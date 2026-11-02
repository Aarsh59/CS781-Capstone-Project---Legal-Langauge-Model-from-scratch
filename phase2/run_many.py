#!/usr/bin/env python3
"""Run several train_ner.py configs over the available GPUs IN PARALLEL (plumbing; declare it).

  python run_many.py --gpus 0,1 configs/B_full.json configs/C_full.json ...

One worker per GPU; every worker runs one training process at a time on its own GPU (CUDA_VISIBLE_DEVICES), so with
--gpus 0,1 two fine-tuning runs train simultaneously, one per T4. Scheduling uses a SHARED queue: whenever a GPU becomes
free it takes the first runnable config in list order (no GPU sits idle while work remains). A config with "kd" is only
runnable once its teacher has written summary.json; until then the free GPU takes the next job instead. If the teacher
fails (or is not part of this batch and has no summary.json), the KD run is marked FAILED instead of waiting forever.

Logs: <out_dir>/<name>.stdout (+ <out_dir>/<name>/log.jsonl). A status line every --every seconds, a summary at the end.
Exit code 1 if any run failed.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Scheduler:
    def __init__(self, cfg_paths):
        self.lock = threading.Lock()
        self.cfgs = {}
        for p in cfg_paths:
            c = json.load(open(p))
            self.cfgs[c["name"]] = (p, c)
        self.pending = list(self.cfgs)                        # names, in the order given
        self.status = {n: "queued" for n in self.pending}
        self.started = {}

    def _teacher(self, cfg):
        if not cfg.get("kd"):
            return None, None
        d = cfg["kd"]["teacher_run"].rstrip("/")
        return os.path.basename(d), os.path.join(d, "summary.json")

    def next_job(self):
        """-> (name, path, cfg) to run now | "wait" (only KD runs whose teacher is still training) | None (nothing left)."""
        with self.lock:
            if not self.pending:
                return None
            for name in list(self.pending):
                path, cfg = self.cfgs[name]
                teacher, done = self._teacher(cfg)
                if teacher is not None and not os.path.exists(done):
                    t_status = self.status.get(teacher)
                    if t_status is None or t_status.startswith(("FAILED", "done")):
                        self.pending.remove(name)
                        self.status[name] = f"FAILED (teacher {teacher} has no summary.json)"
                        continue
                    self.status[name] = f"waiting for teacher {teacher}"
                    continue
                self.pending.remove(name)
                return name, path, cfg
            return "wait" if self.pending else None

    def worker(self, gpu):
        while True:
            job = self.next_job()
            if job is None:
                return
            if job == "wait":
                time.sleep(10)
                continue
            name, path, cfg = job
            os.makedirs(cfg["out_dir"], exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2")
            with self.lock:
                self.status[name] = f"running on GPU {gpu}"
                self.started[name] = time.time()
            with open(os.path.join(cfg["out_dir"], f"{name}.stdout"), "w") as log:
                rc = subprocess.call([sys.executable, os.path.join(HERE, "train_ner.py"), path], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
            mins = (time.time() - self.started[name]) / 60
            with self.lock:
                self.status[name] = f"done on GPU {gpu} in {mins:.1f} min" if rc == 0 else \
                    f"FAILED (exit {rc}) on GPU {gpu} after {mins:.1f} min"


def last_log_line(cfg):
    logf = os.path.join(cfg["out_dir"], cfg["name"], "log.jsonl")
    if not os.path.exists(logf):
        return None
    lines = open(logf).read().strip().splitlines()
    return lines[-1][:250] if lines else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--every", type=int, default=300)
    args = ap.parse_args()
    gpus = [g for g in args.gpus.split(",") if g != ""]
    sched = Scheduler(args.configs)
    print(f"{len(sched.cfgs)} runs on GPUs {gpus} ({len(gpus)} in parallel): {list(sched.cfgs)}", flush=True)
    threads = [threading.Thread(target=sched.worker, args=(g,), daemon=True, name=f"gpu{g}") for g in gpus]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=args.every / len(threads))
        with sched.lock:
            status = dict(sched.status)
        print(f"\n{time.strftime('%H:%M:%S')} (+{(time.time() - t0) / 60:.1f} min)", flush=True)
        for name, st in status.items():
            line = last_log_line(sched.cfgs[name][1]) if st.startswith("running") else None
            print(f"  {name:<20} {st}" + (f"\n      {line}" if line else ""), flush=True)

    print("\nsummary")
    failed = []
    for name, (path, cfg) in sched.cfgs.items():
        sp = os.path.join(cfg["out_dir"], name, "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp))
            print(f"  {name:<20} val macro-F1 {s.get('val_macro_f1')}  trainable {s['params']['trainable_pct']}%  "
                  f"train {s['train_seconds'] / 60:.1f} min  peak {s['peak_mem_GB']} GB  ({sched.status[name]})")
        else:
            print(f"  {name:<20} no summary ({sched.status[name]})")
        if not sched.status[name].startswith("done"):
            failed.append(name)
    if failed:
        sys.exit(f"failed runs: {failed} (see <out_dir>/<name>.stdout)")


if __name__ == "__main__":
    main()
