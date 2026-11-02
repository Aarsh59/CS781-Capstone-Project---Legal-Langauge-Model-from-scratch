#!/usr/bin/env python3
"""Steps 1-2 (+ step 4.1-4.4 error counts): reproduce baselines, pairwise diversity, duplicates, anchor error analysis.

Outputs (ens_family/out/): baselines.csv, per_class_baselines.csv, prediction_diversity.csv,
run_inventory.csv, anchored_error_analysis.csv. Family grouping is NOT decided here (see model_family_audit later).
"""
import csv
import itertools
import os

from common import ANCHOR, Fail, flat, load, scores, vote

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
ctx = load()
labels, runs, M = ctx["labels"], ctx["runs"], ctx["M"]
print(f"loaded {len(runs)} runs; dev {len(ctx['gold'])} docs / {sum(ctx['nsent'].values())} sentences; "
      f"test {len(ctx['tnsent'])} docs / {sum(ctx['tnsent'].values())} sentences; label order = label_vocab")

# ---- step 1: reproduce
S = {n: scores(ctx, M[n]["dev"]) for n in runs}
for n in runs:
    sm = M[n]["summary"]
    if (round(S[n]["dev"], 4), round(S[n]["CL"], 4), round(S[n]["IT"], 4)) != \
            (round(sm["dev_macro_f1"], 4), round(sm["dev_CL_macro_f1"], 4), round(sm["dev_IT_macro_f1"], 4)):
        raise Fail(f"{n}: recomputed dev scores differ from summary.json")
rank = sorted(runs, key=lambda n: -S[n]["dev"])                      # members ranked by dev; ties -> higher rank
base = {"single " + ANCHOR: M[ANCHOR]["dev"]}
for k in (5, 7, 27):
    base[f"top{k} hard vote" if k < 27 else "all27 hard vote"] = vote([M[n]["dev"] for n in rank[:k]], [1.0] * k)
expected = {"single " + ANCHOR: 0.5109, "top5 hard vote": 0.5192, "top7 hard vote": 0.5195, "all27 hard vote": 0.4952}
with open(os.path.join(OUT, "baselines.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "members", "dev", "CL", "IT", "macro_P", "macro_R", "reproduces_reported"])
    for name, P in base.items():
        s = scores(ctx, P)
        k = 1 if name.startswith("single") else int(name.split()[0].replace("top", "").replace("all", ""))
        ok = round(s["dev"], 4) == expected[name]
        if not ok:
            raise Fail(f"{name}: {s['dev']:.4f} does not reproduce {expected[name]}")
        w.writerow([name, " ".join(rank[:k]) if k > 1 else ANCHOR, f"{s['dev']:.4f}", f"{s['CL']:.4f}", f"{s['IT']:.4f}",
                    f"{s['P']:.4f}", f"{s['R']:.4f}", ok])
        print(f"  reproduced {name:34s} dev {s['dev']:.4f}  CL {s['CL']:.4f}  IT {s['IT']:.4f}")
with open(os.path.join(OUT, "per_class_baselines.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method"] + labels)
    for name, P in base.items():
        s = scores(ctx, P); w.writerow([name] + [s["per"].get(l, {}).get("f1", "") for l in labels])

# ---- inventory
with open(os.path.join(OUT, "run_inventory.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "backbone", "technique", "kd", "head", "best_epoch", "dev", "CL", "IT", "dev_rank"])
    for n in runs:
        r = M[n]; w.writerow([n, r["backbone"], r["technique"], r["kd"], r["head"], r["summary"]["best_epoch"],
                              f"{S[n]['dev']:.4f}", f"{S[n]['CL']:.4f}", f"{S[n]['IT']:.4f}", rank.index(n) + 1])

# ---- step 2: pairwise diversity + duplicates
gold_flat = flat(ctx, ctx["gold"])
F = {n: flat(ctx, M[n]["dev"]) for n in runs}
T = {n: flat(ctx, M[n]["test"], "test") for n in runs}
A = F[ANCHOR]
rows, dups = [], []
for a, b in itertools.combinations(runs, 2):
    dag = sum(x == y for x, y in zip(F[a], F[b])) / len(A)
    tag = sum(x == y for x, y in zip(T[a], T[b])) / len(T[a])
    row = {"run_a": a, "run_b": b, "dev_agreement": round(dag, 4), "test_agreement": round(tag, 4),
           "dev_disagreements": sum(x != y for x, y in zip(F[a], F[b])), "test_disagreements": sum(x != y for x, y in zip(T[a], T[b])),
           "near_duplicate(>=0.99 dev and test)": dag >= 0.99 and tag >= 0.99}
    for l in labels:                                   # per-class disagreement: among sentences either run labels l
        idx = [i for i in range(len(A)) if F[a][i] == l or F[b][i] == l]
        row[f"disagree_{l}"] = round(sum(F[a][i] != F[b][i] for i in idx) / len(idx), 3) if idx else ""
    if ANCHOR in (a, b):                               # does the other run fix or break anchor predictions?
        o = b if a == ANCHOR else a
        row["vs_anchor_fixes"] = sum(A[i] != gold_flat[i] and F[o][i] == gold_flat[i] for i in range(len(A)))
        row["vs_anchor_breaks"] = sum(A[i] == gold_flat[i] and F[o][i] != gold_flat[i] for i in range(len(A)))
    rows.append(row)
    if row["near_duplicate(>=0.99 dev and test)"]:
        dups.append((a, b, dag, row["test_disagreements"]))
keys = list(rows[0]) + ["vs_anchor_fixes", "vs_anchor_breaks"]
keys = list(dict.fromkeys(keys))
with open(os.path.join(OUT, "prediction_diversity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"\npairwise rows: {len(rows)}; near-duplicate pairs (>=0.99 agreement on dev AND test):")
for a, b, dag, td in dups:
    print(f"  {a} ~ {b}: dev agreement {dag:.4f}, test sentences differing {td}")

# ---- step 4.1-4.4: anchor error analysis (all / CL / IT)
dom = [ctx["domain"][k] for k in ctx["gold"] for _ in ctx["gold"][k]]
err = [i for i in range(len(A)) if A[i] != gold_flat[i]]
print(f"\nanchor {ANCHOR}: {len(err)} wrong of {len(A)} dev sentences "
      f"(CL {sum(dom[i] == 'CL' for i in err)} / IT {sum(dom[i] == 'IT' for i in err)})")
arows = []
for o in runs:
    if o == ANCHOR:
        continue
    r = {"run": o, "dev": round(S[o]["dev"], 4)}
    for D in ("all", "CL", "IT"):
        idx = [i for i in range(len(A)) if D == "all" or dom[i] == D]
        fixes = sum(A[i] != gold_flat[i] and F[o][i] == gold_flat[i] for i in idx)
        breaks = sum(A[i] == gold_flat[i] and F[o][i] != gold_flat[i] for i in idx)
        r[f"{D}_fixes"], r[f"{D}_breaks"], r[f"{D}_net"] = fixes, breaks, fixes - breaks
        r[f"{D}_fix_rate_of_anchor_errors"] = round(fixes / max(1, sum(A[i] != gold_flat[i] for i in idx)), 3)
    arows.append(r)
arows.sort(key=lambda r: -r["all_fixes"])
with open(os.path.join(OUT, "anchored_error_analysis.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)
print("top 10 runs by anchor errors corrected (fixes / breaks; CL and IT):")
for r in arows[:10]:
    print(f"  {r['run']:28s} dev {r['dev']:.4f} | all {r['all_fixes']:4d}/{r['all_breaks']:4d} | "
          f"CL {r['CL_fixes']:4d}/{r['CL_breaks']:4d} | IT {r['IT_fixes']:4d}/{r['IT_breaks']:4d}")
