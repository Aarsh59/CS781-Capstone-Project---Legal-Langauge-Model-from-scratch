#!/usr/bin/env python3
"""Threshold selection and probability-averaging ensembles for CJPE runs (plumbing; declare it).

  python cjpe_select.py --name FINAL --out ensembles --members runs/E1_B2_full [runs/E6_B2_lorma ...] \
      [--weights 1 1] [--select-on dev_all] [--tmin 0.05 --tmax 0.95 --tstep 0.01]

P(label 1) per document = weighted mean of the members' dev_probs.json / test_probs.json (one member = threshold
tuning only). The threshold is chosen on DEV only (highest macro-F1 on --select-on; ties -> the threshold closest to
0.5), then applied unchanged to test. Reports dev macro-F1 at 0.5 and at the chosen threshold on dev_all and
dev_excl_test, and the agreement between members on dev. Writes <out>/<name>/: summary.json, dev_probs.json,
test_probs.json, prediction.json ([{"id", "pred"}]) and submission_<name>.zip. Test labels are never read.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                   # noqa: E402
from cjpe_data import binary_scores               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--select-on", default="dev_all", choices=("dev_all", "dev_excl_test"))
    ap.add_argument("--data-dir", help="Phase_4_CJPE folder (default: from the first member's config)")
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.01)
    a = ap.parse_args()
    w = a.weights or [1.0] * len(a.members)
    assert len(w) == len(a.members), "one weight per member"
    cfg0 = json.load(open(os.path.join(a.members[0], "config.json")))
    D = a.data_dir or cfg0["data_dir"]
    dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test")
    cjpe_data.mark_dev_excl_test(dev, test)
    dp = [json.load(open(os.path.join(m, "dev_probs.json"))) for m in a.members]
    tp = [json.load(open(os.path.join(m, "test_probs.json"))) for m in a.members]
    for m, d, t in zip(a.members, dp, tp):
        assert set(d) == {x["id"] for x in dev}, f"{m}: dev ids differ from dev.json"
        assert set(t) == {x["id"] for x in test}, f"{m}: test ids differ from test.json"
    W = np.array(w) / sum(w)
    dev_p = {x["id"]: float(sum(wi * d[x["id"]] for wi, d in zip(W, dp))) for x in dev}
    test_p = {x["id"]: float(sum(wi * t[x["id"]] for wi, t in zip(W, tp))) for x in test}

    def scores(th, key):
        its = [x for x in dev if key == "dev_all" or not x["in_test"]]
        return binary_scores([x["label"] for x in its], [int(dev_p[x["id"]] >= th) for x in its])

    grid = np.round(np.arange(a.tmin, a.tmax + 1e-9, a.tstep), 4)
    curve = [(float(t), scores(t, a.select_on)["macro_f1"]) for t in grid]
    best_f1 = max(f for _, f in curve)
    th = min((t for t, f in curve if f == best_f1), key=lambda t: abs(t - 0.5))
    agree = None
    if len(a.members) > 1:
        preds = np.array([[int(d[x["id"]] >= 0.5) for x in dev] for d in dp])
        agree = {f"{os.path.basename(a.members[i].rstrip('/'))}~{os.path.basename(a.members[j].rstrip('/'))}":
                 round(float((preds[i] == preds[j]).mean()), 4)
                 for i in range(len(preds)) for j in range(i + 1, len(preds))}
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    pred = [{"id": x["id"], "pred": int(test_p[x["id"]] >= th)} for x in test]
    assert len(pred) == len(test) and len({p["id"] for p in pred}) == len(pred)
    assert all(isinstance(p["pred"], int) and p["pred"] in (0, 1) for p in pred)
    json.dump(pred, open(os.path.join(out, "prediction.json"), "w"))
    json.dump(dev_p, open(os.path.join(out, "dev_probs.json"), "w"))
    json.dump(test_p, open(os.path.join(out, "test_probs.json"), "w"))
    with zipfile.ZipFile(os.path.join(out, f"submission_{a.name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out, "prediction.json"), arcname="prediction.json")
    summ = {"name": a.name, "members": a.members, "weights": list(map(float, w)), "select_on": a.select_on,
            "threshold": th, "threshold_grid": [a.tmin, a.tmax, a.tstep],
            "dev_at_0.5": {k: scores(0.5, k) for k in ("dev_all", "dev_excl_test")},
            "dev_at_threshold": {k: scores(th, k) for k in ("dev_all", "dev_excl_test")},
            "dev_macro_f1": scores(th, a.select_on)["macro_f1"], "member_agreement_dev": agree,
            "test_pred_share_1": round(float(np.mean([p["pred"] for p in pred])), 4), "curve": curve}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
