"""Phase 2 (L-NER) data loading: documents, gold spans, labels, held-out split (plumbing; declare it).

Internally every span is (start, end, label) with an EXCLUSIVE end, i.e. text[start:end] is the entity.
The spec's own example ([18, 40, "COURT"] on "REPORTABLE IN THE SUPREME COURT OF INDIA") is exclusive;
load_gold() also checks the convention against the entity texts in the gold files and stores it in
END_CONVENTION so predictions are written back in the same convention.
"""
import json
import os
import random
from collections import Counter

END_CONVENTION = {"inclusive": False}      # set by load_gold(); used by to_submission()


def data_root(root):
    """Accepts .../Phase_2_LNER or any parent that contains it."""
    for cand in (root, os.path.join(root, "Phase_2_LNER")):
        if os.path.isdir(os.path.join(cand, "Train")):
            return cand
    raise FileNotFoundError(f"no Phase_2_LNER/Train under {root} (run fetch_data.py first)")


def load_labels(root):
    with open(os.path.join(data_root(root), "Train", "labels.txt"), encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels


def load_texts(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):                    # tolerate [{"id":..., "text":...}, ...]
        data = {str(d.get("id", d.get("doc_id"))): d["text"] for d in data}
    return {str(k): (v["text"] if isinstance(v, dict) else v) for k, v in data.items()}


def _parse_entity(e):
    """-> (start, end, label, text_or_None) from a list/tuple or a dict entry."""
    if isinstance(e, dict):
        s = e.get("start", e.get("start_char", e.get("begin")))
        t = e.get("end", e.get("end_char"))
        lab = e.get("label", e.get("type", e.get("tag")))
        txt = e.get("text", e.get("entity_text", e.get("entity")))
        return int(s), int(t), str(lab), txt
    s, t, lab = e[0], e[1], e[2]
    return int(s), int(t), str(lab), (e[3] if len(e) > 3 else None)


def load_gold(path, texts, labels=None, verbose=True):
    """-> {doc_id: [(start, end_exclusive, label), ...]} sorted by start.
    Detects whether the file's end offsets are exclusive or inclusive from the entity texts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {str(d.get("id", d.get("doc_id"))): d.get("entities", d.get("spans", d.get("labels")))
                for d in data}
    raw = {str(k): [_parse_entity(e) for e in v] for k, v in data.items()}

    excl = incl = with_text = 0
    for doc_id, ents in raw.items():
        text = texts.get(doc_id, "")
        for s, t, _, txt in ents:
            if txt is None:
                continue
            with_text += 1
            excl += text[s:t] == txt
            incl += text[s:t + 1] == txt
    inclusive = with_text > 0 and incl > excl
    END_CONVENTION["inclusive"] = inclusive

    gold, unknown = {}, Counter()
    for doc_id, ents in raw.items():
        spans = []
        for s, t, lab, _ in ents:
            if labels is not None and lab not in labels:
                unknown[lab] += 1
                continue
            spans.append((s, t + 1 if inclusive else t, lab))
        gold[doc_id] = sorted(set(spans))
    if verbose:
        n = sum(len(v) for v in gold.values())
        print(f"{os.path.basename(path)}: {len(gold)} docs, {n} spans | end offsets "
              f"{'INCLUSIVE' if inclusive else 'exclusive'} (text match: exclusive {excl}/{with_text}, "
              f"inclusive {incl}/{with_text})" + (f" | unknown labels dropped: {dict(unknown)}" if unknown else ""))
    missing = [d for d in gold if d not in texts]
    assert not missing, f"{len(missing)} gold docs have no text, e.g. {missing[:3]}"
    return gold


def load_train(root, verbose=True):
    """-> texts {id: str}, gold {id: spans}, fold_of {id: 'fold1'|'fold2'}, labels [..]"""
    root = data_root(root)
    labels = load_labels(root)
    texts, gold, fold_of = {}, {}, {}
    for fold in ("fold1", "fold2"):
        t = load_texts(os.path.join(root, "Train", f"{fold}-text.json"))
        g = load_gold(os.path.join(root, "Train", f"{fold}-gold.json"), t, labels, verbose)
        dup = set(t) & set(texts)
        assert not dup, f"doc ids appear in both folds: {list(dup)[:3]}"
        texts.update(t)
        for d in t:
            gold[d] = g.get(d, [])
            fold_of[d] = fold
    return texts, gold, fold_of, labels


def load_test(root):
    return load_texts(os.path.join(data_root(root), "Test", "fold3-text.json"))


def split_docs(doc_ids, fold_of, split):
    """split = {"mode": "random", "val_frac": float, "seed": int} | {"mode": "fold", "val_fold": "fold2"}
    | {"mode": "none"} (train on everything, no validation).  -> (train_ids, val_ids)"""
    ids = sorted(doc_ids)
    mode = split["mode"]
    if mode == "none":
        return ids, []
    if mode == "fold":
        val = [d for d in ids if fold_of[d] == split["val_fold"]]
        return [d for d in ids if d not in set(val)], val
    if mode == "random":
        rng = random.Random(split["seed"])
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(ids) * split["val_frac"]))
        val = sorted(shuffled[:n_val])
        return [d for d in ids if d not in set(val)], val
    raise ValueError(f"unknown split mode {mode}")


def to_submission(pred):
    """{id: [(s, e_exclusive, label)]} -> {id: [[s, e, label]]} in the gold files' end convention."""
    shift = 1 if END_CONVENTION["inclusive"] else 0
    return {d: [[int(s), int(e) - shift, lab] for s, e, lab in spans] for d, spans in pred.items()}


def label_stats(gold, labels):
    c = Counter(lab for spans in gold.values() for _, _, lab in spans)
    return {lab: c.get(lab, 0) for lab in labels}
