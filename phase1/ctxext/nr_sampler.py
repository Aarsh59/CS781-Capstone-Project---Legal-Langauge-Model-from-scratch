"""Deterministic no-repeat window sampler for train_ctxext.py (plumbing; declare it).

Windows
  A token file (uint16, documents separated by </s> exactly as the data build wrote them) of n tokens is cut into
  n_w = (n - 1) // L fixed windows; window w is tokens [w*L, w*L + L + 1): inputs w*L .. w*L+L-1, targets one to the
  right. Input positions of different windows never overlap, so every token position of a file is an input at most once
  and a target at most once. The </s> separators stay where the build put them; a window may span a document boundary
  (packed training, as in Phase 1). The last (n - 1) - n_w*L tokens of a file are never used.

Order (fixed by seed, file lengths, weights and budget; identical on every rank and in every session)
  * budget: n_windows = max_tokens // L (whole windows only; never more than max_tokens input tokens).
  * allocation: windows per file proportional to the weights (largest remainder); a file that has fewer windows than
    its share gives everything it has and the rest is re-allocated to the other files in proportion to their weights.
    If all files together have fewer windows than the budget: error (would need repeats).
  * per file: the first n_alloc entries of a seeded permutation of its window ids (each id at most once).
  * source interleaving: a seeded permutation of the multiset of source labels, so the mixture is uniform over the run.
  The global order is the array pair (src[k], wid[k]), k = 0 .. n_windows-1. Step s consumes positions
  [s*S, min((s+1)*S, n_windows)), S = windows per optimiser step; rank r takes the r-th block of per_rank positions.
"""
import hashlib

import numpy as np


def allocate(avail, weights, n_total):
    """windows per file: proportional to weights, capped by availability, remainder redistributed."""
    avail = np.asarray(avail, np.int64)
    w = np.asarray(weights, float)
    assert (w > 0).all() and len(w) == len(avail)
    if avail.sum() < n_total:
        raise ValueError(f"budget needs {n_total} windows but the files have only {int(avail.sum())}: lower max_tokens "
                         "or add data (the no-repeat sampler never repeats a window)")
    alloc = np.zeros(len(avail), np.int64)
    open_ = np.ones(len(avail), bool)
    left = n_total
    while left > 0:
        ww = np.where(open_, w, 0.0)
        share = ww / ww.sum() * left
        base = np.floor(share).astype(np.int64)
        rem = left - base.sum()
        order = np.argsort(-(share - base), kind="stable")    # largest remainder, ties by file order
        for i in order[:rem]:
            base[i] += 1
        room = avail - alloc
        take = np.minimum(base, room)
        alloc += take
        left -= int(take.sum())
        open_ &= (avail - alloc) > 0
    assert alloc.sum() == n_total and (alloc <= avail).all()
    return alloc


def build_order(file_lengths, weights, max_tokens, L, seed):
    n_avail = np.array([(n - 1) // L for n in file_lengths], np.int64)
    n_total = int(max_tokens // L)
    alloc = allocate(n_avail, weights, n_total)
    wid = np.empty(n_total, np.int64)
    src = np.repeat(np.arange(len(alloc), dtype=np.int8), alloc)
    src = src[np.random.default_rng([seed, 1_000_003]).permutation(n_total)]
    for f, a in enumerate(alloc):
        perm = np.random.default_rng([seed, 7_919, f]).permutation(n_avail[f])[:a]
        wid[src == f] = perm
    order = {"src": src, "wid": wid, "n_avail": n_avail, "alloc": alloc, "L": int(L), "seed": int(seed),
             "max_tokens": int(max_tokens), "file_lengths": np.asarray(file_lengths, np.int64)}
    check_order(order)
    return order


def check_order(order):
    """assertions: every window id valid and used at most once per file; counts match the allocation."""
    src, wid = order["src"], order["wid"]
    assert len(src) == len(wid) == int(order["alloc"].sum())
    for f, a in enumerate(order["alloc"]):
        ids = wid[src == f]
        assert len(ids) == a, (f, len(ids), a)
        assert len(ids) == 0 or (ids.min() >= 0 and ids.max() < order["n_avail"][f]), f"window id out of range in file {f}"
        assert len(np.unique(ids)) == len(ids), f"repeated window id in file {f}"


def order_digest(order):
    h = hashlib.sha256()
    for k in ("src", "wid", "alloc", "file_lengths"):
        h.update(np.ascontiguousarray(order[k]).tobytes())
    h.update(repr((order["L"], order["seed"], order["max_tokens"])).encode())
    return h.hexdigest()


def step_positions(step, S, n_total, world, rank):
    """global order positions this rank trains on at this step (may be empty in a final partial step)."""
    per_rank = S // world
    lo = step * S + rank * per_rank
    return np.arange(max(lo, 0), max(min(lo + per_rank, n_total, (step + 1) * S), lo), dtype=np.int64)


def per_source_counts(order, upto):
    """windows consumed per file among order positions [0, upto)."""
    return np.bincount(order["src"][:upto], minlength=len(order["alloc"])).astype(np.int64)
