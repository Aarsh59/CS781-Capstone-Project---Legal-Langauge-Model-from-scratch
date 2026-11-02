"""Character <-> token alignment for the legal ByteBPE, span <-> tag conversion, windows (plumbing; declare it).

tokenize_with_offsets() reproduces ByteBPE.encode() exactly (same whitespace split, same legal pre-tokenisation,
same merges) while recording, for every token, the character span [start, end) it covers in the ORIGINAL text.
No text normalisation is applied, so offsets refer to the raw document. A token that is only the end-of-word
marker </w> (no bytes) is given the span of the last character of its word.

Tag schemes: "BIO" (B-X, I-X) and "BIOES" (adds E-X for the last token, S-X for single-token entities).
"""
import bisect
import re

import numpy as np

from a1_tokenizer import PRETOKENIZE

WORD = re.compile(r"\S+")          # str.split() semantics: maximal runs of non-whitespace


# ---------------------------------------------------------------------------
# tokenisation with character offsets
# ---------------------------------------------------------------------------

PUNCT_RUN = re.compile(r"[^\w\s]{2,}")


def _word_pieces(word, split_punct=False):
    """(piece, start_in_word, is_last) exactly as a1_tokenizer.split_word, with positions.
    split_punct=True additionally splits every punctuation run ("]." -> "]", ".") into single characters, so an
    entity boundary inside a run (e.g. "...SC 1728]." with the entity ending at "]") falls on a token boundary.
    Every single character is in the vocabulary, but the ids then differ from ByteBPE.encode() at such runs."""
    ms = [(m.group(), m.start()) for m in PRETOKENIZE.finditer(word)]
    if not ms:
        return [(word, 0, True)]
    if split_punct:
        ms = [(ch, p0 + j) for piece, p0 in ms
              for j, ch in (enumerate(piece) if PUNCT_RUN.fullmatch(piece) else [(0, piece)])]
    last = len(ms) - 1
    return [(piece, p0, i == last) for i, (piece, p0) in enumerate(ms)]


class OffsetTokenizer:
    def __init__(self, tok, split_punct=False):
        self.tok = tok
        self.split_punct = split_punct
        self._cache = {}

    def _encode_word(self, word):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        tok = self.tok
        ids, st, en = [], [], []
        for piece, p0, is_last in _word_pieces(word, self.split_punct):
            sym = tok._bpe(tok._piece_symbols(piece, is_last))
            cum = [0]
            for ch in piece:
                cum.append(cum[-1] + len(ch.encode("utf-8")))
            b = 0
            for t in sym:
                nb = len(tok.token_bytes[t])
                if nb == 0:                                  # bare </w>: last char of the piece
                    s_c, e_c = len(piece) - 1, len(piece)
                else:
                    s_c = bisect.bisect_right(cum, b) - 1    # char containing the first byte
                    e_c = bisect.bisect_left(cum, b + nb)    # first char boundary at/after the last byte
                    b += nb
                ids.append(t)
                st.append(p0 + s_c)
                en.append(p0 + e_c)
        out = (ids, st, en)
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[word] = out
        return out

    def __call__(self, text):
        """-> ids, starts, ends as int32 numpy arrays (char offsets into `text`, end exclusive)."""
        ids, st, en = [], [], []
        for m in WORD.finditer(text):
            w_ids, w_st, w_en = self._encode_word(m.group())
            off = m.start()
            ids.extend(w_ids)
            st.extend(off + s for s in w_st)
            en.extend(off + e for e in w_en)
        return (np.asarray(ids, dtype=np.int32), np.asarray(st, dtype=np.int32), np.asarray(en, dtype=np.int32))


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------

def tag_list(labels, scheme):
    prefixes = {"BIO": ("B", "I"), "BIOES": ("B", "I", "E", "S")}[scheme]
    return ["O"] + [f"{p}-{lab}" for lab in labels for p in prefixes]


def spans_to_tags(spans, starts, ends, tag2id, scheme):
    """Gold char spans -> one tag id per token. A token belongs to an entity if it overlaps it.
    Returns tags (int64 array) and the number of entities skipped because a token was already taken
    (overlapping/nested gold entities cannot be expressed with one tag per token)."""
    n = len(starts)
    tags = np.zeros(n, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    skipped = 0
    for s, e, lab in spans:
        lo = bisect.bisect_right(ends, s)                    # first token with end > s
        hi = bisect.bisect_left(starts, e)                   # first token with start >= e
        if lo >= hi or taken[lo:hi].any():
            skipped += 1
            continue
        taken[lo:hi] = True
        if scheme == "BIOES" and hi - lo == 1:
            tags[lo] = tag2id[f"S-{lab}"]
            continue
        tags[lo] = tag2id[f"B-{lab}"]
        tags[lo + 1:hi] = tag2id[f"I-{lab}"]
        if scheme == "BIOES":
            tags[hi - 1] = tag2id[f"E-{lab}"]
    return tags, skipped


def tags_to_spans(tag_ids, starts, ends, id2tag, repair="lenient"):
    """Token tags -> char spans [(start, end_exclusive, label)].
    repair="lenient": an I-/E- tag that does not continue an open entity of the same label starts a new one.
    repair="strict" : such tags are treated as O."""
    spans = []
    cur_lab, cur_s, cur_e = None, None, None

    def close():
        nonlocal cur_lab
        if cur_lab is not None:
            spans.append((int(cur_s), int(cur_e), cur_lab))
        cur_lab = None

    for i, t in enumerate(tag_ids):
        if i > 0 and starts[i] < ends[i - 1]:
            continue        # token adds no new characters (bare </w>, second half of a multi-byte char): keep state
        tag = id2tag[int(t)]
        if tag == "O":
            close()
            continue
        p, lab = tag.split("-", 1)
        if p == "S":
            close()
            spans.append((int(starts[i]), int(ends[i]), lab))
        elif p == "B":
            close()
            cur_lab, cur_s, cur_e = lab, starts[i], ends[i]
        else:                                                 # I or E
            if cur_lab == lab:
                cur_e = ends[i]
            elif repair == "lenient":
                close()
                cur_lab, cur_s, cur_e = lab, starts[i], ends[i]
            else:
                close()
                continue
            if p == "E":
                close()
    close()
    return spans


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def window_starts(n, window, stride):
    """Start offsets of windows of `window` tokens every `stride` tokens; the last window is shifted left so it
    ends exactly at n (no padding needed). A document shorter than `window` is one window of length n."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


def merge_weights(length, mode):
    """Per-position preference inside one window for overlap merging.
    "center": distance to the nearer window edge (tokens see context on both sides)
    "left_context": position in the window (most left context: natural for causal models)
    "mean": uniform (logits of all covering windows are averaged)"""
    pos = np.arange(length)
    if mode == "center":
        return np.minimum(pos, length - 1 - pos)
    if mode == "left_context":
        return pos
    if mode == "mean":
        return np.zeros(length)
    raise ValueError(f"unknown merge mode {mode}")


# ---------------------------------------------------------------------------
# optional span decoding: snap to word boundaries
# ---------------------------------------------------------------------------

def snap_to_words(text, spans):
    """Optional decoding step (off unless enabled): a span that starts or ends inside a word (alphanumeric on both
    sides of the boundary) is extended to the full word; labels never change. If two snapped spans then overlap,
    same-label spans are merged into one; different-label spans keep the longer one (ties: the earlier one).
    spans: [(start, end_exclusive, label)] -> sorted list of non-overlapping spans."""
    out = []
    for s, e, lab in spans:
        while s > 0 and s < len(text) and text[s - 1].isalnum() and text[s].isalnum():
            s -= 1
        while 0 < e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
            e += 1
        out.append((s, e, lab))
    out = sorted(set(out))
    kept = []
    for sp in out:
        if kept and sp[0] < kept[-1][1]:                       # overlaps the previous kept span
            prev = kept[-1]
            if sp[2] == prev[2]:
                kept[-1] = (prev[0], max(prev[1], sp[1]), prev[2])
            elif sp[1] - sp[0] > prev[1] - prev[0]:
                kept[-1] = sp
            continue
        kept.append(sp)
    return kept
