# Legal byte-level BPE (your A1 tokenizer, optimised)
# tokenizer.py
#
# Byte-level BPE with an end-of-word marker (A1 design), adapted for the CS781 legal corpus.
#
# Changes vs. the A1 version (original kept in Assignment3/a1_tokenizer.py):
#   * Tokens are integer ids backed by raw bytes, not strings. Previously the byte "<", "/",
#     "w", ">" could merge into a token identical to the "</w>" marker (or to "<pad>"), and
#     decode() would then insert spurious spaces. Ids make that collision impossible.
#   * Legal-text pre-tokenisation inside each whitespace word: letter runs, digit groups of
#     at most 3, and punctuation runs are merged separately ("s.302(1)," -> s . 302 ( 1 ) ,),
#     so "court", "court," and "court." share one vocabulary entry instead of three, and
#     years / section numbers / citations don't flood the vocabulary. Only the last piece of
#     each whitespace word carries </w>, so decode(encode(x)) == " ".join(x.split()).
#   * Incremental BPE training (pair counts + inverted index + lazy max-heap): each merge
#     only touches the words that contain the merged pair, instead of recounting every pair
#     in the corpus for every merge. 32k merges on a legal sample take minutes, not days.
#   * Encoding uses merge ranks (GPT-2 style, same result as applying merges in order) and a
#     bounded per-word cache (the old lru_cache on a method also cached `self` and never freed).
#   * Extra reserved special tokens for later phases (cls/sep/mask, span-corruption sentinels).
#   * encode_many() encodes many documents in parallel worker processes.
#
# Assignment 3 contract is unchanged: train_tokenizer(train_text, vocab_size) returns an object
# with encode(text) -> list[int], decode(ids) -> str and a vocab_size attribute.

import re
import os
import heapq
import string
import json
from collections import Counter, defaultdict


# ============================================================
# SPECIAL TOKENS
# ============================================================

N_SENTINELS = 25

SPECIALS = (
    ["<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>"]
    + [f"<extra_id_{i}>" for i in range(N_SENTINELS)]
)   # 32 specials: ids 0..31

MIN_COUNT = 2

# Pieces inside one whitespace word: letter runs | digit groups (<= 3) | punctuation runs | underscores.
# The alternatives cover every non-space character, so "".join(pieces) == word always holds.
PRETOKENIZE = re.compile(r"[^\W\d_]+|\d{1,3}|[^\w\s]+|_+")

MAX_CACHE_WORDS = 2_000_000


# ============================================================
# TEXT CLEANING (A1 helpers, unchanged; not used by the BPE)
# ============================================================

def check_valid_en(word):
    is_en = False
    is_hi = False
    has_digit = False

    for ch in word:
        code = ord(ch)

        # English alphabet
        if 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A:
            is_en = True

        # Arabic numerals
        elif 0x0030 <= code <= 0x0039:
            has_digit = True

        # Devanagari numerals
        elif 0x0966 <= code <= 0x096F:
            has_digit = True

        # Devanagari
        elif 0x0900 <= code <= 0x097F:
            is_hi = True

    if has_digit:
        return False

    return is_en and not is_hi


def clean_text_and_tokenize_en(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return [word for word in tokens if check_valid_en(word)]


# ============================================================
# BYTE-LEVEL BPE
# ============================================================

def split_word(word):
    """Whitespace word -> list of (piece, is_last_piece)."""
    pieces = PRETOKENIZE.findall(word) or [word]
    last = len(pieces) - 1
    return [(p, i == last) for i, p in enumerate(pieces)]


class ByteBPE:

    def __init__(self, vocab_size=16000):

        self.vocab_size = vocab_size

        # Id layout: specials | 256 bytes | </w> | one id per learned merge
        self.n_special = len(SPECIALS)
        self.byte_offset = self.n_special
        self.eow_id = self.byte_offset + 256

        self.merges = []            # list of (left_id, right_id); merge k creates id eow_id + 1 + k
        self._reset_tables()

    # --------------------------------------------------------
    # VOCAB TABLES
    # --------------------------------------------------------

    def _reset_tables(self):
        self.token_bytes = [b""] * self.n_special + [bytes([b]) for b in range(256)] + [b""]
        self.token_eow = [False] * (self.n_special + 256) + [True]
        for left, right in self.merges:
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])
            self.token_eow.append(self.token_eow[right])
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.merge_to_id = {pair: self.eow_id + 1 + rank for rank, pair in enumerate(self.merges)}
        # decode table: bytes of each token, plus a space if it ends a word; specials decode to nothing
        self._decode_table = [
            b"" if i < self.n_special else tb + (b" " if eow else b"")
            for i, (tb, eow) in enumerate(zip(self.token_bytes, self.token_eow))
        ]
        self.itos = list(SPECIALS) + [self._display(i) for i in range(self.n_special, len(self.token_bytes))]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab = self.itos
        self._cache = {}

    def _display(self, idx):
        """Human-readable token string (for inspection only; never used for encoding)."""
        text = self.token_bytes[idx].decode("utf-8", errors="backslashreplace")
        return text + ("</w>" if self.token_eow[idx] else "")

    @property
    def special_ids(self):
        return {tok: i for i, tok in enumerate(SPECIALS)}

    @property
    def eot_id(self):
        """Document separator used in the pretraining stream."""
        return SPECIALS.index("</s>")

    def get_vocab_size(self):
        return len(self.token_bytes)

    # --------------------------------------------------------
    # PIECE -> INITIAL SYMBOLS
    # --------------------------------------------------------

    def _piece_symbols(self, piece, is_last):
        symbols = [self.byte_offset + b for b in piece.encode("utf-8")]
        if is_last:
            symbols.append(self.eow_id)
        return symbols

    # --------------------------------------------------------
    # TRAIN BPE
    # --------------------------------------------------------

    @staticmethod
    def count_words(texts):
        """Whitespace-word frequencies of an iterable of texts."""
        counts = Counter()
        for text in texts:
            counts.update(str(text).split())
        return counts

    def fit(self, tokenized_corpus=None, target_vocab_size=None, min_count=MIN_COUNT,
            word_counts=None, min_word_freq=1, verbose=True):
        """Learn merges.

        tokenized_corpus: iterable of lists of whitespace words (A1/A3 interface), or
        word_counts:      a precomputed {whitespace word: frequency} mapping (faster for big corpora).
        min_word_freq:    ignore whitespace words rarer than this while learning merges (speed knob).
        """
        target = self.vocab_size if target_vocab_size is None else target_vocab_size
        self.vocab_size = target

        if word_counts is None:
            word_counts = Counter()
            for sentence_tokens in tokenized_corpus:
                word_counts.update(sentence_tokens)

        # whitespace words -> (piece, is_last) frequencies
        piece_freq = Counter()
        for word, count in word_counts.items():
            if count < min_word_freq:
                continue
            for piece in split_word(word):
                piece_freq[piece] += count

        words = [self._piece_symbols(p, last) for p, last in piece_freq]
        freqs = list(piece_freq.values())
        del piece_freq

        if verbose:
            print("=" * 60)
            print("BUILDING BYTE-LEVEL BPE")
            print("=" * 60)
            print(f"Unique pre-token pieces: {len(words):,}")
            print(f"Initial vocabulary size: {self.eow_id + 1:,}")
            print(f"Target vocabulary size: {target:,}")

        # pair -> total count, pair -> set of word indices containing it
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, (symbols, f) in enumerate(zip(words, freqs)):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)

        # max-heap on count with deterministic tie-break on ids; stale entries are skipped lazily
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = []
        next_id = self.eow_id + 1

        while next_id < target and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current != -neg:
                if current > 0:
                    heapq.heappush(heap, (-current, pair))
                continue
            if current < min_count:
                if verbose:
                    print(f"\nStopping: best pair frequency = {current}")
                break

            left, right = pair
            new_id = next_id
            next_id += 1
            self.merges.append(pair)
            touched = Counter()

            for wi in list(where.pop(pair, ())):
                symbols, f = words[wi], freqs[wi]
                # remove this word's old pairs
                for p in zip(symbols, symbols[1:]):
                    pair_counts[p] -= f
                    touched[p] += 0
                    s = where.get(p)
                    if s is not None:
                        s.discard(wi)
                # merge left-to-right
                merged, i, n = [], 0, len(symbols)
                while i < n:
                    if i < n - 1 and symbols[i] == left and symbols[i + 1] == right:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                words[wi] = merged
                # add the new pairs
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    touched[p] += 0
                    where[p].add(wi)

            pair_counts.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c <= 0:
                    pair_counts.pop(p, None)
                    where.pop(p, None)
                elif new_id in p:
                    heapq.heappush(heap, (-c, p))   # only pairs with the new symbol can grow

            if verbose and len(self.merges) % 1000 == 0:
                print(f"Merges: {len(self.merges):,} | Vocab: {next_id:,} | Latest pair count: {current:,}")

        self._reset_tables()

        if verbose:
            print("\n" + "=" * 60)
            print("BPE TRAINING COMPLETE")
            print("=" * 60)
            print(f"Final vocabulary size: {self.get_vocab_size():,}")
            print(f"Total merges learned: {len(self.merges):,}")

        return self

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    def _bpe(self, symbols):
        """Repeatedly merge the lowest-rank adjacent pair (== applying merges in learned order)."""
        ranks = self.ranks
        while len(symbols) > 1:
            best, best_rank = None, None
            for pair in zip(symbols, symbols[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id = self.merge_to_id[best]
            left, right = best
            merged, i, n = [], 0, len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == left and symbols[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def encode_word(self, word):
        ids = self._cache.get(word)
        if ids is None:
            ids = []
            for piece, is_last in split_word(word):
                ids.extend(self._bpe(self._piece_symbols(piece, is_last)))
            ids = tuple(ids)
            if len(self._cache) >= MAX_CACHE_WORDS:
                self._cache.clear()
            self._cache[word] = ids
        return ids

    def encode_tokens(self, sentence_tokens):
        ids = []
        for word in sentence_tokens:
            ids.extend(self.encode_word(word))
        return ids

    def encode(self, text):
        """text -> list of integer ids (whitespace is normalised to single spaces)."""
        cache = self._cache
        ids = []
        for word in str(text).split():
            w = cache.get(word)
            if w is None:
                w = self.encode_word(word)
            ids.extend(w)
        return ids

    def pool(self, workers=None):
        """Process pool whose workers each hold a copy of this tokenizer (and keep their word cache).
        Use with the module-level worker functions, e.g. pool.imap(encode_array_worker, texts)."""
        import multiprocessing as mp
        workers = workers or os.cpu_count() or 1
        return mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(self,))

    def encode_many(self, texts, workers=None, chunksize=64):
        """Encode many documents in parallel processes; yields lists of ids in input order."""
        with self.pool(workers) as pool:
            yield from pool.imap(_worker_encode, texts, chunksize=chunksize)

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    def decode(self, ids):
        table = self._decode_table
        n = len(table)
        out = b"".join(table[int(i)] if 0 <= int(i) < n else b"" for i in ids)
        return out.decode("utf-8", errors="replace").strip()

    def ends_word(self, idx):
        """True if token idx closes a whitespace word (used to count generated words)."""
        return self.token_eow[idx]

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "legal-bytebpe-v2",
                    "specials": SPECIALS,
                    "merges": self.merges,
                    "vocab_size": self.get_vocab_size(),
                    "itos": self.itos,          # readable, informational only
                },
                f,
                ensure_ascii=False,
            )
        print(f"Saved tokenizer to: {path}")

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != "legal-bytebpe-v2":
            raise ValueError(f"{path} was not written by this tokenizer version")
        if data["specials"] != SPECIALS:
            raise ValueError("special tokens differ from this tokenizer version")
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._reset_tables()
        return tokenizer


# worker-process helpers for encode_many (module level so they pickle)
_WORKER_TOKENIZER = None


def _worker_init(tokenizer):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _worker_encode(text):
    return _WORKER_TOKENIZER.encode(text)


def encode_array_worker(text):
    """(uint16 id array, whitespace word count) for one document; for tokenizer.pool().imap."""
    import numpy as np
    return np.asarray(_WORKER_TOKENIZER.encode(text), dtype=np.uint16), text.count(" ") + 1


# ============================================================
# ASSIGNMENT 3 TRAINING FUNCTION
# ============================================================

def train_tokenizer(train_text: str, vocab_size: int):
    text = str(train_text)
    print(f"Training tokenizer on {len(text):,} characters")
    tokenizer = ByteBPE(vocab_size=vocab_size)
    tokenizer.fit(
        word_counts=ByteBPE.count_words(text.splitlines()),
        target_vocab_size=vocab_size,
        min_count=MIN_COUNT,
    )
    return tokenizer


# ============================================================
# SIMPLE SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TOKENIZER SMOKE TEST")
    print("=" * 60)

    train_text = """
    the court has jurisdiction over this matter under Section 302(1) of the Code.
    the court shall consider the evidence, vide AIR 1966 SC 1882, dated 18.12.1961.
    the evidence presented before the court </w> <pad> was 12,345 pages.
    """

    tokenizer = train_tokenizer(train_text, vocab_size=400)

    for test_text in [
        "the court has jurisdiction",
        "Section 302(1), AIR 1966 SC 1882 </w><pad> naïve — ₹5,00,000/- (u/s 138)",
    ]:
        ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(ids)
        print("\nInput:  ", test_text)
        print("Ids:    ", ids)
        print("Pieces: ", [tokenizer.itos[i] for i in ids])
        print("Decoded:", decoded)
        assert decoded == " ".join(test_text.split()), "round trip failed"

    print("\nVocabulary size:", tokenizer.get_vocab_size())

    tokenizer.to_json("bpe_smoke_test.json")
    loaded = ByteBPE.from_json("bpe_smoke_test.json")
    assert loaded.encode(test_text) == tokenizer.encode(test_text)
    os.remove("bpe_smoke_test.json")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
