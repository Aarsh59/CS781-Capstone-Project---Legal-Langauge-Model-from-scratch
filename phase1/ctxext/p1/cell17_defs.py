def normalize_ws(text):
    return " ".join(text.split())


def doc_hash(text):
    return int(hashlib.md5(text[:2000].encode("utf-8", errors="ignore")).hexdigest(), 16)


def _parse_line(line):
    line = line.strip().rstrip(",")
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def iter_texts(path):
    """Stream the "text" field of a one-record-per-line JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = _parse_line(line)
            if rec and rec.get("text"):
                yield rec["text"]


def sample_texts_random_seek(path, n_chars, seed):
    """Random byte seeks -> resync to the next line -> one record. Spreads the sample over the whole file."""
    rng = random.Random(seed)
    size = os.path.getsize(path)
    texts, seen, total, misses = [], set(), 0, 0
    with open(path, "rb") as f:
        while total < n_chars and misses < 5000:
            f.seek(rng.randrange(0, size))
            f.readline()
            raw = f.readline()
            rec = _parse_line(raw.decode("utf-8", errors="ignore")) if raw else None
            text = rec.get("text") if rec else None
            if not text:
                misses += 1
                continue
            h = doc_hash(text)
            if h in seen:
                misses += 1
                continue
            seen.add(h)
            texts.append(text)
            total += len(text)
            misses = 0
    if total < n_chars:
        print(f"  WARNING: {os.path.basename(path)} gave only {total / 1e6:.1f}M of {n_chars / 1e6:.1f}M chars")
    return texts


def fertility(tok, texts):
    n_tok = sum(len(tok.encode(t)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(normalize_ws(t)) for t in texts)
    return n_tok / n_words, n_chars / n_tok