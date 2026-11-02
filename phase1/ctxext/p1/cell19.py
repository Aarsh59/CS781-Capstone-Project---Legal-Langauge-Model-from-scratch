# build train.bin / val.bin / val_prompts.json with the legal BPE (skipped if they exist)
# Same recipe as your earlier pipeline: whitespace-normalised docs; all Supreme Court (repeated up to
# SC_MAX_EPOCHS, hash-split slice held out for validation); Acts up to ACTS_MAX_EPOCHS; the rest filled with
# the most SC-like High Court / Tribunal docs (DSIR-style unigram score). Encoding runs on all CPU cores.
# Change vs. before: HC/Tribunals are scored on a random sample of HT_SCAN_FACTOR x the needed tokens instead
# of the whole 17 GB, which keeps this stage around an hour in pure Python.
SCRATCH = "/tmp/cs781_scratch"


_encode_doc = a1_tokenizer.encode_array_worker   # runs in worker processes: (uint16 ids, word count)


def encode_docs(name, texts, score_profile=None):
    """Encode docs to a scratch .bin; returns offsets, per-doc DSIR scores and stats."""
    bin_path = os.path.join(SCRATCH, f"{name}.bin")
    lengths, scores, stats = [], [], Counter()
    t0 = time.time()
    with open(bin_path, "wb") as out, tokenizer.pool(N_WORKERS) as pool:
        for arr, n_words in pool.imap(_encode_doc, texts, chunksize=8):
            stats["docs_seen"] += 1
            if name != "supreme_court" and len(arr) / n_words > MAX_FERTILITY:
                stats["dropped_fertility"] += 1
                continue
            arr.tofile(out)
            lengths.append(len(arr))
            if score_profile is not None:
                i64 = arr.astype(np.int64)
                scores.append(float(profiles["supreme_court"][i64].mean() - score_profile[i64].mean()))
            if len(lengths) % 50_000 == 0:
                print(f"  [{name}] {len(lengths):,} docs | {sum(lengths) / 1e6:,.0f}M tokens | "
                      f"{sum(lengths) / (time.time() - t0) / 1e6:.2f}M tok/s", flush=True)
    offsets = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))])
    stats.update(docs_kept=len(lengths), unique_tokens=int(offsets[-1]), encode_minutes=round((time.time() - t0) / 60, 2))
    print(f"[{name}] {dict(stats)}", flush=True)
    return {"bin": bin_path, "offsets": offsets, "scores": np.asarray(scores, np.float32), "stats": dict(stats)}


def clean_stream(texts, stats, val_sink=None):
    for raw in texts:
        text = normalize_ws(raw)
        if text.count(" ") + 1 < MIN_WORDS:
            stats["dropped_short"] += 1
            continue
        if val_sink is not None and doc_hash(text) % 100 < SC_VAL_PERCENT:
            val_sink.append(text)
            continue
        stats["chars"] += len(text)
        yield text


if NEED_DATA:
    os.makedirs(SCRATCH, exist_ok=True)
    free_gb = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"Free scratch disk: {free_gb:.1f} GB")
    t_all = time.time()

    # unigram profiles for the DSIR-style score
    def unigram_logprobs(texts):
        counts = np.zeros(V, dtype=np.float64)
        with tokenizer.pool(N_WORKERS) as pool:
            for arr, _ in pool.imap(_encode_doc, (normalize_ws(t) for t in texts), chunksize=8):
                counts += np.bincount(arr, minlength=V)
        return np.log((counts + 1) / (counts.sum() + V)).astype(np.float32)

    profiles = {s: unigram_logprobs(sample_texts_random_seek(p, PROFILE_CHARS, seed=SEED + 2)) for s, p in SOURCES.items()}
    print(f"profiles built ({(time.time() - t_all) / 60:.1f} min)")

    encoded, stream_stats, val_texts = {}, {}, []
    for name in ("supreme_court", "acts"):
        stream_stats[name] = Counter()
        sink = val_texts if name == "supreme_court" else None
        encoded[name] = encode_docs(name, clean_stream(iter_texts(SOURCES[name]), stream_stats[name], sink))
    sc_chars_per_token = stream_stats["supreme_court"]["chars"] / encoded["supreme_court"]["stats"]["unique_tokens"]

    rng = np.random.default_rng(SEED)

    def doc_len(src, i):
        off = encoded[src]["offsets"]
        return int(off[i + 1] - off[i])

    def repeat_docs(src, budget):
        """Whole epochs over all docs, then a random subset for the fractional epoch."""
        n_docs = len(encoded[src]["offsets"]) - 1
        unique = encoded[src]["stats"]["unique_tokens"]
        full = int(budget // unique) if unique else 0
        picks, used = [(src, i) for _ in range(full) for i in range(n_docs)], full * unique
        for i in rng.permutation(n_docs):
            if used >= budget:
                break
            picks.append((src, int(i)))
            used += doc_len(src, int(i))
        return picks, used

    sc_budget = min(SC_MAX_EPOCHS * encoded["supreme_court"]["stats"]["unique_tokens"], SC_MAX_SHARE * TOTAL_TRAIN_TOKENS)
    acts_budget = min(ACTS_MAX_EPOCHS * encoded["acts"]["stats"]["unique_tokens"], ACTS_MAX_SHARE * TOTAL_TRAIN_TOKENS)
    sc_picks, sc_used = repeat_docs("supreme_court", sc_budget)
    acts_picks, acts_used = repeat_docs("acts", acts_budget)
    rest_budget = max(0, TOTAL_TRAIN_TOKENS - sc_used - acts_used)

    # High Courts + Tribunals: random sample, scored, pooled ranking, each doc used at most once
    sizes = {s: os.path.getsize(SOURCES[s]) for s in ("high_courts", "tribunals")}
    scan_chars = HT_SCAN_FACTOR * rest_budget * sc_chars_per_token
    for s in sizes:
        stream_stats[s] = Counter()
        sample = sample_texts_random_seek(SOURCES[s], int(scan_chars * sizes[s] / sum(sizes.values())), seed=SEED + 3)
        encoded[s] = encode_docs(s, clean_stream(sample, stream_stats[s]), score_profile=profiles[s])
        del sample
    pool_rank = sorted(((float(sc), s, i) for s in sizes for i, sc in enumerate(encoded[s]["scores"])), key=lambda x: -x[0])
    ht_picks, ht_used, score_cutoff = [], 0, None
    for sc, s, i in pool_rank:
        if ht_used >= rest_budget:
            break
        ht_picks.append((s, i))
        ht_used += doc_len(s, i)
        score_cutoff = sc

    all_picks = sc_picks + acts_picks + ht_picks
    order = rng.permutation(len(all_picks))

    # train.bin: shuffled docs separated by </s>
    src_maps = {s: np.memmap(encoded[s]["bin"], dtype=np.uint16, mode="r") for s in encoded}
    train_len = sum(doc_len(s, i) + 1 for s, i in all_picks)
    train_mm = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="w+", shape=(train_len,))
    pos = 0
    for j in order:
        s, i = all_picks[j]
        off = encoded[s]["offsets"]
        n = int(off[i + 1] - off[i])
        train_mm[pos:pos + n] = src_maps[s][off[i]:off[i + 1]]
        train_mm[pos + n] = EOT_ID
        pos += n + 1
    train_mm.flush()
    del train_mm
    print(f"train.bin: {train_len:,} tokens")

    # val.bin + test-like val prompts (512-word prompt, 128-word reference) from held-out SC docs
    with tokenizer.pool(N_WORKERS) as pool:
        val_arrs = [a for a, _ in pool.imap(_encode_doc, val_texts, chunksize=8)]
    val_arr = np.concatenate([np.append(a, np.uint16(EOT_ID)) for a in val_arrs]).astype(np.uint16)
    val_arr.tofile(os.path.join(DATA_DIR, "val.bin"))
    val_prompts, prng = {}, np.random.default_rng(SEED + 7)
    for idx in prng.permutation(len(val_texts)):
        words = val_texts[idx].split()
        if len(words) < 640:
            continue
        st = int(prng.integers(0, len(words) - 640 + 1))
        val_prompts[str(len(val_prompts))] = {"text": " ".join(words[st:st + 512]), "reference": " ".join(words[st + 512:st + 640])}
        if len(val_prompts) >= N_VAL_PROMPTS:
            break
    json.dump(val_prompts, open(os.path.join(DATA_DIR, "val_prompts.json"), "w", encoding="utf-8"), ensure_ascii=False)

    used = {"supreme_court": sc_used, "acts": acts_used}
    for s, i in ht_picks:
        used[s] = used.get(s, 0) + doc_len(s, i)
    manifest = {
        "tokenizer": "legal_bpe.json", "vocab_size": V, "eot_id": EOT_ID, "dtype": "uint16",
        "train_tokens": int(train_len), "val_tokens": int(len(val_arr)), "val_prompts": len(val_prompts),
        "test_fertility": fert, "whitespace_normalised": True, "hc_trib_score_cutoff": score_cutoff,
        "sources": {s: {**encoded[s]["stats"], "tokens_used": int(used.get(s, 0)),
                        "share": round(used.get(s, 0) / max(sum(used.values()), 1), 4)} for s in encoded},
        "build_minutes": round((time.time() - t_all) / 60, 1),
    }
    json.dump(manifest, open(os.path.join(DATA_DIR, "manifest.json"), "w"), indent=2)
    del src_maps
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(json.dumps(manifest, indent=2))
else:
    print("Using existing data:", sorted(os.listdir(DATA_DIR)))
    print(open(os.path.join(DATA_DIR, "manifest.json")).read()[:1500])