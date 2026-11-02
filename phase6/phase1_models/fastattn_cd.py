# Plumbing: fused SDPA attention + KV caches + generation around your #3 / #4 modules
"""Efficiency plumbing around the hand-written encoder-decoder (#3) and prefix-LM (#4) (written with Claude; declare it).

Nothing here changes an architecture: parameters, shapes and the maths of every model stay exactly as defined in
your files. `patch_model` swaps each attention module's forward for one that computes the *same* function with
F.scaled_dot_product_attention (memory-efficient fused kernel on T4), and adds optional caches for generation:

  encoder SelfAttention (non_causal_self_attention.py)  -> bidirectional, no cache (the encoder runs once per prompt)
  decoder CausalSelfAttention (attention3.py)           -> causal, self-attention KV cache
  CrossAttention (cross_attention.py)                   -> K/V of the encoder memory computed once per prompt, cached
  prefix CausalSelfAttention (prefix_attention.py)      -> mask = causal OR [:p, :p] block, KV cache for decoding

`check_equivalence_*` (run in the notebook before training) verify the patched model against your original forward,
and cached step-by-step decoding against a full forward pass.
"""
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# attention forwards
# ---------------------------------------------------------------------------

def _dropout_p(self):
    return self.attn_dropout.p if self.training else 0.0


def _project_self(self, x, start):
    """q, k, v of a self-attention module, RoPE at absolute positions start..start+T (as in your code)."""
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
    if getattr(self, "use_rope", False):
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]
        q = self._apply_rope(q.float(), cos, sin).to(v.dtype)
        k = self._apply_rope(k.float(), cos, sin).to(v.dtype)
    return q, k, v


def _use_cache(self, k, v, start):
    cache = getattr(self, "_kv_cache", None)
    if cache is None:
        return k, v
    T = k.shape[2]
    cache["k"][:, :, start:start + T] = k
    cache["v"][:, :, start:start + T] = v
    return cache["k"][:, :, :start + T], cache["v"][:, :, :start + T]


def _expand(self, k, v):
    if self.n_rep > 1:
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
    return k, v


def _finish(self, y):
    B, H, T, D = y.shape
    return self.resid_dropout(self.out_proj(y.transpose(1, 2).reshape(B, T, H * D)))


def _sdpa(self, q, k, v, attn_mask=None, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                          dropout_p=_dropout_p(self), scale=self.attention_scale)


def _start(self):
    cache = getattr(self, "_kv_cache", None)
    return cache["pos"] if cache is not None else 0


def _prefix_hook(self, k, v, B):
    """prefix-tuning hook (Phase 3 plumbing): optional self.prefix_kv(B) -> (pk, pv), each (B, n_kv_heads, P, d_head),
    prepended to keys/values (no RoPE on them). Returns (k, v, P)."""
    fn = getattr(self, "prefix_kv", None)
    if fn is None:
        return k, v, 0
    # Phase 6: also used with the KV cache (prefix vectors are re-attached each step; they are not cached)
    pk, pv = fn(B)
    assert pk.shape[0] == B and pk.shape[1] == k.shape[1] and pk.shape[3] == k.shape[3], \
        f"prefix_kv must return (B, n_kv_heads, P, d_head) = ({B}, {k.shape[1]}, P, {k.shape[3]}), got {tuple(pk.shape)}"
    return torch.cat([pk.to(k.dtype), k], dim=2), torch.cat([pv.to(v.dtype), v], dim=2), pk.shape[2]


def _with_prefix_cols(mask_TT, P):
    """(T, T) bool mask -> (T, P + T): the P prefix columns are visible to every query."""
    T = mask_TT.shape[0]
    return torch.cat([torch.ones(T, P, dtype=torch.bool, device=mask_TT.device), mask_TT], dim=1)


def _fwd_bidirectional(self, x):
    """encoder SelfAttention: no mask."""
    q, k, v = _project_self(self, x, 0)
    k, v, P = _prefix_hook(self, k, v, x.shape[0])
    k, v = _expand(self, k, v)
    return _finish(self, _sdpa(self, q, k, v))


def _fwd_causal(self, x):
    """decoder CausalSelfAttention: causal; one-token steps attend to the whole cache."""
    start = _start(self)
    q, k, v = _project_self(self, x, start)
    k, v = _use_cache(self, k, v, start)
    k, v, P = _prefix_hook(self, k, v, x.shape[0])
    k, v = _expand(self, k, v)
    T = x.shape[1]
    if T > 1 and start > 0:
        raise ValueError("multi-token steps are only supported from an empty cache")
    if P and T > 1:
        mask = _with_prefix_cols(torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), P)
        return _finish(self, _sdpa(self, q, k, v, attn_mask=mask))
    return _finish(self, _sdpa(self, q, k, v, is_causal=T > 1 and not P))


def _prefix_mask(self, T, p, device):
    key = (T, int(p), str(device))
    masks = self.__dict__.setdefault("_mask_cache", {})
    if key not in masks:
        m = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        m[:p, :p] = True                       # same rule as your prefix_attention.py
        masks[key] = m
    return masks[key]


def _fwd_prefix(self, x, p):
    """prefix attention: causal OR [:p, :p]; one-token decoding steps attend to the whole cache."""
    start = _start(self)
    q, k, v = _project_self(self, x, start)
    k, v = _use_cache(self, k, v, start)
    k, v, P = _prefix_hook(self, k, v, x.shape[0])
    k, v = _expand(self, k, v)
    T = x.shape[1]
    if start == 0:
        mask = _prefix_mask(self, T, p, x.device)
        if P:
            mask = _with_prefix_cols(mask, P)
        return _finish(self, _sdpa(self, q, k, v, attn_mask=mask))
    if T == 1:
        return _finish(self, _sdpa(self, q, k, v))
    raise ValueError("multi-token steps are only supported from an empty cache")


def _fwd_cross(self, x, memory):
    """CrossAttention: q from the decoder state, k/v from the encoder memory (cached once per prompt)."""
    assert not getattr(self, "use_rope", False), "RoPE in cross-attention is not supported (your design: use_rope=False)"
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    cache = getattr(self, "_kv_cache", None)
    if cache is not None and "k" in cache:
        k, v = cache["k"], cache["v"]
    else:
        S = memory.shape[1]
        k = self.k_proj(memory).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
    k, v = _expand(self, k, v)
    return _finish(self, _sdpa(self, q, k, v))


_FORWARDS = {
    ("non_causal_self_attention", "SelfAttention"): _fwd_bidirectional,
    ("attention3", "CausalSelfAttention"): _fwd_causal,
    ("cross_attention", "CrossAttention"): _fwd_cross,
    ("prefix_attention", "CausalSelfAttention"): _fwd_prefix,
}


def patch_model(model):
    """Swap every attention forward for the fused/cached version (same parameters)."""
    n = 0
    for mod in model.modules():
        fwd = _FORWARDS.get((type(mod).__module__, type(mod).__name__))
        if fwd is None:
            continue
        mod._kv_cache = None
        if getattr(mod, "use_rope", False):
            mod._apply_rope = sys.modules[type(mod).__module__].apply_rope   # your own apply_rope
        mod.forward = types.MethodType(fwd, mod)
        n += 1
    assert n > 0, "no attention module recognised; did a module/class name change?"
    return model


# ---------------------------------------------------------------------------
# generation sessions: prefill once, then one token per step
# ---------------------------------------------------------------------------

def _core(m):
    """Phase 6: the patched attention module itself, also when a technique wrapped it (e.g. adapters)."""
    for sub in m.modules():
        if hasattr(sub, "_kv_cache"):
            return sub
    return m


def _alloc_self_cache(attn, B, max_len, dtype, device):
    attn = _core(attn)
    shape = (B, attn.n_kv_heads, max_len, attn.d_head)
    attn._kv_cache = {"pos": 0, "k": torch.empty(shape, device=device, dtype=dtype),
                      "v": torch.empty(shape, device=device, dtype=dtype)}
    return attn._kv_cache


@contextmanager
def prefix_session(model, ids, max_new_tokens, dtype):
    """#4: prefill the prompt (mask uses the model's own cfg.prefix), then decode with the KV cache."""
    B, L = ids.shape
    dev = ids.device
    caches = [_alloc_self_cache(b.attn, B, L + max_new_tokens, dtype, dev) for b in model.blocks]
    full_pe = getattr(model, "pos_emb", None)

    def run(tok):
        pos = caches[0]["pos"]
        if full_pe is not None:
            model.pos_emb = full_pe[pos:]     # DecoderLM adds pos_emb[:T]; shift it to rows pos..pos+T
        logits = model(tok)
        for c in caches:
            c["pos"] += tok.shape[1]
        return logits[:, -1]

    try:
        yield run(ids), run
    finally:
        for b in model.blocks:
            _core(b.attn)._kv_cache = None
        if full_pe is not None:
            model.pos_emb = full_pe


@contextmanager
def encdec_session(model, ids, max_new_tokens, dtype, start_id=None):
    """#3: encode the prompt once, cache every decoder layer's cross-attention K/V, decode token by token.
    start_id=None -> the decoder starts from the last source token, else from start_id (e.g. <s>)."""
    B = ids.shape[0]
    dev = ids.device
    memory = model.encode(ids)
    caches = [_alloc_self_cache(b.attn, B, max_new_tokens + 1, dtype, dev) for b in model.decblocks]
    for b in model.decblocks:
        _core(b.crattn)._kv_cache = {}
    full_pe = getattr(model, "pos_emb_dec", None)

    def run(tok):
        pos = caches[0]["pos"]
        if full_pe is not None:
            model.pos_emb_dec = full_pe[pos:]
        logits = model.decode(tok, memory)
        for c in caches:
            c["pos"] += tok.shape[1]
        return logits[:, -1]

    first = ids[:, -1:] if start_id is None else torch.full((B, 1), start_id, device=dev, dtype=ids.dtype)
    try:
        yield run(first), run
    finally:
        for b in model.decblocks:
            _core(b.attn)._kv_cache = None
            _core(b.crattn)._kv_cache = None
        if full_pe is not None:
            model.pos_emb_dec = full_pe


# ---------------------------------------------------------------------------
# sampling + word-budgeted generation
# ---------------------------------------------------------------------------

def sample_next(logits, temperature, top_k, top_p, banned_ids, prev_ids=None, repetition_penalty=1.0):
    logits = torch.nan_to_num(logits.float(), nan=-1e4)   # a diverged checkpoint must not crash sampling
    if banned_ids is not None:
        logits[:, banned_ids] = -float("inf")
    if repetition_penalty != 1.0 and prev_ids is not None:
        seen = torch.gather(logits, 1, prev_ids)
        seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
        logits = logits.scatter(1, prev_ids, seen)
    if not temperature:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p and top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(1)


@torch.inference_mode()
def generate_words(model, tokenizer, prompts, *, kind, max_prompt, target_words, max_new_tokens, batch_size,
                   temperature, top_k, top_p, repetition_penalty, device, amp_dtype, start_id=None, log_every=10):
    """Continue each prompt by `target_words` whitespace words. Returns list[str] in input order.

    kind = "prefix" (#4) or "encdec" (#3). Prompts are bucketed by token length so a batch needs no padding: each
    prompt keeps its last L tokens, L = shortest prompt in the batch (capped at max_prompt)."""
    model.eval()
    eow = torch.tensor(tokenizer.token_eow, device=device, dtype=torch.bool)
    has_bytes = torch.tensor([len(b) > 0 for b in tokenizer.token_bytes], device=device, dtype=torch.bool)
    banned = torch.arange(tokenizer.n_special, device=device)     # never emit pad/unk/</s>/...
    encoded = [tokenizer.encode(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results = [None] * len(prompts)
    cache_dtype = amp_dtype if device.type == "cuda" else torch.float32

    for b, start in enumerate(range(0, len(order), batch_size)):
        idxs = order[start:start + batch_size]
        L = min(min(len(encoded[i]) for i in idxs), max_prompt)
        ids = torch.tensor([encoded[i][-L:] for i in idxs], device=device, dtype=torch.long)
        B = len(idxs)
        words = torch.zeros(B, dtype=torch.int32, device=device)
        in_word = torch.zeros(B, dtype=torch.bool, device=device)     # prompts end on a full word
        done = torch.zeros(B, dtype=torch.bool, device=device)
        out = []
        session = (prefix_session(model, ids, max_new_tokens, cache_dtype) if kind == "prefix"
                   else encdec_session(model, ids, max_new_tokens, cache_dtype, start_id))
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"), session as (logits, step):
            history = ids
            for t in range(max_new_tokens):
                logits = logits.float().clone()
                logits[~in_word, tokenizer.eow_id] = -float("inf")       # a bare </w> after a finished word
                logits[:, len(tokenizer.token_bytes):] = -float("inf")   # ids the tokenizer can't decode
                nxt = sample_next(logits, temperature, top_k, top_p, banned,
                                  history if repetition_penalty != 1.0 else None, repetition_penalty)
                out.append(nxt)
                open_ = in_word | has_bytes[nxt]
                words += (eow[nxt] & open_ & ~done).int()
                in_word = open_ & ~eow[nxt]
                done |= words >= target_words
                if bool(done.all()) or t == max_new_tokens - 1:
                    break
                if repetition_penalty != 1.0:
                    history = torch.cat([history, nxt[:, None]], 1)
                logits = step(nxt[:, None])
        gen = torch.stack(out, 1).tolist()
        for row, i in enumerate(idxs):
            text = tokenizer.decode(gen[row])
            results[i] = " ".join(text.split()[:target_words])
        if log_every and (b + 1) % log_every == 0:
            print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# verification helpers (run in the notebook before training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_equivalence_encdec(make_model, vocab_size, src_len=20, tgt_len=12, rtol=1e-4):
    """Your forward vs patched forward vs the generation path (encdec_session: encode once, cached cross-attention
    K/V, one decoder token per step). Returns relative max errors (fp32, CPU)."""
    torch.manual_seed(0)
    ref = make_model().eval()
    fast = make_model().eval()
    fast.load_state_dict(ref.state_dict())
    patch_model(fast)
    src = torch.randint(0, vocab_size, (2, src_len))
    tgt = torch.randint(0, vocab_size, (2, tgt_len))
    tgt[:, 0] = src[:, -1]                    # session with start_id=None starts from the last source token
    a = ref(src, tgt)
    scale = a.abs().max().item()
    err_full = (a - fast(src, tgt)).abs().max().item() / scale
    with encdec_session(fast, src, tgt_len, torch.float32, start_id=None) as (first, step):
        outs = [first[:, None]]
        for t in range(1, tgt_len):
            outs.append(step(tgt[:, t:t + 1])[:, None])
    err_cache = (a - torch.cat(outs, 1)).abs().max().item() / scale
    assert err_full < rtol and err_cache < rtol, (err_full, err_cache)
    return err_full, err_cache


@torch.no_grad()
def check_equivalence_prefix(make_model, vocab_size, prefix, seq_len=None, rtol=1e-4):
    """Your forward vs patched forward vs (prefill >= prefix tokens, then cached one-token steps)."""
    torch.manual_seed(0)
    ref = make_model().eval()
    fast = make_model().eval()
    fast.load_state_dict(ref.state_dict())
    patch_model(fast)
    seq_len = seq_len or prefix + 12
    idx = torch.randint(0, vocab_size, (2, seq_len))
    a = ref(idx)
    scale = a.abs().max().item()
    err_full = (a - fast(idx)).abs().max().item() / scale
    n_new = seq_len - prefix
    with prefix_session(fast, idx[:, :prefix], n_new, torch.float32) as (first, step):
        outs = [first[:, None]]
        for t in range(prefix, seq_len - 1):
            outs.append(step(idx[:, t:t + 1])[:, None])
    err_cache = (a[:, prefix - 1:seq_len - 1] - torch.cat(outs, 1)).abs().max().item() / scale
    assert err_full < rtol and err_cache < rtol, (err_full, err_cache)
    return err_full, err_cache
