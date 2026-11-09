# Plumbing: fused SDPA attention + KV cache + generation around your modules
"""Efficiency plumbing around the hand-written A3 models (written with Claude; declare it).

Nothing here changes an architecture: parameters, shapes and the maths of every model stay
exactly as defined in arch1/arch2/arch3. `patch_model` swaps each attention module's forward
for one that computes the *same* function with fused kernels, and adds an optional KV cache
for generation. `check_equivalence` in the notebook verifies outputs match the original code.

* Training: F.scaled_dot_product_attention(is_causal=True). On T4 (sm_75) PyTorch cannot use
  the FlashAttention-2 kernel (needs sm_80+); it dispatches to the memory-efficient fused
  kernel, which gives the same benefit: no (T x T) score matrix materialised in HBM.
* Generation: per-layer KV cache (keys/values for MHA/GQA, the compressed latent for MLA),
  prefill once, then one token per step.
"""
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# attention forward (drop-in for all three hand-written attention classes)
# ---------------------------------------------------------------------------

def _sdpa(q, k, v, causal, dropout_p, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p, scale=scale)


def _fast_attention_forward(self, x):
    B, T, C = x.shape
    cache = getattr(self, "_kv_cache", None)
    start = cache["pos"] if cache is not None else 0
    n_heads, d_head = self.n_heads, self.d_head
    dropout_p = self.attn_dropout.p if self.training else 0.0

    q = self.q_proj(x).view(B, T, n_heads, d_head).transpose(1, 2)
    n_prefix = 0                                            # prefix tuning is only wired for MHA / GQA

    if hasattr(self, "kv_down_proj"):                       # MLA: cache the shared latent
        latent = self.kv_down_proj(x)
        if cache is not None:
            cache["latent"][:, start:start + T] = latent
            latent = cache["latent"][:, :start + T]
        S = latent.shape[1]
        k = self.k_up_proj(latent).view(B, S, n_heads, d_head).transpose(1, 2)
        v = self.v_up_proj(latent).view(B, S, n_heads, d_head).transpose(1, 2)
    else:                                                   # MHA / GQA (+ optional RoPE)
        n_kv = getattr(self, "n_kv_heads", n_heads)
        k = self.k_proj(x).view(B, T, n_kv, d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n_kv, d_head).transpose(1, 2)
        if getattr(self, "use_rope", False):
            cos = self.rope_cos[start:start + T]
            sin = self.rope_sin[start:start + T]
            q = self._apply_rope(q.float(), cos, sin).to(v.dtype)
            k = self._apply_rope(k.float(), cos, sin).to(v.dtype)
        if cache is not None:
            cache["k"][:, :, start:start + T] = k
            cache["v"][:, :, start:start + T] = v
            k = cache["k"][:, :, :start + T]
            v = cache["v"][:, :, :start + T]
        k, v, n_prefix = _add_prefix(self, k, v, B, cache)
        n_rep = n_heads // n_kv
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

    # prefill (start == 0) is causal; a single decode step attends to the whole cache
    causal = T > 1
    if causal and start > 0:
        raise ValueError("multi-token steps are only supported from an empty cache")
    if n_prefix:                                            # prefix tuning: every query also sees the P prefix keys
        mask = _causal_with_prefix(T, n_prefix, x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.attention_scale)
    else:
        y = _sdpa(q, k, v, causal, dropout_p, self.attention_scale)
    y = y.transpose(1, 2).reshape(B, T, n_heads * d_head)
    return self.resid_dropout(self.out_proj(y))


# ---------------------------------------------------------------------------
# prefix-tuning hook (Phase 3 plumbing): an attention module may carry `prefix_kv`, a callable
# prefix_kv(batch_size) -> (pk, pv), each (B, n_kv_heads, P, d_head). Those P key/value vectors are prepended to the
# keys/values (no RoPE is applied to them) and every query may attend to them. Without `prefix_kv`: unchanged.
# ---------------------------------------------------------------------------

def _add_prefix(attn, k, v, B, cache):
    fn = getattr(attn, "prefix_kv", None)
    if fn is None:
        return k, v, 0
    assert cache is None, "prefix tuning is not supported together with the generation KV cache"
    pk, pv = fn(B)
    assert pk.shape[0] == B and pk.shape[1] == k.shape[1] and pk.shape[3] == k.shape[3], \
        f"prefix_kv must return (B, n_kv_heads, P, d_head) = ({B}, {k.shape[1]}, P, {k.shape[3]}), got {tuple(pk.shape)}"
    return torch.cat([pk.to(k.dtype), k], dim=2), torch.cat([pv.to(v.dtype), v], dim=2), pk.shape[2]


def _causal_with_prefix(T, P, device):
    m = torch.ones(T, P + T, dtype=torch.bool, device=device)
    m[:, P:] = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    return m


def attention_modules(model):
    return [block.attn for block in model.blocks]


def patch_model(model):
    """Swap every attention forward for the fused/cached version (same parameters)."""
    for attn in attention_modules(model):
        attn._kv_cache = None
        if getattr(attn, "use_rope", False):
            # the user's own apply_rope, from the module that defines this attention class
            attn._apply_rope = sys.modules[type(attn).__module__].apply_rope
        attn.forward = types.MethodType(_fast_attention_forward, attn)
    return model


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

@contextmanager
def kv_cache(model, batch_size, max_len, dtype):
    """Allocate a per-layer cache. Inside the context call `step(model, ids)` repeatedly."""
    dev = next(model.parameters()).device
    caches = []
    for attn in attention_modules(model):
        c = {"pos": 0}
        if hasattr(attn, "kv_down_proj"):
            c["latent"] = torch.empty(batch_size, max_len, attn.kv_latent_dim, device=dev, dtype=dtype)
        else:
            n_kv = getattr(attn, "n_kv_heads", attn.n_heads)
            shape = (batch_size, n_kv, max_len, attn.d_head)
            c["k"] = torch.empty(shape, device=dev, dtype=dtype)
            c["v"] = torch.empty(shape, device=dev, dtype=dtype)
        attn._kv_cache = c
        caches.append(c)
    full_pe = getattr(model, "pos_emb", None)
    model._fast_state = {"caches": caches, "full_pe": full_pe}
    try:
        yield
    finally:
        for attn in attention_modules(model):
            attn._kv_cache = None
        if full_pe is not None:
            model.pos_emb = full_pe
        del model._fast_state


def step(model, ids):
    """Run `ids` (B, T) through the model at the current cache position; returns logits (B, T, V)."""
    state = model._fast_state
    pos = state["caches"][0]["pos"]
    if state["full_pe"] is not None:
        # DecoderLM.forward adds self.pos_emb[:T]; shift the table so rows pos..pos+T are used
        model.pos_emb = state["full_pe"][pos:]
    logits = model(ids)
    for c in state["caches"]:
        c["pos"] += ids.shape[1]
    return logits


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
        drop = probs.cumsum(-1) - probs > top_p          # keep the smallest set with mass >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(1)


def ngram_ban(seq, n, vocab):
    """(B, V) bool: tokens that would complete an n-gram (in token ids) already present in `seq` (B, S).
    For each row, every earlier occurrence of the last n-1 tokens bans the token that followed it."""
    B, S = seq.shape
    if n < 2 or S < n:
        return torch.zeros(B, vocab, dtype=torch.bool, device=seq.device)
    grams = seq.unfold(1, n, 1)                                   # (B, S-n+1, n)
    hit = (grams[:, :, :-1] == seq[:, None, S - n + 1:]).all(-1)  # earlier (n-1)-gram == current suffix
    cnt = torch.zeros(B, vocab, device=seq.device).scatter_add_(1, grams[:, :, -1], hit.float())
    return cnt > 0


@torch.inference_mode()
def generate_words(model, tokenizer, prompts, *, max_seq_len, target_words, max_new_tokens,
                   batch_size, temperature, top_k, top_p, repetition_penalty, device, amp_dtype,
                   log_every=10, no_repeat_ngram=0, ngram_scope="output"):
    """no_repeat_ngram = n > 1 forbids repeating any n-gram of TOKEN ids (HF no_repeat_ngram_size), counted over
    the generated text only (ngram_scope="output") or over prompt + generated text ("all")."""
    assert ngram_scope in ("output", "all"), ngram_scope
    """Continue each prompt by `target_words` whitespace words. Returns list[str] in input order.

    Prompts are bucketed by token length so a batch needs no padding: every prompt in a batch
    keeps its last L tokens, L = shortest prompt in the batch (capped so prompt + new tokens fit
    max_seq_len). Words are counted from the tokenizer's end-of-word flag, so we stop as soon as
    every row has produced target_words complete words.
    """
    model.eval()
    eow = torch.tensor(tokenizer.token_eow, device=device, dtype=torch.bool)
    has_bytes = torch.tensor([len(b) > 0 for b in tokenizer.token_bytes], device=device, dtype=torch.bool)
    banned = torch.arange(tokenizer.n_special, device=device)     # never emit pad/unk/</s>/...
    encoded = [tokenizer.encode(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    max_prompt = max_seq_len - max_new_tokens
    assert max_prompt > 0, "max_new_tokens must be smaller than max_seq_len"
    results = [None] * len(prompts)

    for b, start in enumerate(range(0, len(order), batch_size)):
        idxs = order[start:start + batch_size]
        L = min(min(len(encoded[i]) for i in idxs), max_prompt)
        ids = torch.tensor([encoded[i][-L:] for i in idxs], device=device, dtype=torch.long)
        B = len(idxs)
        words = torch.zeros(B, dtype=torch.int32, device=device)
        in_word = torch.zeros(B, dtype=torch.bool, device=device)     # prompts end on a full word
        done = torch.zeros(B, dtype=torch.bool, device=device)
        out = []
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"), \
                kv_cache(model, B, L + max_new_tokens, amp_dtype if device.type == "cuda" else torch.float32):
            logits = step(model, ids)[:, -1]
            history = ids
            for t in range(max_new_tokens):
                # a bare </w> right after a finished word never occurs in encoded text: forbid it
                logits = logits.float().clone()
                logits[~in_word, tokenizer.eow_id] = -float("inf")
                logits[:, len(tokenizer.token_bytes):] = -float("inf")   # ids the tokenizer can't decode
                if no_repeat_ngram and no_repeat_ngram > 1:
                    seq = history if ngram_scope == "all" else (torch.stack(out, 1) if out else ids[:, :0])
                    ban = ngram_ban(seq, no_repeat_ngram, logits.size(1))
                    ban &= ~(ban | torch.isinf(logits)).all(-1, keepdim=True)  # never leave a row with no option
                    logits = logits.masked_fill(ban, -float("inf"))
                nxt = sample_next(logits, temperature, top_k, top_p, banned,
                                  history if repetition_penalty != 1.0 else None, repetition_penalty)
                out.append(nxt)
                # a word ends on an end-of-word token that closes something non-empty
                open_ = in_word | has_bytes[nxt]
                words += (eow[nxt] & open_ & ~done).int()
                in_word = open_ & ~eow[nxt]
                done |= words >= target_words
                if bool(done.all()):
                    break
                if repetition_penalty != 1.0 or (no_repeat_ngram and ngram_scope == "all"):
                    history = torch.cat([history, nxt[:, None]], 1)
                logits = step(model, nxt[:, None])[:, -1]
        gen = torch.stack(out, 1).tolist()
        for row, i in enumerate(idxs):
            text = tokenizer.decode(gen[row])
            results[i] = " ".join(text.split()[:target_words])
        if log_every and (b + 1) % log_every == 0:
            print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return results


@torch.inference_mode()
def generate_beam(model, tokenizer, prompts, *, max_seq_len, target_words, max_new_tokens, batch_size,
                  beam_size, device, amp_dtype, no_repeat_ngram=0, ngram_scope="output", length_penalty=1.0,
                  log_every=10):
    """Beam search with the same word budget / token constraints as generate_words.
    Beams are ranked by sum(log p) / length**length_penalty (1.0 = mean log-prob per token). A beam that has
    produced `target_words` words is frozen (score and length fixed) and keeps competing as a finished
    hypothesis. Optional n-gram blocking per beam (same rule as generate_words). Returns list[str] in input order."""
    assert ngram_scope in ("output", "all"), ngram_scope
    model.eval()
    W = int(beam_size)
    eow = torch.tensor(tokenizer.token_eow, device=device, dtype=torch.bool)
    has_bytes = torch.tensor([len(b) > 0 for b in tokenizer.token_bytes], device=device, dtype=torch.bool)
    n_decodable = len(tokenizer.token_bytes)
    encoded = [tokenizer.encode(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    max_prompt = max_seq_len - max_new_tokens
    per_batch = max(1, batch_size // W)
    results = [None] * len(prompts)
    neg_inf = -float("inf")

    for b, start in enumerate(range(0, len(order), per_batch)):
        idxs = order[start:start + per_batch]
        P = len(idxs)
        R = P * W
        L = min(min(len(encoded[i]) for i in idxs), max_prompt)
        prompt_ids = torch.tensor([encoded[i][-L:] for i in idxs], device=device, dtype=torch.long)
        ids = prompt_ids.repeat_interleave(W, 0)                                   # (R, L): row = p * W + beam
        words = torch.zeros(R, dtype=torch.int32, device=device)
        in_word = torch.zeros(R, dtype=torch.bool, device=device)
        done = torch.zeros(R, dtype=torch.bool, device=device)
        cum = torch.zeros(R, device=device)
        length = torch.zeros(R, device=device)
        gen = torch.zeros(R, 0, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"), \
                kv_cache(model, R, L + max_new_tokens, amp_dtype if device.type == "cuda" else torch.float32):
            logits = step(model, ids)[:, -1]
            for t in range(max_new_tokens):
                lp = torch.nan_to_num(logits.float(), nan=-1e4)
                lp[:, :tokenizer.n_special] = neg_inf
                lp[~in_word, tokenizer.eow_id] = neg_inf
                lp[:, n_decodable:] = neg_inf
                if no_repeat_ngram and no_repeat_ngram > 1:
                    seq = gen if ngram_scope == "output" else torch.cat([ids, gen], 1)
                    ban = ngram_ban(seq, no_repeat_ngram, lp.size(1))
                    ban &= ~(ban | torch.isinf(lp)).all(-1, keepdim=True)
                    lp = lp.masked_fill(ban, neg_inf)
                lp = lp.log_softmax(-1)
                V = lp.size(1)
                # candidate scores (length-normalised); finished beams keep their score via one frozen continuation
                cand = (cum[:, None] + lp) / (length[:, None] + 1).pow(length_penalty)
                frozen_tok = lp.argmax(-1)
                frozen = torch.full_like(cand, neg_inf)
                frozen.scatter_(1, frozen_tok[:, None], (cum / length.clamp(min=1).pow(length_penalty))[:, None])
                cand = torch.where(done[:, None], frozen, cand)
                if t == 0:                                                         # identical beams: expand beam 0 only
                    cand.view(P, W, V)[:, 1:] = neg_inf
                top_s, top_i = cand.view(P, W * V).topk(W, dim=-1)                 # (P, W)
                parent = (top_i // V) + torch.arange(P, device=device)[:, None] * W
                parent, tok = parent.reshape(-1), (top_i % V).reshape(-1)
                was_done = done[parent]
                # reorder beam state + KV cache to the chosen parents
                for c in model._fast_state["caches"]:
                    pos = c["pos"]
                    c["k"][:, :, :pos] = c["k"][parent, :, :pos]
                    c["v"][:, :, :pos] = c["v"][parent, :, :pos]
                words, in_word, done = words[parent], in_word[parent], done[parent].clone()
                cum = torch.where(was_done, cum[parent], cum[parent] + lp[parent, tok])
                length = torch.where(was_done, length[parent], length[parent] + 1)
                gen = torch.cat([gen[parent], tok[:, None]], 1)
                open_ = in_word | has_bytes[tok]
                words = words + (eow[tok] & open_ & ~was_done).int()
                in_word = open_ & ~eow[tok]
                done = was_done | (words >= target_words)
                if bool(done.all()):
                    break
                logits = step(model, tok[:, None])[:, -1]
        score = (cum / length.clamp(min=1).pow(length_penalty)).view(P, W)
        score = torch.where(done.view(P, W), score, score - 1e6)                     # prefer finished beams
        best = score.argmax(-1) + torch.arange(P, device=device) * W
        rows = gen[best].tolist()
        for row, i in enumerate(idxs):
            text = tokenizer.decode(rows[row])
            results[i] = " ".join(text.split()[:target_words])
        if log_every and (b + 1) % log_every == 0:
            print(f"  beam generated {min(start + per_batch, len(order))}/{len(order)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# verification helpers (run in the notebook before training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_equivalence(make_model, vocab_size, seq_len=48, rtol=1e-4):
    """Original forward vs patched forward vs KV-cached incremental decoding, fp32 on CPU.
    Returns max |difference| relative to the largest |logit| of the original model."""
    torch.manual_seed(0)
    ref = make_model().eval()
    fast = make_model().eval()
    fast.load_state_dict(ref.state_dict())
    patch_model(fast)
    idx = torch.randint(0, vocab_size, (2, seq_len))
    a = ref(idx)
    scale = a.abs().max().item()
    err_full = (a - fast(idx)).abs().max().item() / scale
    with kv_cache(fast, 2, seq_len, torch.float32):
        pre = step(fast, idx[:, :seq_len // 2])
        steps = [step(fast, idx[:, t:t + 1]) for t in range(seq_len // 2, seq_len)]
    err_cache = (a - torch.cat([pre] + steps, 1)).abs().max().item() / scale
    assert err_full < rtol and err_cache < rtol, (err_full, err_cache)
    return err_full, err_cache