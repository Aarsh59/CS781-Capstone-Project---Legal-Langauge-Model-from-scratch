"""Phase-1 checkpoints -> generation models for Phase-6 summarization (plumbing; declare it).

Your Phase-1 model code is used unchanged (phase1_models/). Differences to the Phase 2-5 Backbone:
  * the LM head is KEPT, but moved out of the backbone: GenModel = .backbone (what your technique files receive) +
    .head (the Phase-1 nn.Linear, still tied to the token embedding). Technique target matching (e.g. "q_proj") can
    therefore never hit the head; a frozen embedding (freeze(backbone)) also freezes the tied head.
  * kind "encdec" (C) uses BOTH encoder and decoder (encdec_mode = "seq2seq"); kind "prefix" (D) gets its prefix
    length per call (= source + separator).
  * prompt tuning: backbone.n_virtual_tokens = P -> P placeholder ids are prepended to the decoder input (A/B/D) or the
    encoder input (C). For A/B/D the P output positions are dropped again; for C the encoder memory keeps them (the
    decoder may cross-attend to the soft prompt).
  * prefix tuning: the fastattn prefix_kv hook now also works with the generation KV cache.

Backbone options (cfg["backbone"]):
  truncate_layers       keep the first k blocks (A/B/D blocks; C: k encoder AND k decoder blocks) - KD students.
  dropout               override every nn.Dropout p; None = keep the checkpoint's value.
  max_seq_len_override  A/B/D: int; C: {"encoder": int, "decoder": int}. Rebuilds the model with longer positional
                        tables / masks (they are not stored in the checkpoint). Positions beyond the Phase-1 length
                        were never trained (extrapolation).
"""
import copy
import importlib
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "phase1_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastattn_ab   # noqa: E402
import fastattn_cd   # noqa: E402


def _clean_state(sd):
    out = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def build_lm(spec, override=None):
    spec = copy.deepcopy(spec)
    kind = spec.get("kind", "decoder")
    arch = importlib.import_module(spec["module"])
    if kind in ("decoder", "prefix"):
        if override:
            spec["model"]["max_seq_len"] = int(override)
        lm = arch.DecoderLM(arch.LMConfig(**spec["model"]))
    elif kind == "encdec":
        if override:
            spec["encoder"]["max_seq_len"] = int(override.get("encoder") or spec["encoder"]["max_seq_len"])
            spec["decoder"]["max_seq_len"] = int(override.get("decoder") or spec["decoder"]["max_seq_len"])
        lm = arch.EncDecLM(arch.EncLMConfig(**spec["encoder"]), arch.DecLMConfig(**spec["decoder"]))
    else:
        raise ValueError(f"unknown model kind {kind}")
    return lm, kind, spec


def load_phase1(path, override=None, load_weights=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    lm, kind, spec = build_lm(ck["config"], override)
    if load_weights:
        sd = _clean_state(ck["model"])
        if override:     # positional buffers / masks are rebuilt for the new length: skip shape-mismatched buffers
            own = lm.state_dict()
            bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
            params = dict(lm.named_parameters())
            assert not [k for k in bad if k in params], f"override changes parameter shapes: {bad}"
            sd = {k: v for k, v in sd.items() if k not in bad}
            missing, unexpected = lm.load_state_dict(sd, strict=False)
            assert not [k for k in missing if k in params] and not unexpected, (missing, unexpected)
        else:
            lm.load_state_dict(sd)
    info = dict(path=path, kind=kind, module=spec["module"], tag=spec.get("tag"), step=ck.get("step"),
                best_val=ck.get("best_val"))
    return lm, kind, spec, info


class GenBackbone(nn.Module):
    """Attributes for technique code (same names as Phases 2-5): .lm, .kind, .blocks, .d_model, .max_len,
    .n_virtual_tokens, .encdec_mode ("seq2seq" for C), get_input_embedding / set_input_embedding."""

    def __init__(self, lm, kind, truncate_layers=None, dropout=None):
        super().__init__()
        self.kind = kind
        self.lm = lm
        self.encdec_mode = "seq2seq" if kind == "encdec" else None
        if kind == "encdec":
            self.d_model = lm.cfg1.d_model
            self.max_len = lm.cfg1.max_seq_len          # encoder (source) limit
            self.max_tgt_len = lm.cfg2.max_seq_len      # decoder limit
        else:
            self.d_model = lm.cfg.d_model
            self.max_len = lm.cfg.max_seq_len
            self.max_tgt_len = None
        self.n_virtual_tokens = 0
        if truncate_layers:
            k = int(truncate_layers)
            if kind == "encdec":
                lm.encblocks = nn.ModuleList(list(lm.encblocks)[:k])
                lm.decblocks = nn.ModuleList(list(lm.decblocks)[:k])
            else:
                lm.blocks = nn.ModuleList(list(lm.blocks)[:k])
        if dropout is not None:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)

    @property
    def blocks(self):
        if self.kind == "encdec":
            return list(self.lm.encblocks) + list(self.lm.decblocks)
        return list(self.lm.blocks)

    def get_input_embedding(self):
        return self.lm.tok_emb1 if self.kind == "encdec" else self.lm.tok_emb

    def set_input_embedding(self, module):
        if self.kind == "encdec":
            self.lm.tok_emb1 = module
        else:
            self.lm.tok_emb = module

    def _with_virtual(self, ids):
        P = self.n_virtual_tokens
        if not P:
            return ids
        return torch.cat([torch.zeros(ids.shape[0], P, dtype=ids.dtype, device=ids.device), ids], dim=1)

    # ---- teacher-forced forwards (training) ----
    def hidden(self, ids, prefix_len=None):
        """A/B/D: ids (B, T) -> final hidden states (B, T, d). D: bidirectional over the first prefix_len real tokens."""
        P = self.n_virtual_tokens
        if self.kind == "prefix":
            assert prefix_len is not None
            self.lm.cfg.prefix = int(prefix_len) + P
        h = self.lm(self._with_virtual(ids))
        return h[:, P:] if P else h

    def encode(self, src):
        return self.lm.encode(self._with_virtual(src))

    def decode_hidden(self, tgt_in, memory):
        return self.lm.decode(tgt_in, memory)


class GenModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def target_logits(self, batch):
        """batch from summ_data.collate -> (logits (N, V) at every labelled position, labels (N,))."""
        bb = self.backbone
        if bb.kind == "encdec":
            h = bb.decode_hidden(batch["dec_in"], bb.encode(batch["src"]))
        else:
            h = bb.hidden(batch["ids"], batch.get("prefix_len"))
        sel = batch["labels"] != -100
        return self.head(h[sel]), batch["labels"][sel]


def build_model(ckpt_path, bcfg, vocab_size=None, load_weights=True, fast_attention=True):
    lm, kind, spec, info = load_phase1(ckpt_path, bcfg.get("max_seq_len_override"), load_weights)
    v = spec["model"]["vocab_size"] if "model" in spec else spec["encoder"]["vocab_size"]
    if vocab_size is not None:
        assert v == vocab_size, f"{ckpt_path}: vocab {v} != tokenizer vocab {vocab_size} (different tokenizer?)"
    if fast_attention:
        (fastattn_ab if kind == "decoder" else fastattn_cd).patch_model(lm)
    head = lm.head                       # tied to tok_emb / tok_emb2 (same Parameter object)
    lm.head = nn.Identity()
    bb = GenBackbone(lm, kind, truncate_layers=bcfg.get("truncate_layers"), dropout=bcfg.get("dropout"))
    info["phase1_max_len"] = (spec["model"]["max_seq_len"] if "model" in spec
                              else {"encoder": spec["encoder"]["max_seq_len"], "decoder": spec["decoder"]["max_seq_len"]})
    return GenModel(bb, head), spec, info


# ---------------------------------------------------------------------------
# generation sessions: prefill(src) once, then step(tok) -> hidden (B, d) of the last position
# ---------------------------------------------------------------------------

class _StepEmbedding(nn.Module):
    """During cached decoding with prompt tuning: run your soft-prompt embedding on [P placeholders + ids] and keep
    only the real positions, so the wrapper's contract (first P positions = prompt) is respected."""

    def __init__(self, wrapped, P):
        super().__init__()
        self.wrapped, self.P = wrapped, P

    def forward(self, ids):
        z = torch.zeros(ids.shape[0], self.P, dtype=ids.dtype, device=ids.device)
        return self.wrapped(torch.cat([z, ids], 1))[:, self.P:]


@contextmanager
def session(model, src, max_new_tokens, dtype, start_id=None):
    """src: (B, S) real source ids (decoder kinds: already include the separator at the end).
    Yields (h_last (B, d), step) where step(tok (B, 1)) -> h_last (B, d)."""
    bb = model.backbone
    lm, P = bb.lm, bb.n_virtual_tokens
    ids = bb._with_virtual(src)
    if bb.kind == "encdec":
        with fastattn_cd.encdec_session(lm, ids, max_new_tokens, dtype, start_id=start_id) as (h0, run):
            yield h0, run
        return
    if bb.kind == "prefix":
        lm.cfg.prefix = ids.shape[1]                     # whole (virtual + source + separator) block bidirectional
        with fastattn_cd.prefix_session(lm, ids, max_new_tokens, dtype) as (h0, run):
            if P:
                emb = bb.get_input_embedding()
                bb.set_input_embedding(_StepEmbedding(emb, P))
                try:
                    yield h0, run
                finally:
                    bb.set_input_embedding(emb)
            else:
                yield h0, run
        return
    with fastattn_ab.kv_cache(lm, ids.shape[0], ids.shape[1] + max_new_tokens, dtype):
        h0 = fastattn_ab.step(lm, ids)[:, -1]
        emb = bb.get_input_embedding()
        if P:
            bb.set_input_embedding(_StepEmbedding(emb, P))
        try:
            yield h0, (lambda tok: fastattn_ab.step(lm, tok)[:, -1])
        finally:
            if P:
                bb.set_input_embedding(emb)
