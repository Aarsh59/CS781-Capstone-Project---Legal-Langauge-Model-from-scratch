"""Phase-1 checkpoints -> token feature extractors for PCR (prior case retrieval) (plumbing; declare it).

Your Phase-1 model code is used unchanged (phase1_models/: arch1/arch3 for A/B, enc_dec + parts for C,
prefix_lm + prefix_attention for D). The LM head is replaced by nn.Identity, so YOUR forward() returns the
final-LayerNorm hidden states (B, T, d_model) instead of vocabulary logits. Attention runs through the same
fastattn patches as Phase 1 (fused SDPA, identical maths; smoke_test.py re-checks equality).

Backbone options (cfg["backbone"]):
  prefix         D only: "full" -> prefix = window length (every token sees the whole window),
                 int p -> bidirectional over the first p tokens, causal after (your Phase-1 D used 640),
                 0 -> fully causal.
  encdec_mode    C only: "encoder" -> encoder states only (decoder weights dropped; bidirectional),
                 "decoder" -> the window is fed to the encoder AND to the decoder (causal self-attention +
                 cross-attention to the encoder states); features = decoder final states.
  truncate_layers keep only the first k transformer blocks (e.g. a smaller KD student); None = all.
  dropout        override every nn.Dropout p in the backbone; None = keep the checkpoint's value.
"""
import importlib
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "phase1_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastattn_ab   # noqa: E402  (A/B attention: arch1 / arch3)
import fastattn_cd   # noqa: E402  (C/D attention: encoder / decoder / cross / prefix)


def _clean_state(sd):
    out = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def build_lm(spec):
    """spec = the 'config' dict saved inside a Phase-1 best.pt -> (lm, kind)."""
    kind = spec.get("kind", "decoder")
    arch = importlib.import_module(spec["module"])
    if kind == "decoder":
        lm = arch.DecoderLM(arch.LMConfig(**spec["model"]))
    elif kind == "prefix":
        lm = arch.DecoderLM(arch.LMConfig(**spec["model"]))
    elif kind == "encdec":
        lm = arch.EncDecLM(arch.EncLMConfig(**spec["encoder"]), arch.DecLMConfig(**spec["decoder"]))
    else:
        raise ValueError(f"unknown model kind {kind}")
    return lm, kind


def load_phase1(path, load_weights=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = ck["config"]
    lm, kind = build_lm(spec)
    if load_weights:
        lm.load_state_dict(_clean_state(ck["model"]))
    info = dict(path=path, kind=kind, module=spec["module"], tag=spec.get("tag"), step=ck.get("step"),
                best_val=ck.get("best_val"))
    return lm, kind, spec, info


class Backbone(nn.Module):
    """forward(ids (B, T) long) -> hidden states (B, T, d_model).
    Attributes for technique code: .lm (your Phase-1 module), .kind, .blocks (list of transformer blocks,
    input -> output order; C "decoder" mode = encoder blocks then decoder blocks), .d_model, .max_len."""

    def __init__(self, lm, kind, prefix=None, encdec_mode=None, truncate_layers=None, dropout=None):
        super().__init__()
        self.kind = kind
        if kind == "encdec":
            assert encdec_mode in ("encoder", "decoder"), "C needs backbone.encdec_mode = 'encoder' | 'decoder'"
            self.encdec_mode = encdec_mode
            if encdec_mode == "encoder":
                for name in ("decblocks", "tok_emb2", "ln_f_dec", "head"):
                    delattr(lm, name)
                lm.ln_f_dec = lm.head = None
            else:
                lm.head = nn.Identity()
            self.d_model = lm.cfg1.d_model
            self.max_len = lm.cfg1.max_seq_len if encdec_mode == "encoder" else min(lm.cfg1.max_seq_len,
                                                                                    lm.cfg2.max_seq_len)
        else:
            if kind == "prefix":
                assert prefix is not None, "D needs backbone.prefix = 'full' | int (0 = causal)"
            self.prefix = prefix
            lm.head = nn.Identity()
            self.d_model = lm.cfg.d_model
            self.max_len = lm.cfg.max_seq_len
        self.lm = lm
        self.n_virtual_tokens = 0          # prompt tuning (Phase 3): set by your technique; see forward()
        if truncate_layers:
            self._truncate(int(truncate_layers))
        if dropout is not None:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)

    @property
    def blocks(self):
        lm = self.lm
        if self.kind == "encdec":
            return list(lm.encblocks) + (list(lm.decblocks) if self.encdec_mode == "decoder" else [])
        return list(lm.blocks)

    def _truncate(self, k):
        lm = self.lm
        if self.kind == "encdec":
            assert self.encdec_mode == "encoder", "truncate_layers for C is only supported in encoder mode"
            lm.encblocks = nn.ModuleList(list(lm.encblocks)[:k])
        else:
            lm.blocks = nn.ModuleList(list(lm.blocks)[:k])

    # ---- prompt-tuning hook (Phase 3 plumbing) ------------------------------------------------------------------
    # If a technique sets self.n_virtual_tokens = P > 0, forward() prepends P placeholder ids (<pad>) to every window
    # and drops the first P output states again, so callers always get (B, T, d) for their real tokens. The technique
    # makes those P positions learnable by wrapping the input embedding (get_input_embedding / set_input_embedding):
    # its wrapper returns the normal embeddings but replaces positions [:, :P] with its learned prompt vectors.
    # Positional encodings / masks see a sequence of length P + T, so windows must satisfy P + T <= max_len.
    def get_input_embedding(self):
        return self.lm.tok_emb1 if self.kind == "encdec" else self.lm.tok_emb

    def set_input_embedding(self, module):
        if self.kind == "encdec":
            self.lm.tok_emb1 = module      # encoder input (the decoder in "decoder" mode keeps its own embedding)
        else:
            self.lm.tok_emb = module

    def forward(self, ids):
        P = self.n_virtual_tokens
        if P:
            ids = torch.cat([torch.zeros(ids.shape[0], P, dtype=ids.dtype, device=ids.device), ids], dim=1)
        out = self._forward(ids)
        return out[:, P:] if P else out

    def _forward(self, ids):
        lm = self.lm
        if self.kind == "decoder":
            return lm(ids)
        if self.kind == "prefix":
            lm.cfg.prefix = ids.shape[1] if self.prefix == "full" else int(self.prefix) + self.n_virtual_tokens
            return lm(ids)
        memory = lm.encode(ids)
        if self.encdec_mode == "encoder":
            return memory
        return lm.decode(ids, memory)


def patch_fast_attention(lm, kind):
    if kind == "decoder":
        fastattn_ab.patch_model(lm)
    else:
        fastattn_cd.patch_model(lm)
    return lm


def build_backbone(ckpt_path, bcfg, vocab_size=None, load_weights=True, fast_attention=True):
    lm, kind, spec, info = load_phase1(ckpt_path, load_weights)
    if vocab_size is not None:
        v = spec["model"]["vocab_size"] if "model" in spec else spec["encoder"]["vocab_size"]
        assert v == vocab_size, f"{ckpt_path}: vocab {v} != tokenizer vocab {vocab_size} (different tokenizer?)"
    if fast_attention:
        patch_fast_attention(lm, kind)
    bb = Backbone(lm, kind, prefix=bcfg.get("prefix"), encdec_mode=bcfg.get("encdec_mode"),
                  truncate_layers=bcfg.get("truncate_layers"), dropout=bcfg.get("dropout"))
    return bb, spec, info
