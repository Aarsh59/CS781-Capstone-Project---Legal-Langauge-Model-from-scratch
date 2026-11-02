# Architecture 1 — attention1.py + decoder1.py (your code)
# ======== attention1.py (your A3 code; cross-file import removed, FIX lines marked) ========
"""Multi-head causal self-attention, implemented from raw tensor ops."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Hook for Q1.4. None means the standard Transformer scaling 1/sqrt(d_head).
        self.attention_scale = (1.0 / math.sqrt(self.d_head)) if attention_scale is None else float(attention_scale)

        # TODO: define self.q_proj, self.k_proj, self.v_proj, self.out_proj as nn.Linear
        # (no bias). All should map d_model -> d_model.
        self.q_proj = nn.Linear(d_model,d_model,bias=False)
        self.k_proj = nn.Linear(d_model,d_model,bias=False)
        self.v_proj = nn.Linear(d_model,d_model,bias=False)
        self.out_proj = nn.Linear(d_model,d_model,bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # TODO: register a (max_seq_len, max_seq_len) lower-triangular boolean mask as a
        # buffer named `causal_mask` (True = allowed to attend, False = masked out), so
        # you don't rebuild it on every forward call.
        causal_mask = torch.tril(
            torch.ones(max_seq_len,max_seq_len,dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, C = x.shape

        # TODO 1: project to q, k, v  (each (B, T, C))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # TODO 2: reshape each to (B, n_heads, T, d_head) — split C into (n_heads, d_head)
        #         and move n_heads next to the batch dim.
        q = torch.reshape(q,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        k = torch.reshape(k,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        v = torch.reshape(v,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        
        # TODO 3: compute attention scores = (q @ k^T) * self.attention_scale
        #         -> (B, n_heads, T, T). Do NOT hard-code 1/sqrt(d_head): Q1.4
        #         changes this hook for the scaling ablation.
        attention_scores = (q@k.transpose(2,3))*self.attention_scale
        # TODO 4: apply the causal mask (set disallowed positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(
            ~self.causal_mask[:T,:T],float("-inf")
        )
        # TODO 5: softmax over the last dim, apply self.attn_dropout
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        # TODO 6: weighted sum with v -> (B, n_heads, T, d_head)
        weighted_sum = attention_weights@v
        # TODO 7: reshape back to (B, T, C) and apply self.out_proj + self.resid_dropout
        weighted_sum = torch.reshape(weighted_sum.transpose(1,2),(B,T,C))
        output = self.out_proj(weighted_sum)
        output = self.resid_dropout(output)
        
        return output


# ======== decoder1.py (your A3 code; cross-file import removed, FIX lines marked) ========
"""Decoder-only Transformer LM, built from scratch (no nn.Transformer*)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4          # == n_heads for plain MHA; used in Question 2
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"   # or "rope", used in Question 3
    attention_scale: float = None      # Q1.4 hook; None -> 1/sqrt(d_head)
    norm_first: bool = False            # norm_first=True is for pre-norm else post-norm


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return a (max_len, d_model) tensor of the classic Transformer sinusoidal PE."""
    PE = torch.zeros(max_len, d_model)

    for pos in range(max_len):
        PE[pos, 0::2] = torch.sin(
        pos / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    )
        PE[pos, 1::2] = torch.cos(
        pos / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    )
    return PE


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: two linear layers with a GELU in between, plus dropout
        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Pre-norm Transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).
    OR
    Post-norm Transformer block: LN(x + Attn(x)), then LN(x + FFN(x)).
    """

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout,
                                        attention_scale=cfg.attention_scale)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        # TODO: implement pre-norm when norm_first=True,
        # cfg.norm_first = true for pre norm
        if self.cfg.norm_first is True:
            y1 = self.ln1(x)
            y2 = self.attn(y1)+x
            y3 = self.ln2(y2)
            y4 = self.ffn(y3)+y2
            output=y4
        else:
            y1 = self.attn(x)
            y2 = self.ln1(y1+x)
            y3 = self.ffn(y2)
            y4 = self.ln2(y3+y2)
            output=y4
            
        
        # and post-norm when norm_first=False.
        return output


class DecoderLM(nn.Module):
    """The model class the autograder imports directly. Do not rename it."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)
        # (RoPE is applied inside attention instead of here — see Question 3)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) LongTensor of token ids -> logits (B, T, vocab_size)."""
        B, T = idx.shape
        # TODO: x = token embedding + positional encoding (if sinusoidal)
        x = self.tok_emb(idx)
        if(self.cfg.pos_emb_type == "sinusoidal"):
            x=x+self.pos_emb[:T]        # TODO: run x through every block, then ln_f, then self.head
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        
        return x