"""Decoder-only Transformer LM, built from scratch (no nn.Transformer*)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CausalSelfAttention


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