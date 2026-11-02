"""Decoder-only Transformer LM -- Block now passes use_rope through to attention, and
DecoderLM only builds/adds the sinusoidal table when pos_emb_type == "sinusoidal"
(everything else unchanged from Question 2)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention3 import CausalSelfAttention
from cross_attention import CrossAttention


@dataclass
class DecLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"   # "sinusoidal" or "rope"
    attention_scale: float = None      # preserve Q1.4 hook
    norm_first: bool = False


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
    return PE  # unchanged from Question 1


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DecBlock(nn.Module):
    def __init__(self, cfg: DecLMConfig):
        super().__init__()
        self.cfg=cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        # TODO: pass use_rope=(cfg.pos_emb_type == "rope") to CausalSelfAttention
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
                                        cfg.dropout, use_rope=(cfg.pos_emb_type == "rope"),
                                        attention_scale=cfg.attention_scale)
        self.crattn = CrossAttention(cfg.d_model , cfg.n_heads,cfg.n_kv_heads,cfg.max_seq_len,
        cfg.dropout,use_rope=False,attention_scale=cfg.attention_scale)

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x , memory):
        if self.cfg.norm_first is True:
            y1 = self.ln1(x)
            y2 = self.attn(y1)+x
            y3 = self.ln2(y2)
            y4 = self.crattn(y3,memory)+y2
            y5 = self.ln3(y4)
            y6 = self.ffn(y5)+y4
            output=y6
        else:
            y1 = self.attn(x)
            y2 = self.ln1(y1+x)
            y3 = self.crattn(y2,memory)
            y4 = self.ln2(y3+y2)
            y5 = self.ffn(y4)
            y6 = self.ln3(y5+y4)
            output=y6
            
        
        # and post-norm when norm_first=False.
        return output


class DecoderLM(nn.Module):
    def __init__(self, cfg: DecLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)
        # NOTE: when cfg.pos_emb_type == "rope", we deliberately do NOT build a
        # sinusoidal table at all -- position information is injected inside every
        # attention layer instead of once at the input.
        self.blocks = nn.ModuleList([DecBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor , memory:torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        # TODO: x = token embedding + positional encoding (if sinusoidal)
        x = self.tok_emb(idx)
        if(self.cfg.pos_emb_type == "sinusoidal"):
            x=x+self.pos_emb[:T]        # TODO: run x through every block, then ln_f, then self.head
        for block in self.blocks:
            x = block(x,memory)
        x = self.ln_f(x)
        x = self.head(x)
        
        return x  # unchanged from Question 1 (just remember: only add
        # self.pos_emb when self.cfg.pos_emb_type == "sinusoidal")
