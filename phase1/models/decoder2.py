# TODO 2.2: integrate n_kv_heads through the decoder architecture in this cell.
"""Decoder-only Transformer LM -- Block/DecoderLM updated to thread cfg.n_kv_heads
through to attention (everything else is unchanged from Question 1)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CausalSelfAttention, MultiHeadLatentAttention


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False
    attention_type: str = "standard"  # Q2 only: "standard" or "mla"
    kv_latent_dim: int = None


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
        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model),
            nn.Dropout(dropout)
        ) 

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        # TODO: pass cfg.n_kv_heads as the new required argument to CausalSelfAttention
        if cfg.attention_type == "mla":
            if cfg.pos_emb_type != "sinusoidal":
                raise ValueError("MLA is restricted to Q2 with sinusoidal positions")
            self.attn = MultiHeadLatentAttention(
                cfg.d_model, cfg.n_heads, cfg.kv_latent_dim, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale)
        elif cfg.attention_type == "standard":
            self.attn = CausalSelfAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale)
        else:
            raise ValueError("attention_type must be 'standard' or 'mla'")
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
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
      # unchanged from Question 1


class DecoderLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

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