# Architecture 2 option — rope.py + attention3.py + decoder3.py (your code: GQA + RoPE)
# ======== rope.py (your A3 code; cross-file import removed, FIX lines marked) ========
"""Rotary Positional Embeddings, implemented from scratch."""
import torch


def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0):
    """
    Returns (cos, sin) tensors of shape (max_seq_len, d_head) ready to be applied to
    query/key tensors of head dimension d_head. d_head must be even. This scaffold uses
    the split-half rotate_half convention: channel i is paired with channel i+d_head/2.
    """
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    # TODO: compute inverse frequencies: 1 / base^(2i/d_head) for i in [0, d_head/2)
    inverse = [1/(base**(2*i/d_head)) for i in range(0,d_head//2)]
    inverse = torch.tensor(inverse, dtype=torch.float32)  # FIX (Claude): torch.outer needs a tensor, not a list
    # TODO: outer product of positions [0..max_seq_len) with inverse frequencies -> (T, d_head/2)
    pos = torch.arange(max_seq_len, dtype=torch.float32)  # FIX (Claude): float, to match `inverse` in torch.outer
    outer = torch.outer(pos,inverse)
    # TODO: construct angles = torch.cat([freqs, freqs], dim=-1) so the shape
    #       becomes (T, d_head) and matches the split-half rotate_half convention.
    #       Do NOT use repeat_interleave here; it produces a different channel layout.
    #       Then return angles.cos(), angles.sin().
    angles = torch.cat([outer,outer],dim=-1)
    return angles.cos() , angles.sin()
    


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and return (-x2, x1) concatenated -- the 90-degree
    rotation step used by the "rotate_half" formulation of RoPE."""
    x1,x2 = torch.chunk(x,2,dim=-1)
    x2=-x2
    x = torch.cat([x2,x1],dim=-1)
    return x



def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, n_heads, T, d_head) -- a query or key tensor.
    cos, sin: (T, d_head), sliced from build_rope_cache's output for the current T.
    Returns the rotated tensor, same shape as x.
    Formula: x_rotated = x * cos + rotate_half(x) * sin
    """
    # TODO: broadcast cos/sin to (1, 1, T, d_head) and apply the formula above
    cos = torch.unsqueeze(cos,dim=0)
    cos = torch.unsqueeze(cos,dim=0)
    sin = torch.unsqueeze(sin,dim=0)
    sin = torch.unsqueeze(sin,dim=0)
    x_rotated = x*cos+rotate_half(x)*sin
    
    return x_rotated


# ======== attention3.py (your A3 code; cross-file import removed, FIX lines marked) ========
"""Attention supporting MHA/MQA/GQA (Q2) with a choice of positional scheme: RoPE applied
to q/k inside attention, or none (if positions are handled by sinusoidal PE at the input,
as in Q1)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, dropout=0.0, use_rope=False, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        # Preserve the Q1.4 hook. None -> standard 1/sqrt(d_head).
        self.attention_scale = (1.0 / math.sqrt(self.d_head)) if attention_scale is None else float(attention_scale)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            # TODO: apply_rope to q and k (NOT v) using self.rope_cos[:T], self.rope_sin[:T]
            q = apply_rope(q,self.rope_cos[:T],self.rope_sin[:T])
            k = apply_rope(k,self.rope_cos[:T],self.rope_sin[:T])
        

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # TODO: same attention as before, but compute scores as
        #       (q @ k.transpose(-2, -1)) * self.attention_scale before masking/softmax
        attention_scores = (q@k.transpose(-2,-1))*self.attention_scale
       
        attention_scores = attention_scores.masked_fill(
            ~self.causal_mask[:T,:T],float("-inf")
        )
       
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
    
        weighted_sum = attention_weights@v
   
        weighted_sum = torch.reshape(weighted_sum.transpose(1,2),(B,T,C))
        output = self.out_proj(weighted_sum)
        output = self.resid_dropout(output)
        
        return output


# ======== decoder3.py (your A3 code; cross-file import removed, FIX lines marked) ========
"""Decoder-only Transformer LM -- Block now passes use_rope through to attention, and
DecoderLM only builds/adds the sinusoidal table when pos_emb_type == "sinusoidal"
(everything else unchanged from Question 2)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg=cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        # TODO: pass use_rope=(cfg.pos_emb_type == "rope") to CausalSelfAttention
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
                                        cfg.dropout, use_rope=(cfg.pos_emb_type == "rope"),
                                        attention_scale=cfg.attention_scale)
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


class DecoderLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)
        # NOTE: when cfg.pos_emb_type == "rope", we deliberately do NOT build a
        # sinusoidal table at all -- position information is injected inside every
        # attention layer instead of once at the input.
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        # TODO: x = token embedding + positional encoding (if sinusoidal)
        x = self.tok_emb(idx)
        if(self.cfg.pos_emb_type == "sinusoidal"):
            x=x+self.pos_emb[:T]        # TODO: run x through every block, then ln_f, then self.head
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        
        return x  # unchanged from Question 1 (just remember: only add
        # self.pos_emb when self.cfg.pos_emb_type == "sinusoidal")
