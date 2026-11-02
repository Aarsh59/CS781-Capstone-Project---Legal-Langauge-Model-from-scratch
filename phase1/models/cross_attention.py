"""Cross Attention supporting MHA/MQA/GQA (Q2) with a choice of positional scheme: RoPE applied
to q/k inside attention, or none (if positions are handled by sinusoidal PE at the input,
as in Q1)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache, apply_rope


class CrossAttention(nn.Module):
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

        

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x , memory):
        B, T, C = x.shape
        _,S,_ = memory.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(memory).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

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
       
       
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
    
        weighted_sum = attention_weights@v
   
        weighted_sum = torch.reshape(weighted_sum.transpose(1,2),(B,T,C))
        output = self.out_proj(weighted_sum)
        output = self.resid_dropout(output)
        
        return output