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

