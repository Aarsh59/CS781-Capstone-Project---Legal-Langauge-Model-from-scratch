"""Attention supporting MHA / MQA / GQA via n_kv_heads."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # how many query heads share each kv head
        self.d_head = d_model // n_heads
        # Preserve the Q1.4 hook in later questions. None -> standard 1/sqrt(d_head).
        self.attention_scale = (1.0 / math.sqrt(self.d_head)) if attention_scale is None else float(attention_scale)

        # TODO: q_proj maps d_model -> n_heads * d_head (as before).
        # TODO: k_proj, v_proj map d_model -> n_kv_heads * d_head (SMALLER than before
        #       whenever n_kv_heads < n_heads -- this is where the parameter savings
        #       and (in a real serving system) the KV-cache savings come from).
        self.q_proj = nn.Linear(d_model,n_heads*d_head,bias=False)
        self.k_proj = nn.Linear(d_model,n_kv_heads*d_head,bias=False)
        self.v_proj = nn.Linear(d_model,n_kv_heads*d_head,bias=False)
        self.out_proj = nn.Linear(d_model,d_model,bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # TODO 1: project q -> (B, n_heads, T, d_head); k, v -> (B, n_kv_heads, T, d_head)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = torch.reshape(q,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        k = torch.reshape(k,(B,T,self.n_kv_heads,self.d_head)).transpose(1,2)
        v = torch.reshape(v,(B,T,self.n_kv_heads,self.d_head)).transpose(1,2)
        # TODO 2: expand k and v from n_kv_heads to n_heads by repeating each kv head
        #         `self.n_rep` times (torch.repeat_interleave along the head dim is the
        #         natural tool here) so shapes line up for a standard batched matmul.

        k = k.repeat_interleave(self.n_rep,dim=1)
        v = v.repeat_interleave(self.n_rep,dim=1)

        # TODO 3: same attention as Q1, using (q @ k^T) * self.attention_scale,
        #         then causal mask + softmax + weighted sum.
        # TODO 4: reshape back to (B, T, C), apply out_proj.



        attention_scores = (q@k.transpose(2,3))*self.attention_scale
       
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
        
     
      


class MultiHeadLatentAttention(nn.Module):
    """Q2-only joint KV compression; use the equations in Q2.1."""
    def __init__(self, d_model, n_heads, kv_latent_dim, max_seq_len,
                 dropout=0.0, attention_scale=None):
        super().__init__()
        # TODO: validate d_model % n_heads == 0 and 0 < kv_latent_dim < d_model.
        assert d_model%n_heads==0
        assert kv_latent_dim>0 and kv_latent_dim<d_model
        # Expose n_heads, d_head, kv_latent_dim, and attention_scale.
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.attention_scale = (1.0 / math.sqrt(self.d_head)) if attention_scale is None else float(attention_scale)
        
        # None means 1/sqrt(d_head); preserve explicit attention_scale overrides.
        # All five projections must be bias-free nn.Linear modules with these names:
        self.q_proj = nn.Linear(d_model,d_model,bias=False)        # d_model -> d_model
        self.kv_down_proj = nn.Linear(d_model,kv_latent_dim,bias=False)  # d_model -> kv_latent_dim (shared by K and V)
        self.k_up_proj = nn.Linear(kv_latent_dim,d_model,bias=False)     # kv_latent_dim -> d_model
        self.v_up_proj = nn.Linear(kv_latent_dim,d_model,bias=False)     # kv_latent_dim -> d_model
        self.out_proj = nn.Linear(d_model,d_model,bias=False)      # d_model -> d_model
        # TODO: attention/residual dropout and a nonpersistent causal_mask buffer.
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        

    def forward(self, x):
        # TODO: compress once, reconstruct per-head K/V from the same latent,
        # compute causal scaled attention, concatenate heads, and project output.
        # Input/output shape: (B, T, d_model). Do not add positions in this module.
        B,T,d_model = x.shape
        q = self.q_proj(x)
        latent = self.kv_down_proj(x)
        k = self.k_up_proj(latent)
        v = self.v_up_proj(latent)
        q = torch.reshape(q,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        k = torch.reshape(k,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        v = torch.reshape(v,(B,T,self.n_heads,self.d_head)).transpose(1,2)
        attention_scores = (q@k.transpose(2,3))*self.attention_scale
       
        attention_scores = attention_scores.masked_fill(
            ~self.causal_mask[:T,:T],float("-inf")
        )
       
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
    
        weighted_sum = attention_weights@v
   
        weighted_sum = torch.reshape(weighted_sum.transpose(1,2),(B,T,d_model))
        output = self.out_proj(weighted_sum)
        output = self.resid_dropout(output)
        
        return output
        
        
        
       