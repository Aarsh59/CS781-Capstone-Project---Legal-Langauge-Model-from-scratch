"""YOUR FILE: Prefix Tuning — Li & Liang (2021) [4].

WHAT IT IS
  The backbone is frozen. Every self-attention layer gets P learnable key vectors and P learnable value vectors
  (a "prefix") that all queries can attend to, in addition to the real tokens. Unlike prompt tuning (input layer
  only), this steers every layer. The paper reparameterises the prefix through a small MLP during training for
  stability (prefix = MLP(embedding)), which you may or may not do.

PLUMBING HOOK ALREADY IN PLACE (phase1_models/fastattn_ab.py and fastattn_cd.py)
  If an attention module has an attribute `prefix_kv`, the fused attention calls
        pk, pv = attn.prefix_kv(batch_size)        # each (B, n_kv_heads, P, d_head)
  prepends them to that layer's keys/values (BEFORE the GQA head repetition; no RoPE is applied to them) and extends
  the attention mask so every query sees all P prefix positions (causal / bidirectional / prefix-LM rules unchanged
  for the real tokens). Without `prefix_kv` attention is exactly as before. Supported: self-attention of A/B
  (fastattn_ab), C encoder/decoder self-attention and D prefix-LM attention (fastattn_cd); not cross-attention.
  Shapes: A has n_kv_heads = n_heads = 12; B/C/D have n_kv_heads = 4; d_head = 64 everywhere.

WHAT THE PLUMBING CALLS
  apply(backbone, **cfg) -> None, e.g. config {"name": "prefix_tuning", "n_prefix": ..., "reparam_hidden": ...}

SUGGESTED STRUCTURE
  class LayerPrefix(nn.Module):
      __init__(self, n_kv_heads, d_head, P, ...): self.k = nn.Parameter((n_kv_heads, P, d_head)), self.v = same
          (or an embedding + MLP producing them, for the reparameterised version)
      forward / __call__(self, B): return (self.k.expand(B, ...), self.v.expand(B, ...))
  def apply(backbone, n_prefix, ...):
      freeze(backbone)
      for block in backbone.blocks:                       # input -> output order; C "decoder" mode: enc + dec blocks
          attn = block.attn
          prefix = LayerPrefix(attn.n_kv_heads if hasattr(attn, "n_kv_heads") else attn.n_heads, attn.d_head, n_prefix)
          attn.prefix_module = prefix                     # register as a submodule so it is trained and saved
          attn.prefix_kv = prefix                         # the hook the fused attention calls
  IMPORTANT: assign the module as a submodule (setattr on the attention module) so its parameters are part of
  model.parameters() / state_dict; `prefix_kv` must be callable with the batch size.
"""



import torch
import torch.nn as nn

from inject import freeze, unfreeze   # noqa: F401


class LayerPrefix(nn.Module):
  def __init__(self,n_kv_heads,d_head,P):
    super().__init__()
    self.n_kv_heads = n_kv_heads
    self.d_head = d_head
    self.P=P
    self.k = nn.Parameter(torch.empty(n_kv_heads,P,d_head))
    self.v = nn.Parameter(torch.empty(n_kv_heads,P,d_head))
    nn.init.normal_(self.k,mean=0,std=0.02)
    nn.init.normal_(self.v,mean=0,std=0.02)
  def forward(self,B):
    return (self.k.unsqueeze(0).expand(B,-1,-1,-1),self.v.unsqueeze(0).expand(B,-1,-1,-1))







def apply(backbone,n_prefix):
  freeze(backbone)
  for block in backbone.blocks:
    attn=block.attn
    prefix = LayerPrefix(attn.n_kv_heads if hasattr(attn, "n_kv_heads") else attn.n_heads, attn.d_head, n_prefix)
    attn.prefix_module = prefix                     # register as a submodule so it is trained and saved
    attn.prefix_kv = prefix                         # the hook the fused attention calls

    
