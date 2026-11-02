"""YOUR FILE: Prompt Tuning — Lester, Al-Rfou & Constant (2021) [5].

WHAT IT IS
  The whole backbone is frozen. P learnable "soft prompt" vectors (each of size d_model) are prepended to the
  input embeddings of every window; only those P x d_model numbers (plus the head) train.

PLUMBING HOOK ALREADY IN PLACE (backbones.Backbone)
  * backbone.n_virtual_tokens: set it to P in apply(). From then on the backbone itself prepends P placeholder ids
    (<pad>, id 0) to every window and removes the first P output positions again, so pooling / heads are unchanged.
    The plumbing checks window_len + P <= max_seq_len.
  * backbone.get_input_embedding() / backbone.set_input_embedding(module): the token embedding the prompt goes into
    (A/B/D: lm.tok_emb; C: the encoder's lm.tok_emb1).
  Your job: make positions [:, :P] of the embedding output learnable.

WHAT THE PLUMBING CALLS
  apply(backbone, **cfg) -> None, e.g. config {"name": "prompt_tuning", "n_tokens": ..., "init": ...}
      -> def apply(backbone, n_tokens, init=...):

SUGGESTED STRUCTURE
  class SoftPromptEmbedding(nn.Module):
      __init__(self, base_emb: nn.Embedding, n_tokens, init):
          keep base_emb (frozen); self.prompt = nn.Parameter(tensor of shape (n_tokens, d_model))
          init options from the paper: random (small normal) or copied from the embeddings of real vocabulary tokens
      forward(self, ids):  e = base_emb(ids)                     # (B, P + T, d), first P ids are placeholders
                           return torch.cat([self.prompt.expand(B, -1, -1).to(e.dtype), e[:, P:]], dim=1)
  def apply(backbone, n_tokens, init):
      freeze(backbone)
      backbone.set_input_embedding(SoftPromptEmbedding(backbone.get_input_embedding(), n_tokens, init))
      backbone.n_virtual_tokens = n_tokens
  Note: A/B/D tie the LM head to tok_emb; the head is not used in Phase 3 (replaced by Identity), so wrapping is safe.
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze   # noqa: F401

class SoftPromptEmbedding(nn.Module):
  def __init__(self,base_emb:nn.Embedding,n_tokens,init):
      super().__init__()
      self.base_emb = base_emb
      self.d_model = base_emb.embedding_dim
      self.n_tokens = n_tokens
     
      self.prompt = nn.Parameter(torch.empty(self.n_tokens, self.d_model))

      if init == "random":
            nn.init.normal_(self.prompt, mean=0.0, std=0.02)

      else:
            raise ValueError(f"Unknown prompt initialization: {init}")



  def forward(self,ids):
    e = self.base_emb(ids)
    B,_,_ = e.shape
    return torch.cat([self.prompt.expand(B, -1, -1).to(e.dtype), e[:, self.n_tokens:]], dim=1)
def apply(backbone,n_tokens,init):
  freeze(backbone)
  backbone.set_input_embedding(SoftPromptEmbedding(backbone.get_input_embedding(), n_tokens, init))
  backbone.n_virtual_tokens = n_tokens


    
