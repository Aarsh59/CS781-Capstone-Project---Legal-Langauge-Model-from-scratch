"""YOUR FILE: IA³ — "Infused Adapter by Inhibiting and Amplifying Inner Activations", Liu et al. (2022) [13].

WHAT IT IS
  The backbone is frozen; the only new parameters are three learned VECTORS per block that rescale activations
  element-wise (no matrices, so very few parameters):
      keys:    k' = l_k ⊙ k        l_k ∈ R^{dim of k_proj output}
      values:  v' = l_v ⊙ v        l_v ∈ R^{dim of v_proj output}
      FFN:     h' = l_ff ⊙ act(W1 x)   applied to the FFN's inner activation, i.e. to the INPUT of the second FFN
                                        linear (ffn.net.2), l_ff ∈ R^{d_ff}
  All vectors are initialised to ONES, so at the start the model is exactly the pretrained one.
  (Note: in your models k_proj / v_proj output n_kv_heads * d_head — 256 for B/C/D, 768 for A — use out_features.)

WHAT THE PLUMBING CALLS
  apply(backbone, **cfg) -> None      (same contract as your other techniques)
      e.g. config {"name": "ia3"} (no options), or your own options such as which blocks / which of k, v, ffn.
      On exit only the l_k / l_v / l_ff vectors (plus nothing else) should have requires_grad=True.

SUGGESTED STRUCTURE
  class ScaleOutput(nn.Module):  wraps an nn.Linear; forward(x) = base(x) * self.scale   (scale = ones(out_features))
      -> use for attn.k_proj and attn.v_proj via replace_linears
  class ScaleInput(nn.Module):   wraps an nn.Linear; forward(x) = base(x * self.scale)   (scale = ones(in_features))
      -> use for ffn.net.2 (its input IS act(W1 x)), via replace_linears
  def apply(backbone, **cfg):
      freeze(backbone); replace_linears(... k_proj / v_proj -> ScaleOutput ...); replace_linears(... "ffn.net.2" -> ScaleInput ...)
  Parameter count check: per block len(l_k) + len(l_v) + d_ff (B: 256 + 256 + 3456 = 3,968 per block).
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401

class ScaledOutput(nn.Module):
    def __init__(self,base:nn.Linear):
        super().__init__()
        self.base = base
        self.out_features = base.out_features
        self.in_features = base.in_features
        self.scale = nn.Parameter(torch.ones(self.out_features))
    def forward(self,x:torch.Tensor):
        x = self.base(x)
        return x*self.scale

class ScaledInput(nn.Module):
    def __init__(self,base:nn.Linear):
        super().__init__()
        self.base = base
        self.out_features = base.out_features
        self.in_features = base.in_features
        self.scale = nn.Parameter(torch.ones(self.in_features))
    def forward(self,x:torch.Tensor):
        x = self.base(self.scale*x)
        return x



def apply(backbone):
    freeze(backbone)

    replace_linears(
        backbone,
        lambda name, lin: name.endswith(("k_proj", "v_proj", "ffn.net.2")),
        lambda name, lin: (
            ScaledOutput(lin)
            if name.endswith(("k_proj", "v_proj"))
            else ScaledInput(lin)
        )
    )

