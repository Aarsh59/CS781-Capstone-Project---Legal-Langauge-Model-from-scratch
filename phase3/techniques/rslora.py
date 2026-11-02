"""YOUR FILE: Rank-Stabilized LoRA (rsLoRA) — Kalajdzievski (2023) [11].

WHAT IT IS
  Identical to LoRA (h = W0 x + scale * B(A(x)), A: r x in, B: out x r, B zero-initialised, W0 frozen) except for the
  scaling factor:
      LoRA:   scale = alpha / r
      rsLoRA: scale = alpha / sqrt(r)
  With alpha / r the update shrinks as the rank grows, so high ranks learn slowly; alpha / sqrt(r) keeps the update
  magnitude stable across ranks (that is the paper's point). Everything else (targets, init, freezing) as your LoRA.

WHAT THE PLUMBING CALLS
  apply(backbone, **cfg) -> None
      Called once after the Phase-1 weights are loaded. On entry every backbone parameter has requires_grad=True;
      on exit only your inserted A/B matrices should be trainable (the head is handled by the plumbing).
      cfg = the config's "technique" dict minus "name", e.g. {"name": "rslora", "r": ..., "alpha": ..., "targets": [...]}
      -> write it as  def apply(backbone, r, alpha, targets):  so a typo in the config is caught before training.

SUGGESTED STRUCTURE (mirrors your lora.py; you can import your LoRALinear and only change the scale)
  class RsLoRALinear(nn.Module):
      __init__(self, base: nn.Linear, r, alpha): keep `base` frozen, create A (r, in_features) and B (out_features, r),
          init A like your LoRA (kaiming_uniform_, a=sqrt(5)) and B = 0, scaling = alpha / math.sqrt(r)
      forward(self, x): base(x) + scaling * B(A(x))
  def apply(backbone, r, alpha, targets):
      freeze(backbone); replace_linears(backbone, lambda name, lin: <name matches targets>,
                                        lambda name, lin: RsLoRALinear(lin, r, alpha))

HELPERS: inject.freeze / replace_linears (see inject.py for the linear-layer names of A/B/C/D).
"""
import math

import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401

class RSLoRALinear(nn.Module):
    def __init__(self,base:nn.Linear,r,alpha):
        super().__init__()
        self.base_layer=base
        self.in_features = base.in_features
        self.out_features = base.out_features
        for param in self.base_layer.parameters():
            param.requires_grad = False
        # define A and B as dimension B is out features cross r and A is r cross in features
        self.A = nn.Parameter(torch.randn(r, self.in_features))
        
        self.A = nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.out_features,r))
        self.scaling  = alpha/math.sqrt(r)
    def forward (self,x):
        base_output = self.base_layer(x)
        scaled_output = (x@self.A.T)@self.B.T
        return base_output+self.scaling*scaled_output
        



def apply(backbone, r, alpha, targets):
    freeze(backbone)
    replaced = replace_linears(backbone,
                               lambda name, lin: any(target in name for target in targets),
                               lambda name, lin: RSLoRALinear(lin, r, alpha))




