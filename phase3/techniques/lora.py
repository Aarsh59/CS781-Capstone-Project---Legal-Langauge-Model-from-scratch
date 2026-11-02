"""YOUR FILE: LoRA (Hu et al., 2021) [3].

Contract: see techniques/__init__.py.  apply(backbone, **cfg) -> None
e.g. freeze(backbone); replace_linears(backbone, <which projections>, <your LoRALinear around nn.Linear>)
cfg = the config's "technique" dict minus "name" (your options, e.g. rank / alpha / targets / ...).
"""
import torch
import torch.nn as nn
import math
from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401  (helpers, optional)

class LoRALinear(nn.Module):
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
        self.scaling  = alpha/r
    def forward (self,x):
        base_output = self.base_layer(x)
        scaled_output = (x@self.A.T)@self.B.T
        return base_output+self.scaling*scaled_output
        



def apply(backbone, r, alpha, targets):
    freeze(backbone)
    replaced = replace_linears(backbone,
                               lambda name, lin: any(target in name for target in targets),
                               lambda name, lin: LoRALinear(lin, r, alpha))

