"""YOUR FILE: Houlsby / Pfeiffer bottleneck adapters [6, 7].

Contract: see techniques/__init__.py.  apply(backbone, **cfg) -> None
e.g. freeze(backbone); wrap_modules(backbone, <name endswith '.attn' / '.ffn'>, <your adapter wrapper>)
cfg = the config's "technique" dict minus "name" (your options, e.g. rank / alpha / targets / ...).
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401  (helpers, optional)

class Adapter(nn.Module):
    def __init__(self,d_model,bottleneck):
        super().__init__()
        self.d_model = d_model
        self.bottleneck=bottleneck
        self.down = nn.Linear(d_model,bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck,d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self,x:torch.Tensor):
        return x+self.up(self.act(self.down(x)))
class AdapterWrapper(nn.Module):
    def __init__(self,sub,d_model,bottleneck):
        super().__init__()
        self.sub=sub
        self.d_model=d_model
        self.bottleneck=bottleneck
        self.adapter =Adapter(d_model,bottleneck)
    def forward(self,*args,**kwargs):
        y= self.sub(*args,**kwargs)
        return self.adapter(y)

def apply(backbone, bottleneck):
    freeze(backbone)
    replaced = wrap_modules(
        backbone,
        lambda name, m: name.endswith((".ffn",".attn")),            # your placement choice (see below)
        lambda name, m: AdapterWrapper(m, backbone.d_model, bottleneck),
    )
