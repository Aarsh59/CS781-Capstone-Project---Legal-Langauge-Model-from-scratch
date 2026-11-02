"""YOUR FILE: BitFit (Ben Zaken et al., 2022) [8].

Contract: see techniques/__init__.py.  apply(backbone, **cfg) -> None
decide which parameters stay trainable (note: attention projections are bias=False; FFN linears and LayerNorms have biases)
cfg = the config's "technique" dict minus "name" (your options, e.g. rank / alpha / targets / ...).
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401  (helpers, optional)


def apply(backbone, **cfg):
    freeze(backbone)
    for name , param in backbone.named_parameters():
        if name.endswith("bias") is True:
            param.requires_grad = True
        
