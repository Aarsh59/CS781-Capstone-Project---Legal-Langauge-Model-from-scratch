"""YOUR FILE: Houlsby / Pfeiffer bottleneck adapters [6, 7].

Contract: see techniques/__init__.py.  apply(backbone, **cfg) -> None
e.g. freeze(backbone); wrap_modules(backbone, <name endswith '.attn' / '.ffn'>, <your adapter wrapper>)
cfg = the config's "technique" dict minus "name" (your options, e.g. rank / alpha / targets / ...).
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401  (helpers, optional)


def apply(backbone, **cfg):
    raise NotImplementedError("techniques/adapters.py: implement apply(backbone, **cfg)")
