"""YOUR FILE: Partial / layer-wise fine-tuning.

Contract: see techniques/__init__.py.  apply(backbone, **cfg) -> None
e.g. freeze(backbone); then unfreeze the parts you choose (backbone.blocks is input -> output order)
cfg = the config's "technique" dict minus "name" (your options, e.g. rank / alpha / targets / ...).
"""
import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears, wrap_modules   # noqa: F401  (helpers, optional)


def apply(backbone,k):
    freeze(backbone)
    if(k>0):
        
    # unfreeze the final norm layer
        if(backbone.kind == "encdec"):
            if backbone.encdec_mode == "encoder":
                unfreeze(backbone.lm.ln_f_enc)
            else:
                unfreeze(backbone.lm.ln_f_dec)


            
           
        else:
            unfreeze(backbone.lm.ln_f)
        for block in  backbone.blocks[-k:]:
            unfreeze(block)
        
        
    
    




   