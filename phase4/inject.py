"""Generic module-surgery helpers for your fine-tuning techniques (plumbing; declare it).

These only find / replace / freeze modules; they contain no adaptation method. Typical use in your files:

    from inject import freeze, replace_linears, wrap_modules
    freeze(backbone)                                            # base weights frozen
    replace_linears(backbone, lambda name, lin: name.endswith(("q_proj", "v_proj")),
                    lambda name, lin: YourWrapper(lin, ...))    # your LoRA / LoRMA module around nn.Linear
    wrap_modules(backbone, lambda name, m: name.endswith(".ffn"),
                 lambda name, m: YourAdapterWrapper(m, ...))    # your adapter after a sublayer

Linear layer names in the Phase-1 models (relative to backbone, e.g. "lm.blocks.3.attn.q_proj"):
  A/B/D blocks:  attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj, ffn.net.0 (d->d_ff), ffn.net.2 (d_ff->d)
  C encoder:     lm.encblocks.<i>.attn.{q,k,v,out}_proj, lm.encblocks.<i>.ffn.net.{0,2}
  C decoder:     lm.decblocks.<i>.attn.*, lm.decblocks.<i>.crattn.{q,k,v,out}_proj, lm.decblocks.<i>.ffn.net.{0,2}
Attention projections have no bias (bias=False); FFN linears and LayerNorms do.
Sublayer call signatures: attn(x) (D: attn(x, p)), crattn(x, memory), ffn(x) -> a wrapper should take
(*args, **kwargs) and pass them through to the wrapped module.
"""
import torch.nn as nn


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)
    return module


def _set_child(root, name, new):
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def list_linears(root):
    return [name for name, m in root.named_modules() if isinstance(m, nn.Linear)]


def replace_linears(root, match, factory):
    """Replace every nn.Linear for which match(name, linear) is True by factory(name, linear). -> replaced names."""
    targets = [(n, m) for n, m in root.named_modules() if isinstance(m, nn.Linear) and match(n, m)]
    for name, lin in targets:
        _set_child(root, name, factory(name, lin))
    return [n for n, _ in targets]


def wrap_modules(root, match, factory):
    """Replace every submodule for which match(name, module) is True by factory(name, module). -> replaced names.
    Matches are collected first, so a wrapper containing the original module is not matched again."""
    targets = [(n, m) for n, m in root.named_modules() if n and match(n, m)]
    for name, mod in targets:
        _set_child(root, name, factory(name, mod))
    return [n for n, _ in targets]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / max(1, total), 4)}
