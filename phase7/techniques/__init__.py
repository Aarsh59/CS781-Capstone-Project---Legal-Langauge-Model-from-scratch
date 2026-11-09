"""Technique registry (plumbing). The config's technique.name selects techniques/<name>.py.

Contract for every technique file (YOUR code, course rule 9):
    apply(backbone, **cfg) -> None
        Called once, after the Phase-1 weights are loaded and before the optimizer is built. On entry EVERY backbone
        parameter has requires_grad=True. On exit, requires_grad must be True exactly on what should train
        (plus any modules you inserted). The head is separate and always trained.
        backbone: backbones.Backbone (.lm = your Phase-1 module, .blocks, .kind, .d_model). See inject.py for
        freeze / replace_linears / wrap_modules helpers and the linear-layer names.
    optional: param_groups(model, base_lr) -> list of optimizer param-group dicts (e.g. a different lr for your
        inserted modules). If absent, every trainable backbone parameter uses train.lr and the head uses
        train.head_lr (or train.lr).
"""
import importlib


def get(name):
    return importlib.import_module(f"techniques.{name}")
