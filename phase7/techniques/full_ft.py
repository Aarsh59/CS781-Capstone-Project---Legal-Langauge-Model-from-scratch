"""Full fine-tuning: every backbone parameter trains (plumbing-level: nothing to implement)."""


def apply(backbone, **cfg):
    assert not cfg, f"full_ft takes no options, got {cfg}"
    for p in backbone.parameters():
        p.requires_grad_(True)
