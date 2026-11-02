
"""
AdaLoRA — Zhang et al. (2023)

SVD-shaped low-rank adaptation:
    W_delta = alpha * P @ diag(lam * mask) @ Q

Shapes:
    P: (out_features, r)
    lam: (r,)
    Q: (r, in_features)

Features:
    - Frozen pretrained base weights
    - Adaptive global rank pruning
    - EMA-based importance and uncertainty
    - Orthogonality regularization
    - Persistent pruning masks stored as buffers
"""

import torch
import torch.nn as nn

from inject import freeze, replace_linears


class AdaLoRALinear(nn.Module):
    def __init__(
        self,
        base,
        r,
        scale,
        target_rank,
        t_init,
        t_final,
        delta_t,
        beta1,
        beta2,
        orth_reg,
    ):
        super().__init__()

        self.base_layer = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        if r <= 0:
            raise ValueError("r must be positive")
        if not 0 <= target_rank <= r:
            raise ValueError("target_rank must be between 0 and r")
        if delta_t <= 0:
            raise ValueError("delta_t must be positive")

        self.r = r
        self.alpha = scale / r

        self.P = nn.Parameter(
            torch.empty(self.out_features, self.r)
        )
        self.Q = nn.Parameter(
            torch.empty(self.r, self.in_features)
        )
        nn.init.xavier_normal_(self.P, gain=0.1)
        nn.init.xavier_normal_(self.Q, gain=0.1)

        # Zero initialization means the adapter initially contributes zero.
        self.lam = nn.Parameter(torch.zeros(self.r))

        self.register_buffer(
            "mask", torch.ones(self.r)
        )
        self.register_buffer(
            "ipt_ema", torch.zeros(self.r)
        )
        self.register_buffer(
            "unc_ema", torch.zeros(self.r)
        )

        self.target_rank = target_rank
        self.t_init = t_init
        self.t_final = t_final
        self.delta_t = delta_t
        self.beta1 = beta1
        self.beta2 = beta2
        self.orth_reg = orth_reg

    def forward(self, x):
        base_output = self.base_layer(x)

        # (..., in_features) -> (..., r)
        z = x @ self.Q.T

        # Masked singular values stay inactive.
        z = z * (self.lam * self.mask)

        # (..., r) -> (..., out_features)
        update = z @ self.P.T

        return base_output + self.alpha * update


def _get_modules(model):
    return [
        m for m in model.modules()
        if isinstance(m, AdaLoRALinear)
    ]


def extra_loss(model):
    """Orthogonality regularizer added to the task loss."""
    mods = _get_modules(model)

    if not mods:
        return next(model.parameters()).new_zeros(())

    loss = mods[0].P.new_zeros(())

    for m in mods:
        # Only regularize active components. This avoids penalizing
        # the geometry of components that have already been pruned.
        active = m.mask.bool()
        idx = torch.nonzero(active, as_tuple=True)[0]

        if idx.numel() == 0:
            continue

        P_active = m.P[:, idx]
        Q_active = m.Q[idx, :]

        I = torch.eye(
            idx.numel(),
            device=m.P.device,
            dtype=m.P.dtype,
        )

        loss = loss + (
            torch.norm(P_active.T @ P_active - I, p="fro") ** 2
        )
        loss = loss + (
            torch.norm(Q_active @ Q_active.T - I, p="fro") ** 2
        )

    return mods[0].orth_reg * loss


def on_step(model, step, total_steps):
    """
    Update importance EMAs and apply global rank pruning.

    Called after optimizer.step(), while gradients are still available.
    Assumes `step` uses the same convention as the training loop and
    `total_steps` is the total number of optimizer steps.
    """
    mods = _get_modules(model)

    if not mods:
        return

    # All modules should share one rank schedule.
    first = mods[0]
    r = first.r
    target_rank = first.target_rank
    t_init = first.t_init
    t_final = first.t_final
    delta_t = first.delta_t

    if any(
        m.r != r
        or m.target_rank != target_rank
        or m.t_init != t_init
        or m.t_final != t_final
        or m.delta_t != delta_t
        for m in mods
    ):
        raise ValueError(
            "All AdaLoRA modules must share rank schedule settings"
        )

    if total_steps <= t_init + t_final:
        raise ValueError(
            "total_steps must exceed t_init + t_final"
        )

    # Update importance only when all relevant gradients are available
    # and finite. If not, skip this optimizer step's AdaLoRA update.
    params = [
        p for m in mods for p in (m.P, m.Q, m.lam)
    ]

    if any(
        p.grad is None or not torch.isfinite(p.grad).all()
        for p in params
    ):
        return

    with torch.no_grad():
        # 1. Update importance and uncertainty EMAs.
        for m in mods:
            p_score = (m.P * m.P.grad).abs().mean(dim=0)
            q_score = (m.Q * m.Q.grad).abs().mean(dim=1)
            lam_score = (m.lam * m.lam.grad).abs()

            sensitivity = p_score + q_score + lam_score

            old_ipt = m.ipt_ema.clone()

            m.ipt_ema.mul_(m.beta1).add_(
                sensitivity, alpha=1.0 - m.beta1
            )

            uncertainty = (sensitivity - old_ipt).abs()

            m.unc_ema.mul_(m.beta2).add_(
                uncertainty, alpha=1.0 - m.beta2
            )

        # 2. Warm-up: keep the initial rank.
        if step < t_init:
            return

        # 3. Only prune at delta_t intervals.
        if step % delta_t != 0:
            return

        n = len(mods)
        initial_budget = r * n
        target_budget = target_rank * n

        # 4. Cubic rank-budget schedule.
        if step >= total_steps - t_final:
            budget = target_budget
        else:
            progress = (
                (step - t_init)
                / (total_steps - t_init - t_final)
            )
            progress = max(0.0, min(1.0, progress))

            scheduled_budget = (
                target_budget
                + (initial_budget - target_budget)
                * (1.0 - progress) ** 3
            )
            budget = int(scheduled_budget)

        budget = max(0, min(budget, initial_budget))

        # 5. Gather global importance scores.
        score_list = []

        for m in mods:
            score = m.ipt_ema * m.unc_ema
            score = score.masked_fill(
                m.mask == 0, float("-inf")
            )
            score_list.append(score)

        all_scores = torch.cat(score_list)

        active = int(
            sum(m.mask.sum().item() for m in mods)
        )

        # Never reactivate pruned components.
        budget = min(budget, active)

        if active <= budget:
            return

        # 6. Keep the globally highest-importance components.
        keep_idx = torch.topk(
            all_scores, k=budget
        ).indices

        keep = torch.zeros_like(
            all_scores, dtype=torch.bool
        )
        keep[keep_idx] = True

        # 7. Apply each module's portion of the global mask.
        offset = 0

        for m in mods:
            keep_slice = keep[offset:offset + m.r]

            m.mask.mul_(
                keep_slice.to(dtype=m.mask.dtype)
            )

            # Ensure pruned singular values are exactly zero.
            m.lam.mul_(m.mask)

            offset += m.r

        # Temporary diagnostic.
        if step % 200 == 0:
            active_after = int(
                sum(m.mask.sum().item() for m in mods)
            )
            print(
                f"[AdaLoRA] step={step}, "
                f"budget={budget}, active={active_after}"
            )


def apply(
    backbone,
    r,
    alpha,
    targets,
    target_rank,
    t_init,
    t_final,
    delta_t,
    beta1,
    beta2,
    orth_reg,
):
    """Inject AdaLoRA into matching linear layers."""
    freeze(backbone)

    def match(name, lin):
        return any(target in name for target in targets)

    def factory(name, lin):
        return AdaLoRALinear(
            base=lin,
            r=r,
            scale=alpha,
            target_rank=target_rank,
            t_init=t_init,
            t_final=t_final,
            delta_t=delta_t,
            beta1=beta1,
            beta2=beta2,
            orth_reg=orth_reg,
        )

    replace_linears(backbone, match, factory)

    return backbone
