"""YOUR FILE: AdaLoRA — Zhang et al. (2023) [10].

WHAT IT IS
  LoRA with an ADAPTIVE rank per weight matrix. Each adapted linear gets an SVD-shaped update
      h = W0 x + scale * P diag(lam) Q x          P: (out, r), lam: (r,), Q: (r, in); lam starts at 0
  Training starts with an initial rank r_init everywhere and, on a schedule, prunes singular values (sets entries of
  lam to 0 / masks them) in the LESS important matrices until a total budget (target average rank) is reached, so
  important layers keep more rank. Importance of each triplet (P[:, i], lam[i], Q[i, :]) is computed from
  |param * grad| (sensitivity), smoothed with exponential moving averages (EMA of sensitivity and of its uncertainty).
  An orthogonality regulariser keeps P and Q close to orthogonal:  R = ||P^T P - I||_F^2 + ||Q Q^T - I||_F^2.

PLUMBING HOOKS (train_rr.py) — define what you need:
  apply(backbone, **cfg) -> None
      insert your AdaLoRALinear modules (freeze(backbone) + replace_linears), e.g. config
      {"name": "adalora", "r_init": ..., "target_rank": ..., "alpha": ..., "targets": [...], "t_init": ..., "t_final": ..., "orth_reg": ...}
  extra_loss(model) -> scalar tensor                                   (optional; called every micro-batch)
      return orth_reg * sum of the orthogonality terms over all AdaLoRA modules; it is ADDED to the task loss.
  on_step(model, step, total_steps) -> None                            (optional; called after every optimizer step)
      Called AFTER optimizer.step() and BEFORE the gradients are zeroed, so p.grad is still available here.
      (If train.grad_clip is None the gradients are still multiplied by the fp16 GradScaler factor; that factor is
      the same for every parameter, so importance RANKINGS from |p * p.grad| are unaffected.)
      1. update the importance EMAs from |lam * lam.grad| (and the P, Q terms if you use them);
      2. following the cubic budget schedule between t_init and total_steps - t_final, compute the current budget
         and mask the lowest-importance singular values globally across modules (lam[i] = 0 and keep them 0).
  Find the AdaLoRA modules from `model` with e.g. [m for m in model.modules() if isinstance(m, AdaLoRALinear)].
  Keep the masks as buffers so they are saved with the checkpoint.
"""
import math

import torch
import torch.nn as nn

from inject import freeze, unfreeze, replace_linears   # noqa: F401

class AdaLoRALinear(nn.Module):
    def __init__(self,base,r,scale,target_rank, t_init, t_final, delta_t, beta1, beta2, orth_reg):
        super().__init__()
        self.base_layer = base 
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.alpha=scale/r
        self.r = r 
        self.P = nn.Parameter(torch.empty(self.out_features,self.r))
        self.Q = nn.Parameter(torch.empty(self.r,self.in_features))
        self.P=nn.init.xavier_normal_(self.P, gain=0.1)
        self.Q=nn.init.xavier_normal_(self.Q, gain=0.1)
        self.lam = nn.Parameter(torch.zeros(self.r))
        mask = torch.ones(self.r)
        ipt_ema = torch.zeros(self.r)
        unc_ema = torch.zeros(self.r)
        self.register_buffer('mask',mask)
        self.register_buffer('ipt_ema',ipt_ema)
        self.register_buffer('unc_ema',unc_ema)
        self.target_rank=target_rank
        self.t_init = t_init
        self.t_final=t_final
        self.delta_t=delta_t
        self.beta1 = beta1
        self.beta2=beta2
        self.orth_reg = orth_reg
    def forward(self,x:torch.Tensor):
        frozen = self.base_layer(x)
        proj_down = x@self.Q.T
        proj_down = proj_down*(self.lam*self.mask)
        proj_up = proj_down@self.P.T
        return frozen+self.alpha*proj_up

def extra_loss(model):
    mods = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]

    if not mods:
        return next(model.parameters()).new_zeros(())

    loss = 0.0

    for m in mods:
        P = m.P
        Q = m.Q
        I = torch.eye(m.r, device=P.device, dtype=P.dtype)

        loss = loss + torch.norm(P.T @ P - I, p="fro") ** 2
        loss = loss + torch.norm(Q @ Q.T - I, p="fro") ** 2

    return mods[0].orth_reg * loss
def on_step(model, step, total_steps):
    mods = [
        m for m in model.modules()
        if isinstance(m, AdaLoRALinear)
    ]

    if not mods:
        return

    params = [
        p for m in mods for p in (m.P, m.Q, m.lam)
    ]

    # Skip the update if any required gradient is missing
    # or non-finite.
    if any(
        p.grad is None or not torch.isfinite(p.grad).all()
        for p in params
    ):
        return

    with torch.no_grad():
        # Update importance EMAs EVERY step.
        for m in mods:
            p_score = (m.P * m.P.grad).abs().mean(dim=0)
            q_score = (m.Q * m.Q.grad).abs().mean(dim=1)
            lam_score = (m.lam * m.lam.grad).abs()

            sensitivity = p_score + q_score + lam_score

            old_ipt = m.ipt_ema.clone()

            m.ipt_ema.mul_(m.beta1).add_(
                sensitivity, alpha=1 - m.beta1
            )

            uncertainty = (sensitivity - old_ipt).abs()

            m.unc_ema.mul_(m.beta2).add_(
                uncertainty, alpha=1 - m.beta2
            )

        # Only update the rank budget and masks every delta_t steps.
        if step % mods[0].delta_t != 0:
            return

        # Keep your existing cubic budget schedule and
        # global pruning code here.





def apply(backbone,r,alpha,targets,target_rank,t_init,t_final,delta_t,beta1,beta2,orth_reg):
    freeze(backbone)
    replaced = replace_linears(backbone,
                               lambda name, lin: any(target in name for target in targets),
                               lambda name, lin: AdaLoRALinear(lin, r, alpha,target_rank,t_init,t_final,delta_t,beta1,beta2,orth_reg))
    


# Optional hooks (delete them if you do not use them; the plumbing only calls what exists):
# def extra_loss(model): ...
# def on_step(model, step, total_steps): ...
