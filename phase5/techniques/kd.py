"""YOUR FILE: knowledge-distillation loss (Hinton et al., 2015) [2] -- COMPULSORY experiment.

Used when the config has a "kd" block: {"teacher_run": <run dir of a finished teacher>, ...your options}.
The plumbing loads the teacher (frozen, eval mode), runs it on exactly the same windows as the student and calls

kd_loss(student_logits, teacher_logits, tags, **cfg) -> scalar
    student_logits, teacher_logits: (B, T, n_tags) float32 (same tag set: checked)
    tags: (B, T) long gold tags, -100 = ignore
    cfg: the "kd" dict minus "teacher_run" (e.g. your temperature / mixing weight)
This loss REPLACES the normal task loss for the student, so include the gold-label term yourself if you want it.
The student's own adaptation is chosen as usual with "technique" (e.g. full_ft on a smaller backbone).
"""

"""YOUR FILE: knowledge-distillation loss (Hinton et al., 2015) [2]."""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, tags, **cfg):
    """
    Knowledge-distillation loss for token classification / NER.

    Args:
        student_logits: (B, T, C) student logits.
        teacher_logits: (B, T, C) teacher logits.
        tags: (B, T) gold tag IDs; -100 means ignore.
        cfg:
            temperature: softmax temperature (default 2.0).
            alpha: weight for KD loss (default 0.5).
                   Total loss = alpha * KD + (1-alpha) * CE.

    Returns:
        Scalar loss.
    """
    temperature = cfg.get("temperature", 2.0)
    alpha = cfg.get("alpha", 0.5)

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have the same shape"
        )

    mask = tags != -100

  
    if not mask.any():
        return student_logits.sum() * 0.0

    
    teacher_logits = teacher_logits.detach()

  
    student_log_probs = F.log_softmax(
        student_logits / temperature, dim=-1
    )
    teacher_probs = F.softmax(
        teacher_logits / temperature, dim=-1
    )

    
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none"
    ).sum(dim=-1)

    kd_term = kl_per_token[mask].mean() * (temperature ** 2)

   
    ce_term = F.cross_entropy(
        student_logits[mask],
        tags[mask],
        weight=cfg.get("weight"),
    )

   
    return alpha * kd_term + (1.0 - alpha) * ce_term
