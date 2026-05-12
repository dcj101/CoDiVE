from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)


def cd_loss(
    student_clean_logits: torch.Tensor,
    teacher_clean_logits: torch.Tensor,
    student_perturb_logits: torch.Tensor,
    teacher_perturb_logits: torch.Tensor,
    temperature: float = 3.0,
    delta_weight: float = 0.7,
) -> torch.Tensor:
    loss_clean = distillation_loss(student_clean_logits, teacher_clean_logits, temperature)
    loss_perturb = distillation_loss(student_perturb_logits, teacher_perturb_logits, temperature)
    delta_teacher = F.softmax(teacher_clean_logits, dim=-1) - F.softmax(teacher_perturb_logits, dim=-1)
    delta_student = F.softmax(student_clean_logits, dim=-1) - F.softmax(student_perturb_logits, dim=-1)
    loss_delta = F.mse_loss(delta_student, delta_teacher)
    return loss_clean + loss_perturb + delta_weight * loss_delta
