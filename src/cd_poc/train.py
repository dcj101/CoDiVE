from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .losses import cd_loss, distillation_loss


@dataclass
class LogitExample:
    clean_features: list[float]
    perturb_features: list[float]
    teacher_clean_logits: list[float]
    teacher_perturb_logits: list[float]


class LogitDataset(Dataset):
    def __init__(self, rows: Iterable[dict]) -> None:
        self.rows = [LogitExample(**row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "clean_features": torch.tensor(row.clean_features, dtype=torch.float32),
            "perturb_features": torch.tensor(row.perturb_features, dtype=torch.float32),
            "teacher_clean_logits": torch.tensor(row.teacher_clean_logits, dtype=torch.float32),
            "teacher_perturb_logits": torch.tensor(row.teacher_perturb_logits, dtype=torch.float32),
        }


class TinyStudentHead(nn.Module):
    """Small trainable head for smoke-testing CD loss before full VL fine-tuning."""

    def __init__(self, input_dim: int, num_options: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_options),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_logit_head(
    rows: list[dict],
    method: str,
    epochs: int,
    lr: float,
    temperature: float,
    delta_weight: float,
    grad_clip_norm: float,
) -> list[float]:
    if not rows:
        raise ValueError("No training rows provided")
    input_dim = len(rows[0]["clean_features"])
    dataset = LogitDataset(rows)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = TinyStudentHead(input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(epochs):
        for batch in loader:
            clean_logits = model(batch["clean_features"])
            perturb_logits = model(batch["perturb_features"])
            if method == "vanilla_kd":
                loss = distillation_loss(clean_logits, batch["teacher_clean_logits"], temperature)
            elif method == "aug_kd":
                loss = distillation_loss(clean_logits, batch["teacher_clean_logits"], temperature)
                loss = loss + distillation_loss(perturb_logits, batch["teacher_clean_logits"], temperature)
            elif method == "counter_aug_only":
                loss = distillation_loss(perturb_logits, batch["teacher_perturb_logits"], temperature)
            elif method == "delta":
                loss = cd_loss(
                    clean_logits,
                    batch["teacher_clean_logits"],
                    perturb_logits,
                    batch["teacher_perturb_logits"],
                    temperature=temperature,
                    delta_weight=delta_weight,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
    return losses
