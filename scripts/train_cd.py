from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from cd_poc.config import ensure_dirs, load_config
from cd_poc.train import train_logit_head
from cd_poc.utils import read_json, set_seed, write_json


METHODS = ["vanilla_kd", "aug_kd", "counter_aug_only", "delta"]


def _stable_vec(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


def _mock_teacher_logits(row: dict, kind: str) -> list[float]:
    answer = row.get("answer", "A")
    idx = max(0, min(3, ord(answer[:1]) - ord("A"))) if answer else 0
    logits = [-1.0, -1.0, -1.0, -1.0]
    if kind == "clean":
        logits[idx] = 3.0
    elif kind == "black":
        logits = [0.0, 0.0, 0.0, 0.0]
    else:
        logits[idx] = 0.5
    return logits


def build_smoke_rows(rows: list[dict], perturb_kind: str = "black") -> list[dict]:
    examples = []
    for row in rows:
        examples.append(
            {
                "clean_features": _stable_vec(row["video_path"] + row["prompt"]),
                "perturb_features": _stable_vec(row["video_path"] + row["prompt"] + perturb_kind),
                "teacher_clean_logits": _mock_teacher_logits(row, "clean"),
                "teacher_perturb_logits": _mock_teacher_logits(row, perturb_kind),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test CD losses on cached/mock logits.")
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--input", default=None, help="Defaults to sampled_json from config.")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    rows = read_json(args.input or cfg.paths.sampled_json)
    train_rows = build_smoke_rows(rows)
    methods = METHODS if args.method == "all" else [args.method]
    results = {}

    for method in methods:
        losses = train_logit_head(
            train_rows,
            method=method,
            epochs=cfg.training.epochs,
            lr=cfg.training.lr,
            temperature=cfg.training.temperature,
            delta_weight=cfg.training.delta_weight,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )
        results[method] = {
            "steps": len(losses),
            "first_loss": losses[0] if losses else math.nan,
            "last_loss": losses[-1] if losses else math.nan,
            "loss_has_nan": any(math.isnan(x) for x in losses),
        }
        print(method, results[method])

    out = Path(cfg.paths.outputs_dir) / "train_smoke_losses.json"
    write_json(results, out)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
