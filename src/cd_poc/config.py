from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    raw_data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    perturb_dir: str = "data/perturbed"
    outputs_dir: str = "outputs"
    sampled_json: str = "data/processed/sampled_500.json"
    split_json: str = "data/processed/split_500.json"
    metrics_json: str = "outputs/stage1_metrics.json"


@dataclass
class ModelsConfig:
    student_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    teacher_name: str = "REPLACE_WITH_ONETHINKER_PATH"
    dtype: str = "float16"
    device_map: str = "auto"
    frame_stride: int = 10
    video_fps: float = 0.5
    video_min_pixels: int = 12544
    video_max_pixels: int = 50176
    max_new_tokens: int = 32
    dry_run: bool = False


@dataclass
class PerturbConfig:
    enabled: list[str] = field(default_factory=lambda: ["black", "wrong", "shortcut"])
    wrong_text: str = "A red balloon"
    shortcut_period: int = 10
    shortcut_on_frames: int = 5


@dataclass
class TrainingConfig:
    epochs: int = 3
    lr: float = 1e-5
    temperature: float = 3.0
    delta_weight: float = 0.7
    grad_clip_norm: float = 1.0
    batch_size: int = 1


@dataclass
class Stage1Config:
    seed: int = 42
    sample_size: int = 500
    train_size: int = 400
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    perturbations: PerturbConfig = field(default_factory=PerturbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dataclass(cls: type, values: dict[str, Any]) -> Any:
    allowed = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in values.items() if key in allowed}
    return cls(**kwargs)


def load_config(path: str | Path) -> Stage1Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Stage1Config(
        seed=raw.get("seed", 42),
        sample_size=raw.get("sample_size", 500),
        train_size=raw.get("train_size", 400),
        paths=_merge_dataclass(PathsConfig, raw.get("paths", {})),
        models=_merge_dataclass(ModelsConfig, raw.get("models", {})),
        perturbations=_merge_dataclass(PerturbConfig, raw.get("perturbations", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
    )
    return cfg


def ensure_dirs(cfg: Stage1Config) -> None:
    for path in [
        cfg.paths.raw_data_dir,
        cfg.paths.processed_dir,
        cfg.paths.perturb_dir,
        cfg.paths.outputs_dir,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
