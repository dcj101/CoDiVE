from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from cd_poc.config import ensure_dirs, load_config
from cd_poc.infer import build_model, predict_letter
from cd_poc.metrics import summarize_predictions
from cd_poc.perturb import output_path_for
from cd_poc.utils import read_json, write_json


def iter_eval_points(rows: list[dict], perturb_dir: str, kinds: list[str]):
    for row in rows:
        videos = row.get("videos") or {"clean": row["video_path"]}
        yield row, "clean", videos.get("clean", row["video_path"])
        for kind in kinds:
            yield row, kind, videos.get(kind) or output_path_for(row["video_path"], perturb_dir, kind)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zero-shot clean/perturbed evaluation.")
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--input", default=None, help="Manifest with perturbation paths.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock model.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rows = read_json(args.input or Path(cfg.paths.processed_dir) / "sampled_500_with_perturbations.json")
    model_name = cfg.models.student_name if args.model == "student" else cfg.models.teacher_name
    model = build_model(
        model_name,
        dry_run=args.dry_run or cfg.models.dry_run,
        dtype=cfg.models.dtype,
        device_map=cfg.models.device_map,
        frame_stride=cfg.models.frame_stride,
        video_fps=cfg.models.video_fps,
        video_max_pixels=cfg.models.video_max_pixels,
        max_new_tokens=cfg.models.max_new_tokens,
    )

    predictions = []
    points = list(iter_eval_points(rows, cfg.paths.perturb_dir, cfg.perturbations.enabled))
    for row, kind, video_path in tqdm(points, desc=f"eval-{args.model}"):
        pred, raw = predict_letter(model, video_path, row["prompt"])
        predictions.append(
            {
                "id": row["id"],
                "kind": kind,
                "video_path": video_path,
                "answer": row["answer"],
                "prediction": pred,
                "raw_prediction": raw,
                "question_type": row.get("question_type", "unknown"),
            }
        )

    summary = summarize_predictions(predictions)
    out_prefix = Path(cfg.paths.outputs_dir) / f"zero_shot_{args.model}"
    write_json(predictions, out_prefix.with_suffix(".predictions.json"))
    write_json(summary, out_prefix.with_suffix(".metrics.json"))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
