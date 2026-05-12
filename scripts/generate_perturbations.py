from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from cd_poc.config import ensure_dirs, load_config
from cd_poc.perturb import apply_perturbation, output_path_for
from cd_poc.utils import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate black/wrong/shortcut videos.")
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--input", default=None, help="Defaults to sampled_json from config.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rows = read_json(args.input or cfg.paths.sampled_json)
    generated = []

    for row in tqdm(rows, desc="perturb"):
        variants = {"clean": row["video_path"]}
        for kind in cfg.perturbations.enabled:
            out_path = output_path_for(row["video_path"], cfg.paths.perturb_dir, kind)
            if args.force or not Path(out_path).exists():
                apply_perturbation(
                    row["video_path"],
                    out_path,
                    kind,
                    wrong_text=cfg.perturbations.wrong_text,
                    shortcut_period=cfg.perturbations.shortcut_period,
                    shortcut_on_frames=cfg.perturbations.shortcut_on_frames,
                )
            variants[kind] = out_path
        new_row = dict(row)
        new_row["videos"] = variants
        generated.append(new_row)

    out = Path(cfg.paths.processed_dir) / "sampled_500_with_perturbations.json"
    write_json(generated, out)
    print(f"saved perturbation manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
