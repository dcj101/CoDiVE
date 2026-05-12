from __future__ import annotations

import argparse
from pathlib import Path

from cd_poc.config import ensure_dirs, load_config
from cd_poc.data import load_next_gqa, split_train_eval, stratified_sample
from cd_poc.utils import set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/sample Next-GQA rows for stage 1.")
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--dataset", default="docci/next-gqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-json", default=None, help="Use local JSON instead of Hugging Face.")
    parser.add_argument("--video-root", default=None, help="Prefix for relative video paths.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    if args.input_json:
        import json

        rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        rows = load_next_gqa(args.dataset, args.split)

    sampled = stratified_sample(rows, cfg.sample_size, cfg.seed, video_root=args.video_root)
    split = split_train_eval(sampled, cfg.train_size, cfg.seed)
    write_json(sampled, cfg.paths.sampled_json)
    write_json(split, cfg.paths.split_json)
    print(f"saved {len(sampled)} samples -> {cfg.paths.sampled_json}")
    print(f"saved train/eval split -> {cfg.paths.split_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
