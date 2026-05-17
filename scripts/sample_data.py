from __future__ import annotations

"""阶段一数据采样脚本。

主要功能：
1. 从 Hugging Face 数据集或本地 JSON 标注文件读取原始视频问答样本。
2. 将不同来源的数据字段统一成项目内部格式，例如 video_path、question、options、answer。
3. 按配置文件里的 sample_size 抽取小规模样本，便于先跑通阶段一流程。
4. 按 train_size 生成 train/eval 划分，供后续蒸馏训练和评估使用。
5. 输出 sampled_500.json 和 split_500.json，后续扰动生成、评测、训练脚本都会读取它们。
"""

import argparse
from pathlib import Path

from cd_poc.config import ensure_dirs, load_config
from cd_poc.data import load_next_gqa, split_train_eval, stratified_sample
from cd_poc.utils import set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/sample Next-GQA rows for stage 1.")
    # 配置文件里定义了输出路径、采样数量、训练集大小和随机种子。
    parser.add_argument("--config", default="configs/stage1.yaml")
    # Hugging Face 数据集 ID；仅适用于能被 datasets.load_dataset() 直接读取的数据集。
    parser.add_argument("--dataset", default="sming256/NeXTGQA")
    # 当前公开的 NextGQA 镜像通常只提供 test split。
    parser.add_argument("--split", default="test")
    # 如果你已经下载了官方标注文件，可以用本地 JSON 代替 Hugging Face 数据集。
    parser.add_argument("--input-json", default=None, help="Use local JSON instead of Hugging Face.")
    # 相对视频路径的根目录，例如样本里的 "1101/xxx.mp4" 会拼到这个目录下面。
    parser.add_argument("--video-root", default=None, help="Prefix for relative video paths.")
    args = parser.parse_args()

    # 创建输出目录，并固定随机种子，保证多次采样结果可复现。
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    # 数据来源优先级：
    # 1. 如果传入 --input-json，就读取本地标注 JSON。
    # 2. 否则，从 Hugging Face 下载/读取指定 split。
    if args.input_json:
        import json

        rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        rows = load_next_gqa(args.dataset, args.split)

    # 先把原始字段统一成项目内部格式，再抽取阶段一需要的小样本集合。
    sampled = stratified_sample(rows, cfg.sample_size, cfg.seed, video_root=args.video_root)
    # 额外保存 train/eval 划分，方便后续蒸馏训练和评估复用。
    split = split_train_eval(sampled, cfg.train_size, cfg.seed)
    # 后续生成扰动视频、零样本评测、训练 smoke test 默认读取这些文件。
    write_json(sampled, cfg.paths.sampled_json)
    write_json(split, cfg.paths.split_json)
    print(f"saved {len(sampled)} samples -> {cfg.paths.sampled_json}")
    print(f"saved train/eval split -> {cfg.paths.split_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
