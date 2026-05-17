from __future__ import annotations

"""阶段一视频扰动生成脚本。

主要功能：
1. 读取 sample_data.py 生成的 sampled_500.json。
2. 根据配置文件中启用的扰动类型，为每个原始视频生成反事实视频。
3. 当前支持 black、wrong、shortcut 三种扰动。
4. 保存一个带 videos 字段的新 manifest，供后续零样本评测和蒸馏训练读取。
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from cd_poc.config import ensure_dirs, load_config
from cd_poc.perturb import apply_perturbation, output_path_for
from cd_poc.utils import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate black/wrong/shortcut videos.")
    # 配置文件决定输入路径、输出目录、启用哪些扰动以及扰动参数。
    parser.add_argument("--config", default="configs/stage1.yaml")
    # 默认读取 cfg.paths.sampled_json；也可以手动指定其他 sampled JSON。
    parser.add_argument("--input", default=None, help="Defaults to sampled_json from config.")
    # 默认跳过已经生成过的扰动视频；加 --force 会重新生成并覆盖。
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # 创建必要目录，并加载已经标准化的样本列表。
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rows = read_json(args.input or cfg.paths.sampled_json)
    generated = []

    # 每条 row 对应一个原始视频；脚本会为它生成 clean + 多个扰动版本。
    for row in tqdm(rows, desc="perturb"):
        variants = {"clean": row["video_path"]}
        for kind in cfg.perturbations.enabled:
            # 例如 data/perturbed/black/xxx_black.mp4。
            out_path = output_path_for(row["video_path"], cfg.paths.perturb_dir, kind)
            if args.force or not Path(out_path).exists():
                # 底层 OpenCV 逻辑在 cd_poc/perturb.py：
                # black 生成全黑帧，wrong 叠加误导文字，shortcut 叠加闪烁红框。
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
        # videos 字段把 clean 和所有扰动视频路径放在一起，评测脚本会优先读取它。
        new_row["videos"] = variants
        generated.append(new_row)

    # manifest 是“样本 + 扰动视频路径”的索引文件，不是视频本身。
    out = Path(cfg.paths.processed_dir) / "sampled_500_with_perturbations.json"
    write_json(generated, out)
    print(f"saved perturbation manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
