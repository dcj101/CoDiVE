# Counterfactual Distillation Stage 1 POC

这是一个面向“阶段一：小规模概念验证”的项目骨架，用 500 条视频问答样本跑通：数据采样、视频扰动、零样本评测、指标统计和反事实蒸馏 loss 的 smoke test。

## 目录结构

```text
configs/stage1.yaml              # 阶段一配置
scripts/check_env.py             # 环境检查
scripts/sample_data.py            # 下载/采样 500 条样本
scripts/generate_perturbations.py # 生成 black/wrong/shortcut 视频
scripts/zero_shot_eval.py         # clean 与扰动视频零样本评测
scripts/train_cd.py               # 4 种蒸馏损失的最小训练验证
src/cd_poc/                       # 可复用代码包
```

## 1. 安装

推荐使用 Conda 管理环境。GPU 服务器使用默认环境文件：

```bash
conda env create -f environment.yml
conda activate codive
pip install -e .
python scripts/check_env.py
```

如果只在 Mac 本机做 dry-run、代码调试或小规模数据处理，使用 Mac 环境文件：

```bash
conda env create -f environment-mac.yml
conda activate codive-mac
pip install -e .
python scripts/check_env.py
```

如果不使用 Conda，也可以退回 pip 安装：

```bash
pip install -r requirements.txt
pip install -e .
```

服务器上建议确认 CUDA 和 Hugging Face 登录：

```bash
python -c "import torch; print(torch.cuda.is_available())"
huggingface-cli whoami
```

## 2. 配置模型

编辑 `configs/stage1.yaml`：

```yaml
models:
  student_name: Qwen/Qwen2-VL-2B-Instruct
  teacher_name: REPLACE_WITH_ONETHINKER_PATH
  dry_run: false
```

如果你只想先验证代码管线，不下载大模型，可以给脚本加 `--dry-run`，或把配置里的 `dry_run` 改成 `true`。

## 3. 采样 500 条数据

优先从 Hugging Face 数据集采样：

```bash
python scripts/sample_data.py --config configs/stage1.yaml --dataset docci/next-gqa --split train --video-root /path/to/videos
```

如果你已经有本地 JSON，可用：

```bash
python scripts/sample_data.py --input-json data/raw/next_gqa_train.json --video-root /path/to/videos
```

输出：

```text
data/processed/sampled_500.json
data/processed/split_500.json
```

## 4. 生成扰动视频

```bash
python scripts/generate_perturbations.py --config configs/stage1.yaml
```

输出 manifest：

```text
data/processed/sampled_500_with_perturbations.json
```

会为每条样本生成：

```text
data/perturbed/black/*_black.mp4
data/perturbed/wrong/*_wrong.mp4
data/perturbed/shortcut/*_shortcut.mp4
```

## 5. 零样本评测

学生模型：

```bash
python scripts/zero_shot_eval.py --config configs/stage1.yaml --model student
```

老师模型：

```bash
python scripts/zero_shot_eval.py --config configs/stage1.yaml --model teacher
```

先不加载真实模型时：

```bash
python scripts/zero_shot_eval.py --config configs/stage1.yaml --model student --dry-run
```

核心指标会写入：

```text
outputs/zero_shot_student.metrics.json
outputs/zero_shot_teacher.metrics.json
```

指标含义：

| 指标 | 解释 |
| --- | --- |
| `clean_acc` | 正常视频准确率，越高越好 |
| `black_acc` | 全黑视频准确率，越低越好 |
| `wrong_retention` | 错误证据视频中仍选择目标答案的比例，越低越好 |
| `crg_clean_minus_black` | `clean_acc - black_acc`，用于观察扰动敏感度 |

## 6. 蒸馏训练 Smoke Test

这个脚本先验证 4 种 loss 都能跑、不会 NaN、loss 能更新。它默认使用 mock teacher logits 和一个很小的学生头，不等价于真正微调 Qwen2-VL，但能快速排除 loss/训练循环 bug。

```bash
python scripts/train_cd.py --config configs/stage1.yaml --method all
```

输出：

```text
outputs/train_smoke_losses.json
```

支持的方法：

| 方法 | 含义 |
| --- | --- |
| `vanilla_kd` | 只学 clean teacher logits |
| `aug_kd` | clean + perturb，但扰动仍学 clean teacher |
| `counter_aug_only` | 只学扰动 teacher logits |
| `delta` | clean + perturb + delta 变化量 |

## 7. 接入真实蒸馏

真实 Qwen/OneThinker 微调通常需要按模型官方 demo 构造 video processor 输入，并从答案 token 或选项 token 提取 logits。建议扩展点：

- `src/cd_poc/infer.py`：替换或修正 `HFVideoQAModel.generate()`，适配你的多模态模型输入格式。
- `src/cd_poc/train.py`：把 `TinyStudentHead` 替换成真实 student 前向，并让 batch 返回 clean/perturb 的选项 logits。
- `src/cd_poc/losses.py`：`distillation_loss()` 和 `cd_loss()` 已按阶段一公式实现，可直接复用。

## 8. 推荐执行顺序

```bash
conda activate codive
python scripts/check_env.py
python scripts/sample_data.py --video-root /path/to/videos
python scripts/generate_perturbations.py
python scripts/zero_shot_eval.py --model student --dry-run
python scripts/train_cd.py --method all
```

dry-run 全部通过后，再把 `configs/stage1.yaml` 的 teacher/student 路径改成真实模型，运行真实零样本评测。

## 9. Conda 环境说明

| 文件 | 适用场景 |
| --- | --- |
| `environment.yml` | Linux + NVIDIA GPU 服务器，包含 `pytorch-cuda=12.1` |
| `environment-mac.yml` | macOS 本机开发、dry-run、轻量数据处理，不包含 CUDA |
| `requirements.txt` | pip 备用安装方案 |

如果服务器 CUDA 版本不是 12.1，请按实际驱动版本调整 `environment.yml` 里的 `pytorch-cuda`，例如 `11.8` 或 `12.4`。
