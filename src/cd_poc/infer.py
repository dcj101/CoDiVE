from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from .data import ANSWER_LETTERS, parse_prediction


class VideoQAModel(Protocol):
    def generate(self, video_path: str, prompt: str) -> str: ...


@dataclass
class MockVideoQAModel:
    """Deterministic model used to test the pipeline without downloading large checkpoints."""

    name: str = "mock"

    def generate(self, video_path: str, prompt: str) -> str:
        import re

        digest = hashlib.sha256(f"{self.name}|{video_path}|{prompt}".encode()).digest()
        letters = re.findall(r"^([A-Z])\)", prompt, flags=re.MULTILINE)
        candidates = letters or ANSWER_LETTERS[:4]
        return candidates[digest[0] % len(candidates)]


class HFVideoQAModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        frame_stride: int = 10,
        video_fps: float | None = None,
        video_min_pixels: int = 112 * 112,
        video_max_pixels: int = 224 * 224,
        max_new_tokens: int = 32,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = self._load_model(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.frame_stride = frame_stride
        self.video_fps = video_fps
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _load_model(model_name: str, **kwargs):
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        return Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    def _target_device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _video_fps(self) -> float:
        if self.video_fps is not None:
            return max(0.05, float(self.video_fps))
        # Qwen-VL utils accepts fps rather than frame stride. This keeps long videos cheap.
        return max(0.2, min(1.0, 30.0 / max(1, self.frame_stride)))

    def _build_qwen_inputs(self, video_path: str, prompt: str):
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "真实 Qwen2-VL 视频推理需要安装 qwen-vl-utils。"
                "请运行 `pip install qwen-vl-utils` 或重新创建 conda 环境。"
            ) from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self._video_fps(),
                        "min_pixels": self.video_min_pixels,
                        "max_pixels": self.video_max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self._target_device())

    def generate(self, video_path: str, prompt: str) -> str:
        import torch

        inputs = self._build_qwen_inputs(video_path, prompt)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def build_model(model_name: str, dry_run: bool, **kwargs) -> VideoQAModel:
    if dry_run:
        return MockVideoQAModel(model_name)
    return HFVideoQAModel(model_name, **kwargs)


def predict_letter(model: VideoQAModel, video_path: str, prompt: str) -> tuple[str, str]:
    raw = model.generate(video_path, prompt)
    options = [match.group(2).strip() for match in re.finditer(r"^([A-Z])\)\s*(.+)$", prompt, re.MULTILINE)]
    return parse_prediction(raw, options), raw
