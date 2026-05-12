from __future__ import annotations

import hashlib
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
        digest = hashlib.sha256(f"{self.name}|{video_path}|{prompt}".encode()).digest()
        return ANSWER_LETTERS[digest[0] % len(ANSWER_LETTERS)]


class HFVideoQAModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        frame_stride: int = 10,
        max_new_tokens: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.frame_stride = frame_stride
        self.max_new_tokens = max_new_tokens

    def _read_frames(self, video_path: str):
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) == 0:
            raise ValueError(f"Empty video: {video_path}")
        indices = list(range(0, len(vr), max(1, self.frame_stride)))
        frames = vr.get_batch(indices)
        return [Image.fromarray(frame.asnumpy()) for frame in frames]

    def generate(self, video_path: str, prompt: str) -> str:
        import torch

        frames = self._read_frames(video_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=frames, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
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
    return parse_prediction(raw), raw
