from __future__ import annotations

import importlib.util
import subprocess


PACKAGES = ["torch", "transformers", "datasets", "accelerate", "cv2", "decord"]


def main() -> int:
    for package in PACKAGES:
        ok = importlib.util.find_spec(package) is not None
        print(f"{package:14s}: {'OK' if ok else 'MISSING'}")

    try:
        import torch

        print(f"cuda_available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device    : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch check failed: {exc}")

    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"huggingface    : {result.stdout.strip() or result.stderr.strip()}")
    except Exception as exc:
        print(f"huggingface    : unavailable ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
