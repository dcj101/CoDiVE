from __future__ import annotations

from pathlib import Path


def _open_video(video_path: str):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video size for {video_path}: {width}x{height}")
    return cap, fps, width, height


def _writer(output_path: str, fps: float, width: int, height: int):
    import cv2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise OSError(f"Cannot write video: {output_path}")
    return out


def apply_black(video_path: str, output_path: str) -> None:
    import numpy as np

    cap, fps, width, height = _open_video(video_path)
    out = _writer(output_path, fps, width, height)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame_count = int(cap.get(7))
    if frame_count > 0:
        for _ in range(frame_count):
            out.write(black_frame)
    else:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            out.write(black_frame)
    cap.release()
    out.release()


def apply_wrong(video_path: str, output_path: str, wrong_text: str = "A red balloon") -> None:
    import cv2

    cap, fps, width, height = _open_video(video_path)
    out = _writer(output_path, fps, width, height)
    scale = max(width, height) / 360
    thickness = max(2, int(scale * 2))
    origin = (max(10, width // 12), max(40, height // 2))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, wrong_text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
        cv2.putText(frame, wrong_text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 1)
        out.write(frame)
    cap.release()
    out.release()


def apply_shortcut(
    video_path: str,
    output_path: str,
    period: int = 10,
    on_frames: int = 5,
) -> None:
    import cv2

    cap, fps, width, height = _open_video(video_path)
    out = _writer(output_path, fps, width, height)
    frame_idx = 0
    period = max(1, period)
    on_frames = max(1, min(on_frames, period))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % period < on_frames:
            x1, y1 = width // 12, height // 12
            x2, y2 = min(width - 1, x1 + width // 4), min(height - 1, y1 + height // 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), max(3, width // 160))
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()


def output_path_for(video_path: str, perturb_dir: str, kind: str) -> str:
    src = Path(video_path)
    safe_stem = src.stem.replace("/", "_")
    return str(Path(perturb_dir) / kind / f"{safe_stem}_{kind}.mp4")


def apply_perturbation(video_path: str, output_path: str, kind: str, **kwargs) -> None:
    if kind == "black":
        apply_black(video_path, output_path)
    elif kind == "wrong":
        apply_wrong(video_path, output_path, wrong_text=kwargs.get("wrong_text", "A red balloon"))
    elif kind == "shortcut":
        apply_shortcut(
            video_path,
            output_path,
            period=int(kwargs.get("shortcut_period", 10)),
            on_frames=int(kwargs.get("shortcut_on_frames", 5)),
        )
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")
