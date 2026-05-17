from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


ANSWER_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _first_existing(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def normalize_item(row: dict[str, Any], video_root: str | None = None) -> dict[str, Any]:
    video = _first_existing(row, ["video_path", "video", "video_id", "filename", "file_name"])
    if video is None:
        raise ValueError(f"Cannot find video field in row keys: {sorted(row)}")
    video_path = str(video)
    if video_root and not Path(video_path).is_absolute():
        video_path = str(Path(video_root) / video_path)

    question = str(_first_existing(row, ["question", "query", "q"], ""))
    options = _first_existing(row, ["options", "choices", "candidates"], [])
    if isinstance(options, dict):
        options = [options.get(letter, "") for letter in ANSWER_LETTERS if letter in options]
    if isinstance(options, str):
        options = re.split(r"\s*[A-Z][\).:]\s*", options)
        options = [opt for opt in options if opt]
    options = list(options) if options is not None else []

    answer = _first_existing(row, ["answer", "label", "correct", "correct_answer"], "")
    question_type = _first_existing(row, ["question_type", "type", "category"], "unknown")

    item = {
        "id": str(_first_existing(row, ["id", "qid", "question_id"], len(str(row)))),
        "video_path": video_path,
        "question": question,
        "options": options,
        "answer": normalize_answer(answer, options),
        "question_type": str(question_type),
    }
    item["prompt"] = build_prompt(item["question"], item["options"])
    return item


def build_prompt(question: str, options: list[str] | None = None) -> str:
    lines = [f"Question: {question}"]
    if options:
        lines.append("Options:")
        for letter, option in zip(ANSWER_LETTERS, options):
            lines.append(f"{letter}) {option}")
        valid_letters = ", ".join(ANSWER_LETTERS[: len(options)])
        lines.append(f"Answer with only the option letter ({valid_letters}).")
    else:
        lines.append("Answer concisely.")
    return "\n".join(lines)


def normalize_answer(answer: Any, options: list[str] | None = None) -> str:
    text = str(answer).strip()
    if not text:
        return text
    option_count = len(options or [])
    valid_letters = ANSWER_LETTERS[:option_count] if option_count else ANSWER_LETTERS
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in valid_letters:
        return match.group(1)
    if options:
        lowered = text.casefold()
        for idx, option in enumerate(options):
            if lowered == str(option).strip().casefold():
                return ANSWER_LETTERS[idx]
    if match:
        return match.group(1)
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(valid_letters):
            return ANSWER_LETTERS[idx]
        if 1 <= idx <= len(valid_letters):
            return ANSWER_LETTERS[idx - 1]
    return text


def parse_prediction(text: str, options: list[str] | None = None) -> str:
    if options:
        lowered = text.strip().casefold()
        for idx, option in enumerate(options):
            option_text = str(option).strip().casefold()
            if lowered == option_text or option_text in lowered:
                return ANSWER_LETTERS[idx]
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1)
    return text.strip()[:1].upper()


def load_next_gqa(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    video_root: str | None = None,
) -> list[dict[str, Any]]:
    normalized = [normalize_item(row, video_root=video_root) for row in rows]
    df = pd.DataFrame(normalized)
    if len(df) <= sample_size:
        return df.to_dict("records")

    stratify = df["question_type"] if df["question_type"].nunique() > 1 else None
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=seed,
        stratify=stratify,
    )
    return sampled.reset_index(drop=True).to_dict("records")


def split_train_eval(rows: list[dict[str, Any]], train_size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    if train_size >= len(rows):
        return {"train": rows, "eval": []}
    df = pd.DataFrame(rows)
    stratify = df["question_type"] if "question_type" in df and df["question_type"].nunique() > 1 else None
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None
    train_df, eval_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    return {
        "train": train_df.reset_index(drop=True).to_dict("records"),
        "eval": eval_df.reset_index(drop=True).to_dict("records"),
    }
