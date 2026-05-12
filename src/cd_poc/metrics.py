from __future__ import annotations

from collections import defaultdict
from typing import Any

from .data import parse_prediction


def accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if parse_prediction(row["prediction"]) == row["answer"])
    return correct / len(rows)


def wrong_retention(rows: list[dict[str, Any]]) -> float:
    """Approximation: rate of choosing the injected wrong option on wrong-evidence videos.

    If a row has `wrong_answer`, use that. Otherwise this falls back to the clean answer, which is a
    conservative placeholder until you define explicit wrong labels per question.
    """

    if not rows:
        return 0.0
    retained = 0
    for row in rows:
        target = row.get("wrong_answer") or row.get("answer")
        retained += int(parse_prediction(row["prediction"]) == target)
    return retained / len(rows)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row.get("kind", "clean")].append(row)

    clean_acc = accuracy(by_kind.get("clean", []))
    black_acc = accuracy(by_kind.get("black", []))
    summary: dict[str, Any] = {
        "clean_acc": clean_acc,
        "black_acc": black_acc,
        "wrong_retention": wrong_retention(by_kind.get("wrong", [])),
        "crg_clean_minus_black": clean_acc - black_acc,
        "counts": {kind: len(items) for kind, items in by_kind.items()},
    }
    for kind, items in by_kind.items():
        summary[f"{kind}_acc"] = accuracy(items)
    return summary
