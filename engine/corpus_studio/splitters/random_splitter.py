import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def random_split(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> SplitResult:
    rows_copy = list(rows)
    random.Random(seed).shuffle(rows_copy)

    total = len(rows_copy)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    return SplitResult(
        train=rows_copy[:train_end],
        validation=rows_copy[train_end:validation_end],
        test=rows_copy[validation_end:],
    )
