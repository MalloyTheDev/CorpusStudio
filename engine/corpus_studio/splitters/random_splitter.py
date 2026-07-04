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
    if train_ratio < 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio > 1.0 + 1e-9:
        raise ValueError(
            "train_ratio + validation_ratio must not exceed 1 "
            f"(got {train_ratio} + {validation_ratio}); the remainder is the test split"
        )

    rows_copy = list(rows)
    random.Random(seed).shuffle(rows_copy)

    total = len(rows_copy)
    train_end = int(total * train_ratio)
    validation_count = int(total * validation_ratio)
    # A small dataset floors the validation fraction to zero (e.g. 15 rows * 0.05 -> 0),
    # silently producing an empty validation set. Give validation at least one row when a
    # validation fraction was requested and one can be spared without emptying train.
    if validation_ratio > 0 and validation_count == 0 and train_end >= 1 and total - train_end >= 1:
        validation_count = 1
    validation_end = train_end + validation_count

    return SplitResult(
        train=rows_copy[:train_end],
        validation=rows_copy[train_end:validation_end],
        test=rows_copy[validation_end:],
    )
