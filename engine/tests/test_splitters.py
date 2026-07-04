import pytest

from corpus_studio.splitters.random_splitter import random_split


def _rows(count: int) -> list[dict]:
    return [{"i": index} for index in range(count)]


def test_split_is_deterministic_for_a_seed():
    rows = _rows(100)
    first = random_split(rows, seed=7)
    second = random_split(rows, seed=7)
    assert first == second


def test_split_partitions_every_row_exactly_once():
    rows = _rows(100)
    result = random_split(rows, train_ratio=0.8, validation_ratio=0.1)
    combined = result.train + result.validation + result.test
    assert len(combined) == 100
    assert sorted(combined, key=lambda row: row["i"]) == rows


def test_split_ratio_counts():
    result = random_split(_rows(100), train_ratio=0.8, validation_ratio=0.1)
    assert len(result.train) == 80
    assert len(result.validation) == 10
    assert len(result.test) == 10


def test_split_handles_empty_and_single_row():
    empty = random_split([])
    assert empty.train == [] and empty.validation == [] and empty.test == []

    one = random_split(_rows(1))
    assert len(one.train) + len(one.validation) + len(one.test) == 1


def test_split_does_not_mutate_input():
    rows = _rows(10)
    snapshot = list(rows)
    random_split(rows, seed=1)
    assert rows == snapshot


# --- item 14: ratio guards + non-empty validation on small datasets ----------

def test_ratios_summing_over_one_are_rejected():
    with pytest.raises(ValueError):
        random_split(_rows(100), train_ratio=0.9, validation_ratio=0.5)


def test_negative_ratio_is_rejected():
    with pytest.raises(ValueError):
        random_split(_rows(100), train_ratio=-0.1, validation_ratio=0.2)


def test_small_dataset_still_gets_a_validation_row():
    # 15 * 0.05 = 0.75 floors to 0; validation must still receive at least one row,
    # and every row is partitioned exactly once with train left non-empty.
    result = random_split(_rows(15), train_ratio=0.9, validation_ratio=0.05, seed=3)
    assert len(result.validation) >= 1
    assert len(result.train) >= 1
    combined = result.train + result.validation + result.test
    assert sorted(row["i"] for row in combined) == list(range(15))


def test_validation_bump_never_empties_train():
    # A single row cannot spare a validation row without emptying train — no bump, no crash.
    one = random_split(_rows(1), train_ratio=0.9, validation_ratio=0.05)
    assert len(one.train) + len(one.validation) + len(one.test) == 1
