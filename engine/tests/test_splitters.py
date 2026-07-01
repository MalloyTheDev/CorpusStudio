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
