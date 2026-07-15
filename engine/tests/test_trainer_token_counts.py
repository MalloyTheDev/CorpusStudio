"""Pure, torch-free tests for the trainer's token-accounting helpers.

The trainer counts the real non-padding and supervised tokens it consumed per optimizer step so a run
can report token throughput as paper evidence. The counting is purely observational (it never mutates a
batch), which is exactly what lets it be tested here without the training stack: the ``run_training``
wiring that feeds it live batches is exercised only on the managed GPU smoke.
"""

from __future__ import annotations

from corpus_studio.training.trainer import (
    _TokenAccumulator,
    _flatten_ints,
    count_batch_tokens,
)


class _FakeTensor:
    """Stands in for a torch/numpy tensor: exposes ``tolist`` returning nested python lists."""

    def __init__(self, value: list) -> None:
        self._value = value

    def tolist(self) -> list:
        return self._value


def test_flatten_ints_walks_nested_python_sequences() -> None:
    assert list(_flatten_ints([[1, 2], [3]])) == [1, 2, 3]
    assert list(_flatten_ints((0, (1, (2,)))) ) == [0, 1, 2]


def test_flatten_ints_uses_tolist_for_tensor_like() -> None:
    assert list(_flatten_ints(_FakeTensor([[1, 0], [1, 1]]))) == [1, 0, 1, 1]


def test_count_batch_tokens_counts_mask_and_supervised() -> None:
    batch = {
        "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "attention_mask": [[1, 1, 1, 0], [1, 1, 0, 0]],
        "labels": [[-100, 5, 6, -100], [-100, 9, -100, -100]],
    }
    # non-padding = mask ones (3 + 2); supervised = labels != -100 (2 + 1).
    assert count_batch_tokens(batch) == (5, 3)


def test_count_batch_tokens_falls_back_to_input_ids_without_mask() -> None:
    # Absent an attention mask, every input_ids position is a real (non-padding) token.
    assert count_batch_tokens({"input_ids": [[1, 2, 3], [4, 5, 6]]}) == (6, 0)


def test_count_batch_tokens_handles_tensor_like_and_missing_fields() -> None:
    batch = {
        "attention_mask": _FakeTensor([[1, 1, 0]]),
        "labels": _FakeTensor([[-100, 4, -100]]),
    }
    assert count_batch_tokens(batch) == (2, 1)
    # Nothing to count -> zeros, never an error.
    assert count_batch_tokens({}) == (0, 0)


def test_accumulator_sums_microbatches_then_resets_on_flush() -> None:
    batch = {"attention_mask": [[1, 1, 0]], "labels": [[-100, 7, -100]]}
    acc = _TokenAccumulator()
    acc.observe(batch)
    acc.observe(batch)
    assert acc.flush() == (4, 2)  # two microbatches of (2, 1)
    assert acc.flush() == (0, 0)  # flush reset the running total


def test_accumulator_observe_swallows_a_counting_fault() -> None:
    class _Explosive:
        def get(self, _key, _default=None):  # noqa: ANN001 - test double
            raise RuntimeError("boom")

    acc = _TokenAccumulator()
    acc.observe(_Explosive())  # must not raise - token accounting can never disturb training
    assert acc.flush() == (0, 0)
