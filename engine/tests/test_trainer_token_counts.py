"""Pure, torch-free tests for the trainer's token-accounting helpers.

The trainer counts the real non-padding and supervised tokens it consumed per optimizer step so a run
can report token throughput as paper evidence. The counting is purely observational (it never mutates a
batch), which is exactly what lets it be tested here without the training stack: the ``run_training``
wiring that feeds it live batches is exercised only on the managed GPU smoke.
"""

from __future__ import annotations

from corpus_studio.training.trainer import (
    _TokenAccumulator,
    _count_all,
    _count_ne,
    _flatten_ints,
    count_batch_tokens,
)


class _FakeTensor:
    """Stands in for a torch/numpy tensor: exposes ``tolist`` returning nested python lists."""

    def __init__(self, value: list) -> None:
        self._value = value

    def tolist(self) -> list:
        return self._value


class _VectorTensor:
    """Stands in for a torch tensor that supports vectorized ``!=``/``.sum()``/``.numel()`` without
    materializing python lists - exercises the fast counting path in ``_count_ne`` / ``_count_all``. A
    ``!=`` yields another ``_VectorTensor`` of 0/1 flags; ``.sum()`` returns a scalar-like with ``.item``.
    ``tolist`` deliberately raises so a test proves the vectorized path is taken, not the flatten path."""

    def __init__(self, flat: list[int]) -> None:
        self._flat = list(flat)

    def __ne__(self, other: object) -> "_VectorTensor":  # type: ignore[override]
        return _VectorTensor([1 if v != other else 0 for v in self._flat])

    def sum(self) -> "_Scalar":
        return _Scalar(sum(self._flat))

    def numel(self) -> int:
        return len(self._flat)

    def tolist(self) -> list:  # pragma: no cover - must never be reached on the vectorized path
        raise AssertionError("vectorized path must not fall back to tolist()")


class _Scalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
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


def test_count_ne_uses_vectorized_path_for_tensor_like() -> None:
    # A tensor-like with ``!=``/``.sum()`` is counted without materializing a python list; ``tolist``
    # would raise, so reaching a real count proves the vectorized path was taken.
    assert _count_ne(_VectorTensor([1, 1, 0, 1]), 0) == 3
    assert _count_ne(_VectorTensor([-100, 5, 6, -100]), -100) == 2


def test_count_ne_falls_back_to_flatten_for_python_lists() -> None:
    assert _count_ne([[1, 0], [1, 1]], 0) == 3
    assert _count_ne([-100, 4, -100], -100) == 1


def test_count_all_prefers_numel_then_falls_back() -> None:
    assert _count_all(_VectorTensor([9, 8, 7])) == 3  # numel(), no tolist()
    assert _count_all([[1, 2, 3], [4, 5, 6]]) == 6  # flatten fallback


def test_accumulator_sums_microbatches_and_counts_observations_then_resets() -> None:
    batch = {"attention_mask": [[1, 1, 0]], "labels": [[-100, 7, -100]]}
    acc = _TokenAccumulator()
    acc.observe(batch)
    acc.observe(batch)
    # (nonpadding, supervised, observed_microbatches): two microbatches of (2, 1), observed twice.
    assert acc.flush() == (4, 2, 2)
    assert acc.flush() == (0, 0, 0)  # flush reset every running total, including the observed count


def test_accumulator_observe_swallows_a_counting_fault_and_does_not_count_it() -> None:
    class _Explosive:
        def get(self, _key, _default=None):  # noqa: ANN001 - test double
            raise RuntimeError("boom")

    acc = _TokenAccumulator()
    acc.observe(_Explosive())  # must not raise - token accounting can never disturb training
    # A swallowed fault is NOT an observed microbatch: the count stays 0 so it surfaces downstream as
    # UNAVAILABLE (null), never a fabricated measured zero.
    assert acc.flush() == (0, 0, 0)
