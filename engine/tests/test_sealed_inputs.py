"""The shared consumption-time dataset verification (``training.sealed_inputs``, #862) and the two read
primitives it relies on: the bounded stable read (``execution_config.stable_file_bytes``) and the
byte-level JSONL parser (``jsonl_importer.read_jsonl_bytes``), whose line semantics must match the path
reader that planning uses. Torch-free and deterministic."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import corpus_studio.training.sealed_inputs as sealed_inputs
from corpus_studio.importers.jsonl_importer import read_jsonl, read_jsonl_bytes
from corpus_studio.platform.execution_config import (
    ExecutionConfigurationError,
    local_input_binding,
    stable_file_bytes,
    stable_file_sha256,
)
from corpus_studio.training.sealed_inputs import (
    MAX_DATASET_PROGRESS_EVENTS,
    SealedInputError,
    VerifiedDataset,
    bounded_byte_progress,
    read_verified_dataset,
    require_verified_dataset,
    verify_sealed_dataset_bytes,
)

_ROWS = [{"instruction": f"i{index}", "output": f"o{index}"} for index in range(3)]
_MIB = 1024 * 1024


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _bind(path: Path):
    return local_input_binding(kind="dataset", location=str(path), ref_id="dataset", directory=False)


def _large_rows(minimum_bytes: int) -> list[dict]:
    # Enough rows that the file spans more than one 1 MiB read chunk.
    row = {"instruction": "x" * 1000, "output": "y"}
    count = minimum_bytes // (len(json.dumps(row)) + 1) + 1
    return [row] * count


# --- read_verified_dataset: one read, hash, compare, parse the same bytes ---------------------------


def test_read_verified_dataset_returns_rows_parsed_from_the_hashed_bytes(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    binding = _bind(data)

    verified = read_verified_dataset(binding)

    assert verified.content_sha256 == stable_file_sha256(data) == binding.content_sha256
    assert verified.byte_count == data.stat().st_size
    assert list(verified.rows) == list(read_jsonl(data)) == _ROWS
    assert verified.row_count == len(_ROWS)
    assert verified.location == binding.location


def test_read_verified_dataset_refuses_post_plan_mutation(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    binding = _bind(data)
    _write_rows(data, [{"instruction": "TAMPERED", "output": "o"}])

    with pytest.raises(SealedInputError, match="dataset bytes changed after the execution configuration"):
        read_verified_dataset(binding)


def test_read_verified_dataset_refuses_a_size_changing_mid_read_mutation(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _large_rows(2 * _MIB))
    binding = _bind(data)
    appended: list[int] = []

    def _append_once(completed: int, total: int) -> None:
        if not appended:
            appended.append(completed)
            with data.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"instruction": "EVIL", "output": "o"}) + "\n")

    with pytest.raises(SealedInputError, match="changed while hashing"):
        read_verified_dataset(binding, progress_callback=_append_once)
    assert appended


def test_read_verified_dataset_refuses_a_same_size_mid_read_rewrite_with_restored_mtime(tmp_path):
    # An in-place rewrite of a later chunk that keeps the size and restores mtime evades every stat
    # identity check; the digest comparison still refuses it because the parsed bytes ARE the hashed bytes.
    data = tmp_path / "train.jsonl"
    _write_rows(data, _large_rows(2 * _MIB))
    binding = _bind(data)
    original = data.stat()
    rewritten: list[int] = []

    def _rewrite_tail_once(completed: int, total: int) -> None:
        if rewritten:
            return
        rewritten.append(completed)
        with data.open("r+b") as handle:
            handle.seek(total - 10)
            handle.write(b"Z" * 8)
        os.utime(data, ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(SealedInputError, match="dataset bytes changed after the execution configuration"):
        read_verified_dataset(binding, progress_callback=_rewrite_tail_once)
    assert rewritten == [_MIB]


def test_read_verified_dataset_refuses_missing_link_and_directory_inputs(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    binding = _bind(data)

    data.unlink()
    with pytest.raises(SealedInputError, match="does not exist"):
        read_verified_dataset(binding)

    data.mkdir()
    with pytest.raises(SealedInputError, match="does not exist"):
        read_verified_dataset(binding)
    data.rmdir()

    target = tmp_path / "real.jsonl"
    _write_rows(target, _ROWS)
    try:
        data.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")
    with pytest.raises(SealedInputError, match="cannot be a link"):
        read_verified_dataset(binding)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"instruction": "a", "output": "b"}\n{bad\n', "sealed dataset is invalid: line 2: Invalid JSON"),
        (b'{"instruction": "a", "output": "b"}\n[1, 2]\n', "line 2: expected a JSON object, got list"),
        (b"\xff\xfe not utf-8\n", "sealed dataset is invalid: dataset is not valid UTF-8"),
        (b"\n  \n\t\n", "the sealed dataset contains no rows"),
        (b"", "the sealed dataset contains no rows"),
    ],
)
def test_read_verified_dataset_refuses_sealed_but_unusable_content(tmp_path, content, message):
    # The bytes match the seal, so only the parse can refuse them - and it must, before any worker runs.
    data = tmp_path / "train.jsonl"
    data.write_bytes(content)
    with pytest.raises(SealedInputError, match=message):
        read_verified_dataset(_bind(data))


def test_read_verified_dataset_refuses_a_non_dataset_binding_and_a_missing_digest(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    model_binding = local_input_binding(
        kind="model", location=str(data), ref_id="model", directory=False
    )
    with pytest.raises(SealedInputError, match="not a pinned local dataset file"):
        read_verified_dataset(model_binding)
    with pytest.raises(SealedInputError, match="omitted the dataset content digest"):
        verify_sealed_dataset_bytes(str(data), None)


def test_verify_sealed_dataset_bytes_returns_the_exact_bytes_and_digest(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    content, digest = verify_sealed_dataset_bytes(str(data), stable_file_sha256(data))
    assert content == data.read_bytes()
    assert digest == stable_file_sha256(data)


# --- require_verified_dataset: the worker-side binding guard ---------------------------------------


def test_require_verified_dataset_rejects_missing_or_foreign_rows(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    binding = _bind(data)
    verified = read_verified_dataset(binding)

    assert require_verified_dataset(verified, binding) is verified
    for missing in (None, list(_ROWS)):
        with pytest.raises(SealedInputError, match="requires rows from the verified sealed dataset read"):
            require_verified_dataset(missing, binding)
    wrong_digest = VerifiedDataset(
        rows=verified.rows, content_sha256="0" * 64, byte_count=1, location=verified.location
    )
    wrong_location = VerifiedDataset(
        rows=verified.rows,
        content_sha256=verified.content_sha256,
        byte_count=verified.byte_count,
        location=str(tmp_path / "other.jsonl"),
    )
    for foreign in (wrong_digest, wrong_location):
        with pytest.raises(SealedInputError, match="does not belong to this sealed execution"):
            require_verified_dataset(foreign, binding)


# --- bounded_byte_progress: honest, capped progress ------------------------------------------------


def test_bounded_byte_progress_emits_at_most_the_cap_monotonically():
    messages: list[str] = []
    progress = bounded_byte_progress(messages.append)
    total = 10_000 * _MIB
    for chunk in range(1, 10_001):
        progress(chunk * _MIB, total)

    assert 1 <= len(messages) <= MAX_DATASET_PROGRESS_EVENTS
    completed = [int(message.split()[3].split("/")[0]) for message in messages]
    assert completed == sorted(set(completed))
    assert messages[-1] == f"read and hashed {total}/{total} sealed dataset bytes"


def test_bounded_byte_progress_ignores_an_empty_total_and_honors_a_custom_cap():
    messages: list[str] = []
    progress = bounded_byte_progress(messages.append, max_events=2)
    progress(0, 0)
    assert messages == []
    for completed in (1, 2, 3, 4):
        progress(completed, 4)
    assert messages == [
        "read and hashed 1/4 sealed dataset bytes",
        "read and hashed 4/4 sealed dataset bytes",
    ]


def test_sealed_inputs_import_pulls_no_heavy_training_stack():
    code = (
        "import sys\n"
        "import corpus_studio.platform\n"
        "import corpus_studio.training.sealed_inputs\n"
        "heavy = ('torch', 'transformers', 'peft', 'trl', 'datasets', 'bitsandbytes', 'safetensors')\n"
        "print(sorted(name for name in heavy if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert completed.stdout.strip() == "[]"


# --- stable_file_bytes: the read is bounded by the size observed at open ----------------------------


def test_stable_file_bytes_refuses_a_file_that_keeps_growing_after_open(tmp_path):
    data = tmp_path / "growing.jsonl"
    data.write_bytes(b"a" * (2 * _MIB + 5))
    calls: list[tuple[int, int]] = []

    def _append_every_time(completed: int, total: int) -> None:
        calls.append((completed, total))
        with data.open("ab") as handle:
            handle.write(b"b" * _MIB)

    with pytest.raises(ExecutionConfigurationError, match="changed while hashing"):
        stable_file_bytes(data, progress_callback=_append_every_time)
    # Bounded by the size seen at open (three chunks), never by how long the writer keeps appending.
    assert [total for _completed, total in calls] == [2 * _MIB + 5] * 3
    assert calls[-1][0] == 2 * _MIB + 5


def test_stable_file_bytes_refuses_a_file_truncated_during_the_read(tmp_path):
    data = tmp_path / "shrinking.jsonl"
    data.write_bytes(b"a" * (2 * _MIB))

    def _truncate(completed: int, total: int) -> None:
        with data.open("r+b") as handle:
            handle.truncate(completed)

    with pytest.raises(ExecutionConfigurationError, match="changed while hashing"):
        stable_file_bytes(data, progress_callback=_truncate)


# --- read_jsonl_bytes: the same line semantics as read_jsonl ----------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        # raw U+2028 / U+2029 / U+0085 are legal inside a JSON string and must not split a row
        '{"text": "a\u2028b"}\n{"text": "c\u2029d"}\n{"text": "e\u0085f"}\n'.encode("utf-8"),
        b'{"a": 1}\r\n{"a": 2}\r\n',
        b'{"a": 1}\r{"a": 2}\r',
        '\ufeff{"a": 1}\n{"a": 2}\n'.encode("utf-8"),
        b'{"a": 1}\n   \n\t\n{"a": 2}\n',
        b'{"a": 1}\n{"a": 2}',
    ],
    ids=["unicode-line-separators", "crlf", "lone-cr", "bom", "blank-lines", "no-trailing-newline"],
)
def test_read_jsonl_bytes_matches_read_jsonl_line_semantics(tmp_path, content):
    data = tmp_path / "rows.jsonl"
    data.write_bytes(content)
    assert list(read_jsonl_bytes(content)) == list(read_jsonl(data))


@pytest.mark.parametrize(
    "content",
    [
        '{"text": "a\u2028b"}\n[1]\n'.encode("utf-8"),
        '{"text": "a\u2029b"}\n{bad\n'.encode("utf-8"),
    ],
)
def test_read_jsonl_bytes_reports_the_same_error_line_numbers(tmp_path, content):
    data = tmp_path / "rows.jsonl"
    data.write_bytes(content)
    with pytest.raises(ValueError) as from_path:
        list(read_jsonl(data))
    with pytest.raises(ValueError) as from_bytes:
        list(read_jsonl_bytes(content))
    assert str(from_path.value).startswith("line 2: ")
    assert str(from_bytes.value) == str(from_path.value)


def test_read_jsonl_bytes_refuses_invalid_utf8():
    with pytest.raises(ValueError, match="dataset is not valid UTF-8"):
        list(read_jsonl_bytes(b'{"a": 1}\n\xff\n'))


@pytest.mark.parametrize("bom", [b"", codecs.BOM_UTF8])
def test_read_jsonl_bytes_names_the_absolute_offset_of_invalid_utf8(bom):
    # Far past the incremental decoder's first chunk, so a chunk-relative position would differ.
    body = b'{"instruction": "a", "output": "b"}\n' * 1000
    content = bom + body + b"\xff\n"
    offset = len(bom) + len(body)
    assert content[offset] == 0xFF

    with pytest.raises(ValueError) as refused:
        list(read_jsonl_bytes(content))

    message = str(refused.value)
    assert message == f"dataset is not valid UTF-8 at byte offset {offset}: invalid start byte"
    assert message.isascii()


# --- the adapter SFT trainer shares the same comparison ---------------------------------------------


def _sealed_train_config(dataset: Path, digest: str | None):
    from corpus_studio.training.trainer import TrainRunConfig

    return TrainRunConfig(
        base_model="model",
        dataset_path=str(dataset),
        dataset_sha256=digest,
        execution_configuration_hash="a" * 64,
        model_revision="b" * 40,
        package_versions={},
        attn_implementation="sdpa",
        attention_kernel="torch_sdpa_math",
        flash_sdp_enabled=False,
        mem_efficient_sdp_enabled=False,
        math_sdp_enabled=True,
        device_map={"": "cuda:0"},
    )


def test_sealed_runtime_delegates_dataset_verification_to_the_shared_helper(tmp_path, monkeypatch):
    from corpus_studio.training.trainer import TrainerError, verify_sealed_runtime

    data = tmp_path / "train.jsonl"
    _write_rows(data, _ROWS)
    calls: list[str] = []
    real = sealed_inputs.verify_sealed_dataset_bytes

    def _spy(location, expected_sha256, *, progress_callback=None):
        calls.append(location)
        return real(location, expected_sha256, progress_callback=progress_callback)

    monkeypatch.setattr(sealed_inputs, "verify_sealed_dataset_bytes", _spy)

    sealed = _sealed_train_config(data, stable_file_sha256(data))
    assert verify_sealed_runtime(sealed) == data.read_bytes()
    assert calls == [str(data)]
    with pytest.raises(TrainerError, match="omitted the dataset content digest"):
        verify_sealed_runtime(_sealed_train_config(data, None))
    _write_rows(data, [{"instruction": "TAMPERED", "output": "o"}])
    with pytest.raises(TrainerError, match="dataset bytes changed after the execution configuration"):
        verify_sealed_runtime(sealed)
