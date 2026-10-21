"""Cross-platform record-terminator bytes for every JSONL/JSON writer in the record path.

Every site here appends or writes newline-terminated records in **text** mode. Without
``newline=""`` Python's text layer rewrites each ``"\\n"`` to the platform's native sequence, so on
Windows every record terminator becomes CRLF. The cost is not a parse failure (the readers are
deliberately CRLF-tolerant) but two honesty problems:

* a Windows run's record bytes and raw-record sha256s can never byte-match a Linux run's for
  identical logical content, and
* ``row_store.store_line``'s docstring claims to return "the exact on-disk line", which is false
  when the writer rewrites the terminator underneath it.

These assertions are platform-independent: they demand LF on Linux *and* on Windows, so they fail on
the platform that regresses rather than quietly passing on the other one. The tolerance tests at the
bottom pin the other half of the contract: records already written with CRLF must keep parsing.

Deterministic: no GPU, no torch, synthetic probes and injected clocks only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from corpus_studio.platform import checkpoint as ck
from corpus_studio.platform.contracts import (
    CheckpointFileEntry,
    CheckpointManifest,
    SealedTrainingState,
)
from corpus_studio.platform.runners import demo_training_plan
from corpus_studio.platform.subprocess_supervisor import execute_run_subprocess
from corpus_studio.platform.supervisor import (
    RUN_EVENTS_FILENAME,
    EchoRunner,
    demo_run_plan,
    execute_run,
    run_record_directory,
)
from corpus_studio.platform.telemetry import (
    SampleReading,
    TelemetrySampler,
    append_jsonl,
    load_events,
    load_samples,
)
from corpus_studio.platform.worker_protocol import PROTOCOL_VERSION
from corpus_studio.versions.row_store import (
    append_rows,
    load_row_id_set,
    load_rows_by_id,
    row_id,
    row_store_path,
    store_line,
)
from corpus_studio.versions.version_registry import capture_dataset

_CR = b"\r"


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def _assert_lf_only(
    path: Path, *, expected_lines: int | None = None, newline_terminated: bool = True
) -> bytes:
    """The whole point: no CR anywhere, and one LF per record, on every platform.

    ``newline_terminated`` is False for the one writer here that emits a single pretty-printed JSON
    document rather than a JSONL stream (the checkpoint manifest ends at its closing brace)."""

    data = path.read_bytes()
    assert _CR not in data, (
        f"{path.name} contains {data.count(_CR)} CR byte(s); record terminators must be LF on "
        "every platform, Windows included"
    )
    if newline_terminated:
        assert data.endswith(b"\n"), f"{path.name} is not newline-terminated"
    if expected_lines is not None:
        assert data.count(b"\n") == expected_lines
    return data


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _sampler(record_dir: Path) -> TelemetrySampler:
    return TelemetrySampler(
        "run-newline",
        record_dir,
        probe=lambda: SampleReading(unavailable=["probe:synthetic"]),
        interval_ms=50,
        monotonic_ns=lambda: 0,
        clock=lambda: "2026-07-15T00:00:00+00:00",
    )


def _hello_body(plan: object) -> dict[str, object]:
    from corpus_studio.platform.backends import get_worker_backend

    backend = get_worker_backend(plan.backend_ref.id)  # type: ignore[attr-defined]
    assert backend is not None
    return {
        "worker_id": "fake-worker",
        "backend": backend.model_dump(mode="json"),
        "environment_ref": plan.environment_ref.model_dump(mode="json"),  # type: ignore[attr-defined]
        "environment": None,
    }


def _fake_worker_argv(messages: list[tuple[str, dict[str, object]]], hello: dict[str, object]) -> list[str]:
    """A minimal handshake-capable child: enough real protocol traffic to make the parent write
    RunEvents.jsonl. Inlined (not imported from another test module) so this file stands alone."""

    encoded_messages = json.dumps(messages)
    encoded_hello = json.dumps(hello)
    script = (
        "import sys,json\n"
        "def s(t,b,mid,corr):\n"
        f" e={{'protocol_version':{PROTOCOL_VERSION!r},'message_id':mid,"
        "'direction':'worker_to_core','type':t,'body':b}\n"
        " if corr is not None:e['correlation_id']=corr\n"
        " sys.stdout.write(json.dumps(e)+chr(10));sys.stdout.flush()\n"
        f"s('hello',json.loads({encoded_hello!r}),'hello',None)\n"
        "line=sys.stdin.readline()\n"
        "dispatch=json.loads(line) if line.strip() else {}\n"
        "corr=dispatch.get('message_id')\n"
        f"for i,item in enumerate(json.loads({encoded_messages!r})):\n"
        " s(item[0],item[1],'post-'+str(i),corr)\n"
    )
    return [sys.executable, "-c", script]


def _sealed_checkpoint(directory: Path) -> CheckpointManifest:
    directory.mkdir(parents=True, exist_ok=True)

    def entry(name: str, role: str, data: bytes) -> CheckpointFileEntry:
        (directory / name).write_bytes(data)
        return CheckpointFileEntry(
            path=name, role=role, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
        )

    files = sorted(
        [entry("optimizer.pt", "optimizer", b"opt"), entry("rng.pt", "rng", b"rng")],
        key=lambda e: e.path,
    )
    return ck.seal_checkpoint_manifest(
        CheckpointManifest(
            checkpoint_id="ckpt-000001",
            checkpoint_manifest_hash="0" * 64,
            source_run_id="run-parent01",
            created_at="2026-07-15T00:00:00+00:00",
            bound=ck.bound_identities_from_plan(demo_training_plan(plan_id="demo-newline")),
            state=SealedTrainingState(
                scheduler_captured=True, scaler_captured=False, rng_captured=True,
                sampler_state_captured=True, rng_algorithm="philox", epoch=0.5,
                global_optimizer_step=6, microstep_within_step=0,
                gradient_accumulation_steps=4, consumed_microsteps=24,
            ),
            files=files,
        )
    )


# --------------------------------------------------------------------------------------------------
# Every writer emits LF, on every platform
# --------------------------------------------------------------------------------------------------
def test_row_store_append_writes_lf_terminators(tmp_path: Path) -> None:
    rows = [{"text": f"row {i}"} for i in range(3)]
    written = append_rows(tmp_path, [(row_id(r), r) for r in rows], set())
    assert written == 3
    _assert_lf_only(row_store_path(tmp_path), expected_lines=3)


def test_store_line_is_byte_exact_on_disk(tmp_path: Path) -> None:
    """``store_line``'s docstring claims to return the exact on-disk line. Hold it to that: the
    file must be the concatenation of those strings' UTF-8 bytes, with nothing rewritten."""

    rows = [{"text": "ascii"}, {"text": "non-ascii \u00e9\u4e2d"}, {"nested": {"b": 1, "a": [2]}}]
    pairs = [(row_id(r), r) for r in rows]
    append_rows(tmp_path, pairs, set())
    expected = "".join(store_line(rid, row) for rid, row in pairs).encode("utf-8")
    assert row_store_path(tmp_path).read_bytes() == expected


def test_capture_dataset_store_append_writes_lf_terminators(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_lf(examples, _jsonl([{"text": f"cap {i}"} for i in range(4)]))
    capture = capture_dataset(examples, tmp_path, store_rows=True)
    assert capture.rows_stored is True
    assert capture.row_count == 4
    _assert_lf_only(row_store_path(tmp_path), expected_lines=4)


def test_telemetry_sampler_writes_lf_terminators(tmp_path: Path) -> None:
    sampler = _sampler(tmp_path / "record")
    try:
        sampler.take_sample()
        sampler.take_sample()
    finally:
        sampler.stop()
    data = _assert_lf_only(sampler.samples_path, expected_lines=2)
    # Still parseable, and the digested line count matches the LF count.
    assert len(load_samples(sampler.samples_path)) == 2
    assert data.count(b"\n") == 2


def test_telemetry_append_jsonl_writes_lf_terminators(tmp_path: Path) -> None:
    target = tmp_path / "record" / "Lines.jsonl"
    append_jsonl(target, json.dumps({"a": 1}))
    append_jsonl(target, json.dumps({"a": 2}))
    data = _assert_lf_only(target, expected_lines=2)
    assert data == b'{"a": 1}\n{"a": 2}\n'


def test_supervisor_run_events_written_with_lf_terminators(tmp_path: Path) -> None:
    result = execute_run(
        demo_run_plan(), runner=EchoRunner(), run_id="run-newline-sup", out_dir=tmp_path
    )
    assert result.manifest.state == "succeeded"
    events_path = run_record_directory(tmp_path, "run-newline-sup") / RUN_EVENTS_FILENAME
    data = _assert_lf_only(events_path, expected_lines=len(result.events))
    assert len(load_events(events_path)) == len(result.events)
    assert data.count(b"\n") == len(result.events)


def test_subprocess_supervisor_run_events_written_with_lf_terminators(tmp_path: Path) -> None:
    plan = demo_run_plan()
    messages: list[tuple[str, dict[str, object]]] = [
        ("run_accepted", {"run_id": plan.plan_id, "pid": 1}),
        (
            "event",
            {
                "contract_version": "1.0.0",
                "event_type": "stage",
                "run_id": plan.plan_id,
                "seq": 0,
                "emitted_at": "2026-07-15T00:00:00+00:00",
                "stage": "process_start",
                "message": "newline probe",
            },
        ),
    ]
    execute_run_subprocess(
        plan,
        run_id=plan.plan_id,
        worker_argv=_fake_worker_argv(messages, _hello_body(plan)),
        out_dir=tmp_path,
        silence_timeout_s=30,
    )
    events_path = run_record_directory(tmp_path, plan.plan_id) / RUN_EVENTS_FILENAME
    data = _assert_lf_only(events_path)
    assert data.count(b"\n") >= 1
    assert len(load_events(events_path)) == data.count(b"\n")


def test_checkpoint_manifest_written_with_lf_line_breaks(tmp_path: Path) -> None:
    sealed = _sealed_checkpoint(tmp_path / "ckpt")
    written = ck.write_checkpoint_manifest(sealed, tmp_path / "ckpt")
    data = _assert_lf_only(written, newline_terminated=False)
    # Byte-identical to the serialization the manifest hash was computed over.
    assert data == sealed.model_dump_json(indent=2).encode("utf-8")
    assert ck.load_checkpoint_manifest(written) == sealed


# --------------------------------------------------------------------------------------------------
# The other half of the contract: records already written with CRLF must keep parsing
# --------------------------------------------------------------------------------------------------
def test_row_store_reader_still_tolerates_crlf_records(tmp_path: Path) -> None:
    rows = [{"text": f"legacy {i}"} for i in range(3)]
    pairs = [(row_id(r), r) for r in rows]
    path = row_store_path(tmp_path)
    _write_lf(path, "".join(store_line(rid, row).replace("\n", "\r\n") for rid, row in pairs))
    assert b"\r\n" in path.read_bytes()
    ids = {rid for rid, _ in pairs}
    assert load_row_id_set(tmp_path) == ids
    assert load_rows_by_id(tmp_path, ids) == {rid: row for rid, row in pairs}


def test_appending_lf_after_a_crlf_store_keeps_every_record_readable(tmp_path: Path) -> None:
    """A store written by an older Windows build ends up mixed-ending once LF appends land on it.
    Both halves must stay readable: the fix must not trade a cosmetic win for lost rows."""

    legacy = [{"text": f"legacy {i}"} for i in range(2)]
    legacy_pairs = [(row_id(r), r) for r in legacy]
    path = row_store_path(tmp_path)
    _write_lf(
        path, "".join(store_line(rid, row).replace("\n", "\r\n") for rid, row in legacy_pairs)
    )

    fresh = [{"text": f"fresh {i}"} for i in range(2)]
    fresh_pairs = [(row_id(r), r) for r in fresh]
    assert append_rows(tmp_path, fresh_pairs, load_row_id_set(tmp_path)) == 2

    data = path.read_bytes()
    assert data.count(b"\r\n") == 2, "legacy CRLF records must be left untouched"
    assert data.count(b"\n") == 4
    every = dict(legacy_pairs) | dict(fresh_pairs)
    assert load_row_id_set(tmp_path) == set(every)
    assert load_rows_by_id(tmp_path, set(every)) == every


def test_telemetry_loaders_still_tolerate_crlf_records(tmp_path: Path) -> None:
    sampler = _sampler(tmp_path / "record")
    try:
        sampler.take_sample()
    finally:
        sampler.stop()
    lf_bytes = sampler.samples_path.read_bytes()
    crlf_path = tmp_path / "legacy" / "TelemetrySamples.jsonl"
    crlf_path.parent.mkdir(parents=True, exist_ok=True)
    crlf_path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    assert load_samples(crlf_path) == load_samples(sampler.samples_path)
