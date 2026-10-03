"""Row-store GC is serialized with version capture/publication (#859).

The invariant under test: a version published while GC runs keeps every row its manifest names,
because capture -> manifest -> record and GC's scan -> replace are mutually exclusive under the
cross-process version-store lock. The lock is same-thread reentrant, so every "concurrent" actor
here runs on a separate thread or in a separate process (a same-thread nesting would re-enter the
lock rather than wait for it). Children are spawned with ``sys.executable``, never forked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from typer.testing import CliRunner

import corpus_studio.versions.gc as gc_mod
import corpus_studio.versions.row_store as rs
import corpus_studio.versions.version_registry as vr
from corpus_studio.cli import app
from corpus_studio.storage import examples_writer
from corpus_studio.storage.examples_writer import (
    ExamplesLockedError,
    read_existing_lines,
    single_writer_lock,
)
from corpus_studio.storage.file_lock import FileLockTimeoutError
from corpus_studio.versions import store_lock
from corpus_studio.versions.gc import (
    IncompleteReferenceScanError,
    RowStoreGcRefusedError,
    collect_referenced_row_ids,
    gc_row_store,
)
from corpus_studio.versions.row_store import (
    append_rows,
    load_row_id_set,
    row_id,
    row_store_path,
    store_line,
    terminate_torn_tail,
)
from corpus_studio.versions.store_lock import (
    VERSION_STORE_LOCK_FILENAME,
    VersionStoreBusyError,
    VersionStoreLockError,
    version_store_lock,
)
from corpus_studio.versions.version_registry import (
    _truncate_row_store,
    capture_dataset,
    create_dataset_version,
    fingerprint_dataset,
    list_version_records,
    load_row_manifest,
    manifest_path,
    publish_dataset_version,
    record_path,
    registry_dir,
    save_row_manifest,
)
from corpus_studio.versions.version_restore import reconstruct_version_lines

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX lock-file semantics")

runner = CliRunner()

ROW_A = {"instruction": "a", "output": "1"}
ROW_B = {"instruction": "b", "output": "2"}
ROW_C = {"instruction": "c", "output": "3"}


# --- helpers ----------------------------------------------------------------------------------


def _write_examples(project: Path, rows: list[dict[str, Any]]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _project(project: Path) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": "instruction"}), encoding="utf-8"
    )
    return project


def _append_orphan(project: Path, row: dict[str, Any]) -> None:
    with row_store_path(project).open("a", encoding="utf-8") as handle:
        handle.write(store_line(row_id(row), row))


def _short_lock_timeout(monkeypatch: pytest.MonkeyPatch, seconds: float = 0.05) -> None:
    monkeypatch.setattr(store_lock, "DEFAULT_VERSION_STORE_LOCK_TIMEOUT_SECONDS", seconds)


def _probe(project: Path) -> str:
    """Try the version-store lock from ANOTHER thread: 'acquired' or 'busy'."""

    outcome: dict[str, str] = {}

    def run() -> None:
        try:
            with version_store_lock(project, operation="probe", timeout_seconds=0.05):
                outcome["result"] = "acquired"
        except VersionStoreBusyError:
            outcome["result"] = "busy"

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(10)
    return outcome["result"]


class _Worker(threading.Thread):
    """Runs ``target`` on its own thread and keeps its result or exception for the test thread."""

    def __init__(self, target: Callable[[], Any]) -> None:
        super().__init__(daemon=True)
        self._target_fn = target
        self.result: Any = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self._target_fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the asserting thread
            self.error = exc


@contextmanager
def _held_by_another_thread(project: Path) -> Iterator[None]:
    acquired, release = threading.Event(), threading.Event()

    def hold() -> None:
        with version_store_lock(project, operation="test holder"):
            acquired.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert acquired.wait(10)
    try:
        yield
    finally:
        release.set()
        holder.join(10)


def _assert_every_manifest_row_is_stored(project: Path) -> None:
    stored = load_row_id_set(project)
    for manifest in registry_dir(project).glob("*.rows"):
        for line in manifest.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                assert line.strip() in stored, f"{manifest.name} lost row {line.strip()}"


def _wait_for(path: Path, child: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        assert child.poll() is None, child.communicate()
        assert time.monotonic() < deadline, "child never reached the barrier"
        time.sleep(0.01)


# --- the lock itself --------------------------------------------------------------------------


def test_lock_is_reentrant_bounded_and_invisible_to_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lock_timeout(monkeypatch)
    with version_store_lock(tmp_path, operation="outer"):
        with version_store_lock(tmp_path, operation="inner"):  # same thread re-enters
            assert _probe(tmp_path) == "busy"
    assert _probe(tmp_path) == "acquired"
    assert (registry_dir(tmp_path) / VERSION_STORE_LOCK_FILENAME).is_file()
    # The lock file matches neither the record (*.json) nor the manifest (*.rows) globs.
    assert list_version_records(tmp_path) == []
    assert collect_referenced_row_ids(tmp_path) == set()


def test_busy_message_is_ascii_and_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lock_timeout(monkeypatch)
    with _held_by_another_thread(tmp_path):
        with pytest.raises(VersionStoreBusyError) as caught:
            with version_store_lock(tmp_path, operation="row-store GC"):
                pass
    message = str(caught.value)
    assert message.isascii()
    assert "busy" in message and "row-store GC" in message and "do not delete" in message


def test_unavailable_lock_file_is_a_lock_error_not_busy(tmp_path: Path) -> None:
    (registry_dir(tmp_path) / VERSION_STORE_LOCK_FILENAME).mkdir(parents=True)
    with pytest.raises(VersionStoreLockError, match="unavailable") as caught:
        with version_store_lock(tmp_path, operation="dataset capture"):
            pass
    assert not isinstance(caught.value, VersionStoreBusyError)


def test_only_acquisition_failures_are_translated(tmp_path: Path) -> None:
    with pytest.raises(FileLockTimeoutError):
        with version_store_lock(tmp_path, operation="body"):
            raise FileLockTimeoutError("raised by the body, not by acquisition")


# --- GC vs capture/publication (in-process, separate threads) ---------------------------------


def test_capture_started_inside_gc_window_waits_and_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's interleaving: GC takes its reference snapshot, then version B is captured and
    # published before GC reads and replaces the store. B must wait, then survive intact.
    _write_examples(tmp_path, [ROW_A])
    version_a = create_dataset_version(tmp_path, label="A")
    real_collect = gc_mod.collect_referenced_row_ids
    workers: list[_Worker] = []

    def snapshot_then_capture(project_dir: Path | str) -> set[str]:
        snapshot = real_collect(project_dir)
        _write_examples(tmp_path, [ROW_A, ROW_B])
        worker = _Worker(lambda: create_dataset_version(tmp_path, label="B"))
        worker.start()
        workers.append(worker)
        worker.join(0.3)
        assert worker.is_alive(), "capture was not excluded by the GC in progress"
        return snapshot

    monkeypatch.setattr(gc_mod, "collect_referenced_row_ids", snapshot_then_capture)
    result = gc_row_store(tmp_path)
    workers[0].join(30)

    assert workers[0].error is None
    version_b = workers[0].result
    assert result.pruned_rows == 0
    assert reconstruct_version_lines(tmp_path, version_a.version_id)
    assert len(reconstruct_version_lines(tmp_path, version_b.version_id)) == 2
    _assert_every_manifest_row_is_stored(tmp_path)


def test_capture_refused_while_gc_holds_the_store_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    real_collect = gc_mod.collect_referenced_row_ids
    workers: list[_Worker] = []

    def snapshot_then_refused_capture(project_dir: Path | str) -> set[str]:
        snapshot = real_collect(project_dir)
        _write_examples(tmp_path, [ROW_A, ROW_B])
        _short_lock_timeout(monkeypatch)
        worker = _Worker(lambda: create_dataset_version(tmp_path, label="B"))
        worker.start()
        worker.join(30)
        workers.append(worker)
        return snapshot

    monkeypatch.setattr(gc_mod, "collect_referenced_row_ids", snapshot_then_refused_capture)
    gc_row_store(tmp_path)

    assert isinstance(workers[0].error, VersionStoreBusyError)
    assert len(list_version_records(tmp_path)) == 1  # only A
    assert len(list(registry_dir(tmp_path).glob("*.rows"))) == 1
    assert row_id(ROW_B) not in load_row_id_set(tmp_path)


def test_gc_waits_for_an_in_flight_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    _append_orphan(tmp_path, ROW_C)  # a genuine orphan GC should prune
    _write_examples(tmp_path, [ROW_A, ROW_B])
    real_read = vr.read_jsonl
    workers: list[_Worker] = []

    def read_then_start_gc(path: Path | str) -> Iterator[Any]:
        for index, row in enumerate(real_read(path)):
            yield row
            if index == 1 and not workers:  # ROW_B is in the store, its manifest is not yet
                worker = _Worker(lambda: gc_row_store(tmp_path))
                worker.start()
                workers.append(worker)
                worker.join(0.3)
                assert worker.is_alive(), "GC was not excluded by the capture in progress"

    monkeypatch.setattr(vr, "read_jsonl", read_then_start_gc)
    version_b = create_dataset_version(tmp_path, label="B")
    workers[0].join(30)

    assert workers[0].error is None
    assert workers[0].result.pruned_rows == 1  # only the orphan ROW_C
    assert len(reconstruct_version_lines(tmp_path, version_b.version_id)) == 2
    _assert_every_manifest_row_is_stored(tmp_path)


def test_capture_rollback_does_not_erase_a_concurrent_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failing capture rolls the store back to ITS start size; a capture that appended in between
    # would lose its rows. Under the lock the second capture waits for the rollback instead.
    _write_examples(tmp_path, [ROW_A])
    base = create_dataset_version(tmp_path, label="base")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(ROW_C) + "\n{ torn\n", encoding="utf-8")
    real_read = vr.read_jsonl
    workers: list[_Worker] = []

    def read_then_start_capture(path: Path | str) -> Iterator[Any]:
        for index, row in enumerate(real_read(path)):
            yield row
            if Path(path) == bad and index == 0:
                _write_examples(tmp_path, [ROW_A, ROW_B])
                worker = _Worker(lambda: create_dataset_version(tmp_path, label="B"))
                worker.start()
                workers.append(worker)
                worker.join(0.3)
                assert worker.is_alive()

    monkeypatch.setattr(vr, "read_jsonl", read_then_start_capture)
    failed = capture_dataset(bad, tmp_path, store_rows=True)
    workers[0].join(30)

    assert failed.content_fingerprint is None  # rolled back
    assert workers[0].error is None
    assert reconstruct_version_lines(tmp_path, base.version_id)
    assert len(reconstruct_version_lines(tmp_path, workers[0].result.version_id)) == 2
    assert row_id(ROW_C) not in load_row_id_set(tmp_path)


def test_capture_and_publication_hold_the_lock_but_fingerprint_only_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    reads: list[str] = []
    records: dict[str, str] = {}
    real_read = vr.read_jsonl
    real_save_record = vr.save_version_record

    def probing_read(path: Path | str) -> Iterator[Any]:
        reads.append(_probe(tmp_path))
        yield from real_read(path)

    def probing_save_record(project_dir: Path | str, record: Any) -> Path:
        records[record.label] = _probe(tmp_path)
        return real_save_record(project_dir, record)

    monkeypatch.setattr(vr, "read_jsonl", probing_read)
    monkeypatch.setattr(vr, "save_version_record", probing_save_record)

    capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=True)
    capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=False)
    assert reads == ["busy", "acquired"]

    publish_dataset_version(tmp_path, label="stored")
    publish_dataset_version(tmp_path, label="fingerprint-only", store_rows=False)
    assert records == {"stored": "busy", "fingerprint-only": "acquired"}


def test_version_store_paths_never_request_the_examples_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lock order is writer lock -> version-store lock, never the reverse. Structural proof: every
    # version-store operation completes with the writer lock made unusable.
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the examples writer lock was requested under the version store")

    monkeypatch.setattr(examples_writer, "single_writer_lock", forbidden)
    _write_examples(tmp_path, [ROW_A])
    capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=True)
    create_dataset_version(tmp_path)
    publish_dataset_version(tmp_path, label="x")
    append_rows(tmp_path, [(row_id(ROW_B), ROW_B)], set())
    save_row_manifest(tmp_path, "manual", [row_id(ROW_A)])
    gc_row_store(tmp_path)
    gc_row_store(tmp_path, dry_run=True)


# --- two real processes ----------------------------------------------------------------------

_GC_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    import corpus_studio.versions.gc as gc_mod

    project, barrier = Path(sys.argv[1]), Path(sys.argv[2])
    real_collect = gc_mod.collect_referenced_row_ids

    def paused_after_snapshot(project_dir):
        snapshot = real_collect(project_dir)
        (barrier / "snapshot_taken").write_text("1")
        deadline = time.monotonic() + 60
        while not (barrier / "go").exists():
            if time.monotonic() > deadline:
                raise SystemExit("barrier timeout")
            time.sleep(0.01)
        return snapshot

    gc_mod.collect_referenced_row_ids = paused_after_snapshot
    from corpus_studio.cli import app
    app(["dataset-version-gc", str(project), "--json"])
    """
)

_CLI_CHILD = textwrap.dedent(
    """
    import sys
    import corpus_studio.versions.store_lock as store_lock

    store_lock.DEFAULT_VERSION_STORE_LOCK_TIMEOUT_SECONDS = float(sys.argv[1])
    from corpus_studio.cli import app
    app(sys.argv[2:])
    """
)


def _cli_child(timeout: float, args: list[str]) -> list[str]:
    return [sys.executable, "-c", _CLI_CHILD, str(timeout), *args]


@pytest.mark.parametrize("writer", ["dataset-version-create", "import-commit"])
def test_two_process_capture_vs_gc_preserves_both_versions(tmp_path: Path, writer: str) -> None:
    project = _project(tmp_path / "proj")
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _write_examples(project, [ROW_A])
    version_a = create_dataset_version(project, label="A")
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(ROW_B) + "\n", encoding="utf-8")
    if writer == "dataset-version-create":
        _write_examples(project, [ROW_A, ROW_B])
        args = ["dataset-version-create", str(project), "--label", "B"]
    else:  # nests the examples writer lock -> version-store lock inside the child
        args = ["import-commit", str(project), "--from", str(staging), "--json"]
    examples_before = (project / "examples.jsonl").read_bytes()

    gc_child = subprocess.Popen(
        [sys.executable, "-c", _GC_CHILD, str(project), str(barrier)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    patient: subprocess.Popen[str] | None = None
    try:
        _wait_for(barrier / "snapshot_taken", gc_child)
        # GC holds the store: a capture with a short wait is refused and publishes nothing.
        refused = subprocess.run(
            _cli_child(0.2, args), capture_output=True, text=True, timeout=60
        )
        assert refused.returncode == 1, refused.stdout + refused.stderr
        assert "busy" in refused.stderr and "Traceback" not in refused.stderr
        assert len(list_version_records(project)) == 1
        assert (project / "examples.jsonl").read_bytes() == examples_before
        # A patient capture waits for GC, then publishes.
        patient = subprocess.Popen(
            _cli_child(60, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    finally:
        (barrier / "go").write_text("1")
        gc_out, gc_err = gc_child.communicate(timeout=60)
    assert patient is not None
    patient_out, patient_err = patient.communicate(timeout=60)

    assert gc_child.returncode == 0, gc_err
    assert json.loads(gc_out)["pruned_rows"] == 0
    assert patient.returncode == 0, patient_err
    payload = json.loads(patient_out)
    version_b = payload["version_id"]
    assert reconstruct_version_lines(project, version_a.version_id)
    assert len(reconstruct_version_lines(project, version_b)) == 2
    _assert_every_manifest_row_is_stored(project)


_STORE_HOLDER_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from corpus_studio.versions.store_lock import version_store_lock

    project, ready, release = (Path(arg) for arg in sys.argv[1:4])
    with version_store_lock(project, operation="test holder"):
        ready.write_text("1")
        deadline = time.monotonic() + 60
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    """
)


_SIGNALLING_CLI_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import corpus_studio.versions.store_lock as store_lock

    signal = Path(sys.argv[1])
    real_lock = store_lock.version_store_lock

    def signalling_lock(project_dir, **kwargs):
        signal.write_text("1")  # about to request the version-store lock
        return real_lock(project_dir, **kwargs)

    store_lock.version_store_lock = signalling_lock
    from corpus_studio.cli import app
    app(sys.argv[2:])
    """
)


def test_writer_lock_then_version_store_lock_nests_across_processes_without_deadlock(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(ROW_B) + "\n", encoding="utf-8")
    ready, release, requesting = tmp_path / "ready", tmp_path / "release", tmp_path / "requesting"

    holder = subprocess.Popen(
        [sys.executable, "-c", _STORE_HOLDER_CHILD, str(project), str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    committer: subprocess.Popen[str] | None = None
    try:
        _wait_for(ready, holder)
        committer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _SIGNALLING_CLI_CHILD,
                str(requesting),
                "import-commit",
                str(project),
                "--from",
                str(staging),
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for(requesting, committer)
        # The committer now holds the writer lock (L1) and waits for the version store (L2, held
        # by the other process): the nested state a lock-order deadlock would need.
        with pytest.raises(ExamplesLockedError):
            with single_writer_lock(project):
                pass
        time.sleep(0.2)
        assert committer.poll() is None  # still waiting on L2, not failed
    finally:
        release.write_text("1")
        holder.communicate(timeout=60)
    assert committer is not None
    out, err = committer.communicate(timeout=60)

    assert holder.returncode == 0
    assert committer.returncode == 0, err
    payload = json.loads(out)
    assert payload["committed"] == 1 and payload["version_id"]
    assert len(reconstruct_version_lines(project, payload["version_id"])) == 2


_CRASHING_CAPTURE_CHILD = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import corpus_studio.versions.version_registry as vr

    def die_before_the_manifest(*_args, **_kwargs):
        os._exit(9)  # the rows are appended and fsynced; the manifest never appears

    vr.save_row_manifest = die_before_the_manifest
    vr.create_dataset_version(Path(sys.argv[1]), label="crashed")
    """
)


def test_crashed_capture_releases_the_lock_and_leaves_a_recoverable_store(tmp_path: Path) -> None:
    _write_examples(tmp_path, [ROW_A])
    version_a = create_dataset_version(tmp_path, label="A")
    _write_examples(tmp_path, [ROW_A, ROW_B])

    crashed = subprocess.run(
        [sys.executable, "-c", _CRASHING_CAPTURE_CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert crashed.returncode == 9
    assert row_id(ROW_B) in load_row_id_set(tmp_path)  # the orphan row the crash left behind
    with version_store_lock(tmp_path, operation="probe", timeout_seconds=0.5):
        pass  # the kernel released the dead holder's lock; nothing to delete
    assert len(list_version_records(tmp_path)) == 1
    result = gc_row_store(tmp_path)
    assert result.pruned_rows == 1
    assert reconstruct_version_lines(tmp_path, version_a.version_id)


# --- interrupted states and fail-closed rollback ----------------------------------------------


def test_failed_manifest_publication_releases_the_lock_and_gc_prunes_only_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    version_a = create_dataset_version(tmp_path, label="A")
    _write_examples(tmp_path, [ROW_A, ROW_B])

    def disk_full(*_args: Any, **_kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(vr, "save_row_manifest", disk_full)
    with pytest.raises(OSError, match="disk full"):
        create_dataset_version(tmp_path, label="B")
    monkeypatch.undo()

    assert _probe(tmp_path) == "acquired"
    assert len(list_version_records(tmp_path)) == 1
    result = gc_row_store(tmp_path)
    assert result.pruned_rows == 1 and result.kept_rows == 1
    assert reconstruct_version_lines(tmp_path, version_a.version_id)


def test_record_less_manifest_keeps_its_rows_pinned(tmp_path: Path) -> None:
    # A publication interrupted after its manifest but before its record: GC keeps those rows.
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    _append_orphan(tmp_path, ROW_B)
    _append_orphan(tmp_path, ROW_C)
    save_row_manifest(tmp_path, "20260101T000000-interrupted", [row_id(ROW_B)])

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1  # only ROW_C
    assert load_row_id_set(tmp_path) == {row_id(ROW_A), row_id(ROW_B)}
    with pytest.raises(FileNotFoundError):
        reconstruct_version_lines(tmp_path, "20260101T000000-interrupted")


def test_torn_store_tail_is_terminated_before_the_next_capture(tmp_path: Path) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    with row_store_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"row": {"instruction": "half')  # a writer died mid-append
    _write_examples(tmp_path, [ROW_A, ROW_B])

    version_b = create_dataset_version(tmp_path, label="B")

    assert len(reconstruct_version_lines(tmp_path, version_b.version_id)) == 2
    lines = row_store_path(tmp_path).read_text(encoding="utf-8").split("\n")
    assert '{"row": {"instruction": "half' in lines  # the fragment is its own line
    gc_row_store(tmp_path)
    assert '{"row": {"instruction": "half' in row_store_path(tmp_path).read_text(encoding="utf-8")
    assert len(reconstruct_version_lines(tmp_path, version_b.version_id)) == 2


def test_append_rows_terminates_a_torn_tail(tmp_path: Path) -> None:
    path = row_store_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"row_id": "torn', encoding="utf-8")
    assert append_rows(tmp_path, [(row_id(ROW_A), ROW_A)], set()) == 1
    assert load_row_id_set(tmp_path) == {row_id(ROW_A)}


def test_terminate_torn_tail_leaves_complete_or_absent_stores_alone(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    assert terminate_torn_tail(path) is False  # absent
    path.write_bytes(b"")
    assert terminate_torn_tail(path) is False  # empty
    path.write_bytes(b"{}\n")
    assert terminate_torn_tail(path) is False  # complete
    path.write_bytes(b"{}\n{")
    assert terminate_torn_tail(path) is True
    assert path.read_bytes() == b"{}\n{\n"


def test_unrepairable_torn_tail_stores_nothing_but_keeps_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    before = row_store_path(tmp_path).read_bytes()
    _write_examples(tmp_path, [ROW_A, ROW_B])

    def unwritable(_path: Path) -> bool:
        raise OSError("read-only store")

    monkeypatch.setattr(rs, "terminate_torn_tail", unwritable)
    capture = capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=True)

    assert capture.rows_stored is False
    assert capture.content_fingerprint == fingerprint_dataset(tmp_path / "examples.jsonl")[0]
    assert row_store_path(tmp_path).read_bytes() == before


def test_rollback_never_extends_the_store(tmp_path: Path) -> None:
    store = row_store_path(tmp_path)
    store.parent.mkdir(parents=True)
    store.write_bytes(b"row\n")
    _truncate_row_store(store, 100)  # a stale, larger start size must not NUL-pad the store
    assert store.read_bytes() == b"row\n"
    _truncate_row_store(tmp_path / "absent.jsonl", 0)  # a vanished store is not an error


# --- GC refuses an incomplete reference scan --------------------------------------------------


def _store_with_version(tmp_path: Path) -> tuple[str, bytes]:
    _write_examples(tmp_path, [ROW_A, ROW_B])
    version = create_dataset_version(tmp_path, label="A+B")
    _append_orphan(tmp_path, ROW_C)
    return version.version_id, row_store_path(tmp_path).read_bytes()


def test_undecodable_manifest_refuses_gc(tmp_path: Path) -> None:
    _version_id, before = _store_with_version(tmp_path)
    (registry_dir(tmp_path) / "garbled.rows").write_bytes(b"\xff\xfe")

    with pytest.raises(IncompleteReferenceScanError, match="not valid UTF-8"):
        gc_row_store(tmp_path)
    assert row_store_path(tmp_path).read_bytes() == before

    result = runner.invoke(app, ["dataset-version-gc", str(tmp_path)])
    assert result.exit_code == 1
    assert "GC aborted" in result.output and "nothing was pruned" in result.output
    assert row_store_path(tmp_path).read_bytes() == before


def test_torn_manifest_line_refuses_gc(tmp_path: Path) -> None:
    version_id, before = _store_with_version(tmp_path)
    manifest = manifest_path(tmp_path, version_id)
    ids = manifest.read_text(encoding="utf-8").split()
    manifest.write_text(ids[0] + "\n" + ids[1][:20] + "\n", encoding="utf-8")

    with pytest.raises(IncompleteReferenceScanError, match="line 2 is not a sha256 row id"):
        gc_row_store(tmp_path)
    assert row_store_path(tmp_path).read_bytes() == before


def test_manifest_shorter_than_its_record_refuses_gc(tmp_path: Path) -> None:
    version_id, before = _store_with_version(tmp_path)
    manifest = manifest_path(tmp_path, version_id)
    manifest.write_text(manifest.read_text(encoding="utf-8").split()[0] + "\n", encoding="utf-8")

    with pytest.raises(IncompleteReferenceScanError, match="lists 1 row id.* declares 2"):
        gc_row_store(tmp_path)
    assert row_store_path(tmp_path).read_bytes() == before


def test_unreadable_record_keeps_its_manifest_live(tmp_path: Path) -> None:
    version_id, _before = _store_with_version(tmp_path)
    record_path(tmp_path, version_id).write_text("{ not a record", encoding="utf-8")

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1  # only the orphan ROW_C
    assert load_row_id_set(tmp_path) == {row_id(ROW_A), row_id(ROW_B)}
    assert load_row_manifest(tmp_path, version_id) == [row_id(ROW_A), row_id(ROW_B)]


def test_undecodable_store_line_is_kept_byte_for_byte_not_a_refusal(tmp_path: Path) -> None:
    version_id, _before = _store_with_version(tmp_path)
    with row_store_path(tmp_path).open("ab") as handle:
        handle.write(b"\xff\xfe\n")

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1  # only the orphan ROW_C
    assert b"\xff\xfe" in row_store_path(tmp_path).read_bytes().split(b"\n")
    assert reconstruct_version_lines(tmp_path, version_id)


def test_gc_replace_uses_a_unique_temp_and_refuses_cleanly_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _version_id, before = _store_with_version(tmp_path)
    sources: list[str] = []

    def failing_replace(src: Any, dst: Any) -> None:
        sources.append(Path(src).name)
        raise OSError("the store is open elsewhere")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(RowStoreGcRefusedError, match="could not be replaced"):
        gc_row_store(tmp_path)
    monkeypatch.undo()

    assert len(sources) == 1
    assert sources[0].startswith("row_store.jsonl.") and sources[0].endswith(".tmp")
    assert sources[0] != "row_store.jsonl.tmp"  # never a fixed, shared temp name
    assert row_store_path(tmp_path).read_bytes() == before
    assert list(registry_dir(tmp_path).glob("row_store.jsonl*.tmp")) == []

    assert gc_row_store(tmp_path).pruned_rows == 1  # a retry succeeds


def test_gc_without_a_version_store_creates_nothing(tmp_path: Path) -> None:
    result = gc_row_store(tmp_path)
    assert result.pruned_rows == 0 and result.referenced_row_ids == 0
    assert collect_referenced_row_ids(tmp_path) == set()
    assert not registry_dir(tmp_path).exists()


def test_blank_manifest_lines_and_a_missing_store_are_not_refusals(tmp_path: Path) -> None:
    # Blank lines carry no id (the count cross-check ignores them too); a manifest set with no
    # store yet is a clean no-op, not an incomplete scan.
    manifest = manifest_path(tmp_path, "20260101T000000-blank")
    manifest.parent.mkdir(parents=True)
    manifest.write_text("\n" + row_id(ROW_A) + "\n\n", encoding="utf-8")

    result = gc_row_store(tmp_path)

    assert result.referenced_row_ids == 1 and result.pruned_rows == 0
    assert not row_store_path(tmp_path).exists()


# --- CLI refusals: busy store, nothing changed -------------------------------------------------


def test_cli_gc_refuses_when_the_store_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _version_id, before = _store_with_version(tmp_path)
    _short_lock_timeout(monkeypatch)
    with _held_by_another_thread(tmp_path):
        result = runner.invoke(app, ["dataset-version-gc", str(tmp_path)])
    assert result.exit_code == 1
    assert "GC refused, nothing was pruned" in result.output and "busy" in result.output
    assert row_store_path(tmp_path).read_bytes() == before


def test_cli_create_refuses_when_the_store_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    _short_lock_timeout(monkeypatch)
    with _held_by_another_thread(tmp_path):
        result = runner.invoke(app, ["dataset-version-create", str(tmp_path)])
    assert result.exit_code == 1
    assert "Could not capture a dataset version, nothing was written" in result.output
    assert list_version_records(tmp_path) == []
    assert not row_store_path(tmp_path).exists()


def test_cli_create_checks_the_stamp_run_before_capturing(tmp_path: Path) -> None:
    _write_examples(tmp_path, [ROW_A])
    result = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--stamp-run", "nope"])
    assert result.exit_code == 1
    assert "No training run 'nope'" in result.output
    # No orphan manifest, record, or store rows from a refused create.
    assert not registry_dir(tmp_path).exists()


def test_import_commit_refuses_when_the_store_is_busy_and_commits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    before = (project / "examples.jsonl").read_bytes()
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(ROW_B) + "\n", encoding="utf-8")
    _short_lock_timeout(monkeypatch)
    with _held_by_another_thread(project):
        result = runner.invoke(app, ["import-commit", str(project), "--from", str(staging)])
    assert result.exit_code == 1
    assert "Refusing to commit, nothing committed" in result.output
    assert (project / "examples.jsonl").read_bytes() == before
    assert list_version_records(project) == []

    # --no-version captures nothing, so it does not need (or wait for) the version store.
    with _held_by_another_thread(project):
        result = runner.invoke(
            app, ["import-commit", str(project), "--from", str(staging), "--no-version"]
        )
    assert result.exit_code == 0, result.output
    assert len(read_existing_lines(project)) == 2


def test_import_commit_holds_the_version_store_across_the_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path / "proj")
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(ROW_A) + "\n", encoding="utf-8")
    real_append = examples_writer.append_examples_locked
    seen: list[str] = []

    def probing_append(project_dir: Path | str, rows: list[Any]) -> int:
        seen.append(_probe(project))
        return real_append(project_dir, rows)

    monkeypatch.setattr(examples_writer, "append_examples_locked", probing_append)
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(staging), "--json"])
    assert result.exit_code == 0, result.output
    assert seen == ["busy"]


def _busy_capture(*_args: Any, **_kwargs: Any) -> Any:
    raise VersionStoreBusyError("the dataset version store is busy (test)")


def test_examples_delete_refuses_when_no_undo_can_be_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A, ROW_B])
    before = (project / "examples.jsonl").read_bytes()
    monkeypatch.setattr(
        "corpus_studio.versions.version_registry.create_dataset_version", _busy_capture
    )
    result = runner.invoke(app, ["examples-delete", str(project), "--row", "1"])
    assert result.exit_code == 1
    assert "could not capture an undo version" in result.output
    assert "nothing changed" in result.output
    assert (project / "examples.jsonl").read_bytes() == before


def test_inplace_restore_refuses_when_no_undo_can_be_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    target = create_dataset_version(tmp_path, label="target")
    _write_examples(tmp_path, [ROW_A, ROW_B])
    before = (tmp_path / "examples.jsonl").read_bytes()
    monkeypatch.setattr(
        "corpus_studio.versions.version_registry.create_dataset_version", _busy_capture
    )
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", target.version_id, "--in-place"],
    )
    assert result.exit_code == 1
    assert "Refusing --in-place: could not capture an undo version" in result.output
    assert (tmp_path / "examples.jsonl").read_bytes() == before
