"""The subprocess worker protocol — the parent that can KILL a hung run (the thing the in-process
watchdog can't). The worker is unit-tested in-memory (a StringIO out); the parent is integration-
tested against REAL child processes: an echo roundtrip, a hung child (killed → KERNEL_STALL), and a
crashed child (→ ENVIRONMENT_FAILURE). No torch/GPU needed — the fakes are tiny ``python -c`` scripts
and the echo runner."""

import io
import json
import sys
import time

import pytest

from corpus_studio.platform.subprocess_supervisor import (
    _dispatch_line,
    execute_run_subprocess,
)
from corpus_studio.platform.supervisor import demo_run_plan
from corpus_studio.platform.worker import run_worker

_PLAN = demo_run_plan()


def _worker_out(runner_name: str = "echo") -> list[dict]:
    """Drive run_worker in-process with an in-memory out stream; return the emitted WorkerMessages."""
    out = io.StringIO()
    dispatch = _dispatch_line(_PLAN, "run-1", 30)
    rc = run_worker(dispatch, runner_name=runner_name, out=out)
    messages = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    return [{"rc": rc}, *messages]


# ---- the worker (in-memory, no subprocess) ----------------------------------


def test_worker_streams_accepted_events_and_terminal():
    rc, *messages = _worker_out("echo")
    assert rc["rc"] == 0
    types = [m["type"] for m in messages]
    assert types[0] == "run_accepted"
    assert types[-1] == "terminal_result"
    assert "event" in types
    accepted = messages[0]["body"]
    assert accepted["run_id"] == "run-1" and isinstance(accepted["pid"], int)
    terminal = messages[-1]["body"]
    assert terminal["outcome"] == "PASS"
    assert terminal["run_manifest"]["state"] == "succeeded"
    # every message is a well-formed worker->core WorkerMessage envelope
    for m in messages:
        assert m["direction"] == "worker_to_core"
        assert m["protocol_version"] == "1.0.0"


def test_worker_forwards_run_events_in_order():
    _rc, *messages = _worker_out("echo")
    metrics = [m["body"] for m in messages if m["type"] == "event" and m["body"]["event_type"] == "metric"]
    assert [m["optimizer_step"] for m in metrics] == [1, 2, 3]


def test_worker_rejects_a_malformed_dispatch():
    out = io.StringIO()
    rc = run_worker("this is not json", runner_name="echo", out=out)
    assert rc == 2
    msg = json.loads(out.getvalue().splitlines()[0])
    assert msg["type"] == "run_rejected"


def test_build_runner_selects_the_runner():
    from corpus_studio.platform.runners import TrainingRunner
    from corpus_studio.platform.supervisor import EchoRunner
    from corpus_studio.platform.worker import _build_runner

    assert isinstance(_build_runner("echo", None), EchoRunner)
    trainer = _build_runner("cpu_toy", 5)
    assert isinstance(trainer, TrainingRunner)
    assert trainer.cpu_toy is True and trainer.max_steps == 5


def test_worker_main_runs_from_stdin(monkeypatch, capsys):
    from corpus_studio.platform import worker

    monkeypatch.setattr("sys.stdin", io.StringIO(_dispatch_line(_PLAN, "run-main", 30) + "\n"))
    monkeypatch.setattr("sys.argv", ["corpus-studio-worker", "--runner", "echo"])
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 0
    types = [json.loads(line)["type"] for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert types[0] == "run_accepted" and types[-1] == "terminal_result"


def _fake_worker(messages):
    """A stand-in worker child (python -c) that emits the given (type, body) WorkerMessages then exits —
    for exercising the parent's handling of misbehaving/protocol-drifting children."""
    script = (
        "import sys,json\n"
        "def s(t,b):\n"
        " sys.stdout.write(json.dumps({'protocol_version':'1.0.0','message_id':'x',"
        "'direction':'worker_to_core','type':t,'body':b})+chr(10));sys.stdout.flush()\n"
        + "".join(f"s({t!r},{b!r})\n" for t, b in messages)
    )
    return [sys.executable, "-c", script]


def test_malformed_terminal_result_is_a_protocol_failure_not_a_fake_crash():
    # A terminal_result arrived but its run_manifest doesn't validate → an honest protocol failure, NOT
    # "crashed (code 0)" and NEVER a fake success.
    argv = _fake_worker([
        ("run_accepted", {"run_id": "r", "pid": 1}),
        ("terminal_result", {"run_id": "r", "outcome": "PASS", "run_manifest": {"bogus": 1}}),
    ])
    result = execute_run_subprocess(_PLAN, worker_argv=argv, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert "malformed terminal_result" in result.manifest.failure.message


def test_run_rejected_is_classified_with_the_workers_reason():
    # The worker rejects the dispatch: its taxonomy + message must flow through, not be relabeled as a
    # generic crash.
    argv = _fake_worker([
        ("run_rejected", {"run_id": "r", "taxonomy": "UNSUPPORTED_CONFIGURATION", "message": "nope"}),
    ])
    result = execute_run_subprocess(_PLAN, worker_argv=argv, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "UNSUPPORTED_CONFIGURATION"
    assert result.manifest.failure.message == "nope"


def test_a_raising_sink_propagates_without_hanging_and_reaps_the_child():
    # A sink that raises must not orphan the child or deadlock — the try/finally reaps it and the error
    # propagates. (Before the fix, the exception skipped the reap and left a live worker.)
    def _boom(_event):
        raise RuntimeError("sink boom")

    with pytest.raises(RuntimeError, match="sink boom"):
        execute_run_subprocess(_PLAN, runner_name="echo", sink=_boom, silence_timeout_s=30)


def test_worker_main_empty_stdin_rejects(monkeypatch, capsys):
    from corpus_studio.platform import worker

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.argv", ["corpus-studio-worker", "--runner", "echo"])
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 2
    assert "run_rejected" in capsys.readouterr().out


# ---- the parent (REAL child processes) --------------------------------------


def test_echo_worker_roundtrip_through_a_real_subprocess():
    events = []
    result = execute_run_subprocess(_PLAN, runner_name="echo", sink=events.append, silence_timeout_s=30)
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "echo"
    assert [e.event_type for e in result.events] == [
        "stage", "metric", "metric", "metric", "stage", "terminal",
    ]
    assert [e.metrics.loss for e in result.events if e.event_type == "metric"] == [1.0, 0.5, 0.3333]
    assert events == result.events  # the sink saw the live stream


def test_hung_worker_is_killed_and_classified_kernel_stall():
    # A child that emits nothing and sleeps forever — the sm_120-deadlock stand-in. The parent owns the
    # process, so after the silence timeout it KILLS it and classifies KERNEL_STALL (impossible
    # in-process). The kill must happen promptly (well under the child's 120s sleep).
    hang = [sys.executable, "-c", "import time; time.sleep(120)"]
    start = time.monotonic()
    result = execute_run_subprocess(_PLAN, worker_argv=hang, silence_timeout_s=0.75)
    elapsed = time.monotonic() - start
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "KERNEL_STALL"
    assert elapsed < 15  # killed promptly, not after the 120s sleep


def test_crashed_worker_is_environment_failure():
    crash = [sys.executable, "-c", "import sys; sys.exit(3)"]
    result = execute_run_subprocess(_PLAN, worker_argv=crash, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert result.manifest.failure.exit_code == 3


def test_worker_that_emits_non_json_then_exits_is_a_crash():
    # A child that writes junk (not a WorkerMessage) then exits without a terminal_result: the junk is
    # dropped, and the missing terminal is classified as a crash — never a fake "success".
    noisy = [sys.executable, "-c", "print('hello from a broken worker'); print('{not json}')"]
    result = execute_run_subprocess(_PLAN, worker_argv=noisy, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert result.events == []  # nothing parsed as a RunEvent


def test_subprocess_writes_the_manifest_when_out_dir_given(tmp_path):
    result = execute_run_subprocess(
        _PLAN, runner_name="echo", out_dir=str(tmp_path), silence_timeout_s=30
    )
    written = tmp_path / "RunManifest.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["state"] == result.manifest.state
