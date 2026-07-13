"""The subprocess worker protocol — the parent that can KILL a hung run (the thing the in-process
watchdog can't). The worker is unit-tested in-memory (a StringIO out); the parent is integration-
tested against REAL child processes: an echo roundtrip, a hung child (killed → KERNEL_STALL), and a
crashed child (→ ENVIRONMENT_FAILURE). No torch/GPU needed — the fakes are tiny ``python -c`` scripts
and the echo runner."""

import io
import json
import subprocess
import sys
import time

import pytest

from corpus_studio.platform.subprocess_supervisor import (
    _dispatch_line,
    execute_run_subprocess,
    worker_identity_argv,
)
from corpus_studio.platform.supervisor import demo_run_plan
from corpus_studio.platform.worker_protocol import PROTOCOL_VERSION
from corpus_studio.platform.worker import run_worker

_PLAN = demo_run_plan()


def _worker_out(runner_name: str = "echo") -> list[dict]:
    """Drive run_worker in-process with an in-memory out stream; return the emitted WorkerMessages."""
    out = io.StringIO()
    dispatch = _dispatch_line(_PLAN, "run-1", 30)
    rc = run_worker(
        dispatch,
        runner_name=runner_name,
        backend_id=_PLAN.backend_ref.id,
        environment_ref=_PLAN.environment_ref,
        out=out,
    )
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
        assert m["protocol_version"] == PROTOCOL_VERSION
        assert m["correlation_id"] == "c-run-1"


def test_worker_forwards_run_events_in_order():
    _rc, *messages = _worker_out("echo")
    metrics = [m["body"] for m in messages if m["type"] == "event" and m["body"]["event_type"] == "metric"]
    assert [m["optimizer_step"] for m in metrics] == [1, 2, 3]


def test_worker_rejects_a_malformed_dispatch():
    out = io.StringIO()
    rc = run_worker(
        "this is not json",
        runner_name="echo",
        backend_id=_PLAN.backend_ref.id,
        environment_ref=_PLAN.environment_ref,
        out=out,
    )
    assert rc == 2
    msg = json.loads(out.getvalue().splitlines()[0])
    assert msg["type"] == "run_rejected"


def test_worker_rejects_a_well_formed_but_tampered_plan():
    envelope = json.loads(_dispatch_line(_PLAN, "run-1", 30))
    envelope["body"]["plan"]["seed"] += 1
    out = io.StringIO()
    rc = run_worker(
        json.dumps(envelope),
        runner_name="echo",
        backend_id=_PLAN.backend_ref.id,
        environment_ref=_PLAN.environment_ref,
        out=out,
    )
    assert rc == 2
    msg = json.loads(out.getvalue().splitlines()[0])
    assert msg["type"] == "run_rejected"
    assert "plan_hash" in msg["body"]["message"]


def test_worker_events_survive_a_trainer_stdout_redirect(monkeypatch):
    # The real trainer wraps trainer.train() in redirect_stdout(sys.stderr). The protocol channel IS
    # stdout, so if the per-step sink looked up sys.stdout AT CALL TIME it would land on stderr during
    # training and the parent (reading the stdout pipe) would see silence → false KERNEL_STALL. The
    # worker binds the real stdout up front. Reproduce: emit a step from INSIDE a redirect, out=None.
    import contextlib

    from corpus_studio.platform.runners import demo_training_plan
    from corpus_studio.training.trainer import TrainResult

    real_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", real_stdout)
    monkeypatch.setattr("sys.stderr", fake_stderr)

    def _trainer_redirecting(config, *, progress_callback=None, stage_callback=None, **_kw):
        with contextlib.redirect_stdout(sys.stderr):  # exactly what run_training does around .train()
            if progress_callback is not None:
                progress_callback(1, 1, 0.5)
        return TrainResult(
            output_dir="o", adapter_path="o", base_model=config.base_model, cpu_toy=True, steps=1
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _trainer_redirecting)
    training_plan = demo_training_plan()
    run_worker(
        _dispatch_line(training_plan, "run-r", 30),
        runner_name="cpu_toy",
        backend_id=training_plan.backend_ref.id,
        environment_ref=training_plan.environment_ref,
        out=None,
    )
    types = [json.loads(line)["type"] for line in real_stdout.getvalue().splitlines() if line.strip()]
    assert "event" in types  # the metric reached the REAL stdout (the pipe), not the redirected stderr
    assert types[-1] == "terminal_result"


def test_training_worker_echoes_the_sealed_execution_hash(monkeypatch):
    from corpus_studio.platform.runners import demo_training_plan
    from corpus_studio.training.trainer import TrainResult

    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        lambda config, **_kwargs: TrainResult(
            output_dir="o",
            adapter_path="o",
            base_model=config.base_model,
            cpu_toy=True,
        ),
    )
    plan = demo_training_plan()
    assert plan.resolved_execution is not None
    out = io.StringIO()
    rc = run_worker(
        _dispatch_line(plan, "sealed-run", 30),
        runner_name="cpu_toy",
        backend_id=plan.backend_ref.id,
        environment_ref=plan.environment_ref,
        out=out,
    )
    messages = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert rc == 0
    accepted = next(item for item in messages if item["type"] == "run_accepted")
    assert (
        accepted["body"]["execution_configuration_hash"]
        == plan.resolved_execution.configuration_hash
    )


def test_worker_rejects_training_plan_on_echo_lane_before_acceptance():
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    out = io.StringIO()
    rc = run_worker(
        _dispatch_line(plan, "wrong-lane", 30),
        runner_name="echo",
        backend_id=plan.backend_ref.id,
        environment_ref=plan.environment_ref,
        out=out,
    )
    messages = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert rc == 2
    assert [item["type"] for item in messages] == ["run_rejected"]
    assert "sealed lane" in messages[0]["body"]["message"]


def test_parent_refuses_wrong_runner_lane_before_spawning_worker():
    from corpus_studio.platform.runners import demo_training_plan

    result = execute_run_subprocess(
        demo_training_plan(),
        runner_name="echo",
        worker_argv=[sys.executable, "-c", "raise SystemExit(99)"],
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy.value == "UNSUPPORTED_CONFIGURATION"
    assert "sealed lane" in result.manifest.failure.message


def test_build_runner_selects_the_runner():
    from corpus_studio.platform.runners import TrainingRunner
    from corpus_studio.platform.supervisor import EchoRunner
    from corpus_studio.platform.worker import _build_runner

    assert isinstance(_build_runner("echo"), EchoRunner)
    trainer = _build_runner("cpu_toy")
    assert isinstance(trainer, TrainingRunner)
    assert trainer.cpu_toy is True and trainer.max_steps is None


def test_worker_main_runs_from_stdin(monkeypatch, capsys):
    from corpus_studio.platform import worker

    monkeypatch.setattr("sys.stdin", io.StringIO(_dispatch_line(_PLAN, "run-main", 30) + "\n"))
    monkeypatch.setattr(
        "sys.argv",
        ["corpus-studio-worker", "--runner", "echo", *worker_identity_argv(_PLAN)],
    )
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 0
    types = [json.loads(line)["type"] for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert types[:2] == ["hello", "run_accepted"] and types[-1] == "terminal_result"


def _hello_body(plan=_PLAN):
    from corpus_studio.platform.backends import get_worker_backend

    backend = get_worker_backend(plan.backend_ref.id)
    assert backend is not None
    return {
        "worker_id": "fake-worker",
        "backend": backend.model_dump(mode="json"),
        "environment_ref": plan.environment_ref.model_dump(mode="json"),
        "environment": None,
    }


def _fake_worker(
    messages,
    *,
    hello_body=None,
    hello_protocol=PROTOCOL_VERSION,
    hello_direction="worker_to_core",
    post_protocol=PROTOCOL_VERSION,
    post_direction="worker_to_core",
    post_correlation="dispatch",
    duplicate_post_ids=False,
):
    """A handshake-capable fake child for protocol/state-machine conformance tests."""
    encoded_messages = json.dumps(messages)
    encoded_hello = json.dumps(hello_body or _hello_body())
    script = (
        "import sys,json\n"
        "def s(t,b,mid,corr,version,direction):\n"
        " e={'protocol_version':version,'message_id':mid,'direction':direction,'type':t,'body':b}\n"
        " if corr is not None:e['correlation_id']=corr\n"
        " sys.stdout.write(json.dumps(e)+chr(10));sys.stdout.flush()\n"
        f"hello=json.loads({encoded_hello!r})\n"
        f"s('hello',hello,'hello',None,{hello_protocol!r},{hello_direction!r})\n"
        "line=sys.stdin.readline()\n"
        "dispatch=json.loads(line) if line.strip() else {}\n"
        f"messages=json.loads({encoded_messages!r})\n"
        f"mode={post_correlation!r}\n"
        "corr=dispatch.get('message_id') if mode=='dispatch' else mode\n"
        "for i,item in enumerate(messages):\n"
        f" mid='post' if {duplicate_post_ids!r} else 'post-'+str(i)\n"
        f" s(item[0],item[1],mid,corr,{post_protocol!r},{post_direction!r})\n"
    )
    return [sys.executable, "-c", script]


def test_parent_rejects_wrong_execution_hash_before_training_events():
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    argv = _fake_worker(
        [
            (
                "run_accepted",
                {
                    "run_id": plan.plan_id,
                    "pid": 1,
                    "execution_configuration_hash": "f" * 64,
                },
            )
        ],
        hello_body=_hello_body(plan),
    )
    result = execute_run_subprocess(
        plan, run_id=plan.plan_id, worker_argv=argv, silence_timeout_s=10
    )
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert "execution configuration hash" in result.manifest.failure.message


def test_malformed_terminal_result_is_a_protocol_failure_not_a_fake_crash():
    # A terminal_result arrived but its run_manifest doesn't validate → an honest protocol failure, NOT
    # "crashed (code 0)" and NEVER a fake success.
    argv = _fake_worker([
        ("run_accepted", {"run_id": _PLAN.plan_id, "pid": 1}),
        ("terminal_result", {"run_id": _PLAN.plan_id, "outcome": "PASS", "run_manifest": {"bogus": 1}}),
    ])
    result = execute_run_subprocess(
        _PLAN, run_id=_PLAN.plan_id, worker_argv=argv, silence_timeout_s=10
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert "protocol violation" in result.manifest.failure.message
    assert "terminal_result" in result.manifest.failure.message


def test_run_rejected_is_classified_with_the_workers_reason():
    # The worker rejects the dispatch: its taxonomy + message must flow through, not be relabeled as a
    # generic crash.
    argv = _fake_worker([
        ("run_rejected", {"run_id": _PLAN.plan_id, "taxonomy": "UNSUPPORTED_CONFIGURATION", "message": "nope"}),
    ])
    result = execute_run_subprocess(
        _PLAN, run_id=_PLAN.plan_id, worker_argv=argv, silence_timeout_s=10
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "UNSUPPORTED_CONFIGURATION"
    assert result.manifest.failure.message == "nope"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"hello_protocol": "99.0.0"}, "protocol version"),
        ({"hello_direction": "core_to_worker"}, "requires direction"),
        ({"post_correlation": None}, "correlation_id"),
    ],
)
def test_parent_rejects_protocol_direction_and_correlation_drift(kwargs, expected):
    messages = [("run_accepted", {"run_id": _PLAN.plan_id, "pid": 1})]
    result = execute_run_subprocess(
        _PLAN,
        run_id=_PLAN.plan_id,
        worker_argv=_fake_worker(messages, **kwargs),
        silence_timeout_s=10,
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert expected in result.manifest.failure.message


@pytest.mark.parametrize("identity", ["backend", "environment"])
def test_parent_rejects_worker_identity_mismatch_before_dispatch(identity):
    hello = json.loads(json.dumps(_hello_body()))
    if identity == "backend":
        hello["backend"]["backend_id"] = "different-backend"
    else:
        hello["environment_ref"]["id"] = "different-environment"
    result = execute_run_subprocess(
        _PLAN,
        worker_argv=_fake_worker([], hello_body=hello),
        silence_timeout_s=10,
    )
    assert result.manifest.state == "failed"
    assert identity in result.manifest.failure.message


def test_parent_rejects_duplicate_message_ids():
    messages = [
        ("run_accepted", {"run_id": _PLAN.plan_id, "pid": 1}),
        (
            "failure",
            {
                "run_id": _PLAN.plan_id,
                "taxonomy": "ENVIRONMENT_FAILURE",
                "message": "failed",
            },
        ),
    ]
    result = execute_run_subprocess(
        _PLAN,
        run_id=_PLAN.plan_id,
        worker_argv=_fake_worker(messages, duplicate_post_ids=True),
        silence_timeout_s=10,
    )
    assert "duplicate worker message_id" in result.manifest.failure.message


def test_parent_rejects_event_before_acceptance_and_nonmonotonic_sequences():
    _, *worker_messages = _worker_out("echo")
    event = dict(
        next(message["body"] for message in worker_messages if message["type"] == "event")
    )
    event["run_id"] = _PLAN.plan_id

    before = execute_run_subprocess(
        _PLAN,
        run_id=_PLAN.plan_id,
        worker_argv=_fake_worker([("event", event)]),
        silence_timeout_s=10,
    )
    assert "before run_accepted" in before.manifest.failure.message

    repeated = execute_run_subprocess(
        _PLAN,
        run_id=_PLAN.plan_id,
        worker_argv=_fake_worker(
            [
                ("run_accepted", {"run_id": _PLAN.plan_id, "pid": 1}),
                ("event", event),
                ("event", event),
            ]
        ),
        silence_timeout_s=10,
    )
    assert "is not greater than prior seq" in repeated.manifest.failure.message


def test_parent_rejects_terminal_manifest_linkage_mismatch():
    from corpus_studio.platform.contracts import RunManifest

    manifest = RunManifest(
        run_id=_PLAN.plan_id,
        plan_ref={"id": "wrong-plan", "hash": {"value": "0" * 64}},
        environment_ref=_PLAN.environment_ref,
        dataset_ref=_PLAN.dataset_ref,
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:00+00:00",
        state="succeeded",
        base_model="none",
        target="echo",
    )
    result = execute_run_subprocess(
        _PLAN,
        run_id=_PLAN.plan_id,
        worker_argv=_fake_worker(
            [
                ("run_accepted", {"run_id": _PLAN.plan_id, "pid": 1}),
                (
                    "terminal_result",
                    {
                        "run_id": _PLAN.plan_id,
                        "outcome": "PASS",
                        "run_manifest": manifest.model_dump(mode="json"),
                        "artifacts": [],
                        "failure": None,
                    },
                ),
            ]
        ),
        silence_timeout_s=10,
    )
    assert "does not link to the dispatched RunPlan" in result.manifest.failure.message


@pytest.mark.parametrize(
    ("artifact_case", "expected_error"),
    [
        ("rogue_path", "run-scoped output"),
        ("descriptor_only", "weight bytes do not match"),
    ],
)
def test_parent_rejects_false_training_artifact_success(
    tmp_path, artifact_case, expected_error
):
    from corpus_studio.platform.artifacts import build_artifact_manifest
    from corpus_studio.platform.common import HashRef
    from corpus_studio.platform.contracts import RunEvent, RunManifest
    from corpus_studio.platform.execution_config import (
        execution_configuration_hash_for,
        run_scoped_training_output,
    )
    from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    changed = execution.model_copy(update={"output_dir": str(tmp_path / "output-root")})
    changed = changed.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(changed)}
    )
    draft = plan.model_copy(
        update={
            "resolved_execution": changed,
            "export": plan.export.model_copy(update={"output_dir": changed.output_dir}),
        }
    )
    plan = draft.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))}
    )
    rid = "run-false-artifact"
    expected_output = run_scoped_training_output(changed, rid)
    artifact_path = (
        tmp_path / "rogue-adapter"
        if artifact_case == "rogue_path"
        else expected_output
    )
    artifact_path.mkdir(parents=True)
    if artifact_case == "rogue_path":
        (artifact_path / "adapter_model.safetensors").write_bytes(b"weights")
    else:
        (artifact_path / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")
    artifact = build_artifact_manifest(
        artifact_id="run-false-artifact-adapter-deadbeef",
        path=str(artifact_path),
        run_id=rid,
        now="2026-07-13T00:00:00+00:00",
    )
    manifest = RunManifest(
        run_id=rid,
        plan_ref={"id": plan.plan_id, "hash": HashRef(value=plan.plan_hash)},
        environment_ref=plan.environment_ref,
        dataset_ref=plan.dataset_ref,
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:00+00:00",
        state="succeeded",
        base_model=plan.base_model,
        target=plan.backend_ref.id,
        output_dir=str(artifact_path),
        artifact_ids=[artifact.artifact_id],
    )
    event = RunEvent(
        event_type="metric",
        run_id=rid,
        seq=0,
        emitted_at="2026-07-13T00:00:00+00:00",
        optimizer_step=1,
    )
    messages = [
        (
            "run_accepted",
            {
                "run_id": rid,
                "pid": 1,
                "execution_configuration_hash": changed.configuration_hash,
            },
        ),
        ("event", event.model_dump(mode="json")),
        (
            "terminal_result",
            {
                "run_id": rid,
                "outcome": "PASS",
                "run_manifest": manifest.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json")],
                "failure": None,
            },
        ),
    ]
    result = execute_run_subprocess(
        plan,
        run_id=rid,
        worker_argv=_fake_worker(messages, hello_body=_hello_body(plan)),
        silence_timeout_s=10,
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert expected_error in result.manifest.failure.message


def test_worker_rejects_backend_identity_before_building_runner(monkeypatch):
    from corpus_studio.platform.common import HashRef, Ref
    from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload

    mismatched = _PLAN.model_copy(
        update={"backend_ref": Ref(id="echo", hash=HashRef(value="0" * 64))}
    )
    mismatched = mismatched.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(mismatched))}
    )
    monkeypatch.setattr(
        "corpus_studio.platform.worker._build_runner",
        lambda *_args: pytest.fail("runner must not be built before identity validation"),
    )
    out = io.StringIO()
    rc = run_worker(
        _dispatch_line(mismatched, "run-identity", 30),
        runner_name="echo",
        backend_id="echo",
        environment_ref=mismatched.environment_ref,
        out=out,
    )
    assert rc == 2
    assert "backend manifest identity" in out.getvalue()


def test_parent_refuses_a_tampered_plan_before_spawning_worker(tmp_path):
    launched = tmp_path / "worker-launched"
    plan = _PLAN.model_copy(update={"seed": _PLAN.seed + 1})
    result = execute_run_subprocess(
        plan,
        worker_argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(launched)!r}).write_text('launched')",
        ],
        silence_timeout_s=10,
    )
    assert result.manifest.failure.taxonomy.value == "UNSUPPORTED_CONFIGURATION"
    assert "hash verification failed" in result.manifest.failure.message
    assert not launched.exists()


def test_worker_protocol_import_does_not_load_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import corpus_studio.platform.worker_protocol; "
            "print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


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
    monkeypatch.setattr(
        "sys.argv",
        ["corpus-studio-worker", "--runner", "echo", *worker_identity_argv(_PLAN)],
    )
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


def test_heartbeat_spam_cannot_mask_a_hung_run():
    hello = json.dumps(_hello_body())
    script = (
        "import json,sys,time\n"
        "def s(t,b,mid,corr=None):\n"
        f" e={{'protocol_version':{PROTOCOL_VERSION!r},'message_id':mid,"
        "'direction':'worker_to_core','type':t,'body':b}\n"
        " if corr is not None:e['correlation_id']=corr\n"
        " print(json.dumps(e),flush=True)\n"
        f"s('hello',json.loads({hello!r}),'hello')\n"
        "dispatch=json.loads(sys.stdin.readline());corr=dispatch['message_id'];"
        "rid=dispatch['body']['run_id']\n"
        "s('run_accepted',{'run_id':rid,'pid':1},'accepted',corr)\n"
        "i=0\n"
        "while True:\n"
        " s('heartbeat',{'run_id':rid,'pid_alive':True},'hb-'+str(i),corr)\n"
        " i+=1;time.sleep(0.02)\n"
    )
    result = execute_run_subprocess(
        _PLAN,
        worker_argv=[sys.executable, "-c", script],
        silence_timeout_s=0.2,
    )
    assert result.manifest.failure.taxonomy.value == "KERNEL_STALL"
    assert "no run progress" in result.manifest.failure.message


def test_hung_worker_termination_kills_its_descendant(tmp_path):
    ready = tmp_path / "descendant-ready"
    orphan = tmp_path / "descendant-survived"
    child_script = (
        "import pathlib,time;"
        f"pathlib.Path({str(ready)!r}).write_text('ready');"
        "time.sleep(1);"
        f"pathlib.Path({str(orphan)!r}).write_text('orphan')"
    )
    hello = json.dumps(_hello_body())
    worker_script = (
        "import json,pathlib,subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{child_script!r}],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        f"ready=pathlib.Path({str(ready)!r})\n"
        "deadline=time.monotonic()+5\n"
        "while not ready.exists() and time.monotonic()<deadline:time.sleep(0.01)\n"
        "def s(t,b,mid,corr=None):\n"
        f" e={{'protocol_version':{PROTOCOL_VERSION!r},'message_id':mid,"
        "'direction':'worker_to_core','type':t,'body':b}\n"
        " if corr is not None:e['correlation_id']=corr\n"
        " print(json.dumps(e),flush=True)\n"
        f"s('hello',json.loads({hello!r}),'hello')\n"
        "dispatch=json.loads(sys.stdin.readline());corr=dispatch['message_id'];"
        "rid=dispatch['body']['run_id']\n"
        "s('run_accepted',{'run_id':rid,'pid':1},'accepted',corr)\n"
        "time.sleep(120)\n"
    )
    result = execute_run_subprocess(
        _PLAN,
        worker_argv=[sys.executable, "-c", worker_script],
        silence_timeout_s=0.2,
    )
    assert ready.exists(), "the worker must launch its descendant before the timeout"
    assert result.manifest.failure.taxonomy.value == "KERNEL_STALL"
    time.sleep(1.1)
    assert not orphan.exists(), "a timed-out worker descendant survived process-tree termination"


def test_crashed_worker_is_environment_failure():
    crash = [sys.executable, "-c", "import sys; sys.exit(3)"]
    result = execute_run_subprocess(_PLAN, worker_argv=crash, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert result.manifest.failure.exit_code == 3


def test_worker_that_emits_non_json_is_a_protocol_failure():
    # stdout is exclusively the wire channel. Junk is protocol drift, never ignored telemetry.
    noisy = [sys.executable, "-c", "print('hello from a broken worker'); print('{not json}')"]
    result = execute_run_subprocess(_PLAN, worker_argv=noisy, silence_timeout_s=10)
    assert result.manifest.state == "failed"
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert "protocol violation" in result.manifest.failure.message
    assert result.events == []  # nothing parsed as a RunEvent


def test_subprocess_persists_the_childs_artifact_manifests(tmp_path):
    # The child builds ArtifactManifests but (running execute_run without out_dir) doesn't write them;
    # the PARENT must persist them under --out, so the CLI's "wrote N artifact manifest(s)" isn't a lie
    # and the manifest's artifact_ids aren't dangling.
    from corpus_studio.platform.artifacts import build_artifact_manifest
    from corpus_studio.platform.contracts import RunManifest

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    am = build_artifact_manifest(
        artifact_id="run-x-adapter", path=str(adapter), run_id="run-x", now="2026-07-12T00:00:00+00:00"
    )
    rm = RunManifest(
        run_id="run-x",
        plan_ref={"id": _PLAN.plan_id, "hash": {"value": _PLAN.plan_hash}},
        environment_ref=_PLAN.environment_ref,
        dataset_ref=_PLAN.dataset_ref,
        created_at="2026-07-12T00:00:00+00:00",
        updated_at="2026-07-12T00:00:00+00:00",
        started_at="2026-07-12T00:00:00+00:00",
        finished_at="2026-07-12T00:00:00+00:00",
        state="succeeded",
        base_model="m",
        target="echo",
        output_dir="o",
        artifact_ids=["run-x-adapter"],
    )
    argv = _fake_worker([
        ("run_accepted", {"run_id": "run-x", "pid": 1}),
        ("terminal_result", {"run_id": "run-x", "outcome": "PASS",
                             "run_manifest": rm.model_dump(mode="json"),
                             "artifacts": [am.model_dump(mode="json")], "failure": None}),
    ])
    out = tmp_path / "out"
    result = execute_run_subprocess(
        _PLAN,
        run_id="run-x",
        worker_argv=argv,
        out_dir=str(out),
        silence_timeout_s=10,
    )
    assert result.manifest.state == "succeeded"
    assert len(result.artifacts) == 1
    assert (
        out / "runs" / "run-x" / "artifacts" / "run-x-adapter.json"
    ).exists()  # actually persisted, not just reported


def test_subprocess_writes_the_manifest_when_out_dir_given(tmp_path):
    result = execute_run_subprocess(
        _PLAN, runner_name="echo", out_dir=str(tmp_path), silence_timeout_s=30
    )
    written = tmp_path / "runs" / result.manifest.run_id / "RunManifest.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["state"] == result.manifest.state
