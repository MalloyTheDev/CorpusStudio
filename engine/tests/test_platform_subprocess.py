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

from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
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


def test_worker_env_strips_import_shadowing_vars(monkeypatch):
    """The worker child never inherits PYTHONPATH/PYTHONHOME, so an inherited path cannot shadow the
    sealed wheel with another checkout (together with -P in the argv). Other vars are preserved."""
    from corpus_studio.platform.subprocess_supervisor import _worker_env

    monkeypatch.setenv("PYTHONPATH", "/tmp/other-checkout/engine")
    monkeypatch.setenv("PYTHONHOME", "/tmp/other-home")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = _worker_env()
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env.get("CUDA_VISIBLE_DEVICES") == "0"


def test_dispatch_carries_resume_only_when_resuming():
    """The run_dispatch line embeds the resume instruction only for an actual resume; an ordinary
    from-scratch dispatch is byte-for-byte unchanged, so a worker that never consumes resume receives
    an identical line and the resume path stays opt-in rather than a silent default."""
    from corpus_studio.platform.contracts import CheckpointResumeRequest

    plain = json.loads(_dispatch_line(_PLAN, "run-1", 30))
    assert "resume" not in plain["body"]
    # The new parameter must not perturb the from-scratch dispatch bytes (default vs explicit None).
    assert _dispatch_line(_PLAN, "run-1", 30) == _dispatch_line(_PLAN, "run-1", 30, None)

    req = CheckpointResumeRequest(
        checkpoint_id="run-1-ckpt-step-00000002",
        checkpoint_manifest_hash="a" * 64,
        checkpoint_dir="/checkpoints/step-00000002",
    )
    resuming = json.loads(_dispatch_line(_PLAN, "run-1", 30, req))
    assert resuming["body"]["resume"]["checkpoint_dir"] == "/checkpoints/step-00000002"
    assert resuming["body"]["resume"]["checkpoint_manifest_hash"] == "a" * 64


def _write_min_checkpoint(directory) -> "CheckpointResumeRequest":  # noqa: F821
    """Seal a minimal real checkpoint at ``directory`` and return the resume request that names it."""
    import hashlib

    from corpus_studio.platform import checkpoint as ck
    from corpus_studio.platform.contracts import (
        CheckpointFileEntry,
        CheckpointManifest,
        CheckpointResumeRequest,
        SealedTrainingState,
    )
    from corpus_studio.platform.runners import demo_training_plan

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "optimizer.pt").write_bytes(b"opt")
    (directory / "rng.pt").write_bytes(b"rng")
    files = sorted(
        [
            CheckpointFileEntry(
                path="optimizer.pt", role="optimizer",
                sha256=hashlib.sha256(b"opt").hexdigest(), size_bytes=3,
            ),
            CheckpointFileEntry(
                path="rng.pt", role="rng",
                sha256=hashlib.sha256(b"rng").hexdigest(), size_bytes=3,
            ),
        ],
        key=lambda e: e.path,
    )
    sealed = ck.seal_checkpoint_manifest(
        CheckpointManifest(
            checkpoint_id="run-parent01-ckpt-step-00000002",
            checkpoint_manifest_hash="0" * 64,
            source_run_id="run-parent01",
            created_at="2026-07-15T00:00:00+00:00",
            bound=ck.bound_identities_from_plan(demo_training_plan(plan_id="demo-ckpt")),
            state=SealedTrainingState(
                scheduler_captured=True, scaler_captured=False, rng_captured=True,
                sampler_state_captured=True, rng_algorithm="philox", epoch=0.5,
                global_optimizer_step=2, microstep_within_step=0,
                gradient_accumulation_steps=1, consumed_microsteps=2,
            ),
            files=files,
        )
    )
    ck.write_checkpoint_manifest(sealed, directory)
    return CheckpointResumeRequest(
        checkpoint_id=sealed.checkpoint_id,
        checkpoint_manifest_hash=sealed.checkpoint_manifest_hash,
        checkpoint_dir=str(directory),
    )


def test_worker_forwards_resume_from_dispatch_to_execute_run(monkeypatch, tmp_path):
    """Reachability: a resume carried by the dispatch reaches ``execute_run`` - the managed
    --subprocess path is wired end to end, not just the in-process one. Before this wiring the
    subprocess worker silently ignored resume and re-ran from scratch."""
    import corpus_studio.platform.supervisor as supervisor_mod

    real_execute_run = supervisor_mod.execute_run
    captured: dict = {}

    def _spy(plan, runner, **kwargs):
        captured["resume"] = kwargs.get("resume")
        return real_execute_run(plan, runner, **kwargs)

    monkeypatch.setattr(supervisor_mod, "execute_run", _spy)

    req = _write_min_checkpoint(tmp_path / "step-00000002")
    out = io.StringIO()
    rc = run_worker(
        _dispatch_line(_PLAN, "run-1", 30, req),
        runner_name="echo",
        backend_id=_PLAN.backend_ref.id,
        environment_ref=_PLAN.environment_ref,
        out=out,
    )
    assert rc == 0
    assert captured["resume"] is not None
    assert captured["resume"].checkpoint_dir == str(tmp_path / "step-00000002")
    # The resumed run's manifest records the parent lineage read from that checkpoint (F2).
    messages = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    terminal = next(m for m in messages if m["type"] == "terminal_result")
    lineage = terminal["body"]["run_manifest"].get("resume_lineage")
    assert lineage is not None and lineage["parent_run_id"] == "run-parent01"
    assert lineage["resumed_from_global_step"] == 2
    # And the from-scratch dispatch forwards resume=None (no accidental resume on ordinary runs).
    captured.clear()
    run_worker(
        _dispatch_line(_PLAN, "run-1", 30),
        runner_name="echo",
        backend_id=_PLAN.backend_ref.id,
        environment_ref=_PLAN.environment_ref,
        out=io.StringIO(),
    )
    assert captured["resume"] is None


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
    from corpus_studio.platform.runners import PretrainingRunner, TrainingRunner
    from corpus_studio.platform.supervisor import EchoRunner
    from corpus_studio.platform.worker import _build_runner

    assert isinstance(_build_runner("echo"), EchoRunner)
    trainer = _build_runner("cpu_toy")
    assert isinstance(trainer, TrainingRunner)
    assert trainer.cpu_toy is True and trainer.max_steps is None
    # the workload_verified pretraining lanes route to the full-parameter PretrainingRunner, never SFT
    assert isinstance(_build_runner("training"), TrainingRunner)
    pretrain = _build_runner("pretraining")
    assert isinstance(pretrain, PretrainingRunner) and pretrain.cpu_toy is False
    pretrain_toy = _build_runner("pretraining_cpu_toy")
    assert isinstance(pretrain_toy, PretrainingRunner) and pretrain_toy.cpu_toy is True
    # corpus_root threads to the pretraining runner (relative shard anchor); default is CWD.
    assert _build_runner("pretraining", corpus_root="/data/corpus").corpus_root == "/data/corpus"


def test_build_lane_runner_is_the_shared_factory():
    # The SINGLE lane->Runner mapping that BOTH the in-process platform_run path and the subprocess
    # worker call, so they cannot drift - the exact defect where the in-process CLI path lacked the
    # pretraining lanes while the worker had them. max_steps is the CLI-only trainer cap; corpus_root
    # anchors a pretraining plan's relative shards.
    from corpus_studio.platform.runners import (
        FullFinetuneRunner,
        PreferenceRunner,
        PretrainingRunner,
        TrainingRunner,
        build_lane_runner,
    )
    from corpus_studio.platform.supervisor import EchoRunner

    assert isinstance(build_lane_runner("echo"), EchoRunner)
    train = build_lane_runner("training", max_steps=7)
    assert isinstance(train, TrainingRunner) and train.cpu_toy is False and train.max_steps == 7
    assert build_lane_runner("cpu_toy").cpu_toy is True
    assert isinstance(build_lane_runner("preference"), PreferenceRunner)
    assert isinstance(build_lane_runner("full_finetune"), FullFinetuneRunner)
    pre = build_lane_runner("pretraining", corpus_root="/data/corpus")
    assert isinstance(pre, PretrainingRunner) and pre.cpu_toy is False
    assert pre.corpus_root == "/data/corpus"
    assert build_lane_runner("pretraining_cpu_toy").cpu_toy is True


def test_worker_arg_parser_accepts_the_pretraining_lanes():
    # Regression: the pretraining subprocess lane shipped broken because worker.main's argparse choices
    # omitted it - _build_runner mapped it, but argparse exited 2 ("invalid choice") first, so a managed
    # pretraining plan (forced to --subprocess) could never run. Gate on what the parser ACCEPTS.
    from corpus_studio.platform.worker import _build_arg_parser

    parser = _build_arg_parser()
    base = ["--backend-id", "b", "--environment-id", "e"]
    # Every lane build_lane_runner routes must be an accepted --runner choice, or it is a dead subprocess
    # lane: preference (DPO, workload_verified) + full_finetune both route, so both must parse.
    for lane in (
        "echo", "cpu_toy", "training", "preference", "full_finetune",
        "pretraining", "pretraining_cpu_toy",
    ):
        assert parser.parse_args(["--runner", lane, *base]).runner == lane
    assert parser.parse_args(base).corpus_root == "."
    assert parser.parse_args([*base, "--corpus-root", "/data/corpus"]).corpus_root == "/data/corpus"
    with pytest.raises(SystemExit):
        parser.parse_args(["--runner", "not-a-lane", *base])


def test_bind_protocol_stream_isolates_the_protocol_from_fd1_writes():
    # #10 regression: a NATIVE/C write straight to fd 1 (which a Python-level redirect_stdout cannot
    # catch - the pretraining worker's tokenizer training / transformers paths do this) must NOT corrupt
    # the framed protocol. It must land on stderr, while the protocol stream (a private dup of the real
    # stdout) still reaches the stdout the parent reads. Run in a SUBPROCESS so the fd-level dup2 never
    # touches this test process's own fd 1.
    import os as _os
    import subprocess
    import textwrap

    engine_dir = _os.path.dirname(_os.path.dirname(__file__))
    script = textwrap.dedent(
        """
        import os
        from corpus_studio.platform.worker import _bind_protocol_stream
        stream = _bind_protocol_stream()
        os.write(1, b"NATIVE-NOISE\\n")   # native fd-1 write; bypasses sys.stdout
        print("python-print")             # sys.stdout -> fd 1 -> stderr after the redirect
        stream.write("PROTOCOL\\n"); stream.flush()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=engine_dir
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PROTOCOL"  # ONLY the protocol reaches the real stdout
    assert "NATIVE-NOISE" in result.stderr and "python-print" in result.stderr  # all fd-1 -> stderr


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


def _timed_training_worker(plan, actions: str):
    """A training-identity fake whose action block can exercise real supervisor deadlines."""

    hello = json.dumps(_hello_body(plan))
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
        "execution_hash=dispatch['body']['plan']['resolved_execution']['configuration_hash']\n"
        "s('run_accepted',{'run_id':rid,'pid':1,'execution_configuration_hash':execution_hash},"
        "'accepted',corr)\n"
        "seq=0\n"
        "def event(stage,optimizer_step=None):\n"
        " global seq\n"
        " body={'contract_version':'1.0.0','event_type':'stage','run_id':rid,'seq':seq,"
        "'emitted_at':'2026-07-15T00:00:00+00:00','stage':stage,'message':stage}\n"
        " if optimizer_step is not None:\n"
        "  body['event_type']='metric';body['optimizer_step']=optimizer_step\n"
        " s('event',body,'event-'+str(seq),corr);seq+=1\n"
        "def heartbeat(index):\n"
        " s('heartbeat',{'run_id':rid,'pid_alive':True},'hb-'+str(index),corr)\n"
        + actions
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


def test_parent_rejects_terminal_event_state_mismatch_without_exposing_success():
    from corpus_studio.platform.contracts import RunManifest

    rid = "run-terminal-mismatch"
    manifest = RunManifest(
        run_id=rid,
        plan_ref={"id": _PLAN.plan_id, "hash": {"value": _PLAN.plan_hash}},
        environment_ref=_PLAN.environment_ref,
        dataset_ref=_PLAN.dataset_ref,
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:00+00:00",
        state="succeeded",
        base_model=_PLAN.base_model,
        target=_PLAN.backend_ref.id,
    )
    terminal_event = {
        "contract_version": "1.0.0",
        "event_type": "terminal",
        "run_id": rid,
        "seq": 0,
        "emitted_at": "2026-07-13T00:00:00+00:00",
        "payload": {"state": "failed"},
    }
    observed = []
    result = execute_run_subprocess(
        _PLAN,
        run_id=rid,
        sink=observed.append,
        worker_argv=_fake_worker(
            [
                ("run_accepted", {"run_id": rid, "pid": 1}),
                ("event", terminal_event),
                (
                    "terminal_result",
                    {
                        "run_id": rid,
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

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert "terminal RunEvent state" in result.manifest.failure.message
    assert [event.payload for event in observed if event.event_type == "terminal"] == [
        {"state": "failed"}
    ]


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
    from corpus_studio.platform.contracts import (
        RunEvent,
        RunManifest,
        TrainingSuccessEvidence,
    )
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
        base_model=plan.base_model,
        now="2026-07-13T00:00:00+00:00",
    )
    execution_evidence = {
        "trainable_state": {
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "trainable_tensor_count": 1,
            "trainable_tensor_names": ["adapter.weight"],
            "changed_tensor_count": 1,
            "changed_tensor_names": ["adapter.weight"],
        },
        "adapter_export_state": {
            "before_sha256": "c" * 64,
            "after_sha256": "d" * 64,
            "tensor_count": 1,
            "tensor_names": ["adapter.lora_A.weight"],
            "changed_tensor_count": 1,
            "changed_tensor_names": ["adapter.lora_A.weight"],
            "adapter_config_semantic_sha256": "e" * 64,
        },
        "gradient_coverage": {
            "eligible_tensor_count": 1,
            "eligible_tensor_names": ["adapter.weight"],
            "observed_tensor_count": 1,
            "observed_tensor_names": ["adapter.weight"],
        },
        "optimizer_created": True,
        "completed_optimizer_steps": 2,
        "step_losses": [
            {"optimizer_step": 1, "loss": 0.9},
            {"optimizer_step": 2, "loss": 0.5},
        ],
    }
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
        training_success_evidence=TrainingSuccessEvidence(
            execution=execution_evidence,
            output_path_verified=True,
            adapter_bytes_verified=True,
            artifact_integrity_verified=True,
            adapter_safetensors_sha256="f" * 64,
            adapter_config_sha256="0" * 64,
        ),
    )
    events = [
        RunEvent(
            event_type="stage",
            run_id=rid,
            seq=0,
            emitted_at="2026-07-13T00:00:00+00:00",
            stage="optimizer_created",
        ),
        RunEvent(
            event_type="metric",
            run_id=rid,
            seq=1,
            emitted_at="2026-07-13T00:00:00+00:00",
            optimizer_step=1,
            metrics={"loss": 0.9},
        ),
        RunEvent(
            event_type="metric",
            run_id=rid,
            seq=2,
            emitted_at="2026-07-13T00:00:00+00:00",
            optimizer_step=2,
            metrics={"loss": 0.5},
        ),
    ]
    messages = [
        (
            "run_accepted",
            {
                "run_id": rid,
                "pid": 1,
                "execution_configuration_hash": changed.configuration_hash,
            },
        ),
        *(("event", event.model_dump(mode="json")) for event in events),
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
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
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


def test_a_raising_sink_isolated_without_hanging_and_reaps_the_child():
    # An observer cannot rewrite a valid terminal outcome or orphan the child.
    def _boom(_event):
        raise RuntimeError("sink boom")

    result = execute_run_subprocess(
        _PLAN, runner_name="echo", sink=_boom, silence_timeout_s=30
    )
    assert result.manifest.state == "succeeded"
    assert result.manifest.notes == "event sink failures were isolated: RuntimeError"


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"silence_timeout_s": 0}, "silence_timeout_s"),
        ({"silence_timeout_s": float("nan")}, "silence_timeout_s"),
        ({"silence_timeout_s": float("inf")}, "silence_timeout_s"),
        ({"preflight_timeout_s": 0}, "preflight_timeout_s"),
        ({"preflight_timeout_s": float("nan")}, "preflight_timeout_s"),
        ({"preflight_timeout_s": float("inf")}, "preflight_timeout_s"),
        ({"heartbeat_interval_s": 0}, "heartbeat_interval_s"),
    ],
)
def test_nonpositive_deadlines_are_rejected_before_worker_spawn(monkeypatch, kwargs, expected):
    monkeypatch.setattr(
        "corpus_studio.platform.subprocess_supervisor.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not spawn for an invalid deadline"),
    )
    with pytest.raises(ValueError, match=expected):
        execute_run_subprocess(_PLAN, **kwargs)


def test_training_preflight_can_outlive_the_ordinary_silence_budget():
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    actions = (
        "event('process_start')\n"
        "time.sleep(0.7)\n"
        "s('failure',{'run_id':rid,'taxonomy':'ENVIRONMENT_FAILURE',"
        "'stage':'model_load','message':'synthetic preflight failure'},'failure',corr)\n"
    )
    result = execute_run_subprocess(
        plan,
        runner_name="cpu_toy",
        worker_argv=_timed_training_worker(plan, actions),
        silence_timeout_s=0.5,
        preflight_timeout_s=2,
    )

    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy.value == "ENVIRONMENT_FAILURE"
    assert result.manifest.failure.message == "synthetic preflight failure"
    assert result.manifest.failure.stage == StageMarker.model_load


def test_repeated_preflight_progress_cannot_extend_the_absolute_deadline():
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    actions = (
        "index=0\n"
        "spam_until=time.monotonic()+0.8\n"
        "while time.monotonic()<spam_until:\n"
        " event('dataset_formatting')\n"
        " heartbeat(index);index+=1;time.sleep(0.02)\n"
        "time.sleep(2)\n"
    )
    start = time.monotonic()
    result = execute_run_subprocess(
        plan,
        runner_name="cpu_toy",
        worker_argv=_timed_training_worker(plan, actions),
        silence_timeout_s=0.5,
        preflight_timeout_s=0.2,
    )
    elapsed = time.monotonic() - start

    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy.value == "TIMEOUT"
    assert result.manifest.failure.stage.value == "dataset_formatting"
    assert "non-extendable" in result.manifest.failure.message
    assert elapsed < 5


def test_heartbeat_spam_after_optimizer_creation_cannot_mask_a_stall():
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan()
    actions = (
        "event('process_start')\n"
        "event('optimizer_created')\n"
        "index=0\n"
        "while True:\n"
        " heartbeat(index);index+=1;time.sleep(0.02)\n"
    )
    result = execute_run_subprocess(
        plan,
        runner_name="cpu_toy",
        worker_argv=_timed_training_worker(plan, actions),
        silence_timeout_s=0.15,
        preflight_timeout_s=1,
    )

    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy.value == "KERNEL_STALL"
    assert result.manifest.failure.stage.value == "optimizer_created"


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
        artifact_id="run-x-adapter", path=str(adapter), run_id="run-x",
        base_model=_PLAN.base_model, now="2026-07-12T00:00:00+00:00"
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
        base_model=_PLAN.base_model,
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


def test_subprocess_artifact_persistence_failure_cannot_leave_succeeded_manifest(
    tmp_path, monkeypatch
):
    from corpus_studio.platform.artifacts import build_artifact_manifest
    from corpus_studio.platform.contracts import RunManifest

    adapter = tmp_path / "adapter-failure"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    artifact = build_artifact_manifest(
        artifact_id="run-persist-fail-adapter",
        path=str(adapter),
        run_id="run-persist-fail",
        base_model=_PLAN.base_model,
        now="2026-07-12T00:00:00+00:00",
    )
    child_manifest = RunManifest(
        run_id="run-persist-fail",
        plan_ref={"id": _PLAN.plan_id, "hash": {"value": _PLAN.plan_hash}},
        environment_ref=_PLAN.environment_ref,
        dataset_ref=_PLAN.dataset_ref,
        created_at="2026-07-12T00:00:00+00:00",
        updated_at="2026-07-12T00:00:00+00:00",
        state="succeeded",
        base_model=_PLAN.base_model,
        target="echo",
        output_dir="o",
        artifact_ids=[artifact.artifact_id],
    )
    argv = _fake_worker(
        [
            ("run_accepted", {"run_id": "run-persist-fail", "pid": 1}),
            (
                "terminal_result",
                {
                    "run_id": "run-persist-fail",
                    "outcome": "PASS",
                    "run_manifest": child_manifest.model_dump(mode="json"),
                    "artifacts": [artifact.model_dump(mode="json")],
                    "failure": None,
                },
            ),
        ]
    )
    monkeypatch.setattr(
        "corpus_studio.platform.subprocess_supervisor.write_artifact_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    out = tmp_path / "records"
    result = execute_run_subprocess(
        _PLAN,
        run_id="run-persist-fail",
        worker_argv=argv,
        out_dir=out,
        silence_timeout_s=10,
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
    persisted = RunManifest.model_validate_json(
        (out / "runs" / "run-persist-fail" / "RunManifest.json").read_text()
    )
    assert persisted.state == "failed"


def test_parent_defers_terminal_success_until_run_manifest_is_durable(tmp_path, monkeypatch):
    from corpus_studio.platform.contracts import RunManifest
    import corpus_studio.platform.subprocess_supervisor as supervisor_module

    original = supervisor_module.write_run_manifest
    writes = 0

    def _fail_first(manifest, out_dir):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("synthetic terminal write failure")
        return original(manifest, out_dir)

    monkeypatch.setattr(supervisor_module, "write_run_manifest", _fail_first)
    observed = []
    result = execute_run_subprocess(
        _PLAN,
        runner_name="echo",
        run_id="run-terminal-durable",
        out_dir=tmp_path / "records",
        sink=observed.append,
        silence_timeout_s=30,
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    terminal = [event for event in observed if event.event_type == "terminal"]
    assert len(terminal) == 1
    assert terminal[0].payload == {"state": "failed"}
    persisted = RunManifest.model_validate_json(
        (
            tmp_path
            / "records"
            / "runs"
            / "run-terminal-durable"
            / "RunManifest.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted.state == "failed"


def test_parent_preserves_failure_fields_and_fills_only_missing_stage():
    stage_event = {
        "contract_version": "1.0.0",
        "event_type": "stage",
        "run_id": "run-rich-failure",
        "seq": 0,
        "emitted_at": "2026-07-15T00:00:00+00:00",
        "stage": "model_load",
    }
    failure = {
        "run_id": "run-rich-failure",
        "taxonomy": "ENVIRONMENT_FAILURE",
        "exit_code": 17,
        "signal": "SIGABRT",
        "message": "framework aborted",
        "detail": "sealed detail",
        "exception_type": "CudaRuntimeError",
        "detected_at": "2026-07-15T00:01:00+00:00",
        "memory_at_failure": {"shared_gpu_bytes": 4096},
        "remediation": "preserve evidence",
        "reconciled": True,
    }
    result = execute_run_subprocess(
        _PLAN,
        run_id="run-rich-failure",
        worker_argv=_fake_worker(
            [
                ("run_accepted", {"run_id": "run-rich-failure", "pid": 1}),
                ("event", stage_event),
                ("failure", failure),
            ]
        ),
        silence_timeout_s=10,
    )

    observed = result.manifest.failure
    assert observed is not None
    assert observed.stage == StageMarker.model_load
    assert observed.exit_code == 17
    assert observed.signal == "SIGABRT"
    assert observed.exception_type == "CudaRuntimeError"
    assert observed.detected_at == "2026-07-15T00:01:00+00:00"
    assert observed.memory_at_failure is not None
    assert observed.memory_at_failure.shared_gpu_bytes == 4096
    assert observed.reconciled is True


def test_subprocess_writes_the_manifest_when_out_dir_given(tmp_path):
    result = execute_run_subprocess(
        _PLAN, runner_name="echo", out_dir=str(tmp_path), silence_timeout_s=30
    )
    written = tmp_path / "runs" / result.manifest.run_id / "RunManifest.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["state"] == result.manifest.state


def test_spawn_failure_is_classified_environment_failure(tmp_path):
    # #510: a worker that cannot be spawned (a missing interpreter) must return a classified
    # ENVIRONMENT_FAILURE manifest, not let the OSError escape the supervisor - and it must persist a
    # durable manifest (proving the events log was not left leaked/unclosed on the failure path).
    result = execute_run_subprocess(
        _PLAN,
        run_id="spawn-fail",
        worker_argv=["/nonexistent/interpreter-xyz", "-c", "pass"],
        out_dir=str(tmp_path),
        silence_timeout_s=10,
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    written = tmp_path / "runs" / result.manifest.run_id / "RunManifest.json"
    assert written.exists()


def test_non_utf8_worker_stdout_does_not_strand_the_reader(tmp_path):
    # #508: a worker that emits a raw non-UTF-8 byte on stdout then crashes must be classified promptly
    # by its exit, NOT mislabeled KERNEL_STALL/TIMEOUT by a dead reader thread that stranded the main
    # loop on the silence budget. Without errors="replace" + the reader finally, the reader dies on the
    # decode and the loop waits out the full silence_timeout.
    argv = [
        sys.executable,
        "-c",
        "import os,sys; os.write(1, b'\\xff\\xfe not utf-8\\n'); sys.exit(7)",
    ]
    started = time.monotonic()
    result = execute_run_subprocess(
        _PLAN,
        run_id="binary-stdout",
        worker_argv=argv,
        out_dir=str(tmp_path),
        silence_timeout_s=30,
    )
    elapsed = time.monotonic() - started
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy not in {
        FailureTaxonomy.KERNEL_STALL,
        FailureTaxonomy.TIMEOUT,
    }
    assert elapsed < 30


def test_capture_worker_stderr_decision_is_tty_aware(monkeypatch):
    # #509: an explicit True/False is honored; the default (None) is tty-aware.
    from corpus_studio.platform.subprocess_supervisor import _capture_worker_stderr

    assert _capture_worker_stderr(True) is True
    assert _capture_worker_stderr(False) is False

    class _Interactive:
        def isatty(self):
            return True

    class _Piped:
        def isatty(self):
            return False

    monkeypatch.setattr(
        "corpus_studio.platform.subprocess_supervisor.sys.stderr", _Interactive()
    )
    assert _capture_worker_stderr(None) is False  # a tty -> inherit, keep live progress
    monkeypatch.setattr("corpus_studio.platform.subprocess_supervisor.sys.stderr", _Piped())
    assert _capture_worker_stderr(None) is True  # no tty (background/UI) -> capture

    # a stderr with no usable isatty() (None here -> AttributeError) fails safe to capture
    monkeypatch.setattr("corpus_studio.platform.subprocess_supervisor.sys.stderr", None)
    assert _capture_worker_stderr(None) is True


def test_worker_stderr_is_captured_to_the_run_record_when_requested(tmp_path):
    # #509: a background/UI run persists the child's stderr for post-mortem instead of losing it to a
    # terminal nobody is watching.
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('TORCH TRACEBACK boom\\n'); sys.stderr.flush(); sys.exit(5)",
    ]
    result = execute_run_subprocess(
        _PLAN,
        run_id="stderr-capture",
        worker_argv=argv,
        out_dir=str(tmp_path),
        capture_stderr=True,
        silence_timeout_s=10,
    )
    assert result.manifest.state == "failed"
    log = tmp_path / "runs" / result.manifest.run_id / "worker-stderr.log"
    assert log.exists()
    assert "TORCH TRACEBACK boom" in log.read_text(encoding="utf-8")


def test_worker_stderr_is_inherited_when_capture_disabled(tmp_path):
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('x\\n'); sys.exit(0)"]
    result = execute_run_subprocess(
        _PLAN,
        run_id="stderr-inherit",
        worker_argv=argv,
        out_dir=str(tmp_path),
        capture_stderr=False,
        silence_timeout_s=10,
    )
    log = tmp_path / "runs" / result.manifest.run_id / "worker-stderr.log"
    assert not log.exists()  # inherited, so no capture file is created
