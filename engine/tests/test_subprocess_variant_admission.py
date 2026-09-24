"""Variant-specific success evidence at the subprocess parent boundary (#860).

The parent used to re-derive success only for adapter SFT (``resolved_execution``): a succeeded
terminal for any newer variant (DPO, reward, full-parameter SFT, pretraining, on-policy RL) or for
an echo plan was admitted on identity checks alone, and ``run_accepted`` could bind only the SFT
hash.
These tests pin the fix: one central variant binding (``resolved_execution_binding``) drives the
pre-spawn seal check, the ``run_accepted`` echo on both sides, and a torch-free terminal admission
that requires exactly the dispatched variant's evidence family and re-verifies its saved bytes.

Handshakes and end-to-end forged terminals run against a real replaying child process; genuine
artifacts come from the REAL worker entrypoint with only the lane's ML training function and the
worker-side torch reload substituted. The per-forgery table drives the parent's terminal admission
directly so each rejection is attributable to one forged property. No torch or GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.abc
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import _variant_admission_fixtures as fx
from corpus_studio.platform.artifacts import build_artifact_manifest
from corpus_studio.platform.common import HashRef, MemoryMetrics, Ref
from corpus_studio.platform.contracts import (
    ArtifactManifest,
    FailureRecord,
    RunManifest,
    RunPlan,
    TerminalResultBody,
)
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import (
    RESOLVED_EXECUTION_FIELDS,
    ROLLOUT_NOT_EXECUTABLE_REASON,
    SUCCESS_EVIDENCE_FIELDS,
    ExecutionConfigurationError,
    execution_configuration_hash_for,
    required_runner_lane,
    resolved_execution_binding,
    run_scoped_training_output,
)
from corpus_studio.platform.planner import (
    compute_plan_hash,
    run_plan_hash_payload,
    verify_run_plan_hash,
)
from corpus_studio.platform.subprocess_supervisor import (
    _admit_terminal_success,
    _dispatch_line,
    _measured_failure_fit,
    _parse_terminal,
    execute_run_subprocess,
    worker_identity_argv,
)
from corpus_studio.platform.supervisor import RunnerFailure, demo_run_plan
from corpus_studio.platform.watchdog import reconcile_measured_fit
from corpus_studio.platform.worker import run_worker
from corpus_studio.platform.worker_protocol import WorkerProtocolError
from test_platform_subprocess import _fake_worker, _hello_body

TS = "2026-09-24T00:00:00+00:00"
HEAVY_MODULES = frozenset(
    {"torch", "transformers", "peft", "trl", "safetensors", "datasets", "bitsandbytes"}
)
PEAK = MemoryMetrics(
    torch_peak_reserved_bytes=1 << 30, dedicated_gpu_bytes=12 << 30, shared_gpu_bytes=0
)
# Lanes the subprocess parent can dispatch today, keyed by variant name -> sealed runner lane.
EXECUTABLE_LANES = {
    "preference": "preference",
    "reward": "reward",
    "full_finetune": "full_finetune",
    "pretraining": "pretraining_cpu_toy",
}
# Non-SFT variants the admission table covers; rollout is refused before spawn until on_policy_rl is
# promoted, so it is exercised at the terminal-admission level.
VARIANT_LANES = (*EXECUTABLE_LANES, "rollout")
EVIDENCE_FIELD = {
    "training": "training_success_evidence",
    "preference": "preference_success_evidence",
    "reward": "reward_success_evidence",
    "rollout": "rollout_success_evidence",
    "full_finetune": "full_finetune_success_evidence",
    "pretraining": "pretraining_success_evidence",
}


# ---- hash-valid plans for every resolved variant -------------------------------------------------


def _reseal(payload: dict[str, Any]) -> RunPlan:
    draft = RunPlan.model_validate(payload)
    plan = draft.model_copy(update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))})
    assert verify_run_plan_hash(plan)
    return plan


def _plan(lane: str, output_root: Path) -> RunPlan:
    """A sealed RunPlan for ``lane`` whose run-scoped outputs live under ``output_root``."""

    from corpus_studio.platform.runners import demo_training_plan

    root = str(output_root)
    sft = demo_training_plan()
    payload = sft.model_dump(mode="json")
    payload["export"]["output_dir"] = root
    if lane == "training":
        assert sft.resolved_execution is not None
        changed = sft.resolved_execution.model_copy(update={"output_dir": root})
        changed = changed.model_copy(
            update={"configuration_hash": execution_configuration_hash_for(changed)}
        )
        payload["resolved_execution"] = changed.model_dump(mode="json")
        return _reseal(payload)
    if lane == "preference":
        from test_preference_execution_config import _dpo_config, _dpo_plan_payload

        _sft, _dpo, payload = _dpo_plan_payload()
        payload["resolved_preference_execution"] = _dpo_config(output_dir=root).model_dump(
            mode="json"
        )
        payload["export"]["output_dir"] = root
        return _reseal(payload)
    payload["resolved_execution"] = None
    if lane == "reward":
        from test_reward_execution_config import _reward_config

        payload["resolved_reward_execution"] = _reward_config(output_dir=root).model_dump(
            mode="json"
        )
        payload["task_type"] = "reward"
    elif lane == "rollout":
        from test_rollout_execution_config import _rollout_config

        payload["resolved_rollout_execution"] = _rollout_config(output_dir=root).model_dump(
            mode="json"
        )
        payload["task_type"] = "grpo"
    elif lane == "full_finetune":
        from test_full_finetune_execution_config import _ff_config

        payload["resolved_full_finetune_execution"] = _ff_config(output_dir=root).model_dump(
            mode="json"
        )
        payload["task_type"] = "sft"
    else:
        from test_pretraining_execution_config import _pretrain_config

        config = _pretrain_config(output_dir=root)
        payload["resolved_pretraining_execution"] = config.model_dump(mode="json")
        payload["task_type"] = "pretraining"
        payload["precision"] = config.precision.forward_compute_dtype.value
        payload["quantization"] = config.precision.quantized_storage_format.value
        payload["loss_impl"] = config.loss_impl.value
        payload["checkpoint_policy"] = config.checkpoint_policy.model_dump(mode="json")
        payload["export"]["format"] = config.export_format.value
    return _reseal(payload)


def _binding(plan: RunPlan):
    binding = resolved_execution_binding(plan)
    assert binding is not None
    return binding


def _manifest(plan: RunPlan, rid: str, *, state: str = "succeeded", **extra: Any) -> RunManifest:
    return RunManifest(
        run_id=rid,
        plan_ref=Ref(id=plan.plan_id, hash=HashRef(value=plan.plan_hash)),
        environment_ref=plan.environment_ref,
        dataset_ref=plan.dataset_ref,
        created_at=TS,
        updated_at=TS,
        state=state,
        base_model=plan.base_model,
        target=plan.backend_ref.id,
        **extra,
    )


def _artifact(
    plan: RunPlan, rid: str, path: Path, kind: str, suffix: str = "0"
) -> ArtifactManifest:
    return build_artifact_manifest(
        artifact_id=f"{rid}-{kind}-{suffix}",
        path=str(path),
        kind=kind,
        run_id=rid,
        base_model=plan.base_model,
        now=TS,
    )


def _genuine_terminal(
    lane: str,
    tmp_path: Path,
    *,
    rid: str | None = None,
    peak: MemoryMetrics | None = None,
) -> tuple[RunPlan, str, RunManifest, list[ArtifactManifest]]:
    """A terminal exactly as execute_run builds it for ``lane``: genuine-shaped export bytes at the
    run-scoped path, the lane's matching evidence, and an integrity-checked ArtifactManifest."""

    plan = _plan(lane, tmp_path / "output-root")
    binding = _binding(plan)
    rid = rid or f"run-{lane}"
    out = run_scoped_training_output(binding.config, rid, leaf=binding.artifact_kind)
    evidence = fx.write_lane_export(lane, out, binding.config.schedule.max_steps or 1)
    if peak is not None:
        evidence = evidence.model_copy(update={"measured_peak": peak})
    artifact = _artifact(plan, rid, out, binding.artifact_kind)
    manifest = _manifest(
        plan,
        rid,
        output_dir=str(out),
        artifact_ids=[artifact.artifact_id],
        final_fit=reconcile_measured_fit(peak, proven=True) if peak is not None else None,
        **{binding.evidence_field: evidence},
    )
    return plan, rid, manifest, [artifact]


def _admit(
    plan: RunPlan,
    rid: str,
    manifest: RunManifest,
    artifacts: list[ArtifactManifest],
    admitted: list[ArtifactManifest] | None = None,
) -> RunManifest:
    # model_construct: the parent must hold on its own, even for a shape (two evidence families)
    # that contract validation would already refuse on the wire.
    body = TerminalResultBody.model_construct(
        run_id=rid,
        outcome=FailureTaxonomy.PASS,
        run_manifest=manifest,
        artifacts=artifacts,
        failure=None,
    )
    return _parse_terminal(body, plan, rid, [], admitted if admitted is not None else [])


def _terminal_message(rid: str, manifest: RunManifest, artifacts: list[ArtifactManifest]):
    failure = manifest.failure.model_dump(mode="json") if manifest.failure else None
    return (
        "terminal_result",
        {
            "run_id": rid,
            "outcome": "PASS" if failure is None else failure["taxonomy"],
            "run_manifest": manifest.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            "failure": failure,
        },
    )


def _accepted(rid: str, execution_hash: str | None):
    return (
        "run_accepted",
        {"run_id": rid, "pid": 1, "execution_configuration_hash": execution_hash},
    )


class _HeavyImportTripwire(importlib.abc.MetaPathFinder):
    """Records any attempt to import a heavy ML module while installed."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in HEAVY_MODULES:
            self.attempts.append(fullname)
        return None


# ---- the central binding -------------------------------------------------------------------------


def test_binding_table_covers_every_resolved_variant_and_evidence_family():
    # A new resolved variant (or evidence family) cannot ship without a parent admission row.
    assert set(RESOLVED_EXECUTION_FIELDS) == {
        name for name in RunPlan.model_fields if name.startswith("resolved_")
    }
    assert set(SUCCESS_EVIDENCE_FIELDS) == {
        name for name in RunManifest.model_fields if name.endswith("_success_evidence")
    }
    assert len(RESOLVED_EXECUTION_FIELDS) == len(SUCCESS_EVIDENCE_FIELDS) == 6


@pytest.mark.parametrize(
    ("lane", "plan_field", "artifact_kind"),
    [
        ("training", "resolved_execution", "adapter"),
        ("preference", "resolved_preference_execution", "adapter"),
        ("reward", "resolved_reward_execution", "adapter"),
        ("rollout", "resolved_rollout_execution", "adapter"),
        ("full_finetune", "resolved_full_finetune_execution", "model"),
        ("pretraining", "resolved_pretraining_execution", "model"),
    ],
)
def test_binding_selects_the_one_sealed_variant(tmp_path, lane, plan_field, artifact_kind):
    plan = _plan(lane, tmp_path)
    binding = _binding(plan)
    assert binding.plan_field == plan_field
    assert binding.evidence_field == EVIDENCE_FIELD[lane]
    assert binding.artifact_kind == artifact_kind
    assert binding.config is getattr(plan, plan_field)
    assert binding.configuration_hash == getattr(plan, plan_field).configuration_hash
    assert binding.verify_configuration_hash()
    tampered = dataclasses.replace(
        binding, config=binding.config.model_copy(update={"seed": binding.config.seed + 1})
    )
    assert not tampered.verify_configuration_hash()


def test_echo_plan_has_no_binding_and_two_authorities_are_refused(tmp_path):
    assert resolved_execution_binding(demo_run_plan()) is None
    sft = _plan("training", tmp_path)
    dpo = _plan("preference", tmp_path)
    doubled = sft.model_copy(
        update={"resolved_preference_execution": dpo.resolved_preference_execution}
    )
    with pytest.raises(ExecutionConfigurationError, match="more than one resolved execution"):
        resolved_execution_binding(doubled)


# ---- worker side: run_accepted echoes the selected variant hash (WORKER CHANGE) ------------------


def _run_worker_in_memory(plan: RunPlan, runner: str) -> tuple[int, list[dict[str, Any]]]:
    out = io.StringIO()
    rc = run_worker(
        _dispatch_line(plan, "run-echo-hash", 30),
        runner_name=runner,
        backend_id=plan.backend_ref.id,
        environment_ref=plan.environment_ref,
        out=out,
    )
    return rc, [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


@pytest.mark.parametrize("lane", sorted(EXECUTABLE_LANES))
def test_worker_run_accepted_echoes_the_selected_variant_hash(monkeypatch, tmp_path, lane):
    plan = _plan(lane, tmp_path)
    runner = EXECUTABLE_LANES[lane]

    class _StubRunner:
        name = runner

        def run(self, ctx: Any) -> list[Any]:
            raise AssertionError("the runner-type gate must refuse a non-first-party runner")

    # No ML import: execute_run's first-party runner-type gate refuses the stub after acceptance.
    monkeypatch.setattr(
        "corpus_studio.platform.worker._build_runner", lambda *_args, **_kwargs: _StubRunner()
    )
    rc, messages = _run_worker_in_memory(plan, runner)
    assert rc == 0
    accepted = next(item for item in messages if item["type"] == "run_accepted")
    assert accepted["body"]["execution_configuration_hash"] == _binding(plan).configuration_hash
    assert messages[-1]["type"] == "terminal_result"


def test_worker_echo_plan_accepts_with_a_null_hash():
    rc, messages = _run_worker_in_memory(demo_run_plan(), "echo")
    assert rc == 0
    accepted = next(item for item in messages if item["type"] == "run_accepted")
    # The wire encoding omits null fields, so an absent key IS the null echo.
    assert accepted["body"].get("execution_configuration_hash") is None


def test_worker_rejects_the_not_yet_executable_rollout_lane(tmp_path):
    rc, messages = _run_worker_in_memory(_plan("rollout", tmp_path), "rollout")
    assert rc == 2
    assert [item["type"] for item in messages] == ["run_rejected"]
    assert ROLLOUT_NOT_EXECUTABLE_REASON[:40] in messages[0]["body"]["message"]


@pytest.mark.parametrize(
    ("lane", "runner", "expected"),
    [
        ("training", "cpu_toy", "resolved execution configuration hash mismatch"),
        ("preference", "preference", "resolved preference execution configuration hash mismatch"),
    ],
)
def test_worker_rejects_a_variant_whose_seal_does_not_reverify(
    monkeypatch, tmp_path, lane, runner, expected
):
    # A dispatched plan is re-validated on parse, so force the seal re-check itself to fail.
    import corpus_studio.platform.execution_config as execution_config

    real = execution_config.resolved_execution_binding
    monkeypatch.setattr(
        execution_config,
        "resolved_execution_binding",
        lambda plan: dataclasses.replace(real(plan), verifier=lambda _config: False),
    )
    monkeypatch.setattr(
        "corpus_studio.platform.worker._build_runner",
        lambda *_args, **_kwargs: pytest.fail("runner must not be built for a broken seal"),
    )
    rc, messages = _run_worker_in_memory(_plan(lane, tmp_path), runner)
    assert rc == 2
    assert [item["type"] for item in messages] == ["run_rejected"]
    assert expected in messages[0]["body"]["message"]


# ---- parent entry: every variant's seal and schedule are enforced before spawn -------------------


def _marker_worker(marker: Path) -> list[str]:
    return [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('spawned')"]


@pytest.mark.parametrize("lane", VARIANT_LANES)
def test_parent_refuses_a_broken_variant_seal_before_spawn(tmp_path, lane):
    plan = _plan(lane, tmp_path / "output-root")
    binding = _binding(plan)
    config = binding.config.model_copy(update={"seed": binding.config.seed + 1})
    draft = plan.model_copy(update={binding.plan_field: config})
    tampered = draft.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))}
    )
    assert verify_run_plan_hash(tampered)  # the plan seal alone does not catch it
    marker = tmp_path / "spawned"
    result = execute_run_subprocess(
        tampered, worker_argv=_marker_worker(marker), out_dir=tmp_path / "records"
    )
    failure = result.manifest.failure
    assert failure is not None
    assert failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert f"resolved {binding.label} execution configuration hash verification failed" == (
        failure.message
    )
    assert not marker.exists()
    durable = tmp_path / "records" / "runs" / result.manifest.run_id / "RunManifest.json"
    assert json.loads(durable.read_text(encoding="utf-8"))["state"] == "failed"


def test_parent_refuses_a_plan_with_two_execution_authorities_before_spawn(tmp_path):
    sft = _plan("training", tmp_path)
    doubled = sft.model_copy(
        update={
            "resolved_preference_execution": _plan(
                "preference", tmp_path
            ).resolved_preference_execution
        }
    )
    doubled = doubled.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(doubled))}
    )
    marker = tmp_path / "spawned"
    result = execute_run_subprocess(doubled, worker_argv=_marker_worker(marker))
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "more than one resolved execution" in result.manifest.failure.message
    assert not marker.exists()


@pytest.mark.parametrize("lane", sorted(EXECUTABLE_LANES))
def test_parent_refuses_a_max_steps_override_for_every_variant(tmp_path, lane):
    plan = _plan(lane, tmp_path / "output-root")
    sealed = _binding(plan).config.schedule.max_steps
    assert sealed is not None
    marker = tmp_path / "spawned"
    refused = execute_run_subprocess(
        plan, max_steps=sealed + 7, worker_argv=_marker_worker(marker), silence_timeout_s=10
    )
    assert refused.manifest.failure is not None
    assert refused.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "max_steps cannot override the sealed execution schedule" in (
        refused.manifest.failure.message
    )
    assert not marker.exists()
    # The sealed value itself is not an override: the worker is spawned.
    allowed = execute_run_subprocess(
        plan, max_steps=sealed, worker_argv=_marker_worker(marker), silence_timeout_s=10
    )
    assert marker.exists()
    assert allowed.manifest.failure is not None
    assert "max_steps" not in allowed.manifest.failure.message


# ---- parent handshake: run_accepted is bound to the dispatched variant ---------------------------


def _failed_terminal_messages(plan: RunPlan, rid: str, accepted_hash: str | None):
    failure = FailureRecord(run_id=rid, taxonomy=FailureTaxonomy.FAIL, message="stop")
    manifest = _manifest(plan, rid, state="failed", failure=failure)
    return [_accepted(rid, accepted_hash), _terminal_message(rid, manifest, [])]


@pytest.mark.parametrize("lane", ["training", *sorted(EXECUTABLE_LANES)])
@pytest.mark.parametrize("echo", ["variant", "absent", "other_variant"])
def test_parent_binds_run_accepted_to_the_dispatched_variant(tmp_path, lane, echo):
    plan = _plan(lane, tmp_path / "output-root")
    other = _plan("preference" if lane == "training" else "training", tmp_path / "other")
    accepted_hash = {
        "variant": _binding(plan).configuration_hash,
        "absent": None,
        # A pre-change worker echoes the SFT hash only; any other variant's hash is equally foreign.
        "other_variant": _binding(other).configuration_hash,
    }[echo]
    rid = f"run-{lane}-{echo}"
    result = execute_run_subprocess(
        plan,
        run_id=rid,
        worker_argv=_fake_worker(
            _failed_terminal_messages(plan, rid, accepted_hash), hello_body=_hello_body(plan)
        ),
        silence_timeout_s=10,
    )
    failure = result.manifest.failure
    assert failure is not None
    if echo == "variant":
        # Accepted: the worker's own classified failure flows through unchanged.
        assert failure.taxonomy == FailureTaxonomy.FAIL
        assert failure.message == "stop"
    else:
        assert failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
        assert "execution configuration hash does not match the dispatched" in failure.message
        assert _binding(plan).label in failure.message


def test_parent_requires_a_null_accepted_hash_for_an_echo_plan():
    plan = demo_run_plan()
    rid = "run-echo-hash"
    result = execute_run_subprocess(
        plan,
        run_id=rid,
        worker_argv=_fake_worker(_failed_terminal_messages(plan, rid, "f" * 64)),
        silence_timeout_s=10,
    )
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert "must be null for a plan without a resolved execution" in (
        result.manifest.failure.message
    )


# ---- genuine artifacts pass the parent, through the REAL worker, without a heavy import ----------


@pytest.mark.parametrize("lane", sorted(EXECUTABLE_LANES))
def test_genuine_variant_success_is_admitted_through_the_real_worker(tmp_path, lane):
    plan = _plan(lane, tmp_path / "output-root")
    runner = required_runner_lane(plan)
    assert runner == EXECUTABLE_LANES[lane]
    records = tmp_path / "records"
    rid = f"run-{lane}-genuine"
    tripwire = _HeavyImportTripwire()
    before = set(sys.modules)
    sys.meta_path.insert(0, tripwire)
    try:
        result = execute_run_subprocess(
            plan,
            run_id=rid,
            runner_name="auto",
            worker_argv=[
                sys.executable,
                "-c",
                fx.child_worker_script(runner, worker_identity_argv(plan)),
            ],
            out_dir=records,
            silence_timeout_s=120,
            capture_stderr=True,
        )
    finally:
        sys.meta_path.remove(tripwire)
    stderr_log = records / "runs" / rid / "worker-stderr.log"
    assert result.manifest.state == "succeeded", (
        result.manifest.failure,
        stderr_log.read_text(encoding="utf-8") if stderr_log.exists() else "",
    )
    assert tripwire.attempts == []
    assert not {name.split(".")[0] for name in set(sys.modules) - before} & HEAVY_MODULES
    field = EVIDENCE_FIELD[lane]
    assert getattr(result.manifest, field) is not None
    assert all(
        getattr(result.manifest, name) is None for name in SUCCESS_EVIDENCE_FIELDS if name != field
    )
    assert [artifact.kind for artifact in result.artifacts] == [_binding(plan).artifact_kind]
    persisted = records / "runs" / rid / "artifacts" / f"{result.artifacts[0].artifact_id}.json"
    assert persisted.exists()
    durable = json.loads((records / "runs" / rid / "RunManifest.json").read_text(encoding="utf-8"))
    assert durable["state"] == "succeeded"
    assert durable["run_id"] == rid


def test_parent_admission_imports_no_heavy_module_in_a_fresh_interpreter(tmp_path):
    # The definitive boundary check: a clean interpreter with an import tripwire runs the full
    # parent admission of genuine adapter- and model-kind terminals; no heavy import is attempted.
    cases = []
    for lane in ("preference", "pretraining"):
        plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path / lane, peak=PEAK)
        body = TerminalResultBody(
            run_id=rid, outcome=FailureTaxonomy.PASS, run_manifest=manifest, artifacts=artifacts
        )
        cases.append(
            {"plan": plan.model_dump(mode="json"), "rid": rid, "body": body.model_dump(mode="json")}
        )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    script = (
        "import importlib.abc, json, sys\n"
        f"HEAVY = {sorted(HEAVY_MODULES)!r}\n"
        "attempts = []\n"
        "class Tripwire(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in HEAVY:\n"
        "            attempts.append(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Tripwire())\n"
        "from corpus_studio.platform.contracts import RunPlan, TerminalResultBody\n"
        "from corpus_studio.platform.subprocess_supervisor import _parse_terminal\n"
        f"cases = json.load(open({str(cases_path)!r}, encoding='utf-8'))\n"
        "states = []\n"
        "for case in cases:\n"
        "    plan = RunPlan.model_validate(case['plan'])\n"
        "    body = TerminalResultBody.model_validate(case['body'])\n"
        "    admitted = []\n"
        "    states.append(_parse_terminal(body, plan, case['rid'], [], admitted).state)\n"
        "    assert len(admitted) == 1\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in HEAVY)\n"
        "print(json.dumps({'states': states, 'attempts': attempts, 'heavy': loaded}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "states": ["succeeded", "succeeded"],
        "attempts": [],
        "heavy": [],
    }


# ---- forged terminals: every rejected before admission -------------------------------------------


def _weights(binding) -> str:
    if binding.artifact_kind == "adapter":
        return "adapter_model.safetensors"
    return "model.safetensors"


def _config_file(binding) -> str:
    return "adapter_config.json" if binding.artifact_kind == "adapter" else "config.json"


def _digest_fields(binding) -> tuple[str, str, str]:
    if binding.artifact_kind == "adapter":
        return "adapter_safetensors_sha256", "adapter_config_sha256", "adapter_export_state"
    return "model_safetensors_sha256", "model_config_sha256", "model_export_state"


def _with_evidence(manifest: RunManifest, binding, **update: Any) -> RunManifest:
    evidence = getattr(manifest, binding.evidence_field)
    return manifest.model_copy(update={binding.evidence_field: evidence.model_copy(update=update)})


def _with_export(manifest: RunManifest, binding, **update: Any) -> RunManifest:
    evidence = getattr(manifest, binding.evidence_field)
    export_field = _digest_fields(binding)[2]
    execution = evidence.execution.model_copy(
        update={export_field: getattr(evidence.execution, export_field).model_copy(update=update)}
    )
    return _with_evidence(manifest, binding, execution=execution)


def _rebuild(plan, rid, manifest, artifacts, binding):
    """Re-seal the artifact integrity and evidence digests over the CURRENT bytes, so only the
    property under test (not a stale hash) can reject the terminal."""

    root = Path(artifacts[0].path)
    rebuilt = _artifact(plan, rid, root, binding.artifact_kind)
    weights_field = _digest_fields(binding)[0]
    target = root / _weights(binding)
    digest = hashlib.sha256(target.resolve().read_bytes()).hexdigest()
    return _with_evidence(manifest, binding, **{weights_field: digest}), [rebuilt]


def _forge(case: str, plan, rid, manifest, artifacts, tmp_path):
    binding = _binding(plan)
    root = Path(artifacts[0].path)
    weights_field, config_field, _export_field = _digest_fields(binding)
    evidence = getattr(manifest, binding.evidence_field)
    if case == "missing_evidence":
        return manifest.model_copy(update={binding.evidence_field: None}), artifacts
    if case == "foreign_family":
        # The same proof filed under another family: the sibling full-model family for a model
        # lane (identical evidence type), the adapter-SFT family for an adapter lane.
        foreign = {
            "pretraining_success_evidence": "full_finetune_success_evidence",
            "full_finetune_success_evidence": "pretraining_success_evidence",
        }.get(binding.evidence_field, "training_success_evidence")
        return (
            manifest.model_copy(update={binding.evidence_field: None, foreign: evidence}),
            artifacts,
        )
    if case == "extra_family":
        foreign = (
            "reward_success_evidence"
            if binding.evidence_field == "rollout_success_evidence"
            else "rollout_success_evidence"
        )
        return manifest.model_copy(update={foreign: evidence}), artifacts
    if case == "no_artifacts":
        return manifest.model_copy(update={"artifact_ids": []}), []
    if case == "two_artifacts":
        second = artifacts[0].model_copy(update={"artifact_id": f"{rid}-second"})
        return (
            manifest.model_copy(update={"artifact_ids": [*manifest.artifact_ids, f"{rid}-second"]}),
            [*artifacts, second],
        )
    if case == "wrong_kind":
        return manifest, [artifacts[0].model_copy(update={"kind": "checkpoint"})]
    if case == "rogue_path":
        rogue = tmp_path / "rogue" / root.name
        shutil.copytree(root, rogue)
        return manifest, [artifacts[0].model_copy(update={"path": str(rogue)})]
    if case == "output_dir_mismatch":
        return manifest.model_copy(update={"output_dir": str(tmp_path)}), artifacts
    if case == "deleted_output":
        shutil.rmtree(root)
        return manifest, artifacts
    if case in {"integrity_none", "integrity_modified", "no_content_hash", "no_metadata_hash"}:
        integrity = artifacts[0].integrity
        assert integrity is not None
        changed = {
            "integrity_none": None,
            "integrity_modified": integrity.model_copy(update={"current_integrity": "modified"}),
            "no_content_hash": integrity.model_copy(update={"content_hash": None}),
            "no_metadata_hash": integrity.model_copy(update={"metadata_hash": None}),
        }[case]
        return manifest, [artifacts[0].model_copy(update={"integrity": changed})]
    if case == "altered_weight_bytes":
        weights = root / _weights(binding)
        raw = bytearray(weights.read_bytes())
        raw[-1] ^= 0xFF  # same size, so only a byte-exact hash can notice
        weights.write_bytes(bytes(raw))
        return manifest, artifacts
    if case == "wrong_weights_digest":
        return _with_evidence(manifest, binding, **{weights_field: "0" * 64}), artifacts
    if case == "wrong_config_digest":
        return _with_evidence(manifest, binding, **{config_field: "0" * 64}), artifacts
    if case == "altered_config_bytes":
        config = root / _config_file(binding)
        config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return manifest, artifacts
    if case == "missing_config":
        (root / _config_file(binding)).unlink()
        return manifest, artifacts
    if case == "wrong_tensor_state":
        return _with_export(manifest, binding, after_sha256="e" * 64), artifacts
    if case == "wrong_tensor_names":
        export = getattr(evidence.execution, _digest_fields(binding)[2])
        reordered = list(reversed(export.tensor_names))
        return _with_export(manifest, binding, tensor_names=reordered), artifacts
    if case == "invalid_safetensors":
        (root / _weights(binding)).write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{garbage")
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "linked_weights":
        outside = tmp_path / "outside.safetensors"
        shutil.copyfile(root / _weights(binding), outside)
        (root / _weights(binding)).unlink()
        os.symlink(outside, root / _weights(binding))
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "nested_weights":
        nested = root / "nested"
        nested.mkdir()
        fx.write_safetensors(nested / "extra.safetensors", {"x": 1.0})
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "linked_extra_file":
        # A second weights payload that is a link escaping the run scope: the content hash would
        # follow it and bind bytes that live outside the artifact.
        outside = tmp_path / "outside-secret.bin"
        outside.write_bytes(b"bytes living outside the run scope")
        os.symlink(outside, root / "pytorch_model.bin")
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "linked_directory":
        outside_dir = tmp_path / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "notes.txt").write_text("outside", encoding="utf-8")
        os.symlink(outside_dir, root / "linked", target_is_directory=True)
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "checkpoint_directory":
        (root / "checkpoint-1").mkdir()
        (root / "checkpoint-1" / "trainer_state.json").write_text("{}", encoding="utf-8")
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case in {"alternate_bin", "sharding_index", "extra_shard", "tf_alternate", "flax_alternate"}:
        name = {
            "alternate_bin": "pytorch_model.bin",
            "sharding_index": f"{_weights(binding)}.index.json",
            "extra_shard": "model-00002-of-00002.safetensors",
            "tf_alternate": "tf_model.h5",
            "flax_alternate": "flax_model.msgpack",
        }[case]
        (root / name).write_bytes(b"x")
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "hardlinked_training_args":
        outside = tmp_path / "outside-training-args.bin"
        outside.write_bytes(b"x")
        os.link(outside, root / "training_args.bin")
        return _rebuild(plan, rid, manifest, artifacts, binding)
    if case == "schedule_mismatch":
        execution = evidence.execution.model_copy(
            update={"completed_optimizer_steps": evidence.execution.completed_optimizer_steps + 1}
        )
        return _with_evidence(manifest, binding, execution=execution), artifacts
    if case == "fit_without_peak":
        return (
            manifest.model_copy(update={"final_fit": reconcile_measured_fit(PEAK, proven=True)}),
            artifacts,
        )
    if case == "peak_with_mismatched_fit":
        forged = _with_evidence(manifest, binding, measured_peak=PEAK)
        return (
            forged.model_copy(update={"final_fit": reconcile_measured_fit(PEAK, proven=False)}),
            artifacts,
        )
    raise AssertionError(case)


_EXPORT = (FailureTaxonomy.ARTIFACT_FAILURE, StageMarker.export)
# case -> (lanes it applies to, expected exception, (taxonomy, stage) or None, message substring)
_FORGERIES: dict[str, tuple[tuple[str, ...], type[Exception], Any, str]] = {
    "missing_evidence": (
        VARIANT_LANES,
        RunnerFailure,
        (FailureTaxonomy.UPDATE_FAILURE, StageMarker.optimizer_step),
        "has no sealed",
    ),
    "foreign_family": (VARIANT_LANES, WorkerProtocolError, None, "admits only"),
    "extra_family": (VARIANT_LANES, WorkerProtocolError, None, "admits only"),
    "no_artifacts": (VARIANT_LANES, RunnerFailure, _EXPORT, "exactly one"),
    "two_artifacts": (VARIANT_LANES, RunnerFailure, _EXPORT, "exactly one"),
    "wrong_kind": (VARIANT_LANES, RunnerFailure, _EXPORT, "exactly one"),
    "rogue_path": (VARIANT_LANES, RunnerFailure, _EXPORT, "run-scoped output"),
    "output_dir_mismatch": (VARIANT_LANES, RunnerFailure, _EXPORT, "run-scoped output"),
    "deleted_output": (VARIANT_LANES, RunnerFailure, _EXPORT, "run-scoped output"),
    "integrity_none": (VARIANT_LANES, RunnerFailure, _EXPORT, "no integrity-checked"),
    "integrity_modified": (VARIANT_LANES, RunnerFailure, _EXPORT, "no integrity-checked"),
    "no_content_hash": (VARIANT_LANES, RunnerFailure, _EXPORT, "no integrity-checked"),
    "no_metadata_hash": (fx.ADAPTER_LANES, RunnerFailure, _EXPORT, "no integrity-checked"),
    "altered_weight_bytes": (VARIANT_LANES, RunnerFailure, _EXPORT, "weight bytes do not match"),
    "wrong_weights_digest": (
        VARIANT_LANES,
        RunnerFailure,
        _EXPORT,
        "Safetensors bytes do not match the proposed digest",
    ),
    "wrong_config_digest": (
        VARIANT_LANES,
        RunnerFailure,
        _EXPORT,
        "config bytes do not match the proposed digest",
    ),
    "altered_config_bytes": (VARIANT_LANES, RunnerFailure, _EXPORT, "config bytes"),
    "missing_config": (VARIANT_LANES, RunnerFailure, _EXPORT, "failed admission"),
    "wrong_tensor_state": (VARIANT_LANES, RunnerFailure, _EXPORT, "tensor state differs"),
    "wrong_tensor_names": (VARIANT_LANES, RunnerFailure, _EXPORT, "tensor state differs"),
    "invalid_safetensors": (VARIANT_LANES, RunnerFailure, _EXPORT, "Safetensors is invalid"),
    "linked_weights": (VARIANT_LANES, RunnerFailure, _EXPORT, "linked or irregular file"),
    "nested_weights": (VARIANT_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "linked_extra_file": (VARIANT_LANES, RunnerFailure, _EXPORT, "linked or irregular file"),
    "linked_directory": (VARIANT_LANES, RunnerFailure, _EXPORT, "linked or irregular directory"),
    "checkpoint_directory": (VARIANT_LANES, RunnerFailure, _EXPORT, "intermediate checkpoint"),
    "alternate_bin": (VARIANT_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "sharding_index": (VARIANT_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "extra_shard": (VARIANT_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "tf_alternate": (fx.MODEL_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "flax_alternate": (fx.MODEL_LANES, RunnerFailure, _EXPORT, "alternate or nested"),
    "hardlinked_training_args": (VARIANT_LANES, RunnerFailure, _EXPORT, "hard-linked"),
    "schedule_mismatch": (
        VARIANT_LANES,
        RunnerFailure,
        (FailureTaxonomy.OPTIMIZER_FAILURE, StageMarker.optimizer_step),
        "sealed schedule",
    ),
    "fit_without_peak": (VARIANT_LANES, RunnerFailure, _EXPORT, "raw peak-memory evidence"),
    "peak_with_mismatched_fit": (VARIANT_LANES, RunnerFailure, _EXPORT, "parent-reconstructed"),
}
_FORGERY_CASES = [
    (lane, case) for case, (lanes, *_rest) in _FORGERIES.items() for lane in lanes
]


@pytest.mark.parametrize(("lane", "case"), _FORGERY_CASES)
def test_forged_success_terminal_is_rejected(tmp_path, lane, case):
    plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path)
    # The unforged terminal is admitted, so each rejection below is caused by the forgery alone.
    assert _admit(plan, rid, manifest, artifacts).state == "succeeded"
    _lanes, error, classification, message = _FORGERIES[case]
    forged_manifest, forged_artifacts = _forge(case, plan, rid, manifest, artifacts, tmp_path)
    admitted: list[ArtifactManifest] = []
    with pytest.raises(error) as info:
        _admit(plan, rid, forged_manifest, forged_artifacts, admitted)
    assert message in str(info.value)
    if classification is not None:
        assert isinstance(info.value, RunnerFailure)
        assert (info.value.taxonomy, info.value.stage) == classification
    assert admitted == []  # nothing from a rejected terminal is queued for persistence


@pytest.mark.parametrize("lane", VARIANT_LANES)
def test_export_tree_policy_runs_before_any_weight_byte_is_hashed(monkeypatch, tmp_path, lane):
    # The content hash follows linked files, so a link escaping the run scope must be refused
    # before hashing can read (and bind) bytes that live outside the artifact.
    import corpus_studio.training.artifact_registry as artifact_registry

    plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path)
    forged_manifest, forged_artifacts = _forge(
        "linked_extra_file", plan, rid, manifest, artifacts, tmp_path
    )

    def refuse_hashing(path: str) -> str | None:
        raise AssertionError(f"weight bytes were hashed before the tree policy ran: {path}")

    monkeypatch.setattr(artifact_registry, "compute_weight_content_hash", refuse_hashing)
    with pytest.raises(RunnerFailure, match="artifact failed admission"):
        _admit(plan, rid, forged_manifest, forged_artifacts)


@pytest.mark.parametrize("lane", VARIANT_LANES)
def test_benign_export_metadata_is_admitted(tmp_path, lane):
    # What save_pretrained, tokenizer.save_pretrained and Trainer.save_model genuinely write next
    # to the one root Safetensors file must not false-fail the tree policy.
    plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path)
    binding = _binding(plan)
    root = Path(artifacts[0].path)
    (root / "training_args.bin").write_bytes(b"\x80\x04}\x94.")  # never deserialized
    for name in ("generation_config.json", "tokenizer_config.json", "special_tokens_map.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "tokenizer.model").write_bytes(b"sentencepiece")
    (root / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
    rebuilt_manifest, rebuilt = _rebuild(plan, rid, manifest, artifacts, binding)
    admitted: list[ArtifactManifest] = []
    assert _admit(plan, rid, rebuilt_manifest, rebuilt, admitted).state == "succeeded"
    assert admitted == rebuilt


@pytest.mark.parametrize("lane", VARIANT_LANES)
def test_genuine_success_with_a_measured_peak_is_admitted_with_its_proven_fit(tmp_path, lane):
    plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path, peak=PEAK)
    admitted: list[ArtifactManifest] = []
    result = _admit(plan, rid, manifest, artifacts, admitted)
    assert result.final_fit == reconcile_measured_fit(PEAK, proven=True)
    assert admitted == artifacts


def test_epoch_scheduled_variant_requires_at_least_one_completed_step(tmp_path):
    plan, rid, manifest, artifacts = _genuine_terminal("preference", tmp_path)
    binding = _binding(plan)
    schedule = binding.config.schedule.model_copy(update={"max_steps": None, "num_train_epochs": 1})
    epoch_binding = dataclasses.replace(
        binding, config=binding.config.model_copy(update={"schedule": schedule})
    )
    _admit_terminal_success(epoch_binding, plan, rid, manifest, artifacts, [])
    evidence = manifest.preference_success_evidence
    assert evidence is not None
    zero = manifest.model_copy(
        update={
            "preference_success_evidence": evidence.model_copy(
                update={
                    "execution": evidence.execution.model_copy(
                        update={"completed_optimizer_steps": 0}
                    )
                }
            )
        }
    )
    with pytest.raises(RunnerFailure, match="epoch-scheduled preference admitted zero") as info:
        _admit_terminal_success(epoch_binding, plan, rid, zero, artifacts, [])
    assert info.value.taxonomy == FailureTaxonomy.OPTIMIZER_FAILURE


@pytest.mark.parametrize("lane", sorted(EXECUTABLE_LANES))
@pytest.mark.parametrize("forgery", ["evidence_free", "altered_bytes"])
def test_forged_terminal_through_a_real_child_leaves_failed_durable_truth(
    tmp_path, lane, forgery
):
    plan, rid, manifest, artifacts = _genuine_terminal(lane, tmp_path)
    binding = _binding(plan)
    if forgery == "evidence_free":
        # The review's reproduction: a bare PASS with no evidence and no artifacts.
        manifest = _manifest(plan, rid)
        artifacts = []
    else:
        weights = Path(artifacts[0].path) / _weights(binding)
        raw = bytearray(weights.read_bytes())
        raw[-1] ^= 0xFF
        weights.write_bytes(bytes(raw))
    records = tmp_path / "records"
    result = execute_run_subprocess(
        plan,
        run_id=rid,
        worker_argv=_fake_worker(
            [
                _accepted(rid, binding.configuration_hash),
                _terminal_message(rid, manifest, artifacts),
            ],
            hello_body=_hello_body(plan),
        ),
        out_dir=records,
        silence_timeout_s=10,
    )
    failure = result.manifest.failure
    assert result.manifest.state == "failed"
    assert failure is not None
    assert failure.taxonomy == (
        FailureTaxonomy.UPDATE_FAILURE
        if forgery == "evidence_free"
        else FailureTaxonomy.ARTIFACT_FAILURE
    )
    assert all(getattr(result.manifest, name) is None for name in SUCCESS_EVIDENCE_FIELDS)
    assert result.artifacts == []
    assert not list((records / "runs" / rid).glob("artifacts/*.json"))
    durable = json.loads((records / "runs" / rid / "RunManifest.json").read_text(encoding="utf-8"))
    assert durable["state"] == "failed"


# ---- adapter SFT keeps its admission; a foreign family is a protocol violation -------------------


def _genuine_sft_terminal(monkeypatch, tmp_path):
    from test_platform_runners import _fake_run_training

    from corpus_studio.platform.runners import TrainingRunner, demo_training_plan
    from corpus_studio.platform.supervisor import execute_run

    # The demo plan seals a relative output root; anchor it (and the parent's re-check) in tmp_path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
    plan = demo_training_plan()
    result = execute_run(plan, TrainingRunner(cpu_toy=True), run_id="run-sft", clock=lambda: TS)
    assert result.manifest.state == "succeeded"
    body = TerminalResultBody(
        run_id="run-sft",
        outcome=FailureTaxonomy.PASS,
        run_manifest=result.manifest,
        artifacts=result.artifacts,
    )
    return plan, body, result.events


def test_genuine_sft_terminal_is_still_admitted_by_the_parent(monkeypatch, tmp_path):
    plan, body, events = _genuine_sft_terminal(monkeypatch, tmp_path)
    admitted: list[ArtifactManifest] = []
    manifest = _parse_terminal(body, plan, "run-sft", events, admitted)
    assert manifest.state == "succeeded"
    assert manifest.training_success_evidence is not None
    assert [artifact.kind for artifact in admitted] == ["adapter"]


def test_sft_terminal_with_a_foreign_family_is_a_protocol_violation(monkeypatch, tmp_path):
    plan, body, events = _genuine_sft_terminal(monkeypatch, tmp_path)
    _p, _r, dpo_manifest, _a = _genuine_terminal("preference", tmp_path / "dpo")
    forged = body.run_manifest.model_copy(
        update={
            "training_success_evidence": None,
            "preference_success_evidence": dpo_manifest.preference_success_evidence,
        }
    )
    with pytest.raises(WorkerProtocolError, match="admits only training_success_evidence"):
        _parse_terminal(
            body.model_copy(update={"run_manifest": forged}), plan, "run-sft", events, []
        )


def test_sft_evidence_that_differs_from_reconstruction_is_rejected(monkeypatch, tmp_path):
    plan, body, events = _genuine_sft_terminal(monkeypatch, tmp_path)
    evidence = body.run_manifest.training_success_evidence
    assert evidence is not None
    forged = body.run_manifest.model_copy(
        update={
            "training_success_evidence": evidence.model_copy(
                update={"adapter_config_sha256": "0" * 64}
            )
        }
    )
    with pytest.raises(RunnerFailure, match="does not match reconstructed admission"):
        _parse_terminal(
            body.model_copy(update={"run_manifest": forged}), plan, "run-sft", events, []
        )


# ---- echo and non-success terminals --------------------------------------------------------------


def test_echo_success_cannot_claim_evidence_artifacts_or_a_fit(tmp_path):
    plan = demo_run_plan()
    rid = "run-echo"
    assert _admit(plan, rid, _manifest(plan, rid), []).state == "succeeded"
    _p, _r, dpo_manifest, dpo_artifacts = _genuine_terminal("preference", tmp_path)
    with pytest.raises(WorkerProtocolError, match="admits no success evidence"):
        _admit(
            plan,
            rid,
            _manifest(
                plan, rid, preference_success_evidence=dpo_manifest.preference_success_evidence
            ),
            [],
        )
    artifact = dpo_artifacts[0].model_copy(
        update={
            "artifact_id": f"{rid}-adapter",
            "producer_run_ref": Ref(id=rid),
            "base_model": plan.base_model,
        }
    )
    with pytest.raises(RunnerFailure, match="cannot claim artifacts or a measured fit"):
        _admit(plan, rid, _manifest(plan, rid, artifact_ids=[artifact.artifact_id]), [artifact])
    with pytest.raises(RunnerFailure, match="cannot claim artifacts or a measured fit"):
        _admit(
            plan,
            rid,
            _manifest(plan, rid, final_fit=reconcile_measured_fit(PEAK, proven=False)),
            [],
        )


@pytest.mark.parametrize("state", ["failed", "cancelled"])
@pytest.mark.parametrize("lane", ["echo", "training", *VARIANT_LANES])
def test_non_success_terminals_need_no_success_evidence(tmp_path, lane, state):
    plan = demo_run_plan() if lane == "echo" else _plan(lane, tmp_path)
    rid = f"run-{lane}-{state}"
    failure = FailureRecord(run_id=rid, taxonomy=FailureTaxonomy.FAIL, message=state)
    manifest = _manifest(plan, rid, state=state, failure=failure)
    body = TerminalResultBody(
        run_id=rid,
        outcome=FailureTaxonomy.FAIL,
        run_manifest=manifest,
        artifacts=[],
        failure=failure,
    )
    assert _parse_terminal(body, plan, rid, [], []).state == state


# ---- persistence of admitted artifacts (migrated off the echo plan, which can claim none) --------


def _replay_genuine_dpo(tmp_path, *, peak=None, monkeypatch=None, fail_persistence=False):
    plan, rid, manifest, artifacts = _genuine_terminal("preference", tmp_path, peak=peak)
    if fail_persistence:
        assert monkeypatch is not None
        monkeypatch.setattr(
            "corpus_studio.platform.subprocess_supervisor.write_artifact_manifest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
    records = tmp_path / "records"
    result = execute_run_subprocess(
        plan,
        run_id=rid,
        worker_argv=_fake_worker(
            [
                _accepted(rid, _binding(plan).configuration_hash),
                _terminal_message(rid, manifest, artifacts),
            ],
            hello_body=_hello_body(plan),
        ),
        out_dir=records,
        silence_timeout_s=10,
    )
    return rid, artifacts, records, result


def test_subprocess_persists_the_childs_admitted_artifact_manifests(tmp_path):
    # The child builds ArtifactManifests but (running execute_run without out_dir) does not write
    # them; the PARENT persists them under --out once admitted, so artifact_ids never dangle.
    rid, artifacts, records, result = _replay_genuine_dpo(tmp_path)
    assert result.manifest.state == "succeeded"
    assert result.manifest.preference_success_evidence is not None
    assert len(result.artifacts) == 1
    assert (records / "runs" / rid / "artifacts" / f"{artifacts[0].artifact_id}.json").exists()


def test_artifact_persistence_failure_cannot_leave_a_succeeded_variant_manifest(
    tmp_path, monkeypatch
):
    rid, _artifacts, records, result = _replay_genuine_dpo(
        tmp_path, peak=PEAK, monkeypatch=monkeypatch, fail_persistence=True
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
    assert "artifact manifest persistence failed" in result.manifest.failure.message
    # The admitted variant's measured peak survives the downgrade as an unproven fit, never a null.
    assert result.manifest.final_fit == reconcile_measured_fit(PEAK, proven=False)
    persisted = RunManifest.model_validate_json(
        (records / "runs" / rid / "RunManifest.json").read_text(encoding="utf-8")
    )
    assert persisted.state == "failed"


@pytest.mark.parametrize("lane", ["training", *VARIANT_LANES])
def test_measured_failure_fit_reads_every_variant_family(lane):
    peak_evidence = type("Evidence", (), {"measured_peak": PEAK})()
    manifest = _manifest(demo_run_plan(), "run-fit").model_copy(
        update={EVIDENCE_FIELD[lane]: peak_evidence}
    )
    assert _measured_failure_fit(manifest) == reconcile_measured_fit(PEAK, proven=False)
    assert _measured_failure_fit(_manifest(demo_run_plan(), "run-fit")) is None


# ---- the admission stays control-plane-only ------------------------------------------------------


def test_parent_admission_module_stays_outside_the_worker_closure():
    repo_root = Path(__file__).resolve().parents[2]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from assurance.worker_reachability import DEFAULT_WORKER_ROOTS, reachable_from

    def _read(path: str) -> bytes | None:
        candidate = repo_root / path
        return candidate.read_bytes() if candidate.is_file() else None

    closure = reachable_from(DEFAULT_WORKER_ROOTS, _read).reachable
    assert "engine/corpus_studio/platform/worker.py" in closure
    assert "engine/corpus_studio/platform/execution_config.py" in closure
    assert "engine/corpus_studio/platform/subprocess_supervisor.py" not in closure
