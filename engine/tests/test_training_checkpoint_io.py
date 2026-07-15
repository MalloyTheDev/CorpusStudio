"""Deterministic tests for the checkpoint write + resume ENGINE (torch half of #440).

These run WITHOUT torch: a tiny fake ``torch`` module serializes state via pickle, so the real file
writing, hashing, atomic publish, control-plane sealing, fail-closed verification, live-parameter
identity, cadence, retention, and lineage logic are all exercised in CI. The end-to-end training
equivalence under a real torch is proven separately in ``test_training_checkpoint_integration.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

from corpus_studio.platform import checkpoint as ck
from corpus_studio.platform.contracts import CheckpointResumeRequest
from corpus_studio.platform.runners import demo_training_plan
from corpus_studio.training import checkpoint_io as cio


# --------------------------------------------------------------------------------------------------
# Fakes: a torch-shaped module + model/optimizer/scheduler/scaler with stable object identity
# --------------------------------------------------------------------------------------------------
class _Param:
    def __init__(self, name: str, requires_grad: bool = True) -> None:
        self.name = name
        self.requires_grad = requires_grad


class _Model:
    def __init__(self, params: list[tuple[str, _Param]]) -> None:
        self._params = params

    def named_parameters(self):  # noqa: ANN201 - torch-shaped
        return list(self._params)


class _Optimizer:
    def __init__(self, params: list[_Param], state: dict | None = None) -> None:
        self.param_groups = [{"params": params, "lr": 0.1}]
        self._state = state if state is not None else {"step": 0}

    def state_dict(self) -> dict:
        return {"state": dict(self._state), "param_groups": [{"lr": 0.1}]}

    def load_state_dict(self, payload: dict) -> None:
        self._state = dict(payload.get("state", {}))


class _Scheduler:
    def __init__(self) -> None:
        self._s = {"last_epoch": 0}

    def state_dict(self) -> dict:
        return dict(self._s)

    def load_state_dict(self, payload: dict) -> None:
        self._s = dict(payload)


class _Scaler:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def state_dict(self) -> dict:
        return {"scale": 2.0} if self._enabled else {}

    def load_state_dict(self, payload: dict) -> None:
        self._enabled = bool(payload)


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available
        self._rng = [b"cuda-rng-0"]

    def is_available(self) -> bool:
        return self._available

    def get_rng_state_all(self):  # noqa: ANN201
        return list(self._rng)

    def set_rng_state_all(self, states) -> None:  # noqa: ANN001
        self._rng = list(states)


class _Torch:
    def __init__(self, cuda: _Cuda | None = None) -> None:
        self.cuda = cuda
        self._rng = b"cpu-rng-0"

    def save(self, obj, path: str) -> None:  # noqa: ANN001
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)

    def load(self, path: str, map_location=None, weights_only=False):  # noqa: ANN001, ARG002
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def get_rng_state(self):  # noqa: ANN201
        return self._rng

    def set_rng_state(self, state) -> None:  # noqa: ANN001
        self._rng = state


def _bound(plan=None):  # noqa: ANN001
    return ck.bound_identities_from_plan(plan or demo_training_plan(plan_id="demo-ckpt"))


def _save(tmp: Path, *, torch=None, plan=None, **over):  # noqa: ANN001
    torch = torch or _Torch()
    model_params = [_Param("w"), _Param("b")]
    optimizer = _Optimizer(model_params, state={"step": 5})
    kwargs = dict(
        torch_module=torch,
        final_dir=tmp / "step-000006",
        adapter_state={"w": [1.0, 2.0], "b": [3.0]},
        optimizer=optimizer,
        position=cio.StepPosition(
            epoch=0.5, global_optimizer_step=6, gradient_accumulation_steps=4, consumed_microsteps=24
        ),
        bound=_bound(plan),
        source_run_id="run-parent01",
        checkpoint_id="run-parent01-ckpt-step-00000006",
        created_at="2026-07-15T00:00:00+00:00",
        rng_state=cio.capture_rng_state(torch),
        sampler_state={"order": [3, 1, 2, 0], "cursor": 2},
        trainer_state={"global_optimizer_step": 6},
    )
    kwargs.update(over)
    return torch, kwargs, cio.save_checkpoint(**kwargs)


# --------------------------------------------------------------------------------------------------
# Import boundary
# --------------------------------------------------------------------------------------------------
def test_checkpoint_io_import_is_torch_free() -> None:
    # Check in a FRESH interpreter: importing the engine module must not pull torch, even in a venv
    # where torch happens to be installed (a co-running test may already have imported it).
    import subprocess

    probe = (
        "import sys, corpus_studio.training.checkpoint_io;"
        "heavy=[m for m in ('torch','transformers','trl','peft','bitsandbytes') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "", f"checkpoint_io pulled: {result.stdout.strip()}"


# --------------------------------------------------------------------------------------------------
# Save + atomic publish + integrity
# --------------------------------------------------------------------------------------------------
def test_save_publishes_a_verifiable_sealed_checkpoint(tmp_path: Path) -> None:
    torch, kwargs, manifest = _save(tmp_path)
    final = kwargs["final_dir"]
    assert manifest.complete is True
    assert manifest.state.global_optimizer_step == 6
    assert manifest.state.rng_captured and manifest.state.sampler_state_captured
    # No orphan temp dir remains; the final dir is published.
    assert final.is_dir()
    assert not any(p.name.startswith(".step-000006.tmp") for p in tmp_path.iterdir())
    verified = ck.verify_checkpoint_integrity(final)
    assert verified.checkpoint_manifest_hash == manifest.checkpoint_manifest_hash
    roles = {entry.role for entry in verified.files}
    assert {"adapter_weights", "optimizer", "rng", "sampler", "trainer_state"} <= roles


def test_save_records_scheduler_and_scaler_when_present(tmp_path: Path) -> None:
    torch, kwargs, manifest = _save(
        tmp_path, lr_scheduler=_Scheduler(), scaler=_Scaler(enabled=True)
    )
    assert manifest.state.scheduler_captured and manifest.state.scaler_captured
    roles = {e.role for e in manifest.files}
    assert {"scheduler", "scaler"} <= roles


def test_empty_scaler_is_not_captured(tmp_path: Path) -> None:
    _, _, manifest = _save(tmp_path, scaler=_Scaler(enabled=False))
    assert manifest.state.scaler_captured is False
    assert not any(e.role == "scaler" for e in manifest.files)


def test_save_refuses_to_overwrite_an_existing_checkpoint(tmp_path: Path) -> None:
    _save(tmp_path)
    with pytest.raises(ck.CheckpointError) as exc:
        _save(tmp_path)  # same final_dir
    assert exc.value.reason == "incompatible"


def test_cuda_rng_is_captured_and_described_when_available(tmp_path: Path) -> None:
    torch = _Torch(cuda=_Cuda(available=True))
    state = cio.capture_rng_state(torch)
    assert "cuda" in state and "torch_cpu" in state
    assert cio.rng_algorithm_descriptor(state) == "torch-cpu-mt19937+torch-cuda-philox+python-mt19937"
    _, _, manifest = _save(tmp_path, torch=torch, rng_state=state)
    assert manifest.state.rng_algorithm.startswith("torch-cpu-mt19937+torch-cuda-philox")


def test_numpy_rng_capture_and_descriptor() -> None:
    np = pytest.importorskip("numpy")
    torch = _Torch()
    state = cio.capture_rng_state(torch, include_numpy=True)
    assert "numpy" in state
    assert "numpy-mt19937" in cio.rng_algorithm_descriptor(state)
    # restore round-trips (draw, snapshot, advance, restore, same draw)
    np.random.seed(0)
    snap = cio.capture_rng_state(torch, include_numpy=True)
    first = np.random.rand(3).tolist()
    np.random.rand(5)
    cio.restore_rng_state(snap, torch)
    assert np.random.rand(3).tolist() == first


def test_rng_descriptor_none_when_empty() -> None:
    assert cio.rng_algorithm_descriptor({}) == "none"


# --------------------------------------------------------------------------------------------------
# Restore: verify-before-load, request pin, identities, live parameters
# --------------------------------------------------------------------------------------------------
def _restore(tmp_path: Path, torch, plan, model, *, over_request=None, **over):  # noqa: ANN001
    manifest = ck.verify_checkpoint_integrity(tmp_path / "step-000006")
    request = over_request or CheckpointResumeRequest(
        checkpoint_id=manifest.checkpoint_id,
        checkpoint_manifest_hash=manifest.checkpoint_manifest_hash,
        checkpoint_dir=str(tmp_path / "step-000006"),
    )
    live_params = [p for _, p in model.named_parameters() if p.requires_grad]
    kwargs = dict(
        torch_module=torch,
        request=request,
        plan=plan,
        model=model,
        apply_adapter_state=lambda m, sd: None,
        build_optimizer=lambda: _Optimizer(live_params),
    )
    kwargs.update(over)
    return cio.restore_checkpoint(**kwargs)


def test_restore_roundtrip_rebuilds_over_live_params(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, _ = _save(tmp_path, plan=plan)
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    result = _restore(tmp_path, torch, plan, model)
    assert result.resumed_from_global_step == 6
    assert result.position.global_optimizer_step == 6
    assert result.sampler_state == {"order": [3, 1, 2, 0], "cursor": 2}
    # The optimizer owns exactly the model's live trainable params.
    assert cio.optimizer_param_ids(result.optimizer) == cio.trainable_param_ids(model)


def test_restore_rejects_optimizer_over_stale_params(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, _ = _save(tmp_path, plan=plan)
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    # A builder that returns an optimizer over FRESH (stale) params, not the model's live objects.
    with pytest.raises(ck.CheckpointError) as exc:
        _restore(tmp_path, torch, plan, model, build_optimizer=lambda: _Optimizer([_Param("w"), _Param("b")]))
    assert exc.value.reason == "incompatible"


def test_restore_pins_the_exact_requested_checkpoint(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, manifest = _save(tmp_path, plan=plan)
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    bad = CheckpointResumeRequest(
        checkpoint_id=manifest.checkpoint_id,
        checkpoint_manifest_hash="b" * 64,  # wrong hash
        checkpoint_dir=str(tmp_path / "step-000006"),
    )
    with pytest.raises(ck.CheckpointError) as exc:
        _restore(tmp_path, torch, plan, model, over_request=bad)
    assert exc.value.reason == "incompatible"


def test_restore_rejects_incompatible_plan(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, _ = _save(tmp_path, plan=plan)
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    other = plan.model_copy(update={"plan_hash": "b" * 64})
    with pytest.raises(ck.CheckpointError) as exc:
        _restore(tmp_path, torch, other, model)
    assert exc.value.reason == "incompatible"


def test_restore_rejects_mismatched_worker_wheel(tmp_path: Path) -> None:
    # Bind a worker wheel into the checkpoint, then resume with a different wheel.
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch = _Torch()
    bound = _bound(plan).model_copy(update={"worker_wheel_sha256": "a" * 64})
    _, _, _ = _save(tmp_path, torch=torch, plan=plan, bound=bound)
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    with pytest.raises(ck.CheckpointError) as exc:
        _restore(tmp_path, torch, plan, model, worker_wheel_sha256="c" * 64)
    assert exc.value.reason == "incompatible"


def test_restore_requires_scheduler_builder_when_captured(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, _ = _save(tmp_path, plan=plan, lr_scheduler=_Scheduler())
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    with pytest.raises(ck.CheckpointError) as exc:
        _restore(tmp_path, torch, plan, model, build_lr_scheduler=None)
    assert exc.value.reason == "incompatible"


def test_restore_restores_scheduler_and_scaler(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    torch, _, _ = _save(tmp_path, plan=plan, lr_scheduler=_Scheduler(), scaler=_Scaler())
    model = _Model([("w", _Param("w")), ("b", _Param("b"))])
    result = _restore(
        tmp_path, torch, plan, model,
        build_lr_scheduler=lambda opt: _Scheduler(),
        build_scaler=lambda: _Scaler(),
    )
    assert result.lr_scheduler is not None and result.scaler is not None


# --------------------------------------------------------------------------------------------------
# Fail-closed: interruption / incomplete / tamper / corruption / unsafe members
# --------------------------------------------------------------------------------------------------
def test_interruption_before_publish_leaves_no_readable_checkpoint(tmp_path: Path) -> None:
    # Simulate a crash: only the temp dir exists (rename never happened). The final path is absent, so
    # verification fails closed - a torn write is never mistaken for a resumable checkpoint.
    _, kwargs, _ = _save(tmp_path)
    (kwargs["final_dir"]).rename(tmp_path / ".step-000006.tmp.999")  # move final back to a temp name
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(tmp_path / "step-000006")
    assert exc.value.reason == "missing_file"


def test_tampered_state_file_fails_closed(tmp_path: Path) -> None:
    _, kwargs, _ = _save(tmp_path)
    (kwargs["final_dir"] / cio.OPTIMIZER_FILE).write_bytes(b"corrupt")
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(kwargs["final_dir"])
    assert exc.value.reason == "external_change"


def test_symlink_member_is_refused(tmp_path: Path) -> None:
    _, kwargs, _ = _save(tmp_path)
    final = kwargs["final_dir"]
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    target = final / cio.SAMPLER_FILE
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(final)
    assert exc.value.reason == "unsafe_path"


def test_hardlink_member_is_refused(tmp_path: Path) -> None:
    _, kwargs, _ = _save(tmp_path)
    final = kwargs["final_dir"]
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    target = final / cio.SAMPLER_FILE
    target.unlink()
    import os

    os.link(outside, target)  # hard link shares bytes with an external file
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(final)
    assert exc.value.reason == "unsafe_path"


# --------------------------------------------------------------------------------------------------
# Coordinator: cadence, checkpoint-free, retention, lineage chain
# --------------------------------------------------------------------------------------------------
def _coordinator(tmp_path: Path, *, cadence, keep_last=None, torch=None):  # noqa: ANN001
    clock = iter(f"2026-07-15T00:00:{i:02d}+00:00" for i in range(60))
    return cio.CheckpointCoordinator(
        torch_module=torch or _Torch(),
        checkpoints_root=tmp_path / "checkpoints",
        source_run_id="run-child01",
        bound=_bound(),
        clock=lambda: next(clock),
        cadence_optimizer_steps=cadence,
        keep_last=keep_last,
    )


def _tick(coord, step):  # noqa: ANN001
    params = [_Param("w")]
    return coord.maybe_checkpoint(
        global_optimizer_step=step,
        epoch=float(step),
        gradient_accumulation_steps=4,
        adapter_state={"w": [float(step)]},
        optimizer=_Optimizer(params, state={"step": step}),
        rng_state={"torch_cpu": b"r"},
    )


def test_checkpoint_free_coordinator_never_writes(tmp_path: Path) -> None:
    coord = _coordinator(tmp_path, cadence=None)
    assert coord.enabled is False
    for step in range(1, 13):
        assert _tick(coord, step) is None
    assert not (tmp_path / "checkpoints").exists()


def test_cadence_writes_only_on_due_steps_and_chains_lineage(tmp_path: Path) -> None:
    coord = _coordinator(tmp_path, cadence=3)
    manifests = [(_tick(coord, step)) for step in range(1, 10)]
    written = [m for m in manifests if m is not None]
    assert [m.state.global_optimizer_step for m in written] == [3, 6, 9]
    # Lineage chains: step 6's parent is step 3's checkpoint; step 9's parent is step 6's.
    assert written[0].parent_checkpoint_id is None
    assert written[1].parent_checkpoint_id == written[0].checkpoint_id
    assert written[1].parent_checkpoint_hash == written[0].checkpoint_manifest_hash
    assert written[2].parent_checkpoint_id == written[1].checkpoint_id
    # Each published checkpoint verifies.
    ck.verify_checkpoint_integrity(tmp_path / "checkpoints" / "step-00000009")


def test_retention_prunes_oldest_beyond_keep_last(tmp_path: Path) -> None:
    coord = _coordinator(tmp_path, cadence=1, keep_last=2)
    for step in range(1, 6):
        _tick(coord, step)
    kept = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert kept == ["step-00000004", "step-00000005"]  # only the freshest 2 remain


def test_coordinator_rejects_bad_cadence_and_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ck.CheckpointError):
        _coordinator(tmp_path, cadence=0)
    with pytest.raises(ck.CheckpointError):
        _coordinator(tmp_path, cadence=1, keep_last=0)


# --------------------------------------------------------------------------------------------------
# Sealed policy resolution (checkpoint-free stays unchanged; checkpoint-enabled requires a cadence)
# --------------------------------------------------------------------------------------------------
def test_resolve_checkpoint_policy_free_and_enabled() -> None:
    from corpus_studio.training.trainer import (
        TrainRunConfig,
        resolve_checkpoint_execution_policy,
    )

    free = resolve_checkpoint_execution_policy(TrainRunConfig(base_model="m", dataset_path="d"))
    assert free.enabled is False and free.cadence_optimizer_steps is None

    enabled = resolve_checkpoint_execution_policy(
        TrainRunConfig(
            base_model="m", dataset_path="d", save_strategy="steps", save_steps=4, save_total_limit=2
        )
    )
    assert enabled.enabled and enabled.cadence_optimizer_steps == 4 and enabled.keep_last == 2


def test_resolve_checkpoint_policy_rejects_inconsistent_spellings() -> None:
    from corpus_studio.training.trainer import (
        TrainerError,
        TrainRunConfig,
        resolve_checkpoint_execution_policy,
    )

    # save_strategy="steps" always requires a cadence (the contract also enforces this at the config
    # layer; the resolver fails closed regardless of how a config was constructed).
    cfg = TrainRunConfig(base_model="m", dataset_path="d", save_strategy="steps", save_steps=4)
    object.__setattr__(cfg, "save_steps", None)
    with pytest.raises(TrainerError):
        resolve_checkpoint_execution_policy(cfg)
