"""Reference-backend Environment Manager lifecycle tests.

All package installation and framework behavior is faked in temporary directories. The real command
runner is tested only with tiny stdlib Python commands; default CI needs no network and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Event
from typing import Any

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli_module
import corpus_studio.platform.environment_manager as manager_module
from corpus_studio.cli import app
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    EnvironmentInstallation,
    PythonRuntime,
    RunPlan,
)
from corpus_studio.platform.environment_manager import (
    CommandOutcome,
    EnvironmentManager,
    EnvironmentManagerError,
    SubprocessCommandRunner,
    discover_python_runtimes,
    locked_environment_ref,
    probe_python_runtime,
    verify_run_plan_environment,
)
from corpus_studio.platform.enums import EnvironmentState, FailureTaxonomy, OperatingSystem
from corpus_studio.platform.environments import get_recipe, resolution_digest


def _host_os() -> OperatingSystem:
    return OperatingSystem.windows if os.name == "nt" else OperatingSystem.linux


def _record_hash(name: str, version: str) -> str:
    return hashlib.sha256(f"{name}=={version}".encode()).hexdigest()


def _package(name: str, version: str = "1.0") -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "record_sha256": _record_hash(name, version),
        "direct_url": None,
        "installer": "pip",
        "requested": True,
        "dependencies": ["packaging>=23"],
    }


@dataclass
class FakeEnvironmentRunner:
    cuda: bool = False
    import_ok: bool = True
    functional_ok: bool = True
    hardware_ok: bool = True
    fail_phase: str | None = None
    timeout_phase: str | None = None
    cancel_phase: str | None = None
    native_build_output: bool = False
    create_interpreter: bool = True
    raise_phase: str | None = None
    malformed_phase: str | None = None
    packages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            _package("pip", "25.1"),
            _package("torch", "2.7.1+cu128"),
            _package("transformers", "4.52.0"),
            _package("peft", "0.15.0"),
            _package("trl", "0.18.0"),
            _package("accelerate", "1.7.0"),
            _package("datasets", "3.6.0"),
            _package("bitsandbytes", "0.46.0"),
        ]
    )
    calls: list[dict[str, Any]] = field(default_factory=list)
    env_python: str = ""
    compute_capability: str | None = "12.0"
    cuda_runtime: str | None = "12.8"

    def __call__(
        self,
        argv,
        *,
        cwd,
        environment,
        timeout_seconds,
        stdout_path,
        stderr_path,
        cancel,
    ) -> CommandOutcome:
        assert isinstance(argv, list)
        phase = self._phase(argv)
        self.calls.append(
            {
                "phase": phase,
                "argv": list(argv),
                "cwd": str(cwd),
                "environment": dict(environment),
                "timeout_seconds": timeout_seconds,
            }
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        if phase == self.raise_phase:
            raise OSError("simulated runner crash")
        stdout = ""
        stderr = ""

        if phase == "create_venv":
            root = Path(argv[-1])
            python_path = (
                root / "Scripts" / "python.exe"
                if _host_os() == OperatingSystem.windows
                else root / "bin" / "python"
            )
            if self.create_interpreter:
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("fake interpreter", encoding="utf-8")
                self.env_python = str(python_path)
        elif phase == "lock":
            stdout = json.dumps(self._lock_payload()) + "\n"
        elif phase == "import_probe":
            modules = (
                "corpus_studio.platform.worker",
                "torch",
                "transformers",
                "peft",
                "trl",
                "accelerate",
                "datasets",
                "bitsandbytes",
            )
            results = {
                name: {"ok": self.import_ok, "version": "1"}
                if self.import_ok
                else {"ok": False, "error": "ImportError: simulated"}
                for name in modules
            }
            stdout = json.dumps({"results": results}) + "\n"
        elif phase == "functional_probe":
            stdout = json.dumps(
                {
                    "ok": self.functional_ok,
                    "minimal_forward": self.functional_ok,
                    "minimal_backward": self.functional_ok,
                    "checkpoint_reload": self.functional_ok,
                    "error": None if self.functional_ok else "simulated failure",
                }
            ) + "\n"
        elif phase == "hardware_probe":
            stdout = json.dumps(
                {
                    "ok": self.cuda and self.hardware_ok,
                    "cuda_available": self.cuda,
                    "cuda_allocation": self.cuda and self.hardware_ok,
                    "compute_capability": self.compute_capability if self.cuda else None,
                    "bf16_supported": self.cuda,
                    "four_bit_construction": self.cuda and self.hardware_ok,
                    "minimal_forward": self.cuda and self.hardware_ok,
                    "minimal_backward": self.cuda and self.hardware_ok,
                    "attention_backend": "math" if self.cuda and self.hardware_ok else None,
                    "optional_kernels": {"flash_sdp_enabled": True},
                    "error": None if self.hardware_ok else "simulated hardware failure",
                }
            ) + "\n"
        elif phase == "capability_probe":
            signature = "c" * 64
            gpus = []
            if self.cuda:
                gpus.append(
                    {
                        "index": 0,
                        "kind": "cuda",
                        "name": "Synthetic managed GPU",
                        "compute_capability": self.compute_capability,
                        "compute_capability_major": 12,
                        "supported_dtypes": ["fp32", "fp16", "bf16"],
                    }
                )
            stdout = json.dumps(
                {
                    "profile": {
                        "environment_signature": signature,
                        "host": {"os": _host_os().value},
                        "gpus": gpus,
                    },
                    "capability_report": {
                        "backend_id": "corpus_studio",
                        "environment_ref": {"id": signature},
                        "readiness": "ready" if self.cuda else "cpu_toy_only",
                        "bitsandbytes_ok": self.cuda,
                        "effective_capabilities": {
                            "precision_modes": ["bf16"] if self.cuda else ["fp32"],
                            "quantization_modes": ["nf4"] if self.cuda else ["none"],
                            "attention_impls": ["math"],
                            "adapter_methods": ["qlora"] if self.cuda else ["lora"],
                        },
                    },
                }
            ) + "\n"

        if self.native_build_output and phase == "install":
            stderr += "Building wheel for simulated-native-package\n"
        if phase == self.malformed_phase:
            stdout = "not structured json\n"
        if phase == self.fail_phase:
            stderr += "simulated command failure\n"
            exit_code = 1
        else:
            exit_code = 0
        timed_out = phase == self.timeout_phase
        cancelled = phase == self.cancel_phase or (cancel is not None and cancel.is_set())
        if timed_out or cancelled:
            exit_code = -1
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return CommandOutcome(exit_code, timed_out=timed_out, cancelled=cancelled)

    def _phase(self, argv: list[str]) -> str:
        joined = " ".join(argv)
        if len(argv) >= 3 and argv[1:3] == ["-m", "venv"]:
            return "create_venv"
        if "-m pip" in joined and "check" in argv:
            return "dependency_probe"
        if "-m pip" in joined and "--upgrade" in argv:
            return "upgrade_pip"
        if "-m pip" in joined and "install" in argv:
            return "install"
        script = argv[-1] if len(argv) >= 3 and argv[-2] == "-c" else ""
        if "metadata.distributions" in script:
            return "lock"
        if "corpus_studio.platform.worker" in script and "bitsandbytes" in script:
            return "import_probe"
        if "checkpoint_reload" in script:
            return "functional_probe"
        if '"cuda_available"' in script:
            return "hardware_probe"
        if "build_environment_profile" in script and "run_capability_probes" in script:
            return "capability_probe"
        return "unknown"

    def _lock_payload(self) -> dict[str, Any]:
        return {
            "runtime": {
                "executable": self.env_python,
                "version": "3.12.10",
                "implementation": "CPython",
                "architecture": "64-bit",
                "platform": "test-platform",
                "os": _host_os().value,
                "is_virtual_environment": True,
                "venv_available": True,
            },
            "packages": self.packages,
            "torch": {
                "version": "2.7.1+cu128",
                "build": "fake-git-build",
                "cuda": self.cuda_runtime if self.cuda else None,
                "compute_capability": self.compute_capability if self.cuda else None,
            },
        }


def _manager_and_resolution(tmp_path: Path, runner: FakeEnvironmentRunner, env_id: str = "ref-env"):
    recipe = get_recipe("backend-corpus-studio")
    assert recipe is not None
    runtime = PythonRuntime(
        runtime_id="python-test",
        executable=str(tmp_path / "base-python"),
        version="3.12.10",
        implementation="CPython",
        architecture="64-bit",
        platform="test-platform",
        os=_host_os(),
        venv_available=True,
        compatible=True,
    )
    manager = EnvironmentManager(
        tmp_path / "manager",
        runner=runner,
        runtime_probe=lambda executable, requirement: runtime,
    )
    resolution = manager.preview(
        recipe.recipe_id,
        env_id=env_id,
        runtime_executable=runtime.executable,
        accelerator_tag="cu128" if runner.cuda else "cpu",
    )
    assert resolution.resolution_hash
    return manager, resolution


def _create(tmp_path: Path, runner: FakeEnvironmentRunner, env_id: str = "ref-env"):
    manager, resolution = _manager_and_resolution(tmp_path, runner, env_id)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    return manager, resolution, result


def test_reference_environment_full_cpu_lifecycle_is_recorded(tmp_path):
    fake = FakeEnvironmentRunner(native_build_output=True)
    manager, resolution, result = _create(tmp_path, fake)

    assert result.descriptor.state == EnvironmentState.functional_probe_passed
    assert result.lock.lock_hash and len(result.lock.lock_hash) == 64
    assert result.lock.runtime and result.lock.runtime.is_virtual_environment
    assert result.lock.torch_version == "2.7.1+cu128"
    assert {package.name for package in result.lock.packages} >= {"torch", "transformers"}
    torch = next(package for package in result.lock.packages if package.name == "torch")
    assert torch.source == "unknown"
    assert torch.hash and torch.dependencies == ["packaging>=23"]
    assert result.descriptor.lock_ref and result.descriptor.lock_ref.hash
    assert result.health.state == EnvironmentState.functional_probe_passed
    assert result.health.probe_results[-1].outcome == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert any(command.native_build_occurred for command in result.installation.commands)
    assert all(isinstance(command.argv, list) for command in result.installation.commands)
    assert all(command.stdout_path and command.stderr_path for command in result.installation.commands)
    assert (manager.environment_root("ref-env") / ".corpusstudio-owner.json").is_file()
    assert not list(manager.root.rglob("*.tmp"))
    assert resolution.resolution_hash == result.installation.resolution_ref.hash.value


def test_cuda_hardware_probe_earns_hardware_verified_only_after_math_path(tmp_path):
    _, _, result = _create(tmp_path, FakeEnvironmentRunner(cuda=True))
    assert result.descriptor.state == EnvironmentState.hardware_verified
    hardware = result.health.probe_results[-1]
    assert hardware.outcome == FailureTaxonomy.PASS
    assert hardware.measured["attention_backend"] == "math"
    assert result.lock.cuda_runtime_version == "12.8"
    assert result.lock.compute_capability == "12.0"


def test_capability_snapshot_is_proved_inside_the_managed_interpreter(tmp_path):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, result = _create(tmp_path, fake)
    profile, report = manager.capability_snapshot("ref-env")
    assert profile.environment_signature == "c" * 64
    assert report.readiness == "ready"
    call = next(call for call in reversed(fake.calls) if call["phase"] == "capability_probe")
    assert call["argv"][0] == result.descriptor.python_executable


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (FakeEnvironmentRunner(import_ok=False), EnvironmentState.degraded),
        (FakeEnvironmentRunner(fail_phase="dependency_probe"), EnvironmentState.degraded),
        (FakeEnvironmentRunner(functional_ok=False), EnvironmentState.degraded),
        (
            FakeEnvironmentRunner(cuda=True, hardware_ok=False),
            EnvironmentState.incompatible,
        ),
    ],
)
def test_probe_failures_have_honest_non_supported_states(tmp_path, fake, expected):
    _, _, result = _create(tmp_path, fake)
    assert result.descriptor.state == expected
    assert result.health.state == expected
    assert result.health.probe_results[-1].outcome != FailureTaxonomy.PASS


def test_confirmation_and_resolution_seal_block_all_mutation(tmp_path):
    manager, resolution = _manager_and_resolution(tmp_path, FakeEnvironmentRunner())
    with pytest.raises(EnvironmentManagerError, match="exact resolution hash"):
        manager.create(resolution, confirmed_resolution_hash="0" * 64)
    assert not manager.root.exists()

    tampered = resolution.model_copy(update={"warnings": ["changed after review"]})
    with pytest.raises(EnvironmentManagerError, match="modified after review"):
        manager.create(
            tampered, confirmed_resolution_hash=resolution.resolution_hash or ""
        )
    assert not manager.root.exists()

    unsafe_step = resolution.install_steps[1].model_copy(
        update={"argv": [sys.executable, "-c", "print('unsafe')"]}
    )
    unsafe = resolution.model_copy(
        update={
            "install_steps": [resolution.install_steps[0], unsafe_step]
            + resolution.install_steps[2:],
            "resolution_hash": None,
        }
    )
    unsafe = unsafe.model_copy(update={"resolution_hash": resolution_digest(unsafe)})
    with pytest.raises(EnvironmentManagerError, match="canonical plan"):
        manager.create(
            unsafe, confirmed_resolution_hash=unsafe.resolution_hash or ""
        )
    assert not manager.root.exists()


def test_actionable_preview_is_concrete_explicit_and_does_not_mutate(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_STUDIO_TEST_SECRET", "must-not-leak")
    engine_source = tmp_path / "engine-source"
    engine_source.mkdir()
    (engine_source / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    manager = EnvironmentManager(tmp_path / "manager", engine_source=engine_source)
    resolution = manager.preview(
        "backend-corpus-studio",
        env_id="preview-env",
        runtime_executable=sys.executable,
        accelerator_tag="cpu",
    )
    assert resolution.resolution_hash and resolution.resolvable
    assert resolution.environment_root == str(manager.environment_root("preview-env"))
    assert all(
        "<ENV_ROOT>" not in token and "<BASE_PYTHON>" not in token
        for step in resolution.install_steps
        for token in step.argv
    )
    assert all(step.environment.get("PYTHONUTF8") == "1" for step in resolution.install_steps)
    assert all(step.environment.get("PIP_CONFIG_FILE") == os.devnull for step in resolution.install_steps)
    assert all("CORPUS_STUDIO_TEST_SECRET" not in step.environment for step in resolution.install_steps)
    worker_step = resolution.install_steps[-1]
    assert worker_step.argv[-2:] == ["--no-deps", str(engine_source.resolve())]
    assert worker_step.network_required is True
    assert not manager.root.exists()


def test_preview_reports_unknown_recipe_missing_source_and_incompatible_runtime(
    tmp_path, monkeypatch
):
    manager = EnvironmentManager(tmp_path / "manager", engine_source=tmp_path / "missing")
    with pytest.raises(EnvironmentManagerError, match="unknown environment recipe"):
        manager.preview(
            "missing", env_id="x", runtime_executable=sys.executable
        )
    blocked = manager.preview(
        "backend-corpus-studio",
        env_id="blocked",
        runtime_executable=sys.executable,
    )
    assert blocked.resolvable is False
    assert any("pyproject.toml" in reason for reason in blocked.blocking_reasons)

    incompatible = PythonRuntime(
        runtime_id="python-old",
        executable=str(tmp_path / "old-python"),
        version="3.10.0",
        implementation="CPython",
        architecture="64-bit",
        platform="test",
        os=_host_os(),
        venv_available=False,
        compatible=False,
        incompatibility_reasons=["too old", "venv unavailable"],
    )
    monkeypatch.setattr(manager_module, "probe_python_runtime", lambda *a, **k: incompatible)
    blocked_runtime = EnvironmentManager(
        tmp_path / "other-manager", engine_source=Path(__file__).parents[1]
    ).preview(
        "backend-corpus-studio",
        env_id="old-runtime",
        runtime_executable=incompatible.executable,
    )
    assert blocked_runtime.resolvable is False
    assert "too old" in blocked_runtime.blocking_reasons


@pytest.mark.parametrize(
    "fake",
    [
        FakeEnvironmentRunner(create_interpreter=False),
        FakeEnvironmentRunner(raise_phase="install"),
        FakeEnvironmentRunner(malformed_phase="lock"),
    ],
)
def test_missing_outputs_runner_crashes_and_malformed_lock_are_structured(tmp_path, fake):
    manager, resolution = _manager_and_resolution(tmp_path, fake)
    with pytest.raises(EnvironmentManagerError):
        manager.create(
            resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
        )
    descriptor = manager.load_descriptor("ref-env")
    report = manager.load_health("ref-env")
    assert descriptor.state == EnvironmentState.broken
    assert report.failure and report.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    journal_path = (
        manager.registry_root
        / "ref-env"
        / "installations"
        / f"{descriptor.installation_ref.id}.json"
    )
    journal = EnvironmentInstallation.model_validate_json(journal_path.read_text(encoding="utf-8"))
    assert journal.commands
    assert any(command.failure is not None for command in journal.commands)

def test_existing_directory_is_never_silently_overwritten(tmp_path):
    manager, resolution = _manager_and_resolution(tmp_path, FakeEnvironmentRunner())
    env_root = manager.environment_root("ref-env")
    env_root.mkdir(parents=True)
    precious = env_root / "user-file.txt"
    precious.write_text("keep", encoding="utf-8")
    with pytest.raises(EnvironmentManagerError, match="already exists"):
        manager.create(
            resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
        )
    assert precious.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("outcome_field", "taxonomy"),
    [("timeout_phase", FailureTaxonomy.TIMEOUT), ("cancel_phase", FailureTaxonomy.ENVIRONMENT_FAILURE)],
)
def test_failed_timeout_and_cancelled_installs_are_recoverable_broken_states(
    tmp_path, outcome_field, taxonomy
):
    fake = FakeEnvironmentRunner()
    setattr(fake, outcome_field, "install")
    manager, resolution = _manager_and_resolution(tmp_path, fake)
    with pytest.raises(EnvironmentManagerError) as error:
        manager.create(
            resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
        )
    assert error.value.failure.taxonomy == taxonomy
    descriptor = manager.load_descriptor("ref-env")
    assert descriptor.state == EnvironmentState.broken
    assert descriptor.installation_ref
    journal_path = (
        manager.registry_root
        / "ref-env"
        / "installations"
        / f"{descriptor.installation_ref.id}.json"
    )
    journal = EnvironmentInstallation.model_validate_json(journal_path.read_text(encoding="utf-8"))
    assert journal.retry_requires_recreate is True
    assert journal.failure and journal.failure.taxonomy == taxonomy
    assert manager.environment_root("ref-env").exists()


def test_safe_removal_requires_exact_confirmation_marker_and_containment(tmp_path):
    manager, _, result = _create(tmp_path, FakeEnvironmentRunner())
    env_root = Path(result.descriptor.root_path)
    marker_path = env_root / ".corpusstudio-owner.json"
    marker = marker_path.read_text(encoding="utf-8")
    precious = env_root / "precious.txt"
    precious.write_text("keep", encoding="utf-8")

    with pytest.raises(EnvironmentManagerError, match="exact environment id"):
        manager.remove("ref-env", confirmed_env_id="wrong")
    marker_path.unlink()
    with pytest.raises(EnvironmentManagerError, match="ownership marker"):
        manager.remove("ref-env", confirmed_env_id="ref-env")
    assert precious.is_file()
    marker_path.write_text(marker, encoding="utf-8")
    wrong_marker = json.loads(marker)
    wrong_marker["manager_root"] = str(tmp_path / "another-manager")
    marker_path.write_text(json.dumps(wrong_marker), encoding="utf-8")
    with pytest.raises(EnvironmentManagerError, match="does not match"):
        manager.remove("ref-env", confirmed_env_id="ref-env")
    marker_path.write_text(marker, encoding="utf-8")

    removed = manager.remove("ref-env", confirmed_env_id="ref-env")
    assert removed.state == EnvironmentState.not_installed
    assert not env_root.exists()
    assert (manager.registry_root / "ref-env").is_dir()
    assert manager.load_health("ref-env").state == EnvironmentState.not_installed
    health = manager.health("ref-env")
    assert health.state == EnvironmentState.not_installed
    assert health.environment_missing is False

    for unsafe in ("../escape", "..", "bad/name"):
        with pytest.raises(EnvironmentManagerError):
            manager.environment_root(unsafe)


def test_recreate_requires_both_deletion_and_new_plan_confirmations(tmp_path):
    fake = FakeEnvironmentRunner()
    manager, resolution, first = _create(tmp_path, fake)
    with pytest.raises(EnvironmentManagerError, match="exact environment id"):
        manager.recreate(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
            confirmed_remove_env_id="wrong",
        )
    assert Path(first.descriptor.root_path).exists()
    with pytest.raises(EnvironmentManagerError, match="exact resolution hash"):
        manager.recreate(
            resolution,
            confirmed_resolution_hash="0" * 64,
            confirmed_remove_env_id="ref-env",
        )
    assert Path(first.descriptor.root_path).exists()
    second = manager.recreate(
        resolution,
        confirmed_resolution_hash=resolution.resolution_hash or "",
        confirmed_remove_env_id="ref-env",
    )
    assert second.descriptor.state == EnvironmentState.functional_probe_passed
    assert second.installation.installation_id != first.installation.installation_id


def test_live_health_detects_version_source_hash_recipe_lock_and_hardware_drift(
    tmp_path, monkeypatch
):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, result = _create(tmp_path, fake)
    torch = next(package for package in fake.packages if package["name"] == "torch")
    torch["version"] = "9.9"
    torch["record_sha256"] = _record_hash("torch", "9.9")
    torch["direct_url"] = {
        "url": "https://example.invalid/torch-9.9.whl",
        "archive_info": {},
    }
    fake.compute_capability = "8.9"
    fake.packages = [package for package in fake.packages if package["name"] != "datasets"]
    fake.packages.append(_package("unexpected-package", "1.2.3"))
    recipe = get_recipe("backend-corpus-studio")
    assert recipe is not None
    monkeypatch.setattr(
        manager_module,
        "get_recipe",
        lambda recipe_id: recipe.model_copy(update={"description": "recipe changed"}),
    )

    report = manager.health("ref-env")
    assert report.state == EnvironmentState.drifted
    assert report.drift_detected is True
    assert report.recipe_drift_detected is True
    assert report.hardware_mismatch is True
    assert report.cuda_mismatch is False
    assert any("torch" in item for item in report.drifted_packages)
    assert any("torch" in item for item in report.changed_package_sources)
    assert any("datasets" in item and "missing" in item for item in report.drifted_packages)
    assert any("unexpected-package" in item for item in report.drifted_packages)
    assert manager.load_descriptor("ref-env").state == EnvironmentState.drifted
    assert result.lock.lock_hash


def test_health_detects_tampered_or_missing_lock_and_missing_runtime(tmp_path):
    fake = FakeEnvironmentRunner()
    manager, _, result = _create(tmp_path, fake)
    lock_path = manager.registry_root / "ref-env" / "locks" / f"{result.lock.lock_id}.json"
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["manager_version"] = "tampered"
    lock_path.write_text(json.dumps(raw), encoding="utf-8")
    assert manager.health("ref-env").lock_mismatch is True

    lock_path.unlink()
    missing_lock = manager.health("ref-env")
    assert missing_lock.state == EnvironmentState.broken
    assert missing_lock.lock_mismatch is True

    # Restore a valid registry, then remove only the fake interpreter.
    lock_path.write_text(result.lock.model_dump_json(), encoding="utf-8")
    Path(result.descriptor.python_executable).unlink()
    missing_python = manager.health("ref-env")
    assert missing_python.state == EnvironmentState.broken
    assert missing_python.interpreter_missing is True


def test_health_detects_external_environment_removal(tmp_path):
    manager, _, result = _create(tmp_path, FakeEnvironmentRunner())
    shutil.rmtree(result.descriptor.root_path)
    report = manager.health("ref-env")
    assert report.state == EnvironmentState.broken
    assert report.environment_missing is True


def test_health_command_and_each_live_probe_failure_are_structured(tmp_path):
    lock_failure = FakeEnvironmentRunner()
    manager, _, _ = _create(tmp_path / "lock", lock_failure)
    lock_failure.fail_phase = "lock"
    report = manager.health("ref-env")
    assert report.state == EnvironmentState.broken
    assert report.failure

    cases = [
        (FakeEnvironmentRunner(), "import_ok", False, EnvironmentState.degraded),
        (FakeEnvironmentRunner(), "fail_phase", "dependency_probe", EnvironmentState.degraded),
        (FakeEnvironmentRunner(), "functional_ok", False, EnvironmentState.degraded),
        (FakeEnvironmentRunner(cuda=True), "hardware_ok", False, EnvironmentState.incompatible),
    ]
    for index, (fake, field_name, value, expected) in enumerate(cases):
        case_manager, _, _ = _create(tmp_path / f"case-{index}", fake)
        setattr(fake, field_name, value)
        health = case_manager.health("ref-env")
        assert health.state == expected


def test_runtime_probe_and_discovery_are_bounded_and_multi_runtime(tmp_path):
    current = probe_python_runtime(sys.executable)
    assert current.compatible and current.venv_available
    assert current.executable

    def fake_probe(candidate, requirement):
        name = Path(candidate).name
        if name == "broken":
            raise RuntimeError("broken")
        version = "3.12.4" if name == "new" else "3.10.9"
        compatible = version.startswith("3.12")
        return PythonRuntime(
            runtime_id=f"python-{name}",
            executable=str(candidate),
            version=version,
            implementation="CPython",
            architecture="64-bit",
            platform="test",
            os=_host_os(),
            venv_available=True,
            compatible=compatible,
            incompatibility_reasons=[] if compatible else [f"does not satisfy {requirement}"],
        )

    runtimes = discover_python_runtimes(
        candidates=[tmp_path / "old", tmp_path / "new", tmp_path / "new", tmp_path / "broken"],
        python_requires=">=3.11",
        probe=fake_probe,
    )
    assert [runtime.runtime_id for runtime in runtimes] == ["python-new", "python-old"]
    assert runtimes[0].compatible and not runtimes[1].compatible


def test_runtime_and_provenance_helper_edge_cases(tmp_path, monkeypatch):
    assert manager_module._operating_system("Windows") == OperatingSystem.windows
    assert manager_module._operating_system("Darwin") == OperatingSystem.macos
    assert manager_module._operating_system("Linux", "microsoft-standard") == OperatingSystem.wsl
    assert manager_module._operating_system("Linux") == OperatingSystem.linux
    assert manager_module._operating_system("Plan9") == OperatingSystem.unknown

    runtime = manager_module._runtime_from_payload(
        {
            "executable": "python",
            "version": "3.9",
            "implementation": "CPython",
            "architecture": "64-bit",
            "platform": "test",
            "os": "not-an-os",
            "venv_available": False,
        },
        ">=3.11",
    )
    assert runtime.os == OperatingSystem.unknown
    assert len(runtime.incompatibility_reasons) == 2

    assert manager_module._package_source(None)[0] == "unknown"
    assert manager_module._package_source({"url": "git+x", "vcs_info": {}})[0] == "vcs"
    assert manager_module._package_source({"url": "file:///tmp/pkg", "dir_info": {}})[0] == "local"
    wheel = manager_module._package_source(
        {"url": "https://example.invalid/pkg.whl", "archive_info": {}}
    )
    assert wheel == ("wheel", "https://example.invalid/pkg.whl", "pkg.whl")
    assert manager_module._package_source(
        {"url": "https://example.invalid/pkg.tar.gz", "archive_info": {}}
    )[0] == "sdist"

    assert manager_module._read_tail(tmp_path / "missing") == ""
    with pytest.raises(json.JSONDecodeError):
        manager_module._last_json_object("\nnot json\n[]\n")

    completed = manager_module.subprocess.CompletedProcess(
        ["python"], 3, stdout="", stderr="probe exploded"
    )
    monkeypatch.setattr(manager_module.subprocess, "run", lambda *a, **k: completed)
    with pytest.raises(RuntimeError, match="probe exploded"):
        probe_python_runtime("python")
    assert manager_module.default_manager_root().name == "environment-manager"


def test_real_subprocess_runner_success_timeout_and_cancellation(tmp_path):
    runner = SubprocessCommandRunner()
    success = runner(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        environment={"PYTHONUTF8": "1"},
        timeout_seconds=10,
        stdout_path=tmp_path / "ok.out",
        stderr_path=tmp_path / "ok.err",
        cancel=None,
    )
    assert success.exit_code == 0
    assert (tmp_path / "ok.out").read_text(encoding="utf-8").strip() == "ok"

    timeout = runner(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        environment={},
        timeout_seconds=1,
        stdout_path=tmp_path / "timeout.out",
        stderr_path=tmp_path / "timeout.err",
        cancel=None,
    )
    assert timeout.timed_out is True

    cancelled = Event()
    cancelled.set()
    cancel_result = runner(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        environment={},
        timeout_seconds=10,
        stdout_path=tmp_path / "cancel.out",
        stderr_path=tmp_path / "cancel.err",
        cancel=cancelled,
    )
    assert cancel_result.cancelled is True


def _run_plan(environment_ref: Ref) -> RunPlan:
    from corpus_studio.platform.backends import backend_manifest_ref, get_backend

    backend = get_backend("corpus_studio")
    assert backend is not None
    draft = RunPlan(
        plan_id="plan-1",
        plan_hash="0" * 64,
        backend_ref=backend_manifest_ref(backend),
        environment_ref=environment_ref,
        dataset_ref=Ref(id="dataset-1"),
        task_type="sft",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        precision="bf16",
        quantization="nf4",
        adapter={"method": "qlora", "lora_r": 16, "lora_alpha": 32},
        optimizer={"impl": "paged_adamw_8bit", "learning_rate": 2e-4},
        loss_impl="liger_fused_ce",
        attention_backend="math",
        sequence={"max_sequence_len": 1024},
        batching={
            "micro_batch_size": 1,
            "supervised_token_accumulation_target": 4096,
        },
        checkpoint_policy={"impl": "adapter_only"},
        export={"format": "adapter_peft"},
    )
    from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload

    return draft.model_copy(update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))})


def test_run_plan_pins_lock_hash_and_resume_verifies_state(tmp_path):
    _, _, result = _create(tmp_path, FakeEnvironmentRunner())
    environment_ref = locked_environment_ref(result.descriptor, result.lock)
    plan = _run_plan(environment_ref)
    assert verify_run_plan_environment(plan, result.descriptor, result.lock) == []

    wrong = plan.model_copy(
        update={"environment_ref": Ref(id=result.descriptor.env_id, hash=HashRef(value="0" * 64))}
    )
    assert "lock hash" in verify_run_plan_environment(
        wrong, result.descriptor, result.lock
    )[0]
    degraded = result.descriptor.model_copy(update={"state": EnvironmentState.degraded})
    assert any("not functionally verified" in item for item in verify_run_plan_environment(
        plan, degraded, result.lock
    ))
    wrong_id = plan.model_copy(
        update={"environment_ref": Ref(id="other-env", hash=plan.environment_ref.hash)}
    )
    assert any("environment id" in item for item in verify_run_plan_environment(
        wrong_id, result.descriptor, result.lock
    ))
    with pytest.raises(EnvironmentManagerError, match="unsealed"):
        locked_environment_ref(
            result.descriptor, result.lock.model_copy(update={"lock_hash": None})
        )
    with pytest.raises(EnvironmentManagerError, match="do not match"):
        locked_environment_ref(
            result.descriptor.model_copy(update={"lock_ref": Ref(id="wrong")}),
            result.lock,
        )
    with pytest.raises(EnvironmentManagerError, match="do not match"):
        locked_environment_ref(
            result.descriptor.model_copy(
                update={
                    "lock_ref": Ref(
                        id=result.lock.lock_id,
                        hash=HashRef(value="0" * 64),
                    )
                }
            ),
            result.lock,
        )

    recipe_mismatch = result.lock.model_copy(
        update={"recipe_ref": Ref(id="backend-corpus-studio", hash=HashRef(value="0" * 64))}
    )
    recipe_blockers = verify_run_plan_environment(
        plan, result.descriptor, recipe_mismatch
    )
    assert any("recipe refs do not match" in item for item in recipe_blockers)
    assert any("lock recipe hash" in item for item in recipe_blockers)

    from corpus_studio.platform.backends import backend_manifest_ref, get_backend

    unsloth = get_backend("unsloth")
    assert unsloth is not None
    wrong_backend = plan.model_copy(update={"backend_ref": backend_manifest_ref(unsloth)})
    assert any(
        "recipe target" in item
        for item in verify_run_plan_environment(
            wrong_backend, result.descriptor, result.lock
        )
    )


def test_registry_listing_skips_corruption_and_unknown_loads_are_structured(tmp_path):
    manager, _, result = _create(tmp_path, FakeEnvironmentRunner())
    corrupt = manager.registry_root / "bad" / "EnvironmentDescriptor.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("not json", encoding="utf-8")
    assert [item.env_id for item in manager.list_descriptors()] == [result.descriptor.env_id]
    with pytest.raises(EnvironmentManagerError, match="unknown managed environment"):
        manager.load_descriptor("missing")


def test_lifecycle_cli_surfaces_status_lock_and_safe_remove(tmp_path, monkeypatch):
    fake = FakeEnvironmentRunner()
    manager, resolution = _manager_and_resolution(tmp_path, fake, "cli-env")
    monkeypatch.setattr(
        cli_module,
        "_build_environment_resolution",
        lambda *args, **kwargs: (manager, resolution),
    )
    cli = CliRunner()
    created = cli.invoke(
        app,
        ["env-create", "--confirm", resolution.resolution_hash or "", "--json"],
    )
    assert created.exit_code == 0, created.output
    assert json.loads(created.stdout)["descriptor"]["state"] == "FUNCTIONAL_PROBE_PASSED"
    monkeypatch.setattr(
        manager_module, "EnvironmentManager", lambda root=None: manager
    )

    listed = cli.invoke(
        app, ["env-status", "--manager-root", str(manager.root)]
    )
    assert listed.exit_code == 0
    assert "cli-env  FUNCTIONAL_PROBE_PASSED" in listed.stdout

    status = cli.invoke(
        app,
        ["env-status", "cli-env", "--manager-root", str(manager.root), "--json"],
    )
    assert status.exit_code == 0
    assert json.loads(status.stdout)["descriptor"]["env_id"] == "cli-env"

    status_text = cli.invoke(
        app, ["env-status", "cli-env", "--manager-root", str(manager.root)]
    )
    assert status_text.exit_code == 0
    assert "Drift detected: False" in status_text.stdout

    probed = cli.invoke(
        app, ["env-probe", "cli-env", "--manager-root", str(manager.root)]
    )
    assert probed.exit_code == 0
    assert "reference_backend_functional: PASS" in probed.stdout
    probed_json = cli.invoke(
        app,
        ["env-probe", "cli-env", "--manager-root", str(manager.root), "--json"],
    )
    assert probed_json.exit_code == 0
    assert json.loads(probed_json.stdout)["state"] == "FUNCTIONAL_PROBE_PASSED"

    lock = cli.invoke(
        app, ["env-lock", "cli-env", "--manager-root", str(manager.root)]
    )
    assert lock.exit_code == 0
    assert json.loads(lock.stdout)["lock_hash"]

    missing_lock = cli.invoke(
        app, ["env-lock", "missing", "--manager-root", str(manager.root)]
    )
    assert missing_lock.exit_code == 2

    recreated = cli.invoke(
        app,
        [
            "env-recreate",
            "--confirm",
            resolution.resolution_hash or "",
            "--confirm-remove",
            "cli-env",
        ],
    )
    assert recreated.exit_code == 0, recreated.output
    assert "FUNCTIONAL_PROBE_PASSED" in recreated.stdout

    refused = cli.invoke(
        app,
        [
            "env-remove",
            "cli-env",
            "--manager-root",
            str(manager.root),
            "--confirm",
            "wrong",
        ],
    )
    assert refused.exit_code == 2
    removed = cli.invoke(
        app,
        [
            "env-remove",
            "cli-env",
            "--manager-root",
            str(manager.root),
            "--confirm",
            "cli-env",
        ],
    )
    assert removed.exit_code == 0
    assert "NOT_INSTALLED" in removed.stdout


def test_environment_cli_runtime_and_creation_text_paths(tmp_path, monkeypatch):
    compatible = PythonRuntime(
        runtime_id="python-new",
        executable=str(tmp_path / "python-new"),
        version="3.12.10",
        implementation="CPython",
        architecture="64-bit",
        platform="test",
        os=_host_os(),
        venv_available=True,
        compatible=True,
    )
    incompatible = compatible.model_copy(
        update={
            "runtime_id": "python-old",
            "version": "3.10.9",
            "compatible": False,
            "incompatibility_reasons": ["Python is too old"],
        }
    )
    monkeypatch.setattr(
        manager_module,
        "discover_python_runtimes",
        lambda **kwargs: [compatible, incompatible],
    )
    cli = CliRunner()
    runtimes = cli.invoke(app, ["env-runtimes"])
    assert runtimes.exit_code == 0
    assert "python-new  compatible" in runtimes.stdout
    assert "Python is too old" in runtimes.stdout
    runtimes_json = cli.invoke(app, ["env-runtimes", "--json"])
    assert runtimes_json.exit_code == 0
    assert len(json.loads(runtimes_json.stdout)) == 2
    unknown = cli.invoke(app, ["env-runtimes", "--recipe", "missing"])
    assert unknown.exit_code == 2
    monkeypatch.setattr(
        manager_module, "discover_python_runtimes", lambda **kwargs: []
    )
    empty = cli.invoke(app, ["env-runtimes"])
    assert empty.exit_code == 1
    assert "No working Python runtimes" in empty.stdout

    recipes = cli.invoke(app, ["env-recipes"])
    assert recipes.exit_code == 0
    assert "backend-corpus-studio" in recipes.stdout
    assert "verification:" in recipes.stdout

    fake = FakeEnvironmentRunner()
    manager, resolution = _manager_and_resolution(tmp_path / "create", fake, "text-env")
    monkeypatch.setattr(
        cli_module,
        "_build_environment_resolution",
        lambda *args, **kwargs: (manager, resolution),
    )
    wrong = cli.invoke(app, ["env-create", "--confirm", "0" * 64])
    assert wrong.exit_code == 2
    created = cli.invoke(
        app, ["env-create", "--confirm", resolution.resolution_hash or ""]
    )
    assert created.exit_code == 0, created.output
    assert "Environment: text-env" in created.stdout
    assert "Installation journal:" in created.stdout


def test_platform_run_verifies_lock_and_dispatches_with_managed_interpreter(
    tmp_path, monkeypatch
):
    fake = FakeEnvironmentRunner()
    manager, _, result = _create(tmp_path, fake, "run-env")
    plan = _run_plan(locked_environment_ref(result.descriptor, result.lock))
    plan_path = tmp_path / "RunPlan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr(
        manager_module, "EnvironmentManager", lambda root=None: manager
    )
    cli = CliRunner()

    in_process = cli.invoke(app, ["platform-run", str(plan_path)])
    assert in_process.exit_code == 2
    assert "must run with --subprocess" in in_process.output

    captured: dict[str, Any] = {}
    import corpus_studio.platform.subprocess_supervisor as subprocess_module
    from corpus_studio.platform.supervisor import EchoRunner, execute_run

    def fake_subprocess(run_plan, **kwargs):
        captured["worker_argv"] = kwargs["worker_argv"]
        return execute_run(run_plan, EchoRunner())

    monkeypatch.setattr(subprocess_module, "execute_run_subprocess", fake_subprocess)
    dispatched = cli.invoke(
        app, ["platform-run", str(plan_path), "--subprocess"]
    )
    assert dispatched.exit_code == 0, dispatched.output
    worker_argv = captured["worker_argv"]
    assert worker_argv[0] == result.descriptor.python_executable
    assert worker_argv[1:3] == ["-m", "corpus_studio.platform.worker"]
    assert worker_argv[worker_argv.index("--backend-id") + 1] == "corpus_studio"
    assert worker_argv[worker_argv.index("--environment-id") + 1] == "run-env"
    assert worker_argv[worker_argv.index("--environment-hash") + 1] == result.lock.lock_hash

    shutil.rmtree(result.descriptor.root_path)
    missing = cli.invoke(
        app, ["platform-run", str(plan_path), "--subprocess"]
    )
    assert missing.exit_code == 2
    assert "not functionally verified" in missing.output


def test_platform_plan_profiles_the_managed_interpreter_and_pins_its_lock(
    tmp_path, monkeypatch
):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, result = _create(tmp_path, fake, "plan-env")
    monkeypatch.setattr(
        manager_module, "EnvironmentManager", lambda root=None: manager
    )
    cli = CliRunner()
    planned = cli.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--environment",
            "plan-env",
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.stdout)["run_plan"]
    assert plan["environment_ref"]["id"] == "plan-env"
    assert plan["environment_ref"]["hash"]["value"] == result.lock.lock_hash
    capability_call = next(
        call for call in reversed(fake.calls) if call["phase"] == "capability_probe"
    )
    assert capability_call["argv"][0] == result.descriptor.python_executable
