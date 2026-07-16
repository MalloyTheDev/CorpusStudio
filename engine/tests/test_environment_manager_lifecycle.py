"""Reference-backend Environment Manager lifecycle tests.

All package installation and framework behavior is faked in temporary directories. The real command
runner is tested only with tiny stdlib Python commands; default CI needs no network and no GPU.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
from threading import Event, Thread
import time
from typing import Any
import zipfile

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli_module
import corpus_studio.platform.build_provenance as build_provenance
import corpus_studio.platform.environment_manager as manager_module
from corpus_studio.cli import app
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    EnvironmentLock,
    EnvironmentInstallation,
    InstallStep,
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
from corpus_studio.platform.process_control import terminate_process_tree
from corpus_studio.platform.enums import EnvironmentState, FailureTaxonomy, OperatingSystem
from corpus_studio.platform.environments import PYPI_INDEX_URL, get_recipe, resolution_digest


def _host_os() -> OperatingSystem:
    return OperatingSystem.windows if os.name == "nt" else OperatingSystem.linux


def _record_hash(name: str, version: str) -> str:
    return hashlib.sha256(f"{name}=={version}".encode()).hexdigest()


def _package(name: str, version: str = "1.0") -> dict[str, Any]:
    return {
        "name": name,
        "normalized_name": name.lower().replace("_", "-"),
        "version": version,
        "record_sha256": _record_hash(name, version),
        "direct_url": None,
        "installer": "pip",
        "requested": True,
        "record_integrity": "verified",
        "record_count_semantics": "all_record_rows_v2",
        "record_entries": 2,
        "record_verified_entries": 2,
        "record_failed_entries": [],
        "installed_files_sha256": _record_hash(f"{name}-files", version),
        "installed_file_count": 2,
        "metadata_sha256": None,
        "dependencies": ["packaging>=23"],
        "direct_url_parse_error": False,
    }


@dataclass
class FakeEnvironmentRunner:
    cuda: bool = False
    import_ok: bool = True
    functional_ok: bool = True
    hardware_ok: bool = True
    complete_probe_ok: bool = False
    # Independent complete-tuple identities: math readiness-v2 vs flash readiness-v1.
    complete_probe_name: str = "cuda_qlora_math_execution"
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
        elif phase == "framework_inventory":
            stdout = json.dumps(self._torch_payload()) + "\n"
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
            stdout = json.dumps(self._capability_payload()) + "\n"

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
        self._write_pip_report(argv)
        return CommandOutcome(exit_code, timed_out=timed_out, cancelled=cancelled)

    def _write_pip_report(self, argv: list[str]) -> None:
        if "--report" not in argv:
            return
        path = Path(argv[argv.index("--report") + 1])
        pytorch_prerequisites = {
            "cuda-pathfinder",
            "jinja2",
            "markupsafe",
            "setuptools",
            "typing-extensions",
        }
        if "upgrade-pip" in path.name:
            selected = [item for item in self.packages if item["name"].lower() == "pip"]
        elif "install-pytorch-prerequisites" in path.name:
            selected = [
                item
                for item in self.packages
                if item["normalized_name"] in pytorch_prerequisites
            ]
        elif "install-torch" in path.name:
            selected = [item for item in self.packages if item["name"].lower() == "torch"]
        elif "install-worker" in path.name:
            selected = [
                item
                for item in self.packages
                if item["normalized_name"] == "corpus-studio-engine"
            ]
        else:
            selected = [
                item
                for item in self.packages
                if item["name"].lower() not in {"pip", "torch"}
                and item["normalized_name"] != "corpus-studio-engine"
                and item["normalized_name"] not in pytorch_prerequisites
            ]
        installs = []
        for item in selected:
            worker = item["normalized_name"] == "corpus-studio-engine"
            url = (
                Path(argv[-1]).resolve().as_uri()
                if worker
                else f"https://download.pytorch.org/whl/fake/{item['name']}-{item['version']}.whl"
                if item["normalized_name"] == "torch"
                else f"https://files.pythonhosted.org/{item['name']}-{item['version']}.whl"
            )
            installs.append(
                {
                    "download_info": {
                        "url": url,
                        "archive_info": {
                            "hashes": {
                                "sha256": manager_module._hash_file(Path(argv[-1]))
                                if worker
                                else _record_hash(item["name"], item["version"])
                            }
                        },
                    },
                    "is_direct": worker,
                    "requested": True,
                    "metadata": {"name": item["name"], "version": item["version"]},
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": "1", "install": installs}), encoding="utf-8")

    def _capability_payload(self) -> dict[str, Any]:
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
        if not self.complete_probe_ok:
            return {
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
        flash = self.complete_probe_name == "cuda_qlora_sdpa_flash_execution"
        attention_impl = "sdpa" if flash else "math"
        attention_kernel = "torch_sdpa_flash" if flash else "torch_sdpa_math"
        combination = {
            "runtime_mode": "training",
            "device": "cuda",
            "precision": "bf16",
            "quantization": "nf4",
            "adapter_method": "qlora",
            "attention_impl": attention_impl,
            "attention_kernel": attention_kernel,
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": self.complete_probe_name,
        }
        package_versions = {
            item["normalized_name"]: item["version"]
            for item in self.packages
            if item["normalized_name"]
            in {
                "accelerate",
                "bitsandbytes",
                "datasets",
                "peft",
                "safetensors",
                "tokenizers",
                "torch",
                "transformers",
                "trl",
            }
        }
        memory = {
            "gpu_allocator_scope": "pytorch_cuda_allocator_process",
            "gpu_device_scope": "nvidia_smi_current_process",
            "host_memory_scope": "current_process_rss",
            "baseline_gpu_allocated_bytes": 1,
            "baseline_gpu_reserved_bytes": 2,
            "peak_gpu_allocated_bytes": 3,
            "peak_gpu_reserved_bytes": 4,
            "baseline_nvidia_smi_process_bytes": 5,
            "peak_nvidia_smi_process_bytes": 6,
            "baseline_host_rss_bytes": 7,
            "peak_host_rss_bytes": 8,
            "duration_seconds": 0.5,
        }
        if flash:
            memory.update(
                {
                    "forward_duration_seconds": 0.1,
                    "backward_duration_seconds": 0.2,
                    "optimizer_step_duration_seconds": 0.05,
                    "gpu_temperature_celsius": 42.0,
                    "gpu_power_watts": 80.0,
                }
            )
        tuple_result = {
            "probe": self.complete_probe_name,
            "outcome": "PASS",
            "measured": {
                "loss": 1.0,
                "reload_loss": 1.0,
                "adapter_weight_bytes": 128,
                "adapter_round_trip_verified": True,
                "runtime": {"packages": package_versions},
                "memory": memory,
                "configuration": {
                    "compute_dtype": "bf16",
                    "forward_autocast": "bf16",
                    "quantization": "nf4",
                    "double_quantization": True,
                    "attention_api": "sdpa",
                    "attention_kernel": attention_kernel,
                    "forced_sdp_backend": "FLASH_ATTENTION" if flash else "MATH",
                    "attention_toggles": {
                        "flash_sdp_enabled": flash,
                        "memory_efficient_sdp_enabled": False,
                        "math_sdp_enabled": not flash,
                    },
                    "attention_toggles_during": {
                        "flash_sdp_enabled": flash,
                        "memory_efficient_sdp_enabled": False,
                        "math_sdp_enabled": not flash,
                    },
                    "device_map": {"": "cuda:0"},
                    "target_modules": "all-linear",
                    "gradient_checkpointing": True,
                    "optimizer": "adamw_torch",
                    "batch_size": 1,
                    "sequence_length": 8,
                    "lora_r": 2,
                    "lora_alpha": 4,
                    "seed": 0,
                },
            },
            "proves": {
                "precision": ["bf16"],
                "quantization": ["nf4"],
                "attention": ["sdpa"] if flash else ["math", "sdpa"],
                "attention_kernel": [attention_kernel],
                "adapter": ["qlora"],
                "loss": ["cross_entropy"],
                "optimizer": ["adamw_torch"],
                "checkpoint": ["adapter_only"],
            },
            "execution_combinations": [combination],
        }
        bnb_result = {
            "probe": "bnb_4bit_load",
            "outcome": "PASS",
            "proves": {"quantization": ["nf4"]},
        }
        return {
            "profile": {
                "environment_signature": signature,
                "host": {"os": _host_os().value},
                "gpus": gpus,
            },
            "capability_report": {
                "backend_id": "corpus_studio",
                "environment_ref": {"id": signature},
                "readiness": "ready",
                "bitsandbytes_ok": True,
                "probe_results": [bnb_result, tuple_result],
                "effective_capabilities": {
                    "precision_modes": ["bf16"],
                    "quantization_modes": ["nf4"],
                    "attention_impls": ["sdpa"] if flash else ["math", "sdpa"],
                    "attention_kernels": [attention_kernel],
                    "adapter_methods": ["qlora"],
                    "loss_impls": ["cross_entropy"],
                    "optimizers": ["adamw_torch"],
                    "checkpoint_impls": ["adapter_only"],
                    "execution_combinations": [combination],
                },
            },
        }

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
        script = argv[argv.index("-c") + 1] if "-c" in argv else ""
        if "metadata.distributions" in script:
            return "lock"
        if "framework_file" in script:
            return "framework_inventory"
        if "corpus_studio.platform.worker" in script and "bitsandbytes" in script:
            return "import_probe"
        if "checkpoint_reload" in script:
            return "functional_probe"
        if "build_environment_profile" in script and "run_capability_probes" in script:
            return "capability_probe"
        if '"cuda_available"' in script:
            return "hardware_probe"
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
            "torch": {"version": None, "build": None, "cuda": None, "compute_capability": None},
        }

    def _torch_payload(self) -> dict[str, Any]:
        torch_version = next(
            item["version"]
            for item in self.packages
            if item["normalized_name"] == "torch"
        )
        return {
            "version": torch_version,
            "build": "fake-git-build",
            "cuda": self.cuda_runtime if self.cuda else None,
            "compute_capability": self.compute_capability if self.cuda else None,
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


# Fixed, canonical (40-char lowercase hex) synthetic identities for fixture wheels. The env-manager
# admission gate requires EMBEDDED canonical build provenance carrying BOTH a source_commit and a
# required_git_ancestor floor; fixtures embed both so they exercise the real gate rather than bypassing
# it. `with_provenance=False` builds the no-provenance shape; `provenance_ancestor=None` builds the
# inadmissible source-commit-only shape.
_FIXTURE_SOURCE_COMMIT = "b17e57ed0b17e57ed0b17e57ed0b17e57ed0b17e"
_FIXTURE_REQUIRED_ANCESTOR = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def _worker_wheel(
    tmp_path: Path,
    *,
    marker: str = "worker-v1",
    entry_points: str | None = None,
    with_provenance: bool = True,
    provenance_commit: str = _FIXTURE_SOURCE_COMMIT,
    provenance_ancestor: str | None = _FIXTURE_REQUIRED_ANCESTOR,
) -> Path:
    path = tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = "corpus_studio_engine-1.3.0.dist-info/METADATA"
    record_path = "corpus_studio_engine-1.3.0.dist-info/RECORD"
    members = {
        metadata_path: b"Metadata-Version: 2.1\nName: corpus-studio-engine\nVersion: 1.3.0\n",
        "corpus_studio/readiness_marker.txt": marker.encode("utf-8"),
    }
    if entry_points is not None:
        members["corpus_studio_engine-1.3.0.dist-info/entry_points.txt"] = (
            entry_points.encode("utf-8")
        )
    record_lines = []
    for member_name, member_bytes in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(member_bytes).digest()).decode().rstrip("=")
        record_lines.append(f"{member_name},sha256={digest},{len(member_bytes)}")
    record_lines.append(f"{record_path},,")
    members[record_path] = ("\n".join(record_lines) + "\n").encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, member_bytes in members.items():
            archive.writestr(member_name, member_bytes)
    if with_provenance:
        extra = (
            {"required_git_ancestor": provenance_ancestor}
            if provenance_ancestor is not None
            else None
        )
        build_provenance.stamp_wheel_with_provenance(
            path,
            build_provenance.build_provenance_document(
                source_commit=provenance_commit, extra=extra
            ),
            external_copy=False,
        )
    return path


def _readiness_packages() -> list[dict[str, Any]]:
    versions = {
        "pip": "26.1.2",
        "cuda-pathfinder": "1.2.2",
        "setuptools": "78.1.0",
        "typing-extensions": "4.15.0",
        "jinja2": "3.1.6",
        "markupsafe": "3.0.3",
        "pydantic": "2.13.4",
        "typer": "0.26.8",
        "orjson": "3.11.9",
        "torch": "2.11.0+cu128",
        "transformers": "5.13.1",
        "peft": "0.19.1",
        "trl": "1.8.0",
        "accelerate": "1.14.0",
        "datasets": "5.0.0",
        "bitsandbytes": "0.49.2",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "corpus-studio-engine": "1.3.0",
    }
    return [_package(name, version) for name, version in versions.items()]


def _manager_and_readiness_resolution(
    tmp_path: Path,
    *,
    complete_probe_ok: bool = True,
    recipe_id: str = "backend-corpus-studio-readiness-v2",
    complete_probe_name: str = "cuda_qlora_math_execution",
) -> tuple[EnvironmentManager, Any, FakeEnvironmentRunner, Path]:
    wheel = _worker_wheel(tmp_path / "artifacts")
    packages = _readiness_packages()
    worker_package = next(
        item for item in packages if item["normalized_name"] == "corpus-studio-engine"
    )
    worker_package["direct_url"] = {
        "url": wheel.resolve().as_uri(),
        "archive_info": {"hashes": {"sha256": manager_module._hash_file(wheel)}},
    }
    worker_identity = manager_module._worker_artifact_identity(wheel)
    assert worker_identity.metadata_hash is not None
    worker_package["metadata_sha256"] = worker_identity.metadata_hash.value
    worker_package["installed_file_manifest"] = [
        [path, digest]
        for path, digest in sorted(
            manager_module._worker_wheel_payload_manifest(worker_identity).items()
        )
    ]
    runner = FakeEnvironmentRunner(
        cuda=True,
        complete_probe_ok=complete_probe_ok,
        complete_probe_name=complete_probe_name,
        packages=packages,
    )
    runtime = PythonRuntime(
        runtime_id="python-readiness",
        executable=str(tmp_path / "base-python"),
        version="3.12.10",
        implementation="CPython",
        architecture="64-bit",
        platform="test-platform",
        os=OperatingSystem.linux,
        venv_available=True,
        compatible=True,
    )
    manager = EnvironmentManager(
        tmp_path / "manager",
        runner=runner,
        runtime_probe=lambda executable, requirement: runtime,
    )
    resolution = manager.preview(
        recipe_id,
        env_id=recipe_id,
        runtime_executable=runtime.executable,
        accelerator_tag="cu128",
        worker_wheel=wheel,
    )
    assert resolution.resolvable and resolution.resolution_hash
    return manager, resolution, runner, wheel


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
    assert torch.source == "wheel"
    assert torch.source_index_url
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


def test_scientific_admission_accepts_wheel_with_embedded_provenance(tmp_path):
    # End-to-end: a worker wheel carrying EMBEDDED canonical provenance is admitted through the real
    # create() path, and its identity overlay (built from the same sealed artifact) carries the commit -
    # no post-hoc overlay. This exercises build (stamp) -> artifact admission -> environment admission ->
    # telemetry identity in one path.
    from corpus_studio.platform.telemetry import worker_identity_overlay

    manager, resolution, _, wheel = _manager_and_readiness_resolution(tmp_path)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.descriptor.state in {
        EnvironmentState.hardware_verified,
        EnvironmentState.functional_probe_passed,
    }
    assert resolution.worker_artifact is not None
    overlay = worker_identity_overlay(resolution.worker_artifact)
    assert overlay.repository_commit == _FIXTURE_SOURCE_COMMIT


def _assert_create_refuses_inadmissible_wheel(tmp_path: Path, wheel: Path) -> None:
    # Drive the real create() path and assert admission refuses BEFORE any mutation (no env root, no
    # registry entry, no lock file created).
    packages = _readiness_packages()
    worker_package = next(
        item for item in packages if item["normalized_name"] == "corpus-studio-engine"
    )
    worker_package["direct_url"] = {
        "url": wheel.resolve().as_uri(),
        "archive_info": {"hashes": {"sha256": manager_module._hash_file(wheel)}},
    }
    worker_identity = manager_module._worker_artifact_identity(wheel)
    assert worker_identity.metadata_hash is not None
    worker_package["metadata_sha256"] = worker_identity.metadata_hash.value
    worker_package["installed_file_manifest"] = [
        [path, digest]
        for path, digest in sorted(
            manager_module._worker_wheel_payload_manifest(worker_identity).items()
        )
    ]
    runner = FakeEnvironmentRunner(cuda=True, packages=packages)
    runtime = PythonRuntime(
        runtime_id="python-readiness",
        executable=str(tmp_path / "base-python"),
        version="3.12.10",
        implementation="CPython",
        architecture="64-bit",
        platform="test-platform",
        os=OperatingSystem.linux,
        venv_available=True,
        compatible=True,
    )
    manager = EnvironmentManager(
        tmp_path / "manager",
        runner=runner,
        runtime_probe=lambda executable, requirement: runtime,
    )
    resolution = manager.preview(
        "backend-corpus-studio-readiness-v2",
        env_id="backend-corpus-studio-readiness-v2",
        runtime_executable=runtime.executable,
        accelerator_tag="cu128",
        worker_wheel=wheel,
    )
    with pytest.raises(manager_module.EnvironmentManagerError, match="inadmissible build provenance"):
        manager.create(resolution, confirmed_resolution_hash=resolution.resolution_hash or "")
    # Non-mutating: no environment root and no lock were created by the refused admission.
    assert not manager.environment_root("backend-corpus-studio-readiness-v2").exists()


def test_scientific_admission_refuses_wheel_without_embedded_provenance(tmp_path):
    # The v7 defect shape (no embedded provenance at all) is refused at admission, non-mutatingly.
    _assert_create_refuses_inadmissible_wheel(
        tmp_path, _worker_wheel(tmp_path / "artifacts", with_provenance=False)
    )


def test_scientific_admission_refuses_source_commit_only_wheel(tmp_path):
    # A wheel embedding source_commit but NO required_git_ancestor (the exact shape the manager gate
    # sees, since it supplies neither an expected floor nor a repo) is refused BEFORE any mutation.
    _assert_create_refuses_inadmissible_wheel(
        tmp_path, _worker_wheel(tmp_path / "artifacts", provenance_ancestor=None)
    )


def test_readiness_v2_plan_is_stable_hash_bound_and_plan_only(tmp_path):
    manager, first, _, wheel = _manager_and_readiness_resolution(tmp_path)
    second = manager.preview(
        "backend-corpus-studio-readiness-v2",
        env_id="backend-corpus-studio-readiness-v2",
        runtime_executable=first.runtime.executable if first.runtime else "python",
        accelerator_tag="cu128",
        worker_wheel=wheel,
    )
    assert first == second
    assert first.worker_artifact and first.worker_artifact.content_hash.value
    assert first.required_execution_probe is not None
    assert not manager.root.exists()
    original_hash = first.resolution_hash
    _worker_wheel(wheel.parent, marker="worker-v2")
    changed = manager.preview(
        "backend-corpus-studio-readiness-v2",
        env_id="backend-corpus-studio-readiness-v2",
        runtime_executable=first.runtime.executable if first.runtime else "python",
        accelerator_tag="cu128",
        worker_wheel=wheel,
    )
    assert changed.resolution_hash != original_hash
    with pytest.raises(EnvironmentManagerError, match="canonical plan|worker wheel changed"):
        manager.create(first, confirmed_resolution_hash=original_hash or "")
    assert not manager.root.exists()


def test_readiness_install_records_hash_backed_pytorch_prerequisites_before_torch(tmp_path):
    manager, resolution, _, _ = _manager_and_readiness_resolution(tmp_path)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    prerequisite_command = next(
        command
        for command in result.installation.commands
        if any(
            token.endswith("install-pytorch-prerequisites.json")
            for token in command.argv
        )
    )
    torch_command = next(
        command
        for command in result.installation.commands
        if any(token.endswith("install-torch.json") for token in command.argv)
    )
    assert result.installation.commands.index(
        prerequisite_command
    ) < result.installation.commands.index(torch_command)
    assert "--no-deps" in prerequisite_command.argv

    by_name = {
        item.normalized_name: item
        for item in result.installation.package_install_evidence
    }
    for name in (
        "cuda-pathfinder",
        "setuptools",
        "typing-extensions",
        "jinja2",
        "markupsafe",
    ):
        evidence = by_name[name]
        assert evidence.source == "pypi"
        assert evidence.source_index_url == PYPI_INDEX_URL
        assert evidence.artifact_hash is not None


def test_readiness_worker_wheel_inside_replacement_target_is_blocked(tmp_path):
    runner = FakeEnvironmentRunner(cuda=True, packages=_readiness_packages())
    runtime = PythonRuntime(
        runtime_id="python-readiness",
        executable=str(tmp_path / "base-python"),
        version="3.12.10",
        implementation="CPython",
        architecture="64-bit",
        platform="test-platform",
        os=OperatingSystem.linux,
        venv_available=True,
        compatible=True,
    )
    manager = EnvironmentManager(
        tmp_path / "manager",
        runner=runner,
        runtime_probe=lambda executable, requirement: runtime,
    )
    wheel = _worker_wheel(
        manager.environment_root("backend-corpus-studio-readiness-v2") / "authorization"
    )
    resolution = manager.preview(
        "backend-corpus-studio-readiness-v2",
        env_id="backend-corpus-studio-readiness-v2",
        runtime_executable=runtime.executable,
        accelerator_tag="cu128",
        worker_wheel=wheel,
    )
    assert resolution.resolvable is False
    assert any("cannot live inside" in reason for reason in resolution.blocking_reasons)

def test_readiness_v2_seals_only_after_complete_tuple_and_records_memory(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.descriptor.state == EnvironmentState.hardware_verified
    assert result.lock is not None and result.descriptor.lock_ref is not None
    assert result.lock.probe_evidence is not None
    assert result.installation.pre_probe_inventory is not None
    assert result.installation.post_probe_inventory is not None
    assert (
        result.installation.pre_probe_inventory.evidence_hash
        == result.installation.post_probe_inventory.evidence_hash
    )
    assert result.installation.pre_probe_inventory.evidence_id.startswith("inventory-")
    assert result.lock.lock_id.startswith("lock-")
    memory = result.lock.probe_evidence.memory
    assert memory.gpu_allocator_scope == "pytorch_cuda_allocator_process"
    assert memory.gpu_device_scope == "nvidia_smi_current_process"
    assert memory.host_memory_scope == "current_process_rss"
    assert memory.peak_gpu_allocated_bytes >= memory.baseline_gpu_allocated_bytes
    assert memory.peak_host_rss_bytes >= memory.baseline_host_rss_bytes
    command_phases = [item.phase for item in result.installation.commands]
    capability_index = command_phases.index("capability_probe")
    inventory_indexes = [
        index for index, phase in enumerate(command_phases) if phase == "inventory"
    ]
    assert inventory_indexes[1] < capability_index < inventory_indexes[2]
    assert runner.calls[-1]["phase"] == "framework_inventory"
    assert all(item.source_evidence_reason or item.source != "unknown" for item in result.lock.packages)
    raw_lock = result.lock.model_dump(mode="json")
    for field_name in (
        "forward_duration_seconds",
        "backward_duration_seconds",
        "optimizer_step_duration_seconds",
        "gpu_temperature_celsius",
        "gpu_power_watts",
    ):
        raw_lock["probe_evidence"]["memory"].pop(field_name)
    restored = EnvironmentLock.model_validate(raw_lock)
    assert manager._lock_digest(restored) == result.lock.lock_hash
    assert restored.probe_evidence is not None
    assert (
        manager._probe_evidence_digest(restored.probe_evidence)
        == restored.probe_evidence.evidence_hash
    )


def test_failed_or_independently_unionable_probes_never_seal_readiness_v2(tmp_path):
    manager, resolution, _, _ = _manager_and_readiness_resolution(
        tmp_path, complete_probe_ok=False
    )
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.descriptor.state == EnvironmentState.incompatible
    assert result.lock is None
    assert result.descriptor.lock_ref is None
    assert not (manager.registry_root / result.descriptor.env_id / "locks").exists()
    assert result.health.probe_results[-1].probe == "cuda_qlora_math_execution"
    assert result.installation.retry_requires_recreate is True


def test_worker_pip_hash_mismatch_stops_before_any_installed_import(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    write_report = runner._write_pip_report

    def write_tampered_report(argv):
        write_report(argv)
        if "--report" not in argv:
            return
        report_path = Path(argv[argv.index("--report") + 1])
        if "install-worker" not in report_path.name:
            return
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "f" * 64
        report_path.write_text(json.dumps(payload), encoding="utf-8")

    runner._write_pip_report = write_tampered_report  # type: ignore[method-assign]
    with pytest.raises(EnvironmentManagerError, match="reviewed worker wheel"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_pip_report_cannot_replace_source_evidence_from_an_earlier_step(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    write_report = runner._write_pip_report

    def write_colliding_report(argv):
        write_report(argv)
        if "--report" not in argv:
            return
        report_path = Path(argv[argv.index("--report") + 1])
        if "install-worker" not in report_path.name:
            return
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["install"].append(
            {
                "download_info": {
                    "url": "https://files.pythonhosted.org/torch-forged.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": False,
                "requested": False,
                "metadata": {"name": "torch", "version": "2.11.0+cu128"},
            }
        )
        report_path.write_text(json.dumps(payload), encoding="utf-8")

    runner._write_pip_report = write_colliding_report  # type: ignore[method-assign]
    with pytest.raises(EnvironmentManagerError, match="repeats a distribution"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_unverified_pre_probe_files_stop_before_any_installed_import(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    torch_package = next(item for item in runner.packages if item["name"] == "torch")
    torch_package["record_integrity"] = "failed"
    torch_package["record_count_semantics"] = None
    torch_package["record_failed_entries"] = ["torch/library.py"]
    torch_package["installed_files_sha256"] = None
    with pytest.raises(EnvironmentManagerError, match="before readiness probes"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_inventory_rejects_non_scalar_record_count_semantics(tmp_path):
    fake = FakeEnvironmentRunner()
    manager, _, result = _create(tmp_path, fake)
    payload = json.loads(json.dumps(fake._lock_payload()))
    payload["packages"][0]["record_count_semantics"] = ["all_record_rows_v2"]

    with pytest.raises(EnvironmentManagerError, match="invalid RECORD count semantics"):
        manager._inventory_from_payload(
            result.descriptor,
            result.lock.recipe_ref,
            result.lock.index_urls,
            payload,
        )


def test_installed_worker_payload_must_match_the_reviewed_wheel(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    worker_package = next(
        item
        for item in runner.packages
        if item["normalized_name"] == "corpus-studio-engine"
    )
    worker_package["installed_file_manifest"][0][1] = "0" * 64
    with pytest.raises(EnvironmentManagerError, match="files differ from the reviewed wheel"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_installed_worker_cannot_claim_unreviewed_importable_files(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    worker_package = next(
        item
        for item in runner.packages
        if item["normalized_name"] == "corpus-studio-engine"
    )
    worker_package["installed_file_manifest"].append(["torch.py", "0" * 64])
    with pytest.raises(EnvironmentManagerError, match="absent from the reviewed wheel"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "framework_inventory" for call in runner.calls)
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_installed_worker_cannot_claim_an_undeclared_generated_script(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    worker_package = next(
        item
        for item in runner.packages
        if item["normalized_name"] == "corpus-studio-engine"
    )
    worker_package["installed_file_manifest"].append(
        ["../../../bin/not-declared-by-entry-points", "0" * 64]
    )
    with pytest.raises(EnvironmentManagerError, match="absent from the reviewed wheel"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert all(call["phase"] != "import_probe" for call in runner.calls)


def test_readiness_flash_v1_seals_only_after_forced_flash_tuple(tmp_path):
    manager, resolution, _, _ = _manager_and_readiness_resolution(
        tmp_path,
        recipe_id="backend-corpus-studio-readiness-flash-v1",
        complete_probe_name="cuda_qlora_sdpa_flash_execution",
    )
    assert resolution.required_execution_probe is not None
    assert resolution.required_execution_probe.probe == "cuda_qlora_sdpa_flash_execution"
    assert resolution.required_execution_probe.flash_sdp_enabled is True
    assert resolution.required_execution_probe.math_sdp_enabled is False
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.descriptor.state == EnvironmentState.hardware_verified
    assert result.lock is not None and result.lock.probe_evidence is not None
    assert result.lock.probe_evidence.tuple_result.probe == "cuda_qlora_sdpa_flash_execution"
    memory = result.lock.probe_evidence.memory
    assert memory.forward_duration_seconds == 0.1
    assert memory.backward_duration_seconds == 0.2
    assert memory.optimizer_step_duration_seconds == 0.05
    assert memory.gpu_temperature_celsius == 42.0
    assert memory.gpu_power_watts == 80.0


def test_failed_flash_tuple_never_seals_readiness_flash_v1(tmp_path):
    manager, resolution, _, _ = _manager_and_readiness_resolution(
        tmp_path,
        complete_probe_ok=False,
        recipe_id="backend-corpus-studio-readiness-flash-v1",
        complete_probe_name="cuda_qlora_sdpa_flash_execution",
    )
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.descriptor.state == EnvironmentState.incompatible
    assert result.lock is None
    assert result.health.probe_results[-1].probe == "cuda_qlora_sdpa_flash_execution"
    assert result.installation.retry_requires_recreate is True
    assert result.health.remediation is not None
    assert "env-recreate" in result.health.remediation


def test_complete_probe_configuration_is_bound_to_the_required_tuple(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(
        tmp_path,
        recipe_id="backend-corpus-studio-readiness-flash-v1",
        complete_probe_name="cuda_qlora_sdpa_flash_execution",
    )
    required = resolution.required_execution_probe
    assert required is not None
    payload = runner._capability_payload()
    report = payload["capability_report"]
    tuple_result = next(
        item
        for item in report["probe_results"]
        if item["probe"] == "cuda_qlora_sdpa_flash_execution"
    )
    tuple_result["measured"]["configuration"]["forward_autocast"] = "fp32"
    with pytest.raises(EnvironmentManagerError, match="forward_autocast"):
        manager._complete_probe_evidence(payload, required)


def test_legacy_math_probe_shape_remains_a_narrow_rollback_identity(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    required = resolution.required_execution_probe
    assert required is not None
    payload = runner._capability_payload()
    tuple_result = next(
        item
        for item in payload["capability_report"]["probe_results"]
        if item["probe"] == "cuda_qlora_math_execution"
    )
    configuration = tuple_result["measured"]["configuration"]
    for field_name in (
        "forward_autocast",
        "attention_kernel",
        "forced_sdp_backend",
        "attention_toggles_during",
        "batch_size",
        "sequence_length",
        "lora_r",
        "lora_alpha",
        "seed",
    ):
        configuration.pop(field_name)
    tuple_result["measured"].pop("adapter_round_trip_verified")
    with pytest.raises(EnvironmentManagerError):
        manager._complete_probe_evidence(payload, required)
    _, evidence = manager._complete_probe_evidence(
        payload,
        required,
        legacy_math_configuration=True,
    )
    assert evidence.required_spec.probe == "cuda_qlora_math_execution"
    legacy_lock = EnvironmentLock(
        lock_id="lock-legacy-math",
        recipe_ref=resolution.recipe_ref,
        manager_version="1.1.0",
        probe_evidence=evidence,
    )
    assert manager._locked_probe_evidence_mismatch(legacy_lock, required) is False
    assert (
        manager._locked_probe_evidence_mismatch(
            legacy_lock.model_copy(update={"manager_version": "1.2.0"}),
            required,
        )
        is True
    )


def test_new_readiness_lock_requires_adapter_equality_evidence(tmp_path):
    manager, resolution, runner, _ = _manager_and_readiness_resolution(tmp_path)
    required = resolution.required_execution_probe
    assert required is not None
    _, evidence = manager._complete_probe_evidence(runner._capability_payload(), required)
    lock = EnvironmentLock(
        lock_id="lock-current-math",
        recipe_ref=resolution.recipe_ref,
        manager_version=manager_module.MANAGER_VERSION,
        probe_evidence=evidence,
    )
    assert manager._locked_probe_evidence_mismatch(lock, required) is False
    measured = dict(evidence.tuple_result.measured)
    measured.pop("adapter_round_trip_verified")
    weakened_result = evidence.tuple_result.model_copy(update={"measured": measured})
    weakened = evidence.model_copy(
        update={"tuple_result": weakened_result, "evidence_hash": "0" * 64}
    )
    weakened = weakened.model_copy(
        update={"evidence_hash": manager._probe_evidence_digest(weakened)}
    )
    assert (
        manager._locked_probe_evidence_mismatch(
            lock.model_copy(update={"probe_evidence": weakened}), required
        )
        is True
    )


def test_lock_finalization_refuses_missing_complete_probe_evidence(tmp_path):
    manager, resolution, _, _ = _manager_and_readiness_resolution(tmp_path)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.lock is not None
    assert result.installation.post_probe_inventory is not None
    recipe = get_recipe("backend-corpus-studio-readiness-v2")
    assert recipe is not None
    with pytest.raises(EnvironmentManagerError, match="cannot be finalized"):
        manager._finalize_lock(
            result.installation.post_probe_inventory,
            resolution=resolution,
            recipe=recipe,
            installation=result.installation,
            probe_evidence=None,
        )


def test_readiness_v2_worker_artifact_and_dependency_drift_are_detected(tmp_path):
    manager, resolution, runner, wheel = _manager_and_readiness_resolution(tmp_path)
    result = manager.create(
        resolution, confirmed_resolution_hash=resolution.resolution_hash or ""
    )
    assert result.lock is not None
    _worker_wheel(wheel.parent, marker="tampered-worker")
    report = manager.health(result.descriptor.env_id)
    assert report.state == EnvironmentState.drifted
    assert any("worker artifact identity changed" in item for item in report.drifted_packages)

    _worker_wheel(wheel.parent)
    torch_package = next(item for item in runner.packages if item["name"] == "torch")
    torch_package["version"] = "2.11.1+cu128"
    torch_package["record_sha256"] = _record_hash("torch", torch_package["version"])
    report = manager.health(result.descriptor.env_id)
    assert report.state == EnvironmentState.drifted
    assert any("torch" in item.lower() for item in report.drifted_packages)


def test_capability_snapshot_is_proved_inside_the_managed_interpreter(
    tmp_path, monkeypatch
):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, result = _create(tmp_path, fake)
    raw_payload = fake._capability_payload

    def version_only_payload():
        payload = raw_payload()
        payload["profile"]["packages"] = [
            {"name": "torch", "version": "2.7.1+cu128"},
            {"name": "liger-kernel", "version": None},
        ]
        payload["capability_report"]["installed_packages"] = [
            {"name": "torch", "version": "2.7.1+cu128"}
        ]
        return payload

    monkeypatch.setattr(fake, "_capability_payload", version_only_payload)
    profile, report = manager.capability_snapshot("ref-env")
    assert profile.environment_signature == "c" * 64
    assert report.readiness == "ready"
    sealed_torch = next(item for item in result.lock.packages if item.name == "torch")
    assert profile.packages[0] == sealed_torch
    assert report.installed_packages == [sealed_torch]
    assert profile.packages[1].version is None
    assert profile.packages[1].record_integrity == "missing"
    assert profile.packages[1].record_entries == 0
    call = next(call for call in reversed(fake.calls) if call["phase"] == "capability_probe")
    assert call["argv"][0] == result.descriptor.python_executable


def test_capability_snapshot_rejects_version_only_identity_outside_the_sealed_lock(
    tmp_path, monkeypatch
):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, _ = _create(tmp_path, fake)
    raw_payload = fake._capability_payload

    def mismatched_payload():
        payload = raw_payload()
        payload["profile"]["packages"] = [
            {"name": "torch", "version": "0.0"}
        ]
        payload["capability_report"]["installed_packages"] = [
            {"name": "torch", "version": "0.0"}
        ]
        return payload

    monkeypatch.setattr(fake, "_capability_payload", mismatched_payload)
    with pytest.raises(
        EnvironmentManagerError,
        match="package identity does not match the sealed lock: torch",
    ):
        manager.capability_snapshot("ref-env")


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


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_environment_lock_timeout_must_be_finite_and_positive(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        EnvironmentManager(tmp_path / "manager", lock_timeout_seconds=timeout)


def test_environment_lock_directory_cannot_redirect_through_a_symlink(tmp_path):
    manager = EnvironmentManager(tmp_path / "manager", lock_timeout_seconds=0.05)
    manager.root.mkdir(parents=True)
    outside = tmp_path / "outside-locks"
    outside.mkdir()
    try:
        (manager.root / ".locks").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows may deny symlink creation.
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(EnvironmentManagerError, match="lock path escapes|symbolic link"):
        with manager.environment_lease("safe-env"):
            pass


def test_environment_lease_is_cross_process_bounded_and_released(tmp_path):
    root = tmp_path / "manager"
    ready = tmp_path / "child-ready"
    release = tmp_path / "child-release"
    script = "\n".join(
        [
            "import pathlib, sys, time",
            "from corpus_studio.platform.environment_manager import EnvironmentManager",
            "root, ready, release = map(pathlib.Path, sys.argv[1:])",
            "manager = EnvironmentManager(root, lock_timeout_seconds=5.0)",
            "with manager.environment_lease('shared-env', operation='child lease'):",
            "    ready.write_text('ready', encoding='utf-8')",
            "    while not release.exists():",
            "        time.sleep(0.01)",
        ]
    )
    process = subprocess.Popen(  # noqa: S603 - fixed test interpreter and local literal script.
        [sys.executable, "-c", script, str(root), str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=1)

        contender = EnvironmentManager(root, lock_timeout_seconds=0.05)
        with pytest.raises(EnvironmentManagerError, match="is busy") as captured:
            with contender.environment_lease("shared-env", operation="competing lease"):
                pass
        assert captured.value.failure.taxonomy == FailureTaxonomy.TIMEOUT
    finally:
        release.write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)

    # The OS lock and process-local guard both release on context exit.
    with EnvironmentManager(root, lock_timeout_seconds=0.2).environment_lease("shared-env"):
        pass


def test_competing_creator_cannot_corrupt_the_winning_installation(tmp_path):
    fake = FakeEnvironmentRunner()
    manager, resolution = _manager_and_resolution(tmp_path, fake, "race-env")
    manager.lock_timeout_seconds = 2.0
    started = Event()
    release = Event()
    original_runner = manager.runner

    def blocking_runner(argv, **kwargs):
        if fake._phase(argv) == "create_venv":
            started.set()
            assert release.wait(timeout=3)
        return original_runner(argv, **kwargs)

    manager.runner = blocking_runner
    contender = EnvironmentManager(
        manager.root,
        runner=blocking_runner,
        runtime_probe=manager.runtime_probe,
        engine_source=manager.engine_source,
        lock_timeout_seconds=0.05,
    )
    completed: list[Any] = []
    failures: list[BaseException] = []

    def create_winner() -> None:
        try:
            completed.append(
                manager.create(
                    resolution,
                    confirmed_resolution_hash=resolution.resolution_hash or "",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports the captured failure.
            failures.append(exc)

    thread = Thread(target=create_winner)
    thread.start()
    assert started.wait(timeout=2)
    try:
        with pytest.raises(EnvironmentManagerError, match="is busy") as captured:
            contender.create(
                resolution,
                confirmed_resolution_hash=resolution.resolution_hash or "",
            )
        assert captured.value.failure.taxonomy == FailureTaxonomy.TIMEOUT
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert len(completed) == 1
    assert completed[0].descriptor.state == EnvironmentState.functional_probe_passed
    assert manager.load_descriptor("race-env") == completed[0].descriptor
    installations = list((manager.registry_root / "race-env" / "installations").glob("*.json"))
    assert len(installations) == 1


def test_manager_lock_serializes_distinct_environment_mutations(tmp_path):
    fake = FakeEnvironmentRunner()
    manager, first = _manager_and_resolution(tmp_path, fake, "first-env")
    assert first.runtime is not None
    second = manager.preview(
        "backend-corpus-studio",
        env_id="second-env",
        runtime_executable=first.runtime.executable,
        accelerator_tag=first.accelerator_tag,
    )
    started = Event()
    release = Event()
    original_runner = manager.runner

    def blocking_runner(argv, **kwargs):
        if fake._phase(argv) == "create_venv":
            started.set()
            assert release.wait(timeout=3)
        return original_runner(argv, **kwargs)

    manager.runner = blocking_runner
    contender = EnvironmentManager(
        manager.root,
        runner=blocking_runner,
        runtime_probe=manager.runtime_probe,
        engine_source=manager.engine_source,
        lock_timeout_seconds=0.05,
    )
    failures: list[BaseException] = []

    def create_first() -> None:
        try:
            manager.create(first, confirmed_resolution_hash=first.resolution_hash or "")
        except BaseException as exc:  # pragma: no cover - assertion reports the captured failure.
            failures.append(exc)

    thread = Thread(target=create_first)
    thread.start()
    assert started.wait(timeout=2)
    try:
        with pytest.raises(EnvironmentManagerError, match="environment manager is busy"):
            contender.create(second, confirmed_resolution_hash=second.resolution_hash or "")
        assert not contender.environment_root("second-env").exists()
        assert not (contender.registry_root / "second-env").exists()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []


def test_environment_lease_is_reentrant_for_one_complete_health_transaction(tmp_path):
    manager, _, result = _create(tmp_path, FakeEnvironmentRunner(), "lease-env")
    with manager.environment_lease("lease-env", operation="outer transaction"):
        assert manager.load_descriptor("lease-env") == result.descriptor
        assert manager.load_lock("lease-env") == result.lock
        assert manager.health("lease-env").state == EnvironmentState.functional_probe_passed


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
    assert "--no-deps" in worker_step.argv
    assert worker_step.argv[-1] == str(engine_source.resolve())
    assert worker_step.evidence_path
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
    manager, resolution, result = _create(tmp_path, FakeEnvironmentRunner())
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
    with pytest.raises(EnvironmentManagerError, match="identity 'ref-env' is already recorded"):
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert manager.load_descriptor("ref-env") == removed

    for unsafe in ("../escape", "..", "bad/name"):
        with pytest.raises(EnvironmentManagerError):
            manager.environment_root(unsafe)


def test_recreate_requires_confirmations_and_refuses_sealed_identity_reuse(tmp_path):
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
    with pytest.raises(EnvironmentManagerError, match="sealed environment.*new environment id"):
        manager.recreate(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
            confirmed_remove_env_id="ref-env",
        )
    assert manager.load_descriptor("ref-env") == first.descriptor
    assert Path(first.descriptor.root_path).exists()


def test_recreate_recovers_an_unsealed_failed_attempt(tmp_path):
    fake = FakeEnvironmentRunner()
    fake.timeout_phase = "install"
    manager, resolution = _manager_and_resolution(tmp_path, fake)
    with pytest.raises(EnvironmentManagerError) as captured:
        manager.create(
            resolution,
            confirmed_resolution_hash=resolution.resolution_hash or "",
        )
    assert captured.value.failure.taxonomy == FailureTaxonomy.TIMEOUT
    failed = manager.load_descriptor("ref-env")
    assert failed.lock_ref is None

    fake.timeout_phase = None
    recovered = manager.recreate(
        resolution,
        confirmed_resolution_hash=resolution.resolution_hash or "",
        confirmed_remove_env_id="ref-env",
    )
    assert recovered.descriptor.state == EnvironmentState.functional_probe_passed
    assert recovered.installation.installation_id != failed.installation_ref.id


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
    assert report.remediation is not None
    assert "new environment id" in report.remediation
    assert "env-recreate" not in report.remediation
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


def test_install_source_evidence_sanitizes_credentials_and_preserves_unknown(tmp_path):
    report = tmp_path / "pip-report.json"
    report.write_text(
        json.dumps(
            {
                "version": "1",
                "install": [
                    {
                        "download_info": {
                            "url": "https://user:secret@example.invalid/pkg.whl?X-Amz-Signature=secret",
                            "archive_info": {"hashes": {"sha256": "a" * 64}},
                        },
                        "is_direct": True,
                        "requested": True,
                        "metadata": {"name": "Known_Pkg", "version": "1.0"},
                    },
                    {
                        "download_info": {
                            "url": "https://cdn.invalid/unknown.whl",
                            "archive_info": {"hashes": {"sha256": "b" * 64}},
                        },
                        "is_direct": False,
                        "requested": False,
                        "metadata": {"name": "Unknown.Pkg", "version": "2.0"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    step = InstallStep(
        phase="install",
        argv=["python", "-m", "pip", "install"],
        configured_index_urls=[],
    )
    captured = manager_module._install_evidence_from_report(
        report, step=step, command_id="command-001"
    )
    known = next(item for item in captured if item.normalized_name == "known-pkg")
    assert known.direct_url == "https://example.invalid/pkg.whl"
    assert "secret" not in known.model_dump_json().lower()
    unknown = next(item for item in captured if item.normalized_name == "unknown-pkg")
    assert unknown.source == "unknown"
    assert unknown.source_evidence_reason


def test_install_source_evidence_classifies_proven_local_vcs_index_and_sdist_sources(
    tmp_path,
):
    report = tmp_path / "pip-report.json"
    report.write_text(
        json.dumps(
            {
                "version": "1",
                "install": [
                    {
                        "download_info": {
                            "url": "https://files.pythonhosted.org/packages/indexed.whl",
                            "archive_info": {"hashes": {"sha256": "a" * 64}},
                        },
                        "is_direct": False,
                        "requested": True,
                        "metadata": {"name": "indexed", "version": "1.0"},
                    },
                    {
                        "download_info": {
                            "url": "file:///tmp/local-package",
                            "dir_info": {"editable": True},
                        },
                        "is_direct": True,
                        "requested": False,
                        "metadata": {"name": "local-package", "version": "2.0"},
                    },
                    {
                        "download_info": {
                            "url": "https://git.example.invalid/repository.git",
                            "vcs_info": {
                                "vcs": "git",
                                "commit_id": "deadbeef",
                                "requested_revision": "release-1",
                            },
                        },
                        "is_direct": True,
                        "metadata": {"name": "vcs-package", "version": "3.0"},
                    },
                    {
                        "download_info": {
                            "url": "https://example.invalid/source.tar.gz",
                            "archive_info": {"hashes": {"sha256": "d" * 64}},
                        },
                        "is_direct": True,
                        "metadata": {"name": "source-package", "version": "4.0"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    step = InstallStep(
        phase="install",
        argv=["python", "-m", "pip", "install"],
        configured_index_urls=["https://pypi.org/simple"],
    )

    captured = {
        item.normalized_name: item
        for item in manager_module._install_evidence_from_report(
            report, step=step, command_id="command-001"
        )
    }

    assert captured["indexed"].source == "pypi"
    assert captured["indexed"].source_index_url == "https://pypi.org/simple"
    assert captured["local-package"].source == "local"
    assert captured["local-package"].editable is True
    assert captured["vcs-package"].source == "vcs"
    assert captured["vcs-package"].vcs_commit == "deadbeef"
    assert captured["source-package"].source == "sdist"


def test_url_sanitizer_rejects_malformed_ports_and_strips_sensitive_parts():
    assert manager_module._sanitize_url(None) is None
    assert manager_module._sanitize_url("https://example.invalid:bad/simple") is None
    assert manager_module._sanitize_url("user:secret@example.invalid/path") is None
    assert manager_module._sanitize_url("https://example.invalid/path\nsecret") is None
    assert manager_module._sanitize_url("https://example.invalid/a path") is None
    assert manager_module._sanitize_url(r"https://example.invalid\@evil.invalid/pkg") is None
    assert manager_module._sanitize_url("https://example.invalid/%ZZ/pkg") is None
    assert manager_module._sanitize_url("file:relative-wheel.whl") is None
    assert manager_module._sanitize_url("https://[::1]:8443/simple") == "https://[::1]:8443/simple"
    assert (
        manager_module._sanitize_url(
            "https://user:secret@example.invalid:8443/simple?token=secret#fragment"
        )
        == "https://example.invalid:8443/simple"
    )


@pytest.mark.parametrize(
    ("direct_url", "message"),
    [
        ("not-an-object", "malformed direct_url"),
        ({"url": "relative.whl", "archive_info": {}}, "malformed direct artifact URL"),
        (
            {
                "url": "https://example.invalid/pkg.whl",
                "archive_info": {},
                "dir_info": {},
            },
            "exactly one source kind",
        ),
        (
            {
                "url": "https://example.invalid/pkg.whl",
                "archive_info": {},
                "subdirectory": "unsafe\nvalue",
            },
            "malformed subdirectory",
        ),
        (
            {
                "url": "file:///tmp/pkg",
                "dir_info": {"editable": "yes"},
            },
            "malformed editable flag",
        ),
        (
            {
                "url": "https://example.invalid/pkg.whl",
                "archive_info": {"hashes": []},
            },
            "malformed archive hashes",
        ),
        (
            {
                "url": "https://example.invalid/pkg.whl",
                "archive_info": {"hash": 7},
            },
            "malformed archive hash",
        ),
        (
            {
                "url": "https://example.invalid/repository.git",
                "vcs_info": {"vcs": "git!", "commit_id": "deadbeef"},
            },
            "malformed VCS identity",
        ),
        (
            {
                "url": "https://example.invalid/repository.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "deadbeef",
                    "requested_revision": "unsafe\nrevision",
                },
            },
            "malformed VCS revision",
        ),
    ],
)
def test_installed_direct_url_metadata_fails_closed(direct_url, message):
    with pytest.raises(EnvironmentManagerError, match=message):
        manager_module._validate_installed_direct_url(direct_url)


@pytest.mark.skipif(os.name == "nt", reason="symlink inventory test")
def test_lock_probe_rejects_symlinked_distribution_metadata(tmp_path):
    environment_root = tmp_path / "managed-env"
    site_root = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_root.mkdir(parents=True)
    external = tmp_path / "outside.dist-info"
    external.mkdir()
    (external / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: outside\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (external / "RECORD").write_text("outside.dist-info/RECORD,,\n", encoding="utf-8")
    (site_root / "outside-1.0.dist-info").symlink_to(external, target_is_directory=True)

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and reviewed embedded probe.
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            manager_module._LOCK_PROBE,
            str(environment_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "symbolic link" in completed.stderr


def test_lock_probe_rejects_unrecorded_site_package_files(tmp_path):
    environment_root = tmp_path / "managed-env"
    site_root = (
        environment_root
        / ("Lib" if os.name == "nt" else "lib")
        / ("site-packages" if os.name == "nt" else f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    )
    dist_info = site_root / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    metadata_bytes = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
    (dist_info / "METADATA").write_bytes(metadata_bytes)
    digest = base64.urlsafe_b64encode(hashlib.sha256(metadata_bytes).digest()).decode().rstrip("=")
    (dist_info / "RECORD").write_text(
        f"demo-1.0.dist-info/METADATA,sha256={digest},{len(metadata_bytes)}\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    (site_root / "unrecorded.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and reviewed embedded probe.
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            manager_module._LOCK_PROBE,
            str(environment_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "unrecorded file" in completed.stderr


def test_lock_probe_sanitizes_direct_url_before_writing_inventory_logs(tmp_path):
    environment_root = tmp_path / "managed-env"
    site_root = (
        environment_root
        / ("Lib" if os.name == "nt" else "lib")
        / (
            "site-packages"
            if os.name == "nt"
            else f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
    )
    dist_info = site_root / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    files = {
        "demo-1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
        ),
        "demo-1.0.dist-info/direct_url.json": json.dumps(
            {
                "url": (
                    "https://user:secret@example.invalid/demo.whl"
                    "?token=secret#secret"
                ),
                "archive_info": {"hashes": {"sha256": "a" * 64}},
                "private": "secret",
            }
        ).encode(),
    }
    record_lines = []
    for relative, content in files.items():
        target = site_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        record_lines.append(f"{relative},sha256={digest},{len(content)}")
    record_lines.append("demo-1.0.dist-info/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and reviewed embedded probe.
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            manager_module._LOCK_PROBE,
            str(environment_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "secret" not in (completed.stdout + completed.stderr).casefold()
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["packages"][0]["direct_url"] == {
        "url": "https://example.invalid/demo.whl",
        "archive_info": {},
    }


def test_lock_probe_proves_generated_worker_bytecode_matches_its_source(tmp_path):
    environment_root = tmp_path / "managed-env"
    site_root = (
        environment_root
        / ("Lib" if os.name == "nt" else "lib")
        / (
            "site-packages"
            if os.name == "nt"
            else f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
    )
    package_root = site_root / "corpus_studio"
    dist_info = site_root / "corpus_studio_engine-1.3.0.dist-info"
    package_root.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    source = package_root / "verified.py"
    source.write_text("VALUE = 'reviewed'\n", encoding="utf-8")
    pyc = package_root / "__pycache__" / f"verified.{sys.implementation.cache_tag}.pyc"
    pyc.parent.mkdir()
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    metadata = b"Metadata-Version: 2.1\nName: corpus-studio-engine\nVersion: 1.3.0\n"
    (dist_info / "METADATA").write_bytes(metadata)

    source_relative = source.relative_to(site_root).as_posix()
    pyc_relative = pyc.relative_to(site_root).as_posix()
    metadata_relative = (dist_info / "METADATA").relative_to(site_root).as_posix()
    record_relative = (dist_info / "RECORD").relative_to(site_root).as_posix()
    record_lines = []
    for relative, content in (
        (source_relative, source.read_bytes()),
        (metadata_relative, metadata),
    ):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        record_lines.append(f"{relative},sha256={digest},{len(content)}")
    record_lines.extend([f"{pyc_relative},,", f"{record_relative},,"])
    (dist_info / "RECORD").write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    def inspect() -> dict[str, Any]:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and reviewed embedded probe.
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                manager_module._LOCK_PROBE,
                str(environment_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout.splitlines()[-1])["packages"][0]

    verified = inspect()
    assert verified["record_integrity"] == "verified"
    assert verified["record_entries"] == 4
    assert verified["record_verified_entries"] == verified["record_entries"]
    assert verified["installed_file_count"] == verified["record_entries"]

    source.write_text("VALUE = 'unreviewed'\n", encoding="utf-8")
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    source.write_text("VALUE = 'reviewed'\n", encoding="utf-8")
    tampered = inspect()
    assert tampered["record_integrity"] == "failed"
    assert tampered["record_failed_entries"] == [pyc_relative]


@pytest.mark.parametrize(
    "install_entries",
    [
        ["not-an-object"],
        [{"metadata": {"name": "pkg", "version": "1"}}],
        [
            {
                "download_info": {
                    "url": "https://example.invalid:bad/pkg.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": "yes",
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {"url": "https://example.invalid/pkg.whl"},
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "file:///tmp/pkg",
                    "dir_info": {},
                },
                "is_direct": False,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "file:///tmp/pkg",
                    "dir_info": {"editable": "yes"},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg",
                    "dir_info": {},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/repository.git",
                    "vcs_info": {"vcs": "git!", "commit_id": "deadbeef"},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/repository.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "deadbeef",
                        "requested_revision": "unsafe\nrevision",
                    },
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg%0A.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                    "vcs_info": "not-an-object",
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": True,
                "requested": "yes",
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "file:///tmp/pkg",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                    "dir_info": {},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/pkg.whl",
                    "archive_info": {"hashes": {"sha256": "short"}},
                },
                "is_direct": True,
                "metadata": {"name": "pkg", "version": "1"},
            }
        ],
        [
            {
                "download_info": {
                    "url": "https://example.invalid/a.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "is_direct": True,
                "metadata": {"name": "Same_Pkg", "version": "1"},
            },
            {
                "download_info": {
                    "url": "https://example.invalid/b.whl",
                    "archive_info": {"hashes": {"sha256": "b" * 64}},
                },
                "is_direct": True,
                "metadata": {"name": "same.pkg", "version": "2"},
            },
        ],
    ],
)
def test_malformed_or_colliding_pip_report_entries_fail_closed(tmp_path, install_entries):
    report = tmp_path / "pip-report.json"
    report.write_text(
        json.dumps({"version": "1", "install": install_entries}), encoding="utf-8"
    )
    step = InstallStep(
        phase="install",
        argv=["python", "-m", "pip", "install"],
        configured_index_urls=[],
    )
    with pytest.raises(EnvironmentManagerError, match="pip install evidence"):
        manager_module._install_evidence_from_report(
            report, step=step, command_id="command-001"
        )


def test_pip_report_schema_version_is_required(tmp_path):
    report = tmp_path / "pip-report.json"
    report.write_text(json.dumps({"version": "future", "install": []}), encoding="utf-8")
    step = InstallStep(phase="install", argv=["python", "-m", "pip", "install"])
    with pytest.raises(EnvironmentManagerError, match="install list"):
        manager_module._install_evidence_from_report(
            report, step=step, command_id="command-001"
        )


def test_pip_report_rejects_malformed_configured_index(tmp_path):
    report = tmp_path / "pip-report.json"
    report.write_text(json.dumps({"version": "1", "install": []}), encoding="utf-8")
    step = InstallStep(
        phase="install",
        argv=["python", "-m", "pip", "install"],
        configured_index_urls=["https://example.invalid:bad/simple"],
    )
    with pytest.raises(EnvironmentManagerError, match="malformed index URL"):
        manager_module._install_evidence_from_report(
            report, step=step, command_id="command-001"
        )


def test_index_source_classification_uses_exact_artifact_hostname(tmp_path):
    report = tmp_path / "pip-report.json"
    report.write_text(
        json.dumps(
            {
                "version": "1",
                "install": [
                    {
                        "download_info": {
                            "url": "https://not-pypi.org/pkg.whl",
                            "archive_info": {"hashes": {"sha256": "a" * 64}},
                        },
                        "is_direct": False,
                        "metadata": {"name": "pkg", "version": "1"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    step = InstallStep(
        phase="install",
        argv=["python", "-m", "pip", "install"],
        configured_index_urls=["https://pypi.org/simple"],
    )
    evidence = manager_module._install_evidence_from_report(
        report, step=step, command_id="command-001"
    )
    assert evidence[0].source == "unknown"
    assert evidence[0].source_index_url is None


def test_worker_wheel_archive_and_metadata_identities_fail_closed(tmp_path):
    with pytest.raises(EnvironmentManagerError, match="unavailable"):
        manager_module._worker_artifact_identity(tmp_path / "missing.whl")

    traversal = tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr(
            "corpus_studio_engine-1.3.0.dist-info/METADATA",
            "Name: corpus-studio-engine\nVersion: 1.3.0\n",
        )
        archive.writestr("../escape", "bad")
    with pytest.raises(EnvironmentManagerError, match="unsafe archive member"):
        manager_module._worker_artifact_identity(traversal)

    valid = _worker_wheel(tmp_path / "valid-wheel")
    mismatch = tmp_path / "corpus_studio_engine-9.9.9-py3-none-any.whl"
    shutil.copy2(valid, mismatch)
    with pytest.raises(EnvironmentManagerError, match="identit"):
        manager_module._worker_artifact_identity(mismatch)

    valid = _worker_wheel(tmp_path / "record-valid")
    tampered = tmp_path / "record-tampered" / valid.name
    tampered.parent.mkdir()
    with zipfile.ZipFile(valid) as source:
        members = {name: source.read(name) for name in source.namelist()}
    members["corpus_studio/readiness_marker.txt"] = b"bytes-not-matching-record"
    with zipfile.ZipFile(tampered, "w") as archive:
        for member_name, member_bytes in members.items():
            archive.writestr(member_name, member_bytes)
    with pytest.raises(EnvironmentManagerError, match="RECORD does not verify"):
        manager_module._worker_artifact_identity(tampered)


def test_worker_wheel_structural_ambiguities_fail_closed(tmp_path, monkeypatch):
    filename = "corpus_studio_engine-1.3.0-py3-none-any.whl"
    metadata_path = "corpus_studio_engine-1.3.0.dist-info/METADATA"
    record_path = "corpus_studio_engine-1.3.0.dist-info/RECORD"
    valid_metadata = b"Name: corpus-studio-engine\nVersion: 1.3.0\n"

    def record_for(members):
        rows = []
        for name, content in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
            rows.append(f"{name},sha256={digest},{len(content)}")
        rows.append(f"{record_path},,")
        return ("\n".join(rows) + "\n").encode()

    def write_wheel(name, members):
        path = tmp_path / name / filename
        path.parent.mkdir()
        with zipfile.ZipFile(path, "w") as archive:
            for member_name, member_bytes in members.items():
                archive.writestr(member_name, member_bytes)
        return path

    not_a_wheel = tmp_path / "not-a-wheel.txt"
    not_a_wheel.write_text("not a wheel", encoding="utf-8")
    with pytest.raises(EnvironmentManagerError, match="concrete wheel"):
        manager_module._worker_artifact_identity(not_a_wheel)

    unreadable = tmp_path / "unreadable" / filename
    unreadable.parent.mkdir()
    unreadable.write_bytes(b"not a zip archive")
    with pytest.raises(EnvironmentManagerError, match="worker wheel is unreadable"):
        manager_module._worker_artifact_identity(unreadable)

    bounded = _worker_wheel(tmp_path / "expanded-limit")
    with monkeypatch.context() as patch:
        patch.setattr(manager_module, "_MAX_WORKER_WHEEL_EXPANDED_BYTES", 1)
        with pytest.raises(EnvironmentManagerError, match="bounded expanded size"):
            manager_module._worker_artifact_identity(bounded)

    duplicate = tmp_path / "duplicate" / filename
    duplicate.parent.mkdir()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr(metadata_path, valid_metadata)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(metadata_path, valid_metadata)
    with pytest.raises(EnvironmentManagerError, match="duplicate archive members"):
        manager_module._worker_artifact_identity(duplicate)

    missing_metadata = write_wheel("missing-metadata", {record_path: b""})
    with pytest.raises(EnvironmentManagerError, match="exactly one.*METADATA"):
        manager_module._worker_artifact_identity(missing_metadata)

    nested_metadata = write_wheel(
        "nested-metadata",
        {"nested/corpus_studio_engine-1.3.0.dist-info/METADATA": valid_metadata},
    )
    with pytest.raises(EnvironmentManagerError, match="METADATA is not at wheel root"):
        manager_module._worker_artifact_identity(nested_metadata)

    identity_members = {metadata_path: valid_metadata, "other-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n"}
    identity_members[record_path] = record_for(identity_members)
    extra_identity = write_wheel("extra-identity", identity_members)
    with pytest.raises(EnvironmentManagerError, match="more than one dist-info identity"):
        manager_module._worker_artifact_identity(extra_identity)

    missing_record = write_wheel("missing-record", {metadata_path: valid_metadata})
    with pytest.raises(EnvironmentManagerError, match="exactly one.*RECORD"):
        manager_module._worker_artifact_identity(missing_record)

    nested_record_path = "nested/corpus_studio_engine-1.3.0.dist-info/RECORD"
    nested_record = write_wheel(
        "nested-record",
        {metadata_path: valid_metadata, nested_record_path: b""},
    )
    with pytest.raises(EnvironmentManagerError, match="RECORD does not match"):
        manager_module._worker_artifact_identity(nested_record)

    invalid_record_encoding = write_wheel(
        "invalid-record-encoding",
        {metadata_path: valid_metadata, record_path: b"\xff"},
    )
    with pytest.raises(EnvironmentManagerError, match="RECORD is malformed"):
        manager_module._worker_artifact_identity(invalid_record_encoding)

    metadata_digest = base64.urlsafe_b64encode(hashlib.sha256(valid_metadata).digest()).decode().rstrip("=")
    self_hashed_record = (
        f"{metadata_path},sha256={metadata_digest},{len(valid_metadata)}\n"
        f"{record_path},sha256={'a' * 43},1\n"
    ).encode()
    self_hashed = write_wheel(
        "self-hashed",
        {metadata_path: valid_metadata, record_path: self_hashed_record},
    )
    with pytest.raises(EnvironmentManagerError, match="self-entry must be unhashed"):
        manager_module._worker_artifact_identity(self_hashed)

    missing_member_record = (
        f"{metadata_path},sha256={metadata_digest},{len(valid_metadata)}\n"
        "ghost.py,sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,1\n"
        f"{record_path},,\n"
    ).encode()
    missing_member = write_wheel(
        "missing-member",
        {metadata_path: valid_metadata, record_path: missing_member_record},
    )
    with pytest.raises(EnvironmentManagerError, match="names a missing archive member"):
        manager_module._worker_artifact_identity(missing_member)

    malformed_digest_record = (
        f"{metadata_path},sha256=short,not-a-size\n"
        f"{record_path},,\n"
    ).encode()
    malformed_digest = write_wheel(
        "malformed-digest",
        {metadata_path: valid_metadata, record_path: malformed_digest_record},
    )
    with pytest.raises(EnvironmentManagerError, match="malformed digest or size"):
        manager_module._worker_artifact_identity(malformed_digest)

    malformed_record = write_wheel(
        "malformed-record",
        {metadata_path: valid_metadata, record_path: b"malformed,row\n"},
    )
    with pytest.raises(EnvironmentManagerError, match="RECORD contains malformed rows"):
        manager_module._worker_artifact_identity(malformed_record)

    unrecorded_members = {metadata_path: valid_metadata, "unrecorded.py": b"VALUE = 1\n"}
    unrecorded_members[record_path] = record_for({metadata_path: valid_metadata})
    unrecorded = write_wheel("unrecorded", unrecorded_members)
    with pytest.raises(EnvironmentManagerError, match="does not inventory every archive file"):
        manager_module._worker_artifact_identity(unrecorded)

    duplicate_name_metadata = (
        b"Name: corpus-studio-engine\nName: corpus-studio-engine\nVersion: 1.3.0\n"
    )
    duplicate_name_members = {metadata_path: duplicate_name_metadata}
    duplicate_name_members[record_path] = record_for(duplicate_name_members)
    duplicate_name = write_wheel("duplicate-name", duplicate_name_members)
    with pytest.raises(EnvironmentManagerError, match="exactly one Name and Version"):
        manager_module._worker_artifact_identity(duplicate_name)

    invalid_utf8_metadata = b"Name: corpus-studio-engine\nVersion: 1.3.0\n\xff"
    invalid_utf8_members = {metadata_path: invalid_utf8_metadata}
    invalid_utf8_members[record_path] = record_for(invalid_utf8_members)
    invalid_utf8 = write_wheel("invalid-metadata-encoding", invalid_utf8_members)
    with pytest.raises(EnvironmentManagerError, match="METADATA is invalid UTF-8"):
        manager_module._worker_artifact_identity(invalid_utf8)

    wrong_identity_metadata = b"Name: unrelated-package\nVersion: 1.3.0\n"
    wrong_identity_members = {metadata_path: wrong_identity_metadata}
    wrong_identity_members[record_path] = record_for(wrong_identity_members)
    wrong_identity = write_wheel("wrong-identity", wrong_identity_members)
    with pytest.raises(EnvironmentManagerError, match="corpus-studio-engine distribution"):
        manager_module._worker_artifact_identity(wrong_identity)


@pytest.mark.skipif(os.name == "nt", reason="worker artifact symlink test")
def test_worker_wheel_artifact_symlink_fails_closed(tmp_path):
    wheel = _worker_wheel(tmp_path / "artifact")
    link = tmp_path / wheel.name
    link.symlink_to(wheel)

    with pytest.raises(EnvironmentManagerError, match="cannot be a symbolic link"):
        manager_module._worker_artifact_identity(link)


@pytest.mark.parametrize("name", [None, "bad name", "bad\nname"])
def test_distribution_evidence_names_fail_closed(name):
    with pytest.raises(EnvironmentManagerError, match="invalid distribution name"):
        manager_module._validated_package_name(name)


def test_worker_wheel_declared_entry_points_are_the_only_generated_script_allowlist(
    tmp_path,
):
    wheel = _worker_wheel(
        tmp_path,
        entry_points=(
            "# reviewed entry points\n"
            "[console_scripts]\n"
            "corpus-studio = corpus_studio.cli:app\n"
            "\n"
            "[unrelated.group]\n"
            "ignored = corpus_studio.cli:ignored\n"
            "[gui_scripts]\n"
            "corpus-studio-gui = corpus_studio.desktop:main\n"
        ),
    )
    artifact = manager_module._worker_artifact_identity(wheel)

    console, gui = manager_module._worker_wheel_entry_point_scripts(artifact)

    assert console == {"corpus-studio"}
    assert gui == {"corpus-studio-gui"}


@pytest.mark.parametrize(
    "entry_points",
    [
        "[console_scripts]\nmissing-equals\n",
        "[console_scripts]\n!unsafe = corpus_studio.cli:app\n",
        (
            "[console_scripts]\nduplicate = corpus_studio.cli:app\n"
            "[gui_scripts]\nduplicate = corpus_studio.desktop:main\n"
        ),
    ],
)
def test_worker_wheel_malformed_entry_points_fail_closed(tmp_path, entry_points):
    wheel = _worker_wheel(tmp_path, entry_points=entry_points)
    artifact = manager_module._worker_artifact_identity(wheel)

    with pytest.raises(EnvironmentManagerError, match="entry_points.txt"):
        manager_module._worker_wheel_entry_point_scripts(artifact)


def test_lock_probe_record_paths_are_contained_and_symlinks_fail_closed():
    assert "RECORD path escapes the managed environment" in manager_module._LOCK_PROBE
    assert "RECORD path crosses a symbolic link" in manager_module._LOCK_PROBE
    assert 'algorithm != "sha256"' in manager_module._LOCK_PROBE
    assert "installed_files_hash" in manager_module._LOCK_PROBE


def test_legacy_environment_lock_digest_remains_compatible(tmp_path):
    package = manager_module.PackageLock(
        name="torch",
        version="2.7.1+cu128",
        hash=HashRef(value="a" * 64),
        installer="pip",
    )
    legacy = EnvironmentLock(
        lock_id="lock-legacy",
        recipe_ref=Ref(id="backend-corpus-studio", hash=HashRef(value="b" * 64)),
        manager_version="1.0.0",
        python_version="3.12.3",
        packages=[package],
    )
    body = legacy.model_dump(
        mode="json", exclude={"lock_id", "created_at", "lock_hash"}
    )
    for field_name in (
        "resolution_ref",
        "package_install_evidence",
        "worker_artifact",
        "probe_evidence",
    ):
        body.pop(field_name)
    for field_name in (
        "normalized_name",
        "source_index_url",
        "artifact_hash",
        "direct",
        "editable",
        "vcs_repository",
        "vcs_commit",
        "source_evidence_reason",
        "record_integrity",
        "record_entries",
        "record_verified_entries",
        "record_failed_entries",
        "installed_files_hash",
        "installed_file_count",
    ):
        body["packages"][0].pop(field_name)
    expected = manager_module._canonical_sha256(body)
    sealed = legacy.model_copy(update={"lock_hash": expected})
    manager = EnvironmentManager(tmp_path / "manager")
    assert manager._lock_digest(sealed) == expected


def test_manager_12_partial_count_lock_is_readable_but_health_refusal_is_non_mutating(
    tmp_path,
):
    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, _ = _create(tmp_path, fake, "legacy-count-env")
    current = manager.load_lock("legacy-count-env")
    legacy_packages = [
        item.model_copy(
            update={
                "record_count_semantics": None,
                "record_verified_entries": max(1, (item.record_entries or 1) - 1),
            }
        )
        for item in current.packages
    ]
    draft = current.model_copy(
        update={
            "manager_version": "1.2.0",
            "packages": legacy_packages,
            "lock_hash": "0" * 64,
        }
    )
    legacy = draft.model_copy(update={"lock_hash": manager._lock_digest(draft)})
    descriptor = manager.load_descriptor("legacy-count-env")
    assert descriptor.lock_ref is not None
    descriptor = descriptor.model_copy(
        update={
            "manager_version": "1.2.0",
            "lock_ref": descriptor.lock_ref.model_copy(
                update={"hash": HashRef(value=legacy.lock_hash)}
            ),
        }
    )
    manager._write_lock("legacy-count-env", legacy)
    manager._write_descriptor(descriptor)

    registry = manager._registry_dir("legacy-count-env")
    lock_path = registry / "locks" / f"{legacy.lock_id}.json"
    descriptor_path = registry / "EnvironmentDescriptor.json"
    health_path = registry / "EnvironmentHealthReport.json"
    before = {
        path: path.read_bytes() for path in (lock_path, descriptor_path, health_path)
    }
    calls_before = len(fake.calls)

    parsed = manager.load_lock("legacy-count-env")
    assert manager._lock_digest(parsed) == parsed.lock_hash
    assert all(not item.has_complete_record_count_evidence() for item in parsed.packages)
    report = manager.health("legacy-count-env")
    assert report.state == EnvironmentState.degraded
    assert report.failure is not None
    assert report.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert len(fake.calls) == calls_before
    assert {path: path.read_bytes() for path in before} == before

    tampered = legacy.model_copy(update={"lock_hash": "f" * 64})
    manager._write_lock("legacy-count-env", tampered)
    tampered_before = {
        path: path.read_bytes() for path in (lock_path, descriptor_path, health_path)
    }
    report = manager.health("legacy-count-env")
    assert report.state == EnvironmentState.broken
    assert report.lock_mismatch is True
    assert report.failure is not None
    assert report.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert len(fake.calls) == calls_before
    assert {path: path.read_bytes() for path in tampered_before} == tampered_before


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group escalation test")
def test_process_tree_termination_escalates_after_leader_exits(tmp_path):
    ready = tmp_path / "stubborn-child-ready"
    survived = tmp_path / "stubborn-child-survived"
    child_script = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).write_text('ready');"
        "time.sleep(0.8);"
        f"pathlib.Path({str(survived)!r}).write_text('survived')"
    )
    parent_script = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_script!r}]);"
        "time.sleep(120)"
    )
    process = subprocess.Popen(  # noqa: S603 - fixed test interpreter and literal script.
        [sys.executable, "-c", parent_script],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        terminate_process_tree(process, wait_timeout_seconds=0.15)
        time.sleep(0.85)
        assert process.poll() is not None
        assert not survived.exists(), (
            "a SIGTERM-ignoring descendant survived process-group escalation"
        )
    finally:
        terminate_process_tree(process, wait_timeout_seconds=0.15)


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


def _resolved_run_plan(environment_ref: Ref) -> RunPlan:
    from corpus_studio.platform.execution_config import execution_configuration_hash_for
    from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload
    from corpus_studio.platform.runners import demo_training_plan

    plan = demo_training_plan("managed-plan")
    assert plan.resolved_execution is not None
    execution = plan.resolved_execution.model_copy(
        update={
            "configuration_hash": "0" * 64,
            "environment_ref": environment_ref,
            "environment_binding": "managed_lock",
        }
    )
    execution = execution.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(execution)}
    )
    body = plan.model_dump(mode="json")
    body["plan_hash"] = "0" * 64
    body["environment_ref"] = environment_ref.model_dump(mode="json")
    body["resolved_execution"] = execution.model_dump(mode="json")
    draft = RunPlan.model_validate(body)
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
    assert recreated.exit_code == 2
    assert "sealed environment" in recreated.output
    assert "new environment id" in recreated.output

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
    plan = _resolved_run_plan(locked_environment_ref(result.descriptor, result.lock))
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
    from corpus_studio.platform.supervisor import EchoRunner, demo_run_plan, execute_run

    def fake_subprocess(run_plan, **kwargs):
        captured["worker_argv"] = kwargs["worker_argv"]
        lease_probe = "\n".join(
            [
                "import pathlib, sys",
                "from corpus_studio.platform.environment_manager import (",
                "    EnvironmentManager, EnvironmentManagerError",
                ")",
                "manager = EnvironmentManager(pathlib.Path(sys.argv[1]), lock_timeout_seconds=0.05)",
                "try:",
                "    with manager.environment_lease(sys.argv[2], operation='competing run'):",
                "        raise SystemExit(0)",
                "except EnvironmentManagerError as exc:",
                "    print(exc.failure.taxonomy.value)",
                "    raise SystemExit(7)",
            ]
        )
        probe = subprocess.run(  # noqa: S603 - fixed test interpreter and local literal script.
            [sys.executable, "-c", lease_probe, str(manager.root), "run-env"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        captured["lease_probe"] = probe
        return execute_run(demo_run_plan(), EchoRunner())

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
    assert captured["lease_probe"].returncode == 7
    assert captured["lease_probe"].stdout.strip() == FailureTaxonomy.TIMEOUT.value

    shutil.rmtree(result.descriptor.root_path)
    missing = cli.invoke(
        app, ["platform-run", str(plan_path), "--subprocess"]
    )
    assert missing.exit_code == 2
    assert "not functionally verified" in missing.output


def test_platform_plan_profiles_the_managed_interpreter_and_pins_its_lock(
    tmp_path, monkeypatch
):
    from corpus_studio.platform.backends import get_backend
    from corpus_studio.platform.contracts import (
        EffectiveCapabilities,
        ExecutionCapabilityCombination,
        ProbeResult,
    )

    fake = FakeEnvironmentRunner(cuda=True)
    manager, _, result = _create(tmp_path, fake, "plan-env")
    original_capability_snapshot = manager.capability_snapshot
    first_party = get_backend("corpus_studio")
    assert first_party is not None

    def sealed_capability_snapshot(env_id):
        profile, report = original_capability_snapshot(env_id)
        combination = ExecutionCapabilityCombination.model_validate(
            {
                "runtime_mode": "training",
                "device": "cuda",
                "precision": "bf16",
                "quantization": "nf4",
                "adapter_method": "qlora",
                "attention_impl": "math",
                "attention_kernel": "torch_sdpa_math",
                "optimizer": "adamw_torch",
                "loss_impl": "cross_entropy",
                "checkpoint_impl": "adapter_only",
                "export_format": "adapter_peft",
                "execution_contract_version": "1.0.0",
                "probe": "synthetic_execution",
            }
        )
        probe_results = [
            ProbeResult(
                probe="synthetic_axes",
                outcome="PASS",
                proves={
                    "adapter": ["qlora"],
                    "attention": ["math"],
                    "attention_kernel": ["torch_sdpa_math"],
                    "checkpoint": ["adapter_only"],
                    "loss": ["cross_entropy", "liger_fused_ce"],
                    "optimizer": ["adamw_torch", "paged_adamw_8bit"],
                    "precision": ["bf16"],
                },
            ),
            ProbeResult(
                probe="bnb_4bit_load", outcome="PASS", proves={"quantization": ["nf4"]}
            ),
            ProbeResult(
                probe="trainer_contract",
                outcome="PASS",
                proves={
                    "trainer_field": first_party.trainer_fields,
                    "trainer_init_field": first_party.trainer_init_fields,
                },
            ),
            ProbeResult(
                probe="synthetic_execution",
                outcome="PASS",
                execution_combinations=[combination],
            ),
        ]
        effective = EffectiveCapabilities(
            precision_modes=["bf16"],
            quantization_modes=["nf4"],
            attention_impls=["math"],
            attention_kernels=["torch_sdpa_math"],
            adapter_methods=["qlora"],
            optimizers=["adamw_torch", "paged_adamw_8bit"],
            loss_impls=["cross_entropy", "liger_fused_ce"],
            checkpoint_impls=["adapter_only"],
            execution_contract_versions=["1.0.0"],
            execution_combinations=[combination],
            trainer_fields=first_party.trainer_fields,
            trainer_init_fields=first_party.trainer_init_fields,
        )
        return profile, report.model_copy(
            update={
                "installed_packages": result.lock.packages,
                "backend_version": first_party.backend_version,
                "probe_results": probe_results,
                "effective_capabilities": effective,
            }
        )

    monkeypatch.setattr(manager, "capability_snapshot", sealed_capability_snapshot)
    monkeypatch.setattr(
        manager_module, "EnvironmentManager", lambda root=None: manager
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Say hello.", "output": "Hello."}) + "\n",
        encoding="utf-8",
    )
    cli = CliRunner()
    planned = cli.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "model",
            "--model-revision",
            "1" * 40,
            "--dataset",
            str(dataset),
            "--environment",
            "plan-env",
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.stdout)["run_plan"]
    assert plan["environment_ref"]["id"] == "plan-env"
    assert plan["environment_ref"]["hash"]["value"] == result.lock.lock_hash
    resolved = plan["resolved_execution"]
    assert resolved["environment_ref"] == plan["environment_ref"]
    assert resolved["trainer_interface"]["package_versions"]
    capability_call = next(
        call for call in reversed(fake.calls) if call["phase"] == "capability_probe"
    )
    assert capability_call["argv"][0] == result.descriptor.python_executable
