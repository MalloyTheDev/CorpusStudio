"""Side-effectful lifecycle manager for isolated backend environments.

The manager executes only a previously sealed :class:`DependencyResolution`. A caller must echo the
resolution hash before any directory is created, which binds mutation to the exact reviewed argv,
runtime, package index, and target path. Heavy framework imports happen only in the managed Python
subprocess; importing this module remains dependency-light.

The first supported creation target is ``backend-corpus-studio``. Recipes for other backends can be
previewed, but they do not become "supported" merely because the resolver can render commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Literal, Protocol
from urllib.parse import unquote, urlparse
import uuid

from .common import HashRef, PackageLock, PackageSource, Ref
from .contracts import (
    CapabilityReport,
    DependencyResolution,
    EnvironmentCommandRecord,
    EnvironmentDescriptor,
    EnvironmentHealthReport,
    EnvironmentInstallation,
    EnvironmentLock,
    EnvironmentProfile,
    EnvironmentRecipe,
    FailureRecord,
    InstallStep,
    ProbeResult,
    PythonRuntime,
    RunPlan,
)
from .enums import DependencyLayer, EnvironmentState, FailureTaxonomy, OperatingSystem
from .environments import (
    get_recipe,
    recipe_digest,
    resolution_digest,
    resolve_dependencies,
)

MANAGER_VERSION = "1.0.0"
REFERENCE_RECIPE_ID = "backend-corpus-studio"
SUPPORTED_CREATION_RECIPES = frozenset({REFERENCE_RECIPE_ID})

_OWNER_FILENAME = ".corpusstudio-owner.json"
_OWNER_KIND = "corpus-studio-managed-environment-v1"
_DESCRIPTOR_FILENAME = "EnvironmentDescriptor.json"
_HEALTH_FILENAME = "EnvironmentHealthReport.json"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_PROBE_TAIL_LIMIT = 32_000
_LOCK_OUTPUT_LIMIT = 16_000_000
_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)

EnvironmentCommandPhase = Literal[
    "create_venv",
    "upgrade_pip",
    "install",
    "verify",
    "lock",
    "import_probe",
    "dependency_probe",
    "functional_probe",
    "hardware_probe",
    "health_probe",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(now: Callable[[], datetime]) -> str:
    return now().astimezone(timezone.utc).isoformat()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _command_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """A small, non-secret, explicit subprocess environment suitable for Python/pip."""
    selected = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENVIRONMENT_ALLOWLIST
    }
    selected.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    selected.update(extra or {})
    return dict(sorted(selected.items(), key=lambda item: item[0].casefold()))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    # Windows indexers/AV can briefly retain a just-written JSON handle. Keep the transition atomic,
    # but tolerate that short sharing violation rather than leaving the durable state stale.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def _atomic_model(path: Path, model: object) -> None:
    payload = model.model_dump_json(indent=2)  # type: ignore[attr-defined]
    _atomic_write(path, payload + "\n")


def default_manager_root() -> Path:
    """User-owned manager state; on Windows this is the internal system drive by default."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "CorpusStudio" / "environment-manager"
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "corpusstudio" / "environment-manager"


def _operating_system(system: str, release: str = "") -> OperatingSystem:
    if system.lower() == "windows":
        return OperatingSystem.windows
    if system.lower() == "darwin":
        return OperatingSystem.macos
    if system.lower() == "linux":
        if "microsoft" in release.lower() or os.environ.get("WSL_DISTRO_NAME"):
            return OperatingSystem.wsl
        return OperatingSystem.linux
    return OperatingSystem.unknown


_RUNTIME_PROBE = r"""
import importlib.util, json, os, platform, struct, sys
system = platform.system()
release = platform.release()
if system.lower() == "windows": os_name = "windows"
elif system.lower() == "darwin": os_name = "macos"
elif system.lower() == "linux" and ("microsoft" in release.lower() or os.environ.get("WSL_DISTRO_NAME")): os_name = "wsl"
elif system.lower() == "linux": os_name = "linux"
else: os_name = "unknown"
print(json.dumps({
  "executable": sys.executable,
  "version": platform.python_version(),
  "implementation": platform.python_implementation(),
  "architecture": f"{struct.calcsize('P') * 8}-bit",
  "platform": platform.platform(),
  "os": os_name,
  "is_virtual_environment": sys.prefix != sys.base_prefix,
  "venv_available": importlib.util.find_spec("venv") is not None,
}, sort_keys=True))
""".strip()


def _minimum_python(specifier: str) -> tuple[int, int] | None:
    match = re.match(r"^>=\s*(\d+)\.(\d+)", specifier.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def _version_pair(value: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", value.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def _runtime_from_payload(payload: Mapping[str, object], python_requires: str) -> PythonRuntime:
    executable = str(payload.get("executable") or "")
    version = str(payload.get("version") or "")
    reasons: list[str] = []
    floor = _minimum_python(python_requires)
    current = _version_pair(version)
    if floor is not None and (current is None or current < floor):
        reasons.append(f"Python {version or '?'} does not satisfy {python_requires}")
    if not bool(payload.get("venv_available")):
        reasons.append("the stdlib venv module is unavailable")
    identity = _canonical_sha256(
        {"executable": executable.casefold(), "version": version}
    )[:20]
    raw_os = str(payload.get("os") or OperatingSystem.unknown.value)
    try:
        os_value = OperatingSystem(raw_os)
    except ValueError:
        os_value = OperatingSystem.unknown
    return PythonRuntime(
        runtime_id=f"python-{identity}",
        executable=executable,
        version=version,
        implementation=str(payload.get("implementation") or ""),
        architecture=str(payload.get("architecture") or ""),
        platform=str(payload.get("platform") or ""),
        os=os_value,
        is_virtual_environment=bool(payload.get("is_virtual_environment")),
        venv_available=bool(payload.get("venv_available")),
        compatible=not reasons,
        incompatibility_reasons=reasons,
    )


def probe_python_runtime(
    executable: str | Path, *, python_requires: str = ">=3.11", timeout_seconds: int = 10
) -> PythonRuntime:
    """Inspect one interpreter with a bounded argv command; no environment is created."""
    completed = subprocess.run(  # noqa: S603 - explicit interpreter selected by the user/discovery
        [str(executable), "-c", _RUNTIME_PROBE],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        shell=False,
        env=_command_environment(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"runtime probe failed for {executable}: {detail or completed.returncode}")
    payload = _last_json_object(completed.stdout)
    return _runtime_from_payload(payload, python_requires)


RuntimeProbe = Callable[[str | Path, str], PythonRuntime]


def _default_runtime_probe(executable: str | Path, python_requires: str) -> PythonRuntime:
    return probe_python_runtime(executable, python_requires=python_requires)


def _launcher_candidates() -> list[str]:
    launcher = shutil.which("py")
    if os.name != "nt" or launcher is None:
        return []
    try:
        completed = subprocess.run(  # noqa: S603 - resolved Windows launcher, no shell
            [launcher, "-0p"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            shell=False,
            env=_command_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    candidates: list[str] = []
    for line in completed.stdout.splitlines():
        match = re.search(r"([A-Za-z]:\\.*?python(?:\.exe)?)\s*$", line, re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())
    return candidates


def discover_python_runtimes(
    *,
    candidates: Sequence[str | Path] | None = None,
    python_requires: str = ">=3.11",
    probe: RuntimeProbe | None = None,
) -> list[PythonRuntime]:
    """Discover and compatibility-check all reachable Python installations.

    Broken candidates are ignored rather than making discovery fail. Tests inject candidates and a
    probe, so default CI never depends on the host's launcher layout.
    """
    if candidates is None:
        found: list[str | Path] = [sys.executable]
        found.extend(_launcher_candidates())
        for command in ("python3", "python"):
            path = shutil.which(command)
            if path:
                found.append(path)
        candidates = found
    probe_fn = probe or _default_runtime_probe
    runtimes: list[PythonRuntime] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        try:
            runtime = probe_fn(candidate, python_requires)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
            continue
        runtimes.append(runtime)
    return sorted(runtimes, key=lambda item: (not item.compatible, item.version, item.executable))


@dataclass(frozen=True)
class CommandOutcome:
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False


class CancellationToken(Protocol):
    def is_set(self) -> bool: ...


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdout_path: Path,
        stderr_path: Path,
        cancel: CancellationToken | None,
    ) -> CommandOutcome: ...


class SubprocessCommandRunner:
    """Bounded, cancellable, no-shell subprocess execution with file-backed output."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdout_path: Path,
        stderr_path: Path,
        cancel: CancellationToken | None,
    ) -> CommandOutcome:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        process_environment = dict(environment)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, \
                stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            process = subprocess.Popen(  # noqa: S603 - reviewed argv, shell is explicitly disabled
                list(argv),
                cwd=str(cwd),
                env=process_environment,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                shell=False,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            started = time.monotonic()
            timed_out = False
            cancelled = False
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        _terminate(process)
                        break
                    if time.monotonic() - started >= timeout_seconds:
                        timed_out = True
                        _terminate(process)
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                cancelled = True
                _terminate(process)
            if process.poll() is None:
                _terminate(process)
            return CommandOutcome(
                exit_code=process.returncode, timed_out=timed_out, cancelled=cancelled
            )


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Pip/build backends may spawn compiler children. Kill the new process group as a tree so a
        # timeout never leaves an installer writing into the managed environment in the background.
        system_root = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
        taskkill = system_root / "System32" / "taskkill.exe"
        try:
            subprocess.run(  # noqa: S603 - fixed OS utility and integer pid, no shell
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    else:
        kill_process_group = getattr(os, "killpg")
        try:
            kill_process_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            kill_process_group = getattr(os, "killpg")
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            try:
                kill_process_group(process.pid, kill_signal)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - OS failed to reap a killed process
            pass


class EnvironmentManagerError(RuntimeError):
    def __init__(
        self,
        message: str,
        failure: FailureRecord | None = None,
        installation: EnvironmentInstallation | None = None,
    ):
        super().__init__(message)
        self.failure = failure or FailureRecord(
            taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
            message=message,
        )
        self.installation = installation


@dataclass(frozen=True)
class EnvironmentCreationResult:
    descriptor: EnvironmentDescriptor
    lock: EnvironmentLock
    health: EnvironmentHealthReport
    installation: EnvironmentInstallation


def _read_tail(path: Path, limit: int = _PROBE_TAIL_LIMIT) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit), os.SEEK_SET)
            data = stream.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise json.JSONDecodeError("no JSON object found in command output", text, 0)


def _native_build_evidence(stderr: str, stdout: str) -> bool:
    combined = (stderr + "\n" + stdout).casefold()
    markers = (
        "building wheel",
        "running build_ext",
        "microsoft visual c++",
        "gcc ",
        "clang ",
        "cmake",
        "ninja",
    )
    return any(marker in combined for marker in markers)


_LOCK_PROBE = r"""
import hashlib, importlib.metadata as metadata, json, os, platform, struct, sys
packages = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or getattr(dist, "name", "")
    if not name: continue
    direct_text = dist.read_text("direct_url.json")
    try: direct = json.loads(direct_text) if direct_text else None
    except Exception: direct = None
    record = dist.read_text("RECORD")
    installer = dist.read_text("INSTALLER")
    packages.append({
      "name": name,
      "version": dist.version,
      "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None,
      "direct_url": direct,
      "installer": installer.strip() if installer else None,
      "requested": dist.read_text("REQUESTED") is not None,
      "dependencies": sorted(dist.requires or []),
    })
torch_data = {"version": None, "build": None, "cuda": None, "compute_capability": None}
try:
    import torch
    torch_data["version"] = str(torch.__version__)
    torch_data["build"] = str(getattr(torch.version, "git_version", None) or torch.__version__)
    torch_data["cuda"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        torch_data["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
except Exception as exc:
    torch_data["error"] = str(exc)
system = platform.system(); release = platform.release()
if system.lower() == "windows": os_name = "windows"
elif system.lower() == "darwin": os_name = "macos"
elif system.lower() == "linux" and ("microsoft" in release.lower() or os.environ.get("WSL_DISTRO_NAME")): os_name = "wsl"
elif system.lower() == "linux": os_name = "linux"
else: os_name = "unknown"
print(json.dumps({
  "runtime": {
    "executable": sys.executable,
    "version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "architecture": f"{struct.calcsize('P') * 8}-bit",
    "platform": platform.platform(),
    "os": os_name,
    "is_virtual_environment": sys.prefix != sys.base_prefix,
    "venv_available": True,
  },
  "packages": sorted(packages, key=lambda p: p["name"].lower()),
  "torch": torch_data,
}, sort_keys=True))
""".strip()


_IMPORT_PROBE = r"""
import importlib, json
modules = [
  "corpus_studio.platform.worker", "torch", "transformers", "peft", "trl", "accelerate",
  "datasets", "bitsandbytes"
]
results = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        results[name] = {"ok": True, "version": str(getattr(module, "__version__", ""))}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps({"results": results}, sort_keys=True))
""".strip()


_FUNCTIONAL_PROBE = r"""
import json, tempfile
result = {"minimal_forward": False, "minimal_backward": False, "checkpoint_reload": False}
try:
    import torch
    layer = torch.nn.Linear(8, 4)
    value = layer(torch.randn(2, 8)).sum()
    result["minimal_forward"] = bool(torch.isfinite(value).item())
    value.backward()
    result["minimal_backward"] = layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all().item())
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/probe.pt"
        torch.save({"weight": layer.weight.detach()}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)["weight"]
        result["checkpoint_reload"] = bool(torch.equal(layer.weight.detach(), loaded))
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
result["ok"] = bool(result["minimal_forward"] and result["minimal_backward"] and result["checkpoint_reload"])
print(json.dumps(result, sort_keys=True))
""".strip()


_HARDWARE_PROBE = r"""
import json, platform
result = {
  "cuda_available": False, "cuda_allocation": False, "compute_capability": None,
  "bf16_supported": False, "four_bit_construction": False, "minimal_forward": False,
  "minimal_backward": False, "attention_backend": None, "optional_kernels": {}, "ok": False,
}
try:
    import torch
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["optional_kernels"] = {
      "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
      "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
      "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    if result["cuda_available"]:
        capability = torch.cuda.get_device_capability(0)
        result["compute_capability"] = ".".join(map(str, capability))
        result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        tensor = torch.ones(8, device="cuda")
        result["cuda_allocation"] = bool(tensor.sum().item() == 8)
        layer = torch.nn.Linear(8, 4, device="cuda")
        value = layer(torch.randn(2, 8, device="cuda")).sum()
        result["minimal_forward"] = bool(torch.isfinite(value).item())
        value.backward()
        result["minimal_backward"] = layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all().item())
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch.nn.functional as functional
        q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        # Math is mandatory on native-Windows Blackwell and safe everywhere; never execute the known
        # fused sm_120 WDDM deadlock as a default environment probe.
        with sdpa_kernel([SDPBackend.MATH]):
            attention = functional.scaled_dot_product_attention(q, k, v)
        attention.sum().backward()
        result["attention_backend"] = "math"
        try:
            from bitsandbytes.nn import Linear4bit
            quantized = Linear4bit(8, 4, bias=False, compute_dtype=torch.float16).cuda()
            quantized(torch.randn(2, 8, device="cuda", dtype=torch.float16))
            result["four_bit_construction"] = True
        except Exception as exc:
            result["four_bit_error"] = f"{type(exc).__name__}: {exc}"
        result["ok"] = all(result[key] for key in (
          "cuda_allocation", "minimal_forward", "minimal_backward", "four_bit_construction"
        )) and result["attention_backend"] == "math"
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
""".strip()


_CAPABILITY_SNAPSHOT = r"""
import contextlib, json, sys
from corpus_studio.platform.profiler import build_environment_profile
from corpus_studio.platform.probes import run_capability_probes
with contextlib.redirect_stdout(sys.stderr):
    profile = build_environment_profile()
    report = run_capability_probes(profile)
print(json.dumps({
  "profile": profile.model_dump(mode="json"),
  "capability_report": report.model_dump(mode="json"),
}, sort_keys=True))
""".strip()


def _package_source(direct: object) -> tuple[PackageSource, str | None, str | None]:
    if not isinstance(direct, dict):
        # INSTALLER=pip does not prove which configured index supplied the artifact. Keep the source
        # unknown unless PEP 610 gives us direct evidence; the lock separately records reviewed URLs.
        return ("unknown", None, None)
    url = str(direct.get("url") or "") or None
    source: PackageSource = "unknown"
    if "vcs_info" in direct:
        source = "vcs"
    elif "dir_info" in direct:
        source = "local"
    elif "archive_info" in direct:
        source = "wheel" if url and url.lower().endswith(".whl") else "sdist"
    artifact = None
    if url:
        artifact = Path(unquote(urlparse(url).path)).name or None
    return source, url, artifact


class EnvironmentManager:
    """Own isolated worker environments and their durable registry/evidence."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        runner: CommandRunner | None = None,
        runtime_probe: RuntimeProbe | None = None,
        now: Callable[[], datetime] = _utcnow,
        engine_source: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_manager_root()
        self.environments_root = self.root / "environments"
        self.registry_root = self.root / "registry"
        self.runner = runner or SubprocessCommandRunner()
        self.runtime_probe = runtime_probe or _default_runtime_probe
        self.now = now
        self.engine_source = (
            Path(engine_source)
            if engine_source is not None
            else Path(__file__).resolve().parents[2]
        )

    def environment_root(self, env_id: str) -> Path:
        self._validate_env_id(env_id)
        root = (self.environments_root / env_id).resolve(strict=False)
        parent = self.environments_root.resolve(strict=False)
        if root.parent != parent or root == parent:
            raise EnvironmentManagerError("environment path escapes the managed root")
        return root

    def preview(
        self,
        recipe_id: str,
        *,
        env_id: str,
        runtime_executable: str | Path,
        accelerator_tag: str = "cpu",
    ) -> DependencyResolution:
        recipe = get_recipe(recipe_id)
        if recipe is None:
            raise EnvironmentManagerError(f"unknown environment recipe '{recipe_id}'")
        runtime = self.runtime_probe(runtime_executable, recipe.python_requires)
        resolution = resolve_dependencies(
            recipe,
            os_value=runtime.os,
            accelerator_tag=accelerator_tag,
            python_version=runtime.version,
            runtime=runtime,
            environment_id=env_id,
            environment_root=str(self.environment_root(env_id)),
            manager_version=MANAGER_VERSION,
        )
        if recipe_id == REFERENCE_RECIPE_ID and recipe.layer == DependencyLayer.backend_worker:
            environment_python = self._environment_python(
                self.environment_root(env_id), runtime.os
            )
            worker_step = InstallStep(
                phase="install",
                description="Install the CorpusStudio worker package from the reviewed local source",
                argv=[
                    environment_python,
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--no-deps",
                    str(self.engine_source.resolve(strict=False)),
                ],
                working_directory=str(self.environment_root(env_id)),
                environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                timeout_seconds=900,
                network_required=True,
            )
            resolution = resolution.model_copy(
                update={
                    "install_steps": resolution.install_steps + [worker_step],
                    "resolution_hash": None,
                }
            )
            if not (self.engine_source / "pyproject.toml").is_file():
                resolution = resolution.model_copy(
                    update={
                        "resolvable": False,
                        "blocking_reasons": resolution.blocking_reasons
                        + [
                            "CorpusStudio worker source is missing pyproject.toml at "
                            f"{self.engine_source}"
                        ],
                    }
                )
        concrete = resolution.model_copy(
            update={
                "install_steps": [
                    step.model_copy(
                        update={"environment": _command_environment(step.environment)}
                    )
                    for step in resolution.install_steps
                ],
                "resolution_hash": None,
            }
        )
        resolution = concrete.model_copy(
            update={"resolution_hash": resolution_digest(concrete)}
        )
        if runtime.compatible:
            return resolution
        blocked = resolution.model_copy(
            update={
                "resolvable": False,
                "blocking_reasons": resolution.blocking_reasons
                + runtime.incompatibility_reasons,
                "resolution_hash": None,
            }
        )
        return blocked.model_copy(update={"resolution_hash": resolution_digest(blocked)})

    def create(
        self,
        resolution: DependencyResolution,
        *,
        confirmed_resolution_hash: str,
        cancel: CancellationToken | None = None,
    ) -> EnvironmentCreationResult:
        """Execute one sealed reference-backend plan and persist every transition/evidence record."""
        recipe, env_id, env_root = self._validate_creation(
            resolution, confirmed_resolution_hash
        )
        resolution_hash = resolution.resolution_hash
        assert resolution_hash is not None  # narrowed by _validate_creation
        if env_root.exists():
            raise EnvironmentManagerError(
                f"environment '{env_id}' already exists; use env-recreate after inspecting it"
            )

        created_at = _timestamp(self.now)
        installation_id = f"install-{uuid.uuid4().hex[:20]}"
        resolution_ref = Ref(
            id=f"resolution-{resolution_hash[:20]}",
            hash=HashRef(value=resolution_hash),
        )
        descriptor = EnvironmentDescriptor(
            env_id=env_id,
            recipe_ref=resolution.recipe_ref,
            layer=recipe.layer,
            root_path=str(env_root),
            python_executable=self._environment_python(env_root, resolution.os),
            state=EnvironmentState.installing,
            resolution_ref=resolution_ref,
            installation_ref=Ref(id=installation_id),
            manager_version=MANAGER_VERSION,
            created_at=created_at,
            updated_at=created_at,
        )
        installation = EnvironmentInstallation(
            installation_id=installation_id,
            environment_ref=Ref(id=env_id),
            recipe_ref=resolution.recipe_ref,
            resolution_ref=resolution_ref,
            started_at=created_at,
        )

        # Persist intent before touching the environment root. A host crash between transitions then
        # leaves a discoverable INSTALLING attempt instead of an unowned mystery directory.
        self._write_descriptor(descriptor)
        self._write_installation(env_id, installation)
        self._write_health(
            env_id,
            EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                state=EnvironmentState.installing,
                checked_at=created_at,
            ),
        )

        try:
            env_root.mkdir(parents=True, exist_ok=False)
            self._write_owner(env_root, env_id, created_at)
            for step in resolution.install_steps:
                installation = self._execute_step(
                    env_id, installation, step, cancel=cancel
                )
            descriptor = self._update_descriptor(
                descriptor, state=EnvironmentState.installed_unchecked
            )

            lock, installation = self._capture_lock(
                descriptor, resolution, installation, cancel=cancel
            )
            descriptor = self._update_descriptor(
                descriptor,
                lock_ref=Ref(id=lock.lock_id, hash=HashRef(value=lock.lock_hash)),
            )
            probe_results, final_state, installation = self._run_creation_probes(
                descriptor, installation, cancel=cancel
            )
            descriptor = self._update_descriptor(descriptor, state=final_state)
            finished = _timestamp(self.now)
            installation = installation.model_copy(
                update={"state": final_state, "finished_at": finished}
            )
            self._write_installation(env_id, installation)
            health = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=final_state,
                python_version=lock.python_version,
                checked_at=finished,
                installed_packages=lock.packages,
                probe_results=probe_results,
                remediation=self._probe_remediation(final_state),
            )
            self._write_health(env_id, health)
            return EnvironmentCreationResult(descriptor, lock, health, installation)
        except Exception as exc:
            if isinstance(exc, EnvironmentManagerError) and exc.installation is not None:
                installation = exc.installation
            failure = self._failure_from_exception(exc)
            failed_at = _timestamp(self.now)
            installation = installation.model_copy(
                update={
                    "state": EnvironmentState.broken,
                    "finished_at": failed_at,
                    "failure": failure,
                    "retry_requires_recreate": True,
                }
            )
            self._write_installation(env_id, installation)
            descriptor = self._update_descriptor(
                descriptor,
                state=EnvironmentState.broken,
                notes=descriptor.notes
                + ["Creation failed; inspect the installation journal, then explicitly recreate."],
            )
            health = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=failed_at,
                failure=failure,
                remediation="Inspect command stderr, then use env-recreate with a newly reviewed plan.",
            )
            self._write_health(env_id, health)
            if isinstance(exc, EnvironmentManagerError):
                raise
            raise EnvironmentManagerError(failure.message, failure) from exc

    def health(
        self, env_id: str, *, cancel: CancellationToken | None = None
    ) -> EnvironmentHealthReport:
        """Re-probe imports/functionality and compare the live package set to the sealed lock."""
        descriptor = self.load_descriptor(env_id)
        env_root = Path(descriptor.root_path)
        checked_at = _timestamp(self.now)
        if not env_root.exists():
            deliberately_removed = descriptor.state == EnvironmentState.not_installed
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.not_installed
                if deliberately_removed
                else EnvironmentState.broken,
                checked_at=checked_at,
                environment_missing=not deliberately_removed,
                remediation="Recreate this managed environment from a reviewed plan."
                if not deliberately_removed
                else None,
            )
            self._write_health(env_id, report)
            if not deliberately_removed:
                self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report
        try:
            self._assert_owned(env_root, env_id)
        except EnvironmentManagerError as exc:
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=checked_at,
                failure=exc.failure,
                remediation=(
                    "Restore the ownership marker only from trusted registry evidence, or remove "
                    "the path manually after review."
                ),
            )
            self._write_health(env_id, report)
            self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report
        python_path = Path(descriptor.python_executable)
        if not python_path.is_file():
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=checked_at,
                interpreter_missing=True,
                failure=FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="managed environment interpreter is missing",
                ),
                remediation="Recreate the managed environment; do not repair arbitrary files in place.",
            )
            self._write_health(env_id, report)
            self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report

        try:
            lock = self.load_lock(env_id)
        except EnvironmentManagerError as exc:
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=checked_at,
                lock_mismatch=True,
                failure=exc.failure,
                remediation="Recreate from a reviewed plan; the recorded lock is unavailable.",
            )
            self._write_health(env_id, report)
            self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report
        try:
            live_lock = self._inspect_live_lock(
                descriptor, lock.recipe_ref, lock.index_urls, cancel
            )
        except EnvironmentManagerError as exc:
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=checked_at,
                failure=exc.failure,
                remediation="Inspect health-probe logs, then recreate if the interpreter is broken.",
            )
            self._write_health(env_id, report)
            self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report
        drifted, changed_sources = self._package_drift(lock.packages, live_lock.packages)
        recipe = get_recipe(descriptor.recipe_ref.id)
        recipe_drift = (
            recipe is None
            or descriptor.recipe_ref.hash is None
            or descriptor.recipe_ref.hash.value != recipe_digest(recipe)
        )
        descriptor_lock_hash = (
            descriptor.lock_ref.hash.value
            if descriptor.lock_ref is not None and descriptor.lock_ref.hash is not None
            else None
        )
        lock_mismatch = (
            lock.lock_hash != self._lock_digest(lock)
            or descriptor_lock_hash != lock.lock_hash
        )
        cuda_mismatch = lock.cuda_runtime_version != live_lock.cuda_runtime_version
        hardware_mismatch = lock.compute_capability != live_lock.compute_capability

        probe_results, probe_state = self._health_probes(descriptor, cancel=cancel)
        drift = bool(
            drifted
            or changed_sources
            or recipe_drift
            or lock_mismatch
            or cuda_mismatch
            or hardware_mismatch
        )
        state = EnvironmentState.drifted if drift else probe_state
        report = EnvironmentHealthReport(
            environment_ref=Ref(id=env_id),
            recipe_ref=descriptor.recipe_ref,
            lock_ref=descriptor.lock_ref,
            state=state,
            python_version=live_lock.python_version,
            checked_at=checked_at,
            installed_packages=live_lock.packages,
            drifted_packages=drifted,
            changed_package_sources=changed_sources,
            drift_detected=drift,
            recipe_drift_detected=recipe_drift,
            lock_mismatch=lock_mismatch,
            hardware_mismatch=hardware_mismatch,
            cuda_mismatch=cuda_mismatch,
            probe_results=probe_results,
            remediation="Recreate from a newly reviewed plan; the live environment no longer matches its lock."
            if drift
            else self._probe_remediation(probe_state),
        )
        self._write_health(env_id, report)
        if descriptor.state != state:
            self._update_descriptor(descriptor, state=state)
        return report

    def list_descriptors(self) -> list[EnvironmentDescriptor]:
        if not self.registry_root.is_dir():
            return []
        descriptors: list[EnvironmentDescriptor] = []
        for path in sorted(self.registry_root.glob(f"*/{_DESCRIPTOR_FILENAME}")):
            try:
                descriptors.append(
                    EnvironmentDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
        return descriptors

    def capability_snapshot(
        self, env_id: str, *, cancel: CancellationToken | None = None
    ) -> tuple[EnvironmentProfile, CapabilityReport]:
        """Profile and prove capabilities inside the managed interpreter, never the control plane."""
        health = self.health(env_id, cancel=cancel)
        if health.state not in {
            EnvironmentState.functional_probe_passed,
            EnvironmentState.hardware_verified,
        }:
            raise EnvironmentManagerError(
                f"managed environment '{env_id}' is {health.state.value}, not functionally verified"
            )
        descriptor = self.load_descriptor(env_id)
        payload = self._run_unjournaled_json_probe(
            descriptor,
            "capabilities",
            _CAPABILITY_SNAPSHOT,
            600,
            cancel,
        )
        raw_profile = payload.get("profile")
        raw_report = payload.get("capability_report")
        if not isinstance(raw_profile, dict) or not isinstance(raw_report, dict):
            raise EnvironmentManagerError(
                "managed capability probe did not emit a profile and capability report",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="managed capability probe did not emit a profile and capability report",
                    detail=str(payload.get("error") or payload),
                ),
            )
        try:
            profile = EnvironmentProfile.model_validate(raw_profile)
            report = CapabilityReport.model_validate(raw_report)
        except ValueError as exc:
            raise EnvironmentManagerError(
                "managed capability probe emitted invalid contracts",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="managed capability probe emitted invalid contracts",
                    detail=str(exc),
                ),
            ) from exc
        if report.backend_id != "corpus_studio" or report.environment_ref.id != profile.environment_signature:
            raise EnvironmentManagerError(
                "managed capability report does not match its environment profile"
            )
        return profile, report

    def load_descriptor(self, env_id: str) -> EnvironmentDescriptor:
        path = self._registry_dir(env_id) / _DESCRIPTOR_FILENAME
        try:
            return EnvironmentDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EnvironmentManagerError(f"unknown managed environment '{env_id}'") from exc
        except (OSError, ValueError) as exc:
            raise EnvironmentManagerError(
                f"managed descriptor for '{env_id}' is unreadable: {exc}"
            ) from exc

    def load_lock(self, env_id: str) -> EnvironmentLock:
        descriptor = self.load_descriptor(env_id)
        if descriptor.lock_ref is None:
            raise EnvironmentManagerError(f"environment '{env_id}' has no lock")
        path = self._registry_dir(env_id) / "locks" / f"{descriptor.lock_ref.id}.json"
        try:
            return EnvironmentLock.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EnvironmentManagerError(
                f"environment lock for '{env_id}' is missing or unreadable: {exc}"
            ) from exc

    def load_health(self, env_id: str) -> EnvironmentHealthReport:
        path = self._registry_dir(env_id) / _HEALTH_FILENAME
        try:
            return EnvironmentHealthReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EnvironmentManagerError(
                f"environment health for '{env_id}' is missing or unreadable: {exc}"
            ) from exc

    def remove(self, env_id: str, *, confirmed_env_id: str) -> EnvironmentDescriptor:
        """Delete only an owned, contained environment; registry evidence is retained."""
        if confirmed_env_id != env_id:
            raise EnvironmentManagerError(
                f"removal requires the exact environment id confirmation '{env_id}'"
            )
        descriptor = self.load_descriptor(env_id)
        env_root = self.environment_root(env_id)
        if Path(descriptor.root_path).resolve(strict=False) != env_root:
            raise EnvironmentManagerError("descriptor root does not match the managed path")
        if env_root.exists():
            self._assert_owned(env_root, env_id)
            # The path was resolved and proven to be a direct child of environments_root above.
            shutil.rmtree(env_root)
        descriptor = self._update_descriptor(
            descriptor,
            state=EnvironmentState.not_installed,
            notes=descriptor.notes + [f"Removed by CorpusStudio at {_timestamp(self.now)}."],
        )
        self._write_health(
            env_id,
            EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.not_installed,
                checked_at=_timestamp(self.now),
            ),
        )
        return descriptor

    def recreate(
        self,
        resolution: DependencyResolution,
        *,
        confirmed_resolution_hash: str,
        confirmed_remove_env_id: str,
        cancel: CancellationToken | None = None,
    ) -> EnvironmentCreationResult:
        if resolution.environment_ref is None:
            raise EnvironmentManagerError("resolution has no environment identity")
        # Validate the complete new seal before destructive removal. The create call validates again
        # after removal, but this first pass guarantees a typo/tampered/blocked plan preserves the
        # currently working environment.
        self._validate_creation(resolution, confirmed_resolution_hash)
        self.remove(
            resolution.environment_ref.id,
            confirmed_env_id=confirmed_remove_env_id,
        )
        return self.create(
            resolution,
            confirmed_resolution_hash=confirmed_resolution_hash,
            cancel=cancel,
        )

    def _validate_creation(
        self, resolution: DependencyResolution, confirmed_hash: str
    ) -> tuple[EnvironmentRecipe, str, Path]:
        if not resolution.resolution_hash or confirmed_hash != resolution.resolution_hash:
            raise EnvironmentManagerError(
                "creation requires the exact resolution hash printed by env-plan"
            )
        if resolution_digest(resolution) != resolution.resolution_hash:
            raise EnvironmentManagerError("the dependency resolution was modified after review")
        if not resolution.resolvable:
            raise EnvironmentManagerError(
                "the dependency resolution is blocked: " + "; ".join(resolution.blocking_reasons)
            )
        if resolution.manager_version != MANAGER_VERSION:
            raise EnvironmentManagerError("the plan was produced by a different manager version")
        recipe = get_recipe(resolution.recipe_ref.id)
        if recipe is None or recipe.recipe_id not in SUPPORTED_CREATION_RECIPES:
            raise EnvironmentManagerError(
                "creation is currently supported only for backend-corpus-studio"
            )
        expected_recipe_hash = recipe_digest(recipe)
        if (
            resolution.recipe_ref.hash is None
            or resolution.recipe_ref.hash.value != expected_recipe_hash
        ):
            raise EnvironmentManagerError("the recipe changed after this plan was reviewed")
        if recipe.layer != DependencyLayer.backend_worker:
            raise EnvironmentManagerError("only isolated backend-worker recipes may be created")
        if resolution.environment_ref is None or resolution.environment_root is None:
            raise EnvironmentManagerError("the plan does not pin an environment id and root")
        if resolution.runtime is None or not resolution.runtime.compatible:
            raise EnvironmentManagerError("the plan does not pin a compatible Python runtime")
        env_id = resolution.environment_ref.id
        env_root = self.environment_root(env_id)
        if Path(resolution.environment_root).resolve(strict=False) != env_root:
            raise EnvironmentManagerError("the planned environment root is outside manager ownership")
        expected = self.preview(
            recipe.recipe_id,
            env_id=env_id,
            runtime_executable=resolution.runtime.executable,
            accelerator_tag=resolution.accelerator_tag,
        )
        if expected != resolution:
            raise EnvironmentManagerError(
                "the dependency resolution no longer matches the manager's canonical plan"
            )
        placeholders = ("<BASE_PYTHON>", "<CONTROL_PLANE_PYTHON>", "<ENV_ROOT>")
        for step in resolution.install_steps:
            if any(
                placeholder in token
                for token in step.argv
                for placeholder in placeholders
            ):
                raise EnvironmentManagerError("the install plan still contains unresolved placeholders")
        return recipe, env_id, env_root

    def _execute_step(
        self,
        env_id: str,
        installation: EnvironmentInstallation,
        step: InstallStep,
        *,
        cancel: CancellationToken | None,
    ) -> EnvironmentInstallation:
        return self._execute_recorded(
            env_id,
            installation,
            phase=step.phase,
            argv=step.argv,
            cwd=Path(step.working_directory or self.environment_root(env_id)),
            environment=step.environment,
            timeout_seconds=step.timeout_seconds,
            expected_outputs=step.expected_outputs,
            cancel=cancel,
        )

    def _execute_recorded(
        self,
        env_id: str,
        installation: EnvironmentInstallation,
        *,
        phase: EnvironmentCommandPhase,
        argv: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        expected_outputs: Sequence[str] = (),
        cancel: CancellationToken | None,
        raise_on_failure: bool = True,
    ) -> EnvironmentInstallation:
        number = len(installation.commands) + 1
        log_dir = self._registry_dir(env_id) / "logs" / installation.installation_id
        stdout_path = log_dir / f"{number:02d}-{phase}.stdout.log"
        stderr_path = log_dir / f"{number:02d}-{phase}.stderr.log"
        started = _timestamp(self.now)
        log_dir.mkdir(parents=True, exist_ok=True)
        runner_failure: FailureRecord | None = None
        try:
            outcome = self.runner(
                list(argv),
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cancel=cancel,
            )
        except Exception as exc:
            outcome = CommandOutcome(exit_code=None)
            runner_failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message=f"environment command could not start during {phase}",
                detail=str(exc),
                exception_type=type(exc).__name__,
                remediation="Inspect the recorded argv and host permissions, then explicitly recreate.",
            )
            stderr_path.write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8", errors="replace"
            )
        finished = _timestamp(self.now)
        stdout = _read_tail(stdout_path)
        stderr = _read_tail(stderr_path)
        failure = runner_failure
        if failure is None and outcome.cancelled:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                exit_code=outcome.exit_code,
                message=f"environment command cancelled during {phase}",
                detail=stderr or stdout or None,
                detected_at=finished,
                remediation="Inspect the partial environment, then explicitly recreate it.",
            )
        elif failure is None and outcome.timed_out:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.TIMEOUT,
                exit_code=outcome.exit_code,
                message=f"environment command timed out during {phase}",
                detail=stderr or stdout or None,
                detected_at=finished,
                remediation="Inspect logs and network/toolchain health; never silently change sources.",
            )
        elif failure is None and outcome.exit_code != 0:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                exit_code=outcome.exit_code,
                message=f"environment command failed during {phase}",
                detail=stderr or stdout or None,
                detected_at=finished,
                remediation="Inspect the recorded argv and stderr; retry requires explicit recreate.",
            )
        elif failure is None and expected_outputs:
            missing_outputs = [
                output
                for output in expected_outputs
                if not (Path(output) if Path(output).is_absolute() else cwd / output).exists()
            ]
            if missing_outputs:
                failure = FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    exit_code=outcome.exit_code,
                    message=f"environment command did not produce expected output during {phase}",
                    detail="missing: " + ", ".join(missing_outputs),
                    detected_at=finished,
                    remediation="Inspect the command logs and partial environment, then recreate.",
                )
        record = EnvironmentCommandRecord(
            command_id=f"command-{number:03d}",
            phase=phase,
            argv=list(argv),
            working_directory=str(cwd),
            environment=dict(environment),
            timeout_seconds=timeout_seconds,
            expected_outputs=list(expected_outputs),
            started_at=started,
            finished_at=finished,
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            cancelled=outcome.cancelled,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            native_build_occurred=_native_build_evidence(stderr, stdout),
            failure=failure,
        )
        installation = installation.model_copy(
            update={"commands": installation.commands + [record]}
        )
        self._write_installation(env_id, installation)
        if failure is not None and raise_on_failure:
            raise EnvironmentManagerError(failure.message, failure, installation)
        return installation

    def _capture_lock(
        self,
        descriptor: EnvironmentDescriptor,
        resolution: DependencyResolution,
        installation: EnvironmentInstallation,
        *,
        cancel: CancellationToken | None,
    ) -> tuple[EnvironmentLock, EnvironmentInstallation]:
        installation = self._execute_recorded(
            descriptor.env_id,
            installation,
            phase="lock",
            argv=[descriptor.python_executable, "-c", _LOCK_PROBE],
            cwd=Path(descriptor.root_path),
            environment=_command_environment(),
            timeout_seconds=120,
            cancel=cancel,
        )
        output = Path(installation.commands[-1].stdout_path or "")
        try:
            payload = _last_json_object(_read_tail(output, _LOCK_OUTPUT_LIMIT))
            lock = self._lock_from_payload(
                descriptor,
                resolution.recipe_ref,
                resolution.resolved_index_urls,
                payload,
            )
        except EnvironmentManagerError as exc:
            installation = self._mark_last_command_failure(
                descriptor.env_id, installation, exc.failure
            )
            exc.installation = installation
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message="the environment lock probe returned malformed data",
                detail=str(exc),
            )
            installation = self._mark_last_command_failure(
                descriptor.env_id, installation, failure
            )
            raise EnvironmentManagerError(
                "the environment lock probe returned malformed data",
                failure,
                installation,
            ) from exc
        self._write_lock(descriptor.env_id, lock)
        return lock, installation

    def _inspect_live_lock(
        self,
        descriptor: EnvironmentDescriptor,
        recipe_ref: Ref,
        index_urls: Sequence[str],
        cancel: CancellationToken | None,
    ) -> EnvironmentLock:
        registry = self._registry_dir(descriptor.env_id)
        log_dir = registry / "logs" / "health"
        stamp = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stamp}-lock.stdout.log"
        stderr_path = log_dir / f"{stamp}-lock.stderr.log"
        try:
            outcome = self.runner(
                [descriptor.python_executable, "-c", _LOCK_PROBE],
                cwd=Path(descriptor.root_path),
                environment=_command_environment(),
                timeout_seconds=120,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cancel=cancel,
            )
        except Exception as exc:
            raise EnvironmentManagerError(
                "failed to start live environment lock inspection",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="failed to start live environment lock inspection",
                    detail=str(exc),
                    exception_type=type(exc).__name__,
                ),
            ) from exc
        if outcome.exit_code != 0 or outcome.timed_out or outcome.cancelled:
            raise EnvironmentManagerError(
                "failed to inspect the live environment for drift",
                FailureRecord(
                    taxonomy=FailureTaxonomy.TIMEOUT
                    if outcome.timed_out
                    else FailureTaxonomy.ENVIRONMENT_FAILURE,
                    exit_code=outcome.exit_code,
                    message="failed to inspect the live environment for drift",
                    detail=_read_tail(stderr_path) or None,
                ),
            )
        try:
            payload = _last_json_object(_read_tail(stdout_path, _LOCK_OUTPUT_LIMIT))
            return self._lock_from_payload(descriptor, recipe_ref, index_urls, payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise EnvironmentManagerError(
                "the live environment lock probe returned malformed data",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="the live environment lock probe returned malformed data",
                    detail=str(exc),
                ),
            ) from exc

    def _lock_from_payload(
        self,
        descriptor: EnvironmentDescriptor,
        recipe_ref: Ref,
        index_urls: Sequence[str],
        payload: Mapping[str, object],
    ) -> EnvironmentLock:
        raw_runtime = payload.get("runtime")
        raw_packages = payload.get("packages")
        raw_torch = payload.get("torch")
        if not isinstance(raw_runtime, dict) or not isinstance(raw_packages, list):
            raise EnvironmentManagerError("the environment lock probe returned malformed data")
        runtime = _runtime_from_payload(raw_runtime, ">=3.11")
        packages: list[PackageLock] = []
        for raw in raw_packages:
            if not isinstance(raw, dict):
                continue
            installer = str(raw.get("installer") or "") or None
            source, direct_url, artifact = _package_source(raw.get("direct_url"))
            record_hash = str(raw.get("record_sha256") or "") or None
            packages.append(
                PackageLock(
                    name=str(raw.get("name") or ""),
                    version=str(raw.get("version") or "") or None,
                    hash=HashRef(value=record_hash) if record_hash else None,
                    source=source,
                    direct_url=direct_url,
                    artifact=artifact,
                    installer=installer,
                    requested=bool(raw.get("requested")),
                    dependencies=sorted(str(item) for item in raw.get("dependencies") or []),
                )
            )
        packages.sort(key=lambda package: package.name.casefold())
        torch_data = raw_torch if isinstance(raw_torch, dict) else {}
        draft = EnvironmentLock(
            lock_id="lock-pending",
            recipe_ref=recipe_ref,
            created_at=_timestamp(self.now),
            manager_version=MANAGER_VERSION,
            runtime=runtime,
            python_version=runtime.version,
            platform_tag=runtime.platform,
            architecture=runtime.architecture,
            implementation=runtime.implementation,
            torch_version=str(torch_data.get("version") or "") or None,
            torch_build=str(torch_data.get("build") or "") or None,
            cuda_runtime_version=str(torch_data.get("cuda") or "") or None,
            compute_capability=str(torch_data.get("compute_capability") or "") or None,
            index_urls=list(index_urls),
            packages=packages,
        )
        digest = self._lock_digest(draft)
        return draft.model_copy(
            update={"lock_id": f"lock-{digest[:20]}", "lock_hash": digest}
        )

    def _lock_digest(self, lock: EnvironmentLock) -> str:
        return _canonical_sha256(
            lock.model_dump(
                mode="json",
                exclude={"lock_id", "created_at", "lock_hash"},
            )
        )

    def _run_creation_probes(
        self,
        descriptor: EnvironmentDescriptor,
        installation: EnvironmentInstallation,
        *,
        cancel: CancellationToken | None,
    ) -> tuple[list[ProbeResult], EnvironmentState, EnvironmentInstallation]:
        results: list[ProbeResult] = []
        installation, import_result = self._json_probe(
            descriptor,
            installation,
            phase="import_probe",
            script=_IMPORT_PROBE,
            timeout_seconds=180,
            cancel=cancel,
        )
        raw_imports = import_result.get("results")
        imports_ok = isinstance(raw_imports, dict) and all(
            isinstance(value, dict) and bool(value.get("ok"))
            for value in raw_imports.values()
        )
        results.append(
            ProbeResult(
                probe="reference_backend_imports",
                outcome=FailureTaxonomy.PASS if imports_ok else FailureTaxonomy.ENVIRONMENT_FAILURE,
                detail="all reference backend modules imported" if imports_ok else "one or more imports failed",
                measured=import_result,
            )
        )
        if not imports_ok:
            return results, EnvironmentState.degraded, installation

        descriptor = self._update_descriptor(descriptor, state=EnvironmentState.importable)
        installation = self._execute_recorded(
            descriptor.env_id,
            installation,
            phase="dependency_probe",
            argv=[descriptor.python_executable, "-m", "pip", "check"],
            cwd=Path(descriptor.root_path),
            environment=_command_environment(),
            timeout_seconds=180,
            cancel=cancel,
            raise_on_failure=False,
        )
        dependency_failure = installation.commands[-1].failure
        results.append(
            ProbeResult(
                probe="pip_check",
                outcome=FailureTaxonomy.PASS
                if dependency_failure is None
                else FailureTaxonomy.ENVIRONMENT_FAILURE,
                detail="pip dependency graph is consistent"
                if dependency_failure is None
                else dependency_failure.detail or dependency_failure.message,
            )
        )
        if dependency_failure is not None:
            return results, EnvironmentState.degraded, installation
        descriptor = self._update_descriptor(
            descriptor, state=EnvironmentState.dependency_probe_passed
        )

        installation, functional = self._json_probe(
            descriptor,
            installation,
            phase="functional_probe",
            script=_FUNCTIONAL_PROBE,
            timeout_seconds=180,
            cancel=cancel,
        )
        functional_ok = bool(functional.get("ok"))
        results.append(
            ProbeResult(
                probe="reference_backend_functional",
                outcome=FailureTaxonomy.PASS if functional_ok else FailureTaxonomy.FAIL,
                detail="tiny forward/backward/checkpoint round-trip passed"
                if functional_ok
                else str(functional.get("error") or "functional operation failed"),
                measured=functional,
            )
        )
        if not functional_ok:
            return results, EnvironmentState.degraded, installation
        descriptor = self._update_descriptor(
            descriptor, state=EnvironmentState.functional_probe_passed
        )

        installation, hardware = self._json_probe(
            descriptor,
            installation,
            phase="hardware_probe",
            script=_HARDWARE_PROBE,
            timeout_seconds=240,
            cancel=cancel,
        )
        cuda_available = bool(hardware.get("cuda_available"))
        hardware_ok = bool(hardware.get("ok"))
        if not cuda_available:
            outcome = FailureTaxonomy.UNSUPPORTED_CONFIGURATION
            detail = "CPU functional path passed; no CUDA hardware was available to verify"
            state = EnvironmentState.functional_probe_passed
        elif hardware_ok:
            outcome = FailureTaxonomy.PASS
            detail = "CUDA allocation, 4-bit construction, forward/backward, and math attention passed"
            state = EnvironmentState.hardware_verified
        else:
            outcome = FailureTaxonomy.FAIL
            detail = str(hardware.get("error") or hardware.get("four_bit_error") or "hardware probe failed")
            state = EnvironmentState.incompatible
        results.append(
            ProbeResult(
                probe="reference_backend_hardware",
                outcome=outcome,
                detail=detail,
                measured=hardware,
            )
        )
        return results, state, installation

    def _health_probes(
        self,
        descriptor: EnvironmentDescriptor,
        *,
        cancel: CancellationToken | None,
    ) -> tuple[list[ProbeResult], EnvironmentState]:
        results: list[ProbeResult] = []
        imports = self._run_unjournaled_json_probe(
            descriptor, "import", _IMPORT_PROBE, 180, cancel
        )
        raw_imports = imports.get("results")
        imports_ok = isinstance(raw_imports, dict) and all(
            isinstance(value, dict) and bool(value.get("ok"))
            for value in raw_imports.values()
        )
        results.append(
            ProbeResult(
                probe="reference_backend_imports",
                outcome=FailureTaxonomy.PASS if imports_ok else FailureTaxonomy.ENVIRONMENT_FAILURE,
                measured=imports,
            )
        )
        if not imports_ok:
            return results, EnvironmentState.degraded
        dependency_outcome, dependency_detail = self._run_unjournaled_command(
            descriptor,
            "dependency",
            [descriptor.python_executable, "-m", "pip", "check"],
            180,
            cancel,
        )
        dependency_ok = (
            dependency_outcome.exit_code == 0
            and not dependency_outcome.timed_out
            and not dependency_outcome.cancelled
        )
        results.append(
            ProbeResult(
                probe="pip_check",
                outcome=FailureTaxonomy.PASS
                if dependency_ok
                else FailureTaxonomy.ENVIRONMENT_FAILURE,
                detail="pip dependency graph is consistent"
                if dependency_ok
                else dependency_detail or "pip check failed",
            )
        )
        if not dependency_ok:
            return results, EnvironmentState.degraded
        functional = self._run_unjournaled_json_probe(
            descriptor, "functional", _FUNCTIONAL_PROBE, 180, cancel
        )
        functional_ok = bool(functional.get("ok"))
        results.append(
            ProbeResult(
                probe="reference_backend_functional",
                outcome=FailureTaxonomy.PASS if functional_ok else FailureTaxonomy.FAIL,
                measured=functional,
            )
        )
        if not functional_ok:
            return results, EnvironmentState.degraded
        hardware = self._run_unjournaled_json_probe(
            descriptor, "hardware", _HARDWARE_PROBE, 240, cancel
        )
        if not bool(hardware.get("cuda_available")):
            outcome = FailureTaxonomy.UNSUPPORTED_CONFIGURATION
            state = EnvironmentState.functional_probe_passed
        elif bool(hardware.get("ok")):
            outcome = FailureTaxonomy.PASS
            state = EnvironmentState.hardware_verified
        else:
            outcome = FailureTaxonomy.FAIL
            state = EnvironmentState.incompatible
        results.append(
            ProbeResult(
                probe="reference_backend_hardware",
                outcome=outcome,
                measured=hardware,
            )
        )
        return results, state

    def _json_probe(
        self,
        descriptor: EnvironmentDescriptor,
        installation: EnvironmentInstallation,
        *,
        phase: Literal["import_probe", "functional_probe", "hardware_probe"],
        script: str,
        timeout_seconds: int,
        cancel: CancellationToken | None,
    ) -> tuple[EnvironmentInstallation, dict[str, object]]:
        installation = self._execute_recorded(
            descriptor.env_id,
            installation,
            phase=phase,
            argv=[descriptor.python_executable, "-c", script],
            cwd=Path(descriptor.root_path),
            environment=_command_environment(),
            timeout_seconds=timeout_seconds,
            cancel=cancel,
        )
        path = Path(installation.commands[-1].stdout_path or "")
        try:
            return installation, _last_json_object(_read_tail(path))
        except json.JSONDecodeError as exc:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message=f"{phase} did not emit a structured result",
                detail=_read_tail(path) or None,
            )
            installation = self._mark_last_command_failure(
                descriptor.env_id, installation, failure
            )
            raise EnvironmentManagerError(
                f"{phase} did not emit a structured result",
                failure,
                installation,
            ) from exc

    def _mark_last_command_failure(
        self,
        env_id: str,
        installation: EnvironmentInstallation,
        failure: FailureRecord,
    ) -> EnvironmentInstallation:
        if not installation.commands:
            return installation
        commands = list(installation.commands)
        commands[-1] = commands[-1].model_copy(update={"failure": failure})
        updated = installation.model_copy(update={"commands": commands})
        self._write_installation(env_id, updated)
        return updated

    def _run_unjournaled_json_probe(
        self,
        descriptor: EnvironmentDescriptor,
        label: str,
        script: str,
        timeout_seconds: int,
        cancel: CancellationToken | None,
    ) -> dict[str, object]:
        log_dir = self._registry_dir(descriptor.env_id) / "logs" / "health"
        stamp = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stamp}-{label}.stdout.log"
        stderr_path = log_dir / f"{stamp}-{label}.stderr.log"
        try:
            outcome = self.runner(
                [descriptor.python_executable, "-c", script],
                cwd=Path(descriptor.root_path),
                environment=_command_environment(),
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cancel=cancel,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if outcome.exit_code != 0 or outcome.timed_out or outcome.cancelled:
            return {"ok": False, "error": _read_tail(stderr_path) or "probe command failed"}
        try:
            return _last_json_object(_read_tail(stdout_path))
        except json.JSONDecodeError:
            return {"ok": False, "error": "probe did not emit structured JSON"}

    def _run_unjournaled_command(
        self,
        descriptor: EnvironmentDescriptor,
        label: str,
        argv: Sequence[str],
        timeout_seconds: int,
        cancel: CancellationToken | None,
    ) -> tuple[CommandOutcome, str]:
        log_dir = self._registry_dir(descriptor.env_id) / "logs" / "health"
        stamp = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stamp}-{label}.stdout.log"
        stderr_path = log_dir / f"{stamp}-{label}.stderr.log"
        try:
            outcome = self.runner(
                list(argv),
                cwd=Path(descriptor.root_path),
                environment=_command_environment(),
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cancel=cancel,
            )
        except Exception as exc:
            return CommandOutcome(exit_code=None), f"{type(exc).__name__}: {exc}"
        return outcome, _read_tail(stderr_path) or _read_tail(stdout_path)

    def _package_drift(
        self, expected: Sequence[PackageLock], actual: Sequence[PackageLock]
    ) -> tuple[list[str], list[str]]:
        expected_by_name = {item.name.casefold(): item for item in expected}
        actual_by_name = {item.name.casefold(): item for item in actual}
        drifted: list[str] = []
        sources: list[str] = []
        for name in sorted(expected_by_name.keys() | actual_by_name.keys()):
            before = expected_by_name.get(name)
            after = actual_by_name.get(name)
            if before is None:
                assert after is not None
                drifted.append(f"{after.name}: unexpected package")
                continue
            if after is None:
                drifted.append(f"{before.name}: missing (locked {before.version})")
                continue
            if before.version != after.version:
                drifted.append(f"{before.name}: {before.version} -> {after.version}")
            before_hash = before.hash.value if before.hash else None
            after_hash = after.hash.value if after.hash else None
            if before_hash != after_hash:
                drifted.append(f"{before.name}: installed RECORD hash changed")
            if before.source != after.source or before.direct_url != after.direct_url:
                sources.append(f"{before.name}: {before.source} -> {after.source}")
        return drifted, sources

    def _write_owner(self, env_root: Path, env_id: str, created_at: str) -> None:
        _atomic_write(
            env_root / _OWNER_FILENAME,
            json.dumps(
                {
                    "kind": _OWNER_KIND,
                    "env_id": env_id,
                    "manager_root": str(self.root.resolve(strict=False)),
                    "created_at": created_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def _assert_owned(self, env_root: Path, env_id: str) -> None:
        expected = self.environment_root(env_id)
        if env_root.resolve(strict=False) != expected:
            raise EnvironmentManagerError("refusing deletion outside the managed environments root")
        try:
            marker = json.loads((env_root / _OWNER_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentManagerError(
                "refusing deletion: CorpusStudio ownership marker is missing or invalid"
            ) from exc
        if (
            marker.get("kind") != _OWNER_KIND
            or marker.get("env_id") != env_id
            or Path(str(marker.get("manager_root"))).resolve(strict=False)
            != self.root.resolve(strict=False)
        ):
            raise EnvironmentManagerError(
                "refusing deletion: ownership marker does not match this manager"
            )

    def _registry_dir(self, env_id: str) -> Path:
        self._validate_env_id(env_id)
        path = (self.registry_root / env_id).resolve(strict=False)
        if path.parent != self.registry_root.resolve(strict=False):
            raise EnvironmentManagerError("registry path escapes the managed root")
        return path

    def _write_descriptor(self, descriptor: EnvironmentDescriptor) -> None:
        _atomic_model(
            self._registry_dir(descriptor.env_id) / _DESCRIPTOR_FILENAME,
            descriptor,
        )

    def _write_installation(
        self, env_id: str, installation: EnvironmentInstallation
    ) -> None:
        _atomic_model(
            self._registry_dir(env_id)
            / "installations"
            / f"{installation.installation_id}.json",
            installation,
        )

    def _write_lock(self, env_id: str, lock: EnvironmentLock) -> None:
        _atomic_model(
            self._registry_dir(env_id) / "locks" / f"{lock.lock_id}.json",
            lock,
        )

    def _write_health(self, env_id: str, health: EnvironmentHealthReport) -> None:
        _atomic_model(self._registry_dir(env_id) / _HEALTH_FILENAME, health)

    def _update_descriptor(
        self,
        descriptor: EnvironmentDescriptor,
        *,
        state: EnvironmentState | None = None,
        lock_ref: Ref | None = None,
        notes: list[str] | None = None,
    ) -> EnvironmentDescriptor:
        update: dict[str, object] = {"updated_at": _timestamp(self.now)}
        if state is not None:
            update["state"] = state
        if lock_ref is not None:
            update["lock_ref"] = lock_ref
        if notes is not None:
            update["notes"] = notes
        descriptor = descriptor.model_copy(update=update)
        self._write_descriptor(descriptor)
        return descriptor

    @staticmethod
    def _environment_python(env_root: Path, os_value: OperatingSystem) -> str:
        if os_value == OperatingSystem.windows:
            return str(env_root / "Scripts" / "python.exe")
        return str(env_root / "bin" / "python")

    @staticmethod
    def _probe_remediation(state: EnvironmentState) -> str | None:
        if state == EnvironmentState.functional_probe_passed:
            return "CPU functionality passed; run env-probe on a CUDA host for hardware verification."
        if state in {EnvironmentState.degraded, EnvironmentState.incompatible}:
            return "Inspect probe evidence and recreate from a compatible, newly reviewed plan."
        return None

    @staticmethod
    def _failure_from_exception(exc: Exception) -> FailureRecord:
        if isinstance(exc, EnvironmentManagerError):
            return exc.failure
        return FailureRecord(
            taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
            message=f"environment creation failed: {exc}",
            exception_type=type(exc).__name__,
        )

    @staticmethod
    def _validate_env_id(env_id: str) -> None:
        if not env_id or not _ID_PATTERN.fullmatch(env_id) or env_id in {".", ".."}:
            raise EnvironmentManagerError(
                "environment id must contain only letters, numbers, dot, underscore, or hyphen"
            )


def locked_environment_ref(
    descriptor: EnvironmentDescriptor, lock: EnvironmentLock
) -> Ref:
    """Pin a RunPlan to the immutable lock, never to the mutable environment directory."""
    if lock.lock_hash is None:
        raise EnvironmentManagerError("cannot reference an unsealed environment lock")
    if descriptor.lock_ref is None or descriptor.lock_ref.id != lock.lock_id:
        raise EnvironmentManagerError("descriptor and lock do not match")
    return Ref(id=descriptor.env_id, hash=HashRef(value=lock.lock_hash))


def verify_run_plan_environment(
    plan: RunPlan, descriptor: EnvironmentDescriptor, lock: EnvironmentLock
) -> list[str]:
    """Return resume/dispatch blockers when the plan's pinned environment no longer matches."""
    blockers: list[str] = []
    expected = locked_environment_ref(descriptor, lock)
    if plan.environment_ref.id != expected.id:
        blockers.append(
            f"plan environment id {plan.environment_ref.id} != managed environment {expected.id}"
        )
    plan_hash = plan.environment_ref.hash.value if plan.environment_ref.hash else None
    if plan_hash != lock.lock_hash:
        blockers.append("plan environment lock hash does not match the live managed lock")
    if descriptor.state not in {
        EnvironmentState.functional_probe_passed,
        EnvironmentState.hardware_verified,
    }:
        blockers.append(
            f"environment state {descriptor.state.value} is not functionally verified"
        )
    return blockers
