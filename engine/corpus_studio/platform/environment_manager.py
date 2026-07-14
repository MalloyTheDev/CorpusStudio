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
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import cast, Literal, Protocol
from urllib.parse import unquote, urlparse, urlsplit, urlunsplit
import uuid
import zipfile

from .backends import backend_manifest_digest, get_worker_backend
from .common import HashRef, PackageLock, PackageSource, Ref
from .contracts import (
    CapabilityReport,
    DependencyResolution,
    EnvironmentCommandRecord,
    EnvironmentDescriptor,
    EnvironmentHealthReport,
    EnvironmentInstallation,
    EnvironmentLock,
    EnvironmentProbeEvidence,
    EnvironmentProfile,
    EnvironmentRecipe,
    FailureRecord,
    InstalledEnvironmentEvidence,
    InstallStep,
    PackageInstallEvidence,
    ProbeMemoryEvidence,
    ProbeResult,
    PythonRuntime,
    QloraExecutionProbeSpec,
    RunPlan,
    WorkerArtifactIdentity,
)
from .enums import DependencyLayer, EnvironmentState, FailureTaxonomy, OperatingSystem
from .environments import (
    get_recipe,
    READINESS_FLASH_V1_RECIPE_ID,
    READINESS_V2_RECIPE_ID,
    recipe_digest,
    resolution_digest,
    resolve_dependencies,
)
from .process_control import (
    process_group_creation_flags,
    start_new_process_session,
    terminate_process_tree,
)

MANAGER_VERSION = "1.1.0"
REFERENCE_RECIPE_ID = "backend-corpus-studio"
SUPPORTED_CREATION_RECIPES = frozenset(
    {REFERENCE_RECIPE_ID, READINESS_V2_RECIPE_ID, READINESS_FLASH_V1_RECIPE_ID}
)

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
    "inventory",
    "import_probe",
    "dependency_probe",
    "functional_probe",
    "hardware_probe",
    "capability_probe",
    "health_probe",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(now: Callable[[], datetime]) -> str:
    return now().astimezone(timezone.utc).isoformat()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sanitize_url(value: str | None) -> str | None:
    """Remove credentials, query parameters, and fragments from persisted source evidence."""
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if not parsed.scheme:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_artifact_identity(path: str | Path) -> WorkerArtifactIdentity:
    wheel = Path(path).expanduser().resolve(strict=True)
    if not wheel.is_file() or wheel.suffix.casefold() != ".whl":
        raise EnvironmentManagerError("the readiness worker artifact must be a concrete wheel")
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = sorted(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            if len(metadata_names) != 1:
                raise EnvironmentManagerError(
                    "the worker wheel must contain exactly one dist-info/METADATA file"
                )
            metadata_bytes = archive.read(metadata_names[0])
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise EnvironmentManagerError(f"the worker wheel is unreadable: {exc}") from exc
    metadata = Parser().parsestr(metadata_bytes.decode("utf-8", errors="strict"))
    name = str(metadata.get("Name") or "")
    version = str(metadata.get("Version") or "")
    if not name or not version or _normalized_package_name(name) != "corpus-studio-engine":
        raise EnvironmentManagerError(
            "the worker wheel METADATA must identify the corpus-studio-engine distribution"
        )
    return WorkerArtifactIdentity(
        distribution_name=name,
        normalized_name=_normalized_package_name(name),
        version=version,
        filename=wheel.name,
        path=str(wheel),
        size_bytes=wheel.stat().st_size,
        content_hash=HashRef(value=_hash_file(wheel)),
        metadata_hash=HashRef(value=hashlib.sha256(metadata_bytes).hexdigest()),
    )


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
                creationflags=process_group_creation_flags(),
                start_new_session=start_new_process_session(),
            )
            started = time.monotonic()
            timed_out = False
            cancelled = False
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        terminate_process_tree(process)
                        break
                    if time.monotonic() - started >= timeout_seconds:
                        timed_out = True
                        terminate_process_tree(process)
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                cancelled = True
                terminate_process_tree(process)
            if process.poll() is None:
                terminate_process_tree(process)
            return CommandOutcome(
                exit_code=process.returncode, timed_out=timed_out, cancelled=cancelled
            )

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
    # Probe failures deliberately return an unsealed result. A lock exists only after required
    # evidence passes and the post-probe inventory is stable.
    lock: EnvironmentLock | None
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
import base64, csv, hashlib, importlib.metadata as metadata, io, json, os, platform, re, struct, sys
packages = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or getattr(dist, "name", "")
    if not name: continue
    direct_text = dist.read_text("direct_url.json")
    try: direct = json.loads(direct_text) if direct_text else None
    except Exception: direct = None
    record = dist.read_text("RECORD")
    installer = dist.read_text("INSTALLER")
    record_entries = 0
    verified_entries = 0
    failed_entries = []
    if record:
        for row in csv.reader(io.StringIO(record)):
            if not row: continue
            record_entries += 1
            relative = row[0]
            hash_spec = row[1] if len(row) > 1 else ""
            if not hash_spec: continue
            try:
                algorithm, expected = hash_spec.split("=", 1)
                path = dist.locate_file(relative)
                digest = hashlib.new(algorithm, path.read_bytes()).digest()
                actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
                if actual == expected:
                    verified_entries += 1
                else:
                    failed_entries.append(relative)
            except Exception:
                failed_entries.append(relative)
    if not record:
        record_integrity = "missing"
    elif failed_entries:
        record_integrity = "failed"
    elif verified_entries:
        record_integrity = "verified"
    else:
        record_integrity = "unknown"
    packages.append({
      "name": name,
      "normalized_name": re.sub(r"[-_.]+", "-", name).lower(),
      "version": dist.version,
      "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None,
      "record_integrity": record_integrity,
      "record_entries": record_entries,
      "record_verified_entries": verified_entries,
      "record_failed_entries": sorted(failed_entries),
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


_CAPABILITY_SNAPSHOT_TEMPLATE = r"""
import contextlib, json, sys
from corpus_studio.platform.profiler import build_environment_profile
from corpus_studio.platform.probes import run_capability_probes
with contextlib.redirect_stdout(sys.stderr):
    profile = build_environment_profile()
    report = run_capability_probes(profile, probes=__PROBES__)
print(json.dumps({
  "profile": profile.model_dump(mode="json"),
  "capability_report": report.model_dump(mode="json"),
}, sort_keys=True))
""".strip()


def _capability_snapshot_script(probes: Sequence[str] | None) -> str:
    return _CAPABILITY_SNAPSHOT_TEMPLATE.replace(
        "__PROBES__", json.dumps(list(probes) if probes is not None else None, ensure_ascii=True)
    )


def _package_source(direct: object) -> tuple[PackageSource, str | None, str | None]:
    if not isinstance(direct, dict):
        # INSTALLER=pip does not prove which configured index supplied the artifact. Keep the source
        # unknown unless PEP 610 gives us direct evidence; the lock separately records reviewed URLs.
        return ("unknown", None, None)
    raw_url = str(direct.get("url") or "") or None
    url = _sanitize_url(raw_url)
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


def _direct_metadata(direct: object) -> tuple[bool | None, bool | None, str | None, str | None]:
    if not isinstance(direct, dict):
        return None, None, None, None
    dir_info = direct.get("dir_info")
    editable = bool(dir_info.get("editable")) if isinstance(dir_info, dict) else False
    vcs_info = direct.get("vcs_info")
    repository: str | None = None
    commit: str | None = None
    if isinstance(vcs_info, dict):
        repository = _sanitize_url(str(direct.get("url") or ""))
        commit = str(vcs_info.get("commit_id") or "") or None
    return True, editable, repository, commit


def _install_evidence_from_report(
    report_path: Path,
    *,
    step: InstallStep,
    command_id: str,
) -> list[PackageInstallEvidence]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentManagerError(f"pip install evidence is missing or malformed: {exc}") from exc
    raw_installs = payload.get("install") if isinstance(payload, dict) else None
    if not isinstance(raw_installs, list):
        raise EnvironmentManagerError("pip install evidence has no install list")
    configured_indexes = [
        sanitized
        for value in step.configured_index_urls
        if (sanitized := _sanitize_url(value)) is not None
    ]
    evidence: list[PackageInstallEvidence] = []
    for raw in raw_installs:
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata")
        download = raw.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            continue
        name = str(metadata.get("name") or "")
        version = str(metadata.get("version") or "")
        if not name or not version:
            continue
        raw_url = str(download.get("url") or "") or None
        sanitized_url = _sanitize_url(raw_url)
        is_direct = bool(raw.get("is_direct"))
        archive_info = download.get("archive_info")
        dir_info = download.get("dir_info")
        vcs_info = download.get("vcs_info")
        hashes = archive_info.get("hashes") if isinstance(archive_info, dict) else None
        sha256 = str(hashes.get("sha256") or "") if isinstance(hashes, dict) else ""
        artifact = (
            Path(unquote(urlparse(sanitized_url).path)).name
            if sanitized_url
            else None
        ) or None
        if isinstance(vcs_info, dict):
            source: PackageSource = "vcs"
        elif is_direct and isinstance(dir_info, dict):
            source = "local"
        elif is_direct and isinstance(archive_info, dict):
            source = "wheel" if artifact and artifact.casefold().endswith(".whl") else "sdist"
        elif configured_indexes:
            if any("pypi.org" in item for item in configured_indexes):
                source = "pypi"
            elif artifact and artifact.casefold().endswith(".whl"):
                source = "wheel"
            else:
                source = "unknown"
        else:
            source = "unknown"
        reason = None
        if source == "unknown":
            reason = (
                "pip report did not prove an index or direct source"
                if not configured_indexes
                else "configured index is recorded but its source class is not recognized"
            )
        evidence.append(
            PackageInstallEvidence(
                normalized_name=_normalized_package_name(name),
                version=version,
                source=source,
                source_index_url=configured_indexes[0]
                if not is_direct and len(configured_indexes) == 1
                else None,
                direct_url=sanitized_url if is_direct else None,
                artifact_filename=artifact,
                artifact_hash=HashRef(value=sha256) if sha256 else None,
                requested=bool(raw.get("requested")),
                direct=is_direct,
                editable=bool(dir_info.get("editable"))
                if isinstance(dir_info, dict)
                else False,
                vcs_repository=sanitized_url if isinstance(vcs_info, dict) else None,
                vcs_commit=str(vcs_info.get("commit_id") or "") or None
                if isinstance(vcs_info, dict)
                else None,
                installer_command_id=command_id,
                configured_index_urls=configured_indexes,
                source_evidence_reason=reason,
            )
        )
    return sorted(evidence, key=lambda item: (item.normalized_name, item.version))


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
        worker_wheel: str | Path | None = None,
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
        if recipe.requires_worker_wheel and recipe.layer == DependencyLayer.backend_worker:
            worker_artifact: WorkerArtifactIdentity | None = None
            blocking = list(resolution.blocking_reasons)
            if worker_wheel is None:
                blocking.append(
                    "readiness recipes require --worker-wheel pointing to a reviewed CorpusStudio wheel"
                )
            else:
                try:
                    worker_artifact = _worker_artifact_identity(worker_wheel)
                except EnvironmentManagerError as exc:
                    blocking.append(str(exc))
            if worker_artifact is not None:
                evidence_path = str(
                    self.environment_root(env_id)
                    / ".corpusstudio-install-evidence"
                    / "install-worker.json"
                )
                environment_python = self._environment_python(
                    self.environment_root(env_id), runtime.os
                )
                worker_step = InstallStep(
                    phase="install",
                    description="Install the exact hash-bound CorpusStudio worker wheel",
                    argv=[
                        environment_python,
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-index",
                        "--no-deps",
                        "--report",
                        evidence_path,
                        worker_artifact.path,
                    ],
                    working_directory=str(self.environment_root(env_id)),
                    environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                    timeout_seconds=900,
                    network_required=False,
                    evidence_path=evidence_path,
                )
                resolution = resolution.model_copy(
                    update={
                        "install_steps": resolution.install_steps + [worker_step],
                        "worker_artifact": worker_artifact,
                        "resolution_hash": None,
                    }
                )
            if blocking:
                resolution = resolution.model_copy(
                    update={
                        "resolvable": False,
                        "blocking_reasons": blocking,
                        "resolution_hash": None,
                    }
                )
        elif recipe_id == REFERENCE_RECIPE_ID and recipe.layer == DependencyLayer.backend_worker:
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
                    "--report",
                    str(
                        self.environment_root(env_id)
                        / ".corpusstudio-install-evidence"
                        / "install-worker-source.json"
                    ),
                    str(self.engine_source.resolve(strict=False)),
                ],
                working_directory=str(self.environment_root(env_id)),
                environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                timeout_seconds=900,
                network_required=True,
                evidence_path=str(
                    self.environment_root(env_id)
                    / ".corpusstudio-install-evidence"
                    / "install-worker-source.json"
                ),
                configured_index_urls=["https://pypi.org/simple"],
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
                    env_id,
                    installation,
                    step,
                    cancel=cancel,
                    require_evidence=recipe.requires_worker_wheel,
                )
            installation = installation.model_copy(
                update={"worker_artifact": resolution.worker_artifact}
            )
            self._write_installation(env_id, installation)
            descriptor = self._update_descriptor(
                descriptor, state=EnvironmentState.installed_unchecked
            )

            pre_probe, installation = self._capture_inventory(
                descriptor, resolution, installation, cancel=cancel
            )
            installation = installation.model_copy(
                update={"pre_probe_inventory": pre_probe}
            )
            self._write_installation(env_id, installation)
            probe_results, final_state, installation, probe_evidence = self._run_creation_probes(
                descriptor,
                installation,
                recipe=recipe,
                cancel=cancel,
            )
            installation = installation.model_copy(
                update={
                    "probe_results": probe_results,
                    "probe_evidence": probe_evidence,
                }
            )
            self._write_installation(env_id, installation)

            required_probe_passed = (
                recipe.required_execution_probe is None
                or (
                    final_state == EnvironmentState.hardware_verified
                    and probe_evidence is not None
                )
            )
            legacy_probe_passed = final_state in {
                EnvironmentState.functional_probe_passed,
                EnvironmentState.hardware_verified,
            }
            if not required_probe_passed or not legacy_probe_passed:
                descriptor = self._update_descriptor(descriptor, state=final_state)
                finished = _timestamp(self.now)
                installation = installation.model_copy(
                    update={"state": final_state, "finished_at": finished}
                )
                self._write_installation(env_id, installation)
                health = EnvironmentHealthReport(
                    environment_ref=Ref(id=env_id),
                    recipe_ref=descriptor.recipe_ref,
                    state=final_state,
                    python_version=pre_probe.python_version,
                    checked_at=finished,
                    installed_packages=pre_probe.packages,
                    probe_results=probe_results,
                    probe_evidence=probe_evidence,
                    remediation=self._probe_remediation(final_state),
                )
                self._write_health(env_id, health)
                return EnvironmentCreationResult(descriptor, None, health, installation)

            post_probe, installation = self._capture_inventory(
                descriptor, resolution, installation, cancel=cancel
            )
            installation = installation.model_copy(
                update={"post_probe_inventory": post_probe}
            )
            self._write_installation(env_id, installation)
            inventory_drift, inventory_sources = self._package_drift(
                pre_probe.packages, post_probe.packages
            )
            if inventory_drift or inventory_sources:
                final_state = EnvironmentState.drifted
                descriptor = self._update_descriptor(descriptor, state=final_state)
                finished = _timestamp(self.now)
                installation = installation.model_copy(
                    update={"state": final_state, "finished_at": finished}
                )
                self._write_installation(env_id, installation)
                health = EnvironmentHealthReport(
                    environment_ref=Ref(id=env_id),
                    recipe_ref=descriptor.recipe_ref,
                    state=final_state,
                    python_version=post_probe.python_version,
                    checked_at=finished,
                    installed_packages=post_probe.packages,
                    drifted_packages=inventory_drift,
                    changed_package_sources=inventory_sources,
                    drift_detected=True,
                    probe_results=probe_results,
                    probe_evidence=probe_evidence,
                    remediation="Recreate from a newly reviewed plan; probing changed the environment.",
                )
                self._write_health(env_id, health)
                return EnvironmentCreationResult(descriptor, None, health, installation)

            lock = self._finalize_lock(
                post_probe,
                resolution=resolution,
                recipe=recipe,
                installation=installation,
                probe_evidence=probe_evidence,
            )
            self._write_lock(descriptor.env_id, lock)
            descriptor = self._update_descriptor(
                descriptor,
                lock_ref=Ref(id=lock.lock_id, hash=HashRef(value=lock.lock_hash)),
            )

            # A final live inventory is compared with the just-sealed state. This is the first point
            # at which health/drift is evaluated against a final lock.
            live_inventory = self._inspect_live_inventory(
                descriptor, lock.recipe_ref, lock.index_urls, cancel
            )
            drifted, changed_sources = self._package_drift(
                lock.packages, live_inventory.packages
            )
            worker_drift = self._worker_artifact_drift(lock.worker_artifact)
            if worker_drift:
                drifted.append(worker_drift)
            if drifted or changed_sources:
                final_state = EnvironmentState.drifted
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
                drifted_packages=drifted,
                changed_package_sources=changed_sources,
                drift_detected=bool(drifted or changed_sources),
                probe_results=probe_results,
                probe_evidence=probe_evidence,
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
            live_inventory = self._inspect_live_inventory(
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
        drifted, changed_sources = self._package_drift(
            lock.packages, live_inventory.packages
        )
        worker_drift = self._worker_artifact_drift(lock.worker_artifact)
        if worker_drift:
            drifted.append(worker_drift)
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
        cuda_mismatch = lock.cuda_runtime_version != live_inventory.cuda_runtime_version
        hardware_mismatch = lock.compute_capability != live_inventory.compute_capability

        probe_results, probe_state, probe_evidence = self._health_probes(
            descriptor, cancel=cancel
        )
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
            python_version=live_inventory.python_version,
            checked_at=checked_at,
            installed_packages=live_inventory.packages,
            drifted_packages=drifted,
            changed_package_sources=changed_sources,
            drift_detected=drift,
            recipe_drift_detected=recipe_drift,
            lock_mismatch=lock_mismatch,
            hardware_mismatch=hardware_mismatch,
            cuda_mismatch=cuda_mismatch,
            probe_results=probe_results,
            probe_evidence=probe_evidence,
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
        recipe = get_recipe(descriptor.recipe_ref.id)
        probes = (
            recipe.capability_probes
            if recipe is not None and recipe.required_execution_probe is not None
            else None
        )
        payload = self._run_unjournaled_json_probe(
            descriptor,
            "capabilities",
            _capability_snapshot_script(probes),
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
                "creation is currently supported only for CorpusStudio managed-worker recipes"
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
        if recipe.requires_worker_wheel:
            if resolution.worker_artifact is None:
                raise EnvironmentManagerError("the plan does not bind a CorpusStudio worker wheel")
            current_worker = _worker_artifact_identity(resolution.worker_artifact.path)
            if current_worker != resolution.worker_artifact:
                raise EnvironmentManagerError(
                    "the CorpusStudio worker wheel changed after this plan was reviewed"
                )
        expected = self.preview(
            recipe.recipe_id,
            env_id=env_id,
            runtime_executable=resolution.runtime.executable,
            accelerator_tag=resolution.accelerator_tag,
            worker_wheel=resolution.worker_artifact.path
            if resolution.worker_artifact is not None
            else None,
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
        require_evidence: bool = False,
    ) -> EnvironmentInstallation:
        evidence_path = Path(step.evidence_path) if step.evidence_path else None
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
        installation = self._execute_recorded(
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
        if evidence_path is None:
            return installation
        if not evidence_path.is_file():
            if not require_evidence:
                return installation
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message="pip did not produce required dependency-source evidence",
                detail=str(evidence_path),
            )
            installation = self._mark_last_command_failure(env_id, installation, failure)
            raise EnvironmentManagerError(failure.message, failure, installation)
        try:
            captured = _install_evidence_from_report(
                evidence_path,
                step=step,
                command_id=installation.commands[-1].command_id,
            )
        except EnvironmentManagerError as exc:
            installation = self._mark_last_command_failure(env_id, installation, exc.failure)
            exc.installation = installation
            raise
        finally:
            evidence_path.unlink(missing_ok=True)
        by_name = {
            item.normalized_name: item for item in installation.package_install_evidence
        }
        by_name.update({item.normalized_name: item for item in captured})
        installation = installation.model_copy(
            update={
                "package_install_evidence": sorted(
                    by_name.values(), key=lambda item: item.normalized_name
                )
            }
        )
        self._write_installation(env_id, installation)
        return installation

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

    def _capture_inventory(
        self,
        descriptor: EnvironmentDescriptor,
        resolution: DependencyResolution,
        installation: EnvironmentInstallation,
        *,
        cancel: CancellationToken | None,
    ) -> tuple[InstalledEnvironmentEvidence, EnvironmentInstallation]:
        installation = self._execute_recorded(
            descriptor.env_id,
            installation,
            phase="inventory",
            argv=[descriptor.python_executable, "-c", _LOCK_PROBE],
            cwd=Path(descriptor.root_path),
            environment=_command_environment(),
            timeout_seconds=120,
            cancel=cancel,
        )
        output = Path(installation.commands[-1].stdout_path or "")
        try:
            payload = _last_json_object(_read_tail(output, _LOCK_OUTPUT_LIMIT))
            inventory = self._inventory_from_payload(
                descriptor,
                resolution.recipe_ref,
                resolution.resolved_index_urls,
                payload,
                package_install_evidence=installation.package_install_evidence,
                worker_artifact=resolution.worker_artifact,
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
                message="the environment inventory probe returned malformed data",
                detail=str(exc),
            )
            installation = self._mark_last_command_failure(
                descriptor.env_id, installation, failure
            )
            raise EnvironmentManagerError(
                "the environment inventory probe returned malformed data",
                failure,
                installation,
            ) from exc
        return inventory, installation

    def _inspect_live_inventory(
        self,
        descriptor: EnvironmentDescriptor,
        recipe_ref: Ref,
        index_urls: Sequence[str],
        cancel: CancellationToken | None,
    ) -> InstalledEnvironmentEvidence:
        registry = self._registry_dir(descriptor.env_id)
        log_dir = registry / "logs" / "health"
        stamp = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stamp}-inventory.stdout.log"
        stderr_path = log_dir / f"{stamp}-inventory.stderr.log"
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
                "failed to start live environment inventory inspection",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="failed to start live environment inventory inspection",
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
            return self._inventory_from_payload(descriptor, recipe_ref, index_urls, payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise EnvironmentManagerError(
                "the live environment inventory probe returned malformed data",
                FailureRecord(
                    taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                    message="the live environment inventory probe returned malformed data",
                    detail=str(exc),
                ),
            ) from exc

    def _inventory_from_payload(
        self,
        descriptor: EnvironmentDescriptor,
        recipe_ref: Ref,
        index_urls: Sequence[str],
        payload: Mapping[str, object],
        *,
        package_install_evidence: Sequence[PackageInstallEvidence] = (),
        worker_artifact: WorkerArtifactIdentity | None = None,
    ) -> InstalledEnvironmentEvidence:
        raw_runtime = payload.get("runtime")
        raw_packages = payload.get("packages")
        raw_torch = payload.get("torch")
        if not isinstance(raw_runtime, dict) or not isinstance(raw_packages, list):
            raise EnvironmentManagerError("the environment inventory probe returned malformed data")
        runtime = _runtime_from_payload(raw_runtime, ">=3.11")
        evidence_by_name = {
            item.normalized_name: item for item in package_install_evidence
        }
        packages: list[PackageLock] = []
        for raw in raw_packages:
            if not isinstance(raw, dict):
                continue
            installer = str(raw.get("installer") or "") or None
            source, direct_url, artifact = _package_source(raw.get("direct_url"))
            direct, editable, vcs_repository, vcs_commit = _direct_metadata(
                raw.get("direct_url")
            )
            record_hash = str(raw.get("record_sha256") or "") or None
            name = str(raw.get("name") or "")
            normalized_name = str(raw.get("normalized_name") or "") or _normalized_package_name(
                name
            )
            install_evidence = evidence_by_name.get(normalized_name)
            if install_evidence is not None:
                source = install_evidence.source
                direct_url = install_evidence.direct_url
                artifact = install_evidence.artifact_filename
                direct = install_evidence.direct
                editable = install_evidence.editable
                vcs_repository = install_evidence.vcs_repository
                vcs_commit = install_evidence.vcs_commit
            artifact_hash = (
                install_evidence.artifact_hash if install_evidence is not None else None
            )
            if worker_artifact is not None and normalized_name == worker_artifact.normalized_name:
                artifact = worker_artifact.filename
                artifact_hash = worker_artifact.content_hash
            raw_integrity = str(raw.get("record_integrity") or "unknown")
            record_integrity = cast(
                Literal["verified", "failed", "missing", "unknown"],
                raw_integrity
                if raw_integrity in {"verified", "failed", "missing", "unknown"}
                else "unknown",
            )
            packages.append(
                PackageLock(
                    name=name,
                    normalized_name=normalized_name,
                    version=str(raw.get("version") or "") or None,
                    hash=HashRef(value=record_hash) if record_hash else None,
                    source=source,
                    source_index_url=install_evidence.source_index_url
                    if install_evidence is not None
                    else None,
                    direct_url=direct_url,
                    artifact=artifact,
                    artifact_hash=artifact_hash,
                    installer=installer,
                    requested=bool(raw.get("requested")),
                    direct=direct,
                    editable=editable,
                    vcs_repository=vcs_repository,
                    vcs_commit=vcs_commit,
                    source_evidence_reason=install_evidence.source_evidence_reason
                    if install_evidence is not None
                    else (
                        None
                        if source != "unknown"
                        else "installed metadata and pip reports did not prove the source"
                    ),
                    record_integrity=record_integrity,
                    record_entries=int(raw.get("record_entries") or 0),
                    record_verified_entries=int(raw.get("record_verified_entries") or 0),
                    record_failed_entries=sorted(
                        str(item) for item in raw.get("record_failed_entries") or []
                    ),
                    dependencies=sorted(str(item) for item in raw.get("dependencies") or []),
                )
            )
        packages.sort(key=lambda package: package.name.casefold())
        torch_data = raw_torch if isinstance(raw_torch, dict) else {}
        draft = InstalledEnvironmentEvidence(
            evidence_id="inventory-pending",
            recipe_ref=recipe_ref,
            captured_at=_timestamp(self.now),
            runtime=runtime,
            python_version=runtime.version,
            platform_tag=runtime.platform,
            architecture=runtime.architecture,
            implementation=runtime.implementation,
            torch_version=str(torch_data.get("version") or "") or None,
            torch_build=str(torch_data.get("build") or "") or None,
            cuda_runtime_version=str(torch_data.get("cuda") or "") or None,
            compute_capability=str(torch_data.get("compute_capability") or "") or None,
            index_urls=[item for value in index_urls if (item := _sanitize_url(value))],
            packages=packages,
            package_install_evidence=list(package_install_evidence),
            worker_artifact=worker_artifact,
            evidence_hash="0" * 64,
        )
        digest = _canonical_sha256(
            draft.model_dump(
                mode="json", exclude={"evidence_id", "captured_at", "evidence_hash"}
            )
        )
        return draft.model_copy(
            update={
                "evidence_id": f"inventory-{digest[:20]}",
                "evidence_hash": digest,
            }
        )

    def _lock_digest(self, lock: EnvironmentLock) -> str:
        body = lock.model_dump(
            mode="json",
            exclude={"lock_id", "created_at", "lock_hash"},
        )
        legacy_packages = all(
            not item.normalized_name
            and item.source_index_url is None
            and item.artifact_hash is None
            and item.direct is None
            and item.editable is None
            and item.vcs_repository is None
            and item.vcs_commit is None
            and item.source_evidence_reason is None
            and item.record_integrity == "unknown"
            and item.record_entries is None
            and item.record_verified_entries is None
            and not item.record_failed_entries
            for item in lock.packages
        )
        if (
            lock.resolution_ref is None
            and not lock.package_install_evidence
            and lock.worker_artifact is None
            and lock.probe_evidence is None
            and legacy_packages
        ):
            body.pop("resolution_ref", None)
            body.pop("package_install_evidence", None)
            body.pop("worker_artifact", None)
            body.pop("probe_evidence", None)
            package_fields = {
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
            }
            for package in body.get("packages", []):
                for field_name in package_fields:
                    package.pop(field_name, None)
        return _canonical_sha256(body)

    def _finalize_lock(
        self,
        inventory: InstalledEnvironmentEvidence,
        *,
        resolution: DependencyResolution,
        recipe: EnvironmentRecipe,
        installation: EnvironmentInstallation,
        probe_evidence: EnvironmentProbeEvidence | None,
    ) -> EnvironmentLock:
        """Seal a lock only after every recipe-required post-install fact is present."""
        if recipe.required_execution_probe is not None:
            if probe_evidence is None:
                raise EnvironmentManagerError(
                    "the environment lock cannot be finalized before the complete QLoRA probe passes"
                )
            if probe_evidence.required_spec != recipe.required_execution_probe:
                raise EnvironmentManagerError("probe evidence does not match the reviewed recipe tuple")
            tuple_result = probe_evidence.tuple_result
            if (
                tuple_result.outcome != FailureTaxonomy.PASS
                or recipe.required_execution_probe.execution_combination
                not in tuple_result.execution_combinations
            ):
                raise EnvironmentManagerError(
                    "independent probe results cannot replace the required complete QLoRA tuple"
                )
            if probe_evidence.evidence_hash != self._probe_evidence_digest(probe_evidence):
                raise EnvironmentManagerError("the complete QLoRA probe evidence hash is invalid")
            bad_records = [
                item.name
                for item in inventory.packages
                if item.record_integrity != "verified"
            ]
            if bad_records:
                raise EnvironmentManagerError(
                    "the environment lock cannot be finalized without verified installed RECORDs: "
                    + ", ".join(sorted(bad_records, key=str.casefold))
                )
            installed_versions = {
                item.normalized_name or _normalized_package_name(item.name): item.version
                for item in inventory.packages
            }
            version_mismatches: list[str] = []
            for requirement in recipe.dependency_requirements:
                specifier = requirement.specifier or ""
                exact = specifier.split(";", 1)[0].strip()
                if not exact.startswith("=="):
                    continue
                expected_version = exact[2:]
                observed_version = installed_versions.get(
                    _normalized_package_name(requirement.name)
                )
                if observed_version != expected_version:
                    version_mismatches.append(
                        f"{requirement.name}: expected {expected_version}, observed {observed_version}"
                    )
            if (
                recipe.bootstrap_pip_version is not None
                and installed_versions.get("pip") != recipe.bootstrap_pip_version
            ):
                version_mismatches.append(
                    "pip: expected "
                    f"{recipe.bootstrap_pip_version}, observed {installed_versions.get('pip')}"
                )
            if version_mismatches:
                raise EnvironmentManagerError(
                    "installed dependencies do not match the exact recipe: "
                    + "; ".join(version_mismatches)
                )
            if resolution.worker_artifact is None:
                raise EnvironmentManagerError("the environment lock has no worker artifact identity")
            worker_install = next(
                (
                    item
                    for item in installation.package_install_evidence
                    if item.normalized_name == resolution.worker_artifact.normalized_name
                ),
                None,
            )
            expected_worker_url = _sanitize_url(
                Path(resolution.worker_artifact.path).resolve(strict=True).as_uri()
            )
            if (
                worker_install is None
                or worker_install.source != "wheel"
                or worker_install.direct is not True
                or worker_install.direct_url != expected_worker_url
                or worker_install.artifact_filename != resolution.worker_artifact.filename
                or worker_install.artifact_hash != resolution.worker_artifact.content_hash
            ):
                raise EnvironmentManagerError(
                    "pip evidence does not prove installation of the reviewed worker wheel"
                )
            worker_package = next(
                (
                    item
                    for item in inventory.packages
                    if item.normalized_name == resolution.worker_artifact.normalized_name
                ),
                None,
            )
            if (
                worker_package is None
                or worker_package.version != resolution.worker_artifact.version
                or worker_package.artifact_hash != resolution.worker_artifact.content_hash
            ):
                raise EnvironmentManagerError(
                    "the installed CorpusStudio worker does not match the reviewed wheel identity"
                )
        draft = EnvironmentLock(
            lock_id="lock-pending",
            recipe_ref=inventory.recipe_ref,
            resolution_ref=installation.resolution_ref,
            created_at=_timestamp(self.now),
            manager_version=MANAGER_VERSION,
            runtime=inventory.runtime,
            python_version=inventory.python_version,
            platform_tag=inventory.platform_tag,
            architecture=inventory.architecture,
            implementation=inventory.implementation,
            torch_version=inventory.torch_version,
            torch_build=inventory.torch_build,
            cuda_runtime_version=inventory.cuda_runtime_version,
            compute_capability=inventory.compute_capability,
            index_urls=inventory.index_urls,
            packages=inventory.packages,
            package_install_evidence=installation.package_install_evidence,
            worker_artifact=resolution.worker_artifact,
            probe_evidence=probe_evidence,
        )
        digest = self._lock_digest(draft)
        return draft.model_copy(
            update={"lock_id": f"lock-{digest[:20]}", "lock_hash": digest}
        )

    def _probe_evidence_digest(self, evidence: EnvironmentProbeEvidence) -> str:
        return _canonical_sha256(
            evidence.model_dump(mode="json", exclude={"evidence_id", "evidence_hash"})
        )

    def _worker_artifact_drift(
        self, artifact: WorkerArtifactIdentity | None
    ) -> str | None:
        if artifact is None:
            return None
        try:
            current = _worker_artifact_identity(artifact.path)
        except (EnvironmentManagerError, OSError) as exc:
            return f"{artifact.distribution_name}: worker artifact unavailable ({exc})"
        if current != artifact:
            return f"{artifact.distribution_name}: worker artifact identity changed"
        return None

    def _run_creation_probes(
        self,
        descriptor: EnvironmentDescriptor,
        installation: EnvironmentInstallation,
        *,
        recipe: EnvironmentRecipe,
        cancel: CancellationToken | None,
    ) -> tuple[
        list[ProbeResult],
        EnvironmentState,
        EnvironmentInstallation,
        EnvironmentProbeEvidence | None,
    ]:
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
            return results, EnvironmentState.degraded, installation, None

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
            return results, EnvironmentState.degraded, installation, None
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
            return results, EnvironmentState.degraded, installation, None
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
        if recipe.required_execution_probe is None:
            return results, state, installation, None
        if state != EnvironmentState.hardware_verified:
            return results, EnvironmentState.incompatible, installation, None

        installation, capability_payload = self._json_probe(
            descriptor,
            installation,
            phase="capability_probe",
            script=_capability_snapshot_script(recipe.capability_probes),
            timeout_seconds=900,
            cancel=cancel,
        )
        try:
            capability_results, evidence = self._complete_probe_evidence(
                capability_payload, recipe.required_execution_probe
            )
        except EnvironmentManagerError as exc:
            results.append(
                ProbeResult(
                    probe=recipe.required_execution_probe.probe,
                    outcome=FailureTaxonomy.FAIL,
                    detail=str(exc),
                )
            )
            return results, EnvironmentState.incompatible, installation, None
        results.extend(capability_results)
        return results, EnvironmentState.hardware_verified, installation, evidence

    def _complete_probe_evidence(
        self,
        payload: Mapping[str, object],
        required: QloraExecutionProbeSpec,
    ) -> tuple[list[ProbeResult], EnvironmentProbeEvidence]:
        raw_profile = payload.get("profile")
        raw_report = payload.get("capability_report")
        if not isinstance(raw_profile, dict) or not isinstance(raw_report, dict):
            raise EnvironmentManagerError(
                "the complete QLoRA probe did not emit a profile and capability report"
            )
        try:
            profile = EnvironmentProfile.model_validate(raw_profile)
            report = CapabilityReport.model_validate(raw_report)
        except ValueError as exc:
            raise EnvironmentManagerError(
                f"the complete QLoRA probe emitted invalid contracts: {exc}"
            ) from exc
        if (
            report.backend_id != "corpus_studio"
            or report.environment_ref.id != profile.environment_signature
        ):
            raise EnvironmentManagerError(
                "the complete QLoRA capability report does not match its environment profile"
            )
        tuple_results = [item for item in report.probe_results if item.probe == required.probe]
        if len(tuple_results) != 1:
            raise EnvironmentManagerError(
                "the capability report must contain exactly one complete QLoRA tuple result"
            )
        tuple_result = tuple_results[0]
        if (
            tuple_result.outcome != FailureTaxonomy.PASS
            or required.execution_combination not in tuple_result.execution_combinations
        ):
            raise EnvironmentManagerError(
                "the required complete QLoRA tuple did not pass as one probe"
            )
        runtime = tuple_result.measured.get("runtime")
        package_versions = runtime.get("packages") if isinstance(runtime, dict) else None
        if not isinstance(package_versions, dict):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple did not record dependency runtime versions"
            )
        observed_names = {_normalized_package_name(str(name)) for name in package_versions}
        missing = sorted(set(required.required_distributions) - observed_names)
        if missing:
            raise EnvironmentManagerError(
                "the complete QLoRA tuple omitted runtime versions for: " + ", ".join(missing)
            )
        raw_memory = tuple_result.measured.get("memory")
        try:
            memory = ProbeMemoryEvidence.model_validate(raw_memory)
        except ValueError as exc:
            raise EnvironmentManagerError(
                f"the complete QLoRA tuple emitted invalid memory evidence: {exc}"
            ) from exc
        report_hash = _canonical_sha256(report.model_dump(mode="json"))
        draft = EnvironmentProbeEvidence(
            evidence_id="probe-evidence-pending",
            required_spec=required,
            profile_signature=profile.environment_signature,
            capability_report_hash=HashRef(value=report_hash),
            tuple_result=tuple_result,
            memory=memory,
            evidence_hash="0" * 64,
        )
        evidence_hash = self._probe_evidence_digest(draft)
        evidence = draft.model_copy(
            update={
                "evidence_id": f"probe-evidence-{evidence_hash[:20]}",
                "evidence_hash": evidence_hash,
            }
        )
        return report.probe_results, evidence

    def _health_probes(
        self,
        descriptor: EnvironmentDescriptor,
        *,
        cancel: CancellationToken | None,
    ) -> tuple[list[ProbeResult], EnvironmentState, EnvironmentProbeEvidence | None]:
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
            return results, EnvironmentState.degraded, None
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
            return results, EnvironmentState.degraded, None
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
            return results, EnvironmentState.degraded, None
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
        recipe = get_recipe(descriptor.recipe_ref.id)
        required = recipe.required_execution_probe if recipe is not None else None
        if required is None:
            return results, state, None
        assert recipe is not None
        if state != EnvironmentState.hardware_verified:
            return results, EnvironmentState.incompatible, None
        capability_payload = self._run_unjournaled_json_probe(
            descriptor,
            "capability",
            _capability_snapshot_script(recipe.capability_probes),
            900,
            cancel,
        )
        try:
            capability_results, evidence = self._complete_probe_evidence(
                capability_payload, required
            )
        except EnvironmentManagerError as exc:
            results.append(
                ProbeResult(
                    probe=required.probe,
                    outcome=FailureTaxonomy.FAIL,
                    detail=str(exc),
                )
            )
            return results, EnvironmentState.incompatible, None
        results.extend(capability_results)
        return results, EnvironmentState.hardware_verified, evidence

    def _json_probe(
        self,
        descriptor: EnvironmentDescriptor,
        installation: EnvironmentInstallation,
        *,
        phase: Literal[
            "import_probe", "functional_probe", "hardware_probe", "capability_probe"
        ],
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
        expected_by_name = {
            item.normalized_name or _normalized_package_name(item.name): item
            for item in expected
        }
        actual_by_name = {
            item.normalized_name or _normalized_package_name(item.name): item
            for item in actual
        }
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
            if after.record_integrity == "failed":
                drifted.append(f"{before.name}: installed RECORD file integrity failed")
            elif (
                before.record_integrity not in {"unknown", "missing"}
                and after.record_integrity != before.record_integrity
            ):
                drifted.append(
                    f"{before.name}: RECORD integrity {before.record_integrity} -> "
                    f"{after.record_integrity}"
                )
            if before.direct_url != after.direct_url and (
                before.direct_url is not None or after.direct_url is not None
            ):
                sources.append(f"{before.name}: direct source changed")
            elif (
                before.source != "unknown"
                and after.source != "unknown"
                and before.source != after.source
            ):
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
    descriptor_lock_hash = (
        descriptor.lock_ref.hash.value
        if descriptor.lock_ref is not None and descriptor.lock_ref.hash is not None
        else None
    )
    descriptor_lock_algo = (
        descriptor.lock_ref.hash.algo
        if descriptor.lock_ref is not None and descriptor.lock_ref.hash is not None
        else None
    )
    if (
        descriptor.lock_ref is None
        or descriptor.lock_ref.id != lock.lock_id
        or descriptor_lock_algo != "sha256"
        or descriptor_lock_hash != lock.lock_hash
    ):
        raise EnvironmentManagerError("descriptor and lock do not match")
    if descriptor.recipe_ref != lock.recipe_ref:
        raise EnvironmentManagerError("descriptor and lock recipe refs do not match")
    return Ref(id=descriptor.env_id, hash=HashRef(value=lock.lock_hash))


def verify_run_plan_environment(
    plan: RunPlan, descriptor: EnvironmentDescriptor, lock: EnvironmentLock
) -> list[str]:
    """Return resume/dispatch blockers when the plan's pinned environment no longer matches."""
    blockers: list[str] = []
    try:
        expected = locked_environment_ref(descriptor, lock)
    except EnvironmentManagerError as exc:
        blockers.append(str(exc))
        expected = Ref(
            id=descriptor.env_id,
            hash=HashRef(value=lock.lock_hash) if lock.lock_hash is not None else None,
        )
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
    recipe = get_recipe(descriptor.recipe_ref.id)
    if descriptor.layer != DependencyLayer.backend_worker:
        blockers.append("managed run environment is not a backend_worker environment")
    if recipe is None:
        blockers.append(f"managed environment recipe {descriptor.recipe_ref.id!r} is unknown")
    else:
        if recipe.layer != DependencyLayer.backend_worker:
            blockers.append(
                f"managed environment recipe {recipe.recipe_id!r} is not a backend worker recipe"
            )
        if recipe.target != plan.backend_ref.id:
            blockers.append(
                f"environment recipe target {recipe.target!r} != plan backend "
                f"{plan.backend_ref.id!r}"
            )
        current_recipe_hash = recipe_digest(recipe)
        for owner, recipe_ref in (
            ("descriptor", descriptor.recipe_ref),
            ("lock", lock.recipe_ref),
        ):
            actual = recipe_ref.hash.value if recipe_ref.hash else None
            actual_algo = recipe_ref.hash.algo if recipe_ref.hash else None
            if actual_algo != "sha256" or actual != current_recipe_hash:
                blockers.append(f"{owner} recipe hash does not match the current recipe")

    backend = get_worker_backend(plan.backend_ref.id)
    if backend is None:
        blockers.append(f"plan backend {plan.backend_ref.id!r} is not registered")
    else:
        actual_backend_hash = plan.backend_ref.hash.value if plan.backend_ref.hash else None
        actual_backend_algo = plan.backend_ref.hash.algo if plan.backend_ref.hash else None
        if (
            actual_backend_algo != "sha256"
            or actual_backend_hash != backend_manifest_digest(backend)
        ):
            blockers.append("plan backend_ref hash does not match the current backend manifest")
    return blockers
