"""Side-effectful lifecycle manager for isolated backend environments.

The manager executes only a previously sealed :class:`DependencyResolution`. A caller must echo the
resolution hash before any directory is created, which binds mutation to the exact reviewed argv,
runtime, package index, and target path. Heavy framework imports happen only in the managed Python
subprocess; importing this module remains dependency-light.

The first supported creation target is ``backend-corpus-studio``. Recipes for other backends can be
previewed, but they do not become "supported" merely because the resolver can render commands.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import BinaryIO, cast, Literal, Protocol
from urllib.parse import unquote, urlparse, urlsplit, urlunsplit
import uuid
import zipfile

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from .app_paths import corpusstudio_data_home
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

MANAGER_VERSION = "1.3.0"
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
_MAX_WORKER_WHEEL_BYTES = 256 * 1024 * 1024
_MAX_WORKER_WHEEL_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_WORKER_ARCHIVE_MEMBERS = 10_000
_MAX_DISTRIBUTION_METADATA_BYTES = 4 * 1024 * 1024
_MAX_DISTRIBUTION_RECORD_BYTES = 16 * 1024 * 1024
_DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05
_LOCKS_DIRNAME = ".locks"
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

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCK_STATE = threading.local()

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


def _validated_package_name(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise EnvironmentManagerError("package evidence contains an invalid distribution name")
    normalized = _normalized_package_name(value)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized):
        raise EnvironmentManagerError("package evidence contains an invalid distribution name")
    return value, normalized


def _sanitize_url(value: str | None) -> str | None:
    """Remove credentials, query parameters, and fragments from persisted source evidence."""
    if not value:
        return None
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ) or "\\" in value or re.search(r"%(?![0-9A-Fa-f]{2})", value):
        return None
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname
        parsed_port = parsed.port
    except (TypeError, ValueError):
        return None
    if not scheme or not re.fullmatch(r"[a-z][a-z0-9+.-]*", scheme):
        return None
    if scheme != "file" and not hostname:
        return None
    if scheme == "file" and (not parsed.path.startswith("/") or parsed_port is not None):
        return None
    hostname = hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed_port}" if parsed_port is not None else ""
    return urlunsplit((scheme, f"{hostname}{port}", parsed.path, "", ""))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_artifact_identity(path: str | Path) -> WorkerArtifactIdentity:
    candidate = Path(path).expanduser()
    try:
        if candidate.is_symlink():
            raise EnvironmentManagerError(
                "the readiness worker artifact cannot be a symbolic link"
            )
        wheel = candidate.resolve(strict=True)
    except EnvironmentManagerError:
        raise
    except (OSError, RuntimeError) as exc:
        raise EnvironmentManagerError(f"the readiness worker artifact is unavailable: {exc}") from exc
    if not wheel.is_file() or wheel.suffix.casefold() != ".whl":
        raise EnvironmentManagerError("the readiness worker artifact must be a concrete wheel")
    try:
        before = wheel.stat()
        if (
            before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_WORKER_WHEEL_BYTES
        ):
            raise EnvironmentManagerError(
                "the readiness worker artifact must be a singly linked bounded file"
            )
        with wheel.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            wheel_bytes = stream.read(_MAX_WORKER_WHEEL_BYTES + 1)
            opened_after = os.fstat(stream.fileno())
        if len(wheel_bytes) > _MAX_WORKER_WHEEL_BYTES:
            raise EnvironmentManagerError(
                "the readiness worker artifact exceeds the bounded wheel size"
            )
        after = wheel.stat()
        identities = {
            (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            for item in (before, opened_before, opened_after, after)
        }
        if len(identities) != 1:
            raise EnvironmentManagerError(
                "the readiness worker artifact changed while its identity was captured"
            )
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            names = archive.namelist()
            if len(names) > _MAX_WORKER_ARCHIVE_MEMBERS:
                raise EnvironmentManagerError("the worker wheel contains too many archive members")
            if len(names) != len(set(names)):
                raise EnvironmentManagerError("the worker wheel contains duplicate archive members")
            expanded_bytes = 0
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                raw_member = info.filename[:-1] if info.is_dir() else info.filename
                unix_mode = info.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                expanded_bytes += info.file_size
                if (
                    not raw_member
                    or member.is_absolute()
                    or ".." in member.parts
                    or "\\" in info.filename
                    or member.as_posix() != raw_member
                    or any(ord(character) < 32 or ord(character) == 127 for character in raw_member)
                    or info.flag_bits & 0x1
                    or (not info.is_dir() and file_type not in {0, stat.S_IFREG})
                ):
                    raise EnvironmentManagerError(
                        "the worker wheel contains an unsafe archive member"
                    )
            if expanded_bytes > _MAX_WORKER_WHEEL_EXPANDED_BYTES:
                raise EnvironmentManagerError(
                    "the worker wheel exceeds the bounded expanded size"
                )
            metadata_names = sorted(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            if len(metadata_names) != 1:
                raise EnvironmentManagerError(
                    "the worker wheel must contain exactly one dist-info/METADATA file"
                )
            metadata_path = PurePosixPath(metadata_names[0])
            if (
                len(metadata_path.parts) != 2
                or archive.getinfo(metadata_names[0]).file_size
                > _MAX_DISTRIBUTION_METADATA_BYTES
            ):
                raise EnvironmentManagerError("the worker wheel METADATA is not at wheel root")
            if any(
                member.parts
                and member.parts[0].endswith(".dist-info")
                and member.parts[0] != metadata_path.parent.name
                for member in (PurePosixPath(name) for name in names)
            ):
                raise EnvironmentManagerError(
                    "the worker wheel contains more than one dist-info identity"
                )
            metadata_bytes = archive.read(metadata_names[0])
            record_names = sorted(name for name in names if name.endswith(".dist-info/RECORD"))
            if len(record_names) != 1:
                raise EnvironmentManagerError(
                    "the worker wheel must contain exactly one dist-info/RECORD file"
                )
            record_path = PurePosixPath(record_names[0])
            if (
                len(record_path.parts) != 2
                or record_path.parent != metadata_path.parent
                or archive.getinfo(record_names[0]).file_size
                > _MAX_DISTRIBUTION_RECORD_BYTES
            ):
                raise EnvironmentManagerError(
                    "the worker wheel RECORD does not match its root METADATA identity"
                )
            try:
                record_rows = list(
                    csv.reader(
                        io.StringIO(
                            archive.read(record_names[0]).decode("utf-8", errors="strict")
                        )
                    )
                )
            except (UnicodeError, csv.Error) as exc:
                raise EnvironmentManagerError(f"the worker wheel RECORD is malformed: {exc}") from exc
            archive_files = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            recorded_files: set[str] = set()
            for row in record_rows:
                if len(row) != 3 or not row[0] or row[0] in recorded_files:
                    raise EnvironmentManagerError("the worker wheel RECORD contains malformed rows")
                record_member, hash_spec, size_text = row
                record_info = archive_files.get(record_member)
                if record_info is None:
                    raise EnvironmentManagerError(
                        "the worker wheel RECORD names a missing archive member"
                    )
                recorded_files.add(record_member)
                if record_member == record_names[0]:
                    if hash_spec or size_text:
                        raise EnvironmentManagerError(
                            "the worker wheel RECORD self-entry must be unhashed"
                        )
                    continue
                try:
                    algorithm, expected = hash_spec.split("=", 1)
                    expected_size = int(size_text)
                    decoded_expected = base64.urlsafe_b64decode(
                        expected + "=" * (-len(expected) % 4)
                    )
                except (ValueError, TypeError) as exc:
                    raise EnvironmentManagerError(
                        "the worker wheel RECORD contains a malformed digest or size"
                    ) from exc
                actual_bytes = archive.read(record_member)
                if (
                    algorithm != "sha256"
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", expected)
                    or len(decoded_expected) != hashlib.sha256().digest_size
                    or expected_size != len(actual_bytes)
                    or hashlib.sha256(actual_bytes).digest() != decoded_expected
                ):
                    raise EnvironmentManagerError(
                        "the worker wheel RECORD does not verify its archive bytes"
                    )
            if recorded_files != set(archive_files):
                raise EnvironmentManagerError(
                    "the worker wheel RECORD does not inventory every archive file"
                )
    except EnvironmentManagerError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
        raise EnvironmentManagerError(f"the worker wheel is unreadable: {exc}") from exc
    try:
        metadata = Parser().parsestr(metadata_bytes.decode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise EnvironmentManagerError(f"the worker wheel METADATA is invalid UTF-8: {exc}") from exc
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if len(names) != 1 or len(versions) != 1:
        raise EnvironmentManagerError(
            "the worker wheel METADATA must contain exactly one Name and Version"
        )
    name, normalized_name = _validated_package_name(str(names[0]))
    version = str(versions[0])
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version)
        or normalized_name != "corpus-studio-engine"
    ):
        raise EnvironmentManagerError(
            "the worker wheel METADATA must identify the corpus-studio-engine distribution"
        )
    filename_parts = wheel.stem.split("-")
    dist_info_identity = metadata_path.parts[0].removesuffix(".dist-info").rsplit("-", 1)
    if (
        len(filename_parts) < 5
        or _normalized_package_name(filename_parts[0]) != normalized_name
        or filename_parts[1] != version
        or len(dist_info_identity) != 2
        or _normalized_package_name(dist_info_identity[0]) != normalized_name
        or dist_info_identity[1] != version
    ):
        raise EnvironmentManagerError(
            "the worker wheel filename, dist-info directory, and METADATA identity disagree"
        )
    return WorkerArtifactIdentity(
        distribution_name=name,
        normalized_name=normalized_name,
        version=version,
        filename=wheel.name,
        path=str(wheel),
        size_bytes=len(wheel_bytes),
        content_hash=HashRef(value=hashlib.sha256(wheel_bytes).hexdigest()),
        metadata_hash=HashRef(value=hashlib.sha256(metadata_bytes).hexdigest()),
    )


def _worker_wheel_payload_manifest(
    artifact: WorkerArtifactIdentity,
) -> dict[str, str]:
    """Return immutable wheel-member SHA-256s, excluding RECORD which pip rewrites."""

    if _worker_artifact_identity(artifact.path) != artifact:
        raise EnvironmentManagerError("the reviewed worker wheel identity changed")
    wheel = Path(artifact.path)
    try:
        wheel_bytes = wheel.read_bytes()
        if (
            len(wheel_bytes) != artifact.size_bytes
            or hashlib.sha256(wheel_bytes).hexdigest() != artifact.content_hash.value
        ):
            raise EnvironmentManagerError("the reviewed worker wheel identity changed")
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            record_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
            ]
            if len(record_names) != 1:
                raise EnvironmentManagerError("the reviewed worker wheel RECORD is ambiguous")
            record_name = record_names[0]
            rows = csv.reader(
                io.StringIO(archive.read(record_name).decode("utf-8", errors="strict"))
            )
            manifest: dict[str, str] = {}
            for row in rows:
                if len(row) != 3 or not row[0] or row[0] in manifest:
                    raise EnvironmentManagerError("the reviewed worker wheel RECORD is malformed")
                if row[0] == record_name:
                    continue
                algorithm, encoded = row[1].split("=", 1)
                decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                if algorithm != "sha256" or len(decoded) != hashlib.sha256().digest_size:
                    raise EnvironmentManagerError(
                        "the reviewed worker wheel RECORD has an unsupported digest"
                    )
                manifest[row[0]] = decoded.hex()
    except EnvironmentManagerError:
        raise
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        raise EnvironmentManagerError(
            f"the reviewed worker wheel payload is unreadable: {exc}"
        ) from exc
    return manifest


def _worker_wheel_entry_point_scripts(
    artifact: WorkerArtifactIdentity,
) -> tuple[set[str], set[str]]:
    """Return reviewed console/gui script names that pip may generate outside site-packages."""

    manifest = _worker_wheel_payload_manifest(artifact)
    metadata_members = [
        PurePosixPath(path)
        for path in manifest
        if path.endswith(".dist-info/METADATA")
    ]
    if len(metadata_members) != 1:
        raise EnvironmentManagerError("the reviewed worker wheel has ambiguous metadata")
    entry_points_path = metadata_members[0].parent / "entry_points.txt"
    if entry_points_path.as_posix() not in manifest:
        return set(), set()
    try:
        wheel_bytes = Path(artifact.path).read_bytes()
        if (
            len(wheel_bytes) != artifact.size_bytes
            or hashlib.sha256(wheel_bytes).hexdigest() != artifact.content_hash.value
        ):
            raise EnvironmentManagerError("the reviewed worker wheel identity changed")
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            content = archive.read(entry_points_path.as_posix()).decode(
                "utf-8", errors="strict"
            )
    except EnvironmentManagerError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
        raise EnvironmentManagerError(
            f"the reviewed worker entry points are unreadable: {exc}"
        ) from exc

    groups: dict[str, set[str]] = {"console_scripts": set(), "gui_scripts": set()}
    section: str | None = None
    all_names: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        section_match = re.fullmatch(r"\[([A-Za-z0-9_.-]+)\]", line)
        if section_match:
            section = section_match.group(1)
            continue
        if section not in groups:
            continue
        if "=" not in line:
            raise EnvironmentManagerError(
                "the reviewed worker entry_points.txt is malformed"
            )
        name, target = (part.strip() for part in line.split("=", 1))
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or not target
            or len(target) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in target)
            or name in all_names
        ):
            raise EnvironmentManagerError(
                "the reviewed worker entry_points.txt contains an unsafe script identity"
            )
        groups[section].add(name)
        all_names.add(name)
    return groups["console_scripts"], groups["gui_scripts"]


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
    """User-owned manager state; on Windows this is the internal system drive by default.

    Derives from the shared CorpusStudio application-data base so the manager root and the default run
    output root cannot diverge (both live outside any source checkout)."""
    return corpusstudio_data_home() / "environment-manager"


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


def _bind_capability_package_integrity(
    profile: EnvironmentProfile,
    report: CapabilityReport,
    sealed_packages: Sequence[PackageLock],
) -> tuple[EnvironmentProfile, CapabilityReport]:
    """Join a managed capability snapshot to the already-verified sealed inventory.

    The dependency-light profiler intentionally records only presence and version.  A managed
    environment has stronger evidence: creation and every capability snapshot verify the live
    inventory against the immutable lock.  Do not discard that evidence before the planner seals
    the required trainer packages into a RunPlan.

    The join is deliberately limited to package identities the profile/probe already observed.  It
    does not add packages, combine independent functional probes, or change the stable environment
    signature's name/version projection.
    """

    sealed_by_name = {
        item.normalized_name or _normalized_package_name(item.name): item
        for item in sealed_packages
    }

    def bind(item: PackageLock, *, allow_missing: bool) -> PackageLock:
        normalized_name = item.normalized_name or _normalized_package_name(item.name)
        if item.version is None:
            if not allow_missing:
                raise EnvironmentManagerError(
                    "managed capability report labels an installed package as absent"
                )
            return item.model_copy(
                update={
                    "normalized_name": normalized_name,
                    "record_integrity": "missing",
                    "record_entries": 0,
                    "record_verified_entries": 0,
                    "installed_file_count": 0,
                }
            )

        sealed = sealed_by_name.get(normalized_name)
        if sealed is None or sealed.version != item.version:
            raise EnvironmentManagerError(
                "managed capability package identity does not match the sealed lock: "
                f"{normalized_name}"
            )
        if (
            not sealed.has_complete_record_count_evidence()
            or sealed.hash is None
            or sealed.hash.value is None
            or sealed.installed_files_hash is None
            or sealed.installed_files_hash.value is None
            or sealed.artifact_hash is None
            or sealed.artifact_hash.value is None
        ):
            raise EnvironmentManagerError(
                "managed capability package lacks sealed artifact, RECORD, or installed-file "
                f"integrity evidence: {normalized_name}"
            )
        return sealed

    bound_profile_packages = [bind(item, allow_missing=True) for item in profile.packages]
    if [
        (item.name, item.version) for item in bound_profile_packages
    ] != [(item.name, item.version) for item in profile.packages]:
        raise EnvironmentManagerError(
            "managed capability integrity binding changed the environment signature projection"
        )
    bound_report_packages = [
        bind(item, allow_missing=False) for item in report.installed_packages
    ]
    return (
        profile.model_copy(update={"packages": bound_profile_packages}),
        report.model_copy(update={"installed_packages": bound_report_packages}),
    )


def _lock_timeout(scope: str, operation: str) -> EnvironmentManagerError:
    message = f"managed {scope} is busy; could not start {operation} within the lock timeout"
    return EnvironmentManagerError(
        message,
        FailureRecord(
            taxonomy=FailureTaxonomy.TIMEOUT,
            message=message,
            remediation=(
                "Wait for the active managed-environment operation to finish; do not remove lock "
                "files or start a competing lifecycle command."
            ),
        ),
    )


def _process_lock_for(key: str) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _thread_held_locks() -> set[str]:
    held = getattr(_THREAD_LOCK_STATE, "held", None)
    if held is None:
        held = set()
        _THREAD_LOCK_STATE.held = held
    return cast(set[str], held)


def _open_lock_stream(path: Path) -> BinaryIO:
    """Open one owned regular lock file without following a final-component symlink."""

    lock_dir = path.parent
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        if lock_dir.is_symlink():
            raise EnvironmentManagerError("managed lock directory cannot be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
    except EnvironmentManagerError:
        raise
    except OSError as exc:
        raise EnvironmentManagerError(f"managed lock file is unavailable: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise EnvironmentManagerError(
                "managed lock file must be one singly linked regular file"
            )
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise EnvironmentManagerError("managed lock file is owned by another user")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return os.fdopen(descriptor, "r+b", buffering=0)
    except Exception:
        os.close(descriptor)
        raise


def _try_os_lock(stream: BinaryIO) -> bool:
    try:
        stream.seek(0)
        if sys.platform == "win32":
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise
        return False


def _unlock_os_lock(stream: BinaryIO) -> None:
    stream.seek(0)
    if sys.platform == "win32":
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float,
    scope: str,
    operation: str,
) -> Iterator[None]:
    """Bounded cross-process lock with same-thread reentrancy and process-local exclusion."""

    key = str(path.resolve(strict=False))
    deadline = time.monotonic() + timeout_seconds
    process_lock = _process_lock_for(key)
    if not process_lock.acquire(timeout=timeout_seconds):
        raise _lock_timeout(scope, operation)
    held = _thread_held_locks()
    if key in held:
        try:
            yield
        finally:
            process_lock.release()
        return

    stream: BinaryIO | None = None
    os_locked = False
    try:
        stream = _open_lock_stream(path)
        while not (os_locked := _try_os_lock(stream)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _lock_timeout(scope, operation)
            time.sleep(min(_LOCK_POLL_SECONDS, remaining))
        held.add(key)
        yield
    finally:
        held.discard(key)
        if stream is not None:
            if os_locked:
                try:
                    _unlock_os_lock(stream)
                except OSError:
                    # Closing the descriptor releases the OS lock even if an explicit unlock fails.
                    pass
            stream.close()
        process_lock.release()


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
import base64, csv, hashlib, importlib.metadata as metadata, importlib.util, io, json, marshal, os, platform, re, stat, struct, sys, types
from email.parser import Parser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
environment_root = Path(sys.argv[1]).resolve(strict=True)
if os.name == "nt":
    site_candidates = [environment_root / "Lib" / "site-packages"]
else:
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_candidates = [environment_root / "lib" / version_dir / "site-packages", environment_root / "lib64" / version_dir / "site-packages"]
site_roots = []
def is_linklike(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())

for candidate in site_candidates:
    if not candidate.is_dir(): continue
    cursor = Path(os.path.abspath(candidate)); symlinked_candidate = False
    while cursor != environment_root:
        if is_linklike(cursor): symlinked_candidate = True; break
        parent = cursor.parent
        if parent == cursor: raise RuntimeError("site-packages is outside the managed environment")
        cursor = parent
    if symlinked_candidate: continue
    resolved_site = candidate.resolve(strict=True)
    if environment_root not in resolved_site.parents: raise RuntimeError("site-packages escapes the managed environment")
    if str(resolved_site) not in site_roots: site_roots.append(str(resolved_site))
if not site_roots: raise RuntimeError("managed site-packages is missing")

def safe_label(value):
    return re.sub(r"[^A-Za-z0-9._/+ -]", "?", str(value))[:200]

def sanitize_url(value):
    if not isinstance(value, str) or not value: raise RuntimeError("direct URL is malformed")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value) or "\\" in value or re.search(r"%(?![0-9A-Fa-f]{2})", value):
        raise RuntimeError("direct URL is malformed")
    try:
        parsed = urlsplit(value); scheme = parsed.scheme.casefold(); hostname = parsed.hostname; parsed_port = parsed.port
    except Exception as exc: raise RuntimeError("direct URL is malformed") from exc
    if not scheme or not re.fullmatch(r"[a-z][a-z0-9+.-]*", scheme): raise RuntimeError("direct URL is malformed")
    if scheme != "file" and not hostname: raise RuntimeError("direct URL is malformed")
    if scheme == "file" and (not parsed.path.startswith("/") or parsed_port is not None): raise RuntimeError("direct URL is malformed")
    hostname = hostname or ""
    if ":" in hostname and not hostname.startswith("["): hostname = f"[{hostname}]"
    port = f":{parsed_port}" if parsed_port is not None else ""
    return urlunsplit((scheme, f"{hostname}{port}", parsed.path, "", ""))

def sanitize_direct_metadata(value):
    if value is None: return None
    if not isinstance(value, dict): raise RuntimeError("direct_url metadata is malformed")
    kinds = [key for key in ("archive_info", "dir_info", "vcs_info") if key in value]
    if len(kinds) != 1 or not isinstance(value[kinds[0]], dict): raise RuntimeError("direct_url metadata is malformed")
    sanitized = {"url": sanitize_url(value.get("url"))}
    kind = kinds[0]; details = value[kind]
    if kind == "archive_info":
        sanitized[kind] = {}
    elif kind == "dir_info":
        editable = details.get("editable")
        if editable is not None and not isinstance(editable, bool): raise RuntimeError("direct_url metadata is malformed")
        sanitized[kind] = {"editable": bool(editable)}
    else:
        vcs = details.get("vcs"); commit = details.get("commit_id")
        if not isinstance(vcs, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,31}", vcs): raise RuntimeError("direct_url VCS identity is malformed")
        if not isinstance(commit, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", commit): raise RuntimeError("direct_url VCS identity is malformed")
        sanitized[kind] = {"vcs": vcs, "commit_id": commit}
    return sanitized

def hash_regular_file(path):
    before = path.stat(); digest = hashlib.sha256(); size = 0
    with path.open("rb") as stream:
        opened_before = os.fstat(stream.fileno())
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk: break
            digest.update(chunk); size += len(chunk)
        opened_after = os.fstat(stream.fileno())
    after = path.stat()
    identities = {(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) for item in (before, opened_before, opened_after, after)}
    if (
        len(identities) != 1
        or not stat.S_ISREG(opened_after.st_mode)
        or opened_after.st_nlink != 1
    ):
        raise RuntimeError("installed file changed while it was inventoried")
    return digest.digest(), size

def read_bounded_regular_file(path, limit):
    if is_linklike(path): raise RuntimeError("installed file crosses a symbolic link or junction")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit: raise RuntimeError("installed file is not a bounded regular file")
    with path.open("rb") as stream:
        opened_before = os.fstat(stream.fileno()); content = stream.read(limit + 1); opened_after = os.fstat(stream.fileno())
    after = path.stat()
    identities = {(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) for item in (before, opened_before, opened_after, after)}
    if len(content) > limit or len(identities) != 1 or opened_after.st_nlink != 1:
        raise RuntimeError("installed file changed or exceeded its bound")
    return content

def verify_source_bytecode(path):
    match = re.fullmatch(r"(.+)\.(?:cpython|pypy)-[0-9]+(?:\.opt-([0-9]+))?\.pyc", path.name)
    if match is None or path.parent.name != "__pycache__": raise RuntimeError("worker bytecode has no canonical source identity")
    source_path = path.parent.parent / f"{match.group(1)}.py"
    cursor = Path(os.path.abspath(source_path))
    while cursor != environment_root:
        if is_linklike(cursor): raise RuntimeError("worker bytecode source crosses a symbolic link or junction")
        parent = cursor.parent
        if parent == cursor: raise RuntimeError("worker bytecode source is outside the managed environment")
        cursor = parent
    resolved_source = source_path.resolve(strict=True)
    if environment_root not in resolved_source.parents: raise RuntimeError("worker bytecode source escapes the managed environment")
    pyc_bytes = read_bounded_regular_file(path, 16 * 1024 * 1024)
    source_bytes = read_bounded_regular_file(resolved_source, 16 * 1024 * 1024)
    if len(pyc_bytes) < 16 or pyc_bytes[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(pyc_bytes[4:8], "little") not in {0, 1, 3}:
        raise RuntimeError("worker bytecode header is invalid")
    marshalled = io.BytesIO(pyc_bytes[16:])
    try: observed = marshal.load(marshalled)
    except Exception as exc: raise RuntimeError("worker bytecode payload is malformed") from exc
    if marshalled.read(1) or not isinstance(observed, types.CodeType): raise RuntimeError("worker bytecode payload is malformed")
    if not isinstance(observed.co_filename, str) or len(observed.co_filename) > 4096 or "\x00" in observed.co_filename:
        raise RuntimeError("worker bytecode filename is malformed")
    optimize = int(match.group(2) or 0)
    try: expected = compile(source_bytes, observed.co_filename, "exec", dont_inherit=True, optimize=optimize)
    except Exception as exc: raise RuntimeError("worker bytecode source cannot be compiled") from exc
    def code_signature(code):
        def constant_signature(value):
            if isinstance(value, types.CodeType):
                return code_signature(value)
            if isinstance(value, tuple):
                return ("tuple", tuple(constant_signature(item) for item in value))
            if isinstance(value, frozenset):
                return (
                    "frozenset",
                    tuple(sorted(repr(constant_signature(item)) for item in value)),
                )
            return (type(value).__name__, repr(value))
        return (
            code.co_argcount,
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            tuple(constant_signature(item) for item in code.co_consts),
            code.co_names,
            code.co_varnames,
            code.co_freevars,
            code.co_cellvars,
            getattr(code, "co_exceptiontable", b""),
        )
    if code_signature(observed) != code_signature(expected): raise RuntimeError("worker bytecode differs from reviewed source")

def read_metadata_file(distribution_path, filename, limit, required=False):
    path = distribution_path / filename
    if is_linklike(path): raise RuntimeError("distribution metadata contains a symbolic link or junction")
    try: before = path.stat()
    except FileNotFoundError:
        if required: raise RuntimeError(f"distribution metadata is missing {filename}")
        return None
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise RuntimeError(f"distribution metadata {filename} is not a bounded regular file")
    with path.open("rb") as stream:
        opened_before = os.fstat(stream.fileno()); content = stream.read(limit + 1); opened_after = os.fstat(stream.fileno())
    after = path.stat()
    identities = {(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) for item in (before, opened_before, opened_after, after)}
    if len(content) > limit or len(identities) != 1:
        raise RuntimeError(f"distribution metadata {filename} changed or exceeded its bound")
    try: return content.decode("utf-8", errors="strict")
    except UnicodeError as exc: raise RuntimeError(f"distribution metadata {filename} is not UTF-8") from exc

packages = []
owned_record_files = {}
seen_distribution_names = set()
for dist in metadata.distributions(path=site_roots):
    distribution_location = getattr(dist, "_path", None)
    if distribution_location is None: raise RuntimeError("distribution metadata has no concrete path")
    distribution_path = Path(distribution_location)
    cursor = Path(os.path.abspath(distribution_path))
    while cursor != environment_root:
        if is_linklike(cursor): raise RuntimeError("distribution metadata crosses a symbolic link or junction")
        parent = cursor.parent
        if parent == cursor: raise RuntimeError("distribution metadata is outside the managed environment")
        cursor = parent
    resolved_distribution = distribution_path.resolve(strict=True)
    if not resolved_distribution.is_dir() or not any(Path(root) in resolved_distribution.parents for root in site_roots):
        raise RuntimeError("distribution metadata escapes managed site-packages")
    metadata_text = read_metadata_file(distribution_path, "METADATA", 4 * 1024 * 1024, required=True)
    metadata_headers = Parser().parsestr(metadata_text)
    names = metadata_headers.get_all("Name", [])
    versions = metadata_headers.get_all("Version", [])
    if len(names) != 1 or len(versions) != 1:
        raise RuntimeError("distribution metadata has an ambiguous identity")
    name = str(names[0]); version = str(versions[0])
    normalized_name = re.sub(r"[-_.]+", "-", name).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized_name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
        raise RuntimeError("distribution metadata has an invalid identity")
    if normalized_name in seen_distribution_names:
        raise RuntimeError("managed site-packages contains duplicate normalized distributions")
    seen_distribution_names.add(normalized_name)
    direct_text = read_metadata_file(distribution_path, "direct_url.json", 1024 * 1024)
    direct_parse_error = False
    try:
        direct = sanitize_direct_metadata(json.loads(direct_text)) if direct_text else None
    except Exception:
        direct = None
        direct_parse_error = True
    if direct_parse_error:
        raise RuntimeError("installed direct_url metadata is malformed")
    record = read_metadata_file(distribution_path, "RECORD", 16 * 1024 * 1024)
    installer = read_metadata_file(distribution_path, "INSTALLER", 64 * 1024)
    dependencies = metadata_headers.get_all("Requires-Dist", [])
    if any(any(ord(character) < 32 and character not in "\t" for character in item) for item in dependencies):
        raise RuntimeError("distribution metadata has malformed dependencies")
    record_entries = 0
    verified_entries = 0
    failed_entries = []
    installed_files = []
    seen_record_paths = set()
    if record:
        try: rows = list(csv.reader(io.StringIO(record)))
        except Exception: rows = [] ; failed_entries.append("<malformed RECORD CSV>")
        for row in rows:
            if not row: continue
            record_entries += 1
            relative = row[0] if row else ""
            try:
                if len(row) != 3 or not relative or relative in seen_record_paths:
                    raise ValueError("malformed or duplicate RECORD row")
                seen_record_paths.add(relative)
                if any(ord(character) < 32 or ord(character) == 127 for character in relative):
                    raise ValueError("unsafe RECORD path")
                hash_spec, size_text = row[1], row[2]
                relative_path = Path(relative)
                if relative_path.is_absolute():
                    raise ValueError("absolute RECORD path")
                path = Path(dist.locate_file(relative))
                resolved = path.resolve(strict=True)
                if resolved != environment_root and environment_root not in resolved.parents:
                    raise ValueError("RECORD path escapes the managed environment")
                cursor = Path(os.path.abspath(path))
                while cursor != environment_root:
                    if is_linklike(cursor):
                        raise ValueError("RECORD path crosses a symbolic link or junction")
                    parent = cursor.parent
                    if parent == cursor:
                        raise ValueError("RECORD path is outside the managed environment")
                    cursor = parent
                if not resolved.is_file():
                    raise ValueError("RECORD entry is not a regular file")
                digest, installed_size = hash_regular_file(resolved)
                resolved_key = str(resolved)
                prior_owner = owned_record_files.get(resolved_key)
                if prior_owner is not None and prior_owner[1:] != (digest.hex(), installed_size):
                    raise ValueError("multiple RECORD entries own one installed file with different bytes")
                owned_record_files[resolved_key] = (name, digest.hex(), installed_size)
                installed_files.append((relative, digest.hex()))
                if (
                    normalized_name == "corpus-studio-engine"
                    and relative_path.name.endswith(".pyc")
                ):
                    verify_source_bytecode(resolved)
                if hash_spec:
                    algorithm, expected = hash_spec.split("=", 1)
                    if algorithm != "sha256" or not re.fullmatch(r"[A-Za-z0-9_-]+", expected):
                        raise ValueError("unsupported or malformed RECORD digest")
                    actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
                    if actual != expected or not size_text or int(size_text) != installed_size:
                        raise ValueError("RECORD digest or size mismatch")
                elif not (
                    relative.endswith(".dist-info/RECORD")
                    or ("/__pycache__/" in f"/{relative}" and relative.endswith(".pyc"))
                ):
                    raise ValueError("unexpected unhashed RECORD entry")
                # A row is verified when all path, ownership, byte hashing, and (when supplied)
                # RECORD digest/size checks above succeeded.  RECORD itself and generated pyc rows
                # are intentionally unhashed by installers, but their exact bytes are still bound by
                # installed_files_hash, so they count as completely verified rows here.
                verified_entries += 1
            except Exception:
                failed_entries.append(safe_label(relative or "<empty RECORD path>"))
    if not record:
        record_integrity = "missing"
    elif failed_entries:
        record_integrity = "failed"
    elif verified_entries == record_entries and record_entries > 0:
        record_integrity = "verified"
    else:
        record_integrity = "unknown"
    installed_files_hash = None
    if record_integrity == "verified":
        installed_files_hash = hashlib.sha256(json.dumps(sorted(installed_files), separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    packages.append({
      "name": name,
      "normalized_name": normalized_name,
      "version": version,
      "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None,
      "record_integrity": record_integrity,
      "record_count_semantics": "all_record_rows_v2" if record_integrity == "verified" else None,
      "record_entries": record_entries,
      "record_verified_entries": verified_entries,
      "record_failed_entries": sorted(set(failed_entries)),
      "installed_files_sha256": installed_files_hash,
      "installed_file_count": len(installed_files),
      "installed_file_manifest": sorted(installed_files) if normalized_name == "corpus-studio-engine" else None,
      "metadata_sha256": hashlib.sha256(metadata_text.encode("utf-8")).hexdigest() if metadata_text else None,
      "direct_url": direct,
      "direct_url_parse_error": direct_parse_error,
      "installer": installer.strip() if installer else None,
      "requested": (distribution_path / "REQUESTED").is_file(),
      "dependencies": sorted(dependencies),
    })
for site_root in site_roots:
    for current_root, directory_names, file_names in os.walk(site_root, followlinks=False):
        current = Path(current_root)
        for entry_name in [*directory_names, *file_names]:
            if is_linklike(current / entry_name):
                raise RuntimeError("managed site-packages contains a symbolic link or junction")
        for file_name in file_names:
            installed_path = (current / file_name).resolve(strict=True)
            if not installed_path.is_file() or str(installed_path) not in owned_record_files:
                raise RuntimeError("managed site-packages contains an unrecorded file")
torch_data = {"version": None, "build": None, "cuda": None, "compute_capability": None}
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
    "is_virtual_environment": environment_root != Path(sys.base_prefix).resolve(strict=True),
    "venv_available": True,
  },
  "packages": sorted(packages, key=lambda p: p["name"].lower()),
  "torch": torch_data,
}, sort_keys=True))
""".strip()


_TORCH_INVENTORY_PROBE = r"""
import json, os, sys, tempfile
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
environment_root = Path(sys.argv[1]).resolve(strict=True)
if os.name == "nt":
    candidates = [environment_root / "Lib" / "site-packages"]
else:
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [environment_root / "lib" / version_dir / "site-packages", environment_root / "lib64" / version_dir / "site-packages"]
site_roots = [candidate.resolve(strict=True) for candidate in candidates if candidate.is_dir()]
if not site_roots or any(environment_root not in root.parents for root in site_roots):
    raise RuntimeError("managed site-packages is missing or escapes the environment")
sys.path[:0] = [str(root) for root in site_roots]
result = {"version": None, "build": None, "cuda": None, "compute_capability": None}
try:
    with tempfile.TemporaryDirectory(prefix="corpusstudio-inventory-pyc-") as pycache_root:
        sys.pycache_prefix = pycache_root; sys.dont_write_bytecode = True
        torch_distribution = metadata.distribution("torch")
        torch_files = {
            torch_distribution.locate_file(item).resolve(strict=True)
            for item in (torch_distribution.files or ())
        }
        torch_spec = find_spec("torch")
        if torch_spec is None or torch_spec.origin is None:
            raise RuntimeError("installed torch has no import origin")
        expected_framework_file = Path(torch_spec.origin).resolve(strict=True)
        if expected_framework_file not in torch_files:
            raise RuntimeError("torch import origin is not owned by the torch distribution")
        import torch
        framework_file = Path(torch.__file__).resolve(strict=True)
        if framework_file != expected_framework_file or not any(root in framework_file.parents for root in site_roots):
            raise RuntimeError("torch imported outside managed site-packages")
        result["version"] = str(torch.__version__)
        result["build"] = str(getattr(torch.version, "git_version", None) or torch.__version__)
        result["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            result["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
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


def _validate_installed_direct_url(direct: object) -> None:
    """Fail closed on malformed PEP 610 metadata before classifying its source."""

    if direct is None:
        return
    if not isinstance(direct, dict):
        raise EnvironmentManagerError(
            "environment inventory contains malformed direct_url metadata"
        )
    raw_url = direct.get("url")
    if not isinstance(raw_url, str) or _sanitize_url(raw_url) is None:
        raise EnvironmentManagerError(
            "environment inventory contains a malformed direct artifact URL"
        )
    kinds = [key for key in ("archive_info", "dir_info", "vcs_info") if key in direct]
    if len(kinds) != 1 or not isinstance(direct[kinds[0]], dict):
        raise EnvironmentManagerError(
            "environment inventory direct_url must identify exactly one source kind"
        )
    subdirectory = direct.get("subdirectory")
    if subdirectory is not None and (
        not isinstance(subdirectory, str)
        or any(ord(character) < 32 for character in subdirectory)
    ):
        raise EnvironmentManagerError(
            "environment inventory direct_url contains a malformed subdirectory"
        )
    if kinds[0] == "dir_info":
        editable = direct["dir_info"].get("editable")
        if editable is not None and not isinstance(editable, bool):
            raise EnvironmentManagerError(
                "environment inventory direct_url contains a malformed editable flag"
            )
    if kinds[0] == "archive_info":
        archive_info = direct["archive_info"]
        hashes = archive_info.get("hashes")
        legacy_hash = archive_info.get("hash")
        if hashes is not None and (
            not isinstance(hashes, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in hashes.items()
            )
        ):
            raise EnvironmentManagerError(
                "environment inventory direct_url contains malformed archive hashes"
            )
        if legacy_hash is not None and not isinstance(legacy_hash, str):
            raise EnvironmentManagerError(
                "environment inventory direct_url contains a malformed archive hash"
            )
    if kinds[0] == "vcs_info":
        vcs_info = direct["vcs_info"]
        vcs = vcs_info.get("vcs")
        commit_id = vcs_info.get("commit_id")
        if (
            not isinstance(vcs, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,31}", vcs)
            or not isinstance(commit_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", commit_id)
        ):
            raise EnvironmentManagerError(
                "environment inventory direct_url contains malformed VCS identity"
            )
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision is not None and (
            not isinstance(requested_revision, str)
            or len(requested_revision) > 1024
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in requested_revision
            )
        ):
            raise EnvironmentManagerError(
                "environment inventory direct_url contains a malformed VCS revision"
            )


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
    if (
        not isinstance(payload, dict)
        or payload.get("version") != "1"
        or not isinstance(raw_installs, list)
    ):
        raise EnvironmentManagerError("pip install evidence has no install list")
    configured_indexes: list[str] = []
    for value in step.configured_index_urls:
        sanitized = _sanitize_url(value)
        if sanitized is None:
            raise EnvironmentManagerError("pip install evidence contains a malformed index URL")
        configured_indexes.append(sanitized)
    evidence: list[PackageInstallEvidence] = []
    seen_names: set[str] = set()
    for position, raw in enumerate(raw_installs):
        if not isinstance(raw, dict):
            raise EnvironmentManagerError(
                f"pip install evidence entry {position} is not an object"
            )
        metadata = raw.get("metadata")
        download = raw.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise EnvironmentManagerError(
                f"pip install evidence entry {position} lacks metadata or download_info"
            )
        name, normalized_name = _validated_package_name(metadata.get("name"))
        version_value = metadata.get("version")
        if (
            not isinstance(version_value, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version_value)
        ):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} contains an invalid version"
            )
        version = version_value
        if normalized_name in seen_names:
            raise EnvironmentManagerError(
                f"pip install evidence contains duplicate normalized distribution {normalized_name}"
            )
        seen_names.add(normalized_name)
        raw_url_value = download.get("url")
        if not isinstance(raw_url_value, str) or not raw_url_value:
            raise EnvironmentManagerError(f"pip install evidence for {name} has no artifact URL")
        raw_url = raw_url_value
        sanitized_url = _sanitize_url(raw_url)
        if sanitized_url is None:
            raise EnvironmentManagerError(
                f"pip install evidence for {name} contains a malformed artifact URL"
            )
        is_direct_value = raw.get("is_direct")
        if not isinstance(is_direct_value, bool):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} has an invalid is_direct flag"
            )
        is_direct = is_direct_value
        archive_info = download.get("archive_info")
        dir_info = download.get("dir_info")
        vcs_info = download.get("vcs_info")
        if any(
            key in download and not isinstance(download.get(key), dict)
            for key in ("archive_info", "dir_info", "vcs_info")
        ):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} contains malformed source metadata"
            )
        source_kinds = sum(
            isinstance(item, dict) for item in (archive_info, dir_info, vcs_info)
        )
        if source_kinds != 1:
            raise EnvironmentManagerError(
                f"pip install evidence for {name} must identify exactly one source kind"
            )
        if not is_direct and (isinstance(dir_info, dict) or isinstance(vcs_info, dict)):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} marks a local or VCS source as indirect"
            )
        hashes = archive_info.get("hashes") if isinstance(archive_info, dict) else None
        sha256 = str(hashes.get("sha256") or "") if isinstance(hashes, dict) else ""
        if isinstance(archive_info, dict) and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} lacks a valid artifact SHA-256"
            )
        sha256 = sha256.lower()
        artifact = (
            Path(unquote(urlparse(sanitized_url).path)).name
            if sanitized_url
            else None
        ) or None
        if artifact is not None and any(
            ord(character) < 32 or ord(character) == 127 for character in artifact
        ):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} contains an unsafe artifact filename"
            )
        requested_value = raw.get("requested")
        if requested_value is not None and not isinstance(requested_value, bool):
            raise EnvironmentManagerError(
                f"pip install evidence for {name} has an invalid requested flag"
            )
        if isinstance(dir_info, dict):
            editable_value = dir_info.get("editable")
            if editable_value is not None and not isinstance(editable_value, bool):
                raise EnvironmentManagerError(
                    f"pip install evidence for {name} has an invalid editable flag"
                )
            if urlparse(sanitized_url).scheme.casefold() != "file":
                raise EnvironmentManagerError(
                    f"pip install evidence for {name} has a non-file local source"
                )
        if isinstance(vcs_info, dict):
            vcs = vcs_info.get("vcs")
            commit_id = vcs_info.get("commit_id")
            if (
                not isinstance(vcs, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,31}", vcs)
                or not isinstance(commit_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", commit_id)
            ):
                raise EnvironmentManagerError(
                    f"pip install evidence for {name} has malformed VCS identity"
                )
            requested_revision = vcs_info.get("requested_revision")
            if requested_revision is not None and (
                not isinstance(requested_revision, str)
                or len(requested_revision) > 1024
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in requested_revision
                )
            ):
                raise EnvironmentManagerError(
                    f"pip install evidence for {name} has a malformed VCS revision"
                )
        source_index_url: str | None = None
        if isinstance(vcs_info, dict):
            source: PackageSource = "vcs"
        elif is_direct and isinstance(dir_info, dict):
            source = "local"
        elif is_direct and isinstance(archive_info, dict):
            source = "wheel" if artifact and artifact.casefold().endswith(".whl") else "sdist"
        elif configured_indexes:
            artifact_host = (urlparse(sanitized_url).hostname or "").casefold()
            matching_indexes: list[tuple[str, bool]] = []
            for index_url in configured_indexes:
                index_host = (urlparse(index_url).hostname or "").casefold()
                pypi = index_host in {"pypi.org", "www.pypi.org"}
                if artifact_host == index_host or (
                    pypi and artifact_host in {"files.pythonhosted.org", "pypi.org"}
                ):
                    matching_indexes.append((index_url, pypi))
            if len(matching_indexes) == 1:
                source_index_url, pypi = matching_indexes[0]
                source = (
                    "pypi"
                    if pypi
                    else "wheel"
                    if artifact and artifact.casefold().endswith(".whl")
                    else "sdist"
                )
            else:
                source = "unknown"
        else:
            source = "unknown"
        reason = None
        if source == "unknown":
            reason = (
                "pip report did not prove an index or direct source"
                if not configured_indexes
                else "artifact host did not identify exactly one configured index"
            )
        evidence.append(
            PackageInstallEvidence(
                normalized_name=normalized_name,
                version=version,
                source=source,
                source_index_url=source_index_url,
                direct_url=sanitized_url if is_direct else None,
                artifact_filename=artifact,
                artifact_hash=HashRef(value=sha256) if sha256 else None,
                requested=requested_value,
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


def _assert_worker_install_evidence(
    evidence: Sequence[PackageInstallEvidence], artifact: WorkerArtifactIdentity
) -> None:
    worker_installs = [
        item for item in evidence if item.normalized_name == artifact.normalized_name
    ]
    expected_url = _sanitize_url(Path(artifact.path).resolve(strict=True).as_uri())
    if (
        len(worker_installs) != 1
        or worker_installs[0].source != "wheel"
        or worker_installs[0].direct is not True
        or worker_installs[0].direct_url != expected_url
        or worker_installs[0].artifact_filename != artifact.filename
        or worker_installs[0].artifact_hash != artifact.content_hash
        or worker_installs[0].version != artifact.version
    ):
        raise EnvironmentManagerError(
            "pip evidence does not prove installation of the reviewed worker wheel"
        )


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
        lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be finite and positive")
        self.root = Path(root) if root is not None else default_manager_root()
        self.environments_root = self.root / "environments"
        self.registry_root = self.root / "registry"
        self.runner = runner or SubprocessCommandRunner()
        self.runtime_probe = runtime_probe or _default_runtime_probe
        self.now = now
        self.lock_timeout_seconds = float(lock_timeout_seconds)
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

    def _lock_path(self, name: str) -> Path:
        lock_dir = self.root / _LOCKS_DIRNAME
        manager_root = self.root.resolve(strict=False)
        resolved_dir = lock_dir.resolve(strict=False)
        if resolved_dir.parent != manager_root or resolved_dir == manager_root:
            raise EnvironmentManagerError("managed lock path escapes the manager root")
        return lock_dir / name

    @contextmanager
    def _mutation_guard(self, env_id: str, operation: str) -> Iterator[None]:
        self._validate_env_id(env_id)
        with _exclusive_file_lock(
            self._lock_path("manager.lock"),
            timeout_seconds=self.lock_timeout_seconds,
            scope="environment manager",
            operation=operation,
        ):
            with self.environment_lease(env_id, operation=operation):
                yield

    @contextmanager
    def environment_lease(self, env_id: str, *, operation: str = "environment use") -> Iterator[None]:
        """Prevent mutation of one managed environment for the full caller-owned operation."""

        self._validate_env_id(env_id)
        with _exclusive_file_lock(
            self._lock_path(f"environment-{env_id}.lock"),
            timeout_seconds=self.lock_timeout_seconds,
            scope=f"environment '{env_id}'",
            operation=operation,
        ):
            yield

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
                artifact_path = Path(worker_artifact.path)
                target_root = self.environment_root(env_id)
                if artifact_path == target_root or target_root in artifact_path.parents:
                    blocking.append(
                        "the readiness worker wheel cannot live inside the environment being replaced"
                    )
                    worker_artifact = None
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
        if resolution.environment_ref is None:
            raise EnvironmentManagerError("resolution has no environment identity")
        env_id = resolution.environment_ref.id
        # Invalid or stale authorization must remain wholly non-mutating, including lock files.
        # Validate again under the locks in _create_unlocked to close the validation/acquisition race.
        self._validate_creation(resolution, confirmed_resolution_hash)
        with self._mutation_guard(env_id, "environment creation"):
            return self._create_unlocked(
                resolution,
                confirmed_resolution_hash=confirmed_resolution_hash,
                cancel=cancel,
            )

    def _create_unlocked(
        self,
        resolution: DependencyResolution,
        *,
        confirmed_resolution_hash: str,
        cancel: CancellationToken | None = None,
        allow_recorded_env_id: bool = False,
    ) -> EnvironmentCreationResult:
        """Execute one sealed reference-backend plan and persist every transition/evidence record."""
        recipe, env_id, env_root = self._validate_creation(
            resolution, confirmed_resolution_hash
        )
        resolution_hash = resolution.resolution_hash
        assert resolution_hash is not None  # narrowed by _validate_creation
        if env_root.exists():
            raise EnvironmentManagerError(
                f"environment '{env_id}' already exists; inspect it, create sealed replacements "
                "under a new environment id, or use env-recreate only for an unsealed failed "
                "attempt"
            )
        if not allow_recorded_env_id and self._registry_dir(env_id).exists():
            raise EnvironmentManagerError(
                f"environment identity '{env_id}' is already recorded; create a replacement "
                "under a new environment id, or use env-recreate only for an unsealed failed "
                "attempt"
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
                if (
                    resolution.worker_artifact is not None
                    and resolution.worker_artifact.path in step.argv
                    and _worker_artifact_identity(resolution.worker_artifact.path)
                    != resolution.worker_artifact
                ):
                    raise EnvironmentManagerError(
                        "the CorpusStudio worker wheel changed immediately before installation"
                    )
                installation = self._execute_step(
                    env_id,
                    installation,
                    step,
                    cancel=cancel,
                    require_evidence=recipe.requires_worker_wheel,
                )
                if (
                    resolution.worker_artifact is not None
                    and resolution.worker_artifact.path in step.argv
                ):
                    # Compare pip's hash before launching any newly installed interpreter process.
                    # A wheel swapped after the pre-step stat check can never reach an import probe.
                    _assert_worker_install_evidence(
                        installation.package_install_evidence,
                        resolution.worker_artifact,
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
            if recipe.requires_worker_wheel:
                unverified = [
                    item.name
                    for item in pre_probe.packages
                    if not item.has_complete_record_count_evidence()
                    or item.installed_files_hash is None
                ]
                if unverified:
                    raise EnvironmentManagerError(
                        "installed package bytes must verify before readiness probes may import them: "
                        + ", ".join(sorted(unverified, key=str.casefold))
                    )
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
                    update={
                        "state": final_state,
                        "finished_at": finished,
                        "retry_requires_recreate": True,
                    }
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
                    update={
                        "state": final_state,
                        "finished_at": finished,
                        "retry_requires_recreate": True,
                    }
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
                    remediation=self._recovery_remediation(
                        sealed=False,
                        problem="Probing changed the unsealed environment",
                    ),
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
                descriptor,
                lock.recipe_ref,
                lock.index_urls,
                cancel,
                worker_artifact=lock.worker_artifact,
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
                remediation=(
                    self._recovery_remediation(
                        sealed=True,
                        problem="The sealed post-probe inventory drifted",
                    )
                    if final_state == EnvironmentState.drifted
                    else self._probe_remediation(final_state)
                ),
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
                remediation=self._recovery_remediation(
                    sealed=descriptor.lock_ref is not None,
                    problem="Inspect the command stderr and installation journal",
                ),
            )
            self._write_health(env_id, health)
            if isinstance(exc, EnvironmentManagerError):
                raise
            raise EnvironmentManagerError(failure.message, failure) from exc

    def health(
        self, env_id: str, *, cancel: CancellationToken | None = None
    ) -> EnvironmentHealthReport:
        with self.environment_lease(env_id, operation="environment health probe"):
            return self._health_unlocked(env_id, cancel=cancel)

    def _health_unlocked(
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
                remediation=self._recovery_remediation(
                    sealed=descriptor.lock_ref is not None,
                    problem="The managed environment root is missing",
                )
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
                remediation=self._recovery_remediation(
                    sealed=descriptor.lock_ref is not None,
                    problem="The managed interpreter is missing; do not repair arbitrary files in place",
                ),
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
                remediation=self._recovery_remediation(
                    sealed=descriptor.lock_ref is not None,
                    problem="The recorded lock is unavailable",
                ),
            )
            self._write_health(env_id, report)
            self._update_descriptor(descriptor, state=EnvironmentState.broken)
            return report
        descriptor_lock_hash = (
            descriptor.lock_ref.hash.value
            if descriptor.lock_ref is not None and descriptor.lock_ref.hash is not None
            else None
        )
        lock_identity_mismatch = (
            lock.lock_hash != self._lock_digest(lock)
            or descriptor_lock_hash != lock.lock_hash
        )
        legacy_record_counts = sorted(
            item.name
            for item in lock.packages
            if item.version is not None and not item.has_complete_record_count_evidence()
        )
        if legacy_record_counts:
            # The lock is truthful historical evidence, not a corrupt lock and not manager-1.3
            # admission evidence. Refuse before importing installed code, and deliberately do not
            # rewrite the descriptor or historical health record under a different classification.
            if lock_identity_mismatch:
                return EnvironmentHealthReport(
                    environment_ref=Ref(id=env_id),
                    recipe_ref=descriptor.recipe_ref,
                    lock_ref=descriptor.lock_ref,
                    state=EnvironmentState.broken,
                    checked_at=checked_at,
                    lock_mismatch=True,
                    failure=FailureRecord(
                        taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                        message="historical environment lock identity verification failed",
                    ),
                    remediation=(
                        "Preserve the environment and registry evidence; reconstruct the lock and "
                        "descriptor identities before any use."
                    ),
                )
            return EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.degraded,
                checked_at=checked_at,
                failure=FailureRecord(
                    taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                    message=(
                        "historical environment lock predates complete all-row RECORD count evidence"
                    ),
                    detail=(
                        "packages requiring a replacement lock: "
                        + ", ".join(legacy_record_counts)
                    ),
                ),
                remediation=(
                    "Preserve this environment and its evidence unchanged; create a replacement "
                    "under a new environment ID before new health, planning, or execution."
                ),
            )
        try:
            live_inventory = self._inspect_live_inventory(
                descriptor,
                lock.recipe_ref,
                lock.index_urls,
                cancel,
                worker_artifact=lock.worker_artifact,
            )
        except EnvironmentManagerError as exc:
            report = EnvironmentHealthReport(
                environment_ref=Ref(id=env_id),
                recipe_ref=descriptor.recipe_ref,
                lock_ref=descriptor.lock_ref,
                state=EnvironmentState.broken,
                checked_at=checked_at,
                failure=exc.failure,
                remediation=self._recovery_remediation(
                    sealed=True,
                    problem="Inspect the health-probe logs; the sealed interpreter is broken",
                ),
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
        lock_mismatch = lock_identity_mismatch
        if recipe is not None and self._locked_probe_evidence_mismatch(
            lock, recipe.required_execution_probe
        ):
            lock_mismatch = True
        cuda_mismatch = lock.cuda_runtime_version != live_inventory.cuda_runtime_version
        hardware_mismatch = lock.compute_capability != live_inventory.compute_capability
        drift = bool(
            drifted
            or changed_sources
            or recipe_drift
            or lock_mismatch
            or cuda_mismatch
            or hardware_mismatch
        )
        probe_results: list[ProbeResult] = []
        probe_state = descriptor.state
        probe_evidence: EnvironmentProbeEvidence | None = None
        if not drift:
            # Never import a package set that already differs from its lock. After clean probes, take
            # a second isolated (-I -S) inventory so probe-side mutation cannot be reported healthy.
            probe_results, probe_state, probe_evidence = self._health_probes(
                descriptor,
                lock=lock,
                cancel=cancel,
            )
            try:
                post_probe_inventory = self._inspect_live_inventory(
                    descriptor,
                    lock.recipe_ref,
                    lock.index_urls,
                    cancel,
                    worker_artifact=lock.worker_artifact,
                )
            except EnvironmentManagerError as exc:
                report = EnvironmentHealthReport(
                    environment_ref=Ref(id=env_id),
                    recipe_ref=descriptor.recipe_ref,
                    lock_ref=descriptor.lock_ref,
                    state=EnvironmentState.broken,
                    checked_at=checked_at,
                    probe_results=probe_results,
                    probe_evidence=probe_evidence,
                    failure=exc.failure,
                    remediation=self._recovery_remediation(
                        sealed=True,
                        problem="The post-probe inventory failed; inspect the health logs",
                    ),
                )
                self._write_health(env_id, report)
                self._update_descriptor(descriptor, state=EnvironmentState.broken)
                return report
            post_drifted, post_sources = self._package_drift(
                lock.packages, post_probe_inventory.packages
            )
            post_worker_drift = self._worker_artifact_drift(lock.worker_artifact)
            if post_worker_drift:
                post_drifted.append(post_worker_drift)
            drifted = sorted(set(drifted + post_drifted))
            changed_sources = sorted(set(changed_sources + post_sources))
            live_inventory = post_probe_inventory
            cuda_mismatch = lock.cuda_runtime_version != live_inventory.cuda_runtime_version
            hardware_mismatch = lock.compute_capability != live_inventory.compute_capability
            try:
                post_probe_lock = self.load_lock(env_id)
            except EnvironmentManagerError:
                lock_mismatch = True
            else:
                lock_mismatch = lock_mismatch or post_probe_lock != lock
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
            remediation=self._recovery_remediation(
                sealed=True,
                problem="The live environment no longer matches its lock",
            )
            if drift
            else self._probe_remediation(
                probe_state,
                sealed=descriptor.lock_ref is not None,
            ),
        )
        self._write_health(env_id, report)
        if descriptor.state != state:
            self._update_descriptor(descriptor, state=state)
        return report

    def list_descriptors(self) -> list[EnvironmentDescriptor]:
        with _exclusive_file_lock(
            self._lock_path("manager.lock"),
            timeout_seconds=self.lock_timeout_seconds,
            scope="environment manager",
            operation="environment listing",
        ):
            return self._list_descriptors_unlocked()

    def _list_descriptors_unlocked(self) -> list[EnvironmentDescriptor]:
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
        with self.environment_lease(env_id, operation="capability snapshot"):
            return self._capability_snapshot_unlocked(env_id, cancel=cancel)

    def _capability_snapshot_unlocked(
        self, env_id: str, *, cancel: CancellationToken | None = None
    ) -> tuple[EnvironmentProfile, CapabilityReport]:
        """Profile and prove capabilities inside the managed interpreter, never the control plane."""
        health = self._health_unlocked(env_id, cancel=cancel)
        if health.state not in {
            EnvironmentState.functional_probe_passed,
            EnvironmentState.hardware_verified,
        }:
            raise EnvironmentManagerError(
                f"managed environment '{env_id}' is {health.state.value}, not functionally verified"
            )
        descriptor = self.load_descriptor(env_id)
        lock = self.load_lock(env_id)
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
        post_probe_inventory = self._inspect_live_inventory(
            descriptor,
            lock.recipe_ref,
            lock.index_urls,
            cancel,
            worker_artifact=lock.worker_artifact,
        )
        drifted, changed_sources = self._package_drift(
            lock.packages, post_probe_inventory.packages
        )
        worker_drift = self._worker_artifact_drift(lock.worker_artifact)
        if worker_drift:
            drifted.append(worker_drift)
        if drifted or changed_sources or self.load_lock(env_id) != lock:
            raise EnvironmentManagerError(
                "managed capability probing changed the sealed environment; create its "
                "replacement under a new environment id"
            )
        return _bind_capability_package_integrity(profile, report, lock.packages)

    def load_descriptor(self, env_id: str) -> EnvironmentDescriptor:
        with self.environment_lease(env_id, operation="descriptor read"):
            return self._load_descriptor_unlocked(env_id)

    def _load_descriptor_unlocked(self, env_id: str) -> EnvironmentDescriptor:
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
        with self.environment_lease(env_id, operation="lock read"):
            return self._load_lock_unlocked(env_id)

    def _load_lock_unlocked(self, env_id: str) -> EnvironmentLock:
        descriptor = self._load_descriptor_unlocked(env_id)
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
        with self.environment_lease(env_id, operation="health record read"):
            return self._load_health_unlocked(env_id)

    def _load_health_unlocked(self, env_id: str) -> EnvironmentHealthReport:
        path = self._registry_dir(env_id) / _HEALTH_FILENAME
        try:
            return EnvironmentHealthReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EnvironmentManagerError(
                f"environment health for '{env_id}' is missing or unreadable: {exc}"
            ) from exc

    def remove(self, env_id: str, *, confirmed_env_id: str) -> EnvironmentDescriptor:
        if confirmed_env_id != env_id:
            raise EnvironmentManagerError(
                f"removal requires the exact environment id confirmation '{env_id}'"
            )
        with self._mutation_guard(env_id, "environment removal"):
            return self._remove_unlocked(env_id, confirmed_env_id=confirmed_env_id)

    def _remove_unlocked(
        self, env_id: str, *, confirmed_env_id: str
    ) -> EnvironmentDescriptor:
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
        env_id = resolution.environment_ref.id
        if confirmed_remove_env_id != env_id:
            raise EnvironmentManagerError(
                f"removal requires the exact environment id confirmation '{env_id}'"
            )
        # Preserve the current environment and avoid even lock-file mutation for an invalid plan.
        self._validate_creation(resolution, confirmed_resolution_hash)
        with self._mutation_guard(env_id, "environment recreation"):
            # Validate the complete new seal before destructive removal. A sealed identity is
            # immutable even after explicit removal; its replacement must use a new ID so the old
            # lock remains an unambiguous rollback/evidence identity.
            self._validate_creation(resolution, confirmed_resolution_hash)
            descriptor = self._load_descriptor_unlocked(env_id)
            if descriptor.lock_ref is not None:
                raise EnvironmentManagerError(
                    f"refusing in-place recreation of sealed environment '{env_id}'; create the "
                    "replacement under a new environment id and preserve this lock identity"
                )
            self._remove_unlocked(
                env_id,
                confirmed_env_id=confirmed_remove_env_id,
            )
            return self._create_unlocked(
                resolution,
                confirmed_resolution_hash=confirmed_resolution_hash,
                cancel=cancel,
                allow_recorded_env_id=True,
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
        collisions = sorted(
            item.normalized_name for item in captured if item.normalized_name in by_name
        )
        if collisions:
            failure = FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message="pip install evidence repeats a distribution across install steps",
                detail=", ".join(collisions),
            )
            installation = self._mark_last_command_failure(env_id, installation, failure)
            raise EnvironmentManagerError(failure.message, failure, installation)
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
            argv=[
                descriptor.python_executable,
                "-I",
                "-S",
                "-c",
                _LOCK_PROBE,
                descriptor.root_path,
            ],
            cwd=Path(descriptor.root_path),
            environment=_command_environment(),
            timeout_seconds=120,
            cancel=cancel,
        )
        inventory_output = Path(installation.commands[-1].stdout_path or "")
        try:
            payload = _last_json_object(
                _read_tail(inventory_output, _LOCK_OUTPUT_LIMIT)
            )
            # Validate all non-executable metadata and worker bytes before importing torch from the
            # newly installed environment in a separate process.
            self._inventory_from_payload(
                descriptor,
                resolution.recipe_ref,
                resolution.resolved_index_urls,
                payload,
                package_install_evidence=installation.package_install_evidence,
                worker_artifact=resolution.worker_artifact,
                require_framework_identity=False,
            )
            installation = self._execute_recorded(
                descriptor.env_id,
                installation,
                phase="inventory",
                argv=[
                    descriptor.python_executable,
                    "-I",
                    "-S",
                    "-c",
                    _TORCH_INVENTORY_PROBE,
                    descriptor.root_path,
                ],
                cwd=Path(descriptor.root_path),
                environment=_command_environment(),
                timeout_seconds=120,
                cancel=cancel,
            )
            framework_output = Path(installation.commands[-1].stdout_path or "")
            framework_payload = _last_json_object(_read_tail(framework_output))
            payload = dict(payload)
            payload["torch"] = framework_payload
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
        *,
        worker_artifact: WorkerArtifactIdentity | None = None,
    ) -> InstalledEnvironmentEvidence:
        registry = self._registry_dir(descriptor.env_id)
        verifiable_worker_artifact = worker_artifact
        if self._worker_artifact_drift(worker_artifact) is not None:
            # Preserve package/file drift inspection when the external authorization wheel moved or
            # disappeared. The caller records that worker-artifact drift separately.
            verifiable_worker_artifact = None
        log_dir = registry / "logs" / "health"
        stamp = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stamp}-inventory.stdout.log"
        stderr_path = log_dir / f"{stamp}-inventory.stderr.log"
        try:
            outcome = self.runner(
                [
                    descriptor.python_executable,
                    "-I",
                    "-S",
                    "-c",
                    _LOCK_PROBE,
                    descriptor.root_path,
                ],
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
            self._inventory_from_payload(
                descriptor,
                recipe_ref,
                index_urls,
                payload,
                worker_artifact=verifiable_worker_artifact,
                require_framework_identity=False,
            )
            framework_stdout = log_dir / f"{stamp}-framework.stdout.log"
            framework_stderr = log_dir / f"{stamp}-framework.stderr.log"
            try:
                framework_outcome = self.runner(
                    [
                        descriptor.python_executable,
                        "-I",
                        "-S",
                        "-c",
                        _TORCH_INVENTORY_PROBE,
                        descriptor.root_path,
                    ],
                    cwd=Path(descriptor.root_path),
                    environment=_command_environment(),
                    timeout_seconds=120,
                    stdout_path=framework_stdout,
                    stderr_path=framework_stderr,
                    cancel=cancel,
                )
            except Exception as exc:
                raise EnvironmentManagerError(
                    "failed to start verified torch runtime inspection",
                    FailureRecord(
                        taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                        message="failed to start verified torch runtime inspection",
                        detail=str(exc),
                        exception_type=type(exc).__name__,
                    ),
                ) from exc
            if (
                framework_outcome.exit_code != 0
                or framework_outcome.timed_out
                or framework_outcome.cancelled
            ):
                raise EnvironmentManagerError(
                    "failed to inspect the verified torch runtime",
                    FailureRecord(
                        taxonomy=FailureTaxonomy.TIMEOUT
                        if framework_outcome.timed_out
                        else FailureTaxonomy.ENVIRONMENT_FAILURE,
                        exit_code=framework_outcome.exit_code,
                        message="failed to inspect the verified torch runtime",
                        detail=_read_tail(framework_stderr) or None,
                    ),
                )
            framework_payload = _last_json_object(_read_tail(framework_stdout))
            payload = dict(payload)
            payload["torch"] = framework_payload
            return self._inventory_from_payload(
                descriptor,
                recipe_ref,
                index_urls,
                payload,
                worker_artifact=verifiable_worker_artifact,
            )
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
        require_framework_identity: bool = True,
    ) -> InstalledEnvironmentEvidence:
        raw_runtime = payload.get("runtime")
        raw_packages = payload.get("packages")
        raw_torch = payload.get("torch")
        if (
            not isinstance(raw_runtime, dict)
            or not isinstance(raw_packages, list)
            or not raw_packages
            or not isinstance(raw_torch, dict)
        ):
            raise EnvironmentManagerError("the environment inventory probe returned malformed data")
        runtime = _runtime_from_payload(raw_runtime, ">=3.11")
        evidence_by_name: dict[str, PackageInstallEvidence] = {}
        for item in package_install_evidence:
            if item.normalized_name in evidence_by_name:
                raise EnvironmentManagerError(
                    "package install evidence contains duplicate normalized distribution "
                    f"{item.normalized_name}"
                )
            evidence_by_name[item.normalized_name] = item
        packages: list[PackageLock] = []
        seen_names: set[str] = set()
        for position, raw in enumerate(raw_packages):
            if not isinstance(raw, dict):
                raise EnvironmentManagerError(
                    f"environment inventory package entry {position} is not an object"
                )
            raw_installer = raw.get("installer")
            if raw_installer is not None and (
                not isinstance(raw_installer, str)
                or any(ord(character) < 32 for character in raw_installer)
            ):
                raise EnvironmentManagerError(
                    "environment inventory contains malformed installer metadata"
                )
            installer = raw_installer or None
            direct_payload = raw.get("direct_url")
            if raw.get("direct_url_parse_error") is not False:
                raise EnvironmentManagerError(
                    "environment inventory could not parse installed direct_url metadata"
                )
            _validate_installed_direct_url(direct_payload)
            source, direct_url, artifact = _package_source(direct_payload)
            direct, editable, vcs_repository, vcs_commit = _direct_metadata(direct_payload)
            raw_record_hash = raw.get("record_sha256")
            if raw_record_hash is not None and (
                not isinstance(raw_record_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", raw_record_hash)
            ):
                raise EnvironmentManagerError(
                    "environment inventory contains a malformed RECORD SHA-256"
                )
            record_hash = raw_record_hash
            raw_installed_hash = raw.get("installed_files_sha256")
            if raw_installed_hash is not None and (
                not isinstance(raw_installed_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", raw_installed_hash)
            ):
                raise EnvironmentManagerError(
                    "environment inventory contains a malformed installed-files SHA-256"
                )
            name, normalized_name = _validated_package_name(raw.get("name"))
            emitted_normalized = raw.get("normalized_name")
            if not isinstance(emitted_normalized, (str, type(None))) or (
                emitted_normalized not in {None, "", normalized_name}
            ):
                raise EnvironmentManagerError(
                    f"environment inventory normalized name disagrees for {name}"
                )
            if normalized_name in seen_names:
                raise EnvironmentManagerError(
                    "environment inventory contains duplicate normalized distribution "
                    f"{normalized_name}"
                )
            seen_names.add(normalized_name)
            raw_version = raw.get("version")
            if (
                not isinstance(raw_version, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", raw_version)
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains an invalid version for {name}"
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
                observed_metadata_hash = raw.get("metadata_sha256")
                if (
                    worker_artifact.metadata_hash is None
                    or not isinstance(observed_metadata_hash, str)
                    or observed_metadata_hash != worker_artifact.metadata_hash.value
                ):
                    raise EnvironmentManagerError(
                        "installed CorpusStudio worker METADATA differs from the reviewed wheel"
                    )
                raw_manifest = raw.get("installed_file_manifest")
                if not isinstance(raw_manifest, list):
                    raise EnvironmentManagerError(
                        "installed CorpusStudio worker omitted its file manifest"
                    )
                installed_manifest: dict[str, str] = {}
                for entry in raw_manifest:
                    if (
                        not isinstance(entry, list)
                        or len(entry) != 2
                        or not isinstance(entry[0], str)
                        or not isinstance(entry[1], str)
                        or entry[0] in installed_manifest
                        or not re.fullmatch(r"[0-9a-f]{64}", entry[1])
                    ):
                        raise EnvironmentManagerError(
                            "installed CorpusStudio worker emitted a malformed file manifest"
                        )
                    installed_manifest[entry[0]] = entry[1]
                expected_manifest = _worker_wheel_payload_manifest(worker_artifact)
                console_scripts, gui_scripts = _worker_wheel_entry_point_scripts(
                    worker_artifact
                )
                mismatched_worker_files = sorted(
                    path
                    for path, expected_hash in expected_manifest.items()
                    if installed_manifest.get(path) != expected_hash
                )
                if mismatched_worker_files:
                    raise EnvironmentManagerError(
                        "installed CorpusStudio worker files differ from the reviewed wheel: "
                        + ", ".join(mismatched_worker_files[:5])
                    )
                metadata_members = [
                    PurePosixPath(path)
                    for path in expected_manifest
                    if path.endswith(".dist-info/METADATA")
                ]
                if len(metadata_members) != 1:
                    raise EnvironmentManagerError(
                        "reviewed CorpusStudio worker has ambiguous metadata"
                    )
                dist_info = metadata_members[0].parent

                if runtime.os == OperatingSystem.windows:
                    generated_script_directory = "scripts"
                    casefold_generated_scripts = True
                    generated_script_names = {
                        *(f"{name}.exe".casefold() for name in console_scripts | gui_scripts),
                        *(f"{name}-script.py".casefold() for name in console_scripts),
                        *(f"{name}-script.pyw".casefold() for name in gui_scripts),
                    }
                else:
                    generated_script_directory = "bin"
                    casefold_generated_scripts = False
                    generated_script_names = console_scripts | gui_scripts

                def _allowed_generated_worker_file(path: str) -> bool:
                    member = PurePosixPath(path)
                    if (
                        len(member.parts) >= 2
                        and (
                            member.parts[-2].casefold()
                            if casefold_generated_scripts
                            else member.parts[-2]
                        )
                        == generated_script_directory
                        and member.parts[:-2]
                        and all(part == ".." for part in member.parts[:-2])
                        and (
                            member.name.casefold()
                            if casefold_generated_scripts
                            else member.name
                        )
                        in generated_script_names
                    ):
                        return True
                    if member.parent == dist_info and member.name in {
                        "RECORD",
                        "INSTALLER",
                        "REQUESTED",
                        "direct_url.json",
                    }:
                        return True
                    pyc_match = re.fullmatch(
                        r"(.+)\.(?:cpython|pypy)-[0-9]+(?:\.opt-[0-9]+)?\.pyc",
                        member.name,
                    )
                    if pyc_match and member.parent.name == "__pycache__":
                        source = member.parent.parent / f"{pyc_match.group(1)}.py"
                        return source.as_posix() in expected_manifest
                    return False

                unexpected_worker_files = sorted(
                    path
                    for path in installed_manifest
                    if path not in expected_manifest
                    and not _allowed_generated_worker_file(path)
                )
                if unexpected_worker_files:
                    raise EnvironmentManagerError(
                        "installed CorpusStudio worker contains files absent from the reviewed wheel: "
                        + ", ".join(unexpected_worker_files[:5])
                    )
            raw_metadata_hash = raw.get("metadata_sha256")
            if raw_metadata_hash is not None and (
                not isinstance(raw_metadata_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", raw_metadata_hash)
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains a malformed METADATA SHA-256 for {name}"
                )
            raw_integrity = str(raw.get("record_integrity") or "unknown")
            if raw_integrity not in {"verified", "failed", "missing", "unknown"}:
                raise EnvironmentManagerError(
                    f"environment inventory contains invalid RECORD status for {name}"
                )
            record_integrity = cast(
                Literal["verified", "failed", "missing", "unknown"],
                raw_integrity,
            )
            raw_count_semantics = raw.get("record_count_semantics")
            if raw_count_semantics is not None and not isinstance(
                raw_count_semantics, str
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains invalid RECORD count semantics for {name}"
                )
            if raw_count_semantics not in (None, "all_record_rows_v2"):
                raise EnvironmentManagerError(
                    f"environment inventory contains invalid RECORD count semantics for {name}"
                )
            if record_integrity == "verified" and raw_count_semantics != "all_record_rows_v2":
                raise EnvironmentManagerError(
                    f"environment inventory omitted complete RECORD count semantics for {name}"
                )
            if record_integrity != "verified" and raw_count_semantics is not None:
                raise EnvironmentManagerError(
                    f"environment inventory attached complete counts to unverified package {name}"
                )
            raw_failed_entries = raw.get("record_failed_entries") or []
            raw_dependencies = raw.get("dependencies") or []
            if not isinstance(raw_failed_entries, list) or not isinstance(
                raw_dependencies, list
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains malformed lists for {name}"
                )
            if any(
                not isinstance(item, str) or any(ord(character) < 32 for character in item)
                for item in [*raw_failed_entries, *raw_dependencies]
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains malformed list values for {name}"
                )
            count_values: dict[str, int] = {}
            for field_name in (
                "record_entries",
                "record_verified_entries",
                "installed_file_count",
            ):
                value = raw.get(field_name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise EnvironmentManagerError(
                        f"environment inventory contains an invalid {field_name} for {name}"
                    )
                count_values[field_name] = value
            if count_values["record_verified_entries"] > count_values["record_entries"]:
                raise EnvironmentManagerError(
                    f"environment inventory RECORD counts disagree for {name}"
                )
            if record_integrity == "verified" and (
                record_hash is None
                or raw_installed_hash is None
                or count_values["record_entries"] <= 0
                or count_values["record_verified_entries"] != count_values["record_entries"]
                or count_values["installed_file_count"] != count_values["record_entries"]
                or raw_failed_entries
            ):
                raise EnvironmentManagerError(
                    f"environment inventory lacks complete verified-file evidence for {name}"
                )
            raw_requested = raw.get("requested")
            if not isinstance(raw_requested, bool):
                raise EnvironmentManagerError(
                    f"environment inventory contains an invalid requested flag for {name}"
                )
            packages.append(
                PackageLock(
                    name=name,
                    normalized_name=normalized_name,
                    version=raw_version,
                    hash=HashRef(value=record_hash) if record_hash else None,
                    source=source,
                    source_index_url=install_evidence.source_index_url
                    if install_evidence is not None
                    else None,
                    direct_url=direct_url,
                    artifact=artifact,
                    artifact_hash=artifact_hash,
                    installer=installer,
                    requested=raw_requested,
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
                    record_count_semantics=(
                        "all_record_rows_v2" if raw_count_semantics is not None else None
                    ),
                    record_entries=count_values["record_entries"],
                    record_verified_entries=count_values["record_verified_entries"],
                    record_failed_entries=sorted(raw_failed_entries),
                    installed_files_hash=HashRef(value=raw_installed_hash)
                    if raw_installed_hash
                    else None,
                    installed_file_count=count_values["installed_file_count"],
                    dependencies=sorted(raw_dependencies),
                )
            )
        missing_installs = sorted(set(evidence_by_name) - seen_names)
        if missing_installs:
            raise EnvironmentManagerError(
                "package install evidence names distributions absent from the inventory: "
                + ", ".join(missing_installs)
            )
        version_mismatches = [
            item.normalized_name
            for item in packages
            if (evidence := evidence_by_name.get(item.normalized_name)) is not None
            and item.version != evidence.version
        ]
        if version_mismatches:
            raise EnvironmentManagerError(
                "package install evidence versions disagree with the inventory: "
                + ", ".join(sorted(version_mismatches))
            )
        packages.sort(key=lambda package: package.name.casefold())
        torch_data = raw_torch
        if require_framework_identity and (
            torch_data.get("error") is not None
            or not isinstance(torch_data.get("version"), str)
            or not torch_data.get("version")
            or not isinstance(torch_data.get("build"), str)
            or not torch_data.get("build")
        ):
            raise EnvironmentManagerError(
                "environment inventory could not verify the installed torch runtime"
            )
        for field_name in ("version", "build", "cuda", "compute_capability"):
            value = torch_data.get(field_name)
            if value is not None and (
                not isinstance(value, str)
                or any(ord(character) < 32 for character in value)
            ):
                raise EnvironmentManagerError(
                    f"environment inventory contains malformed torch {field_name} metadata"
                )
        if require_framework_identity:
            torch_packages = [
                item for item in packages if item.normalized_name == "torch"
            ]
            if (
                len(torch_packages) != 1
                or torch_data.get("version") != torch_packages[0].version
            ):
                raise EnvironmentManagerError(
                    "imported torch identity disagrees with installed distribution metadata"
                )
        sanitized_indexes: list[str] = []
        for value in index_urls:
            sanitized = _sanitize_url(value)
            if sanitized is None:
                raise EnvironmentManagerError("environment inventory contains a malformed index URL")
            sanitized_indexes.append(sanitized)
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
            index_urls=sanitized_indexes,
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
        # Manager 1.1 locks predate the additive installed-file tree digest. Preserve their seals;
        # manager 1.2 locks populate these fields and therefore bind them normally.
        if all(
            item.installed_files_hash is None and item.installed_file_count is None
            for item in lock.packages
        ):
            for package in body.get("packages", []):
                package.pop("installed_files_hash", None)
                package.pop("installed_file_count", None)
        optional_flash_memory_fields = (
            "forward_duration_seconds",
            "backward_duration_seconds",
            "optimizer_step_duration_seconds",
            "gpu_temperature_celsius",
            "gpu_power_watts",
        )
        if lock.probe_evidence is not None and all(
            getattr(lock.probe_evidence.memory, field_name) is None
            for field_name in optional_flash_memory_fields
        ):
            memory_body = body.get("probe_evidence", {}).get("memory", {})
            for field_name in optional_flash_memory_fields:
                memory_body.pop(field_name, None)
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
            and item.installed_files_hash is None
            and item.installed_file_count is None
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
                "installed_files_hash",
                "installed_file_count",
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
        bad_records = [
            item.name
            for item in inventory.packages
            if item.version is not None and not item.has_complete_record_count_evidence()
        ]
        if bad_records:
            raise EnvironmentManagerError(
                "the environment lock cannot be finalized without complete all-row RECORD "
                "evidence: " + ", ".join(sorted(bad_records, key=str.casefold))
            )
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
            installed_versions = {
                item.normalized_name or _normalized_package_name(item.name): item.version
                for item in inventory.packages
            }
            version_mismatches: list[str] = []
            runtime_payload = probe_evidence.tuple_result.measured.get("runtime")
            runtime_packages = (
                runtime_payload.get("packages") if isinstance(runtime_payload, dict) else None
            )
            if not isinstance(runtime_packages, dict):
                raise EnvironmentManagerError(
                    "the complete QLoRA tuple omitted runtime package identity"
                )
            for runtime_name, runtime_version in runtime_packages.items():
                _, normalized_runtime_name = _validated_package_name(runtime_name)
                if installed_versions.get(normalized_runtime_name) != runtime_version:
                    version_mismatches.append(
                        f"{normalized_runtime_name}: probe observed {runtime_version}, "
                        f"inventory observed {installed_versions.get(normalized_runtime_name)}"
                    )
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
            _assert_worker_install_evidence(
                installation.package_install_evidence,
                resolution.worker_artifact,
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
        body = evidence.model_dump(
            mode="json", exclude={"evidence_id", "evidence_hash"}
        )
        # Manager 1.1 math evidence predates the additive flash timing/thermal fields. Preserve the
        # nested evidence seal just as _lock_digest preserves the enclosing lock seal.
        optional_flash_memory_fields = (
            "forward_duration_seconds",
            "backward_duration_seconds",
            "optimizer_step_duration_seconds",
            "gpu_temperature_celsius",
            "gpu_power_watts",
        )
        if all(
            getattr(evidence.memory, field_name) is None
            for field_name in optional_flash_memory_fields
        ):
            memory_body = body.get("memory", {})
            for field_name in optional_flash_memory_fields:
                memory_body.pop(field_name, None)
        return _canonical_sha256(body)

    @staticmethod
    def _validate_probe_configuration(
        configuration: Mapping[str, object],
        required: QloraExecutionProbeSpec,
        *,
        legacy_math_configuration: bool,
    ) -> None:
        """Validate the execution meaning recorded by one complete readiness tuple."""

        expected_kernel = required.execution_combination.attention_kernel.value
        expected_backend = (
            "FLASH_ATTENTION"
            if required.probe == "cuda_qlora_sdpa_flash_execution"
            else "MATH"
        )
        expected_toggles = {
            "flash_sdp_enabled": required.flash_sdp_enabled,
            "memory_efficient_sdp_enabled": required.memory_efficient_sdp_enabled,
            "math_sdp_enabled": required.math_sdp_enabled,
        }
        expected_configuration: dict[str, object] = {
            "compute_dtype": required.compute_dtype,
            "quantization": required.quantization,
            "double_quantization": required.double_quantization,
            "attention_api": required.attention_api,
            "device_map": {"": required.device},
            "target_modules": required.target_modules,
            "gradient_checkpointing": required.gradient_checkpointing,
            "optimizer": required.optimizer,
        }
        if legacy_math_configuration:
            if required.probe != "cuda_qlora_math_execution":
                raise EnvironmentManagerError(
                    "legacy complete-tuple evidence is valid only for the sealed math rollback"
                )
        else:
            expected_configuration.update(
                {
                    # BF16 forward autocast is part of the already-sealed compute policy, not free
                    # text for every new readiness environment.
                    "forward_autocast": required.compute_dtype,
                    "attention_kernel": expected_kernel,
                    "forced_sdp_backend": expected_backend,
                    # These constants define the bounded evidence scope. They remain manager
                    # acceptance policy so the readiness-v2 recipe digest itself stays stable.
                    "batch_size": 1,
                    "sequence_length": 8,
                    "lora_r": 2,
                    "lora_alpha": 4,
                    "seed": 0,
                }
            )
        mismatched_configuration = sorted(
            key
            for key, expected in expected_configuration.items()
            if configuration.get(key) != expected
        )
        if mismatched_configuration:
            raise EnvironmentManagerError(
                "the complete QLoRA tuple measured configuration disagrees with the recipe: "
                + ", ".join(mismatched_configuration)
            )
        toggle_keys = (
            ("attention_toggles",)
            if legacy_math_configuration
            else ("attention_toggles", "attention_toggles_during")
        )
        for key in toggle_keys:
            if configuration.get(key) != expected_toggles:
                raise EnvironmentManagerError(
                    f"the complete QLoRA tuple did not enforce sealed SDPA settings in {key}"
                )

    def _locked_probe_evidence_mismatch(
        self,
        lock: EnvironmentLock,
        required: QloraExecutionProbeSpec | None,
    ) -> bool:
        """Reject a weak/tampered readiness seal before health probes import installed code."""

        if required is None:
            return lock.probe_evidence is not None
        evidence = lock.probe_evidence
        if evidence is None or evidence.required_spec != required:
            return True
        if evidence.evidence_hash != self._probe_evidence_digest(evidence):
            return True
        tuple_result = evidence.tuple_result
        if (
            tuple_result.outcome != FailureTaxonomy.PASS
            or required.execution_combination not in tuple_result.execution_combinations
        ):
            return True
        configuration = tuple_result.measured.get("configuration")
        if not isinstance(configuration, dict):
            return True
        legacy_math_configuration = (
            lock.manager_version == "1.1.0"
            and required.probe == "cuda_qlora_math_execution"
            and "forward_autocast" not in configuration
        )
        try:
            self._validate_probe_configuration(
                configuration,
                required,
                legacy_math_configuration=legacy_math_configuration,
            )
            measured_memory = ProbeMemoryEvidence.model_validate(
                tuple_result.measured.get("memory")
            )
        except (EnvironmentManagerError, ValueError):
            return True
        if measured_memory != evidence.memory:
            return True
        return (
            not legacy_math_configuration
            and tuple_result.measured.get("adapter_round_trip_verified") is not True
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
        *,
        legacy_math_configuration: bool = False,
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
        configuration = tuple_result.measured.get("configuration")
        if not isinstance(configuration, dict):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple did not record its measured execution configuration"
            )
        self._validate_probe_configuration(
            configuration,
            required,
            legacy_math_configuration=legacy_math_configuration,
        )
        for key in ("loss", "reload_loss"):
            value = tuple_result.measured.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                raise EnvironmentManagerError(
                    f"the complete QLoRA tuple did not record a finite {key}"
                )
        adapter_bytes = tuple_result.measured.get("adapter_weight_bytes")
        if (
            required.require_adapter_round_trip
            and (
                isinstance(adapter_bytes, bool)
                or not isinstance(adapter_bytes, int)
                or adapter_bytes <= 0
            )
        ):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple did not prove a Safetensors adapter round trip"
            )
        if (
            not legacy_math_configuration
            and tuple_result.measured.get("adapter_round_trip_verified") is not True
        ):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple did not compare reloaded adapter bytes"
            )
        runtime = tuple_result.measured.get("runtime")
        package_versions = runtime.get("packages") if isinstance(runtime, dict) else None
        if not isinstance(package_versions, dict):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple did not record dependency runtime versions"
            )
        observed_versions: dict[str, str] = {}
        for raw_name, raw_version in package_versions.items():
            _, normalized_name = _validated_package_name(raw_name)
            if normalized_name in observed_versions:
                raise EnvironmentManagerError(
                    "the complete QLoRA tuple recorded duplicate normalized runtime packages"
                )
            if (
                not isinstance(raw_version, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", raw_version)
            ):
                raise EnvironmentManagerError(
                    f"the complete QLoRA tuple recorded an invalid runtime version for {normalized_name}"
                )
            observed_versions[normalized_name] = raw_version
        missing = sorted(set(required.required_distributions) - set(observed_versions))
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
        if (
            memory.peak_gpu_allocated_bytes < memory.baseline_gpu_allocated_bytes
            or memory.peak_gpu_reserved_bytes < memory.baseline_gpu_reserved_bytes
            or memory.peak_host_rss_bytes < memory.baseline_host_rss_bytes
            or (
                memory.baseline_nvidia_smi_process_bytes is not None
                and memory.peak_nvidia_smi_process_bytes is not None
                and memory.peak_nvidia_smi_process_bytes
                < memory.baseline_nvidia_smi_process_bytes
            )
        ):
            raise EnvironmentManagerError(
                "the complete QLoRA tuple reported a peak below its measurement baseline"
            )
        finite_measurements = {
            "duration_seconds": memory.duration_seconds,
            "forward_duration_seconds": memory.forward_duration_seconds,
            "backward_duration_seconds": memory.backward_duration_seconds,
            "optimizer_step_duration_seconds": memory.optimizer_step_duration_seconds,
            "gpu_temperature_celsius": memory.gpu_temperature_celsius,
            "gpu_power_watts": memory.gpu_power_watts,
        }
        non_finite = sorted(
            name
            for name, value in finite_measurements.items()
            if value is not None and not math.isfinite(value)
        )
        if non_finite:
            raise EnvironmentManagerError(
                "the complete QLoRA tuple reported non-finite measurements: "
                + ", ".join(non_finite)
            )
        if memory.gpu_device_scope == "unavailable" and (
            memory.baseline_nvidia_smi_process_bytes is not None
            or memory.peak_nvidia_smi_process_bytes is not None
        ):
            raise EnvironmentManagerError(
                "unavailable nvidia-smi scope cannot carry process-residency measurements"
            )
        if memory.gpu_device_scope == "nvidia_smi_current_process" and (
            memory.baseline_nvidia_smi_process_bytes is None
            and memory.peak_nvidia_smi_process_bytes is None
        ):
            raise EnvironmentManagerError(
                "nvidia-smi process scope requires at least one process-residency measurement"
            )
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
        lock: EnvironmentLock,
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
            sealed_configuration = (
                lock.probe_evidence.tuple_result.measured.get("configuration")
                if lock.probe_evidence is not None
                else None
            )
            legacy_math_configuration = (
                lock.manager_version == "1.1.0"
                and required.probe == "cuda_qlora_math_execution"
                and isinstance(sealed_configuration, dict)
                and "forward_autocast" not in sealed_configuration
            )
            capability_results, evidence = self._complete_probe_evidence(
                capability_payload,
                required,
                legacy_math_configuration=legacy_math_configuration,
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
        drifted: list[str] = []
        sources: list[str] = []

        def _by_name(
            packages: Sequence[PackageLock], label: str
        ) -> dict[str, PackageLock]:
            indexed: dict[str, PackageLock] = {}
            duplicates: set[str] = set()
            for item in packages:
                name = item.normalized_name or _normalized_package_name(item.name)
                if name in indexed:
                    duplicates.add(name)
                else:
                    indexed[name] = item
            drifted.extend(
                f"{name}: duplicate normalized distribution in {label} inventory"
                for name in sorted(duplicates)
            )
            return indexed

        expected_by_name = _by_name(expected, "locked")
        actual_by_name = _by_name(actual, "live")
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
            before_files_hash = (
                before.installed_files_hash.value if before.installed_files_hash else None
            )
            after_files_hash = (
                after.installed_files_hash.value if after.installed_files_hash else None
            )
            if before_files_hash is not None and before_files_hash != after_files_hash:
                drifted.append(f"{before.name}: installed file bytes changed")
            if before.record_count_semantics != after.record_count_semantics:
                drifted.append(f"{before.name}: RECORD count semantics changed")
            if (
                before.record_entries is not None
                and before.record_entries != after.record_entries
            ):
                drifted.append(f"{before.name}: installed RECORD entry count changed")
            if (
                before.record_verified_entries is not None
                and before.record_verified_entries != after.record_verified_entries
            ):
                drifted.append(f"{before.name}: verified RECORD entry count changed")
            if (
                before.installed_file_count is not None
                and before.installed_file_count != after.installed_file_count
            ):
                drifted.append(f"{before.name}: installed RECORD file count changed")
            before_artifact_hash = (
                before.artifact_hash.value if before.artifact_hash else None
            )
            after_artifact_hash = after.artifact_hash.value if after.artifact_hash else None
            if (
                before_artifact_hash is not None
                and after_artifact_hash is not None
                and before_artifact_hash != after_artifact_hash
            ):
                drifted.append(f"{before.name}: installed artifact hash changed")
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
            elif before.vcs_repository != after.vcs_repository or before.vcs_commit != after.vcs_commit:
                sources.append(f"{before.name}: VCS source changed")
            elif (before.direct is True or after.direct is True) and (
                before.editable != after.editable
            ):
                sources.append(f"{before.name}: editable-source status changed")
            elif (
                before.source_index_url is not None
                and after.source_index_url is not None
                and before.source_index_url != after.source_index_url
            ):
                sources.append(f"{before.name}: package index changed")
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
    def _recovery_remediation(*, sealed: bool, problem: str) -> str:
        problem = problem.rstrip(".")
        if sealed:
            return (
                f"{problem}. Create a replacement under a new environment id from a reviewed "
                "plan and preserve this sealed identity."
            )
        return (
            f"{problem}. Recover this unsealed attempt with env-recreate and a newly reviewed "
            "plan."
        )

    @classmethod
    def _probe_remediation(
        cls,
        state: EnvironmentState,
        *,
        sealed: bool = False,
    ) -> str | None:
        if state == EnvironmentState.functional_probe_passed:
            return "CPU functionality passed; run env-probe on a CUDA host for hardware verification."
        if state in {EnvironmentState.degraded, EnvironmentState.incompatible}:
            return cls._recovery_remediation(
                sealed=sealed,
                problem="Inspect the incompatible or degraded probe evidence",
            )
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
