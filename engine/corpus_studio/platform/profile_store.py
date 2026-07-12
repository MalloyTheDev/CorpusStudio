"""The host profile / capability store — platform slice 9.

Persists an :class:`EnvironmentProfile` + its :class:`CapabilityReport` keyed by the deterministic
``environment_signature`` (a sha256 over the host/env characterization that EXCLUDES volatile fields
like free memory + timestamps). Running the functional capability probes is expensive — they load
torch and execute kernels — so on an UNCHANGED host the cached report is reused and the probes are
skipped. When the signature changes (a driver/torch/GPU swap), the cache misses and the host is
re-profiled + re-probed (recalibrate-on-change).

Dependency-light: stdlib + platform contracts only. The profiler / prober are INJECTED into
:func:`resolve_capabilities` (the real ones lazy-import torch inside), so this module never imports
the heavy stack itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from corpus_studio.platform.contracts import CapabilityReport, EnvironmentProfile

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_PROFILE_FILE = "EnvironmentProfile.json"
_REPORT_FILE = "CapabilityReport.json"


def default_store_dir() -> Path:
    """The default on-disk cache location (``~/.corpus_studio/profiles``)."""
    return Path.home() / ".corpus_studio" / "profiles"


def _entry_dir(store_dir: str | Path, signature: str) -> Path:
    return Path(store_dir) / signature


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_environment(
    profile: EnvironmentProfile, report: CapabilityReport, store_dir: str | Path
) -> Path:
    """Persist the profile + report under ``<store_dir>/<environment_signature>/`` (atomic writes).
    Returns the entry directory."""
    directory = _entry_dir(store_dir, profile.environment_signature)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / _PROFILE_FILE, profile.model_dump_json(indent=2))
    _atomic_write(directory / _REPORT_FILE, report.model_dump_json(indent=2))
    return directory


def load_profile(signature: str, store_dir: str | Path) -> EnvironmentProfile | None:
    """The cached EnvironmentProfile for ``signature``, or ``None`` when absent/unreadable."""
    path = _entry_dir(store_dir, signature) / _PROFILE_FILE
    return _load(path, EnvironmentProfile)


def load_report(signature: str, store_dir: str | Path) -> CapabilityReport | None:
    """The cached CapabilityReport for ``signature``, or ``None`` when absent/unreadable."""
    path = _entry_dir(store_dir, signature) / _REPORT_FILE
    return _load(path, CapabilityReport)


def _load(path: Path, model: type[_ModelT]) -> _ModelT | None:
    if not path.is_file():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError, ValueError):
        # A corrupt / half-written / stale-schema cache entry is a MISS, never a crash.
        return None


def list_signatures(store_dir: str | Path) -> list[str]:
    """Every environment signature with a complete cached entry (profile + report), sorted."""
    directory = Path(store_dir)
    if not directory.is_dir():
        return []
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir()
        and (entry / _PROFILE_FILE).is_file()
        and (entry / _REPORT_FILE).is_file()
    )


@dataclass
class ResolvedEnvironment:
    """The outcome of :func:`resolve_capabilities`: the current profile, its capability report, and
    whether the report came from the cache (``cached=True`` means the probes were NOT re-run)."""

    profile: EnvironmentProfile
    report: CapabilityReport
    cached: bool


def resolve_capabilities(
    store_dir: str | Path,
    *,
    build_profile: Callable[[], EnvironmentProfile],
    run_probes: Callable[[EnvironmentProfile], CapabilityReport],
    refresh: bool = False,
) -> ResolvedEnvironment:
    """Profile the host (cheap), then reuse the cached CapabilityReport when the signature is
    unchanged — skipping the expensive probes — or re-probe + persist on a cache miss / ``refresh``.
    ``build_profile`` / ``run_probes`` are injected so this stays torch-free and unit-testable."""
    profile = build_profile()
    if not refresh:
        cached = load_report(profile.environment_signature, store_dir)
        if cached is not None:
            return ResolvedEnvironment(profile=profile, report=cached, cached=True)
    report = run_probes(profile)
    save_environment(profile, report, store_dir)
    return ResolvedEnvironment(profile=profile, report=report, cached=False)
