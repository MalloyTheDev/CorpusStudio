"""Project-local provider policy overrides (local-first, inspectable JSON).

A user approves generation for a specific local model/route by writing an entry
into ``provider_policy_overrides.json`` in the project directory. The engine
merges these over the built-in defaults in ``resolve_policy``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from corpus_studio.providers.policy import most_specific_override_key

OVERRIDES_FILENAME = "provider_policy_overrides.json"


def overrides_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / OVERRIDES_FILENAME


def load_overrides(project_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load overrides for a project, or an empty mapping when absent/unreadable."""

    path = overrides_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_overrides(project_dir: Path | str, overrides: dict[str, dict[str, Any]]) -> Path:
    """Write overrides atomically (temp file + replace) to avoid partial writes."""

    path = overrides_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def approve_generation(
    project_dir: Path | str,
    provider_id: str,
    model_id: str | None = None,
    route_id: str | None = None,
    outputs_trainable: bool = True,
) -> str:
    """Approve trainable generation for a specific model/route; returns the key.

    This only records intent; the engine's role policy still decides whether the
    provider *can* generate (a blocked frontier provider stays blocked), and
    generated candidates still require human review before they can be saved.
    """

    overrides = load_overrides(project_dir)
    key = most_specific_override_key(provider_id, model_id, route_id)
    overrides[key] = {
        "outputs_trainable": outputs_trainable,
        "user_approved_generation": True,
    }
    save_overrides(project_dir, overrides)
    return key


def revoke_generation(
    project_dir: Path | str,
    provider_id: str,
    model_id: str | None = None,
    route_id: str | None = None,
) -> bool:
    """Remove an approval entry; returns True if one was removed."""

    overrides = load_overrides(project_dir)
    key = most_specific_override_key(provider_id, model_id, route_id)
    if key in overrides:
        del overrides[key]
        save_overrides(project_dir, overrides)
        return True
    return False
