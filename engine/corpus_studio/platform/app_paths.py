"""Per-user application-data roots for CorpusStudio runtime state.

Single source of truth for where CorpusStudio keeps machine-local, non-source state - the Environment
Manager's state and sealed run output. These roots live OUTSIDE any source checkout: on Windows under
``%LOCALAPPDATA%``, elsewhere under ``$XDG_DATA_HOME`` (each with the standard per-user fallback). Both
the environment-manager root and the default output root derive from ``corpusstudio_data_home`` so they
can never diverge. Torch-free control-plane code (``os`` + ``pathlib`` only)."""
from __future__ import annotations

import os
from pathlib import Path


def corpusstudio_data_home() -> Path:
    """Base application-data directory for CorpusStudio, outside any source checkout."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "CorpusStudio"
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "corpusstudio"
