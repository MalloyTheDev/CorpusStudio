"""Reproducibility manifest for a training run.

A one-stop shop must be able to answer "what exactly produced this model?" — so a
run record should pin the *recipe*, not just point at paths that can change. This
captures, at run start, the identity of the inputs and the environment:

* the **dataset fingerprint** (the engine's canonical content hash of
  ``examples.jsonl``) + its row count — proves *which data* trained the model,
  independent of the file path or a later edit;
* the **config SHA-256** — proves *which config*, byte-for-byte;
* the **engine version** and **platform / Python** — the environment that
  generated the run.

Together with the fields the run record already carries (the exact ``argv``, the
``base_model``, the dataset-version back-link, and the before/after eval), this is
the auditable recipe behind a produced model.

Known limitation: the generated training config does not yet emit a random
**seed**, so this manifest pins the *inputs* (data + config + environment) but not
the exact weight initialisation — treat it as data/config provenance, not
bit-exact weight reproduction. (Seed capture is a follow-up.)
"""

import hashlib
import platform
from pathlib import Path

from pydantic import BaseModel

from corpus_studio import __version__
from corpus_studio.versions.version_registry import fingerprint_dataset


class RunProvenance(BaseModel):
    """The reproducibility manifest attached to a training run record."""

    dataset_fingerprint: str | None = None
    dataset_row_count: int = 0
    config_sha256: str | None = None
    engine_version: str = ""
    platform: str = ""
    python_version: str = ""


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_run_provenance(project_dir: Path | str, config_path: Path | str) -> RunProvenance:
    """Build the reproducibility manifest for a run. Best-effort: a missing dataset
    or config leaves the corresponding field ``None`` rather than raising, so a run
    is never blocked by manifest capture."""
    examples_path = Path(project_dir) / "examples.jsonl"
    if examples_path.exists():
        fingerprint, row_count = fingerprint_dataset(examples_path)
    else:
        fingerprint, row_count = None, 0

    return RunProvenance(
        dataset_fingerprint=fingerprint,
        dataset_row_count=row_count,
        config_sha256=_sha256_file(Path(config_path)),
        engine_version=__version__,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )
