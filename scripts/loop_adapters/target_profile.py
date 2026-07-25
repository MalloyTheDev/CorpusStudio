"""The TARGET PROFILE: what the write-capable loop may change in a given repository.

Through 7.1.5 the write adapter was hardcoded to CorpusStudio's shape - the writable surface was
``engine/corpus_studio/**/*.py``, the safe-basename rule was Python-only, the assurance tooling was assumed
to live inside the target, and worker-reachability (a CorpusStudio-specific concept) always ran. Pointing
the loop at a second repository proved none of that is portable: every candidate was refused because
nothing matched the surface, and ``cs_assure`` could not run because the target had no policy bundle.

THE ONE NON-NEGOTIABLE RULE: a profile is OPERATOR-OWNED and lives OUTSIDE the target repository. If the
target owned its own profile, a candidate could WIDEN ITS OWN WRITABLE SURFACE in the same change the
surface is supposed to bound - the exact self-modification the assurance plane separates trusted-base
policy from candidate policy to prevent. Profiles therefore ship with the loop (``scripts/loop/profiles/``)
or are supplied by absolute path; they are never read from the candidate.

THREE ROOTS THAT WERE CONFLATED, now separate:
  * ``repo_root``        - the repository being CHANGED (may be any repo)
  * ``assurance_root``   - where the TRUSTED ``cs_assure`` lives (this repo), independent of the target
  * the profile itself   - operator-owned, outside both

FAIL-CLOSED BY CONSTRUCTION: ``writable_globs`` has NO default. A missing, unreadable or malformed profile
yields no writable surface at all, so forgetting configuration can only ever refuse - never widen.

stdlib-only, like the rest of the adapter layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(__file__).resolve().parent.parent / "loop" / "profiles"

# Keys a profile may declare. An unknown key is REFUSED rather than ignored: a typo in
# "writable_globs" would otherwise silently produce an empty surface that looks like a working config.
_ALLOWED_KEYS = frozenset({
    "name", "description", "writable_globs", "safe_basename", "max_changed_paths", "max_changed_lines",
    "max_changed_bytes", "max_line_bytes", "max_blob_bytes", "max_rationale_bytes",
    "require_worker_reachability", "obligations_policy",
})


class ProfileError(ValueError):
    """A target profile could not be read or is not trustworthy (fail-closed)."""


@dataclass(frozen=True)
class TargetProfile:
    """What the loop may change in ONE repository, and the bounds it must stay inside.

    Defaults are the CONSERVATIVE end of what 7.1.x measured: they are the bounds that were shown not to
    trip on real code while still refusing every measured attack. ``writable_globs`` deliberately has no
    default - a profile that does not say what is writable makes nothing writable."""

    name: str
    writable_globs: tuple[str, ...] = ()
    # Which basenames are acceptable. Python-shaped by default because that is what the measured attacks
    # targeted (conftest injection, stdlib shadowing); a TypeScript or Rust target supplies its own.
    safe_basename: str = r"(?:__[a-z0-9_]+__|[a-z0-9][a-z0-9_]*)\.py\Z"
    max_changed_paths: int = 2
    max_changed_lines: int = 60
    max_changed_bytes: int = 8 * 1024
    max_line_bytes: int = 400
    max_blob_bytes: int = 1 << 20
    max_rationale_bytes: int = 4096
    # CorpusStudio-specific: the worker import closure. Meaningless for a repo with no ML worker, and
    # requiring it there would fail closed forever - so it is OPT-IN.
    require_worker_reachability: bool = False
    # A repo-relative obligations policy for cs_assure. None -> the target's own default location, which
    # is correct for CorpusStudio and absent for most other repos (see _assure_candidate's handling).
    obligations_policy: str | None = None
    description: str = ""
    _basename_re: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            compiled = re.compile(self.safe_basename)
        except re.error as exc:
            raise ProfileError(f"profile {self.name!r} has an invalid safe_basename: {exc}") from exc
        object.__setattr__(self, "_basename_re", compiled)

    def basename_ok(self, basename: str) -> bool:
        return bool(self._basename_re.match(basename))


def load_profile(source: str | Path) -> TargetProfile:
    """Load a profile by NAME (from the shipped ``scripts/loop/profiles/``) or by explicit path.

    Fail-closed: an unreadable file, malformed JSON, an unknown key, or a non-list ``writable_globs`` all
    raise rather than degrading to a default - because the only safe degradation would be an empty surface,
    and silently publishing nothing is indistinguishable from a broken loop."""
    path = Path(source)
    if not path.suffix:
        path = PROFILE_DIR / f"{source}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProfileError(f"target profile {path} could not be read: {exc}") from exc
    except ValueError as exc:
        raise ProfileError(f"target profile {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"target profile {path} is not an object")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise ProfileError(f"target profile {path} has unknown key(s) {unknown}; refusing to guess")
    globs = raw.get("writable_globs")
    # An EMPTY list is refused, not accepted as "nothing is writable". Both refuse every candidate, but a
    # loop that silently publishes nothing is indistinguishable from a broken one - so a profile that
    # declares no surface is a configuration error, surfaced as one.
    if not isinstance(globs, list) or not globs or not all(isinstance(g, str) and g for g in globs):
        raise ProfileError(f"target profile {path} needs a non-empty list of string writable_globs")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ProfileError(f"target profile {path} has no name")
    fields: dict[str, Any] = {k: v for k, v in raw.items() if k not in ("writable_globs", "name")}
    return TargetProfile(name=name, writable_globs=tuple(globs), **fields)
