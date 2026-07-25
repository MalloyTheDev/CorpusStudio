"""The TARGET PROFILE (scripts/loop_adapters/target_profile.py) - what makes the write loop portable.

Pins the property that matters most: a profile is OPERATOR-OWNED and fail-closed. If a target repository
could supply its own profile, a candidate could widen its own writable surface in the very change the
surface exists to bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import loop_adapters.single_agent_write as saw  # noqa: E402
from loop_adapters.target_profile import PROFILE_DIR, ProfileError, TargetProfile, load_profile  # noqa: E402


def test_the_shipped_profiles_load_and_declare_a_surface() -> None:
    for path in sorted(PROFILE_DIR.glob("*.json")):
        profile = load_profile(path)
        assert profile.writable_globs, f"{path.name} declares no writable surface"
        assert profile.name


def test_corpusstudio_is_itself_just_a_profile() -> None:
    """The portability proof: the repo the loop was built against is expressed as an ordinary profile,
    not as hardcoded constants with escape hatches."""
    p = load_profile("corpusstudio")
    assert p.writable_globs == ("engine/corpus_studio/**/*.py",)
    assert p.require_worker_reachability is True          # the worker closure IS meaningful here
    assert saw._path_is_writable("engine/corpus_studio/cli.py", p)
    assert not saw._path_is_writable("src/clockvault/audit.py", p)


def test_a_foreign_profile_moves_the_surface(tmp_path: Path) -> None:
    """A different repo gets a different surface - and CorpusStudio's own paths are NOT writable under it."""
    p = load_profile("chaos-desktop-pet")
    assert saw._path_is_writable("src/pet/brain.py", p)
    assert not saw._path_is_writable("engine/corpus_studio/cli.py", p)
    assert p.require_worker_reachability is False         # no ML worker exists there


@pytest.mark.parametrize("body,why", [
    ('{"name": "x"}', "no writable_globs -> nothing is writable"),
    ('{"name": "x", "writable_globs": []}', "an empty surface is refused, not silently accepted"),
    ('{"name": "x", "writable_globs": ["a"], "typo_key": 1}', "an unknown key is refused, never ignored"),
    ('{"writable_globs": ["a"]}', "an unnamed profile is refused"),
    ('not json at all', "malformed JSON fails closed"),
    ('["not", "an", "object"]', "a non-object fails closed"),
])
def test_a_bad_profile_fails_closed(body: str, why: str, tmp_path: Path) -> None:
    """Fail-closed by construction: the only safe degradation is an empty surface, and a loop that
    silently publishes nothing is indistinguishable from a broken one - so it raises instead."""
    f = tmp_path / "p.json"
    f.write_text(body)
    with pytest.raises(ProfileError):
        load_profile(f)


def test_an_invalid_safe_basename_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"name": "x", "writable_globs": ["a/**"], "safe_basename": "("}))
    with pytest.raises(ProfileError, match="safe_basename"):
        load_profile(f)


def test_bounds_are_profile_driven_not_global() -> None:
    """Bounds move with the target: a repo with longer lines or bigger changes declares its own, and the
    classifier reads them from the profile rather than module constants."""
    tight = TargetProfile(name="tight", writable_globs=("src/**/*.py",), max_line_bytes=10,
                          max_changed_paths=1)
    loose = TargetProfile(name="loose", writable_globs=("src/**/*.py",), max_line_bytes=4000)
    assert tight.max_line_bytes != loose.max_line_bytes
    assert tight.basename_ok("mod.py") and not tight.basename_ok("Mod.py")


def test_a_profile_declaring_a_non_python_surface_uses_its_own_basename_rule() -> None:
    ts = TargetProfile(name="ts", writable_globs=("src/**/*.ts",),
                       safe_basename=r"[a-z0-9][a-z0-9_.-]*\.ts\Z")
    assert ts.basename_ok("client.ts") and not ts.basename_ok("client.py")
    assert saw._path_is_writable("src/api/client.ts", ts)
