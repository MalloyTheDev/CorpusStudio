"""Model fetch — the PURE license classification (the honesty-critical part).

The actual download (`fetch_model`) needs huggingface_hub (a training-runtime dep), so it is verified
on a real box; here we lock the license logic: permissive → OK, and everything unknown/custom/NC/
missing → NOT permissive (restricted-until-verified), mirroring the provenance gate's fail-closed stance.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from corpus_studio.training.model_fetch import classify_license, fetch_model, license_from_readme


@pytest.mark.parametrize("license_id", ["mit", "MIT", "apache-2.0", "Apache-2.0", "bsd-3-clause", "cc0-1.0"])
def test_permissive_licenses_pass(license_id):
    permissive, note = classify_license(license_id)
    assert permissive is True
    assert "permissive" in note.lower()


@pytest.mark.parametrize(
    "license_id",
    [None, "", "   ", "other", "unknown", "llama3", "llama3.1", "gemma", "cc-by-nc-4.0", "openrail", "proprietary"],
)
def test_non_permissive_licenses_are_flagged(license_id):
    permissive, note = classify_license(license_id)
    assert permissive is False
    assert note  # always an explanatory reason


def test_missing_license_is_restricted_not_blank():
    permissive, note = classify_license(None)
    assert permissive is False
    assert "all-rights-reserved" in note or "verify" in note


def test_noncommercial_is_called_out():
    permissive, note = classify_license("cc-by-nc-4.0")
    assert permissive is False
    assert "NON-COMMERCIAL" in note.upper() or "non-commercial" in note.lower()


def test_llama_custom_license_flagged():
    permissive, note = classify_license("llama3.1")
    assert permissive is False
    assert "llama" in note.lower()


# ---- local README license parse (robust, no network) -------------------------


def test_license_from_readme_frontmatter(tmp_path):
    (tmp_path / "README.md").write_text(
        "---\nlibrary_name: transformers\nlicense: apache-2.0\ntags:\n- text\n---\n# Model\n",
        encoding="utf-8",
    )
    assert license_from_readme(tmp_path) == "apache-2.0"


def test_license_from_readme_quoted(tmp_path):
    (tmp_path / "README.md").write_text('---\nlicense: "mit"\n---\nhi\n', encoding="utf-8")
    assert license_from_readme(tmp_path) == "mit"


def test_license_from_readme_absent_or_no_field(tmp_path):
    assert license_from_readme(tmp_path) is None  # no README
    (tmp_path / "README.md").write_text("# no frontmatter here\n", encoding="utf-8")
    assert license_from_readme(tmp_path) is None
    (tmp_path / "README.md").write_text("---\ntags: [a]\n---\nbody\n", encoding="utf-8")
    assert license_from_readme(tmp_path) is None


# --- fetch_model: resilience + exit-code (WBG tester run) — huggingface_hub is mocked ---

def _install_fake_hub(monkeypatch, snapshot_impl):
    """Inject a fake huggingface_hub so fetch_model's lazy `from huggingface_hub import snapshot_download`
    resolves to our impl (the real hub is a [train] dep, absent in the core test venv)."""
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = snapshot_impl
    fake.model_info = lambda *a, **k: types.SimpleNamespace(card_data=None, tags=[])
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def _write_repo(target, *, weights=True):
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    if weights:
        (target / "model.safetensors").write_text("x" * 128, encoding="utf-8")
    return str(target)


def test_fetch_model_retries_through_resets_then_succeeds(tmp_path, monkeypatch):
    # WinError 10054 resets on the first two attempts, success on the third — the resume-loop grinds through.
    calls = {"n": 0}

    def snapshot(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("[WinError 10054] connection reset by remote host")
        return _write_repo(tmp_path / "repo")

    _install_fake_hub(monkeypatch, snapshot)
    result = fetch_model("org/model")

    assert calls["n"] == 3
    assert result.license == "mit"
    assert "model.safetensors" in result.weight_files


def test_fetch_model_zero_weights_raises_for_exit_2(tmp_path, monkeypatch):
    # Only tokenizer/config landed (dropped connection) — a weightless fetch is a FAILED fetch.
    _install_fake_hub(monkeypatch, lambda **k: _write_repo(tmp_path / "repo", weights=False))
    with pytest.raises(RuntimeError, match="No weight files"):
        fetch_model("org/model")


def test_fetch_model_raises_after_exhausting_attempts(tmp_path, monkeypatch):
    calls = {"n": 0}

    def snapshot(**kwargs):
        calls["n"] += 1
        raise OSError("[WinError 10054] connection reset by remote host")

    _install_fake_hub(monkeypatch, snapshot)
    with pytest.raises(RuntimeError, match="after 5 attempts"):
        fetch_model("org/model")
    assert calls["n"] == 5


def test_fetch_model_defaults_hf_transfer_off(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    _install_fake_hub(monkeypatch, lambda **k: _write_repo(tmp_path / "repo"))
    fetch_model("org/model")
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"  # resilient Python downloader, not fragile hf_transfer


def test_fetch_model_respects_an_explicit_hf_transfer_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    _install_fake_hub(monkeypatch, lambda **k: _write_repo(tmp_path / "repo"))
    fetch_model("org/model")
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"  # setdefault respects the user's opt-in
