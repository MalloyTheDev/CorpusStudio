"""Model fetch — the PURE license classification (the honesty-critical part).

The actual download (`fetch_model`) needs huggingface_hub (a training-runtime dep), so it is verified
on a real box; here we lock the license logic: permissive → OK, and everything unknown/custom/NC/
missing → NOT permissive (restricted-until-verified), mirroring the provenance gate's fail-closed stance.
"""

from __future__ import annotations

import pytest

from corpus_studio.training.model_fetch import classify_license, license_from_readme


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
