"""Image content inspection for ``image_path`` fields (schema-validate --check-images).

Fixtures are GENERATED, never committed: a binary blob in the repository cannot be reviewed, and the
interesting cases (a truncated PNG, a CMYK JPEG, an animated GIF) are a few lines of Pillow each.

Pillow is an optional extra, so the whole module skips without it. The dependency-light boundary is
asserted separately in ``test_image_support_boundary``, which must hold whether or not the extra is
installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.quality.images import (
    ImageSupportUnavailable,
    image_content_problems,
    inspect_image,
)

Image = pytest.importorskip("PIL.Image", reason="image inspection needs the optional images extra")

runner = CliRunner()

SCHEMA = {
    "id": "sprite",
    "name": "Sprite",
    "version": "0.1.0",
    "fields": [
        {"name": "image", "type": "image_path", "required": True},
        {"name": "caption", "type": "text", "required": True},
    ],
}


def _project(tmp_path: Path, schema: dict | None = None) -> Path:
    """A project whose schemas/ holds the sprite schema, as a project-local schema."""
    root = tmp_path / "projects"
    result = runner.invoke(
        app, ["new-project", "sprites", "Sprites", "image_caption", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    project_dir = root / "sprites"
    schemas = project_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    payload = schema or SCHEMA
    (schemas / f"{payload['id']}.schema.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return project_dir


def _rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _sprite(path: Path, *, size: tuple[int, int] = (16, 16), colors: int = 4) -> Path:
    """A small palette-quantised image, which is what a pixel-art row actually looks like."""
    image = Image.new("P", size)
    palette: list[int] = []
    for index in range(colors):
        palette.extend([index * 40 % 256, (index * 70) % 256, (index * 110) % 256])
    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (x + y) % colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


# ---- facts -------------------------------------------------------------------------------------


def test_inspect_reports_dimensions_mode_and_palette(tmp_path):
    facts = inspect_image(_sprite(tmp_path / "s.png", size=(16, 16), colors=4))
    assert (facts.width, facts.height) == (16, 16)
    assert facts.mode == "P"
    assert facts.detected_format == "PNG"
    assert facts.frames == 1
    assert facts.distinct_colors == 4
    assert facts.colors_over_cap is False
    assert facts.size_bytes > 0


def test_inspect_detects_an_integer_upscale(tmp_path):
    # A 4x nearest upscale of an 8x8 grid must report scale 4, because that is what decides whether
    # a later resize can be lossless instead of smearing the pixel edges.
    base = Image.open(_sprite(tmp_path / "base.png", size=(8, 8), colors=4)).convert("RGB")
    upscaled = tmp_path / "up.png"
    base.resize((32, 32), Image.NEAREST).save(upscaled)
    assert inspect_image(upscaled).integer_scale == 4


def test_inspect_reports_no_upscale_for_a_native_grid(tmp_path):
    native = tmp_path / "native.png"
    Image.open(_sprite(tmp_path / "s.png", size=(15, 15), colors=5)).convert("RGB").save(native)
    assert inspect_image(native).integer_scale == 1


def test_inspect_reports_alpha(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (8, 8), (10, 20, 30, 128)).save(path)
    assert inspect_image(path).has_alpha is True


def test_inspect_refuses_a_truncated_file(tmp_path):
    whole = _sprite(tmp_path / "whole.png", size=(16, 16))
    broken = tmp_path / "broken.png"
    broken.write_bytes(whole.read_bytes()[: len(whole.read_bytes()) // 2])
    with pytest.raises(Exception):  # noqa: B017 - Pillow's exact type is version-dependent
        inspect_image(broken)


def test_inspect_refuses_a_non_image_with_an_image_extension(tmp_path):
    decoy = tmp_path / "decoy.png"
    decoy.write_text("<html>404</html>", encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017
        inspect_image(decoy)


# ---- problems ----------------------------------------------------------------------------------


def test_multi_frame_image_is_a_problem(tmp_path):
    path = tmp_path / "anim.gif"
    # The frames must differ in actual pixel data, or the GIF writer collapses them into one and
    # the fixture would not exercise the multi-frame path at all.
    palette = [0, 0, 0, 255, 255, 255] + [0] * 762
    frames = []
    for index in (0, 1):
        frame = Image.new("P", (8, 8), index)
        frame.putpalette(palette)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    facts = inspect_image(path)
    assert facts.frames > 1
    assert any("frames" in problem for problem in image_content_problems(facts, path))


def test_mode_needing_conversion_is_a_problem(tmp_path):
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (16, 16), (0, 0, 0, 0)).save(path)
    facts = inspect_image(path)
    assert facts.mode == "CMYK"
    assert any("conversion" in problem for problem in image_content_problems(facts, path))


def test_extension_that_contradicts_the_format_is_a_problem(tmp_path):
    # A real PNG named .jpg: decodable, so only a content check can catch it.
    mislabelled = tmp_path / "actually_png.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(mislabelled, format="PNG")
    facts = inspect_image(mislabelled)
    problems = image_content_problems(facts, mislabelled)
    assert any("does not match the detected format" in problem for problem in problems)


def test_jpg_suffix_accepts_jpeg_format(tmp_path):
    path = tmp_path / "real.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path, format="JPEG")
    assert image_content_problems(inspect_image(path), path) == []


def test_minimum_dimension_is_not_enforced_by_default(tmp_path):
    tiny = _sprite(tmp_path / "tiny.png", size=(3, 3), colors=2)
    facts = inspect_image(tiny)
    assert image_content_problems(facts, tiny) == []
    assert image_content_problems(facts, tiny, min_dimension=16) != []


# ---- CLI ---------------------------------------------------------------------------------------


def test_check_images_reports_facts_and_passes_a_clean_dataset(tmp_path):
    project = _project(tmp_path)
    root = tmp_path / "assets"
    _sprite(root / "images" / "a.png")
    data = _rows(tmp_path / "d.jsonl", [{"image": "images/a.png", "caption": "a sprite"}])

    result = runner.invoke(
        app,
        [
            "schema-validate",
            str(project),
            "sprite",
            "--data",
            str(data),
            "--path-root",
            str(root),
            "--check-images",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["image_content"]["checked"] is True
    assert payload["image_content"]["images_inspected"] == 1
    assert payload["image_content"]["checked_fields"] == ["image"]
    assert payload["image_content"]["facts"][0]["width"] == 16


def test_check_images_fails_the_row_and_exits_nonzero(tmp_path):
    project = _project(tmp_path)
    root = tmp_path / "assets"
    (root / "images").mkdir(parents=True)
    (root / "images" / "bad.png").write_text("not an image", encoding="utf-8")
    data = _rows(tmp_path / "d.jsonl", [{"image": "images/bad.png", "caption": "x"}])

    result = runner.invoke(
        app,
        [
            "schema-validate",
            str(project),
            "sprite",
            "--data",
            str(data),
            "--path-root",
            str(root),
            "--check-images",
        ],
    )
    assert result.exit_code == 1
    assert "could not be read" in result.output


def test_image_fields_are_reported_as_unchecked_without_the_flag(tmp_path):
    project = _project(tmp_path)
    root = tmp_path / "assets"
    _sprite(root / "images" / "a.png")
    data = _rows(tmp_path / "d.jsonl", [{"image": "images/a.png", "caption": "a sprite"}])

    result = runner.invoke(
        app,
        ["schema-validate", str(project), "sprite", "--data", str(data), "--path-root", str(root)],
    )
    assert result.exit_code == 0, result.output
    # The honesty requirement: a dataset whose images were never opened must not read as one whose
    # images were checked.
    assert "Image content NOT checked" in result.output


def test_check_images_requires_a_path_root(tmp_path):
    project = _project(tmp_path)
    data = _rows(tmp_path / "d.jsonl", [{"image": "images/a.png", "caption": "x"}])
    result = runner.invoke(
        app,
        ["schema-validate", str(project), "sprite", "--data", str(data), "--check-images"],
    )
    assert result.exit_code == 1
    assert "--check-images needs --path-root" in result.output


def test_an_uncontained_image_path_is_never_opened(tmp_path):
    """A row naming a path outside the root is refused, not inspected.

    Dataset content is untrusted. Opening an uncontained path would let a third-party JSONL file
    read arbitrary files on the validating machine, which is a strictly larger exposure than the
    existence probe the same boundary already refuses.
    """
    project = _project(tmp_path)
    root = tmp_path / "assets"
    (root / "images").mkdir(parents=True)
    outside = tmp_path / "secret.png"
    _sprite(outside)
    data = _rows(tmp_path / "d.jsonl", [{"image": "../secret.png", "caption": "x"}])

    result = runner.invoke(
        app,
        [
            "schema-validate",
            str(project),
            "sprite",
            "--data",
            str(data),
            "--path-root",
            str(root),
            "--check-images",
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["image_content"]["images_inspected"] == 0
    assert any("escapes the path root" in e["message"] for e in payload["row_errors"])


def test_absolute_image_path_is_refused(tmp_path):
    project = _project(tmp_path)
    root = tmp_path / "assets"
    (root / "images").mkdir(parents=True)
    target = _sprite(tmp_path / "abs.png")
    data = _rows(tmp_path / "d.jsonl", [{"image": str(target), "caption": "x"}])

    result = runner.invoke(
        app,
        [
            "schema-validate",
            str(project),
            "sprite",
            "--data",
            str(data),
            "--path-root",
            str(root),
            "--check-images",
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["image_content"]["images_inspected"] == 0


def test_missing_extra_refuses_rather_than_reporting_a_pass(tmp_path, monkeypatch):
    """With the extra absent the run refuses. It must never report a pass for checks that did not run."""
    import corpus_studio.quality.images as images_module

    def _no_pillow() -> None:
        raise ImageSupportUnavailable(
            "Image content inspection needs the optional 'images' extra. "
            "Install it with: pip install corpus-studio-engine[images]"
        )

    monkeypatch.setattr(images_module, "ensure_image_support", _no_pillow)

    project = _project(tmp_path)
    root = tmp_path / "assets"
    _sprite(root / "images" / "a.png")
    data = _rows(tmp_path / "d.jsonl", [{"image": "images/a.png", "caption": "a sprite"}])

    result = runner.invoke(
        app,
        [
            "schema-validate",
            str(project),
            "sprite",
            "--data",
            str(data),
            "--path-root",
            str(root),
            "--check-images",
        ],
    )
    assert result.exit_code == 1
    assert "images" in result.output


def test_image_support_boundary():
    """Importing the inspection module must pull no image library.

    Mirrors ``test_platform_import_is_torch_free``: the engine core stays dependency-light, and an
    optional extra is only imported at the call site.
    """
    for module in [name for name in sys.modules if name == "PIL" or name.startswith("PIL.")]:
        del sys.modules[module]
    import importlib

    importlib.reload(importlib.import_module("corpus_studio.quality.images"))
    assert "PIL" not in sys.modules, "importing quality.images pulled PIL"
