"""Image content inspection for datasets whose schema declares an ``image_path`` field.

Schema validation proves a declared image path is a contained, existing file; it never opens it. A
row therefore passes today when the file is a truncated PNG, a 3-pixel thumbnail, a CMYK JPEG, or an
HTML error page saved with a ``.png`` extension. A dataset that validates clean and then fails inside
a trainer is what the honesty invariants exist to prevent, so this module opens the image and reports
what it actually is.

Two boundaries matter here and neither is optional:

**Containment.** Every path reaches this module already resolved by
``validators.basic_validator.resolve_contained_path``. Dataset content is untrusted, and a module
that OPENS files is a strictly larger exposure than one that only probes existence, so the same
trust boundary applies and is never reimplemented locally.

**Dependency-light.** Pillow is an optional extra (``corpus-studio-engine[images]``), imported at
the call site only. Importing this module pulls no image library, exactly as importing
``corpus_studio.platform`` pulls no torch. When the extra is absent the caller gets
``ImageSupportUnavailable`` and refuses; it never reports a pass for checks that did not run.

Pixel-art datasets (the first consumer) need two facts a generic image check would not produce: the
palette size, because a sprite is usually palette-quantised, and whether the image is already an
integer upscale of a smaller grid, because that decides whether a later resize can be lossless
instead of smearing the pixel edges.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Inspecting the palette and the integer scale touches every pixel, so both are skipped above this
# budget and REPORTED as not attempted rather than silently omitted. A sprite sheet is far below it;
# a 4K photograph is not, and spending minutes per row to characterise one is not a useful trade.
PIXEL_BUDGET = 4_000_000

# Largest integer upscale factor considered. Pixel art is authored at small grids and scaled by
# modest factors; searching further costs a full pixel pass per candidate for no realistic gain.
MAX_SCALE_FACTOR = 64

# Modes a diffusion or vision pipeline cannot consume without an explicit conversion. Palette ("P")
# is deliberately absent: it is normal, often preferred, for pixel art.
UNUSABLE_MODES = frozenset({"CMYK", "YCbCr", "I", "F", "I;16", "I;16B", "I;16L", "LAB", "HSV"})

ALPHA_MODES = frozenset({"LA", "PA", "RGBA", "La", "RGBa"})

# Suffixes whose spelling differs from the format name Pillow reports.
_SUFFIX_ALIASES = {"jpg": "jpeg", "tif": "tiff"}


class ImageSupportUnavailable(RuntimeError):
    """Raised when image inspection is requested but the optional ``images`` extra is absent."""


class ImageFacts(BaseModel):
    """What the image measurably is. A field is ``None`` when its check was not attempted."""

    model_config = {"extra": "forbid"}

    width: int
    height: int
    # The format Pillow detected from the CONTENT, which is what makes an extension mismatch
    # detectable at all.
    detected_format: str | None
    mode: str
    has_alpha: bool
    frames: int
    size_bytes: int
    # Distinct colours, or None when the image has more than ``color_cap`` of them. The exact count
    # past the cap is not a fact anyone acts on, so it is reported as over-cap rather than counted.
    distinct_colors: int | None
    colors_over_cap: bool
    # 1 means no integer upscale detected. None means the check was not attempted (pixel budget).
    integer_scale: int | None


def _require_pillow() -> Any:
    """Import Pillow lazily, or refuse naming the extra to install."""
    try:
        # noqa: PLC0415 - optional extra, imported at the call site so importing this module (and
        # the engine core through it) pulls no image library.
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised by the absent-extra path
        raise ImageSupportUnavailable(
            "Image content inspection needs the optional 'images' extra. "
            "Install it with: pip install corpus-studio-engine[images]"
        ) from exc
    return Image


def ensure_image_support() -> None:
    """Raise ``ImageSupportUnavailable`` unless the optional extra is installed.

    Callers probe once up front rather than discovering the gap on the first row, so a run either
    inspects every image or refuses before touching any of them.
    """
    _require_pillow()


def _detect_integer_scale(image: Any, image_module: Any) -> int:
    """Largest k above 1 for which the image is made of uniform k-by-k blocks, else 1.

    Implemented as a nearest-neighbour round trip rather than a pixel loop: downscaling by k picks
    one pixel per block, and upscaling back reproduces the original exactly when, and only when,
    every block was uniform. That keeps the check at library speed instead of minutes per image.
    """
    width, height = image.size
    common = math.gcd(width, height)
    if common < 2:
        return 1
    reference = image.tobytes()
    factors = sorted(
        (d for d in range(2, min(common, MAX_SCALE_FACTOR) + 1) if common % d == 0),
        reverse=True,
    )
    for factor in factors:
        small = image.resize((width // factor, height // factor), image_module.NEAREST)
        if small.resize((width, height), image_module.NEAREST).tobytes() == reference:
            return factor
    return 1


def inspect_image(path: Path, *, color_cap: int = 4096) -> ImageFacts:
    """Open ``path`` and report what it is. Raises on a file that cannot be decoded.

    ``path`` MUST already be contained by ``resolve_contained_path``. This function performs no
    containment of its own and will open whatever it is handed.
    """
    image_module = _require_pillow()
    size_bytes = path.stat().st_size

    # verify() detects truncation and corruption but consumes the file object, so the image is
    # reopened afterwards to read anything from it. That reopen is Pillow's documented sequence,
    # not redundancy.
    with image_module.open(path) as probe:
        probe.verify()

    with image_module.open(path) as image:
        width, height = image.size
        mode = image.mode
        detected_format = image.format
        frames = int(getattr(image, "n_frames", 1))
        has_alpha = mode in ALPHA_MODES or "transparency" in image.info

        distinct_colors: int | None = None
        colors_over_cap = False
        integer_scale: int | None = None
        if width * height <= PIXEL_BUDGET:
            # Decode once; both remaining checks need the pixels.
            image.load()
            counted = image.getcolors(maxcolors=color_cap)
            if counted is None:
                colors_over_cap = True
            else:
                distinct_colors = len(counted)
            integer_scale = _detect_integer_scale(image, image_module)

    return ImageFacts(
        width=width,
        height=height,
        detected_format=detected_format,
        mode=mode,
        has_alpha=has_alpha,
        frames=frames,
        size_bytes=size_bytes,
        distinct_colors=distinct_colors,
        colors_over_cap=colors_over_cap,
        integer_scale=integer_scale,
    )


def image_content_problems(
    facts: ImageFacts,
    path: Path,
    *,
    min_dimension: int = 0,
) -> list[str]:
    """Problems that make the image unusable as a training sample. Facts alone are not problems.

    ``min_dimension`` defaults to 0, meaning not enforced: a size floor is a per-dataset policy, and
    failing rows against a guessed default would be the opposite of honest. Dimensions are reported
    whether or not a floor is set.
    """
    problems: list[str] = []
    if facts.width == 0 or facts.height == 0:
        problems.append("Image has a zero dimension.")
    if min_dimension > 0 and min(facts.width, facts.height) < min_dimension:
        problems.append(
            f"Image is {facts.width}x{facts.height}, below the minimum dimension {min_dimension}."
        )
    if facts.frames > 1:
        problems.append(
            f"Image has {facts.frames} frames; a multi-frame file is rarely one training sample."
        )
    if facts.mode in UNUSABLE_MODES:
        problems.append(f"Image mode '{facts.mode}' needs an explicit conversion before training.")
    suffix = path.suffix.lower().lstrip(".")
    if facts.detected_format is not None and suffix:
        expected = _SUFFIX_ALIASES.get(suffix, suffix)
        if facts.detected_format.lower() != expected:
            problems.append(
                f"Extension '.{suffix}' does not match the detected format "
                f"'{facts.detected_format}'."
            )
    return problems
