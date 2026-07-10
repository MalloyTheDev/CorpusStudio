"""Generate the CorpusStudio app icon (.ico + .png) from the branding design.

Draws the "CS" monogram on the blue gradient rounded square that matches the in-app
avatar, and writes a multi-size Windows .ico plus a 256px .png. Run after editing the
design (kept in sync with assets/branding/corpusstudio.svg):

    python tools/generate_icon.py

Pillow is a BUILD-TIME tool only (not an app/engine runtime dependency); the committed
.ico / .png are what the heads consume. Install it ad hoc: `pip install pillow`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 256
RADIUS = 56
TOP = (0x25, 0x63, 0xEB)  # #2563EB
BOTTOM = (0x1D, 0x4E, 0xD8)  # #1D4ED8
TEXT = "CS"
FONT_CANDIDATES = ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf")
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

REPO_ROOT = Path(__file__).resolve().parent.parent
BRANDING = REPO_ROOT / "assets" / "branding"
# Both heads consume a local copy so the csproj/axaml references stay simple.
ICO_TARGETS = [
    BRANDING / "corpusstudio.ico",
    REPO_ROOT / "apps" / "desktop" / "CorpusStudio.Desktop" / "app.ico",
    REPO_ROOT / "apps" / "desktop" / "CorpusStudio.Avalonia" / "app.ico",
]
PNG_TARGET = BRANDING / "corpusstudio-256.png"


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def render(size: int = SIZE) -> Image.Image:
    scale = size / SIZE
    # Vertical gradient.
    gradient = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / (size - 1)
        gd.line(
            [(0, y), (size, y)],
            fill=tuple(round(TOP[i] * (1 - t) + BOTTOM[i] * t) for i in range(3)) + (255,),
        )

    # Rounded-square mask.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=round(RADIUS * scale), fill=255)

    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(gradient, (0, 0), mask)

    # Centered monogram.
    draw = ImageDraw.Draw(icon)
    font = _load_font(round(132 * scale))
    box = draw.textbbox((0, 0), TEXT, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((size - tw) / 2 - box[0], (size - th) / 2 - box[1]), TEXT, font=font, fill=(255, 255, 255, 255))
    return icon


def main() -> None:
    master = render(SIZE)
    BRANDING.mkdir(parents=True, exist_ok=True)
    master.save(PNG_TARGET)
    for target in ICO_TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        master.save(target, format="ICO", sizes=ICO_SIZES)
        print(f"wrote {target.relative_to(REPO_ROOT)}")
    print(f"wrote {PNG_TARGET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
