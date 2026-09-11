#!/usr/bin/env python3
"""
Generate BetLedger's app icons from the palette in `src/theme/tokens.ts`.

The mark is the app's own signature: a rising profit curve over a dashed zero line,
which is exactly what the dashboard shows. Re-run after changing the colours below:

    python3 scripts/generate-icons.py

Outputs (all under assets/):
    icon.png                      1024x1024 RGB, no alpha  — iOS app icon
    android-icon-background.png    512x512  RGB            — adaptive icon background
    android-icon-foreground.png    512x512  RGBA           — adaptive icon foreground
    android-icon-monochrome.png    432x432  RGBA, white    — Android 13+ themed icon
    splash-icon.png               1024x1024 RGBA           — splash mark
    favicon.png                     48x48   RGBA           — web favicon
"""

from __future__ import annotations

import os
from PIL import Image, ImageDraw

# --- Palette (matches src/theme/tokens.ts) ----------------------------------
INK_TOP = (19, 28, 43)      # #131C2B
INK_BOTTOM = (8, 12, 19)    # #080C13
MINT = (37, 211, 160)       # #25D3A0 — colors.primary
GLOW = (37, 211, 160)

# --- The mark ---------------------------------------------------------------
# A realistic P/L curve — up, with one drawdown. Points are in a unit space and
# are fitted to the content box at render time, so tweaking them cannot clip.
CURVE = [(0.00, 1.00), (0.26, 0.44), (0.45, 0.62), (0.72, 0.14), (1.00, 0.00)]

SS = 4  # supersampling factor; everything is drawn big and downsampled

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def gradient_background(size: int) -> Image.Image:
    """Vertical ink gradient with a soft mint glow behind the curve's peak."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    assert pixels is not None
    for y in range(size):
        t = y / max(size - 1, 1)
        row = (
            lerp(INK_TOP[0], INK_BOTTOM[0], t),
            lerp(INK_TOP[1], INK_BOTTOM[1], t),
            lerp(INK_TOP[2], INK_BOTTOM[2], t),
        )
        for x in range(size):
            pixels[x, y] = row

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_pixels = glow.load()
    assert glow_pixels is not None
    cx, cy = size * 0.74, size * 0.30
    radius = size * 0.62
    for y in range(size):
        for x in range(size):
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance >= radius:
                continue
            falloff = (1 - distance / radius) ** 3
            glow_pixels[x, y] = (*GLOW, int(round(52 * falloff)))

    image = Image.alpha_composite(image.convert("RGBA"), glow)
    return image.convert("RGB")


def draw_mark(
    canvas: Image.Image,
    inset: float,
    color: tuple[int, int, int],
    *,
    area_fill: bool,
) -> None:
    """Draw the profit curve into `canvas`, inset by `inset` of the canvas size."""
    size = canvas.size[0]
    stroke = max(int(round(size * (1 - 2 * inset) * 0.088)), 2)
    cap = stroke / 2
    head = stroke * 0.62  # the "latest point" marker, sized not to read as a pin

    # Fit the unit curve inside the content box, leaving room for the round caps
    # and the head marker so nothing touches the edge.
    margin = max(cap, head)
    left = size * inset + margin
    span = size * (1 - 2 * inset) - 2 * margin
    points = [(left + x * span, left + y * span) for x, y in CURVE]
    floor = left + span + cap

    if area_fill:
        # Mirrors the dashboard chart: mint fading to nothing at the baseline.
        polygon = [*points, (points[-1][0], floor), (points[0][0], floor)]
        mask = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)

        top = min(y for _, y in points)
        gradient = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        gradient_pixels = gradient.load()
        assert gradient_pixels is not None
        for y in range(canvas.size[1]):
            t = (y - top) / max(floor - top, 1)
            alpha = int(round(96 * max(0.0, 1.0 - t) ** 1.5))
            if alpha <= 0:
                continue
            row = (*color, alpha)
            for x in range(canvas.size[0]):
                gradient_pixels[x, y] = row

        canvas.alpha_composite(Image.composite(gradient, Image.new("RGBA", canvas.size, (0, 0, 0, 0)), mask))

    draw = ImageDraw.Draw(canvas)
    draw.line(points, fill=(*color, 255), width=stroke, joint="curve")
    # PIL leaves square caps, so round every vertex by hand.
    for px, py in points:
        draw.ellipse([px - cap, py - cap, px + cap, py + cap], fill=(*color, 255))

    # Emphasise the latest point, as the dashboard chart does.
    hx, hy = points[-1]
    draw.ellipse([hx - head, hy - head, hx + head, hy + head], fill=(*color, 255))


def render(size: int, *, background: bool, inset: float, color=MINT,
           area_fill=False) -> Image.Image:
    big = size * SS
    if background:
        canvas = gradient_background(big).convert("RGBA")
    else:
        canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw_mark(canvas, inset, color, area_fill=area_fill)
    return canvas.resize((size, size), Image.LANCZOS)


def save(image: Image.Image, name: str, *, rgb: bool = False) -> None:
    path = os.path.join(ASSETS, name)
    if rgb:
        # iOS rejects icons with an alpha channel.
        flat = Image.new("RGB", image.size, INK_BOTTOM)
        flat.paste(image, mask=image.split()[3])
        flat.save(path, "PNG")
    else:
        image.save(path, "PNG")
    print(f"  {name}  {image.size[0]}x{image.size[1]}")


def main() -> None:
    print("Writing icons to assets/")

    # iOS / store icon: full bleed, no alpha, no rounded corners (Apple masks it).
    save(render(1024, background=True, inset=0.18, area_fill=False), "icon.png", rgb=True)

    # Android adaptive icon. The launcher masks to the central 66%, so the
    # foreground needs far more padding than the iOS icon.
    save(
        gradient_background(512 * SS).resize((512, 512), Image.LANCZOS).convert("RGBA"),
        "android-icon-background.png",
        rgb=True,
    )
    save(render(512, background=False, inset=0.29, area_fill=False), "android-icon-foreground.png")

    # Themed icon: a flat white silhouette the system tints itself.
    save(
        render(
            432,
            background=False,
            inset=0.30,
            color=(255, 255, 255),
            area_fill=False,
        ),
        "android-icon-monochrome.png",
    )

    # Splash: the mark alone, centred on the splash background colour.
    save(render(1024, background=False, inset=0.24, area_fill=False), "splash-icon.png")

    # Web favicon.
    save(render(48, background=True, inset=0.16, area_fill=False), "favicon.png")


if __name__ == "__main__":
    main()
