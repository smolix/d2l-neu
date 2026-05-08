#!/usr/bin/env python3
"""Render a one-shot static university map using a Google Maps background.

This script intentionally does not add the generated image to the page. It
downloads a single Google Maps Static API background, then overlays the red
university dots locally using coordinates from tools/universities.json.

Required:
  GOOGLE_MAPS_API_KEY with Maps Static API enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
UNIVERSITIES = REPO / "tools" / "universities.json"
OUT_DIR = REPO / "static" / "landing"
BACKGROUND = OUT_DIR / "universities-map-google-background.png"
OUT = OUT_DIR / "universities-map-google.png"
OUT_SVG = OUT_DIR / "universities-map-google.svg"

WIDTH = 1280
HEIGHT = 674
STATIC_SIZE = "640x337"
STATIC_SCALE = 2
FINAL_WIDTH = 1536
FINAL_HEIGHT = 810
CROP_BOX = (128, 90, 1152, 630)


def mercator_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Return global pixel coordinates in Web Mercator at the scaled output size."""
    siny = math.sin(math.radians(max(min(lat, 85.05112878), -85.05112878)))
    world = 256 * (2**zoom) * STATIC_SCALE
    x = (lon + 180.0) / 360.0 * world
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world
    return x, y


def local_xy(lat: float, lon: float, center_lat: float, center_lon: float, zoom: int) -> tuple[float, float]:
    cx, cy = mercator_xy(center_lat, center_lon, zoom)
    px, py = mercator_xy(lat, lon, zoom)
    return WIDTH / 2 + (px - cx), HEIGHT / 2 + (py - cy)


def static_map_url(api_key: str, center: tuple[float, float], zoom: int, maptype: str) -> str:
    params = {
        "center": f"{center[0]},{center[1]}",
        "zoom": str(zoom),
        "size": STATIC_SIZE,
        "scale": str(STATIC_SCALE),
        "maptype": maptype,
        "format": "png",
        "key": api_key,
    }
    # Quiet overlays so dense red dots remain legible. Terrain still preserves
    # relief/color, while administrative/date-line geometry is suppressed.
    params["style"] = [
        "feature:poi|visibility:off",
        "feature:transit|visibility:off",
        "feature:administrative|visibility:off",
        "feature:administrative.country|element:geometry.stroke|visibility:off",
        "feature:administrative.province|visibility:off",
        "feature:road|visibility:off",
    ]
    if maptype == "roadmap":
        params["style"].extend(
            [
                "feature:landscape|color:0xf4f0e8",
                "feature:water|color:0xcddce9",
            ]
        )
    return "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params, doseq=True)


def load_locations() -> list[tuple[str, float, float]]:
    entries = json.loads(UNIVERSITIES.read_text(encoding="utf-8"))
    rows = []
    for entry in entries:
        loc = entry.get("location") or {}
        if loc.get("lat") is None or loc.get("lon") is None:
            continue
        rows.append((entry["name"], float(loc["lat"]), float(loc["lon"])))
    return rows


def final_xy(lat: float, lon: float, center: tuple[float, float], zoom: int) -> tuple[float, float]:
    x, y = local_xy(lat, lon, center[0], center[1], zoom)
    left, top, right, bottom = CROP_BOX
    return (
        (x - left) * FINAL_WIDTH / (right - left),
        (y - top) * FINAL_HEIGHT / (bottom - top),
    )


def crop_upscale_image(image) -> object:
    from PIL import Image

    image = image.convert("RGBA")
    image = image.crop(CROP_BOX)
    return image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)


def draw_dots(background: Path, output: Path, rows: list[tuple[str, float, float]], center: tuple[float, float], zoom: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required for PNG compositing. Use `.venv/bin/python` or `uv run python`.") from exc

    image = Image.open(background).convert("RGBA")
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image = crop_upscale_image(image)

    scale = 4
    overlay = Image.new("RGBA", (FINAL_WIDTH * scale, FINAL_HEIGHT * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for _, lat, lon in rows:
        x, y = final_xy(lat, lon, center, zoom)
        if -12 <= x <= FINAL_WIDTH + 12 and -12 <= y <= FINAL_HEIGHT + 12:
            sx, sy = x * scale, y * scale
            shadow_r = 6.8 * scale
            outer_r = 5.6 * scale
            inner_r = 3.9 * scale
            draw.ellipse(
                (sx - shadow_r + 1.5 * scale, sy - shadow_r + 2.2 * scale, sx + shadow_r + 1.5 * scale, sy + shadow_r + 2.2 * scale),
                fill=(72, 20, 20, 46),
            )
            draw.ellipse((sx - outer_r, sy - outer_r, sx + outer_r, sy + outer_r), fill=(255, 255, 255, 235))
            draw.ellipse((sx - inner_r, sy - inner_r, sx + inner_r, sy + inner_r), fill=(222, 37, 43, 232))
            edge_r = inner_r - 0.25 * scale
            draw.ellipse((sx - edge_r, sy - edge_r, sx + edge_r, sy + edge_r), outline=(150, 14, 20, 175), width=max(1, int(0.55 * scale)))

    overlay = overlay.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)
    image.alpha_composite(overlay)
    add_attribution(image)
    image.convert("RGB").save(output, quality=95)


def add_attribution(image) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
        small = font

    width, height = image.size
    draw.rounded_rectangle((12, height - 56, 160, height - 12), radius=6, fill=(255, 255, 255, 190))
    draw.text((22, height - 54), "Google", fill=(80, 80, 80, 255), font=font)
    label = "Map data ©2026 Google"
    bbox = draw.textbbox((0, 0), label, font=small)
    text_width = bbox[2] - bbox[0]
    draw.rectangle((width - text_width - 24, height - 48, width, height), fill=(255, 255, 255, 205))
    draw.text((width - text_width - 12, height - 43), label, fill=(70, 70, 70, 255), font=small)


def crop_upscale(path: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required for PNG postprocessing. Use `.venv/bin/python` or `uv run python`.") from exc

    image = crop_upscale_image(Image.open(path))
    add_attribution(image)
    image.convert("RGB").save(path, quality=95)


def write_svg_overlay(background: Path, output: Path, rows: list[tuple[str, float, float]], center: tuple[float, float], zoom: int) -> None:
    dots = []
    for name, lat, lon in rows:
        x, y = local_xy(lat, lon, center[0], center[1], zoom)
        if -8 <= x <= WIDTH + 8 and -8 <= y <= HEIGHT + 8:
            title = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5">'
                f"<title>{title}</title></circle>"
            )
    rel = background.relative_to(output.parent)
    output.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Universities teaching Dive into Deep Learning on Google Maps background">
  <image href="{rel.as_posix()}" x="0" y="0" width="{WIDTH}" height="{HEIGHT}" preserveAspectRatio="none"/>
  <g fill="#d62728" stroke="#ffffff" stroke-width="2" opacity="0.88">
    {"".join(dots)}
  </g>
</svg>
''',
        encoding="utf-8",
    )


def fetch_background(api_key: str, background: Path, center: tuple[float, float], zoom: int, maptype: str) -> None:
    url = static_map_url(api_key, center, zoom, maptype)
    request = urllib.request.Request(url, headers={"User-Agent": "d2l-university-map/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        sys.stderr.write(body[:1200] + "\n")
        raise SystemExit(f"Google Maps Static API request failed with HTTP {exc.code}.") from exc
    if "image" not in content_type:
        sys.stderr.write(body.decode("utf-8", "replace")[:1200] + "\n")
        raise SystemExit("Google Maps Static API did not return an image.")
    background.write_bytes(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", default="18,0", help="Map center as lat,lon.")
    parser.add_argument("--zoom", type=int, default=1)
    parser.add_argument("--maptype", default="terrain", choices=["roadmap", "terrain", "satellite", "hybrid"])
    parser.add_argument("--background-only", action="store_true")
    parser.add_argument("--reuse-background", action="store_true")
    parser.add_argument("--svg-only", action="store_true", help="Write SVG overlay instead of composited PNG.")
    args = parser.parse_args()

    try:
        center_lat, center_lon = (float(v.strip()) for v in args.center.split(",", 1))
    except ValueError as exc:
        raise SystemExit("--center must be formatted as lat,lon") from exc

    rows = load_locations()
    if len(rows) == 0:
        raise SystemExit("No university locations found.")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key and not args.reuse_background:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY to fetch a Google Maps background.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.reuse_background:
        fetch_background(api_key or "", BACKGROUND, (center_lat, center_lon), args.zoom, args.maptype)
        print(f"wrote {BACKGROUND}")

    if args.background_only:
        return

    if not BACKGROUND.exists():
        raise SystemExit(f"Missing background image: {BACKGROUND}")
    if args.svg_only:
        write_svg_overlay(BACKGROUND, OUT_SVG, rows, (center_lat, center_lon), args.zoom)
        print(f"wrote {OUT_SVG} with {len(rows)} dots")
    else:
        draw_dots(BACKGROUND, OUT, rows, (center_lat, center_lon), args.zoom)
        crop_upscale(BACKGROUND)
        print(f"wrote {OUT} with {len(rows)} dots")


if __name__ == "__main__":
    main()
