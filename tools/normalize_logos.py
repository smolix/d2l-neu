#!/usr/bin/env python3
"""Normalize every logo in static/landing/universities/ to a 256x256 PNG.

For each file:
  1. Load (rasterize SVG via rsvg-convert at width 1024).
  2. Compute content bbox = pixels that are NOT (alpha < 16 OR
     near-white where R>240, G>240, B>240).
  3. Crop to bbox, pad short axis with transparent to square.
  4. Resize to 256x256 (bicubic).
  5. Save as <slug>.png. If source extension != .png, delete original.

After processing, update tools/universities.json so each `logo` field
points to <slug>.png when that file exists on disk.

Usage: python3 tools/normalize_logos.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
LOGO_DIR = REPO / "static" / "landing" / "universities"
JSON_PATH = REPO / "tools" / "universities.json"

TARGET_SIZE = 256
SVG_RASTER_WIDTH = 1024
ALPHA_THRESHOLD = 16
NEAR_WHITE = 240


def rasterize_svg(svg_path: Path) -> Path | None:
    """Rasterize SVG to a tmp PNG. Return path or None on failure."""
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="logonorm-")
    os.close(fd)
    try:
        result = subprocess.run(
            [
                "rsvg-convert",
                "-w",
                str(SVG_RASTER_WIDTH),
                str(svg_path),
                "-o",
                tmp,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None
        return Path(tmp)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None


def content_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of "content" pixels.

    Content = NOT (alpha < ALPHA_THRESHOLD OR
                   (R>NEAR_WHITE AND G>NEAR_WHITE AND B>NEAR_WHITE)).
    """
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()

    # Mask of content: start with alpha >= threshold
    # then AND NOT near-white.
    # Pillow's point() returns L mode.
    alpha_mask = a.point(lambda v: 255 if v >= ALPHA_THRESHOLD else 0)
    r_mask = r.point(lambda v: 255 if v > NEAR_WHITE else 0)
    g_mask = g.point(lambda v: 255 if v > NEAR_WHITE else 0)
    b_mask = b.point(lambda v: 255 if v > NEAR_WHITE else 0)

    from PIL import ImageChops

    near_white = ImageChops.multiply(ImageChops.multiply(r_mask, g_mask), b_mask)
    # not_near_white
    not_white = near_white.point(lambda v: 0 if v == 255 else 255)
    content = ImageChops.multiply(alpha_mask, not_white)
    return content.getbbox()


def normalize_one(path_str: str) -> tuple[str, bool, str]:
    """Process one file. Returns (slug, success, message)."""
    path = Path(path_str)
    slug = path.stem
    ext = path.suffix.lower()

    try:
        if ext == ".svg":
            raster = rasterize_svg(path)
            if raster is None:
                return slug, False, f"rsvg-convert failed for {path.name}"
            try:
                img = Image.open(raster)
                img.load()
            finally:
                try:
                    os.unlink(raster)
                except OSError:
                    pass
        else:
            img = Image.open(path)
            img.load()

        img = img.convert("RGBA")

        bbox = content_bbox(img)
        if bbox is None:
            # No non-white opaque pixels (e.g. all-white-on-transparent
            # logo). Fall back to alpha-only bbox so we at least keep the
            # logo intact.
            bbox = img.split()[-1].getbbox()
        if bbox is None:
            return slug, False, f"empty bbox in {path.name}"

        cropped = img.crop(bbox)
        w, h = cropped.size

        # Pad to square (transparent).
        side = max(w, h)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(cropped, ((side - w) // 2, (side - h) // 2))

        # Resize to 256x256 with bicubic.
        out = canvas.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)

        out_path = LOGO_DIR / f"{slug}.png"
        out.save(out_path, format="PNG", optimize=True)

        # Delete original if it was not the .png we just wrote.
        if ext != ".png" and path.exists() and path.resolve() != out_path.resolve():
            try:
                path.unlink()
            except OSError as e:
                return slug, True, f"normalized but failed to remove original {path.name}: {e}"

        return slug, True, ""
    except Exception as e:  # noqa: BLE001
        return slug, False, f"{type(e).__name__}: {e}"


def update_json() -> int:
    """Update universities.json so each entry's `logo` points at <slug>.png if present."""
    with open(JSON_PATH) as f:
        data = json.load(f)

    changed = 0
    for entry in data:
        slug = entry.get("slug")
        if not slug:
            continue
        png_path = LOGO_DIR / f"{slug}.png"
        if png_path.exists():
            new_logo = f"{slug}.png"
            if entry.get("logo") != new_logo:
                entry["logo"] = new_logo
                changed += 1

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return changed


def main() -> int:
    files = sorted(
        p for p in LOGO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    )
    print(f"Processing {len(files)} files...", flush=True)

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor() as ex:
        futs = {ex.submit(normalize_one, str(p)): p for p in files}
        for fut in as_completed(futs):
            slug, ok, msg = fut.result()
            if ok:
                successes.append(slug)
                if msg:
                    print(f"  WARN {slug}: {msg}", flush=True)
            else:
                failures.append((slug, msg))
                print(f"  FAIL {slug}: {msg}", flush=True)

    print(f"\nNormalized: {len(successes)}")
    print(f"Failed:     {len(failures)}")
    if failures:
        print("Failed slugs:")
        for slug, msg in failures:
            print(f"  - {slug}: {msg}")

    changed = update_json()
    print(f"\nuniversities.json: {changed} logo fields updated.")
    return 0 if not failures else 0  # do not fail the process; user wants to review


if __name__ == "__main__":
    sys.exit(main())
