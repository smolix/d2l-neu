#!/usr/bin/env python3
"""Tight-crop standalone SVGs to their rendered content.

The JS diagram pipeline (diagrams/*.mjs) draws on fixed canvases whose
margins were hand-guessed, so many figures carry dead borders — and dead
canvas means the page scales the figure down and the type shrinks with it
(docs/figure-style-guide.md §6, the density rule).  This tool rasterizes an
SVG with rsvg-convert, measures the non-white bounding box with PIL, and
rewrites the SVG's viewBox/width/height to hug the content plus PAD.

Deterministic: rsvg output is deterministic for a given SVG, and the bbox
is rounded to integers, so re-running is byte-idempotent.

Usage:  python3 tools/figstyle/tightcrop.py FILE.svg [FILE2.svg ...]
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys

from PIL import Image

PAD = 8.0      # content padding in viewBox units
SCALE = 2      # raster oversampling for bbox precision


def crop(path: str) -> str:
    with open(path) as fh:
        svg = fh.read()

    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"', svg)
    if not m:
        return f"{path}: no viewBox — skipped"
    vx, vy, vw, vh = (float(g) for g in m.groups())

    png = subprocess.run(
        ["rsvg-convert", "-z", str(SCALE), path],
        check=True, capture_output=True).stdout
    # rsvg renders a TRANSPARENT background: composite over white before
    # diffing, or the alpha channel makes the whole canvas look like content.
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    im = Image.alpha_composite(white, rgba).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    from PIL import ImageChops
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return f"{path}: blank — skipped"

    # pixel bbox -> viewBox units (raster spans the whole viewBox)
    sx, sy = vw / im.size[0], vh / im.size[1]
    nx = max(vx, vx + bbox[0] * sx - PAD)
    ny = max(vy, vy + bbox[1] * sy - PAD)
    nx2 = min(vx + vw, vx + bbox[2] * sx + PAD)
    ny2 = min(vy + vh, vy + bbox[3] * sy + PAD)
    nw, nh = round(nx2 - nx), round(ny2 - ny)
    nx, ny = round(nx), round(ny)

    if (nx, ny, nw, nh) == (vx, vy, vw, vh):
        return f"{path}: already tight"

    svg = svg.replace(m.group(0), f'viewBox="{nx} {ny} {nw} {nh}"')
    # keep explicit width/height (if present) in sync with the viewBox
    svg = re.sub(r'width="[\d.]+" height="[\d.]+"',
                 f'width="{nw}" height="{nh}"', svg, count=1)
    with open(path, "w") as fh:
        fh.write(svg)
    return f"{path}: {vw:.0f}x{vh:.0f} -> {nw}x{nh}"


def main() -> None:
    for path in sys.argv[1:]:
        print(crop(path))


if __name__ == "__main__":
    main()
