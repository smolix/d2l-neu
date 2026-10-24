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
import math
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

    # pixel bbox -> viewBox units (raster spans the whole viewBox).  Each
    # edge is rounded OUTWARD to an integer (padding never drops below PAD),
    # which also makes a re-crop of a cropped file a no-op: the raster grid
    # then stays aligned and the clamped edges reproduce themselves.
    sx, sy = vw / im.size[0], vh / im.size[1]
    nx = max(vx, math.floor(vx + bbox[0] * sx - PAD))
    ny = max(vy, math.floor(vy + bbox[1] * sy - PAD))
    nx2 = min(vx + vw, math.ceil(vx + bbox[2] * sx + PAD))
    ny2 = min(vy + vh, math.ceil(vy + bbox[3] * sy + PAD))
    nx, ny, nw, nh = (_num(v) for v in (nx, ny, nx2 - nx, ny2 - ny))

    if (nx, ny, nw, nh) == (vx, vy, vw, vh):
        return f"{path}: already tight"

    svg = svg.replace(m.group(0), f'viewBox="{nx} {ny} {nw} {nh}"', 1)
    # keep the ROOT element's explicit width/height (if present) in sync with
    # the viewBox, preserving its unit and scale (matplotlib writes "pt");
    # never touch a width/height pair further down (e.g. a clipPath rect).
    root = re.search(r"<svg\b[^>]*>", svg)
    tag = root.group(0)
    new_tag = tag
    for attr, old, new in (("width", vw, nw), ("height", vh, nh)):
        a = re.search(rf'\s{attr}="([\d.]+)([a-z%]*)"', tag)
        if a:
            val = _num(float(a.group(1)) * new / old)
            new_tag = new_tag.replace(a.group(0), f' {attr}="{val}{a.group(2)}"', 1)
    svg = svg[:root.start()] + new_tag + svg[root.end():]
    with open(path, "w") as fh:
        fh.write(svg)
    return f"{path}: {vw:g}x{vh:g} -> {nw}x{nh}"


def _num(v: float):
    """int when integral, else a short fixed-precision float (deterministic)."""
    r = round(v, 6)
    return int(r) if r == int(r) else float(f"{r:.6f}")


def main() -> None:
    for path in sys.argv[1:]:
        print(crop(path))


if __name__ == "__main__":
    main()
