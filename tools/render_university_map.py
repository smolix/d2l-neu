#!/usr/bin/env python3
"""Render a static SVG world map of D2L university adoption.

The renderer uses latitude/longitude stored in tools/universities.json. It does
not call any map APIs and does not require runtime dependencies.
"""

import json
import math
from pathlib import Path

REPO = Path("/home/smola/d2l/d2l-neu")
UNIVERSITIES = REPO / "tools" / "universities.json"
OUT = REPO / "static" / "landing" / "universities-map.svg"

WIDTH = 1600
HEIGHT = 820


LAND = [
    # North America
    [(-168, 72), (-140, 70), (-125, 54), (-118, 49), (-101, 50), (-82, 46), (-67, 48), (-52, 58), (-60, 68), (-96, 74), (-130, 72)],
    [(-124, 49), (-111, 32), (-99, 26), (-82, 25), (-80, 31), (-92, 41), (-105, 49)],
    [(-86, 22), (-78, 9), (-83, 8), (-91, 16), (-98, 20)],
    # South America
    [(-81, 12), (-66, 8), (-50, -2), (-36, -14), (-44, -24), (-53, -36), (-60, -54), (-72, -50), (-76, -30), (-80, -12)],
    # Europe
    [(-10, 36), (2, 44), (14, 46), (30, 59), (42, 56), (32, 45), (20, 40), (10, 36)],
    [(-8, 58), (2, 61), (8, 56), (2, 50), (-6, 51)],
    # Africa
    [(-18, 35), (10, 37), (34, 31), (51, 12), (42, -18), (30, -34), (18, -35), (4, -28), (-10, -6), (-17, 15)],
    # Asia
    [(28, 40), (45, 55), (72, 58), (94, 55), (120, 50), (145, 45), (150, 30), (132, 18), (112, 20), (100, 8), (78, 22), (62, 25), (45, 20), (34, 32)],
    [(67, 24), (78, 8), (89, 20), (80, 30)],
    [(96, 15), (106, 8), (103, -4), (94, 4)],
    [(116, 6), (126, 2), (124, -7), (113, -4)],
    [(139, 45), (146, 38), (142, 32), (136, 34)],
    # Australia / New Zealand
    [(112, -11), (154, -12), (153, -38), (141, -44), (115, -34)],
    [(166, -35), (178, -38), (174, -46), (166, -45)],
    # Greenland
    [(-52, 60), (-31, 68), (-22, 78), (-44, 83), (-62, 76), (-72, 66)],
]


def project(lat: float, lon: float) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * WIDTH
    # Web Mercator, clipped to keep poles finite.
    lat = max(min(lat, 82.0), -82.0)
    rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * HEIGHT
    return x, y


def path_for(poly: list[tuple[float, float]]) -> str:
    pts = [project(lat, lon) for lon, lat in poly]
    head, *tail = pts
    d = [f"M {head[0]:.1f} {head[1]:.1f}"]
    d.extend(f"L {x:.1f} {y:.1f}" for x, y in tail)
    d.append("Z")
    return " ".join(d)


def main() -> None:
    entries = json.loads(UNIVERSITIES.read_text(encoding="utf-8"))
    located = [e for e in entries if e.get("location") and e["location"].get("lat") is not None and e["location"].get("lon") is not None]

    dots = []
    for e in located:
        loc = e["location"]
        x, y = project(float(loc["lat"]), float(loc["lon"]))
        name = (
            e["name"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2"><title>{name}</title></circle>')

    land_paths = "\n    ".join(f'<path d="{path_for(poly)}"/>' for poly in LAND)
    dot_markup = "\n    ".join(dots)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Universities teaching Dive into Deep Learning">
  <defs>
    <radialGradient id="ocean" cx="50%" cy="45%" r="75%">
      <stop offset="0%" stop-color="#f8fbff"/>
      <stop offset="100%" stop-color="#e6edf7"/>
    </radialGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1" stdDeviation="1.6" flood-color="#7f1d1d" flood-opacity="0.35"/>
    </filter>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#ocean)"/>
  <g fill="#d7e2d1" stroke="#ffffff" stroke-width="1.4" opacity="0.98">
    {land_paths}
  </g>
  <g fill="none" stroke="#c8d5e5" stroke-width="0.8" opacity="0.45">
    {''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="{HEIGHT}"/>' for x in range(0, WIDTH + 1, 200))}
    {''.join(f'<line x1="0" y1="{y}" x2="{WIDTH}" y2="{y}"/>' for y in range(100, HEIGHT, 120))}
  </g>
  <g fill="#df2b2b" stroke="#ffffff" stroke-width="1.1" filter="url(#shadow)" opacity="0.92">
    {dot_markup}
  </g>
  <text x="48" y="70" font-family="Arial, Helvetica, sans-serif" font-size="34" font-weight="700" fill="#203040">Universities teaching Dive into Deep Learning</text>
  <text x="48" y="106" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="#526173">{len(located)} located universities from tools/universities.json</text>
</svg>
'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} with {len(located)} dots")


if __name__ == "__main__":
    main()
