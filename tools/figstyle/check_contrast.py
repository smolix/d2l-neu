"""Validate the token palette: WCAG contrast + pairwise distinguishability.

Run:  python3 tools/figstyle/check_contrast.py

Rules enforced (see docs/figure-style-guide.md §Color):
  * every Accent.dark  >= 4.5:1 against PAPER and against its own tint
  * every Accent.base  >= 3.0:1 against PAPER (non-text graphics threshold)
  * INK >= 4.5:1 on PAPER, PANEL, and every tint (labels sit on washes)
  * adjacent colors in CYCLE differ by >= 20 CIEDE2000-ish (approx via CIE76)
"""

from __future__ import annotations

import sys

from tokens import ACCENTS, CYCLE, INK, MUTED, PANEL, PAPER


def _srgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h: str) -> float:
    r, g, b = (_lin(c) for c in _srgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _lab(h: str):
    # sRGB D65 -> CIELAB (good enough for pairwise-distance sanity checks)
    r, g, b = (_lin(c) for c in _srgb(h))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def de76(a: str, b: str) -> float:
    la, lb = _lab(a), _lab(b)
    return sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5


def main() -> int:
    ok = True

    def check(cond: bool, msg: str):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("text contrast (>= 4.5:1)")
    check(contrast(INK, PAPER) >= 4.5, f"INK on PAPER: {contrast(INK, PAPER):.2f}")
    check(contrast(INK, PANEL) >= 4.5, f"INK on PANEL: {contrast(INK, PANEL):.2f}")
    check(contrast(MUTED, PAPER) >= 4.5, f"MUTED on PAPER: {contrast(MUTED, PAPER):.2f}")
    for name, a in ACCENTS.items():
        check(contrast(a.dark, PAPER) >= 4.5,
              f"{name}.dark on PAPER: {contrast(a.dark, PAPER):.2f}")
        check(contrast(a.dark, a.tint) >= 4.5,
              f"{name}.dark on {name}.tint: {contrast(a.dark, a.tint):.2f}")
        check(contrast(INK, a.tint) >= 4.5,
              f"INK on {name}.tint: {contrast(INK, a.tint):.2f}")

    print("stroke contrast (>= 3.0:1)")
    for name, a in ACCENTS.items():
        check(contrast(a.base, PAPER) >= 3.0,
              f"{name}.base on PAPER: {contrast(a.base, PAPER):.2f}")

    print("cycle separability (adjacent CIE76 >= 20)")
    for c1, c2 in zip(CYCLE, CYCLE[1:]):
        check(de76(c1, c2) >= 20, f"{c1} vs {c2}: dE {de76(c1, c2):.0f}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
