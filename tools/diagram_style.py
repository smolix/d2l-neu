#!/usr/bin/env python3
"""Reusable primitives for original D2L explanatory diagrams.

This module supplements ``gen_mdl_figures.py``. The older module remains the
compatibility layer for existing figures; this one makes the higher-level rules
in ``docs/visual-design-guide.md`` explicit for new and redesigned work.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class Palette:
    paper: str = "#FFFFFF"
    surface: str = "#F5F7FA"
    ink: str = "#1B2533"
    muted: str = "#5C6878"
    border: str = "#AAB4C0"
    grid: str = "#D9E0E8"
    blue: str = "#276FBF"
    blue_tint: str = "#DCEBFA"
    orange: str = "#C96A18"
    orange_tint: str = "#F9E7D5"
    green: str = "#23845D"
    green_tint: str = "#DCEFE7"
    purple: str = "#7655B5"
    purple_tint: str = "#EAE2F5"
    red: str = "#B84A4A"
    red_tint: str = "#F5DFDF"


@dataclass(frozen=True)
class TypeScale:
    label: float = 12.5
    small: float = 11.0
    panel: float = 14.5
    title: float = 17.0


@dataclass(frozen=True)
class StrokeScale:
    primary: float = 2.0
    secondary: float = 1.4
    construction: float = 0.9
    boundary: float = 1.2


COLORS, TYPE, STROKE = Palette(), TypeScale(), StrokeScale()
STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "figure.facecolor": COLORS.paper,
    "savefig.facecolor": COLORS.paper,
    "font.family": "DejaVu Sans",
    "font.size": TYPE.label,
    "text.color": COLORS.ink,
    "axes.labelcolor": COLORS.ink,
    "axes.edgecolor": COLORS.border,
    "axes.titlesize": TYPE.panel,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": COLORS.muted,
    "ytick.color": COLORS.muted,
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "path",
    "svg.hashsalt": "d2l-diagram-v1",
}


@contextmanager
def style_context():
    """Apply the visual tokens without mutating global Matplotlib settings."""
    with plt.rc_context(STYLE):
        yield


def canvas(width=10.0, height=5.8, *, xlim=None, ylim=None, equal=True):
    """Return a clean diagram canvas with predictable data coordinates."""
    xlim = xlim or (0.0, width)
    ylim = ylim or (0.0, height)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=xlim, ylim=ylim)
    if equal:
        ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def module(ax, xy, width, height, label, *, edge=COLORS.blue,
           face=COLORS.paper, text=COLORS.ink, radius=0.10,
           linewidth=STROKE.boundary, fontsize=TYPE.label, zorder=4):
    """Draw a learned or named processing module."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.03,rounding_size={radius}", facecolor=face,
        edgecolor=edge, linewidth=linewidth, zorder=zorder)
    ax.add_patch(patch)
    if label:
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                color=text, fontsize=fontsize, zorder=zorder + 1)
    return patch


def group(ax, xy, width, height, label=None, *, edge=COLORS.border,
          face=COLORS.surface, linestyle="-", linewidth=STROKE.boundary,
          radius=0.16, label_color=COLORS.muted, zorder=1):
    """Draw a subsystem container behind modules and edges."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.04,rounding_size={radius}", facecolor=face,
        edgecolor=edge, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    ax.add_patch(patch)
    if label:
        ax.text(x + 0.18, y + height - 0.18, label, ha="left", va="top",
                color=label_color, fontsize=TYPE.small, zorder=zorder + 1)
    return patch


def arrow(ax, tail, tip, *, color=COLORS.ink, linewidth=STROKE.primary,
          linestyle="-", mutation_scale=14, connection="arc3", zorder=5):
    """Draw a primary or secondary flow with consistent arrowheads."""
    patch = FancyArrowPatch(
        tail, tip, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, linestyle=linestyle, color=color,
        connectionstyle=connection, shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(patch)
    return patch


def residual(ax, points, *, color=COLORS.purple,
             linewidth=STROKE.secondary, linestyle="--", zorder=3):
    """Draw a rectilinear residual or feedback path ending in an arrow."""
    if len(points) < 2:
        raise ValueError("a residual path needs at least two points")
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, linewidth=linewidth,
            linestyle=linestyle, zorder=zorder)
    return arrow(ax, points[-2], points[-1], color=color, linewidth=linewidth,
                 linestyle=linestyle, zorder=zorder + 1)


def merge(ax, center, symbol="+", *, radius=0.14, edge=COLORS.ink,
          face=COLORS.paper, fontsize=TYPE.label, zorder=6):
    """Draw an explicit algebraic merge node."""
    patch = Circle(center, radius=radius, facecolor=face, edgecolor=edge,
                   linewidth=STROKE.boundary, zorder=zorder)
    ax.add_patch(patch)
    ax.text(*center, symbol, ha="center", va="center", fontsize=fontsize,
            color=edge, zorder=zorder + 1)
    return patch


def panel_label(ax, label, xy=(0.0, 1.0), *, title=None, color=COLORS.ink):
    """Place a panel marker and optional descriptive heading."""
    text = f"{label}  {title}" if title else label
    return ax.text(*xy, text, transform=ax.transAxes, ha="left", va="top",
                   fontsize=TYPE.panel, fontweight="semibold", color=color,
                   clip_on=False, zorder=20)


def callout(ax, text, xy, xytext, *, color=COLORS.muted,
            face=COLORS.paper, align="left", zorder=10):
    """Attach a short annotation with a quiet leader."""
    return ax.annotate(
        text, xy=xy, xytext=xytext, ha=align, va="center",
        fontsize=TYPE.small, color=COLORS.ink,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=face,
                  edgecolor=color, linewidth=STROKE.construction),
        arrowprops=dict(arrowstyle="-", color=color,
                        linewidth=STROKE.construction), zorder=zorder)


def inset_link(ax, source_xy, source_width, source_height, inset_xy,
               inset_width, inset_height, *, color=COLORS.muted):
    """Mark a source region and connect it to a larger detail inset."""
    source = group(ax, source_xy, source_width, source_height, edge=color,
                   face="none", linestyle=(0, (2, 2)), zorder=7)
    inset = group(ax, inset_xy, inset_width, inset_height, edge=color,
                  face=COLORS.paper, linestyle=(0, (2, 2)), zorder=2)
    sx, sy = source_xy
    ix, iy = inset_xy
    for source_y, inset_y in ((sy + source_height, iy + inset_height), (sy, iy)):
        ax.plot([sx + source_width, ix], [source_y, inset_y], color=color,
                linewidth=STROKE.construction, linestyle=(0, (2, 3)), zorder=1)
    return source, inset


def end_label(ax, x, y, text, *, color=COLORS.ink, dx=0.10,
              fontsize=TYPE.small, va="center"):
    """Direct-label a line or trajectory near its endpoint."""
    return ax.text(x + dx, y, text, ha="left", va=va, color=color,
                   fontsize=fontsize, clip_on=False)


def _xml_escape(value):
    return (value.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _inject_accessibility(svg, title, description):
    svg = re.sub(r"<dc:date>.*?</dc:date>\s*", "", svg, flags=re.DOTALL)
    root = re.search(r"<svg\b[^>]*>", svg)
    if not root:
        raise ValueError("Matplotlib output does not contain an SVG root")
    tag = root.group(0)[:-1] + ' role="img" aria-labelledby="title desc">'
    payload = (f"\n <title id=\"title\">{_xml_escape(title)}</title>"
               f"\n <desc id=\"desc\">{_xml_escape(description)}</desc>")
    return svg[:root.start()] + tag + payload + svg[root.end():]


def save_svg(fig, path, *, title, description, transparent=False):
    """Save a deterministic SVG with a concise accessible description."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=transparent,
                metadata={"Date": None})
    plt.close(fig)
    content = path.read_text(encoding="utf-8")
    if '<title id="title">' not in content:
        path.write_text(_inject_accessibility(content, title, description),
                        encoding="utf-8")
    return path
