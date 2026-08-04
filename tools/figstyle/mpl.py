"""matplotlib theme + primitives for d2l mathematical illustrations.

Usage in a generator (replaces the rcParams block at the top of
tools/gen_mdl_figures.py and the ad-hoc constants in its consumers):

    import matplotlib
    matplotlib.use("svg")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from figstyle.mpl import use_style, save, arrow, box, halo_line
    from figstyle import tokens as T

    use_style(hashsalt="mdl-la")
    ...
    save(fig, "mdl-la-vectors")

Compatibility: the legacy house-style names (BLUE/ORANGE/GREEN/GRAY/LIGHT)
are re-exported here mapped onto the unified palette, so a consumer script
can switch its import line and keep every drawing call unchanged.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import tokens as T

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_DIR = os.path.join(REPO_ROOT, "static", "fonts")
IMG_DIR = os.path.join(REPO_ROOT, "img")

# Legacy house-style names -> unified palette (drop-in for gen_mdl_figures).
BLUE = T.BLUE.base
ORANGE = T.ORANGE.base
GREEN = T.GREEN.base
GRAY = T.MUTED
LIGHT = T.HAIRLINE
INK = T.INK

WRITTEN: list[str] = []


def use_style(hashsalt: str = "figstyle") -> None:
    """Register bundled fonts and apply the house rcParams.

    Text is emitted as outline paths (svg.fonttype: path) so rendering is
    identical on every machine and the PDF pipeline needs no fonts; a fixed
    hashsalt + Date:None keeps output byte-idempotent (clean git diffs).
    """
    for fname in ("SourceSans3-Regular.ttf", "SourceSans3-Bold.ttf",
                  "SourceSans3-Italic.ttf", "Inconsolata-Regular.ttf",
                  "Inconsolata-Bold.ttf"):
        fpath = os.path.join(FONT_DIR, fname)
        if os.path.exists(fpath):
            font_manager.fontManager.addfont(fpath)

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        # Typography: the site's own sans for labels; CM mathtext to match
        # the MathJax-set math in the surrounding prose.
        "font.family": T.FONT_SANS,
        "font.size": T.FS_LABEL,
        "axes.titlesize": T.FS_TITLE,
        "axes.labelsize": T.FS_LABEL,
        "xtick.labelsize": T.FS_TINY,
        "ytick.labelsize": T.FS_TINY,
        "legend.fontsize": T.FS_SMALL,
        "mathtext.fontset": T.FONT_MATH,
        # Color: ink-not-black structure, unified categorical cycle.
        "text.color": T.INK,
        "axes.edgecolor": T.INK,
        "axes.labelcolor": T.INK,
        "xtick.color": T.MUTED,
        "ytick.color": T.MUTED,
        "axes.prop_cycle": matplotlib.cycler(color=T.CYCLE),
        "axes.linewidth": T.SW_HAIR,
        "grid.color": T.HAIRLINE,
        "grid.linewidth": T.SW_HAIR,
        "lines.linewidth": T.SW_LINE,
        "patch.linewidth": T.SW_BOX,
        # Chrome discipline: no box, no legend frame, light ticks.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # Reproducibility (the repo's byte-idempotence norm).
        "svg.fonttype": "path",
        "svg.hashsalt": hashsalt,
    })


def save(fig, name: str, out_dir: str = IMG_DIR) -> str:
    """Write ``<out_dir>/<name>.svg`` tightly cropped, deterministically."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight",
                pad_inches=T.PAD_CANVAS / 100, metadata={"Date": None})
    plt.close(fig)
    with open(path) as fh:
        head = fh.read(512)
    assert "<svg" in head, f"{path}: not an SVG?"
    WRITTEN.append(path)
    return path


# --------------------------------------------------------------------------- #
# Primitives the old house style lacked (the root cause of every consumer     #
# script hand-rolling its own _box/_tok_box with ad-hoc pastels).             #
# --------------------------------------------------------------------------- #

def arrow(ax, tail, tip, *, color=BLUE, lw=T.SW_LINE, ls="-", alpha=1.0,
          mut=15):
    """Vector/data-flow arrow with the house head shape."""
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=alpha, shrinkA=0,
                                shrinkB=0, mutation_scale=mut))


def leader(ax, frm, to, *, color=T.MUTED, lw=T.SW_HAIR):
    """Dotted annotation leader (never use for data flow)."""
    ax.annotate("", xy=tuple(to), xytext=tuple(frm),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=(0, (1.5, 2.5)), shrinkA=2,
                                shrinkB=2, mutation_scale=10))


def box(ax, xy, w, h, *, role: str | None = None,
        accent: T.Accent | None = None, label: str | None = None,
        fontsize=T.FS_LABEL, dash=False, novelty=False, r=0.12,
        zorder=3):
    """Rounded semantic block; the matplotlib twin of figstyle.svg.block."""
    from matplotlib.patches import FancyBboxPatch

    a = accent or (T.ROLE[role] if role else None)
    if novelty:
        fc, ec, tc = T.NOVELTY_FILL, "none", T.PAPER
    elif a:
        fc, ec, tc = a.tint, a.base, T.INK
    else:
        fc, ec, tc = T.PAPER, T.INK, T.INK
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0,rounding_size={r * min(w, h):.3f}",
        facecolor=fc, edgecolor=ec, linewidth=T.SW_BOX,
        linestyle=(0, (4, 3)) if dash else "-", zorder=zorder)
    ax.add_patch(patch)
    if label:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=tc, zorder=zorder + 1)
    return patch


def halo_line(ax, xs, ys, *, accent: T.Accent = T.BLUE, lw=T.SW_HEAVY,
              zorder=3):
    """Emphasized path: tint under-halo + base line (weight, not new hue)."""
    ax.plot(xs, ys, color=accent.tint, lw=T.SW_HALO, zorder=zorder - 1,
            solid_capstyle="round")
    ax.plot(xs, ys, color=accent.base, lw=lw, zorder=zorder,
            solid_capstyle="round")


def panel_label(ax, x, y, text, *, color=T.MUTED):
    """Letterspaced ALL-CAPS group label (top-left of a panel)."""
    spaced = " ".join(text.upper()) if len(text) < 24 else text.upper()
    ax.text(x, y, spaced, fontsize=T.FS_TINY, color=color,
            ha="left", va="top")


def right_angle(ax, corner, d1, d2, *, size=0.18, color=T.MUTED,
                lw=T.SW_HAIR):
    """Small square marking a right angle at ``corner`` between directions."""
    import numpy as np
    corner = np.asarray(corner, float)
    d1 = np.asarray(d1, float)
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.asarray(d2, float)
    d2 = d2 / np.linalg.norm(d2)
    p0 = corner
    p1 = corner + size * d1
    p2 = corner + size * (d1 + d2)
    p3 = corner + size * d2
    ax.plot([p0[0], p1[0], p2[0], p3[0], p0[0]],
            [p0[1], p1[1], p2[1], p3[1], p0[1]], color=color, lw=lw)


def clean_axes(ax, lim=None, hide=False, equal=True):
    """Geometric-figure look: equal aspect, minimal or hidden axes."""
    if equal:
        ax.set_aspect("equal")
    if lim is not None:
        (x0, x1), (y0, y1) = lim
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    if hide:
        ax.axis("off")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def axis_cross(ax, xr, yr, *, color=T.MUTED, lw=T.SW_HAIR):
    """Faint x/y axes through the origin for a schematic plane.

    Structure is neutral: the house style draws coordinate scaffolding in
    MUTED gray, never full black, so the colored data reads as figure.
    """
    ax.annotate("", xy=(xr[1], 0), xytext=(xr[0], 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=9))
    ax.annotate("", xy=(0, yr[1]), xytext=(0, yr[0]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=9))
