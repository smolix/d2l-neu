#!/usr/bin/env python3
"""Generate the new illustrative figures of the "Multivariable Calculus" section
(``chapter_mdl-calculus/mdl-multivariable-calculus.md``) in the shared house
style of ``gen_mdl_figures.py`` and ``gen_mdl_calculus_figures.py``.

* ``mdl-cal-backward-sweep``: the forward pass (node values, edge derivatives)
  and the backward pass (adjoints) on the four-input graph of
  ``eq_mdl-multi_func_def`` at (w, x, y, z) = (1, 1, -2, 1).  Every number is
  computed here from the graph, not typed in.
* ``mdl-cal-second-derivative-test``: contours of the quadratic model
  q(v) = v^T H v / 2 for the four Hessians of the second-derivative test, with
  eigenvector directions and the descent field -H v.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_calculus_multivariable_figures.py

The output is byte-idempotent (``save`` fixes the SVG hash salt and date).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from figstyle import tokens as T

np, plt = fl.np, fl.plt
GRAY, LIGHT, INK = fl.GRAY, fl.LIGHT, fl.INK
RED = T.ROLE["grad"]          # gradients and the backward pass
FAINT = T.FAINT

from matplotlib.patches import FancyBboxPatch

FS = 14          # node text and panel labels
FS_EDGE = 12.5   # edge-derivative labels


# --------------------------------------------------------------------------- #
# The backward sweep on the four-input graph                                  #
# --------------------------------------------------------------------------- #

def _graph_numbers():
    """Node values, edge derivatives and adjoints of
    f = (u+v)^2, u = (a+b)^2, v = (a-b)^2, a = (w+x+y+z)^2, b = (w+x-y-z)^2."""
    w, x, y, z = 1, 1, -2, 1
    a, b = (w + x + y + z) ** 2, (w + x - y - z) ** 2
    u, v = (a + b) ** 2, (a - b) ** 2
    f = (u + v) ** 2
    val = dict(w=w, x=x, y=y, z=z, a=a, b=b, u=u, v=v, f=f)
    sa, sb = 2 * (w + x + y + z), 2 * (w + x - y - z)
    edge = {("w", "a"): sa, ("x", "a"): sa, ("y", "a"): sa, ("z", "a"): sa,
            ("w", "b"): sb, ("x", "b"): sb, ("y", "b"): -sb, ("z", "b"): -sb,
            ("a", "u"): 2 * (a + b), ("b", "u"): 2 * (a + b),
            ("a", "v"): 2 * (a - b), ("b", "v"): -2 * (a - b),
            ("u", "f"): 2 * (u + v), ("v", "f"): 2 * (u + v)}
    adj = {"f": 1}
    for n in ["u", "v", "a", "b", "w", "x", "y", "z"]:   # reverse order
        adj[n] = sum(adj[c] * d for (p, c), d in edge.items() if p == n)
    return val, edge, adj


POS = {"w": (0, 3), "x": (0, 2), "y": (0, 1), "z": (0, 0),
       "a": (4.7, 2.25), "b": (4.7, 0.75),
       "u": (8.1, 2.25), "v": (8.1, 0.75),
       "f": (11.3, 1.5)}
# one width per column, wide enough for the adjoints of panel (b)
BOX_W = {"w": 2.35, "x": 2.35, "y": 2.35, "z": 2.35, "a": 2.05, "b": 2.05,
         "u": 1.75, "v": 1.75, "f": 1.9}
BOX_H = 0.56


def _box(ax, name, text):
    cx, cy = POS[name]
    w = BOX_W[name]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - BOX_H / 2), w, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=LIGHT, edgecolor="black", lw=1.2,
                                zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK,
            zorder=4)


PARENTS = {"a": "wxyz", "b": "wxyz", "u": "ab", "v": "ab", "f": "uv"}
CHILDREN = {"w": "ab", "x": "ab", "y": "ab", "z": "ab", "a": "uv", "b": "uv",
            "u": "f", "v": "f"}


def _port(name, other, right):
    """Point on the right (or left) side of box ``name`` where the edge to
    ``other`` attaches; edges sharing a side are spread over its height so
    their arrowheads do not pile up."""
    cx, cy = POS[name]
    peers = CHILDREN[name] if right else PARENTS[name]
    k, m = peers.index(other), len(peers)
    dy = 0.0 if m == 1 else (0.5 - k / (m - 1)) * 0.6 * BOX_H
    return np.array([cx + (BOX_W[name] / 2 if right else -BOX_W[name] / 2),
                     cy + dy])


def _edge(ax, p, c, color, reverse=False):
    """Edge from the right side of parent ``p`` to the left side of child ``c``
    (drawn the other way round in the backward pass); returns (start, end) in
    the forward direction for placing labels."""
    t0, t1 = _port(p, c, True), _port(c, p, False)
    tail, head = (t1, t0) if reverse else (t0, t1)
    fl.arrow(ax, tail, head, color=color, lw=1.4, mut=11)
    return t0, t1


def fig_backward_sweep():
    """(a) forward pass: node values and edge derivatives; (b) backward pass:
    the same graph with reversed red arrows and adjoints in the nodes."""
    val, edge, adj = _graph_numbers()
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.2))
    sym = {n: n for n in POS}

    for k, ax in enumerate(axes):
        backward = k == 1
        for (p, c), d in edge.items():
            t0, t1 = _edge(ax, p, c, RED.base if backward else GRAY,
                           reverse=backward)
            if backward:
                continue
            # label inputs' edges only on the w and y rows (x, z repeat them)
            if p in ("x", "z"):
                continue
            # near the source, where a node's outgoing edges are separated
            frac = 0.16 if p in ("w", "y") else 0.28 if p in ("a", "b") else 0.5
            pos = t0 + frac * (t1 - t0)
            ax.text(pos[0], pos[1], f"${d}$", ha="center", va="center",
                    fontsize=FS_EDGE, color=INK, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        for n in POS:
            if backward:
                text = rf"$\bar {sym[n]}={adj[n]}$"
            else:
                text = f"${sym[n]}={val[n]}$"
            _box(ax, n, text)
        ax.text(-1.2, 3.45, "(b)" if backward else "(a)", fontsize=FS,
                color=INK, ha="left", va="bottom")
        ax.set_xlim(-1.25, 12.3); ax.set_ylim(-0.4, 3.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(hspace=0.04)
    fl.save(fig, "mdl-cal-backward-sweep")


# --------------------------------------------------------------------------- #
# The four cases of the second-derivative test                                #
# --------------------------------------------------------------------------- #

def fig_second_derivative_test():
    """Contours of q(v) = v^T H v / 2 for (a) the minimum and (b) the maximum of
    x e^{-x^2-y^2}, (c) the saddle of x^4 + y^4 - 4xy at the origin, and (d) the
    singular log-sum-exp Hessian at (0, log 2); dashed eigenvector directions
    labelled with their eigenvalues; red descent arrows -H v on a circle."""
    e = np.exp(-0.5)
    # eigenvalue labels: (text, x, y, ha, va), placed beside the end of each
    # dashed eigenvector line, clear of the line and of the descent arrows
    axis_lab = lambda lx, ly: [(lx, 0.97, 0.05, "right", "bottom"),
                               (ly, 0.08, 0.97, "left", "top")]
    diag_lab = lambda l11, l1m: [(l11, 0.62, 0.97, "right", "top"),
                                 (l1m, 0.62, -0.97, "right", "bottom")]
    cases = [
        (e * np.diag([2 * np.sqrt(2), np.sqrt(2)]), "(a)",
         axis_lab(r"$\lambda=1.72$", r"$\lambda=0.86$")),
        (-e * np.diag([2 * np.sqrt(2), np.sqrt(2)]), "(b)",
         axis_lab(r"$\lambda=-1.72$", r"$\lambda=-0.86$")),
        (np.array([[0.0, -4.0], [-4.0, 0.0]]), "(c)",
         diag_lab(r"$\lambda=-4$", r"$\lambda=4$")),
        (2 / 9 * np.array([[1.0, -1.0], [-1.0, 1.0]]), "(d)",
         diag_lab(r"$\lambda=0$", r"$\lambda=\frac{4}{9}$")),
    ]
    lim = 1.0
    xs = np.linspace(-lim, lim, 301)
    X, Y = np.meshgrid(xs, xs)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for ax, (H, tag, labels) in zip(axes, cases):
        Q = 0.5 * (H[0, 0] * X ** 2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y ** 2)
        lam, V = np.linalg.eigh(H)
        lam_abs = np.max(np.abs(lam))
        if np.all(lam > 1e-12) or np.all(lam < -1e-12):
            # ellipses through radii 0.3 .. 1.05 along the steeper axis
            levels = 0.5 * lam_abs * np.array([0.3, 0.55, 0.8, 1.05]) ** 2
            levels = np.sort(np.sign(lam[0]) * levels)
        elif np.min(lam) < -1e-12:
            levels = np.array([-2.0, -0.8, -0.2, 0.2, 0.8, 2.0])
        else:
            d = np.array([0.35, 0.7, 1.05, 1.4])          # |x - y| = d
            levels = 0.5 * lam_abs * (d / np.sqrt(2)) ** 2
        ax.contour(X, Y, Q, levels=levels, colors=GRAY, linewidths=1.0,
                   linestyles="solid", zorder=2)
        # eigenvector directions (dashed), labelled with their eigenvalues
        for i in range(2):
            q = V[:, i]
            ax.plot([-1.5 * q[0], 1.5 * q[0]], [-1.5 * q[1], 1.5 * q[1]],
                    ls=(0, (4, 3)), color=FAINT, lw=1.3, zorder=1)
        for txt, lx, ly, ha, va in labels:
            ax.text(lx, ly, txt, fontsize=FS, color=INK, ha=ha, va=va,
                    zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        # descent arrows -H v at eight points of the circle of radius 0.62,
        # with one scale per panel so lengths compare within a panel
        th = np.arange(8) * np.pi / 4
        P = 0.62 * np.stack([np.cos(th), np.sin(th)], axis=1)
        G = -(P @ H.T)
        s = 0.22 / np.max(np.linalg.norm(G, axis=1))
        for pt, g in zip(P, G):
            ax.plot([pt[0]], [pt[1]], "o", color=RED.base, ms=3.0, zorder=4)
            if np.linalg.norm(g) * s > 1e-3:
                fl.arrow(ax, pt, pt + s * g, color=RED.base, lw=2.0, mut=10)
        ax.plot([0], [0], "o", color=INK, ms=4, zorder=5)
        ax.text(-0.97, 0.97, tag, fontsize=FS, color=INK, ha="left", va="top",
                zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT)
    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-cal-second-derivative-test")


FIGURES = [fig_backward_sweep, fig_second_derivative_test]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[start:]:
        assert os.path.getsize(p) > 0, p
        print(f"  {os.path.basename(p):36s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
