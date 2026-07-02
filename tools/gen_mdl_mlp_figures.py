#!/usr/bin/env python3
"""Generate the illustrative figures for the "Multilayer Perceptrons" chapter
(``chapter_multilayer-perceptrons``) in the one shared house style defined in
``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Schematic figures use ``set_aspect("equal")`` and real
numbers where the picture must be exact (e.g. the parameter count).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_mlp_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch


def fig_mlp_arch():
    """Data-flow diagram for *this section's* concrete MLP: the batched tensor
    X (n x 784) is flattened and mapped by an affine layer to a 256-dim hidden
    representation, passed through a ReLU, then mapped by a second affine layer
    to 10 logits.  Tensor states are boxes; the named operations live on the
    arrows between them.  The exact parameter count is computed, not guessed."""
    n_in, n_hid, n_out = 784, 256, 10
    p1 = n_in * n_hid + n_hid           # W1 (784 x 256) + b1 (256)
    p2 = n_hid * n_out + n_out          # W2 (256 x 10)  + b2 (10)
    total = p1 + p2
    print(f"  layer 1: {p1:,} params   layer 2: {p2:,} params   "
          f"total {total:,}")
    assert total == 203_530, total      # keep the slide deck's number honest

    fig, ax = plt.subplots(figsize=(4.4, 5.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.7)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 3.0
    box_w, box_h = 4.2, 1.25

    def state(cy, color, head, sub):
        """A rounded tensor-state box with a bold heading and a shape caption."""
        x, y = cx - box_w / 2, cy - box_h / 2
        for fc, a in [(color, 0.12), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.04,rounding_size=0.18",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, cy + 0.24, head, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
        ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                fontsize=9.5, color="black")

    # Three tensor states, top to bottom. Centers spaced 2.75 apart: tight
    # enough to read as one compact stack, with room for the arrow labels.
    y_x, y_h, y_o = 6.6, 3.85, 1.1
    state(y_x, BLUE,   r"$\mathbf{X}$",  r"input  $n \times 784$")
    state(y_h, GREEN,  r"$\mathbf{H}$",  r"hidden  $n \times 256$")
    state(y_o, ORANGE, r"$\mathbf{O}$",  r"logits  $n \times 10$")

    half = box_h / 2

    def op_arrow(y_top, y_bot, labels):
        """Vertical arrow between two states, with stacked operation labels to
        the right of the shaft."""
        tail = (cx, y_top - half)
        tip = (cx, y_bot + half)
        fl.arrow(ax, tail, tip, color=GRAY, lw=2.0, mut=16)
        ymid = (tail[1] + tip[1]) / 2
        gap = 0.42
        y0 = ymid + gap * (len(labels) - 1) / 2
        for i, (txt, col, weight) in enumerate(labels):
            ax.text(cx + 0.45, y0 - i * gap, txt, ha="left", va="center",
                    fontsize=10.5, color=col, fontweight=weight)

    # X -> H : affine (784->256) then the nonlinearity.
    op_arrow(y_x, y_h, [
        (r"affine  $784 \to 256$", "black", "normal"),
        (r"ReLU", ORANGE, "bold"),
    ])
    # H -> O : second affine (256->10).
    op_arrow(y_h, y_o, [
        (r"affine  $256 \to 10$", "black", "normal"),
    ])

    ax.text(cx, 0.2,
            f"{total:,} parameters "
            r"($\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, "
            r"\mathbf{W}^{(2)}, \mathbf{b}^{(2)}$)",
            ha="center", va="center", fontsize=9.5, color=GRAY)
    fl.save(fig, "mdl-mlp-arch")


def fig_uat_hinges():
    """Universal approximation, one hinge at a time (mlp.md, §5.1).

    Left panel: the three hinge functions a_k * ReLU(x - t_k) contributed by a
    width-3 hidden layer, joints marked on the x-axis.  Right panel: base line
    + sum of the hinges = the piecewise linear interpolant of a smooth target
    at the joints, with the approximation error shaded and the "width D =>
    at most D+1 pieces" annotation.

    The construction is *computed*, not freehand: the target is a full sine
    wave g(x) = 0.5 + 0.4 sin(2 pi x) on [0, 1]; the polyline interpolating g
    at the knots {0, t1, t2, t3, 1} is decomposed exactly into base-slope +
    hinge coefficients (a_k = slope change at t_k), so the right panel really
    is the sum of the left panel's hinges plus the base line.  The joints are
    deliberately non-uniform so the three hinges have visibly different signs
    and magnitudes."""
    g = lambda x: 0.5 + 0.4 * np.sin(2 * np.pi * x)
    lo, hi = 0.0, 1.0
    joints = np.array([0.2, 0.45, 0.8])
    knots = np.concatenate([[lo], joints, [hi]])
    gk = g(knots)
    slopes = np.diff(gk) / np.diff(knots)         # slope of each polyline piece
    a = np.diff(slopes)                           # hinge coefficient = slope change
    m0, c0 = slopes[0], gk[0]                     # base line through the start
    print(f"  joints {np.round(joints, 3)}  hinge coeffs {np.round(a, 3)}")
    assert len(set(np.sign(a))) > 1               # mixed up/down hinges

    x = np.linspace(lo, hi, 400)
    relu = lambda z: np.maximum(z, 0.0)
    hinges = [a[k] * relu(x - joints[k]) for k in range(len(joints))]
    poly = c0 + m0 * (x - lo) + np.sum(hinges, axis=0)
    err = np.max(np.abs(poly - g(x)))
    print(f"  max |polyline - target| = {err:.4f}")
    assert err < 0.2, err                         # 4 pieces track the target

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    # ---- left: one hinge per hidden unit -------------------------------
    colors = [BLUE, GREEN, ORANGE]
    for k, (hk, col) in enumerate(zip(hinges, colors)):
        axl.plot(x, hk, color=col, lw=2.2,
                 label=rf"$a_{k+1}\,\mathrm{{ReLU}}(x - t_{k+1})$")
        axl.plot([joints[k]], [0], "o", color=col, ms=5, zorder=5,
                 clip_on=False)
        axl.annotate(rf"$t_{k+1}$", xy=(joints[k], 0),
                     xytext=(joints[k] - 0.02, 0.22), ha="center",
                     va="bottom", fontsize=9.5, color=col)
    axl.axhline(0, color=GRAY, lw=0.9)
    axl.set_title("each unit: one hinge", fontsize=10.5)
    axl.legend(fontsize=7.5, loc="lower left", frameon=False)
    hmin = min(np.min(hk) for hk in hinges)
    hmax = max(np.max(hk) for hk in hinges)
    axl.set_ylim(hmin - 0.15, hmax + 0.55)

    # ---- right: their sum approximates a smooth target -----------------
    axr.plot(x, g(x), color=GRAY, lw=2.2, label="target $f(x)$")
    axr.plot(x, poly, color=BLUE, lw=2.4, label="sum of hinges")
    axr.fill_between(x, poly, g(x), color=ORANGE, alpha=0.30, lw=0,
                     label="error")
    for k, col in enumerate(colors):
        axr.plot([joints[k]], [0], "o", color=col, ms=5, zorder=5,
                 clip_on=False)
    axr.axhline(0, color=GRAY, lw=0.9)
    axr.set_title(r"width $D$  $\Rightarrow$  at most $D+1$ linear pieces",
                  fontsize=10.5)
    axr.legend(fontsize=7.5, loc="upper right", frameon=False)
    axr.set_ylim(-0.12, 1.18)

    for ax in (axl, axr):
        ax.set_xlim(lo, hi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("$x$", fontsize=10)

    fig.tight_layout()
    fl.save(fig, "mdl-mlp-uat-hinges")


def fig_grokking():
    """Grokking, schematically (generalization-deep.md, §5.5) — the companion
    piece to mdl-mlp-double-descent.svg, same axes styling (schematic axes,
    GRAY = training curve, BLUE = validation curve, dashed vertical markers).

    Train accuracy saturates within ~10^2.5 steps; validation accuracy sits at
    chance for several further decades of optimization, then rises sharply
    around 10^5.3 steps (cf. Power et al. 2022, Fig. 1).  Curves are logistic
    in log10(steps), so the shape is parametric and reproducible."""
    ls = np.linspace(0.0, 6.5, 1300)              # log10(optimization steps)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))

    chance = 0.01                                 # ~1/97 for modular addition
    a_mem, a_gen = 2.2, 5.0                       # centres of the two rises
    train = chance + (1 - chance) * sig(4.2 * (ls - a_mem))
    val = chance + (1 - chance) * sig(4.8 * (ls - a_gen))
    # validation barely twitches while the network memorizes
    val += 0.020 * np.exp(-((ls - a_mem) ** 2) / 0.5)

    # markers: where each curve crosses 95% accuracy
    l_mem = ls[np.argmax(train >= 0.95)]
    l_gen = ls[np.argmax(val >= 0.95)]
    print(f"  memorization at 10^{l_mem:.2f} steps, "
          f"generalization at 10^{l_gen:.2f} steps "
          f"({l_gen - l_mem:.1f} decades apart)")
    assert l_gen - l_mem > 2.5                    # the gap IS the phenomenon

    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    # the long memorized-but-not-generalizing stretch, faintly shaded
    ax.axvspan(l_mem, l_gen, color=LIGHT, alpha=0.18, lw=0)
    for lx in (l_mem, l_gen):
        ax.axvline(lx, color=GRAY, ls="--", lw=1.2, zorder=2)

    ax.plot(ls, 100 * train, color=GRAY, lw=2.2, zorder=3)
    ax.plot(ls, 100 * val, color=BLUE, lw=2.6, zorder=4)

    # curve labels on uncluttered stretches
    ax.text(1.15, 78, "training accuracy", color=GRAY, fontsize=10,
            ha="center", va="bottom", rotation=52)
    ax.text(3.55, 7, "validation accuracy", color=BLUE, fontsize=10,
            ha="center", va="bottom")

    # marker labels
    ax.text(l_mem, 106, "memorization", color=GRAY, fontsize=9,
            ha="center", va="bottom")
    ax.text(l_gen, 106, "generalization", color=GRAY, fontsize=9,
            ha="center", va="bottom")
    ax.text((l_mem + l_gen) / 2, 55,
            "training set interpolated,\nyet no generalization\n"
            "for ~3 decades of steps",
            color=GRAY, fontsize=9, ha="center", va="center")

    # chance level
    ax.axhline(100 * chance, color=LIGHT, lw=1.0, zorder=1)
    ax.text(0.12, 100 * chance + 2, "chance", color=GRAY, fontsize=8.5,
            ha="left", va="bottom")

    ax.set_xlim(ls[0], ls[-1])
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([rf"$10^{k}$" for k in range(7)], fontsize=9)
    ax.set_xlabel("optimization steps (log scale)", fontsize=10.5)
    ax.set_ylabel("accuracy", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fl.save(fig, "mdl-mlp-grokking")


FIGURES = [fig_mlp_arch, fig_uat_hinges, fig_grokking]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
