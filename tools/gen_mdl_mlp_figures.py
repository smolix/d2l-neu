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
                fontsize=12, color="black")

    # Three tensor states, bottom to top (network stacks flow upward, as in
    # the figure style guide). Centers spaced 2.75 apart: tight enough to read
    # as one compact stack, with room for the arrow labels.
    y_x, y_h, y_o = 1.1, 3.85, 6.6
    state(y_x, BLUE,   r"$\mathbf{X}$",  r"input  $n \times 784$")
    state(y_h, GREEN,  r"$\mathbf{H}$",  r"hidden  $n \times 256$")
    state(y_o, ORANGE, r"$\mathbf{O}$",  r"logits  $n \times 10$")

    half = box_h / 2

    def op_arrow(y_from, y_to, labels):
        """Upward arrow between two states, with the operation labels to the
        right of the shaft, stacked bottom to top in the order they apply."""
        tail = (cx, y_from + half)
        tip = (cx, y_to - half)
        fl.arrow(ax, tail, tip, color=GRAY, lw=2.0, mut=16)
        ymid = (tail[1] + tip[1]) / 2
        gap = 0.42
        y0 = ymid - gap * (len(labels) - 1) / 2
        for i, (txt, col, weight) in enumerate(labels):
            ax.text(cx + 0.45, y0 + i * gap, txt, ha="left", va="center",
                    fontsize=12, color=col, fontweight=weight)

    # X -> H : affine (784->256) then the nonlinearity.
    op_arrow(y_x, y_h, [
        (r"affine  $784 \to 256$", "black", "normal"),
        (r"ReLU", ORANGE, "bold"),
    ])
    # H -> O : second affine (256->10).
    op_arrow(y_h, y_o, [
        (r"affine  $256 \to 10$", "black", "normal"),
    ])

    ax.text(cx, -0.08,
            f"{total:,} parameters "
            r"($\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, "
            r"\mathbf{W}^{(2)}, \mathbf{b}^{(2)}$)",
            ha="center", va="center", fontsize=12, color="black")
    fl.save(fig, "mdl-mlp-arch")


def fig_uat_hinges():
    """Universal approximation, one hinge at a time (mlp.md).

    Left panel: the three hinge functions a_j * ReLU(x - t_j) contributed by a
    width-3 hidden layer, joints marked on the x-axis.  Right panel: base line
    + sum of the hinges = the piecewise linear interpolant of a smooth target
    at the joints, with the approximation error shaded.

    The construction is *computed*, not freehand: the target is a full sine
    wave g(x) = 0.5 + 0.4 sin(2 pi x) on [0, 1]; the polyline interpolating g
    at the knots {0, t1, t2, t3, 1} is decomposed exactly into base-slope +
    hinge coefficients (a_j = slope change at t_j), so the right panel really
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

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # ---- left: one hinge per hidden unit -------------------------------
    colors = [BLUE, GREEN, ORANGE]
    for k, (hk, col) in enumerate(zip(hinges, colors)):
        axl.plot(x, hk, color=col, lw=2.2)
        axl.plot([joints[k]], [0], "o", color=col, ms=5, zorder=5,
                 clip_on=False)
        axl.annotate(rf"$t_{k+1}$", xy=(joints[k], 0),
                     xytext=(joints[k] - 0.02, 0.22), ha="center",
                     va="bottom", fontsize=15, color=col)
    axl.axhline(0, color="black", lw=0.9)
    # the caption already says "Left: each ... contributes a single hinge",
    # so no redundant title is drawn here
    # one generic label in the empty lower-left corner, below the falling
    # blue hinge; the colored t_j name the three hinges
    hmin = min(np.min(hk) for hk in hinges)
    hmax = max(np.max(hk) for hk in hinges)
    axl.text(0.03, hmin + 0.05, r"hinge $a_j\,\mathrm{ReLU}(x - t_j)$",
             fontsize=15, color=fl.T.INK, ha="left", va="bottom")
    axl.set_ylim(hmin - 0.15, hmax + 0.55)

    # ---- right: their sum approximates a smooth target -----------------
    axr.plot(x, g(x), color=GRAY, lw=2.2, label="target $f(x)$")
    axr.plot(x, poly, color=BLUE, lw=2.4, label="sum of hinges")
    axr.fill_between(x, poly, g(x), color=ORANGE, alpha=0.30, lw=0,
                     label="error")
    for k, col in enumerate(colors):
        axr.plot([joints[k]], [0], "o", color=col, ms=5, zorder=5,
                 clip_on=False)
    axr.axhline(0, color="black", lw=0.9)
    # the caption states the piece count; no redundant title
    axr.legend(fontsize=13, loc="upper right", frameon=False)
    axr.set_ylim(-0.12, 1.18)

    for ax in (axl, axr):
        ax.set_xlim(lo, hi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("$x$", fontsize=15)

    fig.tight_layout()
    fl.save(fig, "mdl-mlp-uat-hinges")


def fig_grokking():
    """Grokking, schematically (generalization-deep.md): training accuracy
    (gray) saturates within ~10^2.5 steps; validation accuracy (blue) stays
    near chance for more than ten times as many steps and reaches the training
    accuracy two to three orders of magnitude later, around 10^5.5
    steps (cf. Power et al. 2022, Fig. 1).  Curves are logistic in
    log10(steps), so the shape is parametric and reproducible.  Annotation
    text and axes are black; curve labels take their curve's color."""
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

    ink = fl.T.INK
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    # the long memorized-but-not-generalizing stretch, faintly shaded
    ax.axvspan(l_mem, l_gen, color=LIGHT, alpha=0.35, lw=0)
    # markers stop at the plateau (101), below the labels above them
    for lx in (l_mem, l_gen):
        ax.plot([lx, lx], [0, 101], color=ink, ls="--", lw=1.2, zorder=2)

    ax.plot(ls, 100 * train, color=GRAY, lw=2.4, zorder=3)
    ax.plot(ls, 100 * val, color=BLUE, lw=2.6, zorder=4)

    # curve labels in the empty corners, colored like their curves
    ax.text(0.12, 90, "training\naccuracy", color=ink, fontsize=14,
            ha="left", va="center", linespacing=1.15)
    ax.text(5.72, 40, "validation\naccuracy", color=fl.T.BLUE.dark,
            fontsize=14, ha="left", va="center", linespacing=1.15)

    # marker labels above the truncated dashed lines
    ax.text(l_mem, 106, "memorization", color=ink, fontsize=14,
            ha="center", va="bottom")
    ax.text(l_gen, 106, "generalization", color=ink, fontsize=14,
            ha="center", va="bottom")
    ax.text(3.95, 58, "two to three\norders of\nmagnitude\nmore steps",
            color=ink, fontsize=14, ha="center",
            va="center", linespacing=1.15)

    # chance level
    ax.axhline(100 * chance, color=fl.T.FAINT, lw=1.0, zorder=1)
    ax.text(0.12, 100 * chance + 3, "chance", color=ink, fontsize=13,
            ha="left", va="bottom")

    ax.set_xlim(ls[0], ls[-1])
    ax.set_ylim(0, 117)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([rf"$10^{k}$" for k in range(7)])
    ax.set_xlabel("optimization steps (log scale)", fontsize=14, color=ink)
    ax.set_ylabel("accuracy", fontsize=14, color=ink)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink)
    ax.tick_params(colors=ink, labelsize=13)

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


def _black_axes(ax, labelsize=13):
    """Black coordinate axes and tick labels (house rule 1)."""
    ax.spines["left"].set_color(fl.T.INK)
    ax.spines["bottom"].set_color(fl.T.INK)
    ax.tick_params(colors=fl.T.INK, labelsize=labelsize)


def fig_activations():
    """The four activation functions of mlp.md and their derivatives.

    Columns: the function phi(z) (left) and its derivative phi'(z) (right).
    Rows: the rectifiers ReLU and GELU (top), the squashing functions sigmoid
    and tanh (bottom).  Both derivative panels share one vertical scale with
    ticks 0, 1/4, 1, so the reader can compare gradient sizes directly: ReLU
    passes 1 for every positive input, tanh' peaks at 1, and sigmoid' never
    exceeds 1/4; the squashing derivatives vanish in both tails.  ReLU's
    derivative is drawn with the frameworks' convention ReLU'(0) = 0 (filled
    dot) and an open dot at (0, 1).  All curves are computed from their
    closed forms (GELU(z) = z Phi(z), Phi the standard normal CDF)."""
    from math import erf, sqrt, pi
    T = fl.T
    z = np.linspace(-4.0, 4.0, 801)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    pdf = np.exp(-z ** 2 / 2) / sqrt(2 * pi)
    sig = 1.0 / (1.0 + np.exp(-z))
    curves = {  # name: (phi, phi', color)
        "ReLU": (np.maximum(z, 0.0), (z > 0).astype(float), T.BLUE),
        "GELU": (z * cdf, cdf + z * pdf, T.ORANGE),
        "sigmoid": (sig, sig * (1 - sig), T.GREEN),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2, T.PURPLE),
    }
    assert abs((sig * (1 - sig)).max() - 0.25) < 1e-6
    assert abs((cdf + z * pdf).max() - 1.1289) < 1e-3

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.9),
                             gridspec_kw=dict(hspace=0.32, wspace=0.34))
    rows = [("ReLU", "GELU"), ("sigmoid", "tanh")]
    for r, names in enumerate(rows):
        axf, axd = axes[r]
        for name in names:
            f, df, acc = curves[name]
            axf.plot(z, f, color=acc.base, lw=T.SW_LINE, zorder=3)
            if name == "ReLU":   # a step: draw the jump as two pieces
                axd.plot(z[z < 0], df[z < 0], color=acc.base, lw=T.SW_LINE,
                         zorder=3, solid_capstyle="butt")
                axd.plot(z[z > 0], df[z > 0], color=acc.base, lw=T.SW_LINE,
                         zorder=3, solid_capstyle="butt")
                axd.plot([0], [0], "o", color=acc.base, ms=6, zorder=4)
                axd.plot([0], [1], "o", mfc="white", mec=acc.base, mew=1.8,
                         ms=6, zorder=4)
            else:
                axd.plot(z, df, color=acc.base, lw=T.SW_LINE, zorder=3)
        for ax in (axf, axd):
            # the line phi = 0 is the horizontal coordinate axis: black
            ax.axhline(0, color=T.INK, lw=0.9, zorder=1)
            ax.set_xlim(-4, 4)
            ax.set_xticks([-4, -2, 0, 2, 4])
            _black_axes(ax)
        axf.set_ylabel(r"$\phi(z)$", fontsize=15)
        axd.set_ylabel(r"$\phi'(z)$", fontsize=15)
        axd.set_ylim(-0.25, 1.3)
        axd.set_yticks([0, 0.25, 1])
        axd.set_yticklabels(["0", "1/4", "1"])
    axes[1, 0].set_xlabel("$z$", fontsize=15)
    axes[1, 1].set_xlabel("$z$", fontsize=15)

    # top row: ReLU and GELU
    axf, axd = axes[0]
    axf.set_ylim(-1.0, 4.2)
    axf.set_yticks([0, 2, 4])
    lab = dict(fontsize=15, zorder=5)
    axf.text(-3.9, 0.25, "ReLU", color=T.BLUE.dark, ha="left", va="bottom",
             **lab)
    axf.text(-0.75, -0.36, "GELU", color=T.ORANGE.dark, ha="center", va="top",
             **lab)
    axd.text(2.0, 0.83, "ReLU", color=T.BLUE.dark, ha="center", va="top",
             **lab)
    axd.text(-2.2, 0.12, "GELU", color=T.ORANGE.dark, ha="center",
             va="bottom", **lab)
    # bottom row: sigmoid and tanh
    axf, axd = axes[1]
    axf.set_ylim(-1.25, 1.25)
    axf.set_yticks([-1, 0, 1])
    axf.text(-3.9, 0.12, "sigmoid", color=T.GREEN.dark, ha="left",
             va="bottom", **lab)
    axf.text(-3.9, -0.82, "tanh", color=T.PURPLE.dark, ha="left",
             va="bottom", **lab)
    axd.text(0.55, 0.98, "tanh", color=T.PURPLE.dark, ha="left",
             va="center", **lab)
    axd.text(-2.3, 0.3, "sigmoid", color=T.GREEN.dark, ha="center",
             va="bottom", **lab)
    fl.save(fig, "mdl-mlp-activations")


def fig_depth_folding():
    """Depth folds the input: composing a two-unit ReLU layer with itself.

    The tent g(z) = 2 ReLU(z) - 4 ReLU(z - 1/2) maps [0, 1] onto [0, 1],
    rising on the first half and falling on the second; it is the XOR network
    of mlp.md read along s = 2z.  Each composition is one more hidden layer of
    two units, and it doubles the number of linear pieces: g, g o g, g o g o g
    have 2, 4, 8 pieces (counted below from the slopes, and asserted).  A
    single hidden layer with h units has at most h + 1 pieces, so it would need
    7 units for the 8 pieces that three layers of 2 units produce.  In panels 2
    and 3 the previous composition is drawn dashed: every piece of it is split
    in two."""
    T = fl.T
    relu = lambda u: np.maximum(u, 0.0)
    g = lambda u: 2 * relu(u) - 4 * relu(u - 0.5)
    z = np.linspace(0.0, 1.0, 2 ** 10 + 1)        # dyadic grid: exact kinks
    comps = [g(z)]
    for _ in range(2):
        comps.append(g(comps[-1]))

    def pieces(y):
        s = np.sign(np.diff(y))
        return 1 + int(np.count_nonzero(np.diff(s)))
    counts = [pieces(y) for y in comps]
    print(f"  pieces of g, g o g, g o g o g: {counts}")
    assert counts == [2, 4, 8], counts

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55), sharey=True,
                             gridspec_kw=dict(wspace=0.2))
    titles = [r"$g$", r"$g \circ g$", r"$g \circ g \circ g$"]
    for k, ax in enumerate(axes):
        if k > 0:
            ax.plot(z, comps[k - 1], color=T.FAINT, lw=T.SW_HAIR,
                    ls=(0, (4, 3)), zorder=2)
        ax.plot(z, comps[k], color=T.BLUE.base, lw=T.SW_LINE, zorder=3)
        ax.set_title(f"{titles[k]}: {counts[k]} pieces", fontsize=16,
                     color=T.INK, pad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.06)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "1/2", "1"])
        ax.set_yticks([0, 1])
        ax.set_xlabel("$z$", fontsize=16)
        _black_axes(ax, labelsize=14)
    fl.save(fig, "mdl-mlp-depth-folding")


FIGURES += [fig_activations, fig_depth_folding]


if __name__ == "__main__":
    main()
