#!/usr/bin/env python3
"""Generate the illustrative figures for the "Linear Neural Networks for
Classification" chapter (``chapter_linear-classification``) in the one shared
house style defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs). Figures that show a computed result use real numerical
computation so the pictures are exact, not sketched.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_classification_figures.py

All figures are written to ``img/mdl-clf-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch


def fig_loss_accuracy():
    """The two roles of a classifier's scores. One logit vector feeds two
    branches. The TOP (training) branch softmaxes the logits into
    probabilities and reads off the cross-entropy loss of the true class,
    which is smooth in the parameters and is what gradient descent minimizes.
    The BOTTOM (evaluation) branch takes the argmax to a hard decision,
    compares it with the label, and counts it toward accuracy, a step
    function of the parameters. The logits (1.0, 2.2, 0.3) are the running
    example of the softmax section (cat, chicken, dog; the image is a
    chicken), and every number shown is computed, not typed. Laid out at
    text-column width with 13-15 pt type so it stays legible on the page."""
    T = fl.T
    o = np.array([1.0, 2.2, 0.3])             # logits of cat, chicken, dog
    names = ["cat", "chicken", "dog"]
    y = 1                                      # true class: chicken
    p = np.exp(o - o.max())                    # numerically stable softmax
    p = p / p.sum()
    yhat = int(np.argmax(o))                   # hard decision
    loss = float(-np.log(p[y]))                # cross-entropy of the label
    print(f"  logits o = {o.tolist()}  softmax = {np.round(p, 3).tolist()}")
    print(f"  argmax = {names[yhat]}  label = {names[y]}  loss = {loss:.3f}")

    W, H = 6.7, 2.75                           # canvas in inches (aspect 1)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x0, y0, w, h, acc):
        fc, ec = (T.PANEL, T.INK) if acc is None else (acc.tint, acc.base)
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=T.SW_BOX, zorder=2))
        return dict(l=x0, r=x0 + w, b=y0, t=y0 + h, cx=x0 + w / 2,
                    cy=y0 + h / 2)

    def label(x, y0, s, **kw):
        ax.text(x, y0, s, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
                fontsize=kw.pop("fontsize", 15), color=kw.pop("color", T.INK),
                zorder=3, **kw)

    def flow(x0, x1, yy, acc, text=""):
        fl.arrow(ax, (x0, yy), (x1, yy), color=acc.base, lw=T.SW_LINE, mut=15)
        if text:
            label((x0 + x1) / 2, yy + 0.1, text, va="bottom", fontsize=13,
                  color=acc.dark)

    y_top, y_bot, bh = 2.08, 0.74, 0.62
    # the logits, one row per class
    L = box(0.04, 0.36, 1.62, 2.10, None)
    label(L["cx"], 2.12, r"logits $\mathbf{o}$")
    for k, (nm, v) in enumerate(zip(names, o)):
        label(0.22, 1.64 - 0.44 * k, nm, ha="left")
        label(1.48, 1.64 - 0.44 * k, f"${v:.1f}$", ha="right")
    # training: softmax -> probabilities -> loss
    P = box(2.55, y_top - bh / 2, 2.05, bh, T.ORANGE)
    label(P["cx"], P["cy"], r"$\hat{\mathbf{y}} = ("
          + ",\\ ".join(f"{v:.2f}" for v in p) + ")$")
    Lo = box(5.0, y_top - bh / 2, 1.66, bh, T.ORANGE)
    label(Lo["cx"], Lo["cy"], rf"$-\log {p[y]:.2f} = {loss:.2f}$")
    flow(L["r"], P["l"], y_top, T.ORANGE, "softmax")
    flow(P["r"], Lo["l"], y_top, T.ORANGE)
    # evaluation: argmax -> predicted class -> compared with the label
    D = box(2.55, y_bot - bh / 2, 2.05, bh, T.BLUE)
    label(D["cx"], D["cy"], f"predicted: {names[yhat]}")
    A = box(5.0, y_bot - bh / 2, 1.66, bh, T.BLUE)
    label(A["cx"], A["cy"] + 0.12, f"label: {names[y]}", fontsize=13)
    label(A["cx"], A["cy"] - 0.13, "correct" if yhat == y else "wrong",
          weight="bold")
    flow(L["r"], D["l"], y_bot, T.BLUE, "argmax")
    flow(D["r"], A["l"], y_bot, T.BLUE)
    # what each number is, and what it is for
    label(Lo["cx"], Lo["t"] + 0.08, "loss", va="bottom", fontsize=13,
          color=T.ORANGE.dark, weight="bold")
    label(A["cx"], A["t"] + 0.08, "accuracy", va="bottom", fontsize=13,
          color=T.BLUE.dark, weight="bold")
    label((P["l"] + Lo["r"]) / 2, P["b"] - 0.1,
          "smooth in the parameters: training minimizes it", va="top",
          fontsize=13, color=T.ORANGE.dark, style="italic")
    label((D["l"] + A["r"]) / 2, D["b"] - 0.1,
          "a step in the parameters: evaluation counts it", va="top",
          fontsize=13, color=T.BLUE.dark, style="italic")
    fl.save(fig, "mdl-clf-loss-accuracy")


def fig_decision_regions():
    """What a (multiclass) linear classifier's decision LOOKS like. Left: a
    3-class softmax regression partitions the plane into three convex
    polyhedral regions --- the argmax of three affine score functions --- whose
    straight pairwise boundaries (the ties o_j = o_k) meet at the single point
    where all three scores are equal. Right: the binary special case. The
    predicted probability y_hat_1 = sigma(o) depends on x only through the
    logit difference o = (w_1 - w_2).x + b_1 - b_2, so its level sets are
    parallel lines perpendicular to w_1 - w_2, with the decision boundary at
    probability 1/2. The level lines carry inline labels on white patches, so
    no label sits on a line. Regions/boundaries are computed exactly from the
    affine scores, not sketched."""
    T = fl.T
    # --- three concrete affine score functions (generic, no symmetry) ---
    W = np.array([[0.30, 1.00],      # class 1
                  [-1.00, -0.55],    # class 2
                  [0.95, -0.60]])    # class 3
    b = np.array([0.15, -0.10, -0.05])
    lim = 3.0

    # the point where all three scores tie: (w1-w2).x = b2-b1, (w1-w3).x = b3-b1
    A = np.array([W[0] - W[1], W[0] - W[2]])
    p0 = np.linalg.solve(A, np.array([b[1] - b[0], b[2] - b[0]]))
    print(f"  triple point (all scores equal): {np.round(p0, 3).tolist()}")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    # ---------------- LEFT: 3-class argmax regions ----------------
    n = 601
    gx = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(gx, gx)
    scores = np.einsum("kd,dij->kij", W, np.stack([X, Y])) + b[:, None, None]
    region = np.argmax(scores, axis=0)
    from matplotlib.colors import ListedColormap
    axl.contourf(X, Y, region, levels=[-0.5, 0.5, 1.5, 2.5],
                 cmap=ListedColormap([BLUE, ORANGE, GREEN]), alpha=0.14)

    # exact pairwise boundary rays out of the triple point: on the tie o_i=o_j,
    # keep the half where the third class is NOT maximal.
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    for i, j, k in pairs:
        d = W[i] - W[j]
        t = np.array([-d[1], d[0]])
        t = t / np.linalg.norm(t)
        probe = p0 + 1e-3 * t
        if (W[k] @ probe + b[k]) > (W[i] @ probe + b[i]):
            t = -t
        far = p0 + t * 2.5 * lim
        axl.plot([p0[0], far[0]], [p0[1], far[1]], color=GRAY, lw=1.8)
    axl.plot(*p0, "o", ms=7, color=T.INK, zorder=5)
    axl.annotate(r"$o_1 = o_2 = o_3$", p0, xytext=(p0[0] - 1.9, p0[1] - 1.25),
                 fontsize=15, color=T.INK, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.2,
                                 shrinkA=9, shrinkB=4))

    # label each region at a representative interior point
    for cls, col, pos in [(0, BLUE, (0.15, 2.1)), (1, ORANGE, (-2.0, 0.35)),
                          (2, GREEN, (1.95, -1.45))]:
        axl.text(*pos, f"class {cls + 1}", ha="center", va="center",
                 fontsize=15, fontweight="bold", color=col)
        axl.text(pos[0], pos[1] - 0.5,
                 rf"$o_{cls + 1} \geq o_k\ \forall k$",
                 ha="center", va="center", fontsize=15, color=col)
    fl.clean_axes(axl, lim=((-lim, lim), (-lim, lim)))
    axl.set_xticks([]), axl.set_yticks([])
    axl.set_xlabel(r"$x_1$"), axl.set_ylabel(r"$x_2$")
    # no per-panel title: the caption's "Left: ... Right: ..." already names
    # each panel, so a repeated in-figure heading would be redundant.

    # ---------------- RIGHT: binary sigma level sets ----------------
    w2, b2 = np.array([1.3, 0.9]), 0.0     # w_1 - w_2 and b_1 - b_2
    wn = w2 / np.linalg.norm(w2)
    tang = np.array([-wn[1], wn[0]])
    sigma_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    lab_row = -1.55 * tang          # the level labels form one row across
    for s in sigma_levels:
        o = np.log(s / (1 - s))                     # logit: w.x + b = o
        c = (o - b2) / np.linalg.norm(w2)           # signed distance along w
        pt = c * wn
        far1, far2 = pt + tang * 3 * lim, pt - tang * 3 * lim
        main = abs(s - 0.5) < 1e-9
        axr.plot([far1[0], far2[0]], [far1[1], far2[1]],
                 color=BLUE if main else GRAY, lw=2.6 if main else 1.4,
                 ls="-" if main else (0, (5, 3)), zorder=2)
        axr.text(*(lab_row + c * wn), f"{s}", fontsize=14,
                 color=T.BLUE.dark if main else T.INK, ha="center",
                 va="center", zorder=4,
                 bbox=dict(fc="white", ec="none", pad=1.5))
    c0 = (np.log(0.1 / 0.9) - b2) / np.linalg.norm(w2)
    axr.text(*(lab_row + (c0 - 0.75) * wn), r"$\hat{y}_1 =$", fontsize=15,
             color=T.INK, ha="center", va="center", zorder=4)
    # w_1 - w_2, drawn from a point on the decision boundary
    base = 1.9 * tang
    tip = base + 0.95 * wn
    fl.arrow(axr, base, tip, color=ORANGE, lw=2.4)
    axr.text(*(tip + np.array([0.05, 0.42])), r"$\mathbf{w}_1 - \mathbf{w}_2$",
             fontsize=15, color=T.ORANGE.dark, ha="center", va="center",
             zorder=4, bbox=dict(fc="white", ec="none", pad=1.5))
    # region labels in the empty corners beyond every level line
    axr.text(2.0, 2.45, r"$\hat{y}_1 > \frac{1}{2}$", fontsize=15,
             color=T.BLUE.dark, ha="center", va="center")
    axr.text(-2.2, -1.6, r"$\hat{y}_1 < \frac{1}{2}$", fontsize=15,
             color=T.INK, ha="center", va="center")
    fl.clean_axes(axr, lim=((-lim, lim), (-lim, lim)))
    axr.set_xticks([]), axr.set_yticks([])
    axr.set_xlabel(r"$x_1$"), axr.set_ylabel(r"$x_2$")
    # no per-panel title: the caption already names "Left: ... Right: ..."

    fig.subplots_adjust(wspace=0.18)
    fl.save(fig, "mdl-clf-decision-regions")


def fig_temperature():
    """One 3-class score vector pushed through softmax(o / T) at three
    temperatures. Cooling (T < 1) sharpens toward the hard argmax, T = 1 is
    the plain softmax, heating (T > 1) flattens toward uniform --- and the
    ORDER of the bars never changes, since 1/T scales all logits alike. Uses
    the section's running logits (1.0, 2.2, 0.3) for cat, chicken, and dog,
    so the middle panel repeats numbers the reader has already seen. All bar
    heights are the exact computed softmax values."""
    o = np.array([1.0, 2.2, 0.3])
    names = ["cat", "chicken", "dog"]
    temps = [0.25, 1.0, 4.0]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    for ax, temp in zip(axes, temps):
        z = o / temp
        p = np.exp(z - z.max())
        p = p / p.sum()
        print(f"  T = {temp:>4}: softmax(o/T) = {np.round(p, 3).tolist()}")
        ax.bar([1, 2, 3], p, width=0.62, color=[BLUE, ORANGE, GREEN],
               alpha=0.85)
        for x, v in zip([1, 2, 3], p):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(names, fontsize=14)
        title = {0.25: r"$T = 0.25$ (sharper)", 1.0: r"$T = 1$ (softmax)",
                 4.0: r"$T = 4$ (flatter)"}[temp]
        ax.set_title(title, fontsize=17)
    axes[0].set_ylabel(r"$\mathrm{softmax}(\mathbf{o}/T)$")
    # no figure suptitle: the caption already states "the same three scores
    # ... at three temperatures"; the per-panel T= titles stay since they are
    # the only thing that identifies which panel is which.
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-clf-temperature")


def fig_density_ratio():
    """Importance weights under covariate shift, for the x_1-marginal of the
    section's running example: source p_S = N(0, 1), target p_T = N(2, 1).
    Top: the two densities.  Bottom (log scale, shared x): the weight
    beta = p_T/p_S = exp(2 x_1 - 2), a straight line on this scale
    because the log-ratio of two equal-variance Gaussians is linear.  It is
    tiny where only the source has mass, equals 1 where the densities cross
    (x_1 = 1), and grows by e^2 per unit of x_1 where the source thins out; the
    dashed segment is the clipped weight min(beta, c) with c = 5.  All curves
    are computed exactly, not sketched."""
    x = np.linspace(-3.0, 5.5, 601)
    mu_s, mu_t, c = 0.0, 2.0, 5.0
    gauss = lambda m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
    p_s, p_t = gauss(mu_s), gauss(mu_t)
    beta = p_t / p_s                                  # = exp(2 x_1 - 2)
    x_eq = (mu_t + mu_s) / 2                          # beta = 1
    x_clip = x_eq + np.log(c) / (mu_t - mu_s)         # beta = c
    print(f"  beta = 1 at x = {x_eq:.3f};  beta = c = {c} at x = {x_clip:.3f}")

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(7.2, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))
    blue, green, orange = fl.T.BLUE, fl.T.GREEN, fl.T.ORANGE
    top.plot(x, p_s, color=blue.base)
    top.fill_between(x, 0, p_s, color=blue.base, alpha=0.10, lw=0)
    top.plot(x, p_t, color=green.base)
    top.fill_between(x, 0, p_t, color=green.base, alpha=0.10, lw=0)
    top.text(-1.25, 0.30, r"source $p_S$", ha="right", fontsize=15,
             color=blue.dark)
    top.text(3.25, 0.30, r"target $p_T$", ha="left", fontsize=15,
             color=green.dark)
    top.set_ylabel("density")
    top.set_ylim(0, 0.45)
    top.set_yticks([0, 0.2, 0.4])

    bot.plot(x, beta, color=orange.base)
    keep = x >= x_clip
    bot.plot(x[keep], np.full(keep.sum(), c), color=orange.base,
             ls=(0, (4, 3)))
    bot.axhline(1.0, color=fl.T.FAINT, lw=fl.T.SW_HAIR, ls=":")
    bot.set_yscale("log")
    bot.set_ylim(1e-3, 3e3)
    bot.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    bot.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    bot.set_ylabel(r"weight $\beta$")
    bot.set_xlabel(r"$x_1$")
    bot.set_xlim(x[0], x[-1])
    bot.text(-0.9, 150, r"$\beta = \dfrac{p_T}{p_S} = e^{2x_1 - 2}$",
             fontsize=15, color=orange.dark, ha="center", va="center")
    bot.text(-2.85, 1.6, r"$\beta = 1$", fontsize=14, color="black",
             ha="left", va="bottom")
    bot.text(4.1, 7.0, rf"clipped at $c = {c:.0f}$", fontsize=14,
             color="black", ha="center", va="bottom")
    # the densities cross exactly where beta = 1: one guide through both panels
    for ax in (top, bot):
        ax.axvline(x_eq, color=fl.T.FAINT, lw=fl.T.SW_HAIR, ls=":")
    bot.plot([x_eq], [1.0], "o", ms=5, color="black", zorder=5)

    fl.save(fig, "mdl-clf-density-ratio")


def _shift_panel(ax, x, joint0, joint1, color0, color1):
    """One panel of the shift-types figure: the class-wise joint densities
    p(x, y=0) and p(x, y=1), their sum p(x) as a faint outline, and a dotted
    line at the decision boundary (where the two joint densities cross)."""
    ax.plot(x, joint0 + joint1, color=fl.T.FAINT, lw=fl.T.SW_HAIR)
    ax.fill_between(x, 0, joint0, color=color0.base, alpha=0.18, lw=0)
    ax.fill_between(x, 0, joint1, color=color1.base, alpha=0.18, lw=0)
    ax.plot(x, joint0, color=color0.base)
    ax.plot(x, joint1, color=color1.base)
    diff = joint1 - joint0
    k = np.flatnonzero(np.diff(np.sign(diff)) > 0)[0]       # first 0 -> 1 cross
    xb = x[k] - diff[k] * (x[k + 1] - x[k]) / (diff[k + 1] - diff[k])
    ax.axvline(xb, color="black", lw=fl.T.SW_HAIR, ls=":")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 0.52)
    ax.set_xticks([]), ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return xb


def fig_shift_types():
    """Three kinds of distribution shift on one input coordinate x.  Columns:
    covariate, label, concept shift; rows: source and target.  Every panel
    draws the same object, the class-wise joint densities p(x, y=0) and
    p(x, y=1) (their sum, p(x), is the faint outline), so the reader sees
    which factor of p(x, y) moved and which stayed.  Covariate: p(x) moves,
    the boundary P(y=1|x) = 1/2 stays.  Label: the class proportions move, the
    class shapes p(x|y) stay (so the boundary moves).  Concept: p(x) stays,
    the boundary moves.  The dotted line marks where the classes cross."""
    x = np.linspace(-4.0, 4.5, 600)
    gauss = lambda m, s=1.0: np.exp(-0.5 * ((x - m) / s) ** 2) / (
        s * np.sqrt(2 * np.pi))
    sig = lambda t: 1 / (1 + np.exp(-t))
    c0, c1 = fl.T.PURPLE, fl.T.GOLD
    cases = {
        # covariate shift: p(x) moves, P(y=1|x) = sig(3x) stays
        "covariate": [(gauss(0.0) * (1 - sig(3 * x)), gauss(0.0) * sig(3 * x)),
                      (gauss(1.3) * (1 - sig(3 * x)), gauss(1.3) * sig(3 * x))],
        # label shift: p(x|y) stays, p(y=1) goes from 0.5 to 0.8
        "label": [(0.5 * gauss(-1.2, 0.8), 0.5 * gauss(1.2, 0.8)),
                  (0.2 * gauss(-1.2, 0.8), 0.8 * gauss(1.2, 0.8))],
        # concept shift: p(x) stays, the boundary moves from 0 to 1.2
        "concept": [(gauss(0.3, 1.1) * (1 - sig(3 * x)),
                     gauss(0.3, 1.1) * sig(3 * x)),
                    (gauss(0.3, 1.1) * (1 - sig(3 * (x - 1.2))),
                     gauss(0.3, 1.1) * sig(3 * (x - 1.2)))],
    }
    # column titles name the shift and the factor of p(x, y) it keeps fixed
    titles = {"covariate": "covariate shift\n" + r"fixed $p(y \mid x)$",
              "label": "label shift\n" + r"fixed $p(x \mid y)$",
              "concept": "concept shift\n" + r"fixed $p(x)$"}

    def draw(axes, keys):
        """Fill a 2 x len(keys) grid of panels (rows: source, target)."""
        for col, key in enumerate(keys):
            for row in range(2):
                xb = _shift_panel(axes[row, col], x, *cases[key][row], c0, c1)
                print(f"  {key:9s} {'source' if row == 0 else 'target'}: "
                      f"boundary at x = {xb:+.2f}")
            axes[0, col].set_title(titles[key], fontsize=15, linespacing=1.4)
            axes[1, col].set_xlabel(r"$x$")
        axes[0, 0].set_ylabel("source", fontsize=15)
        axes[1, 0].set_ylabel("target", fontsize=15)
        # class names once, in the top-left panel, beside their own bumps
        axes[0, 0].text(-2.55, 0.33, r"$y = 0$", ha="center", fontsize=14,
                        color=c0.dark)
        axes[0, 0].text(2.55, 0.33, r"$y = 1$", ha="center", fontsize=14,
                        color=c1.dark)

    # the book's three-column figure ...
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 3.9), sharex=True,
                             gridspec_kw=dict(hspace=0.16, wspace=0.10))
    draw(axes, list(cases))
    fl.save(fig, "mdl-clf-shift-types")
    # ... and one column each for the covariate and label slides of the deck
    for key in ["covariate", "label"]:
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 3.9), sharex=True,
                                 squeeze=False, gridspec_kw=dict(hspace=0.16))
        draw(axes, [key])
        fl.save(fig, f"mdl-clf-shift-{key}")


def fig_feedback_loop():
    """A deployed model's decisions change the data it is retrained on.  The
    predictive-policing loop as four blocks in a cycle: the model predicts
    more crime in area A -> more patrols go to A -> more crimes are recorded
    in A -> the next training set holds more crime in A -> back to the model.
    Each arrow is labeled with the step it stands for.  Built with the
    figstyle SVG composer (text outlined, byte-idempotent)."""
    from figstyle.svg import Figure
    T = fl.T
    f = Figure()
    xl, xr, yt, yb = 0.0, 390.0, 0.0, 120.0
    w = 300.0
    boxes = {
        "model": f.block(xl, yt, "model predicts more crime in A",
                         role="stream", min_w=w),
        "patrol": f.block(xr, yt, "more patrols go to A", min_w=w),
        "record": f.block(xr, yb, "more crimes recorded in A", min_w=w),
        "data": f.block(xl, yb, "training data hold more crime in A",
                        min_w=w),
    }
    gap = 6.0
    (mx, my, mw, mh) = boxes["model"]
    (px, py, pw, ph) = boxes["patrol"]
    (rx, ry, rw, rh) = boxes["record"]
    (dx, dy, dw, dh) = boxes["data"]
    # top: model -> patrols; right: patrols -> records; bottom: records ->
    # data; left: data -> model (clockwise)
    f.arrow(mx + mw + gap, yt, px - gap, yt)
    f.arrow(xr, py + ph + gap, xr, ry - gap)
    f.arrow(rx - gap, yb, dx + dw + gap, yb)
    f.arrow(xl, dy - gap, xl, my + mh + gap)
    lab = dict(size=T.FS_SMALL, color=T.INK)
    f.text((mx + mw + px) / 2, yt - 16, "decision", **lab)
    f.text(xr + 14, (yt + yb) / 2, "observation", anchor="start", **lab)
    f.text((rx + dx + dw) / 2, yb + 16, "logging", **lab)
    f.text(xl - 14, (yt + yb) / 2, "retraining", anchor="end", **lab)
    f.save("mdl-clf-feedback-loop",
           desc="Predictive-policing feedback loop: predictions direct "
                "patrols, patrols determine which crimes are recorded, and "
                "the records become the next training data.")


FIGURES = [fig_loss_accuracy, fig_decision_regions, fig_temperature,
           fig_density_ratio, fig_shift_types, fig_feedback_loop]


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


def fig_margin_losses():
    """Why we train on cross-entropy but report accuracy. Both losses of one
    example as functions of its margin m = o_y - max_{j != y} o_j. The error
    (0-1 loss) is a step: 1 for m < 0, 0 for m > 0, with zero derivative
    wherever it has one. The two-class cross-entropy log(1 + e^{-m}), divided
    by log 2 so that it is measured in bits, is smooth and decreasing, passes
    through (0, 1), and lies above the step everywhere, so the summed loss in
    bits bounds the number of training errors. It never reaches 0 for m > 0:
    the loss keeps rewarding confidence after the decision is right."""
    T = fl.T
    m = np.linspace(-3.0, 4.0, 701)
    ce_bits = np.log1p(np.exp(-m)) / np.log(2)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axvspan(-3.0, 0.0, color=T.PANEL, zorder=0, lw=0)   # misclassified
    ax.plot([-3.0, 0.0], [1.0, 1.0], color=T.INK, lw=T.SW_LINE, zorder=3,
            solid_capstyle="butt")
    ax.plot([0.0, 4.0], [0.0, 0.0], color=T.INK, lw=T.SW_LINE, zorder=3,
            solid_capstyle="butt")
    ax.plot([0.0, 0.0], [0.0, 1.0], color=T.INK, lw=T.SW_HAIR,
            ls=(0, (1.5, 2.5)), zorder=3)
    ax.plot(m, ce_bits, color=T.BLUE.base, lw=T.SW_LINE, zorder=4)
    ax.plot([0.0], [1.0], "o", ms=6, color=T.BLUE.base, zorder=5)
    ax.text(-2.85, 1.12, "error", ha="left", va="bottom", fontsize=16,
            color=T.INK)
    ax.text(1.0, 0.95, r"cross-entropy $\log_2(1 + e^{-m})$", ha="left",
            va="bottom", fontsize=16, color=T.BLUE.dark)
    ax.text(-0.15, 2.9, "wrong", ha="right", va="top", fontsize=14,
            color=T.INK)
    ax.text(0.15, 2.9, "right", ha="left", va="top", fontsize=14,
            color=T.INK)
    ax.set_xlim(-3.0, 4.0)
    ax.set_ylim(-0.15, 3.0)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel(r"margin $m$")
    ax.set_ylabel("loss")
    ax.spines["left"].set_color(T.INK)
    ax.spines["bottom"].set_color(T.INK)
    ax.tick_params(colors=T.INK)
    fl.save(fig, "mdl-clf-margin-losses")


FIGURES.append(fig_margin_losses)


if __name__ == "__main__":
    main()
