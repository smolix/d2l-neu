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
    """The two roles of a classifier's scores. One forward pass produces a
    logit vector o; from there the picture forks. The TOP (training) branch
    softmaxes o into probabilities and reads off the differentiable
    cross-entropy loss that gradient descent minimizes. The BOTTOM (evaluation)
    branch takes the argmax to a single hard decision, compares it with the
    label, and counts it for accuracy --- a discrete number with no useful
    gradient. Both branches read the SAME o, so the numbers in the picture are
    a real, exact softmax / cross-entropy of one concrete logit vector."""
    # One concrete example: 3-class logits, true label y = 1 (the middle class).
    o = np.array([1.0, 2.2, 0.3])           # the model's score vector (logits)
    y = 1                                     # ground-truth class index
    p = np.exp(o - o.max())                   # numerically stable softmax
    p = p / p.sum()
    yhat = int(np.argmax(p))                  # hard decision
    loss = float(-np.log(p[y]))               # cross-entropy of the true class
    correct = int(yhat == y)
    # Strings shown in the figure are formatted from the SAME computed numbers,
    # so the picture can never drift from the real softmax / cross-entropy.
    o_str = "(" + ",\\ ".join(f"{v:.1f}" for v in o) + ")"
    p_str = "(" + ",\\ ".join(f"{v:.2f}" for v in p) + ")"
    loss_str = f"{loss:.2f}"
    print(f"  logits o = {o.tolist()}")
    print(f"  softmax p = {np.round(p, 3).tolist()}  ->  argmax y_hat = {yhat}")
    print(f"  true y = {y}   cross-entropy = {loss:.3f}   correct = {correct}")

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(cx, cy, w, h, title, sub, color, fc_alpha=0.10, title_fs=12.5,
            sub_fs=9.5, mono=False):
        """Rounded box centred at (cx, cy) with a bold title and a smaller
        second line (rendered in math/mono). Returns useful anchor points."""
        x, y = cx - w / 2, cy - h / 2
        for fc, a in [(color, fc_alpha), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        if sub:
            ax.text(cx, cy + 0.30, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=color)
            ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                    fontsize=sub_fs, color="black",
                    family="monospace" if mono else None)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=color)
        return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)

    y_mid = 3.0
    y_top = 4.55
    y_bot = 1.45

    # --- shared trunk: input x -> model -> logits o ---
    xin = box(1.35, y_mid, 1.7, 1.0, r"input $\mathbf{x}$", "", GRAY,
              fc_alpha=0.08, title_fs=12)
    model = box(4.0, y_mid, 2.2, 1.2, "model", r"$f_{\mathbf{w}}$", BLUE,
                title_fs=12.5, sub_fs=14)
    # logits box shows the actual score vector
    logits = box(6.95, y_mid, 2.7, 1.3, r"logits $\mathbf{o}$",
                 rf"${o_str}$", BLUE, title_fs=12, sub_fs=12)

    fl.arrow(ax, (xin["r"], y_mid), (model["l"], y_mid), color=GRAY, lw=1.8)
    fl.arrow(ax, (model["r"], y_mid), (logits["l"], y_mid), color=GRAY, lw=1.8)

    # --- the fork: create the branch boxes FIRST, so the fork arrows can
    #     terminate exactly at their left edges (no overshoot through the box) ---
    fork_x = logits["r"]
    sm = box(10.5, y_top, 2.05, 0.95, "softmax", "", ORANGE, fc_alpha=0.10,
             title_fs=11.5)
    am = box(10.5, y_bot, 2.05, 0.95, "argmax", "", GREEN, fc_alpha=0.10,
             title_fs=11.5)
    # top (training) branch and bottom (evaluation) branch
    fl.arrow(ax, (fork_x, y_mid + 0.18), (sm["l"], y_top), color=ORANGE, lw=1.9)
    fl.arrow(ax, (fork_x, y_mid - 0.18), (am["l"], y_bot), color=GREEN, lw=1.9)

    # ---------- TOP branch: softmax -> p_hat -> cross-entropy loss ----------
    phat = box(13.6, y_top, 2.9, 1.15, r"probs $\hat{\mathbf{y}}$",
               rf"${p_str}$", ORANGE, title_fs=11.5, sub_fs=11)
    fl.arrow(ax, (sm["r"], y_top), (phat["l"], y_top), color=ORANGE, lw=1.8)

    # ---------- BOTTOM branch: argmax -> y_hat -> compare with y ----------
    cmp_sub = "correct" if correct else "wrong"
    cmp = box(13.6, y_bot, 2.9, 1.15, rf"$\hat{{y}}={yhat}$  vs  $y={y}$",
              cmp_sub, GREEN, title_fs=11, sub_fs=10.5)
    fl.arrow(ax, (am["r"], y_bot), (cmp["l"], y_bot), color=GREEN, lw=1.8)

    # ---------- branch end-labels: loss (top) and accuracy (bottom) ----------
    ax.text(13.6, y_top + 1.02,
            rf"cross-entropy loss $\ell = {loss_str}$", ha="center", va="center",
            fontsize=10.5, color=ORANGE, fontweight="bold")
    ax.text(13.6, y_bot - 1.02,
            "accuracy (count correct)", ha="center", va="center",
            fontsize=10.5, color=GREEN, fontweight="bold")

    # ---------- the one teaching line per branch ----------
    ax.text(11.55, y_top - 1.18,
            "differentiable\nused for gradient descent",
            ha="center", va="center", fontsize=9, color=ORANGE, style="italic")
    ax.text(11.55, y_bot + 1.18,
            "discrete (zero gradient)\nused for benchmarks",
            ha="center", va="center", fontsize=9, color=GREEN, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fl.save(fig, "mdl-clf-loss-accuracy")


def fig_decision_regions():
    """What a (multiclass) linear classifier's decision LOOKS like. Left: a
    3-class softmax regression partitions the plane into three convex
    polyhedral regions --- the argmax of three affine score functions --- whose
    straight pairwise boundaries (the ties o_i = o_j) meet at the single point
    where all three scores are equal. Right: the binary special case. The
    predicted probability sigma(o) depends on x only through the affine logit
    gap o = w.x + b, so its level sets are parallel lines perpendicular to w,
    with the decision boundary at probability 1/2. Regions/boundaries are
    computed exactly from the affine scores, not sketched."""
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
    axl.plot(*p0, "o", ms=6, color="black", zorder=5)
    axl.annotate(r"$o_1 = o_2 = o_3$", p0, xytext=(p0[0] - 2.0, p0[1] - 1.1),
                 fontsize=9.5, color="black",
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    # label each region at a representative interior point
    for cls, col, pos in [(0, BLUE, (0.15, 2.1)), (1, ORANGE, (-2.2, 0.2)),
                          (2, GREEN, (2.0, -1.4))]:
        axl.text(*pos, f"class {cls + 1}", ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=col)
        axl.text(pos[0], pos[1] - 0.42,
                 rf"$o_{cls + 1} \geq o_j\ \forall j$",
                 ha="center", va="center", fontsize=9, color=col)
    fl.clean_axes(axl, lim=((-lim, lim), (-lim, lim)))
    axl.set_xticks([]), axl.set_yticks([])
    axl.set_xlabel(r"$x_1$"), axl.set_ylabel(r"$x_2$")
    axl.set_title("three classes: convex regions, linear boundaries")

    # ---------------- RIGHT: binary sigma level sets ----------------
    w2, b2 = np.array([1.3, 0.9]), 0.0
    wn = w2 / np.linalg.norm(w2)
    tang = np.array([-wn[1], wn[0]])
    sigma_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    for s in sigma_levels:
        o = np.log(s / (1 - s))                     # logit: w.x + b = o
        c = (o - b2) / np.linalg.norm(w2)           # signed distance along w
        pt = c * wn
        far1, far2 = pt + tang * 3 * lim, pt - tang * 3 * lim
        main = abs(s - 0.5) < 1e-9
        axr.plot([far1[0], far2[0]], [far1[1], far2[1]],
                 color=BLUE if main else GRAY, lw=2.4 if main else 1.2,
                 ls="-" if main else (0, (5, 3)))
        rot = np.degrees(np.arctan2(tang[1], tang[0]))
        rot = rot - 180 if rot > 90 else (rot + 180 if rot <= -90 else rot)
        lbl_pos = pt + tang * (-2.0) + wn * 0.16
        axr.text(*lbl_pos, rf"$\hat y = {s}$", fontsize=8.5, color="black",
                 ha="center", va="center", rotation=rot)
    # weight vector from a point on the decision boundary
    base = ((np.log(1) - b2) / np.linalg.norm(w2)) * wn + tang * 1.15
    fl.arrow(axr, base, base + wn * 1.1, color=ORANGE, lw=2.2)
    axr.text(*(base + wn * 1.1 + np.array([0.28, 0.05])), r"$\mathbf{w}$",
             fontsize=12, color=ORANGE, ha="center", va="center")
    axr.text(-1.55, 1.9, r"$\hat y > \frac{1}{2}$", fontsize=11, color=BLUE,
             ha="center")
    axr.text(1.55, -2.1, r"$\hat y < \frac{1}{2}$", fontsize=11, color=GRAY,
             ha="center")
    fl.clean_axes(axr, lim=((-lim, lim), (-lim, lim)))
    axr.set_xticks([]), axr.set_yticks([])
    axr.set_xlabel(r"$x_1$"), axr.set_ylabel(r"$x_2$")
    axr.set_title(r"two classes: level sets of $\hat y = \sigma(o)$")

    fig.subplots_adjust(wspace=0.18)
    fl.save(fig, "mdl-clf-decision-regions")


def fig_temperature():
    """One 3-class score vector pushed through softmax(o / T) at three
    temperatures. Cooling (T < 1) sharpens toward the hard argmax, T = 1 is
    the plain softmax, heating (T > 1) flattens toward uniform --- and the
    ORDER of the bars never changes, since 1/T scales all logits alike. Uses
    the same logits (1.0, 2.2, 0.3) as the loss/accuracy figure, so the middle
    panel repeats numbers the reader has already seen. All bar heights are the
    exact computed softmax values."""
    o = np.array([1.0, 2.2, 0.3])
    temps = [0.25, 1.0, 4.0]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, T in zip(axes, temps):
        z = o / T
        p = np.exp(z - z.max())
        p = p / p.sum()
        print(f"  T = {T:>4}: softmax(o/T) = {np.round(p, 3).tolist()}")
        bars = ax.bar([1, 2, 3], p, width=0.62, color=[BLUE, ORANGE, GREEN],
                      alpha=0.85)
        for x, v in zip([1, 2, 3], p):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([r"$\hat y_1$", r"$\hat y_2$", r"$\hat y_3$"])
        title = {0.25: r"$T = 0.25$ (sharpened)", 1.0: r"$T = 1$ (softmax)",
                 4.0: r"$T = 4$ (flattened)"}[T]
        ax.set_title(title)
    axes[0].set_ylabel(r"$\mathrm{softmax}(\mathbf{o}/T)$")
    fig.suptitle(r"the same scores $\mathbf{o} = (1.0,\ 2.2,\ 0.3)$ at three temperatures",
                 y=1.04, fontsize=12)
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-clf-temperature")


def fig_density_ratio():
    """The geometry of importance weighting under covariate shift. One panel:
    the source density q (where the training data lives), the target density p
    (where the risk puts its weight), and the importance weight
    beta(x) = p(x)/q(x), which is ~0 where only the source has mass, crosses 1
    where the densities agree, and explodes exponentially where the target
    outweighs a vanishing source. A dashed line shows the clipped weight
    min(beta, c): a little bias for much less variance. All curves are the
    exact computed densities/ratio for two unit-variance Gaussians."""
    x = np.linspace(-3.5, 6.0, 601)
    mu_q, mu_p = 0.0, 2.5
    q = np.exp(-0.5 * (x - mu_q) ** 2) / np.sqrt(2 * np.pi)
    p = np.exp(-0.5 * (x - mu_p) ** 2) / np.sqrt(2 * np.pi)
    beta = p / q                       # = exp(2.5 x - 3.125) here
    c = 5.0
    # beta = 1 exactly where the two densities cross
    x_eq = (mu_p**2 - mu_q**2) / (2 * (mu_p - mu_q))
    x_clip = (np.log(c) + (mu_p**2 - mu_q**2) / 2) / (mu_p - mu_q)
    print(f"  beta = 1 at x = {x_eq:.3f};  beta = c = {c} at x = {x_clip:.3f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, q, color=BLUE, lw=2.2)
    ax.fill_between(x, 0, q, color=BLUE, alpha=0.10)
    ax.plot(x, p, color=GREEN, lw=2.2)
    ax.fill_between(x, 0, p, color=GREEN, alpha=0.10)
    ax.text(mu_q, 0.415, r"source $q(x)$" + "\n(training data)", ha="center",
            fontsize=10, color=BLUE)
    ax.text(mu_p, 0.415, r"target $p(x)$" + "\n(deployment)", ha="center",
            fontsize=10, color=GREEN)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 0.50)

    # the weight on a twin axis (its scale is different in kind)
    ax2 = ax.twinx()
    ax2.plot(x, beta, color=ORANGE, lw=2.4)
    ax2.plot(x, np.minimum(beta, c), color=ORANGE, lw=2.0, ls=(0, (5, 3)))
    ax2.axhline(1.0, color=GRAY, lw=0.9, ls=":")
    ax2.plot([x_eq], [1.0], "o", ms=5, color=GRAY)
    ax2.text(x_eq - 0.35, 1.35, r"$\beta = 1$", fontsize=9.5, color="black",
             ha="center")
    ax2.text(4.45, 6.6, r"$\beta(x) = \dfrac{p(x)}{q(x)}$", fontsize=12,
             color=ORANGE, ha="center")
    ax2.text(4.7, 4.35, r"clipped $\min(\beta, c)$", fontsize=9.5,
             color=ORANGE, ha="center", style="italic")
    ax2.set_ylim(0, 8)
    ax2.set_ylabel(r"importance weight $\beta$", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ORANGE)
    ax2.spines["top"].set_visible(False)

    fl.save(fig, "mdl-clf-density-ratio")


FIGURES = [fig_loss_accuracy, fig_decision_regions, fig_temperature,
           fig_density_ratio]


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
