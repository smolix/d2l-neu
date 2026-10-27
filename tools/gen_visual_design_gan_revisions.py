#!/usr/bin/env python3
"""Generate restrained revisions of three authored GAN figures.

Figure 16.1.1 is intentionally excluded: its distribution samples, terminology,
and update paths already explain the adversarial loop effectively. The revisions
here preserve the content of the remaining figures while changing only layouts
that obscure a relationship.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

import diagram_style as ds


ROOT = Path(__file__).resolve().parent.parent


def _text(ax, x, y, value, *, size=12.5, color=ds.COLORS.ink,
          weight="normal", ha="left", va="center", zorder=10):
    return ax.text(x, y, value, ha=ha, va=va, fontsize=size, color=color,
                   fontweight=weight, zorder=zorder)


def _method(ax, center, width, label, *, color, tint, fontsize=12.5):
    x, y = center
    return ds.module(ax, (x - width / 2, y - 0.34), width, 0.68, label,
                     edge=color, face=tint, fontsize=fontsize, radius=0.12,
                     linewidth=ds.STROKE.secondary)


def objective_template(output: Path) -> Path:
    """Flatten the template plane while preserving its two axes and members."""
    with ds.style_context():
        fig, ax = ds.canvas(10.6, 6.1)

        origin = (1.45, 1.18)
        right, top = 8.65, 4.62
        xs = [2.75, 5.10, 7.45]
        y_linear, y_nonlinear = 2.03, 3.73

        # The light rectangle is the domain of the objective template itself.
        ax.add_patch(FancyBboxPatch(
            origin, right - origin[0], top - origin[1],
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=ds.COLORS.surface, edgecolor=ds.COLORS.border,
            linewidth=ds.STROKE.construction, zorder=1))

        # Linear payoffs form the IPM family; the shaded band has a precise role.
        ax.add_patch(FancyBboxPatch(
            (origin[0] + 0.02, 1.50), right - origin[0] - 0.04, 1.05,
            boxstyle="square,pad=0", facecolor="#EEF4F2", edgecolor="none",
            zorder=2))
        _text(ax, 8.88, y_linear,
              "integral probability metrics\n$a(T)=b(T)=T$",
              size=12.5, color=ds.COLORS.muted, va="center")

        ds.arrow(ax, origin, (right + 0.30, origin[1]),
                 linewidth=ds.STROKE.secondary, mutation_scale=13)
        ds.arrow(ax, origin, (origin[0], top + 0.28),
                 linewidth=ds.STROKE.secondary, mutation_scale=13)
        _text(ax, 5.05, 0.52, "critic class $\\mathcal{T}$", size=14.5,
              weight="semibold", ha="center")
        _text(ax, 0.68, 3.15, "payoff\n$(a,b)$", size=14.5,
              weight="semibold", ha="center")

        for x, label in zip(xs, ("unrestricted", "Lipschitz ball", "RKHS ball")):
            ax.plot([x, x], [origin[1] - 0.10, origin[1] + 0.10],
                    color=ds.COLORS.ink, linewidth=ds.STROKE.construction)
            _text(ax, x, 0.92, label, size=12.5, ha="center", va="top")
        for y, label in ((y_linear, "linear"), (y_nonlinear, "nonlinear")):
            ax.plot([origin[0] - 0.10, origin[0] + 0.10], [y, y],
                    color=ds.COLORS.ink, linewidth=ds.STROKE.construction)
            _text(ax, 1.18, y, label, size=12.5, ha="right")

        ax.plot([xs[0]], [y_nonlinear], "o", color=ds.COLORS.blue,
                markersize=9, zorder=6)
        _text(ax, xs[0] + 0.18, y_nonlinear,
              "GAN, $f$-GAN ($f$-divergences)", size=12.5,
              color=ds.COLORS.blue)
        ax.plot([xs[1]], [y_linear], "o", color=ds.COLORS.green,
                markersize=9, zorder=6)
        _text(ax, xs[1], y_linear + 0.32, "$W_1$", size=12.5,
              color=ds.COLORS.green, ha="center")
        ax.plot([xs[2]], [y_linear], "o", color=ds.COLORS.orange,
                markersize=9, zorder=6)
        _text(ax, xs[2], y_linear + 0.32, "MMD", size=12.5,
              color=ds.COLORS.orange, ha="center")

        # This annotation is detached because paired scoring is not a third
        # payoff choice inside the template.
        ds.module(ax, (2.00, 5.05), 5.50, 0.62,
                  "relativistic pairing · scores a pair · outside this template",
                  edge=ds.COLORS.purple, face=ds.COLORS.purple_tint,
                  fontsize=12.5, radius=0.12,
                  linewidth=ds.STROKE.secondary)
        ds.residual(ax, [(xs[0], y_nonlinear + 0.10),
                         (xs[0], 5.05)], color=ds.COLORS.purple,
                    linewidth=ds.STROKE.secondary)

        return ds.save_svg(
            fig, output,
            title="Revised adversarial-objective template",
            description=(
                "A flat two-axis template places GAN and f-GAN at nonlinear "
                "payoff with an unrestricted critic, Wasserstein-1 at linear "
                "payoff with a Lipschitz critic, and MMD at linear payoff with "
                "an RKHS critic. A shaded linear band identifies integral "
                "probability metrics. Relativistic pairing is detached from the "
                "template because it scores a pair."
            ),
        )


def _score_y(value):
    lower, upper = -1.20, 2.20
    return 0.95 + (value - lower) / (upper - lower) * 2.75


def _score_axis(ax, x, *, label):
    ds.arrow(ax, (x, 0.82), (x, 3.93), linewidth=ds.STROKE.secondary,
             mutation_scale=12)
    zero = _score_y(0.0)
    ax.plot([x - 0.11, x + 0.11], [zero, zero], color=ds.COLORS.ink,
            linewidth=ds.STROKE.construction)
    _text(ax, x - 0.16, zero, "$0$", size=12.5, ha="right")
    _text(ax, x, 0.54, label, size=12.5, ha="center", va="top")


def _score_point(ax, x, value, label, outcome, color, *, side="right", dy=0.0):
    y = _score_y(value)
    ax.plot([x], [y], "o", color=color, markersize=9, zorder=6)
    direction = 1 if side == "right" else -1
    _text(ax, x + direction * 0.16, y + dy, label, size=12.5, color=color,
          ha="left" if direction > 0 else "right")
    if outcome:
        _text(ax, x + direction * 0.16, y - 0.28, outcome, size=12.5,
              color=ds.COLORS.ink,
              ha="left" if direction > 0 else "right")
    return y


def _difference(ax, x, low, high, label, *, side="right"):
    y0, y1 = _score_y(low), _score_y(high)
    direction = 1 if side == "right" else -1
    bx = x + direction * 0.28
    ax.plot([bx, bx], [y0, y1], color=ds.COLORS.ink,
            linewidth=ds.STROKE.secondary)
    ax.plot([bx - 0.08, bx + 0.08], [y0, y0], color=ds.COLORS.ink,
            linewidth=ds.STROKE.secondary)
    ax.plot([bx - 0.08, bx + 0.08], [y1, y1], color=ds.COLORS.ink,
            linewidth=ds.STROKE.secondary)
    _text(ax, bx + direction * 0.15, (y0 + y1) / 2, label, size=12.5,
          ha="left" if direction > 0 else "right")


def pairing(output: Path) -> Path:
    """Keep the original score axes while making the cancellation explicit."""
    with ds.style_context():
        fig, ax = ds.canvas(10.4, 5.0, equal=False)
        ax.plot([5.20, 5.20], [0.38, 4.62], color=ds.COLORS.grid,
                linewidth=ds.STROKE.construction)

        _text(ax, 0.35, 4.63, "(a)  one sample: is $D(x)$ above $0$?",
              size=14.5, weight="semibold", va="top")
        _text(ax, 5.47, 4.63, "(b)  a pair: is $D(x)$ above $D(x')$?",
              size=14.5, weight="semibold", va="top")

        # Pointwise threshold: the score crosses a fixed zero level.
        ax.plot([0.48, 4.84], [_score_y(0.0), _score_y(0.0)],
                color=ds.COLORS.ink, linewidth=ds.STROKE.construction,
                linestyle="--")
        _text(ax, 4.75, _score_y(0.0) + 0.17, "threshold",
              size=12.5, ha="right")
        _score_axis(ax, 1.25, label="critic $D$")
        _score_axis(ax, 3.85, label="shifted critic $D+c$")
        y_before = _score_point(ax, 1.25, -0.70, "$D(x)=-0.70$",
                                "generated", ds.COLORS.blue, side="right")
        y_after = _score_point(ax, 3.85, 0.80, "$D(x)+c=0.80$",
                               "real", ds.COLORS.blue, side="right")
        ds.arrow(ax, (1.55, y_before + 0.12), (3.53, y_after - 0.12),
                 color=ds.COLORS.purple, linewidth=ds.STROKE.secondary,
                 mutation_scale=12)
        _text(ax, 2.66, (y_before + y_after) / 2 + 0.24,
              "every score $+c$", size=12.5, color=ds.COLORS.purple,
              ha="center")

        # Pair ranking: both endpoints shift and the bracket length is fixed.
        _score_axis(ax, 6.38, label="critic $D$")
        _score_axis(ax, 9.00, label="shifted critic $D+c$")
        y_fake = _score_point(ax, 6.38, -0.95, "$D(x')$", "",
                              ds.COLORS.orange, side="left")
        y_real = _score_point(ax, 6.38, 0.45, "$D(x)$", "",
                              ds.COLORS.blue, side="left")
        y_fake_shift = _score_point(ax, 9.00, 0.55, "$D(x')+c$", "",
                                    ds.COLORS.orange, side="left", dy=-0.24)
        y_real_shift = _score_point(ax, 9.00, 1.95, "$D(x)+c$", "",
                                    ds.COLORS.blue, side="left", dy=0.24)
        _difference(ax, 6.38, -0.95, 0.45, "$1.40$", side="right")
        _difference(ax, 9.00, 0.55, 1.95, "$1.40$", side="right")
        ds.arrow(ax, (6.60, y_fake), (8.75, y_fake_shift),
                 color=ds.COLORS.orange, linewidth=ds.STROKE.construction,
                 mutation_scale=10)
        ds.arrow(ax, (6.60, y_real), (8.75, y_real_shift),
                 color=ds.COLORS.blue, linewidth=ds.STROKE.construction,
                 mutation_scale=10)
        _text(ax, 7.69, 4.02,
              "$[D(x)+c]-[D(x')+c]=D(x)-D(x')$",
              size=12.5, color=ds.COLORS.purple, ha="center")

        return ds.save_svg(
            fig, output,
            title="Revised pointwise and relativistic critic comparison",
            description=(
                "The left panel shows one score crossing the fixed zero "
                "threshold after an additive shift. The right panel shows real "
                "and generated scores both shifting by the same amount; brackets "
                "of equal length and the cancellation equation show that their "
                "difference remains 1.40."
            ),
        )


def _lane(ax, y, *, color, tint, heading, methods, tradeoff):
    # A short vertical rule marks the branch. The method boxes remain physically
    # separate because they are peer examples, not sequential stages.
    ax.plot([3.05, 3.05], [y - 0.42, y + 0.42], color=color,
            linewidth=ds.STROKE.primary, zorder=2)
    _text(ax, 3.05, y + 0.68, heading, size=13.5, color=color,
          weight="semibold")
    centers = [4.45, 8.00, 11.55]
    widths = [2.60, 2.90, 2.90]
    for center, width, method in zip(centers, widths, methods):
        _method(ax, (center, y), width, method, color=color, tint=tint,
                fontsize=12.5)
    _text(ax, 3.05, y - 0.67, tradeoff, size=12.5,
          color=ds.COLORS.muted, va="top")


def training_strategies(output: Path) -> Path:
    """Show each response as a set of peer methods, not a pipeline."""
    with ds.style_context():
        fig, ax = ds.canvas(13.8, 6.6, equal=False)
        root_x, root_y, root_w, root_h = 0.35, 2.68, 2.15, 1.20
        ds.module(ax, (root_x, root_y), root_w, root_h,
                  "unstable\nminimax training", edge=ds.COLORS.border,
                  face=ds.COLORS.surface, fontsize=12.5, radius=0.12)

        lanes = [
            (5.20, ds.COLORS.blue, ds.COLORS.blue_tint,
             "1. fix the game",
             ["Mescheder\n$R_1+R_2$", "convergent objective\n(RpGAN)", "R3GAN"],
             "cost: one more knob ($\\gamma$) and a minimax loop that remains"),
            (3.20, ds.COLORS.orange, ds.COLORS.orange_tint,
             "2. constrain the critic",
             ["spectral norm", "frozen pretrained\nfeatures",
              "controlled augmentation\n(ADA)"],
             "cost: the critic is no longer free—it inherits the features' bias"),
            (1.20, ds.COLORS.green, ds.COLORS.green_tint,
             "3. remove the game",
             ["MMD, IMLE", "diffusion and flow\nas regression",
              "the adversarial term\nreturns at distillation"],
             "cost: a fixed discrepancy weakens in high dimensions; sampling can be slow"),
        ]

        for y, color, tint, heading, methods, tradeoff in lanes:
            start_y = root_y + root_h / 2
            ds.arrow(ax, (root_x + root_w, start_y), (3.02, y),
                     color=color, linewidth=ds.STROKE.secondary,
                     mutation_scale=12,
                     connection="arc3,rad=0.14" if y > start_y else
                                ("arc3,rad=-0.14" if y < start_y else "arc3"))
            _lane(ax, y, color=color, tint=tint, heading=heading,
                  methods=methods, tradeoff=tradeoff)

        return ds.save_svg(
            fig, output,
            title="Revised responses to unstable adversarial training",
            description=(
                "Unstable minimax training branches into three responses: fix "
                "the game, constrain the critic, or remove the game. Each "
                "colored rail contains three peer examples with no arrows "
                "between them, followed by the corresponding cost."
            ),
        )


REDESIGNS = {
    "gan-objective-template-redesign": objective_template,
    "gan-relativistic-pairing-redesign": pairing,
    "gan-training-strategies-redesign": training_strategies,
}


def build_all(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, builder in REDESIGNS.items():
        written.append(builder(output_dir / f"{stem}.svg"))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "visual-design-examples" / "redesigned-gan",
    )
    args = parser.parse_args()
    for path in build_all(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
