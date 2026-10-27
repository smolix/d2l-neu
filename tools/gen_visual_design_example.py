#!/usr/bin/env python3
"""Generate an original example of the D2L diagram grammar.

The example is documentation, not a model architecture. It demonstrates a
dominant tensor path, compressed repetition, a residual edge, a zoomed detail,
semantic accents, and short external callouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import diagram_style as ds


def build(output: Path) -> Path:
    with ds.style_context():
        fig, ax = ds.canvas(12.0, 6.6)

        # Overview: one dominant path, compressed repetition, and a residual.
        ds.group(ax, (0.35, 0.55), 6.25, 5.30)
        ax.text(0.45, 6.20, "Model overview", ha="left", va="top",
                fontsize=ds.TYPE.panel, fontweight="semibold")
        ds.module(ax, (0.80, 0.95), 1.45, 0.66, "input tokens",
                  edge=ds.COLORS.border)
        ds.arrow(ax, (1.525, 1.61), (1.525, 2.05))
        ds.module(ax, (0.70, 2.05), 1.65, 0.78, "embedding",
                  edge=ds.COLORS.blue, face=ds.COLORS.blue_tint)
        ds.arrow(ax, (1.525, 2.83), (1.525, 3.20))

        ds.group(ax, (0.62, 3.20), 4.78, 1.55,
                 edge=ds.COLORS.blue, face=ds.COLORS.blue_tint)
        ax.text(5.15, 4.58, "repeated block × N", ha="right", va="top",
                fontsize=ds.TYPE.small, color=ds.COLORS.muted)
        ds.module(ax, (0.88, 3.55), 1.30, 0.72, "mix")
        ds.arrow(ax, (2.18, 3.91), (2.62, 3.91))
        ds.module(ax, (2.62, 3.55), 1.44, 0.72, "transform")
        ds.arrow(ax, (4.06, 3.91), (4.56, 3.91))
        ds.merge(ax, (4.78, 3.91))
        ds.residual(ax, [(1.05, 3.55), (1.05, 3.37), (4.78, 3.37),
                         (4.78, 3.75)])

        ds.arrow(ax, (1.525, 4.75), (1.525, 5.00))
        ds.module(ax, (0.70, 5.00), 1.65, 0.62, "output",
                  edge=ds.COLORS.green, face=ds.COLORS.green_tint)
        ds.callout(ax, "The main path uses\none visual direction.",
                   xy=(1.525, 1.82), xytext=(3.30, 1.40),
                   color=ds.COLORS.border)
        # Detail inset: the same highlighted component at a larger scale.
        ds.inset_link(ax, (0.82, 3.48), 1.42, 0.87,
                      (7.05, 0.55), 4.45, 5.30)
        ax.text(7.15, 6.20, "Detail of the mixing block", ha="left", va="top",
                fontsize=ds.TYPE.panel, fontweight="semibold")
        ds.module(ax, (7.45, 0.95), 1.25, 0.62, "input",
                  edge=ds.COLORS.border)
        ds.arrow(ax, (8.075, 1.57), (8.075, 2.05))
        ds.module(ax, (7.35, 2.05), 1.45, 0.72, "normalize")
        ds.arrow(ax, (8.075, 2.77), (8.075, 3.25))
        ds.module(ax, (7.28, 3.25), 1.60, 0.76, "mix features",
                  face=ds.COLORS.blue_tint)
        ds.arrow(ax, (8.88, 3.63), (9.45, 3.63))
        ds.merge(ax, (9.68, 3.63))
        ds.residual(ax, [(8.075, 1.57), (9.68, 1.57), (9.68, 3.47)])
        ds.arrow(ax, (9.68, 3.79), (9.68, 4.40))
        ds.module(ax, (9.03, 4.40), 1.30, 0.62, "output",
                  edge=ds.COLORS.green, face=ds.COLORS.green_tint)
        ds.callout(ax, "State tensor\n$B \\times T \\times d$",
                   xy=(8.08, 1.82), xytext=(10.55, 2.05),
                   color=ds.COLORS.border)

        return ds.save_svg(
            fig, output,
            title="D2L architecture-diagram grammar",
            description=(
                "A model overview shows input tokens flowing upward through an "
                "embedding, a repeated block, and an output. A dashed purple "
                "residual path bypasses two operations. Dotted leaders connect "
                "the repeated block to a larger detail view on the right."
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/d2l-visual-design-example.svg"),
        help="destination SVG (default: /tmp/d2l-visual-design-example.svg)",
    )
    args = parser.parse_args()
    print(f"wrote {build(args.output)}")


if __name__ == "__main__":
    main()
