#!/usr/bin/env python3
"""Build a downloadable visual-design review bundle.

The bundle contains original style specimens, four current GAN diagrams, three
one-to-one revisions, and direct comparison sheets. External reference artwork
is never copied.
"""

from __future__ import annotations

import argparse
from math import erf, pi, sqrt
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import diagram_style as ds
from gen_visual_design_example import build as build_wide_architecture
from gen_visual_design_gan_revisions import build_all as build_gan_redesigns


ROOT = Path(__file__).resolve().parent.parent
GAN_FIGURES = {
    "figure-16.1.1-gan-architecture": "mdl-gan-architecture.svg",
    "gan-objective-template": "mdl-gan-template.svg",
    "gan-relativistic-pairing": "mdl-gan-pairing.svg",
    "gan-training-strategies": "mdl-gan-exits.svg",
}
GAN_REDESIGNS = {
    "gan-objective-template": "gan-objective-template-redesign",
    "gan-relativistic-pairing": "gan-relativistic-pairing-redesign",
    "gan-training-strategies": "gan-training-strategies-redesign",
}



def render_png(svg: Path, png: Path, width: int = 1800) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to render the PNG bundle")
    subprocess.run(
        [converter, "-w", str(width), str(svg), "-o", str(png)],
        check=True,
    )


def build_narrow_architecture(output: Path) -> Path:
    """A narrow composition showing how a labeled diagram can reflow."""
    with ds.style_context():
        fig, ax = ds.canvas(6.2, 9.0)
        ax.text(0.35, 8.72, "Narrow architecture composition",
                ha="left", va="top", fontsize=ds.TYPE.title,
                fontweight="semibold")
        ax.text(0.35, 8.32, "Main path", ha="left", va="top",
                fontsize=ds.TYPE.panel, fontweight="semibold")

        # Top-to-bottom overview, appropriate for a narrow reading column.
        ds.module(ax, (2.10, 7.45), 2.00, 0.62, "input tokens",
                  edge=ds.COLORS.border)
        ds.arrow(ax, (3.10, 7.45), (3.10, 6.92))
        ds.module(ax, (2.10, 6.30), 2.00, 0.62, "embedding",
                  face=ds.COLORS.blue_tint)
        ds.arrow(ax, (3.10, 6.30), (3.10, 5.82))
        ds.group(ax, (0.72, 4.48), 4.76, 1.34,
                 edge=ds.COLORS.blue, face=ds.COLORS.blue_tint)
        ax.text(0.95, 5.62, "repeated block × N", ha="left", va="top",
                fontsize=ds.TYPE.small, color=ds.COLORS.muted)
        ds.module(ax, (1.05, 4.78), 1.15, 0.58, "mix")
        ds.arrow(ax, (2.20, 5.07), (2.60, 5.07))
        ds.module(ax, (2.60, 4.78), 1.42, 0.58, "transform")
        ds.arrow(ax, (4.02, 5.07), (4.48, 5.07))
        ds.merge(ax, (4.70, 5.07), radius=0.13)
        ds.residual(ax, [(1.30, 4.78), (1.30, 4.62), (4.70, 4.62),
                         (4.70, 4.91)])
        ds.arrow(ax, (3.10, 4.48), (3.10, 3.98))
        ds.module(ax, (2.10, 3.36), 2.00, 0.62, "output",
                  edge=ds.COLORS.green, face=ds.COLORS.green_tint)

        # A separate detail panel avoids microscopic inset labels.
        ax.plot([0.35, 5.85], [3.02, 3.02], color=ds.COLORS.grid,
                linewidth=ds.STROKE.construction)
        ax.text(0.35, 2.78, "Block detail", ha="left", va="top",
                fontsize=ds.TYPE.panel, fontweight="semibold")
        detail_font = 9.5
        ds.module(ax, (0.35, 1.52), 0.90, 0.58, "input",
                  edge=ds.COLORS.border, fontsize=detail_font)
        ds.arrow(ax, (1.25, 1.81), (1.52, 1.81))
        ds.module(ax, (1.52, 1.52), 1.28, 0.58, "normalize",
                  fontsize=detail_font)
        ds.arrow(ax, (2.80, 1.81), (3.10, 1.81))
        ds.module(ax, (3.10, 1.52), 1.50, 0.58, "mix features",
                  face=ds.COLORS.blue_tint, fontsize=detail_font)
        ds.arrow(ax, (4.60, 1.81), (4.84, 1.81))
        ds.merge(ax, (5.03, 1.81), radius=0.13)
        ds.arrow(ax, (5.16, 1.81), (5.55, 1.81))
        ds.residual(ax, [(0.80, 1.52), (0.80, 1.15), (5.03, 1.15),
                         (5.03, 1.65)])
        ax.text(3.00, 0.80, "dashed purple: residual or feedback",
                ha="center", va="center", fontsize=ds.TYPE.small,
                color=ds.COLORS.purple)

        return ds.save_svg(
            fig, output,
            title="Narrow D2L architecture composition",
            description=(
                "A top-to-bottom model overview is followed by a separate, "
                "larger block-detail panel. The layout preserves readable "
                "labels in a narrow column instead of shrinking a wide canvas."
            ),
        )


def normal_density(x, mean):
    return np.exp(-0.5 * (x - mean) ** 2) / sqrt(2 * pi)


def gaussian_js(separations):
    grid = np.linspace(-10.0, 16.0, 30001)
    p = normal_density(grid, 0.0)
    values = []
    for separation in separations:
        q = normal_density(grid, separation)
        mixture = 0.5 * (p + q)
        left = np.where(p > 0, p * np.log(p / mixture), 0.0)
        right = np.where(q > 0, q * np.log(q / mixture), 0.0)
        values.append(0.5 * np.trapezoid(left + right, grid))
    return np.asarray(values)


def build_plot_specimen(output: Path) -> Path:
    """Two exact GAN quantities in a restrained, directly labeled plot."""
    with ds.style_context():
        fig, (left, right) = plt.subplots(2, 1, figsize=(7.6, 7.6))
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.94,
                            hspace=0.52)

        logits = np.linspace(-6.0, 6.0, 500)
        saturating = 1.0 / (1.0 + np.exp(-logits))
        nonsaturating = 1.0 - saturating
        left.plot(logits, saturating, color=ds.COLORS.orange,
                  linewidth=ds.STROKE.primary)
        left.plot(logits, nonsaturating, color=ds.COLORS.blue,
                  linewidth=ds.STROKE.primary, linestyle="--")
        left.axvline(0, color=ds.COLORS.grid,
                     linewidth=ds.STROKE.construction)
        left.set(xlabel="critic logit $D(x')$", ylabel="per-sample weight",
                 xlim=(-6, 6), ylim=(-0.02, 1.08))
        left.text(5.55, saturating[-1] - 0.01, "saturating",
                  ha="right", va="top", color=ds.COLORS.orange,
                  fontsize=ds.TYPE.small)
        left.text(5.55, nonsaturating[-1] + 0.04, "non-saturating",
                  ha="right", va="bottom", color=ds.COLORS.blue,
                  fontsize=ds.TYPE.small)
        left.set_title("(a)  Generator weights", loc="left")

        separation = np.linspace(0.0, 6.0, 100)
        js = gaussian_js(separation)
        tv = np.array([erf(value / (2 * sqrt(2))) for value in separation])
        right.plot(separation, tv, color=ds.COLORS.blue,
                   linewidth=ds.STROKE.primary)
        right.plot(separation, js, color=ds.COLORS.orange,
                   linewidth=ds.STROKE.primary, linestyle="--")
        right.axhline(np.log(2), color=ds.COLORS.grid,
                      linewidth=ds.STROKE.construction)
        right.text(0.18, np.log(2) + 0.025, "$\\log 2$ ceiling",
                   color=ds.COLORS.muted, fontsize=ds.TYPE.small,
                   ha="left", va="bottom")
        ds.end_label(right, separation[-1], tv[-1], "total variation",
                     color=ds.COLORS.blue, dx=0.08)
        ds.end_label(right, separation[-1], js[-1], "Jensen–Shannon",
                     color=ds.COLORS.orange, dx=0.08)
        right.set(xlabel="mean separation $|\\mu_p-\\mu_q|$",
                  ylabel="discrepancy", xlim=(0, 7.0), ylim=(-0.02, 1.08))
        right.set_title("(b)  Equal-variance Gaussians", loc="left")

        for axis in (left, right):
            axis.yaxis.grid(True, color=ds.COLORS.grid,
                            linewidth=ds.STROKE.construction, alpha=0.65)
            axis.set_axisbelow(True)
            axis.spines["left"].set_color(ds.COLORS.border)
            axis.spines["bottom"].set_color(ds.COLORS.border)

        return ds.save_svg(
            fig, output,
            title="D2L experimental-plot style specimen",
            description=(
                "The left panel directly labels the saturating and "
                "non-saturating GAN sample weights. The right panel directly "
                "labels total variation and Jensen–Shannon divergence between "
                "equal-variance Gaussian distributions as their means separate."
            ),
        )


def build_contact_sheet(pngs, output):
    montage = shutil.which("montage")
    if montage is None:
        return
    subprocess.run(
        [montage, *map(str, pngs), "-thumbnail", "900x650",
         "-tile", "2x", "-geometry", "+24+36", "-background", "white",
         "-set", "label", "%t", str(output)],
        check=True,
    )

def build_pair_sheet(current, redesign, output):
    montage = shutil.which("montage")
    if montage is None:
        return
    subprocess.run(
        [montage,
         "-label", "CURRENT", str(current),
         "-label", "PROPOSED REDESIGN", str(redesign),
         "-thumbnail", "1200x1000", "-tile", "2x1",
         "-geometry", "+42+46", "-background", "white", str(output)],
        check=True,
    )



def build_bundle(output_dir: Path) -> None:
    proposed = output_dir / "proposed-style"
    existing = output_dir / "existing-gan"
    redesigned = output_dir / "redesigned-gan"
    comparisons = output_dir / "comparisons"
    proposed.mkdir(parents=True, exist_ok=True)
    existing.mkdir(parents=True, exist_ok=True)
    redesigned.mkdir(parents=True, exist_ok=True)
    comparisons.mkdir(parents=True, exist_ok=True)

    wide_svg = proposed / "architecture-overview-detail.svg"
    narrow_svg = proposed / "architecture-narrow.svg"
    plot_svg = proposed / "gan-plot-style.svg"
    build_wide_architecture(wide_svg)
    build_narrow_architecture(narrow_svg)
    build_plot_specimen(plot_svg)

    style_pngs = []
    for svg in (wide_svg, narrow_svg, plot_svg):
        png = svg.with_suffix(".png")
        render_png(svg, png)
        style_pngs.append(png)

    current_pngs = {}
    for stem, filename in GAN_FIGURES.items():
        source = ROOT / "img" / filename
        target = existing / f"{stem}.svg"
        shutil.copy2(source, target)
        png = target.with_suffix(".png")
        render_png(target, png)
        current_pngs[stem] = png

    redesign_pngs = {}
    build_gan_redesigns(redesigned)
    for redesign_stem in GAN_REDESIGNS.values():
        svg = redesigned / f"{redesign_stem}.svg"
        png = svg.with_suffix(".png")
        render_png(svg, png)
        redesign_pngs[redesign_stem] = png

    paired_pngs = []
    for current_stem, redesign_stem in GAN_REDESIGNS.items():
        current = current_pngs[current_stem]
        redesign = redesign_pngs[redesign_stem]
        paired_pngs.extend((current, redesign))
        build_pair_sheet(
            current, redesign,
            comparisons / f"{current_stem}-current-vs-redesign.png",
        )

    build_contact_sheet(
        paired_pngs, output_dir / "comparison-contact-sheet.png")
    build_contact_sheet(
        style_pngs, output_dir / "style-specimens-contact-sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "visual-design-examples",
        help="review-bundle directory",
    )
    args = parser.parse_args()
    build_bundle(args.output_dir)
    print(f"wrote visual review bundle to {args.output_dir}")


if __name__ == "__main__":
    main()
