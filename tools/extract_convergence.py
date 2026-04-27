#!/usr/bin/env python3
"""Extract per-framework training-loss curves from executed d2l notebooks.

For each notebook with a matplotlib-SVG plot in its last code-cell output
(this is how d2l's ProgressBoard renders training curves), parse the
polyline points encoded in the SVG and emit one row per (notebook,
framework, curve) covering:

    label, color, first_y_px, last_y_px, min_y_px, max_y_px, n_points,
    rel_drop = (first_y_px - last_y_px) / (max_y_px - min_y_px + 1e-9)

`rel_drop` is +1.0 when the curve plummeted from its max start to its
min end (textbook convergence, since SVG y-pixels grow downward). It is
~0 when the curve stayed flat. It is negative when the curve went up
(divergence).

We can't recover absolute data-coord loss values without parsing the
matplotlib axis transform, but for cross-framework convergence
*comparison* we only need the relative behavior: does the curve in
framework X drop similarly to frameworks Y/Z, or did it stay flat?

Usage:
    python tools/extract_convergence.py [--out convergence.tsv]
"""

import argparse
import json
import os
import re
import sys
from glob import glob

# matplotlib's default qualitative palette — d2l uses these in order for
# train_loss / val_loss / val_acc / etc. The order in which curves are
# drawn depends on what d2l.ProgressBoard registers first; we detect by
# stroke color and then map color → conventional curve name where
# possible.
PALETTE = {
    "#1f77b4": "blue",       # often train_loss
    "#ff7f0e": "orange",     # often val_loss
    "#2ca02c": "green",      # often val_acc
    "#d62728": "red",
    "#9467bd": "purple",
    "#8c564b": "brown",
    "#e377c2": "pink",
    "#7f7f7f": "gray",
    "#bcbd22": "olive",
    "#17becf": "cyan",
}

# matplotlib emits each Line2D as either:
#   <path d="M ... L ..."  style="fill:none;stroke:#xxxxxx;..."/>
# or:
#   <g style="..."><use ... xlink:href="#m..." x="..." y="..."/></g>
# The first form is what we want. The second is for marker symbols.

# A practical regex: find every <path ...> tag, look for fill:none and
# a stroke color in its style attribute, and parse the d= path data.
PATH_RE = re.compile(
    r'<path\b[^>]*\bd="(?P<d>[^"]+)"[^>]*\bstyle="(?P<style>[^"]+)"',
    re.DOTALL,
)
PATH_RE_REVERSE = re.compile(
    r'<path\b[^>]*\bstyle="(?P<style>[^"]+)"[^>]*\bd="(?P<d>[^"]+)"',
    re.DOTALL,
)
COORD_RE = re.compile(r"[ML]\s*([\-0-9.eE]+)\s+([\-0-9.eE]+)")
STROKE_RE = re.compile(r"stroke:\s*(#[0-9a-fA-F]{3,6})")
FILL_NONE = re.compile(r"fill:\s*none\b")


def parse_lines(svg_text):
    """Return list of dicts: {'color': '#rrggbb', 'points': [(x, y), ...]}.

    Skips paths shorter than 3 points (those are markers/ticks/etc.) and
    skips paths whose color is in the gridline set ('#cccccc',
    '#000000', plus very-light grays).
    """
    seen = set()
    lines = []
    for pat in (PATH_RE, PATH_RE_REVERSE):
        for m in pat.finditer(svg_text):
            style = m.group("style")
            if not FILL_NONE.search(style):
                continue
            sm = STROKE_RE.search(style)
            if not sm:
                continue
            color = sm.group(1).lower()
            # Skip gridlines and axis lines.
            if color in ("#000000", "#cccccc", "#dddddd", "#eeeeee", "#ffffff", "#b0b0b0"):
                continue
            d = m.group("d")
            coords = COORD_RE.findall(d)
            if len(coords) < 3:
                continue
            pts = [(float(x), float(y)) for x, y in coords]
            # Dedupe by color + first/last to avoid re-counting the same
            # path matched by both regex orderings.
            key = (color, pts[0], pts[-1], len(pts))
            if key in seen:
                continue
            seen.add(key)
            lines.append({"color": color, "points": pts})
    return lines


def parse_axis_text_labels(svg_text):
    """Extract numeric tick labels appearing as <text>...</text>.

    Returns a sorted-unique list of floats. We can use min/max as a rough
    bracket on the y-axis data range; the x-axis ticks contaminate this
    list but for our purposes (relative comparison) the min and max of
    the combined set roughly bracket the loss values.
    """
    nums = []
    for m in re.finditer(r">([\-0-9eE.]+)<", svg_text):
        s = m.group(1).strip()
        if not s:
            continue
        try:
            nums.append(float(s))
        except ValueError:
            pass
    return sorted(set(nums))


def find_legend_color_to_label(svg_text):
    """Parse matplotlib's legend block to map curve color -> label.

    Matplotlib emits pairs in the legend like:
        <g id="line2d_NNN">
          <path d="M ..." style="... stroke: #color; stroke-width: 1.5"/>
        </g>
        <g id="text_NN">
          <!-- label_text -->
          ... glyph uses ...
        </g>

    The path under `line2d_*` in the legend is a tiny 20px-wide swatch.
    The immediately-following `text_*` block carries the human label as
    a `<!-- label -->` comment that matplotlib emits before the glyphs.
    """
    color_to_label = {}
    pat = re.compile(
        r'<g id="line2d_\d+">\s*'
        r'<path[^>]*stroke:\s*(#[0-9a-fA-F]{3,6})[^>]*stroke-width:\s*1\.5"\s*/>\s*'
        r'</g>\s*'
        r'<g id="text_\d+">\s*'
        r'<!--\s*([^>]+?)\s*-->',
        re.DOTALL,
    )
    for m in pat.finditer(svg_text):
        color = m.group(1).lower()
        label = m.group(2).strip()
        if color not in color_to_label:
            color_to_label[color] = label
    return color_to_label


def find_plot_outputs(nb):
    """Yield (cell_idx, svg_text) for every code cell that has an
    image/svg+xml output."""
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            data = o.get("data", {})
            v = data.get("image/svg+xml")
            if v is None:
                continue
            if isinstance(v, list):
                v = "".join(v)
            yield i, v


def has_training(nb):
    """Quick check: is this a training notebook (with `trainer.fit` or
    `d2l.train_*`)?"""
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "trainer.fit(" in src or "d2l.train_" in src or "train_ranking" in src:
            return True
    return False


def summarize_notebook(nb_path):
    """Return list of curve summaries from the LAST plot cell with
    training-curve-shaped output (i.e., a plot cell that is also a
    training cell — we approximate by taking the last cell with SVG output
    in a training notebook).
    """
    with open(nb_path) as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError:
            return None
    if not has_training(nb):
        return None
    last = None
    for cell_idx, svg in find_plot_outputs(nb):
        last = (cell_idx, svg)
    if last is None:
        return None
    cell_idx, svg = last
    lines = parse_lines(svg)
    axis_nums = parse_axis_text_labels(svg)
    color_to_label = find_legend_color_to_label(svg)
    # d2l's ProgressBoard re-emits a fresh path for the curve at each
    # animation step, so a single SVG can contain dozens of overlapping
    # paths per color. Keep only the LONGEST path per color — that is the
    # final, fully-drawn curve.
    longest_per_color = {}
    for line in lines:
        c = line["color"]
        if c not in longest_per_color or len(line["points"]) > len(longest_per_color[c]["points"]):
            longest_per_color[c] = line
    lines = list(longest_per_color.values())
    out = []
    for line in lines:
        ys = [p[1] for p in line["points"]]
        first_y = ys[0]
        last_y = ys[-1]
        miny = min(ys)
        maxy = max(ys)
        rng = maxy - miny + 1e-9
        rel_drop = (first_y - last_y) / rng
        label = color_to_label.get(line["color"], "")
        out.append(
            {
                "color": line["color"],
                "color_name": PALETTE.get(line["color"], "?"),
                "label": label,
                "n_points": len(ys),
                "first_y_px": first_y,
                "last_y_px": last_y,
                "min_y_px": miny,
                "max_y_px": maxy,
                "rel_drop": rel_drop,
                "axis_min": min(axis_nums) if axis_nums else None,
                "axis_max": max(axis_nums) if axis_nums else None,
                "cell_idx": cell_idx,
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--notebooks-dir", default="/home/smola/d2l/d2l-neu/_notebooks"
    )
    p.add_argument("--out", default=None, help="TSV output path; default stdout")
    args = p.parse_args()

    fws = ["pytorch", "tensorflow", "jax", "mxnet"]
    rows = []
    for fw in fws:
        fw_root = os.path.join(args.notebooks_dir, fw)
        if not os.path.isdir(fw_root):
            continue
        for nb_path in sorted(glob(os.path.join(fw_root, "chapter_*/*.ipynb"))):
            rel = os.path.relpath(nb_path, fw_root)
            try:
                summary = summarize_notebook(nb_path)
            except Exception as e:
                print(f"# error reading {nb_path}: {e}", file=sys.stderr)
                continue
            if not summary:
                continue
            for s in summary:
                rows.append(
                    {
                        "notebook": rel,
                        "framework": fw,
                        **s,
                    }
                )

    cols = [
        "notebook",
        "framework",
        "color",
        "color_name",
        "label",
        "n_points",
        "first_y_px",
        "last_y_px",
        "min_y_px",
        "max_y_px",
        "rel_drop",
        "axis_min",
        "axis_max",
        "cell_idx",
    ]
    out = open(args.out, "w") if args.out else sys.stdout
    out.write("\t".join(cols) + "\n")
    for r in rows:
        out.write(
            "\t".join(str(r.get(c, "")) for c in cols) + "\n"
        )
    if args.out:
        out.close()
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
