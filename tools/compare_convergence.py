#!/usr/bin/env python3
"""Compare per-framework convergence across notebooks.

Reads the TSV produced by `extract_convergence.py` and reports cases
where one framework's training curve disagrees with the others. The
agreement metric is `rel_drop`: roughly +1 when the curve descended
sharply (so for a loss curve, "the loss went down a lot in data space"
— SVG y-pixels grow downward, so first_y < last_y means data went up
in figure-space, but matplotlib inverts y so data y went DOWN, which is
"loss decreased"). We flag frameworks whose `rel_drop` deviates from
the per-(notebook, color) median by more than a threshold.

Note: the SVG-pixel comparison assumes both frameworks rendered with
the same axis range. d2l ProgressBoard auto-scales, so a framework that
produced very different absolute losses will still have a normalized
`rel_drop` close to ±1 if the curve monotonically descended/ascended.
What we're really detecting is: "this framework's curve is FLAT
(rel_drop near 0) or going the WRONG direction (sign flip vs. peers)."

Usage:
    python tools/compare_convergence.py [--in convergence-all.tsv]

Outputs to stdout a Markdown report grouped by chapter with notebook ×
framework × color cells.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict


def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                r["rel_drop"] = float(r["rel_drop"])
                r["n_points"] = int(r["n_points"])
                r["first_y_px"] = float(r["first_y_px"])
                r["last_y_px"] = float(r["last_y_px"])
                r["min_y_px"] = float(r["min_y_px"])
                r["max_y_px"] = float(r["max_y_px"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def chapter_of(notebook):
    return notebook.split("/", 1)[0] if "/" in notebook else "(unknown)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="logs/convergence-all.tsv")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="rel_drop deviation from median to flag as anomaly")
    p.add_argument("--flat-threshold", type=float, default=0.2,
                   help="abs(rel_drop) below this is considered FLAT")
    p.add_argument("--min-points", type=int, default=5,
                   help="min n_points required to compare a curve (skip 1-2 epoch demos)")
    args = p.parse_args()

    rows = load_rows(args.inp)
    # Group by (notebook, color)
    grouped = defaultdict(dict)  # {(nb, color): {fw: row}}
    for r in rows:
        if r["n_points"] < args.min_points:
            continue
        grouped[(r["notebook"], r["color"])].setdefault(r["framework"], r)

    issues = []  # list of (chapter, notebook, color, framework, msg, details)

    for (nb, color), fw_rows in sorted(grouped.items()):
        if len(fw_rows) < 2:
            continue  # nothing to compare against
        rel_drops = {fw: row["rel_drop"] for fw, row in fw_rows.items()}
        median = statistics.median(rel_drops.values())
        # Detect:
        #  (1) sign flip vs. median
        #  (2) magnitude very different (|x - median| > threshold)
        #  (3) FLAT when others are descending/ascending
        for fw, rel in rel_drops.items():
            other_rels = [v for f, v in rel_drops.items() if f != fw]
            if not other_rels:
                continue
            other_med = statistics.median(other_rels)
            sign_flip = (rel * other_med < 0
                         and abs(rel) > 0.05
                         and abs(other_med) > 0.3)
            big_dev = abs(rel - other_med) > args.threshold
            flat = (abs(rel) < args.flat_threshold
                    and abs(other_med) > 0.6)
            if sign_flip or big_dev or flat:
                tag = []
                if sign_flip: tag.append("SIGN-FLIP")
                if flat: tag.append("FLAT")
                if big_dev and not (sign_flip or flat): tag.append("DRIFT")
                issues.append({
                    "chapter": chapter_of(nb),
                    "notebook": nb,
                    "color": color,
                    "color_name": fw_rows[fw]["color_name"],
                    "framework": fw,
                    "rel_drop": rel,
                    "others": rel_drops,
                    "tag": "/".join(tag),
                    "n_points": fw_rows[fw]["n_points"],
                })

    # Print Markdown report
    print("# Convergence Comparison Report\n")
    print(f"Source: `{args.inp}`. Threshold: |rel_drop − median| > "
          f"{args.threshold}; FLAT means |rel_drop| < {args.flat_threshold} "
          f"while peers have |rel_drop| > 0.6. Minimum {args.min_points} "
          f"points per curve.\n")
    print(f"Total notebooks compared: "
          f"{len({nb for (nb, _) in grouped})}\n")
    print(f"Total flagged (notebook, framework, curve) tuples: {len(issues)}\n")

    if not issues:
        print("**No anomalies found.** All flagged curves agreed across frameworks.\n")
        return

    by_ch = defaultdict(list)
    for i in issues:
        by_ch[i["chapter"]].append(i)

    for ch in sorted(by_ch):
        print(f"\n## {ch}\n")
        ch_issues = sorted(by_ch[ch], key=lambda x: (x["notebook"], x["color"], x["framework"]))
        for i in ch_issues:
            others_str = ", ".join(
                f"{f}={v:+.2f}" for f, v in sorted(i["others"].items()))
            print(f"- **`{i['notebook']}`** color={i['color']} ({i['color_name']}), "
                  f"`{i['framework']}` rel_drop={i['rel_drop']:+.2f} "
                  f"[{i['tag']}]; peers: {others_str} "
                  f"(n_pts={i['n_points']})")


if __name__ == "__main__":
    main()
