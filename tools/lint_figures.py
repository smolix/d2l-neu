#!/usr/bin/env python3
"""Lint illustrative-figure conventions in chapter source .md files.

Enforces the figure rules from CLAUDE.md → "Content authoring" and the
`mdl-figure` skill: illustrative figures are pre-generated SVGs included with a
well-formed caption + attached `:label:`, with no drawing code in the notebook
and one consistent style per chapter.

Output format matches tools/lint_source.py — `path:line:col: error|warning: msg`
— so it feeds the same Problems pane / pre-commit hook.

  python3 tools/lint_figures.py                       # all chapter_*/*.md
  python3 tools/lint_figures.py chapter_x/foo.md ...  # a subset
  python3 tools/lint_figures.py --strict              # warnings fail too

Exit: 1 if any ERROR (or, with --strict, any WARNING); else 0.

Checks
  ERROR  image references ../img/<id>.svg that does not exist
  ERROR  image caption contains '[' or ']' (truncates markdown alt-text, detaches
         the label) — write matrices as \\begin{smallmatrix}…\\end{smallmatrix}
  ERROR  image not immediately followed by a :label:`fig_…` line
  WARN   referenced SVG carries a date/timestamp (non-idempotent; noisy diffs)
  WARN   notebook code cell contains figure-DRAWING primitives (should be a
         pre-generated figure, not inline matplotlib)
  WARN   chapter mixes generated figures with inline drawing code (one style/ch)
  WARN   (full-tree scan only) generated img/mdl-*.svg referenced by nothing
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# ── patterns ────────────────────────────────────────────────────────────────
IMG_WELL_FORMED = re.compile(r'!\[[^\[\]]*\]\((\.\./img/[^)]+)\)')
IMG_LOOSE = re.compile(r'!\[')
URL_IN_LINE = re.compile(r'\((\.\./img/[^)]+)\)')
LABEL_LINE = re.compile(r'^\s*:label:`(fig_[^`]+)`\s*$')
NUMREF = re.compile(r':numref:`(fig_[^`]+)`')
FENCE = re.compile(r'^```')
SVG_DATE = re.compile(r'<dc:date>|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
# Drawing primitives that mean "this cell draws a figure" rather than "this cell
# computes/plots a teaching result" (d2l.plot / plt.plot of computed data is fine).
DRAW_PRIMS = re.compile(
    r'\b(add_patch|FancyArrow\w*|Polygon\(|Arc\(|Rectangle\(|Circle\(|'
    r'quiver\(|annotate\(\s*["\']\s*["\']|\.savefig\()')


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lint_file(path, issues, referenced):
    root = repo_root()
    try:
        lines = open(path, encoding='utf-8').read().split('\n')
    except OSError as e:
        issues.append((path, 1, 1, 'error', f'cannot read: {e}'))
        return
    in_fence = False
    fence_buf, fence_start = [], 0
    has_generated_fig = False
    has_drawing_cell = False

    for i, line in enumerate(lines, 1):
        if FENCE.match(line):
            if in_fence:                       # closing fence → inspect the cell
                body = '\n'.join(fence_buf)
                if DRAW_PRIMS.search(body):
                    has_drawing_cell = True
                    issues.append((path, fence_start, 1, 'warning',
                        'code cell contains figure-drawing primitives — '
                        'pre-generate it via tools/gen_mdl_figures.py instead '
                        '(see the mdl-figure skill)'))
                fence_buf = []
            else:
                fence_start = i
            in_fence = not in_fence
            continue
        if in_fence:
            fence_buf.append(line)
            continue

        if IMG_LOOSE.search(line) and URL_IN_LINE.search(line):
            has_generated_fig = True
            url = URL_IN_LINE.search(line).group(1)
            stem = os.path.splitext(os.path.basename(url))[0]
            referenced.add(stem)
            # caption integrity: a stray [ or ] inside the alt truncates it
            if not IMG_WELL_FORMED.search(line):
                issues.append((path, i, 1, 'error',
                    "image caption contains '[' or ']' (truncates alt-text and "
                    "detaches the :label:); write matrices as "
                    "\\begin{smallmatrix}…\\end{smallmatrix}"))
            # file existence
            svg = os.path.join(root, 'img', f'{stem}.svg')
            if not os.path.exists(svg):
                issues.append((path, i, 1, 'error',
                    f'references ../img/{stem}.svg which does not exist '
                    f'(run tools/gen_mdl_figures.py?)'))
            else:
                try:
                    head = open(svg, encoding='utf-8', errors='replace').read(4000)
                    if SVG_DATE.search(head):
                        issues.append((os.path.relpath(svg, root), 1, 1, 'warning',
                            'SVG carries a date/timestamp — non-idempotent, will '
                            'churn in git; regenerate via save() (metadata Date:None)'))
                except OSError:
                    pass
            # the label must be on the next non-empty line
            j = i
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            m = LABEL_LINE.match(lines[j]) if j < len(lines) else None
            if not m:
                issues.append((path, i, 1, 'error',
                    'image not immediately followed by a :label:`fig_…` line '
                    '(the label will not attach)'))

    if has_generated_fig and has_drawing_cell:
        issues.append((path, 1, 1, 'warning',
            'chapter mixes generated figures with inline drawing code — keep '
            'one figure style per chapter (reproduce stragglers in the generator)'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='source .md files (default: all chapter_*/*.md)')
    ap.add_argument('--strict', action='store_true', help='warnings fail too')
    args = ap.parse_args()

    root = repo_root()
    full_scan = not args.files
    files = args.files or sorted(glob.glob(os.path.join(root, 'chapter_*', '**', '*.md'),
                                           recursive=True))
    # dedupe inputs (overlapping globs would double every issue)
    files = list(dict.fromkeys(os.path.normpath(f) for f in files))

    issues, referenced = [], set()
    for f in files:
        lint_file(f, issues, referenced)
    issues = list(dict.fromkeys(issues))  # collapse identical (e.g. one SVG, many refs)

    # Orphan generated figures — only meaningful on a full-tree scan.
    if full_scan:
        for svg in sorted(glob.glob(os.path.join(root, 'img', 'mdl-*.svg'))):
            stem = os.path.splitext(os.path.basename(svg))[0]
            if stem not in referenced:
                issues.append((os.path.relpath(svg, root), 1, 1, 'warning',
                    'generated figure is referenced by no chapter (orphan)'))

    issues.sort(key=lambda t: (t[0], t[1]))
    for path, ln, col, sev, msg in issues:
        rel = os.path.relpath(path, root) if os.path.isabs(path) else path
        print(f'{rel}:{ln}:{col}: {sev}: {msg}')

    errors = [i for i in issues if i[3] == 'error']
    warns = [i for i in issues if i[3] == 'warning']
    print(f'\nfigure lint: {len(errors)} error(s), {len(warns)} warning(s) '
          f'across {len(files)} file(s).', file=sys.stderr)
    return 1 if errors or (args.strict and warns) else 0


if __name__ == '__main__':
    sys.exit(main())
