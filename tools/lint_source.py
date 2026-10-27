#!/usr/bin/env python3
"""Single-pass linter for d2l source `.md` files.

Catches structural errors before they fail late in the build. Outputs
GCC-style messages (`path:line:col: severity: message`) so VS Code's
problem-matcher can surface them inline.

Checks performed (per file):
  - Unbalanced markers: `:begin_tab:` / `:end_tab:`,
    `::: {.slide}` / `:::`, `::: {.subslide}` / `:::`,
    `::: {.fragment}` / `:::`.
  - `@<id>` and `@<id>@<fw>` placeholders inside `.slide` divs
    reference an existing code fence ID.
  - Code fence IDs unique per file, OR framework-specific variants
    with non-overlapping `#@tab` sets.
  - Framework names valid (pytorch | tensorflow | jax | mxnet) in
    `#@tab`, `%%tab`, `tab.interact_select`, `:begin_tab:`, and
    placeholder `@<fw>` suffixes.
  - Unknown `:directive:` constructs.
  - Display math: a `\\\\` row break needs an alignment environment, and a
    multi-line display followed by `:eqlabel:` needs `$$` alone on its
    opening and closing lines (otherwise the label does not attach).
  - A continuation line of a list item that begins with a cross-reference
    followed by `.` or `)` (warn): pandoc reads `@sec-x.`, `@sec-x)` or
    `(@sec-x)` there as the start of a new block, which splits the item.
  - `:begin_tab:` / `:end_tab:` sharing a line with other text (error):
    the marker then renders literally.
  - A wrapped list-item line that starts `<n>. ` or `<n>) ` (warn): pandoc
    starts a new (or nested) list item there. `1.` and the next sibling
    number are exempt.
  - Cell-ID derivation drift (warn): the existing ID no longer matches
    the slug derivation from its surrounding section.

Cross-corpus checks (with `--corpus`):
  - `:numref:` / `:eqref:` / `:cite:` / `:ref:` point to existing
    labels.
  - `:label:` / `:eqlabel:` IDs are globally unique.

Performance target: <100 ms per file, <5 s for the whole corpus.

Usage:
    python tools/lint_source.py [--corpus] [files...]

Exit code 0 iff there are no errors. Warnings don't block.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (CHAPTER_NUMBERING, FRAMEWORKS,
                             extract_tab, is_python_block, is_boilerplate)
from add_cell_ids import (slugify, file_slug, collect_fences,
                           HEADING_RE)


# ──────────────────────────────────────────────────────────
# Issue collection
# ──────────────────────────────────────────────────────────

class Issue:
    __slots__ = ('path', 'line', 'col', 'severity', 'message')
    def __init__(self, path, line, col, severity, message):
        self.path = str(path)
        self.line = line
        self.col = col
        self.severity = severity  # 'error' or 'warning'
        self.message = message

    def emit(self):
        return (f'{self.path}:{self.line}:{self.col}: '
                f'{self.severity}: {self.message}')


# ──────────────────────────────────────────────────────────
# Per-file lint
# ──────────────────────────────────────────────────────────

# Allowed top-level directive names that can appear as `:foo:...`.
_KNOWN_DIRECTIVES = {
    'label', 'eqlabel', 'numref', 'eqref', 'cite', 'citet', 'ref',
    'exlabel', 'exref',
    'begin_tab', 'end_tab', 'class', 'func', 'mod',
    'width', 'height', 'bibliography',
}

_FW_NAMES = set(FRAMEWORKS) | {'all'}

_FENCE_OPEN_RE = re.compile(r'^```(.*)$')
_FENCE_CLOSE_RE = re.compile(r'^```\s*$')
_TAB_LINE_RE = re.compile(r'^(?:#@tab|%%tab)\s+(.+)$')
_INTERACT_RE = re.compile(r'tab\.interact_select\(([^)]*)\)')
_BEGIN_TAB_RE = re.compile(r':begin_tab:`([^`]+)`')
_DIRECTIVE_RE = re.compile(r'^:(\w[\w-]*):')

# Inside slide divs
_PLACEHOLDER_RE = re.compile(r'^@([a-z][a-z0-9-]*)(?:@(\w+))?\s*$')

# Markers
_SLIDE_OPEN_RE = re.compile(r'^:{3,}\s*\{\.slide(?:\s|\})')
_SUBSLIDE_OPEN_RE = re.compile(r'^:{3,}\s*\{\.subslide(?:\s|\})')
_DIV_OPEN_RE = re.compile(r'^:{3,}\s*\{')
_DIV_CLOSE_RE = re.compile(r'^:{3,}\s*$')


def lint_file(path: Path, corpus_labels=None):
    """Return list[Issue] for one file. corpus_labels is dict of
    label → first-defining-path used for cross-corpus dedup."""
    issues = []
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    chap_slug = file_slug(path)

    # ── Collect fences (for ID and tab checks) ──
    fences = collect_fences(lines, chap_slug)
    fence_ids = {}  # id → list of (line, tab)
    for f in fences:
        cid = _extract_id(f['info'])
        if cid:
            fence_ids.setdefault(cid, []).append((f['start'] + 1, f['tab']))

    # Validate fence IDs for uniqueness / framework-variant non-overlap
    for cid, occurrences in fence_ids.items():
        if len(occurrences) > 1:
            tabs_seen = set()
            collision = None
            for line_no, tab in occurrences:
                if tab is None or tab == 'all':
                    fws = {'all'}
                else:
                    fws = {t.strip() for t in tab.split(',')}
                if fws & tabs_seen:
                    collision = line_no
                    break
                tabs_seen |= fws
            if 'all' in tabs_seen and len(tabs_seen) > 1:
                # 'all' overlaps with everything
                collision = occurrences[-1][0]
            if collision:
                issues.append(Issue(path, collision, 1, 'error',
                    f'duplicate cell ID #{cid} '
                    f'(overlapping #@tab framework sets)'))

    # Validate framework names in #@tab / %%tab / tab.interact_select
    in_fence = False
    fence_info = ''
    for i, line in enumerate(lines, start=1):
        if not in_fence:
            m = _FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                fence_info = m.group(1).strip()
                in_fence = True
                continue
            for m in _INTERACT_RE.finditer(line):
                args = m.group(1)
                # Each arg is a quoted string
                for fw in re.findall(r"['\"]([^'\"]+)['\"]", args):
                    if fw not in _FW_NAMES:
                        col = line.index(fw) + 1
                        issues.append(Issue(path, i, col, 'error',
                            f'unknown framework `{fw}` in tab.interact_select'))
            for m in _BEGIN_TAB_RE.finditer(line):
                key = m.group(1)
                for fw in [t.strip() for t in key.split(',')]:
                    if fw not in _FW_NAMES:
                        col = line.index(fw) + 1
                        issues.append(Issue(path, i, col, 'error',
                            f'unknown framework `{fw}` in :begin_tab:'))
        else:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
                continue
            mt = _TAB_LINE_RE.match(line)
            if mt:
                for fw in [t.strip() for t in mt.group(1).split(',')]:
                    if fw not in _FW_NAMES:
                        col = line.index(fw) + 1
                        issues.append(Issue(path, i, col, 'error',
                            f'unknown framework `{fw}` in tab directive'))

    # ── Marker balance ──
    issues += _check_balance(path, lines)

    # ── Slide div placeholder validation ──
    issues += _check_placeholders(path, lines, fence_ids)

    # ── Unknown directives ──
    issues += _check_directives(path, lines)

    # ── Display math: `\\` needs an alignment environment ──
    issues += _check_display_math(path, lines)

    # ── List continuation lines that start with a cross-reference ──
    issues += _check_xref_line_starts(path, lines)

    # ── Tab markers sharing their line with text ──
    issues += _check_tab_marker_lines(path, lines)

    # ── A wrapped list-item line that pandoc reads as a new list item ──
    issues += _check_numbered_line_starts(path, lines)

    # ── Optional: cross-corpus label uniqueness ──
    if corpus_labels is not None:
        for i, line in enumerate(lines, start=1):
            for m in re.finditer(r':(?:label|eqlabel):`([^`]+)`', line):
                label = m.group(1)
                if label in corpus_labels and corpus_labels[label] != str(path):
                    col = line.index(label) + 1
                    issues.append(Issue(path, i, col, 'error',
                        f'label `{label}` already defined in '
                        f'{corpus_labels[label]}'))
                else:
                    corpus_labels.setdefault(label, str(path))

    return issues


def _extract_id(info):
    m = re.search(r'(?:^|[\s{])#([a-z][a-z0-9-]*)(?=\s|\}|$)', info)
    return m.group(1) if m else None


def _check_balance(path, lines):
    """Check :begin_tab: / :end_tab: pairs and `::: {.slide}` / `:::`."""
    issues = []
    in_fence = False

    # :begin_tab: / :end_tab:
    tab_stack = []  # list of (line, key)
    for i, line in enumerate(lines, start=1):
        if not in_fence:
            m = _FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                in_fence = True
                continue
            mb = _BEGIN_TAB_RE.search(line)
            if mb:
                tab_stack.append((i, mb.group(1)))
            if ':end_tab:' in line:
                if not tab_stack:
                    issues.append(Issue(path, i, 1, 'error',
                        ':end_tab: without matching :begin_tab:'))
                else:
                    tab_stack.pop()
        else:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
    for line_no, key in tab_stack:
        issues.append(Issue(path, line_no, 1, 'error',
            f':begin_tab:`{key}` without matching :end_tab:'))

    # Slide / subslide / fragment div balance
    in_fence = False
    div_stack = []  # list of (line, kind)
    for i, line in enumerate(lines, start=1):
        if not in_fence:
            m = _FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                in_fence = True
                continue
            if _SLIDE_OPEN_RE.match(line):
                div_stack.append((i, 'slide'))
            elif _SUBSLIDE_OPEN_RE.match(line):
                div_stack.append((i, 'subslide'))
            elif _DIV_OPEN_RE.match(line):
                div_stack.append((i, 'div'))
            elif _DIV_CLOSE_RE.match(line):
                if div_stack:
                    div_stack.pop()
                else:
                    issues.append(Issue(path, i, 1, 'error',
                        '::: closer without matching opener'))
        else:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
    for line_no, kind in div_stack:
        issues.append(Issue(path, line_no, 1, 'error',
            f'::: {{.{kind}}} without matching closer'))

    return issues


def _check_placeholders(path, lines, fence_ids):
    """Verify @<id> and @<id>@<fw> placeholders inside .slide divs."""
    issues = []
    in_slide_depth = 0
    for i, line in enumerate(lines, start=1):
        if _SLIDE_OPEN_RE.match(line) or _SUBSLIDE_OPEN_RE.match(line):
            in_slide_depth += 1
            continue
        if _DIV_CLOSE_RE.match(line) and in_slide_depth:
            in_slide_depth -= 1
            continue
        if not in_slide_depth:
            continue
        m = _PLACEHOLDER_RE.match(line.strip())
        if not m:
            continue
        cid, forced_fw = m.group(1), m.group(2)
        if forced_fw and forced_fw not in FRAMEWORKS:
            col = line.index(forced_fw) + 1
            issues.append(Issue(path, i, col, 'error',
                f'unknown framework `{forced_fw}` in placeholder @{cid}@{forced_fw}'))
            continue
        if cid not in fence_ids:
            issues.append(Issue(path, i, 1, 'warning',
                f'placeholder @{cid} references unknown cell ID'))
            continue
        if forced_fw:
            # Verify a variant tagged for forced_fw exists
            ok = False
            for _, tab in fence_ids[cid]:
                if tab is None or tab == 'all':
                    continue
                if forced_fw in [t.strip() for t in tab.split(',')]:
                    ok = True
                    break
            if not ok:
                issues.append(Issue(path, i, 1, 'warning',
                    f'placeholder @{cid}@{forced_fw} has no matching variant'))
    return issues


_MATH_ENVS = ('aligned', 'align', 'alignat', 'split', 'array', 'matrix',
              'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix',
              'smallmatrix', 'psmallmatrix', 'cases', 'gathered', 'gather',
              'multline', 'substack', 'subarray')


def _check_display_math(path, lines):
    """A `\\\\` row break inside `$$...$$` needs an alignment environment.

    Without one, pandoc emits `\\[a\\\\b\\]` and TeX sets every row on a single
    line, which silently runs off the right margin in the PDF.

    A multi-line display followed by `:eqlabel:` must have `$$` alone on its
    opening and closing lines; otherwise the preprocessor does not attach the
    label, the equation gets an auto label, and every `:eqref:` to it is
    unresolved.
    """
    def _label_check(start, end):
        if end < len(lines) and lines[end].strip().startswith(':eqlabel:'):
            return [Issue(path, start, 1, 'error',
                'multi-line display math with :eqlabel: must have `$$` '
                'alone on its opening and closing lines, or the label '
                'does not attach')]
        return []

    def _flag(start, text):
        if '\\\\' in text and not any(
                '\\begin{%s}' % e in text for e in _MATH_ENVS):
            return [Issue(path, start, 1, 'error',
                'display math uses `\\\\` without an alignment '
                'environment; wrap the body in \\begin{aligned}...'
                '\\end{aligned} or TeX sets it on one long line')]
        return []

    issues = []
    in_fence = False
    in_math = False
    start = 0
    body = []
    for i, line in enumerate(lines, start=1):
        if _FENCE_OPEN_RE.match(line) and not in_fence and not line.startswith('````'):
            in_fence = True
            continue
        if in_fence and _FENCE_CLOSE_RE.match(line):
            in_fence = False
            continue
        if in_fence:
            continue
        stripped = line.strip()
        # Single-line display: `$$ ... $$` opened and closed on one line.
        if not in_math and stripped.startswith('$$') and stripped.endswith('$$') \
                and len(stripped) > 4:
            issues += _flag(i, stripped[2:-2])
            continue
        if stripped == '$$':
            if not in_math:
                in_math, start, body, shared = True, i, [], False
            else:
                in_math = False
                issues += _flag(start, '\n'.join(body))
                if shared:
                    issues += _label_check(start, i)
            continue
        # Fenced-off forms where content shares the `$$` line.
        if not in_math and stripped.startswith('$$'):
            in_math, start, body, shared = True, i, [stripped[2:]], True
            continue
        if in_math and stripped.endswith('$$'):
            in_math = False
            body.append(stripped[:-2])
            issues += _flag(start, '\n'.join(body))
            issues += _label_check(start, i)
            continue
        if in_math:
            body.append(line)
    return issues


_LIST_ITEM_RE = re.compile(r'^\s*(?:\d+[.)]|[-*+])\s')
_XREF_ROLE = r':(?:numref|eqref|ref|exref):`{1,2}[^`\s]+`{1,2}'
_XREF_LINE_START_RE = re.compile(
    r'^\s*(?:\(' + _XREF_ROLE + r'\)|' + _XREF_ROLE + r'[.)])(?=\s|$)')


def _check_xref_line_starts(path, lines):
    """Warn on a wrapped list-item line that begins with a cross-reference.

    The preprocessor turns `:numref:`x`` into `@sec-x` (and `:exref:` into
    a number in notebooks and the PDF), so a continuation line that starts
    `:numref:`x`.`, `:numref:`x`)` or `(:numref:`x`)`, followed by a space
    or the end of the line, begins `@sec-x. ` and the like. Pandoc takes
    that for a list marker and ends the paragraph:
    the reference becomes a stray paragraph inside the item, or, under a
    two-digit marker with a shallow indent, ends the list. Rewrap the line
    so the reference does not start it. Slides (after `<!-- slides -->`)
    keep the role text, so they are not checked.
    """
    issues = []
    in_fence = False
    in_list = False
    prev_blank = True
    for i, line in enumerate(lines, start=1):
        if line.startswith('<!-- slides -->'):
            break
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            prev_blank = True
            continue
        if _LIST_ITEM_RE.match(line):
            in_list = True
        elif not line[0].isspace() and (prev_blank or line.startswith('#')):
            in_list = False  # an unindented block after a blank line
        elif in_list and _XREF_LINE_START_RE.match(line):
            issues.append(Issue(path, i, len(line) - len(line.lstrip()) + 1,
                'warning',
                'list-item line starts with a cross-reference followed by '
                '`.` or `)`, which pandoc reads as a new block; rewrap so '
                'the reference does not begin the line'))
        prev_blank = False
    return issues


_TAB_MARKER_RE = re.compile(r':begin_tab:`[^`]*`|:end_tab:')


def _check_tab_marker_lines(path, lines):
    """`:begin_tab:` and `:end_tab:` must each stand alone on their line.

    The prose-tab converters match `:begin_tab:`fw`` followed directly by a
    newline and `:end_tab:` at the start of a line. Text after
    `:begin_tab:`fw`` or before `:end_tab:` defeats the match, and the
    marker renders literally (or swallows the text) in HTML, notebooks and
    the PDF. Markers quoted inside inline code are ignored.
    """
    issues = []
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if not in_fence:
            if _FENCE_OPEN_RE.match(line) and not line.startswith('````'):
                in_fence = True
                continue
        else:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
            continue
        bare = re.sub(r'``[^`]*``', '', line)
        for m in _TAB_MARKER_RE.finditer(bare):
            before = bare[:m.start()]
            # A marker inside inline code (an odd number of backticks
            # before it) is quoted syntax, not a marker.
            if before.count('`') % 2:
                continue
            if before.strip() or bare[m.end():].strip():
                issues.append(Issue(path, i, m.start() + 1, 'error',
                    f'`{m.group(0)}` must stand alone on its line; text '
                    f'beside it makes the marker render literally'))
    return issues


_LIST_MARKER_RE = re.compile(r'^(\s*)(?:(\d+)[.)]|[-*+])(?:\s|$)')


def _check_numbered_line_starts(path, lines):
    """Warn on a wrapped list-item line that pandoc reads as a new list item.

    Inside a list item, a line that begins `<n>. ` or `<n>) ` starts a list
    item even without a blank line before it: indented deeper, it opens a
    nested ordered list numbered from n; at the item's own indent it becomes
    the next sibling item and the number is lost. An exercise wrapped as
    "... each of width" / "   256. Before running anything ..." rendered as
    an ordered list starting at 256 in the middle of the exercise. A line
    starting `1. ` is left alone (almost always a list the author meant),
    as is a sibling item numbered one past the previous item. Outside lists
    pandoc does not interrupt a paragraph this way, so plain paragraphs are
    not checked. Rewrap so the number does not begin the line.
    """
    issues = []
    in_fence = False
    in_math = False
    items = {}          # marker indent → last number (int) or 'ul'
    prev_text = False   # previous line is non-blank text (no blank between)
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            prev_text = False
            continue
        if in_fence:
            continue
        if stripped == '$$' or (stripped.startswith('$$')
                                and not stripped.endswith('$$')):
            in_math = not in_math
            continue
        if in_math:
            if stripped.endswith('$$'):
                in_math = False
            continue
        if not stripped:
            prev_text = False
            continue
        m = _LIST_MARKER_RE.match(line)
        if m:
            indent, num = len(m.group(1)), m.group(2)
            if prev_text and items and num and num != '1':
                if indent in items:
                    last = items[indent]
                    bad = last == 'ul' or int(num) != last + 1
                else:
                    bad = True
                if bad:
                    issues.append(Issue(path, i, indent + 1, 'warning',
                        f'list-item line starts with `{num}{line[indent + len(num)]}`'
                        f'; pandoc reads it as a new list item numbered '
                        f'{num}. Rewrap so the number does not begin the line'))
            items = {k: v for k, v in items.items() if k <= indent}
            items[indent] = int(num) if num else 'ul'
        elif not line[0].isspace() and (not prev_text or stripped.startswith('#')):
            items = {}   # an unindented block after a blank line ends the list
        prev_text = True
    return issues


def _check_directives(path, lines):
    """Warn on unknown :directive: constructs."""
    issues = []
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if _FENCE_OPEN_RE.match(line) and not in_fence and not line.startswith('````'):
            in_fence = True
            continue
        if in_fence and _FENCE_CLOSE_RE.match(line):
            in_fence = False
            continue
        if in_fence:
            continue
        m = _DIRECTIVE_RE.match(line)
        if m:
            name = m.group(1)
            if name not in _KNOWN_DIRECTIVES:
                issues.append(Issue(path, i, 1, 'warning',
                    f'unknown directive :{name}:'))
    return issues


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Lint d2l source .md files (GCC-style output)')
    parser.add_argument('files', nargs='*', type=Path)
    parser.add_argument('--corpus', action='store_true',
                        help='Walk all files; check cross-file label uniqueness')
    parser.add_argument('--source', type=Path, default=Path('.'))
    args = parser.parse_args()

    if args.files:
        files = list(args.files)
    else:
        files = [args.source / rel for rel in CHAPTER_NUMBERING.keys()]
        files = [f for f in files if f.exists()]

    corpus_labels = {} if args.corpus else None

    error_count = 0
    warning_count = 0
    for f in files:
        for issue in lint_file(f, corpus_labels):
            print(issue.emit())
            if issue.severity == 'error':
                error_count += 1
            else:
                warning_count += 1

    if error_count or warning_count:
        sev_summary = []
        if error_count:
            sev_summary.append(f'{error_count} error(s)')
        if warning_count:
            sev_summary.append(f'{warning_count} warning(s)')
        print(f'\n{", ".join(sev_summary)} across {len(files)} file(s).',
              file=sys.stderr)
    sys.exit(1 if error_count else 0)


if __name__ == '__main__':
    main()
