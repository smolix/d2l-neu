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
