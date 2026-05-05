#!/usr/bin/env python3
"""Remove `#@tab all` and `%%tab all` directives from chapter_*/*.md.

Untagged Python cells default to "all frameworks", so the explicit
`all` tag carries no information. Strips 338 occurrences in the
current corpus. Idempotent.

Only the *first non-blank line* of a Python code fence body is
considered, matching `extract_tab()` semantics in d2l_preprocess.py.
Other lines that happen to match the pattern are left alone.

Usage:
    python tools/strip_tab_all.py [--check] [--verbose] [files...]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING, is_python_block


FENCE_OPEN_RE = re.compile(r'^```(.*)$')
FENCE_CLOSE_RE = re.compile(r'^```\s*$')
TAB_ALL_RE = re.compile(r'^(#@tab|%%tab)\s+all\s*$')


def process_file(path: Path, dry_run=False, verbose=False) -> int:
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    out = []
    in_python_fence = False
    seen_nonblank = False  # within current Python fence body
    removed = 0

    for i, line in enumerate(lines):
        if not in_python_fence:
            m = FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                info = m.group(1).strip()
                if is_python_block(info):
                    in_python_fence = True
                    seen_nonblank = False
            out.append(line)
        else:
            if FENCE_CLOSE_RE.match(line):
                in_python_fence = False
                out.append(line)
                continue
            stripped = line.strip()
            if not seen_nonblank and stripped:
                seen_nonblank = True
                if TAB_ALL_RE.match(stripped):
                    if verbose:
                        print(f'  {path}:{i+1}: drop `{line}`')
                    removed += 1
                    # Drop this line entirely. Reset seen_nonblank so the
                    # NEXT non-blank line isn't treated as the directive
                    # candidate (extract_tab only checks the first one).
                    continue
            out.append(line)

    if removed and not dry_run:
        path.write_text('\n'.join(out), encoding='utf-8')

    return removed


def main():
    parser = argparse.ArgumentParser(
        description='Strip `#@tab all` / `%%tab all` from .md sources')
    parser.add_argument('files', nargs='*', type=Path)
    parser.add_argument('--check', action='store_true',
                        help='Report what would change but do not write')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--source', type=Path, default=Path('.'))
    args = parser.parse_args()

    if args.files:
        files = list(args.files)
    else:
        files = [args.source / rel for rel in CHAPTER_NUMBERING.keys()]
        files = [f for f in files if f.exists()]

    total = 0
    files_changed = 0
    for f in files:
        n = process_file(f, dry_run=args.check, verbose=args.verbose)
        if n:
            files_changed += 1
            total += n

    action = 'would strip' if args.check else 'stripped'
    print(f'\nDone. {action} {total} `tab all` line(s) across '
          f'{files_changed}/{len(files)} files.')


if __name__ == '__main__':
    main()
