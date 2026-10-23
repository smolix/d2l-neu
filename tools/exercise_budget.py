#!/usr/bin/env python3
"""Report the size of each section's exercises relative to its body.

The writing guide (docs/writing-guide.md, §12.2) uses about 30% of a section's
body prose as a guide for the size of its `## Exercises`. It shows balance; it
is not a cap. Good problems stay even when a section exceeds it; filler goes
even when a section is under it.

Words are counted outside code fences. The body is everything before
`## Exercises`; the exercise block ends at the slides marker or the
`[Discussions]` link or a `## Discussions` heading.

    python tools/exercise_budget.py                      # every chapter in _quarto.yml
    python tools/exercise_budget.py chapter_attention    # one chapter directory
    python tools/exercise_budget.py --over               # only sections over the guide
"""
import argparse
import re
import sys
from pathlib import Path

TARGET = 0.30
FENCE = re.compile(r'^```.*?^```', re.S | re.M)
PROBLEM = re.compile(r'^\d+\. ', re.M)


def split(text):
    """(body, exercises) prose of one section, code fences removed."""
    text = text.split('<!-- slides -->')[0]
    parts = re.split(r'^## Exercises\b.*$', text, maxsplit=1, flags=re.M)
    if len(parts) < 2:
        return FENCE.sub('', text), ''
    ex = re.split(r'^(?:\[Discussions\]|## Discussions?\b)', parts[1], maxsplit=1, flags=re.M)[0]
    return FENCE.sub('', parts[0]), FENCE.sub('', ex)


def sections(root, dirs):
    order = re.findall(r'(chapter_[\w-]+/[\w-]+)\.qmd', (root / '_quarto.yml').read_text())
    for rel in order:
        path = root / f'{rel}.md'
        if path.exists() and (not dirs or rel.split('/')[0] in dirs):
            yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('chapters', nargs='*', help='chapter directories (default: all)')
    ap.add_argument('--over', action='store_true', help='list only sections over the guide')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    dirs = {d.rstrip('/') for d in args.chapters}
    print(f'{"section":58s} {"body":>6s} {"exer":>6s} {"ratio":>6s} {"probs":>5s}')
    for path in sections(root, dirs):
        body, ex = split(path.read_text())
        b, e = len(body.split()), len(ex.split())
        if not e:
            continue
        ratio = e / max(b, 1)
        if args.over and ratio <= TARGET:
            continue
        flag = '  over' if ratio > TARGET else ''
        print(f'{str(path.relative_to(root)):58s} {b:6d} {e:6d} {ratio:6.0%} '
              f'{len(PROBLEM.findall(ex)):5d}{flag}')


if __name__ == '__main__':
    sys.exit(main())
