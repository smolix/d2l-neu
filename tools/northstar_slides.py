#!/usr/bin/env python3
"""Single source of truth for which slide decks are "north-star".

The slide migration is gradual: we rewrite one source file's
`<!-- slides -->` block at a time into the north-star vocabulary
(cover/divider/kicker cards, `@fig:` diagrams, `.cols` layout, In/Out
cards). Decks whose source has been upgraded should be presented as
north-star; the rest stay as their legacy decks until upgraded.

Rather than maintain a hand-edited allowlist, a deck is considered
north-star **iff its source slides block uses north-star markers**. So a
deck becomes eligible automatically the moment its block is rewritten —
"substitute the north-star deck whenever one is available."

Markers (any one is sufficient):
  ::: {.cover}        title-slide treatment
  ::: {.divider}      section divider
  [text]{.kicker}     eyebrow section label
  @fig:<id>           inlined diagram
  ::: {.d2l-note}     callout card
  ::: {.cols}         two-column body

Usage:
    python tools/northstar_slides.py <source_dir> [--list | --count]
    # default prints "<n> / <total> decks migrated" then the list
"""

import argparse
import re
from pathlib import Path

# Any one of these inside the slides block flags a deck as north-star.
_NS_MARKERS = re.compile(
    r'\{\.cover\}|\{\.divider\}|\]\{\.kicker\}|@fig:|\{\.d2l-note|\{\.cols')

_SLIDES_MARKER = '<!-- slides -->'


def slides_block(text: str) -> str:
    """Return the `<!-- slides -->` block (to end of file), or ''."""
    i = text.find(_SLIDES_MARKER)
    return text[i:] if i != -1 else ''


def is_northstar(md_path) -> bool:
    """True if the source file's slides block uses north-star vocabulary."""
    p = Path(md_path)
    if not p.exists():
        return False
    block = slides_block(p.read_text(encoding='utf-8'))
    return bool(block) and bool(_NS_MARKERS.search(block))


def has_slides(md_path) -> bool:
    p = Path(md_path)
    return p.exists() and _SLIDES_MARKER in p.read_text(encoding='utf-8')


def northstar_set(source_dir) -> set:
    """Set of 'chapter_dir/stem' for every north-star source deck."""
    src = Path(source_dir)
    out = set()
    for md in src.glob('chapter_*/*.md'):
        if md.stem == 'index':
            continue
        if is_northstar(md):
            out.add(f'{md.parent.name}/{md.stem}')
    return out


def all_slide_decks(source_dir) -> set:
    src = Path(source_dir)
    return {f'{md.parent.name}/{md.stem}'
            for md in src.glob('chapter_*/*.md')
            if md.stem != 'index' and has_slides(md)}


def main():
    ap = argparse.ArgumentParser(description='List north-star slide decks')
    ap.add_argument('source', nargs='?', default='.',
                    help='source dir containing chapter_*/ (default: .)')
    ap.add_argument('--list', action='store_true',
                    help='print only the north-star deck paths, one per line')
    ap.add_argument('--count', action='store_true',
                    help='print only "<migrated>/<total>"')
    args = ap.parse_args()

    ns = sorted(northstar_set(args.source))
    total = len(all_slide_decks(args.source))

    if args.list:
        print('\n'.join(ns))
        return
    if args.count:
        print(f'{len(ns)}/{total}')
        return

    print(f'north-star slides: {len(ns)} / {total} decks migrated\n')
    for d in ns:
        print(f'  ✦ {d}')


if __name__ == '__main__':
    main()
