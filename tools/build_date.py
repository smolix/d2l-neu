#!/usr/bin/env python3
"""The book's version label: the date it was built, as YYYY.MM.DD.

The site and the PDFs name their version by build date instead of an edition.
The date is taken in the authors' time zone (America/Los_Angeles); set
D2L_BUILD_DATE=YYYY.MM.DD to pin it (e.g. to rebuild an earlier release).

    python3 tools/build_date.py                      # print the date
    python3 tools/build_date.py --stamp _book/index.html

`--stamp` fills every `<span class="d2l-build-date">…</span>` in the given
HTML files with the date. The html recipe runs it on the rendered landing
page right after `quarto render`, so the label is the date of that render,
not of the last time index.md changed. The PDFs get the date from
tools/fix_latex.py and tools/build_part_pdfs.py, on the page after the cover.
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SPAN_RE = re.compile(r'(<span class="d2l-build-date">)[^<]*(</span>)')


def build_date() -> str:
    pinned = os.environ.get('D2L_BUILD_DATE', '').strip()
    if pinned:
        if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2}', pinned):
            raise SystemExit(f'D2L_BUILD_DATE must be YYYY.MM.DD, not {pinned!r}')
        return pinned
    return datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y.%m.%d')


def version_page_tex() -> str:
    """LaTeX for the back of a cover: blank, with the version at the foot."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import i18n
    version = i18n.t('pdf.version',
                     book='\\textit{' + i18n.t('book.title') + '}',
                     date=build_date())
    return ('\\null\\thispagestyle{empty}\\vfill\n'
            '\\noindent{\\small' + version + '}\\newpage')


def stamp(paths) -> int:
    date, n = build_date(), 0
    for path in paths:
        path = Path(path)
        html = path.read_text(encoding='utf-8')
        new, k = SPAN_RE.subn(rf'\g<1>{date}\g<2>', html)
        if k:
            path.write_text(new, encoding='utf-8')
            n += k
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stamp', nargs='+', metavar='HTML',
                        help='fill the d2l-build-date spans in these files')
    args = parser.parse_args()
    if args.stamp:
        n = stamp(args.stamp)
        print(f'build date {build_date()}: stamped {n} span(s)')
    else:
        print(build_date())


if __name__ == '__main__':
    main()
