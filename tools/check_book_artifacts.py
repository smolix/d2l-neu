#!/usr/bin/env python3
"""Post-render integrity gate for `_book/` — the canonical "is this book
actually shippable?" check, run by `make check-all-artifacts`.

The old check only asserted that two index HTML files existed, so a badly
broken build still "passed": the V8-heap OOM that silently dropped ~40% of
slide decks, and unmaterialized Git-LFS pointer images that shipped as broken
figures, both sailed through. This verifies the things that actually broke:

  1. core pages exist            — _book/index.html + _book/slides/index.html
  2. no broken images            — zero unmaterialized Git-LFS pointer assets
  3. PDFs built                  — one per framework, each a plausible size
  4. slide-deck coverage         — every framework rendered every deck the
                                   source defines for it (catches mass drops)

Exit non-zero (with an actionable message) on any failure.

Usage:  python tools/check_book_artifacts.py [--book _book] [--store outputs]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING
from northstar_slides import has_slides

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']
_IMG_EXTS = ('.svg', '.png', '.jpg', '.jpeg', '.gif')
_LFS_SIG = b'version https://git-lfs.github.com/spec/v1'
MIN_PDF_BYTES = 1_000_000   # a real per-framework book PDF is tens of MB
MIN_ZIP_BYTES = 500_000     # a real per-framework notebook zip is several MB


def pointer_images(root: Path):
    out = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in _IMG_EXTS or not p.is_file() or p.is_symlink():
            continue
        try:
            if p.stat().st_size < 300 and _LFS_SIG in p.read_bytes()[:120]:
                out.append(p)
        except OSError:
            continue
    return out


def expected_decks(source: Path, store: Path):
    """For each framework, the set of `chap/stem` decks the source defines for
    it: the source file has a slides block AND that framework has a committed
    output manifest (the same availability rule build_slides_index uses)."""
    exp = {fw: set() for fw in FRAMEWORKS}
    for rel in CHAPTER_NUMBERING:
        relp = Path(rel)
        if relp.stem == 'index':
            continue
        if not has_slides(source / rel):
            continue
        for fw in FRAMEWORKS:
            if (store / fw / relp.parent.name / f'{relp.stem}.json').exists():
                exp[fw].add(f'{relp.parent.name}/{relp.stem}')
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--book', default='_book')
    ap.add_argument('--store', default='outputs')
    ap.add_argument('--source', default='.')
    args = ap.parse_args()
    book, store, source = Path(args.book), Path(args.store), Path(args.source)
    errors = []

    # 1. core pages
    for rel in ('index.html', 'slides/index.html'):
        if not (book / rel).is_file():
            errors.append(f'missing {book / rel}')

    # 2. no pointer images
    ptrs = pointer_images(book)
    if ptrs:
        errors.append(f'{len(ptrs)} unmaterialized Git-LFS pointer image(s) in '
                      f'{book}/ (broken figures) — `git lfs pull` then re-render. '
                      f'e.g. {ptrs[0].relative_to(book)}')

    # 3. PDFs (only assert if the PDF tree was built at all)
    if (book / 'pdf').is_dir():
        for fw in FRAMEWORKS:
            pdf = book / 'pdf' / f'Dive-into-Deep-Learning-{fw}.pdf'
            if not pdf.is_file():
                errors.append(f'missing PDF {pdf}')
            elif pdf.stat().st_size < MIN_PDF_BYTES:
                errors.append(f'PDF {pdf} is only {pdf.stat().st_size} bytes '
                              f'(< {MIN_PDF_BYTES}) — render likely truncated')
    else:
        print(f'  note: {book}/pdf absent — skipping PDF check (html-only build)')

    # 4. slide-deck coverage per framework
    if (book / 'slides').is_dir() and store.is_dir():
        exp = expected_decks(source, store)
        for fw in FRAMEWORKS:
            want = exp[fw]
            got = {f'{p.parent.name}/{p.stem}'
                   for p in (book / 'slides' / fw).rglob('*.html')}
            miss = want - got
            if miss:
                ex = ', '.join(sorted(miss)[:5])
                errors.append(f'{fw}: {len(miss)} slide deck(s) defined in source '
                              f'but NOT rendered (e.g. {ex})')
            else:
                print(f'  ✓ {fw}: {len(want)} slide decks rendered')

    # 5. Notebook download zips (only assert if the tree was built at all)
    if (book / 'notebooks').is_dir():
        import zipfile
        for fw in FRAMEWORKS:
            z = book / 'notebooks' / f'd2l-{fw}.zip'
            if not z.is_file():
                errors.append(f'missing notebook zip {z}')
            elif z.stat().st_size < MIN_ZIP_BYTES:
                errors.append(f'notebook zip {z} is only {z.stat().st_size} bytes '
                              f'(< {MIN_ZIP_BYTES}) — likely empty/truncated')
            elif not zipfile.is_zipfile(z):
                errors.append(f'notebook zip {z} is not a valid zip archive')
            else:
                n = sum(1 for e in zipfile.ZipFile(z).namelist()
                        if e.endswith('.ipynb'))
                if n == 0:
                    errors.append(f'notebook zip {z} contains no .ipynb files')
                else:
                    print(f'  ✓ {fw}: notebook zip ({n} notebooks)')
    else:
        print(f'  note: {book}/notebooks absent — skipping notebook-zip check')

    if errors:
        print('\nFAIL (check-all-artifacts):', file=sys.stderr)
        for e in errors:
            print(f'  ✗ {e}', file=sys.stderr)
        return 1
    print('Verified full build artifacts: pages, images, PDFs, and slide coverage.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
