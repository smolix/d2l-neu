"""Build the CJK fallback font of the PDF book: a subset of Noto Sans SC.

The PDF fonts (Source Serif 4, Source Sans 3, Inconsolata) have no Chinese
glyphs, and the Language Models part prints Chinese in code cells and cell
outputs. static/d2l-preamble.tex switches to this font for CJK characters
only. The subset keeps

* the 6,763 hanzi of GB 2312 (simplified Chinese as commonly written),
* CJK symbols and punctuation (U+3000-U+303F) and fullwidth ASCII forms
  (U+FF01-U+FF5E),
* the space (fontspec reads a font's interword space from it),
* every other CJK character that occurs in a chapter source or in the
  committed outputs/ store, so a new output that uses a rarer character
  only needs a re-run of this script.

Source: NotoSansSC-Regular.otf of github.com/notofonts/noto-cjk at commit
NOTO_COMMIT (SIL Open Font License 1.1; the license travels with the font as
static/fonts/NotoSansSC-LICENSE.txt). The output is byte-identical for the
same input font and character set.

    .venv-lm/bin/python tools/make_cjk_font_subset.py path/to/NotoSansSC-Regular.otf
"""
import pathlib
import re
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'static/fonts/NotoSansSC-Regular-subset.otf'
NOTO_COMMIT = 'f8d157532fbfaeda587e826d4cd5b21a49186f7c'
# Code-point ranges that static/d2l-preamble.tex sends to this font; keep the
# two lists in step.
CJK_RANGES = [(0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
              (0xF900, 0xFAFF), (0xFF00, 0xFFEF)]


def is_cjk(cp):
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def gb2312_hanzi():
    """The 6,763 hanzi of GB 2312: rows 16-87, cells 1-94."""
    chars = set()
    for row in range(16, 88):
        for cell in range(1, 95):
            try:
                chars.add(bytes([0xA0 + row, 0xA0 + cell]).decode('gb2312'))
            except UnicodeDecodeError:
                pass
    return chars


def corpus_chars():
    """CJK characters in the chapter sources and the outputs store."""
    files = list(ROOT.glob('chapter_*/*.md')) + list(
        (ROOT / 'outputs').rglob('*.json'))
    chars = set()
    for f in files:
        text = f.read_text(encoding='utf-8', errors='ignore')
        if f.suffix == '.json':  # outputs escape non-ASCII as \uXXXX
            text += ''.join(chr(int(h, 16)) for h in
                            re.findall(r'\\u([0-9a-fA-F]{4})', text))
        chars.update(c for c in text if is_cjk(ord(c)))
    return chars


def main(src):
    font = TTFont(src, recalcTimestamp=False)
    cmap = font.getBestCmap()
    wanted = {ord(c) for c in gb2312_hanzi() | corpus_chars()}
    wanted |= {cp for lo, hi in [(0x3000, 0x303F), (0xFF01, 0xFF5E)]
               for cp in range(lo, hi + 1)}
    wanted |= {0x20, 0xA0}  # fontspec reads the interword space of a font
    missing = sorted(cp for cp in wanted if cp not in cmap)
    if missing:
        print(f'{len(missing)} characters not in the font, e.g.',
              ' '.join(chr(cp) for cp in missing[:10]))
    options = subset.Options()
    options.layout_features = ['*']
    options.name_IDs = ['*']
    options.name_languages = ['*']
    options.notdef_outline = True
    options.recalc_timestamp = False
    sub = subset.Subsetter(options)
    sub.populate(unicodes=[cp for cp in wanted if cp in cmap])
    sub.subset(font)
    font.save(OUT)
    print(f'{OUT.relative_to(ROOT)}: {len(wanted) - len(missing)} '
          f'characters, {OUT.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main(sys.argv[1])
