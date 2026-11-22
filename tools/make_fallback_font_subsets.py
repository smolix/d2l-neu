"""Build the non-CJK fallback fonts of the PDF book.

The PDF fonts (Source Serif 4, Source Sans 3, Inconsolata) cover Latin and,
except Inconsolata, Greek; they have no Devanagari, no mathematical
alphanumerics (U+1D400-1D7FF) and patchy coverage of symbols: Inconsolata
lacks Greek and the `ﬁ` ligature, the serif and sans lack block elements
such as `▁` (the SentencePiece word marker) and `●`. XeTeX drops a missing
glyph with only a `Missing character` line in the log, so
static/d2l-preamble.tex sends these characters to the fonts built here:

* NotoSansDevanagari-Regular-subset.ttf: the whole Devanagari block
  (U+0900-097F), Devanagari Extended, ZWNJ/ZWJ and the dotted circle that
  the shaper inserts for a stray mark. Devanagari runs switch to this font
  as a whole, so conjuncts and vowel signs shape correctly.
* DejaVuSans-subset.ttf, DejaVuSansMono-subset.ttf, NotoSansMath-Regular-
  subset.ttf: every glyph each font has in FALLBACK_RANGES. The preamble
  sets a character of these ranges in the current font when that font has
  it, and otherwise in the first of these fonts that does (DejaVu Sans Mono
  first inside code).

The subsets keep each font's own name and licence: SIL Open Font License 1.1
for the Noto fonts (static/fonts/Noto-OFL-LICENSE.txt), the Bitstream Vera /
DejaVu licence for DejaVu (static/fonts/DejaVu-LICENSE.txt). Sources: the
Noto 2.001 and DejaVu 2.37 releases, as shipped by Ubuntu 24.04's
fonts-noto-core and fonts-dejavu-core packages (the default paths below).
The output is byte-identical for the same input fonts.

    .venv-build/bin/python tools/make_fallback_font_subsets.py [FONTDIR]

FONTDIR, if given, holds NotoSansDevanagari-Regular.ttf,
NotoSansMath-Regular.ttf, DejaVuSans.ttf and DejaVuSansMono.ttf.
"""
import pathlib
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'static/fonts'

# Devanagari code points; static/d2l-preamble.tex gives the same ranges the
# Devanagari class (\bookdevaclass). Keep the two lists in step.
DEVA_RANGES = [(0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x200C, 0x200D)]
DEVA_EXTRA = [0x25CC, 0x0020, 0x00A0, 0x002D]
# Code points that get the per-character fallback test (\bookfbclass in the
# preamble). Keep the two lists in step.
FALLBACK_RANGES = [
    (0x0370, 0x03FF),    # Greek and Coptic (Inconsolata has none)
    (0x0400, 0x052F),    # Cyrillic and Cyrillic Supplement
    (0x1D00, 0x1DBF),    # phonetic extensions: modifier letters such as ᵀ
    (0x2016, 0x2016),    # double vertical line ‖
    (0x202F, 0x202F),    # narrow no-break space
    (0x2070, 0x209F),    # superscripts and subscripts
    (0x2100, 0x218F),    # letterlike symbols, number forms
    (0x2190, 0x23FF),    # arrows, mathematical operators, misc. technical
    (0x2400, 0x24FF),    # control pictures, enclosed alphanumerics
    (0x2500, 0x27BF),    # box drawing, blocks, geometric shapes, symbols
    (0x27C0, 0x2BFF),    # supplemental arrows and mathematical operators
    (0xFB00, 0xFB06),    # Latin ligatures (ﬁ, ﬂ)
    (0xFEFF, 0xFEFF),    # zero-width no-break space (byte-order mark)
    (0xFFF9, 0xFFFD),    # specials, including the replacement character
    (0x1D400, 0x1D7FF),  # mathematical alphanumeric symbols (𝑥, 𝐚, 𝔼)
]

FONTS = {
    # output name: (source file, ranges, extra code points)
    'NotoSansDevanagari-Regular-subset.ttf':
        ('NotoSansDevanagari-Regular.ttf', DEVA_RANGES, DEVA_EXTRA),
    'DejaVuSans-subset.ttf': ('DejaVuSans.ttf', FALLBACK_RANGES, [0x20]),
    'DejaVuSansMono-subset.ttf':
        ('DejaVuSansMono.ttf', FALLBACK_RANGES, [0x20]),
    'NotoSansMath-Regular-subset.ttf':
        ('NotoSansMath-Regular.ttf', FALLBACK_RANGES, [0x20]),
}
DEFAULT_DIRS = ['/usr/share/fonts/truetype/noto',
                '/usr/share/fonts/truetype/dejavu']


def find(name, fontdir):
    dirs = [fontdir] if fontdir else DEFAULT_DIRS
    for d in dirs:
        p = pathlib.Path(d) / name
        if p.exists():
            return p
    sys.exit(f'{name} not found in {", ".join(dirs)}')


def build(src, out, ranges, extra):
    font = TTFont(src, recalcTimestamp=False)
    cmap = font.getBestCmap()
    wanted = {cp for lo, hi in ranges for cp in range(lo, hi + 1)}
    wanted |= set(extra)  # fontspec reads the interword space of a font
    keep = sorted(cp for cp in wanted if cp in cmap)
    options = subset.Options()
    options.layout_features = ['*']  # Devanagari needs its shaping tables
    options.name_IDs = ['*']
    options.name_languages = ['*']
    options.notdef_outline = True
    options.recalc_timestamp = False
    sub = subset.Subsetter(options)
    sub.populate(unicodes=keep)
    sub.subset(font)
    font.save(out)
    print(f'{out.relative_to(ROOT)}: {len(keep)} characters, '
          f'{out.stat().st_size / 1e3:.0f} kB')


def main(fontdir=None):
    for out, (src, ranges, extra) in FONTS.items():
        build(find(src, fontdir), OUTDIR / out, ranges, extra)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
