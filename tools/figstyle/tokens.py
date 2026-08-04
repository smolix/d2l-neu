"""Design tokens for d2l figures — the single source of truth.

Every generated figure (matplotlib or the pure-SVG composer) draws its colors,
type sizes, stroke widths, radii, and spacing from here.  The values implement
the house style documented in docs/figure-style-guide.md; change them here and
re-run ``gmake figures`` to restyle the corpus.

Provenance.  This layer *unifies* the repo's previously divergent palettes
around the maintainer-approved "gallery style" (docs/convnet-rewrite/
figure-style.md, tools/arch_diagrams.py): its accent blue (#0B6BB2), tint
(#CDE8FA), amber (#B45309/#FBE8D3), container grays, and Source Sans 3 +
Inconsolata typography are kept verbatim and extended into a full semantic
system.  Neutrals are aligned with the site's own SCSS tokens ($ink =
#15181C, $ink-3 = #6A717B).  The design principles follow the exemplars
analyzed in the style guide (Distill, Anthropic research figures, Raschka's
architecture gallery): neutral structure + semantic accents, tint/base/dark
color trios, flat fills, solid-vs-dashed as meaning, light annotation
leaders, emphasis by stroke weight before hue.

Contrast is enforced by tools/figstyle/check_contrast.py — run it after any
palette edit.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Neutrals — aligned with the site SCSS (_d2l-theme.scss) so figures read as  #
# part of the page, not pasted onto it.                                       #
# --------------------------------------------------------------------------- #

INK = "#15181C"        # primary in-figure text + structural strokes ($ink)
MUTED = "#565D66"      # secondary labels, units, de-emphasis (darkened from
                       # $ink-3 #6A717B for legibility at mobile scale)
FAINT = "#8F98A1"      # construction lines, grids, ghosted structure
HAIRLINE = "#D9DEE4"   # separators, neutral panel borders
PANEL = "#F0F2F5"      # generic grouping-panel wash (non-architecture)
PAPER = "#FFFFFF"      # figure canvas — identical to the page background

# Architecture-diagram structural fills (approved gallery style — verbatim).
NOVELTY_FILL = "#3B3B3B"    # the one "new op" box: near-black, white text
CONTAINER_FILL = "#E4E4E4"  # outermost whole-network container
INSET_FILL = "#ECECEC"      # dashed zoom-in inset panels


@dataclass(frozen=True)
class Accent:
    """A semantic color as a fill/stroke/text trio.

    tint  — area fill (light wash; INK stays >= 4.5:1 readable on top)
    base  — strokes, arrows, plotted lines, markers (>= 3:1 vs white)
    dark  — colored *text* and small glyphs (>= 4.5:1 vs white AND vs tint)
    """

    tint: str
    base: str
    dark: str


# --------------------------------------------------------------------------- #
# Accents.  blue and amber trios embed the approved gallery values            #
# (tint #CDE8FA / dark #0B6BB2; tint #FBE8D3 / dark #B45309); blue.base is    #
# the site link blue, bridging figures and prose.  The rest extend the same   #
# temperature and saturation logic to the roles the corpus already uses.      #
# --------------------------------------------------------------------------- #

# PALETTE REVISION 2026-08-04 (Alex): the corpus moves to the ORIGINAL
# appendix palette (the matplotlib tab10 hues the pre-restyle mdl-* figures
# used) — "brighter and more cheerful", applied book-wide for consistency.
# Bases are tab10 verbatim; tints are a 12% wash of each base over white
# (matching the old alpha-fill look); darks are the same hue darkened just
# until colored TEXT clears 4.5:1 on white and on its own tint.
# CONTRAST WAIVER (recorded deliberately): the orange (2.5:1), green
# (3.4:1), and teal (2.3:1) bases sit below the 2026-08-03 mobile rule of
# >= 4.3:1 against white for line work; Alex accepted the trade for the
# brighter look.  Text remains fully compliant via the dark variants.
BLUE = Accent(tint="#E4EFF6", base="#1F77B4", dark="#1D70A9")     # primary
ORANGE = Accent(tint="#FFF0E2", base="#FF7F0E", dark="#AD560A")   # contrast
GREEN = Accent(tint="#E6F4E6", base="#2CA02C", dark="#227D22")    # go/output
PURPLE = Accent(tint="#F2EDF7", base="#9467BD", dark="#825BA6")   # memory
RED = Accent(tint="#FAE5E5", base="#D62728", dark="#C92526")      # loss/grad
TEAL = Accent(tint="#E3F7F9", base="#17BECF", dark="#0F7A84")     # state
GOLD = Accent(tint="#FBF2D4", base="#9A7212", dark="#6E5108")     # highlight
SLATE = Accent(tint=PANEL, base=MUTED, dark=INK)                  # neutral

ACCENTS = {
    "blue": BLUE, "orange": ORANGE, "green": GREEN, "purple": PURPLE,
    "red": RED, "teal": TEAL, "gold": GOLD, "slate": SLATE,
}

# Categorical cycle for data series (matplotlib prop_cycle, multi-series
# diagrams).  Order tuned for adjacent-pair separability, deuteranopia
# included: blue / orange / green / purple / red / teal / gold / gray.
CYCLE = [BLUE.base, ORANGE.base, GREEN.base, PURPLE.base,
         RED.base, TEAL.base, GOLD.base, MUTED]

# --------------------------------------------------------------------------- #
# Semantic roles — one meaning, one color, across the entire book.  These     #
# continue the conventions the corpus already teaches readers: the residual   #
# stream is blue, attention orange, feed-forward green.                       #
# --------------------------------------------------------------------------- #

ROLE = {
    "stream": BLUE,       # residual stream, main data flow, activations
    "attention": ORANGE,  # attention blocks, scores, heads
    "ffn": GREEN,         # feed-forward / MLP blocks, elementwise transforms
    "norm": SLATE,        # normalization, bookkeeping ops — neutral!
    "embed": PURPLE,      # embeddings, vocab tables, memory, KV-cache
    "state": TEAL,        # recurrent / hidden state carried across time
    "grad": RED,          # gradients, losses, backward pass, errors
    "highlight": GOLD,    # "look here" wash behind the element under discussion
    "params": SLATE,      # weights/parameters drawn as data
}

# --------------------------------------------------------------------------- #
# Typography.  One sans family everywhere (the site's own), mono for tokens/  #
# code/shapes, Computer Modern mathtext for math (matches MathJax on the      #
# page).  Sizes are px in SVG user units at DESIGN width; keep the apparent   #
# (displayed) label size in the 11.5-14px band — see the guide §Sizing.       #
# --------------------------------------------------------------------------- #

FONT_SANS = "Source Sans 3"
FONT_MONO = "Inconsolata"
FONT_MATH = "cm"                   # matplotlib mathtext fontset

FS_TITLE = 19.0     # per-panel labels "(a) post-LN" — used sparingly
FS_LABEL = 17.0     # DEFAULT: block labels, node names, axis labels
FS_SMALL = 14.5     # secondary annotations, edge labels, dims ("d = 512")
FS_TINY = 13.0      # letterspaced ALL-CAPS group/stage labels, tick labels
LETTERSPACE_CAPS = 0.08  # em tracking for TINY caps labels
FS_MIN = 12.0       # hard floor — nothing in a figure renders smaller

# --------------------------------------------------------------------------- #
# Strokes.  Emphasis ladder: change weight before you change hue.             #
# --------------------------------------------------------------------------- #

SW_HAIR = 1.5       # grids, construction geometry, panel borders
SW_BOX = 2.0        # block/pill outlines (up from the gallery's 1.2 —
                    # Alex, 2026-08-03 (twice): heavier lines, mobile first)
SW_LINE = 2.5       # DEFAULT: edges, arrows, plotted schematic lines
SW_HEAVY = 4.0      # the one emphasized path (residual stream, main flow)
SW_HALO = 10.0      # soft under-halo (tint color) beneath an emphasized line

# --------------------------------------------------------------------------- #
# Shape language.  Two-tier radius encodes hierarchy: containers rounder      #
# than leaves; small ops are full pills.  Flat fills only — no gradients,     #
# no shadows.                                                                 #
# --------------------------------------------------------------------------- #

R_BLOCK = 6.0       # leaf blocks (ops, layers)
R_PANEL = 12.0      # grouping panels / containers
# pills (op circles, tags) use radius = height/2

DASH_SOFT = "4 3"      # dashed: optional / candidate / reference structure
DOT_LEADER = "1.5 2.5"  # dotted: annotation leaders — never data flow

# Arrowheads: small filled triangle for DATA FLOW (formal, directed);
# open chevron for ANNOTATION leaders (light pointer, not a stop sign).
ARROW_L = 9.5       # data-flow head length at SW_LINE (scales with width)
ARROW_W = 7.8       # head width across the base
CHEVRON_L = 7.5     # annotation chevron arm length

# --------------------------------------------------------------------------- #
# Spacing & canvas.  8px base grid; canvas is tight around content.           #
# --------------------------------------------------------------------------- #

GAP = 8.0           # base spacing unit; use multiples
PAD_BLOCK_X = 15.0  # horizontal padding inside a leaf block
PAD_BLOCK_Y = 9.0   # vertical padding inside a leaf block
PAD_PANEL = 14.0    # padding inside grouping panels
PAD_CANVAS = 6.0    # outer margin of the whole figure

# Canonical figure widths (SVG user units ~= CSS px at 1:1 display).
# Snap to one of these instead of bespoke sizes; chapter includes then set
# width= to the same value (or omit it) so labels render at design size.
W_NARROW = 420.0    # small inline diagram
W_MEDIUM = 560.0    # DEFAULT single-concept diagram
W_COLUMN = 720.0    # full text-column figure
W_WIDE = 960.0      # multi-panel comparison (displayed at column width)

__all__ = [n for n in dir() if n.isupper()] + ["Accent"]
