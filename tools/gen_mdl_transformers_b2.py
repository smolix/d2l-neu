#!/usr/bin/env python3
"""Generate the whole-model figure of chapter_transformers/gpt.md (B2) with the
deterministic SVG composer (figstyle.svg).

Figure:
  mdl-transformers-gpt-stack.svg -- a GPT from bottom to top: token ids, the
      token embedding E, the residual stream through L pre-norm blocks
      (causal attention, feed-forward network), the final normalization, and
      the tied head that reuses E to produce next-token logits.

Colors follow the semantic roles of docs/figure-style-guide.md section 4.3:
residual stream blue, attention orange, feed-forward green, embedding purple,
normalization neutral.  Coordinates are px (1 SVG unit = 1 CSS px), y
downward.  The figure is written to img/ (book) and img/auto/ (slides).

Run with the repo's pytorch venv (it has fontTools):

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b2.py

Byte-idempotent: no timestamps, no randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import tokens as T  # noqa: E402
from figstyle.svg import IMG_DIR, Figure, Span, sub  # noqa: E402

STREAM = T.ROLE["stream"]
ATTN = T.ROLE["attention"]
FFN = T.ROLE["ffn"]
EMBED = T.ROLE["embed"]

SX = 0.0           # x of the residual stream (and of the spine blocks)
BC = 205.0         # x of the branch column (sublayer box centers)
BW = 214.0         # sublayer box width
NW = 76.0          # branch norm box width
SPINE_W = 176.0    # width of the spine blocks (embedding, norm, head)


def sublayer(f, y_read, y_box, y_add, label, accent):
    """A pre-norm sublayer: read the stream at y_read, normalize, apply the
    sublayer at y_box, add the result back into the stream at y_add."""
    _, ny, _, nh = f.block(BC, y_read, "norm", min_w=NW, size=T.FS_SMALL)
    f.line(SX, y_read, BC - NW / 2, y_read, stroke=accent.base)
    x, by, w, bh = f.block(BC, y_box, label, accent=accent, min_w=BW)
    f.arrow(BC, ny, BC, by + bh, stroke=accent.base)
    f.polyline([(BC, by), (BC, y_add), (SX + 13, y_add)], stroke=accent.base)
    f.pill_op(SX, y_add, "+", r=11)
    return x + w


def fig_gpt_stack() -> None:
    f = Figure()

    # top: logits over the vocabulary
    f.text(SX, 0, "next-token logits", size=T.FS_LABEL, va="middle")

    # tied head, final norm
    _, hy, hw, hh = f.block(SX, 52, [Span("tied head "), Span("E", "b")],
                            accent=EMBED, min_w=SPINE_W)
    f.arrow(SX, hy, SX, 13, stroke=STREAM.base, sw=T.SW_HEAVY)
    _, ny, _, nh = f.block(SX, 112, "final norm", min_w=SPINE_W)
    f.arrow(SX, ny, SX, hy + hh, stroke=STREAM.base, sw=T.SW_HEAVY)

    # the block: two pre-norm sublayers on the stream, inside a panel
    p_top, p_bot = 148.0, 420.0
    f.panel(SX - 44, p_top, BC + BW / 2 + 18 - (SX - 44), p_bot - p_top)
    # stream through the panel, under the (+) pills
    f.line(SX, p_bot + 30, SX, ny + nh + 9, stroke=STREAM.base,
           sw=T.SW_HEAVY)
    f.arrow(SX, ny + nh + 12, SX, ny + nh, stroke=STREAM.base,
            sw=T.SW_HEAVY)
    right = sublayer(f, 392, 333, 285, "causal attention (RoPE)", ATTN)
    sublayer(f, 252, 205, 168, "feed-forward network", FFN)
    f.text(right + 30, (p_top + p_bot) / 2, [Span("× "), Span("L", "i")],
           size=T.FS_TITLE, anchor="start")

    # embedding and tokens at the bottom
    ex, ey, ew, eh = f.block(SX, 470, [Span("token embedding "),
                                       Span("E", "b")],
                             accent=EMBED, min_w=SPINE_W)
    f.line(SX, ey, SX, p_bot + 30, stroke=STREAM.base, sw=T.SW_HEAVY)
    f.text(SX + 14, 436, "learned positions are added here",
           size=T.FS_SMALL, color=T.INK, anchor="start")
    f.text(SX, 532, [Span("token ids "), Span("x", "i"), sub("1"),
                     Span(", …, "), Span("x", "i"), sub("T")],
           size=T.FS_LABEL)
    f.arrow(SX, 518, SX, ey + eh, stroke=T.INK)

    # the shared table: a dashed connector on the left
    lx = ex - 26
    f.polyline([(ex, ey + eh / 2), (lx, ey + eh / 2), (lx, hy + hh / 2),
                (ex, hy + hh / 2)], stroke=EMBED.base, sw=T.SW_BOX,
               dash=T.DASH_SOFT)
    f.text(lx - 8, (ey + hy + hh) / 2 - 10, "same", size=T.FS_SMALL,
           color=EMBED.dark, anchor="end")
    f.text(lx - 8, (ey + hy + hh) / 2 + 10, "matrix", size=T.FS_SMALL,
           color=EMBED.dark, anchor="end")

    desc = ("A GPT from bottom to top: token ids, the token embedding E, "
            "L pre-norm transformer blocks with causal attention and a "
            "feed-forward network on the residual stream, a final norm, and "
            "a tied head that reuses E to produce next-token logits.")
    svg = f.render(desc=desc)
    for out in (IMG_DIR, os.path.join(IMG_DIR, "auto")):
        with open(os.path.join(out, "mdl-transformers-gpt-stack.svg"), "w") as fh:
            fh.write(svg)
        print(f"wrote {os.path.relpath(os.path.join(out, 'mdl-transformers-gpt-stack.svg'))}")


if __name__ == "__main__":
    fig_gpt_stack()
