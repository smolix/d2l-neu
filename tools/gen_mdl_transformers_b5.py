#!/usr/bin/env python3
"""Generate the architecture figure of chapter_transformers/vision-transformer.md
(B5) with the deterministic SVG composer (figstyle.svg).

Figure:
  mdl-transformers-vit.svg -- the vision transformer from bottom to top: a
      96 x 96 Fashion-MNIST-like image cut into a 6 x 6 grid of 16 x 16
      patches, the strided-convolution patch embedding, the 36 patch tokens
      with a prepended <cls> token and added position embeddings, L pre-norm
      encoder blocks (multi-head attention, MLP) on the residual stream, the
      37 output tokens of which only <cls> is read, and the head.

Colors follow the semantic roles of docs/figure-style-guide.md section 4.3:
residual stream and tokens blue, attention orange, MLP green, embeddings
(patch embedding, <cls>) purple, normalization neutral.  The layout mirrors
mdl-transformers-gpt-stack.svg (gen_mdl_transformers_b2.py) so the two
whole-model figures of the chapter read alike.  Coordinates are px (1 SVG
unit = 1 CSS px), y downward.  The figure is written to img/ (book) and
img/auto/ (slides).

Run with the repo's pytorch venv (it has fontTools):

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b5.py

Byte-idempotent: no timestamps, no randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import tokens as T  # noqa: E402
from figstyle.svg import IMG_DIR, Figure, Span  # noqa: E402

STREAM = T.ROLE["stream"]
ATTN = T.ROLE["attention"]
FFN = T.ROLE["ffn"]
EMBED = T.ROLE["embed"]

SX = 0.0           # x of the residual stream (center of the token rows)
BC = 205.0         # x of the branch column (sublayer box centers)
BW = 214.0         # sublayer box width
NW = 76.0          # branch norm box width

TOK_W, TOK_H = 26.0, 26.0   # one token square
CLS_W = 40.0                # the wider <cls> square
TOK_GAP = 5.0
ELL_W = 22.0                # room for the ellipsis


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


def token_row(f, cy, *, outputs=False):
    """<cls>, 1, 2, 3, ..., 36 as a row of squares centered on SX.

    Returns the center x of the <cls> square and the x extents (x0, x_p, x1)
    of the row, where x_p is the left edge of the first patch token.
    In the output row, the patch outputs are drawn faint: the head never
    reads them."""
    labels = ["1", "2", "3", None, "35", "36"]
    width = CLS_W + TOK_GAP + sum(
        (ELL_W if t is None else TOK_W) + TOK_GAP for t in labels) - TOK_GAP
    x = SX - width / 2
    x0 = x
    # <cls>: a learned embedding (purple)
    f.rect(x, cy - TOK_H / 2, CLS_W, TOK_H, fill=EMBED.tint,
           stroke=EMBED.base, sw=T.SW_HEAVY if outputs else T.SW_BOX,
           r=4.0)
    f.text(x + CLS_W / 2, cy, [Span("cls", "m")], size=T.FS_SMALL)
    cls_x = x + CLS_W / 2
    x += CLS_W + TOK_GAP
    xp = x
    for t in labels:
        if t is None:
            f.text(x + ELL_W / 2, cy, "…", size=T.FS_LABEL,
                   color=T.FAINT if outputs else T.INK)
            x += ELL_W + TOK_GAP
            continue
        if outputs:
            f.rect(x, cy - TOK_H / 2, TOK_W, TOK_H, fill=T.PAPER,
                   stroke=T.FAINT, sw=T.SW_HAIR, r=4.0)
            f.text(x + TOK_W / 2, cy, [Span(t, "m")], size=T.FS_MIN,
                   color=T.FAINT)
        else:
            f.rect(x, cy - TOK_H / 2, TOK_W, TOK_H, fill=STREAM.tint,
                   stroke=STREAM.base, sw=T.SW_BOX, r=4.0)
            f.text(x + TOK_W / 2, cy, [Span(t, "m")], size=T.FS_MIN)
        x += TOK_W + TOK_GAP
    return cls_x, (x0, xp, x - TOK_GAP)


def bracket(f, x0, x1, y, tick, color, sw=T.SW_BOX):
    """A square bracket from x0 to x1 at height y; its ends turn by `tick`
    (positive = downward) toward the row of tokens it gathers."""
    f.polyline([(x0, y + tick), (x0, y), (x1, y), (x1, y + tick)],
               stroke=color, sw=sw)


def image(f, cx, top, side):
    """A 96 x 96 grayscale image (a T-shirt silhouette) cut into 6 x 6
    patches."""
    x0 = cx - side / 2
    f.rect(x0, top, side, side, fill=T.PANEL, stroke="none", sw=0)
    shirt = [(0.30, 0.10), (0.40, 0.10), (0.45, 0.17), (0.55, 0.17),
             (0.60, 0.10), (0.70, 0.10), (0.93, 0.29), (0.84, 0.43),
             (0.75, 0.37), (0.75, 0.92), (0.25, 0.92), (0.25, 0.37),
             (0.16, 0.43), (0.07, 0.29)]
    pts = [(x0 + u * side, top + v * side) for u, v in shirt]
    d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
    f.path(d, x0, top, x0 + side, top + side, fill=T.FAINT, stroke="none",
           sw=0)
    # patch grid
    for k in range(7):
        t = k * side / 6
        f.line(x0 + t, top, x0 + t, top + side, stroke=T.PAPER, sw=T.SW_BOX)
        f.line(x0, top + t, x0 + side, top + t, stroke=T.PAPER, sw=T.SW_BOX)
    f.rect(x0, top, side, side, fill="none", stroke=T.MUTED, sw=T.SW_HAIR)
    return x0


def fig_vit() -> None:
    f = Figure()

    # ---- top: logits, head, final norm above the <cls> output ----------
    y_out = 150.0                       # output token row
    # the output row is centered on SX; find the <cls> column first
    cls_x, (ox0, _, ox1) = token_row(f, y_out, outputs=True)
    y_ob = y_out + TOK_H / 2 + 9          # bracket under the output row
    bracket(f, ox0, ox1, y_ob, -6, STREAM.base)
    f.text(cls_x, 0, "class logits", size=T.FS_LABEL, va="middle")
    _, hy, _, hh = f.block(cls_x, 46, "linear head", min_w=150)
    f.arrow(cls_x, hy, cls_x, 13, stroke=STREAM.base, sw=T.SW_HEAVY)
    _, ny, _, nh = f.block(cls_x, 98, "final norm", min_w=150)
    f.arrow(cls_x, ny, cls_x, hy + hh, stroke=STREAM.base, sw=T.SW_HEAVY)
    f.arrow(cls_x, y_out - TOK_H / 2 - 1, cls_x, ny + nh,
            stroke=STREAM.base, sw=T.SW_HEAVY)

    # ---- the encoder block (x L) on the stream ------------------------
    p_top, p_bot = 210.0, 474.0
    f.panel(SX - 44, p_top, BC + BW / 2 + 18 - (SX - 44), p_bot - p_top)
    f.line(SX, p_bot + 22, SX, p_top - 8, stroke=STREAM.base,
           sw=T.SW_HEAVY)
    f.arrow(SX, p_top - 6, SX, y_ob + 1, stroke=STREAM.base, sw=T.SW_HEAVY)
    right = sublayer(f, 446, 392, 352, "multi-head attention", ATTN)
    sublayer(f, 322, 268, 230, "MLP (GELU)", FFN)
    f.text(right + 30, (p_top + p_bot) / 2, [Span("× "), Span("L", "i")],
           size=T.FS_TITLE, anchor="start")

    # ---- input tokens: <cls> + 36 patch tokens + positions --------------
    y_in = 518.0
    _, (rx0, rxp, rx1) = token_row(f, y_in)
    y_ib = y_in - TOK_H / 2 - 9           # bracket over the input row
    bracket(f, rx0, rx1, y_ib, 6, STREAM.base)
    f.line(SX, y_ib + 1, SX, p_bot + 22, stroke=STREAM.base,
           sw=T.SW_HEAVY)
    f.text(rx1 + 14, y_in - 9, "+ learned position", size=T.FS_SMALL,
           color=T.MUTED, anchor="start")
    f.text(rx1 + 14, y_in + 9, "embeddings", size=T.FS_SMALL,
           color=T.MUTED, anchor="start")
    f.text(rx0 - 10, y_in, "37 tokens", size=T.FS_SMALL, color=T.MUTED,
           anchor="end")

    # ---- patch embedding, fed from the image on its right ---------------
    patch_cx = (rxp + rx1) / 2              # center of the 36 patch tokens
    y_emb = 585.0
    bx, ey, bw, eh = f.block(patch_cx, y_emb,
                             "patch embedding: 16 × 16 conv, stride 16",
                             accent=EMBED)
    y_pb = y_in + TOK_H / 2 + 9           # bracket under the patch tokens
    bracket(f, rxp, rx1, y_pb, -6, T.INK)
    f.arrow(patch_cx, ey, patch_cx, y_pb + 1, stroke=T.INK)
    side = 108.0
    img_cx = bx + bw + 40 + side / 2
    x0 = image(f, img_cx, y_emb - side / 2, side)
    f.arrow(x0 - 1, y_emb, bx + bw, y_emb, stroke=T.INK)
    f.text(img_cx, y_emb + side / 2 + 14, "96 × 96 image, 6 × 6 patches",
           size=T.FS_SMALL, color=T.MUTED, va="top")

    desc = ("The vision transformer from bottom to top: a 96 by 96 image "
            "cut into a 6 by 6 grid of 16 by 16 patches; a strided "
            "convolution that embeds each patch as a token; a learned cls "
            "token prepended to the 36 patch tokens and learned position "
            "embeddings added; L pre-norm encoder blocks with multi-head "
            "attention and a GELU MLP on the residual stream; and a final "
            "norm and linear head that read the cls output only.")
    svg = f.render(desc=desc)
    for out in (IMG_DIR, os.path.join(IMG_DIR, "auto")):
        path = os.path.join(out, "mdl-transformers-vit.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    fig_vit()
