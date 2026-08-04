#!/usr/bin/env python3
"""Chapter 16 (GANs) legacy-figure remake — chapter-by-chapter review loop
(docs/figure-style-guide.md §9).

Writes -> img/:
  gan.svg   the generator/discriminator loop (original composition,
            token colors: networks blue, data boxes neutral)

Run:  python3 tools/figstyle/remake_ch16.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, var

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")


def fig_gan():
    """POINT: the generator maps noise to fakes; the discriminator sees
    fakes and real data and must tell them apart.  Original composition
    (vertical left column, real data joining from the right), token
    colors only: the two NETWORKS blue, the data boxes neutral."""
    f = Figure()
    LX, RX = 110.0, 350.0            # left (generator) / right (real) columns
    DX = (LX + RX) / 2               # discriminator, centred between them
    y_top, y_disc, y_data, y_gen, y_noise = 20.0, 110.0, 200.0, 290.0, 380.0

    # data boxes neutral, network boxes blue (as in the original)
    f.block(LX, y_noise, [Span("Noise ", "r"), var("z")], min_w=170)
    f.block(LX, y_gen, "Generator", accent=T.BLUE, min_w=170)
    f.block(LX, y_data, [Span("Fake ", "r"), var("G"), Span("(", "r"),
                         var("z"), Span(")", "r")], min_w=170)
    f.block(RX, y_data, [Span("Real ", "r"), var("x")], min_w=170)
    f.block(DX, y_disc, "Discriminator", accent=T.BLUE, min_w=220)
    f.block(DX, y_top, "Is real or fake", min_w=190)

    # left column, bottom -> top
    f.arrow(LX, y_noise - 20, LX, y_gen + 20)
    f.arrow(LX, y_gen - 20, LX, y_data + 20)
    # fake and real feed the discriminator (tips ON its bottom edge)
    f.arrow(LX, y_data - 20, DX - 40, y_disc + 20)
    f.arrow(RX, y_data - 20, DX + 40, y_disc + 20)
    # verdict
    f.arrow(DX, y_disc - 20, DX, y_top + 20)

    f.save("gan", out_dir=IMG,
           desc="A GAN: the generator maps noise to fakes; the "
                "discriminator must tell fakes from real data.")


def main():
    fig_gan()
    print("wrote chapter-16 remake to", IMG)


if __name__ == "__main__":
    main()
