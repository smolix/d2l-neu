#!/usr/bin/env python3
"""Chapter 1 (introduction) figures — APPROVED by Alex 2026-08-03; this
script now owns the five live SVGs (chapter-by-chapter review loop,
docs/figure-style-guide.md §9).

Writes -> img/:
  ml-loop.svg               the train/evaluate iteration cycle
  supervised-learning.svg   training phase vs inference phase
  data-collection.svg       supervised learning fed by an environment
  rl-environment.svg        agent-environment interaction loop
  wake-word.svg             speech -> model -> {yes, no}

Photos (koebel.jpg, death-cap.jpg, ...) are out of scope — never redraw.

Density rule (Alex, 2026-08-03): the canvas is sized by the content —
connectors are just long enough to read (~GAP*5), blocks auto-size to their
labels, and a figure must not be shrinkable without violating the spacing
minima.  Arrows are therefore wired from the blocks' RETURNED rects, never
from guessed coordinates.

Run:  python3 tools/figstyle/remake_ch01.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, sub, var

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")

ARM = 42.0          # standard connector length between adjacent blocks
TIP = 3.0           # arrowhead standoff from the target edge


def caps(f, x, y, text, anchor="middle"):
    f.text(x, y, text.upper(), size=T.FS_TINY, color=T.MUTED, anchor=anchor,
           tracking=T.LETTERSPACE_CAPS)


def hlink(f, ra, rb, y):
    """Horizontal arrow FROM rect ra TO rect rb at height y (direction
    inferred from the rects' positions)."""
    ca, cb = ra[0] + ra[2] / 2, rb[0] + rb[2] / 2
    if ca < cb:   # rightward
        f.arrow(ra[0] + ra[2] + 1, y, rb[0] - TIP, y, stroke=T.INK)
    else:         # leftward
        f.arrow(ra[0] - 1, y, rb[0] + rb[2] + TIP, y, stroke=T.INK)


def stack(f, cx, cy, label, *, w=152.0, h=44.0, n=3, off=5.0):
    """A 3-card stack (a dataset); returns the front card's rect."""
    for i in range(n - 1, 0, -1):
        f.rect(cx - w / 2 + i * off, cy - h / 2 - i * off, w, h,
               fill=T.PAPER, stroke=T.MUTED, sw=T.SW_HAIR, r=T.R_BLOCK)
    f.rect(cx - w / 2, cy - h / 2, w, h, fill=T.PAPER, stroke=T.INK,
           sw=T.SW_BOX, r=T.R_BLOCK)
    f.text(cx, cy, label, size=T.FS_LABEL)
    return (cx - w / 2, cy - h / 2, w, h)


# --------------------------------------------------------------------------- #
# ml-loop — POINT: model development is a CYCLE (design once, then iterate    #
# grab -> update -> check -> grab ...).                                       #
# --------------------------------------------------------------------------- #

def fig_ml_loop():
    f = Figure()
    Y0, Y1 = 0.0, 92.0
    design = f.block(0, Y0, "design a model")
    grab = f.block(design[0] + design[2] + ARM + 68, Y0, "grab new data",
                   role="stream")
    update = f.block(grab[0] + grab[2] + ARM + 83, Y0, "update the model")
    check = f.block(update[0] + update[2] / 2 - 10, Y1, "check if good enough")
    hlink(f, design, grab, Y0)
    hlink(f, grab, update, Y0)
    f.arrow(update[0] + update[2] / 2, Y0 + 22, update[0] + update[2] / 2,
            Y1 - 24, stroke=T.INK)
    # the loop back: check -> grab (this edge IS the figure's point)
    gx = grab[0] + grab[2] / 2
    f.ortho_arrow([(check[0] - 1, Y1), (gx, Y1), (gx, Y0 + 24)], stroke=T.INK)
    f.save("ml-loop", out_dir=OUT,
           desc="Machine learning is iterative: grab data, update, check.")


# --------------------------------------------------------------------------- #
# supervised-learning — POINT: two phases; training data (inputs + labels)    #
# produce the model, which then maps new inputs to outputs.                   #
# --------------------------------------------------------------------------- #

def _supervised(f: Figure, env: bool):
    Y0, Y1 = 0.0, 96.0
    inputs = stack(f, 76, Y0, "training inputs")
    learn = f.block(280, Y0, "supervised learning", role="attention")
    labels = stack(f, 484, Y0, "training labels")
    hlink(f, inputs, learn, Y0)
    hlink(f, labels, learn, Y0)

    inp = f.block(76, Y1, "input", min_w=130)
    model = f.block(280, Y1, "model", role="stream", min_w=140)
    outp = f.block(484, Y1, "output", min_w=130)
    f.arrow(280, Y0 + 22, 280, Y1 - 24, stroke=T.INK)  # learning -> model
    hlink(f, inp, model, Y1)
    hlink(f, model, outp, Y1)

    caps(f, -14, Y0, "training", anchor="end")
    caps(f, -14, Y1, "inference", anchor="end")

    if env:
        YE = -98.0
        e = f.block(280, YE, "environment", accent=T.SLATE)
        # the environment supplies BOTH the inputs and the labels
        top = Y0 - 22 - 2 * 5.0    # top edge of the stacks' back card
        f.ortho_arrow([(e[0] - 1, YE), (76, YE), (76, top - TIP)],
                      stroke=T.INK)
        f.ortho_arrow([(e[0] + e[2] + 1, YE), (484, YE), (484, top - TIP)],
                      stroke=T.INK)


def fig_supervised_learning():
    f = Figure()
    _supervised(f, env=False)
    f.save("supervised-learning", out_dir=OUT,
           desc="Supervised learning: training on labeled data, then inference.")


def fig_data_collection():
    f = Figure()
    _supervised(f, env=True)
    f.save("data-collection", out_dir=OUT,
           desc="Training data — inputs and labels — comes from an environment.")


# --------------------------------------------------------------------------- #
# rl-environment — POINT: a closed interaction loop; the agent acts on the    #
# environment, which returns observations and rewards.                        #
# --------------------------------------------------------------------------- #

def fig_rl_environment():
    f = Figure()
    AX, MX, EX = 0.0, 262.0, 524.0
    Y0, Y1, Y2 = 0.0, 92.0, 184.0

    agent = f.block(AX, Y1, "agent", role="stream", min_w=110)
    envir = f.block(EX, Y1, "environment", accent=T.SLATE)
    action = f.block(MX, Y0, "action", role="ffn", min_w=120)
    reward = f.block(MX, Y1, "reward", accent=T.GOLD, min_w=120)
    obs = f.block(MX, Y2, "observation")

    f.ortho_arrow([(AX, agent[1] - 1), (AX, Y0), (action[0] - TIP, Y0)],
                  stroke=T.INK)
    f.ortho_arrow([(action[0] + action[2] + 1, Y0), (EX, Y0),
                   (EX, envir[1] - TIP - 1)], stroke=T.INK)
    hlink(f, envir, reward, Y1)
    hlink(f, reward, agent, Y1)
    f.ortho_arrow([(EX, envir[1] + envir[3] + 1), (EX, Y2),
                   (obs[0] + obs[2] + TIP, Y2)], stroke=T.INK)
    f.ortho_arrow([(obs[0] - 1, Y2), (AX, Y2),
                   (AX, agent[1] + agent[3] + TIP + 1)], stroke=T.INK)

    f.save("rl-environment", out_dir=OUT,
           desc="Reinforcement learning: agent and environment in a loop.")


# --------------------------------------------------------------------------- #
# wake-word — POINT: raw speech goes in, a binary decision comes out.         #
# (Approved 2026-08-03 — do not change without re-review.)                    #
# --------------------------------------------------------------------------- #

def fig_wake_word():
    f = Figure()
    # speech waveform (the data -> blue)
    heights = [12, 26, 40, 24, 48, 32, 18, 30, 12]
    for i, h in enumerate(heights):
        x = i * 13.0
        f.rect(x - 3.2, -h / 2, 6.4, h, fill=T.BLUE.base, stroke="none",
               sw=0, r=3.2)
    caps(f, 52, 44, "speech")

    # microphone glyph (ink)
    mx = 165.0
    f.rect(mx - 8, -26, 16, 30, fill=T.INK, stroke="none", sw=0, r=8)
    f.path(f"M {mx - 14} -6 A 14 14 0 0 0 {mx + 14} -6",
           mx - 14, -6, mx + 14, 9, fill="none", stroke=T.INK, sw=T.SW_BOX)
    f.line(mx, 8, mx, 18, stroke=T.INK, sw=T.SW_BOX)
    f.line(mx - 9, 18, mx + 9, 18, stroke=T.INK, sw=T.SW_BOX)

    f.arrow(mx + 30, 0, mx + 95, 0, stroke=T.INK)
    f.block(mx + 200, 0, "wake-word model", role="stream", min_w=195)
    f.arrow(mx + 300, 0, mx + 365, 0, stroke=T.INK)
    f.text(mx + 375, 0, [Span("{yes, no}", "m")], size=T.FS_LABEL,
           color=T.INK, anchor="start")

    f.save("wake-word", out_dir=OUT,
           desc="Wake-word detection: speech in, yes/no out.")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_ml_loop()
    fig_supervised_learning()
    fig_data_collection()
    fig_rl_environment()
    fig_wake_word()
    print("wrote chapter-1 figures to", OUT)


if __name__ == "__main__":
    main()
