#!/usr/bin/env python3
"""Chapter 1 (introduction) figures — APPROVED by Alex 2026-08-03; this
script now owns the five live SVGs (chapter-by-chapter review loop,
docs/figure-style-guide.md §9).

Writes -> img/:
  ml-loop.svg               the train/evaluate iteration cycle
  supervised-learning.svg   training phase vs inference phase
  data-collection.svg       supervised learning fed by an environment
  rl-environment.svg        agent-environment interaction loop
  wake-word.svg             two ends of the pipeline: on-device wake-word
                            model (yes/no), then the large assistant model
  intro-speech.svg          synthetic waveform of "deep learning"; each word
                            spans thousands of samples
  intro-next-token.svg      next-token prediction as 100,000-way
                            classification (five classes shown)
  intro-pretrain-adapt.svg  pretrain once on web text, then adapt the base
                            model three ways
  intro-generative.svg      discriminative vs generative, as mirror rows
  intro-scaling.svg         schematic loss vs compute per model size, with
                            the lower envelope

The six figures from wake-word.svg on were added/extended 2026-11 for the
introduction rewrite (review proposal section 6) and await Alex's review.

Photos (koebel.jpg, death-cap.jpg, ...) are out of scope — never redraw.

Density rule (Alex, 2026-08-03): the canvas is sized by the content —
connectors are just long enough to read (~GAP*5), blocks auto-size to their
labels, and a figure must not be shrinkable without violating the spacing
minima.  Arrows are therefore wired from the blocks' RETURNED rects, never
from guessed coordinates.

Run:  python3 tools/figstyle/remake_ch01.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, sub, var
from figstyle.textmetrics import measure

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
# Shared helpers for the 2026-11 introduction figures.                        #
# --------------------------------------------------------------------------- #

def runs(f, x, y, parts, *, size=T.FS_SMALL, color=T.INK, anchor="start",
         kind="r"):
    """Typeset text with superscripts kept at or above FS_MIN.

    ``parts`` is a list of strings and ``("^", exp)`` tuples.  Span-based
    superscripts shrink to 0.72x, which drops a FS_SMALL exponent below the
    FS_MIN floor; this lays the runs out by hand with the exponent at
    max(FS_MIN, 0.72 * size).  Returns the bbox (x0, y0, x1, y1).
    """
    fam = "mono" if kind == "m" else "sans"
    esz = max(T.FS_MIN, 0.72 * size)
    items = []
    for p in parts:
        if isinstance(p, tuple):
            items.append((p[1], esz, -0.42 * size))
        else:
            items.append((p, size, 0.0))
    total = sum(measure(s, sz, fam).width for s, sz, _ in items)
    x0 = {"middle": x - total / 2, "end": x - total}.get(anchor, x)
    base = y + measure("Hg", size, fam).cap_height / 2
    cx, y0, y1 = x0, math.inf, -math.inf
    for s, sz, dy in items:
        bb = f.text(cx, base + dy, [Span(s, kind)], size=sz, color=color,
                    anchor="start", va="baseline")
        y0, y1 = min(y0, bb[1]), max(y1, bb[3])
        cx = bb[2]
    return (x0, y0, cx, y1)


def mic(f, mx, my=0.0):
    """Microphone glyph (ink), capsule centred on (mx, my - 11)."""
    f.rect(mx - 8, my - 26, 16, 30, fill=T.INK, stroke="none", sw=0, r=8)
    f.path(f"M {mx - 14} {my - 6} A 14 14 0 0 0 {mx + 14} {my - 6}",
           mx - 14, my - 6, mx + 14, my + 9, fill="none", stroke=T.INK,
           sw=T.SW_BOX)
    f.line(mx, my + 8, mx, my + 18, stroke=T.INK, sw=T.SW_BOX)
    f.line(mx - 9, my + 18, mx + 9, my + 18, stroke=T.INK, sw=T.SW_BOX)


def box(f, x, y, w, h, label, *, accent=None, size=T.FS_LABEL):
    """Fixed-size rounded block with top-left (x, y); returns its rect."""
    a = accent
    f.rect(x, y, w, h, fill=a.tint if a else T.PAPER,
           stroke=a.base if a else T.INK, sw=T.SW_BOX, r=T.R_BLOCK)
    f.text(x + w / 2, y + h / 2, label, size=size)
    return (x, y, w, h)


def bw(label, size=T.FS_LABEL, fam="sans"):
    """Width f.block() gives a one-line label."""
    return measure(label, size, fam).width + 2 * T.PAD_BLOCK_X


# --------------------------------------------------------------------------- #
# wake-word — POINT: the two ends of one pipeline.  A tiny always-on model on  #
# the phone answers yes/no; only then does the request go to a model a        #
# million times larger in a data center, which answers in words.  The size    #
# contrast of the two model boxes is the content.                             #
# (2026-11 extension of the 2026-08-03 approved figure; row 1 is unchanged.)  #
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

    mx = 165.0
    mic(f, mx)

    # row 1: on the phone
    small = f.block(mx + 30 + ARM + bw("wake-word model") / 2, 0,
                    "wake-word model", role="stream")
    f.arrow(mx + 30, 0, small[0] - TIP, 0, stroke=T.INK)
    yx = small[0] + small[2] + ARM + 8
    f.arrow(small[0] + small[2] + 1, 0, yx - 8, 0, stroke=T.INK)
    f.text(yx, 0, [Span("{yes, no}", "m")], size=T.FS_LABEL, anchor="start")
    yes_cx = (yx + measure("{", T.FS_LABEL, "mono").width
              + measure("yes", T.FS_LABEL, "mono").width / 2)
    runs(f, small[0] + small[2] / 2, small[1] + small[3] + 14,
         ["on the phone, ~10", ("^", "5"), " parameters"], color=T.MUTED,
         anchor="middle")

    # row 2: in the data center
    Y2 = 112.0
    BW, BH = 236.0, 84.0
    req = f.text(mx, Y2, [Span("“Will it rain?”", "m")], size=T.FS_LABEL)
    f.arrow(mx, 26, mx, req[1] - 4, stroke=T.INK)
    big = box(f, req[2] + ARM + 4, Y2 - BH / 2, BW, BH, "assistant model",
              accent=T.BLUE)
    f.arrow(req[2] + 8, Y2, big[0] - TIP, Y2, stroke=T.INK)
    assert big[0] + 24 < yes_cx < big[0] + big[2] - 24, "yes misses the box"
    f.arrow(yes_cx, 13, yes_cx, big[1] - TIP, stroke=T.INK)
    ax = big[0] + big[2] + ARM + 8
    f.arrow(big[0] + big[2] + 1, Y2, ax - 8, Y2, stroke=T.INK)
    f.text(ax, Y2, [Span("“Yes, at 3 pm.”", "m")], size=T.FS_LABEL,
           anchor="start")
    runs(f, big[0] + big[2] / 2, big[1] + big[3] + 14,
         ["in a data center, ~10", ("^", "11"), " parameters"],
         color=T.MUTED, anchor="middle")

    f.save("wake-word", out_dir=OUT,
           desc="A small wake-word model on the phone answers yes or no; "
                "the request then goes to a large assistant model.")


# --------------------------------------------------------------------------- #
# intro-speech — POINT: one spoken word is thousands of numbers.  A synthetic #
# one-second recording of "deep learning" at 44.1 kHz; brackets mark each     #
# word and how many samples it spans.                                         #
# --------------------------------------------------------------------------- #

FS_AUDIO = 44_100


def _speech_signal():
    """Deterministic synthetic 'deep learning' (1 s at 44.1 kHz)."""
    import numpy as np
    n = FS_AUDIO
    t = np.arange(n) / FS_AUDIO
    f0 = 150.0 - 35.0 * t                      # falling pitch
    ph = 2 * np.pi * np.cumsum(f0) / FS_AUDIO
    voiced = sum(w * np.sin(k * ph + 0.7 * k)
                 for k, w in [(1, 1.0), (2, 0.6), (3, 0.45), (5, 0.3),
                              (7, 0.2)])
    voiced /= np.abs(voiced).max()
    noise = np.random.default_rng(0).standard_normal(n)
    noise /= np.abs(noise).max()
    # (time, amplitude) envelopes: voiced sounds and bursts/fricatives
    v_env = np.interp(t, [0, .075, .10, .27, .30, .44, .47, .50, .60, .65,
                          .68, .71, .74, .82, .85, .93, .96, 1.0],
                         [0, 0, .95, .85, 0, 0, .45, .9, .8, .5,
                          .38, .45, .75, .6, .3, .25, 0, 0])
    b_env = np.interp(t, [0, .058, .062, .075, .335, .338, .36, 1.0],
                         [0, 0, .55, 0, 0, .4, 0, 0])
    trem = 1 + 0.12 * np.sin(2 * np.pi * 11 * t)
    return v_env * trem * voiced + b_env * noise + 0.012 * noise


def fig_speech():
    import numpy as np
    sig = _speech_signal()
    f = Figure()
    W, H = 600.0, 46.0                       # px for 1 s; half-height
    cols = int(W)
    edges = np.linspace(0, len(sig), cols + 1).astype(int)
    hi = [sig[a:b].max() for a, b in zip(edges[:-1], edges[1:])]
    lo = [sig[a:b].min() for a, b in zip(edges[:-1], edges[1:])]
    top = " ".join(f"L {i + .5:.1f} {-H * v:.2f}" for i, v in enumerate(hi))
    bot = " ".join(f"L {i + .5:.1f} {-H * v:.2f}"
                   for i, v in reversed(list(enumerate(lo))))
    d = "M" + top[1:] + " " + bot + " Z"
    f.path(d, 0, -H, W, H, fill=T.BLUE.base, stroke=T.BLUE.base, sw=0.6)

    YB = H + 12.0                              # bracket line
    for word, t0, t1 in [("deep", .058, .36), ("learning", .44, .96)]:
        x0, x1 = t0 * W, t1 * W
        f.polyline([(x0, YB - 7), (x0, YB), (x1, YB), (x1, YB - 7)],
                   stroke=T.INK, sw=T.SW_HAIR)
        n = round((t1 - t0) * FS_AUDIO, -3)
        f.text((x0 + x1) / 2, YB + 18, word, size=T.FS_LABEL)
        f.text((x0 + x1) / 2, YB + 40, f"≈ {int(n):,} samples",
               size=T.FS_SMALL, color=T.MUTED)
    f.save("intro-speech", out_dir=OUT,
           desc="One second of speech, 44,100 samples: each word spans "
                "thousands of them.")


# --------------------------------------------------------------------------- #
# intro-next-token — POINT: a language model is a classifier.  The prompt is  #
# the input; the output is a probability for every token in a vocabulary of  #
# ~100,000 classes (five shown).  Probabilities are illustrative.            #
# --------------------------------------------------------------------------- #

def fig_next_token():
    f = Figure()
    probs = [("Paris", .71), ("a", .08), ("the", .05), ("located", .03),
             ("Lyon", .01)]
    prompt = f.block(0, 0, [Span("The capital of France is", "m")])
    PITCH, BH, L = 30.0, 18.0, 250.0
    tw = max(measure(w, T.FS_LABEL, "mono").width for w, _ in probs)
    xl = prompt[0] + prompt[2] + ARM + 8 + tw    # token labels end here
    xb = xl + 10                                  # bars start here
    f.arrow(prompt[0] + prompt[2] + 1, 0, xl - tw - 8, 0, stroke=T.INK)
    n = len(probs)
    for i, (tok, p) in enumerate(probs):
        y = (i - (n - 1) / 2) * PITCH
        f.text(xl, y, [Span(tok, "m")], size=T.FS_LABEL, anchor="end")
        f.rect(xb, y - BH / 2, max(p * L, 2.0), BH, fill=T.BLUE.base,
               stroke="none", sw=0)
        f.text(xb + max(p * L, 2.0) + 7, y, f"{p:.2f}", size=T.FS_SMALL,
               color=T.MUTED, anchor="start")
    y_top = -(n - 1) / 2 * PITCH - BH / 2 - 4
    y_bot = (n - 1) / 2 * PITCH + BH / 2 + 4
    f.line(xb, y_top, xb, y_bot, stroke=T.INK, sw=T.SW_HAIR)
    caps(f, xb + 0.5 * L * 0.71, y_bot + 18, "100,000 classes, five shown")
    f.save("intro-next-token", out_dir=OUT,
           desc="Next-token prediction is classification over the "
                "vocabulary.")


# --------------------------------------------------------------------------- #
# intro-pretrain-adapt — POINT: one expensive pretraining run on unlabeled    #
# text makes a base model; three cheap adaptations make it a classifier, an   #
# assistant, or an agent.  Processes are arrow labels, things are boxes; the  #
# heavy arrow is the expensive step.                                          #
# --------------------------------------------------------------------------- #

def fig_pretrain_adapt():
    f = Figure()
    # the corpus: a large grey stack
    SW_, SH, NC, OFF = 136.0, 84.0, 5, 5.0
    for i in range(NC - 1, 0, -1):
        f.rect(-SW_ / 2 + i * OFF, -SH / 2 - i * OFF, SW_, SH, fill=T.PANEL,
               stroke=T.MUTED, sw=T.SW_HAIR, r=T.R_BLOCK)
    f.rect(-SW_ / 2, -SH / 2, SW_, SH, fill=T.PANEL, stroke=T.MUTED,
           sw=T.SW_BOX, r=T.R_BLOCK)
    f.text(0, -20, "web text", size=T.FS_LABEL)
    runs(f, 0, 3, ["~10", ("^", "13"), " tokens"], anchor="middle")
    f.text(0, 24, "no labels", size=T.FS_SMALL)
    sx1 = SW_ / 2 + (NC - 1) * OFF

    # pretraining: the one heavy arrow
    lab1 = "pretraining:"
    lab2 = "predict the next token"
    span = measure(lab2, T.FS_SMALL).width + 2 * 12
    base = f.block(sx1 + span + bw("base model") / 2, 0, "base model",
                   role="stream")
    f.arrow(sx1 + 4, 0, base[0] - TIP, 0, stroke=T.INK, sw=T.SW_HEAVY)
    mid = (sx1 + base[0]) / 2
    f.text(mid, -36, lab1, size=T.FS_SMALL)
    f.text(mid, -16, lab2, size=T.FS_SMALL)

    # three light adaptations
    ROW = 62.0
    xt = base[0] + base[2] + 20
    lw = max(measure(s, T.FS_SMALL).width for s in
             ["fine-tune on examples", "learn from preferences",
              "give tools"])
    xr = xt + 12 + lw + 12                     # target left edge
    TW = max(bw(s) for s in ["classifier", "assistant", "agent"])
    for k, (how, what) in enumerate([("fine-tune on examples", "classifier"),
                                     ("learn from preferences", "assistant"),
                                     ("give tools", "agent")]):
        y = (k - 1) * ROW
        f.block(xr + TW / 2, y, what, role="stream", min_w=TW)
        pts = [(base[0] + base[2] + 1, 0), (xt, 0), (xt, y), (xr - TIP, y)]
        if y == 0:
            pts = [pts[0], pts[-1]]
        f.ortho_arrow(pts, stroke=T.INK)
        f.text(xt + 12, y - 15, how, size=T.FS_SMALL, anchor="start")
    f.save("intro-pretrain-adapt", out_dir=OUT,
           desc="Pretrain once on unlabeled text, then adapt the base model "
                "in several cheap ways.")


# --------------------------------------------------------------------------- #
# intro-generative — POINT: the two directions mirror each other.  A          #
# discriminative model maps an image to a label; a generative model maps a    #
# description to an image.                                                    #
# --------------------------------------------------------------------------- #

def _tile(f, cx, cy, s):
    f.rect(cx - s / 2, cy - s / 2, s, s, fill=T.PANEL, stroke=T.INK,
           sw=T.SW_BOX, r=T.R_BLOCK)


def _cat_head(f, cx, cy, r):
    """Line-art cat head of radius r centred at (cx, cy)."""
    for sgn in (-1, 1):                        # ears, behind the head
        pts = [(cx + sgn * 0.92 * r, cy - 0.30 * r),
               (cx + sgn * 0.78 * r, cy - 1.30 * r),
               (cx + sgn * 0.18 * r, cy - 0.88 * r)]
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
        f.path(d, cx - r, cy - 1.35 * r, cx + r, cy, fill=T.PAPER,
               stroke=T.INK, sw=T.SW_BOX)
    f.circle(cx, cy, r, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX)
    for sgn in (-1, 1):
        f.circle(cx + sgn * 0.38 * r, cy - 0.08 * r, max(1.6, 0.12 * r),
                 fill=T.INK, stroke="none", sw=0)
    nose = [(cx - 0.13 * r, cy + 0.24 * r), (cx + 0.13 * r, cy + 0.24 * r),
            (cx, cy + 0.38 * r)]
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in nose) + " Z"
    f.path(d, cx - r, cy, cx + r, cy + r, fill=T.INK, stroke=T.INK, sw=1.0)
    if r >= 14:
        for sgn in (-1, 1):
            for dy in (0.36, 0.56):
                f.line(cx + sgn * 0.30 * r, cy + 0.42 * r,
                       cx + sgn * 1.25 * r, cy + dy * r - 0.1 * r,
                       stroke=T.INK, sw=1.2, cap="round")


def _cat_tile(f, cx, cy, s):
    _tile(f, cx, cy, s)
    _cat_head(f, cx, cy + 0.08 * s, 0.24 * s)


def _bicycle_tile(f, cx, cy, s):
    _tile(f, cx, cy, s)
    u = s / 92.0
    wy, wr = cy + 22 * u, 13 * u
    rx, fx = cx - 21 * u, cx + 21 * u          # rear, front hubs
    seat, bar = (cx - 8 * u, cy + 2 * u), (cx + 14 * u, cy + 2 * u)
    crank = (cx + 1 * u, wy)
    for hx in (rx, fx):
        f.circle(hx, wy, wr, fill="none", stroke=T.INK, sw=T.SW_BOX)
    f.polyline([(rx, wy), seat, bar, (fx, wy)], stroke=T.INK, sw=T.SW_BOX)
    f.polyline([seat, crank, (rx, wy)], stroke=T.INK, sw=T.SW_BOX)
    f.polyline([crank, bar], stroke=T.INK, sw=T.SW_BOX)
    f.polyline([bar, (cx + 13 * u, cy - 6 * u), (cx + 19 * u, cy - 7 * u)],
               stroke=T.INK, sw=T.SW_BOX)
    # the cat: body on the saddle, paw to the handlebar, head on top
    f.raw(f'<ellipse cx="{cx - 6 * u:.2f}" cy="{cy - 8 * u:.2f}" '
          f'rx="{8 * u:.2f}" ry="{10 * u:.2f}" fill="{T.PAPER}" '
          f'stroke="{T.INK}" stroke-width="{T.SW_BOX}"/>',
          cx - 14 * u, cy - 18 * u, cx + 2 * u, cy + 2 * u)
    f.line(cx - 1 * u, cy - 9 * u, cx + 13 * u, cy - 5 * u, stroke=T.INK,
           sw=T.SW_BOX, cap="round")
    f.path(f"M {cx - 14 * u:.2f} {cy - 4 * u:.2f} "
           f"Q {cx - 26 * u:.2f} {cy - 6 * u:.2f} "
           f"{cx - 24 * u:.2f} {cy - 18 * u:.2f}",
           cx - 26 * u, cy - 18 * u, cx - 14 * u, cy - 4 * u,
           stroke=T.INK, sw=T.SW_BOX)
    _cat_head(f, cx - 3 * u, cy - 25 * u, 9 * u)


def fig_generative():
    f = Figure()
    S = 92.0
    Y0, Y1 = 0.0, S + 14.0
    desc = "“a cat on a bicycle”"
    xL = 0.0                                   # left column right edge
    mw = max(bw("discriminative model"), bw("generative model"))
    xm = xL + ARM + 8 + mw / 2
    for y, lab in [(Y0, "discriminative model"), (Y1, "generative model")]:
        m = f.block(xm, y, lab, role="stream", min_w=mw)
        f.arrow(xL + 8, y, m[0] - TIP, y, stroke=T.INK)
        f.arrow(m[0] + m[2] + 1, y, m[0] + m[2] + ARM, y, stroke=T.INK)
    xR = m[0] + m[2] + ARM + 8                 # right column left edge
    _cat_tile(f, xL - S / 2, Y0, S)
    f.text(xR, Y0, [Span("“cat”", "m")], size=T.FS_LABEL, anchor="start")
    f.text(xL, Y1, [Span(desc, "m")], size=T.FS_LABEL, anchor="end")
    _bicycle_tile(f, xR + S / 2, Y1, S)
    f.save("intro-generative", out_dir=OUT,
           desc="A discriminative model maps data to a label; a generative "
                "model maps a description to data.")


# --------------------------------------------------------------------------- #
# intro-scaling — POINT (schematic): for each model size, loss falls with     #
# compute and then levels off; bigger models level off lower.  The best      #
# loss at each budget, the lower envelope, is a straight line in log-log.    #
# Curves follow L = A N^-a + B (C/N)^-b, whose envelope is an exact power    #
# law; the constants are chosen for legibility, not fitted to anything.      #
# --------------------------------------------------------------------------- #

def fig_scaling():
    import numpy as np
    f = Figure()
    PW, PH = 400.0, 240.0
    C0, C1, LMAX = 18.0, 24.0, 1.0             # log10 compute; log10 loss
    A_, AL, B_, BE = 1.80, 0.16, 7.00, 0.60

    def loglos(c, u):
        return np.log10(10 ** (A_ - AL * u) + 10 ** (B_ - BE * (c - u)))

    px = lambda c: (c - C0) / (C1 - C0) * PW
    py = lambda l: PH - l / LMAX * PH

    # axes (black) with power-of-ten ticks
    f.line(0, 0, 0, PH, stroke=T.INK, sw=T.SW_HAIR)
    f.line(0, PH, PW, PH, stroke=T.INK, sw=T.SW_HAIR)
    for c in range(18, 25, 2):
        f.line(px(c), PH, px(c), PH + 5, stroke=T.INK, sw=T.SW_HAIR)
        runs(f, px(c), PH + 20, ["10", ("^", str(c))], anchor="middle")
    for l, e in [(0.0, "0"), (1.0, "1")]:
        f.line(-5, py(l), 0, py(l), stroke=T.INK, sw=T.SW_HAIR)
        runs(f, -10, py(l), ["10", ("^", e)], anchor="end")
    f.text(PW / 2, PH + 46, "training compute", size=T.FS_LABEL)
    f.text(-10, -24, "loss", size=T.FS_LABEL, anchor="start")

    cs = np.linspace(C0, C1, 241)
    sizes = [8, 9, 10, 11]
    for k, u in enumerate(sizes):
        ls = loglos(cs, u)
        keep = ls <= LMAX
        pts = [(px(c), py(l)) for c, l in zip(cs[keep], ls[keep])]
        acc = [T.BLUE, T.ORANGE, T.GREEN, T.PURPLE][k]
        f.polyline(pts, stroke=acc.base, sw=T.SW_LINE)
        runs(f, PW + 10, pts[-1][1], ["10", ("^", str(u)), " parameters"],
             color=acc.dark)
    us = np.linspace(4, 16, 1201)
    env = np.array([loglos(c, us).min() for c in cs])
    keep = env <= LMAX
    f.polyline([(px(c), py(l)) for c, l in zip(cs[keep], env[keep])],
               stroke=T.INK, sw=T.SW_LINE, dash=T.DASH_SOFT)
    # label the envelope from below, in the empty lower-left wedge
    lx = 12.0
    lw = max(measure(s, T.FS_SMALL).width for s in ["best model",
                                                     "for each budget"])
    c_end = C0 + (lx + lw) / PW * (C1 - C0)
    ytop = py(float(np.interp(c_end, cs, env))) + 12
    f.text(lx, ytop + 9, "best model", size=T.FS_SMALL, anchor="start")
    f.text(lx, ytop + 29, "for each budget", size=T.FS_SMALL,
           anchor="start")
    f.save("intro-scaling", out_dir=OUT,
           desc="Schematic: loss against training compute for models of "
                "increasing size, and their lower envelope.")

def main():
    os.makedirs(OUT, exist_ok=True)
    fig_ml_loop()
    fig_supervised_learning()
    fig_data_collection()
    fig_rl_environment()
    fig_wake_word()
    fig_speech()
    fig_next_token()
    fig_pretrain_adapt()
    fig_generative()
    fig_scaling()
    print("wrote chapter-1 figures to", OUT)


if __name__ == "__main__":
    main()
