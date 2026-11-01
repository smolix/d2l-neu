#!/usr/bin/env python3
"""Generate the illustrative figures for the "Recurrent Neural Networks"
chapter (``chapter_recurrent-neural-networks``) in the one shared house style
defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Figures that show a *computed* result (e.g. a loss curve
comparing truncated-BPTT horizons, a hidden-state trajectory) should use real
numerical computation so the pictures are exact, not sketches; purely
schematic figures (unrolled-graph diagrams, gate wiring, ...) use
``set_aspect("equal")`` and the shared drawing helpers
(``arrow``/``vlabel``/``clean_axes``/``axis_cross``/``right_angle``).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_rnn_figures.py

All figures are written to ``img/mdl-rnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
TEAL = fl.T.TEAL.base   # hidden-state role (guide §4.3)

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


# =========================================================================== #
# "Working with Sequences": autoregression vs. latent autoregression.         #
# =========================================================================== #

def fig_ar_vs_latent():
    """Two schematics of how a sequence model makes its input length fixed.

    (left) autoregression: a sliding window of the last m observations feeds
    the prediction; older observations are discarded.
    (right) latent autoregression: a recurrent state h_t summarizes the whole
    past in fixed size and is carried forward step by step.

    Equal aspect on both; y-spans match and gridspec width_ratios match the
    x-spans, so the panels render at the same height and share one scale.
    """
    fig = plt.figure(figsize=(10.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[8.9, 5.7], wspace=0.12)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    LBL = 14        # in-figure math labels
    TTL = 15        # panel titles (functional, not "(a)/(b)")

    # ------------------------------------------------------------------ #
    # (left) fixed window / autoregression                               #
    # ------------------------------------------------------------------ #
    # a smooth deterministic signal sampled at integer time steps 1..8
    t = np.arange(1, 9)
    y = 1.55 + 0.72 * np.sin(0.85 * t + 0.4)
    win_idx = [5, 6, 7]                 # the window: last m observed points
    pred_t = 8                          # the step we predict

    # faint signal line through the observed points 1..7
    axa.plot(t[:7], y[:7], "-", color=LIGHT, lw=1.6, zorder=1)
    # observed points: earlier ones faded, windowed ones solid blue
    for ti, yi in zip(t[:7], y[:7]):
        faded = ti < win_idx[0]
        axa.plot(ti, yi, "o", color=BLUE, ms=8, zorder=4,
                 alpha=0.30 if faded else 1.0)

    # shade the window over the last m observations
    x0, x1 = win_idx[0] - 0.45, win_idx[-1] + 0.45
    ylo, yhi = 0.55, 2.75
    axa.add_patch(fl.Rectangle((x0, ylo), x1 - x0, yhi - ylo, facecolor=BLUE,
                               alpha=0.10, edgecolor=BLUE, lw=1.4, zorder=2))
    axa.text((x0 + x1) / 2, yhi + 0.16,
             r"window of length $m$", color="black", ha="center",
             va="bottom", fontsize=LBL)
    axa.text((x0 + x1) / 2, ylo - 0.14,
             r"$x_{t-m},\ \ldots,\ x_{t-1}$", color="black", ha="center",
             va="top", fontsize=LBL)

    # the prediction: an open marker at the next step, arrow from the window
    yp = y[pred_t - 1]
    axa.plot(pred_t, yp, "o", mfc="white", mec=GREEN, mew=2.2, ms=11, zorder=5)
    fl.arrow(axa, (x1, (ylo + yhi) / 2 + 0.15), (pred_t - 0.22, yp),
             color=GREEN, lw=2.0)
    axa.text(pred_t + 0.12, yp + 0.30, r"$\hat{x}_t$", color=GREEN, ha="left",
             va="bottom", fontsize=LBL + 1)

    # a light time axis along the bottom, below the window's label
    fl.arrow(axa, (0.3, -0.25), (8.5, -0.25), color="black", lw=1.0)
    axa.text(8.55, -0.25, r"$t$", color="black", ha="left", va="center",
             fontsize=LBL)

    axa.set_title("autoregression", fontsize=TTL, color="black")
    axa.set_xlim(0.1, 9.0)
    axa.set_ylim(-0.45, 3.45)
    axa.set_aspect("equal")
    axa.axis("off")

    # ------------------------------------------------------------------ #
    # (right) latent state / latent autoregression                       #
    # ------------------------------------------------------------------ #
    r = 0.30
    xs = [1.2, 2.6, 4.0]               # three time steps
    y_in, y_h, y_out = 0.55, 1.75, 2.95

    def node(ax, cx, cy, text, edge):
        ax.add_patch(plt.Circle((cx, cy), r, facecolor="white", edgecolor=edge,
                                 lw=2.0, zorder=4))
        ax.text(cx, cy, text, color="black", ha="center", va="center",
                fontsize=LBL, zorder=5)

    def link(ax, p, q, color="black", lw=1.8):
        p = np.asarray(p, float); q = np.asarray(q, float)
        d = q - p
        d = d / np.linalg.norm(d)
        fl.arrow(ax, p + r * d, q - r * d, color=color, lw=lw)

    labs_x = [r"$x_1$", r"$x_2$", r"$x_3$"]
    labs_h = [r"$\mathbf{h}_1$", r"$\mathbf{h}_2$", r"$\mathbf{h}_3$"]
    for cx, lx, lh in zip(xs, labs_x, labs_h):
        node(axb, cx, y_in, lx, BLUE)
        node(axb, cx, y_h, lh, TEAL)
        link(axb, (cx, y_in), (cx, y_h))          # x_i -> h_i

    for c0, c1 in zip(xs[:-1], xs[1:]):
        link(axb, (c0, y_h), (c1, y_h))            # recurrence h_i -> h_{i+1}

    # incoming state from earlier history (into h_1) and continuation after h_3
    fl.arrow(axb, (0.25, y_h), (xs[0] - r, y_h), color="black", lw=1.8)
    axb.text(0.18, y_h, r"$\cdots$", color="black", ha="right", va="center",
             fontsize=LBL)
    fl.arrow(axb, (xs[-1] + r, y_h), (xs[-1] + 0.85, y_h), color="black", lw=1.8)
    axb.text(xs[-1] + 0.95, y_h, r"$\cdots$", color="black", ha="left",
             va="center", fontsize=LBL)

    # prediction from the current state
    node(axb, xs[-1], y_out, r"$\hat{x}_4$", GREEN)
    link(axb, (xs[-1], y_h), (xs[-1], y_out), color=GREEN, lw=2.0)

    axb.set_title("latent autoregression", fontsize=TTL, color="black")
    axb.set_xlim(-0.35, 5.35)
    axb.set_ylim(-0.45, 3.45)
    axb.set_aspect("equal")
    axb.axis("off")

    fl.save(fig, "mdl-rnn-ar-vs-latent")


# =========================================================================== #
# "From Text to Tokens": granularity spectrum, BPE merge tree, and the        #
# pre-tokenization pipeline.                                                  #
# =========================================================================== #

def fig_rnn_granularity_spectrum():
    """Sequence length vs vocabulary size: one smooth trade-off curve with the
    three token granularities marked on it.  The BPE segment of the curve is
    anchored at ratios measured on *The Time Machine* (see text-sequence.md);
    the right end flattens toward the ~1 token/word plateau of English text.
    """
    from scipy.interpolate import PchipInterpolator

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # Anchor points: (vocab size, sequence length relative to raw bytes).
    # 256..4096 measured with the section's own tokenizer; the tail flattens
    # toward the ~0.19 plateau (about one token per English word).
    xs = np.array([256, 512, 1024, 2048, 4096, 5e4, 2e5, 8e5])
    ys = np.array([1.0, 0.459, 0.348, 0.275, 0.221, 0.20, 0.193, 0.19])
    grid = np.geomspace(256, 8e5, 400)
    curve = PchipInterpolator(np.log(xs), ys)(np.log(grid))

    ax.plot(grid, curve, color=BLUE, lw=2.5, zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(130, 2.6e6)
    ax.set_ylim(0, 1.14)
    ax.set_xlabel('vocabulary size', fontsize=14, color='black')
    ax.set_ylabel('sequence length (relative to bytes)', fontsize=13,
                  color='black')
    ax.tick_params(labelsize=12, colors='black')

    # The three granularities, as points on the curve.
    ax.plot([256], [1.0], 'o', color=BLUE, ms=8, zorder=4)
    ax.annotate('bytes (vocab 256)', xy=(256, 1.0), xytext=(400, 1.01),
                fontsize=13, color='black', ha='left', va='center')
    ax.annotate('long sequences,\nnever out of vocabulary',
                xy=(430, 0.88), fontsize=13, color='black', ha='left', va='top')

    band = (grid >= 512) & (grid <= 2e5)
    ax.fill_between(grid[band], curve[band] - 0.06, curve[band] + 0.06,
                    color=BLUE, alpha=0.15, lw=0, zorder=2)
    ax.annotate('subword (BPE):\nchoose the operating point',
                xy=(9e3, 0.25), xytext=(4.5e3, 0.52), fontsize=13,
                color='black', ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.1))

    ax.plot([8e5], [0.19], 's', color=ORANGE, ms=8, zorder=4)
    ax.annotate('words', xy=(8e5, 0.19), xytext=(8e5, 0.30), fontsize=13,
                color='black', ha='center', va='bottom')
    ax.annotate('short sequences, but a\nfixed list: rare words\n'
                'are out of vocabulary',
                xy=(6.5e5, 0.68), fontsize=13, color='black', ha='center',
                va='top')
    fl.save(fig, 'mdl-rnn-granularity-spectrum')


def _tok_box(ax, x, y, text, w=0.72, h=0.5, fc=fl.T.BLUE.tint, ec=fl.T.BLUE.base,
             fontsize=14, mono=True):
    """A token box centered at (x, y); returns (x, y) for line anchoring."""
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                               edgecolor=ec, lw=1.1, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            family='monospace' if mono else None, color='black', zorder=4)
    return x, y


def fig_rnn_merge_tree():
    """The three merges learned on the toy corpus 'hug pug pun bun hugs hug
    hug pug', drawn as a tree: u+g -> ug, h+ug -> hug, hug+space -> 'hug '.
    Leaves are byte tokens; each internal node carries its merge rank."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    leaf_fc, node_fc = fl.T.BLUE.tint, fl.T.ORANGE.tint   # token tints
    # Leaves (byte tokens).
    lx = {'h': 0.0, 'u': 1.0, 'g': 2.0, 'sp': 3.0}
    for name, x in lx.items():
        _tok_box(ax, x, 0.0, '␣' if name == 'sp' else name, fc=leaf_fc)

    # Internal nodes: (x, y, label, rank, children x/y).
    nodes = [
        (1.5, 1.0, 'ug', 1, [(1.0, 0.0), (2.0, 0.0)]),
        (0.75, 2.0, 'hug', 2, [(0.0, 0.0), (1.5, 1.0)]),
        (1.875, 3.0, 'hug␣', 3, [(0.75, 2.0), (3.0, 0.0)]),
    ]
    for x, y, label, rank, children in nodes:
        w = 0.72 + 0.24 * (len(label) - 2)
        for cx, cy in children:
            ax.plot([x, cx], [y - 0.25, cy + 0.25], color=GRAY, lw=1.4,
                    zorder=1)
        _tok_box(ax, x, y, label, w=w, fc=node_fc)
        ax.add_patch(plt.Circle((x + w / 2 + 0.28, y), 0.19, facecolor='white',
                                edgecolor=ORANGE, lw=1.4, zorder=4))
        ax.text(x + w / 2 + 0.28, y, str(rank), ha='center', va='center',
                fontsize=12, color='black', zorder=5)

    ax.text(4.6, 3.0, 'merge rank', fontsize=13, color='black', ha='left',
            va='center')
    ax.annotate('', xy=(2.42, 3.0), xytext=(4.5, 3.0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.1))

    # How the learned tree tokenizes new words.
    ax.text(4.6, 1.6, 'new text:', fontsize=13, color='black', ha='left')
    ax.text(4.6, 1.0, 'hugs → hug | s', fontsize=14, family='monospace',
            color='black', ha='left')
    ax.text(4.6, 0.4, 'bun → b | u | n', fontsize=14, family='monospace',
            color='black', ha='left')

    fl.clean_axes(ax, lim=((-0.7, 7.6), (-0.55, 3.55)), hide=True)
    fl.save(fig, 'mdl-rnn-merge-tree')


def fig_rnn_pretokenization_pipeline():
    """Text -> regex chunks -> BPE within chunks, on a real sentence.  Chunk
    splits and sub-tokens are the actual output of the section's GPT-2-pattern
    tokenizer (vocab 1,024) trained on The Time Machine."""
    # (chunk, [sub-tokens]) pairs; U+2423 marks leading spaces.
    chunks = [
        ('The', ['The']),
        (' traveller', [' t', 'raveller']),
        ("'s", ["'s"]),
        (' clock', [' c', 'lock']),
        (' struck', [' stru', 'ck']),
        (' 12', [' ', '1', '2']),
        (',', [',']),
        ('345', ['3', '4', '5']),
        (' times', [' t', 'imes']),
        ('.', ['.']),
    ]
    sentence = ''.join(c for c, _ in chunks)

    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    char_w, gap, h = 0.155, 0.20, 0.62
    y1, y2, y3 = 2.9, 1.55, 0.2

    def digits(s):
        return any(ch.isdigit() for ch in s)

    # Row 1: the raw string in one long box.
    total = sum(len(c) for c, _ in chunks) * char_w + (len(chunks) - 1) * gap
    x0 = 0.0
    ax.add_patch(plt.Rectangle((x0, y1 - h / 2), total, h, facecolor='white',
                               edgecolor='black', lw=1.1))
    ax.text(x0 + total / 2, y1, sentence, ha='center', va='center',
            fontsize=13, family='monospace', color='black')

    # Rows 2 and 3: chunk boxes, and sub-token boxes aligned beneath them.
    x = x0
    for chunk, subs in chunks:
        w = len(chunk) * char_w
        fc = fl.T.ORANGE.tint if digits(chunk) else fl.T.BLUE.tint
        shown = chunk.replace(' ', '␣')
        ax.add_patch(plt.Rectangle((x, y2 - h / 2), w, h, facecolor=fc,
                                   edgecolor='black', lw=1.1))
        ax.text(x + w / 2, y2, shown, ha='center', va='center', fontsize=13,
                family='monospace', color='black')
        sx = x
        for sub in subs:
            sw = max(len(sub), 1) * char_w * (w / (len(chunk) * char_w))
            sw = len(sub) / len(chunk) * w
            ax.add_patch(plt.Rectangle((sx, y3 - h / 2), sw, h,
                                       facecolor='white', edgecolor=GRAY,
                                       lw=1.1))
            ax.text(sx + sw / 2, y3, sub.replace(' ', '␣'), ha='center',
                    va='center', fontsize=12, family='monospace',
                    color='black')
            sx += sw
        x += w + gap

    # Stage arrows + labels on the left margin.
    ax.annotate('', xy=(-0.35, y2 + h / 2 - 0.05), xytext=(-0.35, y1 - h / 2 + 0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    ax.annotate('', xy=(-0.35, y3 + h / 2 - 0.05), xytext=(-0.35, y2 - h / 2 + 0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    ax.text(total / 2, (y1 + y2) / 2, 'split with the pre-tokenization regex',
            ha='center', va='center', fontsize=13, color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.text(total / 2, (y2 + y3) / 2, 'BPE merges, applied within each chunk',
            ha='center', va='center', fontsize=13, color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

    fl.clean_axes(ax, lim=((-0.7, total + 0.25), (y3 - 0.55, y1 + 0.55)),
                  hide=True, equal=False)
    fl.save(fig, 'mdl-rnn-pretokenization-pipeline')


# =========================================================================== #
# "Language Models" (sec_language-model): partitioning a BPE token-id        #
# stream into overlapping input/target subsequences.                         #
#                                                                              #
# One restyled carryover, written under a new house-style name so the old    #
# hand-drawn SVG (``img/lang-model-data.svg``) is left untouched on disk.    #
# =========================================================================== #

# The first 8 tokens of the corpus, from its title line "The Time Machine, by
# H. G. Wells", under d2l.TimeMachine's BPE-1024 tokenizer (trained on the raw
# text with the GPT-2 pattern); leading spaces marked "␣".
_PARTITION_TOKENS = ["The", "␣Time", "␣Machine", ",", "␣by", "␣H", ".", "␣G"]
_PARTITION_N = 5             # window length T (num_steps)
_PARTITION_STARTS = [1, 2]   # the first two windows: they overlap in T - 1


def _token_cells(ax, edges, y, h, lo, hi, fc, ec):
    """Token boxes lo..hi (1-based, inclusive) of the stream at height y:
    monospace text in boxes whose width grows with the token length."""
    for i in range(lo - 1, hi):
        x0, x1 = edges[i], edges[i + 1]
        ax.add_patch(plt.Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=fc,
                                   edgecolor=ec, linewidth=1.2, zorder=2))
        ax.text((x0 + x1) / 2, y, _PARTITION_TOKENS[i], ha="center",
                va="center", fontsize=14, family="monospace", color="black",
                zorder=4)


def fig_partitioning():
    """The first two (input, target) pairs cut from one BPE token stream.

    Top row: the stream x_1, x_2, ... with its position indices.  Below it,
    one pair of rows per window: the input window (blue) of T = 5 tokens and
    its target (orange), the same window shifted forward by one token.  The
    columns align with the stream, so the one-token shift between input and
    target, and the overlap of consecutive windows in all but one token, can
    be read off directly."""
    widths = [0.19 * len(t) + 0.55 for t in _PARTITION_TOKENS]
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    n, T = len(_PARTITION_TOKENS), _PARTITION_N

    h, step = 0.62, 0.82                    # box height, row pitch
    y_stream, y_idx = 0.0, -0.55
    rows = [y_idx - 0.75 - k * step - (0.35 if k >= 2 else 0.0)
            for k in range(4)]              # input 1, target 1, input 2, ...
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # the stream, its position indices, and the row labels (black)
    _token_cells(ax, edges, y_stream, h, 1, n, "white", GRAY)
    for i in range(n):
        ax.text((edges[i] + edges[i + 1]) / 2, y_idx, rf"$x_{{{i + 1}}}$",
                ha="center", va="center", fontsize=14, color="black")
    ax.text(-0.3, y_stream, "stream", ha="right", va="center", fontsize=14,
            color="black")

    for k, t in enumerate(_PARTITION_STARTS):
        y_in, y_tgt = rows[2 * k], rows[2 * k + 1]
        _token_cells(ax, edges, y_in, h, t, t + T - 1, fl.T.BLUE.tint, BLUE)
        _token_cells(ax, edges, y_tgt, h, t + 1, t + T, fl.T.ORANGE.tint,
                     ORANGE)
        ax.text(-0.3, y_in, f"input {k + 1}", ha="right", va="center",
                fontsize=14, color="black")
        ax.text(-0.3, y_tgt, f"target {k + 1}", ha="right", va="center",
                fontsize=14, color="black")

    ax.set_xlim(-2.0, edges[-1] + 0.1)
    ax.set_ylim(rows[-1] - h / 2 - 0.1, y_stream + h / 2 + 0.1)
    fl.save(fig, "mdl-rnn-partitioning")


# =========================================================================== #
# "Recurrent Neural Networks" (sec_rnn): the unrolled RNN and the             #
# shift-by-one language-model training picture.                              #
#                                                                              #
# Two restyled carryovers, written under new house-style names so the old    #
# hand-drawn SVGs (``img/rnn.svg``, ``img/rnn-train.svg``) are left           #
# untouched.                                                                  #
# =========================================================================== #

def _box(ax, cx, cy, w, h, text, color, fontsize=15, text_color="black",
         weight="bold"):
    """A rounded state box (faint fill + solid coloured edge) with centred text,
    matching the house schematic style (cf. gen_mdl_mlp_figures.fig_mlp_arch)."""
    x, y = cx - w / 2, cy - h / 2
    for fc, a in [(color, 0.12), ("none", 1.0)]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)


def fig_unfolded():
    """The RNN unrolled over three adjacent time steps.  Bottom row: inputs
    X (blue).  Middle row: hidden states H (green), linked left-to-right by the
    recurrence.  Top row: outputs O (orange).  The weight labels W_xh, W_hh,
    W_hq repeat identically at every step to show that one set of parameters is
    shared across time; dashed stubs at the far left/right show the chain
    continuing beyond the window."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.4, 5.2, 8.0]                # three time steps
    y_in, y_hid, y_out = 0.9, 3.0, 5.1  # three rows
    bw, bh = 1.15, 0.9
    subs = ["t-1", "t", "t+1"]

    # left-hand row labels (black)
    ax.text(0.35, y_out, "outputs", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.35, y_hid, "hidden\nstate", ha="left", va="center", fontsize=13,
            color="black", linespacing=1.15)
    ax.text(0.35, y_in, "inputs", ha="left", va="center", fontsize=13,
            color="black")

    half = bh / 2
    for cx, s in zip(xs, subs):
        _box(ax, cx, y_in, bw, bh, rf"$\mathbf{{X}}_{{{s}}}$", BLUE)
        _box(ax, cx, y_hid, bw, bh, rf"$\mathbf{{H}}_{{{s}}}$", TEAL)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{{{s}}}$", ORANGE)
        # input -> hidden, and hidden -> output (same weights every step)
        fl.arrow(ax, (cx, y_in + half), (cx, y_hid - half), color=GRAY, lw=2.0,
                 mut=15)
        fl.arrow(ax, (cx, y_hid + half), (cx, y_out - half), color=GRAY, lw=2.0,
                 mut=15)
        ax.text(cx + 0.14, (y_in + y_hid) / 2, r"$\mathbf{W}_\mathrm{xh}$",
                ha="left", va="center", fontsize=14, color="black")
        ax.text(cx + 0.14, (y_hid + y_out) / 2, r"$\mathbf{W}_\mathrm{hq}$",
                ha="left", va="center", fontsize=14, color="black")

    # recurrent links between hidden states, labelled W_hh above the shaft
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (cx0 + bw / 2, y_hid), (cx1 - bw / 2, y_hid), color=GRAY,
                 lw=2.0, mut=15)
        ax.text((cx0 + cx1) / 2, y_hid + 0.24, r"$\mathbf{W}_\mathrm{hh}$",
                ha="center", va="bottom", fontsize=14, color="black")

    # dashed continuation stubs, so the chain reads as unbounded in time
    fl.arrow(ax, (xs[0] - 1.0, y_hid), (xs[0] - bw / 2, y_hid), color=GRAY,
             lw=1.6, ls="--", mut=13)
    fl.arrow(ax, (xs[-1] + bw / 2, y_hid), (xs[-1] + 1.0, y_hid), color=GRAY,
             lw=1.6, ls="--", mut=13)

    ax.set_xlim(0.2, 9.6)
    ax.set_ylim(0.2, 5.8)
    fl.save(fig, "mdl-rnn-unfolded")


def fig_lm_shift():
    """RNN language model trained by teacher forcing over BPE subword tokens.
    The input tokens feed the recurrence; each hidden state emits a next-token
    distribution O_t; the target sequence is the input shifted forward by one
    token, so the per-step cross-entropy trains the model to predict the next
    token.  The tokens are what d2l.TimeMachine's BPE-1024 tokenizer produces
    for the corpus phrase "thing that struck me": "struck" is not in the
    vocabulary and becomes the two tokens "␣stru" and "ck"."""
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.5, 4.9, 7.3, 9.7]                       # four time steps
    y_in, y_hid, y_out, y_tgt = 0.9, 3.0, 5.1, 6.8  # four rows
    bw, bh = 1.75, 0.9
    inp = ["␣thing", "␣that", "␣stru", "ck"]
    tgt = ["␣that", "␣stru", "ck", "␣me"]

    # left-hand row labels (black)
    ax.text(0.15, y_tgt, "targets", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.15, y_out, "outputs", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.15, y_hid, "hidden\nstate", ha="left", va="center", fontsize=13,
            color="black", linespacing=1.15)
    ax.text(0.15, y_in, "inputs", ha="left", va="center", fontsize=13,
            color="black")

    half = bh / 2
    for i, cx in enumerate(xs):
        # token boxes: tokens are literal data, so monospace (style guide 5.1)
        for y, tok, color in ((y_in, inp[i], BLUE), (y_tgt, tgt[i], GRAY)):
            _box(ax, cx, y, bw, bh, "", color)
            ax.text(cx, y, tok, ha="center", va="center", fontsize=15,
                    family="monospace", color="black")
        _box(ax, cx, y_hid, bw, bh, rf"$\mathbf{{H}}_{{{i+1}}}$", TEAL)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{{{i+1}}}$", ORANGE)
        fl.arrow(ax, (cx, y_in + half), (cx, y_hid - half), color=GRAY, lw=2.0,
                 mut=15)
        fl.arrow(ax, (cx, y_hid + half), (cx, y_out - half), color=GRAY, lw=2.0,
                 mut=15)
        # compare prediction against the target above it (cross-entropy loss)
        fl.arrow(ax, (cx, y_out + half), (cx, y_tgt - half), color=GRAY,
                 lw=1.6, ls="--", mut=13)

    # recurrent links between hidden states
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (cx0 + bw / 2, y_hid), (cx1 - bw / 2, y_hid), color=GRAY,
                 lw=2.0, mut=15)

    # name the per-step loss once, beside the first prediction->target link
    ax.text(xs[0] + 0.18, (y_out + y_tgt) / 2, "loss", ha="left", va="center",
            fontsize=13, color="black")

    # brace under the two subword tokens of "struck"
    yb = y_in - 0.72
    ax.plot([xs[2] - bw / 2, xs[3] + bw / 2], [yb, yb], color=GRAY, lw=1.3)
    for cx in (xs[2], xs[3]):
        ax.plot([cx, cx], [yb, y_in - half - 0.04], color=GRAY, lw=1.0)
    ax.text((xs[2] + xs[3]) / 2, yb - 0.25, 'one word, "struck"',
            ha="center", va="top", fontsize=13, color="black")

    ax.set_xlim(0.0, 10.75)
    ax.set_ylim(-0.85, 7.5)
    fl.save(fig, "mdl-rnn-lm-shift")


# =========================================================================== #
# "Backpropagation Through Time" (sec_bptt): the unrolled linear RNN with its  #
# forward and backward passes, and full BPTT vs. regular truncation, where the #
# state is carried forward across a segment boundary but the gradient is not. #
#                                                                              #
# Restyled carryovers of the hand-drawn ``img/rnn-bptt.svg`` and              #
# ``img/truncated-bptt.svg`` (left untouched on disk).  The old figure's      #
# randomized-truncation row is dropped.                                       #
# =========================================================================== #

def _segment(ax, x0, x1, yc, h=0.78):
    """A faint rounded rectangle marking one backpropagation *segment*."""
    ax.add_patch(FancyBboxPatch(
        (x0, yc - h / 2), x1 - x0, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=GRAY, alpha=0.12, zorder=0))


def _lanes(ax, xs, yc, back, r=0.09, off=0.13):
    """Hidden-state nodes at ``xs``: grey forward arrows on the upper lane
    between every pair, red backward (gradient) arrows on the lower lane
    between the pairs whose index is in ``back``."""
    red = fl.T.RED.base
    for x in xs:
        ax.add_patch(plt.Circle((x, yc), r, facecolor=TEAL, edgecolor="none",
                                zorder=3))
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        fl.arrow(ax, (x0 + r + 0.03, yc + off), (x1 - r - 0.03, yc + off),
                 color=GRAY, lw=1.8, mut=12)
        if i in back:
            fl.arrow(ax, (x1 - r - 0.03, yc - off), (x0 + r + 0.03, yc - off),
                     color=red, lw=2.0, mut=12)


def fig_truncated_bptt():
    """Two ways of computing the gradient across a token sequence.  Both rows
    run the same forward pass (grey): the hidden state is carried from token to
    token along the whole sequence.  TOP: full backpropagation through time --
    the gradient (red) flows back along the entire chain.  BOTTOM: regular
    truncation -- the sequence is cut into equal-length segments and the state
    is *detached* at every boundary: its value still crosses forward, but no
    gradient flows back across."""
    # BPE-1024 tokens of the corpus phrase "the thing that struck me most was"
    # (d2l.TimeMachine's tokenizer; "␣" marks a leading space)
    tokens = ["␣the", "␣thing", "␣that", "␣stru", "ck", "␣me", "␣most", "␣was"]
    n = len(tokens)
    seg_len = 4                              # regular-truncation segment length
    xs = np.linspace(2.35, 10.35, n)
    dx = xs[1] - xs[0]
    pad = 0.42 * dx                          # segment box overhang past the nodes

    y_tok, y_full, y_trunc, y_leg = 3.2, 2.2, 0.85, -0.25

    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    ax.set_aspect("equal")
    ax.axis("off")

    # token strip, shared by both rows; row labels on the left (black text)
    for x, tok in zip(xs, tokens):
        ax.text(x, y_tok, tok, ha="center", va="center", fontsize=14,
                family="monospace", color="black")
    for y, lab in [(y_full, "full"), (y_trunc, "truncated")]:
        ax.text(0.0, y, lab, ha="left", va="center", fontsize=14,
                color="black")

    # TOP: one segment spanning the whole sequence, gradient everywhere
    _segment(ax, xs[0] - pad, xs[-1] + pad, y_full)
    _lanes(ax, xs, y_full, back=range(n - 1))

    # BOTTOM: equal segments; the gradient stops at each boundary
    cuts = set(range(seg_len - 1, n - 1, seg_len))   # pair index at a boundary
    for s in range(0, n, seg_len):
        blk = xs[s:s + seg_len]
        _segment(ax, blk[0] - pad, blk[-1] + pad, y_trunc)
    _lanes(ax, xs, y_trunc, back=set(range(n - 1)) - cuts)
    for i in sorted(cuts):
        xb = (xs[i] + xs[i + 1]) / 2
        c, h = (xb, y_trunc - 0.13), 0.11           # a cross where the
        for sgn in (1, -1):                         # gradient would pass
            ax.plot([c[0] - h, c[0] + h], [c[1] - sgn * h, c[1] + sgn * h],
                    color=fl.T.RED.dark, lw=2.4, solid_capstyle="round",
                    zorder=4)
        ax.text(xb, y_trunc - 0.6, "detach", ha="center", va="center",
                fontsize=13, color="black")

    # legend: the two arrow colours
    for x0, color, lab in [(2.4, GRAY, "state (forward)"),
                           (6.2, fl.T.RED.base, "gradient (backward)")]:
        ends = [(x0, y_leg), (x0 + 0.6, y_leg)]   # red points backwards
        fl.arrow(ax, *(ends if color == GRAY else ends[::-1]), color=color,
                 lw=2.0, mut=13)
        ax.text(x0 + 0.75, y_leg, lab, ha="left", va="center", fontsize=13,
                color="black")

    ax.set_xlim(-0.1, xs[-1] + pad + 0.2)
    ax.set_ylim(y_leg - 0.35, y_tok + 0.35)
    fl.save(fig, "mdl-rnn-truncated-bptt")


def fig_rnn_bptt():
    """The linear RNN of the BPTT section unrolled over three time steps, with
    the forward pass (grey) and the backward pass (red) drawn side by side.

    Bottom row: inputs X_t (blue); middle row: hidden states H_t (teal); top
    rows: outputs O_t (orange) and the per-step losses l_t.  Forward arrows
    carry the weights W_xh, W_hh, W_hq (row-vector layout of the RNN section,
    H_t = X_t W_xh + H_{t-1} W_hh, O_t = H_t W_hq).  The red arrows run the
    other way: the gradient reaching H_t is the sum of the one coming down from
    its own output (through W_hq^T) and the one coming back from H_{t+1}
    (through W_hh^T), which is the recursion the section unrolls into powers of
    W_hh^T."""
    RED = fl.T.RED.base
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.3, 5.6, 8.9]                          # time steps t = 1, 2, 3
    y_in, y_hid, y_out, y_loss = 0.8, 2.95, 5.1, 6.85
    bw, bh = 1.2, 0.86
    lw_box = 0.9                                  # loss boxes are narrower
    off = 0.2                                     # forward/backward lane offset
    fs_lab = 15                                   # weight labels on the edges

    def fwd(p, q):
        fl.arrow(ax, p, q, color=GRAY, lw=2.0, mut=15)

    def bwd(p, q):
        fl.arrow(ax, p, q, color=RED, lw=2.2, mut=15)

    # left-hand row labels (black)
    for y, lab in [(y_loss, "losses"), (y_out, "outputs"),
                   (y_hid, "hidden\nstate"), (y_in, "inputs")]:
        ax.text(-1.1, y, lab, ha="left", va="center", fontsize=14,
                color="black", linespacing=1.15)

    for t, cx in enumerate(xs, start=1):
        _box(ax, cx, y_in, bw, bh, rf"$\mathbf{{X}}_{t}$", BLUE)
        _box(ax, cx, y_hid, bw, bh, rf"$\mathbf{{H}}_{t}$", TEAL)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{t}$", ORANGE)
        _box(ax, cx, y_loss, lw_box, 0.72, rf"$l_{t}$", GRAY)
        # forward: X -> H -> O -> l on the right lane
        fwd((cx + off, y_in + bh / 2), (cx + off, y_hid - bh / 2))
        fwd((cx + off, y_hid + bh / 2), (cx + off, y_out - bh / 2))
        fwd((cx + off, y_out + bh / 2), (cx + off, y_loss - 0.36))
        ax.text(cx + off + 0.14, (y_in + y_hid) / 2, r"$\mathbf{W}_\mathrm{xh}$",
                ha="left", va="center", fontsize=fs_lab, color="black")
        ax.text(cx + off + 0.14, (y_hid + y_out) / 2, r"$\mathbf{W}_\mathrm{hq}$",
                ha="left", va="center", fontsize=fs_lab, color="black")
        # backward: l -> O -> H on the left lane
        bwd((cx - off, y_loss - 0.36), (cx - off, y_out + bh / 2))
        bwd((cx - off, y_out - bh / 2), (cx - off, y_hid + bh / 2))
        ax.text(cx - off - 0.14, (y_hid + y_out) / 2, r"$\mathbf{W}_\mathrm{hq}^\top$",
                ha="right", va="center", fontsize=fs_lab, color=fl.T.RED.dark)

    # recurrence: forward on the upper lane, backward on the lower lane
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fwd((cx0 + bw / 2, y_hid + off), (cx1 - bw / 2, y_hid + off))
        bwd((cx1 - bw / 2, y_hid - off), (cx0 + bw / 2, y_hid - off))
        xm = (cx0 + cx1) / 2
        ax.text(xm, y_hid + off + 0.12, r"$\mathbf{W}_\mathrm{hh}$", ha="center",
                va="bottom", fontsize=fs_lab, color="black")
        ax.text(xm, y_hid - off - 0.12, r"$\mathbf{W}_\mathrm{hh}^\top$", ha="center",
                va="top", fontsize=fs_lab, color=fl.T.RED.dark)

    # the initial state enters from the left (forward only)
    fl.arrow(ax, (xs[0] - 1.15, y_hid + off), (xs[0] - bw / 2, y_hid + off),
             color=GRAY, lw=1.6, ls="--", mut=13)
    ax.text(xs[0] - 1.2, y_hid + off, r"$\mathbf{H}_0$", ha="right",
            va="center", fontsize=fs_lab, color="black")

    # legend: the two arrow colours
    y_leg = -0.25
    for x0, color, lab in [(3.0, GRAY, "forward pass"),
                           (6.3, RED, "backward pass (gradient)")]:
        ends = [(x0, y_leg), (x0 + 0.7, y_leg)]   # red points backwards
        fl.arrow(ax, *(ends if color == GRAY else ends[::-1]), color=color,
                 lw=2.2, mut=15)
        ax.text(x0 + 0.85, y_leg, lab, ha="left", va="center", fontsize=14,
                color="black")

    ax.set_xlim(-1.2, xs[-1] + bw / 2 + 0.9)
    ax.set_ylim(y_leg - 0.4, y_loss + 0.5)
    fl.save(fig, "mdl-rnn-bptt")


# =========================================================================== #
# "Decoding and Generation" (sec_decoding): the worked beam-search tree       #
# and the maximization-vs-sampling decoding task map.                        #
#                                                                              #
# The beam-search tree is restyled from the retired hand-drawn                #
# ``img/beam-search.svg`` of the old ``chapter_recurrent-modern/beam-         #
# search.md``; the task map is new.                                          #
# =========================================================================== #

def fig_beam_tree():
    """Beam search with beam size 2 over the vocabulary {A, B, C, D, E}.

    Columns are generation steps t = 1, 2, 3.  Only kept (blue) prefixes are
    expanded, each over all five tokens; of the 2x5 candidates per step the
    two best survive.  Kept path: A, C -> AB, CE -> ABD, CED.
    """
    tokens = ["A", "B", "C", "D", "E"]
    x_root, x1, x2, x3 = 0.0, 2.1, 4.2, 6.3
    s = 0.62                                   # vertical slot spacing
    r = 0.24                                   # node radius

    # ten-slot grid used by the 2x5 candidate columns (t = 2 and t = 3);
    # a slightly wider gap separates the two parents' expansion groups
    ggap = 0.34
    rows10 = [(4.5 - i) * s + (ggap / 2 if i < 5 else -ggap / 2)
              for i in range(10)]
    # five evenly spaced slots for t = 1 (all expansions of the root)
    rows5 = list(np.linspace(rows10[0] - s, rows10[-1] + s, 5))

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    ax.set_aspect("equal")
    ax.axis("off")

    def node(x, y, text, kept):
        face = BLUE if kept else "white"
        edge = BLUE if kept else GRAY
        col = "white" if kept else GRAY
        ax.add_patch(plt.Circle((x, y), r, facecolor=face, edgecolor=edge,
                                linewidth=1.4, zorder=3))
        ax.text(x, y, f"${text}$", ha="center", va="center", fontsize=13,
                color=col, zorder=4)

    def edge(p, q, kept):
        (x0, y0), (x1_, y1_) = p, q
        d = np.hypot(x1_ - x0, y1_ - y0)
        ux, uy = (x1_ - x0) / d, (y1_ - y0) / d
        ax.plot([x0 + ux * (r + 0.04), x1_ - ux * (r + 0.04)],
                [y0 + uy * (r + 0.04), y1_ - uy * (r + 0.04)],
                color=BLUE if kept else GRAY, lw=2.2 if kept else 1.0,
                alpha=1.0 if kept else 0.45, zorder=2)

    # root: the (empty) prefix
    ax.add_patch(FancyBboxPatch(
        (x_root - 0.62, -0.30), 1.24, 0.60,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=GRAY, alpha=0.15, edgecolor="none", zorder=3))
    ax.text(x_root, 0.0, "prefix", ha="center", va="center", fontsize=13,
            color="black", zorder=4)
    root = (x_root + 0.68, 0.0)

    # ---- t = 1: expand the root over the whole vocabulary; keep A and C ----
    kept1 = {"A", "C"}
    pos1 = {}
    for tok, y in zip(tokens, rows5):
        pos1[tok] = (x1, y)
        edge((root[0] - 0.04, root[1]), (x1, y), tok in kept1)
        node(x1, y, tok, tok in kept1)

    # ---- t = 2: expand A (top five slots) and C (bottom five); keep AB, CE --
    kept2 = {("A", "B"), ("C", "E")}
    pos2 = {}
    for gi, parent in enumerate(["A", "C"]):
        for ti, tok in enumerate(tokens):
            y = rows10[gi * 5 + ti]
            pos2[(parent, tok)] = (x2, y)
            edge(pos1[parent], (x2, y), (parent, tok) in kept2)
            node(x2, y, tok, (parent, tok) in kept2)

    # ---- t = 3: expand AB and CE; keep ABD and CED -------------------------
    kept3 = {("A", "B", "D"), ("C", "E", "D")}
    for gi, parent in enumerate([("A", "B"), ("C", "E")]):
        for ti, tok in enumerate(tokens):
            y = rows10[gi * 5 + ti]
            seq = parent + (tok,)
            edge(pos2[parent], (x3, y), seq in kept3)
            node(x3, y, tok, seq in kept3)

    # column headers (functional: they name the generation steps)
    ytop = rows10[0] + 0.78
    for x, lab in [(x1, "$t = 1$"), (x2, "$t = 2$"), (x3, "$t = 3$")]:
        ax.text(x, ytop, lab, ha="center", va="center", fontsize=14,
                color="black")

    # legend: kept vs pruned candidates
    handles = [
        Line2D([], [], marker="o", ls="none", markersize=9,
               markerfacecolor=BLUE, markeredgecolor=BLUE,
               label="kept in the beam ($w = 2$)"),
        Line2D([], [], marker="o", ls="none", markersize=9,
               markerfacecolor="white", markeredgecolor=GRAY,
               label="scored, then pruned"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=11.5, ncol=2,
              frameon=False, bbox_to_anchor=(0.5, -0.085),
              handletextpad=0.35, columnspacing=1.4)

    ax.set_xlim(x_root - 0.75, x3 + 0.55)
    ax.set_ylim(rows10[-1] - 0.95, ytop + 0.32)
    fl.save(fig, "mdl-rnn-beam-tree")


def fig_decoding_taskmap():
    """One trained language model, two families of decoding strategies.

    A root box (the model's next-token distribution) feeds a maximization
    branch (one nearly-right answer: greedy, beam search) and a sampling
    branch (many acceptable answers: temperature, top-k/top-p/min-p), each
    with a short stack of the tasks it serves.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")

    def box(cx, cy, w, h, face, alpha, edge="none", lw=0.0, z=2):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=face, alpha=alpha, edgecolor=edge, linewidth=lw,
            zorder=z))

    # root: the trained model's next-token distribution
    rx, ry, rw, rh = 5.0, 4.05, 4.3, 0.95
    box(rx, ry, rw, rh, GRAY, 0.13)
    ax.text(rx, ry + 0.18, "trained language model", ha="center", va="center",
            fontsize=14, color="black", zorder=3)
    ax.text(rx, ry - 0.22, r"$P(x_t \mid x_1, \ldots, x_{t-1})$", ha="center",
            va="center", fontsize=13, color="black", zorder=3)

    # the two family boxes
    fam = dict(w=4.6, h=1.5, y=2.15)
    lx, rx2 = 2.45, 7.55
    for cx, color, title, goal, methods in [
            (lx, BLUE, "maximization", "one (nearly) right answer",
             "greedy decoding,  beam search"),
            (rx2, ORANGE, "sampling", "many acceptable answers",
             "temperature,  top-$k$,  top-$p$,  min-$p$")]:
        box(cx, fam["y"], fam["w"], fam["h"], color, 0.14)
        ax.text(cx, fam["y"] + 0.42, title, ha="center", va="center",
                fontsize=15, color="black", zorder=3)
        ax.text(cx, fam["y"], goal, ha="center", va="center",
                fontsize=12.5, color="black", style="italic", zorder=3)
        ax.text(cx, fam["y"] - 0.42, methods, ha="center", va="center",
                fontsize=12, color="black", zorder=3)

    # arrows from the root to each family
    for cx in (lx, rx2):
        fl.arrow(ax, (rx if cx == lx else rx, ry - rh / 2 - 0.04),
                 (cx, fam["y"] + fam["h"] / 2 + 0.06),
                 color=BLUE if cx == lx else ORANGE, lw=2.0, mut=15)

    # task chips under each family
    chips = {
        lx: (BLUE, ["machine translation", "speech recognition",
                    "code, constrained output"]),
        rx2: (ORANGE, ["chat and dialogue", "story writing",
                       "brainstorming"]),
    }
    ch, cw, gap = 0.56, 3.6, 0.16
    y0 = fam["y"] - fam["h"] / 2 - 0.42
    for cx, (color, labels) in chips.items():
        for i, lab in enumerate(labels):
            cy = y0 - i * (ch + gap)
            box(cx, cy, cw, ch, color, 0.07, edge=color, lw=1.0)
            ax.text(cx, cy, lab, ha="center", va="center", fontsize=12,
                    color="black", zorder=3)

    ax.set_xlim(0.1, 9.9)
    ax.set_ylim(y0 - 2 * (ch + gap) - ch / 2 - 0.25, ry + rh / 2 + 0.25)
    fl.save(fig, "mdl-rnn-decoding-taskmap")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_ar_vs_latent,
    fig_rnn_granularity_spectrum,
    fig_rnn_merge_tree,
    fig_rnn_pretokenization_pipeline,
    fig_partitioning,
    fig_unfolded,
    fig_lm_shift,
    fig_truncated_bptt,
    fig_rnn_bptt,
    fig_beam_tree,
    fig_decoding_taskmap,
]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
