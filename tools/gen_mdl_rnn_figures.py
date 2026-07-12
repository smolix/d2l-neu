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

from matplotlib.patches import FancyBboxPatch


# =========================================================================== #
# 9.1 "Working with Sequences": autoregression vs. latent autoregression.     #
# =========================================================================== #

def fig_ar_vs_latent():
    """Two schematics of how a sequence model makes its input length fixed.

    (left) autoregression: a sliding window of the last tau observations feeds
    the prediction; older observations are discarded.
    (right) latent autoregression: a recurrent state h_t summarizes the whole
    past in fixed size and is carried forward step by step.

    Equal aspect on both; y-spans match and gridspec width_ratios match the
    x-spans, so the panels render at the same height and share one scale.
    """
    fig = plt.figure(figsize=(10.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[8.4, 5.2], wspace=0.12)
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
    tau_idx = [5, 6, 7]                 # the window: last tau observed points
    pred_t = 8                          # the step we predict

    # faint signal line through the observed points 1..7
    axa.plot(t[:7], y[:7], "-", color=LIGHT, lw=1.6, zorder=1)
    # observed points: earlier ones faded, windowed ones solid blue
    for ti, yi in zip(t[:7], y[:7]):
        faded = ti < tau_idx[0]
        axa.plot(ti, yi, "o", color=BLUE, ms=8, zorder=4,
                 alpha=0.30 if faded else 1.0)

    # shade the window over the last tau observations
    x0, x1 = tau_idx[0] - 0.45, tau_idx[-1] + 0.45
    ylo, yhi = 0.55, 2.75
    axa.add_patch(fl.Rectangle((x0, ylo), x1 - x0, yhi - ylo, facecolor=BLUE,
                               alpha=0.10, edgecolor=BLUE, lw=1.4, zorder=2))
    axa.text((x0 + x1) / 2, yhi + 0.16,
             r"window of length $\tau$", color="black", ha="center",
             va="bottom", fontsize=LBL)
    axa.text((x0 + x1) / 2, ylo - 0.14,
             r"$x_{t-\tau},\ \ldots,\ x_{t-1}$", color="black", ha="center",
             va="top", fontsize=LBL)

    # the prediction: an open marker at the next step, arrow from the window
    yp = y[pred_t - 1]
    axa.plot(pred_t, yp, "o", mfc="white", mec=GREEN, mew=2.2, ms=11, zorder=5)
    fl.arrow(axa, (x1, (ylo + yhi) / 2 + 0.15), (pred_t - 0.22, yp),
             color=GREEN, lw=2.0)
    axa.text(pred_t + 0.12, yp + 0.30, r"$\hat{x}_t$", color=GREEN, ha="left",
             va="bottom", fontsize=LBL + 1)

    # a light time axis along the bottom
    fl.arrow(axa, (0.3, 0.15), (8.5, 0.15), color="black", lw=1.0)
    axa.text(8.55, 0.15, r"$t$", color="black", ha="left", va="center",
             fontsize=LBL)

    axa.set_title("autoregression", fontsize=TTL, color="black")
    axa.set_xlim(0.1, 9.0)
    axa.set_ylim(-0.05, 3.45)
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
    labs_h = [r"$h_1$", r"$h_2$", r"$h_3$"]
    for cx, lx, lh in zip(xs, labs_x, labs_h):
        node(axb, cx, y_in, lx, BLUE)
        node(axb, cx, y_h, lh, ORANGE)
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
    axb.set_ylim(-0.05, 3.45)
    axb.set_aspect("equal")
    axb.axis("off")

    fl.save(fig, "mdl-rnn-ar-vs-latent")


# =========================================================================== #
# 9.2 "From Text to Tokens": granularity spectrum, BPE merge tree, and the    #
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
    ax.tick_params(labelsize=11, colors='black')

    # The three granularities, as points on the curve.
    ax.plot([256], [1.0], 'o', color=BLUE, ms=8, zorder=4)
    ax.annotate('bytes (vocab 256)', xy=(256, 1.0), xytext=(400, 1.01),
                fontsize=13, color='black', ha='left', va='center')
    ax.annotate('long sequences,\nnothing ever\nout of vocabulary',
                xy=(430, 0.88), fontsize=11, color=GRAY, ha='left', va='top')

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
    ax.annotate('short sequences, but a\nfixed list: rare words fall\n'
                'out (out of vocabulary)',
                xy=(6.5e5, 0.65), fontsize=11, color=GRAY, ha='center',
                va='top')
    fl.save(fig, 'mdl-rnn-granularity-spectrum')


def _tok_box(ax, x, y, text, w=0.72, h=0.5, fc='#dbe9f6', ec='black',
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

    leaf_fc, node_fc = '#dbe9f6', '#fde3c8'   # light blue / light orange
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
        fc = '#fde3c8' if digits(chunk) else '#dbe9f6'
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
# 9.3 "Language Models" (sec_language-model): partitioning a BPE token-id     #
# stream into overlapping input/target subsequences.                         #
#                                                                              #
# One restyled carryover, written under a new house-style name so the old    #
# hand-drawn SVG (``img/lang-model-data.svg``) is left untouched on disk.    #
# =========================================================================== #

# The actual first 12 BPE tokens of "The Time Traveller (for so it will be
# convenient ..." under the section's tokenizer; leading spaces marked "␣".
_PARTITION_TOKENS = ["The", "␣Time", "␣Traveller", "␣for", "␣so", "␣it",
                     "␣will", "␣be", "␣con", "ven", "i", "ent"]
_PARTITION_N = 5             # window length (num_steps)
_PARTITION_OFFSETS = [0, 6]  # two windows sampled into one minibatch


def _token_row(ax, y, edges, h, fill_upto=None):
    """One copy of the token stream: monospace text in thin boxes whose width
    is proportional to the token length (as in the 9.2 tokenizer figures)."""
    for i, tok in enumerate(_PARTITION_TOKENS):
        x0, x1 = edges[i], edges[i + 1]
        ax.add_patch(plt.Rectangle((x0, y - h / 2), x1 - x0, h,
                                   facecolor="white", edgecolor=GRAY,
                                   linewidth=1.0, zorder=2))
        ax.text((x0 + x1) / 2, y, tok, ha="center", va="center",
                fontsize=13, family="monospace", color="black", zorder=4)


def _window(ax, edges, lo, hi, y, h, color, label, label_side):
    """A heavy rounded window outline with faint fill spanning token boxes
    lo..hi (inclusive), plus its bold subsequence label above/below."""
    x0, x1 = edges[lo], edges[hi + 1]
    pad = 0.045
    ax.add_patch(FancyBboxPatch(
        (x0 + pad, y - h / 2 - 0.10), x1 - x0 - 2 * pad, h + 0.20,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=2.2, edgecolor=color, facecolor=color, alpha=0.14,
        zorder=1))
    ax.add_patch(FancyBboxPatch(
        (x0 + pad, y - h / 2 - 0.10), x1 - x0 - 2 * pad, h + 0.20,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=2.2, edgecolor=color, facecolor="none", zorder=3))
    ly = y + h / 2 + 0.32 if label_side == "above" else y - h / 2 - 0.34
    va = "bottom" if label_side == "above" else "top"
    ax.text((x0 + x1) / 2, ly, label, ha="center", va=va,
            fontsize=14, color=color, zorder=4)


def fig_partitioning():
    """Partitioning one BPE token-id stream into (input, target) pairs.

    Top row: the stream with two sampled length-5 input windows (blue),
    labelled x_0 and x_6.  Bottom row: the same stream with the target
    windows (orange), which are the input windows shifted forward by one
    token.  Position indices x_0 .. x_11 sit between the two rows, shared
    by both copies since the columns align."""
    # box edges: width proportional to token length + constant padding
    widths = [0.185 * len(t) + 0.34 for t in _PARTITION_TOKENS]
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    total = edges[-1]

    h = 0.62                    # token box height
    y_in, y_tgt = 1.55, -0.10   # the two stream copies
    fig, ax = plt.subplots(figsize=(11.2, 3.1))
    ax.set_aspect("equal")
    ax.axis("off")

    _token_row(ax, y_in, edges, h)
    _token_row(ax, y_tgt, edges, h)

    # shared position indices between the rows (columns align)
    y_idx = (y_in + y_tgt) / 2
    for i in range(len(_PARTITION_TOKENS)):
        ax.text((edges[i] + edges[i + 1]) / 2, y_idx, rf"$x_{{{i}}}$",
                ha="center", va="center", fontsize=11, color="black")

    # input windows (blue, labels above) and shifted target windows (orange,
    # labels below)
    for t in _PARTITION_OFFSETS:
        _window(ax, edges, t, t + _PARTITION_N - 1, y_in, h, BLUE,
                rf"$\mathbf{{x}}_{{{t}}}$", "above")
        _window(ax, edges, t + 1, t + _PARTITION_N, y_tgt, h, ORANGE,
                rf"$\mathbf{{x}}_{{{t + 1}}}$", "below")

    # row labels on the left, in black per the house checklist
    ax.text(-0.25, y_in, "inputs", ha="right", va="center", fontsize=13,
            color="black")
    ax.text(-0.25, y_tgt, "targets", ha="right", va="center", fontsize=13,
            color="black")

    ax.set_xlim(-1.75, total + 0.15)
    ax.set_ylim(y_tgt - h / 2 - 0.75, y_in + h / 2 + 0.72)
    fl.save(fig, "mdl-rnn-partitioning")


# =========================================================================== #
# 9.4 "Recurrent Neural Networks" (sec_rnn): the unrolled RNN and the         #
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
        _box(ax, cx, y_hid, bw, bh, rf"$\mathbf{{H}}_{{{s}}}$", GREEN)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{{{s}}}$", ORANGE)
        # input -> hidden, and hidden -> output (same weights every step)
        fl.arrow(ax, (cx, y_in + half), (cx, y_hid - half), color=GRAY, lw=2.0,
                 mut=15)
        fl.arrow(ax, (cx, y_hid + half), (cx, y_out - half), color=GRAY, lw=2.0,
                 mut=15)
        ax.text(cx + 0.14, (y_in + y_hid) / 2, r"$\mathbf{W}_{xh}$",
                ha="left", va="center", fontsize=12, color="black")
        ax.text(cx + 0.14, (y_hid + y_out) / 2, r"$\mathbf{W}_{hq}$",
                ha="left", va="center", fontsize=12, color="black")

    # recurrent links between hidden states, labelled W_hh above the shaft
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (cx0 + bw / 2, y_hid), (cx1 - bw / 2, y_hid), color=GRAY,
                 lw=2.0, mut=15)
        ax.text((cx0 + cx1) / 2, y_hid + 0.24, r"$\mathbf{W}_{hh}$",
                ha="center", va="bottom", fontsize=12, color="black")

    # dashed continuation stubs, so the chain reads as unbounded in time
    fl.arrow(ax, (xs[0] - 1.35, y_hid), (xs[0] - bw / 2, y_hid), color=GRAY,
             lw=1.6, ls="--", mut=13)
    fl.arrow(ax, (xs[-1] + bw / 2, y_hid), (xs[-1] + 1.35, y_hid), color=GRAY,
             lw=1.6, ls="--", mut=13)

    ax.set_xlim(0.2, 9.6)
    ax.set_ylim(0.2, 5.8)
    fl.save(fig, "mdl-rnn-unfolded")


def fig_lm_shift():
    """RNN language model trained by teacher forcing over BPE subword tokens.
    The input tokens (the, time, mach, ine) feed the recurrence; each hidden
    state emits a next-token distribution O_t; the target sequence is the input
    shifted forward by one token (time, mach, ine, by), so the per-step
    cross-entropy trains the model to predict the next token.  The word
    'machine' splitting into the two subword tokens 'mach' and 'ine' shows the
    BPE tokenization of sec_text-sequence."""
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.5, 4.9, 7.3, 9.7]                       # four time steps
    y_in, y_hid, y_out, y_tgt = 0.9, 3.0, 5.1, 6.8  # four rows
    bw, bh = 1.55, 0.9
    inp = ["the", "time", "mach", "ine"]
    tgt = ["time", "mach", "ine", "by"]

    # left-hand row labels (black)
    ax.text(0.15, y_tgt, "targets", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.15, y_out, "predict", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.15, y_hid, "hidden\nstate", ha="left", va="center", fontsize=13,
            color="black", linespacing=1.15)
    ax.text(0.15, y_in, "inputs", ha="left", va="center", fontsize=13,
            color="black")

    half = bh / 2
    for i, cx in enumerate(xs):
        _box(ax, cx, y_in, bw, bh, inp[i], BLUE, fontsize=15)
        _box(ax, cx, y_hid, bw, bh, rf"$\mathbf{{H}}_{{{i+1}}}$", GREEN)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{{{i+1}}}$", ORANGE)
        _box(ax, cx, y_tgt, bw, bh, tgt[i], GRAY, fontsize=15)
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
            fontsize=11, color="black")

    # brace under the two subword tokens of "machine"
    yb = y_in - 0.72
    ax.plot([xs[2] - bw / 2, xs[3] + bw / 2], [yb, yb], color=GRAY, lw=1.3)
    for cx in (xs[2], xs[3]):
        ax.plot([cx, cx], [yb, y_in - half - 0.04], color=GRAY, lw=1.0)
    ax.text((xs[2] + xs[3]) / 2, yb - 0.30, '"machine"', ha="center", va="top",
            fontsize=12, color="black")

    ax.set_xlim(0.0, 10.6)
    ax.set_ylim(-0.75, 7.5)
    fl.save(fig, "mdl-rnn-lm-shift")


# =========================================================================== #
# 9.6 "Backpropagation Through Time" (sec_bptt): full BPTT vs. regular        #
# truncation, gradient chains severed (detached) at segment boundaries.       #
#                                                                              #
# One restyled carryover, written under a new house-style name so the old    #
# hand-drawn SVG (``img/truncated-bptt.svg``) is left untouched.  The old     #
# three-row figure's randomized-truncation row is dropped (that strategy is  #
# retired from the section).                                                 #
# =========================================================================== #

def _segment(ax, x0, x1, yc, h=0.62):
    """A faint rounded rectangle marking one backpropagation *segment*."""
    ax.add_patch(FancyBboxPatch(
        (x0, yc - h / 2), x1 - x0, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=GRAY, alpha=0.12, zorder=0))


def _chain(ax, xs, yc, r=0.085, color=BLUE):
    """Nodes (green hidden states) linked left-to-right by dependency arrows."""
    for x in xs:
        ax.add_patch(plt.Circle((x, yc), r, facecolor=GREEN, edgecolor="none",
                                 zorder=3))
    for x0, x1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (x0 + r + 0.02, yc), (x1 - r - 0.02, yc), color=color,
                 lw=2.0, mut=13)


def fig_truncated_bptt():
    """Two ways of computing the gradient across a token sequence.  TOP: full
    backpropagation through time -- one unbroken dependency chain spans the
    whole sequence (a single segment).  BOTTOM: regular truncation -- the
    sequence is cut into equal-length segments (here tau = 4) and the state is
    *detached* at every boundary, so the gradient never crosses it."""
    tokens = ["the", "time", "mach", "ine", "by", "h", "g", "wells"]
    n = len(tokens)
    tau = 4                                  # regular-truncation segment length
    xs = np.linspace(1.15, 9.35, n)
    dx = xs[1] - xs[0]
    pad = 0.42 * dx                          # segment box overhang past the nodes

    y_tok, y_full, y_trunc = 3.02, 2.05, 0.72

    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set_aspect("equal")
    ax.axis("off")

    # token strip, shared by both rows (black text)
    for x, tok in zip(xs, tokens):
        ax.text(x, y_tok, tok, ha="center", va="center", fontsize=13,
                color="black")

    # TOP: full BPTT -- one segment spanning the whole sequence, unbroken chain
    _segment(ax, xs[0] - pad, xs[-1] + pad, y_full)
    _chain(ax, xs, y_full)

    # BOTTOM: regular truncation -- equal segments, chain severed at boundaries
    for s in range(0, n, tau):
        blk = xs[s:s + tau]
        _segment(ax, blk[0] - pad, blk[-1] + pad, y_trunc)
        _chain(ax, blk, y_trunc)

    # mark every detached boundary (here just one, between the two segments)
    for s in range(tau, n, tau):
        xb = (xs[s - 1] + xs[s]) / 2
        ax.plot([xb, xb], [y_trunc - 0.31, y_trunc + 0.31], color=ORANGE,
                lw=2.6, zorder=4)
        ax.annotate("detach", xy=(xb, y_trunc + 0.33),
                    xytext=(xb, y_trunc + 0.80), ha="center", va="bottom",
                    fontsize=11.5, color="black",
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))

    ax.set_xlim(xs[0] - pad - 0.35, xs[-1] + pad + 0.35)
    ax.set_ylim(0.18, 3.42)
    fl.save(fig, "mdl-rnn-truncated-bptt")


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
