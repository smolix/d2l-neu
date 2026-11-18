#!/usr/bin/env python3
"""Generate the self-attention worked example for
chapter_attention/multihead-attention.md (A9) in the shared house style
defined in ``gen_mdl_figures.py``.

Figures:
  mdl-attention-self-attention.svg -- one head of self-attention on the
      sentence "Alice gave Bob the book" (fig_self-attention). Every token
      emits a query, a key and a value; the query of "gave" is scored against
      every key, a softmax turns the five scores into weights, and the output
      of "gave" is the weighted sum of the five values. The matrices are the
      hand-set ones of the section's cell, and every number in the figure is
      computed here from them, so the figure equals the cell's printout.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a9.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Deterministic (no
random draws, no timestamps), hence byte-idempotent via the shared ``save()``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from gen_mdl_attention_a7 import _box

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle


# The section's hand-set example (cell multihead-attention-self-attention).
TOKENS = ["Alice", "gave", "Bob", "the", "book"]
# Role of each token: person, action, thing, function word.
ROLES = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                  [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
X = np.concatenate([ROLES, np.eye(5)], axis=1)   # role part, then the word
W_q = np.concatenate([[[1, -1], [1, 1], [1, -1], [0, 0]], np.zeros((5, 2))])
W_k = np.concatenate([[[1, 1], [1, -1], [1, 1], [0, 0]], np.zeros((5, 2))])
W_v = np.concatenate([np.zeros((4, 5)), np.eye(5)])


def fig_self_attention() -> None:
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    focus = TOKENS.index("gave")
    scores = Q[focus] @ K.T / np.sqrt(Q.shape[1]) + 0.0  # 1.41 or 0 (no -0)
    w = np.exp(scores) / np.exp(scores).sum()            # 0.29 or 0.07
    o = w @ V
    cols = [BLUE, ORANGE, GREEN]

    def vec(x):
        return "(" + ", ".join(f"{c:g}" for c in x) + ")"

    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    ys = [4.4 - 1.05 * i for i in range(5)]          # one row per token
    x_tok, x_q, x_k = 0.75, 2.35, 4.55
    x_s, x_bar, ulen = 6.05, 6.95, 1.5
    x_v, x_o = 10.05, 13.15
    cw, ch = 1.3, 0.62                                 # chip size

    for i, (tok, y) in enumerate(zip(TOKENS, ys)):
        on = i == focus
        _box(ax, x_tok, y, 1.2, ch, tok, GRAY, fontsize=14,
             weight="bold" if on else "normal")
        # every token emits a query; only the query of "gave" is used here
        _box(ax, x_q, y, cw, ch, vec(Q[i]), BLUE if on else LIGHT,
             fontsize=13, lw=2.4 if on else 1.2, ls="-" if on else "--",
             fc_alpha=0.12 if on else 0.0)
        _box(ax, x_k, y, cw, ch, vec(K[i]), ORANGE, fontsize=13)
        fl.arrow(ax, (x_q + cw / 2 + 0.04, ys[focus]),
                 (x_k - cw / 2 - 0.05, y), color=BLUE, lw=1.3, mut=11)
        # score of the query of "gave" against key i
        ax.text(x_s, y, f"{scores[i]:.2f}", ha="center", va="center",
                fontsize=13, color="black")
        fl.arrow(ax, (x_k + cw / 2 + 0.04, y), (x_s - 0.38, y),
                 color=GRAY, lw=1.3, mut=11)
        fl.arrow(ax, (x_s + 0.38, y), (x_bar - 0.06, y),
                 color=GRAY, lw=1.3, mut=11)
        # softmax weight as a fraction of a unit bar
        ax.add_patch(Rectangle((x_bar, y - 0.15), ulen, 0.30,
                               facecolor="none", edgecolor=GRAY, lw=1.0,
                               ls="--", zorder=2))
        ax.add_patch(Rectangle((x_bar, y - 0.15), ulen * w[i], 0.30,
                               facecolor=BLUE, edgecolor="none", alpha=0.85,
                               zorder=3))
        ax.text(x_bar + ulen / 2, y + 0.19, f"{w[i]:.2f}", ha="center",
                va="bottom", fontsize=13, color="black")
        fl.arrow(ax, (x_bar + ulen + 0.08, y), (x_v - 1.03, y),
                 color=GRAY, lw=1.3, mut=11)
        _box(ax, x_v, y, 1.95, ch, vec(V[i]), GREEN, fontsize=13)
        # each value flows into the output with its weight (width ~ weight)
        fl.arrow(ax, (x_v + 1.02, y), (x_o - 1.12, ys[2] + 0.3 * (2 - i)),
                 color=GREEN, lw=0.8 + 6.0 * w[i], mut=12)

    head_y = ys[0] + 0.62
    for x, text in ((x_tok, "token"), (x_q, r"query $\mathbf{q}$"),
                    (x_k, r"key $\mathbf{k}$"),
                    (x_s, "score\n" + r"$\mathbf{q}_{\mathrm{gave}} \cdot \mathbf{k}/\sqrt{2}$"),
                    (x_bar + ulen / 2, r"weight $\alpha$"),
                    (x_v, r"value $\mathbf{v}$")):
        ax.text(x, head_y, text, ha="center", va="bottom", fontsize=14,
                color="black")

    _box(ax, x_o, ys[2], 2.15, 1.75,
         "output of gave\n" + r"$\Sigma_j\,\alpha_j \mathbf{v}_j$" + "\n"
         + "(" + ", ".join(f"{c:.2f}" for c in o[:3]) + ",\n"
         + " " + ", ".join(f"{c:.2f}" for c in o[3:]) + ")",
         "black", fontsize=13, fc_alpha=0.06)

    ax.set_xlim(0.0, 14.3)
    ax.set_ylim(ys[-1] - 0.45, head_y + 0.85)
    fl.save(fig, "mdl-attention-self-attention")


if __name__ == "__main__":
    fig_self_attention()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path)}")
