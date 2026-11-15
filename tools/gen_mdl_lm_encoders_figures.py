#!/usr/bin/env python3
"""Generate the illustrative figures of ``chapter_lm-encoders`` (Chapter 19,
"Encoders, Embeddings and Retrieval") in the shared house style.

Four figures (spec section 5 of 19.1, 19.2, 19.4 and 19.5), all written to
``img/mdl-lm-encoders-<id>.svg``:

  sgns-pmi       Skip-gram with negative sampling: a window of five words on
                 each side of a center word, the center table V and the
                 context table U, the 1 + K scored slots of one observed pair
                 with K = 5 noise words drawn from n(o)^0.75, and the shifted
                 PMI matrix of the section's toy corpus that the optimum
                 reproduces (computed here from the toy corpus).
  encoder-heads  One encoder, three heads: a pooled sentence head, a pair read
                 jointly with no segment ids, and a per-token tagging head
                 that labels each word's first token (ModernBERT-base tokens).
  ivf-pq         Product quantization on the section's hand case (m = 4,
                 M = 2, K = 2): codebooks, code, the look-up table of one
                 question and the distance as a sum of M entries; beside it an
                 inverted file with n_probe = 2 cells probed (k-means computed
                 here on seeded points).
  rag-stages     The stages of the retrieval-augmented generator, the
                 measurement that sees each and the failure class it causes.

Literal data the figures show is checked on every run: the toy-corpus counts
and PMI against the numbers the text quotes, the product-quantization hand case
against its code, table and distances, and the k-means of the inverted-file
panel against a probe that misses the exact neighbor.  The token splits are
those of ModernBERT-base at revision 8949b909 (answerdotai/ModernBERT-base),
checked with the pinned tokenizer on 2026-11-15:

  'a charming film'          [CLS] a Ġcharming Ġfilm [SEP]
  'It rains', 'It is dry'    [CLS] It Ġrains [SEP] It Ġis Ġdry [SEP]
                             (fields input_ids, attention_mask: no
                             token_type_ids)
  'Peter Blackburn spoke'    [CLS] Peter ĠBlack burn Ġspoke [SEP]

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_lm_encoders_figures.py

Byte-idempotent: no timestamps, seeded randomness only, fixed svg.hashsalt
(through gen_mdl_figures.save).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (applies the shared rcParams)
import figstyle.mpl as fm  # noqa: E402
from figstyle import tokens as T  # noqa: E402
from figstyle.textmetrics import measure  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

np, plt = fl.np, fl.plt
INK, MUTED, FAINT = T.INK, T.MUTED, T.FAINT
BLUE, ORANGE, GREEN, PURPLE, RED, SLATE = (T.BLUE, T.ORANGE, T.GREEN,
                                           T.PURPLE, T.RED, T.SLATE)
MONO, SANS = T.FONT_MONO, T.FONT_SANS
FS, FS_S, FS_T = 15.0, 13.5, 13.0     # labels, secondary, small caps / mono
PREFIX = "mdl-lm-encoders"
# light-to-saturated blue for the PMI matrix; white numbers on its dark half
PMI_CMAP = fl._LSC.from_list("pmi", [BLUE.tint, BLUE.base])


# --------------------------------------------------------------------------- #
# Helpers: a canvas in inches, measured chips, rounded boxes                   #
# --------------------------------------------------------------------------- #

def canvas(w, h, y0=0.0):
    """A figure whose one hidden axes uses inches as data units; the
    visible window is x in [0, w], y in [y0, h]."""
    fig = plt.figure(figsize=(w, h - y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(y0, h)
    ax.axis("off")
    return fig, ax


def tw(text, fs, family="sans"):
    """Width in inches of ``text`` set at ``fs`` points."""
    return measure(text, fs, family).width / 72


def rbox(ax, x, y, w, h, accent=None, lw=1.6, dash=False, fill=None,
         edge=None, r=0.07, zorder=3):
    """Rounded rectangle; ``accent`` gives tint fill and base edge."""
    fc = fill or (accent.tint if accent else T.PAPER)
    ec = edge or (accent.base if accent else INK)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, lw=lw,
        linestyle=(0, (4, 3)) if dash else "-", zorder=zorder))


def chip(ax, x, y, text, fs=FS_T, mono=True, accent=None, h=0.34, pad=0.07,
         color=INK, lw=1.3, **kw):
    """A token chip with its left edge at ``x``; returns its width."""
    w = tw(text, fs, "mono" if mono else "sans") + 2 * pad
    rbox(ax, x, y, w, h, accent=accent, lw=lw, **kw)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=color,
            family=MONO if mono else SANS, ha="center",
            va="center_baseline", zorder=5)
    return w


def chips(ax, x, y, tokens, gap=0.06, **kw):
    """A row of chips from ``x``; returns the (left, right) of every chip.
    ``tokens`` holds strings or (string, chip-kwargs) pairs."""
    spans = []
    for t in tokens:
        text, extra = (t, {}) if isinstance(t, str) else t
        w = chip(ax, x, y, text, **{**kw, **extra})
        spans.append((x, x + w))
        x += w + gap
    return spans


def flow(ax, tail, tip, color=INK, lw=1.6, mut=13):
    fm.arrow(ax, tail, tip, color=color, lw=lw, mut=mut)


# --------------------------------------------------------------------------- #
# 19.1  Skip-gram with negative sampling and the shifted PMI                   #
# --------------------------------------------------------------------------- #

# The toy corpus of word-embeddings.md (cell word-embeddings-toy-pmi), K = 5.
TOY = ("ice ice steam gas steam ice solid ice steam gas ice solid steam "
       "solid steam gas gas steam gas steam steam ice gas steam solid gas "
       "steam solid solid ice solid ice ice solid gas steam gas steam gas "
       "solid").split()
TOY_WORDS = ["ice", "steam", "solid", "gas"]
K_NOISE = 5


def toy_shifted_pmi():
    """pmi - log K of the toy corpus, window 1 (d2l.cooccurrence_counts)."""
    ids = [TOY_WORDS.index(w) for w in TOY]
    X = np.zeros((4, 4))
    for a, b in zip(ids[:-1], ids[1:]):
        X[a, b] += 1
        X[b, a] += 1
    n_w, n_o, total = X.sum(1), X.sum(0), X.sum()
    pmi = np.log(X * total / np.outer(n_w, n_o))
    # the numbers the text quotes: 78 pairs, steam-gas 13 times, rows 24
    # and 20, pmi about 0.75; every pair occurs
    s, g = TOY_WORDS.index("steam"), TOY_WORDS.index("gas")
    assert total == 78 and X[s, g] == 13 and (n_w[s], n_w[g]) == (24, 20)
    assert abs(pmi[s, g] - 0.75) < 0.005 and (X > 0).all()
    return pmi - np.log(K_NOISE)


def fig_sgns_pmi():
    W, H = 9.1, 4.45
    fig, ax = canvas(W, H)

    # ---- the window: five words on each side of the center word ----------
    left = ["when", "hot", "water", "turns", "into"]
    right = ["that", "drives", "a", "big"]
    toks = (["…"] + left + [("steam", dict(accent=BLUE, lw=1.8))] + right
            + [("piston", dict(accent=GREEN, lw=1.8))] + ["…"])
    fs_w, gap, ty = 14.5, 0.08, 3.5
    widths = [tw(t if isinstance(t, str) else t[0], fs_w) + 0.16
              for t in toks]
    x, spans = (W - (sum(widths) + gap * (len(toks) - 1))) / 2, []
    for w in widths:
        spans.append((x, x + w))
        x += w + gap
    wl, wr = spans[1][0] - 0.07, spans[-2][1] + 0.07
    rbox(ax, wl, ty - 0.09, wr - wl, 0.54, fill=BLUE.tint, edge=BLUE.tint,
         lw=0, r=0.12, zorder=1)
    for t, (xa, xb) in zip(toks, spans):
        text, extra = (t, {}) if isinstance(t, str) else t
        if text == "…":
            ax.text((xa + xb) / 2, ty + 0.18, text, fontsize=fs_w,
                    color=MUTED, ha="center", va="center_baseline")
        else:
            chip(ax, xa, ty, text, fs=fs_w, mono=False, h=0.36, pad=0.08,
                 **extra)
    for a, b in ((spans[1][0], spans[5][1]), (spans[7][0], spans[11][1])):
        ax.plot([a, a, b, b], [ty + 0.5, ty + 0.58, ty + 0.58, ty + 0.5],
                color=INK, lw=1.2)
        ax.text((a + b) / 2, ty + 0.64, "five words", fontsize=FS_S,
                color=INK, ha="center", va="bottom")
    c = spans[6]
    ax.text((c[0] + c[1]) / 2, ty + 0.64, r"center $w$", fontsize=FS_S,
            color=BLUE.dark, ha="center", va="bottom")

    # ---- the two tables, rows in one shared vocabulary order ---------------
    noise = ["the", "season", "of", "river", "album"]   # K = 5 draws
    vocab = ["piston"] + noise + ["steam", "water"]
    row_h, cell_w, ncol, top = 0.34, 0.17, 4, 2.72
    lab_w = max(tw(v, FS_S) for v in vocab) + 0.1

    def table(xl, title, mark):
        gx = xl + lab_w
        ax.text(gx + ncol * cell_w / 2, top + 0.1, title, fontsize=FS,
                ha="center", va="bottom")
        rows = {}
        for i, v in enumerate(vocab):
            y = top - (i + 1) * row_h
            acc = mark.get(v)
            ax.text(gx - 0.08, y + row_h / 2, v, fontsize=FS_S,
                    color=acc.dark if acc else INK, ha="right",
                    va="center_baseline")
            for j in range(ncol):
                ax.add_patch(Rectangle(
                    (gx + j * cell_w, y), cell_w, row_h,
                    facecolor=acc.tint if acc else T.PAPER,
                    edgecolor=acc.base if acc else FAINT,
                    lw=1.2 if acc else 0.8, zorder=3 if acc else 2))
            rows[v] = y + row_h / 2
        return gx, gx + ncol * cell_w, rows

    vx0, vx1, vrows = table(0.0, r"$\mathbf{V}$", {"steam": BLUE})
    ax.text(vx1 + 0.07, vrows["steam"], r"$\mathbf{v}_w$", fontsize=FS,
            color=BLUE.dark, ha="left", va="center")

    # ---- the 1 + K slots of the pair (steam, piston) -----------------------
    sx, sw = 2.15, 1.95
    slots = [("piston", GREEN, "1")] + [(n, ORANGE, "0") for n in noise]
    for i, (word, acc, target) in enumerate(slots):
        yc = vrows[word]
        rbox(ax, sx, yc - 0.14, sw, 0.28, accent=acc, lw=1.3, r=0.06)
        ax.text(sx + 0.1, yc, f"steam, {word}", fontsize=FS_S, ha="left",
                va="center_baseline", zorder=5)
        ax.text(sx + sw - 0.12, yc, target, fontsize=FS_S, color=acc.dark,
                ha="right", va="center_baseline", zorder=5)
        # v_w enters every slot
        ax.plot([vx1 + 0.42, sx - 0.03], [vrows["steam"], yc],
                color=BLUE.base, lw=1.0, zorder=1)
    ax.text(sx + sw / 2, top + 0.1, r"$\sigma(\mathbf{u}_o^\top"
            r"\mathbf{v}_w)$, target", fontsize=FS, ha="center", va="bottom")
    ax.text(sx + sw / 2, vrows["steam"], r"$1 + K = 6$ slots",
            fontsize=FS_S, color=INK, ha="center", va="center")
    ax.text(sx + sw / 2, vrows["water"],
            r"noise $o_k \sim P(o) \propto n(o)^{0.75}$", fontsize=FS_S,
            color=ORANGE.dark, ha="center", va="center")

    umark = {"piston": GREEN, **{n: ORANGE for n in noise}}
    ux0, ux1, urows = table(sx + sw + 0.35, r"$\mathbf{U}$", umark)
    for word, acc, _ in slots:
        fm.arrow(ax, (ux0 - lab_w + 0.02, urows[word]),
                 (sx + sw + 0.03, urows[word]), color=acc.base, lw=1.2,
                 mut=10)
    ax.text(ux1 + 0.07, urows["piston"], r"$\mathbf{u}_o$", fontsize=FS,
            color=GREEN.dark, ha="left", va="center")
    ax.text(ux1 + 0.07, urows["season"], r"$\mathbf{u}_{o_k}$",
            fontsize=FS, color=ORANGE.dark, ha="left", va="center")

    # ---- the shifted PMI the optimum reproduces ----------------------------
    S = toy_shifted_pmi()
    cw, px0 = 0.5, 6.95
    py_top = top - 0.32
    lo, hi = S.min(), S.max()
    ax.text(px0 + 2 * cw, top + 0.1, r"$s^{\!*} = \mathrm{pmi} - \log K$",
            fontsize=FS, ha="center", va="bottom")
    for i, wv in enumerate(TOY_WORDS):
        ax.text(px0 - 0.08, py_top - (i + 0.5) * cw, wv, fontsize=FS_S,
                ha="right", va="center_baseline")
        ax.text(px0 + (i + 0.5) * cw, py_top + 0.06, wv, fontsize=FS_S,
                ha="center", va="bottom")
        for j in range(4):
            u = (S[i, j] - lo) / (hi - lo)
            ax.add_patch(Rectangle((px0 + j * cw, py_top - (i + 1) * cw),
                                   cw, cw, facecolor=PMI_CMAP(u),
                                   edgecolor=T.PAPER, lw=1.5))
            ax.text(px0 + (j + 0.5) * cw, py_top - (i + 0.5) * cw,
                    f"{S[i, j]:.2f}".replace("-", "\u2212"), fontsize=12.5,
                    color=T.PAPER if u > 0.5 else INK, ha="center",
                    va="center_baseline")
    ax.text(px0 + 2 * cw, py_top - 4 * cw - 0.12,
            "toy corpus, noise by frequency", fontsize=FS_S, color=MUTED,
            ha="center", va="top")
    fl.save(fig, f"{PREFIX}-sgns-pmi")


# --------------------------------------------------------------------------- #
# 19.2  One encoder, three heads                                               #
# --------------------------------------------------------------------------- #

# ModernBERT-base token splits (see the module docstring for the check).
SENT_TOKENS = ["[CLS]", "a", "Ġcharming", "Ġfilm", "[SEP]"]
PAIR_TOKENS = ["[CLS]", "It", "Ġrains", "[SEP]", "It", "Ġis", "Ġdry", "[SEP]"]
TAG_TOKENS = ["[CLS]", "Peter", "ĠBlack", "burn", "Ġspoke", "[SEP]"]
TAG_WORD_IDS = [None, 0, 1, 1, 2, None]          # Peter, Blackburn, spoke
TAG_WORD_LABELS = ["B-PER", "I-PER", "O"]


def align_first(word_ids, word_labels):
    """d2l.align_token_labels(scheme='first'), with None for -100."""
    out, prev = [], None
    for w in word_ids:
        out.append(word_labels[w] if w is not None and w != prev else None)
        prev = w
    return out


def fig_encoder_heads():
    W, H = 7.75, 6.0
    fig, ax = canvas(W, H)
    ch_h, enc_lo, enc_hi = 0.32, 0.46, 0.8
    top_lo, top_hi = 0.94, 1.26
    head_x = 4.6            # every row's head starts in one column
    head_w = tw("dense, GELU, norm, linear", FS_S) + 0.3
    pooled = dict(accent=SLATE)

    def encoder_row(y0, tokens, title, special=None):
        ax.text(0.0, y0 + top_hi + 0.28, title, fontsize=FS_T,
                color=MUTED, ha="left", va="center")
        spans = chips(ax, 0.0, y0, [
            (t, dict(accent=ORANGE, lw=1.6)) if special and i in special
            else t for i, t in enumerate(tokens)], h=ch_h, pad=0.06)
        xa, xb = spans[0][0], spans[-1][1]
        for a, b in spans:
            fm.arrow(ax, ((a + b) / 2, y0 + ch_h + 0.02),
                     ((a + b) / 2, y0 + enc_lo - 0.02), color=INK, lw=1.1,
                     mut=8)
        rbox(ax, xa, y0 + enc_lo, xb - xa, enc_hi - enc_lo, accent=BLUE,
             lw=1.6)
        ax.text((xa + xb) / 2, y0 + (enc_lo + enc_hi) / 2,
                "pretrained encoder", fontsize=FS_S,
                ha="center", va="center_baseline", zorder=5)
        for a, b in spans:
            fm.arrow(ax, ((a + b) / 2, y0 + enc_hi + 0.02),
                     ((a + b) / 2, y0 + top_lo - 0.02), color=BLUE.base,
                     lw=1.1, mut=8)
        return spans

    def pooled_head(y0, spans, n_logits):
        xa, xb = spans[0][0], spans[-1][1]
        rbox(ax, xa, y0 + top_lo, xb - xa, top_hi - top_lo, **pooled,
             lw=1.4)
        ym = y0 + (top_lo + top_hi) / 2
        ax.text((xa + xb) / 2, ym, "mean over tokens", fontsize=FS_S,
                ha="center", va="center_baseline", zorder=5)
        flow(ax, (xb + 0.04, ym), (head_x - 0.04, ym))
        rbox(ax, head_x, y0 + top_lo, head_w, top_hi - top_lo,
             accent=GREEN, lw=1.4)
        ax.text(head_x + head_w / 2, ym, "dense, GELU, norm, linear",
                fontsize=FS_S, ha="center", va="center_baseline", zorder=5)
        flow(ax, (head_x + head_w + 0.04, ym), (head_x + head_w + 0.34, ym))
        ax.text(head_x + head_w + 0.4, ym, f"{n_logits} logits",
                fontsize=FS_S, ha="left", va="center_baseline")

    # (1) sentence, (2) pair, (3) tagging, from the top down
    y_sent, y_pair, y_tag = 4.42, 2.38, 0.0
    sp = encoder_row(y_sent, SENT_TOKENS, "SENTENCE  ·  SST-2")
    pooled_head(y_sent, sp, 2)
    sp = encoder_row(y_pair, PAIR_TOKENS, "PAIR  ·  MNLI", special={3})
    pooled_head(y_pair, sp, 3)
    a, b = sp[3]
    ax.text(sp[-1][1] + 0.2, y_pair + ch_h / 2,
            "no segment ids: the separator\nalone marks the second text",
            fontsize=FS_S, color=ORANGE.dark, ha="left", va="center",
            linespacing=1.0)

    sp = encoder_row(y_tag, TAG_TOKENS, "")
    xa, xb = sp[0][0], sp[-1][1]
    rbox(ax, xa, y_tag + top_lo, xb - xa, top_hi - top_lo, accent=GREEN,
         lw=1.4)
    ax.text((xa + xb) / 2, y_tag + (top_lo + top_hi) / 2,
            "dense, GELU, norm, linear at every token", fontsize=FS_S,
            ha="center", va="center_baseline", zorder=5)
    labels = align_first(TAG_WORD_IDS, TAG_WORD_LABELS)
    for (a, b), lab in zip(sp, labels):
        fm.arrow(ax, ((a + b) / 2, y_tag + top_hi + 0.02),
                 ((a + b) / 2, y_tag + top_hi + 0.2), color=GREEN.base,
                 lw=1.1, mut=8)
        ax.text((a + b) / 2, y_tag + top_hi + 0.24,
                lab or "\u2212100", fontsize=FS_T, family=MONO,
                color=INK if lab else FAINT, ha="center", va="bottom")
    ax.text(0.0, y_tag + top_hi + 0.72, "TAGGING  ·  CONLL-2003",
            fontsize=FS_T, color=MUTED, ha="left", va="center")
    ax.text(xb + 0.3, y_tag + (enc_lo + top_hi) / 2,
            "9 logits per token;\na word is read at its first token;\n"
            "\u2212100: not scored",
            fontsize=FS_S, color=INK, ha="left", va="center",
            linespacing=1.0)
    fl.save(fig, f"{PREFIX}-encoder-heads")


# --------------------------------------------------------------------------- #
# 19.4  Product quantization and the inverted file                             #
# --------------------------------------------------------------------------- #

# The hand case of retrieval.md (cell retrieval-pq-check): m = 4, M = 2, K = 2.
PQ_CODEBOOKS = np.array([[[1., 0.], [0., 1.]], [[1., 1.], [-1., 0.]]])
PQ_X = np.array([0.9, 0.2, -0.8, 0.1])
PQ_Y = np.array([1., 1., 0., 0.])


def pq_hand_case():
    """Code, reconstruction, look-up table and distances of the hand case,
    checked against the numbers the text quotes."""
    xb, yb = PQ_X.reshape(2, 2), PQ_Y.reshape(2, 2)
    d2x = ((xb[:, None, :] - PQ_CODEBOOKS) ** 2).sum(-1)   # (M, K)
    code = d2x.argmin(1)
    x_hat = PQ_CODEBOOKS[np.arange(2), code].ravel()
    table = ((yb[:, None, :] - PQ_CODEBOOKS) ** 2).sum(-1)  # T_j[i]
    adc = table[np.arange(2), code].sum()
    exact = ((PQ_Y - PQ_X) ** 2).sum()
    assert code.tolist() == [0, 1] and x_hat.tolist() == [1, 0, -1, 0]
    assert np.allclose(d2x, [[0.05, 1.45], [4.05, 0.05]])
    assert np.allclose(table, [[1, 1], [2, 1]]) and adc == 2
    assert abs(exact - 1.30) < 1e-9
    return d2x, code, x_hat, table, adc, exact


def kmeans2d(X, k, rng, iters=50):
    """k-means++ seeds, then Lloyd's iterations (as d2l.kmeans does)."""
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = ((X[:, None] - np.array(C)[None]) ** 2).sum(-1).min(1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[a == j].mean(0) for j in range(k)])
    return C


def voronoi_cell(C, i, box):
    """The cell of centroid i inside the box, by clipping the box polygon
    with the half-planes |x - c_i|^2 <= |x - c_j|^2."""
    (x0, x1), (y0, y1) = box
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for j in range(len(C)):
        if j == i:
            continue
        n = C[j] - C[i]                       # keep n.x <= b
        b = (C[j] @ C[j] - C[i] @ C[i]) / 2
        out = []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            fp, fq = n @ p - b, n @ q - b
            if fp <= 0:
                out.append(p)
            if fp * fq < 0:
                t = fp / (fp - fq)
                out.append(tuple(np.asarray(p) + t * (np.asarray(q)
                                                      - np.asarray(p))))
        poly = out
    return np.array(poly)


def ivf_scene():
    """Points, 8 k-means cells, and a question whose exact nearest neighbor
    lies in a cell that n_probe = 2 does not scan."""
    rng = np.random.default_rng(19)
    mu = rng.uniform(0.15, 0.85, size=(6, 2))
    X = np.clip(mu[rng.integers(6, size=260)]
                + 0.09 * rng.standard_normal((260, 2)), 0.02, 0.98)
    C = kmeans2d(X, 8, np.random.default_rng(0))
    best = None
    for qx in np.linspace(0.1, 0.9, 81):
        for qy in np.linspace(0.1, 0.9, 81):
            q = np.array([qx, qy])
            probed = np.argsort(((C - q) ** 2).sum(1))[:2]
            dX = ((X - q) ** 2).sum(1)
            nn = dX.argmin()
            cell = ((C - X[nn]) ** 2).sum(1).argmin()
            # missed, and the neighbor clearly nearer than any probed point
            inside = np.isin(((X[:, None] - C[None]) ** 2).sum(-1)
                             .argmin(1), probed)
            margin = dX[inside].min() - dX[nn]
            gap = np.sqrt(dX[nn])
            if cell not in probed and margin > 0.004 and gap > 0.06:
                score = ((q - 0.5) ** 2).sum()
                if best is None or score < best[0]:
                    best = (score, q, probed, nn)
    assert best is not None
    _, q, probed, nn = best
    return X, C, q, probed, nn


def fig_ivf_pq():
    W, H = 9.0, 4.7
    fig, ax = canvas(W, H, y0=0.32)
    d2x, code, x_hat, table, adc, exact = pq_hand_case()
    acc = [BLUE, ORANGE]
    lab_x, col0, col_w, cw, rh = 0.0, 1.05, 1.72, 0.46, 0.36
    cols = [col0, col0 + col_w + 0.15]

    def num(v):
        return (f"{v:g}" if v != int(v) else f"{int(v)}").replace("-",
                                                                 "−")

    def row_label(y, text, **kw):
        ax.text(lab_x, y + rh / 2, text, fontsize=FS, ha="left",
                va="center", **kw)

    # ---- encode: the vector, its two blocks, the codebooks, the code ------
    y = 4.05
    ax.text(cols[0] + 1.0 * cw, y + rh + 0.12,
            r"block $j = 1$", fontsize=FS_S, color=BLUE.dark,
            ha="center", va="bottom")
    ax.text(cols[1] + 1.0 * cw, y + rh + 0.12,
            r"block $j = 2$", fontsize=FS_S, color=ORANGE.dark,
            ha="center", va="bottom")
    row_label(y, r"$\mathbf{x}$")
    for j in range(2):
        for k in range(2):
            v = PQ_X[2 * j + k]
            ax.add_patch(Rectangle((cols[j] + k * cw, y), cw, rh,
                                   facecolor=acc[j].tint,
                                   edgecolor=acc[j].base, lw=1.2))
            ax.text(cols[j] + (k + 0.5) * cw, y + rh / 2, num(v),
                    fontsize=FS_S, ha="center", va="center_baseline")
    # the codebooks: centroid, its squared distance to the block
    for i in range(2):
        yc = y - (i + 1) * (rh + 0.1) - 0.05
        row_label(yc, rf"$\mathbf{{c}}_{{j,{i}}}$")
        for j in range(2):
            c = PQ_CODEBOOKS[j, i]
            chosen = code[j] == i
            rbox(ax, cols[j], yc, 2 * cw, rh,
                 accent=acc[j] if chosen else None,
                 edge=None if chosen else FAINT, lw=1.8 if chosen else 1.0,
                 r=0.05)
            ax.text(cols[j] + cw, yc + rh / 2,
                    "(" + ", ".join(num(t) for t in c) + ")", fontsize=FS_S,
                    ha="center", va="center_baseline", zorder=5)
            ax.text(cols[j] + 2 * cw + 0.1, yc + rh / 2,
                    f"{d2x[j, i]:.2f}", fontsize=FS_S,
                    color=INK if chosen else MUTED, ha="left",
                    va="center_baseline")
    yk = y - 3 * (rh + 0.1) - 0.1
    row_label(yk, r"code $i_j$")
    for j in range(2):
        ax.add_patch(Rectangle((cols[j] + cw / 2, yk), cw, rh,
                               facecolor=acc[j].base, edgecolor=acc[j].base,
                               lw=1.2))
        ax.text(cols[j] + cw, yk + rh / 2, str(code[j]), fontsize=FS_S,
                color=T.PAPER, family=MONO, ha="center",
                va="center_baseline")
    ax.text(cols[1] + 2 * cw + 0.1, yk + rh / 2,
            r"$M\;\log_2 K$ bits", fontsize=FS_S, color=MUTED, ha="left",
            va="center")

    # ---- search: the question's table, the distance as a sum --------------
    ys = yk - 0.42
    ax.plot([lab_x, cols[1] + 2 * cw + 1.0], [ys, ys], color=T.HAIRLINE,
            lw=1.2)
    yq = ys - 0.18 - rh
    row_label(yq, r"$\mathbf{y}$")
    for j in range(2):
        for k in range(2):
            ax.add_patch(Rectangle((cols[j] + k * cw, yq), cw, rh,
                                   facecolor=T.PANEL, edgecolor=MUTED,
                                   lw=1.0))
            ax.text(cols[j] + (k + 0.5) * cw, yq + rh / 2,
                    num(PQ_Y[2 * j + k]), fontsize=FS_S, ha="center",
                    va="center_baseline")
    yt = yq - rh - 0.12
    row_label(yt, r"$T_j[i]$")
    for j in range(2):
        for i in range(2):
            chosen = code[j] == i
            ax.add_patch(Rectangle(
                (cols[j] + i * cw, yt), cw, rh,
                facecolor=acc[j].tint if chosen else T.PAPER,
                edgecolor=acc[j].base if chosen else FAINT,
                lw=1.8 if chosen else 1.0, zorder=3 if chosen else 2))
            ax.text(cols[j] + (i + 0.5) * cw, yt + rh / 2,
                    num(table[j, i]), fontsize=FS_S, ha="center",
                    va="center_baseline", zorder=5)
            ax.text(cols[j] + (i + 0.5) * cw, yt - 0.05, f"$i = {i}$",
                    fontsize=12.5, color=MUTED, ha="center", va="top")
    ax.text(lab_x, yt - 0.62,
            rf"$\|\mathbf{{y}} - \hat{{\mathbf{{x}}}}\|^2 = T_1[0] + T_2[1]"
            rf" = 1 + 1 = {adc:g}$,   exact $\|\mathbf{{y}} - \mathbf{{x}}"
            rf"\|^2 = {exact:.2f}$", fontsize=FS_S, ha="left", va="center")

    # ---- the inverted file -------------------------------------------------
    X, C, q, probed, nn = ivf_scene()
    side, ox, oy = 3.3, W - 3.3, 0.85
    box = ((0, 1), (0, 1))
    to = lambda P: np.c_[ox + side * P[..., 0], oy + side * P[..., 1]]
    for i in range(len(C)):
        poly = to(voronoi_cell(C, i, box))
        ax.add_patch(plt.Polygon(poly, closed=True,
                                 facecolor=BLUE.tint if i in probed
                                 else T.PAPER, edgecolor=MUTED, lw=1.0,
                                 zorder=1))
    cell_of = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    P = to(X)
    inside = np.isin(cell_of, probed)
    ax.scatter(P[~inside, 0], P[~inside, 1], s=7, color=FAINT, lw=0,
               zorder=2)
    ax.scatter(P[inside, 0], P[inside, 1], s=9, color=BLUE.base, lw=0,
               zorder=2)
    Cp = to(C)
    ax.scatter(Cp[:, 0], Cp[:, 1], s=46, marker="D", color=INK,
               edgecolor=T.PAPER, lw=0.8, zorder=4)
    qp, np_ = to(q[None])[0], to(X[nn][None])[0]
    ax.scatter([np_[0]], [np_[1]], s=110, facecolor="none",
               edgecolor=ORANGE.base, lw=2.0, zorder=5)
    ax.scatter([qp[0]], [qp[1]], s=150, marker="*", color=RED.base,
               edgecolor=T.PAPER, lw=0.6, zorder=6)
    # labels pushed away from the segment between the question and the
    # neighbor it misses
    u = (qp - np_) / np.linalg.norm(qp - np_)
    lq, ln = qp + 0.2 * u, np_ - 0.22 * u
    ax.text(lq[0], lq[1], r"$\mathbf{y}$", fontsize=FS, color=RED.dark,
            ha="left" if u[0] > 0 else "right", va="center", zorder=6)
    ax.text(ln[0], ln[1], "nearest point,\nnot scanned", fontsize=FS_S,
            color=ORANGE.dark, ha="left" if u[0] < 0 else "right",
            va="center", zorder=6, linespacing=1.0,
            bbox=dict(facecolor=T.PAPER, edgecolor="none", pad=1.0))
    ax.text(ox + side / 2, oy - 0.12,
            r"$n_{\mathrm{list}} = 8$ cells, $n_{\mathrm{probe}} = 2$ scanned",
            fontsize=FS_S, ha="center", va="top")
    fl.save(fig, f"{PREFIX}-ivf-pq")
    return q, X[nn]


# --------------------------------------------------------------------------- #
# 19.5  The stages of the RAG pipeline, their measurements and failures        #
# --------------------------------------------------------------------------- #

# (stage, what it decides, accent, dashed, measured by, failure classes);
# the rows of tab_lm_rag_stages and tab_lm_rag_fixes in rag.md
RAG_STAGES = [
    ("query", "the question, or its rewrite", SLATE, False,
     "the stages after it", []),
    ("first-stage retrieval", "BM25 and dense rankings, fused (RRF)", BLUE,
     False, "Recall@$k$", ["not retrieved"]),
    ("reranking (optional)", "a cross-encoder reorders the top", BLUE, True,
     "Recall@$k$", ["ranked low"]),
    ("context assembly", "order, token budget, near-duplicates", BLUE,
     False, "coverage at a budget,\ncov@$T$", None),
    ("grounded generation", "an answer citing [1], [2], ..., or abstain",
     GREEN, False, "citation precision and recall,\nabstention rates",
     ["ignored", "unsupported", "false answer", "false abstention"]),
    ("judge", "correct? faithful? (checked on labels)", PURPLE, False,
     "accuracy, faithfulness", []),
]


def fig_rag_stages():
    W, H = 9.1, 5.75
    fig, ax = canvas(W, H)
    bx, bw, bh, pitch = 0.0, 3.55, 0.66, 0.84
    mx, fx = 3.85, 6.35
    y_head = H - 0.14
    for x, t in ((bx, "STAGE"), (mx, "MEASURED BY"), (fx, "FAILURE CLASS")):
        ax.text(x, y_head, t, fontsize=FS_T, color=MUTED, ha="left",
                va="top")
    y = y_head - 0.42
    centers = []
    for name, what, acc, dash, meas, fails in RAG_STAGES:
        tall = fails is not None and len(fails) > 2
        h = bh + (0.14 if tall else 0)
        y_lo = y - h
        rbox(ax, bx, y_lo, bw, h, accent=acc, dash=dash, lw=1.6, r=0.08)
        yc = y_lo + h / 2
        ax.text(bx + 0.14, yc + 0.13, name, fontsize=FS, ha="left",
                va="center_baseline", zorder=5)
        ax.text(bx + 0.14, yc - 0.16, what, fontsize=FS_S, color=MUTED,
                ha="left", va="center_baseline", zorder=5)
        ax.text(mx, yc, meas, fontsize=FS_S, ha="left", va="center",
                linespacing=1.0)
        if fails and fails[0] == "ranked low":   # drawn below, spanning
            pass
        elif fails:
            per_row = 2 if tall else 1
            for i, f in enumerate(fails):
                r, c = divmod(i, per_row)
                fw = tw(f, FS_S) + 0.2
                col0 = max(tw(g, FS_S) for g in fails[::per_row]) + 0.2
                px = fx + c * (col0 + 0.12)
                py = yc + (0.19 if tall else 0) - r * 0.38 - 0.15
                rbox(ax, px, py, fw, 0.3, accent=RED, lw=1.2, r=0.12)
                ax.text(px + fw / 2, py + 0.15, f, fontsize=FS_S,
                        color=RED.dark, ha="center", va="center_baseline",
                        zorder=5)
        centers.append((y, y_lo, yc))
        y = y_lo - (pitch - bh)
    # arrows down the chain
    for (_, lo, _), (hi, _, _) in zip(centers, centers[1:]):
        flow(ax, (bx + bw / 2, lo - 0.02), (bx + bw / 2, hi + 0.02),
             lw=1.6, mut=12)
    # "ranked low" blames reranking and assembly: one bracket over both
    (_, _, c3), (_, _, c4) = centers[2], centers[3]
    ax.plot([fx + 0.08, fx, fx, fx + 0.08], [c3, c3, c4, c4],
            color=RED.base, lw=1.2)
    f = "ranked low"
    fw = tw(f, FS_S) + 0.2
    ax.plot([fx, fx + 0.12], [(c3 + c4) / 2] * 2, color=RED.base, lw=1.2)
    rbox(ax, fx + 0.12, (c3 + c4) / 2 - 0.15, fw, 0.3, accent=RED, lw=1.2,
         r=0.12)
    ax.text(fx + 0.12 + fw / 2, (c3 + c4) / 2, f, fontsize=FS_S,
            color=RED.dark, ha="center", va="center_baseline", zorder=5)
    assert centers[-1][1] > 0.02, "rows run off the canvas"
    fl.save(fig, f"{PREFIX}-rag-stages")


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

FIGURES = [
    fig_sgns_pmi,
    fig_encoder_heads,
    fig_ivf_pq,
    fig_rag_stages,
]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):40s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
