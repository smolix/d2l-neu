# Chapter 7 "Convolutional Neural Networks" (the basics) — Cross-Cutting Landscape Review

**Reviewer role:** Big-picture / structure / missing-topics (whole chapter).
**Date:** 2026-06-14. **Repo:** `d2l-neu`. **Primary lens:** PyTorch 2.x (4-fw book: pytorch / jax / tensorflow / mxnet).
**Scope reminder:** §7 is the *foundations* chapter for convolutions: it sits between
the MLP chapter (Ch. 5) and the modern-architecture chapter (Ch. 8, modern CNNs). It
teaches the convolution operation and the first full ConvNet (LeNet). It is **not** the
modern-architectures chapter (that is §8), and vision transformers are a separate,
not-yet-written section.

This is a **landscape review, not a change spec.** It sets direction and priority; the
two section-level reports own the line edits:
- `01-convolution-mechanics.md` — why-conv, conv-layer, padding-and-strides.
- `02-composition-and-lenet.md` — channels, pooling, lenet (+ the section arc).

---

## 1. Executive read on the chapter's current shape

The chapter is **6 content sections**, in this order:

1. `why-conv.md` — **From Fully-Connected Layers to Convolutions** (translation
   equivariance + locality reduce a dense weight tensor to a shared local kernel).
2. `conv-layer.md` — **Convolutions for Images** (cross-correlation, edge-detection
   demo, learning a kernel from data, feature map, receptive field).
3. `padding-and-strides.md` — **Padding and Stride** (output-size arithmetic).
4. `channels.md` — **Multiple Input and Output Channels** (channel arithmetic, the 1x1
   convolution as a per-pixel MLP).
5. `pooling.md` — **Pooling** (max/avg, downsampling, invariance).
6. `lenet.md` — **Convolutional Neural Networks (LeNet)** (the first full ConvNet on
   Fashion-MNIST).

**Level and freshness.** Overall grade **B+**. The *code* is correct and teaches (the
`corr2d` for-loop primitive, the edge-detection finite-difference, the learn-a-kernel
SGD recovery of `[1, -1]` across all four tabs, the LeNet shape trace) and should be
kept. The chapter is **closer to the bar than a per-file read suggests**, because
conv-layer already carries equivariance, receptive-field, and feature-map material. The
debt is **not the code**; it is four things: one correctness/vocabulary contradiction,
one missing derivation, a modern narrative that lives only in the slides, and a
chapter-wide figure-style violation.

**The four real gaps (priority order):**

1. **`why-conv` contradicts itself on its own central concept.** The section *derives*
   translation **equivariance** (a shifted input yields a shifted feature map) but calls
   it **invariance** in the section title, prose, and summary. The desiderata box gets
   it right; the derivation and headings do not. This is the single most important fix
   in the chapter because it is the chapter's central idea, and the equivariance (here)
   vs. invariance (realized by pooling, §7.5) distinction is what the whole part hangs
   on.
2. **The output-size formula is asserted, never derived** — in the one file whose entire
   job is that formula. `padding-and-strides` states `floor((n-k+p+s)/s)+1` without the
   count-the-kernel-placements argument, and uses only legacy static SVGs where the
   canonical Dumoulin & Visin convolution-arithmetic visualizations should anchor the
   teaching. `Dumoulin.Visin.2016` is already in the bib but uncited.
3. **The modern story and the §8 bridge live only in the slides.** The slides already
   carry (a) the honest "strided convolution and global average pooling have largely
   replaced pooling," and (b) the LeNet -> AlexNet "30 years of progress" substitution
   table and the conv-encoder + head template. A *book reader* never sees either. These
   are promotion problems (slide -> prose), the cheapest high-value wins in the chapter.
4. **There is no chapter-7 figure generator, and zero `mdl-cnn-*` figures.** Every
   illustrative figure in §7 is a legacy upstream asset, which violates CLAUDE.md's
   one-house-style-per-chapter rule chapter-wide. (The Waldo JPEGs and the
   Hubel-Wiesel/`field-visual.png` photographic reference images are correctly exempt.)
   This is the largest *new-asset* item and gates several figure-led teaching fixes.

**Overall:** solidly at the bar for *mechanics*; the work is prose promotions, two
derivations, a handful of forward-pointers, four new bib entries, and standing up a
house-style figure generator. No reorder, no code rewrites.

---

## 2. How the best current programs/resources teach this material (2026)

- **Stanford CS231n** (Convolutional Networks notes). The canonical treatment: motivates
  parameter sharing + locality, derives the output-size arithmetic, and makes the
  receptive field central. d2l matches the spirit but skips the derivation.
- **Dumoulin & Visin, "A guide to convolution arithmetic for deep learning" (2016).**
  The gold-standard *animations* for padding/stride/dilation and the output-size formula.
  This is exactly what `padding-and-strides` should lead with (as a multi-panel figure
  in the house style, since slides cannot animate).
- **distill.pub feature visualization (Olah et al. 2017) + Zeiler & Fergus (2014).** The
  modern, honest "what does a conv layer learn" story (edges -> textures -> parts),
  which §7's receptive-field material gestures at but stops at 1959-1987 biology.
- **Springenberg et al., "Striving for Simplicity: The All Convolutional Net" (2015).**
  The primary source for "strided convolution can replace max-pooling with no accuracy
  loss" — the honest modern framing the pooling section needs. **Not currently in
  `d2l.bib`; must be added.**
- **LeCun et al. (1998), LeNet.** Already cited; the framing to strengthen is "this is
  the template AlexNet scaled up" (the §8 bridge).
- **Effective receptive field (Luo et al. 2016).** The result that the *effective* RF is
  Gaussian and grows like sqrt(L), not the full theoretical box — a high-value upgrade to
  the RF subsection.

Net finding: the elite treatments **derive the arithmetic, lead with the
convolution-arithmetic animations, make the receptive field and feature hierarchy
central, and are honest that pooling has receded.** d2l has all the pieces; several are
trapped in the slides or asserted without derivation.

---

## 3. Missing-topics / weak-spot audit (each with a verdict)

Legend: **FIX** (correctness) / **DERIVE** (present but asserted) / **PROMOTE**
(exists in slides, not prose) / **ADD** (genuinely missing) / **FIGURE** (needs a
house-style asset).

| # | Topic | Verdict | Home / rationale |
|---|---|---|---|
| 1 | equivariance vs. invariance | **FIX** | `why-conv`: rename the section, fix prose + summary, add a one-line shift-operator box. The chapter's central idea. |
| 2 | output-size formula | **DERIVE** | `padding-and-strides`: count kernel placements (no-pad -> +p -> +stride); cite `Dumoulin.Visin.2016`; add the integer/floor caveat. |
| 3 | convolution-arithmetic figures | **FIGURE** | `padding-and-strides`: house-style `mdl-cnn-pad` / `mdl-cnn-stride` panels replacing the legacy SVGs. |
| 4 | receptive field + **effective** RF | **DERIVE/ADD** | `conv-layer`: add the closed form (2L+1 for stacked 3x3) and the effective-RF result (Gaussian, ~sqrt(L)); cite `Luo.Li.Urtasun.ea.2016`. |
| 5 | pooling vs. strided conv (modern) | **PROMOTE** | `pooling`: add a "Pooling versus strided convolution" subsection (All-Conv finding; GAP as the modern head; ConvNeXt's no-max-pool stem). Cite `Springenberg.ea.2015` (**add to bib**). Soften the flat "max > avg" claim to historical framing. |
| 6 | LeNet -> §8 bridge | **PROMOTE** | `lenet`: build the substitution map in prose (sigmoid->ReLU, avg-pool->strided-conv, none->BatchNorm, dense head->GAP) with section pointers; quantify the flatten bottleneck (400->120 = 48k weights) and forward-point GAP. |
| 7 | 1x1 convolution lineage | **PROMOTE** | `channels`: promote the 1x1 -> ResNet-bottleneck / MobileNet-pointwise lineage from slide to prose; name grouped + depthwise + depthwise-separable where the text already alludes to block-diagonal mixing. (1x1 is teased in `conv-layer`, fully treated in `channels` — keep that handoff.) |
| 8 | feature visualization | **ADD/FIGURE** | `conv-layer`: upgrade the dated RF/feature-viz note with a modern edges->textures->parts forward-point (Zeiler-Fergus deconv, distill.pub) + a new `mdl-cnn-feature-hierarchy` figure. |
| 9 | dilated / atrous convolution | **ADD (pointer)** | `padding-and-strides` or `conv-layer`: a short forward-pointer (exponential RF growth); cite `Yu.Koltun.2016`. Paragraph, not implementation. |
| 10 | depthwise-separable convolution | **ADD (pointer)** | `channels`: forward-pointer (MobileNet/Xception); cite `Howard.ea.2017`, `Chollet.2017`. Paragraph, not implementation. |
| 11 | two 3x3 = one 5x5 (VGG synthesis) | **PROMOTE/FIGURE** | `pooling`/`conv-layer`: the receptive-field + parameter-count argument exists in pieces but never as prose; optional `mdl-cnn-stacked-vs-large-kernel` figure. Sets up §8 VGG. |
| 12 | chapter-7 figure generator | **FIGURE (foundational)** | Stand up `tools/gen_mdl_cnn_figures.py` (importing the shared `gen_mdl_figures` style) + the `mdl-cnn-*` set: `mlp-to-conv`, `correlation`, `pad`, `stride`, `reuse`, `receptive-field` (theoretical box + Gaussian effective blob), optional `dilation`, `feature-hierarchy`, `pool-vs-strided`. Gates items 3, 5, 8, 11. |

**Four new bib entries required:** `Luo.Li.Urtasun.ea.2016` (effective RF),
`Yu.Koltun.2016` (dilated conv), `Howard.Zhu.Chen.ea.2017` (MobileNet),
`Chollet.2017` (Xception), plus `Springenberg.ea.2015` (All-Conv) for item 5.

---

## 4. Restructure / reorder — recommendation

**Do NOT reorder or split.** The six-file progression (motivation -> operation ->
arithmetic -> channels -> pooling -> first net) is canonical and correct, and matches
CS231n's own sequence. The fixes are **derivations + promotions + a figure generator +
the §8 bridge**, not surgery.

Two handshakes to keep consistent across the two section reports:
- **1x1 convolution:** teased in `conv-layer`, fully treated in `channels` — keep the
  teaser light and the payoff in `channels`.
- **equivariance/invariance:** equivariance is established in `why-conv`; the
  *invariance* payoff is realized in `pooling`. Use the same vocabulary in both so the
  loop closes (this is why item 1 must be fixed first).

---

## 5. Prioritized work order

**FIX (correctness, do first):**
1. equivariance vs. invariance across `why-conv` (item 1).

**DERIVE / PROMOTE (high value, low risk, mostly prose):**
2. output-size derivation in `padding-and-strides` (item 2).
3. pooling-vs-strided modern subsection + soften max>avg (item 5).
4. LeNet -> §8 bridge in prose (item 6).
5. 1x1 lineage + grouped/depthwise naming in `channels` (item 7).
6. receptive-field + effective-RF upgrade (item 4).

**FIGURE (foundational, unblocks figure-led fixes):**
7. stand up `tools/gen_mdl_cnn_figures.py` + the `mdl-cnn-*` set (item 12), then swap in
   items 3, 8, 11 figures.

**ADD (forward-pointers, cheap):**
8. dilated conv, depthwise-separable, feature-visualization pointers (items 9, 10, 8).

**Bib:** add the five entries above.

---

## 6. Dispatch manifest

Line-edit specs and per-file dispatch live in the two section reports; this is the
consolidated work order.

| WID | Work item | File(s) | New assets | Code/notebook | Refs | Depends on |
|---|---|---|---|---|---|---|
| 7-W1 | Fix equivariance/invariance | `why-conv.md` | — | none (prose) | CS231n | — |
| 7-W2 | Derive output-size formula | `padding-and-strides.md` | `mdl-cnn-pad`, `mdl-cnn-stride` | none | Dumoulin.Visin.2016 | 7-W7 (figures) |
| 7-W3 | Pooling-vs-strided + soften max>avg | `pooling.md` | optional `mdl-cnn-pool-vs-strided` | none | Springenberg.ea.2015 (add to bib) | 7-W7 |
| 7-W4 | LeNet -> §8 bridge in prose | `lenet.md` | — | none | LeCun.1998 | — |
| 7-W5 | 1x1 lineage + grouped/depthwise naming | `channels.md` | — | none | Howard.ea.2017, Chollet.2017 | — |
| 7-W6 | RF + effective-RF upgrade; feature-viz | `conv-layer.md` | `mdl-cnn-feature-hierarchy` | none | Luo.ea.2016, Olah.2017, Zeiler.Fergus.2014 | 7-W7 |
| 7-W7 | **Stand up CNN figure generator** | `tools/gen_mdl_cnn_figures.py` | `mdl-cnn-*` set | `make figures` (CPU, byte-idempotent) | — | — (do early; gates W2/W3/W6) |
| 7-W8 | Add 5 bib entries | `d2l.bib` | — | none | — | — |

All work is **no-GPU** (prose, figures, slide->prose promotion). No notebook
re-execution is required for §7; the existing teaching code stays.

---

### Sources
- Stanford [CS231n](https://cs231n.stanford.edu/) Convolutional Networks notes.
- Dumoulin & Visin, [A guide to convolution arithmetic](https://arxiv.org/abs/1603.07285) (2016).
- Springenberg et al., [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) (2015).
- Luo et al., [Understanding the Effective Receptive Field](https://arxiv.org/abs/1701.04128) (2016).
- Yu & Koltun, [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) (2016).
- Howard et al., [MobileNets](https://arxiv.org/abs/1704.04861) (2017); Chollet, [Xception](https://arxiv.org/abs/1610.02357) (2017).
- Olah et al., [Feature Visualization](https://distill.pub/2017/feature-visualization/) (distill.pub, 2017); Zeiler & Fergus, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) (2014).
- LeCun et al., Gradient-Based Learning Applied to Document Recognition (1998).
