# Chapter 8 "Modern Convolutional Neural Networks" — Cross-Cutting Landscape Review

**Reviewer role:** Big-picture / structure / missing-topics / state-of-the-art (whole chapter).
**Date:** 2026-06-14. **Repo:** `d2l-neu`. **Primary lens:** PyTorch 2.x (4-fw book: pytorch / jax / tensorflow / mxnet).
**Scope reminder:** §8 is the *modern convolutional architectures* chapter, between the
CNN-basics chapter (Ch. 7) and the RNN/attention chapters. It is **pre-transformer for
vision by design**: ViT / DeiT / Swin / attention-as-backbone belong to a **separate,
not-yet-written** section. **ConvNeXt is in scope** (it is a pure ConvNet that imports
transformer-era *design choices*) and is the natural capstone + the clean handoff point
to that future section.

This is a **landscape review, not a change spec.** The two section-level reports own the
detail:
- `01-existing-content-review.md` — per-page keep/fix/cut/add for the 8 current pages.
- `02-sota-gaps-and-roadmap.md` — obsolete / missing / new notebooks / full roadmap + an
  8-row dispatch manifest with primary-source citations and verified numbers.

---

## 1. Executive read on the chapter's current shape

The chapter is **8 content sections**: `alexnet`, `vgg`, `nin`, `googlenet`,
`batch-norm`, `resnet`, `densenet`, `cnn-design`.

**Level and freshness.** The existing pages are **mostly KEEP**, and the code genuinely
computes (real shape pyramids, learned gamma/beta, channel arithmetic) with no
wall-of-matplotlib anywhere. But the chapter teaches to a **~2020 cutoff**, and two
things hold it back:

1. **One real correctness bug (highest-value single fix): `batch-norm`'s "why it
   works."** The page still leads with the discredited **internal-covariate-shift**
   story, cites Santurkar et al. (2018) but never states its actual finding (BN works by
   **smoothing the loss landscape** / lowering the loss's effective Lipschitz constant).
   This is the one defect that is *wrong*, not merely dated. It is prose + 2 slides on
   one page, no GPU, with a crisp well-sourced replacement. **Do it first.**
2. **The 2020-2026 ConvNet story is absent.** The chapter ends at RegNet design spaces
   and never tells the modern arc: efficient primitives (depthwise-separable, inverted
   residual, SE), the training-recipe revolution ("the recipe moves accuracy as much as
   the architecture"), and ConvNeXt (the transformer-informed modern ConvNet that
   *matches* Swin). Without these, a 2026 student leaves thinking the ConvNet story
   stopped at ResNet/RegNet.

Everything else is **one paragraph of currency per page**, along a single through-line:
**"architecture matters less than scale + training recipe + design space."** Almost all
of it is **no-GPU prose/slide work**; only the `googlenet` trim needs a 4-framework
re-run.

---

## 2. How the best programs teach this + the 2026 SOTA landscape

The elite treatments (CS231n; timm/`pytorch-image-models`; the ConvNeXt, RegNet, and
"ResNet strikes back" papers) converge on: **the residual block is the anchor; the
training recipe is a first-class lever; and the modern ConvNet (ConvNeXt) is the bridge
to transformers.** The primary sources that should ground the additions:

- **ConvNeXt** (Liu et al. 2022, "A ConvNet for the 2020s") + **ConvNeXt V2** (2023) —
  the modernized-ConvNet capstone; a pure ConvNet matches Swin.
- **RegNet / "Designing Network Design Spaces"** (Radosavovic et al. 2020) — already the
  basis of `cnn-design`; assess depth and make it the roadmap on-ramp.
- **EfficientNet** (Tan & Le 2019) + **V2** (2021) — compound scaling.
- **"ResNet strikes back"** (Wightman et al. 2021) and **Bello et al. "Revisiting
  ResNets"** (2021) — the training recipe (RandAugment, Mixup, CutMix, label smoothing,
  stochastic depth, EMA, cosine schedule, longer training) moves a fixed ResNet-50 from
  ~76% to ~80.4% top-1 with **no architecture change**.
- **Squeeze-and-Excitation** (Hu et al. 2018) + **CBAM** — channel attention in ConvNets.
- **GroupNorm** (Wu & He 2018), and **normalizer-free nets / NFNets** (Brock et al.
  2021) — normalization beyond BatchNorm.
- **MobileNet** (v1-v3), **Xception**, **ShuffleNet** — depthwise-separable + inverted
  residual efficient primitives.
- **He et al. "Bag of Tricks"** (2019) — the training-recipe precursor.

(Numbers were cross-checked in `02-...`; three common web errors are flagged there:
SE-ResNet-50 is 76.71% not 77.62%, NFNet SOTA is 86.5%, and the ResNet-RS decomposition
baselines on ResNet-200.)

---

## 3. Obsolete / didn't stand the test of time (section-level calls)

None of these is a *cut*; they are **reframes** that keep the history while being honest
about 2026 relevance.

| Topic | Verdict | Rationale |
|---|---|---|
| **NiN, GoogLeNet** | **TRIM / REFRAME** (do not expand) | Keep as the *origin* of the 1x1 conv (NiN) and the stem/body/head pattern (GoogLeNet). Multi-branch Inception is a dead *design strategy*; frame both as history, not frontier. `googlenet`: collapse the ~150-line x 4-tab hand-build (channel numbers the prose itself calls "arbitrary") to a block figure + full-net figure + one worked stage. |
| **DenseNet** | **REFRAME** | Keep, honest about memory cost; add one "for 2026" sentence: a road-not-taken whose feature-reuse idea was influential, but additive ResNet blocks won on hardware-efficiency (params != the cost that matters). Sets up the efficiency thread. |
| **BatchNorm "why it works"** | **REWRITE** | The internal-covariate-shift framing is the one *incorrect* item; replace with the loss-landscape-smoothing finding (Santurkar 2018). |
| **`cnn-design` Discussion** | **REFRAME** | The "scalability trumps inductive biases" framing is transformer-triumphalist and aged into a half-truth post-ConvNeXt/MetaFormer. Refresh the dated hardware aside (Hopper -> Blackwell). |
| **AlexNet hardware section** | **TRIM** | The A100/M1/Graviton TFLOPs spec numbers are ~2021-stale; keep the durable cores/power argument, drop the specifics. |
| **ShiftNet name-drop** | **CUT** | Depthwise-separable conv won that race. |
| **"10-epoch SGD smoke test" framing** | **REFRAME** | Understates the architectures and hides that the recipe is a lever; set it up for the new training-recipe notebook. |

---

## 4. Missing topics (ranked) + proposed new notebooks

**Missing concepts, ranked by value:** (1) modern training recipe, (2)
depthwise-separable convolution, (3) inverted residual + linear bottleneck, (4)
Squeeze-and-Excitation, (5) ConvNeXt, (6) normalization beyond BN (BN/LN/IN/GN taxonomy
+ GroupNorm), (7) compound scaling, (8) structural reparameterization / RepVGG, (9)
large-kernel ConvNets, (10) normalizer-free nets, (11) self-supervised pretraining
(pointer only), (12) MetaFormer framing.

**Proposed new notebooks** (all fit the small-compute model: tiny nets on
Fashion-MNIST / CIFAR-scale, minutes on one RTX-4090; ImageNet numbers cited, never
trained; 4-framework feasibility noted in `02-...`):

- **A. `efficient.md`** — depthwise-separable conv (the `1/N + 1/Dk^2` cost arithmetic,
  verified) + inverted residual + linear bottleneck + SE, building a tiny MobileNetV2;
  compound scaling. *Covers missing #2/#3/#4/#7.*
- **B. `training-recipe.md`** — the recipe moves accuracy as much as architecture:
  Mixup/CutMix, label smoothing, stochastic depth, cosine schedule, AdamW, EMA, as one
  ablation table on a fixed small ResNet. *Covers missing #1; this is the conceptual
  unlock for the whole modern story.*
- **C. `convnext.md` (capstone)** — a pure ConvNet modernized toward Swin: build the
  block, train a tiny one, present the 78.8 -> 82.0 modernization roadmap; **the handoff
  slide to vision transformers.** *Covers missing #5; the single highest-value add.*
- **D. `repvgg.md` (optional)** — train a multi-branch block, fold it to a single 3x3
  conv, numerically verify the equivalence. *Covers missing #8; ship as an exercise
  first.*

**Single highest-value addition: the ConvNeXt capstone (C).** It closes the 2020 -> 2022
gap, retroactively justifies the whole chronological arc, trains at small scale, and is
the designed handoff to the future vision-transformer section. Its prerequisite and
close second is the training-recipe notebook (B).

---

## 5. Proposed modern §8 structure (8 -> 11 sections)

`index` (reframe) · `alexnet` · `vgg` · `nin` (reframe) · `googlenet` (reframe) ·
**`batch-norm` (rewrite "why" + expand to a BN/LN/IN/GN normalization taxonomy +
GroupNorm)** · `resnet` (add the architecture-vs-recipe paragraph; promote the bottleneck
from exercise to body) · `densenet` (reframe) · **`efficient` (NEW)** ·
**`training-recipe` (NEW)** · **`convnext` (NEW)** · `cnn-design` (reframe, roadmap
on-ramp).

**Through-line:** AlexNet scale-up -> VGG/NiN blocks -> BN trainability -> ResNet/ResNeXt
depth -> efficient primitives -> training recipe -> ConvNeXt -> RegNet design spaces ->
**handoff** at the end of `convnext.md` into
`chapter_attention-mechanisms-and-transformers/vision-transformer` (`sec_vision-transformer`).
(A NiN + GoogLeNet merge is offered in `02-...` as a length release-valve only.)

---

## 6. Figures

The section **pre-dates the figure house-style: zero `mdl-` figures, no generator** —
all schematics are legacy d2l SVGs plus a few rasters. This is the same
`tools/gen_mdl_cnn_figures.py` generator §7 needs (see `section-7/00`); §8 shares it.

- **Highest-value new figure:** a **BN / LN / IN / GN "what axis is normalized"** N x C x
  H x W diagram on `batch-norm` (anchors the normalization-taxonomy expansion).
- **One genuine defect:** `regnet-fig.png` is a raster but load-bearing (the 4
  empirical-CDF design-space panels) — re-plot as an SVG. `regnet-paper-fig5/7.png`
  appear unused (confirm + drop).
- Keep photographic reproductions (e.g. `filters.png`) as-is.

---

## 7. Dispatch manifest

Per-page line edits live in `01-existing-content-review.md`; the SOTA roadmap, new
notebooks, and the source-cited 8-row manifest live in `02-sota-gaps-and-roadmap.md`.
Consolidated work order:

| WID | Work item | File(s) | Kind | GPU? | Depends on |
|---|---|---|---|---|---|
| 8-W1 | **Rewrite BatchNorm "why it works"** (landscape-smoothing) | `batch-norm.md` | prose + 2 slides | no | — (do first) |
| 8-W2 | Per-page currency reframes (NiN/GoogLeNet/DenseNet/cnn-design/AlexNet) | the 5 pages | prose | no | — |
| 8-W3 | `googlenet` trim (collapse 5-stage hand-build to figure + 1 stage) | `googlenet.md` | prose + code-trim | **yes** (4-fw re-run + re-capture) | — |
| 8-W4 | `resnet` architecture-vs-recipe paragraph + promote bottleneck | `resnet.md` | prose (+ small code) | maybe | — |
| 8-W5 | Expand `batch-norm` to BN/LN/IN/GN taxonomy + GroupNorm | `batch-norm.md` | prose + figure | no | 8-W1, 8-W8 |
| 8-W6 | **NEW `efficient.md`** (depthwise-sep + inverted residual + SE + tiny MobileNetV2 + scaling) | new file | new notebook | **yes** | 8-W8 |
| 8-W7 | **NEW `training-recipe.md`** (ablation table on fixed small ResNet) | new file | new notebook | **yes** | — |
| 8-W8 | **NEW `convnext.md` capstone** + ViT handoff | new file | new notebook + figures | **yes** | 8-W7 (recipe), generator |
| 8-W9 | Stand up / share `tools/gen_mdl_cnn_figures.py`; BN-axes figure; re-plot `regnet-fig` as SVG | `tools/...`, `batch-norm.md`, `cnn-design.md` | figures | no | shared with §7-W7 |
| 8-W10 | Book-wide flags (route to the right owner, not §8-local): `flax.linen` -> NNX, "TensorFlow" -> Keras 3, `Lazy*` as default | many | prose/code | n/a | separate track |
| 8-W11 | Numbering: this section is §8; ensure new files land in `CHAPTER_NUMBERING` in the §5 order | `tools/d2l_preprocess.py`, `_quarto.yml` | config | no | 8-W6/7/8 |

**Sequencing:** 8-W1 (correctness, no-GPU) -> 8-W2/W4/W5 (currency, mostly no-GPU) ->
8-W9 (figure generator) -> 8-W7 (training recipe, the conceptual prerequisite) -> 8-W6 +
8-W8 (efficient + ConvNeXt capstone, GPU) -> 8-W3 (googlenet trim, GPU re-run). The
GPU-bound items (W3/W6/W7/W8) are the only ones needing the 4xGPU box; everything else
is CPU/prose.

---

### Sources
- Liu et al., [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545) (2022); [ConvNeXt V2](https://arxiv.org/abs/2301.00808) (2023).
- Radosavovic et al., [Designing Network Design Spaces (RegNet)](https://arxiv.org/abs/2003.13678) (2020).
- Tan & Le, [EfficientNet](https://arxiv.org/abs/1905.11946) (2019); [EfficientNetV2](https://arxiv.org/abs/2104.00298) (2021).
- Wightman et al., [ResNet strikes back](https://arxiv.org/abs/2110.00476) (2021); Bello et al., [Revisiting ResNets](https://arxiv.org/abs/2103.07579) (2021).
- He et al., [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) (2019).
- Santurkar et al., [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) (2018).
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (2018).
- Wu & He, [Group Normalization](https://arxiv.org/abs/1803.08494) (2018); Brock et al., [High-Performance Large-Scale Image Recognition Without Normalization (NFNets)](https://arxiv.org/abs/2102.06171) (2021).
- Howard et al., [MobileNets](https://arxiv.org/abs/1704.04861) (2017); Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381) (2018); Chollet, [Xception](https://arxiv.org/abs/1610.02357) (2017).
- timm — [pytorch-image-models](https://github.com/huggingface/pytorch-image-models).
