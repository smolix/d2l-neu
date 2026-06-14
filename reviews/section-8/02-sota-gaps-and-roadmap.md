# §8 "Modern Convolutional Neural Networks" — SOTA Gap Analysis & Improvement Roadmap

**Reviewer role:** Section-level outside view (forward-looking gap analysis + improvement roadmap).
**Date:** 2026-06-13. **Repo:** `d2l-neu`. **Primary lens:** PyTorch 2.x (4-fw book).
**Companion review:** a per-page teaching-quality pass owns line edits to the existing 8 files; **this report owns the section-level calls** (what is obsolete, what is missing, which new notebooks to add, the reordered structure, and a dispatch manifest).

**Section today (`chapter_convolutional-modern/`, 8 files, `_quarto.yml` ll. 105-113):**
`index` · `alexnet` · `vgg` · `nin` · `googlenet` · `batch-norm` · `resnet` (incl. ResNeXt) · `densenet` · `cnn-design` (AnyNet/RegNet).

**Stated problem (from the author):** the technical cutoff is **~2020**. The section ends at RegNet design spaces and *name-drops* ConvNeXt, MobileNetV3, SE, and EfficientNet in prose without teaching any of them. The job is to bring it to **state of the art for pure convnets** while staying **pre-transformer-for-vision** (ViT/DeiT/Swin/attention-as-backbone live in the separate, not-yet-written §"Attention Mechanisms and Transformers" → `vision-transformer.qmd`, `sec_vision-transformer`). **ConvNeXt is in scope** as the capstone that imports transformer-era *design choices* into a pure convnet.

---

## 0. Executive read

The existing arc is **excellent and almost entirely worth keeping**. It tells one clean story: *AlexNet (scale + learned features) → VGG (repeated 3×3 blocks) → NiN (1×1 conv, global average pool) → GoogLeNet (multi-branch, stem/body/head) → BatchNorm (trainability) → ResNet/ResNeXt (depth via identity, grouped conv) → DenseNet (concatenation) → RegNet (design spaces).* The code is already modernized to current APIs (Lazy layers, `init_cnn`), and the slides are strong.

But the section has **three structural gaps**, all consequences of the 2020 cutoff:

1. **No "modern training recipe" content at all.** Every notebook trains for 10 epochs with plain SGD and basic augmentation. The single most important lesson of 2020-2022 convnet research is that **the training recipe moves accuracy as much as the architecture**: a vanilla ResNet-50 goes from ~76% to **80.4%** with no architecture change ("ResNet strikes back"), and **~3/4** of ResNet-RS's gain is training method, not architecture ("Revisiting ResNets"). This thread (Mixup/CutMix, RandAugment, label smoothing, stochastic depth, cosine schedule, AdamW, EMA) is **absent**, and it is a prerequisite for honestly teaching ConvNeXt (whose first +2.7% is *just the recipe*).

2. **No efficiency / mobile-convnet thread.** Depthwise-separable convolution, the inverted residual (MobileNetV2), and SE attention are **foundational primitives** that the section's *own capstone candidates* (EfficientNet's MBConv, ConvNeXt's inverted bottleneck) are built from — yet they appear nowhere. ResNeXt's grouped convolution is taught; its natural limit (depthwise) is not. SE is name-dropped in `cnn-design` as "a precursor to Transformers" but never defined.

3. **The capstone is missing.** The section stops at RegNet (2020) and *mentions* ConvNeXt in two sentences. ConvNeXt (2022) is the natural ending: it is a **pure convnet** that systematically imports ViT/Swin design choices and matches them at equal FLOPs. It is the cleanest possible "what convnets learned from the transformer era" story and the **right handoff point** to the future transformer-vision section.

**The fix is additive, not surgical.** Keep all 8 sections (with two trims and one merge-candidate). Add **three new notebooks** — *efficient convnets* (depthwise-separable + inverted residual + SE), *modern training recipes*, and *ConvNeXt* — plus a short *normalization-beyond-BatchNorm* expansion folded into `batch-norm`. The result is a section that ends at the genuine 2026 frontier for pure convnets and hands off cleanly to vision transformers.

**Single highest-value addition:** the **ConvNeXt capstone notebook** (new `convnext.md`), because it (a) closes the 2020→2022 gap, (b) retroactively justifies the entire chronological arc by ablating each historical idea's modern descendant, (c) is the designed handoff to the transformer section, and (d) trains at small scale. It is closely followed by the **modern-training-recipe notebook**, which is the prerequisite that makes ConvNeXt's story honest.

---

## 1. OBSOLETE / didn't stand the test of time (section-level calls)

Legend: **CUT** (remove) · **TRIM** (shrink/demote) · **REFRAME** (keep, change framing) · **KEEP** (defensible as-is, listed to preempt over-cutting).

### 1.1 NiN and GoogLeNet are over-weighted vs 2026 relevance — **TRIM + REFRAME, do not cut**

NiN and GoogLeNet are each a full section (~15 KB and ~29 KB). In 2026 their *living* contributions are narrow but real:
- **NiN's living ideas** — the **1×1 convolution** as a per-pixel channel-mixing MLP, and **global average pooling** as the classifier head — are foundational and used in *every* subsequent architecture (they are literally the two pointwise convs in every depthwise-separable / inverted-residual / ConvNeXt block, and the GAP head is universal). **Keep NiN**, but reframe its summary to point forward explicitly: "the 1×1 conv you meet here is the channel-mixing half of every modern block (MobileNet, EfficientNet, ConvNeXt)." NiN earns its place precisely as the origin of the 1×1 conv.
- **GoogLeNet/Inception's living idea** — the **stem/body/head decomposition** and **multi-branch** design — the former persists everywhere (it is the organizing principle of `cnn-design`'s AnyNet); the latter (hand-tuned multi-branch with a "smorgasbord" of kernel sizes) is **largely dead** as a design strategy (replaced by uniform blocks: ResNeXt's "same transform in all branches," then depthwise). The Inception *block* is a historical artifact; the stem/body/head *framing* is load-bearing.

**Verdict:** Both stay (the chronological narrative needs them, and the per-page reviewer should keep their teaching quality high). But at the section level, **resist any future pressure to expand them**, ensure each ends with a forward-pointer to its modern descendant, and consider the **NiN+GoogLeNet merge** option in §4 if the section gets too long after the three adds. Their relative weight should *shrink* against the new ResNet/efficiency/ConvNeXt material — they are history, not frontier.

### 1.2 DenseNet — **KEEP but REFRAME as a road-not-taken**

DenseNet (concatenation instead of addition) is elegant and the Taylor-expansion motivation is lovely teaching. But in 2026 DenseNet is **not** a mainstream backbone: its memory cost (concatenation blows up activation memory; this is exactly the "activations, not FLOPs, predict GPU runtime" lesson RegNet later formalizes) made it lose to ResNet variants in practice. **Keep it** (the concatenation idea recurs in U-Net, feature pyramids, and the "reuse all features" intuition), but reframe the discussion to be honest: DenseNet is a beautiful idea whose **memory profile** kept it from winning — a perfect setup for the efficiency thread (activations matter) and a teachable example that elegance ≠ deployment. Do not present it as a live default.

### 1.3 Dated training advice / framing — **REFRAME (cross-cutting)**

- **"10 epochs, plain SGD, resize Fashion-MNIST to 224" is treated as just-how-it's-done.** Across `alexnet`/`vgg`/`nin`/`googlenet`/`resnet`/`cnn-design`, training is a 10-epoch SGD smoke test with no augmentation, no schedule, no regularization beyond dropout. In 2026 this *understates* what these architectures achieve and, worse, leaves the student believing architecture is the only lever. **Reframe** (lightly, per-page; deeply in the new recipe notebook): each training cell should acknowledge it is a *smoke test* and forward-point to the modern-recipe notebook. The recipe notebook then makes the "training matters as much as architecture" point explicitly.
- **`cnn-design` Discussion is a transformer-triumphalist sign-off that has aged into a half-truth.** It currently says "*scalability trumps inductive biases*," "vision Transformers by now lead," and "recent hardware optimizations have only widened the gap in favor of Transformers." Post-ConvNeXt (2022), ConvNeXt V2 (2023), RepLKNet/SLaK/PeLK (2022-2024), and the MetaFormer framing (2022), this is **outdated and one-sided**. The honest 2026 statement: *at matched FLOPs/compute, well-engineered pure convnets (ConvNeXt, large-kernel nets, InternImage) remain competitive with ViTs; the field has largely converged on a shared macro-architecture (the "MetaFormer") where the token-mixer (attention vs conv vs pooling) matters less than the overall block template.* **REFRAME** this Discussion (the per-page reviewer should flag it; the ConvNeXt notebook supersedes it as the section's true conclusion).
- **The batch-norm "internal covariate shift" debunking is good and current — KEEP.** That discussion (Rahimi's "alchemy," Lipton & Steinhardt, Santurkar et al.) is exactly right and well done.

### 1.4 ShiftNet name-drop — **CUT from prose**

`resnet.md` and `index.md` cite ShiftNet (`wu2018shift`) as a "trick" for efficient networks ("mimics a 3×3 conv by shifting activations, no computational cost"). ShiftNet is a historical curiosity that did **not** stand the test of time — depthwise-separable convolution won that race decisively. Replace the ShiftNet mention with a forward-pointer to **depthwise-separable convolution** (the efficiency idea that *did* win), which the new efficient-convnets notebook teaches. **CUT** the ShiftNet sentences.

### 1.5 Things to explicitly KEEP (preempting over-correction)

- **The ResNet function-class / nested-classes motivation** — timeless, one of the best pieces of pedagogy in the book. KEEP verbatim.
- **BatchNorm from-scratch + concise** — KEEP; the train/eval-mode distinction is essential and recurs.
- **ResNeXt + grouped convolution** — KEEP; it is the direct on-ramp to depthwise (groups = channels) and to RegNet's group-width parameter.
- **AnyNet/RegNet design-space methodology** (CDF of error, multi-fidelity, "design distributions not networks") — KEEP; it is unique among textbooks and genuinely current as a *methodology*. (It can go *deeper*; see §2.)

---

## 2. MISSING concepts (2020-2026 convnets), ranked

Each: why it matters, where it slots, how to teach it at the d2l bar. Ranked by value-to-the-section.

| Rank | Concept | Why it matters | Where it slots | Teach-at-bar |
|---|---|---|---|---|
| **1** | **Modern training recipe** (Mixup, CutMix, RandAugment, label smoothing, **stochastic depth**, cosine LR, AdamW, EMA) | The headline lesson of the era: recipe ≈ architecture. Vanilla ResNet-50: 76%→**80.4%** with no arch change. Prerequisite for ConvNeXt (first +2.7% is recipe). | **New notebook** after `resnet` (so there's a real net to train). | Ablation table on a small ResNet/Fashion-MNIST: stack ingredients, show accuracy climb. All ingredients are small-scale demonstrable. |
| **2** | **Depthwise-separable convolution** | The atom under MobileNet, Xception, EfficientNet, ConvNeXt. Generalizes ResNeXt's grouped conv to its limit (groups=channels). FLOP ratio `1/N + 1/D_k²` ≈ 1/9 for 3×3. | **New "efficient convnets" notebook** after `resnet`/ResNeXt (grouped conv is the on-ramp). | Pure shape/FLOP walkthrough on tiny tensors — no training needed for the arithmetic; then train a small depthwise net. |
| **3** | **Inverted residual + linear bottleneck** (MobileNetV2) | The single most influential block of the era: became EfficientNet's MBConv and inspired ConvNeXt's inverted bottleneck. "Expand → depthwise → project," residual on the *thin* ends; linear (no ReLU) projection preserves info in low dim. | Same **efficient-convnets** notebook. | Shape walkthrough (t=6 expand → 3×3 DW → linear 1×1) + a 2-D toy showing ReLU collapsing a low-dim manifold. |
| **4** | **Squeeze-and-Excitation (channel attention)** | The simplest attention in vision; reused in MobileNetV3, EfficientNet, RegNetY. Already *named* in `cnn-design` but never defined. SE-ResNet-50: 75.20→**76.71%** top-1, ~free FLOPs. | Same **efficient-convnets** notebook (it pairs with the inverted residual, as in MBConv); referenced back from `cnn-design`'s RegNetY mention. | Bolt SE onto a small CNN; show tiny param delta + accuracy bump on Fashion-MNIST. 3-line formula. |
| **5** | **ConvNeXt** (the capstone) | The "convnet for the 2020s": a pure convnet modernized step-by-step toward Swin, matching it at equal FLOPs. Closes the 2020→2022 gap and is the designed handoff to vision transformers. | **New capstone notebook**, last architecture section (after `densenet`, before/merged-with `cnn-design`). | Build the ConvNeXt block; train a tiny ConvNeXt on Fashion-MNIST; present the modernization-roadmap accuracy table (cite paper numbers). |
| **6** | **Normalization beyond BatchNorm** (the BN/LN/IN/GN taxonomy; why ConvNeXt uses LayerNorm; GroupNorm for small batches) | BN has real downsides (batch dependence, train/test gap, small-batch collapse: BN 34.7% vs GN 24.1% at batch 2 on ResNet-50). ConvNeXt's BN→LN swap is part of the capstone story; the taxonomy unifies it. | **Expand `batch-norm`** (already mentions LayerNorm) with the 4-norm taxonomy + a GroupNorm subsection. | The 4-panel taxonomy is a perfect pre-generated house-style **figure** (GroupNorm Fig. 2). GN-vs-BN-at-small-batch is a small reproducible ablation. |
| **7** | **Compound scaling** (EfficientNet) | Principled way to build a model *family* from one base: scale depth/width/resolution together (`α·β²·γ²≈2`). Still the textbook reference for "balance all three axes." | A **subsection in the efficient-convnets notebook** (MBConv → EfficientNet) OR a strong exercise; the *block* (MBConv) is the priority, scaling is the named idea on top. | The rule + the single-axis-vs-compound ablation (≈79% vs 81% at fixed FLOPs) is demonstrable on tiny nets / as a discussion. |
| **8** | **Structural reparameterization** (RepVGG) | Beautiful, *highly teachable* idea: train a multi-branch block, **fold it algebraically into a single 3×3 conv** for inference. First plain model >80% ImageNet. Reconnects to VGG ("VGG great again"). | **Optional** subsection/notebook, or a strong exercise on `vgg` or in efficient-convnets. | Excellent small notebook: build 3-branch block, fold it, **numerically verify** identical outputs. No GPU. |
| **9** | **Large-kernel convnets** (RepLKNet 31×31, SLaK 51×51, PeLK 101×101) | The ViT-inspired pure-conv line: a few very large depthwise kernels give ViT-like effective receptive fields. Part of why ConvNeXt's 7×7 helps; the 2022-2024 frontier. | **Discussion/forward-pointer** in the ConvNeXt notebook (ConvNeXt uses 7×7; these push further). | Narrative + figure (effective receptive field). Not a from-scratch notebook (large-scale). |
| **10** | **Normalizer-free nets / NFNets** (Scaled Weight Standardization, AGC) | Proof that BN is *removable* at SOTA (NFNet-F5 86.5% no extra data); AGC + weight standardization are reusable. Reinforces the "BN has downsides" thread. | **Forward-pointer** in the expanded `batch-norm` Discussion. | Concept-level (variance-propagation argument, AGC in a few lines). Not a reproduction. |
| **11** | **Self-supervised convnet pretraining** (SimCLR/MoCo contrastive on ResNet; FCMAE masked on ConvNeXt V2) | The backbones in this chapter can be pretrained *without labels*; ConvNeXt V2 brought MAE-style masking to convnets. | **One-line forward-pointer only** (in ConvNeXt notebook + section index), home is a representation-learning chapter. | Do not teach here. Flag and defer. |
| **12** | **MetaFormer framing** ("the macro-architecture matters more than the token-mixer") | The cleanest way to defuse the conv-vs-attention dichotomy; PoolFormer (attention→avg-pool) still hits 82.1%. Reframes the whole conv-vs-ViT debate. | **Discussion** in the ConvNeXt notebook and the reframed `cnn-design` conclusion. | Narrative. A unifying idea, not a notebook. |

**Net priority for new *content*:** (1) recipe, (2-4) depthwise-separable + inverted residual + SE bundled into one efficiency notebook, (5) ConvNeXt capstone, (6) norm taxonomy folded into `batch-norm`. Items 7-12 are subsections, forward-pointers, exercises, and figures within those.

---

## 3. NEW NOTEBOOKS to add (concrete proposals)

All proposals obey the d2l compute model: **small models, small datasets (Fashion-MNIST / CIFAR-scale), minutes on one RTX-4090.** Real-paper numbers are cited, never trained. Code **computes/demonstrates**, never draws (schematic figures are pre-generated SVGs via the `mdl-figure` skill). PyTorch is authored first and must be portable to jax/tensorflow/mxnet; per-framework feasibility noted.

---

### NOTEBOOK A — `efficient.md`: "Efficient ConvNets: Depthwise-Separable Convolutions, Inverted Residuals, and Squeeze-and-Excitation"

- **Slot/order:** after `resnet.md` (ResNeXt's grouped convolution is the on-ramp: depthwise is the limit groups=channels), before the recipe notebook. New `sec_efficient-cnn`.
- **Learning objective:** understand how convnets are made *cheap enough to deploy*, via the three primitives that every modern efficient/mobile net and even ConvNeXt are built from: depthwise-separable convolution, the inverted residual with linear bottleneck, and SE channel attention.
- **Architecture/recipe covered:** depthwise-separable conv (MobileNetV1); inverted residual + linear bottleneck (MobileNetV2, `t=6`); SE block (`r=16`); the MBConv block (inverted residual + SE, the EfficientNet atom); compound scaling as a closing subsection/discussion (EfficientNet `α·β²·γ²≈2`). Mention MobileNetV3's h-swish and NAS as forward-pointers; ShuffleNet's channel-shuffle and the "FLOPs ≠ latency" lesson as a discussion/exercise.
- **What the code computes/demonstrates:**
  1. **The FLOP/parameter arithmetic, computed and verified.** A cell that takes a standard `Conv2d(M→N, 3×3)` and a depthwise-separable equivalent, prints both parameter counts (`sum(p.numel())`) and verifies the ratio `1/N + 1/D_k² ≈ 1/9`. This is the load-bearing teaching moment and needs no training.
  2. **A depthwise-separable block** (`nn.Conv2d(..., groups=in_channels)` then `1×1`) — a shape walkthrough showing it produces the same output shape as a standard conv at a fraction of the cost.
  3. **The inverted-residual block** (`InvertedResidual`): expand 1×1 (×t) → BN/ReLU6 → depthwise 3×3 → BN/ReLU6 → project 1×1 → BN (**linear**, no activation) → residual on the thin ends. A `layer_summary` walkthrough; a short demonstration that the residual connects the narrow tensors.
  4. **An `SEBlock`** (GAP → FC `C→C/r` → ReLU → FC `C/r→C` → sigmoid → channel-wise scale): print its parameter cost relative to the block it augments (show it is ~free), then train a small CNN with and without SE on Fashion-MNIST and report the accuracy delta (the d2l "verify it" move).
  5. **A tiny MobileNetV2-style net** (a handful of inverted-residual blocks) trained on Fashion-MNIST as a smoke test, plus a `d2l.plot` loss curve. Cite the real MobileNetV2 (72.0% / 3.4M params) and EfficientNet-B0 (77.1% / 5.3M / 0.39 GFLOPs) numbers.
  6. **Optional compute-savings demonstration:** a small table comparing parameter counts of a standard-conv net vs the depthwise-separable version at matched architecture.
- **Dataset:** Fashion-MNIST (resize 96×96, as the existing CNN notebooks do) for the trained pieces; the arithmetic/shape cells use random tensors.
- **Framework coverage:** **pytorch** (`groups=`, full), **jax/flax** (`feature_group_count=` for depthwise — already used in the ResNeXt code, so portable), **tensorflow** (`Conv2D(groups=)` / `DepthwiseConv2D`, `tf.keras.layers` SE is trivial), **mxnet** (`nn.Conv2D(groups=)`; SE via gluon). All four feasible; SE and inverted-residual are framework-agnostic. **Author pytorch + jax first; tensorflow/mxnet portable.**
- **GPU cost:** ~2-4 small training runs at Fashion-MNIST/96×96 scale, ~1-3 min each on one RTX-4090. Cheap. The arithmetic cells are instant.
- **Schematic figures (mdl-figure):** `mdl-cnn-depthwise-separable.svg` (standard conv vs depthwise+pointwise factorization), `mdl-cnn-inverted-residual.svg` (ResNet bottleneck wide-ends vs inverted-residual thin-ends, side by side), `mdl-cnn-se-block.svg` (squeeze→excite→scale data flow).
- **External refs:** MobileNetV1 (arXiv:1704.04861), MobileNetV2 (arXiv:1801.04381), MobileNetV3 (arXiv:1905.02244), Xception (arXiv:1610.02357), SENet (arXiv:1709.01507), EfficientNet (arXiv:1905.11946), ShuffleNetV2 (arXiv:1807.11164).

---

### NOTEBOOK B — `training-recipe.md`: "How You Train Matters: Modern Recipes for ConvNets"

- **Slot/order:** after `efficient.md` (needs a real net to train; ResNet/efficient blocks already in hand), before ConvNeXt (which *depends* on this — its first +2.7% is the recipe). New `sec_training-recipe`. **This is a slightly unusual section for the book** (it is method, not architecture) — frame it as the bridge from "designing networks" to "training them well," and as the prerequisite for the capstone.
- **Learning objective:** internalize that the training procedure (augmentation, regularization, schedule, optimizer, averaging) often moves accuracy as much as the architecture, and learn the standard 2020s ingredients. Drive home the benchmarking lesson: *architecture comparisons are confounded by recipe; fix the recipe to compare fairly.*
- **Architecture/recipe covered:** a fixed small ResNet (reuse `d2l.ResNet18`) held constant while the recipe is varied. Ingredients, each demonstrated: **label smoothing** (`ε=0.1`), **Mixup** (`α=0.2`), **CutMix** (`α=1.0`), **Cutout/Random Erasing**, **stochastic depth** (survival `p_ℓ = 1 − (ℓ/L)(1−p_L)`, `p_L=0.5`), **cosine LR schedule**, **AdamW** (decoupled weight decay) vs Adam, **EMA of weights**. RandAugment as a "what magnitude M does" demo (its gain is muted at small scale — be honest).
- **What the code computes/demonstrates:**
  1. **Mixup/CutMix implemented from scratch** (a few lines: sample `λ~Beta(α,α)`, blend a batch, blend labels) with a **visualization cell** that shows blended Fashion-MNIST images (this is a *data* demonstration of the transform, allowed — it computes the actual augmented batch the model sees).
  2. **Label smoothing** as a one-line loss modification; show the soft-target vector.
  3. **Stochastic depth** as a small `nn.Module` wrapper (DropPath) around a residual block; print the per-layer survival probabilities; show it reduces to identity at eval.
  4. **A cosine LR schedule** plotted (`d2l.plot` of LR vs step — a computed curve, allowed) and wired into the Trainer.
  5. **The headline ablation table:** train the *same* `ResNet18` on Fashion-MNIST (or CIFAR-10) under (a) baseline SGD, (b) + cosine + AdamW, (c) + label smoothing, (d) + Mixup/CutMix, (e) + stochastic depth, (f) + EMA — and report the validation-accuracy climb in a table. This is the entire point of the notebook made concrete and reproducible.
  6. **An EMA wrapper** (shadow weights, `decay=0.999`); compare raw-vs-EMA validation accuracy per epoch.
  - Cite the anchor results: "ResNet strikes back" A1 = **80.4%** (vanilla ResNet-50, no arch change; A2 79.8% / A3 78.1%); ResNet-RS additive decomposition (baseline ResNet-200 79.0%; training-only +3.2 → 82.2%, arch +1.2 → 83.4% — **~3/4 from training**). Quote the regularization rule: *train longer → more regularization; more data → less*.
- **Dataset:** Fashion-MNIST and/or CIFAR-10 (CIFAR-10 makes the augmentation gains more visible — Cutout/Random-Erasing's headline results are CIFAR; consider CIFAR-10 here even though the rest of the section uses Fashion-MNIST, with a one-line justification, OR keep Fashion-MNIST for consistency and accept smaller deltas). Recommend **CIFAR-10** for this notebook specifically (32×32, still tiny) because augmentation effects are larger and more pedagogically legible; note the dataset switch explicitly.
- **Framework coverage:** **pytorch** first (all ingredients native: `CrossEntropyLoss(label_smoothing=)`, `CosineAnnealingLR`, `AdamW`, easy Mixup/EMA). **jax/optax** (optax has `cosine_decay_schedule`, `adamw`; Mixup/label-smoothing are array ops; EMA via `optax.ema`) — feasible. **tensorflow** (`tf.keras` has label smoothing, cosine decay, AdamW; Mixup manual) — feasible. **mxnet** — feasible but more manual; this notebook is the **most pytorch/jax-leaning**; author pytorch fully, jax second, mark tf/mxnet as "core ingredients portable, some helpers pytorch-only." Per CLAUDE.md, pytorch + portability is the floor; note feasibility honestly.
- **GPU cost:** the ablation is ~6 short training runs of a small ResNet on CIFAR-10/Fashion-MNIST, ~2-5 min each on one RTX-4090 → ~15-30 min total per framework. The dominant cost in the section but still well within the small-compute model. Keep epochs modest (10-20) and the net small.
- **Schematic figures (mdl-figure):** `mdl-cnn-mixup.svg` (two images + blended result, schematic), `mdl-cnn-stochastic-depth.svg` (residual stack with some blocks dropped → identity, survival-prob gradient by depth), `mdl-cnn-cosine-schedule.svg` (optional — the LR curve is better as a computed `d2l.plot` cell). Prefer computing the LR curve in code; reserve mdl-figures for the schematic transforms.
- **External refs:** ResNet strikes back (arXiv:2110.00476), Revisiting ResNets/ResNet-RS (arXiv:2103.07579), label smoothing (arXiv:1906.02629), Mixup (arXiv:1710.09412), CutMix (arXiv:1905.04899), RandAugment (arXiv:1909.13719), Random Erasing (arXiv:1708.04896), Cutout (arXiv:1708.04552), stochastic depth (arXiv:1603.09382), SWA (arXiv:1803.05407), SGDR/cosine (arXiv:1608.03983), AdamW (arXiv:1711.05101).

---

### NOTEBOOK C — `convnext.md`: "ConvNeXt: A ConvNet for the 2020s" (THE CAPSTONE — highest value)

- **Slot/order:** the **last architecture section**. Place after `densenet.md` and **before `cnn-design.md`**, OR after `cnn-design` as the literal finale. Recommended: **`densenet` → `convnext` → `cnn-design`** so the design-space methodology (a *meta* topic) follows the architectural climax — but a strong alternative is to make ConvNeXt the very last section since it is the genuine endpoint and the handoff. (See §4 for the recommended ordering and rationale.) New `sec_convnext`.
- **Learning objective:** see how a **pure convnet**, modernized step-by-step with transformer-era design choices, matches Vision Transformers at equal compute — and understand *which* choices mattered. This is the synthesis: every earlier idea (VGG blocks, NiN's 1×1, ResNet's residual, ResNeXt→depthwise, the recipe, the inverted residual, LayerNorm) reappears as one ingredient in the ConvNeXt block.
- **Architecture/recipe covered:** the ConvNeXt block (**depthwise 7×7 → LayerNorm → 1×1 expand ×4 → GELU → 1×1 project → LayerScale → DropPath residual**); the "patchify" stem (4×4 stride-4 conv); stage compute ratio (3,3,9,3); separate downsampling layers (2×2 stride-2 conv + LN); the **modernization roadmap** (the famous 78.8→82.0 bar chart). Forward-point to ConvNeXt V2 (GRN, FCMAE masked pretraining) and large-kernel nets (RepLKNet/SLaK/PeLK).
- **What the code computes/demonstrates:**
  1. **The ConvNeXt block, built and shape-walked** (`ConvNeXtBlock`): depthwise 7×7 (`groups=dim`, `padding=3`) → LayerNorm (channels-last) → Linear `dim→4·dim` → GELU → Linear `4·dim→dim` → LayerScale (learnable per-channel γ, init 1e-6) → residual with DropPath. Print the block's output shape; contrast its **single norm, single activation, inverted bottleneck, depthwise-first** structure against the ResNet bottleneck (which the student already knows) — ideally a side-by-side parameter/shape comparison cell.
  2. **The patchify stem** as a single `Conv2d(3, C, kernel_size=4, stride=4)`; show it 4×-downsamples like a ViT patch embed (connect to NiN's 1×1 and VGG's stem).
  3. **A tiny ConvNeXt** (`ConvNeXt` with depths e.g. (1,1,3,1) or (2,2,2,2) and small widths, far smaller than ConvNeXt-T's (3,3,9,3)/(96,192,384,768)) assembled stem→4 stages→GAP+LN+linear head; a `layer_summary` walkthrough.
  4. **Train the tiny ConvNeXt on Fashion-MNIST** (reuse the modern recipe from notebook B — this is the payoff of ordering B before C) and plot the loss curve.
  5. **The modernization-roadmap table** presented as data (a markdown table or a `d2l.plot` bar chart of the published per-step accuracies — citing the paper, NOT retrained): ResNet-50 76.1 → recipe 78.8 → stage-ratio 79.4 → patchify 79.5 → depthwise+width 80.5 → inverted-bottleneck 80.6 → 7×7 kernel 80.6 → fewer-activations 81.3 → fewer-norms 81.4 → BN→LN 81.5 → separate-downsampling **82.0** (= ConvNeXt-T; Swin-T is 81.3 at equal FLOPs). This table IS the chapter's thesis.
  6. **Final config table** (cite): ConvNeXt-T/S/B/L params/FLOPs/top-1; ConvNeXt-XL 87.8% (IN-22K); ConvNeXt V2 Atto 3.7M/76.7% → Huge 650M/88.9%.
- **Dataset:** Fashion-MNIST (resize 96×96 or keep 224 for one shape demo) for the trained tiny model; cite real ImageNet numbers for everything else.
- **Framework coverage:** **pytorch** first (the block is straightforward; `nn.GELU`, `nn.LayerNorm`, depthwise via `groups=`, a small `DropPath` module, LayerScale as an `nn.Parameter`). **jax/flax** (LayerNorm, GELU, depthwise via `feature_group_count`, channels-last is natural in flax) — feasible. **tensorflow** (`tf.keras.layers.LayerNormalization`, `gelu`, `DepthwiseConv2D`) — feasible. **mxnet** (gluon has LayerNorm, GELU; depthwise via groups) — feasible but lowest priority (MXNet is archived; per the project memory it is KEPT as a co-equal tab, so port it, but author pytorch+jax first). All four feasible.
- **GPU cost:** one tiny-ConvNeXt training run on Fashion-MNIST/96×96, ~2-4 min on one RTX-4090, ×4 frameworks. The roadmap/config tables cost nothing (cited data). Cheap.
- **Schematic figures (mdl-figure):** `mdl-cnn-convnext-block.svg` (ResNet bottleneck vs ConvNeXt block, side by side — the single most important figure in the new material), `mdl-cnn-patchify-stem.svg` (ResNet 7×7+maxpool stem vs 4×4-stride-4 patchify), `mdl-cnn-modernization-roadmap.svg` (optional schematic of the step ladder; the accuracy bar chart is better as a computed `d2l.plot` cell from cited numbers). The roadmap accuracy chart should be a **code cell plotting hardcoded published values** (a computed plot of real data, in the spirit of "data plots that teach a computed result"), clearly labeled as paper numbers.
- **External refs:** ConvNeXt (arXiv:2201.03545), ConvNeXt V2 (arXiv:2301.00808), LayerScale/CaiT (arXiv:2103.17239), Swin (arXiv:2103.14030, as the comparison point — *not* taught here), RepLKNet (arXiv:2203.06717), SLaK (arXiv:2207.03620), PeLK (arXiv:2403.07589), MetaFormer/PoolFormer (arXiv:2210.13452), InternImage (arXiv:2211.05778).

---

### NOTEBOOK D (OPTIONAL / stretch) — `repvgg.md` or a strong exercise: "Structural Reparameterization (RepVGG)"

- **Slot/order:** optional; either a short standalone section after `vgg` (thematic callback: "VGG great again") or, more economically, a **strong exercise** appended to `vgg.md` or `efficient.md`. New `sec_repvgg` if standalone.
- **Learning objective:** a training-time multi-branch block and an inference-time plain block can be **exactly equivalent**; you can train with rich gradient flow and deploy a fast, plain 3×3 stack.
- **What the code computes/demonstrates:** build the RepVGG train-time block (parallel 3×3 conv + 1×1 conv + identity, each with BN); implement the **fold** (fuse conv+BN → conv-with-bias; pad 1×1 to 3×3; identity → one-hot 3×3 kernel; sum the three kernels and biases); then **numerically verify** the folded single 3×3 conv produces outputs equal (to float tolerance) to the multi-branch block on a random input. This is a near-perfect d2l demonstration: a surprising claim, verified in code, no GPU, no training.
- **Dataset:** none needed for the core demo (random tensors); optional tiny train on Fashion-MNIST.
- **Framework coverage:** pytorch (clean); the algebra ports to all frameworks but the demo is most elegant in pytorch. Author pytorch; mark others optional.
- **GPU cost:** ~zero (the demo is a forward-pass equivalence check on CPU-scale tensors).
- **Schematic figure (mdl-figure):** `mdl-cnn-repvgg-fold.svg` (3-branch train block → folded single 3×3 conv).
- **External refs:** RepVGG (arXiv:2101.03697).
- **Recommendation:** include as an **exercise** in v1 (keeps the section length in check) and promote to a standalone notebook later if desired. It is high-delight, low-cost, but not essential to the SOTA arc.

---

## 4. FULL IMPROVEMENT PLAN / ROADMAP

### 4.1 The explicit through-line (the story the reordered section tells)

> **AlexNet** (depth + scale + learned features beat hand-crafted) → **VGG** (stack identical 3×3 blocks) → **NiN** (1×1 conv as channel-mixing MLP; global average pooling head) → **GoogLeNet** (multi-branch; the stem/body/head decomposition) → **BatchNorm** (+ the normalization family: BN/LN/IN/GN — trainability and its modern alternatives) → **ResNet / ResNeXt** (depth via identity; grouped convolution) → **Efficient convnets** (grouped → depthwise-separable; inverted residual + linear bottleneck; SE channel attention; compound scaling — the deployable-net primitives) → **Modern training recipe** (the recipe moves accuracy as much as the architecture) → **ConvNeXt** (a pure convnet that imports transformer-era design choices and matches ViTs at equal compute) → **RegNet / design spaces** (designing *families* of networks, not single networks) → **handoff to Vision Transformers**.

Two through-line threads are woven in rather than siloed:
- **Efficiency thread:** channels are expensive (ResNeXt) → groups → depthwise (efficient) → activations-not-FLOPs predict runtime (DenseNet's memory cost; RegNet's finding) → compute-aware design (RegNet). 
- **Training-recipe thread:** every architecture's smoke-test training forward-points to the recipe notebook, which then shows the recipe is a first-class lever and is the prerequisite for ConvNeXt.

### 4.2 Proposed new §8 structure (reordered file list with adds/cuts/merges)

**Bold = new. *Italic* = changed framing/scope.** `(keep)` = unchanged content, per-page reviewer owns quality.

| # | File | Status | One-line charter |
|---|---|---|---|
| 0 | `index.md` | *REFRAME* | Cover. Update the closing paragraph: the tour now **ends at ConvNeXt / the 2020s frontier** and hands off to vision transformers; cut the ShiftNet/over-eager-Transformer framing; add the efficiency + training-recipe threads to the ToC narrative. |
| 1 | `alexnet.md` | (keep) | Depth + scale + representation learning beat hand-crafted features. |
| 2 | `vgg.md` | (keep) | Repeated identical 3×3 blocks; architecture-as-a-tuple. (Optional RepVGG exercise.) |
| 3 | `nin.md` | *REFRAME* | 1×1 conv (channel-mixing MLP) + global average pooling. End with a forward-pointer: these two ideas are in every modern block. |
| 4 | `googlenet.md` | *REFRAME* | Multi-branch Inception; the durable stem/body/head decomposition. Frame multi-branch as historical, stem/body/head as load-bearing. |
| 5 | `batch-norm.md` | *EXPAND* | BatchNorm (from-scratch + concise; keep). **Add: the BN/LN/IN/GN normalization taxonomy (figure) + a GroupNorm subsection (small-batch); forward-point to normalizer-free nets.** |
| 6 | `resnet.md` | (keep) | Residual/nested-classes; ResNet; ResNeXt + grouped convolution. (Cut the ShiftNet sentence; add a forward-pointer to depthwise.) |
| 7 | `densenet.md` | *REFRAME* | Concatenation / feature reuse. Honest framing: a beautiful idea whose **memory cost** kept it from winning (sets up the efficiency thread). |
| 8 | **`efficient.md`** | **NEW (A)** | **Depthwise-separable conv + inverted residual + linear bottleneck + SE + (compound scaling). The deployable-convnet primitives.** |
| 9 | **`training-recipe.md`** | **NEW (B)** | **The training procedure moves accuracy as much as the architecture: Mixup/CutMix/RandAugment, label smoothing, stochastic depth, cosine, AdamW, EMA. The benchmarking lesson.** |
| 10 | **`convnext.md`** | **NEW (C)** | **The capstone: a pure convnet modernized toward Swin, matching ViTs at equal compute. The modernization roadmap. Handoff to vision transformers.** |
| 11 | `cnn-design.md` | *REFRAME* | AnyNet/RegNet design spaces (keep the methodology; can go deeper — quantized-width rule, depth≈20, activations-not-FLOPs). **Rewrite the Discussion** away from transformer-triumphalism toward the 2026 "MetaFormer / convnets remain competitive" view. |
| (opt) | **`repvgg.md`** | **NEW (D), optional** | **Structural reparameterization: train multi-branch, fold to a plain 3×3 stack (numerically verified).** Ship as an exercise first. |

**Ordering rationale for the three adds (8-9-10):** efficiency primitives (8) come right after ResNeXt's grouped conv (its natural continuation); the recipe (9) needs a real net to train and is the prerequisite for the capstone; ConvNeXt (10) is the synthesis that *uses* the inverted bottleneck from (8), the recipe from (9), and the LayerNorm from (5). Placing `cnn-design`/RegNet (11) *after* ConvNeXt is deliberate: design-space methodology is a *meta* lesson that reads well as the reflective coda after the architectural climax, and RegNet's "activations not FLOPs" finding is reinforced by everything in 8-10. (If the author prefers ConvNeXt as the literal last word, swap 10↔11; both are defensible. The recommended order keeps the chronological spine — RegNet 2020 is *earlier* than ConvNeXt 2022, so a purely chronological reader might want 11 before 10, but the *pedagogical* climax argues for ConvNeXt-then-design-spaces. Flag this as an author decision; default to the table above.)

**Merge option (only if length becomes a problem):** `nin.md` + `googlenet.md` → one "Network-in-Network and Multi-Branch Designs" section. Both are historical, share the 1×1-conv and stem/body/head threads, and trimming them buys room for the three adds without losing the narrative. **Do not merge in v1** unless the section feels bloated after the adds; it is a release valve, not a goal.

**Section length check:** 8 → 11 (or 12 with RepVGG) sections. That is comparable to the §6 "Builders' Guide" (7) and the CNN-basics chapter (7). Acceptable; the merge option exists if needed.

### 4.3 The handoff point to vision transformers (explicit)

The clean handoff is **at the end of `convnext.md`**, into `chapter_attention-mechanisms-and-transformers/vision-transformer.qmd` (`sec_vision-transformer`). ConvNeXt is the designed bridge: it adopts the ViT/Swin *recipe and macro-design* (patchify stem, inverted bottleneck, LayerNorm, GELU, stage ratios) while keeping convolution as the spatial mixer. The handoff paragraph should say, in effect:

> *ConvNeXt shows that most of the ViT/Swin accuracy advantage was design choices and training recipe, not self-attention per se — a pure convnet matches them at equal compute. The remaining question is what self-attention as the spatial mixer buys you (global receptive field from layer one, input-dependent mixing, a token interface that unifies vision and language). We take up Vision Transformers in :numref:`sec_vision-transformer`.*

Everything attention-as-backbone (ViT, DeiT, Swin, attention mechanics) stays out of §8. The MetaFormer framing (block template matters more than the mixer) is the perfect *conceptual* hinge and belongs in both ConvNeXt's Discussion and the reframed `cnn-design` conclusion — it sets up the transformer section without teaching attention.

### 4.4 Dependencies and sequencing for authoring

- **B (recipe) before C (ConvNeXt):** ConvNeXt's training reuses the recipe; its thesis depends on the recipe lesson.
- **A (efficient) before C:** ConvNeXt's inverted bottleneck is introduced in A.
- **`batch-norm` expansion (5) before C:** the LN/GN taxonomy is referenced by ConvNeXt's BN→LN swap.
- **Figures before notebooks:** each notebook's `mdl-cnn-*` SVGs should be generated (new `tools/gen_mdl_cnn_figures.py` importing the shared style in `tools/gen_mdl_figures.py`) before or alongside the prose, per the `mdl-figure` workflow; run `figure-style-audit` after.
- **Reframes (0, 3, 4, 7, 11) are independent** and can proceed in parallel with the new-notebook authoring; they are light prose edits.
- **`d2l` library:** the inverted-residual, SE, DropPath, and ConvNeXt blocks likely warrant `#@save` so later notebooks (and the recipe ablation) can import them — coordinate with `make lib` (a library rebuild affects all frameworks' notebooks; rerun affected ones).

---

## 5. Dispatch manifest (ordered work items for authoring agents)

Each row is parcelable to one agent. **FW** = framework coverage to author (pytorch always first; portability per CLAUDE.md). **GPU** = does the work item require a GPU box, or run on CPU/Apple-Silicon. Deps reference other rows.

| # | Work item | New/edited file | Figures (mdl-figure → `img/…svg`) | Notebook? + FW | GPU need | External refs (arXiv) | Deps |
|---|---|---|---|---|---|---|---|
| **W1** | **`batch-norm` expansion**: add BN/LN/IN/GN taxonomy subsection + GroupNorm small-batch subsection (+ optional GN-vs-BN ablation) + normalizer-free forward-pointer; reframe to set up "BN alternatives." | edit `batch-norm.md` | `mdl-cnn-norm-taxonomy.svg` (4-panel N×C grid, shaded norm regions — from GroupNorm Fig. 2) | yes (add a small GN-vs-BN@batch-2 cell), pytorch + jax + tf + mxnet | CPU to render; **GPU** to (re)execute the small ablation | GroupNorm 1803.08494; NFNets 2102.06171; NF-ResNets 2101.08692 | — |
| **W2** | **`efficient.md` (Notebook A)**: depthwise-separable arithmetic + block; inverted residual + linear bottleneck; SE block + with/without ablation; tiny MobileNetV2; compound-scaling closing subsection. | **new** `efficient.md` | `mdl-cnn-depthwise-separable.svg`, `mdl-cnn-inverted-residual.svg`, `mdl-cnn-se-block.svg` | yes, pytorch + jax first; tf + mxnet portable | **GPU** for the ~3 small trainings; arithmetic cells CPU | MobileNetV1 1704.04861; V2 1801.04381; V3 1905.02244; Xception 1610.02357; SENet 1709.01507; EfficientNet 1905.11946; ShuffleNetV2 1807.11164 | (`#@save` SE/inverted-residual → used by W3/W4) |
| **W3** | **`training-recipe.md` (Notebook B)**: Mixup/CutMix (from scratch + viz), label smoothing, stochastic depth (DropPath), cosine schedule (plotted), AdamW-vs-Adam, EMA; the headline ablation table on a fixed small ResNet. | **new** `training-recipe.md` | `mdl-cnn-mixup.svg`, `mdl-cnn-stochastic-depth.svg` (cosine curve = computed `d2l.plot` cell) | yes, pytorch full + jax; tf/mxnet "core ingredients portable" | **GPU** — the ablation is ~6 short runs (dominant cost of the section; keep small/short) | ResNet-strikes-back 2110.00476; ResNet-RS 2103.07579; label-smooth 1906.02629; Mixup 1710.09412; CutMix 1905.04899; RandAugment 1909.13719; RandomErasing 1708.04896; Cutout 1708.04552; stochastic-depth 1603.09382; SWA 1803.05407; SGDR 1608.03983; AdamW 1711.05101 | W2 (reuses blocks; optionally `#@save` DropPath/EMA) |
| **W4** | **`convnext.md` (Notebook C) — CAPSTONE**: ConvNeXt block (shape walk + ResNet-bottleneck contrast); patchify stem; tiny ConvNeXt trained with W3's recipe; modernization-roadmap accuracy chart (cited data); config tables; ConvNeXt-V2 + large-kernel + MetaFormer forward-pointers; handoff to `sec_vision-transformer`. | **new** `convnext.md` | `mdl-cnn-convnext-block.svg` (vs ResNet bottleneck — top-priority figure), `mdl-cnn-patchify-stem.svg` (roadmap bar chart = computed `d2l.plot` of published numbers) | yes, pytorch + jax first; tf + mxnet portable | **GPU** for the one tiny-ConvNeXt training ×FW; tables/charts CPU | ConvNeXt 2201.03545; ConvNeXt-V2 2301.00808; LayerScale/CaiT 2103.17239; Swin 2103.14030 (comparison only); RepLKNet 2203.06717; SLaK 2207.03620; PeLK 2403.07589; MetaFormer 2210.13452; InternImage 2211.05778; SimCLR 2002.05709; MoCo 1911.05722 (SSL pointer) | W2, W3, W1 |
| **W5** | **`cnn-design` reframe + deepen**: rewrite the Discussion away from transformer-triumphalism to the 2026 MetaFormer/"convnets remain competitive" view; optionally deepen the RegNet methodology (quantized-width rule `w_j=w_0·w_m^{⌊s_j⌉}`, depth≈20, b=1, activations-not-FLOPs, RegNetY=RegNetX+SE, 5× GPU speedup vs EfficientNet-B5). | edit `cnn-design.md` | (none new; existing `regnet-*` figures suffice) | prose-only (no new code execution required) | CPU only | RegNet 2003.13678; Fast-and-Accurate-Scaling 2103.06877; MetaFormer 2210.13452; ConvNeXt 2201.03545 | W4 (so the conclusion can point at ConvNeXt) |
| **W6** | **Light reframes**: `index.md` closing paragraph (ends at ConvNeXt; drop ShiftNet/over-eager-Transformer framing; add efficiency+recipe threads to ToC narrative + add the 3 new files to the `toc` block); `nin.md` summary forward-pointer (1×1 conv → modern blocks); `googlenet.md` framing (multi-branch historical, stem/body/head durable); `densenet.md` memory-cost honest framing; `resnet.md` cut ShiftNet sentence + add depthwise forward-pointer. | edit `index.md`, `nin.md`, `googlenet.md`, `densenet.md`, `resnet.md` | (none) | prose-only | CPU only | (uses refs already cited in those files) | independent; do after W2/W4 exist so forward-pointers resolve |
| **W7** | **(Optional) `repvgg` exercise/notebook (D)**: train-time 3-branch block; fold to single 3×3 conv; numerically verify equivalence. Ship as an exercise on `vgg.md` or `efficient.md` first; promote to `repvgg.md` later. | edit `vgg.md` (exercise) or new `repvgg.md` | `mdl-cnn-repvgg-fold.svg` (if standalone) | yes (equivalence check), pytorch; others optional | CPU (forward-pass equivalence, no training) | RepVGG 2101.03697 | independent |
| **W8** | **Figure generator + audit**: create `tools/gen_mdl_cnn_figures.py` (imports shared style from `tools/gen_mdl_figures.py`) producing all `mdl-cnn-*.svg` above; run `figure-style-audit` on the section; ensure byte-idempotent SVGs committed. | new `tools/gen_mdl_cnn_figures.py`; commit `img/mdl-cnn-*.svg` | (produces all figures for W1-W4, W7) | n/a (matplotlib generator, not a notebook) | CPU only | — | precedes/parallels W1-W4 (figures wanted before prose) |

**Suggested execution order:** W8 (figures) ∥ W1 (norm expansion) → W2 (efficient) → W3 (recipe) → W4 (ConvNeXt capstone) → W5 (cnn-design reframe) → W6 (light reframes) ∥ W7 (optional RepVGG). W6's forward-pointers should land after W2/W4 exist so the `:numref:` targets resolve. After all notebooks: `make lib` (if blocks were `#@save`d), regenerate affected notebooks, execute on the GPU box, `make capture-outputs` (scope `--frameworks` if a single-fw re-run, per the project memory note about capture clobbering other frameworks), then `make html`.

**Verification gates per CLAUDE.md:** new `.md` only (never edit `.qmd`); add the 3 new files to `_quarto.yml` (under the CNN part, ll. 105-113) and to `CHAPTER_NUMBERING` in `tools/d2l_preprocess.py` (else they render unnumbered); run `tools/lint_source.py`; obey the PDF tripwires (no `]` in captions; `$` not followed by a digit; multi-line display equations with `$$` alone on their lines + `:eqlabel:` next line); no `---` em-dashes; no figure-drawing code in notebooks (all schematics via W8's generator).

---

## 6. Notes, caveats, and fidelity flags for authoring agents

- **SE-ResNet-50 numbers:** cite the **paper's** pair **75.20 → 76.71% top-1** (= 24.80 → 23.29% error). Do *not* cite "77.62%" to the SENet paper — that is torchvision's reference number, not the paper's.
- **NFNet SOTA:** the no-extra-data SOTA is **86.5%** (NFNet-F5), not 86.0%.
- **ResNet-RS decomposition baseline is ResNet-200 (79.0%)**, not ResNet-50 — quote it as such ("~3/4 of the 4.4-point gain is training method").
- **Mixup's standalone ImageNet gain is small** at 90 epochs (23.5→23.3%); its real value shows when stacked with CutMix/BCE/long schedules. Be honest in the ablation framing (the *stack* is the lesson, not any single ingredient).
- **RegNet specific quotes safe to use** (from the paper): "∼20 blocks (60 layers)," "w_m … ∼2.5," "bottleneck ratio b of 1.0," "RegNetX-8000 is 5× faster than EfficientNet-B5." The activation/FLOP correlation r-values (≈0.93/0.75/0.71) circulate in *secondary* summaries — cite as "widely reported" or present the paper's *graphical* claim, not verbatim coefficients.
- **ConvNeXt roadmap deltas:** present steps 4 ("depthwise conv," a temporary −1.2 dip) and 7 ("move up depthwise conv," a temporary −0.7 dip) honestly as intentional intermediate dips that *enable* the following gains (width↑, large-kernel). The headline arc is 78.8→82.0; the dips are a great teaching moment ("a change can hurt alone yet enable the next gain"). The kernel benefit **saturates at 7×7** (≥9 gives nothing) — that is *why* ConvNeXt picks 7×7, and the on-ramp to the large-kernel-nets forward-pointer (which needs reparameterization to go further).
- **Dataset for the recipe notebook:** recommend **CIFAR-10** specifically (augmentation deltas are larger and more legible than on Fashion-MNIST; Cutout/Random-Erasing's headline results are CIFAR), with a one-line note on the switch. The rest of the section stays on Fashion-MNIST.
- **Compute model discipline:** every trained model is *tiny* (depths/widths far below the paper configs). The recipe ablation (~6 short runs) is the section's compute peak — keep epochs ≤20 and the net small; it still fits minutes-per-run on one RTX-4090. All ImageNet-scale numbers are **cited, never trained.**
- **MXNet:** KEEP as a co-equal tab (project memory). Author pytorch + jax first for all three new notebooks; port tensorflow + mxnet. All proposed blocks (depthwise via groups, SE, inverted residual, LayerNorm/GELU, DropPath) are feasible in all four frameworks; the recipe notebook is the most pytorch/jax-leaning (mark tf/mxnet helpers honestly).

---

### Sources (primary, title — URL)

**Capstone / modernized convnets**
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt) — https://arxiv.org/abs/2201.03545
- Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders* — https://arxiv.org/abs/2301.00808
- Touvron et al., *Going deeper with Image Transformers* (CaiT / LayerScale) — https://arxiv.org/abs/2103.17239

**Design spaces / scaling**
- Radosavovic et al., *Designing Network Design Spaces* (RegNet) — https://arxiv.org/abs/2003.13678
- Dollár et al., *Fast and Accurate Model Scaling* — https://arxiv.org/abs/2103.06877
- Tan & Le, *EfficientNet* — https://arxiv.org/abs/1905.11946
- Tan & Le, *EfficientNetV2* — https://arxiv.org/abs/2104.00298

**Efficient / mobile convnets**
- Howard et al., *MobileNets* (V1) — https://arxiv.org/abs/1704.04861
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks* — https://arxiv.org/abs/1801.04381
- Howard et al., *Searching for MobileNetV3* — https://arxiv.org/abs/1905.02244
- Chollet, *Xception* — https://arxiv.org/abs/1610.02357
- Zhang et al., *ShuffleNet* — https://arxiv.org/abs/1707.01083
- Ma et al., *ShuffleNet V2* — https://arxiv.org/abs/1807.11164

**Attention in convnets**
- Hu et al., *Squeeze-and-Excitation Networks* — https://arxiv.org/abs/1709.01507
- Woo et al., *CBAM: Convolutional Block Attention Module* — https://arxiv.org/abs/1807.06521
- Park et al., *BAM: Bottleneck Attention Module* — https://arxiv.org/abs/1807.06514

**Normalization**
- Wu & He, *Group Normalization* — https://arxiv.org/abs/1803.08494
- Brock et al., *Characterizing signal propagation … unnormalized ResNets* (NF-ResNets) — https://arxiv.org/abs/2101.08692
- Brock et al., *High-Performance Large-Scale Image Recognition Without Normalization* (NFNets) — https://arxiv.org/abs/2102.06171

**Modern training recipe**
- Wightman et al., *ResNet strikes back* — https://arxiv.org/abs/2110.00476
- Bello et al., *Revisiting ResNets* (ResNet-RS) — https://arxiv.org/abs/2103.07579
- Müller et al., *When Does Label Smoothing Help?* — https://arxiv.org/abs/1906.02629
- Zhang et al., *mixup* — https://arxiv.org/abs/1710.09412
- Yun et al., *CutMix* — https://arxiv.org/abs/1905.04899
- Cubuk et al., *RandAugment* — https://arxiv.org/abs/1909.13719
- Zhong et al., *Random Erasing* — https://arxiv.org/abs/1708.04896
- DeVries & Taylor, *Cutout* — https://arxiv.org/abs/1708.04552
- Huang et al., *Deep Networks with Stochastic Depth* — https://arxiv.org/abs/1603.09382
- Izmailov et al., *SWA* — https://arxiv.org/abs/1803.05407
- Loshchilov & Hutter, *SGDR* (cosine) — https://arxiv.org/abs/1608.03983
- Loshchilov & Hutter, *Decoupled Weight Decay* (AdamW) — https://arxiv.org/abs/1711.05101

**Reparameterization / large kernels / framing / SSL (forward-pointers)**
- Ding et al., *RepVGG* — https://arxiv.org/abs/2101.03697
- Ding et al., *Scaling Up Your Kernels to 31×31* (RepLKNet) — https://arxiv.org/abs/2203.06717
- Liu et al., *More ConvNets in the 2020s* (SLaK) — https://arxiv.org/abs/2207.03620
- Chen et al., *PeLK* — https://arxiv.org/abs/2403.07589
- Ding et al., *UniRepLKNet* — https://arxiv.org/abs/2311.15599
- Wang et al., *InternImage* (DCNv3) — https://arxiv.org/abs/2211.05778
- Yu et al., *MetaFormer Is Actually What You Need for Vision* (PoolFormer) — https://arxiv.org/abs/2210.13452
- Chen et al., *SimCLR* — https://arxiv.org/abs/2002.05709
- He et al., *MoCo* — https://arxiv.org/abs/1911.05722
