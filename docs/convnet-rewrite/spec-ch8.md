# Chapter 8 (chapter_convolutional-modern) — restructure spec

Goal (Alex's words): chapters 7+8 together must be enough to teach the state
of the art in convolutional networks — the most important topics, **not
encyclopedic**. Transformers are acknowledged as superior for many tasks, but
this chapter is about convnets. The chapter currently stops in 2020; it even
cites ConvNeXt three times ("we will cover this in due course") without ever
covering it. The fix: compress the 2012–2015 history from four sections into
two, keep ResNet as the spine, and spend the reclaimed budget on the three
things that define convnets in 2026 — training recipes, ConvNeXt, and
efficient/edge convnets — plus an honest closing verdict.

## 1. New table of contents and file map

| # | File | Title (working) | Fate |
|---|------|-----------------|------|
| 8.0 | `index.md` | Modern Convolutional Neural Networks | rewrite |
| 8.1 | `alexnet.md` | The ImageNet Moment: AlexNet | compress ~40% |
| 8.2 | `blocks.md` **(new)** | Blocks, Bottlenecks, and Branches: VGG, NiN, GoogLeNet | merges `vgg.md` + `nin.md` + `googlenet.md` |
| 8.3 | `batch-norm.md` | Normalization Layers | keep + extend |
| 8.4 | `resnet.md` | Residual Networks: ResNet, ResNeXt — and DenseNet | keep + absorb `densenet.md` |
| 8.5 | `training-recipes.md` **(new)** | Training Recipes Matter | new |
| 8.6 | `convnext.md` **(new)** | ConvNeXt: A ConvNet for the 2020s | new |
| 8.7 | `efficient-convnets.md` **(new)** | Efficient ConvNets: Depthwise Separability, Mobile Architectures, Re-parameterization | new |
| 8.8 | `cnn-design.md` | Design Spaces and the Big Picture | keep AnyNet/RegNet, rewrite closing |

Retired files: `vgg.md`, `nin.md`, `googlenet.md`, `densenet.md` (contents
absorbed; delete the `.md`, their generated `.qmd`, and their
`outputs/*/chapter_convolutional-modern/{vgg,nin,googlenet,densenet}.json`
manifests in the same commit as the additions, so the tree never half-exists).

**Label preservation (verified crossref counts from outside the chapter):**
`sec_alexnet` (4 refs), `sec_batch_norm` (9), `sec_resnet` (8),
`sec_googlenet` (2), `sec_vgg` (1), `sec_nin` (1) must keep resolving.
`sec_vgg`, `sec_nin`, `sec_googlenet` become `##`-section labels inside
`blocks.md`; `sec_densenet` becomes a subsection label inside `resnet.md`.
Also keep `subsec_resnext`, `fig_resnext_block`, `sec_cnn-design` (referenced
inside ch7/8). Before finalizing, re-grep the whole book for every label
defined in the four retired files and make sure each survives somewhere.

Numbering-relative ordering rationale: batch-norm before resnet (dependency),
recipes after resnet (uses ResNet-18 as the testbed), convnext after recipes
(ConvNeXt = modern recipe + modernized architecture — teach the recipe first
so the architecture's gains aren't confounded).

## 2. Per-file specs

### 8.0 `index.md` — rewrite

Reframe as a two-era story: (i) the architecture race 2012–2015 (AlexNet →
VGG/NiN/GoogLeNet → BN → ResNet), (ii) the maturation 2016–2026 (design
spaces, recipes, ConvNeXt, efficiency), with the honest framing up front:
since ~2021 transformers lead large-scale image classification (pointer to
`chap_attention-and-transformers`); convnets remain the workhorse where
latency, small data, or dense prediction dominate — and this chapter brings
the reader to that 2026 state of the art. Fix the "It is only recently that
Transformers have begun to displace" phrasing. Drop the ShiftNet namedrop;
keep MobileNet/RegNet/ConvNeXt mentions, now with real sections to point to.

### 8.1 `alexnet.md` — compress

- Keep `sec_alexnet` and the overall shape (Representation Learning →
  AlexNet → Training → Discussion → Exercises).
- The 250-line preamble (lines ~10–256) compresses to ~120: keep the
  pre-2012 pipeline story (SIFT/kernel methods), the "data + compute +
  representation learning" thesis, and the learned-filters figure; cut
  redundant asides.
- **Purge datable hardware claims**: "NVIDIA's latest Ampere chips…6912 CUDA
  cores" (~line 225), "A100…Graviton 3…M1" (~229), "480 MFLOPS GeForce 256 vs
  today's 1000 TFLOPs" — replace with order-of-magnitude statements anchored
  to explicit years ("between 1999 and 2012, GPU throughput grew by roughly
  three orders of magnitude"), which cannot rot.
- Reword the "it was TensorFlow that changed the situation" claim (~line 491)
  to a neutral statement about autodiff frameworks (Theano → TF → PyTorch/JAX).
- New figure: `arch-alexnet.svg` (gallery style; LeNet vs. AlexNet side by
  side is allowed as the one comparison, or AlexNet alone with callouts for
  ReLU/dropout/its 11×11 stem — the writer decides after looking at both).
- Keep the training code and exercises (ex. 3 "why do engineers no longer
  benchmark on AlexNet" stays — it's good). New deck.

### 8.2 `blocks.md` — new file, merged history

One connected narrative: after AlexNet, progress came from *organizing*
convolutions — repetition (VGG), pointwise mixing + GAP (NiN), multi-branch
(GoogLeNet). Three `##` sections carrying the old labels:

1. **`sec_vgg`** — blocks as the unit of design. Keep the VGG-block +
   VGG-11 implementation and training essentially as today (this is
   load-bearing: every later `arch` tuple is this idea). Keep the stacked
   3×3 receptive-field argument, now referencing the ch7 formula. Cut the
   ParNet "exciting development" bet (aged badly). Fold the
   foundation-models VLSI analogy down to a sentence.
2. **`sec_nin`** — 1×1 convolutions and global average pooling. Compressed
   from `nin.md` (already the tightest file): NiN block, the two surviving
   ideas, the phone-RAM argument reworded without device-specific numbers.
   Keep the short training run (it's cheap and shows GAP works). Point back
   to ch7's 1×1 section rather than re-deriving.
3. **`sec_googlenet`** — multi-branch design and the stem/body/head
   vocabulary. Keep: stem/body/head terminology (still the universal
   architecture vocabulary — this is where the book introduces it), the
   Inception block implementation, the 1×1-bottleneck-before-expensive-conv
   idea, and a *shape-level* walkthrough. Cut: the b1…b5 stage-by-stage
   channel arithmetic (the text itself admits the ratios are arbitrary) and
   the full-model training run. Assemble the full net programmatically in
   ~15 lines for a `layer_summary` shape check only. Honest epitaph
   sentence: the Inception lineage is dead; multi-branch survives in
   ResNeXt/grouped convs (pointer) and, at train-time-only, in RepVGG
   (pointer to 8.7).
- End-of-file "what survived" mini-table: blocks → everything; 1×1 → channel
  mixing everywhere; GAP → default head; multi-branch → grouped conv.
- Figures: `arch-vgg.svg` (AlexNet vs VGG columns — replaces legacy
  `vgg.svg`), `arch-nin.svg`, `arch-inception-block.svg` (pilot figure,
  see figure-style.md §5). New deck (one deck for the merged file).
- Exercises: prune to the best of the three files' sets + the new
  depthwise-separable-VGG-block costing exercise if not already in ch7;
  replace googlenet ex. 1 (Inception-v2/v3 variants — busywork) with
  "add an SE gate to the Inception block" and "replace the Inception block's
  branches with a single 7×7 depthwise conv; compare cost".

### 8.3 `batch-norm.md` — keep + extend

- Do **not** touch the internal-covariate-shift debunking (lines ~891–936);
  it has aged perfectly.
- Rename displayed title to "Normalization Layers" (file/label unchanged).
- Add a `##` "Beyond Batch Normalization" after the concise implementation:
  - The two real problems of BN: minibatch coupling (small/variable batch,
    dense prediction) and train/serve discrepancy — the file already hints,
    make it explicit.
  - **GroupNorm** with code in all four tabs (`nn.GroupNorm` /
    `flax nn.GroupNorm` / `keras GroupNormalization` / gluon `nn.GroupNorm`):
    normalize within channel groups, batch-independent; the default in
    detection/segmentation heads and diffusion U-Nets. Cite `wu2018groupnorm`.
  - LayerNorm-in-convnets: one paragraph — channels-last per-position LN is
    ConvNeXt's choice, pointer to 8.6.
  - Normalizer-free nets: one paragraph on NFNets (`brock2021nfnet`) —
    adaptive gradient clipping + weight standardization can replace BN
    entirely; matters later for the scaling story in 8.8.
- Extend the deck accordingly; everything else unchanged.

### 8.4 `resnet.md` — keep + absorb DenseNet

- Core (function classes → residual block → ResNet-18 → training → ResNeXt)
  unchanged in substance. Two touch-ups: the Discussion's list of
  residual-connection consumers gains "transformer blocks in LLMs and
  diffusion models — by parameter count, most residual blocks in the world
  now live outside convnets"; exercise 4 (pre-activation ordering) gains
  "this ordering is essentially what ConvNeXt adopts — see
  `:numref:`sec_convnext``".
- New `##` subsection after ResNeXt: **"Concatenation instead of addition:
  DenseNet"** carrying `sec_densenet` (plus `fig_densenet_block` if
  referenced). Compressed from `densenet.md`: the Taylor-expansion framing in
  ~3 sentences, dense block + transition layer code (keep — it's a nice
  concatenation exercise), **no full-model build/training**, and the honest
  epitaph the old file already contains (memory cost of concatenation is why
  addition won at scale).
- Figures: `arch-resnet-block.svg` (two variants side by side, replaces
  legacy), `arch-resnext-block.svg`, `arch-densenet-block.svg` (compact),
  keep/replace `resnet18-90.svg` with `arch-resnet18.svg` (full-model column
  with stage containers + `(c, r)` annotations — good stress test of the
  style). New deck covers the merged scope.

### 8.5 `training-recipes.md` — new

The single highest-value addition. Thesis: **an unmodified ResNet-50 goes
76.1% → 80.4% ImageNet top-1 from the training recipe alone** ("ResNet
strikes back", `wightman2021resnet`: A1 = 600 epochs/LAMB/BCE + heavy
augmentation; A2 = 300; A3 = 100 epochs/78.1%) — a bigger jump than most
"architecture generations", and the reason pre-2021 and post-2021 paper
numbers are incomparable.

Structure:
1. **What changed between 2015 and 2022 recipes.** Table: optimizer
   (SGD+momentum → AdamW/LAMB; cite `loshchilov2019adamw`), schedule (step →
   cosine + warmup), augmentation (flips/crops → RandAugment + Mixup +
   CutMix + random erasing; cite `cubuk2020randaugment`, `zhang2018mixup`,
   `yun2019cutmix`), regularization (weight decay only → + label smoothing +
   stochastic depth `huang2016stochasticdepth`), duration (90 → 300–600
   epochs), evaluation (last checkpoint → EMA of weights).
2. **Implement the pieces** (code teaches): label smoothing (a loss
   one-liner), cosine-with-warmup schedule (plot it), Mixup/CutMix (~15
   lines each, framework-portable array ops on the batch), stochastic depth
   (a wrapper around any residual block — reuse `d2l.ResNeXtBlock`), EMA (a
   ~10-line parameter-shadow class).
3. **The experiment**: same ResNet-18, Fashion-MNIST at 96×96, recipe A
   (2015: SGD, step decay) vs recipe B (modern: AdamW + cosine + label
   smoothing + Mixup, moderately longer). Report the delta in a small table.
   *Design note:* pilot this on the GPU box first and pick an epoch budget
   where the gap is clearly visible and reproducible (expect ~1–2 points at
   ~20–30 epochs; if Fashion-MNIST saturates too easily, downsize to 64×64 or
   subsample the training set — the *pedagogical* requirement is a visible,
   honest gap, stated as such in prose). Do not promise ImageNet numbers from
   our demo; cite RSB for those.
4. **Benchmark literacy** (prose): ImageNet top-1 is saturated and
   recipe-confounded; ImageNet-V2 drop is largely a label-noise artifact
   (ReaL labels close it — `beyer2020imagenetreal`, `recht2019imagenetv2`);
   COCO/ADE20K transfer is what still discriminates between backbones.
5. Exercises: ablate one ingredient at a time; schedule-vs-optimizer
   interaction; why BCE-with-soft-targets pairs with Mixup; EMA decay sweep.

Framework parity: pytorch/jax/tf all have AdamW and the array ops needed.
MXNet: verify `mx.optimizer.AdamW` exists in the 2.0 wheel (believed yes; if
not, implement the 10-line decoupled update or run the mxnet tab with SGD +
cosine and say so honestly — reduced-variant precedent is ch6 §6.5).
Label: `sec_training_recipes`. Figures: none needed beyond `d2l.plot`
outputs (schedule plot, accuracy table); optionally `arch-recipe-timeline.svg`
only if a figure genuinely helps — default no.

### 8.6 `convnext.md` — new

The marquee section: **ConvNeXt is a controlled ablation that turns
ResNet-50 into a network that matches Swin-T (82.0 vs 81.3 top-1 at equal
FLOPs)** — and every step uses a concept this book has already taught.
Structure:

1. **The modernization roadmap** (`liu2022convnet`, already in bib): narrate
   the steps with their approximate ImageNet deltas as a table — stage ratio
   (3,4,6,3)→(3,3,9,3); patchify stem (4×4 s4 conv — pointer to ch7
   padding/strides "patterns"); depthwise conv (ch7 §7.4) + width increase;
   inverted bottleneck (NiN/MobileNetV2 lineage); large 7×7 kernel (receptive
   field, ch7 formula; note gains saturate past 7×7); ReLU→GELU, fewer
   activations, fewer norms, BN→LN (8.3); separate downsampling layers.
   Frame explicitly: part recipe (8.5), part architecture — cite the RSB
   baseline to keep the comparison honest.
2. **Implementation**: ConvNeXt block (depthwise 7×7 → LN → 1×1 expand ×4 →
   GELU → 1×1 project, residual, optional layer scale + stochastic depth from
   8.5), patchify stem, stage assembly via the familiar `arch` tuple: a
   scaled-down ConvNeXt (atto-ish widths) trained on Fashion-MNIST 96×96 with
   the 8.5 modern recipe. All four tabs (flax: `nn.Conv` with
   `feature_group_count`; keras: `DepthwiseConv2D` + `LayerNormalization`;
   gluon: `num_group=channels`, `nn.LayerNorm`, GELU activation — verify
   gluon GELU name on the box).
3. **ConvNeXt V2 in brief** (`woo2023convnextv2`): masked-autoencoder
   pretraining for convnets (FCMAE, sparse conv) + Global Response
   Normalization to fix feature collapse; 88.9% top-1 public-data SOTA at the
   time; one paragraph + GRN as an exercise (it's ~5 lines).
4. **Large kernels, honestly** (one subsection, mention-level):
   RepLKNet 31×31 (`ding2022replknet`), SLaK 51×51 (`liu2022slak`),
   InternImage/DCNv3 deformable (`wang2023internimage`), UniRepLKNet
   cross-modal (`ding2024unireplknet`). Verdict: the *idea* (large effective
   receptive field) stuck; giant dense kernels did not become defaults;
   deformable/adaptive operators carried it further.
5. 2026 standing (short): still a default strong-CNN backbone (OpenCLIP
   ConvNeXt towers, detection/segmentation backbones).

Label: `sec_convnext`. Figures: `arch-resnet-vs-convnext-block.svg`
(**pilot figure**, two-accent comparison), `arch-convnext.svg` (full column
with patchify stem + stage containers). Exercises: implement GRN; swap 7×7↔3×3
and measure; count params vs ResNet-18; layer-scale ablation.

### 8.7 `efficient-convnets.md` — new

Where convnets still win outright: latency- and memory-constrained
deployment. Three parts:

1. **Depthwise-separable networks.** Recap the ch7 cost ratio; MobileNetV1
   idea (`howard2017mobilenet`), inverted bottleneck of MobileNetV2
   (`sandler2018mobilenetv2`) — note ConvNeXt reused it; implement a
   mini-MobileNet (DS-block stack) and train briefly on Fashion-MNIST;
   compare params/accuracy against the 8.2 VGG-style net at similar budget.
2. **Scaling and searching**: EfficientNetV2 (`tan2021efficientnetv2`,
   compound scaling + progressive resizing, NAS-found; ties back to
   `tan2019efficientnet` already in bib) and MobileNetV4
   (`qin2024mobilenetv4`, universal inverted bottleneck, cross-hardware
   Pareto) — prose only, no implementation.
3. **Structural re-parameterization** (`ding2021repvgg`): train a block with
   3×3 + 1×1 + identity branches (each with BN), then *algebraically fuse*
   into a single 3×3 conv for inference. **Implement the fusion and
   `allclose`-verify equivalence** — a perfect "code teaches" demo, pure
   weight algebra, portable to all four frameworks. One honest caveat
   paragraph: naive INT8 quantization of fused RepVGG collapses (75.1% →
   40.2%; quantization-aware variants fix it) — the real world adds
   constraints papers don't model. Mention MobileOne/FastViT
   (`vasu2023mobileone`, `vasu2023fastvit`) as the idea industrialized.
4. Closing prose: the 2026 edge landscape — plain depthwise convnets on the
   cheapest tier, conv-attention hybrids on flagship phones, pure ViT only
   with dedicated NPU budget.

Label: `sec_efficient_cnns`. Figures: `arch-dws-block.svg` (depthwise
separable block), `arch-repvgg-reparam.svg` (train-time three-branch →
inference-time single conv; two-panel same-scale). Exercises: fuse a
Conv-BN pair by hand and verify; derive the DS cost ratio for k=5;
latency-vs-FLOPs (why they diverge; memory-bound ops).

### 8.8 `cnn-design.md` — keep + rewrite closing

- AnyNet/RegNet content stays (trim prose ~20%, code unchanged — it teaches
  stem/body/head and empirical design methodology, and ConvNeXt/8.6 hangs
  off that vocabulary). Add one framing sentence: RegNet's "distributions
  over networks" thinking is the ancestor of today's scaling-law-driven
  design.
- SE-networks: expand from one sentence to a short paragraph + tiny code
  snippet (channel attention in ~8 lines) since SE gates persist across
  EfficientNet and friends. Optional; skip if the section runs long.
- **Rewrite the Discussion (currently lines ~363–373) as a `##` "The Big
  Picture: ConvNets and Transformers"**, replacing the dated "hardware
  widened the gap" close. Content:
  - The scaling resolution: NFNets trained at JFT-4B compute match ViTs at
    equal compute — "ConvNets match Vision Transformers at scale"
    (`smith2023convnets`); the CNN-vs-ViT gap was mostly recipe + scale, not
    representation.
  - The 2026 division of labor: ViT owns foundation-scale pretraining and
    multimodal stacks (engineering interop with the transformer ecosystem);
    convnets own edge/latency, small-data, and much dense prediction
    (nnU-Net still wins controlled medical-segmentation benchmarks —
    `isensee2021nnunet`); conv stems persist inside transformers (Whisper —
    `radford2023whisper`).
  - The same story elsewhere: diffusion models moved U-Net → DiT
    (`peebles2023dit`) at the frontier while conv U-Nets persist in
    deployed/smaller systems.
  - Replace the LAION-5B namedrop (dataset was withdrawn amid 2023 CSAM
    findings; don't cite it uncritically) with a neutral "billion-scale
    image–text corpora" phrasing.
  - Close the chapter: inductive bias is a data-efficiency dial, not a
    ceiling; pointer to `chap_attention-and-transformers`.
- Update exercises: keep 1–3; replace ex. 4 (MLP design space) with "apply
  the AnyNet methodology to a ConvNeXt-block design space" tying 8.8 to 8.6.
- Figure: replace legacy `anynet.svg` with `arch-anynet.svg` (gallery style,
  stage containers + `(c, r)` labels). Keep `regnet-fig.png` (reproduced
  paper figure, exempt).

## 3. Bibliography additions

Add to `d2l.bib` (grep for near-duplicates first; some CV-chapter entries may
already cover mixup/cutmix/randaugment — reuse existing keys if present).
Verify metadata against arXiv before committing:

`wightman2021resnet` (2110.00476), `woo2023convnextv2` (2301.00808),
`ding2021repvgg` (2101.03697), `ding2022replknet` (2203.06717),
`liu2022slak` (2207.03620), `ding2024unireplknet` (2311.15599),
`wang2023internimage` (2211.05778), `vasu2023mobileone` (2206.04040),
`vasu2023fastvit` (2303.14189), `tan2021efficientnetv2` (2104.00298),
`qin2024mobilenetv4` (2404.10518), `brock2021nfnet` (2102.06171),
`smith2023convnets` (2310.16764), `peebles2023dit` (2212.09748),
`wu2018groupnorm` (1803.08494), `howard2017mobilenet` (1704.04861),
`sandler2018mobilenetv2` (1801.04381), `chollet2017xception` (1610.02357),
`loshchilov2019adamw` (1711.05101), `huang2016stochasticdepth` (1603.09382),
`zhang2018mixup` (1710.09412), `yun2019cutmix` (1905.04899),
`cubuk2020randaugment` (1909.13719), `beyer2020imagenetreal` (2006.07159),
`recht2019imagenetv2` (1902.10811), `yu2016dilated` (1511.07122),
`isensee2021nnunet` (Nature Methods 18, 2021), `radford2023whisper`
(2212.04356), `touvron2022deit3` (2204.07118), optional `zhang2019making`
(blur-pool, 1904.11486).

## 4. Config & promotion mechanics (order matters)

Follow the ch6 promotion playbook
(`chapter_builders-guide/_outline/promotion-notes.md`) and the two standing
memories: **verify staging before commit** (never `git add 2>/dev/null`;
diff the staged set against the intended change list) and **grep paths
before renaming files**.

1. `tools/d2l_preprocess.py` `CHAPTER_NUMBERING`: replace the ch8 block with
   the new 8.0–8.8 file list (numbers above).
2. `_quarto.yml`: replace the ch8 `.qmd` list to match.
3. Repo-wide grep for the retired stems (`nin`, `googlenet`, `densenet`,
   plus `vgg` if renamed) across `tools/ docs/ _quarto.yml Makefile
   .vscode-extension/` — `tools/runtime_env.py` `MULTI_GPU_NOTEBOOKS` has no
   ch7/ch8 entries today (verified), but check anyway; also
   `docs/notebook-scheduler.md` examples and any `d2l/` docstring
   "Defined in" references (those regenerate via `make lib`).
4. `make lib` after content lands if any `#@save` moved (none is planned to
   move; ResNeXtBlock stays in `resnet.md`; verify byte-identity of carried
   `#@save` blocks like ch6 did — zero downstream churn expected).
5. Notebook generation/execution/capture per framework; **capture with
   explicit `--frameworks pytorch,jax,tensorflow,mxnet`** (comma-separated;
   never bless a partial run over the others' store).
6. Delete retired outputs manifests + orphaned legacy `img/*.svg` (grep for
   references first — including slide decks and `chapter_computer-vision`).
7. Full battery: `lint_source.py`, preprocess `. .` check, `audit_outputs.py
   --verify-fresh`, `make html`, `make pdfs` (watch the `$`-before-digit and
   caption-`]` tripwires from `docs/build-system.md` §6.6), slides build for
   all rewritten files.

## 5. Framework-parity risk register

| Risk | Where | Mitigation |
|---|---|---|
| MXNet AdamW availability | 8.5 | verify on box; else hand-rolled decoupled update or honest SGD+cosine variant |
| gluon GELU naming | 8.6 | `nn.Activation('gelu')` or `npx.leaky_relu(act_type='gelu')` — check wheel |
| keras 3 GroupNormalization axis semantics | 8.3 | test channels-last explicitly |
| flax stochastic depth (rng plumbing) | 8.5/8.6 | use deterministic scaling variant at eval; keep the implementation minimal |
| Fashion-MNIST too easy to show recipe gap | 8.5 | pilot on box; shrink train set or resolution until the gap is visible and honest |
| MXNet DataLoader thread caps | all | scheduler already caps MXNET_GPU_SLOTS=2; nothing to do |

Where a framework genuinely can't express a piece, ship a reduced variant
with an explicit `:begin_tab:` note (ch6 §6.5 precedent) rather than faking
parity.
