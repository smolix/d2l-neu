# Review — §8 Modern Convolutional Neural Networks (existing-content pass)

*Implementation spec for the AUTHORING agent. Scope: the **8 source `.md`
pages that already exist** in `chapter_convolutional-modern/`
(`alexnet`, `vgg`, `nin`, `googlenet`, `batch-norm`, `resnet`, `densenet`,
`cnn-design`). I review teaching quality, technical correctness, and currency
of framing **within each page**. Net-new architectures / notebooks (e.g. a
mobile/efficient-nets page, a normalization-zoo page, a transfer-learning page)
are the SOTA-gap teammate's job — I only flag dated framing inside the existing
pages and forward-pointers that belong in them.*

*Source `.md` is the source of truth; `.qmd` is generated, never edited. These
are GPU notebooks — outputs were checked in `outputs/pytorch/chapter_convolutional-modern/<file>.json`.
Primary lens PyTorch 2.x; book is 4-framework (pytorch / jax / tf / mxnet),
**MXNet kept as a co-equal tab**. No `---` em-dashes in authored prose.*

---

## Executive verdict

This is a strong, coherent section — arguably the best-arced part of the book.
It tells the modern-CNN story in true chronological order, each page builds an
architecture *in runnable code* across four frameworks, the shape-inspection +
train-on-Fashion-MNIST rhythm is consistent, and the slides are uniformly
excellent (often ahead of the prose). The "code computes, never draws" rule is
**already honored**: every code cell either constructs a net, prints a layer
summary, or trains; the only image-producing cells are `d2l.plot` loss curves
(which teach a computed result and are allowed), and all schematic figures are
pre-generated SVGs included with `![…](../img/…)`. The technical level is at or
above CS231n for the architectures it covers.

Three things hold it back from best-in-class, and they are the spine of this
spec:

1. **The batch-norm "why it works" story is still wrong** (highest-priority
   correctness item in the whole section). The page *critiques* internal
   covariate shift in its Discussion and even cites Santurkar et al. 2018 — but
   it never delivers the actual modern finding (BN **smooths the loss
   landscape** / reduces the Lipschitz constant of the loss and its gradients),
   and the **slide "Why it works" and the prose intro both still lead with the
   ICS intuition** as the explanation. A reader who skims comes away believing
   the discredited story. This must be reframed: intuition-first, *then* the
   landscape-smoothing result as the established correction.

2. **The section pre-dates the figure house-style and has no `mdl-` figures at
   all.** Every schematic is a legacy d2l SVG (`alexnet.svg`, `vgg.svg`,
   `nin.svg`, `inception.svg`, `functionclasses.svg`, `residual-block.svg`,
   `resnext-block.svg`, `densenet-block.svg`, `anynet.svg`) plus three raster
   PNGs (`filters.png`, `regnet-fig.png`, `regnet-paper-fig*.png`). There is no
   `tools/gen_mdl_convolutional-modern_figures.py`. Bringing the whole section
   to one consistent house style is a large, separate figure project; for this
   pass I flag **which figures are load-bearing and worth regenerating** vs.
   keep-as-is, and the **one raster that should become an SVG or be cut**
   (`regnet-fig.png`). Coordinate the full restyle with the figure teammate.

3. **Currency: the pages stop at ~2021 and under-forward-point.** Several pages
   end at the architecture and never connect to the 2022-2026 frame the reader
   lives in: the "architecture vs. training-recipe" disentanglement
   (**Bag of Tricks**, He et al. 2019; **ResNet strikes back**, Wightman et al.
   2021 — vanilla ResNet-50 jumps 76% → ~80.4% from training alone),
   **timm** as the de-facto model zoo, and **ConvNeXt** as the "CNNs modernized
   with Transformer-era *training and macro-design*" capstone. Most of these are
   one-paragraph forward-pointers, not new content.

**Grade: B+ / A−.** Correct and assignable today; the batch-norm fix is the one
true *correctness* defect, the rest is currency and figure-consistency polish.

A note on what is NOT a problem: I checked the training cells. They genuinely
*compute* (AlexNet/VGG/NiN/GoogLeNet/ResNet/DenseNet/RegNet layer summaries show
real shape pyramids; the batch-norm `gamma`/`beta` readout shows learned
values; ResNeXt/DenseNet block demos show real channel arithmetic). No "wall of
matplotlib" anywhere. Keep all of it.

Cross-page conventions worth fixing once, everywhere (detail in §"Cross-page"):
`flax.linen` is legacy (NNX is current); "TensorFlow" tabs are really Keras 3;
`LazyConv2d`/`LazyLinear`/`LazyBatchNorm2d` are PyTorch-experimental and used as
the default throughout; and the four-tab depth occasionally inverts (MXNet/TF
get more elaborate class scaffolding than PyTorch).

---

# Per-file specification

Each file: **verdict**, then **keep / fix / cut-or-trim / add** with the *actual
technical content* and the grounding source. Section names match the `##`
headings in the source.

---

## 1. `alexnet.md` — "Deep Convolutional Neural Networks (AlexNet)"

**Verdict: KEEP, light modernization.** The "ImageNet moment = data (ImageNet) +
compute (GPUs) + ReLU + Dropout + augmentation" story is framed well and is
still the right story. Historical context is honest (it explicitly says it
streamlines the two-GPU split, uses Fashion-MNIST as a stand-in, notes upsampling
28→224 is wasteful). Sections: Representation Learning / AlexNet / Training /
Discussion. Good density.

**Keep:**
- The classical-CV-pipeline framing (SIFT/SURF/HOG, "features were crafted")
  and the "representation was the bottleneck" pivot (§Representation Learning).
  This is the single best motivation for representation learning in the book.
- The honest streamlining note (lines 268-271, 305-306) and the Discussion's
  "Achilles heel = the two 4096 FC layers dominate params" (lines 488-489) —
  this sets up NiN/GAP perfectly. Keep and lean into it.
- The `layer_summary` shape-walk cell (`alexnet-capacity-control-and-preprocessing-2`);
  the committed output shows the real `224→54→26→…→6×6×256` pyramid. Keep.
- `fig_filters` (`filters.png`, first-layer learned filters from the paper). A
  photographic/scientific reproduction — keep as-is per the figures-house-style
  scope (do NOT redraw learned filters).

**Fix (currency / correctness):**
- **Hardware paragraphs are dated to ~2021 and partly stale** (lines 225-251:
  "NVIDIA's latest Ampere," "A100 … 300 TFLOPs BF16," "Apple M1," "Graviton 3").
  In 2026 this reads as old. Don't chase exact TFLOPs (they age instantly);
  reframe one paragraph to make the *durable* point (CPU = few fat
  latency-optimized cores; GPU = thousands of throughput cores; the gap has only
  widened with Hopper/Blackwell and the move to low-precision matmul/tensor
  cores) and drop the spec-sheet numbers, or move them into a single "as of
  2025" sentence. Source for the durable framing: any current GPU-architecture
  overview; the point is conceptual, not the numbers.
- **"the most important part of the pipeline was the representation … up until
  2012 the representation was calculated mostly mechanically"** is good; add one
  sentence forward-pointing that *learned representations* are now the default
  substrate for *all* of vision/language (foundation models), closing the loop
  the VGG page opens with `bommasani2021opportunities`.
- **Exercise 3 ("Why do engineers no longer report benchmarks on AlexNet?")** is
  good; keep. Consider adding an exercise that has the reader compute AlexNet's
  param count and confirm the FC layers hold ~96% of them (it motivates NiN).

**Trim:** the GPU-vs-CPU subsection (lines 203-243) is ~40 lines and a bit of a
digression now that the audience is GPU-native. Tighten to ~half; the cost/power
argument (power ∝ frequency², so many slow cores win) is the keeper.

**Figures:** `alexnet.svg` (LeNet→AlexNet) is load-bearing and fine; if the
section gets a house-style pass, regenerate it in the shared style, but it is
NOT a priority (it's a clean schematic already). `filters.png` keep as-is.

---

## 2. `vgg.md` — "Networks Using Blocks (VGG)"

**Verdict: KEEP, strongest of the "early" pages.** The **3×3-stacks-for-depth**
insight is made *correctly and quantitatively* — this is exactly right and still
foundational. The receptive-field/parameter argument is present in both prose
(lines 79-84: two 3×3 = one 5×5 receptive field at `2·9c²` vs `25c²` params;
three 3×3 = 7×7 at `27c²` vs `49c²`) and on a dedicated slide with the
arithmetic `r_L = 1 + Σ(k_ℓ−1)`. Sections: VGG Blocks / VGG Network / Training /
Summary.

**Keep:**
- The parameter/receptive-field arithmetic (lines 79-84) and the slide
  "Receptive field arithmetic." This is the page's reason to exist; it is
  precise and well done. Keep verbatim.
- The "architecture as a family / tuple of `(n_convs, channels)` blocks" framing
  (lines 161-166, 273-275) — VGG-as-a-family is the durable lesson and the slide
  recap nails it ("VGG, ResNet, EfficientNet, ConvNeXt all use it").
- The `log₂ d` depth-limit motivation for blocks (lines 71-75) is a nice,
  rigorous touch. Keep.

**Fix (currency):**
- The Summary's only forward-point is **ParNet** (lines 323-324,
  `Goyal.Bochkovskiy.Deng.ea.2021`, "competitive with a shallow architecture").
  ParNet is a minor curiosity in 2026; this is a weak note to end on. Replace /
  augment with the **durable** forward-point: the 3×3-stack orthodoxy was
  *deliberately revisited* by **ConvNeXt** (Liu et al. 2022), which found that a
  **large-kernel depthwise 7×7** conv (Transformer-inspired) is *better* once
  training and macro-design are modernized. The page already name-drops
  `liu2022convnet` at line 83 — promote it to a real one-sentence point: "the
  3×3 default held for ~8 years until ConvNeXt showed large depthwise kernels
  win at scale." Source: *A ConvNet for the 2020s*, Liu et al., CVPR 2022,
  arxiv.org/abs/2201.03545.

**Trim:** nothing major. The page is tight.

**Figures:** `vgg.svg` (AlexNet→VGG blocks) is load-bearing and clean. Keep
as-is for now; restyle with the section, not urgently.

---

## 3. `nin.md` — "Network in Network (NiN)"

**Verdict: KEEP — and ELEVATE the two ideas, FRAME the architecture as
historical.** This is exactly the right call per the brief: **1×1 convolutions**
and **global average pooling** are hugely influential and the page already says
so; NiN-the-architecture is a historical footnote and the page is honest about
it ("Despite never winning a major benchmark, NiN's ideas are in every ConvNet
that came after" — slide recap). Sections: NiN Blocks / NiN Model / Training /
Summary.

**Keep:**
- The two-problems framing (FC layers eat params + can't add nonlinearity early
  without destroying spatial structure) → two-ideas solution (1×1 conv + GAP).
  Lines 16-32. Clean and correct.
- The "1×1 conv = a fully-connected layer applied per pixel location" intuition
  (lines 70-73). This is *the* mental model for 1×1 convs and the rest of the
  section depends on it (bottlenecks in GoogLeNet/ResNeXt). Keep, and the
  cnn-design page should point back here.
- The GAP-as-classifier-head idea and the surprise that it doesn't hurt accuracy
  (lines 135-137, 284). Keep; this is the more influential of the two ideas.

**Fix (correctness / dates):**
- **Date inconsistency:** the prose cites `Lin.Chen.Yan.2013` (correct — arXiv
  1312.4400, Dec 2013) but the slide title says "Lin et al., 2014" (the ICLR
  publication year). Pick one convention across the section; the body's 2013 is
  fine, fix the slide to match. Source: arxiv.org/abs/1312.4400 (ICLR 2014).
- **Elevate GAP's modern reach explicitly.** The Summary says 1×1 and GAP
  "significantly influenced subsequent CNN designs" — make it concrete in one
  sentence: GAP is the standard classifier head in *every* subsequent net in
  this chapter (GoogLeNet, ResNet, DenseNet, RegNet all use it), and it is the
  hook that later enables **CAM/Grad-CAM** localization (Zhou et al. 2016) — a
  great "this idea kept paying off" note. 1×1 convs became the universal
  channel-mixing / bottleneck primitive (GoogLeNet reductions, ResNet/ResNeXt
  bottlenecks, depthwise-separable convs, SE blocks). Source for CAM: Zhou et
  al., *Learning Deep Features for Discriminative Localization*, CVPR 2016.

**Trim:** nothing; the page is appropriately short.

**Figures:** `nin.svg` (VGG vs NiN blocks) is load-bearing. Keep.

---

## 4. `googlenet.md` — "Multi-Branch Networks (GoogLeNet)"

**Verdict: KEEP the concepts, TRIM the hand-built depth.** This is the page the
brief flags for trim-vs-keep. My call: **keep, but cut roughly a third.**
GoogLeNet's *concepts* are still taught everywhere (multi-branch/multi-scale,
1×1 bottlenecks, the stem/body/head decomposition it introduced), so the page
earns its place. But the page currently builds the *full* 9-block, 5-stage
network with ~10 framework-specific `@d2l.add_to_class` cells full of magic
channel numbers (`Inception(192, (96, 208), (16, 48), 64)` ×9), and the prose
itself admits these numbers are "relatively arbitrary" (line 515-518). That is a
lot of student attention spent transcribing 2014 channel allocations that teach
nothing. Sections: Inception Blocks / GoogLeNet Model / Training / Discussion.

**Keep:**
- The **stem / body / head** decomposition (lines 12) — GoogLeNet introduced it
  and *every* later page (ResNet b1, AnyNet) relies on it. This is the most
  durable thing on the page. Keep and make sure cnn-design points back to it.
- The **Inception block** itself (`googlenet-inception-blocks`, all 4 tabs) and
  `fig_inception`: four parallel branches, 1×1 bottleneck reductions,
  channel-concat. This is the keeper. Keep verbatim.
- The Discussion's honest "cheaper to compute than predecessors; beginning of
  deliberate cost-vs-accuracy design; manual hyperparameters because no
  auto-tooling yet" (lines 576-582) — this is the bridge to cnn-design. Keep.

**Trim (this is the main action):**
- Collapse the **five `b1`-`b5` stage builders** (lines 204-513, the bulk of the
  page) into **`fig_inception_full` + the Inception block + one compact
  assembled `GoogLeNet`** that the `layer_summary` walks. Keep ONE worked stage
  to show the multi-block-then-pool pattern; relegate the exact channel
  allocations of all nine blocks to the figure caption / a single config
  list, not nine prose paragraphs + nine code cells per framework. This removes
  ~150 lines × 4 tabs of transcription with no conceptual loss and brings the
  page to the density of nin/vgg.
- The detailed per-block channel-ratio prose (lines 301-313, 356-371) is the
  trimmable core — it explains arithmetic (`2:4:1:1` etc.) that the reader
  cannot generalize from. Compress to "channels are split across branches by a
  hand-tuned ratio; see the figure."

**Fix (currency):**
- Exercise 1 already lists the Inception-v2/v3/v4 + BN + label-smoothing +
  residual lineage as *exercises* (lines 586-590). Good. Add one
  **forward-pointing sentence in the Discussion**: the multi-branch idea's
  living descendant is less Inception-vN and more the **multi-branch +
  concat/bottleneck template** that recurs in feature-pyramid and
  attention-style designs; the *specific* hand-tuned Inception block has largely
  been retired in favor of regular ResNet/ConvNeXt blocks. Keep it honest about
  limited 2026 relevance (the slide recap already gestures here).

**Figures:** `inception.svg` (the block) is essential — keep. `inception-full-90.svg`
(the full 22-layer net) is only worth keeping if the full build stays; if you
trim the build as recommended, this figure does the teaching *instead of* the
nine code cells, so KEEP it and lean on it. (`inception-full.svg` is an unused
duplicate — ignore.)

---

## 5. `batch-norm.md` — "Batch Normalization"  ⭐ highest-priority page

**Verdict: KEEP the mechanics, REWRITE the "why it works" story.** The mechanics
are excellent: the BN equation, the FC-vs-conv axis distinction, the from-scratch
implementation, the train-vs-eval (moving-average) distinction, and the
LeNet+BN demo are all correct and well taught. The defect is the *explanation*,
and it is split across three places that disagree with each other. Sections:
Training Deep Networks / Batch Normalization Layers (FC / Conv / LayerNorm /
Prediction) / Implementation from Scratch / LeNet with BN / Concise / Discussion.

### What's wrong, precisely
The page is in a half-migrated state on "why BN works":
- The **Discussion** (lines 877-957) *does* the right critique: it says ICS is a
  misnomer, cites Rahimi's "alchemy" talk and Lipton & Steinhardt 2018, and
  cites Santurkar et al. 2018 as "claiming BN's success comes despite exhibiting
  behavior opposite to the original paper's claims" (lines 927-930). Good — but
  it **states the old explanation is wrong without ever stating the new one.**
  Santurkar's actual finding (BN smooths the loss landscape) never appears.
- The **prose intro** (§Training Deep Networks, lines 67-79) still *leads* with
  the ICS intuition ("the inventors postulated … this drift in the distribution
  … could hamper convergence") as the motivating story, only hedging "while this
  is not quite the reasoning."
- The **slide "Why it works"** (lines 1006-1014) lists "gradients stay
  well-conditioned," "higher LRs," "mildly regularizing" — decent, but it
  *omits the landscape-smoothing mechanism entirely* and the slide deck's
  framing reads as "BN is magic that stabilizes." A skimmer never learns the
  corrected story.

### The fix (this is the single most important change in the section)
Reframe "why BN works" **intuition-first, then the established result**, in the
Discussion *and* fix the slide, and soften the intro's ICS lead:

1. **Intro (§Training Deep Networks):** keep the three motivating intuitions
   (preprocessing-inside-the-net, numerical stability, regularization-by-noise)
   — they are good *guiding* intuitions and the page is careful to call them
   that. Just demote the ICS sentence from "the motivation" to "the inventors'
   original (informal) motivation, which we'll see does not survive scrutiny
   (see Discussion)." Forward-point the Discussion.

2. **Discussion — replace the "we know it's not ICS but won't say what it is"
   gap with the actual finding.** Add (intuition-first):
   - **The empirical refutation:** Santurkar et al. *added non-stationary
     Gaussian noise (random mean/variance each step) AFTER the BN layer*,
     deliberately *reintroducing* internal covariate shift. The network's
     activations became *more* unstable than an unnormalized net's — yet it
     **trained just as fast as standard BN**. So distributional stability is not
     the mechanism. (This concrete experiment is far more convincing than the
     current vague "behavior opposite to the claims.")
   - **The actual mechanism:** BN **makes the optimization landscape
     significantly smoother**. Formally, it **reduces the Lipschitz constant of
     the loss** and improves the *predictiveness* of the gradient (it reduces
     the second-order/β-smoothness term, so the gradient stays accurate over a
     larger step — which is *why* larger learning rates become safe). State this
     as the current best-supported explanation, still flagged as an active area
     (the page's existing epistemic-humility framing — "separate intuitions from
     established fact" — is great and should stay).
   - Keep the existing "noise injection → regularization," the Teye/Luo Bayesian
     connections, and the moderate-minibatch (50-100) sweet-spot discussion —
     those are correct and complementary.
   Source: Santurkar, Tsipras, Ilyas, Madry, *How Does Batch Normalization Help
   Optimization?*, NeurIPS 2018, arxiv.org/abs/1805.11604; lab summary with the
   noise-injection figure: gradientscience.org/batchnorm/. Original BN: Ioffe &
   Szegedy 2015, arxiv.org/abs/1502.03167.

3. **Slide "Why it works" + "Recap":** add one bullet making the mechanism
   primary — "**Why:** it *smooths the loss landscape* (lowers the loss's
   Lipschitz constant and makes gradients more predictive), which is what lets
   you use higher learning rates — *not* the originally-claimed 'internal
   covariate shift.'" This one bullet is the highest-value slide edit in the
   section.

### Also fix on this page
- **The train/eval bug — elevate it.** The page *explains* the train-vs-eval
  distinction correctly (lines 160-180, 255-271) and the from-scratch code keys
  on `torch.is_grad_enabled()` / `autograd.is_training()` / `deterministic`. But
  this is **the** classic BN bug in practice (forgetting `model.eval()` →
  garbage inference, BatchNorm using batch stats at test time). Add an explicit
  named callout: "**A common and costly bug:** running inference without
  switching to eval mode, so BN normalizes single test examples by their own
  (degenerate) statistics. In the frameworks, `model.eval()` / `training=False`
  flips this." This is high-value, in-lane, one paragraph. (CS231n and every
  practitioner guide stress it.)
- **LayerNorm subsection (lines 234-253) is a good forward-point — keep and
  sharpen.** It already motivates LN as batch-independent and scale-invariant.
  Add the *why it matters now*: **LayerNorm is the normalization of the
  Transformer era** (it is what the attention chapters will use), precisely
  because it is batch-size- and train/eval-independent. The slide already says
  "LayerNorm (per-example, used in Transformers)" — bring that into the prose.
- **GroupNorm — add as a one-line forward-point (currently absent).** The page
  gives BN and LN but not GN, and GN is the standard answer to BN's biggest
  practical failure mode: **BN degrades badly at small batch sizes** (the stats
  get noisy). One sentence: "When the batch is tiny (detection/segmentation at
  high resolution, where a batch may be 1-2 images), BN's statistics are too
  noisy; **GroupNorm** (Wu & He 2018) normalizes over channel *groups*
  independently of batch size and is the common fix." Source: Wu & He, *Group
  Normalization*, ECCV 2018, arxiv.org/abs/1803.08494 (ResNet-50: GN beats BN by
  ~10% error at batch size 2). This sets up the normalization story the
  Transformer section needs without leaving the CNN lane.
- The Discussion already has the nice `wang2022removing` "for robustness,
  consider removing BN" pointer (line 957) — keep.

**Keep (do not lose):**
- The from-scratch `batch_norm` + `BatchNorm` module split and the
  "math-in-a-function, bookkeeping-in-the-layer" design-pattern note (lines
  389-402). Genuinely good pedagogy.
- The `momentum`-is-a-misnomer note (line 556) and the **Flax momentum
  convention comment** (lines 816-818: Flax decays OLD stats at 0.99, PT/MX use
  0.1-on-NEW = 0.9-decay-of-OLD) — that is exactly the kind of cross-framework
  honesty that makes this book valuable. Keep.
- The `gamma`/`beta` readout cell (`…-4`) showing the layer *learned* a
  non-trivial scale/shift. Keep — it teaches a computed result.

**Trim:** the FC-layer BN subsection can lose a sentence or two of repetition
with the conv subsection, but it's minor. The page is long but earns it.

**Figures:** none currently (the only asset is the training loss curve, which is
fine). **Consider adding ONE house-style figure** if the section gets a figure
pass: a schematic of "what axes BN vs LN vs GN normalize over" (the canonical
N×C×H×W box diagram from the GroupNorm paper). That single figure would teach
the normalization-family relationship better than prose. Flag to the figure
teammate as the highest-value *new* figure in the section; not required for the
correctness fix.

---

## 6. `resnet.md` — "Residual Networks (ResNet) and ResNeXt"  ⭐ anchor page

**Verdict: KEEP — strongest architecture page; tighten currency and add two
in-lane pointers.** The brief asks this to be the strongest page, and it nearly
is. The function-class / nested-classes motivation is rigorous and genuinely
illuminating (the `F ⊆ F'` argument for why "add a layer that can be identity"
guarantees non-degradation). The residual block, the 1×1 shortcut for
shape-matching, the degradation-problem framing, and ResNeXt (grouped conv /
cardinality with the correct cost arithmetic) are all present and correct.
Sections: Function Classes / Residual Blocks / ResNet Model / Training / ResNeXt
/ Summary and Discussion.

**Keep (this is the gold):**
- The **Function Classes** section + `fig_functionclasses` (nested vs non-nested
  function classes). This is the best *conceptual* justification of residual
  learning in any textbook — most treatments just say "skip connections help
  gradients." Keep verbatim. The inductive-bias reframing in the Summary
  ("changes simple-functions from `f(x)=0` to `f(x)=x`", lines 753) is the
  payoff. Keep.
- The residual block (`resnet-residual-blocks-1`, all 4 tabs) with the
  auto-enable-1×1-on-downsample logic and the explanatory comment. Keep.
- **ResNeXt** (§subsec_resnext) with the grouped-conv cost arithmetic
  (`O(c_i·c_o) → O(c_i·c_o/g)`, lines 572) and the block-diagonal-matrix
  interpretation (line 772). This is the cardinality story done right and it
  *sets up cnn-design's AnyNet*. Keep — it is load-bearing for page 8.
- The honest training note (lines 521-523: large train/val gap → "more data
  would help"). Keep.

**Fix (correctness emphasis + currency):**
- **The "why it trains" mechanism is under-stated relative to the function-class
  story.** The page leads (correctly) with expressiveness (nested classes), but
  the *optimization* reason residual nets train — **gradients flow through the
  identity shortcut undiminished**, defeating the vanishing-gradient/degradation
  problem — is only implicit ("inputs can forward propagate faster," lines
  116). The slide says it crisply ("Gradients flow through the skip at full
  strength"); bring that into the prose as a co-equal reason. The two together
  (identity is *representable* AND its *gradient path is clean*) are the full
  answer. Pair with the original degradation evidence (He et al. 2016: a 56-layer
  plain net has *higher training error* than a 20-layer one — the motivating
  plot). Source: He et al., *Deep Residual Learning*, CVPR 2016,
  arxiv.org/abs/1512.03385.
- **Pre-activation and the bottleneck block are exercises only (lines 781-782).**
  Both are production-standard (every ResNet-50/101/152 uses the
  1×1→3×3→1×1 bottleneck; pre-activation is the v2 default). Promote at least the
  **bottleneck block** from exercise to a short body subsection or a worked
  paragraph — it is what "ResNet-50" actually means and the reader will meet it
  constantly. Pre-activation can stay a forward-point with the cite. Source:
  He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016,
  arxiv.org/abs/1603.05027.
- **Add the architecture-vs-training-recipe forward-point (high value, one
  paragraph).** This is the most important currency add in the section: the
  *same* ResNet-50 architecture went from 76.1% (2015 recipe) to **~80.4%**
  top-1 purely by modernizing the *training* (longer schedules, RandAugment,
  Mixup/CutMix, label smoothing, better LR schedules) — "ResNet strikes back"
  (Wightman et al. 2021) and "Bag of Tricks" (He et al. 2019). The lesson: a
  decade later ResNet-50 is *still* a competitive backbone, and much of
  "architecture progress" was actually training progress. This directly
  motivates cnn-design's "scalability/training trumps inductive bias" thesis and
  is the perfect bridge. Sources: Wightman, Touvron, Jégou, *ResNet strikes
  back*, arxiv.org/abs/2110.00476; He et al., *Bag of Tricks for Image
  Classification*, CVPR 2019, arxiv.org/abs/1812.01187.
- **timm pointer.** Where the Summary says ResNet remains "one of the most
  popular off-the-shelf architectures," name the modern *vehicle*: pretrained
  ResNets (and almost every CNN/ViT) are one line away via **timm**
  (`pytorch-image-models`), the de-facto model zoo. One sentence. Source:
  github.com/huggingface/pytorch-image-models.

**Trim:** nothing. The page is long but every part earns its place. (If anything,
ResNeXt could get *one* more sentence connecting cardinality to the modern
depthwise-conv limit `g = c` — see below — rather than less.)

- **Optional depth-add (in-lane, ties ResNeXt to the rest of vision):** note that
  the **extreme of grouped convolution, `groups = channels`, is the depthwise
  convolution** that powers MobileNet/Xception/EfficientNet/ConvNeXt. ResNeXt's
  cardinality and depthwise-separable convs are the same idea at different
  granularity. One sentence in the ResNeXt discussion; it makes the efficient-net
  family (teammate's page) land later. Source: Chollet, *Xception*, CVPR 2017,
  arxiv.org/abs/1610.02357.

**Figures:** `functionclasses.svg`, `residual-block.svg`, `resnet-block.svg`,
`resnet18-90.svg`, `resnext-block.svg` are all load-bearing and good. Keep. If
the house-style pass happens, `functionclasses.svg` and `residual-block.svg` are
the two most worth regenerating in the shared style (they're the conceptual
ones); the architecture diagrams can stay.

---

## 7. `densenet.md` — "Densely Connected Networks (DenseNet)"

**Verdict: KEEP, honest and correct; minor currency.** Dense connectivity,
feature reuse, and the concatenation-vs-addition contrast are well taught, and
the page is **honest about the memory cost** (the brief's specific ask): the
Discussion explicitly says concatenation "leads to heavy GPU memory consumption"
and points to memory-efficient implementations (`pleiss2017memory`), and an
exercise has the reader measure it empirically. Sections: From ResNet to DenseNet
/ Dense Blocks / Transition Layers / DenseNet Model / Training / Summary.

**Keep:**
- The **Taylor-expansion analogy** (lines 47-59): ResNet = `x + g(x)` (two terms);
  DenseNet = keep *all* the terms via concatenation. It is a genuinely elegant
  framing and intuition-first. Keep.
- The explicit channel-arithmetic demo (`densenet-dense-blocks-3`: 3 + 10 + 10 =
  23 channels) and the **growth-rate** definition (line 214). Keep — teaches a
  computed result.
- The dense-block / transition-layer decomposition and the "transition layers
  rein in the channel explosion via 1×1 conv + avg-pool" point. Keep.
- The memory-cost honesty (lines 523-527) and the empirical exercise (lines
  534-536). Keep — this is exactly the intellectual honesty the book wants.

**Fix (currency / framing):**
- **Position DenseNet accurately for 2026.** The page (and slide) imply DenseNet
  is a live competitor ("competitive ImageNet accuracy with far fewer
  parameters"). True in 2017; in 2026 DenseNet is **less used in practice than
  ResNet** precisely because the concat pattern is memory- and
  bandwidth-unfriendly on modern accelerators (fewer-params ≠ faster/less-memory:
  the activations and the gather/concat traffic dominate). Add one honest
  sentence: DenseNet's *idea* (feature reuse) is influential and lives on, but
  ResNet-style additive blocks won in practice for hardware-efficiency reasons —
  a concrete instance of the "params aren't the cost that matters; memory
  bandwidth is" lesson the AlexNet page started. This also reinforces the
  section's recurring compute-vs-params theme. (No new source needed beyond the
  page's own `pleiss2017memory`; the framing is the point.)

**Trim:** the four-framework `DenseBlock`/`TransitionBlock`/`__init__` cells are
necessarily repetitive but fine; no trim needed (the page is already the
shortest architecture page).

**Figures:** `densenet-block.svg` (ResNet-add vs DenseNet-concat) and
`densenet.svg` (dense connectivity growing with depth) are load-bearing. Keep.

---

## 8. `cnn-design.md` — "Designing Convolutional Network Architectures"

**Verdict: KEEP — the design-space idea is excellent and timeless; the *framing*
needs a currency refresh and the section's roadmap hook lives here.** This is the
section's existing 2020-era SOTA (AnyNet/RegNet design spaces) and it is *very*
good pedagogy: it teaches the meta-skill (search over *distributions* of
networks, not a single net), it is intuition-first (the empirical-CDF /
"majorization" argument), and it derives interpretable rules (tie bottleneck
ratio and group width; grow width/depth across stages; `k=1`, i.e. no
bottleneck). Sections: AnyNet Design Space / Distributions and Parameters /
RegNet / Training / Discussion.

**Keep:**
- The whole **design-space methodology**: the four assumptions (lines 261-266),
  the empirical CDF `F̂(e, Z)` and the majorization argument (lines 268-279),
  and the step-by-step constraint discovery (tie `k`, tie `g`, grow `c`, grow
  `d`) with the panel-by-panel evidence in `fig_regnet-fig`. This is the best
  treatment of NAS-adjacent design thinking in any intro textbook. Keep.
- The **stem/body/head + stages + blocks** generalization (lines 68-69) that
  unifies *everything* in the chapter (VGG → ResNeXt) under one template, reusing
  the ResNeXt block from page 6. This is the section's capstone synthesis. Keep.
- The honest "bottlenecks turn out not to help, `k=1` is best" finding (lines
  297) — a nice counter-to-intuition result. Keep.
- The Discussion's "scalability trumps inductive biases" thesis and the explicit
  forward-point to the vision-Transformer chapters (lines 363-373). Keep — this
  is the right note to end the *section* on (it hands off to the not-yet-written
  ViT section cleanly).

**Fix (currency — this page ages fastest because it claims to be the frontier):**
- **The Discussion's hardware/SOTA claims are 2021-dated.** Lines 366: "recent
  hardware optimizations (NVIDIA Ampere and Hopper) have only widened the gap in
  favor of Transformers." In 2026 this needs: (a) updating the hardware
  generation (Hopper → Blackwell), and more importantly (b) the **ConvNeXt
  correction**. The page cites `liu2022convnet` only as "ViT tricks *can* be
  backported to CNNs at higher cost" (line 366) — but ConvNeXt's actual headline
  is stronger and changes the takeaway: a **pure ConvNet, modernized in training
  and macro-design (patchify stem, depthwise 7×7, inverted bottleneck, fewer
  norms/activations, LayerNorm+GELU), *matches or beats* Swin Transformers** on
  ImageNet/COCO/ADE20K at comparable FLOPs. Reframe the Discussion to the more
  accurate 2026 picture: it is **not** "Transformers strictly win"; it is
  "architecture family matters less than scale + training recipe + a good design
  space — ConvNeXt and ViT converge." Source: Liu et al., *A ConvNet for the
  2020s*, CVPR 2022, arxiv.org/abs/2201.03545.
- **Make this page the explicit roadmap hook for the gap teammate.** The page
  already name-checks SENet (line 24), NAS/EfficientNet (lines 27-31), and
  ConvNeXt — these are exactly the *missing dedicated pages* the SOTA-gap
  teammate will scope (efficient/mobile nets, attention-in-convnets, NAS). Add
  one or two sentences positioning them as "covered in depth later / further
  reading," so this page reads as the on-ramp to the section's future rather
  than a dead end. (Coordinate the exact pointers with the teammate so the
  forward-references resolve.)
- **EfficientNet deserves one real sentence, not just a name-drop** (line 31).
  Its compound-scaling rule (scale depth/width/resolution *together* by
  `d=α^φ, w=β^φ, r=γ^φ` with `α·β²·γ²≈2`) is the natural complement to RegNet's
  design-space result and a clean idea. One sentence + cite. Source: Tan & Le,
  *EfficientNet*, ICML 2019, arxiv.org/abs/1905.11946.

**Trim:** the AnyNet build cells (`…-1/-2/-3`, four tabs) are necessary; keep.
Nothing major to cut.

**Figures (action needed):** `anynet.svg` is load-bearing and good — keep.
**`regnet-fig.png` and `regnet-paper-fig5/7.png` are RASTER screenshots of paper
figures.** `fig_regnet-fig` (the four-panel empirical-CDF comparison) is
*load-bearing* (the whole constraint-discovery argument refers to its four
panels) but is a low-res PNG with "Figure courtesy of" attribution. Two options
for the figure teammate: (a) re-plot the four CDF panels as a clean house-style
SVG from the paper's data/description (best — it's just four empirical CDFs), or
(b) keep the PNG but upgrade resolution. This is the **one figure-quality defect
worth fixing** in the section. (`regnet-paper-fig5/7.png` appear unused in the
`.md` — confirm and drop if so.)

---

# Cross-page observations (top items)

1. **The batch-norm "why it works" fix is the one true correctness bug in the
   section** and the single highest-value change. Everything else is currency or
   consistency. Do this first; it touches prose intro + Discussion + two slides
   on one page (§5) and has a crisp, well-sourced replacement (Santurkar 2018
   noise-injection + loss-landscape-smoothing).

2. **The section pre-dates the figure house-style: zero `mdl-` figures, no
   generator.** All schematics are legacy d2l SVGs + 3 raster PNGs. A full
   restyle is a separate figure project (coordinate with the figure teammate +
   `mdl-figure` / `figure-style-audit` skills). For THIS pass: (a) the one
   genuine defect is **`regnet-fig.png`** (raster, load-bearing — re-plot as
   SVG); (b) the highest-value *new* figure is a **BN/LN/GN "what axis is
   normalized" N×C×H×W diagram** on the batch-norm page; (c) the conceptual SVGs
   most worth regenerating in-style if/when the pass happens are
   `functionclasses.svg` and `residual-block.svg`. Keep all photographic/paper
   reproductions as-is (`filters.png`).

3. **Currency is a one-paragraph-per-page job, not new content, and it has a
   single through-line: "architecture matters less than scale + training recipe
   + design space."** The same three modern anchors recur and should be wired in
   consistently: **Bag of Tricks / ResNet-strikes-back** (training-recipe
   disentanglement; resnet page), **timm** (the model zoo; resnet page),
   **ConvNeXt** (CNNs modernized to match ViTs; vgg + cnn-design pages), with
   **GroupNorm/LayerNorm** as the normalization forward-points (batch-norm page).
   These make the section feel 2026 while staying entirely within modern
   *convnets* (no ViT content — that's the separate section). Bonus framing wins
   that are free: AlexNet's "FC layers dominate params" → NiN's GAP →
   GoogLeNet/ResNet bottlenecks → ResNeXt cardinality → depthwise convs is a
   clean efficiency thread already half-present; one sentence per page makes it
   explicit.

*Also flag (book-wide, not section-specific, defer to orchestrator):*
`flax.linen` → `flax.nnx` currency (every JAX tab); "TensorFlow" → "Keras 3"
labeling; `LazyConv2d`/`LazyLinear`/`LazyBatchNorm2d` used as the default
everywhere despite being PyTorch-experimental (consistent with the rest of the
book per the §6 review, so a book-wide decision, not this section's to make).

---

# Per-file dispatch manifest

Format: **file** → change → figures → code-cell edits + framework coverage + GPU
need → refs → deps. "No re-run" = prose/figure/slide only, no notebook
execution. GPU notes flag where a recommended change would require re-executing
(all these notebooks are GPU notebooks; a CPU box can render but defers
execution per the capability-aware gate).

### `alexnet.md`
- **Change:** modernize/trim the GPU-vs-CPU hardware subsection (drop stale
  TFLOPs, keep the durable cores/power argument); add one forward-point sentence
  (learned representations are now the universal substrate). Prose only.
- **Figures:** keep `alexnet.svg`, `filters.png` as-is. (Optional house-style
  regen of `alexnet.svg` — defer to figure teammate.)
- **Code-cell edits:** none. **GPU:** none (no re-run).
- **Refs:** none required (conceptual).
- **Deps:** none.

### `vgg.md`
- **Change:** replace/augment the ParNet end-note with a ConvNeXt forward-point
  (3×3 orthodoxy revisited by large depthwise kernels). Promote `liu2022convnet`
  (already cited) to a real sentence. Fix slide year drift if any. Prose +
  slide.
- **Figures:** keep `vgg.svg`.
- **Code-cell edits:** none. **GPU:** none.
- **Refs:** Liu et al. 2022 (ConvNeXt), arxiv.org/abs/2201.03545.
- **Deps:** none.

### `nin.md`
- **Change:** fix the 2013-vs-2014 date inconsistency (slide vs body); add one
  sentence elevating GAP's modern reach (universal head; enables CAM/Grad-CAM)
  and 1×1 as the universal channel-mixer/bottleneck. Prose + slide.
- **Figures:** keep `nin.svg`.
- **Code-cell edits:** none. **GPU:** none.
- **Refs:** Lin et al. 2013 (arxiv.org/abs/1312.4400); Zhou et al. 2016 (CAM).
- **Deps:** the "1×1 = per-pixel FC" framing is referenced by googlenet/resnet/
  cnn-design — keep it intact.

### `googlenet.md`
- **Change (largest structural edit):** TRIM the five-stage hand-build (~150
  lines × 4 tabs) to Inception-block + `fig_inception_full` + one worked stage +
  a compact assembled net for `layer_summary`; move the nine blocks' channel
  allocations into the figure/config. Add a Discussion sentence on limited 2026
  relevance (block retired; multi-branch+bottleneck template endures).
- **Figures:** keep `inception.svg`; **keep + lean on `inception-full-90.svg`**
  (it replaces the trimmed code). Ignore unused `inception-full.svg`.
- **Code-cell edits:** delete/merge `googlenet-googlenet-model-2..6` down to a
  minimal assembly across **all 4 tabs**; keep `…-inception-blocks`, `…-model-1`
  (stem), `…-model-7` (summary), `…-training`. **GPU: YES** — trimming/merging
  the model-construction cells changes code fingerprints; the page must be
  re-executed (all 4 frameworks) and re-captured (`make -B
  _notebooks/<fw>/chapter_convolutional-modern/googlenet.executed` then
  `capture-outputs FILES=chapter_convolutional-modern/googlenet.md
  --frameworks …`). Modest cost (one model, 10 epochs at 96×96, ×4 fw).
- **Refs:** Szegedy et al. 2015 (already cited); Inception-v2/3/4 already in
  exercises.
- **Deps:** verify the trimmed assembly still produces the layer-summary the
  slides reference (`@googlenet-googlenet-model-7`); update slide code-pointers
  if cells are renamed/removed.

### `batch-norm.md`  ⭐
- **Change (highest priority):** rewrite "why BN works" — (1) demote ICS in the
  intro to "original informal motivation, doesn't survive scrutiny"; (2) in the
  Discussion, replace the "it's-not-ICS-but-we-won't-say-why" gap with the
  Santurkar noise-injection refutation + the loss-landscape-smoothing /
  Lipschitz mechanism; (3) fix the slide "Why it works" + "Recap" to make
  smoothing the primary reason. Also: add a named **`model.eval()` train/eval
  bug** callout; sharpen LayerNorm as the Transformer-era norm; add a one-line
  **GroupNorm** small-batch forward-point. Prose + slides.
- **Figures:** none required. **Optional high-value NEW figure:** BN/LN/GN
  "normalized-axis" N×C×H×W diagram → would be `img/mdl-bn-norm-axes.svg` via a
  new/extended generator — **flag to figure teammate**, not required for the
  correctness fix.
- **Code-cell edits:** none (mechanics code is correct). **GPU: none** for the
  text fix (no code change → no re-run). (If the optional figure is added, it's a
  committed SVG generator, still no notebook re-run.)
- **Refs:** Santurkar et al. 2018 (arxiv.org/abs/1805.11604; lab post
  gradientscience.org/batchnorm/); Ioffe & Szegedy 2015
  (arxiv.org/abs/1502.03167); Wu & He 2018 GroupNorm (arxiv.org/abs/1803.08494);
  Ba et al. 2016 LayerNorm (already cited); Lipton & Steinhardt 2018 (already
  cited).
- **Deps:** none. This is the keystone change — schedule first.

### `resnet.md`  ⭐
- **Change:** (1) add the gradient-flows-through-identity optimization reason as
  co-equal with the function-class expressiveness reason, with the degradation
  evidence; (2) promote the **bottleneck block** from exercise to a short body
  treatment (pre-activation stays a forward-point); (3) add the
  architecture-vs-training-recipe paragraph (Bag of Tricks / ResNet-strikes-back,
  76→80.4%); (4) name **timm** as the model-zoo vehicle; (5) optional: connect
  ResNeXt cardinality to depthwise convs (`g=c`). Prose (+ optional slide bullet
  for the training-recipe point).
- **Figures:** keep all 5 SVGs. (House-style regen of `functionclasses.svg` /
  `residual-block.svg` if the pass happens — defer.)
- **Code-cell edits:** none *required* for prose changes. If the bottleneck block
  is promoted to a runnable cell (recommended), that's a **new code cell across
  4 tabs** → **GPU: YES** to execute + capture (a single small block shape demo,
  cheap; or keep it as a non-executed snippet to avoid a re-run — author's
  call). The training-recipe/timm pointers are prose-only (no re-run).
- **Refs:** He et al. 2016 ResNet (arxiv.org/abs/1512.03385); He et al. 2016
  pre-activation (arxiv.org/abs/1603.05027); He et al. 2019 Bag of Tricks
  (arxiv.org/abs/1812.01187); Wightman et al. 2021 ResNet-strikes-back
  (arxiv.org/abs/2110.00476); timm (github.com/huggingface/pytorch-image-models);
  Chollet 2017 Xception (arxiv.org/abs/1610.02357, optional).
- **Deps:** ResNeXt block here is reused by cnn-design — keep its API stable.

### `densenet.md`
- **Change:** add one honest "for 2026" framing sentence (feature-reuse idea
  influential, but additive ResNet blocks won in practice on hardware-efficiency
  grounds; params ≠ the cost that matters). Keep the memory honesty. Prose (+
  optional slide tweak).
- **Figures:** keep `densenet-block.svg`, `densenet.svg`.
- **Code-cell edits:** none. **GPU:** none.
- **Refs:** none beyond existing `pleiss2017memory`.
- **Deps:** none.

### `cnn-design.md`
- **Change:** refresh the Discussion currency — update hardware generation; fix
  the ConvNeXt framing (pure ConvNet *matches* Swin, not just "tricks can be
  backported"); reframe takeaway to "family matters less than scale+training+
  design space; ConvNeXt and ViT converge." Add one real EfficientNet
  compound-scaling sentence. Position SENet/NAS/EfficientNet/ConvNeXt as the
  roadmap on-ramp to the gap teammate's future pages. Prose (+ slide recap
  tweak).
- **Figures:** keep `anynet.svg`. **Action: re-plot `regnet-fig.png` as a
  house-style SVG** (four empirical-CDF panels) — the section's one real
  figure-quality defect; flag to figure teammate. Confirm `regnet-paper-fig5/7.png`
  are unused and drop.
- **Code-cell edits:** none. **GPU:** none.
- **Refs:** Liu et al. 2022 ConvNeXt (arxiv.org/abs/2201.03545); Tan & Le 2019
  EfficientNet (arxiv.org/abs/1905.11946); existing SENet/NAS/EfficientNet cites.
- **Deps:** **coordinate forward-references with the SOTA-gap teammate** so the
  "covered later / further reading" pointers resolve to their new pages.

---

## Sequencing for the authoring agent
1. **`batch-norm.md` "why it works" rewrite** (correctness; no GPU; self-contained). Do first.
2. Prose-only currency + framing adds: **`resnet.md`**, **`cnn-design.md`**,
   **`vgg.md`**, **`nin.md`**, **`densenet.md`**, **`alexnet.md`** (no GPU; any order).
3. **`googlenet.md` trim** (needs a 4-framework re-run + re-capture; schedule the
   GPU work last and update slide code-pointers).
4. Hand the figure items (`regnet-fig.png` → SVG; optional BN/LN/GN axis figure;
   optional conceptual-SVG restyle) to the figure teammate via `mdl-figure` /
   `figure-style-audit`; none block the prose work.

## Sources
- Santurkar, Tsipras, Ilyas, Madry, *How Does Batch Normalization Help Optimization?*, NeurIPS 2018 — https://arxiv.org/abs/1805.11604 ; lab summary (noise-injection figure) — https://gradientscience.org/batchnorm/
- Ioffe & Szegedy, *Batch Normalization*, 2015 — https://arxiv.org/abs/1502.03167
- Wu & He, *Group Normalization*, ECCV 2018 — https://arxiv.org/abs/1803.08494
- He et al., *Deep Residual Learning*, CVPR 2016 — https://arxiv.org/abs/1512.03385
- He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016 — https://arxiv.org/abs/1603.05027
- He et al., *Bag of Tricks for Image Classification*, CVPR 2019 — https://arxiv.org/abs/1812.01187
- Wightman, Touvron, Jégou, *ResNet strikes back*, 2021 — https://arxiv.org/abs/2110.00476
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022 — https://arxiv.org/abs/2201.03545
- Tan & Le, *EfficientNet*, ICML 2019 — https://arxiv.org/abs/1905.11946
- Lin, Chen, Yan, *Network In Network*, ICLR 2014 — https://arxiv.org/abs/1312.4400
- Chollet, *Xception*, CVPR 2017 — https://arxiv.org/abs/1610.02357
- timm / pytorch-image-models — https://github.com/huggingface/pytorch-image-models
