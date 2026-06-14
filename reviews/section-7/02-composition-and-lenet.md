# Review — Composition & First Net (§7.4 channels.md + §7.5 pooling.md + §7.6 lenet.md)

*Implementation spec. Files: `chapter_convolutional-neural-networks/channels.md`
("Multiple Input and Multiple Output Channels"), `.../pooling.md` ("Pooling"),
`.../lenet.md` ("Convolutional Neural Networks (LeNet)"). Plus a **Section-7 arc &
cross-file gaps** analysis spanning all six files (why-conv, conv-layer,
padding-and-strides reviewed by a teammate; I read their headings/intros/summaries
for the arc and deep-reviewed my three). Primary lens PyTorch 2.x; book is
4-framework (pytorch/jax/tf/mxnet), MXNet kept as a co-equal tab. Source `.md` is
truth; I skimmed `outputs/pytorch/.../<file>.json` to confirm computed results.*

*This is a **change spec**, not a directional review: every item names the file +
section, what to add/cut/rewrite, the actual technical content to write, and the
grounding source. A downstream authoring agent should be able to execute it.*

---

## Executive verdict

These three files are the chapter's payoff: channels turns a single filter into a
**feature bank**, pooling **downsamples** and buys translation tolerance, and LeNet
**assembles the first working CNN** and trains it. The mechanics are correct: I
re-derived the shaded-cell arithmetic in every figure against the committed
PyTorch outputs and they match exactly — channels' `corr2d_multi_in` returns
`56`, the multi-output stack returns `[[56,72],[104,120]], [[76,100],[148,172]],
[[96,128],[192,224]]`, the 1×1-conv-as-matmul `assert` passes, pooling returns
`4,5,7,8` (max) and `2,3,5,6` (avg), and LeNet's `layer_summary` traces the exact
`28→28→14→10→5→400→120→84→10` pyramid the figure promises. The 1×1-convolution
section is genuinely good (per-pixel MLP framing + a verifying matmul equivalence),
and the LeNet slides are excellent and already forward-point to §8 in a way the
prose does not.

**Measured against the "assignable at a top-5 program, above most courses" bar,
the prose is a faithful 2021 treatment with three systematic problems:**

1. **The honest modern forward-points live only in the slides, not the prose.**
   The slides already say the right 2020s things — "modern nets mix pooling with
   strided convs and end with global average pool" (pooling deck), "modern CNNs
   replace the dense stack with global average pooling" (LeNet deck), the
   1×1-conv→bottleneck/MobileNet lineage (channels deck). **None of this is in the
   page bodies.** A reader of the rendered book (who does not see the decks) gets a
   1990s story. The single highest-value pattern across all three files is *promote
   the slides' modernity into the prose.*

2. **`pooling.md` actively misleads on the current state of the field.** Its
   Summary asserts "max-pooling is preferable to average pooling" as a flat fact,
   and the whole page frames pooling as essential. The actual 2020s consensus is
   the opposite at both ends: **strided convolution has largely replaced pooling
   for downsampling** (Springenberg et al. 2015; ConvNeXt 2022 has *no* max-pool),
   and **global average pooling**, an *average*, is the standard classifier head.
   CS231n states outright: "It seems likely that future architectures will feature
   very few to no pooling layers." The page never tells the reader this in the
   body; it needs a forward-pointing subsection and a softened Summary.

3. **The §8 handoff is asserted, not built.** LeNet's Summary gestures ("greater
   amounts of computation enabled more complex architectures... journey down this
   rabbit hole") but never names the **conv-encoder + head template** as the thing
   §8 scales, and never lays out the component-by-component substitution table that
   *is* §8's syllabus. The LeNet slides have exactly this table; the prose should
   too. This is the chapter's closing paragraph — it should land the bridge, not
   wave at it.

**Grades: channels.md B+ (strong, needs a depthwise/grouped-conv forward-point and
a small bias-arithmetic fix). pooling.md B− (correct but dated; the All-Conv
forward-point is non-negotiable at this bar). lenet.md B+ (clean and correct;
needs the explicit §8-template bridge and a GAP-vs-flatten aside).** None of the
fixes is large; all are in-lane; most are *promote-the-slide-into-the-prose*.

Calibration: I checked the load-bearing currency claims against primary sources
(below). The repo is not behind upstream d2l.ai — but upstream is itself a 2021
artifact here, and "match upstream" is below the bar this project set. The
opportunity to *exceed* is concentrated in the three forward-points above.

### Sources consulted (currency check)
- Springenberg, Dosovitskiy, Brox, Riedmiller, **"Striving for Simplicity: The All
  Convolutional Net," ICLR Workshop 2015**, arXiv:1412.6806 —
  https://arxiv.org/abs/1412.6806 . "max-pooling can simply be replaced by a
  convolutional layer with increased stride without loss in accuracy." **Not
  currently in `d2l.bib`** (the only `Springenberg` entry is the Gato/Reed et al.
  paper) — must be added; see channels/pooling dispatch.
- **CS231n, "Convolutional Neural Networks (CNNs / ConvNets)"** —
  https://cs231n.github.io/convolutional-networks/ . "Many people dislike the
  pooling operation and think that we can get away without it... It seems likely
  that future architectures will feature very few to no pooling layers."
- Lin, Chen, Yan, **"Network In Network," 2013**, arXiv:1312.4400 —
  https://arxiv.org/abs/1312.4400 . Origin of both the 1×1 conv and **global
  average pooling** as a classifier head. Already in `d2l.bib` as `Lin.Chen.Yan.2013`.
- Liu et al., **"A ConvNet for the 2020s" (ConvNeXt), CVPR 2022**,
  arXiv:2201.03545 — patchify stem (4×4 stride-4 conv) replaces the 7×7-stride-2 +
  max-pool stem; downsampling is strided conv. (Verified via the ConvNeXt
  literature; confirm exact arXiv id before citing.)
- LeNet papers already present and correctly keyed: `LeCun.Bottou.Bengio.ea.1998`
  (gradient-based learning / LeNet-5), `LeCun.Boser.Denker.ea.1989` (backprop-trained
  CNN), `LeCun.Jackel.Bottou.ea.1995`. Pooling history keys all valid:
  `Riesenhuber.Poggio.1999`, `Yamaguchi.Sakamoto.Akabane.ea.1990`,
  `Zeiler.Fergus.2013` (stochastic pooling), `Graham.2014` (fractional max-pool).

---

## File 1 — `channels.md` ("Multiple Input and Multiple Output Channels")

### 1.1 Current-state assessment

**What it teaches.** Multi-input channels (kernel grows a 3rd axis; sum
per-channel cross-correlations) with a hand-rolled `corr2d_multi_in`;
multi-output channels (stack `c_o` filter banks → 4-D kernel
`c_o × c_i × k_h × k_w`) with `corr2d_multi_in_out`; the **1×1 convolution** as a
per-pixel fully-connected layer across channels, verified by reshaping to a single
matmul (`corr2d_multi_in_out_1x1`); a Discussion with the
`O(h·w·k²·c_i·c_o)` cost and a ResNeXt block-diagonal forward-pointer.
Structure: 4 `##` sections (Multiple Input / Multiple Output / 1×1 / Discussion) +
exercises. Clean and well-sequenced.

**Strong, keep:**
- The **1×1-conv section (lines 184–269)** is the best thing in the file and is
  load-bearing for §8 (NiN, GoogLeNet, ResNet bottlenecks). The "fully connected
  layer applied at every pixel location" framing (lines 210–212), the
  weight-tying note (213–214), the explicit `c_o × c_i` parameter count, and the
  *verifying* `corr2d_multi_in_out_1x1` matmul are exactly right. The note that
  convs are followed by nonlinearities so 1×1s "cannot simply be folded into other
  convolutions" (216–218) is a subtle, correct point most treatments omit. Keep all.
- The **joint-feature caveat (lines 132–136)** — "rather than mapping a single
  channel to an edge detector, it may simply mean that some direction in channel
  space corresponds to detecting edges" — is unusually sophisticated and correct.
  Keep; it sets up the feature-visualization add (see arc §A.3).
- The **`O(h·w·k²·c_i·c_o)` cost + the 53-billion-MAC worked example (line 275)**
  is excellent quantitative grounding and motivates §8's efficiency tricks. Keep.
- `corr2d_multi_in` / `corr2d_multi_in_out` are tight and *teach* (they compute the
  figure's numbers, verified against `channels.json`). Keep.

**Dated / incomplete / to fix:**

- **The 1×1 → modern-architecture lineage is in the slide, absent from the prose.**
  The "Why 1×1 convs everywhere" deck (lines 426–439) lists bottlenecks (ResNet),
  pointwise convs (MobileNet depthwise-separable), and SE/attention channel mixing.
  The *body* stops at "sometimes included in the designs of complex deep networks"
  (line 191) with a bare cite. For a top-bar basics chapter this is the moment that
  makes 1×1 convs feel essential rather than curious. **Fix:** add a short closing
  paragraph to the 1×1 section (after line 218) promoting the slide's content —
  see dispatch C1.

- **Bias arithmetic is glossed.** The Discussion's parameter discussion and the
  slide's "$c_o c_i k_h k_w + c_o$" (line 390) include the bias, but the 1×1 body
  says the layer "requires $c_\textrm{o}\times c_\textrm{i}$ weights (plus the
  bias)" (line 215) without saying there are `c_o` biases (one per output channel).
  Minor, but at this bar the count should be exact and consistent with the slide.
  **Fix:** "requires $c_o \times c_i$ weights plus $c_o$ biases" — dispatch C2.

- **No depthwise / grouped convolution, even as a pointer.** The Discussion
  forward-points to ResNeXt via "block-diagonal" channel mixing (line 275) and
  Exercise 7 explores block-diagonal speedups — but the reader is never given the
  name **grouped convolution** (the block-diagonal case) or **depthwise
  convolution** (the extreme: one group per channel), which together with the 1×1
  *pointwise* conv constitute the **depthwise-separable** factorization that
  dominates efficient vision (MobileNet/EfficientNet). The 1×1 section is the
  natural home for the pointwise half; the Discussion is the home for the grouped/
  depthwise half. **Fix:** name them where the concepts already appear — dispatch C3.

- **Exercise 1 is excellent and unanswered in the body — by design, fine — but the
  body should plant the seed.** Ex. 1 ("two stacked kernels with no nonlinearity =
  one convolution") is the composition-of-linear-maps fact that motivates *why
  nonlinearities matter* and why VGG stacks small kernels. The 1×1 body already
  half-states the converse (nonlinearities prevent folding, line 216–218). No change
  required to the exercise; optionally cross-reference it from line 218 — dispatch C4
  (low priority).

### 1.2 Correctness notes (channels.md)
- Shape arithmetic all correct. `c_i × k_h × k_w` per output channel,
  `c_o × c_i × k_h × k_w` total: correct (lines 143–147).
- The 1×1 matmul equivalence is correct and the `assert ... < 1e-5` passes in
  `channels.json`. The JAX/TF tabs use the right per-framework RNG/shape idioms.
- `corr2d_multi_in_out` uses `d2l.stack(..., 0)` — correct across frameworks
  (the d2l shim dispatches). No issue.

---

## File 2 — `pooling.md` ("Pooling")

### 2.1 Current-state assessment

**What it teaches.** Max/avg pooling as a parameter-free sliding-window reduction;
the downsampling-and-receptive-field motivation; the translation-invariance story
(1-pixel shift, edge detector); a hand-rolled `pool2d`; framework `MaxPool2d`
with padding/stride defaults (stride = window); multi-channel pooling (per-channel,
channel count unchanged); a Summary with stochastic/fractional pooling pointers.
Structure: 4 `##` (Max/Avg / Padding & Stride / Multiple Channels / Summary) +
exercises. Correct throughout; verified against `pooling.json` (max `4,5,7,8`; avg
`2,3,5,6`; framework `MaxPool2d(3)` → `[[10]]`; padded stride-2 → `[[5,7],[13,15]]`).

**Strong, keep:**
- The **invariance worked example (lines 119–129)**: 2×2 max-pool still fires when
  the convolutional edge-detector output shifts by ≤1 element. Concrete and correct.
  Keep — but see the honesty fix below.
- The **history paragraph (lines 86–93)**: avg pooling as denoising downsampling,
  max-pooling from Riesenhuber-Poggio (cognitive neuroscience) with the
  Yamaguchi speech-recognition antecedent. Good scholarship; both keys valid. Keep.
- The **framework-defaults point (lines 221–223)**: pooling defaults stride =
  window (non-overlapping), unlike conv. Genuinely useful, often-missed. Keep.
- The **multi-channel contrast** (pooling pools per channel, does *not* mix like
  conv, lines 311–318) is exactly the right thing to contrast against channels.md.
  Keep.

**Dated / misleading / to fix — this is the file with the most work:**

- **[HIGH] The Summary overstates max-pooling and understates the field's move away
  from pooling.** Line 379: "of the two popular pooling choices, max-pooling is
  preferable to average pooling, as it confers some degree of invariance to output."
  And line 93: "In almost all cases, max-pooling ... is preferable to average
  pooling." Both are stated as flat facts. This is dated in two directions at once:
  (a) **average pooling is the standard final-layer operation** (global average
  pooling, Lin et al. 2013) in essentially every modern classifier head — ResNet,
  ConvNeXt, ViT; (b) **for downsampling, strided convolution has largely displaced
  pooling of either kind** (All-Conv 2015; ConvNeXt 2022 uses *no* max-pool). The
  page reads as if pooling-choice is the live question; the live question in 2026 is
  *whether to pool at all*. **Fix:** soften both flat claims to "max-pooling was the
  default for intermediate downsampling in classic CNNs" and add the forward-point
  subsection below — dispatch P1 + P2.

- **[HIGH] Missing: an honest "Pooling vs. strided convolution" forward-point in
  the BODY.** The "Where pooling sits in modern architectures" slide (lines
  518–530) already says it all — classic CNNs pool every few layers; ResNet/modern
  pool less and use `stride=2` convs; global average pooling ends the net. **None of
  this is in the page body.** This is the single most important add to the file and
  the prompt's explicit ask. **Fix:** add a new `##` subsection (before Summary)
  titled e.g. "Pooling versus strided convolution" that (i) states the All-Conv
  finding with the cite, (ii) gives the trade-off — pooling is parameter-free and
  injects a fixed invariance prior; strided conv *learns* its downsampling and
  preserves representational capacity at the cost of parameters/compute, (iii)
  introduces **global average pooling** as the modern classifier head with the NiN
  cite, (iv) notes ConvNeXt as the "no max-pool" datapoint, (v) quotes/echoes the
  CS231n stance that future nets may have very few pooling layers. Full content in
  dispatch P2.

- **[MED] Bib gap blocks P2.** Add the **All-Conv** entry to `d2l.bib`:
  ```
  @InProceedings{Springenberg.Dosovitskiy.Brox.ea.2015,
    title     = {Striving for Simplicity: The All Convolutional Net},
    author    = {Springenberg, Jost Tobias and Dosovitskiy, Alexey and
                 Brox, Thomas and Riedmiller, Martin},
    booktitle = {ICLR Workshop},
    year      = {2015},
    url       = {https://arxiv.org/abs/1412.6806}
  }
  ```
  Optionally add ConvNeXt (`Liu.Mao.Wu.ea.2022`, arXiv:2201.03545) if P2 cites it
  by reference rather than by name. — dispatch P3.

- **[MED] The intro motivation conflates two distinct jobs pooling does.** Lines
  9–39 open with "global question (does it contain a cat?)" → aggregation → coarser
  maps → receptive field, then pivot to translation invariance. These are *two*
  separable motivations (spatial reduction for receptive-field growth & compute, vs.
  local invariance) and the prose runs them together. The slide separates them
  cleanly ("Spatial aggregation" vs "Translation invariance," lines 420–426). **Fix:**
  restructure the intro to name the two jobs explicitly, mirroring the slide —
  dispatch P4 (low-risk clarity edit).

- **[LOW] The invariance claim deserves a one-line honesty caveat.** Max-pooling
  gives invariance *only* to sub-window shifts and only at that one scale; it is not
  a general invariance and stacked pooling's cumulative invariance is a known source
  of the "texture bias / loss of spatial precision" critique that motivated
  strided-conv and, later, attention. One sentence after line 129 keeps the section
  from overselling. — dispatch P5.

### 2.2 Correctness notes (pooling.md)
- `pool2d` valid-window arithmetic (`X.shape - p + 1`) correct; matches the
  hand-traced figure and `pooling.json`. The JAX functional-update and TF `Variable`
  variants are correct per-framework idioms. No bugs.
- The TF channels-last `:begin_tab:` notes (lines 204–207, 322–325, 370–375) are
  accurate and helpful. Keep.
- Exercises are strong (avg-via-conv; max-not-expressible-as-conv; max via ReLU;
  pooling cost; softmax pooling). Exercise 1 ("implement average pooling through a
  convolution") quietly anticipates the strided-conv equivalence and is a nice
  bridge to P2 — consider cross-referencing. No correctness issue.

---

## File 3 — `lenet.md` ("Convolutional Neural Networks (LeNet)")

### 3.1 Current-state assessment

**What it teaches.** LeNet-5 as conv-encoder (two conv→sigmoid→avgpool blocks) +
dense head (120→84→10); the history (LeCun, Bell Labs, ATMs); a 4-framework
`Sequential` implementation with Xavier init; a `layer_summary` shape-trace; and
training on Fashion-MNIST with cross-entropy + SGD. Structure: 3 `##` (LeNet /
Training / Summary) + exercises. Correct: the shape trace in `lenet.json` is exactly
`28→28→14→10→5→400→120→84→10`.

**Strong, keep:**
- **The shape-trace `layer_summary` cell (lines 254–295)** is excellent pedagogy —
  it makes the architecture *verifiable* against the figure, and the prose
  walk-through (lines 297–317) is clear. Keep; it is the model of "code that teaches."
- **The honest reproduction caveat (lines 216–222)**: softmax for the Gaussian
  decoder, sigmoid for scaled-tanh. Correct and well-judged. Keep.
- **The MNIST 32→28 trimming aside (lines 303–306)** is a delightful, correct
  historical detail ("save space when megabytes mattered"). Keep.
- The **"some ATMs still run LeCun & Bottou's 1990s code" hook (lines 39–40)** is
  great and current. Keep.
- The **slides are outstanding** — the "What 30 years of progress changed" table
  (lines 493–508) and the "LeNet sets the CNN template" framing (lines 412–422) are
  exactly the §8 bridge the prose lacks. They should be promoted (below).

**Dated / incomplete / to fix:**

- **[HIGH] The §8 handoff is asserted, not built — and the material to build it is
  sitting in the slides.** The Summary (lines 373–377) says LeNet "remains
  meaningful," compares error rates to ResNet, and promises a "rabbit hole" — but it
  never (a) names the **conv-encoder + head template** as the invariant §8 scales,
  nor (b) enumerates the component substitutions §8 actually performs. The
  "What 30 years of progress changed" slide already has the table:
  sigmoid→ReLU/GELU, avg-pool→max-pool/strided-conv, none→BatchNorm/LayerNorm,
  Xavier→He, ~60k→millions of params, dense head→global-average-pool. **Fix:** add a
  short paragraph (or promote the table) to the Summary that states the template
  explicitly and maps each LeNet component to its §8 successor *with the §8 section
  it is treated in* (AlexNet, VGG, NiN, BatchNorm, ResNet). This is the chapter's
  closing bridge and should land it. — dispatch L1.

- **[HIGH] "the template AlexNet later scaled up" is the framing the prompt asked
  for, and it is missing from the prose.** The intro (lines 23–41) frames LeNet
  historically but never says, in the body, "this is the template AlexNet (§8.1)
  scaled up with more data and compute." The LeNet "template" slide says it; the
  prose should open or close with it. **Fix:** add one sentence to the intro (after
  line 41) or the Summary: AlexNet is essentially LeNet made deeper and wider, run
  on GPUs over ImageNet — same skeleton, bigger everything. — dispatch L2.

- **[MED] The flatten bottleneck is in the slide, not the prose.** The "Two
  takeaways" slide (lines 444–452) makes the sharp, quantitative point that
  `400 × 120 = 48000` weights — the bulk of LeNet's parameters — live in the
  flatten→first-dense transition, and that **global average pooling** is how modern
  nets eliminate it. The body never says this. It is a perfect, concrete motivation
  for GAP that ties back to pooling.md's P2 add. **Fix:** add a sentence/short
  paragraph to the shape-trace discussion (after line 317) or Summary quantifying the
  flatten cost and forward-pointing to GAP. — dispatch L3.

- **[MED] No parameter count for the whole network.** At this bar, a first-CNN
  section should state LeNet-5's total parameter count (~60k, dominated by the dense
  head) — it makes the "CNNs are parsimonious vs MLPs" claim (lines 20–21) concrete
  and sets up the AlexNet/VGG explosion. The frameworks expose this trivially
  (`sum(p.numel() ...)` / `nn.tabulate` for JAX, already used). **Fix:** either a
  one-line prose figure or extend `layer_summary` to also print param counts —
  dispatch L4. (If adding a print, keep it computing, not drawing — it stays "code
  that teaches.")

- **[LOW] The training cell shows the curve but the prose under-reads it.** The
  Summary says LeNet "remains meaningful" and the slide claims "convolutional
  inductive bias clearly beats the dense MLP" — but the body never states the actual
  Fashion-MNIST result or compares it numerically to the MLP from §5. The training
  cell produces a loss/accuracy curve (the `d2l.Trainer` plot); the prose should name
  the ballpark val accuracy and contrast it with the §5 MLP, making the inductive-bias
  win quantitative rather than asserted. **Fix:** one or two sentences after line 371
  citing the achieved accuracy and the §5 comparison — dispatch L5.

### 3.2 Correctness notes (lenet.md)
- Architecture is correct and matches `lenet.json`. PyTorch uses `LazyConv2d`/
  `LazyLinear` (consistent with the rest of the book; note the §6 review flags
  `Lazy*` as experimental — out of scope here but worth a project-level decision).
- Conv1 padding=2 (SAME for 5×5) and Conv2 no-padding are correctly explained
  (lines 300–309). The channels-last note for TF/JAX (lines 96–98) is accurate.
- `init_cnn` Xavier-uniform on `Conv2d`/`Linear` is correct; the JAX tab uses
  `nn.initializers.xavier_uniform` consistently. No bugs.
- Exercises are strong and already point forward (Ex. 1 "modernize LeNet:
  max-pool + ReLU"; Ex. 4–5 "display activations / feed random noise" — the latter is
  a feature-visualization seed; see arc §A.3). Keep; Ex. 1 should be cross-referenced
  from the L1 substitution paragraph.

---

## Section-7 arc & cross-file gaps

*Spanning all six files: why-conv → conv-layer → padding-and-strides → channels →
pooling → lenet. The first three are deep-reviewed by a teammate; I read their
intros/headings/summaries for the arc and deep-reviewed channels/pooling/lenet.*

### A.1 Is the progression optimal?

**Mostly yes, and it should be kept.** The macro-arc is the canonical and correct
one: *motivation (why-conv) → operation (conv-layer) → output-shape control
(padding-and-strides) → feature banks (channels) → downsampling (pooling) → first
assembled net (lenet).* It moves from principle to mechanism to architecture, each
file consuming the previous. Specific arc strengths:

- **why-conv derives conv from invariance + locality first principles** and already
  carefully distinguishes **translation equivariance** (early layers) from
  **translation invariance** (after pooling + classification) — lines 100–110. This
  is a genuinely above-bar distinction most courses blur, and it correctly *sets up*
  pooling. Keep and lean on it: pooling.md's invariance story should explicitly cite
  back to this equivariance→invariance framing (`:numref:`sec_why-conv``), which it
  currently does not.
- **conv-layer already covers feature maps AND receptive fields** (its
  "Feature Map and Receptive Field" `##`, `:label:`field_visual``) and includes the
  Hubel-Wiesel / Field-1987 biological-filter figure plus a Kuzovkin-2018
  forward-point to trained-net features. **This substantially closes what would
  otherwise be the section's biggest gap** (see A.3). The arc is therefore better
  than a first read of "my three files" suggests.
- channels logically follows padding-and-strides (you need the single-channel
  operation and its shape arithmetic before generalizing the kernel to 4-D).

**One ordering wrinkle worth a deliberate decision (not a mandate):** the
**receptive-field** treatment is in conv-layer (file 2), but its *payoff* — "stack
layers / downsample to grow the receptive field" — is exactly pooling's motivation
(file 5) and padding-and-strides' stride motivation (file 3). The concept is
introduced 3 files before it pays off. This is defensible (receptive field belongs
with the conv definition), but the later files should **explicitly call back** to
`:numref:` the receptive-field discussion rather than re-motivating from scratch.
Concretely: pooling.md lines 13–19 re-explain receptive-field growth without citing
conv-layer's `field_visual` section — they should cross-reference it. — see dispatch
ARC1.

### A.2 The §8 (Modern CNNs) handoff — under-built and mis-placed

The handoff to §8 is the weakest seam in the section. It is currently:
- **index.md**: a forward-pointer ("In the next chapter ... full implementations of
  popular CNN architectures") — fine as a chapter promise.
- **lenet.md Summary**: a vague "rabbit hole" gesture (L1/L2 above).

What's missing at the *section* level is a single, explicit statement of **the
template §8 inherits and the axes along which it scales it** (depth, width,
normalization, activation, downsampling, head). The LeNet slides have this exactly
(the "30 years of progress" table and the "conv encoder + head" template framing) —
**the section's job is to get that into the lenet.md prose** (dispatch L1/L2), so the
last thing a reader sees before §8 is the map of §8. No new file needed; this is the
single most important cross-file fix and it lands in lenet.md.

### A.3 Section-level content gaps (ranked)

1. **[HIGHEST] The honest "pooling is on the way out / strided conv + GAP" story is
   section-wide and lives only in slides.** It surfaces in pooling.md (P2),
   channels.md (the 1×1→GAP-adjacent lineage), padding-and-strides (its slide
   already has the "Halve = standard downsample" strided-conv recipe and even the
   ViT patchify pattern), and lenet.md (L3, the flatten→GAP point). Right now a
   *book reader* (no slides) never learns that the field largely replaced pooling
   with strided convolution and the dense head with global average pooling. **This
   is the section's defining modernization gap and the prompt's central ask.** It
   should be fixed coherently: pooling.md gets the primary forward-point (P2);
   lenet.md echoes it for the head (L3); padding-and-strides' strided-conv-as-
   downsampler framing should be promoted from slide to prose by the teammate. One
   consistent narrative thread, three files. Grounding: All-Conv (1412.6806),
   CS231n, NiN (1312.4400), ConvNeXt (2201.03545).

2. **[HIGH] Feature visualization is *present but thin and dated* — upgrade, don't
   add from scratch.** conv-layer already has the receptive-field + Hubel-Wiesel/
   Field-1987 figure and a one-line Kuzovkin-2018 pointer (closing what would
   otherwise be a glaring hole). But the treatment stops at 1959–1987 biology + one
   2018 cite; it never shows *what trained CNN filters actually look like* in the
   modern feature-visualization sense (Zeiler-Fergus deconv 2014 — `Zeiler.Fergus`
   is in the bib only as stochastic-pooling, the viz paper is a *different* Zeiler
   work; distill.pub "Feature Visualization," Olah et al. 2017). This is a
   high-teachability, low-cost upgrade. **Recommendation:** in conv-layer (teammate's
   file) extend the `field_visual` section with one modern feature-visualization
   forward-point (early layers = edges/colors, mid = textures/patterns, late =
   object parts), citing the deconv/optimization viz line and distill.pub, and
   forward-point to it from lenet.md Ex. 4–5 (which already ask the reader to display
   activations). A **new pre-generated SVG** schematic of the
   edges→textures→parts hierarchy would be the ideal house-style figure — see
   "New figures" below. — dispatch ARC2 (primary owner: conv-layer/teammate; lenet
   cross-ref is L-side).

3. **[MED] "Why small stacked kernels beat one big kernel" is never stated, though
   the pieces exist.** conv-layer's receptive-field section shows depth grows the
   receptive field; channels' Exercise 1 shows two kernels compose; padding-and-
   strides quantifies shrinkage. The synthesis — *two stacked 3×3s have the
   receptive field of one 5×5 with fewer parameters and an extra nonlinearity*
   (the VGG argument) — is the natural capstone that ties three files together and
   directly motivates §8's VGG section. It is currently nowhere in the prose.
   **Recommendation:** one short paragraph, best placed at the end of channels.md's
   Discussion or conv-layer's receptive-field section, forward-pointing to VGG. —
   dispatch ARC3.

4. **[MED] No section-level cost/efficiency through-line.** channels.md has the
   excellent 53-billion-MAC example and a ResNeXt pointer; pooling has a cost
   exercise; lenet has no parameter count (L4). The section never connects these
   into the arc that *drives* §8 (compute/parameters → the efficiency innovations:
   1×1 bottlenecks, grouped/depthwise conv, GAP). The channels Discussion is the
   right anchor; lenet's param count (L4) is the concrete payoff. **Recommendation:**
   ensure L4 lands and have lenet's Summary connect "LeNet ~60k params, dominated by
   the dense head" to channels' cost discussion and §8's efficiency arc. — dispatch
   ARC4.

5. **[LOW] Redundancy: the framework padding/stride mechanics are taught twice.**
   padding-and-strides teaches conv padding/stride; pooling re-teaches the *same*
   padding/stride knobs for `MaxPool2d` (lines 193–309). This is mostly justified
   (pooling's defaults differ — stride = window), but the pooling padding/stride
   subsection could be trimmed to *just the difference* (defaults + the channels-last
   wrinkle) and `:numref:` back to padding-and-strides for the mechanics, saving ~40
   lines and tightening the arc. — dispatch ARC5 (optional, low priority).

### A.4 Arc verdict

**The six-file progression is sound and should be preserved — do not reorder.** The
section is closer to the bar than a per-file read suggests, because conv-layer
already carries the equivariance/receptive-field/feature-map material that would
otherwise be the section's biggest hole. The two things keeping it from best-in-class
are both *promotion problems, not authoring-from-scratch problems*: (1) the modern
"strided conv + GAP largely replaced pooling" narrative exists only in the slides and
must be promoted into the prose across pooling/padding/lenet (A.3 #1), and (2) the
§8 template-and-substitution bridge exists only in the LeNet slides and must be
promoted into lenet.md's Summary (A.2). Fix those two threads and add the
small-stacked-kernel synthesis (A.3 #3) and a modern feature-viz forward-point
(A.3 #2), and the section is a top-5-program treatment.

---

## New figures flagged (house-style, pre-generated SVG via `tools/gen_mdl_*`)

Per CLAUDE.md: schematic figures are pre-generated SVGs, **no drawing code in
notebooks**, one style per chapter, added via the **mdl-figure** skill. There is
currently **no** `tools/gen_mdl_*conv*` / chapter-7 figure generator; the existing
CNN figures (`conv-1x1.svg`, `conv-multi-in.svg`, `pooling.svg`, `lenet.svg`, etc.)
are pre-existing hand-authored assets, not house-style generated. Recommendations:

1. **[for A.3 #2, high value] `img/mdl-cnn-feature-hierarchy.svg`** — a schematic of
   the edges→textures→object-parts feature hierarchy across CNN depth (three or four
   stacked panels, increasing abstraction), to anchor the feature-visualization
   forward-point in conv-layer. Pure schematic; ideal house-style figure. Owner:
   conv-layer/teammate, but flagged here as the section's top figure add.

2. **[for P2, medium value] `img/mdl-cnn-pool-vs-strided.svg`** — a side-by-side
   schematic contrasting 2×2 max-pool (parameter-free, fixed) with a stride-2 conv
   (learned kernel) producing the same output resolution, to anchor pooling.md's
   "Pooling versus strided convolution" subsection. Makes the "learned vs fixed
   downsampling" point visual.

3. **[for A.3 #3, optional] `img/mdl-cnn-stacked-vs-large-kernel.svg`** — two stacked
   3×3 kernels vs one 5×5, same receptive field, annotated with parameter counts
   (18 vs 25 per channel-pair) — the VGG argument as a figure.

If the project prefers to keep chapter-7 figures in their current hand-authored
style rather than introducing a `gen_mdl_cnn_figures.py`, note that the **one-style-
per-chapter rule** then applies to whichever style is chosen — do not mix a single
new house-style SVG into a chapter of hand-authored ones. Decision for the author.

---

## Per-file dispatch manifest

*Priority: H(igh) / M(ed) / L(ow). "Promote" = move existing slide content into the
prose body. All line numbers are against the source `.md` as read.*

### channels.md
| ID | Pri | Where | Action |
|----|-----|-------|--------|
| C1 | M | after 1×1 section, ~line 218 | **Promote** the "Why 1×1 convs everywhere" slide into a closing prose paragraph: 1×1 convs as the channel-mixing workhorse of modern nets — ResNet bottlenecks (squeeze→3×3→expand), MobileNet pointwise (the 1×1 half of depthwise-separable), SE/attention channel mixing. Cite `Lin.Chen.Yan.2013`, `Szegedy.Liu.Jia.ea.2015`. |
| C2 | L | line 215 | Fix bias count: "requires $c_o \times c_i$ weights **plus $c_o$ biases**". |
| C3 | M | Discussion, ~line 275 + 1×1 section | Name **grouped convolution** (block-diagonal channel mixing, the ResNeXt case the text already alludes to) and **depthwise convolution** (one group per channel); note that depthwise 3×3 + pointwise 1×1 = **depthwise-separable** conv (MobileNet/EfficientNet). Forward-point to §8. |
| C4 | L | line 218 | Optional: cross-reference Exercise 1 (stacked-linear-convs fold without nonlinearity) from the "cannot be folded" sentence. |

### pooling.md
| ID | Pri | Where | Action |
|----|-----|-------|--------|
| P1 | H | lines 93, 379 | **Soften** the flat "max-pooling is preferable to average pooling" to historical framing: "max-pooling was the default for *intermediate* downsampling in classic CNNs; average pooling dominates the modern *final* layer (global average pooling)." |
| P2 | H | new `##` before Summary | **Add "Pooling versus strided convolution"** (promote the "Where pooling sits" slide): (i) All-Conv finding — strided conv replaces max-pool with no accuracy loss (cite Springenberg et al. 2015); (ii) trade-off — pooling parameter-free + fixed invariance prior vs strided conv learned + preserves capacity; (iii) **global average pooling** as modern head (cite `Lin.Chen.Yan.2013`); (iv) ConvNeXt as the no-max-pool datapoint; (v) CS231n stance: future nets may have very few pooling layers. |
| P3 | M | `d2l.bib` | **Add** `Springenberg.Dosovitskiy.Brox.ea.2015` (arXiv:1412.6806); optionally ConvNeXt `Liu.Mao.Wu.ea.2022` (arXiv:2201.03545). Required for P2. |
| P4 | M | lines 9–39 | Restructure intro to name pooling's **two distinct jobs** (spatial reduction / receptive-field growth & compute; vs local translation invariance), mirroring the slide; cross-reference why-conv's equivariance→invariance framing (`:numref:`sec_why-conv``). |
| P5 | L | after line 129 | One-sentence honesty caveat: invariance is only to sub-window shifts at one scale; cumulative pooling's loss of spatial precision is a known critique (motivates strided conv / attention). |

### lenet.md
| ID | Pri | Where | Action |
|----|-----|-------|--------|
| L1 | H | Summary, ~line 377 | **Build the §8 bridge** (promote the "30 years of progress" table): name the **conv-encoder + head template** as the §8 invariant; map each LeNet component to its §8 successor with the section it's treated in — sigmoid→ReLU (AlexNet §8.1), avg-pool→max-pool/strided-conv, none→BatchNorm (§8.5), Xavier→He, dense head→global-average-pool (NiN §8.3), shallow→deep (VGG §8.2 / ResNet §8.6). Cross-ref Exercise 1. |
| L2 | H | intro ~line 41 or Summary | Add: **AlexNet is essentially LeNet scaled up** — deeper, wider, ReLU, GPUs, ImageNet — same skeleton, bigger everything. The prompt's "template AlexNet later scaled up" framing. |
| L3 | M | after line 317 or Summary | **Promote** the flatten-bottleneck slide: quantify `16·5·5=400 → 120 = 48,000` weights as the bulk of LeNet's parameters and forward-point to **global average pooling** as the modern fix (ties to pooling P2). |
| L4 | M | line 21 / `layer_summary` | State LeNet-5's total parameter count (~60k, dominated by the dense head) to make "parsimonious vs MLP" concrete. Prefer a prose figure or a *computing* extension of `layer_summary` (param counts) — keep code teaching, not drawing. |
| L5 | L | after line 371 | Read the training result: name the achieved Fashion-MNIST val accuracy and contrast numerically with the §5 MLP, making the inductive-bias win quantitative not asserted. |

### Cross-file / section-level (ARC)
| ID | Pri | Where | Action |
|----|-----|-------|--------|
| ARC1 | M | pooling.md 13–19 (+ padding-and-strides) | Cross-reference conv-layer's receptive-field section (`:numref:`field_visual``) instead of re-motivating receptive-field growth from scratch; tighten the recurring receptive-field thread. |
| ARC2 | H | conv-layer `field_visual` section (teammate) + lenet Ex. 4–5 | Extend the biological-filter treatment with a **modern feature-visualization** forward-point (edges→textures→parts; cite Zeiler-Fergus deconv 2014 — note the bib's `Zeiler.Fergus.2013` is the *stochastic-pooling* paper, the viz paper is separate; + distill.pub Olah et al. 2017). New SVG `mdl-cnn-feature-hierarchy.svg`. Forward-point from lenet Ex. 4–5. |
| ARC3 | M | end of channels Discussion or conv-layer RF section | Add the **small-stacked-kernels** synthesis: two 3×3s = one 5×5 receptive field, fewer params + extra nonlinearity (the VGG argument); forward-point to §8 VGG. Optional figure `mdl-cnn-stacked-vs-large-kernel.svg`. |
| ARC4 | M | lenet Summary + channels Discussion | Connect the cost/efficiency through-line: LeNet param count (L4) ↔ channels' 53-B-MAC cost ↔ §8 efficiency innovations (1×1 bottleneck, grouped/depthwise, GAP). |
| ARC5 | L | pooling.md 193–309 | Optional: trim the pooling padding/stride subsection to *just the differences from conv* (defaults = window; channels-last) and `:numref:` back to padding-and-strides, saving ~40 lines. |

---

*Note on scope: ARC2/ARC3 and the padding-and-strides slide→prose promotion (A.3
#1) touch the teammate's three files (why-conv/conv-layer/padding-and-strides);
flagged here for the section synthesizer to route. Everything in the channels/
pooling/lenet dispatch tables is in-scope for this review and ready to execute.*
