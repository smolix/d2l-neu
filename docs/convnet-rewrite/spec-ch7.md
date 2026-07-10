# Chapter 7 (chapter_convolutional-neural-networks) — change spec

Character of this pass: **surgical additions, no restructure.** No file is
renamed or retired; section order is unchanged; every existing `:label:`
stays. The chapter's job is unchanged (fundamentals before the modern
chapter), but a 2026 reader needs three operations the book never defines
(depthwise-separable, dilation, conv-as-matmul) and an honest account of
where pooling stands. Several fixes are literally promotions of content that
already exists in the slide decks into the prose.

Slides: Alex explicitly approved rewriting slide decks. Where a section's
prose changes, refresh its deck; conventions in `docs/slides.md`.

## 7.0 `index.md`

- Lines 22–23 ("CNN-based architectures are now ubiquitous in the field of
  computer vision"): rewrite to the honest 2026 framing — convnets dominated
  vision roughly 2012–2021; today they share the field with vision
  transformers (forward pointer to `:numref:`chap_attention-and-transformers``)
  and remain the default where latency, small datasets, or dense prediction
  dominate. One or two sentences, not an essay; ch8 owns the full story.
- Update the closing roadmap paragraph to mention that the chapter now also
  covers dilation and depthwise-separable convolutions.

## 7.1 `why-conv.md`

- Line ~149 ("$10^{12}$ parameters, far beyond what computers currently can
  handle"): frontier models passed $10^{12}$ parameters; the argument is not
  infeasibility but *waste* — rephrase around "a million times more
  parameters than there are images to train on / than the task requires".
- Formalize equivariance vs. invariance in two lines of operator notation
  where the Waldo discussion makes the informal distinction: equivariance
  $f(T_v x) = T_v f(x)$, invariance $f(T_v x) = f(x)$; note conv layers are
  equivariant and the *head* (pooling/aggregation) buys invariance. Keep it
  to a short paragraph; this sharpens exercise 3.
- Citation check at line ~85: `Long.Shelhamer.Darrell.2015` is a segmentation
  paper cited as evidence about "object detection and segmentation"; keep it
  for segmentation and add a detection-appropriate citation (e.g. the R-CNN
  or YOLO entry already in `d2l.bib`) or soften the sentence.
- **Add the missing slide deck.** This is the only ch7 file with no
  `<!-- slides -->` block (an incomplete earlier pass). ~10–12 slides:
  flattening loses structure → the two principles (translation invariance,
  locality) → constraining the MLP step by step → convolution definition →
  equivariance vs. invariance → channels → recap.

## 7.2 `conv-layer.md`

- **Promote receptive-field arithmetic into prose.** The concept is taught
  but the growth formula lives only in the slide deck (line ~715). Add to the
  "Feature Map and Receptive Field" section: for a stack of $L$ layers with
  kernel $k_i$ and stride $s_i$, the receptive field is
  $r = 1 + \sum_{i=1}^{L} (k_i - 1) \prod_{j<i} s_j$; specialize to $L$
  stacked 3×3 stride-1 layers → $(2L{+}1) \times (2L{+}1)$. Two paragraphs +
  the formula; this is load-bearing for ch8 (VGG stacking argument, ConvNeXt
  7×7 kernels, RepLKNet callout).
- **Add a short "Convolution as matrix multiplication" example.** im2col is
  currently *asked twice as an exercise* (here ex. 4 and `channels.md` ex. 5)
  and never taught, and the Summary's "convolutions are hardware-friendly"
  claim is asserted without evidence. Add ~15 lines: build the unfolded
  patch matrix explicitly with basic indexing/stacking (framework-agnostic:
  same 6-line loop in all four tabs, or `%%tab pytorch` demo of
  `F.unfold` *after* the generic version), matmul against the flattened
  kernel, `allclose` against `corr2d`. Keep the exercises — they now ask the
  reader to extend a demonstrated idea (strides, multi-channel) instead of
  inventing it.
- Trim the biological-plausibility closer (lines ~544–555) by roughly half;
  keep Hubel & Wiesel and the Field (1987) figure, cut the meandering
  qualifiers. Fix "heralded the *recent* success in deep learning".

## 7.3 `padding-and-strides.md`

- **Add a "Dilation" section** after Stride: the third knob. Content: dilated
  kernel definition; effective kernel size $k + (k-1)(d-1)$; the point of it
  (receptive field grows exponentially with depth at constant compute, used
  in dense prediction — forward pointer to the FCN/DeepLab material in
  `:numref:`chap_cv``, which currently uses the term undefined); one new
  mechanics figure (`img/arch-conv-dilation.svg`, grid family, matching
  `conv-pad.svg`'s look — see figure-style.md §1); code in all four tabs
  (`dilation=` / `dilation_rate=` / `kernel_dilation=` / `dilate=`) showing
  output shape and an effective-receptive-field check. Update the output
  shape formula once to the general padded/strided/dilated form. Cite
  `yu2016dilated` (add to bib: Yu & Koltun, "Multi-Scale Context Aggregation
  by Dilated Convolutions", arXiv:1511.07122).
- Exercise 6 ("stride of ½") gains an explicit
  `:numref:`sec_transposed_conv`` pointer (verify the label name in
  `chapter_computer-vision/transposed-conv.md` before writing it).
- Expand the padding-artifacts citation-drop (line ~353,
  `Alsallakh...2020`) into 3–4 sentences: zero padding is a boundary
  condition; it leaks absolute position; one-sentence nod to
  reflect/replicate padding. No new figure needed.
- Add a dilation exercise (compute effective receptive field of a dilation
  schedule 1,2,4,8; when does gridding become a problem?).

## 7.4 `channels.md`

- **Add "Grouped, Depthwise, and Depthwise-Separable Convolutions"** as a
  new `##` section after the 1×1 section. This is the single biggest gap in
  the whole book (verified: depthwise-separable is defined nowhere). Content:
  - Grouped conv: split $c_i$ channels into $g$ groups, convolve each
    separately; cost drops by $g$; brief note that ResNeXt (ch8) builds on
    this. Reuse/absorb the framing of existing exercise 8 (block-diagonal
    kernels), and rewrite that exercise to build on the now-taught concept.
  - Depthwise conv as the $g = c_i = c_o$ extreme; depthwise-separable =
    depthwise $k\times k$ + pointwise 1×1. Cost ratio vs. dense conv:
    $\tfrac{1}{c_o} + \tfrac{1}{k^2}$ (≈ 8–9× cheaper for $k{=}3$, large
    $c_o$). Cite `howard2017mobilenet`, `chollet2017xception` (add to bib).
  - Code (all four tabs): build a dense 3×3 conv and its depthwise-separable
    factorization (`groups=`/`feature_group_count=`/`num_group=`/keras
    `DepthwiseConv2D`), compare parameter counts programmatically, check
    output shapes match.
  - One mechanics figure `img/arch-conv-depthwise.svg` (channel-diagram,
    grid family): dense conv mixing all channels vs. depthwise per-channel +
    pointwise mixing.
  - Forward pointers: ResNeXt (`:numref:`subsec_resnext``), ConvNeXt and
    MobileNet (new ch8 sections).
- Discussion: extend the existing cost paragraph with one sentence noting
  the depthwise-separable ratio just derived.
- New exercise: FLOPs and parameters of a depthwise-separable VGG block vs.
  the standard one.

## 7.5 `pooling.md`

- **Rewrite the Summary/Discussion to say what the slide deck already
  says**: modern convnets do most downsampling with strided convolutions
  (learned "pooling"); pooling survives chiefly as (a) global average
  pooling at the head — the NiN idea, now the default classifier head — and
  (b) max pooling in some stems and detection necks. The prose currently
  leaves the impression pooling is the default downsampler; that is
  backwards in 2026. 2–3 paragraphs; mention aliasing in one sentence and
  add an exercise on it (why can naive stride-2 downsampling alias, what
  would a blur-pool fix look like — cite `zhang2019making` if added to bib,
  else leave uncited).
- Keep everything else; the mechanics content is fine.

## 7.6 `lenet.md`

- **Promote the "what 30 years changed" comparison into prose.** The slide
  deck's table (sigmoid→ReLU/GELU, avg-pool→max/strided conv, no
  normalization→BN/LN, dense head→GAP+linear, Xavier→He init) becomes a
  small markdown table in the Summary with one sentence per row and the
  explicit line that each substitution is a section of ch8. This is the ramp
  into the modern chapter — end the chapter pointing forward.
- No architecture change, no code change otherwise. LeNet stays the
  capstone.

## Verification (after all ch7 edits)

Per-file: `tools/lint_source.py` clean; regenerate + execute the six changed
notebooks in all four frameworks (`make -B _notebooks/<fw>/chapter_convolutional-neural-networks/<f>.executed`),
then `make capture-outputs FILES=...` **with `--frameworks` covering all
four** (comma-separated!). Chapter-level: `audit_outputs.py --verify-fresh`
clean; `make html` renders; slide decks build for changed files
(`make -B slides-pytorch SLIDES_FILTER=chapter_convolutional-neural-networks/<f>.md`
spot-check). Cross-refs: `grep -rn ":numref:\`sec_" chapter_convolutional-neural-networks/`
targets all resolve.
