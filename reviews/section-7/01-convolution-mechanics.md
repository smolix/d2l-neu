# Review — Convolution Mechanics (§7.1 why-conv.md + §7.2 conv-layer.md + §7.3 padding-and-strides.md)

*Implementation spec for the AUTHORING agent. Files (source `.md`, the truth — never edit `.qmd`):
`chapter_convolutional-neural-networks/why-conv.md` ("From Fully Connected Layers to
Convolutions"), `conv-layer.md` ("Convolutions for Images"), `padding-and-strides.md`
("Padding and Stride"). Primary lens PyTorch 2.x; book ships 4 co-equal tabs
(pytorch / jax / tensorflow / mxnet — MXNet deliberately retained). Every recommendation
below names the file + section, says add/cut/rewrite, gives the actual technical content
(derivation / definition / figure spec / code cell + framework coverage), and the external
source that grounds it. No `---` em-dashes per house style. Researched June 2026.*

---

## Executive verdict

This is the **conceptual heart of the CNN chapter**, and the three files form a clean arc:
*why* convolution (derive it), *what* it computes (cross-correlation + a demo that actually
computes something + learning a kernel), and *how* to control its geometry (output-size
arithmetic, padding, stride). The bones are excellent and largely correct:

- **why-conv.md** genuinely *derives* the convolutional layer from an MLP by imposing two
  principles, and even carries a running parameter count ($10^{12} \to 4\times10^6 \to
  4\Delta^2$) that makes the inductive-bias payoff visceral. This is the strongest part of
  the three files and is at the bar.
- **conv-layer.md** is honest about cross-correlation vs convolution, the edge-detection demo
  *computes* a finite difference (good — code teaches), and "Learning a Kernel" is a
  first-rate payoff (gradient descent recovers `[1, -1]`). The receptive-field subsection and
  the Hubel and Wiesel / Field biology tie-in are good.
- **padding-and-strides.md** has the right three-act structure (boundary loss → padding →
  stride) and the right special cases.

**But measured against the "assignable at Stanford/MIT/CMU, above most courses" bar, three
structural problems keep these pages below best-in-class, and they are concentrated in the
two places the chapter most needs to be airtight:**

1. **The two principles are named inconsistently and one of them is named *wrong*.**
   why-conv.md's desiderata (lines 98–106) correctly distinguish **translation equivariance**
   (early conv layers) from **translation invariance** (pooling + output). Then the derivation
   section is titled **"Translation Invariance"** (line 151) and the prose repeatedly calls the
   *equivariance* property "invariance" (lines 154, 198, 317). The property the math derives
   ("a shift in the input leads to a shift in the hidden representation", line 155) is exactly
   **equivariance**, not invariance. This is *the* central concept of the chapter and the file
   contradicts itself on it. This is the single highest-value fix. (Source: this is the
   standard modern distinction; CS231n and every current treatment use *equivariance* for the
   conv map and reserve *invariance* for the pooled/output prediction —
   https://cs231n.github.io/convolutional-networks/.)

2. **The output-size formula is stated, never derived — in the one file whose entire job is
   that formula.** padding-and-strides.md asserts the padded shape (line 95) and the strided
   floor formula $\lfloor (n - k + p + s)/s \rfloor$ (line 279) by fiat. A top-program
   treatment *derives* "how many kernel placements fit," which is exactly the content of the
   canonical reference for this material — Dumoulin and Visin, *A guide to convolution
   arithmetic for deep learning* (2016), https://arxiv.org/abs/1603.07285 (already in
   `d2l.bib` as `Dumoulin.Visin.2016`, **currently uncited in all three files**). This is also
   the one place the chapter should *anchor on the Dumoulin–Visin animations* and currently
   uses only static legacy SVGs.

3. **The forward-pointing ceiling is missing entirely.** None of dilated/atrous convolution,
   the *effective* receptive field, depthwise-separable convolution, or a real treatment of
   the 1×1 convolution appears anywhere in these three files (verified by grep: zero hits for
   `dilat|atrous|separable|depthwise|effective receptive|exponential`). The receptive-field
   subsection teaches the *theoretical* RF as if it were the whole story; the 2016 result that
   the *effective* RF is Gaussian and grows like $\sqrt{L}$ (Luo et al.) is the natural,
   in-lane upgrade and is genuinely surprising to students.

**A figure-conventions problem cuts across all three files.** Every illustrative SVG these
files use (`correlation.svg`, `conv-pad.svg`, `conv-stride.svg`, `conv-reuse.svg`,
`waldo-mask.jpg`) is a **legacy upstream d2l asset**, not the house style: there is **no
`tools/gen_mdl_cnn_figures.py`** and **zero `img/mdl-cnn-*.svg`** files (grep-confirmed),
whereas the sibling core chapters already ship `mdl-mlp-*` and `mdl-clf-*` generators. The
CLAUDE.md "one figure style per chapter" rule is therefore violated for the entire CNN
chapter. Bringing the *mechanics* figures into the house style (and adding the missing
conceptual figures) is a real, in-lane body of work, flagged per-figure below and consolidated
in the dispatch manifest.

**Grade: B+.** Correct derivation, honest framing, demos that compute, good biology tie-in.
Held back by the equivariance/invariance naming contradiction, an asserted-not-derived
output-size formula, a missing forward-pointing ceiling, and a wholesale figure-style gap. All
fixes are in-lane and none is large.

Calibration: I checked CS231n, Dumoulin–Visin, Luo et al. 2016 (effective RF), Yu and Koltun
2016 (dilated), and the Xception/MobileNet depthwise-separable line. The repo is *not* behind
upstream d2l (upstream has the same gaps), but "match upstream" is below this project's stated
bar; the room to *exceed* is concrete and unclaimed.

---

## 1. `why-conv.md` — "From Fully Connected Layers to Convolutions"

**What it teaches.** The $10^9$-parameter blow-up of a dense layer on a megapixel image; the
"Where's Waldo" spatial-structure motivation; the desiderata (equivariance, locality,
growing-scale); the *derivation* of the conv layer by constraining a 4th-order weight tensor
with translation symmetry (→ weight sharing) then locality (→ finite kernel), with a running
parameter count; the continuous/discrete convolution definition and the cross-correlation
footnote; the channels generalization. Structure: 6 `##` sections (Invariance / Constraining
the MLP / Convolutions / Channels / Summary / Exercises). The derivation is the best thing in
the three files.

**Strong — keep:**
- The **running parameter count** ($10^{12} \to 4\times10^6 \to 4\Delta^2$, lines 149/169/185)
  is excellent and rare; it makes "inductive bias buys sample efficiency" quantitative. Keep
  verbatim.
- The **two-step derivation** (translation symmetry collapses $\mathsf{V}_{i,j,a,b}$ to
  $\mathbf{V}_{a,b}$; locality truncates to $|a|,|b|\le\Delta$, lines 157–182) is exactly the
  right pedagogy and matches CS231n's parameter-sharing argument. Keep the structure.
- The **Waldo "what Waldo looks like does not depend on where Waldo is"** hook (lines 79–88)
  is a strong, correct motivation for *equivariant detection*. Keep.
- The **historical breadcrumbs** (TDNN `Waibel...1989`, Neocognitron `Fukushima.1982`,
  `Zhang.ea.1988`) are good and correctly placed.

**Fix — concrete change items:**

**1.1 (P1, highest value) Resolve the equivariance vs invariance contradiction.** The file
uses "invariance" for the property the math actually derives (equivariance), in the section
title (line 151), the derivation prose (lines 154, 198), and the summary (line 317). Yet the
desiderata (lines 98–104) already define the pair correctly.
- *Rename* the subsection "### Translation Invariance" (line 151) → **"### Translation
  Equivariance"**.
- *Rewrite* line 154 "translation invariance :cite:`Zhang.ea.1988`. This implies that a shift
  in the input … should simply lead to a shift in the hidden representation" → keep the
  sentence but call it **equivariance**: *a shift in the input produces an equal shift in the
  hidden representation; this property is translation equivariance.*
- *Rewrite* lines 197–200 ("our features are now translation invariant") → "our features are
  now translation **equivariant**" and add one sentence: *Invariance — a prediction that does
  not move when the input shifts — is recovered later by pooling and the global readout (see*
  `:numref:`sec_pooling``*), not by the convolution itself.*
- *Fix* the summary (line 317) likewise.
- Add a **precise two-line definition box** right after the desiderata: for an operator $T_s$
  that shifts an image by $s$, a layer $f$ is **equivariant** if $f(T_s x) = T_s f(x)$ and
  **invariant** if $f(T_s x) = f(x)$; convolution is equivariant, global pooling is invariant.
  *Source:* https://cs231n.github.io/convolutional-networks/ (parameter sharing / translation
  invariance discussion); the equivariance-vs-invariance precision is standard modern usage.

**1.2 (P1) Add an illustrative figure that *shows* the dense→shared→local reduction.** The
derivation is currently pure algebra. A single multi-panel house-style SVG would carry it. Spec
in the manifest as **`mdl-cnn-mlp-to-conv`**: three stacked panels over the same small input
grid — (a) *dense*: one output pixel wired to every input pixel, with the count $n^2 \cdot n^2$;
(b) *shared*: the same local stencil drawn at two different output positions with **identical
colors**, annotated "same weights everywhere" → count $4\times10^6$ shape; (c) *local*: the
stencil truncated to a $3\times3$ neighborhood, count $\Delta^2$ shape. This visually *is* the
two principles and replaces a wall of subscripts with a picture (CLAUDE.md: illustrative
figures pre-generated, intuition-first). *No drawing code in the notebook.*

**1.3 (P2) Make the channels teaser end with a forward-pointer to 1×1, not just NiN in an
exercise.** The channels section (lines 247–303) correctly generalizes to $\mathsf{V}_{a,b,c,d}$.
Exercise 1 (lines 327–330) already derives that $\Delta=0$ gives a per-pixel MLP across
channels and cites NiN (`Lin.Chen.Yan.2013`, in bib). Add **one sentence** at the end of the
channels section: *The special case $\Delta = 0$ — a $1 \times 1$ convolution — does no spatial
mixing at all; it is a per-pixel linear map across channels, and turns out to be one of the most
useful primitives in modern CNNs (see* `:numref:`sec_channels``*).* This forward-points without
stealing the channels file's content. *Source:* CS231n 1×1 discussion
(https://cs231n.github.io/convolutional-networks/); Lin, Chen, Yan 2013 (in bib).

**1.4 (P2) Tighten the "Convolutions" math section.** Lines 214–244 define continuous and
discrete convolution and correctly note the operation is really cross-correlation. Good, but
the "flip" intuition ("overlap … when one function is flipped and shifted", line 222) is never
*shown*. Add a one-line parenthetical that the flip is what makes mathematical convolution
**commutative** ($f*g = g*f$, which is Exercise 6, line 339) whereas cross-correlation is not —
this connects the section to its own exercise and to the conv-layer.md flip discussion.

**1.5 (cleanup) The two "Waldo" photos** (`waldo-football.jpg`, `waldo-mask.jpg`) are legacy
JPEGs. Keep the photographs (per project memory: keep photographic/reference images; the
one-style rule targets schematic figures). No change needed beyond noting they are exempt.

---

## 2. `conv-layer.md` — "Convolutions for Images"

**What it teaches.** The 2D cross-correlation operation (worked numerically: $0{\cdot}0 +
1{\cdot}1 + 3{\cdot}2 + 4{\cdot}3 = 19$); a from-scratch `corr2d` (4 tabs); a `Conv2D` layer
class (4 tabs); edge detection with the `[1, -1]` finite-difference kernel (computes a real
edge map, then vanishes on the transpose); **learning** that kernel from (X, Y) by SGD (4
tabs); the cross-correlation-vs-convolution flip argument; feature map + receptive field; the
Hubel and Wiesel / Field biology tie-in. Structure: 6 `##` sections + slides. Strong file.

**Strong — keep:**
- The **edge-detection demo genuinely computes** (lines 252–307): it builds a vertical-edge
  image, applies `[1, -1]`, gets $\pm 1$ at transitions, then shows the *same* kernel detects
  nothing on the transpose. This is "code teaches, never draws" done right. Keep.
- The **finite-difference framing** (lines 285) — "this kernel is a special case of a finite
  difference operator … approximates $-\partial_j f$" — is a lovely, correct mathematical
  upgrade over upstream's bare "edge detector." Keep verbatim.
- **"Learning a Kernel"** (lines 309–458): random init → 10 SGD steps → recovers $[1,-1]$.
  This is the payoff of the whole file and is excellent across all four tabs. Keep. (Note the
  JAX tab's small-stddev-init comment, lines 408–410, is a correct, honest fix — keep it.)
- The **flip argument** (lines 460–495): "since kernels are learned, cross-correlation vs
  convolution does not matter for the output." Correct and the right amount of honesty. Keep.
- The **receptive-field stacking argument** (lines 520–541) and the **biology tie-in**
  (Hubel and Wiesel, Field 1987 figure) are good. Keep the biology.

**Fix — concrete change items:**

**2.1 (P1) Replace the asserted output size with a one-line derivation, and bring the figure
into the house style.** Lines 91–100 state $(n_h - k_h + 1)\times(n_w - k_w + 1)$ with the
hand-wave "we need enough space to shift the kernel." Add the *count*: along a row of width
$n$, a kernel of width $k$ can start at columns $0, 1, \ldots, n-k$, i.e. **$n - k + 1$
placements** — that *is* the output width. This is the seed of the general formula derived in
§3 and costs one sentence. *Source:* Dumoulin–Visin 2016 Relationship 1 (no padding, unit
stride: $o = (i - k) + 1$), https://arxiv.org/abs/1603.07285.

**2.2 (P1) Exploit the cross-correlation framing instead of just noting it.** The file flags
that the op is cross-correlation, shows the flip turns it into convolution, then drops it. A
top treatment *uses* it: add a short paragraph at the end of "Cross-Correlation and
Convolution" (after line 495) noting the practical consequence — frameworks all implement
**cross-correlation** under the name "conv" (PyTorch `nn.Conv2d`, TF/Keras, Flax, MXNet all do
correlation), so a kernel you inspect or initialize by hand is **not** flipped relative to what
you wrote. This is the thing that actually bites students (a hand-set Sobel/Gaussian behaves as
written, not flipped). One paragraph, no code. *Source:* CS231n
(https://cs231n.github.io/convolutional-networks/ uses "convolution" for the correlation op and
says so); Dumoulin–Visin §1.

**2.3 (P1) Upgrade the receptive-field subsection: theoretical RF formula + the *effective* RF
result.** Currently (lines 501–541) the RF is taught only by the 2×2-then-2×2 example ("$z$
sees all 9 inputs"). Add:
- The **closed-form theoretical RF** for a stack of $L$ stride-1 layers with kernel $k$:
  $r_L = L(k-1) + 1$, so $L$ layers of $3\times3$ give $r = 2L+1$. State the general
  stride-aware recurrence $r_{l-1} = s_l\, r_l + (k_l - s_l)$ for completeness. *Source:*
  https://theaisummer.com/receptive-field/ (closed form $r = \sum (k_i-1)\prod s_j + 1$; the
  $2L+1$ special case).
- A **forward-pointing paragraph on the *effective* receptive field**: Luo et al. 2016 showed
  the RF that actually carries gradient is **Gaussian-shaped**, occupies only a *fraction* of
  the theoretical box, and grows like **$\sqrt{L}$** (not linearly). Practical upshot: stacking
  more layers buys less reach than the formula suggests, motivating dilation (next item) and
  larger kernels. This is surprising and in-lane. *Source:* Luo, Li, Urtasun, Zemel,
  *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*, NeurIPS
  2016, https://arxiv.org/abs/1701.04128. **Add bib key `Luo.Li.Urtasun.ea.2016`.**
- Spec a house-style figure **`mdl-cnn-receptive-field`** (manifest): left panel, two stacked
  $3\times3$ layers with the $5\times5$ theoretical RF outlined on the input; right panel, a
  Gaussian heat blob inside that same $5\times5$ box labeled "effective RF $\sim\sqrt{L}$".
  This replaces prose-only RF with the picture that makes the $\sqrt{L}$ point land.

**2.4 (P2) Add a dilated-convolution forward-pointer.** Right after the RF upgrade, one short
paragraph: a **dilated (atrous)** convolution inserts gaps of size $d-1$ between kernel taps,
giving an *effective* kernel size $k + (k-1)(d-1)$ at no extra parameters, so the RF can grow
**exponentially** with depth when $d$ doubles each layer. This is the standard fix for the
$\sqrt{L}$ problem and the basis of segmentation nets. *Source:* Yu and Koltun,
*Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016,
https://arxiv.org/abs/1511.07122 ("dilated convolutions support exponential expansion of the
receptive field without loss of resolution"). **Add bib key `Yu.Koltun.2016`.** Optionally
spec figure **`mdl-cnn-dilation`** (a $3\times3$ kernel at $d=1$ vs $d=2$ over the same grid,
showing the spread taps and the larger effective window), or reuse a Dumoulin–Visin-style
single panel.

**2.5 (P3, optional code) The for-loop `corr2d` is the right teaching primitive — keep it —
but add one sentence** noting real implementations vectorize via *im2col* (unfold the input
into patch columns, then a single matrix multiply), which is *exactly* Exercise 4 (line 577,
"represent cross-correlation as a matrix multiplication"). Connect the exercise to the prose;
no new code cell required. *Source:* CS231n "Implementation as Matrix Multiplication"
(https://cs231n.github.io/convolutional-networks/).

**2.6 (cleanup) Figure style.** `correlation.svg` (line 65) is legacy. Re-draw as
**`mdl-cnn-correlation`** in the house style (same numbers, same shaded-window semantics) so it
matches the rest of the chapter. The `field-visual.png` biology figure is photographic/reference
— **keep as-is** (project-memory exemption).

---

## 3. `padding-and-strides.md` — "Padding and Stride"

**What it teaches.** Why convs shrink ($240\to200$ over ten $5\times5$ layers); padding to
preserve size and to use boundary pixels; the padded output shape; the odd-kernel /
center-alignment convention; per-axis padding code (4 tabs); stride to downsample; the strided
floor formula and its special cases; per-axis stride code (4 tabs); a discussion of non-zero
padding. Structure: 4 `##` sections + slides. Right structure; the arithmetic is asserted, not
derived.

**Strong — keep:**
- The **"$240\to200$, 30% of the area, all from the boundary"** motivation (lines 27–32) is
  vivid and correct. Keep.
- The **odd-kernel / centered-window rationale** (lines 100–128) is the right convention and
  well explained. Keep.
- The **special cases** of the stride formula (lines 281–286: $p=k-1$ ⇒
  $\lfloor (n+s-1)/s \rfloor$; divisible ⇒ $n/s$) are exactly the useful ones. Keep.
- The **`comp_conv2d` helper** (4 tabs) cleanly isolates the shape question from batch/channel
  bookkeeping. Keep. (Note the TF tab's `ZeroPadding2D` workaround for asymmetric padding,
  lines 331–338, is a correct, honest fix — keep.)
- The **non-zero-padding discussion** (line 353, `Alsallakh...2020`, "CNNs can encode implicit
  position from where the whitespace is") is a genuinely sophisticated point. Keep.

**Fix — concrete change items:**

**3.1 (P1, highest value for this file) *Derive* the output-size formula; do not assert it.**
This is the file whose entire purpose is this formula, and it currently states the padded shape
(line 95) and the strided floor (line 279) with no derivation. Add a short derivation, built up
exactly as Dumoulin–Visin do it:
- *No padding, unit stride:* a width-$k$ kernel starts at positions $0,\ldots,n-k$ →
  **$n-k+1$** outputs (Relationship 1).
- *Add $p$ total padding:* the input is effectively $n+p$ wide → **$n-k+p+1$** (Relationship
  with zero padding; their $o = i - k + 2p + 1$ with $p$ = per-side, here $p$ = total).
- *Add stride $s$:* of the $n-k+p+1$ valid start columns we keep every $s$-th one, starting at
  0, giving **$\lfloor (n - k + p)/s \rfloor + 1$** outputs. Show this equals the file's
  $\lfloor (n - k + p + s)/s \rfloor$ (they differ only in algebra:
  $\lfloor (a)/s \rfloor + 1 = \lfloor (a+s)/s \rfloor$). **Recommend switching the file's
  display to the cleaner standard form** $\big\lfloor \tfrac{n-k+p}{s}\big\rfloor + 1$ (which
  is what CS231n and Dumoulin–Visin print, and what the slide recap on line 526 *already*
  uses), and noting the integer/divisibility caveat (a non-integer means the last kernel
  placement hangs off the input and is dropped). *Sources:* Dumoulin–Visin 2016 Relationships
  1–6 (https://arxiv.org/abs/1603.07285); CS231n $(W - F + 2P)/S + 1$, "must be an integer"
  (https://cs231n.github.io/convolutional-networks/). `Dumoulin.Visin.2016` is **already in
  bib** — cite it here (it is the canonical reference for this exact section and is currently
  uncited).

**3.2 (P1) Anchor the teaching on the Dumoulin–Visin *animations* via house-style multi-panel
SVGs.** The three legacy SVGs (`conv-reuse.svg`, `conv-pad.svg`, `conv-stride.svg`) are static
and off-style. The Dumoulin–Visin conv_arithmetic figures (blue input grid, green output grid,
shaded sliding kernel) are *the* canonical way this is taught
(https://github.com/vdumoulin/conv_arithmetic). Since the book is static (no GIFs), capture the
key frames as **multi-panel house-style SVGs** (CLAUDE.md explicitly allows "animation concept
captured as a multi-panel SVG"):
- **`mdl-cnn-pad`** (replaces `conv-pad.svg`): the $3\times3$-padded-to-$5\times5$ input with
  the $2\times2$ kernel at the top-left placement, output $4\times4$, the worked $0$ value.
- **`mdl-cnn-stride`** (replaces `conv-stride.svg`): the stride-(3,2) example with two output
  positions and the two worked values ($8$ and $6$), arrows showing the 3-row / 2-col jumps.
- **`mdl-cnn-reuse`** (replaces `conv-reuse.svg`): the pixel-utilization heatmap for $1\times1$,
  $2\times2$, $3\times3$ kernels (corners cold, center hot).
All in the shared `gen_mdl_figures.py` style so the chapter is one consistent look. *Source:*
https://github.com/vdumoulin/conv_arithmetic; Dumoulin–Visin 2016.

**3.3 (P2) Add the modern "three patterns" table to the prose.** The **slides already have it**
(lines 504–516: Preserve $k3/p1/s1$, Halve $k3/p1/s2$, Patchify $k/0/k$, with ResNet / standard
downsample / ViT annotations) but the *prose body* never states these named recipes. Promote
the table into the Summary (after line 351). This is the highest-leverage modernization in the
file: it connects the arithmetic to what students actually build (ResNet stems, ViT patchify)
and is a one-table lift from the slides. *Source:* the patchify-as-strided-conv view is
standard post-ViT practice (Dosovitskiy et al. 2020; the slide already asserts it).

**3.4 (P2) Forward-point strided conv vs pooling vs transposed conv.** One sentence in the
Summary: strided convolution is one of three ways to change resolution — it *learns* its
downsampling (contrast fixed **pooling**, `:numref:`sec_pooling``), and its inverse,
**transposed convolution** (fractional stride, used for upsampling in segmentation/generation),
is covered later (`:numref:`sec_transposed_conv`` if present). The file's Exercise 6 (line 364,
"implement a stride of 1/2") *is* transposed convolution — connect the exercise to the name.
*Source:* Dumoulin–Visin 2016 §4 (transposed convolution as the gradient of convolution);
trans_conv SVGs already exist in `img/`.

**3.5 (P2) Mention dilation here too (the other geometry knob).** padding-and-strides.md is the
"convolution geometry" file, and dilation is the third geometric parameter alongside padding and
stride (PyTorch `nn.Conv2d` exposes `padding`, `stride`, `dilation` together). After deriving
the strided formula, add the **dilated** generalization in one line: effective kernel
$k' = k + (k-1)(d-1)$, so the output size becomes
$\big\lfloor (n + p - k - (k-1)(d-1))/s \big\rfloor + 1$. This makes the formula complete and
sets up the conv-layer.md dilation forward-pointer (2.4). *Source:* Dumoulin–Visin 2016
Relationship 15 (dilation); Yu and Koltun 2016.

---

## 4. Cross-file: forward-pointing ceiling (depthwise-separable) and the figure-style debt

**4.1 (P2, where?) Depthwise-separable convolution as the efficiency forward-pointer.** This is
the natural "modern efficient conv" capstone and fits best as a **one-paragraph pointer in
conv-layer.md's Summary** (or coordinate with the channels file if the teammate prefers it
there — flag in dispatch). Content: a standard conv mixes space *and* channels at once, costing
$\propto k^2 \cdot C_\text{in} \cdot C_\text{out}$; a **depthwise-separable** conv factors this
into a per-channel spatial conv (depthwise) followed by a $1\times1$ channel mix (pointwise),
cutting cost by roughly $\tfrac{1}{C_\text{out}} + \tfrac{1}{k^2}$ (≈ 8–9× for $3\times3$). This
is what makes MobileNet/Xception efficient. *Sources:* Howard et al.,
*MobileNets* (2017), https://arxiv.org/abs/1704.04861; Chollet, *Xception* (2017),
https://arxiv.org/abs/1610.02357. **Add bib keys `Howard.Zhu.Chen.ea.2017` and
`Chollet.2017`.** This is explicitly a *forward-pointer* (no implementation in the basics
chapter), so keep it to a paragraph.

**4.2 (P1, chapter-wide) Stand up `tools/gen_mdl_cnn_figures.py` and migrate the mechanics
figures to the house style.** This is the cross-cutting figure-conventions fix. There is no CNN
figure generator and no `mdl-cnn-*` SVG today, while sibling core chapters
(`gen_mdl_mlp_figures.py`, the `mdl-clf-*` set) already exist. Per the **mdl-figure** skill and
CLAUDE.md "one style per chapter", create `tools/gen_mdl_cnn_figures.py` that
`import gen_mdl_figures as fl` (shared palette + `save()` + helpers) and emits, at minimum, the
figures specified above: `mdl-cnn-mlp-to-conv`, `mdl-cnn-correlation`, `mdl-cnn-pad`,
`mdl-cnn-stride`, `mdl-cnn-reuse`, `mdl-cnn-receptive-field`, and optionally `mdl-cnn-dilation`.
Run `make figures`; commit byte-idempotent SVGs; then run the **figure-style-audit** skill over
the chapter. Keep the two Waldo JPEGs and `field-visual.png` (photographic/reference,
exempt). *Naming convention confirmed:* core chapters use `img/mdl-<chapter>-<id>.svg` with a
short prefix (e.g. `mdl-mlp-`, `mdl-clf-`), so the CNN prefix is **`mdl-cnn-`**.

---

## 5. What the best current treatments cover here (researched, cited)

Each entry: URL + why it is a good model for *this* material. All fetched/searched June 2026.

**Canonical references for this exact content**
- **Dumoulin and Visin, *A guide to convolution arithmetic for deep learning* (2016)** —
  https://arxiv.org/abs/1603.07285. The reference for output-size arithmetic: numbered
  Relationships for no-padding/unit-stride ($o=i-k+1$), zero-padded ($o=i-k+2p+1$), strided
  ($o=\lfloor(i-k+2p)/s\rfloor+1$), half/full padding, and dilation (effective kernel
  $k+(k-1)(d-1)$). **Already in `d2l.bib` as `Dumoulin.Visin.2016`; currently uncited in these
  files** — it should anchor §3. Companion animations:
  https://github.com/vdumoulin/conv_arithmetic (blue input, green output, shaded kernel).
- **Stanford CS231n, *Convolutional Networks*** —
  https://cs231n.github.io/convolutional-networks/. The model for: the
  local-connectivity/parameter-sharing derivation; the output formula $(W-F+2P)/S+1$ with the
  "must be an integer" caveat and $P=(F-1)/2$ "same" convention; receptive field; im2col;
  dilation; and the precise translation-invariance-via-sharing argument. The single best
  calibration for depth and framing of all three files.

**Receptive field**
- **Luo, Li, Urtasun, Zemel, *Understanding the Effective Receptive Field…*, NeurIPS 2016** —
  https://arxiv.org/abs/1701.04128. The ERF is Gaussian, occupies a fraction of the theoretical
  box, and grows like $\sqrt{L}$. Basis for 2.3. **Add to bib.**
- **AI Summer, *Understanding the receptive field…*** — https://theaisummer.com/receptive-field/.
  Clean closed-form theoretical RF ($r=\sum(k_i-1)\prod s_j + 1$; $L$ layers of 3×3 → $2L+1$)
  and a readable summary of the Luo et al. result. Good model for how to present 2.3 compactly.

**Dilated / atrous**
- **Yu and Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016** —
  https://arxiv.org/abs/1511.07122. Exponential RF growth at linear parameter cost; the basis
  for 2.4 / 3.5. **Add to bib.**

**Efficient convolution (forward-pointer)**
- **Howard et al., *MobileNets* (2017)** — https://arxiv.org/abs/1704.04861, and
  **Chollet, *Xception* (2017)** — https://arxiv.org/abs/1610.02357. Depthwise-separable
  factorization and the $\tfrac{1}{C_\text{out}}+\tfrac{1}{k^2}$ cost saving. Basis for 4.1.
  **Add both to bib.**

**Interpretability / what filters learn (enriches the biology tie-in in conv-layer.md)**
- **Olah et al., *Feature Visualization*, Distill (2017)** —
  https://distill.pub/2017/feature-visualization/, and **Olah et al., *The Building Blocks of
  Interpretability*, Distill (2018)** — https://distill.pub/2018/building-blocks/. The modern,
  evidence-based complement to the Hubel–Wiesel/Field figure: optimization-based visualization
  shows early filters → edges/textures, deeper → parts/objects. A one-line forward-pointer in
  conv-layer.md's receptive-field/biology paragraph would modernize a 1987-anchored discussion
  with a 2017 method. *Optional P3.*

---

## 6. Keep — do not lose this in any revision

- why-conv.md's **running parameter count** ($10^{12}\to4\times10^6\to4\Delta^2$) and the
  **two-step symmetry→sharing, locality→truncation derivation**. Best content in the three files.
- conv-layer.md's **edge-detection demo** (computes a real finite difference, vanishes on the
  transpose), the **finite-difference / $-\partial_j f$ framing**, and **"Learning a Kernel"**
  (SGD recovers $[1,-1]$, all 4 tabs).
- conv-layer.md's **cross-correlation honesty** and the **flip argument**.
- The **Hubel–Wiesel / Field 1987 biology tie-in** and the `field-visual.png` figure (keep the
  photo as-is).
- padding-and-strides.md's **$240\to200$ boundary-loss motivation**, **odd-kernel rationale**,
  the **stride special cases**, and the **`comp_conv2d` helper** (incl. the honest per-tab
  fixes: TF `ZeroPadding2D`, JAX small-stddev init).
- The **slide decks** for all three files — they are *ahead* of the prose (the "three patterns"
  table, the stacked-RF $2L+1$ recap, the ViT/patchify framing). Several recommendations above
  are literally "promote the slide content into the body."

---

## 7. Per-file dispatch manifest (ordered work items)

Legend: **file → change → new figures (via mdl-figure) → code cells + framework coverage →
external refs → dependencies/notes.**

### A. why-conv.md
1. **Fix equivariance vs invariance throughout** (rename §, lines 151/154/197–200/317; add the
   $T_s$ definition box after the desiderata). *Figures:* none. *Code:* none. *Refs:* CS231n.
   *Deps:* none. **P1, do first** (it is the chapter's central concept).
2. **Add `mdl-cnn-mlp-to-conv`** (3-panel dense→shared→local figure for the derivation).
   *Figures:* `mdl-cnn-mlp-to-conv` (new generator). *Code:* none. *Refs:* CS231n.
   *Deps:* requires task **G1** (generator scaffold).
3. **1×1 forward-pointer sentence** at end of Channels (`:numref:`sec_channels``). *Figures:*
   none. *Code:* none. *Refs:* CS231n, `Lin.Chen.Yan.2013` (in bib). *Deps:* coordinate with
   teammate (channels file owns the full 1×1 treatment).
4. **Tighten "Convolutions" math** (flip ⇒ commutativity, tie to Exercise 6). *Figures:* none.
   *Code:* none. *Refs:* `Rudin.1973` (in bib). *Deps:* none. **P2.**

### B. conv-layer.md
5. **Derive $n-k+1$** (one sentence, lines 91–100). *Figures:* none. *Code:* none.
   *Refs:* Dumoulin–Visin (in bib). *Deps:* none. **P1.**
6. **Cross-correlation practical consequence** paragraph after line 495 (frameworks do
   correlation; hand-set kernels not flipped). *Figures:* none. *Code:* none. *Refs:* CS231n,
   Dumoulin–Visin. *Deps:* none. **P1.**
7. **Receptive-field upgrade**: add $r_L=L(k-1)+1$ / $2L+1$ + the effective-RF ($\sqrt{L}$,
   Gaussian) paragraph. *Figures:* `mdl-cnn-receptive-field` (2-panel). *Code:* none.
   *Refs:* Luo et al. 2016 (**add bib `Luo.Li.Urtasun.ea.2016`**), AI Summer. *Deps:* G1; bib.
   **P1.**
8. **Dilated-conv forward-pointer** paragraph after the RF upgrade. *Figures:* optional
   `mdl-cnn-dilation`. *Code:* none. *Refs:* Yu and Koltun 2016 (**add bib `Yu.Koltun.2016`**).
   *Deps:* G1 (if figure); bib. **P2.**
9. **Depthwise-separable forward-pointer** paragraph in Summary (or hand to channels teammate).
   *Figures:* none. *Code:* none. *Refs:* Howard 2017, Chollet 2017 (**add bib
   `Howard.Zhu.Chen.ea.2017`, `Chollet.2017`**). *Deps:* bib; teammate handshake. **P2.**
10. **im2col sentence** tying prose to Exercise 4. *Figures:* none. *Code:* none (keep the
    for-loop `corr2d`). *Refs:* CS231n. *Deps:* none. **P3.**
11. **Migrate `correlation.svg` → `mdl-cnn-correlation`** (house style, same numbers).
    *Figures:* `mdl-cnn-correlation`. *Code:* none. *Refs:* none. *Deps:* G1.
12. *(optional P3)* **Distill feature-visualization pointer** in the biology paragraph.
    *Refs:* Olah et al. 2017/2018 (distill.pub).

### C. padding-and-strides.md
13. **Derive the output-size formula** (no-pad → +p → +stride), switch display to
    $\lfloor(n-k+p)/s\rfloor+1$, add the integer caveat, **cite `Dumoulin.Visin.2016`**.
    *Figures:* none. *Code:* none. *Refs:* Dumoulin–Visin (in bib), CS231n. *Deps:* none.
    **P1, do first in this file.**
14. **Migrate the three legacy SVGs to house-style multi-panel figures** capturing the
    Dumoulin–Visin animation frames. *Figures:* `mdl-cnn-pad`, `mdl-cnn-stride`,
    `mdl-cnn-reuse`. *Code:* none. *Refs:* conv_arithmetic repo; Dumoulin–Visin. *Deps:* G1.
    **P1.**
15. **Promote the "three patterns" table** (Preserve / Halve / Patchify) from slides into the
    Summary. *Figures:* none. *Code:* none. *Refs:* slide already has it; ViT (Dosovitskiy
    2020). *Deps:* none. **P2.**
16. **Forward-point pooling / transposed conv**, tie Exercise 6 (stride 1/2) to transposed
    convolution. *Figures:* none (trans_conv SVGs already exist). *Code:* none.
    *Refs:* Dumoulin–Visin §4. *Deps:* none. **P2.**
17. **Add the dilated output-size generalization** (one line, $k'=k+(k-1)(d-1)$). *Figures:*
    none. *Code:* none. *Refs:* Dumoulin–Visin R15; Yu and Koltun. *Deps:* item 8. **P2.**

### G. Chapter-wide (do before any figure task above)
- **G1 — Create `tools/gen_mdl_cnn_figures.py`** importing `gen_mdl_figures` (shared style),
  emitting all `mdl-cnn-*` SVGs above; `make figures`; commit byte-idempotent SVGs; run the
  **figure-style-audit** skill. Keep Waldo JPEGs + `field-visual.png` (exempt). **Blocks items
  2, 7, 8, 11, 14.** Use the **mdl-figure** skill.
- **G2 — Add bib entries** to `d2l.bib`: `Luo.Li.Urtasun.ea.2016`, `Yu.Koltun.2016`,
  `Howard.Zhu.Chen.ea.2017`, `Chollet.2017`. (`Dumoulin.Visin.2016` and `Lin.Chen.Yan.2013`
  already present.) **Blocks items 7, 8, 9.**

**Suggested order:** G2 + G1 (scaffold) → A1, B5, B6, C13 (prose-only P1s, no figure dep) →
A2, B7, B11, C14 (P1 figures) → B8, C17, A3, B9, C15, C16, A4 (P2) → B10, B12 (P3).

**No code-cell changes are required by this review** (the notebook code is correct and should
be kept); all work is prose, citations, and house-style figures. The one place to *not* touch
is the for-loop `corr2d` — it is the teaching primitive (item 10 only adds a sentence pointing
at the vectorized form).
