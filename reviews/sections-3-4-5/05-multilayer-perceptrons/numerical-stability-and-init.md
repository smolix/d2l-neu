# Review — chapter_multilayer-perceptrons/numerical-stability-and-init.md  (§5.4 "Numerical Stability and Initialization")

**Role in the chapter:** The "why training deep nets is hard, and the first lever for fixing it" section. It establishes vanishing/exploding gradients as a product-of-Jacobians phenomenon, motivates random (symmetry-breaking) initialization, and derives Xavier/Glorot init from variance preservation. It sits between `backprop.md` (which builds the chain-rule machinery this section multiplies) and `generalization-deep.md`/`dropout.md`.

**Verdict:** The bones are good — the Jacobian-product framing, the sigmoid-saturation demo, the exploding-product demo (a memorable ~$10^{24}$ payoff), and a real (not asserted) Xavier derivation put it ahead of most introductory treatments. But it falls short of the top-program bar in three concrete ways: (1) it is a **ReLU-centric chapter that never teaches He/Kaiming init** — the variance argument it carefully builds for Xavier stops exactly where the chapter's own activation choice demands the $\mathrm{Var}=2/n_\text{in}$ correction; the slide deck already teaches He, so the *body lags its own slides*; (2) the **"Default Initialization" subsection is a near-empty stub** (three sentences and five blank lines); (3) the Xavier backward-pass step and a key variance identity are **asserted, not shown**. The single highest-value change is **NSI-1: add the He/Kaiming paragraph + ReLU variance argument** — it is squarely in scope (this is the correct home for it, per the scope map), the citation already exists in the bib, and it closes the most visible currency gap.

**Grade:** B−. Assignable with caveats today, but a Stanford/CMU student who learns only Xavier here and then opens any PyTorch ReLU net (whose default is Kaiming) meets an immediate, unexplained gap. NSI-1 + NSI-2 lift it to a solid A−.

**Top priorities (ranked):**
1. [P0] **NSI-1** — Add He/Kaiming initialization (the ReLU variance argument + formula + framework-default note). *authored.*
2. [P0] **NSI-2** — Fill or fold the empty "Default Initialization" stub into a substantive paragraph. *authored.*
3. [P1] **NSI-3** — Tighten the Xavier derivation: justify the backward-pass condition (don't assert "same reasoning") and the $E[w^2]=\sigma^2$ step. *authored.*
4. [P1] **NSI-4** — Add forward-pointers that currently exist only in the slides: symmetry→`sec_dropout`, and a one-line normalization pointer to `sec_batch_norm`. *mechanical.*
5. [P1] **NSI-5** — MXNet: tombstone the archived-framework tab (book-wide decision) and note the stray `no GPUs found` stderr in the executed output. *judgment.*
6. [P1] **NSI-6** — Strengthen the exercise set (it omits the obvious He-derivation and variance-preservation problems the section now sets up). *authored.*
7. [P2] **NSI-7** — Prose: the intro paragraph is a wall of hedged "you might…/we delve…" filler; tighten. Summary should gain one He sentence. *authored.*

---

## 1. Coverage

### Add

**He/Kaiming initialization (P0, in scope — this is its home).** The task brief, the research digest (§B, §E.ii), and the scope map all agree: because this chapter "uses **ReLU** throughout," the ReLU counterpart to Xavier — $\sigma^2 = 2/n_\text{in}$ — **belongs here, not in `mlp.md`**. It is currently absent from the body (confirmed in the rendered page: "He/Kaiming initialization: Not mentioned in the main body text"). This is doubly awkward because:
- The chapter's **own slide deck already teaches it** correctly (lines 519–536: "Kaiming / He (2015): $\sigma^2 = 2/n_\text{in}$ … Default for modern CNNs and Transformers"). The prose body should not lag its slides.
- The citation `He.Zhang.Ren.ea.2015` ("Delving deep into rectifiers") **already exists in `d2l.bib`** (line 1145) and is already cited by `mlp.md` for pReLU — so this is a no-new-bib add.
- The framework chapter that owns the init *API*, `chapter_builders-guide/init-param.md`, demonstrates `xavier_uniform_`/`init.Xavier` in code but contains **no derivation** — so the *theory* gap is real and this file is where it should be filled, forward-pointing to Builders' Guide for the API call.

The right size is one tight subsection (`### He Initialization`) after Xavier: state that ReLU zeroes ~half its inputs, halving post-activation variance; the one-line consequence is that to preserve forward variance you double the weight variance to $2/n_\text{in}$; give the formula; note it is the default for ReLU nets and ships as `kaiming_normal_`/`He` in the frameworks. Drafted in full in **NSI-1**.

**Substance for "Default Initialization" (P0).** Lines 277–289 are a stub: three sentences ("the framework will use a default … which often works well for moderate problem sizes") followed by five blank lines. At the top-program bar this either earns a real paragraph (what the defaults actually *are* — small Gaussian/uniform scaled by fan-in, i.e. the frameworks already do a Xavier-like thing — and the crucial caveat that defaults degrade with depth, which is precisely why the rest of the section exists) or it folds into the Xavier subsection. Drafted in **NSI-2**.

**Forward-pointers that exist only in slides (P1).** The slide deck has a clean "Modern building blocks" slide (lines 551–567: BatchNorm/LayerNorm/residual/RMSNorm/mixed-precision) and an explicit "(SGD alone doesn't [break symmetry])" note. The *body* has neither. Per the scope map, normalization is **forward-point only** (`sec_batch_norm` exists at `chapter_convolutional-modern/batch-norm.md:7`; LayerNorm at `subsec_layer-normalization-in-bn`). One sentence in the "Beyond" subsection — "normalization layers (:numref:`sec_batch_norm`) and residual connections largely remove the burden from initialization in very deep nets" — closes the gap without importing the topic. And the symmetry paragraph already promises "dropout regularization (to be introduced later)" with no `:numref:`; `sec_dropout` exists. Both in **NSI-4**.

### Remove / trim

- **MXNet tab (P1, book-wide).** Apache MXNet was archived by the ASF (2023). The four-equal-tabs presentation is misleading in 2026. This is a book-wide call (flag for the overview), but this file specifically has the `npx`/`autograd.record()` idiom (lines 124–131) and a **visible stderr artifact** in its committed output (see §3 MXNet). See **NSI-5**.
- Nothing else to cut. The section is lean; the two demos and three conceptual subsections all earn their place.

### Reorder / restructure

The spine is sound: §5.4.1 problem (vanishing/exploding/symmetry) → §5.4.2 fix (initialization) → Summary. The only structural change is **inserting `### He Initialization` between Xavier (5.4.2.2) and "Beyond" (5.4.2.3)** (NSI-1), which keeps the natural progression Xavier→He→frontier. Optionally retitle the parent `## Vanishing and Exploding Gradients` is fine as-is; do **not** flatten — the nested `###` structure is exactly the guide's recommended shape.

---

## 2. Teaching quality

### Structure & flow

Good 2-`##` + nested-`###` shape, matching the house style. The logical arc (product of Jacobians → why it blows up/dies → break symmetry → scale variance) is clean. The one weak link is the **Default → Xavier → Beyond** sub-sequence: Default is empty, Xavier is strong, "Beyond" is a graceful hand-wave. NSI-1/NSI-2 give the middle real weight.

### Figures

- **One figure** (confirmed in render): the sigmoid + gradient plot (cell `numerical-stability-and-init-vanishing-gradients`). It is a **computed data plot** (`d2l.plot`), teaches the saturation directly, and is correctly *not* a hand-drawn schematic — keep it. Caption is implicit (legend `['sigmoid','gradient']`); fine.
- The exploding-gradient demo is **text output**, not a figure — appropriate; the ~$10^{24}$ entries (verified in all four manifests) are more visceral as raw numbers than a plot would be.
- **Missing figure (P2, judgment, optional).** The variance-preservation story is the one place a schematic would unlock the idea: a small diagram showing a signal's variance staying flat through depth under Xavier/He vs. shrinking (vanishing) / growing (exploding) under naive init — i.e. three depth-vs-$\mathrm{Var}[h^{(\ell)}]$ traces. This is exactly the kind of *illustrative* figure the house style pre-generates via a committed generator. **Caveat:** the MLP chapter has **no figure generator yet** (`tools/gen_mdl_*` exist only for the math appendix); adding one is an L-effort infrastructure step. I scope this as optional/judgment in NSI (see §4) rather than P0/P1 — the chapter teaches adequately without it, and the cost is real. Do **not** add inline matplotlib for it.

### Prose & clarity

- **Intro (lines 10–29) is the worst offender.** It is ~20 lines of throat-clearing: "Until now, we took the initialization scheme for granted, glossing over the details… You might have even gotten the impression that these choices are not especially important." Three sentences say the same thing. A top-text opens with a hook. Tighten to ~8 lines (drafted in NSI-7).
- Line 365 "the assumption for **nonexistence of nonlinearities**" is clumsy phrasing — read it as "the assumption that there are no nonlinearities." Minor; rolled into NSI-3.
- The symmetry argument (lines 227–266) is **crisp and correct** — keep. One tiny improvement: it asserts "dropout … would [break it]" without a pointer (NSI-4).
- The Xavier derivation prose is mostly good but **asserts two steps** a top course would show (NSI-3): (a) the variance line jumps to $\sum E[w_{ij}^2]E[x_j^2]$ using $E[w^2]=\sigma^2$ without flagging that this uses the zero-mean assumption ($\mathrm{Var}[w]=E[w^2]-E[w]^2=\sigma^2$); (b) "Using the same reasoning as for forward propagation, we see that the gradients' variance can blow up unless $n_\text{out}\sigma^2=1$" — the backward direction is **stated, never derived**. For the assignable-at-a-top-program bar this should be a one-line justification, not an appeal to symmetry the reader must reconstruct.

### Exercises

The four exercises (lines 405–408) are thin for this section's content and **omit the two most natural problems the material now sets up**:
- No exercise asks the reader to **derive the He factor of 2** — the single best consolidation exercise for the new content.
- No exercise on **variance preservation in practice** (e.g. initialize a deep linear stack three ways, measure $\mathrm{Var}[h^{(\ell)}]$ vs. depth).
- Ex. 2 ("Can we initialize all weights in linear/softmax regression to the same value?") is good and subtle (the answer is *yes* for a single linear layer — there's no permutation symmetry to break) — keep.
- Ex. 4 (LARS, `You.Gitman.Ginsburg.2017`) is a fine pointer; keep.

Strengthened set drafted in **NSI-6**.

---

## 3. Code & examples

### Does the code teach?

Yes — both code cells *compute* something the prose discusses (sigmoid gradient saturation; matrix-product explosion). No figure-drawing boilerplate, no walls of code. The imports cell is correctly single-per-framework at the top. This is well done.

### PyTorch

- **Vanishing (lines 134–142):** `requires_grad=True`, `y.backward(torch.ones_like(x))` to get the elementwise derivative, then `.detach().numpy()` for plotting — correct and idiomatic for torch 2.x. Fine.
- **Exploding (lines 200–207):** `torch.normal(0, 1, size=(4,4))`, `M @ ...` — clean, modern. Output verified: explodes to ~$10^{24}$–$10^{25}$. 
- Minor: `print('a single matrix \n',M)` (line 203) has no space after the comma and embeds `\n` in the label — cosmetic, not worth a change.

### JAX

- **Vanishing (lines 153–159):** `vmap(grad(jax.nn.sigmoid))` is the elegant, idiomatic way to get the per-element derivative — arguably the nicest of the four. Good.
- **Exploding (lines 218–224):** uses `d2l.get_key()` for split RNG keys, `jnp.matmul`. Correct functional-JAX style. Output explodes to ~$10^{22}$–$10^{23}$. This is modern JAX (no Linen/stax involved here), so no currency issue in this file.

### TensorFlow

- **Vanishing (lines 144–151):** `tf.GradientTape`, `tf.nn.sigmoid` — correct TF2 idiom. Fine.
- **Exploding (lines 209–215):** `tf.random.normal`, `tf.matmul` — fine. Output explodes to ~$10^{24}$–$10^{25}$. No Keras-3 surface in this file, so no TF currency flag here.

### MXNet

- **Vanishing (lines 123–131):** `x.attach_grad()` + `autograd.record()` + `npx.sigmoid` — legacy MXNet idiom. Code is correct.
- **Executed-output artifact (flag):** the committed MXNet manifest for this cell has `kind: "mixed"` with a **stderr line** `"[..] GPU context requested, but no GPUs found."` This is a CPU-capture artifact (the other three frameworks emit no such line) and will render as a stray warning above the sigmoid plot. Note in NSI-5; it disappears if the MXNet tab is tombstoned, otherwise the capture should be redone CPU-clean.
- Otherwise MXNet outputs match the others (~$10^{24}$–$10^{26}$). Weight PyTorch/JAX most heavily per the guide.

### Cross-framework consistency & d2l conventions

- The four exploding-gradient cells differ only in the **`print` label formatting** (`'a single matrix'` vs `'a single matrix \n'`) — MXNet line 194 omits the `\n` the other three have. Gratuitous but harmless; optionally normalize (P2, low value).
- Stable cell IDs present and consistent across tabs. Single imports cell. `d2l.plot` used over raw matplotlib. Conventions are clean.

---

## 4. Implementation spec (the executable part)

### NSI-1 — Add He/Kaiming initialization (ReLU variance argument + formula)  ·  [P0] · [M] · [authored]
- **Type:** coverage / currency
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md` — insert a new `### He Initialization` subsection **between** the end of the Xavier subsection (after line 369, the paragraph ending "…turns out to work well in practice.") and the `### Beyond` heading (line 372).
- **Change:** insert the following verbatim:

```
### He Initialization
:label:`subsec_he_init`

The Xavier analysis above assumed a layer *without nonlinearities*. The argument
breaks in a specific, fixable way once we insert a ReLU. Recall that
$\textrm{ReLU}(z) = \max(0, z)$ zeroes every negative pre-activation. If the
pre-activations are symmetric about zero---as they are when weights have zero
mean---then ReLU discards, in expectation, *half* of them, and for the surviving
half it passes the value through unchanged. Its effect on the variance of a
zero-mean, symmetric signal is therefore to **halve** it:
$\textrm{Var}[\textrm{ReLU}(z)] = \tfrac{1}{2}\textrm{Var}[z]$.

Propagating this through the same forward computation as before, the variance of
the layer output is now $\textrm{Var}[o_i] = \tfrac{1}{2} n_\textrm{in}\,\sigma^2\,\gamma^2$,
with the extra factor of $\tfrac{1}{2}$ coming from the rectifier. To keep the
variance fixed across layers we must compensate by *doubling* the weight
variance:

$$\sigma^2 = \frac{2}{n_\textrm{in}}.$$

This is *He* (or *Kaiming*) *initialization* :cite:`He.Zhang.Ren.ea.2015`, and it is
the standard choice for the ReLU-family activations this chapter uses throughout.
Because Xavier and He differ only by this factor of two and by which fan size
they key on, they are easy to confuse; the rule of thumb is **Xavier for
$\tanh$/sigmoid, He for ReLU**. Most frameworks ship both as named initializers
(e.g. PyTorch's `kaiming_normal_`, in fact the default for `nn.Linear`), and we
return to invoking them through the parameter-initialization API in
:numref:`sec_init`.
```

- **Touches:** none (citation `He.Zhang.Ren.ea.2015` already in `d2l.bib:1145`; reference `sec_init` is the label of `chapter_builders-guide/init-param.md` — verify that file's `:label:` is `sec_init`; if it differs, substitute the actual label or drop the trailing clause). No bib edit, no figure.
- **Done when:** the rendered page shows a "5.4.2.x He Initialization" subsection containing the boxed formula $\sigma^2 = 2/n_\textrm{in}$; the `:cite:` resolves (no `??`); `make html` clean; the body now matches the existing slide (lines 527–533).
- **Depends on:** none. (Coordinate with NSI-7 so the new Summary sentence references it.)

### NSI-2 — Fill the empty "Default Initialization" stub  ·  [P0] · [S] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md`, the `### Default Initialization` subsection, lines 277–289 (the three-sentence body + five trailing blank lines, ending "…moderate problem sizes." followed by blank lines 286–290).
- **Change:** replace the body (the text from "In the previous sections" through "moderate problem sizes." and the trailing blank lines) with:

```
In the previous sections, e.g., in :numref:`sec_linear_concise`, we initialized
weights by drawing them from a normal distribution with a small, fixed standard
deviation. If we do not specify an initialization method at all, every framework
falls back to a default scheme---and those defaults are *not* arbitrary: each
samples from a distribution whose spread is tied to the layer's fan-in (a
Xavier- or He-like rule of exactly the kind we derive below). These defaults
work well for moderately sized networks. They become unreliable, however, as
depth grows: the variance analysis that follows explains both *why* they work
and *where* they break, and what to reach for instead.
```

- **Touches:** none (`sec_linear_concise` resolves — `chapter_linear-regression/linear-regression-concise.md:7`).
- **Done when:** the "Default Initialization" subsection renders with a single substantive paragraph, no run of blank lines, and motivates the Xavier/He derivation that follows.
- **Depends on:** none (reads naturally before NSI-1/NSI-3 in the same subsection group).

### NSI-3 — Tighten the Xavier derivation (justify backward step + variance identity)  ·  [P1] · [S] · [authored]
- **Type:** teaching / prose
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md`, the Xavier variance derivation, lines 318–347.
- **Change (two edits):**
  1. After the variance display block (the `\end{aligned}` at line 326–327) and before "One way to keep the variance fixed" (line 329), insert a one-line justification of the identity used:
     ```
     where we used $E[w_{ij}^2] = \textrm{Var}[w_{ij}] = \sigma^2$ (the weights have
     zero mean) and likewise $E[x_j^2] = \gamma^2$.
     ```
  2. Replace the asserted backward-pass sentence (lines 331–337, "Now consider backpropagation. There we face a similar problem … where $n_\textrm{out}$ is the number of outputs of this layer.") with a derivation-grounded version:
     ```
     Now consider backpropagation. A gradient signal flowing *back* through this
     layer is multiplied by $\mathbf{W}^\top$, so by the identical variance
     computation---now summing over the $n_\textrm{out}$ outputs the layer feeds---its
     variance is scaled by $n_\textrm{out}\,\sigma^2$. Keeping the *backward* signal's
     variance fixed therefore requires $n_\textrm{out}\,\sigma^2 = 1$.
     ```
- **Touches:** none.
- **Done when:** the derivation states the zero-mean origin of $E[w^2]=\sigma^2$ and gives a one-sentence reason (multiplication by $\mathbf{W}^\top$, sum over $n_\textrm{out}$) for the backward condition rather than "the same reasoning"; renders cleanly (watch the multi-line `aligned` blocks — no `$` immediately followed by a digit, per the PDF tripwire).
- **Depends on:** none.

### NSI-4 — Add the two missing forward-pointers (symmetry→dropout, normalization)  ·  [P1] · [S] · [mechanical]
- **Type:** coverage / cross-reference
- **Where & change (two edits):**
  1. Symmetry paragraph, line 265: `dropout regularization (to be introduced later) would!` → `dropout regularization (:numref:`sec_dropout`) would!`
  2. "Beyond" subsection — append one sentence after line 385 ("…using a carefully-designed initialization method."), before the "If the topic interests you…" paragraph:
     ```
     In very deep networks, normalization layers (:numref:`sec_batch_norm`) and
     residual connections (:numref:`sec_resnet`) largely remove this burden from
     initialization by re-centering activations during training; we cover them in
     later chapters.
     ```
- **Touches:** none. Verify labels resolve: `sec_dropout` (`dropout.md:7` ✓), `sec_batch_norm` (`chapter_convolutional-modern/batch-norm.md:7` ✓), `sec_resnet` (verify the ResNet file's label; if absent, drop the residual clause and keep only the batch-norm pointer).
- **Done when:** both `:numref:` references resolve in HTML (no `??`); the symmetry→dropout promise is now linked; the normalization pointer exists in prose (not just slides). Apply blindly *except* the `sec_resnet` existence check.
- **Depends on:** none.

### NSI-5 — MXNet tab: tombstone + clean the stray stderr  ·  [P1] · [S/M] · [judgment]
- **Type:** currency / code
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md` MXNet cells (imports lines 31–37; vanishing lines 123–131; exploding lines 191–198); and the committed manifest `outputs/mxnet/chapter_multilayer-perceptrons/numerical-stability-and-init.json` (cell `numerical-stability-and-init-vanishing-gradients`, which carries `kind:"mixed"` and a stderr line `"GPU context requested, but no GPUs found."`).
- **Change:** Apply the **book-wide MXNet decision** (defer to the cross-cutting overview — do not unilaterally delete here). Whatever the decision (drop tab / collapse to a tombstone note / demote ordering), this file is in-scope for it. *Independently of that decision:* the MXNet vanishing-gradients output should not carry a `no GPUs found` stderr line that the other three frameworks lack — if the MXNet tab is retained, re-capture that notebook CPU-clean so the artifact disappears; if dropped, this resolves automatically.
- **Touches:** book-wide MXNet policy; possibly `make capture-outputs FILES=chapter_multilayer-perceptrons/numerical-stability-and-init.md` if re-capturing.
- **Done when:** consistent with the book-wide MXNet decision; and the rendered MXNet tab (if kept) shows no stray "GPU context requested" warning above the sigmoid figure.
- **Depends on:** book-wide MXNet-tab decision (overview).

### NSI-6 — Strengthen the exercise set  ·  [P1] · [S] · [authored]
- **Type:** teaching
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md`, Exercises, lines 405–408. Keep existing items; add two new ones (insert as items between current 2 and 3, or append — numbering is auto via the `1.` markdown convention).
- **Change:** add:
  ```
  1. The Xavier derivation assumed a linear layer. Repeat it for a layer followed
     by a ReLU: show that, for zero-mean symmetric pre-activations,
     $\textrm{Var}[\textrm{ReLU}(z)] = \tfrac{1}{2}\textrm{Var}[z]$, and conclude that
     preserving forward variance requires $\sigma^2 = 2/n_\textrm{in}$ (He
     initialization). Where does the factor of two come from intuitively?
  1. Initialize a deep stack of linear layers (say 50 layers, width 100) three
     ways---weights $\sim\mathcal{N}(0,1)$, Xavier, and He---feed in a unit-variance
     input, and plot $\textrm{Var}[\mathbf{h}^{(\ell)}]$ as a function of depth
     $\ell$. Which scheme keeps the variance flat? Now insert a ReLU after each
     layer and repeat. Do your observations match the theory?
  ```
- **Touches:** none. (These are paper/notebook exercises; no solution slot is required by the house style, but if a solutions appendix exists, supply: factor-of-two = ReLU discards half the probability mass in expectation; He keeps Var flat with ReLU, Xavier keeps it flat without.)
- **Done when:** the exercise list contains a He-derivation problem and a variance-preservation experiment, both consistent with NSI-1's notation.
- **Depends on:** NSI-1 (shares the He notation/result).

### NSI-7 — Prose polish: intro + Summary  ·  [P2] · [S] · [authored]
- **Type:** prose
- **Where:** intro paragraph lines 10–29; Summary lines 395–401.
- **Change (two edits):**
  1. Replace the intro (lines 10–29) with a tighter version:
     ```
     Every model so far has required us to initialize its parameters from some
     chosen distribution, and we have taken those choices for granted. They are
     not innocuous. The initialization scheme interacts with the choice of
     activation function to determine whether gradients flow at a usable scale or
     instead *vanish* (so learning stalls) or *explode* (so it diverges)---and
     hence how fast, or whether, optimization converges at all. This section makes
     the failure modes concrete and develops the variance-preserving heuristics
     (Xavier and He initialization) that fix them.
     ```
  2. Summary (after line 400, the Xavier sentence) add one sentence:
     ```
     For ReLU networks, He initialization scales the weight variance to $2/n_\textrm{in}$ to compensate for the rectifier halving the activation variance.
     ```
- **Touches:** none.
- **Done when:** the intro is ~8 lines with a clear thesis (no "you might have gotten the impression…" hedging); the Summary mentions He alongside Xavier. Renders clean.
- **Depends on:** NSI-1 (Summary sentence mirrors the added subsection).

### NSI-FIG (optional) — Variance-through-depth schematic  ·  [P2] · [L] · [judgment]
- **Type:** figure
- **Where:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md`, candidate placement at the head of the Xavier subsection (~line 293).
- **Change:** a *pre-generated* (never inline) illustrative figure: three traces of $\mathrm{Var}[\mathbf{h}^{(\ell)}]$ vs. layer index $\ell$ on a log-$y$ axis — one decaying to ~0 (naive small init → vanishing), one growing off-scale (naive large init → exploding), one flat (Xavier/He). Caption ties it to the forward-variance equation. Include with `![…](../img/<id>.svg)` + `:label:`fig_…`` and **no drawing code in any notebook cell**.
- **Touches:** requires creating an MLP-chapter figure generator (none exists today — `tools/gen_mdl_*` cover only the math appendix), e.g. `tools/gen_mlp_figures.py` importing the shared house style in `tools/gen_mdl_figures.py`, plus `make figures` and a committed `img/<id>.svg`. Use the `mdl-figure` skill.
- **Done when:** the SVG is committed, byte-idempotent, referenced by `:numref:`, passes `figure-style-audit`, and renders in HTML+PDF.
- **Depends on:** none. **Explicitly optional** — the section teaches adequately without it; only undertake if the figure-generator infrastructure is being added for the MLP chapter anyway.

---

## 5. Keep — what is already excellent (do not lose this)

- **The product-of-Jacobians framing** (lines 64–93) — the `$\mathbf{M}^{(L)}\cdots\mathbf{M}^{(l+1)}\mathbf{v}^{(l)}$` decomposition with the `\underbrace` defs is exactly the right level of rigor and is the conceptual core. Keep verbatim.
- **The exploding-gradient demo** (lines 191–225) — multiplying 100 random $4\times4$ Gaussians to ~$10^{24}$ is a memorable, correct, four-framework-consistent payoff (verified in all four manifests). Do not replace with a figure.
- **The sigmoid saturation plot** (the one figure) — a genuine teaching data-plot, correctly computed not hand-drawn.
- **The symmetry-breaking argument** (lines 227–266) — crisp, correct, and it makes the right subtle point (SGD alone won't break it, dropout will). Ex. 2 (same-value init for linear/softmax) is a genuinely good, subtle question — keep.
- **The Xavier derivation** itself (the forward variance computation and the $\tfrac12(n_\text{in}+n_\text{out})\sigma^2=1$ compromise, lines 294–363) — real, not asserted; the uniform-distribution corollary ($U(\pm\sqrt{6/(n_\text{in}+n_\text{out})})$) is a nice concrete touch. NSI-3 only tightens two steps; do not rewrite it.
- **The slide deck** (lines 427–583) — out of scope for editing, but note it already contains the correct He/Kaiming treatment (lines 519–536) and the normalization/residual pointers (551–567); NSI-1/NSI-4 bring the *body* up to the slides, so the two will be consistent afterward.
- **Single-imports-cell convention and stable cell IDs** — clean; do not disturb.
