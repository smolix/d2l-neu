# §5 Multilayer Perceptrons — slide-deck content plan

**Scope:** the 7 decks in `chapter_multilayer-perceptrons/`: `mlp.md`,
`mlp-implementation.md`, `backprop.md`, `numerical-stability-and-init.md`,
`generalization-deep.md`, `dropout.md`, `kaggle-house-price.md`.

**This is a planning doc only.** No chapter `.md`/`.qmd` was edited, no
`<!-- slides -->` block touched, no slides generated, no build run.

---

## Executive summary (read this first)

Unlike a typical legacy-deck upgrade, **all seven §5 decks already carry a
full north-star `<!-- slides -->` block** (cover, dividers, kickers,
`.cols`/`.col`, `.d2l-note`/`.rule`/`.warn`, `@id` code refs, `@!`/`@-`,
`. . .` fragments, `only=`/`except=` framework scoping). They are **not
shallow** in the sense the brief feared: the hard technical content the depth
rubric demands is, in almost every case, **already present and staged
elegantly**:

- `mlp`: XOR fold figure + a verified hand-built XOR net; the universal
  approximation theorem stated *with* its three caveats; every activation
  paired with **its gradient** and the vanishing-gradient setup.
- `backprop`: the `prod`/chain-rule rule; a fully worked **numeric** example
  ($e=(a+b)c$ then a 2-2-1 net) staged with fragments; the **dead-ReLU as a
  zero gradient row**; reverse-mode framing; the memory cost.
- `numerical-stability`: gradient as a **product of per-layer Jacobians**;
  the **full Xavier variance derivation** and the **He factor-of-two**;
  symmetry breaking; the activation effect.
- `generalization-deep`: double descent, interpolation regime, implicit bias
  (max-margin), grokking, NTK — genuinely modern and honest.
- `dropout`: the **three views** (thinned net / $2^n$ ensemble /
  co-adaptation), the **inverted-scaling expectation derivation**, and the
  train-vs-eval switch (the classic bug) called out.
- `kaggle`: end-to-end pipeline, leakage warning, **log-RMSE rationale**,
  K-fold figure, and — importantly — the **MLP is now trained competently**
  (the earlier "undertrained" defect is fixed; see kaggle verdict).

So this is a **tightening / consistency pass**, not a rebuild. The work that
remains is real but bounded: **one figure-style inconsistency to fix**
(`backprop` mixes the legacy hand-drawn `img/forward.svg` with the house-style
`mdl-mlp-backprop-graph.svg`), a few **overflow risks** on dense
derivation/code slides, a couple of **missing computed payoffs** (the ReLU vs
sigmoid forward-variance demo the chapter's own exercise 4 asks for would make
the init deck *show* rather than *assert*), and assorted small curations.

**Cross-deck headline (top 3):**

1. **Figure-style consistency.** Six decks are uniformly house-style
   (`img/mdl-mlp-*.svg`). The lone exception is `backprop`'s forward-graph
   slide using `img/forward.svg` (old matplotlib/pgf glyph-outline SVG —
   different fonts, different look) one slide before the house-style
   `mdl-mlp-backprop-graph.svg`. **Flag a new figure** `mdl-mlp-forward-graph`
   to replace it (details in the backprop deck below). This is the single most
   visible quality gap across the seven.
2. **The same five figures recur and carry the whole part.**
   `mdl-mlp-arch.svg` (the 784-256-10 MLP) appears in *five* of seven decks as
   the connective tissue; `mdl-mlp-xor`, `mdl-mlp-backprop-graph`,
   `mdl-mlp-double-descent`, `mdl-mlp-kfold` each anchor their deck.
   `dropout2.svg` is the one extra. This consistency is a strength — preserve
   it; do not introduce one-off figures where these already serve.
3. **`only="pytorch"` scoping is already correct where it matters and should
   be extended in two spots.** XOR (`mlp`), the whole MLP-vs-baseline arc and
   `KaggleMLP` (`kaggle`) are pytorch-only cells already scoped `only="pytorch"`.
   The gaps: in `numerical-stability` the **exploding-matrix `print` output**
   differs per framework (and is genuinely framework-flavoured) — keep the
   slide shared but verify all four inject; in `kaggle` the from-scratch
   `k_fold`/model-selection cells are **pytorch-only by construction** (mxnet/
   tf/jax run a different `k_fold(... lr)` signature and never define
   `KaggleMLP`) — the deck already splits these; preserve the split.

A note on the brief's depth rubric: measured against it, the current decks
grade **good** (5 of 7) or **adequate-plus** (2 of 7). The plan below gives,
per deck, the verdict, the 1–3 concrete gaps, the spine, the must-keep
derivation, the lead figures, the earning-their-place code, cross-links, and
the slide arc (which for these decks is largely "keep + the listed edits"
rather than "author from scratch").

---

## Deck 1 — `mlp.md` (§5.1 Multilayer Perceptrons)

**Depth verdict: GOOD.** This is the strongest deck of the seven and close to
exemplary. It leads with *why a line fails* (temperature/cats-dogs/XOR),
derives the affine-collapse, introduces the nonlinearity, *proves the point
with a verified XOR computation*, states the UAT with all three caveats, and
gives every activation **with its gradient**.

Gaps vs rubric (minor):
1. The XOR construction matrix ($\mathbf{W}^{(1)},\mathbf{b}^{(1)},\mathbf{w}^{(2)}$)
   is shown only in the chapter body, not on the verify slide; the slide jumps
   to the result. Add the construction as a one-line `.d2l-note .rule` on the
   "hand-built XOR" slide so the audience sees *what* was constructed before
   the In/Out card proves it.
2. The activation **cheat-sheet table** is excellent but dense; verify it does
   not overflow 720px (6 rows + note). It is a table-only slide, so likely OK,
   but it is the overflow suspect here.
3. Forward-point is present (GELU/Swish/SwiGLU in the cheat sheet + recap) but
   buried; one explicit "modern default = ReLU hidden, GELU in Transformers"
   line already exists — keep it, it satisfies rubric §5.

**Core spine (3 ideas):** (a) one affine map ⇒ one straight boundary, too
rigid; (b) hidden layer **+ nonlinearity** is the minimal fix, and XOR is the
smallest proof; (c) one wide layer is a universal approximator but depth is
what makes that power parameter-efficient — and the activation's *gradient*
decides trainability.

**Must-keep derivation/theorem (stage with fragments):**
- The **affine collapse** (already a fragment slide): substitute $\mathbf{H}$
  into $\mathbf{O}$, get $\mathbf{O}=\mathbf{X}\mathbf{W}+\mathbf{b}$ with
  $\mathbf{W}=\mathbf{W}^{(1)}\mathbf{W}^{(2)}$ — "the hidden layer added zero
  expressive power." Keep the `\underbrace` staging.
- The **UAT statement + 3 caveats** verbatim: a fit *exists*; says nothing
  about SGD finding it; nor about generalization; the single layer can be
  *exponentially* wide. (This is the rubric's headline item for this deck and
  it is already done correctly — protect it.)

**Lead diagram(s) (reuse, do not draw):**
- Cover + motivation + XOR slides: `img/mdl-mlp-xor.svg` (the input-space vs
  hidden-space fold). Used 3× appropriately.
- "Insert hidden layers" slide: `img/mlp.svg` (the generic 4-5-3 MLP). NB this
  is the *one* non-`mdl-` figure here; it is the canonical d2l MLP schematic
  and is fine to keep (consistent with the chapter body `fig_mlp`).

**Code that earns its place:**
- `@mlp-xor` (**`only="pytorch"`**, already scoped): computes XOR on all four
  inputs; output `[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]` — the third column *is*
  XOR. This is the model "code earns its place by computing a result" cell.
  Confirmed output matches the executed notebook.
- `@mlp-relu-function-1` / `-2`, `@mlp-sigmoid-function-1` / `-2`,
  `@mlp-tanh-function-1` / `-2`: each function **and its gradient** as a plot.
  These are *teaching* plots (the gradient shape is the lesson, e.g. sigmoid
  peaks at 0.25), so they earn their place — keep all six. No framework
  scoping needed (code differs per fw, framing identical; the `#@tab`
  mechanism handles it).

**Cross-links / forward-points:** back to §4 softmax (the collapse target);
forward to §5.3 backprop (gradients), §5.4 init (vanishing gradients — the
sigmoid-0.25 slide explicitly sets this up), and Transformers (GELU/SwiGLU).
All present; keep.

**Slide arc (≈18 slides — keep current, apply edits 1–3):**
1. Cover — §5.1 MLPs.
2. Motivation — a line draws one boundary (temp/cats-dogs/XOR) · `mdl-mlp-xor`.
3. Divider 01 — From Linear to Nonlinear.
4. The idea: insert hidden layers · `mlp.svg`.
5. One hidden layer, written out (the two affine equations) + fragment.
6. The catch: two affine maps collapse (derivation + `\underbrace`) + warn.
7. The fix: an elementwise nonlinearity $\sigma$ + fragment.
8. Divider 02 — A Concrete Win: XOR.
9. XOR: impossible for a line, easy after a fold · `mdl-mlp-xor`.
10. Hand-built two-unit XOR net (**only=pytorch**) — *add the construction
    matrices as a `.rule` note*, then `@mlp-xor`.
11. Universal approximation: statement + 3 caveats (warn) + depth fragment.
12. Divider 03 — Activation Functions.
13. ReLU + `@mlp-relu-function-1`.
14. ReLU gradient (on/off switch) + dead-ReLU warn + `@mlp-relu-function-2`.
15. Sigmoid + `@mlp-sigmoid-function-1`.
16. Why sigmoid stalls (grad peaks 0.25, vanishes) + `@mlp-sigmoid-function-2`.
17. Tanh + `@mlp-tanh-function-1` (+ its gradient, or fold `-2` in).
18. Divider 04 — Wrap-up → cheat-sheet table → Recap.

---

## Deck 2 — `mlp-implementation.md` (§5.2 Implementation)

**Depth verdict: GOOD** (with one caveat that is intrinsic to the topic). This
is an *implementation* section, so by the brief's own rule "exactly what
changes from softmax regression" is the right altitude, and the deck nails it:
the architecture diagram, the parameter-count arithmetic, the hand-rolled
ReLU, the two-line forward, then the concise `Sequential`, then the four open
questions that set up the rest of the part.

Gaps vs rubric:
1. **Code-density risk.** Two slides ("Parameters: two weights, two biases"
   and the concise model) put a full multi-line class beside a narrow note.
   These are the overflow suspects. Mitigation: the params slide already pairs
   code with the shape/param-count note — verify 720px; if tight, make the
   scratch-params cell `@-` (code-only, no output) since its output is
   uninteresting, or drop the mxnet `attach_grad` loop visually.
2. **No computed payoff on a slide.** The training cells
   (`@mlp-implementation-training`, `-training-2`) inject a loss/acc curve
   image; the deck *asserts* "≈0.87 val accuracy" in a note rather than
   showing the number. Acceptable (the curve is the artifact), but if a single
   scalar can be surfaced, the "Train it" slide becomes a *result*, not a
   claim. Low priority.
3. The "Why these sizes?" slide reuses `mdl-mlp-arch.svg` a second time
   immediately after the "whole model" slide used it. Two adjacent slides, same
   figure. Consider dropping the figure from the second (it is a design-choices
   slide; the note + arithmetic carry it) to avoid visual repetition.

**Core spine (3 ideas):** (a) an MLP is softmax-regression + one hidden layer
+ a ReLU — *that ReLU is the entire difference*; (b) from-scratch makes the
four parameter tensors and two-line forward concrete; (c) `nn.Sequential`
absorbs the bookkeeping and computes the *same* function, with the *same*
training loop.

**Must-show "derivation":** not a theorem here, but the **parameter-count
arithmetic** $784\cdot256+256+256\cdot10+10 = 203{,}530$ is the quantitative
anchor (already present) — keep it; it makes "256 hidden units" concrete.

**Lead diagram:** `img/mdl-mlp-arch.svg` (the 784→256→ReLU→10 pipeline). The
deck's spine figure. Use once prominently (whole-model slide); thin the second
use per gap 3.

**Code that earns its place:**
- `@mlp-implementation-initializing-model-parameters` — the four parameter
  tensors (shapes are the point; pair with the shape note). Framework code
  differs (nn.Parameter / tf.Variable / flax param / mxnet attach_grad); the
  `except=`/`only=` mechanism already handles framing where it diverges.
- `@mlp-implementation-model-1` — the hand-rolled ReLU ($\max(x,0)$). The "no
  magic" cell; keep.
- `@mlp-implementation-model-2` — the two-line forward (flatten, affine-ReLU,
  affine). This is the cell that *is* the lesson; keep, paired with the
  $\mathbf{H},\mathbf{O}$ equations already on the slide.
- `@mlp-implementation-model-2-2` — the concise stack; the deck **already
  splits `except="jax"` (Sequential) vs `only="jax"` (`@nn.compact`)**, which
  is correct because Flax's construction framing genuinely differs. Preserve.
- `@mlp-implementation-training` / `-training-2` — same-loop training; keep one
  each, they show the modularity payoff.

**Cross-links / forward-points:** explicit "four open questions" slide →
§5.3/§5.4/§5.5/§5.6 (init, backprop, generalization, regularization). Already
present and is exactly the rubric's "cross-links to neighbouring chapters."
Keep verbatim.

**Slide arc (≈15 slides — keep, apply edits 1 & 3):** cover → whole-model
(`mdl-mlp-arch`) → why-these-sizes (drop 2nd figure) → divider From Scratch →
imports → parameters (+shape note; `@-` if tight) → ReLU by hand → forward (2
lines) → train → divider Concise → declared model (jax split) → same loop →
divider What's next → four open questions → recap.

---

## Deck 3 — `backprop.md` (§5.3 Forward/Backward Propagation)

**Depth verdict: GOOD on content, ADEQUATE on figure consistency.** The
mathematical content is excellent and exactly what the rubric asks: forward
pass as stored intermediates, the `prod`/chain-rule rule, the
add-passes/multiply-scales primitive, a **fully worked numeric example** (the
tiny $e=(a+b)c$ graph *then* the 2-2-1 ReLU net with real numbers), the
**dead-ReLU as a literal zero row** of $\partial L/\partial\mathbf{W}^{(1)}$,
the reverse-mode-autodiff framing, and the memory cost. This is the deck most
in danger of being a "wall of algebra," and it avoids that by staging with
fragments and grounding in numbers.

Gaps vs rubric:
1. **FIGURE-STYLE INCONSISTENCY (the one real defect in the seven decks).**
   The "Forward propagation as a graph" slide embeds **`img/forward.svg`** —
   the legacy d2l hand-drawn/pgf SVG (glyph-outlined fonts, `pt`-only viewBox,
   visibly different typography). One slide later the worked example uses the
   house-style **`img/mdl-mlp-backprop-graph.svg`**. Two computational-graph
   figures, two different styles, adjacent. This violates the "one figure
   style per chapter" rule.
   **FLAG A NEW FIGURE:** `mdl-mlp-forward-graph` — a house-style forward-only
   computational graph for the **same** one-hidden-layer net the deck uses:
   nodes $\mathbf{x}\to\mathbf{z}=\mathbf{W}^{(1)}\mathbf{x}\to
   \mathbf{h}=\phi(\mathbf{z})\to\mathbf{o}=\mathbf{W}^{(2)}\mathbf{h}\to
   L=l(\mathbf{o},y)$, plus the regularizer branch
   $\mathbf{W}^{(1)},\mathbf{W}^{(2)}\to s\to J=L+s$ (squares=variables,
   circles=operators, arrows rightward/upward). It should match
   `mdl-mlp-backprop-graph.svg`'s visual language so the two graphs read as a
   pair (forward-only, then forward+backward). Add it via the `mdl-figure`
   skill → generator `tools/gen_mdl_backprop_figures.py` (which already exists
   and produces `mdl-mlp-backprop-graph`). **Do not draw it in this plan.**
   *Alternative if a new figure is out of scope:* reuse
   `mdl-mlp-backprop-graph.svg` for the forward slide too (showing only the
   black/forward pass conceptually) and cut `forward.svg` — but a dedicated
   forward-only house-style graph is the better answer and the generator is
   already there.
2. **Overflow risk on the "Backward through our network" slide** — it stacks
   three local-derivative equations *and* two parameter-gradient equations
   with a fragment. That is a lot of display math. Verify 720px; if tight,
   split into the two it already fragments into (local derivatives slide; then
   parameter gradients slide).
3. **`layout="figure"` on the backward-sweep slide** — the worked-graph figure
   is wide (635×247pt). The brief warns wide-short figures are tiny in a
   column; here it is full-bleed (`layout="figure"`), which is the right call.
   Keep full-bleed; ensure the caption does not push it over 720px.

**Core spine (3 ideas):** (a) forward = compute and *store* every
intermediate; (b) backprop = the **same graph in reverse**, and the whole
algorithm is one move (add passes the gradient, multiply scales it); (c) it is
reverse-mode autodiff — cheap for many-params/one-scalar-loss — and storing
the intermediates is *why training is memory-hungry*.

**Must-keep derivation (the spine of the deck):**
- The `prod` chain rule: $\partial\mathsf Z/\partial\mathsf X =
  \mathrm{prod}(\partial\mathsf Z/\partial\mathsf Y,
  \partial\mathsf Y/\partial\mathsf X)$.
- The **worked numbers**, staged with fragments: forward
  $\mathbf z=[-1,2]^\top,\ \mathbf h=[0,2]^\top$ (first unit dead),
  $o=-2,\ L=2$; backward $\partial L/\partial o=-2$,
  $\partial L/\partial\mathbf z=[-4,2]^\top\odot[0,1]^\top=[0,2]^\top$,
  $\partial L/\partial\mathbf W^{(1)}=\begin{smallmatrix}0&0\\2&4\end{smallmatrix}$
  with the **all-zero top row** = the dying ReLU in one matrix. This is the
  rubric's "the ONE derivation that matters, staged elegantly" — already done;
  protect it.

**Lead diagram(s):**
- Forward slide: **NEW `mdl-mlp-forward-graph`** (per gap 1), replacing
  `forward.svg`.
- Backward-sweep slide: `img/mdl-mlp-backprop-graph.svg` (forward black +
  backward blue, the dead unit annotated). The payoff figure; keep full-bleed.
- Motivation slide: `img/mdl-mlp-arch.svg` (the net whose gradients we chase).

**Code that earns its place:** **none, intentionally.** This deck is
derivation-driven; the chapter has no code cells (it is pure math + figures).
The rubric's "code earns its place" criterion is satisfied by *absence* here —
adding autodiff code would be decoration. The deck instead points to §2.5
autograd ("confirm every number with `backward()`"). Correct call; keep
code-free.

**Cross-links / forward-points:** to §2.5 autograd (reverse vs forward mode),
to §5.1 (the dying ReLU first met there), to §5.4 (the memory/Jacobian-product
thread continues). All present.

**Slide arc (≈16 slides — keep, apply edits 1–2):** cover → motivation
(`mdl-mlp-arch`) → divider Forward → forward equations (+J=L+s fragment) →
**forward graph (NEW figure)** → divider Backprop → chain rule (`prod`) →
backward through the net (split if tight) → one move (add/multiply, 2-col
rules) → divider Worked Example → tiny graph by hand (fragment) → forward with
numbers → **backward sweep (full-bleed `mdl-mlp-backprop-graph`)** → dying-ReLU
matrix (warn) → divider Why It Matters → reverse-mode autodiff → fwd/bwd
interlock + memory warn → recap.

---

## Deck 4 — `numerical-stability-and-init.md` (§5.4)

**Depth verdict: GOOD.** Hits every anchor the brief lists: gradient as a
**product of per-layer Jacobians** $\mathbf M^{(L)}\cdots\mathbf M^{(\ell+1)}$;
spectral-radius <1 vanishes / >1 explodes with the $\rho^{L-\ell}$ compounding
note; the sigmoid-saturation demo; the exploding random-matrix demo; the
**symmetry-breaking** argument; and the **full Xavier variance derivation**
($\mathrm{Var}[o_i]=n_\text{in}\sigma^2\gamma^2$ → forward/backward
compromise → He's factor of two). The "three crashes you'll actually see"
slide is a great honest-gotchas addition.

Gaps vs rubric:
1. **A computed payoff is missing and the chapter's own exercise 4 hands it to
   us.** The deck *asserts* "$0.25^{10}\approx10^{-6}$" and shows the exploding
   matrix, but never *shows* variance staying flat vs collapsing under the
   three init schemes. Exercise 4 ("initialize a 50-layer stack three ways
   [N(0,1), Xavier, He], plot Var[h] vs depth") is exactly the demo that would
   turn the variance-derivation section from *told* to *shown*.
   **Decision needed (flag, do not author):** either (a) add a small computed
   cell + a `d2l.plot` of $\mathrm{Var}[\mathbf h^{(\ell)}]$ vs $\ell$ for the
   three schemes (this is *teaching code that computes a result* — squarely
   rubric §4), capture its output, and give it a slide between "Xavier/He" and
   "Init is the floor"; or (b) keep the deck assertion-only and note the demo
   lives in the exercises. Option (a) is the single highest-value content add
   across the seven decks. If pursued it needs a new `#id` cell in the source
   `.md` (out of scope for *this* doc — flag for the author) and a notebook
   re-run + capture.
2. **The exploding-matrix slide depends on injected `print` output that
   differs per framework.** Confirmed the PyTorch output ("a single matrix … /
   after multiplying 100 matrices …") **is present in the committed
   `outputs/pytorch/.../numerical-stability-and-init.json`** store, so it will
   inject. But the numbers differ across pytorch/tf/jax/mxnet and the slide
   shows raw entries; verify all four inject (the scratch `_notebooks/` copy I
   skimmed had it captured in the store, not inline). Keep the slide **shared**
   (framing is identical), but this is the one slide to eyeball per framework.
3. **Two adjacent figures, both `mdl-mlp-arch`-family.** Motivation uses
   `mdl-mlp-arch.svg`; the "gradient is a product down the chain" slide uses
   `mdl-mlp-backprop-graph.svg`. Different figures, so this is fine (not a
   repeat) — but note the backprop-graph here is doing double duty as a
   "Jacobian per layer" illustration, which is apt. Keep.

**Core spine (3 ideas):** (a) a deep gradient is a **product** of per-layer
Jacobians, so it vanishes or explodes geometrically; (b) the cure is to
**preserve variance** layer-to-layer — derive $\sigma^2=1/n_\text{in}$
forward, reconcile with backward, get Xavier; ReLU halves variance, so He
doubles $\sigma^2$; (c) and you must **break symmetry** (random, never
constant) — init is the floor, normalization+residuals are the ceiling.

**Must-keep derivation (the rubric's headline for this deck — already done):**
- $\mathbb E[o_i]=0,\ \mathrm{Var}[o_i]=n_\text{in}\sigma^2\gamma^2$ (with the
  $E[w^2]=\mathrm{Var}[w]=\sigma^2$ step) → forward demand
  $n_\text{in}\sigma^2=1$, backward demand $n_\text{out}\sigma^2=1$, **Xavier**
  averages them: $\sigma^2=2/(n_\text{in}+n_\text{out})$.
- **He:** $\mathrm{Var}[\mathrm{ReLU}(z)]=\tfrac12\mathrm{Var}[z]$ ⇒
  $\sigma^2=2/n_\text{in}$. Keep the "one factor of two apart" 2-col slide.

**Lead diagram(s):** `img/mdl-mlp-arch.svg` (motivation) and
`img/mdl-mlp-backprop-graph.svg` (the Jacobian-per-layer / product slide). Both
house-style; keep.

**Code that earns its place:**
- `@!numerical-stability-and-init-vanishing-gradients` (**output-only**,
  already `@!`): sigmoid + its gradient on one axis — shows the 0.25 peak and
  the flat tails. Teaching plot; keep.
- `@numerical-stability-and-init-exploding-gradients`: prints a single matrix
  then the product of 100 — entries run away. *Computes the failure*; keep
  (see gap 2 re per-framework injection).
- **(proposed, gap 1)** a variance-vs-depth demo cell — would be the deck's
  strongest "code computes a result" slide if the author adds it.

**Cross-links / forward-points:** to §5.1 (sigmoid/vanishing first met), to
§5.3 (the Jacobian product extends the backprop chain), forward to BatchNorm/
LayerNorm and ResNet (the "init is the floor" slide already names both). All
present.

**Slide arc (≈14 slides — keep; decide gap 1):** cover → motivation (3 ideas;
`mdl-mlp-arch`) → divider Unstable Gradients → gradient = product of Jacobians
(`mdl-mlp-backprop-graph`) → two ways it misbehaves (spectral radius, frags) →
vanishing: sigmoid saturates (`@!…vanishing`) → exploding: random matrices
(`@…exploding`) → three crashes you'll see (frags) → symmetry breaking (warn)
→ divider Variance-Preserving Init → keep variance constant (derivation +
frag) → forward/backward compromise → Xavier vs He (2-col, factor of two) →
**[optional NEW: variance-vs-depth demo — gap 1]** → init is the floor
(BN/residuals) → recap.

---

## Deck 5 — `generalization-deep.md` (§5.5)

**Depth verdict: GOOD.** This is a conceptual/no-code section and the deck is
genuinely modern and honest — exactly the "modernize but forward-point" the
memory file asks for. It covers: the interpolation regime, **double descent**
(with the house figure), why-bigger-is-better (room for the optimizer to find
a simple interpolant), **implicit bias** (SGD → max-margin on separable data,
*provably*), **grokking**, the **NTK**, early stopping (fit-clean-first), and
weight decay reinterpreted as inductive bias. The bias-variance decomposition
is written out. This satisfies the rubric's "double descent, forward-point
honestly" anchor fully.

Gaps vs rubric:
1. **No computed result anywhere** — but this is *correct* for this section
   (it has zero code cells; it is a survey of phenomena). The rubric's code
   criterion is N/A; do **not** manufacture a double-descent toy run here (it
   would be a multi-hour sweep that teaches less than the figure). Keep
   code-free. (Contrast kaggle, where computation is the point.)
2. **`mdl-mlp-double-descent.svg` is used twice** — motivation (`width=100%`)
   and the dedicated full-bleed "two valleys" slide (`layout="figure"`,
   `width=70%`). The motivation use is justified (it is the hook), but two
   uses of one figure in one deck is the repetition to watch. Consider
   cropping the motivation use to a teaser (it already has a different,
   punchier caption) — acceptable as is; low priority.
3. **Density on the bias-variance slide** — the decomposition equation with
   three `\underbrace` terms beside a note. Verify 720px; it is a single
   equation so likely fine.

**Core spine (3 ideas):** (a) deep nets **interpolate** (fit anything,
including random labels), so classical complexity bounds can't explain
generalization and nearly all gains come from closing the generalization gap;
(b) **double descent** — past the interpolation threshold, more capacity helps
again, because there is *more room for the optimizer to pick a simple
interpolant*; (c) the **optimizer is the regularizer** (implicit max-margin
bias; grokking; NTK) and explicit tricks merely *nudge* that bias.

**Must-keep "derivation":** the **bias-variance decomposition**
$\text{error}=\text{bias}^2+\text{variance}$ (with the
↓-with-capacity / ↑-with-capacity annotations) as the *classical* baseline the
double-descent figure then violates — already staged; keep. The provable
statement "GD on logistic loss → max-margin separator on separable data" is
the deck's rigor anchor (a `.rule` callout); keep.

**Lead diagram:** `img/mdl-mlp-double-descent.svg` — the U-curve vs the
two-valley curve with the interpolation threshold marked. *The* figure of the
deck; carries the whole through-line. Keep full-bleed on the dedicated slide.

**Code that earns its place:** none (by design — see gap 1).

**Cross-links / forward-points:** to §4 generalization-basics
(`fig_capacity_vs_error`, the U-curve), to the VC-dim/Rademacher chapter, to
§5.6 dropout ("the next such tool"), forward to BatchNorm. Plus genuinely
current references (Nakkiran double descent, Soudry implicit bias, Power et al.
grokking, Jacot NTK). All present; this is the deck's strength.

**Slide arc (≈16 slides — keep as-is):** cover → motivation (`double-descent`,
fitting-is-easy) → divider Overfitting → no-free-lunch/inductive-bias →
deep nets break the classical picture (frags + VC warn) → divider Double
Descent → classical bias-variance (decomposition) → **two-valley figure
(full-bleed)** → why bigger is better past threshold → divider Why GD finds
simple solutions → implicit bias (max-margin rule) → grokking → NTK →
divider Regularization in practice → early stopping → classical penalties
reinterpreted → recap.

---

## Deck 6 — `dropout.md` (§5.6)

**Depth verdict: GOOD.** Hits every dropout anchor: the **three views**
(thinned subnetwork / implicit $2^n$ ensemble / broken co-adaptation), the
**inverted-dropout expectation derivation** ($E[h']=p\cdot0+(1-p)\frac{h}{1-p}=h$,
"$1/(1-p)$ is the *unique* constant"), the **train-vs-eval switch called out as
the thing to get right**, where to place it (after activation), and an honest
currency note (CNNs→BatchNorm, Transformers use it lightly). The from-scratch
sanity check *shows the numbers*.

Gaps vs rubric:
1. **The sanity-check slide is the best "code computes a result" moment — make
   sure its three-way output reads cleanly.** `@dropout-implementation-from-scratch-2`
   prints $p=0$ (identity), $p=0.5$ (≈half zeroed, survivors **doubled**:
   confirmed output shows `0,2,4,6,8,10,12,14` and a row with zeros + `22,24,
   28`), $p=1$ (all zero). The deck's fragment bullets already interpret this.
   This is a 2×8 tensor printed three times — **verify it does not overflow**
   (three labelled tensors). If tight, use `output-lines=` or show $p=0.5$ and
   $p=1$ only (the $p=0$ identity is the least informative). The doubling at
   $p=0.5$ is the visual proof of the $1/(1-p)$ rescaling — that row must stay.
2. **Two structural figures, both apt, but check the dropout-placement slide.**
   `dropout2.svg` (the before/after-dropout 5-unit net with $h_2,h_5$ removed)
   anchors View 1 — perfect, it is *the* dropout figure. Then
   `mdl-mlp-arch.svg` anchors "where dropout goes." Different figures, fine.
   Note `dropout2.svg` is a legacy d2l figure (not `mdl-`); it is the canonical
   dropout schematic and matches the chapter body `fig_dropout2`, so keep it —
   but it is the second non-house-style figure in the part (after `mlp.svg`).
   These two legacy schematics are acceptable per the memory note (keep
   canonical reference schematics); the *only* style fix needed is backprop's
   `forward.svg` (deck 3).
3. **The JAX dropout-key slide is correctly `only="jax"`** — Flax needs the
   named `dropout` PRNG key threaded through `apply`, a genuine framing
   difference. Keep. The from-scratch JAX cell has a long teaching comment
   about the fixed-key gotcha; on the *slide* that comment will dominate —
   consider `@-` or trusting that the slide shows the pytorch variant for the
   non-jax decks (the `#@tab` resolver handles this; jax deck shows jax code).

**Core spine (3 ideas):** (a) dropout = inject noise by zeroing units, rescale
survivors by $1/(1-p)$ to stay unbiased; (b) **three lenses, one mechanism** —
thinned net / $2^n$ ensemble / anti-co-adaptation; (c) it is **off at test
time** (the full net runs) — getting that switch wrong is the classic bug.

**Must-keep derivation:** the **inverted-scaling expectation**
$E[h']=h$ and "$1/(1-p)$ is the unique factor" — the rubric's headline item;
already a `.rule` slide. Keep. The $2^n$-masks → cheap-model-averaging argument
is the conceptual core; keep.

**Lead diagram(s):** `img/dropout2.svg` (View 1, the thinned net) and
`img/mdl-mlp-arch.svg` (placement). Keep both.

**Code that earns its place:**
- `@dropout-implementation-from-scratch-1` — the 3-line layer (Bernoulli mask
  from a uniform draw, multiply, rescale). The mechanism in code; keep.
- `@dropout-implementation-from-scratch-2` (**output is the payoff** — see gap
  1): the three-`p` sanity check. *This is the deck's "code computes a result"
  slide.* Keep, watch overflow.
- `@dropout-defining-the-model` — the two-hidden-layer net with dropout gated
  on `self.training` (the train/eval switch *in code*). Keep — it makes "off at
  test time" concrete.
- `@dropout-concise-implementation-1` — `nn.Dropout(p)` stack; keep.
- `@dropout-concise-implementation-2` (**only="jax"**, already scoped) — the
  loss threading the `dropout` key. Keep scoped.
- `@dropout-training` / `-concise-implementation-3` — same-loop training; the
  deck notes "train and val curves track closely." Keep one each.

**Cross-links / forward-points:** to §5.5 (the over-parametrized/double-descent
setup — reuses `mdl-mlp-double-descent.svg` as the motivation hook), to §4
weight decay (Tikhonov/Bishop), forward to BatchNorm and to Transformers
(currency note), and to Monte-Carlo-dropout uncertainty (exercise 5). All
present.

**Slide arc (≈18 slides — keep, apply edits 1 & 3):** cover → motivation
(room to memorize; `double-descent`) → the idea (the Srivastava recipe, frag)
→ divider Why It Works → View 1 thinned net (`dropout2`) → View 2 $2^n$
ensemble (frags) → View 3 co-adaptation/smoothness → the arithmetic (expectation
`.rule`) → divider From Scratch → setup → 3-line layer → sanity check
(`@…-2`, watch overflow) → where it goes (`mdl-mlp-arch`) → the model
(training switch) → train it → divider Concise → `nn.Dropout` → JAX key
(**only=jax**) → train concise → dropout today (currency) → summary.

---

## Deck 7 — `kaggle-house-price.md` (§5.7)

**Depth verdict: GOOD — and the prior "undertrained MLP" defect is ALREADY
FIXED.** The brief flagged that an earlier review found this MLP undertrained;
this version explicitly addresses it: the linear baseline is trained
**competently (100 epochs, lr 0.03)** and the MLP uses the *same* budget.
**Confirmed against the executed PyTorch notebook:** linear K-fold log-mse
`= 0.0325`, MLP `= 0.0281` — the deck's claimed "~0.032 vs ~0.028, MLP a hair
lower" is exact, and the honest caveat ("gradient-boosted trees would still
win on tabular") is stated twice. This is now a model of an honest applied
deck. The pipeline (leakage warning, standardization, one-hot, log-RMSE
rationale, K-fold figure) is complete.

Gaps vs rubric:
1. **This is the most code-heavy deck (it is an applied section) and therefore
   the biggest overflow surface.** Several slides put a multi-line method
   beside a narrow note: the preprocessing method
   (`@-kaggle-house-price-data-preprocessing-2`, already `@-` code-only — good
   call, its output is uninteresting), the K-fold loop
   (`@-…-k-fold-cross-validation-2`, also `@-`), the submission cell
   (`@-…-submitting-predictions-on-kaggle`, also `@-`). The author has already
   used `@-` (code-only) on the three verbose cells — **this is exactly right**
   and shows overflow was considered. Verify the remaining shown-output cells
   (`-accessing-…-2` shape print, `-data-preprocessing-1` head, `-3` shape,
   `-model-selection-linear`, `-mlp-select`) fit; the two K-fold result prints
   are one line each (`average validation log mse = …`) so fine.
2. **The two K-fold *result* numbers are the payoff and should be shown, not
   just asserted.** `@kaggle-house-price-model-selection-linear` and
   `@kaggle-house-price-mlp-select` each inject the `average validation log mse
   = …` print **plus** a 350×250 loss-curve figure. The deck currently shows
   these cells (good). Ensure the *number* (0.0325 / 0.0281) is visible on the
   slide, not cropped to just the curve — that contrast is the entire "does the
   MLP beat the baseline?" payoff. This is the rubric's "cross-validate" code
   earning its place.
3. **Framework-scoping is already correct and load-bearing — preserve it.**
   The baseline-trained-competently slide and the whole MLP arc
   (`KaggleMLP`, `mlp-select`) are **`only="pytorch"`** because the
   from-scratch `k_fold(trainer, data, k, model_fn=…)` signature, the
   `KaggleMLP` class, and the 100-epoch competent-baseline story are
   **pytorch-only** in the source (mxnet/tf/jax use a different
   `k_fold(trainer, data, k, lr)` that builds a `LinearRegression` internally
   and never define an MLP). The deck **already** has paired slides:
   `only="pytorch"` (competent baseline + MLP) and `except="pytorch"`
   (the 10-epoch linear-only baseline). **This split is essential and correct;
   do not collapse it.** It is the cleanest example in the part of *framing*,
   not just code, diverging per framework.
4. **Two non-`mdl-` figures (`kaggle.png`, `house-pricing.png`,
   `kaggle-submit2.png`) are screenshots** — these are reference screenshots
   of the Kaggle UI, not schematic figures, so the "one house style" rule does
   not apply (per the memory note: keep photographic/reference images). Keep.
   The one house-style schematic, `mdl-mlp-kfold.svg`, is used appropriately.

**Core spine (3 ideas):** (a) real ML is **mostly the pipeline** (impute,
standardize-on-train-only, one-hot), not the model; (b) **match the loss to the
metric** — predict $\log$ price, score log-RMSE, because we care about relative
error; (c) **K-fold CV** buys a stable estimate on ~1500 rows and doubles as
the HP search — and an honestly-trained linear baseline is the bar a small MLP
must clear (it does, barely).

**Must-show "derivation"/rationale:** the **log-RMSE justification**
$|\log y-\log\hat y|\le\delta \Rightarrow e^{-\delta}\le \hat y/y\le e^\delta$
(relative error) and the RMSLE formula — already on the "score the logarithm"
slide; keep. Plus the **standardization → mean-imputation-becomes-zero**
coherence point (a nice subtle argument) — keep in the preprocessing note.

**Lead diagram:** `img/mdl-mlp-kfold.svg` (the K=5 fold partition, held-out
fold orange). The one schematic; carries the cross-validation idea. Keep
full-bleed on the K-fold slide. Plus the three Kaggle screenshots for the
competition/submit slides (reference images, keep).

**Code that earns its place (curated; `@-` used well already):**
- `@kaggle-house-price-accessing-and-reading-the-dataset-2` — shape print
  `(1460,81)/(1459,80)`; sets the scale. Keep.
- `@kaggle-house-price-data-preprocessing-1` — the head (numbers + categories +
  Id + SalePrice); motivates preprocessing. Keep.
- `@-kaggle-house-price-data-preprocessing-2` (**code-only**) — the one method
  doing impute/standardize/one-hot. Keep `@-`.
- `@kaggle-house-price-data-preprocessing-3` — shape `(1460,331)`: the "79→331"
  payoff of one-hot. Keep (one-line output).
- `@kaggle-house-price-error-measure` — the loader returning $\log$ price. Keep.
- `@-kaggle-house-price-k-fold-cross-validation-1` / `-2` — the fold slicer and
  loop. `-2` is `@-` (good). Keep.
- `@kaggle-house-price-model-selection-linear` (**only="pytorch"** path) — the
  competent 100-epoch baseline; **shows `log mse = 0.0325`**. Keep, surface the
  number (gap 2).
- `@-kaggle-house-price-mlp-model` (**only="pytorch"**, code-only) — `KaggleMLP`
  (32 units, dropout 0.1, wd 1e-4). Keep `@-`.
- `@kaggle-house-price-mlp-select` (**only="pytorch"**) — same loop, MLP;
  **shows `log mse = 0.0281`**. Keep, surface the number — this is the deck's
  climax (does depth help? a little, honestly).
- `@-kaggle-house-price-submitting-predictions-on-kaggle` (code-only) — the
  ensemble-and-write-CSV. Keep `@-`.

**Cross-links / forward-points:** to §4 weight decay & §5.6 dropout (both used
in `KaggleMLP`), to §5.4 init, to §2 pandas, and a genuine forward-point to
**gradient-boosted trees (XGBoost/LightGBM) winning tabular**
(Grinsztajn 2022, Shwartz-Ziv 2022) — the honest "where nets do *not* shine"
note. All present; the tree caveat is exactly the "forward-point honestly" the
memory file wants.

**Slide arc (≈22 slides — keep; apply edits 2–3, watch overflow per 1):**
cover → motivation (heterogeneous/missing/10×/1500 rows) → divider The
competition → Kaggle in 30s (`kaggle.png`) → competition page
(`house-pricing.png`) → divider Reading & preprocessing → imports+read →
DataModule (frag) → a few rows (`-preprocessing-1`) → three transforms (warn:
train-only) → transforms in code (`@-…-2`) + width 79→331 (`-3`) → divider The
right loss → score the logarithm (RMSLE) → loss in code (`error-measure`) →
divider K-fold → K-fold idea (`mdl-mlp-kfold`) → K-fold in code (`-1`,
`@-…-2`) → divider Model selection → competent baseline (**only=pytorch**,
show 0.0325) / OR underfit-baseline (**except=pytorch**) → can a small MLP do
better? (**only=pytorch**, `@-mlp-model`) → same loop, MLP (**only=pytorch**,
show 0.0281) → submit: ensemble + CSV (`@-…submitting`, `kaggle-submit2.png`)
→ the general recipe → recap.

---

## Consolidated action list (what a regeneration pass should actually do)

These decks do **not** need rebuilding. The bounded, concrete work:

1. **[Figure — the one real defect]** Replace `backprop`'s legacy
   `img/forward.svg` with a new house-style **`mdl-mlp-forward-graph`**
   (forward-only computational graph matching `mdl-mlp-backprop-graph.svg`).
   Add via the `mdl-figure` skill + `tools/gen_mdl_backprop_figures.py`.
   Fallback: reuse `mdl-mlp-backprop-graph.svg` and cut `forward.svg`.
2. **[Content — highest-value add, needs author + re-run]** Consider adding the
   **variance-vs-depth demo** (N(0,1)/Xavier/He, with and without ReLU) to
   `numerical-stability` (the chapter's own exercise 4). Turns the variance
   derivation from *asserted* to *shown*. Requires a new source `#id` cell +
   notebook re-run + capture — flag for the author; out of scope for slide
   regeneration alone.
3. **[Overflow sweep]** Run the 720px sweep on the dense slides specifically:
   `backprop` "Backward through our network" (5 equations); `mlp` activation
   cheat-sheet table; `mlp-implementation` parameter/concise class slides;
   `dropout` 3-way sanity check; `kaggle` head/result slides. Fix by `@-` /
   split / `output-lines=`, not scrollbars.
4. **[Show the numbers]** On `kaggle` ensure the two `log mse` prints
   (0.0325 / 0.0281) are visible (not cropped to the curve) — the baseline-vs-
   MLP contrast is the payoff. On `mlp-implementation` consider surfacing the
   ≈0.87 val-acc scalar.
5. **[Preserve scoping]** Do not collapse the existing `only=`/`except=` splits:
   `mlp` XOR (only=pytorch); `mlp-implementation` concise jax split; `dropout`
   jax-key (only=jax); `kaggle` competent-baseline + MLP arc (only=pytorch) vs
   linear-only (except=pytorch). All are *framing* divergences, correctly
   scoped.
6. **[Small curations]** `mlp` — add the XOR construction matrices to the
   verify slide. `mlp-implementation` — drop the 2nd adjacent `mdl-mlp-arch`
   use. `dropout` — `@-` the long JAX from-scratch comment if it dominates the
   slide.
7. **[Verify per-framework injection]** The `numerical-stability` exploding-
   matrix slide and any shown-output cell: confirm all four frameworks inject
   (the pytorch output is confirmed in the committed `outputs/` store).

**What NOT to do:** do not add toy double-descent or grokking runs to
`generalization-deep` (survey section, code-free by design); do not add autodiff
code to `backprop` (derivation-driven by design); do not redraw the canonical
legacy schematics `mlp.svg` / `dropout2.svg` / the Kaggle screenshots (per the
house-style scope note: keep canonical reference/photographic images).
