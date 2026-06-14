# §23 Calculus — slide-deck content plan / depth review

**Scope.** The four §23 ("Mathematics for Deep Learning" → Calculus) decks, source
`.md` in `chapter_mdl-calculus/`:

- `mdl-single-variable-calculus.md`
- `mdl-multivariable-calculus.md`
- `mdl-matrix-calculus-autodiff.md`
- `mdl-integral-calculus.md`

**Method.** Read each chapter's prose, every `#id` code fence, and the existing
`<!-- slides -->` block; cross-read the §2 reference decks
(`chapter_preliminaries/calculus.md`, `linear-algebra.md`, `ndarray.md`) as the
bar, and `docs/slides-northstar-design.md` / `docs/slides.md`; spot-checked the
committed `outputs/pytorch/chapter_mdl-calculus/*.json` store for the cells the
slides show.

---

## TL;DR — the headline finding

**The four decks are already at the north-star bar.** Each already exists as a
full north-star `<!-- slides -->` block (22–24 slides), is **picture-first**
(each leads with an `mdl-cal-*.svg` house-style figure), stages its key
derivation with fragments, shows **only code that computes/verifies something**
(every shown compute cell prints a result that lands the point — and those
outputs are present and correct in the committed store), and **cross-links
downstream** (gradient descent, backprop, Newton, normalizing flows,
expectations). This is the rewrite the task wanted to *plan*; it has, by all
appearances, **already landed** (the source `.md` files were last touched
2026-06-13, after the "technically shallow" complaint).

So this document is **not** a regenerate-from-scratch plan. It is:

1. a **depth verdict per deck** (all **good**), recording the spine, the
   must-show derivation, the figures, and the earns-its-place code **so a
   regeneration preserves them rather than regressing**, and
2. a short list of **concrete, real defects to fix on any regeneration** — the
   one substantive one being **inconsistent cover kickers / section numbers**
   across the four decks.

If the intent is genuinely to *re-run the generator*, treat each deck's existing
block as the **target**, not a source of ideas to replace, and apply the
cross-deck fixes in the last section.

---

## Cross-deck observations (top 3)

1. **Inconsistent cover kickers and section numbers — the one real defect.**
   The four cover slides disagree on both the deck-title prefix and the number:
   - single-variable: `Dive into Deep Learning · §22.1`  ← **stale number**
   - multivariable:   `Dive into Deep Learning · Appendix · Calculus`  ← no number
   - matrix/autodiff: `Math for Deep Learning · §23 · Calculus`  ← different prefix
   - integral:        `Dive into Deep Learning · §23.4`
   Per `CLAUDE.md` (current status) the chapter is **§23** (Calculus), the third
   of the six MDL chapters. **Fix:** make all four read one consistent prefix +
   `§23.1 … §23.4`, e.g. `Dive into Deep Learning · §23.1 · Calculus` …
   `§23.4`. This is a one-line edit per deck inside the `.cover` div; it is the
   single highest-value change here. (Pick the canonical prefix with the author;
   the §2 reference decks use `Dive into Deep Learning · §2.x`, which argues for
   that form.)

2. **Outputs are correct in the store but the `_notebooks/` scratch copy is
   empty — a build-hygiene gate, not a content gap.** The compute cells the
   decks show (`…-tape-check`, `…-softmax-jacobian`, `…-dual-eval`,
   `…-jacobian-finite-diff`, the integral `@!…` plots/prints, etc.) carry their
   results in `outputs/<fw>/chapter_mdl-calculus/*.json` (verified: e.g. *"softmax
   Jacobian matches autograd: True"*, *"our tape … torch …"* matching, dual ==
   exact). But `_notebooks/pytorch/chapter_mdl-calculus/*.ipynb` currently has
   **0 outputs** on those cells, and `inject_outputs.py slides` matches by cell id
   against `_notebooks/<fw>/`. **Before regenerating/ rendering, repopulate
   `_notebooks/` from the store** (regen + inject, or re-execute the CPU
   notebooks) or the rendered decks will show code with **no result** — which
   would gut the "matches autograd" payoff that makes the autodiff deck land.
   This is a render-pipeline step, not a slide-source edit. (Note also the
   memory item *capture clobbers other frameworks*: scope `--frameworks` if you
   re-run a single fw.)

3. **The four decks form one deliberate through-line; keep the hand-offs
   intact.** local linear model (deck 1) → the gradient as that model in many
   dimensions + the chain rule as a sum over paths (deck 2) → the chain rule as
   **Jacobian composition** and why reverse-mode is cheap (deck 3) → integration
   as the *other* limit, ending at **expectation = integral** and Monte Carlo
   (deck 4). The forward-points are explicit and correct (deck 2's "*why* this is
   reverse-mode autodiff … is the matrix-calculus section" hands directly to deck
   3; deck 1's descent lemma and Newton set up deck 2's Hessian test). A
   regeneration must **preserve these cross-references**, not just the within-deck
   content. Minor: decks 1, 3 use the diagram-engine `@fig:mdl-cal-…` form for a
   few figures (resolved from `img/auto/`, all present and OK), while most
   figures use the plain `![](../img/mdl-cal-…svg)` form (resolved from `img/`);
   both work, but the mix is worth normalizing for consistency if touched.

---

## Deck 1 — `mdl-single-variable-calculus.md`

**Depth verdict: GOOD (already at the bar).** 24 slides, four divider-fenced
sections, cover + recap bookends. No rubric gaps. If anything it is *long*; the
risk on regeneration is dropping content, not adding it.

**Core spine.** The **small-change identity** $f(x+\epsilon)\approx
f(x)+\epsilon f'(x)$ is the one engine, worn four ways: (1) the derivative
itself (zoom → line → secant → tangent), (2) the **gradient-descent step**
(choose $\epsilon=-\eta f'$), (3) **curvature** (second derivative, MVT, Newton,
Taylor + remainder rate), (4) **corners** (subgradients, why SGD shrugs).

**Must-show derivation/theorem (already staged).**
- The **descent lemma**, derived from the FTC + an $L$-Lipschitz slope:
  $$f(x-\eta f'(x)) \le f(x) - \eta\bigl(1-\tfrac{L\eta}{2}\bigr)[f'(x)]^2,$$
  strict drop for $0<\eta<2/L$, best at $\eta=1/L$ — and the **quadratic-ceiling**
  picture that proves it. (This is the technical high point and is present.)
- The **Lagrange remainder** $R_n=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$,
  with the $n=0$ case = the Mean Value Theorem, made *quantitative* by the
  error-rate code below.

**Lead diagrams (all exist, all house-style).** `mdl-cal-zoom-sequence` (cover +
"every smooth curve is a line"), `mdl-cal-secant-to-tangent`, `mdl-cal-gd-step`,
`mdl-cal-descent-lemma`, `mdl-cal-pos/neg/zero-second` (the three curvature
signs, side by side), `mdl-cal-mvt`, `mdl-cal-taylor-quadratic`,
`mdl-cal-smooth-not-analytic`, `mdl-cal-relu-corner`. No new figure needed.

**Code that earns its place (all compute a result).**
- `single-variable-calculus-differential-calculus-4` — secant slope → 8 as
  $\epsilon$ shrinks; paired (`. . .`) with
  `single-variable-calculus-autograd-check` (autograd returns 8 exactly).
- `single-variable-calculus-gradient-descent` — five step sizes → five regimes
  (creep/one-shot/zig-zag/bounce/diverge); the table + the trajectory plot.
- `single-variable-calculus-taylor-error-rate` — halve the window, error ÷
  $2^{n+1}$; measured 4.1/8.2/16.3 vs predicted 4/8/16 (the remainder made
  empirical — a standout slide).
- `single-variable-calculus-one-sided` — the $|x|$ one-sided quotients never
  agree (the corner, numerically).
- output-only plots: `…-linear-approximation` (tangents on $\sin$),
  `…-taylor-series` ($e^x$ Taylor degrees).
  All four frameworks are `#@tab`-tabbed code/output swaps; **no `only=`/`except=`
  scoping needed** (concept identical across fw; autograd-check differs only in
  API and is already per-fw tabbed).

**Cross-links / forward-points (present).** GD step ⇒ §sec_gd; Newton + Hessian
⇒ deck 2 + §optimization; chain rule run reverse ⇒ backprop; subgradients ⇒
convex-analysis chapter; "autograd computes it exactly" ⇒ deck 3 / `sec_autograd`.

**Slide arc (existing, keep):** cover → "which way is downhill?" → DIV 01
Derivative → smooth-is-a-line → secant→tangent → watch the slope settle (+
autograd) → small-change identity → DIV 02 → tangent line → GD step → descent
lemma (picture) → five regimes → DIV 03 Curvature → second-derivative sign test
→ MVT → best parabola→Newton → Taylor $P_n$ → each derivative buys a power
(error-rate) → smooth≠analytic → DIV 04 → corners/subgradients → split in the
quotient → why SGD shrugs → recap.

---

## Deck 2 — `mdl-multivariable-calculus.md`

**Depth verdict: GOOD (already at the bar).** 24 slides, four sections. Hits
every §23 multivariable anchor.

**Core spine.** The **gradient is the derivative in many dimensions**
($\boldsymbol\epsilon\cdot\nabla L$ is the first-order change) → its **geometry**
(directional derivative, steepest descent via Cauchy–Schwarz, $\nabla\perp$ level
sets, tangent plane, Lagrange) → the **multivariate chain rule as a sum over
paths**, run backward = backprop → the **Hessian** (best quadratic, Clairaut
symmetry, second-derivative test by definiteness, the saddle argument).

**Must-show derivations (already staged).**
- **Steepest ascent from the directional derivative:** along unit $\mathbf v$,
  change $=\|\nabla L\|\cos\theta$, so by Cauchy–Schwarz $-\nabla L$ is steepest
  descent. (The rubric's required derivation — present.)
- **Chain rule = sum over directed paths** of products of edge derivatives;
  forward sweep gives one input's derivative, **backward sweep reuses everything**
  for all inputs = backpropagation.
- **Second-order Taylor** $f\approx f_0+\nabla f\cdot\delta+\tfrac12
  \delta^\top\mathbf H\delta$; **saddle prevalence**: a min needs all $n$
  eigenvalues $>0$, "coin-flip" signs ⇒ prob $2^{-n}$ → high-D critical points
  are overwhelmingly saddles.

**Lead diagrams (all exist).** `mdl-cal-gradient-field` (cover + steepest),
`mdl-cal-tangent-plane` (two faces of the linearization), `mdl-cal-chain-net1`
and `mdl-cal-chain-net2` (the dependency graphs; net2 shows the skip edge ⇒
LSTM/residual gradient flow), `mdl-cal-taylor-quadratic` (best quadratic surface),
and it **reuses `mdl-la-psd`** for the definiteness/eigenvalue test (a nice
linear-algebra cross-link). No new figure needed.

**Code that earns its place.**
- `multivariable-calculus-higher-dimensional-differentiation` — first-order
  prediction vs true small-step value, agree to several digits.
- `multivariable-calculus-the-backpropagation-algorithm-1/2/3` — forward sweep
  (one derivative) vs backward sweep (all derivatives) vs `f.backward()` matching
  the by-hand sweep. (Three cells that *are* the backprop story.)
- `multivariable-calculus-hessians` — the $f$ − Taylor-quadratic gap is **third
  order** (double the step, 8× the gap).
- output-only: `…-a-note-on-mathematical-optimization` (critical points of a
  quartic). All code/output swaps; **no `only=`/`except=` needed.**

**Cross-links / forward-points (present).** Lagrange ⇒ KKT/duality; backward
sweep ⇒ deck 3 (vector–Jacobian products) + §backprop; Hessian test ⇒ §gd;
"every optimizer reads the local Taylor expansion" recap line ties GD / momentum
/ Adam / Newton together.

**Slide arc (existing, keep):** cover → motivation → DIV 01 → gradient is the
derivative → does the linear approx hold? → directional derivative (every
direction at once) → DIV 02 Geometry → steepest (Cauchy–Schwarz) → tangent plane
/ contours → critical points → Lagrange → DIV 03 Chain rule → compositions are
graphs → sum over paths → forward sweep → backward sweep → what the framework
runs → DIV 04 Hessian → curvature/Clairaut → best quadratic → hold the quadratic
to account → second-derivative test (psd) → saddles not bad minima → recap.

---

## Deck 3 — `mdl-matrix-calculus-autodiff.md`  ★ the deepest, most ML-relevant

**Depth verdict: GOOD — arguably the strongest of the four.** 23 slides, five
sections. This is the autodiff deck and it is excellent: it derives both AD modes
from associativity, *builds* both in a few dozen lines, and lands the
cheap-gradient principle.

**Core spine.** The **Jacobian** is the best local linear map (scalar map ⇒
gradient, Hessian ⇒ Jacobian of the gradient field) → the **chain rule is
Jacobian multiplication**, $\mathbf J=\mathbf J_L\cdots\mathbf J_1$ → matrix
multiply is **associative**, so bracket-order *is* forward- vs reverse-mode →
**forward mode** (dual numbers, JVP, one column/pass, cheap when tall) vs
**reverse mode** (the tape, VJP, one row/pass, cheap when wide) → a scalar loss
is maximally wide ⇒ **backprop is reverse-mode AD**, full gradient at 2–4× one
forward pass.

**Must-show derivations / identities (already staged).**
- **The cost argument (the rubric's required "why reverse mode"):**
  associativity ⇒ right-to-left = forward (per input), left-to-right = reverse
  (per output); a scalar loss = single-row Jacobian ⇒ **one** backward sweep =
  whole gradient, independent of $n$. The **cheap-gradient principle** is stated
  as a `.rule` callout.
- **Dual numbers:** adjoin $\varepsilon\ne0,\ \varepsilon^2=0$; $(a+b\varepsilon)
  (c+d\varepsilon)=ac+(ad+bc)\varepsilon$ **is** the product rule; running on
  $x+1\cdot\varepsilon$ yields $f(x)+f'(x)\varepsilon$.
- **Layer factor structure:** $\mathbf J=\operatorname{diag}(\varphi_L')\mathbf
  W_L\cdots\operatorname{diag}(\varphi_1')\mathbf W_1$ — dense weights interleaved
  with cheap diagonal activation masks (ReLU mask = 0/1 ⇒ $O(n)$ backward).
- **softmax∘cross-entropy collapse to $\mathbf p-\mathbf y$** (Jacobian
  $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$, verified vs autograd).
- **The tape is a diamond, not a chain:** a reused value's adjoint arrives twice
  ⇒ accumulate with `+=` (chain rule's "sum over paths").

**Lead diagrams (all exist).** `mdl-cal-jacobian-ellipse` (cover + "vectors in,
vectors out": a circle → ellipse under $\mathbf J$, leftover bend = $o(\|\delta\|)$),
`mdl-cal-fwd-vs-rev` (forward JVP vs reverse VJP — used on both the forward-mode
and reverse-mode "cost" slides), `mdl-cal-tape-dag` (the diamond tape). No new
figure needed.

**Code that earns its place — every cell prints a verification.**
- `matrix-calculus-autodiff-jacobian-finite-diff` — halve $\delta$, error ÷4
  (the $o(\|\delta\|)$ fingerprint); store shows `error/|delta| ≈ 1.6e-3`.
- `matrix-calculus-autodiff-softmax-jacobian` — store shows *"softmax Jacobian
  matches autograd: True"* and `d loss/dz == p − y`.
- `matrix-calculus-autodiff-dual-eval` — **15-line** forward-mode AD; store shows
  `dual 3.360101 vs exact 3.360101`.
- `matrix-calculus-autodiff-tape-check` — **~30-line** reverse-mode AD; store
  shows `our tape … torch …` agreeing (`dy/du=16, dy/dv=-16`).
  The finite-diff and dual-eval cells are **all-framework, plain-Python (no
  `#@tab`)**; softmax-Jacobian and tape-check are **`#@tab`-tabbed per framework**
  (they call `requires_grad`/`backward` etc.).
  **PyTorch-only scoping note:** the two tabbed compute cells already resolve to
  the deck's own framework variant via `#@tab`, so no `@id@pytorch` force is
  needed. **But** if a particular framework's variant is missing or its autograd
  API diverges enough that the slide *prose* ("Check against the framework's own
  autograd") reads oddly, the clean fix is to scope those two slides
  `only="pytorch"` (or force `@…-tape-check@pytorch` / `@…-softmax-jacobian@pytorch`
  so every deck shows the PyTorch reference). **Verify all four
  `outputs/<fw>/…json` carry these two cells before generating**; the PyTorch
  store does (confirmed). This is the only place in the four decks where
  per-framework scoping is even a question.

**Cross-links / forward-points (present).** Hessian = Jacobian of $\nabla f$
(deck 2); HVP via **forward-over-reverse** ⇒ Newton/CG scaling; "never form the
Jacobian — compose `jvp`/`vjp`" ⇒ practical autograd; cheap-gradient principle ⇒
the 1986 backprop history in deck 1.

**Slide arc (existing, keep):** cover → from a slope to a matrix → DIV 01 Jacobian
→ vectors in/out (ellipse) → three faces (scalar/vector/Hessian) → numerical
signature (finite-diff) → DIV 02 → composition multiplies Jacobians → two shapes
(dense/elementwise) → associativity → DIV 03 Identities → four identities → the
softmax∘CE cancellation → DIV 04 Forward mode → dual algebra → 15-line dual AD →
one pass per input (JVP) → DIV 05 Reverse mode → why reverse is the right cost
model (VJP) → the tape diamond → 30 lines reproduce `loss.backward()` → never
form the Jacobian (HVP) → recap.

---

## Deck 4 — `mdl-integral-calculus.md`

**Depth verdict: GOOD (already at the bar; the reworked figures lead the spine).**
22 slides, four sections. The author flagged the integral figures as previously
poor and reworked; the deck now **leads each idea with the improved figure**
(`mdl-cal-riemann`, `mdl-cal-sub-area`, `mdl-cal-rect-trans`,
`mdl-cal-cov-jacobian`, `mdl-cal-sum-order`, plus `@fig:mdl-cal-bell-surface`).
No rubric gaps.

**Core spine.** Integral = **accumulation / signed area**, a limit of Riemann
sums ($\sum\to\int$) → the **Fundamental Theorem, both halves** (area-so-far
$F'=f$; and $\int_a^b f=G(b)-G(a)$), plus by-parts → **change of variables** (1-D
stretch $du/dx$; $n$-D Jacobian determinant $|\det D\phi|$) → **integration meets
probability** (density $\int p=1$, expectation $\int g\,p$, Monte Carlo vs the
curse of dimensionality).

**Must-show derivations (already staged).**
- **FTC, both directions** — the sliver argument for $F'=f$, then antiderivative
  evaluation $\int_a^b f=G(b)-G(a)$, **verified numerically** by differencing a
  cumulative Riemann sum (telescoping → float roundoff only).
- **The Gaussian integral via the polar trick** — square it, Fubini-fuse the
  copies, polar coordinates contribute Jacobian $r$:
  $(\int e^{-x^2}dx)^2=\iint e^{-x^2-y^2}=\int_0^\infty\!\int_0^{2\pi} r
  e^{-r^2}=\pi$, so $\int e^{-x^2}dx=\sqrt\pi$. (Ties change-of-variables,
  Fubini, and the density normalizer together — the deck's high point.)
- **∫ vs Σ / curse of dimensionality** — a grid to resolution $\epsilon$ in $d$
  dims costs $\epsilon^{-d}$ and decays $N^{-2/d}$; Monte Carlo gives $N^{-1/2}$
  in every dimension. (The rubric's required ∫-vs-expectation link, made
  quantitative.)

**Lead diagrams (all exist; the reworked set).** `mdl-cal-riemann` (cover + the
definite integral as a limit), `mdl-cal-sub-area` (FTC sliver), `mdl-cal-rect-trans`
(1-D substitution stretch), `mdl-cal-cov-jacobian` (the volume-scaling
determinant), `mdl-cal-sum-order` (Fubini grid), `@fig:mdl-cal-bell-surface`
(the $e^{-x^2-y^2}$ bell — present in **both** `img/` and `img/auto/`, so the
`@fig:` form resolves). No new figure needed. *(Normalize the lone `@fig:` ref to
the `![](../img/…)` form, or vice-versa, if you want one figure-reference style.)*

**Code that earns its place — all output-only (`@!`) compute/verify cells.**
- `integral-riemann-converge` — left-rule sum → $\tfrac12\log 5$, first-order
  (÷10 spacing ⇒ ÷10 error).
- `integral-ftc-check` — finite-difference of the cumulative sum vs $f$ (error =
  float roundoff, telescoping).
- `integral-improper` — partials $1-e^{-b}$ vs left-Riemann; two independent
  knobs (truncation vs discretization).
- `integral-box-volume` — box sum over $[-2,2]^2$ (the $\operatorname{erf}$
  hides; whole plane → $\pi$).
- `integral-gaussian` — $\int e^{-x^2}=\sqrt\pi$ numerically.
- `integral-density` — $p=\tfrac1{\sqrt\pi}e^{-x^2}$ integrates to 1, mean 0.
- `integral-monte-carlo` — MC vs grid, $N^{-1/2}$ vs $N^{-2/d}$.
  `box-volume` and `monte-carlo` are **all-framework, plain-Python (no `#@tab`)**;
  the rest are `#@tab`-tabbed. **No `only=`/`except=` scoping needed.**

**Cross-links / forward-points (present).** By-parts ⇒ score matching's
Hyvärinen identity; change-of-variables / $-\log|\det D\phi|$ ⇒ **normalizing
flows**; density + expectation ⇒ the probability chapter; the FTC integral form
is the same one used to prove deck 1's descent lemma (nice back-reference).

**Slide arc (existing, keep):** cover → why integration → DIV 01 → definite
integral as a limit → watching it converge → FTC ($F'=f$) → integration is
differentiation reversed (numeric check) → improper integrals → two errors two
knobs → integration by parts → DIV 02 Change of variables → 1-D substitution →
$n$-D Jacobian determinant → DIV 03 Higher dimensions → multiple integrals (bell)
→ Fubini → box volume numerically → the Gaussian integral → DIV 04 Probability →
densities & expectations → Monte Carlo beats the curse → recap.

---

## What to do on regeneration (checklist)

1. **Treat the existing `<!-- slides -->` blocks as the target.** They meet the
   §8 acceptance checklist and the depth rubric; regenerating from a thinner
   outline would regress them. Preserve spine, derivations, figures, cross-links.
2. **Fix the cover kickers / section numbers** to one consistent prefix +
   `§23.1 … §23.4` (the single substantive content fix; see cross-deck obs #1).
3. **Repopulate `_notebooks/<fw>/chapter_mdl-calculus/` from the committed store
   before rendering** so `inject_outputs.py` has outputs to inject (cross-deck
   obs #2). The store is correct; the scratch notebooks are stale.
4. **Matrix-calculus deck only:** confirm all four `outputs/<fw>/…json` carry
   `…-tape-check` and `…-softmax-jacobian`; if any framework's autograd variant
   is missing/awkward, scope those two slides `only="pytorch"` (or force
   `@…@pytorch`). No other deck needs framework scoping.
5. **Optional polish:** normalize the figure-reference style (mixed `@fig:` vs
   `![](../img/…)`); run the 720 px overflow sweep on deck 1 (the longest) after
   render.
6. **No new figures are required for any deck.** Every figure the spine needs
   already exists in the house style.
