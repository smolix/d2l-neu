# §24 Optimization — slide-deck content plan (depth audit + targeted polish)

Planning doc for the four reveal.js decks of the **§24 Optimization** part of the
"Mathematics for Deep Learning" appendix. Read-and-plan only: no `.md`/`.qmd`
edits, no slide generation, no builds. Scope:

- `chapter_mdl-optimization/mdl-gradient-based-optimization.md`
- `chapter_mdl-optimization/mdl-convexity.md`
- `chapter_mdl-optimization/mdl-constrained-optimization-duality.md`
- `chapter_mdl-optimization/mdl-numerical-stability-conditioning.md`

The bar: `docs/slides/north-star.html` and the §2 math decks
(`chapter_preliminaries/calculus.md`, `linear-algebra.md`); governed by
`docs/slides-northstar-design.md` (§3 style, §8 checklist) and `docs/slides.md`.

---

## Headline finding (read this first)

**All four `<!-- slides -->` blocks are already written to the north-star bar.**
The source `.md` files were all touched on 2026-06-13 (the day this plan was
written), and each block already uses the full vocabulary the §2 reference decks
established: `.cover`, `.divider`/`.dnum`/`.dtitle`/`.dsub`, `.kicker`,
`.cols .vc` + `.fig`/`.fig .big`/`.narrow`, `.d2l-note`/`.d2l-note .rule`/`.warn`,
`. . .` fragments, `@-`/`@!` code/output controls, two-column recap closers, and
geometric-figure-first slides. They lead with the geometry, state the theorem
that matters with a staged proof-sketch, run code that **computes a real result**
shown on the slide, cross-link each other, and carry honest gotchas. This is not
the "restated-API, code-with-no-result, pasted-caption" shallowness the rubric
targets — that shallowness is **absent** here.

**So the brief inverts.** This is not a rebuild plan; it is a *depth-confirmation
audit with narrow, high-value polish*. Regenerating these decks from scratch
risks regressing work that is already at the bar. The right move is: keep the
blocks, apply the small targeted changes below, then `make -j4 slides` and run the
§8 overflow sweep.

### Evidence the decks are real and accurate, not stale
- **All 11 `img/mdl-opt-*.svg` figures exist on disk; all 11 are referenced; zero
  unused, zero dangling.** (`mdl-opt-gd-bowl-vs-valley`, `-momentum-damping`,
  `-sgd-noise-ball`, `-convex-vs-nonconvex-set`, `-chord-above-graph`,
  `-local-equals-global`, `-lagrange-tangency`, `-kkt-active-set`,
  `-primal-dual-gap`, `-conditioning-ellipse`, `-fp-number-line`.)
- **The committed `outputs/pytorch/chapter_mdl-optimization/*.json` store holds
  real executed outputs, captured under `torch==2.11.0` (the pinned version),
  and they match the numbers woven into the slide captions to the digit.** Spot
  check (convexity): store has `worst chord violation -2.46e-03`,
  `Hessian eigenvalues [6.273 18.6557]`, `E[exp(X)]=1.6466 vs 0.9998`,
  `271 runs -> -1.0575 / 229 runs -> 0.9304`, `gap ratios [...0.25 0.25 0.25]` —
  every one appears verbatim in the deck prose. The decks are faithful to
  executed results.
- **The `_notebooks/.../*.ipynb` scratch copies currently show no cell outputs.**
  That is the expected not-yet-injected state: the slide build runs
  `inject_outputs.py slides`, which populates outputs from the committed
  `outputs/` store by cell id at render time. **Not a defect; do not treat the
  empty notebooks as "stale."** (If a reviewer wants to *see* outputs in the
  notebook, run `python tools/inject_outputs.py slides --framework pytorch`.)

### Framework scoping — settled, no pytorch-only problem anywhere
The chapters are deliberately **framework-free plain NumPy** for every *teaching*
cell; only the imports cell is `#@tab`-split. Confirmed from `#@tab` counts:
gradient-based-optimization and duality split **only imports** (1 cell × 4 fw);
convexity has **zero** `#@tab` (pure NumPy). So three of the four decks need
**zero `only=`/`except=` scoping**, and that is exactly what they have. **No
pytorch-only cell needs `only="pytorch"`** in any deck — the code is identical
across frameworks. The lone exception is conditioning (below), which *already*
scopes its genuinely-divergent slides correctly.

---

## Cross-deck observations (top 3)

1. **The §24 "must-hit" technical anchors are essentially all present** — steepest
   descent as a theorem (Cauchy–Schwarz), the learning-rate trade-off shown both
   ways (too small = glacial, too large = diverges) with the κ story, Jensen +
   picture + 2-line proof, the three lenses with the first/second-order
   conditions *derived* and connected, local=global with the picture and the
   chord-ambush intuition, the Lagrangian + all four KKT conditions +
   complementary slackness, the tangency picture, weak/strong duality + Slater,
   the condition number / error-amplification / preconditioning / fp pitfalls
   story, and the conditioning↔convergence tie ("κ: one number, two
   consequences"). The cross-links convexity↔GD↔conditioning↔duality are wired in
   both directions. **One anchor is under-served: the weight-decay / norm-ball
   constrained view of ℓ2 (cross-link §3.7).** See duality deck, gap D1.

2. **One stylistic inconsistency, in exactly one deck.** Conditioning embeds its
   two figures with raw pandoc image lines —
   `![](../img/mdl-opt-fp-number-line.svg){width=100%}` and
   `...mdl-opt-conditioning-ellipse.svg...` — instead of the `@fig:` inline
   mechanism the other three decks (and the §2 reference decks) use. The `@fig:`
   path inlines the SVG into a `{=html}` raw block so it inherits the deck's
   loaded fonts (Source Sans 3 / JetBrains Mono); a bare `<img>` falls back to
   generic fonts (`docs/slides.md` §5.4 / §"Diagrams"). The three figure-led
   decks read crisper for this reason. **Caveat:** `@fig:<id>` resolves to
   `img/auto/<id>.svg`, but these optimization figures live in **`img/`** (they
   are `gen_mdl_*figures.py` book figures, not `diagrams/` engine SVGs). So
   converting to `@fig:` is **not** a drop-in — it needs either (a) a copy/symlink
   of the two SVGs into `img/auto/`, or (b) leaving the raw-`<img>` form as-is
   (functional, just lower font fidelity). Flag for the author to decide; do **not**
   silently move files. This is polish, not a correctness bug.

3. **Density / overflow is the main risk, not shallowness.** Several slides are
   close to full: they pair a multi-line display equation, a `.d2l-note .rule`,
   and a code cell whose output is multi-row (the κ-sweep "six rows", the SVM
   dual, the Hilbert table). The §2 decks keep one idea per slide and lean harder
   on the figure column; a few §24 slides carry a theorem statement *and* a code
   cell *and* a callout. Before sign-off, run the `docs/slides.md` overflow sweep
   (`scrollHeight > Reveal height`) on all four decks in all four frameworks and
   trim the offenders (make a verbose cell `@-`, or split). Specific suspects are
   listed per deck. This is the §8 checklist item most likely to fail.

---

## Deck 1 — `mdl-gradient-based-optimization.md`

**Depth verdict: GOOD (already north-star).** This is arguably the strongest of
the four and a model for the others. It opens with the bowl-vs-valley figure,
poses the why/how-fast/what-breaks-it triad, threads the condition number κ
through the whole deck, and ends with a through-line recap. No rebuild.

Gaps vs rubric (minor):
- The "six rows" η-sweep slide and the momentum-race slide each pair prose +
  a wide code/output cell — **overflow-watch** (trim intro lines if the sweep
  output wraps). 
- The valley picture appears **twice** (cover/motivation `@fig:...gd-bowl-vs-valley`
  *and* the "valley picture" slide). That is acceptable as a bookend callback (the
  §2 decks reuse the rank-ladder similarly), but if space is tight, the second
  instance can drop to `.fig` from `.fig .big`.

**Core spine (already realized):**
1. *Where to step, how far to trust it* — steepest descent is a theorem;
   L-smoothness erects a quadratic ceiling → the descent lemma.
2. *The condition number κ* — on a quadratic GD decouples into per-mode factors
   `(1-ηλ_i)`; stability `η<2/L`, optimal step contracts by `(κ-1)/(κ+1)`, cost
   linear in κ.
3. *Making it practical* — momentum (damped oscillator, √κ law), SGD (unbiased,
   1/b variance, noise ball, decay), and "why not Newton" (κ-immune but O(d³)).

**Must-show theorem + proof-sketch (already staged):** the **descent lemma**,
`f(x_{k+1}) ≤ f(x_k) − η(1−Lη/2)‖∇f‖²`, with the "gain grows linearly, curvature
tax grows quadratically → break-even at 2/L, best at 1/L" reading on a `. . .`
fragment. Keep. The **steepest-descent proposition** (Cauchy–Schwarz, unique
minimizer `−∇f/‖∇f‖`) leads. Both are theorems, not definitions — exactly the
rubric's ask.

**Lead diagrams (reused, correct):** `img/mdl-opt-gd-bowl-vs-valley.svg` (cover +
valley slide), `img/mdl-opt-momentum-damping.svg`, `img/mdl-opt-sgd-noise-ball.svg`.
No new figure needed.

**Code that earns its place (all compute a shown result):**
`gradient-based-optimization-steepest-direction` (3600-direction scan lands on
`−∇f/‖∇f‖`), `-backtracking` (fixed step diverges, Armijo accepts 0.031 then
doubles), `-eta-sweep` (the six-row stability table), `-momentum` (GD 6908 vs
heavy-ball 315 at κ=1000), `-sgd-schedule` (fixed-step floor vs 1/k decay,
log-log plot), `-newton` (one-step kill + doubling digits). Framework deltas:
**none** beyond imports — no `only=` needed (confirmed; the deck uses none).

**Cross-links / forward-points (present):** convexity (the two convex-rate
theorems are *stated here, proved next*), conditioning (κ returns as a numerical
villain), and forward to the main-book optimizer chapters (`sec_momentum`,
`sec_adam`, `sec_minibatch_sgd`). Keep.

**Slide arc (current, 22 slides — sound; only trim for overflow):**
cover → motivation (why/how-fast/what + bowl-valley fig) → DIVIDER 01 → steepest
descent is a theorem → every negative slope descends (code) → L-smoothness ceiling
+ descent lemma (frag) → the one guarantee deep nets keep (frag) → backtracking
(code) → DIVIDER 02 → GD decouples into modes → the six-row η-sweep (code) →
valley picture (fig) → convexity upgrades stationarity (rule cards) → DIVIDER 03 →
momentum is a damped oscillator (fig) → inertia buys √κ (code) → minibatch noise
1/b → noise ball (fig) → decay reaches optimum (code) → why not Newton (code+warn)
→ recap. **Action: none required; overflow-sweep the three code-dense slides.**

---

## Deck 2 — `mdl-convexity.md`

**Depth verdict: GOOD (already north-star).** Leads each act with the right
picture, states Jensen and local=global as theorems with the picture first and a
genuine 2-line proof-sketch staged on a fragment, and closes with an honest
reality-check (non-convexity by construction → PL → implicit bias). Hits every
convexity anchor in the brief. No rebuild.

Gaps vs rubric (minor):
- **The second-order lens / PSD-Hessian condition is stated but compressed.** The
  "Three lenses, one verdict" slide shows `∇²f ⪰ 0` and the practical "pick the
  cheapest lens" reading, which is good. The brief asks to *derive/connect* the
  first- and second-order conditions; the chapter body has the full equivalence
  proof (chord ⇔ first-order ⇔ Hessian), but the deck does not stage even the key
  step (first-order ⇐ second-order via Taylor + nonnegative remainder). **Optional
  enrichment:** add one `. . .` fragment to the "Three lenses, one verdict" slide
  giving the one-line bridge ("along any segment `g''(t)=v^T∇²f v ≥ 0`, so Taylor
  drops the tangent below the graph") — turns a *stated* equivalence into a
  *shown* one. Only if it fits 720 px; otherwise leave.
- The "Log-sum-exp Hessian is a covariance" pair (statement slide + sampling
  slide) is dense — **overflow-watch**.

**Core spine (already realized):**
1. *Convex sets* — chords stay inside; **intersection is the factory** (simplex,
   PSD cone, polyhedra).
2. *Three lenses* — chord / tangent-under-estimator / PSD Hessian, plus
   subgradients for kinks; strong convexity adds the κ=L/μ floor.
3. *Jensen* — chord lifted to expectations → KL≥0, AM–GM, ELBO gap.
4. *Payoff* — **every local min is global**, stationary ⇒ optimal; local steps →
   global rates O(1/k) and (1−μ/L)^k.
5. *Recognizing + limits* — the convexity calculus; lse = softmax covariance;
   deep nets non-convex but PL keeps the rate, implicit bias picks the minimum.

**Must-show theorems + proof-sketches (already staged):**
- **Jensen** `f(E[X]) ≤ E[f(X)]` with the subgradient-at-the-mean + linearity
  proof on a fragment, and the "spreading X out pushes E[f(X)] up" mnemonic.
  Keep — this is the rubric's exemplar.
- **Local minima are global** (picture first via `mdl-opt-local-equals-global`,
  then the chord-ambush argument on a fragment). Keep — this is *the* key
  proof-sketch the brief names.
- (Background, well-handled:) the smooth-convex O(1/k) and strongly-convex
  (1−μ/L)^k rates appear as a two-equation slide; the full telescoping proofs
  live in the body. Fine for slides.

**Lead diagrams (reused, correct):** `img/mdl-opt-convex-vs-nonconvex-set.svg`,
`img/mdl-opt-chord-above-graph.svg` (the two-lens figure — leads the lenses act),
`img/mdl-opt-local-equals-global.svg`. No new figure needed.

**Code that earns its place (all compute a shown result):**
`convexity-three-lenses` (1000 chords + 1000 tangents both clean, Hessian eigs
positive), `convexity-jensen-mc` (√e gap, AM≥GM, KL≥0), `convexity-basins` (500
starts: bowl → one point at machine-ε, double well → 271/229 split),
`convexity-lse-hessian` (eigs incl. one numerical zero, covariance match to
6e-4), `convexity-pl-rate` (PL μ≈0.1755, min f''=−4, gap ratios → 0.25). Framework
deltas: **none** (zero `#@tab`). No `only=` needed.

**Cross-links / forward-points (present):** GD (the descent lemma it chains; "the
second half of the proof never used convexity" → PL), conditioning (lse shift
invariance ↔ stable softmax; κ=L/μ), duality (the convex conjugate bridges to the
dual function and to info-theory variational forms), info theory (KL≥0, ELBO).
Keep.

**Slide arc (current, 26 slides — sound):** cover → why convexity (motivation) →
DIVIDER 01 sets → chords-stay-inside (fig) → new sets from old (intersection
factory) → DIVIDER 02 lenses → chord+first-order (fig, frag) → three lenses one
verdict → subgradients → checked numerically (code) → DIVIDER 03 Jensen →
Jensen+2-line proof (frag) → one inequality three classics (code) → DIVIDER 04
payoff → local=global (fig, frag) → dividing line experiment (code) → local steps
→ global rates → DIVIDER 05 recognizing → calculus of convex fns (frag) → lse is a
covariance → confirmed by sampling (code) → DIVIDER 06 reality check →
non-convex by construction → PL survives (code) → implicit bias → recap.
**Action: optional 2nd-order-lens fragment (above); overflow-sweep the lse pair.**

---

## Deck 3 — `mdl-constrained-optimization-duality.md`

**Depth verdict: GOOD (north-star), with ONE genuine content gap.** Leads with
the Lagrange-tangency figure, builds the "no feasible descent direction" principle
into Lagrange → KKT → projections → duality, has the KKT active-set figure and the
primal-dual-gap figure, and grounds it in three runnable duals (SVM, water-filling,
visible gap). The author flagged the *prior* KKT picture as "too dense"; the
current deck leads the KKT act with `mdl-opt-kkt-active-set` and keeps the
active-set / cone story clean, so that concern is addressed by the existing spine.

### Gap D1 (the one real §24-anchor miss): weight decay ↔ the ℓ2 norm-ball
The brief explicitly wants "the connection to weight decay (the norm-ball /
constrained view of ℓ2), cross-link §3.7." The **chapter body develops this
precisely** — the intro names "the regularization-as-constraint equivalence behind
weight decay" (line 7), and the Discussions state it exactly: *the penalty form
`min L + λ‖w‖²` and the constraint form `min L s.t. ‖w‖² ≤ r²` are linked by the
Lagrangian; the weight-decay coefficient of `sec_weight_decay` (§3.7) is the
multiplier of the norm constraint, and sweeping one traces the other* (lines
1027–1031). The **shadow-price proposition** (body §"Multipliers Are Shadow
Prices", `λ* = −∂p*/∂u`, `eq_mdl-opt-shadow-price`) is the exact machine. **But no
slide surfaces this.** The deck's only weight-decay-adjacent content is the
passing "gradient clipping projects onto a ball / max-norm caps" note on the
projections slide.

**Recommendation (add ONE slide), placed in the Duality act right after
"Strong duality and shadow prices":**

> `::: {.slide title="Weight decay is a norm constraint"}` `[Duality at work]{.kicker}`
> Two-column. **Left (prose + the two equivalent problems):** penalty form
> `min_w L(w) + λ‖w‖²` ⇔ constraint form `min_w L(w) s.t. ‖w‖² ≤ r²`, linked by
> the Lagrangian; λ **is** the multiplier of the norm constraint, and the
> shadow-price reading `λ = −∂p*/∂r²` says λ prices the budget — sweeping λ traces
> the regularization path. Forward-point: this is the weight decay of §3.7
> (`sec_weight_decay`), seen from the constraint side. **Right (`.fig`):** the
> ridge/lasso geometry figure (below). A `.d2l-note` can note the ℓ1 corner →
> sparsity contrast.

**Lead figure for D1 — reuse, do NOT draw:** `img/mdl-linreg-ridge-geometry.svg`
already exists and is *exactly* this picture — elliptical squared-loss contours
centred on `ŵ`, meeting the **ℓ2 ball tangentially off-axis (ridge)** and the
**ℓ1 diamond at a corner (lasso, → sparsity)**. Generated by
`tools/gen_mdl_linreg_figures.py::fig_ridge_geometry`, house style, currently
unreferenced by any optimization deck. It is a *linreg* figure id, so embed it the
same way conditioning embeds its figures — a raw `![](../img/mdl-linreg-ridge-geometry.svg){width=100%}`
line — **unless** D2 (below) standardizes on `@fig:` + `img/auto/`, in which case
copy it there too. No new figure is needed; flag only the reuse.

(Optional, smaller: the projections slide could add half a line — "the ℓ2 ball
projection *is* weight decay's constraint set" — to plant the idea before the new
slide pays it off. Not required if the slide above is added.)

### Other notes (minor)
- **Overflow-watch:** "Four conditions, one workhorse" (the 4-line KKT `.rule`
  card + complementary-slackness prose + a `. . .` convex-sufficiency line) is
  dense; the SVM-dual slide pairs a max-QP display + update rule + output. Sweep.
- Three of the worked-dual slides use `@!` output-only (SVM dual, water-filling,
  duality gap) — correct (`@!constrained-...`), the computed result is the point
  and the code would overflow. Keep.

**Core spine (already realized):** (1) no feasible descent → Lagrange/tangency;
(2) inequalities → KKT + active set + complementary slackness; (3) projections
make it an algorithm (PGD, simplex = sparsemax); (4) duality — bounds for free,
Slater closes the gap, multipliers are shadow prices; worked: SVM dual,
water-filling, a visible gap. **+ add the weight-decay slide (D1) in act 4.**

**Must-show theorem + proof-sketch (already present):** the **KKT conditions**
(all four, as a `.rule` card) with the geometry "at the optimum `−∇f` is a
nonnegative combination of the active constraint gradients (in their cone)" — the
brief's exact ask — led by `mdl-opt-kkt-active-set`. The **shadow-price**
proposition is the proof-sketch to lean on for the new D1 slide.

**Lead diagrams (reused, correct):** `img/mdl-opt-lagrange-tangency.svg` (cover +
Lagrange act), `img/mdl-opt-kkt-active-set.svg`, `img/mdl-opt-primal-dual-gap.svg`.
**+ reuse `img/mdl-linreg-ridge-geometry.svg` for the new D1 slide.**

**Code that earns its place:** `constrained-simplex-projection` (3 coords → exactly
zero, KKT residuals at machine precision), `constrained-svm-dual` (4 nonzero α =
support vectors at margin 1, primal=dual at 1e-16), `constrained-water-filling`
(common water level, dry high-noise channels), `constrained-duality-gap`
(p*=−1/4, d*=−1/2, a real gap). Framework deltas: **none** beyond imports. No
`only=`.

**Cross-links / forward-points (present + the new one):** convexity (KKT
sufficiency, Slater, the conjugate), GD (stationarity, descent lemma, PGD inherits
GD's guarantees), conditioning (its consequences continue there). **Add the §3.7
weight-decay forward-link via D1.**

**Slide arc (current 18 → 19 with D1):** cover → one idea three guises (fig) →
DIVIDER 01 → no-feasible-descent tangency (fig) → Lagrangian packages it (frag) →
DIVIDER 02 → active/inactive + cone (fig) → four KKT conditions (frag) → DIVIDER
03 → project-then-algorithm → simplex projection (code) → DIVIDER 04 → dual
function bounds-for-free (fig) → strong duality + shadow prices →
**[INSERT: weight decay is a norm constraint (fig: ridge-geometry)]** → SVM dual
(code) → water-filling (fig) → a duality gap you can see (fig) → recap. **Action:
add D1; overflow-sweep KKT + SVM slides.**

---

## Deck 4 — `mdl-numerical-stability-conditioning.md`

**Depth verdict: ADEQUATE→GOOD (north-star content; one consistency fix + one
small enrichment).** Content is strong and genuinely deep: it splits the blame
correctly into backward error vs conditioning, has the brilliant
framework-divergent cross-entropy-failure slides (the *one* place in §24 where
framing truly diverges, scoped exactly right), and ties conditioning back to GD
convergence with the memorable "κ: one number, two consequences — ridge pays
twice (accurate solves *and* fast gradient descent)." That tie-back is precisely
the brief's anchor for this deck. No rebuild.

### Gap N1 (consistency, the deck the rubric's "picture-first" flag fits least cleanly)
This is the one deck whose figures are embedded as **raw pandoc image lines**, not
`@fig:`:
- `![](../img/mdl-opt-fp-number-line.svg){width=100%}` (motivation + floating-point
  slides), and
- `![](../img/mdl-opt-conditioning-ellipse.svg){width=100%}` (backward/forward-error
  slide + ridge slide).

Functionally fine and the figures *are* present (my first `@fig:`-only grep
**missed** them — they are there). But the other three decks and the §2 reference
decks use `@fig:`, which inlines the SVG so it inherits deck fonts; a raw `<img>`
falls back to generic fonts (`docs/slides.md` §5.4). **Recommendation:** for
visual parity, either (a) the author copies/symlinks these two `img/*.svg` into
`img/auto/` and the deck switches to `@fig:mdl-opt-fp-number-line` /
`@fig:mdl-opt-conditioning-ellipse`, or (b) accept the raw-`<img>` form as a known
minor font-fidelity delta. **Do not move files unilaterally** — `@fig:` resolves
to `img/auto/`, and these are `gen_mdl_*` book figures living in `img/`. Flag for
a decision. (Same mechanism question applies to the duality D1 figure.)

### Gap N2 (small enrichment): make the conditioning↔convergence tie picture-first
The "Ridge regularization is preconditioning" slide already reuses
`mdl-opt-conditioning-ellipse.svg` and states "the elongated valley rounds into a
bowl … the same λ pays twice." That is the exact cross-link to §24.1's
bowl-vs-valley. **Optional enrichment:** add a one-line explicit pointer to
`:numref:fig_mdl-opt-gd-bowl-vs-valley` ("this is the §24.1 valley, now rounded")
or, if a tighter visual callback is wanted, a `.cols` putting the conditioning
ellipse beside a thumbnail reference to the GD valley. Low priority; the verbal
tie is already strong.

### Other notes
- **Overflow risk is highest in this deck.** Many slides carry a display equation
  + a `.d2l-note .rule`/`.warn` + a code/output cell (floating-point formats,
  stable softmax, log1p, Welford, Hilbert, normal equations, ridge). The
  cross-entropy-failure slides are **split four ways by framework already**, which
  helps. Run the overflow sweep in **all four** frameworks; the float-format
  (`-finfo`) and cross-entropy (`-cross-entropy`) cells differ per framework and
  may overflow in some but not others.
- The Welford slide is correctly `except="mxnet"` / `only="mxnet"` (MXNet's naive
  variance comes out *negative* — a different punchline). Keep — good framing
  divergence.

**Core spine (already realized):** (1) floating point is a gappy number system
(relative precision, doubling gaps, overflow cliffs); (2) stable softmax — subtract
the max, log-sum-exp is exact, cross-entropy from logits (with the 4-framework
failure gallery); (3) catastrophic cancellation — reformulate not add bits (log1p,
Welford); (4) conditioning — forward ≤ κ × backward error; normal equations square
κ; **ridge lowers κ and that is preconditioning, the §24.1 tie**.

**Must-show theorem/derivation (already present):** the **condition-number
error-amplification bound** `‖x̂−x‖/‖x̂‖ ≤ κ(A)·ε` with the rule-of-thumb "correct
digits ≈ format digits − log₁₀κ," demonstrated on Hilbert matrices (backward error
stays at the 1e-16 floor — the *matrix*, not the algorithm, amplifies) and the
`κ(AᵀA)=κ(A)²` normal-equations result. This is the deck's core derivation and it
is shown, not asserted.

**Lead diagrams:** `img/mdl-opt-fp-number-line.svg`,
`img/mdl-opt-conditioning-ellipse.svg` (reused on two slides). Both present.
**No new figure needed**; see N1 re: embedding mechanism.

**Code that earns its place:** `-finfo` (three formats' eps/range), `-spacing`
(overflow points 88.7 / 11.1), `-stable-softmax`, `-logsumexp` (logits near 1000
effortless in log space), `-cross-entropy` (the 4-fw gallery), `-log1p`, `-welford`
(naive off by hundreds in float64; MXNet variant → negative variance), `-hilbert`
(digits fall, backward error blameless), `-normal-equations` (5 extra digits lost),
`-ridge` (κ → 1). Framework deltas: **genuinely divergent** — `-finfo` and
`-cross-entropy` differ per framework, and the deck **correctly** scopes the
cross-entropy punchline with `only="pytorch,mxnet"` / `only="jax"` /
`only="tensorflow"` and the Welford slide with `except/only="mxnet"`. This is the
model for when `only=` is warranted. No further scoping needed.

**Cross-links / forward-points (present):** convexity (lse convex / shift
invariance), GD/§24.1 (κ governs convergence; ridge = preconditioning = rounding
the valley), and the main book (mixed-precision loss scaling = underflow
management; BatchNorm running moments = Welford; `lstsq` solves by QR/SVD). Keep.

**Slide arc (current 23 slides — sound):** cover → math-right-loss-NaN
(fig) → DIVIDER 01 floating point → number-system-with-gaps (fig) → three formats
(code) → where the cliffs are (code) → DIVIDER 02 softmax → subtract the max
(frag+code) → log-sum-exp identity (code) → pass logits not probabilities (code) →
four-framework failure gallery (×3 `only=` slides) → DIVIDER 03 cancellation →
subtraction annihilates digits (code) → Welford (×2 mxnet-scoped, code) → DIVIDER
04 conditioning → backward/forward error (fig) → Hilbert κ eats digits (code) →
normal equations square κ (code) → ridge is preconditioning (fig, code) → recap.
**Action: decide N1 (`@fig:` vs raw img); optional N2; overflow-sweep heavily in
all four fw.**

---

## Consolidated action list (for the regeneration/polish pass)

Priority order. None of these is a rebuild.

1. **DUALITY — add the weight-decay slide (D1).** One new `.slide` in the Duality
   act, two-column, reusing `img/mdl-linreg-ridge-geometry.svg`, stating the
   penalty⇔constraint equivalence and λ = the norm-constraint multiplier =
   shadow price, forward-linked to §3.7 `sec_weight_decay`. This closes the one
   genuine §24-anchor gap. (Spec + figure named above.)
2. **CONDITIONING + DUALITY-D1 — resolve the figure-embedding mechanism (N1).**
   Decide raw `![](../img/…svg)` vs `@fig:` + a copy into `img/auto/`. If `@fig:`
   is chosen, it applies to `mdl-opt-fp-number-line`, `mdl-opt-conditioning-ellipse`,
   and `mdl-linreg-ridge-geometry`. Author decision; do not move files in this
   pass.
3. **ALL FOUR — overflow sweep, all four frameworks.** Run the `docs/slides.md`
   sweep; trim the dense suspects called out per deck (κ-sweep, SVM-QP, KKT card,
   lse pair, the conditioning code-dense slides). Fix by `@-` / split / shorter
   intro, never per-slide scrollbars.
4. **CONVEXITY — optional 2nd-order-lens fragment** (one `. . .` line giving the
   Taylor bridge first-order ⇐ second-order), only if it fits 720 px.
5. **CONDITIONING — optional explicit `:numref:` callback** from the ridge slide
   to `fig_mdl-opt-gd-bowl-vs-valley` (N2).
6. **Build & verify:** after edits, `make -j4 slides` then `tools/audit_slides.py`;
   confirm injection from the committed `outputs/` store populates the `@`/`@!`
   cells (the store already holds the matching executed outputs under torch
   2.11.0). No notebook re-execution is required — outputs are captured.

## What NOT to do
- Do **not** regenerate the blocks from scratch — they are at the bar; a rebuild
  risks regressing accurate, figure-led work.
- Do **not** add `only="pytorch"` anywhere — the teaching code is framework-free
  NumPy; only conditioning has genuine per-framework framing, already scoped.
- Do **not** treat the empty `_notebooks/*.ipynb` outputs as stale — they inject
  from the committed `outputs/` store at render time.
- Do **not** draw a new figure — every needed picture exists in `img/`
  (the 11 `mdl-opt-*` plus `mdl-linreg-ridge-geometry` for D1).
