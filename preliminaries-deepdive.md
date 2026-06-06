# Preliminaries chapter — content/pedagogy deepdive

An in-depth editorial review of the **Preliminaries** chapter (the "survival
skills" on-ramp), along the five axes requested: **clarity**, **teaching /
pedagogical value**, **coverage** (missing topics), **irrelevant content**, and
**missing references**. This is a *content* review — the prose, exposition,
examples, exercises, and structure — not the code review (that's `deepdive.md`).

**Sections reviewed** (reading order): `index` → `ndarray` (Data Manipulation)
→ `pandas` (Data Preprocessing) → `linear-algebra` → `calculus` → `autograd`
(Automatic Differentiation) → `probability` (Probability & Statistics) →
`lookup-api` (Documentation).

**Method.** linear-algebra and probability were read in full by hand;
ndarray/pandas, calculus/autograd, and lookup-api/index were close-read in
parallel; a whole-chapter cross-cutting pass checked ordering, coverage, and the
preview/depth split against the *Mathematics for Deep Learning* (MDL) part. The
headline structural and clarity claims were verified directly against the source
(noted inline). Findings carry `file:line` and a concrete suggestion.

**Framing.** This chapter is, by design, a fast on-ramp that *previews* math the
MDL part develops in full (`index.md:44`). So "missing depth X" is a finding only
when X is needed downstream or its absence misleads — not when it's correctly
deferred. The chapter is genuinely good; most findings are polish, and the few
structural ones are cheap to fix.

---

## TL;DR — the five things worth deciding on first

1. **The calculus section never lands its own thesis in prose.** Its stated
   purpose is "just enough calculus to determine which direction to adjust each
   parameter to decrease the loss." But the words *steepest ascent / descent* and
   $-\nabla f$ **appear only in the slide deck** (calculus.md:470,649,653,750),
   never in the body text (verified: nothing before the `<!-- slides -->` marker
   at line 451). The body defines the gradient mechanically and stops. **Add two
   sentences** to the body: $\nabla f$ points in the direction of steepest
   increase, so optimizers step along $-\nabla f$. This is the single highest-
   value content fix in the chapter.
2. **The preview→depth contract with the MDL part is never wired up in-text.**
   The chapter promises (index.md:44) that the MDL part "develops in full" what
   these sections "only preview," yet **not one** of linear-algebra / calculus /
   probability contains a `:numref:` forward-pointer to its MDL counterpart.
   linear-algebra even sends the eager reader to *external* books (Strang, Kolter,
   Petersen — linear-algebra.md:1107) while ignoring the in-book chapter built for
   exactly this. **Add a one-line "for the full treatment, see …" pointer** to
   each math section's Discussion. Cheap, and it closes the loop the chapter
   advertises.
3. **The chapter makes a GPU promise it never delivers or cross-references.**
   ndarray.md:30 sells GPU acceleration as the tensor's killer feature, but the
   chapter never shows `.to(device)`/device placement and (verified) has **no
   `:numref:` to the builders-guide GPU section**. Either show 3 lines or add a
   forward-pointer so the promise is visibly cashed elsewhere.
4. **Section quality is uneven, and ndarray is the weak link.** ndarray has **2
   exercises** (both "change one symbol" busywork) and a one-paragraph summary;
   pandas has **7** graded ones, linear-algebra/probability 10+ (verified: 2 vs
   7). ndarray also skips two on-ramp topics it implicitly promises — **dtype/
   precision** (`float32` appears unexplained in its first code cell) and **device
   placement**. lookup-api has **zero** exercises despite being the one section
   whose entire point is a *practiced habit*.
5. **In-text citations are absent exactly where a historical/empirical claim is
   made.** Corrected counts (verified, incl. `:citet:`): autograd 4, probability
   4, linear-algebra 3, but **calculus 0, ndarray 0, pandas 0**. calculus opens
   with an uncited Archimedes anecdote and never cites Newton/Leibniz or a text;
   ndarray/pandas name NumPy/pandas/broadcasting with no in-text cite (the
   `index.md` Further Reading list partly compensates). Two autograd citations
   are also mis-stated (see Missing references).

The chapter works as an on-ramp: the seven-skill framing (index.md:4–24) is
convincing, the ordering is logical, the figures are pre-generated SVGs (house
style), and pandas, autograd, and the probability worked-examples are genuinely
strong teaching.

---

## 1. Clarity

**Strong overall.** The prose is clean and the hard disambiguations are handled
well (e.g., linear-algebra's *order* vs *dimensionality* note, ndarray's
zero-based vs one-based indexing caution). Specific issues:

- **calculus.md:329–334** `[high]` — A Jacobian / denominator-layout aside
  ("$\nabla_{\mathbf{x}}\mathbf{f}$ denotes the transpose of the Jacobian (the
  denominator-layout convention)") is dropped on a reader who has not yet met the
  word *Jacobian* (first defined in autograd.md, the *next* section). Notation
  before definition. *Suggest:* cut to the MDL matrix-calculus chapter; the
  scalar-output gradient is all this section needs.
- **calculus.md:373–377** `[med]` — The multivariate chain rule jumps to
  "$\nabla_{\mathbf{x}} y = \mathbf{A}\nabla_{\mathbf{u}} y$, where $\mathbf{A}$ …
  contains the derivative of $\mathbf{u}$ w.r.t. $\mathbf{x}$" but never writes
  $A_{ij}=\partial u_j/\partial x_i$, so the reader can't see the scalar sum
  collapse into the matrix form. *Suggest:* state the entries in one line.
- **probability.md:554–558, 599** `[med]` — `P(X,Y) = P(X)P(Y)` is used as the
  running example of "notation shorthand," but that equation is the *independence*
  statement, introduced ~150 lines before independence is defined
  (probability.md:744). A reader can come away thinking the factorization always
  holds. *Suggest:* use a non-independent shorthand (e.g. $P(X,Y)$ meaning "for
  all $i,j$, $P(X{=}i, Y{=}j)$") as the notation example; keep the factorization
  for the independence section.
- **autograd.md:434–437** `[med]` — The detach payoff is hard to parse: "taking
  the gradient of `z = x * u` will yield `u`, (not `3*x*x` as you might have
  expected since `z = x*x*x`)" forces the reader to juggle three facts at once.
  *Suggest:* split into two sentences and show $\partial z/\partial x = u = x^2$.
- **linear-algebra.md:763–764** `[med]` — "multiplication with a matrix … as a
  transformation that **projects** vectors from $\mathbb{R}^n$ to $\mathbb{R}^m$."
  *Projection* has a specific meaning (idempotent, $P^2=P$); a general linear map
  isn't one. In a linear-algebra section this word choice is imprecise. *Suggest:*
  "maps"/"transforms" rather than "projects."
- **Terminology drift — *order* vs *rank*** `[low, cross-cutting]` — The prose
  standardizes on *order* (number of axes; linear-algebra.md:200–206), but the
  slides use *rank* (linear-algebra.md:1182, ndarray slides). A reader meeting
  "rank" elsewhere won't connect them, and *rank* already means something else in
  linear algebra (column rank). *Suggest:* note the synonym once, or keep *order*
  consistently.
- **ndarray.md:856–857** `[low]` — "invoke the `item` function or Python's
  built-in functions" — "built-in functions" is vague until the code shows
  `float(a)`. Name them.
- **probability.md:805** `[low]` — Section titled just "An Example" — vague.
  *Suggest:* "Worked Example: HIV Testing."

## 2. Teaching / pedagogical value

**Many highlights.** pandas threads one house-price example through inspect →
split → impute → encode → scale → tensorize and motivates *why* at each step
(leakage, false ordering, ill-conditioning) — the best teaching in the chapter.
autograd's "many inputs, one scalar loss → reverse mode" motivation
(autograd.md:776–791) and its batch-loss justification for `sum().backward()`
(323–347) are exactly right. probability's symmetry argument for a fair coin,
the explaining-away discussion, and the two-test HIV Bayes worked example (with
natural-frequencies figure) are excellent. Issues:

- **calculus body — the gradient→descent payoff is missing from prose** `[high]`
  (TL;DR #1, verified). The Discussion (calculus.md:386–409) foreshadows backprop
  *before* ever stating which direction reduces loss. *Suggest:* lead the
  Discussion with the steepest-descent statement (mirror the slide Recap ordering,
  which is better), then the backprop foreshadowing.
- **ndarray exercises are busywork** `[high]` (verified: 2, both trivial). They
  cover only comparison and broadcasting; nothing on reshape, reductions,
  indexing, memory, or conversion. *Suggest:* add 3–4 load-bearing ones — a
  reshape/`-1` task, a reduction-along-axis shape-prediction task, a
  broadcasting-*failure* case, and a memory one (`X[:] =` keeps `id`, `X = X +`
  doesn't).
- **lookup-api has no exercises** `[med]` — the one section that's a *habit*
  needs them most. *Suggest:* 2–3 "use `dir` to find softmax → read its signature
  with `?` → verify on a length-3 vector" drills.
- **calculus exercises over-index on proving rules from limits** `[med]`
  (calculus.md:411–433) — that's MDL-level rigor; there's no exercise that takes a
  gradient *step* or relates a gradient to a descent direction, the one idea the
  section most wants to reinforce. *Suggest:* add "for $f(\mathbf{x})=\|\mathbf{x}\|^2$,
  compute $\mathbf{x}-\eta\nabla f$ and show it moves toward the minimum."
- **ndarray random init under-motivated** `[low]` (ndarray.md:294–303) — says
  params are "initialized randomly" but never *why not zeros* (symmetry breaking),
  though the slide does. Add the clause.
- **Missing inter-section bridges** `[low]` — section *intros* motivate
  themselves well, but hand-offs are thin: only autograd back-references calculus.
  A one-sentence bridge (calculus → "we now differentiate the linear-algebra
  objects from the last section") would knit the on-ramp together.

## 3. Coverage gaps

The math sections are appropriately scoped (linear-algebra correctly defers
inverse / rank / eigendecomposition to MDL; calculus stays minimal). Genuine
gaps, ranked:

- **GPU / device placement — promised, never delivered or pointed to** `[high]`
  (TL;DR #3, verified). *Suggest:* show `.to(device)` briefly, or add a
  `:numref:` to the builders-guide GPU section at the ndarray intro.
- **dtype / numerical precision — absent as a topic** `[high]` — `float32`
  appears unexplained in ndarray's first code cell (ndarray.md:144) and pandas
  explicitly casts to float32 "to save memory" (pandas.md:208), but no section
  explains default dtypes or why float32 dominates DL. *Suggest:* a short "Data
  Types" paragraph in ndarray (`.dtype`/`.astype`, float32-vs-float64).
- **Numerical stability — absent entirely** `[med]` — no mention anywhere of
  overflow/underflow or the log-sum-exp trick, despite softmax-adjacent ground
  coming soon and probability doing heavy floating-point sampling. A two-line note
  or forward-pointer would serve the on-ramp.
- **Reproducibility / seeding — absent** `[med]` — probability samples
  extensively (coin tosses, multinomial draws) with no `manual_seed`/`PRNGKey`
  reproducibility note; JAX's explicit-key model is even *shown* but never framed
  as the reproducibility lesson it naturally is. *Suggest:* a one-paragraph "set a
  seed" note in ndarray or probability.
- **Named distributions + MLE — not introduced, used immediately downstream**
  `[med]` — probability never names the Gaussian/normal or Bernoulli as objects
  (Gaussian appears only in passing at probability.md:1021), and MLE is alluded to
  ("estimator/consistency") but never shown. Both are needed within a chapter or
  two (Gaussian noise in linear regression; Bernoulli/softmax in classification).
  MDL's probability-statistics chapter covers them — so a **pointer** suffices,
  but right now there's no breadcrumb.
- **Broadcasting rule stated, not shown** `[med]` (ndarray.md:609–666) — the
  happy path is shown but the alignment rule (compare shapes right-to-left; each
  axis equal or 1) and a *failure* case are not, though the slide has the rule.
  Readers hit broadcast errors within the first hour. *Suggest:* state the rule +
  one non-broadcastable example/exercise.
- **reshape returns a view, not a copy** `[low]` (ndarray.md:199–245) — never
  stated in prose (the slide says it); a real footgun once readers mutate. One
  sentence tying it to the Saving-Memory section.
- **Notation pointer — missing** `[low]` — the chapter introduces scalar/vector/
  matrix/tensor and probability notation but never points to a notation appendix.
  A single `:numref:` near the linear-algebra intro.
- **Information theory / entropy** `[low]` — correctly omitted (now in MDL), but
  add it to the same MDL forward-pointer fix so there's a breadcrumb.

## 4. Irrelevant / cuttable / should-move content

**KEEP**
- **calculus.md:192–280 "Visualization Utilities"** `[high]` (flagged
  independently by two reviewers) — ~65 lines of matplotlib plumbing
  (`use_svg_display`, `set_figsize`, `set_axes`, a 25-line generic `plot` whose
  own comment admits "Much of the code here is just ensuring that the sizes and
  shapes of inputs match"). It teaches matplotlib, not calculus, and directly
  violates the project's "Code teaches; it does not draw" rule. It's also the only
  inline-matplotlib in an otherwise figure-driven chapter (mixes styles).
  *Suggest:* move the `#@save` definitions into the `d2l` library / a utilities
  page; keep only the one teaching cell (the tangent-line plot at calculus.md:272–278,
  which genuinely shows "slope = derivative").
**CUT TO MDL**
- **calculus.md:329–334** (Jacobian/denominator-layout aside) — MDL material
  parachuted into the on-ramp (also under Clarity). Cut to MDL.
**CUT AND ADD FORWARD POINTER**
- **ndarray.md:761–796 (TensorFlow `tf.function` digression)** `[med]` — a
  ~10-line decorated function teaching graph pruning / allocation reuse is more TF
  memory semantics than a "survival skills" reader needs. *Suggest:* trim to a
  2-line note or forward-pointer. (Per-framework, so low blast radius.)
**KEEP**
- **probability.md — depth that overshoots "preview"** `[low, judgment call]` —
  this is the heaviest section (~1,130 lines). The Kolmogorov-axioms treatment
  (448–528) is fine and well-pitched, but the aleatoric/epistemic discussion and
  Chebyshev's inequality in the Discussion (1052–1127) read like MDL depth.
  Defensible to keep (they're well written), but they're the most trimmable-to-a-
  pointer content if you want the section closer to the others' altitude.
**KEEP**
- **linear-algebra.md:340–341** `[low]` — $\nabla_{\mathbf{X}}\|\mathbf{X}\|_F^2 =
  2\mathbf{X}$ for a matrix argument is never used in the section and pushes past
  "just enough." Minor; defensible to keep as it pairs with the vector identity.

Everything else earns its place. pandas, autograd, and linear-algebra's core all
stay at the right altitude.

## 5. Missing references

Corrected counts (verified, includes `:citet:`): **autograd 4, probability 4,
linear-algebra 3; calculus 0, ndarray 0, pandas 0, lookup-api/index in-text 0**
(index has a curated Further-Reading list). The gaps:

- **The structural fix (highest value):** add in-text `:numref:` pointers from
  each math Discussion to its **in-book MDL counterpart** (Linear Algebra,
  Calculus, Probability & Statistics). linear-algebra.md:1104–1107 currently cites
  Strang/Kolter/Petersen but not chapter_mdl-linear-algebra — the very chapter
  index.md:44 advertises as the depth path. This is the reference change that
  closes the preview/depth loop. (verified: no MDL `:numref:` in any of the three.)
- **calculus is the only zero-cite math section** `[med]` — it opens with the
  Archimedes circle-area/exhaustion anecdote (calculus.md:9–24) with no citation
  and never cites Newton/Leibniz or a standard text (Spivak/Apostol, or the
  in-list MML-book). *Suggest:* a closing "to go deeper" cite mirroring
  linear-algebra's pattern, plus the MDL forward-pointer.
- **autograd.md:37–39 — backprop attribution is potentially mis-stated** `[med]`
  — "The core ideas behind modern backpropagation date to a PhD thesis from 1980
  :cite:`Speelpenning.1980`." Speelpenning is reverse-mode *AD*; a reader reads
  "backpropagation" as the *neural-net* algorithm, whose standard attribution
  (Rumelhart–Hinton–Williams 1986; Linnainmaa 1970 for reverse-mode AD) is absent.
  *Suggest:* either soften to "reverse-mode automatic differentiation," or add the
  Rumelhart/Linnainmaa cites where the NN-backprop claim is made.
- **autograd.md:42–43 — Julia citation overstated** `[med]` — "the Julia
  programming language employs forward propagation :cite:`Revels.Lubin.Papamarkou.2016`."
  That paper is ForwardDiff.jl (one forward-mode package); Julia's flagship
  (Zygote.jl) is reverse-mode. *Suggest:* "Julia's ForwardDiff.jl implements
  forward-mode AD :cite:`Revels...`", which is what the citation supports.
- **ndarray/pandas first-mention cites** `[low]` — NumPy (ndarray.md:20–27),
  broadcasting as a named mechanism (ndarray.md:609), and pandas (pandas.md:18)
  are introduced without in-text cites though the sources sit in index.md's
  Further Reading (NumPy Nature paper, McKinney). *Suggest:* add the in-text cite
  at first mention, per house style.
- **autograd.md:372–373** `[low]` — a Medium blog post is cited as an inline
  hyperlink (link-rot risk, inconsistent with the chapter's `:cite:` style).
  *Suggest:* replace with the PyTorch docs on the `gradient` argument.
- **probability — LLN/CLT** `[low]` — named theorems stated without citation
  (probability.md:337–343); acceptable given how canonical they are, but a cite
  would match the section's otherwise good referencing.

---

## Per-section scorecard

| Section | Clarity | Teaching | Coverage | Notable issue |
|---|---|---|---|---|
| **ndarray** | good | **weak exercises** | dtype + device gaps | 2 trivial exercises; GPU promise unmet; broadcasting rule unshown |
| **pandas** | strong | **excellent** | well-scoped | leakage caveat is a highlight; no in-text cites |
| **linear-algebra** | strong | strong | well-scoped (defers to MDL) | "projects" imprecision; no MDL pointer; order/rank drift |
| **calculus** | good w/ leaks | **thesis missing from prose** | minimal (right) | steepest-descent slides-only; viz plumbing; 0 cites; Archimedes uncited |
| **autograd** | mostly good | **excellent** | complete for on-ramp | detach passage hard; 2 mis-stated cites |
| **probability** | strong | **excellent examples** | heaviest; no distributions/MLE pointer | independence-as-shorthand confusion; seeding absent; "An Example" title |
| **lookup-api** | good | good, current | — | **no exercises**; MXNet docs pinned to **1.8.0** (repo runs MXNet 2.0) |
| **index** | convincing framing | — | — | strong Further Reading list; lacks in-text MDL pointers from sections |

## Suggested triage order

**Tier 1 — cheap, high-value (do first):**
1. Add the steepest-ascent/$-\nabla f$ statement to the **calculus body**
   (TL;DR #1). Two sentences; it's the section's whole point.
2. Add `:numref:` **MDL forward-pointers** from each math Discussion (TL;DR #2).
3. Add a **GPU forward-pointer** (or 3-line device demo) at the ndarray intro.
4. Fix the two **autograd citations** (Speelpenning/backprop, Julia/ForwardDiff).
5. Update **lookup-api MXNet docs** 1.8.0 → 2.x.

**Tier 2 — section strengthening:**
6. Rewrite **ndarray exercises** (2 → ~6 load-bearing) and add **lookup-api
   exercises**.
7. Add a short **dtype/precision** paragraph and a **seeding/reproducibility**
   note in ndarray/probability; state the **broadcasting rule** + a failure case.
8. Add a **distributions/MLE** breadcrumb (pointer to MDL) in probability.

**Tier 3 — polish & cuts:**
9. Move calculus **"Visualization Utilities"** plumbing to the library.
10. Clarity fixes: probability's independence-as-shorthand; linear-algebra's
    "projects"; autograd's detach passage; calculus's Jacobian aside; order/rank
    consistency; "An Example" → "Worked Example: HIV Testing."
11. Optional: trim probability's aleatoric/epistemic + Chebyshev to a pointer;
    trim the TF `tf.function` digression in ndarray.

None of this is breakage — the chapter is solid. Tier 1 items 1–3 are the ones
that most improve the on-ramp for the least effort: they make the section deliver
on its own stated promises (the gradient→descent link, the preview→depth path,
and the GPU pitch).
