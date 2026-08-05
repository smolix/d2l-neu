# Chapter Overview — chapter_preliminaries

**Scope:** 7 sections with `## Exercises` (ndarray, pandas, linear-algebra, calculus,
autograd, probability, lookup-api), book order per `_quarto.yml`; `index.md` has no
exercises. All 7 are already strong by the prior style review (109-file group audit):
no clarity defects flagged in linear-algebra/calculus/autograd/probability/lookup-api,
only mild "vary X, see what happens" softness in pandas (ex6-7) and one item in
linear-algebra (ex6). No file in this chapter uses named/tagged exercises — bare
`1.` lists throughout, consistent with the whole book's legacy style.

**Best external sources found:** VMLS (Boyd & Vandenberghe, free PDF, actively
maintained "Additional Exercises" — updated Dec 2025) is the standout: real, dated,
citable problem numbers spanning vectors/norms/matrix-products, several with the
exact same pedagogical shape as this book's own (applied word-problems in
matrix/vector notation). MIT 18.05 PSet 7 (Spring 2022, solutions posted) is
similarly excellent for probability — direct Bayes'-theorem/diagnostic-reasoning
matches. UC Berkeley Data 100 (Sp25, live GitHub student repo) and Kaggle Learn's
"Intermediate Machine Learning" course together cover the full pandas pipeline
(messy real data, missing-value strategy comparison via held-out MAE, cardinality
tradeoffs) better than any single source. CS231n's "An Exercise in Backpropagation"
handout (2026) is a near-exact structural match for autograd's computational-graph
exercises. MIT 18.06 PSet 1 (Strang) supplies a Markov-matrix repeated-multiplication
problem that deepens the book's own new eigenvalue subsection.

**Coverage gaps:** lookup-api has essentially no external exercise tradition — "how
to read documentation" is taught, not examined, everywhere else (confirmed: MIT
Missing Semester covers the tools but sets no exercises on the *process* of
discovery/verification). Calculus's ML-flavored items (numerical cancellation in
finite differences, gradient-descent step size) have no counterpart in pure MIT
calculus problem sets — those drill rule mechanics only, not the floating-point or
optimization framing that make this book's versions distinctive. Autograd's
forward-vs-reverse-mode cost tradeoff is checked-but-thin outside this book.

**How existing sets fare:** ndarray, calculus, autograd, and probability need almost
no rescue work — the improvements below are additions/upgrades, not replacements,
per the rubric's guidance for already-strong chapters. pandas and linear-algebra
each have one or two genuinely underspecified items worth tightening. Totals below.

## chapter_preliminaries/ndarray.md — Data Manipulation

**Topic:** Tensors as the core data structure — creation, indexing/slicing (read and
write), elementwise ops, broadcasting, reductions, memory/aliasing (`id()`,
in-place ops), conversion to/from NumPy.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — review found no
clarity defects; every item is a concrete run/predict/verify task already anchored
to real API calls. Strongest section in the chapter for existing quality.

**External sources found:**
- Nicolas P. Rougier, "100 numpy exercises" (community-maintained, difficulty-rated
  ★☆☆–★★★, n/d) — #35 computes `(A+B)*(-A/2)` in place without allocating a new
  array; #43 makes an array read-only and confirms writes then raise; #62/#71 drill
  broadcasting (`(1,3)+(3,1)`; multiplying a `(5,5,3)` array by a `(5,5)` array); #67
  sums the last two axes of a 4-D array at once — https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md
- Stanford CS231n, Assignment 1 Q1 (k-nearest neighbors, recurring yearly, 2026
  offering) — implement the same pairwise-distance computation with 0, 1, and 2
  Python for-loops, building vectorization proficiency by comparing runtimes —
  https://cs231n.github.io/assignments2026/assignment1/
- UC Berkeley Data 100, "Numpy_Review" notebook (Fall 2017) — predict-then-verify
  exercises: predict a reshape's layout before running it, predict what `x[:, 0]`
  and `x[0::2, :]` slice out, predict `A[A > 3]`'s shape/values, then replace a
  `-999.0` sentinel with `np.nan` — https://ds100.org/fa17/assets/notebooks/numpy/Numpy_Review.html
- fast.ai, "Deep Learning from the Foundations" Part 2, Lesson 8 (2019) — rewrites a
  single matrix multiplication three times (nested loops → broadcasting → einsum),
  timing each, to make the cost of unvectorized code visceral —
  https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-2-8.md
- Stanford CS231n, "Python NumPy Tutorial" — canonical broadcasting exposition;
  checked directly and confirmed reference-only, no embedded exercises, so it is
  cited here as *background reading*, not as a problem source —
  https://cs231n.github.io/python-numpy-tutorial/

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Comparison operator swap.** Change `X == Y` to `X < Y` and
   `X > Y` in this section's code and report what kind of tensor each produces.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Broadcasting in three dimensions.** Replace the two
   broadcasting operands with 3-D tensors of your choosing and check whether the
   result matches your prediction of the broadcast shape.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Inferred reshape axis.** Build `x = arange(24)` and reshape it
   to `(2, 3, 4)` using `-1` for one component. State which component the
   framework infers and why at most one `-1` is allowed, before checking in code.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Predict-then-verify reduction shapes.** For a `(3, 4)` tensor,
   predict the shape of its sum along `axis=0`, along `axis=1`, and with
   `keepdims=True`; check each prediction against the code.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Fixing a broadcast mismatch.** Add two tensors of shape `(3,
   2)` and `(2, 3)`, read the resulting error against the from-the-right
   alignment rule, then find a reshape of one operand that makes the addition
   valid.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Object identity vs. storage.** Use `id()` to confirm that
   `X[:] = X + Y` (or `X += Y`) preserves the Python object while `X = X + Y`
   rebinds it, then explain why identity alone does not prove the underlying
   storage was reused, checking a storage/buffer API where your framework
   exposes one.
   *Provenance:* original (book's existing exercise).
1. [short-code] **In-place composition and read-only arrays.** Compute
   `(A + B) * (-A / 2)` for two same-shaped tensors while allocating no new
   array (only pre-allocated buffers/`Z[:] = ...` writes), verifying via `id()`
   that no intermediate rebinds. Then, in a framework that supports it, mark an
   array read-only and confirm that a write now raises — and relate this to
   why JAX arrays are unconditionally immutable.
   *Provenance:* adapted from Rougier, "100 numpy exercises" #35 and #43 (overlap
   med — same tasks, reframed around this section's `id()`/aliasing discussion).
1. [extended] **Loops vs. broadcasting vs. a library primitive.** Compute all
   pairwise squared Euclidean distances between two small sets of vectors (e.g.,
   50 points in 8 dimensions each) three ways: (a) two nested Python loops, (b)
   one loop plus broadcasting, (c) no explicit loop (broadcasting or a library
   `cdist`-style call). Confirm the three give the same result up to floating-point
   tolerance, then time all three and report the speedup ratios.
   *Provenance:* adapted from Stanford CS231n Assignment 1 Q1 (kNN 0/1/2-loop
   progression, overlap med) and fast.ai DL-from-Foundations Lesson 8 (loops →
   broadcasting → einsum progression, overlap med).

---

## chapter_preliminaries/pandas.md — Data Preprocessing

**Topic:** Turning a raw CSV into a model-ready tensor — inspect, split
inputs/targets, handle missing values, one-hot encode, standardize, and convert,
with an explicit warning about train/test leakage.
**Current exercises:** 7; disposition: keep 4, rewrite 3, drop 0 — items 1, 2, 4,
and 5 each already state a concrete deliverable and are kept unchanged; item 3
(imputation strategies) is upgraded with an external MAE-comparison methodology
rather than left as visual comparison alone; items 6–7 ("how would you
handle...", "what alternatives can you think of") were open brainstorm prompts
with no artifact per the prior review, and are rewritten with concrete
deliverables below.

**External sources found:**
- UC Berkeley Data 100, HW 2A "Food Safety" (Spring 2025, live student repo) —
  cleans real San Francisco restaurant-inspection data: detects invalid ZIP codes
  with `isin()`, handles a `-9999` missing-value sentinel, extracts business IDs
  from composite ID strings, parses dates with `pd.to_datetime`, and checks address
  leading-digits against Benford's law — https://github.com/DS-100/sp25-student/blob/main/hw/hw02A/hw02A.ipynb
- UC Berkeley Data 100, Lab 2B "Data Cleaning and EDA" (Spring 2025) — `groupby`
  + `agg`/`filter`, `pivot_table`, `str.split` to extract a substring column,
  `pd.merge` across two real datasets, and `fillna` to zero out a pivoted table's
  gaps — https://github.com/DS-100/sp25-student/blob/main/lab/lab02B/lab02B.ipynb
- UC Berkeley Data 100, Project A1 "Housing" (Spring 2025) — log-transforms two
  right-skewed numeric features, removes outliers via the IQR rule, engineers a
  new numeric feature by regex-extracting bathroom counts from free text, and
  one-hot encodes a categorical wall-material column for linear regression —
  https://github.com/DS-100/sp25-student/blob/main/proj/projA1/projA1.ipynb
- Kaggle Learn, "Intermediate Machine Learning" course, exercise "Missing Values"
  (Alexis Cook) — on the Ames Housing dataset, implements drop-columns,
  mean-imputation, and imputation-with-indicator-column side by side, then
  compares them by the *held-out validation MAE* of a downstream model rather
  than by inspection alone — https://www.kaggle.com/code/alexisbcook/missing-values
- Kaggle Learn, "Intermediate Machine Learning" course, exercise "Categorical
  Variables" (Alexis Cook) — compares drop / ordinal-encode / one-hot-encode, and
  explicitly walks through *cardinality* as the reason to route some columns to
  ordinal encoding instead of one-hot — https://www.kaggle.com/code/alexisbcook/categorical-variables

**Proposed problem set** (8 problems, our reference format):
1. [short-code] **A messier real dataset.** Load the UCI Abalone dataset and
   report what fraction of values are missing and what fraction of columns are
   numerical, categorical, or text.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Indexing by name.** Redo this section's column selection using
   name-based indexing instead of `iloc`, citing the specific pandas indexing
   method you used.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Imputation strategies, judged by outcome.** On `NumRooms`,
   compare mean imputation, median imputation, and a "was-missing" indicator
   column. State which assumption each makes about *why* data is missing, then
   — going beyond visual comparison — fit the same downstream model on each
   variant and report which gives the lowest held-out error.
   *Provenance:* adapted from Kaggle Learn, "Intermediate Machine Learning,"
   exercise "Missing Values" (overlap med — same three strategies, but that
   exercise supplies the MAE-comparison methodology this book's original item
   lacked).
1. [conceptual] **Where leakage hides.** Explain why standardizing on statistics
   from the whole dataset (rather than the training split alone) leaks
   information, and what breaks if a feature has zero variance under this
   standardization.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Scaling limits.** Estimate how large a CSV you could load this
   way on your own machine, considering read time, in-memory representation, and
   processing; identify what breaks first as the file grows.
   *Provenance:* original (book's existing exercise).
1. [short-code] **High-cardinality encoding, quantified.** Pick a categorical
   column and compute how many new columns one-hot encoding would add as a
   function of its number of unique values. For a column where every value is
   unique (an identifier), decide — and justify — whether to include it at all,
   drop it, or route it to ordinal/embedding treatment instead.
   *Provenance:* adapted from Kaggle Learn, "Intermediate Machine Learning,"
   exercise "Categorical Variables" (overlap med — same cardinality-tradeoff
   reasoning, made quantitative and given a concrete decision to make).
1. [short-code] **Pandas alternatives, tried once.** Load the same `house_tiny`
   values two other ways — via `numpy.load` on a saved `.npy` array and, for one
   image-like feature, via Pillow — and report one concrete limitation each
   alternative has that pandas does not.
   *Provenance:* original (book's existing exercise, given a concrete deliverable
   in place of the open "what alternatives can you think of" framing).
1. [extended] **A real messy dataset end to end.** Using the Data 100 Food Safety
   business/inspection files (or a similarly messy public dataset), run the full
   pipeline this section teaches: detect a non-standard missing-value sentinel (a
   value like `-9999` rather than a blank field), decide a handling strategy and
   justify it, one-hot or ordinal encode at least one categorical column with a
   cardinality judgment call, standardize the numeric columns, and convert the
   result to a tensor — reporting the shape and dtype of the final tensor.
   *Provenance:* adapted from UC Berkeley Data 100, HW 2A "Food Safety" (overlap
   med — same dataset family and sentinel-value/cleaning tasks, restructured
   around this section's own pipeline).

---

## chapter_preliminaries/linear-algebra.md — Linear Algebra

**Topic:** Vectors/matrices/tensors as objects; elementwise arithmetic and
reductions; dot products, matrix–vector and matrix–matrix products; norms; a
first look at eigenvalues via repeated multiplication.
**Current exercises:** 12; disposition: keep 7, rewrite 0, drop 5 — item 6 ("run
`A / A.sum(axis=1)`... can you analyze the results?") has no stated comparison
target per the prior review and is dropped outright rather than patched; items 2
(sum/transpose commute), 5 (which axis does `len()` correspond to), 8 (shapes of
`sum` along each axis of a `(2,3,4)` tensor), and 11 (`AB` vs `AC^T` speed) are
each close enough in substance to a kept sibling item that dropping them tightens
the set and makes room for two additions below.

**External sources found:**
- Boyd & Vandenberghe, "Additional Exercises for VMLS" (updated Dec 14, 2025;
  used in Stanford ENGR108 and UCLA EE133A) — 3.11 asks whether the triangle
  inequality generalizes to three vectors; 3.16 asks what `‖a+b‖ < ‖a‖` implies
  about the angle between `a` and `b`; 10.2 and 10.6 are applied matrix-product
  word problems (a customer-purchase matrix's `C1`, `CᵀC`, `CCᵀ`; a social-network
  "friend matrix" expressing common-friend counts as `FFᵀ`) —
  https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf
- MIT 18.06, Problem Set 1 solutions (Gilbert Strang, Spring 2010) — Section 1.2
  Problem 23 derives `cos(θ) = v·w/(‖v‖‖w‖)` from the angle-sum trig identity, the
  same cosine/dot-product relationship this section states; Section 1.2 Problem 28
  asks whether three vectors in the plane can be pairwise obtuse (`u·v<0`,
  `v·w<0`, `u·w<0` simultaneously) — surprising answer: yes; Section 2.1 Problems
  29–30 repeatedly multiply a vector by a fixed matrix and ask what the reader
  notices about the resulting sequence, converging to a dominant eigenvector —
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/22297c2a6dcf06d82e93ee4af115e91a_MIT18_06S10_pset1_s10_soln.pdf

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Transpose and symmetry proofs.** Prove `(Aᵀ)ᵀ = A`, and prove
   that `A + Aᵀ` is always symmetric for any square `A`, using only the transpose
   identities established in this section.
   *Provenance:* original (book's existing exercises, merged).
1. [conceptual] **Predicting `len()` on a higher-order tensor.** For the `(2, 3,
   4)` tensor defined in this section, predict the output of `len(X)` without
   running code, then check your answer.
   *Provenance:* original (book's existing exercise).
1. [short-code] **A pairwise-obtuse triple.** Can three vectors in the plane
   satisfy `u·v < 0`, `v·w < 0`, and `u·w < 0` simultaneously? Give a concrete
   numeric example or a short argument for why none exists, then confirm your
   answer numerically.
   *Provenance:* adapted from MIT 18.06 PSet 1, Section 1.2 Problem 28 (overlap
   high — same question and construction).
1. [conceptual] **Manhattan distance.** When traveling between two points in
   downtown Manhattan, express the distance to cover in terms of the two points'
   avenue/street coordinates. Can you travel diagonally?
   *Provenance:* original (book's existing exercise).
1. [short-code] **What does `norm` compute for a 3-axis tensor?** Feed a tensor
   with three or more axes to `linalg.norm` and observe its output. State in
   words what quantity this function computes for tensors of arbitrary shape.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Matrix-product order and cost.** For
   $\mathbf{A}\in\mathbb{R}^{2^{10}\times 2^{16}}$,
   $\mathbf{B}\in\mathbb{R}^{2^{16}\times2^5}$, and
   $\mathbf{C}\in\mathbb{R}^{2^5\times2^{14}}$ with Gaussian entries, is there a
   difference in memory footprint and speed between computing $(\mathbf{AB})\mathbf{C}$
   and $\mathbf{A}(\mathbf{BC})$? Justify your answer using the operation counts
   from this section's matrix-multiplication cost discussion.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Stack and slice.** Given three matrices
   $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^{100\times200}$, stack them
   into a single 3-axis tensor, report its dimensionality, then slice out the
   second coordinate along the stacking axis and confirm it recovers $\mathbf{B}$
   exactly.
   *Provenance:* original (book's existing exercise).
1. [extended] **Watching an eigenvector emerge.** Using the symmetric matrix
   $\mathbf{S}$ from this section, implement power iteration by hand: start from
   a random vector, repeatedly multiply by $\mathbf{S}$ and renormalize, and
   track the vector across iterations (not just the norm ratio this section
   already computes). Report at what iteration the *direction* stabilizes (e.g.,
   cosine similarity between successive iterates exceeds 0.999), then compare
   your converged vector to the top eigenvector returned by `linalg.eigh`.
   *Provenance:* adapted from MIT 18.06 PSet 1, Section 2.1 Problems 29–30
   (overlap med — same repeated-multiplication-to-a-fixed-direction idea, applied
   here to this section's own symmetric matrix and eigenvalue discussion instead
   of a Markov transition matrix).

---

## chapter_preliminaries/calculus.md — Calculus

**Topic:** The limit definition of the derivative and its numerical behavior;
derivative rules; partial derivatives and the gradient as steepest ascent; a
handful of gradient identities; the chain rule as a backprop preview.
**Current exercises:** 11; disposition: keep 8, rewrite 0, drop 3 — the prior
review flagged no clarity defects in this file at all (the strongest-reviewed
section in the chapter); items 3 (constant-multiple rule as a special case of
the product rule), 5 (meaning of `f'(x)=0`, give an example), and 10 (derivative
of an inverse function) are dropped only to fit the 5–8 target, each being a
minor/self-contained aside relative to the kept items' spread across rule-proof,
plotting, gradient, and optimization-preview categories.

**External sources found:**
- MIT 18.01SC, Problem Set 1 (Single Variable Calculus, Fall 2010) — covers
  slope/derivative, limits and continuity, and differentiation formulas
  (polynomials, products, quotients) as separate problem groups, the same rule
  families this section's exercises 1–2 drill —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/problem-set-1/
- MIT 18.02SC, Problem Set 4 (Multivariable Calculus, Fall 2010) — topic is
  "level curves, partial derivatives, and tangent plane," matching this section's
  partial-derivative/gradient material, though the specific problem text was not
  retrievable in this session (only the topic label, confirmed live) —
  https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/resources/mit18_02sc_pset4/
- Stanford CS231n, "An Exercise in Backpropagation" section handout (2026) —
  its warm-up is a tiny computational graph $f(x,y,z)=(x+y)\cdot z$ whose
  forward/backward passes are traced by hand, the same chain-rule-on-a-graph
  skill this section's chain-rule discussion sets up for autograd —
  https://cs231n.stanford.edu/slides/2026/section_2_backprop.pdf (cited fully
  under autograd.md below, where the overlap is closer)

**Verdict on external tradition:** thin for this section's most distinctive
material. MIT's calculus problem sets drill the same rule mechanics (derivative
rules, partial derivatives) but have no counterpart for two things this book
does that are genuinely its own: turning the limit definition into a
*numerical* experiment that exposes floating-point cancellation, and using a
gradient-descent step as a first taste of the learning-rate tradeoff. Neither
appears in pure-math problem sets; the latter belongs to ML/optimization courses
that assume more than this section has introduced. No external adoption is
proposed for those two items as a result — they are kept as the book's own.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Derivative rules from the limit definition.** Using the limit
   definition, prove the derivative rules for (i) $f(x)=c$, (ii) $f(x)=x^n$,
   (iii) $f(x)=e^x$, and (iv) $f(x)=\log x$.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Product, sum, and quotient rules from first principles.**
   Prove each rule directly from the limit definition of the derivative (not by
   citing the rules themselves).
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Derivative of $x^x$.** Compute $\frac{d}{dx}x^x$.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Tangent line at a point.** Plot $f(x) = x^3 - \frac{1}{x}$
   together with its tangent line at $x=1$.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Gradient of a two-variable function.** Find the gradient of
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Gradient of the $\ell_2$ norm.** What is the gradient of
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$? What happens at $\mathbf{x}=\mathbf{0}$,
   and why?
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Multivariable chain rule by hand.** Write out the chain rule
   for $u = f(x,y,z)$ where $x=x(a,b)$, $y=y(a,b)$, and $z=z(a,b)$.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Learning rate as a knob.** Starting from
   $\mathbf{x}=[1,1]^\top$, take one gradient-descent step
   $\mathbf{x}\leftarrow\mathbf{x}-\eta\nabla f(\mathbf{x})$ on
   $f(\mathbf{x})=\|\mathbf{x}\|_2^2$ with $\eta=0.1$, then $\eta=1$, then
   $\eta=2$, verifying in each case whether $f$ decreases. What does this show
   about the role of the learning rate?
   *Provenance:* original (book's existing exercise).

---

## chapter_preliminaries/autograd.md — Automatic Differentiation

**Topic:** The autograd workflow (attach/track, record, backward, read the
gradient); non-scalar backward calls; detaching a graph; turning off tracking;
differentiating through control flow; higher-order derivatives; forward- vs.
reverse-mode cost.
**Current exercises:** 8; disposition: keep 3, rewrite 4 (merged into one
multi-part item), drop 1 — items 5–8 form an explicit chain (each references
"the aforementioned function" or "the graph of exercise 5" in plain prose), which
the prior review flagged as a stale cross-reference risk if the list is ever
reordered; merging them into one item with lettered sub-parts keeps every piece
of content while removing that fragility. Item 4 (plot $\sin(x)$ and its
derivative via autograd) is dropped as redundant with calculus.md's own
plot-plus-tangent-line exercise, freeing room for two gap-filling additions
below (this section has whole subsections on detaching a graph and on turning
off gradient tracking, but no exercise on either).

**External sources found:**
- Stanford CS231n, "An Exercise in Backpropagation" section handout (Favour
  Nerrise, Spring 2026) — a warm-up computational graph
  $f(x,y,z)=(x+y)\cdot z$ with concrete values ($x=-2,y=5,z=-4$): compute the
  forward pass, then trace $\partial f/\partial x$, $\partial f/\partial y$,
  $\partial f/\partial z$ backward through the `+` and `×` gates by hand —
  https://cs231n.stanford.edu/slides/2026/section_2_backprop.pdf
- Andrej Karpathy, micrograd (github.com/karpathy/micrograd; also the basis of
  his "Neural Networks: Zero to Hero" series) — a complete, ~100-line
  scalar-valued reverse-mode autograd engine, presented as a from-scratch build
  that demystifies exactly the record-forward/sweep-backward workflow this
  section teaches; note this is a reference implementation to study or
  reproduce rather than a set of discrete graded exercises —
  https://github.com/karpathy/micrograd
- JAX official docs ("Automatic differentiation," "The Autodiff Cookbook") and
  the official PyTorch autograd tutorial — both checked directly and confirmed
  to be worked-example/cookbook style with no embedded exercises or "try it
  yourself" prompts, so they are cited as background reference only, not as
  problem sources — https://docs.jax.dev/en/latest/automatic-differentiation.html,
  https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **Cost of the second derivative.** Why is computing a second
   derivative substantially more expensive than computing the first?
   *Provenance:* original (book's existing exercise).
1. [short-code] **Running backward twice.** After running the backpropagation
   function once, run it again on the same graph. What happens, and how does
   the behavior differ across frameworks?
   *Provenance:* original (book's existing exercise).
1. [conceptual] **From scalar to vector input.** In the control-flow example
   that differentiates `d` with respect to `a`, what changes if `a` is a random
   vector or matrix instead of a scalar? The result of `f(a)` is then no longer
   a scalar — what happens to the gradient computation, and how would you
   analyze it?
   *Provenance:* original (book's existing exercise).
1. [short-code] **A dependency graph, built and traced by hand.** For
   $f(x) = (\log x^2 \cdot \sin x) + x^{-1}$: (a) draw the dependency graph from
   $x$ to $f(x)$; (b) apply the chain rule along that graph to compute
   $\frac{df}{dx}$, placing each intermediate term on the graph; (c) evaluate the
   graph once sweeping forward (forward mode) and once sweeping backward from
   $f$ to $x$ (reverse mode); (d) count the operations performed and
   intermediate values stored by each mode, and describe how that comparison
   would change for a function with many inputs, or with many outputs.
   *Provenance:* original (book's existing exercises 5–8, merged into one item
   with lettered sub-parts to remove the stale plain-text cross-references
   between them).
1. [short-code] **Trace a graph, then let autograd check you.** By hand, compute
   the local gradients at each node of the graph $f(x,y,z) = (x+y)\cdot z$ for
   $x=-2, y=5, z=-4$: find $\partial f/\partial x$, $\partial f/\partial y$, and
   $\partial f/\partial z$ by multiplying local gradients backward through the
   `+` and `×` nodes. Then verify every value against your framework's autograd.
   *Provenance:* adapted from Stanford CS231n, "An Exercise in Backpropagation"
   section handout (overlap high — same function, same values, same hand-trace
   task, restructured to end in a verification step this book's own
   discover→verify style favors).
1. [short-code] **Detach and check.** Given `y = x * x` and `z = x * u` where `u`
   is `y` detached (or `stop_gradient`-wrapped) from the graph, verify that
   $\partial z/\partial x$ equals $u$ (treating `u` as a constant) rather than
   the $3x^2$ you would get without detaching. Then confirm that the graph
   leading to `y` itself is unaffected: $\partial y/\partial x$ still equals
   $2x$.
   *Provenance:* original (fills a gap: this section has a whole subsection on
   detaching computation but no dedicated exercise on it).
1. [conceptual] **Why turn tracking off?** Wrap a computation in your
   framework's no-tracking context (e.g., `no_grad`, `stop_recording`, or simply
   not calling `grad`) and confirm the result carries no gradient information.
   Then estimate, in words, why skipping this bookkeeping matters at prediction
   time for a model with millions of parameters.
   *Provenance:* original (fills a gap: this section motivates turning off
   gradient tracking for inference but has no exercise exercising it directly).

---

## chapter_preliminaries/probability.md — Probability and Statistics

**Topic:** Sample spaces and axioms; random variables; joint/conditional/marginal
probability and Bayes' theorem (worked HIV-testing example); independence and
conditional independence; expectation, variance, covariance; Markov's and
Chebyshev's tail bounds; aleatoric vs. epistemic uncertainty.
**Current exercises:** 8; disposition: keep 7, rewrite 0, drop 1 — the prior
review found no clarity defects anywhere in this file; item 6 (simplify
$P(A,B,C)$ for a Markov chain $A\to B\to C$) is dropped only to make room for a
new addition, being the one item most disconnected from this section's two
running examples (coin tosses, the two-test HIV diagnosis) that the rest of the
set builds on.

**External sources found:**
- MIT 18.05, Problem Set 7 solutions (Spring 2022, Orloff & Bloom) — Problem 1
  ("Monty Hall: Sober and Drunk") builds a Bayes table under two different
  likelihood models for the *same* observed data and shows the best strategy
  changes; Problem 3 ("Odds") runs a full prior→likelihood→posterior update on a
  drawer of biased coins; Problem 4 ("Courtroom fallacies") is the prosecutor's
  fallacy: a lawyer conflates $P(\text{guilty}\mid\text{evidence alone})$ with
  $P(\text{guilty}\mid\text{all evidence})$, and the solution shows precisely
  where the conflation breaks —
  https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_pset07_sol.pdf
- Harvard Stat 110 (Blitzstein) — confirmed to exist and to be exactly the kind
  of source this topic calls for (Bayes' rule, conditional independence,
  odds-form updates), but its homework/strategic-practice PDFs returned HTTP 403
  in this session and could not be independently verified; not cited with a
  specific problem number as a result, per the no-fabrication rule.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Uncertainty driven to zero.** Give an example where observing
   more data can reduce uncertainty about the outcome to an arbitrarily low
   level.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Uncertainty with a floor.** Give an example where observing
   more data reduces uncertainty only up to a point, and no further. Explain why,
   and where you expect that point to occur.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Variance of the coin-toss estimator.** Calculate the variance
   of the estimated head-probability after $n$ tosses. (a) How does the
   variance scale with $n$? (b) Use Chebyshev's inequality to bound the
   deviation from the true probability. (c) How does this relate to the central
   limit theorem?
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Does Chebyshev apply term by term?** Given $m$ iid samples
   $x_i$ from a zero-mean, unit-variance distribution and their running average
   $z_m = m^{-1}\sum_{i=1}^m x_i$, can Chebyshev's inequality be applied to each
   $z_m$ independently? Why or why not?
   *Provenance:* original (book's existing exercise).
1. [conceptual] **Union and intersection bounds.** Given $P(\mathcal{A})$ and
   $P(\mathcal{B})$, derive upper and lower bounds on $P(\mathcal{A}\cup\mathcal{B})$
   and $P(\mathcal{A}\cap\mathcal{B})$.
   *Provenance:* original (book's existing exercise).
1. [short-code] **A second, correlated test.** Redo the two-test HIV example
   assuming the tests are *not* conditionally independent given $H=0$: each test
   alone has a 10% false-positive and 1% false-negative rate, tests are
   conditionally independent given $H=1$, but $P(D_1{=}D_2{=}1\mid H=0)=0.02$.
   (a) Work out the joint table for $(D_1,D_2)$ given $H=0$. (b) Find
   $P(H=1\mid D_1=1)$. (c) Find $P(H=1\mid D_1=1, D_2=1)$.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Portfolio return and risk.** For a portfolio with weights
   $\boldsymbol{\alpha}$ summing to 1 over assets with mean return
   $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$: (a) compute the
   expected portfolio return; (b) state how to choose $\boldsymbol{\alpha}$ to
   maximize it alone; (c) compute the portfolio variance; (d) formulate (without
   solving) the Markowitz problem that maximizes return subject to a variance
   cap.
   *Provenance:* original (book's existing exercise).
1. [conceptual] **The prosecutor's fallacy.** A suspect had a documented history
   of abusing his now-murdered wife. His lawyer argues: "statistically, only one
   in a thousand wife-abusers goes on to murder his wife — so the history of
   abuse is weak evidence, and you should find him innocent." Using $M$ = "the
   suspect murdered his wife," $K$ = "his wife was killed," and $B$ = "he had a
   history of abusing her," express the lawyer's $1/1000$ figure and the correct
   posterior of guilt in these terms, and explain precisely which probability the
   lawyer has confused with which.
   *Provenance:* adapted from MIT 18.05 PSet 7, Problem 4 "Courtroom fallacies"
   (overlap high — same problem, same hint structure, credited to
   Mackay's *Information Theory, Inference, and Learning Algorithms* in the
   original solution).

---

## chapter_preliminaries/lookup-api.md — Documentation

**Topic:** A four-step procedure for an unfamiliar API — discover names with
`dir()`, inspect a signature with `help`/`?`/`??`, and verify behavior with a
small runnable example — plus treating a coding assistant's suggestion as an
unverified candidate.
**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — the prior
review found no clarity defects; all three already state a concrete action and a
checkable outcome.

**External sources found:** genuinely none with a real primary-source exercise
tradition. This is a meta-skill ("how to learn a library"), not a subject with
its own course unit anywhere checked: MIT's "Missing Semester" teaches
documentation-reading tools (`man`, `--help`) but sets no exercise on the
*process* of discover→inspect→verify itself, only on configuring tools
(confirmed directly, no matching exercise found); the official NumPy/PyTorch/JAX
tutorials are reference material, not problem sets (confirmed for JAX and
PyTorch directly while researching autograd.md, above); and no software-engineering
course was found with a graded exercise on critically verifying an AI coding
assistant's suggested API call — exercise 3 in this section is, as far as this
research could tell, ahead of any external tradition rather than behind one.
This absence is itself the finding for this section.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Discover, inspect, verify: uniform sampling.** Use `dir()` on
   your framework's random-number module to find the routine that samples from a
   uniform distribution. Read its signature with `help` (or `?`), then call it
   to draw a $3\times3$ tensor and confirm the values lie in $[0,1)$.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Discover, inspect, verify: reducing along an axis.** Look up
   your framework's `sum` (or `reduce_sum`) with `help`, identify the argument
   that selects the axis, and verify on a $2\times3$ tensor that summing over
   each axis gives the shape you predicted.
   *Provenance:* original (book's existing exercise).
1. [short-code] **Checking a coding assistant's answer.** Ask a coding assistant
   how to concatenate two tensors along a new axis in your framework. Run its
   answer through the discover→inspect→read→verify loop: does the suggested
   function exist (`dir`)? Does its signature match the claim (`help`/`?`)? Does
   a tiny example behave as expected?
   *Provenance:* original (book's existing exercise).
1. [short-code] **Reading the source, not just the docstring.** Pick a function
   you have used in this chapter whose docstring does not fully explain its
   behavior on an edge case you care about (e.g., what `reshape` does when a
   dimension does not evenly divide the requested shape). Use `??` (or your
   editor's "go to definition") to read its source, find the line that decides
   the edge case, and confirm your reading with a small example.
   *Provenance:* original (fills a gap: this section demonstrates `??` in the
   text but has no exercise that specifically exercises source-reading, as
   distinct from docstring-reading).
1. [conceptual] **When the assistant is confidently wrong.** Ask a coding
   assistant for the name of a plausible-sounding but nonexistent function in
   your framework (for example, invent a name in the style of a real one, such
   as asking for a function that "sorts a tensor's axes by size"). Confirm via
   `dir()`/`hasattr` that no such name exists, and describe, in one or two
   sentences, what about the assistant's answer would have fooled you if you had
   skipped the discover step.
   *Provenance:* original (extends the section's own closing point — that an
   assistant's suggestion is an unverified candidate — to the harder case where
   the suggestion is an outright hallucination rather than a real-but-unverified
   function).
