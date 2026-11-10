# Writing pass log: Concentration Inequalities

Source: `chapter_mdl-probability-statistics/mdl-concentration-generalization.md`.
Prose only; no code cell, cell ID, figure include or slide changed. Not split,
no markers. Numeric checks were run with `python3` and NumPy (no SciPy, no
PyTorch on this laptop); chi-square and chi quantities were computed from
`math.lgamma` series and a numerical integral of the chi density.

## Words

Prose words (checker count): 6719 before, 6710 after (-9). Body without
exercises (own count): 5081 before, about 5070 after. The checker ends "OK:
only prose changed", no em-dash warning; `tools/lint_source.py` is clean. The
section is not shorter by much: the cuts (restated Markov, second Gaussian
proof, second pair count, repeated table readings, the closing recap, the
expert remark after Bernstein) were spent on repairs that add text (CON-9,
CON-10, CON-13, CON-7, CON-4).

## Correctness changes

- **CON-13** (sure; verified by hand). Said: for $\varepsilon>1$ "the tail
  becomes $e^{-d\varepsilon/8}$", after a proposition about the norm. Now: the
  restriction $\varepsilon\le1$ matters only for the squared norm, whose tail
  beyond 1 is exponential; the norm keeps its Gaussian tail, because a norm
  above $(1+\varepsilon)\sqrt d$ means a squared norm above
  $(1+2\varepsilon+\varepsilon^2)d$, with probability at most
  $e^{-d(2\varepsilon+\varepsilon^2)/8}\le e^{-d\varepsilon^2/8}$. Check: the
  sub-exponential bound with $\nu=2$, $\alpha=4$, $n=d$ and
  $t=2\varepsilon+\varepsilon^2>1$ is in the linear regime, exponent
  $-dt/8$; one-sided, so no factor 2. The proposition's statement is
  unchanged.
- **CON-9** (verified numerically). Said: Bernstein's $\pm0.39\%$ "in exchange
  for validity at every $n$", computed with the measured rate in the
  variance. Now: "These two half-widths are not yet guarantees", the reason,
  and the repair (true rate at most $\hat p+t$, solve with variance
  $(\hat p+t)(1-\hat p-t)$), giving 0.43% and 0.79%. Fixed-point iteration
  gives 0.4308% and 0.7933%; the lower sides are 0.36% and 0.73% (not
  quoted). The text quotes the larger, upper side as the half-width.
- **CON-16**. "about 570 directions are pairwise within" is now "this
  guarantees about 570 directions ... These are guarantees from a loose
  bound, not estimates of how many such directions exist."
  ($\sqrt{0.05}\,e^{7.84}=568$, $\sqrt{0.05}\,e^{17.64}=1.02\times10^7$.)
- **CON-2**. The sentence on Chebyshev's majorant now agrees with panel (a):
  "Chebyshev's uses $g(x)=(x/t)^2$, in practice with the deviation $X-\mu$ in
  place of $X$ ... Panel (a) draws the three functions for one nonnegative
  $X$."
- **CON-10**. The proxy is now distinguished from the variance in words (never
  smaller, second-order comparison; 0.0196 against $\tfrac14$ at 2% error).
  The symbol $\sigma^2$ is not changed (shared; see notation). $\nu^2$ and
  $1/\alpha$ are explained at the definition.
- **CON-4**. The circular sentence ("as long as the sum rarely strays ...") is
  replaced by what limits $\lambda$; the half-sentence in "The Chernoff
  Method" about raising $\lambda$ is removed.
- **CON-7**. Hoeffding with unequal ranges is stated as a two-line remark
  after the proof, with a new label, and used for the sign vectors.

No finding disagreed with.

## Six before/after pairs

1. Before: "Chebyshev's bound comes from Markov's inequality, so we restate
   both." After: "Chebyshev's bound is Markov's inequality
   :eqref:`eq_mdl-markov` applied to a squared deviation."
2. Before: "How far apart are Chebyshev's bound and the exact tail?" After:
   "At $n=1{,}000$ Chebyshev allows a miss in one experiment out of forty.
   Before reading the table, guess how often such a miss really happens."
3. Before: "We now set Hoeffding's bound $2e^{-2nt^2}$ beside the exact tail
   and Chebyshev's bound for the fair coin at $t=0.1$, and plot all three
   against $n$ on a log scale. Every number below is deterministic." After:
   "We add Hoeffding's bound $2e^{-2nt^2}$ as a third column to the table for
   the fair coin at $t=0.1$, and plot all three quantities against $n$ on a
   log scale."
4. Before: "We want the coin's bound for every bounded variable, and the
   proof rests on one fact about variances." After: "We want the coin's bound
   for every bounded variable. The lemma that gives it is the hard step of
   this section, and its proof rests on one fact about variances."
5. Before: "The chapter's coin, 9 heads in 13 flips, shows the other end."
   After: "At small $n$ no bound helps. The coin of the chapter introduction
   landed heads 9 times in 13 flips, a deviation of $t=2.5/13$ from
   $\tfrac12$."
6. Before: "We can now size a test set from a required precision and failure
   probability, and predict the norms and angles of random high-dimensional
   vectors." After (end of the last subsection): "Every bound in this section
   protects a quantity fixed *before* the data were drawn. A learner picks the
   best of many candidates *after* seeing the data, and the best of many fair
   coins looks biased."

## Repetition

Removed or reduced:

- Markov's inequality: display and restatement removed, one clause with
  `eq_mdl-markov` (owner: Random Variables).
- Union bound: no longer defined for the two tails ("adding the two tails
  gives the factor 2"); defined once, in general form, at the count of random
  directions. Uniform Convergence still defines it at its line 182; that file
  should reduce to a clause (not mine to edit).
- Near-orthogonality: one proof (signs, via the unequal-range form); the
  Gaussian case is one sentence with the exercise. The opening restatement of
  the linear-algebra result is one clause.
- Pair count: once (directions). The distances paragraph cites it and gives
  only the result; its display and the Chebyshev-against-Hoeffding sentences
  (thesis of Uniform Convergence) are gone.
- Coin table at $n=1{,}000$: read once in full, after the first table. "The
  Fair Coin" keeps only the bound's own values; "The Fair Coin in Numbers"
  keeps the new observations (Chebyshev tighter at 100, factors 15 and 57,
  slopes, 13 flips).
- 66 heads teaser: gone; "the best of many fair coins looks biased".
- Many-models motive in the introduction: one sentence with a forward
  reference (CON-1, S3).
- Panel (b) of the majorant figure is discussed after the display that
  produces the parabola (CON-3).
- Expert remark after Bernstein (CON-11), "plain NumPy" sentence (S6), the
  concentration-of-measure paragraph (CON-14, cut), the three exact standard
  deviations of the norm (the cell shows them), the detail of the
  $\tfrac18$-is-tight paragraph (it is exercise 2b).

Kept because this file owns them: the exponential tail for random directions,
the union bound's definition, the pair count, the first coin table.

## Ranking, order, labels

- No Proposition block demoted. The six kept (variance of a bounded
  variable, Hoeffding's lemma, Hoeffding's inequality, sub-exponential tail,
  Bernstein, norm concentration) are all cited later.
- The introduction names the three results to remember; Hoeffding's lemma is
  called the hard step; the proof of the sub-exponential tail bound is marked
  skippable on a first reading.
- S2: prose of "Sub-Gaussian and Sub-Exponential Variables" reordered
  (Bernstein and its numbers first, with the 8.3% answer of CON-12; then
  sub-Gaussian as a name; then $x^2$ as the motive for sub-exponential). The
  two cells keep their order and their place relative to all other cells. The
  sub-exponential definition and proposition were not moved into "The Thin
  Shell" (headings and labels stay put). The heading now undersells the
  subsection, which opens with Bernstein; a rename is the author's call.
- Label added: `eq_mdl-hoeffding-unequal`. The display that carried no label
  in the near-orthogonality subsection is this one, moved (display count 52
  to 48: Markov, the moved one merged, the distances display, and the
  averaging computation now inline).
- New references, all resolve: `eq_mdl-markov`, `sec_mdl-exponential-family`
  (CON-5), `sec_mdl-generalization-bounds`.

## Cells to cut or redesign on the development machine

- `mdl-distributions-multivariate-gaussian-1`: mostly confirms the
  tangent-line standard deviation and $\sqrt{2/(\pi d)}$; it earns its place
  only through the $d=2$ row. Prose now presents it as two predictions with
  one row that will not fit. Its ID is a leftover from Distributions.
- `mdl-concentration-generalization-chebyshev-s-rate-2` and
  `...-the-tail-race-in-code` (S4): the second table reprints the first two
  columns. Drop the first two columns' repetition or merge the cells.
- `...-sub-gaussian-and-sub-exponential-variables` (CON-9): add the
  self-consistent half-width so that the printed numbers are the guarantees
  (0.43%, 0.79%).
- Imports cell loads unused framework modules (S6).

## Deferred

- Exercises left as they are: 1(a) now overlaps the remark after Hoeffding's
  proof (the text gives the result, the exercise the proof); 1(c), 2(a) and
  "Mode versus mass" (a), (b) repeat the text (review, Exercises).
- A sentence pointing to the two-classifiers exercise (review suggestion): not
  added, for length.
- Moving the sub-exponential block to the thin shell (S2, step 3).
- "law" for "distribution" (tilted law, three-point law) left.

## Claims added or changed (all checked)

- One miss in forty at $n=1{,}000$ (Chebyshev 0.025).
- Bernstein: 3.5 times tighter than Hoeffding (1.36/0.39), within a factor
  1.4 of the Gaussian; 0.43% and 0.79% self-consistent; "within about 0.8%"
  for the 8.3% classifier.
- Proxy never smaller than the variance; 0.0196 against $\tfrac14$.
- $1/\alpha$ is the half-width of the window of $\lambda$.
- Norm tail beyond $\varepsilon=1$ (CON-13 above).
- Unequal-range Hoeffding for sums; denominator $4\sum u_i^2=4$.
- Union bound holds "whether or not the events are independent".
- Hoeffding does not cover correlated Markov-chain draws (chapter note on the
  hand-off from Bayesian Computation).
- Numbers kept that no cell prints, recomputed: $10^{-110}$ (log10 = -109.9),
  $5\times10^{-11}$, $10^{-170}$, 570 and $10^7$, 0.028, 0.0201, 0.77 and
  0.27 for 13 flips, 18,445 and 9,604, 2.72 and 2.5, 10% and 0.5 and "factor
  of two" for depth. Dropped: exact sds 0.655/0.703/0.707, exact shell mass
  14.7%, miss probability $3\times10^{-10}$ (all correct, removed for
  length).

## Slides against the body

- Slide "…lower tail; the union bound adds the two": the body now says the
  two tails add and defines the union bound later.
- Slide "Bernstein … ±0.39% Instead of ±1.36%" and the recap line: the body
  now says 0.39% assumes the variance known and gives 0.43%.
- Slide "≈570 directions": the body says "guarantees".
- Last slide "the best of 1,000 fair coins shows 66% heads": the body no
  longer gives the number.
- Cover kicker "§25.5" is rewritten by `gen_slides.py`.
