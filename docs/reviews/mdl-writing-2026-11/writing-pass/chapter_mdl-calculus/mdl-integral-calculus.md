# Writing pass log: Integral Calculus

File: `chapter_mdl-calculus/mdl-integral-calculus.md`. Base `4deb96d6`.

Prose words (checker): 8712 before, 8661 after (-51). The body shrank by
about 50 words net although about 300 words were added on request of the
review (left-rule derivation IC-4, variance expansion IC-29, hypotheses
IC-13, the opening of the future second section). Checker: "OK: only prose
changed", no em-dash warning, 1 split marker; `lint_source.py` clean. Display
equations 43 to 42 (two school-calculus displays inlined, one added).

## Correctness changes

| ID | Before | After | Verified |
|---|---|---|---|
| IC-10 | "A heavy-tailed density is one whose tail is not bounded by any exponential" (defined on the density) | No second definition. "Power laws are the standard example of a heavy-tailed distribution (defined in `sec_mdl-random_variables` ...)" | Chapter 41 defines it on the tails (`mdl-random-variables.md`, "Heavy Tails"); that subsection has no label, so the section is cited. |
| IC-11 | "The two errors are independent ... neither change touches the other." | "To leading order the two errors add: a larger $b$ removes the first, a smaller $\epsilon$ the second." | Bias is $\tfrac\epsilon2(1-e^{-b})$: $3.16\times10^{-4}$ at $b=1$, $5.00\times10^{-4}$ at $b=20$, as in the stored output. |
| IC-13 | Stein identity by "integration by parts on $[-b,b]$ and $b\to\infty$", condition $gw\to0$ only | "Let $g$ be continuously differentiable with $g(z)w(z)\to0$ ..., and let the two integrals below converge. Integration by parts on $[-a,b]$, followed by the separate limits ..." ; "Both conditions hold for every polynomial $g$." | By hand. |
| IC-17 | $\int_{\phi(\Omega)}f(\mathbf x)\,d\mathbf x=\int_\Omega f(\phi(\mathbf x))\lvert\det J\rvert\,d\mathbf x$ | Left side now in $\mathbf y$; the cube "is sent to a tiny parallelepiped ... at $\mathbf y=\phi(\mathbf x)$". | Dummy rename only. Step II of the Gaussian integral still names the variables $(r,\theta)$ (unchanged). |
| IC-20 | "by $b=3$ the volume outside the square is below $10^{-3}$" | "about $10^{-4}$" | Stored output: $\pi-3.141454=1.4\times10^{-4}$. |
| IC-24 | "Across four decades the measured error tracks the slope $-\tfrac12$ ... with random excursions on both sides" | "A single running average wanders around the prediction, because its errors at nearby $n$ share most of their samples: it can stay on one side of the line for a long stretch, and it dips sharply wherever the error changes sign. What falls with slope $-\tfrac12$ is the envelope ..." and a pointer to the RMS experiment of exercise 9. | True of any draw (ordinary draw; prose changed, no re-capture needed). Table text likewise: rows "can lie a factor of a few to either side" (stored ratios 1.7, 2.1, 0.64, 0.6, 0.3). |
| IC-28 | "(a Lipschitz $g$ that is differentiable almost everywhere suffices)" | "(a Lipschitz $g$ suffices when $X$ has a density)" | Rademacher; kinks need probability zero. |
| IC-30 | "about $0.55$ against $0.66$"; "$0.71$ against $2.1$"; "within a percent or two" | "about $0.5$ against $0.7$"; "about $0.7$ against $2.1$"; "to within a few percent" | Stored: 0.54/0.66, 0.71/2.12; sd off by 0.9% and 1.2%. |
| IC-4 | Left-rule bias "follows from a first-order expansion (exercise ...)" before the fundamental theorem | After the first table only: "The constant $0.2$ is half of $f(2)-f(0)=0.4$ ... we derive it once the fundamental theorem is available." Derivation given in a run-in paragraph "The error of the left rule" after the evaluation rule. | Slice excess $\tfrac{\epsilon^2}2f'(x_i)$, sum $\to\tfrac\epsilon2(f(b)-f(a))$. Also removed the claim that the running example's sums fall short "because left ends are the low ends": $f$ is not increasing on $[1,2]$. |
| IC-29 | Three standard deviations quoted, derivation in exercise | $\textrm{Var}(2X)=4$ shown; display $\mathbb E[(\theta+z)^4z^2]=\theta^4+18\theta^2+15$, minus $(2\theta)^2$. | Recomputed, including $8\theta^2+6$ and $b^\star=\theta^2+3$. |
| IC-27 | "completing the square gives $\log\mathbb E[e^{\theta Z}]=\theta^2/2$" | the one line $\theta z-\tfrac12z^2=\tfrac12\theta^2-\tfrac12(z-\theta)^2$ is shown | By hand. |
| (mine) | Lebesgue integral "agrees with this one on every function in this book" | "agrees with the Riemann integral whenever the latter exists" | Standard for proper Riemann integrals. |

No finding disagreed with. Numbers kept that no cell prints (all recomputed):
$0.316$, $\sqrt\pi=1.7725$, $\sqrt{2\pi}=2.5066$, $\varphi(1.5)=0.13$,
$p(0)=0.40$, ratio $3.6=7.18/2$. Dropped: $2\pi=6.28$, the three hand values
of the $x^2$ sums (the cell prints them), "tails beyond $\pm8$ hold less than
$10^{-14}$".

## Six before/after pairs

1. Before: "The cell below checks the formula. It defines the **left-endpoint
   rule**: ..." After: "The function `left_sum` implements this
   **left-endpoint rule** for any $f$ ... We use it for every grid sum of the
   section, and try it first on $x^2$."
2. Before: "How fast do the sums approach $\tfrac12\log5$? The cell refines
   the slices by factors of ten and prints the error beside $0.2\,\epsilon$."
   After: "How fast do the sums approach $\tfrac12\log5$? For $x^2$ the error
   was $\tfrac1{2N}=\tfrac\epsilon2$. If the first order is typical, slices
   ten times thinner should give an error ten times smaller, and the table
   prints the error beside $0.2\,\epsilon$ for comparison."
3. Before: "The next cell builds $F$ as a cumulative left sum ... and compares
   it with $G(x)-G(0)$ at every grid point." (bias explained after the cell)
   After: "The two cannot agree exactly, because the left sum on $[0,x]$
   carries the bias $\tfrac\epsilon2(f(0)-f(x))$. It is largest where $f$
   peaks, at $x=1$ with $f(1)=\tfrac12$, so the largest gap should be
   $\tfrac\epsilon4$." (prediction before the cell)
4. Before: "The opening constant comes from the one-dimensional integral ...
   The value over the whole line comes from a detour through two dimensions,
   in three steps." After: "Surprisingly, the value over the whole line can
   be found all the same, by a detour through two dimensions. The argument
   takes three steps and is the most elegant of the section."
5. Before: "The pathwise estimator does not always have the smaller
   variance." After: "It would be wrong to conclude that the pathwise
   estimator always has the smaller variance."
6. Before: "We can now evaluate an integral exactly when an antiderivative or
   a symmetry is available, and estimate it by sampling ..." After (end of
   the last subsection): "Neither estimator wins everywhere, then. The
   pathwise estimator differentiates $g$ and suffers where $g'$ is large; the
   score-function estimator multiplies $g$ by noise and suffers where $g$
   itself is large."

## Repetition

Removed or reduced to a reference:

- Signed against unsigned change of variables (IC-15): told once, after
  `eq_mdl-change_var_nd`. Removed from Signed Area and from the end of
  Substitution.
- Variance of the sample mean (IC-21, S2): Proposition and proof removed;
  the equation keeps its label `eq_mdl-mc-variance` (exercises 9 and 10 cite
  it) and is stated as "proved as `eq_mdl-var_mean` in
  `sec_mdl-random_variables`".
- Densities (S2): probability of an interval cites `eq_mdl-pdf_int_int`; the
  marginal-density paragraph is gone (Fubini's paragraph names the use);
  "variance and independence are defined there" is said in the opening of
  the part.
- Law of large numbers (IC-22): one clause with `sec_mdl-statistics`.
- Heavy tails: definition removed (IC-10); the finite-variance caveat points
  to chapter 41 and exercise 10.
- Change of a density under a map (IC-18): one sentence naming normalizing
  flows, citing `eq_mdl-cov_density`.
- "We can now ..." recap deleted (S6); its open question (the constant
  $\sigma$, importance sampling by name, `sec_mdl-bayesian-computation`) is
  the end of the last subsection.
- Roadmap cut (IC-3); Dirichlet function to one clause (IC-6); flip counting
  cut (IC-7); the $x^n$ and $e^x$ displays inlined; second statement that
  $\sum i^2$ needed a closed form removed; mean-absolute-error parenthesis
  (IC-23) cut; trapezoid rule dropped from the grid paragraph.
- Positional names (IC-1): no "opening ...", "smallest case", "the X
  subsection", "in words", or "the cell" remain in the body.

Kept because this file owns them: the Gaussian integral (untouched except
its first paragraph, IC-19), the gradient estimators in full, the Leibniz
rule, the Fubini counterexample, the left-rule bias (now derived here).

## Ranking

- Demoted from Proposition/Proof blocks: integration by parts (label
  `eq_mdl-parts` kept on the equation; one-line justification in a
  sentence); variance of the sample mean (above). Kept as blocks: the
  fundamental theorem, the Leibniz rule.
- Ranking sentences added: the three results that carry the first half and
  what can be skimmed (introduction); the left-rule formula as "numerical
  analysis, not theory", kept because it explains three more outputs and is
  the foil for sampling; the swap as what "both gradient estimators below
  rest on"; the comparison of the two estimators as "what this part is for".
- Labels added: none.
- Heading reworded (IC-26, no label on it): "The Leibniz Rule" to
  "Differentiation under the Integral Sign"; the swap is presented as the
  result and Leibniz as the correction when the limits move. "The swap" is
  now introduced as a name (IC-25).

## Split

Marker at line 827, before `## Integration in Probability`:
`<!-- split: new section "Expectations and Their Gradients" starts here -->`.

- Part one now answers both questions of its introduction: the computation
  $P(-1\le Z\le1)=\operatorname{erf}(1/\sqrt2)=0.683$ (prose and display,
  no cell) moved from "Densities and Expectations" to the end of "The
  Gaussian Integral", which closes part one.
- Part two opens on its own problem, $\ell(\theta)=\mathbb E[X^2]$ with
  $X\sim\mathcal N(\theta,1)$ (display moved up from the Leibniz
  subsection), poses the two questions, and says which belongs to chapter 41.
- Candour about quoted theorems split: two named at the start of "Multiple
  Integrals", the third (dominated convergence) where it is used.
- Part one refers forward only by label (`eq_mdl-normal-normalizer` etc.,
  all inside part one) and once by "measured later" for sampling.
- Summary in two bold-led groups, in order.
- For the later move to chapter 41: "Densities and Expectations" and "Monte
  Carlo Estimation" are now short statements with references; the three
  cells there (`integral-density`, `integral-monte-carlo`,
  `mdl-integral-calculus-monte-carlo-estimation-1`) and the grid-against-
  sampling paragraph are what would move. Exercises 9 and 10 cite
  `subsec_mdl-integration-in-probability` and `eq_mdl-mc-variance`.

## Cells to cut or redesign on the development machine

- `mdl-integral-calculus-the-theorem-1` (IC-9, S5): autodiff confirms a
  one-line chain rule; prose reduced to one sentence each side. Cut.
- `mdl-integral-stochastic-gradient-estimators` (S5): confirms
  $\partial_\theta(\theta+z)^2=2x$; it also creates `x`, `z`, `pathwise`
  for the next two cells, so it must be replaced by a NumPy cell, not
  just deleted. Removing both makes the section framework-free.
- `mdl-integral-calculus-monte-carlo-estimation-1` (IC-24): plot of one
  running average; replace by RMS error over about 200 runs, or cut.
- `integral-density`: confirms three proved numbers; prose is one sentence
  each side. Candidate to cut when the subsection moves.
- `mdl-integral-calculus-riemann-sums-1`: its loop confirms a closed form;
  the definition of `left_sum` is needed.

## Other deferred findings

- S7(a) integration by parts in several dimensions: new material, would
  lengthen the section.
- IC-14 (caption): the caption of `fig_mdl-rect-transform` still says "the
  $x$ side (the right-hand side of the formula)"; figure includes were not
  to be touched. The paragraph was rewritten as the review proposed.
- Exercises: 1(a) reworded (the derivation is now in the text) and 11(d)
  reduced to "verify by simulation" (the text proves it), both named by the
  review. Not done: splitting 4(d), retitling 11, renaming $b$, $z$,
  $\sigma$ in 12, grading marks, the two missing exercises.
- S3: the fundamental theorem keeps its full squeeze proof.
- Figure `mdl-cal-signed-area.svg` still illustrates a subsection that is
  now four sentences; small labels in `mdl-cal-cov-jacobian.svg`.

## Claims added or changed (beyond the table above)

- "it [the left-rule formula] accounts for the residue in three more outputs
  of this section" (cumulative check, improper integral, Gaussian grid).
- "The theorem [Fubini] is at work whenever a variable is integrated out of a
  joint density and whenever an expectation changes places with another
  integral."
- "a sample pushed through ..." was dropped; the flows sentence only says
  that this is how a normalizing flow computes the density of its output.
- Pathwise estimator "suffers where $g'$ is large", score-function estimator
  "where $g$ itself is large" (a reading of the two examples, not a theorem).
- $\sigma$ "depends on the distribution the samples are drawn from, and
  choosing that distribution to make $\sigma$ small ... is called importance
  sampling".
- Main-text reference added: `subsec_normal_distribution_and_squared_loss`
  (IC-2).

## Slides against the body

- Cover kicker still lists the section as one; after the split the deck
  covers two sections.
- Slide "Leibniz: Swap Inside, Then Add What the Moving Edge Sweeps" matches
  the body's content but the subsection is now titled "Differentiation under
  the Integral Sign".
- A slide says "the opening's $68\%$" (positional name removed from the body).
- The slide with $\textrm{Var}(\tfrac1n\sum g(X_i))=\sigma^2/n$ presents as a
  result of this section what the body now borrows from chapter 41.
- Not checked line by line beyond a search for the changed numbers
  ($0.55$, "percent or two", "four decades", "$10^{-3}$"): none occur.

## Notation and neighbours

- `mdl-random-variables.md` is being revised in parallel; this file now
  relies on its labels `eq_mdl-pdf_int_int`, `eq_mdl-var_mean`,
  `eq_mdl-cov_density` and on its definition of "heavy-tailed" staying there.
- Exercise 12 still reuses $b$, $z$, $\sigma$ with other meanings than the
  text.
