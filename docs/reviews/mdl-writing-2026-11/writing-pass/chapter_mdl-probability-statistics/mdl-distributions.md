# Writing pass log: Distributions

Source: `chapter_mdl-probability-statistics/mdl-distributions.md`. Prose only;
no code cell, cell ID, figure include or slide changed. Not split, no markers.

## Words

Prose words (checker count): 8494 before, 8412 after (-82). The checker ends
"OK: only prose changed", no em-dash warning; `tools/lint_source.py` is clean.

## Correctness changes

- **DIST-9** (sketch called a theorem; sum from a non-integer mean). The text
  summed steps "from the mean to $np+x$" with $\sum_{j=0}^{x-1}$, which has no
  meaning at $np=6.5$, and then said "This is the de Moivre–Laplace theorem".
  Now: the approximate log-ratio $-x/(npq)-1/(np)$ is compared with the step of
  the parabola $-x^2/(2npq)$ between $x$ and $x+1$, which is
  $-x/(npq)-1/(2npq)$; the two differ by the constant $(p-q)/(2npq)$, zero for
  a fair coin and of order $1/\sqrt n$ in total over $\sqrt{npq}$ steps. This
  holds on any lattice, integer or half-integer. The text then says "this is a
  sketch and not a proof. It is the idea behind the de Moivre–Laplace theorem
  ... which makes the approximation precise for $x$ of order $\sqrt{npq}$."
  Verified: algebra by hand, and numerically ($n=1000$, $p=0.3$: the difference
  between the approximate log-ratio and the parabola step equals
  $(p-q)/(2npq)$ at four offsets; the normalized bell is within 0.01 in log of
  the exact pmf out to two standard deviations).
- **DIST-10 / structural 1** (normalizer and variance used about 90 lines
  before they are derived). Prose reordered; the three cells keep their order.
  New order: log-ratio and bell; normalizer (polar-coordinate integral, now
  motivated: probabilities sum to one, spacing 1, so
  $e^C\sqrt{2\pi npq}=1$); worked coin value and cell
  `mdl-distributions-gaussian-2`; standard Gaussian with its variance;
  $\mathcal N(\mu,\sigma^2)$ with "the two parameters are the first two
  moments" stated where the density is defined; central limit theorem; sums of
  uniforms (cell); finite-variance caveat (Cauchy cell); squared error. The
  squared-error paragraph stays last (the review would put it before the
  theorem) because the Laplace subsection opens from it.
- **DIST-14 / structural 2** (two propositions proved in a loop). Now three
  steps: "Proposition (invertible linear maps)" with the change-of-variables
  proof; "Proposition (residual and marginal)", which cites it; "Corollary
  (wide matrices)" in the Conditioning subsection, from both. The sentence
  "without circularity" is gone. The sampler only needs the square case, and
  the text now says that a square factor $\mathbf L$ with
  $\mathbf L\mathbf L^\top=\boldsymbol\Sigma$ is invertible. Checked: the
  completed matrix maps $\mathbf x$ to a Gaussian whose first block has mean
  $\mathbf A\boldsymbol\mu+\mathbf b$ and covariance
  $\mathbf A\boldsymbol\Sigma\mathbf A^\top$; reordering blocks is an
  invertible map.
- **DIST-6**. "each tenfold increase in $n$ cuts it about tenfold" (the values
  are 20, 100, 1000) is now "on these three values it falls in proportion to
  $1/n$".
- **DIST-12**. "assumes the noise $y\sim\mathcal N(\hat y,\sigma^2)$ has the
  negative log-probability" is now "models the label as ... assigns the
  observed label the negative log-probability". Same repair in the Laplace
  paragraph, where "the absolute error plus a constant" is now "the absolute
  error divided by $b$ plus a constant". The unreferenced sentence on
  generative-model priors is cut.
- **DIST-8**. The Gaussian no longer opens with "counting orderings there is
  hopeless" (the ratio had already made that false); it opens with the
  reviewer's question about the shape.
- **DIST-5**. The two Poisson assumptions are stated as assumptions, and the
  overdispersion remark now says which of them fails.
- **DIST-3**. One short paragraph separates the fair coin of the question from
  the plug-in coin $p=9/13$ of the Bernoulli cell; "our coin" in the binomial
  subsection is "the fair coin".
- **DIST-16**. "likelihood" is no longer used before the chapter defines it
  (closing paragraph and the Laplace median sentence).
- Stored output: the Gaussian fraction at $n=10$ in the Cauchy cell is 0.0022
  in the store, exact value 0.0016; prose says "about 0.002".

No finding I disagreed with. Numbers kept that no cell prints, all
recomputed: $(1-4/n)^n$ = 0.0115, 0.0169, 0.0182; median wait 10.4 s;
$e^{-3\sqrt2}\approx0.014$ against 0.0027 (ratio 5.3); bell value 0.0846;
variance 42.25 of the copied flip; $P(T_4>1)\approx0.43$ (also printed).

## Six before/after pairs

1. Before: "How likely are 9 heads in 13 fair flips? The cell builds the pmf by
   the ratio, in a function we reuse below, and compares it with direct
   counting." After: "The recursion takes three lines of code. We keep it as a
   function, because the Poisson limit and the Gaussian bell below are both
   measured against it."
2. Before: "Does one simulated stream of events give both laws at once? The
   cell draws $60{,}000$ exponential gaps at rate $4$ ..." After: "If the
   construction is right, it can be run backwards: draw exponential gaps, and
   Poisson counts must come out. The cell draws $60{,}000$ gaps at rate $4$
   ..."
3. Before: "Does the residual argument hold on samples? The cell reuses the
   Cholesky samples, forms $r=x_1-\tfrac12x_2$, and then keeps only the samples
   whose $x_2$ is within $0.05$ of $2$." After: "Conditioning can also be done
   on samples, with no formula at all: keep only the samples whose $x_2$ is
   within $0.05$ of $2$ and look at their $x_1$."
4. Before: "We found $P(X=9)$ for the coin by counting 715 orderings. A test
   set of $10{,}000$ examples also has a binomial number of errors, and
   counting orderings there is hopeless." After: "The successive ratio computes
   any binomial probability, but it does not tell us what the distribution
   looks like. Is there a formula for its shape near the peak? There is, and it
   turns out that only the mean and the variance of the coin enter it. This is
   the most important distribution of the section."
5. Before: "**The normalizer.** Why the constant $1/\sqrt{2\pi}$? Let
   $I=\int e^{-x^2/2}\,dx$." After: "The function $e^{-x^2/2}$ has no
   elementary antiderivative, and the integral looks hopeless. The trick is to
   compute $I^2$ instead."
6. Before: "The proof used only square invertible maps. Its second claim ...
   therefore completes the proof of the linear-map proposition for wide
   matrices without circularity." After: "The one idea of the proof is the
   choice of $\mathbf B$, which makes the residual uncorrelated with
   $\mathbf x_2$."

## Repetition

Removed or reduced:
- The four losses: gone from the opening and the Summary; each stays in its
  subsection, now with a main-text label (`subsec_softmax-regression-loss-func`,
  `subsec_normal_distribution_and_squared_loss`, `subsec_linreg-loss-menu`).
- Bernoulli moments and "the fair coin is least predictable": one sentence
  citing `sec_prob`; the display is gone.
- Cauchy averaging: the paragraph is two sentences citing
  `sec_mdl-random_variables`; the cell stays.
- "High dimension": replaced by two sentences citing
  `sec_mdl-concentration-generalization`.
- Roadmap lists at the head of "Discrete Distributions" and "Continuous
  Distributions" (the opening and the family tree already give them).
- The "We can now" half of the closing paragraph; the two open questions stay
  as the end of the last subsection. Summary cut from seven sentences to five.
- "law" is "distribution" throughout the body and the table header, except
  "law of rare events" and the laws of total expectation and variance.

Kept because this file owns them: the Poisson/exponential/Gamma construction,
the bell from the ratio, the multivariate Gaussian, and the Student-$t$
exercise. Exercise 7 now says where the distribution is used
(`sec_mdl-statistics`), points to `sec_mdl-random_variables` for the Cauchy,
and states the law of total variance itself instead of citing exercise 3.

## Ranking, blocks, labels

- Opening names the three results that matter (Poisson limit with waiting
  times, central limit theorem, Gaussian conditioning) and three things to
  skim (discrete uniform, Gamma, Laplace).
- No Proposition demoted. The linear-map proposition is restated for
  invertible maps, and the wide case is a Corollary. Both propositions now
  carry names.
- Labels added: `subsec_mdl-mvn-contours`, `subsec_mdl-mvn-linear-maps`,
  `subsec_mdl-mvn-conditioning` (structural finding 4).
- Added under the summary table: one sentence saying where the Cauchy,
  log-normal, Beta, Dirichlet, negative binomial, geometric and Student-$t$
  live (structural finding 5).
- Added: one clause naming $\boldsymbol\mu+\mathbf L\mathbf z$ as the
  reparameterization, citing `subsec_mdl-stochastic-gradient-estimators`
  (DIST-15), and `chap_gp` for Gaussian process regression.

## Cells to cut or redesign on the development machine

- `distributions-bernoulli` (DIST-4, structural 3): prints $p(1-p)$ at three
  values and a frequency. Cut, or use the fair coin.
- `distributions-categorical`, last line (DIST-7): $13!/(9!\,4!)=715$. Drop
  the line.
- `distributions-discrete-uniform` (DIST-7): sums against closed forms derived
  from the same sums. Cut, or show $h^2(n^2-1)/12\to(b-a)^2/12$.
- `distributions-continuous-uniform` (structural 3): equal-length intervals of
  a uniform sample. Cut.
- `distributions-gaussian` (DIST-11): repeats `random-variables-cauchy-diverges`.
  Cut; the Gaussian comparison follows from $\mathcal N(0,1/n)$.

## Other deferred findings

- Structural 4: give the Gamma distribution and the Poisson process their own
  headings (heading changes left to the author).
- Structural 7: the multivariate Gaussian as its own section (not to be split
  in this pass).
- DIST-1 in figure captions: the family-tree caption still says "law" twice
  (figure includes were not to be touched).
- DIST-17: unused framework imports; a `log_prob` cell; legend over the $n=1$
  plateau in the stored plot.
- Exercise 5(a) needs "the density of a sum is a convolution", which the
  chapter never states (belongs to Random Variables or the exercise).
- Exercise 2's leftover label `ex_mdl-distributions-using-conditional-formula-eqref`
  (labels may not be renamed).
- The review counts 33 bold run-in headings; two were removed ("Derivation.",
  "Construction."), the rest stay.

## Claims added or changed

- The parabola-step identity and the size $(p-q)/(2npq)$ of the neglected
  term (above).
- "Counts are spaced one apart, so the sum of the bell over all counts is
  close to the integral" (heuristic, labelled as part of the sketch).
- "$e^{-x^2/2}$ has no elementary antiderivative."
- "If a square matrix $\mathbf L$ satisfies
  $\mathbf L\mathbf L^\top=\boldsymbol\Sigma$, it is invertible."
- Overdispersion means "the rate varies from minute to minute, or requests
  trigger further requests".
- Student's $t$ is what `sec_mdl-statistics` uses for the mean of a small
  sample.
- Opening: which results matter and which can be skimmed (a judgment).

## Slides that now disagree with the body

- The slides use "law" throughout; the body says "distribution".
- The slide on linear maps states the result for full row rank in one bullet;
  the body now has the invertible case as the proposition and the wide case as
  a corollary (no contradiction, different packaging).
- The Cauchy check slide gives the full argument that the body now leaves to
  `sec_mdl-random_variables`.
- The opening and closing slides list the four losses, which the body's
  opening and Summary no longer do.
