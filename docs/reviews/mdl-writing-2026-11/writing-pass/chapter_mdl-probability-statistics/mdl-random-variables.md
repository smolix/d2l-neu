# Writing pass log: Random Variables

Source: `chapter_mdl-probability-statistics/mdl-random-variables.md`.
Checker: "OK: only prose changed", no em-dash warning; `tools/lint_source.py`
clean.

## Words

Prose words (checker count, body and exercises to the slides marker):
10,892 before, 10,937 after (+45). The cuts removed about 700 words; the
additions are the repairs of RV-1, RV-5, RV-8, RV-9, RV-10, one sentence that
closes the proof of the one-dimensional change of variables, the two
"missing" items (Jensen pointer, convolution), and the stand-alone opening of
the second half. Without the correctness repairs the section is shorter. If
strict non-growth is wanted, the four-cell covariance example (about 150
words, used again only for $\rho=2\theta-1$) is the candidate to cut.

## Correctness changes

- **RV-1** (two definitions of "has a density", near-circular $F'=p$).
  Before: a `**Definition.**` by the limit of $P(x\le X<x+\epsilon)/\epsilon$,
  then the integral rule "derived" by a Riemann sum, then a Proposition
  $F'=p$ that assumed the integral form. Now: the limit is introduced in
  "From Bins to a Density" as the density at a point ("wherever this limit
  exists", with the remark that the definition proper follows); "Probability
  as Area" says the Riemann-sum argument is a motivation and not a proof
  (about $1/\epsilon$ pieces, errors must be uniformly small) and then states
  the **Definition** by the integral property `eq_mdl-pdf_int_int`; the dart
  is checked against it ($\int_a^b 2r\,dr=b^2-a^2$). The Proposition gains a
  second statement (the ratio of the bins tends to $p(x)$ where $p$ is
  continuous), with a proof: FTC, plus points have probability zero because
  $P(X=x)\le\int_{x-\delta}^x p\to0$, so $P(x\le X<x+\epsilon)=F(x+\epsilon)-F(x)$.
  Verified by hand. The joint density in the second half is worded the same
  way (ratio as description, integral as definition).
- **Proof of the one-dimensional change of variables** (consequence of
  RV-1): added the closing step that $F_Y$ is continuously differentiable on
  $g(I)$, which carries all the probability, so its derivative is a density
  by the fundamental theorem of calculus.
- **RV-7**: "are the extreme case" is now "are the standard example".
- **RV-9**: additivity $\textrm{Cov}(X+Z,Y)=\textrm{Cov}(X,Y)+\textrm{Cov}(Z,Y)$
  added with its one-line reason.
- **RV-10**: "That $\bar X_n$ converges to the mean is the law of large
  numbers. The rule needs a finite variance ..." is now "The $\sigma/\sqrt n$
  rule needs a finite variance. The law of large numbers itself ... needs
  only a finite mean (:numref:`sec_mdl-statistics`). The Cauchy distribution
  has neither ...". Checked against `mdl-statistics.md` ("a finite mean
  suffices").
- **RV-14**: the closing question is no longer assigned to the law of large
  numbers; it now points to `subsec_mdl-gaussian-clt` (the Gaussian value)
  and `sec_mdl-concentration-generalization` (the exponential rate).
- **RV-4**: "In a picture:" is now "Geometrically,".
- **RV-5**: the Gaussian density formula and a reference to
  `subsec_mdl-gaussian-clt` at its first use (the Chebyshev table).
- **RV-13**: the integral change-of-variables theorem is stated once, in the
  letters used afterwards ($\mathbf x=g^{-1}(\mathbf y)$, regions
  $g^{-1}(\mathcal D)$ and $\mathcal D$); the auxiliary map
  $\boldsymbol\phi$, $V$, $W$ is gone.
- Three-atom example (RV-3): now requires $0<q\le\tfrac12$ (was
  $0\le q$), since Chebyshev's proposition needs nonzero variance.
- Log-normal cell: "for a net extra area of $0.65$" after the cell became a
  prediction before it; after the cell the text now says the naive curve is
  the true density times $y$ (so below for $y<1$, above for $y>1$).
  Recomputed: deficit 0.238 on $y<1$, excess 0.887 on $y>1$, net 0.649.
- Exercise 1(d): "$\mathbb{E}[X]$" now reads "the mean of the non-negative
  variable $X$ of part (a)". Exercise 5(c): the measure-zero covering
  argument replaced by the two-line route the review proposed (the pair lies
  on the lines $y=\pm x$; a double integral over a line is zero).

No finding was rejected. **RV-8 is only half done, see Deferred.**

## Six before/after pairs

1. Before: "We can also check the ratio by experiment, without the area
   formula. ... Does the fraction of darts in a bin, divided by the bin's
   width, settle as the bin narrows?"
   After: "The table came from the area formula. A simulation needs no
   formula. ... We count the fraction of darts in each of the three bins of
   the table and divide by the bin's width."
2. Before: "Do the samples $\sqrt U$ match the darts thrown at the board, and
   does the naive $R=U$ fail?"
   After: "We now have two ways to produce the dart's distance, the darts
   thrown at the board above and the samples $\sqrt U$. The cell compares
   them, with the shortcut alongside."
3. Before: "What happens when we average many Cauchy draws? The next cell
   plots the running mean ... Its last line repeats the average of $1000$
   Cauchy draws $1000$ times and reports how often that average lies farther
   than $1$ from $0$."
   After: "With no mean to converge to, what does an average of Cauchy draws
   do? One might still hope that an average of $1000$ draws lies closer to
   $0$ than a single draw, which lies farther than $1$ from $0$ with
   probability $P(|X|>1)=1-\tfrac2\pi\arctan1=\tfrac12$."
4. Before: "Do the two parts add up to the total in a sample?"
   After: "In a sample of a million draws the two parts will come out near
   $1$ and $2.56$. Should their sum match the variance of the sample only as
   roughly as the parts match their targets, or exactly?"
5. Before: "To measure spread we average the distance from the mean. The
   deviation $X-\mu_X$ is sometimes positive and sometimes negative and
   averages to zero, so we square it before averaging."
   After: "The mean says nothing about spread. As in :numref:`sec_prob`, we
   measure spread by the average squared distance from the mean, the
   *variance*".
6. Before: "The first uses only the mean, and applies to variables that are
   never negative."
   After: the same, followed by "Its proof is one line, and nearly every tail
   bound in this chapter is obtained by applying it to a well-chosen function
   of the variable."

## Repetition

Removed:
- "A density is a rate, not a probability": the opening of the c.d.f.
  subsection and the sentence after the proof of $F'=p$. Kept once, the speed
  analogy. The caption of `fig_mdl-prob-pdf-area` still says it (captions
  were not touched; see Deferred).
- "Uncorrelated does not imply independent": the forward remark closing the
  independence subsection and the discrete $X=Y^2$ example (now a clause
  citing `sec_prob`). Kept: the dart under "Covariance", panel (c) of the
  correlation figure, one clause in the Summary. The sentence under Eve's law
  ("the best predictor ... is the constant $0$, although $X$ and $Y$ are
  dependent") stays; it is on the review's list of best passages and says
  something different.
- Against `sec_prob`: the profit example for "the mean alone is not enough",
  the motivation of the squared deviation, the units paragraph (now one
  clause), the sign reading of covariance, the cents-and-dollars sentence
  under Correlation (the figure paragraph under Covariance keeps it once),
  the rod with discrete masses.
- Three-atom example: introduced once, where Chebyshev needs it; its mean,
  variance and standard deviation are no longer computed in three earlier
  places. The offset $a$ stays because the figure and its caption use it.
- RV-12: the PCA / multivariate Gaussian / elliptical contours paragraph is
  one sentence citing `sec_mdl-svd-low-rank` and `sec_mdl-distributions`.
- The "We can now compute ..." recap is deleted; the open question it held is
  the ending.
- Dart mean $0.667$, quartiles, and "three quarters beyond $0.5$" are each
  derived once and cited where needed.

Kept because this file owns them: the Cauchy averaging experiment with the
probability $\tfrac12$ and "the average of $n$ independent Cauchy draws is
again a standard Cauchy variable"; Markov's inequality `eq_mdl-markov` with
proof; the dart under "Covariance"; the speed analogy.

## Ranking

Demoted from Proposition/Proof blocks to a sentence with its labelled
equation (all labels kept): computational form of the variance
(`eq_mdl-var_comp`), variance under affine maps (`eq_mdl-var_affine`),
variance of a sum (`eq_mdl-var_sum`), the independence equivalence, and the
Corollary to Chebyshev (now an unlabelled display, as before). Blocks: 14
before, 9 after (F'=p, inverse-transform sampling, linearity, Markov,
Chebyshev, total expectation and variance, best predictor, correlation
bound, one-dimensional change of variables).

The opening now says what repeats `sec_prob` and what is new, and names the
two results of the first half to remember (`eq_mdl-cdf_deriv`,
`eq_mdl-markov`); the opening of the second half names its three
(`eq_mdl-tower`, `eq_mdl-best_predictor`, `eq_mdl-cov_density`) and says the
covariance matrix and the cosine remark can be skipped on a first reading.
The best-predictor proof says the cross term is the only step that needs
thought.

Labels added: none.

## Split

One marker, line 696, before `## Several Variables`:
`<!-- split: new section "Joint Distributions and Transformations" starts here -->`

To make the halves stand alone:
- The first half ends on "Heavy Tails" (heavy-tailed gradient noise and the
  exercise that tests it).
- The second half opens by restating the dart in one sentence (uniform on the
  board, distance density $2r$), its question, its three main results.
- Backward dependencies removed: "the three-atom example ... tail bounds
  below" (moved); "we prove this ... in the covariance subsection below" (now
  cites `sec_prob` and `subsec_mdl-covariance` by label); "the log-normal
  distribution at the end of this section" (now cites
  `subsec_mdl-change-of-variables`); "promised earlier" (twice, gone);
  "for the third time in this section" (now "once more").
- Summary: two paragraphs, "One variable." and "Several variables.".
- Remaining after the split, for the development machine: (a) the proof of
  linearity for densities says the joint density is defined "when we come to
  several variables"; (b) the opening's sentence "The second half of the
  section ..." must go; (c) the second half needs its own imports cell and
  `rng` (its two cells use `rng`, `np`, `d2l` from the first half);
  (d) the second half recalls the dart, $R=\sqrt U$ and the Cauchy without a
  `:numref:` to the first half, because a self-reference would be wrong
  today; add `sec_mdl-random_variables` references after the split;
  (e) the mixture example refers to `fig_mdl-prob-pdf-area`, which stays in
  the first half.

## Cells

Cells to cut or redesign on the development machine:
- `random-variables-imports` (RV-15): four tabs of unused framework imports;
  one untabbed NumPy cell would do.
- None of the five computing cells is a pure confirmation by the review's
  account. Closest: `random-variables-density-to-probability` confirms the
  ratio of the table; the prose now reads it for what the table lacks (the
  noise of a narrow bin) and it also creates `r_dart` for the next cell.
- `mdl-distributions` has the duplicate Cauchy cell (RV-6 / DIST-11); this
  file owns the experiment.

## Deferred

- **RV-8 (target $X$, input $Y$): not swapped.** A code cell and a figure
  depend on the letters: `mdl-random-variables-conditional-expectation`
  prints `Var(X)` and comments `E[X | Z]`, and
  `fig_mdl-prob-conditional-slice` draws $p(x\mid y_0)$. A swap confined to
  the theory would leave $X$ as input in the propositions and as target in
  the mixture example three paragraphs later. Applied the review's fallback:
  one clause at the top of "Conditional Expectation" saying which letter is
  which and that the main text writes the pair the other way round. A full
  swap needs the cell's print string, the figure, exercise 10 and three
  slides changed together.
- RV-2, fourth instance: the caption of `fig_mdl-prob-pdf-area` ("The density
  itself is not a probability ...") left as is (figure includes untouched).
- RV-4: a small figure for Markov (step under the line $y=x$) not added.
- RV-15: imports cell.
- Structural 5(c): the Gaussian is still used (running-mean cell, mixture)
  before `sec_mdl-distributions`; only the formula and reference were added.
- Leftover label `ex_mdl-random-variables-using-textrmvar-x-y` not renamed
  (labels may not be renamed in this pass).
- Exercise 5 against the neighbour's use of $(X,SX)$: untouched here.

## Claims added or changed

- A density is defined by $P(a<X\le b)=\int_a^b p$; the Riemann-sum argument
  needs uniformly small errors.
- $P(X=x)\le\int_{x-\delta}^x p\to0$; the ratio $P(x\le X<x+\epsilon)/\epsilon\to p(x)$
  at points of continuity.
- "Nearly every tail bound in this chapter is obtained by applying
  [Markov's inequality] to a well-chosen function of the variable" (true of
  Chebyshev, Cantelli, Chernoff/Hoeffding; a judgment).
- "At most $1\%$ of any distribution lies ten or more standard deviations
  from its mean" (was the same with "$\sigma_X>0$" spelled out).
- $\mathbb{E}[g(X)]\ne g(\mathbb{E}[X])$ in general; Jensen for convex $g$
  (`subsec_mdl-jensen`).
- Convolution: $(X,X+Y)$ has joint density $p_X(x)p_Y(s-x)$ at $(x,s)$ for
  independent $X,Y$, so $p_{X+Y}(s)=\int p_X(x)p_Y(s-x)\,dx$ (stated without
  proof of the first step; the shear has Jacobian 1).
- Covariance is additive in each argument.
- The law of large numbers needs only a finite mean; the Cauchy has neither
  mean nor variance.
- `sec_linear_regression` reads its model as $\mathbb{E}[y\mid\mathbf x]$
  (checked, line 138 there); `sec_prob` states $-1\le\rho\le1$ and has the
  $Y=X^2$ example (checked).
- Eve's-law cell: "sampling noise in the third decimal" (stored output
  1.001 and 2.560; standard errors about 0.002 and 0.004).
- Naive log-normal curve equals the true density times $y$.
- Numbers quoted that no cell prints, all recomputed and kept: 3.1 million
  darts on the board and about 3,000 in the narrowest bin; Gaussian tails
  4.6% and 0.27%; Cauchy truncated second moments 5.43, 62.7, 636 and the
  limit $\log2/\pi\approx0.221$; dart Chebyshev value 0.038; 556 darts;
  $e^{1/2}\approx1.65$.

## Slides against the body

No slide now contradicts the body. Differences of presentation only: the
slides still introduce the three-atom example and $\mathbb{E}[X\mid Y]$ in
the body's letters (unchanged there), and one slide repeats "A density is a
rate, not a probability".
