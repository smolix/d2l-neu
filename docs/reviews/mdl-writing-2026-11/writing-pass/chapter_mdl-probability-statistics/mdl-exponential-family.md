# Writing pass: The Exponential Family

File: `chapter_mdl-probability-statistics/mdl-exponential-family.md`. Not split.
Checker: "OK: only prose changed"; `tools/lint_source.py` clean; no em dashes.

## Words

- Checker count (prose and exercises): 7,880 before, 7,876 after. This counter
  is sensitive to line wrapping of inline math.
- Counted paragraph by paragraph with inline math removed: body 6,258 to
  6,193; with exercises 7,695 to 7,634.
- The section is only slightly shorter. The cuts of the review (probit
  integral, logit against probit, normalizability, Gamma–Poisson, the recap,
  the second telling of the four facts) removed about 400 words; the
  correctness repairs and the ranking sentences put about 340 back.
- Display equations: 38 to 37 (the multi-option probit integral is gone).

## Correctness changes

- **EXP-9 (canonical link).** Was: a GLM "makes its natural parameter a linear
  function of the input", and the Summary said every GLM trains with
  prediction minus observation. Now: the GLM subsection names the choice
  $\boldsymbol\eta=\mathbf W^\top\mathbf x$ as the *canonical link* and says
  the simple gradient belongs to it; the probit model "passes the same linear
  function through a Gaussian CDF, and its gradient carries an extra weight
  that depends on the prediction". Summary: "A generalized linear model with
  the canonical link ...; other links, such as the probit, do not." Verified:
  for the probit loss the derivative with respect to the linear predictor $m$
  is $(\Phi(m)-y)\,\phi(m)/(\Phi(m)(1-\Phi(m)))$. I did **not** say the probit
  loses convexity (its loss is convex), only the simple gradient. Also added
  the reviewer's note that the Gaussian row is the fixed-variance family with
  $h(y)\propto e^{-y^2/2}$.
- **EXP-14 (pseudo-counts).** One new paragraph after the posterior mean states
  the convention: $a$, $b$ are the pseudo-counts of the posterior mean, $a-1$,
  $b-1$ those of the mode (mode $(a+x-1)/(a+b+n-2)$ when both posterior
  parameters exceed 1); Beta(1,1) adds one flip of each kind to the mean and
  none to the mode. Repeated in one clause in the Summary. Exercise 9(b) now
  reads "its one imagined head and one imagined tail in the mode". Checked:
  Beta(15,2) has mode 14/15 = (13+1)/(13+2).
- **EXP-8.** The maximum-entropy proposition now requires a density in the
  continuous case and a finite $\mathbb{E}_q[T]$.
- **EXP-10.** "which requires Cov(T) to be positive definite ... no linear
  combination is constant" is now "which holds when ... no nontrivial linear
  combination of the statistics is constant on the support".
- **EXP-6.** Student-$t$ cites Exercise `ex_mdl-distributions-student-t`. The
  claim that no fixed number of sums summarizes its likelihood is now labelled
  as the Pitman–Koopman–Darmois theorem, stated without proof ("among regular
  families with a fixed support, only exponential families have a sufficient
  statistic of fixed dimension"). Stated from memory, with "regular" as the
  hedge; no citation key added.
- **EXP-11.** The promise "what the curvature tells us" is now kept: the loss
  is about half a nat above its minimum at distance $1/\sqrt{2.77}\approx0.60$
  from $\hat\eta$ (recomputed: 0.46 and 0.53 nats at $\hat\eta\pm0.6$), with a
  pointer to the Fisher information in the maximum-likelihood section.
- **EXP-15.** "the $1/\sqrt n$ law" is now "shrinks like $1/\sqrt n$, as the
  standard error of the Statistics section will".
- **EXP-12 ($T$ twice).** Not renamed (the Gumbel cell prints `T =`, and the
  main text uses $T$). One sentence before the cell says that until the end
  of the softmax part $T$ is a temperature. Logged as a notation clash.
- No finding disputed. EXP-5 treated as the chapter review reads it (not an
  error): the scaling remark was replaced by a forward clause on the redundant
  logit.

## Entropy, KL, Gibbs (author decision 5)

Kept here, once: the definition of KL in one sentence, Gibbs' inequality as a
lemma with the three-line tangent proof, the proposition, and the uniform,
Gaussian and Poisson/geometric corollaries. The lemma is now stated directly
as $D_{\mathrm{KL}}(q\|p)\ge0$ (before, KL was named only after the proof).
One sentence sends the reader to `sec_mdl-information_theory` for entropy and
KL in full. **Label added:** `eq_mdl-exp_family_gibbs` on the lemma's display.
The proposition is still reached by `subsec_mdl-max-entropy-derivation`.

## Six before/after pairs

1. Opening. Before: "The previous section built its catalogue of distributions
   one law at a time: a coin, the number of heads in many flips, ..." After:
   the section opens on "Linear regression, logistic regression and softmax
   regression each come with a loss of their own." and adds "The catalogue of
   :numref:`sec_mdl-distributions`, in which each mean and variance needed its
   own trick, raises the same suspicion."
2. Ranking. Before: "These facts help whenever a new kind of label comes along
   ... The form tells us which loss to train with and guarantees ..." After:
   "The second fact is the one a deep learning reader came for, and the first
   is what proves it ... Two parts can be skipped on a first reading: maximum
   entropy ..., and the derivations of the softmax from noisy choices and from
   physics."
3. Die cell. Before: "How large is $\eta$, and how much entropy does the
   constraint remove?" After: "The die of threes and sixes gave up
   $\log6-\log2\approx1.10$ of the fair die's $1.79$ nats to reach this mean.
   Before running the cell, guess how much the tilt has to give up."
4. Autograd cell. Before: "Does automatic differentiation, applied twice to
   $A$, return these numbers?" After: "Since the moments are derivatives, each
   framework's automatic differentiation computes them for any $A$ that we can
   write as code. We try it on these two cases."
5. Grid cell. Before: "Is the normalized product of prior and likelihood the
   $\mathrm{Beta}(10,5)$ density, and does it have mean $2/3$ ...?" After:
   "Conjugacy spared us the integral in Bayes' rule. Without it we would
   multiply the prior by the likelihood on a fine grid of $\theta$ and
   normalize the sum, which here serves as a check of the closed form."
6. GLM. Before: "prediction minus observation, whatever the family. The loss
   is convex in $\mathbf W$ ..." After: "... whatever the family. This is the
   common rule that the opening asked about, and with the moment property in
   hand it took two lines."

Also: "The sigmoid of logistic regression is thus nothing more than the map
from the coin's natural parameter back to its mean. Nobody had to choose it."

## Repetition

- Four facts: now in the introduction and the Summary only. The second
  paragraph's restatement and the "We can now" recap are gone.
- One ending: the recap is deleted. The last subsection ends on what is
  missing (posteriors outside the conjugate pairs; no error bar for $9/13$)
  and hands over to Statistics.
- Summary: nine sentences to six.
- Cut: the multi-option probit integral (to two sentences); logit against
  probit (to two sentences, Exercise 4 has the rest); the normalizability
  paragraph (EXP-16); the Gamma–Poisson pair to four sentences with a pointer
  to Exercise 7 (the label `subsec_mdl-other-conjugate-pairs` stays, the
  exercise cites it); the variance-$\sigma^2$ and softmax-gradient retelling
  of the main text in the GLM subsection; "appear later in the book"
  (EXP-17), replaced by a pointer to naive Bayes for add-one smoothing.
- Kept because this file owns them: the Beta posterior, pseudo-counts and
  Laplace's rule; Gibbs and maximum entropy (above); the Gumbel proof (the
  main text points here).

## Ranking of results

No Proposition block was demoted. The five blocks (Gibbs lemma, maximum
entropy, moment property, Gumbel, conjugate prior) are each cited or among
the four facts. Ranking is done in prose: introduction (which fact matters,
what can be skipped), GLM subsection, start of the softmax part ("The third
is the shortest, and the only one that the rest of the section uses").

## Cells to cut or redesign on the development machine

- `distributions-conjugate` (grid posterior against Beta(10,5), agreement
  $2\times10^{-14}$): confirms two lines of algebra. Overview finding 4; no
  EXP id (the section review praised it). Redesign as a non-conjugate prior
  on the same grid, or cut.
- `distributions-exp-family` (autograd twice on $A$, four tabs): confirms the
  moment property to four digits. Keep at most one of this and the previous.
- `mdl-exponential-family-noisy-choices-and-gumbel-noise`: confirms the Gumbel
  proposition; now presented as the sampler in use. Keep.
- EXP-3: the PyTorch imports cell loads `nn`, `F`, `d2l` unused, and there is
  an MXNet-only `nd` import cell. I deleted the housekeeping sentence about
  NumPy; the cells are untouched.

## Deferred

- **Reorder (structural findings 1, 2):** moving maximum entropy to the end
  would move the die cell past three others, so it was not done. The
  introduction now says that part can be skipped.
- EXP-17 in full (drop Gamma–Poisson to the exercise): kept short because
  Exercise 7 cites the subsection for the Gamma posterior.
- Exercise 3(a) repeats the text; Exercise 1 against the main-text Gumbel
  exercise; seven `ex_mdl-distributions-*` labels: untouched.
- Student-$t$ local minima: the reviewer's numbers were not added (I
  reproduced minima at $\pm4.69$ for $\nu=3$, clusters at $\pm5$).

## Claims added or changed (all checked)

- Die of threes and sixes gives up $\log 3\approx1.10$ nats; the tilt 0.18
  (stored output: $H=1.614$, $\log 6=1.792$).
- Probit gradient has an extra prediction-dependent weight.
- Loss about half a nat above the minimum at distance 0.60.
- Mode of the Beta posterior and the two pseudo-count readings.
- Pitman–Koopman–Darmois statement (from memory).
- Gaussian GLM row: $h(y)\propto e^{-y^2/2}$.
- Gumbel frequencies differ "by a few thousandths at most" (stored: 0.001;
  Monte Carlo standard error about 0.0015).
- "the classifier of naive Bayes uses the case $K=2$" of add-one smoothing.
- Numbers kept that no cell prints, recomputed: 8.02 nats, 0.617, 2.77,
  $|\mathrm{sigmoid}(1.7x)-\Phi(x)|<0.01$ (max 0.0096), 0.118, 0.040, 2.9,
  average logit 1.49.

## Neighbouring files

- `mdl-maximum-likelihood.md` ("Beta Prior and Pseudo-Counts") says "The prior
  acts as $a-1$ pseudo-counts of heads and $b-1$ of tails" without "for the
  mode". It agrees with the convention here once it says so and cites this
  section.
- The remark on $p$ and $\theta$ now reads "we write it $\theta$, as the
  following sections do"; Statistics has its own remark (STA-3).
- Cross-file exercise reference added: `ex_mdl-distributions-student-t`.

## Slides against the body

- Slide "Every GLM Trains by Input Times (Prediction $-$ Observation)": true
  for the canonical link only; the slide does state
  $\boldsymbol\eta=\mathbf W^\top\mathbf x$ but the title says "Every GLM".
- Slide with "$a$ and $b$ are pseudo-counts": correct for the mean; the
  body now adds the mode convention.
- Slides still show the Lemma in its old form ($\sum q\log(p/q)\le0$), if at
  all; not checked line by line.
