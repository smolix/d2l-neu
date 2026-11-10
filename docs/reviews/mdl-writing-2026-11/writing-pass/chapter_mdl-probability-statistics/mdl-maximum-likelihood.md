# Writing pass: Maximum Likelihood

File: `chapter_mdl-probability-statistics/mdl-maximum-likelihood.md`.
Checker: "OK: only prose changed", no em-dash warning; `tools/lint_source.py`
exits 0. `.venv-pytorch/bin/python` is missing on this machine; numerical
checks were run with `python3` and NumPy 2.5.1 in the scratchpad.

## Words

Prose words (checker count, body and exercises): 10,658 before, 10,626 after
(-32). The body is shorter by more than that in the repeated material (about
450 words cut) and longer by about 420 words of additions the review asked
for (misspecification worked with one number, what survives for a network,
the held-out floor, the support remark on the ELBO, ranking sentences). If
the section must shrink further, the next candidates are the two-parameter
Gaussian Fisher paragraph and the third EM limit.

## Correctness changes

- **ML-3.** The float32 aside ("stops growing at $2^{24}$, after about twenty
  million of the billion terms") is cut. The reviewer measured about 23
  million; the aside was about summation, not likelihood.
- **ML-6.** "so it is not the NLL of any conditional distribution" now reads
  "..., even up to an additive constant". Agrees with exercise 4.
- **ML-9 / ML-8.** Was: "mean score within a few hundredths of 0 ... within
  about 0.02 of 0.5 ... Monte Carlo error ... about 0.01". Now the Cauchy
  information is derived by $u=\tan\varphi$ with $\varphi$ uniform,
  $s=\sin2\varphi$, $I=\mathbb{E}[\sin^22\varphi]=1/2$, and the standard
  errors $0.007$, $0.004$, $0.008$ are stated before the cell; after it,
  "within two or three of these standard errors". Verified by simulation
  ($10^7$ draws: sd of score 0.707, of score squared 0.354, of curvature
  0.791; $|s|\le1$). Stored output: 0.004, 0.495, 0.501.
- **ML-10.** "In general, a training run can compute the curvature ..." now
  begins "For a model that satisfies the conditions above, the same holds
  approximately at large $n$", and adds that a deep network does not satisfy
  them, with a pointer to `subsec_mdl-mle-limits`.
- **ML-14.** "The figure also shows the two limits" is now "The same picture
  explains the two limits".
- **ML-15.** The condition "$q(z)>0$ exactly where the posterior is positive"
  is removed from the definition of the ELBO. New sentence: the sums run over
  $q(z)>0$, and if $q$ vanishes where the joint is positive the first
  equality becomes $\ge$. The gap proposition now assumes only that $q$ puts
  no mass outside the posterior's support (so the KL term is finite). Checked
  by hand.
- **Gradient-descent cell.** "lands on $0.97131$ ... to the precision of
  single-precision arithmetic" is now "$0.9713$, ... to the digits printed"
  (the stored output prints four digits).
- The regularity bullet "the Fisher information defined below" is now "the
  Fisher information" (the list moved after the definition, ML-7).

Disagreed with: none. Not adopted as proposed: ML-13 asked to delete the
subsection "Posterior Mode and Posterior Distribution"; the brief forbids
removing headings and labels, so the subsection stays and now holds the one
account of parameterization dependence (see below).

## Six before/after pairs

1. Before: "Every model in this book was trained by minimizing a loss:
   squared error for regression, cross-entropy for classification."
   After: "Suppose we flip a coin thirteen times and see nine heads and four
   tails." (ML-18; the losses come in the second paragraph.)
2. Before: "Thus the MLE of a coin's bias is the observed frequency of
   heads." After: "The MLE of a coin's bias is the observed frequency of
   heads, as anyone would have guessed."
3. Before: "Does more data move the peak, or does it sharpen it? The cell
   below plots the likelihood of $9$ heads in $13$ flips and of $90$ heads in
   $130$ flips, ..." After: "Suppose the same frequency had come from ten
   times as many flips, $90$ heads in $130$. The estimate cannot change,
   since only the ratio enters it. What changes is how firmly the data single
   it out."
4. Before: "Do these identities hold on data? The cell below draws
   $10{,}000$ Cauchy observations ..." After: "A model without such a trick
   still yields both forms of $I$ to automatic differentiation. The cell
   draws $10{,}000$ Cauchy observations ... By the uniform angle, their Monte
   Carlo standard errors are about $0.007$, $0.004$ and $0.008$."
5. Before: "Thus the error of the MLE is approximately Gaussian, ..."
   After: "We do not prove it beyond the three steps above. It is the central
   result of the section: the error bar, the efficiency of the MLE and the
   limits of the theory all hang on it. The error of the MLE is approximately
   Gaussian, ..."
6. Before: "Does EM climb, and how fast? We start from a poor guess, ...
   check that the trace never decreases, ..." After: "The proposition says
   that the trace cannot go down. It says nothing about speed. We start from
   a poor guess, ..."

"Thus" opening a gloss: 13 before, 0 after.

## Repetition

Removed or reduced to a clause with a reference:

- Entropy, KL and Gibbs (S2, author decision): the definitions are recalled
  in one sentence with `subsec_mdl-max-entropy-derivation` and
  `sec_mdl-information_theory`; the Proposition and tangent-line proof are
  gone; Gibbs is cited as `eq_mdl-exp_family_gibbs` here, in the consistency
  argument and in the ELBO-gap proof. Kept (owned here): the regrouping
  identity, $\textrm{CE}=H+D_{\textrm{KL}}$ in one display, the coin's 0.617,
  0.076 and 0.164 nats, the figure.
- Gaussian NLL (S3): stated as a running derivation with a reference to
  `subsec_normal_distribution_and_squared_loss`; `eq_mdl-gaussian-nll` kept.
  The CLT justification is left to the main text.
- Gaussian prior (S3): one derivation in running text with a reference to
  `sec_weight_decay` and `eq_wd-objective`; the two displays of the old proof
  and the separate averaged-loss display are merged.
- Beta pseudo-counts (EXP-14): the text now says "for the mode, the prior
  counts as $a-1$ imaginary heads and $b-1$ imaginary tails" and cites the
  convention of the exponential-family section ($a$, $b$ for the mean); the
  posterior-mean formula and the derivation of Laplace's rule are left to
  that section, cited by `eq_mdl-beta_posterior`.
- Mode depends on the parameterization (ML-13): told once, in "Posterior Mode
  and Posterior Distribution", which now leads with it; the paragraph in the
  Beta subsection is gone.
- The coin's $\pm0.13$ (ML-17): now only in the error-bar subsection. The
  opening asks the question without the number; the closing paragraph is
  deleted; the Summary does not quote it.
- Learning rate against loss scale (ML-4): two sentences with
  `subsec_mdl-quadratic-model` and the exercise; the text no longer gives
  away exercise 1(b) and 1(c).
- Consistency, weak law, CLT (ML-11): one sentence with references in place
  of a paragraph of definitions.
- "We can now ..." closing paragraph: deleted. The hand-over to Bayesian
  Computation is the last sentence of part one (posterior averages) and of
  part two (the same bound as the objective of variational inference).

Kept because this file owns them: NLL as cross-entropy from the empirical
distribution, the $\epsilon$-window argument, the hinge counterexample, mean
against median, Fisher information and the information identity, the error
bar of the coin, asymptotic normality, Cramér–Rao, the ELBO and its gap, EM.

## Ranking

Demoted from Proposition blocks to running text: discrete MLE = minimum
cross-entropy; Gibbs' inequality (now cited); Gaussian NLL = squared error;
Gaussian prior = $L_2$ penalty; consistency (now a cited statement without
proof, hypotheses kept). Kept as Propositions: asymptotic normality (named as
the central result), Cramér–Rao, the ELBO gap, EM monotonicity. Labels added:
none. Labels removed: none.

Other moves inside the prose (no cell moved): the regularity conditions now
follow the two derivations that use them (ML-7); misspecification is promoted
from a bullet to a paragraph (ML-12), followed by "what survives for a
network" (S5).

## Split

Marker `<!-- split: new section "Latent Variables, EM, and the ELBO" starts
here -->` at line 1202, immediately before `## Latent Variables, EM, and the
ELBO` (`sec_mdl-latent-em-elbo`). Part one ends on its own argument (MAP is
one point; posterior averages go to Bayesian Computation). Part two opens
with its own question and cites part one by `sec_mdl-maximum_likelihood`; the
recap "The last two parts concerned ..." is gone. Part one does not need part
two (exercise 3b cites it by label only). The Summary has two paragraphs, one
per future section. Exercise 11 belongs to part two. To do at the cut: part
two needs its own imports cell (it uses `np` only), and the phrase "the
coin's counting estimator" in part two wants a label reference.

## Cells to cut or redesign on the development machine

- `maximum-likelihood-numerical-optimization-and-the-negative-log-likelihood`
  (ML-4): gradient descent on a function with a known minimizer. Kept as the
  first NLL minimized numerically; prose reduced. Candidate to cut, or to
  move with exercise 1 to chapter 40.
- `maximum-likelihood-fisher-information` (ML-9): confirms two proved
  identities. Prose now presents it as a demonstration of per-example
  automatic differentiation, with predicted standard errors. Could go.
- `mdl-maximum-likelihood-expectation-maximization-1`: the `assert` confirms
  the monotonicity proposition; the cell is kept for the speed of the climb.

## Deferred

- ML-15, second part: the ELBO figure writes $p(x;\theta)$ in its labels; the
  figure cannot change in this pass.
- ML-14 alternative (a second, flatter prior in the figure): not done.
- Exercises 1, 5 and 12 (move, split, shorten): left; 12(b) is now also
  worked in the text, so it should be trimmed when the exercises are revised.
- S1: the cut itself and re-capture.
- The table separator was changed from `|---|---|---|` to `|----|----|----|`
  only to silence the checker's em-dash pattern; rendering is unchanged.

## Numbers and claims added or changed

- Cauchy: $\varphi=\arctan u$ uniform, $s=\sin2\varphi$, $|s|\le1$, $I=1/2$;
  standard errors 0.007, 0.004, 0.008 for $10{,}000$ draws (simulated).
- Misspecification: a Poisson fit to counts with mean $m$ and variance
  $m+m^2/\alpha$ has $\textrm{Var}(\hat\lambda)=(m+m^2/\alpha)/n$ against
  $m/n$ from the Fisher information, a factor $1+m/\alpha$, equal to 3 for
  $m=4$, $\alpha=2$ (by hand; it is exercise 12b). Variance $B/A^2$ in the
  scalar case.
- With every training input distinct, the empirical floor of the training
  loss is zero; the held-out floor is the conditional entropy
  (`subsec_mdl-irreducible-loss`, chapter 42).
- "The largest stable step shrinks in proportion" to $1/n$ for a summed loss
  (from $\eta<2/L$, `subsec_mdl-quadratic-model`).
- The Fisher matrix is the Hessian of the KL divergence between neighbouring
  members of a family (`sec_mdl-divergences-distances`) and the natural-
  gradient preconditioner (`sec_mdl-adaptive-stochastic-methods`). Both
  sections state this; I did not verify how those files read after their own
  passes.
- A network "is both misspecified and non-identifiable": a judgment, stated
  in the limits subsection and the Summary.
- Numbers kept that no cell prints, recomputed: $2^{-13}=1.2\times10^{-4}$,
  $3.3\times10^{-4}$, ratio 2.7; 0.617, 0.076, 0.164 nats; se 0.128,
  curvature 61.0; 330 flips; $\pi^2/4=2.47$; $1/(1+e)=0.27$.

## Slides against the body

- "A summed loss needs `lr` $=10^{-9}$: learning rates are tuned to a loss
  scale": the body now says this in one clause and sends it to chapter 40.
- "$D_{\textrm{KL}}\ge0$ since $\log u\le u-1$": the body no longer proves
  this; it cites the exponential-family section.
- The recap slide and the $\pm0.13$ slide are consistent with the body. The
  kicker "§25.3" is stale (rewritten by the build).
