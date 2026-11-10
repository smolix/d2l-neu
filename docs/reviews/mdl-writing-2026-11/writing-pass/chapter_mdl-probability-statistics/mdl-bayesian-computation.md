# Writing pass log: Bayesian Computation

Source: `chapter_mdl-probability-statistics/mdl-bayesian-computation.md`. Prose
only; no code cell, cell ID, figure include or slide changed. Not split, no
markers.

## Words

Prose words (checker count): 9142 before, 9141 after (-1). The section did not
shrink, because the review asked for additions (BC-7, BC-8, BC-13, BC-15, S3,
S5) and found little to cut (S6); every addition is paid for by a cut. The
checker ends "OK: only prose changed" with no em-dash warning;
`tools/lint_source.py` is clean. (The baseline em-dash warning came from the
table separator `|---|`, now `|----|`; the table is otherwise unchanged
except for one cell, see S4.)

Numeric checks were run with `python3` and NumPy (no SciPy, no PyTorch) in
scratch files outside the repository: 400 simulated data sets for the running
example with a 161-point grid, 300 more for the evidence. Closed forms were
checked by hand or with `math.factorial`.

## Correctness changes

- **S1 (atypical stored data set): prose left, deferred.** I verified that the
  general claims are true of typical draws (400 data sets): plug-in at
  $x_\star=-3$ median 0.003 (quartiles 0.001 to 0.010); predictive over
  plug-in median 5.0, 5% to 95% range 2.4 to 11.5, 80% of runs inside
  [3, 10]; Laplace sds median 1.03 and 0.49; correlation median $-0.86$;
  difference at $x_\star=2$ at most 0.016 in 95% of runs; slope mean minus MAP
  median 0.25 sd. "Typically three to ten times" stays. As the review asked,
  one clause was added after the predictive cell: "a draw whose fitted slope
  comes out shallow gives a factor nearer two" (verified: log ratio and MAP
  slope correlate at 0.88; the runs with ratio below 2.5 have MAP slopes 0.3
  to 0.8 against a median of 1.15). The stored run is such a draw (slope
  0.53, ratio 1.7). See "Deferred".
- **BC-1.** "a probability of about one in a few hundred for a typical draw of
  the data ... is several times larger" is now "typically gives the label 1 a
  probability of a few in a thousand ... is about five times larger"
  (medians 0.003 and 5.0 above).
- **BC-2.** "Each answer repairs a failure of the one before" is now "Each of
  the first three answers repairs a failure of the one before; the fourth
  replaces sampling by optimization."
- **BC-5.** "matches the posterior's location, slope and curvature at the
  mode" is now "matches the position and the curvature of the posterior's
  peak". (The figure caption still says "value, slope and curvature"; it is a
  figure include and was not touched.)
- **BC-8.** "It is exact whenever the parameters are themselves drawn from the
  prior" now says what is exact: if nature draws the parameters from the
  prior and the labels from the model, then among the worlds that produce
  these twenty points the label at $x_\star$ is 1 with this probability.
- **BC-10.** The ESS "estimates, roughly, the number of independent posterior
  draws ..., and we shall see two ways in which it misleads".
- **BC-11.** "The smallest eigenvalue of the Laplace covariance is far below 2
  in every run, a few hundredths in a typical one" is now "Both eigenvalues
  ... are below 2 in nearly every run, typically about 1.2 and 0.05, so its
  weights have infinite variance in every direction, not along one thin
  axis." Verified: the larger eigenvalue is below 2 in 99.75% of 400 runs
  (median 1.25, 95th percentile 1.71); the smaller has median 0.052.
- **S4 (ELBO called a check).** The table's diagnostic for variational
  inference is now "comparison with samples" (was "ELBO, comparison with
  samples"); the closing list of checks is gone; the Summary says the
  samplers carry their own checks and the two Gaussian approximations are
  checked against samples. The sentence "A converged ELBO shows that the
  optimization within the family has stopped ..." is kept.
- **Exercise 8b.** "The Laplace value comes out a few percent low in every
  run" is now "The Laplace estimate of $p(\mathcal D)$ comes out a few
  percent low, about $0.04$ in $\log p(\mathcal D)$, in every run." Verified
  on 300 data sets: Laplace minus grid log evidence between $-0.071$ and
  $-0.022$, median $-0.042$, never positive.
- **Cross-reference.** Langevin dynamics now cites
  `sec_mdl-langevin-sampling` (the subsection that defines it) instead of the
  whole section `sec_mdl-score-matching-diffusion-flow`.

No finding was disputed.

## Six before/after pairs

1. Before: "Every model in the Basics part was trained to a single parameter
   vector, by maximum likelihood or, with a penalty, by the maximum a
   posteriori (MAP) estimate, the peak of the posterior."
   After: "The Basics part trains each model to a single parameter vector, by
   maximum likelihood or, with a penalty, by the maximum a posteriori (MAP)
   estimate, the peak of the posterior."
2. Before: "The next cell checks that this gives the same numbers at
   $\boldsymbol\theta^\star$." (and after the cell: "The differences are at
   the level of rounding error.")
   After: "For a larger model we would write only $\log\widetilde p$ and
   leave its gradient and Hessian to automatic differentiation, which here
   returns the same numbers at $\boldsymbol\theta^\star$ up to rounding."
3. Before: "The cell applies them to the four chains, compares the MCMC mean
   with the grid mean in units of the MCSE, and compares the measured
   $\kappa$ ..."
   After: "Now the test. If the error bar is honest, the MCMC mean of our
   four chains should differ from the grid mean by about one MCSE."
4. Before: "The cell reruns the chains with the proposal scale multiplied by
   0.05 and by 10."
   After: "We rerun the chains with the proposal scale set to 0.05 and to 10
   in place of 1.1. Before reading the output, guess which of the two runs is
   worth fewer draws."
5. Before: "The cell computes both on the draws of the rejection sampler."
   After: "We reweight the draws that the rejection sampler made, instead of
   discarding two thirds of them."
6. Before: "This balance of flows holds for every pair of states, and it is
   the reason Metropolis works."
   After: the same, followed by "If the reader keeps one result of this
   section, it should be this one."

Other cell introductions changed: model cell ("In the code, `log_joint` is
..."), MAP cell ("In code, ten Newton steps from zero find the mode, and one
matrix inverse gives the covariance."), predictive cell, rejection cell,
importance cell (a question), Metropolis cell, diagnostics definitions (no
lead-in), trace plot, variational cell, comparison plot. No lead-in in the
body begins "The cell" any more.

## Repetition removed; items kept

- **ELBO and its gap (BC-14).** Owner is `sec_mdl-latent-em-elbo`. The
  derivation by Bayes' rule and add-and-subtract is gone. The subsection now
  opens with the identification (parameters as the latent variable, the data
  set as the observation), cites `eq_mdl-elbo-gap`, and states the identity in
  this section's notation. Both displays and their labels
  (`eq_mdl-bayes-elbo-gap`, `eq_mdl-bayes-vi-elbo`) stay because exercises 7
  and 8 cite them. The variational autoencoder sentence moved to the closing.
- **BC-3.** "The MAP is finite in every run, separable data included" removed
  after the MAP cell; the statement under the Hessian stays.
- **BC-6.** The coin's Laplace paragraph is reduced to two sentences that tie
  it to `subsec_mdl-coin-error-bar`: under a flat prior the Laplace
  approximation is the MLE with its standard error 0.128. The exact mean and
  standard deviation, and the curvature 61, are no longer restated.
- **BC-16 / one ending.** The "We can now ..." paragraph is gone. Its scaling
  verdict (with the four reasons) and its hand-over are kept as the end of
  the comparison section.
- **Dead-end framing (S3).** "None scales" is said once in the body (closing
  of the comparison section) and once in the Summary; it is gone from the
  opening, which now says in one sentence why a small posterior is the
  teaching case and points to `sec_mdl-bayes-decision-map`. The rejection
  paragraph's "which it is not for the posterior of a network" is now "which
  it rarely is beyond toy models".
- Removed after confirm-only cells: "The acceptance rate is close to 0.305,
  and the kept draws have the Beta(10,5) mean ..." and "The ESS fraction is
  close to 0.42, and the weighted mean and standard deviation match the
  posterior."
- Removed: the two-state flows restated after the proof; "The chain spends a
  quarter of its time ... as it should"; the last sentence of the MCMC part
  (Monte Carlo error, one draw in $\kappa$), which the next section's first
  sentence says; "Every number in this subsection came from the grid"; the
  sentence announcing the table; the hedge "the order flips in about one run
  in a thousand" (S2).
- Geyer truncation (S6) cut to one sentence; the formula is in the code.
- Kept because this file owns them: the Laplace approximation, rejection and
  importance sampling with the ESS limit, detailed balance, the diagnostics,
  the mean-field proposition, the paragraph on the Laplace proposal's
  infinite variance (untouched apart from BC-11).

## Results demoted; labels added

- The rejection-sampling statement (kept draws have the posterior density;
  acceptance probability $p(\mathcal D)/M$) is no longer a
  Proposition/Proof block. It is a sentence with its three-line argument;
  `eq_mdl-bayes-rejection` keeps its label.
- Kept as propositions: the ESS limit, detailed balance (named in the text as
  the one result to keep), the mean-field fit.
- No labels added.

## Deferred to the development machine

- **S1, re-capture with a fixed data set.** Seed or list the twenty points in
  cell `bayesian-computation-model` (data draw only; samplers stay unseeded),
  then re-capture every cell that depends on the data:
  `bayesian-computation-map-predictive`, `bayesian-computation-grid`,
  `mdl-bayesian-computation-the-plug-in-and-the-posterior-predictive-1` and
  `-2`, `bayesian-computation-importance`,
  `mdl-bayesian-computation-detailed-balance-and-stationarity-1`,
  `mdl-bayesian-computation-diagnostics-for-markov-chain-mixing-1` and `-2`,
  `mdl-bayesian-computation-proposal-scale-and-mixing-failures`,
  `bayesian-computation-variational`, `bayesian-computation-comparison-plot`.
  Stored today: plug-in 0.0960, predictive 0.1648 (factor 1.7), Laplace sds
  0.826 and 0.357, 90% range [0.0032, 0.6323] which excludes the true 0.0006.
  After the re-capture the prose can quote this posterior's numbers and drop
  the remaining hedges about runs (S2): "in nearly every run", "in most
  runs", "in about nine runs out of ten", "about one run in thirty".
- **Cells to cut or redesign:**
  - `mdl-bayesian-computation-the-gradient-and-the-hessian-2` (BC-4, S6):
    four-tab autodiff check of a two-line derivative; prose now one clause.
    Move to an exercise.
  - `mdl-bayesian-computation-rejection-sampling` (confirm-only): confirms
    0.305 and the Beta moments; kept because the next cell reuses its draws.
    Prose now one sentence.
  - `mdl-bayesian-computation-the-effective-sample-size` (confirm-only):
    confirms the limit 0.42. Could merge with the previous cell.
- **S4, alternative remedy:** print the ELBO of the Laplace and mean-field
  Gaussians beside the grid's log evidence (new cell).
- **BC-6, exercise 4a** still asks for the mean and standard deviation that
  the text now gives in passing (0.692 is written as $9/13$, 0.128 is
  quoted). Start the exercise at the interval and the mass outside $[0,1]$.
- **BC-12** done in prose (definition in words before the display; warm-up
  moved to the first run). **BC-9** done.
- Figure caption of `fig_mdl-prob-bayes-laplace` says "value, slope and
  curvature" (BC-5); figure includes were out of scope.

## Claims added or changed (all checked)

- Typical plug-in "a few in a thousand", ratio "about five times" (simulation).
- "a draw whose fitted slope comes out shallow gives a factor nearer two"
  (simulation).
- BC-7: for small probabilities $\operatorname{sigmoid}(a)\approx e^a$, and
  $\mathbb{E}[e^a]=e^{\mu+s^2/2}$ for a Gaussian logit; "Averaging raises the
  log-probability by about $s^2/2$, and by less once the sigmoid flattens."
  The first is the Gaussian moment generating function. In the simulation the
  Laplace logit variance at $x_\star=-3$ has median 5.8, so $s^2/2\approx2.9$,
  while the measured log-odds shift has median 1.6: the formula is an upper
  guide, as the text's "by less" says. **This is the statement I am least
  sure is worded strongly enough**; the cap is large here.
- Laplace on the coin under a flat prior equals the MLE $9/13$ with standard
  error $0.128=1/\sqrt{61}$ (by hand; the number is in
  `subsec_mdl-coin-error-bar`).
- Both Laplace eigenvalues below $\tau^2/2=2$, typically 1.2 and 0.05
  (simulation).
- $\widehat R$: the numerator estimates the posterior variance and overstates
  it while chains disagree; $W$ understates it while each chain has seen only
  part of the posterior (standard, Gelman and Rubin).
- The forward KL needs expectations under the posterior, the reverse only
  draws from $q$ (BC-15).
- The decision paragraph (S5): grid for a handful of parameters; Laplace as a
  first answer and a proposal; MCMC for an expectation with an error bar;
  variational inference inside a training loop. A judgment, not a theorem.
- A network's symmetries copy every mode "once for every permutation and sign
  flip of its hidden units" (exercise 9d; true as stated for odd activations
  such as tanh).
- Where the book uses each idea, labels verified with `git grep`: the ELBO in
  `sec_diffusion-latent` (the variational autoencoder, which cites
  `sec_mdl-latent-em-elbo`) and `sec_diffusion-discrete`; importance weights
  and the ESS in `sec_ppo`; the Langevin sampler in `sec_diffusion-denoising`
  and `sec_mdl-langevin-sampling`.
- Hand-over: "how far such an average can be trusted when the draws are
  independent" (the chapter review notes that concentration covers
  independent draws only).
- Exercise 8b: 0.04 nats (simulation).

Numbers in prose that no cell prints, kept after recomputation: acceptance
0.305 and ESS limit 0.42 for the coin ($B(10,5)/M=0.3051$,
$B(10,5)^2/B(19,9)=0.4210$); "about one run in thirty" separable and "nine
runs out of ten" rest on the reviewer's 1,500 runs, which I did not repeat.
Not rechecked by me: the rates 0.10 and 0.03 for ten and a hundred times the
data, the Gaussian ESS limits 0.83, 0.66, 0.46, the importance-sampling
ranges (1% to 10%, 55% to 95%), "a few runs out of a hundred"; the reviewer
confirmed these and I left the sentences as they were.

## Notation

- $N_{\mathrm{eff}}$ is used for the weight ESS and the chain ESS; the text
  says so in a clause. No clash with other sections found.
- This section writes the ELBO as $\mathcal L(\psi)$, the owner as
  $\mathcal L(q,\boldsymbol\theta)$, where $\boldsymbol\theta$ is a model
  parameter and $z$ the latent. Here $\boldsymbol\theta$ is the latent. The
  new opening sentence of the subsection states the mapping.

## Slides against the body

- Slide "Choose a Method by How It Fails": row "mean-field VI | fitted $q$ | ELBO | too
  narrow" still names the ELBO as the check; the body no longer does.
- Slide "Far From the Data ...": "typically 3–10× the plug-in" agrees with
  the body (and, like the body, not with the stored figure; S1).
- The "Check: Laplace on the Coin by Hand" slide agrees with the shortened
  coin paragraph.
- No slide mentions where the book reuses the ideas.
