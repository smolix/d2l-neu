# Writing pass log: Score Matching and Diffusion Models

File: `chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md`. Reviews
followed: `43-dynamics/mdl-score-matching-diffusion-flow.md` (structural
findings 1 to 5, SMD-1 to SMD-18) and `43-dynamics.md`. Not split; no
markers.

## Words and checks

Checker count (prose with exercises): 8,390 before, 8,574 after (+184).
**The count went up, against the brief.** Where it comes from:

- About 50 of the +184 are label tokens: the pass added some 25
  `:numref:`/`:eqref:` references (to chapter 17, chapter 42 and the
  neighbouring sections), and the checker counts each label as several
  words. With role markup stripped the count is 8,016 to 8,151 (+135); the
  body without exercises is 6,201 to 6,317 (+116).
- Additions asked for by the review: the curvature proposition with proof
  (SMD-15, about 130 words), the ancestral step (SMD-8, about 70), the
  convergence sentence (SMD-5, about 45), the floor under the likelihood
  weight (SMD-10, about 45), the statement of what is fitted (SMD-1, about
  55), the honest reading of the Langevin run (SMD-6, about 60 net), the
  notation warnings (SMD-9, SMD-13, about 60).
- Cuts (about 420 words): the guidance derivations and the paragraph on the
  clean-level tilt, the prose around the KS cell, the MALA sentence, two of
  three "variance-preserving" explanations, the variational weight formula,
  the EDM sentence, the sign-preservation sentence, the "We can now" recap,
  two proposition blocks, the note on the next section's clock.

If the length rule is strict, the next candidates are exercise 4 parts 2 to
4 and exercise 6 parts 3 and 4 (the review proposes to trim both; I left
exercises alone as instructed), about 330 words.

Checker: "OK: only prose changed", no em-dash warning, display delimiters 56
to 60 (two new displays: the curvature identity and the ancestral step).
`lint_source.py` exits 0 with no messages. All added references resolve
(checked with `git grep` for each label).

Numeric checks: `python3` with NumPy in scratch files outside the repository
(no `.venv-pytorch` here); the algebra by hand.

## Correctness changes

- **SMD-5.** Was: stationarity proved, then the experiment asked whether
  chains from other starts "reproduce $p$". Now, after the proof:
  "Stationarity says nothing yet about a chain started elsewhere, and a
  sampler always starts elsewhere. Under conditions of this kind the law of
  $\mathbf{X}_t$ does converge to $p$ from every starting point
  :cite:`Roberts.Tweedie.1996`. We do not prove it; how long the convergence
  takes is what the experiment below measures." Source of the claim:
  Roberts and Tweedie (1996), Theorem 2.1 (non-explosive Langevin diffusion
  with $C^1$ score converges in total variation from every start), quoted
  from memory; the key is already in the bibliography. I wrote "conditions
  of this kind" because their growth condition is not word for word ours.
- **SMD-6.** Was: "From the broad start, about half the chains end in each
  mode", read as a check of the sampler. Now the text says the even split
  "is no evidence for the sampler": start and target are symmetric, the
  proportions are those of the start, and "with mode weights $0.8$ and $0.2$
  the same broad start would still split about evenly", with a pointer to
  `sec_diffusion-denoising`, which measures it. Verified: the cell's sampler
  with weights 0.8/0.2 (right mode 0.2), same start, $h$, steps gives
  $P(X>0)=0.489$, $\mathbb{E}X^2=4.26$; the symmetric case gives 0.502 and
  1.0% crossing. "Limits of a Single Noise Level" now counts three failures
  (the third: the score hardly records the weights, cited to
  `sec_mdl-fisher-divergence`, where the swapped-weights experiment is).
  The annealing sentence no longer says the ladder "gets the proportions
  approximately right" on this mixture; it cites
  `sec_diffusion-gaussian-paths` for unequal weights.
- **SMD-10.** Was: "$T$ times the loss, plus the KL divergence ..., bounds
  ...". Now: "$T$ times the excess of the loss over its floor, plus ...",
  and "The floor must be subtracted: for the VP process under this weight it
  grows like $d/t$ as $t \to 0$ and is not integrable, which is also why
  training starts at a small $\varepsilon > 0$". By hand: the loss is
  $\tfrac1T\int\tfrac12 g^2\,\mathbb{E}\|\mathbf{s}_\theta+\epsilon/\sigma_t\|^2dt$,
  the regression lemma splits off the floor, and the remainder times $T$ is
  the integral in exercise 9(c); floor times weight is
  $\beta(t)d/\sigma_t^2\approx d/t$ (exercise 5).
- **SMD-14** (stored output, ordinary draw). Was: "to about $1.03$ and
  $1.07$ ... and narrows it slightly, from $0.24$ to about $0.23$" and "at
  the cost of diversity". Now: "to about $1.03$ and $1.08$ ... The width
  stays within a few percent of $0.24$", and the closing sentence says a
  single Gaussian mode has little diversity to lose, with a pointer to
  `subsec_diffusion-images-guided`. Stored: means 1.027, 1.077; std 0.243,
  0.236, 0.237. Exercise 7 quoted "about $1.07$"; changed to "about $1.08$"
  to agree.
- **SMD-3.** "about $2.1$" is now "about $2.1$ to $2.2$", and the gap to
  the floor 2.11 is named as the sampling error of a 16,384-pair average
  (standard error about 0.02, computed as $\sqrt2\cdot2.11/\sqrt{16384}$).
- **SMD-7.** The question "Which forward processes give a Gaussian kernel?"
  is replaced by a statement, and the next subsection answers it: "Every SDE
  whose drift is linear in $\mathbf{x}$ and whose noise amplitude does not
  depend on $\mathbf{x}$ has Gaussian transition kernels" (standard fact for
  linear SDEs; the OU case is `eq_mdl-sde-ou-kernel`).
- **SMD-1, SMD-2.** See "claims added" below. Heading "A Score Network in
  One Dimension" is now "A Least-Squares Score in One Dimension" (it has no
  label).
- No finding disputed. SMD-9 applied only in part (see deferred).

## Stored outputs compared

`outputs/pytorch/chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.json`:
$w=2.03$; 3.207 and 0.801; fit 2.15, exact 2.15, floor 2.11, excess 0.008,
fit at 8 is $-6.4$; Langevin 0.503, 4.25, 0.011; chain variances 0.998,
1.011, 1.032 and KS 0.008, 0.013, 0.020; DDIM 0.832, 0.964, 0.998, gaps
0.084/0.081 and 0.015/0.019; guidance as above. All prose statements agree.

Numbers no cell prints, kept after recomputing: $\sigma\ge1.94$; modes
$\pm0.92$ and standard deviation $0.40$ at $t=100$; $\bar\sigma$ from 0.01
to 157; $\bar\alpha_N\approx4\times10^{-5}$; guidance thresholds 1.0625 and
1.09; "Heun wins only once the strides are short" (my run: mean gap, Heun
with 5/10/20/40 strides 0.22/0.040/0.011/0.003 against DDIM with twice the
strides 0.083/0.038/0.019/0.009). Dropped: "No sample changes mode at any
stride count: each stride map is odd and ... increasing" (no cell shows it
and I did not prove monotonicity).

## Six before/after pairs

1. Before: "weighting, and the samplers of the previous section run
   unchanged once the learned field replaces the exact score."
   After: "No network is trained in this section. We fit a score once, by
   linear least squares on 17 fixed features at a single noise level, to see
   the regression work. Every sampler then runs on the exact score of the
   mixture, so that each error we measure is the sampler's own."
2. Before: "The cell below asks whether a sample average finds the same
   answer: it evaluates $J$ on 10,000 draws from $\mathcal{N}(0, 2)$ over a
   grid of $w$."
   After: "In practice the expectation is a sample average. The cell
   replaces $v$ in $J$ by the mean of $x^2$ over 10,000 draws from
   $\mathcal{N}(0, 2)$ and minimizes over a grid of $w$."
3. Before: "It asks two questions. Do chains started from a broad
   distribution reproduce $p$? And do chains started in the left mode ever
   reach the right one?"
   After: "... twice: from a broad start $\mathcal{N}(0, 3^2)$, and with
   every chain started at the left mean. Before reading the output, guess
   how many of the 10,000 chains of the second run reach the right mode."
4. Before: "The model is a fixed set of 17 features ... and a linear
   combination of them."
   After: "The model is not a network. It is a linear combination of 17
   fixed features ... We choose it because fitting the DSM loss is then
   ordinary least squares ... any gap between the fit and the score is
   sampling error, with no optimizer to blame."
5. Before: "The cell checks the $\bar\alpha$-marginal on data that are not
   Gaussian." / "The variance stays within a few hundredths of $1$ ..."
   After: "The cell sets up what the samplers below use ... It also runs the
   chain once ..." / "The induction left this output nothing to decide:
   chain and formula agree to within the sampling error of 4,096 points. The
   rows are more useful as a picture of the levels."
6. Before: "The cell asks what the guided sampler produces instead."
   After: "What does the guided sampler produce, if not samples of a noised
   class?"

Ranking sentences added: the roadmap names the three results chapter 17
relies on (Vincent's theorem, DDPM is the VP process, DDIM is Euler) and
what can be skipped (the proof of Hyvärinen's identity, guidance); "This is
the result of the subsection to remember" (exact VP transition and
marginal); the DDIM proof now says which step is algebra and where the
substance is.

## Repetition and ownership

Kept, this file owns: Hyvärinen's identity in $d$ dimensions (chapter 42
points here for it), the regression lemma, Vincent's theorem, Tweedie, the
floor, Langevin stationarity and bias, the $\bar\alpha$-marginal, DDIM is
Euler with both corollaries, and the new curvature bound.

Reduced to a clause with a reference:

- Fisher divergence: definition by name, `sec_mdl-fisher-divergence`. The
  one-dimensional step of Hyvärinen's proof is identified as Stein's
  identity `eq_mdl-stein-identity` with $g=s_\theta$ (chapter 42 says the
  same from its side). I kept the three-line computation, since chapter 42
  sends the reader here for the $d$-dimensional case.
- Mixture score: cited as `eq_mdl-dyn-tanh-score`, formula kept (the review
  lists the restatement as a strength).
- Reverse SDE and probability-flow ODE: the display stays because
  `eq_mdl-score-samplers` is a label and the DDIM proof uses it; the prose
  now cites `eq_mdl-dyn-reverse-sde` and `eq_mdl-dyn-pf-ode` and no longer
  restates their properties. The family of reverse processes is cited as
  `eq_mdl-dyn-reverse-family`.
- "Variance-preserving" (SMD-11): one clause with `sec_mdl-sdes`; the
  computation after the marginal is gone.
- Euler-Maruyama agreement of the DDPM step: one clause.
- Hutchinson (SMD-4): named, `sec_mdl-continuous-normalizing-flows`.

Against chapter 17 (kept able to bear its citations; I checked each citing
passage):

- `sec_mdl-ddim-guidance` is cited for (a) DDIM is Euler and the two
  extreme cases, (b) "the rule on a one-dimensional mixture", (c) the
  curvature bound and its violation "at most noise levels". All three are
  still in that part: (a) unchanged, (b) Bayes' rule, the two guided fields
  and the experiment, (c) the bound (now proved, by reference to the new
  proposition) with thresholds and "only at about ten of the thousand
  levels" for $\gamma=3$.
- Cut from the guidance subsection as chapter 17's material: how the
  classifier and the label-dropping network are trained, the $\gamma=0,1$
  reading, the noise-prediction form of the rule, the "tilt at the clean
  level" paragraph (it also answered exercise 7 part 1). Pointers:
  `sec_diffusion-images`, `subsec_diffusion-images-guided`.
- Limits of one level and annealing: kept the unimodality threshold and the
  bias formula; pointers to `sec_diffusion-denoising` and
  `sec_diffusion-gaussian-paths`.
- DDPM weight history (SMD-12): ends at "the same regression with a
  different weight on each level", `sec_diffusion-design-space`.
- Samplers as recipes: one sentence, `sec_diffusion-samplers`.

Inside the file: the "We can now" sentence is gone (SMD-16), the bridge to
the next section stays as the end of the body; the note on the next
section's clock is gone (SMD-17); the Summary no longer repeats the numbers
3.2 and 0.83.

## Propositions and labels

Demoted to sentences: "the DDPM step is an exact VP transition" (now a
paragraph with its two-line reason) and "the DDPM loss is reweighted DSM"
(now the display with one sentence before and after). Both stay under
`sec_mdl-ddpm-discretized-sde`, which is what chapter 17 cites.

Promoted (SMD-15): **Proposition (curvature of a noised density)**, placed
after the floor in "Denoising Score Matching".

Label added: `eq_mdl-noised-curvature`.

## Cells to cut or redesign on the development machine

- `#score-matching-diffusion-flow-langevin` (SMD-6): give the mixture
  unequal weights in this cell, so that the broad-start line tests
  something; the printed "truth 0.5" cannot fail today.
- `#score-matching-diffusion-flow-ddpm-marginal` (structural finding 4):
  confirms a four-line induction. The KS loop can go; the cell must keep
  `N`, `beta`, `alpha_bar`, `ks`, `x0`, which later cells use.
- `#mdl-score-matching-diffusion-flow-guidance-steering-with-bayes-rule`
  (SMD-14, structural finding 2): the review would move it to chapter 17. I
  kept it because chapter 17 cites this section for "the rule on a
  one-dimensional mixture". A class with two modes would show the loss of
  diversity that one mode cannot.
- `#mdl-score-matching-diffusion-flow-a-normalizer-free-objective`:
  confirms a computation done by hand; this is the "once or twice" case and
  I left its prose.
- `#score-matching-diffusion-flow-imports` (structural finding 3): four
  framework tabs, no framework used.
- Missing experiment (SMD-1): a trained noise predictor driving `ddim`
  (exercise 10 part 1 and 2).

## Other deferred findings

- **SMD-9**, done in part. The text now says that in the DDPM and DDIM
  subsections $t$ counts levels and stands for the time $t\Delta$, the mixed
  formula is explained, step (ii) of the DDIM proof is flagged as continuous
  time, and the warning about Ho et al.'s $\alpha_t$ and factor
  $\tfrac12$ is in. Not done: re-indexing the levels by $k$. It would touch
  every display of two subsections, the exercises, the slides and the
  printed output ("t = 100"), and chapter 17 uses continuous $t$.
- **SMD-13**: the cheap fix (a sentence on which bars mean what). The
  symbols are shared with `samplers.md`.
- **Title** (structural finding 2: "Score Matching" plus two identification
  theorems): not changed; the title is in the slides and cited widely.
- **Exercises**: 3 (unequal weights), 4 and 6 (trim), 10 (after SMD-1) left
  as they are. Changed: cell IDs replaced by "the cell that defines ..."
  in five places (SMD-18); exercise 7 "1.07" to "1.08"; exercise 8 part 1
  now starts from the new proposition instead of deriving it (same display,
  same conclusion, parts 2 and 3 untouched; chapter 17 cites the exercise).
- Exercise labels and cell IDs with the old prefix: untouched.

## Claims added or changed (for checking)

1. "No network is trained in this section ... linear least squares on 17
   fixed features at a single noise level ... Every sampler then runs on the
   exact score of the mixture." True of all eight cells.
2. Stein's identity with $g=s_\theta$ is the one-dimensional step of
   Hyvärinen's proof (chapter 42 states the same).
3. Curvature proposition:
   $\nabla^2\log p_\sigma=-\mathbf I/\sigma^2+\mathrm{Cov}[\mathbf x\mid\tilde{\mathbf x}]/\sigma^4$
   "for any data distribution". Proof by hand (Bayes' rule gives
   $\nabla_{\tilde x}\log p(x\mid\tilde x)=(x-\mathbb E[x\mid\tilde x])/\sigma^2$);
   the posterior has all moments because of the Gaussian factor. Checked on
   the Gaussian ($-1/(v+\sigma^2)$) and numerically on the noised mixture
   (largest $-(\log p_\sigma)''$ is 2.0 against the bound 4.0 at
   $\sigma=0.5$). The guidance subsection applies it to
   $\sqrt{\bar\alpha_t}\mathbf x_0$ noised at scale $\sigma_t$.
4. "At $\gamma=3$ the bound holds only at about ten of the thousand levels,
   the least noisy ones, and at $\gamma=10$ at fewer still": threshold
   $1+v_0v_t/(\sigma_t^2m^2)$ is below 3 from index 9 of `alpha_bar` on and
   below 10 from index 3 on (NumPy). The formula itself is not printed in
   the body, since exercise 8 part 3 asks for it.
5. Convergence of Langevin dynamics from every start (SMD-5, cited).
6. Weights 0.8/0.2 split about evenly from the broad start (recomputed).
7. Linear SDEs have Gaussian kernels (SMD-7).
8. The floor under $w=g^2$ for the VP process grows like $d/t$ (SMD-10).
9. Ancestral step
   $\mathbf x_{t-1}=(\mathbf x_t+\beta_t\mathbf s_\theta)/\sqrt{1-\beta_t}+\sqrt{\beta_t}\boldsymbol\xi$,
   no noise in the last step, equal to first order to the Euler-Maruyama
   step of the reverse VP SDE. This is the line in `cfg_sample`; by hand,
   $1/\sqrt{1-\beta}=1+\beta/2+O(\beta^2)$.
10. Ho et al. write $\alpha_t$ for $1-\beta_t$ and their simple loss has no
    factor $\tfrac12$ (from memory of the paper; the review says the same).
11. The variational route "yields the same regression with a different
    weight on each level" (the review's wording).
12. "`sec_diffusion-gaussian-paths` shows that the ladder recovers unequal
    weights" and "`sec_diffusion-denoising` measures it" (read in both
    files: weights within 0.03 of 0.5/0.3/0.2; Langevin keeps the start's
    thirds).
13. DDIM is the $\lambda=0$ member of `eq_mdl-dyn-reverse-family`.

## Slides now out of step with the body

- "But Chains Do Not Cross the Valley": "Two failures: no pairs far out, no
  crossing." The body counts three (mode weights).
- "Past $\gamma=1$ the Class Sharpens Beyond Itself" and its note
  "Mean $0.97\to1.03\to1.07$"; Recap "mean $0.97\to1.07$". The body says the
  mode moves to about 1.08 and the width hardly changes.
- "DDPM Is the VP-SDE Discretized, with an Exact Marginal" is fine; any
  slide wording that speaks of three propositions would be stale (I found
  none).
- The slide title "Least Squares on 16,384 Noisy Pairs Reaches the Floor"
  agrees with the body.

## For neighbouring files

- `mdl-flow-matching-transport.md` says "The previous section trained a
  network to predict the noise" (and two related sentences, FMT-1). This
  section now states at the top that no network is trained here.
- `chapter_diffusion-models/gaussian-paths.md:600` cites
  `sec_mdl-ddim-guidance` for a fact that is in `subsec_mdl-fm-two-point`
  (found by the reviewer; not mine to edit).
- `image-diffusion.md` cites exercise `guided-marginals` for the curvature
  bound; it could now cite `eq_mdl-noised-curvature`.
