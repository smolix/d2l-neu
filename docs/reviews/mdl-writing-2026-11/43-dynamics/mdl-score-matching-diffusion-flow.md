# Score Matching and Diffusion Models

Source: `chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md` (lines 1 to 1452; about 7,300 words of body prose and mathematics, about 2,300 words of exercises, 8 code cells, all NumPy).

Context the brief did not have: chapter 17 (`chapter_diffusion-models/`) is no longer a placeholder. It holds ten files and about 83,000 words, cites this section 14 times, and restates most of its results. The scope judgement below is made against that chapter as it stands.

## Verdict

The first half of this section (Hyvärinen's identity, the regression lemma, Vincent's theorem, the Gaussian worked by hand, the loss floor) is at the level of the model texts: one line of argument, each abstraction followed by a number the reader can compute, proofs of three to eight lines, and an author who says what a result means. I found no error in any proposition or proof, and I recomputed every worked number. The second half (DDPM, DDIM, guidance) is correct and carefully built but is three topics in sequence, and with chapter 17 written it is no longer clear which of the two texts teaches them. The most important thing to fix is a mismatch between what the section says it does and what it does: the opening and the next section say that a network is trained and that the samplers run on the learned field, but the only fit in the section is a 17-feature least-squares fit at one noise level, and every sampler (Langevin, the reverse chain, DDIM, guided sampling) runs on an exact score. Either say so in the opening, as a deliberate choice that isolates sampler error from learning error, or add the missing run. Second in importance: the Langevin experiment's headline check (half the mass in each mode) is true by symmetry and would pass with a wrong sampler.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening poses a problem the reader feels.** "How do we learn a field that we cannot evaluate at even a single point?" follows directly from the previous section's closing question. The second paragraph gives the history as an argument (Hyvärinen removed the normalizer, Vincent "noticed something simpler still") instead of a list of names.
- **The Gaussian worked by hand, twice.** After Hyvärinen's identity: $J(w) = v/(2w^2) - 1/w$, minimum at $w = v$, and "the objective never saw the data score and still returns the variance". After Vincent's theorem: the posterior, the conditional mean of the target, and the floor $1/\sigma^2 - 1/(v+\sigma^2) = 3.2$, then a two-line cell that prints $3.207$ and $0.801$. This is the predict-then-run method the Basics review asked for. Protect both passages.
- **The reading of the two terms** ("The divergence term $-1/w$ rewards a field that points inward at the samples, a sink where the data are ... The squared norm ... stops the field from growing without bound"). A reader can quote this a week later.
- **The floor as a corollary with consequences**: it explains why the loss does not go to zero, why a small $\sigma$ is badly conditioned, and (later) why DDPM's weight $\sigma_t^2$ is the natural one. One fact, used three times, derived once.
- **"DDIM is Euler's method"** is the right theorem to prove, the change of coordinates is motivated by a question ("What, then, does DDIM integrate, and how large an error does a long stride make?"), and the two corollaries (a point mass is integrated exactly; Gaussian data shrink by $\cos(\phi_t - \phi_{t'})$) bracket the Euler error with closed forms. The stored output ($0.832$, $0.964$, $0.998$) matches the text. Figure `mdl-dyn-ddim-strides` carries the argument: panel (a) answers the literal misreading of the update.
- **Candour about limits**: the boundary-term failure of implicit score matching, the arbitrary extrapolation of the fit at $x = 8$ ("it can even be positive, pointing away from the data"), "that row is a limit, not a test".
- **Exercises 1, 8 and 9** (boundary terms; guided densities are not noised marginals; the likelihood bound from the Fokker-Planck equation) are real problems with depth.
- All cross-reference labels resolve. The section restates the mixture score formula instead of sending the lookup reader back for it.

## Structural findings

1. **Nothing is trained, and the text says otherwise.** See SMD-1. This is the Basics pattern (prose claims more than the run shows) in a milder form: the mathematics is all proved, but the reader never sees a learned multi-level score drive a sampler. Exercise 10, tagged [extended], carries the section's central demonstration.

2. **Mathematics background or the topic itself?** Split by subsection:
   - *Mathematics, and this is its home*: Hyvärinen's identity; the regression lemma; Vincent's theorem; Tweedie and the floor; stationarity of Langevin dynamics and the discretization bias; the three DDPM propositions; DDIM is Euler; the curvature bound on noised marginals (now only an exercise). Chapter 17 states each of these and points here for the proof, which is the division §7.6 of the guide asks for.
   - *The topic itself, taught a second time*: the guidance subsection. Chapter 17 (`image-diffusion.md`, "Guidance") gives the same derivation in nearly the same words (Bayes' rule on scores, classifier guidance, "read ... from right to left", the tilted density, the curvature bound) and then runs it on images. Classifier-free guidance is a modelling technique, not background mathematics. Keep here only what chapter 17 cites: the bound and the mixture thresholds, as a proposition (SMD-13). Move or cut the rest.
   - *Borderline*: "Limits of a Single Noise Level" and annealed Langevin dynamics are told again in `denoising.md` ("Limits of a Single Noise Level", same heading) and `gaussian-paths.md` ("Annealing through Noise Levels"). Chapter 17's version is better because its mixture has unequal weights (SMD-6). The DDPM history and weight formula at the end of the DDPM subsection belongs to `design-space.md`.
   - The title promises "Diffusion Models". After the moves above the section is "Score Matching" plus two identification theorems (DDPM is the VP process; DDIM is Euler), and should be titled for that.

3. **Four framework tabs for a NumPy notebook.** The imports cell loads MXNet, PyTorch, TensorFlow and JAX in four tabs; no cell uses any of them. Either drop the framework imports (one untagged imports cell with `numpy` and `d2l`) or use them for the missing training run.

4. **Length and proportion.** The first two `##` parts (learning; sampling with one level) take about 510 lines and are tight. The third and fourth take about 570 lines for three topics. Candidates to cut with no loss to the argument: the KS-statistic cell that re-verifies the $\bar\alpha$-marginal after a four-line induction has proved it (keep its last sentence, "Training therefore never simulates the chain"); the DDPM weight formula; the guidance experiment (moved); the paragraph before the Summary's first sentence.

5. **Lookup use.** A reader sent from chapter 17 to `sec_mdl-ddpm-discretized-sde` or `sec_mdl-ddim-guidance` lands in a subsection that depends on notation set 100 to 300 lines earlier ($\alpha_t$, $\sigma_t$, `mixture_score`, the standardized mixture `m`, `v0`). The DDIM subsection restates what it needs; the guidance subsection does not say what "the ancestral chain" is (SMD-8).

## Passage-level findings

**SMD-1** | major | pedagogy, thought
Anchor (opening): "weighting, and the samplers of the previous section run unchanged once the"
Also (Scores Across Noise Levels): "that takes the noise level as a second input, train it at every level at"
The section announces a network $\mathbf{s}_{\boldsymbol\theta}(\mathbf{x}, t)$ trained at every level and samplers run on the learned field. No such network is trained. The one fit is linear least squares on 17 fixed features at $\sigma = 0.5$; Langevin, the DDPM chain, DDIM and the guided chain all use `mixture_score`. The next section then says "The previous section trained a network to predict the noise", which is false. A reader who wants to see that a learned score, with its error, still samples the mixture has to do exercise 10.
Fix (writing pass): state the design in the opening, for example: "We fit the score once, at one noise level, to see the regression work. From then on the samplers run on the exact score of the mixture, so that every error we measure is the sampler's own; :numref:`chap_diffusion` trains the networks." Change "train it at every level at once" to "which is trained at every level at once by the loss below". Fix the sentence in the next section (FMT-1). Deferred (experiment): a short cell that trains the noise predictor of exercise 10 and runs `ddim` on it would close the gap and give the four framework tabs a purpose.

**SMD-2** | moderate | pedagogy
Anchor (heading): "### A Score Network in One Dimension"
The model is "a fixed set of 17 features ... and a linear combination of them", solved by `np.linalg.lstsq`. It is a good choice (no optimizer, the regression lemma is exact) but it is not a network, and the heading raises the expectation that SMD-1 describes.
Fix: "### A Least-Squares Score in One Dimension", and one clause saying why a linear model is used: the fit is then exact least squares, so any gap to the floor is sampling error and not optimization error.

**SMD-3** | minor | thought
Anchor (same subsection): "The fit and the exact score both have mean squared error about $2.1$ on"
The stored output prints `fit 2.15, exact score 2.15, population floor 2.11`. The reader sees 2.15 beside 2.11 and the text calls both "about 2.1" without saying why they differ by 0.04. The reason (the sample average over 16,384 pairs has a standard error of about 0.02 to 0.03) is left to exercise 4.
Fix: add one clause: "about $2.1$ to $2.2$; the gap to the population floor of $2.11$ is the sampling error of a 16,384-pair average (exercise ...)".

**SMD-4** | minor | explanation
Anchor (A Normalizer-Free Objective): "passes in all, and $d$ is in the millions for images. Random projections"
"Random projections estimate the trace with one pass but add variance" names neither the estimator nor where the reader has met it. Hutchinson's estimator is taught in `sec_mdl-continuous-normalizing-flows`.
Fix: "Hutchinson's estimator (:numref:`sec_mdl-continuous-normalizing-flows`) replaces the trace by one vector-Jacobian product per random direction, at the price of variance."

**SMD-5** | minor | correctness (confidence high that a sentence is missing, not that anything is false)
Anchor (Langevin Dynamics): "If $\mathbf{X}_0 \sim p$, then $\mathbf{X}_t \sim p$ for all $t \ge 0$.*"
The proposition proves invariance: started in $p$, the process stays in $p$. The experiment that follows starts from $\mathcal{N}(0, 9)$ and from a point mass and asks whether the chains "reproduce $p$", which needs convergence, a different and unproved statement. The text slides from one to the other; "The claim is that $p$ is its stationary law" does not tell the reader that stationarity alone promises nothing about other starts.
Fix: after the proof add: "Stationarity says nothing yet about a chain started elsewhere. Under mild conditions the law of $\mathbf{X}_t$ does converge to $p$ from any start, which we do not prove; how fast is the question the experiment asks."

**SMD-6** | major | thought
Anchor (Langevin Dynamics): "From the broad start, about half the chains end in each mode and the second"
Code anchor: "print(f'spread-out start: P(X > 0) = {(warm > 0).mean():.3f}, '"
The check "P(X > 0) = 0.5 (truth 0.5)" cannot fail. The target is symmetric and the start $\mathcal{N}(0, 3^2)$ is symmetric, so the chains split evenly whatever the dynamics do between the modes; and the second half of the same cell shows that chains do not move between modes in 2,000 steps (about 1% cross). So the proportions are those of the initial distribution, not of $p$. I reran the cell's sampler on the same mixture with weights $0.8/0.2$: from the same broad start it gives $P(X > 0) \approx 0.48$ against the truth $0.2$. The experiment therefore hides the best-known failure of single-level Langevin sampling, the one chapter 17 puts first ("Blindness to Mode Weights" in `denoising.md`, on a mixture whose "unequal weights give a sampler something to get wrong"). The next subsection then lists "Two failures" and omits it, and says annealing "gets the proportions approximately right" on a mixture where any symmetric procedure does.
Fix (writing pass): say what the run shows. "From the broad start the second moment matches $4.25$: each chain equilibrates inside the mode it falls into. The even split between the modes is inherited from the symmetric start, not produced by the dynamics. The second run shows why: chains do not move between modes in this time. Had the modes unequal weights, the broad start would still split about evenly (:numref:`sec_diffusion-denoising` measures this)." In "Limits of a Single Noise Level", make the count three. Deferred (experiment): give the running mixture unequal weights here, or only in this cell.

**SMD-7** | minor | thought
Anchor (Limits of a Single Noise Level): "forward processes give a Gaussian kernel, and hence an explicit DSM target,"
The subsection ends on the question "Which forward processes give a Gaussian kernel ... at every level?" The next subsection answers "Both standard forward processes", which is not an answer to "which". The reader is owed one sentence: any SDE whose drift is linear in $\mathbf{x}$ and whose noise amplitude does not depend on $\mathbf{x}$ (`sec_mdl-sdes` proved the Gaussian transition for the OU case).
Fix: either answer in that form at the start of "Forward Processes and the Noise-Conditional Loss", or end the previous subsection on a statement ("A forward process supplies both: a continuum of levels, and its reverse is the sampler.").

**SMD-8** | moderate | explanation
Anchor (DDPM subsection): "noise-prediction form, and ancestral sampling, which steps through the"
"Ancestral sampling" is named here, used as the benchmark at the start of the next part ("Ancestral sampling calls the network once per level"), placed at "the other end" of the DDIM family, and run in `cfg_sample`. Its update is never written in the prose. The only definition is the code line `xt = (xt + b * (...)) / np.sqrt(1 - b)` plus `np.sqrt(b)` noise. DDPM is described as three things (forward process, loss, sampler) and three propositions cover the first two.
Fix: display the step after the third proposition,
$\mathbf{x}_{t-1} = (\mathbf{x}_t + \beta_t\, \mathbf{s}_{\boldsymbol\theta}(\mathbf{x}_t, t))/\sqrt{1 - \beta_t} + \sqrt{\beta_t}\, \boldsymbol\xi$,
with one sentence: to first order in $\beta_t$ it is the Euler-Maruyama step of the reverse SDE :eqref:`eq_mdl-score-samplers` for the VP process.

**SMD-9** | moderate | explanation
Anchor (DDPM subsection): "the step $\Delta t = T/N$ and $\beta_t = \beta((t-1)\Delta t)\, \Delta t$, the"
In this subsection $t$ is an integer level from 1 to $N$; in the subsection before and in the proof of the DDIM proposition it is continuous time; here both appear in one formula ($\beta_t$ the number, $\beta(\cdot)$ the function, $t$ an index inside a continuous argument). The first proposition repeats the clash ("a time interval on which $\int \beta(r)\, dr = -\log(1 - \beta_t)$"). Two further traps for a reader who opens the DDPM paper: this section's $\alpha_t$ is $\sqrt{\bar\alpha_t}$, while the paper's $\alpha_t$ is $1 - \beta_t$; and the section's simple loss carries a factor $\tfrac12$ the paper's does not.
Fix: index the discrete levels by $k$ (or $i$) throughout the DDPM and DDIM subsections and reserve $t$ for continuous time; add one sentence: "Ho et al. write $\alpha_t$ for $1 - \beta_t$; our $\alpha_t$ is the signal coefficient $\sqrt{\bar\alpha_t}$." Chapter 17 uses continuous $t$ with $\bar\alpha_t := \alpha_t^2$, so the change also brings the two chapters into line.

**SMD-10** | minor | correctness (confidence medium)
Anchor (Forward Processes and the Noise-Conditional Loss): "forward SDE, ties the loss to the model's likelihood. With this weight, $T$"
The bound of exercise 9 is on the explicit score error $\tfrac12\int g^2\, \mathbb{E}\|\mathbf{s}_{\boldsymbol\theta} - \nabla\log p_t\|^2\,dt$. "The loss" in this sentence is the DSM loss :eqref:`eq_mdl-ncsm-loss`, which exceeds the explicit error by the floor. The statement is still true as an upper bound, but by the section's own exercise 5 the floor behaves like $d/t$ under $w = g^2$ and is not integrable at $t = 0$, so on $[0, T]$ the right-hand side is infinite and the bound says nothing.
Fix: "With this weight, $T$ times the excess of the loss over its floor, plus the KL divergence ..., bounds ...", and one clause that the floor itself diverges at $t \to 0$ under this weight, which is why training starts at a small $\varepsilon > 0$ (exercise 5).

**SMD-11** | minor | language (repetition, guide §7.7)
Anchor (DDPM subsection): "The name \"variance-preserving\" follows: for unit-variance data,"
The reason for the name is given in `mdl-sdes.md` ("This is what **variance-preserving** (VP) means"), again 150 lines earlier in this section ("so that unit-variance data keep unit variance at every $t$"), and here a third time with the computation.
Fix: delete the sentence here; keep the kernel identification $\alpha_t = \sqrt{\bar\alpha_t}$, $\sigma_t = \sqrt{1-\bar\alpha_t}$ and "because the kernel is Gaussian, DSM applies directly".

**SMD-12** | minor | thought, length
Anchor (DDPM subsection): "$\beta_t/(2(1 - \beta_t)(1 - \bar\alpha_t))$ when the reverse step has"
The weight formula of the variational route is a detail of `sec_diffusion-design-space`, which the same sentence cites. Here it is an unexplained expression (which reverse variance, why that ratio) at the end of a paragraph whose job is to say that DDPM was first derived differently.
Fix: end the paragraph at "the route that :numref:`sec_diffusion-design-space` follows; it yields the same regression with a different weight on each level."

**SMD-13** | moderate | explanation
Anchor (DDIM subsection): "the last under the standard schedule. The bar means \"divided by"
Three bars with three meanings are in play at once: $\bar\alpha_t$ (a cumulative product, where the bar does not mean "divided by $\sqrt{\bar\alpha_t}$"), $\bar{\mathbf{x}}_t$ and $\bar\sigma_t$ (rescaled), and $\bar{\mathbf{W}}$ (backward Brownian motion, 200 lines earlier); the previous section also has $\bar t = T - t$. The sentence that explains the bar is true of two of the five.
Fix: say it exactly: "We mark the rescaled point and its noise level with a bar, $\bar{\mathbf{x}}_t$ and $\bar\sigma_t$; the bar on $\bar\alpha_t$ is DDPM's and means something else." Chapter 17 (`samplers.md`) uses the same $\bar\sigma$, $\bar{\mathbf{x}}$, so a change of symbol would have to be made in both places; the sentence is the cheap fix.

**SMD-14** | moderate | thought (claim against stored output)
Anchor (Guidance with Bayes' Rule): "class boundary, and narrows it slightly, from $0.24$ to about $0.23$."
Stored output: std $0.243$ at $\gamma = 1$, $0.236$ at $\gamma = 3$, $0.237$ at $\gamma = 10$. To two digits these are $0.24$, $0.24$, $0.24$; the narrowing is 3% and does not grow with $\gamma$. The subsection then concludes "at the cost of diversity". The one-dimensional example shows the shift of the mode ($0.970 \to 1.03 \to 1.08$) clearly and the loss of diversity hardly at all, since the class has one mode and nothing to lose.
Fix: "and leaves its width almost unchanged ($0.243$ to about $0.237$)". Replace the last sentence by what the example shows: "Guidance with $\gamma > 1$ thus moves samples away from the other class, beyond where the class itself sits. A class with a single Gaussian mode has little diversity to lose; on images the loss is measurable (:numref:`subsec_diffusion-images-guided`)."

**SMD-15** | moderate | pedagogy, scope
Anchor (Guidance with Bayes' Rule): "noise: no eigenvalue of $-\nabla^2 \log p_t$ exceeds $1/\sigma_t^2$. Raising"
The one piece of mathematics in the guidance subsection, and the one chapter 17 cites ("on the mixture of the appendix it does so at most noise levels"), is asserted in the body and proved only in exercise 8. It is a two-line consequence of Tweedie's formula differentiated once: $\nabla^2\log p_t = -\mathbf{I}/\sigma_t^2 + (\bar\alpha_t/\sigma_t^4)\,\mathrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$, and a covariance is positive semidefinite. I checked the thresholds: $1 + v_0 v_t/(\sigma_t^2 m^2)$ gives $1.0625$ at the noisiest level, $1.09$ at $t = 100$, and falls below 3 from the tenth level on, so "at most noise levels" is right (99% of them).
Fix: promote the bound to a proposition with its two-line proof, placed with Tweedie's formula in "Denoising Score Matching" (it is a property of noised marginals, not of guidance), and keep the mixture thresholds as its example. Then the guidance subsection can shrink to a paragraph or move to chapter 17 (structural finding 2) without leaving chapter 17's citation dangling.

**SMD-16** | minor | language (guide §12.2, closings)
Anchor (end of body): "We can now learn a score from samples by least squares, at one noise level or"
The body ends with a recap sentence in the fixed form "We can now ...", followed by a Summary that restates the same results. Every section of this chapter closes with "We can now" (eight occurrences in six sections). The rest of the paragraph is a good bridge (the forward process was chosen for its kernels, not for easy integration) and should stay.
Fix: delete the first sentence; begin at "Every sampler calls the network many times".

**SMD-17** | minor | language
Anchor (after eq_mdl-score-samplers): "The next section takes $T = 1$, so that the noise sits at $t = 1$ for scores"
A note about the next section's convention interrupts the paragraph that states the section's organizing idea ("A generative model is thus a triple") and introduces the figure. The next section has a callout for exactly this.
Fix: delete here.

**SMD-18** | minor | language (pattern)
Exercises refer to cells by internal ID in running text, for example "`#mdl-score-matching-diffusion-flow-ddim-trading-noise-for-speed`" and "`#mdl-score-matching-diffusion-flow-guidance-steering-with-bayes-rule`" (five in this section). Two prefixes are in use (`score-matching-diffusion-flow-` and `mdl-score-matching-diffusion-flow-`), and the IDs carry old heading text ("trading noise for speed", "steering with Bayes' rule") that no longer matches any heading. IDs are frozen by house rule, so the fix is in the prose: refer to the cell by what it defines ("the cell that defines `ddim`"), as exercise 10 already does with `langevin`, `ddim`, `cfg_sample`.

## Checks made

Recomputed by hand or numerically, all in agreement with the text: the implicit objective and its minimum; the floor $3.2$ and slope $0.8$; the mixture score $4\tanh(4x) - 2x$ and its values $\pm 2.9$ at $\pm 0.45$; the unimodality threshold $\sigma \ge 1.94$; the stationary variance $v/(1 - h/(4v))$; the crossing rate of about 1% (Kramers' estimate gives $0.8\%$ in 2,000 steps); the exact-transition interval $0.0202$; the standardized mixture ($\pm 0.970$, $0.243$) and its level-100 values ($0.897$, $\pm 0.92$, $0.40$); $\bar\sigma$ from $0.01$ to $157$; all three steps of the DDIM proof; the Gaussian factor $\cos(\phi_t - \phi_{t'})$; the guidance thresholds; the variational weight formula against Ho et al. Stored PyTorch outputs agree with the prose except as noted in SMD-3 and SMD-14. Figures `mdl-dyn-tweedie` and `mdl-dyn-ddim-strides` were rendered; both match their captions (the posterior in the Tweedie figure has modes at $-0.9$ and $1.1$ and mean near $0.77$, as it should for $\tilde x = 0.2$).

Not checked: the TensorFlow, JAX and MXNet outputs (the cells are framework-free, so they should be identical in distribution); the statements quoted from Kynkäänniemi et al. and Song et al. (2021) in exercises 5 and 7.

## Exercises

Ten problems, inside the band of 8 to 12, graded from a pencil exercise to a small project. The set is strong.

- **Keep as they are**: 1 (boundary terms; the third part connects to pixels), 2 (discretization bias; part 2 asks when an experiment can resolve a bias, which is the right habit), 3 (annealing), 5 (weightings; it carries the divergence that SMD-10 needs), 8, 9.
- **Fix**: 3 should use, or add a part with, unequal mode weights, so that "the fraction of chains with $X > 0$" tests something symmetric procedures do not get for free (SMD-6).
- **Trim**: 4 (the Bayes-risk floor). Part 1 is the derivation the body cites and must stay. Parts 2 to 4 are about the standard error of two printed averages and the variance of their difference; they are careful, but they are a statistics exercise about one printout, three parts long. Keep part 1 and one of the others.
- **Trim or move**: 6 (DDIM against Heun) has four parts, the last two of which need new reference solutions at three component variances and a stability argument. The next section's body runs Euler against Heun on the same mixture, and chapter 17 does it on images. Keep parts 1 and 2 here; move 3 and 4 to wherever the solver comparison ends up (see the flow-matching review, FMT-11).
- **7** part 3 quotes an empirical finding on image models and asks which parts a one-dimensional mixture can reproduce; fine, but it depends on SMD-14's correction to what the example shows.
- **10** is the section's missing experiment. If the experiment is added to the body (SMD-1), parts 2 and 3 remain a good exercise.

## Still open from October

The October review asked to split this section into diffusion and flow matching (done), to cut EDM, log-SNR, v-prediction, stochastic interpolants and interacting particles (done: moved to Further Reading or to the next section's table), and to remove the notation collisions (mostly done through the "one clock" convention; the discrete/continuous $t$ and the bars remain, SMD-9 and SMD-13). Its last request, "coordinate it with ch. 17, which covers the same ground", is still open and is now the main structural question.

## For the chapter review

- **Fit.** The section follows directly from the Fokker-Planck section's closing ("the only unknown is the score") and hands the next section a real question (the paths were chosen for their kernels, not for integration). The order is right. It assumes the OU kernel and Euler-Maruyama from `mdl-sdes.md`, the reverse SDE, probability-flow ODE and mixture score from the Fokker-Planck section, and the Fisher divergence from chapter 42; all are cited by label.
- **Appendix or main text.** With chapter 17 written, sections 5 and 6 of this chapter are the proof store for chapter 17 and also a second, smaller course on the same topics. The clean division: the appendix keeps theorems, their smallest worked cases and the exact-score experiments that isolate one error; chapter 17 keeps everything a practitioner chooses (guidance, weights, step grids, model families). From this section that means moving or cutting the guidance experiment and the DDPM weight history, promoting the curvature bound to a proposition, and renaming the section. `CLAUDE.md` still calls chapter 17 a placeholder; that line is stale.
- **Pattern: exact scores everywhere.** No sampler in this section, and only one experiment in the next (two moons), runs on a learned field. As a design this is defensible and even elegant, but it must be stated, and the neighbouring sentences that speak of trained networks must match.
- **Pattern: "We can now".** Every section of the chapter ends its body with a paragraph beginning "We can now ..." (found at `mdl-odes-solvers.md:1111` and `:1184`, `mdl-neural-odes.md:1052`, `mdl-sdes.md:1314`, `mdl-fokker-planck-probability-flow.md:1221`, this section `:1153`, `mdl-flow-matching-transport.md:1430`). The guide (§12.2) names this formula.
- **Pattern: symmetric running mixture.** The chapter's running example $\tfrac12\mathcal{N}(\pm 2, 0.5^2)$ is symmetric, so no experiment on it can detect an error in mode weights. Chapter 17 chose unequal weights for this reason. Worth checking in the Fokker-Planck section's reverse-sampling experiment too.
- **A wrong pointer in chapter 17**, found while tracing references: `chapter_diffusion-models/gaussian-paths.md:600` cites `sec_mdl-ddim-guidance` for "one step of length one sends every start to the mean of the data". That fact is in `subsec_mdl-fm-two-point` of the next section.
