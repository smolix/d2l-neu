# Chapter 43, Dynamics: writing review

Sources: the eight `.md` files of `chapter_mdl-dynamics/`, each read in full by a section reviewer up to the slides marker. Three reviewers worked in parallel: one on the introduction, the ODE section and the neural ODE section; one on the SDE and Fokker-Planck sections; one on the score-matching, flow-matching and further-reading sections. By their counts the chapter has about 43,000 words of body prose and mathematics and 13,000 words of exercises (63 problems), about 60 code cells, and a 1,500-word reading guide. For this file I read all eight section reviews, and checked the cross-section claims against the sources: the references between this chapter and `chapter_diffusion-models/` and `chapter_recurrent-modern/ssm.md`, the closing formula, the sentences about trained networks, the repeated sampler equations, and two stored outputs.

Per-section reviews:
[index](43-dynamics/index.md) ·
[mdl-odes-solvers](43-dynamics/mdl-odes-solvers.md) ·
[mdl-neural-odes](43-dynamics/mdl-neural-odes.md) ·
[mdl-sdes](43-dynamics/mdl-sdes.md) ·
[mdl-fokker-planck-probability-flow](43-dynamics/mdl-fokker-planck-probability-flow.md) ·
[mdl-score-matching-diffusion-flow](43-dynamics/mdl-score-matching-diffusion-flow.md) ·
[mdl-flow-matching-transport](43-dynamics/mdl-flow-matching-transport.md) ·
[further-reading](43-dynamics/further-reading.md)

**A correction to the reviewers' brief.** Two of the three reviewers were told that the main-text diffusion chapter is a placeholder. It is not. `chapter_diffusion-models/` holds ten files and about 83,000 words, and it cites this chapter 51 times (my count, below). The third reviewer found this and judged scope against the chapter as it stands. The scope statements in this file use the real state. Where the first two reviewers wrote that a use is "named without a label" they are right; where one wrote that the ODE section "is the only section of the chapter with main-text inbound references" he is wrong (see "Contradictions between section reviews"). The line in `CLAUDE.md` that calls chapter 17 a placeholder is stale.

## Verdict

This is a strong chapter, the mathematics is sound, and long stretches of it meet the bar. All three reviewers recomputed the proofs and worked numbers and none found an error in a proposition or a proof. What they found are loose sentences, hypotheses that do not cover the chapter's own examples, and one recurring gap (a solution of the Fokker-Planck equation is taken to be the marginal of the SDE without the uniqueness that licenses it). The chapter has a real line of argument, stated by the last reviewer in one sentence: integrate a field, train through a solver, add noise, follow the density, learn the score, choose the path. Each section ends on the question the next one takes up, and the hand-offs are the best in the book so far. The method is the one the Basics review asked for: a number predicted by hand, then a cell that prints it. The Fokker-Planck section, the first half of the score-matching section, the two-point example of flow matching and the first and third parts of the ODE section are pages one would be glad to find in MacKay.

It is not yet at the bar for four reasons. (1) The chapter was written as if it were the book's only treatment of diffusion, and it is not. Its last two sections are at once the proof store for chapter 17 and a second, smaller course on the same topics (guidance, annealed Langevin sampling, solver budgets, a table of model families). The division of labour has to be decided once and applied to both chapters. (2) The prose says a network is trained where none is. Every sampler in the score-matching section runs on an exact score, and the following section says "The previous section trained a network to predict the noise". (3) Several verification cells cannot fail or confirm a line proved just above, and the best-known failure of single-level Langevin sampling is hidden by a symmetric running example. (4) The routine around the mathematics is uniform: every section ends its body with a "We can now ..." recap followed by a Summary that says the same, almost every cell is introduced by a question, and results with a home elsewhere in the appendix are taught again here with a new example.

What matters most is the first point. It decides what is cut from sections 5 and 6, what must be promoted from exercises to the body, and what the further-reading file points to.

## Scores

As the section reviewers gave them, 1 to 5; 5 means the page would be at home in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](43-dynamics/index.md) | 4 | 3 | 3 | 4 | 4 |
| [mdl-odes-solvers](43-dynamics/mdl-odes-solvers.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-neural-odes](43-dynamics/mdl-neural-odes.md) | 4 | 4 | 4 | 4 | 3 |
| [mdl-sdes](43-dynamics/mdl-sdes.md) | 4 | 4 | 4 | 4 | 3 |
| [mdl-fokker-planck-probability-flow](43-dynamics/mdl-fokker-planck-probability-flow.md) | 4 | 4 | 5 | 4 | 4 |
| [mdl-score-matching-diffusion-flow](43-dynamics/mdl-score-matching-diffusion-flow.md) | 4 | 4 | 3 | 4 | 3 |
| [mdl-flow-matching-transport](43-dynamics/mdl-flow-matching-transport.md) | 4 | 4 | 4 | 4 | 4 |
| [further-reading](43-dynamics/further-reading.md) | 4 | 4 | 4 | 4 | 4 |

The three reviewers calibrated alike: nobody gave a 5 for language or enjoyment, and the differences between sections are in pedagogy and enjoyment, where the texts do differ. The one 5 (pedagogy, Fokker-Planck) and the one 3 among the long sections (pedagogy, score matching) are each argued in the section file and I see no reason to doubt either. The ODE reviewer notes that the first and third parts of his section are close to 5 for explanation and pedagogy and the middle part (linear stability) is a 3.

## Correctness

No proposition or proof in the chapter was found to be wrong. All findings with dimension `correctness` are listed here, the most serious first.

Gaps in arguments and hypotheses:

- **FPK-4** (moderate; confident that the argument has a gap, the result is right). Superposition shows that the mixture of evolved Gaussians solves the Fokker-Planck equation. It does not show that this solution is the SDE's marginal without uniqueness, which is never stated. A one-line probabilistic argument (condition on the component) gives the result without the PDE.
- **FPK-7** (minor; confident). The deterministic-twin proposition takes "a smooth positive solution of the Fokker-Planck equation" where it needs the marginal density of the SDE. One hypothesis in the statement is not used in the proof.
- **FPK-11** (minor; confident). The proposition on the family of reverse processes states no regularity hypotheses, though its proof uses them, and has the same solution-against-marginal wording.
- **FMT-5** (minor; confidence medium-high). The reflow proposition and the Benamou-Brenier theorem assume a velocity field that is Lipschitz uniformly in $t$. The field of the two-point example and of any data with atoms is not (the slope is unbounded as $t \to 0$), yet the text applies the proposition to the two points in the next paragraph.
- **FMT-14** (minor; confidence medium). The Benamou-Brenier theorem ends in an equality with a minimum that reads as unconditional. Under the stated definition of admissibility the minimum need not be attained, and it is not for the two points or the moons.
- **FMT-6** (minor; confidence high). Uniqueness for the continuity equation, on which part (i) of the reflow proposition rests, is a one-clause indication presented as a proof step.
- **SMD-5** (minor; confidence high that a sentence is missing). The Langevin proposition proves invariance (started in $p$, the process stays in $p$). The experiment that follows starts elsewhere and so tests convergence, which is not proved and not said to be unproved.
- **SMD-10** (minor; confidence medium). "ties the loss to the model's likelihood": the bound is on the explicit score error, and the denoising loss exceeds it by a floor that is not integrable at $t = 0$ under the weight $g^2$, so as written the bound says nothing on $[0, T]$.
- **SDE-9** (minor; confident). Itô's lemma is applied to $\log x$, which is not $C^2$ on the line, under an unstated assumption that $X_t > 0$.

False or overstated sentences:

- **SDE-11** (moderate; fairly confident). "This is also why data are normalized to unit" variance: pixel-space models scale images to $[-1, 1]$, which gives a variance well below one. True of latent diffusion only.
- **SDE-14** (minor; confident). Rademacher data "start as far from the Gaussian as unit-variance data can": false, since fourth moments of unit-variance laws are unbounded above.
- **ODE-5** (minor; sure). Linear fields "describe every other field near a point where it vanishes": only near hyperbolic fixed points, as the section says itself later.
- **NODE-3** (minor; fairly sure). "the cost we set out to avoid" confuses the $O(d^3)$ determinant with the $d$ reverse passes for a trace.
- **NODE-13** (minor; moderately sure; terminology). "Jacobi's formula" names two different identities, in the body and in exercise 9.
- **ODE-15** (minor; sure, pedantic). The stability region is defined with a strict inequality and A-stability with a weak one, so the "since" for backward Euler does not quite follow.
- **ODE-11** (minor; sure, pedantic). The transient-growth eigenvectors are $174.3^\circ$ apart as written; the lines they span are $5.7^\circ$ apart; a sum is described as a difference.
- **FRD-2** (minor; confidence medium). "Repeated reflow converges to a pairing whose straight paths do not cross" is stronger than what the chapter or the cited paper proves.

Findings that were not tagged `correctness` but concern truth:

- **Statements about the chapter's own content that are false.** "The previous section trained a network to predict the noise" (FMT-1; SMD-1; I confirmed the sentence at line 359 of `mdl-flow-matching-transport.md`, and two related ones at lines 35 and 499). "The previous section ended with two observations", one of which that section does not make (NODE-1). A figure caption says the Euler steps lie "on the exact flow" when the figure shows them leaving it (NODE-4).
- **Stored outputs that contradict the prose.** FPK-13: the prose says the reverse sampler "splits the mass within about $0.01$ of one half" and the committed output prints `mass on the left: 0.487` and a KS distance of `0.0148` against "usually below $0.01$" (I confirmed both numbers in the store; the reviewer's 40 reruns gave 0.492 to 0.507, so the stored run is an unlucky draw and needs re-capture, not new prose). SMD-14: guidance "narrows it slightly, from $0.24$ to about $0.23$" against stored $0.243$, $0.236$, $0.237$, and the conclusion "at the cost of diversity" is not shown by a class with one mode. SMD-3: fit and exact score print $2.15$ beside a floor of $2.11$ and the text calls both "about $2.1$".
- **Numbers that no cell prints.** NODE-14: the single-probe standard deviation "of about $4$ to $5$". FPK-14: finer steps "removes nearly nine tenths of the bias" (true, 87% by the reviewer's computation). FMT-12: about ten quantitative statements in two paragraphs that explain the solver table ("less than a tenth of their length", "about an eighth of their total size", and others). FMT-13: the minibatch-pairing numbers $0.06$ and "a few thousandths" come from an exercise, not from a cell (the reviewer ran the exercise and they are right). NODE-15: "for any draw of the weights" about an unseeded cell.
- **Checks that cannot fail.** SMD-6 (major): "P(X > 0) = 0.5 (truth 0.5)" holds by the symmetry of target and start; with weights $0.8/0.2$ the same sampler gives about $0.48$ against a truth of $0.2$. SDE-2, SDE-16, SDE-19 and structural finding 1 of the SDE review: cells that check a random number generator, a formula visible on sight, or an effect the cell is said to be unable to see.
- **Reasons that are too strong.** SDE-1 (noise alone presented as a failure, then as a standard family); SDE-5 (Itô's convention justified by "cannot look ahead", though Stratonovich equations are simulated forward routinely); SDE-17 (the orders of Euler-Maruyama "follow" from a count that ignores error propagation).

## Arc and order

**One line of argument: yes.** The sections are in the right order and every reviewer says so for his part. The introduction's route paragraph hides this by giving six sentences of one shape, one per section (IDX-1). The chapter has three movements of two sections each: deterministic motion and how to compute and differentiate it; noise and the density it produces, ending in the discovery that reversal needs one unknown field; learning that field, or a velocity in its place, from samples. The introduction should say so.

**The best link in the arc is not made.** The neural ODE section trains a ten-step model, integrates the same field with a thousand steps, and finds that the error rises tenfold: the trained field is tied to its training step count (NODE-S2). That is the strongest motivation in the chapter for learning a field by a regression that involves no solver, which is what sections 5 and 6 do. Neither end says so. One sentence in the neural ODE section and one back-reference at the head of the flow-matching section would tie the first third of the chapter to the last.

**Sections that are more than one.**

- `mdl-odes-solvers` is three sections under one title, each with its own `sec_` label and roadmap (ODE-S1). The main text points at the third part (`sec_mdl-euler-runge-kutta`, three times) and at the first once. A split after "Linearization at Fixed Points" into "Ordinary Differential Equations" and "Numerical Solvers" is the reviewer's proposal and I agree with it, the more so because the solver half is where the missing exact step for a driven linear equation belongs (ODE-S3).
- `mdl-flow-matching-transport` is one and a half sections. Its last `##` part, "Sampling with a Learned Field", leaves paths and pairings for solver budgets on the previous section's object (FMT-11, FMT-12, structural finding 1 of that review).
- `mdl-score-matching-diffusion-flow` is two halves of different character: score matching (the home of its theorems, tight) and DDPM, DDIM and guidance (three topics in sequence, partly taught again in chapter 17). After the moves proposed below it is "Score Matching" plus two identification theorems and should be titled for that (structural finding 2 of that review).
- `mdl-sdes` is four topics held together by one running recurrence; `mdl-neural-odes` and `mdl-fokker-planck-probability-flow` are each one section. None of the three should be split.

**Where boundaries leak.**

- The SDE section's OU cloud from $x_0 = 2$ is run again at the start of the Fokker-Planck section, and the kernel has two equation labels (`eq_mdl-sde-ou-kernel`, `eq_mdl-dyn-ou-marginal`) (FPK-2, structural finding 1 of that review).
- The probability-flow ODE and the reverse SDE are derived in the Fokker-Planck section, displayed again in the score-matching section (`eq_mdl-score-samplers`) and a third time, unlabeled, at the head of "Sampling with a Learned Field" (FMT-11; I confirmed the three displays).
- Solver comparisons at equal cost appear three times: RK4 against Euler in the ODE section, Euler against Heun on the VP probability-flow ODE in the flow-matching section, and DDIM against Heun as exercise 6 of the score-matching section. Chapter 17 (`samplers.md`, "Second Order and the Step Grid") runs a fourth.
- The dictionary subsection "Score, Noise, and Velocity" sits between the two-point example and the question that example ends on (FMT-4). It belongs directly after the conditional flow matching theorem.
- The notation table for the whole chapter sits in the first section, 21 of its 34 rows meaningless at that point (ODE-S5). It should move to the introduction or the end. `chapter_notation/index.md` cites its label `tab_mdl-dyn-notation`, so the label must move with it.
- The continuity equation is named in the neural ODE section and derived two sections later with no forward pointer (NODE-17).
- The vector-calculus preliminaries (divergence, Laplacian, integration by parts) live in the Fokker-Planck section; the calculus chapter has none of them and does not point here (structural finding 3 of that review). They are used at once and can stay, with a pointer from the calculus chapter.

## The most important findings

1. **Decide the division of labour with chapter 17 and apply it.** The appendix keeps theorems, their smallest worked cases, and the exact-field experiments that isolate one error. Chapter 17 keeps what a practitioner chooses: guidance, weights, step grids, model families. From this follow: the guidance subsection shrinks to the curvature bound (SMD-15, structural finding 2 of the score-matching review); the DDPM weight history goes (SMD-12); "Deterministic and Stochastic Samplers" and "Comparison of Model Families" go (FMT-11); the solver-order experiment gets one home (FMT-12); two further-reading entries become pointers (FRD-1). Details in "Connection to the main text".
2. **Nothing is trained in sections 5 and 6 except the two-moons flow, and the prose says otherwise** (SMD-1, SMD-2, FMT-1). The design (exact scores, so that every measured error is the sampler's own) is defensible and even elegant. It must be stated once, in the chapter introduction and in the opening of the score-matching section, and the three sentences about trained networks must be corrected.
3. **The symmetric running mixture hides the failure it should show** (SMD-6). The half-and-half split under Langevin sampling is inherited from the start, not produced by the dynamics. Chapter 17 uses unequal weights for exactly this reason. The same check, "mass on the left ... (target 0.5)", appears in the Fokker-Planck section's reverse sampler (the cell of FPK-13) on the same symmetric mixture, where it has the same weakness: it can detect sampling noise and nothing else. The Fokker-Planck reviewer did not raise this. Unequal weights in the running mixture would repair both; that is an experiment change and is deferred under the author's decision, but the prose can say now what the runs show.
4. **The Hutchinson experiment appears to show that the method does not work** (NODE-14). It uses $10^5$ probes at $d = 50$, where the exact trace costs 50 passes, and the text does not say why one probe per step suffices in training (unbiasedness inside a stochastic gradient). Chapter 17's `design-space.md` cites both `eq_mdl-ode-hutchinson` and the exercise on its variance, so the main text leans on the very point the body leaves open.
5. **Uniqueness for the Fokker-Planck equation is used and never stated** (FPK-4, FPK-7, FPK-11, with FPK-9 on the status of Anderson's theorem). One sentence under "From Itô's Lemma to the PDE" covers all three; the section is otherwise the model of honesty about what is proved.
6. **The driven linear ODE and its exact step are missing** (ODE-S3, IDX-3). The State Space Models chapter builds on exactly this and has no reference to the appendix. See "Connection to the main text".
7. **Two experiments re-teach other appendix chapters** (ODE-17, major: the learning-rate sweep of the optimization chapter, same quadratic, same start, same ceiling; ODE-10: transient growth from the linear algebra chapter, in full with a second matrix). By guide section 7.7 each has one home, and it is not this chapter.
8. **Confirmation cells and repeated identifications in the SDE section** (structural findings 1 and 2 of that review; SDE-12, SDE-13, SDE-16, SDE-19). Five of thirteen cells verify what was just proved. Cutting them and the fourfold re-identification of the opening recurrence takes the section from 8,300 words to under 7,000 and leaves the four experiments that teach.
9. **Explanations no cell supports, and a conclusion wider than its evidence** (FMT-12). "Where the steps go matters more than the order of the solver" rests on one one-dimensional mixture under one schedule, explained by 27 lines of unmeasured mechanism. This is the Basics pattern in its clearest form in the chapter.
10. **The decisive counterexample of the transport part is in an exercise** (FMT-10, FRD-3). The straight quarter-turn pairing (straight, constant speed, crossing-free, a fixed point of reflow, cost 4 against $W_2^2 = 0$) is what shows that straight and crossing-free is not optimal. The body has a rotating flow instead, and the further-reading file cites the section for an example a reader will not find in its body.

## Repetition

Inside the chapter:

| What | Where | Proposed single home |
|---|---|---|
| OU cloud from $x_0 = 2$ against the kernel; the kernel under two labels | SDE section, Fokker-Planck section (FPK-2) | SDE section; the Fokker-Planck section keeps the histogram and cites `eq_mdl-sde-ou-kernel` |
| "The recurrence is Euler-Maruyama for OU", four times; the fixed point $1.005$, three times | SDE section (SDE-12) | The OU definition; the fixed point under "The Stationary Distribution" |
| Why the VP process is called variance-preserving | SDE section, score-matching section twice (SMD-11) | SDE section |
| Probability-flow ODE and reverse SDE displayed | Fokker-Planck, score matching, flow matching (FMT-11) | Fokker-Planck section; later uses by label |
| Regression lemma displayed in full twice | Score matching, flow matching (FMT-3) | Score-matching section |
| Discrete adjoint against continuous adjoint, stated four times in a page and a half | Neural ODE section (NODE-11) | Once, after the two recursions |
| The cost of a $3072 \times 3072$ determinant | Neural ODE section, three times (NODE-2) | Where the formula is |
| What a Neural ODE gains, listed three times before any is shown | Neural ODE section (NODE-S1) | The subsection that takes the limit |
| Solver comparison at equal cost | ODE section, flow-matching section, score-matching exercise 6 (FMT-12) | RK4 against Euler stays in the ODE section; the diffusion-specific comparison goes to `samplers.md` in chapter 17, which already runs it |
| Chapter opening | Introduction and first section (IDX-4, ODE-1) | Introduction |
| Each result in body, "We can now" recap, and Summary | All six sections (ODE-18, NODE-18, SDE-20, FPK-15, SMD-16, FMT-15) | Body and Summary; drop the recap, keep the hand-off sentences |

With other appendix chapters:

| What | Where else | Proposed home |
|---|---|---|
| Learning-rate sweep on $\mathrm{diag}(10, 1)$ | `sec_mdl-gradient-based-optimization` (ODE-17) | Optimization chapter; this chapter keeps the three-step derivation and the identifications "condition number is stiffness", "divergence is solver instability" |
| Transient growth | `subsec_mdl-transient-growth` in linear algebra (ODE-10) | Linear algebra; keep here the closed form $10(e^{-t} - e^{-2t})$ and a pointer |
| Matrix exponential: series, eigenvalue formula, stability reading | `mdl-eigendecomposition.md` (ODE-6) | This chapter (definition and proofs); the linear algebra paragraph becomes a pointer |
| The score and its geometry | `sec_mdl-fisher-divergence` in information theory (noted under "Still open from October" in the Fokker-Planck review) | One of the two; the Fokker-Planck section already points there |
| Divergence, Laplacian, integration by parts | Nowhere else; a reader will look in Calculus | Stay here, with a pointer from the calculus chapter |

With the main text: see the next section.

## Connection to the main text

### The diffusion chapter (17)

`chapter_diffusion-models/` consists of `index.md`, `denoising.md`, `gaussian-paths.md`, `image-diffusion.md`, `samplers.md`, `latent-diffusion.md`, `discrete-diffusion.md`, `few-step.md`, `design-space.md` and `further-reading.md`. Its introduction states the contract: "The mathematics lives in :numref:`chap_mdl-dynamics`, to which the chapter points instead of repeating it".

**What it cites.** A grep of the ten files (text before the slides marker) finds 51 citations of 36 distinct labels defined in this chapter. All resolve. By home section:

| Appendix section | Citations | Labels cited from chapter 17 |
|---|---|---|
| Score matching | 20 | `sec_mdl-ddim-guidance` (4), `sec_mdl-denoising-score-matching` (3), `sec_mdl-score-based-generative-modeling` (2), `sec_mdl-langevin-sampling` (2), `sec_mdl-ddpm-discretized-sde` (2), `sec_mdl-score-matching`, `fig_mdl-dyn-tweedie`, `fig_mdl-dyn-ddim-strides`, and four exercises (Bayes-risk floor, annealing, guided marginals, guidance levels) |
| Flow matching | 12 | `subsec_mdl-fm-two-point` (2), `sec_mdl-flow-matching` (2), `tab_mdl-score-noise-velocity`, `subsec_mdl-reflow`, `sec_mdl-rectified-flow`, `sec_mdl-sampling-learned-dynamics`, `sec_mdl-flow-matching-transport`, `fig_mdl-dyn-time-conventions`, and two exercises (late decisions, target conditioning) |
| Fokker-Planck | 8 | the mode-merging exercise (3), `sec_mdl-time-reversal` (2), `sec_mdl-probability-flow-ode`, `sec_mdl-fokker-planck-probability-flow`, `fig_mdl-dyn-lambda-family` |
| ODEs and solvers | 4 | `sec_mdl-euler-runge-kutta` (3), `sec_mdl-vector-fields-trajectories` |
| Neural ODEs | 3 | `sec_mdl-continuous-normalizing-flows`, `eq_mdl-ode-hutchinson`, the Hutchinson-variance exercise |
| SDEs | 2 | `sec_mdl-ornstein-uhlenbeck`, `sec_mdl-euler-maruyama` |
| Further reading, chapter label | 2 | `sec_mdl-dynamics-further-reading`, `chap_mdl-dynamics` |

By citing file: `gaussian-paths.md` 22, `denoising.md` 7, `few-step.md` 6, `design-space.md` 5, `samplers.md` 5, `image-diffusion.md` 4, `index.md` and `further-reading.md` 1 each. `latent-diffusion.md` and `discrete-diffusion.md` cite other appendix chapters only. Outside chapter 17 the only main-text citations are `chapter_generative-adversarial-networks/convergence.md` (`sec_mdl-ot-connection`, for Benamou-Brenier), the preface and the notation page.

**What the appendix must therefore supply**, with proofs, and must not lose in revision:

- Denoising score matching: Hyvärinen's identity, the regression lemma, Vincent's theorem, Tweedie's formula and its figure, the loss floor.
- Langevin dynamics: stationarity, discretization bias, and (missing today, SMD-5) one sentence on convergence.
- The identification theorems: DDPM is the discretized VP process; DDIM is Euler's method in rescaled coordinates, with its figure.
- The curvature bound on noised marginals, which chapter 17's guidance section relies on and which the appendix proves only in an exercise (SMD-15).
- Time reversal, the family of reverse processes and its figure, and the probability-flow ODE.
- The OU kernel and Euler-Maruyama.
- The marginal-velocity proposition, the conditional flow matching theorem, the two-point example, the score-noise-velocity table, the "one clock" figure, reflow, and Benamou-Brenier.
- Euler, Heun and their orders; the instantaneous change of variables and Hutchinson's estimator.

**Ten of the 51 citations point at exercises.** The main text rests on results that the appendix sets as problems: when two Gaussian modes merge (cited three times), the variance of Hutchinson's estimator, the Bayes-risk floor, annealing, guided densities and guidance levels, late decisions, the conditioning of prediction targets. This supports three section findings from a direction their authors could not see. The mode-merging criterion, the Hutchinson variance (NODE-14) and the curvature bound (SMD-15) should be stated in the body, as a proposition or a worked remark, and the exercise kept as its extension. It also constrains the exercise edits proposed in the section files: any exercise that is trimmed or reworked (exercises 3, 4 and 6 of the score-matching section) must keep its label and the part chapter 17 uses.

**What is duplicated**, that is, taught at about the same level in both places:

- Guidance. `image-diffusion.md` ("Bayes' Rule on Scores", "Classifier-Free Guidance", "Guidance as a Tilt") gives the derivation of the appendix's "Guidance with Bayes' Rule" in nearly the same words and then runs it on images. Home: chapter 17. The appendix keeps the bound.
- Limits of one noise level and annealing. `denoising.md` has a `##` part with the same heading as the appendix subsection ("Limits of a Single Noise Level"), and `gaussian-paths.md` has "Annealing through Noise Levels". Chapter 17's version is the better one, because its mixture has unequal weights (SMD-6). Home: chapter 17; the appendix keeps the unimodality threshold and the bias formula.
- Marginal velocity, the conditional flow matching identity and the score-velocity identity. `gaussian-paths.md` ("Conditional Targets and Marginal Fields", "Four Targets, One Oracle") states each with a one-line reason and a display, so on these three the main text is not lighter than the appendix, only without proofs (reported in the flow-matching review). This is acceptable under the contract if `gaussian-paths.md` shortens its reasons to pointers; it is a matter for a revision of chapter 17, not of this chapter.
- Solver order and step grids. The appendix's "Solver Order at Equal Cost" and `samplers.md` ("Second Order and the Step Grid"), which quotes the appendix result and then runs its own. Home: chapter 17 (FMT-12). Note that `samplers.md` cites `sec_mdl-sampling-learned-dynamics`, so if that part is cut the citation must be redirected.
- The table of model families. `sec_mdl-unifying-table` in the appendix and `tab_diffusion_models` in `gaussian-paths.md`. Home: chapter 17 (FMT-11).
- The DDPM variational weight. Appendix (SMD-12) and `design-space.md`. Home: chapter 17.
- Further reading. The appendix file presents the design space and few-step generators as "topics they had no room for"; `sec_diffusion-design-space` and `sec_diffusion-few-step` are full sections (FRD-1).

**What the appendix teaches that chapter 17 does not use**, and which therefore stands on other grounds or on none: the adjoint method and its drift; order preservation; linear stability, phase portraits, transient growth, stiffness and implicit methods (not cited from chapter 17, although its samplers face stiffness near $t = 0$); Brownian motion, quadratic variation, the Itô integral and Itô's lemma (needed inside the appendix to derive what chapter 17 cites); the derivation of the Fokker-Planck equation itself. None of this is a reason to cut, but it shows where length is spent for the appendix's own sake, and it bears on ODE-S2.

**Missing or wrong references.**

- Appendix to main text. The whole chapter has six references to main-text labels: `chap_diffusion` four times (introduction, score matching, flow matching, further reading), `sec_diffusion-design-space` once, and `subsec_activation_checkpointing` once. The ODE, SDE and Fokker-Planck sections have none. Uses named without a label: "diffusion-model samplers use this freedom", "how likelihoods of diffusion models are reported", "the forward step of DDPM", "Many diffusion samplers use this two-stage method", residual blocks, gradient descent (ODE-S4, ODE-13; structural finding 5 of the SDE review and of the Fokker-Planck review). Each should get the label of the chapter 17 section that already points back: `sec_diffusion-samplers`, `sec_diffusion-gaussian-paths`, `sec_diffusion-denoising`, `sec_diffusion-images`.
- Main text to appendix, wrong. `gaussian-paths.md` cites `sec_mdl-ddim-guidance` for "one step of length one sends every start to the mean of the data". That fact is in `subsec_mdl-fm-two-point` (reported in the score-matching review; I confirmed the passage).
- Residual networks. `chapter_convolutional-modern/resnet.md` does not mention the Euler reading of a residual block (I found no occurrence of "Euler" or "differential" in it), and the neural ODE section does not cite the residual-network section by label (ODE-S4 and the closing notes of the neural ODE review). One reference in each direction.
- Inside the appendix, stale. `mdl-matrix-calculus-autodiff.md` cites `sec_mdl-odes-solvers` for the adjoint method, which now lives in `sec_mdl-adjoint-method` (reported in the ODE review; the citation is still there).
- Exercise labels. Seven of the nine exercise labels of the neural ODE section carry the prefix `ex_mdl-odes-solvers-`, one flow-matching exercise carries the score-matching prefix, and chapter 17 cites one of the former (`ex_mdl-odes-solvers-hutchinson-variance`). Invisible to readers; a rename would have to update chapter 17.

### The State Space Models chapter (12)

`chapter_recurrent-modern/ssm.md` announces that it will "write the recurrence as a differential equation that is sampled at discrete steps". Its subsection "Discretization" (`subsec_zoh`) takes the scalar equation with an input, holds the input fixed over a step (zero-order hold) and solves the step exactly: $\bar a = e^{\Delta\lambda}$, $\bar b = (e^{\Delta\lambda} - 1)/\lambda$. It then argues that every $\lambda < 0$ gives $\bar a \in (0, 1)$ for every step size, allows complex rates ("the state decays while it rotates"), and sets an exercise that compares the zero-order hold with forward Euler, backward Euler and the bilinear rule and asks which can make a stable continuous system unstable.

Every one of these is the subject of the ODE section, and the connection is absent in both directions. I confirmed that `chapter_recurrent-modern/` contains no reference to any `mdl-` label, and that the appendix chapter contains neither a driven linear equation, nor variation of constants, nor the exact step: "integrating factor" occurs once, in the proof of the OU kernel in the SDE section, as a tool "that solves the decay ODE" which no earlier section has shown; exact transitions and exponential integrators occur only in an SDE exercise and in the further-reading file. The introduction names diffusion as the chapter's one customer (IDX-3).

What the SSM chapter needs from here, in about half a page (ODE-S3):

1. The solution of $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{b}$ by variation of constants, and the exact step over an interval $h$ with the input held fixed.
2. The observation that the exact factor $e^{h\lambda}$ lies inside the unit circle for every $h$ whenever $\operatorname{Re}\lambda < 0$, set beside Euler's $1 + h\lambda$ and backward Euler's $1/(1 - h\lambda)$ in the existing stability figure.
3. A label for it, a reference from `subsec_zoh` to that label, and a sentence in this chapter's introduction that names the SSM chapter as a reader.

The same half page serves three other places: the OU proof (which uses the integrating factor), the exact DDPM transition (SDE-15), and the exponential integrators of the diffusion samplers, which the further-reading file describes.

This need qualifies ODE-S2. The reviewer proposes to shorten the linear-stability part because "nothing downstream uses" the phase portraits, Jordan blocks and transient growth. That is true for the rest of this chapter and for chapter 17. But "the real part decides decay", complex eigenvalues as a damped rotation, and the stability regions are exactly what the SSM chapter relies on, so the cut should fall on the portrait table, the Jordan paragraph and the duplicate transient-growth treatment, and the space should go to the driven equation.

### The optimization chapters

The ODE section's "Gradient Descent as a Solver" names no main-text label, and the optimization chapter of the main text does not point here for the reading of a diverging learning rate as a solver instability (ODE-S4). Exercise 5 of the ODE section (implicit gradient regularization) is the best connection to deep learning in that section and is not mentioned in the body.

## Recurring patterns

**Closings.** Every one of the six sections ends its body with a paragraph that begins "We can now ...", followed by a Summary that restates the same results: eight occurrences in six sections, which I confirmed by grep (two each in the ODE and Fokker-Planck sections). Findings ODE-18, NODE-18, SDE-20, FPK-15, SMD-16, FMT-15. In each case the last two or three sentences of the paragraph are a real hand-off and should stay.

**A question before every cell.** 5 of 5 cells in the ODE section and 6 of 6 experiment cells in the neural ODE section (ODE-20, NODE-19); 9 of 13 in the SDE section, with the fixed continuation "The next cell ..." (structural finding 1 of that review). The method is right and should not be removed. As the only way to introduce a cell it is a visible template by the third use, and where the answer has just been proved the question is empty. State the prediction as a claim some of the time.

**Verification cells that cannot fail or that confirm a proof.** SDE section: `#sdes-brownian-paths`, `#sdes-brownian-covariance`, `#sdes-isometry-check`, `#sdes-euler-check`, `#sdes-ou-stationary` (SDE-2, SDE-16, SDE-19). Score matching: the half-and-half split (SMD-6), the KS cell that re-verifies the $\bar\alpha$ marginal after a four-line induction. Flow matching: the cosine-path check of an identity proved six lines earlier (FMT-4). Fokker-Planck: the mass split of the reverse sampler on a symmetric target (my observation, see finding 3). Against these stand cells that do teach by the same method (the exact bias recursion, the weak order without sampling, the ten-step model at a thousand steps), so the fault is selection, not method.

**Exact fields everywhere.** No sampler in the score-matching section runs on a learned field, and only two moons does in the flow-matching section (closing notes of both reviews). State the design once.

**Opening phrases as signatures.** "The smallest case ..." opens four worked examples in the Fokker-Planck section and two in the SDE section (FPK-16). "The one step that is not routine ..." and its variants, four times across the first two sections (ODE-7). Stacked roadmaps, five in the ODE section (ODE-19).

**"Proposition" for three kinds of statement.** Proved results, Itô's lemma (heuristic argument, proof cited) and the orders of Euler-Maruyama (no proof) carry the same label (SDE-8). The Fokker-Planck section has the same issue with a subsection titled "Anderson's Theorem" that states no theorem (FPK-9), and the flow-matching section with Benamou-Brenier (FMT-14). One convention for the chapter: "Proposition ... Proof ... $\blacksquare$" for what is proved here, "Theorem (name; proof in ...)" for what is cited.

**Framework tabs on NumPy-only notebooks.** The SDE, Fokker-Planck and score-matching sections each load four frameworks in tabbed import cells and use none (structural finding 6 of the SDE review, 3 of the score-matching review). The flow-matching section spends about 330 source lines on four-tab copies of its training code, and its prose carries framework-neutral plumbing because of it (FMT-7). The house policy gives the Advanced part PyTorch and JAX only; this chapter is the companion of an Advanced chapter. A decision for the author.

**"For any draw" on unseeded cells** (NODE-15) and unseeded outcomes quoted as single figures in exercises ("settles near $0.03$").

**Notation drift.**

- $h$: step size in the first three sections and the notation table, a scalar field in the Fokker-Planck section, where $\Delta h$ is then its Laplacian a page after $\Delta t$ was an increment (FPK-3).
- $\theta$: the OU rate in the SDE section, the network parameters in the neural ODE section (closing notes of the SDE review).
- $\lambda$: an eigenvalue in the notation table, the sampler's noise scale in the Fokker-Planck section; $q$ has three meanings there (FPK-3).
- $t$: an integer level and continuous time in one formula in the DDPM subsection; this chapter's $\alpha_t$ is the DDPM paper's $\sqrt{\bar\alpha_t}$, unremarked (SMD-9).
- Bars: $\bar\alpha_t$, $\bar{\mathbf{x}}_t$, $\bar\sigma_t$, $\bar{\mathbf{W}}$, $\bar t$ with three meanings, and a sentence that explains the bar correctly for two of the five (SMD-13).
- Step counts: $n$, $N$, $k$, $l$ across the first two sections; `H` for hidden units against $\mathbf{H}$ for the Hessian (NODE-20).
- "Energy distance" for a quantity defined and plotted as its square (FMT-8).
- Exercises that refer to cells by internal ID with stale heading text (SMD-18).

**Prerequisites understated.** The introduction says the chapter assumes "the Basics part"; the first section uses complex eigenvalues, the eigendecomposition, the operator norm and several-variable Taylor expansions, and the last two use MMD, couplings and the Fisher divergence from chapter 42 (IDX-2). Name the appendix chapters the chapter builds on.

## What to cut

Summing the figures the reviewers give, roughly 5,000 to 6,000 words of body, none of it argument:

- **ODE section** (about 1,500 words): the gradient-descent sweep cell and its paragraph (ODE-17); transient growth down to the closed form and a pointer (ODE-10); the portrait table (ODE-9) and the Jordan paragraph (ODE-S2); the closing recap (ODE-18); exercise 2a and most of 8a, which the text answers.
- **Neural ODE section**: two of the three statements of the gains (NODE-S1); three of the four statements of the discrete-adjoint conclusion (NODE-11); the tab-bridging sentence (NODE-16); the recap (NODE-18).
- **SDE section** (about 1,300 words): the Euler check, the stationary-quantile cell, the Rademacher fourth-moment cell and table (SDE-13, SDE-16, SDE-19); one Brownian check merged into the other; the local-time paragraph (SDE-10); three of the four identifications of the recurrence (SDE-12); the variance-exploding paragraph down to a sentence (SDE-1).
- **Fokker-Planck section**: little. "Pathwise and Ensemble Views" down to its one new idea (FPK-2); the recap (FPK-15).
- **Score-matching section**: the guidance experiment and derivation (to chapter 17), keeping the bound as a proposition (SMD-15); the DDPM weight formula (SMD-12); the KS cell after the induction; the third explanation of "variance-preserving" (SMD-11); the note on the next section's clock (SMD-17).
- **Flow-matching section** (about 1,500 words): "Deterministic and Stochastic Samplers" down to a sentence and the model-families table (FMT-11); the two mechanism paragraphs (FMT-12); the plumbing before the training cell (FMT-7); the second display of the regression lemma (FMT-3); the fifth opening paragraph (FMT-2).
- **Further reading**: the two entries that chapter 17 now teaches, replaced by a pointer (FRD-1); the restatement inside the reflow entry (FRD-3).

## What is missing

- The driven linear ODE, variation of constants, and the exact (zero-order hold) step, with a reference from the SSM chapter (ODE-S3).
- A statement, in the introduction and at the head of section 5, that the chapter's samplers run on exact fields by design, and where the networks are trained (SMD-1).
- Uniqueness for the Fokker-Planck equation, as one sentence used without proof (FPK-4).
- Why the reverse process is legitimate although the backward heat equation is ill-posed (FPK-12), and that the equation is never solved on a grid in practice (today only in exercise 8d of that section).
- Why one Hutchinson probe per step suffices in training, and when the estimator is good or bad (NODE-14).
- What must change for "a step count chosen after training" to hold, and the link from there to simulation-free training (NODE-S2).
- The ancestral sampling step, written as an equation (SMD-8).
- Convergence of Langevin dynamics from other starts, as a cited statement (SMD-5), and blindness to mode weights as the third limit of one noise level (SMD-6).
- The curvature bound on noised marginals as a proposition with its two-line proof (SMD-15).
- The straight quarter-turn pairing in the body (FMT-10).
- Label references to the main text from the ODE, SDE and Fokker-Planck sections, and the Euler reading in `resnet.md`.
- In further reading: how accurate the score must be (sampling-error bounds in terms of the $L^2$ score error), predictor-corrector sampling, and Girsanov's theorem for the likelihood exercise (structural finding 2 of that review).
- In the introduction: the three movements, the real prerequisites, and the other chapters this one serves (IDX-1, IDX-2, IDX-3).

## What is best and should set the model

- **The opening of the ODE section.** $\dot x = -50x$, the factor $1 - 50h$, two step sizes, and "The equation is stable; the steps are not." The whole section in miniature before any definition, and then reused honestly in the stability figure and the stiffness table.
- **Bounds checked against the truth.** "the bound is $\tfrac e2(e-1)h = 2.3\,h$, against the true error $1.36\,h$: the right order, with a pessimistic constant." It teaches what a bound is.
- **The scalar adjoint.** One example worked to the end in closed form, the general proof that "keeps the shape of the scalar one", and a figure of a constant product. This is how a proposition in the appendix should be prepared.
- **The experiment that reports its own failure.** The ten-step model whose error rises at a thousand steps: "Training fitted the ten-step program."
- **The opening recurrence of the SDE section** and the sentence that follows it: the noise is of order $\sqrt h$, "its square is of order $h$, the same order as the drift. Ordinary calculus discards squares of small changes, so this one fact requires a new calculus."
- **Honesty about level in the SDE section.** "This is not a proof (it rules out a derivative at each fixed time, while "nowhere" concerns all times at once), but it shows the obstruction." What is proved, what is heuristic and what is cited are told apart in the prose. This should be the register of the whole part.
- **Diffusion as transport**, the centre of the Fokker-Planck section: the heat flow solved in closed form, "every particle doubles its distance from the center", and the factor of two explained by a process that "smooths its own density, so its drift must undo two smoothings".
- **Monte Carlo replaced by an exact recursion** where sampling error would hide the effect: the weak order of Euler-Maruyama and the width bias of the reverse sampler.
- **The Gaussian worked by hand, twice,** in the score-matching section: $J(w) = v/(2w^2) - 1/w$ with "the objective never saw the data score and still returns the variance", and the floor $3.2$ predicted and then printed as $3.207$.
- **Two points in one dimension** in the flow-matching section: a closed form, two values to check by hand, and two sentences a reader keeps ("One Euler step ... sends *every* sample to $0$"; "the flow chooses a mode late").
- **The reflow experiment.** A proposition, then a measurement in which every number in the prose is in the stored output, and candour about what the method cannot do.
- **The exercises.** Each of the three reviewers independently called the exercise set of one of his sections the strongest he had read. Across 63 problems almost none is a drill and almost all extend the text. The chapter's exercises are a model for the part.
- **The further-reading file**, where every entry states the one thing it changes relative to what the chapter taught.

## Contradictions between section reviews

- **Inbound references.** The ODE review says its section "is the only section of the chapter with main-text inbound references". The other two reviewers, and my grep, show that chapter 17 cites all six sections, the score-matching section most of all (20 of 51 citations). The ODE and neural ODE reviews' statements about the thin connection to the main text are right about outbound references and wrong about inbound ones.
- **Where the solver-order experiment goes.** The flow-matching review proposes to move it next to "DDIM is Euler's method" in the score-matching section or to chapter 17; the score-matching review (same reviewer) finds that section's second half already too long and proposes to move part of its exercise 6 "to wherever the solver comparison ends up". The two are consistent only if the home is chapter 17, which is what I propose above.
- **How much of the linear-stability part is unused.** ODE-S2 calls it a detour that "nothing downstream uses", while ODE-S3 and IDX-3 (same reviewer) show that the SSM chapter needs its core. Resolved above: cut the catalogue, keep the eigenvalue reading, add the driven equation.
- **The running mixture.** The score-matching reviewer shows that a symmetric mixture cannot test mode weights and asks whether the Fokker-Planck section has the same problem. It does, in the cell of FPK-13, whose reviewer treated the split as a meaningful check that an unlucky draw happened to miss.
- **Citation counts.** The score-matching reviewer counts 14 citations of his section from chapter 17 and "about ten" of the flow-matching section; my grep, which includes figure, table and exercise labels, gives 20 and 12.

No two reviews disagree about a mathematical statement, a score, or the order of the sections.

## Could not check

- TensorFlow, JAX and MXNet outputs: all three reviewers compared prose with the PyTorch store only.
- Rendered HTML and PDF pages: sources, stored outputs and rasterized figures were read, not the site.
- Claims that need a new run: the experiments proposed under SMD-1 (a trained multi-level score driving a sampler), SMD-6 (unequal mode weights), NODE-14 (a contracting field for Hutchinson), FMT-12 (signed per-step errors) and ODE-17 (backward Euler or RK4 on the gradient flow) are untested; each section file says what to verify.
- Statements quoted from papers in exercises (Kynkäänniemi et al., Song et al. 2021), the fixed-point result attributed to `Liu.2022`, and the clustering theorem of Geshkovski et al.
- The duplication inside chapter 17 was established from headings, citations and the passages the reviewers quote, with spot checks. Chapter 17 itself has not had a writing review.
