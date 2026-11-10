# Further Reading (Dynamics)

Source: `chapter_mdl-dynamics/further-reading.md` (229 lines, about 1,500 words, no code, no exercises).

## Verdict

A useful and accurate reading guide, and a good instance of the form the writing guide (§12.3) describes: sixteen short topics in the order of the chapter, each opening with the problem it addresses and the one thing it changes relative to what the chapter taught, with citations that carry a clause. It is within the length budget and nothing in the chapter depends on it. I checked the technical statements I could and found none wrong. The one thing to fix is its relation to chapter 17, which was written after it: two of its topics ("The Diffusion Design Space", "Few-Step Generators") are presented as material the book had no room for, while the main text now gives each a full section.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **Every entry states a difference.** "A symplectic method ... does not try to follow the flow more closely. It preserves phase-space volume instead"; "An exponential integrator solves the linear part exactly ... and approximates only the nonlinear remainder"; "Its pairing is the plan of entropy-regularized optimal transport, computed by Sinkhorn iterations ... rather than an assignment, and the generator is an SDE." A reader learns what each method is for in two sentences and can decide whether to follow the citation.
- **Entries tie back to a named place in the chapter**: the symplectic exercise, the stiffness section, the adjoint, the OU stationary law, Hutchinson's estimator. This is what makes it a guide to this chapter and not a generic bibliography.
- **"Wasserstein Gradient Flows"** ends by explaining something about the chapter itself: "This is why transport, from the continuity equation to Benamou-Brenier, keeps reappearing in the chapter." The JKO step as "the analogue for densities of the backward Euler step" is the kind of connection the model texts make.
- **"Stochastic Gradient Descent as an SDE"** states its own limits ("The model needs small learning rates and nearly Gaussian gradient noise, and both assumptions can fail in practice").
- **"Physical Noise and the Wong-Zakai Theorem"** answers a question a careful reader of the SDE section will have (which integral is the physical one) and says when it matters ("They agree for the additive noise of diffusion models").

## Structural findings

1. **Coordination with chapter 17.** See FRD-1. The division the two notebooks need: this one covers mathematics beyond the appendix chapter (solvers, continuous-time models, noise, transport); chapter 17's own Further Reading covers diffusion practice. The group "Learned Generators" straddles the line.
2. **Missing topics a reader of this chapter would look for.**
   - *How accurate must the score be?* The chapter proves a KL bound for the reverse SDE in an exercise and then stops. The natural sequel is the line of results that bound the sampling error of a diffusion model in terms of the $L^2$ score error and the step size under weak assumptions on the data. One entry would tell the reader that the question has an answer and where.
   - *Predictor-corrector sampling.* The score-matching section mentions "a *corrector* that repairs the error of the main step" with no citation and no later treatment.
   - *Path measures and Girsanov's theorem*, the tool the likelihood exercise says the original proof uses ("The paper proves the bound with path measures"). One sentence and a reference (Øksendal is already in the chapter's resource list) would close that loop.
   None of these is required; the first is the one I would add.
3. **Order and grouping are right.** Solvers, continuous-time models, noise, learned generators follow the chapter. "Transformers as Interacting Particles" is an outlier in "Learned Generators" (it says itself that the model "does not generate data"); it would sit better as the last entry of "Continuous-Time Models".

## Passage-level findings

**FRD-1** | moderate | thought, scope
Anchor (Few-Step Generators): "the first model. Latent and discrete diffusion, the forms used for images"
Anchor (The Diffusion Design Space): "be chosen freely. :citet:`Kingma.Salimans.Poole.ea.2021` show that the"
The notebook says it collects "topics they had no room for". For the design space (targets, schedules, log-SNR, EDM) and for few-step generation (consistency models, distillation) the book did make room: `sec_diffusion-design-space` and `sec_diffusion-few-step` are full sections of chapter 17, with experiments. The only pointer to chapter 17 is for "Latent and discrete diffusion", which leaves the impression that the other two are not taught anywhere.
Fix: replace the two entries by one short paragraph: "Schedules, loss weightings and the choice of prediction target (:numref:`sec_diffusion-design-space`), generation in one or a few steps (:numref:`sec_diffusion-few-step`), and latent and discrete diffusion are taught in :numref:`chap_diffusion`." If a sentence of content is wanted, keep the Kingma et al. fact (the likelihood bound depends on the schedule only through the signal-to-noise ratio at the two ends), which is mathematics and is the reason for the log-SNR clock.

**FRD-2** | minor | correctness (confidence medium)
Anchor (Straight Couplings and Optimal Couplings): "Repeated reflow converges to a pairing whose straight paths do not cross. In"
The chapter shows that the cost is non-increasing and bounded below, hence convergent, and that a fixed point has $C = 0$ (exercise 9 of the flow-matching section). Convergence of the pairings themselves to a crossing-free one is stronger than that and stronger than what Liu et al. prove (a measure of non-straightness tends to zero along the sequence at rate $1/k$ in an averaged sense).
Fix: "Each round of reflow lowers the transport cost, and a pairing that reflow leaves unchanged has straight paths that do not cross."

**FRD-3** | minor | explanation
Anchor (same entry): "pairing $\mathbf x_1=\mathbf R\mathbf x_0$ of a standard Gaussian in"
The entry cites `sec_mdl-ot-connection` for the quarter-turn pairing and its cost of $4$. In that section the body's example is a rotating flow with kinetic energy $\pi^2/2$; the straight quarter-turn pairing is one sentence after the Benamou-Brenier proof, and the cost $4$ appears only in an exercise. A reader following the reference does not find what is cited. (The flow-matching review, FMT-10, recommends moving the example into the body, which would also repair this.) The entry also restates a result of the chapter at length; its new content is only the last two sentences, on gradient fields.
Fix: shorten to the new content: "A pairing that reflow leaves unchanged need not be optimal (the quarter-turn pairing of :numref:`sec_mdl-ot-connection`). :citet:`Liu.2022` restricts the velocity field to a gradient $\nabla\phi_t$; each round still does not raise the cost, and the fixed points are then exactly the optimal couplings."

**FRD-4** | minor | thought
Anchor (Reversible Solvers and Straighter Flows): "also cannot reverse the order of points (:numref:`subsec_mdl-flow-order`);"
This entry holds three unrelated topics under one heading joined by "A separate cost is" and "also": reversible solvers (a memory and accuracy question for the adjoint), kinetic-energy regularization (straightness), and augmented neural ODEs (expressiveness). The guide asks that each topic open with its problem; the second and third do not get an opening.
Fix: split into "Reversible Solvers" and "Regularized and Augmented Flows", or move the kinetic-energy sentence to the transport entries, where the reader has the vocabulary.

**FRD-5** | minor | explanation
Anchor (Stochastic Interpolants): "derives the one-sided case."
The exercise cited does more than the one-sided case: its parts 3 and 4 treat $c_t > 0$ and derive the score formula quoted in this entry. The entry and the exercise also repeat each other's setup almost word for word.
Fix: "Exercise ... works through the construction", and trim the entry to the claim that matters: the same marginals can be generated by an ODE or a family of SDEs, between any two endpoint laws.

**FRD-6** | minor | language
Anchor (opening): "The sections of this chapter integrate a given field, train a field through"
The first sentence lists five verbs for six sections and the second lists four topic groups; together they are a table of contents in prose. The useful sentences are the third and the last two.
Fix: "This section collects, in the order of the chapter, topics the chapter had no room for. Each states the problem it addresses and what it changes relative to the chapter. Resources for what the chapter does teach are in its introduction (:numref:`chap_mdl-dynamics`); nothing else in the book depends on this section."

## Checks made

Confirmed against what I know of the sources: bounded energy error of symplectic methods over exponentially long times and the modified-equation reading; DEIS and DPM-Solver as exponential integrators that reach ten to twenty evaluations; triangular Jacobians for coupling and autoregressive flows, the power-series log-determinant of invertible residual networks; the Itô-Stratonovich drift difference $\tfrac12 g\,\partial_x g$ and their agreement for additive noise; the SGD-as-SDE scaling with $\sqrt{\eta}$ and the learning-rate to batch-size ratio; the free energy and the JKO step (consistent with the chapter's Fokker-Planck equation for $\mathbf{f} = -\nabla V$); Hutchinson's estimator in sliced score matching; the SNR-endpoint result of Kingma et al.; EDM's $\alpha_t = 1$, $\sigma_t = t$; the consistency-model definition; Sinkhorn and iterative proportional fitting for Schrödinger bridges; the score formula $-\mathbb{E}[\mathbf{w} \mid \mathbf{x}_t]/c_t$ for interpolants; the quarter-turn cost $4$. All labels and exercise references resolve.

Not checked: the precise statement of the fixed-point result attributed to `Liu.2022`, and the clustering theorem of Geshkovski et al., beyond their general form.

## Still open from October

Not applicable: the notebook postdates the October review, and it is where that review's "cut" list for the score-matching section went (EDM, stochastic interpolants, interacting particles, JKO). They are better here.

## For the chapter review

- The notebook does its job and needs only light edits. It is the model for how the survey tails removed from sections 4 and 5 should be kept: short, each with a stated difference.
- Its one structural problem is the one that runs through the second half of the chapter: it was written as if the appendix were the book's only treatment of diffusion. Two entries should become pointers to chapter 17.
- The chapter introduction's "Resources and Further Reading" list does not repeat this notebook (it lists books, courses, papers and explainers for what the chapter teaches), as the guide requires. The list writes its entries as "Title --- Author" with a triple hyphen, which the house style avoids in prose; that is a matter for the index review.
- If the flow-matching section moves the quarter-turn pairing into its body (FMT-10), FRD-3 resolves itself.
