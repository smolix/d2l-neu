# Writing pass: chapter 43 introduction and Further Reading

Files: `chapter_mdl-dynamics/index.md`, `chapter_mdl-dynamics/further-reading.md`.
Reviews followed: `43-dynamics/index.md` (IDX-1 to IDX-5),
`43-dynamics/further-reading.md` (FRD-1 to FRD-6, structural findings 1 to 3),
`43-dynamics.md`. Read beforehand: the current openings and Summaries of the
six sections and their writing-pass logs, and `chapter_diffusion-models/`
(introduction, further reading, section headings, citations of this chapter).

Premise corrected as instructed: `chapter_diffusion-models/` is a full
chapter that cites this one, so this chapter is its proof store.

## Words

| File | Before | After |
|:--|--:|--:|
| `index.md` (checker, whole file) | 1087 | 1088 |
| `index.md` (introduction before the toc block, `wc -w`) | 369 | 358 |
| `further-reading.md` (checker) | 1716 | 1496 |

The checker's +1 on `index.md` comes from six added label references; the
introduction is shorter by `wc`. Checker: "OK: only prose changed" for both.
The em-dash warning on `index.md` is the ` --- ` title/author separators of
the Resources list, which I left alone; `further-reading.md` has no warning.
`tools/lint_source.py`: clean for both. The toc block, the Resources list
and all labels are untouched. No split markers (neither file is split).

## Introduction (`index.md`)

- **IDX-1.** The six ":numref: asks how" sentences are replaced by a route
  that ranks. Three results "carry the chapter": the stability limit on a
  step, the reversal of a density through the score, and least squares on
  noisy samples learning the score or a velocity. The rest is placed around
  them in one sentence. Skippable, for a reader headed for diffusion models:
  the adjoint method and optimal transport (neither is cited from
  `chapter_diffusion-models/`; `sec_mdl-ot-connection` is cited only by the
  GAN chapter).
- **Survives the split of ODEs and solvers.** The route never says "first
  two sections" or gives a count for the deterministic part. It cites
  `sec_mdl-stiffness-implicit` (stays with the solver half) and
  `subsec_mdl-ode-driven-linear` (stays with the ODE half), and no longer
  cites `sec_mdl-odes-solvers` at all, so nothing needs editing after the
  split. No positional phrase is left; every section is cited by label.
- **IDX-3, main-text uses** (labels verified by `git grep`): `sec_ssm`
  (`chapter_recurrent-modern/ssm.md`) for the driven linear equation and its
  exact step; `sec_resnet` for the residual network as Euler steps;
  `chap_diffusion`. Gradient descent is named without a label ("learning
  rates included") to save words; the opening of the ODE section cites
  `sec_gd`, and "Gradient Descent as a Solver" sits inside
  `sec_mdl-stiffness-implicit`.
- **Relation to the diffusion chapter.** New paragraph of three sentences:
  this chapter holds the proofs; small results are in both; the long
  derivations (time reversal, probability-flow ODE, the theorems behind
  score and flow matching) are only here and the diffusion chapter can be
  read without them. It replaces the old sentence "Discrete diffusion for
  text and the engineering of image generators are treated in ...".
- **IDX-2.** Prerequisites now name `sec_mdl-eigendecompositions` and
  `sec_mdl-multivariable_calculus` next to Gaussians and conditional
  expectation. There is no `chap_mdl-probability` label that I could cite
  for the probability chapter, so it is named by content only.
- **IDX-4.** Already resolved by the section pass (the first section now
  opens on the diverging run), so the chapter opening stays.
- **IDX-5.** "It calls a network a few dozen times" is now "each call to a
  network returns a direction".
- **Cut to stay within length:** the sentence on Newton's planets and
  Einstein's particles ("Differential equations were invented to predict
  motion ... The second example is the one that matters here."). The
  protected sentence ("A cloud of noisy particles ...") and the three limits
  of the Basics tools are verbatim. If the history is wanted back, the
  Fokker-Planck section opening carries its own.

Claims added or changed in the introduction: "The eigenvalues of an equation
limit the step that can follow it, learning rates included" (ODE Summary:
forward Euler stable for $|1+h\lambda|<1$, $\eta<2/\lambda_{\max}$); "a
linear equation with an input solves to the recurrence of a state space
model" (`eq_mdl-ode-zoh`, ODE Summary); a Neural ODE as "the limit of a
residual network as its steps shrink" (Neural ODE Summary); "Small results,
such as the regression target, appear in both" (`denoising.md` and
`sec_mdl-denoising-score-matching`); "that chapter can be read without them"
(its introduction: "The mathematics lives in ... to which the chapter points
instead of repeating it").

## Further Reading (`further-reading.md`)

**Opening (FRD-6).** The template paragraph, including "Each topic states the
problem it addresses and what it changes relative to the chapter", is gone.
The new opening (a) separates this file from the introduction's list (that
list teaches again what the chapter taught; this file points past it), (b)
states the rule towards the diffusion chapter (what it or its further
reading covers gets a pointer and no more), and (c) ranks: exponential
integrators, Wasserstein gradient flows and Schrödinger bridges first.
Within each group the ranked entry now comes first.

**Reduced to pointers because the diffusion chapter covers them (FRD-1)**

| Entry | Was | Now |
|:--|:--|:--|
| The Diffusion Design Space | full entry (targets, $v$, schedules, EDM, Heun) | one sentence at the head of "Learned Generators" pointing to `sec_diffusion-design-space`, keeping only the citation of Kingma et al. for the log-SNR coordinate |
| Few-Step Generators | full entry (consistency models, guidance distillation) | one clause pointing to `sec_diffusion-few-step` |
| Exponential Integrators | DEIS and DPM-Solver described with a step count | the mathematics stays (now tied to `eq_mdl-ode-variation-of-constants`, the label the ODE pass added); DEIS and DPM-Solver are one sentence pointing to `sec_diffusion-samplers` |
| Stochastic Interpolants | own entry repeating the exercise's setup (FRD-5) | three lines inside the Schrödinger bridge entry; the exercise "works through the construction" |

**Reduced because the chapter itself now teaches them:** augmented Neural
ODEs (body of the Neural ODE section, `subsec_mdl-flow-order`), the FFJORD
sentence, the restatement of the quarter-turn example (now in the body of
`sec_mdl-ot-connection`, so FRD-3's dangling reference is resolved; the
entry keeps one clause with the cost $4$ against $W_2^2=0$, which I checked
against the section text).

**Other findings**

- **FRD-2.** "Repeated reflow converges to a pairing whose straight paths do
  not cross" is now "A pairing that reflow leaves unchanged has straight
  paths that do not cross" (what the flow-matching section proves).
- **FRD-4.** The three-topic entry is now "Reversible Solvers and
  Regularized Flows" in two paragraphs, each opening with its problem; the
  third topic (augmentation) is gone, see above.
- **Structural finding 3.** "Transformers as Interacting Particles" moved to
  the end of "Continuous-Time Models".
- **Structural finding 2, Girsanov.** One sentence added to "Neural SDEs":
  for two diffusions with the same noise, Girsanov's theorem gives the
  divergence between path laws as an integral of the squared difference of
  the drifts (`Oksendal.2003`, already in the bibliography). Standard; I did
  not point to the likelihood-bound exercise because the current section
  text no longer mentions path measures.
- **Not done, deferred:** an entry on how accurate the score must be
  (sampling-error bounds in terms of the $L^2$ score error). The natural
  citation is not in `d2l.bib` and I cannot add keys in this pass. Also
  predictor-corrector sampling: the diffusion chapter's further reading has
  correctors (UniPC), so it does not belong here.

**Claims about papers softened or dropped (not verifiable here)**

- DEIS and DPM-Solver: "brings the cost of a sample down from hundreds of
  network evaluations to ten or twenty" dropped (the diffusion chapter
  measures this itself).
- "a stiff linear decay" is now "a linear term known in closed form"; I
  could not confirm that the linear part is stiff for every schedule.
- `Liu.2022`: "the fixed points are then exactly the optimal couplings" is
  now "argues that the pairings this restricted reflow leaves unchanged are
  the optimal ones"; "Each round still does not raise the transport cost"
  dropped. The reviewer had not checked the statement either.
- Geshkovski et al.: "prove that the tokens cluster as depth grows" is now
  "study how the tokens cluster as depth grows"; "mean-field" dropped.
- Reversible solver: "recovers the forward trajectory exactly" is now
  "recovers its own forward steps from the endpoint up to rounding".
- Kingma et al.: the claim that the likelihood bound depends on the schedule
  only through the signal-to-noise ratio at its two ends is no longer stated
  here (`design-space.md` derives the log-SNR form); only "the log-SNR
  coordinate of Kingma et al. orders the choices" remains.
- Dropped with their entries, not for doubt: EDM's $\alpha_t=1$,
  $\sigma_t=t$; the consistency-model definition; Meng et al.; "A student
  distilled from a teacher inherits its errors"; the score formula
  $-\mathbb E[\mathbf w\mid\mathbf x_t]/c_t$ (it is part of the exercise).
- Left as they were, consistent with what I know: symplectic energy
  conservation over exponentially long times; Barrett and Dherin; Wong and
  Zakai and the drift $\tfrac12 g\,\partial_x g$; Li, Tai and E; Mandt et
  al.; JKO; Li et al. 2020; De Bortoli et al.; sliced score matching.

**Labels added as references** (all resolve): `chap_diffusion`,
`sec_diffusion-further-reading`, `sec_diffusion-samplers`,
`sec_diffusion-design-space`, `sec_diffusion-few-step`,
`eq_mdl-ode-variation-of-constants`, `subsec_mdl-sinkhorn`. In the
introduction: `sec_mdl-stiffness-implicit`, `subsec_mdl-ode-driven-linear`,
`sec_ssm`, `sec_resnet`, `sec_mdl-eigendecompositions`,
`sec_mdl-multivariable_calculus`.

## Before and after (verbatim)

1. Before: ":numref:`sec_mdl-odes-solvers` asks when a differential equation
   has exactly one solution, and how accurately and stably a computer can
   follow it."
   After: "Three results carry the chapter. The eigenvalues of an equation
   limit the step that can follow it, learning rates included
   (:numref:`sec_mdl-stiffness-implicit`)."
2. Before: ":numref:`sec_mdl-score-matching-diffusion-flow` asks how to
   learn the score from samples and how to sample with it."
   After: "Least squares on noisy samples learns that field, or a velocity
   in its place".
3. Before: "Discrete diffusion for text and the engineering of image
   generators are treated in :numref:`chap_diffusion`."
   After: "This chapter holds the proofs for :numref:`chap_diffusion`, which
   points here instead of deriving its results."
4. Before: "We assume calculus, Gaussians and conditional expectation at the
   level of the Basics part, and we leave out the measure-theoretic
   foundations of stochastic calculus."
   After: "We assume eigendecompositions
   (:numref:`sec_mdl-eigendecompositions`), Jacobians
   (:numref:`sec_mdl-multivariable_calculus`), Gaussians and conditional
   expectation, and no measure theory."
5. Before: "Each topic states the problem it addresses and what it changes
   relative to the chapter."
   After: "What remains is mathematics, and it is not of equal weight."
6. Before: "Repeated reflow converges to a pairing whose straight paths do
   not cross."
   After: "A pairing that reflow leaves unchanged has straight paths that do
   not cross, and in more than one dimension it need not be optimal".

## Left inconsistent with other files (not mine to edit)

- `chapter_diffusion-models/further-reading.md` says that "the design space
  in continuous time, Schrödinger bridges, optimal couplings and sliced
  score matching" are in this file. The last three still are. The design
  space is now a pointer back to `sec_diffusion-design-space`, so "the
  design space in continuous time" should be removed from that sentence, or
  the two files point at each other.
- `chapter_recurrent-modern/ssm.md` and `chapter_convolutional-modern/resnet.md`
  still do not point to this chapter; the introduction now points to them
  (the ODE log lists the sentences to add there).
- The slides sections of both files: neither file has one that I changed;
  nothing to reconcile.
