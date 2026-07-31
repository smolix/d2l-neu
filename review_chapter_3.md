# Chapter 3 style review: Linear Classification in Neural Networks

## Scope

Reviewed every tracked Markdown source in `chapter_linear-classification`: `index.md`, `softmax-regression.md`, `image-classification-dataset.md`, `classification.md`, `softmax-regression-scratch.md`, `softmax-regression-concise.md`, `generalization-classification.md`, and `environment-and-distribution-shift.md`. The review covers prose, headings, equations, captions, code and experiments, exercises, and all 151 slide blocks. Evidence points to current line numbers.

## Executive assessment

The chapter's main conceptual transfer from regression is clear: outputs now parameterize a categorical distribution and the likelihood yields cross-entropy. The best passages state that transfer explicitly, derive stable fused loss computation, and use a confusion matrix to show why one scalar accuracy is inadequate. The covariate-shift code is also unusually valuable because it builds a problem with a known density ratio and compares the learned discriminator against it.

The chapter is much less controlled at its boundaries. `softmax-regression.md` begins with a lengthy return to regression variants and ends with an advanced compression survey unrelated to the next implementation step. `generalization-classification.md` embeds a long second-person story where a shorter statistical argument would be clearer. `environment-and-distribution-shift.md` expands from distribution shift into batch/online learning, bandits, control, reinforcement learning, deployment ethics, and feedback loops. These topics matter, but together they turn the final section into a second chapter without a single recoverable question. Several repeated claims that 82–83% is *the* linear ceiling on Fashion-MNIST are not established by the experiment and are amplified across slides.

## Scores

| Dimension | Score | Basis |
|---|---:|---|
| Writing quality | 7.1/10 | Strong local explanations coexist with polemical summaries, second-person narrative, filler, and an overextended final section. |
| Explanation quality | 7.6/10 | The softmax/cross-entropy chain and diagnostic examples work well; scope and experimental interpretation often obscure the main dependency. |
| Technical quality | 8.0/10 | Core derivations are solid, but layout claims, metric aggregation, empirical ceilings, and distribution-shift qualifications need correction or scoping. |

## Chapter architecture and logical order

The table of contents follows a reasonable high-level path: theory -> data -> reusable classifier interface -> scratch and concise implementations -> generalization -> shift. Two sections break that path internally.

First, `softmax-regression.md` spends lines 8-27 on log-price, count, and survival-regression distinctions before presenting classification, then adds a research-level survey of structured compression at lines 485-509 after its summary has begun. The first passage should be reduced to the one contrast needed here; the latter belongs in computational performance or further reading.

Second, `environment-and-distribution-shift.md` has at least three independent arcs: shift taxonomy/correction, learning-problem taxonomy, and responsible deployment. The text itself concedes that its correction material is skippable (`:298-300`), a sign that the dependency structure is not resolved. Keep this chapter's close on distribution shift: what changes, what remains invariant, what can be estimated, and when reweighting fails. Move online/bandit/control/RL to their respective chapters and give feedback loops/fairness a dedicated deployment section with enough definitions and evidence.

## Section- and file-level issues

| ID | Severity | Evidence | Violated style-guide rule | Diagnosis | Concrete revision direction |
|---|---|---|---|---|---|
| C3-01 | Medium | `chapter_linear-classification/softmax-regression.md:8-27` discusses log-price, counts, and survival modeling before classification | Start from the chapter's concrete problem and introduce only prerequisites required by the next claim. | The regression qualification is correct but too elaborate for a transition. It delays the categorical object and introduces a specialist field that is not used. | Retain one sentence: different target distributions call for different likelihoods; here the target is categorical. Move immediately to the image example. |
| C3-02 | Low | `chapter_linear-classification/softmax-regression.md:31-36`: example list includes “Which section of the book are you going to read next?” | Examples should clarify the formal distinction, not call attention to the text as artifact. | The meta example weakens the otherwise concrete classification set and does not help distinguish single-label, ordinal, or multilabel tasks. | Keep two contrasting real tasks and explicitly identify their label spaces. Remove the book-navigation example. |
| C3-03 | High | `chapter_linear-classification/softmax-regression.md:464-509`: summary declares completion, then launches Fourier/structured/quaternion compression | Conclusions should reconstruct the argument; preserve dependency and scope. | An advanced cost-reduction survey begins after the conceptual argument has ended. It neither prepares Fashion-MNIST nor follows from the summary. | End with logits -> softmax -> likelihood -> cross-entropy -> gradient. Move compression to computational performance/further reading. |
| C3-04 | High | `chapter_linear-classification/image-classification-dataset.md:124` correctly distinguishes layouts, but summary at `:293` states all images have `(batch, channels, height, width)` | Maintain technical and notational consistency across framework variants. | The summary contradicts TensorFlow/JAX channel-last behavior explained earlier and reinforced by exercise 1. | State both layouts in the summary or use a layout-neutral notation with named batch/spatial/channel axes. |
| C3-05 | Medium | `chapter_linear-classification/image-classification-dataset.md:223-229`: one pass establishes that loading “is not the bottleneck” and future CNN compute is slower | Performance claims require hardware/software conditions, synchronization, repeated measurements, and bounded conclusions. | A single local loop cannot establish I/O behavior for different disks, worker counts, devices, augmentation pipelines, or later models. | Call this a loader smoke test. Report environment and repeated throughput; say only whether loading was the bottleneck in this run. |
| C3-06 | Low | `chapter_linear-classification/image-classification-dataset.md:118-124`: native 28x28 images are resized to 32x32 without explaining why | Explain transformations by purpose and consequence. | Resizing changes the data and cost but appears as an arbitrary constructor choice. | State that 32x32 prepares dimensions used by later convolution examples (if that is the reason), and note interpolation plus the 31% increase in pixels. |
| C3-07 | Medium | `chapter_linear-classification/classification.md:43-56`: the validation-averaging paragraph is duplicated verbatim for JAX | State shared concepts once; framework tabs should contain only genuine differences. | The duplicate paragraph increases length while hiding the JAX-specific state/reporting distinction. It also normalizes an avoidable biased aggregation when the last batch differs. | Explain example-weighted metric aggregation once outside the tabs and implement it correctly. Keep only NNX state/reporting behavior in the JAX tab. |
| C3-08 | High | `chapter_linear-classification/softmax-regression-scratch.md:623-636`; repeated at `:923-938` and `:1024-1026`: 82–83% is “the ceiling” and “not a tuning artifact” | Do not infer a universal performance limit from one implementation/run; match claims to evidence. | A confusion matrix explains error structure but does not prove optimal linear separability or exclude preprocessing, optimization, regularization, or evaluation choices. | Report the observed range for this setup. If a linear-model ceiling is important, define the hypothesis class and provide a systematic hyperparameter/optimization study with uncertainty. |
| C3-09 | High | `chapter_linear-classification/softmax-regression-concise.md:294-301`: “blessing and a curse,” engineers without statistics, “protective padding,” “little muscle memory” | Use neutral, precise prose; do not speculate about readers or user competence. | The summary becomes a polemic about framework users instead of reconstructing fused loss and the scratch/concise tradeoff. | Replace it with a technical comparison: which operations the fused API combines, what instability it avoids, and when custom components still require understanding logits and gradients. |
| C3-10 | High | `chapter_linear-classification/generalization-classification.md:301-358`: long second-person 3 a.m. story about repeated test-set use | Avoid false intimacy and dramatized narrative; state the statistical dependency directly. | The story consumes many lines, predicts emotions, and delays multiple-testing/adaptive-overfitting mechanics. | Use a compact sequence: evaluate one fixed model -> inspect test result -> choose a new model adaptively -> test statistic is no longer independent. Then give the union-bound consequence. |
| C3-11 | Medium | `chapter_linear-classification/generalization-classification.md:421-444`: “test sets are all that we really have”; “you might now be sufficiently primed” | Avoid universal slogans and reader-state narration; define the question precisely. | The passage overstates the role of test sets and frames learning theory as an emotional reveal. | State the distinction between post hoc evaluation and a priori uniform guarantees, then introduce model-class complexity as the next mathematical object. |
| C3-12 | High | `chapter_linear-classification/environment-and-distribution-shift.md:289-300`, `:634-716` | One section should answer one question; order by dependency and state scope. | Shift correction, batch/online/bandit/control/RL taxonomy, and fairness/deployment are separate subjects. A “skip this” instruction admits that the internal order is optional rather than causal. | Split the material. Keep only shift definitions, diagnostics, correction assumptions, and monitoring here. Move the taxonomies and expand responsible deployment elsewhere. |
| C3-13 | Medium | `chapter_linear-classification/environment-and-distribution-shift.md:384-403`, algorithm at `:421-424` | State sampling assumptions beside the estimator; connect derivation to implementation. | The density-ratio odds identity assumes balanced domain priors; unequal sample sizes change the odds by a constant. The text notes this, but the algorithm does not say that samples must be balanced or weights normalized. | Make domain-prior correction an explicit algorithm step: balance the discriminator dataset or multiply odds by the source/target prior ratio, then normalize weights used in optimization. |
| C3-14 | Medium | `chapter_linear-classification/environment-and-distribution-shift.md:445-512`: one constructed run yields “no better than a coin flip,” “above 90%,” and clipping “helps” | Interpret outputs as evidence from the stated experiment, not general properties of importance weighting. | The experiment is seeded and useful, but exact performance and the benefit of clipping depend on this draw, model misspecification, overlap, and clipping level. | Label the numbers as this run, repeat over seeds with dispersion, and state the bias-variance tradeoff rather than implying clipping improves target accuracy generally. |
| C3-15 | Low | `chapter_linear-classification/environment-and-distribution-shift.md:1077`: “there is no clever reweighting, the old answers are simply wrong.” | Slides require precise syntax and bounded claims. | The comma splice combines two claims, and “concept shift” can sometimes be adapted with new labels or structured assumptions even though covariate reweighting alone is insufficient. | Write: “Covariate reweighting cannot repair a changed conditional `P(y|x)`; adaptation requires information about the new labeling relation.” |

## Mathematics and notation

- The transfer from Gaussian likelihood/squared loss to categorical likelihood/cross-entropy is the chapter's mathematical backbone. Make the shared pattern explicit once and remove surrounding digressions.
- At `softmax-regression.md:791`, the slide says “prediction minus truth” is shared by *every* exponential-family model. Qualify this as the gradient with respect to canonical natural parameters being expected sufficient statistics minus observed sufficient statistics.
- The density-ratio derivation should keep `p`/target and `q`/source naming stable through prose, figure, algorithm, and code. Include the domain-prior constant in the operational recipe, not only a parenthesis.
- Precision is undefined when no positive predictions are made (`classification.md:221`), which the prose correctly states. Exercises and metric code should preserve explicit zero-denominator behavior rather than silently choose a convention.
- The generalization bounds need a visible distinction among a fixed classifier, a finite predeclared class, and adaptively chosen models. That distinction should organize the section rather than emerge after the story.

## Figures, captions, and slides

The confusion matrix, softmax geometry, temperature diagram, and density-ratio figure are explanatory and should remain. The density-ratio caption at `environment-and-distribution-shift.md:435` is self-contained, though “explodes” could be replaced by “diverges as `q(x)` approaches zero with `p(x)>0`.” The dataset visualizations should state that the displayed palette is false color and the underlying data are grayscale, as the summary already does.

All 151 slide blocks were reviewed. This is a very dense deck for seven content sections, and repeated framework/result slides inflate it further. The most important slide corrections are the universal 82–83% ceiling, “every exponential-family model,” and distribution-shift slogans. Descriptive titles such as “Only differences of logits matter” work well; titles should state the relation without adding claims stronger than the derivation.

## Code and experiment pedagogy

- The naive softmax overflow demonstration and the fused-logit loss provide a strong failure -> remedy sequence. The concise summary should center it.
- The Fashion-MNIST loader timing needs benchmarking discipline and should not predict later CNN bottlenecks from one pass.
- Aggregate validation metrics by example count, not by an unweighted mean of batch means; the final partial batch is a simple opportunity to teach the distinction.
- The confusion matrix supplies richer evidence than accuracy, but it does not prove an optimum over all linear models.
- The covariate-shift experiment is exemplary in using a known true log-density ratio. Add repetitions and separate recovery of the ratio from downstream classification accuracy.
- When correction depends on access to unlabeled target data, state that deployment assumption before the algorithm and discuss how a finite, possibly drifting target sample affects the estimate.

## Recurring artifacts

- Long boundary digressions before or after the core argument.
- Reader simulation and dramatization: “get our feet wet,” “you wake up at 3am,” “sufficiently primed.”
- Polemical abstractions: “blessing and curse,” “protective padding,” assumptions about engineers.
- Experimental results promoted to limits: “the ceiling,” “not a tuning artifact,” “exactly.”
- Topic accumulation in the final section instead of a bounded scope.

## Positive patterns to preserve

- The chapter introduction states precisely what changes from regression and what remains the same.
- The softmax invariance and fused cross-entropy explanations connect formal identities to numerical implementation.
- The disease-screening example makes accuracy failure concrete before introducing precision and recall.
- The confusion matrix is reused as both a diagnostic and an object in label-shift correction, creating a useful dependency across sections.
- The covariate-shift example has a known density ratio, visible support shift, and an explicit comparison among unweighted, weighted, and clipped training.
- Several slide titles are strong declarative claims, especially “Only differences of logits matter” and “Maximum likelihood becomes a sum of losses.”

## Prioritized revision plan

1. Split and rescope `environment-and-distribution-shift.md`; keep this chapter's final arc on shift, correction assumptions, failure modes, and monitoring.
2. Remove the compression survey from `softmax-regression.md` and rebuild its opening around one categorical example.
3. Replace all “linear ceiling” claims with scoped observations or supply the missing controlled evidence.
4. Rewrite the concise implementation summary and the repeated-test-set passage in neutral, compact, technical prose.
5. Correct the channel-layout summary and metric aggregation; make the domain-prior correction operational in the covariate-shift algorithm.
6. Consolidate the 151-slide deck, removing duplicated framework/result slides and auditing every absolute claim against the main derivation.

**Issue count: 15 total (6 high, 6 medium, 3 low).**
