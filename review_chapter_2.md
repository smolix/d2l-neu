# Chapter 2 style review: Linear Regression in Neural Networks

## Scope

Reviewed every tracked Markdown source in `chapter_linear-regression`: `index.md`, `linear-regression.md`, `oo-design.md`, `synthetic-regression-data.md`, `linear-regression-scratch.md`, `linear-regression-concise.md`, `generalization.md`, and `weight-decay.md`. The review includes prose, headings, derivations, captions, code and output interpretation, exercises, and all 131 slide blocks. Line references are current.

## Executive assessment

The chapter has an excellent conceptual spine: a concrete house-price problem leads to a linear model, a loss, optimization, a known-answer synthetic experiment, implementation, generalization, and regularization. The revised normal-equation discussion is technically responsible, and the noise-model view of loss selection is one of the chapter's strongest explanations. The polynomial and ridge spectral experiments also connect abstractions to measurable quantities.

The main weakness is architectural. A long section on notebook utilities and object-oriented infrastructure appears immediately after the mathematical model, before the synthetic dataset or first training implementation. It suspends the learning problem for hundreds of lines. The framework architecture is necessary for the book, but its current placement and breadth make the chapter feel as though it changes subjects. Several experiments then describe outcomes from a particular random run as universal laws. Slide language strengthens those claims further (“exactly,” “relentlessly,” “everything shrinks”), even when the main text contains more nuance.

## Scores

| Dimension | Score | Basis |
|---|---:|---|
| Writing quality | 7.7/10 | Strong concrete prose in the model and data sections, offset by infrastructure digressions, conversational filler, and absolute slide slogans. |
| Explanation quality | 7.8/10 | The model/loss and ridge explanations are strong; dependency order and overinterpretation of demonstrations reduce recoverability. |
| Technical quality | 8.3/10 | Normal equations, loss likelihoods, and ridge are handled carefully; experimental scope, initialization, and framework-general claims need qualification. |

## Chapter architecture and logical order

The chapter introduction correctly states the complete learning procedure (`chapter_linear-regression/index.md:4-14`). The table of contents does not enact that procedure cleanly: `oo-design.md` comes between the mathematical definition and the data/implementation that would make the model run. Its utilities, background plotting, base classes, and incomplete trainer occupy a full section before the learner has trained a model.

A better dependency order is: linear regression and its loss -> synthetic data -> a compact scratch implementation -> extract the repeated loop into `Module`/`DataModule`/`Trainer` -> concise framework implementation -> generalization -> weight decay. If the library API must be introduced before scratch code, reduce `oo-design.md` here to the three interfaces and move notebook metaprogramming plus asynchronous plotting to the builders/tools material.

Inside `linear-regression.md`, the Gaussian likelihood arrives after analytic optimization, SGD, and vectorization (`:576`). Since it explains why squared error is the chosen objective, it belongs immediately after the loss definition. The “Biology” subsection (`:800`) is historically interesting but does not answer a dependency of the learning procedure; it should be a short sidebar or be removed from this chapter.

## Section- and file-level issues

| ID | Severity | Evidence | Violated style-guide rule | Diagnosis | Concrete revision direction |
|---|---|---|---|---|---|
| C2-01 | High | `chapter_linear-regression/index.md:19-25` places `oo-design` before synthetic data and scratch training; `oo-design.md:67-175` spends over 100 lines on utilities before models | Order material by dependency; maintain one recoverable chapter question. | The chapter pauses the regression argument to build notebook infrastructure. The learner sees decorators and asynchronous plotting before obtaining data or fitting the model. | Move synthetic data and a minimal training loop earlier. Introduce the three base-class responsibilities only when duplication has become visible; relocate utility implementation details. |
| C2-02 | Medium | `chapter_linear-regression/linear-regression.md:190-206` defines squared loss; probabilistic justification begins only at `:576` | Explain an equation's purpose and assumptions when it is introduced. | The loss is used for optimization long before its Gaussian-noise interpretation and alternatives are available. The reason for this objective is split across distant parts of the section. | Place the Gaussian conditional model and negative-log-likelihood derivation directly after squared loss, followed by the loss/noise table. Then proceed to optimization. |
| C2-03 | Medium | `chapter_linear-regression/linear-regression.md:800-848`: biological-neuron history after the operational network interpretation | Keep scope explicit; sections should answer a question required by the chapter. | The historical analogy interrupts the model-to-implementation transition and adds no mathematical or operational consequence. | Condense to a sidebar after the single-neuron figure, or move it to historical notes. End the main argument at the computational graph interpretation. |
| C2-04 | Medium | `chapter_linear-regression/linear-regression.md:558-570`: one timing followed by “dramatically faster” and “roughly tenfold to a thousandfold” | Code experiments need conditions, output interpretation, and evidence proportional to claims. | The qualitative vectorization conclusion is correct, but the numeric range is not established by the shown single timing and depends on size, device, synchronization, and warm-up. | State the measured environment or remove the range. Use warm-up and repeated timings, synchronize accelerators, report a median, and restrict the conclusion to the tested sizes. |
| C2-05 | Medium | `chapter_linear-regression/oo-design.md:67-126`: generic `add_to_class`, `HyperParameters`, and `ProgressBoard` precede the model interface | Present abstractions after a concrete instance and avoid exposing book-production machinery as subject matter. | The utilities explain how the notebook is authored more than how regression is learned. Their detail dominates the learner-facing abstraction. | Show one compact regression class first. Explain `Module`, `DataModule`, and `Trainer` from that instance; move decorator and plotting implementation to an appendix or tools chapter. |
| C2-06 | High | `chapter_linear-regression/oo-design.md:126`: plotting “never slows down training”; `:141-159` generalizes compilation/asynchrony across frameworks | Qualify system claims by framework, execution mode, hardware, and measurement. | Queueing reduces blocking but is not free; host transfer, queue contention, rendering, and synchronization can still affect throughput. “Compiled steps have to be pure” also differs across APIs. | Replace “never” with a measured, conditional claim. Separate JAX purity, TensorFlow graph behavior, PyTorch compilation/graph breaks, and eager execution; state what the queue actually guarantees. |
| C2-07 | Medium | `chapter_linear-regression/linear-regression-scratch.md:54-64`: “The magic number 0.01 often works well in practice” | Explain constants and assumptions; avoid “magic” as a substitute for mechanism. | The initialization scale is introduced without reference to this model's input scale or later initialization theory. “Often works” is too broad. | Say that a small nonzero scale suffices for this convex linear example and does not materially control convergence here; point forward to variance-preserving initialization for deep networks. |
| C2-08 | Low | `chapter_linear-regression/linear-regression-scratch.md:357-365`; `linear-regression-concise.md:301-317`: “Now that we have all the pieces...” and reader directives | Transitions should carry dependency, not stage the conversation. | These paragraphs announce readiness and predict what the reader should understand instead of stating what changes between scratch and framework code. | Replace with the concrete invariant: the same batch-level computation is now delegated to `Trainer.fit`, and list which responsibilities remain visible. |
| C2-09 | Medium | `chapter_linear-regression/synthetic-regression-data.md:24-27`, `:381-384`: synthetic data is “the first test for any new learning method” | Avoid universal prescriptions; state the diagnostic's limits. | Some algorithms cannot be meaningfully assessed by this generator, and agreement with synthetic truth does not validate performance under misspecification. | Call it an early implementation check for methods compatible with the generator. Explicitly state what passing the check does and does not establish. |
| C2-10 | Medium | `chapter_linear-regression/synthetic-regression-data.md:367-376`: JAX drops the partial batch and says recompilation “can cost minutes per epoch” | Explain operational tradeoffs with scoped evidence. | Dropping data is a material semantic choice, while the timing claim is unmeasured here. Readers may infer that fixed batch shapes always justify discarding examples. | State the exact examples dropped, reshuffling consequence, and alternatives (padding/masking or shape-polymorphic handling). Remove or document the “minutes” claim. |
| C2-11 | High | `chapter_linear-regression/generalization.md:408-450`: one constructed sweep “traces out exactly” the U-curve; variance “grows relentlessly”; slides repeat at `:823-848` | Do not turn an illustrative experiment into a universal law; align claim strength with evidence. | The behavior depends on the sampled inputs, noise draws, degree range, and numerical conditioning. Repeated draws estimate the decomposition, but no uncertainty is shown. | Call the plot an instance, seed it, report variability across repeated datasets or confidence bands, and distinguish the finite-sample observation from the expected bias-variance theorem. |
| C2-12 | Medium | `chapter_linear-regression/weight-decay.md:743-759`: summary omits the spectral and MAP explanations; the exact MAP scaling is deferred to an exercise | Conclusions should reconstruct the argument; central assumptions should not be left only to exercises. | The section's most explanatory material—direction-dependent shrinkage and the relation between `lambda`, noise variance, sample size, and prior scale—disappears from the summary. | Put the MAP scaling in the main derivation, distinguish summed and averaged objectives there, and rewrite the summary around constrained geometry -> update -> spectral effect -> validation choice. |
| C2-13 | Medium | `chapter_linear-regression/weight-decay.md:826-835`: “everything shrinks”; “lasso does feature selection and ridge does not” | Slides must preserve qualifications from the main text. | The two-dimensional geometry is suggestive, not a universal coordinate-wise statement; correlated designs and special optima complicate the slogan. | Say that ridge generally applies continuous shrinkage without inducing sparsity, whereas lasso's corners can produce exact zeros. Tie the claim to the displayed geometry. |
| C2-14 | Low | `chapter_linear-regression/oo-design.md:817-820`: “is the point of the next slide” | Avoid self-referential presentation language; titles/transitions should state dependencies. | The note describes slide choreography rather than the conceptual reason asynchronous reporting is discussed. | Replace it with the dependency: fetching a device value may synchronize execution, which motivates separating measurement from rendering. |

## Mathematics and notation

- Preserve the normal-equation caveat at `linear-regression.md:328-342`: it separates an analytic formula from a sound numerical implementation and states the rank-deficient case.
- Move the Gaussian likelihood adjacent to squared loss, with the conditional model, independence assumption, variance convention, and negative-log-likelihood all visible before the equivalence claim.
- The ridge section handles the summed/averaged objective conversion explicitly at `weight-decay.md:500-502`; use the same convention in the MAP paragraph rather than leaving the exact relation to exercise 6.
- The bias-variance experiment should distinguish the theoretical expectation from a Monte Carlo estimate based on 200 noise draws. “Expected test error” at `generalization.md:449-450` needs its conditioning and averaging convention stated.
- Avoid changing notation between conceptual and implementation sections: `lambda`, `lambd`, and `tilde{lambda}` are justifiable, but each conversion should be restated where code is compared with equations.

## Figures, captions, and slides

The house-price setup, loss/noise figure, ridge geometry, and bias-variance plots are useful because they expose relations rather than decorate the prose. The single-neuron caption at `linear-regression.md:782` is too thin to stand alone; it should say that every feature contributes through one weighted edge and the output is their affine sum.

All 131 slide blocks were reviewed. The decks cover the source dependencies, but some slides turn demonstrations into slogans: “Deep nets just stack many of them” (`linear-regression-scratch.md:822`), “figures ... are exactly what this produces” (`:949`), “variance grows relentlessly” (`generalization.md:824-848`), and “everything shrinks” (`weight-decay.md:834`). Replace these with scoped declarative claims. The `oo-design` deck also contains self-referential sequencing and spends many slides on authoring utilities before the learning loop.

## Code and experiment pedagogy

- The synthetic generator is a strong known-answer test. Pair parameter recovery with an explicit tolerance and distinguish statistical error from implementation error.
- The scratch training explanation at `linear-regression-scratch.md:313-323` clearly identifies gradient clearing and no-tracking as invariants; preserve it.
- Timing demonstrations need warm-up, synchronization, repetition, and environmental scope. A single elapsed-time print should support only a local observation.
- For the polynomial experiment, seed both inputs and noise, show several datasets or an uncertainty band, and state which plot is empirical versus theoretical.
- JAX's fixed-shape batching decision should be presented as one engineering option. The code explanation should mention that dropping a remainder changes epoch coverage.
- The concise implementation should explicitly compare outputs against the scratch version, not only celebrate fewer lines of code.

## Recurring artifacts

- Readiness transitions: “Now that we have all the pieces,” “we are ready.”
- Reader simulation: “You might have noticed,” “the astute reader might wonder.”
- Authoring-process exposition: notebook cell length, slide sequence, and library scaffolding displace the model argument.
- Universal experimental conclusions: “exactly,” “relentlessly,” “any new method,” “never slows.”
- Slide compression that removes statistical or system conditions.

## Positive patterns to preserve

- The house-price opening in `linear-regression.md:9-31` grounds every term in one concrete record.
- The loss/noise-model table makes objective selection principled and points naturally to classification.
- The normal-equation discussion separates mathematical identity from numerical advice.
- `synthetic-regression-data.md` clearly identifies three possible sources of failure and explains why known truth is useful.
- The scratch optimizer paragraph states two operational invariants and their failure modes.
- The spectral view of ridge explains *which* directions shrink, not merely that a norm becomes smaller.

## Prioritized revision plan

1. Reorder or sharply reduce `oo-design.md` so the learner reaches data and a fitted model before notebook infrastructure.
2. Move the Gaussian likelihood and loss-selection explanation next to the squared-loss definition; demote the biology material.
3. Rework the generalization and performance demonstrations so observations, conditions, and theoretical claims are explicitly separated.
4. Put the exact MAP correspondence and objective scaling in the main weight-decay argument; rebuild its summary around the section's strongest explanation.
5. Audit all 131 slides for absolute wording and restore the conditions present in the main prose.
6. Replace conversational/readiness transitions and strengthen thin captions during the final line edit.

**Issue count: 14 total (3 high, 9 medium, 2 low).**
