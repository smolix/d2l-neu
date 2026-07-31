# Chapter 7 Style Review: Modern Convnets

## Scope and files reviewed

Diagnosis only. I reviewed every tracked Markdown file in `chapter_convolutional-modern`: `index.md`, `alexnet.md`, `blocks.md`, `batch-norm.md`, `resnet.md`, `training-recipes.md`, `convnext.md`, `efficient-convnets.md`, and `cnn-design.md`. The review covers prose and heading hierarchy, mathematical claims, architecture diagrams and captions, code/experiment setup and interpretation, summaries, and all slide blocks.

## Executive assessment

The chapter’s best new sections (`training-recipes.md`, `convnext.md`, and much of `efficient-convnets.md`) use controlled comparisons to separate architectural changes from training procedure and deployment cost. Older inherited material is conspicuously less disciplined: several sections rely on promotional historical narration, vague “truth” functions, unsupported causal explanations, and dense paragraphs that mix motivation, theorem-like claims, and chronology. The slide deck for batch normalization is materially more confident than the chapter’s own discussion. The chapter also lacks a consistent closing architecture: several files use “Discussion,” some use “Summary and Discussion,” and `cnn-design.md` has no synthesis before exercises.

## Scores (0–10)

| Dimension | Score | Rationale |
|---|---:|---|
| Writing quality | 5 | Strong recent prose coexists with long inherited sentences, promotional headings, and visible authorial staging. |
| Explanation quality | 6 | Ablation-driven sections are effective; older sections often introduce history or code before the technical question is made precise. |
| Technical quality | 6 | Core architectures are represented correctly, but causal explanations and empirical generalizations are often stronger than their evidence. |

## Architecture and logical order

Chronology is defensible because designs respond to predecessors, but the intellectual dependency should remain explicit: AlexNet establishes learned representations; VGG/NiN/GoogLeNet introduce reusable blocks; normalization and residual connections address optimization; recipes show that training confounds architecture comparisons; ConvNeXt, efficient networks, and RegNet revisit design under controlled budgets. `batch-norm.md` belongs next to the optimization problem it answers but currently spends too long on input preprocessing. `cnn-design.md` should conclude the chapter only if it reconstructs what systematic design adds beyond the preceding one-off architectures; the missing summary prevents that payoff.

## Section/file issue table

| ID | Severity | Evidence | Excerpt / description | Violated rule | Diagnosis | Concrete revision |
|---|---|---|---|---|---|---|
| C7-01 | M | `alexnet.md:6` | Title: “The ImageNet Moment: AlexNet” | §14.3: avoid literary/promotional titles | “Moment” advertises historical drama rather than the section’s intellectual work. | Retitle “AlexNet and Learned Image Representations” or “AlexNet: Data, Compute, and ReLU at ImageNet Scale.” |
| C7-02 | M | `alexnet.md:10–40` | Three historical paragraphs precede the technical architecture | §§5.1, 5.4: concrete problem before chronology | The opening delays the explanatory tension: hand-engineered features versus jointly learned representation, and the data/compute/training conditions needed for the latter. | Compress history into the concrete classical pipeline, identify the three missing conditions, then introduce AlexNet; move remaining chronology to a short context box. |
| C7-03 | H | `alexnet.md:408–410` | “hardly any overfitting ... This is due to ... dropout” | §§13.3–13.4, 16.1: match causal claim to evidence | A single train/validation curve does not isolate dropout, and nearly equal losses may reflect preprocessing, task difficulty, or implementation. “Accuracy (dropout)” is also too causal. | Describe only the observed gap; cite an ablation or rephrase dropout as one plausible contributor. Report accuracy/loss and run conditions, not “hardly any” globally. |
| C7-04 | H | `batch-norm.md:48–76` | Input preprocessing, unit norms, bounds, “plays nicely,” and internal normalization are combined | §§6.2, 7, 9.6: one inferential step; exact versus intuition | The analogy between fixed input standardization and batch normalization is presented through several loosely connected claims. Bounded-norm generalization results do not by themselves motivate minibatch normalization. | Separate (1) the empirical optimization problem in deep activations, (2) BN’s exact transform, and (3) input-standardization analogy explicitly labeled as analogy. Remove the tangential survey sentence. |
| C7-05 | H | prose `batch-norm.md:983–1001` versus slides `batch-norm.md:1097–1115` | Prose calls explanations speculative; slide says “single-biggest stability win” and “gradients stay well-conditioned” under “Why it works” | §§9.6, 13.4, 19, 20.10: preserve evidential strength across formats | The deck converts an unresolved mechanism into a confident causal explanation and adds an unmeasured superlative. | Title the slide “Observed effects and normalization variants”; distinguish empirical effects from proposed mechanisms and remove “single-biggest.” |
| C7-06 | M | `blocks.md:21–27`, `53–59` | Long backend memory/autotuning comments precede the VGG question | §13.1: explain computational purpose first; do not let implementation dominate concept | Backend-specific memory tuning is useful for reproducibility but interrupts the architecture opening and contains measured claims with no local environment record. | Add a short prose note explaining why memory settings are needed; move detailed values to an implementation note and state hardware/software conditions. |
| C7-07 | M | `blocks.md:70–79` | “key idea,” “quest,” “gold standard,” “Back to VGG” in very long paragraphs | §§8.2, 15.2, 17.1–17.4 | The argument from receptive field and parameter count is valuable but is buried in historical promotion and self-conscious transition. | Split into: pooling depth limit; repeated-convolution block; exact parameter/receptive-field comparison; empirical result and limitation. Use descriptive verbs. |
| C7-08 | H | `resnet.md:47–74` | Undefined “truth” (f^*), “best bet,” “better,” and claim that more data generally improves (f^*_{\mathcal F}) | §§9.2, 9.10, 16.1: define objects and claim strength | (f^*_{\mathcal F}) is defined using training loss, while prose treats distance to an unspecified truth and generalization as the same quantity. Nested classes guarantee non-increasing empirical optimum, not better population performance or optimization. | Define the objective and comparison metric. State only the containment result for optimal empirical loss, then separately discuss optimization/generalization limitations. Redraw/re-caption the figure accordingly. |
| C7-09 | H | `resnet.md:540` | “powerful and flexible”; loss gap; “more training data would” close it and improve accuracy | §§13.3–13.4, 16.2 | A single Fashion-MNIST curve does not show that data scarcity causes the gap or that more data closes it. | Report the observed training/validation gap; identify data augmentation, regularization, and capacity as hypotheses; test one or avoid the causal prescription. |
| C7-10 | H | `cnn-design.md:251–258` | “CDF ... majorizes ... superior,” “no effect at all,” “equally harmless,” “verify both” | §§13.4, 16.1–16.3: local experiment does not prove universal superiority | Finite sampled error distributions under one protocol cannot establish no effect, harmlessness, or general superiority. “Majorizes” is not defined and appears to mean stochastic dominance. | Define empirical CDF and first-order stochastic dominance; say the sampled distributions are indistinguishable/shifted under the paper’s budget, sample size, and training protocol. Include uncertainty. |
| C7-11 | M | `cnn-design.md:427` | Exercises follow immediately after comparison with transformers; no summary | §§5.6, 6.3: summary reconstructs argument | The final section never states what the design-space experiment established, under which budget, or how RegNet changes the chapter’s view of architecture. | Add a summary that connects sampled design distributions, tied stage parameters, quantized linear widths, and the limits of the search protocol. |

## Math and notation

- Resolve the mismatch in `resnet.md` between a training-loss argmin and geometric “distance to truth.” The current diagram and formula do not share a defined metric.
- In `batch-norm.md`, attach axes to every mean/variance definition and keep training statistics, running estimates, and per-example normalization distinct.
- In `cnn-design.md`, define `e_i`, sampling distribution `\mathcal Z`, sample size `n`, and the stochastic-order relation before claiming one CDF dominates another.
- Parameter/FLOP comparisons in `efficient-convnets.md` are useful; state whether multiply-adds count as one or two operations and preserve that convention across tables.
- Major ConvNeXt and re-parameterization equations generally follow the equation protocol well and should serve as local models for older sections.

## Figures, captions, and slides

Architecture captions in the recently added material are unusually strong: they identify branches, shapes, and the relevant comparison. `resnet.md:74` is weaker because “closer to truth” has no defined distance and the caption claims nested classes avoid an issue that only concerns representational containment. `cnn-design.md:255` puts an entire four-panel interpretation into the caption and overstates null results; move inference and uncertainty to prose. Slide captions such as “NiN vs. VGG: ... radically different head” use promotional modifiers where a literal difference suffices. Most urgently, repair the batch-normalization slide contradiction (C7-05).

## Code and experiment pedagogy

`training-recipes.md` and `convnext.md` clearly state what is held fixed and what accumulates; preserve this controlled-comparison pattern. `alexnet.md` and `resnet.md` interpret single curves causally without an intervention (C7-03, C7-09). `blocks.md` exposes measured backend memory numbers before hardware/software conditions (C7-06). For each training comparison, state seed count, dataset subset, metric, budget, and whether the result verifies an identity, illustrates a mechanism, or estimates general performance.

## Recurring artifacts

- Promotional history: “watershed,” “moment,” “quest,” “gold standard,” “powerful.”
- Vague agent/metric language: “truth,” “better,” “best bet,” “harmless.”
- Overloaded paragraphs with several semicolons and chronology mixed into derivation.
- Section-closing inconsistency (“Discussion,” “Summary,” “Summary and Discussion,” or none).
- Slides that turn qualified prose into slogans or causal claims.

## What already works

- `index.md` identifies the architecture-versus-recipe confound that organizes the chapter.
- `training-recipes.md` begins with a concrete 76.1% to 80.4% comparison and states what remains unchanged.
- `convnext.md` uses a cumulative ablation table with held-cost context.
- `efficient-convnets.md` connects factorization, scaling, and exact re-parameterization to deployment constraints.
- Many new architecture captions are self-contained and operational.

## Prioritized revision plan

1. Correct C7-03, C7-05, C7-08, C7-09, and C7-10; these are evidential or conceptual, not cosmetic.
2. Rebuild `batch-norm.md` and `resnet.md` around exact objects and explicit limitations.
3. Harmonize every slide with the corresponding prose’s claim strength.
4. Compress historical narration in `alexnet.md` and `blocks.md`; retain only history that explains a design dependency.
5. Add the missing `cnn-design.md` synthesis and standardize section endings.
6. Audit experiments for seeds, conditions, uncertainty, and causal language.
