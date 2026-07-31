# Review of Chapter 25: Calculus and Automatic Differentiation

## Scope

Reviewed `chapter_mdl-calculus/index.md`, `mdl-single-variable-calculus.md`, `mdl-multivariable-calculus.md`, `mdl-matrix-calculus-autodiff.md`, and `mdl-integral-calculus.md`, including all equations, proofs, examples, code, figures, summaries, exercises, and slides.

## Executive assessment

The chapter has a strong teaching idea: derivatives are local linear maps, and backpropagation is structured composition of those maps. It frequently gives the reader a picture, a formula, and a computation. Its main weaknesses are occasional technical shortcuts in foundational definitions, a few overconfident slogans, long proof paragraphs, and a mismatch between the chapter order described in the index and the conceptual dependencies of the sections. The slide decks are generally coherent but still contain topic headings and dense proof slides.

Scores (0–10): **writing quality 7.8**, **explanation/pedagogy 8.3**, **technical/logical quality 7.8**.

## Architecture and logical order

Single-variable → multivariable → matrix calculus/autodiff → integration is coherent for optimization, but the index says the “final section develops integration” while the prose also points integration forward to probability and dynamics. Make that change of purpose explicit. Within single-variable calculus, gradient descent appears before a clean account of the derivative’s assumptions; within multivariable calculus, backprop is introduced and then substantially reintroduced in matrix calculus. Define the first treatment as scalar-graph intuition and the second as computational cost/JVP–VJP generalization.

## Detailed issues

| ID | Severity | Location | Problem and violated guide rule | Concrete revision direction |
|---|---|---|---|---|
| C25-01 | Moderate | `chapter_mdl-calculus/index.md:4-10` | The chapter map lists topics but does not tell the reader why integration follows autodiff or which sections are prerequisites versus optional reference. | Add a two-track map: differentiation for training, integration for probability/continuous models. |
| C25-02 | Critical | `mdl-single-variable-calculus.md:30-35` | The motivating function is `sin(x^x)` “on [0,3]”; the usual real expression `x^x` is undefined at `x=0` unless a continuous extension is declared. A foundational figure should not begin with a hidden domain exception. | Use a smooth function defined on the entire plotted interval, or explicitly define the extension and avoid distracting domain issues. |
| C25-03 | Moderate | `mdl-single-variable-calculus.md:38` | “Locally, a smooth function is a line” is memorable but literally false; the function is locally approximated by an affine map, with an error condition. The following conversational questions add flair without precision. | Say “at a differentiable point, the first-order error is negligible relative to the step,” then connect that definition to the picture. |
| C25-04 | Moderate | `mdl-single-variable-calculus.md:48-90` | Imports interrupt the secant-to-tangent explanation before the derivative is formally defined. This makes setup, not the concept, control the flow. | Complete the definition and first example before moving setup into a collapsed cell. |
| C25-05 | Moderate | `mdl-single-variable-calculus.md:439-490` | “The best quadratic is a better optimizer” and “simply jump there” overstate Newton’s local model. The quadratic can be indefinite, curvature can be near zero, and a full Newton step can increase the objective. | State the conditions under which the model has a local minimum and preview damping/trust regions; distinguish solving the local model from improving the true function. |
| C25-06 | Moderate | `mdl-single-variable-calculus.md:520-546` | The Taylor-remainder proof is a single very long paragraph with several cancellations and a hidden auxiliary-function strategy. It is rigorous but difficult to learn from. | Preview the proof idea, break it into numbered steps, and isolate the telescoping derivative in a display. |
| C25-07 | Moderate | `mdl-single-variable-calculus.md:768-775` and slide `:1173` | The summary/slide claim that the small-change identity “generates everything” is the same totalizing artifact rejected in the style guide. It erases assumptions and the separate role of limits, curvature, and nonsmooth analysis. | Replace with a bounded statement: it organizes the derivative rules and local optimization arguments developed here. |
| C25-08 | Major | `mdl-multivariable-calculus.md:29-75` | The coordinate-at-a-time derivation initially discards cross terms before differentiability is established. The later caveat is correct, but readers can retain the false implication that existing partial derivatives automatically give a linear approximation. | State the differentiability assumption first, derive the gradient from it, then use coordinate perturbations to identify components; keep the counterexample adjacent. |
| C25-09 | Moderate | `mdl-multivariable-calculus.md:207-327` | Several geometric consequences—steepest direction, level sets, tangent planes, and linearization—repeat one another at length. The central norm dependence of “steepest” can be missed. | Consolidate around the directional derivative and explicitly say steepest descent depends on the chosen norm/inner product. |
| C25-10 | Moderate | `mdl-multivariable-calculus.md:402-687` | The chain rule and backprop section mixes manual scalar derivatives, four framework implementations, and algorithmic interpretation. Repeated setup comments crowd the conceptual path. | Use one backend-neutral graph and table of local derivatives in the main text; move framework variants to tabs after the conclusion. |
| C25-11 | Moderate | `mdl-multivariable-calculus.md:821-923` | “Finish the story” and later “decides everything” turn the Hessian test into a dramatic ending. Positive/negative definiteness decides strict local behavior only under stated smoothness, while semidefinite cases are inconclusive. | Lead with the exact theorem and a three-way decision table, including the inconclusive semidefinite case. |
| C25-12 | Moderate | `mdl-matrix-calculus-autodiff.md:56-99` | The local-linear-map explanation is excellent, but “up close, a differentiable map is a linear map” should retain “approximately”; the circle-to-ellipse picture is first-order and fails at finite radius. | Keep approximation language in the claim and state what the residual does as radius shrinks. |
| C25-13 | Major | `mdl-matrix-calculus-autodiff.md:226-261` | Layout conventions appear after Jacobians, gradients, and chain rules have already been used. Readers may have silently assumed the opposite convention. | State the convention at first use, then keep this subsection as a comparison/reference box. |
| C25-14 | Moderate | `mdl-matrix-calculus-autodiff.md:693-944` | “The dimensions of the Jacobian determine which mode is more efficient” is useful but incomplete: graph structure, number of requested products, batching, sparsity, and memory also matter. | Present the input/output dimension rule as the dense full-Jacobian baseline and name the exceptions. |
| C25-15 | Critical | `mdl-integral-calculus.md:56-82` | The Riemann integral is defined as `ε→0` for an unspecified choice of sample points and equal slices. A rigorous definition needs a sequence of tagged partitions/mesh size; the claim that bounded piecewise-continuous functions cover “everything” in ML is false (unbounded densities and improper integrals occur immediately). | Define the equal-grid Riemann sum as an introductory construction, then state the proper mesh-limit definition or explicitly mark the restricted setting. Remove the universal coverage claim. |
| C25-16 | Major | `mdl-integral-calculus.md:65` and `:353-371` | The first caption calls the integral “the exact signed area” before signed area is explained much later. For functions below the axis, ordinary geometric area and the integral differ. | Say “net signed area” and introduce the sign convention at the first definition, not hundreds of lines later. |
| C25-17 | Moderate | `mdl-integral-calculus.md:509-555` | Multidimensional change of variables is called “the key that unlocks” the Gaussian integral. The phrase is self-conscious and the determinant’s geometric role deserves a direct explanation. | Explain locally: the Jacobian determinant converts small volume elements; then show why polar coordinates separate the Gaussian integral. |
| C25-18 | Moderate | Slide blocks across all four files | Many titles are strong claims, but generic “Summary,” capitalization inconsistencies, and dense proof/code slides remain. Some slides reproduce notebook code rather than teach the mathematical invariant. | Normalize sentence case, use result-oriented titles, and replace long code with pseudocode or one highlighted line plus output. |

## Math and notation

The chapter’s unifying notation is a strength. Correct the `sin(x^x)` domain, formalize the integral definition, state norm dependence for steepest descent, and place Jacobian layout conventions earlier. Preserve the distinction between differentiability and existence of partial derivatives and the JVP/VJP cost analysis.

## Figures, captions, and slides

Most figures genuinely support the argument. Captions should preserve approximation language and identify limiting behavior. Slides need pruning where code or proofs are too dense to parse in presentation time.

## Code and experiment pedagogy

The backend comparisons are useful for readers who implement these ideas, but repeated imports and near-identical cells obscure the mathematics. Keep one mathematical target per experiment, state the expected scaling/error before the cell, and interpret discrepancies (finite differences, dtype, nondifferentiable conventions) afterward.

## Recurring artifacts

- Totalizing slogans (“generates everything,” “decides everything”).
- “Simply,” “obvious,” “finish the story,” and “key that unlocks.”
- Proofs compressed into a single long paragraph.
- Setup cells inserted in the middle of a conceptual explanation.

## Strengths to preserve

- Local linear approximation as a consistent conceptual spine.
- Explicit counterexamples and error-order checks.
- Clear bridge from computational graphs to JVPs and VJPs.
- Treatment of nonsmooth points and framework derivative conventions.
- Integration connected to probability and gradient estimators.

## Prioritized revision plan

1. Fix the domain and integral-definition issues.
2. Reframe Newton’s method with conditions and failure modes.
3. Clarify differentiability before the coordinate derivation and move layout conventions earlier.
4. Break long proofs into idea–steps–conclusion form.
5. Prune setup/code from the narrative and simplify slides.
