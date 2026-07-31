# Style review: Chapter 12, Modern Recurrent Architectures

## Scope

Reviewed every tracked Markdown source in `chapter_recurrent-modern`: `index.md`,
`lstm.md`, `ssm.md`, `matrix-state.md`, `deltanet.md`, `mamba.md`, `hybrids.md`,
and `test-time-regression.md`, including prose, displayed mathematics, captions,
experiments, summaries, exercises, and all slide blocks. Line references below
refer to the current source files.

## Executive assessment

The chapter has a valuable organizing idea: recurrent models can be compared by
their state, update rule, readout, and parallel training form. Its derivations are
usually careful and its small experiments often expose exactly the intended
failure mode. The prose, however, repeatedly replaces technical transitions with
the same dramatic vocabulary--*ladder*, *rung*, *wall*, *price*, *debt*, and
*promise*. Across nearly ten thousand lines this becomes a parallel narrative
that obscures the actual architecture. Several sections also put synthetic
diagnostics, small language-model runs, published results, and current-practice
claims on the same evidentiary footing. The largest improvement would come from
shortening the chapter, making each comparison explicit, and letting equations
and measured results carry the argument.

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Writing | 6/10 | Generally grammatical and energetic, but repetitive metaphors, fragments, slogans, and oversized paragraphs are pervasive. |
| Explanation | 7/10 | Strong mechanisms and diagnostic examples; the accumulated taxonomy and repeated scoreboards make the route through the chapter harder to follow. |
| Technical | 8/10 | Substantively careful in most derivations, but evidence classes, assumptions, and limits need more consistent labeling. |

## Architecture and order

The progression from gated scalar memory to structured state spaces, matrix
state, data-dependent recurrence, and hybrids is defensible. It is nevertheless
too long for a single chapter and repeatedly restarts the same comparison. Put a
single state/update/readout/parallelization table near the beginning; make each
subsequent section answer one new question against that table. Separate
foundational mechanisms (`lstm`, `ssm`, `matrix-state`) from the fast-moving model
survey (`deltanet`, `mamba`, `hybrids`, test-time regression), or explicitly label
the latter as a dated survey. Move repeated scoreboards and implementation
details to one comparison section or appendix.

## Issue inventory

| ID | Severity | Evidence | Excerpt or description | Violated rule | Diagnosis and concrete revision |
|---|---|---|---|---|---|
| C12-01 | High | `chapter_recurrent-modern/index.md:17-25` | The introduction says the chapter will “carry the story” through a “decay ladder” and a “promised duality.” | Prefer concrete problems and ordinary transitions; avoid self-conscious narration. | State the comparison directly: which state representation each family uses, what can be parallelized, and what dependency it preserves. Remove “story,” “ladder,” and “promise.” |
| C12-02 | High | `chapter_recurrent-modern/index.md:59-60` | Attention “hides” a deficit and “pays” a growing cache. | Avoid economic and concealment metaphors when exact resource quantities are available. | Give the inference-memory complexity and the dependency it buys in literal terms; name the sequence-length and head-dimension assumptions. |
| C12-03 | Medium | `chapter_recurrent-modern/index.md:84` | The overview ends at a “collapsed wall.” | Titles and transitions should describe content, not manufacture drama. | Replace with the actual limiting case and point to the section that analyzes it. |
| C12-04 | High | `chapter_recurrent-modern/lstm.md:75` | A baseline has “terrible memory” because it remembers “everything forever.” | Avoid anthropomorphism and universal claims. | Explain that a unit forget factor produces no selective decay and retains all coordinates at the same rate; then state why that is unsuitable. |
| C12-05 | Medium | `chapter_recurrent-modern/lstm.md:703-823` | Several transitions say a construction “pays,” “unlocks,” or is the “first rung.” | Use a need-driven technical sequence rather than a repeated figurative scaffold. | Replace each with the operation enabled: content-dependent retention, selective overwrite, or parallel scan. Consolidate the comparison in a table. |
| C12-06 | Medium | `chapter_recurrent-modern/lstm.md:928` | The slide sequence relies on clipped questions and fragments. | Slides must meet the same sentence and explanatory standards as prose. | Turn slide titles into descriptive claims and supply the condition/result pair on each slide. |
| C12-07 | Medium | `chapter_recurrent-modern/ssm.md:196` | The text says one can “pay a logarithmic factor for code that fits on one slide.” | State tradeoffs in technical terms; do not address presentation convenience as evidence. | Compare scan work, depth, and storage with the sequential recurrence. Remove the remark about fitting on a slide. |
| C12-08 | High | `chapter_recurrent-modern/ssm.md:336` | Parallel convolution is “the moment an SSM trains like a CNN.” | Avoid slogans that collapse distinct mechanisms. | Say precisely that a time-invariant linear recurrence is equivalent to a causal convolution for a fixed kernel, then list the boundary conditions and computational consequence. |
| C12-09 | High | `chapter_recurrent-modern/ssm.md:1191-1217` | A local timing experiment supports broad statements about SSM training behavior. | Match conclusion strength to experimental scope. | Report device, shapes, compilation/warm-up, precision, repetitions, and variability; restrict the conclusion to the tested implementations. |
| C12-10 | Medium | `chapter_recurrent-modern/ssm.md:1232-1462` | “Time to collect,” “the punchline,” and “Everything…” frame the synthesis. | Avoid presenter language and universalizing summaries. | Organize the synthesis by exact claims: convolutional equivalence, discretization, stability, and measured cost. |
| C12-11 | High | `chapter_recurrent-modern/matrix-state.md:202-292` | One passage repeatedly says capacity “collapses,” a number is “everything,” and a ladder’s next rung “makes good on” a promise. | Separate definitions, propositions, and interpretations; avoid drama. | Split the passage after the capacity calculation. State the assumptions on keys, give the result, and interpret collision error without the metaphor chain. |
| C12-12 | Medium | `chapter_recurrent-modern/matrix-state.md:756` | Table header: “origin, in one line.” | Tables should use stable, descriptive field names. | Rename the column to the actual property, such as “update derivation” or “objective inducing the update,” and give full mathematical conditions in the cells or text. |
| C12-13 | High | `chapter_recurrent-modern/deltanet.md:354-535` | “Training does not rescue” the limitation; one task is “harder than everything.” | Avoid categorical causal claims unsupported by the diagnostic. | Describe what the fixed setup failed to learn, report run variation, and replace “everything” with the relevant distractor or interference regime. |
| C12-14 | Medium | `chapter_recurrent-modern/deltanet.md:676-715` | A long ladder/rung/price passage substitutes for the sequence of update-rule variants. | Make architecture changes explicit and comparable. | Use aligned equations or a table showing normalization, erase term, gate, state size, and consequence for each variant. |
| C12-15 | High | `chapter_recurrent-modern/deltanet.md:1190-1285` | “One eigenvalue” is called the “entire content”; a later diagnostic “collapses like guesswork.” | Interpret measurements precisely and distinguish result from rhetoric. | State which spectral statistic controls which bound, and report measured accuracy against a chance baseline with uncertainty. |
| C12-16 | High | `chapter_recurrent-modern/mamba.md:806-1136` | “The debt falls due,” “One question, three answers,” and “pays” frame three distinct mechanisms. | Avoid economic metaphors and slogan captions; captions must identify evidence. | Name the limitation of a fixed kernel, then distinguish selective parameters, scan implementation, and hardware-aware layout. Rewrite the caption to identify plotted variables and conclusion. |
| C12-17 | High | `chapter_recurrent-modern/hybrids.md:139-173` | Attention and recurrence are described using “bill,” “rent,” and “price.” | Use literal resource and modeling terms. | Compare cache memory, arithmetic intensity, receptive path, and state compression explicitly, with the same notation for all branches. |
| C12-18 | Medium | `chapter_recurrent-modern/hybrids.md:569-772` | “The thesis in one picture,” a “hidden deficit,” and components that “pay” continue the chapter-wide metaphor chain. | Figures should support a proposition, not serve as a rhetorical climax. | Make the caption self-contained: architecture variants, axes, controlled quantities, and the observed tradeoff. Refer to that evidence in restrained prose. |
| C12-19 | High | `chapter_recurrent-modern/test-time-regression.md:56-59` | “Nothing was learned” is immediately called “the trap.” | Avoid absolutes and manufactured suspense. | Say which parameters remain fixed, which state changes at inference, and why calling the state update “learning” requires a declared convention. |
| C12-20 | High | `chapter_recurrent-modern/test-time-regression.md:1001-1007` | “One question” and “not merely” compress several equivalence and modeling claims. | Do not conflate algebraic identity, algorithmic interpretation, and empirical advantage. | Give separate paragraphs for the regression objective, the induced update identity, and any empirical hypothesis. State assumptions for each. |

## Mathematics and notation

- The recurring state/update/readout notation is the chapter's strongest
  organizing device, but it should be declared once and reused exactly. Several
  later sections introduce a fresh vocabulary before mapping back to it.
- Label algebraic equivalences (recurrence/convolution, delta update/regression)
  as identities under stated assumptions. Keep them separate from stability,
  memory-capacity, or optimization claims.
- For discretized SSMs, collect assumptions on continuous-time stability,
  discretization rule, input holding, and complex parameterization near the first
  formula instead of distributing them across the later interpretation.
- Capacity diagnostics based on random or isotropic keys are useful, but their
  assumptions must appear in theorem/proposition statements and summaries.

## Figures, captions, and slides

Many diagnostic captions successfully name the quantity and expected behavior.
The weaker captions act as taglines--notably “One question, three answers” and
the ladder/rung captions--without stating axes, controls, or inference. Replace
these with self-contained captions. Slides frequently reintroduce fragments and
metaphors removed from nearby prose; review them independently rather than as
abridged duplicates. Current-model family trees should carry an “as of” date and
citations.

## Code and experiment pedagogy

The code often follows a productive pattern: implement the recurrence, verify an
equivalent form, then probe a failure mode. Preserve that. Standardize every
benchmark with purpose, hypothesis, tensor shapes, device/software/precision,
warm-up and compilation treatment, repetitions, variability, and the exact scope
of the conclusion. Small character-level language models and synthetic recall
tasks should be labeled diagnostics, not evidence of general model quality.

## Recurring artifacts

- Repeated “ladder/rung/wall” architecture metaphors.
- Repeated economic vocabulary: “price,” “pay,” “bill,” “rent,” and “debt.”
- “Everything,” “nothing,” “the punchline,” and “the story” used where a bounded
  claim is needed.
- Slide titles that pose theatrical questions or use fragments.
- Multiple local scoreboards that duplicate the chapter-level comparison.

## Positive patterns to preserve

- State, update, and readout provide a coherent cross-family comparison.
- The recurrence/convolution and sequential/scan checks make abstract identities
  executable.
- Synthetic tasks frequently isolate interference, decay, or selective recall
  instead of reporting an uninterpretable aggregate score.
- Summaries often reconnect equations to compute and memory consequences.

## Prioritized revision plan

1. Replace the recurring metaphor system with a single literal comparison table.
2. Split or explicitly partition foundational material from the dated model
   survey; remove repeated introductions and scoreboards.
3. Separate identities, theoretical results, synthetic diagnostics, local
   benchmarks, and cited large-scale evidence.
4. Rewrite C12-09, C12-13, C12-15, and all broad experimental conclusions with
   complete protocols and bounded claims.
5. Audit captions and every slide independently for descriptive titles, complete
   sentences, and stated evidence.
6. Perform a final notation and assumption pass across all eight files.
