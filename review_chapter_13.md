# Style review: Chapter 13, Computational Performance

## Scope

Reviewed every tracked Markdown source in `chapter_computational-performance`:
`index.md`, `performance-model.md`, `hardware.md`, `memory-precision.md`,
`compilation.md`, `multiple-gpus.md`, `multi-gpu-practice.md`,
`fast-transformer.md`, and build-only `legacy-multigpu-lib.md`. The review covers
prose, mathematics, captions, code and benchmarks, summaries, exercises, and all
slide blocks. Line references are to the current sources.

## Executive assessment

This is a technically useful performance chapter with unusually good coverage of
roofline reasoning, memory traffic, compilation, distributed collectives, and
end-to-end transformer optimization. The code generally makes costs observable
instead of treating speed as magic. Its main stylistic problem is an insistent
metaphor system--*bill*, *price*, *rung*, *ladder*, *wall*, *cure*--that appears in
almost every file and is densest in `fast-transformer.md`. The repetition makes
many distinct resource tradeoffs sound identical. The chapter also mixes
portable reasoning with machine-specific measurements and fast-changing hardware
facts without always marking the boundary. A revision should lead with invariant
cost models, state measurement conditions beside results, and reserve survey
claims for clearly dated passages.

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Writing | 6/10 | Direct in many derivations, but repetitive economic/climbing metaphors and slogan slides dominate the chapter's voice. |
| Explanation | 8/10 | Strong progression from models to measurements; some very long implementation sequences need explicit intermediate conclusions. |
| Technical | 8/10 | Good operational content, with insufficiently standardized benchmark metadata and several quickly aging hardware claims. |

## Architecture and order

The order from cost models to hardware, precision, compilation, distribution,
and a transformer case study is sound. The two multi-GPU files overlap and should
be given distinct jobs: one for collective algorithms and scaling equations, one
for a reproducible training recipe. `fast-transformer.md` is long enough to
function as a second chapter; either split it or turn it into a capstone that
refers back to one shared cost table instead of rebuilding a “ladder” of
optimizations. Keep `legacy-multigpu-lib.md` explicitly marked as build-only or
remove it from the tracked chapter sources.

## Issue inventory

| ID | Severity | Evidence | Excerpt or description | Violated rule | Diagnosis and concrete revision |
|---|---|---|---|---|---|
| C13-01 | High | `chapter_computational-performance/index.md:32-41` | The overview introduces a “communication bill” and a compilation “rung.” | Explain the concrete problem before abstraction; avoid recurring economic/climbing metaphors. | Name communicated bytes and launch/graph overhead directly, then map each later section to one measurable bottleneck. |
| C13-02 | High | `chapter_computational-performance/performance-model.md:102-156` | “One line” is called the “workhorse,” followed by “only half the story.” | Avoid presenter narration and vague claims of completeness. | State the roofline bound, define its assumptions, then give a separate paragraph for omissions such as latency, occupancy, and fusion. |
| C13-03 | Medium | `chapter_computational-performance/performance-model.md:377-458` | “Armed with” and “genuinely elegant” culminate in silicon “earning its price.” | Use neutral interpretation; do not evaluate a derivation aesthetically. | Replace the sequence with the observed arithmetic intensity, limiting resource, and predicted direction of optimization. |
| C13-04 | High | `chapter_computational-performance/performance-model.md:533-538` | A bottleneck “bleeds” performance and its “cure” fits in one line. | Avoid medical/dramatic metaphors and overselling compact formulas. | Identify the redundant traffic or launch explicitly and state how the proposed transformation changes the model. |
| C13-05 | High | `chapter_computational-performance/hardware.md:70-88` | Hardware is organized as a ladder of rungs. | Titles and structure should describe concepts, not a rhetorical scaffold. | Organize by compute throughput, bandwidth, hierarchy, parallelism, and communication. Use one comparison table with units. |
| C13-06 | High | `chapter_computational-performance/hardware.md:178-232` | A memory “wall,” several rungs, a “catch,” and what each step “buys” occur in one short span. | Prefer exact causal explanation to compressed metaphor. | For every example, state working-set size, hierarchy level, bandwidth/latency constraint, and the resulting bound. |
| C13-07 | High | `chapter_computational-performance/hardware.md:339-389` | “Everything” and “where performance dies” introduce a final ladder. | Avoid universal and catastrophic language. | Restrict claims to the measured kernel and hardware generation; use descriptive subsection titles. |
| C13-08 | High | `chapter_computational-performance/hardware.md:492` | Slide title: “Energy explains everything twice.” | Slides require bounded, descriptive claims. | Name the two effects--data movement energy and thermal/power limits--and quantify or cite them. |
| C13-09 | High | `chapter_computational-performance/memory-precision.md:184-283` | A “format ladder” is “promised” and “cashed,” followed by “one sentence” that is called philosophical. | Separate representation, numerical error, and hardware support; avoid self-commentary. | Replace with a table of exponent/mantissa/range, accumulation rule, supported kernels, and observed error. |
| C13-10 | Medium | `chapter_computational-performance/compilation.md:154-386` | “The contrast in one line,” a repaid “price,” “the catch,” and repeated “pays” narrate the examples. | Technical transitions should say what changed and why. | After each graph or fusion example, list eliminated intermediates/launches, compilation cost, shape constraints, and measured outcome. |
| C13-11 | High | `chapter_computational-performance/compilation.md:491` | Summary: a “bleeding chain” is “cured in one line.” | Summaries should reconstruct mechanisms without slogans. | Summarize tracing, fusion, specialization, guard failures, and amortization in literal terms. |
| C13-12 | Medium | `chapter_computational-performance/multiple-gpus.md:455-488` | A collective has a “catch,” is “elegant,” and later “pays.” | Avoid aesthetic judgments and economic metaphors. | Compare transferred bytes, number of rounds, overlap opportunity, topology assumptions, and failure conditions. |
| C13-13 | High | `chapter_computational-performance/multi-gpu-practice.md:351` | “Agreement is the result and the price.” | Do not substitute symmetry for an explanation. | Say which replicas must agree, when synchronization occurs, and the communication and staleness consequences. |
| C13-14 | High | `chapter_computational-performance/multi-gpu-practice.md:550-678` | A benchmark is framed as a “reveal” and ends with “the punchline.” | Experiments need purpose, protocol, result, and bounded interpretation. | Add a benchmark table with topology, interconnect, framework/version, precision, batch sizes, warm-up, repetitions, variability, and scaling efficiency. |
| C13-15 | High | `chapter_computational-performance/fast-transformer.md:64-234` | The opening establishes an optimization ladder with rungs, walls, prices, and “honest” measurement. | Avoid moralized terminology and a metaphor replacing chapter structure. | Define baseline semantics and environment once; number optimizations by mechanism (fusion, layout, attention kernel, compilation), not by “rung.” |
| C13-16 | High | `chapter_computational-performance/fast-transformer.md:353-490` | Dense sequences of “pays,” “wall,” and “rung” blur several separate kernel changes. | Each experiment should answer one declared question. | Give each change its own hypothesis, invariant correctness test, timing result with error, and explanation linked to the cost model. |
| C13-17 | High | `chapter_computational-performance/fast-transformer.md:547-764` | The cumulative optimization narrative gives machine-specific timings without one complete reproducibility record. | Evidence must be reproducible and conclusion strength proportional. | Add a single environment/protocol table and a cumulative ablation table; report distributions, not isolated timings. Restrict conclusions to tested shapes and hardware. |
| C13-18 | Medium | `chapter_computational-performance/fast-transformer.md:1346-1436` | Slides say “Nothing new—that is the point,” “Measure Honestly,” and call a rung “discipline not cast.” | Slides must use descriptive titles and avoid slogans/moral language. | Retitle with the actual validation principle, baseline definition, and transformation; turn fragments into complete claims. |
| C13-19 | High | `chapter_computational-performance/hardware.md:204-232` | Device examples and performance figures are presented as current hardware facts. | Fast-changing facts require dates, sources, and separation from invariant principles. | Add model, release/generation, source, and “as of” date to tables and captions; keep the derivation hardware-neutral. |
| C13-20 | Low | `chapter_computational-performance/legacy-multigpu-lib.md:1-132` | A tracked legacy source is excluded from the rendered chapter but still reads as chapter material. | Scope and navigation should be explicit. | Add an unmistakable archival/build-only notice and cross-reference the maintained replacement, or remove it in a separate cleanup. |

## Mathematics and notation

- Keep units visible in all operational-intensity, bandwidth, throughput, and
  collective-volume equations; distinguish ideal upper bounds from empirical
  fits.
- State whether FLOP conventions count fused multiply-add as one or two
  operations and use the convention consistently.
- Collective-cost equations need topology, link concurrency, and message-size
  assumptions beside the formulas.
- Numerical-format sections should distinguish representable range, rounding
  model, accumulation precision, loss scaling, and empirical training stability.

## Figures, captions, and slides

Roofline and memory-traffic figures are useful when their hardware parameters and
units are explicit. Every machine-specific plot should identify device, software,
precision, tensor shape, and timing statistic in the caption or adjacent table.
Avoid captions that merely announce a “ladder” or “wall.” Slide blocks require an
independent pass: several preserve slogan titles and moralized language absent
from the derivation. Hardware diagrams and tables should be date-stamped.

## Code and experiment pedagogy

Correctness checks before timing are an excellent pattern and should remain.
Standardize timing helpers: synchronization, warm-up, compilation exclusion or
inclusion, repetition count, summary statistic, and uncertainty. Make semantic
equivalence tests explicit after every fusion or reduced-precision change. A
cumulative capstone needs a stable baseline, per-step ablation, and a note that
optimized rankings may change with shapes and hardware.

## Recurring artifacts

- “Bill/price/pay/buy” for unrelated compute, memory, communication, and accuracy
  tradeoffs.
- “Ladder/rung/wall” as a substitute for descriptive organization.
- “Catch,” “punchline,” “cure,” “bleeding,” and “dies” as manufactured drama.
- “Honest” used to moralize baseline or measurement choices.
- Portable theory and dated device measurements presented without a visible
  boundary.

## Positive patterns to preserve

- Performance is derived from bytes, operations, synchronization, and dependency
  structure rather than attributed to framework magic.
- Code frequently checks numerical agreement before comparing speed.
- The chapter connects local kernel changes to end-to-end training behavior.
- Multi-GPU material treats communication as part of the algorithm.

## Prioritized revision plan

1. Replace the chapter-wide ladder/economic metaphor system with descriptive
   resource categories and one shared cost table.
2. Add a standard reproducibility record to every benchmark and one complete
   environment table to the capstone.
3. Split or compress `fast-transformer.md`; turn its steps into controlled
   ablations tied to the earlier performance model.
4. Date and source all hardware-specific claims, figures, and tables.
5. Clarify the distinct purposes of the two multi-GPU sections and archive the
   legacy source explicitly.
6. Rewrite captions, summaries, and slide titles after the structural pass.
