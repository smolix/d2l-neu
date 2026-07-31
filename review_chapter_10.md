# Chapter 10 Style Review: Attention

## Scope and files reviewed

Diagnosis only. I reviewed every tracked Markdown file in `chapter_attention`: `index.md`, `queries-keys-values.md`, `attention-scoring.md`, `multihead-attention.md`, `positional-information.md`, `attention-at-scale.md`, and `what-attention-computes.md`. The audit covers prose and titles, equations/proofs and notation, captions/tables, code/experiment explanations, summaries, and every slide block.

## Executive assessment

The chapter’s core technical progression is excellent: differentiable lookup, score normalization, multiple heads, position, scale, and learned circuits. It frequently follows the four-level explanation pattern and contains unusually explicit theorem limitations. The main issues are concentrated in the chapter index, where a dense roadmap overstates several conclusions; the database analogy, which includes an undefined approximate match; apples-to-oranges complexity accounting; and mechanistic-interpretability claims that shift between exact algebra, observational evidence, and causal evidence. Several slide figures lack captions.

## Scores (0–10)

| Dimension | Score | Rationale |
|---|---:|---|
| Writing quality | 7 | Mostly direct and controlled; index narration, anthropomorphic ownership, and a few slogans remain. |
| Explanation quality | 8 | Strong motivation–equation–experiment links and clear small counterexamples. |
| Technical quality | 7 | Good derivations, but complexity comparisons, approximation conditions, experiment scope, and mechanistic evidence need sharper labels. |

## Architecture and logical order

The file order follows necessity well. Fixed-state retrieval motivates attention; scoring makes weights learnable; multi-head attention removes a restricted one-mixture bottleneck; position repairs permutation equivariance; scale addresses quadratic interactions; circuit analysis asks what trained heads implement. The index should state this dependency in six short moves. Its present section-by-section paragraph is too long and includes results before their assumptions. Within `queries-keys-values.md`, kernel regression should be introduced immediately after soft lookup; the four-regime list can be shortened. Within `what-attention-computes.md`, distinguish the exact algebraic framework first, the behavioral hypothesis second, and causal/weight tests third.

## Section/file issue table

| ID | Severity | Evidence | Excerpt / description | Violated rule | Diagnosis | Concrete revision |
|---|---|---|---|---|---|---|
| C10-01 | M | `index.md:23–51` | One paragraph summarizes every section, including “proves ... smallest task” and “clearest laboratory example” | §§5.1, 8.2, 15.3: roadmap should follow problem and avoid over-signposting | The 29-line sentence sequence overloads the opening and advertises results before conditions. “Clearest” is promotional. | Reduce to a dependency map: lookup → scoring → heads → position → cost → circuits. Move exact results and caveats to their sections. |
| C10-02 | M | `index.md:53–66` | Run-time logistics; “deliberate economy”; appendix sections “own” topics | §§8.1, 17.2, 17.7: discuss subject, not authorial staging; avoid anthropomorphism | The paragraph foregrounds editorial choices and personifies chapters instead of giving prerequisites/scope. | State concrete scope and exclusions in two sentences; move runtime estimates next to experiments; replace “owns” with cross-references. |
| C10-03 | H | `queries-keys-values.md:12–31` | Missing exact key would retrieve “Lipton” under approximate matching | §§10–11, 16.1: examples require defined mechanism | No similarity metric or threshold is defined, so the asserted approximate answer is arbitrary. The example also spends many proper names on a mechanism a two-key toy case could show. | Use a small numeric/string-key example with an explicit similarity score; show exact match, soft weights, and the returned weighted value. |
| C10-04 | M | `queries-keys-values.md:55–89` | Defines arbitrary real weights, then four regimes and softmax | §§6.2, 9.7: avoid serial definitions without relationship | The general linear-combination definition is broader than the normalized nonnegative attention used throughout, and the cone/one-hot/average list delays the operational definition. | Define score → softmax weights → weighted value first. Present unnormalized/signed variants later as extensions and state which chapters use them. |
| C10-05 | H | `attention-scoring.md:53–68` | Independent zero-mean unit-variance query/key elements justify (1/\sqrt d) | §§9.1, 9.6, 16.2: assumptions adjacent; intuition versus exact result | The variance calculation is exact under independence but learned projections generally violate it. “Keeps variance at 1 regardless of dimension” is not a trained-model guarantee. | Label this an initialization heuristic under stated iid assumptions; say scaling removes the nominal (d) dependence and test actual score variance after training separately. |
| C10-06 | M | `multihead-attention.md:31–65`, limitation at `141–145` | Proposition headline says a single head loses half; restriction appears later | §§8.8, 9.8, 16.2: burden-bearing assumptions before theorem | The proof is good, but readers encounter a broad title/claim before value-blind fixed keys, one layer, no residual, Gaussian values, and linear readout are consolidated. | Put all assumptions in the proposition statement and retitle “A One-Mixture Bottleneck with Position-Only Keys.” Keep the later limitation paragraph immediately after the proposition. |
| C10-07 | H | `attention-at-scale.md:61–84` | CNN includes (d\times d) maps, RNN includes state maps, attention reports only (O(n^2d)) “mixing only” | §§13.4, 16.1: matched comparison | The table omits Q/K/V/output projections (O(nd^2)) only for attention, so regimes with (n<d) are misrepresented. It also calls path length a predictor of learnability more strongly than established. | Give total attention cost (O(nd^2+n^2d)) and a separate mixing-only row for all methods; define path length as an architectural proxy, not a guarantee. |
| C10-08 | H | `attention-at-scale.md:832–846` | “about 4 MiB per head at (n=8192) in fp32”; “growing linearly forever”; production interleaving claim | §§13.4, 16.2: dimensions and evidence | KV-cache memory depends on head dimension and stores both K and V; the stated 4 MiB is not reproducible without (d_h). “Forever” is rhetorical and production trend needs evidence. | Show (2nd_h\times4) bytes and plug in the local (d_h); say linear in context length; cite concrete hybrid architectures or qualify as an observed design pattern. |
| C10-09 | H | `what-attention-computes.md:61–72` | Fixed patterns make the map linear, therefore “complete mechanistic analysis becomes possible” | §§9.6, 16.1, 17.4 | Algebraic expansion is possible, but complete explanation does not follow: attention patterns themselves depend on input, path sums can be enormous, and feature meaning remains unresolved. | Say exact path expansion is available conditional on fixed patterns, which makes certain circuit hypotheses testable. Reserve “complete” for a demonstrated exhaustive account. |
| C10-10 | H | `what-attention-computes.md:599–657`, `704–719` | OV diagonal check; summary later calls period changes/ablations “causal evidence” | §§13.4, 16.1: classify evidence precisely | The shown OV test is partial and correlational, as the text admits. Period changes are input interventions, but “ablating model components” must name which ablations and controls support the circuit rather than the task generally. | Add an explicit evidence table: behavioral, activation/attention pattern, OV weight test, targeted head/path ablation, and counterfactual patching. Report results/controls or downgrade “causal.” |
| C10-11 | M | `positional-information.md:336–370`, experiment `558–578` | “hope perplexity survives”; five schemes trained once with fixed seed; conclusions about extrapolation | §§13.3–13.4, 17.7 | The experiment is a valuable illustration but one corpus/model/seed cannot rank positional schemes generally; “hope” personifies the evaluation. | State the test criterion mathematically, run/report multiple seeds or mark it as one controlled instance, and keep conclusions local to context 128 and this model. |
| C10-12 | M | slides `attention-at-scale.md:1059,1093` | Empty captions on architecture and online-softmax figures | §§12.2, 19.5 | The audience is not told which edge/path or memory object to compare. | Add a compact conclusion beneath each figure: path/sequential tradeoff for the first; current block plus running max/sum/output state for the second. |

## Math and notation

- `queries-keys-values.md` should settle dimensions of queries, keys, values, and output before the general sum; values need not share key/query width.
- In `attention-scoring.md`, distinguish key/query width (d_k) from model width (d); the scale is (1/\sqrt{d_k}).
- `multihead-attention.md` uses single-vector notation and then sequence FLOPs. Define matrix shapes (Q\in\mathbb R^{n_q\times d}), (K,V\in\mathbb R^{n_k\times d}) before counting.
- `positional-information.md` gives a clean permutation-equivariance proof; specify that dropout is disabled and no causal mask/position-dependent bias is present.
- `attention-at-scale.md` must include projection costs and KV-cache dimensions (C10-07, C10-08).
- `what-attention-computes.md` should state row/column orientation for (W_{QK}), (W_{OV}), and embedding matrices consistently across PyTorch and JAX code.

## Figures, captions, and slides

Main-text captions are generally strong and interpretive, especially the one-head bottleneck and induction-circuit diagrams. The CNN/RNN/attention caption is too generic for a figure carrying three complexity concepts; say what edges and path lengths represent. The alignment figure correctly says it is schematic. Slide copies of the cost figures are missing captions (C10-12), and abbreviated captions should preserve key assumptions such as “one layer, position-only keys.”

## Code and experiment pedagogy

The chapter often explains purpose before code and interprets outputs afterward. Improve experimental discipline where results become broad claims: report seed count for positional extrapolation and circuit formation; give device/dtype/synchronization for time/memory; distinguish a numerical identity check from evidence for a learned mechanism. In circuit analysis, targeted interventions are more probative than attention heatmaps. Private/specialized accelerator APIs should have availability/fallback notes.

## Recurring artifacts

- Dense roadmap sentences and visible editorial narration.
- Anthropomorphic chapter/model language: sections “own” topics; attention “ignores” or “remembers” without immediately naming equivariance/state.
- Exact algebra followed by overly broad interpretability conclusions.
- Unmatched complexity or memory comparisons.
- Empty slide captions that remove the main text’s interpretation.

## What already works

- The chapter opens with a concrete fixed-state retrieval bottleneck.
- `attention-scoring.md` explicitly separates Gaussian equivalence at equal key norms from adopting dot product as a modeling choice.
- `multihead-attention.md` supplies a small theorem, proof, experiment, and an explicit limitation paragraph.
- `positional-information.md` proves permutation equivariance before introducing position schemes.
- `what-attention-computes.md:648–657` candidly describes what the OV check does not establish.

## Prioritized revision plan

1. Correct C10-03, C10-05, C10-07 through C10-10.
2. Rewrite the index as a compact technical dependency and remove editorial staging.
3. Standardize dimensions and cost accounting across scoring, multi-head, and scale sections.
4. Put theorem/experiment assumptions before claims and keep conclusions local to protocols.
5. Strengthen circuit evidence labels and add targeted causal tests or downgrade wording.
6. Restore compact interpretive captions in slides and preserve main-text caveats.
