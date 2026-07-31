# Chapter 11 Review: Transformers

## Scope

Reviewed every tracked Markdown source in `chapter_transformers/`:
`index.md`, `transformer-block.md`, `gpt.md`, `kv-cache.md`,
`encoders-decoders.md`, `vision-transformer.md`, `moe.md`, and
`scaling-laws.md`. The review covered prose, headings, equations and
notation, captions, code and experiment explanations, summaries, exercises,
and all slide blocks. Line numbers refer to the current sources.

## Executive assessment

This is a technically ambitious and generally coherent chapter. Its strongest
feature is the use of one configurable implementation to connect block design,
GPT, caching, alternative wirings, vision, MoE, and scaling. Most openings now
state concrete computational questions, important figures are unusually
self-contained, and experiment claims are often scoped to the small teaching
setting.

The principal weakness is uneven restraint. Several later sections and many
slides revert to theatrical titles, compressed slogans, and economic
metaphors. A second weakness is evidential calibration: current-model surveys
and broad claims about modern practice are presented alongside controlled
notebook experiments without always marking the different evidential status.
The chapter also occasionally lets implementation detail precede the conceptual
question, especially in cache compression and scaling.

## Scores

- Writing quality: **7/10**
- Explanation quality: **8/10**
- Technical quality: **8/10**

## Chapter architecture and logical order

The dependency order is mostly sound: block anatomy precedes GPT; GPT motivates
the KV cache; the shared block then supports alternate encoder/decoder wirings,
vision, and MoE; parameter and compute accounting close the chapter. The main
architectural weakness is that scaling laws combine three jobs—cost accounting,
a small empirical scaling study, and a survey of current configurations. The
survey is useful, but it weakens the section's main question and ages faster
than the derivation. The vision and MoE sections also read partly as standalone
case studies rather than consequences of a specific unresolved question from
the preceding section.

The chapter overview is clear, but lines 35–43 of `index.md` begin a historical
survey before completing a concise scope statement. Move unstable “current
model” material to a dated comparison table or resource note and keep the main
argument centered on reusable architectural decisions.

## Section- and file-level issues

| ID | Severity | Evidence | Violated rule | Diagnosis | Revision direction |
|---|---|---|---|---|---|
| C11-01 | medium | `index.md:35-43`: “What survived is the block; nearly every choice around it changed” | §5 opening sequence; §16 claim strength | The broad historical claim is plausible but not scoped by model family, date, or evidence, and it interrupts the roadmap. | Replace with a dated, qualified statement tied to the configurations actually tabulated; move the broader survey after the chapter's scope. |
| C11-02 | medium | `transformer-block.md:289`: “rank collapse ... caught in the act” | §8.4 ordinary verbs; §17.8 drama | A standard technical term is followed by theatrical narration. The experiment measures token similarity at initialization, not the full training phenomenon in the cited paper. | State the measured quantity and its relation to the cited definition; distinguish an initialization diagnostic from a demonstrated training failure. |
| C11-03 | medium | `transformer-block.md:375-404`: “nothing measurable ... collapses” | §13.4 evidence; §16.2 qualifications near claims | Framework-specific, small-stack measurements are mixed with causal explanations about norm placement. | Separate identity/architecture statements from the observed timings and gradient measurements; add an explicit experimental-scope paragraph before the summary. |
| C11-04 | low | `transformer-block.md:847`: slide title “RMSNorm: drop what you don't need” | §19.2 slide titles; §14.3 | The title is a teaser and implies that mean subtraction is generally unnecessary. | Use “RMSNorm Omits Mean Centering” and state the conditions under which the comparison is made. |
| C11-05 | medium | `gpt.md:335`: “gradient descent has nothing left to learn ... except the book itself” | §17.7 anthropomorphism; §17.8 drama | The sentence substitutes wit for a precise diagnosis of memorization and validation degradation. | Name the observed training/validation divergence and connect it to corpus reuse and capacity. |
| C11-06 | low | `gpt.md:856`: slide title “Did it work?” | §14.2 vague questions; §19.2 | The title does not identify the validation operation. | Rename it “Validating the Loaded GPT-2 Checkpoint” and retain the perplexity and completion checks. |
| C11-07 | high | `kv-cache.md:405`: “pays off precisely when that term dominates”; `kv-cache.md:512`: “entire bill” | §8.4 ordinary verbs; §17.8 metaphor | Repeated economic metaphor obscures which term is compute, memory traffic, allocation, or latency. | Replace each occurrence with the resource and asymptotic term it denotes; keep “cost” only with units or a formula. |
| C11-08 | medium | `kv-cache.md:779-856`: “Cache Against Quality” experiment | §13.4 evidence; §16.1 claim category | A tiny character model is used to assess GQA quality, and “nothing we can measure” risks being read as no quality cost. | State sample size, seeds, uncertainty, and detection limit; phrase the result as no detectable difference in this run, not equivalence. |
| C11-09 | medium | `kv-cache.md:1104-1235`: low-rank compression, sliding windows, sinks | §6 one main question; §10 four levels | Several production ideas appear in rapid succession, but assumptions and implementation differences are compressed. | Give each method a uniform comparison contract: what dimension is reduced, what information is discarded, complexity, training requirement, and evidence source. |
| C11-10 | medium | `kv-cache.md:1316`: slide title “Generation recomputes everything”; `kv-cache.md:1344`: “First duty ... change nothing” | §19.2; §17.1 slogans | These are dramatic or normative slogans rather than content descriptions. | Rename to “Naive Autoregressive Generation Recomputes the Prefix” and “Cached and Uncached Logits Agree.” |
| C11-11 | low | `encoders-decoders.md:89`: “Predicting from Both Sides”; `encoders-decoders.md:245`: “What the Second Side Is Worth” | §14.1 descriptive titles | The headings are conversational and make the comparison less immediately recoverable. | Use “Bidirectional Context for Masked-Token Prediction” and “Loss by Available Context.” |
| C11-12 | medium | `encoders-decoders.md:787` caption | §12.2 self-contained captions | The caption states asymptotic costs but omits the number of latent self-attention layers; “all further processing” can be misread as one operation. | Include the latent depth or express cost per layer, and state which dimensions are held fixed. |
| C11-13 | high | `vision-transformer.md:16-20`: “When pretrained on 300 million images, it outperformed contemporary CNNs”; `vision-transformer.md:526-534`: “Scale flips the verdict” | §16 claim strength; §13.4 evidence | The narrative compresses dataset, pretraining recipe, architecture size, and evaluation protocol into a single causal claim about scale. | Attribute the exact comparison to the cited ViT setting; distinguish data scale from compute, augmentation, and pretraining. |
| C11-14 | high | `vision-transformer.md:539-542`: “the standard vision backbone has become the plain ViT” | §16.3 universal claims; §16.4 unresolved points | This time-sensitive field-wide statement lacks domain and date and conflicts with continuing use of hierarchical and convolutional backbones. | Qualify by application and model family, cite supporting surveys or deployment examples, or remove the ranking claim. |
| C11-15 | medium | `vision-transformer.md:576`: slide title “Images were CNN country”; `vision-transformer.md:649`: “The grid, barely” | §14.3; §19.2 | Literary titles manufacture a historical contest and do not identify the measured result. | Rename to “Convolutional Priors in Vision” and “Learned Position-Embedding Similarity.” |
| C11-16 | medium | `moe.md:597-609`: “collapse, caught red-handed”; “quieter half of the story” | §17.2 self-conscious narration; §17.8 drama | The analysis contains useful seed sensitivity, but the rhetorical framing distracts from utilization, loss, and variance. | Lead with the measured expert-load distribution and seed range; reserve “routing collapse” for the defined phenomenon. |
| C11-17 | low | `moe.md:768-783`: summary repeats “many narrow experts” | §17.10 repetition | The summary contains a literal duplicated phrase and mixes published facts with the chapter's experiments. | Remove duplication and split the summary into mechanism, experiment, and production evidence. |
| C11-18 | high | `scaling-laws.md:430-493`: miniature power law to published law | §9.6 exact vs intuition; §13.4 evidence | A five-size, fixed-data experiment is placed immediately beside a multi-variable published scaling law, inviting overgeneralization from the toy curve. | Explicitly state that the experiment demonstrates data limitation qualitatively and cannot estimate the published exponents or irreducible term. |
| C11-19 | medium | `scaling-laws.md:674-681`: “Chinchilla result suggesting about twenty tokens per parameter” | §16.2 qualifications; technical currency | The ratio is presented as a portable rule despite dependence on model, data quality, optimizer, and accounting conventions. | Present it as the cited study's setting and discuss why newer repeated-data and data-quality results change the allocation. |
| C11-20 | low | `scaling-laws.md:646`: “Where the Field Is Moving” | §14.1 titles; §16 current claims | The heading is vague and rapidly becomes stale. | Use a dated descriptive title such as “Reported Configurations, 2023–2025,” matching the table's scope. |

## Math and notation

- The chapter generally defines shapes well, especially in cache and MoE
  sections. Preserve the explicit head-count and head-width notation.
- `scaling-laws.md:100-136` should distinguish parameter FLOPs from attention
  score FLOPs before introducing the memorable (6ND) approximation. The
  current correction appears later and may be forgotten.
- `encoders-decoders.md:787` uses (M), (N), and (d) in a caption before
  restating all held-fixed quantities. Add layer count and batch assumptions.
- `kv-cache.md:1250-1265` gives a useful cache formula, but the summary should
  explicitly state whether (b) is bytes per element and whether the count is
  per sequence and excludes allocator overhead.
- `moe.md` switches among router probabilities, selected weights, loads, and
  bias controls. A compact notation table before the balancing objectives would
  reduce cognitive load.

## Figures, captions, and slides

Captions are a major strength: `transformer-block.md:75`,
`kv-cache.md:78`, and `moe.md:142` identify what varies and what the reader
should compare. Preserve that standard.

The slide deck is less disciplined than the prose. Teaser titles noted above,
the compressed slogan at `moe.md:845`, and “Recap” slides often replace an
argument with fragments. Give every recap one explicit relationship or
limitation, and use the same evidential qualifiers as the chapter prose.

## Code and experiment pedagogy

The configurable block/GPT interface aligns code with concepts well. Identity
checks for cache correctness and parameter mappings are exemplary. Weaknesses
are uneven reporting of seeds and uncertainty, framework-dependent claims
without a shared protocol, and occasional use of a tiny corpus to motivate
production conclusions. Each experiment should state before the code: the
hypothesis, controlled variables, seed count, and expected failure mode. After
the output, separate “implementation identity confirmed” from “empirical trend
observed.”

## Recurring artifacts

- Economic metaphors for compute and memory: “pays,” “bill,” and “price.”
- “Nothing/everything” as emphatic universal quantifiers.
- Slide titles framed as contests, surprises, or questions.
- Current-practice claims without a stable date and domain.
- Small teaching experiments placed too close to production-scale conclusions.

## Positive patterns to preserve

- One configurable implementation carries concepts across the chapter.
- Important captions are analytical and largely self-contained.
- Cache equivalence and checkpoint-loading checks use assertions rather than
  rhetorical claims.
- Summaries usually reconstruct mechanisms instead of listing terminology.
- Exercises ask for derivations, ablations, and interpretation rather than code
  transcription.

## Prioritized revision plan

1. **Calibrate broad claims.** Fix C11-13, C11-14, C11-18, and C11-19 so
   production, cited, and notebook evidence are visibly distinct.
2. **Reorganize scaling laws.** Separate invariant accounting from the dated
   configuration survey and scope the miniature experiment.
3. **Standardize cache/MoE comparisons.** Use common fields for resource reduced,
   assumptions, quality evidence, and training requirements.
4. **Rewrite slide titles and slogans.** Address C11-04, C11-06, C11-10,
   C11-15, and the MoE title strip.
5. **Perform a sentence-level restraint pass.** Replace economic metaphors and
   universal “nothing/everything” claims with explicit quantities.
6. **Finish with an evidence audit.** Add seeds, variability, and detection
   limits wherever a run supports an architectural conclusion.
