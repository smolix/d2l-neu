# Chapter 9 Style Review: Optimization Algorithms

## Scope and files reviewed

Diagnosis only. I reviewed every tracked Markdown file in `chapter_optimization`: `index.md`, `optimization-intro.md`, `gd.md`, `sgd.md`, `minibatch-sgd.md`, `momentum.md`, `adam.md`, `adamw.md`, `lr-scheduler.md`, `muon.md`, `batch-size.md`, `scaling.md`, and `practice.md`. The review includes prose, derivations and assumptions, empirical comparisons, captions/tables, summaries, code explanations, and all slide blocks.

## Executive assessment

The chapter has a strong organizing idea—direction, step size, and gradient noise—and unusually good continuity across experiments. Many sections explicitly distinguish proof, small controlled experiment, and unsettled production evidence. Remaining defects are concentrated but important: the index is syntactically broken and overpromotional; several tiny-testbed observations are generalized to transformers or “practice”; current hardware and production claims lack enough conditions; and a few generated-writing slogans remain. Slides often drop the informative captions of the main text.

## Scores (0–10)

| Dimension | Score | Rationale |
|---|---:|---|
| Writing quality | 7 | Strong technical voice overall; slogans, long roadmaps, and promotional frontier language remain. |
| Explanation quality | 8 | Concrete quadratics and matched sweeps usually connect formal and operational levels well. |
| Technical quality | 7 | Derivations are largely sound, but several empirical and “default practice” claims exceed their protocols. |

## Architecture and logical order

The sequence from landscape to deterministic/stochastic descent, batching, momentum, and adaptive methods is coherent. Adam introduces a shared transformer testbed that supports the applied second half. AdamW and schedules should precede Muon; batch size and scaling then explain how choices change with compute; practice synthesizes them. The index currently turns this into two very long catalogues. Replace those with the three decisions plus a short dependency map. The mathematical appendix bridge is valuable but its sentence is grammatically malformed.

## Section/file issue table

| ID | Severity | Evidence | Excerpt / description | Violated rule | Diagnosis | Concrete revision |
|---|---|---|---|---|---|---|
| C9-01 | H | `index.md:45–51` | “develops ... and the convex analysis underneath them are developed” | §§7–8: grammatical, one coherent move | The subject/predicate duplicates “develops,” and the long list obscures what this chapter assumes versus defers. | Rewrite as two sentences: the main chapter develops empirical intuition; the appendix proves named results under stated assumptions. |
| C9-02 | M | `index.md:30–43`, `53–62` | “credible challenger to Adam’s decade,” “younger than it looks,” rapid Muon story | §§5.1, 8.4, 17.4: roadmap after need; avoid promotion | The index mixes dependency, current news, and editorial verdicts before readers see the methods. “Credible” and “celebrated” make the author’s evaluation visible. | Keep a concise method dependency map; move evidence status and chronology to `muon.md`/`practice.md`, expressed with concrete benchmarks. |
| C9-03 | M | `gd.md:39–42` | “Everything interesting hides in the word ‘should’” | §§17.1–17.2: no grand slogan or self-conscious narration | The sentence advertises a caveat rather than naming the missing smoothness/step-size condition. | State that the Taylor argument is local and that a global descent guarantee requires an (L)-Lipschitz gradient and `0<\eta<2/L` (or defer the exact bound explicitly). |
| C9-04 | H | `adam.md:10–11` | Adam “has been the default optimizer of deep learning for a decade” | §16.1–16.3: avoid universal claims from local practice | SGD with momentum remained standard in vision for much of the period, while Adam/AdamW dominated language and transformers. | Scope by domain and date: Adam-family methods became common defaults for transformer training; SGD remained competitive/default in several vision regimes. |
| C9-05 | H | `adam.md:655–660` | Tiny sweep makes SGD’s “knife-edge characteristic of SGD on transformers,” attributed to heavy-tailed gradients | §§13.4, 16.3: finite experiment does not establish a general mechanism | Four learning rates on one tiny model do not diagnose gradient tails or characterize all transformers. | Describe the observed grid and instability only; separately cite evidence for heavy tails and label it a possible explanation, or measure the gradient distribution here. |
| C9-06 | H | `adamw.md:4–6` | AdamW uses “typically `\lambda=0.1`” on most parameters | §§16.2, 16.3: qualifications near claim | The value depends on task, LR convention, training length, batch, parameter group, and implementation; 0.1 is common in some language-model recipes, not AdamW’s definition. | Define AdamW without a typical value in the opening. Discuss 0.1 later as a reported recipe with examples and effective shrinkage (\prod_t(1-\eta_t\lambda)). |
| C9-07 | M | `minibatch-sgd.md:19–34` | Current CPU/GPU FLOP and bandwidth orders, “roughly two orders” | §§13.4, 16.1: evidence and temporal scope | The numbers age quickly, mix precision modes, and have no named device or source. “Current server” is not reproducible. | Use a concrete CPU/GPU example with year, dtype, theoretical/achieved distinction, and citation; make arithmetic intensity, not headline hardware, the durable lesson. |
| C9-08 | M | `lr-scheduler.md:451–453`, `548–605` | “The catch”; “shrugs off”; “magic trick”; one run “reproducible across restarts” | §§13.3–13.4, 17.8 | The prose retains generated-style drama and reports reproducibility without seed count/error range. The attribution of chance performance to parameters being “destroyed” is inferential. | Replace with direct limitation language; report number of restarts and range; describe the loss/accuracy evidence and mark the parameter-displacement mechanism as hypothesis unless measured. |
| C9-09 | H | `muon.md:11–18`, `90–102` | Spectral norm is “the natural way” to measure a matrix; sign descent “recovers, in essence, Adam” | §§9.6, 16.1: exact versus interpretation | Spectral norm controls worst-case activation change, which makes it one motivated geometry, not uniquely natural. Adam also has moment estimation, coordinate scales, bias correction, and `\epsilon`; sign descent is only a limiting directional analogy. | Say which optimization model each norm solves exactly, then label the Adam relation as an interpretation/limit and state what is omitted. |
| C9-10 | H | `practice.md:47–70` | Four incomplete reports yield “consensus core”; unstated fields treated as defaults; “state of the art in one table” | §§13.4, 16.1–16.4 | Two rows supply betas/clipping/decay, one row uses Muon, and dashes are missing evidence—not evidence of hidden defaults. The conclusion overgeneralizes a selected sample. | Call these four case studies; calculate each conclusion only from disclosed cells; separate “reported in 2/4” from common external defaults and add selection limitations. |
| C9-11 | M | `practice.md:181–198`, `493–498` | “first steps tell the story,” “heart of the matter,” “Everything ... somehow” | §§17.1–17.2, 17.6 | Slogans replace the actual evidence and make the author’s staging visible. | Name the evidence directly: clipping changes six of 2,000 updates and prevents overflow in this run; tuning choices must be recorded and nuisance variables retuned. |
| C9-12 | M | slide figures `optimization-intro.md:422–452` | Five figures have empty captions | §§12.2, 19.5 | Main-text captions contain the mathematical conclusion, but slide copies provide no comparison instruction. | Restore compact interpretive captions (“stationary inflection: zero first and second derivative, no extremum,” etc.). |

## Math and notation

- In `gd.md`, replace the informal `\lessapprox` conclusion with a precisely labeled first-order approximation or the descent lemma under (L)-smoothness.
- In `adamw.md`, distinguish the schematic preconditioner display from the actual coupled moment recursion; the existing parenthesis helps, but the equation should be explicitly labeled schematic before it appears.
- In `muon.md`, retain the constrained-ball versus quadratic-regularized distinction when calling a solution the “SGD direction.”
- In `batch-size.md`, state with-replacement sampling assumptions adjacent to the `1/b` variance law and note finite-population correction without replacement.
- In `scaling.md`, identify which muP variant and parameter-group rules are implemented; “maximal update parametrization” is not one framework-independent three-line recipe.

## Figures, captions, and slides

Main-text captions are mostly excellent, especially the critical-damping and norm-ball figures: they state variables and the comparison. The river-valley figure is explicitly a picture/analogy but risks carrying a causal inference; keep the formal warmup evidence separate. Slide copies of optimization landscape figures omit captions entirely (C9-12). Slide titles are generally descriptive, although “Why it moves” in scaling and similar vague headings should name the width-dependent update being explained.

## Code and experiment pedagogy

Matched sweeps and fixed testbeds are a major strength. Tighten the claim boundary: a sweep demonstrates behavior for its grid, model, seed set, and budget, not “SGD on transformers” (C9-05). Whenever prose says “reproducible,” give restart count and variability. Hardware timing should include device, dtype, warmup/synchronization, and whether memory is allocated or reserved. Production tables should separate disclosed values from inferred conventions.

## Recurring artifacts

- “Everything,” “the story,” “heart of the matter,” “the catch,” “magic trick.”
- Frontier/news narration and editorial rankings in an opening roadmap.
- Domain-wide defaults inferred from transformer-language-model recipes.
- Mechanistic causes inferred from a single loss curve.
- Empty slide captions that discard the main text’s strongest explanations.

## What already works

- The direction/step/noise framework gives the chapter a recoverable main question.
- `optimization-intro.md` distinguishes training objective from population risk early.
- `momentum.md` connects a two-dimensional trajectory to damping and condition number.
- `batch-size.md` poses a quantitative “steps to target” experiment rather than relying on throughput alone.
- `practice.md` explicitly warns that reported recipes are not ablations; keep that restraint and make the subsequent synthesis match it.

## Prioritized revision plan

1. Correct C9-01, C9-04 through C9-06, C9-09, and C9-10.
2. Reframe all empirical generalizations with protocol, seeds, and uncertainty.
3. Rewrite the index as a concise dependency map and remove current-news promotion.
4. Replace remaining slogans with exact mechanisms or observations.
5. Normalize assumptions and notation in GD, AdamW, batch-size, Muon, and muP.
6. Restore interpretive captions in the slide deck and audit slide claims against prose.
