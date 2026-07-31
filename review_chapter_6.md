# Chapter 6 Style Review: Convolutional Neural Networks

## Scope and files reviewed

Diagnosis only. I reviewed every tracked Markdown file in `chapter_convolutional-neural-networks`: `index.md`, `why-conv.md`, `conv-layer.md`, `padding-and-strides.md`, `channels.md`, `pooling.md`, and `lenet.md`. The review includes chapter and section openings, derivations, notation, code setup and output interpretation, captions, summaries, exercises where they affect exposition, and every block after `<!-- slides -->`.

## Executive assessment

The chapter has a sound dependency chain: spatial structure motivates locality and translation equivariance; those assumptions produce convolution; padding, stride, channels, and pooling complete the operator; LeNet assembles the pieces. Several recently written openings and captions already follow the style guide well. The main prose is nevertheless uneven. Older passages are wordy or conversational, several conclusions overstate what the derivation proves, and the slide deck sometimes contradicts the chapter it is meant to summarize. The largest revision need is consistency across prose and slides, followed by sharper technical qualification of equivariance, pooling, and expressive-power claims.

## Scores (0–10)

| Dimension | Score | Rationale |
|---|---:|---|
| Writing quality | 6 | Generally readable, but promotional adjectives, jokes, awkward sentences, and slide rhetoric interrupt the calm technical voice. |
| Explanation quality | 7 | The main progression and examples are effective; several captions, summaries, and code passages do not complete the interpretive step. |
| Technical quality | 7 | Core mathematics is sound, but claims about expressivity, padding, pooling, and current practice require narrower conditions. |

## Architecture and logical order

The chapter-level order is appropriate. `why-conv.md` supplies the structural assumptions before `conv-layer.md` defines the operation. `padding-and-strides.md`, `channels.md`, and `pooling.md` extend the primitive before `lenet.md` composes a network. The index should present this dependency rather than a catalogue of applications. Within `why-conv.md`, the ending anticipates channels before the intervening convolution mechanics and reads partly as a second roadmap. Within `channels.md`, grouped/depthwise convolution is introduced before the cost argument that most clearly motivates it; move the dense-convolution cost calculation immediately before the factorization. The `lenet.md` ending is a useful bridge to Chapter 7.

## Section/file issue table

| ID | Severity | Evidence | Excerpt / description | Violated style-guide rule | Diagnosis | Concrete revision |
|---|---|---|---|---|---|---|
| C6-01 | M | `index.md:17` | “a powerful family ... designed for precisely this purpose” | §§8.4, 17.4: avoid promotional adjectives; state the mechanism | “Powerful” adds no information after a concrete spatial problem has already been established. | Replace with a direct claim: CNNs preserve spatial organization through local connectivity and shared weights. |
| C6-02 | M | `index.md:29–42` | Broad catalogue from biology and group theory through audio, text, time series, graphs, and recommenders | §§4.3, 5.1, 15.3: opening should establish scope without over-signposting | The catalogue delays the chapter’s actual dependency and makes several current-practice claims without saying under what constraints CNNs are preferred. | Condense to one scoped paragraph about parameter sharing and parallel computation; move non-image applications to further reading or the summary. |
| C6-03 | M | `why-conv.md:16–20` | “Unless we have lots of GPUs, a talent for distributed optimization, and an extraordinary amount of patience” | §§8.1–8.4, 17.8: concrete subjects and restrained prose | The joke interrupts a useful numerical example and substitutes personality for the computational conclusion. | State the memory/compute implication of (10^9) parameters, preferably with bytes or FLOPs under a stated dtype. |
| C6-04 | H | `why-conv.md:322` | “reduce the number of parameters ... without limiting its expressive power, at least whenever certain assumptions ... hold” | §§9.6, 16.1–16.3: separate exact statements from intuition and scope claims locally | Locality and translation equivariance do restrict the function class; the vague trailing caveat does not identify the functions for which no useful expressivity is lost. | Say that the restrictions preserve the desired class only when the target mapping is local and translation equivariant; give a counterexample such as an absolute-position-dependent label. |
| C6-05 | M | `channels.md:9–24` | A long recap followed by “channels is as old as CNNs themselves” and “we will take a deeper look” | §§6.1, 15.1–15.3, 17.9: section opening should state the unresolved issue | History and generic signposting obscure the precise need: one filter must combine all input channels and a layer must produce many learned features. | Open with the shape mismatch left by the single-channel operator, then introduce (c_i) and (c_o) and their kernel shape. Move the LeNet history to a later note or remove it. |
| C6-06 | M | `pooling.md:367–370` | “appears at first glance to be different, however numerically the same results are presented as ...” | §§7, 8.2, 8.9: one clear logical move per sentence | The sentence is ungrammatical and does not name the actual layout convention. “Reading vertically” is not a tensor-shape explanation. | State both layouts explicitly (for example NCHW versus NHWC), show the shape permutation, and say that values agree after transposition. |
| C6-07 | M | `conv-layer.md:661` | Caption begins “Figure and caption taken from ...” and contains Left/Right/Middle instructions plus sampling details | §§12.2–12.3: caption identifies the comparison; argument stays in prose | The caption is an imported mini-essay, orders panels confusingly, and does not state what the reader should infer for CNN receptive fields. | Give attribution in one clause, identify panels in visual order, and state the relevant comparison. Move sensor-sampling details to prose. |
| C6-08 | M | `lenet.md:31–37` | “outstanding results ... To this day, some ATMs still run the code ...!” | §§8.3, 16.1, 17.4: qualify claims and avoid exclamatory promotion | The historical anecdote is vivid but uncited and stronger than the supplied evidence; it also delays the architecture. | Cite a source that supports deployment longevity and remove the exclamation, or reduce to the documented check-reading deployment in the 1990s. |
| C6-09 | H | `pooling.md:383` versus slide `pooling.md:454–457` | Prose: modern networks mainly downsample with strided convolution. Slide: “Max is the default in modern nets” | §20.10 and §19: claims must remain consistent across formats | The deck teaches the opposite of the section summary and confuses “default pooling reduction” with “default downsampling layer.” | Revise the slide to distinguish local max-pooling in older/stem designs, global average pooling in heads, and strided convolution as common learned downsampling. |
| C6-10 | H | slide `padding-and-strides.md:486–510` | “A plain convolution always shrinks”; “padding ... fight the shrink”; “Padding fixes ... underweighting” | §§16.1–16.3, 17.7–17.8, 19.2 | The universal claim ignores (1\times1), padded, circular, and other boundary conventions. “Underweighting” is not generally fixed by zeros, which introduce their own boundary effect. | State the exact unpadded shape condition, replace metaphors with the two controlled quantities, and qualify zero padding as increasing boundary participation rather than equalizing information. |

## Math and notation

- `why-conv.md` usefully derives the structured weight tensor, but the summary must distinguish restricting a function class from preserving functions that satisfy the stated symmetries (C6-04).
- `channels.md` should introduce the complete kernel shape (c_o\times c_i\times k_h\times k_w) before moving among per-channel equations and framework layouts. State once that TensorFlow uses channels-last while the mathematical convention is channels-first.
- `padding-and-strides.md` uses (p_h,p_w) as total padding in equations while libraries often accept per-side padding. Keep the distinction adjacent to each formula, including slides.
- `pooling.md` needs one explicit statement that max-pooling is nonlinear and average pooling is linear; this would connect the exercises to the operator definition.
- Major equations are usually followed by an operational reading. Preserve that strength when revising prose.

## Figures, captions, and slides

Main-text captions are usually specific and self-contained, especially the numerical captions in `channels.md` and `padding-and-strides.md`. The Field figure is the principal exception (C6-07). Slide captions are shorter, but several slides use marketing syntax (“at a glance”), bold emphasis, em dashes, and categorical claims where the chapter itself is qualified. The deck also duplicates figures without always preserving the prose’s limitations. Audit all slides for the same boundary assumptions, layout conventions, and modern-practice claims used in the chapter.

## Code and experiment pedagogy

The framework tabs make cross-library shape conventions valuable, but explanations should map conventions rather than say outputs merely “look different” (C6-06). Repeated code comments in `padding-and-strides.md:133–198` narrate helper behavior four times; explain the helper once in prose, then reserve tab comments for framework-specific differences. `conv-layer.md` gives the learned edge kernel a concrete target and interprets convergence well. `lenet.md` should report the observed train/validation behavior with the run’s seed/variability instead of relying on a single curve as evidence for architecture-wide claims.

## Recurring artifacts

- Promotional modifiers: “powerful,” “outstanding,” “critical.”
- Conversational fillers: “Note that,” “at first glance,” “we will take a deeper look.”
- Slide rhetoric: “fight,” “lean into it,” “at a glance,” categorical “default” claims.
- Long single-paragraph summaries that mix result, historical aside, next-chapter roadmap, and current-practice survey.

## What already works

- `conv-layer.md:8–11` is an excellent dependency-based opening.
- `padding-and-strides.md:17–29` states concrete spatial consequences before naming the three controls.
- `why-conv.md` begins with a quantitative parameter-count problem and develops locality/equivariance from it.
- Many captions compute the exact highlighted output rather than saying only what a figure “shows.”
- `lenet.md` closes with a useful comparison table whose third column gives the operational reason for each modernization.

## Prioritized revision plan

1. Resolve C6-04, C6-09, and C6-10 so prose, slides, and technical qualifications agree.
2. Rewrite the index and section openings around the concrete unresolved problem, removing catalogues and generic roadmaps.
3. Normalize notation and framework layout explanations across `channels.md`, `pooling.md`, and `padding-and-strides.md`.
4. Rewrite the imported/underspecified captions and audit every slide for the main text’s assumptions.
5. Remove promotional and conversational artifacts; split overloaded summaries into result, limitation, and bridge.
6. Re-read code commentary and output interpretation, retaining only conceptual purpose and framework-specific differences.
