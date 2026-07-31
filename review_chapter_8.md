# Chapter 8 Style Review: Sequence Models

## Scope and files reviewed

Diagnosis only. I reviewed every tracked Markdown file in `chapter_recurrent-neural-networks`: `index.md`, `sequence.md`, `text-sequence.md`, `language-model.md`, `rnn.md`, `rnn-implementation.md`, `bptt.md`, and `decoding.md`. The review includes prose, headings, probability and gradient exposition, captions, all code/experiment interpretation, summaries, and all slide blocks.

## Executive assessment

This chapter has a clear running application and a mostly strong dependency chain from general sequences to tokenization, language models, recurrence, implementation, gradient propagation, and decoding. Its best passages use small computations to expose a limitation before introducing the next method. The chief problems are categorical statements where conditional ones are needed, a few genuine grammatical defects, generated-style slogans in transitions, and slippage between a mathematical interface and claims about all sequence models or all production systems. Slides often omit captions entirely even when the main text has a good self-contained caption.

## Scores (0–10)

| Dimension | Score | Rationale |
|---|---:|---|
| Writing quality | 6 | Usually energetic and clear, but universal slogans, theatrical transitions, and a handful of broken sentences remain. |
| Explanation quality | 7 | Concrete examples and operational code are strong; some major abstractions are introduced with imprecise assumptions. |
| Technical quality | 6 | Core algorithms are correct, but independence, hidden-state capacity, decoding objectives, tokenizer prevalence, and BPTT notation need tighter scope. |

## Architecture and logical order

The file order is one of the chapter’s strengths. `sequence.md` defines dependence and autoregression; `text-sequence.md` creates observations for the running language task; `language-model.md` defines its probability and metric; `rnn.md` removes the fixed-window limitation; `rnn-implementation.md` makes the recurrence operational; `bptt.md` analyzes its optimization pathology; `decoding.md` separates modeling from generation. At chapter level the index repeats this order twice: first as “two main ideas,” then as a section-by-section roadmap. Keep the conceptual two-idea map and shorten the catalogue. Within `sequence.md`, move the synthetic experiment immediately after the fixed-window objective and let the multistep failure motivate latent state.

## Section/file issue table

| ID | Severity | Evidence | Excerpt / description | Violated rule | Diagnosis | Concrete revision |
|---|---|---|---|---|---|---|
| C8-01 | H | `index.md:10–20` | “The first assumption was ... i.i.d.”; “every element depends”; “two examples rarely share” length | §§16.1–16.3: avoid universal claims | Earlier chapters did not require every real dataset to be i.i.d., many sequence elements are conditionally independent, and minibatched sequences are commonly padded to equal shapes. | State the modeling contrast conditionally: earlier objectives often treat examples as exchangeable fixed-shape records; sequence tasks retain within-example order and may have variable lengths. |
| C8-02 | M | `index.md:33–39` | Hidden state is “a fixed-size summary of everything it has read” | §§8.1, 16.1: literal mechanism and scoped claim | A finite vector is computed from the past but need not summarize every part of it; “everything” clashes with the immediately stated bottleneck. | Define (h_t=f(x_t,h_{t-1})) as a fixed-size function of the prefix and say training determines which information it retains. |
| C8-03 | M | `sequence.md:54–73` | “Sequences are everywhere once you look”; “word ... depends on all words”; “If entries were unrelated there would be nothing to predict” | §§17.1, 17.9, 16.3 | The passage uses slogans and necessity claims instead of separating order, dependence, variable length, and nonstationarity. Independent symbols can still form a sequence and permit marginal prediction. | Use one concrete example, then list the exact properties modeled. Replace “nothing to predict” with “conditioning on neighboring entries would give no advantage.” |
| C8-04 | M | `sequence.md:229` and heading `sequence.md:450` | “Enough theory; let us fit ...”; “Why This Matters Everywhere” | §§14.2, 15.2, 17.8 | The transition is self-conscious and the heading is a vague universal teaser. | Replace with the dependency: “The fixed-window model now permits a controlled test of one-step and recursive prediction.” Retitle “Error Accumulation in Autoregressive Rollouts.” |
| C8-05 | H | `text-sequence.md:18–22` | BPE is “behind essentially every production tokenizer”; “implementation ... reproduce” | §§8.2, 16.1–16.3 | The sentence has subject–verb disagreement and overstates BPE prevalence (WordPiece, Unigram, byte-level and learned tokenizers exist). | Write “many widely deployed tokenizers use BPE or related subword schemes”; correct “reproduces”; name the GPT-2 compatibility scope. |
| C8-06 | H | `text-sequence.md:632–658` | “Everything a deployed tokenizer knows fits in one table”; GPT-2 artifact “entire” | §§16.3, 17.1 | The claim ignores normalization, pre-tokenization, special-token policy, byte mapping, added vocabulary, and model metadata—even the next sentence names a regex separately. | Scope the statement specifically to GPT-2’s mergeable-rank table plus byte mapping, regex, and special tokens; contrast with tokenizers that include normalization/model components. |
| C8-07 | M | `language-model.md:22–26` | “reduces to the conditional prediction problem estimate ... Every language model ... exactly” | §§8.2, 16.1 | The sentence is grammatically incomplete. “Exactly” blurs normalized autoregressive LMs with masked, energy-based, diffusion, or other language models. | Insert “of estimating”; scope to autoregressive language models in this book; distinguish exact chain-rule factorization from approximate parameterization. |
| C8-08 | H | `rnn.md:9–18` | (P(x_t\mid x_{<t})\approx P(x_t\mid h_{t-1})), then “not an approximation at all” if (f) stores everything | §§9.1–9.2, 9.6, 16.1 | The condition depends on an injective/unbounded-precision state and an output map capable of recovering the conditional, not merely “sufficiently powerful (f).” It also conflicts with the fixed-size motivation. | State the sufficiency condition (P(x_t\mid x_{<t})=P(x_t\mid h_{t-1})); explain that finite practical states generally impose an information bottleneck and that universal real-valued encoding is not operationally useful. |
| C8-09 | H | `bptt.md:61–85`, then `bptt.md:134` | Notation “does not distinguish scalars, vectors, and matrices”; “Everything now hinges ...” | §§9.2, 9.10, 17.1 | Calling this the “shape” while suppressing shapes makes derivative products ambiguous; (\partial h_t/\partial w_h) changes rank for vector/matrix objects. The slogan overstates one factor before loss/output Jacobians are discussed. | Either use scalar notation explicitly as a scalar warm-up or give vector dimensions and Jacobian shapes. Replace the slogan with the exact numerical issue: repeated state Jacobians can contract or amplify components. |
| C8-10 | H | `decoding.md:68–100` | Atom comparison; argmax “exactly right” for translation/transcription; sampling/maximization split “organize the whole section” | §§8.3, 16.1–16.3, 17.8 | Search-space size is sufficient without a cosmic analogy. Sequence-level probability does not make argmax uniquely appropriate for tasks evaluated by utility, calibration, or multiple references; beam search can prefer length-biased outputs. | State complexity with vocabulary/horizon only; distinguish model MAP decoding from task-optimal decision rules; present maximization and sampling as useful families, not exhaustive task laws. |
| C8-11 | M | slide `text-sequence.md:978` and several slides at `sequence.md:531`, `decoding.md:714`, `decoding.md:762` | Empty image captions `![](...)` | §§12.2, 19.5 | The deck discards the self-contained captions present in the main text, so the audience is not told what to compare. | Add one-sentence slide captions stating the comparison and conclusion; keep them shorter than main-text captions. |

## Math and notation

- Distinguish exact chain-rule identities from model approximations throughout `language-model.md` and `rnn.md`.
- Define whether (x_t) is a token, token id, one-hot vector, or embedding at each transition; current files reuse (x_t) across these levels.
- Repair BPTT shapes (C8-09). A scalar warm-up followed by a matrix case is acceptable if the shift is labeled.
- `decoding.md` should specify whether scores include EOS and length normalization before comparing sequence probabilities.
- `language-model.md` introduces perplexity well, but “surprise” should be connected immediately to average negative log likelihood and units; bits per byte must keep log base explicit.

## Figures, captions, and slides

Main-text figures are generally excellent: captions name the transformation and relevant limitation, especially the tokenization spectrum, partitioning diagram, unrolled RNN, and truncated BPTT. Slide copies frequently replace these with empty captions or shortened descriptions that lose the conclusion (C8-11). The decoding task map also encodes a binary task taxonomy more strongly than the prose should claim; redraw or caption it as common tendencies rather than a rule.

## Code and experiment pedagogy

`rnn-implementation.md` is strong: it states inputs/outputs, maps scratch and concise implementations, reports framework-specific performance, and acknowledges run-to-run movement in TensorFlow. Keep this as the chapter model. Improve reproducibility by stating seeds for generated samples and multiple training runs where the prose says behavior is typical. `text-sequence.md` uses private `tiktoken` fields; explain that the inspection is version-sensitive and not a stable API. `decoding.md` compares samples qualitatively; explicitly label such output as illustrative rather than evidence of general decoder quality.

## Recurring artifacts

- Universal quantifiers: “every,” “everything,” “any,” “exactly right.”
- Dramatic transition phrases: “Enough theory,” “Everything now hinges,” “staggering,” atom comparisons.
- Anthropomorphic shorthand: models “judge,” “write,” “remember,” or “know” without local operational definitions.
- Missing slide captions despite strong source captions.
- Scope drift from autoregressive models/tokenizers used here to all language models or deployed tokenizers.

## What already works

- The chapter uses one corpus and one task to connect abstractions operationally.
- `decoding.md:4–12` clearly separates a probability model from a token-selection rule.
- `rnn-implementation.md:621–633` explains truncation, state carry, and detachment before code.
- `bptt.md` gives a concrete linear recurrence after the general graph and connects eigenstructure to numerical behavior.
- Most main-text captions are self-contained and state the comparison readers should make.

## Prioritized revision plan

1. Fix C8-05 through C8-10: scope and formal correctness first.
2. Rewrite the index and early `sequence.md` claims around precise modeling assumptions rather than universals.
3. Normalize the token/token-id/vector/state notation across files.
4. Add interpretive captions to all slide figures and weaken the decoding taxonomy.
5. Remove theatrical transitions and cosmic comparisons while preserving the useful quantitative examples.
6. Retain `rnn-implementation.md`’s code-to-concept pattern and apply it to tokenizer and decoding experiments.
