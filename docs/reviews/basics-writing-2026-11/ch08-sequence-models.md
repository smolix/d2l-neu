# Chapter 8: Sequence Models

Sources: `chapter_recurrent-neural-networks/` (index, sequence, text-sequence, language-model, rnn, rnn-implementation, bptt, decoding), read in full up to the slides marker, with the stored PyTorch outputs and the rendered figures. About 24,000 words of prose (including the 870-word resource list of the introduction) and 5,750 words of exercises.

Per-notebook reviews:
[index](ch08-sequence-models/index.md),
[sequence](ch08-sequence-models/sequence.md),
[text-sequence](ch08-sequence-models/text-sequence.md),
[language-model](ch08-sequence-models/language-model.md),
[rnn](ch08-sequence-models/rnn.md),
[rnn-implementation](ch08-sequence-models/rnn-implementation.md),
[bptt](ch08-sequence-models/bptt.md),
[decoding](ch08-sequence-models/decoding.md).

## Verdict

This is a good chapter, and in three places (the chain-rule passage of `sequence.md`, the perplexity and bits-per-byte subsections of `language-model.md`, the opening and the clipping argument of `bptt.md`) it reaches the bar the authors set. The language is the cleanest I would expect to find in the part: plain declarative sentences, no dashes, almost no stock phrases, numbers quoted at the precision the stored outputs support, and I found no number in the prose that a stored PyTorch output contradicts. One corpus, one tokenizer and one metric run through all seven sections, so results in different sections can be compared, and they are.

What separates it from Cover and Thomas or MacKay is not the sentences. It is that several experiments do not show the thing the surrounding prose says, and the text then states the lesson anyway: error accumulation in `sequence.md`, the need for clipping and the usefulness of the hidden state in `rnn-implementation.md`, the special virtue of min-$p$ in `decoding.md`, and a spectral-radius experiment in `bptt.md` that confirms an identity. In each case the evidence that would support the claim sits in an exercise. The second gap is `rnn-implementation.md`, which carries six jobs and a research detour, and is the one section a reader will find long. Fixing these is mostly a matter of moving four experiments from exercises into the text and cutting about a thousand words of verification and plumbing.

## Scores

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 4 |
| sequence | 4 | 3 | 4 | 4 | 4 |
| text-sequence | 4 | 4 | 4 | 4 | 4 |
| language-model | 4 | 4 | 3 | 4 | 4 |
| rnn | 4 | 4 | 4 | 4 | 3 |
| rnn-implementation | 3 | 3 | 3 | 4 | 3 |
| bptt | 4 | 4 | 4 | 4 | 4 |
| decoding | 4 | 3 | 4 | 4 | 3 |

No 5s: each section has at least one place where a claim outruns its evidence or the order works against the reader. `bptt.md` and the second half of `language-model.md` are the nearest.

## Arc and order

The arc is sound: predict a sequence from its past (sequence), decide what the units are (text-sequence), estimate by counting and learn to measure (language-model), replace the window by a state (rnn), build and train it (rnn-implementation), understand its gradients (bptt), turn it into text (decoding). Each section opens from the previous one and closes with what is still open. That is rare and should be preserved.

Three order problems:

1. **Inside `language-model.md`** (LM-S1). Smoothing is derived before perplexity exists, and Zipf's law comes after the sparsity it explains. Reordering to: counting, sparsity with Zipf, perplexity, Laplace (evaluated), Kneser-Ney (evaluated), bits per byte, pipeline, would let every method answer a number the reader has just seen.

2. **`rnn-implementation.md` before `bptt.md`.** The implementation needs clipping, truncation and detaching, introduces each with a forward reference ("`sec_bptt` analyzes both cases", "`sec_bptt` analyzes what it gives up"), and `bptt.md` then explains each again (the clipping formula twice in the chapter, the truncate, carry, detach paragraph twice, nearly verbatim: IMP-11). `rnn.md` ends "One question remains before we train it" and points to `bptt.md`, as if that section came next (RNN-5); the chapter introduction lists the other order (IDX-3). And "Text Generation" in the implementation is separated from "Decoding" by the gradient analysis. I recommend moving `bptt.md` before `rnn-implementation.md`: theory (rnn, bptt), then practice (implementation, decoding). The opening of `bptt.md` does not need the trained model; its scalar example stands alone, and it would lose only the sentence "trained only because we clipped", which is unsupported as it stands (BPTT-1). The implementation could then use clipping and truncation as known tools and spend its experiment on measuring them (gradient norms, the gradient against lag). If the authors prefer to keep practice before theory, so that the reader meets the problem before its analysis, then the implementation must show the problem (IMP-7) instead of asserting it, and the duplicated paragraphs must go.

3. **The classical character and word pipeline inside `text-sequence.md`** sits after BPE has replaced it (see the structural finding there).

On proportion: sections run from 2,300 to 4,400 words of prose. `text-sequence.md` and `language-model.md` are long but their length is content. `rnn-implementation.md` is long because it has too many jobs.

## The chapter-specific question

*Does the chapter give a clean mental model of autoregressive sequence modelling, perplexity, the recurrent state, and gradient pathologies, without leaning forward?*

**Autoregressive modelling: yes.** "Thus a model of the next entry is a model of the entire sequence", the two uses (score, generate), teacher forcing against generation, and the stateless `step_fn` of `decoding.md` ("Any model that predicts next tokens can fill this interface") add up to exactly the picture the Advanced part and the Language Models part need. Two blemishes: `sequence.md` reserves "autoregression" for the window model (SEQ-3), and its rollout experiment teaches a slightly different lesson from the one it announces (SEQ-1).

**Perplexity: yes, and better than that.** Motivation (*War and Peace* against *The Little Prince*), definition, the branching-factor reading, three reference values, a result that overturns the reader's intuition (the fluent trigram is the worst model), and then bits per byte as the tokenizer-independent form, re-taught at the moment it is needed in the implementation. This is the part of the chapter later parts will lean on most and it will bear the weight.

**The recurrent state: defined cleanly, demonstrated weakly.** `rnn.md` says what the state is, how it differs from a hidden layer, that it is lossy, and what it costs (512 bytes, whatever the prefix). What the chapter never shows is a state doing something a window cannot. The constructions that show it (parity; a shift register inside a recurrence) are exercises. The trained model ties a trigram, its embedding table is close to a random projection, and no plot shows the loss improving with the amount of context. A reader could finish the chapter believing the state is a fixed-size summary without having seen it summarize anything. Since the Advanced part's chapters on attention and state space models are organized around what a fixed-size state can and cannot hold, this is the gap most worth closing (IMP-S2: plot loss against position in the window; promote parity into `rnn.md`).

**Gradient pathologies: the linear theory is clean; the link to the real model is left to exercises.** The scalar example, the sum over lags, $\rho^k$, why clipping treats explosion and cannot treat vanishing, and what truncation removes are all there and all well put. Missing from the text: what the $\tanh$ changes (BPTT-4), any measurement on the trained model (BPTT-3), and any evidence that the model's gradients did explode (IMP-7).

**Leaning forward: mostly well judged, with one exception.** The pointers to attention (`rnn.md`), to gated cells and linear recurrences (`bptt.md`) and to state space models (`sequence.md`, introduction) each describe the later idea by the one thing it changes and carry no part of the present argument. That is how forward references should be written. The exception is Adam. `rnn-implementation.md` diagnoses that SGD leaves the embeddings untrained, adopts a workaround, and says twice that the remedy is the optimizer of `sec_adam` in the next part ("as an exercise shows with Adam"; "Trained with Adam ... the RNN alone also edges past the trigram; an exercise asks you to try it"). Here the chapter's main experiment depends on a tool the part has not introduced (IMP-13). Two smaller leans: "spectral radius" is used one section before it is defined (IMP-2), and two sections name `sec_machine_translation`, which now sits in the Attic, as the consumer of special tokens and beam search (TOK-10, DEC-10).

*Is the balance between `rnn.md` and `rnn-implementation.md` right?*

By source lines the ratio is three to one, but that is four framework tabs of code. By prose the PyTorch reader sees about 2,350 words against about 3,800, a ratio of 1.6. An implementation section that also trains, evaluates and generates may fairly be longer than the definition. So the ratio is not the problem; the allocation is. `rnn.md` is a little too thin on the one idea that matters (what a state can do) and spends 200 words checking a block-matrix identity (RNN-2). `rnn-implementation.md` is too thick with things that are neither implementation nor the experiment: a third telling of one-hot against lookup (IMP-4), shape helpers (IMP-3), the embedding diagnosis (IMP-13), a first treatment of temperature that `decoding.md` repeats (IMP-16), and a framework timing study (IMP-18). After the cuts listed under IMP-S1 it would be about 2,900 words, and with the two measurements added (gradient norms, loss by position) about 3,200, which is proportionate.

## The most important findings

1. **Experiments that do not show their stated lesson.** SEQ-1 (rollout: the model damps, the text concludes accumulation), IMP-7 (clipping: gradient spikes described with numbers that no cell prints), IMP-S2 (no evidence the state uses context), BPTT-3 (the experiment plots $\rho^k$ for a matrix whose power has norm $\rho^k$), DEC-6 and DEC-8 (min-$p$ credited with a property top-$p$ shows equally; "truncated samplers sit in between" not visible in the samples). In each case a short measurement would settle it, and in four of the five that measurement already exists as an exercise.
2. **`rnn-implementation.md` has no single idea.** IMP-S1, IMP-13. Decide what its experiment is for, cut the rest.
3. **Order within `language-model.md`.** LM-S1. Perplexity before smoothing; Zipf with sparsity.
4. **Order of implementation and BPTT, and the repetition it causes.** RNN-5, IMP-11, IDX-3.
5. **The embedding promise is dropped.** `language-model.md` ends by saying that learned representations are where neural models win ("cat" and "feline"); `rnn.md` introduces the embedding as a fix for arbitrary ids (RNN-4); `rnn-implementation.md` finds the embeddings barely train (IMP-13). The reader never sees a learned representation.
6. **The word-level baseline knows the held-out vocabulary.** LM-4. The best row of the bits-per-byte table is disqualified by the sentence after it.
7. **Teaching code optimized for speed.** TOK-2 (`_merge` and `train`), with forward-declared arguments in the first class cell (TOK-3, IMP-5).
8. **The legacy pipeline in the middle of the BPE story.** `text-sequence.md`, structural finding; three "later sections use it" justifications.
9. **A non sequitur about production vocabularies**, next to a plot in which a 4,096-token vocabulary out-compresses GPT-2's. TOK-6.
10. **The forward/backward switch in the BPTT derivation.** BPTT-5. One bridging sentence.

## Recurring patterns

- **Claims handed to exercises.** At least nine places where the text asserts something and says an exercise will show it: "exercise ... asks how" (a linear model can continue a sine, SEQ-9); scheduled sampling "narrows the gap" (SEQ-8); "no choice of $\alpha$ rescues it (the exercises ask you to try)" (LM-8); "for the reason that exercise ... explores" (RNN-2); "(an exercise asks you to watch what goes wrong)" (IMP-12); "as an exercise shows with Adam" and "an exercise asks you to try it" (IMP-13); "The exercises explore what this changes" (BPTT-4); and the gradient spikes of IMP-7. Exercises should extend what the text has shown, not carry its evidence. I suspect this pattern is book-wide.
- **Infrastructure for other chapters, justified as such.** "several later sections reuse it", "which some later sections still use", "several later sections call it" (`text-sequence.md`); "which later chapters use for character-level models" (`language-model.md`); "later sections use them as well" (`rnn-implementation.md`); special tokens, the end-of-sequence handling and length normalization, none exercised in this chapter. Six instances. Each asks the reader to learn something now for a use they cannot see.
- **Verification cells of low value.** The concatenation identity (RNN-2), `allclose` for one-hot against lookup (IMP-4), `check_len` and `check_shape` (IMP-3), fifteen random draws per cutoff (DEC-9), the untrained sample (IMP-17). They read as holdovers from an earlier edition. Compare the checks that do teach: the initial perplexity of 1,027, the toy BPE trace reproduced by the code, the `assert` against `tiktoken`.
- **Notation for the conditioning context.** $x_1, \ldots, x_{t-1}$ (`sequence.md`, top of `language-model.md`), $x_{t-1}, \ldots, x_1$ (rest of `language-model.md`, `rnn.md`), $x_{<t}$ (`decoding.md`). Adopt $x_{<t}$ throughout. Smaller clashes: $b$ as minibatch size and as a recursion coefficient, $g$ as update function and gradient (BPTT-2); $\mathbf{x}$ for parameters (IMP-9); $\beta$ in prose and `alpha` in code (DEC-4).
- **Repetition across sections.** One-hot equals lookup (three tellings, two checks); truncation and detaching (twice); the clipping formula (twice); greedy decoding and temperature with outputs (twice); $10^{150}$ (twice in forty lines). The chain rule appears four times, which October flagged; I think those restatements are the right kind and would leave them.
- **Numbers reported ahead of their experiment.** The parenthesis in `rnn.md` about the 740,355-parameter model (RNN-3).
- **Framework plumbing in prose.** Device synchronization, cuDNN bias vectors, launch overhead (IMP-10, IMP-18). The PyTorch reader is mostly spared tab fragmentation in this chapter; `rnn-implementation.md` is the exception, with 19 tab blocks and a timing subsection whose four tabs reach different conclusions.
- **Whole-framework import cells for one numpy cell.** `rnn.md` and `bptt.md` each open with a tab selector and four import cells to support a single small computation.
- **Language.** Very little to report. "Contract" six times in `rnn-implementation.md` (IMP-1); "face the vocabulary" three times in `rnn.md` (RNN-7). "Thus" is used six times, each time to read an equation in a plain sentence, which is the Cover and Thomas habit the guide asks for. No em dashes in body prose. Headings are noun phrases; "Tokenizers in the Wild" is the only one with a flourish.

## What to cut and what is missing

Cut or shrink (about 1,500 words in all): the concatenation check in `rnn.md`; in `rnn-implementation.md` the one-hot check, the shape helpers, most of "Embeddings under SGD", temperature sampling, the untrained sample, the timing subsection; the Pitman-Yor paragraph in `language-model.md` (LM-5); random truncation in `bptt.md` (BPTT-8); the glitch-token paragraph and possibly the `Vocab` class in `text-sequence.md`; the duplicated preview in the opening of `decoding.md` (DEC-1).

Missing (about 600 words and five short cells):
- one construction showing a state beating every window (parity), in `rnn.md`;
- the gradient norm per update, with the clipping threshold, in `rnn-implementation.md`;
- validation loss against position in the window, in `rnn-implementation.md`;
- the gradient norm against lag for the trained model, in `bptt.md`, with three sentences on what the $\tanh$ changes;
- a worked number for the Kneser-Ney recursion (LM-6);
- a reason to want $P(x_1, \ldots, x_T)$ at all (SEQ-2);
- one sentence in `decoding.md` connecting sampling's derailment to the rollout of `sequence.md`.

Not missing: LSTMs, GRUs, bidirectional and deep recurrent networks, sequence-to-sequence models. The chapter is right to leave them out and says where the first two went.

## What is best and should set the model

- **The opening of `bptt.md`**: a question, the mechanism in one sentence, an everyday image, the smallest case with two numbers the reader can check, and why it matters, all before the first heading.
- **"From Conditionals to Sequences" in `sequence.md`**: meaning, formula, "Thus ...", a numerical instance, and the two uses.
- **Perplexity in `language-model.md`**: motivation by a failure of the obvious measure, definition, a reading in words, reference values, and then a measurement that surprises.
- **The BPE hand trace and the GPT-2 check in `text-sequence.md`**: the smallest case solved completely, and a verification against the real world that a reader will remember.
- **The clipping paragraph in `bptt.md`**: a remedy explained by exactly what it does and does not change in an equation the reader has just derived.
- **The exercises of `rnn.md`**: short, on paper, each testing one idea, with the two best (parity, the window inside a recurrence) good enough to be text.
- **Honest reporting of results** throughout: the RNN ties the trigram and the text says so. Keep that habit while making sure each experiment shows what its paragraph claims.

## Still open from October

October called this "the best-organized chapter in Basics"; I agree. Its notation complaints are resolved ($\tau$ for temperature, $w$ for beam width, a consistent $\mathbf{W}_\textrm{xh}$ layout). The chain rule still appears four times, as restatements. "Tokenizers in the Wild" is still in `text-sequence.md`; I would keep its first half (the rank table and the verification) and move only the anecdotal material. The three passages October marked "keep" are intact.
