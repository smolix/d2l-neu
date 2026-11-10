# Recurrent Neural Networks

Source: `chapter_recurrent-neural-networks/rnn.md`. About 2,350 words of prose (all shared), 875 words of exercises, two schematic figures, one table, two code cells.

## Verdict

A clear, economical definition of the recurrent network: the problem first, the MLP the reader knows, one new term, a scalar example worked to three decimals, then the language model with its parameters counted. It does what a definition section should and is easy to read. It is also a little thin where it matters most. The reader is told that the state "must learn what to keep" but is never shown a state doing something a window cannot; that demonstration lives in two excellent exercises. Meanwhile 200 words and two cells verify that a block matrix product equals the sum of two products. Trade the second for the first.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- **The opening question.** "a character introduced on the first page of a novel still shapes how its last page reads. How can a model take the whole prefix into account without growing as the prefix grows?" Then the obvious fix is shown to fail with a number ($1024^{20} \approx 10^{60}$, correct).
- **"A state of fixed size cannot keep everything, so it must learn what to keep."** The sentence the chapter's later sections (and the Advanced part) rest on.
- **The two meanings of "hidden".** "A hidden layer sits between the input and the output of one computation. A hidden state is carried from one time step to the next". It anticipates a real confusion in three sentences.
- **What changed.** "Compared with :eqref:`rnn_h_without_state`, the only new term is $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$". The new model is presented as a one-term difference from a known one, as the guide's MacKay lesson asks.
- **The smallest case.** $d = h = 1$, input $1, 0, 0$, states $0.76$, $0.36$, $0.18$ (all correct), ending "The recurrent weight sets how long the past persists." This prepares `bptt.md` without saying so.
- **The parameter table.** 222,915 parameters, itemized and correct, with the reading "The recurrent layer itself holds 24,704 parameters, 11% of the total." A reader learns where a language model's weight sits before training one.
- **The teacher-forcing figure's example.** "struck" splitting into `␣stru` and `ck`, so that at step 3 the model "must predict `ck`, completing the word it has started", makes subword prediction concrete.

## Structural findings

- **The state's power is asserted, not shown.** The section's title idea is that a fixed-size state can do what no window can. The proof-by-construction is in exercise 4 (an RNN whose state is a shift register reproduces the window model) and exercise 6 (two hidden units compute parity over any length, which no window can). One of these, in three or four sentences, belongs in the text; parity is the sharper, because it shows a single bit of state beating every window.
- **The embedding gets one sentence of motivation.** `language-model.md` ended by promising that neural models "share statistical strength between related words" ("cat" and "feline"). The embedding is the mechanism, and it is introduced here as a fix for arbitrary ids, inside a paragraph that also introduces the neural $n$-gram and a parameter count (RNN-6). The promise is never picked up.
- **The closing hand-off points at the wrong section** (RNN-5).
- If the concatenation check goes (RNN-2), the section has no code and needs no import cells in four frameworks.

## Passage-level findings

- **RNN-1** (minor, language). Opening. Quote: "$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid \mathbf{h}_{t-1}).$$". Reversed conditioning order, see LM-2.
- **RNN-2** (moderate, pedagogy). "Recurrent Neural Networks with Hidden States". Quotes: "The two products in :eqref:`rnn_h_with_state` can be computed as one." and "The two (3, 4) results differ by at most a few millionths, the rounding". A paragraph, a displayed identity, two code cells in four tabs and a sentence about 32-bit rounding establish that $[\mathbf{X}, \mathbf{H}]$ times the stacked weights equals the sum of the two products, which is the definition of a block product. The reason it is mentioned at all is deferred ("for the reason that exercise :exref:`ex_rnn-input-product` explores"). This is the one place the section drags. Fix: one sentence ("Equivalently, the layer is a fully connected layer applied to the concatenation $[\mathbf{X}_t, \mathbf{H}_{t-1}]$"), no code, and give the reason in a clause (the input product can be computed for all steps at once during training).
- **RNN-3** (moderate, pedagogy). "Model Size and Computation". Quote: "embeddings to $d = 512$ and adds dropout: 740,355 parameters, still 89%". A parenthesis reports the size of a second model that the next section will build for reasons the reader does not yet know. It is correct (740,355; 89%; 81,920) and useless here. Fix: delete; the next section states the count where the model is built.
- **RNN-4** (minor, explanation). "Neural Networks without Hidden States". Quote: "A token id is not a useful input by itself, since the". The paragraph introduces the embedding table, the neural $n$-gram and the cost of widening the window. The embedding deserves its own short paragraph with the reason it matters: tokens that occur in similar contexts can receive similar vectors, so what the model learns about one transfers to the other. That is the answer to the "cat" and "feline" problem of the previous section and should be said in those words.
- **RNN-5** (moderate, thought). End of "Teacher Forcing". Quote: "and count its parameters and computation. One question remains before we". The text continues "train it" and points to :numref:`sec_bptt` for the gradient analysis, but the next section is the implementation, which trains the model first, and the opening of this section says ":numref:`sec_rnn-scratch` then implements and trains it." The two ends of the section disagree about what comes next. Fix depends on the order chosen (see the chapter file): either move `bptt.md` forward, in which case this paragraph is right, or rewrite it to hand off to the implementation and mention that training will need one precaution, clipping, explained after.
- **RNN-6** (minor, thought). "Teacher Forcing". Quote: "continues from, and errors can compound over a long generation. Teacher". This leans on "the gap between one-step and multistep prediction that :numref:`sec_sequence` measured on a sine wave", where compounding was not what the experiment showed (SEQ-1). Keep the reference to the gap; soften "compound" unless `sequence.md` is changed to show it.
- **RNN-7** (minor, language). "Model Size and Computation" and Summary. Quote: "other 89% face the vocabulary, in the embedding table and the output". "Face the vocabulary" and "vocabulary-facing layers" are coined here and used three times. They are understandable, but "the embedding table and the output layer, whose sizes scale with the vocabulary" says it without a coinage.
- **RNN-8** (minor, pedagogy). "Constant Memory per Step". Quote: "is *attention* (:numref:`chap_attention`), whose memory and computation per". A well-judged forward pointer: it names the alternative by what it changes and does not depend on it. Keep as is. Noted here only because it is the model for how the chapter should lean forward.

## Exercises

The best set in the chapter and among the best in the part. Exercise 1 compares the growth in $m$ of a count table, a neural $n$-gram and an RNN with numbers. Exercise 2 answers the question the text left open about separate products. Exercise 3 (exposure bias: $0.99^{100}$) is a one-line calculation that makes teacher forcing's weakness vivid. Exercise 4 (a window inside a recurrence, with $\mathbf{W}_\textrm{hh}^m = \mathbf{0}$) and exercise 6 (parity) are the conceptual core that the text lacks. Exercise 5 (embedding lookup against a one-hot product) duplicates a check that `rnn-implementation.md` also runs; keep it here and drop it there.

## Still open from October

October's note that $\mathbf{W}_\textrm{hx}$ flipped layout is resolved: the layout is $\mathbf{W}_\textrm{xh} \in \mathbb{R}^{d \times h}$ throughout the chapter.
