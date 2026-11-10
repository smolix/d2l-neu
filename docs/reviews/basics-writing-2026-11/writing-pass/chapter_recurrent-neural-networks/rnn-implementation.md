# Writing pass: `chapter_recurrent-neural-networks/rnn-implementation.md`

Section 5 of 7 in chapter 8 (after `sequence`, `text-sequence`,
`language-model`, `rnn`; before `bptt`, `decoding`). Prose only: no code
cell, label, figure, or slide changed. Checker: "OK: only prose changed";
`tools/lint_source.py` reports nothing.

Prose words: 5579 before, 5466 after (-113).

## Before/after pairs

1. Before: "The code for these equations keeps one contract."
   After: "The code for them has a simple interface."

2. Before: "With weights this small, the recurrent weight has spectral
   radius about $\sigma\sqrt{h} \approx 0.11$, so the untrained cell forgets
   its state within a few steps, and training grows the weights from there."
   After: "With weights this small, the largest eigenvalue of the recurrent
   weight matrix has magnitude about $\sigma\sqrt{h} \approx 0.11$. Recall
   the scalar example of :numref:`sec_rnn`, where the past persisted scaled
   by about $w_\textrm{hh}$ per step: at 0.11 per step, the untrained cell
   forgets its state within a few steps. Training has to grow the weights
   from there."

3. Before: "With a vocabulary of 27 characters this is harmless. With our
   BPE vocabulary of $|\mathcal{V}| = 1{,}027$ ids it is wasteful, since
   each step multiplies a 1,027-wide vector that is zero everywhere but one
   entry."
   After: "Since multiplying a one-hot vector by a matrix selects a row of
   it (:numref:`sec_rnn`), this is nothing but a lookup, computed the
   expensive way. With a vocabulary of 27 characters nobody minds. With our
   BPE vocabulary of 1,027 ids, each step would multiply a vector that is
   zero everywhere but in one entry."

4. Before: "Whenever it acts, clipping changes the update, so training no
   longer follows the gradient, and its effect on convergence is hard to
   analyze."
   After: "Note that this is a rather crude device. Whenever it acts,
   clipping changes the update, so training no longer follows the gradient,
   and its effect on convergence is hard to analyze."

5. Before: "The cell therefore reads its inputs through token vectors that
   stay close to their random initial values; rows of tokens that never
   occur in the training text do not move at all."
   After: "Rows of tokens that never occur in the training text do not move
   at all. This is an awkward finding. The table that was meant to let the
   model choose a vector for each token is, in this configuration, little
   more than a fixed random projection."

6. Before: "The RNN alone comes within a few hundredths of a bit per byte of
   the Kneser–Ney trigram, and the mixture beats both by a clear margin,
   because the two models are confident on different tokens."
   After: "After all this effort, the RNN alone does not beat a trigram. It
   comes within a few hundredths of a bit per byte of Kneser–Ney, on the
   wrong side. The mixture is the more interesting number: it is better than
   the trigram by about 0.06 bits per byte, which could not happen if the
   two models assigned the same probabilities to the same tokens."

7. Before: "The model we have built reads a window of 32 tokens and keeps a
   128-number state. With a wider embedding and dropout, it predicts the
   next token about as well as a Kneser–Ney trigram, and averaged with that
   trigram it beats both. A plain RNN still has trouble carrying information
   across many steps, because its gradients pass through the same transition
   at every step."
   After: "A tie with a trigram is less than the hidden state seemed to
   promise, and nothing in this section shows the state using context that
   two tokens do not provide. What we do know is that a plain RNN has
   trouble carrying information across many steps, because its gradients
   pass through the same transition at every step."

8. Before: "Before training, we let the untrained model continue a prefix.
   The continuation consists of random tokens, with replacement characters
   where the generated ids decode to bytes that do not form valid UTF-8. It
   still checks the wiring from the tokenizer through the model and back."
   After: "Before training, we let the untrained model continue a prefix, to
   check the wiring from the tokenizer through the model and back. Expect
   noise: random tokens, with replacement characters where the generated ids
   decode to bytes that do not form valid UTF-8."

## Repetition removed

- **One-hot product equals row lookup** (owner: `rnn.md`). Two paragraphs
  reduced to one; the equivalence is now one clause with a reference. The
  check cell stays (code), introduced by one sentence.
- **Kneser–Ney mechanism** (owner: `language-model.md`). The retelling of
  discounting and distinct predecessors is now a name and a reference.
- **222,915 parameters** (derived in `rnn.md`). Cited once, where the
  overfitting argument needs it; the first mention is now "the one whose
  parameters :numref:`sec_rnn` counted".
- **Closing paragraph** no longer restates the Kneser–Ney result; it ends on
  what the section did not show. The Summary is the only recap and went from
  eight sentences to six.
- **Adam hand-off** told twice, now once (after the comparison), with the
  one-clause description of what Adam does.
- **Trainer hook for clipping**: the sentence repeated in three tab blocks
  is stated once in shared prose; each tab keeps one sentence.
- **"Both times also include the same data loading and validation"**
  (four tabs; already in the shared sentence above the cell) removed from
  the tabs.
- "Contract" (six times) replaced by "interface" or by the thing itself.

Kept here under the earliest-section rule (this section precedes `bptt.md`
and `decoding.md`): the clipping argument and formula, truncation at the
window boundary with the carry-and-detach variant (tightened), greedy
decoding and temperature.

## Review findings

Applied: IMP-1 (contract), IMP-2 (spectral radius glossed through the scalar
example of `rnn.md`), IMP-4 (prose part), IMP-5 (dropout rates announced
where the class is introduced), IMP-8 (learning rate 4; see claims below),
IMP-9 (parameters are $\mathbf{w}$, not $\mathbf{x}$, in the clipping
argument; the symbol occurs nowhere else in the chapter), IMP-10 (device
synchronization dropped), IMP-11 (partly: variant tightened, not removed,
see above), IMP-12 (prose: the parenthetical hand-off to an exercise
removed), IMP-13 (option c in spirit: said plainly that the table is close
to a fixed random projection and that the wide embedding is a workaround;
one Adam hand-off instead of two), IMP-14 (prose only: Kneser–Ney retelling
cut, purpose of the alignment said first), IMP-15 (no "clear margin", no
asserted "because"), IMP-17 (prose shortened), IMP-18 (tab prose trimmed;
the shared sentence about "seconds to tens of seconds" removed since the
TensorFlow times are equal), IMP-19 (summary).

Already addressed before this pass: IMP-7 (the paragraph saying that the
gradient norm is not measured here was kept unchanged).

Deferred:
- IMP-S1, IMP-3, IMP-4 (cell), IMP-17 (cell), IMP-18 (cell and subsection):
  need cells removed.
- IMP-S2: needs a new cell (loss against position in the window).
- IMP-7 (the measurement itself): new cell and plot.
- IMP-6: code change in `forward` and `output_layer`.
- IMP-12 (default split): code change in `language-model.md`.
- IMP-13 options (a) and (b): new configuration and re-capture.
- IMP-14 (helper): code change.
- IMP-15 (evidence that the models differ token by token): new cell.
- IMP-16: this section is the earliest to explain temperature, so it keeps
  it under the parallel-writing rule; removing the branch from `predict`
  is a code change.
- Order of this section and `bptt.md`: chapter-level decision.

## Factual claims added or changed

- "whenever clipping acts, the update has length $\eta\theta = 4$, whatever
  the gradient was." True for plain SGD without momentum or weight decay,
  which is what `configure_optimizers` returns.
- "the largest eigenvalue of the recurrent weight matrix has magnitude about
  $\sigma\sqrt{h} \approx 0.11$" (same number as before, "spectral radius"
  spelled out), linked to the scalar example of `rnn.md`.
- The mixture "is better than the trigram by about 0.06 bits per byte"
  (stored: 2.03 against 2.09 in all four frameworks), replacing "by a clear
  margin, because the two models are confident on different tokens". The new
  reason given is only the logical one: the mixture could not beat both if
  the models assigned the same probabilities.
- "the RNN alone does not beat a trigram ... on the wrong side" (stored:
  2.11 or 2.12 against 2.09 in all four).
- The embedding table "is, in this configuration, little more than a fixed
  random projection" (stored: rows move 4.2% to 4.4% of their length).
- Greedy output: "has the look of English for a few words, and then it gets
  stuck" replaces "with real words, punctuation, and line breaks, but its
  grammar holds over a few words at most" (the PyTorch sample contains
  invented words; all four stored samples loop). The concise model's
  "samples read the same" became "its greedy continuation gets stuck in the
  same way" (true of all four stored outputs).
- TensorFlow timing tab now says "the two times are about equal" (stored:
  36.1 s and 37.7 s).
- Adam is now described in one clause ("an optimizer that adapts the step to
  each parameter"); the claims that it moves the rows much further and lets
  the RNN edge past the trigram are the old ones, not re-checked.
- Removed: "At this model size the difference is seconds to tens of
  seconds"; the notes on transposed weight storage, the fused parameter
  vector, and the optax cache key.

## Slides that now differ from the body

- The slides still say "contract" (cover line, "The contract" kickers, "The
  Framework Layer Keeps the Contract and the Initialization", the wrap-up);
  the body says "interface".
- Slide "An Embedding Lookup Is a One-Hot Product Without the Zeros" keeps
  "One-hot = an embedding whose table is fixed at the identity"; the body no
  longer has this sentence (it says the one-hot product is a lookup computed
  the expensive way).
- Slide line "Adam instead of SGD moves the rows much further and does
  better still" now corresponds to the paragraph after the Kneser–Ney
  comparison, not to "Embeddings under SGD".
- No slide uses $\mathbf{x}$ for the parameters, so the change to
  $\mathbf{w}$ creates no mismatch.
