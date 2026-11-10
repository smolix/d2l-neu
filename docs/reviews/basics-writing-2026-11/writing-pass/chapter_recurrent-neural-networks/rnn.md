# Writing pass: `chapter_recurrent-neural-networks/rnn.md` (section 8.4)

Prose words: 3061 before, 3104 after (+43; checker: "OK: only prose changed";
`lint_source.py` clean). The checker's em-dash warning is the table rule
`|---|---|--:|` of the parameter table and was present before the pass.

The section grew slightly, against the brief. The review asked for two
additions (a state doing what no window can, RNN structural finding; the
embedding as the answer to "cat" and "feline", RNN-4), about 115 words
together, and offered the concatenation check as the cut that pays for them.
That cut needs the two code cells removed, which this pass may not do, so
only its prose was trimmed. Everything else removed (about 170 words) is
repetition or the misplaced parenthesis of RNN-3.

Untouched: the opening question, the three "hidden" sentences, "the only
new term", the scalar example and its last sentence, the figure captions,
the parameter table, the `␣stru`/`ck` example, the attention pointer
(RNN-8), exercises, slides.

## Before/after pairs

1. - Before: "Widening the window does not answer this."
   - After: "Widening the window is the obvious remedy, and it does not get
     far."
2. - Before: "The approximation is exact only if $\mathbf{h}_{t-1}$ retains
     everything about the prefix that matters for predicting $x_t$. A state
     of fixed size cannot keep everything, so it must learn what to keep."
   - After: "Note, however, that the approximation is exact only if
     $\mathbf{h}_{t-1}$ retains everything about the prefix that matters for
     predicting $x_t$. That is a great deal to ask. A state of fixed size
     cannot keep everything, so it must learn what to keep."
3. - Before: "A token id is not a useful input by itself, since the ids are
     arbitrary labels."
   - After: "What should the input be? A token id will not do, since the ids
     are arbitrary labels."
4. - Before: "A state of 128 numbers cannot store an arbitrarily long text,
     so to record something new the network must overwrite something old,
     and it has to learn what to keep."
   - After: "No novel fits into 512 bytes, so to record something new the
     network must overwrite something old."
5. - Before: "The recurrent layer itself holds 24,704 parameters, 11% of the
     total. The other 89% face the vocabulary, in the embedding table and
     the output layer."
   - After: "The recurrent layer, which does all the remembering, holds only
     24,704 parameters, 11% of the total. The other 89% sit in the embedding
     table and the output layer, whose sizes scale with the vocabulary."
6. - Before: "Teacher forcing trains the model only on the first situation,
     and :numref:`sec_decoding` studies generation in the second."
   - After: "Teacher forcing never shows the model its own mistakes, and
     :numref:`sec_decoding` studies generation, where the model has to live
     with them."
7. - Before: "We can now write down an RNN language model and its training
     objective, and count its parameters and computation. One question
     remains before we train it. [...] :numref:`sec_bptt` shows that such
     products tend to vanish or explode as the number of steps grows."
   - After: "One precaution is needed before we train this model. The loss
     at step $t$ depends on the weights through every earlier step of the
     recurrence, so its gradient contains products of one factor per step,
     and such products tend to vanish or explode as the number of steps
     grows. :numref:`sec_rnn-scratch` therefore limits the size of the
     gradient, and :numref:`sec_bptt` analyzes the products themselves."
8. - Before: "Current libraries keep the two products separate, for the
     reason that exercise :exref:`ex_rnn-input-product` explores."
   - After: "Current libraries nonetheless keep the two products separate:
     during training the inputs of all steps are known in advance, so their
     product with $\mathbf{W}_{\textrm{xh}}$ can be computed in one
     multiplication, outside the loop over time steps (exercise
     :exref:`ex_rnn-input-product`)."

## Repetition removed

- "Every additional token of context adds weights" was told three times
  (opening, the neural $n$-gram paragraph with the number 8,192, "Constant
  Memory per Step"). It now lives only with its number in the neural
  $n$-gram paragraph.
- "Ten tokens or a million" appeared in the opening and with the 512 bytes.
  Kept only with the 512 bytes.
- "It must learn what to keep" appeared in the opening and again in
  "Constant Memory per Step". Kept in the opening (the sentence the review
  protects).
- "Does not depend on the length of the sequence" was stated for the
  parameters, again after the table, and in the summary. The sentence after
  the table is gone.
- "Thus the new state is a hidden layer whose inputs are ..." restated the
  first sentence of its own subsection. Removed.
- The list of the five parameter symbols before the table that lists them,
  and the sentence that reads :numref:`fig_rnn` back as the equations.
  Shortened.
- The formulaic closing recap ("We can now write down ...").

## Review findings

Applied:
- **Structural (state's power asserted, not shown):** one paragraph on
  parity after the scalar example, pointing to the last exercise for the
  weights.
- **RNN-2 (partly):** prose around the concatenation check cut from about
  120 to about 75 words; the reason for separate products is now given in
  the text instead of being left to the exercise.
- **RNN-3:** parenthesis about the 740,355-parameter model deleted
  (`rnn-implementation.md` states that count where the model is built).
- **RNN-4:** the embedding has its own paragraph and picks up "cat" and
  "feline" from `language-model.md`.
- **RNN-5:** the closing now hands off to `sec_rnn-scratch` (which trains,
  with the gradient limited) and then to `sec_bptt`, in the chapter's
  actual order.
- **RNN-7:** "face the vocabulary" and "vocabulary-facing layers" replaced
  in body and summary.

Deferred or skipped:
- **RNN-1** ($x_{t-1}, \ldots, x_1$ against $x_{<t}$): needs one decision
  for the chapter.
- **RNN-2 (rest):** removing the two check cells and the four import cells
  is a code change.
- **RNN-6:** already addressed ("errors can compound"), and `sequence.md`
  now says itself that it measured the gap and not accumulation. Left.
- **RNN-8:** "keep as is". Kept.

## Factual claims added or changed

- Embeddings: "tokens that occur in similar contexts can receive similar
  vectors, and what the model learns about 'cat' then transfers to
  'feline'." Stated as a possibility ("can"). Note that
  `rnn-implementation.md` finds the embeddings barely train under SGD, so
  the chapter still never shows this happening (chapter finding 5).
- Parity: no model reading the last $m$ inputs computes the running parity
  (flipping a bit outside the window flips the answer); one bit of state
  suffices; "two hidden units with a threshold activation suffice". The last
  claim is taken from exercise 6. I checked it: with $p = h_1 - h_2$ of the
  previous step, $h_1 = [x + p \geq 1]$ and $h_2 = [x + p \geq 2]$ give
  output $h_1 - h_2 = x \oplus p$.
- Separate products: the reason is now asserted in the text (the input
  product can be computed for all steps at once during training). The old
  text implied it through the exercise. "Some early implementations computed
  it this way" was dropped.
- "No novel fits into 512 bytes" replaces "a state of 128 numbers cannot
  store an arbitrarily long text".
- Closing: "its gradient contains products of one factor per step" (was "is
  a product"), and the statement that `sec_rnn-scratch` "limits the size of
  the gradient" (it clips the norm; clipping is not named here).
- Not changed: "at most a few millionths" for the check. The stored values
  are $2.4 \times 10^{-7}$ (PyTorch, TensorFlow, JAX) and
  $4.8 \times 10^{-7}$ (MXNet), so the bound holds and is loose.

## Slides against the body

No slide contradicts the revised body. Differences in wording only:
- Slides "89% of the 222,915 Parameters Face the Vocabulary" and the recap
  ("89% vocabulary-facing") keep the coinage that the body dropped (RNN-7).
- The slides have no parity slide and no "cat"/"feline" motivation for the
  embedding; the body now has both.
- Slide "A Window Forgets ..." keeps "each extra token adds $dh$ weights",
  which the body now says once (in the neural $n$-gram paragraph), not in
  the opening.
- The "Next" line of the recap slide leads to the gradient analysis, while
  the body's closing now leads to the implementation first.
