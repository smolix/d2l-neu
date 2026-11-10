# Writing pass: `chapter_recurrent-neural-networks/decoding.md`

Section 8.7, the last of chapter 8 and of the Basics part. Prose only; no
code cell, label, figure, exercise, or slide was changed.

## Words

Prose words (checker count, exercises included): 4455 before, 4452 after.
The opening lost about 90 words and the summary about 25; the same amount
went into the prose fixes for DEC-3, DEC-4, DEC-8, DEC-12 and the new ending.
Checker: "OK: only prose changed". `tools/lint_source.py`: clean.

## Before and after

1. Opening.
   - Before: "The two simplest rules, tried in :numref:`sec_rnn-scratch`,
     both fell short. How should the choice be made?"
   - After: "The two simplest rules, tried in :numref:`sec_rnn-scratch`,
     both fell short. Always taking the most probable token ran into loops,
     and drawing tokens at random from the model's distribution produced
     invented words. How, then, should the choice be made?"
2. The decoding problem.
   - Before: "One answer is the *most probable* sequence. Finding it is
     intractable."
   - After: "The obvious answer is the *most probable* one. Unfortunately,
     finding it is out of the question."
3. Why greedy decoding loops (DEC-3).
   - Before: "Once a phrase has occurred, a language model tends to assign
     its recurrence a higher probability, because text that repeats tends to
     keep repeating."
   - After: "Greedy decoding feeds its own argmax back in as context, and
     nothing in it is random: the next token is a function of the hidden
     state, and the next state is a function of the state and that token.
     Once the state has settled into a cycle, no chance event can dislodge
     it. Is this a defect of our small model? Unfortunately not."
4. Temperature.
   - Before: "Temperature reshapes the head and the tail together, so
     cooling the distribution enough to suppress the tail also removes
     legitimate variety at the head."
   - After: "Temperature is a blunt tool. It reshapes the head and the tail
     together, so cooling the distribution enough to suppress the tail also
     removes legitimate variety at the head."
5. Top-p against min-p.
   - Before: "it keeps the tokens within a factor of twenty of the best, a
     few dozen at most."
   - After: the same, followed by "Which of the two is right, the counts
     cannot say."
6. The five samples (DEC-8).
   - Before: "Our small model cannot make any of these continuations good,
     but it makes their differences visible. ... The truncated samplers sit
     in between: they keep the variety of sampling and cut the tail on which
     pure sampling trips."
   - After: "Our small model cannot make any of these continuations good,
     and one sample per rule is thin evidence. ... Only greedy decoding
     loops. Among the samplers these lines show no ranking, since invented
     words appear with and without truncation. The truncated samplers are
     designed to keep the variety of sampling and cut the tail on which pure
     sampling trips".
7. Cost (DEC-12).
   - Before: "For a recurrent network the quadratic part is pure waste."
   - After: "We built `step_fn` without a state so that it would fit any
     model, and so it runs the network over the whole prefix for every token
     it generates. ... The sequential part cannot be avoided. The quadratic
     part, for a recurrent network, is pure waste."
8. Ending of the section, the chapter, and the part.
   - Before: "We can now turn a language model into a text generator and
     choose between searching for probable text and sampling varied text.
     What we cannot yet do is measure the quality of open-ended text without
     human readers, and our small model limits how good any decoder can make
     its output."
   - After: "The same state is the model's limit: whatever it remembers of
     the prefix must fit into 128 numbers
     (:numref:`subsec_rnn-constant-memory`). No decoding rule can put into
     the text what the model's distribution does not contain. The decoders,
     however, never looked inside the model, and they will serve unchanged
     when a better one supplies the distribution."

Kept untouched because they already have voice: the thesis paragraph ("Even
if we could find it ..."), the myopia example, "Score, take the argmax,
append, repeat.", "Greedy decoding commits immediately; exhaustive search
never commits at all.", "The two averages disagree.", and "perplexity selects
the model, and human judgment, or a proxy for it, selects the decoding
strategy."

## Repetition removed

- The count $1027^{50} > 10^{150}$ was in the opening and again in "The
  Decoding Problem". It is now only in the subsection (DEC-1).
- The opening previewed greedy decoding, beam search and sampling before
  the first subsection stated them again. The opening is now two paragraphs:
  the problem with the two failures seen in `sec_rnn-scratch`, and the two
  families in one sentence each.
- Greedy decoding and temperature are owned by `rnn-implementation.md`, the
  earlier section. Here the greedy loop is tied to that sample in one clause
  ("as the greedy sample of :numref:`sec_rnn-scratch` did"), and the three
  sentences on $\tau = 1$, $\tau \to 0$ and large $\tau$ are one sentence
  introduced by "which we met in :numref:`sec_rnn-scratch`". The equation
  `eq_temperature` stays, since the sampler and an exercise refer to it.
- The closing paragraph restated the body (choose between search and
  sampling; no measure without human readers). Removed; the summary is the
  only recap, and it is shorter by one sentence (the length-normalized score
  is no longer restated there).
- Tab notes for TensorFlow and JAX cut to what the reader of that tab needs
  to read its code (guide §9).

## Review findings

Applied:

- DEC-1: opening shortened, count stated once.
- DEC-2: the roadmap sentence no longer lists four undefined names.
- DEC-3: circular explanation of repetition replaced by the determinism
  argument, then Holtzman et al. for large models.
- DEC-4 (prose part only): the text now says that without an end-of-sequence
  token all candidates have 30 tokens, the correction changes no ranking,
  and the printed score is the log-probability divided by
  $30^{0.75} \approx 12.8$.
- DEC-5: the prefix $x_{\leq t_0}$ is dropped from the toy example and its
  displayed equations; $t_0$ is introduced where the score needs it.
- DEC-8 (prose part only): the claim that truncated samplers "sit in
  between" is now a statement of design, and the text says the five samples
  do not rank the samplers.
- DEC-10: the reference to `sec_machine_translation` (now in the Attic) is
  dropped.
- DEC-11: top-$k$ is cited to Fan, Lewis and Dauphin (2018) together with
  the GPT-2 report (key `Fan.Lewis.Dauphin.2018` exists in the bibliography).
- DEC-12: the stateless interface is given its reason where it is
  introduced and again in the cost subsection.
- Structural finding: one sentence links sampling's derailment to the
  rollout of `sec_sequence`.

Already addressed in the file, kept: DEC-6, DEC-7.

Deferred:

- DEC-4 (experiment): running beam search with a newline as `eos_id`, and
  renaming `alpha` to `beta`, need code and new outputs.
- DEC-8 (experiment): adding pure, top-$k$ and top-$p$ samplers to the
  evaluation table needs a changed cell and new outputs.
- DEC-9: replacing fifteen random draws by a printed kept set needs a
  changed cell.
- Structural: the section retrains the model; it follows from one notebook
  per section.
- Structural (IMP-16): temperature is still demonstrated in both sections.
  Under the earliest-section-owns rule this section reduced its explanation;
  the temperature sweep cell stays because cells may not be removed.

## Factual claims added or changed

Checked against the stored outputs of all four frameworks where they
describe an output.

- "Always taking the most probable token ran into loops, and drawing tokens
  at random ... produced invented words" (of `sec_rnn-scratch`): the greedy
  sample there loops in all four stores, and that section's prose says both.
- "Both continuations end in a loop": true in all four stores.
- "the next token is a function of the hidden state, and the next state is
  a function of the state and that token. Once the state has settled into a
  cycle, no chance event can dislodge it": new mechanism for the loop. I
  wrote "settled into a cycle" and not "once a state recurs", because a
  continuous state approaches a cycle and need not recur exactly.
- Top-$k$ citation now includes Fan, Lewis and Dauphin (2018).
- "a top probability between 0.1 and 0.2" replaces "near 0.1": the stores
  give 0.19, 0.16, 0.09, 0.21.
- "every candidate has 30 tokens and the length correction changes no
  ranking here: the score is the log-probability divided by
  $30^{0.75} \approx 12.8$": follows from the code (`num_tokens=30`, no
  `eos_id`).
- "Only greedy decoding loops" and "invented words appear with and without
  truncation" (the five samples): true in all four stores. In the MXNet
  store the top-$k$ sample happens to contain no invented word, so the
  sentence is worded about the set of samples, not about every sample.
- "The sequential part cannot be avoided": of autoregressive generation.
- "whatever it remembers of the prefix must fit into 128 numbers": the 128
  hidden units, as in `subsec_rnn-constant-memory`.
- "This is the compounding of errors in a rollout (:numref:`sec_sequence`),
  with tokens in place of numbers": `sequence.md` describes this problem in
  prose and says its own experiment does not demonstrate it.
- Removed: "This constant cost and memory ... is a property of every
  recurrent architecture" is now "is what a fixed-size state provides".
- Removed from the TensorFlow and JAX tab notes: the remarks on Python
  overhead when run eagerly and on retracing the scan when uncompiled.

## Slides that now disagree with the body

- "Greedy Decoding Takes the Best Token ..." slide: "A phrase that has
  occurred becomes more probable: text that repeats tends to keep
  repeating." The body no longer gives this explanation (DEC-3); it gives
  the determinism argument.
- "The problem" slide repeats the opening's old wording ("generic text that
  tends to repeat", "varied text that sometimes derails") and the
  $10^{150}$ count. Not wrong, but the body's opening now names loops and
  invented words, and gives the count in the first subsection.
- The slides do not mention that length normalization has no effect in this
  section's runs, which the body now states.
- No slide claims that the truncated samplers "sit in between", and the
  top-$p$ against min-$p$ slide already matches the body.
