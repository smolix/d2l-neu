# Writing pass: `chapter_recurrent-neural-networks/sequence.md`

First section of chapter 8 after the introduction, so it owns every fact it
shares with later sections (chain rule, Markov condition, rollout).

Prose words (checker): 3303 before, 3376 after (+73). The section grew
slightly because four review findings ask for added content (SEQ-2, SEQ-4,
SEQ-7 as a prose stopgap, SEQ-9; about 170 words together); about 100 words
of repetition and connective prose were removed against that. Checker:
"OK: only prose changed"; `lint_source.py` clean.

## Before/after pairs

1. Before: "This dependence is what makes the past useful: if the entries
   were independent, conditioning on earlier entries would not improve a
   prediction of the next one."
   After: "Far from being a nuisance, this dependence is what makes the past
   useful: were the entries independent, conditioning on earlier ones would
   not improve a prediction of the next."
2. Before: "Before handling targets of any kind, we model the sequence
   itself: we estimate how probable a given sequence is, $P(x_1, \ldots, x_T)$."
   After: the same, followed by "Why would anyone want this number?
   Estimating it needs no labels, since every entry serves as the target for
   the entries before it, and a model that knows which sequences are probable
   can generate new ones and choose between candidates, say two
   transcriptions of one recording."
3. Before: "A window is simple and easy to parallelize."
   After: "A window is simple, and since every prediction depends only on
   its own window, all of them can be computed at once."
4. Before: "One choice above was implicit: we factored the sequence left to
   right." (under "Stationarity")
   After: "The chain rule leaves us one choice, and we made it without
   saying so: we factored the sequence left to right." (moved to the end of
   "From Conditionals to Sequences")
5. Before: "Dynamics with this property are called *stationary*."
   After: "Dynamics with this property are called *time-homogeneous* or,
   more loosely and far more often, *stationary*."
6. Before: "The learned weights show why."
   After: "How can a model with so small a one-step error fail so
   thoroughly? The learned weights tell us."
7. Before: "A linear model *could* continue a sine wave far better;
   exercise :exref:`ex_sequence-window-length` asks how."
   After: "Linear models are not to blame. Without noise, the sine satisfies
   $x_t = 2\cos(0.01)\, x_{t-1} - x_{t-2}$ exactly, so two lags with weights
   of nearly 2 and $-1$ would continue it forever. On the noisy series those
   weights would multiply the noise variance by five, and the fit prefers to
   average (exercise :exref:`ex_sequence-window-length`)."
8. Before: "Two remedies exist. First, ... Second, exposing a model to its
   own predictions during training, not only to observed histories, narrows
   the gap ..."
   After: "Two lines of work address it. ... And a model can be trained on
   its own predictions as well as on observed histories, so that the
   multistep setting it is used in is no longer new to it. Whether that
   helps our linear model is left to exercise
   :exref:`ex_sequence-scheduled-sampling`."

## Repetition removed

- $n$-grams as the window idea: told under "Fixed Windows", again under "The
  Markov Condition", again in the Summary. The first is cut; the Markov
  paragraph (which says what an $n$-gram model does) and the Summary clause
  remain.
- "A one-step loss rewards only one step": told in the weights paragraph,
  again in two sentences after the error table, again in the Summary. The
  second sentence after the table ("Averaging the last four values is close
  to the best one-step rule ...") is cut.
- "The error is how far the sine moves in $k$ steps": derived before the
  integral, restated two paragraphs later; the restatement is now a clause.
- The stationarity assumption was stated in the introduction to "Markov
  Models and Stationarity" and again, nearly verbatim, as the first sentence
  of the subsection; the introduction now only names it.
- Summary: dropped the definition of a sequence, the clause "fed its own
  outputs it only carries the last observed level forward", and the
  restatement that rollouts feed predictions back.
- "Sequential Data": dropped the recap of the opening examples.

## Review findings

Applied: SEQ-2 (reason for wanting $P(x_1,\ldots,x_T)$; no speech example,
because `language-model.md` opens with it), SEQ-3 ("Both are autoregressive.
They differ in what they keep of the past."; the window is "the classical
form of autoregression"; Summary no longer splits the names), SEQ-4
(time-homogeneous; fixed rule does not mean identically distributed), SEQ-5
(left-to-right paragraph moved), SEQ-6 (rollout written with $t+1,\ldots,t+5$;
604 named only at the loop), SEQ-8 (no untested claim; exercise 7's
wording changed from "states that ... narrows the gap. Test the claim" to
"mentions ... as a way to narrow the gap. Test the idea" so that it agrees),
SEQ-9 (two-lag identity), SEQ-10.

Already addressed before this pass, kept: SEQ-1.

Deferred:
- SEQ-7: needs a changed plot cell. Stopgap in prose: "(The plot shows the
  one-step predictions in place of the series, which they track closely.)"
  Remove that sentence when the plot is redrawn.
- SEQ-1 option (b), a second model whose rollout does accumulate error: new
  experiment.
- Heading "Error Growth with the Horizon" kept (it is true of the output,
  and exercise 4 carries the same title).
- Figure caption of `fig_ar-vs-latent` still says "Autoregression" (left)
  against "Latent autoregression" (right): caption is outside this pass
  (SEQ-3 remainder).
- Exercise 2 (momentum strategies) left as is.
- Notation $x_1,\ldots,x_{t-1}$ against $x_{<t}$: chapter-wide decision.

## Factual claims added or changed

- A noise-free sine satisfies $x_t = 2\cos(0.01)x_{t-1} - x_{t-2}$ exactly
  (from $\sin(a+h)+\sin(a-h) = 2\sin a\cos h$), so two lags suffice. This
  gives away the answer to exercise 1b; the exercise still asks why.
- With weights $2\cos(0.01)$ and $-1$ the noise variance of the prediction
  is $(4+1)\cdot 0.04 = 0.2$, "multiplied by five" (against 0.01 for the
  average of four).
- The rule's constancy is "time-homogeneous", "more loosely and far more
  often" called stationary.
- A fixed transition rule does not make the entries identically distributed.
- Estimating $P(x_1,\ldots,x_T)$ needs no labels; such a model can choose
  between candidate transcriptions.
- "all of them can be computed at once" (window predictions are
  independent of each other).
- Scheduled sampling: the text no longer asserts that it narrows the gap.
- Unchanged and checked against all four stored outputs: weights near 1/4;
  errors about 0.05, 0.05, 0.07, 0.23 to 0.27; "four to six times" (ratios
  4.8 to 5.4); "roughly half the variance" (0.48).

## Slides that now disagree with the body

- Rollout slide: "Observed up to step 604; to reach $\hat{x}_{609}$" with
  indices 601 to 609; the body now uses $t+1,\ldots,t+5$ (not wrong, only
  different).
- "Beyond the sine" slide: "Remedies: ... train on the model's own
  predictions (*scheduled sampling*)" states as a remedy what the body now
  leaves open.
- Recap slide: "**Autoregression:** ... window ...; **latent
  autoregression:** ... state" keeps the naming split that SEQ-3 removed
  from the body's Summary.
- Stationarity slide: "*Stationary*" without "time-homogeneous".
