# Writing pass: Reproducibility and Inspection (5.8)

Source: `chapter_builders-guide/reproducibility-inspection.md`, the last
section of chapter 5. Prose words (checker, all four tabs): **6784 before,
6725 after** (-59). Checker: "OK: only prose changed". `lint_source.py`: clean.
No code cell, label, figure, exercise, or slide was touched.

## Before/after pairs

1. Before: "For the classifiers of :numref:`chap_classification` and
   :numref:`chap_perceptrons`, they differ by a few tenths of a point, because
   each run draws different initial weights and visits the examples in a
   different order."
   After: "For the softmax regression of :numref:`sec_softmax_scratch` we have
   the answer already: repeated runs moved its accuracy by about a point,
   because each run draws different initial weights and visits the examples in
   a different order. A gain of 0.3 points is well inside that range."
2. Before: "... so seeding the generators fixes the draws; repeating a run bit
   for bit also requires that the arithmetic adds numbers in a fixed order."
   After: "... so seeding the generators fixes the draws. That is the easy
   part. Repeating a run bit for bit also requires that the arithmetic adds
   numbers in a fixed order, which a GPU does not promise."
3. Before: "Where the random functions get their state from is the first thing
   to know about a framework: a global generator that they consult unless told
   otherwise, or keys passed explicitly."
   After: "A framework's random functions either consult a global generator or
   take their state as an argument. Which of the two a framework does decides
   how a run is seeded."
4. Before: "Two failures are common."
   After: "Two failures are common, and neither raises an error."
5. Before: "The smallest example takes three float32 numbers."
   After: "Three float32 numbers suffice to show it."
6. Before: "This growth with depth is why the residual branches of deep
   networks are often initialized with smaller weights
   (:numref:`sec_init_param`)."
   After: "Eight blocks should multiply 0.69 by $1.5^4 \approx 5$, which gives
   3.5, and the table agrees to within the scatter of one random draw. This is
   the calculation of :numref:`sec_init_param` with a factor of 1.5 where He
   initialization gave 2, and it is the reason why the residual branches of
   deep networks are often initialized with smaller weights."
7. Before: "To test this, we put a NaN into one weight of the fourth layer of
   the stack, the third residual block:"
   After: "To test this, we put a NaN into one weight of the third residual
   block, the module at index 3 of the stack:" and, after the output, "A
   planted NaN is the easy case, since we knew where to look. In a real run
   the finder names the first symptom, and the cause lies at or before it:
   activations may grow over several layers before one of them overflows."
8. Before: "We can now repeat a run bit for bit on one machine, say which
   sources of randomness and which operations stand in the way, and observe
   any layer of an unmodified model during a run."
   After (chapter ending): "The 8-billion-parameter model of the chapter
   introduction has not become any smaller: its weights still take 32 GB in
   fp32. But we can now say how such a model is built, initialized, stored,
   placed on a device, and observed, each time by a walk over the same tree of
   modules."

## Repetition removed

- "`net(X)` calls `__call__`, which wraps `forward`" (owned by 5.1): two
  sentences in the MXNet and PyTorch tabs reduced to one clause with a
  reference.
- Residual block definition (owned by 5.1): reduced to one sentence; the
  double colon is gone (REP-9).
- "The function below draws everything ... so one call fixes all its draws":
  stated in three tabs, now once in the shared prose.
- "No generator objects / no `generator=`" in MXNet: told in three tab blocks,
  now once; the MXNet and JAX tab blocks after the generator cell are removed
  (both repeated the tab before the cell).
- Variance growth per block: tied in one clause to the derivation in 5.3
  instead of standing as an unrelated second derivation.
- Closing paragraph no longer restates the section's results; the Summary is
  the only recap and lost its list of randomness sources (already in the body).
- Tab prose trimmed throughout (Gluon `Block.summary()`, `Trainer.allreduce_grads`,
  backward-hook tuple/return-value details, JAX borrowed loaders, TF stateless
  seed pairs).

## Review findings

Applied (prose only): REP-1 (claim replaced by the range the book records),
REP-2 (softened to "serve one purpose"; the closing now admits the replay was
not carried out), REP-3, REP-4 (forward clause), REP-5 (prose now says the
cell shows only the shuffle order and points to the exercise), REP-6 (version
trivia cut), REP-7 (prose says the operation was chosen only because it
fails), REP-8 (reference moved to the spacing fact), REP-9, REP-10 (prose
explains that the model is unseeded), REP-11, REP-12 (index wording and the
symptom/cause sentence), REP-13 (PyTorch reading added; single-layer norms
given a purpose), REP-14 (question first, signature prose cut), REP-15
(partly: "has no" openings reworded or cut where they were workaround notes).
Structural finding 5: chapter ending added (three sentences).

Deferred:
- Structural 1 / REP-1 experiment (five seeds of one classifier): new cell.
- Structural 2, REP-5, REP-7 (show worker failure, show GPU nondeterminism;
  drop the worker-initializer cell): new or changed cells.
- Structural 3 (capstone replay, or moving hooks to 5.3): code and file moves.
- Structural 4 (feature-matrix form of the first half): needs restructuring
  across cells; only trimmed.
- REP-9 second half (save `ResidualBlock` once in `d2l`): code.
- REP-10 (remove `torch.seed()` line): code.
- REP-12 (promote the divergence monitor into the body): new cell.
- REP-13 for MXNet, TensorFlow, JAX: their cells print one layer's norms only,
  so the per-block reading exists in the PyTorch tab alone.
- Exercise suggestions (seed-spread exercise, paper exercise, untabbing): the
  brief excludes exercises.

## Factual claims added or changed

1. Opening: run-to-run difference changed from "a few tenths of a point"
   (chapters 3 and 4, unsupported) to "about a point" for softmax regression,
   citing `sec_softmax_scratch`, whose text says repeated runs land between
   82% and 84%.
2. "which a GPU does not promise" (fixed order of additions); the section
   body supports it.
3. "Two failures are common, and neither raises an error."
4. Input layer: Xavier weights of variance $2/84$, output variance
   $20 \cdot 2/84 \approx 0.48$, standard deviation 0.69 (stored outputs:
   0.69 to 0.70 in all four frameworks). Eight blocks: $0.69 \times 1.5^4
   \approx 3.5$ (stored: 3.25 to 3.92).
5. "The calculation of `sec_init_param` with a factor of 1.5 where He
   initialization gave 2" (5.3 derives $1 + g^2 = 2$ for $g = 1$).
6. "We do not seed this model. Its weights are drawn afresh in every run."
   Checked per tab: `torch.seed()`, `SystemRandom` in MXNet and TensorFlow,
   and `d2l.get_key()` whose master key is drawn from `SystemRandom`.
7. NaN finder: "the finder names the first symptom, and the cause lies at or
   before it: activations may grow over several layers before one of them
   overflows."
8. MXNet NaN tab: "inside the same kernel, and that kernel maps NaN to 0"
   reworded to "inside the layer, and that ReLU maps NaN to 0" (avoids
   "kernel" for an operator; same claim, not re-verified by running MXNet).
9. PyTorch gradient reading: "the gradient norm grows by a fifth to a third
   per block, severalfold over the stack" (stored: 0.236 to 1.26, ratios 1.20
   to 1.33), with the explanation that the backward pass also adds the body's
   signal to the bypass. The run is unseeded, so the range is from one draw.
10. Chapter ending: 8-billion-parameter model, 32 GB of weights in fp32 (as in
    the chapter introduction); every operation as a walk over the module tree.

## Slides that now disagree with the body

- Slide "A 0.3-Point Gain Means Nothing If Two Runs Differ by as Much": "Two
  runs of the **unchanged** script already differ by a few tenths". The body
  now says about a point, per `sec_softmax_scratch`.
- MXNet determinism slide: "avoiding the operations whose kernels add in a
  varying order"; the body no longer says "kernels" (wording only).
