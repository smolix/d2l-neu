# Review — chapter_linear-classification/softmax-regression-scratch.md  (§4.4 "Softmax Regression Implementation from Scratch")

**Role in the chapter:** The pedagogical demystification step: implement softmax
regression entirely by hand — softmax function, cross-entropy loss, parameter
initialization, forward pass — then train on Fashion-MNIST using the inherited
`Trainer` scaffold. It exists to make the magic of the concise sibling
(§4.5) legible by showing every moving part first.

**Verdict:** Functional and clear at the sentence level, but thin as a
*teaching* document. The code cells do their job, and the numerical-instability
caveat is present (a recent addition with the clip comment). What is missing is
any *discussion* of what is happening — there is no section explaining why
cross-entropy + softmax is the right loss, no prose reflecting on the ~83%
accuracy ceiling that a linear model imposes on Fashion-MNIST, no connection
forward to the concise sibling's log-sum-exp fix (the clip in `cross_entropy` is
a band-aid, and the text says "go to the concise version" but never gives a
`:numref:`). The summary is the weakest in the chapter: a single throw-away
paragraph with the "state of the art of 1960–1970s" phrase. The single
highest-value change is **adding a proper Summary/Discussion section** that
closes the scratch→concise loop, names the ~83% linear ceiling, and forward-
points to the log-sum-exp derivation in `:numref:subsec_softmax-implementation-revisited`.

**Grade:** **B–.** The code is clean and correct (with the recent clip patch),
and the indexing trick is explained, but the file barely teaches beyond the
code — it reads as a notebook stub with light prose glue, not a chapter section.
A top program could assign it, but students would not learn *why* these choices
were made. One focused pass on Discussion + a `:numref:` forward pointer would
lift it to B+.

**Top priorities (ranked):**
1. **[P0] SMS-1** — Add a proper `## Summary and Discussion` that: (a) explains
   the ~83% linear ceiling and what it means; (b) closes the numerical-stability
   loop with a `:numref:` to `subsec_softmax-implementation-revisited`; (c) states
   the scratch→concise purpose explicitly.
2. **[P1] SMS-2** — Add a `:numref:` forward pointer to
   `subsec_softmax-implementation-revisited` immediately after the clipping
   paragraph (the current prose says "production code uses log-softmax" but never
   says *where in this book* that is derived).
3. **[P1] SMS-3** — Add a brief explanation before the cross-entropy loss cell of
   *why* this loss is appropriate: negative log-likelihood = MLE for the
   categorical model (one sentence restating the theoretical link from §4.1,
   keeping the reader grounded).
4. **[P1] SMS-4** — Upgrade the accuracy reporting in the prediction section to
   also print the overall test accuracy (not just the shape of predictions), so
   the reader sees the ~83% number without having to read the training curve.
5. **[P2] SMS-5** — MXNet `prediction-1` emits a massive stderr wall
   (`Storage type fallback detected: operator = stack`) — noisy and confusing.
   Either suppress with `MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0` in the cell, or
   note it is benign.
6. **[P2] SMS-6** — The `## Summary` heading (l. 402) should be renamed
   `## Summary and Discussion` and the stub text replaced (see SMS-1).
7. **[P2] SMS-7** — Exercise 2 ("Implement a cross_entropy function that follows
   the definition...") asks something the *current* code already does — the
   existing `cross_entropy` *is* that function. The exercise needs to be
   clarified or repointed to a genuinely different variant (e.g., the full
   one-hot version `−∑_i y_i log ŷ_i` vs. the index-based shortcut) so it
   teaches the equivalence rather than requesting something already present.

---

## 1. Coverage

### Add

- **Summary / Discussion section — MISSING, P0.** The current `## Summary`
  (ll. 402–411) is four sentences long and says nothing beyond "we trained
  softmax regression." A top-program section needs a closing section that:
  (a) names and explains the accuracy ceiling (~83% on Fashion-MNIST is
  well-known for linear models; the SVG axis confirms values in the 0.8–0.9
  range); (b) explicitly closes the stability loop — the naive clip is good
  enough for learning but the reader should be told the fused-loss fix lives
  in `:numref:subsec_softmax-implementation-revisited`; (c) says what the
  scratch→concise arc buys (muscle memory for the moving parts before they
  disappear into a framework call). This is the most important gap.

- **Why cross-entropy here? — missing, P1.** The section `## The Cross-Entropy
  Loss` (ll. 209–323) launches immediately into the indexing trick without a
  single sentence recalling *why* cross-entropy is the right loss (MLE for
  categorical distributions, derived in §4.1). A top textbook always re-grounds
  the reader before showing the code. One sentence suffices:
  "Recall from :numref:`subsec_softmax-regression-loss-func` that minimizing
  cross-entropy is equivalent to maximizing the log-likelihood of the correct
  labels under our model — so we compute $-\log \hat y_y$ for each example
  and average."

- **`:numref:` to `subsec_softmax-implementation-revisited` — missing, P1.**
  Lines 300–301 explain why clipping is needed and promise "production code
  typically uses a log-softmax layer," but never give a forward reference to
  where *in this book* the log-sum-exp / fused-loss fix is derived. The concise
  sibling's `subsec_softmax-implementation-revisited` is precisely that place.
  One mechanical change: append "See :numref:`subsec_softmax-implementation-revisited`
  for the derivation." This is the key coordination point the task watch-points
  flag.

- **Overall test-accuracy number, P1.** The prediction section shows
  `preds.shape` (256,) and visualizes misclassified images, but never prints a
  number. The reader finishes the section not knowing whether they got 75% or
  83%. Add a one-liner `model.accuracy(data.val_dataloader())` or equivalent
  to make the ceiling tangible. (The training SVG confirms val_acc converges to
  ~0.83 by epoch 10, matching the Fashion-MNIST linear-model benchmark.)

### Remove / trim

- **The "partition function" aside (ll. 73–76).** The two-sentence historical
  note about statistical physics is fine color but is tacked onto the middle
  of the softmax definition cell explanation. It reads as an orphaned aside.
  Either move it to a footnote or cut it — it adds no teaching value here and
  the proper treatment is in §4.1.

- **Nothing else needs cutting.** The file is already short. The priority is
  addition, not subtraction.

### Reorder / restructure

- The `## Prediction` section (ll. 354–400) is currently just a code dump with
  two sentences of prose. Promote it to actually *discuss* the prediction output:
  what classes does the linear model confuse and why (shirt vs. pullover are
  visually similar to a linear model; this is a classic pedagogical point for
  Fashion-MNIST). Even two sentences would give the prediction code purpose.

- No major restructuring needed; the four sections (Softmax / Model /
  Cross-Entropy Loss / Training) are the natural decomposition. The issue is
  the thinness of the prose within and after Training, not the structure.

---

## 2. Teaching quality

### Structure & flow

Four `##` sections + a stub summary. The spine is logical: define the operation,
build the model, define the loss, train, predict. Each section corresponds to
one code concept. This is correct. The weakness is that Prediction and Summary
are prose-free — a student reading linearly gets the "what" from every section
but rarely the "why." The guide's criterion "Every formal object is given a
geometric or operational meaning" is violated in the prediction and summary
sections.

### Figures

No illustrative figures. The only figures are:
- `softmax-regression-scratch-training-1.svg` — the loss/accuracy training
  curve, produced by the `d2l.Trainer` plot call. This is a *data plot of a
  computed result* and is correct here; it should not be pre-generated.
- `softmax-regression-scratch-prediction-2-1.svg` — the misclassification
  visualization. Correct to be inline.

**Missing figure (P2, judgment):** A figure showing the forward pass
architecture — 784-dimensional input → flatten → $784\times10$ weight matrix →
softmax → 10-dimensional probability output — would visually anchor the
`## The Model` section. This is a schematic/illustrative figure and should be
pre-generated. However, `softmax-regression.md` (§4.1) likely already carries
this architecture figure (`:numref:fig_softmaxreg`). **Check whether
`:numref:fig_softmaxreg` should be cross-referenced here** rather than
duplicating it.

### Prose & clarity

- **Line 89** (the instability warning) is the right sentiment but is one run-on
  sentence buried in a paragraph: "Caution: the code above is *not* robust
  against very large or very small arguments. While it is sufficient to
  illustrate what is happening, you should *not* use this code verbatim for any
  serious purpose. Deep learning frameworks have such protections built in and
  we will be using the built-in softmax going forward." The clamp in the
  `cross_entropy` function (ll. 263–264) softens this but doesn't resolve it —
  the softmax itself is still naive. The text should say the clamp in
  `cross_entropy` mitigates the worst symptom for training but the proper fix
  is the fused log-sum-exp loss in the concise sibling, with the `:numref:`.

- **Lines 219–229** (the cross-entropy indexing motivation) are the best prose
  in the file — clear, motivates the trick, gives a 2-example walkthrough.
  Keep exactly as is.

- **Lines 327–345** (the Training section intro) over-explains hyperparameters
  generically ("Note that the number of epochs... and learning rate... are
  adjustable hyperparameters"). This is true of every model in the book and
  repeats §3.3's content verbatim in spirit. Trim to two sentences: note that
  we re-use `fit` from §3.3 and set batch_size=256, lr=0.1, 10 epochs — enough
  for Fashion-MNIST to converge — and let the curve show the result.

- **Lines 402–411** (Summary) — placeholder-quality. See SMS-1.

### Exercises

The five exercise items are reasonable but have two issues:

1. **Exercise 2 ambiguity (P2).** Exercise 2 asks to "implement a
   `cross_entropy` function that follows the definition $-\sum_i y_i \log
   \hat{y}_i$." The current code *already does this* (selecting
   `y_hat[range(len(y)), y]` is equivalent for one-hot `y`). The exercise
   should be clarified to ask students to implement the *one-hot dot-product
   form* (iterate over classes, use the actual one-hot label) and then verify
   it is mathematically equivalent to the indexing form. That makes it
   a genuine discovery exercise about equivalent implementations, not a
   repetition of what the notebook already shows.

2. **Missing convergence analysis exercise (P1, judgment).** There is no
   exercise asking why a linear model plateaus at ~83% on Fashion-MNIST (which
   classes are hardest; what does that tell us about the limits of linear
   separability for this dataset). This is a natural "what just happened?"
   question that connects the training curve to the model's limitations.

3. Exercise 1 (numerical stability of softmax) is well-calibrated and stays.

4. Exercise 3 (most-likely label vs. medical diagnosis) is a good
   real-world extension. Exercise 4 (large vocabulary problem) is a useful
   forward pointer to language models. Exercise 5 (hyperparameter
   experiment) is routine but useful.

---

## 3. Code & examples

### Does the code teach?

Yes, mostly. Each cell has a clear purpose:
- `the-softmax-1`: warms up tensor reduction operations (fine, keeps prereqs fresh).
- `the-softmax-2`: defines naive softmax — compact, legible.
- `the-softmax-3`: verifies outputs are nonnegative and rows sum to 1 — correct teaching check.
- `the-model-1/2`: parameter init and forward pass — teaches the flatten→linear→softmax pipeline.
- `the-cross-entropy-loss-1/2/3`: indexing trick, loss implementation, wiring into model.
- `training`: one-liner using inherited scaffold — correct abstraction level for a scratch implementation.
- `prediction-1/2`: get predictions, visualize errors — functional.

No cell is gratuitous boilerplate. The JAX `loss` method redefines
`cross_entropy` locally (for purity reasons) — this is correct but the
explanation in the `:begin_tab:jax` block is good.

### PyTorch

Clean and idiomatic. `requires_grad=True` on raw tensors is the pre-module
style but is appropriate here (this is a scratch implementation that teaches
the underlying mechanics). `clamp(min=1e-12)` in cross_entropy is correct.
The `parameters()` override returning `[self.W, self.b]` is clean.

No issues.

### JAX

Uses `flax.linen` as `nn` — this is the current Flax API for the book's
setup. `setup()` with `self.param(...)` is standard linen style. The
`@partial(jax.jit, static_argnums=(0))` on `loss` is correct for the d2l
Trainer contract. The `cross_entropy` redefined inside `loss` (for purity)
is explained in the `:begin_tab:jax` note, which is good.

`jnp.take_along_axis` + `expand_dims` + `squeeze` is more verbose than the
PyTorch indexing equivalent. The explanation in the prose (ll. 221–229)
only covers the PyTorch/MXNet index form; it would help to add one sentence
noting the JAX equivalent, or a comment in the cell.

No correctness issues.

### TensorFlow

Uses `tf.gather(..., batch_dims=1)` for the indexing step — idiomatic for
TF. `tf.Variable` init is standard. No issues.

### MXNet

**P2 noise:** The `prediction-1` cell (and `prediction-2`) emit a massive
stderr wall:
```
Storage type fallback detected: operator = stack
...
WARNING: Execution of the operator above will fallback to generic implementation...
```
This is a benign MXNet oneDNN warning (unrelated to correctness), but it
floods the notebook output and confuses learners. The standard fix is to set
`MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0` at the top of the cell or in the
environment, or to note in the prose that this warning is expected and
harmless.

**MXNet is archived (ASF retirement 2023).** The present file includes MXNet
as a co-equal framework tab. Per the review guide, flag this: the MXNet tab
here should eventually be de-emphasized or dropped (it still executes
correctly, as evidenced by the output manifest, but is a dead framework for
new users). This is a book-wide decision, not this file's call alone.

### Cross-framework consistency & d2l conventions

- **One imports cell per framework** at the top — correct.
- **No `#@save` abuse** — `cross_entropy` is `#@save` which is correct (it's
  reused by the d2l library).
- **Stable cell IDs** — all cells have stable `#`-prefixed IDs.
- **`d2l.reduce_sum`, `d2l.reduce_mean`, `d2l.log`, `d2l.matmul`** used
  consistently — good abstraction.
- **JAX `jax.random.uniform` in `the-softmax-3`** vs `d2l.rand` in other
  frameworks — unavoidable (JAX requires explicit key); clean as-is.
- The PyTorch/MXNet/TF cell for cross-entropy-loss-1 uses
  `%%tab mxnet, pytorch, jax` together, but the actual JAX indexing uses
  `take_along_axis` which is different from `y_hat[[0,1], y]`. This is fine
  because it is in a single tab group, but the *prose* description of "using
  `y` as the indices" describes the non-JAX version. The JAX cell shows the
  same result but through different code — the prose should acknowledge this
  briefly (even "JAX uses `jnp.take_along_axis`; see the JAX tab" would
  suffice).

---

## 4. Implementation spec

### SMS-1 — Rewrite Summary section as Summary and Discussion  ·  [P0] · M · authored

- **Type:** teaching / coverage
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  anchor: `## Summary` (l. 402), verbatim: `"By now we are starting to get some
  experience\nwith solving linear regression\nand classification problems."`
- **Change:** Replace the entire `## Summary` block (ll. 402–411) with the
  following drafted section:

```markdown
## Summary and Discussion

In this section we built softmax regression entirely from scratch: the softmax
operation, the cross-entropy loss, parameter initialization, the forward pass, and
training on Fashion-MNIST. Breaking each piece open by hand is the purpose — once
you have seen these five moving parts separately, the one-liner in
:numref:`sec_softmax_concise` is not magic but notation.

**What the training curve tells you.** After 10 epochs with minibatch SGD the
model converges to roughly 83% validation accuracy. That ceiling is not a
hyperparameter problem — it is the limit of linear separability on Fashion-MNIST.
The ten classes are not linearly separable in pixel space (shirts and pullovers
look nearly identical to a linear model). The misclassification visualization at
the end of the section makes this concrete. Replacing the flat linear layer with
even a single hidden layer (Chapter 5) will push past 87%.

**Why the clip is only a band-aid.** Our `cross_entropy` clips the softmax output
away from zero before taking the log. This prevents the worst NaN failures, but
the naive `softmax` function itself can overflow for large logits (`exp(100)` is
infinity in float32). The right fix — subtracting $\max_k o_k$ before
exponentiating, then fusing softmax and log into a single numerically stable
operation — is derived in :numref:`subsec_softmax-implementation-revisited`. The
concise implementation applies that fix automatically; always use the framework's
built-in cross-entropy when not explicitly studying the internals.
```

- **Touches:** none (prose-only change, no build step).
- **Done when:** `## Summary` heading reads `## Summary and Discussion`, the
  four-sentence stub is replaced by the drafted three-paragraph text, `make html`
  renders clean, and the `:numref:` link to
  `subsec_softmax-implementation-revisited` resolves.
- **Depends on:** none.

---

### SMS-2 — Add `:numref:` forward pointer after clipping note  ·  [P1] · S · mechanical

- **Type:** coverage / correctness
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  anchor: verbatim text ending at "...without changing its mathematical form."
  (l. 300–301, the paragraph beginning "Note that we clip $\hat{y}$ away from
  zero...").
- **Change:** Append the following sentence to that paragraph (after the final
  period on "...without changing its mathematical form."):

  old → new (append to end of the paragraph):

  ```
  old: "...without changing its mathematical form."
  new: "...without changing its mathematical form. The proper fix — fusing softmax and cross-entropy via the log-sum-exp trick — is derived in :numref:`subsec_softmax-implementation-revisited`."
  ```

- **Touches:** none.
- **Done when:** The paragraph ends with the `:numref:` link and that link
  resolves to the "Softmax Revisited" subsection of `softmax-regression-concise.md`
  in the rendered HTML. Verify with `make html`.
- **Depends on:** none (the label `subsec_softmax-implementation-revisited`
  already exists in `softmax-regression-concise.md` l. 141).

---

### SMS-3 — Add MLE grounding sentence before cross-entropy code  ·  [P1] · S · authored

- **Type:** teaching / prose
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  anchor: `## The Cross-Entropy Loss` section, immediately before "Recall that
  cross-entropy takes the negative log-likelihood..." (l. 219).
- **Change:** Insert one sentence between "This may be the most common loss
  function\nin all of deep learning." and "At the moment, applications of deep
  learning...":

  ```
  old: "This may be the most common loss function\nin all of deep learning.\n\nAt the moment, applications of deep learning"
  new: "This may be the most common loss function\nin all of deep learning.\nRecall from :numref:`subsec_softmax-regression-loss-func` that minimizing cross-entropy is equivalent to maximizing the log-likelihood of the correct labels under our categorical model — it is the natural loss for classification.\n\nAt the moment, applications of deep learning"
  ```

- **Touches:** none.
- **Done when:** The sentence appears in the rendered HTML immediately after
  "most common loss function in all of deep learning." and the `:numref:` link
  resolves to the Loss Function subsection of `softmax-regression.md`.
- **Depends on:** none.

---

### SMS-4 — Add overall accuracy line to prediction section  ·  [P1] · S · authored

- **Type:** teaching / code
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  anchor: cell `#softmax-regression-scratch-prediction-1` (all framework tabs),
  and the prose paragraph "We are more interested in the images we label
  *incorrectly*." (l. 388).
- **Change:** Before the "We are more interested in the images we label
  incorrectly" paragraph, insert a brief note that reports the overall test
  accuracy, and a prose sentence that names the ~83% ceiling. Drafted prose
  insert:

  ```
  old: "We are more interested in the images we label *incorrectly*."
  new: "The overall test accuracy is approximately 83%, consistent with the training curve — the ceiling of a linear model on Fashion-MNIST. We are more interested in the images we label *incorrectly*."
  ```

  Additionally, add a short code cell after `prediction-1` (before `prediction-2`)
  that computes and prints the scalar accuracy:

  ```python
  # All-framework cell (no %%tab needed if d2l.Classifier.accuracy is abstract)
  # For pytorch tab:
  ```

  Since `d2l.Classifier` already tracks val_acc in the Trainer, the simplest
  approach is to just print `trainer.train_batch_idx` or re-use the final epoch's
  accuracy from the plot. **Judgment call for the implementing agent:** if
  `d2l.Classifier` exposes an `accuracy()` method taking a dataloader, use it;
  otherwise add a one-liner that computes `(preds == y).float().mean()` on the
  full val set. The goal is a single printed scalar in the output (e.g.,
  `"Test accuracy: 0.832"`).

- **Touches:** none (prose + optional small code cell; no library change required).
- **Done when:** The rendered HTML shows a printed accuracy scalar before the
  misclassification grid, and the prose notes the ~83% ceiling.
- **Depends on:** none.

---

### SMS-5 — Suppress MXNet storage-fallback warning in prediction cells  ·  [P2] · S · mechanical

- **Type:** code / currency
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  cell `#softmax-regression-scratch-prediction-1`, `%%tab mxnet` variant
  (l. 381–386).
- **Change:** Add `import os; os.environ['MXNET_STORAGE_FALLBACK_LOG_VERBOSE'] = '0'`
  before the prediction call in the MXNet tab, or add a prose note that the
  warning is benign. Preferred: add the env-var suppression to the MXNet
  imports cell at the top (cell
  `#softmax-regression-scratch-softmax-regression-implementation-from-scratch`,
  `%%tab mxnet`):

  ```
  old (mxnet imports cell):
  from d2l import mxnet as d2l
  from mxnet import autograd, np, npx, gluon
  npx.set_np()

  new:
  import os
  os.environ.setdefault('MXNET_STORAGE_FALLBACK_LOG_VERBOSE', '0')
  from d2l import mxnet as d2l
  from mxnet import autograd, np, npx, gluon
  npx.set_np()
  ```

- **Touches:** Re-capture MXNet outputs (`make -B _notebooks/mxnet/.../softmax-regression-scratch.executed && make capture-outputs FILES=chapter_linear-classification/softmax-regression-scratch.md`).
- **Done when:** The MXNet output manifest for `prediction-1` no longer contains
  the `Storage type fallback detected` stderr block.
- **Depends on:** none.

---

### SMS-6 — Fix Exercise 2 to be a genuine discovery exercise  ·  [P2] · S · authored

- **Type:** teaching / exercises
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  Exercise 2 (ll. 419–424), verbatim anchor: "Implement a `cross_entropy`
  function that follows the definition of the cross-entropy loss function
  $-\sum_i y_i \log \hat{y}_i$."
- **Change:** Replace exercise 2 with a version that makes the equivalence
  between the one-hot sum form and the index form the *discovery*:

  ```
  old:
  1. Implement a `cross_entropy` function that follows the definition of the cross-entropy loss function $-\sum_i y_i \log \hat{y}_i$.
      1. Try it out in the code example of this section.
      1. Why do you think it runs more slowly?
      1. Should you use it? When would it make sense to?
      1. What do you need to be careful of? Hint: consider the domain of the logarithm.

  new:
  1. The cross-entropy loss is defined as $-\sum_j y_j \log \hat{y}_j$ where $\mathbf{y}$ is a one-hot vector.
      1. Show algebraically that for one-hot $\mathbf{y}$ this simplifies to $-\log \hat{y}_{y_{\text{true}}}$, the form used in this section's `cross_entropy`.
      1. Implement the sum form explicitly (loop over all $j$, multiply by the one-hot indicator) and verify it produces the same result as the indexing form.
      1. Which is faster in practice? Why?
      1. Both forms require $\hat{y}_j > 0$. What happens if a softmax output underflows to exactly zero? How does our `clamp(min=1e-12)` address this, and what is a more principled solution? (See :numref:`subsec_softmax-implementation-revisited`.)
  ```

- **Touches:** none.
- **Done when:** Exercise 2 reads as drafted above, with the `:numref:` link
  resolving in HTML.
- **Depends on:** none.

---

### SMS-7 — Add missing convergence/linear-ceiling exercise  ·  [P2] · S · authored

- **Type:** teaching / exercises
- **Where:** `chapter_linear-classification/softmax-regression-scratch.md` —
  after Exercise 5 (l. 428), before the discussion links.
- **Change:** Add a sixth exercise:

  ```
  6. Inspect the misclassification grid produced by the prediction code.
      1. Which pairs of Fashion-MNIST classes does the model most often confuse? Why would a linear model struggle to separate them?
      1. Train the model for 20 epochs instead of 10. Does the validation accuracy continue to improve, or does it plateau? What does this tell you about the capacity of a linear model on this dataset?
      1. What is the minimum number of parameters needed for a linear model that maps 784-dimensional inputs to 10 classes? How does this compare to the number of training examples (60,000)?
  ```

- **Touches:** none.
- **Done when:** Exercise 6 appears before the discussion links in the rendered HTML.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The cross-entropy indexing trick explanation (ll. 219–229).** The two-example
  walkthrough (`y = [0, 2]`, `y_hat = [[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]`,
  picking out `y_hat[[0,1], y]`) is exactly right — concrete, minimal, and
  immediately shows what the code does before the code appears. Do not
  over-generalize or replace with abstraction.

- **The numerical-instability warning (l. 89).** Recent, honest, and appropriately
  calibrated: the naive softmax is presented for clarity, the warning is present,
  and the clip in `cross_entropy` is a documented workaround. This is better than
  most textbooks that ignore the issue entirely. Strengthen it with a `:numref:`
  (SMS-2) but do not remove it.

- **The clip comment in `cross_entropy` (ll. 262–264, all frameworks).** Providing
  the clip as an explicit, commented fix (rather than silently adding 1e-9) is
  pedagogically correct. The comment "Tiny clip to keep log finite when softmax
  outputs underflow to 0" is clear and accurate.

- **The JAX `loss` method re-defining `cross_entropy` locally for purity, with
  the explanation in `:begin_tab:jax`.** This is the right thing to do for JAX
  and the explanation is honest about *why* without burdening non-JAX readers.

- **The misclassification visualization (`prediction-2`).** Showing the errors
  (not just the accuracy) with "actual / predicted" labels is a teaching win —
  it makes the model's failure mode tangible. Do not remove; extend with prose
  (see SMS-4 and SMS-7).

- **The scaffold reuse pattern.** Opening Training with "We reuse the `fit`
  method defined in :numref:`sec_linear_scratch`" and running the exact same
  Trainer loop as in linear regression is the whole point of the chapter's
  architecture. This continuity is a genuine strength.
