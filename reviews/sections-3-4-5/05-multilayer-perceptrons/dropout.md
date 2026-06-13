# Review — chapter_multilayer-perceptrons/dropout.md  (§5.6 "Dropout")

**Role in the chapter:** Introduces dropout as the chapter's second explicit regularizer
(after weight decay), motivating it via noise-as-smoothness (Bishop 1995) and
co-adaptation, deriving inverted-dropout scaling, and demonstrating scratch + concise
implementations on Fashion-MNIST. It is the penultimate content section before the
Kaggle capstone.

**Verdict:** The mechanics are correct and clearly explained — inverted dropout is derived
properly, the $1/(1-p)$ rescaling is motivated with $\mathbb{E}[h'] = h$, and the
train-vs-test switch is handled correctly in all four frameworks. The main gaps are a
threadbare Summary (3 lines, no forward pointers), absence of the ensemble-of-subnetworks
intuition (the third canonical view from Srivastava et al.), no mention of where dropout
fits in 2026 (less central than it was in 2014), a buried and pedagogically costly
long-comment kludge in the JAX scratch implementation, and a flat two-section
structure that lacks the `###` articulation the rest of the chapter uses. MXNet should
be flagged as archived.

**Grade:** B — Assignable, and the core teaching is sound, but the Summary is
embarrassingly thin, the ensemble view is missing, and the currency note is absent.
A top-program text should not end with "dropout is yet another tool."

---

**Top priorities (ranked):**
1. [P0] DROP-1 — Expand the Summary into a real Discussion with the ensemble view, forward pointers, and a currency note.
2. [P1] DROP-2 — Add the ensemble/subnetwork interpretation to the intuition section (in-scope, in the original paper).
3. [P1] DROP-3 — JAX scratch: replace the mutable-default key hack with a proper `key` argument; move the explanation from a 10-line inline comment to a short prose paragraph after the cell.
4. [P1] DROP-4 — Flag MXNet as archived (one-line note in the MXNet tab).
5. [P2] DROP-5 — Add an exercise on MC Dropout (uncertainty estimation); shorten exercise 7.
6. [P2] DROP-6 — Flatten the prose heading structure: add one `##` section break to separate the intuition/theory from the scratch implementation.

---

## 1. Coverage

### Add

**Ensemble / subnetwork interpretation (P1, in-scope).** The original paper's third view —
that training with dropout is equivalent to training an exponential ensemble of $2^n$
weight-sharing thinned networks, with test time averaging them — is the most cited
modern framing of *why* dropout works. The chapter presents two views (noise/Tikhonov and
co-adaptation) but entirely omits the ensemble view that every current reference
(Goodfellow Ch.7, CS231n, Nielsen Ch.3, Srivastava et al. §5) covers. It belongs in the
"Dropout in Practice" section as a third bullet or paragraph. (See DROP-2.)

**Currency: where dropout stands in 2026 (P0, one paragraph in Summary).** Dropout was
transformative in 2014 for fully-connected vision networks. By 2026 the picture is more
nuanced: CNNs with BatchNorm rarely use dropout (BatchNorm provides similar regularization
via noise); large language models use it lightly (typically 0.0–0.1) or not at all in
their attention/FFN sublayers; dropout is most alive at final classifier heads (0.5 is a
standard recipe) and in transformer-era models where it is a tunable knob. None of this
appears in the chapter. The task prompt explicitly flags this as in-scope; the summary is
the right place for a two-sentence honest note + forward pointer to BatchNorm (later
chapter). (See DROP-1.)

**MC Dropout exercise (P2).** :citet:`Gal.Ghahramani.2016` showed that applying dropout
at test time and averaging many forward passes is a practical Bayesian approximation.
The text already mentions this exception ("some researchers use dropout at test time as a
heuristic for estimating uncertainty") but the existing exercises do not ask the student to
try it. A new exercise asking students to implement and compare MC Dropout against a
single forward pass would be concrete and teach both the uncertainty-estimation idea and
the train/eval mode switch. (See DROP-5.)

### Remove / trim

**JAX inline comment (P1).** The JAX `dropout_layer` function carries a 10-line
explanatory comment embedded in the function body explaining why the mutable-default key
pattern is a limitation of the from-scratch version. This is the right information but
the wrong place — inline comments inside a displayed code cell are pedagogically
counter-productive (they dominate the display and distract from the algorithm). The
comment belongs in the prose immediately following the cell, and the function signature
should be cleaned up. (See DROP-3.)

**Exercise 7 (P2 trim).** "Invent another technique for injecting random noise at each
layer that is different from the standard dropout technique. Can you develop a method
that outperforms dropout on the Fashion-MNIST dataset?" This is open-ended to the point
of being unanchored — what architecture, what budget, what metric? Either sharpen it
("DropConnect — apply dropout to individual *weights* rather than activations — was
proposed by Wan et al. 2013. Implement DropConnect and compare on Fashion-MNIST.") or
cut it in favor of the MC Dropout exercise. Note that exercise 6 already covers weight
masking, making exercise 7 partially redundant.

### Reorder / restructure

**No structural change needed**, but the two top-level sections ("Dropout in Practice"
and "Implementation from Scratch") and the flat run of `###` subsections under
Implementation could benefit from one added `##` section called "Why Dropout Works" (or
"Intuition") that lifts the Bishop/co-adaptation/ensemble discussion out of the preamble
and into its own named home. The current preamble before the first `##` is unusually long
(138 lines) and mixes background motivation with the formal definition. A `## Intuition`
heading before the preamble's second paragraph would improve navigability without
restructuring the content. (See DROP-6.)

---

## 2. Teaching quality

### Structure & flow

The section structure is: long preamble (no heading) → `## Dropout in Practice` →
`## Implementation from Scratch` with `### Defining the Model` / `### Training` →
`## Concise Implementation` → `## Summary`. This is workable but the long preamble is an
anomaly — 138 lines of unnested content before the first sub-`##` header. The
"Dropout in Practice" section (lines 139–167) reads as a continuation of the preamble
rather than a distinct section. The cleaner shape would be:

```
## Intuition and Theory
   (Bishop → co-adaptation → ensemble view → formal definition + formula)
## Dropout in Practice
   (where to place dropout, train vs. test, typical p values)
## Implementation from Scratch
   ### The Layer
   ### A Dropout MLP
   ### Training
## Concise Implementation
## Summary / Discussion
```

This requires more than a mechanical edit — it is a judgment restructuring. Flag as P2
but note that even without restructuring, adding a `## Discussion` section (DROP-1) is
P0.

### Figures

**`fig_dropout2` (present, correct).** The figure shows an MLP before and after dropout,
with two hidden units zeroed. It is clean, labeled, and the caption is accurate. The
reference at line 147 (`In :numref:`fig_dropout2`, $h_2$ and $h_5$ are removed`) is
correct. The figure carries the right idea.

**Missing: a figure showing the ensemble interpretation.** A schematic showing three
different thinned subnetworks being averaged at test time (or the exponential fan-out)
would make the ensemble view concrete. Given the scope map, this is a P2 addition — worth
doing but not essential if the ensemble view is present in text.

**No matplotlib figure-drawing code in notebook cells.** The chapter is clean on this
front — the training curve cells (`#dropout-training`, `#dropout-concise-implementation-3`)
produce SVG plots via `d2l.Trainer.fit` (data-result plots), which is appropriate.

### Prose & clarity

**Lines 56–72: sexual reproduction analogy.** The text says "we are imposing our own
narrative with the link to Bishop" and then describes the co-adaptation / sexual
reproduction analogy. This meta-commentary works but slightly undercuts the argument.
The sentence "While such a justification of this theory is certainly up for debate"
(line 69) is accurate but trails off without connecting to the ensemble view that
*is* on firmer theoretical ground. The fix is to add the ensemble view immediately after.

**Lines 106–107: "By design, the expectation remains unchanged, i.e., $E[h'] = h$."**
This is stated without proof. The proof is a one-liner:
$\mathbb{E}[h'] = p \cdot 0 + (1-p) \cdot \frac{h}{1-p} = h$.
Adding this inline makes the claim self-contained and teaches the key algebra students
need to understand why the rescaling factor is $1/(1-p)$ and not some other constant.
This is important — the "classic bug" is scaling by $(1-p)$ instead of $1/(1-p)$, or
forgetting to rescale at test time instead of train time (non-inverted dropout). The
proof makes the correct approach unforgettable. (See DROP-1 or as a small inline fix.)

**Lines 503–509: Summary.** Three lines, no forward pointers, no connection to later
material, no honest note about modern relevance. This is well below the standard set by
the rest of the chapter. The calculus.md and linear-algebra.md gold-standard files each
have 8–15 bullet-point summaries plus a "Discussion" or "Further Reading" block.

**Lines 163–167: test-time behavior.** The text correctly notes that dropout is off at
test time and briefly mentions MC Dropout as a heuristic for uncertainty. This is fine.
The one-sentence nod to :citet:`Gal.Ghahramani.2016` here (it currently lacks a
citation) would complete the gesture. (See DROP-5.)

### Exercises

Exercise 1 (compare dropout probabilities, swap layers) — good mechanical experiment.
Exercise 2 (increase epochs, compare dropout vs. no dropout) — fine but trivial.
Exercise 3 (variance of activations) — excellent, asks for a plot, teaches something.
Exercise 4 (why not dropout at test time) — conceptual, good.
Exercise 5 (dropout + weight decay interaction) — strong, asks for quantitative + qualitative.
Exercise 6 (dropout on weights vs. activations) — asking students to implement DropConnect; good.
Exercise 7 (invent new noise injection method) — see trim recommendation above.

The exercise set is above average. The main gap is the absence of an MC Dropout exercise
(see DROP-5). Exercise 2 could be sharpened: "Train without dropout and show the
train/test loss gap; overlay with the dropout run to make the regularization effect
visible."

---

## 3. Code & examples

### Does the code teach?

Yes. The scratch `dropout_layer` → scratch `DropoutMLPScratch` → concise `DropoutMLP`
progression teaches the right thing. The mask-then-rescale pattern is short and clear.
The exception is the JAX scratch function, whose 10-line inline comment overwhelms the
algorithm and should be moved to prose.

### PyTorch

**`#dropout-defining-the-model` (PyTorch):** `if self.training:` correctly uses
`nn.Module.training` (the standard flag; set by `.train()` / `.eval()`). Correct.

**`#dropout-concise-implementation-1` (PyTorch):** `nn.Dropout(dropout_1)` — correct,
idiomatic, uses `p` positional argument. The `nn.Sequential` layout is clean. No issues.

**Outputs (`#dropout-training`, `#dropout-concise-implementation-3`):** SVG assets
produced cleanly; sizes ~21.5 KB (consistent across frameworks), indicating successful
training convergence. No stray warnings in the text outputs.

### JAX

**`#dropout-implementation-from-scratch-1` (JAX):** The `key=d2l.get_key()` mutable
default argument is a pedagogical liability. The long comment (lines 219–230) explains
the issue but does so inside the function body — the worst possible location for
explanatory text in a displayed cell. The from-scratch function is only used to demo the
mechanics on a static input (lines 259–263); it never runs inside a training loop.
This is a significant teaching friction point. (See DROP-3.)

**`#dropout-defining-the-model` (JAX):** The `DropoutMLPScratch` class carries
`training: bool = True` as a class attribute (Flax struct-style). When `self.training`
is `True` the custom dropout_layer is called with the fixed key — so the training run
actually uses a deterministic mask across all calls. This is a correctness concern for
the *scratch training run*: the network trains with the same dropout mask every step,
which is not what the prose describes. The concern is mitigated because the scratch run
is intended only to show the concept before the concise version; the concise version
uses Flax's `nn.Dropout` correctly. But it should be flagged in prose rather than buried
in a function-body comment.

**`#dropout-concise-implementation-1` (JAX):**
`nn.Dropout(self.dropout_1, deterministic=not self.training)(x)` — the `deterministic`
kwarg is the correct Flax `linen` API (not deprecated in linen; it was renamed to `use`
in Flax NNX but linen still supports `deterministic`). The chapter uses linen throughout,
so this is correct for the present codebase. No issue.

**`#dropout-concise-implementation-2` (JAX):** The overridden `loss` function
passes `rngs={'dropout': state.dropout_rng}` — correct Flax linen RNG threading. The
prose explanation (lines 465–478) is the right level of detail for a JAX learner.

**Outputs:** SVG assets produced cleanly for both JAX training cells; sizes ~21 KB,
consistent with other frameworks.

### TensorFlow

**`#dropout-defining-the-model` (TF):** `if self.training:` — here `self.training` is a
`d2l.Classifier` attribute, presumably set during `trainer.fit`. This is correct as long
as the base class sets it appropriately. Since the concise version uses
`tf.keras.layers.Dropout` which handles training mode natively, the scratch version
works but demonstrates a subtlety not explained in prose: TF Keras Dropout checks the
`training` argument passed at call time, not a module attribute. The scratch version's
reliance on `self.training` (a d2l book convention) sidesteps Keras's built-in
mechanism, which is fine for pedagogy but worth a one-line note.

**`#dropout-concise-implementation-1` (TF):** `tf.keras.layers.Dropout(dropout_1)` in a
`tf.keras.models.Sequential` — idiomatic, correct. No issues.

### MXNet

**Status:** Apache MXNet was retired by the ASF and archived in 2023/2024. Presenting it
as a co-equal fourth framework is misleading to students in 2026. Every section in the
chapter should add a one-line note to the MXNet tab. (See DROP-4.)

**`#dropout-defining-the-model` (MXNet):** `if autograd.is_training():` — this is the
correct MXNet Gluon idiom for checking training mode and is consistent with how the base
class drives training. No correctness issue.

**`#dropout-implementation-from-scratch-1` (MXNet):** `np.random.uniform(0, 1, X.shape)
> dropout` — correct for `p ∈ (0,1)`. The `dropout == 0` case falls through to the
mask path: all values are `> 0`, so mask is all True, and `X / 1.0 == X`. Correct.
No short-circuit for `p == 0` (unlike PyTorch/JAX/TF which also lack one, as all handle
it correctly via the mask).

**Outputs:** All training SVG assets present, ~21.7 KB.

### Cross-framework consistency & d2l conventions

**Imports cell (`#dropout`):** All four frameworks have their imports in the shared
`#dropout` cell, correctly split by `%%tab`. No re-imports. Clean.

**`#dropout-implementation-from-scratch-2` (test demo):** The MXNet variant uses
`np.arange(16).reshape(2, 8)` (integer) while PyTorch uses `torch.arange(16,
dtype=torch.float32)`. This causes the MXNet version to produce integer outputs which
then get cast to float32 inside the function. Functionally correct but the dtype
inconsistency in the demo input is cosmetically distracting. Minor (P2).

**Unnecessary divergence:** The PyTorch scratch `DropoutMLPScratch.forward` calls
`X.reshape((X.shape[0], -1))` directly while TF uses `tf.reshape(X, (tf.shape(X)[0],
-1))` — framework-forced. No gratuitous divergence.

**Cell ID hygiene:** All cell IDs are stable, consistently named across tabs. Clean.

---

## 4. Implementation spec

### DROP-1 — Expand Summary into Discussion  ·  P0 · M · authored

- **Type:** coverage / prose / teaching
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — anchor `## Summary` (line 503)
- **Change:** Replace the existing 3-line Summary with the following drafted content:

```
## Summary

* **Inverted dropout** replaces each hidden activation $h$ with a random variable
  $h'$ that is zero with probability $p$ and $h/(1-p)$ otherwise.
  The rescaling by $1/(1-p)$ keeps $\mathbb{E}[h'] = h$, so the network's
  expected behaviour at test time matches training without any change to the
  test-time code.
* Dropout is **off at test time**: the full network runs; there is no masking
  and no rescaling.
* Three complementary views explain why dropout helps:
  (1) *noise injection*: training with random zeroing is equivalent to Tikhonov
  regularization on the function (Bishop 1995);
  (2) *anti-co-adaptation*: no hidden unit can rely on any specific partner, so
  each unit learns broadly useful features;
  (3) *implicit ensemble*: each training step trains a different thinned
  subnetwork, and the full-network test evaluation approximates averaging $2^n$
  such subnetworks :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`.
* As of the early 2020s, dropout is less central than it was in the 2014–2017
  era: convolutional networks typically replace it with batch normalization
  (see :numref:`sec_batch_norm`), and large transformer-based language models
  use it lightly (rate 0.0–0.1) or not at all in their core layers, reserving
  it for final classifier heads.  It remains a reliable, cheap regularizer that
  combines well with weight decay and data augmentation.
```

- **Touches:** none (prose only).
- **Done when:** `make html` clean; the Summary section renders with 5 bullets as
  drafted; the citation `:cite:`Srivastava.Hinton.Krizhevsky.ea.2014`` resolves in
  the HTML reference list; `sec_batch_norm` cross-reference resolves.
- **Depends on:** none.

---

### DROP-2 — Add ensemble/subnetwork intuition view  ·  P1 · S · authored

- **Type:** coverage / prose
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — anchor at line 73:
  `The key challenge is how to inject this noise.`
  Insert the new paragraph *before* that line (i.e., after the co-adaptation paragraph
  ending at line 72).
- **Change:** Insert the following paragraph between line 72 and line 74:

```
A third, perhaps more illuminating, perspective is due to
:citet:`Srivastava.Hinton.Krizhevsky.ea.2014` themselves.
Dropout with $n$ hidden units implicitly trains an ensemble of $2^n$
*thinned* networks that share weights.
On each training step, a different subset of units is active,
so the update moves the shared weights in the direction that helps
one particular thinned network.
At test time, running the full network with weights scaled down by $1-p$
is an approximation to averaging the predictions of all $2^n$ networks.
This ensemble view explains why dropout tends to reduce variance:
it is cheap model averaging.
```

- **Touches:** none.
- **Done when:** `make html` clean; paragraph renders in HTML; citation resolves;
  the ensemble view appears before the "key challenge" paragraph.
- **Depends on:** none.

---

### DROP-3 — Fix JAX scratch dropout_layer: move explanation to prose  ·  P1 · M · authored

- **Type:** code / teaching
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — cell `#dropout-implementation-from-scratch-1`, JAX tab (lines 216–234).
- **Change:** Two-part edit.

**Part A** — Replace the JAX `dropout_layer` function body (remove the 10-line comment,
require `key` as a positional argument with no default):

Old:
```python
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    # Note: `key` is bound at function-definition time (mutable default
    # pattern), so this educational from-scratch dropout uses one fixed
    # key for all calls — i.e. the dropout mask is identical on every
    # call, which is *not* what real training wants. That keeps the
    # function JIT-traceable — calling `d2l.get_key()` at call time would
    # mutate `d2l._master_key` and leak a tracer when invoked inside a
    # JIT'd loss. Real per-step dropout instead threads a *fresh* key for
    # each call, e.g. derive one deterministically with
    # `jax.random.fold_in(key, step)`, or — better — let Flax's
    # `nn.Dropout` handle it, which pulls a new key per step via
    # `rngs={"dropout": ...}`.
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

New:
```python
%%tab jax
def dropout_layer(X, dropout, key):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

**Part B** — Update the test cell `#dropout-implementation-from-scratch-2` (JAX tab)
to pass an explicit key:

Old:
```python
%%tab jax
X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

New:
```python
%%tab jax
X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
key = d2l.get_key()
print('dropout_p = 0:', dropout_layer(X, 0, key))
print('dropout_p = 0.5:', dropout_layer(X, 0.5, key))
print('dropout_p = 1:', dropout_layer(X, 1, key))
```

**Part C** — Update the `DropoutMLPScratch` JAX `forward` to pass a key. Since the
scratch model is only used to demo the concept (not for rigorous training), use
`jax.random.PRNGKey(0)` or `d2l.get_key()` at call time:

Old `forward` body for JAX (lines 364–371):
```python
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

New:
```python
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1, d2l.get_key())
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2, d2l.get_key())
        return self.lin3(H2)
```

**Part D** — Add a short prose note immediately after the `#dropout-implementation-from-scratch-1` cell (before the test demo). Insert after the sentence ending "dividing the survivors by `1.0-dropout`." (line 186):

```
:begin_tab:`jax`
JAX functions must be *pure* — the same input must always produce the same output
so that JIT compilation and automatic differentiation remain well-defined.
Dropout therefore requires an explicit PRNG `key` argument; the function has no
internal state. In a real training loop one would derive a fresh key each step
(e.g., `jax.random.fold_in(train_key, step)`) or rely on Flax's `nn.Dropout`
which handles key threading automatically via `rngs={'dropout': ...}`.
:end_tab:
```

- **Touches:** The change to `dropout_layer` signature requires updating the two callers in `#dropout-implementation-from-scratch-2` (JAX tab) and `DropoutMLPScratch.forward` (JAX tab). No other files affected.
- **Done when:** `make html` clean; the JAX function body is 5 lines without any
  inline comment; the prose tab note appears in the rendered page; the JAX demo cell
  output is unchanged (still shows p=0 identity, p=0.5 partial zero, p=1 all-zero).
- **Depends on:** none.

---

### DROP-4 — Flag MXNet as archived  ·  P1 · S · mechanical

- **Type:** currency
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — each MXNet tab that has a
  `:begin_tab:`mxnet`` block. There is none currently (the file uses `%%tab mxnet`
  inline but no explicit begin_tab note). Add a single `:begin_tab:`mxnet`` note after
  the concise implementation cell.

  Specifically, add after the `:begin_tab:`jax`` block ending at line 478
  (after the JAX loss-override note) a new parallel note:

```
:begin_tab:`mxnet`
**Note:** Apache MXNet was retired by the Apache Software Foundation in 2023 and
its repository archived. The MXNet tab is preserved for reference but is no longer
actively maintained. For new projects, use PyTorch or JAX.
:end_tab:
```

- **Touches:** none.
- **Done when:** `make html` clean; the rendered MXNet tab shows the retirement note
  in the Concise Implementation section.
- **Depends on:** none.

---

### DROP-5 — Add MC Dropout exercise + sharpen exercise 2  ·  P2 · S · authored

- **Type:** teaching / exercises
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — anchor `## Exercises` (line 511).
- **Change:**

Replace exercise 2 (line 514):

Old:
```
1. Increase the number of epochs and compare the results obtained when using dropout with those when not using it.
```

New:
```
1. Train the same architecture without dropout for the same number of epochs.
   Plot the train and test loss curves for both runs on the same axes.
   How wide is the train–test gap with and without dropout?
```

Add a new exercise after exercise 4 (after the current line 516 "Why is dropout not typically used at test time?"):

```
1. **MC Dropout.** At test time, instead of disabling dropout, run $T=20$ forward
   passes and average the softmax outputs.  Compare accuracy and calibration
   (expected confidence vs. actual accuracy) against the standard one-pass
   evaluation.  How does MC Dropout relate to ensemble methods?
   (See :citet:`Gal.Ghahramani.2016`.)
```

- **Touches:** Requires adding `Gal.Ghahramani.2016` to the BibTeX references if not
  already present. Check `references.bib` (or the book's central bib file).
- **Done when:** `make html` clean; the new exercise appears between exercises 4 and 5
  (renumbering subsequent exercises); the citation resolves.
- **Depends on:** none.

---

### DROP-6 — Add inline expectation proof  ·  P2 · S · mechanical

- **Type:** prose / teaching
- **Where:** `chapter_multilayer-perceptrons/dropout.md` — line 106:
  `By design, the expectation remains unchanged, i.e., $E[h'] = h$.`
- **Change:**

Old:
```
By design, the expectation remains unchanged, i.e., $E[h'] = h$.
```

New:
```
By design, the expectation remains unchanged, i.e., $E[h'] = h$,
since $p \cdot 0 + (1-p) \cdot \frac{h}{1-p} = h$.
This is why we divide by $1-p$ and not by any other constant:
it is the unique factor that restores the original expected value.
```

- **Touches:** none.
- **Done when:** `make html` clean; the inline derivation appears immediately after the
  formula.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **Inverted-dropout derivation and formula.** The display equation (lines 96–105) and
  the statement that `E[h']  = h` are correct and clearly positioned. This is the classic
  bug surface and the chapter gets it exactly right.

- **Train-vs-test switch in all four frameworks.** PyTorch uses `self.training`,
  MXNet uses `autograd.is_training()`, TF uses `self.training`, JAX uses a struct
  attribute — all are correct and each is the idiomatic choice for its framework.
  The symmetry across frameworks is clean and the concise version further reinforces
  that the framework's built-in Dropout layer handles this automatically.

- **Bishop 1995 link.** The opening motivation via noise-as-smoothness/Tikhonov
  regularization (lines 33–39) is an intellectually elegant framing that most
  treatments skip. Keep it.

- **Scratch → concise pairing.** The two-section structure (from-scratch then high-level
  API) is the pedagogically correct approach for dropout: seeing the mask-then-rescale
  logic before seeing `nn.Dropout(p)` makes the latter meaningful rather than magic.

- **JAX concise implementation with `dropout_rng` threading.** The prose explanation
  of why JAX requires explicit RNG keys (lines 465–478) and the overridden `loss`
  function (cell `#dropout-concise-implementation-2`) are unusually careful and accurate.
  This is the right amount of detail for a learner encountering JAX's functional random
  number model for the first time.

- **Co-adaptation discussion.** Acknowledging the original paper's sexual-reproduction
  analogy and then honestly noting "such a justification is certainly up for debate"
  (line 69) is intellectually honest and sets a good tone for how to read research
  claims. Keep this voice.

- **Test-time uncertainty note.** The one-sentence mention of using dropout at test time
  for uncertainty estimation (lines 163–167) is a valuable forward-pointer. It just needs
  a citation to :citet:`Gal.Ghahramani.2016`.
