# Review — chapter_linear-regression/synthetic-regression-data.md  (§3.3 "Synthetic Regression Data")

**Role in the chapter:** Introduces the `SyntheticRegressionData` class — a reusable `DataModule` subclass that generates a ground-truth linear dataset — and teaches the data-loading pipeline by first rolling a hand-written minibatch iterator, then replacing it with framework built-ins. This class is imported by every downstream section in the chapter; the page is infrastructure as much as exposition.

**Verdict:** Functionally complete and mechanically correct. The code works across all four frameworks, the `DataModule` subclass pattern is clean, and the JAX PRNGKey and `drop_remainder` handling are genuinely good. The page falls short of the best-textbook bar on three fronts: (1) the opening motivation is weak — it states "known parameters → can check recovery" but never sells *why that matters* as a scientific discipline; (2) the "ex post facto OOP" tangent is a distraction from the data-pipeline story; (3) the undocumented `len()` discrepancy (JAX returns 31, all others 32) will puzzle every reader, yet goes unmentioned in prose. The Summary wanders into composition/pipeline generalities and misses the point of the section.

**Grade:** B− — assignable as a reading, but a curious student will be confused by the batch-count mismatch, uninspired by the motivation, and left with a summary that doesn't tell them what they actually learned.

**Top priorities (ranked):**
1. [P0] SD-1 — Document and explain the JAX `len()` discrepancy (31 vs 32 batches)
2. [P1] SD-2 — Sharpen the opening motivation: *why* synthetic data is scientifically valuable
3. [P1] SD-3 — Cut the "ex post facto" OOP tangent; tighten the efficiency paragraph
4. [P1] SD-4 — Rewrite the Summary to reflect what the section actually taught
5. [P1] SD-5 — Add a missing exercise: vary noise σ and check parameter recovery
6. [P2] SD-6 — Note MXNet archival status with a one-line callout
7. [P2] SD-7 — Rename the local `noise` variable in all four tabs to avoid shadowing the hyperparameter
8. [P2] SD-8 — Add a missing exercise: seeding for reproducibility across frameworks

---

## 1. Coverage

### Add

**Why synthetic data — the scientific argument (P1).** The opening paragraph's motivation is accurate but thin: "we can check that our model can in fact recover them." A top-tier treatment (Murphy §2.5, Bishop §1.1, CS229 Lecture 1) makes a stronger point: synthetic data is the only setting where we can distinguish *algorithm failure* from *model mismatch* from *data pathology* — in real data, all three are confounded. The key pedagogical value is **identifiability**: we know the data-generating process exactly, so a failure to recover `w` indicts the optimizer or implementation, full stop. One or two sentences on this would transform the paragraph.

**Noise σ and signal-to-noise ratio (P1 as an exercise, not a section).** The file fixes σ = 0.01 with no discussion of what that implies. Students benefit from understanding that small σ makes recovery easy, large σ makes it hard. This belongs as an exercise (SD-5 below), not new prose.

**`drop_remainder` is documented in a code comment only (P0 as a prose gap).** The JAX `get_tensorloader` uses `drop_remainder=train`, which causes `len(data.train_dataloader())` to return 31 instead of 32 for JAX (the final 8-example partial batch is dropped). This is a deliberate and correct choice for JIT compilation stability, but it is explained *only* in a multi-line inline comment inside a code cell. The prose after `len(data.train_dataloader())` says nothing about it, so the JAX reader sees an unexplained numerical difference. This is the file's most confusing omission.

### Remove / trim

**"Ex post facto" OOP tangent (P1).** Lines 241–247: "Note that we added a method to the `SyntheticRegressionData` class *after* creating the `data` object. Nonetheless, the object benefits from the *ex post facto* (after-the-fact) addition of functionality to the class." This is a Python metaprogramming curiosity that was explained in `oo-design.md` and does not belong here as the main observation after a data loading demonstration. It breaks the flow from "hand-rolled iterator works" → "but here is why the framework version is better." Cut entirely; replace with a single sentence linking to the efficiency argument.

**Wordy Summary (P1).** The current Summary (lines 349–369) talks at length about data loader composition for image pipelines and the "two-dimensional linear model" being simple. Neither connects back to what this section actually taught: that we know the ground truth, that the `DataModule` pattern abstracts the data contract, and that the framework DataLoader is preferred for all real use. The whole Summary should be replaced (SD-4).

### Reorder / restructure

The three-subsection structure (Generate → Hand-roll → Concise) is correct and well-ordered. No reordering needed. The only structural issue is that the efficiency argument at lines 250–259 is mid-section prose rather than a clean transition, and the "ex post facto" tangent interrupts it; trimming SD-3 resolves this.

---

## 2. Teaching quality

### Structure & flow

The spine is: motivation → generate data (class) → hand-roll iterator → concise (framework) loader → summary. This is logical. The section is appropriately lean for a "utility" file. The main flow interruption is the OOP tangent at line 241 (see SD-3).

The transition between the hand-rolled and concise sections is undercut by a passive "Next let's try to implement the same method using built-in iterators" — a stronger connector would name the specific deficiencies being solved (all-in-memory, Python-level index looping, no prefetch, no worker parallelism).

### Figures

No figures exist in this section. This is acceptable — it is a data-pipeline utility section, not a concepts section. One optional figure that *would* help (but is not P0/P1 here): a diagram of the `DataModule` / `get_dataloader` / `train_dataloader` call chain. However, this belongs in `oo-design.md` if it exists anywhere; do not add it here.

### Prose & clarity

**Line 9–18 (intro paragraph):** "Machine learning is all about extracting information from data." — this is a generic opener. The section should open directly on the scientific utility of synthetic data. See SD-2.

**Line 77–80:** "Note that for object-oriented design we add the code to the `__init__` method of a subclass of `d2l.DataModule` (introduced in :numref:`oo-design-data`). It is good practice to allow the setting of any additional hyperparameters. We accomplish this with `save_hyperparameters()`. The `batch_size` will be determined later." — This is four sentences about OOP bookkeeping that interrupts the explanation of the data generation equation. The cross-reference to `oo-design-data` is correct and useful; the other three sentences should be trimmed to a parenthetical or dropped (the pattern was already taught in §3.2).

**Lines 241–247:** See SD-3 (ex post facto tangent — cut).

**Lines 250–259 (efficiency argument):** Correct and useful, but verbose. "it requires that we load all the data in memory and that we perform lots of random memory access" — the second point is imprecise; the issue is Python-level iteration overhead and absence of prefetching/pipelining, not random-access memory cost per se. SD-3 drafts a tighter replacement.

**Lines 339–346 (after `len()` call):** No prose explains the JAX discrepancy. This is the P0 gap (SD-1).

### Exercises

The four current exercises are uneven in quality:

- Ex 1 (partial-batch behavior): Solid mechanical exercise. Keep.
- Ex 2 (huge dataset, disk-based shuffle): Good systems exercise; the pseudorandom permutation hint is genuinely interesting. Keep.
- Ex 3 (on-the-fly generator): Good generalization exercise. Keep.
- Ex 4 (reproducible random generator): Weak — this is answered immediately (set a seed), and the answer for each framework is a one-liner. The interesting version is: *why* does seeding work the way it does in JAX (functional PRNG) vs. PyTorch/TF (global state)? Elevate to this version (SD-8).

**Missing exercise (P1, SD-5):** Vary `noise` (σ) from 0.001 to 1.0 and train a linear model. How does recovery of `w` degrade as σ grows? This is the key point of synthetic data — it lets you trace the relationship between signal quality and estimation quality — yet no exercise asks students to explore it.

---

## 3. Code & examples

### Does the code teach?

Yes, broadly. The `SyntheticRegressionData.__init__` is compact and clear. The hand-rolled `get_dataloader` is a clean 8-line generator that teaches index shuffling and batch slicing. The `get_tensorloader` is a minimal framework-native wrapper. No cell draws figures or performs unrelated computations. Code cells teach what the surrounding prose claims they teach.

Minor: the local variable `noise` shadows the `noise` hyperparameter in all four tabs (e.g. `noise = d2l.randn(n, 1) * noise`). This is a Python anti-pattern (right-hand side reads the parameter, left-hand side creates a new local tensor — fine at runtime but confusing on first read). Rename to `eps` or `epsilon` across all tabs (SD-7).

### PyTorch

Clean, idiomatic. `torch.utils.data.TensorDataset` + `DataLoader` with `shuffle=train` is the canonical modern pattern (unchanged from PyTorch 1.x to 2.x). `d2l.randn`, `d2l.tensor`, `d2l.matmul`, `d2l.reshape` are d2l shims — no issues. No seed set, but this is a known limitation of the section (not a PyTorch-specific bug).

One minor gap: `DataLoader` is called without `num_workers` or `pin_memory`. Since `DataModule.__init__` already stores `num_workers=4` (from `oo-design.md`), `get_tensorloader` silently ignores it. This is worth a one-line note in prose or passing `num_workers=self.num_workers` (book-wide decision; flag, don't fix here).

Outputs: sensible. `len()` = 32 (correct, ⌈1000/32⌉ = 32 but actually 1000/32 = 31.25 → 32 batches with a partial last batch). The first example features `[-0.9261, -0.0554]` and label `[2.5330]` are plausible given w=[2, -3.4], b=4.2.

### JAX

The JAX tab is the most thoughtfully written in this file. Explicit PRNGKey handling with `jax.random.split` is correct functional-PRNG practice. The `key=None` default (resolved to `PRNGKey(0)` inside `__init__`) is documented by an inline comment explaining it stays deterministic while allowing overrides — this is good JAX pedagogy.

The `drop_remainder=train` in `get_tensorloader` is correct (prevents recompilation on partial final batch) and explained in a detailed comment. However, the comment is inside the code cell, while the observable consequence (len = 31 vs 32) appears in a separate cell with no prose — this is the P0 gap (SD-1).

Using `tensorflow_datasets.as_numpy(tf.data.Dataset...)` as the JAX dataloader is the standard pattern (JAX doesn't bundle a data loader); the `import tensorflow as tf; import tensorflow_datasets as tfds` in the JAX imports cell is correct.

Outputs: `len()` = 31 (expected, due to `drop_remainder=train`).

### TensorFlow

The TF tab uses `tf.data.Dataset.from_tensor_slices(...).shuffle(...).batch(self.batch_size)`. This is idiomatic TF 2.x / Keras 3. The `shuffle_buffer = tensors[0].shape[0]` (full-dataset shuffle buffer) is correct for small in-memory datasets and produces uniform shuffling. For large datasets a fraction of N is conventional, but at 1000 samples this is fine.

Notably, TF tab does NOT use `drop_remainder`, so the last batch of 8 examples IS included: `len()` = 32. This is correct behavior (TF handles variable-batch shapes without recompilation by default). No issues.

Outputs: `len()` = 32, expected.

### MXNet

`gluon.data.ArrayDataset` + `gluon.data.DataLoader` is idiomatic MXNet Gluon. Technically correct.

**Currency issue (P2, SD-6):** Apache MXNet was archived by the ASF in September 2023 and is no longer actively maintained. The text presents it as a co-equal current framework. The book-wide decision on MXNet is outside this file's scope, but at minimum a callout note or footnote should flag archival status here, as this is the first file where students see `gluon.data` code. The suggested language is a short `:begin_tab:`mxnet`` note (identical to what should appear in every MXNet-first-appearance file in the chapter).

### Cross-framework consistency & d2l conventions

**Gratuitous divergence — none.** All four `get_dataloader` tabs are essentially identical (Python stdlib `random.shuffle` + index slicing + `yield`). The only meaningful framework differences are: JAX takes a `key` argument in `__init__`, and the `get_tensorloader` differs appropriately by framework. This is disciplined.

**d2l conventions — clean.** One imports cell per framework at the top (cell ID `synthetic-regression-data`). `#@save` decorators on `SyntheticRegressionData.__init__`, `get_tensorloader`, and the concise `get_dataloader` are correct (these are all reused downstream). No re-imports inside later cells.

**`num_workers` not passed to `DataLoader` (PyTorch/MXNet).** `DataModule.__init__` stores `num_workers=4` but `get_tensorloader` doesn't forward it to `torch.utils.data.DataLoader` or `gluon.data.DataLoader`. This is a book-wide consistency issue that first surfaces here; flagged as P2 since fixing it requires a coordinated change to `oo-design.md` and `synthetic-regression-data.md`. Note in report; let the overview reconcile.

**Batch count discrepancy across frameworks (P0).** PyTorch/TF/MXNet all return `len() = 32`; JAX returns `len() = 31`. This is caused by `drop_remainder=train` in the JAX `get_tensorloader` — a deliberate design choice — but it is never mentioned in prose. A reader switching tabs on the `len()` output will see 31 vs 32 with no explanation.

---

## 4. Implementation spec

### SD-1 — Document JAX `len()` discrepancy  ·  [P0] · [S] · [authored]
- **Type:** teaching / prose
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — after the `len(data.train_dataloader())` cell, before the discussion link block. The surrounding anchor is the text "supports the built-in `__len__` method, so we can query its length, i.e., the number of batches."
- **Change:** Add a framework-specific note immediately after the prose sentence about `__len__`, before the closing Discussion links. Insert the following after line 345 (cell `synthetic-regression-data-concise-implementation-of-the-data-loader-4`):

```
:begin_tab:`jax`
Note that the JAX data loader returns 31 batches rather than 32.
This is because `get_tensorloader` uses `drop_remainder=True` for training:
the final partial batch (8 examples instead of 32) is discarded.
Dropping the partial batch ensures every minibatch has identical shape,
which prevents XLA from recompiling the JIT-compiled training step for the
smaller last batch — a common source of unexpectedly slow training on large
datasets.
:end_tab:
```

- **Touches:** none (prose-only addition to `.md`).
- **Done when:** Rendered HTML shows the callout text inside the JAX tab at the `len()` output; other tabs are unaffected.
- **Depends on:** none.

---

### SD-2 — Sharpen opening motivation for synthetic data  ·  [P1] · [S] · [authored]
- **Type:** prose / coverage
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — lines 9–18, the opening paragraph, anchored by "Machine learning is all about extracting information from data."
- **Change:** Replace the current opening paragraph with the following (tighter, scientifically grounded):

Old text:
```
Machine learning is all about extracting information from data.
So you might wonder, what could we possibly learn from synthetic data?
While we might not care intrinsically about the patterns 
that we ourselves baked into an artificial data generating model,
such datasets are nevertheless useful for didactic purposes,
helping us to evaluate the properties of our learning 
algorithms and to confirm that our implementations work as expected.
For example, if we create data for which the correct parameters are known *a priori*,
then we can check that our model can in fact recover them.
```

New text:
```
Before training a model we need data.
Real datasets are invaluable for applications,
but they conflate three separate sources of failure:
a wrong model, a flawed algorithm, and pathological data.
*Synthetic data* resolves this ambiguity by construction.
When we know the data-generating process exactly
— the true weights $\mathbf{w}^*$, the true bias $b^*$,
and the noise distribution — then any failure to recover
those parameters must be an algorithm or implementation failure,
not a data problem.
This makes synthetic datasets the indispensable debugging tool
for any new learning method:
we first confirm it works on a problem with a known answer,
then we hand it a real one.
```

- **Touches:** none.
- **Done when:** Rendered HTML shows the revised opening; the cross-framework tabs still load; `make html` clean.
- **Depends on:** none.

---

### SD-3 — Cut the "ex post facto" OOP tangent; tighten efficiency paragraph  ·  [P1] · [S] · [authored]
- **Type:** prose / teaching
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — lines 241–259, anchored by "While seemingly innocuous, the invocation of `iter(data.train_dataloader())`".
- **Change:** Replace the entire block from "While seemingly innocuous..." through "...using built-in iterators." with:

Old text:
```
While seemingly innocuous, the invocation 
of `iter(data.train_dataloader())` 
illustrates the power of Python's object-oriented design. 
Note that we added a method to the `SyntheticRegressionData` class
*after* creating the `data` object. 
Nonetheless, the object benefits from 
the *ex post facto* (after-the-fact) addition of functionality to the class.

Throughout the iteration we obtain distinct minibatches
until the entire dataset has been exhausted (try this).
While the iteration implemented above is good for didactic purposes,
it is inefficient in ways that might get us into trouble with real problems.
For example, it requires that we load all the data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in a deep learning framework
are considerably more efficient and they can deal
with sources such as data stored in files, 
data received via a stream, 
and data generated or processed on the fly. 
Next let's try to implement the same method using built-in iterators.
```

New text:
```
Iterating over `data.train_dataloader()` yields distinct minibatches
until the dataset is exhausted.
This hand-rolled implementation is useful for seeing exactly what happens
under the hood, but it has real costs:
all data must fit in memory, iteration is single-threaded Python,
and there is no prefetching to overlap data loading with compute.
Framework data loaders address all of these — they support
parallel workers, prefetch pipelines, memory-mapped files,
and streaming from remote storage.
We now switch to the framework's built-in loader,
which presents an identical interface to the caller.
```

- **Touches:** none.
- **Done when:** The revised prose appears in rendered HTML; the "ex post facto" phrase is absent; the efficiency paragraph is ≤5 sentences.
- **Depends on:** none.

---

### SD-4 — Rewrite the Summary  ·  [P1] · [S] · [authored]
- **Type:** prose / teaching
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — the `## Summary` block, lines 348–369, anchored by "Data loaders are a convenient way of abstracting out".
- **Change:** Replace the current Summary with:

Old text:
```
## Summary

Data loaders are a convenient way of abstracting out 
the process of loading and manipulating data. 
This way the same machine learning *algorithm* 
is capable of processing many different types and sources of data 
without the need for modification. 
One of the nice things about data loaders 
is that they can be composed. 
For instance, we might be loading images 
and then have a postprocessing filter 
that crops them or modifies them in other ways. 
As such, data loaders can be used 
to describe an entire data processing pipeline. 

As for the model itself, the two-dimensional linear model 
is about the simplest we might encounter. 
It lets us test out the accuracy of regression models 
without worrying about having insufficient amounts of data 
or an underdetermined system of equations. 
We will put this to good use in the next section.  
```

New text:
```
## Summary

Synthetic data closes the loop: we know $\mathbf{w}^*$ and $b^*$,
so after training we can check whether the learned parameters agree.
This makes synthetic datasets the first stop for validating any new algorithm.
The `SyntheticRegressionData` class introduced here
encapsulates the data-generating process in a `DataModule` subclass,
separating *where batches come from* from *how the model uses them*.
We also saw two implementations of the same `get_dataloader` protocol:
a transparent hand-rolled iterator that is easy to read but inefficient,
and a framework-native loader that handles shuffling, prefetching,
and parallel workers automatically.
The hand-rolled version teaches; the framework version is what we use going forward.
```

- **Touches:** none.
- **Done when:** New Summary appears in rendered HTML; original Summary text absent; `make html` clean.
- **Depends on:** none.

---

### SD-5 — Add exercise: vary noise σ and observe parameter recovery  ·  [P1] · [S] · [authored]
- **Type:** exercise
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — `## Exercises` block, after exercise 4 (line 378), before the `:begin_tab:`mxnet`` discussion links.
- **Change:** Append a new exercise 5:

```
1. In `SyntheticRegressionData`, vary the noise standard deviation
   `noise` over the range $\{0.001, 0.01, 0.1, 0.5, 1.0\}$.
   After training a linear model on each dataset (using code from
   :numref:`sec_linear-regression-scratch` or :numref:`sec_linear-regression-concise`),
   how closely does the recovered $\hat{\mathbf{w}}$ match the true
   $\mathbf{w}^* = [2, -3.4]^\top$?
   Plot the estimation error $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$
   as a function of $\sigma$.
   What asymptotic behavior do you expect from statistical theory?
```

- **Touches:** none (prose-only).
- **Done when:** A fifth exercise appears in the rendered Exercises section; references to `sec_linear-regression-scratch` and `sec_linear-regression-concise` resolve correctly.
- **Depends on:** none (the forward references to §3.4 and §3.5 are fine since exercises assume reading order).

---

### SD-6 — MXNet archival callout  ·  [P2] · [S] · [mechanical]
- **Type:** currency
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — the `:begin_tab:`mxnet`` block inside the `## Concise Implementation of the Data Loader` section, immediately before the `gluon.data` cell (line ~315).
- **Change:** Add a short note inside the first MXNet tab in this section. After the line `%%tab mxnet` and before the `@d2l.add_to_class` line in the `get_tensorloader` cell, insert a `:begin_tab:`mxnet`` note paragraph above the cell:

Insert before the cell `synthetic-regression-data-concise-implementation-of-the-data-loader-1` (the `%%tab mxnet` block):

```
:begin_tab:`mxnet`
**Note:** Apache MXNet was archived by the Apache Software Foundation in September 2023
and is no longer actively maintained.
The MXNet tabs in this book are preserved for reference and continuity
with earlier editions; for new projects, prefer PyTorch, JAX, or TensorFlow.
:end_tab:
```

- **Touches:** none beyond the single `.md` file.
- **Done when:** The archival note renders in the MXNet tab in the Concise section; `make html` clean.
- **Depends on:** Book-wide decision on MXNet messaging (the overview should coordinate a consistent note text across all chapter files). This entry adopts that note text; if the overview changes the wording, update accordingly.

---

### SD-7 — Rename `noise` local variable to `eps` to avoid hyperparameter shadowing  ·  [P2] · [S] · [mechanical]
- **Type:** code / clarity
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — cell `synthetic-regression-data-generating-the-dataset-1`, all four framework tabs (PyTorch, TF, JAX, MXNet).
- **Change:** In each of the four `%%tab` blocks in that cell, rename the local tensor variable `noise` (the result of the randn call) to `eps`. There are exactly two occurrences per tab (one assignment, one use):

PyTorch tab — old:
```python
        noise = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```
New:
```python
        eps = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

TensorFlow tab — old:
```python
        noise = tf.random.normal((n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```
New:
```python
        eps = tf.random.normal((n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

JAX tab — old:
```python
        noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```
New:
```python
        eps = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

MXNet tab — old:
```python
        noise = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```
New:
```python
        eps = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

Also update the prose at lines 68–75 to use $\boldsymbol{\epsilon}$ (already does, no change) and confirm the equation still reads $\mathbf{y} = \mathbf{X}\mathbf{w} + b + \boldsymbol{\epsilon}$ (it does, no change needed there).

- **Touches:** The `#@save` on `SyntheticRegressionData` means the d2l library will be rebuilt (`make lib`). The change is semantically neutral — outputs will be bit-identical. Run `make lib` after editing.
- **Done when:** No framework tab uses `noise` as both a parameter name and a local variable in the same scope; `make lib` succeeds; a quick execution of the cell produces the same output as before.
- **Depends on:** none.

---

### SD-8 — Elevate exercise 4 to a cross-framework PRNG comparison  ·  [P2] · [S] · [authored]
- **Type:** exercise
- **Where:** `chapter_linear-regression/synthetic-regression-data.md` — exercise 4, line 378, anchored by "How would you design a random data generator that generates *the same* data each time it is called?"
- **Change:** Replace exercise 4 with:

Old:
```
1. How would you design a random data generator that generates *the same* data each time it is called?
```

New:
```
1. **(Reproducibility across frameworks.)** How would you design a random data generator
   that produces the *same* dataset every time it is called?
   In PyTorch and TensorFlow, a global `torch.manual_seed` / `tf.random.set_seed`
   suffices. In JAX, the PRNG is *functional* — there is no global state.
   Explain why passing `key=jax.random.PRNGKey(42)` to `SyntheticRegressionData`
   already gives reproducibility in JAX, and why re-using the same key for both
   $\mathbf{X}$ and $\boldsymbol{\epsilon}$ would be incorrect.
```

- **Touches:** none.
- **Done when:** Exercise 4 in the rendered HTML reads the new text; the question about JAX key reuse is present.
- **Depends on:** SD-7 (renames `noise` → `eps`; the exercise should refer to $\boldsymbol{\epsilon}$ not `noise`, which SD-7 aligns). Can be applied independently.

---

## 5. Keep — what is already excellent (do not lose this)

- **JAX PRNGKey handling.** The `key=None` default resolved to `PRNGKey(0)` inside `__init__`, with `jax.random.split` for independent `X` and `eps` keys, is correct functional-PRNG practice. The inline comment explaining it is good. Do not simplify away.

- **JAX `drop_remainder=train` in `get_tensorloader`.** The three-line comment explaining JIT recompilation avoidance is the only place in the chapter this important practical detail is explained. It is correct and well-motivated. Keep verbatim.

- **Hand-rolled → concise structure.** The pedagogical sequence (show the mechanism first, then show the production version) is exactly right. The hand-rolled iterator is compact enough that it teaches without boring.

- **`@d2l.add_to_class` pattern.** Using it twice (for the hand-rolled `get_dataloader` then overriding with the concise version) naturally demonstrates the pattern and makes the transition concrete.

- **`save_hyperparameters()` with all constructor args.** Storing `w`, `b`, `noise`, `num_train`, `num_val`, `batch_size` as hyperparameters makes the class introspectable and consistent with the `d2l.HyperParameters` convention throughout the book. Keep.

- **Cross-framework consistency in `get_dataloader`.** All four hand-rolled tabs are structurally identical, differing only where the framework forces it. This is disciplined and lowers the cognitive load of tab-switching.

- **The TF `shuffle_buffer = tensors[0].shape[0]` (full-buffer shuffle).** This is correct for small in-memory datasets and produces truly uniform random batches. It is the right choice at this scale and does not need qualification.
