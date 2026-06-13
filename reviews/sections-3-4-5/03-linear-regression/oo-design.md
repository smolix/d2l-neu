# Review — chapter_linear-regression/oo-design.md  (§3.2 "Object-Oriented Design for Implementation")

**Role in the chapter:** This is the infrastructure chapter for the entire book — it introduces `Module`, `DataModule`, `Trainer`, `HyperParameters`, `ProgressBoard`, and the `add_to_class` decorator that all subsequent chapters build upon. It must both explain these abstractions clearly and justify why they exist.

**Verdict:** The skeleton is solid and the ProgressBoard async-decoupling explanation is genuinely excellent prose. However, the file currently lacks the one thing it most needs: a *motivating figure* showing how the three classes relate (a conceptual UML-lite diagram). The intro is choppy and passive where it should be punchy. The JAX `Trainer.fit` carries ~20 lines of batch-norm/TrainState machinery that a reader of §3.2 has never heard of, which is the most disorienting moment in the page. Two minor issues: the PyTorch Lightning URL is stale (the domain moved to `lightning.ai`), and the TF `Module` exposes two dead helper methods (`_report_train`, `_report_val`) that are not part of the abstract interface but sit unexplained in the scaffold shown here. With a class-relationship figure, a tighter intro, and the JAX Trainer trimmed to its minimal form, this section would be best-in-class.

**Grade:** B+. Assignable but the "wall of class code with no picture" is a real teaching deficit for the intended audience; a strong introductory figure would push it to A.

**Top priorities (ranked):**
1. [P0] OO-1 — Add a class-relationship diagram (Module/DataModule/Trainer with their key methods; pre-generated SVG, no notebook drawing code)
2. [P1] OO-2 — Rewrite the choppy two-paragraph intro as a single tight motivating paragraph
3. [P1] OO-3 — Trim the JAX Trainer.fit `batch_stats`/`TrainState` block to its minimal linear-regression form; move the full machinery to a forward comment
4. [P1] OO-4 — Fix the stale PyTorch Lightning URL (`pytorchlightning.ai` → `lightning.ai`)
5. [P1] OO-5 — Remove / comment the TF `_report_train` / `_report_val` dead methods from this scaffold
6. [P2] OO-6 — Add three substantive exercises (current exercises are trivially weak)
7. [P2] OO-7 — Add a note on Flax `linen` (legacy) vs `nnx` (current) to the JAX tab-note
8. [P2] OO-8 — Tighten the Summary section (currently passive and repetitive)

---

## 1. Coverage

### Add

**A motivating figure is the single largest gap.** The page presents four long code blocks for four frameworks before a reader can visualise how these three classes connect. Every strong OO-design presentation — PyTorch Lightning's own docs, fast.ai's deep learning from scratch, CS231n's software engineering segment — opens with a box-and-arrow diagram. Here the prose on lines 35–42 lists `(i)`, `(ii)`, `(iii)` but has no visual correlate. A pre-generated SVG showing three boxes (`Module`, `DataModule`, `Trainer`) with arrows labelled `fit(model, data)`, `training_step(batch)`, `train_dataloader()`, etc., would let a reader orient before reading any code. This is in-scope infrastructure content that every reader needs; it is the canonical example where a figure unlocks the idea.

**The motivation for OO style is understated.** Lines 9–32 answer "what" (we are designing APIs) but barely answer "why" (what does OO buy us that a bag of functions would not?). A single-sentence concrete payoff belongs here: "Because each chapter's model only subclasses `Module` and overrides `training_step`, we can swap the dataset, the optimiser, or the training loop independently — none of the other components change." This belongs in a tightened intro paragraph (see OO-2).

**A forward pointer to Flax NNX is warranted.** Flax's `linen` API (used throughout the JAX tabs) was superseded by `flax.nnx` in 2024 (stable in Flax 0.9+). The tab-note at lines 438–442 describes `linen.Module` without acknowledging that a newer, imperative-style NNX API now exists. The scope-correct action here is not to rewrite the book in NNX (too large) but to add one sentence: "As of 2024, Flax offers a newer `flax.nnx` API with an imperative style closer to PyTorch; `linen` remains fully supported and is what this book uses throughout." That keeps the text honest without requiring a rewrite.

### Remove / trim

**TF `_report_train` / `_report_val` in `Module` (lines 344–352) are dead interface surface.** These two methods do not exist in the PyTorch, JAX, or MXNet variants. They exist because `linear-regression-scratch.md`'s TF `Trainer.fit_epoch` calls `self.model._report_train(loss)` instead of the generic path, a TF-specific workaround for `tf.function` graph-mode tracing. Exposing them in the abstract scaffold shown here — with no explanation — makes the TF `Module` look different from the other three in a way that teaches nothing. Either: (a) remove them from the scaffold here and only introduce them in `linear-regression-scratch.md` where they are first needed (preferred), or (b) add a one-sentence tab-note explaining why TF needs this hook. Option (a) is cleaner — these are implementation details of the concrete subclass, not the base class contract.

**JAX `Trainer.fit` `batch_stats`/`TrainState` block (lines 647–664) is premature.** The 20 lines handling `batch_stats` and the subclassed `TrainState` are required for batch normalisation (chapters 8+), not for the linear regression models this chapter builds. A first-time reader of §3.2 has never heard of batch statistics. The minimal form needed here is just parameter initialisation — the `batch_stats` branch is dead in all §3 examples (always `{}`). The right fix is to trim `fit` to the linear-regression-only form and introduce the `batch_stats` extension point in the relevant later chapter via `@add_to_class`. The current JAX Trainer is more complex than the PyTorch/TF/MXNet Trainers for no reason a §3 reader can follow.

**Double blank line between intro paragraphs (lines 32–35)** — minor formatting noise; one blank line is correct.

### Reorder / restructure

No fundamental reordering needed. The existing 3–5 `##` structure (`Utilities`, `Models`, `Data`, `Training`, `Summary`) is logical and maps directly onto the three classes plus their shared utilities. This is correct.

The one structural concern: the section label on `## Models` is `:label:`subsec_oo-design-models`` (with prefix `subsec_`), while `## Utilities` uses `:label:`oo-design-utilities`` and `## Data` uses `:label:`oo-design-data``. The `subsec_` prefix on the Models label is inconsistent and will produce an inconsistent cross-reference prefix in the rendered numbering. It should be `oo-design-models` to match the pattern.

---

## 2. Teaching quality

### Structure & flow

The `##` spine is sound: Utilities → Models → Data → Training → Summary. Three concerns:

1. **The intro is written as two disconnected paragraphs.** Paragraph 1 (lines 9–32) motivates OO in terms of notebook readability. Paragraph 2 (lines 35–42) pivots to listing the three classes, with a PyTorch Lightning reference dropped in. The double-blank-line gap between them reinforces the disconnect. A single crisp paragraph that (a) states the payoff, (b) names the three classes, and (c) gives the key analogy would be stronger.

2. **`ProgressBoard.draw`'s async design** is explained very well in the current text (lines 156–176) — this is the best writing in the file. The analogy to "keep the hot path pure and compiled" is an idea worth keeping exactly.

3. **JAX dataclass note (lines 197–202)** is brief but adequate. However, it is a `:begin_tab:` that has no counterpart tab for the other frameworks; readers switching frameworks see nothing where the JAX tab shows a substantive note. This asymmetry is fine — just worth noting that the other frameworks could benefit from a one-line tab-note identifying the base class they extend (`nn.Module`, `tf.keras.Model`, `nn.Block`). Those notes do exist later (lines 424–442), but they are *after* the code cell, not before it (where a reader wants orientation). Consider moving those tab-notes to immediately before their code cells.

### Figures

**There are zero conceptual figures in this section.** The only rendered output is the ProgressBoard `sin`/`cos` demo, which is a *data plot* (not a conceptual diagram). This is the most significant teaching gap.

Specifically needed: a **class-relationship diagram** (see OO-1 below). It should show:
- Three boxes: `Module`, `DataModule`, `Trainer`
- Key selected methods in each box (not exhaustive — just the contract: `training_step`, `configure_optimizers` for Module; `train_dataloader` for DataModule; `fit(model, data)` for Trainer)
- Arrows: `Trainer.fit` → `Module` (calls `training_step`), `Trainer.fit` → `DataModule` (calls `train_dataloader`)
- A note showing `@add_to_class` as the extension mechanism

This should be a pre-generated SVG (`img/oo-design-class-diagram.svg`) produced by a committed generator, using the book's standard figure style. No drawing code in the notebook.

### Prose & clarity

**Lines 9–32 (intro):** The current text is fragmented into short grammatical units across many lines, producing a halting rhythm. Example of the problem:

> "In our introduction to linear regression, / we walked through various components / including / the data, the model, the loss function, / and the optimization algorithm. / Indeed, / linear regression is / one of the simplest machine learning models."

This telegraphs rather than argues. Compare to the calculus chapter's opening — it makes a *claim* and immediately grounds it in a picture. The oo-design intro should open with the payoff: "We are going to reuse this training loop — with only the model and the dataset swapped — for almost every chapter in this book. That reuse is only possible if we design the right abstractions upfront." Then name the three classes. See OO-2 for a drafted replacement.

**Line 33 (double blank line):** formatting error — should be one blank line.

**Line 421 (`nn.Block`) / line 439 (`linen.Module`):** the MXNet tab-note says "`Module` is a subclass of `nn.Block`, the base class of neural networks in Gluon." `nn.Block` was the Gluon 1.x API; Gluon's successor (MXNet 2) uses `nn.Block` still, so this is not technically wrong, but the phrase "in Gluon" reads as dated since Gluon was absorbed into MXNet 2. Minor.

**Line 35:** `https://www.pytorchlightning.ai/` is a stale URL — the project rebranded to `lightning.ai`. The domain may redirect but will not do so forever. Fix: update to `https://lightning.ai/` with display text "Lightning" (the framework is now called "Lightning", not "PyTorch Lightning"). See OO-4.

**Lines 712–726 (Summary):** largely restates what was said in the intro, without connecting forward to what follows. A one-sentence payoff per class, plus a forward pointer to `linear-regression-scratch.md` where they are first used concretely, would strengthen this. See OO-8.

### Exercises

**Currently two exercises, both trivially weak:**
1. "Locate the full implementations in the D2L library." This is a search exercise with no intellectual content.
2. "Remove `save_hyperparameters` and see what breaks." This is a one-liner to run.

Neither is worth the problem-set slot at a top program. Three stronger exercises are proposed in OO-6:
- One that asks the reader to extend `Module` with a new method via `@add_to_class` (tests understanding of the decorator pattern)
- One that asks why `configure_optimizers` is a method of `Module` rather than an argument to `Trainer` (tests understanding of the design choice)
- One that asks the reader to adapt `DataModule` to return a third split (test set) and update `Trainer.fit` accordingly (tests understanding of the abstraction's extensibility)

---

## 3. Code & examples

### Does the code teach?

Most cells teach clearly — the `add_to_class` demo with class `A` and instance `a.do()` is a concise, well-chosen example. The `HyperParameters` demo with class `B` and `ignore=['c']` shows exactly what the mixin does. The ProgressBoard `sin`/`cos` demo motivates `every_n` nicely.

The walls-of-plumbing concern is real for the `Module` and `Trainer` cells, which are long (40–70 lines each) and necessarily contain boilerplate. However, the boilerplate is *unavoidable* for scaffold code — the question is whether it is *explained*. The PyTorch `Module.plot` method (lines 220–237) has a good inline comment about the async thunk, but no surrounding prose that explains why `plot` belongs in `Module` at all, rather than in `Trainer`. A one-sentence prose note before or after the cell would help.

The JAX `Trainer.fit` is the only cell where the code actively confuses rather than teaches — the `batch_stats` branch and nested `TrainState` subclass are infrastructure for a feature (batch norm) that doesn't appear until part 4 of the book. This creates the experience of reading boilerplate for an unknown future purpose. See OO-3.

### PyTorch

- **Modern and idiomatic.** The `Module` inherits `nn.Module` correctly via `d2l.nn_Module`. `configure_optimizers` returning an optim is the standard pattern.
- **`torch.compile` mention (line 159)** — correct and current (PyTorch 2.0+). Good.
- The `lambda v=value: d2l.numpy(d2l.to(v, d2l.cpu()))` thunk in `plot` (line 235) is the right pattern for deferring device-to-host transfers; the inline comment explains it well.
- No issues. PyTorch tab is clean.

### JAX

- **`from flax import linen as nn` (line 73)** — `linen` is the legacy Flax API, superseded by `flax.nnx` (stable since Flax 0.9 / 2024). The book's wholesale commitment to `linen` is a reasonable scoping decision (too large to change here), but the tab-note at line 438 should acknowledge this. See OO-7.
- **JAX `Module` (lines 359–416):** Using `@dataclass` semantics (field defaults, `init=False` fields) is correct for Flax `linen`. The `forward`/`__call__` redirect (lines 375–381) to maintain API parity across frameworks is well-motivated in the tab-note.
- **`training_step` signature (line 401):** `(self, params, batch, state)` diverges substantially from PyTorch/TF/MXNet `(self, batch)`. This divergence is *required* by JAX's functional style (params are explicit), but it means JAX readers see a different contract. The prose should explicitly call this out in the tab-note as a fundamental JAX difference, not just let it pass silently.
- **JAX `Trainer.fit` batch_stats block (lines 647–664):** The `batch_stats` check and the nested `TrainState` subclass are premature. See OO-3.
- **`model.configure_optimizers()` called twice** in JAX `fit` (lines 634 and 664): `self.optim = model.configure_optimizers()` at line 634, then `tx=model.configure_optimizers()` at line 664. This creates two separate optimizer instances, discarding the first. This is a latent bug — if `configure_optimizers` had side effects (e.g., parameter-group initialization), calling it twice would break. Should be `tx=self.optim` at line 664.

### TensorFlow

- `Module` inherits `tf.keras.Model` via `d2l.nn_Module`. The `call`→`forward` redirect (lines 318–321) is the correct Keras 3 pattern. The `self.__dict__.pop('loss', None)` on line 309 is needed to prevent Keras from shadowing the `loss` method; the tab-note at line 433 explains this.
- **`_report_train` / `_report_val` (lines 344–352)** are dead-letter interface surface in this abstract scaffold. They have no counterpart in the other three frameworks, are not part of the documented contract (the prose at line 194 doesn't mention them), and exist only because `linear-regression-scratch.md`'s TF fit_epoch uses a different code path. They should either be removed from this base class (introduced where first needed) or documented in a tab-note. See OO-5.
- The `_compile_steps` method mentioned in the Trainer tab-note (line 525) is a good forward reference; the cross-reference to `sec_linear_scratch` (line 532) is correctly placed.

### MXNet

- **Apache MXNet was archived by the ASF in 2023.** This is a live framework tab presented as co-equal with PyTorch, JAX, and TensorFlow. The prose at line 421 refers to `nn.Block` / "Gluon" without any disclaimer. The CLAUDE.md confirms MXNet notebooks still execute (the custom wheel), so the technical content is not broken — but presenting MXNet to students as a primary framework choice in 2026 is misleading. A one-sentence note — "Note: Apache MXNet was archived in 2023 and is no longer under active development; this tab is retained for readers who need to maintain existing MXNet code." — is the minimum needed. This is a book-wide decision (applies to all files with MXNet tabs), flagged here for the overview.
- **`Module.plot` MXNet variant (lines 280–286):** The comment explaining the synchronous device-to-host transfer (MXNet CUDA context bug) is accurate and necessary. Good.

### Cross-framework consistency & d2l conventions

- **DataModule is identical across all four frameworks** (the four `%%tab` variants of `#oo-design-data` are byte-for-byte the same code). This is fine — it demonstrates portability — but it also means there's no reason to show all four tabs. A note "The `DataModule` definition is identical across all frameworks; select any tab." and a single tab would reduce code surface without losing any content. This is a P2 cosmetic improvement.
- **`#oo-design-object-oriented-design-for-implementation` cell ID** (line 44/52/61/69): this is the imports cell. The ID is extremely long; per d2l conventions, stable IDs should be short. Consider `#oo-design-imports`. (The ID is referenced nowhere externally in this file, so renaming is safe.) P2.
- **`%%tab` ordering:** tabs are ordered `mxnet, pytorch, tensorflow, jax` in `tab.interact_select` (line 3) but in the Models cell the `:begin_tab:` notes appear as `mxnet`, `pytorch`, `tensorflow`, `jax` in that order. This is consistent. Fine.
- **`d2l` helpers used appropriately:** `d2l.numpy`, `d2l.to`, `d2l.cpu` used in `plot`. `d2l.plt` and `d2l.set_figsize` not needed here (no plot cells). Conventions are respected.

---

## 4. Implementation spec

### OO-1 — Add class-relationship diagram  ·  P0 · L · authored

- **Type:** figure
- **Where:** `chapter_linear-regression/oo-design.md` — after the intro paragraph block (after line 42, before the imports code cell on line 44); specifically after the sentence "Most code in this book adapts `Module` and `DataModule`..."
- **Change:** Create a pre-generated SVG figure `img/oo-design-class-diagram.svg` using a committed generator `tools/gen_oo_design_figure.py`, then include it in the file. The figure should depict:
  - Three labelled boxes in a horizontal layout: `Module` (left), `Trainer` (center), `DataModule` (right)
  - Inside `Module`: methods `forward(X)`, `loss(y_hat, y)`, `configure_optimizers()`, `training_step(batch)`
  - Inside `DataModule`: methods `train_dataloader()`, `val_dataloader()`
  - Inside `Trainer`: method `fit(model, data)` + `fit_epoch()` (abstract)
  - Arrows: from `Trainer` to `Module` labelled "calls `training_step`"; from `Trainer` to `DataModule` labelled "calls `train_dataloader`"
  - A small note box attached to `Module`/`DataModule`: "@add_to_class — extend in later chapters"
  - Use the book's standard house style (from `tools/gen_mdl_figures.py`: `BLUE="#1f77b4"`, `ORANGE="#ff7f0e"`, font size 11, `figure.dpi=100`)
  
  Insert after line 42 in `oo-design.md`:
  ```
  ![The three base classes and their relationships. Subclasses of `Module` and `DataModule` provide the model and the data; `Trainer.fit` drives the loop.](../img/oo-design-class-diagram.svg)
  :label:`fig_oo_design`
  ```
- **Touches:** new file `tools/gen_oo_design_figure.py`; `make figures` (or run the generator directly); new file `img/oo-design-class-diagram.svg` committed to the repo.
- **Done when:** `img/oo-design-class-diagram.svg` exists and renders cleanly in the HTML page at §3.2 with no broken image; `make html` passes; the SVG is byte-idempotent on re-run.
- **Depends on:** none.

---

### OO-2 — Rewrite intro to be tight and motivating  ·  P1 · S · authored

- **Type:** prose
- **Where:** `chapter_linear-regression/oo-design.md` — lines 9–42 (the two intro paragraphs from "In our introduction to linear regression" through "...optimization algorithms.")
- **Change:** Replace lines 9–42 (the two intro paragraphs) with:

  ```
  Almost every model we build in this book follows the same loop: load data,
  run a forward pass, compute the loss, update the parameters, repeat.  If we
  wrote that loop from scratch for each new model, a small change to the
  training procedure — say, adding gradient clipping or a learning-rate
  schedule — would require touching every chapter.  The solution, borrowed
  from libraries such as [Lightning](https://lightning.ai/), is to fix the
  loop inside a reusable `Trainer` class and let models and datasets vary as
  subclasses of `Module` and `DataModule`:

  - **`Module`** — encapsulates model parameters, the `forward` pass, the
    loss, and the optimizer.  Every model in the book is a subclass.
  - **`DataModule`** — encapsulates a dataset and provides train and
    validation data loaders.  Every dataset is a subclass.
  - **`Trainer`** — owns the epoch loop, feeds batches from `DataModule`
    into `Module.training_step`, and calls the optimizer.

  Before building any of these concretely, we first introduce three small
  utilities that make the OO style work smoothly in notebooks.
  ```

- **Touches:** none.
- **Done when:** the rendered page opens with this paragraph; `make html` is clean; no broken `:numref:` or `:citet:` references.
- **Depends on:** none (but OO-1's figure reference should appear after this paragraph).

---

### OO-3 — Trim JAX Trainer.fit to minimal linear-regression form  ·  P1 · M · authored

- **Type:** code
- **Where:** `chapter_linear-regression/oo-design.md` — cell `#oo-design-training` `%%tab jax`, specifically lines 647–664 (the `batch_stats` / `TrainState` subclass block)
- **Change:** Replace the full `fit` method in the JAX Trainer with the stripped-down version that only does what §3 needs (linear regression: no batch norm, no dropout). Move the `batch_stats` / `TrainState` extension to a comment or to the first chapter that uses it (batch norm in the convolutional part). 

  Replace the `fit` method body (from `dummy_input = ...` through `self.val_batch_idx = 0`) with:
  ```python
      def fit(self, model, data, key=None):
          self.prepare_data(data)
          self.prepare_model(model)
          self.optim = model.configure_optimizers()

          if key is None:
              root_key = d2l.get_key()
          else:
              root_key = key
          params_key, dropout_key = jax.random.split(root_key)
          key = {'params': params_key, 'dropout': dropout_key}

          dummy_input = next(iter(self.train_dataloader))[:-1]
          variables = model.apply_init(dummy_input, key=key)
          params = variables['params']

          self.state = train_state.TrainState.create(
              apply_fn=model.apply,
              params=params,
              tx=self.optim)
          self.epoch = 0
          self.train_batch_idx = 0
          self.val_batch_idx = 0
          for self.epoch in range(self.max_epochs):
              self.fit_epoch()
          self.model.board.flush()
  ```
  
  Also fix the **double `configure_optimizers` call** bug: `self.optim = model.configure_optimizers()` is set at line 634 then called again as `tx=model.configure_optimizers()` at line 664. The fix uses `tx=self.optim` (as shown above).

  Add a prose paragraph before the JAX Trainer code cell:
  > `:begin_tab:\`jax\`` ... "The JAX `Trainer` additionally initialises the model parameters via `apply_init` and wraps them in an Optax `TrainState`. Later chapters extend `TrainState` to hold batch-normalisation statistics and PRNG keys; for now, only the learnable parameters are tracked." `:end_tab:`

- **Touches:** downstream JAX notebooks that depend on `TrainState` having `batch_stats` will need `@add_to_class(d2l.Trainer)` patches in the relevant chapters (batch-norm section). This is the correct pattern for the book's incremental-reveal design.
- **Done when:** JAX `Trainer.fit` in §3.2 has no `batch_stats` / nested `TrainState` subclass; all JAX notebooks in chapter_linear-regression still execute green (`make -B _notebooks/jax/chapter_linear-regression/*.executed`); the `batch_stats` variant is introduced in the first chapter that uses it via `@add_to_class`.
- **Depends on:** must verify that `linear-regression-scratch.md` JAX fit_epoch does not depend on `state.batch_stats`.

---

### OO-4 — Fix stale PyTorch Lightning URL  ·  P1 · S · mechanical

- **Type:** currency
- **Where:** `chapter_linear-regression/oo-design.md` — line 35: `[PyTorch Lightning](https://www.pytorchlightning.ai/)`
- **Change:**
  - old: `[PyTorch Lightning](https://www.pytorchlightning.ai/)`
  - new: `[Lightning](https://lightning.ai/)`
- **Touches:** none.
- **Done when:** the link in the rendered HTML points to `https://lightning.ai/`; `make html` clean.
- **Depends on:** none.

---

### OO-5 — Remove TF dead-letter methods `_report_train` / `_report_val` from base scaffold  ·  P1 · S · judgment

- **Type:** code
- **Where:** `chapter_linear-regression/oo-design.md` — cell `#oo-design-models` `%%tab tensorflow`, lines 344–352:
  ```python
      def _report_train(self, loss):
          self.plot('loss', loss, train=True)

      def _report_val(self, y_hat, batch):
          self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
  ```
- **Change:** Remove these two methods from the `Module` definition shown in §3.2. They are TF-implementation details of `linear-regression-scratch.md`'s `fit_epoch` and do not belong in the abstract base-class scaffold. They should first appear (introduced via `@add_to_class(d2l.Module)`) in `linear-regression-scratch.md` directly before the TF `fit_epoch` that calls them. Add a one-line tab-note in `linear-regression-scratch.md` explaining why the TF path needs these hooks (graph-mode tracing constraint).
- **Touches:** `chapter_linear-regression/linear-regression-scratch.md` — add an `@add_to_class(d2l.Module)` cell for `%%tab tensorflow` defining `_report_train` and `_report_val` before the TF `fit_epoch`; also update `d2l/tensorflow.py` if `_report_train`/`_report_val` are #@saved from the scratch file rather than oo-design.
- **Done when:** TF `Module` in §3.2 has the same four methods as PyTorch/MXNet variants (`loss`, `forward`, `plot`, `training_step`, `validation_step`, `configure_optimizers` — and no extra methods); TF notebooks in chapter_linear-regression still execute green.
- **Depends on:** none, but coordinate with linear-regression-scratch.md review.

---

### OO-6 — Add three substantive exercises  ·  P2 · M · authored

- **Type:** teaching
- **Where:** `chapter_linear-regression/oo-design.md` — `## Exercises` section (lines 729–731); replace the existing two exercises with the following five:
- **Change:** Replace the existing exercise list with:

  ```markdown
  1. The `add_to_class` decorator works by calling `setattr(Class, obj.__name__, obj)`.
     (a) Add a method `greet(self)` to the existing class `A` *after* the instance `a`
     has been created, using `@add_to_class(A)`. Verify that `a.greet()` works.
     (b) What happens if you define `greet` *without* the decorator and then call
     `a.greet()`? Why?

  1. The `Module` class keeps the optimizer in `configure_optimizers`, which is a
     *method of the model*, rather than passing the optimizer as an argument to
     `Trainer`. What are the advantages of this design choice?  Can you think of a
     case where putting the optimizer on the model is problematic?

  1. Extend `DataModule` with a `test_dataloader` method and extend `Trainer.fit` to
     run a final evaluation pass on the test set after training. What invariant must
     the test loader satisfy that the validation loader does not?

  1. The current `save_hyperparameters` implementation uses Python's `inspect` module
     to capture the caller's local variables. Can you implement a version that does
     *not* use `inspect` — for example, by requiring the caller to pass the local
     namespace explicitly? What are the trade-offs?

  1. (Advanced) The `ProgressBoard.draw` method is *asynchronous*: it hands values to
     a background thread rather than plotting immediately. Sketch an alternative
     synchronous implementation. Under what conditions would the synchronous version
     be slower? Under what conditions would they perform identically?
  ```

- **Touches:** none.
- **Done when:** five exercises appear in the rendered page; the first exercise can be run interactively in the notebook.
- **Depends on:** none.

---

### OO-7 — Add Flax NNX currency note to JAX tab-note  ·  P2 · S · authored

- **Type:** currency
- **Where:** `chapter_linear-regression/oo-design.md` — `:begin_tab:\`jax\`` note at lines 438–442 (the tab-note after the `#oo-design-models` JAX code cell)
- **Change:** Append one sentence to the existing JAX tab-note:

  Current last sentence: "Here we also redirect `__call__` to the `forward` method. We do this to make our code more similar to other framework implementations."
  
  Add after: "As of 2024, Flax offers a newer, imperative `flax.nnx` API that is closer in style to PyTorch; this book uses the `linen` API throughout, which remains fully supported."

- **Touches:** none.
- **Done when:** the rendered JAX tab note contains this sentence; `make html` clean.
- **Depends on:** none.

---

### OO-8 — Tighten the Summary section  ·  P2 · S · authored

- **Type:** prose
- **Where:** `chapter_linear-regression/oo-design.md` — lines 712–726 (`## Summary` through the blank line before `## Exercises`)
- **Change:** Replace the current summary with:

  ```markdown
  ## Summary

  Three base classes form the scaffold for every model in this book:
  `Module` (model + loss + optimizer), `DataModule` (data loaders),
  and `Trainer` (the epoch loop).  The `add_to_class` decorator lets us
  spread a class definition across multiple cells — essential for
  notebook-sized chunks of pedagogy.  `HyperParameters.save_hyperparameters`
  eliminates constructor boilerplate.  `ProgressBoard` provides live
  loss-curve monitoring without slowing the training loop, by decoupling
  the hot training path from the (slow) plotting thread.

  In the next section we build a concrete dataset — synthetic linear
  regression data — then use these classes directly in
  :numref:`sec_linear_scratch` and :numref:`sec_linear_concise`.
  ```

- **Touches:** none.
- **Done when:** the summary is present, forward-pointing `:numref:` references resolve, and `make html` is clean.
- **Depends on:** none.

---

### OO-9 — Fix inconsistent `subsec_` label prefix on `## Models`  ·  P2 · S · mechanical

- **Type:** prose
- **Where:** `chapter_linear-regression/oo-design.md` — line 196: `:label:\`subsec_oo-design-models\``
- **Change:**
  - old: `:label:`subsec_oo-design-models``
  - new: `:label:`oo-design-models``
- **Touches:** any file that references `:numref:`subsec_oo-design-models`` must be updated. Run: `grep -rn "subsec_oo-design-models" /path/to/repo/` to find all callsites.
- **Done when:** `make html` produces no undefined reference warning for `subsec_oo-design-models`; the section is cross-referenceable as `oo-design-models`.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The `ProgressBoard` async design explanation (lines 156–176):** The prose explaining *why* `draw` is asynchronous — "keep the hot path pure and compiled, push logging off to the side" — is some of the best writing in the chapter. It names the two rules of compiled training (purity, async device execution) and explains how the queue-based design respects both. This should not be changed.

- **The `add_to_class` demo with class `A`:** Three cells, each small: define the class and instance, define the method with the decorator, call it. The output `"Class attribute 'b' is 1"` is a perfect smoke-test. Clear, minimal, reusable as a mental model.

- **The `HyperParameters.save_hyperparameters` demo with `ignore`:** The `B(a=1, b=2, c=3)` example showing that `c` is not saved is the right level of concreteness. The actual `inspect`-based implementation is deliberately deferred to `sec_utils` — this is the right choice; showing it here would be a wall of introspection code that teaches nothing about deep learning.

- **MXNet `Module.plot` comment (lines 280–286):** The explanation of why MXNet needs a synchronous device-to-host transfer (CUDA context corruption risk from a foreign thread) is accurate, technically precise, and useful for anyone maintaining MXNet code. Worth keeping even if the MXNet tab is eventually de-emphasised.

- **TF `call`→`forward` redirect + the `__dict__.pop('loss')` fix (lines 318–321, 309):** Both are correct and necessary workarounds for Keras 3 semantics. The tab-note (lines 429–436) explains the `loss` shadowing issue clearly.

- **JAX `Module.training_step` signature discrepancy** — the fact that JAX requires `(params, batch, state)` vs. `(batch)` for other frameworks is surfaced correctly (the code differs visibly). The book does not hide this; it is the honest JAX-specific interface.
