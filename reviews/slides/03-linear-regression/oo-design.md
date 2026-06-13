# Slide outline — §3.2 `oo-design`

**Source:** `chapter_linear-regression/oo-design.md`
**Status:** ready to build (no stale content). Legacy `<!-- slides -->` block exists
(reasonable structure, but no cover/divider/kicker/`@fig:` and several
wall-of-code slides). Rebuild to the bar.
**Frameworks:** all 11 cells present in all four tabs. **Big overflow risk:** the
three base-class cells (`oo-design-models`, `-data`, `-training`) are long, and the
JAX variants are *much* longer (JAX `Trainer.fit` ~60 lines, `Module` uses
dataclasses). These must be handled with `@-` (code-only) and/or trimming, never shown
whole. See flags.

## What this section teaches
A reusable three-class scaffold — `Module` (model), `DataModule` (data), `Trainer`
(loop) — plus two notebook ergonomics tricks (`add_to_class`, `HyperParameters`) and a
live-plotting helper (`ProgressBoard`). The *idea* is the architecture and the
decoupling, not any one class body. This is a design/architecture deck: lead with the
class-relationship diagram; keep code to the few short, genuinely-illustrative cells.

## Code-cell inventory (notebook order)
| id | what | output | use |
|---|---|---|---|
| `oo-design-object-oriented-design-for-implementation` | imports | none | skip |
| `oo-design-utilities-1` | `def add_to_class` | none | show (6 lines, the trick) |
| `oo-design-utilities-2` | `class A` + instance | none | show (tiny) |
| `oo-design-utilities-3` | `@add_to_class(A) def do` + call | `Class attribute "b" is 1` | show (the payoff) |
| `oo-design-utilities-4` | `HyperParameters` stub | none | show (tiny) |
| `oo-design-utilities-5` | `class B(...)` using `save_hyperparameters` | `self.a = 1 self.b = 2 There is no self.c = True` | show |
| `oo-design-utilities-6` | `ProgressBoard` shell | none | `@-` code-only (long signature) |
| `oo-design-utilities-7` | draw sin/cos demo | `[plot]` | show output (`@!…` or `@…`) |
| `oo-design-models` | `Module` class | none | `@-` code-only, TRIMMED — overflow risk |
| `oo-design-data` | `DataModule` class | none | show (shortest of the three) |
| `oo-design-training` | `Trainer` class | none | `@-` code-only, TRIMMED — overflow risk |

## Diagrams
- **REUSE `img/mdl-linreg-oo-classes.svg`** — already the book figure `fig_oo_design`
  (the three boxes: `Trainer.fit` driving a `Module` over data from a `DataModule`,
  subclasses below). Inline via a plain image line, or — preferred — register it in the
  engine as `linear-regression-oo-classes` so it inherits deck fonts. This is the
  section's anchor figure; it carries the whole "three classes" idea.
- **NEW `linear-regression-progressboard-async` (optional)** — a small flow figure for
  the async-logging idea: hot compiled training step → `draw()` enqueues → background
  thread does device→host + matplotlib, dropping points if behind. Only if slide 8
  needs more than the demo output; the prose idea ("keep the hot path pure; push
  logging off to the side") is a nice candidate for a picture.

## Slide list

1. **Cover** — kicker "§3.2"; title "Object-Oriented Design — one scaffold for every
   model." Teaser `@fig:linear-regression-oo-classes`.
2. **Why a scaffold? (opener)** — `.cols .vc`: left, every model is the same loop
   (load → forward → loss → update → repeat); writing it per-chapter means a one-line
   change touches every chapter; the fix (à la Lightning) is `Trainer` once, `Module`
   / `DataModule` vary. Right `@fig:linear-regression-oo-classes`. `.d2l-note`: most
   chapters subclass only `Module` + `DataModule`.
3. **Divider 01 — Notebook Ergonomics.**
4. **`add_to_class`: split a class across cells** — `.cols`: left the trick
   `@oo-design-utilities-1`; right the use `@oo-design-utilities-2` then
   `@oo-design-utilities-3`. Caption: Python's class namespace is mutable — define the
   shell, attach methods later (so each method fits a slide-sized cell).
5. **`HyperParameters`: kill constructor boilerplate** — `@oo-design-utilities-4` then
   `@oo-design-utilities-5`. Caption: one `save_hyperparameters()` call and every
   ctor arg is `self.<name>`; `ignore=` drops the ones you don't want stored.
6. **`ProgressBoard`: live loss curves** — `.cols .vc`: left `@-oo-design-utilities-6`
   (code-only, the API) + `@oo-design-utilities-7` (the sin/cos demo output); right
   the optional async-flow figure or just the rendered curves. `.d2l-note`: `draw` is
   *asynchronous* — keep the compiled hot path pure, push plotting to a side thread.
7. **Divider 02 — The Three Classes.**
8. **`Module` — the model** — `@-oo-design-models` **trimmed** to the skeleton:
   `__init__` / `loss` / `forward` / `training_step` / `configure_optimizers`
   signatures only (drop `plot`/`validation_step` bodies). Caption: holds params,
   forward, loss, optimizer; every model subclasses it. **Overflow: show signatures,
   not full bodies** (esp. drop the long `plot` method).
9. **`DataModule` — the data** — `@oo-design-data` (it is short enough to show whole —
   `get_dataloader` hook + `train_dataloader`/`val_dataloader`). Caption: encapsulates
   "where do batches come from?"; subclasses override one hook.
10. **`Trainer` — the loop** — `@-oo-design-training` **trimmed** to the PyTorch
    skeleton (`__init__`, `prepare_data`, `prepare_model`, `fit`'s epoch loop;
    `fit_epoch` abstract). Caption: ties model + data, owns the epoch loop, drives the
    steps. **Overflow + framework risk:** the JAX `fit` (TrainState plumbing) is ~3×
    longer — either keep this slide PyTorch-trimmed and framework-neutral, or make it
    `except="jax"` with a short JAX-specific companion noting "JAX threads an explicit
    `TrainState` (params + optimizer state) through `fit` — same loop, functional
    style." Recommend the latter only if the trimmed shared version misleads.
11. **Recap** — three classes form the scaffold for the whole book; `add_to_class`
    (slide-sized cells), `HyperParameters` (no boilerplate), `ProgressBoard` (live
    curves, one call). We enrich these classes throughout the book.

## Notes & flags
- **Overflow is the dominant concern here.** `oo-design-models` and `oo-design-training`
  are walls of code; per §8 they must be `@-` code-only and trimmed, or split. Do NOT
  show the full JAX bodies. Run the overflow sweep after building.
- **Per-framework framing:** the *concept* (three classes) is identical; only the class
  bodies differ. So mostly **no `only=`/`except=`** — show one trimmed shared skeleton.
  The single place framing genuinely diverges is the `Trainer.fit` JAX TrainState
  plumbing (slide 10) — flag for the author to decide whether to scope it. The TF
  `_compile_steps`/`tf.function` detail belongs to §3.4, not here.
- Per-framework omissions: none (all 11 cells exist in all four tabs).
- Don't try to show `oo-design-utilities-6` output (it's a shell, raises
  `NotImplementedError`); use `@-` for code-only.
