# Slide outline — §3.3 `synthetic-regression-data`

**Source:** `chapter_linear-regression/synthetic-regression-data.md`
**Status:** ready to build (no stale content). Legacy `<!-- slides -->` block exists
(decent shape, no cover/divider/kicker/`@fig:`). Rebuild to the bar.
**Frameworks:** all 10 cells present in all four tabs; outputs captured for all four.
One genuine framing divergence (JAX reports **31** batches, others **32**) — one
framework-aware caption or a short scoped slide.

## What this section teaches
Why synthetic data (known $\mathbf w^*,b^*$ → any failure is the algorithm, not the
data), packaged as a `DataModule` subclass; then the *same* dataloader protocol twice —
a transparent hand-rolled minibatch iterator and the framework-native loader
(workers/prefetch/shuffle). The teaching beats are: the generative equation, "known
ground truth," one example's shape, and hand-rolled-vs-builtin (same interface).

## Code-cell inventory (notebook order)
| id | what | output | use |
|---|---|---|---|
| `synthetic-regression-data` | imports | none | skip |
| `…-generating-the-dataset-1` | `SyntheticRegressionData.__init__` (draws X, computes y) | none | show (the generator) |
| `…-generating-the-dataset-2` | instantiate w=[2,-3.4], b=4.2 | none | show (tiny — ground truth) |
| `…-generating-the-dataset-3` | print first example | `features: tensor([-0.2621, -0.2395])` / `label: tensor([4.4742])` | show |
| `…-reading-the-dataset-1` | hand-rolled `get_dataloader` (shuffle + yield) | none | show (the iterator) |
| `…-reading-the-dataset-2` | grab one minibatch, print shapes | `X shape: …[32, 2]` / `y shape: …[32, 1]` | show |
| `…-concise-…-data-loader-1` | `get_tensorloader` (framework builtin) | none | show |
| `…-concise-…-data-loader-2` | re-register `get_dataloader` via builtin | none | show (4 lines) |
| `…-concise-…-data-loader-3` | one minibatch, print shapes | same `[32,2]/[32,1]` | show |
| `…-concise-…-data-loader-4` | `len(dl)` | `32` (pt/tf/mxnet); **`31` (JAX)** | show |

## Diagrams
- **NEW `linear-regression-synthetic-pipeline`** — the generative process as a flow:
  draw $\mathbf X\sim\mathcal N$ (a $1000\times2$ grid chip) → multiply by true
  $\mathbf w=[2,-3.4]$, add $b=4.2$ → add noise $\epsilon\sim\mathcal N(0,\sigma^2)$ →
  labels $\mathbf y$. One picture for "we built it, so we know the answer." Use the
  `matvec` grid idiom; values match `…-generating-the-dataset-2`.
- **NEW `linear-regression-minibatch-shuffle`** — the dataloader idea: a long index
  vector shuffled, then sliced into batches of 32 (last batch partial → 8). Annotate
  31 full + 1 partial = 32 batches (and note JAX drops the partial → 31). Pairs with
  the hand-rolled iterator and the `len(dl)` slide; carries the 31-vs-32 point
  visually. Use `grid` with `opt.fill` to colour batch boundaries.
- No inline computed plots in this section (no figures produced by the cells).

## Slide list

1. **Cover** — kicker "§3.3"; title "Synthetic Regression Data — a problem whose
   answer we already know." Teaser `@fig:linear-regression-synthetic-pipeline`.
2. **Why synthetic? (opener)** — `.cols .vc`: left, real data conflates three failure
   sources (model / optimizer / data); synthetic removes the ambiguity — if we fix
   $\mathbf w^*,b^*,\sigma$, any failure to recover them is the algorithm's fault.
   Right the generative equation
   $\mathbf y=\mathbf X\mathbf w+b+\boldsymbol\epsilon$. `.d2l-note`: the first test
   for any new method.
3. **Divider 01 — Generate.**
4. **A `DataModule` that makes data** — `.cols`: left
   `@…-generating-the-dataset-1` (draws X, computes y in `__init__`); right
   `@fig:linear-regression-synthetic-pipeline`. Caption: `save_hyperparameters()`
   keeps `w, b, noise, splits, batch_size` introspectable.
5. **Known ground truth** — `@…-generating-the-dataset-2` then
   `@…-generating-the-dataset-3`. Caption: true `w=[2,-3.4]`, `b=4.2`; each row of
   `features` ∈ $\mathbb R^2$, each `label` a scalar — verify after training.
6. **Divider 02 — Read in Minibatches.**
7. **A hand-rolled dataloader** — `.cols .vc`: left
   `@…-reading-the-dataset-1` (shuffle indices, yield slices) + `@…-reading-the-dataset-2`
   (shapes `[32,2]/[32,1]`); right `@fig:linear-regression-minibatch-shuffle`.
   Caption: educational but slow — Python index loop, no prefetch, all in memory.
   *(`. . .` only at slide top level if used — not inside `.col`.)*
8. **The framework dataloader** — `@…-concise-…-data-loader-1` then
   `@…-concise-…-data-loader-2`. Caption: wrap X,y in the builtin dataset/loader —
   workers, prefetch, shuffling for free; identical caller interface.
9. **Same interface, batch count** — `@…-concise-…-data-loader-3`
   (same shapes) then `@…-concise-…-data-loader-4` (`len(dl)`). Caption: 1000 train ÷
   32 = 32 batches (31 full + a partial 8). **Framework note (JAX):** the JAX loader
   reports **31** — `drop_remainder=True` discards the partial last batch so every
   `@jax.jit` step keeps one shape. Either bake this into the caption (it renders the
   right number per tab automatically via the cell output) **or** make a one-line
   `only="jax"` companion explaining the drop. Recommend: neutral caption +
   parenthetical, since the output cell already shows 31 vs 32 correctly.
10. **Recap** — synthetic data → ground-truth `w,b` to check against; `DataModule`
    encapsulates "where batches come from," reusable across models; hand-rolled vs
    framework loader — same protocol, framework wins on speed/ergonomics.

## Notes & flags
- **Per-framework framing:** essentially a code/output swap. The ONLY divergence is the
  batch count (31 JAX vs 32 others), and the cell output carries it correctly per tab —
  so a framework-neutral caption suffices; a scoped `only="jax"` slide is optional, not
  required. Lean toward **zero scoped slides** unless the author wants to spotlight the
  `drop_remainder` reasoning.
- The JAX import cell pulls in `tensorflow` + `tensorflow_datasets` (it borrows TF's
  loader). Not slide-worthy; it's setup (skip).
- Per-framework omissions: none (all 10 cells exist in all four tabs).
- Captions must say `n=1000`, `batch_size=32`, true `w=[2,-3.4]`, `b=4.2` — match the
  cells exactly.
