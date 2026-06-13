# Slide outline — §4.2 The Image Classification Dataset (`image-classification-dataset.md`)

**Status: fresh, executed in all four frameworks.** All four notebooks ran (5
cells w/ output + 1 image each in the committed store). The existing
`<!-- slides -->` block is legacy-shaped (10 flat `.slide`s, code-forward, no
cover/divider/kicker/diagram) — treat as a source of ideas, rebuild to the bar.

**One staleness flag (cosmetic, MXNet only):** the **MXNet** store outputs for
the shape/timing cells are polluted with repeated `Storage type fallback
detected: operator = stack …` stderr lines (full local Google-Drive paths leak
into the text). Cells affected: `…loading-the-dataset-3`,
`…reading-a-minibatch-2`, `…reading-a-minibatch-3`. Not an error, but ugly on a
slide. **Flag for the MXNet deck:** trim with `output-lines=` / show only the
`(1,32,32)` result line, or re-capture MXNet with the storage-fallback warning
silenced. Does not block the pytorch/jax/tf decks. (Note: this same noise will
hit the book/PDF too — worth a separate capture cleanup, out of scope here.)

**Cross-framework framing difference — the teaching point of this section:**
channel axis position differs and the outputs prove it. This warrants a
**`only=`/`except=` scoped pair** plus a framework diagram variant:

| cell | pytorch / mxnet (channel-first) | tensorflow / jax (channel-last) |
|---|---|---|
| `…loading-the-dataset-3` | `(1, 32, 32)` = $(c,h,w)$ | `(32, 32, 1)` = $(h,w,c)$ |
| `…reading-a-minibatch-2` | `(64, 1, 32, 32)` | `(64, 32, 32, 1)` |

The `…loading-the-dataset-2` cell also differs in **code** (mxnet/pytorch:
`len(data.train)`; tf/jax: `len(data.train[0])`) but same output `(60000,
10000)` — that is an automatic `#@tab` swap, **no scoping needed**.

**Per-framework cell presence:** all 10 cells present in all four frameworks
(every cell is a 4-way `#@tab` set). **No missing-sibling gaps.** No cells to
port.

**Diagrams:** one NEW engine diagram (`diagrams/image-classification-dataset.mjs`,
register it) — the channel-axis layout, with a `-jax`/channel-last variant.
The dataset *sample grid* is a real computed output (`@!…visualization-2`), not a
diagram.

---

## Outline

### 0. Cover
- `::: {.cover}` — kicker `Dive into Deep Learning · §4.2`.
- "**Fashion-MNIST** — the dataset every classifier in this chapter trains on."
- Teaser: `@!image-classification-dataset-visualization-2` (the 8-tile sample
  grid, output-only). Strong visual cover.

### 1. Why / what opener — "Why Fashion-MNIST, not MNIST"
- One idea: MNIST is too easy (linear models > 95%); Fashion-MNIST is a drop-in
  replacement, same shape/API, but classes are genuinely confusable, so model
  differences show.
- 2-col. Left: bullets — 10 clothing classes, 28×28 grayscale, 60k/10k split,
  drop-in for MNIST, harder; ImageNet is the real-scale benchmark but too big to
  stay interactive. Right (`.fig`): `@!image-classification-dataset-visualization-2`
  (sample grid) — or move grid to cover and put a `.d2l-note` here.

### 2. Setup: a reusable `DataModule`
- One idea: wrap the framework's built-in Fashion-MNIST in a `DataModule` so
  every model in the chapter reuses the same loaders.
- `@image-classification-dataset-the-image-classification-dataset` (imports) +
  `@image-classification-dataset-loading-the-dataset-1` (the `FashionMNIST`
  class). Code-forward but short; consider `@-` for imports to save height.
- `.d2l-note`: all major frameworks ship Fashion-MNIST preprocessed; we only
  wrap it.

### 3. Instantiate (resize to 32×32)
- One idea: instantiate, resizing to 32×32 to match later ConvNet inputs; 60k
  train / 10k test.
- `@image-classification-dataset-loading-the-dataset-2` → `(60000, 10000)`.
  (Code auto-swaps `len(data.train)` vs `len(data.train[0])` by framework.)

### 4. Where does the channel axis live? — **scoped pair**
- One idea: a grayscale image is one channel; **where that axis sits differs by
  framework**, and this is the one cross-framework gotcha to internalize now.
- **Slide 4a `except="tensorflow,jax"`** (pytorch/mxnet, channel-first):
  `@image-classification-dataset-loading-the-dataset-3` → `(1, 32, 32)`; caption
  "$(c,h,w)$ — channel first." Right (`.fig`): **NEW diagram
  `image-classification-dataset-channel-axis`** (channel-first variant) — a
  $1\times32\times32$ box with the channel axis labelled first.
- **Slide 4b `only="tensorflow,jax"`** (channel-last):
  `@image-classification-dataset-loading-the-dataset-3` → `(32, 32, 1)`; caption
  "$(h,w,c)$ — channel last." Right: `@fig:image-classification-dataset-channel-axis@jax`
  (the channel-last variant SVG).
- `.d2l-note` (both): `get_dataloader` produces the right layout per framework,
  so the rest of the chapter never thinks about it again.

### 5. Human-readable labels
- One idea: labels are stored 0–9; a helper maps each to its English name.
- `@image-classification-dataset-loading-the-dataset-4` (`text_labels`). Short;
  pair with a `.d2l-note` listing the 10 names, or show the list as a small
  static grid (no diagram needed).

### 6. Reading a minibatch
- One idea: the dataloader yields shuffled minibatches in a fixed shape per
  framework.
- `@image-classification-dataset-reading-a-minibatch-1` (the `get_dataloader`
  method — long for TF/JAX with the `tf.data` pipeline; consider `@-` code-only
  to avoid overflow, or trim the comment). Fragment (`. . .`):
  `@image-classification-dataset-reading-a-minibatch-2` → batch shape (channel
  position again visible: `(64,1,32,32)` vs `(64,32,32,1)`).
- **Overflow watch:** the TF/JAX `get_dataloader` is the tallest code cell in
  this section. If 4a/4b + this both show it, prefer `@-` here.

### 7. Throughput sanity check
- One idea: time one full epoch; loading should not be the bottleneck (a fwd+bwd
  pass is 10–100× the I/O).
- `@image-classification-dataset-reading-a-minibatch-3` → `'2.49 sec'` (pytorch),
  `'0.15–0.16 sec'` (tf/jax), `'0.87 sec'` (mxnet). **MXNet:** trim the
  storage-fallback stderr (see staleness flag) — show only the timing string.
- `.d2l-note`: if I/O *were* slower than compute, overlap with prefetch / more
  workers.

### 8. Visualize a batch
- One idea: always eyeball your data; the `visualize` helper tiles a batch and
  labels each tile with its class name.
- `@image-classification-dataset-visualization-2` (code + the 8-tile SVG grid).
  Skip `…visualization-1` (the `show_images` stub raises `NotImplementedError` —
  it is a deliberate interface stub, **do not show its "output"**; mention
  `show_images` lives in `d2l` in one line if needed).
- Caption: false-color palette is for visibility; the data is grayscale.

### 9. Recap
- Fashion-MNIST: 10 classes, 28×28 grayscale, harder than MNIST, drop-in API.
- A `DataModule` subclass owns loaders, label decoding, and `visualize`.
- **Channel axis:** $(c,h,w)$ in PyTorch/MXNet, $(h,w,c)$ in TF/JAX — the loader
  hides it from here on.
- Sanity-check throughput; same data API drives every model in §4.

---

## Diagram inventory

Create `diagrams/image-classification-dataset.mjs`, register in `registry.mjs`:

| id | draws | variant |
|---|---|---|
| `image-classification-dataset-channel-axis` | a 32×32 single-channel image tensor with the channel axis labelled, channel-**first** $(1,32,32)$ | base |
| `image-classification-dataset-channel-axis-jax` | same, channel-**last** $(32,32,1)$ | `@jax` (also used for the TF deck via `only=` slide) | 

Both small; reuse `grid`/`block`/`tx`. Engine `grid` already draws a 2-D cell
array — wrap with an annotated channel axis arrow.

## Curation notes (drops from the legacy block)
- Drop `…visualization-1` as a *shown* cell (it is a raising stub).
- Fold the two "what does one example look like" + minibatch shape ideas into
  the scoped channel-axis pair so the cross-framework point lands once, clearly.

## Per-framework summary
- **Scoped slides required:** the channel-axis pair (4a `except=tf,jax` / 4b
  `only=tf,jax`) + the `-jax` diagram variant. Everything else is automatic
  `#@tab` code/output swaps.
- **No cells to port** (all 10 cells exist for all four frameworks).
- **MXNet staleness:** stderr noise in 3 cells' outputs — trim on slides;
  ideally re-capture clean.
