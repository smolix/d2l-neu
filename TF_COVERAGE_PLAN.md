# Plan: TensorFlow Coverage via Keras 3

Goal: close the 32-notebook gap between TF and JAX by writing **Keras-first**
TF code. Prefer `keras.layers` / `keras.Model` / `compile()` / `fit()` over
raw `tf.*` / `tf.GradientTape` / `tf.nn.*`, falling back only when the topic
itself is about low-level mechanics.

Nothing below is implemented yet — this is the blueprint for review.

## Policy

**Default toolkit for new TF code:**

| Need | Preferred | Fallback |
|---|---|---|
| Model definition | `keras.Model` subclass with `__init__`/`call` | `keras.Sequential`, functional API |
| Layers | `keras.layers.*` (Dense, Conv2D, Embedding, MultiHeadAttention, …) | `tf.keras.*` (legacy namespace) |
| Training | `model.compile(...); model.fit(...)` on a `tf.data.Dataset` | Custom `train_step()` override on a `keras.Model` subclass |
| Multi-output / custom losses | Subclass `keras.Model`, override `train_step()` (Keras 3 supports this) | `tf.GradientTape` in a for-loop |
| Gradients / `GradientTape` | Only for chapters about gradients themselves (autograd, scratch implementations, GAN with two optimizers) | — |
| Multi-GPU | `tf.distribute.MirroredStrategy` with `with strategy.scope():` | raw `tf.function` + shard |
| Data loading | `tf.data.Dataset.from_tensor_slices(...).shuffle().batch().prefetch(tf.data.AUTOTUNE)` | — |
| Optimizers | `keras.optimizers.SGD/Adam/…` | — |
| Loss | `keras.losses.*` or pass loss string to `.compile()` | custom callable |
| Checkpoints | `model.save(...)`, `keras.models.load_model(...)` | `tf.train.Checkpoint` |
| Pretrained models | `keras.applications.ResNet50(weights='imagenet', ...)` | — |

`keras.*` and `tf.keras.*` are the same namespace in Keras 3 bundled with
TF 2.16+. Prefer `keras.*` for brevity.

For notebooks that teach low-level mechanics (autograd, scratch SGD/MLP,
GAN with two explicit optimizer updates), keep `tf.GradientTape`. This
matches current usage and the pedagogy.

## Audit of existing TF notebooks

I scanned all 118 `.md` files for `tf.keras` / `keras.*` vs low-level
patterns (`@tf.function`, `tf.GradientTape`, `apply_gradients`).
Most are already Keras-heavy; the low-level outliers are deliberate:

| File | Low-level hits | Verdict |
|---|---|---|
| `autograd.md` | 5× `tf.GradientTape` | **Keep** — topic is autograd itself |
| `linear-regression-scratch.md` | `GradientTape` + `apply_gradients` | **Keep** — "scratch" by design |
| `mlp.md` | 3× `tf.GradientTape` | **Keep** — scratch MLP |
| `maximum-likelihood.md` | 1× `tf.GradientTape` | **Keep** — math-appendix, teaches gradient |
| `multivariable-calculus.md` | 1× `tf.GradientTape` | **Keep** — math |
| `numerical-stability-and-init.md` | 1× `tf.GradientTape` | **Keep** — demonstrates vanishing gradients |
| `conv-layer.md` | 1× `tf.GradientTape` | **Keep** — teaches 2D conv from scratch |
| `utils.md` | 3× `tf.GradientTape` | **Keep** — shared d2l helpers |
| `gan.md` | 7× `GradientTape`, 7× Keras | **Borderline** — could switch to `train_step()` override with two optimizers, but current code is already readable; leave for now |
| `minibatch-sgd.md` | 3× `GradientTape` | **Keep** — teaches mini-batch SGD |
| `batch-norm.md` | 1× `@tf.function`, 10× Keras | **OK** — mostly Keras |
| `ndarray.md` | 1× `@tf.function` | **Keep** — shows graph-mode |
| `dcgan.md` | 1× low, 21× Keras | **OK** — mostly Keras already |

No changes proposed to existing TF code; these are pedagogically
well-placed. The main work is the 32 new notebooks.

## Per-notebook plan for the 32 gaps

Legend: **fit** = `model.compile()` + `model.fit()`; **train_step** =
`keras.Model` subclass with overridden `train_step`; **tape** =
`tf.GradientTape` loop (use only when the topic is gradients);
**MirroredStrategy** = multi-GPU via `tf.distribute`; **(copy)** = data
code mirrors PT/MX versions closely, limited new logic.

### computer-vision (12 notebooks)

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 1 | `multiscale-object-detection` | (copy) | Just reads an image and calls `multibox_prior`; the hard code is already in `anchor.md`. Small port. |
| 2 | `object-detection-dataset` | (copy) | `tf.data` pipeline over banana images, similar to PT's `BananasDataset`. |
| 3 | `rcnn` | fit (none — mostly prose) | Mostly explanation; ~3 code cells. |
| 4 | `semantic-segmentation-and-dataset` | (copy) | `tf.data` pipeline over VOC; crop + one-hot labels. |
| 5 | `fcn` | fit — Keras U-net-style model with `Conv2DTranspose` | Pretrained backbone via `keras.applications.ResNet50(include_top=False)`; classifier head per-pixel; `SparseCategoricalCrossentropy`. |
| 6 | `fine-tuning` | fit — standard transfer learning | Port of the JAX version but with `net.compile/fit` directly (no stateless dance); `keras.applications.ResNet50(weights='imagenet')` backbone. |
| 7 | `image-augmentation` | fit — small CNN on CIFAR-10 + `tf.keras.layers.RandomCrop/Flip/...` | `keras.Sequential` preprocessing layers; `compile/fit` on `tf.data`. |
| 8 | `kaggle-cifar10` | fit | Similar to `image-augmentation` with Kaggle submission file generation. |
| 9 | `kaggle-dog` | fit — `ResNet50` finetune | Same pattern as `fine-tuning`. |
| 10 | `neural-style` | tape | Style transfer is *inherently* custom: two losses on feature maps, optimize the image (not the model). `GradientTape` on a `tf.Variable` image. Keep low-level. |
| 11 | `anchor` | (copy) | Pure NumPy/TF ops; `multibox_prior`, `box_iou`, `assign_anchor_to_bbox`, `multibox_target`. Port the JAX versions to `tf.*`. No training. |
| 12 | `ssd` | train_step | TinySSD: subclass `keras.Model` with `call()` returning `(anchors, cls_preds, bbox_preds)`; override `train_step` to compute the multibox loss. Prediction uses NumPy NMS (same as JAX fix). |

### computational-performance (4 notebooks)

These chapters teach infrastructure. Use `tf.distribute` where possible;
fall back to `tf.function` where the topic is graph compilation.

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 13 | `async-computation` | tape/`@tf.function` | Chapter about async vs sync execution. Show `@tf.function` vs eager; minimal new Keras code. |
| 14 | `auto-parallelism` | `@tf.function` + timer | Uses `tf.function` to demonstrate parallel op scheduling on two GPUs. |
| 15 | `multiple-gpus` | MirroredStrategy + tape | From-scratch multi-GPU teaching chapter; use `tf.distribute.MirroredStrategy` with a custom tf.function train step so the mechanics are visible. |
| 16 | `multiple-gpus-concise` | MirroredStrategy + fit | The "concise" version; `with strategy.scope(): model = ...; model.compile(...); model.fit(...)`. |

### hyperparameter-optimization (3 notebooks)

These use `syne_tune`. The training-function inside each HPO trial can
be a short Keras `compile/fit` call, independent of framework.

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 17 | `hyperopt-intro` | fit (softmax regression) | Trivial Keras training inside the tune objective. |
| 18 | `hyperopt-api` | fit (LeNet) | Same idea with a small CNN. |
| 19 | `sh-intro` | fit (LeNet) | Same. |

For all three, the tune-driver code is `syne_tune`-specific and largely
framework-neutral; only the `objective` function needs TF. Each ~30 LoC.

### natural-language-processing-applications (6 notebooks)

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 20 | `sentiment-analysis-and-dataset` | (copy) | IMDB dataset loader — port the JAX `tf.data` pipeline (already TF!). |
| 21 | `sentiment-analysis-rnn` | fit — BiLSTM classifier | `keras.layers.Bidirectional(keras.layers.LSTM(...))` + `Dense` head. |
| 22 | `sentiment-analysis-cnn` | fit — TextCNN | Parallel `keras.layers.Conv1D` branches + `GlobalMaxPooling1D`, concat, Dense. |
| 23 | `natural-language-inference-and-dataset` | (copy) | SNLI loader, `tf.data` pipeline. |
| 24 | `natural-language-inference-attention` | fit — decomposable attention | Keras subclassed model with MLP attend/compare/aggregate. |
| 25 | `natural-language-inference-bert` | fit — finetune BERT | Same hazard as fine-tuning — load BERT via `keras.layers` / KerasHub; use `compile/fit`. If no bundled BERT, build a minimal Transformer encoder (mirrors JAX). |

### natural-language-processing-pretraining (6 notebooks)

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 26 | `word-embedding-dataset` | (copy) | Port PTB loader; reuses the new `_pad_ptb` helper in numpy. TF version yields pre-padded tensors via `tf.data.Dataset.from_tensor_slices(...)`. |
| 27 | `word2vec-pretraining` | train_step | Skip-gram with negative sampling: define a `keras.Model` subclass (two `Embedding` layers), override `train_step` because the loss is `BinaryCrossentropy(from_logits=True)` with a mask. |
| 28 | `similarity-analogy` | (copy) | GloVe lookup — no training. |
| 29 | `subword-embedding` (PT-only today, but natural to add) | **not in scope** — skipping since goal is JAX-matching coverage; PT version only |
| 30 | `bert-dataset` | (copy) | WikiText-2 MLM/NSP dataset; port the Python logic unchanged (it's framework-agnostic). |
| 31 | `bert` | fit — build BERT encoder | `keras.Model` subclass with `TransformerEncoder` blocks (reuse `transformer.md`'s). Output: `mlm_head`, `nsp_head`. No training here. |
| 32 | `bert-pretraining` | train_step | Custom loss over MLM + NSP; subclass + `train_step` override; `compile(optimizer=Adam(), ...); fit(ds)`. |

### attention-mechanisms-and-transformers (1 notebook)

| # | Notebook | Approach | Notes |
|---|---|---|---|
| 33 | `vision-transformer` | fit | `keras.Model` subclass: patch embed (`Conv2D` with stride=patch_size) + cls token + positional embedding + `TransformerEncoder` stack + head. Use `d2l.Trainer` if consistent with other Transformer notebooks, otherwise `compile/fit`. |

(That's 32 gaps; `subword-embedding` is listed as out-of-scope since PT-only.)

## Approach summary

- **`fit()` notebooks (19)**: standard classifier / transfer learning /
  pre-trained backbone pattern. Each ~50–100 LoC of TF code.
- **`train_step` subclass (4)**: SSD, word2vec, BERT-pretraining, and
  optionally sentiment-CNN if a mask makes `compile()` awkward. Each
  ~80–120 LoC.
- **`tape` (2)**: `neural-style`, `multiple-gpus` — pedagogy demands it.
- **`MirroredStrategy` (2)**: `multiple-gpus(-concise)` — canonical TF
  multi-GPU.
- **`@tf.function`-centric (2)**: `async-computation`, `auto-parallelism` —
  these *are about* graph compilation.
- **"Copy" (8)**: data loaders and prose-heavy chapters; limited new
  logic.

## Dependencies between notebooks

Some notebooks define functions that others reuse via `#@save`:

1. `anchor.md` → defines `multibox_prior`, `box_iou`,
   `assign_anchor_to_bbox`, `multibox_target`, `nms`, `multibox_detection`.
   Consumed by `ssd.md`, `object-detection-dataset.md`,
   `multiscale-object-detection.md`. **Must port first.**
2. `bert.md` → defines `BERTModel`, `MaskLM`, `NextSentencePred`.
   Consumed by `bert-pretraining.md`, `natural-language-inference-bert.md`.
   **Must port before BERT trainers.**
3. `word-embedding-dataset.md` → defines `load_data_ptb`, `_pad_ptb`.
   Consumed by `word2vec-pretraining.md`. **Must port first.**
4. `bert-dataset.md` → defines `_WikiTextDataset`. Consumed by
   `bert-pretraining.md`. **Must port first.**
5. `sentiment-analysis-and-dataset.md` → defines `load_data_imdb`.
   Consumed by `sentiment-analysis-rnn/cnn.md`. **Must port first.**
6. `natural-language-inference-and-dataset.md` → `load_data_snli`.
   Consumed by `nli-attention`, `nli-bert`. **Must port first.**

## Suggested implementation order

Following the dependency graph, in batches of ~5 that can be worked on
in parallel by agents:

**Batch 1 — dependency-free infrastructure (5)**
- `anchor`, `object-detection-dataset`, `multiscale-object-detection`,
  `sentiment-analysis-and-dataset`, `natural-language-inference-and-dataset`

**Batch 2 — data loaders and trainable backbones (5)**
- `image-augmentation`, `fine-tuning`, `bert-dataset`,
  `word-embedding-dataset`, `similarity-analogy`

**Batch 3 — classifiers that use the Batch 1–2 data loaders (6)**
- `sentiment-analysis-rnn`, `sentiment-analysis-cnn`,
  `natural-language-inference-attention`,
  `kaggle-cifar10`, `kaggle-dog`, `rcnn`

**Batch 4 — model-definition notebooks (3)**
- `semantic-segmentation-and-dataset`, `bert`, `vision-transformer`

**Batch 5 — heavier training notebooks (4)**
- `fcn`, `ssd`, `word2vec-pretraining`, `neural-style`

**Batch 6 — infrastructure + BERT trainers (5)**
- `multiple-gpus`, `multiple-gpus-concise`, `async-computation`,
  `auto-parallelism`, `bert-pretraining`

**Batch 7 — HPO (3)**
- `hyperopt-intro`, `hyperopt-api`, `sh-intro`
- These depend on `LeNet` / `SoftmaxRegression` from earlier chapters'
  TF code, all of which already exists.

Each batch = parallel agents on a shared chapter, no file conflicts.
Batches run sequentially because later batches may need to call into
the `#@save` functions from earlier batches' library builds.

## Estimated code volume

~30 notebooks × ~50–100 LoC of new TF code = **~1800 LoC across all
chapters**. About 80% of that is boilerplate that mirrors the PT/JAX
versions.

## Notes on Keras 3 specifics to exploit

- **`keras.layers.MultiHeadAttention`** saves lots of code in
  attention/transformer/ViT/NLI/BERT notebooks; matches what the JAX
  version builds by hand.
- **`keras.layers.Embedding(..., mask_zero=True)`** auto-propagates the
  mask through LSTM/Bidirectional, simplifying NLP trainers vs manual
  `valid_len` threading.
- **`keras.applications.*`** (ResNet50, VGG16, Xception, …) — pretrained
  backbones with `weights='imagenet'`. Big savings in `fine-tuning`,
  `kaggle-*`, `fcn`, `neural-style`.
- **Keras 3 custom `train_step(self, data)`** — lets us keep `fit()`'s
  progress bar, callbacks, metrics while owning the gradient step.
  Useful in SSD, word2vec, BERT-pretraining.
- **`tf.distribute.MirroredStrategy`** — one-line multi-GPU wrap: any
  `compile()`/`fit()` inside `strategy.scope()` uses all visible GPUs.
- **`keras.Metric`** subclass (or `"accuracy"` string) — avoids manual
  metric accumulation that JAX notebooks have to do by hand.
- **`model.save('mymodel.keras')`** — portable `.keras` format; replaces
  `tf.saved_model` dances.

## What to watch for

- **Shape conventions**: TF/Keras is NHWC by default (matches JAX/Flax).
  PT is NCHW. Port needs careful attention to `Conv2D` input shape.
- **Mask handling in NLP**: Keras `Masking`/`mask_zero=True` is more
  idiomatic than the manual `valid_len` pattern; code is shorter but
  pedagogically different. For consistency with the rest of the book,
  consider keeping an explicit mask tensor rather than relying on
  implicit propagation, unless the text is willing to introduce Keras's
  masking concept.
- **ViT / BERT**: Keras 3 does not bundle BERT. Options: (a) build the
  Transformer encoder from `keras.layers.MultiHeadAttention` + a custom
  `TransformerEncoder` layer (parallel to the JAX implementation — keeps
  teaching value); (b) depend on `keras-hub` which does. Plan
  **(a)** — no extra dependency, and the pedagogy matches the existing
  `transformer.md` teaching.
- **BERT dataset size / memory**: existing PT/MXNet versions already
  handle this; port the numpy prep exactly and just convert to
  `tf.data.Dataset` at the end.
- **Pretrained BERT weights**: the current PT notebook uses
  `d2l.load_pretrained_model`, a custom loader. TF version should do the
  same rather than pull HuggingFace — keep the book self-contained.

## Final delivery

When implementing:
- One commit per batch.
- Each batch: regenerate notebooks, run the new TF notebooks (they're
  the only ones affected), commit + push only after they pass.
- Update `coverage.md` after each batch.
