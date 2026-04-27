# Convergence & Speed Audit

Generated 2026-04-27 from `logs/make-all.log` (513 successful notebook
runs across PyTorch, TensorFlow, JAX, MXNet — all 4 frameworks built
cleanly; 0 failures).

This file is a *reporting* artifact: each notebook below is flagged
with what is wrong, **not** how to fix it. Suggested causes are
hypotheses and need verification before any code change.

---

## Summary

- **Convergence:** 4 confirmed cross-framework convergence problems, 2
  previously-flagged issues now resolved, 2 false positives dismissed,
  and ~10 notebooks spot-checked clean.
- **Speed:** 49 distinct notebooks have at least one framework taking
  >120 s. 13 of those have one framework ≥3× slower than its peers;
  most are MXNet (no `hybridize()` call) but TF and JAX outliers exist
  (mostly from missing `@tf.function` / `@jax.jit` and from the JAX tab
  using a TF backbone via host round-trips).

| metric | PT | TF | JAX | MX |
|---|---:|---:|---:|---:|
| notebooks executed | 139 | 123 | 123 | 128 |
| total runtime | 116 min | 182 min | 209 min | 297 min |
| avg per notebook | 50 s | 89 s | 102 s | 139 s |

Source data:
- `logs/timings.tsv` — all (framework, notebook, time_s) rows.
- `logs/framework_outliers.tsv` — notebooks where one framework is ≥3× slower.
- `logs/convergence-all.tsv` — per-curve SVG-pixel summary (369 rows).
- `logs/convergence-compare.md` — auto-generated cross-framework flag list.
- `logs/training_metrics.tsv` — text-extracted final-epoch metrics.
- `logs/metric_compare.tsv` — cross-framework metric spread.

---

## Convergence

### Confirmed issues

#### `chapter_convolutional-modern/nin.ipynb` — JAX NiN diverges after epoch 3
- **Framework:** JAX
- **Symptom:** `val_loss` is recorded for only 3 epochs (then NaN); `val_acc` collapses to ~0% from epoch 4 onward and stays there. PT/TF/MX all converge normally to ~80–90% val_acc at the same `lr=0.05`.
- **Suggested cause:** Numerical instability in the JAX NiN trainer at lr=0.05. After epoch 3 the loss becomes NaN; subsequent argmax over NaN logits yields a constant class. Possibly a missing dropout, an uninitialized parameter, or a gradient-clipping difference vs. the PT/TF d2l Trainer.

#### `chapter_convolutional-modern/cnn-design.ipynb` — MXNet RegNetX32 val_loss plateau
- **Framework:** MXNet
- **Symptom:** Orange (val_loss) curve is essentially flat across 10 epochs (4-pixel shift on a 126-pixel axis vs. PT's 34-pixel drop on the same axis scale). val_acc improves only marginally. PT/TF/JAX all converge strongly.
- **Suggested cause:** The MX Gluon RegNetX32 likely uses a different weight initialisation or BN momentum default than the PT/TF/JAX versions, slowing generalisation despite the same lr=0.05 setup.

#### `chapter_recommender-systems/neumf.ipynb` — MXNet NeuMF metrics flat
- **Framework:** MXNet
- **Symptom:** Both tracked curves are perfectly flat across all 10 epochs. Final printed metrics: MXNet `test hit rate 0.075, test AUC 0.733` vs. PyTorch `test hit rate 0.296, test AUC 0.861`. PT shows monotonic improvement; MX shows no epoch-to-epoch change at all.
- **Suggested cause:** The MX Gluon BPRLoss or NeuMF Gluon model may not propagate gradients correctly for the ranking objective (different autograd graph path or `detach` behaviour vs. PyTorch), leaving the model effectively frozen at initialisation.

#### `chapter_computer-vision/kaggle-cifar10.ipynb` — JAX/TF LR schedule decays per-step instead of per-epoch
- **Framework(s):** JAX (severe), TensorFlow (moderate)
- **Symptom:** Final printed metrics:
  - JAX `train loss 2.108, train acc 0.220, valid acc 0.188` (near-random for 10 classes)
  - TF `train loss 1.426, train acc 0.497, valid acc 0.391`
  - PT `train loss ~0.60, train acc ~0.78, valid acc ~0.45`
  - MX similar to PT
- This was not flagged by the SVG-pixel heuristic because the curves all technically descend.
- **Suggested cause:** PT uses `torch.optim.lr_scheduler.StepLR(step_size=lr_period)` which decays every `lr_period` *epochs* (`scheduler.step()` called once per epoch). The JAX implementation uses `optax.exponential_decay(transition_steps=lr_period, staircase=True)` and TF uses `keras.ExponentialDecay(decay_steps=lr_period)`, both of which count *gradient-update steps*. With `lr_period=4` and ~50 batches/epoch, JAX's LR decays by 0.9× every 4 steps — reaching ~10⁻¹⁶ within two epochs. TF's counting differs slightly, giving intermediate-but-still-poor performance. The fix would be `transition_steps = lr_period * num_batches_per_epoch`.

### Resolved since prior audit (track for regression)

#### `chapter_convolutional-modern/batch-norm.ipynb` — TF BNLeNet now trains
- Previous run had `BNLeNet` (Keras `BatchNormalization`) producing rising val_loss and flat val_acc at lr=0.5. Current run converges normally (val_acc full-range improvement, val_loss descending). The fix likely landed in the d2l TF library or notebook source. **Worth a regression watch.**

#### `chapter_convolutional-modern/densenet.ipynb` — PT DenseNet now converges
- Prior run showed PT val_loss rising (+0.47 rel_drop) while training loss fell — overfitting. Current run: val_loss descends (rel_drop = −0.98), val_acc improves. **Worth a regression watch.**

### Other suspicious metric spreads (text-extracted finals)

These came from grepping `text/plain` outputs in the executed notebooks
(`logs/metric_compare.tsv`); they should be confirmed by inspecting the
training plots before treating them as bugs.

| notebook | metric | PT | TF | JAX | MX | spread | note |
|---|---|---|---|---|---|---|---|
| `chapter_convolutional-neural-networks/conv-layer.ipynb` | loss | 0.029 | 0.28 | 1.21 | 0.004 | 121× | JAX from-scratch `Conv2D` converges much worse than peers; this is the toy 6×8 conv-from-scratch demo, so not a model bug, but worth a note in the prose if the discrepancy is misleading. |
| `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` | loss | — | 0.262 | 0.014 | — | 18× | TF final loss ~18× higher than JAX. Note: differing reduction conventions can inflate this; verify with val_acc before flagging as a bug. |
| `chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb` | loss | — | 0.236 | 0.051 | — | 4.6× | Same caveat as above. |
| `chapter_attention-mechanisms-and-transformers/transformer.ipynb` | bleu | 1.0 | 0.522 | 1.0 | 0.522 | 0.48 | TF and MX cap at BLEU 0.52 while PT/JAX hit 1.0 on the same eval sentences. Likely a tokenisation / decoding difference; needs verification. |
| `chapter_computer-vision/fine-tuning.ipynb` | acc | — | 0.793 | 0.604 | — | 0.19 | JAX final acc ~19 pp lower than TF on the same hot-dog dataset. Already documented in `deep-scan-rest.md` (JAX uses ResNet-50 from-scratch / non-pretrained where TF uses pretrained ImageNet). |

### Heuristic flags dismissed as false positives

- **`chapter_convolutional-modern/resnet.ipynb` (TF, orange DRIFT)** — TF ResNet just starts at a higher initial val_loss than peers, so the relative drop is larger. Final position is comparable; no training failure.
- **`chapter_optimization/sgd.ipynb` (PT, blue SIGN-FLIP)** — The "blue path" is a `d2l.show_trace_2d()` contour trace, not a loss curve. Matplotlib renders the contour from a different starting point on PT vs. peers, flipping the rel_drop sign. The actual SGD trajectory is fine.

### Spot-checked clean

PT/TF/JAX/MX final pixel positions matched within axis tolerance for:
`chapter_convolutional-neural-networks/lenet.ipynb`,
`chapter_recurrent-modern/lstm.ipynb`,
`chapter_recurrent-modern/gru.ipynb`,
`chapter_recurrent-neural-networks/rnn-scratch.ipynb`,
`chapter_linear-classification/softmax-regression-scratch.ipynb`,
`chapter_convolutional-modern/vgg.ipynb`,
`chapter_convolutional-modern/alexnet.ipynb`,
`chapter_convolutional-modern/googlenet.ipynb`,
`chapter_computer-vision/ssd.ipynb`,
`chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb`.

### Convergence audit caveats

- The `rel_drop` heuristic compares pixel-space curve direction; it cannot detect a curve that converged to a *worse* absolute value than peers if both still descended.
- Curves that descend then rebound (overfitting) only register if the last-point comparison catches them.
- Color-based mapping assumes each framework plots curves in the same order; custom training loops (GAN, recommender) don't follow this — those are filtered out by the min-points threshold.
- Final-metric scrape via `logs/training_metrics.tsv` is regex-based; values are best-effort and a few rows may carry early-iteration values rather than the final-epoch print.

---

## Speed

### Per-framework totals

| framework | notebooks | total | avg | %>120s |
|---|---:|---:|---:|---:|
| PyTorch | 139 | 116 min | 50 s | 17 (12 %) |
| TensorFlow | 123 | 182 min | 89 s | 31 (25 %) |
| JAX | 123 | 209 min | 102 s | 30 (24 %) |
| MXNet | 128 | 297 min | 139 s | 36 (28 %) |

### Notebooks where one framework is ≥3× slower than peers (incidental, likely fixable)

| notebook | PT | TF | JAX | MX | offender | likely cause |
|---|---:|---:|---:|---:|---|---|
| `chapter_computer-vision/neural-style.md` | 38 s | 59 s | **1710 s** | 57 s | JAX 45× | Training loop has no `@jax.jit` anywhere; 500 epochs run in pure-Python JAX ops. Also calls a TF VGG backbone per step → host round-trip. |
| `chapter_generative-adversarial-networks/gan.md` | 17 s | 137 s | 309 s | **637 s** | MX 38× | MXNet imperative + `update_D`/`update_G` not hybridized. TF has no `@tf.function`. JAX has no `@jax.jit`. |
| `chapter_multilayer-perceptrons/kaggle-house-price.md` | 13 s | 21 s | 29 s | **279 s** | MX 21× | MXNet imperative dispatch on the small MLP. |
| `chapter_linear-classification/softmax-regression-concise.md` | 40 s | 26 s | 29 s | **496 s** | MX 19× | MXNet imperative. |
| `chapter_appendix-mathematics-for-deep-learning/statistics.md` | 9 s | 8 s | **141 s** | 7 s | JAX 19× | Likely lazy JAX op compilation in a non-trainer chapter. |
| `chapter_linear-classification/softmax-regression-scratch.md` | 41 s | 28 s | 31 s | **491 s** | MX 18× | MXNet imperative. |
| `chapter_recurrent-neural-networks/rnn-concise.md` | 82 s | 121 s | 10 s | **145 s** | MX 15× (vs JAX) | MXNet imperative, but timing differences are bounded. |
| `chapter_multilayer-perceptrons/dropout.md` | 61 s | 52 s | 62 s | **635 s** | MX 12× | MXNet imperative. |
| `chapter_optimization/minibatch-sgd.md` | 24 s | 177 s | **252 s** | 67 s | JAX 11× | JAX trainer not JIT-fused for this notebook's variant. |
| `chapter_computer-vision/ssd.md` | 87 s | 449 s | 125 s | **914 s** | MX 11× | MXNet imperative + Python `multibox_target` per step. TF: `train_step` not under `@tf.function`. |
| `chapter_natural-language-processing-pretraining/bert-pretraining.md` | 22 s | 36 s | **187 s** | 62 s | JAX 8× | JAX retracing on variable BERT-input shapes. |
| `chapter_computer-vision/kaggle-dog.md` | 100 s | 213 s | **776 s** | 282 s | JAX 8× | JAX tab calls TF VGG backbone outside the `@jax.jit` boundary; cross-framework data shuttle every step. |
| `chapter_convolutional-neural-networks/lenet.md` | 47 s | 63 s | 57 s | **354 s** | MX 8× | MXNet imperative. |

### Notebooks where one framework is "incidentally" slow (specific code-pattern issue)

These are notebooks where a code-level fix would likely close the gap.

#### `chapter_computer-vision/neural-style.md` (JAX, 1710 s)
- The `train` loop iterates 500 epochs in pure Python JAX with no `@jax.jit` anywhere. Each iteration calls `extract_features`, `compute_loss` (= `content_loss + style_loss + tv_loss`), and an optimizer update as separate dispatched calls.
- The feature extractor is a TF VGG model called via `features_net.predict(...)`, so each step incurs a host round-trip even after JIT'ing the JAX side.
- A single `@jax.jit` step function combining forward + grad + apply would likely cut this to ~100–200 s (matching PT/MX).

#### `chapter_generative-adversarial-networks/dcgan.md` (TF 1026 s, MX 814 s)
- TF: `d2l.update_D` / `d2l.update_G` are plain Python functions — no `@tf.function`. Each per-batch call retraces a fresh `GradientTape` and returns an eager tensor. With 500 epochs × batches × 2 updates this dominates.
- MX: imperative; no `hybridize()` on the generator/discriminator.
- JAX (198 s) and PT (161 s) confirm the upper bound for the structural cost.

#### `chapter_generative-adversarial-networks/gan.md` (TF 137 s, JAX 309 s, MX 637 s)
- Same root cause as `dcgan.md`: missing `@tf.function` (TF), missing `@jax.jit` on `update_D`/`update_G` in the JAX `#@tab jax` block (the JAX tab defines them as plain Python functions), and MXNet imperative.

#### `chapter_recommender-systems/neumf.md` (PT 153 s, MX 1091 s)
- Both PT and MX use `evaluate_ranking`, which loops over ~944 users and creates a fresh `DataLoader` per user. PT eats it; MX's higher per-call overhead amplifies the cost ~7×. Cross-user inference could be batched into a single forward pass.

#### `chapter_computer-vision/ssd.md` (TF 449 s)
- TF training loop calls `net.train_step(...)` (a Keras method override using `tf.GradientTape`) **without** `@tf.function` decoration and not via `model.fit`, so it runs eagerly per batch for 20 epochs. Metric extraction also calls `int(logs['cls_correct'])` per step, forcing host syncs.

#### `chapter_computer-vision/kaggle-dog.md` (JAX 776 s)
- The JAX training loop calls `extract_features(features_net, features.numpy())` inside the per-step body. `features_net` is a TF backbone, so each step (a) leaves JAX, (b) executes TF inference, (c) copies data back to JAX. The `train_step` itself is `@jax.jit` but the feature extractor overhead dominates. Pre-extracting features once (the PT/TF approach) or porting the backbone to JAX would close the gap.

#### `chapter_natural-language-processing-applications/natural-language-inference-attention.md` (JAX 396 s vs PT 133 s)
- `train_step` and `eval_step` are `@jax.jit`-decorated. The slowdown is most plausibly variable-length sequence padding causing JAX recompilation per distinct shape.

#### `chapter_hyperparameter-optimization/sh-intro.md` (TF 1358 s vs PT 513 s, JAX 785 s)
- TF: `hpo_objective` calls `model.fit` repeatedly with a fresh model per HPO trial. Each `model.fit` rebuilds the TF graph from scratch (no graph caching across calls); this is an inherent TF overhead when `fit` is called many times with new model instances.
- JAX: same pattern, smaller penalty (recompiles per-trial model).

#### `chapter_hyperparameter-optimization/hyperopt-api.md` (TF 533 s vs PT 190 s, JAX 266 s)
- Same cause as `sh-intro.md`: repeated `model.fit` calls with fresh Keras models.

### Notebooks slow in **all** frameworks (structural — model size or epoch count, not a code bug)

`chapter_convolutional-modern/{vgg,nin,alexnet,googlenet,densenet,cnn-design,resnet}.md`,
`chapter_optimization/lr-scheduler.md`,
`chapter_hyperparameter-optimization/sh-intro.md`,
`chapter_natural-language-processing-applications/natural-language-inference-bert.md`,
`chapter_computer-vision/{ssd,neural-style}.md`. These are heavy by
design (10–500 epochs of large CNNs / transformers / BERT fine-tuning).

### Notebooks slow only in MXNet (no `hybridize()` is the blanket cause)

The MXNet `d2l.Trainer` path is purely imperative — no `hybridize()`
call anywhere. For medium-to-large models the per-op Python dispatch
cost accumulates to 4–7× PyTorch.

| notebook | MX | min peer | ratio |
|---|---:|---:|---:|
| `chapter_convolutional-modern/nin.md` | 1242 s | 213 s (PT) | 5.8× |
| `chapter_convolutional-modern/vgg.md` | 941 s | 227 s (PT) | 4.1× |
| `chapter_convolutional-modern/cnn-design.md` | 781 s | 159 s (PT) | 4.9× |
| `chapter_convolutional-modern/googlenet.md` | 694 s | 177 s (PT) | 3.9× |
| `chapter_convolutional-modern/alexnet.md` | 677 s | 181 s (TF) | 3.7× |
| `chapter_convolutional-modern/resnet.md` | 637 s | 173 s (TF) | 3.7× |
| `chapter_convolutional-modern/batch-norm.md` | 575 s | 101 s (PT) | 5.7× |
| `chapter_convolutional-modern/densenet.md` | 469 s | 138 s (PT) | 3.4× |
| `chapter_recommender-systems/neumf.md` | 1091 s | 153 s (PT) | 7.1× |
| `chapter_computer-vision/ssd.md` | 914 s | 87 s (PT) | 10.5× |
| `chapter_multilayer-perceptrons/dropout.md` | 635 s | 52 s (TF) | 12.2× |

`chapter_convolutional-modern/batch-norm.md` is amplified further by a
scratch `BatchNorm` implementation in Python (per-step Python forward
pass on top of MXNet's imperative cost).

---

## Pipeline & data

- **Convergence pipeline:**
  - `tools/extract_convergence.py` parses each notebook's last
    matplotlib SVG, takes the longest path per color (since
    `d2l.ProgressBoard` re-emits at every animation step), and
    computes `rel_drop = (first_y_px − last_y_px) / range`. Output:
    `logs/convergence-all.tsv` (369 rows).
  - `tools/compare_convergence.py` cross-framework medians the
    `rel_drop` per `(notebook, color)` and flags `SIGN-FLIP`,
    `FLAT`, or `DRIFT`. Output: `logs/convergence-compare.md`.
- **Metric scraping:** `_notebooks/<fw>/<chapter>/<nb>.ipynb` text/plain
  outputs are regex-matched for `loss`, `acc`, `ppl`, `bleu` printed
  values. Per (notebook, framework) we keep the *last* match.
  Output: `logs/training_metrics.tsv` and `logs/metric_compare.tsv`.
- **Timing:** `logs/make-all.log` is parsed for `OK (Xs)` lines per
  framework section. Output: `logs/timings.tsv` and
  `logs/framework_outliers.tsv`.

The TSVs and the auto-generated `convergence-compare.md` are
regenerated each `make all` cycle; this `convergence-and-speed.md`
needs to be re-curated against them by hand.
