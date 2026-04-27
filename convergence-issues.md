# Cross-Framework Convergence Issues

Cross-framework comparison of training curves across PT/TF/JAX (MX
pending). For each notebook with a `trainer.fit(...)` plot, we extracted
each curve's first/last y-pixel and the relative drop across the curve's
range; we then compared the same color-coded curve (typically `train_loss`
= blue, `val_loss` = orange, `val_acc` = green) across frameworks and
flagged where one framework's `rel_drop` deviates strongly from the
median of its peers.

Source data: `logs/convergence-all.tsv` (one row per notebook ×
framework × curve color).
Pipeline: `tools/extract_convergence.py` → `tools/compare_convergence.py`.

## Findings (PT, TF, JAX — MX still running)

### Real convergence problems

#### 1. `chapter_convolutional-modern/batch-norm.ipynb` — TF BNLeNet (Keras BN)

**TensorFlow's `BNLeNet` (using `tf.keras.layers.BatchNormalization`)
does not train.** val_loss rises across all 10 epochs and val_acc stays
flat near its initial value. The companion `BNLeNetScratch` (from-scratch
BN) in the same notebook converges normally on TF; PT and JAX both train
the Keras-equivalent BN model fine.

| Framework | val_loss rel_drop | val_acc rel_drop |
|-----------|------------------:|-----------------:|
| PyTorch   | -0.98 (decreasing) | +0.98 (rising) |
| JAX       | -1.00 (decreasing) | +1.00 (rising) |
| TensorFlow | **+0.20 (rising)** | **-0.07 (flat)** |

Likely cause per inspection: `lr=0.5` interacts badly with Keras's
default BatchNormalization momentum (0.99) when used inside the custom
Trainer's `GradientTape` step. The from-scratch BN trains fine at the
same LR.

**Suggested fix:** lower the LR for the Keras-BN variant, or set
`momentum=0.9` (PyTorch's default) on the Keras `BatchNormalization`
layers.

#### 2. `chapter_convolutional-modern/densenet.ipynb` — PT DenseNet

**PyTorch DenseNet's val_loss rises across all 10 epochs and val_acc
stays flat.** Training loss falls normally, so the model learns the
training data — this is overfitting on the training set without
generalizing. TF and JAX DenseNet converge normally.

| Framework | val_loss rel_drop | val_acc rel_drop |
|-----------|------------------:|-----------------:|
| PyTorch   | **+0.47 (rising)** | **-0.08 (flat)** |
| JAX       | -0.99 (decreasing) | +0.90 (rising) |
| TensorFlow | -1.00 (decreasing) | +1.00 (rising) |

**Suggested fix:** investigate whether PT DenseNet has a different
weight init, default dropout, or growth-rate vs. the JAX/TF tabs; the
training loss converging while validation diverges suggests a
regularization gap.

### False positives flagged by the heuristic

These tripped the cross-framework `rel_drop` check but on inspection
are not real divergences:

- **`chapter_convolutional-modern/resnet.ipynb` (JAX)**: JAX ResNet
  converges so fast it pegged at near-optimal in epoch 1; the curve is
  flat at the *converged* end. PT/TF show normal convergence over 10
  epochs but JAX is just earlier to the asymptote.
- **`chapter_optimization/sgd.ipynb`**: the blue line in this notebook
  is a `f(x₁, x₂)` contour from `d2l.show_trace_2d()`, not a training
  loss. Matplotlib renders the contour path in different directions on
  PT vs. TF, which the rel_drop heuristic flags as a sign-flip; the
  actual SGD trajectory (orange) is fine on both frameworks.
- **`chapter_recurrent-modern/seq2seq.ipynb`**: both PT and TF val_loss
  changes are < 10% of the plot range — noise-level oscillation, not
  divergence. TF additionally plots a green `acc` curve (TF's
  `EncoderDecoder` inherits from `Classifier._report_val`, which auto-
  plots accuracy even though it's meaningless for sequence tasks);
  that's a TF-library inheritance quirk, not a training failure.

## Method

`tools/extract_convergence.py` parses each notebook's matplotlib SVG
output, extracting one polyline per curve color (the `LONGEST` path per
color, since `d2l.ProgressBoard` re-emits the SVG at every animation
step and shorter intermediate curves are stale). The legend block is
also parsed to map color → label (`val_loss`, `val_acc`, etc.) when
matplotlib emits a legend entry.

`rel_drop = (first_y_pixel - last_y_pixel) / (max_y_pixel - min_y_pixel)`,
which is **negative** for a textbook-converging loss curve (data
decreased → SVG y-pixel increased, since matplotlib inverts the y-axis
in its rendered SVG) and **positive** for a textbook-converging
accuracy curve.

`tools/compare_convergence.py` computes the median `rel_drop` for each
(notebook, color) tuple across frameworks and flags any framework whose
value:
- has the opposite sign (SIGN-FLIP) of its peers, or
- has |rel_drop| < 0.2 while peers have |rel_drop| > 0.6 (FLAT), or
- deviates by > 0.5 from the median (DRIFT).

## Limitations

- Color-based comparison assumes each framework plots curves in the
  same order so colors line up. d2l's Trainer calls
  `self.plot('train_loss', ...)` then `self.plot('val_loss', ...)`,
  which matches this assumption — but custom training loops (e.g.,
  GAN, recommender) don't follow it and were excluded by min-points
  filtering.
- The heuristic flags relative-drop discrepancies, not absolute
  performance. A framework can converge to a *worse* loss/accuracy
  than peers and the heuristic won't catch it (both still descended).
- The flat detection misses cases where a curve descends then rebounds
  (we only look at first vs. last point, not the trajectory shape).
- MXNet results pending the in-progress `make run-all-notebooks` run.

---

*This file is generated; will be regenerated after MX completes.*
