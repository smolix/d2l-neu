# Non-MXNet Anomaly Diagnostics

This note summarizes the non-MXNet convergence and reporting anomalies found
while comparing current notebook outputs on 2026-05-19. It records the known
cause, affected files, fix direction, and before/after metrics captured so far.

## Summary

| Area | Framework | Before | After | Status |
|------|-----------|--------|-------|--------|
| Transformer translation | JAX | BLEU `0`; decoded output collapsed to `<pad>` | BLEU `1.000` | Fixed |
| Single-shot detection | JAX | class error `3.16e-01` | class error `3.56e-03`; bbox MAE `3.58e-03` | Fixed |
| CIFAR-10 | JAX | loss `1.583`; train acc `0.417`; valid acc `0.359` | loss `1.283`; train acc `0.536`; valid acc `0.453` | Improved |
| Image augmentation | TensorFlow | test acc `0.593` | test acc `0.808` | Improved |
| Fine-tuning | TensorFlow | test acc `0.536` fine-tuned; `0.521` scratch | test acc `0.720` fine-tuned; `0.521` scratch | Improved |
| Fine-tuning | JAX | results were not explicitly labelled fine-tuned vs scratch | test acc `0.879` fine-tuned; `0.827` scratch | Fixed reporting; improved recipe |
| Kaggle dog | TensorFlow | train/valid loss `4.532`/`4.865` | train/valid loss `0.320`/`1.382` | Fixed |
| Kaggle dog | JAX | train/valid loss `4.749`/`4.798` | train/valid loss `3.764`/`4.164` | Improved, still needs work |

## JAX Transformer Translation

Affected files:

- `d2l/jax.py`
- `d2l/_blocks/jax/Seq2Seq.py`
- `chapter_recurrent-modern/seq2seq.md`

Cause:

The JAX sequence-to-sequence loss did not mask padded decoder positions
correctly. The loss therefore trained on `<pad>` targets as ordinary labels,
which let the transformer translation model converge to all-`<pad>` decoded
outputs despite the training loop completing successfully. The visible symptom
was BLEU `0` for all reported examples.

Fix:

The JAX Seq2Seq loss now applies the valid-length mask before reduction, so
padded positions do not contribute to the optimization target. The supporting
JAX block and chapter code were updated to use the masked loss path.

Metrics:

- Before: JAX transformer BLEU was `0`; predictions were all `<pad>`.
- After: JAX transformer BLEU is `1.000` for each of the three printed
  example translations. This is a notebook-output smoke check, not a full
  validation-set BLEU evaluation.

## JAX SSD Prediction Layout

Affected file:

- `chapter_computer-vision/ssd.md`

Cause:

The JAX SSD path handled model predictions with the wrong layout for the loss
and metric code. SSD predictions in this notebook should remain NHWC in the JAX
path. Transposing or interpreting them as channel-first corrupted class and
box alignment, producing a class error roughly two orders of magnitude worse
than peer frameworks.

Fix:

The JAX SSD code keeps predictions in NHWC form for the downstream SSD loss and
metrics.

Metrics:

- Before: class error `3.16e-01`.
- After: class error `3.56e-03`; bbox MAE `3.58e-03`.

## JAX CIFAR-10 Learning Rate

Affected file:

- `chapter_computer-vision/kaggle-cifar10.md`

Cause:

The JAX CIFAR-10 training recipe was under-training with the previous learning
rate. The model was learning, but the curve lagged well behind peer-framework
results and produced weak validation accuracy.

Fix:

The JAX learning rate was raised to `5e-4`.

Metrics:

- Before: loss `1.583`; train acc `0.417`; valid acc `0.359`.
- After: loss `1.283`; train acc `0.536`; valid acc `0.453`.

## TensorFlow Image Augmentation

Affected file:

- `chapter_computer-vision/image-augmentation.md`

Cause:

The TensorFlow image-augmentation notebook was using a weaker or mismatched
model path for the CIFAR-10 augmentation experiment, leaving test accuracy well
below the PyTorch and JAX runs.

Fix:

The TensorFlow path now uses `d2l.resnet18`, aligning the model choice with the
chapter recipe and peer-framework behavior.

Metrics:

- Before: test acc `0.593`.
- After: loss `0.183`; train acc `0.937`; test acc `0.808`.

## TensorFlow Fine-Tuning Preprocessing

Affected file:

- `chapter_computer-vision/fine-tuning.md`

Cause:

The TensorFlow fine-tuning path did not apply the Keras ResNet50
`preprocess_input` transformation expected by the pretrained backbone. This
distribution mismatch prevented the fine-tuned model from benefiting
meaningfully from pretrained ImageNet features.

Fix:

The TensorFlow data path now applies Keras ResNet50 `preprocess_input` for the
fine-tuned model path.

Metrics:

- Before: test acc `0.536` fine-tuned; `0.521` scratch.
- After: test acc `0.720` fine-tuned; `0.521` scratch.

## JAX Fine-Tuning Reporting And BatchNorm

Affected file:

- `chapter_computer-vision/fine-tuning.md`

Cause:

The JAX fine-tuning output did not clearly label fine-tuned versus scratch
results, making the comparison ambiguous. The fine-tuned JAX path also needed
to keep pretrained BatchNorm statistics fixed so the small target dataset would
not overwrite useful ImageNet normalization state.

Fix:

The notebook now reports explicitly labelled fine-tuned and scratch results.
The JAX fine-tuned path keeps pretrained BatchNorm statistics fixed and uses a
learning rate of `1e-4`.

Metrics:

- After: test acc `0.879` fine-tuned; `0.827` scratch.

## Kaggle Dog Features And Preprocessing

Affected file:

- `chapter_computer-vision/kaggle-dog.md`

Cause:

The TensorFlow and JAX Kaggle dog classifiers were not using the input
distribution expected by the pretrained Keras ResNet50 backbone. The JAX path
also used pooled features rather than the ImageNet logit feature representation
that mirrors the PyTorch tab's frozen 1000-way ResNet output. This kept both
train and validation losses high.

Fix:

Both TensorFlow and JAX now apply ResNet50 `preprocess_input`. The TensorFlow
and JAX paths use frozen ImageNet logits from the pretrained model before the
custom dog-breed head.

Metrics:

- TensorFlow before: train/valid loss `4.532`/`4.865`.
- TensorFlow after: train/valid loss `0.320`/`1.382` for the validation-split
  run. The later full-data submission run reports train loss `0.370` and has
  no validation loss.
- JAX before: train/valid loss `4.749`/`4.798`.
- JAX after: train/valid loss `3.764`/`4.164` for the validation-split run.
  The later full-data submission run reports train loss `3.681` and has no
  validation loss.

Status:

TensorFlow is now comparable to PyTorch. JAX remains incomplete: the
improvement confirms the preprocessing and feature path are in the right
direction, but validation loss is still high enough that the JAX Kaggle dog
recipe needs further tuning or a deeper comparison against the stronger
PyTorch/TensorFlow paths.

## Follow-Up

- Re-run only the affected non-MXNet notebooks after the current code changes
  are reviewed and the MXNet runtime work is separated from this analysis.
- Treat JAX Kaggle dog as the remaining open non-MXNet convergence item from
  this set.
- Keep future anomaly notes tied to exact notebook outputs or run logs, because
  several affected chapters report metrics only through generated notebook
  outputs rather than source-controlled text.
