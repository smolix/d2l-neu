# Convergence and Rendered-Output Audit

Date: 2026-05-06

## Remediation Status

Status: remediated in source and rebuilt into `_book` / `_book/slides` on
2026-05-06.

Key fixes applied:

- Fixed binary single-logit accuracy and scalar-loss accounting used by
  PyTorch FM/DeepFM training.
- Initialized PyTorch matrix-factorization embeddings/biases consistently
  with the small-normal MXNet recipe.
- Fixed JAX optimizer full-loss evaluation so `Xs` and `ys` are stacked from
  the same shuffled iterator pass.
- Increased linear-regression scratch training to 10 epochs so all framework
  parameter-error outputs are small and present.
- Replaced brittle translation demo examples and reran sequence notebooks
  across frameworks; rendered pages no longer contain BLEU-0 examples.
- Quieted BERT dataset download output and the TensorFlow Keras `input_shape`
  warning in the HPO example.
- Fixed ResNet slide image reference and slide overlay fallback paths.
- Added explanatory text for tiny CIFAR-10 demo validation variance, GAN loss
  interpretation, and small-workload multi-GPU overhead.

Final rendered checks:

- `_book/chapter_recommender-systems/fm.html`: `loss 0.006, train acc 1.000, test acc 0.924`.
- `_book/chapter_recommender-systems/deepfm.html`: `loss 0.012, train acc 0.996, test acc 0.938`.
- `_book/chapter_recommender-systems/mf.html`: `train loss 0.706, test RMSE 1.048`.
- `_book/chapter_optimization/minibatch-sgd.html`: JAX losses render around `0.244`.
- `_book/chapter_linear-regression/linear-regression-scratch.html`: all framework parameter errors are present; PyTorch `w` error is about `3e-4`.
- `_book/chapter_computer-vision/kaggle-cifar10.html`: demo validation accuracy renders as `0.469` with an explicit tiny-dataset caveat.
- `_book/chapter_computational-performance/multiple-gpus-concise.html`: renders one/two GPU measurements with an explicit small-workload overhead caveat.
- `_book/chapter_generative-adversarial-networks/dcgan.html`: renders the refreshed DCGAN trace with an explicit GAN-loss caveat.

Validation commands completed:

- `python3 -m py_compile tools/gen_notebooks.py tools/inject_outputs.py tools/gen_slides.py d2l/torch.py d2l/tensorflow.py d2l/jax.py d2l/mxnet.py`
- `make lib`
- Targeted notebook executions for affected PyTorch, TensorFlow, JAX, and MXNet notebooks.
- `make -B -j4 slides`: all slide decks rendered with `0 failed`.
- `make -B html`: full 192-page HTML book rendered and `_slides/` integrated into `_book/slides/`.
- Rendered bad-signature scan found no occurrences of the audited stale failures:
  all-zero recommender metrics, `test RMSE 2.564`, JAX `0.7/0.8` optimizer losses,
  `bleu,0.000`, stale Keras warning, missing `resnet18.svg`, `Final-Report.jpg`,
  or the old four-level slide-logo fallback path.

Scope:

- Rendered book HTML under `_book/chapter_*`.
- Compiled slides under `_book/slides/{pytorch,tensorflow,jax,mxnet}` and `_slides`.
- Latest build logs under `logs/`.

Method:

- Searched rendered outputs for failed cells, tracebacks, warnings, progress bars, NaN/inf, missing images, and incomplete result blocks.
- Extracted training metrics from rendered output blocks.
- Compared framework tabs when one backend looked materially worse than siblings.
- Used parallel chapter-group audits for core supervised learning, NLP/sequence models, vision/recommendation/RL/GP/HPO, and mechanical artifact checks.

## Executive Summary

The latest full slide and site renders completed successfully: all slide decks rendered with `0 failed`, and `make html` completed. No rendered book page appears to contain an actual traceback, NaN/inf failure, or missing notebook output caused by a failed cell.

However, several rendered results are not publication quality:

- Recommender-system training is the highest-risk area: FM and DeepFM report all-zero metrics, and MF has very poor RMSE.
- JAX optimization examples often converge much worse than the other frameworks on the same minibatch linear-regression task.
- Several translation examples in seq2seq/attention/Transformer demos produce BLEU 0 outputs.
- Kaggle CIFAR-10 validation accuracy is poor relative to training accuracy.
- Some rendered pages include noisy TensorFlow/Keras warnings or download logs.
- A generated ResNet slide references `resnet18.svg`, but `_book/img` contains `resnet18-90.svg`, so the ResNet-18 figure is missing in compiled slides.

## High Severity

### `_book/chapter_recommender-systems/fm.html`

Issue: Broken training/evaluation metrics.

Evidence:

```text
loss 0.000, train acc 0.000, test acc 0.000
343198.9 examples/sec on [device(type='cuda', index=0)]
```

Why it matters:

The nonzero throughput suggests the training cell ran, but the metrics are nonsensical. This looks like a metric/evaluator wiring bug, not a real converged model.

Recommended action:

Inspect `d2l.train_ch13` compatibility with CTR binary classification outputs and labels. Confirm that the evaluator is appropriate for `BCEWithLogitsLoss` and that batch labels are nonempty and correctly typed.

### `_book/chapter_recommender-systems/deepfm.html`

Issue: Same broken all-zero metric pattern as FM.

Evidence:

```text
loss 0.000, train acc 0.000, test acc 0.000
265973.0 examples/sec on [device(type='cuda', index=0)]
```

Why it matters:

DeepFM uses the same CTR dataset and training path as FM. This likely shares the same metric/evaluator failure.

Recommended action:

Fix FM/DeepFM together. Add a sanity check that positive/negative labels are present and that reported accuracy/AUC is not identically zero.

### `_book/chapter_recommender-systems/mf.html`

Issue: Matrix factorization rating prediction is very poor.

Evidence:

```text
train loss 1.721, test RMSE 2.564
373741.5 examples/sec on [device(type='cuda', index=0)]
```

Why it matters:

For MovieLens ratings, RMSE `2.564` is far too high and conflicts with the later AutoRec result (`test RMSE 0.909`). This makes the baseline comparison misleading.

Recommended action:

Check rating normalization, loss scaling, model initialization, and evaluator definition. Compare with MXNet output and earlier known-good D2L results.

## Medium Severity

### `_book/chapter_optimization/minibatch-sgd.html`

Issue: JAX final losses are much worse than other frameworks.

Evidence:

JAX outputs include:

```text
loss: 0.760
loss: 0.771
loss: 0.765
loss: 0.714
```

Neighboring PyTorch/TensorFlow/MXNet outputs are approximately `0.242-0.254`.

Recommended action:

Inspect the JAX training loop and loss normalization. This pattern repeats across the optimizer chapter and likely has a shared JAX implementation issue.

### `_book/chapter_optimization/adagrad.html`, `_book/chapter_optimization/adadelta.html`, `_book/chapter_optimization/adam.html`, `_book/chapter_optimization/rmsprop.html`, `_book/chapter_optimization/momentum.html`

Issue: Same JAX optimizer convergence problem.

Evidence:

- Adagrad JAX: `loss: 0.784`
- Adadelta JAX: `loss: 0.817`
- Adam JAX: `loss: 0.769`
- RMSProp JAX: `loss: 0.869`
- Momentum JAX: `loss: 0.744`, later `0.649`, `0.580`

Other frameworks are near `0.24` in the same demos.

Recommended action:

Audit the shared JAX linear-regression/minibatch training helper used in these sections. Check parameter updates, optimizer state, gradient averaging, and whether JIT/static argument behavior is freezing state.

### `_book/chapter_linear-regression/linear-regression-scratch.html`

Issue: Final parameter-error output appears only for PyTorch; other framework tabs show the print code but no rendered output.

Evidence:

PyTorch output:

```text
error in estimating w: tensor([ 0.0776, -0.1552])
b: tensor([0.2090])
```

TensorFlow, JAX, and MXNet tabs do not show the corresponding rendered output.

Why it matters:

This is an incomplete rendered result in a foundational notebook. The PyTorch error is also larger than expected for the synthetic regression demo.

Recommended action:

Rerun this notebook for all frameworks and verify the output injection labels. Check whether the source cell label is shared correctly across framework tabs.

### `_book/chapter_computer-vision/kaggle-cifar10.html`

Issue: Weak validation convergence/generalization.

Evidence:

```text
train loss 0.613, train acc 0.781, valid acc 0.391
2199.9 examples/sec on [device(type='cuda', index=0)]
```

Why it matters:

Validation accuracy is poor and far below training accuracy. For a competition-style training notebook, this looks bad even if the text says the run is for demonstration.

Recommended action:

Check train/valid split, augmentation consistency, evaluation mode, and whether validation labels/classes are mapped correctly. Consider increasing epochs or using a known-good smaller recipe.

### `_book/chapter_generative-adversarial-networks/dcgan.html`

Issue: Final GAN losses look highly imbalanced.

Evidence:

```text
loss_D 0.030, loss_G 6.885, 6447.8 examples/sec on cuda:0
```

Why it matters:

The discriminator appears to dominate the generator. GAN losses are not definitive by themselves, but this is a poor teaching signal unless the generated image grid is visibly good.

Recommended action:

Inspect the generated image output. If quality is poor, lower discriminator learning pressure, adjust update balance, or use a curated stable checkpoint/output for the demo.

### `_book/slides/jax/chapter_recurrent-modern/seq2seq.html`

Issue: Slide-only JAX translation demo mostly fails.

Evidence:

```text
go . => ['<unk>', '!'], bleu,0.000
i lost . => ['je', 'refuse', '.'], bleu,0.000
he's calm . => ['elles', 'ont', 'gagné', '.'], bleu,0.000
```

Only `i'm home` reaches BLEU `1.000`.

Recommended action:

Rerun JAX seq2seq training or use more stable demo examples. The book output is better than the slide output, so the slide artifact may be stale or framework-specific.

### `_book/slides/mxnet/chapter_recurrent-modern/seq2seq.html`

Issue: Slide-only MXNet translation demo has multiple bad outputs.

Evidence:

```text
go . => ['va', 'maintenant', '.'], bleu,0.000
he's calm . => ['il', 'court', '.'], bleu,0.000
i'm home . => ['je', 'suis', 'gras', '.'], bleu,0.512
```

Recommended action:

Same as JAX seq2seq: rerun or select stable examples.

### `_book/chapter_computational-performance/multiple-gpus-concise.html`

Issue: Two-GPU result is slower and less accurate than one-GPU result.

Evidence:

```text
test acc: 0.92, 5.8 sec/epoch on [device(type='cuda', index=0)]
test acc: 0.87, 8.7 sec/epoch on [device(type='cuda', index=0), device(type='cuda', index=1)]
```

Why it matters:

This is a performance chapter. A multi-GPU demo that is both slower and less accurate undermines the stated lesson unless explicitly explained.

Recommended action:

Verify that the batch size, learning rate scaling, data loading, and device placement are correct. If hardware overhead dominates, explain this in the text or choose a workload where multi-GPU helps.

## Low Severity

### `_book/chapter_recurrent-modern/seq2seq.html`

Issue: Main book demo has one failed translation.

Evidence:

```text
he's calm . => ['tu', 'cours', '.'], bleu,0.000
```

Other three examples render with BLEU `1.000`.

Recommended action:

Use a stable example set or train longer. This is less severe than the slide-only failures.

### `_book/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html`

Issue: One bad translation demo.

Evidence:

```text
he's calm . => ['<unk>', 'maintenant', '.'], bleu,0.000
```

Recommended action:

Rerun or adjust the example set.

### `_book/chapter_attention-mechanisms-and-transformers/transformer.html`

Issue: One bad translation demo.

Evidence:

```text
he's calm . => ['<unk>', '.'], bleu,0.000
```

Recommended action:

Rerun or adjust the example set.

### `_book/chapter_natural-language-processing-pretraining/bert-pretraining.html`

Issue: Very short BERT pretraining run looks poorly converged.

Evidence:

```text
MLM loss 7.293, NSP loss 0.809
```

Why it matters:

This may be intentional for runtime, but NSP loss is worse than the random binary cross-entropy baseline of about `0.693`.

Recommended action:

Either train enough steps for the metrics to visibly improve, or explicitly label the run as a mechanics-only smoke test.

### `_book/chapter_recurrent-neural-networks/rnn-scratch.html` and `_book/chapter_recurrent-neural-networks/rnn-concise.html`

Issue: Final samples are weak/repetitive.

Evidence:

Examples include:

```text
it has of the the the the
it has and the that the ti
```

Recommended action:

Render perplexity numerically in stdout and consider a slightly longer run or more stable sample prefix.

### `_book/chapter_hyperparameter-optimization/hyperopt-api.html`

Issue: Keras warning is rendered into the page.

Evidence:

```text
UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer.
```

Recommended action:

Suppress this warning during rendering or update the Keras model to use an explicit `Input` layer.

### `_book/chapter_hyperparameter-optimization/sh-intro.html`

Issue: Same rendered Keras warning.

Evidence:

```text
UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer.
```

Recommended action:

Same as `hyperopt-api.html`.

### `_book/chapter_natural-language-processing-pretraining/bert-dataset.html` and `_book/chapter_natural-language-processing-pretraining/bert-pretraining.html`

Issue: Download log text appears in rendered output.

Evidence:

```text
Downloading ../data/train-00000-of-00001.parquet from https://huggingface.co/...
```

Recommended action:

Quiet the dataset download helper during rendered book builds.

### `_book/chapter_optimization/gd.html`

Issue: PyTorch Newton output is split awkwardly.

Evidence:

One output block contains:

```text
epoch 10, x:
```

and a separate output contains:

```text
tensor(0.)
```

Recommended action:

Convert tensor to scalar before formatting so the output matches the other frameworks.

### `_book/slides/*/chapter_convolutional-modern/resnet.html`

Issue: Missing ResNet-18 slide figure.

Evidence:

Compiled slides reference:

```text
../../../img/resnet18.svg
```

but `_book/img` contains `resnet18-90.svg`, not `resnet18.svg`.

Affected generated slides:

- `_book/slides/pytorch/chapter_convolutional-modern/resnet.html`
- `_book/slides/tensorflow/chapter_convolutional-modern/resnet.html`
- `_book/slides/jax/chapter_convolutional-modern/resnet.html`
- `_book/slides/mxnet/chapter_convolutional-modern/resnet.html`

Recommended action:

Either copy `img/resnet18.svg` into `_book/img`, or update the source slide to use `../img/resnet18-90.svg`.

### `_book/index.html`

Issue: One landing-page university logo reference is missing.

Evidence:

```text
static/landing/universities/Final-Report.jpg
```

The source tree does not contain `static/landing/universities/Final-Report.jpg`.

Recommended action:

Remove the bad entry or replace it with a real institution/logo. This is not a convergence issue, but it is a rendered-site defect found by the artifact scan.

## Notes on Intentional Non-Convergence

Some optimizer pages deliberately show bad behavior:

- `chapter_optimization/gd.html` includes overshooting/divergence examples.
- `chapter_optimization/momentum.html` includes an unstable momentum setting with `x2: -1673.365109`.
- `chapter_optimization/adagrad.html` includes slow convergence on an ill-conditioned example.

These are not necessarily defects if the surrounding text explains the point. The flagged JAX optimizer losses are different: they occur in the shared minibatch training examples where sibling frameworks converge to about `0.24`.

## Areas Checked With No Major Issues

No major convergence/render issues were found in the rendered outputs for:

- Gaussian process chapters: GP inference loss decreases plausibly.
- Reinforcement learning chapters: no rendered tracebacks or missing result figures found.
- Fine-tuning and FCN vision demos: final accuracies are plausible.
- NLI attention and BERT fine-tuning: final accuracies are plausible, aside from the short-pretraining caveat above.
- Word2vec pretraining: rendered loss and throughput are plausible.
- AutoRec and SeqRec: metrics are plausible relative to their task.

## Build/Artifact Health

Latest render logs show:

- PyTorch slides: `Rendered 140 / 140 (0 failed)`.
- MXNet slides: `Rendered 136 / 136 (0 failed)`.
- JAX slides: `Rendered 131 / 131 (0 failed)`.
- TensorFlow slides: `Rendered 131 / 131 (0 failed)`.
- Full HTML build completed and integrated `_slides/` into `_book/slides/`.

Known non-fatal build warnings:

- Placeholder warnings for framework-specific cells without variants.
- `Citeproc: citation Zhang.Lipton.Li.ea.2021 not found`.
- Many skipped cells without `#| label` during output injection; these are expected for unlabeled cells, but the count is high and can hide missing-output issues unless monitored.
