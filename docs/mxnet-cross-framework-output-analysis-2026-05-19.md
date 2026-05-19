# MXNet Cross-Framework Output Analysis

Date: 2026-05-19

This report compares current MXNet notebook outputs with the corresponding
PyTorch, TensorFlow, and JAX notebooks. It uses the current `_notebooks` tree,
the clean MXNet run, `tools/audit_notebook_results.py`, curve extraction from
`tools/extract_convergence.py`, and chapter-specific manual inspection.

## Scope

| Framework | Notebooks | Current Stamps | Notebooks With Outputs | Issues |
|-----------|-----------|----------------|------------------------|--------|
| PyTorch | 139 | 134 | 127 | 5 stale/unexecuted current notebooks |
| TensorFlow | 123 | 123 | 117 | none detected by audit |
| JAX | 123 | 123 | 116 | 1 convergence warning |
| MXNet | 128 | 86 | 80 | 42 missing stamps, 1 stamped notebook with error outputs |

MXNet is therefore comparable only for the subset that actually executed. The
largest gaps are exactly the training-heavy GPU notebooks: modern CNNs,
computer vision training, translation, RNN language models, NLP applications,
BERT/Word2Vec pretraining, recommender training, and several multi-GPU demos.

## Conclusion

The MXNet notebooks that produced usable outputs mostly look sensible compared
with the other frameworks. Simple supervised models, linear regression,
softmax/MLP examples, basic optimization demos, deterministic convolution
examples, dataset/preprocessing notebooks, and the toy GAN have outputs in the
same qualitative and numeric range as PyTorch/TensorFlow/JAX.

The main MXNet problem is not bad convergence among completed notebooks. It is
missing coverage: 42 notebooks did not pass, mostly due the MXNet GPU runtime
failures documented in `docs/mxnet-runtime-diagnostics.md`. One additional
artifact-quality problem remains: `chapter_builders-guide/use-gpu.ipynb` has a
passing stamp but contains two stored MXNet GPU error outputs, so it should not
be treated as a clean pass.

## MXNet Outputs That Look Good

- Linear regression scratch/concise converges like the other frameworks.
  MXNet final parameter errors are small:
  `w=[-0.00058794, -0.00038433], b=0.00057793` for scratch and
  `w=[0.00879252, -0.01148272], b=0.01451397` for concise.
- `softmax-regression-concise`, `dropout`, and `mlp-implementation` all have
  loss curves moving in the expected direction and validation accuracy curves
  improving, matching the peer-framework curve shapes.
- `kaggle-house-price` reports average validation log MSE `0.1721`, comparable
  to PyTorch `0.1800`, TensorFlow `0.1770`, and JAX `0.1791`.
- `weight-decay` L2 norms are `0.0112`, `0.00176`, `0.00196`, close to
  TensorFlow/JAX and within expected stochastic variation.
- Optimizer demos that executed (`adadelta`, `adagrad`, `adam`, `momentum`,
  `rmsprop`, `sgd`, `gd`) have trajectories/losses aligned with peers.
- `conv-layer` learns the intended edge detector. MXNet loss decreases
  `4.757 -> 0.799 -> 0.134 -> 0.023 -> 0.004`, and the learned kernel is about
  `[[0.9877, -0.9896]]`, matching PyTorch/JAX behavior.
- Dataset/geometry/example notebooks in preliminaries, appendix, builders
  guide, attention basics, object detection dataset, and segmentation dataset
  have structurally sensible outputs and shapes.
- `gan` reports `loss_D=0.693`, `loss_G=0.694`, matching PyTorch `0.695/0.696`,
  TensorFlow `0.684/0.730`, and JAX `0.693/0.693`. Throughput is lower, but the
  loss values are plausible for this toy GAN.
- NLP/data-only outputs that executed look consistent: BERT tensor shapes,
  attention weights, dataset examples, and similarity/analogy outputs are
  plausible. `similarity-analogy` matches the peers on examples such as
  `chips/intel/electronics`, `babies/boy/girl`, and
  `lovely/gorgeous/wonderful`.

## MXNet Outputs Missing Or Not Comparable

These MXNet notebooks have no passing execution record and therefore cannot be
compared for convergence:

- Attention/translation: `bahdanau-attention`, `transformer`, `seq2seq`.
- Recurrent language models: `deep-rnn`, `gru`, `lstm`, `rnn-concise`,
  `rnn-scratch`.
- NLP applications/pretraining: `natural-language-inference-attention`,
  `natural-language-inference-bert`, `sentiment-analysis-cnn`,
  `sentiment-analysis-rnn`, `bert-pretraining`, `word2vec-pretraining`.
- Computer vision/CNNs: `fcn`, `fine-tuning`, `image-augmentation`,
  `kaggle-cifar10`, `kaggle-dog`, `neural-style`, `ssd`, `lenet`, and all
  modern CNN architecture notebooks (`alexnet`, `batch-norm`, `cnn-design`,
  `densenet`, `googlenet`, `nin`, `resnet`, `vgg`).
- Recommender training: `autorec`, `deepfm`, `fm`, `mf`, `neumf`, `seqrec`.
- Optimization/performance: `lr-scheduler`, `minibatch-sgd`,
  `auto-parallelism`, `multiple-gpus`, `multiple-gpus-concise`.
- GAN: `dcgan`.

For these notebooks the correct comparison result is "not available from
MXNet", not "MXNet converged badly".

## Peer-Framework Anomalies Found While Comparing

These are not MXNet failures, but they were useful cross-framework signals.
They are tracked in more detail in
`docs/non-mxnet-anomaly-diagnostics-2026-05-19.md`.

- JAX `chapter_attention-mechanisms-and-transformers/transformer.ipynb`
  previously emitted repeated `<pad>` predictions and BLEU `0/0/0`; after the
  masked Seq2Seq loss fix it reports BLEU `1.000/1.000/1.000`.
- TensorFlow `image-augmentation` previously had low test accuracy (`0.593`);
  after switching the TensorFlow tab to the same ResNet-18 architecture, it
  reports `loss 0.183`, `train acc 0.937`, `test acc 0.808`.
- JAX `kaggle-cifar10` improved from `loss 1.583`, `train acc 0.417`,
  `valid acc 0.359` to `loss 1.283`, `train acc 0.536`,
  `valid acc 0.453`. This is improved but still comparatively weak.
- In `kaggle-dog`, TensorFlow improved from `train/valid loss 4.532/4.865`
  to `0.320/1.382`. JAX improved from `4.749/4.798` to `3.764/4.164`, but
  remains an open quality gap.
- JAX `ssd` previously had class error `3.16e-01`; after fixing prediction
  layout it reports class error about `3.56e-03`, matching PyTorch/TensorFlow.
- JAX `fine-tuning` now labels the fine-tuned and scratch runs separately and
  reports `test acc 0.879` for fine-tuning versus `0.827` for scratch.

## Method Notes

- The audit report is the execution gatekeeper. Missing/stale/error notebooks
  are excluded from positive convergence claims.
- Text metric extraction is sparse for MXNet because many notebooks report
  only plots. Only `gan` produced a parsed MXNet text metric summary in the
  current audit.
- SVG curve extraction is useful as qualitative triage only. It compares curve
  direction and shape in rendered plot coordinates; it does not recover exact
  data-space values.
- For MXNet, the most important next step is to fix the runtime GPU failures
  and rerun the missing notebooks. Only then can the high-value training
  chapters be compared fairly.
