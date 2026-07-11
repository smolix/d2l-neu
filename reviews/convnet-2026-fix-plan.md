# Chapters 7--8 correction and experiment plan (approved 2026-07-11)

## Constraints

- Keep MXNet, with an explicit reduced implementation where necessary.
- PyTorch and JAX must implement the complete algorithms and experiments.
- Aim for TensorFlow parity; document any concrete limitation rather than hiding it.
- Chapter 8 may remain long. Reorganize only where a dependency or pacing problem
  warrants it.
- Present the implemented LeNet as a teaching variant and state how it differs from
  LeNet-5 rather than reproducing every historical detail.
- Strengthen comparative experiments with matched controls, multiple seeds, and
  uncertainty. A completed run is not evidence of convergence by itself.

## Ordered work

1. Correct Chapter 7's equivariance, parameter-count, padding, pooling, LeNet,
   and convolution-implementation claims, including slides and exercises.
2. Correct normalization mathematics and implementations, residual-function-class
   reasoning, Conv--BN folding, and benchmark interpretation.
3. Move efficient convnets before training recipes and ConvNeXt so that the inverted
   bottleneck is introduced before use.
4. Provide full PyTorch/JAX implementations of stochastic depth, EMA, recipe
   training, ConvNeXt training, and comparative experiments. Add TensorFlow versions
   where the existing trainer supports them; retain honest MXNet reductions.
5. Replace single-run architectural conclusions by seeded comparisons with reported
   mean and spread. Match parameters or compute in the ConvNeXt comparison.
6. Execute and capture every affected notebook in the supported frameworks, inspect
   learning curves and final metrics, then run source, freshness, HTML, PDF, and slide
   checks.

## Out of scope

- Selecting one framework for the book as a whole.
- Reworking Chapter 6.
- Teaching transformer mechanics in these chapters.
- Reproducing historical LeNet-5 quirks in executable code.
