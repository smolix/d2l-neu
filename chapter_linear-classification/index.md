# Linear Neural Networks for Classification
:label:`chap_classification`

Now that you have worked through all of the mechanics
you are ready to apply the skills you have learned to broader kinds of tasks.
Even as we pivot towards classification,
most of the plumbing remains the same:
loading the data, passing it through the model,
generating output, calculating the loss,
taking gradients with respect to weights,
and updating the model.
However, the precise form of the targets,
the parametrization of the output layer,
and the choice of loss function will adapt
to suit the *classification* setting.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```

## Resources and Further Reading {.unnumbered}

The classical machine-learning texts listed in the previous chapter (**Linear Neural Networks for Regression**) cover classification just as thoroughly, so we do not repeat them here; the references below focus on what is specific to the classification setting—softmax regression and cross-entropy, calibration, and the distribution shift introduced at the end of this chapter. All are freely accessible online except where noted.

**Tutorials and notes**

- [CS229 Machine Learning lecture notes — Andrew Ng & Tengyu Ma (Stanford)](https://cs229.stanford.edu/main_notes.pdf) — free PDF; derives logistic and softmax (multinomial) regression and the cross-entropy loss from the exponential-family / GLM viewpoint.
- [Information Theory — *this book's Mathematics for Deep Learning appendix*](../chapter_mdl-information-theory/index.html) — entropy, cross-entropy, and KL divergence developed in full, the information-theoretic background behind the softmax cross-entropy loss.

**Foundational papers**

- [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms — Xiao, Rasul & Vollgraf (2017)](https://arxiv.org/abs/1708.07747) — the dataset used throughout this chapter's image-classification examples.
- [On Calibration of Modern Neural Networks — Guo, Pleiss, Sun & Weinberger (2017)](https://arxiv.org/abs/1706.04599) — shows that a classifier's softmax confidences are often miscalibrated, and that temperature scaling fixes it; essential context for reading softmax outputs as probabilities.

**Books**

- [Dataset Shift in Machine Learning — Quiñonero-Candela, Sugiyama, Schwaighofer & Lawrence (eds.), MIT Press](https://mitpress.mit.edu/9780262545877/dataset-shift-in-machine-learning/) — the standard reference on the distribution-shift problems (covariate, label, and concept shift) discussed in the final section (an open-access edition is available from MIT Press).
- [Machine Learning in Non-Stationary Environments: Introduction to Covariate Shift Adaptation — Sugiyama & Kawanabe, MIT Press](https://mitpress.mit.edu/9780262017091/machine-learning-in-non-stationary-environments/) — a focused, in-depth treatment of covariate shift and importance-weighted adaptation (print edition; not open access).

