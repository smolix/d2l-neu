# Optimization Algorithms
:label:`chap_optimization`

Every deep network in the preceding chapters was trained with a standard recipe—such as minibatch SGD with momentum $0.9$, or AdamW with linear warmup and cosine decay (:numref:`sec_training_recipes`). When a training run diverges to `nan`, we lower the learning rate; when progress stalls, we adjust the schedule or batch size. This chapter opens the black box of the optimizer to answer three practical questions: what determines the largest safe learning rate and the speed of convergence below it, how does each component of a modern training recipe reshape the update step, and how must those hyperparameters scale as models and batch sizes grow?

Gradient descent itself predates modern computing: Augustin-Louis Cauchy proposed it in 1847 to compute planetary orbits, and in 1951 Herbert Robbins and Sutton Monro analyzed root-finding under noisy observations with decaying step sizes. Across both classical and modern settings, two fundamental obstacles limit any gradient-based optimizer:

1. **Ill-conditioning (uneven curvature):** When the loss surface curves sharply in some directions and gently in others, a learning rate small enough to remain stable along the steepest direction makes painfully slow progress along the flattest directions.
2. **Gradient noise:** Because gradients are estimated from finite minibatches rather than the full dataset, every update carries sampling variance that prevents constant-step-size iterates from settling at a minimum.

Every design choice in a modern optimizer targets one of these two obstacles through three levers:

* **Step direction:** Momentum (:numref:`sec_momentum`), adaptive coordinate-wise methods such as Adam and AdamW (:numref:`sec_adam`), and matrix-level orthogonalized methods such as Muon (:numref:`sec_muon`) replace the raw gradient with a preconditioned or smoothed direction that travels faster along ill-conditioned valleys.
* **Step-size schedule:** Learning-rate warmup, cosine or linear decay, and warmup–stable–decay (WSD) schedules (:numref:`sec_scheduler`) navigate the trade-off between rapid early traversal and low final noise.
* **Minibatch size and scaling:** The critical batch size (:numref:`sec_batch_size`) determines when larger minibatches stop reducing the number of optimization steps, while Maximal Update Parametrization ($\mu\text{P}$, :numref:`sec_scaling`) allows learning rates tuned on a small proxy model to transfer directly to much wider networks.

We develop each method by pairing an analytical model—a two-dimensional quadratic valley with condition number $\kappa = 20$—with systematic empirical measurements on two neural network benchmarks introduced in :numref:`sec_test_models`: the convolutional classifier `FashionCNN` and the byte-level Transformer language model `TinyLM`. Finally, :numref:`sec_practice` combines gradient clipping, weight averaging, and hyperparameter scaling into an end-to-end protocol for configuring and diagnosing large-scale training runs.

This chapter assumes familiarity with gradients, eigenvalues of symmetric matrices, and basic minibatch SGD (:numref:`subsec_linear-regression-minibatch-sgd`). We give concise derivations where they directly guide hyperparameter choices and defer longer mathematical proofs to :numref:`chap_mdl-optimization`. Annotated references to textbooks, papers, and practitioner playbooks are collected in :numref:`sec_optimization-further-reading`.

```toc
:maxdepth: 2

gd
sgd
test-models
momentum
adam
muon
lr-scheduler
batch-size
scaling
practice
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9]{.kicker}

How optimizers take their steps<br>
**two problems · the direction of a step · step size and noise over a run**
:::
:::

::: {.slide title="Curvature Limits How Fast a Run Can Go, Noise How Close It Can Get"}
[The problem]{.kicker}

The loss is `nan`. We halve the learning rate $\eta$, and training crawls.
A valley shows why: $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$, with second
derivatives 4 across ($x_2$) and 0.2 along ($x_1$).

| problem | on the valley | number |
|---|---|---|
| **Uneven curvature** | the sharpest direction caps the step, the flattest sets the pace | a step multiplies $x_2$ by $1 - 4\eta$, so $\eta > 0.5$ diverges; the best rate, $\eta = 0.476$, needs 47 steps to bring both coordinates within 1% of their start |
| **Noisy gradients** | at a constant rate, stochastic gradient descent (SGD) stalls in a cloud around the minimum | half the rate halves the variance of the cloud and doubles the steps to reach it |

::: {.d2l-note}
Momentum, AdamW, warmup, decay and the batch size are answers to these two
problems.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- predict from the curvature which learning rates converge, and in how
  many steps;
- predict where SGD with a constant rate stalls, from the rate, the noise
  and the batch size;
- implement momentum, Adam, AdamW and Muon from scratch, and say what each
  changes;
- choose a schedule and a batch size, and carry a learning rate over to a
  wider model;
- configure a run that cannot be repeated, clip its gradients and average
  its weights.

**Running example:** the valley, then `TinyLM`, a small transformer
language model trained on the bytes of WikiText-103, a selection of
Wikipedia articles.

::: {.d2l-note}
Two problems, three decisions: the direction of a step, its size over the
run, and the noise it carries.
:::
:::

::: {.slide title="The Chapter Stays on One Device and Proves Only What Is Short"}
[Scope and place]{.kicker}

- **Assumes:** gradients, eigenvalues of a symmetric matrix, minibatch SGD.
- **Method:** predict a number, then run. Short proofs are given; longer
  ones are in the mathematics appendix, which is optional.
- **Leaves out:** quasi-Newton methods, line search, variance reduction
  (classical methods that deep learning rarely uses); training on many
  devices.
- **Place:** opens the Advanced part. `TinyLM`, its data and its training
  loop return in the chapters on attention, transformers, state space
  models and performance.
:::

::: {.slide title="From a Valley We Can Solve to Networks We Can Only Measure"}
[Chapter map]{.kicker}

| part | decks | measured on |
|---|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes and Gradient Descent · Stochastic and Minibatch Gradient Descent | the valley; a 25-parameter network; linear regression on 1,500 airfoil measurements |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam and AdamW · Muon | the valley, then `TinyLM` (470,784 parameters) and `FashionCNN` (421,642) |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice | the valley; both networks; MLPs of width 128 to 8,192 |

::: {.d2l-note}
The core: Momentum, Adam and AdamW, Schedules and Batch Size. Muon and
Scaling Up can be left for a second pass.
:::
:::
