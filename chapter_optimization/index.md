# Optimization Algorithms
:label:`chap_optimization`

Every network in this book was trained with optimizer settings taken from
a recipe, such as SGD with momentum 0.9 or AdamW with warmup and cosine
decay (:numref:`sec_training_recipes`). When a loss shoots up to `nan`, we
halve the learning rate and try again. Why does a larger step ruin a run,
and why does a smaller one cost so much time? What does each part of the
recipe do, and what must change when the model grows?

Gradient descent is older than computers. Augustin-Louis Cauchy proposed it
in 1847 to solve the equations that determine the orbit of a celestial
body. In 1951 Herbert Robbins and Sutton Monro asked how to find the root of
a function that can only be measured with noise, as a minibatch gradient
is, and answered with steps whose size shrinks over time. Two problems
limit such steps. The curvature of the loss differs between
directions, so a step that is safe along the steepest one barely moves
along the flattest. And the gradient is only an estimate, so every step is
slightly wrong. Momentum, per-coordinate learning rates, schedules and batch
sizes each answer one of these problems. Newer methods also ask how to
measure the size of a step, coordinate by coordinate or as a matrix.

Each answer is one of three decisions an optimizer makes: the *direction*
of a step, which need not be the gradient's, the *step size* and its change
over a run, and the *noise level*, set by batch size and averaging. The
first answers the curvature, the last the noise; the step size must serve
both.

We take the questions one at a time, first on a problem we can solve
exactly, then on small networks where we can only measure. The problem is a
valley that curves twenty times more sharply across than along.
:numref:`sec_test_models` builds the networks: the language model `TinyLM`
and the image classifier `FashionCNN`. `TinyLM` is a transformer, which
:numref:`chap_transformers` explains; to train it you need not know how it
works. On them we choose first the direction, then step size and noise
level, and :numref:`sec_practice` puts the three together.

If you know gradients, eigenvalues and minibatch SGD
(:numref:`subsec_linear-regression-minibatch-sgd`), you know enough. The
chapter proves what is short and measures the rest; longer proofs are in an
appendix (:numref:`chap_mdl-optimization`) you will not need. On a first
reading Muon (:numref:`sec_muon`) and wider models (:numref:`sec_scaling`)
can wait.

```toc
:maxdepth: 2

optimization-intro
gd
sgd
test-models
momentum
adam
adamw
muon
lr-scheduler
batch-size
scaling
practice
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with, and three are marked as the places to start. All are
freely accessible online except where noted. Related topics that the
chapter does not teach, among them second-order methods and further
preconditioners, are collected in
:numref:`sec_optimization-further-reading`. Training on many devices is the
subject of :numref:`chap_performance`, and the optimization chapter of the
mathematical appendix (:numref:`chap_mdl-optimization`) keeps its own list
for the theory, with convex-optimization texts and courses.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — free PDF; the standard reference for the vocabulary this chapter uses informally (conditioning, convergence rates), with the full analysis of gradient descent and Newton's method behind :numref:`sec_gd`, and of steepest descent under a general norm, the idea from which :numref:`sec_muon` starts. It does not treat stochastic gradients.
- [Numerical Optimization — Nocedal & Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5) — paywalled, widely held in university libraries; a comprehensive treatment of Newton's method and preconditioning (:numref:`sec_gd`) and of the methods built on them.

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch — Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — free; the graded version of several exercises of this chapter: implement AdamW as :numref:`sec_adamw` does, account for the memory of the optimizer state (:numref:`subsec_adamw-memory`), and tune the learning rate and batch size of a small transformer language model by sweeps, as the worked configuration of :numref:`sec_practice` does; the accompanying lectures are on YouTube.

**Foundational and current papers**

- [Optimization Methods for Large-Scale Machine Learning — Bottou, Curtis & Nocedal (2018)](https://arxiv.org/abs/1606.04838) — free; the reference for the stochastic side, which Boyd & Vandenberghe leave out: why stochastic gradients suit large data sets, and the proofs that a constant step leaves SGD in a neighborhood of the minimum whose size grows with the step (the noise ball of :numref:`sec_sgd`) and that a decaying step removes it (:numref:`subsec_scheduler-valley`).
- [Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be — Kunstner et al. (2023)](https://arxiv.org/abs/2304.13960) — free; Adam's lead over SGD on transformers persists in full-batch training, where there is no noise to average, and sign descent with momentum shares it. The paper points to Adam's normalized, sign-like step (:numref:`sec_adam`), and not to its handling of noise, as the source of the lead.
- [Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models — Kunstner et al. (2024)](https://arxiv.org/abs/2402.19449) — free; traces the lead to the data: gradient descent makes little progress on the many rare tokens of a language, while Adam makes progress on all of them, even in a linear model (:numref:`sec_adam`).
- [Why Transformers Need Adam: A Hessian Perspective — Zhang et al. (2024)](https://arxiv.org/abs/2402.16788) — free; traces the lead to the architecture: the parameter blocks of a transformer differ far more in curvature than those of a CNN, so no single learning rate suits them all. :numref:`sec_adam` measures the gradient scales of the parameter groups of `TinyLM`, a cruder view of the same heterogeneity.
- [Old Optimizer, New Norm: An Anthology — Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325) — free; the unification that organizes :numref:`sec_muon`: SGD, sign descent and Adam, and Shampoo are each steepest descent under a different norm.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al. (2024)](https://arxiv.org/abs/2410.05192) — free; the river-valley landscape of :numref:`sec_scheduler`, in which the stable phase travels along the river and the decay phase descends its bank; this explains why the loss drops when a warmup–stable–decay schedule begins to decay.
- [An Empirical Model of Large-Batch Training — McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — free; defines the gradient-noise scale and the critical batch size, the two quantities measured in :numref:`sec_batch_size`, and predicts when doubling the batch stops halving the steps.
- [Fantastic Pretraining Optimizers and Where to Find Them — Wen et al. (2025)](https://arxiv.org/abs/2509.02046) — free; re-benchmarks ten optimizers under matched tuning and finds that many reported speedups over AdamW shrink, and that a ranking read before the learning rate has decayed can flip by the end. It is the evidence behind the comparison rules of :numref:`sec_test_models` and the reading of the comparison in :numref:`sec_muon`.

**Tutorials, notes, and interactive**

- [Why Momentum Really Works — Gabriel Goh, Distill (2017)](https://distill.pub/2017/momentum/) — **start here** for the valley, from :numref:`sec_optimization-intro` to :numref:`sec_momentum`; free, interactive; damping and acceleration with sliders for $\eta$ and $\beta$, showing the critical value of $\beta$ and the ringing trajectories of :numref:`sec_momentum`.
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder (2016)](https://www.ruder.io/optimizing-gradient-descent/) — free; a survey of the classical progression from :numref:`sec_sgd` through :numref:`sec_adam`, and a record of common practice in 2016.
- [Muon: An Optimizer for Hidden Layers in Neural Networks — Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/) — free; the original post, with the design decisions, Newton–Schulz coefficients, and ablations behind the optimizer that :numref:`sec_muon` builds from scratch.
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — **start here** for :numref:`sec_muon`; free; a compact derivation of Muon as steepest descent under a norm, the note-form companion to the derivation in that section.
- [The Practitioner's Guide to the Maximal Update Parameterization — EleutherAI](https://blog.eleuther.ai/mutransfer/) — free; muP implemented step by step with the coordinate checks of :numref:`sec_scaling`, including the failures a first implementation runs into.
- [Deep Learning Tuning Playbook — Godbole et al., Google Research](https://github.com/google-research/tuning_playbook) — **start here** for :numref:`sec_practice`; free; a procedure for tuning a new model from a team that ran sweeps at production scale. Its division of hyperparameters into scientific, nuisance, and fixed is the comparison protocol of :numref:`sec_test_models`, and its guide to a new project (choose the optimizer and the batch size, then tune by sweeps) is the long form of the worked configuration that ends :numref:`sec_practice`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9]{.kicker}

How optimizers take their steps<br>
**landscapes · gradient descent · noise · test models · momentum · Adam · AdamW · Muon · schedules · batch size · scaling · practice**
:::
:::

::: {.slide title="The Steepest Direction Caps the Step, the Flattest Sets the Pace"}
[The problem]{.kicker}

The loss is `nan`. We halve the learning rate, and training crawls. A
valley shows both effects.

- $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$: curvature 4 across ($x_2$), 0.2
  along ($x_1$).
- A gradient step with learning rate $\eta$ multiplies $x_2$ by
  $1 - 4\eta$ and $x_1$ by $1 - 0.2\eta$.
- $\eta > 0.5$: $x_2$ diverges. $\eta < 0.5$: $x_1$ shrinks by under a
  tenth per step.
- Best rate $\eta = 0.476$: 47 steps to reach 1% of the start.

::: {.d2l-note}
Two problems limit every step: curvature that differs between directions,
and, on a network, a gradient that is only a minibatch estimate.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- predict from the curvature which learning rates converge, and in how
  many steps;
- predict where stochastic gradient descent (SGD) with a constant rate
  stalls, from the rate, the noise and the batch size;
- implement momentum, Adam, AdamW and Muon from scratch, and say what each
  changes;
- choose a schedule and a batch size, and carry a learning rate over to a
  wider model;
- configure, guard and finish a run that cannot be repeated.

**Running example:** the valley, then `TinyLM`, a small transformer
language model trained on the bytes of Wikipedia.

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
- **Leaves out:** second-order methods, matrix preconditioners other than
  Muon, variance reduction, training on many devices.
- **Place:** opens the Advanced part. `TinyLM`, its data and its training
  loop return in the chapters on attention, transformers, state space
  models and performance.
:::

::: {.slide title="From a Valley We Can Solve to Networks We Can Only Measure"}
[Chapter map]{.kicker}

| movement | sections | measured on |
|---|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch Gradient Descent | the valley; a 25-parameter network; one linear regression |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon | the valley, then `TinyLM` (470,784 parameters) and `FashionCNN` (421,642) |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice | both networks; MLPs of width 128 to 8,192 |

::: {.d2l-note}
On a first pass, Muon and Scaling Up can wait.
:::
:::
