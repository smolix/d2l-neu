# Optimization Algorithms
:label:`chap_optimization`

Every network in this book so far was trained with optimizer settings
taken from a recipe, such as SGD with momentum 0.9 or AdamW with warmup and
cosine decay (:numref:`sec_training_recipes`). When a loss shoots up to
`nan`, we halve the learning rate and try again. Why does a larger step
ruin a run, and why does a smaller one cost so much time? What does each
part of the recipe do, and what must change when the model grows?

Gradient descent is older than computers. Augustin-Louis Cauchy proposed it
in 1847 to compute the orbit of a celestial body. In 1951 Herbert Robbins
and Sutton Monro asked how to find the root of a function that can only be
measured with noise, as a minibatch gradient is, and answered with steps
whose size shrinks over time.

Two problems limit a gradient step, and Cauchy's exact gradients already
had the first. The curvature of the loss differs between directions, so a
step that is safe along the sharpest one barely moves along the flattest.
And the gradient of a network is only an estimate, so every step is
slightly wrong.

What can an optimizer do about them? A step has a direction and a length,
and it is computed from a sample. That makes three decisions: the
*direction* of a step, its size over the run, and the noise it carries.
The first answers the curvature, the last the noise; the step size must
serve both. The direction gets the most room, because the gradient, it
turns out, is rarely the best one: in a narrow valley it points mostly
across, hardly along.

We meet both problems first on an example that we can solve by hand, a
valley that curves twenty times more sharply across than along. After that
we can only measure, on two small networks (:numref:`sec_test_models`): the
image classifier `FashionCNN` and the language model `TinyLM`, a
transformer (:numref:`chap_transformers`) that to an optimizer is only
parameters and their gradients. On both we take the recipe apart: momentum
and Adam for the direction, warmup and decay for the step size, the batch
size for the noise. That is the core of the chapter, and
:numref:`sec_practice` assembles it into a run that we get to do only once.

If you know gradients, eigenvalues and minibatch SGD
(:numref:`subsec_linear-regression-minibatch-sgd`), you know enough. The
chapter proves what is short and measures the rest; longer proofs are in
the mathematical appendix (:numref:`chap_mdl-optimization`), which it does
not require. It stays on one device and leaves the classical methods that
deep learning rarely uses, such as quasi-Newton methods and line search,
to its last section. On a first reading Muon, a recent optimizer for
weight matrices (:numref:`sec_muon`), and wider models
(:numref:`sec_scaling`) can wait.

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

Start with Goh's article for the valley, Bernstein's note for Muon and the
Tuning Playbook for practice. Each entry names the section it goes with,
and all are free online except where noted. Topics that the chapter does
not teach, from quasi-Newton methods to benchmarks of optimizers, are
collected in :numref:`sec_optimization-further-reading`. Training on many
devices is the subject of :numref:`chap_performance`, and texts on the
theory are listed with the appendix (:numref:`chap_mdl-optimization`).

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — the standard reference for the vocabulary that this chapter uses informally (conditioning, convergence rates). It analyzes gradient descent and Newton's method in full (:numref:`sec_gd`), and steepest descent under a general norm, the idea from which :numref:`sec_muon` starts. It does not treat stochastic gradients.
- [Numerical Optimization — Nocedal & Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5) — paywalled, widely held in university libraries; a comprehensive treatment of Newton's method and preconditioning (:numref:`sec_gd`) and of the methods built on them.

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch — Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — the graded version of several exercises of this chapter: implement AdamW (:numref:`sec_adamw`), account for the memory of the optimizer state (:numref:`subsec_adamw-memory`), and find the learning rate and batch size of a small transformer language model by experiment, the task of the worked configuration in :numref:`sec_practice`. The lectures are on YouTube.

**Foundational and current papers**

- [Optimization Methods for Large-Scale Machine Learning — Bottou, Curtis & Nocedal (2018)](https://arxiv.org/abs/1606.04838) — the reference for the stochastic side, which Boyd & Vandenberghe leave out. It proves that a constant step leaves SGD in a neighborhood of the minimum whose size grows with the step (the noise ball of :numref:`sec_sgd`), and that a decaying step removes it (:numref:`subsec_scheduler-valley`).
- [Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be — Kunstner et al. (2023)](https://arxiv.org/abs/2304.13960) — Adam's lead over SGD on transformers persists in full-batch training, where there is no noise to average, and sign descent with momentum shares it. The paper points to Adam's normalized, sign-like step, and not to its handling of noise, as the source of the lead. The explanation in :numref:`sec_adam` starts here.
- [Why Transformers Need Adam: A Hessian Perspective — Zhang et al. (2024)](https://arxiv.org/abs/2402.16788) — why the lead appears on transformers and not on CNNs: the parameter blocks of a transformer differ far more in curvature, so no single learning rate suits them all. :numref:`sec_adam` measures the gradient scales of the parameter groups of `TinyLM`, a cruder view of the same heterogeneity.
- [Old Optimizer, New Norm: An Anthology — Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325) — the unification that organizes :numref:`sec_muon`: with their moving averages switched off, Adam and Shampoo are each steepest descent under a norm of their own, as gradient descent is under the Euclidean norm.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al. (2024)](https://arxiv.org/abs/2410.05192) — the river-valley landscape of :numref:`sec_scheduler`, in which the constant phase travels along the river while it bounces between the banks, and the decay settles the iterate onto the floor. This explains why the loss drops when a warmup–stable–decay schedule begins to decay.
- [An Empirical Model of Large-Batch Training — McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — defines the gradient-noise scale and predicts from it the largest useful batch size. :numref:`sec_batch_size` measures both, the second as the critical batch size, where doubling the batch stops halving the steps.
- [Fantastic Pretraining Optimizers and Where to Find Them — Wen et al. (2025)](https://arxiv.org/abs/2509.02046) — re-benchmarks ten optimizers under matched tuning. Many reported speedups over AdamW shrink, as :numref:`sec_muon` reports beside its own comparison, and a ranking read before the learning rate has decayed can flip by the end, which is why :numref:`sec_scheduler` says when to read one.

**Tutorials, notes, and interactive**

- [Why Momentum Really Works — Gabriel Goh, Distill (2017)](https://distill.pub/2017/momentum/) — **start here** for the valley, from :numref:`sec_optimization-intro` to :numref:`sec_momentum`; interactive; damping and acceleration with sliders for $\eta$ and $\beta$, showing the critical value of $\beta$ and the ringing trajectories of :numref:`sec_momentum`.
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder (2016)](https://www.ruder.io/optimizing-gradient-descent/) — a survey of the classical progression from :numref:`sec_sgd` through :numref:`sec_adam`, and a record of common practice in 2016.
- [Muon: An Optimizer for Hidden Layers in Neural Networks — Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/) — the original post, with the design decisions, Newton–Schulz coefficients, and ablations behind the optimizer that :numref:`sec_muon` builds from scratch.
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — **start here** for :numref:`sec_muon`; a compact derivation of Muon as steepest descent under a norm, the note-form companion to the derivation in that section.
- [The Practitioner's Guide to the Maximal Update Parameterization — EleutherAI](https://blog.eleuther.ai/mutransfer/) — the maximal update parametrization of :numref:`sec_scaling` implemented step by step, with its coordinate checks and the failures that a first implementation runs into.
- [Deep Learning Tuning Playbook — Godbole et al., Google Research](https://github.com/google-research/tuning_playbook) — **start here** for :numref:`sec_practice`; a procedure for tuning a new model, from a team that ran sweeps at production scale. Its three classes of hyperparameters (scientific, nuisance, fixed) are the comparison protocol of :numref:`sec_test_models`, and its guide to a new project is the long form of the worked configuration that ends :numref:`sec_practice`.

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
Momentum, Adam, warmup, decay and the batch size are answers to these two
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
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch Gradient Descent | the valley; a 25-parameter network; linear regression on 1,500 airfoil measurements |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon | the valley, then `TinyLM` (470,784 parameters) and `FashionCNN` (421,642) |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice | the valley; both networks; MLPs of width 128 to 8,192 |

::: {.d2l-note}
The core: Momentum, Adam, AdamW, Schedules and Batch Size. Muon and
Scaling Up can be left for a second pass. Every later deck shows this map
with its own place marked.
:::
:::
