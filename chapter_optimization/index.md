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
measure the size of a step. We take the questions one at a time, on
problems we can solve exactly.

:numref:`sec_optimization-intro` asks what makes a loss hard to minimize.
:numref:`sec_gd` asks how large a gradient step may be, and how rescaling
the gradient removes the limit. :numref:`sec_sgd` asks what changes when the
gradient is estimated from a few examples. :numref:`sec_momentum` asks how
averaging gradients over steps speeds up descent along a narrow valley.
:numref:`sec_adam` asks what a separate learning rate for each coordinate
achieves, and why weight decay must then bypass it. :numref:`sec_scheduler`
asks how the learning rate should change over a run. :numref:`sec_muon` asks
what step a weight matrix should take when the size of a step is measured as
a matrix. :numref:`sec_batch_size` asks how large a batch can grow before
more examples per step stop saving steps. :numref:`sec_scaling` asks how
learning rates carry over from a narrow model to a wide one.
:numref:`sec_practice` asks how to configure a long run, guard it against
failure, and compare optimizers fairly.

The mathematical appendix (:numref:`chap_mdl-optimization`) proves the rates
that this chapter measures, and :numref:`chap_performance` spreads the same
updates across many devices. Methods the chapter does not teach are listed
in :numref:`sec_optimization-further-reading`.

```toc
:maxdepth: 2

optimization-intro
gd
sgd
momentum
adam
lr-scheduler
muon
batch-size
scaling
practice
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online except where noted.
Related topics that the chapter does not teach are collected in
:numref:`sec_optimization-further-reading`, and the optimization chapter of
the mathematical appendix (:numref:`chap_mdl-optimization`) keeps its own
list for the theory, with convex-optimization texts and courses.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — free PDF; the standard reference for the vocabulary this chapter uses informally (conditioning, convergence rates, projections), and the place where the analyses that :numref:`sec_gd` and :numref:`sec_sgd` state without proof are carried out in full.
- [Numerical Optimization — Nocedal & Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5) — paywalled, widely held in university libraries; a comprehensive treatment of Newton's method and preconditioning (:numref:`sec_gd`) and of the methods built on them.

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch — Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — free; the graded version of several exercises of this chapter: implement AdamW as :numref:`sec_adamw` does, account for optimizer-state memory byte by byte, and run the learning-rate and batch-size sweeps that :numref:`sec_practice` turns into a method; the accompanying lectures are on YouTube.

**Foundational and current papers**

- [Old Optimizer, New Norm: An Anthology — Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325) — free; the unification that organizes :numref:`sec_muon`: SGD, sign descent and Adam, and Shampoo are each steepest descent under a different norm.
- [An Empirical Model of Large-Batch Training — McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — free; defines the gradient-noise scale and the critical batch size, the two quantities measured in :numref:`sec_batch_size`, and predicts when doubling the batch stops halving the steps.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al. (2024)](https://arxiv.org/abs/2410.05192) — free; the river-valley landscape of :numref:`sec_scheduler`, in which the stable phase travels along the river and the decay phase descends its bank; this explains why the loss drops when a warmup–stable–decay schedule begins to decay.
- [Fantastic Pretraining Optimizers and Where to Find Them — Wen et al. (2025)](https://arxiv.org/abs/2509.02046) — free; re-benchmarks ten optimizers under matched tuning and finds that many reported speedups over AdamW shrink, the evidence behind the matched comparisons of :numref:`sec_muon` and :numref:`sec_practice`.

**Tutorials, notes, and interactive**

- [Why Momentum Really Works — Gabriel Goh, Distill (2017)](https://distill.pub/2017/momentum/) — free, interactive; damping and acceleration with sliders for $\eta$ and $\beta$, showing the critical value of $\beta$ and the ringing trajectories of :numref:`sec_momentum`.
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder (2016)](https://www.ruder.io/optimizing-gradient-descent/) — free; a survey of the classical progression from :numref:`sec_sgd` through :numref:`sec_adam`, and a record of common practice in 2016.
- [Deep Learning Tuning Playbook — Godbole et al., Google Research](https://github.com/google-research/tuning_playbook) — free; the division into scientific, nuisance, and fixed hyperparameters and the budget-tiered sweeps that :numref:`sec_practice` teaches, from a team that ran them at production scale.
- [Muon: An Optimizer for Hidden Layers in Neural Networks — Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/) — free; the original post, with the design decisions, Newton–Schulz coefficients, and ablations behind the optimizer that :numref:`sec_muon` builds from scratch.
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — free; a compact derivation of Muon as steepest descent under a norm, the note-form companion to the derivation in :numref:`sec_muon`.
- [The Practitioner's Guide to the Maximal Update Parameterization — EleutherAI](https://blog.eleuther.ai/mutransfer/) — free; muP implemented step by step with the coordinate checks of :numref:`sec_scaling`, including the failures a first implementation runs into.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9]{.kicker}

How optimizers take their steps<br>
**landscapes · gradient descent · noise · momentum · Adam and AdamW · schedules · Muon · batch size · scaling · practice**
:::
:::

::: {.slide title="The Steepest Direction Caps the Step, the Flattest Sets the Pace"}
[The problem]{.kicker}

- The valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$: curvature 4 along $x_2$,
  0.2 along $x_1$.
- Gradient descent diverges for $\eta > 2/4 = 0.5$; at such a rate $x_1$
  shrinks by less than a tenth per step.
- At its best learning rate it needs 47 steps to bring both coordinates to
  1% of their starting values.
- A network's curvatures differ by factors of hundreds or more, and each
  gradient is a noisy minibatch estimate.

::: {.d2l-note}
Which direction, which step size, and how much averaging should an
optimizer use, and how do the answers change as models and batches grow?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- predict from the curvature which learning rates converge, and how fast;
- explain the noise ball of SGD and how batch size and decay shrink it;
- derive momentum, Adam, AdamW, and Muon, and implement each from scratch;
- design a learning-rate schedule with warmup and decay;
- measure the critical batch size, and transfer learning rates across
  widths with muP;
- set up, guard, and compare training runs with matched tuning.

**Running example:** the valley, then a tiny transformer language model
and a small CNN.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| Landscapes | what makes a loss hard to minimize? |
| Gradient Descent | how large may a step be, and how does preconditioning help? |
| Stochastic and Minibatch GD | what changes when the gradient is a noisy estimate? |
| Momentum | how does averaging over steps speed up a narrow valley? |
| Adam and AdamW | what does a learning rate per coordinate achieve? |
| Schedules | how should the learning rate change over a run? |
| Muon | what step should a weight matrix take? |
| Batch Size | when do more examples per step stop saving steps? |
| Scaling Up | how do learning rates carry over to wider models? |
| Practice | how do we configure, guard, and compare runs? |
:::

::: {.slide title="Every Section Measures Its Claims on a Small Testbed"}
[Running examples]{.kicker}

| section | testbed |
|---|---|
| Landscapes, Gradient Descent | the valley; a small network at the edge of stability |
| Stochastic and Minibatch GD | the valley with noisy gradients; linear regression on airfoil data |
| Momentum | the valley; the airfoil regression |
| Adam and AdamW | `TinyLM` on the bytes of WikiText-103; `FashionCNN` on Fashion-MNIST |
| Schedules | the noisy valley; LeNet with batch normalization |
| Muon, Batch Size | `TinyLM` and `FashionCNN` |
| Scaling Up | MLPs on Fashion-MNIST, widths 128 to 1,024 |
| Practice | `TinyLM`; LeNet with batch normalization |
:::
