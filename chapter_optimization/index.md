# Optimization Algorithms
:label:`chap_optimization`

Every model in this book has used an optimizer to update its parameters once
per minibatch. This chapter examines the principles behind those updates.
An optimization algorithm makes three related choices: the descent
direction, the step size and its variation over time, and the treatment of
noise in minibatch gradients. The appropriate descent direction depends on
how the size of an update is measured; the step size must account for
curvature and training time; and the gradient variance depends on batching
and averaging.

Two properties of the loss surface make the decisions consequential. The
first is *curvature*: a deep network's loss rises steeply along some
directions of parameter space and slowly along others. A single step size
must serve both, so a large step causes oscillation in steep directions
while a small step makes slow progress in flat directions. The second
is *noise*: an exact gradient costs a full pass over the dataset, so
scalable methods use minibatch estimates whose variance depends on the batch
size. :numref:`sec_optimization-intro` introduces these properties.
The next five sections develop gradient descent and preconditioning
(:numref:`sec_gd`), stochastic gradients and learning-rate decay
(:numref:`sec_sgd`), the computational effects of minibatching
(:numref:`sec_minibatch_sgd`), momentum for ill-conditioned objectives
(:numref:`sec_momentum`), and per-coordinate scaling from AdaGrad through
RMSProp to Adam (:numref:`sec_adam`). The Adam section also builds the tiny
transformer language model used in later experiments.

The second half removes simplifying assumptions from this progression. AdamW
separates shrinkage from adaptive preconditioning, and learning-rate schedules
vary the step over time. Muon changes the geometry used to choose matrix
updates. Batch-size experiments connect gradient variance to parallel compute,
while scaling and practice address hyperparameter transfer, clipping, weight
averaging, and matched comparisons.

This chapter emphasizes computations and controlled experiments. The
mathematical appendix (:numref:`chap_mdl-optimization`) proves the descent
lemma, the condition-number law, momentum's $\sqrt{\kappa}$ acceleration,
the Robbins–Monro conditions, and Adam's bias correction under explicit
assumptions. The two treatments can be read in either order.

Several methods in the second half are recent, and their relative performance
remains protocol-dependent. The comparisons therefore tune each optimizer
under the same budget and distinguish evidence from small testbeds, public
benchmarks, and reported production runs.

```toc
:maxdepth: 2

optimization-intro
gd
sgd
minibatch-sgd
momentum
adam
adamw
lr-scheduler
muon
batch-size
scaling
practice
```

## Resources and Further Reading {.unnumbered}

The references below cover convex foundations, classical methods, current
optimizers, and tuning. All are
freely accessible online except where noted. The optimization chapter of
the mathematical appendix (:numref:`chap_mdl-optimization`) keeps its own
resource list for the theory side — convex-optimization texts and courses
with proofs — and we do not repeat those entries here.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — free PDF; the standard reference behind the vocabulary this chapter uses informally — conditioning, convergence rates, duality, projections — and the right place to see the analyses that :numref:`sec_gd` and :numref:`sec_sgd` state without proof carried out in full.
- [Numerical Optimization — Nocedal & Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5) — a comprehensive treatment of line search, trust regions, and the quasi-Newton methods discussed in :numref:`sec_gd` (paywalled; widely held in university libraries).

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch — Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — free; the graded version of this chapter's exercises: implement AdamW exactly as :numref:`sec_adamw` does, account for optimizer-state memory byte by byte, and run the learning-rate and batch-size sweeps that :numref:`sec_practice` turns into method; the accompanying lectures are on YouTube.

**Foundational and current papers**

- [Old Optimizer, New Norm: An Anthology — Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325) — free; the unification that organizes :numref:`sec_muon`: SGD, sign descent/Adam, and Shampoo are each steepest descent under a different norm, which compares these methods through a shared geometric question.
- [An Empirical Model of Large-Batch Training — McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — free; defines the gradient-noise scale and the critical batch size, the two quantities measured at the center of :numref:`sec_batch_size`, and predicts when doubling the batch stops halving the steps.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al. (2024)](https://arxiv.org/abs/2410.05192) — free; the modern upgrade of the ill-conditioned valley of :numref:`sec_optimization-intro`: a river-valley landscape in which the stable phase travels along the river and the decay phase descends its bank, explaining the WSD loss cliff of :numref:`sec_scheduler`.
- [Fantastic Pretraining Optimizers and Where to Find Them — Stanford (2025)](https://arxiv.org/abs/2509.02046) — free; re-benchmarks ten optimizers under matched tuning and finds that many reported speedups over AdamW shrink; it motivates the matched-comparison discipline that :numref:`sec_muon` and :numref:`sec_practice` adopt as a rule.
- [Benchmarking Neural Network Training Algorithms — Dahl et al. (2023)](https://arxiv.org/abs/2306.07179) — free; the MLCommons AlgoPerf benchmark ([code and results](https://github.com/mlcommons/algorithmic-efficiency)): why optimizer verdicts depend on the comparison protocol, the evidence standard behind the caveats of :numref:`sec_muon` and :numref:`sec_practice`.

**Tutorials, notes, and interactive**

- [Why Momentum Really Works — Gabriel Goh, Distill (2017)](https://distill.pub/2017/momentum/) — free, interactive; an interactive treatment of damping and acceleration with sliders for $\eta$ and $\beta$, illustrating the critical value of $\beta$ and oscillatory trajectories.
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder (2016)](https://www.ruder.io/optimizing-gradient-descent/) — free; a survey of the classical progression from :numref:`sec_sgd` through :numref:`sec_adam` and a useful record of common practice in 2016.
- [Deep Learning Tuning Playbook — Godbole et al., Google Research](https://github.com/google-research/tuning_playbook) — free; the scientific/nuisance/fixed-hyperparameter methodology and budget-tiered sweeps that :numref:`sec_practice` teaches, from the team that ran them at production scale.
- [Muon: An Optimizer for Hidden Layers in Neural Networks — Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/) — free; the original post: design decisions, Newton–Schulz coefficients, and ablations behind the optimizer that :numref:`sec_muon` builds from scratch.
- [modded-nanogpt — Keller Jordan et al.](https://github.com/KellerJordan/modded-nanogpt) — free; the speedrun repository in which Muon was first demonstrated, with documented and reproducible records; its reporting practice informs the evidence standard that :numref:`sec_muon` holds up as a model.
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — free; a compact derivation of Muon from the steepest-descent-under-a-norm principle, the note-form companion to the derivation in :numref:`sec_muon`.
- [The Practitioner's Guide to the Maximal Update Parameterization — EleutherAI](https://blog.eleuther.ai/mutransfer/) — free; muP implemented step by step with the coordinate-check experiments of :numref:`sec_scaling`, including the failure modes a first implementation actually hits.
