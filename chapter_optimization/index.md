# Optimization Algorithms
:label:`chap_optimization`

Every model in this book has been trained by a line we have not yet
examined. Since fitting our first models in :numref:`chap_regression` we
have constructed an optimizer — `SGD` at first, `Adam` soon after — handed
it the parameters and a learning rate, and let the training loop invoke it
once per minibatch, trusting the loss to fall. It nearly always did. This
chapter opens that box. Inside is not one idea but three decisions, made
anew by every algorithm: a *descent direction* — which way counts as
"down", a question whose answer turns out to depend on which norm measures
the size of a step; a *step size over time* — how boldly to move, and how
that boldness should change over a run; and a *way of living with noise* —
since almost every gradient is estimated from a minibatch rather than
computed exactly. The twelve sections of this chapter are these three
decisions unfolded, roughly in the order history made them.

Two properties of the loss surface make the decisions consequential. The
first is *curvature*: a deep network's loss rises steeply along some
directions of parameter space and barely at all along others, and a single
step size must serve both — too bold and the steep directions oscillate
out of control, too timid and the flat directions never arrive. The second
is *noise*: the exact gradient costs a full pass over the dataset, so
any method that scales settles for a minibatch estimate whose variance is
ours to choose. :numref:`sec_optimization-intro` maps this terrain. The
five sections after it climb the classical ladder, each rung repairing a
failure of the one below: gradient descent and the ideal of
preconditioning (:numref:`sec_gd`), stochastic gradients and why learning
rates must decay (:numref:`sec_sgd`), minibatching and what the hardware
has to say about it (:numref:`sec_minibatch_sgd`), momentum against
curvature (:numref:`sec_momentum`), and per-coordinate scaling from
AdaGrad through RMSProp to Adam (:numref:`sec_adam`) — where we also build
the tiny transformer language model on which the rest of the chapter runs
its experiments.

The ladder's top rung dates from 2014, and for years textbooks stopped
there. Practice did not. The second half of the chapter is the layer a
practitioner actually configures today: decoupled weight decay, the rule
of what not to decay, and the memory arithmetic of optimizer state
(:numref:`sec_adamw`); warmup, cosine decay, and the warmup–stable–decay
schedules of large-model training (:numref:`sec_scheduler`); the
realization that steepest descent depends on the norm — under one norm it
recovers sign descent and, in essence, Adam, and under the spectral norm
it yields Muon, a credible challenger to Adam's decade
(:numref:`sec_muon`); how large a batch can grow before more parallelism
stops buying anything (:numref:`sec_batch_size`); how to tune a small
model and transfer the result to one too expensive to tune
(:numref:`sec_scaling`); and the craft of running real training — recipes,
gradient clipping, weight averaging, and how to sweep
(:numref:`sec_practice`).

A word on what this chapter is not. It states results, demonstrates
phenomena, and gives intuition; it does not prove. The descent lemma, the
condition-number law, momentum's $\sqrt{\kappa}$ acceleration, the
Robbins–Monro conditions, Adam's bias correction, and the convex analysis
underneath all of them live in the optimization chapter of the
mathematical appendix (:numref:`chap_mdl-optimization`), which is written
as this chapter's proof volume. Where a section here says "one can show",
the appendix shows it. The two chapters can be read in either order; the
experiments here give the theorems something to predict.

The modern layer is younger than it looks, and parts of it are still
moving. Decoupled weight decay was published in 2017 and became a
universal default only years later; warmup–stable–decay schedules entered
large-model practice around 2024; Muon went from a speed-run leaderboard
in late 2024 to reported trillion-parameter production runs within about a
year — for optimizers, unusually fast. The field has also grown a
benchmarking discipline: matched-tuning comparisons and public benchmarks
now routinely shrink headline claims, and more than one celebrated
optimizer has failed to replicate under them. Where the evidence is
unsettled we say so in place, and the chapter's comparisons follow one
rule throughout: tuned against tuned, never a tuned challenger against a
default baseline.

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

The references below follow the chapter's arc — the convex foundations,
the classical ladder, the modern layer, and the craft of tuning. All are
freely accessible online except where noted. The optimization chapter of
the mathematical appendix (:numref:`chap_mdl-optimization`) keeps its own
resource list for the theory side — convex-optimization texts and courses
with proofs — and we do not repeat those entries here.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — free PDF; the standard reference behind the vocabulary this chapter uses informally — conditioning, convergence rates, duality, projections — and the right place to see the analyses that :numref:`sec_gd` and :numref:`sec_sgd` state without proof done properly.
- [Numerical Optimization — Nocedal & Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5) — the classical machinery in full: line search, trust regions, and the quasi-Newton methods that :numref:`sec_gd` names as the ideal deep learning cannot afford (paywalled, noted; widely held in university libraries).

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch — Assignment 1](https://github.com/stanford-cs336/assignment1-basics) — free; the graded version of this chapter's exercises: implement AdamW exactly as :numref:`sec_adamw` does, account for optimizer-state memory byte by byte, and run the learning-rate and batch-size sweeps that :numref:`sec_practice` turns into method; the accompanying lectures are on YouTube.

**Foundational and current papers**

- [Old Optimizer, New Norm: An Anthology — Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325) — free; the unification that organizes :numref:`sec_muon`: SGD, sign descent/Adam, and Shampoo are each steepest descent under a different norm, which turns the chapter's zoo of methods into one question asked three ways.
- [An Empirical Model of Large-Batch Training — McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — free; defines the gradient-noise scale and the critical batch size, the two quantities measured at the center of :numref:`sec_batch_size`, and predicts when doubling the batch stops halving the steps.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al. (2024)](https://arxiv.org/abs/2410.05192) — free; the modern upgrade of the ill-conditioned valley of :numref:`sec_optimization-intro`: a river-valley landscape in which the stable phase travels along the river and the decay phase descends its bank, explaining the WSD loss cliff of :numref:`sec_scheduler`.
- [Fantastic Pretraining Optimizers and Where to Find Them — Stanford (2025)](https://arxiv.org/abs/2509.02046) — free; re-benchmarks ten optimizers under matched tuning and watches most claimed speedups over AdamW shrink — the fair-comparison discipline that :numref:`sec_muon` and :numref:`sec_practice` adopt as a rule.
- [Benchmarking Neural Network Training Algorithms — Dahl et al. (2023)](https://arxiv.org/abs/2306.07179) — free; the MLCommons AlgoPerf benchmark ([code and results](https://github.com/mlcommons/algorithmic-efficiency)): why optimizer verdicts depend on the comparison protocol, the evidence standard behind the caveats of :numref:`sec_muon` and :numref:`sec_practice`.

**Tutorials, notes, and interactive**

- [Why Momentum Really Works — Gabriel Goh, Distill (2017)](https://distill.pub/2017/momentum/) — free, interactive; the damping and acceleration story of :numref:`sec_momentum` with sliders for $\eta$ and $\beta$ — the fastest way to internalize why the critical $\beta$ exists and what ringing looks like.
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder (2016)](https://www.ruder.io/optimizing-gradient-descent/) — free; the field guide to the classical ladder of :numref:`sec_sgd` through :numref:`sec_adam`, and a historical marker: everything a practitioner needed in 2016, and a measure of how much the modern layer has added since.
- [Deep Learning Tuning Playbook — Godbole et al., Google Research](https://github.com/google-research/tuning_playbook) — free; the scientific/nuisance/fixed-hyperparameter methodology and budget-tiered sweeps that :numref:`sec_practice` teaches, from the team that ran them at production scale.
- [Muon: An Optimizer for Hidden Layers in Neural Networks — Keller Jordan (2024)](https://kellerjordan.github.io/posts/muon/) — free; the original post: design decisions, Newton–Schulz coefficients, and ablations behind the optimizer that :numref:`sec_muon` builds from scratch.
- [modded-nanogpt — Keller Jordan et al.](https://github.com/KellerJordan/modded-nanogpt) — free; the speedrun repository where Muon first proved itself, with every record documented and reproducible — the evidence culture that :numref:`sec_muon` holds up as a model.
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — free; a compact derivation of Muon from the steepest-descent-under-a-norm principle, the note-form companion to the derivation in :numref:`sec_muon`.
- [The Practitioner's Guide to the Maximal Update Parameterization — EleutherAI](https://blog.eleuther.ai/mutransfer/) — free; muP implemented step by step with the coordinate-check experiments of :numref:`sec_scaling`, including the failure modes a first implementation actually hits.
