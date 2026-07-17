# Landscapes
:label:`sec_optimization-intro`

For a deep learning problem we first define a loss function, and once we
have it, an optimization algorithm drives it down. In optimization the loss
is called the *objective function*; by convention we minimize, and if we
ever need to maximize something, flipping the sign of the objective
suffices. None of this is difficult to state. What is difficult is the
surface being minimized: the graph of a deep network's loss over a
parameter space with millions of dimensions, and the shape of that surface
decides which algorithms work. This section surveys the terrain before the
chapter builds the machinery: what minimizing the objective does and does
not accomplish, the places where gradients die, and the two properties of
the surface — curvature and noise — that set the pace of every method that
follows.

One idea organizes the whole chapter. An optimizer is three decisions.
First, a *descent direction*: which way counts as "down" depends on which
norm measures the size of a step — the gradient is the answer under the
Euclidean norm, not the only answer, and changing the norm changes the
algorithm, a thread that pays off in :numref:`sec_muon`. Second, a *step
size over time*: how far to trust the local slope, and how that trust
should shrink or grow over a training run, the subject of
:numref:`sec_scheduler`. Third, a *way of living with noise*: in practice
every gradient is an estimate computed on a minibatch, and the batch size,
together with averaging over time, decides how noisy an estimate we act on
(:numref:`sec_minibatch_sgd`, :numref:`sec_batch_size`). Each method in
this chapter, from gradient descent to Muon, is a particular way of making
these three decisions.

## The Goal of Optimization

Optimization supplies deep learning with a means, but the two have
different ends. Optimization cares about the objective it was handed;
learning cares about performance on data the model has never seen. As
discussed in :numref:`sec_generalization_basics`, training error and
generalization error generally differ, and driving the first toward zero
can even hurt the second. In the vocabulary of
:numref:`subsec_empirical-risk-and-risk`, the *empirical risk* is the
average loss over the training set, while the *risk* is the expected loss
over the whole population. The optimizer only ever sees the former.

```{.python .input #optimization-intro-goal-of-optimization-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input #optimization-intro-goal-of-optimization-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

To make the gap concrete we define two functions: a smooth risk function
`f`, and an empirical risk function `g` that wobbles around it, the way an
average over finitely many training examples wobbles around an
expectation.

```{.python .input #optimization-intro-goal-of-optimization-2}
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

The minimum of the empirical risk need not sit at the minimum of the risk,
and here it does not:

```{.python .input #optimization-intro-goal-of-optimization-3}
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

No optimizer, however good, can close this gap: it is a property of the
data, not of the algorithm, and closing it is the business of the
regularization and model-selection tools met earlier in the book. For the
rest of the chapter we therefore set generalization aside and take the
objective at face value. Even so restricted, the problem is hard. Deep
learning objectives admit no analytical solution of the kind we found for
linear regression in :numref:`sec_linear_regression`, so every algorithm
in this chapter is iterative — and the surface it iterates over is nothing
like a convex bowl.

## Where Gradients Vanish

An iterative method needs a signal to follow, and the gradient is that
signal. The classical hazards of nonconvex optimization are the places
where the signal gives out: local minima, saddle points, and flat regions
of saturated activations. We look at each in turn.

### Local Minima

For an objective function $f(x)$, if the value of $f$ at $x$ is smaller
than at any nearby point, then $x$ is a *local minimum*. If it is smallest
over the entire domain, $x$ is the *global minimum*. For example, the
function

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0$$

has a local minimum that is not global:

```{.python .input #optimization-intro-local-minima}
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Deep learning objectives have many local minima, and an iterate that lands
near one sees its gradient approach zero: from the signal alone, a local
minimum is indistinguishable from the global one. Only some amount of
noise dislodges the parameters — one reason, as we will see below, that
the noise in minibatch gradients is not purely a nuisance.

### Saddle Points

Besides local minima, *saddle points* make gradients vanish: locations
where every gradient component is zero but which are neither a minimum nor
a maximum of the function. Consider $f(x) = x^3$. Its first and second
derivatives both vanish at $x=0$, and optimization can stall there even
though it is no minimum at all:

```{.python .input #optimization-intro-saddle-points-1}
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Saddle points in higher dimensions are more insidious. Consider
$f(x, y) = x^2 - y^2$: its saddle point at $(0, 0)$ is a minimum with
respect to $x$ and a maximum with respect to $y$, and the surface looks
like the saddle that gives the phenomenon its name:

```{.python .input #optimization-intro-saddle-points-2}
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Why saddle points dominate in high dimension is a counting argument.
Suppose the input of a function is a $k$-dimensional vector and its output
a scalar, so its Hessian matrix has $k$ eigenvalues. At a point where the
gradient is zero:

* if all $k$ eigenvalues are positive, we have a local minimum;
* if all are negative, a local maximum;
* if some are positive and some negative, a saddle point.

For a zero-gradient point of a high-dimensional function to be a local
minimum, *every one* of thousands or millions of eigenvalues must be
positive; if signs were even roughly balanced coin flips, nearly every
critical point would be a saddle. Convex functions — those whose Hessian
eigenvalues are nowhere negative — have neither saddle points nor spurious
minima, which is one reason classical optimization theory is built on
them. Deep learning objectives are not convex, but the theory has not
therefore become useless; we return to what it still offers at the end of
this section.

### Vanishing Gradients

The most insidious way to starve the gradient signal involves no critical
point at all. Recall the activation functions of
:numref:`subsec_activation-functions` and suppose we want to minimize
$f(x) = \tanh(x)$ starting from $x = 4$. The derivative is
$f'(x) = 1 - \tanh^2(x)$, so $f'(4) = 0.0013$: the surface is simply very
flat where we happen to stand, and gradient descent barely moves for a
long time before making progress.

```{.python .input #optimization-intro-vanishing-gradients}
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Vanishing gradients made deep networks genuinely hard to train before the
ReLU activation and careful initialization; those fixes belong to model
design (:numref:`subsec_activation-functions`) rather than to the
optimizer. The hazards of this section, then, are real, but two facts
soften them. Deep learning does not need *the* global minimum — a good
approximate local one serves — and, as the next section shows, what
actually limits training speed day to day is usually something else.

## Curvature and Noise

Zero-gradient traps are the textbook picture of why nonconvex
optimization is hard. In daily practice they are rarely what hurts.
Training is slow, or unstable, for two humbler reasons: the gradient is a
poor guide when curvature differs across directions, and we never see the
exact gradient anyway. These two — ill-conditioning and noise — are the
recurring villains of this chapter, and most of its methods exist to fight
one or the other.

### An Ill-Conditioned Valley

Take the simplest curved objective, a quadratic valley

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2,$$

which curves gently along $x_1$ (second derivative $0.2$) and steeply
along $x_2$ (second derivative $4$). Gradient descent updates both
coordinates with the same learning rate $\eta$, and each step multiplies
$x_1$ by $1 - 0.2\,\eta$ and $x_2$ by $1 - 4\eta$. The steep direction
sets a ceiling: for $x_2$ to shrink rather than explode we need
$|1 - 4\eta| < 1$, that is $\eta < 0.5$. The flat direction sets the pace:
for any stable $\eta$, each step keeps more than $90\%$ of $x_1$. To watch
the squeeze we borrow two helpers built in :numref:`sec_gd` —
`d2l.train_2d` iterates an update rule from a fixed starting point, and
`d2l.show_trace_2d` draws the resulting trace over the objective's
contours — and run 30 steps at $\eta = 0.45$, just under the ceiling:

```{.python .input #optimization-intro-an-ill-conditioned-valley}
def f_valley(x1, x2):  # Second derivatives 0.2 and 4
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_valley(x1, x2, s1, s2):
    eta = 0.45  # Just under the stability ceiling of 0.5
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_valley, d2l.train_2d(gd_valley, steps=30))
```

The trace is the signature of ill-conditioning: zig-zag *across* the
valley, crawl *along* it. The steep coordinate overshoots the valley floor
on every step, its sign flipping each iteration, while the flat coordinate
sheds only nine percent of its remaining distance per step — at that rate,
every factor of ten along $x_1$ costs about 24 steps. The number that controls this
squeeze is the ratio of the largest to the smallest curvature, the
*condition number*

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}},$$

here $4/0.2 = 20$. In general the steep curvature caps the learning rate
at $2/\lambda_{\max}$, the flat curvature then contracts by only
$1 - 2/\kappa$ per step, and the iteration count grows *linearly* with
$\kappa$ — the arithmetic is worked out in
:numref:`subsec_mdl-quadratic-model`. For deep networks $\kappa$ is not
$20$; measured values run to the thousands and beyond, and this valley is
the honest cartoon of why plain gradient descent crawls. Much of the
chapter is aimed at exactly this picture: momentum cuts the effective cost
from $\kappa$ to $\sqrt{\kappa}$ (:numref:`sec_momentum`), adaptive
methods rescale each coordinate by its own history (:numref:`sec_adam`),
and Muon rescales whole matrices at once (:numref:`sec_muon`).

### The Edge of Stability

The valley analysis treats curvature as a fixed property of the surface,
and the classical advice follows from it: measure the sharpness
$\lambda_{\max}$, then choose $\eta < 2/\lambda_{\max}$. On real networks
the causality runs backwards. Train a network with full-batch gradient
descent and the sharpness *rises* — "progressive sharpening" — until it
reaches roughly $2/\eta$, and then hovers there, with the loss still
falling, non-monotonically, in the very regime the quadratic analysis
forbids :cite:`Cohen.Kaur.Li.ea.2021`. The stability ceiling behaves less
like a fence the optimizer must stay behind and more like an attractor it
equilibrates onto: you pick $\eta$, and the network adapts its curvature
to your choice. Training does not live in the tidy descent regime that
most of this chapter's stated results (and the appendix's proofs) analyze;
the results remain the right guide to the mechanisms, but this is a gap
worth knowing about, and it is one reason the learning-rate schedules of
:numref:`sec_scheduler` — warmup especially — matter as much as they do.
The phenomenon is easy to check on a 25-parameter network, and
:numref:`subsec_mdl-quadratic-model` does exactly that.

### Noisy Gradients

The second villain is that the gradient we act on is an estimate. The loss
is an average over the training set, so computing its exact gradient costs
a full pass over the data; every practical method instead uses a minibatch
of $b$ examples. The estimate is unbiased, and its variance falls like
$1/b$ — :numref:`sec_sgd` measures exactly this on a real network, three
orders of magnitude of batch size falling neatly on the $1/b$ line. Noise
changes the character of the iteration: with a constant learning rate the
parameters do not converge but rattle around the optimum in a *noise
ball* whose radius scales with $\eta$, which is the fundamental reason
learning rates must decay (:numref:`sec_sgd`, :numref:`sec_scheduler`).
Batch size becomes a second dial next to the learning rate — one with
hardware consequences (:numref:`sec_minibatch_sgd`) and, at scale, a
measurable point of diminishing returns (:numref:`sec_batch_size`) — and
averaging over time, momentum's second job, quiets noise that batching
alone leaves behind (:numref:`sec_momentum`). Nor is noise purely a tax:
it is what bounces the iterate out of the local minima and saddle points
of the previous section, at no charge. Living with noise — spending it,
canceling it, budgeting for it — is the third of the chapter's three
decisions.

## What Convexity Still Buys

Every surface in this section was nonconvex, deliberately so. Yet the
vocabulary we used to describe them — condition number, convergence rate,
noise ball — comes from *convex* analysis, where each of these is a
theorem rather than a cartoon. That is the first thing convexity still
buys: a language, and clean baselines. A convex function has no bad local
minima and no saddle points to hide in, so any weakness an optimizer shows
on a convex problem is intrinsic to the optimizer. If a method misbehaves
on a quadratic, it has no business near a transformer, and throughout this
chapter new methods meet quadratics first.

The second purchase is local. Near a good minimum, a smooth loss is
approximately a quadratic bowl — the bottom of the surface looks locally
convex even when the whole is anything but. This is why the valley
analysis above predicts the late-training behavior of real networks, and
it underwrites practical tricks: averaging iterates near the bottom of the
bowl, as in stochastic weight averaging
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`, is a convex-analysis idea
that transfers to deep networks essentially intact
(:numref:`sec_practice`).

The limits are just as instructive. A deep network's loss cannot be convex
globally: permuting the hidden units of a layer leaves the function
computed unchanged, so every minimum comes with a combinatorial family of
separated copies of itself — the first exercise below makes this precise —
while a convex function's minima form a single connected set. Convexity
for deep learning is therefore a local approximation and a source of
tools, never a global fact. The full treatment — convex sets and
functions, Jensen's inequality, why local minima of convex functions are
global, duality, projections — lives in :numref:`sec_mdl-convexity` of the
mathematical appendix, whose optimization chapter
(:numref:`chap_mdl-optimization`) carries the proofs this chapter owes.

## Summary

Optimization and learning share a loss function but not a goal: the
optimizer minimizes empirical risk, while learning wants low risk, and no
optimizer can close that gap by itself. On the training objective, the
classical hazards are the places where gradients vanish — local minima,
saddle points (overwhelmingly more common in high dimension), and
saturated activations. The hazards that dominate practice are different:
curvature, summarized by the condition number $\kappa$, which forces a
single learning rate to serve directions of very different steepness; and
noise, since minibatch gradients are estimates whose variance we choose
via the batch size. Real training adds a twist to the classical stability
story — sharpness rises until it sits at the edge that the step size
tolerates. Convex analysis survives all this as a source of vocabulary,
baselines, and local approximations. The rest of the chapter builds the
machinery: a descent direction, a step size over time, and a way of
living with noise.

## Exercises

1. Consider a simple MLP with a single hidden layer of, say, $d$
   dimensions in the hidden layer and a single output. Show that for any
   local minimum there are at least $d!$ equivalent solutions that behave
   identically.
1. Assume that we have a symmetric random matrix $\mathbf{M}$ where the
   entries $M_{ij} = M_{ji}$ are each drawn from some probability
   distribution $p_{ij}$. Furthermore assume that $p_{ij}(x) = p_{ij}(-x)$,
   i.e., that the distribution is symmetric (see e.g.,
   :citet:`Wigner.1958` for details).
    1. Prove that the distribution over eigenvalues is also symmetric.
       That is, for any eigenvector $\mathbf{v}$ the probability that the
       associated eigenvalue $\lambda$ satisfies
       $P(\lambda > 0) = P(\lambda < 0)$.
    1. Why does the above *not* imply $P(\lambda > 0) = 0.5$?
1. Assume that you want to balance a (real) ball on a (real) saddle.
    1. Why is this hard?
    1. Can you exploit this effect also for optimization algorithms?
1. Consider the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ from this
   section.
    1. What is the largest learning rate for which gradient descent still
       converges? Verify your answer with `d2l.train_2d`.
    1. At $\eta = 0.45$, by what factor per step do $|x_1|$ and $|x_2|$
       shrink? Roughly how many steps does it take to reduce $|x_1|$ by a
       factor of $100$? Check your prediction numerically.
    1. For $f(\mathbf{x}) = \frac{\lambda_{\min}}{2} x_1^2 +
       \frac{\lambda_{\max}}{2} x_2^2$ with the best stable learning rate,
       show that the number of steps needed grows linearly with the
       condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.
    1. Suppose you were allowed to rescale the coordinate
       $\tilde{x}_1 = \alpha x_1$ before optimizing. Which $\alpha$ makes
       the valley perfectly conditioned? Which sections of this chapter
       estimate such rescalings automatically, from gradients alone?
1. What other challenges involved in deep learning optimization can you
   think of?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/487)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/489)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.1]{.kicker}

What makes deep-net optimization hard<br>
**risk vs. empirical risk · where gradients vanish · curvature and noise · the edge of stability**
:::
:::

::: {.slide title="An optimizer is three decisions"}
[The chapter's frame]{.kicker}

1. A **descent direction** — which way is "down"? Depends on which *norm*
   measures the step. Euclidean → the gradient. Other norms → other
   algorithms (payoff: Muon).
2. A **step size over time** — how far to trust the local slope, and how
   that trust changes over a run (schedules, warmup).
3. A **way of living with noise** — every gradient is a minibatch
   estimate; batch size and averaging set the noise level.

Every method in this chapter, GD through Muon, is one way of making these
three decisions.
:::

::: {.slide title="Optimization vs. learning"}
Optimization minimizes the *empirical risk* (training loss). Learning
wants low *risk* (expected loss on the population). The optimizer only
ever sees the former:

@optimization-intro-goal-of-optimization-1

. . .

@optimization-intro-goal-of-optimization-2

. . .

The two minima sit in different places — and no optimizer can fix that:

@optimization-intro-goal-of-optimization-3
:::

::: {.slide title="Local minima"}
$f(x) = x \cos(\pi x)$ has a local minimum that is not global. Near it,
the gradient goes to zero — the signal cannot tell the two apart:

@optimization-intro-local-minima

Only *noise* can knock the iterate out — minibatch variance does this for
free.
:::

::: {.slide title="Saddle points"}
1D: $f(x) = x^3$ has $f'(0) = 0$, yet no minimum:

@optimization-intro-saddle-points-1

. . .

High-dim: a zero-gradient point is a minimum only if **all** Hessian
eigenvalues are positive — with mixed signs it is a saddle. At $10^6$
parameters, essentially every critical point is a saddle:

@optimization-intro-saddle-points-2
:::

::: {.slide title="Vanishing gradients"}
No critical point needed: $f(x) = \tanh(x)$ at $x = 4$ has
$f'(4) \approx 0.0013$. The surface is just *flat* where we stand:

@optimization-intro-vanishing-gradients

ReLU and good initialization fixed this at the *model* level — not the
optimizer's job.
:::

::: {.slide title="The first villain: curvature"}
$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$: curvatures $0.2$ and $4$, one
learning rate. Steep direction caps $\eta < 0.5$; flat direction then
keeps $> 90\%$ of its value per step:

@optimization-intro-an-ill-conditioned-valley

. . .

Zig-zag across, crawl along. Condition number $\kappa =
\lambda_{\max}/\lambda_{\min} = 20$; iterations scale **linearly with
$\kappa$**. Real networks: $\kappa$ in the thousands.

::: {.d2l-note}
Momentum → $\sqrt{\kappa}$. Adam → per-coordinate rescaling. Muon →
per-matrix rescaling. Most of the chapter fights this picture.
:::
:::

::: {.slide title="The edge of stability"}
Classical advice: measure sharpness $\lambda_{\max}$, pick
$\eta < 2/\lambda_{\max}$.

Measured reality (Cohen et al., 2021): causality runs **backwards** —
training *raises* sharpness ("progressive sharpening") until it reaches
$\approx 2/\eta$, then hovers there. Loss keeps falling, non-monotonically,
in the "forbidden" regime.

- The ceiling is an *attractor*, not a fence: pick $\eta$, the network
  adapts its curvature to it.
- Training does not live in the tidy descent regime the proofs analyze.
- One reason warmup and schedules matter (§ Schedules); measured on a
  25-parameter net in the math appendix.
:::

::: {.slide title="The second villain: noise"}
The gradient is a minibatch estimate: unbiased, variance $\propto 1/b$
(measured on a real network in the SGD section).

- Constant $\eta$ → no convergence: a **noise ball** of radius
  $\propto \eta$. Hence decaying learning rates and schedules.
- Batch size = a second dial, with hardware consequences (Minibatches)
  and diminishing returns at scale (Batch Size).
- Momentum's second job: averaging noise over *time*.
- Not purely a tax — noise kicks the iterate out of local minima and
  saddles for free.
:::

::: {.slide title="What convexity still buys"}
Deep losses are *not* convex — permutation symmetry alone gives every
minimum $d!$ separated copies; convex minima form one connected set.

What survives:

- **Language and baselines**: condition number, rates, noise ball — all
  theorems in the convex world. A method that fails on a quadratic has no
  business near a transformer.
- **Local honesty**: near a good minimum the loss is approximately a
  quadratic bowl — which is why the valley cartoon predicts late-training
  behavior (and why weight averaging works).

Full treatment: the convexity chapter of the math appendix.
:::

::: {.slide title="Recap"}
- Minimizing training loss ≠ minimizing test loss; that gap belongs to
  regularization, not the optimizer.
- Classical hazards: local minima, saddles (dominant in high dim),
  vanishing gradients.
- Practical hazards: **curvature** (condition number $\kappa$) and
  **noise** (minibatch variance) — the chapter's two villains.
- Modern twist: training equilibrates at the edge of stability.
- The toolkit ahead = three decisions: direction, step size over time,
  living with noise.
:::
