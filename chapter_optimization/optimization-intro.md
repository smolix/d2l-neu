# Landscapes
:label:`sec_optimization-intro`

Every loss curve in this book came from one procedure: compute the gradient
of the loss on a minibatch, take a small step against it, and repeat. The
curves were not alike. Some fell within an epoch and others crept for
dozens, and a learning rate that trained one model sent the loss of another
to `nan`. The procedure was the same each time. What differed was the
function it was applied to.

A model with a million parameters defines a loss over a million-dimensional
space, and it helps to think of this function as a landscape on which the
optimizer walks downhill, able to measure only the slope under its feet.
Whether a step makes progress depends on the shape of the surface nearby.
Where the surface is flat, the slope says almost nothing. Where it curves
steeply in one direction and gently in another, a step small enough to be
safe in the steep direction barely moves along the gentle one.

Which of these shapes is to blame when training is slow? The hazard with
the longest reputation is the local minimum, a dip in which the walk ends
too early. This section gives a different answer: uneven curvature. We meet
it on a valley in two variables, small enough to solve by hand, where the
ratio of the largest to the smallest curvature decides how many steps are
needed, whatever the learning rate. We then return to the points where the
slope vanishes and see why they do less harm than their reputation
suggests. The optimizer throughout is the plain gradient step. We analyze
it completely, because every later method of the chapter answers something
that goes wrong on this valley.

```{.python .input #optimization-intro-goal-of-optimization-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #optimization-intro-goal-of-optimization-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Goal of Optimization

In optimization, the function to be minimized is called the *objective
function*, or simply the objective. Optimization texts collect all
parameters into a single vector $\mathbf{x}$, and we follow them in this
chapter: $\mathbf{x}$ holds everything that earlier chapters wrote as
$(\mathbf{w}, b)$. Our objective is the *empirical risk*, the average loss
over the training set. Linear regression had a formula for its minimizer
(:numref:`sec_linear_regression`). A deep network has none, so every method
in this chapter is iterative: it starts somewhere and improves the
parameters step by step.

Learning wants something else, a low *risk*, the expected loss on examples
the model has not seen (:numref:`subsec_empirical-risk-and-risk`). The two
differ because the training set is a finite sample. That difference is not
the optimizer's doing, and no optimizer can remove it; regularization and
model selection address it (:numref:`sec_generalization_basics`). The
optimizer is not a bystander, though. The empirical risk of a network has
many minimizers, they do not all generalize equally well, and the learning
rate and the noise in the gradient help decide which of them a run reaches.
For most of this chapter we ask the narrower question of how fast and how
reliably a method lowers the objective it is given, and we point out the
places where its choice among minimizers shows.

## An Ill-Conditioned Valley

The smallest example of uneven curvature is a quadratic valley in two
variables,

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$
:eqlabel:`eq_opt-intro-valley`

It curves gently along $x_1$, with second derivative $\lambda_1 = 0.2$, and
steeply along $x_2$, with second derivative $\lambda_2 = 4$, so its contours
are long, narrow ellipses around the minimum at the origin. We minimize it
by gradient descent, the update we have used in one form or another since
:numref:`sec_linear_regression`: step against the gradient with a learning
rate $\eta$, that is, $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla
f(\mathbf{x})$. The gradient of the valley is $(0.2 x_1, 4 x_2)$, so each
step multiplies the two coordinates by fixed factors,

$$x_1 \leftarrow (1 - 0.2\eta)\, x_1, \qquad x_2 \leftarrow (1 - 4\eta)\, x_2.$$

The coordinates do not interact, and each makes its own demand on $\eta$.
The steep coordinate sets a ceiling: $x_2$ shrinks only if
$|1 - 4\eta| < 1$, that is, if $\eta < 0.5$. The flat coordinate sets the
pace: below that ceiling, $1 - 0.2\eta$ stays above $0.9$, so each step
removes less than a tenth of the distance along $x_1$. This valley is the
running example of the chapter: gradient descent, stochastic gradient
descent, momentum, and Adam are all tested on it.

Let us take $\eta = 0.45$, just under the ceiling, and start at
$(-5, -2)$. The steep factor is $1 - 4 \cdot 0.45 = -0.8$, so $x_2$ should
overshoot the valley floor on every step, flipping its sign and losing a
fifth of its size. The flat factor is $0.91$, so $x_1$ should shed only
nine percent of its remaining distance per step. At that rate every factor
of ten along $x_1$ takes about 24 steps, and 30 steps should leave $x_1$
near $-5 \cdot 0.91^{30} \approx -0.3$. We run the 30 steps and plot both
coordinates against the step count. The array `h` in the code holds the
two curvatures.

```{.python .input #optimization-intro-an-ill-conditioned-valley}
%%tab pytorch, jax
h = np.array([0.2, 4.0])  # Curvatures h1, h2 along x1 and x2

def gd_valley(eta, steps, x0=(-5.0, -2.0)):
    """Gradient descent on the valley; returns all iterates."""
    xs = [np.array(x0)]
    for _ in range(steps):
        xs.append(xs[-1] - eta * h * xs[-1])  # The gradient is h * x
    return np.stack(xs)

xs = gd_valley(0.45, steps=30)
d2l.plot(np.arange(31), [xs[:, 0], xs[:, 1]], 'step', 'coordinate',
         legend=['$x_1$', '$x_2$'])
```

The plot shows both predictions. The steep coordinate swings from side to
side and is too small to see after twenty steps, while the flat one is
still at $-0.3$ when the run ends. In the plane, the iterate zigzags across
the narrow valley while it creeps along its floor.

Could another learning rate do better? We count the steps until both
coordinates are within 1% of the minimum, measured relative to where they
started.

```{.python .input #optimization-intro-an-ill-conditioned-valley-1}
%%tab pytorch, jax
print('  eta   factor x1   factor x2   steps to 1%')
for eta in [0.1, 0.2, 0.3, 0.4, 0.45, 2 / 4.2, 0.49, 0.5]:
    xs = gd_valley(eta, steps=1000)
    done = np.all(np.abs(xs) <= 0.01 * np.abs(xs[0]), axis=1)
    steps = int(np.argmax(done)) if done.any() else 'never'
    print(f'{eta:5.3f}   {1 - 0.2 * eta:9.3f}   {1 - 4 * eta:9.3f}   '
          f'{steps:>11}')
```

Small learning rates are slow because the flat factor stays near one.
Raising $\eta$ helps until the steep factor, which grows more negative,
overtakes the flat one in size. The best rate balances the two:
$1 - 0.2\eta = -(1 - 4\eta)$ gives $\eta = 2/4.2 \approx 0.476$, where both
factors have size $0.905$, and the count bottoms out at 47 steps. Just
above it the steep coordinate becomes the slow one, and at $\eta = 0.5$ it
stops shrinking altogether. No choice of $\eta$ reaches 1% in fewer than
47 steps.

The table is lopsided around its best row: 49 steps at $\eta = 0.45$, 113
at $\eta = 0.49$. Close to the ceiling, a learning rate slightly too large
costs far more than one slightly too small.

## The Condition Number

The floor is not a peculiarity of this example. Nothing in the argument
depended on the valley being aligned with the axes or on its having two
variables. Take any quadratic
$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x}$, where
the symmetric matrix $\mathbf{H}$ is the *Hessian*, the matrix of second
derivatives (for the valley, $\textrm{diag}(0.2, 4)$). Its gradient is
$\mathbf{H}\mathbf{x}$, so a step of gradient descent maps $\mathbf{x}$ to
$(\mathbf{I} - \eta\mathbf{H})\mathbf{x}$, and an eigenvector of
$\mathbf{H}$ with eigenvalue $\lambda_i$ to $1 - \eta\lambda_i$ times
itself. It is therefore instructive to measure $\mathbf{x}$ along the
eigenvectors, which are orthogonal because $\mathbf{H}$ is symmetric. Each
of these coordinates is multiplied by its own factor $1 - \eta\lambda_i$,
as $x_1$ and $x_2$ were. Every quadratic is a valley like ours, possibly
turned in space, with one curvature per eigenvector. If all eigenvalues are
positive, in which case $\mathbf{H}$ is called *positive definite* and the
origin is the minimum, the largest eigenvalue $\lambda_{\max}$ sets the
ceiling $\eta < 2/\lambda_{\max}$, and the smallest, $\lambda_{\min}$,
sets the pace.

What decides the number of steps is the ratio of the two, the *condition
number*

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}},$$

which is $4/0.2 = 20$ for the valley. A problem with a large $\kappa$ is
called *ill-conditioned*. The best learning rate balances the two extreme
factors, as in the sweep. At $\eta = 2/(\lambda_{\min} + \lambda_{\max})$
both have size $(\kappa - 1)/(\kappa + 1)$, about $1 - 2/\kappa$ for large
$\kappa$, and every direction in between shrinks at least as fast. Each
step thus removes a fraction of about $2/\kappa$ of the error, and
shrinking the error to a fraction $\epsilon$ of its starting value takes
about $(\kappa/2) \log(1/\epsilon)$ steps, with $\log$ the natural
logarithm. For the valley the factor is $19/21 = 0.905$ and the estimate
is $10 \log 100 \approx 46$, against the 47 we measured
(:numref:`subsec_mdl-quadratic-model` gives the argument in full). The
number of steps grows in proportion to $\kappa$, and no learning rate can
change that. To do better we have to change the method, which is what most
of this chapter does.

For a network the definition needs care. It presumes a positive definite
Hessian, and the Hessian of a network's loss is not one: it has a few large
eigenvalues and a bulk of eigenvalues near zero, some of them negative
:cite:`Ghorbani.Krishnan.Xiao.2019`. The ratio
$\lambda_{\max}/\lambda_{\min}$ is then infinite or negative and says
nothing. What carries over is the comparison that the valley taught us to
make. The largest eigenvalue, called the *sharpness* of the network's loss,
limits the learning rate as $\lambda_2$ did on the valley, and below that
limit a direction of curvature $\lambda$ loses only a fraction
$\eta\lambda$ of its error per step. The sharpness exceeds the typical
curvature by orders of magnitude, so a network is a far narrower valley
than ours.

One thing the valley lacks is noise. Its gradient is exact, whereas a
minibatch gradient is an estimate whose error does not vanish at the
minimum, so a constant learning rate leaves the parameters fluctuating
around it (:numref:`subsec_linear-regression-minibatch-sgd`).
:numref:`sec_sgd` adds this noise to the valley and computes how far from
the minimum the iterates stay.

## Critical Points and Plateaus

A reader may object that the valley is too kind. It has a single minimum,
while the loss of a network has many, of different depth. Should we not
worry first about getting stuck in a poor one? The worry is about the
places where a gradient method loses its signal: *critical points*, where
the gradient is zero, and *plateaus*, where it is merely tiny. A *local
minimum*, where $f$ is smaller than at all nearby points, is a critical
point, and the gradient cannot tell it from the *global minimum*, where $f$
is smallest over the whole domain. Most critical points of a network's
loss, however, are not minima at all.

### Saddle Points
:label:`subsec_opt-intro-saddles`

The function $f(x, y) = x^2 - y^2$ has a critical point at the origin that
is neither a minimum nor a maximum (:numref:`fig_mdl-opt-saddle`). Along
the $x$-axis the origin is a minimum, along the $y$-axis it is a maximum,
and the surface has the shape of the saddle that gives such a point its
name, a *saddle point*.

![The saddle $z=x^2-y^2$: a minimum along $x$ and a maximum along $y$ (left), seen in the two slices through the origin (right).](../img/mdl-opt-saddle.svg)
:label:`fig_mdl-opt-saddle`

In high dimensions saddle points are the rule, and the Hessian shows why.
Its eigenvalues are the curvatures along its eigenvectors, as they were on
the valley. At a critical point of a function of $k$ variables, the signs
of the $k$ eigenvalues decide the shape:

* all $k$ eigenvalues positive: a strict local minimum;
* all $k$ negative: a strict local maximum;
* some positive and some negative: a saddle point, called *strict* because
  at least one direction curves downward;
* no negative and some zero eigenvalues, or no positive and some zero: the
  test is inconclusive, and higher-order terms decide. In one variable,
  $x^4$ and $x^3$ both have second derivative zero at the origin, and only
  the first has a minimum there.

A minimum needs every one of the $k$ signs to come out positive. If the
signs were independent fair coin flips, a critical point of a function with
$k = 1000$ parameters would be a minimum with probability
$2^{-1000} \approx 10^{-301}$, and nearly every critical point would be a
saddle. The coin-flip model is only a heuristic. The eigenvalues of a
Hessian are not independent, as
Exercise :exref:`ex_optimization-intro-eigenvalues-symmetric-random-matrices`
shows. Critical points located in small networks show the pattern in the
form that matters: those at high loss are saddles with many descending
directions, the fraction of negative eigenvalues falls as the loss falls,
and those at low loss have few descending directions or none
:cite:`Dauphin.Pascanu.Gulcehre.ea.2014`.

Does a saddle trap gradient descent? Only if the iterate arrives on it
exactly. On $x^2 - y^2$ the valley's rule applies with one curvature of
each sign: a step multiplies $x$ by $1 - 2\eta$ and $y$ by $1 + 2\eta$. The
second factor exceeds one, so any nonzero $y$ grows geometrically, and only
a start with $y = 0$ ends on the saddle. This holds in general: from a
random start, gradient descent with a small enough learning rate avoids
every strict saddle, because the starting points that lead into one form a
set of measure zero :cite:`Lee.Simchowitz.Jordan.ea.2016`. The escape can
take long when the iterate passes close to the saddle. Here the noise of
minibatch gradients is a help and not a nuisance: it pushes the iterate off
the ridge and shortens the escape, as
Exercise :exref:`ex_optimization-intro-saddle-escape-time` measures.

Critical points at high loss are saddles, and gradient methods leave
saddles, so the places where a run can come to rest have low loss. A run
stuck in a poor local minimum, the classical worry, is not what limits the
training of a large network.

### Plateaus

A gradient can also be too small to guide an update at a point that is not
critical at all. Suppose we minimize $f(x) = \tanh(x)$, starting from
$x = 4$. The derivative is $f'(x) = 1 - \tanh^2(x)$, so $f'(4) \approx
0.0013$. The surface is nearly flat at the starting point
(:numref:`fig_mdl-opt-tanh-flat`), and gradient descent barely moves for a
long time before it reaches the slope.

![Minimizing $f(x)=\tanh x$ from $x=4$ stalls: the surface is nearly flat, slope about $0.0013$, a vanishing gradient with no critical point.](../img/mdl-opt-tanh-flat.svg)
:label:`fig_mdl-opt-tanh-flat`

This is the vanishing-gradient problem of
:numref:`sec_numerical_stability` in one variable. In a deep network the
gradient of an early layer is a product of one derivative per later layer,
and a product of many factors below one shrinks fast. ReLU activations and
careful initialization removed most such plateaus
(:numref:`subsec_activation-functions`). They are remedies in the design of
the model, not in the optimizer. Uneven curvature is another matter: it is
present at every step of every run, and dealing with it is the optimizer's
job.

## Convexity as a Local Model

The valley is a convex function, and the loss of a network is not, so it
is fair to ask why what we learn on the one should carry over to the
other. A function is *convex* if the straight line between any two points
of its graph lies on or above the graph, as it does for a bowl. A convex
function has no saddle points and no local minima other than its global
ones, so any weakness an optimizer shows on a convex problem belongs to the
optimizer. A method that fails on a quadratic is unlikely to succeed on a
transformer, which is why the methods of this chapter meet the valley
before they meet a network.

A convex model is also accurate where training ends. Near a minimum with a
positive definite Hessian, a smooth loss is approximately a quadratic bowl,
so the valley describes the last phase of training reasonably well. The
vocabulary of this chapter, from the condition number onward, is that of
convex analysis, which the appendix develops
(:numref:`sec_mdl-convexity`).

The approximation is local and no more. Permuting the hidden units of a
layer leaves the function a network computes unchanged, so every minimum
comes with a large family of copies of itself, one per permutation. If the
loss were convex, the average of these copies would be a minimizer too.
That average is a network whose units in the layer are all identical. It is
no more expressive than a network with one unit and usually fits worse, as
Exercise :exref:`ex_optimization-intro-permutation-symmetric-minima` makes
precise. For deep learning, convexity is a local approximation and a source
of tools, never a global fact.

Even locally, the valley has told us only part of what we need. It gave a
ceiling for the learning rate and a best value, but on a quadratic whose
curvatures we chose ourselves. Nobody tells us the curvatures of a
network's loss. What limits the step on a general function, and what would
the ideal step be if the curvature could be measured?

## Summary

Optimization minimizes the empirical risk. The gap to the risk on new data
is not the optimizer's to close, although the optimizer does choose among
the minimizers of its objective. What slows gradient descent is uneven
curvature: one learning rate must stay below $2/\lambda_{\max}$ for the
steepest direction, and it then removes only a fraction
$\eta\lambda_{\min}$ of the error along the flattest one, so the number of
steps grows in proportion to the condition number
$\kappa = \lambda_{\max}/\lambda_{\min}$. On the valley, with $\kappa = 20$,
no learning rate reaches 1% in fewer than 47 steps. Critical points do less
harm: in high dimensions those at high loss are saddle points, which
gradient methods leave, and the plateaus of saturating activations are
removed by the design of the model. Convexity holds near a minimum, where
the loss is approximately a quadratic like the valley, and never globally.

## Exercises

1. **Critical points and plateaus.** Gradient descent slows or stops at
   each of the following points: the critical points of
   $f(x) = x\cos(\pi x)$ in the interior of $[-1, 2]$, the origin of
   $f(x) = x^3$, the origin of $f(x, y) = x^2 - y^2$, and $x = 4$ for
   $f(x) = \tanh x$.
    1. Compute the gradient and the Hessian at each point, locating the
       critical points of $x\cos(\pi x)$ numerically. Classify each point
       by the eigenvalue test of :numref:`subsec_opt-intro-saddles` as a
       local minimum, local maximum, saddle, inconclusive case, or
       non-critical point.
    1. Gradient descent with learning rate $\eta$ on $f(x) = x^3$ starts
       at $x_0 = \epsilon$ or at $x_0 = -\epsilon$ for a small
       $\epsilon > 0$. For each sign, determine whether the iterates
       approach $0$ or move away, and find the rate at which $x_t$ decays
       on the side that approaches. Explain why the result on strict
       saddles stated in :numref:`subsec_opt-intro-saddles` does not apply
       to this point.
    1. Using $1 - \tanh^2 x \approx 4e^{-2x}$ for large $x$, estimate how
       many steps gradient descent at $\eta = 1$ needs to move from $x = 4$
       to $x = 2$ on $f(x) = \tanh x$.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 6.6.*

1. [code] **Saddle escape time.** Balancing a ball on the saddle
   $f(x, y) = x^2 - y^2$ of :numref:`fig_mdl-opt-saddle` is stable along
   $x$ and unstable along $y$. Gradient descent with learning rate $\eta$
   started at $(1, \delta)$ behaves the same way.
    1. Derive the number of steps until $|y| \geq 1$ as a function of
       $\delta$ and $\eta$. Evaluate it for
       $\delta \in \{10^{-2}, 10^{-6}\}$ at $\eta = 0.1$, check both counts
       numerically, and state what happens for $\delta = 0$.
    1. Add zero-mean Gaussian noise of standard deviation $\sigma$ to every
       gradient evaluation and measure the escape time from $\delta = 0$
       for $\sigma \in \{10^{-3}, 10^{-1}\}$. Reconcile the result with
       the statement in :numref:`subsec_opt-intro-saddles` that gradient
       descent from a random start avoids strict saddles without noise.
   :exlabel:`ex_optimization-intro-saddle-escape-time`

1. [code] **An ill-conditioned valley.** Gradient descent on the valley
   :eqref:`eq_opt-intro-valley` multiplies $x_1$ by $1 - 0.2\eta$ and
   $x_2$ by $1 - 4\eta$ on every step, so it is stable for $\eta < 0.5$.
   The *step count* is the number of steps until both $|x_1|$ and $|x_2|$
   are below $1\%$ of their starting values.
    1. For $f(\mathbf{x}) = \frac{\lambda_{\min}}{2} x_1^2 +
       \frac{\lambda_{\max}}{2} x_2^2$, set $\eta = 1.8/\lambda_{\max}$,
       the same fraction of the stability limit $2/\lambda_{\max}$ as
       $\eta = 0.45$ is of $0.5$. Predict the step count from the two
       factors for $\kappa = \lambda_{\max}/\lambda_{\min} \in
       \{10, 100, 1000\}$, check it numerically, and compare it with the
       count $(\kappa/2)\log 100$ at the best learning rate.
    1. The section found the best rate by balancing the two factors. Prove
       that this is optimal: show that the learning rate minimizing the
       larger of $|1 - \eta\lambda_{\min}|$ and $|1 - \eta\lambda_{\max}|$
       is $\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$, that the
       factor it achieves is $(\kappa - 1)/(\kappa + 1)$, and that a
       direction with an eigenvalue between $\lambda_{\min}$ and
       $\lambda_{\max}$ shrinks at least as fast. Use the result to
       explain why tuning the learning rate cannot remove the dependence
       of the step count on $\kappa$.

1. [code] **Learning rate and sharpness.** A two-layer scalar linear
   network $\hat{y} = abx$ is fit to the single example $(x, y) = (1, 1)$
   with the loss $\ell(a, b) = \frac{1}{2}(ab - 1)^2$. Every point of the
   curve $ab = 1$ is a minimizer.
    1. Show that the Hessian at a minimizer $(a, b)$ has largest
       eigenvalue $\lambda_{\max} = a^2 + b^2$. Which minimizers are
       flattest?
    1. Run 200 steps of gradient descent from $(a, b) = (4, 0.1)$ at
       $\eta \in \{0.05, 0.1, 0.15, 0.2\}$. For each run, report the
       $\lambda_{\max}$ of the minimizer it reaches next to $2/\eta$.
    1. Explain the pattern with the stability condition
       $\eta < 2/\lambda_{\max}$.

1. **Permutation-symmetric minima.** ● Consider a multilayer perceptron with
   one hidden layer of $d \geq 2$ units and a single output, and let
   $\boldsymbol{\theta}^\star$ be a local minimum of its loss at which no
   two hidden units have identical incoming weights.
    1. Show that permuting the hidden units yields $d!$ distinct parameter
       vectors that all compute the same function and are therefore all
       local minima with the same loss.
    1. Suppose the loss were a convex function of the parameters. Show
       that the average of the $d!$ copies would then also be a global
       minimizer and that it computes the same function as a network with
       one hidden unit. Conclude that the loss is not convex whenever
       every network with one hidden unit has a higher loss than
       $\boldsymbol{\theta}^\star$.
    1. For the activation $\tanh$, identify a second symmetry of each
       hidden unit, state the resulting count of equivalent minima, and
       give a condition on $\boldsymbol{\theta}^\star$ under which all of
       them are distinct.
   :exlabel:`ex_optimization-intro-permutation-symmetric-minima`

1. **Eigenvalues of symmetric random matrices.** ● Let $\mathbf{M}$ be a
   random symmetric $k \times k$ matrix whose entries $M_{ij} = M_{ji}$,
   $i \leq j$, are drawn independently from distributions $p_{ij}$ with
   $p_{ij}(x) = p_{ij}(-x)$ :cite:`Wigner.1958`.
    1. Prove that an eigenvalue $\lambda$ drawn uniformly at random from
       the $k$ eigenvalues of $\mathbf{M}$ satisfies
       $P(\lambda > 0) = P(\lambda < 0)$.
    1. Explain why this symmetry alone does not determine the fraction of
       negative eigenvalues in a single realization of $\mathbf{M}$. State
       what the semicircle law of :citet:`Wigner.1958` says about that
       fraction for large $k$ when the off-diagonal entries are
       identically distributed with finite variance.
    1. The coin-flip heuristic of :numref:`subsec_opt-intro-saddles`
       treats the $k$ signs as independent, so all of them are positive
       with probability $2^{-k}$. For $k = 2$, draw the diagonal entries
       from $\mathcal{N}(0, 1)$ and the off-diagonal entry from
       $\mathcal{N}(0, 1/2)$. Compute the probability that both
       eigenvalues are positive, compare it with $2^{-2}$, and state what
       the gap says about the independence assumption.
   :exlabel:`ex_optimization-intro-eigenvalues-symmetric-random-matrices`

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

Landscapes<br>
**the goal of optimization · the ill-conditioned valley · the condition number · saddle points and plateaus · convexity as a local model**
:::
:::

::: {.slide title="One Learning Rate Must Serve Curvatures 0.2 and 4"}
[The problem]{.kicker}

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$. Gradient descent with
learning rate $\eta$ multiplies each coordinate by a fixed factor:

$$x_1 \leftarrow (1 - 0.2\eta)\, x_1, \qquad x_2 \leftarrow (1 - 4\eta)\, x_2.$$

- $\eta \geq 0.5$: the steep coordinate $x_2$ never shrinks.
- $\eta < 0.5$: the flat coordinate $x_1$ keeps more than 90% per step.

::: {.d2l-note}
The best $\eta$ still needs **47 steps** to get within 1% of the minimum.
A network's curvatures differ by orders of magnitude.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- say what an optimizer can and cannot fix about generalization;
- predict each coordinate's factor $1 - \eta\lambda$ from its curvature
  $\lambda$, and the largest stable learning rate;
- compute the best learning rate and the step count from the condition
  number;
- classify a critical point from the signs of its Hessian eigenvalues;
- explain why saddle points and plateaus rarely limit training.

**Running example:** the valley, curvatures 0.2 and 4, start $(-5, -2)$.
**The idea:** uneven curvature, not critical points, sets the speed of
training.
:::

::: {.slide title="Landscapes Opens the Chapter on a Valley We Can Solve by Hand"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | decks | question |
|---|---|---|
| a valley we can solve | **Landscapes** · Gradient Descent · Stochastic and Minibatch GD | what limits a step |
| two small networks | Two Test Models · Momentum · Adam · AdamW · Muon | which direction to step in |
| a whole run | Schedules · Batch Size · Scaling Up · Practice | how the step should change |
:::

::: {.col .narrow}
- **Uses:** gradients, the Hessian and its eigenvalues.
- **Skips:** gradient noise (the SGD deck) and proofs.
:::
:::
:::

::: {.slide title="The Optimizer Sees Only the Empirical Risk"}
[The goal]{.kicker}

- **Objective:** the empirical risk, the average loss on the training
  set, as a function of all parameters $\mathbf{x}$.
- **Learning wants:** a low risk, the expected loss on new data.
- The gap comes from sampling the data. No optimizer closes it.
- A network's objective has many minimizers. The learning rate and the
  gradient noise help choose among them.

::: {.d2l-note}
This deck asks how fast a method lowers the objective it is given.
:::
:::

::: {.slide title="The Steep Coordinate Flips Sign, the Flat One Creeps"}
[The valley]{.kicker}

@!optimization-intro-an-ill-conditioned-valley

$\eta = 0.45$, 30 steps from $(-5, -2)$. Predicted: $x_2$ times $-0.8$ per
step, $x_1$ times $0.91$, so $x_1$ ends near
$-5 \cdot 0.91^{30} \approx -0.3$. A factor of ten along $x_1$ takes about
24 steps.
:::

::: {.slide title="No Learning Rate Beats 47 Steps"}
[The valley]{.kicker}

Steps until both coordinates are within 1% of the minimum, from $(-5, -2)$:

| $\eta$ | 0.1 | 0.45 | 0.476 | 0.49 | 0.5 |
|---|---:|---:|---:|---:|---:|
| factor $x_1$ | 0.98 | 0.91 | 0.905 | 0.902 | 0.9 |
| factor $x_2$ | 0.6 | −0.8 | −0.905 | −0.96 | −1 |
| steps | 228 | 49 | 47 | 113 | never |

::: {.d2l-note}
Best: $1 - 0.2\eta = -(1 - 4\eta)$, so $\eta = 2/4.2 \approx 0.476$ and
both factors have size $0.905$.
:::
:::

::: {.slide title="Check: Which Coordinate Is Slow at η = 0.49?"}
[Check]{.kicker}

On the valley, gradient descent at $\eta = 0.49$ needs 113 steps, more
than twice as many as at $\eta = 0.45$. Which coordinate is to blame?

::: {.fragment .fade-up}
$x_2$: its factor is $1 - 4 \cdot 0.49 = -0.96$, larger in size than the
flat factor $0.902$. Close to the ceiling, a rate slightly too large costs
far more than one slightly too small.
:::
:::

::: {.slide title="Every Quadratic Is a Valley Along Its Eigenvectors"}
[The condition number]{.kicker}

$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{H}\mathbf{x}$ with
Hessian $\mathbf{H}$ (the matrix of second derivatives; for the valley
$\textrm{diag}(0.2, 4)$). One step of gradient descent:

$$\mathbf{x} \leftarrow (\mathbf{I} - \eta\mathbf{H})\,\mathbf{x}$$

Along an eigenvector $\mathbf{v}_i$ of $\mathbf{H}$ with eigenvalue
$\lambda_i$, the coordinate $c_i = \mathbf{v}_i^\top\mathbf{x}$ obeys

$$c_i \leftarrow (1 - \eta\lambda_i)\, c_i$$

::: {.d2l-note}
The largest eigenvalue sets the ceiling, $\eta < 2/\lambda_{\max}$. The
smallest, $\lambda_{\min}$, sets the pace.
:::
:::

::: {.slide title="Steps Grow in Proportion to the Condition Number"}
[The condition number]{.kicker}

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}, \qquad \textrm{valley: } \frac{4}{0.2} = 20$$

- Best rate $\eta = 2/(\lambda_{\min} + \lambda_{\max})$: both extreme
  factors have size $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$, here
  $19/21 = 0.905$.
- Error down to a fraction $\epsilon$: about $(\kappa/2)\ln(1/\epsilon)$
  steps. Valley: $10 \ln 100 \approx 46$, measured 47.

::: {.d2l-note}
No learning rate removes the factor $\kappa$. A faster method must change
the step itself.
:::
:::

::: {.slide title="Check: A Valley with Condition Number 1000"}
[Check]{.kicker}

Curvatures $0.004$ and $4$. With the best fixed learning rate, about how
many steps reduce the error by a factor of 100? And at $\eta = 0.45$?

::: {.fragment .fade-up}
- Best $\eta \approx 2/4.004$: about $(1000/2) \cdot \ln 100 \approx 2300$
  steps.
- $\eta = 0.45$: the flat factor is $1 - 0.0018$, so
  $\ln 100 / 0.0018 \approx 2560$ steps; the steep factor $-0.8$ is no
  bottleneck.
:::
:::

::: {.slide title="A Network Has No Finite κ, Only a Far Narrower Valley"}
[The condition number]{.kicker}

The Hessian of a network's loss (Ghorbani et al., 2019):

- a few large eigenvalues;
- a bulk near zero, some negative: $\lambda_{\max}/\lambda_{\min}$ is
  infinite or negative.

What carries over from the valley:

- the largest eigenvalue, the **sharpness**, limits $\eta$;
- a direction of curvature $\lambda$ loses a fraction $\eta\lambda$ of its
  error per step.

::: {.d2l-note}
The sharpness exceeds the typical curvature by orders of magnitude.
:::
:::

::: {.slide title="A Saddle Curves Up in One Direction and Down in Another"}
[Critical points]{.kicker}

![](../img/mdl-opt-saddle.svg){width=74%}

A **critical point** has zero gradient. At the origin of $z = x^2 - y^2$
it is a minimum along $x$ and a maximum along $y$: a **saddle point**.
:::

::: {.slide title="Hessian Eigenvalue Signs Classify a Critical Point"}
[Critical points]{.kicker}

| eigenvalues of the Hessian | critical point |
|---|---|
| all positive | strict local minimum |
| all negative | strict local maximum |
| some positive, some negative | saddle (strict) |
| zeros, all others one sign | inconclusive |

::: {.d2l-note}
Coin-flip signs, $k = 1000$ parameters: a minimum with probability
$2^{-1000} \approx 10^{-301}$. Real signs are not independent, but
critical points at high loss are saddles (Dauphin et al., 2014).
:::
:::

::: {.slide title="Check: Minimum, Saddle, or Inconclusive?"}
[Check]{.kicker}

The gradient is zero, and the Hessian is

1. $\textrm{diag}(2, 3)$
2. $\textrm{diag}(2, -1)$
3. $\textrm{diag}(0, 3)$

::: {.fragment .fade-up}
1. Strict local minimum: both curvatures positive.
2. Saddle: up along the first axis, down along the second.
3. Inconclusive: the flat direction is decided by higher-order terms
   ($x_1^4$ gives a minimum, $x_1^3$ does not).
:::
:::

::: {.slide title="Gradient Descent Leaves a Saddle Unless It Starts on the Ridge"}
[Critical points]{.kicker}

On $x^2 - y^2$, one step with learning rate $\eta$:

$$x \leftarrow (1 - 2\eta)\, x, \qquad y \leftarrow (1 + 2\eta)\, y$$

- $\eta = 0.1$: $y$ grows by $1.2$ per step. From $y = 10^{-3}$ it reaches
  1 in $\ln 10^3 / \ln 1.2 \approx 38$ steps.
- Only $y = 0$ stays, a set of measure zero (Lee et al., 2016).
- Minibatch noise pushes $y$ off zero and shortens the escape.

::: {.d2l-note}
High-loss critical points are saddles, and descent leaves saddles: a run
comes to rest only at low loss.
:::
:::

::: {.slide title="A Plateau Stalls Descent Without a Critical Point"}
[Plateaus]{.kicker}

![](../img/mdl-opt-tanh-flat.svg){width=56%}

$f(x) = \tanh x$ from $x = 4$: $f'(4) = 1 - \tanh^2 4 \approx 0.0013$.
Deep networks saturate the same way. ReLU and good initialization fix it
in the model, not in the optimizer.
:::

::: {.slide title="Convexity Describes the Neighborhood of a Minimum, Not the Landscape"}
[Convexity]{.kicker}

- **Convex:** the line between two points of the graph lies on or above
  it. No saddles, and every local minimum is global.
- **Locally true:** near a minimum a smooth loss is roughly a quadratic
  bowl, so the valley describes late training.
- **Globally false:** permuting $d$ hidden units gives $d!$ copies of
  every minimum, and their average is a network of identical units.

::: {.d2l-note}
A method that fails on a quadratic is unlikely to succeed on a
transformer.
:::
:::

::: {.slide title="Check: Multiply the Loss by 10. What Changes?"}
[Check]{.kicker}

The valley becomes $f(\mathbf{x}) = x_1^2 + 20 x_2^2$. What are the
ceiling, the best learning rate, the condition number, and the steps to
1%?

::: {.fragment .fade-up}
- Curvatures 2 and 40: ceiling $2/40 = 0.05$, best
  $\eta = 2/42 \approx 0.048$, ten times smaller.
- $\kappa = 40/2 = 20$ and 47 steps, both unchanged.

The learning rate absorbs the scale of the loss. The condition number is
what remains.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- The optimizer minimizes the empirical risk. It cannot close the gap to
  the risk, but it chooses among minimizers.
- The valley, curvatures 0.2 and 4: ceiling $\eta < 0.5$, best
  $\eta \approx 0.476$, never fewer than 47 steps to 1%.
- $\kappa = \lambda_{\max}/\lambda_{\min} = 20$: about
  $(\kappa/2)\ln(1/\epsilon)$ steps at the best rate.
- A network is a far narrower valley: its sharpness against a bulk of
  curvatures near zero.
- Saddles are left behind, plateaus are fixed in the model, and convexity
  holds only near a minimum.

**Next:** what limits the step on a general function, and the ideal step
when the curvature is known.
:::
