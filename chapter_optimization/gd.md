# Gradient Descent
:label:`sec_gd`

Gradient descent itself trains almost nothing in deep learning: every method
in this chapter replaces it with a stochastic variant. We begin with it anyway
because its failure modes carry over intact. A learning rate that diverges on
a one-dimensional parabola diverges on a billion-parameter transformer for the
same reason, and the cure for badly scaled coordinates — preconditioning —
reappears, in estimated form, inside the adaptive methods of
:numref:`sec_adam`. Of the three decisions framed in
:numref:`sec_optimization-intro`, this section isolates direction and step
size; noise enters in :numref:`sec_sgd`.

## One-Dimensional Gradient Descent

Why should stepping against the gradient reduce the objective at all? The
one-dimensional case already contains the answer. Consider a continuously
differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$. Its Taylor
expansion reads

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

For small $\epsilon$ the function is well approximated by its tangent line,
so moving against the derivative should decrease $f$. Pick a fixed step size
$\eta > 0$, choose $\epsilon = -\eta f'(x)$, and substitute:

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Unless the derivative vanishes, the first-order term $\eta f'^2(x) > 0$ pulls
the value down, and we can always choose $\eta$ small enough for the
higher-order remainder to stay negligible. Hence

$$f(x - \eta f'(x)) \lessapprox f(x),$$

and iterating

$$x \leftarrow x - \eta f'(x)$$

should drive the value of $f$ downhill until the gradient becomes small or we
run out of iterations. Everything interesting hides in the word "should": the
guarantee holds only while $\eta$ is small enough for :eqref:`gd-taylor` to be
trusted.

To watch the iteration at work we use $f(x)=x^2$. We know that $x=0$ is the
minimizer, which makes it easy to judge how the iterates behave.

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

```{.python .input #gd-one-dimensional-gradient-descent-2}
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Starting from $x=10$ with $\eta=0.2$, ten iterations bring $x$ close to the
optimum.

```{.python .input #gd-one-dimensional-gradient-descent-3}
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

The progress of optimizing over $x$ can be plotted as follows.

```{.python .input #gd-one-dimensional-gradient-descent-4}
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Learning Rate
:label:`subsec_gd-learningrate`

The learning rate $\eta$ is ours to choose, and both directions of error
cost us. Set it too small and $x$ barely moves: with $\eta = 0.05$, ten
steps leave us far from the solution, and reaching it would take many
more.

```{.python .input #gd-learning-rate-1}
show_trace(gd(0.05, f_grad), f)
```

Set it too large and the step $\left|\eta f'(x)\right|$ outruns the
first-order approximation: the remainder $\mathcal{O}(\eta^2 f'^2(x))$ in
:eqref:`gd-taylor-2` takes over and each update can *increase* the objective.
With $\eta=1.1$ the iterates overshoot the minimum on every step and
gradually diverge.

```{.python .input #gd-learning-rate-2}
show_trace(gd(1.1, f_grad), f)
```

### Local Minima

On a convex parabola the only risk was the step size. Nonconvex functions add
another. Consider $f(x) = x \cdot \cos(cx)$, which has infinitely many local
minima. Depending on the learning rate and on how well conditioned the problem
is, gradient descent settles into one of many solutions — and an
(unrealistically) large learning rate can bounce the iterate into a poor one.

```{.python .input #gd-local-minima}
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Multivariate Gradient Descent

Now consider the general case, $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$,
where the objective $f: \mathbb{R}^d \to \mathbb{R}$ maps vectors to scalars.
Its gradient is the vector of $d$ partial derivatives:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Each entry $\partial f(\mathbf{x})/\partial x_i$ measures the rate of change
of $f$ with respect to input $x_i$. The multivariate Taylor expansion mirrors
the one-dimensional one:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Up to second-order terms, the direction of steepest descent is the negative
gradient $-\nabla f(\mathbf{x})$, and with a suitable learning rate
$\eta > 0$ we obtain the prototypical gradient descent algorithm:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

To see the algorithm in action, take the quadratic
$f(\mathbf{x})=x_1^2+2x_2^2$ with gradient
$\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ — a bowl that curves twice as
steeply in $x_2$ as in $x_1$ — and track the trajectory of $\mathbf{x}$ from
the initial position $[-5, -2]$. We need two helper functions: the first
applies an update rule repeatedly from the fixed starting point, the second
draws the trajectory over a contour plot of the objective. Both will be
reused throughout this chapter.

```{.python .input #gd-multivariate-gradient-descent-1}
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables used by the stateful
    # optimizers (momentum, Adam) later in this chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input #gd-multivariate-gradient-descent-2}
%%tab pytorch
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input #gd-multivariate-gradient-descent-2}
%%tab jax
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

With learning rate $\eta = 0.1$, twenty steps bring $\mathbf{x}$ near its
minimum at $[0, 0]$. Progress is well behaved but slow — and notice the shape
of the path: the trajectory bends because the two coordinates want different
step sizes, a first glimpse of the conditioning problem that
:numref:`sec_momentum` takes up in earnest.

```{.python .input #gd-multivariate-gradient-descent-3}
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Newton's Method

As :numref:`subsec_gd-learningrate` showed, getting the learning rate "just
right" is tricky, and the multivariate demo made things worse: the best rate
differs per coordinate. What if the objective itself told us how far to step?
Methods that consult the *curvature* of the objective — its second
derivatives — do exactly that. They cannot be applied to deep networks
directly, for reasons of cost we quantify below, but they define the ideal
that the practical algorithms later in this chapter approximate.

There was no need to stop the Taylor expansion of
$f: \mathbb{R}^d \rightarrow \mathbb{R}$ after the first term. We can write
it as

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Define $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$, the
Hessian of $f$, a $d \times d$ matrix. For small $d$ and simple problems
$\mathbf{H}$ is easy to compute; for a deep network it is prohibitively large,
with $\mathcal{O}(d^2)$ entries. For now, set that aside and see what
algorithm the expansion suggests.

The minimum of $f$ satisfies $\nabla f = 0$. Taking derivatives of
:eqref:`gd-hot-taylor` with regard to $\boldsymbol{\epsilon}$ (following the
calculus rules of :numref:`subsec_calculus-grad`) and ignoring higher-order
terms we arrive at

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Newton's method is gradient descent with the gradient premultiplied by the
inverse Hessian — the step size problem solved by the objective itself. As a
simple example, for $f(x) = \frac{1}{2} x^2$ we have $\nabla f(x) = x$ and
$\mathbf{H} = 1$, so for any $x$ the update is $\epsilon = -x$: a *single*
step converges perfectly, with no learning rate to tune. We got a bit lucky
here: the Taylor expansion of this $f$ was exact.

Let's see what happens in other problems. Given a convex hyperbolic cosine
function $f(x) = \cosh(cx)$ for some constant $c$, the global minimum at
$x=0$ is reached after a few iterations.

```{.python .input #gd-newton-s-method-1}
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print(f'epoch 10, x: {float(x):f}')
    return results

show_trace(newton(), f)
```

Now let's consider a *nonconvex* function, such as $f(x) = x \cos(c x)$.
Newton's method divides by the Hessian, so wherever the second derivative is
*negative* the update walks toward *increasing* values of $f$ — toward a
maximum. That is a fatal flaw of the algorithm. Let's see what happens in
practice.

```{.python .input #gd-newton-s-method-2}
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

This went spectacularly wrong. How can we fix it? One option is to "repair"
the Hessian by taking its absolute value. Another is to bring back the
learning rate. This seems to defeat the purpose, but not quite: second-order
information still lets us be cautious where curvature is large and take
longer steps where the objective is flat. With a slightly smaller learning
rate, $\eta = 0.5$, the damped iteration converges quickly.

```{.python .input #gd-newton-s-method-3}
show_trace(newton(0.5), f)
```

Two facts about Newton's method are worth carrying away, and both are proved
in the appendix rather than here. First, near a minimum with positive
curvature it converges *quadratically*: the number of correct digits roughly
doubles at every iteration. :numref:`subsec_mdl-why-not-newton` gives the
proof and shows the doubling numerically. Second, nothing rescues the method
at deep-learning scale. Storing the Hessian costs $\mathcal{O}(d^2)$ memory
and solving with it $\mathcal{O}(d^3)$ time — at $d \sim 10^9$ parameters,
exabytes before the first step — and, as the demo above showed in one
dimension, a nonconvex objective hands Newton negative curvature that it
follows toward saddle points and maxima. The classical remedies are cheaper
curvature estimates and safer step rules: quasi-Newton methods such as BFGS
rebuild curvature from successive gradient differences
(:numref:`subsec_mdl-quasi-newton`), line search picks $\eta$ by trial at run
time (:numref:`subsec_mdl-gd-smoothness`), and trust regions bound the step
instead of the rate (:numref:`subsec_mdl-trust-region`)
:cite:`Boyd.Vandenberghe.2004,Nocedal.Wright.2006`. None of them fit deep
learning as-is; a single line-search trial, for instance, evaluates the
objective on the entire dataset.

### Preconditioning

What survives at scale is the underlying idea. Instead of inverting the full
Hessian, rescale the update by a cheap approximation of it — a
*preconditioner*. The cheapest useful choice is the diagonal:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

This amounts to selecting a separate learning rate for every coordinate. To
see why that matters, imagine a model with one parameter measured in
millimeters and another in kilometers. Both natural scales are meters, so the
two gradients differ by orders of magnitude for no meaningful reason; a
single global $\eta$ must fit both, so it fits neither. Preconditioning
removes the mismatch without our ever finding it by hand. This idea drives
much of what follows: diagonal preconditioners estimated from gradients
rather than second derivatives are the core of AdaGrad and Adam
(:numref:`sec_adam`), and preconditioning whole weight *matrices* rather than
individual coordinates leads to Muon (:numref:`sec_muon`).

## Summary

Gradient descent decreases a differentiable function by stepping against the
gradient, and the guarantee holds only while the step is small enough for the
first-order Taylor expansion to be trusted. That proviso carries all the
trouble: a learning rate chosen too small wastes iterations, one too large
overshoots or diverges, and on nonconvex objectives even a well-chosen rate
merely reaches *some* local minimum. Newton's method removes the step-size
problem by dividing out the curvature — the ideal preconditioner — and
converges in one step on a quadratic, but its $\mathcal{O}(d^2)$ cost and its
attraction to saddle points rule it out for deep networks. The rest of this
chapter builds cheap, gradient-estimated stand-ins for that ideal.

## Exercises

1. Experiment with different learning rates and objective functions for gradient descent.
1. Implement line search to minimize a convex function in the interval $[a, b]$.
    1. Do you need derivatives for binary search, i.e., to decide whether to pick $[a, (a+b)/2]$ or $[(a+b)/2, b]$.
    1. How rapid is the rate of convergence for the algorithm?
    1. Implement the algorithm and apply it to minimizing $\log (\exp(x) + \exp(-2x -3))$.
1. Design an objective function defined on $\mathbb{R}^2$ where gradient descent is exceedingly slow. Hint: scale different coordinates differently.
1. Implement the lightweight version of Newton's method using preconditioning:
    1. Use diagonal Hessian as preconditioner.
    1. Use the absolute values of that rather than the actual (possibly signed) values.
    1. Apply this to the problem above.
1. Apply the algorithm above to a number of objective functions (convex or not). What happens if you rotate coordinates by $45$ degrees?

[Discussions](https://d2l.discourse.group/t/351)

<!-- slides -->

::: {.slide title="Gradient Descent"}
Plain gradient descent isn't what trains deep nets — SGD
and its descendants do — but every issue those methods
hit shows up here first, in cleaner form: LR sensitivity,
divergence, local minima, poor conditioning, second-order
corrections.

The rule:

$$x \leftarrow x - \eta \nabla f(x).$$

A first-order Taylor expansion shows that for small enough
$\eta$, this decreases $f$ locally. The art is picking
$\eta$.
:::

::: {.slide title="1D demo: $f(x) = x^2$"}
Setup and define $f$, $f'$:

@gd-one-dimensional-gradient-descent-1

. . .

@gd-one-dimensional-gradient-descent-2
:::

::: {.slide title="GD iteration"}
Start at $x = 10$, $\eta = 0.2$, 10 steps. Converges to 0:

@gd-one-dimensional-gradient-descent-3

. . .

@gd-one-dimensional-gradient-descent-4
:::

::: {.slide title="Learning rate too small"}
$\eta = 0.05$: takes forever to converge:

@gd-learning-rate-1
:::

::: {.slide title="Learning rate too big"}
$\eta = 1.1$: the $\mathcal{O}(\eta^2 f'^2)$ Taylor remainder
dominates and the iterates diverge:

@gd-learning-rate-2
:::

::: {.slide title="Non-convex: trapped in a local min"}
$f(x) = x \cos(cx)$ has infinitely many local minima. Even
with a moderately large learning rate, GD ends up in
whichever basin it falls into:

@gd-local-minima
:::

::: {.slide title="Multivariate GD"}
Same rule on vectors:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Demo on $f(x_1, x_2) = x_1^2 + 2 x_2^2$ — anisotropic,
$x_2$ direction is steeper.

@gd-multivariate-gradient-descent-1

. . .

@gd-multivariate-gradient-descent-2
:::

::: {.slide title="Run it"}
@gd-multivariate-gradient-descent-3

. . .

The path bends: the two coordinates want *different* step
sizes. One global $\eta$ can't satisfy both.
:::

::: {.slide title="Newton's method: second-order"}
Use the Hessian to set the step size automatically. From
the second-order Taylor expansion:

$$\mathbf{x} \leftarrow \mathbf{x} - [\nabla^2 f(\mathbf{x})]^{-1} \nabla f(\mathbf{x}).$$

For $f(x) = \cosh(cx)$, a few steps find the minimum — no
learning rate to tune:

@gd-newton-s-method-1
:::

::: {.slide title="Newton fails on non-convex"}
$f(x) = x \cos(cx)$: Newton divides by the second
derivative, so negative curvature sends it *uphill*,
toward a maximum. Damping ($\eta = 0.5$) restores sanity:

@gd-newton-s-method-2

. . .

@gd-newton-s-method-3
:::

::: {.slide title="Preconditioning: the idea that scales"}
Full Newton at $d \sim 10^9$: $\mathcal{O}(d^2)$ memory,
$\mathcal{O}(d^3)$ solve — exabytes before the first step.

What survives: rescale each update by a *cheap approximation*
of curvature.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta\, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x})$$

= a separate learning rate per coordinate (fixes the
millimeters-vs-kilometers mismatch automatically).

::: {.d2l-note}
Diagonal preconditioners estimated from *gradients* → AdaGrad,
Adam. Per-*matrix* preconditioning → Muon. Both later in this
chapter.
:::
:::

::: {.slide title="Recap"}
- GD update: $x \leftarrow x - \eta \nabla f(x)$.
- Learning rate too small → slow; too large → diverge.
- Local minima trap plain GD on non-convex objectives.
- Newton uses the Hessian as the *ideal* preconditioner —
  one step on quadratics, but $\mathcal{O}(d^2)$ memory and
  unsafe under negative curvature.
- What deep learning keeps is cheap preconditioning:
  per-coordinate (Adam) and per-matrix (Muon) rescaling.
:::
