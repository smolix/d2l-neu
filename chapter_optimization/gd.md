# Gradient Descent
:label:`sec_gd`

Gradient descent is the simplest optimizer and the ancestor of every other
method in this chapter: compute the gradient, take a step against it, and
repeat. It has a single knob, the learning rate $\eta$, and on the valley
$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of :numref:`sec_optimization-intro`
this knob already decides everything. At $\eta = 0.45$ the iterate needs 49
steps to come within 1% of the minimum, at $\eta = 0.5$ it never arrives,
and no learning rate does better than 47 steps. A learning rate that makes
a large network diverge does so for the same reason that it makes a
one-dimensional quadratic diverge, so the small cases are worth
understanding completely.

This section answers three questions. Why does a step against the gradient
lower the objective, and how large may the step be? What would the ideal
step be if we also knew the curvature? And how can we approximate that
ideal cheaply? The answers are a Taylor expansion, Newton's method, and
preconditioning. Preconditioning is the idea behind the adaptive methods of
:numref:`sec_adam`. Throughout the section the gradient is exact;
:numref:`sec_sgd` replaces it with a noisy estimate.

## One-Dimensional Gradient Descent

Why should a step against the gradient reduce the objective at all? The
one-dimensional case already contains the answer. Let
$f: \mathbb{R} \rightarrow \mathbb{R}$ be continuously differentiable. For
a small change $\epsilon$ in its argument, its Taylor expansion reads

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Near $x$ the function is close to its tangent line, so a move downhill
along the tangent should decrease $f$. We pick a step size $\eta > 0$, set
$\epsilon = -\eta f'(x)$, and substitute:

$$f(x - \eta f'(x)) = f(x) - \eta f'(x)^2 + \mathcal{O}(\eta^2 f'(x)^2).$$
:eqlabel:`gd-taylor-2`

The first-order change $-\eta f'(x)^2$ is negative whenever the derivative
is not zero, and for small enough $\eta$ it outweighs the remainder.
Iterating the update

$$x \leftarrow x - \eta f'(x)$$

therefore decreases $f$ as long as the step is small. How small depends on
how fast the slope changes. If $f'$ is $L$-Lipschitz, that is, if the slope
changes by at most $L$ per unit of $x$, then every learning rate with
$0 < \eta < 2/L$ decreases $f$; :numref:`chap_mdl-optimization` proves this.

The quadratic $f(x) = x^2$ shows all of this in closed form. Its derivative
is $2x$, so one step maps $x$ to $x - 2\eta x$:

$$x \leftarrow (1 - 2\eta)\, x.$$

On a quadratic, gradient descent multiplies the distance to the minimum by
a fixed factor at every step. The iterates converge if and only if
$|1 - 2\eta| < 1$, that is, $0 < \eta < 1$, which is the bound $2/L$ with
$L = 2$. With $\eta = 0.2$ the factor is $0.6$, and ten steps from $x = 10$
should leave $10 \cdot 0.6^{10} \approx 0.06$. We check this numerically.

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

```{.python .input #gd-one-dimensional-gradient-descent-2}
%%tab pytorch, jax
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Its derivative
    return 2 * x
```

```{.python .input #gd-one-dimensional-gradient-descent-3}
%%tab pytorch, jax
def gd(eta, f_grad, steps=10):
    """Gradient descent from x = 10; returns all iterates."""
    x = 10.0
    results = [x]
    for _ in range(steps):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'step {steps}, x: {float(x):f}')
    return results

results = gd(0.2, f_grad)
```

The iterates approach the minimum geometrically, each step keeping 60% of
the distance. We plot them on the graph of $f$.

```{.python .input #gd-one-dimensional-gradient-descent-4}
%%tab pytorch, jax
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [f(f_line), [f(x) for x in results]],
             '$x$', '$f(x)$', fmts=['-', '-o'])

show_trace(results, f)
```

### Learning Rate
:label:`subsec_gd-learningrate`

The learning rate is ours to choose, and the choice can go wrong in either
direction. With $\eta = 0.05$ the factor is $0.9$, and ten steps leave
$10 \cdot 0.9^{10} \approx 3.5$: the iterate barely moves, and reaching the
minimum would take many more steps.

```{.python .input #gd-learning-rate-1}
%%tab pytorch, jax
show_trace(gd(0.05, f_grad), f)
```

With $\eta = 1.1$ the factor is $-1.2$. Each step overshoots the minimum and
lands farther away on the other side, so after ten steps the distance has
grown to $10 \cdot 1.2^{10} \approx 62$. In the language of
:eqref:`gd-taylor-2`, the step $|\eta f'(x)|$ is too long for the tangent
line to be a good guide, and the remainder term takes over.

```{.python .input #gd-learning-rate-2}
%%tab pytorch, jax
show_trace(gd(1.1, f_grad), f)
```

The best learning rate for this quadratic is $\eta = 0.5$: the factor is
zero, and a single step lands on the minimum. That rate is the inverse of
the curvature $f''(x) = 2$, an observation that returns with Newton's
method below.

### Local Minima
:label:`subsec_gd-local-minima`

On a convex parabola the only risk is the step size. A nonconvex function
adds another: where the iterate ends up can depend on the learning rate.
Consider $f(x) = x \cos(cx)$ with $c = 0.15\pi$. From $x = 10$ the deep
basin around $x \approx 7.3$ is close by, and a shallow one lies near
$x \approx -1.8$. We run gradient descent with $\eta = 0.5$ and with
$\eta = 2$, and plot the second run.

```{.python .input #gd-local-minima}
%%tab pytorch, jax
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Its derivative
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

gd(0.5, f_grad)  # A moderate learning rate
show_trace(gd(2, f_grad), f)
```

With $\eta = 0.5$ the iterate rolls into the deep basin and stops near
$7.27$. With $\eta = 2$ the first step jumps over that basin to
$x \approx 0.6$, and the iterate ends up bouncing around the shallow
minimum. A learning rate this large is unrealistic for a one-dimensional
problem, but large steps in the early phase of training play the same role
in deep networks: they decide which region of the surface training
explores.

## Multivariate Gradient Descent

The same update works for a vector of parameters
$\mathbf{x} = (x_1, \ldots, x_d)$ and an objective
$f: \mathbb{R}^d \to \mathbb{R}$. The gradient collects the $d$ partial
derivatives,

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f(\mathbf{x})}{\partial x_1}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\right),$$

and each entry measures how fast $f$ changes when its coordinate alone
changes. The multivariate Taylor expansion has the same form as the
one-dimensional one,

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Among all steps $\boldsymbol{\epsilon}$ of a fixed small length $r$, which
one decreases $f$ the most? To first order the change is
$\boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$, and by the
Cauchy–Schwarz inequality this inner product is at least
$-r \|\nabla f(\mathbf{x})\|$, with equality exactly when
$\boldsymbol{\epsilon}$ points along $-\nabla f(\mathbf{x})$. The negative
gradient is therefore the direction of *steepest descent*, and with a
learning rate $\eta > 0$ we obtain gradient descent:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

We return to the valley, $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$, with
curvatures $h_1 = 0.2$ along $x_1$ and $h_2 = 4$ along $x_2$. These are the
eigenvalues $\lambda_{\min}$ and $\lambda_{\max}$ of its Hessian in
:numref:`sec_optimization-intro`, and the gradient is
$(h_1 x_1, h_2 x_2) = (0.2 x_1, 4 x_2)$. To follow the iterates in the plane we write
two helpers that the rest of the chapter reuses. `train_2d` starts at
$(-5, -2)$ and applies an update rule for a given number of steps. Besides
the two coordinates, the update rule receives and returns two state
variables, which the methods with memory in later sections (momentum, Adam)
use; gradient descent ignores them. Hyperparameters such as the learning
rate are passed through as keyword arguments. `show_trace_2d` draws the
path over the contour lines of the objective.

```{.python .input #gd-multivariate-gradient-descent-1}
%%tab pytorch, jax
def train_2d(trainer, steps=20, f_grad=None, **hparams):  #@save
    """Run a 2D update rule from (-5, -2) and return the iterates."""
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1, s2: state of stateful optimizers
    results = [(x1, x2)]
    for _ in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad, **hparams)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, **hparams)
        results.append((x1, x2))
    print(f'step {steps}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input #gd-multivariate-gradient-descent-2}
%%tab pytorch, jax
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables over the contours of f."""
    x1s, x2s = zip(*[(float(a), float(b)) for a, b in results])
    grids = []  # A window around start and minimum, widened to fit the trace
    for (lo, hi), xs in zip([(-5.5, 1.0), (-3.0, 3.0)], (x1s, x2s)):
        lo, hi = min(lo, min(xs) - 0.5), max(hi, max(xs) + 0.5)
        grids.append(d2l.arange(lo, hi, (hi - lo) / 100))
    x1, x2 = d2l.meshgrid(*grids, indexing='ij')
    width, height = (float(g[-1] - g[0]) for g in grids)
    equal = 0.25 < height / width < 4  # Equal scales unless the trace diverged
    d2l.set_figsize((3, 2.8) if equal else (3.5, 2.5))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='0.6', linewidths=0.8)
    d2l.plt.plot(x1s, x2s, '-o', color='C1', markersize=4)
    d2l.plt.xlabel('$x_1$')
    d2l.plt.ylabel('$x_2$')
    if equal:
        d2l.plt.gca().set_aspect('equal')
```

We run the same 30 steps at $\eta = 0.45$ as in
:numref:`sec_optimization-intro`, now drawn in the plane.

```{.python .input #gd-multivariate-gradient-descent-3}
%%tab pytorch, jax
h1, h2 = 0.2, 4.0  # Curvatures of the valley along x1 and x2

def f_valley(x1, x2):
    return h1 / 2 * x1 ** 2 + h2 / 2 * x2 ** 2

def valley_grad(x1, x2):
    return h1 * x1, h2 * x2

def gd_2d(x1, x2, s1, s2, f_grad, eta):
    g1, g2 = f_grad(x1, x2)
    return x1 - eta * g1, x2 - eta * g2, 0, 0

show_trace_2d(f_valley, train_2d(gd_2d, steps=30, f_grad=valley_grad,
                                 eta=0.45))
```

Each step is perpendicular to the contour line through the current point,
as steepest descent must be. On a narrow valley that direction points
mostly across the valley: from $(-5, -2)$ the gradient is $(-1, -8)$, eight
times steeper across than along. The iterate therefore zigzags from wall
to wall and makes slow progress along the floor, and after 30 steps $x_1$
is still $-0.3$. A larger $\eta$ would speed up progress along the floor
but make the oscillation across the valley unstable. What the valley needs is a different learning rate in each
direction.

## Newton's Method
:label:`subsec_gd-newton`

On the quadratic $x^2$ the best learning rate was the inverse of the
curvature. The valley has two curvatures, $0.2$ along $x_1$ and $4$ along
$x_2$, so a learning rate of $1/0.2 = 5$ along $x_1$ and $1/4$ along $x_2$
would solve each coordinate in a single step. Newton's method makes this
idea general: it measures the curvature in every direction and divides the
gradient by it.

The curvature enters through the second-order term of the Taylor
expansion,

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3),$$
:eqlabel:`gd-hot-taylor`

where $\mathbf{H} = \nabla^2 f(\mathbf{x})$ is the *Hessian*, the
$d \times d$ matrix of second derivatives. Ignoring the remainder, the
right-hand side is a quadratic in $\boldsymbol{\epsilon}$. Setting its
gradient with respect to $\boldsymbol{\epsilon}$ to zero (with the rules of
:numref:`subsec_calculus-grad`) gives the step that minimizes it:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0, \quad \textrm{hence} \quad \boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Newton's method is gradient descent with the gradient multiplied by the
inverse Hessian. Each direction receives the step its own curvature calls
for, and there is no learning rate to tune. When $f$ is itself a quadratic,
the expansion is exact and a single Newton step lands on the minimum. On
the valley, $\mathbf{H} = \textrm{diag}(0.2, 4)$ and
$\mathbf{H}^{-1} \nabla f(\mathbf{x}) = \mathbf{x}$, so one step from any
point reaches the origin, whatever the condition number.

On functions that are not quadratic, the method has to iterate. The convex
function $f(x) = \cosh(cx)$ with $c = 0.5$ has its minimum at $x = 0$, and
we run ten Newton steps from $x = 10$.

```{.python .input #gd-newton-s-method-1}
%%tab pytorch, jax
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # First derivative
    return c * d2l.sinh(c * x)

def f_hess(x):  # Second derivative
    return c**2 * d2l.cosh(c * x)

def newton(eta=1, steps=10):
    """Newton's method from x = 10, with an optional step size eta."""
    x = 10.0
    results = [x]
    for _ in range(steps):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print(f'step {steps}, x: {float(x):f}')
    return results

show_trace(newton(), f)
```

The iterate reaches the minimum to machine precision within ten steps, with
no learning rate chosen by hand.

Newton's method divides by the curvature, and the curvature of a nonconvex
function can be negative. Where it is, the step points uphill, toward a
maximum. We return to $f(x) = x \cos(cx)$ with $c = 0.15\pi$, which has a
minimum near $7.27$, close to the start.

```{.python .input #gd-newton-s-method-2}
%%tab pytorch, jax
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # First derivative
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Second derivative
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

The first step overshoots to $x = 5$. The curvature there is positive but
small, so the next step divides a large slope by a small number and jumps
to $x \approx 25$, where the curvature is negative. From there the
iteration climbs to the local maximum near $26.8$, a point where the
gradient vanishes as it does at a minimum. One remedy is to
replace the Hessian by its absolute value, so that every step points
downhill. Another is to bring back a learning rate. With $\eta = 0.5$ the
first step stops at $x = 7.5$, inside the basin around the minimum, where
the curvature is positive, and the iteration converges. It converges only
linearly, however: each step halves the distance to $7.27$. Damping makes
the method safe here, and in exchange it gives up Newton's fast local rate.

```{.python .input #gd-newton-s-method-3}
%%tab pytorch, jax
show_trace(newton(0.5), f)
```

Damping is no guarantee: it only makes a jump into a region of negative
curvature less likely. Two further facts about Newton's method matter
here, and the appendix proves both. First, near a minimum with positive
curvature it converges at least *quadratically*: the number of correct
digits roughly doubles with every step
(:numref:`subsec_mdl-why-not-newton`). Second, the method does not scale to
deep learning.

Storing the Hessian of a model with $d$ parameters takes
$\mathcal{O}(d^2)$ memory and solving a linear system with it takes
$\mathcal{O}(d^3)$ time. For $d = 10^9$ the Hessian has $10^{18}$ entries,
four exabytes in single precision, before the first step. And a deep
network's Hessian has negative eigenvalues, which steer the raw Newton step
toward saddles and maxima. Classical optimization lowers these costs with
curvature estimates built from successive gradients and with safer step
rules (:numref:`subsec_mdl-quasi-newton`). Deep learning keeps a simpler
part of the idea: rescale the gradient, direction by direction, with an
inexpensive estimate of the curvature.

## Preconditioning
:label:`subsec_gd-preconditioning`

A *preconditioner* rescales the gradient before the step. The simplest
useful one keeps only the diagonal of the Hessian,

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}),$$
:eqlabel:`eq_gd-diag-precond`

which divides each coordinate of the gradient by the curvature along that
coordinate. In effect, every coordinate receives its own learning rate.
Storing the diagonal takes as much memory as storing the gradient.

Ill-conditioning of this axis-aligned kind arises from something as
mundane as units. Suppose a linear regression model receives two unrelated
length features of similar size, one recorded in meters and the other in
millimeters. The curvature of the squared loss along a weight is the mean
of the squared feature, so the millimeter weight has a curvature $10^6$
times larger than the meter weight. The units alone give a condition
number of at least $10^6$. A single learning rate small enough for the
millimeter weight leaves the meter weight almost frozen. Dividing by the diagonal curvature
removes the factor $10^6$ without our ever noticing the units.

On the valley, the diagonal of the Hessian is $(0.2, 4)$, and we run the
preconditioned update at $\eta = 0.5$.

```{.python .input #gd-preconditioning}
%%tab pytorch, jax
def precond_gd_2d(x1, x2, s1, s2, f_grad, eta):
    g1, g2 = f_grad(x1, x2)
    return x1 - eta * g1 / h1, x2 - eta * g2 / h2, 0, 0  # diag(H) = (h1, h2)

show_trace_2d(f_valley, train_2d(precond_gd_2d, steps=10,
                                 f_grad=valley_grad, eta=0.5))
```

Both coordinates now shrink by the same factor $1 - \eta = 0.5$ per step,
so the path runs straight to the minimum. Seven steps bring both within 1%,
against 47 for the best plain learning rate, and the count no longer
depends on the condition number. At $\eta = 1$ one step suffices: because
the Hessian of the valley is diagonal, the diagonal preconditioner is
Newton's method.

Two limitations shape what comes next. A deep network does not give us its
Hessian, not even the diagonal, so practical methods estimate a diagonal
scale from the gradients they observe. AdaGrad, RMSProp, and Adam
(:numref:`sec_adam`) differ mainly in how they form that estimate. And a
diagonal preconditioner only helps when the steep and flat directions line
up with the coordinate axes. Exercise :exref:`ex_gd-diagonal-preconditioning`
rotates the valley and shows the preconditioner failing. For this reason
later methods precondition whole weight matrices rather than single
coordinates.

With an exact gradient, then, the step size and the curvature explain
everything we have seen. But an exact gradient of a training loss requires
a pass over the whole dataset. The next section takes up this problem.

## Summary

Gradient descent decreases a differentiable function by stepping against
the gradient, and the decrease is guaranteed only while the step is short
enough for the first-order Taylor expansion to hold. On a quadratic each
step multiplies the error by a fixed factor: a learning rate that is too
small makes the factor close to one, and one that is too large makes it
exceed one in size and the iterates diverge. On nonconvex functions the
learning rate also decides which minimum the iterate reaches. Newton's
method divides the gradient by the curvature and solves a quadratic in one
step, whatever its condition number, but its cost and its attraction to
maxima and saddles rule it out for deep networks. Preconditioning keeps the
useful part: a diagonal rescaling solved the valley in seven steps instead
of 47, and the adaptive methods of later sections estimate such a rescaling
from gradients.

## Exercises

1. [code] **Learning-rate sweep.** Extend `gd` to run at most 100 steps
   and to stop as soon as $|x| < 10^{-3}$. Run it on $f(x) = x^2$,
   $f(x) = |x|^{1.5}$, and $f(x) = x^4$ at
   $\eta \in \{0.01, 0.1, 0.5, 0.9, 1.1\}$.
    1. Tabulate, for each of the fifteen runs, the number of steps until
       the stopping test holds, or, for a run that misses the test within
       100 steps, whether $|x|$ grew without bound or stayed bounded.
    1. The second derivative of $|x|^{1.5}$ is unbounded near the origin.
       Explain what the iterates do close to $0$ at any fixed $\eta$, and
       why the stopping test is reached late or not at all.
    1. The second derivative of $x^4$ vanishes at the origin. Show that
       once the iterates are small, $x_{t+1} \approx x_t (1 - 4\eta x_t^2)$,
       and that $|x_t|$ then decays like $1/\sqrt{8 \eta t}$ rather than
       geometrically. Why can no fixed learning rate restore a geometric
       rate on this function?

1. [code] **Diagonal preconditioning under rotation.** Rotate the valley
   by an angle $\theta$: $f_\theta(\mathbf{x}) = \frac{1}{2}
   \mathbf{x}^\top \mathbf{R}_\theta \textrm{diag}(0.2, 4)
   \mathbf{R}_\theta^\top \mathbf{x}$, where $\mathbf{R}_\theta$ is the
   rotation matrix by $\theta$.
    1. For $\theta = 45^\circ$, run plain gradient descent at
       $\eta = 0.45$, the diagonal preconditioner
       :eqref:`eq_gd-diag-precond` at $\eta = 0.5$, and Newton's method
       with `d2l.train_2d`. Which of the three behave as on the unrotated
       valley? Explain the behavior of the diagonal preconditioner from
       the entries of the rotated Hessian.
    1. The preconditioned update behaves like plain gradient descent on a
       quadratic with Hessian
       $\mathbf{D}^{-1/2} \mathbf{H}_\theta \mathbf{D}^{-1/2}$, where
       $\mathbf{D} = \textrm{diag}(\mathbf{H}_\theta)$. Compute its
       condition number as a function of $\theta$, and check that it is $1$
       at $\theta = 0$ and $20$ at $\theta = 45^\circ$.
    1. A deep network's Hessian is not diagonal in its parameter
       coordinates. What does this exercise suggest about how much a
       diagonal preconditioner can help, and what would a preconditioner
       need in order to undo a rotation?
   :exlabel:`ex_gd-diagonal-preconditioning`

1. [code] **Absolute-curvature preconditioning.** Replace
   $\textrm{diag}(\mathbf{H})^{-1}$ in :eqref:`eq_gd-diag-precond` by
   $|\textrm{diag}(\mathbf{H})|^{-1}$ and run this variant at $\eta = 1$ on
   the nonconvex $f(x_1, x_2) = x_1 \cos(c x_1) + x_2 \cos(c x_2)$ with
   $c = 0.15\pi$ from the starting point of `d2l.train_2d`. Show that every
   update direction is a descent direction. Does every step decrease $f$?
   Explain the size of the largest step of the run.

1. **Certifying the lowest minimum.** ● Modify the objective
   $f(x) = x\cos(cx)$ with $c = 0.15\pi$ of :numref:`subsec_gd-local-minima`
   so that its local minima in $(-20, 20)$ differ in value by at most
   $10^{-3}$ except for one. Explain why any method that uses only $f$ and
   $f'$ at the points it visits must then visit every basin before it can
   certify which local minimum is lowest.

1. [code] **Newton's convergence order.** ● In :numref:`subsec_gd-newton`,
   `newton` converges in a few steps on $f(x) = \cosh(cx)$. Run it with
   $c = 0.5$ from $x_0 = 10$ and record $|x_t|$ after every step.
    1. From the last steps before $x_t$ reaches $0$ in floating point,
       estimate the order $p$ in $|x_{t+1}| \approx C |x_t|^p$. Where does
       this fast phase begin, and what governs the steps before it?
    1. Derive $p$ and $C$ from the Taylor expansion of the update
       $x - \tanh(cx)/c$ around $0$, and reconcile $p$ with the quadratic
       rate stated in :numref:`subsec_mdl-why-not-newton`.

1. [code] **Stale Hessian factorizations.** ● A Newton step in $d$
   dimensions can reuse the Hessian factorization of an earlier step.
   Minimize
   $f(\mathbf{x}) = \sum_{i=1}^{n} \cosh(c\, \mathbf{a}_i^\top \mathbf{x})$
   with $n = 200$, $d = 50$, $c = 0.5$, directions $\mathbf{a}_i$ with
   independent standard Gaussian entries, and $\mathbf{x}_0 = \mathbf{1}$.
   Its minimum is $f^\star = n$ at $\mathbf{x} = \mathbf{0}$. Charge $4nd$
   operations per gradient, $nd^2$ per Hessian evaluation, $d^3/3$ per
   Cholesky factorization, and $2d^2$ per pair of triangular solves.
   Evaluate and factor the Hessian on every step, every 5 steps, and every
   20 steps, reusing the stale factorization in between. Plot
   $f(\mathbf{x}_t) - f^\star$ against the cumulative operation count
   rather than the iteration count. Which variant reaches $10^{-8}$ first,
   and which of the four costs dominates its total?

    *Adapted from Stephen Boyd and Lieven Vandenberghe,
    [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/),
    Exercise 9.31(a).*

[Discussions](https://d2l.discourse.group/t/351)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.2]{.kicker}

Gradient Descent<br>
**the Taylor argument · learning rates · steepest descent · Newton's method · preconditioning**
:::
:::

::: {.slide title="One Learning Rate Decides Between 49 Steps and Never"}
[The problem]{.kicker}

Gradient descent on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$, from
$(-5, -2)$, until both coordinates are within 1%:

| $\eta$ | 0.1 | 0.45 | 0.476 | 0.5 |
|---|---:|---:|---:|---:|
| steps | 228 | 49 | 47 | never |

::: {.d2l-note}
A learning rate that makes a large network diverge does so for the same
reason it makes a quadratic diverge.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive from a Taylor expansion why a small step against the gradient
  descends;
- predict convergence, slowness, or divergence from the factor $1 - \eta h$
  of a curvature $h$;
- explain why the negative gradient zigzags across a narrow valley;
- write Newton's update and say when it fails;
- precondition the valley and count the steps saved.

**Running example:** the valley, curvatures $h_1 = 0.2$ and $h_2 = 4$.
**The idea:** the best step divides the gradient by the curvature.
:::

::: {.slide title="Gradient Descent Is the Second Stop of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| **Gradient Descent** | **step size against curvature** |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| Momentum | averaging over time |
| Adam and AdamW | a step size per coordinate, decoupled decay |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** exact gradients and the Hessian.
- **Skips:** noise (next deck).
:::
:::
:::

::: {.slide title="A Short Step Against the Slope Always Descends"}
[One dimension]{.kicker}

$$f(x - \eta f'(x)) = f(x) - \eta f'(x)^2 + \mathcal{O}(\eta^2 f'(x)^2)$$

- The first-order change $-\eta f'(x)^2$ is negative unless $f'(x) = 0$.
- For small $\eta$ it outweighs the remainder: $x \leftarrow x - \eta f'(x)$
  descends.
- If $f'$ is $L$-Lipschitz, every $0 < \eta < 2/L$ descends.
:::

::: {.slide title="On a Quadratic Each Step Multiplies by 1 − 2η"}
[One dimension]{.kicker}

$f(x) = x^2$: $x \leftarrow x - 2\eta x = (1 - 2\eta)\, x$.

| $\eta$ | factor | $x$ after 10 steps from 10 |
|---|---:|---:|
| 0.05 | 0.9 | 3.49 |
| 0.2 | 0.6 | 0.060 |
| 0.5 | 0 | 0 (one step) |
| 1.1 | −1.2 | 61.9 |

Converges iff $0 < \eta < 1 = 2/L$. Best: $\eta = 1/f''$.
:::

::: {.slide title="η = 0.2 Keeps 60% of the Distance per Step"}
[One dimension]{.kicker}

@!gd-one-dimensional-gradient-descent-4

Ten steps from $x = 10$ leave $10 \cdot 0.6^{10} \approx 0.06$.
:::

::: {.slide title="η = 0.05 Crawls; η = 1.1 Overshoots and Diverges"}
[Learning rate]{.kicker}

::: {.cols .vc}
::: {.col}
@!gd-learning-rate-1
:::
::: {.col}
@!gd-learning-rate-2
:::
:::

Factor $0.9$: still at 3.5. Factor $-1.2$: each step lands farther away.
:::

::: {.slide title="The Learning Rate Decides Which Basin the Iterate Reaches"}
[Local minima]{.kicker}

@!gd-local-minima

$x\cos(0.15\pi x)$ from $x = 10$: $\eta = 0.5$ rolls into the deep basin
at 7.27, $\eta = 2$ jumps over it to the shallow one near $-1.8$.
:::

::: {.slide title="Check: When Does Gradient Descent Diverge?"}
[Check]{.kicker}

$f(x) = 2x^2$, so $f''(x) = 4$.

1. Which learning rates converge?
2. Which learning rate reaches the minimum in one step?

::: {.fragment .fade-up}
1. The factor is $1 - 4\eta$: converges iff $0 < \eta < 0.5$.
2. $\eta = 0.25 = 1/f''$, where the factor is $0$.
:::
:::

::: {.slide title="The Negative Gradient Is the Steepest Direction"}
[Many dimensions]{.kicker}

$$f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$$

- Among steps of length $r$, Cauchy–Schwarz gives
  $\boldsymbol{\epsilon}^\top \nabla f \geq -r \|\nabla f\|$.
- Equality iff $\boldsymbol{\epsilon}$ points along $-\nabla f$.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$
:::

::: {.slide title="Steepest Descent Zigzags Across the Valley"}
[Many dimensions]{.kicker}

@!gd-multivariate-gradient-descent-3

At $(-5, -2)$ the gradient is $(-1, -8)$: eight times steeper across the
valley than along it. After 30 steps at $\eta = 0.45$, $x_1 = -0.3$.
:::

::: {.slide title="Newton's Method Divides the Gradient by the Curvature"}
[Newton's method]{.kicker}

Minimize the second-order model
$f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon}$:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \quad\Longrightarrow\quad \mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1} \nabla f(\mathbf{x})$$

- $\mathbf{H}$: the Hessian, the matrix of second derivatives.
- No learning rate. A quadratic is solved in one step.
- Valley: $\mathbf{H} = \textrm{diag}(0.2, 4)$ and
  $\mathbf{H}^{-1} \nabla f = \mathbf{x}$, so one step reaches the origin,
  for any $\kappa$.
:::

::: {.slide title="On cosh, Ten Newton Steps Reach Machine Precision"}
[Newton's method]{.kicker}

@!gd-newton-s-method-1

$f(x) = \cosh(0.5x)$ from $x = 10$: no learning rate chosen by hand.
:::

::: {.slide title="Negative Curvature Sends Newton Uphill"}
[Newton's method]{.kicker}

::: {.cols .vc}
::: {.col}
@!gd-newton-s-method-2
:::
::: {.col}
@!gd-newton-s-method-3
:::
:::

$x\cos(0.15\pi x)$: the full step lands at $x = 5$ ($f''$ small), jumps
to 25 ($f'' < 0$), and climbs to a maximum at 26.8. Half steps stay in the
basin and halve the distance to 7.27 per step.
:::

::: {.slide title="A Billion Parameters Need Four Exabytes of Hessian"}
[Newton's method]{.kicker}

- Hessian of $d$ parameters: $d^2$ entries; solve: $\mathcal{O}(d^3)$.
- $d = 10^9$: $10^{18}$ entries $\times$ 4 bytes = 4 exabytes.
- Negative eigenvalues steer the raw step to saddles and maxima.

::: {.d2l-note}
Keep the cheap part: rescale each direction by an estimate of its
curvature.
:::
:::

::: {.slide title="Units Alone Can Create a Condition Number of a Million"}
[Preconditioning]{.kicker}

$$\mathbf{x} \leftarrow \mathbf{x} - \eta\, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x})$$

- Linear regression: curvature along a weight = mean squared feature.
- Two unrelated lengths, in meters and in millimeters: curvatures differ
  by $10^6$.
- Dividing by the diagonal removes the factor without knowing the units.
:::

::: {.slide title="Preconditioning Solves the Valley in 7 Steps, Not 47"}
[Preconditioning]{.kicker}

@!gd-preconditioning

$\eta = 0.5$: both coordinates shrink by $0.5$ per step, on a straight
line. At $\eta = 1$ this is Newton's method: one step.
:::

::: {.slide title="Practical Optimizers Estimate the Preconditioner"}
[Where this leads]{.kicker}

| method | preconditioner | from |
|---|---|---|
| Newton | $\mathbf{H}^{-1}$ | exact Hessian, $\mathcal{O}(d^2)$ memory |
| diagonal | $\textrm{diag}(\mathbf{H})^{-1}$ | exact curvatures |
| AdaGrad, RMSProp, Adam | per-coordinate scale | observed gradients |
| Muon | per weight matrix | the gradient matrix |
:::

::: {.slide title="Check: Rotate the Valley by 45°"}
[Check]{.kicker}

The rotated valley has Hessian
$\begin{pmatrix} 2.1 & 1.9 \\ 1.9 & 2.1 \end{pmatrix}$ (eigenvalues 4 and
0.2). What does the diagonal preconditioner do?

::: {.fragment .fade-up}
$\textrm{diag}(\mathbf{H}) = (2.1, 2.1)$: it divides both coordinates by
the same number, a plain change of learning rate. The condition number
stays 20. Only a preconditioner that mixes coordinates, such as
$\mathbf{H}^{-1}$, undoes a rotation.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- A short step against the gradient descends; on $x^2$ it multiplies the
  distance by $1 - 2\eta$: crawl at 0.9, diverge beyond $-1$.
- The learning rate also picks the basin: 7.27 at $\eta = 0.5$, the shallow
  basin at $\eta = 2$.
- Steepest descent zigzags across the valley: 49 steps at $\eta = 0.45$.
- Newton divides by the curvature: one step on a quadratic, uphill under
  negative curvature, $d^2$ memory.
- Diagonal preconditioning: 7 steps instead of 47.

**Next:** the gradient becomes a noisy minibatch estimate.
:::
