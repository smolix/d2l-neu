# Gradient Descent
:label:`sec_gd`

When a training run diverges, we halve the learning rate and try again,
and sooner or later the loss goes down. Once it does, we push the learning
rate back up as far as it will go. Both habits assume a ceiling somewhere,
and neither says where it is or what puts it there.

The valley of :numref:`sec_optimization-intro` answered the question for
one quadratic. Its steep direction, with curvature 4, put a ceiling of
$2/4 = 0.5$ on the learning rate, and its flat direction made gradient
descent take 47 steps even at the best rate below that ceiling. We found
both numbers by multiplying two coordinates by two fixed factors, which
works for a quadratic whose axes we know. A training loss is neither
quadratic nor aligned with any axes we know.

The idea of this section is a single inequality: a step of gradient
descent is safe as long as the learning rate stays below 2 divided by the
largest curvature. We prove it in three lines, for any function whose
slope does not change too fast, and then follow it to three places. At a
minimum, it decides whether gradient descent can stay there, so the
learning rate chooses among the minima of a loss. On a small network, we
watch the curvature rise during training until it meets the limit. And if
curvature is what limits the step, the ideal step should divide by it: we
ask what such a step would cost on a network, and which part of it is
cheap enough to keep. The gradient is exact throughout the section;
:numref:`sec_sgd` replaces it by an estimate.

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #gd-one-dimensional-gradient-descent-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #gd-one-dimensional-gradient-descent-5}
%%tab jax
import functools
```

## The Descent Guarantee

Why should a step against the gradient lower the objective at all? Near a
point $\mathbf{x}$, a differentiable function
$f: \mathbb{R}^d \to \mathbb{R}$ is close to its tangent plane, so a small
step $\boldsymbol{\epsilon}$ changes it by

$$f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}).$$

Among all steps of a fixed small length $r$, which one lowers $f$ the
most? By the Cauchy–Schwarz inequality, the inner product
$\boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$ is at least
$-r \|\nabla f(\mathbf{x})\|$, with equality only when
$\boldsymbol{\epsilon}$ points along $-\nabla f(\mathbf{x})$. The negative
gradient is therefore the direction of *steepest descent*, and a step of
gradient descent with learning rate $\eta > 0$,

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}),$$

changes $f$ by $-\eta \|\nabla f(\mathbf{x})\|^2$ to first order. This is
negative wherever the gradient is not zero.

The words "to first order" hide the difficulty. The tangent plane is a
good guide only near $\mathbf{x}$, and a long step leaves the region where
it can be trusted. How long is too long depends on how fast the slope
changes along the way. One variable is enough to see how.

Let $f$ be a function of one variable whose slope changes by at most $L$
per unit of $x$,

$$|f'(y) - f'(x)| \leq L\, |y - x| \quad \textrm{for all } x \textrm{ and } y.$$

For a function with a second derivative this says $|f''| \leq L$, so $L$
is a bound on the curvature. Now walk a distance $\epsilon > 0$ away from
$x$. After a distance $s$ the slope differs from $f'(x)$ by at most $Ls$,
and $f$ can rise above its tangent line only by the excess slope that
accumulates along the way:

$$f(x + \epsilon) - f(x) - \epsilon f'(x) = \int_0^\epsilon \big(f'(x + s) - f'(x)\big)\, ds \leq \int_0^\epsilon L s \, ds = \frac{L}{2} \epsilon^2.$$

A step to the left obeys the same bound. The function therefore lies
below a parabola of curvature $L$ that touches it at $x$. It remains to
insert the step of gradient descent, $\epsilon = -\eta f'(x)$:

$$f(x - \eta f'(x)) \leq f(x) - \eta \Big(1 - \frac{\eta L}{2}\Big) f'(x)^2.$$
:eqlabel:`gd-guarantee`

Every learning rate with $0 < \eta < 2/L$ therefore lowers $f$ wherever
the slope is not zero, and :eqref:`gd-guarantee` says by how much. The
guaranteed decrease is largest at $\eta = 1/L$, where it equals
$f'(x)^2 / (2L)$: one over the curvature is the natural size of a step, an
observation that returns with Newton's method. At $2/L$ and beyond, the
bound promises nothing.

The same argument, applied along the straight line from $\mathbf{x}$ to
$\mathbf{x} - \eta \nabla f(\mathbf{x})$, gives the result for $d$
variables: :eqref:`gd-guarantee` holds with $\|\nabla f(\mathbf{x})\|^2$
in place of $f'(x)^2$, where $L$ now bounds the curvature in every
direction, that is, the size of every eigenvalue of the Hessian
(:numref:`subsec_mdl-gd-smoothness`). On the valley the Hessian is
$\textrm{diag}(0.2, 4)$ everywhere, so $L = 4$ and the guarantee reads
$\eta < 0.5$. The ceiling we found by hand was no peculiarity of
quadratics.

The guarantee says that $f$ falls, not that it falls quickly: at
$\eta = 1/L = 0.25$ the flat coordinate of the valley still keeps the
factor $0.95$ per step. It is also a worst-case statement, made with no
knowledge but $L$. On the valley, where we knew the flat direction as
well, the best learning rate was $0.476$, almost twice $1/L$.

## Stability of Minima
:label:`subsec_gd-local-minima`

The guarantee assumed one bound $L$ for the whole function. A loss
surface has no such number worth using: it is nearly flat in some regions
and sharply curved in others. So let us fix the learning rate instead and
ask what happens at a minimum whose curvature exceeds $2/\eta$.

Near a minimum $x^*$, a smooth function of one variable is close to a
parabola with curvature $\lambda = f''(x^*)$, and on a parabola each step
of gradient descent multiplies the distance to the minimum by
$1 - \eta\lambda$ (:numref:`sec_optimization-intro`). If
$\lambda < 2/\eta$, this factor is smaller than one in size, and an
iterate that comes close is drawn in. If $\lambda > 2/\eta$, the factor
lies below $-1$: an iterate that comes close,
however close, is thrown to the other side and farther out with every
step. Unless a step happens to land on it exactly, gradient descent cannot
converge to such a minimum, whatever its starting point. A learning rate
therefore does more than set the pace. It sorts the minima of a function
into those sharper than $2/\eta$, which are out of reach, and the rest.

The function $f(x) = x \cos(cx)$ with $c = 0.15\pi$ has minima of both
kinds, a deep one near $x = 7.27$ and a shallow one near $x = -1.83$. We
write a helper `gd` that returns the iterates of gradient descent, and
`show_trace`, which draws them on the graph of $f$.

```{.python .input #gd-one-dimensional-gradient-descent-3}
%%tab pytorch, jax
def gd(eta, f_grad, x=10.0, steps=10):
    """Gradient descent from x; returns all iterates."""
    xs = [x]
    for _ in range(steps):
        x = x - eta * f_grad(x)
        xs.append(float(x))
    return xs

def show_trace(f, *runs, legend=None):
    """Draw the iterates of one or more runs on the graph of f."""
    n = max(abs(x) for xs in runs for x in xs)
    grid = d2l.arange(-n, n, 0.01)
    d2l.plot([grid, *runs], [f(grid), *[[f(x) for x in xs] for xs in runs]],
             '$x$', '$f(x)$', legend=legend, fmts=['-', '-o', '-s'])
```

```{.python .input #gd-local-minima-1}
%%tab pytorch, jax
c = d2l.tensor(0.15 * np.pi)

def f_cos(x):  # Objective function
    return x * d2l.cos(c * x)

def f_cos_grad(x):  # Its derivative
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_cos_hess(x):  # Its second derivative
    return -2 * c * d2l.sin(c * x) - c**2 * x * d2l.cos(c * x)
```

We start at $x = 10$, on the slope above the deep minimum, and run
gradient descent at $\eta = 0.5$ and at $\eta = 2$, whose thresholds
$2/\eta$ are 4 and 1. The second derivative is about $1.8$ at the deep
minimum and just under 1 at the shallow one; the cell prints both. This
is enough to predict the runs. At $\eta = 0.5$ both minima lie far below
the threshold, so the iterate should settle in the first one it meets, the
deep one. At $\eta = 2$ the deep minimum is out of reach. The shallow one
is barely within reach: its factor, $1 - 2 \cdot 0.98 = -0.96$, removes
only 4% of the distance per step, so the iterate should arrive there
slowly. The cell reports each run after 10 and after 100 steps and plots
the first ten.

```{.python .input #gd-local-minima}
%%tab pytorch, jax
minima = [gd(0.5, f_cos_grad, x=x, steps=50)[-1] for x in (10.0, 0.0)]
print(', '.join(f"f''({x:.2f}) = {float(f_cos_hess(x)):.2f}" for x in minima))
runs = {eta: gd(eta, f_cos_grad, steps=100) for eta in (0.5, 2)}
for eta, xs in runs.items():
    print(f'eta {eta}, 2/eta = {2/eta:.0f}: x = {xs[10]:.3f} after 10 steps, '
          f'{xs[100]:.3f} after 100')
show_trace(f_cos, *[xs[:11] for xs in runs.values()],
           legend=['$f$', r'$\eta = 0.5$', r'$\eta = 2$'])
```

With $\eta = 0.5$ the iterate descends into the deep minimum and has
settled at $7.269$ within ten steps. With $\eta = 2$ the first step throws
it to $x \approx 0.6$, past the deep basin altogether, and it then swings
around the shallow minimum: after ten steps it is at $-1.53$, still $0.3$
away, and after a hundred at $-1.821$. The long first jump is not the
reason the run ends there. Had it started right next to the deep minimum,
the factor $1 - 2 \cdot 1.81 = -2.6$ would have driven it out all the
same.

A learning rate of 2 is absurdly large for a problem in one variable, but
the conclusion carries over to surfaces with many minima: the learning
rate chooses among them, and it chooses by curvature, never by depth. On
this function the deeper minimum happens to be the sharper one, so the
larger learning rate ends at the worse value.
Exercise :exref:`ex_gd-minima-within-reach` examines the other minima of
$f$ and what an iterate does just above a threshold.

### The Edge of Stability
:label:`subsec_gd-edge-of-stability`

So far the surface was given and the learning rate was chosen to suit it.
The classical advice has the same order: measure the sharpness of the
loss, the largest eigenvalue $\lambda_{\max}$ of its Hessian
(:numref:`sec_optimization-intro`), and choose $\eta < 2/\lambda_{\max}$.
On a network the dependence can run the other way. :citet:`Cohen.Kaur.Li.ea.2021` trained networks by gradient descent
on the full training set and found that the sharpness rises during
training, a behavior they named *progressive sharpening*, until it reaches
about $2/\eta$, and then stays there. The loss keeps falling in the long
run, although no longer on every step, since the guarantee has run out.
They called this regime the *edge of stability*.

We can watch it happen on a network small enough that its whole Hessian
can be computed. The network has one input, eight $\tanh$ hidden units and
one output, 25 parameters in all, and it fits 16 points of $\sin(3x)$ on
$[-2, 2]$ with the squared loss. We train it by full-batch gradient
descent for 20,000 steps at $\eta = 0.4$ and at $\eta = 0.25$, whose
thresholds $2/\eta$ are 5 and 8, from ten random starts that the two
learning rates share, and we compute the sharpness every 500 steps.

```{.python .input #optimization-intro-the-edge-of-stability-1}
%%tab pytorch
X = torch.linspace(-2, 2, 16)
Y = torch.sin(3 * X)

def mlp_loss(p):  # 8 hidden units: weights, biases, output weights, bias
    W1, b1, W2, b2 = p[:8], p[8:16], p[16:24], p[24]
    return 0.5 * ((torch.tanh(X[:, None] * W1 + b1) @ W2 + b2 - Y) ** 2).mean()

def sharpness(P):  # Largest eigenvalue of the Hessian, for each row of P
    H = torch.func.vmap(torch.func.hessian(mlp_loss))(P)
    return torch.linalg.eigvalsh(H)[:, -1]

def gd_sharpness(eta, P, steps=20000, every=500, last=1000):
    """Full-batch gradient descent from each row of P. Returns the sharpness
    every `every` steps, the final loss, and the fraction of the last `last`
    steps on which the loss rose."""
    grad_and_loss = torch.func.vmap(torch.func.grad_and_value(mlp_loss))
    sharp, losses = [], []
    for k in range(steps + 1):
        if k % every == 0:
            sharp.append(sharpness(P))
        g, loss = grad_and_loss(P)
        losses.append(loss)
        if k < steps:  # The last pass only measures
            P = P - eta * g
    losses = torch.stack(losses[-last - 1:])  # (last + 1, starts)
    return (torch.stack(sharp, 1), losses[-1],
            (losses[1:] > losses[:-1]).float().mean(0))

P0 = 0.5 * torch.randn(10, 25)  # Ten random starts, shared by both runs
```

```{.python .input #optimization-intro-the-edge-of-stability-1}
%%tab jax
X = jnp.linspace(-2, 2, 16)
Y = jnp.sin(3 * X)

def mlp_loss(p):  # 8 hidden units: weights, biases, output weights, bias
    W1, b1, W2, b2 = p[:8], p[8:16], p[16:24], p[24]
    return 0.5 * ((jnp.tanh(X[:, None] * W1 + b1) @ W2 + b2 - Y) ** 2).mean()

def sharpness(P):  # Largest eigenvalue of the Hessian, for each row of P
    return jnp.linalg.eigvalsh(jax.vmap(jax.hessian(mlp_loss))(P))[:, -1]

@functools.partial(jax.jit, static_argnames=('steps', 'every', 'last'))
def gd_sharpness(eta, P, steps=20000, every=500, last=1000):
    """Full-batch gradient descent from each row of P. Returns the sharpness
    every `every` steps, the final loss, and the fraction of the last `last`
    steps on which the loss rose."""
    grad_and_loss = jax.vmap(jax.value_and_grad(mlp_loss))
    def step(P, _):
        loss, g = grad_and_loss(P)
        return P - eta * g, loss
    def block(P, _):  # Measure the sharpness, then take `every` steps
        P_next, losses = jax.lax.scan(step, P, length=every)
        return P_next, (sharpness(P), losses)
    P, (sharp, losses) = jax.lax.scan(block, P, length=steps // every)
    losses = jnp.vstack([losses.reshape(steps, -1), jax.vmap(mlp_loss)(P)])
    losses = losses[-last - 1:]  # (last + 1, starts)
    return (jnp.vstack([sharp, sharpness(P)]).T, losses[-1],
            jnp.mean(losses[1:] > losses[:-1], axis=0))

P0 = 0.5 * jax.random.normal(d2l.get_key(), (10, 25))  # Ten random starts
```

For each start, the next cell prints the sharpness and the loss before
training and after it at each learning rate, together with the fraction
of the last 1,000 steps on which the loss went up. It plots the sharpness
divided by its threshold $2/\eta$: the median over the ten starts, with a
band from the smallest to the largest.

```{.python .input #optimization-intro-the-edge-of-stability-2}
%%tab pytorch, jax
runs = {eta: gd_sharpness(eta, P0) for eta in (0.4, 0.25)}
print('      start      |' + '|'.join(
    f'    eta {eta}, 2/eta = {2 / eta:.0f}     ' for eta in runs))
print('sharpness   loss |' + '|'.join([' sharpness    loss  rising '] * 2))
for i in range(len(P0)):
    print(f'{float(runs[0.4][0][i, 0]):9.2f} {float(mlp_loss(P0[i])):6.3f} |'
          + '|'.join(f' {float(s[i, -1]):9.2f} {float(l[i]):7.1e} '
                     f'{float(r[i]):6.0%} ' for s, l, r in runs.values()))
curves = {rf'$\eta = {eta}$': eta / 2 * s for eta, (s, _, _) in runs.items()}
d2l.plot_curves(curves, 'step', r'sharpness$\,/\,(2/\eta)$',
                x=np.arange(0, 20001, 500), reference=1)
```

The starts have a sharpness of roughly $2.5$ to 6, in most cases below
both thresholds. None of them stays where it began. From every one of the
ten starts the sharpness moves to the threshold of its own learning rate,
and there it remains: the runs at $\eta = 0.4$ all end at $5.0$, and the
runs at $\eta = 0.25$ end between $7.8$ and $8.05$. Most starts climb to
the threshold. A start that happens to be sharper than 5 is brought down
to $5.0$ by the run at $\eta = 0.4$, so the threshold holds from both
sides.
Meanwhile the loss falls by a factor of several hundred or more, to below
$10^{-3}$ in every run.

The last column shows how the loss behaves at the edge, and it takes one
of two values, with rare exceptions in between. In the runs marked 50% the
loss goes up on every second step. The iterate swings from one side of the
sharpest direction to the other, as $x_2$ did on the valley, except that
at a curvature of $2/\eta$ the factor $1 - \eta\lambda$ is $-1$, so the
swing neither dies out nor grows. Each rise of the loss is undone, with a
little to spare, by the step that follows. In the runs marked 0% the swing
has died out, the sharpness rests at the threshold or slightly below it,
and the loss falls on every step. Which runs end in which state varies
with the random starts.

The two learning rates share their starts, yet they end in regions of
different curvature, each as sharp as its learning rate allows. This is
the sorting of minima by curvature again, performed by a network on
itself: among all the places where the loss is low, gradient
descent with learning rate $\eta$ stays only where the sharpness is at
most $2/\eta$.

Gradient descent stays at the edge because it corrects itself. When the
sharpness exceeds $2/\eta$, the swing along the sharpest direction grows,
as a factor below $-1$ demands, and on a quadratic it would grow without
bound. A network's loss is not a quadratic. The third-order term of its
Taylor expansion, which records how the curvature changes from place to
place, turns the growing swing into a drift toward flatter regions, until
the sharpness is back below the threshold
:cite:`Damian.Nichani.Lee.2023`. Then the sharpening resumes. Most
convergence results quoted in this chapter assume a fixed surface and a
loss that falls at every step. The edge of stability lies outside both
assumptions, and training works there all the same. The result is one
about gradient descent with the exact gradient, though: minibatch noise
and the optimizers of later sections change the threshold.

The same self-correction explains a habit of training recipes that might
otherwise pass for superstition: they raise the learning rate gradually
over the first steps instead of starting at its peak. As $\eta$ grows,
the threshold $2/\eta$ falls, and at each stage the mechanism above
lowers the sharpness to the new threshold, so the run arrives at its peak
rate in a region flat enough for it. Started at the peak rate, the same
network can find itself far above the threshold and diverge before the
correction has time to act :cite:`Kalra.Barkeshli.2024`. The ramp is
called *warmup*, and we return to it in :numref:`sec_scheduler`.

## Newton's Method
:label:`subsec_gd-newton`

The bound $\eta < 2/L$ protects the steepest direction and starves all
the others. At the start $(-5, -2)$ of the valley the negative gradient is
$(1, 8)$: it points almost straight at the opposite wall, while the
minimum lies in the direction $(5, 2)$, along the floor. Steepest is not
the same as shortest. The steep coordinate tolerates no learning rate
above $0.5$, whereas the flat one, with curvature $0.2$, would be solved
in a single step by a learning rate of $1/0.2 = 5$. What the valley needs
is a different learning rate in each direction. Newton's method makes this
idea general: it measures the curvature in every direction and divides the
gradient by it.

Later sections return to the valley, so the next cell saves it and its
gradient, together with `train_2d`, which runs a given update rule from
$(-5, -2)$ and returns the iterates, and `gd_2d`, the update rule of
gradient descent.

```{.python .input #gd-multivariate-gradient-descent-4}
%%tab pytorch, jax
VALLEY_CURVATURES = (0.2, 4.0)  #@save
h1, h2 = VALLEY_CURVATURES  # Curvatures of the valley along x1 and x2

def f_valley(x1, x2):  #@save
    """The ill-conditioned quadratic valley h1/2 x1^2 + h2/2 x2^2, with
    (h1, h2) = VALLEY_CURVATURES."""
    h1, h2 = VALLEY_CURVATURES
    return h1 / 2 * x1 ** 2 + h2 / 2 * x2 ** 2

def valley_grad(x1, x2):  #@save
    """Gradient of f_valley."""
    h1, h2 = VALLEY_CURVATURES
    return h1 * x1, h2 * x2

def train_2d(trainer, f_grad, steps=20, **hparams):  #@save
    """Run a 2D update rule from (-5, -2) and return the iterates."""
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1, s2: state of stateful optimizers
    results = [(x1, x2)]
    for _ in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad, **hparams)
        results.append((x1, x2))
    print(f'step {steps}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def gd_2d(x1, x2, s1, s2, f_grad, eta):  #@save
    """One step of gradient descent in 2D; the state stays zero."""
    g1, g2 = f_grad(x1, x2)
    return x1 - eta * g1, x2 - eta * g2, 0, 0
```

The curvature enters through the second-order term of the Taylor
expansion,

$$f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon},$$

where $\mathbf{H} = \nabla^2 f(\mathbf{x})$ is the Hessian, the
$d \times d$ matrix of second derivatives. The right-hand side is a
quadratic in $\boldsymbol{\epsilon}$, and unlike a tangent plane, a
quadratic with positive curvature has a lowest point. Setting its gradient
with respect to $\boldsymbol{\epsilon}$ to zero (with the rules of
:numref:`subsec_calculus-grad`) gives the step that reaches it:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0, \quad \textrm{hence} \quad \boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Newton's method is gradient descent with the inverse Hessian in place of
the learning rate. Each direction receives the step that its own curvature
calls for, and nothing is left to tune. When $f$ is itself a quadratic,
the expansion is exact and a single Newton step lands on the minimum. On
the valley, $\mathbf{H} = \textrm{diag}(0.2, 4)$ and
$\nabla f(\mathbf{x}) = (0.2 x_1, 4 x_2)$, so
$\mathbf{H}^{-1} \nabla f(\mathbf{x}) = \mathbf{x}$, and one step from any
point reaches the origin, whatever the condition number.

On functions that are not quadratic the method has to iterate. In one
variable the Newton step is $-f'(x)/f''(x)$, which the code below takes
literally: `newton` is `gd` with learning rate 1 applied to $f'/f''$. The
convex function $f(x) = \cosh(cx)$ with $c = 0.5$ has its minimum at
$x = 0$. We take ten Newton steps from $x = 10$ and print the iterates.

```{.python .input #gd-newton-s-method-1}
%%tab pytorch, jax
def f_cosh(x, c=d2l.tensor(0.5)):  # Objective function
    return d2l.cosh(c * x)

def f_cosh_grad(x, c=d2l.tensor(0.5)):  # First derivative
    return c * d2l.sinh(c * x)

def f_cosh_hess(x, c=d2l.tensor(0.5)):  # Second derivative
    return c**2 * d2l.cosh(c * x)

def newton(f_grad, f_hess, steps=10):
    """Newton's method from x = 10: gradient descent on f_grad / f_hess."""
    return gd(1, lambda x: f_grad(x) / f_hess(x), steps=steps)

xs = newton(f_cosh_grad, f_cosh_hess)
print('x_t:', ' '.join(f'{x:.2g}' for x in xs))
show_trace(f_cosh, xs)
```

The run has two phases, and both are typical of Newton's method. Far from
the minimum the iterate walks: four steps of almost exactly 2 each take it
from 10 to 4, because out there $f'/f'' = \tanh(cx)/c$ is close to
$1/c = 2$. Near the minimum the error collapses, from $0.53$ to $0.012$
to $10^{-7}$ and then to zero in the precision of the arithmetic. This
second phase made the method's reputation. Near a minimum with positive curvature, Newton's method
converges at least *quadratically*: the error after a step is at most a
constant times the square of the error before it, so the number of correct
digits at least doubles with every step
(:numref:`subsec_mdl-why-not-newton`). On $\cosh$ it does even better,
because the function is symmetric about its minimum
(Exercise :exref:`ex_gd-newton-order`).

Newton's method divides by the curvature, and the curvature of a nonconvex
function can be negative. Where it is, the step points uphill. We return
to $f(x) = x \cos(cx)$ and start again at $x = 10$, with the minimum at
$7.27$ close by.

```{.python .input #gd-newton-s-method-2}
%%tab pytorch, jax
xs = newton(f_cos_grad, f_cos_hess)
print('x_t:', ' '.join(f'{x:.4g}' for x in xs))
show_trace(f_cos, xs)
```

The first step overshoots the minimum and lands at $x = 5$. The curvature
there is positive but small, about $0.12$, so the next step divides a
slope of $-2.4$ by a small number and jumps by 20, to $x \approx 25$,
where the curvature is negative. From there the iteration climbs to the
local maximum at $26.83$ and stays. Seen from the method, nothing went
wrong: Newton's method looks for a point where the gradient vanishes, and
a maximum is such a point as much as a minimum is. One remedy replaces the
Hessian by its absolute value, the matrix with the same eigenvectors and
the sizes of its eigenvalues, so that every direction is treated as if it
curved upward and every step points downhill.

Even where it behaves, Newton's method is out of reach for a network,
though not for the reason usually given. The Hessian of a model with $d$
parameters has $d^2$ entries: for $d = 10^9$ that is $10^{18}$ numbers,
four exabytes in single precision, before the first step. But the method
does not need the matrix. The product of the Hessian with a vector costs
roughly one more gradient (:numref:`subsec_mdl-hvp`), and an iterative
solver finds the Newton step from such products alone. Three other
obstacles remain. The solver needs many products for a single step, each
as costly as a gradient step that would have made progress by itself. In
training, derivatives are computed on a minibatch, and a division by an
estimated curvature is unforgiving: where the true curvature is close to
zero, a small error in the estimate changes the step by a large factor or
reverses it. And the Hessian of a network has negative eigenvalues, which
send the raw step toward saddle points and maxima, as in the run above.
Classical optimization answers each of these with curvature estimates
built from successive gradients and with safer step rules
(:numref:`subsec_mdl-quasi-newton`). Deep learning keeps a simpler part of
the idea: rescale the gradient, direction by direction, with an
inexpensive estimate of the curvature.

## Preconditioning
:label:`subsec_gd-preconditioning`

A *preconditioner* is a matrix that rescales the gradient before the
step. Newton's method uses the inverse Hessian. The simplest useful
preconditioner keeps only the diagonal of the Hessian,

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}),$$
:eqlabel:`eq_gd-diag-precond`

which divides each coordinate of the gradient by the curvature along that
coordinate. In effect, every coordinate receives its own learning rate,
and the diagonal takes no more memory than the gradient itself.

Curvature that differs wildly between coordinates arises from something
as mundane as units. Suppose a linear regression model receives two
unrelated length features of similar size, one recorded in meters and the
other in millimeters. The curvature of the squared loss along a weight is
the mean of the squared feature, so the millimeter weight has a curvature
$10^6$ times larger than the meter weight. The units alone give a
condition number of at least $10^6$. A single learning rate small enough
for the millimeter weight leaves the meter weight almost frozen. Dividing
by the diagonal curvature removes the factor $10^6$ without our ever
noticing the units.

On the valley the diagonal of the Hessian is $(0.2, 4)$, and at
$\eta = 0.5$ the preconditioned update multiplies both coordinates by the
same factor, $1 - \eta = 0.5$. Since $0.5^6 \approx 0.016$ and
$0.5^7 \approx 0.008$, both coordinates should be within 1% of the minimum
after seven steps. The cell counts the steps, and for comparison those of
plain gradient descent at its best learning rate, $2/(0.2 + 4)$. The first
two lines of the output are the last iterates of the two runs, which
`train_2d` prints.

```{.python .input #gd-preconditioning}
%%tab pytorch, jax
def precond_gd_2d(x1, x2, s1, s2, f_grad, eta):
    g1, g2 = f_grad(x1, x2)
    return x1 - eta * g1 / h1, x2 - eta * g2 / h2, 0, 0  # diag(H) = (h1, h2)

def steps_to_1pct(xs):  # First step with both coordinates within 1% of start
    return next(t for t, (x1, x2) in enumerate(xs)
                if abs(x1) <= 0.01 * 5 and abs(x2) <= 0.01 * 2)

plain = train_2d(gd_2d, valley_grad, steps=100, eta=2 / (h1 + h2))
precond = train_2d(precond_gd_2d, valley_grad, steps=10, eta=0.5)
print(f'steps to 1%: plain at its best rate {steps_to_1pct(plain)}, '
      f'preconditioned {steps_to_1pct(precond)}')
d2l.show_trace_2d(f_valley, precond)
```

Seven steps do the work of 47. The path in the plot runs straight to the
minimum, since both coordinates shrink in step and nothing is left to
zigzag. More important than the count is what it no longer depends on: the
factor $1 - \eta$ is the same for every curvature, so the condition number
has dropped out. At $\eta = 1$ one step suffices, because the Hessian of
the valley is diagonal and the diagonal preconditioner is then Newton's
method.

A network offers neither of the two things this run relied on. It does not
give us its Hessian, not even the diagonal, so practical methods estimate
a scale for each coordinate from the gradients they observe. AdaGrad,
RMSProp, and Adam (:numref:`sec_adam`) differ mainly in how they form that
estimate. Nor do its steep and flat directions line up with the coordinate
axes, which a diagonal preconditioner needs. Turn the valley by
$45^\circ$ and the two diagonal entries of its Hessian become equal, both
$2.1$. Dividing by them is then a plain change of learning rate, and the
condition number is 20 again
(Exercise :exref:`ex_gd-diagonal-preconditioning`). This is one motive for
methods that treat a whole weight matrix at once (:numref:`sec_muon`).

One assumption has run through the whole section: the gradient was exact.
For a training loss, an exact gradient costs a pass over the entire
dataset, once for every step. What remains of the ceiling and of the pace
when the gradient is estimated from a few examples is the subject of
:numref:`sec_sgd`.

## Summary

Wherever the curvature of a function is at most $L$, a step of gradient
descent with a learning rate below $2/L$ lowers it; the ceiling of $0.5$
on the valley is this bound with $L = 4$. The same threshold decides where
gradient descent can come to rest: a minimum sharper than $2/\eta$ is
unstable, so the learning rate selects minima by their curvature. Under
full-batch gradient descent, the sharpness of a network rises during
training until it reaches about $2/\eta$ and stays there, while the loss
keeps falling without falling on every step. Dividing the gradient by the
curvature removes the condition number, and Newton's method solves any
quadratic in one step. On a network it would need many gradient-sized
computations per step, it would divide by noisy curvature estimates, and
it is drawn to saddle points and maxima. Diagonal preconditioning keeps
the affordable part: it solved the valley in seven steps instead of 47,
given the curvature of each coordinate and axes aligned with the valley,
neither of which a network supplies.

## Exercises

1. [code] **Three kinds of curvature.** Extend `gd` to stop as soon as
   $|x| < 10^{-3}$, and run it from $x = 1$ for at most 100 steps on
   $f(x) = x^2$, $f(x) = |x|^{1.5}$, and $f(x) = x^4$ at
   $\eta \in \{0.01, 0.1, 0.5, 0.9, 1.1\}$.
    1. Tabulate, for each of the fifteen runs, the number of steps until
       the stopping test holds, or, for a run that misses the test within
       100 steps, whether $|x|$ is still shrinking, cycles, or grows.
       Which entries of the row for $x^2$ does the guarantee
       $\eta < 2/L$ predict? What does the entry at $\eta = 0.01$ show
       about what a guarantee of descent leaves out?
    1. The second derivative of $|x|^{1.5}$ is unbounded near the origin,
       so no bound $L$ exists. Show that at a fixed $\eta$ the iterates
       end in a cycle between $\pm (0.75\eta)^2$, and find the learning
       rates for which the stopping test can be met at all.
    1. The second derivative of $x^4$ is unbounded far from the origin and
       vanishes at it. Show that the update is
       $x \leftarrow x (1 - 4\eta x^2)$, that it shrinks $|x|$ only from
       starts with $x^2 < 1/(2\eta)$, and that $|x_t|$ then decays like
       $1/\sqrt{8 \eta t}$ rather than geometrically. Why can no fixed
       learning rate restore a geometric rate on this function?

1. [code] **Diagonal preconditioning under rotation.** Rotate the valley
   by an angle $\theta$: $f_\theta(\mathbf{x}) = \frac{1}{2}
   \mathbf{x}^\top \mathbf{R}_\theta \textrm{diag}(0.2, 4)
   \mathbf{R}_\theta^\top \mathbf{x}$, where $\mathbf{R}_\theta$ is the
   rotation matrix by $\theta$.
    1. For $\theta = 45^\circ$, run plain gradient descent at
       $\eta = 0.45$, the diagonal preconditioner
       :eqref:`eq_gd-diag-precond` at $\eta = 0.5$, and Newton's method
       with `d2l.train_2d`. Which of the three behave as on the unrotated
       valley? The preconditioned run follows the path of plain gradient
       descent at a different learning rate. Which one?
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

1. [code] **Minima within reach.** The function $f(x) = x\cos(cx)$ with
   $c = 0.15\pi$ of :numref:`subsec_gd-local-minima` has four local minima
   in the interval $(-20, 30)$.
    1. Locate them with `gd` at a small learning rate and compute $f''$ at
       each with `f_cos_hess`. For each minimum, state the largest
       learning rate at which gradient descent can converge to it.
    1. Start `gd` at a distance of $0.01$ from each minimum and run 300
       steps at $0.9$ and at $1.1$ times that learning rate. Above the
       threshold, does the iterate diverge, move to another minimum, or
       do something else? Relate what you see to the runs that
       :numref:`subsec_gd-edge-of-stability` marks with 50%.
    1. Run 2,000 steps at $\eta = 2$ from $x = 20$, $10$, $0$, and $-12$.
       Where do the four runs end? Rank the four minima by their value
       and by their curvature. Which of the two did the learning rate
       select for?
   :exlabel:`ex_gd-minima-within-reach`

1. [code] **Newton's convergence order.** ● In :numref:`subsec_gd-newton`,
   `newton` minimizes $f(x) = \cosh(cx)$ with $c = 0.5$ from $x_0 = 10$
   within ten steps. Record $|x_t|$ after every step.
    1. From the last steps before $x_t$ reaches $0$ in floating point,
       estimate the order $p$ in $|x_{t+1}| \approx C |x_t|^p$. How many
       steps would the walking phase take from $x_0 = 100$, and how many
       the fast phase?
    1. Derive $p$ and $C$ from the Taylor expansion of the update
       $x - \tanh(cx)/c$ around $0$, and reconcile $p$ with the quadratic
       rate stated in :numref:`subsec_mdl-why-not-newton`.
   :exlabel:`ex_gd-newton-order`

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
**the descent guarantee · stable and unstable minima · the edge of stability · Newton's method · preconditioning**
:::
:::

::: {.slide title="On the Valley No Learning Rate Works at 0.5 or Above"}
[The problem]{.kicker}

Gradient descent $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$
on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$, from $(-5, -2)$.
Steps until both coordinates are within 1% of the minimum:

| learning rate $\eta$ | 0.1 | 0.3 | 0.45 | 0.476 | 0.49 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| steps | 228 | 75 | 49 | 47 | 113 | never |

::: {.d2l-note}
A diverging run is cured by halving $\eta$. What puts the ceiling where
it is, on a function that is not a quadratic?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- prove that every learning rate below $2/L$ lowers a function whose
  curvature is at most $L$;
- decide from $f''$ whether gradient descent can settle in a minimum;
- predict the sharpness of a network trained by full-batch gradient
  descent;
- write Newton's update and name what rules it out for a network;
- precondition the valley and count the steps saved.

**Running example:** the valley, curvatures $0.2$ and $4$, condition
number $\kappa = 20$.
**The idea:** curvature sets the step, at most $2/L$, ideally one over
the curvature of each direction.
:::

::: {.slide title="Gradient Descent Turns the Valley Into a General Rule"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | uneven curvature, the valley |
| **Gradient Descent** | **step size against curvature** |
| Stochastic and Minibatch GD | sampled gradients, the noise ball |
| Two Test Models | a language model and a CNN to measure on |
| Momentum | averaging gradients over steps |
| Adam, AdamW | a step size per coordinate, weight decay |
| Muon | a step for a whole weight matrix |
| Schedules | warmup and decay |
| Batch Size, Scaling Up, Practice | larger batches, wider models, a recipe |
:::

::: {.col .narrow}
- **Uses:** gradients, the Hessian and its eigenvalues.
- **Skips:** noisy gradients (next deck), quasi-Newton methods, line
  search.
:::
:::
:::

::: {.slide title="The Negative Gradient Is the Steepest Direction"}
[The descent guarantee]{.kicker}

A small step $\boldsymbol{\epsilon}$ changes $f$ by about
$\boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$.

- Among steps of length $r$, Cauchy–Schwarz gives
  $\boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) \geq -r \|\nabla f(\mathbf{x})\|$.
- Equality only if $\boldsymbol{\epsilon}$ points along
  $-\nabla f(\mathbf{x})$.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}) \quad \textrm{changes } f \textrm{ by about } -\eta \|\nabla f(\mathbf{x})\|^2$$

::: {.d2l-note .warn}
True for a short step only. How short is short enough?
:::
:::

::: {.slide title="I. A Slope That Changes by at Most L Keeps f Below a Parabola"}
[The descent guarantee]{.kicker}

Assume $|f'(y) - f'(x)| \leq L\,|y - x|$ for all $x, y$; with a second
derivative, $|f''| \leq L$.

After a distance $s$ the slope exceeds $f'(x)$ by at most $Ls$:

$$f(x + \epsilon) - f(x) - \epsilon f'(x) = \int_0^\epsilon \big(f'(x + s) - f'(x)\big)\, ds \leq \int_0^\epsilon L s\, ds = \frac{L}{2}\epsilon^2$$

::: {.d2l-note}
$f$ lies below the parabola of curvature $L$ that touches it at $x$.
:::
:::

::: {.slide title="II. Every Learning Rate Below 2/L Lowers f"}
[The descent guarantee]{.kicker}

Insert the gradient step $\epsilon = -\eta f'(x)$:

$$\boxed{f(x - \eta f'(x)) \leq f(x) - \eta \Big(1 - \frac{\eta L}{2}\Big) f'(x)^2}$$

- $0 < \eta < 2/L$: a strict decrease wherever $f'(x) \neq 0$.
- Largest guaranteed decrease at $\eta = 1/L$: $f'(x)^2 / (2L)$.
- In $d$ variables: $\|\nabla f(\mathbf{x})\|^2$ replaces $f'(x)^2$, and
  $L$ bounds the size of every eigenvalue of the Hessian.
:::

::: {.slide title="On the Valley L = 4, and the Guarantee Is the Ceiling 0.5"}
[The descent guarantee]{.kicker}

Hessian $\textrm{diag}(0.2, 4)$ everywhere, so $L = 4$ and $2/L = 0.5$.

| $\eta$ | steep factor $1 - 4\eta$ | flat factor $1 - 0.2\eta$ | guarantee |
|---|---:|---:|---|
| $0.25 = 1/L$ | 0 | 0.95 | holds, largest |
| $0.476$ | −0.905 | 0.905 | holds |
| $0.5 = 2/L$ | −1 | 0.9 | none |

::: {.d2l-note}
The guarantee says that $f$ falls, not how fast: the flat direction
still sets the pace.
:::
:::

::: {.slide title="Check: Which Learning Rates Are Safe on cos x?"}
[Check]{.kicker}

$f(x) = \cos x$.

1. For which learning rates does the guarantee hold?
2. Which learning rate has the largest guaranteed decrease, and how
   large is it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $|f''(x)| = |\cos x| \leq 1$, so $L = 1$: every
$0 < \eta < 2$ lowers $f$. The best guaranteed rate is $\eta = 1$, with
decrease $f'(x)^2/2 = \sin^2(x)/2$.
:::
:::
:::

::: {.slide title="A Minimum Sharper Than 2/η Repels Gradient Descent"}
[Stability of minima]{.kicker}

Near a minimum $x^*$ with curvature $\lambda = f''(x^*)$, each step
multiplies the distance to $x^*$ by $1 - \eta\lambda$.

- $\lambda < 2/\eta$: factor smaller than 1 in size, the minimum
  attracts.
- $\lambda > 2/\eta$: factor below $-1$, the minimum repels from any
  start.

$f(x) = x\cos(0.15\pi x)$ has a deep and a shallow minimum:

| minimum | $f''$ | factor at $\eta = 0.5$ | factor at $\eta = 2$ |
|---|---:|---:|---:|
| $x = 7.27$ (deep) | 1.81 | 0.09 | −2.6 |
| $x = -1.83$ (shallow) | 0.98 | 0.51 | −0.96 |
:::

::: {.slide title="At η = 2 the Deep Minimum Is Out of Reach"}
[Stability of minima]{.kicker}

@!gd-local-minima

From $x = 10$: $\eta = 0.5$ settles in the deep minimum; $\eta = 2$ ends
at the shallow one, slowly (factor $-0.96$). The learning rate selects
minima by curvature, not by depth.
:::

::: {.slide title="On a Network the Sharpness Rises Until It Meets 2/η"}
[Edge of stability]{.kicker}

- *Sharpness*: the largest eigenvalue $\lambda_{\max}$ of the Hessian of
  the loss. Classical advice: measure it, choose $\eta < 2/\lambda_{\max}$.
- Full-batch gradient descent on networks (Cohen et al., 2021): the
  sharpness **rises** during training to about $2/\eta$, then stays.
- The loss keeps falling, but not on every step.

**Test:** a $\tanh$ network with 25 parameters fits 16 points of
$\sin(3x)$; ten random starts; $\eta = 0.4$ and $\eta = 0.25$, thresholds
$2/\eta = 5$ and $8$; 20,000 steps.
:::

::: {.slide title="Ten Starts End at Sharpness 5.0 and 8, One per Learning Rate"}
[Edge of stability]{.kicker}

@!optimization-intro-the-edge-of-stability-2

Plot: sharpness divided by $2/\eta$, median and range over the starts.
Same starts, loss below $10^{-3}$ in every run.
:::

::: {.slide title="At the Edge the Loss Rises on Every Second Step"}
[Edge of stability]{.kicker}

- At curvature $2/\eta$ the factor $1 - \eta\lambda$ is $-1$: the iterate
  swings across the sharpest direction, and the loss rises on 50% of the
  steps.
- Above $2/\eta$ the swing grows; the third-order term of the Taylor
  expansion turns it into a drift toward flatter regions (Damian et al.,
  2023).
- Other runs rest just below the threshold: the loss rises on 0% of the
  steps.

::: {.d2l-note}
**Warmup:** a learning rate that rises gradually lowers the sharpness
stage by stage, so the peak rate arrives in a region flat enough for it
(Kalra and Barkeshli, 2024).
:::
:::

::: {.slide title="Check: What Sharpness Will Training Reach?"}
[Check]{.kicker}

The 25-parameter network is trained by full-batch gradient descent at
$\eta = 0.1$.

1. What sharpness do you expect after long training?
2. A minimum of the loss has sharpness 30. Can this run end there?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** About $2/\eta = 20$. No: 30 exceeds 20, so the minimum
repels the iterate. It is within reach only for $\eta < 2/30 \approx 0.067$.
:::
:::
:::

::: {.slide title="The Negative Gradient Points at the Wall, Not at the Minimum"}
[Newton's method]{.kicker}

Valley, start $(-5, -2)$:

- negative gradient $(1, 8)$, direction to the minimum $(5, 2)$;
- $x_2$ (curvature 4) tolerates no $\eta$ above $0.5$;
- $x_1$ (curvature $0.2$) would be solved in one step by
  $\eta = 1/0.2 = 5$.

::: {.d2l-note}
The valley needs a learning rate per direction: 5 along $x_1$, $1/4$
along $x_2$. One over the curvature.
:::
:::

::: {.slide title="Newton's Method Divides the Gradient by the Curvature"}
[Newton's method]{.kicker}

Minimize the second-order model
$f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon}$
over the step $\boldsymbol{\epsilon}$:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \quad\Longrightarrow\quad \mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1} \nabla f(\mathbf{x})$$

- $\mathbf{H}$: the Hessian, the matrix of second derivatives.
- No learning rate. A quadratic is solved in one step.
- Valley: $\mathbf{H} = \textrm{diag}(0.2, 4)$,
  $\nabla f(\mathbf{x}) = (0.2 x_1, 4 x_2)$, so
  $\mathbf{H}^{-1} \nabla f(\mathbf{x}) = \mathbf{x}$: one step to the
  origin, for any $\kappa$.
:::

::: {.slide title="Far Away Newton Walks, Near the Minimum the Error Collapses"}
[Newton's method]{.kicker}

@!gd-newton-s-method-1

$f(x) = \cosh(0.5x)$ from $x = 10$. Far out
$f'/f'' = \tanh(0.5x)/0.5 \approx 2$: steps of 2. Then $0.53$, $0.012$,
$10^{-7}$: the correct digits at least double per step.
:::

::: {.slide title="Negative Curvature Sends Newton to a Maximum"}
[Newton's method]{.kicker}

@!gd-newton-s-method-2

$x\cos(0.15\pi x)$ from $x = 10$: at $x = 5$ the curvature is $0.12$,
the step is 20 long and lands where $f'' < 0$. Newton finds points with
zero gradient; a maximum is one.
:::

::: {.slide title="Storage Is Not What Rules Newton Out for a Network"}
[Newton's method]{.kicker}

- $d = 10^9$ parameters: $10^{18}$ Hessian entries $\times$ 4 bytes =
  4 exabytes.
- But a Hessian–vector product costs about one more gradient, and an
  iterative solver needs only such products.

What remains:

1. many products per step, each as costly as a gradient step;
2. minibatch noise: dividing by a curvature estimated near zero;
3. negative eigenvalues: steps toward saddle points and maxima.

::: {.d2l-note}
Keep the cheap part: rescale each direction by an estimate of its
curvature.
:::
:::

::: {.slide title="Units Alone Create a Condition Number of a Million"}
[Preconditioning]{.kicker}

$$\mathbf{x} \leftarrow \mathbf{x} - \eta\, \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x})$$

- Linear regression: curvature along a weight = mean squared feature.
- Two unrelated lengths, one in meters, one in millimeters: curvatures
  differ by $10^6$.
- Dividing by the diagonal removes the factor without knowing the units.
:::

::: {.slide title="Preconditioning Solves the Valley in 7 Steps, Not 47"}
[Preconditioning]{.kicker}

@!gd-preconditioning

$\eta = 0.5$: both coordinates keep the factor $1 - \eta = 0.5$, and
$0.5^7 \approx 0.008$. The condition number has dropped out. At
$\eta = 1$ this is Newton's method.
:::

::: {.slide title="Check: Rotate the Valley by 45°"}
[Check]{.kicker}

The rotated valley has Hessian
$\begin{pmatrix} 2.1 & 1.9 \\ 1.9 & 2.1 \end{pmatrix}$, with eigenvalues
4 and $0.2$. What does the diagonal preconditioner do?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\textrm{diag}(\mathbf{H}) = (2.1, 2.1)$: both coordinates
are divided by the same number, a plain change of learning rate. The
condition number stays 20. Only a preconditioner that mixes coordinates,
such as $\mathbf{H}^{-1}$, undoes a rotation.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Curvature at most $L$: every $\eta < 2/L$ lowers $f$. Valley: $L = 4$,
  ceiling $0.5$.
- A minimum sharper than $2/\eta$ repels: at $\eta = 2$ the run leaves
  the minimum with $f'' = 1.81$ for the one with $f'' = 0.98$.
- Full-batch gradient descent on a network: sharpness rises to $2/\eta$
  ($5.0$ at $\eta = 0.4$, about 8 at $\eta = 0.25$) and stays.
- Newton divides by the curvature: one step on the valley; a maximum
  under negative curvature; too costly and too noisy for a network.
- Diagonal preconditioning: 7 steps instead of 47.

**Next:** the gradient becomes a noisy estimate from a minibatch.
:::
