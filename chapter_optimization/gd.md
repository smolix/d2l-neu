# Gradient Descent
:label:`sec_gd`

Every model in this book was trained with a learning rate fixed before
the run, usually copied from a recipe and lowered if the loss diverged.
The previous section showed why the choice is delicate: on a valley that
curves steeply in one direction and gently in another, a step small enough
for the steep direction makes little progress along the gentle one. This
section asks what decides which learning rates work, and how fast the best
of them can go. We answer it for gradient descent, the simplest optimizer
and the ancestor of every other method in this chapter: compute the
gradient, take a step against it, and repeat.

Gradient descent has a single knob, the learning rate $\eta$, and on the
valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of
:numref:`sec_optimization-intro` this knob already decides everything. At
$\eta = 0.45$ the iterate needs 49 steps to come within 1% of the minimum,
at $\eta = 0.5$ it never arrives, and no learning rate does better than 47
steps. A learning rate that makes a large network diverge does so for the
same reason that it makes a one-dimensional quadratic diverge, so the small
cases are worth understanding completely.

We split the question into three. Why does a step against the gradient
lower the objective, and how large may the step be? What would the ideal
step be if we also knew the curvature? And how can we approximate that
ideal cheaply? The answers are a Taylor expansion, Newton's method, and
preconditioning. Preconditioning is the idea behind the adaptive methods of
:numref:`sec_adam`. Throughout the section the gradient is exact;
:numref:`sec_sgd` replaces it with a noisy estimate.

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

## One-Dimensional Gradient Descent

Why should a step against the gradient reduce the objective at all? The
one-dimensional case already contains the answer. Let
$f: \mathbb{R} \rightarrow \mathbb{R}$ be continuously differentiable. For
a small change $\epsilon$ in its argument, its Taylor expansion reads

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + O(\epsilon^2).$$
:eqlabel:`gd-taylor`

Near $x$ the function is close to its tangent line, so a move downhill
along the tangent should decrease $f$. We pick a step size $\eta > 0$, set
$\epsilon = -\eta f'(x)$, and substitute:

$$f(x - \eta f'(x)) = f(x) - \eta f'(x)^2 + O(\eta^2 f'(x)^2).$$
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

<!-- WRITER: cells gd-one-dimensional-gradient-descent-2/-3/-4 (x^2, its ten-step run printing 0.060466 and its plot) are cut (GD-2, GD-3): "We check this numerically" and the next paragraph ("The iterates approach ... We plot them on the graph of f") have no cell any more. The arithmetic 10 * 0.6^10 = 0.06 stands on its own. -->

The iterates approach the minimum geometrically, each step keeping 60% of
the distance. We plot them on the graph of $f$.

### Learning Rate
:label:`subsec_gd-learningrate`

The learning rate is ours to choose, and the choice can go wrong in either
direction. With $\eta = 0.05$ the factor is $0.9$, and ten steps leave
$10 \cdot 0.9^{10} \approx 3.5$: the iterate barely moves, and reaching the
minimum would take many more steps.

<!-- WRITER: cell gd-learning-rate-1 (a plotted ten-step run on x^2, output 3.486784) is cut (GD-2); the sentence before it states the arithmetic, which needs no cell. -->

With $\eta = 1.1$ the factor is $-1.2$. Each step overshoots the minimum and
lands farther away on the other side, so after ten steps the distance has
grown to $10 \cdot 1.2^{10} \approx 62$. In the language of
:eqref:`gd-taylor-2`, the step $|\eta f'(x)|$ is too long for the tangent
line to be a good guide, and the remainder term takes over.

<!-- WRITER: cell gd-learning-rate-2 (a plotted ten-step run on x^2, output 61.917364) is cut (GD-2); the sentence before it states the arithmetic, which needs no cell. -->

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

<!-- WRITER: the cells now (i) define `gd` (returns the iterates, takes a start x, no print) and `show_trace` once, (ii) define f_cos with f_cos_grad and f_cos_hess (moved here from the Newton cell), (iii) print f'' at both minima and where each run is after 10 and 100 steps, and plot the first 10 steps of both runs. "plot the second run" is now false. Measured (identical in both tabs, deterministic): f''(7.27) = 1.81, f''(-1.83) = 0.98; eta 0.5 (2/eta = 4): x = 7.269 after 10 and after 100 steps; eta 2 (2/eta = 1): x = -1.528 after 10 steps, -1.821 after 100. Reading per GD-6: at eta = 2 the deep minimum (f'' = 1.81 > 1) is unstable from any start; the shallow one (0.98 < 1) is barely stable, so the run settles there slowly (factor 1 - 2 * 0.98 = -0.96 per step). -->

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

With $\eta = 0.5$ the iterate rolls into the deep basin and stops near
$7.27$. With $\eta = 2$ the first step jumps over that basin to
$x \approx 0.6$, and the iterate ends up bouncing around the shallow
minimum. A learning rate this large is unrealistic for a one-dimensional
problem, but large steps in the early phase of training play the same role
in deep networks: they decide which region of the surface training
explores.

### The Edge of Stability
:label:`subsec_gd-edge-of-stability`

<!-- WRITER: moved verbatim from optimization-intro.md. "The valley" below is the valley of sec_optimization-intro, which this file reintroduces only later (Multivariate Gradient Descent); the subsection sits under "One-Dimensional Gradient Descent" until the gd.md rebuild reorders the headings. -->

The valley treats curvature as a fixed property of the surface, and the
classical advice follows from it: measure the largest curvature
$\lambda_{\max}$, called the *sharpness*, and choose $\eta < 2/\lambda_{\max}$.
On real networks the dependence can run the other way.
:citet:`Cohen.Kaur.Li.ea.2021` trained networks with full-batch gradient
descent and found that the sharpness rises during training, a behavior
they named *progressive sharpening*, until it reaches about $2/\eta$, and
then stays there. The loss keeps falling, although not on every step, in a
regime that the quadratic analysis calls unstable. They called this regime
the *edge of stability*.

We can watch it happen on a network small enough to compute its full
Hessian. The network has one input, eight $\tanh$ hidden units, and one
output, 25 parameters in all, and it fits 16 points of $\sin(3x)$ on
$[-2, 2]$ with the squared loss. We train it from one random start with
full-batch gradient descent at $\eta = 0.4$ and at $\eta = 0.25$, whose
ceilings $2/\eta$ are 5 and 8, and compute the sharpness every 500 steps.

<!-- WRITER: the experiment now trains from ten random starts at once (vmap), the same ten for both learning rates, and also returns the fraction of the last 1,000 steps on which the loss rose. "from one random start" is false. The plot shows sharpness / (2/eta), median over the ten starts with a min-max band, against the reference line 1; both bands climb to 1 by about step 4,000 and stay. -->

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

Both runs start from the same weights, with a sharpness well below 8 and
usually below 5. Neither stays there. In almost every run the sharpness
climbs until it reaches the ceiling of its own learning rate and then stays
close to $2/\eta$: near 5 for $\eta = 0.4$ and near 8 for $\eta = 0.25$.
Meanwhile the loss keeps falling by orders of magnitude, though not on
every step. The two runs differ only in $\eta$, yet they end in regions of
different curvature, each as sharp as its own learning rate allows.

<!-- WRITER: measured facts for this paragraph (two notebook runs plus probes of 10 x 10 and 50 starts per tab; ranges are over random starts). Initial sharpness 2.4 to 5.4 over the development draws, below 5 in all but about one start in twenty (keep "usually below 5"; a start above 5 at eta = 0.4 has its sharpness pushed down to 5.00); initial loss 0.25 to 2.4. eta = 0.4: every start ends at sharpness 4.99 to 5.01 (all probes 4.99 to 5.06), loss 2e-4 to 9e-4. eta = 0.25: every start ends at 7.6 to 8.05 (notebook runs 7.66 to 8.03), loss 4e-5 to 8e-4, i.e. a fall of three to four orders of magnitude. "though not on every step" now has its number: in a start that sits at the edge the loss rises on exactly 50% of the last 1,000 steps, i.e. on every second step (the period-two swing across the sharpest direction); in the other starts the swing has died out, the sharpness sits just below 2/eta and the loss rises on 0% (rarely in between, e.g. 35%). Which starts are still swinging varies from draw to draw (eta = 0.4: 6 to 10 of 10, 86 of 100 over ten draws; eta = 0.25: 2 to 9 of 10, 60 of 100), so do not quote a count of runs; state the two states. "In almost every run" can become "in every one of the ten starts" for reaching the ceiling. -->

Gradient descent stays at the edge because it corrects itself. When the
sharpness exceeds $2/\eta$, the iterate starts to oscillate along the
sharpest direction, as $x_2$ did on the valley, and this oscillation moves
the iterate toward flatter parts of the surface until the sharpness drops
back below the ceiling :cite:`Damian.Nichani.Lee.2023`. The quadratic
model predicts the oscillation but not what follows. The third-order term
of the Taylor expansion, which the quadratic model drops, turns the
oscillation into a drift toward lower sharpness. The quadratic analysis
misses this change of the surface under the iterate. Most results quoted in this
chapter assume monotone descent on a fixed surface, and the edge of
stability lies outside that assumption.

The start of training is delicate for a related reason. In large networks
the sharpness is often high at initialization and falls early in training,
the reverse of our small network. A learning rate
that suits the later surface can then diverge on the first steps, and
warmup keeps $\eta$ small until the sharpness has fallen
:cite:`Kalra.Barkeshli.2024` (:numref:`sec_scheduler`).

## Multivariate Gradient Descent

The same update works for a vector of parameters
$\mathbf{x} = (x_1, \ldots, x_d)$ and an objective
$f: \mathbb{R}^d \to \mathbb{R}$. The gradient collects the $d$ partial
derivatives,

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f(\mathbf{x})}{\partial x_1}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\right),$$

and each entry measures how fast $f$ changes when its coordinate alone
changes. The multivariate Taylor expansion has the same form as the
one-dimensional one,

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + O(\|\boldsymbol{\epsilon}\|^2).$$
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
$(h_1 x_1, h_2 x_2) = (0.2 x_1, 4 x_2)$. The rest of the chapter returns to
this valley, so we save it and its gradient.

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

To follow the iterates in the plane we write a helper that the rest of the
chapter reuses. `train_2d` starts at $(-5, -2)$ and applies an update rule
with the gradient `f_grad` for a given number of steps. Besides the two
coordinates and the gradient, the update rule receives and returns two
state variables, which the methods with memory in later sections
(momentum, Adam) use; gradient descent ignores them. Hyperparameters such
as the learning rate are passed through as keyword arguments. The plotting
utility `d2l.show_trace_2d` (:numref:`sec_utils`) draws the path over the
contour lines of the objective.

We run the same 30 steps at $\eta = 0.45$ as in
:numref:`sec_optimization-intro`, now drawn in the plane.

<!-- WRITER: train_2d and gd_2d now sit in the one helper cell above (structure 0.8, GD-8): introduce them with one sentence there and let the docstrings do the rest. The 30-step run in the plane (cell gd-multivariate-gradient-descent-3's run line, output x1 = -0.295265) is cut (GD-9): "We run the same 30 steps ..." and the paragraph after it ("Each step is perpendicular to the contour line ... after 30 steps x1 is still -0.3") describe a plot that no longer exists; the zigzag is plotted against the step count in sec_optimization-intro. -->

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

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon} + O(\|\boldsymbol{\epsilon}\|^3),$$
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

The iterate reaches the minimum to machine precision within ten steps, with
no learning rate chosen by hand.

<!-- WRITER: `newton` is now gradient descent with eta = 1 on f'/f'' (one line on top of `gd`) and the cell prints the iterates (identical in both tabs): x_t: 10 8 6 4 2.1 0.53 0.012 1.4e-07 0 0 0. Two phases for GD-11: four steps of almost exactly 1/c = 2 far out, then the error collapses (2.1, 0.53, 0.012, 1.4e-07, then exactly 0 in float32). The Newton run on x cos(cx) prints x_t: 10 5 24.99 27.78 26.8 26.83 ... 26.83. -->

Newton's method divides by the curvature, and the curvature of a nonconvex
function can be negative. Where it is, the step points uphill, toward a
maximum. We return to $f(x) = x \cos(cx)$ with $c = 0.15\pi$, which has a
minimum near $7.27$, close to the start.

```{.python .input #gd-newton-s-method-2}
%%tab pytorch, jax
xs = newton(f_cos_grad, f_cos_hess)
print('x_t:', ' '.join(f'{x:.4g}' for x in xs))
show_trace(f_cos, xs)
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

<!-- WRITER: the damped Newton run (cell gd-newton-s-method-3, output 7.269860) is cut (GD-13); the paragraph above ("Another is to bring back a learning rate ... gives up Newton's fast local rate") and the next one ("Damping is no guarantee") describe a run that no longer exists. `newton` no longer takes a step size eta. -->

Damping is no guarantee: it only makes a jump into a region of negative
curvature less likely. Two further facts about Newton's method matter
here, and the appendix proves both. First, near a minimum with positive
curvature it converges at least *quadratically*: the number of correct
digits roughly doubles with every step
(:numref:`subsec_mdl-why-not-newton`). Second, the method does not scale to
deep learning.

Storing the Hessian of a model with $d$ parameters takes
$O(d^2)$ memory and solving a linear system with it takes
$O(d^3)$ time. For $d = 10^9$ the Hessian has $10^{18}$ entries,
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

def steps_to_1pct(xs):  # First step with both coordinates within 1% of start
    return next(t for t, (x1, x2) in enumerate(xs)
                if abs(x1) <= 0.01 * 5 and abs(x2) <= 0.01 * 2)

plain = train_2d(gd_2d, valley_grad, steps=100, eta=2 / (h1 + h2))
precond = train_2d(precond_gd_2d, valley_grad, steps=10, eta=0.5)
print(f'steps to 1%: plain at its best rate {steps_to_1pct(plain)}, '
      f'preconditioned {steps_to_1pct(precond)}')
d2l.show_trace_2d(f_valley, precond)
```

Both coordinates now shrink by the same factor $1 - \eta = 0.5$ per step,
so the path runs straight to the minimum. Seven steps bring both within 1%,
against 47 for the best plain learning rate, and the count no longer
depends on the condition number. At $\eta = 1$ one step suffices: because
the Hessian of the valley is diagonal, the diagonal preconditioner is
Newton's method.

<!-- WRITER: the cell now prints "steps to 1%: plain at its best rate 47, preconditioned 7" (both tabs; it runs plain gradient descent at eta = 2/(h1 + h2) for 100 steps beside the 10 preconditioned steps), so "Seven steps" is printed (GD-17). -->

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

1. [code] **Learning-rate sweep.** Extend `gd` to stop as soon as
   $|x| < 10^{-3}$, and run it from $x = 1$ for at most 100 steps on
   $f(x) = x^2$, $f(x) = |x|^{1.5}$, and $f(x) = x^4$ at
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

$$f(x - \eta f'(x)) = f(x) - \eta f'(x)^2 + O(\eta^2 f'(x)^2)$$

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

<!-- WRITER: cell gd-one-dimensional-gradient-descent-4 is cut; this slide needs rebuilding without it (parabola runs cut, GD-2/GD-3). -->

Ten steps from $x = 10$ leave $10 \cdot 0.6^{10} \approx 0.06$.
:::

::: {.slide title="η = 0.05 Crawls; η = 1.1 Overshoots and Diverges"}
[Learning rate]{.kicker}

::: {.cols .vc}
::: {.col}
<!-- WRITER: cell gd-learning-rate-1 is cut; this slide needs rebuilding without it (parabola runs cut, GD-2). -->
:::
::: {.col}
<!-- WRITER: cell gd-learning-rate-2 is cut; this slide needs rebuilding without it (parabola runs cut, GD-2). -->
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

::: {.slide title="Classical Advice Fixes the Curvature, Then Picks the Step"}
[Edge of stability]{.kicker}

- Measure the *sharpness* $\lambda_{\max}$, choose $\eta < 2/\lambda_{\max}$.
- Full-batch training of real networks (Cohen et al., 2021): the sharpness
  **rises** during training until it reaches about $2/\eta$, then stays.
- The loss keeps falling, though not on every step.

Test: a 25-parameter $\tanh$ network fits 16 points of $\sin(3x)$, from one
random start, at $\eta = 0.4$ and $\eta = 0.25$.
:::

::: {.slide title="Sharpness Climbs to 2/η and Stays There"}
[Edge of stability]{.kicker}

@!optimization-intro-the-edge-of-stability-2

<!-- WRITER: the cell now runs ten random starts (shared by both rates) and prints a per-start table (start sharpness and loss; per eta: final sharpness, final loss, share of the last 1,000 steps with rising loss); the plot is sharpness / (2/eta) with a min-max band and reference 1. "Same start" here and "from one random start" on the previous slide are false. -->

Same start, two learning rates, final sharpness near 5 and near 8.
:::

::: {.slide title="Gradient Descent Corrects Its Own Overshoot"}
[Edge of stability]{.kicker}

- Sharpness above $2/\eta$: the iterate oscillates along the sharpest
  direction, as $x_2$ did on the valley.
- The quadratic model predicts the oscillation; the third-order term it
  drops turns it into a drift toward lower sharpness (Damian et al., 2023).

::: {.d2l-note}
Large networks often start sharp and flatten early, the reverse of this
small one: **warmup** keeps $\eta$ small until the sharpness has fallen.
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

<!-- WRITER: cell gd-multivariate-gradient-descent-3's run is cut; this slide needs rebuilding without it (third valley run cut, GD-9; the cell now only defines gd_2d inside gd-multivariate-gradient-descent-4). -->

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
<!-- WRITER: cell gd-newton-s-method-3 is cut; this slide needs rebuilding without it (damped Newton run cut, GD-13; the slide text on half steps is now false). -->
:::
:::

$x\cos(0.15\pi x)$: the full step lands at $x = 5$ ($f''$ small), jumps
to 25 ($f'' < 0$), and climbs to a maximum at 26.8. Half steps stay in the
basin and halve the distance to 7.27 per step.
:::

::: {.slide title="A Billion Parameters Need Four Exabytes of Hessian"}
[Newton's method]{.kicker}

- Hessian of $d$ parameters: $d^2$ entries; solve: $O(d^3)$.
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
| Newton | $\mathbf{H}^{-1}$ | exact Hessian, $O(d^2)$ memory |
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
