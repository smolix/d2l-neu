# Curvature Methods: Newton, Quasi-Newton, and Trust Regions
:label:`sec_mdl-curvature-methods`

Gradient descent never measures the curvature of the function it minimizes;
it only endures it. The steepest direction caps the step size, the flattest
direction then crawls, and the number of iterations grows with the condition
number $\kappa$, or with $\sqrt\kappa$ once momentum is added
(:numref:`subsec_mdl-momentum-acceleration`). On our running example
$\tfrac12(x_1^2 + 1000\,x_2^2)$, that came to $6{,}908$ steps of gradient
descent, or $315$ with momentum. Yet anyone can solve this problem by
inspection: divide each coordinate of the gradient by its curvature ($1$ or
$1000$) and a single step lands directly on the minimum. What can an
optimizer achieve when it accounts for curvature instead of merely surviving
it?

The classical answer is Newton's method, which predates gradient descent by
a century and a half. At each iterate, it fits a quadratic approximation to
the objective and jumps straight to the minimizer of that quadratic. In this
section we examine how fast that step converges, what it costs, where it
breaks down, and how much of its curvature information can be recovered from
gradients alone when second derivatives are unavailable. Near a minimum,
Newton's method is remarkably fast; for a neural network, however, forming
the Hessian is far too expensive, and on nonconvex functions the method is
drawn to saddle points just as readily as to minima. We then turn to two
remedies: quasi-Newton methods, which rebuild curvature from successive
gradient differences, and trust-region methods, which restrict each step to
a neighborhood where the quadratic model actually agrees with the function.

For training deep networks, these classical methods are used much less often
than first-order optimizers, and it is worth being candid about why. A full
Hessian is too large to form, and minibatch noise spoils curvature estimates
built from differences of gradients. Even so, the underlying ideas reappear
throughout deep learning: the per-coordinate scaling of Adam acts as a
diagonal stand-in for Newton's inverse Hessian
(:numref:`sec_mdl-adaptive-stochastic-methods`), and the damping added to
curvature matrices plays the same stabilizing role as a trust region. The
central result to keep in mind is the quadratic convergence of Newton's
method :eqref:`eq_mdl-opt-newton-error`; the subsection on quasi-Newton
methods may be skipped on a first reading. From
:numref:`sec_mdl-gradient-based-optimization`, we carry over two ingredients:
the Armijo condition for backtracking line search and the test function
$\tfrac12 x^2 + \cos y$, whose saddle point stalled gradient descent.

```{.python .input #mdl-curvature-methods-curvature-methods-newton-quasi-newton-and-trust-regions}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-curvature-methods-curvature-methods-newton-quasi-newton-and-trust-regions}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-curvature-methods-curvature-methods-newton-quasi-newton-and-trust-regions}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-curvature-methods-curvature-methods-newton-quasi-newton-and-trust-regions}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Newton's Method
:label:`subsec_mdl-why-not-newton`

Gradient descent builds a linear approximation to $f$ at the current point.
Because a linear function has no minimum, the algorithm cannot choose its own
step length and must rely on an external learning rate. A quadratic
approximation, by contrast, has a unique minimum whenever it curves upward,
giving us a step whose length is set automatically by the local geometry.

### The Newton Step

Newton's method lets the local curvature determine both the direction and the
length of the step (:numref:`subsec_gd-newton`). Writing
$\mathbf{g}_k = \nabla f(\mathbf{x}_k)$, we approximate $f$ near
$\mathbf{x}_k$ by its second-order Taylor expansion in the displacement
$\mathbf{p}$:

$$
m_k(\mathbf{p}) = f(\mathbf{x}_k) + \mathbf{g}_k^\top \mathbf{p} + \tfrac12\, \mathbf{p}^\top \nabla^2 f(\mathbf{x}_k)\, \mathbf{p}.
$$
:eqlabel:`eq_mdl-opt-newton-model`

When the Hessian $\nabla^2 f(\mathbf{x}_k)$ is positive definite, the model
$m_k$ is a strictly convex bowl with a unique minimizer, found by setting
$\nabla m_k(\mathbf{p}) = \mathbf{g}_k + \nabla^2 f(\mathbf{x}_k)\,\mathbf{p} = \mathbf{0}$.
Solving for $\mathbf{p}$ and moving there gives **Newton's method**:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \big(\nabla^2 f(\mathbf{x}_k)\big)^{-1} \nabla f(\mathbf{x}_k).
$$
:eqlabel:`eq_mdl-opt-newton-step`

Notice that no step size appears in :eqref:`eq_mdl-opt-newton-step`: the
inverse Hessian rescales each direction according to its curvature. More
broadly, Newton's direction belongs to the family of preconditioned steps
$\mathbf{d} = -\mathbf{P}^{-1}\nabla f$ with a positive definite matrix
$\mathbf{P}$ (here $\mathbf{P} = \nabla^2 f$). Every such direction is a
descent direction, since
$\nabla f^\top \mathbf{d} = -\nabla f^\top \mathbf{P}^{-1}\nabla f < 0$.
In fact, Newton's direction is simply the direction of steepest descent when
step lengths are measured in the norm induced by the Hessian rather than the
Euclidean norm (exercise
:exref:`ex_mdl-gradient-based-optimization-show-that-is-de`).

On a positive definite quadratic $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$,
the Taylor model is exact, so a single Newton step lands immediately on the
minimizer:
$\mathbf{x}_1 = \mathbf{x}_0 - \mathbf{A}^{-1}\mathbf{A}\mathbf{x}_0 = \mathbf{0}$,
regardless of the condition number $\kappa$. The reason goes deeper than
matrix cancellation. Newton's method is **affine-invariant**, meaning that
its trajectory does not depend on the choice of coordinates. Under any
invertible linear change of variables $\mathbf{x} = \mathbf{T}\mathbf{y}$,
running Newton's method on $f(\mathbf{T}\mathbf{y})$ produces the exact
images of Newton's iterates for $f$. Gradient descent, by contrast, changes
its path under every non-orthogonal $\mathbf{T}$, while a scalar multiple of
an orthogonal matrix merely rescales the effective step size (exercise
:exref:`ex_mdl-gradient-based-optimization-newton-show-that-any`). Because
some linear transformation turns any positive definite quadratic into a
perfect sphere, the condition number cannot affect Newton's method. A long
valley is, to Newton's method, a round bowl drawn on stretched paper.

### Quadratic Convergence

For a general non-quadratic objective, the Taylor model is only a local
approximation, so one step is no longer enough. What survives is remarkable
speed. Once the iterate enters a neighborhood of a minimizer, Newton's
method converges much faster than geometrically, and the proof requires
little more than the fundamental theorem of calculus. Let
$\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^\star$ denote the error at
iteration $k$.

**Proposition (quadratic convergence).** *Let $\nabla f(\mathbf{x}^\star) = \mathbf{0}$,
and suppose that on a ball around $\mathbf{x}^\star$ the Hessian satisfies
$\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ and
$\|\nabla^2 f(\mathbf{x}) - \nabla^2 f(\mathbf{y})\| \le L_2\|\mathbf{x} - \mathbf{y}\|$.
If $\mathbf{x}_k$ lies in the ball, then*

$$
\|\mathbf{e}_{k+1}\| \;\le\; \frac{L_2}{2\mu}\, \|\mathbf{e}_k\|^2.
$$
:eqlabel:`eq_mdl-opt-newton-error`

**Proof.** Abbreviate $\nabla^2 f_k = \nabla^2 f(\mathbf{x}_k)$. Because
$\nabla f(\mathbf{x}^\star) = \mathbf{0}$, subtracting $\mathbf{x}^\star$
from the Newton update yields
$\mathbf{e}_{k+1} = \mathbf{e}_k - (\nabla^2 f_k)^{-1}\mathbf{g}_k = (\nabla^2 f_k)^{-1}\big[\nabla^2 f_k\,\mathbf{e}_k - (\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star))\big]$.
Applying the fundamental theorem of calculus along the line segment from
$\mathbf{x}^\star$ to $\mathbf{x}_k$ expresses the gradient difference as
$\int_0^1 \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\,\mathbf{e}_k\,dt$,
so the bracketed term becomes
$\int_0^1 \big[\nabla^2 f_k - \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\big]\mathbf{e}_k\,dt$.
Since $\mathbf{x}_k$ and $\mathbf{x}^\star + t\mathbf{e}_k$ are separated by
$(1 - t)\mathbf{e}_k$, the Lipschitz bound on the Hessian gives this integral
norm at most
$\int_0^1 L_2(1 - t)\|\mathbf{e}_k\|^2\,dt = \tfrac{L_2}{2}\|\mathbf{e}_k\|^2$.
Combining this with $\|(\nabla^2 f_k)^{-1}\| \le 1/\mu$ completes the
proof. $\blacksquare$

To read what :eqref:`eq_mdl-opt-newton-error` implies, multiply both sides by
$L_2/(2\mu)$:
$\tfrac{L_2}{2\mu}\|\mathbf{e}_{k+1}\| \le \big(\tfrac{L_2}{2\mu}\|\mathbf{e}_k\|\big)^2$.
The scaled error squares at every step. If it starts below $1$, it shrinks
monotonically and every subsequent iterate remains in the ball. Once it drops
below $0.1$, it plunges to $10^{-2}$, $10^{-4}$, $10^{-8}$, roughly doubling
the number of accurate digits at each iteration. Compare that with gradient
descent on our running example, which needs about $1{,}150$ steps to gain a
single digit. The catch is written directly into the hypotheses: this
quadratic rate is strictly local and guarantees nothing until the iterate is
already close to $\mathbf{x}^\star$.

Far from $\mathbf{x}^\star$, the pure Newton step can overshoot even when
the Hessian is positive definite. The standard safeguard is **damping**:
update via $\mathbf{x}_k + \alpha_k\mathbf{p}_k$ along the Newton direction
$\mathbf{p}_k = -(\nabla^2 f(\mathbf{x}_k))^{-1}\mathbf{g}_k$, choosing
$\alpha_k \le 1$ by backtracking until the Armijo condition
$f(\mathbf{x}_k + \alpha\mathbf{p}_k) \le f(\mathbf{x}_k) + c\,\alpha\,\mathbf{g}_k^\top\mathbf{p}_k$
holds (the general-direction form of :eqref:`eq_mdl-opt-armijo`). Near a
minimizer with a positive definite Hessian, the full step $\alpha = 1$
automatically passes this test whenever $c < \tfrac12$, so damping adds no
overhead once quadratic convergence kicks in.

To test these predictions in code, we use $\ell_2$-regularized logistic
regression on $n = 200$ synthetic examples in $d = 5$ dimensions,

$$
f(\mathbf{w}) = \frac1n \sum_{i=1}^n \log\big(1 + \exp(-y_i\,\mathbf{x}_i^\top \mathbf{w})\big) + \frac{\nu}{2}\|\mathbf{w}\|^2,
\qquad \nu = 0.1,
$$

with binary labels $y_i = \pm 1$ sampled from a logistic model (we write
$\nu$ for the weight-decay coefficient to reserve $\lambda$ for eigenvalues).
The setup cell below also computes a reference optimum $\mathbf{w}^\star$
using $4{,}000$ gradient steps of size $\eta = 0.5$, which is safely below
$2/L$ because $L = \lambda_{\max}(\mathbf{X}^\top\mathbf{X})/(4n) + \nu$ is
roughly $0.4$ here (exercise
:exref:`ex_mdl-gradient-based-optimization-smoothness-constants`).

```{.python .input #mdl-gradient-based-optimization-newton-s-method-and-its-computational-cost-1}
rng = np.random.default_rng()
n, d, nu = 200, 5, 0.1
w_true = np.array([1.0, -2.0, 1.5, 0.0, 2.0])
X = rng.normal(size=(n, d))
y = np.where(rng.random(n) < 1 / (1 + np.exp(-X @ w_true)), 1.0, -1.0)

def loss(w, X, y, nu):
    return np.logaddexp(0, -y * (X @ w)).mean() + nu / 2 * (w @ w)

def grad_rows(w, X, y, nu):                   # row i: gradient of example i
    s = 1 / (1 + np.exp(y * (X @ w)))         # sigmoid(-y_i x_i^T w)
    return -(y * s)[:, None] * X + nu * w

w_star = np.zeros(d)
for _ in range(4000):                         # full-batch reference optimum
    w_star = w_star - 0.5 * grad_rows(w_star, X, y, nu).mean(axis=0)
```

We can now test both predictions directly. On our running quadratic
$\mathrm{diag}(1, 1000)$, a single Newton step should replace the
$\log 10^6/\log(1001/999) \approx 6{,}908$ steps required by gradient descent
at its optimal fixed step size (:numref:`subsec_mdl-momentum-acceleration`).
On the non-quadratic logistic loss, the exponent of the gradient norm should
roughly double at every iteration once the local phase begins.

```{.python .input #gradient-based-optimization-newton}
A2 = np.diag([1.0, 1000.0])                   # the running example, kappa = 1000
x = np.array([1.0, 1.0])
x_new = x - np.linalg.solve(A2, A2 @ x)       # one Newton step
rate = (1000 - 1) / (1000 + 1)
print(f'Newton on the quadratic: |x_1| = {np.linalg.norm(x_new):.1e} after one step')
print(f'GD at the optimal step contracts by {rate:.4f}; it needs '
      f'{int(np.ceil(np.log(1e-6) / np.log(rate)))} steps for |x_k| <= 1e-6 |x_0|')
w = np.zeros(d)                               # Newton on the logistic loss
for it in range(6):
    s = 1 / (1 + np.exp(y * (X @ w)))
    g = grad_rows(w, X, y, nu).mean(axis=0)
    hess = (X * (s * (1 - s))[:, None]).T @ X / n + nu * np.eye(d)
    w = w - np.linalg.solve(hess, g)
    print(f'Newton iteration {it + 1}:  |grad| = {np.linalg.norm(g):.1e}')
print(f'max |w - w_star| = {np.abs(w - w_star).max():.1e}')
```

Both predictions hold. On the quadratic, a single Newton step drives the
error to zero, whereas optimal fixed-step gradient descent contracts by only
$0.998$ per step and requires $6{,}908$ iterations. On the logistic loss, the
gradient norm starts at a few tenths and drops below $10^{-8}$ by the fifth
iteration, its exponent roughly doubling once it becomes small. After just
six iterations, Newton's iterate agrees with the reference optimum to roughly
$10^{-15}$, matching the precision of the reference itself. Six linear solves
have accomplished what took $4{,}000$ gradient steps.

### The Cost of a Step

Unfortunately, comparing iteration counts alone flatters Newton's method. The
Hessian contains $d^2$ entries, and solving the linear system in
:eqref:`eq_mdl-opt-newton-step` generally requires $O(d^3)$ operations per
step. For $d = 5$ this cost is negligible, but for a neural network with even
a modest $10^8$ parameters, the Hessian has $10^{16}$ entries, ruling out
exact Newton's method before the first step is ever taken
(:numref:`subsec_gd-newton`). Practical alternatives either approximate the
Hessian from gradients, as in the quasi-Newton methods below, or use cheap
diagonal and structured preconditioners
(:numref:`sec_mdl-adaptive-stochastic-methods`).

### Saddle Points

Computational cost is not the only drawback of Newton's method, nor is it the
most instructive one. Because the Newton step solves $\nabla m_k = \mathbf{0}$,
it jumps to the stationary point of the quadratic model whether that point is
a minimum, a maximum, or a saddle. Return to the function
$f(x, y) = \tfrac12 x^2 + \cos y$ from
:numref:`sec_mdl-gradient-based-optimization`, which has a saddle at the
origin and a minimum of $-1$ at $(0, \pi)$, and start at $(0.5, 0.3)$. Here
the gradient is $(0.5, -\sin 0.3) = (0.5, -0.296)$, and the Hessian
$\mathrm{diag}(1, -\cos 0.3) = \mathrm{diag}(1, -0.955)$ is indefinite: the
surface curves upward along $x$ and downward along $y$. Which way does a
Newton step move $y$ from this point, compared with a plain gradient step of
size $\eta = 1$?

```{.python .input #mdl-gradient-based-optimization-newton-s-method-1}
def f_s(v):
    return 0.5 * v[0]**2 + np.cos(v[1])

def grad_s(v):
    return np.array([v[0], -np.sin(v[1])])

def hess_s(v):
    return np.diag([1.0, -np.cos(v[1])])

x_n = x_g = np.array([0.5, 0.3])
ys_n, ys_g = [], []
for _ in range(6):
    x_n = x_n - np.linalg.solve(hess_s(x_n), grad_s(x_n))    # Newton
    x_g = x_g - 1.0 * grad_s(x_g)                            # GD, eta = 1
    ys_n.append(x_n[1])
    ys_g.append(x_g[1])
print('Newton y:', np.array(ys_n[:3]), f'  f = {f_s(x_n):.4f}')
print('GD     y:', np.array(ys_g).round(4), f'  f = {f_s(x_g):.4f}')
```

Newton's first step sends $x$ to $0$ and $y$ to $0.3 - \tan 0.3 = -0.0093$;
the second sends $y$ to $2.7 \times 10^{-7}$. We are seeing Newton's rapid
local convergence (here even cubic, since $y - \tan y \approx -y^3/3$), driving
the iterate straight into the saddle $(0, 0)$ with $f = 1$. Gradient descent
moves $y$ in the opposite direction, through $0.60$, $1.16$, $2.07$, $2.95$,
$3.140$, and $3.1416$, reaching the minimum $(0, \pi)$ with $f = -1$. The
richer model has led to the worse point. Because the curvature along $y$ is
negative, the stationary point of the quadratic model is a saddle, and
Newton's method is blind to the difference between a saddle and a minimum.
A trust region (:numref:`subsec_mdl-trust-region`) fixes this flaw.

## Quasi-Newton Methods

Newton's method requires explicit second derivatives. Can we instead infer
curvature from the gradients we are already computing? When the gradient
changes rapidly along a step, the objective must have large curvature in that
direction. An optimizer that records these changes can gradually assemble a
model of the Hessian without ever evaluating a second derivative.

### The Secant Method

In one dimension the idea is immediate: the slope of the chord connecting
the last two gradients estimates $f''$,

$$
h_k = \frac{f'(x_k) - f'(x_{k-1})}{x_k - x_{k-1}},
\qquad
x_{k+1} = x_k - \frac{f'(x_k)}{h_k}.
$$

This is the **secant method**, which is simply Newton's step with $f''(x_k)$
replaced by a finite difference of gradients. Consider $f(x) = e^x - 2x$,
whose minimizer is $\log 2 \approx 0.6931$, starting from $x_0 = 0$ and
$x_1 = 1$. The first secant slope is $h_1 = (e - 2) - (1 - 2) = e - 1 \approx 1.718$,
giving $x_2 = 1 - (e - 2)/1.718 \approx 0.582$. From there the errors shrink
as $1.1 \times 10^{-1}$, $1.6 \times 10^{-2}$, $9.3 \times 10^{-4}$,
$7.7 \times 10^{-6}$, and $3.6 \times 10^{-9}$. Each error is roughly half
the product of the previous two (the factor of one-half comes from
$f'''/(2f'')$ at the minimizer), so the number of correct digits grows by a
factor approaching $1.6$ per step. That is slightly slower than Newton's
factor of $2$, yet each iteration requires only a single gradient evaluation
and no second derivative. On a one-dimensional quadratic
$\tfrac{\lambda}{2}x^2$, the secant slope equals $\lambda$ exactly, and the
step from $x_1$ lands directly on the minimizer.

In $d$ dimensions, the analogous secant condition is a single vector
equation, which is far from enough to pin down a $d \times d$ matrix: one
step reveals curvature along one direction only. We need a principled rule to
fill in the remaining directions.

### BFGS
:label:`subsec_mdl-quasi-newton`

A **quasi-Newton** method maintains an approximation $\mathbf{M}_k$ to the
inverse Hessian and updates it after each step using the observed change in
the gradient. After moving from $\mathbf{x}_k$ to $\mathbf{x}_{k+1}$, we
record the secant pair

$$
\mathbf s_k=\mathbf x_{k+1}-\mathbf x_k,
\qquad
\mathbf y_k=\nabla f(\mathbf x_{k+1})-\nabla f(\mathbf x_k).
$$

On a quadratic $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, the gradient
difference satisfies $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$, so the true
inverse Hessian maps $\mathbf{y}_k$ back to $\mathbf{s}_k$. We require the
updated approximation $\mathbf{M}_{k+1}$ to obey the same **secant
equation**:

$$
\mathbf{M}_{k+1}\,\mathbf{y}_k = \mathbf{s}_k .
$$
:eqlabel:`eq_mdl-opt-secant`

Among all symmetric matrices satisfying :eqref:`eq_mdl-opt-secant`, **BFGS**
(named after Broyden, Fletcher, Goldfarb, and Shanno) chooses the one closest
to the previous estimate $\mathbf{M}_k$ in a suitably weighted Frobenius
norm:

$$
\mathbf{M}_{k+1}
=(\mathbf{I}-\gamma_k\mathbf s_k\mathbf y_k^\top)\,\mathbf{M}_k\,
 (\mathbf{I}-\gamma_k\mathbf y_k\mathbf s_k^\top)
 +\gamma_k\mathbf s_k\mathbf s_k^\top,
\qquad
\gamma_k=\frac{1}{\mathbf y_k^\top\mathbf s_k}.
$$
:eqlabel:`eq_mdl-opt-bfgs`

Even without deriving :eqref:`eq_mdl-opt-bfgs`, we can read off how it works
(dropping the index $k$ on the pair for clarity). The right factor
$\mathbf{I} - \gamma\mathbf{y}\mathbf{s}^\top$ sends $\mathbf{y}$ to zero,
because $\gamma\,\mathbf{s}^\top\mathbf{y} = 1$, while leaving every vector
orthogonal to $\mathbf{s}$ untouched. Thus the first term erases what
$\mathbf{M}_k$ previously predicted along $\mathbf{y}$, and the second term
writes in the new observation:
$\mathbf{M}_{k+1}\mathbf{y} = \gamma\,\mathbf{s}\,(\mathbf{s}^\top\mathbf{y}) = \mathbf{s}$,
satisfying the secant equation.

Just as importantly, the update preserves positive definiteness, ensuring
that the next search direction $-\mathbf{M}_{k+1}\nabla f(\mathbf{x}_{k+1})$
remains a descent direction. For any nonzero vector $\mathbf{z}$, set
$\mathbf{w} = (\mathbf{I} - \gamma\mathbf{y}\mathbf{s}^\top)\mathbf{z}$, so
that
$\mathbf{z}^\top\mathbf{M}_{k+1}\mathbf{z} = \mathbf{w}^\top\mathbf{M}_k\mathbf{w} + \gamma(\mathbf{s}^\top\mathbf{z})^2$.
When $\mathbf{M}_k \succ 0$ and $\gamma > 0$, both terms are nonnegative and
cannot vanish simultaneously: $\mathbf{w} = \mathbf{0}$ forces
$\mathbf{z} = \gamma(\mathbf{s}^\top\mathbf{z})\,\mathbf{y}$, which requires
$\mathbf{s}^\top\mathbf{z} \neq 0$. The requirement $\gamma > 0$ is the
**curvature condition** $\mathbf{y}_k^\top\mathbf{s}_k > 0$, meaning that the
directional slope increased along the step. It holds automatically on any
strictly convex objective; on general objectives, the line search enforces it
via a second check known as the Wolfe condition :cite:`Nocedal.Wright.2006`.

How quickly do secant pairs reconstruct the true curvature? Let us run BFGS
from $(1, 1)$ on the two-dimensional quadratic $\mathrm{diag}(1, 10)$,
starting with $\mathbf{M}_0 = \mathbf{I}$ and using the exact line search
$\alpha = -\mathbf{g}^\top\mathbf{p}/(\mathbf{p}^\top\mathbf{A}\mathbf{p})$,
and inspect both $\mathbf{x}$ and $\mathbf{M}$ after each step.

```{.python .input #mdl-gradient-based-optimization-bfgs-1}
def bfgs_update(M, s, y):
    gamma = 1.0 / (y @ s)
    V = np.eye(len(s)) - gamma * np.outer(s, y)
    return V @ M @ V.T + gamma * np.outer(s, s)

A = np.diag([1.0, 10.0])
x, M = np.array([1.0, 1.0]), np.eye(2)
for k in range(2):
    g = A @ x
    p = -M @ g
    alpha = -(g @ p) / (p @ A @ p)            # exact line search
    s = alpha * p
    M = bfgs_update(M, s, A @ s)              # y = A s on a quadratic
    x = x + s
    print(f'step {k + 1}: x = {x.round(4)},  M = {M.round(4).tolist()}')
print(f'A^-1 = {np.linalg.inv(A).tolist()},  '
      f'max |M_2 - A^-1| = {np.abs(M - np.linalg.inv(A)).max():.0e}')
```

After a single step, $\mathbf{M}_1 \approx \mathrm{diag}(1.009, 0.100)$ is
already close to $\mathbf{A}^{-1}$ along the steep direction it has just
probed. After the second step, $\mathbf{M}_2 = \mathbf{A}^{-1}$ up to
floating-point rounding and $\mathbf{x}_2 = \mathbf{0}$. This two-dimensional
behavior generalizes cleanly: on any $d$-dimensional positive definite
quadratic with exact line searches, each secant equation remains satisfied
after subsequent updates, and the steps are mutually $\mathbf{A}$-conjugate
($\mathbf{s}_i^\top\mathbf{A}\mathbf{s}_j = 0$ for $i \neq j$). After $d$
steps, $\mathbf{M}_d\mathbf{A} = \mathbf{I}$ and BFGS reaches the exact
minimizer (exercise
:exref:`ex_mdl-gradient-based-optimization-bfgs-quadratic`).

Finite termination, however, holds specifically for quadratics with exact
line searches and should not be taken as a general guarantee. On non-quadratic
objectives, line searches are inexact, the true Hessian varies from step to
step, and stochastic gradients corrupt $\mathbf{y}_k$: the difference between
two minibatch gradients is often dominated by sampling noise rather than
curvature. In addition, full BFGS stores a dense $d \times d$ matrix.
**L-BFGS** avoids that memory cost by retaining only the $m$ most recent
secant pairs and applying the resulting inverse-Hessian approximation (a
scaled identity plus a correction of rank at most $2m$) in $O(md)$ time and
memory :cite:`Liu.Nocedal.1989`. It works very well on deterministic,
medium-scale problems, though it remains awkward when fresh minibatch noise
makes gradient differences unreliable.

## Trust Regions
:label:`subsec_mdl-trust-region`

Newton's method went astray at $(0.5, 0.3)$ because it trusted its local
quadratic model across the entire space. That model was built from derivatives
at a single point and accurately reflects $f$ only nearby. A trust-region
method takes this limitation literally: it keeps the full curvature
information, including directions of negative curvature, and instead bounds the
step to a region where the model can be trusted.

### The Subproblem

At $(0.5, 0.3)$, the quadratic model of $\tfrac12 x^2 + \cos y$ has
curvature $-0.955$ along $y$. Because it curves downward without bound,
asking for its unconstrained minimum makes no sense. A **trust-region
method** instead minimizes the quadratic model inside a ball of radius
$\Delta_k$, which represents how far we believe the approximation holds:

$$
\min_{\|\mathbf p\|\le\Delta_k}
 m_k(\mathbf p)
 =f(\mathbf x_k)+\nabla f(\mathbf x_k)^\top\mathbf p
  +\tfrac12\mathbf p^\top \mathbf{B}_k\mathbf p.
$$
:eqlabel:`eq_mdl-opt-trust-region`

Here $\mathbf{B}_k$ is either the exact Hessian or a symmetric approximation
to it. Restricting $\mathbf{p}$ to a closed ball guarantees that a minimizer
exists even when $\mathbf{B}_k$ has negative eigenvalues, because any
continuous function achieves a minimum on a compact set. Better still, the
exact solutions of this constrained problem have a clean characterization.

**Proposition (solution of the subproblem).** *A step $\mathbf{p}$ with
$\|\mathbf{p}\| \le \Delta$ solves :eqref:`eq_mdl-opt-trust-region` if and only
if there is a $\tau \ge 0$ with*

$$
(\mathbf{B} + \tau\mathbf{I})\,\mathbf{p} = -\mathbf{g},
\qquad \mathbf{B} + \tau\mathbf{I} \succeq \mathbf{0},
\qquad \tau\,(\Delta - \|\mathbf{p}\|) = 0.
$$

This result is Theorem 4.1 of :citet:`Nocedal.Wright.2006`; exercise
:exref:`ex_mdl-gradient-based-optimization-trust-region-saddle` proves the
sufficiency of these conditions. Geometrically, the trust region adds a
uniform shift $\tau$ to every eigenvalue of $\mathbf{B}$, just enough to
make the shifted quadratic convex. Two cases arise. When $\mathbf{B}$ is
positive definite and the unconstrained Newton step already lies inside the
ball, we have $\tau = 0$ and simply take the Newton step. Otherwise the
solution lies on the boundary $\|\mathbf{p}\| = \Delta$, and $\tau$ is the
root of the scalar equation
$\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ over
$\tau > -\lambda_{\min}(\mathbf{B})$, where the left-hand side decreases
strictly as $\tau$ grows.

This scalar equation fails to have a root only in the degenerate *hard case*,
when $\mathbf{g}$ is orthogonal to the eigenvectors associated with the
smallest eigenvalue of $\mathbf{B}$ (explored in part d of exercise
:exref:`ex_mdl-gradient-based-optimization-trust-region-saddle`). As we will
see in :numref:`sec_mdl-constrained-optimization-duality`, the shift $\tau$
is simply the Lagrange multiplier enforcing the constraint
$\tfrac12\|\mathbf{p}\|^2 \le \tfrac12\Delta^2$.

Between iterations, the radius $\Delta_k$ adapts according to how accurately
the model predicted the actual drop in $f$. After solving
:eqref:`eq_mdl-opt-trust-region`, we form the ratio of actual decrease to
predicted decrease:

$$
r_k=
\frac{f(\mathbf x_k)-f(\mathbf x_k+\mathbf p_k)}
     {m_k(\mathbf 0)-m_k(\mathbf p_k)}.
$$
:eqlabel:`eq_mdl-opt-trust-ratio`

A ratio near $1$ indicates that the quadratic model matched the objective
well. In the implementation below, we accept the step whenever $r_k > 0.1$.
Independently, we shrink the radius to $\Delta \leftarrow \Delta/4$ whenever
$r_k < \tfrac14$, and we double it when $r_k > \tfrac34$ and the step hit
the boundary $\|\mathbf{p}_k\| = \Delta_k$.

### The Trust-Region Step at a Saddle

Let us work out one trust-region step at $(0.5, 0.3)$ by hand with
$\Delta = 0.5$. At this point $\mathbf{g} = (0.5, -0.296)$ and
$\mathbf{B} = \mathrm{diag}(1, -0.955)$. Making
$\mathbf{B} + \tau\mathbf{I}$ positive semidefinite requires
$\tau \ge 0.955$, so $\tau = 0$ is impossible and the step must lie on the
boundary of the ball. The optimality system gives
$(1 + \tau)p_1 = -0.5$ and $(\tau - 0.955)p_2 = 0.296$ subject to
$p_1^2 + p_2^2 = 0.25$, whose solution is $\tau \approx 1.60$ and
$\mathbf{p} \approx (-0.19, 0.46)$. Moving to $(0.31, 0.76)$ lowers $f$ from
$1.080$ to $0.771$. Because the model predicted a decrease of $0.316$ and
the actual decrease is $0.309$, the ratio is $r = 0.98$, so the step is
accepted and the radius doubles. Crucially, the trust-region step moves
*away* from the saddle along the negative-curvature direction where the
model drops, whereas the unconstrained Newton step $(-0.5, -0.31)$ marched
straight toward it.

To see where this update rule leads over several iterations, we implement
`tr_step` via bisection on $\tau$ and run six steps from $(0.5, 0.3)$,
printing the iterate, objective value, step norm, shift $\tau$, agreement
ratio $r$, and current radius $\Delta$.

```{.python .input #mdl-gradient-based-optimization-the-trust-region-step-at-a-saddle-1}
def tr_step(g, B, Delta):
    """Minimize g p + p B p / 2 over |p| <= Delta; return (p, tau)."""
    eig_min = np.linalg.eigvalsh(B).min()
    if eig_min > 0 and np.linalg.norm(np.linalg.solve(B, -g)) <= Delta:
        return np.linalg.solve(B, -g), 0.0     # Newton step fits: tau = 0
    lo = hi = max(0.0, -eig_min)
    hi += 1e3
    for _ in range(100):                       # bisection for |p(tau)| = Delta
        tau = (lo + hi) / 2
        p = np.linalg.solve(B + tau * np.eye(len(g)), -g)
        lo, hi = (tau, hi) if np.linalg.norm(p) > Delta else (lo, tau)
    return p, tau
```

```{.python .input #mdl-gradient-based-optimization-the-trust-region-step-at-a-saddle-2}
x, Delta = np.array([0.5, 0.3]), 0.5
print(' k        x            f      |p|    tau      r  Delta')
for k in range(6):
    g, B = grad_s(x), hess_s(x)
    p, tau = tr_step(g, B, Delta)
    r = (f_s(x) - f_s(x + p)) / -(g @ p + 0.5 * p @ B @ p)
    print(f'{k:2d} ({x[0]:6.3f},{x[1]:6.3f}) {f_s(x):7.4f} '
          f'{np.linalg.norm(p):6.3f} {tau:6.3f} {r:6.2f} {Delta:5.2f}')
    if r > 0.1:                                # accept the step
        x = x + p
    if r < 0.25:
        Delta /= 4                             # poor prediction: shrink
    elif r > 0.75 and np.linalg.norm(p) > 0.99 * Delta:
        Delta *= 2                             # good, and on the boundary
print(f'final x = ({x[0]:.4f}, {x[1]:.4f}),  f = {f_s(x):.4f}')
```

The first row matches our hand calculation. The radius doubles from $0.5$ to
$1$ and then to $2$, while the third step (with ratio $0.41$) is accepted
without changing the radius. By the fourth iterate, $(0.04, 3.75)$, the
algorithm has entered the basin where the Hessian is positive definite. From
that point on, the unconstrained Newton step fits inside the trust-region
ball with $\tau = 0$, so the method seamlessly turns into Newton's method and
converges quadratically to the minimum $(0, \pi)$ with $f = -1$. Six
iterations are enough; :numref:`fig_mdl-opt-trust-region-saddle` contrasts
both trajectories.

![Newton's method and a trust-region method from $(0.5,0.3)$ on $\tfrac12 x^2 + \cos y$, where the Hessian is $\mathrm{diag}(1,-0.96)$. (a) The local quadratic model forms a saddle surface: Newton's step moves to its stationary point, whereas minimizing the model over the ball of radius $\Delta = 0.5$ follows the negative-curvature direction away from the saddle. (b) Newton's method converges quadratically to the saddle $(0,0)$ with $f = 1$, while the trust-region method (shown with its initial radius) reaches the minimum $(0,\pi)$ with $f = -1$ in six steps as its radius expands through $0.5, 1, 2$.](../img/mdl-opt-trust-region-saddle.svg)
:label:`fig_mdl-opt-trust-region-saddle`

Solving the trust-region subproblem exactly requires factoring
$\mathbf{B}_k + \tau\mathbf{I}$ for each trial value of $\tau$. Large-scale
implementations instead solve it approximately using Hessian--vector
products, each costing roughly the same as a single gradient evaluation
(:numref:`sec_mdl-matrix-calculus-autodiff`). Even with that speedup, neither
BFGS nor trust-region methods replace stochastic gradient descent on large,
noisy deep learning objectives.

The saddle of $\tfrac12 x^2 + \cos y$ has now tripped up two optimizers:
gradient descent stalled at it, and Newton's method ran straight into it. A
trust region escapes, but only by checking each local model against the true
objective. Yet there is a large class of functions where none of these
precautions are needed: functions with no saddles, no spurious basins, and
where every stationary point is a global minimum. Identifying those
functions is the subject of :numref:`sec_mdl-convexity`.

## Summary

Newton's method minimizes a local quadratic model of $f$ at each iterate. It
solves any positive definite quadratic in a single step regardless of the
condition number, and near a minimizer with a positive definite,
Lipschitz-continuous Hessian it squares the error at every iteration. That
speed comes with two obstacles: each step requires $O(d^2)$ memory and
$O(d^3)$ operations, far beyond what a neural network can afford, and the
unconstrained step is attracted to saddle points just as strongly as to
minima. BFGS avoids explicit second derivatives by reconstructing the inverse
Hessian from past step and gradient differences, recovering it completely in
$d$ steps on a $d$-dimensional quadratic. Trust-region methods make any
quadratic model safe to use by confining each step to a ball, comparing the
predicted and actual decreases in $f$, and expanding or shrinking the ball
accordingly.

## Exercises

1. **Reparametrization and preconditioning.** Let $\mathbf{T}$ be an invertible
   matrix and $h(\mathbf{y}) = f(\mathbf{T}\mathbf{y})$, so that
   $\mathbf{x} = \mathbf{T}\mathbf{y}$ is a change of coordinates.
    1. Express $\nabla h$ and $\nabla^2 h$ through the derivatives of $f$.
       Show that Newton's iterates :eqref:`eq_mdl-opt-newton-step` for $h$,
       started at $\mathbf{y}_0 = \mathbf{T}^{-1}\mathbf{x}_0$, satisfy
       $\mathbf{y}_k = \mathbf{T}^{-1}\mathbf{x}_k$, where $\mathbf{x}_k$ are
       Newton's iterates for $f$.
    1. Run gradient descent with step $\eta$ on $h$ and map every iterate
       back by $\mathbf{x}_k = \mathbf{T}\mathbf{y}_k$. Show that the mapped iterates
       follow $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{P}^{-1} \nabla f(\mathbf{x}_k)$
       for a positive definite matrix $\mathbf{P}$, and find $\mathbf{P}$. By
       problem :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b
       this direction is the steepest one in the norm $\|\cdot\|_{\mathbf{P}}$.
       What is $\|\mathbf{x}\|_{\mathbf{P}}$ in the coordinates $\mathbf{y}$?
    1. For $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top \mathbf{A}\,\mathbf{x}$ with
       $\mathbf{A} = \mathrm{diag}(1, 1000)$, the $\kappa = 1000$ quadratic of the
       Newton subsection, find a diagonal $\mathbf{T}$ for which gradient
       descent on $h$ with $\eta = 1$ reaches the minimizer in one step. What
       does constructing such a $\mathbf{T}$ require for a general positive
       definite $\mathbf{A}$?
   :exlabel:`ex_mdl-gradient-based-optimization-newton-show-that-any`

1. [code] **BFGS on a quadratic.** ● Let
   $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x} - \mathbf{b}^\top\mathbf{x}$
   with $\mathbf{A}$ positive definite, and run BFGS
   :eqref:`eq_mdl-opt-bfgs` with exact line searches from
   $\mathbf{M}_0 = \mathbf{I}$.
    1. Show that an exact line search along $\mathbf{p}_k$ gives
       $\nabla f(\mathbf{x}_{k+1})^\top\mathbf{p}_k = 0$, and that
       $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$.
    1. Show the hereditary property: if $\mathbf{M}_k\mathbf{y}_j = \mathbf{s}_j$
       for all $j < k$ and the steps $\mathbf{s}_0, \ldots, \mathbf{s}_{k-1}$,
       $\mathbf{s}_k$ are $\mathbf{A}$-conjugate, then
       $\mathbf{M}_{k+1}\mathbf{y}_j = \mathbf{s}_j$ for all $j \le k$. (Hint:
       $\mathbf{s}_j^\top\mathbf{y}_k = \mathbf{s}_j^\top\mathbf{A}\mathbf{s}_k = 0$
       for $j < k$.) Conclude that after $d$ linearly independent steps
       $\mathbf{M}_d\mathbf{A} = \mathbf{I}$.
    1. Run the BFGS loop of the text in $d = 6$ dimensions, on a randomly
       rotated quadratic with eigenvalues from $1$ to $1000$. Print
       $\max_{j \le k}\|\mathbf{M}_{k+1}\mathbf{y}_j - \mathbf{s}_j\|$
       after each step, and confirm that it stays at rounding level and that
       $\mathbf{M}_6 = \mathbf{A}^{-1}$. How far from the minimizer is
       gradient descent at its best fixed step after the same six
       gradients?
    1. The symmetric rank-one update
       $\mathbf{M}_{k+1} = \mathbf{M}_k + \frac{(\mathbf{s} - \mathbf{M}_k\mathbf{y})(\mathbf{s} - \mathbf{M}_k\mathbf{y})^\top}{(\mathbf{s} - \mathbf{M}_k\mathbf{y})^\top\mathbf{y}}$
       also satisfies the secant equation. Show by a $2 \times 2$ example that
       it need not stay positive definite even when $\mathbf{y}^\top\mathbf{s} > 0$,
       and say which of the two checks in the text fails for it.
   :exlabel:`ex_mdl-gradient-based-optimization-bfgs-quadratic`

1. [code] **Trust region at a saddle.** ● Consider the model
   $m(\mathbf{p}) = \mathbf{g}^\top\mathbf{p} + \tfrac12\mathbf{p}^\top\mathbf{B}\mathbf{p}$
   with $\mathbf{B} = \mathrm{diag}(1, -1)$ and
   $\mathbf{g} = (\tfrac12, -\tfrac{3}{10})$.
    1. Show that $m$ is unbounded below on $\mathbb{R}^2$ and that the Newton
       point $-\mathbf{B}^{-1}\mathbf{g} = (-\tfrac12, -\tfrac{3}{10})$ is a
       saddle of $m$, not a minimum.
    1. Prove that the conditions of the subproblem proposition are
       sufficient: if $(\mathbf{B} + \tau\mathbf{I})\mathbf{p} = -\mathbf{g}$
       with $\tau \ge 0$, $\mathbf{B} + \tau\mathbf{I} \succeq \mathbf{0}$, and
       $\tau(\Delta - \|\mathbf{p}\|) = 0$, then every $\|\mathbf{q}\| \le \Delta$
       satisfies
       $m(\mathbf{q}) \ge m(\mathbf{p}) + \tfrac{\tau}{2}(\|\mathbf{p}\|^2 - \|\mathbf{q}\|^2) \ge m(\mathbf{p})$.
       (Hint: expand $m(\mathbf{q}) - m(\mathbf{p})$ around $\mathbf{p}$ and use
       $\mathbf{g} = -(\mathbf{B} + \tau\mathbf{I})\mathbf{p}$.)
    1. For $\Delta = \tfrac12$ find $\tau \ge 1$ by bisection on
       $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ (the answer
       is $\tau \approx 1.65$, $\mathbf{p} \approx (-0.19, 0.46)$), and check
       that the step increases the coordinate with negative curvature while
       the Newton step decreases it.
    1. Replace $\mathbf{g}$ by $(\tfrac12, 0)$. Show that
       $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \tfrac12/(1 + \tau) \le \tfrac14$
       for all $\tau > 1$, so the boundary equation has no solution with
       $\tau > 1$. Find the minimizers on the ball directly (they have
       $\tau = 1$ and $p_2 = \pm\sqrt{\Delta^2 - p_1^2}$), and explain in one
       sentence what the ball does with a direction of negative curvature that
       the gradient does not see.
   :exlabel:`ex_mdl-gradient-based-optimization-trust-region-saddle`

[Discussions](https://d2l.discourse.group/t/36879)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.2]{.kicker}

Curvature Methods: Newton, Quasi-Newton, and Trust Regions<br>**Newton's step · affine invariance · quadratic convergence · BFGS · trust-region subproblem**
:::
:::

::: {.slide title="Respecting Curvature vs. Modeling Curvature"}
[Motivation]{.kicker}

First-order methods never measure the curvature of the objective; they merely endure it. On our running benchmark $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ from $(1, 1)$, compare what happens when an optimizer models curvature directly:

| Optimization Strategy | Curvature Information Used | Steps to $10^{-6}$ Accuracy | Why It Behaves This Way |
| :--- | :--- | :--- | :--- |
| **Gradient Descent** | Single scalar ceiling $L = 1000$ | $6{,}908$ steps | Steep axis caps $\eta < 2/L$; flat axis crawls |
| **Heavy-Ball Momentum** | Extreme eigenvalues $\mu = 1, L = 1000$ | $315$ steps | Velocity memory equalizes mode contraction to $\sqrt{\kappa}$ |
| **Exact Curvature Scaling** | Full Hessian $\nabla^2 f = \mathrm{diag}(1, 1000)$ | **$1$ step** | Rescales each gradient entry by its exact curvature |

::: {.d2l-note}
Modeling second-order curvature eliminates the condition number $\kappa$ from the iteration count—but introduces new challenges of $O(d^3)$ compute cost and attraction to saddle points.
:::
:::

::: {.slide title="The Second-Order Taylor Model and Newton Step"}
[Newton's Method]{.kicker}

Because a linear Taylor approximation has no minimum, gradient descent cannot choose its own step length. **Newton's method** fits a local quadratic model $m_k(\mathbf{p})$ in the displacement $\mathbf{p}$ around $\mathbf{x}_k$:

$$m_k(\mathbf{p}) = f(\mathbf{x}_k) + \mathbf{g}_k^\top \mathbf{p} + \tfrac12\,\mathbf{p}^\top \nabla^2 f(\mathbf{x}_k)\,\mathbf{p}, \qquad \mathbf{g}_k = \nabla f(\mathbf{x}_k)$$

- **Minimizing the quadratic model:** When $\nabla^2 f(\mathbf{x}_k) \succ \mathbf{0}$, $m_k(\mathbf{p})$ is a strictly convex bowl whose unique minimizer satisfies $\nabla m_k(\mathbf{p}) = \mathbf{g}_k + \nabla^2 f(\mathbf{x}_k)\mathbf{p} = \mathbf{0}$.
- **The pure Newton update:**
  $$\mathbf{x}_{k+1} = \mathbf{x}_k - \big(\nabla^2 f(\mathbf{x}_k)\big)^{-1}\nabla f(\mathbf{x}_k)$$
- **Natural unit step size:** No external learning rate $\eta$ is needed; the inverse Hessian automatically sets both the direction and the physical units of the step.
:::

::: {.slide title="Preconditioned Steepest Descent and Geometry"}
[Newton's Method]{.kicker}

Newton's update is the prototype of a **preconditioned gradient step** $\mathbf{d} = -\mathbf{P}^{-1}\nabla f(\mathbf{x})$ with a symmetric positive-definite matrix $\mathbf{P} \succ \mathbf{0}$:

- **Guaranteed descent direction:** Whenever $\mathbf{P} \succ \mathbf{0}$, we have $\mathbf{P}^{-1} \succ \mathbf{0}$, so the directional derivative is strictly negative for any $\nabla f(\mathbf{x}) \neq \mathbf{0}$:
  $$\nabla f(\mathbf{x})^\top \mathbf{d} = -\nabla f(\mathbf{x})^\top \mathbf{P}^{-1}\nabla f(\mathbf{x}) < 0$$
- **Steepest descent in the Hessian norm:** If we measure step length in the local Mahalanobis norm $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\mathbf{d}}$ instead of the Euclidean norm $\|\mathbf{d}\|_2$, minimizing $\nabla f(\mathbf{x})^\top \mathbf{d}$ over $\|\mathbf{d}\|_{\mathbf{P}} = 1$ yields $\mathbf{d} \propto -\mathbf{P}^{-1}\nabla f(\mathbf{x})$.
- **Special cases:** $\mathbf{P} = \mathbf{I}$ recovers gradient descent; $\mathbf{P} = \nabla^2 f(\mathbf{x}_k)$ gives Newton's method; diagonal $\mathbf{P}$ gives adaptive methods (AdaGrad, RMSProp, Adam).
:::

::: {.slide title="Affine Invariance Eliminates the Condition Number"}
[Newton's Method]{.kicker}

Why does Newton's method solve *any* positive-definite quadratic $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ in a single step ($\mathbf{x}_1 = \mathbf{x}_0 - \mathbf{A}^{-1}\mathbf{A}\mathbf{x}_0 = \mathbf{0}$), regardless of $\kappa(\mathbf{A})$?

- **Coordinate reparametrization:** Let $\mathbf{x} = \mathbf{T}\mathbf{y}$ for any invertible matrix $\mathbf{T} \in \mathbb{R}^{d \times d}$, and define $h(\mathbf{y}) = f(\mathbf{T}\mathbf{y})$.
- **Chain rule for gradient and Hessian:**
  $$\nabla h(\mathbf{y}) = \mathbf{T}^\top \nabla f(\mathbf{T}\mathbf{y}), \qquad \nabla^2 h(\mathbf{y}) = \mathbf{T}^\top \nabla^2 f(\mathbf{T}\mathbf{y})\,\mathbf{T}$$
- **Trajectory invariance:** One Newton step in $\mathbf{y}$-coordinates maps directly to the Newton step in $\mathbf{x}$-coordinates:
  $$\mathbf{T}\mathbf{y}_{k+1} = \mathbf{T}\mathbf{y}_k - \mathbf{T}\big(\mathbf{T}^\top \nabla^2 f(\mathbf{x}_k)\mathbf{T}\big)^{-1}\mathbf{T}^\top \nabla f(\mathbf{x}_k) = \mathbf{x}_k - \big(\nabla^2 f(\mathbf{x}_k)\big)^{-1}\nabla f(\mathbf{x}_k) = \mathbf{x}_{k+1}$$

::: {.d2l-note .rule}
Because $\mathbf{T} = \mathbf{A}^{-1/2}$ transforms any elongated quadratic valley into an isotropic sphere ($\kappa = 1$), Newton's trajectory is completely blind to the condition number.
:::
:::

::: {.slide title="Local Quadratic Convergence: Error Decomposition"}
[Convergence Proof]{.kicker}

Near a non-quadratic minimizer $\mathbf{x}^\star$ where $\nabla f(\mathbf{x}^\star) = \mathbf{0}$, how rapidly does the error $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^\star$ shrink?

- **Step 1 — Express the new error via the Hessian at $\mathbf{x}_k$:** Subtract $\mathbf{x}^\star$ from the Newton update and factor out $(\nabla^2 f_k)^{-1}$:
  $$\mathbf{e}_{k+1} = \mathbf{e}_k - (\nabla^2 f_k)^{-1}\mathbf{g}_k = (\nabla^2 f_k)^{-1}\Big[\nabla^2 f_k\,\mathbf{e}_k - \big(\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star)\big)\Big]$$
- **Step 2 — Apply the fundamental theorem of calculus:** Write the gradient difference as an integral of the Hessian along the segment $\mathbf{x}^\star + t\mathbf{e}_k$:
  $$\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star) = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\,\mathbf{e}_k\,dt$$
- **Step 3 — Combine inside the integral:** The bracketed residual is $\int_0^1 \big[\nabla^2 f(\mathbf{x}_k) - \nabla^2 f(\mathbf{x}^\star + t\mathbf{e}_k)\big]\mathbf{e}_k\,dt$.
:::

::: {.slide title="Local Quadratic Convergence: Doubling Digits"}
[Convergence Proof]{.kicker}

Suppose on a ball around $\mathbf{x}^\star$ that $\nabla^2 f(\mathbf{x}) \succeq \mu\mathbf{I}$ and the Hessian is $L_2$-Lipschitz: $\|\nabla^2 f(\mathbf{x}) - \nabla^2 f(\mathbf{y})\| \le L_2\|\mathbf{x} - \mathbf{y}\|$.

- **Bounding the integral:** Since $\mathbf{x}_k$ and $\mathbf{x}^\star + t\mathbf{e}_k$ are separated by $(1 - t)\|\mathbf{e}_k\|$, the integral norm is at most $\int_0^1 L_2(1 - t)\|\mathbf{e}_k\|^2\,dt = \tfrac{L_2}{2}\|\mathbf{e}_k\|^2$.
- **Quadratic error recursion:** Combining with $\|(\nabla^2 f_k)^{-1}\| \le 1/\mu$ proves:
  $$\|\mathbf{e}_{k+1}\| \le \frac{L_2}{2\mu}\,\|\mathbf{e}_k\|^2 \iff \left(\frac{L_2}{2\mu}\|\mathbf{e}_{k+1}\|\right) \le \left(\frac{L_2}{2\mu}\|\mathbf{e}_k\|\right)^{\!2}$$
- **Doubling accurate digits:** Once the scaled error drops below $10^{-1}$, successive steps plunge through $10^{-2}$ $\to$ $10^{-4}$ $\to$ $10^{-8}$ $\to$ $10^{-16}$. Far from $\mathbf{x}^\star$, **damped Newton** backtracks $\alpha_k \le 1$ along $\mathbf{p}_k$ using the Armijo condition, automatically accepting $\alpha_k = 1$ near $\mathbf{x}^\star$.
:::

::: {.slide title="Newton's Method on Valleys and Logistic Loss"}
[Empirical Verification]{.kicker}

We test Newton's method on the $\kappa = 1000$ quadratic $\mathrm{diag}(1, 1000)$ and on $\ell_2$-regularized logistic regression ($n = 200$ examples, $d = 5$ weights, $\nu = 0.1$):

@!gradient-based-optimization-newton

- On the $\kappa = 1000$ valley, $1$ Newton step replaces $6{,}908$ gradient descent steps.
- On the non-quadratic logistic loss, $\|\nabla f\|$ squares at each step ($10^{-1}$ $\to$ $10^{-3}$ $\to$ $10^{-6}$ $\to$ $10^{-11}$ $\to$ $10^{-16}$), matching $4{,}000$ gradient descent steps in $6$ linear solves.
:::

::: {.slide title="Why Exact Newton Fails at Deep Learning Scale"}
[Computational Cost]{.kicker}

Comparing iteration counts alone flatters Newton's method. Forming the $d \times d$ Hessian costs $O(d^2)$ memory and solving $(\nabla^2 f_k)\mathbf{p}_k = -\mathbf{g}_k$ costs $O(d^3)$ operations per step:

| Model Scale | Parameter Count $d$ | Hessian Entries $d^2$ | Linear Solve Cost $\sim d^3$ FLOPs | Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Toy Logistic Model** | $5$ | $25$ ($200$ B) | $125$ | Instantaneous |
| **Small Linear Layer** | $10^4$ | $10^8$ ($400$ MB) | $10^{12}$ | Seconds per step |
| **Modest ConvNet / ViT** | $10^8$ | $10^{16}$ ($40$ PB) | $10^{24}$ | Impossible to store or invert |
| **Frontier LLM** | $10^{11}$ | $10^{22}$ ($4 \times 10^7$ PB) | $10^{33}$ | Out of reach by $15$ orders of magnitude |

::: {.d2l-note .warn}
Even a single exact Newton step is impossible for modern neural networks—motivating low-memory quasi-Newton approximations and structured/diagonal preconditioners.
:::
:::

::: {.slide title="Saddle-Point Attraction on Nonconvex Surfaces"}
[Nonconvex Pitfalls]{.kicker}

Cost is not Newton's only flaw. Because the Newton step solves $\nabla m_k(\mathbf{p}) = \mathbf{0}$, it jumps to the stationary point of the local quadratic model *whether that point is a minimum, a maximum, or a saddle*:

- **Indefinite Hessian:** Return to $f(x, y) = \tfrac12 x^2 + \cos y$ at $(0.5, 0.3)$, where $\nabla f = (0.5, -\sin 0.3) = (0.5, -0.296)$ and $\nabla^2 f = \mathrm{diag}(1, -\cos 0.3) = \mathrm{diag}(1, -0.955)$.
- **Wrong-way step along negative curvature:** Along $y$, the surface curves *downward* ($f_{yy} < 0$). Dividing the negative gradient $-\sin y$ by the negative second derivative $-\cos y$ reverses the sign of the update:
  $$y_{k+1} = y_k - \frac{-\sin y_k}{-\cos y_k} = y_k - \tan y_k \approx -\frac{y_k^3}{3}$$
- Instead of moving downhill toward the minimum at $y = \pi$, Newton's method converges cubically *uphill* along $y$ straight into the saddle point $(0, 0)$!
:::

::: {.slide title="Newton vs. Gradient Descent Near a Saddle"}
[Nonconvex Pitfalls]{.kicker}

Starting both methods at $(0.5, 0.3)$ on $f(x, y) = \tfrac12 x^2 + \cos y$ shows how a richer local model can converge to a worse stationary point:

@!mdl-gradient-based-optimization-newton-s-method-1

::: {.d2l-note .warn}
Newton's method is pulled into the saddle $(0, 0)$ with $f = 1$ in three steps, whereas gradient descent ($\eta = 1$) follows the negative gradient downhill to the global minimum $(0, \pi)$ with $f = -1$.
:::
:::

::: {.slide title="Inferring Curvature from Gradients: Secant Rule"}
[Quasi-Newton Methods]{.kicker}

Can we recover curvature information using only first derivatives? When the gradient changes rapidly along a step, the curvature in that direction must be large. In one dimension, the **secant method** replaces $f''(x_k)$ with a finite difference of the last two gradients:

$$h_k = \frac{f'(x_k) - f'(x_{k-1})}{x_k - x_{k-1}}, \qquad x_{k+1} = x_k - \frac{f'(x_k)}{h_k}$$

- **Superlinear convergence without second derivatives:** On $f(x) = e^x - 2x$ (minimizer $x^\star = \log 2 \approx 0.6931$) from $x_0 = 0, x_1 = 1$:
- The error $|x_k - x^\star|$ drops through $1.1 \times 10^{-1} \to 1.6 \times 10^{-2} \to 9.3 \times 10^{-4} \to 7.7 \times 10^{-6} \to 3.6 \times 10^{-9}$.
- Each error is proportional to the product of the previous two ($e_{k+1} \approx C e_k e_{k-1}$), multiplying the number of accurate digits by the golden ratio $\approx 1.618$ per step at the cost of just one gradient evaluation.
:::

::: {.slide title="The Multivariate Secant Equation and BFGS"}
[Quasi-Newton Methods]{.kicker}

In $d$ dimensions, a **quasi-Newton method** maintains a symmetric positive-definite approximation $\mathbf{M}_k \approx (\nabla^2 f(\mathbf{x}_k))^{-1}$ and updates it from the observed step and gradient difference:

$$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k, \qquad \mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$$

- **The secant equation:** On a quadratic $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, $\mathbf{y}_k = \mathbf{A}\mathbf{s}_k$, so we require $\mathbf{M}_{k+1}$ to map $\mathbf{y}_k$ back to $\mathbf{s}_k$:
  $$\mathbf{M}_{k+1}\,\mathbf{y}_k = \mathbf{s}_k$$
- **The BFGS update:** Because $d$ equations cannot uniquely determine a $d \times d$ symmetric matrix, **BFGS** selects the symmetric matrix closest to $\mathbf{M}_k$ (in a weighted Frobenius norm) that satisfies $\mathbf{M}_{k+1}\mathbf{y}_k = \mathbf{s}_k$:
  $$\mathbf{M}_{k+1} = \big(\mathbf{I} - \gamma_k\mathbf{s}_k\mathbf{y}_k^\top\big)\,\mathbf{M}_k\,\big(\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top\big) + \gamma_k\mathbf{s}_k\mathbf{s}_k^\top, \qquad \gamma_k = \frac{1}{\mathbf{y}_k^\top \mathbf{s}_k}$$
:::

::: {.slide title="How BFGS Preserves Positive Definiteness"}
[Quasi-Newton Methods]{.kicker}

Reading the BFGS formula algebraically explains both why it satisfies the secant equation and when it guarantees a descent direction:

- **Erase-and-write mechanism:** Because $\gamma_k(\mathbf{y}_k^\top\mathbf{s}_k) = 1$, the right factor $(\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top)$ maps $\mathbf{y}_k$ to $\mathbf{0}$ while leaving every vector orthogonal to $\mathbf{s}_k$ untouched. The first term erases $\mathbf{M}_k$'s old prediction along $\mathbf{y}_k$, and the second term writes in $\gamma_k\mathbf{s}_k(\mathbf{s}_k^\top\mathbf{y}_k) = \mathbf{s}_k$.
- **Preserving $\mathbf{M}_{k+1} \succ \mathbf{0}$:** For any nonzero $\mathbf{z} \in \mathbb{R}^d$, let $\mathbf{w} = (\mathbf{I} - \gamma_k\mathbf{y}_k\mathbf{s}_k^\top)\mathbf{z}$:
  $$\mathbf{z}^\top \mathbf{M}_{k+1}\mathbf{z} = \mathbf{w}^\top \mathbf{M}_k\mathbf{w} + \gamma_k(\mathbf{s}_k^\top\mathbf{z})^2$$
- **Curvature condition:** If $\mathbf{M}_k \succ \mathbf{0}$ and $\mathbf{y}_k^\top\mathbf{s}_k > 0$ ($\gamma_k > 0$), both terms are nonnegative and cannot vanish simultaneously. Thus $\mathbf{M}_{k+1} \succ \mathbf{0}$ holds whenever the directional slope increases along the step (enforced in practice via the Wolfe line-search conditions).
:::

::: {.slide title="BFGS Recovers the Exact Inverse Hessian in d Steps"}
[Quasi-Newton Methods]{.kicker}

On $\mathbf{A} = \mathrm{diag}(1, 10)$ from $(1, 1)$ with $\mathbf{M}_0 = \mathbf{I}$ and exact line search, BFGS probes the steep direction on step 1 and recovers $\mathbf{A}^{-1} = \mathrm{diag}(1, 0.1)$ exactly on step $d = 2$:

@!mdl-gradient-based-optimization-bfgs-1

- **L-BFGS for large $d$:** Storing a dense $d \times d$ matrix $\mathbf{M}_k$ still costs $O(d^2)$ memory. **L-BFGS** stores only the $m$ most recent pairs $\{(\mathbf{s}_j, \mathbf{y}_j)\}$, applying $\mathbf{M}_k\mathbf{g}_k$ in $O(md)$ time and memory—ideal for deterministic problems, though vulnerable to minibatch noise in $\mathbf{y}_k$.
:::

::: {.slide title="Trust-Region Methods: Bounding the Model Radius"}
[Trust Regions]{.kicker}

Newton's method ran into the saddle of $\tfrac12 x^2 + \cos y$ because it trusted its local Taylor approximation across all of $\mathbb{R}^d$. A **trust-region method** keeps the full quadratic model—including directions of negative curvature—but restricts the step $\mathbf{p}$ to a ball of radius $\Delta_k$:

$$\min_{\|\mathbf{p}\| \le \Delta_k} m_k(\mathbf{p}) = f(\mathbf{x}_k) + \mathbf{g}_k^\top\mathbf{p} + \tfrac12\,\mathbf{p}^\top\mathbf{B}_k\mathbf{p}$$

- **Always well-posed:** Because the closed ball $\{\mathbf{p} : \|\mathbf{p}\| \le \Delta_k\}$ is compact and $m_k$ is continuous, a global minimizer of the subproblem always exists, even when $\mathbf{B}_k = \nabla^2 f(\mathbf{x}_k)$ is indefinite or negative definite.
- **Escaping saddles:** Along any eigenvector of $\mathbf{B}_k$ with a negative eigenvalue, the quadratic model curves downward, so minimizing $m_k$ inside the ball automatically pushes $\mathbf{p}$ all the way to the boundary $\|\mathbf{p}\| = \Delta_k$ away from the saddle.
:::

::: {.slide title="Optimality Conditions for the Trust-Region Ball"}
[Trust Regions]{.kicker}

A vector $\mathbf{p}$ with $\|\mathbf{p}\| \le \Delta$ is a global minimizer of the trust-region subproblem if and only if there exists a scalar shift $\tau \ge 0$ satisfying:

$$(\mathbf{B} + \tau\mathbf{I})\,\mathbf{p} = -\mathbf{g}, \qquad \mathbf{B} + \tau\mathbf{I} \succeq \mathbf{0}, \qquad \tau\,(\Delta - \|\mathbf{p}\|) = 0$$

- **Interior case ($\tau = 0$):** When $\mathbf{B} \succ \mathbf{0}$ and the unconstrained Newton step satisfies $\|\mathbf{B}^{-1}\mathbf{g}\| \le \Delta$, we take the pure Newton step $\mathbf{p} = -\mathbf{B}^{-1}\mathbf{g}$ with $\tau = 0$.
- **Boundary case ($\tau > 0$):** Whenever $\mathbf{B}$ has negative eigenvalues or the Newton step exceeds $\Delta$, the solution sits on the boundary $\|\mathbf{p}\| = \Delta$.
- **Spectral damping:** The multiplier $\tau > -\lambda_{\min}(\mathbf{B})$ shifts every eigenvalue of $\mathbf{B}$ upward until $\mathbf{B} + \tau\mathbf{I} \succ \mathbf{0}$ and $\|(\mathbf{B} + \tau\mathbf{I})^{-1}\mathbf{g}\| = \Delta$ (solved via 1D bisection or Newton root-finding on $\tau$).
:::

::: {.slide title="Escaping a Saddle with a Trust-Region Step"}
[Trust Regions]{.kicker}

Let us solve the trust-region subproblem by hand at $\mathbf{x}_0 = (0.5, 0.3)$ on $f(x,y) = \tfrac12 x^2 + \cos y$ with radius $\Delta = 0.5$:

- **Local derivatives:** $\mathbf{g} = (0.5, -0.296)$ and $\mathbf{B} = \mathrm{diag}(1, -0.955)$.
- **Positivity threshold:** Requiring $\mathbf{B} + \tau\mathbf{I} = \mathrm{diag}(1 + \tau,\, \tau - 0.955) \succeq \mathbf{0}$ forces $\tau \ge 0.955 > 0$, so the step must lie on the boundary $\|\mathbf{p}\| = 0.5$.
- **Boundary system:** Solve $(1 + \tau)p_1 = -0.5$ and $(\tau - 0.955)p_2 = 0.296$ subject to $p_1^2 + p_2^2 = 0.25$:
  $$\tau \approx 1.60, \qquad \mathbf{p} \approx (-0.19,\, +0.46), \qquad \mathbf{x}_1 = \mathbf{x}_0 + \mathbf{p} \approx (0.31,\, 0.76)$$
- **Comparison with Newton:** Unconstrained Newton stepped to $(-0.5, -0.31)$ *toward* the saddle $(0, 0)$, whereas the shifted matrix $(\mathbf{B} + \tau\mathbf{I}) \succ \mathbf{0}$ flips the sign of $p_2$ and steps *away* from the saddle, dropping $f$ from $1.080$ to $0.771$.
:::

::: {.slide title="Adaptive Radius Control via Agreement Ratio"}
[Trust Regions]{.kicker}

After computing the candidate step $\mathbf{p}_k$, the algorithm compares the actual loss reduction to the reduction predicted by the quadratic model:

$$r_k = \frac{f(\mathbf{x}_k) - f(\mathbf{x}_k + \mathbf{p}_k)}{m_k(\mathbf{0}) - m_k(\mathbf{p}_k)}$$

@!mdl-gradient-based-optimization-the-trust-region-step-at-a-saddle-2

- When $r_k > 0.75$ on the boundary, $\Delta_k$ doubles ($0.5 \to 1 \to 2$); once inside the convex basin ($k \ge 3$), $\tau = 0$ and the method becomes pure Newton, reaching $(0, \pi)$ in $6$ steps.
:::

::: {.slide title="Comparing Trajectories: Newton vs. Trust Region"}
[Trust Regions]{.kicker}

::: {.cols .vc}
::: {.col}
- **Panel (a) — Local saddle surface:** At $(0.5, 0.3)$, the quadratic model $m_0(\mathbf{p})$ is a hyperbolic saddle. Pure Newton jumps to its stationary point at the origin, while minimizing $m_0$ inside the ball $\|\mathbf{p}\| \le 0.5$ follows the downward-curving $y$-axis.
- **Panel (b) — Global trajectory:** Trust region escapes the indefinite region in three steps, expands $\Delta$, and enjoys quadratic convergence to the global minimum $(0, \pi)$.
:::

::: {.col .fig}
@fig:mdl-opt-trust-region-saddle
:::
:::
:::

::: {.slide title="Concept Check: Curvature, Cost, and Damping"}
[Concept Check]{.kicker}

Test your understanding of second-order and quasi-Newton methods:

1. Suppose Newton's method has entered its quadratic convergence basin with scaled error $\frac{L_2}{2\mu}\|\mathbf{e}_k\| = 10^{-1}$. How many Newton steps are needed to reach $10^{-8}$, compared with gradient descent at $\kappa = 10^3$?
2. In a trust-region subproblem with $\mathbf{B} = \mathrm{diag}(1, 4)$, $\mathbf{g} = (1, 4)$, and radius $\Delta = 2$, what are the optimal multiplier $\tau$ and step $\mathbf{p}$?
3. Why does BFGS require the curvature condition $\mathbf{y}_k^\top \mathbf{s}_k > 0$?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1) 3 Newton steps:** The scaled error squares at each step ($10^{-1}$ $\to$ $10^{-2}$ $\to$ $10^{-4}$ $\to$ $10^{-8}$). Gradient descent at $\kappa = 10^3$ needs $\approx 1{,}150$ steps per digit, or roughly **$8{,}050$ steps** for seven digits. **(2) $\tau = 0$ and $\mathbf{p} = (-1, -1)$:** Here $\mathbf{B} \succ \mathbf{0}$ and the unconstrained Newton step $-\mathbf{B}^{-1}\mathbf{g} = (-1, -1)$ has norm $\sqrt{2} \approx 1.414 < \Delta = 2$, so it fits strictly inside the ball. **(3) To keep $\mathbf{M}_{k+1} \succ \mathbf{0}$:** In the quadratic form $\mathbf{z}^\top\mathbf{M}_{k+1}\mathbf{z} = \mathbf{w}^\top\mathbf{M}_k\mathbf{w} + (\mathbf{s}_k^\top\mathbf{z})^2 / (\mathbf{y}_k^\top\mathbf{s}_k)$, positive definiteness is preserved if and only if $\mathbf{y}_k^\top\mathbf{s}_k > 0$.
:::
:::

::: {.slide title="Summary: Second-Order and Curvature Methods"}
[Key Takeaways]{.kicker}

| Method | Step / Subproblem | Convergence Rate | Per-Step Memory & Compute | Deep Learning Role & Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Newton** | $\mathbf{p}_k = -(\nabla^2 f_k)^{-1}\mathbf{g}_k$ | Quadratic ($\|\mathbf{e}_{k+1}\| \le \frac{L_2}{2\mu}\|\mathbf{e}_k\|^2$), affine-invariant | $O(d^2)$ memory, $O(d^3)$ solve | Prohibitive for large $d$; attracted to saddle points |
| **Damped Newton** | $\mathbf{x}_k + \alpha_k \mathbf{p}_k$ via Armijo line search | Global descent when $\nabla^2 f_k \succ \mathbf{0}$; local quadratic ($\alpha_k = 1$) | $O(d^2)$ memory, $O(d^3)$ solve | Prevents overshooting far from $\mathbf{x}^\star$, still needs $\nabla^2 f_k \succ \mathbf{0}$ |
| **BFGS / L-BFGS** | Rank-2 update solving $\mathbf{M}_{k+1}\mathbf{y}_k = \mathbf{s}_k$ | Superlinear ($d$ steps on $d$-D quadratic) | BFGS: $O(d^2)$; L-BFGS: $O(md)$ | Standard for deterministic medium-scale fits; sensitive to minibatch noise |
| **Trust Region** | $\min_{\|\mathbf{p}\| \le \Delta_k} m_k(\mathbf{p})$ $\iff$ $(\mathbf{B}_k + \tau\mathbf{I})\mathbf{p} = -\mathbf{g}_k$ | Escapes negative-curvature saddles; local quadratic | Subproblem solve per step | Motivated spectral damping ($\mathbf{B} + \tau\mathbf{I}$) in second-order preconditioners |
:::

