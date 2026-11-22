# Gradient Descent and Curvature
:label:`sec_mdl-gradient-based-optimization`

In practice, a learning rate is usually chosen by trial and error: we pick a
value, watch the loss, and turn it down if training blows up. Because every
step moves opposite the gradient of the loss, the gradient fixes the direction
and the learning rate sets only the distance. Yet a poor choice of distance
turns training either painfully slow or unstable. How far can we trust a step
along the gradient, and how many iterations does that limit force upon us?

A function of just two variables already exposes the difficulty. Consider the
elliptical bowl $f(x, y) = \tfrac12(x^2 + 10\,y^2)$, which curves ten times
more steeply along $y$ than along $x$, and start at $(2, 1)$, where $f = 7$
and $\nabla f = (2, 10)$. Taking a step of size $\eta$ against the gradient
lands at $(2 - 2\eta,\, 1 - 10\eta)$. With $\eta = 0.01$, the loss barely
budges, dropping to $6.01$. With $\eta = 0.1$, it falls sharply to $1.62$.
With $\eta = 0.25$, it *rises* to $12.4$. Every one of those steps pointed
downhill; only the step length changed.

The culprit is curvature. A step short enough to stay safe along the steep
$y$-axis creeps along the flat $x$-axis. Even the best fixed learning rate
needs $69$ iterations to shrink the error by a factor of $10^6$ (six digits of
accuracy). Stretch the curvature ratio to $1000$—our chapter's running example
$\tfrac12(x_1^2 + 1000\,x_2^2)$—and gradient descent needs $6{,}908$ steps.
Adding momentum brings that count down to $315$. We will work by hand on the
gentler bowl, whose Hessian $\mathrm{diag}(1, 10)$ keeps every number easy to
check, and quote the corresponding counts for the stiffer running example.

We build the analysis in three stages. First, we formalize smoothness to turn
the local linear approximation of :numref:`sec_mdl-multivariable_calculus` into
a guarantee on finite steps, yielding the descent lemma
:eqref:`eq_mdl-opt-descent-lemma`. Next, we solve gradient descent exactly on
quadratics, where the ratio of curvatures emerges as the condition number and
fixes the optimal step :eqref:`eq_mdl-opt-optimal-step`. Finally, we introduce
momentum and derive its $\sqrt\kappa$ convergence law
:eqref:`eq_mdl-opt-hb-rate`. Throughout this section, our methods only respect
curvature without modeling it; methods that model curvature directly, such as
Newton's method, appear in :numref:`sec_mdl-curvature-methods`. We also assume
exact gradients here, leaving the noisy gradients of minibatch training to
:numref:`sec_mdl-stochastic-gradient-methods`. All code below is plain NumPy,
so each update can be read directly against its equation.

```{.python .input #gradient-based-optimization-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #gradient-based-optimization-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #gradient-based-optimization-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #gradient-based-optimization-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Descent Directions and Smoothness
:label:`subsec_mdl-gd-smoothness`

Every gradient step settles two questions: which way to move, and how far. The
first has an immediate answer from the first-order Taylor expansion. The second
requires knowing how quickly the gradient itself can change—an assumption that
turns a local linear approximation into a guarantee for steps of finite length.

### The Steepest Direction
:label:`subsec_mdl-descent-directions`

In our opening example, we stepped along $-\nabla f$ without asking whether
another direction might be better. To see why the negative gradient is the
natural choice, expand $f : \mathbb{R}^d \to \mathbb{R}$ to first order. A
small step of size $\eta > 0$ along a direction $\mathbf{d}$ changes the
function value by

$$
f(\mathbf{x} + \eta\,\mathbf{d}) = f(\mathbf{x}) + \eta\, \nabla f(\mathbf{x})^\top \mathbf{d} + o(\eta).
$$

To first order, the change is simply $\eta$ times the directional derivative
$\nabla f(\mathbf{x})^\top \mathbf{d}$. We call $\mathbf{d}$ a **descent
direction** at $\mathbf{x}$ whenever $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$.

**Proposition (descent directions, and the steepest one).** *Let $f$ be
differentiable at $\mathbf{x}$ with $\nabla f(\mathbf{x}) \neq \mathbf{0}$.*

1. *If $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$, then
   $f(\mathbf{x} + \eta\,\mathbf{d}) < f(\mathbf{x})$ for every sufficiently
   small $\eta > 0$.*
2. *Among all unit vectors, the directional derivative is minimized uniquely by
   the normalized negative gradient:*

$$
\begin{aligned}
\min_{\|\mathbf{d}\| = 1} \nabla f(\mathbf{x})^\top \mathbf{d} &= -\|\nabla f(\mathbf{x})\|, \\
\textrm{attained only at }\mathbf{d} &= -\frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-steepest`

**Proof.** For the first claim, define
$\varphi(\eta) = f(\mathbf{x} + \eta\,\mathbf{d})$. By the chain rule,
$\varphi'(0) = \nabla f(\mathbf{x})^\top \mathbf{d} < 0$, so the difference
quotient $(\varphi(\eta) - \varphi(0))/\eta$ is strictly negative for all
sufficiently small $\eta > 0$; hence $\varphi(\eta) < \varphi(0)$. For the
second claim, the Cauchy--Schwarz inequality gives
$\nabla f(\mathbf{x})^\top \mathbf{d} \ge -\|\nabla f(\mathbf{x})\|\,\|\mathbf{d}\| = -\|\nabla f(\mathbf{x})\|$
for every unit vector $\mathbf{d}$, with equality if and only if $\mathbf{d}$
is a negative scalar multiple of $\nabla f(\mathbf{x})$. The unique such unit
vector is $-\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$, and it attains the
bound. $\blacksquare$

On our opening bowl, the steepest unit direction at $(2, 1)$ is
$-(2, 10)/\sqrt{104} = (-0.196, -0.981)$, with slope
$-\|\nabla f\| = -10.198$. The coordinate direction $(-1, 0)$ also points
downhill, but with a much gentler slope of $-2$. In fact, descent directions
fill the entire open half-space of vectors making an acute angle with
$-\nabla f$; the negative gradient simply picks out the steepest one. Because
this result governs slopes—the limit of vanishingly short steps—let us check
how faithfully those slopes predict the actual decrease over a small finite
step, $\eta = 0.01$, across four unit directions.

```{.python .input #gradient-based-optimization-steepest-direction}
def f(v):
    return 0.5 * (v[0]**2 + 10 * v[1]**2)

def grad(v):
    return np.array([v[0], 10 * v[1]])

x = np.array([2.0, 1.0])
g = grad(x)
steepest = -g / np.linalg.norm(g)
candidates = [('steepest -grad/|grad|', steepest),
              ('diagonal -(1,1)/sqrt(2)', -np.ones(2) / np.sqrt(2)),
              ('axis     (-1,0)', np.array([-1.0, 0.0])),
              ('uphill   +(1,1)/sqrt(2)', np.ones(2) / np.sqrt(2))]
eta = 0.01
for name, d in candidates:
    print(f'{name}:  slope = {g @ d:+8.4f},  '
          f'f(x + eta d) - f(x) = {f(x + eta * d) - f(x):+.6f}')
```

At $\eta = 0.01$, each actual change matches $\eta$ times the directional slope
to within one percent, so the first-order model is a reliable guide for short
steps. Differentiability promises nothing beyond that: a decrease for
sufficiently small steps. How small is "sufficiently small" is the question we
turn to next.

### Smoothness and the Descent Lemma

**Gradient descent** repeats the steepest-descent step with a fixed **step
size** (or learning rate) $\eta > 0$:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \nabla f(\mathbf{x}_k).
$$
:eqlabel:`eq_mdl-opt-gd-step`

Throughout this section, $k$ indexes iterations and we abbreviate
$\mathbf{g}_k = \nabla f(\mathbf{x}_k)$. The first-order Taylor expansion
justified the direction $-\mathbf{g}_k$, but it says nothing about how far to
trust it: once the iterate moves far enough that the gradient changes
significantly, $\mathbf{g}_k$ no longer describes $f$. To certify a finite step
size, we need a bound on how fast the gradient can vary.

**Definition ($L$-smoothness).** A differentiable function $f$ is
**$L$-smooth** if its gradient changes by at most $L$ per unit of distance:

$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \;\le\; L\, \|\mathbf{x} - \mathbf{y}\|
\qquad \textrm{for all } \mathbf{x}, \mathbf{y}.
$$
:eqlabel:`eq_mdl-opt-smoothness`

When $f$ is twice continuously differentiable, $L$-smoothness is equivalent to
requiring every eigenvalue of its Hessian to lie in $[-L, L]$. Bounded
eigenvalues imply :eqref:`eq_mdl-opt-smoothness` by integrating the Hessian
along the line segment from $\mathbf{y}$ to $\mathbf{x}$, and the converse
follows by taking a difference-quotient limit (exercise
:exref:`ex_mdl-gradient-based-optimization-smoothness-constants`a). In short,
$L$ caps the curvature in every direction. Our opening bowl has constant
Hessian $\mathrm{diag}(1, 10)$, so $L = 10$. By contrast, a function whose
curvature grows without bound has no global smoothness constant: $\tfrac14 x^4$
has $f''(x) = 3x^2$, so no finite global $L$ exists. We will handle that case
shortly using backtracking line search.

Smoothness upgrades the local tangent plane into a global upper bound—a
quadratic ceiling:

$$
f(\mathbf{y}) \;\le\; f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) + \tfrac{L}{2}\, \|\mathbf{y} - \mathbf{x}\|^2.
$$
:eqlabel:`eq_mdl-opt-quadratic-ceiling`

Geometrically, the graph of $f$ can never rise above its tangent plane at
$\mathbf{x}$ by more than $\tfrac{L}{2}$ times the squared distance. Before we
prove this inequality, plug in a gradient step
$\mathbf{y} = \mathbf{x} - \eta\,\mathbf{g}$ and watch the two terms compete.
The linear first-order decrease $\eta\,\|\mathbf{g}\|^2$ grows proportionally
to $\eta$, whereas the quadratic curvature penalty
$\tfrac{L}{2}\eta^2\|\mathbf{g}\|^2$ grows as $\eta^2$. For small $\eta$, the
linear decrease wins; at $\eta = 2/L$, the two terms exactly cancel.

**Proposition (descent lemma).** *If $f$ is $L$-smooth, then the gradient step
:eqref:`eq_mdl-opt-gd-step` satisfies*

$$
f(\mathbf{x}_{k+1}) \;\le\; f(\mathbf{x}_k) - \eta \left(1 - \tfrac{L\eta}{2}\right) \|\mathbf{g}_k\|^2.
$$
:eqlabel:`eq_mdl-opt-descent-lemma`

*In particular, $f$ strictly decreases whenever $0 < \eta < 2/L$ and
$\mathbf{g}_k \neq \mathbf{0}$. The guaranteed decrease is maximized at
$\eta = 1/L$, where
$f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \tfrac{1}{2L}\|\mathbf{g}_k\|^2$.*

**Proof.** We first establish the quadratic ceiling
:eqref:`eq_mdl-opt-quadratic-ceiling`. Restrict $f$ to the line segment
between $\mathbf{x}$ and $\mathbf{y}$ via
$\psi(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ for $t \in [0, 1]$, so
that $\psi'(t) = \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x}))^\top (\mathbf{y} - \mathbf{x})$.
Applying the fundamental theorem of calculus, Cauchy--Schwarz, and the
Lipschitz condition :eqref:`eq_mdl-opt-smoothness` yields

$$
\begin{aligned}
f(\mathbf{y}) - f(\mathbf{x}) &- \nabla f(\mathbf{x})^\top (\mathbf{y}-\mathbf{x}) \\
&= \int_0^1 \big(\nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x})\big)^\top (\mathbf{y}-\mathbf{x})\, dt \\
&\le \int_0^1 L\,t\, \|\mathbf{y}-\mathbf{x}\|^2\, dt = \tfrac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2.
\end{aligned}
$$

Substituting the gradient step $\mathbf{y} = \mathbf{x}_k - \eta\,\mathbf{g}_k$
gives

$$
f(\mathbf{x}_{k+1}) \;\le\; f(\mathbf{x}_k) - \eta\,\|\mathbf{g}_k\|^2 + \tfrac{L}{2}\,\eta^2\, \|\mathbf{g}_k\|^2.
$$

The factor $\eta(1 - L\eta/2)$ is strictly positive if and only if
$0 < \eta < 2/L$, and it reaches its maximum at $\eta = 1/L$. $\blacksquare$

Notice that the proof only compares gradients between $\mathbf{x}_k$ and points
along the segment to $\mathbf{x}_{k+1}$, so it suffices for $\nabla f$ to be
$L$-Lipschitz on that segment. Let us test the lemma on our opening bowl, where
$\|\mathbf{g}\|^2 = 104$ and $L = 10$. At $\eta = 1/L = 0.1$, the bound
guarantees $f \le 7 - 104/20 = 1.8$. At $\eta = 0.01$, it guarantees
$f \le 7 - 104 \cdot 0.01 \cdot 0.95 = 6.012$. At $\eta = 0.25 > 2/L$, the
factor $1 - L\eta/2$ turns negative and the bound rises above the starting
value, offering no guarantee at all. How tight are these bounds compared with
the values the three steps actually reach?

```{.python .input #mdl-gradient-based-optimization-smoothness-and-the-descent-lemma}
x, L = np.array([2.0, 1.0]), 10.0
g = grad(x)
for eta in (0.01, 0.1, 0.25):
    bound = f(x) - eta * (1 - L * eta / 2) * (g @ g)
    print(f'eta = {eta:4.2f}:  bound {bound:7.4f},  actual {f(x - eta * g):7.4f}')
```

For the two stable steps, the bounds are remarkably tight: $6.0102$ lies just
under $6.012$, and $1.62$ sits cleanly below $1.8$. At $\eta = 0.25$, the bound
$13.5$ lies well above the starting value $f = 7$, and indeed the step overshoots
to $12.375$. :numref:`fig_mdl-opt-descent-lemma-slice` plots both curves as
functions of $\eta$. Along the search direction,
$\mathbf{g}^\top\mathrm{diag}(1, 10)\,\mathbf{g} = 1004$, so the true objective
is the parabola $7 - 104\,\eta + 502\,\eta^2$, while the quadratic ceiling is
$7 - 104\,\eta + 520\,\eta^2$. The two parabolas hug each other so closely
because this gradient points almost along the stiff axis, where the directional
curvature is nearly $L$.

![The descent lemma on the opening example. Along $-\mathbf{g}$ from $(2,1)$, the true objective is the lower parabola and the quadratic ceiling $f-\eta\|\mathbf{g}\|^2+\tfrac{L}{2}\eta^2\|\mathbf{g}\|^2$ is the upper one, tangent at $\eta=0$. Minimizing the ceiling at $\eta=1/L=0.1$ guarantees $f\le 1.8$, and the step achieves $1.62$. Beyond $2/L=0.2$ the ceiling rises above the starting value and promises nothing; at $\eta=0.25$ the objective climbs to $12.4$.](../img/mdl-opt-descent-lemma-slice.svg)
:label:`fig_mdl-opt-descent-lemma-slice`

In short, the step $\eta = 1/L$ minimizes the quadratic ceiling, and $f$ is
guaranteed to lie beneath that ceiling. :numref:`sec_mdl-single_variable_calculus`
develops the same geometric argument in one variable.

### Stationary Points Without Convexity

Surprisingly, the descent lemma alone—without any convexity assumption—already
places a quantitative limit on a long run: the gradient cannot stay large
forever.

**Proposition (gradient descent finds approximate stationary points).** *Let
$f$ be $L$-smooth and bounded below by $f_{\mathrm{low}}$, and run
:eqref:`eq_mdl-opt-gd-step` with $\eta = 1/L$. Then for every $K \ge 1$,*

$$
\min_{0 \le k < K} \|\nabla f(\mathbf{x}_k)\|^2 \;\le\; \frac{2L\,\big(f(\mathbf{x}_0) - f_{\mathrm{low}}\big)}{K}.
$$
:eqlabel:`eq_mdl-opt-stationarity-rate`

**Proof.** At $\eta = 1/L$, the descent lemma rearranges to
$\|\mathbf{g}_k\|^2 \le 2L\,(f(\mathbf{x}_k) - f(\mathbf{x}_{k+1}))$. Summing
over $k = 0, \ldots, K-1$, the right-hand side telescopes to
$2L\,(f(\mathbf{x}_0) - f(\mathbf{x}_K)) \le 2L\,(f(\mathbf{x}_0) - f_{\mathrm{low}})$.
Because the minimum of $K$ numbers is at most their average, dividing by $K$
gives the bound. $\blacksquare$

Thus, among the first $K$ iterates, at least one has squared gradient norm
$O(1/K)$, using nothing beyond a lower bound on $f$ and a smoothness constant.
Keep in mind that this result assumes a global $L$; ReLU networks are not
smooth, so for them the bound serves as an idealized model rather than a formal
guarantee.

A small gradient, however, does not guarantee a minimum. Consider
$f(x, y) = \tfrac12 x^2 + \cos y$, which is bounded below by $-1$ and
$1$-smooth (its Hessian $\mathrm{diag}(1, -\cos y)$ has eigenvalues in
$[-1, 1]$). Starting gradient descent with $\eta = 1$ at $(1, 0)$ drives $x$ to
$0$ in a single step and never changes $y$, since $\partial f/\partial y = -\sin y$
vanishes at $y = 0$. The algorithm halts at $(0, 0)$ with $f = 1$—a saddle
point where $f$ curves upward along $x$ and downward along $y$, far from the
true minimum value $-1$ at $(0, \pi)$. The theorem still holds, because the
gradient vanishes at the saddle, yet it cannot tell us which kind of stationary
point we reached. For convex
objectives, this pathology disappears: every stationary point is automatically
a global minimum (:numref:`sec_mdl-convexity`).

### Backtracking Line Search

Using $\eta = 1/L$ presupposes that we know $L$. In practice we rarely do, and
even when a global $L$ exists, the local curvature varies across the loss
landscape, making a single global constant overly conservative. A **line
search** adapts the step size at run time. Given a candidate step $\eta$, it
accepts the step only if the actual decrease achieves at least a fixed fraction
$c \in (0, 1)$ of the decrease predicted by the first-order model—the **Armijo
condition** (or sufficient decrease condition) :cite:`Armijo.1966`:

$$
f\big(\mathbf{x} - \eta\, \nabla f(\mathbf{x})\big) \;\le\; f(\mathbf{x}) - c\,\eta\, \|\nabla f(\mathbf{x})\|^2.
$$
:eqlabel:`eq_mdl-opt-armijo`

The right-hand side is simply the linear Taylor prediction with its slope
scaled down by $c$. **Backtracking** starts from an optimistic trial step
$\eta_0$ and repeatedly halves $\eta$ until :eqref:`eq_mdl-opt-armijo` is
satisfied. By the descent lemma, the test is guaranteed to pass as soon as
$1 - L\eta/2 \ge c$, or equivalently $\eta \le 2(1-c)/L$. Consequently, the
loop always terminates in finitely many halvings and returns a step
$\eta \ge \min\big(\eta_0,\, (1-c)/L\big)$—within a factor of two of the
theoretical threshold—without ever needing to know $L$ in advance.

Better still, this guarantee only requires $L$ on the region the iterates
actually visit. Take the quartic $f(x) = \tfrac14 x^4$, which has no global
$L$, starting at $x_0 = 3$ with $c = \tfrac12$. Because backtracking never
increases $f$, every iterate remains in $[-3, 3]$, where $f''(x) = 3x^2 \le 27$.
A trial step with $\eta \le 1/27$ sends $x$ to $x(1 - \eta x^2)$, which lies
between $\tfrac23 x$ and $x$, keeping the entire segment inside $[-3, 3]$ so
that $L = 27$ suffices locally. Since the quartic's curvature is steep far from
the origin and vanishes near zero, no single fixed step size suits the entire
trajectory. Let us compare both strategies from $x_0 = 3$: a fixed step
$\eta = 0.3$ (which would be completely safe near the origin) and backtracking
starting from $\eta_0 = 1$.

```{.python .input #gradient-based-optimization-backtracking}
quartic = lambda v: 0.25 * (v ** 4).sum()
quartic_grad = lambda v: v ** 3

x = np.array([3.0])
for _ in range(6):                            # fixed step, too large at x0
    x = x - 0.3 * quartic_grad(x)
print(f'fixed eta = 0.3 from x0 = 3:  |x_6| = {abs(x[0]):.2e}  (diverged)')

x, etas = np.array([3.0]), []
for _ in range(25):                           # backtracking line search
    g = quartic_grad(x)
    eta = 1.0
    while quartic(x - eta * g) > quartic(x) - 0.5 * eta * (g * g).sum():
        eta *= 0.5                            # Armijo fails: halve the step
    x = x - eta * g
    etas.append(eta)
print('accepted steps:', np.array(etas[:8]), '...')
print(f'backtracking from x0 = 3:  f(x_25) = {quartic(x):.2e}  (monotone descent)')
```

The fixed step explodes immediately: after just six iterations,
$|x_6| \approx 6.5 \times 10^{103}$. Because the update
$x \mapsto x(1 - \eta x^2)$ shrinks $|x|$ if and only if $\eta x^2 < 2$, a
fixed step from $x_0 = 3$ converges for $\eta < 2/9 \approx 0.22$ and diverges
above it. Notice
that this stability threshold is $2/9$, not $2/f''(x_0) = 2/27$; the
$2/f''(x_0)$ rule belongs to quadratics. A step toward the origin experiences
the average of $f''$ between $0$ and $x$—which is $x^2$—rather than its peak
value $3x^2$. The descent lemma must guard against the worst-case curvature
along the segment and therefore certifies only a third of the true safe range.

Backtracking, by contrast, accepts $0.031$ on the first step. Its previous
trial, $0.0625$, fails the Armijo test, which on this quartic holds if and only
if $\eta x^2 \le 0.456$ (roughly $\eta \le 0.05$ at $x_0 = 3$). As the iterate
moves toward the origin and the curvature flattens, the accepted steps grow
automatically—$0.0625, 0.125, 0.25, 0.5, 1.0$, each close to $0.3/x^2$—because
the line search probes the local curvature on the fly. Deep learning training
rarely uses backtracking, since a minibatch loss is too noisy to compare two
candidate steps reliably and each trial costs an extra forward pass. Instead,
training usually specifies the step size in advance via a schedule
(:numref:`sec_scheduler`).

## The Quadratic Model and the Condition Number
:label:`subsec_mdl-quadratic-model`

In the opening, we promised to explain why one bowl takes $69$ steps and
another takes $6{,}908$. Both objectives are quadratics, and on a quadratic we
can solve gradient descent in closed form. Remarkably, the iteration count
boils down to a single quantity: the ratio of the largest curvature to the
smallest.

### Ill-Conditioned Valleys

Before writing any algebra, look at :numref:`fig_mdl-opt-gd-bowl-vs-valley`,
which contrasts gradient descent on two quadratics. On a round bowl, the
iterates march almost straight to the minimum. In an elongated valley, the
walls are steep while the floor is flat: any step near the stability ceiling
imposed by the walls overshoots across the valley and bounces from wall to
wall, while barely inching along the floor. The result is the classic zigzag
trajectory. (The valley shown uses $\mathrm{diag}(1, 20)$ with $\eta = 0.097$,
just below its stability ceiling $0.1$, producing the contraction factors noted
in the caption.)

![Gradient descent on two quadratics $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$. Left: on a well-conditioned bowl, the iterates head nearly straight to the minimum. Right: in an ill-conditioned valley, a step near the stability ceiling makes the steep mode overshoot and alternate (factor $-0.94$) while the slow axis contracts by only $0.90$ per step.](../img/mdl-opt-gd-bowl-vs-valley.svg)
:label:`fig_mdl-opt-gd-bowl-vs-valley`

Quadratics deserve a complete analysis because every smooth function looks like
one near a strict minimum. Around a minimizer $\mathbf{x}^\star$ with
positive-definite Hessian $\mathbf{A}$, the second-order Taylor expansion
approximates $f$ by
$f(\mathbf{x}^\star) + \tfrac12 (\mathbf{x}-\mathbf{x}^\star)^\top \mathbf{A}\, (\mathbf{x}-\mathbf{x}^\star)$.
Shifting the minimizer to the origin and dropping the additive constant leaves
our model problem:

$$
\begin{aligned}
f(\mathbf{x}) &= \tfrac12\, \mathbf{x}^\top \mathbf{A}\, \mathbf{x}, \\
&\quad \mathbf{A} \textrm{ symmetric positive definite, with eigenvalues }
0 < \lambda_{\min} = \lambda_1 \le \cdots \le \lambda_d = \lambda_{\max}.
\end{aligned}
$$

Its gradient is $\mathbf{A}\mathbf{x}$ and its Hessian is $\mathbf{A}$, so its
smoothness constant is simply $L = \lambda_{\max}$. The ratio of the largest
curvature to the smallest,

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}},
$$

is the **condition number** of the problem. It equals $10$ for our opening bowl
$\mathrm{diag}(1, 10)$ and $10^3$ for the chapter's running example. Because
$\mathbf{A}$ is symmetric positive definite, $\kappa$ coincides with the
singular-value ratio $\kappa(\mathbf{A})$ introduced in
:numref:`subsec_mdl-condition-number`.

### Per-Mode Contraction and the Stability Ceiling

The zigzagging in :numref:`fig_mdl-opt-gd-bowl-vs-valley` comes directly from
the step-size ceiling enforced by the steepest direction. To pinpoint that
ceiling and see how severely it slows down the flat direction, let us solve
gradient descent on a quadratic exactly—starting, as always, with a single
variable.

Take $f(u) = \tfrac{\lambda}{2} u^2$ with $\lambda > 0$. One gradient step
sends $u_k$ to $u_{k+1} = u_k - \eta\lambda u_k$, so

$$
u_{k+1} = (1 - \eta\lambda)\, u_k, \qquad u_k = (1 - \eta\lambda)^k\, u_0 .
$$

Every iteration simply multiplies the error by the same scalar factor
$1 - \eta\lambda$. Convergence occurs if and only if $|1 - \eta\lambda| < 1$,
meaning $0 < \eta < 2/\lambda$. At the sweet spot $\eta = 1/\lambda$, the
iterate lands on $0$ in a single step. For $1/\lambda < \eta < 2/\lambda$, the
factor is negative, so $u_k$ flips sign at every step while shrinking toward
zero. For $\lambda = 10$, for instance, $\eta = 0.1$ gives factor $0$,
$\eta = 0.19$ gives $-0.9$, and $\eta = 0.21$ gives $-1.1$, which diverges.

The $d$-dimensional quadratic keeps the exact same structure. Since
$\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x}$, the update
:eqref:`eq_mdl-opt-gd-step` becomes

$$
\mathbf{x}_{k+1} = (\mathbf{I} - \eta \mathbf{A})\, \mathbf{x}_k .
$$

Each step applies the same linear operator, so after $k$ steps
$\mathbf{x}_k = (\mathbf{I} - \eta\mathbf{A})^k\mathbf{x}_0$. Because
$\mathbf{A}$ is symmetric, it admits an orthonormal basis of eigenvectors
(:numref:`sec_mdl-eigendecompositions`). Express $\mathbf{x}_k$ in this
eigenbasis with coordinates $u_i^{(k)}$. The matrix
$\mathbf{I} - \eta\mathbf{A}$ shares the same eigenvectors with eigenvalues
$1 - \eta\lambda_i$, so the coordinates decouple completely and evolve
independently:

$$
u_i^{(k+1)} = (1 - \eta \lambda_i)\, u_i^{(k)},
\qquad \textrm{hence} \qquad
u_i^{(k)} = (1 - \eta \lambda_i)^k\, u_i^{(0)}.
$$
:eqlabel:`eq_mdl-opt-per-mode`

Thus, gradient descent on a $d$-dimensional quadratic is nothing more than $d$
independent copies of the one-variable recursion—one per eigenvalue, or
**mode**. Two consequences follow immediately from the factors
$1 - \eta\lambda_i$:

* **Convergence** requires $|1 - \eta \lambda_i| < 1$ for every mode, which
  reduces to $0 < \eta < 2/\lambda_{\max} = 2/L$. This threshold is the
  **stability ceiling**. On quadratics, the bound $2/L$ that the descent lemma
  guaranteed as sufficient is also strictly necessary. Push $\eta$ past it, and
  the stiffest mode's factor drops below $-1$: that coordinate oscillates with
  growing amplitude even while all other modes decay cleanly.
* **Speed** is governed by the slowest mode. The error norm decays as
  $\rho(\eta)^k$, where $\rho(\eta) = \max_i |1 - \eta\lambda_i|$ is the
  **spectral radius** of $\mathbf{I} - \eta\mathbf{A}$—the largest eigenvalue
  magnitude.

On $\mathbf{A} = \mathrm{diag}(1, 10)$, the stability ceiling is $\eta = 0.2$.
Before running the sweep below, predict which step size leaves the smallest
error after $60$ steps from $(1, 1)$: the cautious $0.02$, the descent-lemma
favorite $0.1 = 1/L$, or something closer to the ceiling?

```{.python .input #gradient-based-optimization-eta-sweep}
A = np.diag([1.0, 10.0])                      # eigenvalues 1 and 10, so L = 10
x0 = np.array([1.0, 1.0])

def run_gd(A, x0, eta, steps):
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - eta * (A @ xs[-1]))
    return np.array(xs)

print(' eta    |1-eta*1|  |1-eta*10|  spectral radius   |x_60|')
for eta in [0.02, 0.10, 2 / 11, 0.19, 0.20, 0.21]:
    f1, f10 = abs(1 - eta * 1.0), abs(1 - eta * 10.0)
    print(f'{eta:5.3f}    {f1:6.3f}     {f10:6.3f}        {max(f1, f10):6.3f}     '
          f'{np.linalg.norm(run_gd(A, x0, eta, 60)[-1]):9.1e}')
```

The cautious $\eta = 0.02$ is safe but sluggish ($\rho = 0.98$, leaving an
error of $0.3$ after $60$ steps). At $\eta = 0.1$, the stiff mode vanishes in a
single step, yet the slow mode still limits the overall rate to $\rho = 0.9$.
Interestingly, $\eta = 0.19$ shares that exact same rate $\rho = 0.9$, for the
opposite reason: now the stiff mode oscillates with factor $-0.9$. Between
them, $\eta = 2/11 \approx 0.182$ balances both extremes and beats every other
choice by two orders of magnitude ($\rho = 0.818$, error $8.3 \times 10^{-6}$).
Right at the ceiling $\eta = 0.2$, the stiff mode has factor $-1$, flipping
between $\pm 1$ forever so $\|\mathbf{x}_{60}\| = 1.0$. Step just past it to
$\eta = 0.21$, and sixty iterations inflate the error to $3 \times 10^{2}$.

### The Optimal Step

Within the stable range, which fixed step size $\eta$ converges fastest? The
sweep points directly to the answer: balance the two extreme modes. For any
fixed $\eta$, the map $\lambda \mapsto |1 - \eta\lambda|$ traces a V-shape with
its vertex at $\lambda = 1/\eta$. On the interval
$[\lambda_{\min}, \lambda_{\max}]$, a V-shaped function attains its maximum at
one of the endpoints, so
$\rho(\eta) = \max\big(|1 - \eta\lambda_{\min}|,\, |1 - \eta\lambda_{\max}|\big)$.
Once $\eta$ passes $1/\lambda_{\max}$, increasing it further helps the slow
mode at $\lambda_{\min}$ while hurting the stiff mode at $\lambda_{\max}$.

**Proposition (optimal fixed step on a quadratic).** *For
$f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathbf{A} \mathbf{x}$, the
convergence factor $\rho(\eta) = \max_i |1 - \eta\lambda_i|$ is minimized by*

$$
\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}},
\qquad
\rho(\eta^\star) = \frac{\kappa - 1}{\kappa + 1},
\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}.
$$
:eqlabel:`eq_mdl-opt-optimal-step`

**Proof.** For every $\eta$, we have
$\rho(\eta) \ge \max(1 - \eta\lambda_{\min},\, \eta\lambda_{\max} - 1)$, and
the maximum of two numbers is at least any convex combination of them. Choosing
the weights $\lambda_{\max}/(\lambda_{\min}+\lambda_{\max})$ and
$\lambda_{\min}/(\lambda_{\min}+\lambda_{\max})$ makes the $\eta$-terms cancel
cleanly:

$$
\rho(\eta) \;\ge\; \frac{\lambda_{\max}(1 - \eta\lambda_{\min}) + \lambda_{\min}(\eta\lambda_{\max} - 1)}{\lambda_{\min} + \lambda_{\max}}
= \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}.
$$

Equality holds only when the two terms agree,
$1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$, which solves to
$\eta = \eta^\star$. At $\eta^\star$, both endpoints equal
$(\kappa - 1)/(\kappa + 1)$, and the V-shape keeps every interior eigenvalue
below that height, so the bound is attained. $\blacksquare$

:numref:`fig_mdl-opt-eta-tent` illustrates this proof on
$\mathrm{diag}(1, 10)$: the optimal V is the one whose two endpoints sit at
the same height. Here $\eta^\star = 2/11$ and $\rho = 9/11 = 0.818$, so gaining
six digits takes $\log 10^6/\log(11/9) \approx 69$ steps. On the chapter's
running example ($\kappa = 1000$), $\rho = 999/1001$, and those same six digits
demand $6{,}908$ steps.

![The per-mode contraction factors $|1-\eta\lambda|$ form a V with its vertex at $\lambda = 1/\eta$, and the convergence factor $\rho(\eta)$ is the larger of its values at the extreme eigenvalues $\lambda_{\min} = 1$ and $\lambda_{\max} = 10$. Choosing $\eta = 0.1$ eliminates the stiff mode immediately but leaves the slow mode contracting at $0.9$; the optimal step $\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$ equalizes both endpoints at $(\kappa-1)/(\kappa+1) = 9/11$. Lowering either endpoint would raise the other.](../img/mdl-opt-eta-tent.svg)
:label:`fig_mdl-opt-eta-tent`

When $\kappa = 1$ (a perfectly round bowl), $\rho = 0$ and a single step lands
on the minimum. For large $\kappa$, $(\kappa-1)/(\kappa+1) \approx 1 - 2/\kappa$,
so $\log(1/\rho) \approx 2/\kappa$, and shrinking the error by a factor of
$\epsilon$ takes roughly $\tfrac{\kappa}{2}\log\tfrac1\epsilon$ iterations:
**the iteration count of gradient descent grows linearly with the condition
number**. The bottleneck is easy to see in one sentence: stability forces the
step to obey the stiffest mode, $\eta \lesssim 2/\lambda_{\max}$, leaving the
flattest mode to advance by only $\eta\,\lambda_{\min} \approx 2/\kappa$ per
iteration.

At $\eta^\star$, the factor $\rho$ is even stronger than an asymptotic rate. On
$\mathrm{diag}(1, 10)$, the two per-mode factors are $+9/11$ and $-9/11$ (the
$\eta = 2/11$ row of the sweep): the slow mode contracts monotonically while
the stiff mode alternates in sign. Because both coordinates shrink by the same
magnitude, $\|\mathbf{x}_k\|$ contracts by $(\kappa-1)/(\kappa+1)$ on every
single step, not just on average.

### The Edge of Stability
:label:`subsec_mdl-edge-of-stability`

The stability ceiling suggests a simple recipe for neural networks: compute the
**sharpness**—the largest Hessian eigenvalue $\lambda_{\max}(\nabla^2 f)$—and
set $\eta < 2/\lambda_{\max}$. In practice, training a neural network does not
cooperate, because the Hessian evolves along the trajectory. Under full-batch
gradient descent, the sharpness typically climbs until it hits roughly
$2/\eta$ and then hovers right around that threshold; meanwhile, the loss
decreases over the long run even though it jumps upward on many individual
steps :cite:`Cohen.Kaur.Li.ea.2021`. :numref:`subsec_gd-edge-of-stability` runs
this experiment on a small $\tanh$ network and examines it in detail. Here we
ask a narrower question: how much of this behavior does our quadratic model
explain?

Roughly half. In the Hessian's eigenbasis, the top mode's multiplier is
$1 - \eta\lambda_{\max}$, which drops below $-1$ precisely when
$\lambda_{\max} > 2/\eta$. That coordinate then alternates in sign with growing
amplitude, explaining how a loss that trends downward overall can spike on
individual steps. On a fixed quadratic, however, that growth would run away to
infinity. What pulls the sharpness back down instead lies outside the quadratic
approximation: the third-order Taylor term, which the quadratic model ignores,
turns the oscillation into a gradual drift toward flatter regions of lower
sharpness :cite:`Damian.Nichani.Lee.2023`. Exercise
:exref:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum` shows
that adding momentum shifts this stability threshold to $(2 + 2\beta)/\eta$.

### From Quadratics to Convex Functions

What carries over beyond quadratics is the linear dependence on $\kappa$. As we
will formalize in :numref:`sec_mdl-convexity`, a differentiable function is
convex if and only if it lies on or above every tangent hyperplane, and
$\mu$-strongly convex (:eqref:`eq_mdl-opt-strong-convexity`) if
$f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \tfrac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|_2^2$
for all $\mathbf{x}, \mathbf{y}$. If $f$ is $L$-smooth and $\mu$-strongly
convex, then at every point it is sandwiched
between two quadratics of curvatures $\mu$ and $L$. The ratio $\kappa = L/\mu$
replaces $\lambda_{\max}/\lambda_{\min}$, and gradient descent with
$\eta = 1/L$ satisfies

$$
f(\mathbf{x}_k)-f^\star \le \Big(1-\frac{1}{\kappa}\Big)^k\big(f(\mathbf{x}_0)-f^\star\big),
$$
:eqlabel:`eq_mdl-opt-gd-rate-strongly-convex`

which again requires $O(\kappa\log\tfrac1\epsilon)$ iterations. If $f$ is
smooth and convex without being strongly convex, gradient descent still
converges to the global optimum, though at the slower sublinear rate $O(1/k)$;
drop convexity altogether, and only the stationarity bound
:eqref:`eq_mdl-opt-stationarity-rate` remains. The proofs rely on convexity
and are developed in :numref:`subsec_mdl-convergence-rates`, which also
compares these general bounds against the sharper rate attained on quadratics.

## Momentum and Acceleration
:label:`subsec_mdl-momentum-acceleration`

Gradient descent requires $O(\kappa)$ iterations because a single step size
must simultaneously accommodate the stiffest and flattest modes. Momentum
keeps the same single gradient evaluation per step, yet changes the update rule
so that both modes can contract at the same rate.

### Heavy-Ball Momentum

The geometry of the valley immediately suggests what to change. Along the steep
walls, consecutive gradients point in opposite directions (the zigzagging in
:numref:`fig_mdl-opt-gd-bowl-vs-valley`), so averaging recent gradients cancels
the oscillation. Along the flat floor, consecutive gradients are small but
consistently aligned, so accumulating them builds speed. One simple mechanism
does both: give the iterate a velocity with memory. **Heavy-ball momentum**
:cite:`Polyak.1964` replaces the plain gradient step with

$$
\mathbf{v}_{k+1} = \beta\, \mathbf{v}_k - \eta\, \nabla f(\mathbf{x}_k),
\qquad
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1},
$$
:eqlabel:`eq_mdl-opt-heavy-ball`

where $\beta \in [0, 1)$ is the momentum parameter. The velocity $\mathbf{v}_k$
is an exponentially decaying sum of past gradients. Since
$\mathbf{v}_k = \mathbf{x}_k - \mathbf{x}_{k-1}$, eliminating $\mathbf{v}$
gives the equivalent two-step form

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \nabla f(\mathbf{x}_k) + \beta\,(\mathbf{x}_k - \mathbf{x}_{k-1}).
$$

Each update takes a gradient step and adds $\beta$ times the previous
displacement; setting $\beta = 0$ recovers ordinary gradient descent.

On a quadratic, the modes once again decouple. In a single mode with gradient
$\lambda u_k$, the two-step update reads
$u_{k+1} = (1 + \beta - \eta\lambda)\, u_k - \beta\, u_{k-1}$, so each iterate
depends on its two predecessors. A second-order recurrence of this form is the
discrete counterpart of a damped harmonic oscillator—a heavy ball rolling in a
bowl with friction $1 - \beta$—where $\beta$ controls the damping.

:numref:`fig_mdl-opt-momentum-damping` displays the three resulting regimes on
$\mathrm{diag}(1, 10)$, all using gradient descent's optimal step
$\eta = 2/11$. With too little momentum, the path zigzags much like plain
gradient descent. With just the right amount, it settles smoothly after a
single overshoot. With too much momentum, the ball carries excess inertia and
spirals around the minimum. Next, we attach exact convergence rates to all
three regimes—and discover that achieving the fastest possible rate also calls
for a larger step size.

![Heavy ball on the $\kappa=10$ valley $\tfrac12(x^2+10y^2)$, with all three runs using gradient descent's step $\eta=2/11$. Too little momentum zigzags like gradient descent (the stiff mode's coefficient $1+\beta-\eta\lambda_{\max}$ is negative, giving a negative real root); the critically damped $\beta$ for this step is fastest; too much momentum spirals around the minimum. Reaching the optimal $\sqrt\kappa$ rate of $0.52$ requires the larger step $\eta_{\mathrm{HB}}$ of :eqref:`eq_mdl-opt-hb-tuning`, at which $\beta=0.05$ would diverge.](../img/mdl-opt-momentum-damping.svg)
:label:`fig_mdl-opt-momentum-damping`

### The $\sqrt{\kappa}$ Law

We now show that with the right pair $(\eta, \beta)$, every mode contracts at
the identical rate $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$. For $\kappa = 10$,
this optimal pair is $\beta^\star \approx 0.27$ and
$\eta_{\mathrm{HB}} \approx 0.23$, dropping the contraction factor from
gradient descent's $0.82$ to $0.52$—reaching six digits in $27$ steps instead
of $69$. Remarkably, the entire result follows from a single algebraic fact
about one mode.

**Proposition (heavy ball contracts every mode at $\sqrt\beta$).** *Let
$0 < \beta < 1$. On $f(u) = \tfrac{\lambda}{2}u^2$, the heavy-ball iterates
obey $u_{k+1} = (1 + \beta - \eta\lambda)u_k - \beta u_{k-1}$. Whenever
$(1 - \sqrt\beta)^2 \le \eta\lambda \le (1 + \sqrt\beta)^2$, both roots of the
characteristic polynomial have modulus $\sqrt\beta$, and
$|u_k| \le C\,k\,\beta^{k/2}$ for $k \ge 1$ and a constant $C$ depending on
$u_0$, $u_1$, and $\eta\lambda$. The linear factor $k$ is needed only at the
two endpoints, where the root is repeated.*

**Proof.** Substitute the trial solution $u_k = z^k$ into the recurrence and
divide by $z^{k-1}$ to obtain the characteristic equation

$$
z^2 - (1 + \beta - \eta\lambda)\, z + \beta = 0.
$$
:eqlabel:`eq_mdl-opt-hb-char-poly`

The product of the two roots is $\beta$. They form a complex-conjugate pair
whenever the discriminant $(1 + \beta - \eta\lambda)^2 - 4\beta$ is negative,
that is, when $|1 + \beta - \eta\lambda| < 2\sqrt\beta$, which rearranges to
$(1 - \sqrt\beta)^2 < \eta\lambda < (1 + \sqrt\beta)^2$. Because complex
conjugates $z, \bar z$ satisfy $|z|^2 = z\bar z = \beta$, both roots have
modulus $\sqrt\beta$ regardless of $\lambda$, and the general solution
$u_k = a z^k + b \bar z^k$ satisfies $|u_k| \le (|a| + |b|)\beta^{k/2}$. At
the two endpoints, the discriminant vanishes and we get a double real root
$(1 + \beta - \eta\lambda)/2 = \pm\sqrt\beta$. A repeated root $z$ yields the
general solution $u_k = (a + bk)z^k$, which accounts for the extra factor $k$.
$\blacksquare$

**Corollary (the tuning).** Place the two extreme modes right at the endpoints
of the complex-root interval: $\eta\lambda_{\min} = (1 - \sqrt\beta)^2$ and
$\eta\lambda_{\max} = (1 + \sqrt\beta)^2$. Every intermediate eigenvalue then
falls inside the interval, so every single mode has root modulus $\sqrt\beta$.
Taking square roots gives
$\sqrt{\eta\lambda_{\max}} - \sqrt{\eta\lambda_{\min}} = 2\sqrt\beta$ and
$\sqrt{\eta\lambda_{\max}} + \sqrt{\eta\lambda_{\min}} = 2$. Solving the
second equation for $\eta$ and dividing the first by the second gives
$\sqrt\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$. With the tuning

$$
\eta_{\mathrm{HB}} = \left(\frac{2}{\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}}\right)^{\!2},
\qquad
\beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{\!2},
$$
:eqlabel:`eq_mdl-opt-hb-tuning`

heavy ball contracts every mode at the rate

$$
\rho_{\mathrm{HB}} = \sqrt{\beta^\star} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
\;\approx\; 1 - \frac{2}{\sqrt{\kappa}}.
$$
:eqlabel:`eq_mdl-opt-hb-rate`

Compare this with :eqref:`eq_mdl-opt-optimal-step`: momentum replaces $\kappa$
with $\sqrt\kappa$, slashing the iteration complexity from
$O(\kappa \log \tfrac1\epsilon)$ to $O(\sqrt{\kappa} \log \tfrac1\epsilon)$
:cite:`Polyak.1964`. At finite $k$, the double roots at the two extreme modes
add a modest logarithmic overhead: the error scales as $k\rho^k$, so with
$\log(1/\rho) \approx 2/\sqrt\kappa$, reaching accuracy $\epsilon$ solves
$k \approx \tfrac{\sqrt\kappa}{2}\big(\log\tfrac1\epsilon + \log k\big)$—the
asymptotic law plus a correction of order $\sqrt\kappa\log\kappa$.

:numref:`fig_mdl-opt-hb-eigenvalues` makes this proof visual for $\kappa = 10$.
On the left, the modulus of the dominant root sits flat at $\sqrt\beta$ across
the entire complex-root interval. On the right, in the complex plane, every
mode's root pair lies squarely on the circle of radius
$\sqrt{\beta^\star} = 0.52$, whereas gradient descent's real multipliers
stretch out to $\pm 0.82$. By allowing the roots to move off the real axis,
momentum lets the stiff and flat modes share a single small modulus instead of
forcing the flat mode to crawl at $1 - 2/\kappa$.

The proposition also explains the exact rates of the three trajectories in
:numref:`fig_mdl-opt-momentum-damping`, where $\eta = 2/11$ places the two
modes at $\eta\lambda = 0.18$ and $1.82$. At $\beta = 0.85$, both modes lie
strictly inside the interval: the contraction rate is $\sqrt{0.85} = 0.92$,
and the complex roots produce the spiraling overshoot. The smallest $\beta$
whose interval still captures the slow mode,
$\beta = (1 - \sqrt{2/11})^2 \approx 0.33$, is the fastest choice for this
step size, achieving rate $0.57$. At $\beta = 0.05$, the interval
$[0.60, 1.50]$ misses both modes: the slow mode gets a real root $0.81$, while
the stiff mode gets a negative real root that flips sign on every step—an
alternation that a continuous-time physical oscillator can never produce.

![Heavy ball at the tuning of :eqref:`eq_mdl-opt-hb-tuning` on the $\kappa=10$ example. (a) For fixed $\beta$, the modulus of the larger root is flat at $\sqrt\beta$ wherever the roots are complex; the optimal tuning places the two extreme modes at the ends of that interval. (b) In the complex plane, every mode's root pair (labeled by its eigenvalue $\lambda$) lies on the circle of radius $\sqrt{\beta^\star}=0.52$, while gradient descent's factors span $[-0.82,0.82]$ on the real axis. The extreme modes are double roots, producing the factor $k$ in the error.](../img/mdl-opt-hb-eigenvalues.svg)
:label:`fig_mdl-opt-hb-eigenvalues`

Be aware that this $\sqrt\kappa$ theorem holds strictly for quadratics.
:citet:`Lessard.Recht.Packard.2016` constructed a one-dimensional, smooth,
strongly convex function on which heavy ball with the tuning
:eqref:`eq_mdl-opt-hb-tuning` fails to converge from certain initializations,
trapping the iterates in a stable limit cycle (exercise
:exref:`ex_mdl-gradient-based-optimization-heavy-ball-limit-cycle`). Thus, the
widespread use of heavy-ball momentum in deep learning
:cite:`Sutskever.Martens.Dahl.ea.2013` rests on the local quadratic picture
and empirical success rather than a global convergence theorem.

### Nesterov's Look-Ahead

One small modification recovers a global guarantee: evaluate the gradient not
at the current position $\mathbf{x}_k$, but at the look-ahead point
$\mathbf{x}_k + \beta\,\mathbf{v}_k$ where momentum is about to carry us.
**Nesterov's accelerated gradient** :cite:`Nesterov.1983` updates via

$$
\mathbf{v}_{k+1} = \beta\, \mathbf{v}_k - \eta\, \nabla f(\mathbf{x}_k + \beta\, \mathbf{v}_k),
\qquad
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}.
$$
:eqlabel:`eq_mdl-opt-nesterov`

That shift in evaluation point is the only difference from heavy ball.
Intuitively, if momentum is about to carry the iterate past the minimum, the
gradient at the look-ahead point already points back uphill and applies the
brakes before the overshoot occurs.

**Proposition (Nesterov on quadratics).** *Let $\kappa = L/\mu$. With
$\eta = 1/L$ and $\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$, every mode of a
quadratic with eigenvalues in $[\mu, L]$ has characteristic roots of modulus
at most $1 - 1/\sqrt\kappa$, with a double root at the slow mode
$\lambda = \mu$. Consequently, the error is bounded by
$C\,k\,(1 - 1/\sqrt\kappa)^k$.*

**Proof.** In a single mode with eigenvalue $\lambda$, write
$a = 1 - \eta\lambda \in [0, 1 - 1/\kappa]$. The velocity update becomes
$v_{k+1} = a\beta v_k - \eta\lambda u_k$, giving
$u_{k+1} = a u_k + a\beta v_k$. Substituting $v_k = u_k - u_{k-1}$ yields the
recurrence $u_{k+1} = (1 + \beta)a\,u_k - \beta a\,u_{k-1}$, whose
characteristic polynomial $z^2 - (1 + \beta)a\,z + \beta a$ has discriminant
$a\,[(1 + \beta)^2 a - 4\beta]$. For our choice of $\beta$,
$(1 + \beta)^2 = 4\kappa/(\sqrt\kappa + 1)^2$. At the slow mode
$a = (\kappa - 1)/\kappa$, the bracket evaluates to
$4(\kappa - 1)/(\sqrt\kappa + 1)^2 - 4\beta = 4\beta - 4\beta = 0$, giving a
double root at $(1 + \beta)a/2 = (\sqrt\kappa - 1)/\sqrt\kappa = 1 - 1/\sqrt\kappa$.
Every stiffer mode has a smaller $a$, making the bracket strictly negative and
producing complex roots (or roots at $0$ when $a = 0$) of modulus
$\sqrt{\beta a} < \sqrt{\beta(1 - 1/\kappa)} = 1 - 1/\sqrt\kappa$.
$\blacksquare$

On quadratics, Nesterov's contraction factor is $0.684$ for $\kappa = 10$,
$0.900$ for $\kappa = 100$, and $0.968$ for $\kappa = 1000$—slightly slower
than heavy ball's $0.519$, $0.818$, and $0.939$, yet preserving the same
$\sqrt\kappa$ scaling. Crucially, unlike heavy ball, Nesterov's method keeps
its guarantees beyond quadratics, which we state here without proof
:cite:`Nesterov.2018`. On every $\mu$-strongly convex, $L$-smooth function,
Nesterov's update with the same constant $\beta$ satisfies
$f(\mathbf{x}_k) - f^\star \le C\,(1 - 1/\sqrt\kappa)^k$. (Mind the units:
this bounds the function gap $f - f^\star$, which scales as the squared
distance, so its exponent is half the exponent in the quadratic proposition.)
On smooth convex functions without strong convexity, a schedule
$\beta_k \to 1$ achieves $f(\mathbf{x}_k) - f^\star = O(1/k^2)$, improving on
gradient descent's $O(1/k)$. Moreover, this speed cannot be beaten in general:
no method that builds its iterates solely from past gradients can surpass
$O(1/k^2)$ or the $\sqrt\kappa$ dependence in the worst case, provided the
dimension is large relative to the number of iterations
:cite:`Nemirovski.Yudin.1983`. Exercise
:exref:`ex_mdl-gradient-based-optimization-first-order-lower-bound` constructs
the worst-case function that proves this limit.

Let us now verify the iteration counts we quoted at the start of the section.
On $\mathrm{diag}(1, \kappa)$ starting from $(1, 1)$, gaining six digits
should cost gradient descent roughly $\tfrac{\kappa}{2}\log 10^6 \approx 6.9\,\kappa$
steps and heavy ball somewhat more than $6.9\sqrt\kappa$.

```{.python .input #gradient-based-optimization-momentum}
def iterations(lam, eta, beta=0.0, lookahead=False, tol=1e-6):
    """Steps until |x_k| <= tol |x_0| on f(x) = sum_i lam_i x_i^2 / 2."""
    x, v = np.array([1.0, 1.0]), np.zeros(2)
    n0 = np.linalg.norm(x)
    for k in range(1, 10**6):
        g = lam * (x + beta * v) if lookahead else lam * x
        v = beta * v - eta * g
        x = x + v
        if np.linalg.norm(x) <= tol * n0:
            return k

print('kappa      GD   heavy ball   Nesterov     GD/kappa   HB/sqrt(kappa)')
for kappa in [10.0, 100.0, 1000.0]:
    lam, sk = np.array([1.0, kappa]), np.sqrt(kappa)
    gd = iterations(lam, 2 / (1 + kappa))
    hb = iterations(lam, (2 / (1 + sk))**2, ((sk - 1) / (sk + 1))**2)
    na = iterations(lam, 1 / kappa, (sk - 1) / (sk + 1), lookahead=True)
    print(f'{kappa:5.0f} {gd:7d} {hb:9d} {na:10d}      '
          f'{gd / kappa:7.1f}     {hb / sk:7.1f}')
```

The table confirms every prediction. Gradient descent's step count divided by
$\kappa$ stays pinned at $6.9$, reaching $6{,}908$ iterations at
$\kappa = 1000$, while heavy ball finishes in just $315$. Notice that heavy
ball's ratio $k/\sqrt\kappa$ drifts gently upward—$8.5$, $9.3$, $10.0$—because
each extreme mode has a double root: its $2\times 2$ iteration matrix forms a
Jordan block, a repeated eigenvalue with only a single eigenvector
(:numref:`subsec_mdl-jordan`), whose $k$-th power brings in the factor $k$.
The refined prediction $k \approx \tfrac{\sqrt\kappa}{2}(\log 10^6 + \log k)$
matches closely—$8.6$, $9.2$, $9.8$—and at $\kappa = 10^4$, heavy ball takes
$1{,}056$ steps, a ratio of $10.6$ (exercise
:exref:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`).
Nesterov's $508$ iterations at $\kappa = 1000$ sit cleanly between the two,
just as its contraction factor $0.968$ versus heavy ball's $0.939$ predicts.
Because all three methods evaluate only one gradient per step, momentum's
speedup costs no extra gradient work. Yet $\sqrt\kappa$ is also the limit of
what gradients alone can achieve: by the first-order lower bound, no method
built solely from past gradients can beat it in high dimensions. Removing the
dependence on $\kappa$ altogether requires a step that actually models the
curvature (:numref:`sec_mdl-curvature-methods`).

## Summary

Smoothness caps how far a gradient step can be trusted: on an $L$-smooth
objective, descent is guaranteed up to $\eta = 2/L$. On a quadratic, the best
fixed step contracts the error by $(\kappa - 1)/(\kappa + 1)$ per iteration,
forcing the step count to grow linearly with the condition number $\kappa$
because stability is governed by the stiffest mode while progress is governed
by the flattest. Momentum equalizes every mode at the shared contraction rate
$\sqrt\beta$, and its tuned factor $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ cuts
the iteration count from $O(\kappa)$ to $O(\sqrt\kappa)$ at no extra gradient
cost. On neural networks, sharpness rises to the stability ceiling $2/\eta$ of
the chosen step size, a phenomenon that the quadratic model explains only in
part. Three questions remain open: what an optimizer can achieve when it models
curvature directly (:numref:`sec_mdl-curvature-methods`), when a stationary
point is guaranteed to be a global minimum (:numref:`sec_mdl-convexity`), and
what changes when gradients are noisy
(:numref:`sec_mdl-stochastic-gradient-methods`).

## Exercises

1. **Preconditioned descent directions.** The proposition of
   :numref:`subsec_mdl-descent-directions` measures the length of a direction
   with the Euclidean norm. Choosing a different norm yields a different
   steepest-descent direction.
    1. Show that $\mathbf{d} = -\mathbf{P}^{-1}\nabla f(\mathbf{x})$ is a
       descent direction for every positive definite matrix $\mathbf{P}$.
    1. Measure length by
       $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\, \mathbf{d}}$.
       Show that the minimizer of $\nabla f(\mathbf{x})^\top \mathbf{d}$ over
       $\|\mathbf{d}\|_{\mathbf{P}} = 1$ is proportional to
       $-\mathbf{P}^{-1}\nabla f(\mathbf{x})$, and identify the $\mathbf{P}$
       that recovers Newton's direction
       $-\big(\nabla^2 f(\mathbf{x})\big)^{-1}\nabla f(\mathbf{x})$ (which we
       will study in :numref:`sec_mdl-curvature-methods`,
       :eqref:`eq_mdl-opt-newton-step`). The
       change of variables $\mathbf{u} = \mathbf{P}^{1/2}\mathbf{d}$ reduces
       the problem to the Euclidean case.
   :exlabel:`ex_mdl-gradient-based-optimization-show-that-is-de`

1. [code] **Smoothness constants.** The smoothness constant $L$ of
   :eqref:`eq_mdl-opt-smoothness`
   ($\|\nabla f(\mathbf{y}) - \nabla f(\mathbf{x})\|_2 \le L\|\mathbf{y} - \mathbf{x}\|_2$)
   bounds the curvature of $f$, and for a
   logistic loss it can be computed directly from the data.
    1. Let $f$ be twice continuously differentiable and $L$-smooth. Show that
       $\|\nabla^2 f(\mathbf{x})\,\mathbf{v}\| \le L\,\|\mathbf{v}\|$ for all
       $\mathbf{x}$ and $\mathbf{v}$, and conclude that every Hessian
       eigenvalue lies in $[-L, L]$. This establishes the converse direction of
       the equivalence stated after :eqref:`eq_mdl-opt-smoothness`.
    1. Consider
       $f(\mathbf{w}) = \frac1n \sum_{i=1}^n \log\big(1 + \exp(-y_i\, \mathbf{x}_i^\top \mathbf{w})\big) + \frac{\nu}{2}\|\mathbf{w}\|^2$
       with labels $y_i = \pm 1$ and $\nu = 0.1$—the `loss` function that we
       will use in the data cell of :numref:`subsec_mdl-why-not-newton`, where
       $\mathbf{x}_i^\top$ is the $i$-th row of
       $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n = 200$, $d = 5$, with
       independent standard Gaussian entries). Show that
       $L = \lambda_{\max}(\mathbf{X}^\top \mathbf{X})/(4n) + \nu$ is a valid
       smoothness constant and that the Hessian attains this bound at
       $\mathbf{w} = \mathbf{0}$.
    1. Compute this $L$ for a $200 \times 5$ matrix $\mathbf{X}$ of standard
       Gaussian entries and labels $y_i \in \{\pm 1\}$ drawn with
       $P(y_i = 1) = 1/(1 + \exp(-\mathbf{x}_i^\top \mathbf{w}_{\mathrm{true}}))$
       for $\mathbf{w}_{\mathrm{true}} = (1, -2, 1.5, 0, 2)$. It comes out near $0.4$ for
       every draw; explain why using
       $\lambda_{\max}(\mathbf{X}^\top\mathbf{X})/n \approx (1 + \sqrt{d/n})^2$
       for standard Gaussian entries. Count how many steps $\eta = 0.5$ (the
       reference step) and $\eta = 1/L$ each need from $\mathbf{w} = \mathbf{0}$
       to reach $\|\nabla f(\mathbf{w})\| \le 10^{-10}$. Which is faster, and
       does $\eta = 0.5$ stay below $2/L$?
   :exlabel:`ex_mdl-gradient-based-optimization-smoothness-constants`

1. **Steepest descent in other norms.** ● Problem
   :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b measures length
   by a norm $\|\cdot\|_{\mathbf{A}}$ induced by an inner product. Norms that
   do not come from an inner product produce steepest directions of a very
   different form.
    1. Find the minimizers of $\nabla f(\mathbf{x})^\top \mathbf{d}$ over
       $\|\mathbf{d}\|_\infty = 1$ and over $\|\mathbf{d}\|_1 = 1$, together
       with their minimal slopes, and state when each minimizer is unique.
    1. Let the parameter be a matrix whose gradient $\mathbf{G}$ has reduced
       SVD $\mathbf{G} = \mathbf{U}\, \mathrm{diag}(\sigma_i)\, \mathbf{V}^\top$,
       let the directional slope of $\mathbf{D}$ be
       $\operatorname{tr}(\mathbf{G}^\top \mathbf{D})$, and measure length by
       the spectral norm $\|\mathbf{D}\|_2$. Show that the minimal slope over
       $\|\mathbf{D}\|_2 \le 1$ is $-\sum_i \sigma_i$, attained at
       $\mathbf{D} = -\mathbf{U} \mathbf{V}^\top$—the polar factor that Muon
       applies to its momentum matrix (:numref:`sec_muon`). Writing
       $\operatorname{tr}(\mathbf{G}^\top \mathbf{D}) = \sum_i \sigma_i\, \mathbf{u}_i^\top \mathbf{D}\, \mathbf{v}_i$
       is a helpful first step. This is the steepest-descent perspective of
       :citet:`Bernstein.Newhouse.2024`.
   :exlabel:`ex_mdl-gradient-based-optimization-steepest-other-norms`

1. [code] **Heavy ball: the double root and the factor $k$.** ● Apply heavy
   ball :eqref:`eq_mdl-opt-heavy-ball`
   ($\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})$)
   to $f(u) = \tfrac{\lambda}{2} u^2$.
    1. Write one iteration as a linear map on the state $(u_k, u_{k-1})$ and
       show that the characteristic polynomial of its $2 \times 2$ matrix
       is :eqref:`eq_mdl-opt-hb-char-poly`.
    1. At the tuning :eqref:`eq_mdl-opt-hb-tuning`, the two extreme modes have
       double eigenvalues $\sqrt{\beta^\star}$ and $-\sqrt{\beta^\star}$. Show
       that the matrix then has only a single eigenvector, and that the $k$-th
       power of the Jordan block
       $\left(\begin{smallmatrix}\rho&1\\0&\rho\end{smallmatrix}\right)$ has
       entry $k\rho^{k-1}$ above the diagonal.
    1. With $\|\mathbf{e}_k\| \approx k\rho^k$ and
       $\log(1/\rho) \approx 2/\sqrt\kappa$, solve
       $k = \tfrac{\sqrt\kappa}{2}\big(\log\tfrac1\epsilon + \log k\big)$
       numerically for $\epsilon = 10^{-6}$ and
       $\kappa \in \{10, 100, 1000\}$, and compare $k/\sqrt\kappa$ with the
       momentum cell's column $8.5$, $9.3$, $10.0$.
    1. Run the momentum cell's heavy ball at $\kappa = 10^4$ and compare its
       step count with your prediction. Explain why the ratio $k/\sqrt\kappa$
       never settles to a constant as $\kappa$ grows.
   :exlabel:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`

1. [code] **Heavy ball beyond quadratics.** ●
   :citet:`Lessard.Recht.Packard.2016` define a one-dimensional function
   through its gradient:

   $$
   \nabla f(x) = \begin{cases} 25x, & x < 1, \\ x + 24, & 1 \le x < 2, \\ 25x - 24, & x \ge 2. \end{cases}
   $$

    1. Show that $f$ is $\mu$-strongly convex
       ($f(y) \ge f(x) + \nabla f(x)(y - x) + \tfrac{\mu}{2}(y - x)^2$) with
       $\mu = 1$ and $L$-smooth ($|\nabla f(y) - \nabla f(x)| \le L|y - x|$)
       with $L = 25$.
    1. Run heavy ball with the tuning :eqref:`eq_mdl-opt-hb-tuning` for $300$
       steps from $x_0 = 3.3$ with zero initial velocity, and run Nesterov's
       method :eqref:`eq_mdl-opt-nesterov` with $\eta = 1/L$ and
       $\beta = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ from the same start,
       where $\kappa = L/\mu$. Plot $x_k$ against $k$ for both methods.
    1. The heavy-ball iterates approach a period-three cycle that visits the
       piece $x < 1$ twice and the piece $x \ge 2$ once. Express the
       heavy-ball update across these pieces as a linear system for the three
       cycle points, solve it, and compare your solution with the simulation.
    1. Repeat the heavy-ball run from a grid of starting points on $[-6, 6]$
       and report the fraction that land on the cycle. Every piece of $f$ is a
       quadratic with curvature in $[\mu, L]$; why does the per-mode analysis
       behind :eqref:`eq_mdl-opt-hb-rate` fail to apply to $f$?
   :exlabel:`ex_mdl-gradient-based-optimization-heavy-ball-limit-cycle`

1. [code] **A lower bound for first-order methods.** ● This problem derives
   the $O(1/k^2)$ part of the bound of :citet:`Nemirovski.Yudin.1983` quoted
   in :numref:`subsec_mdl-momentum-acceleration`—the smooth convex case—following
   the construction of :citet:`Nesterov.2018`. Here, a first-order method is
   any algorithm whose iterates satisfy
   $\mathbf{x}_k \in \mathbf{x}_0 + \mathrm{span}\{\nabla f(\mathbf{x}_0), \ldots, \nabla f(\mathbf{x}_{k-1})\}$;
   take $\mathbf{x}_0 = \mathbf{0}$. For $p \le d$, let
   $\mathbf{A}_p \in \mathbb{R}^{d \times d}$ be zero except for its leading
   $p \times p$ block, which is tridiagonal with $2$ on the diagonal and $-1$
   on both off-diagonals, and define

   $$
   f_p(\mathbf{x}) = \frac{L}{4} \left( \tfrac12\, \mathbf{x}^\top \mathbf{A}_p\, \mathbf{x} - x_1 \right).
   $$

    1. Show that $f_p$ is convex and $L$-smooth.
    1. Show that if $\mathbf{x}$ is supported on the first $j$ coordinates,
       then $\nabla f_p(\mathbf{x})$ is supported on the first $j + 1$.
       Conclude that the $k$-th iterate of every first-order method is
       supported on the first $k$ coordinates.
    1. Show that the minimizer of $f_p$ has entries
       $x_i^\star = 1 - i/(p+1)$ for $i \le p$ and zero beyond, that the
       minimum value is $-\frac{L}{8}\big(1 - \frac{1}{p+1}\big)$, and that
       $\|\mathbf{x}^\star\|^2 \le (p+1)/3$.
    1. Let $2k + 1 \le d$. The functions $f_{2k+1}$ and $f_k$ agree on all
       vectors supported on the first $k$ coordinates. Use this fact to show
       that every first-order method applied to $f_{2k+1}$ satisfies, with
       $\mathbf{x}^\star$ the minimizer of $f_{2k+1}$,

       $$
       f_{2k+1}(\mathbf{x}_k) - \min f_{2k+1} \;\ge\; \frac{3L\, \|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{32\,(k+1)^2}.
       $$

       Because the construction requires $d \ge 2k + 1$, the bound governs
       roughly the first $d/2$ iterations. In fixed dimension, long runs can
       eventually do better asymptotically, but for neural networks with $d$ in
       the millions this restriction never bites.
    1. For $k \in \{5, 10, 25, 50, 100, 200, 400\}$, run $k$ steps of gradient
       descent with $\eta = 1/L$ and of Nesterov's update
       :eqref:`eq_mdl-opt-nesterov` with $\eta = 1/L$ and
       $\beta_j = (j-1)/(j+2)$ at step $j = 1, 2, \ldots$ on $f_{2k+1}$ with
       $d = 2k + 1$, and divide each suboptimality gap by the lower bound.
       Which ratio stays bounded as $k$ grows?
   :exlabel:`ex_mdl-gradient-based-optimization-first-order-lower-bound`

1. [code] **Edge of stability with momentum.** ●●
   :citet:`Cohen.Kaur.Li.ea.2021` report that when momentum is added, the
   sharpness settles at a threshold different from $2/\eta$.
    1. Use the iteration matrix from problem
       :exref:`ex_mdl-gradient-based-optimization-write-heavy-ball-eqref`a to
       show that heavy ball :eqref:`eq_mdl-opt-heavy-ball` on
       $f(x) = \tfrac{\lambda}{2} x^2$ with $0 \le \beta < 1$ converges if and
       only if $0 < \eta\lambda < 2 + 2\beta$. Derive the corresponding
       stability interval for Nesterov's update :eqref:`eq_mdl-opt-nesterov`.
    1. Modify the edge-of-stability experiment of
       :numref:`subsec_gd-edge-of-stability` (cell
       `#optimization-intro-the-edge-of-stability-1`) to train with heavy ball
       and with Nesterov's update for $\beta \in \{0.25, 0.5\}$, choosing for
       each run a step size whose predicted threshold lies between $5$ and
       $12$. Track the sharpness against both $2/\eta$ and your threshold from
       (a).
    1. At which threshold does the sharpness settle? What does this imply for
       choosing $\eta$ from a sharpness measured at initialization when
       training with momentum?
   :exlabel:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum`

[Discussions](https://d2l.discourse.group/t/36828)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.1]{.kicker}

Gradient Descent and Curvature<br>**steepest descent · L-smoothness · descent lemma · condition number · heavy-ball & Nesterov momentum**
:::
:::

::: {.slide title="One Downhill Direction, Three Step Lengths"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
On $f(x, y) = \tfrac12(x^2 + 10y^2)$ from $\mathbf{x}_0 = (2, 1)$ ($f = 7, \nabla f = (2, 10)$), $-\nabla f$ fixes the direction while $\eta$ sets the step length:

- **Too timid ($\eta = 0.01$):** Moves to $(1.98, 0.90)$ with $f = 6.01$—safe but painfully slow.
- **Just right ($\eta = 0.10 = 1/L$):** Eliminates the steep $y$-mode in one step ($f = 1.62$), while the flat $x$-mode creeps at $0.9\times$ per step.
- **Too aggressive ($\eta = 0.25 > 2/L$):** Overshoots the steep wall to $(1.50, -1.50)$ ($f = 12.38$) and diverges.
:::

::: {.col .fig .big}
@fig:ch36-opt-three-step-regimes
:::
:::
:::

::: {.slide title="First-Order Taylor Model and Descent Directions"}
[Steepest Descent]{.kicker}

::: {.cols .vc}
::: {.col}
Expanding $f$ around $\mathbf{x}$ along a unit vector $\|\mathbf{d}\| = 1$ for small $\eta > 0$:

$$f(\mathbf{x} + \eta\,\mathbf{d}) = f(\mathbf{x}) + \eta\,\nabla f(\mathbf{x})^\top \mathbf{d} + o(\eta)$$

- **Descent half-space:** Any direction with $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$ lowers $f$ for sufficiently small $\eta > 0$.
- **Cauchy–Schwarz optimality:** $\nabla f(\mathbf{x})^\top \mathbf{d} \ge -\|\nabla f(\mathbf{x})\|$, attained uniquely by $\mathbf{d}^\star = -\nabla f(\mathbf{x}) / \|\nabla f(\mathbf{x})\|$.
:::

::: {.col .fig}
@fig:ch36-opt-steepest-descent-cone
:::
:::
:::

::: {.slide title="Comparing Directional Slopes Over Finite Steps"}
[Steepest Descent]{.kicker}

On $f(x, y) = \tfrac12(x^2 + 10y^2)$ at $(2, 1)$, the normalized negative gradient is $(-0.196, -0.981)$ with directional slope $-\|\nabla f\| = -10.198$. Over a short step $\eta = 0.01$, actual loss changes track $\eta \nabla f^\top \mathbf{d}$ within $1\%$:

@!gradient-based-optimization-steepest-direction

::: {.d2l-note}
Differentiability alone guarantees descent only for *infinitesimally* small steps. To certify a finite step size $\eta > 0$, we must bound how rapidly the gradient itself can change along the step.
:::
:::

::: {.slide title="L-Smoothness and the Global Quadratic Ceiling"}
[Smoothness]{.kicker}

::: {.cols .vc}
::: {.col}
A differentiable function $f$ is **$L$-smooth** if its gradient is $L$-Lipschitz ($\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\|\mathbf{x} - \mathbf{y}\|$):

$$f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \tfrac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

- **Hessian test:** For $C^2$ functions, $L$-smoothness holds iff all eigenvalues of $\nabla^2 f(\mathbf{x})$ lie in $[-L, L]$.
- **Geometric guarantee:** The graph of $f$ never rises above its tangent plane by more than the isotropic parabola $\tfrac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$.
:::

::: {.col .fig}
@fig:ch36-opt-lipschitz-quadratic-ceiling
:::
:::
:::

::: {.slide title="The Descent Lemma and the 2/L Step Ceiling"}
[Smoothness]{.kicker}

Substituting the gradient descent update $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{g}_k$ (where $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$) into the quadratic ceiling yields the **descent lemma**:

$$f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \eta\,\|\mathbf{g}_k\|^2 + \tfrac{L}{2}\eta^2\|\mathbf{g}_k\|^2 = f(\mathbf{x}_k) - \eta\left(1 - \tfrac{L\eta}{2}\right)\|\mathbf{g}_k\|^2$$

- **Linear gain vs. curvature penalty:** The first-order drop $-\eta\|\mathbf{g}_k\|^2$ dominates for small $\eta$, while the worst-case curvature penalty $+\tfrac{L}{2}\eta^2\|\mathbf{g}_k\|^2$ grows quadratically in $\eta$.
- **Strict descent window:** The factor $\eta(1 - L\eta/2)$ is strictly positive if and only if $0 < \eta < 2/L$. At $\eta = 2/L$, the linear gain and curvature penalty cancel completely.
- **Optimal worst-case step:** Minimizing the quadratic ceiling selects $\eta = 1/L$, certifying a per-step decrease of at least $f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \tfrac{1}{2L}\|\mathbf{g}_k\|^2$.
:::

::: {.slide title="Visualizing the Quadratic Ceiling Along a Step"}
[Smoothness]{.kicker}

::: {.cols .vc}
::: {.col}
On $f(x,y) = \tfrac12(x^2 + 10y^2)$ from $(2, 1)$, we have $L = 10$ and $\|\mathbf{g}_0\|^2 = 104$:

@!mdl-gradient-based-optimization-smoothness-and-the-descent-lemma

- At $\eta = 1/L = 0.1$, the ceiling guarantees $f \le 1.8$ and the step achieves $1.62$.
- Beyond $2/L = 0.2$, the ceiling exceeds $f(\mathbf{x}_0) = 7$ and at $\eta = 0.25$ the actual loss climbs to $12.375$.
:::

::: {.col .fig}
@fig:mdl-opt-descent-lemma-slice
:::
:::
:::

::: {.slide title="Finding Stationary Points Without Convexity"}
[Nonconvex Guarantees]{.kicker}

Even on nonconvex landscapes, the descent lemma alone prevents the gradient norm from staying large indefinitely when $f$ is bounded below by $f_{\mathrm{low}}$:

- **Telescoping the per-step drops:** At $\eta = 1/L$, rearrange the descent lemma as $\|\mathbf{g}_k\|^2 \le 2L\big(f(\mathbf{x}_k) - f(\mathbf{x}_{k+1})\big)$ and sum over $k = 0, \ldots, K-1$:
  $$\sum_{k=0}^{K-1} \|\nabla f(\mathbf{x}_k)\|^2 \le 2L\big(f(\mathbf{x}_0) - f(\mathbf{x}_K)\big) \le 2L\big(f(\mathbf{x}_0) - f_{\mathrm{low}}\big)$$
- **Sublinear stationarity rate:** Because the minimum over $K$ steps is at most the average,
  $$\min_{0 \le k < K} \|\nabla f(\mathbf{x}_k)\|^2 \le \frac{2L\big(f(\mathbf{x}_0) - f_{\mathrm{low}}\big)}{K}$$

::: {.d2l-note}
Among the first $K$ iterates, at least one has gradient norm $\|\nabla f(\mathbf{x}_k)\| = O(1/\sqrt{K})$, requiring only $L$-smoothness and a lower bound on the objective.
:::
:::

::: {.slide title="Why Stationarity Does Not Guarantee a Minimum"}
[Nonconvex Pitfalls]{.kicker}

::: {.cols .vc}
::: {.col}
Without convexity, $\nabla f(\mathbf{x}) = \mathbf{0}$ cannot distinguish a minimum from a saddle point:

- **Saddle trap:** $f(x, y) = \tfrac12 x^2 + \cos y$ is bounded below by $-1$ and is globally $1$-smooth ($\nabla^2 f = \mathrm{diag}(1, -\cos y)$).
- **One-step halt:** From $(1, 0)$ with $\eta = 1/L = 1$, gradient descent steps directly to $(0, 0)$ where $\nabla f(0, 0) = \mathbf{0}$.
- **Suboptimal plateau:** GD freezes permanently at the saddle $f(0, 0) = +1$, two units above the global minimum $f(0, \pi) = -1$.
:::

::: {.col .fig .big}
@fig:ch36-opt-saddle-halt
:::
:::
:::

::: {.slide title="Backtracking Line Search and the Armijo Rule"}
[Adaptive Step Sizes]{.kicker}

::: {.cols .vc}
::: {.col}
When $L$ is unknown or varies locally (e.g., $f(x) = \tfrac14 x^4$), **backtracking line search** halves a trial step $\eta \leftarrow \eta/2$ until the Armijo condition holds ($c \in (0, 1)$):

$$f\big(\mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)\big) \le f(\mathbf{x}_k) - c\,\eta\,\|\nabla f(\mathbf{x}_k)\|^2$$

- **Sufficient decrease:** Accepts $\eta$ as soon as actual loss reduction reaches fraction $c$ of the linear Taylor drop.
- **Automatic termination:** Guaranteed to pass once $\eta \le 2(1-c)/L_{\mathrm{loc}}$, returning $\eta \ge \min(\eta_0, (1-c)/L_{\mathrm{loc}})$ without knowing $L_{\mathrm{loc}}$.
:::

::: {.col .fig}
@fig:ch36-opt-armijo-backtracking
:::
:::
:::

::: {.slide title="Backtracking vs. Fixed Step on a Quartic Bowl"}
[Adaptive Step Sizes]{.kicker}

On $f(x) = \tfrac14 x^4$ from $x_0 = 3$ with $c = \tfrac12$, the update $x \mapsto x(1 - \eta x^2)$ contracts $|x|$ if and only if $\eta x_0^2 < 2$ ($\eta < 2/9 \approx 0.22$), whereas local curvature at $x_0 = 3$ is $f''(3) = 27$:

@!gradient-based-optimization-backtracking

::: {.d2l-note .warn}
A fixed step $\eta = 0.3$ (safe near the origin) explodes to $6.5 \times 10^{103}$ in $6$ steps from $x_0 = 3$. Backtracking starts at $\eta = 0.031 < 1/27$ and automatically doubles $\eta$ as the iterate enters flatter regions near zero.
:::
:::

::: {.slide title="The Quadratic Model and the Condition Number"}
[Quadratic Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
Near any strict local minimizer $\mathbf{x}^\star$ with positive-definite Hessian $\mathbf{A} = \nabla^2 f(\mathbf{x}^\star) \succ \mathbf{0}$, shifting $\mathbf{x}^\star$ to the origin yields the canonical quadratic model:

$$f(\mathbf{x}) = \tfrac12\,\mathbf{x}^\top \mathbf{A}\,\mathbf{x}$$

- **Spectrum:** $0 < \lambda_{\min} = \lambda_1 \le \cdots \le \lambda_d = \lambda_{\max} = L$.
- **Condition number:** $\kappa = \lambda_{\max} / \lambda_{\min} \ge 1$ measures the ratio of steepest to flattest curvature ($\kappa = 10$ on our gentle bowl, $\kappa = 1000$ on the chapter benchmark).
:::

::: {.col .fig}
@fig:mdl-opt-gd-bowl-vs-valley
:::
:::
:::

::: {.slide title="Spectral Decoupling into Scalar Recursions"}
[Quadratic Geometry]{.kicker}

Because $\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x}$, gradient descent applies the linear operator $\mathbf{x}_{k+1} = (\mathbf{I} - \eta\mathbf{A})\mathbf{x}_k$. Expanding $\mathbf{x}_k$ in the orthonormal eigenbasis of $\mathbf{A}$ with coordinates $u_i^{(k)}$ decouples the $d$ variables completely:

$$u_i^{(k+1)} = (1 - \eta\lambda_i)\,u_i^{(k)} \implies u_i^{(k)} = (1 - \eta\lambda_i)^k\,u_i^{(0)}$$

- **Exact stability ceiling:** Every mode contracts if and only if $|1 - \eta\lambda_i| < 1$ for all $i$, which requires $0 < \eta < 2/\lambda_{\max} = 2/L$. Thus the descent lemma's sufficient threshold $2/L$ is also *strictly necessary* on quadratics.
- **Spectral radius governs speed:** The worst-case error norm decays as $\rho(\eta)^k$, where $\rho(\eta) = \max_i |1 - \eta\lambda_i| = \max\big(|1 - \eta\lambda_{\min}|,\, |1 - \eta\lambda_{\max}|\big)$ is determined by the two extreme eigenvalues.
:::

::: {.slide title="Sweeping Step Sizes Across the Stability Limit"}
[Quadratic Geometry]{.kicker}

On $\mathbf{A} = \mathrm{diag}(1, 10)$ from $\mathbf{x}_0 = (1, 1)$, the stability ceiling is $2/\lambda_{\max} = 0.20$:

@!gradient-based-optimization-eta-sweep

- At $\eta = 0.10 = 1/L$, the stiff mode vanishes in $1$ step ($1 - 10\eta = 0$), but the slow mode crawls at $1 - \eta = 0.90$.
- At $\eta = 2/11 \approx 0.182$, both modes contract by $9/11 \approx 0.818$ ($+9/11$ and $-9/11$), cutting the $60$-step error by two orders of magnitude.
:::

::: {.slide title="The Optimal Fixed Step Balances Extreme Modes"}
[Quadratic Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
Because $\lambda \mapsto |1 - \eta\lambda|$ is V-shaped on $[\lambda_{\min}, \lambda_{\max}]$, $\rho(\eta)$ is minimized by equalizing the contraction magnitudes at the two endpoints:

$$1 - \eta^\star\lambda_{\min} = \eta^\star\lambda_{\max} - 1 \implies \eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}$$

- **Optimal contraction rate:**
  $$\rho(\eta^\star) = \frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$$
- **Linear iteration complexity:** Reaching $\|\mathbf{x}_k\| \le \epsilon\|\mathbf{x}_0\|$ takes $k \approx \frac{\kappa}{2}\log\frac{1}{\epsilon}$ steps ($69$ at $\kappa=10$, $6{,}908$ at $\kappa=1000$).
:::

::: {.col .fig}
@fig:mdl-opt-eta-tent
:::
:::
:::

::: {.slide title="Neural Network Training at the Edge of Stability"}
[Deep Learning Dynamics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Progressive sharpening:** During full-batch GD on deep networks, top Hessian sharpness $\lambda_{\max}(\nabla^2 f(\mathbf{x}_k))$ rises until it hits the stability limit $2/\eta$ and hovers there.
- **Why loss spikes occur:** Once $\lambda_{\max} > 2/\eta$, the top mode multiplier $1 - \eta\lambda_{\max} < -1$ oscillates with growing amplitude.
- **Self-stabilization beyond quadratics:** Third-order Taylor terms couple the oscillation to changing curvature, steering the iterate toward flatter regions where $\lambda_{\max} \approx 2/\eta$.
:::

::: {.col .fig .big}
@fig:ch36-opt-edge-of-stability
:::
:::
:::

::: {.slide title="How Convexity Upgrades Convergence Guarantees"}
[Rate Hierarchy]{.kicker}

Adding structural assumptions on $f$ upgrades gradient descent ($\eta = 1/L$) from local stationarity to global optimality and geometric convergence:

| Objective Class | Mathematical Assumption | Guaranteed Rate at $\eta = 1/L$ | Steps for Accuracy $\epsilon$ |
| :--- | :--- | :--- | :--- |
| **Smooth Nonconvex** | $\nabla f$ is $L$-Lipschitz, $f \ge f_{\mathrm{low}}$ | $\min_{k < K}\|\nabla f(\mathbf{x}_k)\|^2 \le \frac{2L(f_0 - f_{\mathrm{low}})}{K}$ | $O(1/\epsilon^2)$ for $\|\nabla f\| \le \epsilon$ |
| **Smooth Convex** | $L$-smooth and $\nabla^2 f(\mathbf{x}) \succeq \mathbf{0}$ | $f(\mathbf{x}_k) - f^\star \le \frac{L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}$ | $O(1/\epsilon)$ for $f - f^\star \le \epsilon$ |
| **Strongly Convex** | $\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$ ($\kappa = L/\mu$) | $f(\mathbf{x}_k) - f^\star \le (1 - 1/\kappa)^k(f_0 - f^\star)$ | $O(\kappa \log(1/\epsilon))$ |
| **Quadratic at $\eta^\star$** | $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, $\eta^\star = \frac{2}{\mu + L}$ | $f(\mathbf{x}_k) - f^\star \le \bigl(\frac{\kappa - 1}{\kappa + 1}\bigr)^{2k}(f_0 - f^\star)$ | $\approx \frac{\kappa}{4}\log(1/\epsilon)$ for $f - f^\star$ |
:::

::: {.slide title="Heavy-Ball Momentum: Velocity with Memory"}
[Momentum]{.kicker}

::: {.cols .vc}
::: {.col}
Gradient descent takes $O(\kappa)$ steps because a memoryless step cannot damp steep oscillations without starving flat directions. Polyak's **heavy-ball momentum** maintains an exponentially weighted velocity $\mathbf{v}_k$ and updates $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}$:

$$\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla f(\mathbf{x}_k)$$

- **Two-step recurrence:** Eliminating $\mathbf{v}_k = \mathbf{x}_k - \mathbf{x}_{k-1}$ gives $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\nabla f(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})$.
- **Dual action:** Opposite gradients along steep walls cancel; aligned gradients along the flat floor accumulate speed.
:::

::: {.col .fig}
@fig:mdl-opt-momentum-damping
:::
:::
:::

::: {.slide title="Characteristic Polynomial of a Momentum Mode"}
[Momentum Analysis]{.kicker}

In the eigenbasis of a quadratic $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, each mode $u_k$ with eigenvalue $\lambda$ obeys the second-order linear recurrence $u_{k+1} = (1 + \beta - \eta\lambda)u_k - \beta u_{k-1}$. Substituting $u_k = z^k$ yields the characteristic equation:

$$z^2 - (1 + \beta - \eta\lambda)\,z + \beta = 0$$

- **Complex-conjugate root regime:** The discriminant $(1 + \beta - \eta\lambda)^2 - 4\beta$ is nonpositive whenever:
  $$(1 - \sqrt{\beta})^2 \le \eta\lambda \le (1 + \sqrt{\beta})^2$$
- **Equalized root modulus:** When the roots $z, \bar{z}$ are complex conjugates, their product is the constant term $z\bar{z} = |z|^2 = \beta$, so **both roots have modulus $|z| = \sqrt{\beta}$ regardless of $\lambda$**.
- **Endpoint double roots:** At the two boundaries $\eta\lambda = (1 \pm \sqrt{\beta})^2$, a repeated real root $z = \pm\sqrt{\beta}$ introduces a linear prefactor $|u_k| \le C\,k\,\beta^{k/2}$.
:::

::: {.slide title="Optimal Heavy-Ball Tuning and the √κ Law"}
[Momentum Analysis]{.kicker}

::: {.cols .vc}
::: {.col}
Placing the extreme eigenvalues $\lambda_{\min}$ and $\lambda_{\max}$ right at the endpoints of the complex-root interval, $\eta\lambda_{\min} = (1 - \sqrt{\beta})^2$ and $\eta\lambda_{\max} = (1 + \sqrt{\beta})^2$, forces every mode onto the circle of radius $\sqrt{\beta^\star}$ with step size $\eta_{\mathrm{HB}} = 4 / (\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2$:

$$\beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{\!2}$$

- **Accelerated contraction rate:** Every mode contracts at rate
  $$\rho_{\mathrm{HB}} = \sqrt{\beta^\star} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$$
- **Speedup:** Iteration complexity drops from $O(\kappa\log\tfrac{1}{\epsilon})$ to $O(\sqrt{\kappa}\log\tfrac{1}{\epsilon})$.
:::

::: {.col .fig}
@fig:mdl-opt-hb-eigenvalues
:::
:::
:::

::: {.slide title="Nesterov Acceleration and Global Guarantees"}
[Accelerated Methods]{.kicker}

::: {.cols .vc}
::: {.col}
**Nesterov's accelerated gradient** evaluates $\nabla f$ at the look-ahead point $\tilde{\mathbf{x}}_k = \mathbf{x}_k + \beta\mathbf{v}_k$:

$$\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla f(\mathbf{x}_k + \beta\,\mathbf{v}_k)$$

- **Look-ahead braking:** If momentum carries the iterate uphill, $\nabla f(\tilde{\mathbf{x}}_k)$ points backward and brakes *before* overshooting.
- **Minimax optimal rates:** Achieves $O(\sqrt{\kappa}\log(1/\epsilon))$ globally on all $L$-smooth, $\mu$-strongly convex objectives and $O(1/k^2)$ on smooth convex objectives (matching the Nemirovski–Yudin lower bound).
:::

::: {.col .fig}
@fig:ch36-opt-nesterov-lookahead
:::
:::
:::

::: {.slide title="Benchmarking Iteration Scaling Across κ"}
[Empirical Verification]{.kicker}

Measuring the exact number of iterations required to reach $\|\mathbf{x}_k\| \le 10^{-6}\|\mathbf{x}_0\|$ on $\mathrm{diag}(1, \kappa)$ from $(1, 1)$ confirms the linear $\kappa$ vs. $\sqrt{\kappa}$ scaling laws:

@!gradient-based-optimization-momentum

- **Gradient descent** stays pinned at $k/\kappa \approx 6.9$ ($6{,}908$ steps at $\kappa = 1000$).
- **Heavy ball** finishes in $315$ steps at $\kappa = 1000$; its ratio $k/\sqrt{\kappa}$ drifts gently ($8.5$ $\to$ $10.0$) due to the Jordan-block double-root prefactor $k\rho_{\mathrm{HB}}^k$, matching $k \approx \frac{\sqrt{\kappa}}{2}(\log 10^6 + \log k)$.
:::

::: {.slide title="Concept Check: Step Sizes and Momentum Rates"}
[Concept Check]{.kicker}

Consider the quadratic objective $f(x_1, x_2) = \tfrac12(x_1^2 + 100\,x_2^2)$ with eigenvalues $\lambda_{\min} = 1$, $\lambda_{\max} = 100$ ($\kappa = 100$):

1. What is the maximum stable learning rate for gradient descent, the optimal fixed step $\eta^\star$, and the per-step contraction factor $\rho(\eta^\star)$?
2. Roughly how many gradient descent steps are needed to shrink $\|\mathbf{x}_k\|$ by $10^{-6}$?
3. What contraction factor $\rho_{\mathrm{HB}}$ does tuned heavy-ball momentum achieve, and roughly how many steps does it need (including the double-root logarithmic term)?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** Stability ceiling: $2/L = 0.02$. Optimal step: $\eta^\star = 2/(1 + 100) = 2/101 \approx 0.0198$. Contraction factor: $\rho(\eta^\star) = 99/101 \approx 0.9802$. **(2)** $\log(10^6)/\log(101/99) \approx \frac{100}{2}\ln(10^6) \approx 691$ steps. **(3)** With $\sqrt{\kappa} = 10$, $\rho_{\mathrm{HB}} = (10 - 1)/(10 + 1) = 9/11 \approx 0.818$. Pure geometric decay predicts $69$ steps, and the double-root factor $k\rho_{\mathrm{HB}}^k$ adds $\frac{\sqrt{\kappa}}{2}\ln k \approx 24$ steps, matching the measured $93$ steps.
:::
:::

::: {.slide title="Summary: First-Order Methods and Curvature"}
[Key Takeaways]{.kicker}

| Method / Concept | Step Rule / Formula | Contraction / Guarantee | Core Takeaway |
| :--- | :--- | :--- | :--- |
| **Descent Lemma** | $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\mathbf{g}_k$, $0 < \eta < 2/L$ | $\Delta f \le -\eta(1 - L\eta/2)\|\mathbf{g}_k\|^2$ | Steepest curvature $L$ caps the safe step at $2/L$ |
| **Backtracking (Armijo)** | Halve $\eta$ until $f(\mathbf{x} - \eta\mathbf{g}) \le f(\mathbf{x}) - c\eta\|\mathbf{g}\|^2$ | Finds $\eta \ge \min(\eta_0, (1-c)/L_{\mathrm{loc}})$ | Adapts to local curvature without knowing $L$ |
| **Optimal Fixed-Step GD** | $\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}$ | $\rho_{\mathrm{GD}} = \frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$ | Iteration count scales linearly with $\kappa = L/\mu$ |
| **Heavy-Ball Momentum** | $\mathbf{v}_{k+1} = \beta^\star\mathbf{v}_k - \eta_{\mathrm{HB}}\mathbf{g}_k$ | $\rho_{\mathrm{HB}} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$ | Complex roots equalize all modes at modulus $\sqrt{\beta^\star}$ |
| **Nesterov Look-Ahead** | $\mathbf{v}_{k+1} = \beta\mathbf{v}_k - \frac{1}{L}\nabla f(\mathbf{x}_k + \beta\mathbf{v}_k)$ | $\rho_{\mathrm{NAG}} = 1 - \frac{1}{\sqrt{\kappa}}$ ($O(1/k^2)$ if $\mu=0$) | Achieves optimal $O(\sqrt{\kappa})$ globally beyond quadratics |
:::

