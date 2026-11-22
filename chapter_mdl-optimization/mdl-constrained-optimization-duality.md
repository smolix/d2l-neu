# Constrained Optimization
:label:`sec_mdl-constrained-optimization-duality`

Up to this point, every parameter we optimized was free to take any real value,
so finding a minimum meant hunting for a point where the gradient vanishes.
Many quantities in deep learning do not have that freedom. Attention weights
must be nonnegative and sum to one, gradient clipping caps the norm of an
update at a threshold, a norm budget restricts the size of a weight vector, and
a support vector machine requires every training margin to be at least one.
When the variables must stay inside a feasible set, the gradient rarely
vanishes at the optimum, and the zero-gradient test of the earlier sections
breaks down. How do we recognize an optimal point under constraints, and how do
we compute it?

Gradient clipping offers the simplest concrete example. Suppose we want to clip
the vector $\mathbf{y} = (3, 4)$ so that its norm is at most $1$. Rescaling it
to unit length returns $(0.6, 0.8)$, which is the point of the unit disk
closest to $\mathbf{y}$. In other words, $(0.6, 0.8)$ minimizes the squared
distance $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ over the disk
$\|\mathbf{x}\| \le 1$, reaching the optimal value $f = 8$. Yet the gradient
there is $(-2.4, -3.2)$, nowhere near zero. What makes $(0.6, 0.8)$ optimal is
that no *allowed* direction lowers $f$: every move that keeps
$\|\mathbf{x}\| \le 1$ either increases $f$ or leaves it unchanged to first
order. To handle problems like this, we need both a replacement for the
zero-gradient condition and a replacement for plain gradient descent, which on
its own would step right out of the feasible set.

The replacement for "the gradient is zero" introduces one scalar per
constraint, called its *multiplier*, which measures how hard the optimum
presses against that constraint. At $(0.6, 0.8)$ the boundary pushes back with
strength $4$ along the outward normal, so that
$(-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$. Multipliers give us the
first-order optimality conditions: Lagrange's condition for equality
constraints, and the Karush--Kuhn--Tucker (KKT) conditions for inequalities.
Meanwhile, the projection that mapped $(3, 4)$ to $(0.6, 0.8)$ suggests a
direct algorithm: take an ordinary gradient step, then project back to the
nearest feasible point. And when a hard boundary is replaced by a kinked
penalty, such as the absolute-value penalty that the lasso places on each
weight, the projection generalizes to a *proximal step*, which drives small
weights to exact zeros. Throughout the section, convexity
(:numref:`sec_mdl-convexity`) is what turns these conditions into guarantees of
global optimality and ensures that our algorithms converge.

Three ideas carry the section: the KKT conditions and their sufficiency for
convex problems, the obtuse-angle inequality that characterizes a projection,
and soft-thresholding.

```{.python .input #constrained-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #constrained-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #constrained-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #constrained-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Equality Constraints and Lagrange Multipliers
:label:`subsec_mdl-lagrange-multipliers`

### Tangency and the Lagrange Condition

Equality constraints are simpler than inequalities, so let us start by
replacing the solid disk of the clipping example with its boundary: minimize
$f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ with
$\mathbf{y} = (3, 4)$ over the unit circle, written as
$h(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) = 0$. More generally, we want
to minimize a differentiable objective $f : \mathbb{R}^n \to \mathbb{R}$
subject to a single equality constraint,

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad h(\mathbf{x}) = 0.
$$

Because the feasible set is a surface (a curve in two dimensions) rather than a
full-dimensional region, every feasible move must stay on that surface.

:numref:`fig_mdl-opt-lagrange-tangency` illustrates the geometric signature of
an optimum. Suppose we stand at a feasible point where a level set of $f$
*crosses* the circle. Then $\nabla f$ has a nonzero component along the circle,
so sliding along the circle opposite that component reduces $f$ without leaving
the feasible set. Such a point cannot be optimal. An optimum can occur only
where the level set is *tangent* to the circle, so that every move along the
circle leaves $f$ unchanged to first order. Since tangent curves share parallel
normal vectors, $\nabla f$ must be a scalar multiple of $\nabla h$ at the
optimum.

![Minimizing $f=\tfrac12\|\mathbf{x}-\mathbf{y}\|^2$ with $\mathbf{y}=(3,4)$ on the unit circle $h(\mathbf{x})=0$. Where a level set crosses the circle, $\nabla f$ has a component along the constraint, giving a feasible descent direction. At $\mathbf{x}^\star=(0.6,0.8)$ the level set is tangent to the circle and $\nabla f=-4\,\nabla h$. The antipode $(-0.6,-0.8)$ also satisfies the tangency condition, with $\nu=-6$, and is the maximum.](../img/mdl-opt-lagrange-tangency.svg)
:label:`fig_mdl-opt-lagrange-tangency`

**Proposition (Lagrange condition).** *Let $\mathbf{x}^\star$ be a local
minimum of $f$ on $\{\mathbf{x} : h(\mathbf{x}) = 0\}$. Assume that $f$ is
differentiable at $\mathbf{x}^\star$, that $h$ is continuously differentiable
near $\mathbf{x}^\star$, and that $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$.
Then there exists a unique scalar $\nu^\star$, the **Lagrange multiplier**,
such that*

$$
\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla h(\mathbf{x}^\star) = \mathbf{0}.
$$
:eqlabel:`eq_mdl-opt-lagrange-condition`

In words, the gradient of the objective is balanced by a multiple of the
constraint's normal vector. This is the same Lagrange condition proved in
:numref:`subsec_mdl-lagrange`, where the constraint is written as
$g(\mathbf{x}) = c$ and the condition as $\nabla f = \mu\,\nabla g$. Setting
$h = g - c$ and $\nu = -\mu$ reconciles the two forms; we use the plus sign
here because it matches the sign of the Lagrangian introduced next.

Checking the tangent point $(0.6, 0.8)$ in :numref:`fig_mdl-opt-lagrange-tangency`,
we have $\nabla f = \mathbf{x} - \mathbf{y} = (-2.4, -3.2)$ and
$\nabla h = \mathbf{x} = (0.6, 0.8)$. The two vectors are indeed parallel, with
$\nabla f = -4\,\nabla h$, which gives $\nu^\star = 4$.

The assumption $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ is our first
example of a **constraint qualification**, and it is essential rather than a
technicality. Consider minimizing $f(\mathbf{x}) = x_1$ subject to
$h(\mathbf{x}) = x_1^2 + x_2^2 = 0$. The feasible set consists of the single
point $\mathbf{0}$, which is automatically the minimum. Yet
$\nabla f(\mathbf{0}) = (1, 0)$ while $\nabla h(\mathbf{0}) = \mathbf{0}$, so
no multiplier $\nu^\star$ can satisfy :eqref:`eq_mdl-opt-lagrange-condition`.
Because the feasible set has collapsed to a single point with no tangent
directions, the tangency argument has nothing to grab onto.

### The Lagrangian

Notice that we have two unknowns, $\mathbf{x}$ and $\nu$, and two requirements:
parallel gradients and feasibility. Both requirements can be packaged into the
stationarity of a single function of $(\mathbf{x}, \nu)$, called the
**Lagrangian**, which adds the constraint weighted by its multiplier to the
objective:

$$
\mathcal{L}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu\, h(\mathbf{x}).
$$
:eqlabel:`eq_mdl-opt-lagrangian-eq`

Setting all partial derivatives of $\mathcal{L}$ to zero yields

$$
\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \nu\, \nabla h = \mathbf{0}
\qquad\textrm{and}\qquad
\frac{\partial \mathcal{L}}{\partial \nu} = h(\mathbf{x}) = 0,
$$

recovering :eqref:`eq_mdl-opt-lagrange-condition` and the equality constraint
together. In this way, a constrained optimization problem in $\mathbf{x}$
becomes an unconstrained stationarity problem in $(\mathbf{x}, \nu)$.

Let us solve the unit-circle problem using the Lagrangian. Stationarity with
respect to $\mathbf{x}$ gives
$\nabla_{\mathbf{x}}\mathcal{L} = (\mathbf{x} - \mathbf{y}) + \nu\mathbf{x} = \mathbf{0}$,
so $\mathbf{x} = \mathbf{y}/(1 + \nu)$ must lie on the line through the origin
and $\mathbf{y}$. Enforcing feasibility, $\|\mathbf{x}\| = 1$, then requires
$|1 + \nu| = \|\mathbf{y}\| = 5$, which produces two solutions:

* $\nu = 4$: $\mathbf{x} = (0.6, 0.8)$, at distance $4$ from $\mathbf{y}$, with $f = 8$;
* $\nu = -6$: $\mathbf{x} = (-0.6, -0.8)$, at distance $6$ from $\mathbf{y}$, with $f = 18$.

The second solution is the point on the circle *farthest* from $\mathbf{y}$.
This highlights an important caveat: the Lagrange condition is necessary, not
sufficient, because it holds at every stationary point of $f$ along the
surface, including maxima and saddles. Here the sign of the multiplier
distinguishes the minimum from the maximum. When we move to inequality
constraints in the next section, the KKT conditions will turn that sign into a
general rule.

When there are multiple equality constraints $h_j(\mathbf{x}) = 0$ for
$j = 1, \ldots, p$, each constraint receives its own multiplier and the
Lagrangian becomes $\mathcal{L} = f + \sum_j \nu_j h_j$. Provided that the
constraint gradients $\nabla h_j(\mathbf{x}^\star)$ are linearly independent,
the same reasoning shows that $\nabla f(\mathbf{x}^\star)$ lies in the *span*
of the constraint normals—that is, the objective gradient is balanced by a
linear combination of the constraint gradients.

### The Closest Point on a Hyperplane
:label:`subsec_mdl-closest-point`

The simplest constrained problem with a linear constraint asks for the point on
a hyperplane closest to the origin: minimize
$f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|^2$ subject to
$h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$. Setting the gradient of
$\mathcal{L} = \tfrac12\|\mathbf{x}\|^2 + \nu(\mathbf{a}^\top\mathbf{x} - b)$
to zero gives $\mathbf{x} + \nu\mathbf{a} = \mathbf{0}$, so the closest point
must lie along the normal vector $\mathbf{a}$. Substituting
$\mathbf{x} = -\nu\mathbf{a}$ into the constraint yields
$-\nu\|\mathbf{a}\|^2 = b$, which fixes $\nu^\star = -b/\|\mathbf{a}\|^2$. The
solution and optimal value are therefore

$$
\mathbf{x}^\star = \frac{b}{\|\mathbf{a}\|^2}\,\mathbf{a},
\qquad
p^\star = \frac{b^2}{2\,\|\mathbf{a}\|^2}.
$$

Geometrically, the nearest point sits at distance $b/\|\mathbf{a}\|$ along the
unit normal $\mathbf{a}/\|\mathbf{a}\|$, matching the projection formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops`.

For $\mathbf{a} = (3, 4)$ and $b = 5$, we land on $(0.6, 0.8)$ once again, this
time with $\nu^\star = -0.2$ and $p^\star = \tfrac12$: the line
$3x_1 + 4x_2 = 5$ is tangent to the unit circle at $(0.6, 0.8)$. Notice that
the multiplier is more than an algebraic helper. If we nudge $b$ from $5$ to
$5.1$, pushing the line slightly farther from the origin, the optimal value
rises by about $0.02 = -\nu^\star \cdot 0.1$. As
:numref:`subsec_mdl-shadow-prices` will show in full generality, this is what a
multiplier always measures: the sensitivity of the optimal value to shifting
its constraint.

## Inequality Constraints and the KKT Conditions
:label:`subsec_mdl-kkt-conditions`

### Active and Inactive Constraints

Most constraints in machine learning are one-sided: a norm must be at most $r$,
a probability at least $0$, or a resource budget not exceeded. We write
inequality constraints in the standard form

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m,
$$

so that each $g_i$ is negative strictly inside its region and zero along its
boundary. At a solution $\mathbf{x}^\star$, every inequality falls into one
of two regimes. If $g_i(\mathbf{x}^\star) < 0$, the constraint is
**inactive**: the solution lies strictly inside the region, so locally the
constraint has no effect. If $g_i(\mathbf{x}^\star) = 0$, the constraint is
**active**: the solution sits right on the boundary, where the inequality acts
like an equality.

Gradient clipping illustrates both regimes cleanly. Suppose we project a vector
$\mathbf{y}$ onto the Euclidean ball of radius $r$ by minimizing
$f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ subject to
$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - r^2) \le 0$. Introducing a
multiplier $\lambda$, stationarity of $f + \lambda g$ yields
$(\mathbf{x} - \mathbf{y}) + \lambda\mathbf{x} = \mathbf{0}$, so
$\mathbf{x} = \mathbf{y}/(1 + \lambda)$ is always a radial rescaling of
$\mathbf{y}$. Which regime applies depends on $\|\mathbf{y}\|$:

* If $\|\mathbf{y}\| \le r$, then $\mathbf{y}$ is already feasible and
  minimizes $f$ outright. The constraint is inactive, $\lambda^\star = 0$, and
  $\mathbf{x}^\star = \mathbf{y}$; for example, $\mathbf{y} = (0.3, 0.4)$ with
  $r = 1$ stays untouched.
* If $\|\mathbf{y}\| > r$, no interior point can be optimal, because any
  interior minimum would have to be an unconstrained minimum of $f$, and the
  only unconstrained minimum is the infeasible point $\mathbf{y}$. The
  constraint must therefore be active, and $\|\mathbf{x}^\star\| = r$ forces
  $\lambda^\star = \|\mathbf{y}\|/r - 1 > 0$. For $\mathbf{y} = (3, 4)$ and
  $r = 1$, we get $\lambda^\star = 4$, $\mathbf{x}^\star = (0.6, 0.8)$, and
  $p^\star = \tfrac12 \cdot 4^2 = 8$.

Both cases collapse into a single formula,

$$
\mathbf{x}^\star = \frac{\mathbf{y}}{\max\left(1,\; \|\mathbf{y}\|/r\right)},
$$

which leaves short vectors alone and rescales long ones to norm $r$. Applied to
a gradient vector, this is *gradient clipping*
:cite:`Pascanu.Mikolov.Bengio.2013`. In code, the case split becomes a single
`max`, and we can check directly that at the clipped point the gradient of $f$
is cancelled by $\lambda^\star$ times the constraint gradient
$\nabla g(\mathbf{x}) = \mathbf{x}$.

```{.python .input #mdl-constrained-optimization-duality-active-and-inactive-constraints}
y, r = np.array([3.0, 4.0]), 1.0
lam = max(np.linalg.norm(y) / r - 1, 0.0)    # multiplier: 0 if y is inside the ball
x = y / (1 + lam)                            # x* = y / max(1, |y|/r)
print('x* =', x, ' lambda* =', lam, ' f(x*) =', 0.5 * ((x - y)**2).sum())
print('grad f(x*) =', x - y, ' lambda* grad g(x*) =', lam * x,
      ' sum =', (x - y) + lam * x)
```

Stationarity holds even though $\nabla f$ is far from zero: the objective
gradient $(-2.4, -3.2)$ and the boundary's push $(2.4, 3.2)$ sum to zero, with
the multiplier $\lambda^\star = 4$ giving the ratio of their lengths.

Notice the geometric meaning of the multiplier's sign. Because $g(\mathbf{x})$
increases as we leave the ball, its gradient $\nabla g = \mathbf{x}$ points
outward. Stationarity written as $-\nabla f = \lambda\nabla g$ with
$\lambda > 0$ says that the downhill direction $-\nabla f$ pulls outward,
while the boundary wall pushes back. If $-\nabla f$ pointed inward instead, we
could step into the interior of the ball to lower $f$, so the point could not
be optimal. A negative multiplier is therefore ruled out: $\lambda = -6$ would
select the far point $(-0.6, -0.8)$ on the circle, which is a maximum rather
than a minimum over the disk. Unlike an equality constraint, an inequality wall
pushes in only one direction.

### The Karush--Kuhn--Tucker Conditions

Combining inequality and equality constraints gives the general optimization
problem

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad
g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m, \qquad h_j(\mathbf{x}) = 0,\; j = 1, \ldots, p.
$$
:eqlabel:`eq_mdl-opt-standard-problem`

Its **Lagrangian** assigns one multiplier to each constraint, with the
inequality multipliers required to be nonnegative:

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}) + \sum_{i=1}^m \lambda_i\, g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j\, h_j(\mathbf{x}),
\qquad \lambda_i \ge 0.
$$
:eqlabel:`eq_mdl-opt-lagrangian`

At any local minimum $\mathbf{x}^\star$ where a constraint qualification holds
(we specify which ones in the next subsection), there exist multipliers
satisfying the four **Karush--Kuhn--Tucker (KKT) conditions**
:cite:`Karush.1939,Kuhn.Tucker.1951`:

$$
\begin{aligned}
&\textrm{stationarity:}
&& \nabla f(\mathbf{x}^\star) + \textstyle\sum_i \lambda_i^\star \nabla g_i(\mathbf{x}^\star) + \sum_j \nu_j^\star \nabla h_j(\mathbf{x}^\star) = \mathbf{0}, \\
&\textrm{primal feasibility:}
&& g_i(\mathbf{x}^\star) \le 0, \qquad h_j(\mathbf{x}^\star) = 0, \\
&\textrm{dual feasibility:}
&& \lambda_i^\star \ge 0, \\
&\textrm{complementary slackness:}
&& \lambda_i^\star\, g_i(\mathbf{x}^\star) = 0 \quad \textrm{for every } i.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-kkt`

We have already seen the first three conditions in our examples: a balance of
gradients, satisfaction of the constraints themselves, and the one-way sign
restriction on inequality multipliers. The fourth, **complementary slackness**,
states that for each $i$ at least one of $\lambda_i^\star$ and
$g_i(\mathbf{x}^\star)$ must vanish. Consequently, every inactive constraint
($g_i(\mathbf{x}^\star) < 0$) automatically gets $\lambda_i^\star = 0$ and
drops out of the stationarity equation. Keep in mind a subtle asymmetry: a
positive multiplier guarantees that a constraint is active, but an active
constraint can still have $\lambda_i^\star = 0$ if it happens to touch
$\mathbf{x}^\star$ without exerting any force. In the ball-projection example,
complementary slackness encodes the exact case split we carried out by hand,
and at $(0.6, 0.8)$ we have $\lambda^\star g(\mathbf{x}^\star) = 4 \cdot 0 = 0$.

:numref:`fig_mdl-opt-kkt-active-set` shows the geometry of stationarity and
dual feasibility together. At the optimum, the descent direction $-\nabla f$
must be a nonnegative linear combination of the outward normals of the active
constraints. Equivalently, $-\nabla f$ lies inside the cone spanned by those
outward normals—the *normal cone*—while inactive constraints contribute nothing.

![Geometry of the KKT conditions on a feasible region defined by two inequality constraints. In (a), one constraint is active at $\mathbf{x}^\star$ and the other is inactive: $-\nabla f$ aligns with the active outward normal with $\lambda_1 > 0$, while $\lambda_2 = 0$. In (b), both constraints are active at a corner, so $-\nabla f$ must lie in the shaded normal cone spanned by both outward normals. Orange arcs show level sets of $f$ around its unconstrained minimizer (+).](../img/mdl-opt-kkt-active-set.svg)
:label:`fig_mdl-opt-kkt-active-set`

Like the Lagrange condition, the KKT conditions are first-order conditions and
are not sufficient on their own. Consider minimizing $f(x) = -x^2$ on
$[-1, 1]$, written as $x - 1 \le 0$ and $-1 - x \le 0$. At $x = 0$, the
gradient vanishes, both constraints are inactive, and setting both multipliers
to zero satisfies all four KKT conditions—yet $x = 0$ is the *maximum*. The
true minima $x = \pm 1$ are KKT points too, each with multiplier $2$ on its
active boundary constraint.

### Constraint Qualifications and Sufficiency
:label:`subsec_mdl-constraint-qualifications`

This leaves two complementary questions. First, when is a local minimum
*guaranteed* to satisfy the KKT conditions? Second, when is a KKT point
*guaranteed* to be a minimum?

For the first question, the answer parallels the equality case: a minimum must
satisfy the KKT conditions whenever a constraint qualification holds. The most
common general condition is the *linear independence constraint qualification*
(LICQ). Suppose $\mathbf{x}^\star$ is a local minimum of
:eqref:`eq_mdl-opt-standard-problem` with $f$, $g_i$, and $h_j$ continuously
differentiable. If the gradients of all active inequality constraints together
with the gradients of all equality constraints,
$\{\nabla g_i(\mathbf{x}^\star) : g_i(\mathbf{x}^\star) = 0\} \cup \{\nabla h_j(\mathbf{x}^\star) : j = 1, \ldots, p\}$,
are linearly independent, then multipliers satisfying :eqref:`eq_mdl-opt-kkt`
are guaranteed to exist (:citet:`Nocedal.Wright.2006`, chapter 12). LICQ
directly generalizes $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$, and
Exercise :exref:`ex_mdl-constrained-degenerate-constraints` illustrates what
can go wrong when it fails.

For convex problems, there is a much simpler check that requires finding only
a single feasible point rather than inspecting gradients at the unknown
solution. Call problem :eqref:`eq_mdl-opt-standard-problem` *convex* when $f$
and every $g_i$ are convex and every $h_j$ is affine. A convex problem
satisfies **Slater's condition** :cite:`Slater.1950` if there exists a point
$\bar{\mathbf{x}}$ such that

$$
\begin{aligned}
g_i(\bar{\mathbf{x}}) &< 0 \quad\textrm{for every } g_i \textrm{ that is not affine}, \\
g_i(\bar{\mathbf{x}}) &\le 0 \quad\textrm{for every affine } g_i, \\
h_j(\bar{\mathbf{x}}) &= 0 \quad\textrm{for every } j.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-slater`

In plain terms, $\bar{\mathbf{x}}$ must be feasible and lie strictly inside
every curved inequality constraint; if all constraints are affine, then mere
feasibility already satisfies Slater's condition. (When $f$ or some $g_i$ is
defined only on a subset of $\mathbb{R}^n$, which will not occur here,
$\bar{\mathbf{x}}$ must also lie in the relative interior of that domain.) For
the Euclidean ball, $\bar{\mathbf{x}} = \mathbf{0}$ immediately works because
$g(\mathbf{0}) = -r^2/2 < 0$. Whenever Slater's condition holds for a convex
problem with differentiable $f$ and $g_i$, every minimum is a KKT point
:cite:`Boyd.Vandenberghe.2004`. The deeper reason is strong duality, which
:numref:`subsec_mdl-strong-duality` derives from the very same condition.

Convexity also settles the second question, giving the single most useful
result of this section.

**Proposition (KKT sufficiency under convexity).** *In problem
:eqref:`eq_mdl-opt-standard-problem`, let $f$ and every $g_i$ be convex and
differentiable and every $h_j$ affine. If
$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$
satisfies the KKT conditions :eqref:`eq_mdl-opt-kkt`, then $\mathbf{x}^\star$ is
a global minimum.*

**Proof.** Define
$\varphi(\mathbf{x}) = \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.
Because $\lambda_i^\star \ge 0$, $\varphi$ is a nonnegative combination of
convex functions plus affine terms, so $\varphi$ is convex. Stationarity says
$\nabla\varphi(\mathbf{x}^\star) = \mathbf{0}$, which means $\mathbf{x}^\star$
is a *global* minimizer of $\varphi$ over all of $\mathbb{R}^n$. Now pick any
feasible $\mathbf{x}$. Since $\lambda_i^\star \ge 0$, $g_i(\mathbf{x}) \le 0$,
and $h_j(\mathbf{x}) = 0$, we have

$$
f(\mathbf{x}) \;\ge\; f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})
= \varphi(\mathbf{x}) \;\ge\; \varphi(\mathbf{x}^\star)
= f(\mathbf{x}^\star),
$$

where the final equality follows from complementary slackness
($\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$) and primal feasibility
($h_j(\mathbf{x}^\star) = 0$). $\blacksquare$

Notice a broader principle tucked inside this short proof. For *any* fixed
multipliers with $\boldsymbol{\lambda} \ge 0$, the first inequality holds for
every feasible $\mathbf{x}$, so minimizing the Lagrangian over $\mathbf{x}$
always yields a lower bound on the constrained optimum; at the KKT
multipliers, that lower bound is exact. :numref:`sec_mdl-lagrangian-duality`
builds the entire theory of duality from this observation. For convex
problems, then, the KKT conditions are always sufficient—and under Slater's
condition they are necessary as well, characterizing the global minima
completely.

## Projections and Projected Gradient Descent
:label:`subsec_mdl-projected-gd`

The KKT conditions tell us how to *recognize* an optimum, and for simple
constraints they let us solve for one by hand. To *compute* solutions more
generally, we pair gradient descent with the nearest-feasible-point operation:
take a standard gradient step, and whenever it leaves the feasible set, project
it right back in.

### Projection onto a Convex Set

For a nonempty closed convex set $C \subseteq \mathbb{R}^n$, the
**projection** of a point $\mathbf{y}$ onto $C$ is the unique point of $C$
closest to $\mathbf{y}$,

$$
\Pi_C(\mathbf{y}) = \mathop{\mathrm{argmin}}_{\mathbf{x} \in C}\; \tfrac12 \|\mathbf{x} - \mathbf{y}\|^2.
$$

Why does a unique minimizer always exist? Because we can intersect $C$ with a
large closed ball centered at $\mathbf{y}$ without excluding the optimum; that
intersection is closed and bounded (hence compact), so the continuous objective
attains its minimum there. Strict convexity of the squared distance then rules
out ties, making $\Pi_C$ a well-defined map.

We have already computed two projections in closed form: onto a hyperplane (the
closest-point problem) and onto a ball (the clipping formula). Projection onto
a half-space $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \ge b\}$ follows
immediately from the hyperplane formula. If $\mathbf{y}$ already satisfies the
inequality, we leave it alone; if it violates it, we move straight along
$\mathbf{a}$ onto the boundary hyperplane:

$$
\Pi(\mathbf{y}) = \mathbf{y} + \frac{\max(0,\; b - \mathbf{a}^\top\mathbf{y})}{\|\mathbf{a}\|^2}\,\mathbf{a}.
$$

Projections onto closed convex sets obey a remarkably clean geometric
characterization. Let $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$ be the minimizer
of $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ over $C$. For any other point
$\mathbf{x} \in C$, convexity guarantees that the entire line segment from
$\hat{\mathbf{x}}$ to $\mathbf{x}$ stays inside $C$. Moving from
$\hat{\mathbf{x}}$ toward $\mathbf{x}$ therefore cannot decrease the objective
to first order, so its directional derivative is nonnegative:
$(\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge 0$.
Flipping the sign gives the **obtuse-angle inequality** (or variational
inequality):

$$
\left(\mathbf{y} - \Pi_C(\mathbf{y})\right)^\top \left(\mathbf{x} - \Pi_C(\mathbf{y})\right) \;\le\; 0
\qquad \textrm{for all } \mathbf{x} \in C.
$$
:eqlabel:`eq_mdl-opt-proj-variational`

Geometrically, the *residual* vector $\mathbf{y} - \Pi_C(\mathbf{y})$ forms an
angle of at least $90^\circ$ with the vector from $\Pi_C(\mathbf{y})$ to every
point $\mathbf{x} \in C$ (:numref:`fig_mdl-opt-projection-inequality`, panel
(a)). Put differently, the hyperplane through $\Pi_C(\mathbf{y})$ with normal
$\mathbf{y} - \Pi_C(\mathbf{y})$ is a *supporting hyperplane* of $C$, keeping
the entire set $C$ on its far side.

![(a) The residual $\mathbf{y}-\Pi_C(\mathbf{y})$ makes an angle of at least $90^\circ$ with every direction into $C$, so $C$ lies entirely on the far side of the supporting line through $\Pi_C(\mathbf{y})$. (b) Projection onto a closed convex set never increases distances: $\|\mathbf{u}-\mathbf{v}\|\le\|\mathbf{x}-\mathbf{x}'\|$.](../img/mdl-opt-projection-inequality.svg)
:label:`fig_mdl-opt-projection-inequality`

Let us check this inequality on our running example. For $\mathbf{y} = (3, 4)$
projected onto the unit disk, the residual is
$(2.4, 3.2) = 4\,(0.6, 0.8)$—once again, the multiplier times the outward
normal. For any $\mathbf{x}$ with $\|\mathbf{x}\| \le 1$,

$$
(2.4, 3.2)^\top\bigl(\mathbf{x} - (0.6, 0.8)\bigr) = 4\,(0.6x_1 + 0.8x_2) - 4 \;\le\; 4\,\|\mathbf{x}\| - 4 \;\le\; 0,
$$

where the first inequality uses Cauchy--Schwarz,
$\mathbf{u}^\top\mathbf{v} \le \|\mathbf{u}\|\,\|\mathbf{v}\|$. The supporting
line here is $3x_1 + 4x_2 = 5$, the very hyperplane from our closest-point
example.

Just as importantly, the obtuse-angle inequality is also *sufficient*. Suppose
some point $\hat{\mathbf{x}} \in C$ satisfies
:eqref:`eq_mdl-opt-proj-variational`, and write
$\varphi(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$, whose gradient
at $\hat{\mathbf{x}}$ is $\hat{\mathbf{x}} - \mathbf{y}$. Because a convex
differentiable function always lies above its tangent planes
(:numref:`sec_mdl-convexity`), we have
$\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}}) + (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}})$.
The last term is nonnegative for every $\mathbf{x} \in C$ by assumption, so
$\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}})$ across all of $C$, proving
that $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$. Thus
:eqref:`eq_mdl-opt-proj-variational` *completely characterizes* the projection.
Every property we need for projected algorithms flows directly from it,
beginning with the fact that projections never stretch distances.

**Proposition (projections are nonexpansive).** *For a nonempty closed convex
set $C$ and all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,*

$$
\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\| \;\le\; \|\mathbf{x} - \mathbf{y}\|.
$$

**Proof.** Set $\mathbf{u} = \Pi_C(\mathbf{x})$ and $\mathbf{v} = \Pi_C(\mathbf{y})$.
Applying :eqref:`eq_mdl-opt-proj-variational` twice—first at $\mathbf{x}$ with
$\mathbf{v} \in C$, then at $\mathbf{y}$ with $\mathbf{u} \in C$—gives

$$
(\mathbf{x} - \mathbf{u})^\top(\mathbf{v} - \mathbf{u}) \le 0,
\qquad
(\mathbf{y} - \mathbf{v})^\top(\mathbf{u} - \mathbf{v}) \le 0.
$$

Adding these two inequalities yields
$(\mathbf{x} - \mathbf{y} - (\mathbf{u} - \mathbf{v}))^\top(\mathbf{v} - \mathbf{u}) \le 0$,
which rearranges to

$$
\|\mathbf{u} - \mathbf{v}\|^2 \;\le\; (\mathbf{x} - \mathbf{y})^\top (\mathbf{u} - \mathbf{v})
\;\le\; \|\mathbf{x} - \mathbf{y}\|\, \|\mathbf{u} - \mathbf{v}\|
$$

by Cauchy--Schwarz. Dividing by $\|\mathbf{u} - \mathbf{v}\|$ finishes the
proof (and when $\|\mathbf{u} - \mathbf{v}\| = 0$ the claim is trivial).
$\blacksquare$

Panel (b) of :numref:`fig_mdl-opt-projection-inequality` depicts this visually:
two points outside the disk are always at least as far apart as their
projections. Convexity of $C$ is essential here. Take the nonconvex two-point
set $C = \{-1, 1\}$: then $y = 0.01$ projects to $1$ while $y = -0.01$ projects
to $-1$, turning an input gap of $0.02$ into an output gap of $2$—and at
$y = 0$ the projection is not even unique. Projecting onto a nonconvex set can
be both discontinuous and multivalued.

### Projected Gradient Descent

**Projected gradient descent** (PGD) turns projection into an iterative solver
by projecting after every gradient step:

$$
\mathbf{x}_{k+1} = \Pi_C\left(\mathbf{x}_k - \eta\, \nabla f(\mathbf{x}_k)\right).
$$
:eqlabel:`eq_mdl-opt-pgd`

Every iterate stays feasible by construction. Better yet, the fixed points of
PGD are precisely the constrained stationary points. For any differentiable
$f$, closed convex $C$, and step size $\eta > 0$, the fixed-point equation
$\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta \nabla f(\mathbf{x}^\star))$
holds if and only if $\mathbf{x}^\star \in C$ satisfies the obtuse-angle
inequality :eqref:`eq_mdl-opt-proj-variational` with
$\mathbf{y} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$. Cancelling
$\mathbf{x}^\star$ and dividing by $\eta > 0$ turns that inequality into

$$
\nabla f(\mathbf{x}^\star)^\top (\mathbf{x} - \mathbf{x}^\star) \;\ge\; 0
\qquad \textrm{for all } \mathbf{x} \in C.
$$

In plain words, no feasible direction is a descent direction. This is the
general constrained first-order condition, reducing to
$\nabla f(\mathbf{x}^\star) = \mathbf{0}$ when $C = \mathbb{R}^n$. When $f$ is
convex, the tangent-plane inequality immediately makes this condition
sufficient for a global minimum; when $f$ is nonconvex, it certifies
stationarity. And when $C = \{\mathbf{x} : g_i(\mathbf{x}) \le 0\}$ for
differentiable convex $g_i$ satisfying Slater's condition, it is equivalent to
the KKT conditions.

Does the extra projection step slow gradient descent down? Remarkably, no—and
nonexpansiveness gives a three-line proof.

**Proposition (PGD contracts at least as fast as GD).** *Let $C$ be nonempty,
closed, and convex, let $f$ be twice continuously differentiable with
$\mu\mathbf{I} \preceq \nabla^2 f \preceq L\mathbf{I}$ for some
$0 < \mu \le L$, let $\mathbf{x}^\star$ be the minimizer of $f$ over $C$, and
let $\eta > 0$. Then the iterates of :eqref:`eq_mdl-opt-pgd` satisfy*

$$
\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \;\le\; \rho\, \|\mathbf{x}_k - \mathbf{x}^\star\|,
\qquad \rho = \max\left(|1 - \eta\mu|,\; |1 - \eta L|\right).
$$
:eqlabel:`eq_mdl-opt-pgd-contraction`

**Proof.** (i) Because $\mathbf{x}^\star$ is a fixed point of PGD, we can write
$\mathbf{x}^\star = \Pi_C(\mathbf{b})$ with
$\mathbf{b} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$.
(ii) That choice of $\mathbf{b}$ is the key step: applying nonexpansiveness to
$\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)$ and $\mathbf{b}$ strips away the
projection altogether,

$$
\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \;\le\; \left\|(\mathbf{x}_k - \mathbf{x}^\star) - \eta\left(\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star)\right)\right\|.
$$

(iii) By the fundamental theorem of calculus along the segment joining
$\mathbf{x}^\star$ to $\mathbf{x}_k$, we have
$\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star) = \bar{\mathbf{H}}(\mathbf{x}_k - \mathbf{x}^\star)$,
where
$\bar{\mathbf{H}} = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t(\mathbf{x}_k - \mathbf{x}^\star))\,dt$.
This averaged Hessian is symmetric with all eigenvalues in $[\mu, L]$, so every
eigenvalue of $\mathbf{I} - \eta\bar{\mathbf{H}}$ has the form $1 - \eta h$
with $h \in [\mu, L]$ and magnitude at most $\rho$. The right-hand side is
therefore at most $\rho\,\|\mathbf{x}_k - \mathbf{x}^\star\|$. $\blacksquare$

Once step (ii) removes the projection, all that remains is the contraction of
unconstrained gradient descent. For a quadratic
$f = \tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$, we have
$\bar{\mathbf{H}} = \mathbf{A}$, and step (iii) recovers the exact per-mode
factor $|1 - \eta\lambda_i(\mathbf{A})|$ from
:numref:`sec_mdl-gradient-based-optimization`, which is strictly less than $1$
whenever $0 < \eta < 2/L$. For a convex $L$-smooth $f$ that is not strongly
convex, PGD with $\eta = 1/L$ likewise preserves gradient descent's $O(1/k)$
objective-value rate :cite:`Nesterov.2018`.

Let us test how tight the contraction factor $\rho$ is in practice. Consider
the ill-conditioned quadratic
$f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ with condition number
$\kappa = 10^3$, minimized over the half-space $x_1 + x_2 \ge 1$ (written
$g(\mathbf{x}) = 1 - x_1 - x_2 \le 0$). We can solve the KKT conditions by
hand: the unconstrained minimizer $\mathbf{0}$ is infeasible, so the constraint
must be active. Stationarity, $(x_1, 1000\,x_2) - \lambda(1, 1) = \mathbf{0}$,
gives $x_1 = \lambda$ and $x_2 = \lambda/1000$, and plugging these into
$x_1 + x_2 = 1$ yields $\lambda^\star = 1000/1001 \approx 0.999$,
$\mathbf{x}^\star = (1000/1001,\; 1/1001) \approx (0.999, 0.001)$, and
$p^\star = 500/1001 \approx 0.4995$.

Now run PGD with step size $\eta = 0.0015$, safely below the stability limit
$2/L = 0.002$. The proposition guarantees a per-step factor of
$\rho = \max(1 - \eta,\ |1 - 1000\,\eta|) = 0.9985$. Starting from $(3, 4)$ at
distance $4.47$ from $\mathbf{x}^\star$, that bound predicts up to
$\log(4.47 \times 10^6)/\log(1/\rho) \approx 10{,}200$ steps to reach distance
$10^{-6}$. The code below runs PGD and inspects the worst single-step ratio,
the actual step count, the moment when every step begins hitting the boundary,
the contraction ratio along the boundary, and the final gradient.

```{.python .input #mdl-constrained-optimization-duality-projected-gradient-descent}
A = np.diag([1.0, 1000.0])                         # f(x) = 1/2 x^T A x, kappa = 1000
x_star = np.array([1000 / 1001, 1 / 1001])         # KKT point of x1 + x2 >= 1
project = lambda y: y + max(0.0, 1 - y.sum()) / 2  # onto the half-space x1 + x2 >= 1
eta = 1.5e-3
rho = max(abs(1 - eta), abs(1 - 1000 * eta))       # the GD contraction factor
x, dist, projected = np.array([3.0, 4.0]), [], []
for k in range(1000):
    y = x - eta * A @ x
    projected.append(y.sum() < 1)                  # did the step leave C?
    x = project(y)
    dist.append(np.linalg.norm(x - x_star))
dist, projected = np.array(dist), np.array(projected)
ratios = dist[1:] / dist[:-1]
steps = np.argmax(dist < 1e-6) + 1
on_line = np.flatnonzero(~projected).max() + 2     # every step from here on is projected
print(f'bound rho = {rho:.4f}   largest ratio in a step = '
      f'{ratios[dist[:-1] > 1e-12].max():.4f}')
print(f'steps to |x - x*| < 1e-6: {steps}   every step projected from step {on_line} on')
print(f'ratio per step on the line: {np.median(ratios[on_line:steps - 1]):.4f}'
      f'   1 - 500.5 eta = {1 - 500.5 * eta:.4f}')
print(f'grad f at the end = {A @ x}')
```

The largest single-step ratio, $0.9977$, comes close to $\rho = 0.9985$, so
the worst-case per-step bound is nearly tight. The total step count, however,
is a pleasant surprise: $738$ steps instead of $10{,}200$. The printed
diagnostics reveal why—the trajectory unfolds in two distinct phases. During
the first phase, the constraint is inactive after the initial step: the stiff
coordinate $x_2$ collapses within a few dozen iterations, while the flat
coordinate $x_1$ creeps inward by the factor $1 - \eta$ per step, just as in
unconstrained gradient descent. Reaching the boundary line from $x_1 = 3$
takes roughly $\log 3 / \eta \approx 730$ steps. From step $733$ onward, every
update hits the boundary and is projected onto the line $x_1 + x_2 = 1$. Along
that line, the quadratic has curvature $(1 + 1000)/2 = 500.5$, so the distance
to $\mathbf{x}^\star$ shrinks by $1 - 500.5\,\eta = 0.249$ per step—the exact
rate of one-dimensional gradient descent along the line—and just a handful of
additional steps finish the run. While $\rho$ must guard against the worst
curvature anywhere in the plane, once the constraint becomes active only the
curvature *along the boundary* matters. Finally, notice the gradient at
termination: $(0.999, 0.999) = \lambda^\star(1, 1) = -\lambda^\star\nabla g$.
Far from vanishing, it hands us back the multiplier $\lambda^\star = 1000/1001$
we derived by hand.

PGD is a workhorse whenever $\Pi_C$ is inexpensive to compute. Nonnegativity
constraints project onto the nonnegative orthant simply by zeroing negative
entries, and max-norm constraints on the rows of a weight matrix rescale each
row onto a ball after every update. Perhaps the most famous deep learning
application is generating adversarial examples via projected gradient ascent
on an input perturbation $\boldsymbol{\delta}$ with
$\|\boldsymbol{\delta}\|_\infty \le \epsilon$, where projection simply clips
each entry to $[-\epsilon, \epsilon]$ :cite:`Madry.Makelov.Schmidt.ea.2018`.
Keep in mind that gradient clipping also uses a ball projection, but it
projects the *gradient vector* rather than the parameters, so it is not PGD
for any constraint on the weights
(Exercise :exref:`ex_mdl-constrained-projected-fixed-points` highlights the
difference).

### Projection onto the Simplex

Attention weights and class probabilities live on the **probability simplex**
$\Delta = \{\mathbf{x} : \mathbf{x} \ge 0,\; \sum_i x_i = 1\}$. Given a vector
of real scores $\mathbf{y}$, how do we find the closest point on $\Delta$? The
problem

$$
\min_{\mathbf{x}}\; \tfrac12\|\mathbf{x} - \mathbf{y}\|^2
\quad \textrm{subject to} \quad \textstyle\sum_i x_i = 1, \;\; -x_i \le 0,
$$

is a small **quadratic program** (QP)—a convex quadratic objective subject to
affine constraints—and the KKT conditions collapse it to finding a single
scalar. Introduce a multiplier $\tau$ for the sum constraint and multipliers
$\lambda_i \ge 0$ for the nonnegativity constraints $-x_i \le 0$. Stationarity
gives $x_i - y_i + \tau - \lambda_i = 0$, and complementary slackness splits
the coordinates into two groups. Wherever $x_i > 0$, we must have
$\lambda_i = 0$ and thus $x_i = y_i - \tau$. Wherever $x_i = 0$, stationarity
gives $\lambda_i = \tau - y_i$, and dual feasibility $\lambda_i \ge 0$ forces
$y_i \le \tau$. Putting both cases together gives

$$
x_i^\star = \max(y_i - \tau,\, 0),
\qquad \textrm{with } \tau \textrm{ set by } \sum_i \max(y_i - \tau,\, 0) = 1.
$$
:eqlabel:`eq_mdl-opt-simplex`

In other words, projecting onto the simplex simply shifts every score down by
the same threshold $\tau$ and clips negative results to zero. Remarkably, this
threshold formula automatically satisfies stationarity, complementary slackness,
dual feasibility, and nonnegativity for *any* $\tau$; the unit-sum budget
$\sum_i x_i = 1$ alone pins down $\tau$. Because the left-hand side
$\varphi(\tau) = \sum_i \max(y_i - \tau, 0)$ is continuous, piecewise linear,
and strictly decreasing wherever it is positive, there is a unique $\tau$ that
solves $\varphi(\tau) = 1$.

A quick hand calculation shows how to find $\tau$. Take the already-sorted
vector $\mathbf{y} = (0.9, 0.5, -0.2)$, and guess that the $k$ largest scores
survive above zero. Then the budget $\sum_{j \le k}(y_j - \tau) = 1$ predicts
the candidate threshold $\tau_k = (\sum_{j \le k} y_j - 1)/k$, which is
self-consistent only if the $k$-th score stays strictly above $\tau_k$:

* $k = 1$: $\tau_1 = -0.1$, and $y_1 = 0.9 > \tau_1$;
* $k = 2$: $\tau_2 = 0.2$, and $y_2 = 0.5 > \tau_2$;
* $k = 3$: $\tau_3 \approx 0.067$, but $y_3 = -0.2 < \tau_3$.

The largest valid choice is $k = 2$, giving $\tau = 0.2$ and
$\mathbf{x}^\star = (0.7, 0.3, 0)$ (:numref:`fig_mdl-opt-simplex-threshold`).
For comparison, softmax applied to the same scores yields $(0.50, 0.33, 0.17)$,
keeping every entry strictly positive.

![Projection of $\mathbf{y}=(0.9,0.5,-0.2)$ onto the probability simplex in (a): every score is shifted down by the threshold $\tau=0.2$ and clipped at zero, with $\tau$ uniquely fixed by $\sum_i x_i=1$. Softmax on the same scores in (b) keeps every coordinate strictly positive.](../img/mdl-opt-simplex-threshold.svg)
:label:`fig_mdl-opt-simplex-threshold`

Two simple observations turn this hand check into a general algorithm. First,
the *support* $\{i : x_i^\star > 0\}$ always consists of the largest scores:
if $y_i > y_j$ and $x_j^\star > 0$, then
$x_i^\star \ge y_i - \tau > y_j - \tau = x_j^\star > 0$. (The *active* sign
constraints are the remaining zero coordinates.) Second, once the scores are
sorted as $u_1 \ge \cdots \ge u_n$, the test $u_k > \tau_k$ holds for every
$k$ up to the true support size and fails beyond it. Indeed, rearranging
$u_k > \tau_k$ gives $\sum_{j < k}(u_j - u_k) < 1$, which is just
$\varphi(u_k) < 1 = \varphi(\tau)$; because $\varphi$ is strictly decreasing,
this is equivalent to $u_k > \tau$. We therefore sort $\mathbf{y}$, pick the
largest $k$ with $u_k > \tau_k$, and set $\tau = \tau_k$, taking $O(n \log n)$
time :cite:`Held.Wolfe.Crowder.1974,Duchi.Shalev-Shwartz.Singer.ea.2008`.

When applied to attention scores, this simplex projection is known as
*sparsemax* :cite:`Martins.Astudillo.2016`, because it assigns exact zero
weight to low-scoring positions. Softmax and sparsemax differ by just one
regularization term: expanding the squared distance shows that sparsemax
maximizes $\mathbf{y}^\top\mathbf{x} - \tfrac12\|\mathbf{x}\|^2$ over
$\Delta$, whereas softmax maximizes $\mathbf{y}^\top\mathbf{x} + H(\mathbf{x})$,
replacing the quadratic penalty with Shannon entropy
(:eqref:`eq_mdl-opt-lse-entropy`).

Whenever we implement a projection, it is good practice to verify its output.
The KKT residuals are the natural first test, though we have to be careful
about which ones actually discriminate. The cell below tests six random scores
at both the computed $\tau$ and a perturbed value $\tau + 0.3$, and then checks
the optimality certificate from the obtuse-angle inequality
:eqref:`eq_mdl-opt-proj-variational`. Because the left-hand side of
:eqref:`eq_mdl-opt-proj-variational` is affine in $\mathbf{x}$, its maximum
over $\Delta$ occurs at a vertex, so testing the $n$ standard basis vectors
$\mathbf{e}_i$ is enough to certify the entire simplex.

```{.python .input #constrained-simplex-projection}
def project_simplex(y):
    """Sort-and-threshold projection onto {x >= 0, sum x = 1}; returns x* and tau."""
    u = np.sort(y)[::-1]
    tau_k = (np.cumsum(u) - 1) / np.arange(1, len(y) + 1)   # tau_k for k = 1..n
    k = np.nonzero(u > tau_k)[0][-1]                          # largest k with u_k > tau_k
    return np.maximum(y - tau_k[k], 0.0), tau_k[k]

y = np.random.default_rng().normal(size=6)
x, tau = project_simplex(y)
print('x* =', x.round(3), ' zeros:', int((x == 0).sum()), ' tau =', round(tau, 3))
for t in (tau, tau + 0.3):                     # three conditions hold for every tau ...
    xt, lam = np.maximum(y - t, 0), np.maximum(0, t - y)
    print(f'tau = {t:6.3f}: stationarity {abs(xt - y + t - lam).max():.0e}, '
          f'comp. slack {abs(lam * xt).max():.0e}, dual feas. {lam.min() >= 0}, '
          f'sum x = {xt.sum():.3f}')            # ... only the budget differs
cert = ((y - x) @ (np.eye(6) - x).T).max()     # max_i (y - x*)^T (e_i - x*)
print(f'certificate max_i (y - x*)^T(e_i - x*) = {cert:.0e}  '
      f'(<= 0 up to rounding: {cert <= 1e-12})')
```

Typically three to five of the six coordinates are set to exact zero; these are
the active nonnegativity constraints, each carrying the positive multiplier
$\tau - y_i$. Notice the diagnostic pitfall: stationarity, complementary
slackness, and dual feasibility all pass up to rounding error for *both* the
correct $\tau$ and the wrong one. Only the sum constraint catches the error, so
checking those three residuals alone cannot confirm the solution. The vertex
certificate does: seeing that $\max_i (\mathbf{y} - \mathbf{x}^\star)^\top(\mathbf{e}_i - \mathbf{x}^\star) \le 0$
up to rounding error proves that no point on the simplex lies closer to
$\mathbf{y}$.

## Proximal Operators and Coordinate Descent

### Soft-Thresholding and the Proximal Operator
:label:`subsec_mdl-proximal-operators`

A hard boundary is not the only way to keep weights under control. The
**lasso** :cite:`Tibshirani.1996` replaces a hard wall with a kinked
$\ell_1$ penalty added to a least-squares loss,

$$
F(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2 + \lambda\|\mathbf{w}\|_1,
\qquad \mathbf{A} \in \mathbb{R}^{n \times d},
$$

whose minimizer typically sets many coordinates to exact zero. We write the
penalty weight as $\lambda$ on purpose, because it acts like the Lagrange
multiplier for a norm constraint $\|\mathbf{w}\|_1 \le c$: every minimizer of
$F$ also minimizes the least-squares loss under the $\ell_1$ budget $c$ that it
uses, and conversely every budget $c > 0$ corresponds to some penalty $\lambda$.
:numref:`subsec_mdl-weight-decay-duality` proves both directions, for the
lasso and for $\ell_2$ weight decay alike.

To see what happens numerically, let us plant a sparse vector $\mathbf{w}$ in
$\mathbb{R}^8$ with five zero coordinates, generate $n = 100$ noisy linear
observations, and set $\lambda = 3$. We choose the step size $\eta = 1/L$,
where $L = \lambda_{\max}(\mathbf{A}^\top\mathbf{A})$ is the largest eigenvalue
of $\mathbf{A}^\top\mathbf{A}$ and the smoothness constant of the least-squares
term.

```{.python .input #mdl-constrained-optimization-duality-soft-thresholding-and-the-proximal-operator-1}
rng = np.random.default_rng()
w_planted = np.array([2.0, 0, 0, -1.5, 0, 0, 0, 1.0])
A = rng.normal(size=(100, 8))
b = A @ w_planted + 0.05 * rng.normal(size=100)
lam = 3.0
L = np.linalg.eigvalsh(A.T @ A).max()
eta = 1 / L
F = lambda w: 0.5 * ((A @ w - b)**2).sum() + lam * np.abs(w).sum()
soft = lambda z, t: np.sign(z) * np.maximum(np.abs(z) - t, 0.0)  # derived below
print(f'L = lambda_max(A^T A) = {L:.1f};  |a_j|^2 range from '
      f'{(A**2).sum(0).min():.0f} to {(A**2).sum(0).max():.0f}')
```

Here $L$ typically lands between $130$ and $190$, while the squared column
norms $\|\mathbf{a}_j\|^2$ of $\mathbf{A}$ sit around $100$.

The most naive algorithm to try is subgradient descent. Recall from
:numref:`sec_mdl-convexity` that the subgradients of $|w|$ are
$\mathrm{sign}(w)$ for $w \neq 0$ and the entire interval $[-1, 1]$ at the
kink $w = 0$. Picking $\mathrm{sign}(\mathbf{w})$ at each step gives the update
$\mathbf{w} \leftarrow \mathbf{w} - \eta\left(\mathbf{A}^\top(\mathbf{A}\mathbf{w} - \mathbf{b}) + \lambda\,\mathrm{sign}(\mathbf{w})\right)$.
Does it recover the five zeros?

```{.python .input #mdl-constrained-optimization-duality-soft-thresholding-and-the-proximal-operator-2}
w_sub, signs = np.zeros(8), []
for k in range(50):                                # subgradient descent, step 1/L
    w_sub = w_sub - eta * (A.T @ (A @ w_sub - b) + lam * np.sign(w_sub))
    signs.append(np.sign(w_sub))
null = w_planted == 0
S = np.array(signs)[-20:, null]
print(f'exact zeros: {int((w_sub == 0).sum())} of 8;  sign flips of the null '
      f'coordinates, last 20 steps: {(S[1:] != S[:-1]).sum()} of 95')
print(f'largest null coordinate: {np.abs(w_sub[null]).max():.3f};  '
      f'eta * lam = {eta * lam:.3f}')
```

It fails completely to produce exact zeros. None of the coordinates land on
zero, and the five *null* coordinates—those planted at zero—flip sign on
nearly every iteration, bouncing back and forth at an amplitude of about
$\eta\lambda$. The culprit is the kink at zero. At the true lasso solution, the
smooth gradient on a null coordinate has magnitude strictly smaller than
$\lambda$. Near zero, the penalty term $-\eta\lambda\,\mathrm{sign}(w_j)$
therefore dominates the step and kicks the coordinate across zero by about
$\eta\lambda$ every time. Because subgradient descent commits to a single slope
out of the interval $[-\lambda, \lambda]$ at each step, a fixed step size can
never settle at the exact slope that cancels the smooth gradient.

The cure is simple: instead of linearizing the kinked penalty, handle it
*exactly*. Start in one dimension by minimizing
$\lambda|x| + \tfrac12(x - z)^2$ over $x \in \mathbb{R}$. Because the objective
is convex, $x$ is optimal if and only if $0$ belongs to its set of subgradients:

* If $x \neq 0$, the objective is differentiable at $x$, so
  $\lambda\,\mathrm{sign}(x) + x - z = 0$. When $x > 0$ this gives
  $x = z - \lambda$, which requires $z > \lambda$; when $x < 0$ it gives
  $x = z + \lambda$, which requires $z < -\lambda$.
* If $x = 0$, the subgradients form the interval $[-\lambda, \lambda] - z$,
  which contains $0$ if and only if $|z| \le \lambda$.

These two cases partition the real line and combine into **soft-thresholding**,

$$
\mathrm{prox}_{\lambda|\cdot|}(z) \;=\; \mathrm{sign}(z)\, \max\left(|z| - \lambda,\, 0\right).
$$
:eqlabel:`eq_mdl-opt-soft-threshold`

Any input with magnitude at most $\lambda$ is snapped to exact zero, while
larger inputs are shrunk toward zero by $\lambda$. For example, with
$\lambda = 0.7$, $z = 0.5$ maps to $0$, $z = 2$ maps to $1.3$, and $z = -2$
maps to $-1.3$. That dead zone $[-\lambda, \lambda]$ is precisely how $\ell_1$
regularization creates exact sparsity, whereas $\ell_2$ regularization only
shrinks weights toward zero without ever zeroing them out.

Soft-thresholding is one instance of a much more general operation. The
**proximal operator** of a function $\psi$ trades off lowering $\psi(\mathbf{x})$
against staying close to a reference point $\mathbf{z}$:

$$
\mathrm{prox}_{\psi}(\mathbf{z}) \;=\; \mathop{\mathrm{argmin}}_{\mathbf{x}}\; \left( \psi(\mathbf{x}) + \tfrac12\, \|\mathbf{x} - \mathbf{z}\|^2 \right).
$$
:eqlabel:`eq_mdl-opt-prox`

This map is well defined for every *proper closed convex* function $\psi$—that
is, convex, taking values in $\mathbb{R} \cup \{+\infty\}$, finite at least
somewhere, and having a closed epigraph. Under those conditions, the objective
in :eqref:`eq_mdl-opt-prox` is $1$-strongly convex and therefore has a unique
minimizer :cite:`Parikh.Boyd.2014`.

Three special cases reveal how the proximal operator unifies ideas we already
know:

* When $\psi = 0$, the proximal operator is simply the identity map.
* When $\psi$ is the *indicator* of a closed convex set $C$ (equal to $0$ on
  $C$ and $+\infty$ outside), any minimizer must lie in $C$, so the proximal
  operator reduces to the Euclidean projection $\mathrm{prox}_\psi = \Pi_C$.
  Proximal operators are projections generalized from sets to functions.
* When $\psi = \lambda\|\cdot\|_1$, the objective separates completely across
  coordinates, so the proximal operator applies soft-thresholding
  :eqref:`eq_mdl-opt-soft-threshold` coordinate by coordinate.

Exercise :exref:`ex_mdl-constrained-group-lasso-prox` works out the proximal
operator of the Euclidean norm, which zeroes an entire block of weights at
once, and Exercise :exref:`ex_mdl-convexity-moreau-decomposition` shows that
the proximal operator of a norm and the projection onto its dual ball sum to
the identity.

### Proximal Gradient Descent

The lasso objective has the composite form $F = s + \psi$, combining a smooth
loss $s$ with a nonsmooth penalty $\psi$ whose proximal operator is cheap to
evaluate. Suppose $s$ is convex with an $L$-Lipschitz gradient, $\psi$ is
proper closed convex, and $0 < \eta \le 1/L$. By the descent lemma
(:numref:`sec_mdl-gradient-based-optimization`, with $1/\eta \ge L$), $s$ is
bounded above by the quadratic model
$s(\mathbf{x}_k) + \nabla s(\mathbf{x}_k)^\top(\mathbf{x} - \mathbf{x}_k) + \tfrac{1}{2\eta}\|\mathbf{x} - \mathbf{x}_k\|^2$.
Rather than linearizing $\psi$ as well, we minimize this quadratic upper bound
on $s$ together with the *exact* $\psi$. Completing the square shows that the
minimizer is

$$
\mathop{\mathrm{argmin}}_{\mathbf{x}}\; \psi(\mathbf{x}) + \frac{1}{2\eta}\left\|\mathbf{x} - \left(\mathbf{x}_k - \eta\nabla s(\mathbf{x}_k)\right)\right\|^2
\;=\; \mathrm{prox}_{\eta\psi}\left(\mathbf{x}_k - \eta\, \nabla s(\mathbf{x}_k)\right),
$$

which is an ordinary gradient step on the smooth part followed by a proximal
step on the kinked part. Iterating this update,

$$
\mathbf{x}_{k+1} = \mathrm{prox}_{\eta\psi}\left(\mathbf{x}_k - \eta\, \nabla s(\mathbf{x}_k)\right),
$$
:eqlabel:`eq_mdl-opt-ista`

gives **proximal gradient descent**, known as **ISTA** (the iterative
shrinkage-thresholding algorithm) when $\psi = \lambda\|\cdot\|_1$. It
subsumes both projected gradient descent (take $\psi$ to be the indicator of
$C$) and plain gradient descent (take $\psi = 0$).

Even though $F$ is not differentiable, proximal gradient descent with
$\eta = 1/L$ achieves the same $O(1/k)$ rate as smooth gradient descent:
$F(\mathbf{x}_k) - F^\star \le L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(2k)$
(:citet:`Beck.Teboulle.2009`, Theorem 3.1). Moreover, if $s$ is twice
continuously differentiable and $\mu$-strongly convex
($\mu\mathbf{I} \preceq \nabla^2 s$), our PGD contraction proof carries over
verbatim, because all it needs is that $\mathbf{x}^\star$ is a fixed point and
that $\mathrm{prox}_{\eta\psi}$ is nonexpansive just like $\Pi_C$
(Exercise :exref:`ex_mdl-convexity-accelerated-proximal-gradient` proves this).

For the lasso, the proximal step is soft-thresholding with threshold
$\eta\lambda$, which snaps every coordinate within $\eta\lambda$ of zero
directly to zero. That threshold matches the exact amplitude at which
subgradient descent was chattering, so ISTA should hold the null coordinates
cleanly at zero. Let us check.

```{.python .input #convexity-prox-ista}
w = np.zeros(8)
for k in range(50):                                # ISTA: gradient step, then prox
    w = soft(w - eta * A.T @ (A @ w - b), eta * lam)
print(f'exact zeros: {int((w == 0).sum())} of 8;  support recovered: '
      f'{np.array_equal(w != 0, w_planted != 0)};  '
      f'max deviation on the support: {np.abs(w - w_planted)[~null].max():.3f}')
```

After the same fifty iterations—at the same cost per step as subgradient
descent—ISTA recovers the exact support, setting all five null coordinates to
zero and leaving the three active coordinates nonzero. Those three entries sit
within about $0.05$ of the planted values; they cannot get much closer, because
the $\ell_1$ penalty shrinks even the true lasso solution toward zero by
roughly $\lambda/\|\mathbf{a}_j\|^2 \approx 0.03$.

### Coordinate Descent
:label:`subsec_mdl-coordinate-descent`

You may skip this subsection on a first reading: coordinate descent is the
standard solver for the lasso, though it is not how neural networks are
trained.

On the lasso, minimizing along a single coordinate $w_j$ while holding all
others fixed reduces to the exact one-dimensional problem we just solved. Let
$\mathbf{r}_{-j} = \mathbf{b} - \sum_{i \neq j}\mathbf{a}_i w_i$ be the
residual computed without coordinate $j$. The $j$-th subproblem,
$\tfrac12\|\mathbf{a}_j w_j - \mathbf{r}_{-j}\|^2 + \lambda|w_j|$, equals
$\|\mathbf{a}_j\|^2\left(\tfrac12(w_j - c_j/\|\mathbf{a}_j\|^2)^2 + (\lambda/\|\mathbf{a}_j\|^2)|w_j|\right)$
up to an additive constant, where $c_j = \mathbf{a}_j^\top\mathbf{r}_{-j}$.
Soft-thresholding with threshold $\lambda/\|\mathbf{a}_j\|^2$ solves it in
closed form:

$$
w_j \leftarrow \frac{\mathrm{sign}(c_j)\,\max(|c_j| - \lambda,\, 0)}{\|\mathbf{a}_j\|^2},
\qquad c_j = \mathbf{a}_j^\top\mathbf{r}_{-j}.
$$
:eqlabel:`eq_mdl-opt-lasso-coordinate`

Whenever a feature's correlation $c_j$ with the current residual has magnitude
at most $\lambda$, its weight is set immediately to zero. Cycling through the
coordinates with this update is the algorithm behind the widely used `glmnet`
package :cite:`Friedman.Hastie.Tibshirani.2010`. By updating the residual
incrementally after each coordinate change, one full sweep over all $d$
coordinates costs the same as a single gradient evaluation.

The same principle applies to any differentiable objective $f$ with
coordinate-wise smoothness constants $L_j$,

$$
f(\mathbf x+t\mathbf e_j)
\le f(\mathbf x)+t\,\partial_jf(\mathbf x)+\tfrac12L_jt^2,
$$
:eqlabel:`eq_mdl-coordinate-smoothness`

where minimizing the quadratic upper bound over $t$ produces the
**coordinate-descent** update

$$
x_j\leftarrow x_j-\frac{1}{L_j}\partial_jf(\mathbf x).
$$
:eqlabel:`eq_mdl-coordinate-step`

Because $L_j$ measures curvature along a single axis, it can be substantially
smaller than the global smoothness constant $L$, allowing larger steps: in our
lasso problem, $L_j = \|\mathbf{a}_j\|^2 \approx 100$, compared with
$L \in [130, 190]$. For a positive-definite quadratic
$f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf{A}\mathbf x-\mathbf b^\top\mathbf x$,
the upper bound is exact with $L_j = A_{jj}$, and the update

$$
x_j\leftarrow x_j-\frac{(\mathbf{A}\mathbf x-\mathbf b)_j}{A_{jj}}
$$
:eqlabel:`eq_mdl-quadratic-coordinate`

minimizes $f$ exactly along coordinate $j$. Sweeping cyclically through all
coordinates recovers the classical Gauss--Seidel iteration for
$\mathbf{A}\mathbf x = \mathbf b$.

How much faster is a coordinate sweep than an ISTA step of the same cost? Let
us count how many passes over the data each method takes to push the objective
gap $F(\mathbf{w}) - F^\star$ below $10^{-8}$ on our lasso problem.

```{.python .input #mdl-convexity-coordinate-descent}
w_ref = np.zeros(8)
for k in range(5000):                              # reference optimum by ISTA
    w_ref = soft(w_ref - eta * A.T @ (A @ w_ref - b), eta * lam)
F_star, col2 = F(w_ref), (A**2).sum(0)             # col2[j] = |a_j|^2
w_i, w_c, r = np.zeros(8), np.zeros(8), b.copy()   # r = b - A w_c, kept up to date
gap_i, gap_c = [], []
for sweep in range(30):
    w_i = soft(w_i - eta * A.T @ (A @ w_i - b), eta * lam)   # one ISTA step
    for j in range(8):                                       # one coordinate sweep
        c_j = A[:, j] @ r + col2[j] * w_c[j]                 # a_j^T r_{-j}
        w_new = soft(c_j, lam) / col2[j]
        r -= A[:, j] * (w_new - w_c[j])
        w_c[j] = w_new
    gap_i.append(F(w_i) - F_star); gap_c.append(F(w_c) - F_star)
passes = lambda gap: next((k + 1 for k, g in enumerate(gap) if g < 1e-8),
                          '> 30')                  # not reached in 30 passes
print(f'passes to gap < 1e-8: ISTA {passes(gap_i)}, coordinate descent {passes(gap_c)}')
d2l.plot(np.arange(1, 31), [np.maximum(gap_i, 1e-16), np.maximum(gap_c, 1e-16)],
         'passes over the data', 'F - F*', legend=['ISTA', 'coordinate descent'],
         yscale='log')                             # gaps floored at rounding level
```

On the logarithmic plot, both methods trace straight lines—linear
convergence—but coordinate descent drops much more steeply, typically reaching
a gap of $10^{-8}$ in four to six sweeps compared with fifteen to twenty-five
steps for ISTA. Nothing we proved in this section predicts that exact speedup;
the standard convergence theory for coordinate descent applies to randomly
chosen coordinates :cite:`Nesterov.2012` rather than cyclic sweeps.

Coordinate descent does have an important limitation. If a nonsmooth kink does
not separate across coordinates, coordinate descent can get stuck far from the
optimum: for $f(x, y) = \max(|x|, |y|)$ at $(1, 1)$, moving along $x$ alone or
along $y$ alone cannot decrease $f$, even though the global minimum $0$ sits at
the origin. Coordinate descent succeeds on the lasso specifically because its
nonsmooth penalty $\lambda\sum_j |w_j|$ is separable across coordinates
:cite:`Tseng.2001`. For sparse
linear models and block-alternating minimization, it remains the method of
choice. For training dense neural networks, however, it plays no role, because
modern accelerators are engineered for parallel matrix multiplications rather
than sequential scalar updates.

Notice one pattern running through everything we did in this section: every
time we wanted a multiplier, we first solved for $\mathbf{x}$. In
:numref:`sec_mdl-lagrangian-duality`, we flip the question around and ask how
to solve for the multipliers *directly*—and how any choice of multipliers
provides a certificate proving that no feasible point can beat a given lower
bound. Our clipping example, with $\lambda^\star = 4$ and $p^\star = 8$, will
be its opening case.

## Summary

At a constrained optimum, the gradient need not vanish; instead, no feasible
direction can decrease the objective. For a single equality constraint with
$\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$, this means
$\nabla f + \nu\nabla h = \mathbf{0}$ for a Lagrange multiplier $\nu$ that
measures how hard the solution presses against the constraint. Adding
inequality constraints leads to the four KKT conditions, where dual feasibility
($\lambda_i \ge 0$) enforces one-way walls and complementary slackness
($\lambda_i g_i = 0$) zeroes out the multiplier of every inactive constraint.
For convex problems, every KKT point is a global minimum, and under a
constraint qualification (such as LICQ or Slater's condition) every minimum is
a KKT point.

To compute constrained solutions iteratively, projected gradient descent
follows each gradient step with a Euclidean projection $\Pi_C$. Because
projection onto a closed convex set is unique, characterized by the
obtuse-angle inequality, and nonexpansive, projected gradient descent preserves
the convergence rate of unconstrained gradient descent. Projections onto a
ball, a half-space, and the probability simplex each reduce to solving for a
single scalar multiplier. Finally, the proximal operator generalizes projection
from sets to nonsmooth functions: soft-thresholding is the proximal operator of
$\lambda|\cdot|$, and both ISTA and coordinate descent use it to solve the
lasso with exact zeros that subgradient descent cannot attain.

## Exercises

1. **Degenerate constraints.** The Lagrange condition
   :eqref:`eq_mdl-opt-lagrange-condition` and the KKT conditions
   :eqref:`eq_mdl-opt-kkt` hold at a minimum only under a constraint
   qualification. Each of the following problems violates one.
    1. Minimize $f(\mathbf{x}) = x_1 + x_2^2$ over $\mathbb{R}^3$ subject to
       $h_1(\mathbf{x}) = x_3 = 0$ and $h_2(\mathbf{x}) = x_3 - x_1^2 = 0$.
       Show that the feasible set is a line, find the minimizer, and show
       that both constraint gradients are nonzero everywhere. Show that no
       multipliers $\nu_1, \nu_2$ satisfy
       $\nabla f + \nu_1 \nabla h_1 + \nu_2 \nabla h_2 = \mathbf{0}$ at the
       minimizer. Describe the same line by two other equality constraints
       for which multipliers exist. Is a constraint qualification a property
       of the feasible set or of the functions that describe it?
    1. Minimize $f(x) = x$ over $\mathbb{R}$ subject to $x^2 \le 0$. Find
       $p^\star$ and show that no $\lambda \ge 0$ satisfies the KKT
       stationarity condition at the minimizer. Which of LICQ and Slater's
       condition fails? Explain why for each.
    1. Minimize $f(x) = x$ over $\mathbb{R}$ subject to the constraint
       $-x \le 0$ written twice, $g_1(x) = g_2(x) = -x$. Show that LICQ fails
       and Slater's condition holds. Find all pairs
       $(\lambda_1^\star, \lambda_2^\star)$ that satisfy the KKT conditions at
       the minimizer. Under the relaxed constraints $-x \le u_1$ and
       $-x \le u_2$, compute the optimal value $p^\star(u_1, u_2)$, and verify
       that every KKT pair satisfies
       $p^\star(\mathbf{u}) \ge p^\star(\mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u}$
       for all $\mathbf{u}$. Is $p^\star$ differentiable at $\mathbf{0}$, and what
       takes the place of the relation
       $\boldsymbol{\lambda}^\star = -\nabla p^\star(\mathbf{0})$?
   :exlabel:`ex_mdl-constrained-degenerate-constraints`

1. **Projected fixed points.** For a closed convex set $C$, the fixed points
   of projected gradient descent :eqref:`eq_mdl-opt-pgd` are the constrained
   stationary points, whatever the step size $\eta > 0$.
    1. Let $C = \{\mathbf{x} : \mathbf{x} \ge 0\}$. Show that
       $\mathbf{x}^\star$ is a fixed point if and only if, for every
       coordinate $i$, either $\partial f / \partial x_i (\mathbf{x}^\star) = 0$
       and $x_i^\star \ge 0$, or
       $\partial f / \partial x_i (\mathbf{x}^\star) > 0$ and $x_i^\star = 0$.
       Show that these conditions are the KKT conditions for the constraints
       $-x_i \le 0$, and express the multipliers through
       $\nabla f(\mathbf{x}^\star)$.
    1. Gradient clipping updates
       $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\Pi_{B_c}(\nabla f(\mathbf{x}_k))$,
       where $B_c = \{\mathbf{z} : \|\mathbf{z}\| \le c\}$. Find the fixed
       points of this update and compare them with those of projected
       gradient descent onto the parameter ball
       $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$. Which of the two updates
       solves a constrained problem?
    1. Binary weights restrict a parameter to the nonconvex set
       $C = \{-1, 1\}$, and projecting onto $C$ rounds to the nearer element
       (to $1$ at a tie). Let $f(x) = \tfrac12 (x - a)^2$ with $0 < a < 1$.
       For which step sizes is $x = 1$ a fixed point of
       :eqref:`eq_mdl-opt-pgd`, and for which is $x = -1$ one? What do the
       iterates do when $\eta$ exceeds both ranges? Which point minimizes
       $f$ over $C$? Does either point satisfy
       $f'(x^\star)(x - x^\star) \ge 0$ for all $x \in C$? Which step of the
       fixed-point argument fails for this set?
    1. Run the PGD cell on the half-space with $\eta = 2.2/1000$ instead of
       $0.0015$. The projection is nonexpansive; why does the run still
       diverge? Which factor of the contraction proposition exceeds $1$?
   :exlabel:`ex_mdl-constrained-projected-fixed-points`

1. [code] **Sparsemax and the $\ell_1$ ball.** The function `project_simplex`
   computes the simplex projection :eqref:`eq_mdl-opt-simplex`, which is
   sparsemax; its first output is the projected vector.
    1. Show that adding the same constant to every score leaves the
       projection unchanged. Divide the scores by a temperature $T > 0$, and
       let $u_1 > u_2$ be the two largest entries of $\mathbf{y}$. Show that
       the projection of $\mathbf{y} / T$ is one-hot exactly when
       $T \le u_1 - u_2$, and find its limit as $T \to \infty$. Plot the
       number of nonzero entries of the projection of $\mathbf{y} / T$
       against $T$ for ten standard normal scores.
    1. Let $S$ be the support of $\mathbf{x}^\star$, and suppose that no
       $y_i$ equals the threshold $\tau$. Derive the Jacobian
       $\partial \mathbf{x}^\star / \partial \mathbf{y}$ from
       :eqref:`eq_mdl-opt-simplex` :cite:`Martins.Astudillo.2016` and check
       it against central finite differences. How many arithmetic operations
       does the backward pass through sparsemax need for an input of length
       $n$?
    1. Show that the projection onto the $\ell_1$ ball
       $\{\mathbf{x} : \|\mathbf{x}\|_1 \le z\}$ keeps the signs of
       $\mathbf{y}$ and, when $\|\mathbf{y}\|_1 > z$, equals
       $\mathrm{sign}(\mathbf{y}) \odot z\,\Pi_\Delta(|\mathbf{y}| / z)$
       :cite:`Duchi.Shalev-Shwartz.Singer.ea.2008`. Implement it with
       `project_simplex`. Certify the result with the variational inequality
       $(\mathbf{y} - \hat{\mathbf{x}})^\top(\mathbf{x} - \hat{\mathbf{x}}) \le 0$,
       checked at the $2n$ points $\pm z\,\mathbf{e}_i$. Why do these points
       suffice?

1. **Second-order conditions on the sphere.** ● The Lagrange condition is a
   first-order condition and cannot tell minima from saddles. A second-order
   test must account for the curvature of the constraint.
    1. Let $f$ and $h$ be twice continuously differentiable, and let
       $\mathbf{x}^\star$ satisfy :eqref:`eq_mdl-opt-lagrange-condition` with
       multiplier $\nu^\star$. For every curve $\mathbf{x}(t)$ on
       $\{h = 0\}$ with $\mathbf{x}(0) = \mathbf{x}^\star$ and
       $\dot{\mathbf{x}}(0) = \mathbf{v}$, show that

       $$\left.\frac{d^2}{dt^2} f(\mathbf{x}(t))\right|_{t=0} = \mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}(\mathbf{x}^\star, \nu^\star)\, \mathbf{v},$$

       with the Lagrangian :eqref:`eq_mdl-opt-lagrangian-eq`. Conclude that
       $\mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}\, \mathbf{v} \ge 0$
       for every tangent direction $\mathbf{v}$ at a local minimum. Why is the
       Hessian of $f$ alone the wrong matrix for this test? Conversely,
       positive definiteness on the tangent space implies a strict local
       minimum (:citet:`Nocedal.Wright.2006`, chapter 12).
    1. Let $\mathbf{A}$ be symmetric with distinct eigenvalues
       $a_1 < a_2 < \cdots < a_n$ and unit eigenvectors
       $\mathbf{q}_1, \ldots, \mathbf{q}_n$, and minimize
       $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$ subject to
       $\mathbf{x}^\top \mathbf{x} = 1$. Show that the Lagrange points are
       $\pm \mathbf{q}_i$ with $\nu^\star = -a_i$, and find the eigenvalues of
       $\nabla^2_{\mathbf{x}} \mathcal{L}$ on the tangent space at
       $\mathbf{q}_i$. Which Lagrange points are local minima, which are local
       maxima, and which are saddles? How many independent descent directions
       does $\mathbf{q}_i$ have?
    1. Projected gradient descent onto the sphere,
       $\mathbf{x} \leftarrow (\mathbf{x} - \eta \nabla f(\mathbf{x})) / \|\mathbf{x} - \eta \nabla f(\mathbf{x})\|$,
       projects onto a nonconvex set. Show that for this $f$ it is the power
       iteration for $\mathbf{I} - 2\eta \mathbf{A}$. Assume $a_1 + a_n > 0$.
       From a random start, what do the iterates $\mathbf{x}_k$ and the values
       $f(\mathbf{x}_k)$ do? How does the answer depend on $\eta > 0$?

1. **Block soft-thresholding (the group lasso).** The penalty
   $\lambda\|\mathbf{w}\|_2$ on a whole vector is kinked only at
   $\mathbf{w} = \mathbf{0}$. Its subgradients are $\mathbf{w}/\|\mathbf{w}\|$
   for $\mathbf{w} \neq \mathbf{0}$ and every vector of the unit ball at
   $\mathbf{0}$ (times $\lambda$).
    1. Show that
       $\mathrm{prox}_{\lambda\|\cdot\|_2}(\mathbf{z}) = \max\left(0,\; 1 - \lambda/\|\mathbf{z}\|\right)\mathbf{z}$.
    1. Compare the zero pattern with that of coordinatewise
       soft-thresholding :eqref:`eq_mdl-opt-soft-threshold`. When does the block prox return
       exactly $\mathbf{0}$, and can it zero a single coordinate of a block?
    1. What does the penalty $\lambda\sum_u\|\mathbf{w}_u\|_2$, summed over the
       vectors $\mathbf{w}_u$ of outgoing weights of the hidden units $u$ of a
       network, do to the network?
    1. [code] Verify (a) against a brute-force minimization on a 2-D grid for
       $\mathbf{z} = (0.3, 0.4)$ and $\mathbf{z} = (3, 4)$ with $\lambda = 1$.
       The answers are $\mathbf{0}$ and $(2.4, 3.2)$; where has $(2.4, 3.2)$
       appeared in this section?
   :exlabel:`ex_mdl-constrained-group-lasso-prox`

1. **Moreau decomposition.** ● The *convex conjugate* of $\psi$ is
   $\psi^*(\mathbf{y}) = \sup_{\mathbf{x}}\left(\mathbf{y}^\top\mathbf{x} - \psi(\mathbf{x})\right)$.
   For a proper closed convex $\psi$, two facts connect it with the proximal
   operator :eqref:`eq_mdl-opt-prox`: $\mathbf{x} = \mathrm{prox}_\psi(\mathbf{z})$
   exactly when $\mathbf{z} - \mathbf{x} \in \partial \psi(\mathbf{x})$ (the
   set of subgradients), and $\mathbf{y} \in \partial \psi(\mathbf{x})$
   exactly when $\mathbf{x} \in \partial \psi^*(\mathbf{y})$
   :cite:`Parikh.Boyd.2014`.
    1. Derive the Moreau decomposition
       $\mathbf{z} = \mathrm{prox}_\psi(\mathbf{z}) + \mathrm{prox}_{\psi^*}(\mathbf{z})$.
    1. The conjugate of $\psi = \lambda\|\cdot\|_1$ is the indicator of the box
       $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$. Show that the
       decomposition writes soft-thresholding
       :eqref:`eq_mdl-opt-soft-threshold` as $\mathbf{z}$ minus a clipped
       copy of $\mathbf{z}$.
    1. The conjugate of $\psi = \lambda\|\cdot\|_2$ is the indicator of the
       $\ell_2$ ball of radius $\lambda$, whose prox is the projection onto
       the ball, the clip-by-norm map of gradient clipping. Use the
       decomposition to recover the block soft-thresholding formula of the
       group-lasso exercise from the clipping formula. For $\mathbf{z} = (3, 4)$
       and $\lambda = 1$, which two vectors of this section add up to
       $\mathbf{z}$?
   :exlabel:`ex_mdl-convexity-moreau-decomposition`

1. [code] **Accelerated proximal gradient.** ● FISTA
   :cite:`Beck.Teboulle.2009` adds momentum to the ISTA update
   :eqref:`eq_mdl-opt-ista` for $F = s + \psi$. With
   $\mathbf{v}_1 = \mathbf{x}_0$ and $t_1 = 1$, it iterates for $k \ge 1$

   $$\begin{aligned}\mathbf{x}_k &= \mathrm{prox}_{\eta \psi}\bigl(\mathbf{v}_k - \eta \nabla s(\mathbf{v}_k)\bigr), \\ t_{k+1} &= \tfrac12\Bigl(1 + \sqrt{1 + 4t_k^2}\Bigr), \\ \mathbf{v}_{k+1} &= \mathbf{x}_k + \frac{t_k - 1}{t_{k+1}}\,(\mathbf{x}_k - \mathbf{x}_{k-1}).\end{aligned}$$

   With $\eta = 1/L$ its objective gap satisfies
   $F(\mathbf{x}_k) - F^\star \le 2L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(k+1)^2$,
   against $O(1/k)$ for ISTA.
    1. Show that the prox is nonexpansive,
       $\|\mathrm{prox}_{\eta\psi}(\mathbf{u}) - \mathrm{prox}_{\eta\psi}(\mathbf{v})\| \le \|\mathbf{u} - \mathbf{v}\|$,
       by adding the two subgradient inequalities that characterize
       $\mathrm{prox}_{\eta\psi}(\mathbf{u})$ and
       $\mathrm{prox}_{\eta\psi}(\mathbf{v})$, as in the proof for
       projections.
    1. Add FISTA to the ISTA cell, on the same lasso, and plot
       $F(\mathbf{x}_k) - F^\star$ of both methods over 100 steps on a
       logarithmic axis, with $F^\star$ taken from $10^4$ ISTA steps. Which
       method is ahead after 10 steps, and which after 30? Explain the
       straight line of ISTA with the PGD contraction proposition and part (a).
    1. Is $F(\mathbf{x}_k)$ monotone along the FISTA iterates?
    1. Replace $\mathbf{A}$ by a $30 \times 100$ Gaussian matrix, plant the
       same three nonzero coefficients, and run both methods for 1000 steps.
       Over which range of $k$ is FISTA ahead now? Why does ISTA still end
       with a fast linear phase, although $\mathbf{A}^\top\mathbf{A}$ is
       singular?
   :exlabel:`ex_mdl-convexity-accelerated-proximal-gradient`

[Discussions](https://d2l.discourse.group/t/36831)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.6]{.kicker}

Constrained Optimization<br>**Lagrange multipliers · KKT conditions · Euclidean & simplex projections · proximal operators & ISTA · coordinate descent**
:::
:::

::: {.slide title="When the Optimum Has a Nonzero Gradient"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Projecting $\mathbf{y} = (3, 4)$ onto the unit disk $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - 1) \le 0$ minimizes $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|_2^2$:

$$\nabla f(\mathbf{x}^\star) + 4\,\nabla g(\mathbf{x}^\star) = (-2.4, -3.2) + 4(0.6, 0.8) = \mathbf{0}$$

- **Nonzero gradient at $\mathbf{x}^\star = (0.6, 0.8)$:** $\nabla f(\mathbf{x}^\star) = (-2.4, -3.2) \neq \mathbf{0}$.
- **Force balance:** No feasible move lowers $f$ because the outward boundary normal $\nabla g(\mathbf{x}^\star)$ pushes back with multiplier $\lambda^\star = 4$.
:::

::: {.col .fig}
@fig:ch35-opt-boundary-force-balance
:::
:::
:::

::: {.slide title="Equality Constraints and Tangency of Level Sets"}
[Lagrange Multipliers]{.kicker}

::: {.cols .vc}
::: {.col}
Consider minimizing a differentiable objective $f(\mathbf{x})$ on an equality surface $h(\mathbf{x}) = 0$ (such as the unit circle $h(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) = 0$):

- **Crossing level sets cannot be optimal:** If a level set of $f$ crosses $\{h = 0\}$, then $\nabla f$ has a nonzero component tangent to the surface, providing a feasible descent direction.
- **Tangency condition:** At any local extremum $\mathbf{x}^\star$ where $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$, the level set of $f$ is tangent to $\{h = 0\}$, so their normal vectors are parallel:
  $$\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla h(\mathbf{x}^\star) = \mathbf{0}$$
:::

::: {.col .fig}
@fig:mdl-opt-lagrange-tangency
:::
:::
:::

::: {.slide title="The Lagrangian and Stationarity on the Circle"}
[Lagrange Multipliers]{.kicker}

We package both requirements—parallel normals and feasibility $h(\mathbf{x}) = 0$—into a single function of $(\mathbf{x}, \nu)$, the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu\,h(\mathbf{x}) \implies \nabla_{\mathbf{x}}\mathcal{L} = \nabla f(\mathbf{x}) + \nu\,\nabla h(\mathbf{x}) = \mathbf{0}, \qquad \frac{\partial\mathcal{L}}{\partial\nu} = h(\mathbf{x}) = 0$$

- **Solving the unit-circle problem ($\mathbf{y} = (3, 4)$):**
  1. Stationarity $\nabla_{\mathbf{x}}\mathcal{L} = (\mathbf{x} - \mathbf{y}) + \nu\mathbf{x} = \mathbf{0}$ places $\mathbf{x} = \frac{\mathbf{y}}{1 + \nu}$ on the ray through $\mathbf{y}$.
  2. Feasibility $\|\mathbf{x}\|_2 = 1$ requires $|1 + \nu| = \|\mathbf{y}\|_2 = 5$, yielding two solutions:
     - **$\nu^\star = +4$:** $\mathbf{x}^\star = (0.6, 0.8)$ at distance $4$ from $\mathbf{y}$ ($f = 8$, the **global minimum**).
     - **$\nu = -6$:** $\mathbf{x} = (-0.6, -0.8)$ at distance $6$ from $\mathbf{y}$ ($f = 18$, the **global maximum**).
- **Why $\nabla h(\mathbf{x}^\star) \neq \mathbf{0}$ matters:** On $\min x_1$ subject to $h(x_1, x_2) = x_1^2 + x_2^2 = 0$, the unique feasible point $\mathbf{0}$ has $\nabla f(\mathbf{0}) = (1, 0)$ but $\nabla h(\mathbf{0}) = \mathbf{0}$, so no multiplier $\nu$ exists.
:::

::: {.slide title="Closest Point on a Hyperplane and Shadow Prices"}
[Lagrange Multipliers]{.kicker}

::: {.cols .vc}
::: {.col}
Minimizing $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|_2^2$ on the hyperplane $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$ gives $\mathbf{x} = -\nu\mathbf{a}$ and:

$$\nu^\star = -\frac{b}{\|\mathbf{a}\|^2}, \qquad \mathbf{x}^\star = \frac{b}{\|\mathbf{a}\|^2}\,\mathbf{a}, \qquad p^\star(b) = \frac{b^2}{2\|\mathbf{a}\|^2}$$

- **Shadow price:** Differentiating $p^\star(b)$ with respect to threshold $b$ reveals $\frac{\partial p^\star(b)}{\partial b} = -\nu^\star$.
:::

::: {.col .fig}
@fig:ch35-opt-hyperplane-shadow-price
:::
:::
:::

::: {.slide title="Inequality Constraints: Active vs. Inactive Walls"}
[Inequality Constraints]{.kicker}

::: {.cols .vc}
::: {.col}
Minimizing $\tfrac12\|\mathbf{x} - \mathbf{y}\|_2^2$ over the solid ball $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - r^2) \le 0$ splits into two regimes:

$$\mathbf{x}^\star = \frac{\mathbf{y}}{\max\big(1,\, \|\mathbf{y}\|/r\big)}, \qquad \lambda^\star = \max\!\left(0,\, \frac{\|\mathbf{y}\|}{r} - 1\right)$$

- **Inactive ($g(\mathbf{x}^\star) < 0$, $\|\mathbf{y}\| < r$):** $\mathbf{x}^\star = \mathbf{y}$ lies inside the ball and $\lambda^\star = 0$.
- **Active ($g(\mathbf{x}^\star) = 0$, $\|\mathbf{y}\| \ge r$):** $\mathbf{x}^\star = r\mathbf{y}/\|\mathbf{y}\|$ sits on the boundary sphere with $\lambda^\star \ge 0$.
:::

::: {.col .fig}
@fig:ch35-opt-active-vs-inactive-ball
:::
:::
:::

::: {.slide title="Verifying Force Balance in Gradient Clipping"}
[Inequality Constraints]{.kicker}

For $\mathbf{y} = (3, 4)$ and radius $r = 1$, the active constraint formula yields $\lambda^\star = 5/1 - 1 = 4$ and $\mathbf{x}^\star = (0.6, 0.8)$:

@!mdl-constrained-optimization-duality-active-and-inactive-constraints

::: {.d2l-note .rule}
**Why inequality multipliers must be nonnegative ($\lambda^\star \ge 0$):** Since $\nabla g(\mathbf{x}^\star)$ points *outward* from the feasible set $\{g \le 0\}$, stationarity $-\nabla f(\mathbf{x}^\star) = \lambda^\star \nabla g(\mathbf{x}^\star)$ with $\lambda^\star \ge 0$ ensures that the downhill direction $-\nabla f$ pulls outward into the wall. If $\lambda^\star < 0$, $-\nabla f$ would point *inward*, so stepping into the interior would lower $f$!
:::
:::

::: {.slide title="The Four Karush–Kuhn–Tucker (KKT) Conditions"}
[KKT Conditions]{.kicker}

::: {.cols .vc}
::: {.col}
For $\min_{\mathbf{x}} f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \le 0$ and $h_j(\mathbf{x}) = 0$, form $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$:

$$\nabla f(\mathbf{x}^\star) + \sum_{i=1}^m \lambda_i^\star\nabla g_i(\mathbf{x}^\star) + \sum_{j=1}^p \nu_j^\star\nabla h_j(\mathbf{x}^\star) = \mathbf{0}$$

- **Feasibility:** Primal ($g_i(\mathbf{x}^\star) \le 0, h_j(\mathbf{x}^\star) = 0$) and Dual ($\lambda_i^\star \ge 0$).
- **Complementary slackness:** $\lambda_i^\star g_i(\mathbf{x}^\star) = 0$ for every $i$ (inactive walls exert zero force).
:::

::: {.col .fig}
@fig:ch35-opt-kkt-four-conditions
:::
:::
:::

::: {.slide title="Normal Cone Geometry of Active Constraints"}
[KKT Conditions]{.kicker}

::: {.cols .vc}
::: {.col}
By complementary slackness, inactive constraints ($g_i(\mathbf{x}^\star) < 0$) have $\lambda_i^\star = 0$ and drop out of the stationarity sum. Let $\mathcal{A}(\mathbf{x}^\star) = \{i : g_i(\mathbf{x}^\star) = 0\}$ denote the active set, with $\lambda_i^\star \ge 0$:

$$-\nabla f(\mathbf{x}^\star) = \sum_{i \in \mathcal{A}(\mathbf{x}^\star)} \lambda_i^\star\,\nabla g_i(\mathbf{x}^\star)$$

- **The normal cone:** At an optimum, the steepest descent direction $-\nabla f(\mathbf{x}^\star)$ must lie inside the convex cone spanned by the outward normals $\nabla g_i(\mathbf{x}^\star)$ of the active walls.
- **Nonconvex caveat:** On $\min_{x \in [-1, 1]} -x^2$, the interior point $x = 0$ satisfies all four KKT conditions with $\lambda_1 = \lambda_2 = 0$, yet $x = 0$ is the global *maximum*!
:::

::: {.col .fig}
@fig:mdl-opt-kkt-active-set
:::
:::
:::

::: {.slide title="Constraint Qualifications: LICQ and Slater's Rule"}
[KKT Necessity & Sufficiency]{.kicker}

Two regularity conditions guarantee that every local minimum is a KKT point (preventing degenerate boundary cusps where constraint gradients collapse):

- **1. Linear Independence Constraint Qualification (LICQ — general smooth problems):**
  The gradients of all active inequalities and all equalities at $\mathbf{x}^\star$,
  $$\big\{\nabla g_i(\mathbf{x}^\star) : i \in \mathcal{A}(\mathbf{x}^\star)\big\} \cup \big\{\nabla h_j(\mathbf{x}^\star) : j = 1, \ldots, p\big\},$$
  are linearly independent.
- **2. Slater's Condition (convex problems — $f, g_i$ convex, $h_j$ affine):**
  There exists a single strictly feasible point $\bar{\mathbf{x}}$ satisfying:
  $$g_i(\bar{\mathbf{x}}) < 0 \;\text{ (for all non-affine } g_i\text{)}, \qquad g_i(\bar{\mathbf{x}}) \le 0 \;\text{ (for affine } g_i\text{)}, \qquad h_j(\bar{\mathbf{x}}) = 0 \;\forall\,j$$
  Checking Slater's condition requires finding just one interior point (e.g., $\bar{\mathbf{x}} = \mathbf{0}$ for any norm ball of radius $r > 0$), without knowing $\mathbf{x}^\star$ in advance!
:::

::: {.slide title="Proving KKT Sufficiency Under Convexity"}
[KKT Necessity & Sufficiency]{.kicker}

For convex problems ($f$ and $g_i$ convex and differentiable, $h_j$ affine), the KKT conditions are **always sufficient** for global optimality—requiring no constraint qualification at all:

- **Proposition:** If $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ satisfies the KKT conditions on a convex problem, then $f(\mathbf{x}^\star) \le f(\mathbf{x})$ for every feasible $\mathbf{x}$.
- **Three-line proof:**
  1. Fix the KKT multipliers and consider $\varphi(\mathbf{x}) = \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) = f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})$. Because $\lambda_i^\star \ge 0$, $\varphi(\mathbf{x})$ is a nonnegative weighted sum of convex functions plus affine functions, so $\varphi$ is convex on $\mathbb{R}^n$.
  2. By KKT stationarity, $\nabla\varphi(\mathbf{x}^\star) = \mathbf{0}$, so $\mathbf{x}^\star$ is an unconstrained global minimizer of $\varphi$: $\varphi(\mathbf{x}) \ge \varphi(\mathbf{x}^\star)$ for all $\mathbf{x} \in \mathbb{R}^n$.
  3. For any feasible $\mathbf{x}$ ($g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0$), complementary slackness ($\lambda_i^\star g_i(\mathbf{x}^\star) = 0$) gives:
     $$f(\mathbf{x}) \ge f(\mathbf{x}) + \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^\star h_j(\mathbf{x}) = \varphi(\mathbf{x}) \ge \varphi(\mathbf{x}^\star) = f(\mathbf{x}^\star)$$
:::

::: {.slide title="Euclidean Projection onto a Closed Convex Set"}
[Projections]{.kicker}

For any nonempty closed convex set $C \subseteq \mathbb{R}^n$, the **Euclidean projection** $\Pi_C(\mathbf{y})$ maps $\mathbf{y} \in \mathbb{R}^n$ to the unique closest point in $C$:

$$\Pi_C(\mathbf{y}) = \mathop{\mathrm{argmin}}_{\mathbf{x} \in C}\;\tfrac12\|\mathbf{x} - \mathbf{y}\|_2^2$$

- **Existence and uniqueness:** Intersecting $C$ with a closed ball around $\mathbf{y}$ creates a compact set (guaranteeing existence by continuity), and $1$-strong convexity of $\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ guarantees uniqueness.
- **Closed-form dictionary of elementary projections:**
  - *Euclidean ball $\{ \|\mathbf{x}\|_2 \le r \}$:* $\Pi_C(\mathbf{y}) = \frac{\mathbf{y}}{\max(1,\, \|\mathbf{y}\|_2 / r)}$.
  - *Axis-aligned box $[a_i, b_i]$:* $\Pi_C(\mathbf{y})_i = \min\big(b_i, \max(a_i, y_i)\big)$ (coordinatewise clipping).
  - *Half-space $\{ \mathbf{a}^\top\mathbf{x} \ge b \}$:* $\Pi_C(\mathbf{y}) = \mathbf{y} + \frac{\max(0,\, b - \mathbf{a}^\top\mathbf{y})}{\|\mathbf{a}\|_2^2}\,\mathbf{a}$.
:::

::: {.slide title="The Obtuse-Angle Variational Inequality"}
[Projections]{.kicker}

::: {.cols .vc}
::: {.col}
A point $\hat{\mathbf{x}} \in C$ is the projection $\Pi_C(\mathbf{y})$ if and only if the residual vector $\mathbf{y} - \hat{\mathbf{x}}$ forms an obtuse (or right) angle $\ge 90^\circ$ with every feasible chord $\mathbf{x} - \hat{\mathbf{x}}$ into $C$ ($\forall\,\mathbf{x} \in C$):

$$\big(\mathbf{y} - \Pi_C(\mathbf{y})\big)^\top\big(\mathbf{x} - \Pi_C(\mathbf{y})\big) \le 0$$

- **Necessity:** Moving along $(1-\theta)\hat{\mathbf{x}} + \theta\mathbf{x} \in C$ cannot lower $\varphi(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$, so $\nabla\varphi(\hat{\mathbf{x}})^\top(\mathbf{x} - \hat{\mathbf{x}}) = (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge 0$.
- **Sufficiency:** By the tangent bound for convex $\varphi$, $\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}}) + (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge \varphi(\hat{\mathbf{x}})$.
:::

::: {.col .fig}
@fig:mdl-opt-projection-inequality
:::
:::
:::

::: {.slide title="Proving Projections Are Nonexpansive"}
[Projections]{.kicker}

::: {.cols .vc}
::: {.col}
Projecting two points onto a closed convex set $C$ never increases the Euclidean distance between them:

$$\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\|_2 \le \|\mathbf{x} - \mathbf{y}\|_2$$

- **Three-line proof:** Sum the obtuse-angle inequalities $(\mathbf{x} - \mathbf{u})^\top(\mathbf{v} - \mathbf{u}) \le 0$ and $(\mathbf{y} - \mathbf{v})^\top(\mathbf{u} - \mathbf{v}) \le 0$ (where $\mathbf{u} = \Pi_C(\mathbf{x}), \mathbf{v} = \Pi_C(\mathbf{y})$) to get $\|\mathbf{u} - \mathbf{v}\|_2^2 \le (\mathbf{x} - \mathbf{y})^\top(\mathbf{u} - \mathbf{v})$, and apply Cauchy–Schwarz.
:::

::: {.col .fig}
@fig:ch35-opt-projection-nonexpansive
:::
:::
:::

::: {.slide title="Projected Gradient Descent and Fixed Points"}
[Projected Gradient Descent]{.kicker}

::: {.cols .vc}
::: {.col}
**Projected Gradient Descent (PGD)** alternates a gradient step with Euclidean projection onto $C$:

$$\mathbf{x}_{k+1} = \Pi_C\big(\mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)\big)$$

- **Fixed-point certificate:** $\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star))$ iff $\nabla f(\mathbf{x}^\star)^\top(\mathbf{x} - \mathbf{x}^\star) \ge 0$ for all $\mathbf{x} \in C$ (no feasible descent direction; global optimum when $f$ is convex).
:::

::: {.col .fig}
@fig:ch35-opt-pgd-step-cycle
:::
:::
:::

::: {.slide title="Why PGD Contracts as Fast as Unconstrained GD"}
[Projected Gradient Descent]{.kicker}

Does adding a projection step after every gradient update slow down convergence? On a $\mu$-strongly convex, $L$-smooth function ($\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$), nonexpansiveness guarantees the *exact same* worst-case contraction factor $\rho = \max(|1 - \eta\mu|, |1 - \eta L|)$ as unconstrained gradient descent:

- **Step 1 — Write $\mathbf{x}^\star$ as a fixed point:** $\mathbf{x}^\star = \Pi_C\big(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)\big)$.
- **Step 2 — Strip away $\Pi_C$ via nonexpansiveness:**
  $$\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| = \big\|\Pi_C(\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)) - \Pi_C(\mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star))\big\| \le \big\|(\mathbf{x}_k - \mathbf{x}^\star) - \eta\big(\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}^\star)\big)\big\|$$
- **Step 3 — Apply the fundamental theorem of calculus:** With $\bar{\mathbf{H}} = \int_0^1 \nabla^2 f(\mathbf{x}^\star + t(\mathbf{x}_k - \mathbf{x}^\star))\,dt$ ($\mu\mathbf{I} \preceq \bar{\mathbf{H}} \preceq L\mathbf{I}$),
  $$\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| = \big\|(\mathbf{I} - \eta\bar{\mathbf{H}})(\mathbf{x}_k - \mathbf{x}^\star)\big\| \le \max\big(|1 - \eta\mu|,\, |1 - \eta L|\big)\,\|\mathbf{x}_k - \mathbf{x}^\star\|$$
:::

::: {.slide title="Two-Phase PGD Dynamics on the κ = 1000 Valley"}
[Projected Gradient Descent]{.kicker}

Minimize $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ over the half-space $x_1 + x_2 \ge 1$ ($\mathbf{x}^\star = (\frac{1000}{1001}, \frac{1}{1001})$, $\lambda^\star = \frac{1000}{1001}$) using PGD with $\eta = 1.5 \times 10^{-3}$ from $(3, 4)$:

@!mdl-constrained-optimization-duality-projected-gradient-descent

- **Why $738$ steps instead of $10{,}200$?** For the first $732$ steps, the iterate drifts through the interior at the slow rate $1 - \eta = 0.9985$. Once it hits the boundary line $x_1 + x_2 = 1$ at step $733$, the restricted quadratic along the line has curvature $500.5$, contracting at $1 - 500.5\eta = 0.249$ per step!
:::

::: {.slide title="Projecting onto the Probability Simplex (Sparsemax)"}
[Simplex Projection]{.kicker}

::: {.cols .vc}
::: {.col}
To project a score vector $\mathbf{y} \in \mathbb{R}^n$ onto the probability simplex $\Delta = \{\mathbf{x} \succeq \mathbf{0} : \mathbf{1}^\top\mathbf{x} = 1\}$, solve $\min_{\mathbf{x} \in \Delta}\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$:

- **KKT stationarity & slackness:** With multiplier $\tau$ on $\sum_i x_i = 1$ and $\lambda_i \ge 0$ on $-x_i \le 0$, stationarity $x_i - y_i + \tau - \lambda_i = 0$ and $\lambda_i x_i = 0$ collapse to $x_i^\star = \max(y_i - \tau,\, 0)$, where $\tau$ is uniquely determined by the unit-sum budget:
  $$\sum_{i=1}^n \max(y_i - \tau,\, 0) = 1$$
- **Exact $O(n\log n)$ sort-and-threshold:** Sort $u_1 \ge \cdots \ge u_n$, form $\tau_k = \frac{1}{k}\big(\sum_{j=1}^k u_j - 1\big)$, and pick the largest $k$ with $u_k > \tau_k$.
:::

::: {.col .fig}
@fig:mdl-opt-simplex-threshold
:::
:::
:::

::: {.slide title="Certifying Simplex Projections via Vertex Tests"}
[Simplex Projection]{.kicker}

Because $x_i(\tau) = \max(y_i - \tau, 0)$ and $\lambda_i(\tau) = \max(0, \tau - y_i)$ satisfy stationarity, dual feasibility, and complementary slackness for *every* scalar $\tau$, checking those three KKT residuals alone can be fooled by an incorrect $\tau$:

@!constrained-simplex-projection

- Only the unit-sum budget $\sum_i x_i = 1$ and the obtuse-angle certificate tested at the $n$ simplex vertices $\mathbf{e}_i$, $\max_{1 \le i \le n}(\mathbf{y} - \mathbf{x}^\star)^\top(\mathbf{e}_i - \mathbf{x}^\star) \le 0$, certify that $\mathbf{x}^\star = \Pi_\Delta(\mathbf{y})$.
:::

::: {.slide title="Why Subgradient Descent Fails to Produce Sparsity"}
[Proximal Operators]{.kicker}

The **lasso** minimizes $F(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{w}\|_1$ ($\mathbf{A} \in \mathbb{R}^{100 \times 8}$, $5$ planted zero weights, $\lambda = 3$). What happens if we run subgradient descent $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\big(\mathbf{A}^\top(\mathbf{A}\mathbf{w}_k - \mathbf{b}) + \lambda\,\mathrm{sign}(\mathbf{w}_k)\big)$ at $\eta = 1/L$?

@!mdl-constrained-optimization-duality-soft-thresholding-and-the-proximal-operator-2

::: {.d2l-note .warn}
**Why subgradient steps chatter across zero:** At a planted zero coordinate, the smooth gradient has magnitude $< \lambda$, so the subgradient kick $\pm\eta\lambda$ overpowers it and flips the coordinate's sign on nearly every step, leaving $0$ exact zeros!
:::
:::

::: {.slide title="Soft-Thresholding and the Proximal Operator"}
[Proximal Operators]{.kicker}

::: {.cols .vc}
::: {.col}
The **proximal operator** minimizes a nonsmooth convex penalty $\psi(\mathbf{x})$ against a quadratic tether to $\mathbf{z}$:

$$\mathrm{prox}_\psi(\mathbf{z}) = \mathop{\mathrm{argmin}}_{\mathbf{x} \in \mathbb{R}^d}\left(\psi(\mathbf{x}) + \tfrac12\|\mathbf{x} - \mathbf{z}\|_2^2\right)$$

- **1D soft-thresholding ($\psi(x) = \lambda|x|$):** By Fermat's rule $0 \in \lambda\partial|x| + (x - z)$, $\mathrm{prox}_{\lambda|\cdot|}(z) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$, with an exact zero **dead zone** for $|z| \le \lambda$.
:::

::: {.col .fig}
@fig:ch35-opt-soft-thresholding-prox
:::
:::
:::

::: {.slide title="Proximal Gradient Descent (ISTA) Recovers Exact Zeros"}
[Proximal Operators]{.kicker}

For composite objectives $F(\mathbf{w}) = s(\mathbf{w}) + \psi(\mathbf{w})$ where $s$ is convex and $L$-smooth and $\psi$ is proper closed convex, **proximal gradient descent** minimizes the $L$-smoothness quadratic upper bound of $s$ plus the *exact* penalty $\psi$:

$$\mathbf{w}_{k+1} = \mathrm{prox}_{\eta\psi}\big(\mathbf{w}_k - \eta\,\nabla s(\mathbf{w}_k)\big)$$

@!convexity-prox-ista

- When $\psi(\mathbf{w}) = \lambda\|\mathbf{w}\|_1$, this update is **ISTA** (Iterative Shrinkage-Thresholding Algorithm): its dead zone $[-\eta\lambda, \eta\lambda]$ absorbs the exact chattering amplitude of subgradient descent, recovering all $5$ planted zeros cleanly.
:::

::: {.slide title="Cyclic Coordinate Descent on the Lasso"}
[Coordinate Descent]{.kicker}

::: {.cols .vc}
::: {.col}
Because $\|\mathbf{w}\|_1 = \sum_{j=1}^d |w_j|$ separates across coordinates, minimizing the lasso along a single weight $w_j$ (with partial residual $\mathbf{r}_{-j} = \mathbf{b} - \sum_{i \neq j}\mathbf{a}_i w_i$ and correlation $c_j = \mathbf{a}_j^\top\mathbf{r}_{-j}$) has an exact closed form:

$$w_j \leftarrow \frac{\mathrm{sign}(c_j)\,\max(|c_j| - \lambda,\, 0)}{\|\mathbf{a}_j\|_2^2}$$

- **Why coordinate sweeps beat ISTA:** Each coordinate step uses its own single-column curvature $L_j = \|\mathbf{a}_j\|_2^2 \approx 100$ rather than the global eigenvalue ceiling $L = \lambda_{\max}(\mathbf{A}^\top\mathbf{A}) \approx 160$, reaching $10^{-8}$ in $4$–$6$ passes vs. $\sim 20$ for ISTA.
:::

::: {.col .fig}
@!mdl-convexity-coordinate-descent
:::
:::
:::

::: {.slide title="Concept Check: KKT, Simplex Projection, and Prox"}
[Concept Check]{.kicker}

Work out these three calculations using the KKT and proximal rules:

1. We project $\mathbf{y}_1 = (1.5, 2.0)$ and $\mathbf{y}_2 = (0.3, 0.4)$ onto the unit disk $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|_2^2 - 1) \le 0$. What are the optimal point $\mathbf{x}^\star$ and KKT multiplier $\lambda^\star$ for each?
2. Project the score vector $\mathbf{y} = (0.9, 0.5, -0.2)$ onto the probability simplex $\Delta$: which coordinates survive ($k$), what is the threshold $\tau$, and what is $\Pi_\Delta(\mathbf{y})$?
3. Evaluate the soft-thresholding operator $\mathrm{prox}_{\lambda|\cdot|}(z)$ with $\lambda = 0.7$ at $z = 0.5$ and at $z = -2.0$.

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** $\|\mathbf{y}_1\|_2 = 2.5 > 1$ (active): $\lambda^\star = 2.5 - 1 = \mathbf{1.5}$ and $\mathbf{x}^\star = \mathbf{y}_1 / 2.5 = \mathbf{(0.6, 0.8)}$. $\|\mathbf{y}_2\|_2 = 0.5 < 1$ (inactive): $\lambda^\star = \mathbf{0}$ and $\mathbf{x}^\star = \mathbf{(0.3, 0.4)}$. **(2)** Candidate thresholds $\tau_k = (\sum_{j \le k} y_j - 1)/k$: $\tau_1 = -0.1 < 0.9$ (valid), $\tau_2 = (1.4 - 1)/2 = 0.2 < 0.5$ (valid), $\tau_3 = 0.2/3 \approx 0.067 \not< -0.2$ (invalid). Thus $k = 2$, $\tau = \mathbf{0.2}$, and $\Pi_\Delta(\mathbf{y}) = \mathbf{(0.7, 0.3, 0.0)}$. **(3)** At $z = 0.5$, $|0.5| \le 0.7$ falls inside the dead zone so $\mathrm{prox}_{0.7|\cdot|}(0.5) = \mathbf{0}$. At $z = -2.0$, $\mathrm{prox}_{0.7|\cdot|}(-2.0) = -(2.0 - 0.7) = \mathbf{-1.3}$.
:::
:::

::: {.slide title="Summary: Constrained Optimization and Operators"}
[Key Takeaways]{.kicker}

| Problem / Method | Mathematical Condition / Update | Optimality / Rate Guarantee | Deep Learning & ML Role |
| :--- | :--- | :--- | :--- |
| **Equality Lagrange** | $\nabla f(\mathbf{x}^\star) + \sum_j \nu_j^\star \nabla h_j(\mathbf{x}^\star) = \mathbf{0}$, $h_j = 0$ | Necessary under LICQ; $\nu_j^\star = -\partial p^\star/\partial b_j$ | Hyperplane projections, normalization constraints |
| **Inequality KKT** | Stationarity, primal/dual feas. ($\lambda_i^\star \ge 0$), $\lambda_i^\star g_i(\mathbf{x}^\star) = 0$ | Sufficient for convex problems; necessary under Slater | Gradient clipping, SVMs, norm-budget regularization |
| **Euclidean Projection $\Pi_C$** | $(\mathbf{y} - \Pi_C(\mathbf{y}))^\top(\mathbf{x} - \Pi_C(\mathbf{y})) \le 0\;\forall\mathbf{x}\in C$ | Unique and nonexpansive ($\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\| \le \|\mathbf{x}-\mathbf{y}\|$) | PGD $\mathbf{x}_{k+1} = \Pi_C(\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k))$ contracts at least as fast as GD (same $\rho$) |
| **Simplex Projection** | $x_i^\star = \max(y_i - \tau, 0)$ with $\sum_i x_i^\star = 1$ | Exact $O(n\log n)$ sort-and-threshold | `sparsemax` attention weights with exact zero entries |
| **Proximal Gradient (ISTA)** | $\mathbf{w}_{k+1} = \mathrm{prox}_{\eta\psi}(\mathbf{w}_k - \eta\nabla s(\mathbf{w}_k))$ | $O(1/k)$ convex, linear $(1-\mu/L)^k$ strongly convex | Soft-thresholding $\mathrm{prox}_{\lambda\|\cdot\|_1}$ creates exact weight sparsity |
:::

