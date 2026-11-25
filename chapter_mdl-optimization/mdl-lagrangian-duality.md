# Duality
:label:`sec_mdl-lagrangian-duality`

When an iterative solver stops making progress and reports convergence, a
natural question remains: is the point it returned actually optimal? For an
unconstrained convex function, a zero gradient settles the question at once
(:numref:`sec_mdl-convexity`). With constraints, however, the optimum usually
sits on the boundary with a nonzero gradient
(:numref:`sec_mdl-constrained-optimization-duality`), so that simple test
disappears. Both numerical solvers and mathematical proofs therefore need a
different tool: a reliable lower bound on the optimal value. In this section we
develop a systematic way to construct such bounds and uncover what the Lagrange
multipliers of the KKT conditions actually measure.

To see the difficulty concretely, return to the clipping problem of
:numref:`subsec_mdl-kkt-conditions`. There we minimized the squared distance
$\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ to the target $\mathbf{x}_0 = (3, 4)$
over the unit disk $\|\mathbf{x}\| \le 1$, finding the optimum
$\mathbf{x}^\star = (0.6, 0.8)$ with optimal value $8$. (We write
$\mathbf{x}_0$ here instead of $\mathbf{y}$, reserving $\mathbf{y}$ for labels
and for the argument of the convex conjugate later.) Now imagine an iterative
solver stops at $\mathbf{x} = (0.55, 0.835)$. This point is feasible, since
$\|\mathbf{x}\|^2 = 0.9997 \le 1$, and achieves an objective value of $8.01$.
How can we tell whether $8.01$ is nearly optimal, or whether some other point
in the disk reaches $7$? The gradient at the true optimum is $(-2.4, -3.2)$,
nowhere near zero, and the disk contains infinitely many points, so checking
them one by one is out of the question.

The key idea is to fold the constraint into the objective. Add a nonnegative
multiple $\lambda \ge 0$ of the constraint function
$\tfrac12(\|\mathbf{x}\|^2 - 1)$ to the objective. At every feasible point this
added term is nonpositive, so minimizing the modified objective over *all* of
$\mathbb{R}^2$ must yield a value no larger than the constrained optimum. Each
choice of $\lambda \ge 0$ thus produces a lower bound through an unconstrained
minimization, and $\lambda = 4$ gives a bound of $8$—proving immediately that
the solver's $8.01$ is within $0.01$ of optimal. Searching for the tightest such
bound defines a companion optimization problem, the *dual problem*. Remarkably,
the dual problem is always convex, even when the original problem is not. For a
convex problem with a strictly feasible point, its optimal value matches the
constrained optimum, and its optimal multipliers tell us how rapidly the
optimal value improves when each constraint is relaxed.

In what follows, we first use this bounding idea to prove optimality for the
clipping problem and state *weak duality*, a two-line argument that holds for
every optimization problem. We then establish *strong duality* for convex
problems, whose proof contains the one subtle geometric step in the section,
and interpret the optimal multipliers as *shadow prices*, connecting weight
decay directly to norm constraints. After introducing the *convex conjugate* as
a convenient tool for computing duals of linearly constrained problems (which
can be skimmed on a first reading), we work through three revealing examples:
the support vector machine, water-filling power allocation, and a nonconvex
problem whose best dual bound stays strictly below its optimum. All
computations are small enough for NumPy, and `d2l` handles the plots.

```{.python .input #mdl-lagrangian-duality-duality}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Dual Problem
:label:`subsec_mdl-dual-problem`

Our first goal is to prove that no feasible point of the clipping problem
achieves a value below $8$, without checking feasible points one by one. Every
lower bound in this section rests on a single elementary observation: at any
feasible point, every inequality constraint function is nonpositive.

### A Lower Bound from One Multiplier

Recall the clipping problem: minimize
$f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ with
$\mathbf{x}_0 = (3, 4)$, subject to
$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$. Including the factor of
$\tfrac12$ in $g$ simply makes its gradient equal to $\mathbf{x}$. From the KKT
conditions we already know the solution:
$\mathbf{x}^\star = \mathbf{x}_0 / \|\mathbf{x}_0\| = (0.6, 0.8)$ with optimal
value $p^\star = \tfrac12(5 - 1)^2 = 8$, and optimal multiplier
$\lambda^\star = 4$, determined by the stationarity condition
$\nabla f(\mathbf{x}^\star) + \lambda^\star \mathbf{x}^\star = (-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$.

Now pick any multiplier $\lambda \ge 0$ and any feasible point $\mathbf{x}$.
Because $g(\mathbf{x}) \le 0$ and $\lambda \ge 0$, their product
$\lambda g(\mathbf{x})$ is nonpositive, which means
$f(\mathbf{x}) + \lambda g(\mathbf{x}) \le f(\mathbf{x})$. Minimizing the
left-hand side over all of $\mathbb{R}^2$ can only make it smaller still, so
that unconstrained minimum is at most $f(\mathbf{x})$. Since this holds for
every feasible $\mathbf{x}$, it also holds for the best feasible $\mathbf{x}$:

$$
\min_{\mathbf{x} \in \mathbb{R}^2} \bigl[ f(\mathbf{x}) + \lambda g(\mathbf{x}) \bigr]
\;\le\; \min_{\|\mathbf{x}\| \le 1} f(\mathbf{x}) \;=\; p^\star
\qquad \textrm{for every } \lambda \ge 0 .
$$

Let us evaluate the left-hand side explicitly. The function being minimized is
the Lagrangian
$\mathcal{L}(\mathbf{x}, \lambda) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \tfrac\lambda2(\|\mathbf{x}\|^2 - 1)$.
For any $\lambda \ge 0$, this is a convex quadratic in $\mathbf{x}$ with
Hessian $(1 + \lambda)\mathbf{I}$, so its unique minimizer occurs where the
gradient $(\mathbf{x} - \mathbf{x}_0) + \lambda \mathbf{x}$ vanishes:
$\mathbf{x}(\lambda) = \mathbf{x}_0 / (1 + \lambda)$. Increasing $\lambda$
shrinks $\mathbf{x}_0$ more strongly toward the origin. Substituting
$\mathbf{x}(\lambda)$ back into the Lagrangian—using
$\mathbf{x}(\lambda) - \mathbf{x}_0 = -\lambda\mathbf{x}_0 / (1 + \lambda)$ and
$\|\mathbf{x}_0\|^2 = 25$—gives
$\tfrac12\|\mathbf{x}(\lambda) - \mathbf{x}_0\|^2 = \tfrac{25}{2}\lambda^2/(1+\lambda)^2$
and $\tfrac\lambda2\|\mathbf{x}(\lambda)\|^2 = \tfrac{25}{2}\lambda/(1+\lambda)^2$.
Combining these two fractions cancels a factor of $(1 + \lambda)$, leaving the
minimum value as a simple function of $\lambda$:

$$
q(\lambda) \;=\; \frac{25}{2}\,\frac{\lambda}{1 + \lambda} \;-\; \frac{\lambda}{2}.
$$
:eqlabel:`eq_mdl-opt-ball-dual`

The two terms pull in opposite directions. As $\lambda$ grows, the first term
rises toward $12.5$, while the second term—coming from the constant $-\tfrac12$
inside $g$—drops linearly without bound. The tightest lower bound sits where
these two effects balance.

Plugging in a few numbers makes this balance concrete: $q(0) = 0$,
$q(1) = 5.75$, $q(2) = 7.33$, $q(3) = 7.875$, $q(4) = 8$, $q(5) = 7.92$, and
$q(8) = 7.11$. Every single one of these values is a valid lower bound on
$p^\star$. Differentiating gives
$q'(\lambda) = \tfrac{25}{2(1 + \lambda)^2} - \tfrac12$, which vanishes when
$(1 + \lambda)^2 = 25$, or $\lambda = 4$. Since
$q''(\lambda) = -25/(1 + \lambda)^3 < 0$, $\lambda = 4$ is the unique maximum,
attaining $q(4) = 10 - 2 = 8 = p^\star$. The best lower bound equals the true
optimum, and the multiplier that achieves it is the KKT multiplier. This
certifies that the solver's value of $8.01$ is within $0.01$ of optimal; even
an unoptimized guess like $\lambda = 3$ already certifies that it is within
$0.135$.

Why does $\lambda = 4$ hit the optimum? Look at the constraint value at the
Lagrangian minimizer:
$g(\mathbf{x}(\lambda)) = \tfrac12\bigl(25/(1 + \lambda)^2 - 1\bigr)$. When
$\lambda < 4$, the multiplier is too weak to pull the minimizer inside the
unit disk (for example, $\mathbf{x}(1) = (1.5, 2)$ has norm $2.5$), so
$g(\mathbf{x}(\lambda)) > 0$ and the bound reflects an infeasible point. When
$\lambda > 4$, the multiplier pulls the minimizer too far inside (for example,
$\mathbf{x}(8)$ has norm $\tfrac59$), so $\lambda g(\mathbf{x}(\lambda)) < 0$
drags the bound back down. Right at $\lambda = 4$, the minimizer
$\mathbf{x}(4) = (0.6, 0.8)$ lands on the boundary circle, the penalty term
$\lambda g(\mathbf{x}(\lambda))$ vanishes, and the bound coincides with the
objective of a feasible point.

Let us plot $q(\lambda)$ alongside $p^\star$ across a grid of $\lambda$ values
to see how these bounds behave.

```{.python .input #mdl-lagrangian-duality-a-lower-bound-from-one-multiplier}
x0 = np.array([3.0, 4.0])                             # the point to clip
f = lambda x: 0.5 * np.sum((x - x0) ** 2)             # objective
g = lambda x: 0.5 * (np.sum(x ** 2) - 1.0)            # constraint g(x) <= 0
x_of = lambda lam: x0 / (1 + lam)                     # minimizer of L(., lam)
q = lambda lam: f(x_of(lam)) + lam * g(x_of(lam))     # dual function
p_star = f(x0 / np.linalg.norm(x0))                   # value at the clipped point
lams = np.linspace(0.0, 12.0, 1201)
qs = np.array([q(l) for l in lams])
d2l.plot(lams, [qs, np.full_like(lams, p_star)], 'lambda', 'value',
         legend=['dual function q(lambda)', 'p* = 8'])
```

The resulting curve is concave and stays everywhere below the horizontal line
$p^\star = 8$, touching it once at $\lambda = 4$. Surprisingly, concavity here
has nothing to do with the geometry of the disk: as we now show, the dual
function is concave for *every* optimization problem.

### The Lagrange Dual Function

Nothing in the bounding argument above depended on the clipping problem. Let us
apply the same construction to the general constrained problem from
:numref:`subsec_mdl-kkt-conditions`,

$$
\min_{\mathbf{x} \in \mathcal{D}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad
g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m, \qquad h_j(\mathbf{x}) = 0,\; j = 1, \ldots, p,
$$

with optimal value $p^\star$, defined as the infimum of $f$ over all feasible
points (with $p^\star = +\infty$ if no feasible point exists). Here
$\mathcal{D}$ is the domain on which $f$ and all constraint functions are
defined; it is simply $\mathbb{R}^n$ unless we keep a set as a hard constraint,
as we will do with $[0, 1]$ in the duality-gap example at the end of this
section. The Lagrangian introduces one multiplier per constraint,

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}) + \sum_{i=1}^m \lambda_i\, g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j\, h_j(\mathbf{x}),
$$

with $\lambda_i \ge 0$ for inequalities and $\nu_j \in \mathbb{R}$ unconstrained
in sign for equalities. We write $\boldsymbol{\lambda} \succeq 0$ to mean
$\lambda_i \ge 0$ for every coordinate $i$.

Minimizing the Lagrangian over $\mathbf{x}$ for fixed multipliers turns our
lower-bound construction into a function of $(\boldsymbol{\lambda}, \boldsymbol{\nu})$.
The **Lagrange dual function** is

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}}\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$
:eqlabel:`eq_mdl-opt-dual-function`

In words, $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ asks how low $f$ can go
once we drop the hard constraints and replace them with linear penalties
weighted by $\boldsymbol{\lambda}$ and $\boldsymbol{\nu}$. For the clipping
problem this gave :eqref:`eq_mdl-opt-ball-dual`. In general, the infimum need
not be attained, and it can even be $-\infty$. For example, suppose we minimize
$x$ subject to $-x \le 0$: the Lagrangian $(1 - \lambda)x$ is unbounded below
unless $\lambda = 1$, so $q(1) = 0$ and $q(\lambda) = -\infty$ for all other
$\lambda$. A bound of $-\infty$ is trivially valid but useless. The set of
multipliers where $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty$ is
called the *effective domain* of $q$; when we derive the support vector machine
dual below, this implicit domain restriction will surface as an explicit
equality constraint.

Two fundamental properties of $q$ hold with no assumptions whatsoever on $f$,
$g_i$, or $h_j$—not even convexity or differentiability.

**Proposition (the dual function is concave).** *For any primal problem,
$q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ in :eqref:`eq_mdl-opt-dual-function`
is concave.*

**Proof.** Fix any $\mathbf{x} \in \mathcal{D}$. Because the multipliers enter
$\mathcal{L}$ linearly, the map
$(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$
is affine. The dual function $q$ is the pointwise infimum of this family of
affine functions. Its **hypograph**—the set of points lying on or below its
graph—is therefore the intersection of the half-spaces below these affine
functions. Since any intersection of convex sets is convex, $q$ is concave.
$\blacksquare$

:numref:`fig_mdl-opt-dual-envelope` draws this construction for the clipping
problem. Each dashed line is the Lagrangian of one fixed $\mathbf{x}$ as a
function of $\lambda$, and $q$ is the curve that stays below all of them. The
objective of the clipping problem happens to be convex, but the argument never
used that: a lower envelope of lines is concave whatever shape $f$ and $g$
have.

![The dual function of the clipping problem as a lower envelope. Each fixed $\mathbf{x}$ contributes one line $\lambda \mapsto \mathcal{L}(\mathbf{x}, \lambda)$ (dashed), and $q(\lambda)$ is the lowest value that any of these lines takes at $\lambda$. The lines drawn belong to the minimizers $\mathbf{x}(\lambda) = \mathbf{x}_0/(1+\lambda)$ for several values of $\lambda$, and each touches $q$ at its own $\lambda$ (dots). The line of $\mathbf{x}^\star = \mathbf{x}(4)$ is flat because $g(\mathbf{x}^\star) = 0$; it touches $q$ at the maximum $d^\star = q(4) = p^\star = 8$.](../img/mdl-opt-dual-envelope.svg)
:label:`fig_mdl-opt-dual-envelope`

**Proposition (weak duality).** *If $\mathbf{x}$ is feasible and
$\boldsymbol{\lambda} \succeq 0$, then
$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$. Consequently,*

$$
d^\star \;=\; \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \inf_{\mathbf{x}\ \textrm{feasible}} f(\mathbf{x}) \;=\; p^\star.
$$
:eqlabel:`eq_mdl-opt-weak-duality`

**Proof.** For any feasible $\mathbf{x}$ and any $\boldsymbol{\lambda} \succeq 0$,
we have $\lambda_i g_i(\mathbf{x}) \le 0$ (since $\lambda_i \ge 0$ and
$g_i(\mathbf{x}) \le 0$) and $\nu_j h_j(\mathbf{x}) = 0$. Therefore,

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;=\; f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})
\;\le\; f(\mathbf{x}).
$$

Taking the supremum over $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ with
$\boldsymbol{\lambda} \succeq 0$ on the left and the infimum over feasible
$\mathbf{x}$ on the right yields :eqref:`eq_mdl-opt-weak-duality`. $\blacksquare$

Weak duality is the two-line inequality that turned every $q(\lambda)$ in the
clipping problem into a certified lower bound. It is also the exact engine
behind KKT sufficiency in :numref:`subsec_mdl-kkt-conditions`, where evaluating
the same chain at the KKT multipliers makes both inequalities tight via
complementary slackness.

Maximizing $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ over
$\boldsymbol{\lambda} \succeq 0$ and $\boldsymbol{\nu}$ is called the **dual
problem**. Because it maximizes a concave function over a convex set, the dual
problem is *always* a convex optimization problem, even when the primal problem
is badly nonconvex. Its optimal value $d^\star$ is the tightest lower bound
this construction can provide, and every dual-feasible
$(\boldsymbol{\lambda}, \boldsymbol{\nu})$ gives a valid lower bound. The
nonnegative difference $p^\star - d^\star \ge 0$ is the **duality gap**. For
the clipping problem the gap was zero, whereas our final worked example will
exhibit a strict gap. The natural next question is when we can guarantee in
advance that the gap vanishes.

## Strong Duality
:label:`subsec_mdl-strong-duality`

In the clipping problem, the best dual bound matched the primal optimum
exactly. For a nonconvex problem, however, the dual bound can fall strictly
short, certifying only part of the answer. Fortunately, two mild ingredients—
convexity and a single strictly feasible point—are enough to close the gap. A
simple geometric view of the attainable constraint and objective values shows
why both are needed.

### Slater's Condition

**Proposition (Slater's condition implies strong duality).** *Suppose the
problem is convex ($f$ and every $g_i$ convex, every $h_j$ affine) and
satisfies Slater's condition (:numref:`subsec_mdl-constraint-qualifications`): there exists a
feasible point $\bar{\mathbf{x}}$ in the relative interior of $\mathcal{D}$
such that $g_i(\bar{\mathbf{x}}) < 0$ for every inequality constraint $g_i$
that is not affine. Then $d^\star = p^\star$, and whenever $p^\star$ is finite,
the dual optimum is attained.*

This is the same constraint qualification introduced for convex problems in
:numref:`subsec_mdl-kkt-conditions`: at least one point must sit strictly
inside all curved inequality constraints. (See
:citet:`Boyd.Vandenberghe.2004`, section 5.3.2 for the full proof in multiple
dimensions.) The clipping problem satisfies Slater's condition effortlessly:
both $f$ and $g$ are convex, and the origin $\mathbf{0}$ satisfies
$g(\mathbf{0}) = -\tfrac12 < 0$. Thus $q(4) = p^\star$ was guaranteed before we
ever computed a derivative.

To understand *why* convexity and strict feasibility close the gap, consider a
problem with a single inequality constraint $g(\mathbf{x}) \le 0$. Map every
point $\mathbf{x} \in \mathcal{D}$ to the pair $(g(\mathbf{x}), f(\mathbf{x}))$
of its constraint value and objective value, and collect all points lying above
and to the right of these pairs:

$$
\mathcal{A} = \bigl\{ (u, t) : u \ge g(\mathbf{x}),\ t \ge f(\mathbf{x})
\ \textrm{for some } \mathbf{x} \in \mathcal{D} \bigr\}.
$$

In this $(u, t)$-plane, feasible points correspond to $u \le 0$, so the primal
optimum $p^\star$ is simply the lowest height $t$ that $\mathcal{A}$ reaches on
the vertical axis $u = 0$. Now fix $\lambda \ge 0$. Over all $(u, t)$
contributed by a given $\mathbf{x}$, the linear function $t + \lambda u$ is
minimized at $(g(\mathbf{x}), f(\mathbf{x}))$, so
$q(\lambda) = \inf_{(u, t) \in \mathcal{A}} (t + \lambda u)$. Geometrically, we
take a line $t + \lambda u = c$ of slope $-\lambda$ and push it upward from
below until it just supports $\mathcal{A}$; its intercept on the vertical axis
$u = 0$ is $q(\lambda)$. Because the entire set $\mathcal{A}$ lies on or above
this supporting line, the intercept $q(\lambda)$ can never exceed $p^\star$.
This is weak duality made visible (:numref:`fig_mdl-opt-primal-dual-gap`).

![The supporting-line geometry of duality. Every $\mathbf{x}$ maps to $(g(\mathbf{x}),f(\mathbf{x}))$; the shaded set $\mathcal{A}$ contains these pairs and all points above and to the right of them. Evaluating the dual at $\lambda$ pushes a line of slope $-\lambda$ up until it supports $\mathcal{A}$ from below, and its intercept at $u=0$ is $q(\lambda)$. (a) The clipping problem: $\mathcal{A}$ is convex and $\mathbf{0}$ is strictly feasible, so the optimal supporting line touches at $(0,p^\star)$ with slope $-4$, giving $d^\star=p^\star=8$. (b) Minimizing $-x^2$ on $[0,1]$ subject to $x\le\tfrac12$: $\mathcal{A}$ is nonconvex, so the highest supporting line passes strictly below $(0,p^\star)$, leaving $d^\star=-\tfrac12<-\tfrac14=p^\star$.](../img/mdl-opt-primal-dual-gap.svg)
:label:`fig_mdl-opt-primal-dual-gap`

With this geometry in place, the proof of strong duality for a single
inequality constraint becomes transparent.

**Proof (one inequality constraint with $g(\bar{\mathbf{x}}) < 0$, $p^\star$ finite).**
First, $\mathcal{A}$ is convex. Take any two points $(u_1, t_1)$ and
$(u_2, t_2)$ in $\mathcal{A}$, contributed by $\mathbf{x}_1$ and $\mathbf{x}_2$,
and any $\theta \in [0, 1]$. By convexity of $g$, the midpoint
$\mathbf{x}_\theta = \theta\mathbf{x}_1 + (1 - \theta)\mathbf{x}_2$ satisfies
$g(\mathbf{x}_\theta) \le \theta g(\mathbf{x}_1) + (1 - \theta) g(\mathbf{x}_2) \le \theta u_1 + (1 - \theta) u_2$,
and the same inequality holds for $f$. Hence the convex combination of
$(u_1, t_1)$ and $(u_2, t_2)$ belongs to $\mathcal{A}$.

Second, $(0, p^\star)$ lies on the boundary of $\mathcal{A}$: it is a limit of
points in $\mathcal{A}$, whereas $(0, p^\star - \varepsilon) \notin \mathcal{A}$
for every $\varepsilon > 0$. Every boundary point of a convex set admits a
supporting line: there exists a nonzero vector $(\lambda, \gamma) \neq (0, 0)$
such that $\gamma t + \lambda u \ge \gamma p^\star$ for all
$(u, t) \in \mathcal{A}$ (:citet:`Boyd.Vandenberghe.2004`, section 2.5.2; this
is the same geometric fact that produces subgradients in
:numref:`sec_mdl-convexity`). Because $\mathcal{A}$ extends indefinitely upward
and to the right, we must have $\gamma \ge 0$ and $\lambda \ge 0$.

Could $\gamma = 0$? If so, the supporting line is vertical and $\lambda > 0$,
forcing $\lambda u \ge 0$ (and thus $u \ge 0$) throughout $\mathcal{A}$. That
contradicts the strictly feasible point $\bar{\mathbf{x}}$, which contributes
$u = g(\bar{\mathbf{x}}) < 0$. Therefore $\gamma > 0$. Dividing by $\gamma$
yields $t + (\lambda/\gamma)\,u \ge p^\star$ for all $(u, t) \in \mathcal{A}$,
which means $q(\lambda/\gamma) \ge p^\star$. Combined with weak duality,
$q(\lambda/\gamma) = p^\star$: the dual optimum is attained and equals
$p^\star$. $\blacksquare$

Notice how cleanly the two hypotheses separate in the proof. Convexity is used
once, to make $\mathcal{A}$ convex so that a supporting line exists at
$(0, p^\star)$. Slater's condition is used once, to prevent that supporting
line from being vertical, which would correspond to an infinite multiplier. In
panel (a) of :numref:`fig_mdl-opt-primal-dual-gap`, for the clipping problem,
a point of norm $\|\mathbf{x}\| = r$ has minimum objective $\tfrac12(5 - r)^2$
and constraint value $u = \tfrac12(r^2 - 1)$, so the lower boundary of
$\mathcal{A}$ is the curve $t = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$. Its
slope at $u = 0$ is $-4$, matching the optimal supporting line of slope
$-\lambda^\star = -4$. Panel (b) previews the nonconvex example at the end of
this section: there $\mathcal{A}$ bends the wrong way, so every line below
$\mathcal{A}$ passes strictly underneath $(0, p^\star)$.

### Saddle Points

Strong duality can also be restated purely in terms of the Lagrangian, without
even mentioning the dual function. Observe that we can rewrite the primal
problem as a min--max problem: for any $\mathbf{x} \in \mathcal{D}$,

$$
\sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= \begin{cases} f(\mathbf{x}) & \textrm{if } \mathbf{x} \textrm{ is feasible}, \\ +\infty & \textrm{otherwise}. \end{cases}
$$

If $\mathbf{x}$ violates any constraint, we can scale the corresponding
multiplier to drive $\mathcal{L}$ to $+\infty$; if $\mathbf{x}$ is feasible,
the best the multipliers can do is set $\lambda_i = 0$ on every slack
constraint, leaving $f(\mathbf{x})$. Consequently,
$p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \mathcal{L}$,
while by definition
$d^\star = \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}$.
Weak duality is simply the universal inequality $\sup \inf \le \inf \sup$,
which holds for any function of two variables. Strong duality says that we can
swap the order of minimization and maximization without changing the value.
When both optima are attained, this swapability is equivalent to the existence
of a saddle point of the Lagrangian.

**Proposition (strong duality is a saddle point).** *Suppose strong duality
holds and both optima are attained, at a primal optimum $\mathbf{x}^\star$ and
a dual optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Then
$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ is a*
**saddle point** *of the Lagrangian: for all $\mathbf{x} \in \mathcal{D}$, all
$\boldsymbol{\lambda} \succeq 0$, and all $\boldsymbol{\nu}$,*

$$
\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star).
$$
:eqlabel:`eq_mdl-opt-saddle-point`

*Conversely, any saddle point of $\mathcal{L}$ over
$\mathbf{x} \in \mathcal{D}$ and $(\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu})$
closes the duality gap, with saddle value $p^\star = d^\star$.*

**Proof.** By definition of $q$ and weak duality,

$$
d^\star
= q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; f(\mathbf{x}^\star)
= p^\star,
$$

where the final inequality holds because $\lambda_i^\star \ge 0$ and
$g_i(\mathbf{x}^\star) \le 0$ make every penalty term nonpositive. Strong
duality ($d^\star = p^\star$) forces every inequality in this chain to be an
equality. Equality on the left means $\mathbf{x}^\star$ minimizes
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$,
giving the right-hand inequality of :eqref:`eq_mdl-opt-saddle-point`. Equality
on the right forces $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$, which
is complementary slackness. Hence, for any $\boldsymbol{\lambda} \succeq 0$ and
$\boldsymbol{\nu}$,
$\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}^\star) + \sum_i \lambda_i g_i(\mathbf{x}^\star)
\le f(\mathbf{x}^\star)
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$,
giving the left-hand inequality of :eqref:`eq_mdl-opt-saddle-point`.

Conversely, if $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$
is a saddle point, then

$$
\begin{aligned}
\inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}
&\le \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \\
&= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\le \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L},
\end{aligned}
$$

which shows $p^\star \le d^\star$. Weak duality supplies the reverse
inequality, so $p^\star = d^\star$. $\blacksquare$

In our clipping example, the saddle point is
$(\mathbf{x}^\star, \lambda^\star) = ((0.6, 0.8), 4)$: perturbing $\mathbf{x}$
away from $\mathbf{x}^\star$ increases $\mathcal{L}(\cdot, 4)$, while varying
$\lambda$ leaves $\mathcal{L}(\mathbf{x}^\star, \lambda) = 8$ flat because
$g(\mathbf{x}^\star) = 0$.

Strong duality brings three practical benefits:

- *The dual can be simpler or smaller than the primal.* The support vector
  machine dual has one variable per data point with simple sign constraints and
  at most one equality; the ridge regression dual in
  Exercise :exref:`ex_mdl-lagrangian-duality-ridge-dual` reduces to a linear
  system of size equal to the number of examples rather than the feature
  dimension.
- *Every dual point gives a stopping criterion.* For any primal-feasible
  $\mathbf{x}$ and any dual-feasible $(\boldsymbol{\lambda}, \boldsymbol{\nu})$,
  the gap $f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ upper-bounds
  how far $f(\mathbf{x})$ can be from $p^\star$.
- *The dual solution helps recover the primal solution.* By
  :eqref:`eq_mdl-opt-saddle-point`, $\mathbf{x}^\star$ is *among* the
  minimizers of $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.
  When that minimizer is unique (for instance, when $f$ is strictly convex), it
  immediately gives $\mathbf{x}^\star$, as in the clipping problem where
  $\mathbf{x}(4) = \mathbf{x}_0 / 5 = \mathbf{x}^\star$. When the minimizer is
  not unique, feasibility and complementary slackness pick $\mathbf{x}^\star$
  out of the minimizer set. In a linear program,
  $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star)$ is flat in $\mathbf{x}$; in
  the SVM with an unpenalized offset below, minimizing the Lagrangian leaves
  the offset undetermined, and complementary slackness pins it down.

The saddle-point proposition also settles the necessity of the KKT conditions,
which :numref:`subsec_mdl-kkt-conditions` established as sufficient for convex
problems. Suppose a convex problem with differentiable $f$, $g_i$, and $h_j$ on
$\mathcal{D} = \mathbb{R}^n$ satisfies Slater's condition and attains its
optimum at $\mathbf{x}^\star$. Then strong duality holds with an attained dual
optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Because
$\mathbf{x}^\star$ minimizes
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ over
all of $\mathbb{R}^n$, its gradient $\nabla_{\mathbf{x}} \mathcal{L}$ vanishes
there, and as we just saw, complementary slackness holds as well. Combined with
primal and dual feasibility, these are precisely the four KKT conditions: under
Slater's condition, they are both necessary and sufficient.

Finally, the min--max structure appears even in deep learning problems that have
no explicit constraints. Generative adversarial networks
(:numref:`sec_basic_gan`) and adversarial training are formulated directly as
$\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}$ games
:cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014,Madry.Makelov.Schmidt.ea.2018`.
Without a saddle point, a gap can separate their $\min\max$ and $\max\min$
values. Even when a saddle point does exist, taking simultaneous gradient
descent steps in the minimizing variable and ascent steps in the maximizing
variable can fail to converge: for the simple bilinear objective $xy$, whose
saddle point is the origin, simultaneous descent--ascent spirals outward for
every positive step size
(Exercise :exref:`ex_mdl-lagrangian-duality-descent-ascent`).

## Multipliers as Prices
:label:`subsec_mdl-shadow-prices`

What is a constraint worth? In the clipping problem, if we slightly enlarge the
unit disk, how much closer to $\mathbf{x}_0$ can we get? The optimal Lagrange
multiplier answers that question directly. Even better, the same sensitivity
argument reveals that for a convex loss, weight decay and a norm constraint on
the weights trace out the exact same family of solutions.

### Shadow Prices

To measure the value of each constraint, perturb its right-hand side and track
how the optimal objective changes. For perturbation vectors
$\mathbf{u} \in \mathbb{R}^m$ and $\mathbf{v} \in \mathbb{R}^p$, define the
perturbed optimal value

$$
p^\star(\mathbf{u}, \mathbf{v}) = \inf\,\{ f(\mathbf{x}) : g_i(\mathbf{x}) \le u_i,\; h_j(\mathbf{x}) = v_j \}.
$$

At $(\mathbf{0}, \mathbf{0})$ we recover the original optimum
$p^\star(\mathbf{0}, \mathbf{0})$, while taking $u_i > 0$ relaxes the $i$-th
inequality constraint. We have already seen this function geometrically: for a
single inequality constraint, $p^\star(u)$ is the lowest height $t$ that the
set $\mathcal{A}$ in :numref:`fig_mdl-opt-primal-dual-gap` reaches at
horizontal coordinate $u$. The lower boundary of $\mathcal{A}$ is the graph of
$p^\star(u)$, and the supporting line of slope $-\lambda^\star$ lies everywhere
below it. The next proposition extends that geometric inequality to any number
of constraints.

**Proposition (multipliers are shadow prices).** *Suppose strong duality
holds with dual optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.
Then for all $\mathbf{u}$ and $\mathbf{v}$,*

$$
p^\star(\mathbf{u}, \mathbf{v}) \;\ge\; p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top} \mathbf{u} - \boldsymbol{\nu}^{\star\top} \mathbf{v},
$$
:eqlabel:`eq_mdl-opt-shadow-price`

*and if $p^\star$ is differentiable at $(\mathbf{0}, \mathbf{0})$, then
$\nabla_{\mathbf{u}} p^\star = -\boldsymbol{\lambda}^\star$ and
$\nabla_{\mathbf{v}} p^\star = -\boldsymbol{\nu}^\star$ there.*

**Proof.** Let $\mathbf{x}$ be any point feasible for the perturbed problem, so
$g_i(\mathbf{x}) \le u_i$ and $h_j(\mathbf{x}) = v_j$. Using strong duality,
the definition of $q$ as an infimum, $\lambda_i^\star \ge 0$, and the perturbed
constraints, we obtain

$$
\begin{aligned}
p^\star(\mathbf{0}, \mathbf{0}) &= q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\le f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x}) \\
&\le f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}.
\end{aligned}
$$

Taking the infimum over all perturbed-feasible $\mathbf{x}$ yields
:eqref:`eq_mdl-opt-shadow-price`. If $p^\star$ is differentiable at the origin,
the nonnegative function
$p^\star(\mathbf{u}, \mathbf{v}) - p^\star(\mathbf{0}, \mathbf{0}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$
attains its global minimum of $0$ at $(\mathbf{0}, \mathbf{0})$, so its
gradient must vanish there. $\blacksquare$

The optimal multiplier $\lambda_i^\star$ is called the **shadow price** of
constraint $i$: it measures the rate at which the optimal value improves as the
constraint is relaxed. Notice that :eqref:`eq_mdl-opt-shadow-price` is global,
holding for perturbations of *any* size. Relaxing constraint $i$ by $u_i > 0$
can lower the optimal value by at most $\lambda_i^\star u_i$, whereas
tightening it by the same amount raises the optimal value by at least
$\lambda_i^\star u_i$; to first order, when $p^\star$ is differentiable, the
change is $-\lambda_i^\star u_i$. A slack constraint has $\lambda_i^\star = 0$
by complementary slackness, matching intuition: loosening a constraint that is
not binding cannot improve the optimum.

Let us check this on the clipping problem. Relaxing $g(\mathbf{x}) \le 0$ to
$g(\mathbf{x}) \le u$ expands the disk to radius $r = \sqrt{1 + 2u}$, giving
$p^\star(u) = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$—the lower boundary of
$\mathcal{A}$ we computed earlier—whose derivative at $u = 0$ is indeed
$-4 = -\lambda^\star$. If we relax the constraint by $u = \tfrac12$ (a disk of
radius $\sqrt2$), the new optimum is
$p^\star(\tfrac12) = \tfrac12(5 - \sqrt2)^2 \approx 6.43$, safely above the
global lower bound $8 - 4 \cdot \tfrac12 = 6$ guaranteed by
:eqref:`eq_mdl-opt-shadow-price`.

One subtlety is worth keeping in mind: a multiplier depends on how the
constraint is scaled. Here $\lambda^\star = 4$ prices $u$, which is *half* the
squared radius, rather than the radius $r$ itself. Per unit of radius, the
chain rule gives $\mathrm{d}p^\star / \mathrm{d}r = -(5 - r)$, which equals $r$
times the derivative with respect to $u$ and matches $-\lambda^\star$ only
because $r = 1$. Rescaling a constraint rescales its multiplier accordingly:
writing $\|\mathbf{x}\|^2 - 1 \le 0$ without the factor of $\tfrac12$ would
halve the multiplier to $2$.

The closest-point problem of :numref:`subsec_mdl-closest-point` illustrates the
equality case. There, the minimum of $\tfrac12\|\mathbf{x}\|^2$ over the
hyperplane $\mathbf{a}^\top\mathbf{x} = b$ is
$p^\star(b) = b^2 / (2\|\mathbf{a}\|^2)$, and
$\partial p^\star / \partial b = -\nu^\star$ is simply
:eqref:`eq_mdl-opt-shadow-price` applied to the perturbation
$h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = v$, which shifts the
hyperplane to $b + v$. That explains the increase of $0.02 = 0.2 \cdot 0.1$
that we observed numerically in that section.

### Weight Decay and Norm Constraints
:label:`subsec_mdl-weight-decay-duality`

In deep learning, we regularly control model complexity either by adding weight
decay (:numref:`sec_weight_decay`) to a training loss $\ell(\mathbf{w})$ or by
imposing a hard norm constraint on the weights. Using the book's convention,
these two problems are

$$
\min_{\mathbf{w}}\; \ell(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2
\qquad \textrm{and} \qquad
\min_{\mathbf{w}}\; \ell(\mathbf{w}) \quad \textrm{subject to} \quad \tfrac12\|\mathbf{w}\|^2 \le \tfrac12 r^2 .
$$

At a stationary point of the penalized problem,
$\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$. For the constrained
problem, introducing a multiplier $\lambda \ge 0$ for
$g(\mathbf{w}) = \tfrac12\|\mathbf{w}\|^2 - \tfrac12 r^2$ produces the KKT
stationarity condition $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$:
the very same equation, with the weight-decay coefficient playing the role of
the Lagrange multiplier.

When the loss $\ell$ is *convex*, this equivalence works in both directions.
Going from penalty to constraint, let $\mathbf{w}_\lambda$ minimize the
penalized objective and choose the norm budget $r = \|\mathbf{w}_\lambda\|$.
Then $\mathbf{w}_\lambda$ is feasible, the constraint is active (satisfying
complementary slackness), and KKT stationarity holds automatically. Because the
KKT conditions are sufficient for convex problems
(:numref:`subsec_mdl-kkt-conditions`), $\mathbf{w}_\lambda$ solves the
constrained problem.

Going from constraint to penalty, take any budget $r > 0$. The origin
$\mathbf{w} = \mathbf{0}$ is strictly feasible, so Slater's condition holds. If
the constrained minimum is attained at $\mathbf{w}^\star$, KKT necessity yields
a multiplier $\lambda^\star(r) \ge 0$, and $\mathbf{w}^\star$ minimizes the
convex penalized objective $\ell(\mathbf{w}) + \tfrac{\lambda^\star}{2}\|\mathbf{w}\|^2$.
Whenever $\lambda^\star > 0$, the quadratic term makes this objective strictly
convex, so $\mathbf{w}^\star$ is its unique minimizer; when the budget is
slack, $\lambda^\star = 0$ and the penalized problem reduces to unregularized
minimization. By the shadow-price proposition,
$\lambda^\star(r) = -\partial p^\star / \partial(\tfrac12 r^2)$: the
weight-decay coefficient is the marginal reduction in training loss per unit of
half squared norm.

In fact, neither direction depends on the $\ell_2$ norm specifically. Replace
$\tfrac12\|\mathbf{w}\|^2$ by any convex penalty $\Omega(\mathbf{w})$, such as
the $\ell_1$ norm $\|\mathbf{w}\|_1$ used in the lasso
(:numref:`subsec_mdl-proximal-operators`). Any minimizer $\mathbf{w}_\lambda$
of $\ell + \lambda\Omega$ solves the constrained problem with budget
$\Omega(\mathbf{w}) \le \Omega(\mathbf{w}_\lambda)$, since every $\mathbf{w}$
within that budget satisfies
$\ell(\mathbf{w}) \ge \ell(\mathbf{w}_\lambda) + \lambda\,(\Omega(\mathbf{w}_\lambda) - \Omega(\mathbf{w})) \ge \ell(\mathbf{w}_\lambda)$.
Conversely, for convex $\ell$ and any budget strictly above $\inf \Omega$,
Slater's condition holds, and the saddle-point proposition ensures that any
constrained minimizer also minimizes $\ell + \lambda^\star\Omega$.

Convexity of $\ell$, however, is essential for the full equivalence. For a
nonconvex loss, a penalized stationary point is still a KKT point of the
matching constrained problem, but the global solution sets can differ, and
certain norm budgets may be unattainable by *any* penalty weight $\lambda$
(Exercise :exref:`ex_mdl-lagrangian-duality-penalty-nonconvex`).

It is worth noting a third interpretation of the same quadratic term. For a
squared loss with noise variance $\sigma^2$ and an isotropic Gaussian prior
$\mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ on the weights, the penalty
$\tfrac\lambda2\|\mathbf{w}\|^2$ with $\lambda = \sigma^2/\tau^2$ is the
negative log-prior up to a constant and overall scale
(:numref:`sec_mdl-maximum_likelihood`). Thus a single number $\lambda$ can be
read three ways: as a weight-decay coefficient, as a Lagrange multiplier for a
norm budget, and as a ratio of noise to prior variance.

Let us verify this equivalence numerically. For the convex least-squares loss
$\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2$ on random
data, we solve the penalized problem in closed form via
$(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$,
set $r = \|\mathbf{w}_\lambda\|$, and then solve the constrained problem using
projected gradient descent with step size $1/\lambda_{\max}(\mathbf{A}^\top\mathbf{A})$,
projecting onto the $r$-ball by clipping. Does the constrained solver recover
the same weights, and does the multiplier computed from stationarity—taking the
inner product of $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ with
$\mathbf{w}$ to get $\lambda = -\mathbf{w}^\top\nabla\ell(\mathbf{w}) / r^2$—match
$\lambda$?

```{.python .input #constrained-weight-decay}
rng = np.random.default_rng()
A, b = rng.normal(size=(30, 5)), rng.normal(size=30)
grad = lambda w: A.T @ (A @ w - b)                    # gradient of 1/2 |Aw - b|^2
step = 1.0 / np.linalg.eigvalsh(A.T @ A).max()        # 1 / lambda_max(A^T A)
print(' lambda   r = |w_pen|   |w_con - w_pen|   recovered multiplier')
for lam in [0.1, 1.0, 10.0]:
    w_pen = np.linalg.solve(A.T @ A + lam * np.eye(5), A.T @ b)  # penalty form
    r = np.linalg.norm(w_pen)                         # its norm becomes the budget
    w = np.zeros(5)                                   # constraint form: PGD
    for _ in range(5000):
        w -= step * grad(w)
        w *= min(1.0, r / np.linalg.norm(w))          # project onto the r-ball
    lam_rec = -(w @ grad(w)) / r ** 2                 # KKT: grad l + lam w = 0
    print(f'{lam:7.2f}  {r:10.4f}  {np.linalg.norm(w - w_pen):15.2e}  {lam_rec:20.6f}')
```

In every row, the constrained solution agrees with the penalized one to machine
precision, and the multiplier recovered from stationarity matches $\lambda$ to
all six printed decimals (the exact budgets $r$ vary with the random draw).
Penalty and constraint are simply two parameterizations of the same solution
path, with the Lagrange multiplier translating between them.

## The Convex Conjugate
:label:`subsec_mdl-convex-conjugate`

Evaluating the dual function $q$ normally requires solving a fresh unconstrained
minimization for each choice of multipliers. When the constraints are linear,
however, we can factor out the shared work and perform that minimization once
and for all. For constraints $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$, the
Lagrangian is
$f(\mathbf{x}) + \boldsymbol{\lambda}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$,
so

$$
q(\boldsymbol{\lambda}) = \inf_{\mathbf{x}} \bigl[ f(\mathbf{x}) + (\mathbf{A}^\top\boldsymbol{\lambda})^\top \mathbf{x} \bigr] - \mathbf{b}^\top \boldsymbol{\lambda}.
$$

Notice that the infimum depends on $f$ and on the single vector
$\mathbf{A}^\top\boldsymbol{\lambda}$, not on $\mathbf{A}$ or $\mathbf{b}$
separately. If we compute, once for each common objective $f$, the value of
$\inf_{\mathbf{x}} [f(\mathbf{x}) - \mathbf{y}^\top\mathbf{x}]$ as a function
of $\mathbf{y}$, then the dual of *any* linearly constrained problem with
objective $f$ can be read straight off a lookup table. That lookup table is
the convex conjugate.

### Supporting Lines

Start in one dimension with $f(x) = \tfrac12 x^2$ and pick a slope $y = 2$.
Which lines $t = 2x + c$ of slope $2$ stay everywhere on or below the parabola?
We need $2x + c \le \tfrac12 x^2$ for all $x$, meaning
$c \le \min_x (\tfrac12 x^2 - 2x) = -2$, attained at $x = 2$. The highest such
line, $t = 2x - 2$, touches the parabola tangentially at $(2, 2)$, where
$f'(2) = 2$. Lowering a line of slope $2$ until it touches the graph lands its
vertical intercept at $-2$, and the number
$\max_x (2x - \tfrac12 x^2) = 2$ is minus that intercept, as shown in
:numref:`fig_mdl-opt-conjugate-intercept`(a).

![The convex conjugate as a supporting line. (a) For $f(x)=\tfrac12x^2$ and slope $2$, lowering the dashed line of slope $2$ until it touches the graph lands at $x=2$ with intercept $-2$, giving $f^*(2)=2$. (b) For $f(x)=|x|$, a line of slope $\tfrac12$ supports the graph at the origin with intercept $0$, whereas a line of slope $2$ climbs faster than $|x|$ for every intercept, giving $f^*(\tfrac12)=0$ and $f^*(2)=+\infty$. Thus the conjugate is finite precisely for the slopes that admit a supporting line below the graph.](../img/mdl-opt-conjugate-intercept.svg)
:label:`fig_mdl-opt-conjugate-intercept`

The **convex conjugate** (also known as the Fenchel--Legendre transform) of a
function $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ records this
negated intercept for every slope vector $\mathbf{y}$ (we use an asterisk $*$
for conjugates and a star $\star$ for optima):

$$
f^*(\mathbf{y}) \;=\; \sup_{\mathbf{x}} \;\left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right).
$$
:eqlabel:`eq_mdl-opt-conjugate`

In geometric terms, $f^*(\mathbf{y})$ is minus the vertical intercept of the
highest affine function of slope $\mathbf{y}$ that lies everywhere below the
graph of $f$. Rather than describing $f$ by its point values, the conjugate
describes $f$ by its family of supporting lines. For $f(x) = \tfrac12 x^2$ and
any slope $y$, the supremum is attained at $x = y$, giving
$f^*(y) = \tfrac12 y^2$: the quadratic is its own conjugate.

Allowing $f$ to take the value $+\infty$ is a convenient way to encode a
restricted domain: points where $f(\mathbf{x}) = +\infty$ never contribute to
the supremum, and convexity of such an extended-valued function simply means
convexity of its epigraph.

The supremum itself can also be $+\infty$. Take $f(x) = |x|$ and slope $y = 2$.
For $x > 0$, $2x - |x| = x$ grows without bound: no line of slope $2$ can stay
below the V-shaped graph, no matter how low its intercept. Whenever $|y| \le 1$,
on the other hand, $yx - |x| \le 0$ with equality at $x = 0$, so the line of
slope $y$ through the origin supports the graph. Therefore $f^*(y) = 0$ for
$|y| \le 1$ and $f^*(y) = +\infty$ otherwise
(:numref:`fig_mdl-opt-conjugate-intercept`(b)). In other words, the conjugate
of the absolute value is the *indicator function* of the interval $[-1, 1]$—the
function equal to $0$ inside the set and $+\infty$ outside it.

Two immediate properties follow straight from :eqref:`eq_mdl-opt-conjugate`.
First, $f^*$ is *always* convex, regardless of whether $f$ is convex. For each
fixed $\mathbf{x}$, the expression $\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$
is affine in $\mathbf{y}$, and the pointwise supremum of affine functions is
convex because its epigraph is the intersection of the half-spaces above them
(the exact mirror image of the argument that made $q$ concave).

Second, because $f^*(\mathbf{y})$ is the supremum over all $\mathbf{x}$, we
have $f^*(\mathbf{y}) \ge \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ for every
$\mathbf{x}$ and $\mathbf{y}$. Rearranging gives the **Fenchel--Young
inequality**,

$$
\mathbf{y}^\top \mathbf{x} \;\le\; f(\mathbf{x}) + f^*(\mathbf{y})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y}.
$$
:eqlabel:`eq_mdl-opt-fenchel-young`

Equality holds if and only if $\mathbf{x}$ attains the supremum in
:eqref:`eq_mdl-opt-conjugate`. When $f$ is convex and differentiable, this
happens precisely when $\mathbf{y} = \nabla f(\mathbf{x})$, since
$\mathbf{x} \mapsto \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ is then concave
and maximized where its gradient vanishes. Convexity matters here: the
nonconvex double well $\phi_0(x) = (x^2 - 1)^2$ has $\phi_0'(0) = 0$, yet
$\phi_0^*(0) = -\min \phi_0 = 0$ and $\phi_0(0) + \phi_0^*(0) = 1 > 0$,
because $x = 0$ is a local maximum rather than a point where a horizontal line
supports the graph from below. For the self-conjugate pair
$\tfrac12\|\mathbf{x}\|^2$, Fenchel--Young reduces to the familiar inequality
$\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$.

Does passing to the conjugate lose any information? For any convex function $f$
that is *proper* (never $-\infty$ and not identically $+\infty$) and *closed*
(its epigraph is a closed set), its supporting lines determine the graph
completely, and taking the conjugate a second time recovers $f$: $f^{**} = f$
:cite:`Rockafellar.1970`. When $f$ is nonconvex (and bounded below by at least
one affine function), $f^{**}$ is the tightest closed convex function lying
below $f$—its convex envelope
(Exercise :exref:`ex_mdl-convexity-conjugate-pairs`d).

Here are the four conjugate pairs we use in this section and its exercises:

- $\tfrac12\|\mathbf{x}\|^2$ and $\tfrac12\|\mathbf{y}\|^2$, the self-conjugate
  pair;
- $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ and
  $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$ for $\mathbf{A} \succ 0$
  (attained at $\mathbf{x} = \mathbf{A}^{-1}\mathbf{y}$);
- $e^x$ and $y \log y - y$ for $y > 0$, taking the value $0$ at $y = 0$ and
  $+\infty$ for $y < 0$ (attained at $x = \log y$, so for instance
  $f^*(1) = -1$ and $f^*(e) = 0$);
- $\lambda\|\mathbf{x}\|_1$ and the indicator of the box
  $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$, the vector version of
  the absolute value above.

### Duals of Linearly Constrained Problems

Now return to the linearly constrained problem that motivated the conjugate:
minimize $f(\mathbf{x})$ subject to $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$.
Pulling a minus sign out of the infimum turns it directly into the supremum
defining $f^*$:

$$
\begin{aligned}
q(\boldsymbol{\lambda})
&= \inf_{\mathbf{x}} \left[ f(\mathbf{x}) + (\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x} \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -\sup_{\mathbf{x}} \left[ (-\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x} - f(\mathbf{x}) \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-dual-conjugate`

For equality constraints $\mathbf{A}\mathbf{x} = \mathbf{b}$, the exact same
three lines yield
$q(\boldsymbol{\nu}) = -f^*(-\mathbf{A}^\top\boldsymbol{\nu}) - \mathbf{b}^\top\boldsymbol{\nu}$,
with $\boldsymbol{\nu}$ unconstrained in sign. Notice that this gives a second
explanation for why $q$ is concave: $f^*$ is convex, its argument is linear in
$\boldsymbol{\lambda}$, and the minus sign flips convexity to concavity.

Let us try :eqref:`eq_mdl-opt-dual-conjugate` on the smallest possible example:
finding the point closest to the origin in a half-plane,
$\min_{\mathbf{x}} \tfrac12\|\mathbf{x}\|^2$ subject to
$\mathbf{a}^\top\mathbf{x} \le b$, with $\mathbf{a} = (1, 2)$ and $b = -1$.
Since the origin violates the constraint ($\mathbf{a}^\top\mathbf{0} = 0 > -1$),
the constraint must bind at the optimum. Plugging
$f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$ and $\|\mathbf{a}\|^2 = 5$ into
:eqref:`eq_mdl-opt-dual-conjugate` gives the dual function in one step:

$$
q(\lambda) = -\tfrac12 \lambda^2 \|\mathbf{a}\|^2 - b\lambda = \lambda - \tfrac52 \lambda^2 .
$$

Setting $q'(\lambda) = 1 - 5\lambda = 0$ yields $\lambda^\star = \tfrac15$ and
$d^\star = \tfrac15 - \tfrac1{10} = \tfrac1{10}$. The minimizer of the
Lagrangian at $\lambda^\star$ is the point attaining the conjugate supremum,
$\mathbf{x} = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$. Checking directly,
$\mathbf{a}^\top\mathbf{x} = -1 = b$ and
$f(\mathbf{x}) = \tfrac12(0.04 + 0.16) = \tfrac1{10} = d^\star$, so weak
duality certifies that $\mathbf{x}$ is optimal. This matches the projection
formula $(b / \|\mathbf{a}\|^2)\,\mathbf{a}$ from
:numref:`subsec_mdl-closest-point`, obtained here for a different hyperplane—and
where the primal had two variables, the dual had one and fell in a single line.

### Log-Sum-Exp and Entropy

Throughout deep learning, softmax converts raw scores into probabilities by
exponentiating and normalizing. Why exponentials, rather than some other
positive function? The conjugate of the log-sum-exp function,
$\mathrm{lse}(\mathbf{x}) = \log \sum_i e^{x_i}$, gives a clean answer. Let
$\Delta = \{\mathbf{p} : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$
denote the probability simplex and $H(\mathbf{p}) = -\sum_i p_i \log p_i$ the
Shannon entropy.

To compute
$\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \bigl(\mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})\bigr)$,
first check when the supremum can be finite. If any coordinate $p_i$ is
negative, sending $x_i \to -\infty$ drives $\mathbf{p}^\top\mathbf{x}$ to
$+\infty$ while $\mathrm{lse}(\mathbf{x})$ stays bounded, so the supremum is
$+\infty$. If $\mathbf{1}^\top\mathbf{p} \neq 1$, shifting all coordinates by a
constant $\mathbf{x} = c\mathbf{1}$ gives
$c(\mathbf{1}^\top\mathbf{p} - 1) - \log n$ (since
$\mathrm{lse}(c\mathbf{1}) = c + \log n$), which again blows up as
$c \to \pm\infty$. Hence $\mathrm{lse}^*(\mathbf{p})$ is finite only on the
probability simplex $\Delta$.

For $\mathbf{p}$ in the interior of $\Delta$ (all $p_i > 0$), setting the
gradient with respect to $\mathbf{x}$ to zero gives
$\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$, which is solved by
$\mathbf{x} = \log\mathbf{p}$ (up to an additive constant), yielding the value
$\sum_i p_i \log p_i$. If $\mathbf{p}$ lies on the boundary of $\Delta$ with
some $p_i = 0$, no finite maximizer exists because softmax outputs are always
strictly positive; nevertheless, sending those coordinates $x_i \to -\infty$
approaches the same value, so the formula extends continuously to all of
$\Delta$ under the standard convention $0 \log 0 = 0$. Because $\mathrm{lse}$
is convex and finite everywhere, $\mathrm{lse}^{**} = \mathrm{lse}$, giving the
conjugate pair

$$
\begin{aligned}
\mathrm{lse}^*(\mathbf{p}) &=
\begin{cases}
\sum_i p_i \log p_i = -H(\mathbf{p}), & \mathbf{p} \in \Delta, \\
+\infty, & \textrm{otherwise,}
\end{cases} \\
\mathrm{lse}(\mathbf{x}) &= \max_{\mathbf{p} \in \Delta} \left( \mathbf{p}^\top \mathbf{x} + H(\mathbf{p}) \right) \quad \textrm{(dually)}.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-lse-entropy`

Read the second line of :eqref:`eq_mdl-opt-lse-entropy` in words: log-sum-exp
is the maximum of an expected score $\mathbf{p}^\top\mathbf{x}$ plus an entropy
bonus $H(\mathbf{p})$ over all distributions on $n$ outcomes. Which
distribution achieves this maximum? By the equality condition of Fenchel--Young,
it is $\mathbf{p} = \nabla\mathrm{lse}(\mathbf{x})$, whose $i$-th entry is
$\partial\,\mathrm{lse}/\partial x_i = e^{x_i} / \sum_j e^{x_j}$—precisely the
softmax (whose Hessian we computed in :numref:`sec_mdl-convexity`). Put another
way, among all distributions achieving the same expected score
$\mathbf{p}^\top\mathbf{x}$, softmax is the one with the largest entropy,
matching the maximum-entropy derivation in
:numref:`subsec_mdl-max-entropy-derivation` (and
Exercise :exref:`ex_mdl-lagrangian-duality-max-entropy` derives softmax from
that constraint side). Without the entropy term, maximizing
$\mathbf{p}^\top\mathbf{x}$ over $\Delta$ puts all mass on $\arg\max_i x_i$;
adding entropy softens that hard selection into a smooth probability
distribution. The very same conjugacy construction, applied to the convex
generator of an $f$-divergence, produces the variational bounds used to train
$f$-GANs (:numref:`sec_mdl-f-gan-dual`).

## Worked Examples
:label:`subsec_mdl-worked-duals`

We now work through three examples that highlight what duality reveals in
practice. In the support vector machine, complementary slackness concentrates
the solution on a handful of data points, while the dual exposes the data only
through inner products. In water-filling, the Lagrange multiplier carries
tangible engineering units—communication rate per unit of power. Finally, a
one-dimensional nonconvex problem demonstrates the strict duality gap that
convexity prevents.

### The Support Vector Machine Dual
:label:`subsec_mdl-svm-dual`

Given linearly separable training data $(\mathbf{x}_i, y_i)$ with labels
$y_i \in \{\pm 1\}$, the support vector machine
:cite:`Boser.Guyon.Vapnik.1992`
seeks the separating hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ with the
widest possible margin. Its primal problem is

$$
\min_{\mathbf{w}, b}\; \tfrac12 \|\mathbf{w}\|^2
\quad \textrm{subject to} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \;\; i = 1, \ldots, n.
$$
:eqlabel:`eq_mdl-opt-svm-primal`

Because the signed distance from a point $\mathbf{x}$ to the hyperplane is
$|\mathbf{w}^\top\mathbf{x} + b| / \|\mathbf{w}\|$
(:numref:`sec_mdl-geometry-linear-algebraic-ops`), the two margin hyperplanes
$\mathbf{w}^\top\mathbf{x} + b = \pm 1$ sit at distance $1/\|\mathbf{w}\|$ on
either side of the decision boundary, or $2/\|\mathbf{w}\|$ apart. Minimizing
$\tfrac12\|\mathbf{w}\|^2$ therefore maximizes the margin. This is a convex
quadratic program with one affine inequality per training point. Since the
constraints are affine, Slater's condition requires only a feasible point—which
we obtain by rescaling any separating $(\mathbf{w}, b)$ until every point has
margin at least $1$. Strong duality is thus guaranteed.

Introduce one nonnegative multiplier $\alpha_i \ge 0$ for each constraint and
form the Lagrangian,

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
= \tfrac12\|\mathbf{w}\|^2 + \sum_i \alpha_i \left(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\right).
$$

To compute the dual function
$q(\boldsymbol{\alpha}) = \inf_{\mathbf{w}, b} \mathcal{L}$, notice first that
$\mathcal{L}$ is affine in the unpenalized offset $b$, with coefficient
$-\sum_i \alpha_i y_i$. Unless that coefficient vanishes, sending
$b \to \pm\infty$ drives $\mathcal{L}$ to $-\infty$, so
$q(\boldsymbol{\alpha}) = -\infty$ whenever $\sum_i \alpha_i y_i \neq 0$. Here
we see the effective domain of :eqref:`eq_mdl-opt-dual-function` in action:
the finiteness condition $\sum_i \alpha_i y_i = 0$ becomes an explicit equality
constraint in the dual. On this domain the $b$-term disappears, leaving a
strictly convex quadratic in $\mathbf{w}$ whose unique minimum occurs where
$\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$:

$$
\mathbf{w} = \sum_i \alpha_i y_i\, \mathbf{x}_i,
\qquad\textrm{on the domain}\qquad
\sum_i \alpha_i y_i = 0.
$$

Thus the optimal weight vector is a linear combination of the training inputs,
weighted by their multipliers. Substituting this expression back into
$\mathcal{L}$ yields the **SVM dual**:

$$
\max_{\boldsymbol{\alpha} \succeq 0,\ \sum_i \alpha_i y_i = 0}\;\;
\sum_i \alpha_i \;-\; \tfrac12 \sum_{i,j} \alpha_i \alpha_j\, y_i y_j\, \mathbf{x}_i^\top \mathbf{x}_j.
$$
:eqlabel:`eq_mdl-opt-svm-dual`

Two striking consequences follow before we even choose a solver. First,
complementary slackness, $\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$,
tells us that any point with $\alpha_i > 0$ must satisfy
$y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$, lying right on one of the margin
hyperplanes. These points are the **support vectors**. In
:numref:`fig_mdl-opt-svm-margin` they can only be among the four ringed
points. Their constraints are active and their normals balance $\nabla f$, as
in :numref:`fig_mdl-opt-kkt-active-set`, whereas every point with
$\alpha_i = 0$ drops out of $\mathbf{w}$ completely. Second, the training inputs enter
:eqref:`eq_mdl-opt-svm-dual` exclusively through inner products
$\mathbf{x}_i^\top\mathbf{x}_j$. Replacing those inner products with a kernel
function $k(\mathbf{x}_i, \mathbf{x}_j)$ immediately yields a nonlinear
classifier—a structure the primal formulation hides.

![Maximum-margin separation of the eight points that the code cell below uses. The solid line is the decision boundary $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = 0$ with $\mathbf{w}^\star = (\tfrac47, \tfrac47)$ and $b^\star = -\tfrac37$. The dashed lines are the margin hyperplanes $\mathbf{w}^{\star\top}\mathbf{x} + b^\star = \pm 1$, a distance $2/\|\mathbf{w}^\star\|$ apart. Only the four ringed points, which lie on a margin hyperplane, can carry $\alpha_i > 0$.](../img/mdl-opt-svm-margin.svg)
:label:`fig_mdl-opt-svm-margin`

Notice that the equality constraint $\sum_i \alpha_i y_i = 0$ arose solely
because the unpenalized offset $b$ left the Lagrangian affine in one direction.
To keep the dual feasible set as simple as the nonnegative orthant (where
projection just clips negative entries to zero), we solve a closely related
variant in code by folding the offset into the weights: augment each input with
a constant feature $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$, set
$\tilde{\mathbf{w}} = (\mathbf{w}, b)$, and minimize
$\tfrac12\|\tilde{\mathbf{w}}\|^2$ subject to
$y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$. Because this variant
penalizes $b$ as well, its decision boundary can differ slightly from
:eqref:`eq_mdl-opt-svm-primal`; in return, its Lagrangian is strictly convex in
every primal variable, eliminating the equality constraint. Defining
$\mathbf{Q}_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$, the
dual becomes

$$
\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top \mathbf{Q}\, \boldsymbol{\alpha},
$$
:eqlabel:`eq_mdl-opt-svm-dual-folded`

a concave quadratic over the nonnegative orthant. Applying projected gradient
ascent—the update of :numref:`subsec_mdl-projected-gd` with the step sign
flipped—iterates
$\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - \mathbf{Q}\boldsymbol{\alpha}))$
using the safe step size $\eta = 1/\lambda_{\max}(\mathbf{Q})$. Let us run this
ascent on eight points in the plane and check three predictions: does the
recovered hyperplane separate the data, do all positive multipliers sit on
points of margin $1$, and does the dual value match the primal value?

```{.python .input #constrained-svm-dual}
X = np.array([[1.0, 1.5], [2.0, 0.5], [2.5, 2.0], [1.5, 3.0],
              [-1.0, -0.5], [-2.0, 1.0], [-1.5, -2.0], [0.5, -1.5]])
y = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
Xa = np.hstack([X, np.ones((len(X), 1))])     # fold the offset b into w
G = y[:, None] * Xa                           # rows y_i x~_i
Q = G @ G.T                                   # Q_ij = y_i y_j x~_i^T x~_j
alpha = np.zeros(len(X))
eta = 1.0 / np.linalg.eigvalsh(Q).max()       # 1 / lambda_max(Q)
for _ in range(5000):                         # projected gradient ascent
    alpha = np.maximum(0.0, alpha + eta * (1.0 - Q @ alpha))
w = G.T @ alpha                               # w~* = sum_i alpha_i y_i x~_i
margins = y * (Xa @ w)
dual, primal = alpha.sum() - 0.5 * alpha @ Q @ alpha, 0.5 * w @ w
print('alpha* =', alpha.round(4))
print('rank of Q =', np.linalg.matrix_rank(Q), '  multipliers > 0:', int((alpha > 1e-8).sum()))
print('w* =', w[:2].round(4), ' b* =', w[2].round(4),
      ' separates data:', bool((np.sign(Xa @ w) == y).all()))
print('margins y_i (w^T x_i + b) =', margins.round(4))
print(f'primal = {primal:.6f}, dual = {dual:.6f}, gap = {abs(primal - dual):.1e}, '
      f'comp. slack = {np.abs(alpha * (margins - 1.0)).max():.1e}')
```

All three predictions hold. Four multipliers are positive, and precisely those
four points sit at margin $1$; the other four have margins between $1.29$ and
$2.43$ with $\alpha_i = 0$. The printed digits match the exact solution
$\tilde{\mathbf{w}}^\star = (\tfrac47, \tfrac47, -\tfrac37)$—that is,
$\mathbf{w}^\star = (\tfrac47, \tfrac47)$ and $b^\star = -\tfrac37$—with
$p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$ and a duality gap near
$10^{-16}$. (The equality of the primal and dual values also follows directly
from the identity $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ at
the optimum; see
Exercise :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`a.)

The printout also highlights a subtle point about interpreting dual solutions.
Because the primal objective $\tfrac12\|\tilde{\mathbf{w}}\|^2$ is strictly
convex, the primal optimum $\tilde{\mathbf{w}}^\star$ is unique. The dual
optimum $\boldsymbol{\alpha}^\star$ is *not*. Here
$\mathbf{Q} = \mathbf{G}\mathbf{G}^\top$, where $\mathbf{G}$ is the
$8 \times 3$ matrix with rows $y_i\tilde{\mathbf{x}}_i^\top$, so $\mathbf{Q}$
has rank at most $3$. Because the four support vectors correspond to four
vectors in $\mathbb{R}^3$, they are linearly dependent: there exists a nonzero
vector $\mathbf{d}$ supported on those four indices with
$\mathbf{G}^\top\mathbf{d} = \mathbf{0}$. Moving $\boldsymbol{\alpha}^\star$
along $\mathbf{d}$ leaves the quadratic term unchanged (since
$\mathbf{Q}\mathbf{d} = \mathbf{0}$) and leaves the linear term unchanged as
well, because every support vector has margin $1$:
$\mathbf{1}^\top\mathbf{d} = \sum_i d_i\, y_i\tilde{\mathbf{w}}^{\star\top}\tilde{\mathbf{x}}_i = \tilde{\mathbf{w}}^{\star\top}\mathbf{G}^\top\mathbf{d} = 0$.
Thus an entire line segment of multiplier vectors is optimal, all yielding the
same $\tilde{\mathbf{w}}^\star$ and the same dual value; projected ascent from
$\mathbf{0}$ simply returns one point on that segment. At each endpoint of the
segment, one of the four margin-$1$ points actually has $\alpha_i = 0$
(Exercise :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`c).
Remember that complementary slackness runs in one direction: $\alpha_i > 0$
guarantees margin $1$, but margin $1$ does not force $\alpha_i > 0$.

### Water-Filling

Our second worked example has both a closed-form solution and an unforgettable
physical picture. Suppose a transmitter must divide a total power budget $P$
across $n$ independent communication channels. Channel $i$ has background noise
power $N_i > 0$, and allocating power $P_i \ge 0$ to it yields a transmission
rate of $\log(1 + P_i / N_i)$ nats per use :cite:`Cover.Thomas.1999`. (Cover
and Thomas include a factor of $\tfrac12$ for a real Gaussian channel, which
rescales the Lagrange multiplier below without changing the optimal
allocation.) To maximize the total rate, we solve

$$
\max_{\mathbf{P}}\; U(\mathbf{P}) = \sum_{i=1}^n \log\left(1 + \frac{P_i}{N_i}\right)
\quad \textrm{subject to} \quad P_i \ge 0, \;\; \sum_i P_i = P.
$$
:eqlabel:`eq_mdl-opt-waterfilling`

Maximizing the concave function $U$ is equivalent to minimizing the convex
function $-U$ under affine constraints. Because a strictly positive feasible
allocation exists, Slater's condition holds, and the KKT conditions are both
necessary and sufficient. Introducing a multiplier $\nu$ for the budget
equality $\sum_i P_i - P = 0$ and multipliers $\lambda_i \ge 0$ for the
nonnegativity constraints $-P_i \le 0$, stationarity of the Lagrangian of $-U$
reads

$$
\frac{1}{N_i + P_i} = \nu - \lambda_i .
$$

Now read off the two cases. Whenever channel $i$ receives positive power
($P_i > 0$), complementary slackness forces $\lambda_i = 0$, so
$N_i + P_i = 1/\nu$: every active channel ends up at the exact same total
level $1/\nu$. Whenever channel $i$ receives no power ($P_i = 0$), dual
feasibility $\lambda_i = \nu - 1/N_i \ge 0$ requires $N_i \ge 1/\nu$. Writing
$\theta = 1/\nu$ for this common threshold, the optimal allocation takes the
compact form

$$
P_i^\star = \max(\theta - N_i,\, 0),
\qquad \textrm{with } \theta \textrm{ set by } \sum_i \max(\theta - N_i,\, 0) = P.
$$

This is the classic **water-filling** rule. Imagine each channel as a basin
whose floor sits at height $N_i$, and pour a volume $P$ of water into the
collection of basins. As :numref:`fig_mdl-opt-water-filling` illustrates, the
water settles to a uniform surface level $\theta$, pouring the most power into
the quietest channels and leaving any basin whose floor exceeds $\theta$
completely dry. Because $\sum_i \max(\theta - N_i, 0)$ is continuous, piecewise
linear, zero for $\theta \le \min_i N_i$, and strictly increasing for
$\theta > \min_i N_i$, any budget $P > 0$ determines a unique water level
$\theta$, which bisection finds effortlessly. (The simplex projection in
:numref:`subsec_mdl-projected-gd` has the same threshold structure inverted:
subtract a common level from each coordinate and clip at zero.)

![Water-filling for the noise floors and budget in the code cell below: pouring $P=3$ units of power into basins with floors at $N_i$ fills the three quietest channels to the common level $\theta=1/\nu^\star\approx1.43$, allocating $P_i^\star=\theta-N_i$ to each, while the two basins with floors above $\theta$ receive nothing, as complementary slackness requires.](../img/mdl-opt-water-filling.svg)
:label:`fig_mdl-opt-water-filling`

The shadow-price proposition gives $\nu^\star$ an immediate engineering
meaning. Since maximizing $U$ is minimizing $-U$, relaxing the budget from $P$
to $P + v$ changes $-U^\star$ at rate $-\nu^\star$, so
$\mathrm{d}U^\star / \mathrm{d}P = \nu^\star = 1/\theta$: the optimal
multiplier is the extra communication rate gained per unit of additional power.
Let us compute the water-filling allocation for five channels and check this
shadow-price prediction against finite differences at $P \pm 10^{-4}$.

```{.python .input #constrained-water-filling}
N = np.array([0.1, 0.4, 0.8, 1.6, 2.5])       # channel noise levels
P = 3.0                                       # total power budget

def water_fill(P, N, steps=60):
    lo, hi = N.min(), N.min() + P             # bracket for the water level
    for _ in range(steps):                    # bisection on the level theta
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.maximum(0, mid - N).sum() < P else (lo, mid)
    theta = 0.5 * (lo + hi)
    return theta, np.maximum(0.0, theta - N)

theta, power = water_fill(P, N)
print('water level theta =', f'{theta:.6f}', ' P* =', power.round(4),
      ' total =', f'{power.sum():.6f}')
print('wet channels at the common level:', (N + power)[power > 0].round(6))
print('dry channels above the level:    ', bool((N[power == 0] >= theta).all()))
rate = lambda power: np.log(1.0 + power / N).sum()
dP = 1e-4                                     # shadow price, two ways
nu = 1.0 / theta
sens = (rate(water_fill(P + dP, N)[1]) - rate(water_fill(P - dP, N)[1])) / (2 * dP)
print(f'nu = 1/theta = {nu:.6f} vs finite-difference dU*/dP = {sens:.6f}')
```

The power budget is met to machine precision, the three active channels all sit
at the common level $\theta = (3 + 0.1 + 0.4 + 0.8)/3 \approx 1.4333$, and the
noisy channels at $1.6$ and $2.5$ stay dry, just as dual feasibility requires.
Moreover, the multiplier $\nu^\star = 1/\theta = 0.697674$ matches the
finite-difference sensitivity $\mathrm{d}U^\star / \mathrm{d}P$ to all six
printed digits.

### An Explicit Duality Gap
:label:`subsec_mdl-explicit-duality-gap`

Weak duality holds for every problem, but strong duality does not. A
one-dimensional nonconvex example makes the gap concrete. Over the domain
$\mathcal{D} = [0, 1]$, consider

$$
\min_{x \in [0, 1]}\; f(x) = -x^2
\quad \textrm{subject to} \quad g(x) = x - \tfrac12 \le 0 .
$$

Because $f(x) = -x^2$ is concave rather than convex, the problem is nonconvex.
The feasible set is $[0, \tfrac12]$, and the minimum sits at the right
endpoint, $p^\star = f(\tfrac12) = -\tfrac14$. We dualize only the constraint
$g(x) \le 0$, keeping $\mathcal{D} = [0, 1]$ as a hard domain constraint so
that the infimum in :eqref:`eq_mdl-opt-dual-function` runs over $[0, 1]$ (weak
duality applies unchanged). For any $\lambda \ge 0$, the Lagrangian
$\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ is concave in $x$, so
its minimum on $[0, 1]$ must occur at one of the two endpoints:
$\mathcal{L}(0, \lambda) = -\lambda/2$ or
$\mathcal{L}(1, \lambda) = \lambda/2 - 1$. Therefore,

$$
q(\lambda) = \min\!\left(-\frac{\lambda}{2},\; \frac{\lambda}{2} - 1\right),
$$

which is piecewise linear and concave, as guaranteed. The two lines intersect
at $\lambda^\star = 1$, where $d^\star = -\tfrac12 < -\tfrac14 = p^\star$: a
strict duality gap of $\tfrac14$. Let us plot $q(\lambda)$ alongside $p^\star$
to compare this nonconvex dual curve with the clipping plot from the start of
the section.

```{.python .input #constrained-duality-gap}
xs = np.linspace(0.0, 1.0, 4001)              # the domain D = [0, 1]
fx, gx = -xs**2, xs - 0.5                     # objective and constraint on D
p_star = fx[gx <= 0.0].min()                  # primal optimum: -1/4
lams = np.linspace(0.0, 3.0, 601)
q = np.array([(fx + lam * gx).min() for lam in lams])  # dual function on D
d_star, lam_star = q.max(), lams[q.argmax()]
print(f'p* = {p_star:.4f},  d* = {d_star:.4f} at lambda = {lam_star:.2f},'
      f'  gap = {p_star - d_star:.4f}')
d2l.plot(lams, [q, np.full_like(lams, p_star)], 'lambda', 'value',
         legend=['dual function q(lambda)', 'primal optimum p*'])
```

Unlike the clipping curve, which rose smoothly to touch $p^\star$, this tent
peaks at $-\tfrac12$, a full quarter below the primal optimum. Notice that a
strictly feasible point certainly exists—$x = \tfrac14$ lies in the interior of
$\mathcal{D}$ with $g(\tfrac14) = -\tfrac14 < 0$—so the gap is caused entirely
by the lack of convexity, not by a failure of Slater's condition. Returning to
panel (b) of :numref:`fig_mdl-opt-primal-dual-gap`, the image curve
$\{(u, -(u + \tfrac12)^2) : u \in [-\tfrac12, \tfrac12]\}$ bends downward, so
no supporting line below $\mathcal{A}$ can reach $(0, p^\star)$; the highest
one, of slope $-1$, certifies only $-\tfrac12$.

Two open challenges remain. First, for a nonconvex problem, the dual can bound
the optimum but need not reach it. Second, even when a saddle point exists,
updating $\mathbf{x}$ and $\boldsymbol{\lambda}$ simultaneously can fail to
converge, a central difficulty in minimax training. While training an entire
deep network is nonconvex, many of its building blocks (projections, clipped
updates, last-layer fits, and convex trust-region subproblems) are convex, and
everything in this section applies directly to them. So far, however, every
derivation in this chapter has assumed exact real arithmetic; in the next
section we turn to what happens when finite floating-point precision, rather
than the optimization algorithm itself, determines the answer.

## Summary

For any constrained optimization problem, the Lagrange dual function
$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$
is concave and satisfies weak duality, $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$,
so every choice of $\boldsymbol{\lambda} \succeq 0$ certifies a lower bound on
the primal optimum. When the primal problem is convex and satisfies Slater's
condition (a strictly feasible point for all non-affine inequalities), strong
duality closes the gap ($d^\star = p^\star$) and attained primal--dual optima
form a saddle point of the Lagrangian. For linear constraints, the dual reduces
to the convex conjugate,
$q(\boldsymbol{\lambda}) = -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$,
which turns conjugate pairs such as log-sum-exp and negative entropy directly
into dual formulations. Under strong duality, the optimal multipliers act as
shadow prices ($\lambda_i^\star = -\partial p^\star / \partial u_i$ and
$\nu_j^\star = -\partial p^\star / \partial v_j$ wherever $p^\star$ is
differentiable), explaining why weight decay matches a norm constraint for
convex losses, why SVM multipliers vanish off the support vectors, and why the
water-filling multiplier $1/\theta$ measures communication rate per unit of
power. Without convexity, a strict duality gap can remain, as shown by
minimizing $-x^2$ on $[0, 1]$ subject to $x \le \tfrac12$.

## Exercises

1. **Conjugate pairs.** Compute each conjugate
   from :eqref:`eq_mdl-opt-conjugate`.
    1. $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ with $\lambda > 0$.
    1. $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top \mathbf{A} \mathbf{x}$ with
       $\mathbf{A} \succ 0$. What does the Fenchel--Young inequality
       :eqref:`eq_mdl-opt-fenchel-young` state for this pair?
    1. The softplus $f(t) = \log(1 + e^t)$ on $\mathbb{R}$. The binary
       cross-entropy of a logit $t$ against a soft label $p \in [0, 1]$ is
       $\log(1 + e^t) - pt$. Use Fenchel--Young to show that it is at least
       the binary entropy $H(p) = -p\log p - (1-p)\log(1-p)$, with equality
       exactly when $p = 1/(1 + e^{-t})$, the sigmoid of $t$. Write the
       difference as a KL divergence.
    1. The double well $\phi_0(x) = (x^2 - 1)^2$ of
       :numref:`subsec_mdl-recognizing-convexity` (the function $\phi$ there
       without its tilt $x/2$) is not convex. Find its convex envelope
       $\phi_0^{**}$ and the set on which $\phi_0^{**} < \phi_0$.

   *Adapted from Boyd and Vandenberghe,
   [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/),
   Examples 3.22 and 3.26.*
   :exlabel:`ex_mdl-convexity-conjugate-pairs`

1. **Maximum entropy and temperature.** The entropy of a probability vector
   $\mathbf{p} \in \mathbb{R}^n$ is $H(\mathbf{p}) = -\sum_i p_i \log p_i$.
   This exercise derives the softmax from the constraint side.
    1. Maximize $H$ subject to $\sum_i p_i = 1$ with one Lagrange
       multiplier, ignoring the sign constraints $p_i \ge 0$. Show that the
       only stationary point is the uniform distribution. Why does ignoring
       the sign constraints not change the answer?
    1. Fix scores $s_1, \ldots, s_n$, not all equal, and a target $\bar s$
       strictly between $\min_i s_i$ and $\max_i s_i$, and add the constraint
       $\sum_i p_i s_i = \bar s$. Show that the maximizer is
       $\mathbf{p} = \mathrm{softmax}(\beta\mathbf{s})$ for a scalar $\beta$
       determined by the multiplier of the score constraint. Show that the
       mean score under $\mathrm{softmax}(\beta\mathbf{s})$ increases strictly
       with $\beta$, so that exactly one $\beta$ meets the target.
    1. Let $H^\star(\bar s)$ be the maximal entropy. Show that
       $\mathrm{d}H^\star / \mathrm{d}\bar s = -\beta$, so that the multiplier
       prices the score constraint as the multiplier of
       :numref:`subsec_mdl-closest-point` prices the offset $b$. For which
       targets $\bar s$ is $\beta$ negative? What happens to the softmax
       temperature $1/\beta$ as $\bar s$ approaches $\max_i s_i$?
   :exlabel:`ex_mdl-lagrangian-duality-max-entropy`

1. **A dual certificate for $\ell_1$ minimization.** Basis pursuit,

   $$\min_{\mathbf{x}}\; \|\mathbf{x}\|_1 \quad \textrm{subject to} \quad \mathbf{A}\mathbf{x} = \mathbf{b},$$

   with $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $m < n$, selects a sparse
   solution of an underdetermined linear system. Its constraints are affine,
   so Slater's condition needs only a feasible point and gives
   $d^\star = p^\star$ whenever the system has a solution. The conjugate of
   $\|\cdot\|_1$ is the indicator of the box
   $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le 1\}$.
    1. Use the equality form of :eqref:`eq_mdl-opt-dual-conjugate` to show
       that, after replacing $\boldsymbol{\nu}$ by $-\boldsymbol{\nu}$, the
       dual problem is $\max_{\boldsymbol{\nu}} \mathbf{b}^\top \boldsymbol{\nu}$
       subject to $\|\mathbf{A}^\top \boldsymbol{\nu}\|_\infty \le 1$.
    1. Solve both problems by hand for $\mathbf{A} = (1\;\; 2)$ and $b = 2$,
       and check $d^\star = p^\star$.
    1. Let $\mathbf{x}^\star$ and $\boldsymbol{\nu}^\star$ be optimal. Show
       that $x_i^\star = 0$ wherever
       $|(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i| < 1$, and that
       $(\mathbf{A}^\top \boldsymbol{\nu}^\star)_i = \mathrm{sign}(x_i^\star)$
       wherever $x_i^\star \neq 0$. A candidate $\mathbf{x}$ with
       $\mathbf{A}\mathbf{x} = \mathbf{b}$ is given. Which conditions must a
       vector $\boldsymbol{\nu}$ satisfy to certify that $\mathbf{x}$ is
       optimal without solving the problem?

1. **Support vectors and the dual value.** For the offset-folded SVM dual
   :eqref:`eq_mdl-opt-svm-dual-folded`, let $\boldsymbol{\alpha}^\star$ be
   optimal and
   $\tilde{\mathbf{w}}^\star = \sum_i \alpha_i^\star y_i \tilde{\mathbf{x}}_i$.
    1. Show that $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ by
       writing the KKT conditions of the dual over the orthant. Conclude that
       the dual optimal value equals the primal value
       $\tfrac12 \|\tilde{\mathbf{w}}^\star\|^2$.
    1. Show that deleting a training point with $\alpha_j^\star = 0$ leaves
       the remaining multipliers optimal for the smaller problem and leaves
       $\tilde{\mathbf{w}}^\star$ unchanged. In leave-one-out validation, each
       of the $n$ points is classified by the SVM trained on the other
       $n - 1$ points. Conclude that the leave-one-out error rate is at most
       the fraction of points with $\alpha_i^\star > 0$, for every optimal
       $\boldsymbol{\alpha}^\star$. Which optimal $\boldsymbol{\alpha}^\star$
       gives the best bound?
    1. [code] The matrix $\mathbf{Q}$ of the SVM cell has rank $3$. Let $S$
       be the four indices with $\alpha_i^\star > 0$ and $\mathbf{G}_S$ the
       corresponding rows of `G`. Find a nonzero $\mathbf{d} \in \mathbb{R}^4$
       with $\mathbf{G}_S^\top\mathbf{d} = \mathbf{0}$ (use `np.linalg.svd` of
       $\mathbf{G}_S^\top$), and move $\boldsymbol{\alpha}^\star_S + t\mathbf{d}$ in
       each direction until a coordinate reaches zero. Verify that both
       endpoints are dual feasible, have dual value $\tfrac{41}{98}$, and give
       the same $\tilde{\mathbf{w}}^\star$. Which margin-$1$ point has
       $\alpha_i = 0$ at each endpoint? What does this say about reading
       "support vector" off the multipliers? (You should find the endpoints
       $(0, 0.204, 0, 0, 0, 0.192, 0, 0.441)$ and
       $(0.204, 0, 0, 0, 0, 0.273, 0, 0.359)$.)
   :exlabel:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`

1. [code] **Gradient descent--ascent.** ● Primal--dual methods and GAN
   training update the minimizing variable by gradient descent and the
   maximizing variable by gradient ascent at the same time.
    1. For the bilinear game $\min_x \max_y\, xy$, whose saddle point is the
       origin, write simultaneous descent--ascent,
       $x \leftarrow x - \eta y$ and $y \leftarrow y + \eta x$, as a linear
       map. Show that every nonzero start spirals away from the saddle for
       every $\eta > 0$. Show that the alternating version, in which the
       $y$-update uses the new $x$, is a linear map with determinant $1$.
       What do its iterates do for $\eta < 2$?
    1. Run simultaneous descent--ascent on the Lagrangian
       $\mathcal{L}(\mathbf{x}, \nu) = \tfrac12\mathbf{x}^\top \mathbf{x} + \nu(\mathbf{a}^\top \mathbf{x} - b)$
       of the closest-point problem of :numref:`subsec_mdl-closest-point`, here
       with $\mathbf{a} = (1, 2)$, $b = 1$, and the start
       $\mathbf{x} = \mathbf{0}$, $\nu = 0$. Find the step sizes for which the
       iterates converge to
       $(\mathbf{x}^\star, \nu^\star)$. Derive them from the eigenvalues of
       the update acting on the pair
       $(\mathbf{a}^\top \mathbf{x} / \|\mathbf{a}\|, \nu)$ and on the
       components of $\mathbf{x}$ orthogonal to $\mathbf{a}$.
    1. Add $\tfrac{\epsilon}{2} x^2$ with $0 < \epsilon \le 2$ to the
       objective of the bilinear game. For which step sizes does simultaneous
       descent--ascent converge now? Which term of the Lagrangian of the
       previous part plays the role of $\epsilon$?
   :exlabel:`ex_mdl-lagrangian-duality-descent-ascent`

1. [code] **Penalty versus constraint without convexity.** ● For a convex
   loss, :numref:`subsec_mdl-weight-decay-duality` shows that weight decay and
   a norm constraint trace the same solutions. Take instead the nonconvex
   one-dimensional loss $\ell(w) = (w - 1)^2 (w - 3)^2 - w$, whose local
   minima lie near $w = 1.16$ and $w = 3.11$.
    1. On a fine grid of $w$, compute the penalized minimizers
       $w_\lambda = \mathrm{argmin}_w\, \ell(w) + \tfrac\lambda2 w^2$ for
       $\lambda$ on a logarithmic grid, and the constrained optimum
       $p^\star(r) = \min \{\ell(w) : w^2 \le r^2\}$ for $r \in [0, 3.2]$. Plot
       the points $(w_\lambda^2, \ell(w_\lambda))$ on top of the curve
       $(r^2, p^\star(r))$. Which budgets $r$ are never the norm $|w_\lambda|$
       of a penalized minimizer?
    1. Show that $w_\lambda$ minimizes $\ell(w) + \tfrac\lambda2 w^2$ if and
       only if the line of slope $-\tfrac\lambda2$ through
       $(w_\lambda^2, \ell(w_\lambda))$ lies below every point of the set
       $\{(w^2, \ell(w)) : w \in \mathbb{R}\}$. Use this and the
       supporting-line picture of :numref:`fig_mdl-opt-primal-dual-gap` to
       explain the missing budgets.
    1. Pick a budget $r$ in the missing range at which the constraint is
       active. Compute the KKT multiplier $\lambda$ of the constrained
       solution, with the constraint written $\tfrac12 w^2 \le \tfrac12 r^2$
       as in the text, and find a point where the penalized objective with this
       $\lambda$ is lower. Compute the duality gap $p^\star(r) - d^\star(r)$
       of the constrained problem.
   :exlabel:`ex_mdl-lagrangian-duality-penalty-nonconvex`

1. [extended] **Entropic optimal transport.** ●● Given probability vectors
   $\mathbf{p} \in \mathbb{R}^n$ and $\mathbf{r} \in \mathbb{R}^m$ and a cost
   matrix $\mathbf{C} \in \mathbb{R}^{n \times m}$, optimal transport finds
   the cheapest plan $\boldsymbol{\Gamma}$ that moves the mass $\mathbf{p}$ to
   the mass $\mathbf{r}$,

   $$\mathrm{OT} = \min_{\boldsymbol{\Gamma} \ge 0}\; \sum_{i,j} C_{ij} \Gamma_{ij} \quad \textrm{subject to} \quad \boldsymbol{\Gamma} \mathbf{1} = \mathbf{p}, \;\; \boldsymbol{\Gamma}^\top \mathbf{1} = \mathbf{r},$$

   a linear program with $nm$ variables. Entropic regularization
   :cite:`Cuturi.2013` adds $\varepsilon \sum_{i,j} \Gamma_{ij} (\log \Gamma_{ij} - 1)$
   to the objective, with $\varepsilon > 0$; let $\boldsymbol{\Gamma}_\varepsilon$
   be the regularized optimum. Its solution by Sinkhorn iterations is used to
   compare distributions :cite:`Peyre.Cuturi.2019`.
    1. Form the Lagrangian with the terms
       $\boldsymbol{\phi}^\top (\mathbf{p} - \boldsymbol{\Gamma} \mathbf{1}) + \boldsymbol{\psi}^\top (\mathbf{r} - \boldsymbol{\Gamma}^\top \mathbf{1})$
       for the marginal constraints. Show that
       $\boldsymbol{\Gamma}_\varepsilon = \mathrm{diag}(\mathbf{u})\, \mathbf{K}\, \mathrm{diag}(\mathbf{v})$
       with $K_{ij} = e^{-C_{ij} / \varepsilon}$, $u_i = e^{\phi_i / \varepsilon}$,
       and $v_j = e^{\psi_j / \varepsilon}$. Why can the sign constraints
       $\Gamma_{ij} \ge 0$ be left out of the Lagrangian?
    1. Show that the dual function is

       $$q(\boldsymbol{\phi}, \boldsymbol{\psi}) = \mathbf{p}^\top \boldsymbol{\phi} + \mathbf{r}^\top \boldsymbol{\psi} - \varepsilon \sum_{i,j} e^{(\phi_i + \psi_j - C_{ij}) / \varepsilon},$$

       and that maximizing it exactly over $\boldsymbol{\phi}$ with
       $\boldsymbol{\psi}$ fixed gives the Sinkhorn update
       $\mathbf{u} = \mathbf{p} / (\mathbf{K} \mathbf{v})$, elementwise.
       Conclude that Sinkhorn's algorithm is block-coordinate ascent on a
       concave function and that its dual value never decreases. Why is every
       dual value a lower bound on $\mathrm{OT}$?
    1. Let $H(\boldsymbol{\Gamma}) = -\sum_{i,j} \Gamma_{ij} \log \Gamma_{ij}$,
       and grant that every feasible plan satisfies
       $\max(H(\mathbf{p}), H(\mathbf{r})) \le H(\boldsymbol{\Gamma}) \le H(\mathbf{p}) + H(\mathbf{r})$.
       Show that
       $0 \le \sum_{i,j} C_{ij} (\boldsymbol{\Gamma}_\varepsilon)_{ij} - \mathrm{OT} \le \varepsilon \min(H(\mathbf{p}), H(\mathbf{r}))$.
    1. Let the costs lie in $[0, 1]$. In float64, $e^{-x}$ rounds to zero
       for $x$ above about $745$. For which $\varepsilon$ can entries of
       $\mathbf{K}$ vanish, and what then happens to the multiplicative
       updates $\mathbf{u} = \mathbf{p} / (\mathbf{K}\mathbf{v})$? Explain why
       updating $\boldsymbol{\phi}$ and $\boldsymbol{\psi}$ directly, with a
       log-sum-exp in place of the products with $\mathbf{K}$, avoids the
       problem.
   :exlabel:`ex_mdl-lagrangian-duality-entropic-ot`

1. [code] **The dual of ridge regression.** Write
   $\min_{\mathbf{w}}\tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$,
   with $\mathbf{A} \in \mathbb{R}^{n \times d}$, as
   $\min_{\mathbf{w}, \mathbf{z}}\tfrac12\|\mathbf{z}\|^2 + \tfrac\lambda2\|\mathbf{w}\|^2$
   subject to $\mathbf{A}\mathbf{w} - \mathbf{b} - \mathbf{z} = \mathbf{0}$,
   with multiplier $\boldsymbol{\alpha} \in \mathbb{R}^n$.
    1. Minimize the Lagrangian over $\mathbf{z}$ and $\mathbf{w}$ to get
       $\mathbf{z} = \boldsymbol{\alpha}$, $\mathbf{w} = -\mathbf{A}^\top\boldsymbol{\alpha}/\lambda$,
       and the dual function
       $q(\boldsymbol{\alpha}) = -\tfrac12\|\boldsymbol{\alpha}\|^2 - \tfrac1{2\lambda}\boldsymbol{\alpha}^\top\mathbf{A}\mathbf{A}^\top\boldsymbol{\alpha} - \mathbf{b}^\top\boldsymbol{\alpha}$.
       Why does strong duality hold?
    1. Maximize $q$ to get
       $\boldsymbol{\alpha}^\star = -\lambda(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$
       and
       $\mathbf{w}^\star = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top + \lambda\mathbf{I})^{-1}\mathbf{b}$.
       Explain why the solution is a combination of the rows of $\mathbf{A}$,
       as the SVM weight vector is a combination of the data points, and why
       the data enter only through $\mathbf{A}\mathbf{A}^\top$.
    1. For $n = 20$ examples and $d = 50$ features, verify numerically that
       the primal solve
       $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$
       and the dual formula agree to about $10^{-14}$, and that the primal and
       dual values coincide. Which solve is cheaper when $n \ll d$? What does
       replacing $\mathbf{A}\mathbf{A}^\top$ by a kernel matrix give?
   :exlabel:`ex_mdl-lagrangian-duality-ridge-dual`

[Discussions](https://d2l.discourse.group/t/36860)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.7]{.kicker}

Duality<br>**Lagrange dual function · weak & strong duality · Slater's condition · shadow prices · Fenchel conjugates · SVM & water-filling**
:::
:::

::: {.slide title="Certifying Optimality When the Gradient Is Nonzero"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Suppose an iterative solver for $\min_{\|\mathbf{x}\|_2 \le 1}\tfrac12\|\mathbf{x} - \mathbf{x}_0\|_2^2$ ($\mathbf{x}_0 = (3, 4)$) halts at feasible $\mathbf{x} = (0.55, 0.835)$ with $f(\mathbf{x}) = 8.01 \ge p^\star$:

- **Why unconstrained checks fail:** At the constrained optimum $\mathbf{x}^\star = (0.6, 0.8)$, $\nabla f(\mathbf{x}^\star) = (-2.4, -3.2) \neq \mathbf{0}$.
- **Dual lower-bound certificate:** For any $\lambda \ge 0$, $\lambda g(\mathbf{x}) \le 0$ on the disk, so minimizing $f(\mathbf{x}) + \lambda g(\mathbf{x})$ over all of $\mathbb{R}^2$ certifies an ironclad lower bound $q(\lambda) \le p^\star$.
:::

::: {.col .fig .big}
@fig:ch35-opt-dual-lower-bound-cert
:::
:::
:::

::: {.slide title="Folding Constraints into Unconstrained Lower Bounds"}
[The Dual Problem]{.kicker}

For the clipping problem ($f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$, $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$, $\|\mathbf{x}_0\|^2 = 25$), fix any $\lambda \ge 0$:

$$\min_{\mathbf{x} \in \mathbb{R}^2} \big[f(\mathbf{x}) + \lambda\,g(\mathbf{x})\big] \le \min_{\|\mathbf{x}\| \le 1} \big[f(\mathbf{x}) + \underbrace{\lambda\,g(\mathbf{x})}_{\le 0}\big] \le \min_{\|\mathbf{x}\| \le 1} f(\mathbf{x}) = p^\star$$

- **Closed-form unconstrained minimization:** Setting $\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \lambda) = (\mathbf{x} - \mathbf{x}_0) + \lambda\mathbf{x} = \mathbf{0}$ yields the unique minimizer $\mathbf{x}(\lambda) = \frac{\mathbf{x}_0}{1 + \lambda}$.
- **Evaluating the lower bound $q(\lambda) = \mathcal{L}(\mathbf{x}(\lambda), \lambda)$:**
  $$q(\lambda) = \frac{25}{2}\frac{\lambda^2}{(1+\lambda)^2} + \frac{\lambda}{2}\left(\frac{25}{(1+\lambda)^2} - 1\right) = \frac{25}{2}\frac{\lambda}{1 + \lambda} - \frac{\lambda}{2}$$
- **Tightest lower bound:** Setting $q'(\lambda) = \frac{25}{2(1+\lambda)^2} - \frac12 = 0$ selects $\lambda^\star = 4$, where $q(4) = 10 - 2 = \mathbf{8 = p^\star}$, proving immediately that $8.01$ is within $0.01$ of optimal.
:::

::: {.slide title="Visualizing the Dual Function on the Unit Disk"}
[The Dual Problem]{.kicker}

::: {.cols .vc}
::: {.col}
Every $\lambda \ge 0$ furnishes a valid lower bound $q(\lambda) \le p^\star = 8$:

| $\lambda$ | $0$ | $1$ | $2$ | $3$ | **$4$** | $5$ | $8$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $q(\lambda)$ | $0.00$ | $5.75$ | $7.33$ | $7.88$ | **$8.00$** | $7.92$ | $7.11$ |

- For $\lambda < 4$, $\mathbf{x}(\lambda)$ lies outside the disk ($g > 0$).
- For $\lambda > 4$, $\mathbf{x}(\lambda)$ is pulled strictly inside ($g < 0$).
- At $\lambda^\star = 4$, $\mathbf{x}(4) = (0.6, 0.8)$ lands on the boundary circle ($g = 0$).
:::

::: {.col .fig}
@!mdl-lagrangian-duality-a-lower-bound-from-one-multiplier
:::
:::
:::

::: {.slide title="The Lagrange Dual Function Is Always Concave"}
[The Dual Problem]{.kicker}

::: {.cols .vc}
::: {.col}
The **Lagrange dual function** is the unconstrained infimum of $\mathcal{L}$ over $\mathbf{x} \in \mathcal{D}$:

$$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}}\left[f(\mathbf{x}) + \sum_{i=1}^m \lambda_i\,g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j\,h_j(\mathbf{x})\right]$$

- **Universal concavity:** For each fixed $\mathbf{x}$, $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ is affine in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$. Because a pointwise infimum of affine lines is always concave, **$q$ is concave even when the primal is nonconvex**.
:::

::: {.col .fig}
@fig:ch35-opt-concave-infimum-envelope
:::
:::
:::

::: {.slide title="Weak Duality and the Duality Gap"}
[The Dual Problem]{.kicker}

Maximizing the concave dual function over dual-feasible multipliers ($\boldsymbol{\lambda} \succeq \mathbf{0}$) defines the convex **dual problem**:

$$d^\star = \sup_{\boldsymbol{\lambda} \succeq \mathbf{0},\, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le \inf_{\mathbf{x}\text{ feasible}} f(\mathbf{x}) = p^\star$$

- **Two-line proof of Weak Duality ($d^\star \le p^\star$):** For any primal-feasible $\mathbf{x}$ ($g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0$) and any dual-feasible $(\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu})$, every penalty term satisfies $\lambda_i g_i(\mathbf{x}) \le 0$ and $\nu_j h_j(\mathbf{x}) = 0$:
  $$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{z} \in \mathcal{D}}\mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x}) \le f(\mathbf{x})$$
- **Duality gap and stopping certificates:** The nonnegative difference $p^\star - d^\star \ge 0$ is the **duality gap**. Any dual-feasible pair $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ immediately bounds the suboptimality of any primal-feasible candidate $\mathbf{x}$ via $f(\mathbf{x}) - p^\star \le f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$.
:::

::: {.slide title="Slater's Condition and Strong Duality"}
[Strong Duality]{.kicker}

::: {.cols .vc}
::: {.col}
- **Slater's Condition ($\Rightarrow d^\star = p^\star$):** If the primal problem is convex ($f, g_i$ convex, $h_j$ affine) and there exists a point $\bar{\mathbf{x}} \in \mathrm{relint}(\mathcal{D})$ with $g_i(\bar{\mathbf{x}}) < 0$ (for non-affine $g_i$), $g_i(\bar{\mathbf{x}}) \le 0$ (for affine $g_i$), and $h_j(\bar{\mathbf{x}}) = 0$, then **strong duality holds ($d^\star = p^\star$)** and dual optimal multipliers $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ exist.
- **Affine shortcut:** When all inequalities are linear/affine (LPs, SVMs), mere feasibility suffices.
:::

::: {.col .fig .big}
@fig:ch35-opt-slater-interior-point
:::
:::
:::

::: {.slide title="Supporting-Line Geometry of Strong Duality"}
[Strong Duality]{.kicker}

::: {.cols .vc}
::: {.col}
For a single inequality $g(\mathbf{x}) \le 0$, form the epigraph set of attainable constraint and objective values:

$$\mathcal{A} = \big\{(u, t) : u \ge g(\mathbf{x}),\; t \ge f(\mathbf{x})\text{ for some }\mathbf{x} \in \mathcal{D}\big\}$$

- **Visualizing $p^\star$ and $q(\lambda)$:** $p^\star$ is the lowest height $t$ of $\mathcal{A}$ on the vertical axis $u = 0$. Evaluating $q(\lambda) = \inf_{(u,t)\in\mathcal{A}}(t + \lambda u)$ pushes a line of slope $-\lambda$ up from below until it supports $\mathcal{A}$; its intercept at $u = 0$ is $q(\lambda)$.
- **Why convexity + Slater work:** Convexity makes $\mathcal{A}$ convex so a supporting line exists at $(0, p^\star)$; Slater's point $u = g(\bar{\mathbf{x}}) < 0$ prevents that line from being vertical.
:::

::: {.col .fig}
@fig:mdl-opt-primal-dual-gap
:::
:::
:::

::: {.slide title="Saddle Points and Min–Max Swapability"}
[Strong Duality]{.kicker}

::: {.cols .vc}
::: {.col}
Primal and dual optima are the two orders of a min–max game on $\mathcal{L}$:

$$d^\star = \sup_{\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu}}\inf_{\mathbf{x} \in \mathcal{D}}\mathcal{L} \;\le\; \inf_{\mathbf{x} \in \mathcal{D}}\sup_{\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu}}\mathcal{L} = p^\star$$

- **Saddle-point equivalence:** $d^\star = p^\star$ with attained optima holds iff $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ is a saddle point ($\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$).
- **KKT necessity:** Minimizing over $\mathbf{x}$ gives $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$, and $d^\star = p^\star$ forces $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$.
:::

::: {.col .fig}
@fig:ch35-opt-saddle-minmax-game
:::
:::
:::

::: {.slide title="Min–Max Games in Deep Learning: GANs and Robustness"}
[Strong Duality]{.kicker}

::: {.cols .vc}
::: {.col}
- **Primal recovery & dimension reduction:** When $f$ is strictly convex, $\mathbf{x}^\star = \mathrm{argmin}_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ recovers the primal solution from the $n$-variable dual (e.g., kernel SVMs when $n \ll d$).
- **GAN & adversarial rotational instability:** On a bilinear game $\min_x\max_y xy$, simultaneous gradient descent–ascent multiplies the radius by $\sqrt{1 + \eta^2} > 1$ at every step, spiraling outward unless stabilized by extragradient or damping.
:::

::: {.col .fig .big}
@fig:ch35-opt-bilinear-gda-spiral
:::
:::
:::

::: {.slide title="Lagrange Multipliers as Shadow Prices"}
[Sensitivity & Prices]{.kicker}

How much does the optimal value improve if we relax the constraints to $g_i(\mathbf{x}) \le u_i$ and $h_j(\mathbf{x}) = v_j$? Let $p^\star(\mathbf{u}, \mathbf{v})$ denote the perturbed optimal value:

- **Global sensitivity lower bound:** Under strong duality with dual optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$, any $\mathbf{x}$ feasible for $(\mathbf{u}, \mathbf{v})$ satisfies:
  $$p^\star(\mathbf{0}, \mathbf{0}) = q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) \le f(\mathbf{x}) + \sum_{i=1}^m \lambda_i^\star g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^\star h_j(\mathbf{x}) \le f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$$
  Taking the infimum over perturbed-feasible $\mathbf{x}$ yields:
  $$p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$$
- **Differential shadow price:** Wherever $p^\star(\mathbf{u}, \mathbf{v})$ is differentiable at $(\mathbf{0}, \mathbf{0})$,
  $$\lambda_i^\star = -\frac{\partial p^\star}{\partial u_i}(\mathbf{0}, \mathbf{0}), \qquad \nu_j^\star = -\frac{\partial p^\star}{\partial v_j}(\mathbf{0}, \mathbf{0})$$
:::

::: {.slide title="Equivalence of Weight Decay and Norm Budgets"}
[Sensitivity & Prices]{.kicker}

::: {.cols .vc}
::: {.col}
For convex $\ell(\mathbf{w})$, adding an $\ell_2$ penalty is path-equivalent to constraining weights to a norm ball:

$$\min_{\mathbf{w}}\;\ell(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2 \;\Longleftrightarrow\; \min_{\tfrac12\|\mathbf{w}\|_2^2 \le \tfrac12 r^2}\;\ell(\mathbf{w})$$

- **Identical KKT stationarity:** Both share $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$.
- **Bijective correspondence:** The penalized minimizer $\mathbf{w}_\lambda$ solves the ball problem of radius $r = \|\mathbf{w}_\lambda\|_2$, and the ball multiplier satisfies $\lambda^\star = -\partial p^\star/\partial(\tfrac12 r^2)$.
:::

::: {.col .fig}
@fig:ch35-opt-weight-decay-norm-ball
:::
:::
:::

::: {.slide title="Numerical Verification: Penalty vs. Ball Projection"}
[Sensitivity & Prices]{.kicker}

On a convex least-squares loss $\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2$ ($30 \times 5$), we solve the penalized ridge problem in closed form for $\lambda \in \{0.1, 1.0, 10.0\}$, set $r = \|\mathbf{w}_\lambda\|_2$, and run PGD on the $r$-ball:

@!constrained-weight-decay

- PGD on the norm ball recovers the penalized solution $\mathbf{w}_\lambda$ to machine precision ($10^{-16}$), and the multiplier recovered from $-\mathbf{w}^\top\nabla\ell(\mathbf{w})/r^2$ matches $\lambda$ to all six decimal places.
:::

::: {.slide title="The Fenchel Convex Conjugate: Supporting Lines"}
[Convex Conjugates]{.kicker}

::: {.cols .vc}
::: {.col}
The **convex conjugate** (or Fenchel conjugate) $f^*(\mathbf{y})$ encodes a function $f(\mathbf{x})$ by the intercepts of its supporting hyperplanes of slope $\mathbf{y}$:

$$f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathcal{D}}\big(\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})\big)$$

- **Always convex & biconjugacy:** $f^*$ is a pointwise supremum of affine functions of $\mathbf{y}$, hence always convex. If $f$ is proper, closed, and convex, then $f^{**} = f$.
- **Fenchel–Young inequality ($\forall\,\mathbf{x}, \mathbf{y}$):**
  $$\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$$
  For convex differentiable $f$, equality holds iff $\mathbf{y} = \nabla f(\mathbf{x})$.
:::

::: {.col .fig}
@fig:mdl-opt-conjugate-intercept
:::
:::
:::

::: {.slide title="Four Essential Fenchel Conjugate Pairs"}
[Convex Conjugates]{.kicker}

::: {.cols .vc}
::: {.col}
- **Quadratics:** $\tfrac12\|\mathbf{x}\|_2^2$ is self-conjugate ($\tfrac12\|\mathbf{y}\|_2^2$), and $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ ($\mathbf{A} \succ \mathbf{0}$) has conjugate $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$.
- **Exponential $\leftrightarrow$ entropy:** $f(x) = e^x$ has conjugate $f^*(y) = y\log y - y$ for $y > 0$ (attained at $x^\star = \log y$).
- **$\ell_1$ penalty $\leftrightarrow$ $\ell_\infty$ box:** $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ has conjugate $I_{\{\|\mathbf{y}\|_\infty \le \lambda\}}(\mathbf{y})$.
:::

::: {.col .fig .big}
@fig:ch35-opt-fenchel-conjugate-pairs
:::
:::
:::

::: {.slide title="One-Line Duals of Linearly Constrained Problems"}
[Convex Conjugates]{.kicker}

For any linearly constrained problem $\min_{\mathbf{x}} f(\mathbf{x})$ subject to $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ (or $\mathbf{A}\mathbf{x} = \mathbf{b}$), the dual function $q(\boldsymbol{\lambda})$ reduces directly to the Fenchel conjugate $f^*$:

$$q(\boldsymbol{\lambda}) = \inf_{\mathbf{x}}\big[f(\mathbf{x}) + (\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x}\big] - \mathbf{b}^\top\boldsymbol{\lambda} = -\sup_{\mathbf{x}}\big[(-\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x} - f(\mathbf{x})\big] - \mathbf{b}^\top\boldsymbol{\lambda} = -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$$

- **Worked example (closest point in a half-plane):** Minimize $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|_2^2$ subject to $\mathbf{a}^\top\mathbf{x} \le b$ with $\mathbf{a} = (1, 2)$ ($\|\mathbf{a}\|^2 = 5$) and $b = -1$.
- **One-line dual via $f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|_2^2$:**
  $$q(\lambda) = -f^*(-\lambda\mathbf{a}) - b\lambda = -\tfrac12\lambda^2\|\mathbf{a}\|_2^2 - b\lambda = \lambda - \tfrac52\lambda^2 \quad (\lambda \ge 0)$$
- Setting $q'(\lambda) = 1 - 5\lambda = 0$ gives $\lambda^\star = \tfrac15$, $d^\star = p^\star = \tfrac{1}{10}$, and primal optimum $\mathbf{x}^\star = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$.
:::

::: {.slide title="Log-Sum-Exp and Negative Shannon Entropy"}
[Convex Conjugates]{.kicker}

::: {.cols .vc}
::: {.col}
The Fenchel conjugate of $\mathrm{lse}(\mathbf{x}) = \log\sum_{i=1}^n e^{x_i}$ is negative Shannon entropy on the probability simplex $\Delta$:

$$\mathrm{lse}^*(\mathbf{p}) = \sum_{i=1}^n p_i\log p_i = -H(\mathbf{p}) \quad (\mathbf{p} \in \Delta)$$

- **Biconjugate variational law ($\mathrm{lse}^{**} = \mathrm{lse}$):** $\mathrm{lse}(\mathbf{x}) = \max_{\mathbf{p} \in \Delta}(\mathbf{p}^\top\mathbf{x} + H(\mathbf{p}))$, uniquely attained at $\mathbf{p}^\star = \mathrm{softmax}(\mathbf{x})$.
:::

::: {.col .fig}
@fig:ch35-opt-lse-entropy-duality
:::
:::
:::

::: {.slide title="Deriving the Support Vector Machine (SVM) Dual"}
[Worked Duals: SVM]{.kicker}

For linearly separable data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{\pm 1\}$, the hard-margin SVM maximizes the geometric margin $2/\|\mathbf{w}\|_2$ by solving the convex quadratic program:

$$\min_{\mathbf{w}, b}\;\tfrac12\|\mathbf{w}\|_2^2 \quad \text{subject to} \quad g_i(\mathbf{w}, b) = 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b) \le 0, \quad i = 1, \ldots, n$$

- **Forming and minimizing the Lagrangian ($\alpha_i \ge 0$):**
  $$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \tfrac12\|\mathbf{w}\|_2^2 + \sum_{i=1}^n \alpha_i\big(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big)$$
  1. **Minimize over $b$:** $\partial_b\mathcal{L} = -\sum_{i=1}^n \alpha_i y_i = 0$ (otherwise $\inf_b \mathcal{L} = -\infty$).
  2. **Minimize over $\mathbf{w}$:** $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w}^\star = \sum_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i$.
:::

::: {.slide title="Support Vectors and the Kernel Trick"}
[Worked Duals: SVM]{.kicker}

::: {.cols .vc}
::: {.col}
Substituting $\mathbf{w}^\star = \sum_i \alpha_i^\star y_i \mathbf{x}_i$ and $\sum_i \alpha_i y_i = 0$ into $\mathcal{L}$ yields the **SVM Dual**:

$$\max_{\boldsymbol{\alpha} \succeq \mathbf{0},\, \sum \alpha_i y_i = 0}\;\sum_{i=1}^n \alpha_i - \frac12\sum_{i,j=1}^n \alpha_i \alpha_j\,y_i y_j\,(\mathbf{x}_i^\top\mathbf{x}_j)$$

- **Support vector sparsity:** By complementary slackness, $\alpha_i^\star > 0$ only for active points on the margin $y_i(\mathbf{w}^{\star\top}\mathbf{x}_i + b^\star) = 1$.
- **Kernel trick:** Training and prediction depend on $\mathbf{x}_i$ solely via inner products $\mathbf{x}_i^\top\mathbf{x}_j \mapsto k(\mathbf{x}_i, \mathbf{x}_j)$.
:::

::: {.col .fig}
@fig:ch35-opt-svm-margin-support
:::
:::
:::

::: {.slide title="Solving the SVM Dual via Projected Gradient Ascent"}
[Worked Duals: SVM]{.kicker}

Folding the offset $b$ into $\tilde{\mathbf{w}} = (\mathbf{w}, b)$ with $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ turns the dual into $\max_{\boldsymbol{\alpha} \succeq \mathbf{0}}\big(\mathbf{1}^\top\boldsymbol{\alpha} - \tfrac12\boldsymbol{\alpha}^\top\mathbf{Q}\boldsymbol{\alpha}\big)$ over the nonnegative orthant, solved via projected gradient ascent:

@!constrained-svm-dual

- All $4$ positive multipliers sit at margin $1.0000$, and $p^\star = d^\star = 41/98 \approx 0.418367$ with gap $10^{-16}$.
:::

::: {.slide title="Water-Filling Power Allocation Across Channels"}
[Worked Duals: Water-Filling]{.kicker}

::: {.cols .vc}
::: {.col}
Divide total power $P > 0$ across $n$ channels with noise floors $N_i > 0$ to maximize total Shannon capacity:

$$\max_{\mathbf{P} \succeq \mathbf{0},\; \sum_{i=1}^n P_i = P}\;\; U(\mathbf{P}) = \sum_{i=1}^n \log\!\left(1 + \frac{P_i}{N_i}\right)$$

- **KKT stationarity:** With multiplier $\nu$ on $\sum_i P_i = P$ and $\lambda_i \ge 0$ on $-P_i \le 0$ ($\lambda_i P_i = 0$), stationarity requires $\frac{1}{N_i + P_i} = \nu - \lambda_i$.
- **Uniform water level $\theta = 1/\nu^\star$:** Active channels fill to $\theta$ ($P_i^\star = \max(\theta - N_i,\, 0)$), where $\theta$ solves:
  $$\sum_{i=1}^n \max(\theta - N_i,\, 0) = P$$
:::

::: {.col .fig}
@fig:mdl-opt-water-filling
:::
:::
:::

::: {.slide title="Verifying Water-Filling and Its Shadow Price"}
[Worked Duals: Water-Filling]{.kicker}

With noise floors $\mathbf{N} = (0.1, 0.4, 0.8, 1.6, 2.5)$ and total budget $P = 3.0$, bisection finds the water level $\theta$ and verifies that the Lagrange multiplier $\nu^\star = 1/\theta$ equals the marginal capacity $\mathrm{d}U^\star / \mathrm{d}P$:

@!constrained-water-filling

- The three quietest channels fill to $N_i + P_i^\star = \theta = 1.433333$, the two noisy channels ($N_i > \theta$) stay dry ($P_4^\star = P_5^\star = 0$), and $\nu^\star = 1/\theta = 0.697674$ matches $\mathrm{d}U^\star/\mathrm{d}P$ to six digits.
:::

::: {.slide title="When Strong Duality Fails: An Explicit Duality Gap"}
[Nonconvex Duality]{.kicker}

To see why convexity is required for strong duality, minimize the *concave* objective $f(x) = -x^2$ over $\mathcal{D} = [0, 1]$ subject to $g(x) = x - \tfrac12 \le 0$ ($p^\star = f(\tfrac12) = -\tfrac14$):

$$q(\lambda) = \min_{x \in [0, 1]}\big(-x^2 + \lambda(x - \tfrac12)\big) = \min\!\left(-\frac{\lambda}{2},\; \frac{\lambda}{2} - 1\right) \implies d^\star = q(1) = -\frac12 < -\frac14 = p^\star$$

@!constrained-duality-gap

- Even though $x = 1/4$ is strictly feasible ($g(1/4) = -1/4 < 0$), nonconvexity leaves a strict duality gap $p^\star - d^\star = 1/4$.
:::

::: {.slide title="Concept Check: Dual Bounds, Prices, and Water Levels"}
[Concept Check]{.kicker}

Test your understanding of dual certificates, shadow prices, and water-filling:

1. In the disk-clipping problem ($p^\star = 8$), a solver stops at a feasible point with $f(\mathbf{x}) = 8.01$. Without finding $\lambda^\star = 4$, what suboptimality bound does the unoptimized multiplier $\lambda = 3$ ($q(3) = 7.875$) certify?
2. Clipping $(3, 4)$ to the unit disk ($\tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$) has $p^\star(0) = 8$ and optimal multiplier $\lambda^\star = 4$. If we relax the constraint by $u = \tfrac12$ (enlarging the disk to radius $r = \sqrt{2}$), what lower bound on $p^\star(\tfrac12)$ does the shadow price guarantee, and how does it compare with the true $p^\star(\tfrac12)$?
3. In water-filling with noise floors $(0.1, 0.4, 0.8, 1.6, 2.5)$, how did we calculate the exact water level $\theta = 1.433333$ for $P = 3$ by hand?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** Weak duality guarantees $p^\star \ge q(3) = 7.875$, certifying immediately that $f(\mathbf{x}) - p^\star \le 8.01 - 7.875 = \mathbf{0.135}$. **(2)** The global sensitivity bound guarantees $p^\star(\tfrac12) \ge p^\star(0) - \lambda^\star u = 8 - 4(\tfrac12) = \mathbf{6.00}$. The exact new optimum is $\tfrac12(5 - \sqrt{2})^2 \approx \mathbf{6.43} > 6.00$. **(3)** Guessing that the $3$ quietest channels are active, $\sum_{i=1}^3(\theta - N_i) = 3\theta - (0.1 + 0.4 + 0.8) = 3 \implies 3\theta = 4.3 \implies \theta = \mathbf{4.3/3 \approx 1.4333}$, which is consistent since $0.8 < 1.4333 < 1.6$.
:::
:::

::: {.slide title="Summary: Lagrangian Duality and Conjugates"}
[Key Takeaways]{.kicker}

| Concept / Theorem | Mathematical Statement | Hypotheses Required | Practical Significance |
| :--- | :--- | :--- | :--- |
| **Lagrange Dual $q(\boldsymbol{\lambda}, \boldsymbol{\nu})$** | $\inf_{\mathbf{x} \in \mathcal{D}}\big(f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})\big)$ | None | Always concave in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ even for nonconvex primal |
| **Weak Duality** | $d^\star = \sup_{\boldsymbol{\lambda} \succeq \mathbf{0}, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$ | None | Any dual-feasible $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ certifies a lower bound on $p^\star$ |
| **Strong Duality (Slater)** | $d^\star = p^\star$ and saddle point of $\mathcal{L}$ | Convex primal + strict interior $\bar{\mathbf{x}}$ | Zero duality gap; KKT conditions are necessary & sufficient |
| **Shadow Prices** | $p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$ | Strong duality ($d^\star = p^\star$) | $\lambda_i^\star = -\partial p^\star/\partial u_i$; equates weight decay to a norm budget |
| **Fenchel Conjugate $f^*(\mathbf{y})$** | $\sup_{\mathbf{x}}(\mathbf{y}^\top\mathbf{x} - f(\mathbf{x}))$; $\mathbf{y}^\top\mathbf{x} \le f(\mathbf{x}) + f^*(\mathbf{y})$ | $f^{**} = f$ if proper closed convex | Linearly constrained dual is $-f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$; $\mathrm{lse}^* = -H$ |
:::

