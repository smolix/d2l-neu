# Duality
:label:`sec_mdl-lagrangian-duality`

An iterative solver stops when its progress becomes small and reports that
it has converged. Whether the point it returns is optimal is a separate
question. Without constraints, a zero gradient answers it for a convex
function (:numref:`sec_mdl-convexity`). A constrained optimum usually has a
nonzero gradient (:numref:`sec_mdl-constrained-optimization-duality`), so
that test is gone. Solvers need a lower bound on the optimal value to decide
when to stop, and proofs need one to show that an answer is optimal. This
section asks how to find such a bound, and what the multipliers of the KKT
conditions measure.

The clipping problem of :numref:`subsec_mdl-kkt-conditions` shows the
difficulty. It minimized the squared distance
$\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ to the point $(3, 4)$ over all
$\mathbf{x}$ with $\|\mathbf{x}\| \le 1$ and found
$\mathbf{x}^\star = (0.6, 0.8)$ with value $8$. (That section called the
point $\mathbf{y}$. Here it is $\mathbf{x}_0$, because $\mathbf{y}$ will be
needed for labels and for the argument of a conjugate.) Suppose instead that an
iterative solver stops at $\mathbf{x} = (0.55, 0.835)$. This point is
feasible ($\|\mathbf{x}\|^2 = 0.9997$) and has value $8.01$. Is that good
enough, or is there a feasible point with value $7$? The gradient at the
true optimum is $(-2.4, -3.2)$, not zero, and the disk has infinitely many
points, so we cannot check them all.

The bound comes from moving the constraint into the objective. Add $\lambda$
times the constraint function $\tfrac12(\|\mathbf{x}\|^2 - 1)$, with
$\lambda \ge 0$. At every feasible point the added term is at most zero, so
the minimum of the new objective over *all* of $\mathbb{R}^2$ lies below the
constrained optimum. Each $\lambda$ gives one lower bound, computed by an
unconstrained minimization, and $\lambda = 4$ gives exactly $8$: the
solver's $8.01$ is within $0.01$ of optimal. Choosing the best $\lambda$ is a
second optimization problem, the dual problem. The dual problem is always
convex. For a convex problem with a strictly feasible point, its value equals
the constrained optimum. Its solution, the vector of multipliers, measures
how fast the optimum improves when each constraint is relaxed.

We prove that no feasible point of the clipping problem beats
$\mathbf{x}^\star$, compute duals by hand for a half-plane, the support
vector machine, and a power allocation problem, and show a problem whose
best bound stays strictly below its optimum. Three results carry the
section: weak duality, which costs two lines; strong duality for convex
problems, whose proof holds the one hard step; and the reading of the
multipliers as prices. The conjugate, a tool for computing duals, comes
after the prices and can wait for a second reading. All computations are
small enough for NumPy, and `d2l` supplies the plots.

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

We want to prove that no feasible point of the clipping problem has a value
below $8$, without visiting feasible points. Every bound in this section
comes from one fact: at a feasible point, every inequality constraint
function is at most zero.

### A Lower Bound from One Multiplier

The clipping problem minimizes
$f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ with
$\mathbf{x}_0 = (3, 4)$, subject to
$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$. The factor
$\tfrac12$ makes the constraint's gradient equal to $\mathbf{x}$. The KKT
conditions gave the answer
$\mathbf{x}^\star = \mathbf{x}_0 / \|\mathbf{x}_0\| = (0.6, 0.8)$ with value
$p^\star = \tfrac12(5 - 1)^2 = 8$, and the multiplier $\lambda^\star = 4$ from
the stationarity condition
$\nabla f(\mathbf{x}^\star) + \lambda^\star \mathbf{x}^\star = (-2.4, -3.2) + 4\,(0.6, 0.8) = \mathbf{0}$.

Take any $\lambda \ge 0$ and any feasible $\mathbf{x}$. Since
$g(\mathbf{x}) \le 0$, the product $\lambda g(\mathbf{x})$ is at most zero,
and $f(\mathbf{x}) + \lambda g(\mathbf{x}) \le f(\mathbf{x})$. The minimum of
the left side over all of $\mathbb{R}^2$ is at most its value at this
feasible $\mathbf{x}$, and hence at most $f(\mathbf{x})$. Taking the best
feasible $\mathbf{x}$ on the right gives

$$
\min_{\mathbf{x} \in \mathbb{R}^2} \bigl[ f(\mathbf{x}) + \lambda g(\mathbf{x}) \bigr]
\;\le\; \min_{\|\mathbf{x}\| \le 1} f(\mathbf{x}) \;=\; p^\star
\qquad \textrm{for every } \lambda \ge 0 .
$$

We compute the left side. The function being minimized is the Lagrangian
$\mathcal{L}(\mathbf{x}, \lambda) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \tfrac\lambda2(\|\mathbf{x}\|^2 - 1)$.
For $\lambda \ge 0$ it is a convex quadratic in $\mathbf{x}$ with Hessian
$(1 + \lambda)\mathbf{I}$, so its minimizer is the point where the gradient
$(\mathbf{x} - \mathbf{x}_0) + \lambda \mathbf{x}$ vanishes:
$\mathbf{x}(\lambda) = \mathbf{x}_0 / (1 + \lambda)$. The minimizer shrinks
$\mathbf{x}_0$ toward the origin, more strongly for larger $\lambda$. To
substitute it, we use
$\mathbf{x}(\lambda) - \mathbf{x}_0 = -\lambda\mathbf{x}_0 / (1 + \lambda)$ and
$\|\mathbf{x}_0\|^2 = 25$, so
$\tfrac12\|\mathbf{x}(\lambda) - \mathbf{x}_0\|^2 = \tfrac{25}{2}\lambda^2/(1+\lambda)^2$
and $\|\mathbf{x}(\lambda)\|^2 = 25/(1+\lambda)^2$. The two terms with the
factor $25/(1+\lambda)^2$ add up to
$\tfrac{25}{2}(\lambda^2 + \lambda)/(1+\lambda)^2$, and the minimum value is

$$
q(\lambda) \;=\; \frac{25}{2}\,\frac{\lambda}{1 + \lambda} \;-\; \frac{\lambda}{2}.
$$
:eqlabel:`eq_mdl-opt-ball-dual`

The first term rises toward $12.5$ as $\lambda$ grows. The second term,
$\lambda$ times the constant $-\tfrac12$ in the constraint function, falls
without bound. The best bound balances the two.

A few values make this concrete. The function gives $q(0) = 0$,
$q(1) = 5.75$, $q(2) = 7.33$, $q(3) = 7.875$, $q(4) = 8$, $q(5) = 7.92$ and
$q(8) = 7.11$. Every one of them is a valid lower bound on $p^\star$. Its
derivative $q'(\lambda) = \tfrac{25}{2(1 + \lambda)^2} - \tfrac12$ vanishes
where $(1 + \lambda)^2 = 25$, at $\lambda = 4$, and
$q''(\lambda) = -25/(1 + \lambda)^3 < 0$, so $\lambda = 4$ is the maximum,
with value $q(4) = 10 - 2 = 8 = p^\star$. The best bound equals the optimum,
and the $\lambda$ that attains it is the KKT multiplier. The solver's $8.01$
is therefore within $0.01$ of optimal. Even $\lambda = 3$, picked without
solving anything, proves that it is within $0.135$.

Why is $\lambda = 4$ the right value? The constraint value at the shrunken
point is $g(\mathbf{x}(\lambda)) = \tfrac12\bigl(25/(1 + \lambda)^2 - 1\bigr)$.
For $\lambda < 4$ it is positive: the multiplier is too small to keep the
minimizer in the disk ($\mathbf{x}(1) = (1.5, 2)$ has norm $2.5$), and the
bound is the value of an infeasible point. For $\lambda > 4$ it is negative:
the minimizer is pushed deep inside ($\mathbf{x}(8)$ has norm $\tfrac59$), and
the term $\lambda g(\mathbf{x}(\lambda)) < 0$ lowers the bound. At
$\lambda = 4$ the minimizer $\mathbf{x}(4) = (0.6, 0.8)$ lies on the
circle, the added term vanishes, and the bound is the value of a feasible
point.

The cell evaluates $f + \lambda g$ at $\mathbf{x}(\lambda)$ on a grid of
$\lambda$ and plots the bounds against $p^\star$; the shape of the curve is
what to look at.

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

The curve is concave and stays below the line $p^\star = 8$, touching it
once, at $\lambda = 4$. Concavity is not a property of the disk: the next
subsection proves it for every problem.

### The Lagrange Dual Function

Nothing in this argument used the shape of the disk. The general problem of
:numref:`subsec_mdl-kkt-conditions` minimizes $f$ subject to inequality and
equality constraints,

$$
\min_{\mathbf{x} \in \mathcal{D}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad
g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m, \qquad h_j(\mathbf{x}) = 0,\; j = 1, \ldots, p,
$$

with optimal value $p^\star$, the infimum of $f$ over the feasible points
($p^\star = +\infty$ if there are none). The set $\mathcal{D}$ is the domain
of the problem, where $f$ and all constraint functions are defined: it is
$\mathbb{R}^n$ unless a problem keeps a set as a hard constraint, as the gap
example at the end of this section does with $[0, 1]$. The Lagrangian adds
one multiplier per constraint,

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}) + \sum_{i=1}^m \lambda_i\, g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j\, h_j(\mathbf{x}),
$$

with $\lambda_i \ge 0$ and $\nu_j$ of either sign. We write
$\boldsymbol{\lambda} \succeq 0$ for "every $\lambda_i \ge 0$", the
elementwise sense of $\succeq$ on vectors.

The bound of the clipping problem becomes a function of the multipliers.
The **Lagrange dual function** is the minimum of the Lagrangian over the
domain, for fixed multipliers,

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}}\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$
:eqlabel:`eq_mdl-opt-dual-function`

In words, $q$ asks how low $f$ can go once the constraints are replaced by
penalties with the given weights. For the clipping problem it is
:eqref:`eq_mdl-opt-ball-dual`. The infimum need not be attained or even
finite. Minimize $x$ subject to $-x \le 0$: the Lagrangian $(1 - \lambda)x$
is unbounded below unless $\lambda = 1$, so $q(1) = 0$ and $q(\lambda) = -\infty$
for every other $\lambda$. A value $-\infty$ is a true but useless bound. The
set where $q > -\infty$ is the *effective domain* of $q$; in the SVM dual
below, this restriction turns into an equality constraint.

The dual function has two properties that need no assumption on $f$, $g_i$,
or $h_j$.

**Proposition (the dual function is concave).** *For any primal problem,
$q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ in :eqref:`eq_mdl-opt-dual-function`
is concave.*

**Proof.** For each fixed $\mathbf{x}$, the map
$(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$
is affine, because the multipliers enter linearly. The dual function is the
pointwise infimum of this family of affine functions. Its **hypograph**, the
set of points on or below its graph, is the intersection of the half-spaces
below the affine functions, and an intersection of convex sets is convex.
So $q$ is concave. $\blacksquare$

**Proposition (weak duality).** *If $\mathbf{x}$ is feasible and
$\boldsymbol{\lambda} \succeq 0$, then
$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$. Consequently*

$$
d^\star \;=\; \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} q(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \inf_{\mathbf{x}\ \textrm{feasible}} f(\mathbf{x}) \;=\; p^\star.
$$
:eqlabel:`eq_mdl-opt-weak-duality`

**Proof.** For feasible $\mathbf{x}$, each $\lambda_i g_i(\mathbf{x}) \le 0$
(nonnegative times nonpositive) and each $\nu_j h_j(\mathbf{x}) = 0$, so

$$
q(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;=\; f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})
\;\le\; f(\mathbf{x}).
$$

Take the supremum over $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ on the left
and the infimum over feasible $\mathbf{x}$ on the right. $\blacksquare$

This is the inequality that made every value $q(\lambda)$ of the clipping
problem a bound. It was also the proof of KKT sufficiency in
:numref:`subsec_mdl-kkt-conditions`, which ran the same chain at the KKT
multipliers, where complementary slackness makes the bound tight. The
**dual problem** maximizes $q$ over
$\boldsymbol{\lambda} \succeq 0$ and $\boldsymbol{\nu}$. It is always a convex
problem, a concave function maximized over a convex set, even when the primal
problem is not convex. Its optimal value $d^\star$ is a lower bound on
$p^\star$, and every dual feasible point gives one. The difference
$p^\star - d^\star \ge 0$ is the **duality gap**. It was zero for the
clipping problem, and the last worked example of this section shows a
problem where it is not. When must it be zero? That is the next question.

## Strong Duality
:label:`subsec_mdl-strong-duality`

For the clipping problem the best bound equalled the optimum. For a
nonconvex problem it can fall short, and then the dual certifies only part
of the answer. Convexity and one strictly feasible point
guarantee equality, and a picture of the attainable pairs of constraint and
objective values shows why.

### Slater's Condition

**Proposition (Slater's condition implies strong duality).** *Suppose the
problem is convex ($f$ and every $g_i$ convex, every $h_j$ affine) and
satisfies Slater's condition :eqref:`eq_mdl-opt-slater`: some
feasible point $\bar{\mathbf{x}}$ in the relative interior of $\mathcal{D}$
has $g_i(\bar{\mathbf{x}}) < 0$ for every $g_i$ that is not affine. Then
$d^\star = p^\star$, and if $p^\star$ is finite the dual optimum is attained.*

This is the constraint qualification that :numref:`subsec_mdl-kkt-conditions`
introduced for convex problems: one point strictly inside all the curved
constraints. For the general proof see :citet:`Boyd.Vandenberghe.2004`, section
5.3.2. The clipping problem satisfies it: $f$ and $g$ are convex, and
$\mathbf{0}$ has $g(\mathbf{0}) = -\tfrac12 < 0$. So $q(4) = p^\star$ was
guaranteed before we computed anything.

Now the picture. Take a single inequality constraint $g$ and map each
$\mathbf{x} \in \mathcal{D}$ to the pair $(g(\mathbf{x}), f(\mathbf{x}))$ of constraint value and objective
value. Collect these pairs and every point above and to the right of them,

$$
\mathcal{A} = \bigl\{ (u, t) : u \ge g(\mathbf{x}),\ t \ge f(\mathbf{x})
\ \textrm{for some } \mathbf{x} \in \mathcal{D} \bigr\}.
$$

The optimum $p^\star$ is the lowest $t$ that $\mathcal{A}$ reaches on the line
$u = 0$. For $\lambda \ge 0$, the smallest value of $t + \lambda u$ over the
points that one $\mathbf{x}$ contributes is $f(\mathbf{x}) + \lambda g(\mathbf{x})$,
so $q(\lambda) = \inf_{(u, t) \in \mathcal{A}} (t + \lambda u)$. In other words,
the line $t + \lambda u = q(\lambda)$, of slope $-\lambda$, is lowered until it
touches $\mathcal{A}$ from below, and $q(\lambda)$ is its height at $u = 0$.
Since the line stays below $\mathcal{A}$, its height at $u = 0$ is at most
$p^\star$. This is weak duality drawn as a picture
(:numref:`fig_mdl-opt-primal-dual-gap`).

![The supporting-line geometry of duality. Every $\mathbf{x}$ maps to $(g(\mathbf{x}),f(\mathbf{x}))$; the shaded set $\mathcal{A}$ holds these pairs and everything above and to the right of them. Evaluating the dual at $\lambda$ lowers a line of slope $-\lambda$ until it supports $\mathcal{A}$ from below, and its height at $u=0$ is $q(\lambda)$. (a) The clipping problem: $\mathcal{A}$ is convex and $\mathbf{0}$ is strictly feasible, so the best line touches at $(0,p^\star)$ with slope $-4$: $d^\star=p^\star=8$. (b) Minimizing $-x^2$ on $[0,1]$ subject to $x\le\tfrac12$: $\mathcal{A}$ is not convex, the best line passes under the point $(0,p^\star)$, and $d^\star=-\tfrac12<-\tfrac14=p^\star$.](../img/mdl-opt-primal-dual-gap.svg)
:label:`fig_mdl-opt-primal-dual-gap`

**Proof (one inequality constraint with $g(\bar{\mathbf{x}}) < 0$, $p^\star$ finite).** First, $\mathcal{A}$
is convex. Take $(u_1, t_1)$ and $(u_2, t_2)$ in $\mathcal{A}$, contributed by
$\mathbf{x}_1$ and $\mathbf{x}_2$, and $\theta \in [0, 1]$. By convexity of
$g$, the point $\mathbf{x}_\theta = \theta\mathbf{x}_1 + (1 - \theta)\mathbf{x}_2$
has $g(\mathbf{x}_\theta) \le \theta g(\mathbf{x}_1) + (1 - \theta) g(\mathbf{x}_2) \le \theta u_1 + (1 - \theta) u_2$,
and the same holds for $f$, so the combination of the two points lies in
$\mathcal{A}$. Second, $(0, p^\star)$ is a boundary point of $\mathcal{A}$:
it is a limit of points of $\mathcal{A}$, while $(0, p^\star - \varepsilon)$
is not in $\mathcal{A}$ for any $\varepsilon > 0$. A convex set has a
supporting line at each boundary point: some $(\lambda, \gamma) \neq (0, 0)$
satisfies $\gamma t + \lambda u \ge \gamma p^\star$ for all $(u, t) \in \mathcal{A}$
(:citet:`Boyd.Vandenberghe.2004`, section 2.5.2; it is the fact behind the
subgradients of :numref:`sec_mdl-convexity`). Because $\mathcal{A}$ extends
upward and to the right without limit, $\gamma \ge 0$ and $\lambda \ge 0$. If
$\gamma = 0$, the line is vertical, $\lambda > 0$, and $\lambda u \ge 0$ forces
$u \ge 0$ on all of $\mathcal{A}$; the strictly feasible point contributes
$u = g(\bar{\mathbf{x}}) < 0$, a contradiction. So $\gamma > 0$, and dividing by
$\gamma$ gives $t + (\lambda/\gamma)\,u \ge p^\star$ on $\mathcal{A}$, that is,
$q(\lambda/\gamma) \ge p^\star$. With weak duality, $q(\lambda/\gamma) = p^\star$:
the dual optimum is attained and equals $p^\star$. $\blacksquare$

The proof used convexity once, to make $\mathcal{A}$ convex, and the strictly
feasible point once, to rule out a vertical line, which no finite multiplier
can represent. Panel (a) of the figure is the clipping problem: at
$\|\mathbf{x}\| = r$ the smallest objective is $\tfrac12(5 - r)^2$ and the
constraint value is $\tfrac12(r^2 - 1)$, so the lower edge of $\mathcal{A}$
is $t = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$. Its slope at $u = 0$ is
$-4$, which is why the best line has slope $-\lambda^\star = -4$. Panel (b) is
the nonconvex example worked at the end of this section: its set
$\mathcal{A}$ is not convex, and every line below it passes under
$(0, p^\star)$.

### Saddle Points

Strong duality has an equivalent form in terms of the Lagrangian alone. The
primal problem can be written as a min--max problem: for any $\mathbf{x}$,

$$
\sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= \begin{cases} f(\mathbf{x}) & \textrm{if } \mathbf{x} \textrm{ is feasible}, \\ +\infty & \textrm{otherwise}. \end{cases}
$$

A violated constraint lets its multiplier drive the supremum to $+\infty$,
while for feasible $\mathbf{x}$ the best the multipliers can do is vanish on
every slack constraint. So
$p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \mathcal{L}$,
and by definition
$d^\star = \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}$.
Weak duality is the inequality $\sup \inf \le \inf \sup$, which holds for any
function of two variables, and strong duality says that the order of the two
optimizations does not matter. When both optima are attained, this equality
of the two orders is the same as a saddle point of the Lagrangian.

**Proposition (strong duality is a saddle point).** *Suppose strong duality
holds with both optima attained, at a primal optimum $\mathbf{x}^\star$ and a
dual optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Then
$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ is a*
**saddle point** *of the Lagrangian: for all $\mathbf{x} \in \mathcal{D}$ and
all $\boldsymbol{\lambda} \succeq 0$, $\boldsymbol{\nu}$,*

$$
\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star).
$$
:eqlabel:`eq_mdl-opt-saddle-point`

*Conversely, any saddle point of $\mathcal{L}$ over
$\mathbf{x} \in \mathcal{D}$ and $(\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu})$
closes the duality gap, with saddle value $p^\star = d^\star$.*

**Proof.** Using the definitions and weak duality,

$$
d^\star
= q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; f(\mathbf{x}^\star)
= p^\star,
$$

where the last inequality holds because $\lambda_i^\star \ge 0$ and
$g_i(\mathbf{x}^\star) \le 0$ make every added term nonpositive. Strong
duality turns every inequality in this chain into an equality. Equality in
the first says that $\mathbf{x}^\star$ minimizes
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$,
the right half of :eqref:`eq_mdl-opt-saddle-point`. Equality in the second
forces $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$, which is
complementary slackness. Then for any $\boldsymbol{\lambda} \succeq 0$ and
$\boldsymbol{\nu}$,
$\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}^\star) + \sum_i \lambda_i g_i(\mathbf{x}^\star)
\le f(\mathbf{x}^\star)
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$,
the left half. For the converse, a saddle point gives

$$
\inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}
\;\le\; \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
= \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L},
$$

that is, $p^\star \le d^\star$; weak duality supplies the reverse inequality.
$\blacksquare$

In the clipping problem the saddle point is $(\mathbf{x}^\star, \lambda^\star) = ((0.6, 0.8), 4)$:
moving $\mathbf{x}$ away from it raises $\mathcal{L}(\cdot, 4)$, and changing
$\lambda$ leaves $\mathcal{L}(\mathbf{x}^\star, \lambda) = 8$ unchanged,
because $g(\mathbf{x}^\star) = 0$.

Strong duality has three practical uses.

- *The dual can be the smaller problem.* The SVM dual has one variable per
  data point, sign constraints, and at most one equality; the ridge dual of
  Exercise :exref:`ex_mdl-lagrangian-duality-ridge-dual` is a linear system
  of the size of the data set rather than of the feature space.
- *Every dual point is a stopping test.* For any feasible $\mathbf{x}$ and any
  $\boldsymbol{\lambda} \succeq 0$, $\boldsymbol{\nu}$, the difference
  $f(\mathbf{x}) - q(\boldsymbol{\lambda}, \boldsymbol{\nu})$ bounds how far
  $f(\mathbf{x})$ is above $p^\star$.
- *The dual solution locates the primal solution.* The optimum
  $\mathbf{x}^\star$ is *among* the minimizers of
  $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.
  When that minimizer is unique, for example when $f$ is strictly convex, it
  is $\mathbf{x}^\star$: in the clipping problem,
  $\mathbf{x}(4) = \mathbf{x}_0 / 5 = \mathbf{x}^\star$. Otherwise the
  minimizers can include infeasible points, and feasibility and
  complementary slackness select $\mathbf{x}^\star$ among them. In a linear
  program, $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star)$ is constant in
  $\mathbf{x}$, so every point minimizes it; in the SVM with an offset below,
  the Lagrangian does not determine the offset, and complementary slackness
  does.

The proposition also proves the necessity of the KKT conditions, which
:numref:`subsec_mdl-kkt-conditions` showed sufficient for convex problems,
leaving necessity to a constraint qualification. Suppose a convex problem with
differentiable $f$, $g_i$, $h_j$ on $\mathcal{D} = \mathbb{R}^n$ satisfies
Slater's condition and attains its optimum at $\mathbf{x}^\star$. Then strong
duality holds with an attained dual optimum, $\mathbf{x}^\star$ minimizes
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$, so
$\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{0}$ there, and complementary
slackness holds. Together with primal and dual feasibility these are the four
KKT conditions: under Slater's condition they are necessary as well as
sufficient.

The same min--max form describes training objectives that are not
constrained problems at all. Generative adversarial networks
(:numref:`sec_basic_gan`) and adversarial training are
$\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}$ problems
:cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014,Madry.Makelov.Schmidt.ea.2018`,
and without a saddle point a gap can separate their $\min\max$ and
$\max\min$ values. Even when a saddle point exists, taking gradient descent
steps in the minimizing variable and ascent steps in the maximizing one at
the same time need not reach it: for the bilinear objective $xy$, whose
saddle point is the origin, the iterates spiral outward for every step size
(Exercise :exref:`ex_mdl-lagrangian-duality-descent-ascent`).

## Multipliers as Prices
:label:`subsec_mdl-shadow-prices`

What is a constraint worth? In the clipping problem, if the disk were a
little larger, how much closer to $\mathbf{x}_0$ could we get? The optimal
multiplier answers this question. The same answer shows that, for a convex
loss, weight decay and a norm constraint on the weights produce the same
solutions.

### Shadow Prices

Relax each constraint by a given amount and record the optimal value as a
function of the constraint levels: for $\mathbf{u} \in \mathbb{R}^m$ and
$\mathbf{v} \in \mathbb{R}^p$ let

$$
p^\star(\mathbf{u}, \mathbf{v}) = \inf\,\{ f(\mathbf{x}) : g_i(\mathbf{x}) \le u_i,\; h_j(\mathbf{x}) = v_j \},
$$

so that $p^\star(\mathbf{0}, \mathbf{0})$ is the original optimum and $u_i > 0$
loosens inequality $i$. We have drawn this function already. For one inequality
constraint, the lowest $t$ that the set $\mathcal{A}$ of
:numref:`fig_mdl-opt-primal-dual-gap` reaches at level $u$ is $p^\star(u)$: the
lower edge of $\mathcal{A}$ is its graph, and the supporting line of slope
$-\lambda^\star$ lies below it. The proposition says this for any number of
constraints.

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

**Proof.** Let $\mathbf{x}$ be feasible for the perturbed problem:
$g_i(\mathbf{x}) \le u_i$ and $h_j(\mathbf{x}) = v_j$. Then

$$
p^\star(\mathbf{0}, \mathbf{0}) = q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})
\;\le\; f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v},
$$

using strong duality, the definition of $q$ as an infimum,
$\lambda_i^\star g_i(\mathbf{x}) \le \lambda_i^\star u_i$, and
$h_j(\mathbf{x}) = v_j$. Taking the infimum over all such $\mathbf{x}$ gives
:eqref:`eq_mdl-opt-shadow-price`. If $p^\star$ is differentiable at the
origin, the function
$p^\star(\mathbf{u}, \mathbf{v}) - p^\star(\mathbf{0}, \mathbf{0}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u} + \boldsymbol{\nu}^{\star\top}\mathbf{v}$
is nonnegative and vanishes at the origin, so the origin is its minimum and
its gradient is zero there. $\blacksquare$

The multiplier $\lambda_i^\star$ is the **shadow price** of constraint $i$:
the rate at which the optimal value falls as the constraint is relaxed. The
inequality holds for perturbations of any size. Relaxing constraint $i$ by
$u_i > 0$ improves the optimum by at most $\lambda_i^\star u_i$, and
tightening it by the same amount makes the optimum worse by at least
$\lambda_i^\star u_i$; to first order, when $p^\star$ is differentiable, the
change is exactly $\lambda_i^\star u_i$. A slack constraint has
$\lambda_i^\star = 0$ by complementary slackness: relaxing a constraint that
does not bind does not help.

For the clipping problem, relaxing $g(\mathbf{x}) \le 0$ to
$g(\mathbf{x}) \le u$ enlarges the disk to radius $r = \sqrt{1 + 2u}$, and
$p^\star(u) = \tfrac12\bigl(5 - \sqrt{1 + 2u}\bigr)^2$ is the edge computed
after the proof of strong duality, with slope $-4 = -\lambda^\star$ at
$u = 0$. Relaxing by $u = \tfrac12$, a disk of radius $\sqrt2$, gives
$p^\star = \tfrac12(5 - \sqrt2)^2 = 6.43$, above the bound
$8 - 4 \cdot \tfrac12 = 6$ that the proposition guarantees.

The multiplier depends on how the constraint is written. Here $\lambda^\star$
prices $u$, half the squared radius, not the radius $r$. Per unit of radius,
$\mathrm{d}p^\star / \mathrm{d}r = -(5 - r)$, which is $r$ times the price
per unit of $u$ by the chain rule and agrees with $-\lambda^\star$ only
because $r = 1$. Rescaling a constraint rescales its multiplier: written as
$\|\mathbf{x}\|^2 - 1 \le 0$, the same constraint would have multiplier $2$.

The closest-point example of :numref:`subsec_mdl-closest-point` is the
equality case. There the minimum of $\tfrac12\|\mathbf{x}\|^2$ over the hyperplane
$\mathbf{a}^\top\mathbf{x} = b$ is $p^\star(b) = b^2 / (2\|\mathbf{a}\|^2)$, and
$\partial p^\star / \partial b = -\nu^\star$ is the proposition for the
perturbation $h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = v$, which moves the
hyperplane to $b + v$. This is the rise of $0.02 = 0.2 \cdot 0.1$ that we
observed there without explaining it.

### Weight Decay and Norm Constraints
:label:`subsec_mdl-weight-decay-duality`

Weight decay (:numref:`sec_weight_decay`) adds a penalty to a training loss
$\ell(\mathbf{w})$. A norm constraint limits the weights instead. With the
main text's convention the two problems are

$$
\min_{\mathbf{w}}\; \ell(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2
\qquad \textrm{and} \qquad
\min_{\mathbf{w}}\; \ell(\mathbf{w}) \quad \textrm{subject to} \quad \tfrac12\|\mathbf{w}\|^2 \le \tfrac12 r^2 .
$$

The penalized problem is stationary where
$\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$. The constrained
problem, with multiplier $\lambda \ge 0$ on
$g(\mathbf{w}) = \tfrac12\|\mathbf{w}\|^2 - \tfrac12 r^2$, has KKT
stationarity $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$: the same
equation, with the weight-decay coefficient in the role of the multiplier.

For a *convex* loss the correspondence holds in both directions. From
penalty to constraint, let $\mathbf{w}_\lambda$ minimize the penalized
objective and set the budget to $r = \|\mathbf{w}_\lambda\|$. Then
$\mathbf{w}_\lambda$ is feasible, the constraint is active so complementary
slackness holds, and stationarity is the penalized optimality condition. The
KKT conditions are sufficient for convex problems
(:numref:`subsec_mdl-kkt-conditions`), so $\mathbf{w}_\lambda$ solves the
constrained problem.

From constraint to penalty, take $r > 0$. The point $\mathbf{w} = \mathbf{0}$
is then strictly feasible, so Slater's condition holds. If the constrained minimum is
attained at $\mathbf{w}^\star$, KKT necessity gives a multiplier
$\lambda^\star(r) \ge 0$, and $\mathbf{w}^\star$ is among the minimizers of the
convex function $\ell + \tfrac{\lambda^\star}{2}\|\mathbf{w}\|^2$. It is the
only one when $\lambda^\star > 0$, which makes the penalized objective
strictly convex. When the budget is slack, $\lambda^\star = 0$ and the
penalized problem is the unregularized one. The shadow-price proposition
reads $\lambda^\star(r) = -\partial p^\star / \partial(\tfrac12 r^2)$: the
weight-decay coefficient is the rate at which the best loss falls per unit of
half squared norm.

Neither direction used the shape of the squared norm. Let $\Omega$ be any
convex penalty, for instance $\|\mathbf{w}\|_1$ for the lasso of
:numref:`subsec_mdl-proximal-operators`. A minimizer $\mathbf{w}_\lambda$ of
$\ell + \lambda\Omega$ solves the problem with budget
$\Omega(\mathbf{w}) \le \Omega(\mathbf{w}_\lambda)$, because every $\mathbf{w}$
within that budget has
$\ell(\mathbf{w}) \ge \ell(\mathbf{w}_\lambda) + \lambda\,(\Omega(\mathbf{w}_\lambda) - \Omega(\mathbf{w})) \ge \ell(\mathbf{w}_\lambda)$.
Conversely, for convex $\ell$, a budget above the minimum of $\Omega$ has a
Slater point, and the saddle-point proposition makes a constrained minimizer
a minimizer of $\ell + \lambda^\star\Omega$.

The equivalence needs the convexity of $\ell$. For a nonconvex loss a
penalized stationary point is still a KKT point of the matching constrained
problem, but the two solution sets can differ, and some budgets are reached
by no penalty at all (Exercise :exref:`ex_mdl-lagrangian-duality-penalty-nonconvex`).

The same term has a third reading. For a squared loss with noise variance
$\sigma^2$ and a Gaussian prior $\mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ on the
weights, $\tfrac\lambda2\|\mathbf{w}\|^2$ with $\lambda = \sigma^2/\tau^2$ is the
negative log of the prior, up to a constant and the overall scale
(:numref:`sec_mdl-maximum_likelihood`). The one coefficient $\lambda$ is
thus a decay rate, a multiplier, and a ratio of variances.

Does the constrained problem with budget $r = \|\mathbf{w}_\lambda\|$ return
the penalized solution, and does the multiplier recovered from its solution
equal $\lambda$? The cell below takes the convex loss
$\ell(\mathbf{w}) = \tfrac12\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2$ with random
data, solves the penalized problem in closed form from
$(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^\top\mathbf{b}$,
and solves the constrained problem by projected gradient descent with step
$1/\lambda_{\max}(\mathbf{A}^\top\mathbf{A})$, where the projection onto the
ball is clipping. On the active constraint, the inner product of
stationarity with $\mathbf{w}$ gives
$\lambda = -\mathbf{w}^\top\nabla\ell(\mathbf{w}) / r^2$.

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

In every row the constrained solution matches the penalized one to machine
precision, and the multiplier recovered from the constrained solution
reproduces $\lambda$ to the six printed decimals; the budgets $r$ change with
the random data. Penalty and constraint are two parameterizations of the
same family of solutions, and the multiplier converts one into the other.

## The Convex Conjugate
:label:`subsec_mdl-convex-conjugate`

Computing $q$ needs an unconstrained minimization for every value of the
multipliers. For linear constraints this minimization has a structure that
lets us do it once and for all. With constraints
$\mathbf{A}\mathbf{x} \preceq \mathbf{b}$ the Lagrangian is
$f(\mathbf{x}) + \boldsymbol{\lambda}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$, so

$$
q(\boldsymbol{\lambda}) = \inf_{\mathbf{x}} \bigl[ f(\mathbf{x}) + (\mathbf{A}^\top\boldsymbol{\lambda})^\top \mathbf{x} \bigr] - \mathbf{b}^\top \boldsymbol{\lambda}.
$$

The infimum depends on $f$ and on one vector, $\mathbf{A}^\top\boldsymbol{\lambda}$,
and not otherwise on $\mathbf{A}$ or $\mathbf{b}$. If we tabulate, once for
each $f$, the value of $\inf_{\mathbf{x}} [f(\mathbf{x}) - \mathbf{y}^\top\mathbf{x}]$
as a function of $\mathbf{y}$, then every linearly constrained dual with
objective $f$ can be read off the table. That table is the convex conjugate.

### Supporting Lines

Start in one dimension with $f(x) = \tfrac12 x^2$ and the slope $y = 2$. Which
lines $t = 2x + c$ of slope $2$ stay below the graph? We need
$2x + c \le \tfrac12 x^2$ for every $x$, that is,
$c \le \min_x (\tfrac12 x^2 - 2x) = -2$, attained at $x = 2$. The highest such
line, $t = 2x - 2$, touches the parabola at $(2, 2)$, where its slope matches
$f'(2) = 2$. Lowering a line of slope $2$ until it touches the graph thus
puts its intercept at $-2$, and the number
$\max_x (2x - \tfrac12 x^2) = 2$ is minus that intercept.
:numref:`fig_mdl-opt-conjugate-intercept`(a) draws this.

![The convex conjugate as a supporting line. (a) For $f(x)=\tfrac12x^2$ and slope $2$, the dashed line of slope $2$ is lowered until it touches the graph, at $x=2$; its intercept is $-2$, so $f^*(2)=2$. (b) For $f(x)=|x|$ a line of slope $\tfrac12$ touches at the origin with intercept $0$, while a line of slope $2$ eventually runs above the graph for every intercept: $f^*(\tfrac12)=0$ and $f^*(2)=+\infty$. The conjugate is finite exactly for the slopes that admit a line below the graph.](../img/mdl-opt-conjugate-intercept.svg)
:label:`fig_mdl-opt-conjugate-intercept`

The **convex conjugate** (or Fenchel--Legendre transform) of a function
$f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ records this number for
every slope (an asterisk $*$ marks a conjugate, a star $\star$ an optimum):

$$
f^*(\mathbf{y}) \;=\; \sup_{\mathbf{x}} \;\left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right).
$$
:eqlabel:`eq_mdl-opt-conjugate`

That is, $f^*(\mathbf{y})$ is minus the intercept of the highest affine function
with slope $\mathbf{y}$ that lies below the graph of $f$. The conjugate
describes $f$ by its supporting lines instead of its values. For
$f(x) = \tfrac12 x^2$ and any $y$ the supremum is attained at $x = y$, so
$f^*(y) = \tfrac12 y^2$.

We allow $f$ to take the value $+\infty$, which encodes a domain. Points where
$f = +\infty$ never attain the supremum, and convexity of such a function means
convexity of its epigraph.

The supremum can be $+\infty$. For $f(x) = |x|$ and slope $2$,
$2x - |x| = x$ for $x > 0$, which grows without bound: no line of slope $2$
stays below the V, whatever its intercept. For $|y| \le 1$ instead,
$yx - |x| \le 0$ with equality at $x = 0$, so the line of slope $y$ through
the origin is a supporting line. Hence $f^*(y) = 0$ for $|y| \le 1$ and
$f^*(y) = +\infty$ otherwise
(:numref:`fig_mdl-opt-conjugate-intercept`(b)). The conjugate of the
absolute value is the *indicator* of the interval $[-1, 1]$, the function
that is $0$ on the set and $+\infty$ off it.

Two properties follow directly from the definition. First, $f^*$ is convex
for every $f$, convex or not. For fixed $\mathbf{x}$ the expression
$\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ is affine in $\mathbf{y}$, and a
pointwise supremum of affine functions is convex, because its epigraph is the
intersection of the half-spaces above them. The same argument with an
infimum made $q$ concave.

Second, the definition gives $f^*(\mathbf{y}) \ge \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$
for every $\mathbf{x}$, which rearranges into the **Fenchel--Young
inequality**,

$$
\mathbf{y}^\top \mathbf{x} \;\le\; f(\mathbf{x}) + f^*(\mathbf{y})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y}.
$$
:eqlabel:`eq_mdl-opt-fenchel-young`

Equality holds exactly when $\mathbf{x}$ attains the supremum in
:eqref:`eq_mdl-opt-conjugate`. For a differentiable *convex* $f$ this happens
exactly when $\mathbf{y} = \nabla f(\mathbf{x})$, since
$\mathbf{x} \mapsto \mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ is then concave and
maximized where its gradient vanishes. Convexity is needed. The double well
$\phi_0(x) = (x^2 - 1)^2$ has $\phi_0'(0) = 0$, but
$\phi_0^*(0) = -\min\phi_0 = 0$ and $\phi_0(0) + \phi_0^*(0) = 1 > 0$,
because $x = 0$ is a local maximum of $\phi_0$,
not a point where the line of slope $0$ touches from below. For the
self-conjugate pair $\tfrac12\|\mathbf{x}\|^2$, Fenchel--Young becomes the
familiar
$\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$.

Does the conjugate lose information? For a convex $f$ that is *proper* (never
$-\infty$ and not $+\infty$ everywhere) and *closed* (its epigraph is a closed
set), the supporting lines determine the graph, and applying the conjugate
twice recovers it: $f^{**} = f$ :cite:`Rockafellar.1970`. For a nonconvex
$f$ that has at least one affine function below it, $f^{**}$ is the
largest closed convex function below $f$, its convex envelope
(Exercise :exref:`ex_mdl-convexity-conjugate-pairs`d).

The pairs used in this section and its exercises are:

- $\tfrac12\|\mathbf{x}\|^2$ and $\tfrac12\|\mathbf{y}\|^2$, the self-conjugate
  pair;
- $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ and
  $\tfrac12\mathbf{y}^\top\mathbf{A}^{-1}\mathbf{y}$ for $\mathbf{A} \succ 0$
  (the maximizer is $\mathbf{x} = \mathbf{A}^{-1}\mathbf{y}$);
- $e^x$ and $y \log y - y$ for $y > 0$, with value $0$ at $y = 0$ and
  $+\infty$ for $y < 0$ (the maximizer is $x = \log y$; for instance
  $f^*(1) = -1$ and $f^*(e) = 0$);
- $\lambda\|\mathbf{x}\|_1$ and the indicator of the box
  $\{\mathbf{y} : \|\mathbf{y}\|_\infty \le \lambda\}$, the vector version of
  the absolute value above.

### Duals of Linearly Constrained Problems

We now return to the problem that motivated the conjugate,
$\min f(\mathbf{x})$ subject to $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$.
The one step that is not routine is pulling a minus sign out of the
infimum, which turns it into the supremum of the conjugate's definition:

$$
\begin{aligned}
q(\boldsymbol{\lambda})
&= \inf_{\mathbf{x}} \left[ f(\mathbf{x}) + (\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x} \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -\sup_{\mathbf{x}} \left[ (-\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x} - f(\mathbf{x}) \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}.
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-dual-conjugate`

For equality constraints $\mathbf{A}\mathbf{x} = \mathbf{b}$ the same three
lines give $q(\boldsymbol{\nu}) = -f^*(-\mathbf{A}^\top\boldsymbol{\nu}) - \mathbf{b}^\top\boldsymbol{\nu}$,
with $\boldsymbol{\nu}$ of either sign. The dual function is concave for a
second reason now: $f^*$ is convex and its argument is linear in
$\boldsymbol{\lambda}$.

The smallest case is the closest point of a half-plane to the origin:
minimize $\tfrac12\|\mathbf{x}\|^2$ subject to $\mathbf{a}^\top\mathbf{x} \le b$
with $\mathbf{a} = (1, 2)$ and $b = -1$. The origin violates the constraint
($\mathbf{a}^\top\mathbf{0} = 0 > -1$), so the constraint will be active. With
$f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$ and $\|\mathbf{a}\|^2 = 5$,
:eqref:`eq_mdl-opt-dual-conjugate` gives at once

$$
q(\lambda) = -\tfrac12 \lambda^2 \|\mathbf{a}\|^2 - b\lambda = \lambda - \tfrac52 \lambda^2 .
$$

Setting $q'(\lambda) = 1 - 5\lambda$ to zero gives $\lambda^\star = \tfrac15$
and $d^\star = \tfrac15 - \tfrac1{10} = \tfrac1{10}$. The minimizer of the
Lagrangian at $\lambda^\star$ is the maximizer in the conjugate,
$\mathbf{x} = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$. It satisfies
$\mathbf{a}^\top\mathbf{x} = -1 = b$ and has value
$\tfrac12(0.04 + 0.16) = \tfrac1{10}$, so it is feasible with value $d^\star$,
and weak duality makes it optimal. It is the projection
$(b / \|\mathbf{a}\|^2)\,\mathbf{a}$ of :numref:`subsec_mdl-closest-point`,
here for a different line. The dual had one variable where the primal had two,
and solving it took one line.

### Log-Sum-Exp and Entropy

The softmax turns scores into probabilities by exponentiating and
normalizing. Why exponentials? The conjugate of log-sum-exp,
$\mathrm{lse}(\mathbf{x}) = \log \sum_i e^{x_i}$, answers this. Let
$\Delta = \{\mathbf{p} : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$
be the probability simplex and $H(\mathbf{p}) = -\sum_i p_i \log p_i$ the
entropy.

We compute
$\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})$.
If $\mathbf{p}$ has a negative coordinate, sending that $x_i \to -\infty$
makes the objective unbounded. If $\mathbf{1}^\top\mathbf{p} \neq 1$, moving
$\mathbf{x} = c\mathbf{1}$ toward $\pm\infty$ does the same, because
$\mathrm{lse}(c\mathbf{1}) = c + \log n$. So the supremum is finite only on
the simplex. There, stationarity demands
$\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$, solved by $\mathbf{x} = \log\mathbf{p}$,
and the value is $\sum_i p_i \log p_i$.

The stationarity argument covers $\mathbf{p}$ with all coordinates positive,
since softmax outputs are strictly positive. For $\mathbf{p}$ on the boundary
of the simplex, with some $p_i = 0$, no maximizer exists. Sending those
$x_i \to -\infty$ still approaches the same value, so the formula holds on
the whole simplex with the convention $0 \log 0 = 0$. Because $\mathrm{lse}$
is convex and finite, $\mathrm{lse}^{**} = \mathrm{lse}$, and

$$
\begin{aligned}
\mathrm{lse}^*(\mathbf{p}) &=
\begin{cases}
\sum_i p_i \log p_i = -H(\mathbf{p}), & \mathbf{p} \in \Delta, \\
+\infty, & \textrm{otherwise,}
\end{cases} \\
\textrm{and dually} \qquad
\mathrm{lse}(\mathbf{x}) &= \max_{\mathbf{p} \in \Delta} \left( \mathbf{p}^\top \mathbf{x} + H(\mathbf{p}) \right).
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-lse-entropy`

Log-sum-exp is the largest value of a linear score plus an entropy bonus over
all distributions on $n$ outcomes. The maximizer is the softmax. By the
equality case of Fenchel--Young it is the $\mathbf{p}$ with
$\mathbf{p} = \nabla\mathrm{lse}(\mathbf{x})$, and differentiating gives
$\partial\,\mathrm{lse}/\partial x_i = e^{x_i} / \sum_j e^{x_j}$, the softmax
(the Hessian is computed in :numref:`sec_mdl-convexity`). Equivalently, among
all distributions with the same expected score $\mathbf{p}^\top\mathbf{x}$, it
has the largest entropy, the maximum-entropy property proved in
:numref:`subsec_mdl-max-entropy-derivation`. Exercise
:exref:`ex_mdl-lagrangian-duality-max-entropy` derives the softmax from this
constraint side. Without the entropy term the maximizer of
$\mathbf{p}^\top\mathbf{x}$ is a one-hot vector at $\arg\max_i x_i$. The
entropy replaces this hard selection by a distribution. The same
construction, applied to the convex function that defines an
$f$-divergence, gives the variational representations of divergences used to
train adversarial models (:numref:`sec_mdl-f-gan-dual`).

## Worked Examples
:label:`subsec_mdl-worked-duals`

Three examples show what the dual exposes. In the support vector machine,
complementary slackness makes the solution depend on a few data points, and
the dual sees the data only through inner products. Water-filling is the
standard example of a multiplier with units an engineer cares about, rate
per unit of power. The last example shows the gap that convexity prevents.

### The Support Vector Machine Dual
:label:`subsec_mdl-svm-dual`

Given linearly separable data $(\mathbf{x}_i, y_i)$ with labels
$y_i \in \{\pm 1\}$, the support vector machine :cite:`Cortes.Vapnik.1995`
looks for the separating hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ with
the widest margin. It solves

$$
\min_{\mathbf{w}, b}\; \tfrac12 \|\mathbf{w}\|^2
\quad \textrm{subject to} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \;\; i = 1, \ldots, n.
$$
:eqlabel:`eq_mdl-opt-svm-primal`

A point $\mathbf{x}$ lies at distance $|\mathbf{w}^\top\mathbf{x} + b| / \|\mathbf{w}\|$
from the hyperplane (the signed-distance formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops`), so the two margin
hyperplanes $\mathbf{w}^\top\mathbf{x} + b = \pm 1$ sit at distance
$1/\|\mathbf{w}\|$ on either side of the boundary, $2/\|\mathbf{w}\|$ apart.
Minimizing $\|\mathbf{w}\|$ maximizes the margin. This is a convex quadratic
program with one inequality per data point. The constraints are affine, so
Slater's condition needs only a feasible point, and scaling any separating
$(\mathbf{w}, b)$ up until every margin is at least $1$ gives one. So
strong duality is guaranteed. Form the Lagrangian with
multipliers $\alpha_i \ge 0$,

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
= \tfrac12\|\mathbf{w}\|^2 + \sum_i \alpha_i \left(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\right),
$$

and compute the dual function
$q(\boldsymbol{\alpha}) = \inf_{\mathbf{w}, b} \mathcal{L}$. The Lagrangian is
affine in $b$, with coefficient $-\sum_i \alpha_i y_i$. Unless that
coefficient vanishes, sending $b$ to $\pm\infty$ drives $\mathcal{L}$ to
$-\infty$, so $q(\boldsymbol{\alpha}) = -\infty$ whenever
$\sum_i \alpha_i y_i \neq 0$. This is the effective domain of
:eqref:`eq_mdl-opt-dual-function` at work: the condition
$\sum_i \alpha_i y_i = 0$ reappears as an equality constraint of the dual. On
that domain the $b$-term drops out, leaving a strictly convex quadratic in
$\mathbf{w}$ whose infimum is attained at its stationary point,

$$
\mathbf{w} = \sum_i \alpha_i y_i\, \mathbf{x}_i,
\qquad\textrm{on the domain}\qquad
\sum_i \alpha_i y_i = 0.
$$

The weight vector is a combination of the data points, with the multipliers
as coefficients. Substituting back yields the **SVM dual**:

$$
\max_{\boldsymbol{\alpha} \succeq 0,\ \sum_i \alpha_i y_i = 0}\;\;
\sum_i \alpha_i \;-\; \tfrac12 \sum_{i,j} \alpha_i \alpha_j\, y_i y_j\, \mathbf{x}_i^\top \mathbf{x}_j.
$$
:eqlabel:`eq_mdl-opt-svm-dual`

Two consequences hold before any algorithm is chosen. First, complementary
slackness, $\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$, says that
a positive multiplier forces the margin of point $i$ to equal $1$. Such
points are the **support vectors**, and they lie on the margin hyperplanes. In
the terms of :numref:`fig_mdl-opt-kkt-active-set`, their constraints are
active, and only their constraint normals enter the stationarity balance. Points with
$\alpha_i = 0$ do not contribute to the weight vector. Second, the data enter
:eqref:`eq_mdl-opt-svm-dual` only through the inner products
$\mathbf{x}_i^\top\mathbf{x}_j$. Replacing them by a kernel evaluation
$k(\mathbf{x}_i, \mathbf{x}_j)$ gives a nonlinear classifier, an option the
primal formulation does not expose directly.

The equality constraint $\sum_i \alpha_i y_i = 0$ entered only because the
unpenalized offset $b$ made the Lagrangian affine in one direction. To keep
the dual feasible set an orthant, on which projection clips each coordinate
at zero, the cell below solves a variant with the offset folded into the
weights: with a constant feature $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ and
$\tilde{\mathbf{w}} = (\mathbf{w}, b)$, minimize
$\tfrac12\|\tilde{\mathbf{w}}\|^2$ subject to
$y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$. This is a different
problem, since it also penalizes the offset, and its boundary can differ
slightly from that of :eqref:`eq_mdl-opt-svm-primal`; but its Lagrangian is
strictly convex in every primal variable, so no equality constraint
appears. Writing
$\mathbf{Q}_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$, the dual is

$$
\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top \mathbf{Q}\, \boldsymbol{\alpha},
$$
:eqlabel:`eq_mdl-opt-svm-dual-folded`

a concave quadratic over the orthant. Projected gradient ascent, the
update of :numref:`subsec_mdl-projected-gd` with the sign of the
step reversed, iterates
$\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - \mathbf{Q}\boldsymbol{\alpha}))$
with $\eta = 1/\lambda_{\max}(\mathbf{Q})$, the descent-lemma step for a
gradient that is $\lambda_{\max}(\mathbf{Q})$-Lipschitz. At its limit three
things should hold: the hyperplane separates the data, the positive
multipliers sit on points of margin $1$, and the dual value meets the primal
value. The cell runs the ascent on eight points in the plane.

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

All three hold. Four multipliers are positive, and those four points have
margin $1$; the other four sit at margins $1.29$ to
$2.43$ with $\alpha_i = 0$. The printed digits identify the exact solution,
$\tilde{\mathbf{w}}^\star = (\tfrac47, \tfrac47, -\tfrac37)$, that is,
$\mathbf{w}^\star = (\tfrac47, \tfrac47)$ and $b^\star = -\tfrac37$, with
$p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$ and a gap near
$10^{-16}$. The equal values also follow from an identity,
$\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ at the optimum
(Exercise :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`a).

The printout also shows a limit of reading the dual. The primal solution
$\tilde{\mathbf{w}}^\star$ is unique, because the primal objective is strictly
convex. The dual solution is not: $\mathbf{Q} = \mathbf{G}\mathbf{G}^\top$, with
$\mathbf{G}$ the $8 \times 3$ matrix of rows $y_i\tilde{\mathbf{x}}_i^\top$, has
rank $3$. The four rows of the support vectors are four vectors in
$\mathbb{R}^3$, so some nonzero $\mathbf{d}$ supported on them has
$\mathbf{G}^\top\mathbf{d} = \mathbf{0}$. Along $\mathbf{d}$ the quadratic term
of the dual objective does not change, and neither does the linear term:
$\mathbf{1}^\top\mathbf{d} = \sum_i d_i\, y_i\tilde{\mathbf{w}}^{\star\top}\tilde{\mathbf{x}}_i = \tilde{\mathbf{w}}^{\star\top}\mathbf{G}^\top\mathbf{d} = 0$,
because every support vector has margin $1$. Moving
$\boldsymbol{\alpha}^\star$ along $\mathbf{d}$ therefore keeps
$\tilde{\mathbf{w}}^\star$, the dual value, and feasibility until a coordinate
reaches zero. The optimal
multipliers form a segment, and projected ascent from $\mathbf{0}$ returns one
point of it. At either end of the segment, one of the four margin-$1$ points
has $\alpha_i = 0$ (Exercise :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`c).
Complementary slackness works in one direction: $\alpha_i > 0$ implies margin
$1$, but margin $1$ does not imply $\alpha_i > 0$.

### Water-Filling

The second example has a closed-form solution with a geometric picture. A
transmitter splits a power budget $P$ across $n$ independent channels.
Channel $i$ has noise power $N_i > 0$, and with power $P_i$ it carries
$\log(1 + P_i / N_i)$ nats per use :cite:`Cover.Thomas.1999`. (Cover and
Thomas write $\tfrac12\log$ for a real Gaussian channel; the factor rescales
the multiplier below but not the allocation.) The allocation problem is

$$
\max_{\mathbf{P}}\; U(\mathbf{P}) = \sum_{i=1}^n \log\left(1 + \frac{P_i}{N_i}\right)
\quad \textrm{subject to} \quad P_i \ge 0, \;\; \sum_i P_i = P,
$$
:eqlabel:`eq_mdl-opt-waterfilling`

a concave objective maximized under affine constraints, so minimizing $-U$
is a convex problem. Slater's condition holds because the constraints are
affine and a positive allocation exists, so the KKT conditions characterize
the optimum. With multiplier $\nu$ for the budget and $\lambda_i \ge 0$ for
$P_i \ge 0$, stationarity of the Lagrangian of $-U$ reads

$$
\frac{1}{N_i + P_i} = \nu - \lambda_i .
$$

On a channel with $P_i > 0$, complementary slackness sets $\lambda_i = 0$ and
forces $N_i + P_i = 1/\nu$, the same value for every channel that receives
power. On a channel with $P_i = 0$, dual feasibility requires
$\lambda_i = \nu - 1/N_i \ge 0$, that is, $N_i \ge 1/\nu$. Writing
$\theta = 1/\nu$ for the common value,

$$
P_i^\star = \max(\theta - N_i,\, 0),
\qquad \textrm{with } \theta \textrm{ set by } \sum_i \max(\theta - N_i,\, 0) = P.
$$

This is the **water-filling** solution. Draw each channel as a basin with
floor at height $N_i$ and pour in $P$ units of water. As
:numref:`fig_mdl-opt-water-filling` shows, the water settles at a common level
$\theta$, fills the quiet channels most, and never reaches basins whose floor
is above the level. The left side of the level equation is continuous,
piecewise linear, zero for $\theta \le \min_i N_i$, and strictly increasing
for $\theta > \min_i N_i$, so for $P > 0$ it has exactly one root, and
bisection finds it. The simplex projection of :numref:`subsec_mdl-projected-gd`
has the same threshold structure upside down: there a common level is
subtracted from the coordinates and the result is clipped at zero.

![Water-filling with the cell's noise floors and budget: pouring $P=3$ units of power into basins with floors at $N_i$ fills the three quiet channels to the common level $\theta=1/\nu^\star\approx1.43$, allocating $P_i^\star=\theta-N_i$ to each; the two basins whose floors exceed the level receive nothing, as complementary slackness requires.](../img/mdl-opt-water-filling.svg)
:label:`fig_mdl-opt-water-filling`

The shadow-price proposition gives the multiplier its engineering meaning.
Maximizing $U$ is minimizing $-U$, and relaxing the budget to $P + v$ changes
$-U^\star$ at rate $-\nu^\star$, so
$\mathrm{d}U^\star / \mathrm{d}P = \nu^\star = 1/\theta$: the extra rate from
one more unit of power. The cell below computes the allocation for five
channels and checks the price by re-solving at $P \pm 10^{-4}$.

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

The budget is met to the printed precision, the three wet channels sit at
the same level $\theta = 1.4333$, and the channels with floors $1.6$ and $2.5$
stay dry, as dual feasibility demands. Here the root is exact:
$\theta = (3 + 0.1 + 0.4 + 0.8)/3$. The multiplier $\nu^\star = 1/\theta = 0.697674$
matches the finite-difference sensitivity of the optimal rate to the budget
in all six printed digits.

### An Explicit Duality Gap
:label:`subsec_mdl-explicit-duality-gap`

Weak duality always holds; strong duality need not. A one-dimensional problem
shows a gap. On the domain $\mathcal{D} = [0, 1]$,

$$
\min_{x \in [0, 1]}\; f(x) = -x^2
\quad \textrm{subject to} \quad g(x) = x - \tfrac12 \le 0 .
$$

The objective is concave, so the problem is not convex. The feasible set is
$[0, \tfrac12]$, and the optimum is at its right end,
$p^\star = f(\tfrac12) = -\tfrac14$. Only the constraint $g(x) \le 0$ is
dualized: the domain $[0, 1]$ stays a hard constraint, so the infimum in
:eqref:`eq_mdl-opt-dual-function` runs over $[0, 1]$, and the proof of weak
duality goes through unchanged. The Lagrangian
$\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ is concave in $x$, so
its minimum over $[0, 1]$ is at an endpoint:
$\mathcal{L}(0, \lambda) = -\lambda/2$ and $\mathcal{L}(1, \lambda) = \lambda/2 - 1$.
Hence

$$
q(\lambda) = \min\!\left(-\frac{\lambda}{2},\; \frac{\lambda}{2} - 1\right),
$$

piecewise linear and concave, as it must be. The two pieces cross at
$\lambda^\star = 1$, where $d^\star = -\tfrac12 < -\tfrac14 = p^\star$: a
duality gap of $\tfrac14$. The cell evaluates $p^\star$ and $q$ on grids and
plots the dual function against $p^\star$, the counterpart of the first plot
of this section.

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

Where the curve of the clipping problem touched its line, the tent peaks at
$-\tfrac12$, a quarter below the primal optimum. A strictly feasible point
exists ($x = \tfrac14$ lies inside $\mathcal{D}$ and has
$g(\tfrac14) = -\tfrac14 < 0$), so the gap is caused by the missing convexity,
not by a missing Slater point. In panel (b) of
:numref:`fig_mdl-opt-primal-dual-gap`, the image curve
$\{(u, -(u + \tfrac12)^2) : u \in [-\tfrac12, \tfrac12]\}$ bends the wrong way,
and no line below $\mathcal{A}$ passes through its point at $u = 0$; the best
one, of slope $-1$, certifies only $-\tfrac12$. The gap also depends on how the
problem is written (Exercise
:exref:`ex_mdl-lagrangian-duality-gap-formulations`).

Two questions remain open. A nonconvex problem can keep a gap, so its dual
bounds but does not solve it. And iterating on $\mathbf{x}$ and
$\boldsymbol{\lambda}$ together need not converge even when a saddle point
exists, which is the difficulty behind minimax training. Training a deep
network is not a convex problem, but many of its subproblems are: projections,
clipped updates, last-layer fits, and trust-region steps on a convex model, and
everything in this section applies to them. All of it also assumed exact
arithmetic; the next section asks when finite precision, rather than the
algorithm, decides the answer.

## Summary

The dual function $q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}$
is concave for every problem and never exceeds $p^\star$, so each choice of
multipliers $\boldsymbol{\lambda} \succeq 0$ certifies a lower bound. For a
convex problem with a strictly feasible point (Slater's condition) the best
bound is exact, and attained primal and dual optima form a saddle point of
the Lagrangian. For linear constraints the dual is a conjugate,
$q(\boldsymbol{\lambda}) = -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}$,
which turns a table of conjugates into a table of duals. Under strong
duality the optimal multipliers are shadow prices, with
$\lambda_i^\star = -\partial p^\star / \partial u_i$ and
$\nu_j^\star = -\partial p^\star / \partial v_j$ when $p^\star$ is
differentiable. The SVM multipliers are positive only on support vectors,
the multiplier $1/\theta$ of the power budget in water-filling is the rate
gained per unit of power, and for a convex loss the weight-decay coefficient
is the multiplier of a norm budget. A nonconvex problem can keep a gap, as the concave objective with
$p^\star - d^\star = \tfrac14$ shows.

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

1. [code] **Duality gaps and formulations.** The gap cell of
   :numref:`subsec_mdl-explicit-duality-gap` minimizes the Lagrangian over
   the domain $[0, 1]$ and dualizes only the constraint $x \le \tfrac12$.
    1. Replace the objective by $f(x) = (x - 1)^2$. Compute $p^\star$, the
       dual function, $d^\star$, and $\lambda^\star$ by hand, and confirm them
       with the modified cell. Why is the gap now zero?
    1. Keep $f(x) = -x^2$, but write the constraint as $x^2 \le \tfrac14$,
       which defines the same feasible set on $[0, 1]$. Compute the new dual
       function and $d^\star$ by hand and with the cell.
    1. Keep $f(x) = -x^2$ and the constraint $x \le \tfrac12$, but treat
       $0 \le x$ and $x \le 1$ as constraints with their own multipliers and
       minimize the Lagrangian over all of $\mathbb{R}$. What is $d^\star$
       now? Rank the dual bounds of the three formulations of the same
       problem (the one in the text and those of the two previous parts). Is
       a duality gap a property of the problem or of its formulation?
   :exlabel:`ex_mdl-lagrangian-duality-gap-formulations`

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

1. [code] **Dual ascent and the gradient of the dual.** For the clipping
   problem, $q(\lambda) = \tfrac{25}{2}\tfrac{\lambda}{1 + \lambda} - \tfrac\lambda2$
   and the Lagrangian's minimizer is $\mathbf{x}(\lambda) = \mathbf{x}_0 / (1 + \lambda)$.
    1. Show that $q'(\lambda) = g(\mathbf{x}(\lambda))$. Explain why the same
       holds for any problem whose Lagrangian has a unique minimizer
       $\mathbf{x}(\boldsymbol{\lambda})$ that depends smoothly on
       $\boldsymbol{\lambda}$: differentiate
       $q(\boldsymbol{\lambda}) = \mathcal{L}(\mathbf{x}(\boldsymbol{\lambda}), \boldsymbol{\lambda})$
       and use stationarity to drop the term through $\mathbf{x}$. The
       gradient of the dual is the vector of constraint values at the
       Lagrangian's minimizer.
    1. Run projected dual ascent,
       $\lambda \leftarrow \max\bigl(0, \lambda + \eta\, g(\mathbf{x}(\lambda))\bigr)$,
       from $\lambda = 0$ with $\eta \in \{0.5, 1, 2\}$, and report the
       iterate after $10$, $50$, and $100$ steps. (For $\eta = 0.5$ you
       should see about $5.07$, $4.02$, and $4.0001$.) The first step jumps to
       $12\eta$, far past $\lambda^\star = 4$. Explain this from
       $q''(\lambda) = -25/(1 + \lambda)^3$, which is $-25$ at $\lambda = 0$
       and $-0.2$ at $\lambda = 4$. Why does a step size that is safe near
       $\lambda^\star$ jump so far at the start?
    1. For $\eta = 0.5$, are the points $\mathbf{x}(\lambda_k)$ feasible? Show
       that every feasible $\mathbf{x}(\lambda_k)$ gives an upper bound
       $f(\mathbf{x}(\lambda_k)) \ge p^\star$ while $q(\lambda_k)$ gives a lower
       bound, and print the certified gap $f(\mathbf{x}(\lambda_k)) - q(\lambda_k)$
       along the run. Relate dual ascent to the Sinkhorn updates of
       Exercise :exref:`ex_mdl-lagrangian-duality-entropic-ot`, which maximize
       the same kind of dual exactly in blocks.
   :exlabel:`ex_mdl-lagrangian-duality-dual-ascent`

1. **The chapter's quadratic against a half-plane.** Minimize
   $\tfrac12(x_1^2 + 1000\,x_2^2)$ subject to $x_1 + x_2 \ge 1$.
    1. Write the constraint as $-x_1 - x_2 \le -1$ and use
       :eqref:`eq_mdl-opt-dual-conjugate` with the conjugate of
       $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ from
       Exercise :exref:`ex_mdl-convexity-conjugate-pairs`b to get
       $q(\lambda) = \lambda - \tfrac12\lambda^2(1 + \tfrac1{1000})$. Find
       $\lambda^\star = \tfrac{1000}{1001}$,
       $p^\star = d^\star = \tfrac{500}{1001} \approx 0.4995$, and
       $\mathbf{x}^\star = (0.999, 0.000999)$.
    1. Why does $x_2^\star$ carry almost none of the constraint? By how much,
       to first order, does the optimum rise if the constraint becomes
       $x_1 + x_2 \ge 1 + \delta$?
    1. Replace $1000$ by a curvature ratio $\kappa$ and let $\kappa \to \infty$.
       What do $\lambda^\star$, $p^\star$, and $\mathbf{x}^\star$ tend to? Fixed-step
       gradient descent on the unconstrained quadratic slows down by a factor
       of about $\kappa$ (:numref:`sec_mdl-gradient-based-optimization`).
       Does the ill-conditioning make this constrained problem hard to solve
       through its dual?

[Discussions](https://d2l.discourse.group/t/36860)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.6]{.kicker}

Duality<br>**lower bounds · the conjugate · strong duality · prices · worked duals**
:::
:::

::: {.slide title="A Solver Says 8.01. Is Anything Feasible Below 8?"}
[The problem]{.kicker}

- Clip $\mathbf{x}_0 = (3, 4)$ to the unit disk: minimize
  $f(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ s.t. $\|\mathbf{x}\| \le 1$.
- A solver stops at $(0.55, 0.835)$: feasible, $f = 8.01$.
- At the true answer $(0.6, 0.8)$ the gradient is $(-2.4, -3.2) \neq \mathbf{0}$.

::: {.d2l-note .warn}
The zero-gradient test cannot certify a constrained optimum, and the disk
has infinitely many points to check.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute a dual function and read any of its values as a lower bound;
- compute a conjugate and the dual of a linearly constrained problem in one line;
- decide from convexity and one strictly feasible point that the bound is exact;
- read a multiplier as the price of a constraint;
- derive the SVM dual and the water-filling rule.

**The idea:** add $\lambda \ge 0$ times the constraint to the objective; every
$\lambda$ gives a lower bound, and the best one is the optimum.
**Running example:** clip $(3, 4)$ to the unit disk.
:::

::: {.slide title="Duality Proves That the KKT Point Is Optimal"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gradient Descent and Curvature | safe steps, $\kappa$, momentum |
| Curvature Methods | Newton, BFGS, trust regions |
| Convexity | local = global, rates, Jensen |
| Stochastic Gradient Methods | noise floors, step decay |
| Adaptive Methods | per-coordinate steps |
| Constrained Optimization | KKT, projections, prox |
| **Duality** | **lower bounds, prices** |
| Numerical Stability and Conditioning | rounding, conditioning |
:::

::: {.col .narrow}
- **Uses:** KKT conditions, convexity, projection.
- **Skips:** interior-point solvers; nonconvex duality beyond one example.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Lower bounds]{.dtitle}

[one multiplier, one bound]{.dsub}
:::
:::

::: {.slide title="Adding λ Times the Constraint Gives a Lower Bound"}
[Lower bounds]{.kicker}

- Constraint $g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - 1) \le 0$; take $\lambda \ge 0$.
- Feasible $\mathbf{x}$: $f + \lambda g \le f$, so
  $\min_{\mathbf{x}\in\mathbb{R}^2} (f + \lambda g) \le p^\star = 8$.
- Minimizer $\mathbf{x}(\lambda) = \mathbf{x}_0/(1 + \lambda)$, value
  $q(\lambda) = \tfrac{25}{2}\tfrac{\lambda}{1+\lambda} - \tfrac\lambda2$.

| $\lambda$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $q(\lambda)$ | 5.75 | 7.33 | 7.875 | **8** | 7.92 |

::: {.d2l-note .rule}
Every entry is a bound: $8.01$ is within $0.01$ of optimal.
:::
:::

::: {.slide title="The Dual Function Is Concave and Never Exceeds p*"}
[Lower bounds]{.kicker}

::: {.cols .vc}
::: {.col}
For $\min f$ s.t. $g_i \le 0$, $h_j = 0$:

$$q(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \Bigl[f + \textstyle\sum_i \lambda_i g_i + \sum_j \nu_j h_j\Bigr]$$

- Feasible $\mathbf{x}$, $\boldsymbol{\lambda} \succeq 0$:
  $q \le f + \sum\lambda_i g_i + \sum\nu_j h_j \le f(\mathbf{x})$.
- Infimum of functions affine in the multipliers: concave.
- Dual problem $\max_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} q$: convex,
  value $d^\star \le p^\star$.
:::

::: {.col}
@!mdl-lagrangian-duality-a-lower-bound-from-one-multiplier
:::
:::
:::

::: {.slide title="Check: What Does λ = 3 Certify?"}
[Check]{.kicker}

1. What does $q(\lambda)$ certify for an arbitrary $\lambda \ge 0$?
2. With $q(3) = 7.875$, how far from optimal can the solver's $8.01$ be?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) No feasible point has $f < q(\lambda)$. (2) At most
$8.01 - 7.875 = 0.135$; $\lambda = 4$ tightens this to $0.01$.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Strong duality and prices]{.dtitle}

[when the bound is exact, and what it is worth]{.dsub}
:::
:::

::: {.slide title="Convexity Plus One Strictly Feasible Point Closes the Gap"}
[Strong duality]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
$\mathcal{A}$: all $(u, t)$ above and right of some $(g(\mathbf{x}), f(\mathbf{x}))$.
$q(\lambda)$ is the height at $u = 0$ of the lowest line of slope $-\lambda$ below
$\mathcal{A}$. Left: clipping, $d^\star = p^\star = 8$. Right: concave objective, gap.
:::

::: {.col .fig .big}
@fig:mdl-opt-primal-dual-gap
:::
:::

[Slater: $f, g_i$ convex, $h_j$ affine, and some feasible $\bar{\mathbf{x}}$ with $g_i(\bar{\mathbf{x}}) < 0$ for every $g_i$ that is not affine.]{.small}
:::

::: {.slide title="Strong Duality Is a Saddle Point of the Lagrangian"}
[Strong duality]{.kicker}

- $p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0} \mathcal{L}$
  (a violated constraint makes the sup $+\infty$);
  $d^\star = \sup_{\boldsymbol{\lambda} \succeq 0} \inf_{\mathbf{x}} \mathcal{L}$.
- Strong duality with attained optima:
  $\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}) \le \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star) \le \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star)$.
- $\mathbf{x}^\star$ is among the minimizers of $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star)$;
  unique for strictly convex $f$: $\mathbf{x}(4) = \mathbf{x}_0/5 = (0.6, 0.8)$.

::: {.d2l-note}
GAN objectives have this min--max form. Simultaneous descent--ascent on
$xy$ spirals away from the saddle at the origin for every step size.
:::
:::

::: {.slide title="The Multiplier Is the Price of the Constraint"}
[Prices]{.kicker}

- Relax: $p^\star(\mathbf{u}, \mathbf{v}) = \inf\{f : g_i \le u_i,\ h_j = v_j\}$.
- $p^\star(\mathbf{u}, \mathbf{v}) \ge p^\star(\mathbf{0}, \mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u} - \boldsymbol{\nu}^{\star\top}\mathbf{v}$;
  if differentiable, $\lambda_i^\star = -\partial p^\star / \partial u_i$.
- Clipping relaxed to $\tfrac12(\|\mathbf{x}\|^2 - 1) \le u$, a disk of radius $r = \sqrt{1 + 2u}$:
  $p^\star(u) = \tfrac12(5 - \sqrt{1 + 2u})^2$, slope $-4$ at $u = 0$.
- $\lambda^\star = 4$ prices $u$, half the squared radius, not $r$; slack constraints cost $0$.

[Hypotheses: strong duality; for the derivative, $p^\star$ differentiable.]{.small}
:::

::: {.slide title="Weight Decay Is the Price of a Norm Budget"}
[Prices]{.kicker}

$\min \ell + \tfrac\lambda2\|\mathbf{w}\|^2$ and $\min \ell$ s.t.
$\tfrac12\|\mathbf{w}\|^2 \le \tfrac12 r^2$ share
$\nabla\ell + \lambda\mathbf{w} = \mathbf{0}$; for convex $\ell$ they have
the same solutions. Ridge, $r = \|\mathbf{w}_\lambda\|$, PGD on the ball:

@!constrained-weight-decay

::: {.d2l-note}
One $\lambda$, three readings: decay rate, multiplier, $\sigma^2/\tau^2$ of a
Gaussian prior.
:::
:::

::: {.slide title="Check: What Is a Disk of Radius √2 Worth?"}
[Check]{.kicker}

Clipping $(3, 4)$ to the unit disk has $p^\star = 8$ and $\lambda^\star = 4$.
Enlarge the disk to radius $\sqrt2$, that is, $u = \tfrac12$. What does the
price guarantee, and what is the new optimum?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The price guarantees $p^\star(\tfrac12) \ge 8 - 4 \cdot \tfrac12 = 6$.
The new optimum is $\tfrac12(5 - \sqrt2)^2 = 6.43$: relaxing helps by less
than the price times the amount, because the price falls as the disk grows.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The conjugate]{.dtitle}

[duals read off a table]{.dsub}
:::
:::

::: {.slide title="The Conjugate Records How Far a Line Must Drop to Touch the Graph"}
[The conjugate]{.kicker}

$f^*(\mathbf{y}) = \sup_{\mathbf{x}} \bigl(\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})\bigr)$:
minus the intercept of the highest line of slope $\mathbf{y}$ below $f$.

::: {.cols .vc}
::: {.col .narrow}
$\tfrac12x^2$ at slope $2$: touches at $x = 2$, $f^*(2) = 2$.
$|x|$: $f^* = 0$ on $[-1, 1]$, $+\infty$ outside.
:::

::: {.col .fig .big}
@fig:mdl-opt-conjugate-intercept
:::
:::
:::

::: {.slide title="For Linear Constraints the Dual Is a Conjugate"}
[The conjugate]{.kicker}

$\min f(\mathbf{x})$ s.t. $\mathbf{A}\mathbf{x} \preceq \mathbf{b}$:

$$q(\boldsymbol{\lambda}) = \inf_{\mathbf{x}}\bigl[f + (\mathbf{A}^\top\boldsymbol{\lambda})^\top\mathbf{x}\bigr] - \mathbf{b}^\top\boldsymbol{\lambda}
= -f^*(-\mathbf{A}^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda}.$$

- Half-plane: $f = \tfrac12\|\mathbf{x}\|^2$, $f^* = \tfrac12\|\mathbf{y}\|^2$,
  $\mathbf{a} = (1, 2)$, $b = -1$.
- $q(\lambda) = \lambda - \tfrac52\lambda^2$: $\lambda^\star = \tfrac15$,
  $d^\star = \tfrac1{10}$.
- $\mathbf{x} = -\lambda^\star\mathbf{a} = (-0.2, -0.4)$: feasible, $f = \tfrac1{10}$.

::: {.d2l-note}
One dual variable instead of two primal ones; the answer in one line.
:::
:::

::: {.slide title="Log-Sum-Exp and Negative Entropy Are Conjugates"}
[The conjugate]{.kicker}

- $\mathrm{lse}^*(\mathbf{p}) = \sum_i p_i \log p_i$ on the simplex $\Delta$,
  $+\infty$ off it.
- $\mathrm{lse}(\mathbf{x}) = \max_{\mathbf{p} \in \Delta}\, \mathbf{p}^\top\mathbf{x} + H(\mathbf{p})$,
  maximizer $\mathrm{softmax}(\mathbf{x})$.

::: {.d2l-note .rule}
Softmax is the entropy-regularized argmax.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Worked examples]{.dtitle}

[SVM · water-filling · a gap]{.dsub}
:::
:::

::: {.slide title="Support Vectors Are the Points with Positive Multipliers"}
[Worked examples]{.kicker}

Primal $\min\tfrac12\|\tilde{\mathbf{w}}\|^2$ s.t.
$y_i\tilde{\mathbf{w}}^\top\tilde{\mathbf{x}}_i \ge 1$ (offset folded in);
dual $\max_{\boldsymbol{\alpha} \succeq 0} \mathbf{1}^\top\boldsymbol{\alpha} - \tfrac12\boldsymbol{\alpha}^\top\mathbf{Q}\boldsymbol{\alpha}$,
$\mathbf{Q}_{ij} = y_iy_j\tilde{\mathbf{x}}_i^\top\tilde{\mathbf{x}}_j$; projected ascent:

@!constrained-svm-dual

::: {.d2l-note}
$\tilde{\mathbf{w}}^\star$ is unique; $\boldsymbol{\alpha}^\star$ is not
(rank $3$): a segment of optimal multipliers.
:::
:::

::: {.slide title="Water Fills Every Channel to One Level"}
[Worked examples]{.kicker}

::: {.cols .vc}
::: {.col}
- $\max\sum_i\log(1 + P_i/N_i)$ s.t. $\sum_i P_i = P$, $P_i \ge 0$.
- KKT: $N_i + P_i = \theta = 1/\nu$ on channels with power.
- $P_i^\star = \max(\theta - N_i, 0)$.
- $\nu^\star = 1/\theta = 0.698 = \mathrm{d}U^\star/\mathrm{d}P$.
:::

::: {.col .fig .big}
@fig:mdl-opt-water-filling
:::
:::
:::

::: {.slide title="Check: Which Channels Get Power?"}
[Check]{.kicker}

Noise floors $(0.1, 0.4, 0.8, 1.6, 2.5)$ and budget $P = 3$. Which channels
receive power, and at what level?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Try the three quietest: $\theta = (3 + 0.1 + 0.4 + 0.8)/3 = 1.433$.
The floors $1.6$ and $2.5$ lie above it, so they stay dry, and the guess is
consistent.
:::
:::
:::

::: {.slide title="A Concave Objective Leaves a Gap of 1/4"}
[Worked examples]{.kicker}

::: {.cols .vc}
::: {.col}
$\min -x^2$ on $[0, 1]$ s.t. $x \le \tfrac12$:
$p^\star = -\tfrac14$, $d^\star = -\tfrac12$.

- A strictly feasible point exists; convexity does not.
- Duals of nonconvex problems bound but do not solve.
:::

::: {.col .fig}
@!constrained-duality-gap
:::
:::
:::

::: {.slide title="One Multiplier per Constraint Bounds, Certifies, and Prices"}
[Wrap-up]{.kicker}

- Clipping: $q(4) = 8 = p^\star$; price $4$ per unit of $\tfrac12 r^2$.
- Half-plane: $d^\star = \tfrac1{10}$ from the conjugate in one line.
- $\mathrm{lse}$ and $-H$ are conjugates; softmax attains the maximum.
- Slater closes the gap; a concave objective keeps $\tfrac14$.
- SVM: $\tfrac{41}{98}$, four positive multipliers; water-filling: level
  $1.433$, price $0.698$.

**Next:** when finite precision, not the algorithm, decides the answer.
:::
