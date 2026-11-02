# Constrained Optimization
:label:`sec_mdl-constrained-optimization-duality`

Constraints appear in projected and clipped updates, trust regions, support
vector machines, and regularization. This section develops equality and
inequality constraints, the Karush--Kuhn--Tucker conditions, projected gradient
descent, Lagrangian duality, and strong duality
:cite:`Boyd.Vandenberghe.2004`. Worked examples derive the SVM dual, solve a
water-filling problem by bisection, and exhibit a nonconvex duality gap.

At an unconstrained minimum the gradient vanishes. At a constrained minimum it
may be nonzero, but no feasible infinitesimal direction can decrease the
objective. Lagrange multipliers, the KKT conditions, and fixed points of
projected gradient descent express this condition for different constraint
sets.

We lean on :numref:`sec_mdl-multivariable_calculus` (gradients, level sets, and
the tangency condition used below), :numref:`sec_mdl-gradient-based-optimization`
(stationarity and the descent lemma), and :numref:`sec_mdl-convexity` (convex
sets and functions, required for KKT sufficiency and strong duality). The
standard reference is :citet:`Boyd.Vandenberghe.2004`, chapter 5;
:citet:`Nocedal.Wright.2006` gives the numerical view. The code in
this section is plain NumPy, because the algorithms have compact NumPy
implementations; we load the `d2l` module once for plotting.

```{.python .input #constrained-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #constrained-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #constrained-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #constrained-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## Equality Constraints and Lagrange Multipliers
:label:`subsec_mdl-lagrange-multipliers`

### Feasible Directions at an Equality-Constrained Minimum

Consider the simplest constrained problem: minimize a smooth
$f : \mathbb{R}^n \to \mathbb{R}$ over the surface defined by one smooth
equality constraint,

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad g(\mathbf{x}) = 0.
$$

The geometric argument requires the regularity condition
$\nabla g(\mathbf{x}^\star)\ne\mathbf{0}$. It ensures that the feasible set is
locally a smooth surface with a well-defined tangent space. Without it, the
tangency conclusion may fail; an adjacent counterexample follows the
proposition.

:numref:`sec_mdl-multivariable_calculus` supplies the geometric interpretation: the admissible
moves from a feasible point are, to first order, directions tangent to the
surface $\{g = 0\}$, and at a regular constrained optimum the level set of
$f$ is tangent to the constraint surface. :numref:`fig_mdl-opt-lagrange-tangency`
shows why tangency is forced. At a feasible point where a level curve of $f$
*crosses* the constraint curve, the gradient $\nabla f$ has a nonzero component
along the constraint, so moving along the constraint opposite that component
decreases $f$ while staying feasible. A feasible descent direction exists, and
the point cannot be optimal. Only where the level curve is *tangent* to the
constraint does every feasible direction become neutral to first order, and
tangency of the curves is parallelism of their normals: $\nabla f$ must be a
multiple of $\nabla g$.

![Minimizing $f$ along the constraint curve $g(\mathbf{x})=0$. At a non-optimal feasible point the gradient $\nabla f$ has a component along the constraint, so sliding along the curve descends. At the optimum $\mathbf{x}^\star$ the level set of $f$ is tangent to the constraint and the two gradients are parallel: $\nabla f = -\nu\,\nabla g$.](../img/mdl-opt-lagrange-tangency.svg)
:label:`fig_mdl-opt-lagrange-tangency`

**Proposition (Lagrange condition).** *Let $f$ and $g$ be continuously
differentiable, and let $\mathbf{x}^\star$ be a local minimum of $f$ on
$\{\mathbf{x} : g(\mathbf{x}) = 0\}$ at which $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$.
Then there is a unique scalar $\nu^\star$, called the **Lagrange multiplier**,
such that*

$$
\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla g(\mathbf{x}^\star) = \mathbf{0}.
$$
:eqlabel:`eq_mdl-opt-lagrange-condition`

**Proof.** The first-order feasible directions at $\mathbf{x}^\star$ are the
tangent directions $\mathbf{v}$ with $\nabla g(\mathbf{x}^\star)^\top\mathbf{v} = 0$.
Because $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$, the implicit function
theorem (stated in :numref:`subsec_mdl-implicit-diff`, and taken on faith at
this chapter's level of rigor) makes the feasible set a genuine smooth surface
near $\mathbf{x}^\star$:
every such $\mathbf{v}$ is the velocity of some curve $\mathbf{x}(t)$ that stays
on the surface with $\mathbf{x}(0) = \mathbf{x}^\star$,
$\dot{\mathbf{x}}(0) = \mathbf{v}$. Since $\mathbf{x}^\star$ minimizes $f$ along
each such curve, and the curve can be traversed in *both* directions $\pm\mathbf{v}$,
the derivative at $t = 0$ must vanish rather than merely be nonnegative:

$$
\left.\frac{d}{dt} f(\mathbf{x}(t))\right|_{t=0} = \nabla f(\mathbf{x}^\star)^\top \mathbf{v} = 0
\quad\textrm{for every } \mathbf{v} \perp \nabla g(\mathbf{x}^\star).
$$

So $\nabla f(\mathbf{x}^\star)$ is orthogonal to the entire tangent hyperplane
$\{\mathbf{v} : \nabla g^\top\mathbf{v} = 0\}$, whose orthogonal complement is
the line spanned by $\nabla g(\mathbf{x}^\star)$
(:numref:`sec_mdl-geometry-linear-algebraic-ops`). Hence
$\nabla f(\mathbf{x}^\star) = -\nu^\star \nabla g(\mathbf{x}^\star)$ for exactly
one scalar $\nu^\star$. $\blacksquare$

This is the tangency condition of :numref:`sec_mdl-multivariable_calculus`,
which wrote it as $\nabla f = \mu\,\nabla g$; the two conventions agree
with $\nu = -\mu$, and the plus sign here is the one the Lagrangian below
makes natural.

The hypothesis $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$ is called a
**constraint qualification**, and it can fail. Minimize $f(\mathbf{x}) = x_1$
subject to $g(\mathbf{x}) = x_1^2 + x_2^2 = 0$: the feasible set is the single
point $\mathbf{0}$, which is therefore the minimum, yet
$\nabla f(\mathbf{0}) = (1, 0)$ while $\nabla g(\mathbf{0}) = \mathbf{0}$: no
multiplier exists. The constraint set has collapsed to a point, the "surface" is
not a surface, so the tangent-space argument does not apply. Invoking
multipliers therefore requires verifying that the constraint
gradients do not degenerate at the optimum.

### The Lagrangian

The two optimality requirements (parallel gradients *and* feasibility)
combine into stationarity of a single function of more variables, the
**Lagrangian**

$$
\mathcal{L}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu\, g(\mathbf{x}),
$$
:eqlabel:`eq_mdl-opt-lagrangian-eq`

since

$$
\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \nu \nabla g = \mathbf{0}
\qquad\textrm{and}\qquad
\frac{\partial \mathcal{L}}{\partial \nu} = g(\mathbf{x}) = 0
$$

are exactly :eqref:`eq_mdl-opt-lagrange-condition` and the constraint. The constrained problem in $\mathbf{x}$ is represented as an unconstrained
stationarity problem
in $(\mathbf{x}, \nu)$; the multiplier is a *coordinate* of
the optimality condition. With several equality constraints
$h_j(\mathbf{x}) = 0$, $j = 1, \ldots, p$, each gets its own multiplier,
$\mathcal{L} = f + \sum_j \nu_j h_j$, and the same argument (now requiring the
$\nabla h_j(\mathbf{x}^\star)$ to be linearly independent) puts
$\nabla f(\mathbf{x}^\star)$ in the *span* of the constraint gradients: the
objective's gradient is balanced by a combination of the constraint normals,
with nothing left over along any feasible direction.

### Worked Example: Closest Point on a Hyperplane
:label:`subsec_mdl-closest-point`

Minimize $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{x}$ subject to
$h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$: the point of a hyperplane
nearest the origin. Stationarity of $\mathcal{L} = \mathbf{x}^\top\mathbf{x} + \nu(\mathbf{a}^\top\mathbf{x} - b)$
gives $2\mathbf{x} + \nu\mathbf{a} = \mathbf{0}$, so $\mathbf{x} = -\tfrac{\nu}{2}\mathbf{a}$:
the solution lies along the normal $\mathbf{a}$. Feasibility fixes the scale,
$\nu^\star = -2b/\|\mathbf{a}\|^2$, hence

$$
\mathbf{x}^\star = \frac{b}{\|\mathbf{a}\|^2}\,\mathbf{a},
$$

recovering the projection formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops`. The multiplier also has a sensitivity interpretation: the optimal value is $p^\star(b) = b^2/\|\mathbf{a}\|^2$,
and $\partial p^\star / \partial b = 2b/\|\mathbf{a}\|^2 = -\nu^\star$. The multiplier measures the derivative of the optimal value with respect to the
constraint parameter; this is called a shadow price. The general result appears in
:numref:`subsec_mdl-lagrangian-duality`.

## Inequality Constraints and the KKT Conditions
:label:`subsec_mdl-kkt-conditions`

### Active and Inactive Constraints

Most constraints in practice are one-sided: a norm *at most* $r$, a probability
*at least* $0$, a budget *not exceeded*. Consider

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m.
$$

At the optimum each inequality is in one of two regimes. Either
$g_i(\mathbf{x}^\star) < 0$: the constraint is **inactive**, the optimum lies
strictly inside its region, and locally the constraint might as well not exist.
Or $g_i(\mathbf{x}^\star) = 0$: the constraint is **active** and behaves
like an equality, and contributes to first-order stationarity.

A worked example shows both regimes at once. Project a point onto a ball:
minimize $\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ subject to
$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - r^2) \le 0$. Stationarity of
$\mathcal{L} = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \lambda g(\mathbf{x})$
gives $(\mathbf{x} - \mathbf{x}_0) + \lambda\mathbf{x} = \mathbf{0}$, i.e.
$\mathbf{x} = \mathbf{x}_0 / (1 + \lambda)$: the constrained solution is a radial rescaling of the input vector. If $\|\mathbf{x}_0\| \le r$ the constraint is inactive,
$\lambda^\star = 0$ and $\mathbf{x}^\star = \mathbf{x}_0$; if
$\|\mathbf{x}_0\| > r$ it is active, $\|\mathbf{x}^\star\| = r$ forces
$\lambda^\star = \|\mathbf{x}_0\|/r - 1 > 0$, and

$$
\mathbf{x}^\star = \frac{\mathbf{x}_0}{\max\left(1,\; \|\mathbf{x}_0\|/r\right)}.
$$

*Gradient clipping* applies this projection to the update vector :cite:`Pascanu.Mikolov.Bengio.2013`. And note the
sign of the active multiplier:
$\lambda^\star > 0$, never negative. The nonnegativity of an inequality multiplier distinguishes it from an
unrestricted equality multiplier and enforces the orientation of the active
constraint normal.

### The Karush--Kuhn--Tucker Conditions

Assembling the general problem with both kinds of constraints,

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad
g_i(\mathbf{x}) \le 0,\; i = 1, \ldots, m, \qquad h_j(\mathbf{x}) = 0,\; j = 1, \ldots, p,
$$
:eqlabel:`eq_mdl-opt-standard-problem`

the **Lagrangian** acquires a multiplier per constraint,

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= f(\mathbf{x}) + \sum_{i=1}^m \lambda_i\, g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j\, h_j(\mathbf{x}),
\qquad \lambda_i \ge 0,
$$
:eqlabel:`eq_mdl-opt-lagrangian`

and the first-order optimality conditions are the four
**Karush--Kuhn--Tucker (KKT) conditions** :cite:`Karush.1939,Kuhn.Tucker.1951`:

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

Stationarity balances the objective and constraint gradients; primal
feasibility restates the constraints; dual feasibility requires nonnegative
inequality multipliers. The fourth, **complementary slackness**, says that
for each constraint, *at least one of* $\lambda_i^\star$ and
$g_i(\mathbf{x}^\star)$ is zero: a constraint is either active
($g_i = 0$, multiplier free to be positive) or priced at zero
($\lambda_i = 0$, constraint slack and locally irrelevant). It identifies the active set: in a KKT solution, zero and nonzero multipliers
indicate which constraints contribute to stationarity. In the ball
projection, complementary slackness is precisely the case split we did by hand.

:numref:`fig_mdl-opt-kkt-active-set` draws stationarity plus dual feasibility:
at the optimum, $-\nabla f$ is a nonnegative combination of the active
constraints' outward normals. Equivalently, it belongs to the normal cone
generated by the active constraints; inactive constraints contribute nothing.

![Geometry of the KKT conditions on a feasible region cut out by two inequality constraints. At the optimum $\mathbf{x}^\star$ one constraint is active and one is inactive: $-\nabla f$ points along the active constraint's outward normal with $\lambda_1 > 0$, while the inactive constraint has $\lambda_2 = 0$. At a corner where two constraints are active, $-\nabla f$ must lie in the cone spanned by both normals (shaded: the normal cone). Orange circles are level sets of $f$ around its unconstrained minimizer (+).](../img/mdl-opt-kkt-active-set.svg)
:label:`fig_mdl-opt-kkt-active-set`

How strong are these conditions? In general they are **necessary**: multipliers
satisfying :eqref:`eq_mdl-opt-kkt` exist at any local minimum that satisfies a
constraint qualification (the standard one, *LICQ*, asks the active
constraints' gradients to be linearly independent, the inequality
analogue of $\nabla g \neq \mathbf{0}$ above); see
:citet:`Nocedal.Wright.2006`, chapter 12, for the proof. They are not
sufficient: a non-convex problem can have KKT points that
are saddles or maxima (Exercise :exref:`ex_mdl-constrained-optimization-duality-write-all-four-kkt`). Under convexity, the conditions are also sufficient, as the following proof
shows.

**Proposition (KKT sufficiency under convexity).** *In problem
:eqref:`eq_mdl-opt-standard-problem`, let $f$ and every $g_i$ be convex and
differentiable and every $h_j$ affine. If
$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$
satisfies the KKT conditions :eqref:`eq_mdl-opt-kkt`, then $\mathbf{x}^\star$ is
a global minimum.*

**Proof.** The function
$\varphi(\mathbf{x}) = \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$
is convex: it is a nonnegative combination of convex functions plus affine terms
(:numref:`sec_mdl-convexity`). Stationarity says
$\nabla\varphi(\mathbf{x}^\star) = \mathbf{0}$, so $\mathbf{x}^\star$ is a
*global* minimizer of $\varphi$. Now take any feasible $\mathbf{x}$. Since
$\lambda_i^\star \ge 0$, $g_i(\mathbf{x}) \le 0$, and $h_j(\mathbf{x}) = 0$,

$$
f(\mathbf{x}) \;\ge\; f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})
= \varphi(\mathbf{x}) \;\ge\; \varphi(\mathbf{x}^\star)
= f(\mathbf{x}^\star),
$$

where the last equality uses complementary slackness
($\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$) and feasibility
($h_j(\mathbf{x}^\star) = 0$). $\blacksquare$

For convex problems, the KKT system is necessary (under a
constraint qualification we will meet as Slater's condition) and sufficient.
This extends unconstrained first-order stationarity to constrained problems and
is used in the examples below.

## Projections and Projected Gradient Descent
:label:`subsec_mdl-projected-gd`

The KKT conditions characterize optima; projection provides a corresponding
algorithmic update. Take the ordinary gradient step,
and if it leaves the feasible set, project to the nearest feasible point.

### Projection onto a Convex Set

For a closed convex set $C \subseteq \mathbb{R}^n$, the **projection** of
$\mathbf{y}$ onto $C$ is the nearest feasible point,

$$
\Pi_C(\mathbf{y}) = \mathop{\mathrm{argmin}}_{\mathbf{x} \in C}\; \tfrac12 \|\mathbf{x} - \mathbf{y}\|^2.
$$

A minimizer exists whenever $C$ is nonempty: we may restrict the search to the
intersection of $C$ with a large closed ball around $\mathbf{y}$, which is
compact because $C$ is closed, and the continuous objective attains its minimum
there. It is unique because the objective is strictly convex. So
$\Pi_C$ is a well-defined map. Both examples we have computed are projections:
onto a hyperplane in :numref:`subsec_mdl-lagrange-multipliers`, onto a ball
(the clipping formula) in :numref:`subsec_mdl-kkt-conditions`.

Projections onto closed convex sets have a useful first-order
characterization. The
point $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$ minimizes the convex function
$\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ over the convex set $C$, and "no
feasible descent" over a convex set takes an especially simple form: for any
$\mathbf{x} \in C$ the whole segment from $\hat{\mathbf{x}}$ to $\mathbf{x}$ is
feasible, so the directional derivative along it must be nonnegative, giving
$(\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge 0$,
i.e.

$$
\left(\mathbf{y} - \Pi_C(\mathbf{y})\right)^\top \left(\mathbf{x} - \Pi_C(\mathbf{y})\right) \;\le\; 0
\qquad \textrm{for all } \mathbf{x} \in C.
$$
:eqlabel:`eq_mdl-opt-proj-variational`

Geometrically: the *residual* $\mathbf{y} - \Pi_C(\mathbf{y})$ makes an obtuse
angle with every feasible direction; the set $C$ lies entirely on the far
side of the supporting hyperplane through $\Pi_C(\mathbf{y})$ with normal
$\mathbf{y} - \Pi_C(\mathbf{y})$. The inequality is also *sufficient*: if
$\hat{\mathbf{x}} \in C$ satisfies it, apply the first-order convexity
condition :eqref:`eq_mdl-opt-first-order` to
$\varphi(\mathbf{x}) = \tfrac12\|\mathbf{x} - \mathbf{y}\|^2$, whose gradient
at $\hat{\mathbf{x}}$ is $\hat{\mathbf{x}} - \mathbf{y}$: for every
$\mathbf{x} \in C$,
$\varphi(\mathbf{x}) \ge \varphi(\hat{\mathbf{x}}) + (\hat{\mathbf{x}} - \mathbf{y})^\top(\mathbf{x} - \hat{\mathbf{x}}) \ge \varphi(\hat{\mathbf{x}})$,
so $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$. Thus
:eqref:`eq_mdl-opt-proj-variational` *characterizes* the projection. This inequality implies nonexpansiveness, which is used in projected-method
analysis.

**Proposition (projections are nonexpansive).** *For a closed convex set $C$
and all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,*

$$
\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\| \;\le\; \|\mathbf{x} - \mathbf{y}\|.
$$

**Proof.** Write $\mathbf{u} = \Pi_C(\mathbf{x})$, $\mathbf{v} = \Pi_C(\mathbf{y})$.
Apply :eqref:`eq_mdl-opt-proj-variational` twice: at $\mathbf{x}$ with test
point $\mathbf{v}$, and at $\mathbf{y}$ with test point $\mathbf{u}$:

$$
(\mathbf{x} - \mathbf{u})^\top(\mathbf{v} - \mathbf{u}) \le 0,
\qquad
(\mathbf{y} - \mathbf{v})^\top(\mathbf{u} - \mathbf{v}) \le 0.
$$

Adding them gives
$(\mathbf{x} - \mathbf{y} - (\mathbf{u} - \mathbf{v}))^\top(\mathbf{v} - \mathbf{u}) \le 0$,
which rearranges to

$$
\|\mathbf{u} - \mathbf{v}\|^2 \;\le\; (\mathbf{x} - \mathbf{y})^\top (\mathbf{u} - \mathbf{v})
\;\le\; \|\mathbf{x} - \mathbf{y}\|\, \|\mathbf{u} - \mathbf{v}\|
$$

by the Cauchy--Schwarz inequality :eqref:`eq_mdl-cauchy-schwarz`; divide by
$\|\mathbf{u} - \mathbf{v}\|$ (the claim is trivial when it is zero). $\blacksquare$

Projection onto a convex set does not increase distances. For a nonconvex set,
the projection can be multivalued or discontinuous; even a pair of points gives
nearby inputs whose selected projections are far apart.

### Projected Gradient Descent

**Projected gradient descent** (PGD) interleaves a gradient step with a
projection:

$$
\mathbf{x}_{t+1} = \Pi_C\left(\mathbf{x}_t - \eta\, \nabla f(\mathbf{x}_t)\right).
$$
:eqlabel:`eq_mdl-opt-pgd`

For a nonempty closed convex set $C$, a convex $L$-smooth objective, and
$\eta=1/L$, PGD has an $O(1/k)$ objective-value rate
:cite:`Nesterov.2018`. Nonexpansiveness of the projection is part of that
analysis. More generally, for differentiable $f$, convex $C$, and any
$\eta>0$, the fixed points of :eqref:`eq_mdl-opt-pgd` are exactly the
first-order stationary points for the constrained problem. Indeed,
$\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta \nabla f(\mathbf{x}^\star))$
holds iff plugging $\mathbf{y} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$
into :eqref:`eq_mdl-opt-proj-variational` does, which after cancelling
$\mathbf{x}^\star$ reads

$$
\nabla f(\mathbf{x}^\star)^\top (\mathbf{x} - \mathbf{x}^\star) \;\ge\; 0
\qquad \textrm{for all } \mathbf{x} \in C
$$

This is the constrained first-order condition, reducing to $\nabla f = \mathbf{0}$
when $C = \mathbb{R}^n$. If $f$ is convex, the condition is also sufficient for
global optimality; if $f$ is nonconvex, it certifies only stationarity. Under a
constraint qualification such as Slater's (introduced below), it is equivalent
to the KKT conditions when $C$ is described by convex inequalities. Convexity
of $C$ is also essential: for a nonconvex set the projection may be multivalued
and the fixed-point equivalence can fail.

Non-negativity constraints use projection onto the orthant, and max-norm weight
constraints project rows onto balls after the update. Gradient norm clipping
also projects the *gradient vector* onto a ball, but is not generally PGD for a
constraint on the parameters. PGD is practical whenever $\Pi_C$ is cheap,
which brings us to the projection behind sparse attention.

### Projection onto the Simplex

The **probability simplex**
$\Delta = \{\mathbf{x} : \mathbf{x} \ge 0,\; \sum_i x_i = 1\}$ is where
distributions live, and projecting onto it,

$$
\min_{\mathbf{x}}\; \tfrac12\|\mathbf{x} - \mathbf{y}\|^2
\quad \textrm{subject to} \quad \textstyle\sum_i x_i = 1, \;\; -x_i \le 0,
$$

is a small **quadratic program** (QP: convex quadratic objective with affine
constraints). The KKT conditions reduce its solution to finding one scalar.
With multiplier $\tau$ for the sum and $\lambda_i \ge 0$ for each sign
constraint, stationarity reads $x_i - y_i + \tau - \lambda_i = 0$.
Complementary slackness splits the coordinates: where $x_i > 0$ we get
$\lambda_i = 0$ and $x_i = y_i - \tau$; where $x_i = 0$, dual feasibility
$\lambda_i = \tau - y_i \ge 0$ says exactly that $y_i \le \tau$. Both cases
combine as

$$
x_i^\star = \max(y_i - \tau,\, 0),
\qquad \textrm{with } \tau \textrm{ set by } \sum_i \max(y_i - \tau,\, 0) = 1.
$$
:eqlabel:`eq_mdl-opt-simplex`

Every coordinate is shifted down by the same $\tau$ and clipped at zero; the
normalization condition determines the threshold. The left side of the
$\tau$-equation is continuous, piecewise linear, and strictly decreasing where
positive, so the threshold is unique and found by sorting: with
$u_1 \ge \cdots \ge u_n$ the sorted entries of $\mathbf{y}$, the active set is a
top-$k$ prefix, and $\tau = (\sum_{j \le k} u_j - 1)/k$ for the largest $k$
keeping $u_k > \tau$, an $O(n \log n)$ algorithm
:cite:`Held.Wolfe.Crowder.1974,Duchi.Shalev-Shwartz.Singer.ea.2008`. Applied to
scores instead of a softmax, this map is exactly *sparsemax*
:cite:`Martins.Astudillo.2016`: unlike softmax it produces genuinely
sparse attention weights, with complementary slackness determining which
entries are zero. More broadly, attention itself can be read as a *regularized
argmax* over the simplex: softmax arises from an entropy regularizer, and
sparsemax from the squared Euclidean distance used here
:cite:`Niculae.Blondel.2017`. The cell below implements the sort-and-threshold
projection,
reconstructs all the multipliers, and checks every KKT residual numerically,
plus a sanity check that no random feasible point does better.

```{.python .input #constrained-simplex-projection}
def project_simplex(y):
    """Project y onto the probability simplex by sort-and-threshold."""
    u = np.sort(y)[::-1]                       # sorted descending
    css = np.cumsum(u) - 1.0                   # cumulative sums minus budget
    k = np.nonzero(u * np.arange(1, len(y) + 1) > css)[0][-1]
    tau = css[k] / (k + 1.0)                   # multiplier of sum(x) = 1
    return np.maximum(y - tau, 0.0), tau

rng = np.random.default_rng(0)
y = rng.normal(size=6)
x, tau = project_simplex(y)
lam = np.maximum(0.0, tau - y)                 # multipliers of x >= 0
print('x* =', x.round(4), ' sum =', f'{x.sum():.6f}')
print('KKT residuals: stationarity', f'{np.abs(x - y + tau - lam).max():.1e}',
      '| comp. slack', f'{np.abs(lam * x).max():.1e}',
      '| dual feas.', f'{lam.min():.1e}')
# Optimality check: x* beats 10^5 random feasible points
z = rng.exponential(size=(100000, len(y)))
z /= z.sum(axis=1, keepdims=True)              # random points on the simplex
print('f(x*) =', f'{0.5 * ((x - y)**2).sum():.6f}',
      '<= best random feasible', f'{(0.5 * ((z - y)**2).sum(axis=1)).min():.6f}')
```

The stationarity, complementary-slackness, and dual-feasibility residuals are
at machine precision. Two of the six coordinates are exactly zero; their sign
constraints are active and have strictly positive multipliers. The KKT case
analysis yields the algorithm, and its residuals verify the solution.

## Proximal Operators and Coordinate Descent

### Proximal Operators
:label:`subsec_mdl-proximal-operators`

The conjugate encodes a convex function by its supporting lines; another
construction supplies algorithmic updates for convex functions, unifying
gradient steps with the projections that
:numref:`subsec_mdl-projected-gd` will build on. The **proximal operator** of
a convex $f$ is

$$
\mathrm{prox}_{f}(\mathbf{z}) \;=\; \mathop{\mathrm{argmin}}_{\mathbf{x}}\; \left( f(\mathbf{x}) + \tfrac12\, \|\mathbf{x} - \mathbf{z}\|^2 \right),
$$
:eqlabel:`eq_mdl-opt-prox`

well defined for the closed convex $f$ we use: the objective is $1$-strongly
convex, so the uniqueness proposition above guarantees at most one minimizer, and
closedness supplies the existence that an indicator (which is not continuous)
would otherwise put in doubt. The prox balances lower values of $f$ against a
quadratic penalty for moving away from
$\mathbf{z}$. Two special cases calibrate the definition. If $f = 0$, the prox
is the identity. If $f$ is the *indicator* of a closed convex set $C$ (zero on
$C$, $+\infty$ outside), then minimizing forces $\mathbf{x} \in C$ and the
prox is the nearest feasible point:
$\mathrm{prox}_f = \Pi_C$, the projection operator on which
:numref:`subsec_mdl-projected-gd` will build projected gradient descent.
Proximal operators are projections,
generalized from sets to functions.

A sparsity example is $f(x) = \lambda |x|$ (the operator
acts coordinate-wise on $\lambda\|\cdot\|_1$, so one dimension suffices). The
objective $\lambda|x| + \tfrac12 (x - z)^2$ has a kink at $0$, and the
subgradient optimality criterion $0 \in \partial(\cdot)$
(:eqref:`eq_mdl-subopt` of :numref:`sec_mdl-single_variable_calculus`, restated
for vectors in :numref:`subsec_mdl-three-lenses`) gives the following cases: if the
minimizer $x \neq 0$, stationarity reads $\lambda\,\mathrm{sign}(x) + x - z = 0$,
i.e. $x = z - \lambda\,\mathrm{sign}(z)$, consistent only when $|z| > \lambda$;
if $x = 0$, the criterion asks $z \in [-\lambda, \lambda]$; and the two cases
combine as

$$
\mathrm{prox}_{\lambda|\cdot|}(z) \;=\; \mathrm{sign}(z)\, \max\left(|z| - \lambda,\, 0\right).
$$
:eqlabel:`eq_mdl-opt-soft-threshold`

This is **soft-thresholding**; Exercise :exref:`ex_mdl-convexity-compute-full-subdifferential-f`
runs the same subdifferential argument for the Euclidean norm
$\lambda\|\cdot\|_2$. Inputs smaller than $\lambda$ are
set to *exactly* zero; larger ones are shrunk by $\lambda$. This is the
mechanism by which $\ell_1$ regularization produces genuinely sparse weights
where $\ell_2$ only shrinks them.

The resulting update is compact. For a composite objective
$g(\mathbf{x}) + h(\mathbf{x})$ with $g$ smooth and $h$ convex but kinked
(the lasso :cite:`Tibshirani.1996`, with $g$ the least-squares term and
$h = \lambda\|\cdot\|_1$, is
the prototype), alternate a gradient step on the smooth part with a prox on
the kinked part:

$$
\mathbf{x}_{k+1} = \mathrm{prox}_{\eta h}\left(\mathbf{x}_k - \eta\, \nabla g(\mathbf{x}_k)\right).
$$
:eqlabel:`eq_mdl-opt-ista`

This is **ISTA**, the proximal-gradient method. It converges at the same
$O(1/k)$ rate as plain gradient descent, kink notwithstanding (a theorem we
quote rather than prove; see :citet:`Beck.Teboulle.2009`), and it contains
the projected gradient descent of
:numref:`sec_mdl-constrained-optimization-duality` as the special case where
$h$ is an indicator. The cell verifies :eqref:`eq_mdl-opt-soft-threshold`
against brute-force minimization, then runs twenty ISTA steps on a tiny lasso
and records the resulting exact zeros:

```{.python .input #convexity-prox-ista}
lam = 0.7
prox = lambda z, t: np.sign(z) * np.maximum(np.abs(z) - t, 0.0)
zs = np.linspace(-3.0, 3.0, 13)
grid = np.linspace(-5.0, 5.0, 200001)          # brute-force argmin on a grid
brute = np.array([grid[np.argmin(lam * np.abs(grid) + 0.5 * (grid - z)**2)]
                  for z in zs])
print('max |prox - brute-force argmin| =',
      f'{np.abs(prox(zs, lam) - brute).max():.1e}')
rng = np.random.default_rng(5)                 # a tiny lasso problem
A = rng.normal(size=(30, 8))
b = A @ np.array([2.0, 0, 0, -1.5, 0, 0, 0, 1.0]) + 0.05 * rng.normal(size=30)
eta = 1.0 / np.linalg.eigvalsh(A.T @ A).max()
w = np.zeros(8)
for _ in range(20):                            # ISTA: gradient step, then prox
    w = prox(w - eta * A.T @ (A @ w - b), eta * lam)
print('w after 20 ISTA steps:', w.round(3))
print('exact zeros:', int((w == 0).sum()), 'of 8 coordinates')
```

The closed form matches the brute-force minimizer to grid resolution, and
after twenty proximal-gradient steps five of the eight coordinates are
*exactly* zero, set there by the $\max$ in
:eqref:`eq_mdl-opt-soft-threshold`, while the three nonzero coordinates sit
near the planted values $(2.0, -1.5, 1.0)$. Plain gradient descent does not
produce exact zeros because it uses only smooth steps. The proximal operator
incorporates the nonsmooth term, and one application per step suffices for
sparsity.

### Coordinate and Block Coordinate Descent
:label:`subsec_mdl-coordinate-descent`

A full gradient updates every coordinate at once. When parameters or examples
are sparse, touching one coordinate or one block can be much cheaper. Suppose
$f$ has coordinate-wise smoothness constants $L_j$:

$$
f(\mathbf x+t\mathbf e_j)
\le f(\mathbf x)+t\,\partial_jf(\mathbf x)+\tfrac12L_jt^2.
$$
:eqlabel:`eq_mdl-coordinate-smoothness`

Minimizing this one-dimensional upper bound gives the **coordinate-descent**
step

$$
x_j\leftarrow x_j-\frac{1}{L_j}\partial_jf(\mathbf x),
$$
:eqlabel:`eq_mdl-coordinate-step`

with all other coordinates fixed. The method can cycle through coordinates,
choose the largest partial derivative, or sample $j$. Randomized selection
avoids adversarial orderings and has a clean representative guarantee: for a
$\mu$-strongly convex objective, sampling coordinate $j$ proportional to
$L_j$ and taking :eqref:`eq_mdl-coordinate-step` contracts expected
suboptimality at a rate governed by
$1-\mu/\sum_jL_j$. The comparison with full-gradient descent is not “one
iteration versus one iteration”: a coordinate step reads and changes much
less state, so work, memory traffic, and sparsity determine the useful unit.

For a positive-definite quadratic

$$
f(\mathbf x)=\tfrac12\mathbf x^\top A\mathbf x-\mathbf b^\top\mathbf x,
$$

we can minimize the chosen coordinate *exactly*:

$$
x_j\leftarrow x_j-\frac{(A\mathbf x-\mathbf b)_j}{A_{jj}}.
$$
:eqlabel:`eq_mdl-quadratic-coordinate`

Maintaining the residual $\mathbf r=A\mathbf x-\mathbf b$ makes a coordinate
update cheap: after changing $x_j$ by $\delta$, update
$\mathbf r\leftarrow\mathbf r+\delta A_{:j}$. On a sparse matrix this touches
only the nonzeros in column $j$. A full cyclic sweep is the Gauss--Seidel
iteration for $A\mathbf x=\mathbf b$.

```{.python .input #mdl-convexity-coordinate-descent}
d_cd = 40
A_cd = (2 * np.eye(d_cd) - 0.95 * np.eye(d_cd, k=1)
        - 0.95 * np.eye(d_cd, k=-1))             # SPD, strongly coupled
b_cd = np.ones(d_cd)
x_star_cd = np.linalg.solve(A_cd, b_cd)
f_cd = lambda x: 0.5 * x @ A_cd @ x - b_cd @ x
f_star_cd = f_cd(x_star_cd)
L_cd = np.linalg.eigvalsh(A_cd)[-1]
x_gd, x_cd = np.zeros(d_cd), np.zeros(d_cd)
r_cd = A_cd @ x_cd - b_cd
gap_gd, gap_cd = [], []
for sweep in range(1, 101):
    x_gd -= (A_cd @ x_gd - b_cd) / L_cd           # one full gradient
    for j in range(d_cd):                         # one coordinate sweep
        delta = -r_cd[j] / A_cd[j, j]
        x_cd[j] += delta
        r_cd += delta * A_cd[:, j]
    gap_gd.append(f_cd(x_gd) - f_star_cd)
    gap_cd.append(f_cd(x_cd) - f_star_cd)
d2l.plot(np.arange(1, 101), [gap_gd, gap_cd], 'sweep', 'optimality gap',
         legend=[r'gradient descent, step $1/L$', 'exact coordinate sweep'],
         yscale='log')
```

This coupled quadratic includes interactions between neighboring coordinates:
changing one affects its neighbors. Even so, exact coordinate minimization
uses the updated residual after each coordinate and converges faster per sweep here
than the conservative global step $1/L$. Different matrices can reverse that
comparison; the point is the mechanism, not a universal ranking.

The same idea extends in two directions. **Block coordinate descent** updates a
layer, feature group, or matrix block by solving a small subproblem, exploiting
structure that scalar coordinates miss. For a composite objective
$f(\mathbf x)+\lambda\|\mathbf x\|_1$, the coordinate subproblem includes the
nonsmooth term and its exact update is soft-thresholding, the proximal operator
of :eqref:`eq_mdl-opt-soft-threshold`. This is why coordinate methods are
natural for sparse regression and matrix factorization: each cheap local solve
can create exact zeros, whereas an ordinary smooth gradient step cannot.

Coordinate descent is not the default for dense end-to-end neural training:
accelerators are built for large matrix operations, not sequential scalar
updates. It remains important for sparse linear models, embedding tables,
alternating minimization, and inner problems where a block has a closed-form or
very cheap solve.

## Summary

* A constrained optimum is a point with **no feasible descent direction**. For
  one equality constraint with $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$
  this forces $\nabla f = -\nu \nabla g$; the Lagrangian
  $\mathcal{L} = f + \nu g$ packages the condition and the constraint as joint
  stationarity.
* With inequalities, the **KKT conditions** govern: stationarity, primal and
  dual feasibility ($\lambda_i \ge 0$), and **complementary slackness**
  $\lambda_i g_i = 0$, which finds the active set. KKT is necessary under a
  constraint qualification and *sufficient* for global optimality in convex
  problems.
* **Projections** onto nonempty closed convex sets are unique and
  **nonexpansive**. For convex $L$-smooth objectives, projected gradient descent
  with a suitable step has the usual $O(1/k)$ value guarantee. Its fixed points
  satisfy constrained first-order stationarity; convexity of the objective is
  what upgrades stationarity to global optimality. The simplex projection is a
  KKT-derived sort-and-threshold (sparsemax).
* **Proximal operators** generalize
  projections from sets to functions; soft-thresholding is the prox of
  $\lambda|\cdot|$, and one prox per step (ISTA) optimizes kinked composite
  losses at gradient descent's rate. **Coordinate descent** minimizes one
  coordinate or block at a time; on quadratics it is an exact one-dimensional
  solve, and with $\ell_1$ regularization that solve becomes soft-thresholding.

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
    1. Minimize $f(x) = x$ over $\mathbb{R}$ subject to $x^2 \le 0$. Compute
       $p^\star$, the dual function :eqref:`eq_mdl-opt-dual-function`, and
       $d^\star$. Is the dual optimum attained? Does a KKT multiplier exist at
       the minimizer? Which hypothesis of the Slater proposition in
       :numref:`subsec_mdl-lagrangian-duality` fails?
    1. Minimize $f(x) = x$ over $\mathbb{R}$ subject to the constraint
       $-x \le 0$ written twice, $g_1(x) = g_2(x) = -x$. Which constraint
       qualification holds here, and which fails? Find all pairs
       $(\lambda_1^\star, \lambda_2^\star)$ that satisfy the KKT conditions at
       the minimizer. Compute the optimal value $p^\star(u_1, u_2)$ under the
       perturbed constraints $-x \le u_1$ and $-x \le u_2$, and show that
       every such pair satisfies the shadow-price
       inequality :eqref:`eq_mdl-opt-shadow-price`. Is $p^\star$
       differentiable at $\mathbf{0}$, and what takes the place of the relation
       $\boldsymbol{\lambda}^\star = -\nabla p^\star(\mathbf{0})$?

1. **KKT points that are not minima.** The KKT conditions
   :eqref:`eq_mdl-opt-kkt` are necessary at a local minimum under a
   constraint qualification, but a nonconvex problem can satisfy them at other
   points.
    1. Minimize $f(\mathbf{x}) = x_1^2 - x_2^2$ over the box
       $-1 \le x_1 \le 1$, $-1 \le x_2 \le 1$, written as four inequality
       constraints. Find all KKT points and their multipliers. Classify each
       KKT point as a local minimum, a local maximum, or neither.
    1. The maximum of $f$ on the box is attained at $(\pm 1, 0)$. Which KKT
       condition fails at these points, and what value would the multiplier
       of the active constraint need?
    1. Give a problem in one variable with a single inequality constraint
       $g(x) \le 0$ that has a KKT point $x^\star$ which is a strict local
       maximum. Can the multiplier at such a point be positive when
       $g'(x^\star) \neq 0$?
   :exlabel:`ex_mdl-constrained-optimization-duality-write-all-four-kkt`

1. **Projected fixed points.** For a closed convex set $C$, the fixed points
   of projected gradient descent :eqref:`eq_mdl-opt-pgd` are the constrained
   stationary points, whatever the step size $\eta > 0$.
    1. Let $C = \{\mathbf{x} : \mathbf{x} \succeq 0\}$. Show that
       $\mathbf{x}^\star$ is a fixed point if and only if, for every
       coordinate $i$, either $\partial f / \partial x_i (\mathbf{x}^\star) = 0$
       and $x_i^\star \ge 0$, or
       $\partial f / \partial x_i (\mathbf{x}^\star) > 0$ and $x_i^\star = 0$.
       Show that these conditions are the KKT conditions for the constraints
       $-x_i \le 0$, and express the multipliers through
       $\nabla f(\mathbf{x}^\star)$.
    1. Gradient clipping updates
       $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\Pi_{B_c}(\nabla f(\mathbf{x}_t))$,
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
       fixed-point argument in :numref:`subsec_mdl-projected-gd` fails for
       this set?

1. [code] **Sparsemax and the $\ell_1$ ball.** The function `project_simplex`
   of :numref:`subsec_mdl-projected-gd` computes the simplex projection
   :eqref:`eq_mdl-opt-simplex`, which is sparsemax; its first output is the
   projected vector.
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
       `project_simplex`. Why does it suffice to check the variational
       inequality :eqref:`eq_mdl-opt-proj-variational` at the $2n$ points
       $\pm z\,\mathbf{e}_i$ to certify the result? Run that check.

1. **Second-order conditions on the sphere.** ● The Lagrange condition is a
   first-order condition and cannot tell minima from saddles. A second-order
   test must account for the curvature of the constraint.
    1. Let $f$ and $g$ be twice continuously differentiable, and let
       $\mathbf{x}^\star$ satisfy :eqref:`eq_mdl-opt-lagrange-condition` with
       multiplier $\nu^\star$. For every curve $\mathbf{x}(t)$ on
       $\{g = 0\}$ with $\mathbf{x}(0) = \mathbf{x}^\star$ and
       $\dot{\mathbf{x}}(0) = \mathbf{v}$, show that

       $$\left.\frac{d^2}{dt^2} f(\mathbf{x}(t))\right|_{t=0} = \mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}(\mathbf{x}^\star, \nu^\star)\, \mathbf{v},$$

       with the Lagrangian :eqref:`eq_mdl-opt-lagrangian-eq`. Conclude that
       $\mathbf{v}^\top \nabla^2_{\mathbf{x}} \mathcal{L}\, \mathbf{v} \ge 0$
       for every tangent direction $\mathbf{v}$ at a local minimum. Why is the
       Hessian of $f$ alone the wrong matrix for this test? Conversely,
       positive definiteness on the tangent space implies a strict local
       minimum (:citet:`Nocedal.Wright.2006`, chapter 12).
    1. Let $A$ be symmetric with distinct eigenvalues
       $a_1 < a_2 < \cdots < a_n$ and unit eigenvectors
       $\mathbf{q}_1, \ldots, \mathbf{q}_n$, and minimize
       $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ subject to
       $\mathbf{x}^\top \mathbf{x} = 1$. Show that the Lagrange points are
       $\pm \mathbf{q}_i$ with $\nu^\star = -a_i$, and find the eigenvalues of
       $\nabla^2_{\mathbf{x}} \mathcal{L}$ on the tangent space at
       $\mathbf{q}_i$. Which Lagrange points are local minima, which are local
       maxima, and which are saddles? How many independent descent directions
       does $\mathbf{q}_i$ have?
    1. Projected gradient descent onto the sphere,
       $\mathbf{x} \leftarrow (\mathbf{x} - \eta \nabla f(\mathbf{x})) / \|\mathbf{x} - \eta \nabla f(\mathbf{x})\|$,
       projects onto a nonconvex set. Show that for this $f$ it is the power
       iteration for $I - 2\eta A$. Assume $a_1 + a_n > 0$. From a random
       start, what do the iterates $\mathbf{x}_t$ and the values
       $f(\mathbf{x}_t)$ do? How does the answer depend on $\eta > 0$?

1. **Moreau decomposition.** ● For a closed convex $f$, two facts connect
   the proximal operator :eqref:`eq_mdl-opt-prox` with the conjugate
   :eqref:`eq_mdl-opt-conjugate`: $\mathbf{x} = \mathrm{prox}_f(\mathbf{z})$
   exactly when $\mathbf{z} - \mathbf{x} \in \partial f(\mathbf{x})$, and
   $\mathbf{y} \in \partial f(\mathbf{x})$ exactly when
   $\mathbf{x} \in \partial f^*(\mathbf{y})$ :cite:`Parikh.Boyd.2014`.
    1. Derive the Moreau decomposition
       $\mathbf{z} = \mathrm{prox}_f(\mathbf{z}) + \mathrm{prox}_{f^*}(\mathbf{z})$.
    1. Take $f = \lambda\|\cdot\|_1$ and its conjugate from problem
       :exref:`ex_mdl-convexity-conjugate-pairs`a. Show that the
       decomposition writes soft-thresholding
       :eqref:`eq_mdl-opt-soft-threshold` as $\mathbf{z}$ minus a clipped
       copy of $\mathbf{z}$.
    1. Take $f = \lambda\|\cdot\|_2$, whose conjugate is the indicator of
       the $\ell_2$ ball of radius $\lambda$. The proximal operator of that
       indicator is the projection onto the ball, the clip-by-norm map of
       gradient clipping. Use the decomposition to recover the solution of
       problem :exref:`ex_mdl-convexity-compute-full-subdifferential-f`c.
   :exlabel:`ex_mdl-convexity-moreau-decomposition`

1. [code] **Accelerated proximal gradient.** ● FISTA
   :cite:`Beck.Teboulle.2009` adds momentum to the ISTA update
   :eqref:`eq_mdl-opt-ista` for $F = g + h$. With
   $\mathbf{v}_1 = \mathbf{x}_0$ and $t_1 = 1$, it iterates for $k \ge 1$

   $$\mathbf{x}_k = \mathrm{prox}_{\eta h}\bigl(\mathbf{v}_k - \eta \nabla g(\mathbf{v}_k)\bigr), \qquad t_{k+1} = \tfrac12\Bigl(1 + \sqrt{1 + 4t_k^2}\Bigr), \qquad \mathbf{v}_{k+1} = \mathbf{x}_k + \frac{t_k - 1}{t_{k+1}}\,(\mathbf{x}_k - \mathbf{x}_{k-1}).$$

   With $\eta = 1/L$ its objective gap satisfies
   $F(\mathbf{x}_k) - F^\star \le 2L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2/(k+1)^2$,
   against $O(1/k)$ for ISTA.
    1. Add FISTA to the `#convexity-prox-ista` cell and plot
       $F(\mathbf{x}_k) - F^\star$ of both methods over 100 steps on a
       logarithmic axis, with $F^\star$ taken from $10^5$ ISTA steps. Which
       method is ahead after 10 steps, and which after 30? Explain the ISTA
       curve with problem :exref:`ex_mdl-convexity-distance-contraction`a
       and the nonexpansiveness
       $\|\mathrm{prox}_{\eta h}(\mathbf{u}) - \mathrm{prox}_{\eta h}(\mathbf{v})\| \le \|\mathbf{u} - \mathbf{v}\|$
       of the proximal operator.
    1. Is $F(\mathbf{x}_k)$ monotone along the FISTA iterates?
    1. Replace `A` by a $30 \times 100$ Gaussian matrix and plant the same
       three nonzero coefficients, and run both methods for 1000 steps.
       Over which range of $k$ is FISTA ahead now? Why does ISTA still end
       with a fast linear phase, although $A^\top A$ is singular?
   :exlabel:`ex_mdl-convexity-accelerated-proximal-gradient`

## Discussions

These methods connect directly to applications in the main book. The penalty form of
regularization (add $\lambda\|\mathbf{w}\|^2$ to the loss) and the
constraint form (minimize the loss subject to $\|\mathbf{w}\|^2 \le r^2$)
are linked precisely by the Lagrangian: the weight-decay coefficient of
:numref:`sec_weight_decay` is the multiplier of the norm constraint, and
sweeping one traces out the solutions of the other. Projections support gradient
clipping and constrained updates, and the simplex projection is
sparsemax. Within
this part, the section leans on :numref:`sec_mdl-convexity` (convexity, the
conjugate) and :numref:`sec_mdl-gradient-based-optimization` (descent lemma,
rates), and its conditioning consequences continue in
:numref:`sec_mdl-numerical-stability-conditioning`.

[Discussions](https://d2l.discourse.group/t/constrained-optimization-duality)

[Discussions](https://d2l.discourse.group/t/36831)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §24.4]{.kicker}

Conditions and algorithms for constrained optimization<br>**Lagrange multipliers · KKT · projections · duality**.
:::
:::

::: {.slide title="First-order conditions under constraints"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
At an unconstrained minimum the gradient vanishes. At a *constrained*
minimum it need not, but there is **no feasible descent direction**: no
move that both stays feasible and lowers $f$.

- **Equality** constraints give Lagrange multipliers, $\nabla f = -\nu\,\nabla g$.
- **Inequality** constraints give the KKT conditions.
- **Convex** feasible sets make it an algorithm: projected gradient descent.

::: {.d2l-note}
An optimal multiplier measures sensitivity to relaxation of its constraint.
:::
:::

::: {.col .fig}
@fig:mdl-opt-lagrange-tangency
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Lagrange multipliers]{.dtitle}

[equality constraints and tangency]{.dsub}
:::
:::

::: {.slide title="Feasible Stationarity Implies Tangency"}
[Lagrange]{.kicker}

::: {.cols .vc}
::: {.col}
If a level set of $f$ crosses the constraint, $\nabla f$ has a component
along the feasible direction, which gives local descent and lowers $f$. At a regular
constrained optimum, the curves are **tangent**, so their normals are parallel:

$$\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla g(\mathbf{x}^\star) = \mathbf{0}.$$

Valid wherever $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$, the
**constraint qualification**.
:::

::: {.col .fig}
@fig:mdl-opt-lagrange-tangency
:::
:::
:::

::: {.slide title="The Lagrangian Combines the Conditions"}
[Lagrange]{.kicker}

Parallel gradients and feasibility become joint stationarity of one function
with an additional variable:

$$\mathcal{L}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu\, g(\mathbf{x}),
\qquad
\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0},\;\; \partial_\nu \mathcal{L} = g = 0.$$

. . .

Closest point on a hyperplane $\mathbf{a}^\top\mathbf{x} = b$: stationarity
places $\mathbf{x}$ *along the normal* $\mathbf{a}$, giving
$\mathbf{x}^\star = \tfrac{b}{\|\mathbf{a}\|^2}\,\mathbf{a}$.

::: {.d2l-note}
Already a shadow price: $\partial p^\star/\partial b = -\nu^\star$. The
multiplier tracks how the optimum moves when the constraint moves.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The KKT conditions]{.dtitle}

[inequalities and the active set]{.dsub}
:::
:::

::: {.slide title="Active and Inactive Inequality Constraints"}
[KKT]{.kicker}

::: {.cols .vc}
::: {.col}
With $g_i(\mathbf{x}) \le 0$, each constraint is **inactive**
($g_i < 0$, locally irrelevant) or **active** ($g_i = 0$, contributing
like an equality). At the optimum $-\nabla f$ is a *nonnegative*
combination of the active outward normals, in their **cone**.

An inequality multiplier satisfies $\lambda_i \ge 0$, so active constraint
normals enter stationarity with nonnegative coefficients.
:::

::: {.col .fig}
@fig:mdl-opt-kkt-active-set
:::
:::
:::

::: {.slide title="The four KKT conditions"}
[KKT]{.kicker}

::: {.cols}
::: {.col}
::: {.d2l-note .rule}
**Stationarity:** $\nabla f + \sum_i \lambda_i \nabla g_i + \sum_j \nu_j \nabla h_j = \mathbf{0}$

**Primal feas.:** $g_i \le 0,\;\; h_j = 0$

**Dual feas.:** $\lambda_i \ge 0$

**Comp. slackness:** $\lambda_i\, g_i = 0$
:::
:::

::: {.col}
**Complementary slackness** *finds the active set*: for each $i$, either
$g_i = 0$ (active, $\lambda_i$ free) or $\lambda_i = 0$ (slack, priced at
zero). The pattern of zero multipliers identifies which constraints contribute to
stationarity.
:::
:::

. . .

Convex $f, g_i$ and affine $h_j$: a KKT point is a **global** minimum, by
a direct convexity argument. Thus KKT is sufficient in this setting.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Projections]{.dtitle}

[projection-based constrained updates]{.dsub}
:::
:::

::: {.slide title="Projected Gradient Descent"}
[Projections]{.kicker}

::: {.cols .vc}
::: {.col}
For nonempty closed convex $C$, $\Pi_C(\mathbf{y})$ is the unique nearest
feasible point and is **nonexpansive**
($\|\Pi_C\mathbf{x} - \Pi_C\mathbf{y}\| \le \|\mathbf{x} - \mathbf{y}\|$),
which supports projected-gradient analysis:

$$\mathbf{x}_{t+1} = \Pi_C\!\left(\mathbf{x}_t - \eta\,\nabla f(\mathbf{x}_t)\right).$$

Its fixed points are constrained **stationary points**. They are global optima
when $f$ is convex; otherwise no global conclusion follows.
:::

::: {.col .narrow}
::: {.d2l-note}
Non-negativity and max-norm parameter constraints give PGD steps. Gradient norm
clipping projects the gradient, but is not generally PGD for a parameter
constraint.
:::
:::
:::
:::

::: {.slide title="Simplex projection = sort, shift, clip"}
[Projections]{.kicker}

Projecting onto $\{\mathbf{x} \ge 0,\, \sum_i x_i = 1\}$ is a QP the KKT
conditions solve in closed form up to one threshold $\tau$:

$$x_i^\star = \max(y_i - \tau,\, 0),
\qquad \textstyle\sum_i \max(y_i - \tau,\, 0) = 1.$$

A common shift followed by clipping gives **sparsemax**: genuinely sparse
attention weights, with complementary slackness determining the zeros.

@!constrained-simplex-projection

::: {.d2l-note}
Two coordinates are **exactly zero**, corresponding to active sign
constraints; every KKT residual is at machine precision.
:::
:::

::: {.slide title="Proximal operators for nonsmooth terms"}
[Recognizing convexity]{.kicker}

$\mathrm{prox}_f(\mathbf{z})$ balances lower values of $f$ against a
quadratic penalty for moving away from $\mathbf{z}$: a projection,
generalized from sets to functions. For $\lambda|x|$ the subgradient
criterion solves it in closed form: **soft-thresholding**,

$$\mathrm{prox}_{\lambda|\cdot|}(z) = \mathrm{sign}(z)\,\max(|z| - \lambda,\, 0),$$

and alternating a gradient step on the smooth part with a proximal update on
the nonsmooth part is **ISTA**, which keeps gradient descent's $O(1/k)$ rate:

@!convexity-prox-ista

Five of eight coordinates are exactly zero because of the $\max$ thresholding
operation; an ordinary smooth gradient step does not produce this result. This is how $\ell_1$
sparsifies where $\ell_2$ only shrinks.
:::

::: {.slide title="Constraints replace zero gradients by feasible stationarity"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- A constrained optimum has **no feasible descent direction**, made
  precise by Lagrange and **KKT**; complementary slackness finds the
  active set.
- **Projections** onto convex sets are nonexpansive; projected GD is
  gradient descent followed by projection back to feasibility.
:::

::: {.col}
- The **dual** is always concave and always a lower bound; Slater closes
  the gap; multipliers are **shadow prices**.
- The **SVM dual** and **water-filling** provide computable examples;
  non-convexity can leave a positive gap.
:::
:::

::: {.d2l-note}
Many deep-learning subproblems, including projections, clipped updates, and
last-layer fits, are convex optimization problems.
:::
:::
