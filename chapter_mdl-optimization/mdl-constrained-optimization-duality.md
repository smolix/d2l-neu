# Constrained Optimization and Duality
:label:`sec_mdl-constrained-optimization-duality`

Constraints and their multipliers are everywhere in deep learning, even when they
are not written down as such: projected and clipped updates, trust regions, the
support-vector machine and attention recast as optimization, and the
regularization-as-constraint equivalence behind weight decay. This is the
Boyd-style section of the chapter :cite:`Boyd.Vandenberghe.2004`. It teaches you
to *read* a constrained problem and its dual --- to recognize that a multiplier
is a shadow price, that complementary slackness pins down which constraints are
active, and that a hard problem sometimes becomes easy when viewed from the dual
side. We build from Lagrange multipliers for equalities, to the full Lagrangian
and KKT conditions for inequalities, then make constraints *algorithmic* with
projections and projected gradient descent, develop Lagrangian duality and the
strong-duality guarantee, and ground it all in worked examples: the SVM dual
solved by projected gradient ascent, water-filling solved by bisection, and a
tiny non-convex problem whose duality gap you can see.

One idea powers everything in this section. At an unconstrained minimum the
gradient vanishes; at a *constrained* minimum it need not --- but there must be
**no feasible descent direction**: no infinitesimal move that both respects the
constraints and decreases the objective. Lagrange multipliers, the KKT
conditions, and the fixed points of projected gradient descent are this one
sentence translated into equations for three kinds of constraint sets.

We lean on :numref:`sec_mdl-multivariable_calculus` (gradients, level sets, and
the tangency teaser we are about to grow), :numref:`sec_mdl-gradient-based-optimization`
(stationarity and the descent lemma), and :numref:`sec_mdl-convexity` (convex
sets and functions --- required for KKT sufficiency and strong duality). The
standard reference is :citet:`Boyd.Vandenberghe.2004`, chapter 5; Nocedal &
Wright's *Numerical Optimization* (2006) gives the numerical view. The code in
this section is deliberately framework-free --- plain NumPy --- because every
algorithm is a handful of lines; we load the per-framework `d2l` module once for
plotting.

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

### The Geometry: No Feasible Descent

Consider the cleanest constrained problem there is: minimize a smooth
$f : \mathbb{R}^n \to \mathbb{R}$ over the surface carved out by one smooth
equality constraint,

$$
\min_{\mathbf{x}}\; f(\mathbf{x}) \quad \textrm{subject to} \quad g(\mathbf{x}) = 0.
$$

:numref:`sec_mdl-multivariable_calculus` already showed us the picture: the
admissible moves from a feasible point are, to first order, the directions
*tangent* to the surface $\{g = 0\}$, and at a constrained optimum the level
curves of $f$ *kiss* the constraint surface. :numref:`fig_mdl-opt-lagrange-tangency`
shows why the kiss is forced. At a feasible point where a level curve of $f$
*crosses* the constraint curve, the gradient $\nabla f$ has a nonzero component
along the constraint --- so sliding along the constraint against that component
decreases $f$ while staying feasible. A feasible descent direction exists, and
the point cannot be optimal. Only where the level curve is *tangent* to the
constraint does every feasible direction become neutral to first order --- and
tangency of the curves is parallelism of their normals: $\nabla f$ must be a
multiple of $\nabla g$.

![Minimizing $f$ along the constraint curve $g(\mathbf{x})=0$. At a non-optimal feasible point the gradient $\nabla f$ has a component along the constraint, so sliding along the curve descends. At the optimum $\mathbf{x}^\star$ the level set of $f$ is tangent to the constraint and the two gradients are parallel: $\nabla f = -\nu\,\nabla g$.](../img/mdl-opt-lagrange-tangency.svg)
:label:`fig_mdl-opt-lagrange-tangency`

**Proposition (Lagrange condition).** *Let $f$ and $g$ be continuously
differentiable, and let $\mathbf{x}^\star$ be a local minimum of $f$ on
$\{\mathbf{x} : g(\mathbf{x}) = 0\}$ at which $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$.
Then there is a unique scalar $\nu^\star$ --- the* **Lagrange multiplier** *---
with*

$$
\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla g(\mathbf{x}^\star) = \mathbf{0}.
$$
:eqlabel:`eq_mdl-opt-lagrange-condition`

**Proof.** The first-order feasible directions at $\mathbf{x}^\star$ are the
tangent directions $\mathbf{v}$ with $\nabla g(\mathbf{x}^\star)^\top\mathbf{v} = 0$.
Because $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$, the implicit function
theorem makes the feasible set a genuine smooth surface near $\mathbf{x}^\star$:
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

The hypothesis $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$ is called a
**constraint qualification**, and it is not decoration. Minimize $f(\mathbf{x}) = x_1$
subject to $g(\mathbf{x}) = x_1^2 + x_2^2 = 0$: the feasible set is the single
point $\mathbf{0}$, which is therefore the minimum, yet
$\nabla f(\mathbf{0}) = (1, 0)$ while $\nabla g(\mathbf{0}) = \mathbf{0}$ --- no
multiplier exists. The constraint set has collapsed to a point, the "surface" is
not a surface, and the tangent-space argument has nothing to stand on. Whenever
you invoke multipliers, you are implicitly certifying that the constraint
gradients do not degenerate at the optimum.

### The Lagrangian

The two optimality requirements --- parallel gradients *and* feasibility ---
package into stationarity of a single function of more variables, the
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

are exactly :eqref:`eq_mdl-opt-lagrange-condition` and the constraint. A
constrained problem in $\mathbf{x}$ became an unconstrained stationarity problem
in $(\mathbf{x}, \nu)$ --- the multiplier is not a trick but a *coordinate* of
the optimality condition. With several equality constraints
$h_j(\mathbf{x}) = 0$, $j = 1, \ldots, p$, each gets its own multiplier,
$\mathcal{L} = f + \sum_j \nu_j h_j$, and the same argument (now requiring the
$\nabla h_j(\mathbf{x}^\star)$ to be linearly independent) puts
$\nabla f(\mathbf{x}^\star)$ in the *span* of the constraint gradients: the
objective's gradient is balanced by a combination of the constraint normals,
with nothing left over along any feasible direction.

### Worked Example: Closest Point on a Hyperplane

Minimize $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{x}$ subject to
$h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x} - b = 0$: the point of a hyperplane
nearest the origin. Stationarity of $\mathcal{L} = \mathbf{x}^\top\mathbf{x} + \nu(\mathbf{a}^\top\mathbf{x} - b)$
gives $2\mathbf{x} + \nu\mathbf{a} = \mathbf{0}$, so $\mathbf{x} = -\tfrac{\nu}{2}\mathbf{a}$:
the solution is forced to lie *along the normal* $\mathbf{a}$, which is the
geometry doing the work. Feasibility fixes the scale,
$\nu^\star = -2b/\|\mathbf{a}\|^2$, hence

$$
\mathbf{x}^\star = \frac{b}{\|\mathbf{a}\|^2}\,\mathbf{a},
$$

recovering the projection formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops`. The multiplier already whispers
its future meaning: the optimal value is $p^\star(b) = b^2/\|\mathbf{a}\|^2$,
and $\partial p^\star / \partial b = 2b/\|\mathbf{a}\|^2 = -\nu^\star$. The
multiplier measures *how much the optimum moves when the constraint moves* --- a
shadow price. We will prove this in general in
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
$g_i(\mathbf{x}^\star) < 0$ --- the constraint is **inactive**, the optimum sits
strictly inside its region, and locally the constraint might as well not exist
--- or $g_i(\mathbf{x}^\star) = 0$ --- the constraint is **active** and behaves
like an equality, pushing back against the objective.

A worked example shows both regimes at once. Project a point onto a ball:
minimize $\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ subject to
$g(\mathbf{x}) = \tfrac12(\|\mathbf{x}\|^2 - r^2) \le 0$. Stationarity of
$\mathcal{L} = \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2 + \lambda g(\mathbf{x})$
gives $(\mathbf{x} - \mathbf{x}_0) + \lambda\mathbf{x} = \mathbf{0}$, i.e.
$\mathbf{x} = \mathbf{x}_0 / (1 + \lambda)$: the constraint can only pull the
answer *radially inward*. If $\|\mathbf{x}_0\| \le r$ the constraint is inactive,
$\lambda^\star = 0$ and $\mathbf{x}^\star = \mathbf{x}_0$; if
$\|\mathbf{x}_0\| > r$ it is active, $\|\mathbf{x}^\star\| = r$ forces
$\lambda^\star = \|\mathbf{x}_0\|/r - 1 > 0$, and

$$
\mathbf{x}^\star = \frac{\mathbf{x}_0}{\max\left(1,\; \|\mathbf{x}_0\|/r\right)}.
$$

You have used this formula: *gradient clipping* is exactly this projection
applied to the update vector. And note the sign of the active multiplier:
$\lambda^\star > 0$, never negative. An inequality constraint can only push
*one way* --- into its feasible side --- so its multiplier carries a sign
constraint that an equality multiplier does not.

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
**Karush--Kuhn--Tucker (KKT) conditions** (Karush 1939; Kuhn & Tucker 1951):

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

Three of the four you have already met: stationarity is the balanced-gradient
condition, primal feasibility is the problem statement, dual feasibility is the
one-way push. The fourth is the workhorse. **Complementary slackness** says that
for each constraint, *at least one of* $\lambda_i^\star$ and
$g_i(\mathbf{x}^\star)$ is zero: a constraint is either active
($g_i = 0$, multiplier free to be positive) or priced at zero
($\lambda_i = 0$, constraint slack and locally irrelevant). It is the equation
that *finds the active set* --- solve a KKT system and the pattern of zero
multipliers tells you which constraints actually shaped the answer. In the ball
projection, complementary slackness is precisely the case split we did by hand.

:numref:`fig_mdl-opt-kkt-active-set` draws stationarity plus dual feasibility:
at the optimum, $-\nabla f$ must be a *nonnegative combination of the active
constraints' outward normals*. The objective pushes against the wall of active
constraints, and the wall --- which can only push back outward --- absorbs it
exactly; inactive constraints contribute nothing.

![Geometry of the KKT conditions on a feasible region cut out by two inequality constraints. At the optimum $\mathbf{x}^\star$ one constraint is active and one is inactive: $-\nabla f$ points along the active constraint's outward normal with $\lambda_1 > 0$, while the inactive constraint has $\lambda_2 = 0$. At a corner where two constraints are active, $-\nabla f$ must lie in the cone spanned by both normals.](../img/mdl-opt-kkt-active-set.svg)
:label:`fig_mdl-opt-kkt-active-set`

How strong are these conditions? In general they are **necessary**: at any local
minimum satisfying a constraint qualification (the standard one, *LICQ*, asks
the active constraints' gradients to be linearly independent --- the inequality
analogue of $\nabla g \neq \mathbf{0}$ above), multipliers satisfying
:eqref:`eq_mdl-opt-kkt` exist; see Nocedal & Wright (2006), chapter 12, for the
proof. They are not sufficient: a non-convex problem can have KKT points that
are saddles or maxima (Exercise 3). Under convexity, however, the implication
reverses, with a proof short enough to be memorable.

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

So for convex problems the KKT system *is* the answer: necessary (under a
constraint qualification we will meet as Slater's condition) and sufficient.
This is the bridge from "set the gradient to zero" to constrained problems, and
it is how every example in the rest of this section will be solved.

## Projections and Projected Gradient Descent
:label:`subsec_mdl-projected-gd`

The KKT conditions characterize the answer; this subsection turns them into an
*algorithm*. The idea could not be simpler: take the ordinary gradient step,
and if it leaves the feasible set, snap back to the nearest feasible point.

### Projection onto a Convex Set

For a closed convex set $C \subseteq \mathbb{R}^n$, the **projection** of
$\mathbf{y}$ onto $C$ is the nearest feasible point,

$$
\Pi_C(\mathbf{y}) = \mathop{\mathrm{argmin}}_{\mathbf{x} \in C}\; \tfrac12 \|\mathbf{x} - \mathbf{y}\|^2.
$$

A minimizer exists (the objective is continuous and we may restrict to a compact
ball around $\mathbf{y}$) and is unique (the objective is strictly convex), so
$\Pi_C$ is a well-defined map. Both examples we have computed are projections:
onto a hyperplane in :numref:`subsec_mdl-lagrange-multipliers`, onto a ball ---
the clipping formula --- in :numref:`subsec_mdl-kkt-conditions`.

What makes projections tractable is their first-order characterization. The
point $\hat{\mathbf{x}} = \Pi_C(\mathbf{y})$ minimizes the convex function
$\tfrac12\|\mathbf{x} - \mathbf{y}\|^2$ over the convex set $C$, and "no
feasible descent" over a convex set has an especially clean form: for any
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
angle with every feasible direction --- the set $C$ lies entirely on the far
side of the supporting hyperplane through $\Pi_C(\mathbf{y})$ with normal
$\mathbf{y} - \Pi_C(\mathbf{y})$. From this single inequality follows the
property that makes projected methods safe.

**Proposition (projections are nonexpansive).** *For a closed convex set $C$
and all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,*

$$
\|\Pi_C(\mathbf{x}) - \Pi_C(\mathbf{y})\| \;\le\; \|\mathbf{x} - \mathbf{y}\|.
$$

**Proof.** Write $\mathbf{u} = \Pi_C(\mathbf{x})$, $\mathbf{v} = \Pi_C(\mathbf{y})$.
Apply :eqref:`eq_mdl-opt-proj-variational` twice --- at $\mathbf{x}$ with test
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

by the Cauchy--Schwarz inequality; divide by $\|\mathbf{u} - \mathbf{v}\|$ (the
claim is trivial when it is zero). $\blacksquare$

Projecting can only shrink distances. Convexity is essential: projecting onto a
*non*-convex set (say, a pair of points) can tear nearby inputs far apart, which
is one more reason convex feasible sets are the tractable ones.

### Projected Gradient Descent

**Projected gradient descent** (PGD) interleaves a gradient step with a
projection:

$$
\mathbf{x}_{t+1} = \Pi_C\left(\mathbf{x}_t - \eta\, \nabla f(\mathbf{x}_t)\right).
$$
:eqlabel:`eq_mdl-opt-pgd`

Everything we know about plain gradient descent
(:numref:`sec_mdl-gradient-based-optimization`) survives, because the appended
map $\Pi_C$ is nonexpansive: it cannot amplify the distance between iterates,
and for convex $L$-smooth $f$ with $\eta = 1/L$, PGD converges at the same
$O(1/k)$ rate as the unconstrained method :cite:`Nesterov.2018`. Just as
important, it stops in the right place: the fixed points of
:eqref:`eq_mdl-opt-pgd` are exactly the points with no feasible descent
direction. Indeed, $\mathbf{x}^\star = \Pi_C(\mathbf{x}^\star - \eta \nabla f(\mathbf{x}^\star))$
holds iff plugging $\mathbf{y} = \mathbf{x}^\star - \eta\nabla f(\mathbf{x}^\star)$
into :eqref:`eq_mdl-opt-proj-variational` does, which after cancelling
$\mathbf{x}^\star$ reads

$$
\nabla f(\mathbf{x}^\star)^\top (\mathbf{x} - \mathbf{x}^\star) \;\ge\; 0
\qquad \textrm{for all } \mathbf{x} \in C
$$

--- the constrained first-order condition, reducing to $\nabla f = \mathbf{0}$
when $C = \mathbb{R}^n$, and equivalent to the KKT conditions when $C$ is given
by convex inequalities. Deep-learning practice is full of disguised PGD steps:
gradient clipping projects updates onto a ball, non-negativity is enforced by
clamping (projection onto the orthant), and max-norm weight constraints project
each row back onto a ball after the update. The method earns its keep whenever
$\Pi_C$ is cheap --- which brings us to the one projection every attention
practitioner should know.

### Projection onto the Simplex

The **probability simplex**
$\Delta = \{\mathbf{x} : \mathbf{x} \ge 0,\; \sum_i x_i = 1\}$ is where
distributions live, and projecting onto it,

$$
\min_{\mathbf{x}}\; \tfrac12\|\mathbf{x} - \mathbf{y}\|^2
\quad \textrm{subject to} \quad \textstyle\sum_i x_i = 1, \;\; -x_i \le 0,
$$

is a small QP that the KKT conditions solve *in closed form up to one scalar*.
With multiplier $\tau$ for the sum and $\lambda_i \ge 0$ for each sign
constraint, stationarity reads $x_i - y_i + \tau - \lambda_i = 0$.
Complementary slackness splits the coordinates: where $x_i > 0$ we get
$\lambda_i = 0$ and $x_i = y_i - \tau$; where $x_i = 0$, dual feasibility
$\lambda_i = \tau - y_i \ge 0$ says exactly that $y_i \le \tau$. Both cases
compress into one formula,

$$
x_i^\star = \max(y_i - \tau,\, 0),
\qquad \textrm{with } \tau \textrm{ set by } \sum_i \max(y_i - \tau,\, 0) = 1.
$$
:eqlabel:`eq_mdl-opt-simplex`

*Soft-threshold, then renormalize via the threshold*: every coordinate is
shifted down by the same $\tau$ and clipped at zero. The left side of the
$\tau$-equation is continuous, piecewise linear, and strictly decreasing where
positive, so the threshold is unique and found by sorting: with
$u_1 \ge \cdots \ge u_n$ the sorted entries of $\mathbf{y}$, the active set is a
top-$k$ prefix, and $\tau = (\sum_{j \le k} u_j - 1)/k$ for the largest $k$
keeping $u_k > \tau$ --- an $O(n \log n)$ algorithm (Held, Wolfe & Crowder 1974;
Duchi et al. 2008). This map, applied to scores instead of a softmax, is exactly
*sparsemax* (Martins & Astudillo 2016): unlike softmax it produces genuinely
sparse attention weights, with complementary slackness deciding which entries
are zeroed. The cell below implements the sort-and-threshold projection,
reconstructs all the multipliers, and checks every KKT residual numerically ---
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

The stationarity, complementary-slackness, and dual-feasibility residuals all
sit at machine precision, and three of the six coordinates came back exactly
zero --- the active sign constraints, each carrying a strictly positive
multiplier. KKT did not just certify the answer; its case analysis *was* the
algorithm.

## The Dual Problem
:label:`subsec_mdl-lagrangian-duality`

### The Lagrange Dual Function

So far multipliers were unknowns solved for alongside $\mathbf{x}$. Duality
promotes them to *variables in their own right*. Fix
$(\boldsymbol{\lambda}, \boldsymbol{\nu})$ with $\boldsymbol{\lambda} \succeq 0$
and minimize the Lagrangian :eqref:`eq_mdl-opt-lagrangian` over $\mathbf{x}$
*unconstrained*:

$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}}\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$
:eqlabel:`eq_mdl-opt-dual-function`

This **Lagrange dual function** has two universal properties --- universal
meaning they require *no* assumptions on $f$, $g_i$, $h_j$ whatsoever.

**Proposition (the dual function is concave).** *For any primal problem,
$g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ in :eqref:`eq_mdl-opt-dual-function`
is concave.*

**Proof.** For each fixed $\mathbf{x}$, the map
$(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$
is *affine* --- the multipliers enter linearly. The dual function is the
pointwise infimum of this family of affine functions, and a pointwise infimum
of affine functions is concave: its hypograph is the intersection of the
half-spaces below the affine functions, an intersection of convex sets
(:numref:`sec_mdl-convexity`). $\blacksquare$

**Proposition (weak duality).** *If $\mathbf{x}$ is feasible for
:eqref:`eq_mdl-opt-standard-problem` and $\boldsymbol{\lambda} \succeq 0$, then
$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$. Consequently*

$$
d^\star \;=\; \sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \inf_{\mathbf{x}\ \textrm{feasible}} f(\mathbf{x}) \;=\; p^\star.
$$
:eqlabel:`eq_mdl-opt-weak-duality`

**Proof.** For feasible $\mathbf{x}$: each $\lambda_i g_i(\mathbf{x}) \le 0$
(nonnegative times nonpositive) and each $\nu_j h_j(\mathbf{x}) = 0$, so

$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;=\; f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})
\;\le\; f(\mathbf{x}).
$$

Take the supremum over $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ on the left
and the infimum over feasible $\mathbf{x}$ on the right. $\blacksquare$

Pause on what these two facts buy, because together they are remarkable. The
**dual problem** --- maximize $g$ over $\boldsymbol{\lambda} \succeq 0$ --- is
*always* a convex optimization problem (maximizing a concave function over a
convex set), even when the primal is a hostile non-convex mess. And its optimal
value $d^\star$ is always a *certified lower bound* on the primal optimum.
Every dual feasible point is a certificate: if you exhibit some
$(\boldsymbol{\lambda}, \boldsymbol{\nu})$ with
$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = 17$, no one can ever find a
feasible $\mathbf{x}$ with $f(\mathbf{x}) < 17$. The difference
$p^\star - d^\star \ge 0$ is the **duality gap**: the slack between the best
bound the dual can certify and the truth.

### Strong Duality and Slater's Condition

When is the gap zero? For convex problems, almost always --- the standard
sufficient condition is disarmingly mild.

**Proposition (Slater's condition implies strong duality).** *Suppose the
problem :eqref:`eq_mdl-opt-standard-problem` is convex ($f$, $g_i$ convex,
$h_j$ affine) and there exists a strictly feasible point: some
$\bar{\mathbf{x}}$ with $g_i(\bar{\mathbf{x}}) < 0$ for all $i$ and
$h_j(\bar{\mathbf{x}}) = 0$. Then $d^\star = p^\star$, and the dual optimum is
attained.* For the proof see :citet:`Boyd.Vandenberghe.2004`, section 5.3.2.

The geometric idea is worth carrying even without the proof. Map every
$\mathbf{x}$ to the pair $(g(\mathbf{x}), f(\mathbf{x}))$ of constraint value
and objective value, and consider the resulting region in the plane (with one
constraint, for drawing's sake). The primal optimum $p^\star$ is the lowest
objective value in the left half-plane $g \le 0$. Evaluating the dual function
at $\lambda$ amounts to lowering a line of slope $-\lambda$ until it *supports*
the region from below; its height at $g = 0$ is $g(\lambda)$, a lower bound on
$p^\star$ --- that is weak duality drawn as a picture. For a *convex* problem
the region is convex (more precisely, the set of points above and to the right
of it is), so some supporting line passes through the boundary point at
$(0, p^\star)$; Slater's strictly feasible point guarantees the region pokes
into the open left half-plane, ruling out the one failure mode --- a *vertical*
supporting line, which no finite $\lambda$ can represent. Then the touching
line's $\lambda$ achieves $g(\lambda) = p^\star$: strong duality. For
non-convex problems the region can be dented, every supporting line passes
*below* the dent where $p^\star$ lives, and a gap opens ---
:numref:`fig_mdl-opt-primal-dual-gap` shows both situations, and we will compute
a dented example exactly at the end of this section.

![Primal values, dual values, and the gap. The dual function is concave and sits below $p^\star$ everywhere by weak duality; primal feasible values sit above $p^\star$. For a convex problem satisfying Slater's condition the dual maximum touches $p^\star$ from below: strong duality. For a non-convex problem the dual can top out strictly below $p^\star$, leaving a duality gap.](../img/mdl-opt-primal-dual-gap.svg)
:label:`fig_mdl-opt-primal-dual-gap`

Why care, beyond elegance? Three practical reasons. First, *the dual may be the
easier problem* --- fewer variables, simpler constraints (the SVM dual below has
only sign constraints, where the primal couples all margins). Second, *the dual
certifies*: any feasible dual point bounds your suboptimality, which is how
solvers know when to stop. Third, when strong duality holds, the KKT conditions
:eqref:`eq_mdl-opt-kkt` are satisfied by the primal--dual optimal pair, so
solving the (convex) dual *is* solving the primal: recover $\mathbf{x}^\star$
as the minimizer of
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. One
more connection rounds out the toolkit: for linearly constrained problems the
dual function is a *convex conjugate* in disguise --- for
$\min f(\mathbf{x})$ s.t. $A\mathbf{x} \preceq \mathbf{b}$ one finds
$g(\boldsymbol{\lambda}) = -\mathbf{b}^\top\boldsymbol{\lambda} - f^*(-A^\top\boldsymbol{\lambda})$
with $f^*$ the conjugate of :numref:`subsec_mdl-convex-conjugate` --- so tables
of conjugates are tables of duals.

### Multipliers Are Shadow Prices

The last universal fact explains *what the numbers
$\boldsymbol{\lambda}^\star$ mean*, and it is the reading that economics,
operations research, and machine learning all share. Perturb the constraints:
for $\mathbf{u} \in \mathbb{R}^m$ let

$$
p^\star(\mathbf{u}) = \inf\,\{ f(\mathbf{x}) : g_i(\mathbf{x}) \le u_i,\; h_j(\mathbf{x}) = 0 \},
$$

so $p^\star(\mathbf{0})$ is our original optimum and $u_i > 0$ *relaxes*
constraint $i$.

**Proposition (multipliers are shadow prices).** *Suppose strong duality holds
with dual optimum $(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Then
for every $\mathbf{u}$,*

$$
p^\star(\mathbf{u}) \;\ge\; p^\star(\mathbf{0}) - \boldsymbol{\lambda}^{\star\top} \mathbf{u},
$$
:eqlabel:`eq_mdl-opt-shadow-price`

*and if $p^\star(\cdot)$ is differentiable at $\mathbf{0}$, then
$\lambda_i^\star = -\partial p^\star / \partial u_i$.*

**Proof.** Let $\mathbf{x}$ be feasible for the perturbed problem:
$g_i(\mathbf{x}) \le u_i$, $h_j(\mathbf{x}) = 0$. Then

$$
p^\star(\mathbf{0}) = g(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; f(\mathbf{x}) + \sum_i \lambda_i^\star g_i(\mathbf{x}) + \sum_j \nu_j^\star h_j(\mathbf{x})
\;\le\; f(\mathbf{x}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u},
$$

using strong duality, the definition of $g$ as an infimum, and
$\lambda_i^\star g_i(\mathbf{x}) \le \lambda_i^\star u_i$. Taking the infimum
over all such $\mathbf{x}$ gives
$p^\star(\mathbf{0}) \le p^\star(\mathbf{u}) + \boldsymbol{\lambda}^{\star\top}\mathbf{u}$,
which is :eqref:`eq_mdl-opt-shadow-price`. If $p^\star$ is differentiable at
$\mathbf{0}$, then :eqref:`eq_mdl-opt-shadow-price` says the affine function
$p^\star(\mathbf{0}) - \boldsymbol{\lambda}^{\star\top}\mathbf{u}$ minorizes
$p^\star(\mathbf{u})$ and touches it at $\mathbf{u} = \mathbf{0}$ --- so it is
the tangent there, and matching gradients gives
$\nabla p^\star(\mathbf{0}) = -\boldsymbol{\lambda}^\star$. $\blacksquare$

Read $\lambda_i^\star$ as the **price of constraint $i$**: relax it by a unit
and the achievable optimum improves by about $\lambda_i^\star$. Complementary
slackness becomes an economic platitude --- slack constraints are free
($\lambda_i = 0$); only binding constraints command a positive price. In the
ball projection, $\lambda^\star = \|\mathbf{x}_0\|/r - 1$ prices the radius; in
the hyperplane example, $-\nu^\star$ priced the offset $b$; and in the
water-filling problem below, the equality multiplier is *literally* the
marginal value of transmit power, which we will verify by finite differences.

## Worked Duals: SVM, Water-Filling, and a Visible Gap
:label:`subsec_mdl-worked-duals`

### The Support Vector Machine Dual

The support-vector machine :cite:`Cortes.Vapnik.1995` is the canonical example
where the dual is the better problem. Given linearly separable data
$(\mathbf{x}_i, y_i)$ with labels $y_i \in \{\pm 1\}$, the *maximum-margin*
separating hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ solves

$$
\min_{\mathbf{w}, b}\; \tfrac12 \|\mathbf{w}\|^2
\quad \textrm{subject to} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \;\; i = 1, \ldots, n,
$$
:eqlabel:`eq_mdl-opt-svm-primal`

since the two margin hyperplanes
$\mathbf{w}^\top\mathbf{x} + b = \pm 1$ are a distance $2/\|\mathbf{w}\|$ apart
--- minimizing $\|\mathbf{w}\|$ *is* maximizing the margin. This is a convex QP
with one inequality per data point; Slater holds for strictly separable data
(scale any separating $(\mathbf{w}, b)$ up until all margins exceed $1$), so
strong duality is guaranteed. Form the Lagrangian with multipliers
$\alpha_i \ge 0$,

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
= \tfrac12\|\mathbf{w}\|^2 + \sum_i \alpha_i \left(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\right),
$$

and minimize over the primal variables. Stationarity in $\mathbf{w}$ and $b$
gives

$$
\mathbf{w} = \sum_i \alpha_i y_i\, \mathbf{x}_i,
\qquad
\sum_i \alpha_i y_i = 0,
$$

--- the weight vector is a combination *of the data*, with coefficients the
multipliers. Substituting back yields the **SVM dual**:

$$
\max_{\boldsymbol{\alpha} \succeq 0,\ \sum_i \alpha_i y_i = 0}\;\;
\sum_i \alpha_i \;-\; \tfrac12 \sum_{i,j} \alpha_i \alpha_j\, y_i y_j\, \mathbf{x}_i^\top \mathbf{x}_j.
$$
:eqlabel:`eq_mdl-opt-svm-dual`

Two structural payoffs before any algorithm. *Complementary slackness*,
$\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$, says
$\alpha_i > 0$ only for points sitting *exactly on the margin* --- the
**support vectors**, the active constraints of the active-set picture in
:numref:`fig_mdl-opt-kkt-active-set`. All other points have $\alpha_i = 0$ and
could be deleted without moving the answer. And the data enter
:eqref:`eq_mdl-opt-svm-dual` only through inner products
$\mathbf{x}_i^\top\mathbf{x}_j$ --- replace them by a kernel evaluation
$k(\mathbf{x}_i, \mathbf{x}_j)$ and the same dual trains a nonlinear classifier,
a door the primal never opens.

Now let us *solve* the dual, with the tools of this section and nothing else.
One simplification first: the equality constraint $\sum_i \alpha_i y_i = 0$
came from differentiating in the unpenalized offset $b$. If we instead fold the
offset into the weights --- append a constant feature,
$\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ and
$\tilde{\mathbf{w}} = (\mathbf{w}, b)$, so the primal becomes
$\min \tfrac12\|\tilde{\mathbf{w}}\|^2$ subject to
$y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$ --- then there is no
free $b$ to differentiate out, the equality constraint never appears, and the
dual feasible set is the plain nonnegative orthant
$\boldsymbol{\alpha} \succeq 0$, on which projection is a coordinate-wise clip.
(The price is honesty in labeling: this variant also regularizes the offset, so
its maximum-margin boundary can differ slightly from
:eqref:`eq_mdl-opt-svm-primal`'s; the duality structure is identical.) Writing
$Q_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$, the dual is

$$
\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top Q\, \boldsymbol{\alpha},
$$

a concave quadratic over the orthant --- a problem
:eqref:`eq_mdl-opt-pgd` was born for. Projected gradient *ascent* iterates
$\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - Q\boldsymbol{\alpha}))$
with $\eta = 1/\lambda_{\max}(Q)$, the dual's own descent-lemma step size. The
cell below runs it on a small 2-D toy problem and then audits *everything* this
section promised: the recovered $(\mathbf{w}^\star, b^\star)$ separates the
data, complementary slackness holds to machine precision, and the dual value
meets the primal value --- strong duality, observed.

```{.python .input #constrained-svm-dual}
X = np.array([[1.0, 1.5], [2.0, 0.5], [2.5, 2.0], [1.5, 3.0],
              [-1.0, -0.5], [-2.0, 1.0], [-1.5, -2.0], [0.5, -1.5]])
y_pm = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
Xa = np.hstack([X, np.ones((len(X), 1))])     # fold the offset b into w
G = y_pm[:, None] * Xa
Q = G @ G.T                                   # Q_ij = y_i y_j x_i^T x_j
alpha = np.zeros(len(X))
eta = 1.0 / np.linalg.eigvalsh(Q).max()       # 1/L for the dual objective
for _ in range(5000):                         # projected gradient ascent
    alpha = np.maximum(0.0, alpha + eta * (1.0 - Q @ alpha))
w = G.T @ alpha                               # w* = sum_i alpha_i y_i x_i
margins = y_pm * (Xa @ w)
dual = alpha.sum() - 0.5 * alpha @ Q @ alpha
primal = 0.5 * w @ w
print('alpha* =', alpha.round(4))
print('w* =', w[:2].round(4), ' b* =', w[2].round(4),
      ' separates data:', bool((np.sign(Xa @ w) == y_pm).all()))
print('margins y_i (w^T x_i + b) =', margins.round(4))
print('KKT: comp. slack', f'{np.abs(alpha * (margins - 1.0)).max():.1e}',
      '| worst primal feas.', f'{(1.0 - margins).max():.1e}')
print(f'primal = {primal:.6f}, dual = {dual:.6f}, gap = {abs(primal - dual):.1e}')
```

Read the printout against the theory. Four of the eight multipliers are
strictly positive, and *exactly those four points* have margin $1.0000$ --- the
support vectors, pinned to the margin by complementary slackness; the other
four points sit at margins $1.29$--$2.43$ with $\alpha_i = 0$. The solution is
exact enough to recognize:
$\mathbf{w}^\star = (\tfrac47, \tfrac47)$, $b^\star = -\tfrac37$, with
$p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$ and a primal--dual gap at
$10^{-16}$ --- far inside the $10^{-4}$ we asked for. As a bonus, the printout
verifies an identity you will prove in Exercise 6:
$\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ at the optimum, which
is why the dual and primal values coincide line for line.

### Water-Filling

Our second dual is a closed form with a famous picture. A transmitter splits a
power budget $P$ across $n$ independent channels; channel $i$ has noise level
$n_i > 0$, and the achievable communication rate is
$\sum_i \log(1 + p_i / n_i)$ :cite:`Cover.Thomas.1999`. The allocation problem,

$$
\max_{\mathbf{p}}\; \sum_{i=1}^n \log\left(1 + \frac{p_i}{n_i}\right)
\quad \textrm{subject to} \quad \mathbf{p} \succeq 0, \;\; \sum_i p_i = P,
$$
:eqlabel:`eq_mdl-opt-waterfilling`

is convex (concave objective, affine constraints) with Slater trivially
satisfied, so KKT pins down the global optimum. With multiplier $\mu$ for the
budget and $\lambda_i \ge 0$ for $p_i \ge 0$, stationarity reads

$$
\frac{1}{n_i + p_i} = \mu - \lambda_i.
$$

On channels with $p_i > 0$, complementary slackness kills $\lambda_i$ and
forces $n_i + p_i = 1/\mu$ --- *the same constant for every active channel*.
On dry channels, $p_i = 0$ requires $\lambda_i = \mu - 1/n_i \ge 0$, i.e.
$n_i \ge 1/\mu$. Writing $w = 1/\mu$ for that constant:

$$
p_i^\star = \max(w - n_i,\, 0),
\qquad \textrm{with } w \textrm{ set by } \sum_i \max(w - n_i,\, 0) = P.
$$

This is the **water-filling** solution. Picture each channel as a basin whose
floor sits at height $n_i$ and pour in $P$ units of water: it settles at a
common level $w$, filling the deep (quiet) channels most, and never reaching
basins whose floor is above the waterline. The level is the unique root of a
continuous, nondecreasing, piecewise-linear function of $w$, so *bisection*
finds it --- note the structural rhyme with the simplex projection's threshold
$\tau$, which is the same KKT phenomenon wearing a different objective. The
shadow-price proposition gives the multiplier its engineering meaning:
$\mu^\star = 1/w$ is the marginal rate bought by the *next* watt of power,
which the cell checks by re-solving at $P \pm 10^{-4}$.

```{.python .input #constrained-water-filling}
noise = np.array([0.1, 0.4, 0.8, 1.6, 2.5])   # channel noise floors
P = 3.0                                       # total power budget

def water_fill(P, noise, steps=60):
    lo, hi = noise.min(), noise.min() + P     # bracket for the water level
    for _ in range(steps):                    # bisection on the level
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.maximum(0, mid - noise).sum() < P else (lo, mid)
    level = 0.5 * (lo + hi)
    return level, np.maximum(0.0, level - noise)

level, p = water_fill(P, noise)
print('water level w =', f'{level:.6f}', ' p* =', p.round(4),
      ' total =', f'{p.sum():.6f}')
print('wet channels filled to common level:', (noise + p)[p > 0].round(6))
print('dry channels above the waterline:   ', bool((noise[p == 0] >= level).all()))
rate = lambda q: np.log(1.0 + q / noise).sum()
dP = 1e-4                                     # shadow price, two ways
mu = 1.0 / level
sens = (rate(water_fill(P + dP, noise)[1])
        - rate(water_fill(P - dP, noise)[1])) / (2 * dP)
print(f'mu = 1/w = {mu:.6f} vs finite-difference dU*/dP = {sens:.6f}')
```

The budget is met to ten digits, the three wet channels are filled to the
*identical* level $w = 1.4333$, the two channels with floors $1.6$ and $2.5$
stay dry exactly as dual feasibility demands --- and the multiplier
$\mu = 1/w = 0.6977$ matches the finite-difference sensitivity of the optimal
rate to the budget through six digits. The shadow price is not a metaphor; it
is a number you can measure by perturbing the constraint.

### A Duality Gap You Can See

Weak duality never fails; strong duality can. To watch it fail, take a problem
small enough to solve by looking at it: on the interval $x \in [0, 1]$,

$$
\min_x\; f_0(x) = -x^2
\quad \textrm{subject to} \quad f_1(x) = x - \tfrac12 \le 0.
$$

The objective is *concave* --- this is a non-convex problem --- and the
feasible region is $[0, \tfrac12]$, so the primal optimum is at the boundary:
$p^\star = f_0(\tfrac12) = -\tfrac14$. The dual function is computable by hand:
$\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ is concave in $x$, so
its minimum over $[0, 1]$ is at an endpoint, giving
$g(\lambda) = \min(-\lambda/2,\; \lambda/2 - 1)$ --- piecewise linear, concave
(as it must be), maximized at $\lambda = 1$ with
$d^\star = -\tfrac12 < -\tfrac14 = p^\star$: a duality gap of exactly
$\tfrac14$. The cell verifies all of it numerically and plots the dual function
bumping its head on a ceiling strictly below $p^\star$.

```{.python .input #constrained-duality-gap}
xs = np.linspace(0.0, 1.0, 4001)              # the domain D = [0, 1]
f0, f1 = -xs**2, xs - 0.5                     # objective and constraint
p_star = f0[f1 <= 0.0].min()                  # primal optimum: -1/4
lams = np.linspace(0.0, 3.0, 601)
g = np.array([(f0 + lam * f1).min() for lam in lams])  # dual function
d_star, lam_star = g.max(), lams[g.argmax()]
print(f'p* = {p_star:.4f},  d* = {d_star:.4f} at lambda = {lam_star:.2f},'
      f'  gap = {p_star - d_star:.4f}')
print('weak duality g(lambda) <= p* everywhere:',
      bool((g <= p_star + 1e-12).all()))
d2l.plot(lams, [g, np.full_like(lams, p_star)], 'lambda', 'value',
         legend=['dual function g(lambda)', 'primal optimum p*'])
```

Note the trap this example disarms. A strictly feasible point exists
($x = 0$ has $f_1 = -\tfrac12 < 0$), yet the gap is real --- because Slater's
condition certifies strong duality *only for convex problems*, and $f_0 = -x^2$
is not convex. In the supporting-line picture of
:numref:`fig_mdl-opt-primal-dual-gap`, the curve
$\{(f_1(x), f_0(x))\} = \{(u, -(u + \tfrac12)^2)\}$ is dented from below, every
supporting line passes under the dent at $u = 0$, and the best of them ---
slope $-1$, our $\lambda^\star = 1$ --- certifies only $-\tfrac12$. This is the
general situation for deep learning's loss surfaces: duals of non-convex
training problems still give honest *lower bounds* (and are the engine of
verification and relaxation methods), but the bound need not be tight.

### Coda: A Map of Problem Classes

The examples above were all quadratic programs, but they sit inside a standard
hierarchy of convex problem classes, worth knowing because it tells you which
off-the-shelf solver applies and what to expect from it:

| Class | Template | Deep-learning sightings |
|---|---|---|
| **LP** | linear $f$, constraints $A\mathbf{x} \preceq \mathbf{b}$ | $\ell_1$ / $\ell_\infty$ reformulations, optimal transport |
| **QP** | convex quadratic $f$, affine constraints | ridge, lasso, SVM dual, simplex projection, trust-region step |
| **SOCP** | cones $\|A\mathbf{x} + \mathbf{b}\| \le \mathbf{c}^\top\mathbf{x} + d$ | max-norm weight constraints, robust losses |
| **SDP** | matrix variable $X \succeq 0$ | spectral-norm bounds, nuclear-norm relaxations, certification |

Each class contains the previous ($\mathrm{LP} \subseteq \mathrm{QP} \subseteq
\mathrm{SOCP} \subseteq \mathrm{SDP}$), and all are solvable to high accuracy
in polynomial time by interior-point methods
:cite:`Boyd.Vandenberghe.2004,Nesterov.2018`. Training a deep network lies
*outside* the hierarchy --- the composition of layers destroys convexity
(:numref:`sec_mdl-convexity`) --- but, as this section has shown, many of its
*sub-problems* (projections, clipped updates, trust regions, last-layer fits)
and many of its *analyses* live inside, where multipliers, KKT, and duality do
exact work.

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
* **Projections** onto closed convex sets are unique and **nonexpansive**, and
  **projected gradient descent** $\mathbf{x} \leftarrow \Pi_C(\mathbf{x} - \eta\nabla f)$
  inherits gradient descent's guarantees; its fixed points are exactly the
  constrained first-order optima. The simplex projection is a KKT-derived
  sort-and-threshold (sparsemax).
* The **dual function** $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}$
  is *always concave*, and **weak duality** $d^\star \le p^\star$ *always*
  holds: dual points are certificates. **Slater's condition** (convexity plus a
  strictly feasible point) closes the gap; without convexity a gap can survive
  even with strictly feasible points.
* Multipliers are **shadow prices**: $\lambda_i^\star = -\partial p^\star/\partial u_i$
  measures what relaxing constraint $i$ is worth --- exactly $1/w$ in
  water-filling, and the reason slack constraints cost nothing.
* The SVM dual (support vectors = active constraints, kernels via inner
  products) and water-filling (pour to a common level, bisection on the level)
  show the full pipeline; LP/QP/SOCP/SDP is the solver map for the convex
  sub-problems of deep learning.

## Exercises

1. Derive the Lagrange condition for *two* equality constraints: if
   $\mathbf{x}^\star$ is a local minimum of $f$ on
   $\{h_1 = 0\} \cap \{h_2 = 0\}$ and $\nabla h_1(\mathbf{x}^\star)$,
   $\nabla h_2(\mathbf{x}^\star)$ are linearly independent, show
   $\nabla f(\mathbf{x}^\star) \in \mathrm{span}\{\nabla h_1, \nabla h_2\}$ by
   the no-feasible-descent argument. Then revisit the counterexample
   ($\min x_1$ s.t. $x_1^2 + x_2^2 = 0$) and identify exactly which step of
   your proof fails there.
2. Maximize the entropy $-\sum_i p_i \log p_i$ over the probability simplex
   using one multiplier for $\sum_i p_i = 1$, and show the optimum is the
   uniform distribution. Where did you use that the $p_i > 0$ constraints are
   inactive at the optimum? (Compare :numref:`subsec_mdl-jensen`, which reaches
   the same conclusion through Jensen's inequality.)
3. Write all four KKT conditions for
   $\min\, \tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ s.t. $\|\mathbf{x}\|^2 \le r^2$
   and verify the clipping solution and $\lambda^\star = \max(0, \|\mathbf{x}_0\|/r - 1)$.
   Then exhibit a one-dimensional *non-convex* problem with a KKT point that is
   a local maximum --- necessity without sufficiency.
4. Show that $\mathbf{x}^\star$ is a fixed point of the projected gradient
   update :eqref:`eq_mdl-opt-pgd` with $C = \{\mathbf{x} : \mathbf{x} \succeq 0\}$
   if and only if for each coordinate either $\partial f/\partial x_i = 0$ and
   $x_i^\star \ge 0$, or $\partial f/\partial x_i > 0$ and $x_i^\star = 0$ ---
   and check this is precisely KKT for the constraints $-x_i \le 0$.
5. Prove from the definitions that the dual function of *any* primal problem
   is concave, and that weak duality holds (reproduce the proofs without
   looking). Then compute the dual of the equality-constrained QP
   $\min\, \tfrac12\mathbf{x}^\top A \mathbf{x} - \mathbf{b}^\top\mathbf{x}$
   s.t. $C\mathbf{x} = \mathbf{d}$ with $A \succ 0$ in closed form, and verify
   $d^\star = p^\star$ by solving both sides for
   $A = \mathrm{diag}(1, 2)$, $\mathbf{b} = (1, 1)$, $C = (1\;\; 1)$, $d = 1$.
6. For the offset-folded SVM dual
   $\max_{\boldsymbol{\alpha} \succeq 0} \mathbf{1}^\top\boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top Q \boldsymbol{\alpha}$,
   show that at the optimum
   $\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ where
   $\tilde{\mathbf{w}}^\star = \sum_i \alpha_i^\star y_i \tilde{\mathbf{x}}_i$.
   (*Hint:* multiply the KKT stationarity condition
   $(\mathbf{1} - Q\boldsymbol{\alpha}^\star)_i = -\lambda_i^\star \le 0$ by
   $\alpha_i^\star$, sum, and use complementary slackness.) Conclude that the
   dual optimal value equals $\tfrac12\|\tilde{\mathbf{w}}^\star\|^2$, as the
   `#constrained-svm-dual` cell observed numerically.
7. In water-filling :eqref:`eq_mdl-opt-waterfilling`, show that the optimal
   rate $U^\star(P)$ is concave and piecewise smooth in $P$, that the water
   level $w(P)$ is piecewise linear and increasing, and that the shadow price
   $\mu = 1/w$ therefore *decreases* with the budget --- diminishing returns.
   At which values of $P$ (for the noise floors in the
   `#constrained-water-filling` cell) does the slope change, and why?
8. Rerun the duality-gap demo with the objective $f_0(x) = +x^2$ in place of
   $-x^2$ (everything else unchanged). Compute $p^\star$ and $d^\star$ by hand
   and numerically, and explain in terms of Slater's condition why the gap is
   now zero.

## Discussions

This section is the chapter's bridge to the main book. The penalty form of
regularization --- add $\lambda\|\mathbf{w}\|^2$ to the loss --- and the
constraint form --- minimize the loss subject to $\|\mathbf{w}\|^2 \le r^2$ ---
are linked precisely by the Lagrangian: the weight-decay coefficient of
:numref:`sec_weight_decay` is the multiplier of the norm constraint, and
sweeping one traces out the solutions of the other. Projections power gradient
clipping and constrained updates; the simplex projection is sparsemax;
attention itself can be read as a regularized argmax over the simplex. Within
this part, the section leans on :numref:`sec_mdl-convexity` (convexity, the
conjugate) and :numref:`sec_mdl-gradient-based-optimization` (descent lemma,
rates), and its conditioning consequences continue in
:numref:`sec_mdl-numerical-stability-conditioning`.

[Discussions](https://d2l.discourse.group/t/constrained-optimization-duality)

<!-- slides -->

::: {.slide}
::: {.cover}
[Mathematics for Deep Learning · Optimization]{.kicker}

Reading a constrained problem and its dual<br>**Lagrange multipliers · KKT · projections · duality**.
:::
:::

::: {.slide title="One idea, three guises"}
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
Every multiplier is a **price**: what relaxing its constraint is worth.
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

::: {.slide title="No feasible descent forces tangency"}
[Lagrange]{.kicker}

::: {.cols .vc}
::: {.col}
Where a level set of $f$ *crosses* the constraint, $\nabla f$ has a
component **along** it, so sliding feasibly lowers $f$. Only where the
curves are **tangent** does every feasible move stall, and tangency of
the curves means their normals are parallel:

$$\nabla f(\mathbf{x}^\star) + \nu^\star\,\nabla g(\mathbf{x}^\star) = \mathbf{0}.$$

Valid wherever $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$, the
**constraint qualification**.
:::

::: {.col .fig}
@fig:mdl-opt-lagrange-tangency
:::
:::
:::

::: {.slide title="The Lagrangian packages it"}
[Lagrange]{.kicker}

Parallel gradients *and* feasibility become joint stationarity of one
function of more variables:

$$\mathcal{L}(\mathbf{x}, \nu) = f(\mathbf{x}) + \nu\, g(\mathbf{x}),
\qquad
\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0},\;\; \partial_\nu \mathcal{L} = g = 0.$$

. . .

Closest point on a hyperplane $\mathbf{a}^\top\mathbf{x} = b$: stationarity
forces $\mathbf{x}$ *along the normal* $\mathbf{a}$, giving
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

::: {.slide title="Active, inactive, and a one-way push"}
[KKT]{.kicker}

::: {.cols .vc}
::: {.col}
With $g_i(\mathbf{x}) \le 0$, each constraint is **inactive**
($g_i < 0$, locally irrelevant) or **active** ($g_i = 0$, pushing back
like an equality). At the optimum $-\nabla f$ is a *nonnegative*
combination of the active outward normals, in their **cone**.

An inequality multiplier carries a sign: $\lambda_i \ge 0$. It can push
*one way only*, into the feasible side.
:::

::: {.col .fig}
@fig:mdl-opt-kkt-active-set
:::
:::
:::

::: {.slide title="Four conditions, one workhorse"}
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
zero). The pattern of zero multipliers tells you which constraints shaped
the answer.
:::
:::

. . .

Convex $f, g_i$ and affine $h_j$: a KKT point is a **global** minimum, by
a three-line convexity argument. KKT *is* the answer.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Projections]{.dtitle}

[making constraints an algorithm]{.dsub}
:::
:::

::: {.slide title="Project, then it is an algorithm"}
[Projections]{.kicker}

::: {.cols .vc}
::: {.col}
$\Pi_C(\mathbf{y})$ is the nearest feasible point. On a convex set it is
**nonexpansive** ($\|\Pi_C\mathbf{x} - \Pi_C\mathbf{y}\| \le \|\mathbf{x} - \mathbf{y}\|$),
so appending it keeps gradient descent's guarantees:

$$\mathbf{x}_{t+1} = \Pi_C\!\left(\mathbf{x}_t - \eta\,\nabla f(\mathbf{x}_t)\right).$$

Its **fixed points are exactly the constrained optima**, equivalent to KKT.
:::

::: {.col .narrow}
::: {.d2l-note}
You have run **PGD** in disguise: gradient **clipping** projects onto a
ball; non-negativity is a projection onto the orthant; max-norm weight
caps project each row.
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

Soft-threshold then renormalize. This is **sparsemax**: genuinely sparse
attention weights, with complementary slackness choosing the zeros.

@!constrained-simplex-projection

::: {.d2l-note}
Three coordinates came back **exactly zero**, the active sign constraints;
every KKT residual sits at machine precision.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Duality]{.dtitle}

[bounds for free, and shadow prices]{.dsub}
:::
:::

::: {.slide title="The dual function: bounds for free"}
[Duality]{.kicker}

::: {.cols .vc}
::: {.col}
Minimize the Lagrangian over $\mathbf{x}$ to promote the multipliers to
variables:

$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}}\, \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).$$

- *Always concave* (an infimum of affine functions), **even for a
  non-convex primal**.
- **Weak duality**: $d^\star \le p^\star$ always. Every dual point
  *certifies* a lower bound.
:::

::: {.col .fig}
@fig:mdl-opt-primal-dual-gap
:::
:::
:::

::: {.slide title="Strong duality and shadow prices"}
[Duality]{.kicker}

::: {.cols}
::: {.col}
**Slater:** convex problem $+$ one strictly feasible point $\Rightarrow$
$d^\star = p^\star$, the gap closes.

The dual is *always convex*, often the easier problem, and at strong
duality solving it *solves the primal*.
:::

::: {.col}
::: {.d2l-note .rule}
**Shadow price:**

$$\lambda_i^\star = -\frac{\partial p^\star}{\partial u_i}.$$

Relax constraint $i$ by a unit and the optimum improves by $\lambda_i^\star$.
:::
:::
:::

Slack constraints cost nothing; only binding ones command a price.
:::

::: {.slide title="Worked: the SVM dual"}
[Duality at work]{.kicker}

Eliminate $(\mathbf{w}, b)$ from the max-margin Lagrangian and the dual is
a concave QP over the orthant, made for projected gradient *ascent*:

$$\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top Q\, \boldsymbol{\alpha},
\qquad
\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - Q\boldsymbol{\alpha})).$$

@!constrained-svm-dual

::: {.d2l-note}
The four nonzero $\alpha_i$ are exactly the points at **margin 1**, the
support vectors. Primal meets dual at $10^{-16}$: strong duality, observed.
:::
:::

::: {.slide title="Worked: water-filling"}
[Duality at work]{.kicker}

::: {.cols .vc}
::: {.col}
Allocate power across noisy channels. KKT pours power until every wet
channel reaches a **common level** $w$, leaving high-noise channels dry:

$$p_i^\star = \max(w - n_i,\, 0),
\qquad \textstyle\sum_i \max(w - n_i, 0) = P.$$

Bisection on $w$ finds it. The multiplier $\mu = 1/w$ is the marginal
value of power, measured here by finite differences.
:::

::: {.col .fig}
@!constrained-water-filling
:::
:::
:::

::: {.slide title="A duality gap you can see"}
[Duality at work]{.kicker}

::: {.cols .vc}
::: {.col}
Weak duality never fails; strong duality can. Minimize the *concave*
$f_0 = -x^2$ on $[0,1]$ s.t. $x \le \tfrac12$:

$$p^\star = -\tfrac14, \qquad d^\star = -\tfrac12.$$

A strictly feasible point exists, yet the gap is real: Slater certifies
strong duality only for **convex** problems.
:::

::: {.col .fig}
@!constrained-duality-gap
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- A constrained optimum has **no feasible descent direction**, made
  precise by Lagrange and **KKT**; complementary slackness finds the
  active set.
- **Projections** onto convex sets are nonexpansive; projected GD is
  gradient descent plus a snap back to feasibility.
:::

::: {.col}
- The **dual** is always concave and always a lower bound; Slater closes
  the gap; multipliers are **shadow prices**.
- **SVM dual** and **water-filling** are duality you can run;
  non-convexity leaves a gap you can see.
:::
:::

::: {.d2l-note}
Many sub-problems of deep learning, projections, clipped updates,
last-layer fits, live inside this convex toolkit.
:::
:::
