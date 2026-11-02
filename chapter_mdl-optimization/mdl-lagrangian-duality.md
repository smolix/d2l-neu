# Duality
:label:`sec_mdl-lagrangian-duality`

<!-- Placeholder opening; the section writer replaces it. -->
Every choice of nonnegative multipliers gives a lower bound on the optimal
value of a constrained problem. This section builds that bound from the
convex conjugate and the Lagrange dual function, shows when it is tight,
reads the optimal multipliers as prices, and works the bound out on
complete examples.

```{.python .input #mdl-lagrangian-duality-duality}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #mdl-lagrangian-duality-duality}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## The Convex Conjugate
:label:`subsec_mdl-convex-conjugate`

The convex conjugate connects this section with duality (:numref:`sec_mdl-constrained-optimization-duality`)
and the variational representations of information theory. From here on we
allow $f$ to take the value $+\infty$; convexity for such an extended-real
function is read on its epigraph (the chord inequality is required wherever
the values are finite), and the conjugates below produce $+\infty$ naturally.
The **convex
conjugate** (or Fenchel--Legendre transform) of $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$
is

$$
f^*(\mathbf{y}) \;=\; \sup_{\mathbf{x}} \;\left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right).
$$
:eqlabel:`eq_mdl-opt-conjugate`

Geometrically, $f^*(\mathbf{y})$ asks: among all affine functions with slope
$\mathbf{y}$, what is the largest intercept compatible with remaining below the graph of $f$?
(It records the negative of that intercept.) The conjugate thus encodes $f$ by
its supporting lines instead of its values, and for **closed** convex $f$
(closed meaning the epigraph is a closed set, equivalently $f$ is lower
semicontinuous) the
encoding is lossless: $f^{**} = f$ (:citet:`Boyd.Vandenberghe.2004`, section
3.3). For non-convex $f$ the double transform returns the largest
closed convex function below it, the **convex envelope**, which is what
convex *relaxations* optimize in place of $f$.

Two properties are immediate. First, $f^*$ is *always* convex, whatever
$f$ is: for fixed $\mathbf{x}$ the expression
$\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ is affine in $\mathbf{y}$, and rule
3 says a supremum of affine functions is convex. The same one-line argument,
in the next section, makes the dual function of
:numref:`subsec_mdl-lagrangian-duality` concave, and that is no coincidence:
for linearly constrained problems the dual function can be written in terms of a
conjugate.
Second, the definition rearranges into the **Fenchel--Young inequality**,

$$
\mathbf{y}^\top \mathbf{x} \;\le\; f(\mathbf{x}) + f^*(\mathbf{y})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y},
$$
:eqlabel:`eq_mdl-opt-fenchel-young`

with equality exactly when $\mathbf{x}$ attains the supremum; for
differentiable $f$, when $\mathbf{y} = \nabla f(\mathbf{x})$.

The simplest example sets the pattern: for $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|^2$,
the supremum of $\mathbf{y}^\top\mathbf{x} - \tfrac12\|\mathbf{x}\|^2$ is attained at
$\mathbf{x} = \mathbf{y}$, giving $f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$:
the squared norm is its own conjugate, and Fenchel--Young becomes the familiar
$\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$.

The pair that matters for deep learning is **log-sum-exp $\leftrightarrow$
negative entropy**. Compute
$\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})$.
If $\mathbf{p}$ has a negative coordinate, send that $x_i \to -\infty$ and the
objective is unbounded; if $\mathbf{1}^\top\mathbf{p} \neq 1$, take $\mathbf{x} = c\mathbf{1}$
off to $\pm\infty$ and shift invariance does the same. So the supremum is finite
only for $\mathbf{p}$ in the simplex, where stationarity demands
$\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$, solved by
$\mathbf{x} = \log \mathbf{p}$, and the value is
$\sum_i p_i \log p_i$. The stationarity argument covers $\mathbf{p}$ in the
relative interior of the simplex (softmax outputs are strictly positive); for
boundary $\mathbf{p}$, with some $p_i = 0$, no maximizer exists, but sending
those $x_i \to -\infty$ approaches the same value, so the formula holds on the
whole simplex with the convention $0 \log 0 = 0$. Therefore

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

The dual formula expresses log-sum-exp as the maximum of an entropy-regularized
linear score over distributions, and the maximizing $\mathbf{p}$ is the softmax.
Two consequences are already on this page. Softmax is the gradient of
$\mathrm{lse}$, proved in the Hessian proposition above; and softmax is the
*maximum-entropy* answer, since dropping the bonus leaves
$\max_{\mathbf{p} \in \Delta} \mathbf{p}^\top\mathbf{x}$, whose maximizer is a
hard, one-hot $\arg\max$, and the entropy term replaces this hard selection with a distribution. The identity
is also the prototype
for the variational representations of divergences: Donsker--Varadhan and the
$f$-GAN duals in :numref:`sec_mdl-divergences-distances` are this same
conjugate construction applied to KL and its relatives.

## The Dual Problem
:label:`subsec_mdl-lagrangian-duality`

### The Lagrange Dual Function

So far, multipliers were unknowns solved alongside $\mathbf{x}$. Duality
treats them as optimization variables. Fix
$(\boldsymbol{\lambda}, \boldsymbol{\nu})$ with $\boldsymbol{\lambda} \succeq 0$
(elementwise, in the vector sense of the $\succeq$ convention of
:numref:`sec_mdl-convexity`)
and minimize the Lagrangian :eqref:`eq_mdl-opt-lagrangian` over $\mathbf{x}$
*unconstrained*:

$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}}\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$
:eqlabel:`eq_mdl-opt-dual-function`

The infimum is over all of $\mathbb{R}^n$ and need not be attained or even
finite: for many multiplier values the Lagrangian is unbounded below and
$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$. The dual problem below
implicitly maximizes over the region where $g$ is finite (its *effective
domain*); in the SVM dual, this mechanism produces an equality constraint.

The **Lagrange dual function** has two properties that require no assumptions
on $f$, $g_i$, or $h_j$.

**Proposition (the dual function is concave).** *For any primal problem,
$g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ in :eqref:`eq_mdl-opt-dual-function`
is concave.*

**Proof.** For each fixed $\mathbf{x}$, the map
$(\boldsymbol{\lambda}, \boldsymbol{\nu}) \mapsto \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$
is *affine*: the multipliers enter linearly. The dual function is the
pointwise infimum of this family of affine functions, and a pointwise infimum
of affine functions is concave: its **hypograph** (the set of points on or
below its graph, the mirror image of the epigraph of
:numref:`sec_mdl-convexity`) is the intersection of the
half-spaces below the affine functions, an intersection of convex sets.
$\blacksquare$

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

The
**dual problem** maximizes $g$ over $\boldsymbol{\lambda} \succeq 0$, and it is
*always* a convex optimization problem (maximizing a concave function over a
convex set), even when the primal is non-convex. Its optimal
value $d^\star$ is always a *certified lower bound* on the primal optimum.
Every dual feasible point provides a certificate: if
$(\boldsymbol{\lambda}, \boldsymbol{\nu})$ satisfies
$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = 17$, then there is no feasible
$\mathbf{x}$ with $f(\mathbf{x}) < 17$. The difference
$p^\star - d^\star \ge 0$ is the **duality gap** between the primal and dual
optimal values.

### Strong Duality and Slater's Condition

Slater's condition gives a standard sufficient condition for a convex problem
to have zero duality gap: it requires one strictly feasible point.

**Proposition (Slater's condition implies strong duality).** *Suppose the
problem :eqref:`eq_mdl-opt-standard-problem` is convex ($f$, $g_i$ convex,
$h_j$ affine) and there exists a strictly feasible point: some
$\bar{\mathbf{x}}$ with $g_i(\bar{\mathbf{x}}) < 0$ for all $i$ and
$h_j(\bar{\mathbf{x}}) = 0$. Then $d^\star = p^\star$, and the dual optimum is
attained.* The condition is due to :citet:`Slater.1950`; for the proof see
:citet:`Boyd.Vandenberghe.2004`, section 5.3.2. In the refined form of the
condition, strictness is required only of the non-affine $g_i$: affine
inequality constraints may hold with equality at $\bar{\mathbf{x}}$.

The geometric idea can be sketched without the proof. Map every
$\mathbf{x}$ to the pair $(g(\mathbf{x}), f(\mathbf{x}))$ of constraint value
and objective value, and consider the resulting region in the plane (with one
constraint, for drawing's sake). The primal optimum $p^\star$ is the lowest
objective value in the left half-plane $g \le 0$. Evaluating the dual function
at $\lambda$ amounts to lowering a line of slope $-\lambda$ until it *supports*
the region from below; its height at $g = 0$ is $g(\lambda)$, a lower bound on
$p^\star$: that is weak duality drawn as a picture. For a *convex* problem
the region is convex (more precisely, the set of points above and to the right
of it is), so some supporting line passes through the boundary point at
$(0, p^\star)$; the granted fact of :numref:`sec_mdl-convexity` supplies a
supporting line at a boundary point of a convex set, the same fact
that gives convex functions their subgradients. Slater's strictly feasible
point ensures that the region intersects
the open left half-plane, ruling out a *vertical*
supporting line, which no finite $\lambda$ can represent. Then the touching
line's $\lambda$ achieves $g(\lambda) = p^\star$: strong duality. For
non-convex problems, the boundary can be nonconvex, so every supporting line
may pass below the point associated with $p^\star$, producing a gap;
:numref:`fig_mdl-opt-primal-dual-gap` shows both situations, and we compute
an explicit nonconvex example at the end of this section.

![The supporting-line geometry of duality. Every $x$ maps to the pair $(g(x), f_0(x))$ of constraint and objective value, plotted as $(u, t)$; the shaded region $G$ collects them. Evaluating the dual at $\lambda$ lowers a line of slope $-\lambda$ until it supports $G$ from below, and its height over $u = 0$ is the dual value. Left: for a convex problem with Slater's condition the best line is tangent at $(0, p^\star)$: strong duality. Right: for this section's non-convex example, no supporting line reaches the boundary point associated with the primal optimum, and the best certifies only $d^\star < p^\star$: a positive duality gap.](../img/mdl-opt-primal-dual-gap.svg)
:label:`fig_mdl-opt-primal-dual-gap`

Three consequences are useful. First, *the dual may be the
easier problem*: fewer variables, simpler constraints (the SVM dual below has
only sign constraints, where the primal couples all margins). Second, *the dual
certifies*: any feasible dual point bounds your suboptimality, which is how
solvers know when to stop. Third, when strong duality holds, the KKT conditions
:eqref:`eq_mdl-opt-kkt` are satisfied by the primal--dual optimal pair, so
solving the (convex) dual *is* solving the primal: recover $\mathbf{x}^\star$
as the minimizer of
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. One
more connection rounds out the toolkit: for linearly constrained problems the
dual function is a *convex conjugate* in disguise. For
$\min f(\mathbf{x})$ s.t. $A\mathbf{x} \preceq \mathbf{b}$, the Lagrangian is
$f(\mathbf{x}) + \boldsymbol{\lambda}^\top(A\mathbf{x} - \mathbf{b})$, and
grouping the terms in $\mathbf{x}$,

$$
\begin{aligned}
g(\boldsymbol{\lambda})
&= \inf_{\mathbf{x}} \left[ f(\mathbf{x}) + (A^\top\boldsymbol{\lambda})^\top\mathbf{x} \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -\sup_{\mathbf{x}} \left[ (-A^\top\boldsymbol{\lambda})^\top\mathbf{x} - f(\mathbf{x}) \right] - \mathbf{b}^\top\boldsymbol{\lambda} \\
&= -f^*(-A^\top\boldsymbol{\lambda}) - \mathbf{b}^\top\boldsymbol{\lambda},
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-dual-conjugate`

with $f^*$ the conjugate of :numref:`subsec_mdl-convex-conjugate`. Tables
of conjugates are tables of duals.

### Duality as a Saddle Point

Strong duality has an equivalent formulation that also appears in several
modern training objectives. The primal problem can be written in min--max
form: for any $\mathbf{x}$,

$$
\sup_{\boldsymbol{\lambda} \succeq 0,\, \boldsymbol{\nu}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
= \begin{cases} f(\mathbf{x}) & \textrm{if } \mathbf{x} \textrm{ is feasible}, \\ +\infty & \textrm{otherwise}, \end{cases}
$$

since any violated constraint lets the corresponding multiplier drive the
supremum to $+\infty$, while for feasible $\mathbf{x}$ the best the multipliers
can do is vanish on every slack constraint. So
$p^\star = \inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \mathcal{L}$
and, by definition,
$d^\star = \sup_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \inf_{\mathbf{x}} \mathcal{L}$:
weak duality is the universal inequality $\sup \inf \le \inf \sup$, and
strong duality says that the two optimization orders have equal values. This
equality corresponds to a saddle point.

**Proposition (strong duality is a saddle point).** *Suppose strong duality
holds for :eqref:`eq_mdl-opt-standard-problem` with both optima attained, at a
primal optimum $\mathbf{x}^\star$ and a dual optimum
$(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$. Then
$(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$ is a*
**saddle point** *of the Lagrangian: for all $\mathbf{x}$ and all
$\boldsymbol{\lambda} \succeq 0$, $\boldsymbol{\nu}$,*

$$
\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}, \boldsymbol{\nu})
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star).
$$
:eqlabel:`eq_mdl-opt-saddle-point`

*Conversely, any saddle point of $\mathcal{L}$ over
$\mathbf{x} \times (\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu})$ closes
the duality gap, with saddle value $p^\star = d^\star$.*

**Proof.** Using the definitions and weak duality,

$$
d^\star
= g(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
= \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)
\;\le\; f(\mathbf{x}^\star)
= p^\star,
$$

where the last inequality holds because $\lambda_i^\star \ge 0$ and
$g_i(\mathbf{x}^\star) \le 0$ make every added term nonpositive. Strong
duality makes every inequality in this chain an equality. Equality in the
first inequality says
$\mathbf{x}^\star$ minimizes
$\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$,
the right half of :eqref:`eq_mdl-opt-saddle-point`. Equality in the second
forces $\sum_i \lambda_i^\star g_i(\mathbf{x}^\star) = 0$ (complementary
slackness, re-derived), whence for any $\boldsymbol{\lambda} \succeq 0$ and
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

i.e. $p^\star \le d^\star$; weak duality supplies the reverse inequality.
$\blacksquare$

The saddle-point formulation also describes a family of *minimax* training
objectives. Generative adversarial networks
(:numref:`sec_basic_gan`) and adversarial
training are $\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}$ problems in
the same min--max form
:cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014,Madry.Makelov.Schmidt.ea.2018`,
and equality between the $\min\max$ and $\max\min$ values holds when an
appropriate saddle point exists. A duality gap separates these values. This
formulation also underlies
**primal--dual methods**, which descend in $\mathbf{x}$ and ascend in
$\boldsymbol{\lambda}$ simultaneously and seek the saddle rather than a
minimum. Convexity alone does not make them converge; they reach the saddle
under extra conditions, for example a strongly convex objective and a small
enough step size.

### Sensitivity and Shadow Prices

Optimal multipliers $\boldsymbol{\lambda}^\star$ quantify sensitivity to
constraint perturbations, an interpretation used in economics, operations
research, and machine learning. Perturb the constraints:
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
$p^\star(\mathbf{u})$ and touches it at $\mathbf{u} = \mathbf{0}$, so it is
the tangent there, and matching gradients gives
$\nabla p^\star(\mathbf{0}) = -\boldsymbol{\lambda}^\star$. $\blacksquare$

The multiplier $\lambda_i^\star$ is the **shadow price of constraint $i$**:
relaxing the constraint by one unit improves the optimum by approximately
$\lambda_i^\star$. Complementary slackness sets the multiplier of a slack
constraint to zero; only binding constraints can have a positive multiplier. In the
ball projection, $\lambda^\star = \|\mathbf{x}_0\|/r - 1$ prices the radius; in
the hyperplane example, $-\nu^\star$ priced the offset $b$; and in the
water-filling problem below, the equality multiplier equals the
marginal value of transmit power, as verified below by finite differences.

### Weight Decay and Norm Constraints
:label:`subsec_mdl-weight-decay-duality`

Weight decay (:numref:`sec_weight_decay`) adds a penalty to the training loss,

$$
\min_{\mathbf{w}}\; L(\mathbf{w}) + \lambda \|\mathbf{w}\|^2 ,
$$

and the KKT formulation identifies that penalty as the Lagrangian of a norm
**constraint**,

$$
\min_{\mathbf{w}}\; L(\mathbf{w}) \quad \textrm{subject to} \quad \|\mathbf{w}\|^2 \le r^2 .
$$

Compare stationarity conditions. The penalized problem asks
$\nabla L(\mathbf{w}) + 2\lambda \mathbf{w} = \mathbf{0}$; the constrained
problem's KKT stationarity, with multiplier $\lambda \ge 0$ on
$g(\mathbf{w}) = \|\mathbf{w}\|^2 - r^2$, asks
$\nabla L(\mathbf{w}) + 2\lambda \mathbf{w} = \mathbf{0}$: the *same
equation*, with the weight-decay coefficient playing the multiplier's role.
For **convex** $L$ the correspondence is exact in both directions: the
minimizer $\mathbf{w}_\lambda$ of the penalized problem is the solution of the
constrained problem with budget $r = \|\mathbf{w}_\lambda\|$ (it is feasible,
and KKT sufficiency certifies it), and conversely any active budget $r > 0$
with attained minimizer has a
multiplier $\lambda^\star(r)$ whose penalized problem returns the same point;
this direction uses KKT necessity, which applies because Slater's condition
holds for every $r > 0$ ($\mathbf{w} = \mathbf{0}$ is strictly feasible).
Sweeping $\lambda$ and sweeping $r$ trace the *same* regularization path, just
parameterized differently. The shadow-price proposition gives
$\lambda^\star = -\partial p^\star / \partial (r^2)$, the marginal reduction in
loss from an additional unit of squared-norm budget. When the weight
budget is slack (the unregularized minimizer already satisfies
$\|\mathbf{w}\| \le r$), complementary slackness prices it at
$\lambda^\star = 0$.

Two caveats. First, for the *non-convex* losses of deep networks the
correspondence is local and heuristic, not exact: a stationary point of the
penalized problem is a KKT point of the matching constrained problem, but
without convexity KKT points need not be global minima and the two problems'
solution sets need not coincide; the same gap that separates $p^\star$ from
$d^\star$ in the example below can separate the two formulations. Second, the
same number has a third, statistical reading:
:numref:`sec_mdl-maximum_likelihood` derives the penalty
$\lambda\|\mathbf{w}\|^2$ as the log of a Gaussian prior (MAP estimation).
Thus $\lambda$ can be interpreted as a Lagrange multiplier, a regularization
coefficient, or a parameter of a prior distribution. The cell below
verifies the equivalence numerically on ridge regression
($L(\mathbf{w}) = \|A\mathbf{w} - \mathbf{b}\|^2$, convex, so the theorem
applies in full): for each $\lambda$ it solves the penalized problem in closed
form, uses the resulting norm as the *constrained* problem's budget $r$,
solves that by projected gradient descent onto the $r$-ball, and recovers the
multiplier from the KKT stationarity residual:

```{.python .input #constrained-weight-decay}
rng = np.random.default_rng(3)
A, b = rng.normal(size=(30, 5)), rng.normal(size=30)
grad = lambda w: 2 * A.T @ (A @ w - b)           # gradient of L = |Aw - b|^2
step = 1.0 / (2 * np.linalg.eigvalsh(A.T @ A).max())
print(' lambda    r = |w_pen|   |w_con - w_pen|   recovered multiplier')
for lam in [0.1, 1.0, 10.0]:
    w_pen = np.linalg.solve(A.T @ A + lam * np.eye(5), A.T @ b)  # penalty form
    r = np.linalg.norm(w_pen)                    # its norm becomes the budget
    w = np.zeros(5)                              # constraint form: PGD, radius r
    for _ in range(5000):
        w -= step * grad(w)
        w *= min(1.0, r / np.linalg.norm(w))     # project onto the r-ball
    lam_rec = -(w @ grad(w)) / (2 * r * r)       # KKT: grad L + 2*lam*w = 0
    print(f'{lam:7.2f}  {r:11.4f}  {np.linalg.norm(w - w_pen):15.2e}  '
          f'{lam_rec:17.4f}')
```

Three different decay strengths, and in every row the constrained solution
matches the penalized one to $10^{-16}$ while the multiplier recovered from
the constrained problem's own KKT residual reproduces $\lambda$ to four
decimals. Penalty and constraint give two parameterizations of the same convex solution
family, and the multiplier connects them.

## Worked Duals: SVM, Water-Filling, and an Explicit Gap
:label:`subsec_mdl-worked-duals`

### The Support Vector Machine Dual
:label:`subsec_mdl-svm-dual`

The support-vector machine :cite:`Cortes.Vapnik.1995` is a standard example in
which the dual exposes useful structure. Given linearly separable data
$(\mathbf{x}_i, y_i)$ with labels $y_i \in \{\pm 1\}$, the *maximum-margin*
separating hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ solves

$$
\min_{\mathbf{w}, b}\; \tfrac12 \|\mathbf{w}\|^2
\quad \textrm{subject to} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \;\; i = 1, \ldots, n,
$$
:eqlabel:`eq_mdl-opt-svm-primal`

since, by the signed-distance formula of
:numref:`sec_mdl-geometry-linear-algebraic-ops` (a point $\mathbf{x}$ lies at
distance $|\mathbf{w}^\top\mathbf{x} + b| / \|\mathbf{w}\|$ from the
hyperplane), the two margin hyperplanes
$\mathbf{w}^\top\mathbf{x} + b = \pm 1$ sit at distance $1/\|\mathbf{w}\|$ on
either side of the boundary, $2/\|\mathbf{w}\|$ apart:
minimizing $\|\mathbf{w}\|$ *is* maximizing the margin. This is a convex QP
with one inequality per data point; Slater holds for strictly separable data
(scale any separating $(\mathbf{w}, b)$ up until all margins exceed $1$), so
strong duality is guaranteed. Form the Lagrangian with multipliers
$\alpha_i \ge 0$,

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
= \tfrac12\|\mathbf{w}\|^2 + \sum_i \alpha_i \left(1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\right),
$$

and compute the dual function
$g(\boldsymbol{\alpha}) = \inf_{\mathbf{w}, b} \mathcal{L}$. The Lagrangian is
*affine in $b$*, with coefficient $-\sum_i \alpha_i y_i$: unless that
coefficient vanishes, sending $b$ to $\pm\infty$ drives $\mathcal{L}$ to
$-\infty$, so $g(\boldsymbol{\alpha}) = -\infty$ whenever
$\sum_i \alpha_i y_i \neq 0$. This is the effective-domain mechanism promised
at :eqref:`eq_mdl-opt-dual-function`: multipliers with
$\sum_i \alpha_i y_i \neq 0$ lie outside the dual function's effective
domain, so the
vanishing condition reappears as the dual's *equality constraint*. On that
domain the $b$-term drops out of $\mathcal{L}$ entirely, leaving a strictly
convex quadratic in $\mathbf{w}$ whose infimum is attained at its stationary
point,

$$
\mathbf{w} = \sum_i \alpha_i y_i\, \mathbf{x}_i,
\qquad\textrm{on the domain}\qquad
\sum_i \alpha_i y_i = 0.
$$

The weight vector is a combination *of the data*, with coefficients the
multipliers. Substituting back yields the **SVM dual**:

$$
\max_{\boldsymbol{\alpha} \succeq 0,\ \sum_i \alpha_i y_i = 0}\;\;
\sum_i \alpha_i \;-\; \tfrac12 \sum_{i,j} \alpha_i \alpha_j\, y_i y_j\, \mathbf{x}_i^\top \mathbf{x}_j.
$$
:eqlabel:`eq_mdl-opt-svm-dual`

Two consequences hold before any algorithm is chosen. *Complementary slackness*,
$\alpha_i (1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)) = 0$, says
$\alpha_i > 0$ only for points *exactly on the margin*: the
**support vectors**, the active constraints of the active-set picture in
:numref:`fig_mdl-opt-kkt-active-set`. All other points have $\alpha_i = 0$ and
do not contribute to the recovered weight vector. And the data enter
:eqref:`eq_mdl-opt-svm-dual` only through inner products
$\mathbf{x}_i^\top\mathbf{x}_j$: replace them by a kernel evaluation
$k(\mathbf{x}_i, \mathbf{x}_j)$ and the resulting dual defines a nonlinear
classifier, an option the primal formulation does not expose.

We can solve the dual using the preceding conditions.
One simplification first: the equality constraint $\sum_i \alpha_i y_i = 0$
entered only because the unpenalized offset $b$ made the Lagrangian affine in
one direction. If we instead fold the
offset into the weights (append a constant feature,
$\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ and
$\tilde{\mathbf{w}} = (\mathbf{w}, b)$, so the primal becomes
$\min \tfrac12\|\tilde{\mathbf{w}}\|^2$ subject to
$y_i \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i \ge 1$), then the Lagrangian
is strictly convex in every primal variable, its infimum is finite for every
$\boldsymbol{\alpha}$, the equality constraint never appears, and the
dual feasible set is the plain nonnegative orthant
$\boldsymbol{\alpha} \succeq 0$, on which projection is a coordinate-wise clip.
(One caveat in labeling: this variant also regularizes the offset, so
its maximum-margin boundary can differ slightly from
:eqref:`eq_mdl-opt-svm-primal`'s; the duality structure is identical.) Writing
$Q_{ij} = y_i y_j \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j$, the dual is

$$
\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top Q\, \boldsymbol{\alpha},
$$
:eqlabel:`eq_mdl-opt-svm-dual-folded`

a concave quadratic over the orthant, exactly the setting of
:eqref:`eq_mdl-opt-pgd`. Projected gradient *ascent* iterates
$\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - Q\boldsymbol{\alpha}))$
with $\eta = 1/\lambda_{\max}(Q)$, the dual's own descent-lemma step size. The
cell below runs it on a small 2-D toy problem and then checks this
section's claims: the recovered $(\mathbf{w}^\star, b^\star)$ separates the
data, complementary slackness holds to machine precision, and the dual value
meets the primal value: strong duality, observed.

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

The printout illustrates these properties. Four of the eight multipliers are
strictly positive, and *exactly those four points* have margin $1.0000$: the
support vectors, whose margin equality follows from complementary slackness; the other
four points sit at margins $1.29$--$2.43$ with $\alpha_i = 0$. The solution is
exact enough to recognize:
$\mathbf{w}^\star = (\tfrac47, \tfrac47)$, $b^\star = -\tfrac37$, with
$p^\star = d^\star = \tfrac{41}{98} \approx 0.418367$ and a primal--dual gap at
$10^{-16}$ after five thousand fixed-step projected-ascent iterations. The
primal--dual gap therefore certifies optimality to machine precision. The
printout also verifies an identity derived in Exercise :exref:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`a:
$\sum_i \alpha_i^\star = \|\tilde{\mathbf{w}}^\star\|^2$ at the optimum, which
is why the dual and primal values coincide line for line.

### Water-Filling

The second dual has a closed-form solution with a standard geometric
interpretation. A transmitter splits a
power budget $P$ across $n$ independent channels; channel $i$ has noise level
$n_i > 0$, and the achievable communication rate is
$\sum_i \log(1 + p_i / n_i)$ :cite:`Cover.Thomas.1999`. The allocation problem,

$$
\max_{\mathbf{p}}\; \sum_{i=1}^n \log\left(1 + \frac{p_i}{n_i}\right)
\quad \textrm{subject to} \quad \mathbf{p} \succeq 0, \;\; \sum_i p_i = P,
$$
:eqlabel:`eq_mdl-opt-waterfilling`

is convex (concave objective, affine constraints) with Slater trivially
satisfied, so the KKT conditions characterize the global optimum. With multiplier $\mu$ for the
budget and $\lambda_i \ge 0$ for $p_i \ge 0$, stationarity reads

$$
\frac{1}{n_i + p_i} = \mu - \lambda_i.
$$

On channels with $p_i > 0$, complementary slackness sets $\lambda_i$ to zero and
forces $n_i + p_i = 1/\mu$: *the same constant for every active channel*.
On dry channels, $p_i = 0$ requires $\lambda_i = \mu - 1/n_i \ge 0$, i.e.
$n_i \ge 1/\mu$. Writing $w = 1/\mu$ for that constant:

$$
p_i^\star = \max(w - n_i,\, 0),
\qquad \textrm{with } w \textrm{ set by } \sum_i \max(w - n_i,\, 0) = P.
$$

This is the **water-filling** solution:
represent each channel as a basin with floor height $n_i$ and allocate $P$
units of water. As :numref:`fig_mdl-opt-water-filling` shows, the water
settles at a common level $w$, filling the deep (quiet) channels most, and
never reaching basins whose floor is above the waterline. The level is the
unique root of a continuous, nondecreasing, piecewise-linear function of $w$,
so *bisection* finds it. The threshold structure is the same KKT case split
as the simplex projection's $\tau$, with a different objective.

![Water-filling, drawn with the same noise floors and budget as the cell below. Complementary slackness in a picture: pouring $P = 3$ units of power into basins with floors at the noise levels $n_i$ fills the three quiet channels to the common level $w = 1/\mu \approx 1.43$, allocating $p_i^\star = w - n_i$ to each, while the two channels whose floors exceed the waterline receive zero allocation, $p_i^\star = 0$.](../img/mdl-opt-water-filling.svg)
:label:`fig_mdl-opt-water-filling`

The
shadow-price proposition gives the multiplier its engineering meaning:
$\mu^\star = 1/w$ is the derivative of the optimal rate with respect to the
power budget, which the cell checks by re-solving at $P \pm 10^{-4}$.

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
stay dry exactly as dual feasibility demands, and the multiplier
$\mu = 1/w = 0.6977$ matches the finite-difference sensitivity of the optimal
rate to the budget through six digits. Perturbing the constraint verifies this
shadow-price interpretation.

### An Explicit Duality Gap
:label:`subsec_mdl-explicit-duality-gap`

Weak duality always holds, whereas strong duality need not. The following
one-dimensional problem gives an explicit counterexample: on the interval $x \in [0, 1]$,

$$
\min_x\; f_0(x) = -x^2
\quad \textrm{subject to} \quad f_1(x) = x - \tfrac12 \le 0.
$$

The objective is *concave* (this is a non-convex problem) and the
feasible region is $[0, \tfrac12]$, so the primal optimum is at the boundary:
$p^\star = f_0(\tfrac12) = -\tfrac14$. Only the constraint $f_1(x) \le 0$ is
dualized: the domain $[0, 1]$ stays a hard constraint, so the infimum in
:eqref:`eq_mdl-opt-dual-function` runs over $[0, 1]$ instead of all of
$\mathbb{R}$, and the proof of weak duality goes through unchanged. The dual
function is computable by hand:
$\mathcal{L}(x, \lambda) = -x^2 + \lambda(x - \tfrac12)$ is concave in $x$, so
its minimum over $[0, 1]$ is at an endpoint, giving
$g(\lambda) = \min(-\lambda/2,\; \lambda/2 - 1)$: piecewise linear, concave
(as it must be), and maximized at $\lambda = 1$ with
$d^\star = -\tfrac12 < -\tfrac14 = p^\star$: a duality gap of exactly
$\tfrac14$. The cell verifies all of it numerically and plots the maximum of the dual function
strictly below $p^\star$.

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

This example also shows that a strictly feasible point can exist
($x = 0$ has $f_1 = -\tfrac12 < 0$), yet the gap is real: Slater's
condition certifies strong duality *only for convex problems*, and $f_0 = -x^2$
is not convex. In the supporting-line picture of
:numref:`fig_mdl-opt-primal-dual-gap`, the nonconvex curve
$\{(f_1(x), f_0(x))\} = \{(u, -(u + \tfrac12)^2)\}$ admits no supporting line
through its point at $u = 0$, and the best available line
(slope $-1$, our $\lambda^\star = 1$) certifies only $-\tfrac12$. For
non-convex training problems, duals still give valid *lower bounds* (and are the
engine of
verification and relaxation methods), but the bound need not be tight.

### Convexity and Duality Across Problem Classes

The examples above were all quadratic programs, but they sit inside a standard
hierarchy of convex problem classes :cite:`Boyd.Vandenberghe.2004`; each class
corresponds to standard solver families:

| Class | Template | Machine-learning examples |
|---|---|---|
| **LP** (linear program) | linear $f$, constraints $A\mathbf{x} \preceq \mathbf{b}$ | $\ell_1$ / $\ell_\infty$ reformulations, optimal transport |
| **QP** (quadratic program) | convex quadratic $f$, affine constraints | ridge, lasso, SVM dual, simplex projection |
| **SOCP** (second-order cone program) | cones $\|A\mathbf{x} + \mathbf{b}\| \le \mathbf{c}^\top\mathbf{x} + d$ | max-norm weight constraints, robust losses, trust-region step with a convex model |
| **SDP** (semidefinite program) | matrix variable $X \succeq 0$, the semidefinite sense of $\succeq$ (:numref:`sec_mdl-convexity`) | spectral-norm bounds, nuclear-norm relaxations |

Each class contains the previous ($\mathrm{LP} \subseteq \mathrm{QP} \subseteq
\mathrm{SOCP} \subseteq \mathrm{SDP}$), and all are solvable to high accuracy
in polynomial time by interior-point methods, which replace the constraints by
a barrier and follow the smoothed problems' solutions to the boundary
:cite:`Boyd.Vandenberghe.2004,Nesterov.2018,Nocedal.Wright.2006`. Training a
deep network lies
*outside* the hierarchy, since the composition of layers destroys convexity
(:numref:`sec_mdl-convexity`); but many of its
*sub-problems* are convex, where multipliers, KKT, and duality apply directly:
projections, clipped updates, last-layer fits, and, when the local quadratic
model is convex, the trust-region step, which minimizes that model within a
ball around the current iterate :cite:`Nocedal.Wright.2006`.

## Summary

* The **dual function** $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}$
  is *always concave*, and **weak duality** $d^\star \le p^\star$ *always*
  holds: dual points are certificates. **Slater's condition** (convexity plus a
  strictly feasible point) closes the gap; without convexity a gap can survive
  even with strictly feasible points.
* Strong duality is the same statement as a **saddle point** of the
  Lagrangian ($\inf \sup = \sup \inf$), which provides the basis for
  primal--dual methods and minimax
  objectives such as GANs and adversarial training. **Weight decay** is the
  central application: the penalty
  $\lambda\|\mathbf{w}\|^2$ and the constraint $\|\mathbf{w}\|^2 \le r^2$
  share one KKT stationarity equation. Under convexity and suitable regularity,
  an active constraint and its optimal multiplier connect corresponding
  solutions; inactive constraints, zero multipliers, and nonunique solutions
  prevent a one-to-one correspondence. For deep networks the analogy is local
  and heuristic.
* Multipliers are **shadow prices**: $\lambda_i^\star = -\partial p^\star/\partial u_i$
  measures what relaxing constraint $i$ is worth. It is $1/w$ in
  water-filling, and it is why slack constraints have zero shadow price.
* The SVM dual (support vectors = active constraints, kernels via inner
  products) and water-filling (pour to a common level, bisection on the level)
  show the full pipeline; LP/QP/SOCP/SDP is the solver map for the convex
  sub-problems of deep learning.

## Exercises

1. **Conjugate pairs.** Compute each conjugate
   from :eqref:`eq_mdl-opt-conjugate`.
    1. $f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$ with $\lambda > 0$.
    1. $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top A \mathbf{x}$ with
       $A \succ 0$. What does the Fenchel--Young inequality
       :eqref:`eq_mdl-opt-fenchel-young` state for this pair?
    1. The softplus $f(t) = \log(1 + e^t)$ on $\mathbb{R}$. The binary
       cross-entropy of a logit $t$ against a soft label $p \in [0, 1]$ is
       $\log(1 + e^t) - pt$. Use Fenchel--Young to show that it is at least
       the binary entropy $H(p) = -p\log p - (1-p)\log(1-p)$, with equality
       exactly when $\sigma(t) = p$. Write the difference as a KL divergence.
    1. The double well $g(x) = (x^2 - 1)^2$ of
       :numref:`subsec_mdl-recognizing-convexity` is not convex. Find its
       convex envelope $g^{**}$ and the set on which $g^{**} < g$.

   *Adapted from Boyd and Vandenberghe,
   [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/),
   Examples 3.22 and 3.26.*
   :exlabel:`ex_mdl-convexity-conjugate-pairs`

1. **Maximum entropy and temperature.** The entropy of a probability vector
   $\mathbf{p} \in \mathbb{R}^n$ is $H(\mathbf{p}) = -\sum_i p_i \log p_i$.
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
       $dH^\star / d\bar s = -\beta$, so that the multiplier prices the score
       constraint as the multiplier of :numref:`subsec_mdl-closest-point`
       prices the offset $b$. For which targets $\bar s$ is $\beta$
       negative? What happens to the softmax temperature $1/\beta$ as
       $\bar s$ approaches $\max_i s_i$?

1. **A dual certificate for $\ell_1$ minimization.** Basis pursuit,

   $$\min_{\mathbf{x}}\; \|\mathbf{x}\|_1 \quad \textrm{subject to} \quad A\mathbf{x} = \mathbf{b},$$

   with $A \in \mathbb{R}^{m \times n}$ and $m < n$, selects a sparse solution
   of an underdetermined linear system. Its constraints are affine, so the
   refined Slater condition gives $d^\star = p^\star$ whenever the system has
   a solution.
    1. Adapt the conjugate form :eqref:`eq_mdl-opt-dual-conjugate` of the
       dual function to equality constraints. Use the conjugate of
       $\|\cdot\|_1$ from Exercise :exref:`ex_mdl-convexity-conjugate-pairs`a
       of :numref:`sec_mdl-convexity` to show that, after replacing
       $\boldsymbol{\nu}$ by $-\boldsymbol{\nu}$, the dual problem is
       $\max_{\boldsymbol{\nu}} \mathbf{b}^\top \boldsymbol{\nu}$ subject to
       $\|A^\top \boldsymbol{\nu}\|_\infty \le 1$.
    1. Solve both problems by hand for $A = (1\;\; 2)$ and $b = 2$, and
       check $d^\star = p^\star$.
    1. Let $\mathbf{x}^\star$ and $\boldsymbol{\nu}^\star$ be optimal. Show
       that $x_i^\star = 0$ wherever
       $|(A^\top \boldsymbol{\nu}^\star)_i| < 1$, and that
       $(A^\top \boldsymbol{\nu}^\star)_i = \mathrm{sign}(x_i^\star)$ wherever
       $x_i^\star \neq 0$. A candidate $\mathbf{x}$ with $A\mathbf{x} = \mathbf{b}$
       is given. Which conditions must a vector $\boldsymbol{\nu}$ satisfy to
       certify that $\mathbf{x}$ is optimal without solving the problem?

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
       the fraction of points with $\alpha_i^\star > 0$.
   :exlabel:`ex_mdl-constrained-optimization-duality-offset-folded-svm-dual`

1. [code] **Duality gaps and formulations.** The cell that computes `p_star`
   and `d_star` in :numref:`subsec_mdl-explicit-duality-gap` minimizes the
   Lagrangian over the domain $[0, 1]$ and dualizes only the constraint
   $x \le \tfrac12$.
    1. Replace the objective by $f_0(x) = (x - 1)^2$. Compute $p^\star$, the
       dual function, $d^\star$, and $\lambda^\star$ by hand, and confirm them
       with the modified cell. Why is the gap now zero?
    1. Keep $f_0(x) = -x^2$, but write the constraint as $x^2 \le \tfrac14$,
       which defines the same feasible set on $[0, 1]$. Compute the new dual
       function and $d^\star$ by hand and with the cell.
    1. Keep $f_0(x) = -x^2$ and the constraint $x \le \tfrac12$, but treat
       $0 \le x$ and $x \le 1$ as constraints with their own multipliers and
       minimize the Lagrangian over all of $\mathbb{R}$. What is $d^\star$
       now? Rank the dual bounds of the three formulations of the same
       problem (the one in the body and those of sub-items b and c). Is a
       duality gap a property of the problem or of its formulation?

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
       $\mathcal{L}(\mathbf{x}, \nu) = \mathbf{x}^\top \mathbf{x} + \nu(\mathbf{a}^\top \mathbf{x} - b)$
       of the closest-point problem of :numref:`subsec_mdl-closest-point`, with
       $\mathbf{a} = (1, 2)$, $b = 1$, and the start $\mathbf{x} = \mathbf{0}$,
       $\nu = 0$. Find the step sizes for which the iterates converge to
       $(\mathbf{x}^\star, \nu^\star)$. Derive them from the eigenvalues of
       the update acting on the pair
       $(\mathbf{a}^\top \mathbf{x} / \|\mathbf{a}\|, \nu)$ and on the
       components of $\mathbf{x}$ orthogonal to $\mathbf{a}$.
    1. Add $\tfrac{\epsilon}{2} x^2$ with $0 < \epsilon \le 2$ to the
       objective of the bilinear game. For which step sizes does simultaneous
       descent--ascent converge now? Which term of the Lagrangian of sub-item
       b plays the role of $\epsilon$?

1. [code] **Penalty versus constraint without convexity.** ● For a convex
   loss, :numref:`subsec_mdl-weight-decay-duality` shows that weight decay and
   a norm constraint trace the same solutions. Take instead the nonconvex
   one-dimensional loss $L(w) = (w - 1)^2 (w - 3)^2 - w$, whose local minima
   lie at $w \approx 1.16$ and $w \approx 3.11$.
    1. On a fine grid of $w$, compute the penalized minimizers
       $w_\lambda = \mathrm{argmin}_w\, L(w) + \lambda w^2$ for $\lambda$ on
       a logarithmic grid, and the constrained optimum
       $p^\star(r) = \min \{L(w) : w^2 \le r^2\}$ for $r \in [0, 3.2]$. Plot
       the points $(w_\lambda^2, L(w_\lambda))$ on top of the curve
       $(r^2, p^\star(r))$. Which budgets $r$ are never the norm $|w_\lambda|$
       of a penalized minimizer?
    1. Show that $w_\lambda$ minimizes $L(w) + \lambda w^2$ if and only if
       the line of slope $-\lambda$ through $(w_\lambda^2, L(w_\lambda))$ lies
       below every point of the set $\{(w^2, L(w)) : w \in \mathbb{R}\}$. Use
       this and the supporting-line picture of
       :numref:`fig_mdl-opt-primal-dual-gap` to explain the missing budgets.
    1. Pick a budget $r$ in the missing range at which the constraint is
       active. Compute the KKT multiplier $\lambda$ of the constrained
       solution, and find a point where the penalized objective with this
       $\lambda$ is lower. Compute the duality gap $p^\star(r) - d^\star(r)$
       of the constrained problem.

1. [extended] **Entropic optimal transport.** ●● Given probability vectors
   $\mathbf{p} \in \mathbb{R}^n$ and $\mathbf{q} \in \mathbb{R}^m$ and a cost
   matrix $C \in \mathbb{R}^{n \times m}$, optimal transport finds the
   cheapest plan $\Gamma$ that moves the mass $\mathbf{p}$ to the mass
   $\mathbf{q}$,

   $$\mathrm{OT} = \min_{\Gamma \ge 0}\; \sum_{i,j} C_{ij} \Gamma_{ij} \quad \textrm{subject to} \quad \Gamma \mathbf{1} = \mathbf{p}, \;\; \Gamma^\top \mathbf{1} = \mathbf{q},$$

   a linear program with $nm$ variables. Entropic regularization
   :cite:`Cuturi.2013` adds $\varepsilon \sum_{i,j} \Gamma_{ij} (\log \Gamma_{ij} - 1)$
   to the objective, with $\varepsilon > 0$; let $\Gamma_\varepsilon$ be the
   regularized optimum. :numref:`sec_mdl-divergences-distances` uses the
   resulting Sinkhorn iterations to compare distributions
   :cite:`Peyre.Cuturi.2019`.
    1. Form the Lagrangian with the terms
       $\boldsymbol{\phi}^\top (\mathbf{p} - \Gamma \mathbf{1}) + \boldsymbol{\psi}^\top (\mathbf{q} - \Gamma^\top \mathbf{1})$
       for the marginal constraints. Show that
       $\Gamma_\varepsilon = \mathrm{diag}(\mathbf{u})\, K\, \mathrm{diag}(\mathbf{v})$
       with $K_{ij} = e^{-C_{ij} / \varepsilon}$, $u_i = e^{\phi_i / \varepsilon}$,
       and $v_j = e^{\psi_j / \varepsilon}$. Why can the sign constraints
       $\Gamma_{ij} \ge 0$ be left out of the Lagrangian?
    1. Show that the dual function is

       $$g(\boldsymbol{\phi}, \boldsymbol{\psi}) = \mathbf{p}^\top \boldsymbol{\phi} + \mathbf{q}^\top \boldsymbol{\psi} - \varepsilon \sum_{i,j} e^{(\phi_i + \psi_j - C_{ij}) / \varepsilon},$$

       and that maximizing it exactly over $\boldsymbol{\phi}$ with
       $\boldsymbol{\psi}$ fixed gives the Sinkhorn update
       $\mathbf{u} = \mathbf{p} / (K \mathbf{v})$, elementwise. Conclude that
       Sinkhorn's algorithm is block-coordinate ascent on a concave function
       and that its dual value never decreases. Why is every dual value a
       lower bound on $\mathrm{OT}$?
    1. Let $H(\Gamma) = -\sum_{i,j} \Gamma_{ij} \log \Gamma_{ij}$, and grant
       that every feasible plan satisfies
       $\max(H(\mathbf{p}), H(\mathbf{q})) \le H(\Gamma) \le H(\mathbf{p}) + H(\mathbf{q})$.
       Show that
       $0 \le \sum_{i,j} C_{ij} (\Gamma_\varepsilon)_{ij} - \mathrm{OT} \le \varepsilon \min(H(\mathbf{p}), H(\mathbf{q}))$.
    1. Implement Sinkhorn in the log domain, updating $\boldsymbol{\phi}$ and
       $\boldsymbol{\psi}$ with a numerically stable log-sum-exp, and stop
       when the row sums of the plan are off by less than $10^{-6}$ in
       $\ell_1$ norm. Take two histograms on $50$ equally spaced points
       $x_i$ of $[0, 1]$, for example a Gaussian bump and a mixture of two
       bumps, and the cost $C_{ij} = (x_i - x_j)^2$. Compute $\mathrm{OT}$
       exactly: in one dimension an optimal plan is monotone, filling the
       destinations from left to right with the sources taken from left to
       right. For $\varepsilon \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$,
       tabulate the number of iterations, the transport cost of
       $\Gamma_\varepsilon$, its excess over $\mathrm{OT}$ next to the bound
       of sub-item c, and the final dual value. Then run the multiplicative
       updates $\mathbf{v} = \mathbf{q} / (K^\top \mathbf{u})$,
       $\mathbf{u} = \mathbf{p} / (K \mathbf{v})$. At which $\varepsilon$ do
       they break down, and why?

## Discussions

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.6]{.kicker}

Lagrangian duality<br>**convex conjugate · dual function · strong duality · prices · worked duals**.
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Duality]{.dtitle}

[lower bounds and sensitivity]{.dsub}
:::
:::

::: {.slide title="The Dual Function Gives Lower Bounds"}
[Duality]{.kicker}

::: {.cols .vc}
::: {.col}
Minimizing the Lagrangian over $\mathbf{x}$ defines a function of the
multipliers:

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

Maximizing the concave dual function is a convex optimization problem, and
under strong duality its optimum equals the primal optimum.
:::

::: {.col}
::: {.d2l-note .rule}
**Shadow price:**

$$\lambda_i^\star = -\frac{\partial p^\star}{\partial u_i}.$$

Relax constraint $i$ by a unit and the optimum improves by $\lambda_i^\star$.
:::
:::
:::

Slack constraints have zero shadow price; binding constraints can have a
nonzero price.
:::

::: {.slide title="Strong duality is a saddle point"}
[Duality]{.kicker}

The primal value is $\inf_{\mathbf{x}} \sup_{\boldsymbol{\lambda} \succeq 0} \mathcal{L}$:
a violated constraint makes the supremum $+\infty$. The dual reverses the
optimization order. Weak duality gives $\sup\inf \le \inf\sup$, and strong
duality makes the two values equal:

$$\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda})
\;\le\; \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star)
\;\le\; \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star).$$

. . .

::: {.d2l-note}
This is the formulation used in **minimax training**: GANs and adversarial training
are $\min_{\boldsymbol{\theta}}\max_{\boldsymbol{\phi}}$ problems, and
a saddle point makes the min--max and max--min values equal. Primal--dual
methods descend in $\mathbf{x}$ and ascend in $\boldsymbol{\lambda}$; their
convergence requires additional assumptions.
:::
:::

::: {.slide title="Penalty and Norm-Constrained Forms"}
[Duality at work]{.kicker}

The penalty and constraint forms of $\ell_2$ regularization are linked by the
Lagrangian:

$$\underbrace{\min_{\mathbf{w}}\, L(\mathbf{w}) + \lambda\|\mathbf{w}\|^2}_{\text{penalty}}
\;\Longleftrightarrow\;
\underbrace{\min_{\mathbf{w}}\, L(\mathbf{w}) \;\text{s.t.}\; \|\mathbf{w}\|^2 \le r^2}_{\text{constraint}}$$

Same stationarity equation, $\lambda$ playing the multiplier:
$\lambda = -\partial p^\star/\partial(r^2)$ prices the weight budget.
Verified on ridge regression, where the theorem is exact:

@!constrained-weight-decay

::: {.d2l-note}
The same coefficient appears as **weight decay**, the multiplier of a norm
constraint, and the precision of a Gaussian prior.
:::
:::

::: {.slide title="Worked: the SVM dual"}
[Duality at work]{.kicker}

Eliminate $(\mathbf{w}, b)$ from the max-margin Lagrangian and the dual is
a concave QP over the orthant that can be solved by projected gradient
*ascent*:

$$\max_{\boldsymbol{\alpha} \succeq 0}\;\; \mathbf{1}^\top \boldsymbol{\alpha} - \tfrac12 \boldsymbol{\alpha}^\top Q\, \boldsymbol{\alpha},
\qquad
\boldsymbol{\alpha} \leftarrow \max(\mathbf{0},\, \boldsymbol{\alpha} + \eta\,(\mathbf{1} - Q\boldsymbol{\alpha})).$$

@!constrained-svm-dual

::: {.d2l-note}
The four nonzero $\alpha_i$ are exactly the points at **margin 1**, the
support vectors. The primal and dual values agree to $10^{-16}$.
:::
:::

::: {.slide title="Worked: water-filling"}
[Duality at work]{.kicker}

::: {.cols .vc}
::: {.col}
Allocate power $P$ across noisy channels. KKT gives a **common level** $w$
for every channel receiving power; complementary slackness assigns zero power
to sufficiently noisy channels:

$$p_i^\star = \max(w - n_i,\, 0),
\qquad \textstyle\sum_i \max(w - n_i, 0) = P.$$

The multiplier $\mu = 1/w$ equals the marginal value of power;
the run below confirms it by finite differences, $0.697674$ both ways.
:::

::: {.col .fig .big}
@fig:mdl-opt-water-filling
:::
:::
:::

::: {.slide title="An Explicit Duality Gap"}
[Duality at work]{.kicker}

::: {.cols .vc}
::: {.col}
Weak duality always holds; strong duality can fail. Minimize the *concave*
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
