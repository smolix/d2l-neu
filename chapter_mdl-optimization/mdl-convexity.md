# Convex Sets and Convex Functions
:label:`sec_mdl-convexity`

Convexity is the line between "gradient descent provably finds the answer" and "we
hope." For a convex problem, *every local minimum is global*, there are no saddle
points or spurious basins to get stuck in, and the rates of
:numref:`sec_mdl-gradient-based-optimization` come with global guarantees.
Convexity tells you which deep-learning sub-problems are *easy* --- the last-layer
softmax/logistic loss in the logits, projections onto constraint sets, the
$\ell_1$ and $\ell_2$ regularizers, the SVM dual --- and it tells you precisely
what you forfeit by stacking nonlinear layers. This section develops the three
equivalent lenses for recognizing a convex function, the inequality (Jensen's)
that powers half of information theory, the global guarantees that are the payoff
of the whole theory, a calculus for certifying convexity without computing a
single Hessian, and an honest reality check on why deep networks are non-convex
and what we keep anyway.

We lean on :numref:`sec_mdl-geometry-linear-algebraic-ops` (inner products,
hyperplanes, half-spaces), :numref:`sec_mdl-single_variable_calculus` and
:numref:`sec_mdl-multivariable_calculus` (derivatives, the Hessian, positive
semidefiniteness), and :numref:`sec_mdl-gradient-based-optimization` (the descent
lemma and the condition number $\kappa = L/\mu$). The standard reference is
:citet:`Boyd.Vandenberghe.2004`, chapters 2--3; :citet:`Nesterov.2018` is the
source for the convergence theory. All code in this section is plain NumPy ---
every demonstration is a handful of lines, and none of it needs a framework.

```{.python .input #convexity-imports}
import numpy as np
```

## Convex Sets
:label:`subsec_mdl-convex-sets`

### Segments That Stay Inside

The definition is visual: a set is convex when the segment between any two of its
points stays inside it. Formally, $C \subseteq \mathbb{R}^n$ is **convex** if

$$
\theta \mathbf{x} + (1 - \theta)\, \mathbf{y} \in C
\qquad \textrm{for all } \mathbf{x}, \mathbf{y} \in C \textrm{ and all } \theta \in [0, 1].
$$
:eqlabel:`eq_mdl-opt-convex-set`

As $\theta$ runs from $1$ to $0$, the point $\theta\mathbf{x} + (1-\theta)\mathbf{y}$
walks the straight segment from $\mathbf{x}$ to $\mathbf{y}$; convexity demands
the walk never leaves the set. :numref:`fig_mdl-opt-convex-vs-nonconvex-set`
contrasts a convex set, where every chord lies within, with a non-convex
crescent, where a chord between two of its points slips outside; it also shows
the two convex sets deep learning uses most, the probability simplex (a
hyperplane cut of the nonnegative orthant) and a half-space.

![Convex versus non-convex sets. Left: a convex set, where the segment between any two points stays inside. Middle: a non-convex crescent, where a chord between two of its points passes outside the set. Right: the probability simplex $\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$ and a half-space $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}\leq b\}$, both convex.](../img/mdl-opt-convex-vs-nonconvex-set.svg)
:label:`fig_mdl-opt-convex-vs-nonconvex-set`

A quick non-example calibrates the definition. The annulus
$\{\mathbf{x} : 1 \le \|\mathbf{x}\| \le 2\}$ contains $(\pm \tfrac32, 0)$, but
their midpoint is the origin, which the annulus excludes --- the chord tunnels
through the hole. Holes, dents, and disconnected pieces are exactly what
:eqref:`eq_mdl-opt-convex-set` forbids.

### The Catalog Deep Learning Uses

Four families of convex sets cover most of what this book touches, and each is
convex for a one-line reason.

* **Hyperplanes and half-spaces.** For the hyperplane
  $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} = b\}$ and the half-space
  $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le b\}$, the check is linearity:
  $\mathbf{a}^\top(\theta\mathbf{x} + (1-\theta)\mathbf{y}) = \theta\, \mathbf{a}^\top\mathbf{x} + (1-\theta)\, \mathbf{a}^\top\mathbf{y}$,
  and an average of two numbers equal to $b$ (or at most $b$) is again equal to
  $b$ (or at most $b$).
* **Norm balls.** For any norm, $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$ is
  convex by the triangle inequality plus homogeneity:
  $\|\theta\mathbf{x} + (1-\theta)\mathbf{y}\| \le \theta\|\mathbf{x}\| + (1-\theta)\|\mathbf{y}\| \le r$.
  This covers the $\ell_2$ balls of weight decay and gradient clipping and the
  $\ell_1$ balls of sparsity-inducing regularization alike.
* **The probability simplex.**
  $\Delta = \{\mathbf{p} : \mathbf{p} \succeq 0,\ \mathbf{1}^\top\mathbf{p} = 1\}$,
  where attention weights, class probabilities, and mixture weights live.
* **The positive semidefinite cone.**
  $\mathbb{S}^n_+ = \{A = A^\top : \mathbf{z}^\top A \mathbf{z} \ge 0 \textrm{ for all } \mathbf{z}\}$,
  home of covariance matrices and of every Hessian this section will certify.

The last two we have not yet justified --- deliberately, because the cleanest
proofs use the one structural fact every catalog needs.

### New Convex Sets from Old

**Proposition (intersections preserve convexity).** *Let $\{C_i\}_{i \in I}$ be
any family of convex sets --- finite or infinite. Then
$C = \bigcap_{i \in I} C_i$ is convex.*

**Proof.** Take $\mathbf{x}, \mathbf{y} \in C$ and $\theta \in [0, 1]$. For
every $i$, both points lie in the convex set $C_i$, so
$\theta\mathbf{x} + (1-\theta)\mathbf{y} \in C_i$. A point in every $C_i$ is a
point in their intersection. $\blacksquare$

Unions enjoy no such luck: $[0,1] \cup [2,3]$ is a union of two convex intervals
whose chord from $1$ to $2$ leaves the set. Convexity survives *and*, not *or*.

The proposition is a factory. The probability simplex is the intersection of
one hyperplane ($\mathbf{1}^\top\mathbf{p} = 1$) with $n$ half-spaces
($-p_i \le 0$), hence convex with no further work. The PSD cone is the
intersection of *infinitely many* half-spaces, one per test vector: for each
fixed $\mathbf{z}$, the condition $\mathbf{z}^\top A \mathbf{z} \ge 0$ is
*linear* in the entries of $A$, so
$\mathbb{S}^n_+ = \bigcap_{\mathbf{z}} \{A : \mathbf{z}^\top A \mathbf{z} \ge 0\}$
is convex --- a fact we will lean on when Hessians enter. More generally, every
**polyhedron** $\{\mathbf{x} : A\mathbf{x} \preceq \mathbf{b}\}$ --- the
feasible set of the constrained problems in
:numref:`sec_mdl-constrained-optimization-duality` --- is a finite intersection
of half-spaces.

Two more constructions round out the toolkit. Affine maps preserve convexity in
both directions --- images and preimages of convex sets under
$\mathbf{x} \mapsto A\mathbf{x} + \mathbf{b}$ are convex, because affine maps
send segments to segments. And any point cloud $S$ generates a smallest convex
superset, its **convex hull**: the intersection of every convex set containing
$S$ (convex by the proposition), or concretely the set of all weighted averages
$\sum_i \theta_i \mathbf{x}_i$ with $\theta_i \ge 0$, $\sum_i \theta_i = 1$. The
simplex is the convex hull of the coordinate vectors: every probability
distribution is an average of certainties.

## Convex Functions: Three Lenses
:label:`subsec_mdl-three-lenses`

### The Chord Lens

A function $f : C \to \mathbb{R}$ on a convex domain $C$ is **convex** if its
graph lies on or below every chord:

$$
f\left(\theta \mathbf{x} + (1 - \theta)\, \mathbf{y}\right)
\;\le\; \theta f(\mathbf{x}) + (1 - \theta) f(\mathbf{y})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y} \in C,\ \theta \in [0, 1].
$$
:eqlabel:`eq_mdl-opt-chord`

The left side evaluates $f$ at a point of the segment; the right side is the
height of the chord joining $(\mathbf{x}, f(\mathbf{x}))$ to
$(\mathbf{y}, f(\mathbf{y}))$ above that same point. If the inequality is strict
whenever $\mathbf{x} \neq \mathbf{y}$ and $\theta \in (0, 1)$, $f$ is **strictly
convex**; if $-f$ is convex, $f$ is **concave**. Note that the domain must be
convex for the definition to even parse --- the point
$\theta\mathbf{x} + (1-\theta)\mathbf{y}$ has to be somewhere $f$ is defined.

Sets and functions are one theory, not two: $f$ is convex exactly when its
**epigraph** $\{(\mathbf{x}, t) : t \ge f(\mathbf{x})\}$ --- the region on and
above the graph --- is a convex set. Chords between points of the epigraph stay
above the graph precisely when :eqref:`eq_mdl-opt-chord` holds. This dictionary
lets every fact about convex sets generate a fact about convex functions, and we
will cash it in shortly.

### The First-Order Lens

The chord lens needs no derivatives. When $f$ is differentiable, an equivalent
test reads the tangent instead of the chord: the first-order Taylor
approximation at any point is a *global under-estimator*,

$$
f(\mathbf{y}) \;\ge\; f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y} \in C.
$$
:eqlabel:`eq_mdl-opt-first-order`

The two lenses are two ways of looking at the same picture, shown in
:numref:`fig_mdl-opt-chord-above-graph`. The chord lens reads off the definition
directly: the chord joining any two points on the graph lies *above* the graph.
The first-order lens reads the slope: the tangent at any point lies *below* the
graph. This is the property gradient methods exploit --- a single gradient
evaluation at $\mathbf{x}$ hands you a certificate about *every other point in
the domain*, however far away. In particular, if $\nabla f(\mathbf{x}) = \mathbf{0}$,
then :eqref:`eq_mdl-opt-first-order` says $f(\mathbf{y}) \ge f(\mathbf{x})$ for
all $\mathbf{y}$: a stationary point of a convex function is already a global
minimum. We will make that the headline of this section's fourth act.

![Two equivalent lenses on convexity. Left (chord lens): the chord joining two points lies above the graph, so $f(\theta\mathbf{x}+(1-\theta)\mathbf{y})\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$. Right (first-order lens): the tangent at a point lies below the graph, so $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$.](../img/mdl-opt-chord-above-graph.svg)
:label:`fig_mdl-opt-chord-above-graph`

### The Second-Order Lens

When $f$ is twice differentiable there is a third test, usually the easiest to
run: $f$ is convex if and only if its Hessian is positive semidefinite
everywhere,

$$
\nabla^2 f(\mathbf{x}) \succeq 0 \qquad \textrm{for all } \mathbf{x} \in C.
$$

In one dimension this is the calculus-class criterion $f''(x) \ge 0$ --- the
graph curves upward. In $n$ dimensions it says every one-dimensional slice
curves upward, since
$\mathbf{v}^\top \nabla^2 f(\mathbf{x})\, \mathbf{v}$ is the second derivative
of $f$ along the line through $\mathbf{x}$ in direction $\mathbf{v}$
(:numref:`sec_mdl-multivariable_calculus`). Three instant applications: the
quadratic $\tfrac12\mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top\mathbf{x}$
(with $A$ symmetric) has constant Hessian $A$, so it is convex iff
$A \succeq 0$; the least-squares loss
$\tfrac12\|X\mathbf{w} - \mathbf{y}\|^2$ has Hessian $X^\top X$, a Gram matrix,
which is *always* PSD since
$\mathbf{v}^\top X^\top X \mathbf{v} = \|X\mathbf{v}\|^2 \ge 0$; and
$e^x$, $-\log x$, and $x \log x$ are convex on their domains because their
second derivatives $e^x$, $1/x^2$, and $1/x$ are positive.

**Proposition (the three lenses agree).** *Let $f$ be differentiable on a convex
domain $C$. Then the chord condition :eqref:`eq_mdl-opt-chord` and the
first-order condition :eqref:`eq_mdl-opt-first-order` are equivalent. If $f$ is
twice differentiable, both are equivalent to
$\nabla^2 f(\mathbf{x}) \succeq 0$ on $C$.*

**Proof.** *(Chord $\Rightarrow$ first-order.)* Fix $\mathbf{x}, \mathbf{y}$ and
$\theta \in (0, 1]$. Rewriting :eqref:`eq_mdl-opt-chord` at the point
$\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})$ and subtracting $f(\mathbf{x})$,

$$
\frac{f(\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})) - f(\mathbf{x})}{\theta}
\;\le\; f(\mathbf{y}) - f(\mathbf{x}).
$$

As $\theta \downarrow 0$ the left side converges to the directional derivative
$\nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$, giving
:eqref:`eq_mdl-opt-first-order`.

*(First-order $\Rightarrow$ chord.)* Let
$\mathbf{z} = \theta\mathbf{x} + (1-\theta)\mathbf{y}$ and apply the
under-estimator at $\mathbf{z}$ twice, once toward each endpoint:

$$
f(\mathbf{x}) \ge f(\mathbf{z}) + \nabla f(\mathbf{z})^\top(\mathbf{x} - \mathbf{z}),
\qquad
f(\mathbf{y}) \ge f(\mathbf{z}) + \nabla f(\mathbf{z})^\top(\mathbf{y} - \mathbf{z}).
$$

Multiply by $\theta$ and $1 - \theta$ and add: the gradient terms combine to
$\nabla f(\mathbf{z})^\top(\theta\mathbf{x} + (1-\theta)\mathbf{y} - \mathbf{z}) = 0$,
leaving exactly :eqref:`eq_mdl-opt-chord`.

*(Second-order $\Rightarrow$ first-order.)* Restrict to the segment:
$g(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ has
$g''(t) = (\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))\,(\mathbf{y} - \mathbf{x}) \ge 0$.
Taylor's theorem with the Lagrange remainder
(:numref:`sec_mdl-single_variable_calculus`) gives
$g(1) = g(0) + g'(0) + \tfrac12 g''(\tau)$ for some $\tau \in (0, 1)$, and
dropping the nonnegative remainder is :eqref:`eq_mdl-opt-first-order`.

*(First-order $\Rightarrow$ second-order.)* Suppose instead
$\mathbf{v}^\top \nabla^2 f(\mathbf{x})\, \mathbf{v} < 0$ for some $\mathbf{x}$
and $\mathbf{v}$. Expanding along that direction,

$$
f(\mathbf{x} + t\mathbf{v}) = f(\mathbf{x}) + t\, \nabla f(\mathbf{x})^\top\mathbf{v}
+ \frac{t^2}{2}\, \mathbf{v}^\top \nabla^2 f(\mathbf{x})\, \mathbf{v} + o(t^2),
$$

so for small enough $t > 0$ the left side drops strictly below the tangent value
$f(\mathbf{x}) + t \nabla f(\mathbf{x})^\top\mathbf{v}$, contradicting
:eqref:`eq_mdl-opt-first-order`. $\blacksquare$

The practical reading: *pick whichever lens is cheapest to check*. The chord
lens needs no smoothness (it certifies $\|\mathbf{x}\|_1$ and the hinge loss,
where Hessians do not exist); the first-order lens is what optimization proofs
consume; the second-order lens is the workhorse for smooth losses, where
checking convexity means checking a matrix is PSD ---
:numref:`sec_mdl-eigendecompositions` territory.

### Strong Convexity

Convexity bounds curvature from below by zero. Two refinements sharpen the
geometry. $f$ is **strictly convex** if the chord inequality is strict, ruling
out flat segments. More quantitatively, $f$ is **$\mu$-strongly convex** (for
$\mu > 0$) if $f(\mathbf{x}) - \tfrac{\mu}{2}\|\mathbf{x}\|^2$ is convex ---
equivalently, in the first-order lens,

$$
f(\mathbf{y}) \;\ge\; f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})
+ \frac{\mu}{2}\, \|\mathbf{y} - \mathbf{x}\|^2,
$$
:eqlabel:`eq_mdl-opt-strong-convexity`

or, in the second-order lens, $\nabla^2 f \succeq \mu I$ everywhere. Strong
convexity says the graph does not merely stay above its tangents --- it pulls
away from them at least quadratically, like a bowl with a guaranteed minimum
curvature. Paired with the $L$-smoothness of
:numref:`sec_mdl-gradient-based-optimization` ($\nabla^2 f \preceq L I$), the
function is sandwiched between two quadratics, and the ratio $\kappa = L/\mu$
is precisely the condition number that governed convergence speed there. A
useful source of strong convexity: adding the ridge penalty
$\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ to any convex loss makes the sum
$\lambda$-strongly convex --- one more service weight decay performs.

### The Subgradient

The losses $\|\mathbf{x}\|_1$ and $\max(0, 1 - z)$ have corners, so
:eqref:`eq_mdl-opt-first-order` cannot be checked as written --- there is no
gradient at the kink. Convexity offers a graceful repair. A vector $\mathbf{g}$
is a **subgradient** of $f$ at $\mathbf{x}$ if it provides the same global
under-estimate a gradient would:

$$
f(\mathbf{y}) \;\ge\; f(\mathbf{x}) + \mathbf{g}^\top (\mathbf{y} - \mathbf{x})
\qquad \textrm{for all } \mathbf{y},
$$
:eqlabel:`eq_mdl-opt-subgradient`

and the set of all subgradients at $\mathbf{x}$ is written
$\partial f(\mathbf{x})$. For convex $f$ this set is nonempty at every interior
point of the domain --- some supporting line always fits under the graph ---
and where $f$ is differentiable it collapses to the singleton
$\{\nabla f(\mathbf{x})\}$. At a corner it fans out
(:numref:`fig_mdl-opt-subgradient-fan`): for $f(x) = |x|$ at the
origin, every slope $g \in [-1, 1]$ tucks the line $g\,x$ under the V, so
$\partial f(0) = [-1, 1]$; for the hinge $\max(0, 1 - z)$ at $z = 1$,
$\partial f(1) = [-1, 0]$. The optimality criterion survives verbatim:
$\mathbf{x}^\star$ minimizes $f$ iff
$\mathbf{0} \in \partial f(\mathbf{x}^\star)$ --- which is how $|x|$ is
minimized at its corner, where no gradient exists but the zero slope fits under
the graph. Subgradients make everything in this section --- Jensen's inequality,
local-equals-global, descent methods --- go through for ReLU-style kinks, and
they will quietly power the cleanest proof below.

![At the kink of $f(x) = |x|$ the gradient does not exist, but the subgradient fans out into a set: every slope $g \in \left(-1, 1\right)$ tucks a supporting line under the V, and the two extreme slopes $\pm 1$ lie along the branches themselves, so $\partial f(0)$ is the whole interval from $-1$ to $1$. The zero-slope member (orange) is the optimality certificate $0 \in \partial f(0)$: the corner is a provable minimum, no gradient required.](../img/mdl-opt-subgradient-fan.svg)
:label:`fig_mdl-opt-subgradient-fan`

### Checking the Lenses Numerically

Theory says the three lenses certify the same functions; the code checks all
three on the least-squares loss
$f(\mathbf{w}) = \tfrac12\|X\mathbf{w} - \mathbf{y}\|^2$ in two weights ---
random chords, random tangents, and the Hessian's eigenvalues.

```{.python .input #convexity-three-lenses}
rng = np.random.default_rng(0)
X = rng.normal(size=(20, 2))                   # design matrix, 20 points
y = rng.normal(size=20)
f = lambda w: 0.5 * ((X @ w - y) ** 2).sum()
grad = lambda w: X.T @ (X @ w - y)
worst_chord, worst_tangent = -np.inf, -np.inf
for _ in range(1000):
    u, v = rng.normal(size=(2, 2)) * 3.0       # random pair of weight vectors
    t = rng.uniform()
    worst_chord = max(worst_chord,             # f(chord point) - chord height
                      f(t * u + (1 - t) * v) - (t * f(u) + (1 - t) * f(v)))
    worst_tangent = max(worst_tangent,         # tangent value - f
                        f(u) + grad(u) @ (v - u) - f(v))
print(f'worst chord violation over 1000 trials:   {worst_chord:.2e}')
print(f'worst tangent violation over 1000 trials: {worst_tangent:.2e}')
print('Hessian eigenvalues:', np.linalg.eigvalsh(X.T @ X).round(4))
```

All three lenses report the same verdict. The worst chord "violation" across a
thousand random trials is $-2.46 \times 10^{-3}$ and the worst tangent
violation is $-4.95 \times 10^{-2}$ --- both *negative*, meaning the graph
stayed below every sampled chord and above every sampled tangent with room to
spare --- and the Hessian's eigenvalues, $6.273$ and $18.6557$, are positive.
(Sampling can only ever refute convexity, never prove it; the Hessian check is
the one that constitutes a proof, since $X^\top X$ is the same matrix at every
$\mathbf{w}$.)

## Jensen's Inequality
:label:`subsec_mdl-jensen`

The chord definition compares $f$ at a two-point average against the average of
$f$ at the two points. Nothing about the argument is special to *two*: applying
:eqref:`eq_mdl-opt-chord` repeatedly (Exercise 4) gives the finite form

$$
f\left(\sum_i \theta_i \mathbf{x}_i\right) \;\le\; \sum_i \theta_i\, f(\mathbf{x}_i),
\qquad \theta_i \ge 0,\ \sum_i \theta_i = 1,
$$

and a set of nonnegative weights summing to one is exactly a probability
distribution. The full generalization replaces averages by expectations, and it
is the single most-used inequality in this book's probabilistic chapters.

**Proposition (Jensen's inequality).** *Let $f$ be convex and let $X$ be a
random vector taking values in $f$'s domain, with $\mathbb{E}[X]$ finite and in
the domain's interior, and read $\mathbb{E}[f(X)]$ as an extended real number
(the value $+\infty$ is allowed, and makes the inequality trivially true).
Then*

$$
f\left(\mathbb{E}[X]\right) \;\le\; \mathbb{E}\left[f(X)\right].
$$
:eqlabel:`eq_mdl-opt-jensen`

*If $f$ is strictly convex, equality holds if and only if $X = \mathbb{E}[X]$
almost surely.*

**Proof.** Let $\boldsymbol{\mu} = \mathbb{E}[X]$ and pick a subgradient
$\mathbf{g} \in \partial f(\boldsymbol{\mu})$ --- one exists at an interior
point. The supporting-line inequality :eqref:`eq_mdl-opt-subgradient` holds
pointwise for every outcome:

$$
f(X) \;\ge\; f(\boldsymbol{\mu}) + \mathbf{g}^\top (X - \boldsymbol{\mu}).
$$

Take expectations of both sides: the linear term has mean zero, leaving
$\mathbb{E}[f(X)] \ge f(\boldsymbol{\mu})$. For strictly convex $f$ the
supporting line touches the graph only at $\boldsymbol{\mu}$, so the pointwise
inequality is strict on the event $X \neq \boldsymbol{\mu}$; equality in
expectation therefore forces that event to have probability zero. $\blacksquare$

Two lines, and notice *which* two: the first-order lens (in subgradient form, so
no smoothness is needed) plus linearity of expectation. A mnemonic for the
direction: the graph bends upward, so spreading $X$ out can only push the
average of $f(X)$ up --- *the function of the mean undershoots the mean of the
function*. For concave $f$ the inequality flips:
$\mathbb{E}[f(X)] \le f(\mathbb{E}[X])$.

The corollary that information theory is built on follows immediately, and
:numref:`sec_mdl-information_theory` will lean on it for Gibbs' inequality, the
entropy ceiling $H(X) \le \log k$, and the nonnegativity of mutual information.

**Corollary (nonnegativity of KL divergence).** *For probability distributions
$p$ and $q$ on a common finite alphabet (with $q(x) > 0$ where $p(x) > 0$),*

$$
D_{\mathrm{KL}}(p \,\|\, q) \;=\; \sum_x p(x) \log \frac{p(x)}{q(x)} \;\ge\; 0,
$$

*with equality if and only if $p = q$.*

**Proof.** Apply Jensen's inequality to the strictly convex function $-\log$
and the random variable $R = q(X)/p(X)$ with $X \sim p$:

$$
D_{\mathrm{KL}}(p \,\|\, q)
= \mathbb{E}_p\left[-\log R\right]
\;\ge\; -\log \mathbb{E}_p[R]
= -\log \sum_{x : p(x) > 0} p(x)\, \frac{q(x)}{p(x)}
\;\ge\; -\log 1 = 0,
$$

since $\sum_{x : p(x) > 0} q(x) \le 1$. By the equality case, the bound is tight
iff $R$ is almost surely constant and the $q$-mass off $p$'s support is zero ---
i.e., $q = c\,p$ with $c = 1$: $p = q$. $\blacksquare$

The same one-liner with the concave $\log$ yields the classical
**arithmetic--geometric mean inequality**: for positive $x_1, \ldots, x_n$,
concavity gives
$\log\bigl(\tfrac1n \sum_i x_i\bigr) \ge \tfrac1n \sum_i \log x_i$, and
exponentiating,

$$
\frac{1}{n} \sum_{i=1}^n x_i \;\ge\; \left(\prod_{i=1}^n x_i\right)^{1/n},
$$

with equality iff all $x_i$ coincide ($\log$ is strictly concave). Jensen also
explains a gap you have already trained on: the evidence lower bound of
:numref:`sec_mdl-latent-em-elbo` is Jensen applied to the concave $\log$ of an
expectation, and the slack in the ELBO *is* the Jensen gap. The cell makes the
inequality concrete three ways --- a Monte Carlo estimate of
$\mathbb{E}[e^X]$ versus $e^{\mathbb{E}[X]}$, a bulk test of AM--GM, and the
KL corollary on a thousand random distribution pairs.

```{.python .input #convexity-jensen-mc}
rng = np.random.default_rng(1)
z = rng.normal(size=1_000_000)                 # X ~ N(0, 1)
print(f'E[exp(X)] = {np.exp(z).mean():.4f}  vs  exp(E[X]) = {np.exp(z.mean()):.4f}'
      f'  (theory: sqrt(e) = {np.exp(0.5):.4f} vs 1)')
u = rng.uniform(0.1, 10.0, size=(100_000, 5))  # AM-GM from concavity of log
am, gm = u.mean(axis=1), np.exp(np.log(u).mean(axis=1))
print(f'AM >= GM in all trials: {bool((am >= gm).all())},'
      f'  smallest AM/GM ratio = {(am / gm).min():.4f}')
p = rng.dirichlet(np.ones(10), size=1000)      # 1000 random distribution pairs
q = rng.dirichlet(np.ones(10), size=1000)
print(f'min KL(p||q) = {(p * np.log(p / q)).sum(axis=1).min():.4f} >= 0,'
      f'  max |KL(p||p)| = {np.abs((p * np.log(p / p)).sum(axis=1)).max():.1e}')
```

With $X \sim \mathcal{N}(0, 1)$ and the convex $f(x) = e^x$, a million samples
give $\mathbb{E}[e^X] \approx 1.6466$ against
$e^{\mathbb{E}[X]} \approx 0.9998$ --- the true values are
$\sqrt{e} \approx 1.6487$ and $1$, a Jensen gap of $0.65$ that no amount of
sampling will close, because it is geometry, not noise. AM $\ge$ GM holds in
all $10^5$ random draws (smallest ratio $1.0002$), and across a thousand random
pairs of distributions on ten symbols the smallest KL divergence is $0.1331$,
comfortably nonnegative, while $D_{\mathrm{KL}}(p\,\|\,p)$ prints as
`0.0e+00` --- exactly zero, not merely small, the equality case certifying
itself at machine resolution.

## Why Convexity Matters
:label:`subsec_mdl-why-convexity-matters`

### Every Local Minimum Is Global

Here is the headline theorem, with the picture first.
:numref:`fig_mdl-opt-local-equals-global` shows why convexity is the dividing
line for optimization. On a convex objective, gradient descent reaches the
single global minimum no matter where it starts; on a non-convex landscape the
same algorithm slides into whichever basin it happens to start in, separated by
a saddle --- the local minimum on one side is *not* the global one.

![Why convexity matters. Left: a convex objective has one global minimum, and gradient descent from any start reaches it. Right: a non-convex landscape with two local minima separated by a saddle; gradient descent lands in different minima depending on its starting point, so a local minimum need not be global.](../img/mdl-opt-local-equals-global.svg)
:label:`fig_mdl-opt-local-equals-global`

**Proposition (local minima are global).** *Let $f$ be convex on a convex set
$C$ and let $\mathbf{x}^\star$ be a local minimum: $f(\mathbf{x}^\star) \le f(\mathbf{z})$
for all feasible $\mathbf{z}$ within some radius $r > 0$ of $\mathbf{x}^\star$.
Then $f(\mathbf{x}^\star) \le f(\mathbf{y})$ for* every *$\mathbf{y} \in C$.
Moreover the set of minimizers is convex, and if $f$ is differentiable on open
$C$, then any stationary point ($\nabla f = \mathbf{0}$) is a global minimum.*

**Proof.** Take any $\mathbf{y} \in C$ and slide from $\mathbf{x}^\star$ toward
it: for small $\theta > 0$ the point
$\mathbf{z}_\theta = (1 - \theta)\mathbf{x}^\star + \theta\mathbf{y}$ is
feasible (convexity of $C$) and within $r$ of $\mathbf{x}^\star$, so local
minimality and the chord inequality give

$$
f(\mathbf{x}^\star) \;\le\; f(\mathbf{z}_\theta)
\;\le\; (1 - \theta) f(\mathbf{x}^\star) + \theta f(\mathbf{y}).
$$

Subtract $(1-\theta)f(\mathbf{x}^\star)$ and divide by $\theta$:
$f(\mathbf{x}^\star) \le f(\mathbf{y})$. The minimizer set is the sublevel set
$\{\mathbf{x} \in C : f(\mathbf{x}) \le f^\star\}$, and every sublevel set of a
convex function is convex by the chord inequality. The stationary-point claim is
:eqref:`eq_mdl-opt-first-order` read at $\mathbf{x}$ with
$\nabla f(\mathbf{x}) = \mathbf{0}$. $\blacksquare$

Read the proof's geometry: a convex function cannot ambush you. If anywhere in
the domain there were a strictly better point, the chord toward it would already
be descending *inside your local neighborhood* --- so "no local improvement"
instantly means "no improvement anywhere." Saddle points and spurious basins,
the failure modes that haunt deep landscapes, are structurally impossible. One
more consequence is cheap and worth pocketing:

**Proposition (uniqueness).** *A strictly convex $f$ has at most one minimizer;
a strongly convex $f$ on a closed convex set has exactly one.*

**Proof.** If $\mathbf{x} \neq \mathbf{y}$ were both minimizers with value
$f^\star$, the strict chord inequality at $\theta = \tfrac12$ would give
$f\bigl(\tfrac{\mathbf{x} + \mathbf{y}}{2}\bigr) < f^\star$ --- contradiction.
Strong convexity adds existence: by :eqref:`eq_mdl-opt-strong-convexity` the
function grows at least quadratically away from any fixed point, so sublevel
sets are bounded, and a continuous function on a closed bounded set attains its
minimum. $\blacksquare$

The code stages the dividing line as an experiment: the same gradient-descent
loop, the same step size, the same $500$ random starting points --- on a convex
bowl $f(x) = (x - \tfrac12)^2$ and on the tilted double well
$g(x) = (x^2 - 1)^2 + x/2$.

```{.python .input #convexity-basins}
def gd(grad_fn, x, eta=0.05, steps=600):
    for _ in range(steps):
        x = x - eta * grad_fn(x)
    return x

rng = np.random.default_rng(2)
inits = rng.uniform(-1.5, 1.5, size=500)       # 500 random starting points
bowl = gd(lambda x: 2 * (x - 0.5), inits)      # convex: f(x) = (x - 1/2)^2
g = lambda x: (x**2 - 1)**2 + 0.5 * x          # non-convex: tilted double well
well = gd(lambda x: 4 * x * (x**2 - 1) + 0.5, inits)
left, right = well[well < 0], well[well > 0]
print(f'convex bowl: all {bowl.size} runs end at x = {bowl.mean():.6f}'
      f'  (spread {np.ptp(bowl):.1e})')
print(f'double well: {left.size} runs -> x = {left.mean():.4f}'
      f'  (g = {g(left.mean()):.4f})')
print(f'             {right.size} runs -> x = {right.mean():.4f}'
      f'  (g = {g(right.mean()):.4f})')
```

On the bowl, all $500$ runs collapse onto $x = 0.5$ with a spread of
$6.7 \times 10^{-16}$ --- machine epsilon; the histogram of outcomes is a single
spike. On the double well the histogram has two bars: $271$ runs find the global
minimum at $x \approx -1.06$ with $g \approx -0.515$, while the other $229$ are
captured by the local minimum at $x \approx 0.93$ with $g \approx 0.483$ ---
nearly a full unit worse, and no amount of further descent will fix it. Nothing
about the *algorithm* changed between the two lines; convexity is a property of
the landscape, and it alone decides whether initialization is a detail or
destiny.

### From Local Steps to Global Rates

:numref:`sec_mdl-gradient-based-optimization` proved the **descent lemma**: for
$L$-smooth $f$, a gradient step with $\eta = 1/L$ achieves

$$
f(\mathbf{x}_{t+1}) \;\le\; f(\mathbf{x}_t) - \frac{1}{2L}\, \|\nabla f(\mathbf{x}_t)\|^2,
\qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \tfrac{1}{L} \nabla f(\mathbf{x}_t).
$$

By itself this guarantees only that gradients eventually vanish --- a
*stationary* point, which in general may be a saddle or a poor local minimum.
Convexity upgrades the guarantee from "somewhere flat" to "the global optimum,
this fast." These two theorems are the payoff of the section, and both proofs
are short enough to carry with you :cite:`Nesterov.2018`.

**Proposition (gradient descent on smooth convex functions).** *Let $f$ be
convex and $L$-smooth with a minimizer $\mathbf{x}^\star$. Gradient descent with
$\eta = 1/L$ satisfies, for every $k \ge 1$,*

$$
f(\mathbf{x}_k) - f(\mathbf{x}^\star) \;\le\; \frac{L\, \|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}.
$$
:eqlabel:`eq_mdl-opt-rate-convex`

**Proof.** Chain the descent lemma into the first-order lens
:eqref:`eq_mdl-opt-first-order` aimed at the optimum,
$f(\mathbf{x}_t) - f(\mathbf{x}^\star) \le \nabla f(\mathbf{x}_t)^\top(\mathbf{x}_t - \mathbf{x}^\star)$:

$$
f(\mathbf{x}_{t+1}) - f(\mathbf{x}^\star)
\;\le\; \nabla f(\mathbf{x}_t)^\top (\mathbf{x}_t - \mathbf{x}^\star) - \frac{1}{2L} \|\nabla f(\mathbf{x}_t)\|^2.
$$

Completing the square turns the right side into a difference of distances:

$$
\nabla f(\mathbf{x}_t)^\top (\mathbf{x}_t - \mathbf{x}^\star) - \frac{1}{2L} \|\nabla f(\mathbf{x}_t)\|^2
= \frac{L}{2} \left( \|\mathbf{x}_t - \mathbf{x}^\star\|^2
- \left\|\mathbf{x}_t - \mathbf{x}^\star - \tfrac{1}{L}\nabla f(\mathbf{x}_t)\right\|^2 \right),
$$

and the second norm is exactly $\|\mathbf{x}_{t+1} - \mathbf{x}^\star\|^2$.
Summing over $t = 0, \ldots, k-1$, the right side telescopes:

$$
\sum_{t=0}^{k-1} \left( f(\mathbf{x}_{t+1}) - f(\mathbf{x}^\star) \right)
\;\le\; \frac{L}{2} \left( \|\mathbf{x}_0 - \mathbf{x}^\star\|^2 - \|\mathbf{x}_k - \mathbf{x}^\star\|^2 \right)
\;\le\; \frac{L}{2}\, \|\mathbf{x}_0 - \mathbf{x}^\star\|^2.
$$

The descent lemma makes $f(\mathbf{x}_t)$ nonincreasing, so the last term of the
sum is the smallest: $k \left(f(\mathbf{x}_k) - f(\mathbf{x}^\star)\right)$ is at
most the sum, which gives :eqref:`eq_mdl-opt-rate-convex`. $\blacksquare$

Two remarks. The bound is *dimension-free* --- a million parameters cost no more
iterations than two, which is the deep reason first-order methods scale to
neural networks. And the telescoping display has a bonus reading: every bracket
is nonnegative (its left factor is $f(\mathbf{x}_{t+1}) - f^\star \ge 0$), so
$\|\mathbf{x}_t - \mathbf{x}^\star\|$ never increases --- the iterates do not
just descend in value, they march monotonically closer to the optimum.

With strong convexity, the sublinear $O(1/k)$ sharpens to a geometric decay.

**Proposition (linear rate under strong convexity).** *Let $f$ be
$\mu$-strongly convex and $L$-smooth, with minimizer $\mathbf{x}^\star$.
Gradient descent with $\eta = 1/L$ satisfies*

$$
f(\mathbf{x}_k) - f(\mathbf{x}^\star)
\;\le\; \left(1 - \frac{\mu}{L}\right)^{k} \left( f(\mathbf{x}_0) - f(\mathbf{x}^\star) \right).
$$
:eqlabel:`eq_mdl-opt-rate-strong`

**Proof.** First, strong convexity links the gradient's size to the remaining
suboptimality. Minimize both sides of :eqref:`eq_mdl-opt-strong-convexity` over
$\mathbf{y}$: the left side's infimum is $f(\mathbf{x}^\star)$, and the right
side is a quadratic in $\mathbf{y}$ minimized at
$\mathbf{y} = \mathbf{x} - \tfrac{1}{\mu}\nabla f(\mathbf{x})$, where it equals
$f(\mathbf{x}) - \tfrac{1}{2\mu}\|\nabla f(\mathbf{x})\|^2$. Hence

$$
\frac12\, \|\nabla f(\mathbf{x})\|^2 \;\ge\; \mu \left( f(\mathbf{x}) - f(\mathbf{x}^\star) \right)
\qquad \textrm{for all } \mathbf{x}.
$$
:eqlabel:`eq_mdl-opt-pl`

Now feed :eqref:`eq_mdl-opt-pl` into the descent lemma:

$$
f(\mathbf{x}_{t+1}) - f(\mathbf{x}^\star)
\;\le\; f(\mathbf{x}_t) - f(\mathbf{x}^\star) - \frac{1}{2L} \|\nabla f(\mathbf{x}_t)\|^2
\;\le\; \left(1 - \frac{\mu}{L}\right) \left( f(\mathbf{x}_t) - f(\mathbf{x}^\star) \right),
$$

and iterate $k$ times. $\blacksquare$

The contraction factor is $1 - 1/\kappa$ with $\kappa = L/\mu$ the condition
number: reaching accuracy $\epsilon$ costs $O(\kappa \log(1/\epsilon))$
iterations, against $O(1/\epsilon)$ without strong convexity --- this is the
global, every-start version of the per-mode contraction that
:numref:`sec_mdl-gradient-based-optimization` measured on quadratics. File away
one observation for the final act of this section: the second half of the proof
never used convexity at all. It consumed only the inequality
:eqref:`eq_mdl-opt-pl` --- a fact we will exploit when convexity itself is gone.

These two rates are the payoff of the whole theory, so they should not escape
the pattern every other proposition in this section obeys: state, prove,
*measure*. The cell runs gradient descent at $\eta = 1/L$ on the least-squares
toy of the `#convexity-three-lenses` cell --- strongly convex, with $\mu$ and
$L$ read off the eigenvalues of $X^\top X$ --- and prints the per-step
contraction of $f - f^\star$ next to the theorem's promise:

```{.python .input #convexity-rate-check}
mu, L = np.linalg.eigvalsh(X.T @ X)[[0, -1]]   # curvature floor and ceiling
w_star = np.linalg.solve(X.T @ X, X.T @ y)     # the unique minimizer
w, gaps = np.zeros(2), []
for _ in range(25):                            # gradient descent at eta = 1/L
    w -= grad(w) / L
    gaps.append(f(w) - f(w_star))
r = np.array(gaps[1:]) / np.array(gaps[:-1])
print(f'mu = {mu:.4f}, L = {L:.4f}:  the theorem promises '
      f'contraction <= 1 - mu/L = {1 - mu / L:.4f}')
print('measured per-step contraction of f - f*:', r[::6].round(4))
print(f'worst step: {r.max():.4f}  (the bound holds at every step)')
```

The measured contraction is $0.4406$ at every step, safely inside the promised
$1 - \mu/L = 0.6637$ --- and the gap between the two is itself informative. On
a pure quadratic the slow mode's *distance* to the optimum contracts by
exactly $1 - \mu/L$ per step, and the value gap, being quadratic in the
distance, contracts by its square: $(1 - \mu/L)^2 = 0.4406$, precisely the
printed figure. The theorem's rate is tight for the quantity it bounds and the
class it covers; on any particular function you may go faster, never slower.

## Recognizing Convexity and Its Limits
:label:`subsec_mdl-recognizing-convexity`

### A Calculus of Convex Functions

Checking Hessians is honest work, but most convexity proofs in practice are
assembled, not computed --- a small set of operations preserves convexity, and
real losses are built from convex atoms by exactly these operations.

1. **Nonnegative weighted sums.** If $f_1, \ldots, f_m$ are convex and
   $w_i \ge 0$, then $\sum_i w_i f_i$ is convex: multiply the chord inequality
   for each $f_i$ by $w_i$ and add. (An average of losses over a training set is
   a nonnegative weighted sum.)
2. **Affine pre-composition.** If $f$ is convex then so is
   $\mathbf{x} \mapsto f(A\mathbf{x} + \mathbf{b})$: affine maps send the
   segment between $\mathbf{x}$ and $\mathbf{y}$ to the segment between their
   images, with the same $\theta$, so the chord inequality transfers verbatim.
   (A loss that is convex in a model's *output* is convex in the *weights of a
   linear model* producing that output.)
3. **Pointwise maximum and supremum.** If each $f_i$ is convex, so is
   $\max_i f_i$ --- and the slick proof is the epigraph dictionary: the
   epigraph of a max is the *intersection* of the epigraphs, and intersections
   of convex sets are convex. The upper envelope of any family of lines is
   convex, however large the family.
4. **Monotone convex composition.** If $g$ is convex and $h$ is convex *and
   nondecreasing*, then $h(g(\mathbf{x}))$ is convex (in one dimension:
   $(h \circ g)'' = h''(g)\,(g')^2 + h'(g)\, g'' \ge 0$, each summand
   nonnegative). The monotonicity hypothesis is not decoration ---
   $h(t) = t^2$ and $g(x) = x^2 - 1$ compose to the double well we just watched
   trap gradient descent.

This calculus certifies most of the convex losses in this book in one line
each. The hinge loss $\max(0, 1 - y\,\mathbf{w}^\top\mathbf{x})$ is a maximum
of two affine functions of $\mathbf{w}$ (rules 3 and 2). The $\ell_1$ norm
$\|\mathbf{w}\|_1 = \sum_i \max(w_i, -w_i)$ is a sum of maxima of affine
functions (rules 1, 3, 2). The logistic loss
$\log(1 + e^{-y\,\mathbf{w}^\top\mathbf{x}})$ is the convex
$t \mapsto \log(1 + e^t)$ (second derivative
$\sigma'(t) = \sigma(t)(1-\sigma(t)) > 0$) pre-composed with an affine map, then
summed over data. Ridge-regularized anything is "convex plus
$\lambda$-strongly convex," hence strongly convex (rule 1 applied to
:eqref:`eq_mdl-opt-strong-convexity`). The one conspicuous absence from the
list --- compositions with *nonlinear, non-monotone* inner maps --- is where
deep networks will exit the theory below.

### Log-Sum-Exp and the Softmax Covariance

The canonical worked case --- the function behind every softmax cross-entropy
loss --- deserves its own proposition. Define
$\mathrm{lse}(\mathbf{x}) = \log \sum_{i=1}^n e^{x_i}$, the smooth maximum of
:numref:`sec_mdl-numerical-stability-conditioning`.

**Proposition (log-sum-exp is convex).** *The Hessian of $\mathrm{lse}$ at
$\mathbf{x}$ is*

$$
\nabla^2 \mathrm{lse}(\mathbf{x}) = \mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top,
\qquad \mathbf{s} = \mathrm{softmax}(\mathbf{x}),
$$

*which is the covariance matrix of the one-hot encoding of a class drawn from
$\mathbf{s}$ --- hence PSD, hence $\mathrm{lse}$ is convex.*

**Proof.** Differentiating once gives
$\partial\, \mathrm{lse} / \partial x_i = e^{x_i} / \sum_j e^{x_j} = s_i$: the
gradient of $\mathrm{lse}$ *is* the softmax. Differentiating again,
$\partial s_i / \partial x_j = s_i \delta_{ij} - s_i s_j$, which is the claimed
matrix. Now let $I$ be a random index with $\Pr(I = i) = s_i$ and let
$\mathbf{e}_I$ be its one-hot vector. Then
$\mathbb{E}[\mathbf{e}_I] = \mathbf{s}$ and
$\mathbb{E}[\mathbf{e}_I \mathbf{e}_I^\top] = \mathrm{diag}(\mathbf{s})$, so
$\mathrm{Cov}(\mathbf{e}_I) = \mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$
is exactly the Hessian. Covariances are PSD --- directly,

$$
\mathbf{v}^\top \nabla^2 \mathrm{lse}\, \mathbf{v}
= \sum_i s_i v_i^2 - \left(\sum_i s_i v_i\right)^2
= \mathbb{E}\left[v_I^2\right] - \left(\mathbb{E}[v_I]\right)^2
= \mathrm{Var}(v_I) \;\ge\; 0. \;\blacksquare
$$

The corollary you use daily: the cross-entropy loss *in the logits*,
$\ell(\mathbf{z}) = \mathrm{lse}(\mathbf{z}) - z_y$ for true class $y$, is
log-sum-exp plus an affine function --- convex. Softmax regression
(:numref:`sec_softmax`) composes it with the affine map
$\mathbf{z} = W\mathbf{x} + \mathbf{b}$, so the loss is convex in the weights
(rule 2): the last layer of a classifier is always a convex problem, whatever
the features feeding it. The Hessian formula also predicts one *zero*
eigenvalue, in the direction $\mathbf{1}$: a variance $\mathrm{Var}(v_I)$
vanishes when $v$ is constant, which is the shift invariance
$\mathrm{lse}(\mathbf{x} + c\mathbf{1}) = \mathrm{lse}(\mathbf{x}) + c$ --- the
very identity that powers the numerically stable softmax of
:numref:`sec_mdl-numerical-stability-conditioning`. The cell verifies all of it:
the eigenvalues, the covariance identity by Monte Carlo, and the flat direction.

```{.python .input #convexity-lse-hessian}
lse = lambda x: x.max() + np.log(np.exp(x - x.max()).sum())
rng = np.random.default_rng(3)
x = rng.normal(size=6)                         # six logits
s = np.exp(x - lse(x))                         # softmax(x)
H = np.diag(s) - np.outer(s, s)                # analytic Hessian of lse
print('eigenvalues of H:', np.linalg.eigvalsh(H).round(6))
draws = rng.choice(len(x), size=200_000, p=s)  # class I ~ Categorical(s)
onehot = np.eye(len(x))[draws]
print(f'max |H - covariance of one-hot samples| = '
      f'{np.abs(H - np.cov(onehot.T)).max():.4f}')
print(f'max |H @ 1| = {np.abs(H.sum(axis=1)).max():.1e}  (flat shift direction)')
```

Every eigenvalue is nonnegative, from $0.284324$ down to one numerical zero ---
the predicted flat direction, confirmed by
$\|H\mathbf{1}\|_\infty \approx 1.8 \times 10^{-16}$. And the empirical
covariance of $200{,}000$ one-hot draws matches the analytic Hessian to within
$0.0006$: the Hessian of log-sum-exp really is a covariance matrix you can
sample from.

### The Convex Conjugate
:label:`subsec_mdl-convex-conjugate`

One more construction completes the toolkit, and it is the bridge between this
section and both duality (:numref:`sec_mdl-constrained-optimization-duality`)
and the variational representations of information theory. The **convex
conjugate** (or Fenchel--Legendre transform) of $f : \mathbb{R}^n \to \mathbb{R}$
is

$$
f^*(\mathbf{y}) \;=\; \sup_{\mathbf{x}} \;\left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right).
$$
:eqlabel:`eq_mdl-opt-conjugate`

Geometrically, $f^*(\mathbf{y})$ asks: among all affine functions with slope
$\mathbf{y}$, how high can one be pushed while staying below the graph of $f$?
(It records the negative of that intercept.) The conjugate thus encodes $f$ by
its supporting lines instead of its values --- and for closed convex $f$ the
encoding is lossless: $f^{**} = f$ (:citet:`Boyd.Vandenberghe.2004`, section
3.3). For non-convex $f$ the double transform returns not $f$ but the largest
closed convex function below it --- the **convex envelope** --- which is why
conjugacy is the engine of convex *relaxations*: $f^{**}$ is the tightest
convex stand-in for $f$ that any method restricted to supporting lines can
see. Two properties cost nothing. First, $f^*$ is *always* convex, whatever
$f$ is: for fixed $\mathbf{x}$ the expression
$\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})$ is affine in $\mathbf{y}$, and rule
3 says a supremum of affine functions is convex --- the same one-line argument
that, in the next section, makes the dual function of
:numref:`subsec_mdl-lagrangian-duality` concave, and no coincidence: for
linearly constrained problems the dual function *is* a conjugate in disguise.
Second, the definition rearranges into the **Fenchel--Young inequality**,

$$
\mathbf{y}^\top \mathbf{x} \;\le\; f(\mathbf{x}) + f^*(\mathbf{y})
\qquad \textrm{for all } \mathbf{x}, \mathbf{y},
$$
:eqlabel:`eq_mdl-opt-fenchel-young`

with equality exactly when $\mathbf{x}$ attains the supremum --- for
differentiable $f$, when $\mathbf{y} = \nabla f(\mathbf{x})$.

The simplest example sets the pattern: for $f(\mathbf{x}) = \tfrac12\|\mathbf{x}\|^2$,
the supremum of $\mathbf{y}^\top\mathbf{x} - \tfrac12\|\mathbf{x}\|^2$ sits at
$\mathbf{x} = \mathbf{y}$, giving $f^*(\mathbf{y}) = \tfrac12\|\mathbf{y}\|^2$:
the squared norm is its own conjugate, and Fenchel--Young becomes the familiar
$\mathbf{y}^\top\mathbf{x} \le \tfrac12\|\mathbf{x}\|^2 + \tfrac12\|\mathbf{y}\|^2$.

The pair that matters for deep learning is **log-sum-exp $\leftrightarrow$
negative entropy**. Compute
$\mathrm{lse}^*(\mathbf{p}) = \sup_{\mathbf{x}} \mathbf{p}^\top\mathbf{x} - \mathrm{lse}(\mathbf{x})$.
If $\mathbf{p}$ has a negative coordinate, send that $x_i \to -\infty$ and the
objective blows up; if $\mathbf{1}^\top\mathbf{p} \neq 1$, march $\mathbf{x} = c\mathbf{1}$
off to $\pm\infty$ and shift invariance does the same. So the supremum is finite
only for $\mathbf{p}$ in the simplex, where stationarity demands
$\mathrm{softmax}(\mathbf{x}) = \mathbf{p}$ --- solved by
$\mathbf{x} = \log \mathbf{p}$ --- and the value is
$\sum_i p_i \log p_i$. Therefore

$$
\mathrm{lse}^*(\mathbf{p}) =
\begin{cases}
\sum_i p_i \log p_i = -H(\mathbf{p}), & \mathbf{p} \in \Delta, \\
+\infty, & \textrm{otherwise,}
\end{cases}
\qquad \textrm{and dually} \qquad
\mathrm{lse}(\mathbf{x}) = \max_{\mathbf{p} \in \Delta} \left( \mathbf{p}^\top \mathbf{x} + H(\mathbf{p}) \right).
$$
:eqlabel:`eq_mdl-opt-lse-entropy`

The dual formula is worth reading aloud: log-sum-exp is the best *entropy-bonused*
score over all distributions, and the maximizing $\mathbf{p}$ is the softmax.
This single identity unifies three things you have met separately --- softmax as
the gradient of $\mathrm{lse}$, softmax as the *maximum-entropy* answer (drop
the bonus and the maximizer of $\mathbf{p}^\top\mathbf{x}$ alone is a hard,
one-hot $\arg\max$; the entropy term is what softens it), and the
entropy-regularized policies of reinforcement learning. It is also the prototype
for the variational representations of divergences --- Donsker--Varadhan and the
$f$-GAN duals in :numref:`sec_mdl-divergences-distances` are this same
conjugate trick applied to KL and its relatives.

### Proximal Operators
:label:`subsec_mdl-proximal-operators`

The conjugate encodes a convex function by its supporting lines; one last
construction turns convex functions into *algorithm components*, and it
unifies two things this chapter otherwise treats separately --- gradient steps
and projections. The **proximal operator** of a convex $f$ is

$$
\mathrm{prox}_{f}(\mathbf{z}) \;=\; \mathop{\mathrm{argmin}}_{\mathbf{x}}\; \left( f(\mathbf{x}) + \tfrac12\, \|\mathbf{x} - \mathbf{z}\|^2 \right),
$$
:eqlabel:`eq_mdl-opt-prox`

well defined because the objective is $1$-strongly convex --- the uniqueness
proposition above guarantees exactly one minimizer. Read it as a negotiated
step: move toward lower $f$, but pay quadratically for straying from
$\mathbf{z}$. Two special cases calibrate the definition. If $f = 0$, the prox
is the identity. If $f$ is the *indicator* of a convex set $C$ --- zero on
$C$, $+\infty$ outside --- then minimizing forces $\mathbf{x} \in C$ and the
prox is the nearest feasible point:
$\mathrm{prox}_f = \Pi_C$, the projection operator of
:numref:`subsec_mdl-projected-gd`. Proximal operators are projections,
generalized from sets to functions.

The example that matters for sparsity is $f(x) = \lambda |x|$ (the operator
acts coordinate-wise on $\lambda\|\cdot\|_1$, so one dimension suffices). The
objective $\lambda|x| + \tfrac12 (x - z)^2$ has a kink at $0$, and the
subgradient optimality criterion $0 \in \partial(\cdot)$ from
:numref:`subsec_mdl-three-lenses` dispatches it in three lines: if the
minimizer $x \neq 0$, stationarity reads $\lambda\,\mathrm{sign}(x) + x - z = 0$,
i.e. $x = z - \lambda\,\mathrm{sign}(z)$, consistent only when $|z| > \lambda$;
if $x = 0$, the criterion asks $z \in [-\lambda, \lambda]$; and the two cases
splice into one formula,

$$
\mathrm{prox}_{\lambda|\cdot|}(z) \;=\; \mathrm{sign}(z)\, \max\left(|z| - \lambda,\, 0\right)
$$
:eqlabel:`eq_mdl-opt-soft-threshold`

--- **soft-thresholding**, the same solution Exercise 3 derives from the
subdifferential, now recognized as a prox. Inputs smaller than $\lambda$ are
snapped to *exactly* zero; larger ones are shrunk by $\lambda$. This is the
mechanism by which $\ell_1$ regularization produces genuinely sparse weights
where $\ell_2$ only shrinks them.

The algorithmic payoff is one line long. For a composite objective
$g(\mathbf{x}) + h(\mathbf{x})$ with $g$ smooth and $h$ convex but kinked ---
the lasso, with $g$ the least-squares term and $h = \lambda\|\cdot\|_1$, is
the prototype --- alternate a gradient step on the smooth part with a prox on
the kinked part:

$$
\mathbf{x}_{k+1} = \mathrm{prox}_{\eta h}\left(\mathbf{x}_k - \eta\, \nabla g(\mathbf{x}_k)\right).
$$

This is **ISTA**, the proximal-gradient method: it converges at the same
$O(1/k)$ rate as plain gradient descent, kink notwithstanding, and it contains
the projected gradient descent of
:numref:`sec_mdl-constrained-optimization-duality` as the special case where
$h$ is an indicator. The cell verifies :eqref:`eq_mdl-opt-soft-threshold`
against brute-force minimization, then runs twenty ISTA steps on a tiny lasso
and watches exact zeros appear:

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
*exactly* zero --- not small: zero, snapped there by the $\max$ in
:eqref:`eq_mdl-opt-soft-threshold` --- while the three nonzero coordinates sit
near the planted values $(2.0, -1.5, 1.0)$. No amount of plain gradient
descent can produce an exact zero (a smooth step lands wherever the arithmetic
falls); the prox is the piece of the algorithm that *knows about the kink*,
and one application per step is all sparsity costs.

### Reality Check: Deep Networks Are Non-Convex

Now for the honest part. The convexity calculus had one missing rule ---
composition with non-monotone nonlinear maps --- and a neural network is nothing
but a tower of such compositions. The loss surface of a deep network in its
*parameters* is not convex, and a two-parameter "network" already shows why.
Take $f(a, b) = (ab - 1)^2$: a linear model with two stacked scalar weights and
a squared loss. Its global minima form the hyperbola $\{ab = 1\}$ --- a
*non-convex set*. But our local-equals-global proposition proved the minimizer
set of any convex function is convex; therefore $f$ cannot be convex, and
indeed the average of the minimizers $(1, 1)$ and $(-1, -1)$ is the origin,
where $f = 1$ sits strictly above the minimum $0$. Real networks inherit this
structurally: permuting the hidden units of a layer (with their weights) leaves
the computed function unchanged, so every minimum comes with combinatorially
many symmetric copies scattered across parameter space, and a set containing
all of them and their averages cannot consist of minima alone. Convexity is not
mildly violated by deep learning; it is demolished by the architecture itself.

What survives the demolition? More than the worst case suggests, and the
linear-rate proof already told us where to look. Its second half consumed only
the inequality :eqref:`eq_mdl-opt-pl`, known as the
**Polyak--Łojasiewicz (PL) condition** (after Polyak and Łojasiewicz, who
introduced it independently in 1963):

$$
f \textrm{ is } L\textrm{-smooth} \;\;\textrm{and}\;\;
\frac12\, \|\nabla f(\mathbf{x})\|^2 \;\ge\; \mu \left( f(\mathbf{x}) - f^\star \right)
\;\;\textrm{for all } \mathbf{x}
\qquad \Longrightarrow \qquad
\textrm{GD converges linearly to } f^\star,
$$

with no convexity required :cite:`Karimi.Nutini.Schmidt.2016` (smoothness
stays in the hypothesis --- it is what powers the descent lemma half of the
argument). PL says the
gradient cannot be small unless the *value* is nearly optimal --- flat spots
exist only at the bottom. Strong convexity implies PL (that was the first half
of the proof), but the converse fails: PL functions can have multiple minima,
plateaus of minimizers, and wild non-convexity, as long as every stationary
point is global. The standard example is $f(x) = x^2 + 3\sin^2 x$, whose second
derivative dips to $-4$ (decisively non-convex) yet which satisfies PL with a
healthy constant. The cell verifies both claims and then runs gradient descent,
watching the suboptimality gap $f(x_k) - f^\star$ shrink by a near-constant
factor per step --- the signature of a linear rate.

```{.python .input #convexity-pl-rate}
f = lambda x: x**2 + 3 * np.sin(x)**2          # nonconvex; f* = 0 at x = 0
df = lambda x: 2 * x + 3 * np.sin(2 * x)
xs = np.linspace(-5, 5, 100001)
xs = xs[np.abs(xs) > 1e-3]                     # avoid 0/0 at the minimizer
print(f'PL constant on [-5, 5]: mu = {(0.5 * df(xs)**2 / f(xs)).min():.4f}')
print(f"min f'' = {(2 + 6 * np.cos(2 * xs)).min():.2f}  (f is not convex)")
x, gaps = 4.5, []
for _ in range(80):                            # gradient descent, eta = 1/16
    x -= df(x) / 16.0
    gaps.append(f(x))
ratios = np.array(gaps[1:]) / np.array(gaps[:-1])
print('successive gap ratios:', ratios[::16].round(4))
```

The numerically measured PL constant on $[-5, 5]$ is $\mu \approx 0.1755$, and
the minimum of $f''$ is $-4.00$: genuinely non-convex, genuinely PL. Under
gradient descent the successive gap ratios start at $0.6293$ while the iterate
crosses the slow middle stretch, then settle at exactly $0.25$ --- a constant
contraction, i.e. linear convergence, on a non-convex function. (The asymptotic
$0.25$ is no mystery: near the minimum $f$ looks like $\tfrac12 f''(0) x^2$ with
$f''(0) = 8$, each step scales $x$ by $1 - 8/16 = \tfrac12$, and the gap is
quadratic in $x$.) Conditions of PL type are one current explanation for why
heavily over-parameterized networks train so reliably: near interpolation, wide
networks have been argued to satisfy local PL-style inequalities
:cite:`Liu.Zhu.Belkin.2022`, so first-order methods converge fast to global
*training-loss* minima even though the landscape is nothing like convex.

Non-convexity has one more consequence the convex theory never had to face:
*which* global minimum you reach is up for grabs, and the optimizer itself does
the choosing. When many minimizers exist, gradient descent does not pick
arbitrarily --- it has an **implicit bias**. The cleanest case is
underdetermined least squares: started from $\mathbf{w}_0 = \mathbf{0}$, every
gradient $X^\top(X\mathbf{w} - \mathbf{y})$ lies in the row space of $X$, so
the iterates never leave it, and the limit is the *minimum-norm* interpolant
--- the same solution the pseudoinverse computes
(:numref:`sec_mdl-svd-low-rank`; Exercise 8 has you verify it, and Exercise 9
of :numref:`sec_mdl-gradient-based-optimization` has you derive it). For
logistic regression on separable data, gradient descent's direction converges
to the maximum-margin separator :cite:`Soudry.Hoffer.Nacson.ea.2018` --- the
SVM solution of :numref:`sec_mdl-constrained-optimization-duality`, never asked
for, delivered anyway. Which minimum an optimizer prefers is part of what a
trained model *is*, and it is one reason architecture-plus-optimizer, not
architecture alone, determines generalization.

So the honest framing of this chapter's relationship to practice: convex theory
is the *idealization* that the working optimizer approximates, not a literal
description of deep training. The losses are convex in the last layer and in
many sub-problems (projections, trust regions, duals); the proofs --- descent
lemma, PL, rates --- are the instruments we carry into non-convex territory; and
where the instruments lose their guarantees, they usually keep their shape:
the same step sizes, the same contraction heuristics, the same role for
curvature and conditioning.

## Summary

* A set is **convex** when every chord stays inside; half-spaces, norm balls,
  the simplex, and the PSD cone are the working catalog, and **intersection**
  preserves convexity (unions do not) --- the factory that certifies the simplex,
  the PSD cone, and every polyhedral feasible set.
* A function is convex by any of **three equivalent lenses**: chord above graph,
  tangent below graph (the global under-estimator gradient methods exploit), or
  PSD Hessian. **Strong convexity** ($\nabla^2 f \succeq \mu I$) adds a
  curvature floor; **subgradients** extend the tangent lens to kinked losses
  like $\ell_1$ and the hinge, with $\partial |x|(0) = [-1, 1]$.
* **Jensen's inequality** $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$ is the chord
  inequality lifted to expectations; its two-line proof is a supporting line
  plus linearity of expectation. It yields AM--GM, the ELBO gap, and
  $D_{\mathrm{KL}} \ge 0$ with equality iff the distributions coincide --- the
  cornerstone :numref:`sec_mdl-information_theory` builds on.
* The payoff: for convex $f$, **every local minimum is global** (and stationary
  $\Rightarrow$ optimal); strict convexity gives uniqueness. Gradient descent's
  local progress becomes a global rate: $O(L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2 / k)$
  for smooth convex $f$, and a linear rate $(1 - \mu/L)^k$ under strong
  convexity --- both dimension-free.
* A short **calculus** (nonnegative sums, affine pre-composition, pointwise
  max, monotone composition) certifies hinge, $\ell_1$, logistic, and softmax
  cross-entropy without touching a Hessian; log-sum-exp is convex because its
  Hessian **is the softmax covariance**, and its conjugate is negative entropy
  --- the identity behind softmax-as-maximum-entropy and the variational duals
  of :numref:`chap_mdl-information-theory`. **Proximal operators** generalize
  projections from sets to functions; soft-thresholding is the prox of
  $\lambda|\cdot|$, and one prox per step (ISTA) optimizes kinked composite
  losses at gradient descent's rate.
* Deep networks are **non-convex by construction** (permutation symmetry makes
  the minimizer set non-convex), but the **PL condition** preserves linear
  convergence without convexity, and gradient descent's **implicit bias**
  (min-norm, max-margin) decides which of the many minima you get.

## Exercises

1. Prove that every norm ball $\{\mathbf{x} : \|\mathbf{x}\| \le r\}$ is convex
   directly from the triangle inequality, and exhibit two convex sets whose
   union is not convex. Then show the set of *strictly* positive definite
   matrices is convex, and explain why the PSD cone is its closure.
2. Certify each of the following by naming the lens or calculus rule that does
   it in one line: (a) $x \log x$ on $x > 0$; (b) the hinge loss
   $\sum_i \max(0, 1 - y_i\,\mathbf{w}^\top\mathbf{x}_i)$ in $\mathbf{w}$;
   (c) $\|A\mathbf{x} - \mathbf{b}\|_1$; (d) the *pointwise minimum* of two
   convex functions --- convex or not? Prove or give a counterexample.
3. Compute the full subdifferential $\partial f(x)$ of $f(x) = |x|$ at every
   $x \in \mathbb{R}$, and verify the optimality criterion
   $0 \in \partial f(x^\star)$ picks out $x^\star = 0$. Then show that for
   $f(\mathbf{w}) = \tfrac12\|\mathbf{w} - \mathbf{z}\|^2 + \lambda\|\mathbf{w}\|_1$
   the criterion yields the soft-thresholding solution
   $w_i^\star = \mathrm{sign}(z_i) \max(|z_i| - \lambda, 0)$.
4. Prove Jensen's finite form
   $f(\sum_i \theta_i \mathbf{x}_i) \le \sum_i \theta_i f(\mathbf{x}_i)$ by
   induction on the number of points, starting from the chord inequality.
   Use it (with $-\log$) to derive the full
   harmonic--geometric--arithmetic mean chain
   $\textrm{HM} \le \textrm{GM} \le \textrm{AM}$ for positive numbers.
5. A function is **quasiconvex** if all its sublevel sets are convex. Show
   every convex function is quasiconvex, and that $x \mapsto \sqrt{|x|}$ is
   quasiconvex but not convex --- so convex sublevel sets alone do *not* imply
   the chord inequality. Which step of the local-equals-global proof still
   works for quasiconvex functions, and which fails?
6. In the $O(1/k)$ proof, we observed that
   $\|\mathbf{x}_t - \mathbf{x}^\star\|$ is nonincreasing. Verify this
   numerically for gradient descent on
   $f(\mathbf{w}) = \tfrac12\|X\mathbf{w} - \mathbf{y}\|^2$ (reuse the data of
   the `#convexity-three-lenses` cell). The `#convexity-rate-check` cell
   measured the value-gap contraction at $\eta = 1/L$ and found exactly
   $(1 - \mu/L)^2$; predict what the contraction becomes at the more
   aggressive $\eta = 2/(\mu + L)$ of
   :numref:`sec_mdl-gradient-based-optimization`, then verify by modifying
   the cell.
7. Show that the Polyak--Łojasiewicz condition :eqref:`eq_mdl-opt-pl` implies
   every stationary point is a global minimizer, and that strong convexity
   implies PL (reproduce the proof). Then show PL does *not* imply convexity by
   verifying the PL inequality for $f(x) = x^2 + 3\sin^2 x$ analytically on a
   neighborhood of $0$ --- or numerically on $[-5, 5]$, as the
   `#convexity-pl-rate` cell did.
8. **Implicit bias, verified.** Exercise 9 of
   :numref:`sec_mdl-gradient-based-optimization` derives, on paper, that
   gradient descent on the underdetermined least-squares loss
   $\tfrac12\|X\mathbf{w} - \mathbf{y}\|^2$ started at
   $\mathbf{w}_0 = \mathbf{0}$ stays in the row space of $X$ and converges to
   the minimum-$\ell_2$-norm interpolant $X^+\mathbf{y}$. This exercise is the
   laboratory half: with a random $4 \times 10$ system, (a) compare the GD
   limit against `np.linalg.pinv`; (b) track the component of the iterates
   *orthogonal* to the row space and confirm it stays at exactly zero;
   (c) restart GD from a nonzero $\mathbf{w}_0$ with a component off the row
   space. What changes in the limit, and why?

## Discussions

Within this part, this section is load-bearing in both directions.
:numref:`sec_mdl-gradient-based-optimization` proved what smoothness alone
buys (stationarity); here convexity upgraded those rates to global guarantees,
and the PL condition showed how much of the upgrade survives without convexity.
:numref:`sec_mdl-constrained-optimization-duality` consumes nearly everything
above: convex feasible sets for projections, KKT sufficiency and Slater's
condition require convex objectives and constraints, the dual function's
concavity is the sup-of-affine rule, and its linear-constraint duals are
convex conjugates. Beyond the chapter, Jensen's inequality and the
log-sum-exp/negative-entropy pair are the analytic engine of
:numref:`sec_mdl-information_theory` and the variational bounds of
:numref:`sec_mdl-divergences-distances`, and the main book's
:numref:`sec_convexity`, :numref:`sec_softmax`, and :numref:`sec_weight_decay`
all stand on results proved here.

[Discussions](https://d2l.discourse.group/t/convexity)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §24.3]{.kicker}

The line between "gradient descent provably works" and "we hope"<br>**convex sets · three lenses · Jensen · global rates · what deep nets break**.
:::
:::

::: {.slide title="Why convexity?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
For a **convex** problem the guarantees are airtight:

- every local minimum is *global*: no saddles, no spurious basins
- a single gradient certifies *every* other point in the domain
- the rates of the last section become **global**, every-start

Convexity also names the easy pieces of deep learning (the softmax
loss, the regularizers, projections, the SVM dual), and tells you
exactly what stacking nonlinear layers forfeits.
:::

::: {.col .narrow}
::: {.d2l-note}
All code here is a few lines of plain **NumPy**: no framework, no GPU.
:::
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Convex sets]{.dtitle}

[the segment between any two points stays inside]{.dsub}
:::
:::

::: {.slide title="A set is convex when chords stay inside"}
[Convex sets]{.kicker}

A set $C$ is **convex** if every segment between two of its points never
leaves it: $\theta\mathbf{x} + (1 - \theta)\mathbf{y} \in C$ for all
$\mathbf{x}, \mathbf{y} \in C$, $\theta \in [0, 1]$.

@fig:mdl-opt-convex-vs-nonconvex-set

Left stays inside; the crescent's chord tunnels *outside*. Right: the
simplex and a half-space.
:::

::: {.slide title="New convex sets from old"}
[Convex sets]{.kicker}

::: {.cols .vc}
::: {.col}
**Intersections preserve convexity** (a point in every $C_i$ is a point
in their meet). Unions do not: $[0,1]\cup[2,3]$ has a chord that escapes.

This one fact is a factory:

- simplex = one hyperplane $\cap$ $n$ half-spaces
- PSD cone $= \bigcap_{\mathbf{z}}\{A : \mathbf{z}^\top A\mathbf{z} \ge 0\}$
- every polyhedron $\{A\mathbf{x} \preceq \mathbf{b}\}$
:::

::: {.col .narrow}
::: {.d2l-note .rule}
The catalog: half-spaces, norm balls, the simplex, the PSD cone.

Affine maps and convex hulls round it out.
:::
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Three lenses on a convex function]{.dtitle}

[chord, tangent, Hessian, and the kink repair]{.dsub}
:::
:::

::: {.slide title="The chord lens and the first-order lens"}
[Three lenses]{.kicker}

@fig:mdl-opt-chord-above-graph

. . .

Two pictures of one property: the **chord** joining two points lies
*above* the graph; the **tangent** at any point lies *below* it. A
single gradient at $\mathbf{x}$ thus certifies every $\mathbf{y}$, however far.
:::

::: {.slide title="Three lenses, one verdict"}
[Three lenses]{.kicker}

::: {.cols .vc}
::: {.col}
$$f(\theta\mathbf{x}+(1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta) f(\mathbf{y})$$

is equivalent, for smooth $f$, to the tangent under-estimator, and to

$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \textrm{everywhere.}$$

Pick the cheapest lens: the chord needs no derivatives, the first-order
one feeds every proof, the Hessian is the workhorse for smooth losses.
:::

::: {.col .narrow}
::: {.d2l-note}
**Strong convexity** adds a floor, $\nabla^2 f \succeq \mu I$: a bowl
with guaranteed curvature, and $\kappa = L/\mu$ is the condition number.
:::
:::
:::
:::

::: {.slide title="Subgradients: the tangent lens at a corner"}
[Three lenses]{.kicker}

::: {.cols .vc}
::: {.col}
$\ell_1$ and the hinge have kinks, so no gradient exists there. A
**subgradient** $\mathbf{g}$ keeps the under-estimate anyway:

$$f(\mathbf{y}) \ge f(\mathbf{x}) + \mathbf{g}^\top(\mathbf{y}-\mathbf{x}).$$

At a corner the slopes *fan out*: $\partial|x|(0) = [-1, 1]$. Optimality
survives verbatim: $\mathbf{x}^\star$ is a minimizer iff
$\mathbf{0} \in \partial f(\mathbf{x}^\star)$.
:::

::: {.col .narrow}
::: {.d2l-note .rule}
Subgradients carry Jensen, local-equals-global, and descent through
ReLU-style kinks.
:::
:::
:::
:::

::: {.slide title="All three lenses, checked numerically"}
[Three lenses]{.kicker}

On the least-squares loss in two weights: a thousand random chords, a
thousand random tangents, and the Hessian's eigenvalues.

@!convexity-three-lenses

Both worst "violations" are *negative*, and $X^\top X$ has positive
eigenvalues. Sampling can only ever refute convexity; the Hessian is the
one that proves it.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Jensen's inequality]{.dtitle}

[the chord, lifted to expectations]{.dsub}
:::
:::

::: {.slide title="Jensen: the function of the mean undershoots"}
[Jensen]{.kicker}

The chord inequality, with weights read as a probability distribution
and then as an expectation:

$$f(\mathbb{E}[X]) \;\le\; \mathbb{E}[f(X)].$$

. . .

**Proof in two lines:** take a subgradient at $\boldsymbol{\mu}=\mathbb{E}[X]$,
so $f(X) \ge f(\boldsymbol{\mu}) + \mathbf{g}^\top(X-\boldsymbol{\mu})$
pointwise; take expectations, the linear term has mean zero.

::: {.d2l-note}
The graph bends up, so spreading $X$ out can only push $\mathbb{E}[f(X)]$ above
$f(\mathbb{E}[X])$. For concave $f$ the inequality flips.
:::
:::

::: {.slide title="One inequality, three classics"}
[Jensen]{.kicker}

Apply Jensen to the right convex (or concave) function and out fall the
staples of the probabilistic chapters:

- $-\log$ is convex $\Rightarrow$ $D_{\mathrm{KL}}(p\,\|\,q) \ge 0$, with equality iff $p=q$
- $\log$ is concave $\Rightarrow$ AM $\ge$ GM
- the **ELBO gap** is precisely the slack in Jensen on a concave $\log$

@!convexity-jensen-mc

A Jensen gap of $\sqrt{e}$ vs $1$ that no sampling closes (it is geometry,
not noise), AM $\ge$ GM in every draw, and KL nonnegative throughout.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Why it matters]{.dtitle}

[local equals global, and a global rate]{.dsub}
:::
:::

::: {.slide title="Every local minimum is global"}
[The payoff]{.kicker}

@fig:mdl-opt-local-equals-global

. . .

A convex function cannot ambush you: if a better point existed anywhere,
the chord toward it would already descend *inside your neighborhood*. So
"no local improvement" means "no improvement anywhere," and stationary
$\Rightarrow$ global.
:::

::: {.slide title="The dividing line, as an experiment"}
[The payoff]{.kicker}

The *same* gradient-descent loop and step size, from 500 random starts,
on a convex bowl and on a tilted double well:

@!convexity-basins

The bowl collapses all 500 runs onto one point (spread at machine
epsilon). The double well splits them across two basins. Convexity is a
property of the landscape, and it alone decides whether the start matters.
:::

::: {.slide title="Local steps become global rates"}
[The payoff]{.kicker}

::: {.cols .vc}
::: {.col}
Chaining the descent lemma through the first-order lens upgrades
"eventually flat" to "the optimum, this fast":

$$f(\mathbf{x}_k) - f^\star \le \frac{L\,\|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}$$

and with strong convexity it sharpens to a geometric rate

$$f(\mathbf{x}_k) - f^\star \le \bigl(1 - \tfrac{\mu}{L}\bigr)^{k}\bigl(f(\mathbf{x}_0) - f^\star\bigr).$$
:::

::: {.col .narrow}
::: {.d2l-note .rule}
Both bounds are **dimension-free**: a million parameters cost no more
iterations than two. The deep reason first-order methods scale.
:::
:::
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Recognizing convexity]{.dtitle}

[a calculus, and the softmax workhorse]{.dsub}
:::
:::

::: {.slide title="A calculus of convex functions"}
[Recognizing convexity]{.kicker}

Most convexity proofs are *assembled*, not computed. Four operations
preserve convexity:

::: {.d2l-note .rule}
nonnegative sums · affine pre-composition · pointwise max ·
**monotone** convex composition
:::

. . .

So in one line each: the **hinge** is a max of affines; $\ell_1$ a sum of
such maxes; **logistic** is the convex $\log(1+e^t)$ after an affine map;
ridge-anything is strongly convex. The one missing rule, *non-monotone*
inner maps, is where deep networks exit the theory.
:::

::: {.slide title="Log-sum-exp: its Hessian is a covariance"}
[Recognizing convexity]{.kicker}

::: {.cols .vc}
::: {.col}
The function behind every softmax cross-entropy:

$$\nabla\,\mathrm{lse} = \mathrm{softmax} = \mathbf{s}, \qquad \nabla^2\mathrm{lse} = \mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top.$$

That Hessian is the **covariance of a one-hot draw** from $\mathbf{s}$, so
$\mathbf{v}^\top\nabla^2\mathrm{lse}\,\mathbf{v} = \mathrm{Var}(v_I) \ge 0$:
PSD, hence $\mathrm{lse}$ is convex.
:::

::: {.col .narrow}
::: {.d2l-note}
The one **zero** eigenvalue (direction $\mathbf{1}$) is the shift
invariance behind the stable softmax. Its conjugate is negative entropy.
:::
:::
:::
:::

::: {.slide title="Confirming the covariance, by sampling"}
[Recognizing convexity]{.kicker}

Eigenvalues of the analytic Hessian, the covariance identity by Monte
Carlo, and the predicted flat direction:

@!convexity-lse-hessian

Every eigenvalue is nonnegative down to one numerical zero, and 200k
one-hot draws reproduce the analytic Hessian: it really is a covariance
you can sample from.
:::

::: {.slide}
::: {.divider}
[06]{.dnum}

[Reality check]{.dtitle}

[deep nets are non-convex, and what survives]{.dsub}
:::
:::

::: {.slide title="Deep networks are non-convex by construction"}
[Reality check]{.kicker}

::: {.cols .vc}
::: {.col}
Take $f(a,b) = (ab-1)^2$, a two-weight linear model. Its minima form the
hyperbola $\{ab=1\}$, a **non-convex set**, but minimizer sets of convex
functions *are* convex. So $f$ cannot be convex.

Real networks inherit this: permuting hidden units leaves the function
unchanged, scattering equivalent minima everywhere. Convexity is not
bruised by deep learning, it is demolished by the architecture.
:::

::: {.col .narrow}
::: {.d2l-note .warn}
The minimizers $(1,1)$ and $(-1,-1)$ average to the origin, where
$f = 1 > 0$: not a minimum.
:::
:::
:::
:::

::: {.slide title="What survives: the PL condition"}
[Reality check]{.kicker}

The linear-rate proof never used convexity in its second half, only

$$\tfrac12\|\nabla f\|^2 \ge \mu\,(f - f^\star) \qquad \textrm{(Polyak--Łojasiewicz).}$$

PL says the gradient is small only where the value is near-optimal, so
flat spots sit only at the bottom: linear convergence with *no* convexity.

@!convexity-pl-rate

The gap contracts by a constant factor on $x^2 + 3\sin^2 x$, whose Hessian
dips to $-4$. One current account of why huge overparameterized nets train.
:::

::: {.slide title="Which minimum? Implicit bias decides"}
[Reality check]{.kicker}

When many minima exist, gradient descent does not choose arbitrarily, it
has an **implicit bias**:

- least squares from $\mathbf{w}_0 = \mathbf{0}$: the iterates stay in the
  row space, so the limit is the *minimum-norm* interpolant ($X^+\mathbf{y}$)
- separable logistic regression: the direction converges to the
  *max-margin* separator, the SVM solution never asked for

Which minimum an optimizer prefers is part of what a trained model *is*.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Convex set:** chords stay inside; **intersection** is the factory.
- **Three lenses:** chord, tangent under-estimator, PSD Hessian;
  **subgradients** extend the tangent to kinks.
- **Jensen:** $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$ gives KL $\ge 0$,
  AM $\ge$ GM, the ELBO gap.
:::

::: {.col}
- **Payoff:** local $=$ global; rates $O(1/k)$ and $(1-\mu/L)^k$,
  both dimension-free.
- **Calculus** certifies hinge, $\ell_1$, logistic, softmax; log-sum-exp's
  Hessian *is* the softmax covariance.
- **Deep nets** are non-convex, but **PL** keeps the rate and **implicit
  bias** picks the minimum.
:::
:::

::: {.d2l-note}
Convex theory is the idealization the working optimizer approximates, and
its instruments (descent lemma, PL, rates) we carry into non-convex territory.
:::
:::
