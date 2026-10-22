# Multivariable Calculus
:label:`sec_mdl-multivariable_calculus`

A deep network's loss depends on many parameters.
:numref:`sec_mdl-single_variable_calculus` described the response to a change
in one variable; here we extend the local linear approximation to several
variables. The *gradient* represents the first-order derivative and determines
directions of steepest change. Applying the multivariable chain rule efficiently
gives backpropagation. We conclude with the *Hessian*, whose second-order
information distinguishes minima, maxima, and saddle points.

## From Partial Derivatives to the Gradient

### Partial Derivatives

If we change a *single* weight
$w_1$ and freeze the rest, we are back to a function of one variable, and
:numref:`sec_mdl-single_variable_calculus` applies directly:

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_mdl-part_der`

The derivative in one variable *while holding the others fixed* is the *partial
derivative*, written $\frac{\partial}{\partial w_1}$. The new idea is what
happens when we perturb several coordinates at once.

### The Gradient
:label:`subsec_mdl-gradient`

Assume first that $L$ is differentiable at $\mathbf w$. By definition, its change
under a joint perturbation $\boldsymbol\epsilon$ equals a linear function plus an
error $o(\|\boldsymbol\epsilon\|)$. In Euclidean coordinates that linear function
is a dot product with some vector $\mathbf g$. Setting $\boldsymbol\epsilon=h\mathbf e_i$
identifies $g_i$ with $\partial L/\partial w_i$. Thus differentiability gives

$$
L(w_1+\epsilon_1, \ldots, w_N+\epsilon_N) \approx L(w_1, \ldots, w_N) + \sum_{i=1}^N \epsilon_i \frac{\partial L}{\partial w_i}.
$$

That sum on the right is a dot product. Collecting the perturbations and the
partials into vectors,

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top
\quad\textrm{and}\quad
\nabla_{\mathbf{w}} L = \left[\frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_N}\right]^\top,
$$

we obtain the multivariable analogue of the linear approximation,

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_mdl-nabla_use`

The vector $\nabla_{\mathbf{w}} L$ is the *gradient* of $L$. Equation
:eqref:`eq_mdl-nabla_use` has exactly the shape of the one-dimensional
$f(x+\epsilon) \approx f(x) + \epsilon f'(x)$, with the scalar derivative
replaced by the gradient and ordinary multiplication replaced by a dot product.
The gradient *is* the derivative in many dimensions: it is the unique vector
whose dot product with a perturbation gives the first-order change in $L$. We
treat $\nabla_{\mathbf{w}} L$ as a column vector throughout; the layout
conventions that make this choice precise are settled in
:numref:`sec_mdl-matrix-calculus-autodiff`.

This order of reasoning matters. Existence of all partial derivatives does not
itself imply the joint linear approximation. A convenient sufficient condition
is that the partials exist and are continuous near $\mathbf w$. For a
counterexample, $f(x, y) = 2xy/(x^2+y^2)$ with
$f(0,0) = 0$ has both partials equal to zero at the origin, yet equals $1$
everywhere on the diagonal $x = y \neq 0$, so no linear approximation can hold
there.

### Directional Derivatives

Reading :eqref:`eq_mdl-nabla_use` for a perturbation $\boldsymbol{\epsilon} =
h\,\mathbf{u}$ of size $h$ along a unit direction $\mathbf{u}$ shows how the
gradient determines the derivative in *any* direction:

$$
\frac{L(\mathbf{w} + h\,\mathbf{u}) - L(\mathbf{w})}{h} \;\xrightarrow{\;h\to 0\;}\; \mathbf{u}\cdot\nabla_{\mathbf{w}} L(\mathbf{w}).
$$

The rate of change of $L$ as we move along $\mathbf{u}$ is the projection of
the gradient onto $\mathbf{u}$, and that rate is the *directional derivative*.
The single vector $\nabla_{\mathbf{w}} L$ thus encodes the slope in every
direction simultaneously. The next section turns this one identity into the
geometry of gradient descent.

First, let us check that :eqref:`eq_mdl-nabla_use` really does approximate $L$.

```{.python .input #multivariable-calculus-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import math
npx.set_np()
```

```{.python .input #multivariable-calculus-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import math
```

```{.python .input #multivariable-calculus-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input #multivariable-calculus-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import math
```

Take

$$
f(x, y) = \log(e^x + e^y), \qquad \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right]^\top.
$$

At the point $(0, \log 2)$ we have $f = \log 3$ and $\nabla f = [\tfrac13, \tfrac23]^\top$,
so :eqref:`eq_mdl-nabla_use` predicts
$f(\epsilon_1, \log 2 + \epsilon_2) \approx \log 3 + \tfrac13\epsilon_1 + \tfrac23\epsilon_2$.
We compare this against the true value for a small step. The check is plain
scalar arithmetic.

```{.python .input #multivariable-calculus-higher-dimensional-differentiation}
def f(x, y):
    return math.log(math.exp(x) + math.exp(y))

def grad_f(x, y):
    s = math.exp(x) + math.exp(y)
    return [math.exp(x) / s, math.exp(y) / s]

epsilon = [0.01, -0.03]
x0, y0 = 0, math.log(2)
approx = f(x0, y0) + sum(e * g for e, g in zip(epsilon, grad_f(x0, y0)))
true_value = f(x0 + epsilon[0], y0 + epsilon[1])
print(f'approximation: {approx:.6f}, true value: {true_value:.6f}')
```

The two agree to about three decimal places, with a gap of $1.8\times10^{-4}$.
The gap is not noise, and its size is itself a lesson: a first-order model
discards terms of order $\|\boldsymbol{\epsilon}\|^2$, and here
$\|\boldsymbol{\epsilon}\|^2 = 10^{-3}$: shrink the step tenfold and the gap
shrinks a hundredfold. The discarded term has a name and a formula (the Hessian
quadratic $\tfrac12\boldsymbol{\epsilon}^\top\mathbf{H}f\,\boldsymbol{\epsilon}$),
which we develop at the end of this section.

The same example illustrates a standard verification procedure: compare a
hand-derived gradient against automatic differentiation. Here
autograd should reproduce $\nabla f(0, \log 2) = [\tfrac13, \tfrac23]^\top$.

```{.python .input #mdl-multivariable-calculus-directional-derivatives-1}
#@tab mxnet
# Check the hand-derived gradient at (0, log 2): [1/3, 2/3]
xy = np.array([0.0, math.log(2)])
xy.attach_grad()
with autograd.record():
    out = np.log(np.exp(xy).sum())
out.backward()
print(f'autograd gradient: {xy.grad}')
```

```{.python .input #mdl-multivariable-calculus-directional-derivatives-1}
#@tab pytorch
# Check the hand-derived gradient at (0, log 2): [1/3, 2/3]
xy = torch.tensor([0.0, math.log(2)], requires_grad=True)
torch.logsumexp(xy, 0).backward()
print(f'autograd gradient: {xy.grad}')
```

```{.python .input #mdl-multivariable-calculus-directional-derivatives-1}
#@tab tensorflow
# Check the hand-derived gradient at (0, log 2): [1/3, 2/3]
xy = tf.Variable([0.0, math.log(2.0)])
with tf.GradientTape() as t:
    out = tf.reduce_logsumexp(xy)
print(f'autograd gradient: {t.gradient(out, xy).numpy()}')
```

```{.python .input #mdl-multivariable-calculus-directional-derivatives-1}
#@tab jax
# Check the hand-derived gradient at (0, log 2): [1/3, 2/3]
grad = jax.grad(jax.scipy.special.logsumexp)(jnp.array([0.0, math.log(2.0)]))
print(f'autograd gradient: {grad}')
```

Up to display precision, the printout is our hand-computed
$[\tfrac13, \tfrac23]^\top$.

## The Geometry of Gradients

### Steepest Descent
:label:`subsec_mdl-steepest-descent`

Equation :eqref:`eq_mdl-nabla_use` tells us how $L$ changes in *any* direction,
so we can ask which direction makes it decrease fastest. This is the geometric
content of gradient descent, first introduced in :numref:`sec_linear_regression`:

1. Start from some initial parameters $\mathbf{w}$.
2. Find the unit direction $\mathbf{v}$ along which $L$ decreases most rapidly.
3. Take a small step that way: $\mathbf{w} \leftarrow \mathbf{w} + \eta\mathbf{v}$.
4. Repeat.

To solve step 2, write the gradient's effect on a unit direction
$\mathbf{v}$ using the geometric form of the dot product from
:numref:`sec_mdl-geometry-linear-algebraic-ops`,

$$
L(\mathbf{w} + \mathbf{v}) - L(\mathbf{w}) \approx \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta),
$$

where $\theta$ is the angle between $\mathbf{v}$ and the gradient. The direction
enters only through $\cos\theta$.

**Proposition (steepest ascent and descent).** *At any point where
$\nabla L \neq \mathbf{0}$, among all unit vectors $\mathbf{v}$ the directional
derivative $\mathbf{v}\cdot\nabla L$ is largest when $\mathbf{v}$ points along
$+\nabla L$ and smallest when it points along $-\nabla L$. Thus $+\nabla L$ is
the direction of steepest ascent and $-\nabla L$ the direction of steepest
descent.*


Here “unit” and “steepest” refer to the Euclidean norm and its inner product. With a different norm, the unit ball changes and the minimizing direction is determined by the corresponding dual norm; it need not be the negative Euclidean gradient.

**Proof.** By Cauchy–Schwarz (:eqref:`eq_mdl-cauchy-schwarz`) applied to the
unit vector $\mathbf{v}$,

$$
-\|\nabla L\| \;\le\; \mathbf{v}\cdot\nabla L \;\le\; \|\nabla L\|,
$$

and the Cauchy–Schwarz equality criterion (equality holds exactly for
collinear vectors) says each bound is attained exactly
when $\mathbf{v}$ is collinear with $\nabla L$. The upper bound $+\|\nabla L\|$
is reached by $\mathbf{v} = \nabla L/\|\nabla L\|$ and the lower bound
$-\|\nabla L\|$ by $\mathbf{v} = -\nabla L/\|\nabla L\|$. $\blacksquare$

In the $\cos\theta$ picture this is simply $\theta = 0$ versus $\theta = \pi$.
Steepest descent therefore steps along $-\nabla_{\mathbf{w}} L$, and
the informal recipe becomes the gradient-descent update

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_{\mathbf{w}} L(\mathbf{w}).
$$

First-order optimizers use gradient information, but they need not step along
the current negative gradient. Momentum combines gradients across time, while
RMSProp and Adam rescale coordinates; :numref:`chap_optimization` develops the
resulting directions.

### Gradients and Level Sets

The same expression $L(\mathbf{w} + \mathbf{v}) - L(\mathbf{w}) \approx
\|\nabla L\|\cos\theta$ also reveals what the gradient is *geometrically*, not
just which way to step. Consider moving along a *level set*, the set of points
where $L$ keeps a fixed value $c$. To first order, $L$ does not change along
such a direction, so the directional derivative vanishes.

**Proposition (gradients are normal to level sets).** *At any point, $\nabla L$
is orthogonal to every direction tangent to the level set
$\{\mathbf{x} : L(\mathbf{x}) = c\}$ through that point.*

**Proof.** A direction $\mathbf{v}$ is *tangent* to the level set at
$\mathbf{x}$ if $\mathbf{v} = \boldsymbol{\gamma}'(0)$ for some differentiable
curve $\boldsymbol{\gamma}(t)$ lying in the level set with
$\boldsymbol{\gamma}(0) = \mathbf{x}$. Along such a curve
$L(\boldsymbol{\gamma}(t)) = c$ for all $t$, so the derivative of
$t \mapsto L(\boldsymbol{\gamma}(t))$ at $t = 0$ vanishes. To first order
$\boldsymbol{\gamma}(t) = \mathbf{x} + t\mathbf{v} + o(t)$, so by
:eqref:`eq_mdl-nabla_use` that derivative is $\mathbf{v}\cdot\nabla L(\mathbf{x})$.
Hence $\mathbf{v}\cdot\nabla L = 0$; by the definition of orthogonality from
:numref:`sec_mdl-geometry-linear-algebraic-ops`, $\mathbf{v}\perp\nabla L$.
$\blacksquare$

On a contour map, the gradient crosses each contour at a right angle and points
toward increasing values. Its norm is largest where $L$ changes fastest, as
shown in :numref:`fig_mdl-cal-gradient-field`. Gradient descent moves in the
opposite direction, perpendicular to the contours.

![Contours of a scalar field $L$ with $\nabla L$ at several points. Each gradient is normal to its contour, points toward increasing values, and has greater norm where $L$ changes more rapidly.](../img/mdl-cal-gradient-field.svg)
:label:`fig_mdl-cal-gradient-field`

### Tangent Planes and Linearization

A level-set diagram represents the domain in the *base plane*, with the gradient
normal to each contour. The *graph* $z = f(\mathbf{x})$ gives a second
representation one dimension higher. Reading :eqref:`eq_mdl-nabla_use` as an
equation for the height $z$ rather than as an approximation gives the
*linearization* of $f$ at $\mathbf{x}_0$,

$$
z = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)\cdot(\mathbf{x}-\mathbf{x}_0),
$$
:eqlabel:`eq_mdl-tangent_plane`

a linear function of $\mathbf{x}$ whose graph is a plane (a hyperplane in higher
dimensions). It passes through the point $(\mathbf{x}_0, f(\mathbf{x}_0))$ and
matches every first-order slope of $f$ there, so it is the *tangent plane* to the
surface: the two-dimensional analogue of the tangent line, and the surface's
best linear approximation near $\mathbf{x}_0$.

Rewrite
:eqref:`eq_mdl-tangent_plane` as $\nabla f(\mathbf{x}_0)\cdot(\mathbf{x}-\mathbf{x}_0) - (z - f(\mathbf{x}_0)) = 0$:
this says the augmented vector $[\nabla f(\mathbf{x}_0),\, -1]^\top$ is normal,
*in graph space*, to the tangent plane; the gradient is normal to the surface
once we account for the height direction. Drop the height coordinate, projecting
that normal onto the base plane, and we recover $\nabla f$ crossing
the level curves at right angles. :numref:`fig_mdl-tangent-plane` shows the
tangent plane together with this projected gradient.

![A surface $z = f(\mathbf{x})$ with its tangent plane at a point and $\nabla f$ projected onto the base plane. The tangent plane represents the linearization in graph space; the projected gradient is normal to the corresponding level curve of $f$.](../img/mdl-cal-tangent-plane.svg)
:label:`fig_mdl-tangent-plane`

### Critical Points and the First-Order Test

Throughout this book we minimize losses numerically, because the functions that
arise in deep learning are far too complex to minimize in closed form. But the
geometry above gives a cheap, exact *necessary* condition that every minimum
must satisfy.

A necessary condition can reject a proposed minimizer of $L$ at $\mathbf{x}_0$. Applying
:eqref:`eq_mdl-nabla_use` at $\mathbf{x}_0$, if
$\nabla L(\mathbf{x}_0) \neq \mathbf{0}$, then stepping along
$-\nabla L(\mathbf{x}_0)$ strictly decreases $L$, so $\mathbf{x}_0$ cannot be a
minimum. Contrapositively, **a minimum forces $\nabla L(\mathbf{x}_0) =
\mathbf{0}$.** Points where the gradient vanishes are called *critical points*.

This is occasionally enough to optimize by hand. For example,

$$
f(x) = 3x^4 - 4x^3 - 12x^2, \qquad f'(x) = 12x(x-2)(x+1),
$$

has critical points $x = -1, 0, 2$ with values $-5, 0, -32$, so the minimum is
at $x = 2$, as the plot confirms.

```{.python .input #multivariable-calculus-a-note-on-mathematical-optimization}
import numpy as onp
x = onp.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

But not every critical point is an extremum: $x = 0$ above is a local maximum,
and in higher dimensions a critical point can be a saddle. Telling these apart
needs *second*-order information, the Hessian we develop below.

### Optimizing on a Constraint

The first-order test answers "where can an unconstrained minimum be?" One more
turn of the same geometry answers the constrained question that pervades machine
learning: minimize a loss *subject to* keeping some quantity fixed, say
$g(\mathbf{x}) = c$. Now we are no longer free to step in any direction: the
admissible moves are exactly those tangent to the constraint surface
$\{g = c\}$. At a constrained optimum $\mathbf{x}^\star$, *no* admissible
direction can lower $f$ to first order: otherwise we would slide along the
constraint and improve. By :eqref:`eq_mdl-nabla_use` that means
$\nabla f(\mathbf{x}^\star)$ has zero component along every direction tangent to
$\{g = c\}$; it is orthogonal to that surface. But we proved above that
$\nabla g$ is *also* orthogonal to $\{g = c\}$. One hypothesis makes the
geometry rigorous: provided $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$, the
standard *constraint qualification*, the implicit function theorem (stated in
:numref:`subsec_mdl-implicit-diff`; see :cite:`Rudin.1976`) guarantees that
$\{g = c\}$ really is a smooth surface of codimension one near
$\mathbf{x}^\star$ (one equation removes one degree of freedom, leaving an
$(n{-}1)$-dimensional surface) with a single normal direction; it also supplies
the fact, used tacitly above, that every tangent direction is realized by a
curve lying in $\{g = c\}$, so "sliding along the constraint" is a legitimate
move. Two vectors normal to the same surface must be parallel, so at the
constrained optimum

$$
\nabla f(\mathbf{x}^\star) = \lambda\,\nabla g(\mathbf{x}^\star)
$$
:eqlabel:`eq_mdl-lagrange-condition`

for some scalar $\lambda$, the *Lagrange multiplier*. This single picture is
the first-order condition for constrained optimization and a basis for the KKT
conditions and duality. At the solution, a contour of $f$ is tangent to the
constraint surface and their gradients align
(:numref:`fig_mdl-lagrange-tangency`). :numref:`sec_mdl-constrained-optimization-duality`
develops this condition further.

![Lagrange multipliers as tangency. At the constrained optimum the level set of $f$ is tangent to the constraint curve $g = c$ and the two gradients align, $\nabla f = \lambda \nabla g$. At a non-optimal feasible point the gradients disagree, so $\nabla f$ keeps a component along the constraint and sliding along it still improves $f$.](../img/mdl-cal-lagrange-tangency.svg)
:label:`fig_mdl-lagrange-tangency`

## The Multivariate Chain Rule

Neural networks are deep compositions of simple functions, so computing
gradients means differentiating compositions. Consider four inputs $w, x, y, z$
flowing through intermediate quantities to a scalar output:

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_mdl-multi_func_def`

The dependencies form a graph (:numref:`fig_mdl-chain-1`): each node is a value,
each edge a direct functional dependence.

![The function relations of :eqref:`eq_mdl-multi_func_def`, drawn as a graph where nodes are values and edges show direct functional dependence.](../img/mdl-cal-chain-net1.svg)
:label:`fig_mdl-chain-1`

Direct substitution produces long expressions with repeated subexpressions;
computing another partial derivative repeats much of the same work. The chain
rule organizes these shared computations.

### The Rule as a Sum Over Paths

Take the simplest composite step, $f(u(a,b), v(a,b))$, and perturb $a$ by a
small $\epsilon$. Each intermediate moves by its partial,
$u \to u + \epsilon\frac{\partial u}{\partial a}$ and
$v \to v + \epsilon\frac{\partial v}{\partial a}$, and feeding those into the
first-order expansion of $f$ gives

$$
f\!\left(u + \epsilon\tfrac{\partial u}{\partial a},\, v + \epsilon\tfrac{\partial v}{\partial a}\right)
\approx f(u, v) + \epsilon\left[\frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}\right].
$$

Reading off the coefficient of $\epsilon$ gives the multivariate chain rule,

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

In words, there are two *pathways* by which $a$
influences $f$: $a \to u \to f$ and $a \to v \to f$. Each path contributes the
*product* of the derivatives along its edges, and the total derivative is the
*sum* over paths.

In general, to differentiate the output with respect to an input we **sum, over
every directed path from that input to the output, the product of the edge
derivatives along the path.**

To see the rule at work on a graph that is not a tree, consider a
*different* composition (its own dependency graph, unrelated to
:eqref:`eq_mdl-multi_func_def`) in which one intermediate feeds the output both
directly and through a later node:

$$\begin{aligned}u(x, y) & = x + y, \qquad v(x, y) = x - y, \\a(u) & = u^2, \qquad b(v) = v^2, \\f(a, u, b) & = a + u + b.\end{aligned}$$
:eqlabel:`eq_mdl-multi_func_def_2`

These relations are exactly the edges drawn in :numref:`fig_mdl-chain-2`:
$u$ and $v$ each depend on the input $y$; $a$ depends on $u$ and $b$ on $v$; and
the output $f$ depends on $a$, on $b$, and *also directly on $u$*: the skip-like
edge that gives the middle path below. Tracing every directed route from $y$ to
$f$ (namely $y\to u\to a\to f$, the direct $y\to u\to f$, and
$y\to v\to b\to f$), the rule gives

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

![A second, distinct dependency graph, :eqref:`eq_mdl-multi_func_def_2`: the input $y$ reaches the output $f$ by three directed paths ($y\to u\to a\to f$, the direct skip $y\to u\to f$, and $y\to v\to b\to f$), and the chain rule sums their contributions.](../img/mdl-cal-chain-net2.svg)
:label:`fig_mdl-chain-2`

Here every edge derivative is elementary
($\partial f/\partial a = \partial f/\partial u = \partial f/\partial b = 1$,
$\partial a/\partial u = 2u$, $\partial u/\partial y = 1$, and so on), so the
sum-over-paths value is checkable by hand: it collapses to
$2u + 1 - 2v = 2(x+y) + 1 - 2(x-y) = 1 + 4y$. This "sum over paths" view is
exactly how gradients flow through a network, and
it explains why architectural choices that open or close paths, such as the
gates of an LSTM (:numref:`sec_lstm`) or the skip connections of a residual
block (:numref:`sec_resnet`), shape learning by controlling that gradient flow.

### The Backpropagation Algorithm

Return to :eqref:`eq_mdl-multi_func_def` and ask for $\frac{\partial f}{\partial
w}$. Expanding the chain rule directly pushes $w$ forward through the
graph,

$$
\frac{\partial f}{\partial w} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \qquad
\frac{\partial u}{\partial w} = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \quad \ldots
$$

and the single-step partials are all elementary,

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

In code this is a tidy forward sweep through the graph.

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-1}
# Compute the value of the function from inputs to outputs
w, x, y, z = 1, 1, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

This computes $\frac{\partial f}{\partial w}$ but does not reuse the calculation
for $\frac{\partial f}{\partial x}$. Deep learning instead requires the
derivative of one loss with respect to every parameter. We therefore keep
$\partial f$ in each numerator and traverse the graph backward from the
output:

$$
\begin{aligned}
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \qquad
\frac{\partial f}{\partial b} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}, \\
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \qquad (\textrm{and likewise for } x, y, z).
\end{aligned}
$$

Computing $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}$ once,
then $\frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}$, then all
four input derivatives, *reuses* every intermediate. One backward sweep yields
the gradient with respect to all inputs at once.

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-2}
# Compute the value of the function from inputs to outputs
w, x, y, z = 1, 1, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

Two sanity checks are visible in the printout. The derivatives with respect to
$w$ and $x$ agree, as do those with respect to $y$ and $z$, as they must,
since $f$ reaches its inputs only through the sums $w+x$ and $y+z$. And the two
pairs differ in both magnitude and sign, so the sweep is genuinely telling the
four paths through the graph apart.

Computing derivatives *from $f$ back toward the inputs*, rather than forward from
the inputs, is what gives the algorithm its name: *backpropagation*
:cite:`Rumelhart.Hinton.Williams.ea.1988`. It is two
passes: a *forward pass* that evaluates the function and records the single-step
partials, and a *backward pass* that accumulates $\frac{\partial f}{\partial
\cdot}$ from output to input. This is exactly how the deep learning library
obtains the gradient of the loss with respect to *every* weight in a
single sweep, and it is what the one-line autograd call in the cell below runs
under the hood.

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab mxnet
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(1), np.array(1), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab pytorch
# Initialize as tensors that require gradients
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([1.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab tensorflow
# Initialize as Variables, which the tape tracks automatically
w = tf.Variable(tf.constant([1.]))
x = tf.Variable(tf.constant([1.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape() as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# One backward sweep yields the gradient for all four inputs at once
w_grad, x_grad, y_grad, z_grad = t.gradient(f, [w, x, y, z])

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad.numpy()}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad.numpy()}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad.numpy()}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad.numpy()}')
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab jax
# Define the function to differentiate
def f_comp(w, x, y, z):
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    return ((u + v)**2).squeeze()

w, x, y, z = jnp.array([1.]), jnp.array([1.]), jnp.array([-2.]), jnp.array([1.])

# Compute gradients with respect to all four arguments
grad_f = jax.grad(f_comp, argnums=(0, 1, 2, 3))
w_grad, x_grad, y_grad, z_grad = grad_f(w, x, y, z)

print(f'df/dw at {w}, {x}, {y}, {z} is {w_grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x_grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y_grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z_grad}')
```

The library's answer matches our hand-computed backward pass.
:numref:`sec_mdl-matrix-calculus-autodiff` explains why backprop is
reverse-mode automatic differentiation, a chain of vector–Jacobian products,
and when to prefer it over forward mode.

## Second-Order Structure: the Hessian

The gradient is a first-order, linear approximation; to know whether a critical
point is a minimum we need the *curvature*, which is determined by the second
derivatives. A function of $n$ variables has $n^2$ second partials,

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial}{\partial x_j} f\right),
$$

collected into the *Hessian* matrix

$$\mathbf{H}_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1 \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n \partial x_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_mdl-hess_def`

These $n^2$ entries are not independent: the Hessian is symmetric, a result
known as the Clairaut–Schwarz theorem :cite:`Rudin.1976`.

**Proposition (symmetry of the Hessian; Clairaut–Schwarz).** *If the mixed partials of
$f$ exist and are continuous, then for all $i, j$,*

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i},
\qquad\textrm{equivalently}\qquad \mathbf{H}_f = \mathbf{H}_f^\top.
$$

**Proof.** Both mixed partials are limits of the *same* quantity, the
symmetric second difference

$$
\Delta_h = \frac{f(\mathbf{x}+h\mathbf{e}_i+h\mathbf{e}_j) - f(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x}+h\mathbf{e}_j) + f(\mathbf{x})}{h^2},
$$
:eqlabel:`eq_mdl-second-difference`

which is symmetric in $i$ and $j$ by inspection. Indeed, write
$g(t) = f(\mathbf{x}+t\mathbf{e}_i+h\mathbf{e}_j) - f(\mathbf{x}+t\mathbf{e}_i)$,
so that $\Delta_h = \bigl(g(h)-g(0)\bigr)/h^2$; two applications of the mean
value theorem (first along $x_i$, then along $x_j$) give
$\Delta_h = \frac{\partial^2 f}{\partial x_j \partial x_i}(\boldsymbol{\xi}_h)$
at some point $\boldsymbol{\xi}_h$ within distance $2h$ of $\mathbf{x}$.
Swapping the roles of $i$ and $j$ leaves $\Delta_h$ unchanged and gives
$\Delta_h = \frac{\partial^2 f}{\partial x_i \partial x_j}(\boldsymbol{\xi}'_h)$
at another nearby point. Let $h \to 0$: by continuity the two expressions
converge to the two mixed partials at $\mathbf{x}$, and since they are equal
for every $h$, the limits agree. $\blacksquare$

Symmetry allows the spectral theorem and positive-definiteness results from
:numref:`sec_mdl-eigendecompositions` apply, which is exactly what the
second-derivative test will use.

### The Second-Order Taylor Approximation

Just as the gradient gives the best linear fit, the Hessian gives the best
*quadratic* fit. To see how the coefficients enter, read them off an exact
quadratic. Let $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$.
Evaluating the value, gradient, and Hessian :eqref:`eq_mdl-hess_def` at the
origin gives

$$
f(0,0) = a, \qquad
\nabla f (0,0) = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \qquad
\mathbf{H} f (0,0) = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
$$

and these recover the polynomial exactly:
$f(\mathbf{x}) = f(0) + \nabla f(0) \cdot \mathbf{x} + \tfrac12\mathbf{x}^\top \mathbf{H} f(0)\, \mathbf{x}$.

For a general $f$, twice differentiable near a base point $\mathbf{x}_0$, the
same assembly is no longer exact, but restricting $f$ to a line shows exactly
how close it comes. Fix a displacement $\boldsymbol{\delta} = \mathbf{x} -
\mathbf{x}_0$ and let $g(t) = f(\mathbf{x}_0 + t\boldsymbol{\delta})$, a
function of the single variable $t$. By the multivariate chain rule,
$g'(t) = \nabla f(\mathbf{x}_0 + t\boldsymbol{\delta})\cdot\boldsymbol{\delta}$
and $g''(t) = \boldsymbol{\delta}^\top \mathbf{H} f(\mathbf{x}_0 +
t\boldsymbol{\delta})\,\boldsymbol{\delta}$, so $g'(0)$ and $g''(0)$ are
exactly the gradient term and the Hessian quadratic form at $\mathbf{x}_0$.
Now apply the single-variable Lagrange remainder :eqref:`eq_mdl-lagrange`,
established in the previous section, to $g$ on $[0, 1]$ with $n = 1$: there is
a $\tau \in (0, 1)$ with

$$
f(\mathbf{x}) = g(1) = g(0) + g'(0) + \tfrac{1}{2}g''(\tau) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)\cdot\boldsymbol{\delta} + \tfrac{1}{2}\boldsymbol{\delta}^\top \mathbf{H} f(\mathbf{x}_0 + \tau\boldsymbol{\delta})\,\boldsymbol{\delta}.
$$

This identity is exact; its only blemish is that the Hessian is evaluated at
an intermediate point $\mathbf{x}_0 + \tau\boldsymbol{\delta}$ rather than at
$\mathbf{x}_0$. When the second partials are continuous, moving the evaluation
point back to $\mathbf{x}_0$ changes the quadratic term by
$o(\|\boldsymbol{\delta}\|^2)$, giving the *second-order Taylor approximation*

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$
:eqlabel:`eq_mdl-second_taylor`

This is the best-approximating quadratic to $f$ near $\mathbf{x}_0$ in a
precise sense: its error vanishes faster than $\|\mathbf{x}-\mathbf{x}_0\|^2$,
and it is the only quadratic with that property. For smoother $f$ (three
continuous derivatives) the error is at most *third order* in the step. To see
it in numbers, take $f(x, y) = xe^{-x^2-y^2}$. Assembling its value,
gradient, and Hessian at $\mathbf{x}_0 = [-1, 0]^\top$ via
:eqref:`eq_mdl-second_taylor` gives the approximating quadratic
$q(x, y) = e^{-1}\bigl(-1 - (x+1) + (x+1)^2 + y^2\bigr)$.
:numref:`fig_mdl-taylor-quadratic` plots the surface against this quadratic;
near $[-1, 0]^\top$ they are nearly indistinguishable and separate only farther from the base point.

![The surface $z = xe^{-x^2-y^2}$ (blue) and its second-order Taylor quadratic at the base point $(-1, 0)^\top$ (orange). The two agree in value, slope, and curvature there, so they are nearly indistinguishable in a neighborhood of the base point and separate only farther out.](../img/mdl-cal-taylor-quadratic.svg)
:label:`fig_mdl-taylor-quadratic`

The figure asserts agreement; we can check it numerically, exactly as
we checked the gradient above. Evaluating $f$ and its quadratic $q$ at a few
points stepping away from the base point, the gap should stay tiny nearby and
grow with distance.

```{.python .input #multivariable-calculus-hessians}
def f(x, y):
    return x * math.exp(-x**2 - y**2)

def quad(x, y):  # 2nd-order Taylor of f at the base point (-1, 0)
    return math.exp(-1) * (-1 - (x + 1) + (x + 1)**2 + y**2)

for d in [0.0, 0.05, 0.1, 0.3]:  # step d in each coordinate from (-1, 0)
    x, y = -1 + d, d
    print(f'step {d:.2f}: f = {f(x, y):.6f}, '
          f'quadratic = {quad(x, y):.6f}, gap = {abs(f(x, y) - quad(x, y)):.6f}')
```

The gap vanishes at the base point and is third order in the step (doubling the
step roughly multiplies it by eight), the third-order behavior promised above.
Iterating this idea, repeatedly fitting the local quadratic and jumping to *its*
minimum, is Newton's method, which :numref:`subsec_mdl-newton` introduced in one
variable; :numref:`sec_gd` develops it as a practical optimizer.

### The Second-Derivative Test

At a critical point of a twice continuously differentiable function, a definite
or indefinite Hessian classifies local behavior; a semidefinite Hessian does not. At a
critical point $\mathbf{x}_0$ the gradient term in :eqref:`eq_mdl-second_taylor`
vanishes, so the local picture is purely quadratic,

$$
f(\mathbf{x}) - f(\mathbf{x}_0) \approx \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0)(\mathbf{x}-\mathbf{x}_0).
$$

Stepping a unit direction $\mathbf{v}$ away from $\mathbf{x}_0$ makes the
right-hand side $\tfrac12\mathbf{v}^\top\mathbf{H}\mathbf{v}$: the scalar
$\mathbf{v}^\top\mathbf{H}\mathbf{v}$ is the *second directional derivative* of
$f$ along $\mathbf{v}$, the second-order analogue of $\mathbf{v}\cdot\nabla f$.
At a critical point, where the slope term is gone, it is the curvature of $f$
along $\mathbf{v}$. Whether $f$ goes up or down as we leave $\mathbf{x}_0$
is governed entirely by the sign of this quadratic form, that is, by the
*definiteness* of the symmetric matrix $\mathbf{H}$. The classification is read straight off the
eigenvalues of $\mathbf{H}$ via the PSD/PD criterion of
:numref:`subsec_mdl-psd`:

* $\mathbf{H} \succ 0$ (all eigenvalues positive): $f$ curves *upward* in every
  direction. Quantitatively, the quadratic term is at least
  $\tfrac12\lambda_{\min}\|\mathbf{v}\|^2$ for a step $\mathbf{v}$, where
  $\lambda_{\min} > 0$ is the smallest eigenvalue, and this dominates the
  approximation's $o(\|\mathbf{v}\|^2)$ error for all small $\mathbf{v}$, so
  $\mathbf{x}_0$ is a strict local **minimum**.
* $\mathbf{H} \prec 0$ (all eigenvalues negative): $f$ curves downward
  everywhere, so $\mathbf{x}_0$ is a local **maximum**.
* $\mathbf{H}$ *indefinite* (eigenvalues of both signs): $f$ rises along some
  directions and falls along others: a **saddle**.
* $\mathbf{H} \succeq 0$ with a zero eigenvalue (semidefinite): the quadratic is
  flat along that eigenvector and second order is *inconclusive*; the behavior
  is decided by higher-order terms.

These local shapes (the bowl, the saddle, and the flat-bottomed trough of the
semidefinite case) are exactly the quadratic-form surfaces drawn in
:numref:`fig_mdl-la-psd`.

This is the multivariable generalization of the single-variable test $f'' > 0$:
there, curvature is a single number; here it is a matrix, and "positive
curvature" becomes "positive definite." Identity :eqref:`eq_mdl-quadform` makes
the upward-curving picture precise, writing the quadratic form as a weighted sum
of squares over the eigenvector directions with the eigenvalues as weights.

The test is one we can run. The surface of :numref:`fig_mdl-taylor-quadratic`,
$f(x, y) = xe^{-x^2-y^2}$, has exactly two critical points: setting
$\frac{\partial f}{\partial x} = (1 - 2x^2)\,e^{-x^2-y^2}$ and
$\frac{\partial f}{\partial y} = -2xy\,e^{-x^2-y^2}$ to zero forces $y = 0$ and
$x = \pm 1/\sqrt{2}$. The next cell assembles each Hessian from symmetric second
differences (the same quantity $\Delta_h$ that proved the Clairaut–Schwarz
theorem), reads
off the eigenvalues of a $2 \times 2$ symmetric matrix in closed form, and
prints the classification.

```{.python .input #mdl-multivariable-calculus-the-second-derivative-test}
# Classify the critical points (+-1/sqrt(2), 0) of f(x, y) = x e^(-x^2-y^2)
def hessian(f, x, y, h=1e-4):  # symmetric second differences
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
    fxy = (f(x + h, y + h) - f(x + h, y - h)
           - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
    return fxx, fxy, fyy

for x0 in [-1 / math.sqrt(2), 1 / math.sqrt(2)]:
    fxx, fxy, fyy = hessian(f, x0, 0.0)
    mid, r = (fxx + fyy) / 2, math.sqrt(((fxx - fyy) / 2)**2 + fxy**2)
    lo, hi = mid - r, mid + r  # eigenvalues of [[fxx, fxy], [fxy, fyy]]
    kind = 'minimum' if lo > 0 else 'maximum' if hi < 0 else 'saddle'
    print(f'at ({x0:+.4f}, 0): eigenvalues {lo:+.3f}, {hi:+.3f} -> {kind}')
```

At $(-1/\sqrt{2}, 0)$ both eigenvalues are positive ($\mathbf{H} \succ 0$, a
strict local minimum) and at $(+1/\sqrt{2}, 0)$ both are negative, a local
maximum; a saddle would show one eigenvalue of each sign, as in
exercise :exref:`ex_mdl-multivariable-calculus-classify`. This five-line loop
is the second-derivative test as a program: differentiate twice (here
numerically; autograd Hessians appear in
:numref:`sec_mdl-matrix-calculus-autodiff`), extract eigenvalues, read the signs.

The eigenvalue picture also explains why, in high dimension, saddles are the
rule rather than the exception: a minimum requires *all* $n$ eigenvalues to be
positive at once, and if the signs at a random critical point behaved like
independent coin flips, that would happen with probability $2^{-n}$. The
coin-flip heuristic is not exact, but its conclusion survives a precise
analysis: for random Gaussian landscapes the
statistics of critical points can be computed exactly and minima are
exponentially outnumbered by saddles :cite:`Bray.Dean.2007`, and among the
critical points that Newton's method finds on the loss of a small neural
network, those of high loss are saddles too
:cite:`Dauphin.Pascanu.Gulcehre.ea.2014`, one reason gradient
methods fare better in practice than the old fear of "getting stuck in a bad
local minimum" suggests.

The gradient, Hessian, and multivariable chain rule extend the one-variable
methods to scalar functions of several variables. Neural network layers also
map vectors to vectors and use matrix parameters. Their derivative is a matrix
of partial derivatives, the *Jacobian*, with gradients and Hessians as special
cases. The Mean Value Theorem of
:numref:`sec_mdl-mvt` does *not* survive the passage to vector-valued maps;
only an inequality remains, the mean value inequality :cite:`Rudin.1976`.
:numref:`sec_mdl-matrix-calculus-autodiff` develops the Jacobian machinery, the
layout conventions, and how it all yields backpropagation as reverse-mode
automatic differentiation.

## Summary

* The *gradient* $\nabla_{\mathbf{w}} L$ is the derivative in many dimensions: it
  gives the first-order change $L(\mathbf{w}+\boldsymbol{\epsilon}) \approx
  L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot\nabla_{\mathbf{w}} L$, and its dot
  product with a unit direction is the rate of change along that direction.
* By Cauchy–Schwarz, $+\nabla L$ is the direction of steepest ascent and
  $-\nabla L$ of steepest descent, and the gradient is everywhere orthogonal to
  the level sets of $L$: the geometry behind gradient descent.
* The multivariate *chain rule* sums, over every path from an input to the
  output, the product of edge derivatives. Organizing it from the output
  backward reuses every intermediate, giving *backpropagation*: a forward pass
  followed by a backward pass that yields the gradient with respect to all
  parameters at once.
* The symmetric *Hessian* supplies the second-order Taylor approximation; at a
  critical point its definiteness, read from its eigenvalues, distinguishes a
  minimum, a maximum, and a saddle.

## Exercises

1. **Gradient of log-sum-exp.** Let
   $L(\mathbf{x}) = \log \sum_{i=1}^n e^{x_i}$ for $\mathbf{x} \in \mathbb{R}^n$.
    1. Compute $\nabla L$ and show that its components are positive and sum
       to $1$.
    1. Show that $L(\mathbf{x} + t\mathbf{1}) = L(\mathbf{x}) + t$ for every
       scalar $t$, where $\mathbf{1}$ is the all-ones vector. Derive the
       sum-to-one property from this identity and :eqref:`eq_mdl-nabla_use`,
       without differentiating $L$ coordinate by coordinate.
    1. Between which bounds does the rate of steepest ascent
       $\|\nabla L(\mathbf{x})\|$ lie? At which inputs are the bounds attained
       or approached?

1. **Directional derivatives without a gradient.** Let
   $f(x, y) = x^2 y / (x^2 + y^2)$ for $(x, y) \neq (0, 0)$ and
   $f(0, 0) = 0$.
    1. Show that $f$ is continuous at the origin.
    1. Compute the two partial derivatives of $f$ at the origin and the
       directional derivative there along every unit vector
       $\mathbf{u} = [\cos\theta, \sin\theta]^\top$.
    1. Show that no vector $\mathbf{g}$ gives these directional derivatives
       as $\mathbf{u}\cdot\mathbf{g}$, so that $f$ is not differentiable at
       the origin. Which hypothesis of the sufficient condition for
       differentiability in :numref:`subsec_mdl-gradient` fails? What does
       this function show that the counterexample $2xy/(x^2+y^2)$ there does
       not?

1. **Gradients of an elongated bowl.** Let $f(x, y) = x^2 + \kappa y^2$ with
   $\kappa \ge 1$. Its minimizer is the origin, and $\kappa$ measures how much
   more strongly $f$ curves along $y$ than along $x$.
    1. Find a direction tangent to the level curve of $f$ through $(1, 1)$
       and check that it is orthogonal to $\nabla f(1, 1)$.
    1. Compute the angle between $-\nabla f(1, 1)$ and the direction
       $-[1, 1]^\top$ from $(1, 1)$ to the minimizer. At which points
       $(x, y) \neq (0, 0)$ does the negative gradient point exactly at the
       minimizer?
    1. Find the smallest value of the cosine of this angle over all points
       $(x, y) \neq (0, 0)$, and the points where it is attained. What does
       the result imply for a gradient-descent step on such a loss as
       $\kappa$ grows?

1. **Lagrange multipliers.** The Lagrange condition
   :eqref:`eq_mdl-lagrange-condition` holds at a constrained optimum provided
   $\nabla g \neq \mathbf{0}$ there.
    1. Check that $\nabla g \neq \mathbf{0}$ everywhere for
       $g(x, y) = x + y$. Then use the Lagrange condition to maximize
       $f(x, y) = xy$ subject to $x + y = 1$, and find the multiplier
       $\lambda$.
    1. Replace the constraint by $x + y = 1 + \delta$. Show that the maximum
       value changes by $\lambda\delta$ to first order in $\delta$, where
       $\lambda$ is the multiplier of the unperturbed problem. In what sense
       does the multiplier price the constraint?
    1. Minimize $f(x, y) = x$ subject to $y^2 - x^3 = 0$. Show that the
       minimizer is the origin and that no $\lambda$ satisfies
       :eqref:`eq_mdl-lagrange-condition` there. What does the constraint set
       look like near the origin, and which step of the argument behind
       :eqref:`eq_mdl-lagrange-condition` breaks down there?

1. [code] **Backward sweep for two tanh units.** A network with one input
   $x$, two hidden units, and one output computes, for a target $t$,

   $$h_i = \tanh(w_i x + b_i) \quad (i = 1, 2), \qquad f = v_1 h_1 + v_2 h_2, \qquad \ell = \tfrac12 (f - t)^2.$$

    1. Draw the dependency graph of $\ell$. In plain Python, write a forward
       sweep that stores every intermediate value and a backward sweep that
       returns the partial derivatives of $\ell$ with respect to all six
       parameters $w_1, b_1, w_2, b_2, v_1, v_2$. Check them against
       automatic differentiation at $x = 1.5$, $t = 0.5$, and
       $(w_1, b_1, w_2, b_2, v_1, v_2) = (0.5, 0, -1, 0.2, 1, -0.5)$.
       Which intermediate derivative does every one of the six partials
       reuse?
    1. Set $w_1 = 5$ and $x = 2$, keeping the other values. Compute
       $\partial \ell / \partial w_1$ with the plain-Python backward sweep
       and explain its size from the product of edge derivatives along its
       path. Compare it with the value that automatic differentiation returns
       in single precision, and explain the difference. What does this imply
       for training a unit whose pre-activation is large?
    1. Set $w_1 = w_2$, $b_1 = b_2$, and $v_1 = v_2$. Show from the backward
       sweep that both units receive identical gradients, so that gradient
       descent keeps them identical at every step. What does this imply for
       the initialization of hidden units?

1. [code] **Order of the Taylor gap.**
    1. Let $L(x, y) = \log(e^x + e^y)$ and
       $\boldsymbol{\epsilon} = d\,[1, -3]^\top$. For
       $d = 10^{-1}, 10^{-2}, \ldots, 10^{-5}$, compute the gap between
       $L(\epsilon_1, \log 2 + \epsilon_2)$ and its linear approximation
       :eqref:`eq_mdl-nabla_use` at $(0, \log 2)$. Compute the Hessian of $L$
       at $(0, \log 2)$ by hand and use the quadratic term of
       :eqref:`eq_mdl-second_taylor` to predict the gap as a function of
       $d$. How closely does the prediction match the measured gaps?
    1. Let $f(x, y) = x e^{-x^2 - y^2}$, and let $q$ be its quadratic
       approximation at $(-1, 0)$, implemented as `quad`. Compute the gap
       $|f - q|$ at $(-1, 0) + d\,\mathbf{u}$ for $\mathbf{u} = [1, 1]^\top$,
       $[1, 0]^\top$, and $[0, 1]^\top$ with $d$ from $10^{-1}$ to $10^{-3}$,
       and plot the three gaps against $d$ on log-log axes. Read the order of
       each gap off its slope.
    1. Along one of the three directions the gap shrinks faster than third
       order. Explain this from a symmetry of $f$. Is it the only direction
       with this property?

1. **Classifying critical points.** Let $f(x, y) = x^3 - 3x + y^2$. Show that
   the critical points are $(\pm 1, 0)$. Determine whether each is a minimum,
   a maximum, or a saddle by examining $f$ along the lines $y = 0$ and
   $x = \pm 1$, and confirm the classification with the eigenvalues of the
   Hessian.
   :exlabel:`ex_mdl-multivariable-calculus-classify`

1. **Flat directions.**
    1. Give three functions of $(x, y)$ that have a critical point at the
       origin with Hessian $\operatorname{diag}(2, 0)$ there: one with a
       strict local minimum, one with a local minimum that is not strict, and
       one without a local minimum.
    1. The function $f(x, y) = (x - y^2)(x - 2y^2)$ also has Hessian
       $\operatorname{diag}(2, 0)$ at the origin. Show that its restriction
       to every line through the origin has a strict local minimum there.
       Sketch the region of the plane where $f < 0$, and show that the origin
       is nevertheless not a local minimum of $f$. What does this imply for
       testing a critical point by probing the loss along a few directions?
    1. The *monkey saddle* $f(x, y) = x^3 - 3xy^2$ has a critical point at
       the origin. Compute its Hessian there, write $f$ in polar coordinates,
       and find the directions along which $f$ decreases. How does
       $\|\nabla f\|$ scale with the distance to the origin, compared with the
       saddle $x^2 - y^2$? What does this imply for gradient descent started
       near each of the two saddles?

1. **Steepest descent in other norms.** ● The steepest-descent proposition
   of :numref:`subsec_mdl-steepest-descent` measures the length of a step in
   the Euclidean norm.
   Measure it instead in a norm $\|\cdot\|$ and seek the step $\mathbf{v}$
   with $\|\mathbf{v}\| \le 1$ that minimizes the first-order change
   $\mathbf{v}\cdot\mathbf{g}$, where $\mathbf{g} = \nabla L(\mathbf{w})$ is
   nonzero.
    1. Solve this problem for $\|\mathbf{v}\|_\infty = \max_i |v_i|$ and for
       $\|\mathbf{v}\|_1 = \sum_i |v_i|$. In each case give the minimizing
       step, the condition on $\mathbf{g}$ under which it is unique, and the
       minimal value written as a norm of $\mathbf{g}$. Which of the two steps
       is the update of Adam with its moving averages switched off
       :cite:`Bernstein.Newhouse.2024`?
    1. For a weight matrix, the first-order change under a step
       $\boldsymbol{\Delta}$ is the Frobenius inner product
       $\langle \mathbf{G}, \boldsymbol{\Delta} \rangle = \operatorname{tr}(\mathbf{G}^\top \boldsymbol{\Delta})$
       with the gradient matrix $\mathbf{G}$. Let
       $\mathbf{G} = \sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$
       be a reduced singular value decomposition
       (:numref:`sec_mdl-svd-low-rank`). Show that over all steps with
       spectral norm $\|\boldsymbol{\Delta}\|_2 \le 1$ the minimum of
       $\langle \mathbf{G}, \boldsymbol{\Delta} \rangle$ is attained at
       $\boldsymbol{\Delta} = -\mathbf{U}\mathbf{V}^\top$, the direction along
       which Muon :cite:`Jordan.Jin.Boza.ea.2024` steps, and compute the
       minimal value. (Write
       $\langle \mathbf{G}, \boldsymbol{\Delta} \rangle = \sum_i \sigma_i \mathbf{u}_i^\top \boldsymbol{\Delta} \mathbf{v}_i$
       and bound each term.)

1. **Paths through a residual stack.** ● In a one-dimensional model of the
   residual networks of :numref:`sec_resnet`, scalar blocks compute
   $x_{l+1} = x_l + g_l(x_l)$ for $l = 0, \ldots, m-1$ with differentiable
   functions $g_l$.
    1. Use the chain rule to write $\partial x_m / \partial x_0$ as a product
       of $m$ factors.
    1. Expand the product and match each term to a directed path from $x_0$
       to $x_m$ in the dependency graph. How many paths are there, and how
       many of them pass through exactly $k$ of the functions $g_l$? Compare
       the number of paths with the number of multiplications that a
       backward sweep needs to compute $\partial x_m / \partial x_0$.
    1. For a trained network of $m = 54$ residual blocks,
       :citet:`Veit.Wilber.Belongie.2016` found that almost all of the
       gradient flows along the paths through $5$ to $17$ blocks. What
       fraction of all paths are these? Now assume that
       $g_l'(x_l) = c > 0$ for every block. Which value of $c$ makes the
       paths through about $11$ blocks contribute the most to
       $\partial x_m / \partial x_0$, and what share of the derivative do the
       paths through $5$ to $17$ blocks then carry?

1. [code] **Gradient checks in single precision.** ● A gradient check
   compares a backward sweep with the central difference
   $\bigl(f(w + h, x, y, z) - f(w - h, x, y, z)\bigr) / (2h)$. Let $f$ be the
   function of :eqref:`eq_mdl-multi_func_def`, whose exact partial derivative
   at $(w, x, y, z) = (1, 1, -2, 1)$ is
   $\partial f / \partial w = 73472$.
    1. Implement $f$ in NumPy and compute the central difference at this point
       for $h = 10^{-1}, 10^{-2}, \ldots, 10^{-12}$, once in `float64` and
       once in `float32`. Plot the relative error against $h$ on log-log axes
       for both precisions, and read off the best $h$ and the smallest
       relative error of each.
    1. The error of the central difference is a truncation part of order
       $h^2$ plus a rounding part of order
       $\varepsilon_{\textrm{mach}} |f| / h$, where the machine epsilon
       $\varepsilon_{\textrm{mach}}$ is about $2.2 \times 10^{-16}$ for
       `float64` and $1.2 \times 10^{-7}$ for `float32`. By which factors
       should the best $h$ and the smallest relative error change from
       `float64` to `float32`? Compare the predicted factors with the plot.
    1. Which tolerance on the relative error can a `float32` gradient check
       use without rejecting a correct backward sweep? Why are gradient
       checks run in double precision?
    1. Why does the `float32` estimate become exactly zero for the smallest
       values of $h$?
   :exlabel:`ex_mdl-multivariable-calculus-gradient-check`

1. **Mixed partials that disagree.** ● Second-order methods treat the Hessian
   as a symmetric matrix. Let $f(x, y) = xy(x^2 - y^2)/(x^2 + y^2)$ for
   $(x, y) \neq (0, 0)$ and $f(0, 0) = 0$.
    1. Compute $\frac{\partial f}{\partial x}(0, y)$ and
       $\frac{\partial f}{\partial y}(x, 0)$, and from them the two mixed
       partials at the origin in the notation of :eqref:`eq_mdl-hess_def`.
    1. Evaluate the symmetric second difference
       :eqref:`eq_mdl-second-difference` at the origin for this $f$. Which
       step of the proof of the Clairaut–Schwarz theorem fails?
    1. What does the function `hessian` of this section return as `fxy` at
       the origin? What does a symmetric finite-difference formula miss here?

1. [code] **Random Hessians.** ● A minimum requires all $n$ eigenvalues of the
   Hessian to be positive, and the section notes that independent random
   signs would make this happen with probability $2^{-n}$. Test this
   heuristic on random symmetric matrices
   $\mathbf{A} = (\mathbf{G} + \mathbf{G}^\top)/\sqrt{2}$, where the
   $n \times n$ matrix $\mathbf{G}$ has independent standard normal entries.
    1. For $n = 2$, show that $s = (a_{11} + a_{22})/2$,
       $d = (a_{11} - a_{22})/2$, and $a_{12}$ are independent standard normal
       variables, and that $\mathbf{A} \succ 0$ exactly when
       $s > \sqrt{d^2 + a_{12}^2}$. Use the rotational symmetry of the
       standard normal distribution in three dimensions to compute
       $P(\mathbf{A} \succ 0)$ exactly, and compare it with $2^{-2}$.
    1. Estimate $P(\mathbf{A} \succ 0)$ by Monte Carlo for
       $n = 1, \ldots, 6$, with $10^6$ draws for $n \le 5$ and $10^7$ draws,
       generated in batches, for $n = 6$. Plot $\log P$ against $n$ together
       with $\log 2^{-n}$, and fit a quadratic in $n$ to your estimates.
       :citet:`Dean.Majumdar.2006` prove that
       $\log P \approx -\tfrac{\ln 3}{4} n^2$ for large $n$. Compare this
       coefficient with the quadratic coefficient of your fit. Which property
       of the eigenvalues does the coin-flip heuristic ignore?
    1. For large $n$ the smallest eigenvalue of $\mathbf{A}$ lies close to
       $-2\sqrt{n}$. For $n = 5, 10, 20$, find the shift $\mu$ at which half
       of the shifted matrices $\mathbf{A} + \mu\mathbf{I}$ are positive
       definite, and compare it with $2\sqrt{n}$. In the random landscapes of
       :citet:`Bray.Dean.2007`, the Hessian at a critical point behaves like
       such a shifted matrix, with a shift that grows as the value of the
       random function at the critical point decreases. What does your result
       suggest about critical points of low loss?

1. [extended] **Saddles of a linear network.** ●● Let $\mathbf{A}$ be a
   symmetric $n \times n$ matrix with eigenvalues
   $\lambda_1 > \lambda_2 > \cdots > \lambda_n > 0$ and orthonormal
   eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_n$. A linear network with
   one hidden unit, input weights $\mathbf{v}$, and output weights
   $\mathbf{u}$ maps $\mathbf{x}$ to $\mathbf{u}\mathbf{v}^\top\mathbf{x}$.
   Trained on whitened inputs ($\mathbb{E}[\mathbf{x}\mathbf{x}^\top] = \mathbf{I}$)
   with targets $\mathbf{A}\mathbf{x}$, half its expected squared error is

   $$L(\mathbf{u}, \mathbf{v}) = \tfrac12 \|\mathbf{A} - \mathbf{u}\mathbf{v}^\top\|_F^2.$$

   :citet:`Baldi.Hornik.1989` showed that the local minima of such linear
   networks are global. This problem derives the complete set of critical
   points and compares it with the relation between loss and index that
   :citet:`Dauphin.Pascanu.Gulcehre.ea.2014` measured for nonlinear
   networks; the *index* of a critical point is the fraction of negative
   eigenvalues of its Hessian.
    1. Compute $\nabla_{\mathbf{u}} L$ and $\nabla_{\mathbf{v}} L$. Show that
       the critical points are the origin and the points
       $\mathbf{u} = \alpha\mathbf{q}_i$,
       $\mathbf{v} = (\lambda_i / \alpha)\mathbf{q}_i$ with $\alpha \neq 0$,
       and compute $L$ at each of them.
    1. Compute the eigenvalues of the $2n \times 2n$ Hessian at the balanced
       point $\mathbf{u} = \mathbf{v} = \sqrt{\lambda_i}\,\mathbf{q}_i$ and at
       the origin. (In the eigenvector basis, the Hessian splits into
       $2 \times 2$ blocks, one for each pair of coordinates along
       $\mathbf{q}_j$ in $\mathbf{u}$ and in $\mathbf{v}$.) How many
       eigenvalues are negative at each critical point? Explain the zero
       eigenvalue by a symmetry of $L$, and conclude that every local minimum
       is global.
    1. Take $n = 4$, $(\lambda_1, \ldots, \lambda_4) = (4, 3, 2, 1)$, and a
       random orthonormal basis of eigenvectors. Run Newton's method from
       many random starting points of different scales, and record the loss
       and the index of every critical point it reaches. The Hessian is
       singular at every critical point other than the origin, so compute
       each Newton step by least squares. Plot the loss against the index,
       as in Figure 1 of :citet:`Dauphin.Pascanu.Gulcehre.ea.2014`. How often
       does Newton's method end at each kind of critical point, and where
       does gradient descent from small random initializations end?
    1. Replace $\mathbf{u}\mathbf{v}^\top$ by $\mathbf{U}\mathbf{V}^\top$ with
       two hidden units, so that $\mathbf{U}$ and $\mathbf{V}$ are
       $4 \times 2$ matrices. Predict the loss values of the critical points
       from the rank-one analysis, and confirm them with Newton's method.
       How many zero eigenvalues does the Hessian have at the critical
       points, and which symmetry explains them? Is the index a monotone
       function of the loss?


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1091)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1091)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §23.2]{.kicker}

Differentiation in many variables<br>**the gradient · its geometry · the chain rule · the Hessian**.
:::
:::

::: {.slide title="Derivatives with Many Parameters"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A deep network's loss depends on millions to billions of weights, yet training
rests on a single question: **how does the loss change when we nudge the
parameters?**

- The **gradient** $\nabla_{\mathbf{w}} L$ is the answer, the derivative in many dimensions.
- Its negative gives the direction of steepest **descent**, which underlies gradient-based optimization.
- The **chain rule**, organized by the gradient, *is* backpropagation.
- The **Hessian** tells a minimum from a saddle.
:::

::: {.col .fig}
![](../img/mdl-cal-gradient-field.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[From partials to the gradient]{.dtitle}

[one slope per coordinate, bundled into a vector]{.dsub}
:::
:::

::: {.slide title="The gradient is the derivative"}
[Gradients]{.kicker}

Perturb every coordinate at once and discard the second-order cross terms,
exactly as in one variable. The first-order term is a dot product:

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}), \qquad \nabla_{\mathbf{w}} L = \left[\tfrac{\partial L}{\partial w_1}, \ldots, \tfrac{\partial L}{\partial w_N}\right]^\top.$$

. . .

This is $f(x+\epsilon) \approx f(x) + \epsilon f'(x)$ with the scalar slope
replaced by a vector and the product by a dot product. The **gradient is
the unique vector whose dot product with a step gives the first-order
change** in $L$.
:::

::: {.slide title="Does the linear approximation hold?"}
[Gradients]{.kicker}

For $f(x,y) = \log(e^x+e^y)$ at $(0, \log 2)$, the gradient is
$[\tfrac13, \tfrac23]^\top$. We compare the first-order prediction against
the true value of a small step:

@multivariable-calculus-higher-dimensional-differentiation

. . .

They agree to about three decimals, and the gap of $1.8\times10^{-4}$ is itself the lesson: a first-order model discards terms of order $\|\boldsymbol{\epsilon}\|^2 = 10^{-3}$ (the Hessian quadratic, coming up). Shrink the step tenfold and the gap shrinks a hundredfold.

. . .

Autograd reproduces the hand gradient:

@!mdl-multivariable-calculus-directional-derivatives-1
:::

::: {.slide title="Directional Derivatives"}
[Gradients]{.kicker}

Read the approximation along a unit direction $\mathbf{u}$ and the rate of
change of $L$ there (the **directional derivative**) is the gradient's
projection onto $\mathbf{u}$:

$$\frac{L(\mathbf{w} + h\,\mathbf{u}) - L(\mathbf{w})}{h} \;\xrightarrow{\,h\to 0\,}\; \mathbf{u}\cdot\nabla_{\mathbf{w}} L.$$

::: {.d2l-note}
One vector $\nabla_{\mathbf{w}} L$ encodes the slope in **every** direction
simultaneously. The geometric results follow from this identity.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The geometry of gradients]{.dtitle}

[steepest descent, level sets, tangent planes]{.dsub}
:::
:::

::: {.slide title="Which way is steepest?"}
[Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
Along a unit direction $\mathbf{v}$, the change in $L$ is
$\|\nabla L\|\cos\theta$, so the direction enters only through $\theta$.

::: {.d2l-note .rule}
By Cauchy–Schwarz, $-\|\nabla L\| \le \mathbf{v}\cdot\nabla L \le \|\nabla L\|$:
$+\nabla L$ is **steepest ascent**, $-\nabla L$ **steepest descent**.
:::

Hence the gradient-descent step
$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_{\mathbf{w}} L$.
:::

::: {.col .fig}
![](../img/mdl-cal-gradient-field.svg)
:::
:::
:::

::: {.slide title="Linearization and Gradient Geometry"}
[Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
On the **base plane**, $L$ does not change along a level set, so
$\nabla L$ is **normal to the contours**, longest where they bunch.

One dimension up, on the **graph**, the same equation is the
**tangent plane** $z = f(\mathbf{x}_0) + \nabla f \cdot (\mathbf{x}-\mathbf{x}_0)$.

Drop the height coordinate and the surface normal becomes the gradient
crossing the contours square on.
:::

::: {.col .fig .big}
![](../img/mdl-cal-tangent-plane.svg)
:::
:::
:::

::: {.slide title="Where can a minimum be?"}
[Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
If $\nabla L(\mathbf{x}_0) \neq \mathbf{0}$, stepping along $-\nabla L$
*lowers* $L$, so $\mathbf{x}_0$ is no minimum. Contrapositively:

::: {.d2l-note .rule}
A minimum forces $\nabla L(\mathbf{x}_0) = \mathbf{0}$. Such points are
**critical points**: necessary, not sufficient.
:::

For $f(x) = 3x^4-4x^3-12x^2$, the critical points $x=-1,0,2$ have values
$-5, 0, -32$; the plot confirms the minimum at $x=2$.
:::

::: {.col .fig}
@!multivariable-calculus-a-note-on-mathematical-optimization
:::
:::
:::

::: {.slide title="Optimizing under a constraint"}
[Geometry]{.kicker}

::: {.cols .vc}
::: {.col}
Minimize $f$ subject to $g(\mathbf{x}) = c$. At the optimum no move *along*
the constraint can lower $f$, so $\nabla f$ is normal to $\{g=c\}$, and so
is $\nabla g$. Two normals to one surface are parallel:

$$\nabla f(\mathbf{x}^\star) = \lambda\,\nabla g(\mathbf{x}^\star).$$

::: {.d2l-note}
The level set of $f$ is *tangent* to the constraint at the optimum: $\lambda$ is the
**Lagrange multiplier**, the seed of the KKT conditions and duality.
:::
:::

::: {.col .fig}
![At the optimum the level set is tangent to the constraint and the gradients align; at the non-optimal point $\nabla f$ keeps a component along the constraint, so sliding still improves $f$.](../img/mdl-cal-lagrange-tangency.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The chain rule and backprop]{.dtitle}

[gradients are a sum over paths]{.dsub}
:::
:::

::: {.slide title="Compositions are graphs"}
[Chain rule]{.kicker}

::: {.cols .vc}
::: {.col}
A network is a deep composition of simple functions. Drawn out, the
dependencies form a **graph**: each node a value, each edge a direct
functional dependence.

Substituting the full composite expression and differentiating it repeats the same
subexpressions over and over. The chain rule organizes that waste away.
:::

::: {.col .fig .big}
![](../img/mdl-cal-chain-net1.svg)
:::
:::
:::

::: {.slide title="The rule is a sum over paths"}
[Chain rule]{.kicker}

::: {.cols .vc}
::: {.col}
Perturbing an input moves each intermediate by its partial. Reading off
the first-order coefficient gives the whole rule:

> Sum, over **every directed path** from input to output, the **product**
> of the edge derivatives along it.

Here $y$ reaches $f$ by three paths, including a skip edge $y\to u\to f$,
the same mechanism by which LSTM gates and residual connections shape
gradient flow.
:::

::: {.col .fig .big}
![](../img/mdl-cal-chain-net2.svg)
:::
:::
:::

::: {.slide title="Forward-Mode Derivative Evaluation"}
[Chain rule]{.kicker}

Push an input forward through the graph and the single-step partials multiply out. One forward sweep returns only $\partial f/\partial w$, with no head start on the other inputs:

@!multivariable-calculus-the-backpropagation-algorithm-1
:::

::: {.slide title="Backward Evaluation of the Gradient"}
[Chain rule]{.kicker}

Walk the graph from the output **backward**, keeping $\partial f$ in every *numerator*: compute $\tfrac{\partial f}{\partial u}, \tfrac{\partial f}{\partial v}$ once, reuse them, and **all** four input derivatives fall out in a single sweep. This *is* backpropagation.

@!multivariable-calculus-the-backpropagation-algorithm-2

. . .

Two sanity checks sit in the printout: $\partial f/\partial w = \partial f/\partial x$ and $\partial f/\partial y = \partial f/\partial z$, as they must ($f$ reaches its inputs only through $w+x$ and $y+z$), yet the two pairs differ in size *and* sign, so the sweep genuinely tells the paths apart.
:::

::: {.slide title="What autograd runs"}
[Chain rule]{.kicker}

The one-line autograd call runs exactly this backward pass. The four gradients match our by-hand sweep to the digit:

@!multivariable-calculus-the-backpropagation-algorithm-3

. . .

*Why* it is reverse-mode autodiff, a chain of vector–Jacobian products, is the matrix-calculus section.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Second-order: the Hessian]{.dtitle}

[curvature tells a minimum from a saddle]{.dsub}
:::
:::

::: {.slide title="The Hessian: curvature"}
[Hessian]{.kicker}

The gradient is first-order; the $n^2$ second partials determine curvature,
collected into the **Hessian** $\mathbf{H}_f$.

::: {.d2l-note .rule}
**Clairaut–Schwarz.** If the mixed partials are continuous, order doesn't matter:
$\partial^2 f/\partial x_i\partial x_j = \partial^2 f/\partial x_j\partial x_i$,
so $\mathbf{H}_f = \mathbf{H}_f^\top$.
:::

Because $\mathbf{H}$ is symmetric, the spectral theorem applies and supports
the second-derivative test.
:::

::: {.slide title="The best-fitting quadratic"}
[Hessian]{.kicker}

::: {.cols .vc}
::: {.col}
Just as the gradient gives the best *linear* fit, the Hessian gives the
best *quadratic* fit, the **second-order Taylor approximation**:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f \cdot (\mathbf{x}-\mathbf{x}_0) + \tfrac12 (\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f\, (\mathbf{x}-\mathbf{x}_0).$$

Near $\mathbf{x}_0$ the surface and the quadratic are nearly indistinguishable,
separating only farther out.
:::

::: {.col .fig .big}
![](../img/mdl-cal-taylor-quadratic.svg)
:::
:::
:::

::: {.slide title="Checking the quadratic"}
[Hessian]{.kicker}

Stepping from the base point $(-1, 0)$, the gap between $f$ and its Taylor quadratic $q$ stays tiny nearby and grows with distance:

@!multivariable-calculus-hessians

The gap is **third order** (double the step, eight times the gap), which is what the local quadratic guarantee predicts. Newton's method minimizes this model when it is positive definite and safeguards the step otherwise.
:::

::: {.slide title="The second-derivative test"}
[Hessian]{.kicker}

At a critical point the picture is purely quadratic; the curvature
$\mathbf{v}^\top\mathbf{H}\mathbf{v}$ along $\mathbf{v}$ gives the second-order
test through the **definiteness** of $\mathbf{H}$, read off its eigenvalues:

![](../img/mdl-la-psd.svg)

::: {.d2l-note .rule}
$\mathbf{H}\succ0$ → **minimum** (bowl). $\mathbf{H}\prec0$ → **maximum**.
Mixed signs → **saddle**. A zero eigenvalue → second order is silent.
:::
:::

::: {.slide title="The test, run as a program"}
[Hessian]{.kicker}

The surface $f(x,y) = x\,e^{-x^2-y^2}$ has exactly two critical points, $(\pm 1/\sqrt{2},\, 0)$. Assemble each Hessian from symmetric second differences, read off the $2\times2$ eigenvalues, and classify:

@!mdl-multivariable-calculus-the-second-derivative-test

. . .

Differentiate twice, extract eigenvalues, read the signs: all positive is a minimum, all negative a maximum, and a saddle shows one of each.
:::

::: {.slide title="Saddle Points in High Dimensions"}
[Hessian]{.kicker}

A minimum needs **all** $n$ eigenvalues positive at once. If their signs
behaved like coin flips, that is probability $2^{-n}$, vanishing in high
dimension.

. . .

So the critical points met while training deep nets are overwhelmingly
**saddles**, not the bad local minima once feared, one reason gradient
methods do so well in practice.
:::

::: {.slide title="The gradient represents Euclidean linear change"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- The **gradient** is the derivative in many dimensions: $\boldsymbol{\epsilon}\cdot\nabla L$ is the first-order change, and its projection gives the slope in any direction.
- $-\nabla L$ is **steepest descent**, and $\nabla L \perp$ the level sets: the geometry of gradient descent.
:::

::: {.col}
- The **chain rule** sums products of edge derivatives over paths; run **backward** it reuses everything: that is **backpropagation**.
- The symmetric **Hessian** gives the quadratic fit; its eigenvalues separate minimum, maximum, and saddle.
:::
:::

::: {.d2l-note}
Every optimizer in this book (GD, momentum, Adam, Newton) reads the local
Taylor expansion of the loss and moves against the gradient.
:::
:::
