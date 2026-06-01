# Geometry and Linear Algebraic Operations
:label:`sec_mdl-geometry-linear-algebraic-ops`

In :numref:`sec_linear-algebra`, we encountered the basics of linear algebra
and saw how it could be used to express common operations for transforming our data.
Linear algebra is one of the key mathematical pillars
underlying much of the work that we do in deep learning
and in machine learning more broadly.
While :numref:`sec_linear-algebra` contained enough machinery
to communicate the mechanics of modern deep learning models,
there is a lot more to the subject.
In this section, we will go deeper,
building geometric intuition for vectors, angles, projections,
hyperplanes, and the way matrices reshape space.
These pictures are the foundation for the two matrix decompositions
that run through all of deep learning, which we develop in the
sections that follow: *eigendecomposition*
(:numref:`sec_mdl-eigendecompositions`), the tool for analyzing
stability, PCA, and Hessians; and the *singular value decomposition*
(:numref:`sec_mdl-svd-low-rank`), the tool behind low-rank
approximation, conditioning, and LoRA.

## Vectors and Their Geometry

### Points and Directions

First, we need to discuss the two common geometric interpretations of vectors,
as either points or directions in space.
Fundamentally, a vector is a list of numbers such as the Python list below.

```{.python .input #geometry-linear-algebraic-ops-geometry-of-vectors}
v = [1, 7, 0, 1]
```

We rely on a single set of imports throughout the section; the few code
examples below use the framework's own tensor library and the d2l plotting
helpers.

```{.python .input #geometry-linear-algebraic-ops-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input #geometry-linear-algebraic-ops-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
```

```{.python .input #geometry-linear-algebraic-ops-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #geometry-linear-algebraic-ops-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
```

Mathematicians most often write this as either a *column* or *row* vector, which is to say either as

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

or

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

These often have different interpretations,
where data examples are column vectors
and weights used to form weighted sums are row vectors.
However, it can be beneficial to be flexible.
As we have described in :numref:`sec_linear-algebra`,
though a single vector's default orientation is a column vector,
for any matrix representing a tabular dataset,
treating each data example as a row vector
in the matrix
is more conventional.

Given a vector, the first interpretation
that we should give it is as a point in space.
In two or three dimensions, we can visualize these points
by using the components of the vectors to define
the location of the points in space compared
to a fixed reference called the *origin*.
In parallel, there is a second point of view
that people often take of vectors: as directions in space.
Not only can we think of the vector $\mathbf{v} = [3,2]^\top$
as the location $3$ units to the right and $2$ units up from the origin,
we can also think of it as the direction itself
to take $3$ steps to the right and $2$ steps up.
In this way, we consider all the parallel arrows in :numref:`fig_mdl-la-vectors` the same.

![A vector can be read two ways: as a *point* whose first component is the $x$-coordinate and second is the $y$-coordinate (left), or as a *direction*---an arrow that can start anywhere, so every arrow shown is the same vector $(3,2)^\top$ (right).](../img/mdl-la-vectors.svg)
:label:`fig_mdl-la-vectors`

This geometric point of view allows us to consider the problem on a more abstract level.
No longer faced with some insurmountable seeming problem
like classifying pictures as either cats or dogs,
we can start considering tasks abstractly
as collections of points in space and picturing the task
as discovering how to separate two distinct clusters of points.

One of the benefits of the direction view is that
we can make visual sense of the act of vector addition.
In particular, we follow the directions given by one vector,
and then follow the directions given by the other, as is seen in :numref:`fig_mdl-la-vector-add`.

![We can visualize vector addition by first following one vector, and then another, placing them tip to tail.](../img/mdl-la-vector-add.svg)
:label:`fig_mdl-la-vector-add`

Vector subtraction has a similar interpretation.
By considering the identity that $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
we see that the vector $\mathbf{u}-\mathbf{v}$ is the direction
that takes us from the point $\mathbf{v}$ to the point $\mathbf{u}$.


### Dot Products and Angles
As we saw in :numref:`sec_linear-algebra`,
if we take two column vectors $\mathbf{u}$ and $\mathbf{v}$,
we can form their dot product by computing:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_mdl-dot_def`

Because :eqref:`eq_mdl-dot_def` is symmetric, we will mirror the notation
of classical multiplication and write

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

to highlight the fact that exchanging the order of the vectors will yield the same answer.

The dot product :eqref:`eq_mdl-dot_def` also admits a geometric interpretation: it is closely related to the angle between two vectors.  Consider the angle shown in :numref:`fig_mdl-la-angle`.

![Between any two vectors in the plane there is a well defined angle $\theta$.  We will see this angle is intimately tied to the dot product.](../img/mdl-la-angle.svg)
:label:`fig_mdl-la-angle`

To start, let's consider two specific vectors:

$$
\mathbf{v} = (r,0) \; \textrm{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

The vector $\mathbf{v}$ is length $r$ and runs parallel to the $x$-axis,
and the vector $\mathbf{w}$ is of length $s$ and at angle $\theta$ with the $x$-axis.
If we compute the dot product of these two vectors, we see that

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

In short, for these two specific vectors the dot product, combined with the
norms, tells us the angle between them. Remarkably, the *same identity* holds
for **any** pair of vectors, in any number of dimensions. We can see why with
two short arguments that together both *justify* the formula and pin down
exactly when it makes sense.

The first argument is purely planar. Any two vectors $\mathbf{v}$ and
$\mathbf{w}$ — no matter how many coordinates they have — both lie in the
two-dimensional plane they span, and the angle $\theta$ between them lives in
that plane. So we lose nothing by reasoning in two dimensions. Expanding
$\|\mathbf{v} - \mathbf{w}\|^2$ with the dot product gives

$$
\|\mathbf{v} - \mathbf{w}\|^2
 = (\mathbf{v}-\mathbf{w})\cdot(\mathbf{v}-\mathbf{w})
 = \|\mathbf{v}\|^2 - 2\,\mathbf{v}\cdot\mathbf{w} + \|\mathbf{w}\|^2,
$$

while the planar law of cosines applied to the triangle with sides
$\mathbf{v}$, $\mathbf{w}$, and $\mathbf{v}-\mathbf{w}$ gives

$$
\|\mathbf{v} - \mathbf{w}\|^2
 = \|\mathbf{v}\|^2 + \|\mathbf{w}\|^2 - 2\,\|\mathbf{v}\|\|\mathbf{w}\|\cos\theta.
$$

Equating the two and cancelling $\|\mathbf{v}\|^2 + \|\mathbf{w}\|^2$ leaves the
**geometric formula for the dot product**,

$$
\mathbf{v}\cdot\mathbf{w} = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta,
$$
:eqlabel:`eq_mdl-dot_geom`

which we may solve for the angle:

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_mdl-angle_formula`

Nothing in the computation referenced the ambient dimension, so
:eqref:`eq_mdl-angle_formula` holds in three or three million dimensions
without change.

There is, however, a subtlety we must not skip. The function $\arccos$ is only
defined on the interval $[-1, 1]$, so :eqref:`eq_mdl-angle_formula` is
meaningful *only if* the fraction inside it never escapes that interval. That
this is guaranteed — in every dimension — is the content of one of the most
useful inequalities in all of mathematics.

**Proposition (Cauchy–Schwarz).** *For any vectors $\mathbf{v}, \mathbf{w}$,*

$$
|\mathbf{v}\cdot\mathbf{w}| \le \|\mathbf{v}\|\,\|\mathbf{w}\|,
$$
:eqlabel:`eq_mdl-cauchy-schwarz`

*with equality if and only if $\mathbf{v}$ and $\mathbf{w}$ are collinear
(one is a scalar multiple of the other).*

**Proof.** If $\mathbf{w} = \mathbf{0}$ both sides are zero and there is
nothing to prove, so assume $\mathbf{w} \neq \mathbf{0}$. The trick is to look
at the squared length of $\mathbf{v} - t\mathbf{w}$ as a function of the real
number $t$. A squared length is never negative, so

$$
q(t) = \|\mathbf{v} - t\mathbf{w}\|^2
     = \|\mathbf{w}\|^2\, t^2 - 2(\mathbf{v}\cdot\mathbf{w})\, t + \|\mathbf{v}\|^2
     \;\ge\; 0
     \quad\textrm{for every } t.
$$

This is a quadratic in $t$ with positive leading coefficient $\|\mathbf{w}\|^2$.
A quadratic that stays non-negative cannot have two distinct real roots, so its
discriminant must be $\le 0$:

$$
\bigl(2\,\mathbf{v}\cdot\mathbf{w}\bigr)^2 - 4\,\|\mathbf{w}\|^2\,\|\mathbf{v}\|^2 \le 0,
\qquad\textrm{i.e.}\qquad
(\mathbf{v}\cdot\mathbf{w})^2 \le \|\mathbf{v}\|^2\,\|\mathbf{w}\|^2.
$$

Taking square roots gives :eqref:`eq_mdl-cauchy-schwarz`. Equality forces the
discriminant to vanish, which means $q$ has a (repeated) real root $t^\star$
with $q(t^\star) = \|\mathbf{v} - t^\star\mathbf{w}\|^2 = 0$, that is
$\mathbf{v} = t^\star \mathbf{w}$. $\blacksquare$

The whole argument used nothing but the fact that *a squared length is
non-negative*. Dividing :eqref:`eq_mdl-cauchy-schwarz` by
$\|\mathbf{v}\|\|\mathbf{w}\|$ for nonzero $\mathbf{v}, \mathbf{w}$ yields the
**well-definedness of the angle**:

$$
-1 \;\le\; \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\,\|\mathbf{w}\|} \;\le\; 1,
$$

so the $\arccos$ in :eqref:`eq_mdl-angle_formula` is always defined and
$\theta$ is a genuine angle in $[0, \pi]$, no matter the dimension. The
equality cases are exactly the familiar ones: $\cos\theta = +1$ ($\theta = 0$)
when the vectors point the same way, and $\cos\theta = -1$ ($\theta = \pi$)
when they point in opposite directions — precisely the collinear cases of the
proposition.

Cauchy–Schwarz has a one-picture explanation, shown in
:numref:`fig_mdl-la-projection`. On the left, the projection of $\mathbf{v}$
onto $\mathbf{w}$ has signed length $\|\mathbf{v}\|\cos\theta$, and the residual
$\mathbf{r} = \mathbf{v} - \operatorname{proj}_{\mathbf{w}}\mathbf{v}$ meets
$\mathbf{w}$ at a right angle (we prove both facts in the next section). Because
the right triangle's hypotenuse $\mathbf{v}$ can be no shorter than its leg,
$\|\mathbf{v}\|\,|\cos\theta| \le \|\mathbf{v}\|$, which is exactly
$|\mathbf{v}\cdot\mathbf{w}| \le \|\mathbf{v}\|\|\mathbf{w}\|$. On the right is
the equality case: when $\mathbf{v}$ is collinear with $\mathbf{w}$ the residual
vanishes and the inequality becomes an equality.

![Left: the orthogonal projection of $\mathbf{v}$ onto $\mathbf{w}$ has signed length $\|\mathbf{v}\|\cos\theta$, and the residual $\mathbf{r}$ meets $\mathbf{w}$ at a right angle, so $\mathbf{v}$ is the hypotenuse of a right triangle. Right: the Cauchy–Schwarz equality case, where $\mathbf{v}$ is collinear with $\mathbf{w}$ and the residual vanishes.](../img/mdl-la-projection.svg)
:label:`fig_mdl-la-projection`

A first dividend of Cauchy–Schwarz is the **triangle inequality**, which says
that a detour through a third point is never shorter than going straight.

**Corollary (triangle inequality).** *For any $\mathbf{v}, \mathbf{w}$,*
$\|\mathbf{v} + \mathbf{w}\| \le \|\mathbf{v}\| + \|\mathbf{w}\|$.

**Proof.** Expand and apply :eqref:`eq_mdl-cauchy-schwarz`:

$$
\|\mathbf{v} + \mathbf{w}\|^2
 = \|\mathbf{v}\|^2 + 2\,\mathbf{v}\cdot\mathbf{w} + \|\mathbf{w}\|^2
 \le \|\mathbf{v}\|^2 + 2\,\|\mathbf{v}\|\|\mathbf{w}\| + \|\mathbf{w}\|^2
 = \bigl(\|\mathbf{v}\| + \|\mathbf{w}\|\bigr)^2.
$$

Taking square roots gives the claim. $\blacksquare$

As a simple example, let's see how to compute the angle between a pair of vectors:

```{.python .input #geometry-linear-algebraic-ops-dot-products-and-angles}
#@tab mxnet
def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input #geometry-linear-algebraic-ops-dot-products-and-angles}
#@tab pytorch
def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input #geometry-linear-algebraic-ops-dot-products-and-angles}
#@tab tensorflow
def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

```{.python .input #geometry-linear-algebraic-ops-dot-products-and-angles}
#@tab jax
def angle(v, w):
    return jnp.arccos(jnp.dot(v, w) / (jnp.linalg.norm(v) * jnp.linalg.norm(w)))

angle(jnp.array([0, 1, 2], dtype=jnp.float32), jnp.array([2.0, 3, 4]))
```

Two vectors whose angle is $\pi/2$ (equivalently $90^{\circ}$) are called
*orthogonal*. From :eqref:`eq_mdl-dot_geom`, the angle is $\pi/2$ exactly when
$\cos\theta = 0$, and since $\|\mathbf{v}\|\|\mathbf{w}\| \neq 0$ for nonzero
vectors, this happens if and only if the dot product itself vanishes. We
therefore *define* two vectors to be **orthogonal when**
$\mathbf{v}\cdot\mathbf{w} = 0$. (We take this as the definition because it
extends gracefully to the zero vector, which is orthogonal to everything even
though no angle is defined for it.) This will prove to be a workhorse condition
throughout the chapter.

### Projection and Orthogonality

Cauchy–Schwarz answers "how aligned are two vectors?"; the closely related
operation of *projection* answers "how much of $\mathbf{v}$ points along
$\mathbf{w}$?" Geometrically, we look for the point on the line
$\{t\mathbf{w} : t \in \mathbb{R}\}$ that is closest to $\mathbf{v}$.

**Proposition (orthogonal projection).** *Let $\mathbf{w} \neq \mathbf{0}$. The
closest multiple of $\mathbf{w}$ to $\mathbf{v}$ is*

$$
\operatorname{proj}_{\mathbf{w}}\mathbf{v}
 = \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}\,\mathbf{w},
$$
:eqlabel:`eq_mdl-projection`

*and the residual $\mathbf{r} = \mathbf{v} - \operatorname{proj}_{\mathbf{w}}\mathbf{v}$
is orthogonal to $\mathbf{w}$.*

**Proof.** We minimize the squared distance
$f(t) = \|\mathbf{v} - t\mathbf{w}\|^2
 = \|\mathbf{w}\|^2 t^2 - 2(\mathbf{v}\cdot\mathbf{w})\,t + \|\mathbf{v}\|^2$.
This is a convex parabola in $t$; setting $f'(t) = 2\|\mathbf{w}\|^2 t -
2(\mathbf{v}\cdot\mathbf{w}) = 0$ gives the unique minimizer
$t^\star = \dfrac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}$, which
is :eqref:`eq_mdl-projection`. For orthogonality of the residual, compute

$$
\mathbf{r}\cdot\mathbf{w}
 = \mathbf{v}\cdot\mathbf{w}
   - \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}\,(\mathbf{w}\cdot\mathbf{w})
 = 0. \qquad \blacksquare
$$

Because $\mathbf{r}$ is orthogonal to $\operatorname{proj}_{\mathbf{w}}\mathbf{v}$
(which is a multiple of $\mathbf{w}$), the decomposition
$\mathbf{v} = \operatorname{proj}_{\mathbf{w}}\mathbf{v} + \mathbf{r}$ splits
$\mathbf{v}$ into two perpendicular pieces, and **Pythagoras** applies:

$$
\|\mathbf{v}\|^2 = \|\operatorname{proj}_{\mathbf{w}}\mathbf{v}\|^2 + \|\mathbf{r}\|^2 .
$$
:eqlabel:`eq_mdl-pythagoras`

Two remarks tie this back to the rest of the section. First, the *signed
length* of the projection is

$$
\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{w}\|} = \|\mathbf{v}\|\cos\theta ,
$$

which is exactly the quantity the hyperplane discussion below will use, so the
hyperplane material is now fully self-contained. Second, we just solved a
genuine (if tiny) *least-squares* problem: we found the best approximation of
$\mathbf{v}$ from the one-dimensional subspace spanned by $\mathbf{w}$. The
same idea — project onto a subspace, the residual comes out orthogonal —
scales up to fitting an arbitrary linear model, which is how the singular value
decomposition produces optimal least-squares solutions in
:numref:`sec_mdl-svd-low-rank`.

## Similarity in High Dimensions

It is reasonable to ask why the *angle* — rather than the raw distance — is so
often the right notion of similarity. The answer is invariance to scale.
Consider an image and a copy of it dimmed to $10\%$ brightness. Pixel by pixel
the two are far apart, so their Euclidean distance is large; yet the content is
identical, and a cat/dog classifier should treat them the same. The angle does:
for any vector $\mathbf{v}$, the angle between $\mathbf{v}$ and $0.1\,\mathbf{v}$
is zero, because scaling changes a vector's length but not its direction. This
is why, when the angle is used to compare two vectors, practitioners work with
its cosine and call it **cosine similarity**,

$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\,\|\mathbf{w}\|}
 \;\in\; [-1, 1],
$$
:eqlabel:`eq_mdl-cosine-sim`

equal to $+1$ when the vectors point the same way, $-1$ when opposite, and $0$
when orthogonal. Cosine similarity is the metric behind nearest-neighbor
retrieval over **embeddings**, the scaled dot products inside **attention**
(:numref:`sec_attention-scoring-functions`), and the alignment objective of
**contrastive learning**: in each case we have represented objects as vectors
and we measure relatedness by direction, deliberately discarding magnitude.

This raises a question that turns out to have a striking answer. If we drop two
*unrelated* vectors into a high-dimensional space, what cosine should we expect
between them? The answer is that **in high dimensions, random vectors are almost
always nearly orthogonal** — and the higher the dimension, the more sharply this
holds.

**Proposition (near-orthogonality).** *Fix a unit vector $\mathbf{u} \in
\mathbb{R}^d$ and let $\mathbf{v}$ be drawn uniformly from the unit sphere in
$\mathbb{R}^d$. Then $\cos\theta = \mathbf{u}\cdot\mathbf{v}$ satisfies*

$$
\mathbb{E}[\cos\theta] = 0,
\qquad
\operatorname{Var}(\cos\theta) = \frac{1}{d}.
$$

**Proof.** The uniform distribution on the sphere is invariant under rotations,
so we may rotate coordinates until $\mathbf{u} = \mathbf{e}_1$; then
$\cos\theta = \mathbf{u}\cdot\mathbf{v} = v_1$. By the symmetry $\mathbf{v}
\mapsto -\mathbf{v}$ we have $\mathbb{E}[v_1] = 0$. For the variance, every
coordinate plays the same role by symmetry, so $\mathbb{E}[v_i^2]$ is the same
for all $i$; summing and using $\sum_i v_i^2 = \|\mathbf{v}\|^2 = 1$ gives
$d\,\mathbb{E}[v_1^2] = \mathbb{E}\!\left[\sum_i v_i^2\right] = 1$, hence
$\operatorname{Var}(\cos\theta) = \mathbb{E}[v_1^2] = 1/d$. Chebyshev's
inequality then bounds the chance of a large cosine,

$$
\Pr\bigl(|\cos\theta| \ge \varepsilon\bigr) \le \frac{1}{d\,\varepsilon^2},
$$

which tends to $0$ as $d \to \infty$ for any fixed $\varepsilon > 0$.
$\blacksquare$

So the typical cosine between random directions has standard deviation
$1/\sqrt{d}$, concentrating ever more tightly at $0$. This is a first taste of
*concentration of measure*, the phenomenon that makes high-dimensional geometry
behave very differently from our $2$- and $3$-dimensional intuition. It is also
exactly *why cosine similarity is such a useful signal*: since unrelated items
are nearly orthogonal by default, a cosine that is appreciably above $0$ is
unlikely to be an accident and instead reflects real shared structure — the
working assumption behind embedding-based retrieval and the attention mechanism.

We can watch the concentration happen by sampling random unit vectors and
histogramming their pairwise cosines as the dimension grows, shown in
:numref:`fig_mdl-la-cosine-highd`.

![Histograms of the cosine between independent random unit vectors at dimensions $d = 2$, $10$, and $1000$. The $d = 2$ histogram is broad and flat; by $d = 1000$ it is a narrow spike at $0$ of width $\approx 1/\sqrt{d}$, exactly as the proposition predicts.](../img/mdl-la-cosine-highd.svg)
:label:`fig_mdl-la-cosine-highd`

The higher the dimension, the more sharply the cosine concentrates at $0$:
unrelated directions are almost always nearly orthogonal.

## Hyperplanes and Decision Boundaries

In addition to working with vectors, another key object
that you must understand to go far in linear algebra
is the *hyperplane*, a generalization to higher dimensions
of a line (two dimensions) or of a plane (three dimensions).
In an $d$-dimensional vector space, a hyperplane has $d-1$ dimensions
and divides the space into two half-spaces.

Let's start with an example.
Suppose that we have a column vector $\mathbf{w}=[2,1]^\top$ and a scalar
*bias* $b$. We want to know, "what are the points $\mathbf{v}$ with
$\mathbf{w}\cdot\mathbf{v} = b$?" For concreteness we first take $b = 1$.
By recalling the connection between dot products and angles above :eqref:`eq_mdl-angle_formula`,
we can see that this is equivalent to
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

If we consider the geometric meaning of this expression,
we see that this is equivalent to saying
that the signed length of the projection of $\mathbf{v}$
onto the direction of $\mathbf{w}$ is exactly $1/\|\mathbf{w}\|$ --- recall the
signed projection length $\|\mathbf{v}\|\cos(\theta)$ from
:numref:`fig_mdl-la-projection`.
The set of all points where this is true is a line
at right angles to the vector $\mathbf{w}$.
If we wanted, we could find the equation for this line
and see that it is $2x + y = 1$ or equivalently $y = 1 - 2x$.

More generally, the equation $\mathbf{w}\cdot\mathbf{v} = b$ for any scalar
$b$ describes a line (in higher dimensions, a hyperplane) at right angles to
$\mathbf{w}$. The vector $\mathbf{w}$ is called the *normal* to the hyperplane,
and the bias $b$ controls *where along that normal* the hyperplane sits.
By the same projection argument, every point on it has the same signed
projection $\mathbf{w}\cdot\mathbf{v}/\|\mathbf{w}\| = b/\|\mathbf{w}\|$ onto the
direction of $\mathbf{w}$, so the hyperplane passes at (signed) distance
$b/\|\mathbf{w}\|$ from the origin. Changing $b$ slides the hyperplane along
$\mathbf{w}$ without rotating it; the case $b=0$ is the hyperplane through the
origin. More usefully, for *any* point $\mathbf{x}$ the quantity
$$
\frac{\mathbf{w}\cdot\mathbf{x} - b}{\|\mathbf{w}\|}
$$
is the *signed distance* from $\mathbf{x}$ to the hyperplane: positive on the
side $\mathbf{w}$ points toward, negative on the other, and zero exactly on it.
This signed distance is precisely the *margin* used by linear classifiers. The
derivation is just the projection result of the previous section applied to the
displacement of $\mathbf{x}$ from any point on the hyperplane, which is why the
projection material had to come first. :numref:`fig_mdl-la-hyperplane` collects
these facts: the normal $\mathbf{w}$ as an arrow from the origin, two parallel
level lines $\mathbf{w}\cdot\mathbf{x} = b$ for two offsets $b$ (the larger one
shifted along $\mathbf{w}$ without any rotation), the half-space
$\mathbf{w}\cdot\mathbf{x} > b$ lightly shaded, and the signed distance
$b/\|\mathbf{w}\|$ from the origin to the line.

![The hyperplane $\mathbf{w}\cdot\mathbf{x} = b$ with normal $\mathbf{w}$. Sliding the offset $b$ translates the hyperplane along $\mathbf{w}$ without rotating it; the shaded region is the half-space $\mathbf{w}\cdot\mathbf{x} > b$; and the signed distance from any point to the hyperplane is $(\mathbf{w}\cdot\mathbf{x} - b)/\|\mathbf{w}\|$, the quantity a linear classifier reads off as its margin.](../img/mdl-la-hyperplane.svg)
:label:`fig_mdl-la-hyperplane`

If we now look at what happens when we ask about the set of points with
$\mathbf{w}\cdot\mathbf{v} > b$ or $\mathbf{w}\cdot\mathbf{v} < b$,
we can see that these are cases where the projections
are longer or shorter than $b/\|\mathbf{w}\|$, respectively
(equivalently, the signed distance above is positive or negative).
Thus, those two inequalities define either side of the line, cutting our space
into two halves: all the points on one side have dot product below a threshold,
and the other side above.

The story in higher dimensions is much the same.
If we now take $\mathbf{w} = [1,2,3]^\top$
and ask about the points in three dimensions with $\mathbf{w}\cdot\mathbf{v} = b$,
we obtain a plane at right angles to the given vector $\mathbf{w}$,
offset from the origin by the signed distance $b/\|\mathbf{w}\|$.
The two inequalities again define the two sides of the plane.

While our ability to visualize runs out at this point,
nothing stops us from doing this in tens, hundreds, or billions of dimensions.
This occurs often when thinking about machine learned models.
For instance, we can understand linear classification models
like those from :numref:`sec_softmax`,
as methods to find hyperplanes that separate the different target classes.
In this context, such hyperplanes are often referred to as *decision planes*:
the learned weight vector is the normal $\mathbf{w}$ and the learned bias is
exactly the offset $b$, with the predicted class read off from the sign of
$\mathbf{w}\cdot\mathbf{x} - b$.
The majority of deep learned classification models end
with a linear layer fed into a softmax,
so one can interpret the role of the deep neural network
to be to find a non-linear embedding such that the target classes
can be separated cleanly by hyperplanes.

To give a hand-built example, notice that we can produce a reasonable model
to classify tiny images of t-shirts and trousers from the Fashion-MNIST dataset
(seen in :numref:`sec_fashion_mnist`)
by just taking the vector between their means to define the decision plane
and eyeball a crude threshold.  First we will load the data and compute the averages.

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-1}
#@tab mxnet
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

# In MXNet 2.0 reductions over `float` (== float64) inputs stay float64, but
# many fused kernels still emit float32 — pin everything to float32 up front so
# downstream dot products see matching dtypes.
X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype('float32')
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype('float32')
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype('float32')
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype('float32')

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-1}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-1}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]]), dtype=tf.float32) * 256
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]]), dtype=tf.float32) * 256
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0 or label == 1]]),
    dtype=tf.float32) * 256
y_test = tf.cast(tf.stack([label for label in test_labels
    if label == 0 or label == 1]), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-1}
#@tab jax
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()

X_train_0 = jnp.array(train_images[train_labels == 0], dtype=jnp.float32) * 256
X_train_1 = jnp.array(train_images[train_labels == 1], dtype=jnp.float32) * 256
X_test = jnp.array(
    test_images[(test_labels == 0) | (test_labels == 1)], dtype=jnp.float32) * 256
y_test = jnp.array(
    test_labels[(test_labels == 0) | (test_labels == 1)], dtype=jnp.float32)

# Compute averages
ave_0 = jnp.mean(X_train_0, axis=0)
ave_1 = jnp.mean(X_train_1, axis=0)
```

It can be informative to examine these averages in detail, so let's plot what they look like.  In this case, we see that the average indeed resembles a blurry image of a t-shirt.

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab jax
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(np.array(ave_0.reshape(28, 28)), cmap='Greys')
d2l.plt.show()
```

In the second case, we again see that the average resembles a blurry image of trousers.

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-3}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-3}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-3}
#@tab jax
# Plot average trousers
d2l.plt.imshow(np.array(ave_1.reshape(28, 28)), cmap='Greys')
d2l.plt.show()
```

In a fully machine learned solution, we would learn the threshold from the
dataset. Here we set it geometrically instead: the normal is the difference of
the two class means $\mathbf{w} = \overline{\mathbf{x}}_1 - \overline{\mathbf{x}}_0$,
and the natural decision boundary is the hyperplane that *bisects* the two means
— that is, $\mathbf{w}\cdot\mathbf{x} = b$ with
$b = \mathbf{w}\cdot\tfrac12(\overline{\mathbf{x}}_0 + \overline{\mathbf{x}}_1)$,
the midpoint of the two means' projections onto $\mathbf{w}$. We classify a test
image as class $1$ when it lands on the class-$1$ side, i.e.
$\mathbf{w}\cdot\mathbf{x} > b$. Note that deriving $b$ from the data this way is
*scale-equivariant*: it gives the same boundary whatever convention each
framework uses for pixel intensities, which a hand-picked numeric threshold
would not.

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-4}
#@tab mxnet
# Normal = difference of class means; threshold = midpoint of their projections
w = (ave_1 - ave_0).flatten()
b = np.dot(w, (ave_0 + ave_1).flatten()) / 2
predictions = X_test.reshape(2000, -1).dot(w) > b

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-4}
#@tab pytorch
# Normal = difference of class means; threshold = midpoint of their projections
w = (ave_1 - ave_0).flatten()
b = torch.dot(w, (ave_0 + ave_1).flatten()) / 2
# '@' is the matrix-multiplication operator in PyTorch.
predictions = X_test.reshape(2000, -1) @ w > b

# Accuracy
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-4}
#@tab tensorflow
# Normal = difference of class means; threshold = midpoint of their projections
w = tf.reshape(ave_1 - ave_0, [-1])
b = tf.tensordot(w, tf.reshape(ave_0 + ave_1, [-1]), axes=1) / 2
# Genuine per-example dot product: flatten each image and matvec against w.
predictions = tf.linalg.matvec(tf.reshape(X_test, (2000, -1)), w) > b

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-4}
#@tab jax
# Normal = difference of class means; threshold = midpoint of their projections
w = (ave_1 - ave_0).flatten()
b = jnp.dot(w, (ave_0 + ave_1).flatten()) / 2
predictions = X_test.reshape(2000, -1) @ w > b

# Accuracy
jnp.mean((predictions.astype(y_test.dtype) == y_test).astype(jnp.float32))
```

## Matrices as Linear Maps

### Linear Transformations

Through :numref:`sec_linear-algebra` and the above discussions,
we have a solid understanding of the geometry of vectors, lengths, and angles.
However, there is one important object we have omitted discussing,
and that is a geometric understanding of linear transformations represented by matrices.  Fully internalizing what matrices can do to transform data
between two potentially different high dimensional spaces takes significant practice,
and is beyond the scope of this appendix.
However, we can start building up intuition in two dimensions.

Suppose that we have some matrix:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

If we want to apply this to an arbitrary vector
$\mathbf{v} = [x, y]^\top$,
we multiply and see that

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

This may seem like an odd computation,
where something clear became somewhat impenetrable.
However, it tells us that we can write the way
that a matrix transforms *any* vector
in terms of how it transforms *two specific vectors*:
$[1,0]^\top$ and $[0,1]^\top$.
This is worth considering for a moment.
We have essentially reduced an infinite problem
(what happens to any pair of real numbers)
to a finite one (what happens to these specific vectors).
These vectors are an example of a *basis*,
where we can write any vector in our space
as a weighted sum of these *basis vectors*.

Let's draw what happens when we use the specific matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

If we look at the specific vector $\mathbf{v} = [2, -1]^\top$,
we see this is $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
and thus we know that the matrix $A$ will send this to
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
If we follow this logic through carefully,
say by considering the grid of all integer pairs of points,
we see that what happens is that the matrix multiplication
can skew, rotate, and scale the grid,
but the grid structure must remain as you see in :numref:`fig_mdl-la-linear-map`.

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/mdl-la-linear-map.svg)
:label:`fig_mdl-la-linear-map`

This is the most important intuitive point
to internalize about linear transformations represented by matrices.
Matrices are incapable of distorting some parts of space differently than others.
All they can do is take the original coordinates on our space
and skew, rotate, and scale them.

Some distortions can be severe.  For instance the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

compresses the entire two-dimensional plane down to a single line.
Identifying and working with such transformations are the topic of a later section,
but geometrically we can see that this is fundamentally different
from the types of transformations we saw above.
For instance, the result from matrix $\mathbf{A}$ can be "bent back" to the original grid.  The results from matrix $\mathbf{B}$ cannot
because we will never know where the vector $[1,2]^\top$ came from---was
it $[1,1]^\top$ or $[0, -1]^\top$?

While this picture was for a $2\times2$ matrix,
nothing prevents us from taking the lessons learned into higher dimensions.
If we take similar basis vectors like $[1,0, \ldots,0]$
and see where our matrix sends them,
we can start to get a feeling for how the matrix multiplication
distorts the entire space in whatever dimension space we are dealing with.

### Orthogonal Matrices

A matrix may skew, rotate, and scale, but a special and important family does
*only* the rigid part — it rotates or reflects without any stretching. A square
matrix $\mathbf{Q}$ is called **orthogonal** when its columns are orthonormal,
which we can write compactly as $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$. The
defining property of such maps is that they **preserve lengths and angles**,
because they preserve every dot product:

$$
(\mathbf{Q}\mathbf{x})\cdot(\mathbf{Q}\mathbf{y})
 = (\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y})
 = \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\,\mathbf{y}
 = \mathbf{x}^\top\mathbf{y}
 = \mathbf{x}\cdot\mathbf{y}.
$$

Taking $\mathbf{y} = \mathbf{x}$ shows $\|\mathbf{Q}\mathbf{x}\| =
\|\mathbf{x}\|$, so an orthogonal map is a rigid motion of space. Since
$\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ means $\mathbf{Q}^{-1} =
\mathbf{Q}^\top$, such maps are trivially invertible, and as we will see in the
next section their volume scaling is $\det\mathbf{Q} = \pm 1$ (the sign
distinguishing rotations from reflections). Orthogonal matrices are the
"distortion-free" linear maps, and they will turn out to be the building blocks
of the two decompositions in the sections that follow: the spectral theorem
writes a symmetric matrix as $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$
(:numref:`sec_mdl-eigendecompositions`), and the singular value decomposition
writes *any* matrix as orthogonal–diagonal–orthogonal
(:numref:`sec_mdl-svd-low-rank`).

### Linear Dependence, Rank, and Invertibility

Consider again the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

This compresses the entire plane down to live on the single line $y = 2x$.
The question now arises: is there some way we can detect this
just looking at the matrix itself?
The answer is that indeed we can.
Let's take $\mathbf{b}_1 = [2,4]^\top$ and $\mathbf{b}_2 = [-1, -2]^\top$
be the two columns of $\mathbf{B}$.
Remember that we can write everything transformed by the matrix $\mathbf{B}$
as a weighted sum of the columns of the matrix:
like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
We call this a *linear combination*.
The fact that $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$
means that we can write any linear combination of those two columns
entirely in terms of say $\mathbf{b}_2$ since

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

This means that one of the columns is, in a sense, redundant
because it does not define a unique direction in space.
This should not surprise us too much
since we already saw that this matrix
collapses the entire plane down into a single line.
Moreover, we see that the linear dependence
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ captures this.
To make this more symmetrical between the two vectors, we will write this as

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

In general, we will say that a collection of vectors
$\mathbf{v}_1, \ldots, \mathbf{v}_k$ are *linearly dependent*
if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

In this case, we can solve for one of the vectors
in terms of some combination of the others,
and effectively render it redundant.
Thus, a linear dependence in the columns of a matrix
is a witness to the fact that our matrix
is compressing the space down to some lower dimension.
If there is no linear dependence we say the vectors are *linearly independent*.
If the columns of a matrix are linearly independent,
no compression occurs and the operation can be undone.

#### Rank

If we have a general $n\times m$ matrix,
it is reasonable to ask what dimension space the matrix maps into.
A concept known as the *rank* will be our answer.
In the previous section, we noted that a linear dependence
bears witness to compression of space into a lower dimension
and so we will be able to use this to define the notion of rank.
In particular, the rank of a matrix $\mathbf{A}$
is the largest number of linearly independent columns
amongst all subsets of columns. For example, the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

has $\textrm{rank}(B)=1$, since the two columns are linearly dependent,
but either column by itself is not linearly dependent.
For a more challenging example, we can consider

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

and show that $\mathbf{C}$ has rank two since, for instance,
the first two columns are linearly independent,
however any of the $\binom{5}{3} = 10$ collections of three columns are linearly dependent.

Equivalently, the rank is the dimension of the *column space* (the span of the
columns), and a foundational theorem of linear algebra says this equals the
dimension of the *row space*. A matrix "compresses space" into a lower dimension
exactly when its rank is smaller than its number of columns --- equivalently, when
it has a nontrivial null space (some nonzero $\mathbf{x}$ with $\mathbf{A}\mathbf{x}=\mathbf{0}$).

This procedure, as described, is very inefficient.
It requires looking at every subset of the columns of our given matrix,
and thus is potentially exponential in the number of columns.
Later we will see a more computationally efficient way
to compute the rank of a matrix, but for now,
this is sufficient to see that the concept
is well defined and understand the meaning.

#### Invertibility

We have seen above that multiplication by a matrix with linearly dependent columns
cannot be undone, i.e., there is no inverse operation that can always recover the input.  However, multiplication by a full-rank matrix
(i.e., some $\mathbf{A}$ that is $n \times n$ matrix with rank $n$),
we should always be able to undo it.  Consider the matrix

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

which is the matrix with ones along the diagonal, and zeros elsewhere.
We call this the *identity* matrix.
It is the matrix which leaves our data unchanged when applied.
To find a matrix which undoes what our matrix $\mathbf{A}$ has done,
we want to find a matrix $\mathbf{A}^{-1}$ such that

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

If we look at this as a system, we have $n \times n$ unknowns
(the entries of $\mathbf{A}^{-1}$) and $n \times n$ equations
(the equality that needs to hold between every entry of the product $\mathbf{A}^{-1}\mathbf{A}$ and every entry of $\mathbf{I}$)
so we should generically expect a solution to exist.
Indeed, in the next section we will see a quantity called the *determinant*,
which has the property that as long as the determinant is not zero, we can find a solution.  We call such a matrix $\mathbf{A}^{-1}$ the *inverse* matrix.
As an example, if $\mathbf{A}$ is the general $2 \times 2$ matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

then we can see that the inverse is

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

We can test to see this by seeing that multiplying
by the inverse given by the formula above works in practice.

```{.python .input #geometry-linear-algebraic-ops-invertibility}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input #geometry-linear-algebraic-ops-invertibility}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input #geometry-linear-algebraic-ops-invertibility}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

```{.python .input #geometry-linear-algebraic-ops-invertibility}
#@tab jax
M = jnp.array([[1, 2], [1, 4]], dtype=jnp.float32)
M_inv = jnp.array([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

#### Numerical Issues
While the inverse of a matrix is useful in theory,
we must say that most of the time we do not wish
to *use* the matrix inverse to solve a problem in practice.
In general, there are far more numerically stable algorithms
for solving linear equations like

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

than computing the inverse and multiplying to get

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Just as division by a small number can lead to numerical instability,
so can inversion of a matrix which is close to having low rank.

Moreover, it is common that the matrix $\mathbf{A}$ is *sparse*,
which is to say that it contains only a small number of non-zero values.
If we were to explore examples, we would see
that this does not mean the inverse is sparse.
Even if $\mathbf{A}$ was a $1$ million by $1$ million matrix
with only $5$ million non-zero entries
(and thus we need only store those $5$ million),
the inverse will typically have almost every entry non-zero,
requiring us to store all $1\textrm{M}^2$ entries---that is $1$ trillion entries!

While we do not have time to dive all the way into the thorny numerical issues
frequently encountered when working with linear algebra,
we want to provide you with some intuition about when to proceed with caution,
and generally avoiding inversion in practice is a good rule of thumb.

### The Determinant
The geometric view of linear algebra gives an intuitive way
to interpret a fundamental quantity known as the *determinant*.
Consider the grid image from before, but now with a highlighted region (:numref:`fig_mdl-la-determinant`).

![The determinant as a signed area. The unit square spanned by the basis vectors maps to the parallelogram spanned by the columns of $\mathbf{A}$; its (signed) area is the determinant. A matrix that flips orientation gives a negative determinant, and a matrix that collapses the square to a segment gives a determinant of zero.](../img/mdl-la-determinant.svg)
:label:`fig_mdl-la-determinant`

Look at the highlighted square.  This is a square with edges given
by $(0, 1)$ and $(1, 0)$ and thus it has area one.
After $\mathbf{A}$ transforms this square,
we see that it becomes a parallelogram.
There is no reason this parallelogram should have the same area
that we started with, and indeed in the specific case shown here of

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

it is an exercise in coordinate geometry to compute
the area of this parallelogram and obtain that the area is $5$.

In general, if we have a matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

we can see with some computation that the area
of the resulting parallelogram is $ad-bc$.
This area is referred to as the *determinant*.

Let's check this quickly with some example code.

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab mxnet
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab jax
jnp.linalg.det(jnp.array([[1, -1], [2, 3]], dtype=jnp.float32))
```

The eagle-eyed amongst us will notice
that this expression can be zero or even negative.
For the negative term, this is a matter of convention
taken generally in mathematics:
if the matrix flips the figure,
we say the area is negated.
Let's see now that when the determinant is zero, we learn more.

Let's consider

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

If we compute the determinant of this matrix,
we get $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Given our understanding above, this makes sense.
$\mathbf{B}$ compresses the square from the original image
down to a line segment, which has zero area.
And indeed, being compressed into a lower dimensional space
is the only way to have zero area after the transformation.
Thus we see the following result is true:
a matrix $A$ is invertible if and only if
the determinant is not equal to zero.

This single equivalence is the thread that ties together three notions we have
met separately — *linear dependence*, *invertibility*, and the *determinant* —
and it is worth stating once, cleanly, with a proof we can carry out by hand in
two dimensions.

**Proposition (the unifying theorem).** *For a square matrix $\mathbf{A}$, the
following are equivalent:*
(i) *$\det\mathbf{A} = 0$;*
(ii) *the columns of $\mathbf{A}$ are linearly dependent;*
(iii) *$\mathbf{A}$ is not invertible.*

**Proof.** We give the argument for the $2 \times 2$ matrix
$\mathbf{A} = \bigl[\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr]$,
where every step is a picture; the same chain of reasoning holds in any
dimension with "area" replaced by "$n$-dimensional volume." Write the two
columns as $\mathbf{a}_1 = [a, c]^\top$ and $\mathbf{a}_2 = [b, d]^\top$. As we
saw above, $\det\mathbf{A} = ad - bc$ is the *signed area* of the parallelogram
spanned by $\mathbf{a}_1$ and $\mathbf{a}_2$.

*(i) $\Leftrightarrow$ (ii).* A parallelogram has zero area exactly when its two
spanning edges are collinear, i.e., when one column is a scalar multiple of the
other (including the degenerate case where a column is $\mathbf{0}$). That is
precisely linear dependence of the columns. So $ad - bc = 0$ if and only if
$\mathbf{a}_1$ and $\mathbf{a}_2$ are linearly dependent.

*(ii) $\Leftrightarrow$ (iii).* If the columns are dependent, every output
$\mathbf{A}\mathbf{x} = x_1\mathbf{a}_1 + x_2\mathbf{a}_2$ lies on the single
line spanned by the surviving column, so the whole plane is crushed onto that
line. Distinct inputs collide there (the map is not one-to-one), so no inverse
can recover them, and $\mathbf{A}$ is not invertible. Conversely, if the columns
are independent they span the plane, every target is hit exactly once, and the
map can be undone — concretely, $ad - bc \neq 0$ is exactly the nonvanishing
denominator that made the explicit inverse
$\frac{1}{ad-bc}\bigl[\begin{smallmatrix} d & -b \\ -c & a \end{smallmatrix}\bigr]$
well-defined earlier in this section. $\blacksquare$

The equivalence retroactively justifies the claims we made on credit:
linear dependence (the columns of $\mathbf{B}$ are redundant), the missing
$ad - bc \neq 0$ hypothesis under the $2 \times 2$ inverse, and the present
section's "$\det = 0$ means collapse" all turn out to be three faces of the
same fact.

As a final comment, imagine that we have any figure drawn on the plane.
Thinking like computer scientists, we can decompose
that figure into a collection of little squares
so that the area of the figure is in essence
just the number of squares in the decomposition.
If we now transform that figure by a matrix,
we send each of these squares to parallelograms,
each one of which has area given by the determinant.
We see that for any figure, the determinant gives the (signed) number
that a matrix scales the area of any figure.

This "scale every figure's area by the same factor" reading has an immediate and
powerful consequence for *composing* two transformations.

**Proposition (multiplicativity of the determinant).** *For square matrices
$\mathbf{A}$ and $\mathbf{B}$ of the same size,*

$$
\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\,\det(\mathbf{B}).
$$
:eqlabel:`eq_mdl-det-multiplicative`

**Proof.** Apply $\mathbf{A}\mathbf{B}$ to an arbitrary figure of area $V$ by
running the two maps in turn. First $\mathbf{B}$ acts, and by the volume-scaling
property just established it scales the area to $\det(\mathbf{B})\,V$. Then
$\mathbf{A}$ acts on that result and scales its area by a further factor of
$\det(\mathbf{A})$, giving $\det(\mathbf{A})\,\det(\mathbf{B})\,V$. But the
composite map $\mathbf{A}\mathbf{B}$ is itself a single linear transformation, so
by the very same property it scales the original area by exactly its own
determinant: the final area is $\det(\mathbf{A}\mathbf{B})\,V$. Equating the two
expressions and cancelling $V$ (true for any figure, so for one of nonzero area)
gives the claim. The signed version is consistent too, because each map
contributes its own orientation flip independently. $\blacksquare$

Two consequences follow without any further work. Taking
$\mathbf{B} = \mathbf{A}^{-1}$ in :eqref:`eq_mdl-det-multiplicative` and using
$\det(\mathbf{I}) = 1$ (the identity moves no volume) gives

$$
\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})},
$$

which also re-confirms the unifying theorem: an inverse can exist only when
$\det(\mathbf{A}) \neq 0$, since otherwise the right-hand side is undefined.
And looking ahead, multiplicativity is exactly what makes the determinant
factor cleanly through a diagonalization: once we can write a matrix in terms of
its eigenvalues in :numref:`sec_mdl-eigendecompositions`, this same identity will
show that the determinant is simply the *product of the eigenvalues*,
$\det(\mathbf{A}) = \prod_i \lambda_i$ — the volume scaling is just the product
of the per-axis stretch factors.

Computing determinants for larger matrices can be laborious,
but the  intuition is the same.
The determinant remains the factor
that $n\times n$ matrices scale $n$-dimensional volumes.

## Tensors and Einstein Summation

In :numref:`sec_linear-algebra` the concept of tensors was introduced.
In this section, we will dive more deeply into tensor contractions
(the tensor equivalent of matrix multiplication),
and see how it can provide a unified view
on a number of matrix and vector operations.

With matrices and vectors we knew how to multiply them to transform data.
We need to have a similar definition for tensors if they are to be useful to us.
Think about matrix multiplication:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

or equivalently

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

This pattern is one we can repeat for tensors.
For tensors, there is no one case of what
to sum over that can be universally chosen,
so we need specify exactly which indices we want to sum over.
For instance we could consider

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Such a transformation is called a *tensor contraction*.
It can represent a far more flexible family of transformations
that matrix multiplication alone.

As an often-used notational simplification,
we can notice that the sum is over exactly those indices
that occur more than once in the expression,
thus people often work with *Einstein notation*,
where the summation is implicitly taken over all repeated indices.
This gives the compact expression:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Common Examples from Linear Algebra

Let's see how many of the linear algebraic definitions
we have seen before can be expressed in this compressed tensor notation:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

In this way, we can replace a myriad of specialized notations with short tensor expressions.

### Expressing in Code
Tensors may flexibly be operated on in code as well.
As seen in :numref:`sec_linear-algebra`,
we can create tensors as is shown below.

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-1}
#@tab mxnet
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-1}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-1}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-1}
#@tab jax
# Define tensors
B = jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = jnp.array([[1, 2], [3, 4]])
v = jnp.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

Einstein summation has been implemented directly.
The indices that occur in the Einstein summation can be passed as a string,
followed by the tensors that are being acted upon.
For instance, to implement matrix multiplication,
we can consider the Einstein summation seen above
($\mathbf{A}\mathbf{v} = a_{ij}v_j$)
and strip out the indices themselves to get the implementation:

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab mxnet
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab jax
# Reimplement matrix multiplication
jnp.einsum("ij, j -> i", A, v), A @ v
```

This is a highly flexible notation.
For instance if we want to compute
what would be traditionally written as

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

it can be implemented via Einstein summation as:

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-3}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-3}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-3}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-3}
#@tab jax
jnp.einsum("ijk, il, j -> kl", B, A, v)
```

This notation is readable and efficient for humans,
however bulky if for whatever reason
we need to generate a tensor contraction programmatically.
For this reason, `einsum` provides an alternative notation
by providing integer indices for each tensor.
For example, the same tensor contraction can also be written as:

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-4}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-4}
#@tab pytorch
torch.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-4}
#@tab tensorflow
# TensorFlow does not support this type of notation.
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-4}
#@tab jax
jnp.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

Either notation allows for concise and efficient representation of tensor contractions in code.

## Summary
* Vectors can be interpreted geometrically as either points or directions in space.
* Dot products define the notion of angle to arbitrarily high-dimensional spaces.
* Hyperplanes are high-dimensional generalizations of lines and planes.  They can be used to define decision planes that are often used as the last step in a classification task.
* Matrix multiplication can be geometrically interpreted as uniform distortions of the underlying coordinates. They represent a very restricted, but mathematically clean, way to transform vectors.
* Linear dependence is a way to tell when a collection of vectors are in a lower dimensional space than we would expect (say you have $3$ vectors living in a $2$-dimensional space). The rank of a matrix is the size of the largest subset of its columns that are linearly independent.
* When a matrix's inverse is defined, matrix inversion allows us to find another matrix that undoes the action of the first. Matrix inversion is useful in theory, but requires care in practice owing to numerical instability.
* Determinants allow us to measure how much a matrix expands or contracts a space. A nonzero determinant implies an invertible (non-singular) matrix and a zero-valued determinant means that the matrix is non-invertible (singular).
* Tensor contractions and Einstein summation provide for a neat and clean notation for expressing many of the computations that are seen in machine learning.

## Exercises
1. What is the angle between
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. True or false: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ and $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ are inverses of one another?
3. Suppose that we draw a shape in the plane with area $100\textrm{m}^2$.  What is the area after transforming the figure by the matrix
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Which of the following sets of vectors are linearly independent?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Suppose that you have a matrix written as $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ for some choice of values $a, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0$?
6. The vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ are orthogonal.  What is the condition on a matrix $A$ so that $Ae_1$ and $Ae_2$ are orthogonal?
7. How can you write $\textrm{tr}(\mathbf{A}^4)$ in Einstein notation for an arbitrary matrix $A$?
8. Consider the hyperplane $\mathbf{w}\cdot\mathbf{x} = b$ with $\mathbf{w} = [3,4]^\top$ and $b = 10$.  What is the signed distance from the point $\mathbf{x} = [1,1]^\top$ to this hyperplane, and on which side of it does $\mathbf{x}$ lie?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1085)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1085)
:end_tab:

<!-- slides -->

::: {.slide title="Geometry of Linear Algebra"}
The geometric intuitions behind the linear algebra used
throughout the book. Two viewpoints on a vector $\mathbf{v}$:

- A *position* — a point in space.
- A *direction* — an arrow from the origin.

Most of deep learning works in the second view. From it
we get dot products (similarity), angles, projections,
hyperplanes (decision boundaries), and determinants
(volume changes).
:::

::: {.slide title="Vectors as geometry"}
The same array can name a point or a displacement. Deep learning
mostly uses the displacement view: directions, lengths, and angles.

@geometry-linear-algebraic-ops-geometry-of-vectors
:::

::: {.slide title="Dot products and angles"}
$\mathbf{u}^\top \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$.
Cosine similarity = normalized dot product. The metric
behind kernel methods, attention, and contrastive
learning:

@geometry-linear-algebraic-ops-dot-products-and-angles
:::

::: {.slide title="Hyperplanes as classifiers"}
A hyperplane is the set
$\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} = b\}$.
Linear classifiers split space with one — sign of the dot
product gives the prediction. Most of deep learning is
"learn good features so a hyperplane works":

@geometry-linear-algebraic-ops-hyperplanes-1

. . .

@geometry-linear-algebraic-ops-hyperplanes-2
:::

::: {.slide title="Hyperplanes (cont.)"}
Changing $\mathbf{w}$ rotates the boundary; changing $b$ shifts it.
Normalized distance to the boundary is a margin.

@geometry-linear-algebraic-ops-hyperplanes-3

. . .

@geometry-linear-algebraic-ops-hyperplanes-4
:::

::: {.slide title="Invertibility and determinant"}
Square matrices are invertible iff they don't collapse
volumes. The determinant measures the signed volume scale
factor:

@geometry-linear-algebraic-ops-invertibility

. . .

@geometry-linear-algebraic-ops-determinant
:::

::: {.slide title="In code"}
Translate all of this into NumPy / PyTorch:

@geometry-linear-algebraic-ops-expressing-in-code-1

. . .

@geometry-linear-algebraic-ops-expressing-in-code-2
:::

::: {.slide title="In code (cont.)"}
These final snippets connect the geometric ideas to the actual
linear-algebra APIs for norms, determinants, and inverses.

@geometry-linear-algebraic-ops-expressing-in-code-3

. . .

@geometry-linear-algebraic-ops-expressing-in-code-4
:::

::: {.slide title="Recap"}
- Vectors as directions; dot products = cosine
  similarity; matrices = linear maps; determinant =
  volume scale.
- Hyperplanes are the decision-boundary primitive of every
  linear classifier and every linear layer.
- These geometric pictures keep being useful all the way
  up to attention and high-dim embeddings.
:::
