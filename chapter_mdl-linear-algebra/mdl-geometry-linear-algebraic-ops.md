# Geometry and Linear Algebraic Operations
:label:`sec_mdl-geometry-linear-algebraic-ops`

In :numref:`sec_linear-algebra`, we encountered the basics of linear algebra
and saw how it could be used to express common operations for transforming our data.
Linear algebra is one of the pillars
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

A vector admits two common geometric readings, as a point or as a direction
in space. We use the following imports throughout the section.

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

Read as a *point*, a vector's components give its location relative to a
fixed reference called the *origin*: $\mathbf{v} = [3,2]^\top$ is the point
$3$ units to the right of the origin and $2$ units up. This reading turns a
dataset into a cloud of points, and a task such as telling pictures of cats
from pictures of dogs into geometry: find a way to separate two clusters of
points in space.

Read as a *direction*, the same vector says "take $3$ steps to the right and
$2$ steps up." An arrow with that displacement can start anywhere, so all the
parallel arrows in :numref:`fig_mdl-la-vectors` are the same vector. (On
notation: a single vector is by default a *column*, while in a matrix holding
a tabular dataset each data example is conventionally a *row*, as described in
:numref:`sec_linear-algebra`.)

![A vector can be read two ways: as a *point* whose first component is the $x$-coordinate and second is the $y$-coordinate (left), or as a *direction*, an arrow that can start anywhere, so every arrow shown is the same vector $(3,2)^\top$ (right).](../img/mdl-la-vectors.svg)
:label:`fig_mdl-la-vectors`

The direction view makes vector addition visual: follow one arrow, then the
other, tip to tail, as in :numref:`fig_mdl-la-vector-add`.

![We can visualize vector addition by first following one vector, and then another, placing them tip to tail.](../img/mdl-la-vector-add.svg)
:label:`fig_mdl-la-vector-add`

Vector subtraction has a similar interpretation: from the identity
$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$, the vector
$\mathbf{u}-\mathbf{v}$ is the direction that takes us from the point
$\mathbf{v}$ to the point $\mathbf{u}$.


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
norms, tells us the angle between them. The same identity holds for **any**
pair of vectors, in any number of dimensions. Two short arguments *justify*
the formula and pin down exactly when it makes sense.

The first argument is purely planar. Any two vectors $\mathbf{v}$ and
$\mathbf{w}$, however many coordinates they have, both lie in the
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
this is guaranteed, in every dimension, is the content of the following
inequality.

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
when they point in opposite directions, precisely the collinear cases of the
proposition.

Cauchy–Schwarz also has a one-picture summary, shown in
:numref:`fig_mdl-la-projection`. It is a way to *remember* the inequality
rather than a second proof, since the picture reads off the angle $\theta$
(and with it :eqref:`eq_mdl-dot_geom`) that Cauchy–Schwarz itself makes
legitimate.
On the left, the projection of $\mathbf{v}$
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

Cauchy–Schwarz immediately gives the **triangle inequality**, which says
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

angle(torch.tensor([0.0, 1, 2]), torch.tensor([2.0, 3, 4]))
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
though no angle is defined for it.) We will use this condition throughout the
chapter.

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
*least-squares* problem in one dimension: we found the best approximation of
$\mathbf{v}$ from the subspace spanned by $\mathbf{w}$. We state the general
version below, as soon as the vocabulary of subspaces and bases is in hand.

### Span, Bases, and Subspaces

The projection result spoke of "the one-dimensional subspace spanned by
$\mathbf{w}$," and the planar argument for the dot-product formula reasoned
inside the plane that two vectors span. These words (*span*, *subspace*,
*basis*) are the organizing vocabulary of linear algebra, and the
decompositions later in this chapter lean on them constantly, so let us pin
them down. 

Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$, their **span** is the set
of everything reachable by scaling and adding them:

$$
\operatorname{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)
 = \bigl\{ a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k
   : a_1, \ldots, a_k \in \mathbb{R} \bigr\}.
$$

A weighted sum $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$ is called a
*linear combination*, so the span is the set of all linear combinations. In
$\mathbb{R}^2$ the possibilities are easy to picture
(:numref:`fig_mdl-la-span`): the span of a single nonzero vector is the line
through the origin in its direction, while the span of two vectors that do not
lie on a common line is the entire plane. A span is closed under further
addition and scaling, and any set of vectors with that closure property is
called a **subspace**. The complete list of subspaces of $\mathbb{R}^2$ is
short: the origin alone, the lines through the origin, and $\mathbb{R}^2$
itself; in $\mathbb{R}^3$, the planes through the origin join the list. Note
that every subspace contains the origin (scale any of its elements by zero),
so a line that misses the origin is not a subspace.

![Left: the span of a single nonzero vector $\mathbf{v}$, the set of all of its scalar multiples, is the line through the origin in its direction, a one-dimensional subspace. Right: two vectors not on a common line span the whole plane; the dashed parallelogram construction resolves $\mathbf{x}$ as $2\mathbf{u} + \mathbf{w}$, coordinates that are unique because $\mathbf{u}$ and $\mathbf{w}$ are linearly independent.](../img/mdl-la-span.svg)
:label:`fig_mdl-la-span`

A spanning set can be wasteful. If one of the vectors already lies in the span
of the others, deleting it shrinks the list without shrinking the span. A
collection with no such redundancy (equivalently, one where the only linear
combination producing $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k = \mathbf{0}$
is the trivial one with every $a_i = 0$) is called **linearly independent**.
(We return to the redundant case, *linear dependence*, when we study matrices
and rank below.) A **basis** of a subspace is a linearly independent set that
spans it: enough vectors to reach everything, none to spare. The coordinate
vectors $\mathbf{e}_1 = [1, 0]^\top$ and $\mathbf{e}_2 = [0, 1]^\top$ form the
*standard basis* of $\mathbb{R}^2$, but the slanted pair in
:numref:`fig_mdl-la-span` is an equally valid basis. A fundamental theorem,
which we will use without proof, says that every basis of a given subspace has
the same number of elements, and that any linearly independent set can be
extended to a basis; these are the two facts we take on faith in this chapter.
The common count of basis elements is the subspace's
**dimension**. That gives, at last, a precise meaning to the $d$ in
"$d$-dimensional space": $\mathbb{R}^d$ has dimension $d$ because
$\mathbf{e}_1, \ldots, \mathbf{e}_d$ is a basis for it.

What independence buys is *coordinates*.

**Proposition (coordinates are unique).** *Let $\mathbf{v}_1, \ldots,
\mathbf{v}_k$ be a basis of a subspace $S$. Then every $\mathbf{x} \in S$ can
be written as $\mathbf{x} = a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$ for
exactly one choice of coefficients $a_1, \ldots, a_k$.*

**Proof.** At least one representation exists because the basis spans $S$. If
there were two, say
$\mathbf{x} = \sum_i a_i \mathbf{v}_i = \sum_i b_i \mathbf{v}_i$, then
subtracting gives $\sum_i (a_i - b_i)\,\mathbf{v}_i = \mathbf{0}$, and linear
independence forces $a_i = b_i$ for every $i$. $\blacksquare$

A basis therefore turns an abstract subspace into a concrete copy of
$\mathbb{R}^k$: once the basis is agreed upon, the coefficient list
$(a_1, \ldots, a_k)$ *is* the point. Much of applied linear algebra is the art
of choosing a basis in which a problem's coordinates become simple; the
eigenvector and singular-vector bases of the next two sections are chosen for
exactly this reason.

Finally, two subspaces attach to every matrix $\mathbf{A}$, and they organize
everything matrices do in the remainder of this chapter. The **column space**
of $\mathbf{A}$ is the span of its columns; since the product
$\mathbf{A}\mathbf{v}$ is exactly the linear combination of the columns of
$\mathbf{A}$ weighted by the entries of $\mathbf{v}$ (a fact we put to work
when we take up matrices as maps below), the column space is the set of all
possible *outputs* of $\mathbf{A}$. The **null space** of $\mathbf{A}$ is the
set of inputs sent to zero, $\{\mathbf{x} : \mathbf{A}\mathbf{x} =
\mathbf{0}\}$; it is a subspace because if $\mathbf{A}\mathbf{x} =
\mathbf{A}\mathbf{y} = \mathbf{0}$ then $\mathbf{A}(a\mathbf{x} + b\mathbf{y})
= a\,\mathbf{A}\mathbf{x} + b\,\mathbf{A}\mathbf{y} = \mathbf{0}$. In words:
the column space records what a matrix can produce, and the null space records
what it destroys. We will measure the first when we define the *rank* below,
and meet both again among the SVD's *four fundamental subspaces* in
:numref:`sec_mdl-svd-low-rank`.

### Projection onto a Subspace

The projection proposition dropped $\mathbf{v}$ onto a line. With subspaces
and bases in hand, we can state the general version promised there: the same
formula, one matrix heavier, drops a vector onto an arbitrary subspace. Call a
basis $\mathbf{q}_1, \ldots, \mathbf{q}_k$ of a subspace $S \subseteq
\mathbb{R}^n$ **orthonormal** when each vector has unit length and distinct
vectors are orthogonal; compactly, $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}_k$,
where $\mathbf{Q}$ is the $n \times k$ matrix with the $\mathbf{q}_i$ as
columns.

**Proposition (projection onto a subspace).** *Let $\mathbf{Q}$ have
orthonormal columns spanning $S$, and set $\mathbf{P} =
\mathbf{Q}\mathbf{Q}^\top$. Then for every $\mathbf{x} \in \mathbb{R}^n$:*
(i) *$\mathbf{P}\mathbf{x} \in S$, and the residual $\mathbf{x} -
\mathbf{P}\mathbf{x}$ is orthogonal to every vector of $S$;*
(ii) *$\mathbf{P}\mathbf{x}$ is the unique closest point of $S$ to
$\mathbf{x}$;*
(iii) *$\mathbf{P}^2 = \mathbf{P}$ and $\mathbf{P}^\top = \mathbf{P}$.*

**Proof.** For (i), $\mathbf{P}\mathbf{x} = \mathbf{Q}(\mathbf{Q}^\top\mathbf{x})$
is a linear combination of the columns of $\mathbf{Q}$, hence lies in $S$; and
$$\mathbf{Q}^\top(\mathbf{x} - \mathbf{P}\mathbf{x}) = \mathbf{Q}^\top\mathbf{x}
- (\mathbf{Q}^\top\mathbf{Q})\mathbf{Q}^\top\mathbf{x} = \mathbf{0},$$ 
so the
residual is orthogonal to each $\mathbf{q}_i$ and therefore to everything they
span. For (ii), take any $\mathbf{s} \in S$ and split $\mathbf{x} - \mathbf{s}
= (\mathbf{x} - \mathbf{P}\mathbf{x}) + (\mathbf{P}\mathbf{x} - \mathbf{s})$:
the second piece lies in $S$, the first is orthogonal to it, so Pythagoras
:eqref:`eq_mdl-pythagoras` gives 
$$\|\mathbf{x} - \mathbf{s}\|^2 = \|\mathbf{x}
- \mathbf{P}\mathbf{x}\|^2 + \|\mathbf{P}\mathbf{x} - \mathbf{s}\|^2 \ge
\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2,$$ 
with equality exactly when
$\mathbf{s} = \mathbf{P}\mathbf{x}$. For (iii), $\mathbf{P}^2 =
\mathbf{Q}(\mathbf{Q}^\top\mathbf{Q})\mathbf{Q}^\top = \mathbf{Q}\mathbf{Q}^\top
= \mathbf{P}$, and symmetry is immediate from the form
$\mathbf{Q}\mathbf{Q}^\top$. $\blacksquare$

Property (iii) restates the geometry algebraically: projecting a second time
changes nothing, because after the first projection you are already in $S$.
For $k = 1$ with the single unit column
$\mathbf{q} = \mathbf{w}/\|\mathbf{w}\|$, the matrix
$\mathbf{P} = \mathbf{q}\mathbf{q}^\top$ reproduces the one-dimensional formula
:eqref:`eq_mdl-projection` exactly.

This proposition is the geometry of **least squares**, the fitting problem
behind linear regression. Given a data matrix $\mathbf{A}$ whose columns are
features and a target vector $\mathbf{b}$, least squares asks for

$$
\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 .
$$
:eqlabel:`eq_mdl-least-squares`

As $\mathbf{x}$ ranges over all coefficient vectors, $\mathbf{A}\mathbf{x}$
ranges over the column space of $\mathbf{A}$, so minimizing
:eqref:`eq_mdl-least-squares` means finding the closest point to $\mathbf{b}$
in that column space, and by property (i) the optimal residual is orthogonal
to every column of $\mathbf{A}$. The singular value decomposition solves
:eqref:`eq_mdl-least-squares` in full generality in
:numref:`sec_mdl-svd-low-rank`.

Where does an orthonormal basis come from? The library's `qr` routine (the
*QR factorization*, a matrix form of the Gram–Schmidt process we describe when
we meet orthogonal matrices below) turns any full-rank matrix into one with
orthonormal columns and the same column space. Let us verify the proposition
numerically on a random 3-dimensional subspace of $\mathbb{R}^5$.

```{.python .input #mdl-geometry-linear-algebraic-ops-projection-onto-a-subspace}
#@tab mxnet
# A random 5x3 matrix whose columns span a 3-dim subspace of R^5
np.random.seed(0)
A = np.random.randn(5, 3)
Q, _ = np.linalg.qr(A)  # orthonormal basis of the column space
P = Q.dot(Q.T)          # projection matrix onto that subspace
x = np.random.randn(5)
r = x - P.dot(x)        # residual
(np.linalg.norm(P.dot(P) - P),  # P^2 = P (idempotent)
 np.linalg.norm(Q.T.dot(r)))    # residual is orthogonal to the subspace
```

```{.python .input #mdl-geometry-linear-algebraic-ops-projection-onto-a-subspace}
#@tab pytorch
# A random 5x3 matrix whose columns span a 3-dim subspace of R^5
torch.manual_seed(0)
A = torch.randn(5, 3)
Q, _ = torch.linalg.qr(A)  # orthonormal basis of the column space
P = Q @ Q.T                # projection matrix onto that subspace
x = torch.randn(5)
r = x - P @ x              # residual
(torch.norm(P @ P - P),    # P^2 = P (idempotent)
 torch.norm(Q.T @ r))      # residual is orthogonal to the subspace
```

```{.python .input #mdl-geometry-linear-algebraic-ops-projection-onto-a-subspace}
#@tab tensorflow
# A random 5x3 matrix whose columns span a 3-dim subspace of R^5
tf.random.set_seed(0)
A = tf.random.normal((5, 3))
Q, _ = tf.linalg.qr(A)     # orthonormal basis of the column space
P = Q @ tf.transpose(Q)    # projection matrix onto that subspace
x = tf.random.normal((5,))
r = x - tf.linalg.matvec(P, x)  # residual
(tf.norm(P @ P - P),       # P^2 = P (idempotent)
 tf.norm(tf.linalg.matvec(tf.transpose(Q), r)))  # residual orthogonal to S
```

```{.python .input #mdl-geometry-linear-algebraic-ops-projection-onto-a-subspace}
#@tab jax
# A random 5x3 matrix whose columns span a 3-dim subspace of R^5
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
A = jax.random.normal(key1, (5, 3))
Q, _ = jnp.linalg.qr(A)    # orthonormal basis of the column space
P = Q @ Q.T                # projection matrix onto that subspace
x = jax.random.normal(key2, (5,))
r = x - P @ x              # residual
(jnp.linalg.norm(P @ P - P),  # P^2 = P (idempotent)
 jnp.linalg.norm(Q.T @ r))    # residual is orthogonal to the subspace
```

Both numbers are zero up to floating-point roundoff: $\mathbf{P}$ really is
idempotent, and the residual really is orthogonal to the whole subspace, just
as the proposition demands.

## Similarity in High Dimensions

Why is the *angle*, rather than the raw distance, so often the right notion
of similarity? The answer is invariance to scale.
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
**contrastive learning** :cite:`Oord.Li.Vinyals.2018`: in each case we have
represented objects as vectors and we measure relatedness by direction,
deliberately discarding magnitude.

If we drop two *unrelated* vectors into a high-dimensional space, what cosine
should we expect between them? **In high dimensions, random vectors are almost
always nearly orthogonal**, and the higher the dimension, the more sharply
this holds.

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
inequality, $\Pr(|X - \mathbb{E}X| \ge \varepsilon) \le
\operatorname{Var}(X)/\varepsilon^2$ (we prove it in :numref:`sec_mdl-random_variables`),
then bounds the chance of a large cosine,

$$
\Pr\bigl(|\cos\theta| \ge \varepsilon\bigr) \le \frac{1}{d\,\varepsilon^2},
$$

which tends to $0$ as $d \to \infty$ for any fixed $\varepsilon > 0$.
$\blacksquare$

So the typical cosine between random directions has standard deviation
$1/\sqrt{d}$, concentrating ever more tightly at $0$. This is a first taste of
*concentration of measure*, the phenomenon that makes high-dimensional geometry
behave very differently from our $2$- and $3$-dimensional intuition; we
quantify it with exponential tail bounds in
:numref:`sec_mdl-concentration-generalization`. It is also
why cosine similarity is a useful signal: since unrelated items are nearly
orthogonal by default, a cosine that is appreciably above $0$ is unlikely to
be an accident and instead reflects real shared structure. This is the
working assumption behind embedding-based retrieval and the attention
mechanism.

We can watch the concentration happen by sampling random unit vectors and
histogramming their pairwise cosines as the dimension grows, shown in
:numref:`fig_mdl-la-cosine-highd`. Very low dimensions are genuinely
different: in the plane ($d = 2$) the histogram piles up at $\pm 1$ (the
density is arcsine-shaped), so two random directions are *more* likely to be
nearly aligned or nearly opposed than nearly orthogonal. Raising the dimension
reverses the picture: by $d = 10$ the histogram is a bell centered at $0$ with
standard deviation $1/\sqrt{10} \approx 0.32$, and by $d = 1000$ it has
collapsed into a spike of width $\approx 0.03$.

![Histograms of the cosine between independent random unit vectors at dimensions $d = 2$, $10$, and $1000$. In $d = 2$ the mass piles up near $\pm 1$ in an arcsine-shaped density: random directions in the plane are typically far from orthogonal. In moderate dimension the histogram flattens into a bell around $0$, and by $d = 1000$ it is a narrow spike of width $\approx 1/\sqrt{d}$, exactly as the proposition predicts.](../img/mdl-la-cosine-highd.svg)
:label:`fig_mdl-la-cosine-highd`

This concentration is the operating environment of **dot-product attention**
(:numref:`sec_attention-scoring-functions`). An
attention layer compares one query $\mathbf{q}$ against thousands of keys
$\mathbf{k}_1, \mathbf{k}_2, \ldots$ by dot product, and near-orthogonality is
what makes the comparison informative at scale: the scores of the many
unrelated keys concentrate near zero, so the few keys that genuinely share
structure with the query stand out against a quiet background. A short
variance computation also explains the otherwise mysterious $\sqrt{d}$ in the
attention scores $\mathbf{Q}\mathbf{K}^\top/\sqrt{d}$
:cite:`Vaswani.Shazeer.Parmar.ea.2017`. Let the query and key have
independent, mean-zero, unit-variance entries. Each summand of
$\mathbf{q}\cdot\mathbf{k} = \sum_i q_i k_i$ then has mean zero and variance
$\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$, and
distinct summands are uncorrelated, since independence gives
$\mathbb{E}[(q_i k_i)(q_j k_j)] = 0$ for $i \neq j$. The variances of the $d$
summands therefore add:

$$
\operatorname{Var}(\mathbf{q}\cdot\mathbf{k})
 = \sum_{i=1}^d \operatorname{Var}(q_i k_i) = d.
$$

The raw scores have standard deviation $\sqrt{d}$, growing with the dimension,
and dividing by $\sqrt{d}$ is exactly dividing by that standard deviation: it
standardizes the scores to size $O(1)$ so that the softmax downstream stays in
its responsive range instead of saturating on whichever score happens to be
largest (we study that softmax's derivative in
:numref:`sec_mdl-matrix-calculus-autodiff`). The sphere proposition gives the
same answer as geometric intuition: each vector has length about $\sqrt{d}$
while the cosine between unrelated directions has typical size $1/\sqrt{d}$,
so the product lands at size $\sqrt{d}$.

## Hyperplanes and Decision Boundaries

In addition to working with vectors, another key object
that you must understand
is the *hyperplane*, a generalization to higher dimensions
of a line (two dimensions) or of a plane (three dimensions).
In a $d$-dimensional vector space, a hyperplane has $d-1$ dimensions
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
onto the direction of $\mathbf{w}$ is exactly $1/\|\mathbf{w}\|$; recall the
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
projection material had to come first. :numref:`fig_mdl-la-hyperplane`
collects all of these facts in a single picture.

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

Higher dimensions work the same way.
If we now take $\mathbf{w} = [1,2,3]^\top$
and ask about the points in three dimensions with $\mathbf{w}\cdot\mathbf{v} = b$,
we obtain a plane at right angles to the given vector $\mathbf{w}$,
offset from the origin by the signed distance $b/\|\mathbf{w}\|$.
The two inequalities again define the two sides of the plane.

While our ability to visualize runs out at this point,
nothing stops us from doing this in tens, hundreds, or billions of dimensions (or even infinite dimensional spaces in the case of kernel methods).
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
can be separated by hyperplanes.

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
# many fused kernels still emit float32; pin everything to float32 up front so
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
    [x[0] * 255 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 255 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 255 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
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

The two class means are blurry but recognizable images; let's plot them side
by side.

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab mxnet, pytorch
# Plot the two class means side by side
d2l.set_figsize((6, 3))
_, axes = d2l.plt.subplots(1, 2)
axes[0].imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
axes[0].set_title('mean t-shirt')
axes[1].imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
axes[1].set_title('mean trousers')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab tensorflow
# Plot the two class means side by side
d2l.set_figsize((6, 3))
_, axes = d2l.plt.subplots(1, 2)
axes[0].imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
axes[0].set_title('mean t-shirt')
axes[1].imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
axes[1].set_title('mean trousers')
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-hyperplanes-2}
#@tab jax
# Plot the two class means side by side
d2l.set_figsize((6, 3))
_, axes = d2l.plt.subplots(1, 2)
axes[0].imshow(np.array(ave_0.reshape(28, 28)), cmap='Greys')
axes[0].set_title('mean t-shirt')
axes[1].imshow(np.array(ave_1.reshape(28, 28)), cmap='Greys')
axes[1].set_title('mean trousers')
d2l.plt.show()
```

In a fully machine learned solution, we would learn the threshold from the
dataset. Here we set it geometrically instead: the normal is the difference of
the two class means $\mathbf{w} = \overline{\mathbf{x}}_1 - \overline{\mathbf{x}}_0$,
and the natural decision boundary is the hyperplane that *bisects* the two
means: $\mathbf{w}\cdot\mathbf{x} = b$ with
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

This rule classifies about $92\%$ of the
$2{,}000$ test images correctly, and *nothing was trained*. We computed two
class means, took their difference as the normal $\mathbf{w}$, and asked of
each test image only which side of one hyperplane its $784$-dimensional pixel
vector lies on. To see the geometry of why such a crude rule works, project
every test image onto the normal: reduce each image to the single
number $\mathbf{w}\cdot\mathbf{x}$, its (scaled) signed position along the
direction from "mean t-shirt" to "mean trousers", and histogram the two
classes separately.

```{.python .input #geometry-linear-algebraic-ops-projection-histogram}
#@tab mxnet
# Histogram of the test images' projections onto the normal direction w
proj = X_test.reshape(2000, -1).dot(w)
d2l.set_figsize()
d2l.plt.hist(proj[y_test == 0].asnumpy(), bins=50, alpha=0.6,
             label='t-shirts')
d2l.plt.hist(proj[y_test == 1].asnumpy(), bins=50, alpha=0.6,
             label='trousers')
d2l.plt.axvline(float(b), color='black', linestyle='--', label='threshold')
d2l.plt.xlabel(r'$\mathbf{w}\cdot\mathbf{x}$')
d2l.plt.legend()
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-projection-histogram}
#@tab pytorch
# Histogram of the test images' projections onto the normal direction w
proj = X_test.reshape(2000, -1) @ w
d2l.set_figsize()
d2l.plt.hist(proj[y_test == 0].numpy(), bins=50, alpha=0.6,
             label='t-shirts')
d2l.plt.hist(proj[y_test == 1].numpy(), bins=50, alpha=0.6,
             label='trousers')
d2l.plt.axvline(float(b), color='black', linestyle='--', label='threshold')
d2l.plt.xlabel(r'$\mathbf{w}\cdot\mathbf{x}$')
d2l.plt.legend()
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-projection-histogram}
#@tab tensorflow
# Histogram of the test images' projections onto the normal direction w
proj = tf.linalg.matvec(tf.reshape(X_test, (2000, -1)), w)
d2l.set_figsize()
d2l.plt.hist(tf.boolean_mask(proj, y_test == 0).numpy(), bins=50, alpha=0.6,
             label='t-shirts')
d2l.plt.hist(tf.boolean_mask(proj, y_test == 1).numpy(), bins=50, alpha=0.6,
             label='trousers')
d2l.plt.axvline(float(b), color='black', linestyle='--', label='threshold')
d2l.plt.xlabel(r'$\mathbf{w}\cdot\mathbf{x}$')
d2l.plt.legend()
d2l.plt.show()
```

```{.python .input #geometry-linear-algebraic-ops-projection-histogram}
#@tab jax
# Histogram of the test images' projections onto the normal direction w
proj = X_test.reshape(2000, -1) @ w
d2l.set_figsize()
d2l.plt.hist(np.array(proj[y_test == 0]), bins=50, alpha=0.6,
             label='t-shirts')
d2l.plt.hist(np.array(proj[y_test == 1]), bins=50, alpha=0.6,
             label='trousers')
d2l.plt.axvline(float(b), color='black', linestyle='--', label='threshold')
d2l.plt.xlabel(r'$\mathbf{w}\cdot\mathbf{x}$')
d2l.plt.legend()
d2l.plt.show()
```

This plot summarizes the key points of hyperplane classifiers. Along the single direction
$\mathbf{w}$, the two classes form two well-separated humps, and the dashed
threshold (the value of $\mathbf{w}\cdot\mathbf{x}$ at the midpoint of the
two means) cuts between them; the tails that spill across it are exactly the
$\approx 8\%$ of images the rule misclassifies. A *learned* linear classifier, such as the softmax
regression of :numref:`sec_softmax`, improves on this only by moving and
tilting the same kind of boundary to cut the overlap more cleverly. A deep
network goes one step further: it learns a new representation under which the
two humps separate so widely that a hyperplane between them becomes trivial to
place.

## Matrices as Linear Maps

### Linear Transformations

Through :numref:`sec_linear-algebra` and the above discussions,
we have a solid understanding of the geometry of vectors, lengths, and angles.
The remaining object is a geometric understanding of the linear
transformations that matrices represent, and two dimensions are the place to
build it.

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
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} 
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} 
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

This may seem like an odd computation,
where something clear became somewhat impenetrable.
However, it tells us that we can write the way
that a matrix transforms *any* vector
in terms of how it transforms *two specific vectors*:
$[1,0]^\top$ and $[0,1]^\top$.
We have reduced an infinite problem
(what happens to any pair of real numbers)
to a finite one (what happens to these specific vectors).
The vectors $[1,0]^\top$ and $[0,1]^\top$ are exactly the standard basis
$\mathbf{e}_1, \mathbf{e}_2$ from our discussion of spans and bases: because
every vector is a (unique) weighted sum of basis vectors, knowing where a
matrix sends a basis determines where it sends everything.

Let's draw what happens when we use the specific matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

If we look at the specific vector $\mathbf{v} = [2, -1]^\top$,
we see this is $2\cdot[1,0]^\top - [0,1]^\top$,
and thus we know that the matrix $\mathbf{A}$ will send this to
$2\,\mathbf{A}[1,0]^\top - \mathbf{A}[0,1]^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
If we follow this logic through carefully,
say by considering the grid of all integer pairs of points,
we see that what happens is that the matrix multiplication
can skew, rotate, and scale the grid,
but the grid structure must remain as you see in :numref:`fig_mdl-la-linear-map`.

![The matrix $\mathbf{A} = \bigl(\begin{smallmatrix}1 & 2 \\ -1 & 3\end{smallmatrix}\bigr)$ acting on the plane. The basis vectors are sent to $\mathbf{A}\mathbf{e}_1 = (1, -1)^\top$ and $\mathbf{A}\mathbf{e}_2 = (2, 3)^\top$, and the entire grid is transported along with them: lines stay lines, the origin stays put, and equally spaced cells stay equally spaced. The shaded unit square maps to the shaded parallelogram, whose area will be the subject of the determinant below.](../img/mdl-la-linear-map.svg)
:label:`fig_mdl-la-linear-map`

The point to internalize about linear transformations represented by matrices:
matrices are incapable of distorting some parts of space differently than others.
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
because we will never know where the vector $[1,2]^\top$ came from: was
it $[1,1]^\top$ or $[0, -1]^\top$?

While this picture was for a $2\times2$ matrix,
nothing prevents us from taking the lessons learned into higher dimensions.
If we take similar basis vectors like $[1,0, \ldots,0]$
and see where our matrix sends them,
we can start to get a feeling for how the matrix multiplication
distorts the entire space in whatever dimension space we are dealing with.

Nothing requires the matrix to be square, either. An $m \times n$ matrix takes
vectors with $n$ entries to vectors with $m$ entries: it is a linear map
*between* spaces, from $\mathbb{R}^n$ to $\mathbb{R}^m$, and it is still
determined by where it sends the $n$ basis vectors (whose images are its
columns). A $2 \times 3$ matrix flattens three-dimensional space onto a plane;
a $3 \times 2$ matrix lays the plane into three-dimensional space as a
(generally tilted) plane through the origin. Every fully connected layer of a
neural network is exactly such a map between spaces of different dimensions,
composed with a nonlinearity.

### Orthogonal Matrices

A matrix may skew, rotate, and scale, but a special and important family does
*only* the rigid part: it rotates or reflects without any stretching. A square
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
\mathbf{Q}^\top$, such maps are trivially invertible, and as we will prove
when we meet the determinant at the end of this section, their volume scaling
is $\det\mathbf{Q} = \pm 1$ (the sign distinguishing rotations from
reflections). Orthogonal matrices are the "distortion-free" linear maps, and
they will turn out to be the building blocks
of the two decompositions in the sections that follow: the spectral theorem
writes a symmetric matrix as $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$
(:numref:`sec_mdl-eigendecompositions`), and the singular value decomposition
writes *any* matrix as orthogonal–diagonal–orthogonal
(:numref:`sec_mdl-svd-low-rank`).

Where do orthonormal columns come from in the first place? Any linearly
independent collection can be converted into an orthonormal basis of its span
by the *Gram–Schmidt process*: walk through the vectors in order, subtract
from each one its projection :eqref:`eq_mdl-projection` onto each direction
already produced, and normalize what remains. In matrix form this algorithm is
the *QR factorization*, the `qr` call that produced the orthonormal basis
in our subspace-projection demo earlier in this section. We will not need its
inner workings in this chapter; it is enough to know that orthonormal bases
are cheap to compute, which is one reason the decompositions built from
them are so practical.

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
as a linear combination of the columns of the matrix,
like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$;
in the language of spans, the outputs of $\mathbf{B}$ fill out exactly its
column space.
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

The coefficients $(1, 2)$ of this relation say
precisely that $\mathbf{B}[1, 2]^\top = 1\cdot\mathbf{b}_1 + 2\cdot\mathbf{b}_2
= \mathbf{0}$, so the dependence hands us a nonzero vector in the *null space*
of $\mathbf{B}$. :numref:`fig_mdl-la-null-collapse` shows the two subspaces of
$\mathbf{B}$ at work together: the whole plane is crushed onto the column
space, while the null-space direction is exactly the set of inputs that land
on the origin. (For this particular $\mathbf{B}$ the two subspaces happen to
coincide as sets, both being the line $y = 2x$: a coincidence, equivalent
to $\mathbf{B}^2 = \mathbf{0}$, that makes the picture no less instructive,
since what the matrix *produces* and what it *destroys* are different roles
even when they occupy the same line.)

![The matrix $\mathbf{B} = \left(\begin{smallmatrix}2 & -1 \\ 4 & -2\end{smallmatrix}\right)$ collapsing the plane. Left: the input plane, with three marked points and the null-space direction $(1,2)^\top$ dashed; every input on that line is sent to the origin. Right: the image. The entire grid lands on the column space, the line $y = 2x$ spanned by the columns $\mathbf{b}_1$ and $\mathbf{b}_2$; the marked points land at their images, and the null-space line has collapsed to $\mathbf{0}$. Once distinct inputs collide like this, no inverse can tell them apart.](../img/mdl-la-null-collapse.svg)
:label:`fig_mdl-la-null-collapse`

In general, we will say that a collection of vectors
$\mathbf{v}_1, \ldots, \mathbf{v}_k$ are *linearly dependent*
if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that

$$
\sum_{i=1}^k a_i\mathbf{v}_i = 0.
$$

In this case, we can solve for one of the vectors
in terms of some combination of the others,
and effectively render it redundant.
Thus, a linear dependence in the columns of a matrix
is a witness to the fact that our matrix
is compressing the space down to some lower dimension.
If there is no linear dependence we say the vectors are *linearly
independent*, the same notion we met when defining bases, now read as a
property of a matrix's columns. If the columns of a matrix are linearly
independent, no compression occurs and the operation can be undone.

#### Rank

If we have a general $m\times n$ matrix,
it is reasonable to ask the dimension of the space the matrix maps *onto*,
the dimension of its image.
A concept known as the *rank* will be our answer.
In the previous section, we noted that a linear dependence
bears witness to compression of space into a lower dimension
and so we will be able to use this to define the notion of rank.
In particular, the rank of a matrix $\mathbf{A}$
is the largest number of linearly independent columns
amongst all subsets of columns. For example, our matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}
$$

from above has $\textrm{rank}(\mathbf{B})=1$, since the two columns
$[2, 4]^\top$ and $[-1, -2]^\top$ are linearly dependent,
while each column on its own is linearly independent.
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
however any of the $\binom{5}{3} = 10$ collections of three columns are
linearly dependent. (Exercise: verify this, either by hand or with
`matrix_rank`.)

Equivalently, the rank is the *dimension of the column space* we defined
alongside spans and bases, and a foundational theorem of linear algebra says
this equals the dimension of the *row space*, the span of the rows. A matrix
"compresses space" into a lower dimension exactly when its rank is smaller
than its number of columns, equivalently, when its null space contains some
nonzero vector.

This procedure, as described, is very inefficient.
It requires looking at every subset of the columns of our given matrix,
and thus is potentially exponential in the number of columns.
Later we will see a more computationally efficient way
to compute the rank of a matrix, but for now,
this is sufficient to see that the concept
is well defined and understand the meaning.

The trade-off between compression and survival is exact.

**Proposition (rank--nullity).** *For an $m \times n$ matrix $\mathbf{A}$,*

$$
\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n,
$$

*where $\ker\mathbf{A}$ denotes the null space.*

**Proof.** Choose a basis $\mathbf{w}_1, \ldots, \mathbf{w}_k$ of the null
space and extend it to a basis $\mathbf{w}_1, \ldots, \mathbf{w}_k,
\mathbf{v}_1, \ldots, \mathbf{v}_{n-k}$ of $\mathbb{R}^n$. The images
$\mathbf{A}\mathbf{v}_1, \ldots, \mathbf{A}\mathbf{v}_{n-k}$ span the column
space (the $\mathbf{w}_i$ contribute nothing), and they are independent: if
$\sum_i c_i \mathbf{A}\mathbf{v}_i = \mathbf{0}$ then $\sum_i c_i \mathbf{v}_i$
lies in the null space, hence is a combination of the $\mathbf{w}_j$,
impossible within a basis unless every $c_i = 0$. So the column space has
dimension exactly $n - k$. $\blacksquare$

Read it as conservation of directions: of the $n$ directions coming in,
$\dim\ker\mathbf{A}$ are destroyed and $\operatorname{rank}\mathbf{A}$
survive. $\mathbf{B}$ has $1 + 1 = 2$, one column-space direction
surviving and one null-space
direction crushed, exactly as :numref:`fig_mdl-la-null-collapse` shows. The
singular value decomposition will make this split visible, with orthonormal
bases attached to both halves, in :numref:`sec_mdl-svd-low-rank`.

#### Invertibility

We have seen above that multiplication by a matrix with linearly dependent columns
cannot be undone, i.e., there is no inverse operation that can always recover the input.  However, multiplication by a *full-rank* matrix
(an $n \times n$ matrix $\mathbf{A}$ with rank $n$)
can always be undone.  Consider the matrix

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
"Generically" is made precise by the *determinant*, introduced below: a
solution exists exactly when the determinant is nonzero.
We call such a matrix $\mathbf{A}^{-1}$ the *inverse* matrix.
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
While the matrix inverse is useful in theory, in practice we rarely want to
*compute* it. To solve the linear system $\mathbf{A}\mathbf{x} = \mathbf{b}$,
call `linalg.solve(A, b)`, which factorizes $\mathbf{A}$ without ever forming
its inverse, rather than the tempting but inferior `inv(A) @ b`: just as
division by a small number is numerically unstable, so is inversion of a
matrix that is close to having low rank. Sparsity raises the stakes further.
A matrix with a million rows and columns but only $5$ million non-zero entries
is cheap to store, yet its inverse is typically dense, with on the order of
$10^{12}$ non-zero entries. The quantity that makes "close to low rank"
precise, and that measures exactly how much a solve can amplify error, is the
condition number of :numref:`subsec_mdl-condition-number`.

### The Determinant
The geometric view of linear algebra gives an intuitive way
to interpret a fundamental quantity known as the *determinant*.
Return to the grid picture of :numref:`fig_mdl-la-linear-map` and watch the
shaded unit square, the square with edges $\mathbf{e}_1 = (1,0)^\top$ and
$\mathbf{e}_2 = (0,1)^\top$, hence with area one. The matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}
$$

carries it to the shaded parallelogram with edges
$\mathbf{A}\mathbf{e}_1 = [1, -1]^\top$ and $\mathbf{A}\mathbf{e}_2 = [2, 3]^\top$.
There is no reason this parallelogram should have the same area
that we started with, and indeed an exercise in coordinate geometry
shows that its area is exactly $5$: this particular matrix
quintuples areas.

In general, a matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

sends the unit square to the parallelogram with edges $(a, c)^\top$ and
$(b, d)^\top$, and we can compute that parallelogram's area directly. Suppose
for the moment that all four entries are positive and that $(a, c)^\top$ makes
the smaller angle with the $x$-axis. The parallelogram has vertices $(0,0)$,
$(a, c)$, $(b, d)$, and $(a+b, c+d)$; enclose it in its bounding rectangle
$[0, a+b] \times [0, c+d]$. The part of the rectangle outside the
parallelogram consists of two $b \times c$ rectangles in the corners, two
right triangles with legs $a$ and $c$, and two right triangles with legs $b$
and $d$. Subtracting all six pieces from the rectangle's area leaves

$$
(a+b)(c+d) - 2bc - 2\cdot\tfrac{1}{2}\,ac - 2\cdot\tfrac{1}{2}\,bd = ad - bc.
$$

Working through the other sign configurations changes only the sign of the
result, never its magnitude: the parallelogram's area is always $|ad - bc|$.
The signed quantity $ad-bc$
is referred to as the *determinant*, written $\det\mathbf{A}$.

Let's quickly confirm the worked example in code: for the matrix
$\mathbf{A}$ above the determinant should come out to
$1 \cdot 3 - 2 \cdot (-1) = 5$.

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab mxnet
np.linalg.det(np.array([[1, 2], [-1, 3]]))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab pytorch
torch.det(torch.tensor([[1, 2], [-1, 3]], dtype=torch.float32))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, 2], [-1, 3]], dtype=tf.float32))
```

```{.python .input #geometry-linear-algebraic-ops-determinant}
#@tab jax
jnp.linalg.det(jnp.array([[1, 2], [-1, 3]], dtype=jnp.float32))
```

Note that the expression $ad - bc$ can be zero or even negative, and
:numref:`fig_mdl-la-determinant` shows the three possible regimes side by
side, each for a different matrix: a positive determinant, where the unit
square maps to a parallelogram of area $\det\mathbf{A}$; a negative
determinant, where the parallelogram has area $|\det\mathbf{A}|$ but the map
has *flipped the orientation* of the plane (the images of $\mathbf{e}_1$ and
$\mathbf{e}_2$ have traded sides); and a zero determinant, where the square is
crushed to a segment. The negative case is a matter of convention taken
generally in mathematics: if the matrix flips the figure, we say the area is
negated.

![The determinant as a signed area, shown for three different matrices. The unit square spanned by the basis vectors (dashed) maps to the parallelogram spanned by the columns of each matrix, and the signed area of that parallelogram is the determinant: (a) positive; (b) negative, because the matrix flips orientation; (c) zero, because the matrix collapses the square to a segment.](../img/mdl-la-determinant.svg)
:label:`fig_mdl-la-determinant`

Let's see now that when the determinant is zero, we learn more.
Consider our compressing matrix from before,

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

If we compute the determinant of this matrix,
we get $2\cdot(-2) - (-1)\cdot 4 = 0$.
Given our understanding above, this makes sense.
$\mathbf{B}$ compresses the unit square
down to a segment of the line $y = 2x$, which has zero area:
the situation of panel (c) in :numref:`fig_mdl-la-determinant`, and exactly
the collapse pictured in :numref:`fig_mdl-la-null-collapse`.
And indeed, being compressed into a lower dimensional space
is the only way to have zero area after the transformation.
Thus we see the following result is true:
a matrix $A$ is invertible if and only if
the determinant is not equal to zero. We prove this equivalence below, once
the determinant exists in every dimension.

#### The Determinant in General
:label:`subsec_mdl-determinant-general`

The two-dimensional picture says what the determinant of an $n \times n$
matrix should be: the signed volume of the parallelepiped spanned by its
columns. Rather than wrestle with $n$-dimensional geometry directly, we
distill the three properties of signed volume that determine it completely,
stated for the columns $\mathbf{a}_1, \ldots, \mathbf{a}_n$ of an
$n \times n$ matrix:

1. **Multilinear**: $\det$ is linear in each column separately, with the other
   columns held fixed. (Scaling one edge of a parallelepiped scales its
   volume; splitting one edge as a sum splits the volume accordingly.)
2. **Alternating**: $\det$ is zero whenever two columns are equal. (The
   parallelepiped then flattens into a lower-dimensional slab of zero volume.)
3. **Normalized**: $\det \mathbf{I} = 1$. (The unit cube has volume one.)

The granted fact for this subsection, which we use without proof, is that
there is exactly *one* function of the columns with these three properties;
it is the determinant (see :cite:`Horn.Johnson.2012` for the construction and
the uniqueness argument). Uniqueness comes in a sharper form that we will use
below: *every* function $f$ of the columns satisfying (1) and (2) is a
multiple of the determinant, $f = f(\mathbf{I}) \cdot \det$. To see why, set
$g = f - f(\mathbf{I})\cdot\det$; then $g$ satisfies (1) and (2) with
$g(\mathbf{I}) = 0$, so $g + \det$ satisfies all three properties, hence
equals $\det$ by uniqueness, forcing $g = 0$.

The three properties are a machine for producing identities. First,
**swapping two columns flips the sign of the determinant**. Place the vector
$\mathbf{u}+\mathbf{v}$ in two different column slots and expand by
multilinearity:

$$
\begin{aligned}
0 &= \det(\ldots, \mathbf{u}+\mathbf{v}, \ldots, \mathbf{u}+\mathbf{v}, \ldots) \\
  &= \det(\ldots, \mathbf{u}, \ldots, \mathbf{u}, \ldots)
   + \det(\ldots, \mathbf{u}, \ldots, \mathbf{v}, \ldots)
   + \det(\ldots, \mathbf{v}, \ldots, \mathbf{u}, \ldots)
   + \det(\ldots, \mathbf{v}, \ldots, \mathbf{v}, \ldots).
\end{aligned}
$$

The left-hand side, the first term, and the last term all vanish by the
alternating property, leaving

$$
\det(\ldots, \mathbf{u}, \ldots, \mathbf{v}, \ldots)
 = -\det(\ldots, \mathbf{v}, \ldots, \mathbf{u}, \ldots).
$$

This is where the *sign* of the signed area comes from: exchanging two edges
reverses the orientation of the parallelepiped, and an odd permutation of the
columns negates the determinant.

Second, the three properties pin down a formula. Expand each column in the
standard basis, $\mathbf{a}_j = \sum_i a_{ij}\,\mathbf{e}_i$, and apply
multilinearity column by column: $\det\mathbf{A}$ becomes a sum of terms of
the form $a_{i_1 1} a_{i_2 2} \cdots a_{i_n n}
\det(\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_n})$. Whenever two of the chosen
row indices coincide, the term dies by the alternating property; when all are
distinct, the columns $\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_n}$ are a
reshuffling of the identity matrix's, so repeated column swaps reduce the
determinant factor to $\pm 1$. For a $2 \times 2$ matrix with columns
$\mathbf{a}_1 = a\,\mathbf{e}_1 + c\,\mathbf{e}_2$ and
$\mathbf{a}_2 = b\,\mathbf{e}_1 + d\,\mathbf{e}_2$, the expansion has four
terms; the two with a repeated basis vector vanish and

$$
\det(\mathbf{a}_1, \mathbf{a}_2)
 = ad\,\det(\mathbf{e}_1, \mathbf{e}_2)
 + cb\,\det(\mathbf{e}_2, \mathbf{e}_1)
 = ad - bc,
$$

recovering the parallelogram area we computed by hand. In general the
determinant is a signed sum, over all ways of picking one entry from each row
and each column, of the products of the chosen entries; this is the
permutation formula, and (1)+(2)+(3) leave no other possibility.

For hand computation the sum is organized as *cofactor expansion*: expanding
the first column in the standard basis splits $\det\mathbf{A}$ into $n$
smaller determinants, with alternating signs, of the submatrices obtained by
deleting the first column and one row. Let us work a $3 \times 3$ example,

$$
\mathbf{A} = \begin{bmatrix}
2 & 1 & 0 \\
1 & 3 & 1 \\
0 & 1 & 2
\end{bmatrix},
$$

expanding along the first column:

$$
\det \mathbf{A}
 = 2 \det \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}
 - 1 \det \begin{bmatrix} 1 & 0 \\ 1 & 2 \end{bmatrix}
 + 0 \det \begin{bmatrix} 1 & 0 \\ 3 & 1 \end{bmatrix}
 = 2 \cdot 5 - 1 \cdot 2 + 0 = 8.
$$
:eqlabel:`eq_mdl-det-3x3`

Geometrically, $\mathbf{A}$ maps the unit cube to a parallelepiped of volume
$8$, and it scales the volume of every solid by that same factor. Let us
check :eqref:`eq_mdl-det-3x3` in code, along with two consequences of the
properties above: swapping two columns flips the sign, and the determinant of
a triangular matrix is the product of its diagonal entries.

```{.python .input #mdl-geometry-linear-algebraic-ops-det-3x3}
#@tab mxnet
A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
A_swap = np.array([[1, 2, 0], [3, 1, 1], [1, 0, 2]])  # first two columns swapped
T = np.array([[2, 5, 1], [0, 3, 4], [0, 0, 1]])       # upper triangular
(np.linalg.det(A),       # 8, matching the cofactor expansion
 np.linalg.det(A_swap),  # -8: a column swap flips the sign
 np.linalg.det(T))       # 6 = 2 * 3 * 1, the diagonal product
```

```{.python .input #mdl-geometry-linear-algebraic-ops-det-3x3}
#@tab pytorch
A = torch.tensor([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=torch.float32)
A_swap = torch.tensor([[1, 2, 0], [3, 1, 1], [1, 0, 2]],
                      dtype=torch.float32)  # first two columns swapped
T = torch.tensor([[2, 5, 1], [0, 3, 4], [0, 0, 1]],
                 dtype=torch.float32)       # upper triangular
(torch.det(A),       # 8, matching the cofactor expansion
 torch.det(A_swap),  # -8: a column swap flips the sign
 torch.det(T))       # 6 = 2 * 3 * 1, the diagonal product
```

```{.python .input #mdl-geometry-linear-algebraic-ops-det-3x3}
#@tab tensorflow
A = tf.constant([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=tf.float32)
A_swap = tf.constant([[1, 2, 0], [3, 1, 1], [1, 0, 2]],
                     dtype=tf.float32)  # first two columns swapped
T = tf.constant([[2, 5, 1], [0, 3, 4], [0, 0, 1]],
                dtype=tf.float32)       # upper triangular
(tf.linalg.det(A),       # 8, matching the cofactor expansion
 tf.linalg.det(A_swap),  # -8: a column swap flips the sign
 tf.linalg.det(T))       # 6 = 2 * 3 * 1, the diagonal product
```

```{.python .input #mdl-geometry-linear-algebraic-ops-det-3x3}
#@tab jax
A = jnp.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=jnp.float32)
A_swap = jnp.array([[1, 2, 0], [3, 1, 1], [1, 0, 2]],
                   dtype=jnp.float32)  # first two columns swapped
T = jnp.array([[2, 5, 1], [0, 3, 4], [0, 0, 1]],
              dtype=jnp.float32)       # upper triangular
(jnp.linalg.det(A),       # 8, matching the cofactor expansion
 jnp.linalg.det(A_swap),  # -8: a column swap flips the sign
 jnp.linalg.det(T))       # 6 = 2 * 3 * 1, the diagonal product
```

The triangular rule deserves its one-paragraph proof, since the next section
relies on it. Suppose $\mathbf{T}$ is upper triangular, so $t_{ij} = 0$
whenever $i > j$. Each term of the permutation formula picks one entry from
every column, all in distinct rows. To be nonzero, the entry picked from
column $1$ must come from row $1$ (every lower entry vanishes); the entry from
column $2$ must then come from row $2$ (row $1$ is taken and rows $3, \ldots,
n$ hold zeros); continuing across the columns, only the diagonal choice
survives. That single term carries the sign of no swaps at all, so

$$
\det \mathbf{T} = t_{11} t_{22} \cdots t_{nn},
$$

the product of the diagonal entries. The same argument, run from column $n$
backwards, handles lower triangular matrices.

One more identity rounds out the toolkit: $\det \mathbf{A}^\top =
\det \mathbf{A}$. For $2 \times 2$ matrices this is visible at a glance, since
transposing swaps $b$ and $c$ and leaves $ad - bc$ unchanged. In general,
transposing merely reindexes the permutation formula: each product of $n$
entries, one from every row and every column, appears in both expansions with
the same sign (see :cite:`Horn.Johnson.2012` for the bookkeeping). One
consequence: everything the determinant says about columns holds verbatim for
rows.

#### The Unifying Theorem

The equivalence between a zero determinant, linearly dependent columns, and
failure of invertibility connects three notions we have met separately. We
state it once, with a proof we can carry out by hand in two dimensions.

**Proposition (the unifying theorem).** *For a square matrix $\mathbf{A}$, the
following are equivalent:*
(i) *$\det\mathbf{A} = 0$;*
(ii) *the columns of $\mathbf{A}$ are linearly dependent;*
(iii) *$\mathbf{A}$ is not invertible.*

**Proof (for the $2 \times 2$ case).** We give the argument for the
$2 \times 2$ matrix
$\mathbf{A} = \bigl[\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr]$,
where every step is a picture. Write the two
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
map can be undone; concretely, $ad - bc \neq 0$ is exactly the nonvanishing
denominator that made the explicit inverse
$\frac{1}{ad-bc}\bigl[\begin{smallmatrix} d & -b \\ -c & a \end{smallmatrix}\bigr]$
well-defined earlier in this section. $\blacksquare$

In $n$ dimensions the same equivalences hold, and the axiomatic
characterization of :numref:`subsec_mdl-determinant-general` supplies the two
ingredients that generalize: $\det\mathbf{A}$ is the signed factor by which
$\mathbf{A}$ scales $n$-dimensional volume, and that volume vanishes exactly
when the columns are linearly dependent; see :cite:`Horn.Johnson.2012` for the
full argument. The equivalence also proves claims made earlier without proof:
linear dependence (the columns of $\mathbf{B}$ are redundant), the missing
$ad - bc \neq 0$ hypothesis under the $2 \times 2$ inverse, and the present
section's "$\det = 0$ means collapse" all turn out to be three faces of the
same fact.

#### Multiplicativity

Imagine that we have any figure drawn on the plane.
Thinking like computer scientists, we can decompose
that figure into a collection of little squares
so that the area of the figure is in essence
just the number of squares in the decomposition.
If we now transform that figure by a matrix,
we send each of these squares to parallelograms,
each one of which has area given by the determinant.
We see that for any figure, the determinant gives the (signed) number
that a matrix scales the area of any figure.

This "scale every figure's area by the same factor" reading has an immediate
consequence for *composing* two transformations. First, a fact that deserves
its own sentence: **matrix multiplication is composition of linear
maps**. Applying $\mathbf{B}$ and then $\mathbf{A}$ to an input $\mathbf{v}$
produces $\mathbf{A}(\mathbf{B}\mathbf{v})$, and writing out components,

$$
\bigl(\mathbf{A}(\mathbf{B}\mathbf{v})\bigr)_i
 = \sum_j a_{ij} (\mathbf{B}\mathbf{v})_j
 = \sum_j \sum_k a_{ij} b_{jk} v_k
 = \bigl((\mathbf{A}\mathbf{B})\mathbf{v}\bigr)_i,
$$

so running the two maps in turn is the same as applying the single matrix
$\mathbf{A}\mathbf{B}$. This is the real reason matrix multiplication is
defined by the row-times-column rule, and it explains why the product is
associative but not commutative: composing functions in the other order
generally gives a different function. With composition in hand, we can prove
that the determinant is multiplicative.

**Proposition (multiplicativity of the determinant).** *For square matrices
$\mathbf{A}$ and $\mathbf{B}$ of the same size,*

$$
\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\,\det(\mathbf{B}).
$$
:eqlabel:`eq_mdl-det-multiplicative`

**Geometric intuition.** Apply $\mathbf{A}\mathbf{B}$ to an arbitrary figure
of area $V$ by running the two maps in turn. First $\mathbf{B}$ acts and
scales the area to $\det(\mathbf{B})\,V$; then $\mathbf{A}$ acts and scales
that by a further factor of $\det(\mathbf{A})$. But the composite map
$\mathbf{A}\mathbf{B}$ is itself a single linear transformation, which scales
area by its own determinant, so the two factors must multiply to
$\det(\mathbf{A}\mathbf{B})$. This argument leans on the tiling picture above
(a linear map scales *every* figure's volume by one common factor), which we
made plausible but did not prove; the axioms of
:numref:`subsec_mdl-determinant-general` turn it into a proof.

**Proof.** Fix $\mathbf{A}$ and consider, as a function of $\mathbf{B}$, the
quantity $D(\mathbf{B}) = \det(\mathbf{A}\mathbf{B})$. The $j$-th column of
$\mathbf{A}\mathbf{B}$ is $\mathbf{A}\mathbf{b}_j$, where $\mathbf{b}_j$ is
the $j$-th column of $\mathbf{B}$. Because $\mathbf{A}(\cdot)$ is linear and
$\det$ is multilinear, $D$ is multilinear in the columns of $\mathbf{B}$; and
if two columns of $\mathbf{B}$ are equal, then so are the corresponding
columns of $\mathbf{A}\mathbf{B}$, making $D(\mathbf{B}) = 0$, so $D$ is
alternating. By the scaling form of uniqueness from
:numref:`subsec_mdl-determinant-general`,

$$
\det(\mathbf{A}\mathbf{B}) = D(\mathbf{B}) = D(\mathbf{I})\,\det(\mathbf{B})
 = \det(\mathbf{A})\,\det(\mathbf{B}). \qquad \blacksquare
$$

A corollary is one line long but easy to miss.

**Corollary.** *For square matrices $\mathbf{A}$ and $\mathbf{B}$ of the same
size,*

$$
\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\,\det(\mathbf{B})
 = \det(\mathbf{B})\,\det(\mathbf{A}) = \det(\mathbf{B}\mathbf{A}).
$$

Even though $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ in general, their
determinants always agree, because the two determinants are scalars and
scalars commute. This is a genuinely useful fact: applied to the pair
$\mathbf{W}\mathbf{A}$ and $\mathbf{W}^{-1}$, it gives *similarity
invariance*, $\det(\mathbf{W}\mathbf{A}\mathbf{W}^{-1}) =
\det(\mathbf{W}^{-1}\mathbf{W}\mathbf{A}) = \det(\mathbf{A})$: matrices that
represent the same linear map in different bases have the same determinant,
a fact :numref:`sec_mdl-eigendecompositions` puts to work.

Further consequences follow without any additional work. First, taking
$\mathbf{B} = \mathbf{A}^{-1}$ in :eqref:`eq_mdl-det-multiplicative` and using
$\det(\mathbf{I}) = 1$ (the identity moves no volume) gives

$$
\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})},
$$

which also re-confirms the unifying theorem: an inverse can exist only when
$\det(\mathbf{A}) \neq 0$, since otherwise the right-hand side is undefined.
Second, we can prove the claim made in the orthogonal-matrices
subsection. Combining multiplicativity with $\det\mathbf{Q}^\top =
\det\mathbf{Q}$ from :numref:`subsec_mdl-determinant-general` and applying
both to $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ gives

$$
\det(\mathbf{Q})^2 = \det(\mathbf{Q}^\top)\det(\mathbf{Q})
 = \det(\mathbf{Q}^\top\mathbf{Q}) = \det(\mathbf{I}) = 1,
$$

so $\det\mathbf{Q} = \pm 1$: a rigid motion leaves every area and volume
unchanged in magnitude, and the sign records whether it is a pure rotation
($+1$) or involves a reflection ($-1$).
Third, and looking ahead, similarity invariance is exactly what lets the
determinant pass through a diagonalization: once we can write a matrix in
terms of its eigenvalues in :numref:`sec_mdl-eigendecompositions`, the same
identities will show that the determinant is the *product of the
eigenvalues*, $\det(\mathbf{A}) = \prod_i \lambda_i$, the volume scaling
being the product of the per-axis stretch factors.

## Tensors and Einstein Summation

We close with a compact notation for all of these products at once. Every
product in this section (dot products, matrix–vector products,
matrix products, and the *trace* $\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}$,
the sum of the diagonal entries) follows one pattern: multiply entries, then
sum over the index that appears twice,

$$
\mathbf{v} \cdot \mathbf{w} = \sum_i v_i w_i,
\qquad
(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij} v_j,
\qquad
(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij} b_{jk},
\qquad
\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}.
$$

*Einstein notation* makes the pattern the entire definition: write the indexed
factors, drop the summation sign, and sum over every index that appears more
than once. Thus $\mathbf{v}\cdot\mathbf{w} = v_i w_i$ and
$(\mathbf{A}\mathbf{B})_{ik} = a_{ij}b_{jk}$. The same rule extends unchanged
to tensors with any number of axes, where a general *tensor contraction* such
as $y_{il} = x_{ijkl}\,a_{jk}$ (summing over $j$ and $k$) has no tidy matrix
notation at all. The rule is exposed directly as
`einsum`: spell out the index pattern as a string, and the library performs
the contraction.

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab mxnet
A = np.array([[1.0, 2.0], [-1.0, 3.0]])
v = np.array([2.0, -1.0])
(np.einsum('i,i->', v, v),      # dot product: v.v
 np.einsum('ij,j->i', A, v),    # matrix-vector product: Av
 np.einsum('ij,jk->ik', A, A),  # matrix product: AA
 np.einsum('ii->', A))          # trace: tr(A)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab pytorch
A = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
v = torch.tensor([2.0, -1.0])
(torch.einsum('i,i->', v, v),      # dot product: v.v
 torch.einsum('ij,j->i', A, v),    # matrix-vector product: Av
 torch.einsum('ij,jk->ik', A, A),  # matrix product: AA
 torch.einsum('ii->', A))          # trace: tr(A)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab tensorflow
A = tf.constant([[1.0, 2.0], [-1.0, 3.0]])
v = tf.constant([2.0, -1.0])
(tf.einsum('i,i->', v, v),      # dot product: v.v
 tf.einsum('ij,j->i', A, v),    # matrix-vector product: Av
 tf.einsum('ij,jk->ik', A, A),  # matrix product: AA
 tf.einsum('ii->', A))          # trace: tr(A)
```

```{.python .input #geometry-linear-algebraic-ops-expressing-in-code-2}
#@tab jax
A = jnp.array([[1.0, 2.0], [-1.0, 3.0]])
v = jnp.array([2.0, -1.0])
(jnp.einsum('i,i->', v, v),      # dot product: v.v
 jnp.einsum('ij,j->i', A, v),    # matrix-vector product: Av
 jnp.einsum('ij,jk->ik', A, A),  # matrix product: AA
 jnp.einsum('ii->', A))          # trace: tr(A)
```

The matrix here is $\mathbf{A}$ from
:numref:`fig_mdl-la-linear-map` and the vector is $\mathbf{v} = [2, -1]^\top$,
so the second entry of the output reproduces the worked example
$\mathbf{A}\mathbf{v} = [0, -5]^\top$. Index strings like
`'ij,jk->ik'` recur throughout deep learning: batched matrix products
(`'bij,bjk->bik'`), attention scores, and many custom layers are one `einsum`
call away, and we will reach for the notation whenever a computation is easier
to state in indices than in matrices, as soon as
:numref:`sec_mdl-svd-low-rank`, where a one-line `einsum` will rebuild a matrix
from its singular value decomposition.

Index notation also yields one-line proofs. In indices, the trace of a
product is

$$
\textrm{tr}(\mathbf{A}\mathbf{B}) = a_{ij} b_{ji} = b_{ji} a_{ij}
 = \textrm{tr}(\mathbf{B}\mathbf{A}),
$$
:eqlabel:`eq_mdl-trace-cyclic`

and the middle step costs nothing: matrix entries are numbers, so they
commute, and only the bookkeeping of the indices matters. Note that
:eqref:`eq_mdl-trace-cyclic` holds even when $\mathbf{A}\mathbf{B} \neq
\mathbf{B}\mathbf{A}$, just as with determinants. A corollary is that the
trace is invariant under a change of basis,

$$
\textrm{tr}(\mathbf{W}\mathbf{A}\mathbf{W}^{-1})
 = \textrm{tr}(\mathbf{A}\mathbf{W}^{-1}\mathbf{W})
 = \textrm{tr}(\mathbf{A}),
$$

the identity behind the trace–eigenvalue formula
$\textrm{tr}(\mathbf{A}) = \sum_i \lambda_i$ in
:numref:`sec_mdl-eigendecompositions`.

## Summary
* Vectors can be interpreted geometrically as either points or directions in space.
* Dot products define the notion of angle to arbitrarily high-dimensional spaces.
* Spans, subspaces, and bases organize collections of vectors: a basis assigns every vector of a subspace unique coordinates, and the dimension counts the basis vectors. The column space and null space of a matrix record what it can produce and what it destroys.
* Projection produces the closest point of a line (or, via the projection matrix $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$, of any subspace) and is characterized by an orthogonal residual. This is the geometry of the least-squares problem :eqref:`eq_mdl-least-squares`.
* Hyperplanes are high-dimensional generalizations of lines and planes.  They can be used to define decision planes that are often used as the last step in a classification task.
* Matrix multiplication can be geometrically interpreted as uniform distortions of the underlying coordinates: a matrix skews, rotates, and scales, and it treats every part of space the same way. Multiplying two matrices composes the corresponding maps.
* Orthogonal matrices are the rigid motions: they preserve lengths, angles, and (up to a sign recording reflections) volumes.
* Linear dependence is a way to tell when a collection of vectors are in a lower dimensional space than we would expect (say you have $3$ vectors living in a $2$-dimensional space). The rank of a matrix is the size of the largest subset of its columns that are linearly independent, and rank--nullity is exact: rank plus the dimension of the null space equals the number of columns.
* When a matrix's inverse is defined, matrix inversion allows us to find another matrix that undoes the action of the first. Matrix inversion is useful in theory, but requires care in practice owing to numerical instability.
* The determinant is the signed volume scaling of a matrix. It is characterized by three properties (multilinear, alternating, $\det\mathbf{I} = 1$), it flips sign under a column swap, and it satisfies $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B}) = \det(\mathbf{B}\mathbf{A})$. A nonzero determinant is equivalent to invertibility; a zero determinant means the matrix is singular.
* Tensor contractions and Einstein summation express many of the computations of machine learning in one index pattern, and in index form identities such as $\textrm{tr}(\mathbf{A}\mathbf{B}) = \textrm{tr}(\mathbf{B}\mathbf{A})$ become one-line computations.

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
9. The proof of Cauchy–Schwarz shows that equality holds exactly when one vector is a scalar multiple of the other. Use this to characterize when the triangle inequality $\|\mathbf{v} + \mathbf{w}\| \le \|\mathbf{v}\| + \|\mathbf{w}\|$ holds with equality, and check your characterization on $\mathbf{v} = [1,2]^\top$ paired first with $\mathbf{w} = [2,4]^\top$ and then with $\mathbf{w} = [-1,-2]^\top$.
10. Compute the projection $\operatorname{proj}_{\mathbf{w}}\mathbf{v}$ of $\mathbf{v} = [1,2,3]^\top$ onto $\mathbf{w} = [1,1,1]^\top$. Verify that the residual is orthogonal to $\mathbf{w}$ and confirm the Pythagorean identity :eqref:`eq_mdl-pythagoras` numerically.
11. In code: sample $1{,}000$ pairs of random unit vectors in $d = 10{,}000$ dimensions (draw Gaussian vectors and normalize them) and compute the cosine of the angle for each pair. Compare the empirical standard deviation to the predicted $1/\sqrt{d} = 0.01$, and report the largest $|\cos\theta|$ you observe.
12. Show that an orthogonal matrix $\mathbf{Q}$ preserves angles: for nonzero $\mathbf{x}$ and $\mathbf{y}$, the angle between $\mathbf{Q}\mathbf{x}$ and $\mathbf{Q}\mathbf{y}$ equals the angle between $\mathbf{x}$ and $\mathbf{y}$. Conclude that rotating every embedding vector in a dataset by the same orthogonal matrix leaves all cosine similarities unchanged.
13. Verify that the $4 \times 5$ matrix $\mathbf{C}$ from the rank subsection has rank two: check that the first two columns are linearly independent, and confirm, by hand or with `matrix_rank`, that every collection of three of its columns is linearly dependent.
14. Use multilinearity to prove that $\det(c\mathbf{A}) = c^n \det(\mathbf{A})$ for an $n \times n$ matrix $\mathbf{A}$ and a scalar $c$. Check the identity in code for the $3 \times 3$ matrix of :eqref:`eq_mdl-det-3x3` with $c = 2$.

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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §22.1]{.kicker}

The geometry under the algebra<br>**angles, projections, hyperplanes, and how matrices move space**.
:::
:::

::: {.slide title="Why a geometric view"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A vector is a list of numbers, but reading it as a **picture** is where the
intuition lives.

- A vector is a **point**, or equally an **arrow** (a direction).
- The **dot product** measures alignment, giving angles and similarity.
- A **hyperplane** is the decision boundary of every linear classifier.
- A **matrix** is a map that skews, rotates, and scales space.

::: {.d2l-note}
These pictures are the foundation for the **eigendecomposition** and the
**SVD** in the sections that follow.
:::
:::

::: {.col .fig}
![](../img/mdl-la-hyperplane.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Vectors and their geometry]{.dtitle}

[points, directions, dot products, projection]{.dsub}
:::
:::

::: {.slide title="Two readings of a vector"}
[Vectors]{.kicker}

::: {.cols .vc}
::: {.col}
The same array names a **point** (its coordinates) or a **direction** (an
arrow that may start anywhere). Deep learning mostly uses the direction view.

Reading a vector as a direction makes **addition** visual: follow one arrow,
then the next, tip to tail.
:::

::: {.col .fig .big}
![](../img/mdl-la-vectors.svg)
:::
:::
:::

::: {.slide title="Dot products and the angle"}
[Vectors]{.kicker}

Why is the dot product tied to the **angle**? The two vectors span a plane, so
measure $\|\mathbf{v}-\mathbf{w}\|^2$ two ways and equate.

. . .

Algebra gives $\|\mathbf{v}\|^2 - 2\,\mathbf{v}\cdot\mathbf{w} + \|\mathbf{w}\|^2$; the law of cosines gives $\|\mathbf{v}\|^2 + \|\mathbf{w}\|^2 - 2\,\|\mathbf{v}\|\,\|\mathbf{w}\|\cos\theta$.

. . .

::: {.cols .vc}
::: {.col}
Cancel the common terms: $\;\mathbf{v}\cdot\mathbf{w} = \|\mathbf{v}\|\,\|\mathbf{w}\|\cos\theta$, so $\theta=\arccos\!\big(\mathbf{v}\cdot\mathbf{w}/\|\mathbf{v}\|\,\|\mathbf{w}\|\big)$.

@geometry-linear-algebraic-ops-dot-products-and-angles
:::

::: {.col .fig}
![](../img/mdl-la-angle.svg)
:::
:::
:::

::: {.slide title="Why the angle is always defined"}
[Vectors]{.kicker}

$\arccos$ only accepts inputs in $[-1, 1]$, so we need a guarantee.

::: {.d2l-note .rule}
**Cauchy–Schwarz.** $\;|\mathbf{v}\cdot\mathbf{w}| \le \|\mathbf{v}\|\,\|\mathbf{w}\|$, with equality iff $\mathbf{v}$ and $\mathbf{w}$ are collinear.
:::

. . .

The proof needs only one fact: a squared length is never negative, so
$q(t) = \|\mathbf{v} - t\mathbf{w}\|^2 \ge 0$ is a parabola that cannot dip
below zero, forcing its discriminant $\le 0$. Dividing through shows the
cosine never leaves $[-1, 1]$, so $\theta$ is always a genuine angle.
:::

::: {.slide title="Projection and orthogonality"}
[Vectors]{.kicker}

::: {.cols .vc}
::: {.col}
How much of $\mathbf{v}$ points along $\mathbf{w}$? Drop $\mathbf{v}$ onto the
line through $\mathbf{w}$:

$$\operatorname{proj}_{\mathbf{w}}\mathbf{v}
 = \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{w}\cdot\mathbf{w}}\,\mathbf{w}.$$

The residual $\mathbf{r}$ leaves at a **right angle**, so Cauchy–Schwarz is
just "a leg is no longer than the hypotenuse." Two vectors are **orthogonal**
when $\mathbf{v}\cdot\mathbf{w} = 0$.

This is least squares in one dimension (the best fit leaves an orthogonal
residual); the **SVD** scales the same idea to any matrix.
:::

::: {.col .fig}
![](../img/mdl-la-projection.svg)
:::
:::
:::

::: {.slide title="Projection onto a subspace"}
[Vectors]{.kicker}

The same formula, one matrix heavier: stack an **orthonormal** basis of a
subspace $S$ as the columns of $\mathbf{Q}$ (so $\mathbf{Q}^\top\mathbf{Q} =
\mathbf{I}$), and $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$ drops any vector
onto $S$.

::: {.d2l-note .rule}
$\mathbf{P}\mathbf{x}$ is the unique **closest point** of $S$; the residual
$\mathbf{x} - \mathbf{P}\mathbf{x}$ is orthogonal to *all* of $S$; and
$\mathbf{P}^2 = \mathbf{P}$: projecting twice changes nothing.
:::

. . .

`qr` manufactures the orthonormal basis. Both claims check to roundoff on a
random 3-dimensional subspace of $\mathbb{R}^5$, and residual-orthogonality
*is* least squares: the optimal fit's error is orthogonal to every feature.

@mdl-geometry-linear-algebraic-ops-projection-onto-a-subspace
:::

::: {.slide title="Span, basis, dimension"}
[Vectors]{.kicker}

::: {.cols .vc}
::: {.col}
- **Span**: everything reachable by scaling and adding a set of vectors.
- **Basis**: an independent spanning set, so every vector gets **unique**
  coordinates.
- **Dimension**: the size of any basis.

::: {.d2l-note}
Two subspaces ride on every matrix: the **column space** is what it can
produce, the **null space** is what it sends to zero.
:::
:::

::: {.col .fig}
![](../img/mdl-la-span.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Similarity in high dimensions]{.dtitle}

[cosine similarity and near-orthogonality]{.dsub}
:::
:::

::: {.slide title="Cosine similarity"}
[High dimensions]{.kicker}

Comparing direction, not magnitude, is often what we want: an image and a
dimmed copy point the same way, so they should score as identical.

$$\cos\theta = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\,\|\mathbf{w}\|}
 \;\in\; [-1, 1].$$

. . .

This is the signal behind **embedding retrieval**, the scaled dot products
inside **attention**, and the alignment objective of **contrastive learning**.
:::

::: {.slide title="Random vectors are nearly orthogonal"}
[High dimensions]{.kicker}

::: {.cols .vc}
::: {.col}
Drop two unrelated unit vectors into $\mathbb{R}^d$: what cosine do we expect?

::: {.d2l-note .rule}
For a random unit vector, $\;\mathbb{E}[\cos\theta] = 0$ and
$\operatorname{Var}(\cos\theta) = \tfrac{1}{d}$.
:::

*Why:* rotate so $\mathbf{u}=\mathbf{e}_1$, so $\cos\theta=v_1$. Symmetry
$\mathbf{v}\mapsto-\mathbf{v}$ kills the mean; $\sum_i v_i^2=1$ with all
coordinates alike gives $\mathbb{E}[v_1^2]=\tfrac{1}{d}$.

So the cosine concentrates at $0$ with width $1/\sqrt{d}$: a cosine well above
$0$ is unlikely to be an accident, which is **why** cosine similarity works.
:::

::: {.col .fig}
![](../img/mdl-la-cosine-highd.svg)
:::
:::
:::

::: {.slide title="Why attention divides by √d"}
[High dimensions]{.kicker}

Attention compares one query against thousands of keys by dot product, and
near-orthogonality is its operating environment: the many unrelated keys score
near zero, so the few that share structure with the query stand out against a
quiet background.

. . .

The mysterious $\sqrt{d}$ falls out of our variance result. With entries of
typical size $1$, $\|\mathbf{q}\| \approx \|\mathbf{k}\| \approx \sqrt{d}$
while $\operatorname{sd}(\cos\theta) = 1/\sqrt{d}$, so

$$\operatorname{sd}(\mathbf{q}\cdot\mathbf{k})
\approx \sqrt{d}\cdot\sqrt{d}\cdot\tfrac{1}{\sqrt{d}} = \sqrt{d}.$$

::: {.d2l-note .rule}
$\mathbf{Q}\mathbf{K}^\top/\sqrt{d}$ is **standardization**: dividing the raw
scores by their standard deviation keeps them $O(1)$, so the softmax stays in
its responsive range instead of saturating.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Hyperplanes and decision boundaries]{.dtitle}

[the primitive every classifier shares]{.dsub}
:::
:::

::: {.slide title="A hyperplane as a decision boundary"}
[Hyperplanes]{.kicker}

::: {.cols .vc}
::: {.col}
The set $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} = b\}$ is a plane with
**normal** $\mathbf{w}$, sitting at distance $b/\|\mathbf{w}\|$ from the
origin. Sliding $b$ translates it without rotating.

For any point, the **signed distance**
$(\mathbf{w}\cdot\mathbf{x} - b)/\|\mathbf{w}\|$ is positive on one side and
negative on the other. That number is exactly a linear classifier's **margin**.
:::

::: {.col .fig}
![](../img/mdl-la-hyperplane.svg)
:::
:::
:::

::: {.slide title="A classifier with nothing trained"}
[Hyperplanes]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Take Fashion-MNIST t-shirts and trousers. Average each class: the means are
blurry but recognizable. The line between them, $\mathbf{w} = \overline{\mathbf{x}}_1 - \overline{\mathbf{x}}_0$, is our normal.
:::

::: {.col .fig .big}
@!geometry-linear-algebraic-ops-hyperplanes-2
:::
:::
:::

::: {.slide title="One hyperplane, ~92% correct"}
[Hyperplanes]{.kicker}

Classify each $784$-dimensional image by the side it falls on, with the
threshold at the midpoint of the two means' projections:

@geometry-linear-algebraic-ops-hyperplanes-4

. . .

Over $2{,}000$ test images, this hand-built rule is right about **92%** of the
time, and nothing was learned.
:::

::: {.slide title="The whole picture in one projection"}
[Hyperplanes]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Reduce every image to one number, $\mathbf{w}\cdot\mathbf{x}$, its position
along the normal. The two classes form two humps; the dashed threshold cuts
between them.

A *learned* classifier just tilts this boundary to trim the overlap; a deep
net learns features that pull the humps apart.
:::

::: {.col .fig .big}
@!geometry-linear-algebraic-ops-projection-histogram
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Matrices as linear maps]{.dtitle}

[skew, rotate, scale, and the determinant]{.dsub}
:::
:::

::: {.slide title="A matrix moves the whole grid"}
[Linear maps]{.kicker}

::: {.cols .vc}
::: {.col}
A matrix is fixed by where it sends the **basis vectors**, namely its columns.
Every other vector follows as a weighted sum, so the entire grid is carried
along: lines stay lines, the origin stays put, cells stay evenly spaced.

::: {.d2l-note}
Matrices cannot bend space, only **skew, rotate, and scale** it. Multiplying
two matrices **composes** their maps.
:::
:::

::: {.col .fig .big}
![](../img/mdl-la-linear-map.svg)
:::
:::
:::

::: {.slide title="What a matrix destroys"}
[Linear maps]{.kicker}

::: {.cols .vc}
::: {.col}
$\mathbf{B} = \bigl[\begin{smallmatrix}2&-1\\4&-2\end{smallmatrix}\bigr]$ has
dependent columns, $\mathbf{b}_1 + 2\,\mathbf{b}_2 = \mathbf{0}$. The whole
plane lands on one line (the **column space**), and the direction $(1,2)^\top$
is crushed to the origin (the **null space**). Once inputs collide, no inverse
can tell them apart.

::: {.d2l-note .rule}
**Rank--nullity.** $\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$:
of the $n$ directions coming in, $\dim\ker\mathbf{A}$ are destroyed and
$\operatorname{rank}\mathbf{A}$ survive. For $\mathbf{B}$: $1 + 1 = 2$.
:::
:::

::: {.col .fig .big}
![](../img/mdl-la-null-collapse.svg)
:::
:::
:::

::: {.slide title="Orthogonal matrices: the rigid motions"}
[Linear maps]{.kicker}

A square matrix is **orthogonal** when $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$.
Such maps preserve every dot product, hence all **lengths and angles**:

$$(\mathbf{Q}\mathbf{x})\cdot(\mathbf{Q}\mathbf{y})
 = \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\,\mathbf{y}
 = \mathbf{x}\cdot\mathbf{y}.$$

. . .

They are the pure rotations and reflections: the distortion-free building
blocks of both the spectral theorem and the SVD.
:::

::: {.slide title="The determinant is signed area"}
[Linear maps]{.kicker}

::: {.cols .vc}
::: {.col}
The unit square maps to a parallelogram; its **signed area** is $\det\mathbf{A}$.
The sign records orientation, and a **zero** determinant means space was
crushed to a lower dimension.

For our grid matrix, $\det = 1\cdot3 - 2\cdot(-1) = 5$:

@geometry-linear-algebraic-ops-determinant

Composed maps multiply their area scalings,
$\det(\mathbf{A}\mathbf{B})=\det\mathbf{A}\,\det\mathbf{B}$. Two payoffs:
$\det\mathbf{Q}=\pm1$ for orthogonal $\mathbf{Q}$, and (next section)
$\det\mathbf{A}=\prod_i\lambda_i$, the per-axis stretches multiplied.
:::

::: {.col .fig}
![](../img/mdl-la-determinant.svg)
:::
:::
:::

::: {.slide title="The determinant, defined"}
[Linear maps]{.kicker}

Three properties pin the determinant down in every dimension, as a function
of the columns: **multilinear** (linear in each column), **alternating**
(zero when two columns are equal), **normalized** ($\det\mathbf{I} = 1$).
Exactly one such function exists.

. . .

Consequences: swapping two columns **flips the sign**, and a triangular
determinant is the product of the diagonal:

@mdl-geometry-linear-algebraic-ops-det-3x3

::: {.d2l-note .rule}
$\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\,\det\mathbf{B} =
\det(\mathbf{B}\mathbf{A})$: determinants commute even when matrices do not.
:::
:::

::: {.slide title="One equivalence ties it together"}
[Linear maps]{.kicker}

::: {.d2l-note .rule}
For a square matrix: $\det\mathbf{A} = 0$ $\iff$ the columns are linearly
dependent $\iff$ $\mathbf{A}$ is **not** invertible.
:::

Our collapsing $\mathbf{B}$ scores $2\cdot(-2) - (-1)\cdot 4 = 0$, exactly as
it must.

. . .

When invertible, $\mathbf{A}^{-1}$ undoes the map; the $2\times2$ formula checks out, multiplying back to the identity:

@geometry-linear-algebraic-ops-invertibility

::: {.d2l-note .warn}
Prefer `linalg.solve(A, b)` over `inv(A) @ b`: stabler, and it never forms the inverse.
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Einstein summation]{.dtitle}

[one rule for every product]{.dsub}
:::
:::

::: {.slide title="Sum over the repeated index"}
[Einstein notation]{.kicker}

Dot products, matrix–vector, matrix–matrix, and traces are one pattern,
$(\mathbf{A}\mathbf{B})_{ik} = a_{ij}\,b_{jk}$: multiply entries, sum the
repeated index. `einsum` makes that index string the whole definition (the
matrix–vector call recovers our worked $\mathbf{A}\mathbf{v} = [0, -5]^\top$):

@-geometry-linear-algebraic-ops-expressing-in-code-2
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- A vector is a **point or a direction**; the **dot product** gives angles.
- **Cauchy–Schwarz** makes the angle well-defined; **projection**
  ($\mathbf{P}=\mathbf{Q}\mathbf{Q}^\top$ for a subspace) leaves an orthogonal
  residual: the geometry of least squares.
- In high dimensions, random directions are **nearly orthogonal**: why cosine
  similarity works and why attention divides by $\sqrt{d}$.
:::

::: {.col}
- A **hyperplane** is the decision boundary of every linear classifier.
- A **matrix** skews, rotates, and scales; **rank--nullity** counts what
  survives; the **determinant** is its volume scale, zero exactly when it
  collapses space.
- **Einstein notation** writes every product as one index pattern.
:::
:::

::: {.d2l-note}
These pictures carry straight into the **eigendecomposition** and the **SVD**,
and all the way up to attention and embeddings.
:::
:::
