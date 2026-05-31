# Eigendecompositions
:label:`sec_mdl-eigendecompositions`

Eigenvalues are often one of the most useful notions
we will encounter when studying linear algebra,
however, as a beginner, it is easy to overlook their importance.
Below, we introduce eigendecomposition and
try to convey some sense of just why it is so important.

Suppose that we have a matrix $A$ with the following entries:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

If we apply $A$ to any vector $\mathbf{v} = [x, y]^\top$,
we obtain a vector $\mathbf{A}\mathbf{v} = [2x, -y]^\top$.
This has an intuitive interpretation:
stretch the vector to be twice as wide in the $x$-direction,
and then flip it in the $y$-direction.

However, there are *some* vectors for which something remains unchanged.
Namely $[1, 0]^\top$ gets sent to $[2, 0]^\top$
and $[0, 1]^\top$ gets sent to $[0, -1]^\top$.
These vectors are still in the same line,
and the only modification is that the matrix stretches them
by a factor of $2$ and $-1$ respectively.
We call such vectors *eigenvectors*
and the factor they are stretched by *eigenvalues*.

In general, if we can find a number $\lambda$
and a vector $\mathbf{v}$ such that

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

We say that $\mathbf{v}$ is an eigenvector for $A$ and $\lambda$ is an eigenvalue.

::: {.callout-note title="⟢ Planned — diagram spec (not yet drawn)"}
**Diagram:** `fig_mdl-eig-circle-to-ellipse` — the unit circle (set of all unit
vectors) mapped through a symmetric $\mathbf{A}$ to an ellipse; the semi-axes lie
*along the eigenvectors* with lengths equal to the eigenvalues $|\lambda_i|$, with
a flipped axis where $\lambda_i < 0$. Makes "eigenvectors are the directions only
scaled" visible at a glance and previews the SVD picture.
:::

## Finding Eigenvalues
Let's figure out how to find them. By subtracting off the $\lambda \mathbf{v}$ from both sides,
and then factoring out the vector,
we see the above is equivalent to:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_mdl-eigvalue_der`

For :eqref:`eq_mdl-eigvalue_der` to happen, we see that $(\mathbf{A} - \lambda \mathbf{I})$
must compress some direction down to zero,
hence it is not invertible, and thus the determinant is zero.
Thus, we can find the *eigenvalues*
by finding for what $\lambda$ is $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Once we find the eigenvalues, we can solve
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
to find the associated *eigenvector(s)*.

### An Example
Let's see this with a more challenging matrix

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

If we consider $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$,
we see this is equivalent to the polynomial equation
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$.
Thus, two eigenvalues are $4$ and $1$.
To find the associated vectors, we then need to solve

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

We can solve this with the vectors $[1, -1]^\top$ and $[1, 2]^\top$ respectively.

We can check this in code using the built-in `numpy.linalg.eig` routine.

```{.python .input #eigendecomposition-an-example}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input #eigendecomposition-an-example}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input #eigendecomposition-an-example}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

```{.python .input #eigendecomposition-an-example}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
from IPython import display
import jax
from jax import numpy as jnp

jnp.linalg.eig(jnp.array([[2, 1], [2, 3]], dtype=jnp.float64))
```

Note that the library normalizes the eigenvectors to be of length one,
whereas we took ours to be of arbitrary length.
Additionally, the choice of sign is arbitrary.
However, the vectors computed are parallel
to the ones we found by hand with the same eigenvalues.

## Decomposing Matrices
Let's continue the previous example one step further.  Let

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

be the matrix where the columns are the eigenvectors of the matrix $\mathbf{A}$. Let

$$
\boldsymbol{\Lambda} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

be the matrix with the associated eigenvalues on the diagonal.
Then the definition of eigenvalues and eigenvectors tells us that

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Lambda} .
$$

The matrix $W$ is invertible, so we may multiply both sides by $W^{-1}$ on the right,
we see that we may write

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Lambda} \mathbf{W}^{-1}.$$
:eqlabel:`eq_mdl-eig_decomp`

In the next section we will see some nice consequences of this,
but for now we need only know that such a decomposition
will exist as long as we can find a full collection
of linearly independent eigenvectors (so that $W$ is invertible).

## Operations on Eigendecompositions
One nice thing about eigendecompositions :eqref:`eq_mdl-eig_decomp` is that
we can write many operations we usually encounter cleanly
in terms of the eigendecomposition. As a first example, consider:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Lambda} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Lambda} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Lambda}\cdots\boldsymbol{\Lambda}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Lambda}^n \mathbf{W}^{-1}.
$$

This tells us that for any positive power of a matrix,
the eigendecomposition is obtained by just raising the eigenvalues to the same power.
The same can be shown for negative powers,
so if we want to invert a matrix we need only consider

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Lambda}^{-1} \mathbf{W}^{-1},
$$

or in other words, just invert each eigenvalue.
This will work as long as each eigenvalue is non-zero,
so we see that invertible is the same as having no zero eigenvalues.

Indeed, additional work can show that if $\lambda_1, \ldots, \lambda_n$
are the eigenvalues of a matrix, then the determinant of that matrix is

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

or the product of all the eigenvalues.
This makes sense intuitively because whatever stretching $\mathbf{W}$ does,
$W^{-1}$ undoes it, so in the end the only stretching that happens is
by multiplication by the diagonal matrix $\boldsymbol{\Lambda}$,
which stretches volumes by the product of the diagonal elements.

Finally, recall that the rank was the maximum number
of linearly independent columns of your matrix.
By examining the eigendecomposition closely,
we can see that the rank is the same
as the number of non-zero eigenvalues of $\mathbf{A}$.

The examples could continue, but hopefully the point is clear:
eigendecomposition can simplify many linear-algebraic computations
and is a fundamental operation underlying many numerical algorithms
and much of the analysis that we do in linear algebra.

## The Spectral Theorem: Eigendecompositions of Symmetric Matrices
It is not always possible to find enough linearly independent eigenvectors
for the above process to work. For instance the matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

has only a single eigenvector, namely $(1, 0)^\top$.
To handle such matrices, we require more advanced techniques
than we can cover (such as the Jordan Normal Form, or Singular Value Decomposition).
We will often need to restrict our attention to those matrices
where we can guarantee the existence of a full set of eigenvectors.

The most commonly encountered family are the *symmetric matrices*,
which are those matrices where $\mathbf{A} = \mathbf{A}^\top$.
The guarantee for this family has a name: the *spectral theorem*.
It states that every real symmetric matrix admits a *full orthonormal*
eigenbasis with *real* eigenvalues. In this case, we may take $W$ to be an
*orthogonal matrix*—a matrix whose columns are all length one vectors that are at right angles to one another, where
$\mathbf{W}^\top = \mathbf{W}^{-1}$—and all the eigenvalues will be real.
Thus, in this special case, we can write :eqref:`eq_mdl-eig_decomp` as

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Lambda}\mathbf{W}^\top .
$$

This is the workhorse behind PCA, covariance analysis, and the Hessian-based
view of optimization, all of which involve symmetric matrices. Contrast it with
the defective $\begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}$ above, which has no such
basis—a deficiency the singular value decomposition repairs for *every* matrix
(:numref:`sec_mdl-svd-low-rank`).

## Positive (Semi)Definiteness

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A symmetric matrix's eigenvalues carry a *sign* story: whether the quadratic form $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ is ever negative. This single property—positive (semi)definiteness—is what makes covariance matrices, Gram matrices, and "this is a minimum" Hessian tests work, so it deserves first-class treatment right after the spectral theorem.
**Outline:** 1. Define $\mathbf{A} \succeq 0$ (PSD) as $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ for all $\mathbf{x}$, and $\mathbf{A} \succ 0$ (PD) as strict for $\mathbf{x}\neq 0$ · 2. Prove the equivalence for symmetric $\mathbf{A}$: PSD $\iff$ all $\lambda_i \ge 0$, PD $\iff$ all $\lambda_i > 0$, via the spectral decomposition $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i \lambda_i (\mathbf{w}_i^\top\mathbf{x})^2$ · 3. Gram/covariance: $\mathbf{G} = \mathbf{X}^\top\mathbf{X}$ is always PSD, PD iff $\mathbf{X}$ has full column rank · 4. Hessian relevance: a critical point is a local minimum if the Hessian is PD (forward-ref optimization) · 5. The PSD cone and the geometric "bowl vs. saddle" picture.
**Key results to state:** $\mathbf{A}\succeq 0 \iff \lambda_i \ge 0\ \forall i$; $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i \lambda_i(\mathbf{w}_i^\top\mathbf{x})^2$; $\mathbf{X}^\top\mathbf{X}\succeq 0$ with PD $\iff$ $\operatorname{rank}\mathbf{X}=$ #columns.
**Diagrams:** `fig_mdl-psd-bowl` — the surface $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ rendered as an upward bowl (PD), a trough (PSD, one zero eigenvalue), and a saddle (indefinite).
**Worked example(s):** form $\mathbf{G} = \mathbf{X}^\top\mathbf{X}$ for a small tall $\mathbf{X}$, verify all eigenvalues $\ge 0$, and exhibit a zero eigenvalue when columns are dependent; check $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ numerically against the eigenvalue test.
**Exercises (draft):** (1) prove $\mathbf{X}^\top\mathbf{X}$ is PSD, and PD iff $\mathbf{X}$ has full column rank; (2) show the sum of two PSD matrices is PSD; (3) classify $\begin{bmatrix}2&1\\1&2\end{bmatrix}$ and $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ as PD / PSD / indefinite from their eigenvalues; (4) show a PD matrix is invertible.
**Prereqs / cross-refs:** spectral theorem above; feeds covariance (:numref:`sec_mdl-svd-low-rank` PCA), Hessian-based optimization, and the second-order minimum test in the calculus chapter.
:::

## Gershgorin Circle Theorem
Eigenvalues are often difficult to reason with intuitively.
If presented an arbitrary matrix, there is little that can be said
about what the eigenvalues are without computing them.
There is, however, one theorem that can make it easy to approximate well
if the largest values are on the diagonal.

Let $\mathbf{A} = (a_{ij})$ be any square matrix ($n\times n$).
We will define $r_i = \sum_{j \neq i} |a_{ij}|$.
Let $\mathcal{D}_i$ represent the disc in the complex plane
with center $a_{ii}$ radius $r_i$.
Then, every eigenvalue of $\mathbf{A}$ is contained in one of the $\mathcal{D}_i$.

This can be a bit to unpack, so let's look at an example.
Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

We have $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ and $r_4 = 0.9$.
The matrix is symmetric, so all eigenvalues are real.
This means that all of our eigenvalues will be in one of the ranges of

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


Performing the numerical computation shows
that the eigenvalues are approximately $0.99$, $2.97$, $4.95$, $9.08$,
all comfortably inside the ranges provided.

```{.python .input #eigendecomposition-gershgorin-circle-theorem}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input #eigendecomposition-gershgorin-circle-theorem}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input #eigendecomposition-gershgorin-circle-theorem}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eig(A)
v
```

```{.python .input #eigendecomposition-gershgorin-circle-theorem}
#@tab jax
A = jnp.array([[1.0, 0.1, 0.1, 0.1],
               [0.1, 3.0, 0.2, 0.3],
               [0.1, 0.2, 5.0, 0.5],
               [0.1, 0.3, 0.5, 9.0]])

v, _ = jnp.linalg.eig(A)
v
```

::: {.callout-note title="⟢ Planned — diagram spec (not yet drawn)"}
**Diagram:** `fig_mdl-gershgorin-disks` — the complex plane with the four discs
$\mathcal{D}_i$ (centers $a_{ii}$, radii $r_i$) from the example drawn as circles,
and the computed eigenvalues plotted as points, each landing inside its disc.
Shows the localization theorem visually.
:::

In this way, eigenvalues can be approximated,
and the approximations will be fairly accurate
in the case that the diagonal is
significantly larger than all the other elements.

It is a small thing, but with a complex
and subtle topic like eigendecomposition,
it is good to get any intuitive grasp we can.

## A Useful Application: The Growth of Iterated Maps

Now that we understand what eigenvectors are in principle,
let's see how they can be used to provide a deep understanding
of a problem central to neural network behavior: proper weight initialization.

### Eigenvectors as Long Term Behavior

The full mathematical investigation of the initialization
of deep neural networks is beyond the scope of the text,
but we can see a toy version here to understand
how eigenvalues can help us see how these models work.
As we know, neural networks operate by interspersing layers
of linear transformations with non-linear operations.
For simplicity here, we will assume that there is no non-linearity,
and that the transformation is a single repeated matrix operation $A$,
so that the output of our model is

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

When these models are initialized, $A$ is taken to be
a random matrix with Gaussian entries, so let's make one of those.
To be concrete, we start with a mean zero, variance one Gaussian distributed $5 \times 5$ matrix.

```{.python .input #eigendecomposition-eigenvectors-as-long-term-behavior}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input #eigendecomposition-eigenvectors-as-long-term-behavior}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input #eigendecomposition-eigenvectors-as-long-term-behavior}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

```{.python .input #eigendecomposition-eigenvectors-as-long-term-behavior}
#@tab jax
k = 5
A = jax.random.normal(jax.random.PRNGKey(42), (k, k), dtype=jnp.float64)
A
```

### Behavior on Random Data
For simplicity in our toy model,
we will assume that the data vector we feed in $\mathbf{v}_{in}$
is a random five dimensional Gaussian vector.
Let's think about what we want to have happen.
For context, lets think of a generic ML problem,
where we are trying to turn input data, like an image, into a prediction,
like the probability the image is a picture of a cat.
If repeated application of $\mathbf{A}$
stretches a random vector out to be very long,
then small changes in input will be amplified
into large changes in output---tiny modifications of the input image
would lead to vastly different predictions.
This does not seem right!

On the flip side, if $\mathbf{A}$ shrinks random vectors to be shorter,
then after running through many layers, the vector will essentially shrink to nothing,
and the output will not depend on the input. This is also clearly not right either!

We need to walk the narrow line between growth and decay
to make sure that our output changes depending on our input, but not much!

Let's see what happens when we repeatedly multiply our matrix $\mathbf{A}$
against a random input vector, and keep track of the norm.

```{.python .input #eigendecomposition-behavior-on-random-data-1}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-behavior-on-random-data-1}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-behavior-on-random-data-1}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-behavior-on-random-data-1}
#@tab jax
# Calculate the sequence of norms after repeatedly applying `A`
v_in = jax.random.normal(jax.random.PRNGKey(1), (k, 1), dtype=jnp.float64)

norm_list = [float(jnp.linalg.norm(v_in))]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(float(jnp.linalg.norm(v_in)))

d2l.plot(jnp.arange(0, 100), norm_list, 'Iteration', 'Value')
```

The norm is growing uncontrollably!
Indeed if we take the list of quotients, we will see a pattern.

```{.python .input #eigendecomposition-behavior-on-random-data-2}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-behavior-on-random-data-2}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-behavior-on-random-data-2}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-behavior-on-random-data-2}
#@tab jax
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(jnp.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

If we look at the last portion of the above computation,
we see that the random vector is stretched by a factor that stabilizes
(the exact numerical value depends on the random seed used by each
framework; see :numref:`subsec_mdl-eig-stretch-back` for the connection to the
largest eigenvalue of $\mathbf{A}$).

### Relating Back to Eigenvectors
:label:`subsec_mdl-eig-stretch-back`

We have seen that eigenvectors and eigenvalues correspond
to the amount something is stretched,
but that was for specific vectors, and specific stretches.
Let's take a look at what they are for $\mathbf{A}$.
A bit of a caveat here: it turns out that to see them all,
we will need to go to complex numbers.
You can think of these as stretches and rotations.
By taking the norm of the complex number
(square root of the sums of squares of real and imaginary parts)
we can measure that stretching factor. Let's also sort them.

```{.python .input #eigendecomposition-relating-back-to-eigenvectors}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input #eigendecomposition-relating-back-to-eigenvectors}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input #eigendecomposition-relating-back-to-eigenvectors}
#@tab tensorflow
# Compute the eigenvalues (A is not symmetric in general, so use eig).
eigs = tf.linalg.eig(A)[0].numpy().tolist()
norm_eigs = [abs(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input #eigendecomposition-relating-back-to-eigenvectors}
#@tab jax
# Compute the eigenvalues
eigs = jnp.linalg.eigvals(A).tolist()
norm_eigs = [abs(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### An Observation

We see something a bit unexpected happening here:
that number we identified before for the
long term stretching of our matrix $\mathbf{A}$
applied to a random vector is *exactly*
(accurate to thirteen decimal places!)
the largest eigenvalue of $\mathbf{A}$.
This is clearly not a coincidence!

But, if we now think about what is happening geometrically,
this starts to make sense. Consider a random vector.
This random vector points a little in every direction,
so in particular, it points at least a little bit
in the same direction as the eigenvector of $\mathbf{A}$
associated with the largest eigenvalue.
This is so important that it is called
the *principal eigenvalue* and *principal eigenvector*.
After applying $\mathbf{A}$, our random vector
gets stretched in every possible direction,
as is associated with every possible eigenvector,
but it is stretched most of all in the direction
associated with this principal eigenvector.
What this means is that after applying $A$,
our random vector is longer, and points in a direction
closer to being aligned with the principal eigenvector.
After applying the matrix many times,
the alignment with the principal eigenvector becomes closer and closer until,
for all practical purposes, our random vector has been transformed
into the principal eigenvector!
Indeed this algorithm is the basis
for what is known as the *power iteration*
for finding the largest eigenvalue and eigenvector of a matrix. For details see, for example, :cite:`Golub.Van-Loan.1996`.

::: {.callout-note title="⟢ Planned — diagram spec (not yet drawn)"}
**Diagram:** `fig_mdl-power-iteration-converge` — a 2-D vector being repeatedly
multiplied by $\mathbf{A}$ and renormalized, drawn as a sequence of arrows that
swing toward (and then lock onto) the dominant eigenvector direction; an inset
shows the norm ratio converging to $|\lambda_1|$. Visualizes why repeated
application aligns any input with the principal eigenvector.
:::

### Fixing the Normalization

Now, from above discussions, we concluded
that we do not want a random vector to be stretched or squished at all,
we would like random vectors to stay about the same size throughout the entire process.
To do so, we now rescale our matrix by this principal eigenvalue
so that the largest eigenvalue is instead now just one.
Let's see what happens in this case.

```{.python .input #eigendecomposition-fixing-the-normalization-1}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-fixing-the-normalization-1}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-fixing-the-normalization-1}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input #eigendecomposition-fixing-the-normalization-1}
#@tab jax
# Rescale the matrix `A`
A = A / norm_eigs[-1]

# Do the same experiment again
v_in = jax.random.normal(jax.random.PRNGKey(2), (k, 1), dtype=jnp.float64)

norm_list = [float(jnp.linalg.norm(v_in))]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(float(jnp.linalg.norm(v_in)))

d2l.plot(jnp.arange(0, 100), norm_list, 'Iteration', 'Value')
```

We can also plot the ratio between consecutive norms as before and see that indeed it stabilizes.

```{.python .input #eigendecomposition-fixing-the-normalization-2}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-fixing-the-normalization-2}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-fixing-the-normalization-2}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input #eigendecomposition-fixing-the-normalization-2}
#@tab jax
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(jnp.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Discussion

We now see exactly what we hoped for!
After normalizing the matrices by the principal eigenvalue,
we see that the random data does not explode as before,
but rather eventually equilibrates to a specific value.
It would be nice to be able to do these things from first principles,
and it turns out that if we look deeply at the mathematics of it,
we can say quite a lot about the *spectral radius*---the largest of the
$|\lambda_i|$, which is exactly the stretching factor we measured above.
For an $n\times n$ matrix with independent mean-zero, variance-one entries,
the eigenvalues fill the disk of radius $\sqrt{n}$ in the complex plane
roughly uniformly, a fact known as the *circular law* :cite:`Ginibre.1965`.
The spectral radius is therefore not a fixed number at finite $n$; rather, as
$n \to \infty$ it converges to $\sqrt{n}$ *from above* (the largest $|\lambda_i|$
sits just outside the bulk of the disk). For our small $5\times 5$ example we
should not expect to land exactly on $\sqrt{5} \approx 2.2$---finite-size
fluctuations are substantial---but the value we measured is in the right
ballpark, and the *scaling* with $\sqrt{n}$ is what matters for initialization.
A word of caution: the spectral radius is *not* the same as the largest
*singular value* of such a matrix, which by the Marchenko–Pastur law sits near
$2\sqrt{n}$ at finite $n$; we return to singular values in
:numref:`sec_mdl-svd-low-rank`.
The relationship between the eigenvalues (and the related singular values) of random matrices has been shown to have deep connections to proper initialization of neural networks as was discussed in :citet:`Pennington.Schoenholz.Ganguli.2017` and subsequent works.

## Summary
* Eigenvectors are vectors which are stretched by a matrix without changing direction.
* Eigenvalues are the amount that the eigenvectors are stretched by the application of the matrix.
* The eigendecomposition of a matrix can allow for many operations to be reduced to operations on the eigenvalues.
* The Gershgorin Circle Theorem can provide approximate values for the eigenvalues of a matrix.
* The behavior of iterated matrix powers depends primarily on the size of the largest eigenvalue.  This understanding has many applications in the theory of neural network initialization.

## Exercises
1. What are the eigenvalues and eigenvectors of
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  What are the eigenvalues and eigenvectors of the following matrix, and what is strange about this example compared to the previous one?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Without computing the eigenvalues, is it possible that the smallest eigenvalue of the following matrix is less than $0.5$? *Note*: this problem can be done in your head.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1087)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1087)
:end_tab:

<!-- slides -->

::: {.slide title="Eigenvectors and Dynamics"}
A square matrix $\mathbf{A}$ has **eigenvalue** $\lambda$
and **eigenvector** $\mathbf{v}$ when

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.$$

Geometrically: $\mathbf{A}$ stretches $\mathbf{v}$ by
$\lambda$ but doesn't rotate it. If $\mathbf{A}$ is
diagonalizable: $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$
— a basis change in which the action is just stretching
along axes.

Why we care: matrix powers $\mathbf{A}^t$ are governed by
$\lambda^t$. Repeated application of $\mathbf{A}$ aligns
arbitrary inputs with the dominant eigenvector. That's the
heart of vanishing/exploding gradients in RNNs, of
PageRank, and of every iterative solver.
:::

::: {.slide title="A concrete example"}
Use a small matrix so the geometry is visible: applying
$\mathbf{A}$ to an eigenvector changes scale but not direction.

@eigendecomposition-an-example
:::

::: {.slide title="Gershgorin circles"}
Cheap eigenvalue bounds without computing them:
eigenvalues lie in the union of disks centered at
$a_{ii}$ with radius $\sum_{j \ne i} |a_{ij}|$. Useful for
stability arguments:

@eigendecomposition-gershgorin-circle-theorem
:::

::: {.slide title="Eigenvectors govern long-run behavior"}
Power iteration: keep multiplying by $\mathbf{A}$. The
direction converges to the leading eigenvector; the norm
grows like $\lambda_1^t$:

@eigendecomposition-eigenvectors-as-long-term-behavior

. . .

@eigendecomposition-behavior-on-random-data-1

. . .

@eigendecomposition-behavior-on-random-data-2
:::

::: {.slide title="Relating back"}
After repeated multiplication, normalize the vector to read off the
direction; the scale factor estimates the dominant eigenvalue.

@eigendecomposition-relating-back-to-eigenvectors

. . .

@eigendecomposition-fixing-the-normalization-1

. . .

@eigendecomposition-fixing-the-normalization-2
:::

::: {.slide title="Recap"}
- $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$: $\mathbf{A}$
  acts as scaling along the eigenvector axes.
- Largest $|\lambda|$ controls long-run iterated dynamics.
- Symmetric matrices have orthonormal eigenvectors and
  real eigenvalues — the basis for PCA.
- Vanishing/exploding RNN gradients = "iterated map" with
  bad spectral radius.
:::
