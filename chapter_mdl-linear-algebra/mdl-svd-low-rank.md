# Singular Value Decomposition and Low-Rank Approximation
:label:`sec_mdl-svd-low-rank`

The eigendecomposition of :numref:`sec_mdl-eigendecompositions` was powerful but
*picky*: it needs a square matrix, and to be fully well-behaved---an orthonormal
eigenbasis, real eigenvalues---it really wants a *symmetric* one. The defective
shear there had a repeated eigenvalue but only a one-dimensional eigenspace, so it
admitted no eigenbasis at all. The *singular value decomposition* (SVD) is the
same idea made universal. It applies to **every** matrix---rectangular,
rank-deficient, non-symmetric, defective---and factors it into a clean
*rotate--scale--rotate* form whose scale factors, the *singular values*, are the
right generalization of eigenvalues.

The engine that makes this work is one we already built. The spectral theorem of
:numref:`subsec_mdl-spectral-theorem` guarantees an orthonormal eigenbasis for
*symmetric* matrices, and :numref:`subsec_mdl-psd` showed that
$\mathbf{A}^\top\mathbf{A}$ is always symmetric and positive semidefinite. Feeding
that matrix through the spectral theorem manufactures an SVD for *any* $\mathbf{A}$
whatsoever---so the SVD never fails for the simple reason that Gram matrices are
never defective. From this one factorization, a remarkable amount falls out:
rank and the four fundamental subspaces (resolving the "efficient rank" promise of
:numref:`sec_mdl-geometry-linear-algebraic-ops`), the *optimal* low-rank
approximation, principal component analysis, the pseudoinverse and least squares,
and the condition number that predicts numerical trouble and gradient-descent
zig-zag. It also underwrites a large share of practical deep learning: PCA, the
Eckart--Young truncation behind LoRA and other low-rank adapters, and the spectral
diagnostics used to inspect and constrain trained weights---including spectral
normalization, which estimates the largest singular value with the very power
iteration we met in :numref:`sec_mdl-eigendecompositions`.

We first load the per-framework library so the figures and computations below have
`d2l` and `np` in scope. The proofs and the framework-agnostic figures use plain
NumPy; only the small worked-verification cells branch per framework.

```{.python .input #svd-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np
```

```{.python .input #svd-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np
import torch
```

```{.python .input #svd-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np
import tensorflow as tf
```

```{.python .input #svd-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
from IPython import display
import numpy as np
import jax
from jax import numpy as jnp
```

## The SVD: Rotate--Scale--Rotate
:label:`subsec_mdl-svd-rotate-scale-rotate`

Recall the central picture of :numref:`sec_mdl-eigendecompositions`: a matrix
sends the unit circle to an ellipse. For a *symmetric* matrix, the ellipse's axes
lie along the eigenvectors, and one orthonormal frame does the whole job. A
general matrix bends the circle to an ellipse too---but now the input directions
it stretches cleanly and the output directions those map to are *two different*
orthonormal frames. The SVD names both.

**Theorem (Singular Value Decomposition).** *Every real $m\times n$ matrix
$\mathbf{A}$ can be written as*

$$
\mathbf{A} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top,
$$
:eqlabel:`eq_mdl-svd`

*where $\mathbf{U}\in\mathbb{R}^{m\times m}$ and $\mathbf{V}\in\mathbb{R}^{n\times
n}$ are orthogonal ($\mathbf{U}^\top\mathbf{U}=\mathbf{I}$,
$\mathbf{V}^\top\mathbf{V}=\mathbf{I}$) and $\boldsymbol{\Sigma}\in\mathbb{R}^{m\times
n}$ is "diagonal" with non-negative entries
$\sigma_1\ge\sigma_2\ge\cdots\ge0$ on its main diagonal and zeros elsewhere.* The
columns $\mathbf{v}_i$ of $\mathbf{V}$ are the *right singular vectors*, the
columns $\mathbf{u}_i$ of $\mathbf{U}$ are the *left singular vectors*, and the
$\sigma_i$ are the *singular values*.

We prove existence in :numref:`subsec_mdl-svd-via-ata`; first we read off what it
*means*. Because $\mathbf{V}$ and $\mathbf{U}$ are orthogonal, they are pure
rotations (possibly with a reflection), and :eqref:`eq_mdl-svd` decomposes the
action of $\mathbf{A}$ on a vector $\mathbf{x}$ into three stages applied right to
left:

1. $\mathbf{V}^\top$ **rotates** the input so that the right singular vectors
   $\mathbf{v}_i$ land on the coordinate axes;
2. $\boldsymbol{\Sigma}$ **scales** axis $i$ by $\sigma_i$ (and, if
   $m\neq n$, embeds into or projects onto the output dimension);
3. $\mathbf{U}$ **rotates** the scaled axes into the output frame, placing the
   stretched axis $i$ along the left singular vector $\mathbf{u}_i$.

Equivalently, reading off column $i$ of $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$,

$$
\mathbf{A}\mathbf{v}_i = \sigma_i\,\mathbf{u}_i .
$$
:eqlabel:`eq_mdl-svd-action`

The orthonormal input direction $\mathbf{v}_i$ is sent to the orthonormal output
direction $\mathbf{u}_i$, stretched by $\sigma_i$. The unit circle (sphere) of
right singular vectors becomes an ellipse (ellipsoid) whose semi-axes are the
$\sigma_i\mathbf{u}_i$. This is exactly the eigen-picture of
:numref:`sec_mdl-eigendecompositions`, generalized: there one frame served as both
input and output axes because the map was symmetric; here input frame $\mathbf{V}$
and output frame $\mathbf{U}$ are allowed to differ, which is precisely what lets
the SVD handle every matrix. The figure below draws this for a non-symmetric
$2\times2$ matrix, so that $\mathbf{U}\neq\mathbf{V}$ is visible: the input
singular vectors $\mathbf{v}_1,\mathbf{v}_2$ are orthogonal, and so are the output
axes $\sigma_1\mathbf{u}_1,\sigma_2\mathbf{u}_2$, but the two frames point in
different directions.

```{.python .input #svd-fig-rotate-scale-rotate}
import numpy as np

def plot_svd_action():
    A = np.array([[1.4, 1.2], [0.0, 1.0]])   # non-symmetric: U != V
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    fig, ax = d2l.plt.subplots(1, 2, figsize=(7.8, 3.8))
    # Input: unit circle + right singular vectors v_i
    ax[0].plot(circle[0], circle[1], color='gray', lw=1.2, label='unit circle')
    for i in range(2):
        ax[0].annotate('', xy=(V[0, i], V[1, i]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='C2', lw=2))
        ax[0].text(V[0, i] * 1.12, V[1, i] * 1.12, rf'$v_{i+1}$', color='C2')
    ax[0].set_title('input: right singular vectors', fontsize=10)
    # Output: image ellipse + scaled left singular vectors sigma_i u_i
    ax[1].plot(ellipse[0], ellipse[1], color='C0', lw=2, label='image ellipse')
    for i in range(2):
        ax[1].annotate('', xy=(s[i] * U[0, i], s[i] * U[1, i]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='C3', lw=2))
        ax[1].text(s[i] * U[0, i] * 1.08, s[i] * U[1, i] * 1.08,
                   rf'$\sigma_{i+1}u_{i+1}$', color='C3', fontsize=9)
    ax[1].set_title('output: image $=$ ellipse with semi-axes $\\sigma_i u_i$',
                    fontsize=10)
    for a in ax:
        a.set_aspect('equal'); a.grid(alpha=.3)
        a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6)
    ax[0].legend(fontsize=8, loc='upper left')
    d2l.plt.tight_layout()

plot_svd_action()
```

Notice that the right singular vectors in the left panel are *not* aligned with
the left singular vectors in the right panel---the matrix rotates as well as
stretches, and the SVD separates the two rotations from the single stretch in
between.

**The polar decomposition: rotate--scale--rotate, rigorously.** The phrase
"rotate--scale--rotate" can be made into a clean statement. For square
$\mathbf{A}$, insert $\mathbf{V}\mathbf{V}^\top=\mathbf{I}$:

$$
\mathbf{A}
  = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top
  = (\mathbf{U}\mathbf{V}^\top)\,(\mathbf{V}\boldsymbol{\Sigma}\mathbf{V}^\top)
  = \mathbf{Q}\,\mathbf{P}.
$$
:eqlabel:`eq_mdl-polar`

Here $\mathbf{Q}=\mathbf{U}\mathbf{V}^\top$ is orthogonal (a product of
orthogonal matrices) and $\mathbf{P}=\mathbf{V}\boldsymbol{\Sigma}\mathbf{V}^\top$
is symmetric positive semidefinite (its eigendecomposition has the non-negative
$\sigma_i$ as eigenvalues). This *polar decomposition* says every linear map is a
positive-semidefinite stretch $\mathbf{P}$ followed by a rigid rotation
$\mathbf{Q}$---the exact matrix analog of writing a complex number as
$z=re^{i\theta}$, modulus times phase. It justifies the rotate--scale--rotate
slogan: the "scale" is $\mathbf{P}$ and the net "rotate" is $\mathbf{Q}$.

**Thin SVD and the dyadic sum.** If $r=\operatorname{rank}\mathbf{A}$, only the
first $r$ singular values are nonzero. Discarding the zero columns of
$\boldsymbol{\Sigma}$ and the matching singular vectors gives the *thin* (or
*economy*) SVD $\mathbf{A}=\mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^\top$ with
$\mathbf{U}_r\in\mathbb{R}^{m\times r}$, $\mathbf{V}_r\in\mathbb{R}^{n\times r}$.
Multiplying out, the SVD is a sum of rank-one *dyads*,

$$
\mathbf{A} = \sum_{i=1}^{r} \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,
$$
:eqlabel:`eq_mdl-svd-dyadic`

each term a single outer product weighted by its singular value, ordered from
largest to smallest. This form---a *ranked* list of rank-one ingredients---is the
key to the low-rank approximation of :numref:`subsec_mdl-eckart-young`: keep the
heavy terms, drop the light ones.

## SVD via the Eigendecomposition of $\mathbf{A}^\top\mathbf{A}$
:label:`subsec_mdl-svd-via-ata`

The SVD is not a new mystery to be proved from scratch. It is the spectral theorem
of :numref:`subsec_mdl-spectral-theorem` applied to the symmetric PSD matrix
$\mathbf{A}^\top\mathbf{A}$, which we built precisely for this purpose in the
bridge at the end of that section. The following constructive proof is the heart
of the section; the one elegant step in it---getting orthonormality of the
$\mathbf{u}_i$ for free---is worth pausing on.

**Proof of :eqref:`eq_mdl-svd` (existence).** The matrix
$\mathbf{A}^\top\mathbf{A}$ is symmetric and positive semidefinite, because

$$
\mathbf{x}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{x} = \|\mathbf{A}\mathbf{x}\|^2 \ge 0 .
$$

By the spectral theorem it has an orthonormal eigenbasis
$\mathbf{v}_1,\ldots,\mathbf{v}_n$ with real eigenvalues, and by
:numref:`subsec_mdl-psd` those eigenvalues are non-negative; order them
$\lambda_1\ge\cdots\ge\lambda_n\ge0$ and set

$$
\sigma_i = \sqrt{\lambda_i}, \qquad r = \#\{i : \sigma_i > 0\} .
$$

For each $i\le r$ define $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$. These
left singular vectors are automatically orthonormal---and this single line is the
crux of the whole construction:

$$
\mathbf{u}_i^\top\mathbf{u}_j
  = \frac{1}{\sigma_i\sigma_j}\,\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\,\mathbf{v}_j
  = \frac{1}{\sigma_i\sigma_j}\,\mathbf{v}_i^\top(\lambda_j\mathbf{v}_j)
  = \frac{\lambda_j}{\sigma_i\sigma_j}\,\delta_{ij}
  = \delta_{ij} .
$$

Orthonormality of the *output* frame is inherited from orthonormality of the
*input* frame, routed through $\mathbf{A}^\top\mathbf{A}$. Extend
$\mathbf{u}_1,\ldots,\mathbf{u}_r$ to an orthonormal basis
$\mathbf{u}_1,\ldots,\mathbf{u}_m$ of $\mathbb{R}^m$ (Gram--Schmidt on any
completion). For the remaining right singular vectors, $i>r$, we have
$\|\mathbf{A}\mathbf{v}_i\|^2=\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_i=\lambda_i=0$,
so $\mathbf{A}\mathbf{v}_i=\mathbf 0$. In all cases, then,
$\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ (with $\sigma_i=0$ for $i>r$).
Collecting these columns gives $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$,
and right-multiplying by $\mathbf{V}^\top=\mathbf{V}^{-1}$ yields
$\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. $\blacksquare$

This proof also *computes* the SVD by hand: diagonalize the (small, symmetric)
matrix $\mathbf{A}^\top\mathbf{A}$, take square roots for the singular values, and
push the eigenvectors through $\mathbf{A}$ for the left singular vectors.

**Uniqueness.** The singular *values* are unique---they are the square roots of
the eigenvalues of $\mathbf{A}^\top\mathbf{A}$, which are determined by
$\mathbf{A}$. The singular *vectors* are unique only up to the same ambiguities we
met for eigenvectors in :numref:`sec_mdl-eigendecompositions`: a sign flip on a
pair $(\mathbf{u}_i,\mathbf{v}_i)$ for a simple $\sigma_i$, and an arbitrary
orthonormal rotation within the subspace belonging to a repeated singular value.

**Variational characterization of $\sigma_1$.** There is a second, more
illuminating route to the top singular value that needs no eigendecomposition and
makes the *meaning* of $\sigma_1$ plain. It is the Rayleigh quotient of
:numref:`subsec_mdl-rayleigh` in disguise:

$$
\sigma_1 = \max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\| = \|\mathbf{A}\|_2 ,
$$
:eqlabel:`eq_mdl-sigma1-variational`

because $\|\mathbf{A}\mathbf{x}\|^2=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$
is maximized over unit $\mathbf{x}$ by the Rayleigh proposition at
$\lambda_1=\sigma_1^2$, attained at $\mathbf{x}=\mathbf{v}_1$. So **$\sigma_1$ is
the most the matrix can stretch any unit vector**---its operator (spectral) norm
$\|\mathbf{A}\|_2$. The remaining singular values come from the same problem with
deflation: $\sigma_k=\max\{\|\mathbf{A}\mathbf{x}\| : \|\mathbf{x}\|=1,\ \mathbf{x}\perp\mathbf{v}_1,\ldots,\mathbf{v}_{k-1}\}$.
This "maximum stretch" reading is the one that recurs in Eckart--Young, PCA,
conditioning, and Lipschitz/spectral-norm arguments. The constructive proof is the
better *first* proof (concrete, reusing machinery we have); the variational one is
the better *meaning*.

**Relationship to the eigendecomposition.** Substituting :eqref:`eq_mdl-svd` into
the two Gram matrices gives, since $\mathbf{U}$ and $\mathbf{V}$ are orthogonal,

$$
\mathbf{A}^\top\mathbf{A} = \mathbf{V}\,\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\,\mathbf{V}^\top,
\qquad
\mathbf{A}\mathbf{A}^\top = \mathbf{U}\,\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top\,\mathbf{U}^\top .
$$
:eqlabel:`eq_mdl-svd-gram`

These are *eigendecompositions*: the right singular vectors are the eigenvectors
of $\mathbf{A}^\top\mathbf{A}$, the left singular vectors are the eigenvectors of
$\mathbf{A}\mathbf{A}^\top$, and the squared singular values $\sigma_i^2$ are the
shared eigenvalues. For a symmetric PSD matrix the SVD and the eigendecomposition
coincide. In general they differ, and the cleanest warning against conflating
singular values with eigenvalues is the rotation matrix

$$
\mathbf{A} = \begin{bmatrix} 0 & -2\\ 1 & \phantom{-}0\end{bmatrix},
\qquad
\mathbf{A}^\top\mathbf{A} = \begin{bmatrix} 1 & 0\\ 0 & 4\end{bmatrix},
$$

whose singular values are $\{2,1\}$ (the square roots of $\{4,1\}$), while its
eigenvalues are $\pm i\sqrt2$ with modulus $|\lambda|=\sqrt2$. The singular values
are *not* the eigenvalue magnitudes; here even their *geometric mean*
$\sqrt{\sigma_1\sigma_2}=\sqrt2$ equals $|\lambda|$, which is no accident---it is
the matrix analog of the rotation-with-scaling reading from
:numref:`subsec_mdl-complex-rotation`, with $|\det\mathbf{A}|=\sigma_1\sigma_2=|\lambda_1\lambda_2|=2$.

### The Defective Shear, Finally Decomposed
:label:`subsec_mdl-defective-shear-svd`

We can now keep the promise made twice in :numref:`sec_mdl-eigendecompositions`.
The shear

$$
\mathbf{A} = \begin{bmatrix} 1 & 1\\ 0 & 1\end{bmatrix}
$$

is *defective*: $\lambda=1$ has algebraic multiplicity $2$ but only a
one-dimensional eigenspace, so it has **no eigenbasis** and the eigendecomposition
simply does not exist. The SVD has no such trouble. Form

$$
\mathbf{A}^\top\mathbf{A} = \begin{bmatrix} 1 & 1\\ 1 & 2\end{bmatrix},
$$

a symmetric PSD matrix. Its characteristic polynomial is
$\lambda^2-3\lambda+1$, with roots $\lambda_{1,2}=(3\pm\sqrt5)/2$. Hence the
singular values are

$$
\sigma_1 = \sqrt{\tfrac{3+\sqrt5}{2}} = \frac{1+\sqrt5}{2} = \varphi \approx 1.618,
\qquad
\sigma_2 = \sqrt{\tfrac{3-\sqrt5}{2}} = \frac{1}{\varphi} = \varphi-1 \approx 0.618,
$$

the golden ratio and its reciprocal (consistent with
$\sigma_1\sigma_2=|\det\mathbf{A}|=1$). Both are strictly positive, so the shear
has full rank $2$ and a perfectly clean orthonormal factorization
$\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$---there is nothing
defective about it. The eigendecomposition stumbled because it insists on a
*single* basis that both diagonalizes and is reused for input and output; the SVD
succeeds because it is allowed two different orthonormal frames. This is the
payoff of the whole construction: *the SVD repairs exactly what the
eigendecomposition could not.* The cell below verifies the singular values against
the golden ratio.

```{.python .input #svd-defective-shear}
import numpy as np

A = np.array([[1., 1.], [0., 1.]])
U, s, Vt = np.linalg.svd(A)
phi = (1 + np.sqrt(5)) / 2
print('singular values of the shear:', s.round(6))
print('golden ratio phi, 1/phi:    ', np.array([phi, 1 / phi]).round(6))
print('product sigma_1 * sigma_2 = |det A| =', round(s[0] * s[1], 6))
```

## Rank, Range, and Null Space from the Spectrum
:label:`subsec_mdl-svd-subspaces`

Because $\mathbf{U}$ and $\mathbf{V}$ are invertible, multiplying by them changes
no dimensions, so $\operatorname{rank}\mathbf{A}=\operatorname{rank}\boldsymbol{\Sigma}$,
which is simply the number of nonzero diagonal entries:

$$
\operatorname{rank}\mathbf{A} = \#\{i : \sigma_i > 0\} = r .
$$
:eqlabel:`eq_mdl-rank-sigma`

This is the *efficient* characterization of rank promised in
:numref:`sec_mdl-geometry-linear-algebraic-ops`, where counting independent
columns by elimination was the only tool on offer. The dyadic form
:eqref:`eq_mdl-svd-dyadic` makes the four fundamental subspaces equally explicit.
Splitting the singular vectors at the index $r$,

* the **range** (column space) of $\mathbf{A}$ is
  $\operatorname{span}\{\mathbf{u}_1,\ldots,\mathbf{u}_r\}$, since every output
  $\mathbf{A}\mathbf{x}=\sum_{i\le r}\sigma_i(\mathbf{v}_i^\top\mathbf{x})\mathbf{u}_i$
  is a combination of these;
* the **null space** of $\mathbf{A}$ is
  $\operatorname{span}\{\mathbf{v}_{r+1},\ldots,\mathbf{v}_n\}$, the input
  directions sent to zero;
* the **row space** (range of $\mathbf{A}^\top$) is
  $\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_r\}$;
* the **left null space** (null space of $\mathbf{A}^\top$) is
  $\operatorname{span}\{\mathbf{u}_{r+1},\ldots,\mathbf{u}_m\}$.

The picture is a clean bijection: $\mathbf{A}$ maps the row space to the column
space, sending each $\mathbf{v}_i\mapsto\sigma_i\mathbf{u}_i$ ($i\le r$) one-to-one
and onto, while crushing the null space to zero. Input space splits orthogonally
as row space $\oplus$ null space; output space splits orthogonally as column space
$\oplus$ left-null space. The figure below draws this two-plane map; the
pseudoinverse of :numref:`subsec_mdl-pseudoinverse` will run the bijection
backwards.

```{.python .input #svd-fig-four-subspaces}
import numpy as np

def plot_four_subspaces():
    fig, ax = d2l.plt.subplots(figsize=(7.2, 4.2))
    ax.axis('off')
    # input space (left) and output space (right) as two stacked boxes each
    def box(x, y, w, h, color, label):
        ax.add_patch(d2l.plt.Rectangle((x, y), w, h, fc=color, ec='k',
                                       alpha=.35, lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=9)
    box(0.2, 1.15, 1.6, 0.9, 'C2', r'row space' + '\n' +
        r'$\mathrm{span}\{v_1,\dots,v_r\}$')
    box(0.2, 0.05, 1.6, 0.9, 'C7', r'null space' + '\n' +
        r'$\mathrm{span}\{v_{r+1},\dots,v_n\}$')
    box(4.2, 1.15, 1.6, 0.9, 'C3', r'column space' + '\n' +
        r'$\mathrm{span}\{u_1,\dots,u_r\}$')
    box(4.2, 0.05, 1.6, 0.9, 'C7', r'left null space' + '\n' +
        r'$\mathrm{span}\{u_{r+1},\dots,u_m\}$')
    ax.annotate('', xy=(4.15, 1.6), xytext=(1.85, 1.6),
                arrowprops=dict(arrowstyle='->', color='C0', lw=2))
    ax.text(3.0, 1.78, r'$v_i \mapsto \sigma_i u_i$ (bijective)',
            ha='center', color='C0', fontsize=9)
    ax.annotate('', xy=(4.15, 0.5), xytext=(1.85, 0.95),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                ls='--'))
    ax.text(3.0, 0.55, r'$\mapsto \mathbf{0}$', ha='center', color='gray',
            fontsize=9)
    ax.text(1.0, 2.25, r'input $\mathbb{R}^n$', ha='center', fontsize=10)
    ax.text(5.0, 2.25, r'output $\mathbb{R}^m$', ha='center', fontsize=10)
    ax.set_xlim(-0.1, 6.1); ax.set_ylim(-0.1, 2.5)

plot_four_subspaces()
```

**Numerical rank.** In floating-point arithmetic, a matrix that is mathematically
rank-deficient rarely has exact zero singular values; rounding leaves tiny
$\sigma_i$ of size around $\epsilon_{\text{mach}}\,\sigma_1$ instead. The honest
notion of rank therefore *thresholds*: count the singular values above a tolerance.
The cell below builds a deliberately rank-2 matrix in $\mathbb{R}^{4\times4}$ and
shows two singular values collapse to near machine zero, so a tolerance recovers
the true rank where a test for exact zeros would fail.

```{.python .input #svd-numerical-rank}
import numpy as np

rng = np.random.default_rng(0)
B = rng.standard_normal((4, 2))
C = rng.standard_normal((2, 4))
A = B @ C                       # rank at most 2 by construction
s = np.linalg.svd(A, compute_uv=False)
print('singular values:', s.round(6))
tol = s[0] * max(A.shape) * np.finfo(float).eps
print('numerical rank (tol =', f'{tol:.2e}):', int((s > tol).sum()))
```

## Eckart--Young: Optimal Low-Rank Approximation
:label:`subsec_mdl-eckart-young`

Here is the theorem that makes the SVD indispensable. The dyadic sum
:eqref:`eq_mdl-svd-dyadic` lists the rank-one pieces of $\mathbf{A}$ in order of
importance. Keeping only the top $k$,

$$
\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,
$$
:eqlabel:`eq_mdl-truncated-svd`

is not merely *a* rank-$k$ approximation of $\mathbf{A}$---it is the *provably
best* one, in both the spectral and Frobenius norms. Eckart and Young (1936)
proved the Frobenius case and Mirsky (1960) extended it to every unitarily
invariant norm; we give the spectral-norm statement and proof in full because the
argument is beautiful.

**Theorem (Eckart--Young--Mirsky).** *For every $k<r$,*

$$
\min_{\operatorname{rank}\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_2 = \sigma_{k+1},
$$
:eqlabel:`eq_mdl-eckart-young`

*and the minimum is attained at $\mathbf{B}=\mathbf{A}_k$. In the Frobenius norm,
$\min_{\operatorname{rank}\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_F^2=\sum_{i>k}\sigma_i^2$,
again attained at $\mathbf{A}_k$.*

**Proof.** *The truncation achieves $\sigma_{k+1}$.* The error
$\mathbf{A}-\mathbf{A}_k=\sum_{i>k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ is itself
a (sub-)SVD whose largest singular value is $\sigma_{k+1}$, so by the variational
identity :eqref:`eq_mdl-sigma1-variational`,
$\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$. (And
$\|\mathbf{A}-\mathbf{A}_k\|_F^2=\sum_{i>k}\sigma_i^2$, since the Frobenius norm is
the root-sum-of-squares of the singular values.)

*No rank-$k$ matrix does better (the dimension-counting argument).* Let
$\mathbf{B}$ be any matrix with $\operatorname{rank}\mathbf{B}\le k$. Its null
space has dimension at least $n-k$. Consider also the
$(k{+}1)$-dimensional subspace
$\mathcal{V}=\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}\}$ spanned by
the top $k{+}1$ right singular vectors. Two subspaces of $\mathbb{R}^n$ whose
dimensions add to more than $n$ must intersect nontrivially:

$$
\dim(\ker\mathbf{B}\cap\mathcal{V}) \ge (n-k) + (k+1) - n = 1 .
$$

So there is a *unit* vector $\mathbf{x}\in\ker\mathbf{B}\cap\mathcal{V}$. Write it
in the top singular basis, $\mathbf{x}=\sum_{i\le k+1}c_i\mathbf{v}_i$ with
$\sum c_i^2=1$. Because $\mathbf{x}\in\ker\mathbf{B}$ we have $\mathbf{B}\mathbf{x}=\mathbf 0$,
and because $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ with the $\mathbf{u}_i$
orthonormal,

$$
\|\mathbf{A}\mathbf{x}\|^2 = \Bigl\|\sum_{i\le k+1}c_i\sigma_i\mathbf{u}_i\Bigr\|^2
   = \sum_{i\le k+1} c_i^2\sigma_i^2
   \ge \sigma_{k+1}^2 \sum_{i\le k+1} c_i^2
   = \sigma_{k+1}^2 ,
$$

using $\sigma_i\ge\sigma_{k+1}$ for $i\le k+1$. Therefore

$$
\|\mathbf{A}-\mathbf{B}\|_2^2
  \ge \|(\mathbf{A}-\mathbf{B})\mathbf{x}\|^2
  = \|\mathbf{A}\mathbf{x}\|^2 \ge \sigma_{k+1}^2 ,
$$

so $\|\mathbf{A}-\mathbf{B}\|_2\ge\sigma_{k+1}$ for every rank-$k$ $\mathbf{B}$,
with equality at $\mathbf{A}_k$. $\blacksquare$

The pivot is the collision: the kernel of $\mathbf{B}$ and the top-$(k{+}1)$
singular subspace *overfill* $\mathbb{R}^n$, so they must share a direction. On
that shared vector $\mathbf{B}$ is blind ($\mathbf{B}\mathbf{x}=\mathbf 0$) while
$\mathbf{A}$ still stretches by at least $\sigma_{k+1}$---so no rank-$k$
$\mathbf{B}$ can track $\mathbf{A}$ everywhere. (The Frobenius optimality needs a
little more---Weyl's inequalities for singular values, see
:cite:`Golub.Van-Loan.1996`---but the conclusion and the optimal $\mathbf{A}_k$ are
the same.)

The error formulas give a quantitative *dial*. The fraction of "energy" retained
by the rank-$k$ truncation is the *energy ratio*

$$
\frac{\sum_{i\le k}\sigma_i^2}{\sum_{i}\sigma_i^2}
  = 1 - \frac{\|\mathbf{A}-\mathbf{A}_k\|_F^2}{\|\mathbf{A}\|_F^2} ,
$$
:eqlabel:`eq_mdl-energy-ratio`

so choosing $k$ to capture, say, 95% of the energy is a principled way to set the
rank. When the singular values decay quickly---as they do for natural images and,
empirically, for many trained weight matrices---a small $k$ captures almost
everything, which is exactly the regime in which low-rank compression pays off.

The figure below makes this concrete on a grayscale image. The left panel plots
the singular-value spectrum on a log scale (note the rapid decay); the remaining
panels reconstruct the image at ranks $k=1,5,20$ and full, each labeled with its
rank, compression ratio, and the relative Frobenius error
$\|\mathbf{A}-\mathbf{A}_k\|_F/\|\mathbf{A}\|_F=\sqrt{\sum_{i>k}\sigma_i^2/\sum_i\sigma_i^2}$.
We build a deterministic synthetic image so the figure renders without any
download; the same code works verbatim on `scipy.datasets.ascent()` or any
photograph loaded as a 2-D array.

```{.python .input #svd-fig-eckart-young}
import numpy as np

def make_image(n=160):
    # A deterministic, structured grayscale image (smooth gradients, a disk,
    # bars, a checker patch) so the spectrum decays like a natural image but
    # needs no download. Works identically on scipy.datasets.ascent().
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.5 + 0.4 * np.sin(6 * np.pi * x) * np.cos(2 * np.pi * y)
    img += 0.3 * np.exp(-((x - 0.3) ** 2 + (y - 0.65) ** 2) / 0.02)  # blob
    disk = ((x - 0.7) ** 2 + (y - 0.35) ** 2) < 0.03
    img[disk] = 0.1
    img[10:30, :] = 0.9                                              # bright bar
    checker = (((x * 8).astype(int) + (y * 8).astype(int)) % 2)[120:150, 100:140]
    img[120:150, 100:140] = 0.2 + 0.6 * checker
    return img

def plot_eckart_young():
    A = make_image()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    total = (s ** 2).sum()
    ranks = [1, 5, 20, len(s)]
    fig, ax = d2l.plt.subplots(1, 5, figsize=(12.5, 3.0))
    ax[0].semilogy(np.arange(1, len(s) + 1), s, '.-', ms=3)
    ax[0].set_xlabel('index $i$'); ax[0].set_ylabel(r'$\sigma_i$ (log)')
    ax[0].set_title('singular spectrum', fontsize=9); ax[0].grid(alpha=.3)
    for j, k in enumerate(ranks):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        rel_err = np.sqrt((s[k:] ** 2).sum() / total) if k < len(s) else 0.0
        comp = k * (A.shape[0] + A.shape[1]) / (A.shape[0] * A.shape[1])
        ax[j + 1].imshow(np.clip(Ak, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax[j + 1].set_title(
            f'$k={k}$\n{comp:.0%} storage, err {rel_err:.1%}', fontsize=9)
        ax[j + 1].set_xticks([]); ax[j + 1].set_yticks([])
    d2l.plt.tight_layout()

plot_eckart_young()
```

A rank-20 truncation of this $160\times160$ image already looks essentially
correct while storing only a fraction of the numbers, because the discarded
singular values carry little energy---a visual proof of Eckart--Young.

## PCA as the Worked Example
:label:`subsec_mdl-pca`

Principal component analysis is the most important single application of the SVD,
and Eckart--Young turns it from a recipe into a theorem. Given data points as the
rows of $\mathbf{X}\in\mathbb{R}^{n\times d}$, first *center* them by subtracting
the mean row, $\tilde{\mathbf{X}}=\mathbf{X}-\mathbf 1\bar{\mathbf{x}}^\top$. The
empirical covariance is the symmetric PSD matrix
$\mathbf{C}=\tfrac1n\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ from
:numref:`subsec_mdl-psd`. PCA asks: which unit direction $\mathbf{w}$ captures the
most variance of the projected data?

**Proposition (PCA via Rayleigh).** *The projected variance
$\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$ is
maximized over unit $\mathbf{w}$ by the top eigenvector of $\mathbf{C}$, which is
the top right singular vector $\mathbf{v}_1$ of $\tilde{\mathbf{X}}$, with maximal
variance $\sigma_1^2/n$.*

**Proof.** The projection of a centered point onto $\mathbf{w}$ has coordinate
$\tilde{\mathbf{x}}^\top\mathbf{w}$; the variance of these coordinates is
$\tfrac1n\sum_i(\tilde{\mathbf{x}}_i^\top\mathbf{w})^2=\tfrac1n\|\tilde{\mathbf{X}}\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{C}\mathbf{w}$.
Maximizing this Rayleigh quotient over unit $\mathbf{w}$ gives, by
:numref:`subsec_mdl-rayleigh`, the top eigenvector of $\mathbf{C}$ and the value
$\lambda_1(\mathbf{C})$. By :eqref:`eq_mdl-svd-gram`,
$\mathbf{C}=\tfrac1n\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$,
so its eigenvectors are the right singular vectors $\mathbf{v}_i$ of
$\tilde{\mathbf{X}}$ and its eigenvalues are $\sigma_i^2/n$. $\blacksquare$

Iterating with deflation, the top-$k$ principal directions are
$\mathbf{v}_1,\ldots,\mathbf{v}_k$, and the variance *explained* by component $i$
is $\sigma_i^2/n$. The ranked list of these variances is the *scree plot*, and the
explained-variance ratio is exactly the energy ratio
:eqref:`eq_mdl-energy-ratio` of $\tilde{\mathbf{X}}$. Moreover the rank-$k$
projection $\mathbf{z}=\mathbf{V}_k^\top(\mathbf{x}-\bar{\mathbf{x}})$ is the
*optimal* linear dimensionality reduction: minimizing reconstruction error over
all rank-$k$ linear maps is Eckart--Young--Frobenius applied to $\tilde{\mathbf{X}}$,
whose solution is the top-$k$ truncation. **Centering matters**: skip it and the
first singular vector is dominated by the offset of the cloud from the origin,
capturing where the data *is* rather than how it *varies*.

The cell below runs PCA on a small 2-D correlated cloud, cross-checking the SVD
principal axes against the eigenvectors of the covariance and reporting the
explained-variance ratio. The figure draws the cloud with its two principal axes
scaled by $\sigma_i$.

```{.python .input #svd-pca}
import numpy as np

rng = np.random.default_rng(1)
# correlated 2-D cloud
latent = rng.standard_normal((300, 2)) * np.array([3.0, 0.8])
theta = np.pi / 5
rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
X = latent @ rot.T + np.array([2.0, -1.0])

Xc = X - X.mean(0)                              # center
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
V = Vt.T
cov = (Xc.T @ Xc) / len(Xc)
eigval, eigvec = np.linalg.eigh(cov)            # ascending
print('singular values of Xc:        ', s.round(4))
print('explained variance (sigma^2/n):', (s ** 2 / len(Xc)).round(4))
print('eigenvalues of covariance:     ', eigval[::-1].round(4))  # = sigma^2/n
print('explained-variance ratio:      ',
      (s ** 2 / (s ** 2).sum()).round(4))
```

```{.python .input #svd-fig-pca-axes}
import numpy as np

def plot_pca_axes():
    rng = np.random.default_rng(1)
    latent = rng.standard_normal((300, 2)) * np.array([3.0, 0.8])
    theta = np.pi / 5
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    X = latent @ rot.T + np.array([2.0, -1.0])
    mu = X.mean(0)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    d2l.set_figsize((4.2, 4.2))
    ax = d2l.plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=.35, color='C0')
    for i in range(2):
        scale = s[i] / np.sqrt(len(Xc))            # std dev along axis
        ax.annotate('', xy=(mu[0] + scale * V[0, i], mu[1] + scale * V[1, i]),
                    xytext=(mu[0], mu[1]),
                    arrowprops=dict(arrowstyle='->', color='C3', lw=2.5))
        ax.text(mu[0] + scale * V[0, i] * 1.1, mu[1] + scale * V[1, i] * 1.1,
                rf'$v_{i+1}$', color='C3', fontsize=11)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_title('principal axes scaled by std dev', fontsize=10)

plot_pca_axes()
```

The first principal axis aligns with the long direction of the cloud, exactly the
direction of maximal variance the proposition predicts.

## Pseudoinverse and Least Squares
:label:`subsec_mdl-pseudoinverse`

When $\mathbf{A}\mathbf{x}=\mathbf{b}$ has no solution (too many equations) or
infinitely many (too few), the SVD delivers the principled answer. Define the
*Moore--Penrose pseudoinverse* by inverting the nonzero singular values and
transposing the rotations,

$$
\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top,
\qquad
\boldsymbol{\Sigma}^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i>0,\\ 0 & \sigma_i=0.\end{cases}
$$
:eqlabel:`eq_mdl-pseudoinverse`

**Proposition (min-norm least squares).** *Among all minimizers of
$\|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2$, the one of smallest norm is
$\hat{\mathbf{x}}=\mathbf{A}^{+}\mathbf{b}$.*

**Proof.** Rotate both the input and the target into the singular bases:
$\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ and $\mathbf{c}=\mathbf{U}^\top\mathbf{b}$.
Since $\mathbf{U}$ is orthogonal it preserves norms, so

$$
\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2
   = \|\mathbf{U}^\top(\mathbf{A}\mathbf{x}-\mathbf{b})\|^2
   = \|\boldsymbol{\Sigma}\mathbf{y}-\mathbf{c}\|^2
   = \sum_{i\le r}(\sigma_i y_i - c_i)^2 + \sum_{i> r} c_i^2 .
$$

The rotation has *decoupled* the problem into independent scalar problems. The
first sum is minimized term by term at $y_i=c_i/\sigma_i$ for $i\le r$; the second
sum is a constant we cannot touch (it is the irreducible residual, the part of
$\mathbf{b}$ outside the column space). The coordinates $y_i$ for $i>r$ are *free*:
they do not appear in the residual at all. Setting them to zero gives the unique
solution of *smallest* norm (since $\|\mathbf{x}\|=\|\mathbf{y}\|$). That choice is
exactly $\mathbf{y}=\boldsymbol{\Sigma}^{+}\mathbf{c}$, i.e.
$\hat{\mathbf{x}}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top\mathbf{b}=\mathbf{A}^{+}\mathbf{b}$.
$\blacksquare$

The moral is worth stating: rotating into the SVD basis turns a coupled
least-squares problem into a list of one-dimensional problems, and "small residual"
and "small norm" land on *disjoint* coordinate blocks ($i\le r$ versus $i>r$), so
both can be satisfied at once. Two corollaries follow at once. When $\mathbf{A}$ is
square and invertible, all $\sigma_i>0$ and
$\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top=\mathbf{A}^{-1}$.
And *truncating* the pseudoinverse---dropping terms with tiny $\sigma_i$ instead of
dividing by them---caps the dangerous $1/\sigma_i$ blow-up; this is a form of
regularization, closely related to ridge regression, which we revisit when we
discuss weight decay.

The classical alternative, the *normal equations*
$\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{A}^\top\mathbf{b}$, is mathematically
equivalent for full-rank $\mathbf{A}$ but numerically worse: forming
$\mathbf{A}^\top\mathbf{A}$ *squares* the condition number
(:numref:`subsec_mdl-condition-number`), so an SVD- or QR-based solve is preferred.
The cell below solves an overdetermined system three ways---via `pinv`, via the
library `lstsq`, and via the normal equations---and prints the (matching)
residuals together with the condition numbers, showing the normal-equations matrix
is far worse conditioned.

```{.python .input #svd-least-squares}
#@tab mxnet
A = np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])  # 4 eqns, 2 unknowns
b = np.array([6., 5., 7., 10.])
x_pinv = np.linalg.pinv(A) @ b
x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
x_normal = np.linalg.solve(A.T @ A, A.T @ b)
print('pinv : ', x_pinv.round(4), ' residual',
      round(float(np.linalg.norm(A @ x_pinv - b)), 4))
print('lstsq: ', x_lstsq.round(4), ' residual',
      round(float(np.linalg.norm(A @ x_lstsq - b)), 4))
print('normal:', x_normal.round(4))
print('cond(A)      =', round(float(np.linalg.cond(A)), 3))
print('cond(A^T A)  =', round(float(np.linalg.cond(A.T @ A)), 3),
      '= cond(A)^2')
```

```{.python .input #svd-least-squares}
#@tab pytorch
A = torch.tensor([[1., 1.], [1., 2.], [1., 3.], [1., 4.]], dtype=torch.float64)
b = torch.tensor([6., 5., 7., 10.], dtype=torch.float64)
x_pinv = torch.linalg.pinv(A) @ b
x_lstsq = torch.linalg.lstsq(A, b).solution
x_normal = torch.linalg.solve(A.T @ A, A.T @ b)
print('pinv : ', x_pinv.numpy().round(4), ' residual',
      round(float(torch.linalg.norm(A @ x_pinv - b)), 4))
print('lstsq: ', x_lstsq.numpy().round(4), ' residual',
      round(float(torch.linalg.norm(A @ x_lstsq - b)), 4))
print('normal:', x_normal.numpy().round(4))
print('cond(A)      =', round(float(torch.linalg.cond(A)), 3))
print('cond(A^T A)  =', round(float(torch.linalg.cond(A.T @ A)), 3),
      '= cond(A)^2')
```

```{.python .input #svd-least-squares}
#@tab tensorflow
A = tf.constant([[1., 1.], [1., 2.], [1., 3.], [1., 4.]], dtype=tf.float64)
b = tf.constant([[6.], [5.], [7.], [10.]], dtype=tf.float64)
x_pinv = tf.linalg.pinv(A) @ b
x_lstsq = tf.linalg.lstsq(A, b)
x_normal = tf.linalg.solve(tf.transpose(A) @ A, tf.transpose(A) @ b)
print('pinv : ', x_pinv.numpy().ravel().round(4), ' residual',
      round(float(tf.norm(A @ x_pinv - b)), 4))
print('lstsq: ', x_lstsq.numpy().ravel().round(4), ' residual',
      round(float(tf.norm(A @ x_lstsq - b)), 4))
print('normal:', x_normal.numpy().ravel().round(4))
s = tf.linalg.svd(A, compute_uv=False)
sn = tf.linalg.svd(tf.transpose(A) @ A, compute_uv=False)
print('cond(A)      =', round(float(s[0] / s[-1]), 3))
print('cond(A^T A)  =', round(float(sn[0] / sn[-1]), 3), '= cond(A)^2')
```

```{.python .input #svd-least-squares}
#@tab jax
A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]], dtype=jnp.float64)
b = jnp.array([6., 5., 7., 10.], dtype=jnp.float64)
x_pinv = jnp.linalg.pinv(A) @ b
x_lstsq, *_ = jnp.linalg.lstsq(A, b, rcond=None)
x_normal = jnp.linalg.solve(A.T @ A, A.T @ b)
print('pinv : ', np.asarray(x_pinv).round(4), ' residual',
      round(float(jnp.linalg.norm(A @ x_pinv - b)), 4))
print('lstsq: ', np.asarray(x_lstsq).round(4), ' residual',
      round(float(jnp.linalg.norm(A @ x_lstsq - b)), 4))
print('normal:', np.asarray(x_normal).round(4))
print('cond(A)      =', round(float(jnp.linalg.cond(A)), 3))
print('cond(A^T A)  =', round(float(jnp.linalg.cond(A.T @ A)), 3),
      '= cond(A)^2')
```

## Condition Number and Numerical Stability
:label:`subsec_mdl-condition-number`

The *condition number* is the single SVD-derived scalar that predicts numerical
pain. For an invertible (or full-rank) matrix it is the ratio of the largest to
the smallest nonzero singular value,

$$
\kappa(\mathbf{A}) = \frac{\sigma_1}{\sigma_r} .
$$
:eqlabel:`eq_mdl-condition-number`

It measures how much a linear solve can amplify input error.

**Proposition (error amplification, with tightness).** *Let $\mathbf{A}$ be square
invertible, $\mathbf{A}\mathbf{x}=\mathbf{b}$, and let a perturbation
$\delta\mathbf{b}$ change the solution to $\mathbf{x}+\delta\mathbf{x}$. Then*

$$
\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}
   \le \kappa(\mathbf{A})\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|},
$$
:eqlabel:`eq_mdl-condition-bound`

*and the bound is attained for suitable $\mathbf{b},\delta\mathbf{b}$.*

**Proof.** From $\mathbf{A}\,\delta\mathbf{x}=\delta\mathbf{b}$ we get
$\delta\mathbf{x}=\mathbf{A}^{-1}\delta\mathbf{b}$, and the largest singular value
of $\mathbf{A}^{-1}$ is $1/\sigma_n$, so by :eqref:`eq_mdl-sigma1-variational`
$\|\delta\mathbf{x}\|\le\sigma_n^{-1}\|\delta\mathbf{b}\|$. From
$\mathbf{b}=\mathbf{A}\mathbf{x}$ we get $\|\mathbf{b}\|\le\sigma_1\|\mathbf{x}\|$.
Multiplying the two inequalities,

$$
\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}
   \le \frac{\sigma_1}{\sigma_n}\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}
   = \kappa(\mathbf{A})\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|} .
$$

It is *tight*: align the signal with the least-amplified direction by taking
$\mathbf{b}=\mathbf{u}_1$ (so $\mathbf{x}=\sigma_1^{-1}\mathbf{v}_1$ has the largest
possible norm for its right-hand side) and the error with the most-amplified
direction $\delta\mathbf{b}=\mathbf{u}_n$ (so $\delta\mathbf{x}=\sigma_n^{-1}\mathbf{v}_n$);
then both inequalities are equalities. $\blacksquare$

A few consequences are worth recording. An orthogonal matrix has all singular
values equal to $1$, so $\kappa=1$: rotations and reflections are perfectly
conditioned, which is *why* the SVD's $\mathbf{U},\mathbf{V}$ never amplify error
and why orthogonal initialization is attractive. At the other extreme, forming the
normal-equations matrix squares the conditioning, $\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2$
(its singular values are the $\sigma_i^2$), which is the quantitative reason to
prefer an SVD/QR least-squares solve over the normal equations
(:numref:`subsec_mdl-pseudoinverse`).

Geometrically, a large $\kappa$ means very *elongated* level sets. For the
quadratic bowl $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}$,
the contours are ellipses with axis ratio $\kappa$; when $\kappa\gg1$ the bowl is a
narrow valley, and gradient descent zig-zags across the steep walls while crawling
along the flat floor. This is the same picture that ended the Rayleigh discussion
in :numref:`subsec_mdl-rayleigh`, and it is no coincidence: the very same $\kappa$
controls gradient descent's convergence rate, which contracts like
$(\kappa-1)/(\kappa+1)$ per step---*one number, two consequences*, error
amplification in a solve and slow convergence in optimization. We make this
precise when we analyze gradient descent in :numref:`sec_mdl-gradient-based-optimization`
and study numerical conditioning in :numref:`sec_mdl-numerical-stability-conditioning`.
The figure below contrasts a well-conditioned bowl ($\kappa\approx1$, near-circular
contours, gradient descent heads almost straight to the minimum) with an
ill-conditioned one ($\kappa\gg1$, elongated contours, a zig-zag trajectory).

```{.python .input #svd-fig-condition-contours}
import numpy as np

def gd_path(A2, lr, steps=40):
    x = np.array([4.5, 4.5])
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * (A2 @ x)        # grad of 0.5 x^T A2 x
        path.append(x.copy())
    return np.array(path)

def plot_condition_contours():
    grid = np.linspace(-5, 5, 200)
    Xg, Yg = np.meshgrid(grid, grid)
    cases = [(np.diag([1.0, 1.2]), 'well-conditioned  $\\kappa\\approx1.2$'),
             (np.diag([1.0, 20.0]), 'ill-conditioned  $\\kappa=20$')]
    fig, ax = d2l.plt.subplots(1, 2, figsize=(8.2, 4.0))
    for j, (A2, title) in enumerate(cases):
        Z = 0.5 * (A2[0, 0] * Xg ** 2 + A2[1, 1] * Yg ** 2)
        ax[j].contour(Xg, Yg, Z, levels=20, cmap='Blues', alpha=.7)
        lr = 1.0 / A2[1, 1]          # stable step = 1/largest curvature
        path = gd_path(A2, lr)
        ax[j].plot(path[:, 0], path[:, 1], 'o-', color='C3', ms=3, lw=1,
                   label='gradient descent')
        ax[j].plot(0, 0, '*', color='k', ms=12)
        ax[j].set_title(title, fontsize=10); ax[j].set_aspect('equal')
        ax[j].set_xlim(-5, 5); ax[j].set_ylim(-5, 5)
    ax[0].legend(fontsize=8, loc='lower right')
    d2l.plt.tight_layout()

plot_condition_contours()
```

In the narrow valley on the right, the step size is throttled by the steep
direction (to stay stable) while the flat direction needs many such small steps to
make progress---the zig-zag is the visible cost of a large condition number.

## SVD in Modern Deep Learning
:label:`subsec_mdl-svd-modern-dl`

The SVD is not a historical artifact; low-rank structure is everywhere in
contemporary models, and Eckart--Young is the reason it can be exploited cheaply.

**Low-rank adapters (LoRA).** Fine-tuning a large pretrained model by updating
every weight is expensive. LoRA (Hu et al., 2021) freezes a pretrained weight
$\mathbf{W}\in\mathbb{R}^{m\times n}$ and learns only a low-rank correction
$\Delta\mathbf{W}=\mathbf{B}\mathbf{A}$ with $\mathbf{B}\in\mathbb{R}^{m\times r}$,
$\mathbf{A}\in\mathbb{R}^{r\times n}$, and $r\ll\min(m,n)$. This trades $mn$
trainable parameters for $r(m+n)$, a ratio of

$$
\frac{r(m+n)}{mn} ,
$$
:eqlabel:`eq_mdl-lora-ratio`

which for a $4096\times4096$ layer at rank $r=8$ is $8\cdot8192/4096^2\approx0.39\%$
of the parameters. The hypothesis that fine-tuning has *intrinsically low rank* is
what makes this work, and Eckart--Young is the formal backbone: among all rank-$r$
updates, the truncated SVD is the most expressive one, so a learned rank-$r$
adapter is operating at the optimal trade-off the theorem describes.

**Spectral normalization.** Constraining the largest singular value of each weight
matrix to $1$ caps the per-layer Lipschitz constant
(:eqref:`eq_mdl-sigma1-variational`: $\sigma_1=\|\mathbf{W}\|_2$ is exactly the most
a layer can stretch its input), which stabilizes training---originally for GAN
discriminators (Miyato et al., 2018). The beautiful part is *how* $\sigma_1$ is
estimated: by **power iteration on $\mathbf{W}^\top\mathbf{W}$**, the very algorithm
we analyzed in :numref:`sec_mdl-eigendecompositions`, since
$\sigma_1=\sqrt{\lambda_1(\mathbf{W}^\top\mathbf{W})}$. A couple of iterations per
training step suffice. The same circle closes that the chapter opened: power
iteration finds the dominant eigenvalue, and the dominant eigenvalue of the Gram
matrix *is* the squared top singular value.

**Weight and attention spectra.** Plotting the singular values of a trained layer
(the spectrum panel of the Eckart--Young figure, applied to $\mathbf{W}$ instead of
an image) reveals its *effective rank* via the energy ratio
:eqref:`eq_mdl-energy-ratio` and exposes heavy-tailed spectra that correlate with
generalization. Attention matrices are often empirically near-low-rank, which
motivates linear-attention approximations that replace the full softmax attention
with a low-rank surrogate. The cell below makes the diagnostic concrete: it builds
a synthetic weight matrix with a fast-decaying spectrum and reports the rank
needed for 95% spectral energy and the parameter saving a LoRA of that rank would
give.

```{.python .input #svd-weight-spectrum}
import numpy as np

rng = np.random.default_rng(2)
m, n = 256, 512
# a weight with a quickly decaying spectrum (effective low rank)
true_sigma = np.exp(-np.arange(min(m, n)) / 12.0)
Uw, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
Vw, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
W = (Uw * true_sigma) @ Vw.T
s = np.linalg.svd(W, compute_uv=False)
energy = np.cumsum(s ** 2) / (s ** 2).sum()
r95 = int(np.searchsorted(energy, 0.95) + 1)
print(f'matrix shape: {m} x {n}  ({m * n} params)')
print(f'rank for 95% spectral energy: {r95}')
print(f'LoRA params at that rank: {r95 * (m + n)} '
      f'({r95 * (m + n) / (m * n):.1%} of full)')
```

**Scaling up.** For matrices too large to factor fully, *randomized SVD*
(Halko, Martinsson, and Tropp, 2011) computes an accurate rank-$k$ truncation by
projecting onto a small random subspace, at a fraction of the cost---the standard
tool when only the leading singular triples are needed.

Let us verify the two facts the whole section rests on: that the SVD reconstructs
$\mathbf{A}$, and that $\sigma_i^2$ are the eigenvalues of $\mathbf{A}^\top\mathbf{A}$.

```{.python .input #svd-verify}
#@tab mxnet
A = np.array([[3., 1.], [1., 3.], [0., 2.]])   # a 3x2 matrix
U, s, Vt = np.linalg.svd(A, full_matrices=False)
recon = (U * s) @ Vt
eig = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
print('reconstruction error:', round(float(np.linalg.norm(recon - A)), 12))
print('sigma^2          :', (s ** 2).round(6))
print('eig(A^T A) sorted:', eig.round(6))
```

```{.python .input #svd-verify}
#@tab pytorch
A = torch.tensor([[3., 1.], [1., 3.], [0., 2.]], dtype=torch.float64)
U, s, Vt = torch.linalg.svd(A, full_matrices=False)
recon = (U * s) @ Vt
eig = torch.linalg.eigvalsh(A.T @ A).flip(0)
print('reconstruction error:', round(float(torch.linalg.norm(recon - A)), 12))
print('sigma^2          :', (s ** 2).numpy().round(6))
print('eig(A^T A) sorted:', eig.numpy().round(6))
```

```{.python .input #svd-verify}
#@tab tensorflow
A = tf.constant([[3., 1.], [1., 3.], [0., 2.]], dtype=tf.float64)
s, U, V = tf.linalg.svd(A)
recon = U @ tf.linalg.diag(s) @ tf.transpose(V)
eig = tf.sort(tf.linalg.eigvalsh(tf.transpose(A) @ A), direction='DESCENDING')
print('reconstruction error:', round(float(tf.norm(recon - A)), 12))
print('sigma^2          :', (s.numpy() ** 2).round(6))
print('eig(A^T A) sorted:', eig.numpy().round(6))
```

```{.python .input #svd-verify}
#@tab jax
A = jnp.array([[3., 1.], [1., 3.], [0., 2.]], dtype=jnp.float64)
U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
recon = (U * s) @ Vt
eig = jnp.sort(jnp.linalg.eigvalsh(A.T @ A))[::-1]
print('reconstruction error:', round(float(jnp.linalg.norm(recon - A)), 12))
print('sigma^2          :', np.asarray(s ** 2).round(6))
print('eig(A^T A) sorted:', np.asarray(eig).round(6))
```

The reconstruction error is at the level of floating-point round-off, and the
squared singular values match the eigenvalues of $\mathbf{A}^\top\mathbf{A}$
exactly---the construction of :numref:`subsec_mdl-svd-via-ata` made flesh.

## Summary

* Every matrix factors as $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$
  (rotate--scale--rotate; equivalently the polar form $\mathbf{A}=\mathbf{Q}\mathbf{P}$),
  with $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ and the dyadic sum
  $\mathbf{A}=\sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$.
* The SVD is the spectral theorem applied to the symmetric PSD matrix
  $\mathbf{A}^\top\mathbf{A}$, with $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$;
  it therefore *never fails*, even for the defective shear that had no
  eigenbasis. Equivalently $\sigma_1=\max_{\|\mathbf{x}\|=1}\|\mathbf{A}\mathbf{x}\|=\|\mathbf{A}\|_2$.
* Rank, range, and the four fundamental subspaces read off the spectrum;
  numerical rank thresholds the $\sigma_i$ at $\sim\epsilon_{\text{mach}}\sigma_1$.
* **Eckart--Young--Mirsky:** the top-$k$ truncation $\mathbf{A}_k$ is the optimal
  rank-$k$ approximation, with $\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$ and
  $\|\mathbf{A}-\mathbf{A}_k\|_F^2=\sum_{i>k}\sigma_i^2$. The energy ratio is the
  "how much did we keep" dial.
* **PCA is Eckart--Young on centered data:** principal directions are the right
  singular vectors $\mathbf{v}_i$, explained variance is $\sigma_i^2/n$.
* The pseudoinverse $\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$
  gives the minimum-norm least-squares solution; the condition number
  $\kappa=\sigma_1/\sigma_r$ predicts both error amplification and
  gradient-descent speed, and $\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2$
  is why the normal equations are worse.
* SVD powers PCA, LoRA (rank-$r$ updates at $r(m+n)$ parameters), spectral
  normalization (power iteration on $\mathbf{W}^\top\mathbf{W}$), and weight /
  attention spectral analysis.

## Exercises

1. Compute the SVD of $\operatorname{diag}(3,1)$ by inspection. Then show the
   rotation $\begin{bmatrix}0&-2\\1&0\end{bmatrix}$ has singular values $\{2,1\}$
   even though its eigenvalue magnitudes are both $\sqrt2$---i.e.
   $\sigma\neq|\lambda|$ for non-symmetric matrices.
2. Prove the Eckart--Young spectral-norm error
   $\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$ directly from
   :eqref:`eq_mdl-sigma1-variational`, by identifying the SVD of
   $\mathbf{A}-\mathbf{A}_k$.
3. Prove that $\hat{\mathbf{x}}=\mathbf{A}^{+}\mathbf{b}$ is the minimum-norm
   least-squares solution of $\mathbf{A}\mathbf{x}=\mathbf{b}$, and deduce
   $\mathbf{A}^{+}=\mathbf{A}^{-1}$ when $\mathbf{A}$ is square invertible.
4. Show that the singular values of any orthogonal matrix are all $1$, hence
   $\kappa=1$. Then prove $\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2$
   and explain why this makes the normal equations numerically inferior.
5. Show $\mathbf{A}$ and $\mathbf{A}^\top$ have the same nonzero singular values.
   (*Hint:* relate $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$ via
   :eqref:`eq_mdl-svd-gram`.)
6. Prove the polar decomposition $\mathbf{A}=\mathbf{Q}\mathbf{P}$ has $\mathbf{Q}$
   orthogonal and $\mathbf{P}\succeq0$, and that $\mathbf{P}$ is unique while
   $\mathbf{Q}$ is unique when $\mathbf{A}$ is invertible.
7. For a given weight matrix, find the LoRA rank achieving 95% spectral energy and
   compute the resulting parameter saving relative to a full update (the
   `#svd-weight-spectrum` cell). How does the answer change if the spectrum decays
   more slowly?
8. Show that adding a rank-$r$ update $\mathbf{B}\mathbf{A}$ to $\mathbf{W}$ can
   change at most $r$ of $\mathbf{W}$'s singular values away from their original
   values. (*Hint:* the perturbation has rank $\le r$; use a dimension-counting
   argument like the one in :numref:`subsec_mdl-eckart-young`.)

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/svd)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/svd)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/svd)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/svd)
:end_tab:

<!-- slides -->

::: {.slide title="The SVD: rotate--scale--rotate"}
Every $m\times n$ matrix factors as

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,$$

with $\mathbf{U},\mathbf{V}$ orthogonal and
$\sigma_1\ge\sigma_2\ge\cdots\ge0$. Read right to left:
$\mathbf{V}^\top$ rotates, $\boldsymbol{\Sigma}$ stretches by
$\sigma_i$, $\mathbf{U}$ rotates again. Two frames, one stretch
($\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$) — the eigen-picture
made universal:

@svd-fig-rotate-scale-rotate
:::

::: {.slide title="Where singular values come from"}
The SVD *is* the spectral theorem applied to the symmetric PSD
matrix $\mathbf{A}^\top\mathbf{A}$:

$$\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})},
\qquad \mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i.$$

Gram matrices are never defective, so the SVD **never fails** —
even the defective shear $[[1,1],[0,1]]$ gets a clean SVD
($\sigma = \varphi, 1/\varphi$):

@svd-verify
:::

::: {.slide title="Eckart--Young: optimal low rank"}
Truncating to the top $k$ terms,
$\mathbf{A}_k=\sum_{i\le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$,
is the **best** rank-$k$ approximation:
$\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$. Fast-decaying spectra
compress almost for free:

@svd-fig-eckart-young
:::

::: {.slide title="PCA = Eckart--Young on centered data"}
Principal directions are the right singular vectors $\mathbf{v}_i$
of the centered data; variance explained by component $i$ is
$\sigma_i^2/n$:

@svd-fig-pca-axes

. . .

@svd-pca
:::

::: {.slide title="Least squares & conditioning"}
Pseudoinverse $\mathbf{A}^{+}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$
gives min-norm least squares; condition number
$\kappa=\sigma_1/\sigma_r$ amplifies error and slows gradient
descent. Normal equations square it ($\kappa(\mathbf{A}^\top\mathbf{A})=\kappa^2$):

@svd-fig-condition-contours

. . .

@svd-least-squares
:::

::: {.slide title="SVD in modern deep learning"}
- **LoRA**: $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}$, rank $r$, only
  $r(m+n)$ params; Eckart--Young says rank-$r$ is the best cheap update.
- **Spectral norm**: cap $\sigma_1=\|\mathbf{W}\|_2$ for Lipschitz
  control — estimated by power iteration on $\mathbf{W}^\top\mathbf{W}$.
- **Weight/attention spectra**: effective rank from the energy ratio.

@svd-weight-spectrum
:::

::: {.slide title="Recap"}
- Every matrix is rotate--scale--rotate;
  $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$ — the SVD never fails.
- Top-$k$ truncation is the *optimal* low-rank approximation (Eckart--Young).
- PCA, pseudoinverse, and $\kappa$ all fall out of one factorization.
- $\kappa=\sigma_1/\sigma_r$ is the one number to watch.
:::
