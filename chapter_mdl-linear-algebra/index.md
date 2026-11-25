# Linear Algebra
:label:`chap_mdl-linear-algebra`

Every neural network in this book is built out of matrix products: a linear
layer multiplies its input by a weight matrix, attention multiplies queries by
keys, and a recurrent network multiplies its hidden state by the same matrix at
every step. In :numref:`sec_linear-algebra`, we computed these products without
asking what a matrix actually *does* to the vectors it acts on. This optional
chapter takes up that geometric question, proofs included; nothing in the main
text depends on it, yet seeing the geometry behind matrix products changes how
you read almost every model in the book.

Consider a 784-pixel Fashion-MNIST image mapped to 256 features. How similar
are two images, how much information does the layer discard, and what happens
if we apply the same linear map fifty times in a row? Because we cannot draw in
784 dimensions, our answers come from algebra that carries plane geometry into
any dimension.

A complicated matrix becomes transparent once viewed along the right
directions. Around 1810, Gauss fitted the orbit of the asteroid Pallas to more
observations than unknowns via least squares, a problem our first section
solves and our last revisits. In orthonormal axes, lengths and angles reduce to
dot products; along eigenvectors, a square matrix scales each direction by a
single scalar; and in a pair of singular bases, *any* matrix decomposes into
independent stretches. These ideas explain why attention divides scores by
$\sqrt{d}$ (:numref:`sec_attention-scoring-functions`), when recurrent
gradients explode (:numref:`sec_bptt`), and what a low-rank adapter can
represent.

The chapter moves in two stages. First, we apply a matrix once:
:numref:`sec_mdl-geometry-linear-algebraic-ops` extends lengths, angles, and
orthogonal projections (giving the closest point in a subspace), while
:numref:`sec_mdl-linear-maps` tracks which directions collapse to zero and when
a map is invertible. Next, we choose coordinates tailored to the matrix:
eigenvectors and the spectral theorem for symmetric matrices
(:numref:`sec_mdl-eigendecompositions`), non-normal behavior that eigenvalues
alone miss (:numref:`sec_mdl-spectra-in-practice`), and singular vectors with
optimal low-rank truncation for any shape (:numref:`sec_mdl-svd-low-rank`). Two
small matrices keep us honest throughout:
$\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$, with
complex eigenvalues, and the shear
$\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$, whose
eigenvectors fail to span the plane.

On a first reading, feel free to skim $n$-dimensional determinants and the
discussions of Gershgorin discs, PageRank, and pseudospectra in
:numref:`sec_mdl-spectra-in-practice`. Numerical eigensolvers and linear
solvers are surveyed in :numref:`sec_mdl-linear-algebra-further-reading`.

```toc
:maxdepth: 2

mdl-geometry-linear-algebraic-ops
mdl-linear-maps
mdl-eigendecomposition
mdl-spectra-in-practice
mdl-svd-low-rank
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §33]{.kicker}

Linear Algebra: Geometry, Spectra, and Low-Rank Structure<br>**projections · linear maps · eigendecompositions · spectral stability · singular value decomposition**
:::
:::

::: {.slide title="Why Deep Learning Needs Geometric Linear Algebra"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Beyond Mechanical Multiplication:** Viewing $\mathbf{X}\mathbf{W}$ geometrically reveals what a layer preserves, collapses, or amplifies in $\mathbb{R}^d$.
- **High-Dimensional Intuition:** Dot products carry lengths, angles, and orthogonal shadows into 784-pixel images or 4096-D embeddings.
- **Choosing Natural Axes:** A matrix becomes transparent along orthonormal subspace bases, eigenvectors, or singular vectors.
:::

::: {.col .fig .big}
@fig:ch33-la-motivation-axes
:::
:::
:::

::: {.slide title="Two-Stage Roadmap: Single Maps to Natural Axes"}
[Chapter Structure]{.kicker}

::: {.cols .vc}
::: {.col}
- **Stage 1: Standard Coordinates (§§{sec_mdl-geometry-linear-algebraic-ops}–§{num:sec_mdl-linear-maps}):** Measure similarity, project onto subspaces ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$, $\mathbf{A} = \mathbf{Q}\mathbf{R}$), and track surviving vs. collapsed dimensions ($\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$, $\det\mathbf{A}$).
- **Stage 2: Matrix-Tailored Coordinates (§§{sec_mdl-eigendecompositions}–§{num:sec_mdl-svd-low-rank}):** Diagonalize repeated square maps ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$), analyze power iteration and non-normal growth, and factor any $m \times n$ matrix via SVD ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$).
:::

::: {.col .fig .big}
@fig:ch33-la-roadmap-two-stage
:::
:::
:::

::: {.slide title="Three Landmark Theorems of Chapter §{num:chap_mdl-linear-algebra}"}
[Core Theory]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Orthogonal Projection ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$):** The closest point in $S$ leaves a residual $\mathbf{y} - \mathbf{P}\mathbf{y} \perp S$, turning least squares into a right triangle.
- **2. Spectral Theorem ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$):** Every real symmetric matrix has real eigenvalues and an orthonormal eigenbasis.
- **3. Eckart–Young–Mirsky ($\mathbf{A}_k$):** Truncating the SVD at rank $k$ gives the optimal rank-$k$ approximation in spectral and Frobenius norms.
:::

::: {.col .fig .big}
@fig:ch33-la-three-theorems
:::
:::
:::

::: {.slide title="Running Matrices and Empirical Benchmarks"}
[Pedagogical Thread]{.kicker}

::: {.cols .vc}
::: {.col}
- **Concrete $2 \times 2$ Test Matrices:** Running matrix $\mathbf{A}$ (complex $\lambda = 2 \pm i$, $\det = 5$), defective shear $\mathbf{T}$ ($g = 1 < a = 2$, linear drift), and rank-1 singular matrix $\mathbf{B}$.
- **Real High-Dimensional Data:** Fashion-MNIST in $\mathbb{R}^{784}$ ($92\%$ nearest-mean hyperplane; non-isotropic cosines) and a $400 \times 500$ cat photo ($9\%$ rank-20 SVD storage at $8\%$ error).
:::

::: {.col .fig .big}
@fig:ch33-la-running-benchmarks
:::
:::
:::

::: {.slide title="How to Navigate This Chapter"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **First-Pass Essentials:** Focus on orthogonal projections and $1/\sqrt{d}$ attention scaling (§{sec_mdl-geometry-linear-algebraic-ops}), column/null spaces and rank–nullity (§{sec_mdl-linear-maps}), the spectral theorem and positive definiteness (§{sec_mdl-eigendecompositions}), power iteration and transient growth (§{sec_mdl-spectra-in-practice}), and the full SVD / LoRA / conditioning workflow (§{sec_mdl-svd-low-rank}).
- **What to Skim on First Reading:** The axiomatic $n$-dimensional permutation determinant (§{sec_mdl-linear-maps}), Gershgorin discs, PageRank, and $\varepsilon$-pseudospectra (§{sec_mdl-spectra-in-practice}) can be skipped without losing the main thread.
:::

::: {.col}
- **Real vs. Complex Spaces:** We work in real vector spaces $\mathbb{R}^d$ throughout, bringing in complex numbers $\mathbb{C}$ only when real matrices like rotations or $\mathbf{A}$ force complex eigenvalues.
- **Numerical Mindset:** Every section pairs exact theorems with floating-point realities—why we clip cosines to $[-1, 1]$, use QR/SVD instead of normal equations, call `solve` instead of `inv`, and threshold tiny singular values.
:::
:::

::: {.d2l-note}
**Up next (§{sec_mdl-geometry-linear-algebraic-ops} Vectors, Angles, and Projections):** How the dot product extends planar lengths, angles, orthogonal shadows, and separating hyperplanes to 784 dimensions.
:::
:::

