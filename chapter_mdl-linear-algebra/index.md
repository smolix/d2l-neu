# Linear Algebra
:label:`chap_mdl-linear-algebra`

Every neural network in this book is built out of matrix products. A linear
layer multiplies its input by a weight matrix, attention multiplies queries by
keys, and a recurrent network multiplies its hidden state by the same matrix at
every step. In :numref:`sec_linear-algebra`, we learned how to compute these
products without pausing to ask what a matrix actually *does* to the vectors it
acts on. This chapter takes up that question, proofs included. It is optional
reading, so nothing in the main text depends on it, yet understanding the
geometry behind these products changes how you read almost every model in the
book.

Consider a Fashion-MNIST image, represented as a list of 784 numbers, passing
through a layer that maps it to 256 features. How similar are two images? How
much information does the layer discard? And what happens to a signal if we
feed it through the same linear map fifty times in a row? We cannot draw a
picture in 784 dimensions, so our answers have to come from algebra that
carries the familiar geometry of the plane into any dimension.

The central idea of linear algebra is remarkably simple: a complicated matrix
becomes transparent once you view it along the right directions. Around 1810,
Gauss fitted the orbit of the asteroid Pallas to more observations than
unknowns using least squares, a problem our first section solves and our last
revisits. In orthonormal axes, lengths and angles reduce to dot products. Along
its eigenvectors, a matrix acts simply by scaling each direction by a single
number. With a pair of singular bases, *any* matrix breaks down into a list of
independent stretches. These same geometric ideas explain why attention divides
its scores by $\sqrt{d}$ (:numref:`sec_attention-scoring-functions`), when the
gradients of a recurrent network explode (:numref:`sec_bptt`), and what a
low-rank fine-tuning adapter can and cannot represent.

The chapter moves in two stages. First, we study what happens when a matrix is
applied once: :numref:`sec_mdl-geometry-linear-algebraic-ops` extends lengths,
angles, and projections to arbitrary dimensions, and
:numref:`sec_mdl-linear-maps` asks which inputs a matrix collapses to zero and
when its action can be undone. Next, we choose coordinates tailored to the
matrix itself: eigenvectors for a square matrix applied repeatedly
(:numref:`sec_mdl-eigendecompositions`), the subtler behavior that eigenvalues
alone miss (:numref:`sec_mdl-spectra-in-practice`), and singular vectors for
matrices of any shape (:numref:`sec_mdl-svd-low-rank`). If you remember only
three results from this chapter, let them be these: projection onto a subspace
yields the closest point, every symmetric matrix has an orthonormal basis of
eigenvectors (the spectral theorem), and a truncated singular value
decomposition gives the best approximation of its rank. Along the way, two
small examples will keep us honest: the running matrix
$\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$,
whose eigenvalues are complex, and the shear
$\mathbf{T}=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$, which
does not have enough eigenvectors to span the plane.

On a first reading, feel free to skim the $n$-dimensional determinant and, in
:numref:`sec_mdl-spectra-in-practice`, the discussions of Gershgorin discs,
PageRank, and pseudospectra. Throughout, we work in real finite-dimensional
spaces, bringing in complex numbers only when eigenvalues force our hand. How
numerical libraries actually compute eigenvalues and solve large systems is
left to :numref:`sec_mdl-linear-algebra-further-reading`.

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
- **Stage 1 — Standard Coordinates (§§{sec_mdl-geometry-linear-algebraic-ops}–§{num:sec_mdl-linear-maps}):** Measure similarity, project onto subspaces ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$, $\mathbf{A} = \mathbf{Q}\mathbf{R}$), and track surviving vs. collapsed dimensions ($\operatorname{rank}\mathbf{A} + \dim\ker\mathbf{A} = n$, $\det\mathbf{A}$).
- **Stage 2 — Matrix-Tailored Coordinates (§§{sec_mdl-eigendecompositions}–§{num:sec_mdl-svd-low-rank}):** Diagonalize repeated square maps ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$), analyze power iteration and non-normal growth, and factor any $m \times n$ matrix via SVD ($\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$).
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

