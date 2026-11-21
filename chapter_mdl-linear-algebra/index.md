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

Every neural network layer—from linear projections and attention heads to recurrent state updates—is built from matrix products whose behavior in $\mathbb{R}^d$ is governed by high-dimensional geometry:

::: {.cols}
::: {.col}
- **Beyond Mechanical Multiplication:** Chapter 1 showed how to compute $\mathbf{X}\mathbf{W}$, but understanding *what* a layer preserves, collapses, or amplifies requires viewing matrices as geometric transformations.
- **High-Dimensional Intuition:** We cannot draw a 784-pixel Fashion-MNIST image or a 4096-dimensional embedding, yet dot products carry lengths, angles, and orthogonal shadows into any dimension.
:::

::: {.col}
- **Choosing the Right Coordinates:** A complicated matrix becomes transparent once viewed along its natural axes—orthonormal bases for subspaces, eigenvectors for repeated square maps, and singular vectors for arbitrary rectangular layers.
- **Three Core Questions:** How similar are two representations? Which input directions does a layer destroy? And what happens when we apply a matrix fifty times in a row?
:::
:::

::: {.d2l-note}
Three landmark results anchor the chapter: orthogonal projection yields the closest point, every symmetric matrix has an orthonormal eigenbasis, and truncated SVD gives the optimal low-rank approximation.
:::
:::

::: {.slide title="Two-Stage Roadmap: Single Maps to Natural Axes"}
[Chapter Structure]{.kicker}

The chapter progresses from applying a matrix once in standard coordinates to choosing basis coordinates tailored to the matrix itself:

| Section | Core Mathematical Question | Key Factorization / Theorem | Deep Learning Connection |
| :--- | :--- | :--- | :--- |
| **§33.1 Vectors, Angles & Projections** | How do we measure similarity, distance, and best fits in $\mathbb{R}^d$? | Cauchy–Schwarz, $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$, QR ($\mathbf{A} = \mathbf{Q}\mathbf{R}$) | Attention $1/\sqrt{d}$ scaling, cosine retrieval, linear classifiers |
| **§33.2 Matrices as Linear Maps** | Which inputs collapse to zero, when is a map invertible, and how does volume scale? | Rank–nullity theorem, $\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$ | Bottleneck layers, `einsum` contractions, normalizing flows |
| **§33.3 Eigendecompositions** | What happens when a square matrix is applied repeatedly, $\mathbf{A}^k\mathbf{x}$? | $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$, Spectral Theorem ($\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$) | Markov chains, Hessian curvature, quadratic optimization bowls |
| **§33.4 Spectra in Practice** | How do we find eigenvalues at scale, and when do eigenvalues mislead us? | Gershgorin discs, Power Iteration, Non-normal transient growth | PageRank, exploding/vanishing RNN gradients, weight initialization |
| **§33.5 SVD & Low-Rank Approximation** | How does any $m \times n$ matrix stretch space, and what is its best rank-$k$ approximation? | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, Eckart–Young, Pseudoinverse $\mathbf{A}^+$ | PCA, LoRA fine-tuning, Muon Newton–Schulz, conditioning $\kappa(\mathbf{A})$ |
:::

::: {.slide title="Three Landmark Theorems of Chapter 33"}
[Core Theory]{.kicker}

Almost every derivation in deep learning linear algebra reduces to one of three geometric principles:

- **1. Orthogonal Projection ($\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$):** The closest point in a subspace $S$ to $\mathbf{y}$ is the unique point whose residual $\mathbf{y} - \mathbf{P}\mathbf{y}$ is orthogonal to $S$, turning least squares $\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|^2$ into a right-triangle decomposition.
- **2. The Spectral Theorem ($\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$):** Every real symmetric matrix possesses real eigenvalues and an orthonormal basis of eigenvectors, turning quadratic forms $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i \lambda_i (\mathbf{v}_i^\top \mathbf{x})^2$ into weighted sums of squares.
- **3. Eckart–Young–Mirsky Theorem ($\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$):** Every $m \times n$ matrix maps one orthonormal basis to another with non-negative stretches $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$, and keeping the top $k$ terms yields the best rank-$k$ approximation in both spectral and Frobenius norms.
:::

::: {.slide title="Running Matrices and Empirical Benchmarks"}
[Pedagogical Thread]{.kicker}

Throughout the chapter, concrete $2 \times 2$ test matrices and real image datasets keep every theorem honest and expose edge cases:

| Benchmark / Matrix | Definition | What It Illustrates Across Chapter 33 |
| :--- | :--- | :--- |
| **Running Matrix $\mathbf{A}$** | $\begin{bmatrix}1 & 2 \\ -1 & 3\end{bmatrix}$ | Invertible map ($\det \mathbf{A} = 5$), complex eigenvalues $\lambda = 2 \pm i$ (rotation-scaling), singular values $\sigma = 3.618, 1.382$ ($\sigma_1\sigma_2 = 5$) |
| **Defective Shear $\mathbf{T}$** | $\begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix}$ | Repeated eigenvalue $\lambda = 1$ with only one eigenvector direction ($g = 1 < a = 2$), linear growth $\mathbf{T}^k$, clean SVD with golden ratio $\sigma = \varphi, 1/\varphi$ |
| **Singular Matrix $\mathbf{B}$** | $\begin{bmatrix}2 & 1 \\ 4 & 2\end{bmatrix}$ | Rank-1 collapse onto $y = 2x$, 1D null space along $(1, -2)$, $\det \mathbf{B} = 0$, four fundamental subspaces in SVD |
| **Fashion-MNIST ($d = 784$)** | $70{,}000$ images in $\mathbb{R}^{784}$ | Non-isotropic cosine distribution vs. $1/\sqrt{d}$ random baseline, $92\%$ nearest-mean hyperplane classifier on t-shirts vs. trousers |
| **Cat Photograph ($400 \times 500$)** | $200{,}000$ grayscale pixels | Power-law singular value decay ($\sigma_i \propto i^{-1.1}$), rank-20 truncated SVD retaining $9\%$ storage at $8\%$ relative error |
:::

::: {.slide title="How to Navigate This Chapter"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **First-Pass Essentials:** Focus on orthogonal projections and $1/\sqrt{d}$ attention scaling (§33.1), column/null spaces and rank–nullity (§33.2), the spectral theorem and positive definiteness (§33.3), power iteration and transient growth (§33.4), and the full SVD / LoRA / conditioning workflow (§33.5).
- **What to Skim on First Reading:** The axiomatic $n$-dimensional permutation determinant (§33.2), Gershgorin discs, PageRank, and $\varepsilon$-pseudospectra (§33.4) can be skipped without losing the main thread.
:::

::: {.col}
- **Real vs. Complex Spaces:** We work in real vector spaces $\mathbb{R}^d$ throughout, bringing in complex numbers $\mathbb{C}$ only when real matrices like rotations or $\mathbf{A}$ force complex eigenvalues.
- **Numerical Mindset:** Every section pairs exact theorems with floating-point realities—why we clip cosines to $[-1, 1]$, use QR/SVD instead of normal equations, call `solve` instead of `inv`, and threshold tiny singular values.
:::
:::

::: {.d2l-note}
**Up next (§33.1 Vectors, Angles, and Projections):** How the dot product extends planar lengths, angles, orthogonal shadows, and separating hyperplanes to 784 dimensions.
:::
:::

