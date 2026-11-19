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

## Resources and Further Reading {.unnumbered}

If you would like to revisit this chapter's material in greater depth or from a
different perspective, the books, courses, and notes below pair well with
individual sections, and most are freely available online. For topics that go
beyond the scope of this chapter, see
:numref:`sec_mdl-linear-algebra-further-reading`.

**Books**

- [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal, and Ong: a free textbook whose chapters 2 (Linear Algebra), 3 (Analytic Geometry), and 4 (Matrix Decompositions) cover the entire chapter from a machine-learning perspective, from the projections of :numref:`sec_mdl-geometry-linear-algebraic-ops` to the SVD of :numref:`sec_mdl-svd-low-rank`.
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/) by Boyd and Vandenberghe: a free book with slides and code, organized around least squares and data, pairing well with the fitting problems of :numref:`sec_mdl-geometry-linear-algebraic-ops` and the pseudoinverse of :numref:`sec_mdl-svd-low-rank`.
- [Introduction to Linear Algebra, 6th ed.](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) by Gilbert Strang: the classic first course (a commercial book with sample chapters and problems online), built around the column picture and the four fundamental subspaces of :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-svd-low-rank`.
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) by Gilbert Strang: the SVD-centered companion to MIT 18.065, closely aligned with the low-rank approximation, principal components, and least squares of :numref:`sec_mdl-svd-low-rank`.
- [Linear Algebra Done Right, 4th ed.](https://linear.axler.net/) by Sheldon Axler: a free, rigorous, determinant-free route to eigenvalues and the spectral theorem, ideal for readers who want every proof behind :numref:`sec_mdl-eigendecompositions`.
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) by Trefethen and Bau: an engaging treatment of conditioning, stability, QR, and the floating-point algorithms behind `solve`, `eig`, and `svd`, covering the numerical side of :numref:`sec_mdl-linear-maps` and :numref:`subsec_mdl-condition-number`.

**Courses and video lectures**

- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) by Gilbert Strang: a complete lecture course with notes, exams, and solutions, covering :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-eigendecompositions`.
- [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) by Gilbert Strang: the follow-on course connecting the SVD and low-rank approximation of :numref:`sec_mdl-svd-low-rank` to deep learning and optimization.
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3Blue1Brown: short animated lectures on vectors, linear maps, determinants, and eigenvectors, bringing to life the geometry of :numref:`sec_mdl-geometry-linear-algebraic-ops` through :numref:`sec_mdl-eigendecompositions`.
- [Singular Value Decomposition](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) by Steve Brunton: clear video lectures on the geometry of the SVD, low-rank approximation, principal components, and randomized SVD, accompanying :numref:`sec_mdl-svd-low-rank`.

**Tutorials and notes**

- [Linear Algebra Review and Reference](https://cs229.stanford.edu/section/cs229-linalg.pdf) by Kolter, updated by Do, for Stanford CS229: a compact refresher on the notation of :numref:`sec_mdl-geometry-linear-algebraic-ops` and :numref:`sec_mdl-linear-maps` as used in machine-learning courses.
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) by Ström, Åström, and Akenine-Möller: a free online book with interactive figures of linear maps, determinants, and eigenvectors, reinforcing :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-eigendecompositions`.
