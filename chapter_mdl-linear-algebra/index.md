# Linear Algebra
:label:`chap_mdl-linear-algebra`

Every network in this book is built from matrix products. A linear layer
multiplies its input by a weight matrix, attention multiplies queries by
keys, and a recurrent network multiplies its state by the same matrix at
every step. :numref:`sec_linear-algebra` taught how to compute these
products and rarely asked what a matrix does to the vectors it acts on.
This chapter does, with proofs. It is additional reading, and the main text
does not depend on it.

Take a Fashion-MNIST image, a list of 784 numbers, and a layer that maps it
to 256 features. How similar are two images? How much information does the
layer lose? What happens to a signal that passes through the same map fifty
times? We cannot draw 784 dimensions, so the answers must come from algebra
that keeps the geometry of the plane.

Around 1810 Gauss fitted the orbit of the asteroid Pallas to more
observations than unknowns by least squares, a problem the first section
solves and the last takes up again. The central idea of linear algebra is
that a matrix becomes simple in the right directions. In orthonormal axes,
lengths and angles are dot products. Along its eigenvectors, a matrix acts
as multiplication by a number. With a pair of singular bases, every matrix
becomes a list of independent stretches. These ideas explain why attention divides its scores by $\sqrt d$
(:numref:`sec_attention-scoring-functions`), when the gradients of a
recurrent network explode (:numref:`sec_bptt`), and what a low-rank
fine-tuning adapter can represent.

The first two sections apply a matrix once.
:numref:`sec_mdl-geometry-linear-algebraic-ops` carries lengths, angles, and
projections into any dimension, and :numref:`sec_mdl-linear-maps` finds
which inputs a matrix sends to zero and when it can be undone. The rest of
the chapter chooses coordinates to suit the matrix: eigenvectors for a
square matrix applied repeatedly (:numref:`sec_mdl-eigendecompositions`),
then what eigenvalues leave unsaid (:numref:`sec_mdl-spectra-in-practice`),
and singular vectors for every matrix
(:numref:`sec_mdl-svd-low-rank`). Three results matter most: projection
onto a subspace gives the closest point, a symmetric matrix has an
orthonormal basis of eigenvectors (the spectral theorem), and a truncated
singular value decomposition is the best approximation of its rank. Two
matrices recur: the running matrix
$\mathbf A=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$, with
complex eigenvalues, and the shear
$\mathbf T=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$, with
too few eigenvectors.

A first reading can skim the determinant in $n$ dimensions and, in
:numref:`sec_mdl-spectra-in-practice`, Gershgorin discs, PageRank, and
pseudospectra. We work in real
finite-dimensional spaces and use complex numbers only where eigenvalues
require them. How libraries compute eigenvalues and solve large systems is
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

The books, courses, and notes below teach this chapter's material again, at
greater length or from another side; each names its section, and most are
free online. What lies beyond the chapter is the subject of
:numref:`sec_mdl-linear-algebra-further-reading`.

**Books**

- [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal, and Ong: a free book whose chapters 2 (Linear Algebra), 3 (Analytic Geometry), and 4 (Matrix Decompositions) cover the whole chapter from a machine-learning perspective, from the projections of :numref:`sec_mdl-geometry-linear-algebraic-ops` to the SVD of :numref:`sec_mdl-svd-low-rank`.
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/) by Boyd and Vandenberghe: a free book with slides and code, organized around least squares and data, for the fitting problems of :numref:`sec_mdl-geometry-linear-algebraic-ops` and the pseudoinverse of :numref:`sec_mdl-svd-low-rank`.
- [Introduction to Linear Algebra, 6th ed.](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) by Gilbert Strang: the standard first course (a commercial book with sample chapters and problems online), built on the column picture and the four fundamental subspaces of :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-svd-low-rank`.
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) by Gilbert Strang: the SVD-centered companion to MIT 18.065, close to the low-rank approximation, principal components, and least squares of :numref:`sec_mdl-svd-low-rank`.
- [Linear Algebra Done Right, 4th ed.](https://linear.axler.net/) by Sheldon Axler: a free, rigorous, determinant-free route to eigenvalues and the spectral theorem, for readers who want every proof behind :numref:`sec_mdl-eigendecompositions`.
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) by Trefethen and Bau: conditioning, stability, QR, and the floating-point algorithms behind `solve`, `eig`, and `svd`, the numerical side of :numref:`sec_mdl-linear-maps` and :numref:`subsec_mdl-condition-number`.

**Courses and video lectures**

- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) by Gilbert Strang: a full lecture course with notes, exams, and solutions, covering :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-eigendecompositions`.
- [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) by Gilbert Strang: the follow-on course that connects the SVD and low-rank approximation of :numref:`sec_mdl-svd-low-rank` to deep learning and optimization.
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3Blue1Brown: short animated lectures on vectors, linear maps, determinants, and eigenvectors, the pictures of :numref:`sec_mdl-geometry-linear-algebraic-ops` through :numref:`sec_mdl-eigendecompositions`.
- [Singular Value Decomposition](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) by Steve Brunton: lectures on the geometry of the SVD, low-rank approximation, principal components, and randomized SVD, the material of :numref:`sec_mdl-svd-low-rank`.

**Tutorials and notes**

- [Linear Algebra Review and Reference](https://cs229.stanford.edu/section/cs229-linalg.pdf) by Kolter, updated by Do, for Stanford CS229: a compact refresher on the notation of :numref:`sec_mdl-geometry-linear-algebraic-ops` and :numref:`sec_mdl-linear-maps` as machine-learning courses use it.
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) by Ström, Åström, and Akenine-Möller: a free online book with interactive figures of linear maps, determinants, and eigenvectors, for :numref:`sec_mdl-linear-maps` and :numref:`sec_mdl-eigendecompositions`.
