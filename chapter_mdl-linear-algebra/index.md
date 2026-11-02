# Linear Algebra
:label:`chap_mdl-linear-algebra`

A $28\times28$ image is a vector of 784 numbers, and a layer of a network is
a matrix that maps it to another vector. We cannot draw 784 dimensions, yet
we need to answer geometric questions there: how similar two images are,
which side of a boundary an image falls on, whether a layer loses
information, what happens to a signal that passes through the same map fifty
times, and how much of a large weight matrix a few hundred numbers can
describe.

The Basics introduced vectors, matrix products, norms, and eigenvectors as
tools of computation. They did not say when a map can be undone, why a
recurrence whose eigenvalues all lie inside the unit circle can grow tenfold
before it decays, or which matrix of rank $k$ is closest to a given one. Each
of these questions has a simple answer once we pick the right directions:
orthonormal axes, eigenvectors, or a pair of singular bases, in which the
matrix becomes a list of numbers. One $2\times2$ matrix,
$\mathbf{A}=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$,
and one shear,
$\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$, recur through
the chapter, the first as the typical case and the second as the case where
eigenvectors fail.

:numref:`sec_mdl-geometry-linear-algebraic-ops` asks how lengths, angles,
and projections work in any dimension, and uses them to fit lines and to
classify images with one dot product. :numref:`sec_mdl-linear-maps` asks
what a matrix does to all of space at once: which inputs it sends to zero,
when it can be inverted, and by how much it scales volume.
:numref:`sec_mdl-eigendecompositions` asks what happens when the same square
matrix is applied again and again, and when its eigenvalues give the wrong
answer to that question. :numref:`sec_mdl-svd-low-rank` asks how every
matrix, square or not, splits into independent stretches, and uses the
split to approximate matrices at low rank, to solve least-squares problems,
and to measure how sensitive a solution is to errors in the data.

We work in real finite-dimensional spaces and use complex numbers only where
real matrices force them on us, for rotations. Algorithms appear only as far
as the mathematics explains them; how libraries factor large matrices is
left to the Further Reading.

```toc
:maxdepth: 2

mdl-geometry-linear-algebraic-ops
mdl-linear-maps
mdl-eigendecomposition
mdl-svd-low-rank
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. Most are freely accessible online. Topics that the
chapter does not teach, such as kernel methods, sparse and iterative
solvers, the QR algorithm for eigenvalues, mixing times of Markov chains,
and matrix completion, are collected in
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
