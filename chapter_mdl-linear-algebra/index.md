# Linear Algebra
:label:`chap_mdl-linear-algebra`

Linear algebra is the language in which data, parameters, and the operations of
a network are written. This chapter develops the geometry behind that
language---vectors, dot products, and the way matrices stretch and rotate
space---and then the two decompositions that recur throughout deep learning:
the *eigendecomposition* (behind stability analysis, PCA, and the curvature of
loss surfaces) and the *singular value decomposition* (behind low-rank
approximation, conditioning, and parameter-efficient fine-tuning).

```toc
:maxdepth: 2

mdl-geometry-linear-algebraic-ops
mdl-eigendecomposition
mdl-svd-low-rank
```

## Resources and Further Reading

The references below go deeper on the geometry, decompositions, and numerical machinery sketched in this chapter, with an emphasis on the linear algebra that actually shows up in machine and deep learning; most are freely accessible online.

**Books**

- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — free PDF; chapters 2 (Linear Algebra), 3 (Analytic Geometry), and 4 (Matrix Decompositions) are the most ML-aligned treatment of exactly this material.
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares (VMLS) — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/vmls/) — free PDF, slides, and code; an applications-first text built around least squares and data, ideal for building intuition before the abstraction.
- [Linear Algebra and Learning from Data — Gilbert Strang](https://math.mit.edu/~gs/learningfromdata/) — the SVD-centric companion to MIT 18.065; written specifically around matrix methods for signal processing and deep learning. Of the two Strang books, read this one second: it assumes the foundations and matches this chapter's SVD and low-rank material most closely.
- [Introduction to Linear Algebra, 6th ed. — Gilbert Strang](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) — the classic foundational text (a commercial book; sample chapters and problems are online); the standard first pass on the four subspaces, eigenvalues, and the SVD, and the Strang book to start with.
- [Linear Algebra Done Right, 4th ed. — Sheldon Axler](https://linear.axler.net/) — free, open-access; a rigorous, determinant-free path to eigenvalues and the spectral theorem for readers who want the proofs.
- [Numerical Linear Algebra — Trefethen & Bau](https://people.maths.ox.ac.uk/trefethen/text.html) — the reference on conditioning, stability, QR, and how the SVD and eigendecomposition are actually computed in floating point.

**Courses and video lectures**

- [MIT 18.06 Linear Algebra (OCW) — Gilbert Strang](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — the canonical full-semester lecture series, with notes, exams, and solutions.
- [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning (OCW) — Gilbert Strang](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) — the graduate follow-on that connects linear algebra directly to deep learning and optimization.
- [Essence of Linear Algebra — 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — a short, visually driven series that builds geometric intuition for vectors, transformations, determinants, and eigenvectors.
- [Singular Value Decomposition — Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) — a lecture series devoted entirely to the SVD (geometry, low-rank approximation, PCA, randomized SVD), the video companion to *Data-Driven Science and Engineering*; it covers exactly the material 3Blue1Brown stops short of.

**Tutorials, notes, and visual introductions**

- [Linear Algebra Review and Reference — Stanford CS229 (Kolter, updated by Do)](https://cs229.stanford.edu/section/cs229-linalg.pdf) — a compact PDF refresher covering exactly the notation and matrix calculus assumed in ML courses.
- [Immersive Linear Algebra — Ström, Åström & Akenine-Möller](http://immersivemath.com/ila/index.html) — a free online textbook whose fully interactive figures let you manipulate the geometry of transformations and eigenvectors directly.
- [LoRA: Low-Rank Adaptation of Large Language Models — Hu et al.](https://arxiv.org/abs/2106.09685) — a concrete modern payoff of low-rank structure: adapting large models by training only a low-rank update to the weight matrices.
