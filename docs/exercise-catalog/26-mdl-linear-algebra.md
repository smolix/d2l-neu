# Chapter Overview — chapter_mdl-linear-algebra

Best source by far: **MIT 18.065** (Strang, *Matrix Methods in Data Analysis, Signal
Processing, and Machine Learning*, Spring 2018) — its lecture-keyed problem sets map
almost one-to-one onto our three files (orthogonal matrices/projections →
geometry; eigenvalues/positive-definiteness → eigendecomposition; SVD/Eckart-Young →
svd-low-rank), and its numbering follows the same build-up we use. **MIT 18.06**
(undergraduate Strang) supplies concrete rank/nullspace/Fredholm-alternative
computations one level more elementary than our text. The standout surprise is
**Axler's Linear Algebra Done Right**: its determinant-free eigenvalue chapter and its
polar-decomposition/SVD section (7.D) give short operator-theoretic problems that
cross-check our geometric claims (e.g. $\sigma\neq|\lambda|$) from a different
axiomatic direction. **Boyd & Vandenberghe (VMLS)** is the best source for real-data
cosine-similarity/least-squares problems, since all three of our files' own
data-driven exercises use synthetic Gaussian data. **Trefethen & Bau** supplies
by-hand SVD/unitary-equivalence exercises our conceptual treatment lacks. All three
files' existing exercise sets are strong per the prior style review (0 clarity
defects across 33 items except one flagged ambiguity); external material is used to
add or, in that one case, replace — never to displace strong originals. Coverage
gap: no suggested source has a problem-set tradition for tensor/Einstein-notation
exercises (our geometry file's final subsection) or for pseudospectra specifically
(Trefethen & Bau's own pseudospectra chapter is expository, not exercised) — both
noted as findings below. Stanford CS246 and Berkeley CS189 homework PDFs were
checked directly across several years; none yielded an assignable PageRank/PCA
problem statement stable enough to cite (only SVM/perceptron/exam material was
found), so the IR-book textbook is used in their place for the PageRank aside.

---

## chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md — Geometry and Linear Algebraic Operations

**Topic:** Vector geometry (angles, projections, bases), matrices as linear maps
(rank, invertibility, determinant), and tensor/Einstein notation.
**Current exercises:** 14; disposition: keep 14, rewrite 0, drop 0 — every item is a
direct compute/prove/verify task with a concrete deliverable and no clarity defects
(prior review); this is the strongest exercise set of the three files, so external
material below is purely additive.

**External sources found:**
- MIT 18.06 (Strang), *Linear Algebra*, Problem Set 5, Section 4.1 Problems 7 & 9
  (Spring 2010) — the Fredholm alternative: exactly one of $A\mathbf{x}=\mathbf{b}$
  or ($A^\top\mathbf{y}=\mathbf 0$ with $\mathbf y^\top\mathbf b=1$) is solvable, and
  $A^\top A\mathbf x=\mathbf 0 \Rightarrow A\mathbf x = \mathbf 0$ — both proved via
  orthogonality of the four subspaces. —
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ef8cf047287bc629eb243a691340c823_MIT18_06S10_pset5_s10_soln.pdf
- MIT 18.06, Problem Set 4, Section 3.5 Problem 20 & Section 3.6 Problem 28 (Spring
  2010) — find a basis for a plane-as-nullspace, its intersection with a coordinate
  plane, and its orthogonal complement; separately, find the rank of an 8×8
  checkerboard matrix and a structured "chess" matrix and bases for their row space
  and left nullspace. —
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/4ece22f9c707878e1e57b9840469490e_MIT18_06S10_pset4_s10_soln.pdf
- MIT 18.065 (Strang), Problem Set I.5 (Lecture 3, "Orthonormal Columns"), Problems
  2, 4, 6 (Spring 2018) — draw non-orthogonal unit vectors and Gram–Schmidt one
  against the other; prove $\|Q\mathbf x\|=\|\mathbf x\|$ and
  $(Q\mathbf x)^\top(Q\mathbf y)=\mathbf x^\top \mathbf y$ for orthogonal $Q$; show a
  given permutation matrix is orthogonal. —
  https://www.ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf
- MIT 18.065, Problem Set II.2 (Lecture 10), Problem 17 (Spring 2018) — project
  $\mathbf b=(0,8,8,20)$ onto the line through $\mathbf a=(1,1,1,1)$, verify the
  residual is perpendicular to $\mathbf a$, and report the distance. (Same URL as
  above.)
- Boyd & Vandenberghe, *VMLS* Additional Exercises, Ch. 3 "Norm and distance", Ex.
  3.19–3.20 (2025 revision) — interpret $\angle(x,y)=0$ or $90°$ for word-count
  document vectors, then find the 10 nearest neighbors of a chosen article among 500
  Wikipedia articles by cosine distance on their word-histogram vectors. —
  https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf
- Boyd & Vandenberghe, *VMLS* Additional Exercises, Ch. 5 "Linear independence", Ex.
  5.2, 5.7 (2025 revision) — does reversing the input order of Gram–Schmidt reverse
  the output basis; given $a_1+a_3=a_2+a_4$ and $\{a_1,a_2,a_3\}$ independent,
  classify the dependence of five derived vector sets. (Same URL as above.)

**Proposed problem set** (6 problems):
1. [conceptual] **Fredholm alternative from orthogonal complements.** State and
   prove: for $A\in\mathbb R^{m\times n}$, exactly one of $A\mathbf x=\mathbf b$
   solvable, or $\exists\,\mathbf y$ with $A^\top\mathbf y=\mathbf 0,\,
   \mathbf y^\top\mathbf b\neq0$, holds. Deliverable: a two-line proof from "column
   space and left nullspace are orthogonal complements," plus one explicit $3\times3$
   numerical instance of each case.
   *Provenance:* adapted from MIT 18.06, Problem Set 5, Problems 7 & 9 (overlap med;
   cite on adoption).
2. [short-code] **Rank of a patterned matrix.** Build the $8\times 8$ checkerboard
   matrix $B$ with $B_{ij}=(i+j)\bmod 2$ in code, compute its rank numerically, and
   by hand find a basis for its row space and left nullspace using only the first
   two rows. Verify the code-computed rank matches your hand count.
   *Provenance:* adapted from MIT 18.06, Problem Set 4, Section 3.6 Problem 28
   (overlap med; cite on adoption).
3. [conceptual] **Orthogonality of permutation matrices.** For the 4×4 cyclic
   permutation matrix $P$ with $Pe_i=e_{i+1\bmod 4}$, verify $P^\top P=I$ directly
   from the column definition, then prove every permutation matrix is orthogonal in
   general and conclude $P$ preserves all pairwise angles among a set of vectors.
   *Provenance:* adapted from MIT 18.065, Problem Set I.5, Problem 6 (overlap high;
   cite on adoption).
4. [short-code] **Cosine similarity on real short texts.** Build word-count vectors
   in code for five one-sentence documents you write yourself (a small fixed
   dictionary), compute all pairwise cosine similarities, and find the nearest
   neighbor of each document. Contrast the range of similarities you observe with
   the $1/\sqrt d$ concentration found for random Gaussian vectors in the section's
   own high-dimensional-similarity exercise.
   *Provenance:* inspired by Boyd & Vandenberghe VMLS Ex. 3.19–3.20 (overlap low —
   different, self-authored dataset in place of the Wikipedia corpus).
5. [conceptual] **Projection onto a two-dimensional subspace.** Project
   $\mathbf b=(1,2,3,4)$ onto $\operatorname{span}\{\mathbf a_1,\mathbf a_2\}$ for
   $\mathbf a_1=(1,1,1,1)$, $\mathbf a_2=(0,1,2,3)$ by solving the $2\times2$ normal
   equations for the coefficients, and verify the residual is orthogonal to both
   $\mathbf a_1$ and $\mathbf a_2$.
   *Provenance:* inspired by MIT 18.065, Problem Set II.2, Problem 17 (overlap low —
   generalizes the cited line-projection to a plane, which the source problem does
   not cover).
6. [extended] **A five-document retrieval system.** Collect or write ten short
   documents over a fixed 30–50 word dictionary, build the term-count matrix,
   rank it (are all documents linearly independent as vectors?), and implement
   cosine-similarity nearest-neighbor retrieval for a held-out query sentence built
   from the same dictionary. Report the rank of the document matrix and the top-2
   retrieved documents for two different queries.
   *Provenance:* inspired by Boyd & Vandenberghe VMLS Ex. 3.20 (overlap low — scaled
   into a small end-to-end retrieval exercise rather than a single lookup).

---

## chapter_mdl-linear-algebra/mdl-eigendecomposition.md — Eigendecompositions

**Topic:** Eigenvalues/eigenvectors, diagonalizability and the Jordan form, the
spectral theorem and positive-definiteness, Gershgorin/power iteration, and spectral
radius vs. transient growth.
**Current exercises:** 11; disposition: keep 11, rewrite 0, drop 0 — all 11 are
direct compute/prove/verify tasks with concrete deliverables and no clarity defects
(prior review flagged only a bare-list numbering/formatting quirk, not a content
issue); external material below is purely additive.

**External sources found:**
- MIT 18.065 (Strang), Problem Set I.6 (Lecture 4, "Eigenvalues and Eigenvectors"),
  Problems 2, 11, 15 (Spring 2018) — compute eigenvalues/eigenvectors of $A$ and
  $A^{-1}$ and check the trace; show $A$, $A^\top$ share eigenvalues but not
  eigenvectors; factor two given $2\times2$ matrices as $A=X\Lambda X^{-1}$ and use
  it to get $A^3$ and $A^{-1}$. —
  https://www.ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf
- MIT 18.065, Problem Set I.7 (Lecture 5, "Positive Definite and Semidefinite
  Matrices"), Problems 3, 14, 15 (Spring 2018) — find $b,c$ making three given
  matrices positive definite via $LDL^\top$ pivots; back out a $3\times3$ matrix $S$
  from a given quadratic form; verify that ratios of upper-left determinants equal
  the second and third pivots for a specific $S$. (Same URL as above.)
- MIT 18.065, Problem Set II.4 (Lecture 13), Problem 4 (Spring 2018) — for
  $M=\mathbf 1\mathbf 1^\top$ (the $n\times n$ all-ones matrix), prove $nI-M$ is
  positive semidefinite and find its eigenvalues. (Same URL as above.)
- Axler, *Linear Algebra Done Right* (4th ed.), Exercises 5.B, Problems 7–8 — prove
  $9$ is an eigenvalue of $T^2$ iff $3$ or $-3$ is an eigenvalue of $T$; construct an
  operator on $\mathbb R^2$ with $T^4=-I$ (forcing complex eigenvalues on a real
  space). —
  https://web.math.ucsb.edu/~bigelow/books/axler.pdf
- Axler, Exercises 5.C, Problem 8 — for $T\in\mathcal L(\mathbb F^5)$ with
  $\dim E(8,T)=4$, prove $T-2I$ or $T-6I$ is invertible (an eigenspace-dimension /
  diagonalizability argument with no determinants). (Same URL as above.)
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (Stanford),
  §21.3 "The PageRank Computation" — a worked 7-node web-graph example (teleport
  rate $0.14$) whose computed PageRank vector shows a page with several inbound
  links ranked *lowest* because the random walk drifts away from it. —
  https://nlp.stanford.edu/IR-book/html/htmledition/the-pagerank-computation-1.html

**Proposed problem set** (6 problems):
1. [conceptual] **Spectrum of the all-ones deflation.** For $M=\mathbf1\mathbf1^\top$
   ($n\times n$), prove $nI-M$ is positive semidefinite by writing
   $\mathbf x^\top(nI-M)\mathbf x$ as a sum of squares, then find all eigenvalues of
   $nI-M$ explicitly (one is $0$, the rest are $n$) and identify their eigenvectors.
   *Provenance:* adapted from MIT 18.065, Problem Set II.4, Problem 4 (overlap high;
   cite on adoption).
2. [conceptual] **Eigenvalues under squaring.** Prove that if $\lambda$ is an
   eigenvalue of $A$ then $\lambda^2$ is an eigenvalue of $A^2$, and that the
   converse ($\mu$ an eigenvalue of $A^2$ implies $\pm\sqrt\mu$ an eigenvalue of $A$)
   can fail over $\mathbb R$. Check both directions on
   $A=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$.
   *Provenance:* adapted from Axler, Exercises 5.B, Problem 7 (overlap med; cite on
   adoption).
3. [short-code] **A real matrix, complex rotation.** Find a real
   $2\times 2$ matrix $A$ with $A^4=-I$ (hence purely imaginary eigenvalues), verify
   in code that $A^4$ is numerically $-I$, and plot the images of the unit circle
   under $A,A^2,A^3,A^4$ to show the four successive quarter-turns, connecting to
   the section's "complex eigenvalues are rotations" result.
   *Provenance:* adapted from Axler, Exercises 5.B, Problem 8 (overlap med; cite on
   adoption).
4. [conceptual] **Eigenspace dimension and blocked eigenvalues.** For $T$ on a
   5-dimensional space with $\dim E(8,T)=4$, prove $T-2I$ or $T-6I$ is invertible.
   (*Hint:* eigenspaces of three distinct eigenvalues are always independent, so if
   $8,2,6$ were all eigenvalues their eigenspace dimensions would sum to at least
   $4+1+1=6$, exceeding $\dim V=5$ — a contradiction.) Then
   build a $5\times5$ numerical example with $\dim E(8,T)=4$ where $6$ is the fifth
   eigenvalue, and verify in code that $T-2I$ is invertible while $T-6I$ is not.
   *Provenance:* adapted from Axler, Exercises 5.C, Problem 8 (overlap high; cite on
   adoption).
5. [short-code] **Pivots, ratios, and positive definiteness.** For
   $S=\left[\begin{smallmatrix}2&2&0\\2&5&3\\0&3&8\end{smallmatrix}\right]$, compute
   the three leading-principal-minor determinants by hand, verify the second and
   third pivots equal the ratios of consecutive determinants, and confirm in code
   that all eigenvalues of $S$ are positive.
   *Provenance:* adapted from MIT 18.065, Problem Set I.7, Problem 15 (overlap high;
   cite on adoption).
6. [extended] **PageRank on a hand-built graph.** Construct a 6–7 node
   directed graph in code with at least one dangling page and one node with three
   inbound links, repair dangling columns, form the damped transition matrix with
   $\alpha=0.85$, and run power iteration (reusing the section's own power-iteration
   cell) to convergence. Check whether your three-inbound-link node ends up ranked
   below a node with fewer inbound links, and explain the drift using the graph's
   structure.
   *Provenance:* inspired by Manning, Raghavan & Schütze §21.3 (overlap low — a
   different, self-built graph reproducing the qualitative phenomenon rather than
   their specific 7-node example).

---

## chapter_mdl-linear-algebra/mdl-svd-low-rank.md — Singular Value Decomposition and Low-Rank Approximation

**Topic:** The SVD as rotate-scale-rotate, the four fundamental subspaces, Eckart–
Young low-rank approximation and PCA, the pseudoinverse/least squares/condition
number, and SVD in LoRA/Muon/spectral analysis.
**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — seven items are
direct, well-scoped compute/prove/verify tasks; exercise 7 (LoRA rank/spectral-decay
follow-up) is rewritten per the prior style review's clarity flag (ambiguous whether
to read an existing cell's output or build a new matrix, and an unscoped "how does
the answer change" follow-up with no parameter or metric given).

**External sources found:**
- MIT 18.065 (Strang), Problem Set I.8 (Lecture 6, "Singular Value Decomposition"),
  Problems 1, 6 (Spring 2018) — explain
  $\mathbf x^\top\mathbf x=\sum c_i^2$, $\mathbf x^\top S\mathbf x=\sum\lambda_i c_i^2$
  for symmetric $S$; find $\sigma$'s, $\mathbf v$'s, $\mathbf u$'s for
  $A=\left[\begin{smallmatrix}3&4\\0&5\end{smallmatrix}\right]$. —
  https://www.ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf
- MIT 18.065, Problem Set I.9 (Lecture 7, "Eckart–Young"), Problems 2, 10 (Spring
  2018) — find the closest rank-1 approximation (in $2$-norm or Frobenius norm) to
  three given matrices; for a $2\times2$ $A$ with $\sigma_1\ge\sigma_2>0$, find
  $\|A^{-1}\|_2$ and $\|A^{-1}\|_F^2$. (Same URL as above.)
- Trefethen & Bau, *Numerical Linear Algebra* (SIAM, 1997), Exercises 4.1, 4.4, 4.5 —
  compute the SVD of $\operatorname{diag}(3,-2)$ by hand via the eigendecomposition
  of $A^\top A$; prove unitary equivalence ($A=QBQ^*$, one fixed $Q$ conjugating)
  implies equal singular values, and give a same-singular-value counterexample
  showing the converse fails; show a real matrix's $A^\top A$ is real symmetric,
  hence its SVD can always be taken real. — verified via handwritten solution
  images at
  https://raw.githubusercontent.com/desh2608/numerical-linear-algebra/master/Sol4.pdf
  (community solutions manual restating each problem before solving it; original
  problem numbering confirmed against the textbook's Lecture 4 exercise list).
- Axler, *Linear Algebra Done Right* (4th ed.), Exercises 7.D, Problems 5, 10, 11 —
  find the singular values of $T(x,y)=(-4y,x)$ on $\mathbb C^2$ directly; prove that
  for self-adjoint $T$ the singular values equal $|\lambda_i|$; prove $T$ and $T^*$
  always have the same singular values. —
  https://web.math.ucsb.edu/~bigelow/books/axler.pdf
- Boyd & Vandenberghe, *VMLS* Additional Exercises, Ch. 12 "Least squares", Ex. 12.1
  (2025 revision) — solve a random $20\times10$ least-squares problem three ways
  (built-in solver, normal equations $(A^\top A)^{-1}A^\top b$, pseudoinverse
  $A^{+}b$), verify agreement, and confirm that perturbing $\hat x$ strictly
  increases the residual. —
  https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf

**Proposed problem set** (7 problems):
1. [conceptual] **Hand computation of an SVD.** For $A=\operatorname{diag}(3,-2)$,
   form $A^\top A$, find its eigenvalues and eigenvectors by hand, read off the
   singular values and $V$, then recover $U=AVS^{-1}$ and verify
   $A=U\Sigma V^\top$ by direct multiplication.
   *Provenance:* adapted from Trefethen & Bau, Exercise 4.1 (overlap high; cite on
   adoption).
2. [conceptual] **Singular values of a scaled rotation.** For the real-linear map
   $T(x,y)=(-4y,x)$ on $\mathbb R^2$ (matrix
   $\left[\begin{smallmatrix}0&-4\\1&0\end{smallmatrix}\right]$), find its
   eigenvalues (purely imaginary) and its singular values from $T^\top T$ directly,
   confirming $\sigma\neq|\lambda|$ with a cleaner pair of numbers than the
   section's own rotation-scaling example.
   *Provenance:* adapted from Axler, Exercises 7.D, Problem 5 (overlap high; cite on
   adoption).
3. [short-code] **Least squares three ways.** For a random $20\times10$ matrix $A$
   and $20$-vector $b$, compute $\hat x$ via the platform solver, via
   $(A^\top A)^{-1}A^\top b$, and via the pseudoinverse $A^{+}b$; verify all three
   agree to numerical precision, then perturb $\hat x$ by a small random $\delta$
   and confirm $\|A(\hat x+\delta)-b\|^2>\|A\hat x-b\|^2$.
   *Provenance:* adapted from Boyd & Vandenberghe, VMLS Ex. 12.1 (overlap high; cite
   on adoption).
4. [conceptual] **Closest rank-1 approximation by inspection.** For
   $A=\operatorname{diag}(3,2,1)$ and
   $B=\left[\begin{smallmatrix}0&3\\2&0\end{smallmatrix}\right]$, read the SVD off
   by inspection and give the closest rank-1 approximation in Frobenius norm to
   each, stating the resulting approximation error $\|A-A_1\|_F$ in closed form.
   *Provenance:* adapted from MIT 18.065, Problem Set I.9, Problem 2 (overlap high;
   cite on adoption).
5. [short-code] **LoRA rank under two spectral profiles** (rewrite of exercise 7).
   Using the section's own weight matrix from the `#svd-weight-spectrum` cell,
   report the smallest LoRA rank $r_1$ achieving 95% spectral energy and the
   resulting parameter saving. Then construct a second synthetic weight matrix of
   the same shape whose singular values decay as $\sigma_i\propto i^{-0.5}$ (instead
   of the original's faster decay), find its rank $r_2$ for the same 95% threshold,
   and report the ratio $r_2/r_1$ as the concrete metric of "how the answer
   changes."
   *Provenance:* original (rewrite addressing the prior review's clarity flag on
   this exercise).
6. [conceptual] **Two-sided orthogonal equivalence.**
   Prove that $A,B\in\mathbb R^{n\times n}$ have identical singular values iff
   $A=P B Q$ for *some* orthogonal $P,Q$ (possibly different), then exhibit two
   $2\times2$ matrices with the same singular values for which no *single*
   orthogonal $Q$ satisfies $A=QBQ^\top$.
   *Provenance:* adapted from Trefethen & Bau, Exercise 4.4 (overlap med; cite on
   adoption).
7. [extended] **Image compression via truncated SVD.** Take any $m\times n$ array
   from the section's own tools (e.g. a rendered grayscale figure or a generated
   texture), compute its full SVD, and for $k=1,2,5,10,20$ plot both the
   reconstruction error $\|A-A_k\|_F$ and the retained-energy ratio
   $\sum_{i\le k}\sigma_i^2/\sum_i\sigma_i^2$, verifying the Eckart–Young identity
   numerically. Report the smallest $k$ retaining 99% energy and show the image at
   that $k$.
   *Provenance:* original (extends the section's own Eckart–Young and PCA material
   into an end-to-end compression demo).
