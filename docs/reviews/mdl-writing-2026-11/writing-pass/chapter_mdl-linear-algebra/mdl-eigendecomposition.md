# Writing pass: `chapter_mdl-linear-algebra/mdl-eigendecomposition.md`

Date: 2026-11-09. Base: `4deb96d6`. To be split in two; one marker placed. No
code cell, cell ID, figure include, existing label or slide changed.
`check_mdl_prose.py`: "OK: only prose changed", no em-dash warning, 1 split
marker, display-equation delimiters 102 to 104 (one new display, EIG-4).
`tools/lint_source.py`: clean.

## Words

Checker count (to the slides marker, exercises included): 10,831 before,
10,779 after (-52). Body to the Summary: 8,459 to 8,409; Summary 179 to 177.
Part one (to the seam) 5,219 to 5,212; part two 3,236 to 3,196.

The cut is far below the review's "about 2,000". Reasons: (a) the pass added
what the assignment and the review asked for (ranking paragraph in the
opening, an opening for part two with the ranking of its topics, the repair
of two proofs, the Marchenko-Pastur hypotheses, the reason for $\mu<0$, the
PageRank judgment), about 420 words; (b) the counter counts every label
token, and about 25 `:numref:` were added; (c) the largest items on the cut
list (Pseudospectra, the Euler cell) carry a cell, a label that
`further-reading.md` and exercise 8 cite, and a labelled equation, so only
the prose around them could shrink. Prose actually removed: about 520 words.

## Correctness

- **EIG-6** (unjustified step in the proof of $g\le a$). Was: "By
  :eqref:`eq_mdl-similar-charpoly`, $p(t)=(\lambda-t)^g\det(\mathbf
  Y-t\mathbf I)$". Now the proof says that the similar matrix has the
  characteristic polynomial $p$ of $\mathbf A$, that the variable is written
  $t$ because $\lambda$ is the fixed eigenvalue, and obtains the factor by
  cofactor expansion along the first column, $g$ times: that column holds
  the single entry $\lambda-t$ on the diagonal, so each expansion leaves
  $\lambda-t$ times a determinant of the same form with $g-1$. Cites
  :numref:`subsec_mdl-determinant-general`, which teaches cofactor expansion
  along the first column. Verified by hand (the $(1,1)$ cofactor has sign
  $+1$; the minor is again block upper triangular). I did not cite a block
  rule in linear maps, because I cannot know whether that file's writer adds
  one. If it is added, this proof can cite it in one clause.
- **EIG-7** (proposition used before stated). The proposition "distinct
  eigenvalues give independent eigenvectors" and its proof (text unchanged)
  now come first in the subsection; the definitions of the two
  multiplicities, the proposition $g\le a$ and the diagonalizability
  criterion follow. The parenthesis is replaced by the reviewer's sentence:
  group a vanishing combination by eigenvalue; each group is zero or an
  eigenvector; the proposition forces every group to be zero. The criterion
  now also says "and $g\le a$", which it uses. No code cell is in this
  subsection, so no cell moved.
- **EIG-5**. "with the same multiplicities" is now "with the same
  multiplicities as roots".
- **EIG-13**. Was: "the eigenvalues of $\mathbf C$ fill the interval ...
  with a density that the theorem also gives". Now: "the histogram of the
  eigenvalues of $\mathbf C$ converges to a fixed density supported on the
  interval ..., and with a finite fourth moment the extreme eigenvalues
  converge to its edges". "$d-n$ eigenvalues are exactly zero" is now "at
  least $d-n$ eigenvalues are zero". "A sample covariance is such a Gram
  matrix" is now "The sample covariance of centred data is such a Gram
  matrix". From memory of Bai and Yin (1993); no citation key for it exists,
  so none was added.
- **EIG-14**. The sentence on Gaussian elimination and Jacobi iteration is
  deleted, not reworded (no use in the book; Gershgorin is ranked as
  skippable).
- **EIG-12**. "the smallest sets the slowest rate" was true only for small
  steps. Now: "at step sizes $\eta\le1/\lambda_{\max}$ the smallest sets the
  slowest rate, $1-\eta\lambda_{\min}$". Checked: for $\eta\le1/\lambda_{\max}$
  all factors $1-\eta\lambda_i$ lie in $[0,1)$ and the largest belongs to
  $\lambda_{\min}$.
- **EIG-4**. The factorization $\det(\mathbf A-\lambda\mathbf
  I)=\prod_i(\lambda_i-\lambda)$ is now displayed, so "on the right" and "on
  the left" refer to something.
- **EIG-9** (numbers no cell prints), all recomputed and **kept**: the rate
  $\|\mathbf A^{20}\mathbf x\|^{1/20}=2.252$ ("2.25"); 43% of 20,000
  Gaussian $5\times5$ matrices have a complex dominant pair ("about four in
  ten"); gap ratio of uniform $5\times5$ matrices has 5%/95% quantiles 0.14
  and 0.37, 1%/99% 0.11 and 0.43 ("typically between 0.1 and 0.4"). Also
  rechecked and kept: $6\times10^{-7}$ and below $10^{-30}$ for the surfer;
  $58^\circ$, $-32^\circ$; 13.6 steps per turn; sines 0.316, 0.110, 0.037
  and Rayleigh errors 0.2, 0.024, 0.0027; $3.99\pm0.96i$, 1.03 and the
  distance 0.96; peak at $k\approx6.4$ with value 10.8, $1.4^\circ$,
  $\kappa_2(\mathbf V)=80$, bound 42.5; eigenvalues $1,-\tfrac23,0$ of the
  three-page matrix.
- **EIG-22** (damped PageRank numbers no cell prints). $-0.567$ and
  $(0.303,0.394,0.303)$ are correct (recomputed) but **dropped** from the
  body: exercise 10e asks the reader to find exactly these.
- Disagreements with the review: none.

## Six before/after pairs

1. Before: "Does the library agree? The cell computes the eigenpairs of
   $\mathbf P$ and checks the defining equation for all of them at once."
   After: "In practice a library does this work. NumPy's `eig` returns the
   eigenvalues and, as the columns of a matrix, one eigenvector for each;
   the last line of the cell checks :eqref:`eq_mdl-eigpair` for all pairs at
   once."
2. Before: "scale by $\sqrt5$ and rotate by $26.6^\circ$. The cell computes
   the block from `eig` and follows the iterates $\mathbf A^k(1,0)$." After:
   "scale by $\sqrt5$ and rotate by $26.6^\circ$. If that were the whole
   description, every step would lengthen a vector by the factor
   $\sqrt5\approx2.24$. We compute the block from the eigenvector that `eig`
   returns and then follow the lengths of the iterates $\mathbf A^k(1,0)$."
3. Before: "The spectral theorem states that this holds for every symmetric
   matrix." After: "The spectral theorem states that this holds for every
   symmetric matrix. It is the theorem of this chapter: positive
   definiteness, the Rayleigh quotient, and all of
   :numref:`sec_mdl-svd-low-rank` rest on it."
4. Before: "Which matrices are safe? A matrix with positive entries has a
   real, simple, strictly dominant eigenvalue, by the Perron--Frobenius
   theorem of the next subsection. The cell runs 200 steps of power
   iteration ..." After: "A matrix with positive entries is safe: it has a
   real, simple, strictly dominant eigenvalue, by the Perron--Frobenius
   theorem of :numref:`subsec_mdl-pagerank`. We run 200 steps of power
   iteration ..."
5. Before: "The largest singular value behaves differently:
   $\sigma_{\max}(\mathbf A)/\sqrt n\to2$ :cite:`Yin.Bai.Krishnaiah.1988`.
   The cell checks both for $n=1000$." After: "The spectral norm behaves
   differently: $\|\mathbf A\|_2/\sqrt n\to2$
   :cite:`Yin.Bai.Krishnaiah.1988`. This is the Marchenko--Pastur law of
   :numref:`subsec_mdl-sample-covariance-mp` at aspect ratio $c=1$. The
   largest eigenvalue of $\tfrac1n\mathbf A^\top\mathbf A$ tends to the
   upper edge $(1+\sqrt c)^2=4$, and $\|\mathbf A\|_2^2$ is $n$ times that
   eigenvalue. At $n=1000$ both limits are close."
6. Before: "We can now compute the eigenvalues of a small matrix by hand and
   write $\mathbf A^k\mathbf x$ in closed form whenever an eigenbasis
   exists. ... Two questions remain open. ..." After: "The random matrix
   and $\mathbf M$ make the same point: eigenvalues say little about the
   stretch in a finite number of steps. Nor do they exist for rectangular
   matrices. The singular values of :numref:`sec_mdl-svd-low-rank` measure
   stretch directly, for every matrix."

## Repetition

- **Condition number** (owner: SVD section, `subsec_mdl-condition-number`).
  The definition and the bowl are gone from "The Rayleigh Quotient"; one
  clause with the reference remains, and $L$ is tied to
  :numref:`sec_mdl-gradient-based-optimization`. The $\kappa_2(\mathbf V)$
  of the non-normal bound now cites the same label.
- **Marchenko-Pastur** (owner per the table: one "Random Matrices"
  subsection at the end of this file). The subsection cannot move, because
  its cell would cross six others. It stays in part one, and "Random
  Matrices at Initialization" now derives $\|\mathbf A\|_2/\sqrt n\to2$ from
  it (EIG-19) instead of reporting a separate fact. Deferred: the move.
- **Matrix exponential** (owner: Dynamics). One sentence and
  :numref:`sec_mdl-odes-solvers`; the stability criterion and the
  continuous-time spiral are gone.
- **Weyl / Courant-Fischer** (owner: lemma in SVD). The body now says that
  part (d) of the exercise "asks for the proof and gives a hint". Making it
  an exercise of its own that cites the SVD lemma is deferred (exercises
  were not restructured).
- Inside the file: the norm ratios "1.4 and 3.5, period 13.6" are no longer
  repeated under Power Iteration (a reference instead); the closing recap is
  gone; the Summary no longer repeats the surfer's numbers.
- Kept because this file owns them: the eigen-expansion, the spectral
  theorem, definiteness from eigenvalues, the Rayleigh proposition, power
  iteration, the spectral norm's first definition, transient growth.

## Ranking; Proposition blocks

- No Proposition block was demoted. The five blocks (similar matrices,
  distinct eigenvalues, $g\le a$, definiteness, Rayleigh) are each cited
  later or carry a multi-line proof.
- Ranking added: the opening names the three results that matter (the
  eigen-expansion, the spectral theorem, the quadratic form as a sum of
  squares) and the short route for a reader in a hurry; the spectral theorem
  is called the theorem of the chapter; its proof says where the work is
  (part iii); the Jordan subsection says which single consequence is needed;
  the real Jordan form is one sentence (EIG-10).
- Labels added: none.

## Split

- Marker `<!-- split: new section "Spectra in Practice" starts here -->` at
  line 1058, immediately before `## Locating and Computing Eigenvalues`.
- Part two opens with two new paragraphs under that heading: its two
  questions, then the ranking. Deep learning uses power iteration (spectral
  normalization, `subsec_mdl-spectrum-control`), transient growth and
  products of Jacobians (`sec_bptt`), and random matrices
  (`sec_numerical_stability`). Gershgorin and PageRank are marked as
  classical and skippable, Pseudospectra as skippable (also said in its
  first sentence).
- Backward dependencies removed from part one: the Rayleigh subsection no
  longer cites Power Iteration for the ordering; the opening roadmap
  describes part two in one sentence that can be reworded or cut at the
  split ("The second half of the section, from ... on"). Part one still
  points forward three times, as pointers only: `subsec_mdl-power-iteration`
  (large matrices), `subsec_mdl-transient-growth` (after the Jordan cell),
  `subsec_mdl-normal-matrices` (after the running-matrix cell).
- Part two cites part one by label: the eigen-expansion, the spectral
  theorem (proof of the complex version), `subsec_mdl-complex-rotation`,
  `subsec_mdl-sample-covariance-mp`.
- Part one ends on the Marchenko-Pastur reading (only an eigenvalue above
  the edge is structure). The Summary has two paragraphs, one per part.
- For the split itself: part two needs its own imports cell and section
  label; exercises 3, 4, 6, 8, 9, 10 and 12 use part-two material (3 and 4
  cite the non-normal bound and the normal-power identity).

## Cells to cut or redesign on the development machine

- `eigendecomposition-complex-rotation` (EIG-8): NumPy agrees with Euler's
  formula at one angle. Prose reduced to a clause. Cut.
- `mdl-eigendecomposition-pseudospectra` (EIG-16): two resolvent norms. If
  the subsection moves to Further Reading, keep the number 200.3 against 10
  as two sentences at the end of "Transient Growth" and cut the cell.
- `eigendecomposition-gershgorin-circle-theorem` (no finding ID): confirms
  the theorem on the example whose eigenvalues the text already quotes.
  Prose reduced to one sentence before and one after.
- `eigendecomposition-psd-gram` and the Rayleigh sweep (no finding ID):
  both confirm a proved statement on one small matrix. Prose shortened; one
  of the two would be enough.
- EIG-9: add a printed line for the average rate at $k=20$ in
  `mdl-eigendecomposition-complex-eigenvalues-are-rotations-1`.

## Other deferred findings

- EIG-3 (notation): bold $\mathbf A$ is still the generic matrix, the
  shear, the running matrix, the Gershgorin example and the random matrix;
  $\mathbf P$ is the $2\times2$ and the $3\times3$ link matrix; $\mu$ is an
  eigenvalue, a real number, the Lyapunov exponent and a shift. Shared
  symbols, and code variables are named `A`; not changed.
- EIG-16: the definition of the pseudospectrum stays in the body because
  `eq_mdl-pseudospectrum` and `subsec_mdl-pseudospectra` are cited by
  exercise 8 and `further-reading.md`. Moving it needs both changed.
- EIG-15, exercises: splitting exercise 7d into "Courant-Fischer and Weyl";
  the easy exercises on the core (Fibonacci); exercise 8c.
- EIG-2: the remark that $\mathbf A\mathbf A^\top$ of the triangular example
  is twice the Gram matrix of the shear (why $58.3^\circ$ recurs) was not
  added.
- The continuity argument for Gershgorin components is now one clause with
  a citation; the review wanted it as an exercise.
- Hessians at a critical point still have no `:numref:` into the Calculus or
  Optimization appendix (I did not look for the right label).
- Caption of :numref:`fig_mdl-la-transient-growth` (right panel) not
  shortened: figure includes were left alone.

## Claims added or changed

- Cofactor expansion along the first column gives $(\lambda-t)^g\det(\mathbf
  Y-t\mathbf I)$ (checked by hand).
- "A matrix therefore has $\sum_\lambda g(\lambda)$ independent eigenvectors
  and no more."
- Every real square matrix is built from stretch, shear and rotation with
  scaling, "alone or combined" (the real Jordan form; "combined" covers a
  defective complex pair).
- The Marchenko-Pastur statement as above (density; fourth moment for the
  edges; at least $d-n$ zeros).
- Cholesky "is cheaper than computing the eigenvalues" (replaces the
  operation counts $n^3/3$ and $4n^3/3$).
- $L$ of :numref:`sec_mdl-gradient-based-optimization` is $\lambda_{\max}$
  for the quadratic; slowest rate $1-\eta\lambda_{\min}$ for
  $\eta\le1/\lambda_{\max}$.
- $\|\mathbf A\|_2/\sqrt n\to2$ is the Marchenko-Pastur upper edge at $c=1$:
  $\|\mathbf A\|_2^2=n\,\lambda_{\max}(\tfrac1n\mathbf A^\top\mathbf A)\to4n$
  (simulation at $n=1000$: 1.99).
- $\mu<0$ by Jensen: for a unit vector $\mathbb E\log\|\mathbf A\mathbf
  x\|^2<\log\mathbb E\|\mathbf A\mathbf x\|^2=0$.
- PageRank: "fifty sweeps reduce the error by a factor of more than 3000
  ($0.85^{50}\approx3\times10^{-4}$) on a graph of any size"
  ($0.85^{50}=2.96\times10^{-4}$), and "this bound is what made the
  computation feasible for the web". The second is a judgment and may be
  too strong; the reviewer proposed it (EIG-22).
- "The theorem is nearly sharp here" (Gershgorin: distance 0.96 in a disc of
  radius 1).
- The Lyapunov limit "exists with probability one under mild conditions"
  (shortened from the Furstenberg-Kesten fine print, EIG-18).
- Deep-learning uses named in the ranking paragraph: spectral normalization
  (`subsec_mdl-spectrum-control`, which does cite this section for power
  iteration), `sec_bptt`, `sec_numerical_stability`; gates and clipping now
  cite `subsec_bptt-gradient-pathologies`.
- "Its proof follows the induction of :numref:`subsec_mdl-spectral-theorem`"
  for the complex spectral theorem; the one extra step ($\mathbf A^*\mathbf
  v=\bar\lambda\mathbf v$) is no longer written out.

## Slides against the body

No factual disagreement. Three wordings differ: the slide "Pure-Noise
Covariances Fill [0.25, 2.25]" keeps "fill" (EIG-13); the Jacobian slide
says "Orthogonal init sets every singular value to 1" where the body now
avoids singular values (EIG-17); the Gershgorin slide has no counterpart to
"can be skipped". The slides have no slide on pseudospectra or random
matrices, and nothing on the split.
