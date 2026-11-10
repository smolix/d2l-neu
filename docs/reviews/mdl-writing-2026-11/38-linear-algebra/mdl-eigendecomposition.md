# Eigendecompositions

Source: `chapter_mdl-linear-algebra/mdl-eigendecomposition.md`. About 9,600 words of prose and mathematics to the Summary, plus 2,200 words of exercises (11,800 to the slides marker, code excluded). Four `##` parts, 20 `###` subsections, 14 code cells, 8 figures.

## Verdict

A good section, and in its first half a very good one. It opens on a question the reader can compute with (a surfer between two pages, four iterates, "where is she after 50 clicks?"), answers it in closed form 230 lines later, and keeps returning to the same three small matrices. Every number I recomputed is right, every proof I checked is correct, and every stored output matches the prose. The examples in the second half are the best in the chapter: a Jordan block with eigenvalue 0.99 whose powers climb to 37, a unit vector that is the difference of two eigen-components of length 40, two nilpotent Jacobians whose product has spectral radius 4.

What keeps it from the bar is that it is two sections in one binding. The first (eigenvalues, diagonalization, the spectral theorem, quadratic forms) is the mathematics the rest of the book cites. The second (Gershgorin, PageRank, normality, transient growth, pseudospectra, Lyapunov exponents, three random-matrix laws) is a tour of everything else an expert knows about spectra, at the same level of detail and with no ranking. A reader who arrives from a forward reference to the spectral theorem has to find it at line 685 of 1,630, between the real Jordan form and the Marchenko-Pastur law. The single most important fix is to decide what the core is, say so, and cut or demote the rest: pseudospectra to Further Reading (which already has them), the real Jordan form to a sentence, and the random-matrix material gathered in one place.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 3 | 4 | 4 |

Pedagogy is 3 for the structural problem alone; the local teaching is at 4 or better.

## What works

- **The opening problem is carried to the end.** "Where is the surfer after 50 clicks, and how fast does the answer settle?" The iterates $(0.5,0.5)$, $(0.375,0.625)$, $(0.344,0.656)$ are given before any definition; the eigenvectors are found by inspection, then by the characteristic polynomial, then by `eig`; the closed form :eqref:`eq_mdl-markov-closed-form` reproduces the third iterate ("$(\tfrac13+\tfrac1{96},\ \tfrac23-\tfrac1{96})=(0.34375,0.65625)$, the third iterate of the opening"). This fixes the October finding that the section stated $\mathbf A^n=\mathbf W\boldsymbol\Lambda^n\mathbf W^{-1}$ with no worked recurrence. Protect all of it.
- **Predictions before cells.** "For $\lambda=0.99$ this puts the peak near $k\approx100$, where the corner entry is $100\cdot0.99^{99}\approx37$. The cell checks the prediction." The output then prints 36.977. The same method for $\mathbf M$: peak predicted at $k\approx6.4$ with value 10.8, cell prints 10.788 at $k=6$. This is the book's best habit and it is used consistently here.
- **The cancellation explanation of transient growth.** "A vector of length 1 is the difference of two components of length about 40. Both decay, but at different rates, so their difference first grows." With :numref:`fig_mdl-la-transient-cancel` this is the clearest account of non-normal growth I know at this level. The honest remark on the bound ("The bound holds, but it is four times the actual value 10.8") is in the spirit of Cover and Thomas.
- **Ellipse axes against eigenvectors.** "The ellipse axes are the directions of largest and smallest stretch; the eigenvectors are the directions that are not turned." One sentence that removes the most common confusion in the subject, placed at the definition, with a figure that shows both cases.
- **The norm ratios of the running matrix.** The reader is told the matrix "scales by $\sqrt5$" and then shown that consecutive norm ratios "swing between about 1.4 and 3.5 instead of staying at 2.24", with the reason (the coordinates $(\mathbf v,\mathbf u)$ are not orthonormal). An objection raised and answered.
- **The proofs are the short ones.** Similar matrices (three lines), distinct eigenvalues (minimal dependence relation), the spectral theorem in three labelled parts with the borrowed fact named ("The only fact we borrow is that the characteristic polynomial has at least one complex root"), Gershgorin (largest entry of the eigenvector), Rayleigh ("an average of the eigenvalues with non-negative weights $y_i^2$ that sum to 1"). Each is led by an example on $\mathbf S$ or $\mathbf P$.
- **The Jacobian counterexample.** "Both factors are nilpotent with $\rho=0$, and their product has $\rho=4$". Two $2\times2$ matrices settle the question of whether spectral radii of factors control a product.
- **Candour.** "which we use without proof" (Jordan form, fundamental theorem of algebra); "The condition is sufficient, not necessary: $f(x)=x^4$ has a strict minimum at 0 with Hessian 0."
- **Code.** No cell draws. Every cell computes a number the prose predicted or reads.

## Structural findings

- **Two sections in one (major).** Parts 1 and 2 ("Eigenvalues and Diagonalization", "Symmetric Matrices and Quadratic Forms", about 5,300 words) are the core. Parts 3 and 4 ("Locating and Computing Eigenvalues", "Iteration and Stability", about 4,300 words) are a second section whose subject is "what eigenvalues do not tell you, and how to compute them". Either split the file there (the second section would open naturally with the shear and $\mathbf J_2(0.99)$ and end at random initialization), or keep one file and cut about 2,000 words: Pseudospectra (EIG-16), the real Jordan form (EIG-10), the fine print on Lyapunov exponents (EIG-18), the continuity argument for Gershgorin components (to an exercise), and the Cholesky operation counts. Of the two I would cut. The October review counted "about 3,500 words of pathologies against 1,500 for the diagonalizable case"; the diagonalizable case is now well served, and the pathologies have grown.
- **Random-matrix facts are scattered and not connected (moderate).** Marchenko-Pastur sits under "Symmetric Matrices" (line 1000); the circular law, $\rho/\sqrt n\to1$ and $\sigma_{\max}/\sqrt n\to2$ sit 560 lines later; the SVD section then uses Marchenko-Pastur twice more. The reader is never told that $\sigma_{\max}\to2\sqrt n$ is the Marchenko-Pastur upper edge at $c=1$ (EIG-19). Put the three laws in one subsection, "Random Matrices", at the end, and let the Rayleigh subsection end on gradient descent.
- **Nothing is ranked (moderate, voice).** The spectral theorem is the result on which positive definiteness, the Rayleigh quotient, PCA and the whole SVD section rest. It is introduced with "The spectral theorem states that this holds for every symmetric matrix", in the same register as the real Jordan form and the Kreiss constant. Writing guide 10.2 asks the authors to say what matters. Say here that this is the theorem of the chapter and that a reader in a hurry can go from :numref:`subsec_mdl-diagonalization` straight to it; say of the Jordan form that only its consequence (a polynomial factor in $k$) is needed; say of pseudospectra that they are a diagnostic few readers will compute.
- **A visible template around code (minor, pattern).** Seventeen question marks in the body, eleven of them in the form question, "The cell ...", answer: "Does the library agree? The cell computes", "Does the closed form reproduce 50 matrix multiplications? The cell compares", "Does the rank condition for Gram matrices show up in their eigenvalues?", "Which matrices are safe?", "How large are these quantities for the random weight matrices". "The cell" opens or closes a sentence 11 times. Each instance is fine; the eleventh is a drone. Vary: sometimes state the prediction and let the cell follow without a question.
- **Closing paragraph and Summary say the same thing (moderate).** See EIG-21. The closing paragraph begins "We can now", as do the closing paragraphs of all four sections of this chapter (`mdl-geometry-linear-algebraic-ops.md:1065`, `mdl-linear-maps.md:1169`, `mdl-eigendecomposition.md:1601`, `mdl-svd-low-rank.md:1446`). Writing guide 12.2 names this exact formula as one to avoid.
- **Level against Basics.** :numref:`sec_linear-algebra` teaches the definition, the symmetric $\mathbf S$, the expansion $\sum_i c_i\lambda_i^k\mathbf v_i$ for symmetric matrices and power iteration, without proofs. This section restates the definition in ten lines, says what is new ("This section drops the symmetry"), and reuses $\mathbf S$ and the iterates $(2,1),(5,4),(14,13)$ instead of re-deriving them. That is the right relation. The level inside the section is not even: it runs from "factor out the vector" to Furstenberg-Kesten within one file.
- **Connection to deep learning.** Named and referenced for iteration (deep linear network, gradient descent, RNN gradients, initialization, PageRank). Missing or thin for Gershgorin (no use named), pseudospectra (none), the matrix exponential (EIG-7), and the Jordan form (its only use is the polynomial factor). Hessians at a critical point and gradient-descent step size are named but with no `:numref:` into the Optimization appendix.

## Passage-level findings

**EIG-1** (minor, language). "Eigenvectors and Eigenvalues".

> Its eigenvalues are complex, and we return to them at the end of this part.

"This part" means the `##` part, but the book has Parts, and this file sits in one. Write "in :numref:`subsec_mdl-complex-rotation`".

**EIG-2** (moderate, explanation). "Eigenvectors and Eigenvalues".

> the ellipse axes point at $58^\circ$ and $-32^\circ$. The ellipse axes are the

The angles are asserted. Nothing in this section can produce them; they are the left singular vectors, defined one section later. October noted "singular vectors are used in the section that defines eigenvectors"; the word is now avoided but the numbers still come from nowhere. Add a clause: "(they are computed in :numref:`sec_mdl-svd-low-rank`)". A curiosity worth one more clause or an exercise: $\mathbf A\mathbf A^\top$ for this matrix is twice the Gram matrix $\left(\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right)$ of the shear, which is why the same $58.3^\circ=\arctan\varphi$ reappears there. A reader who notices the repeated angle will otherwise wonder.

**EIG-3** (moderate, language/notation). "Algebraic and Geometric Multiplicity" and "Complex Eigenvalues".

> The chapter's running $2\times2$ matrix has no visible rotation in its

Bold $\mathbf A$ is the generic matrix (definition), then the shear (:eqref:`eq_mdl-defective-shear`, "$\mathbf A^k=\ldots$ grows without bound"), then 100 lines later "the running matrix $\mathbf A$", then the $3\times3$ Gershgorin example, then a random matrix. The chapter introduction promises two recurring matrices; give each its own letter for the whole chapter (the SVD section has the same clash) and keep $\mathbf A$ generic. The same holds, less seriously, for $\mathbf P$: the $2\times2$ surfer for 250 lines, then "Here $\mathbf P$ is the repaired link matrix" for a $3\times3$ one, and $\mu$: an eigenvalue in the spectral theorem, a real number in the complex-eigenvalue argument, the Lyapunov exponent, and a shift in the exercises.

**EIG-4** (minor, explanation). "Diagonalization", trace identity.

> $\det\mathbf A=\prod_i\lambda_i$. On the right, the coefficient of $\lambda^{n-1}$ is

"On the right" and "On the left" refer to an equation that is not displayed; the factorization is inline two sentences earlier and its other side, $\det(\mathbf A-\lambda\mathbf I)$, is not on the page at all. Display $\det(\mathbf A-\lambda\mathbf I)=\prod_{i=1}^n(\lambda_i-\lambda)$ and then compare coefficients.

**EIG-5** (minor, correctness, confident). "Diagonalization".

> Similar matrices therefore have the same eigenvalues with the same

The proposition gives equal algebraic multiplicities. Geometric multiplicities also agree, but by a different (one-line) argument, and the two notions are only defined in the next subsection. Write "with the same multiplicities as roots".

**EIG-6** (moderate, correctness: unjustified step, confident). "Algebraic and Geometric Multiplicity", proof of $g\le a$.

> By :eqref:`eq_mdl-similar-charpoly`, $p(t)=(\lambda-t)^g\det(\mathbf Y-t\mathbf I)$,

The step needs the determinant of a block upper-triangular matrix, $\det\left(\begin{smallmatrix}\mathbf X_{11}&\mathbf X_{12}\\\mathbf 0&\mathbf X_{22}\end{smallmatrix}\right)=\det\mathbf X_{11}\det\mathbf X_{22}$. :numref:`sec_mdl-linear-maps` proves the triangular rule only; the block rule appears there in one exercise for a special case. State the block rule in the determinant section next to the triangular rule (it follows from the permutation formula in the same way: a permutation contributes a nonzero term only if it maps the first $g$ indices among themselves) and cite it here. The proof also changes the variable of $p$ from $\lambda$ to $t$ without comment, because $\lambda$ is now the fixed eigenvalue; say so.

**EIG-7** (moderate, thought/order). "Algebraic and Geometric Multiplicity".

> independent (the next proposition, applied to the components of a dependence

The diagonalizability criterion is derived from a proposition that has not yet been stated, and the step that carries the weight (bases of different eigenspaces are jointly independent) is compressed into a parenthesis that a first-time reader cannot unpack. Move "Proposition (distinct eigenvalues give independent eigenvectors)" before this paragraph, and then spell out the one sentence: "if a combination of the chosen basis vectors vanishes, group it by eigenvalue; each group is zero or an eigenvector, and the proposition forces every group to be zero."

**EIG-8** (minor, pedagogy/code). "Complex Eigenvalues".

> argument $\pm\theta$ records the angle. The cell confirms this for

The cell confirms that NumPy agrees $\cos\theta\pm i\sin\theta=e^{\pm i\theta}$ at one angle. Nothing can go wrong and nothing is learned. Drop it; the next cell (the running matrix, the block, the norm ratios) does real work.

**EIG-9** (minor, code: claims the outputs do not show; all verified). Three places.

> $\|\mathbf A^k\mathbf x\|^{1/k}$ settles at $\sqrt5$ (it is 2.25 at $k=20$).

> rounding level. With Gaussian entries instead, about four draws in ten have a

> digit, on every run. For such matrices the gap ratio $|\lambda_2/\lambda_1|$

None of the three is printed by a cell. I recomputed them: 2.252 at $k=20$; 43% of $5\times5$ Gaussian matrices have a complex dominant pair (20,000 draws); the gap ratio of uniform $5\times5$ matrices has 5% and 95% quantiles 0.14 and 0.37. They are true as written. When cells are next touched, add one printed line for the first; the second would make a good two-line exercise.

**EIG-10** (moderate, pedagogy/scope). "Complex Eigenvalues", last paragraph.

> over $\mathbb R$, to a block-diagonal matrix built from four kinds of blocks.

A taxonomy of the real Jordan form with a $4\times4$ example of a defective complex pair. Nothing later uses it, and it comes straight after the section's most instructive computation (the norm ratios). Reduce to one sentence: "Every real square matrix is built, up to a real change of basis, from the three behaviours we have seen: stretch along a line, shear within a Jordan block, and rotation with scaling in a plane :cite:`Horn.Johnson.2012`." Keep the last sentence on continuous time only if EIG-11 is resolved by a reference.

**EIG-11** (minor, pedagogy). "Diagonalization".

> The same recipe handles other functions of a diagonalizable matrix. If no

Eight lines introduce the matrix exponential, linear differential equations and their stability criterion, with no example and no reference to where the book uses them. Give the pointer by label (the Dynamics chapter, :numref:`chap_mdl-dynamics`, or the state space models chapter), or cut to the inverse alone.

**EIG-12** (minor, explanation). "The Rayleigh Quotient", last paragraph.

> The largest eigenvalue is the curvature constant $L$ that limits the step

$L$ has not been defined in this chapter. Either "the constant called $L$ in :numref:`sec_mdl-gradient-based-optimization`" or drop the letter. The paragraph also defines the condition number ("Their ratio $\kappa=\lambda_{\max}/\lambda_{\min}$, the *condition number*") which :numref:`subsec_mdl-condition-number` defines again, differently worded, 800 lines later, with the same bowl and the same gradient-descent consequence. Per writing guide 7.7 give it one home (the SVD section) and one clause here.

**EIG-13** (minor, correctness: hypothesis, moderately sure). "Sample Covariance and the Marchenko-Pastur Law".

> $c=d/n\le1$ fixed. Then the eigenvalues of $\mathbf C$ fill the interval

Stated for "independent coordinates of mean 0 and variance 1". The limiting density needs only that; the statement the cell tests, that the smallest and largest eigenvalues converge to the edges, needs a finite fourth moment (Bai and Yin). "Fill" is also loose for a statement about an empirical distribution. Suggested: "the histogram of the eigenvalues converges to a fixed density supported on ..., and for entries with a finite fourth moment the extreme eigenvalues converge to its edges". The clause "with a density that the theorem also gives" tells the reader nothing; give the density or drop the clause. Also "A sample covariance is such a Gram matrix divided by the number of samples" (in "Positive Definiteness") holds for centred data; say so, since this subsection then uses uncentred $\tfrac1n\mathbf X^\top\mathbf X$ with true mean zero.

**EIG-14** (minor, correctness: wording, confident). "Gershgorin Discs", corollary.

> diagonally dominant matrices are also the ones on which Gaussian elimination

"The ones on which" reads as a characterization. Strict diagonal dominance is sufficient for elimination without row exchanges and for convergence of Jacobi, not necessary. Write "On such matrices Gaussian elimination needs no row exchanges and the Jacobi iteration converges."

**EIG-15** (minor, language, pattern: 3 instances). 

> Exercise :exref:`ex_mdl-eigendecomposition-spectral-radius-norms` proves the

> :exref:`ex_mdl-eigendecomposition-pseudospectra-transient-growth` proves that

> (Exercise :exref:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`

An exercise does not prove anything; the reader does. "is the subject of", "asks for the proof". The third is worse than wording: the Courant-Fischer theorem is handed to part (d) of a code exercise titled "Accuracy of computed eigenvalues", so a reader who follows the reference lands on `np.linalg.eigvals` and backward stability. Make Courant-Fischer and Weyl an exercise of their own (see Exercises).

**EIG-16** (moderate, pedagogy/scope). "Pseudospectra".

> A second symptom of non-normality is that the eigenvalues are sensitive to

The subsection defines the pseudospectrum and the resolvent, gives two equivalent descriptions whose equivalence is left to an exercise, and runs a two-line cell. No use in deep learning is named, and the reader has already been convinced by $\mathbf M$ that eigenvalues mislead. `further-reading.md` has a topic "Pseudospectra and the Kreiss Matrix Theorem" that depends on this subsection. Recommendation: keep two sentences at the end of "Transient Growth" (the number 200.3 against 10 is a good one: "a perturbation of size 0.005 puts an eigenvalue of $\mathbf M$ on the unit circle, where $\boldsymbol\Lambda$ needs 0.1"), move the definition into the Further Reading topic, and let the exercise define what it needs. The right panel of :numref:`fig_mdl-la-transient-growth` can stay with a shortened caption.

**EIG-17** (moderate, explanation). "Products of Jacobians" and "Random Matrices at Initialization".

> singular value of the initial weight matrix equal to 1, so a linear recurrence

> $\rho(\mathbf A)/\sqrt n\to1$ with probability one, is a separate theorem

Singular values are used here and in "The largest singular value behaves differently" before they are defined (the only definition so far is inside exercise 2c). The spectral norm was defined 200 lines earlier and orthogonal matrices in the previous section, so nothing new is needed: "Orthogonal initialization makes the initial weight matrix orthogonal, so a linear recurrence preserves the length of every vector", and "The spectral norm behaves differently: $\|\mathbf A\|_2/\sqrt n\to2$". Still open from October.

**EIG-18** (minor, pedagogy). "Products of Jacobians".

> factors it equals $\mu$ for every fixed $\mathbf x_0$. In general one reads the

Three sentences of conditions (Furstenberg-Kesten with the matrix norm, Gaussian factors, limsup) for a quantity the section uses in one sentence. The reader needs: "For random factors the limit exists with probability one under mild conditions :cite:`Furstenberg.Kesten.1960`." The rest belongs to the exercise, which restates it anyway.

**EIG-19** (moderate, thought). "Random Matrices at Initialization".

> $\sigma_{\max}(\mathbf A)/\sqrt n\to2$ :cite:`Yin.Bai.Krishnaiah.1988`. The

Presented as a new fact with a new citation. It is the upper Marchenko-Pastur edge of :numref:`subsec_mdl-sample-covariance-mp` at $c=1$: the largest eigenvalue of $\tfrac1n\mathbf A^\top\mathbf A$ tends to $(1+\sqrt1)^2=4$, and $\|\mathbf A\|_2^2$ is $n$ times that. One clause makes the section's two random-matrix passages one idea, and it explains the factor 2 instead of reporting it. The following sentence, "a random matrix stretches some vectors twice as much as its eigenvalues suggest", is the kind of sentence the section needs more of.

**EIG-20** (minor, explanation). "Random Matrices at Initialization".

> A spectral radius near 1 is a side effect, and by

The paragraph makes three claims in five lines: fan-in scaling preserves expected squared length; a spectral radius near 1 follows but is beside the point; products nevertheless shrink, $\mu<0$. The third is surprising and is given no reason, only an exercise. Add the reason in one sentence, since it is short and memorable: "the logarithm is concave, so the average of $\log\|\mathbf D\mathbf x\|$ is below the logarithm of the average, and it is the logarithms that add along a product."

**EIG-21** (moderate, language/structure). End of "Random Matrices at Initialization".

> We can now compute the eigenvalues of a small matrix by hand and write

A recap paragraph under the last subsection heading (where it does not belong), in the "We can now" formula, followed immediately by a Summary that restates the same results at twice the length. The Summary itself is a list of everything mentioned, 11 sentences, including a worked number ("approaches $(1/3,2/3)$ with an error that shrinks by a factor of 4"). Per writing guide 12.2: delete the first half of the closing paragraph, keep its two open questions as the last sentences of the body ("Eigenvalues say little about finite-step stretch ... And eigenvalues do not exist for rectangular matrices at all"), and cut the Summary to four or five sentences that answer what, why, how and where.

**EIG-22** (minor, enjoyment). "PageRank".

> $\boldsymbol\pi=(0.3,0.4,0.3)$. Page 2 ranks first because page 1 sends it

The three-page example is worked by hand and then the damped version is quoted ($-0.567$, $(0.303,0.394,0.303)$) with no cell; the check is exercise 10e. That is acceptable, but this is the one subsection where the opening question of the section (the surfer) reaches its real application, and it is told flatly. One sentence of judgment would help: that the damping factor 0.85 is the whole reason PageRank could be computed for the web in a few dozen sweeps, because it bounds the second eigenvalue whatever the link structure.

## Exercises

Twelve problems, inside the 8 to 12 band, about 2,200 words. The quality is high and several are better than anything in the body of a standard text: the defective limit ($\sqrt\varepsilon\,\kappa_2(\mathbf V_\varepsilon)\to1$, an eigenvalue moving by $10^{-5}$ under a perturbation of $10^{-10}$), spikes above the bulk (the detection threshold $\beta>\sqrt c$, with the nice trap at $n=4000$ where prediction and edge differ by 0.005), and Lyapunov exponents ($\mu\approx-1/(2n)$ although each factor preserves squared length on average). I checked the stated numbers of exercises 3, 5, 7a, 9, 10 and 11; all are right.

- **Balance.** Seven of twelve are `[code]` with a difficulty mark, and six of those run to 150 words or more. Exercises 1 and 2 are the only ones a reader fresh from the definition can do. Add one or two short ones on the core: diagonalize a $2\times2$ non-symmetric matrix and write $\mathbf A^k\mathbf x$ (Fibonacci is the classic and is absent); find a matrix with given eigenvalues and eigenvectors; show that $\mathbf A$ and $\mathbf A^\top$ have the same eigenvalues but not the same eigenvectors, using $\mathbf P$.
- **Split exercise 7.** Parts (a) to (c) are about computed accuracy; part (d) is Courant-Fischer and Weyl, which the body cites as if it were proved. Make (d) its own exercise, "Courant-Fischer and Weyl", and refer to it from :numref:`subsec_mdl-rayleigh` and from the Weyl lemma for singular values in :numref:`subsec_mdl-eckart-young`, which is the same inequality.
- **Exercise 8** (Kreiss constant) and **exercise 9** (shifted inverse iteration, Rayleigh quotient iteration) are numerical-analysis problems. If Pseudospectra moves to Further Reading (EIG-16), keep 8a and 8b with the definition stated in the exercise, and drop 8c. Exercise 9 is good but long; part (d) is the memorable one (the shift 4 is equidistant from $3.99\pm0.96i$, so the iteration never converges; I confirmed the rate 0.02 for the shift $4+i$).
- **Exercise 6** carries a `[code]` tag though (a) and (b) are pencil work; (b) is what the body cites for the squared rate. Fine as is.

## Still open from October

"No starting vector, worked recurrence, or picture" is fixed, and fixed well. "About 3,500 words of pathologies against 1,500 for the diagonalizable case" is half open: the diagonalizable case now has its due, but the pathologies are still there in full (see the first structural finding). "Singular vectors are used in the section that defines eigenvectors" is mostly open (EIG-2, EIG-17).

## For the chapter review

- The section assumes the determinant and its multiplicativity, rank and null space from :numref:`sec_mdl-linear-maps`, and orthogonal matrices; it needs, and does not get from there, the block-triangular determinant (EIG-6). It hands the SVD section the spectral theorem, positive semidefiniteness of $\mathbf A^\top\mathbf A$, the Rayleigh proposition, the Marchenko-Pastur law, power iteration and the spectral norm. Order within the chapter is right.
- Overlap with the SVD section: the condition number and the elongated bowl are taught in both (EIG-12); Marchenko-Pastur is used three times across the two; Weyl's inequality appears as an exercise here and a lemma there with no cross-reference.
- Chapter-wide patterns I can confirm from all four sections' closing paragraphs: each ends with a "We can now ... [limits/questions remain]" recap followed by a Summary that lists everything. Patterns I suspect hold chapter-wide: bold $\mathbf A$ reused for the generic matrix and both running examples; the question, "The cell ...", answer template around every code cell; no sentence that tells the reader which results are central.
- This is the longest section of the chapter and the one where a lookup reader is worst served. If only one structural change is made in chapter 38, it should be here.
- Mathematics: no errors found. Five small precision or justification points (EIG-5, 6, 13, 14, and the forward-used proposition in EIG-7).
