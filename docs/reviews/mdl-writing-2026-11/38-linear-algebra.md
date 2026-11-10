# Chapter 38, Linear Algebra: writing review

Sources: the six `.md` files of `chapter_mdl-linear-algebra/`, each read in full to the slides marker by a section reviewer. About 36,000 words of prose and mathematics (index 830, geometry 7,000, linear maps 8,400, eigendecompositions 9,600, SVD 8,200, further reading 1,800) and about 8,100 words of exercises (42 problems). This file is assembled from the section reviews. I opened the sources only to check the cross-section claims listed under "Checked in the sources" at the end.

Section reviews:
[index](38-linear-algebra/index.md) ·
[mdl-geometry-linear-algebraic-ops](38-linear-algebra/mdl-geometry-linear-algebraic-ops.md) ·
[mdl-linear-maps](38-linear-algebra/mdl-linear-maps.md) ·
[mdl-eigendecomposition](38-linear-algebra/mdl-eigendecomposition.md) ·
[mdl-svd-low-rank](38-linear-algebra/mdl-svd-low-rank.md) ·
[further-reading](38-linear-algebra/further-reading.md)

## Verdict

This is a good chapter, close to the bar in several long stretches and at it in a few (Eckart-Young, the condition number, the cancellation account of transient growth, the measured cosines of real images). Two reviewers recomputed the worked numbers and the stored outputs of all four sections and found no false statement of mathematics and no stored output that contradicts the prose. What the chapter gets right is the thing the model texts get right: a small number of concrete objects (a $2\times2$ matrix, a shear, a two-page surfer, one photograph) are carried through, a number comes before every statement, proofs are the short ones, and bounds are compared with what actually happens. What keeps it from the bar is organization and economy, not substance. The chapter is about a quarter longer than its argument needs; the same facts are taught two, three or four times (the condition number three times, "the Gram matrix squares the condition number" five times, a layer as a stack of hyperplanes four times); two sections order their material by the logic of the proofs or by accumulation and not by what a reader needs (the second half of linear maps, the second half of eigendecompositions); nothing tells the reader which results are central; and every section ends twice, with a "We can now ..." recap followed by a Summary that lists everything. What matters most: decide what the core of the eigendecomposition section is and demote the rest, prove invertibility where it is claimed in linear maps, and give each repeated fact one home.

## Scores

As the section reviewers gave them; 1 to 5, where 5 means the page would be at home in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|:-:|:-:|:-:|:-:|:-:|
| [index](38-linear-algebra/index.md) | 4 | 4 | 4 | 4 | 4 |
| [Vectors, Angles, and Projections](38-linear-algebra/mdl-geometry-linear-algebraic-ops.md) | 4 | 4 | 4 | 4 | 4 |
| [Matrices as Linear Maps](38-linear-algebra/mdl-linear-maps.md) | 4 | 3 | 3 | 4 | 3 |
| [Eigendecompositions](38-linear-algebra/mdl-eigendecomposition.md) | 4 | 4 | 3 | 4 | 4 |
| [SVD and Low-Rank Approximation](38-linear-algebra/mdl-svd-low-rank.md) | 4 | 4 | 4 | 4 | 4 |
| [Further Reading](38-linear-algebra/further-reading.md) | 4 | 4 | 4 | 4 | 3 |

Two reviewers wrote these: one covered the index, geometry and linear maps, the other eigendecompositions, SVD and further reading. Their calibration looks the same (both use 4 as "good textbook prose with local flaws" and reserve 3 for a structural problem). Qualifiers they attached: the geometry section's first half is a 3 on enjoyment and its second half earns the 4; the eigendecomposition section's 3 on pedagogy is "for the structural problem alone"; the Eckart-Young and condition-number subsections of the SVD section "taken alone, are at 5".

## Correctness

No section reviewer found an error of mathematics, a false proposition, or a stored output that contradicts the prose. The findings with dimension `correctness` are gaps of order, missing hypotheses, and loose wording. In order of severity:

1. **MAP-14** (moderate; sure). In "The Determinant in General" the sharper uniqueness statement is proved "by uniqueness" a page before uniqueness is derived; existence of a function with the three properties is asserted in one sentence, with no proof and no note that the proof is omitted; "The proof rests on one fact" does not say which proof.
2. **GEO-8** (moderate; sure). The projection theorem assumes an orthonormal basis of $S$ and is applied at once to the column space of an arbitrary $\mathbf A$, including dependent columns. That every subspace has an orthonormal basis is shown only in the next subsection, and Gram-Schmidt as described there requires independence.
3. **EIG-6** (moderate; confident). The proof that geometric multiplicity is at most algebraic uses the determinant of a block upper-triangular matrix, which the determinant section never states (it proves the triangular rule only). The proof also switches the variable of $p$ from $\lambda$ to $t$ without comment.
4. **FR-3** (moderate; fairly sure). "LoRA starts $\mathbf B\mathbf A$ at zero, so early fine-tuning moves along random directions" is imprecise: $\mathbf B=\mathbf 0$ and $\mathbf A$ is random, so the first updates are confined to a random $k$-dimensional input subspace.
5. **MAP-6** (minor; sure). Rank-nullity uses that an independent list extends to a basis, and the last proposition uses that an $n$-dimensional subspace of $\mathbb R^n$ is all of $\mathbb R^n$. Neither is proved or declared as assumed; the geometry section declares only that all bases have the same size.
6. **EIG-13** (minor; moderately sure). Marchenko-Pastur is stated for mean 0 and variance 1; the convergence of the extreme eigenvalues to the edges, which is what the cell tests, needs a finite fourth moment. "Fill the interval" is loose, and "a sample covariance is such a Gram matrix divided by $n$" holds for centred data only.
7. **SVD-9** (minor; confident). "Weight decay ... is ridge regression for neural networks" is true only for a linear model with squared loss under plain gradient descent.
8. **FR-4** (minor; confident). The mixing-time paragraph defines $\lambda_\star$ as the largest modulus other than 1 and then says Cheeger bounds "the gap $1-\lambda_2$"; these differ for a nearly periodic chain.
9. **EIG-5** (minor; confident). "Similar matrices have the same eigenvalues with the same multiplicities": the proposition gives algebraic multiplicities; geometric ones need a separate line and are defined only in the next subsection.
10. **EIG-14** (minor; confident). "Diagonally dominant matrices are also the ones on which Gaussian elimination ..." reads as a characterization; dominance is sufficient, not necessary.
11. **GEO-16** (minor; sure). A hyperplane with $b\ne0$ is given "dimension $d-1$" although dimension is defined only for subspaces, and the count is not justified (it is rank-nullity, one section later).

Three smaller points sit inside findings filed under other dimensions: the sign in cofactor expansion uses the row-swap rule two paragraphs before it is established (MAP-15, sure); "every pair has a positive dot product" should be "non-negative" (GEO-14, sure); and the diagonalizability criterion is derived from a proposition not yet stated (EIG-7).

**Stored outputs and numbers.** No output contradicts the prose. The following are claims the outputs do not carry, all recomputed by the reviewers and found true:

- EIG-9: three numbers no cell prints (2.25 at $k=20$; "about four draws in ten" have a complex dominant pair; the gap ratio "typically between 0.1 and 0.4").
- SVD-8: three numbers no cell prints (rank 4 keeps 95% at 21% error; rank 8 after centring; decay as $i^{-1.1}$).
- GEO-6: the `nan` from `arccos` on a collinear pair is described in prose and not shown.
- GEO-17: the explanation of the harder class pairs (means close, spread wide) is asserted and not computed, though the previous cell computes exactly those quantities for the first pair.
- EIG-22: the damped PageRank values are quoted with no cell (the check is an exercise).
- MAP-11: the prose first says the condition number "does not predict" the loss and four sentences later that the loss is "as $\kappa$ predicts"; "whichever method we use" flattens a factor of 30 between the two errors that the output shows.

Cells that cannot fail or that confirm their own construction: EIG-8 (NumPy agrees with Euler's formula at one angle), SVD-14 (the 95% rank of a matrix built with singular values $e^{-i/12}$ is $6\ln 20$), SVD-4 (three shapes and `True`), and nine of the thirteen cells of the linear-maps section, which recompute a $2\times2$ or $3\times3$ result printed in the paragraph above (linear maps, structural finding 4).

## Arc and order

**One line of argument: yes.** The introduction states it in a sentence ("a matrix becomes simple in the right directions") and the four sections deliver it in the right order: the geometry of one pair of vectors, what a map does to all of space, what repetition of a map does, and the decomposition that works for every matrix. No reviewer wants the order changed. The dependencies are cited by label and are real: the SVD section rests on the spectral theorem, the Rayleigh proposition, rank-nullity and least squares, each from the section that should own it.

**Hand-overs.** The geometry section ends on the question the next one answers ("which inputs the map destroys, or whether its action can be undone"); the linear-maps section ends with the running matrix's eigenvalues and singular values, both with product 5; the eigendecomposition section ends on what eigenvalues cannot say; the SVD section ends on the circle of rank 2 and the next chapter. These are the chapter's structural strength. The first hand-over is told twice, at the end of one section and at the start of the next.

**Sections that are really two.**

- *Eigendecompositions* is two sections in one binding (first structural finding of that review). Parts 1 and 2 (eigenvalues, diagonalization, the spectral theorem, quadratic forms; about 5,300 words) are what the rest of the book cites. Parts 3 and 4 (Gershgorin, PageRank, normality, transient growth, pseudospectra, Lyapunov exponents, random matrices; about 4,300 words) are a tour at the same level of detail with no ranking. A lookup reader finds the spectral theorem at line 685 of 1,630. The reviewer prefers cutting about 2,000 words to splitting (EIG-10, EIG-16, EIG-18).
- *Vectors, Angles, and Projections* holds three topics under a title that names one: the subspace vocabulary of the whole chapter, least squares with QR, and similarity and hyperplanes in high dimensions. The reviewer does not require a split, but names the clean cut (before "Similarity in High Dimensions") and asks that the title or opening name subspaces, least squares and hyperplanes (structural finding 3 there).
- *Matrices as Linear Maps* should not be split, but its second half needs reordering (below).
- *SVD*: not two sections, but "Existence and Uniqueness" is a container for six topics and should become three subsections, and "The SVD in Deep Learning" can shrink by half.

**Order inside sections.** The recurring structural defect of the chapter is that pieces of an argument arrive in an order that leaves a temporary hole. In linear maps there are nine forward references to a later subsection of the same section for a proof or definition; $\mathbf Q\mathbf Q^\top=\mathbf I$ is promised in one subsection, half proved in a second and completed in a third (MAP-9). The same shape recurs in GEO-8 (orthonormal basis used before it is built), MAP-14 (uniqueness used before it is derived), EIG-7 (a proposition used before it is stated), and SVD-7 (the reader is not told the Eckart-Young proof is half done). The determinant story is interrupted by a `##` on `einsum` and ends in another `##` (MAP-17, MAP-18).

**Boundaries that leak.**

- The condition number is used in linear maps (with a reference), defined in eigendecompositions for positive definite matrices, and defined again in the SVD section (EIG-12, SVD-11, MAP-11).
- Singular values and singular vectors are used in the eigendecomposition section before they exist (EIG-2: the ellipse axes at $58^\circ$ and $-32^\circ$ come from nowhere; EIG-17). Still open from October.
- Orthonormal columns and $\mathbf Q^\top\mathbf Q=\mathbf I$ are defined in geometry and redefined in linear maps (MAP-4).
- Random-matrix facts are scattered across two sections and four places; the reader is never told that $\sigma_{\max}/\sqrt n\to2$ is the Marchenko-Pastur edge at $c=1$ (EIG-19).
- Weyl's inequality is an exercise part in eigendecompositions and a lemma in the SVD section, with no cross-reference (EIG-15, and the exercises note of that review). Confirmed in the sources.
- The trace, which the eigenvalue section needs, is defined in a subordinate clause of the `einsum` subsection (MAP-17).
- The geometry exercise "Tilting the boundary" needs the inverse, positive definiteness and the condition number, all defined later.

**Length against importance.** The determinant gets about 3,000 words, more than any other topic of the first two sections, for the object deep learning uses least, while its one use (change of variables for densities), promised in that section's opening, never appears in the body (linear maps, structural finding 3; MAP-15). The pathologies of eigenvalues get almost as many words as the spectral theorem and everything that rests on it.

## Most important findings

1. **Decide what the core of the eigendecomposition section is, say so, and cut or demote the rest.** Pseudospectra to Further Reading, the real Jordan form to a sentence, the Lyapunov fine print to the exercise, the random-matrix laws gathered in one closing subsection and connected. EIG-10, EIG-16, EIG-18, EIG-19, and the first three structural findings of that review. The reviewer: "If only one structural change is made in chapter 38, it should be here."
2. **Prove invertibility where it is claimed.** A square matrix with independent columns is invertible by rank-nullity alone; move that proposition into "Inverses" and the nine forward references of the linear-maps section disappear. MAP-9, with MAP-13 and MAP-16.
3. **Cut the determinant to what is used and add its use.** Keep area, the three properties, the triangular rule, multiplicativity and the limits; reduce the permutation formula and cofactor expansion; add one paragraph on the change of variables and the log-determinant, and the block-triangular rule that EIG-6 needs. Repair the order of the uniqueness argument while there. MAP-14, MAP-15, EIG-6, linear maps structural finding 3.
4. **Give each repeated fact one home.** Condition number and the bowl (EIG-12, SVD-11); "the Gram matrix squares the condition number" (GEO-10, SVD structural finding on repetition); Marchenko-Pastur; a layer as hyperplanes (GEO-15); the $10^8$ shear (MAP-19). See the table under Repetition.
5. **Settle the division of labour with the main text in two places.** The angle, Cauchy-Schwarz and cosine similarity are taught in :numref:`sec_linear-algebra` and again here (GEO-4, GEO-12); Newton-Schulz and the Muon quintic are taught in full in `chapter_optimization/muon.md` and again here, and the main text does not cite the appendix (SVD-13).
6. **Make the first section one argument.** It opens on Fashion-MNIST, promises to "check it on the images", and shows no image for 3,300 words. Change the promise or keep it with one cell. Then use the section's own proposition on its own measurement: a cosine spread of 0.34 is what random directions give in about nine dimensions. GEO-1, GEO-14.
7. **Remove the double endings.** All four sections close with a "We can now ..." recap under the last subsection heading, followed by a Summary of eleven or twelve sentences. Keep only what is still open in the closing, and cut each Summary to four or five sentences. GEO-18, MAP-18, EIG-21, SVD-16.
8. **Give the running matrices their own letters.** Bold $\mathbf A$ is the generic matrix, the running matrix, the shear, and several further examples; in the SVD section the text apologizes for LoRA's $\mathbf A$. The chapter's main asset is undersold by its notation. MAP-1, MAP-12, EIG-3, SVD-6.
9. **Say what is central.** The spectral theorem is "the theorem of the chapter" and is introduced in the same register as the Kreiss constant; Further Reading lists fifteen topics of equal weight. The judgment the author plainly has (it shows in the introduction and in Further Reading) is missing from the sections. Eigendecomposition structural finding on ranking; further-reading structural finding "Flat".
10. **Close the small logical holes.** GEO-8, MAP-6, EIG-7, GEO-16, and the five minor wording points under Correctness. Each is one or two sentences.

## Repetition

**Inside the chapter.**

| What | Where it is taught | Proposed home |
|---|---|---|
| Condition number, the elongated bowl, the step-size limit of gradient descent | used in linear maps "Solving Linear Systems in Practice" (MAP-11); defined in eigendecompositions "The Rayleigh Quotient" (EIG-12); defined again in SVD "The Condition Number" with the same bowl (SVD-11) | SVD section, with "for a symmetric positive definite matrix this is $\lambda_{\max}/\lambda_{\min}$"; one clause and a reference elsewhere; the rate and zig-zag belong to the Optimization appendix |
| Forming $\mathbf A^\top\mathbf A$ squares the condition number | geometry, end of "Projection onto a Subspace" and again at the end of "Gram-Schmidt and QR" (GEO-10); SVD at source lines 421, 1057, 1156 and in the Summary, proved only the third time | SVD section, where it is proved; a pointer from geometry |
| Marchenko-Pastur interval | derived in eigendecompositions; re-derived for singular values under "Truncation as denoising"; quoted again after the spectrum cell; $\sigma_{\max}/\sqrt n\to2$ presented as a separate fact (EIG-19) | one "Random Matrices" subsection at the end of eigendecompositions; the SVD section cites it |
| A linear layer is a stack of hyperplanes | three times in geometry and once in the first sentence of linear maps (GEO-15) | "Layers as Stacked Hyperplanes" |
| Orthonormal columns, $\mathbf Q^\top\mathbf Q=\mathbf I$ entry by entry | geometry; again under "Orthogonal Matrices" (MAP-4) | geometry; linear maps starts at the square case |
| Area zero if and only if the edges are collinear; the hypothesis $ad-bc\ne0$ | twice in nearly the same words, and four times for the hypothesis (MAP-13) | "Invertibility and the Determinant" |
| Classification of orthogonal $2\times2$ matrices | "Orthogonal Matrices" and consequence 4 of multiplicativity (MAP-16) | "Orthogonal Matrices" |
| The shear with entry $10^8$ | "Rank", "Limits of the Determinant", and an exercise that asks what the text answers twice (MAP-19) | "Limits of the Determinant" |
| "Read from right to left" | four times in linear maps (MAP-18) | "Composition" |
| Weyl's inequality | exercise part in eigendecompositions, lemma in SVD | lemma in SVD; a Courant-Fischer and Weyl exercise of its own in eigendecompositions that cites it (EIG-15) |
| Randomized SVD | SVD "Measuring and Controlling the Spectrum", an SVD exercise, and the sketching topic of Further Reading | Further Reading and the exercise |
| Nuclear norm and matrix completion | SVD exercise 12 and a Further Reading topic that open with nearly the same sentence | shorten the topic to the recovery theorem |
| Solving in practice, numerical rank | linear maps; SVD; Further Reading topic "Elimination, Determinants, and Rank in Practice" | the sections; cut the topic to its two new citations |
| Every closing paragraph against its Summary | all four sections | see finding 7 |

Text that gives away its own exercises: GEO-13 (both probabilities and the answer "$d=4$"), the "about 0.3" for $10^5$ keys, MAP-19, and the null vectors of $\mathbf C$ in linear maps.

**With the main text.**

- :numref:`sec_linear-algebra` (chapter 1) already gives the law of cosines, the angle by reading the identity backward, the legitimacy argument, a proof of Cauchy-Schwarz, cosine similarity and the brightness example. The geometry section cites it four times and repeats them (GEO-4, GEO-12). Home: the Basics own the definition of the angle and cosine similarity; the appendix owns the proof by projection, the objection paragraph, the floating-point clip and the triangle inequality. About 350 words go.
- For eigenvalues the relation is right: the Basics give the definition, the symmetric $\mathbf S$ and power iteration without proofs, and the appendix "drops the symmetry" and reuses $\mathbf S$. For norms and the SVD there is no re-teaching.
- Newton-Schulz: `chapter_optimization/muon.md` (:numref:`subsec_muon-newton-schulz`) covers odd polynomials on singular values, Frobenius normalization, the cubic, the quintic and the band "from about 0.68 to 1.13". The appendix repeats all of it (SVD-13). Home: the appendix keeps the polar decomposition, the proposition with proof, the basin and the step count; the main text keeps the quintic and the motivation, and should cite the appendix for the proof. I confirmed that `muon.md` has no reference to any label of this chapter.
- LoRA: the clearest statement in the book of what Eckart-Young does and does not say about adapters is here. That is a scope question for the overview, not a defect.

**With other appendix chapters.** The bowl, the rate $(\kappa-1)/(\kappa+1)$ and the constant $L$ belong to the Optimization appendix (EIG-12, SVD-11); the cost of contraction order is made again in the Calculus chapter for forward and reverse mode (MAP-17); the matrix exponential and linear differential equations belong to the Dynamics chapter (EIG-11). In each case this chapter should keep one clause and a label.

## Recurring patterns

**The "We can now" closing.** The last paragraph of the body of all four sections begins with these words (source lines 1065, 1169, 1601 and 1446 of the four files; confirmed), and each is followed by a list Summary. Writing guide 12.2 names the formula as one to avoid. The same double ending was the first recurring pattern of the Basics review.

**The question before the cell.** Nine times in geometry, where the reviewer counts it a strength "near the point where a device becomes a template". In eigendecompositions, seventeen question marks in the body, eleven in the form question, "The cell ...", answer, and "The cell" opens or closes a sentence eleven times. In the SVD section, ten of twenty question marks introduce a cell. Good device, overused; vary by stating the prediction and letting the cell follow.

**Prediction before measurement.** The positive counterpart, and the chapter's best habit: $100\cdot0.99^{99}\approx37$ before the cell prints 36.977; $1/\sqrt{784}$ before the measurement; a bound of 1.41 against a realized 1.0. Used consistently in eigendecompositions and SVD, absent from the middle of linear maps, where cells confirm arithmetic.

**Forward references in place of ordering.** Nine in linear maps ("we state this here and prove it in", "settles the two claims left open"); one each in GEO-8, EIG-7, MAP-14.

**"An exercise proves".** Three instances in eigendecompositions (EIG-15); the third hands the Courant-Fischer theorem to part (d) of a code exercise on a different subject.

**Run-in bold headings.** About 25 in the SVD section, one in eigendecompositions, almost none in the first two sections. The SVD reviewer reads this, with "exercise" against "Exercise" in mid-sentence (SVD-17) and the single use of framework tabs (SVD-4), as a sign that the sections were written in separate passes.

**Templates in the frame.** Four consecutive "X asks ..." sentences in the route paragraph (IDX-1); the sentence that opens the resource list is the same in every chapter of the part (IDX-4); "what it changes" as the wording of four Further Reading topics (FR-2); the sentence "Each topic states the problem it addresses ..." (FR-1, and the part-wide note below).

**Unused imports.** The imports cells of geometry and SVD bring in framework modules that nothing uses, after the text says the code is plain NumPy; `F` is then rebound to a float (GEO-19, SVD-2).

**Notation drift.**

- $\mathbf A$: generic matrix, running matrix, shear, $3\times2$ and $3\times4$ examples, Gershgorin example, random matrix, LoRA's down-projection (EIG-3, SVD-6).
- $a,b,c,d$ as entries, with a parenthesis apologizing that $d$ is not a dimension (MAP-1).
- $\mathbf H$: factor in $\mathbf A=\mathbf G\mathbf H$, Hilbert matrix, Householder reflection. $\mathbf R$: QR factor, rotation, right inverse. $\mathbf P$: surfer matrix ($2\times2$ and $3\times3$), projector in PCA, polar stretch. $\mathbf w$: hyperplane normal in geometry, edge of a parallelogram in linear maps (MAP-12). $\mu$: eigenvalue, real number, Lyapunov exponent, shift. $n$: columns everywhere except PCA.
- Dot against transpose for the inner product, unexplained (GEO-3). Bold against italic for defined terms (GEO-7). "this part" for a `##` (EIG-1).
- A wrong equation reference: "By :eqref:`eq_mdl-dot_geom`" under "Hyperplanes and Signed Distance" cites the cosine identity where the unlabelled signed-length display is meant (GEO-5; confirmed at source line 861).

**Voice.** Present in single sentences ("In 784 dimensions there is no protractor.", "The rotation has decoupled the problem into scalar problems.", "a random matrix stretches some vectors twice as much as its eigenvalues suggest") and in the candour about bounds and evidence. Absent as ranking: the sections do not say which results matter. The photograph is of a cat and is called "the photograph" seventeen times (SVD-1).

## Connection to the main text

- **Named and referenced:** the $\sqrt d$ of attention, cosine similarity for retrieval and contrastive learning, the last layer as hyperplanes, deep linear networks, RNN gradients and initialization, PageRank, weight decay, LoRA, spectral normalization, Muon.
- **Named without a use or without a label:** Gershgorin and pseudospectra have no use named; the matrix exponential has no pointer (EIG-11); Hessians at a critical point and the step-size constant $L$ have no `:numref:` into the Optimization appendix (EIG-12); orthogonal initialization is "attractive" without a reason (SVD-10); the rank factorization is the idea behind bottleneck layers and adapters and the text does not say so (MAP-7); `slogdet` is computed without saying what it is for.
- **Promised and missing:** the change of variables for densities, announced in the opening of linear maps and present only in an exercise.
- **Missing in the other direction:** `chapter_optimization/muon.md` does not cite this chapter for Newton-Schulz (SVD-13).
- **Overstated:** weight decay as ridge regression (SVD-9).
- **Defects of plumbing:** two "Discussions" links at the end of the SVD section, one a placeholder (SVD-18); exercise and cell label prefixes in linear maps still read `geometry-linear-algebraic-ops` (invisible to readers).
- **Lookup use.** The chapter serves a front-to-back reader better than one who lands from a reference. Basis, least squares and hyperplane are in a section whose title names none of them; the trace is buried (MAP-17); "singular" is used before it is defined (MAP-10); $\ker$ is never expanded (MAP-5); the spectral theorem sits in the middle of the longest section; $\|\mathbf A\|_2=\sigma_1$ and "singular values are not eigenvalue magnitudes" are inside "Existence and Uniqueness".

## What to cut

About 4,500 to 5,000 words, by the reviewers' own counts:

- *Geometry* (about 600): the law-of-cosines paragraph and its figure (GEO-4); the paragraph "Least squares leaves the target unchanged" (GEO-9); the first paragraph of "The Nearest-Mean Classifier" (GEO-15); the closing recap (GEO-18); most of the first paragraph of "Cosine Similarity" (GEO-12).
- *Linear maps* (about 900): cofactor expansion to two sentences and the permutation formula stated without the uniqueness detour (MAP-15); the duplicated statements (MAP-13, MAP-16); the recap at the head of "Limits" (MAP-18); the redefinition of orthonormal columns (MAP-4); the weakest confirmation cells.
- *Eigendecompositions* (about 2,000): Pseudospectra (EIG-16); the real Jordan form (EIG-10); the Lyapunov conditions (EIG-18); the Euler cell (EIG-8); the continuity argument for Gershgorin components (to an exercise); the Cholesky operation counts; the matrix exponential unless it gets a pointer (EIG-11); half the closing paragraph and half the Summary (EIG-21).
- *SVD* (about 1,000): the framework-conventions cell and tabs (SVD-4); the Muon motivation and the quintic (SVD-13); the circularity paragraph (SVD-5); two of the three statements on squaring the condition number; the bowl paragraph down to what is new (SVD-11); the recap and most of the Summary (SVD-16); run-in titles that only label a paragraph.
- *Further Reading*: the sentence of FR-1; the elimination topic down to two citations.
- *Exercises*: parts whose answers are printed in the text (geometry, linear maps); SVD exercise 7a; Kreiss part 8c if pseudospectra move.

## What is missing

- The change of variables for densities and the log-determinant, with the triangular and block-triangular rules that make flows cheap (linear maps structural finding 3; EIG-6).
- The reading of the image cosines as "about nine dimensions" (GEO-14).
- A sentence that ranks: this is the theorem of the chapter; these four Further Reading topics come first.
- The reason products of fan-in-scaled random matrices shrink, in one sentence (EIG-20), and the judgment that makes PageRank matter (EIG-22).
- The two-vector form of power iteration for spectral normalization (SVD-15); the step from one principal direction to $k$ (SVD-12); the shapes of full, thin and compact SVD on one small case (SVD-3).
- Definitions at first use: singular (MAP-10), trace (MAP-17), kernel (MAP-5), matrix norm in the residual bound (MAP-11).
- Easy exercises on the core: Fibonacci or another $2\times2$ diagonalization, the four subspaces of a small matrix, the pseudoinverse projectors, `solve` against `inv`.
- Real data in the second half: PCA is shown on a synthetic cloud and Fashion-MNIST appears only in an exercise; the spectrum cell compares two constructed matrices (SVD-14). Both are deferred, since they need changed cells.
- Optional, for Further Reading: gradients through `eigh` and `svd`; the graph Laplacian.
- Elimination is named in the introduction as the origin of the subject and is taught nowhere (IDX-2, MAP-8). Either say so or connect the anecdote to least squares.

## What is best and should set the model

- **The running matrices.** The shear is defective in one section and is $\mathbf R\operatorname{diag}(\varphi,1/\varphi)\mathbf R^\top$ in the next, where its singular values are the golden ratio and its reciprocal; the running matrix is read by rows with the input $(2,-1)$ in one section and by columns with the same input in the next, and its eigenvalues and singular values both multiply to 5. This is what a running example should be.
- **Number first, statement second.** $10=5+5$ before Pythagoras; the three-point line fit before the projection theorem; four iterates of the surfer before any definition, and the closed form 230 lines later reproducing the third.
- **Eckart-Young.** Two planes in $\mathbb R^3$ meet in a line; then the figure, the proof, and the reading "In words: $\mathbf B$ must annihilate some direction ...". The house style at its best.
- **Bound against reality.** "The bound holds, but it is four times the actual value 10.8"; the bound allows 1.41 and the realized change is 1.0, with the reason.
- **Data that contradict the reference.** "They do not. Raw images have cosines around 0.6", ending in a judgment about when a cosine counts as large; the nearest-mean rule that works on one pair of classes and then "does not" on harder ones.
- **The cancellation account of transient growth**, and the two nilpotent Jacobians whose product has spectral radius 4.
- **Composition explains the product**, and the rank factorization read as "first computes $r$ numbers from its input ... and then spreads them".
- **Candour.** "(a theorem we use without proof)"; "It says nothing about the adapter that training finds"; "rests on an empirical correlation, not a theorem".
- **The introduction's central sentence** and its three one-line instances, and the Further Reading notebook as a whole, which shows the ranking judgment the sections withhold.

## Checked in the sources

**IDX-3, the "typical case".** Confirmed, and the finding is slightly stronger than the index reviewer could state. `index.md` line 31 says the running matrix and the shear recur, "the first as the typical case and the second as the case where eigenvectors fail". In `mdl-eigendecomposition.md` the running matrix $\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ first appears under "Complex Eigenvalues" (line 578) with $\lambda=2\pm i$ and the complex eigenvector $(1-i,1)$: it has no real eigenvector. It returns at line 1341 as the example of a matrix that is not normal ("This is why its iterates did not move on circles"). The section's Summary lists three ways eigenvalues mislead, "defective, rotating, and non-normal"; the running matrix is the example of two of them. It is typical only in the sense that its eigenvalues are distinct, so it is diagonalizable over $\mathbb C$, and the section never says that. The well-behaved cases of that section are the surfer matrix $\mathbf P$ and the symmetric $\mathbf S$, which the introduction does not mention. Fix in the introduction: drop "typical" and say what each matrix is for (for example, "the first hides a rotation, the second has too few eigenvectors"), and consider naming $\mathbf P$ as the third recurring object.

**The "We can now" closings, the wrong `:eqref:` of GEO-5, the absence of a reference from `muon.md`, the three definitions of the condition number (linear maps line 618, eigendecompositions line 997, SVD line 1096), and Weyl in both sections** were confirmed as the reviewers reported them.

**Part-wide: the sentence in all six Further Reading files.** `grep -n "Each topic states" chapter_mdl-*/further-reading.md` matches every one of the six files, in the opening paragraph each time:

| File | Line |
|---|:-:|
| `chapter_mdl-linear-algebra/further-reading.md` | 8 |
| `chapter_mdl-calculus/further-reading.md` | 10 |
| `chapter_mdl-optimization/further-reading.md` | 10 |
| `chapter_mdl-probability-statistics/further-reading.md` | 10 |
| `chapter_mdl-information-theory/further-reading.md` | 7 |
| `chapter_mdl-dynamics/further-reading.md` | 9 |

The sentence is identical in all six: "Each topic states the problem it addresses and what it changes relative to the chapter." To be exact about what is there: it is not a leftover imperative or a comment addressed to the author. It is a grammatical sentence addressed to the reader that describes the format of the topics, and it paraphrases the writing guide's rule for such sections (`docs/writing-guide.md` line 930: "the problem it addresses and the one thing it changes relative to the method"). The reviewer's point (FR-1) stands in the weaker form: the sentence tells the reader the template instead of anything about the subject, and the reader meets it six times. The whole opening paragraph is in fact one template across the six files: "The sections of this chapter [teach] ... This section collects, in the order of the chapter, topics they had no room for ... Each topic states ... Resources for the material the chapter does teach are listed in its introduction ... Nothing else in the book depends on this section." Recommendation for the overview: delete the sentence in all six, and vary or shorten the rest of the paragraph.

## Disagreements between section reviews

None on a matter of fact. Four tensions a reviser should resolve deliberately:

- **The determinant section is asked to shrink and to grow.** The linear-maps review wants it cut (MAP-15); the eigendecomposition review needs the block-triangular rule added there (EIG-6) and the SVD review wants the dimension formula for two subspaces placed next to rank-nullity (SVD-7). These are compatible if the cut is cofactor expansion and the addition is the block rule, which also serves flows.
- **Questions before cells** are a strength in the geometry review and "a drone" by the eleventh in the eigendecomposition review. Both agree on the remedy: add no more, and vary.
- **Cells.** The eigendecomposition review says "every cell computes a number the prose predicted or reads"; the linear-maps review finds nine of thirteen cells re-running arithmetic. This is a real difference between sections, not between reviewers' standards.
- **Exercise placement.** The geometry review proposes moving "Tilting the boundary" to the SVD section, which already has twelve problems, the top of the band, and whose reviewer wants two more added. The alternative given (a line naming the later sections it needs) avoids the conflict.

## Could not check

- Rendered HTML and PDF; the section reviewers read sources, stored outputs and figures.
- Outputs of the TensorFlow, JAX and MXNet tabs. The code is NumPy in all but one cell, so the risk is small; the one framework-specific cell is SVD-4.
- I did not re-verify the reviewers' numerical recomputations.
