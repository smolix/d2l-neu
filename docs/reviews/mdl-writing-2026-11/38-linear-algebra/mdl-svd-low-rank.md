# Singular Value Decomposition and Low-Rank Approximation

Source: `chapter_mdl-linear-algebra/mdl-svd-low-rank.md`. About 8,200 words of prose and mathematics to the Summary, plus 2,300 words of exercises (10,500 to the slides marker, code excluded). Four `##` parts, 12 `###` subsections, 10 code cells, 9 figures.

## Verdict

The best section of my three and close to the bar in its first two parts. It has a question a reader wants answered ("Twenty well-chosen pieces ... reproduce it with a relative error of 8% ... Which twenty pieces are these, and could any other twenty do better?"), a small example that is a pleasure in its own right (the shear, whose singular values turn out to be the golden ratio and its reciprocal, and whose two rotation angles add to $90^\circ$ for a reason the text gives), and a proof of Eckart-Young that is led by a picture of two planes meeting in a line. The text is unusually honest about what its results do and do not say: the bound allows a relative change of 1.41 and the realized change is 1.0, with the reason; Eckart-Young "says nothing about the adapter that training finds". I recomputed every number in the section, including several that no cell prints, and found no error. Every proof is correct.

Its faults are of organization, not of substance. The subsection "Existence and Uniqueness" is a container for six topics, one of which is a table of framework return conventions. The last part teaches Newton-Schulz orthogonalization, which the main text (:numref:`sec_muon`) teaches in full without referring here. And three facts are told three or four times each. The most important fix is to give the first part a structure a lookup reader can navigate, and to make "The SVD in Deep Learning" add to the main text instead of repeating it.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

The Eckart-Young subsection and the condition-number subsection, taken alone, are at 5.

## What works

- **The opening.** Two requirements stated in plain words ("a description that works for every matrix, and that tells how few numbers suffice to describe a large matrix well"), then the photograph with its count (18,020 of 200,000 numbers), then the idea in one sentence: "The answer is to stop asking one basis to serve as both the input and the output axes." Keep verbatim.
- **The shear, by hand.** $\mathbf A^\top\mathbf A$ has trace 3 and determinant 1, so $\sigma=\varphi,1/\varphi$; $\mathbf v_1\propto(1,\varphi)$, $\mathbf u_1\propto(\varphi,1)$; "The two angles add up to $90^\circ$ because $\mathbf{u}_1$ is $\mathbf{v}_1$ with its coordinates swapped". The same matrix that defeated the eigendecomposition is taken apart completely, and it returns for the dyadic sum, the rank-one truncation, the pseudoinverse, the condition number and the polar factor. This is what a running example should be.
- **"The proof is also an algorithm"**, followed by a cell that carries out the three steps on a $3\times2$ matrix and compares with the library. Code that teaches the proof.
- **Eckart-Young.** "Picture $n=3$ and $k=1$ first. ... Two planes through the origin of $\mathbb{R}^3$ meet in at least a line. On that line $\mathbf{B}$ gives zero while $\mathbf{A}$ stretches by at least $\sigma_2$". Then the figure, then the proof, then "In words: $\mathbf{B}$ must annihilate some direction in the span of the top $k+1$ right singular vectors". Picture, proof, reading: the house style at its best. The remark that the hypothesis "enters here, as $\sigma_{k+1}(\mathbf{B})=0$" tells the reader where to look.
- **The energy ratio, with its trap.** "The ratio needs care. On the photograph $\sigma_1^2$ alone holds 89% of the energy. A constant image has rank one ... so the first dyad is mostly that mean." A warning that practitioners need and textbooks omit.
- **A competitor.** Twenty actual columns against twenty singular triples: 0.31 against 0.08. The theorem says no rank-20 matrix does better; the cell shows how much worse a natural one does.
- **The condition number.** A $2\times2$ system where changing $b_2$ by $10^{-4}$ moves the solution from $(1,1)$ to $(0,2)$, before the definition. After the proposition: "the bound allows a relative change of $1.41$ in $\mathbf{x}$. The realized change is $1.0$. The perturbation $(0,10^{-4})$ has only $1/\sqrt2$ of its length along the most amplified direction". Bound, reality, and the reason for the gap.
- **The proof of minimum-norm least squares.** "The rotation has decoupled the problem into scalar problems." (October asked that this sentence be kept; it has been.) And the reading after it: "a small residual and a small norm concern disjoint coordinates, $i\le r$ and $i>r$, so both can be had at once."
- **LoRA, stated at its true strength.** "It says nothing about the adapter that training finds. ... That such updates exist for many fine-tuning tasks is the empirical finding behind LoRA."
- **The last sentence of the body.** "Points on a circle in the plane form a data matrix of rank 2, yet the circle is one-dimensional, and describing such curved structure needs derivatives, the subject of the next chapter." A limit and a bridge in one image.

## Structural findings

- **"Existence and Uniqueness" is a grab-bag (moderate).** Lines 183 to 500 hold, under one `###` heading: the existence proof, the shear by hand, a rectangular check, uniqueness, the maximum stretch and deflation, the relation to the eigendecomposition with the determinant identity, how libraries compute the SVD, and four framework tabs on return conventions. A reader sent here for "why $\|\mathbf A\|_2=\sigma_1$" or "singular values are not eigenvalue magnitudes" cannot find either from the table of contents. Split into three subsections: "Existence" (proof, shear, the $3\times2$ check), "Uniqueness and the Largest Stretch" (uniqueness, :eqref:`eq_mdl-sigma1-variational`, deflation), and "Singular Values and Eigenvalues" (Gram matrices, the running matrix, the determinant identity, computing). See SVD-4 for the tabs.
- **Run-in headings instead of structure (minor, pattern).** The section has about 25 bold run-in paragraph titles ("**The compact SVD and the dyadic sum.**", "**The maximum stretch.**", "**The photograph.**", "**A competitor.**", "**Truncation as denoising.**", "**The smallest case.**", "**The normal equations.**", "**Condition and the shape of a bowl.**", ...). The eigendecomposition section has one. They are a sign that subsections hold too many topics, and they give the text the look of notes. Where a run-in title names a result a reader will look up (maximum stretch, numerical rank, polar decomposition), promote it; where it only labels the next paragraph ("**The photograph.**", "**A competitor.**"), delete it and let the first sentence do the work, which it already does.
- **Newton-Schulz is taught twice in the book (moderate).** "Orthogonalized Updates" (125 lines and a cell) covers the polar factor, odd polynomials acting on singular values, Frobenius normalization, the cubic, zero singular values, and Muon's quintic with its band "from about $0.68$ to $1.13$". `chapter_optimization/muon.md:177-230` covers odd polynomials, Frobenius normalization, the cubic, zeros, the quintic and the same band, and does not point here. By writing guide 7.6 the appendix holds the proof and the detail; so keep here what the main text lacks (the polar decomposition and its shear example, the proposition with proof, the basin $(0,\sqrt3)$, the step count $\log_{3/2}(1/s_{\min})+4$ and its cell) and cut what it has (the motivation paragraph on momentum, the Muon paragraph, the quintic; see SVD-13). The later revision of :numref:`sec_muon` should then cite :eqref:`eq_mdl-odd-polynomial` for the proof.
- **The proof of Eckart-Young loses its thread (minor).** The theorem is followed by: attribution, picture, figure, "Proof (spectral norm)", a reading, the shear example, a transitional sentence, Weyl's lemma, its proof, a remark, "Proof (Frobenius norm)", and uniqueness. All of it is good and in a defensible order, but a reader does not know after the first $\blacksquare$ that half the theorem is still unproved. Say so at the start: "We prove the spectral-norm statement first. The Frobenius statement needs one more inequality, Weyl's, which the same argument gives." See SVD-7.
- **Repetition (moderate, writing guide 7.7).** (i) "Forming $\mathbf A^\top\mathbf A$ squares the condition number" is said at lines 421, 1057 and 1156 and again in the Summary; it has a proof only the third time. (ii) The condition number, the elongated bowl and the step-size limit of gradient descent are taught in :numref:`subsec_mdl-rayleigh`, again here under "Condition and the shape of a bowl", and a third time (by reference) in the Optimization appendix. (iii) The Marchenko-Pastur interval is derived in the eigendecomposition section, re-derived for singular values under "Truncation as denoising", and quoted again after the spectrum cell. Each should have one home and one-clause citations elsewhere.
- **Level against Basics.** :numref:`sec_linear-algebra` defines the Frobenius and spectral norms and says "Matrices that are not symmetric need more care (:numref:`sec_mdl-svd-low-rank`)". This section delivers exactly that and is deeper throughout; no re-teaching. :numref:`sec_weight_decay` and the regression chapter use least squares and ridge; the filter-factor form here is new and worth its place.
- **Is it two sections?** Not necessarily. But "The SVD in Deep Learning" would lose nothing by shrinking to LoRA, the polar factor, and spectral normalization at about half its present length, with randomized SVD already represented in Further Reading ("CUR Decompositions and Sketching") and in exercise 10.

## Passage-level findings

**SVD-1** (minor, enjoyment). Opening.

> photograph of $400\times500$ pixels is a matrix with 200,000 entries. Like

The photograph is of a cat lying on its back (`img/cat1.jpg`), and the figure shows it emerging from a grey blur at rank 1 to a recognizable animal at rank 20. The text never says what the picture shows; it is "a photograph" or "the photograph" in all 17 mentions. Naming it costs three words and gives the reader something to see: at rank 5 "a pale shape with four limbs", at rank 20 "whiskers". This is the concrete image writing guide 10.2 asks for.

**SVD-2** (minor, code). Imports cell and "Best Low-Rank Approximation".

> F = np.linalg.norm(img)

The PyTorch imports cell brings in `math`, `torch.nn as nn` and `torch.nn.functional as F`, none of which the section uses; the other tabs likewise import `autograd`, `gluon`, `init`, `nnx`, `optax`. This line then rebinds `F` to a float. Reduce the imports to NumPy, the framework (for one `svd` call) and `d2l`, and call the norm `fro`.

**SVD-3** (minor, explanation). "The Factorization and Its Geometry".

> Libraries return, on request, the *thin SVD*, with $\min(m,n)$ columns in $\mathbf{U}$ and

Three variants (full, compact, thin) arrive within ten lines, and the thin one is described by what libraries do. The distinction matters later ("Let $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ be a thin SVD"). Give the shapes for one small case, for example the $3\times2$ matrix used later: full $3\times3,\,3\times2,\,2\times2$; thin $3\times2,\,2\times2,\,2\times2$; compact equals thin when $r=2$.

**SVD-4** (moderate, pedagogy). "Existence and Uniqueness", "Computing the SVD".

> factor is $\mathbf{V}$ or $\mathbf{V}^\top$. What does the framework's routine

The only framework-specific cell in the chapter prints three shapes and `True`, and is followed by four tabs of return-order conventions (`(ut, l, v)`, `(s, u, v)`, float32 in JAX, MXNet's restriction to wide matrices). This is the "framework documentation" the Basics review found in chapter 5, placed in the middle of the section's main mathematical subsection, between the determinant identity and the four subspaces. The author's decision there was that practical content may stay but should be pruned. Here: keep the fact (libraries disagree on order and on $\mathbf V$ versus $\mathbf V^\top$) in one sentence, keep one tab-specific sentence per framework if wanted, drop the cell, and move the paragraph to the end of the subsection so that it does not interrupt.

**SVD-5** (minor, explanation). "Existence and Uniqueness", maximum stretch.

> $\sigma_1$ rather than a second proof that the SVD exists, since the Rayleigh

The paragraph answers an objection ("is this circular?") that the reader has not raised and may not understand, and it does so before saying what the result is for. Shorten to the result and its reading; the pointer to the exercise that builds the SVD from the maximum stretch can stay as one clause.

**SVD-6** (minor, notation). "Existence and Uniqueness", and throughout.

> magnitudes. The chapter's running matrix

$\mathbf A$ is the generic $m\times n$ matrix, the shear ("For $\mathbf{A}=\ldots$ the Gram matrix is"), a $3\times2$ example, the running matrix, a $3\times4$ example, and then LoRA's down-projection, where the text has to apologize: "here $\mathbf{A}$ is the paper's $k\times n$ down-projection, not the generic matrix of this section". Writing guide 8.2 treats such a sentence as a signal to rename. Give the shear and the running matrix their own letters across the chapter, and write the LoRA update with other letters (or as $\mathbf B\mathbf C$ with a note on the paper's notation). Likewise $n$ is the number of columns everywhere except in PCA, where it becomes the number of rows; the text flags it ("here $n$ counts the points and $d$ the features"), which is acceptable, but $\mathbf P$ is a projector in PCA and the polar stretch 400 lines later.

**SVD-7** (minor, explanation). "Best Low-Rank Approximation".

> The same lower bound yields the inequality on which the Frobenius case rests.

This is the only signal that the proof of the theorem is not finished, and it comes after an example that reads as a conclusion ("and no matrix of rank one comes closer"). Move the announcement to the head of the proof (see Structural findings). Also in the proof: "where the equality is the dimension formula for two subspaces" is followed by a one-clause proof of that formula. Either the formula belongs in :numref:`sec_mdl-linear-maps` next to rank-nullity (it is used again in exercise 7d of the previous section) and is cited here, or the clause should become two sentences; as written it is too fast to check and too slow to skip.

**SVD-8** (minor, code: claims the outputs do not show; all verified). "Best Low-Rank Approximation".

> large, so the first dyad is mostly that mean. Rank 4 already

Three numbers are not printed by any cell: rank 4 keeps 95% at a relative error of 21%; after subtracting the mean, 95% takes rank 8; the singular values fall as $i^{-1.1}$. I recomputed all three from `img/cat1.jpg` (energy 0.9554 at rank 4, error 0.211; rank 8 after centring; fitted slope $-1.11$ over the first hundred). They are right. When the cell is next touched, print the first two; they are the point of the paragraph.

**SVD-9** (minor, correctness: overstatement, confident). "The Pseudoinverse and Least Squares".

> $0.083$. Weight decay in :numref:`sec_weight_decay` is ridge regression for

"Weight decay is ridge regression for neural networks" is a slogan, and as a statement it is true only for a linear model with squared loss trained by plain gradient descent; with Adam the two differ, which is the reason AdamW exists and which the book explains elsewhere. Write: "For a linear model with squared loss, the weight decay of :numref:`sec_weight_decay` is ridge regression, and the filter factors show which directions it shrinks."

**SVD-10** (minor, thought). "The Condition Number".

> factors $\mathbf{U}$ and $\mathbf{V}$ are safe to compute with, and

"and orthogonal initialization of weights is attractive" hangs on the end of a sentence about numerical stability, without a reason. The previous section gave the reason (every vector keeps its length). Cite it in a clause or drop the remark.

**SVD-11** (moderate, repetition). "The Condition Number".

> The *condition number* of a matrix of rank $r$ is the ratio of its largest to

Second definition in the chapter; the first is $\kappa=\lambda_{\max}/\lambda_{\min}$ for a positive definite matrix in :numref:`subsec_mdl-rayleigh`, and :numref:`sec_mdl-linear-maps` uses the term before either. Make this the home, add "for a symmetric positive definite matrix this is $\lambda_{\max}/\lambda_{\min}$", and reduce "Condition and the shape of a bowl" to what is new here: contours of least squares have axis ratio $\kappa(\mathbf A)$ and the Hessian has $\kappa(\mathbf A)^2$. The rate $(\kappa-1)/(\kappa+1)$ and the zig-zag belong to the Optimization appendix, which is cited.

**SVD-12** (minor, pedagogy). "Principal Component Analysis".

> By deflation, the top $k$ principal directions are

The proposition is stated and proved for one direction. The step to $k$ directions is these two words, resting on a definition of deflation given in a single clause 500 lines earlier. Add the sentence: "Among unit vectors orthogonal to $\mathbf v_1$, the variance is largest along $\mathbf v_2$, by the same argument on the remaining eigenvectors, and so on."

**SVD-13** (moderate, repetition with the main text). "Orthogonalized Updates".

> $p(\sigma)=3.4445\sigma-4.7750\sigma^3+2.0315\sigma^5$ that gives up

The quintic, its coefficients and its band are the subject of `chapter_optimization/muon.md:206-230`, with a figure. Here they are an afterthought to a cell about the cubic. Replace the last three sentences of the paragraph with one: "Muon uses a tuned quintic in place of the cubic, trading exact convergence for speed (:numref:`sec_muon`)." Likewise the opening paragraph of the subsection ("An optimizer with momentum keeps, for each weight matrix, a running average") re-motivates Muon; one sentence with the reference is enough before "The polar decomposition".

**SVD-14** (moderate, pedagogy). "Measuring and Controlling the Spectrum".

> spectrum, so one measures before truncating. Is low rank a property of

The cell contrasts a Gaussian matrix with a matrix constructed to have singular values $e^{-i/12}$, and reports that the second reaches 95% of its energy at rank 18. That number is $6\ln20$: the cell confirms its own construction. The Gaussian half teaches something (no gap, rank 199 of 256); the other half does not answer the question the paragraph asks, which is about what real matrices hold. The photograph is already in memory and would be an honest second case (rank 4, or 8 after centring), and a trained weight matrix or a fine-tuning update would be the right one. This asks for a changed cell and so falls under the deferred experiments; in the writing pass, change the prose to say that the second matrix is built to have a decaying spectrum and stands for the case the first does not show.

**SVD-15** (minor, explanation). "Measuring and Controlling the Spectrum".

> (:numref:`sec_mdl-eigendecompositions`). Repeated multiplication turns a

Spectral normalization is explained well, but the estimator is given as "power iteration on $\mathbf W^\top\mathbf W$" and the practical two-vector form ($\mathbf v\leftarrow\mathbf W^\top\mathbf u$, $\mathbf u\leftarrow\mathbf W\mathbf v$) appears only in exercise 9. One sentence here would connect them: the two half-steps are one multiplication by $\mathbf W^\top\mathbf W$, and $\|\mathbf W\mathbf v\|$ is the estimate.

**SVD-16** (moderate, language/structure). End of "Measuring and Controlling the Spectrum".

> We can now decompose a $2\times2$ matrix by hand and any matrix with one

A recap under the last subsection heading, in the formula that closes all four sections of the chapter, followed by a Summary of twelve sentences that lists every result of the section including a worked number for the shear. Keep the three limits (cost, low rank cannot be assumed, linear structure only): they are the real ending and the last is the best sentence of the close. Delete the two recap sentences before them. Cut the Summary to five sentences: what the SVD is, where it comes from, what the list of $\sigma_i$ gives (rank, norms, best approximation, shortest least-squares solution, conditioning) in one sentence, and what training uses it for.

**SVD-17** (minor, language). Whole section.

> $\|\mathbf{Y}\|_2$ :cite:`Horn.Johnson.2012`; exercise

In mid-sentence this section writes "exercise :exref:" in lower case (five times) and "problem :exref:" once; the previous section writes "Exercise" with a capital in the same position ("and Exercise :exref:"). Choose one for the part.

**SVD-18** (minor, defect). After the exercises.

> [Discussions](https://d2l.discourse.group/t/36826)

The four framework tabs above this line each hold a placeholder link `https://d2l.discourse.group/t/svd`, and this untabbed link follows them, so a reader sees two "Discussions" links, one of them dead. Remove the tabs or the placeholder.

## Exercises

Twelve problems, about 2,300 words. As in the previous section the quality is high and the weight is toward long computational projects: nine are `[code]` or `[extended]`. I checked the mathematics of exercises 1, 3, 7 and 11 (the hard threshold: the truncation's asymptotic squared error is $2+3/x^2$ against $x^2$ for zero, which gives $x>\sqrt3$ and the singular value $4/\sqrt3$); all are right.

- **Exercise 1** (the SVD of $\left(\begin{smallmatrix}3&0\\4&5\end{smallmatrix}\right)$ by hand, six parts) is the right first exercise. Parts (e) and (f) are separate ideas (SVD of a symmetric matrix; $\sigma_n\le|\lambda|\le\sigma_1$) and would be easier to find as their own problem.
- **Exercise 2a** (building the SVD from the maximum stretch, without the spectral theorem) is excellent and is the proof the body promises "avoids it".
- **Exercise 5** is two exercises: (a) to (d) on centring, a clean piece of analysis, and (e), PCA of Fashion-MNIST t-shirts and trousers, which is a project of its own and the only place the section touches real data. Split them, and consider promoting a short version of (e) into the PCA subsection when experiments are revisited: the synthetic cloud there shows nothing a reader could not predict.
- **Exercise 7a** asks the reader to "Show directly, without :eqref:`eq_mdl-odd-polynomial`" the special case of a proposition proved in the text in three lines. That is a restatement of the text; drop (a).
- **Exercise 8c** (Procrustes alignment of embedding spaces) is a real use and well posed.
- **Exercises 11 and 12** (optimal hard threshold; matrix completion by Soft-Impute) are research-flavoured and long. Both are good. If the set must shrink, 12 is the one that leans furthest from the section: the nuclear norm and soft-thresholding are not taught in the body, and Further Reading has a topic on exactly this, opening with nearly the same sentence ("the convex stand-in for rank ... $\ell_1$ stands in for sparsity").
- **Missing.** A short problem on the four subspaces (find all four for a given $3\times2$ matrix and check the dimensions), and one on the pseudoinverse identities ($\mathbf A\mathbf A^+$ and $\mathbf A^+\mathbf A$ are the orthogonal projectors onto the column and row spaces), which exercise 6 uses without having established.

## Still open from October

October's only note on this section was a sentence to keep ("The rotation has *decoupled* the problem"); it is kept. October's note that singular vectors are used before the SVD is defined concerns the previous section.

## For the chapter review

- The section rests on the spectral theorem, positive semidefiniteness, the Rayleigh proposition, the Marchenko-Pastur law and power iteration from :numref:`sec_mdl-eigendecompositions`, on rank-nullity from :numref:`sec_mdl-linear-maps`, and on least squares from the first section. All are cited by label. It is the right last section: the chapter introduction's "every matrix becomes a list of independent stretches" is delivered here.
- The running matrices work across sections. The shear is defective in the previous section and is $\mathbf R_{31.7^\circ}\operatorname{diag}(\varphi,1/\varphi)\mathbf R_{58.3^\circ}^\top$ here; the running matrix has eigenvalues of modulus $\sqrt5$ and singular values 3.618 and 1.382 whose product is 5. This is the chapter's main asset; the notation ($\mathbf A$ for all of them) undersells it.
- Overlap inside the chapter: condition number and bowl (with the eigendecomposition section), least squares (with the geometry section, cited), numerical rank and `solve` (with linear maps). Overlap with the main text: Newton-Schulz and the Muon quintic (:numref:`sec_muon`); LoRA is treated here at a length the main text may not match, which is a question of scope for the overview: the clearest statement in the book of what Eckart-Young does and does not say about LoRA is in an appendix.
- Style differs from the previous section: many bold run-in headings here, almost none there; "exercise" against "Exercise"; this section uses framework tabs once, that one never. The two read as written in separate passes.
- Chapter-wide patterns confirmed here: the "We can now" closing plus list Summary; the question before each cell (ten of the section's twenty question marks introduce a cell).
- Mathematics: no errors found. One overstatement (SVD-9).
