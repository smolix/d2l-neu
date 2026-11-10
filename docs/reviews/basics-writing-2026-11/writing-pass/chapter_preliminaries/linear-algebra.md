# Writing pass log: `chapter_preliminaries/linear-algebra.md`

Prose only. No code cell, label, figure include, or slide was changed.
Checker: "OK: only prose changed"; `tools/lint_source.py` clean.

## Length

Prose words (checker count, whole file to the slides marker): 5943 before,
5940 after. The cuts (about 330 words) were spent on what the review asked
to add: room for the column view of a matrix, the worked decomposition of
$\mathbf{e}_1$, the identity matrix and the inverse, and the skip notice.
The section is not shorter in total; it is thinner in topics.

## Before/after pairs

1. Before: "Training scores the images in minibatches of, say, 256 at a time,
   and repeats this for thousands of minibatches, so the scoring has to run
   as one call to compiled code rather than as Python loops over the
   numbers."
   After: "Training scores a minibatch of 256 images at a time
   (:numref:`sec_ndarray`), thousands of times over, so Python loops over the
   numbers are out of the question."

2. Before: "This section builds that expression piece by piece and then asks
   two questions that every model raises"
   After: "We have not yet said what a matrix is, let alone a product of two.
   This section builds the expression piece by piece and then asks two
   questions that every model raises"

3. Before: "Linearity means that the map is determined by what it does to the
   *coordinate vectors* $\mathbf{e}_j$, which have a 1 in position $j$ and 0
   elsewhere."
   After: "What does such a map do to the *coordinate vectors*
   $\mathbf{e}_j$, which have a 1 in position $j$ and 0 elsewhere? It returns
   a column: the product $\mathbf{A}\mathbf{e}_j$ is column $j$ of
   $\mathbf{A}$, and for our $2 \times 3$ example
   $\mathbf{A}\mathbf{e}_2 = (1, 4)$. That settles every other vector as
   well."

4. Before: "The definition :eqref:`eq_cos_sim` makes sense only if the ratio
   lies between $-1$ and $1$, as a cosine must."
   After: "Are we entitled to do this? The definition :eqref:`eq_cos_sim`
   makes sense only if the ratio lies between $-1$ and $1$, as a cosine
   must."

5. Before: "A matrix is also a map, which raises a second question: by how
   much can it stretch a vector?"
   After: "The Frobenius norm is rather crude in one respect, though. It
   forgets that a matrix is also a map, and a map raises a different question
   of size: by how much can it stretch a vector?"

6. Before: "For $\mathbf{S}$ and $\mathbf{x} = \mathbf{e}_1$, the iterates
   are $(2, 1)$, $(5, 4)$, $(14, 13)$, and in general
   $\left((3^k + 1)/2,\ (3^k - 1)/2\right)$"
   After: "Half the difference of the two eigenvectors does it:
   $\mathbf{e}_1 = \frac{1}{2}(1, 1) - \frac{1}{2}(-1, 1)$. Every
   multiplication by $\mathbf{S}$ triples the first part and leaves the
   second alone, so $\mathbf{S}^k\mathbf{e}_1 = \ldots$" followed by
   "Nothing in this computation was special to $\mathbf{S}$."

7. Before: "We can now write the computation of a linear model on a whole
   minibatch as one matrix product, read off its shape and cost before
   running it, measure errors and weights with norms, and predict how
   repeated multiplication scales a vector. What we cannot do yet is improve
   the weights."
   After: "Everything in this section took the weights as given, and ours
   were random numbers. What we cannot do yet is improve the weights."

8. Before: "Since $\mathbf{S}$ maps $(1, 1)$ to $(3, 3)$ and $(-1, 1)$ to
   $(-1, 1)$, the directions $(1, 1)$ and $(-1, 1)$ are eigenvectors with
   eigenvalues 3 and 1."
   After: "Its eigenvectors can be found by inspection. Since $\mathbf{S}$
   maps $(1, 1)$ to $(3, 3)$ and $(-1, 1)$ to itself, these two directions
   are eigenvectors, with eigenvalues 3 and 1."

## Repetition removed

- Minibatch of 256 images (owned by `ndarray.md`): the opening now cites it
  in one clause; the "one call to compiled code" motive is stated once (the
  library sentence after the expression), not twice.
- "Shape and cost are known before we compute" was in the first and third
  opening paragraphs; kept in the first only.
- Elementwise operations and broadcasting (owned by `ndarray.md`): the
  re-explanation of what an elementwise operation is, and the second
  statement that $\mathbf{b}$ is added to every row, are gone.
- "(256,) does not broadcast against (256, 784)" (owned by
  `subsec_broadcasting`): the retelling before the centering cell is reduced
  to the shape that `keepdims` produces.
- "A reduction summarizes many numbers by one" was said in the part intro and
  again as the first sentence of "Reductions"; the second is cut.
- The dot product was described three times (intro of "Products", after its
  equation, "Hadamard product followed by a sum"); the middle one is cut.
- Orders 0, 1, 2 of scalars, vectors, matrices: stated in the part intro,
  cut from "Tensors".
- The loss was defined in the norms opening after the same paragraph had
  already said "measure how wrong they are" (LA-8); one statement now.
- Spectral norm: was introduced under "Matrix Norms" and again under "Powers
  of a Matrix"; now defined once, where it is derived.
- "Norms appear throughout the book: as losses, as penalties ..." repeated
  the norms opening; cut.
- The closing "We can now ..." recap; the Summary is the only recap.
- The matrix-vector tab pair repeated one sentence per tab; it is now shared
  prose, and each tab holds only the operator or function name.

## Review findings

Applied: LA-1 (opening trimmed), LA-2 (terminology paragraph reduced to two
sentences), LA-3 (tensor notation $\mathsf{X}$ and channel-order parenthesis
cut), LA-4 (heading now "Keeping the Reduced Axis", label unchanged), LA-5
(column view given two paragraphs and the worked numbers; the layer sentence
moved to the end of "Matrix--Matrix Products"), LA-6 (batched `@` remark
cut), LA-7 (FLOPs-per-byte argument cut; one clause points to
`chap_performance`), LA-8, LA-9 (reference to `sec_weight_decay` added),
LA-10 (spectral norm deferred to "Powers of a Matrix"), LA-11
($\mathbf{e}_1$ decomposed and the iterates derived before the general
formula; one sentence on why the coefficients are dot products), LA-12
(real reason given in one line; paragraph split), LA-13 (softened; the cell
that defines `S` cannot move). Structural: identity matrix and inverse
defined after the cost paragraph; the error vector in "Norms" is now
attributed to "a model that predicts one number per example, such as a
price"; the hand-over says "when a parameter changes"; the eigenvalue part
opens with a notice that it can be skipped on a first reading.

Deferred:
- New exercise on identity and inverse (review, Exercises): exercises are
  out of scope.
- Exercise 6 friendlier sizes: exercises out of scope.
- LA-6 relocation: the batched-`@` fact was cut, not moved to the attention
  chapter (no moves between files). If a later section relies on it, it
  needs a sentence there.
- `A` is $2 \times 3$ in code while $\mathbf{A}$ is square in the eigenvalue
  text: needs a code rename.
- Section still above the 4,500-word guide ceiling; a split is structural.

## Factual claims added or changed

- $\mathbf{A}\mathbf{e}_2 = (1, 4)$ and
  $\mathbf{A}(0, 1, 2) = 1 \cdot (1, 4) + 2 \cdot (2, 5) = (5, 14)$ for the
  $2 \times 3$ example (matches the stored output of `A @ x`).
- Identity matrix: ones on the diagonal, columns are the coordinate vectors,
  $\mathbf{I}\mathbf{x} = \mathbf{x}$. Inverse:
  $\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$; the
  scaling of the figure is undone by scaling by $1/2$ and 2; a matrix that
  sends a nonzero vector to zero has no inverse. Pointer to
  `subsec_mdl-inverses`.
- $\mathbf{e}_1 = \frac{1}{2}(1, 1) - \frac{1}{2}(-1, 1)$ and
  $\mathbf{S}^k\mathbf{e}_1 = \frac{1}{2} 3^k (1, 1) - \frac{1}{2}(-1, 1)$.
- $c_j = \mathbf{v}_j^\top\mathbf{x}$ justified by taking dot products with
  $\mathbf{v}_j$; "any vector is a sum" is justified only by "one per
  dimension" (asserted, as before).
- Spectral norm of a symmetric matrix:
  $\|\mathbf{A}\mathbf{x}\|^2 = \sum_i c_i^2\lambda_i^2 \leq \max_i
  \lambda_i^2 \|\mathbf{x}\|^2$, with equality for the top eigenvector. The
  pointer to `sec_mdl-eigendecompositions` in that sentence was dropped; the
  one to `sec_mdl-svd-low-rank` stays.
- "This is modest, but it does not stay so" about $4 \times 10^6$ FLOPs (a
  judgment; no number attached).
- Skip notice: "The book needs it for the analysis of weight decay
  (`sec_weight_decay`) and again in `sec_numerical_stability`." Checked:
  `weight-decay.md` ("Shrinkage Along Eigenvectors") cites
  `subsec_lin-alg-eigenvalues`. Two exercises in the linear-regression
  scratch and concise sections also mention eigenvalues of a Hessian.
- Removed claims: 5 and 680 FLOPs per byte, 0.8 MB, "Hardware for deep
  learning is built for such large products", "Small products ... are
  limited by moving data"; "some call [the order] the rank"; the `mean`
  equals `sum / 2` remark; "Later/throughout the book norms measure the size
  of a gradient, which training cuts back".

## Slides that now disagree with the body

- "**Skips:** inverses, determinants, and the SVD": the body now defines the
  inverse in one sentence.
- Slide "AB Costs 2mkn FLOPs: About 4 Million for the Minibatch": its table
  has columns "data moved (float32)" and "FLOPs per byte"; the body no
  longer makes that argument.
- Final slide: "**Next:** how the error changes when $\mathbf{W}$ changes";
  the body says "when a parameter changes".
