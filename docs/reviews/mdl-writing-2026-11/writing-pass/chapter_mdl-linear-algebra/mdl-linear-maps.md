# Writing pass log: `chapter_mdl-linear-algebra/mdl-linear-maps.md`

Prose only. No code cell, cell ID, figure include, label or slide text changed.
Checker: "OK: only prose changed", no em-dash warning. `lint_source.py`: clean.

## Words

Checker count (to the slides marker, exercises included): 8,879 before, 8,790
after (-89). Body without exercises, my count: 6,654 before, about 6,560
after. The cuts (cofactor expansion, duplicated area/hypothesis statements,
the "We can now" recap, the second classification of orthogonal matrices,
the redefinition of orthonormal columns, cell read-backs; about 700 words)
were largely spent on repairs the review asked for: the invertibility
proposition proved in place, the candid existence paragraph and the direct
proof of the sharper uniqueness statement, the block-triangular rule (EIG-6),
definitions of singular/kernel, and the two unproved basis facts. The review's
target of 7,500 is not reached; the remaining length is in the determinant
subsection and the confirmation cells, both of which need code changes.

## Correctness changes

- **MAP-14** (verified by rereading the derivation; the expansion uses only
  multilinearity, the alternating property and the swap rule, which itself
  uses only those two). Before: the sharper form "f = f(I) det" was proved
  "by uniqueness" before uniqueness was derived; "Conversely, the formula
  does have the three properties" with no proof and no note; "The proof rests
  on one fact". Now the order is: three properties, swap rule, expansion,
  permutation formula ("there is at most one determinant"), then a paragraph
  that says existence is not proved in full (parity taken from Horn and
  Johnson; multilinearity and normalization read off the formula; alternating
  property "we omit the verification"), then the sharper form proved directly
  by running the same expansion on any multilinear alternating f, giving
  f(A) = f(I) det A as a display. The multiplicativity proof now cites that
  statement in words.
- **MAP-9** (the main reorder; no cell moved). "Inverses" now contains
  **Proposition (invertible matrices)**: invertible, independent columns,
  trivial null space, rank n are equivalent, proved on the spot from fact 2,
  fact 3 and rank-nullity (the construction of X is the old step (iii) to
  (ii), moved verbatim). The corollary Q Q^T = I follows in three lines
  directly after it (Qc = 0 implies c = Q^T Q c = 0). "Invertibility and the
  Determinant" shrinks to one proposition, invertible iff det nonzero, with a
  six-line proof that cites the earlier proposition. The paragraph "The proof
  also settles the two claims left open" is gone. Forward references for a
  proof inside the section: nine before, two after (Orthogonal Matrices to
  Inverses for Q Q^T = I; Orthogonal Matrices to Multiplicativity for the
  sign), both now phrased as plain pointers.
- **2 x 2 inverse** (new argument, checked by hand and numerically:
  adj(A) A = (ad - bc) I). The text used to defer "ad - bc != 0 is the
  condition for an inverse" to the determinant theorem. It now shows it in
  place: if ad - bc = 0 the product adj(A) A is zero, and multiplying by an
  inverse of A would give adj(A) = 0, hence A = 0.
- **MAP-6**. The rank-nullity proof now says in a parenthesis that extending
  an independent list to a basis is used without proof. The claim "an
  n-dimensional subspace of R^n is all of R^n" now has its one-line reason
  inside the new proposition (a vector outside, appended to the basis, gives
  n + 1 independent vectors, whose extension to a basis would have more than
  n elements). The geometry section should widen its "(a theorem we use
  without proof)" to include the extension fact; that is the other writer's
  file.
- **MAP-15** (the sign gap). Cofactor expansion is cut to one sentence plus
  the worked 3 x 3 display (kept because `eq_mdl-det-3x3` is cited by an
  exercise). The justification of the sign by row swaps, which used the row
  rule before it was established, is removed; the sign is stated.
- **MAP-11** (stored output: residuals 5.7e-16 and 4.8e-05, errors 2.7e-04
  and 9.0e-03, kappa 1.6e13; my rerun with NumPy 2.5: residuals 6e-16 and
  2e-04, errors 1.2e-03 and 1.3e-02). Before: "it does not predict the loss"
  followed by "as kappa predicts ... whichever method we use". Now: the bound
  allows thirteen digits; the residual of `inv` is "some eleven orders of
  magnitude larger, within the factor kappa"; "either route loses twelve or
  more of the sixteen digits, close to what the bound allows, and in our runs
  `inv` is the worse of the two by about an order of magnitude". The matrix
  norm is now identified as the spectral norm of
  `subsec_lin-algebra-norms`.
- **MAP-20**. "Rounding errors of that size are unavoidable" now reads: the
  entries are stored exactly, but a computation with them commits rounding
  errors of about the machine precision times the largest entry.
- **MAP-16**. Rank n is now one of the four equivalent statements; the
  opening of "Invertibility and the Determinant" no longer claims "three
  tests".
- **Limits cell**: "the perturbed shear has determinant 0" now "0 up to
  rounding" (the stored output prints 0.0; my rerun prints -2e-17).
- No finding disagreed with.

## Six before/after pairs

1. Before: "We check the three images in code." / "The output lists the two
   columns of A and the image (0, -5)."
   After: "The cell below defines A and v, which the code of this section
   reuses, and applies the matrix to e_1, e_2 and v." / "The first two
   outputs are the columns of A."
2. Before: "The code computes the two products and their action on e_1."
   After: "In code the product is the operator `@`, and for the shear and the
   rotation it returns the two composites that we found by following the
   axes."
3. Before: "When does an inverse exist? A square matrix with linearly
   independent columns is invertible; we state this here and prove it in
   [Invertibility and the Determinant]."
   After: "When does an inverse exist? For a square matrix the answer needs
   only the null space and rank-nullity, and the rest of the chapter uses it
   more than any other result of this section."
4. Before: "The two ranks agree for every matrix."
   After: "This is no coincidence."
5. Before: "We can now read a matrix off its columns and compose maps,
   reading their product from right to left. We can decide in three
   equivalent ways ..."
   After: "A nonzero determinant says that a matrix can be undone. It does
   not tell us how *close* the matrix is to collapsing space, or in which
   direction. Two examples show this from opposite sides."
6. Before: (no introduction; the limits cell followed the examples directly)
   After: "Before running it, guess the rank that the library reports for the
   unchanged shear."

Also added to the opening: "Three results carry the section: rank-nullity,
the equivalence of independent columns and invertibility for a square matrix,
and the multiplicativity of the determinant. The construction of the
determinant in n dimensions is the most technical part and can be skimmed on
a first reading." After the triangular cell: "Of all the rules for computing
determinants this is the one to remember."

## Repetition

Removed or reduced to a clause:

- "A layer is a set of 10 normal vectors" in the first sentence (owner: the
  geometry section).
- Orthonormal columns and Q^T Q = I entry by entry (MAP-4; owner: geometry).
  Now one clause with a reference.
- Area zero iff edges collinear, and the hypothesis ad - bc != 0 (MAP-13):
  kept once, at the head of "Invertibility and the Determinant" (the owner).
  The closing paragraph of "Area of a Parallelogram" is now two sentences
  leading to the next subsection.
- Classification of orthogonal 2 x 2 matrices in consequence 4 (MAP-16): now
  one clause with a reference to "Orthogonal Matrices" (the owner).
- The 10^8 shear in "Rank" (MAP-19): now a reference to "Limits of the
  Determinant" (the owner), without the example.
- "Read from right to left": kept in "Composition" (the owner) and in the
  figure caption (cannot be edited); removed from "Limits" and the Summary.
- The "We can now ..." recap at the head of "Limits" (MAP-18), deleted. What
  it held that the Summary lacked (the determinant does not measure nearness
  to singular; the triangular rule) is now in the Summary.
- The definition of linear dependence in fact 2, the description of the
  identity matrix, the 2 x 2 check of the transpose rule, and the read-backs
  after seven cells.

Kept because this file owns them: rank-nullity, row rank, the invertibility
proposition, the determinant and its multiplicativity, the trace and its
cyclic property, the limits of the determinant.

## Ranking; propositions; labels

- No Proposition block demoted (there were four; all are cited later). One
  added: "Proposition (invertible matrices)" in "Inverses". The proposition
  in "Invertibility and the Determinant" was restated as an "if and only if"
  with a shorter proof.
- The sharper uniqueness statement is now an unlabelled display.
- No labels added, none removed.
- The right inverse R in the uniqueness argument was renamed X (MAP-1: R was
  the rotation and the QR factor).

## Split markers

None (not to be split).

## Cells to cut or redesign on the development machine (structural finding 4)

Prose around each is now a sentence; none is presented as evidence.

- `mdl-linear-maps-columns-as-images-of-the-axes`: confirms hand arithmetic,
  but defines `A`, `e1`, `v` for later cells. Keep the definitions, drop the
  output.
- `mdl-linear-maps-composition`: reprints the two composites. Merge.
- `mdl-linear-maps-orthogonal-matrices`: confirms Q^T Q = I, a length and a
  dot product; defines `F`. Merge or cut.
- `mdl-linear-maps-column-space-and-null-space`: three products done by hand;
  defines `B`. Cut the output.
- `geometry-linear-algebraic-ops-invertibility`: the 2 x 2 inverse; defines
  `A_inv`. Merge.
- `geometry-linear-algebraic-ops-determinant`: 5, 5, -5, 0. Cut or merge.
- `mdl-geometry-linear-algebraic-ops-det-3x3`: 8, -8, 6. Cut or replace by a
  log-determinant of a triangular Jacobian.
- `mdl-linear-maps-multiplicativity`: 25, 25, 1, 0.2, -1. Cut or merge.
- `geometry-linear-algebraic-ops-expressing-in-code-2`: four `einsum` strings
  returning numbers already computed; worth keeping only as syntax.

Cells that teach: rank of C, the Hilbert system, the attention scores, the
limits cell.

## Deferred

- **Reorder of the last two `##` sections** (structural finding 2, MAP-17):
  "Limits of the Determinant" should be the last `###` of "The Determinant"
  and "Tensors and Einstein Summation" should follow "Composition" or come
  last. Either move changes the order of cells, so it was not made. In place:
  the trace has its own sentence and a pointer to the eigenvalue section,
  "tensor" is tied to `sec_ndarray`, and the closing claim about contraction
  order has its numbers (n^3 against 2n^2; same numbers as Further Reading).
- **Names for the running examples** (assignment; MAP-1, MAP-12): not done.
  Bold A and B are in six figure captions, in the code (`A`, `B`, `A_inv`)
  and in the slides, none of which I may touch, and A is "the running matrix
  of this chapter" in three other files. The entries a, b, c, d and the edges
  u, w, n of the area formula are also the code's variable names (`a, b, c,
  d = A.ravel()`, `u, w, n`), so MAP-12 was left too. I removed only the
  parenthesis apologizing for d and renamed the right inverse. Needs one
  chapter-wide decision.
- **Change of variables** (structural finding 3): the opening now points to
  `subsec_mdl-change-of-variables` (verified, probability chapter), and the
  triangular and block-triangular rules end with one sentence on flows. The
  full paragraph the review proposes (density divided by |det A|, sum of n
  logarithms) is owned by the probability chapter and was not written here.
  `slogdet` is explained in a clause.
- **Exercises**: only the rank-one-update exercise was changed (u v^T became
  u w^T throughout, eight replacements, checked numerically including both
  block factorizations). Not changed: the first part of "Maps that preserve
  angles", "Rank and nullity of C" (the text still prints the null vectors,
  since the cell needs them), parts two and three of "Determinant and
  near-singularity", the missing `solve` against `inv` exercise.
- **Length**: see Words.
- Exercise and cell label prefixes `geometry-linear-algebraic-ops` remain.

## Factual or mathematical claims added or changed

- Proposition (invertible matrices), four equivalent statements, with proof.
- A 2 x 2 matrix with ad - bc = 0 has no inverse (direct argument).
- Every multilinear alternating f satisfies f(A) = f(I) det A (direct proof).
- Determinant of a block upper triangular matrix is det T_1 det T_2, with a
  proof sketch from the permutation formula (checked numerically on a random
  5 x 5 example). Added for EIG-6.
- Extension of an independent list to a basis is used without proof.
- "A layer of rank r stores r(m + n) numbers in place of mn", with a pointer
  to `subsec_mdl-low-rank-updates` (MAP-7).
- `matrix_rank` "counts the singular values that exceed a small threshold"
  (before: "counts the nonzero singular values").
- "By hand one uses Gaussian elimination, which this chapter does not teach."
- The Hilbert reading (see MAP-11) and the cost n^3 against 2n^2.
- "The rest of the chapter uses it more than any other result of this
  section" (a judgment, about the invertibility proposition).
- Numbers quoted that no cell prints, kept after recomputation: eigenvalues
  2 +- i and singular values 3.618, 1.382 of A; singular values 1e8 and 1e-8
  of the shear; the threshold 4e-8; 169 degrees.

## Slides that now disagree with the body

- "Three Properties Define the Determinant in Every Dimension": the slide
  still leads with cofactors for M; the body now treats cofactor expansion as
  one sentence. Not wrong.
- The slide on solving says both methods "can lose up to log10 kappa, about
  13, digits", which agrees with the body.
- The cover kicker reads "§24.2"; it is rewritten by `gen_slides.py`.
- No slide states anything the body now contradicts.
