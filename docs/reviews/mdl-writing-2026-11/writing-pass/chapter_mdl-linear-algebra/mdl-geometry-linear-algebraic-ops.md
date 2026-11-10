# Writing pass: Vectors, Angles, and Projections

File: `chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md`.
Not split. Code cells, figures, labels and the slides section are unchanged
(`check_mdl_prose.py`: "OK: only prose changed"; `lint_source.py` clean).

## Words

Checker count (prose and exercises, to the slides marker): 7,843 before,
7,820 after. Body alone, with references and figure captions excluded:
about 5,610 before, 5,560 after. The review expected about 600 words to go.
About 550 went, and about 500 came back as repairs the review asked for
(the opening that names the three parts, GEO-8, GEO-14, GEO-16, GEO-17, the
ranking sentences, and cell introductions that state what is computed). Two
display equations went (the triangle-inequality chain, already displayed in
`sec_linear-algebra`, and none other; the count is in `$$` delimiters).

## Correctness changes

- **GEO-8** (projection theorem used a basis built later). Was: orthonormal
  basis defined, proposition stated, applied to any column space; Gram-Schmidt
  afterwards "needs independence". Now: at the definition, "Such a basis
  always exists: :numref:`subsec_mdl-gram-schmidt` builds one from any finite
  list of vectors that spans $S$ ... and it needs nothing beyond projection
  onto a line"; in Gram-Schmidt, "For an orthonormal basis of the span alone
  we skip such a vector and continue, so any spanning list will do. This is
  the existence claim we used for the projection onto a subspace." Checked:
  the Gram-Schmidt subsection uses only the projection onto a line, so there
  is no circle; skipping a zero remainder leaves the span of the $\mathbf{q}$s
  equal to the span of the vectors processed so far. The claim is made for
  finitely spanned subspaces only (that every subspace of $\mathbb{R}^n$ is
  one is not proved in this section, and I did not assert it).
- **GEO-5** (wrong equation reference). Was: "By :eqref:`eq_mdl-dot_geom`,
  every point on it has the same signed shadow". Now: "Every point
  $\mathbf{x}$ on it has $\mathbf{w}\cdot\mathbf{x} = -b$, so its shadow on
  $\mathbf{w}$ has the same signed length :eqref:`eq_mdl-signed-length`".
  Label `eq_mdl-signed-length` added to the signed-length display.
- **GEO-16** (dimension of a hyperplane). Was: "It has dimension $d - 1$".
  Now: "It is a shifted copy of the subspace $\mathbf{w}\cdot\mathbf{x} = 0$,
  which has dimension $d - 1$ (:numref:`subsec_mdl-rank`)". Rank-nullity is
  in that subsection (checked). The later clause "$b = 0$ gives a hyperplane
  through the origin, which is a subspace" was dropped as now redundant.
- **GEO-14** ("positive dot product"). Now "no pair has a negative dot
  product".
- **GEO-2** (opening argued from the classifier to the geometry). Now: "That
  so simple a rule works says that the two clouds of points lie well apart,
  and that 'closer' can be computed where it cannot be seen."
- **GEO-1** (promise to check each result on the images). Removed; the
  opening now says the third part takes the results to the images.
- **GEO-17** (explanation not shown by the output). Was: "The means of these
  pairs are close, while the images of each class spread widely". Now: "The
  likely reason is that the means of such pairs are close relative to the
  spread of each class along $\mathbf{w}$ ..., though this cell does not
  measure it." The question before the cell no longer presupposes it
  ("classes that look alike" instead of "whose means are close").
- **GEO-15** ("predicts by the sign" is not what the softmax layer does).
  The paragraph is gone from the nearest-mean subsection; its content is in
  "Layers as Stacked Hyperplanes" with "It predicts the class with the
  largest score".
- **GEO-10**: "squares the sensitivity" is now "squares the condition number
  of the problem, its sensitivity to rounding errors
  (:numref:`subsec_mdl-condition-number`)".

No finding was rejected.

## Before and after

1. Before: "If the pictures we draw in the plane did not carry over to $784$
   dimensions, a rule as simple as 'nearest mean' would have no reason to
   work."
   After: "That so simple a rule works says that the two clouds of points
   lie well apart, and that 'closer' can be computed where it cannot be
   seen."
2. Before: "Does the formula return $\pi/4$ on the worked pair, and what
   does it give on a pair in $\mathbb{R}^3$ that we have not drawn?"
   After: "In code the definition takes two lines, tried here on the worked
   pair and on a pair in $\mathbb{R}^3$ that we have not drawn."
3. Before: "Does the two-step process give $(0.6, 0.8)$ and $(0.8, -0.6)$,
   and what does `np.linalg.qr` return?"
   After: "So when we run the two steps above next to `np.linalg.qr`, we
   should expect the same columns, possibly negated."
4. Before: "How often is the nearest-mean hyperplane right on the $2{,}000$
   test images of the two classes?"
   After: "With the two means only $33^\circ$ apart, one might not expect
   much of the rule. But the classifier does not need the means to be
   orthogonal. It uses only their difference. We apply it to the $2{,}000$
   test images of the two classes."
5. Before: "the images vary along far fewer effective directions than $784$."
   After: "By the proposition, it is the spread of random directions in
   eight or nine dimensions ($1/0.34^2$ lies between $8$ and $9$). Measured
   by their cosines, the centered images behave like a cloud in fewer than
   ten dimensions, not $784$, a first sign of the low-rank structure of
   :numref:`sec_mdl-svd-low-rank`."
6. Before: "We can now measure an angle in any dimension, drop a vector onto
   a line or a subspace, fit a line by projection, and classify a
   $784$-dimensional image by the side of one hyperplane. We cannot yet
   describe what a matrix does to all vectors at once. The row view shows
   ..."
   After: "The row view shows a layer as a stack of hyperplane tests, but it
   does not tell us which inputs the map destroys, or whether its action can
   be undone. Reading the same product by columns answers these questions in
   :numref:`sec_mdl-linear-maps`."

The two question-then-answer pairs the review lists as best ("Do two
unrelated Fashion-MNIST images look orthogonal?" / "They do not." and "Does
it survive ...?" / "It does not.") are kept; all other cell questions were
replaced.

## Repetition

Removed or reduced to a clause with a reference:

- With `sec_linear-algebra` (GEO-4, GEO-12): the list of what the Basics
  did (now one clause in the roadmap); the law-of-cosines paragraph (one
  sentence; the figure include had to stay); "turn the planar identity
  around"; the Basics' algebraic proof ("from $\|\mathbf{u}-\mathbf{v}\|^2
  \ge 0$"); the sentence reading :eqref:`eq_mdl-dot_geom` in words; the
  first paragraph of "Cosine Similarity" (now the display and one clause;
  the dimmed image is kept in that clause); the triangle-inequality proof
  (the same display is in the Basics; now a sentence).
- A layer as hyperplanes (GEO-15): said once, in "Layers as Stacked
  Hyperplanes" (the owner). Removed from the head of "Hyperplanes and
  Decision Boundaries" and from the nearest-mean subsection.
- Condition number: this file never defined it. The one remark that the
  Gram matrix squares it now points to `subsec_mdl-condition-number` (owner:
  SVD section). The classical-Gram-Schmidt remark stays as one sentence with
  its exercise.
- "Least squares leaves the target unchanged ..." (GEO-9): cut.
- Closing recap (GEO-18): cut; Summary reduced from eleven sentences to
  five.
- The column-vector convention parenthesis, the foot of the perpendicular
  $(0.4, 0.2)$, the second statement of the origin's distance: cut.

Kept because this file owns them: orthonormal columns and
$\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ entry by entry; the proof of
Cauchy-Schwarz by projection; the objection paragraph; the clip; the
subspace vocabulary; a layer as stacked hyperplanes.

## Ranking

- Demoted from blocks to sentences: "Proposition (coordinates are unique)"
  and "Corollary (triangle inequality)". Neither had a label.
- Kept as Proposition blocks: orthogonal projection, Cauchy-Schwarz,
  projection onto a subspace, near-orthogonality.
- Added: "Of these results, three matter most ..." in the opening; "The
  proposition that follows is the result of this section that the rest of
  the chapter uses most"; "A reader who knows the standard definitions can
  go straight to the next subsection"; "The clip in `angle` is the part
  worth remembering."
- Defined terms are now all italic (they were bold for some and italic for
  others, GEO-7); the other sections of the chapter use italic.

Label added: `eq_mdl-signed-length`.

## Split

None (not to be split). The opening now names the three parts.

## Cells to cut or redesign on the development machine

- `mdl-geometry-linear-algebraic-ops-near-orthogonality-of-random-directions-1`:
  confirms the proposition and the variance $d$ by sampling. Kept as the
  section's one such check and presented as the reference for the image
  cell; could be merged into the image cell.
- `geometry-linear-algebraic-ops-dot-products-and-angles` (GEO-6): replace
  the $\mathbb{R}^3$ pair by the collinear pair with and without the clip so
  that the `nan` is shown.
- `mdl-geometry-linear-algebraic-ops-the-nearest-mean-classifier-2`
  (GEO-17): print $\|\mathbf{w}\|/2$ and the two spreads per pair; the prose
  can then state the reason instead of calling it likely.
- Imports cell (GEO-19): reduce to `numpy` and the `d2l` import.
- GEO-1: one new cell projecting a test t-shirt onto the mean t-shirt
  (deferred, new cell).

## Other deferred findings

- GEO-4: remove :numref:`fig_mdl-la-angle` (a figure include cannot change
  in this pass; the prose around it is one sentence now).
- Exercise "Tilting the boundary": not moved; a parenthesis now names the
  later subsections it needs (`subsec_mdl-inverses`, `subsec_mdl-psd`,
  `subsec_mdl-condition-number`).
- GEO-3 notation: the dot and the transpose are both kept, with one clause
  saying when each is used. $\mathbf{w}$ is the hyperplane normal here and a
  parallelogram edge in linear maps (chapter review); not changed.

## Exercises changed

- Low-dimensional cosine (GEO-13): part 1 now asks to confirm the two
  probabilities quoted in the text (the "why" was answered in the text);
  part 3 now ends "For which $d$ is $|\cos\theta| < 0.1$ more likely than
  $|\cos\theta| > 0.9$?" (checked numerically: tie at $d = 3$, true from
  $d = 4$).
- Random cosines: the text no longer gives "about $0.3$ for $n = 10^5$,
  $d = 256$", which the exercise asks for.

## Claims added or changed

- "Such a basis always exists" for finitely spanned subspaces (above).
- "$0.34$ ... is the spread of random directions in eight or nine
  dimensions": $1/0.34^2 = 8.65$; the stored output is $0.346$, giving
  $8.35$.
- "Each class sits less than two of its standard deviations from the
  boundary, so errors of several percent are no surprise": means at about
  $\mp 3.2$, standard deviations $2.36$ and $1.66$ (stored), so $1.4$ and
  $1.9$ standard deviations; the errors are 10.1% and 6.8%.
- "which for a large index is several times the typical size $1/\sqrt{d}$":
  $\sqrt{2\ln n}$ is $3.7$ at $n = 10^3$ and $4.8$ at $n = 10^5$.
- "On this small problem the two routes agree" (stored output).
- Numbers no cell prints that were kept after recomputing: the `nan` and
  $1 + 2.2\times10^{-16}$ for the collinear pair; softmax weights $0.9997$
  and $0.73$; $0.29$ and $0.06$ in $d = 2$; $\|\mathbf{w}\|/2 \approx 3.2$
  (the reviewer's value, not recomputed here since the data are not on this
  machine).

## Slides section against the body

- The slides still give "about $0.3$ for $n = 10^5$, $d = 256$" for the
  largest unrelated cosine; the body no longer quotes it. No contradiction.
- Nothing in the slides contradicts the revised body that I could find by
  searching for the changed statements.
