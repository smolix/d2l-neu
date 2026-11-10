# Linear Algebra (chapter introduction)

Source: `chapter_mdl-linear-algebra/index.md`. About 400 words of introduction, then an annotated resource list of about 430 words.

## Verdict

A good chapter introduction, close to the template of guide section 12.1 and better than most of the Basics introductions were before their pass. It starts from what the reader has done (matrix products as computations), poses three questions about one concrete object (a Fashion-MNIST image and a 784-to-256 layer), names the central idea in one sentence ("a matrix becomes simple in the right directions"), and says which three facts of deep learning the chapter will explain. The one thing to fix is the route paragraph, which is four consecutive sentences of the form "section X asks ...", the construction the guide singles out.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The first paragraph tells the reader what has been missing so far: "So far we have treated these products as computations and rarely asked what a matrix does to the vectors it acts on." This is the right promise for the chapter and both of the sections I read keep it.
- The three questions of the second paragraph ("How similar are two images? How much information does the layer lose? What happens to a signal that passes through the same map fifty times?") map onto the first three sections, and the closing sentence of that paragraph is quotable: "We cannot draw 784 dimensions, so the answers must come from algebra that keeps the geometry of the plane."
- "The central idea of linear algebra is that a matrix becomes simple in the right directions", followed by three one-line instances (orthonormal axes, eigenvectors, singular bases). An author is ranking the material here. Protect this passage.
- The concrete payoffs are named: the $\sqrt d$ of attention, exploding recurrent gradients, low-rank adapters.
- The announcement of the running matrix and the shear tells the reader to watch for them. In the two sections I read the running matrix is used well (same input $(2,-1)$, same image $(0,-5)$, inverse, determinant 5, trace 4, eigenvalues $2 \pm i$).
- The scope paragraph is two sentences and candid.
- The resource list follows the guide: every entry carries a clause and points to sections by label, and its opening sentence points to the Further Reading notebook.

## Structural findings

None of weight. The introduction is at the length ceiling (about 405 words before the table of contents); any addition must be paid for by a cut.

## Passage-level findings

**IDX-1** (moderate, language). Under the chapter title, route paragraph.
Anchor: `asks what a matrix does to all of space at once`
The route is four sentences in a row with the same skeleton: "X asks how ...", "X asks what ...", "X asks what happens when ...", "X asks how every matrix ...". Guide section 12.1 names this pattern as the one readers skip. The content of the four sentences is good. Fix: group them into two movements and vary the grammar, for example: "The first two sections stay with one application of a matrix: lengths, angles and projections in any dimension (:numref:...), and then what a matrix does to all of space at once, which inputs it sends to zero and when it can be undone (:numref:...). The last two ask what is left when the coordinates are chosen to suit the matrix: repeated application and eigenvalues (:numref:...), and the singular value decomposition, which works for every matrix (:numref:...)."

**IDX-2** (minor, thought). Third paragraph.
Anchor: `resulting equations by elimination. The central idea of linear algebra is`
The Gauss sentence and the "central idea" sentence are unconnected: the anecdote is about least squares and elimination, the central idea is about good bases. Elimination is then never taught in the chapter (the linear-maps section says only "By hand one uses elimination"). Fix: either tie the anecdote to what the chapter does with it ("His problem, more equations than unknowns, is the least-squares problem that opens the chapter and returns with the SVD") or add one clause that turns from history to the idea ("Elimination solves such systems; it does not explain them.").

**IDX-3** (minor, thought). Fourth paragraph.
Anchor: `the first as the typical case and the second as the case where`
The running matrix has the eigenvalues $2 \pm i$ and no real eigenvector. Calling it "the typical case" next to "the case where eigenvectors fail" will puzzle a reader who reaches the eigenvalue section and finds that the typical case has no real eigenvectors either. Check against the eigendecomposition section and say what is meant (diagonalizable, with complex eigenvalues), or drop "typical".

**IDX-4** (minor, language). Second paragraph of the resources.
Anchor: `The resources below accompany the sections of this chapter; each names the`
Fine as is. Noted only because the same sentence opens the list in every chapter of the part; if a reader meets it six times it becomes a template.

## Exercises

None, correctly.

## Still open from October

Not covered by the October review.

## For the chapter review

- The introduction sets up exactly what the first two sections deliver. The promise "these ideas explain why attention divides its scores by $\sqrt d$" is paid in the first section; the other two payoffs belong to the eigenvalue and SVD sections.
- The part introduction (`chapter_mathematics-for-deep-learning/index.md`, 150 words) is weaker than this chapter introduction: it is a list of chapter contents with no reader, no question and no voice. It also states the policy "developed far enough to be useful and no further", which the determinant subsection of the linear-maps section does not follow (cofactor expansion, the permutation formula).
- Section order is right: geometry of one vector pair, then maps, then repeated maps, then the SVD. The first section carries the subspace vocabulary (span, basis, dimension) that the second needs, although its title does not say so.
