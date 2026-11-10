# Writing pass: chapter introduction and Further Reading (chapter 38)

Files: `chapter_mdl-linear-algebra/index.md`,
`chapter_mdl-linear-algebra/further-reading.md`. No code cells, figures or
labels in either; the toc block and all labels are untouched. Checker: "OK:
only prose changed" for both, no em-dash warning; `tools/lint_source.py`
clean (exit 0) for both.

## Words

Checker count (it counts each `:numref:` as about three words):

| File | Before | After |
|---|:-:|:-:|
| `index.md` (introduction and resource list) | 946 | 945 |
| `further-reading.md` | 1906 | 1884 |

Introduction alone, to the toc block: 409 to 424 by the checker's count; 397
to 400 with the reference tokens removed (three label references were
added, which is most of the difference). The resource lead-in pays for it:
the file as a whole is one word shorter.

## Introduction

- **Route (IDX-1).** The four "X asks ..." sentences are now two movements:
  the first two sections "apply a matrix once"; the rest "chooses
  coordinates to suit the matrix". The eigendecomposition reference is
  attached only to "eigenvectors for a square matrix applied repeatedly";
  the clause "then what eigenvalues leave unsaid" carries no reference, so
  after the split it can receive the label of "Spectra in Practice" with no
  other change. No section numbers are named.
- **What the chapter is for.** First paragraph now says what
  :numref:`sec_linear-algebra` did (compute) and what this chapter adds
  (what a matrix does, with proofs), and the author's rule in the chapter's
  terms: "It is additional reading, and the main text does not depend on
  it."
- **Ranking (new).** "Three results matter most": projection gives the
  closest point, the spectral theorem, Eckart-Young. This is my choice from
  the nine results the four sections rank in their own openings; the
  geometry section also names Cauchy-Schwarz and the $1/\sqrt d$ cosine,
  linear maps names rank-nullity. The author may prefer another three.
- **What can be skipped (new).** The determinant in $n$ dimensions (skim),
  Gershgorin discs, PageRank, pseudospectra (skip). Each agrees with what
  the section openings now say (`mdl-linear-maps.md` line 40,
  `mdl-eigendecomposition.md` lines 1074 to 1077). I did not list "The SVD
  in Deep Learning", which the SVD section marks as skippable on a first
  reading, because the introduction names its content as a payoff.
- **Uses in the main text.** Added `sec_attention-scoring-functions` (the
  $\sqrt d$) and `sec_bptt` (exploding gradients); both labels exist and are
  the ones the sections cite. The adapter payoff has no reference:
  `sec_lm_lora` exists but is a skeleton chapter.
- **IDX-2.** The Gauss sentence now ties to the chapter: least squares is "a
  problem the first section solves and the last takes up again" (projection
  in the geometry section, pseudoinverse in the SVD section). "Solving the
  resulting equations by elimination" and "Linear algebra grew out of
  solving systems of linear equations" are gone, so the introduction no
  longer names elimination as the origin of a subject that never teaches
  it.
- **IDX-3.** "the first as the typical case and the second as the case where
  eigenvectors fail" is now "with complex eigenvalues" and "with too few
  eigenvectors" (eigenvalues $2\pm i$: trace 4, determinant 5; the shear
  has the double eigenvalue 1 and one eigenvector direction). The
  parenthetical $\mathbf A=$ is dropped from the display because the
  sections use bold $\mathbf A$ for several matrices (deferred notation
  finding MAP-1/EIG-3).
- **IDX-4 and the name clash.** The lead-in of "Resources and Further
  Reading" now states its purpose in one sentence ("teach this chapter's
  material again, at greater length or from another side") and gives the
  other file one sentence ("What lies beyond the chapter is the subject of
  ..."). The heading itself is unchanged; the same heading is used in all
  six chapters, so renaming it is a part-wide decision.
- Protected: the second paragraph (three questions, "We cannot draw 784
  dimensions ...") and "a matrix becomes simple in the right directions"
  with its three instances, verbatim.

## Further Reading

- **Opening (FR-1).** The template paragraph, including "Each topic states
  the problem it addresses ...", is replaced by: where the chapter stops,
  the order of the entries, the purpose of this file against the list in
  the introduction, and a ranking.
- **Ranking ("Flat").** First for deep learning: random projections,
  iterative solvers, random matrices and trainable depth, sketching, then
  the last two entries (low rank in attention, steepest descent under other
  norms), which point into the main text. For the curious: tensor
  decompositions, the QR algorithm, the Kreiss theorem, mixing times. The
  first four are the section reviewer's list; the rest is my judgment.
- **Merged topic.** "Elimination, Determinants, and Rank in Practice" and
  "Sparse and Iterative Solvers" are one entry, "Solving Linear Systems at
  Scale" (the review's proposal). Kept: the two citations (Trefethen and
  Bau, Higham) and all of the sparse and iterative text. Dropped because
  the sections teach it: LU behind `solve`, the determinant as the product
  of pivots, numerical rank from singular values, and the
  $10^6\times10^6$ example (it is in `subsec_mdl-numerical-issues`, now
  cited). Fourteen entries remain. Neither heading had a label.
- **FR-2.** "The one difference from matrices is that", "The one change is
  to", "changes the initialization", "changes the parameterization instead"
  are reworded; no two entries share the frame.
- **FR-5.** "Parzen-window classifier" is replaced by what it means: "every
  training point votes with weight $k(\mathbf x,\mathbf x_i)$".
- Matrix completion: the opening is shortened by a clause; the recovery
  theorem and Soft-Impute stay.

### Correctness changes

| ID | Before | After | Check |
|---|---|---|---|
| FR-3 | LoRA "starts its update $\mathbf B\mathbf A$ at zero, so early fine-tuning moves along random directions" | "$\mathbf B=\mathbf 0$ and a random $\mathbf A$, so the first updates see the input only through a random $k$-dimensional subspace, unrelated to the pretrained weight" | With $\mathbf B=\mathbf 0$ the gradient with respect to $\mathbf A$ is $\mathbf B^\top(\cdot)=\mathbf 0$, so only $\mathbf B$ moves at first and $\Delta\mathbf W\mathbf x=\mathbf B(\mathbf A\mathbf x)$ depends on $\mathbf x$ through $\mathbf A\mathbf x$ |
| FR-4 | Mixing bounded by $1/(1-\lambda_\star)$, then Cheeger on "the gap $1-\lambda_2$" with no link | Added: for a lazy chain (holding probability at least $1/2$) no eigenvalue is negative and $\lambda_\star=\lambda_2$ | $\mathbf P=(\mathbf I+\mathbf Q)/2$ with $\mathbf Q=2\mathbf P-\mathbf I$ stochastic and reversible, eigenvalues of $\mathbf Q$ in $[-1,1]$ |
| FR-4 | "approaches its stationary distribution at rate $1/4$" | "error ... shrinks by a factor of 4 per click, because its second eigenvalue is $1/4$" | Wording of the section's figure caption |
| FR-6 | "the weights of an untrained network follow it" | "... with independent random initialization ..." | Orthogonal initialization has all singular values 1 |
| mine | Kolda and Bader's survey "covers these decompositions" (CP, Tucker, tensor train) | "covers the CP and Tucker decompositions" | The survey is from 2009, the tensor train from 2011 |
| mine | QR algorithm: "the diagonal converges to the spectrum" | Added "(a complex pair of a real matrix remains as a $2\times2$ block)" | Real Schur form |
| mine | Golub and Van Loan "cover both families" (sparse direct and iterative) | "treat the iterative methods in detail" | The cited 1996 edition has a chapter on iterative methods; I am not certain how much it gives on fill-reducing orderings |
| mine | Levin and Peres "prove the Perron--Frobenius facts" that the PageRank subsection cites | "develop this theory, including the convergence facts" | They prove existence, uniqueness and convergence for stochastic matrices, not the general theorem |
| mine | PiSSA: "training adjusts the principal directions first" | "from the first step" | The principal part is trainable from the start; "first" implied an order |

Claims about papers left as they were and checked only from memory (titles
confirmed in `d2l.bib`, contents not): Dasgupta and Gupta (short proof with
explicit constants); the Kreiss bracket with the constant $e\,n$; Saxe et
al. (depth-independent learning time with orthogonal weights);
Pennington et al. (dynamical isometry with tanh and orthogonal weights, not
with ReLU); Candès and Recht (incoherence condition); Mahoney and Drineas
(leverage scores, constant-factor guarantee); Woodruff ($1+\varepsilon$
regression); Linformer; Bernstein and Newhouse (Adam and Shampoo without
moving averages). I am confident of each. One caveat: the sentence on
ranking models by the tail exponent describes later work by Martin and
Mahoney, not the cited 2021 JMLR paper; the text does not attribute it to
that paper, so I left it.

## Before and after (verbatim)

1. Before: ":numref:`sec_mdl-linear-maps` asks what a matrix does to all of
   space at once: which inputs it sends to zero, when it can be inverted,
   and by how much it scales volume." After: "The first two sections apply
   a matrix once."
2. Before: "So far we have treated these products as computations and rarely
   asked what a matrix does to the vectors it acts on." After:
   ":numref:`sec_linear-algebra` taught how to compute these products and
   rarely asked what a matrix does to the vectors it acts on. This chapter
   does, with proofs."
3. Before: "the first as the typical case and the second as the case where
   eigenvectors fail." After: "with complex eigenvalues, and the shear ...,
   with too few eigenvectors."
4. Before: "Each topic states the problem it addresses and what it changes
   relative to the chapter." After: "The entries are not of equal weight."
5. Before: "The one difference from matrices is that Eckart--Young has no
   analogue:" After: "Eckart--Young has no analogue for tensors:"
6. Before: "The resources below accompany the sections of this chapter; each
   names the section it goes with." After: "The books, courses, and notes
   below teach this chapter's material again, at greater length or from
   another side; each names its section, and most are free online."

## Deferred

- Pseudospectra entry: still opens with ":numref:`subsec_mdl-pseudospectra`
  showed", which is right while the subsection stays in the section (the
  section's pass kept it; see its log, EIG-16). If it moves here, the entry
  needs the definition and the numbers 200.3 against 10.
- Optional new entries (gradients through `eigh`/`svd`, graph Laplacian):
  not added; the file must not grow.
- The heading "Resources and Further Reading" against the file title
  "Further Reading": the two now state different purposes, but the names
  still overlap. A rename ("Resources") would have to be made in all six
  chapters.
- The other five further-reading files should rank the same way so that the
  six openings do not become a new template; this one uses "The entries are
  not of equal weight. For deep learning, read ... first".
- After the eigendecomposition split: add the new section's label to "then
  what eigenvalues leave unsaid" in the route, and check the three "skip"
  items against the new section's opening.
