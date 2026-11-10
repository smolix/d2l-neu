# Writing pass: `chapter_mdl-linear-algebra/mdl-svd-low-rank.md`

Date: 2026-11-09. Base: `4deb96d6`. Not split. No code cell, cell ID, figure
include, equation label or slide changed. `check_mdl_prose.py`: "OK: only
prose changed", no em-dash warning. `tools/lint_source.py`: clean.

## Words

Checker count (to the slides marker, exercises included): 10,240 before,
10,074 after (-166). Body to the Summary: 8,034 to 7,920. The cut is smaller
than the review's "about 1,000" because the pass also added what the review
asked for (ranking, the shapes of the three SVD variants, the step from one
principal direction to $k$, the two-vector power iteration, the announcement
that the Eckart-Young proof has two halves, a checkable dimension formula),
about 280 words in all. The remaining cuts the review lists need cells
removed (SVD-4) and are deferred.

## Correctness

- **SVD-9** (overstatement). Was: "Weight decay in :numref:`sec_weight_decay`
  is ridge regression for neural networks." Now: "For a linear model with
  squared loss, the weight decay of :numref:`sec_weight_decay` is ridge
  regression, and the filter factors show which directions it shrinks."
  Verified by the definition of ridge; the reviewer's wording is used.
- **SVD-8** (numbers no cell prints). All three recomputed from
  `img/cat1.jpg` with the cell's grayscale weights and **kept**: energy
  0.9554 at rank 4 with relative error 0.211 ("95%", "21%"); after
  subtracting the mean pixel value, 95% is first reached at rank 8 (also rank
  8 with column means removed); least-squares slope of $\log\sigma_i$ against
  $\log i$ over $i\le100$ is $-1.107$ ("$i^{-1.1}$"). Also rechecked: 0.886
  for $\sigma_1^2$ (printed), ridge factors 0.9, 0.5, 0.0826, the singular
  values 2.00005 and $5\times10^{-5}$ of $\mathbf K$ and the relative change
  $3.5\times10^{-5}$ of $\mathbf b$.
- **SVD-7** (proof step too fast to check). The one-clause justification of
  the dimension formula inside the Eckart-Young proof is now three short
  sentences: extend a basis of $\mathcal K\cap\mathcal V$ to bases of
  $\mathcal K$ and of $\mathcal V$; the vectors obtained span the sum and are
  independent; count them. Checked by writing out the independence argument
  (a combination of the $\mathcal V$-extension vectors lying in $\mathcal K$
  lies in the intersection). The rest of the proof is untouched.
- **SVD-14** (prose read the cell for more than it shows). The question "Is
  low rank a property of matrices or of the data they hold?" and the
  conclusion "therefore depends on what it holds" are gone. The text now says
  the second matrix is built to have the spectrum $e^{-i/12}$, that its
  numbers follow from the construction (rank 18 is $6\ln 20=17.97$ rounded
  up, verified), and that neither matrix is a trained weight.
- **SVD-11** (bowl). The reduced paragraph adds the hypothesis "with
  $\mathbf A$ of full column rank" for the ellipsoidal level sets (without it
  the level sets are cylinders). Semi-axes proportional to $1/\sigma_i$ along
  $\mathbf v_i$: checked from
  $(\mathbf x-\hat{\mathbf x})^\top\mathbf A^\top\mathbf A(\mathbf x-\hat{\mathbf x})=c$.
- Stored outputs: no disagreement found. The Newton-Schulz draw prints a
  distance of $9.9\times10^{-8}$; the prose's "within $10^{-5}$" is true of
  every draw (the stopping rule bounds the Frobenius distance by
  $2\times10^{-6}$), so it stays.
- No finding disagreed with.

## Six before/after pairs

1. Before: "Does the library reproduce the hand computation? A library may
   return any of the sign choices discussed below, so the cell fixes the
   signs that put $\mathbf{u}_1$ in the first quadrant and make $\mathbf{U}$
   a rotation before printing the factors and their angles."
   After: "A library call should return the same factors, though possibly
   with other signs (:numref:`subsec_mdl-svd-uniqueness`). The cell therefore
   first fixes the signs that put $\mathbf{u}_1$ in the first quadrant and
   make $\mathbf{U}$ a rotation."
2. Before: "**The photograph.** On the photograph, is the spectral error
   equal to $\sigma_{k+1}$, what does each rank cost, and does a plausible
   competitor do as well?"
   After: "Now the photograph. The theorem fixes the spectral error of every
   truncation in advance, at $\sigma_{k+1}$. It does not say how fast the
   error falls with $k$, nor how far a plausible competitor of the same rank
   falls behind."
3. Before: "Do the SVD axes of a fresh cloud of this kind capture the
   variance the proposition promises, and is the discarded variance the
   reconstruction error?"
   After: "For such a cloud the first direction should hold about
   $9/9.64=93\%$ of the variance and lie along the long axis. How closely 300
   points reproduce these population values is what the cell measures on a
   fresh draw."
4. Before: "Does this prediction hold for a random $6\times4$ momentum
   matrix?"
   After: "The estimate is rough, a linear phase glued to a quadratic one, so
   we test it on a random $6\times4$ matrix standing in for the momentum."
5. Before: "The same lower bound yields the inequality on which the Frobenius
   case rests."
   After: "Half of the theorem is still unproved. The lower bound just shown
   yields the inequality on which the Frobenius case rests."
6. Before: "The computations of this section use NumPy. Each framework's own
   `svd` routine appears once, to show what it returns."
   After: "Two results carry the section: the existence of the decomposition,
   and the Eckart--Young theorem, by which cutting the list short gives the
   best approximation of each rank. The pseudoinverse and the condition
   number follow from the first in a few lines. The last part, on the SVD in
   deep learning, can be skipped on a first reading. The computations use
   NumPy, and each framework's own `svd` routine appears once, to show what
   it returns."

Of the ten cells, one is still introduced by a question, the prediction "What do the 'zero' singular values of a
$4\times4$ matrix of rank two look like?" (numerical rank).

## Repetition

Owned here and kept:

- **Condition number** (chapter table: home is this section). Definition,
  proposition, the $2\times2$ system and the bound-against-reality passage
  are untouched. Added after the definition: for a symmetric positive
  definite matrix $\kappa=\lambda_{\max}/\lambda_{\min}$, with a reference to
  :numref:`subsec_mdl-rayleigh`.
- **$\kappa(\mathbf A^\top\mathbf A)=\kappa(\mathbf A)^2$** (home: here,
  where it is proved). The proof stays in "The Condition Number". The two
  earlier mentions ("Computing the SVD", "The normal equations") were already
  one clause with a reference and stay; the Summary no longer repeats it.
- **Ridge filter factors** :eqref:`eq_mdl-ridge-filter` and the **two-line
  ill-conditioned system** (chapter 40 review: home is chapter 38). Kept in
  full.
- **Weyl's inequality** (lemma here), **Eckart-Young**, **the photograph's
  counts**.

Reduced to a clause with a reference:

- **Marchenko-Pastur** (home: eigendecompositions). "Truncation as denoising"
  no longer re-derives the eigenvalue interval; it cites
  :numref:`subsec_mdl-sample-covariance-mp` and states the singular-value
  interval once. After the spectrum cell the interval is cited from that
  paragraph instead of being attributed to the law a second time.
- **The bowl, the rate $(\kappa-1)/(\kappa+1)$ and the zig-zag** (home:
  Optimization appendix). The paragraph keeps only what is new here (contours
  of least squares have axis ratio $\kappa(\mathbf A)$, the Hessian
  $\kappa(\mathbf A)^2$) and points to
  :numref:`sec_mdl-gradient-based-optimization`. The figure include stays
  (not mine to remove) and is cited in one clause.
- **Newton-Schulz against the main text** (SVD-13). The momentum motivation
  paragraph and the paragraph on Muon and steepest descent are replaced by
  one opening paragraph with :numref:`sec_muon`; the quintic, its
  coefficients and its band are replaced by one sentence with
  :numref:`subsec_muon-newton-schulz`. Kept: polar decomposition with the
  shear, the proposition with proof, the basin, the step count and its cell.
  The citation of `Bernstein.Newhouse.2024` left this file (it is in
  `muon.md` and in Further Reading).
- **Randomized SVD** (home: Further Reading and the exercise). Cut from 150
  words to 80: the range finder in one sentence (the term is kept because
  `further-reading.md` and the exercise cite it from here), why it works, and
  references to the exercise and to
  :numref:`sec_mdl-linear-algebra-further-reading`.
- **Inside the file.** The closing paragraph of the first subsection
  (restating the opening's "stop asking one basis to serve as both") is
  deleted; the after-cell text of the condition-number cell no longer repeats
  the bound 1.41 and the realized 1.0 given just before it; the shear
  factorization and the 200-against-18 numbers left the closing and the
  Summary.

## One ending

The two "We can now ..." sentences are deleted. The three limits (cost, low
rank cannot be assumed, linear structure only) remain as the ending, with the
circle sentence last. The Summary went from twelve sentences to five.

## Ranking; Proposition blocks; labels

- A ranking paragraph closes the introduction (pair 6). The Eckart-Young
  proof now says at its head that it has two parts.
- No Proposition block was demoted. The candidates were the odd-polynomial
  proposition (three-line proof), which the review asks to keep and which
  `muon.md` should cite, and Weyl's lemma, which the exercises cite.
- **"Existence and Uniqueness" is now three subsections** (structural
  finding 1): "Existence" (keeps `subsec_mdl-svd-via-ata`), "Uniqueness and
  the Largest Stretch", "Singular Values and Eigenvalues". No cell moved.
  **Labels added:** `subsec_mdl-svd-uniqueness`,
  `subsec_mdl-svd-eigenvalues`.
- Run-in titles removed where they only labelled a paragraph ("Example: the
  shear.", "Uniqueness.", "The maximum stretch.", "Relationship to the
  eigendecomposition.", "The photograph.", "A competitor.", "The smallest
  case."). Kept where a reader looks the term up (compact SVD and dyadic sum,
  computing the SVD, numerical rank, norms as singular values, energy ratio,
  denoising, centering, normal equations, polar decomposition, Newton-Schulz,
  spectral normalization, measuring the spectrum, randomized SVD).

## Other findings applied

- SVD-1: the photograph is named as a cat in the opening, and the text after
  the cell says what the figure shows at ranks 1, 5 and 20 (checked against
  the rendered `img/mdl-la-eckart-young.svg`).
- SVD-3: shapes of full, thin and compact SVD for a $3\times2$ matrix of
  rank 2.
- SVD-4 (prose part): the four framework tabs are cut to two or three sentences each;
  the lead-in is a statement.
- SVD-5: the circularity paragraph is reduced to one sentence pointing to
  Exercise 2a.
- SVD-10: "orthogonal initialization of weights is attractive" now gives the
  reason (every input keeps its length).
- SVD-12: the step from one principal direction to $k$.
- SVD-15: the two-vector form of power iteration, consistent with Exercise 9;
  the reference now goes to `subsec_mdl-power-iteration`.
- SVD-17: "Exercise" capitalized in mid-sentence throughout (as in the
  eigendecomposition section); "problem :exref:" in Exercise 9b likewise.
- SVD-18: the four tabbed placeholder links `https://d2l.discourse.group/t/svd`
  are removed; the real link stays.

## Cells to cut or redesign on the development machine

- `mdl-svd-low-rank-existence-and-uniqueness-1` (SVD-4): prints three shapes
  and `True`; cut, together with the four tabs.
- `svd-weight-spectrum` (SVD-14): the "decaying" half confirms its own
  construction; replace it by the photograph (already in memory) or a trained
  weight or fine-tuning update.
- `svd-pca` (review, Exercises note on 5e): a synthetic cloud whose result is
  predictable; consider a short Fashion-MNIST version.
- `svd-defective-shear`: confirms the hand computation; the prose around it
  is now three sentences. Worth keeping once, as the first library call.
- `mdl-svd-low-rank-rotate-scale-rotate` (SVD-8, SVD-2): print the energy at
  rank 4 and the rank for 95% after centring; rename `F` to `fro`.
- `svd-imports` (SVD-2): unused imports in all four tabs.

## Other deferred findings

- SVD-6 and the chapter's finding 8 (own letters for the shear, the running
  matrix and LoRA's factors): shared symbols, not changed. The apology for
  LoRA's $\mathbf A$ stays.
- SVD-7, second half: the dimension formula would sit better next to
  rank-nullity in `mdl-linear-maps.md`; it is proved here in three sentences
  until then.
- SVD-13, other direction: `chapter_optimization/muon.md` should cite
  :eqref:`eq_mdl-odd-polynomial` for the proof (main-text file, not edited).
- Exercises: 7a (restates the text), splitting 1 and 5, the two missing easy
  problems. Not errors, left.
- The review's suggestion to shrink "The SVD in Deep Learning" by half is
  about two thirds done; the rest is the spectrum cell.

## Notation clashes (logged, not changed)

- **With chapter 40** (`mdl-numerical-stability-conditioning.md`): the ridge
  filter factors are :eqref:`eq_mdl-ridge-filter` here with penalty $\gamma$,
  unknown $\mathbf x$, target $\mathbf b$ and no factor $\tfrac12$; there
  they are :eqref:`eq_mdl-opt-filter-factors` with $\lambda$ and $\mathbf w$.
  The two-line system $x_1+x_2=2$, $x_1+1.0001x_2=2.0001$ appears in both;
  here the matrix is $\mathbf K$. This section is the home of both.
- $\mathbf A$: generic matrix, shear, $3\times2$ and $3\times4$ examples,
  running matrix, LoRA's down-projection. $\mathbf P$: projector in PCA,
  polar stretch. $\mathbf Q$: orthogonal block in the uniqueness list, polar
  factor, range-finder basis. $n$: columns everywhere, rows in PCA (flagged
  in the text). $\mathbf B$: competitor in Eckart-Young, the singular
  $2\times2$ example, the signal in denoising, LoRA's factor.

## Factual or mathematical claims added or changed

1. Shapes of the full and thin SVD of a $3\times2$ matrix of rank 2, and that
   the compact SVD equals the thin one there.
2. Exercise 2a "builds the SVD from the maximum stretch, without the spectral
   theorem" (was: "gives a proof that avoids it").
3. The three-sentence proof of the dimension formula.
4. "the cat is a bright blur at rank 1, a pale body with four paws at rank 5,
   and plainly a cat lying on its back at rank 20" (from the figure).
5. "the singular values of a large noise matrix concentrate in the interval
   $[(\sqrt n-\sqrt m)\sigma_{\text{noise}},(\sqrt n+\sqrt m)\sigma_{\text{noise}}]$"
   (same interval as before; "concentrate in" replaces "fill" for the
   eigenvalue statement that was removed).
6. Among unit vectors orthogonal to $\mathbf v_1$ the variance is largest
   along $\mathbf v_2$ (Rayleigh on the orthogonal complement).
7. The PCA numbers agree "because they are one quantity computed twice".
8. SVD-9 wording on weight decay.
9. For symmetric positive definite matrices
   $\kappa=\lambda_{\max}/\lambda_{\min}$.
10. An orthogonal weight matrix passes every input on with its length
    unchanged.
11. Level sets of least squares with full column rank: ellipsoids with
    semi-axes proportional to $1/\sigma_i$ along $\mathbf v_i$.
12. Muon "steps along the polar factor of its momentum matrix" and "uses a
    tuned quintic in place of the cubic, trading exact convergence for speed"
    (both were in the text at greater length).
13. The alternating form $\mathbf u\leftarrow\mathbf W\mathbf v$,
    $\mathbf v\leftarrow\mathbf W^\top\mathbf u$ is one multiplication by
    $\mathbf W^\top\mathbf W$ and $\|\mathbf W\mathbf v\|$ is the estimate.
14. Rank 18 "is $6\ln20$ rounded up".
15. The Newton-Schulz step estimate is called "rough, a linear phase glued to
    a quadratic one" (a judgment on the derivation just given).

## Slides against the body

No slide statement contradicts the revised body. The slides still carry
three things the body now leaves to their owners: the level-set axis ratio
$\sqrt{\kappa(\mathbf H)}$ and the rate
$(\kappa(\mathbf H)-1)/(\kappa(\mathbf H)+1)$ on the bowl slide, the
motivation "small but consistent directions" on the Muon slide, and Muon's
quintic with its band. All three are true.
