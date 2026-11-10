# Writing pass log: Numerical Stability and Conditioning

File: `chapter_mdl-optimization/mdl-numerical-stability-conditioning.md`.
Reviews followed: `40-optimization/mdl-numerical-stability-conditioning.md`
(NSC-1 to NSC-24, S1 to S6) and `40-optimization.md`.

## Words

Checker count (prose with exercises): 9,456 before, 9,368 after (-88). Body
without exercises: about 7,180 before, 7,075 after. By half: the Conditioning
half went from about 2,650 to 2,350 words (-300; the ridge subsection alone
-240, Backward and Forward Error -125); the first half grew by about 210
words. The growth is the review's own requests (NSC-1, 2, 4, 10, 12, 14, the
Sterbenz reason, the parallel-sum sentence) plus a five-line ranking
paragraph in the opening; the tab notes lost 85 words (NSC-5). The cut in the
second half is smaller than the review's estimate of 1,000 words because the
three labelled displays that other text cites (`eq_mdl-opt-ridge-kappa`,
`eq_mdl-opt-filter-factors`, `eq_mdl-opt-gd-iteration-bound`, all used by
exercises 5 and 6) had to stay with enough words to carry their hypotheses.
Display equations: 62 delimiters before, 56 after (three displays folded into
sentences). Checker: "OK: only prose changed", no em dash warning (the table
separator `|---|` that triggered a false warning is now `|----|`).
`lint_source.py`: clean. No split markers (section is not to be split).

## Correctness changes

- **NSC-17 (Hilbert).** Was: "Rounding the entries $1/3$, $1/7$, … to
  float64 is already a backward error of size $u$: the exact solution of the
  *stored* system differs from $\mathbf 1$ by $7\times10^{-8}$,
  $4.5\times10^{-5}$, and $0.14$". Now: computing $\mathbf b=\mathbf H\mathbf
  1$ rounds every entry of $\mathbf b$; rounding the matrix entries "does no
  harm here, because $\mathbf b$ is built from the matrix as stored"; the
  stored system's exact solution differs from $\mathbf 1$ "by roughly
  $10^{-7}$, $10^{-4}$, and $10^{-1}$". Verified with `fractions.Fraction`
  (stored $\mathbf H$, Gauss-Jordan in exact arithmetic): with $\mathbf b$
  summed exactly the solution is exactly $\mathbf 1$; with $\mathbf b$ in
  fp64 the deviation for $n=8,10,12$ is $1.3\times10^{-7}$,
  $2.4\times10^{-4}$, $0.21$ (BLAS `@`), and between $5\times10^{-8}$ and
  $4\times10^{-7}$, $1.3$ and $2.8\times10^{-4}$, $0.05$ and $0.21$ over
  four summation orders. The old three numbers did not reproduce; only
  orders of magnitude are quoted. The opening's matching clause ("rounding
  its entries in the sixteenth digit") now reads "a rounding error in the
  sixteenth digit of its data".
- **NSC-1.** Was: "so the bound $|\delta|\le u$ is attained". Now: "Half a
  gap, the worst case, does occur; as a relative error it is $u/(1+u)$, just
  inside the bound". (For $x=1+u$, $\mathrm{fl}(x)=1$, $\delta=-u/(1+u)$.)
- **NSC-11.** Was: "The measured error is $3\times10^{-9}$ at $h=10^{-8}$".
  Now: "We measured about $10^{-8}$". Recomputed: $1.4\times10^{-8}$ at
  $h=10^{-8}$ (between $2\times10^{-9}$ and $1.4\times10^{-8}$ for $h$ within
  a factor 2), $4.3\times10^{-5}$ at $10^{-12}$, $4.2\times10^{-3}$ at
  $10^{-2}$.
- **NSC-8.** "`log1p` evaluates a series in $x$ and never forms $1+x$" is
  now "is written so that the rounding error of $1+x$ never reaches the
  result; for tiny $x$ it returns $x$ itself".
- **NSC-12 (Welford stability).** Now labelled informal, with the pointer to
  Higham; the mechanism is stated as the review asks (everything entering
  $M_k$ is a deviation of size $s$; only the error of the center comes from
  the scale $m$); "a few dozen gaps" is "some tens of gaps" (measured: 11 to
  92 gaps over 12 seeds, Welford relative error $4\times10^{-10}$ to
  $5\times10^{-8}$). The second-order statement is now restricted to a
  *constant* error of the center. No new citation key (Chan, Golub and
  LeVeque is not in the bibliography).
- **NSC-13.** Proposition title "(Welford's recursion is exact)" is now
  "(Welford's recursion)"; "makes the recursion exact" is "correct".
- **NSC-14.** "compute ... this way" is "with Welford's recursion"; the
  summation paragraph now covers the range above $2^{20}$ (gap $1/8$, one
  gap, again $0.125$; checked on the fp32 running sum).
- **NSC-18.** "of order $10^{-16}$ in every row" is "at or below about
  $10^{-16}$" (the stored row $n=4$ prints `0.0e+00`).
- **NSC-21 remnant.** "falls below tol" is "is met within" ($\rho^k\le$ tol).
- **NSC-3.** The three `NaN` of the opening are now tied to the log-sum-exp
  cell in one clause ("dividing `inf` by `inf` is what gave the three
  `NaN`"). Recomputed: naive softmax of $(1000,1001,1002)$ in fp64 is
  `[nan nan nan]`.
- **NSC-4.** $3.8\times10^{-44}$ is now explained as $3.7\times10^{-44}$
  "rounded to the subnormal grid with five significant bits left"
  (it is $27\cdot2^{-149}$).
- **Exercise 2 part 3.** Test points $\{-50,0,50,800\}$ are now
  $\{-800,-50,0,50\}$, and the part asks where the written formula overflows
  and where it returns $0$. Checked in fp32: naive gives `inf` at $-800$
  (true 800), $0$ at $50$ (true $1.9\times10^{-22}$); the rewritten form
  gives 800, 50, 0.6931472, $1.93\times10^{-22}$.

No finding disagreed with.

## Numbers no cell prints that were kept (all recomputed)

Finite differences: $h^\star\approx2\times10^{-8}$, predicted
$1.8\times10^{-8}$, measured about $10^{-8}$, $4\times10^{-5}$,
$4\times10^{-3}$; new: fp32 best step near $5\times10^{-4}$ with about three
digits ($2\sqrt{u_{32}}=4.9\times10^{-4}$; measured error $3$ to
$4\times10^{-4}$). Hilbert stored-system deviations (orders of magnitude).
$1.6\,|m|/s$ ($2\,\mathbb E|z|=1.596$). Two-line system: $3.5\times10^{-5}$,
$\kappa\approx4\times10^4$ (40,002), $\omega=2.5\times10^{-5}$, angle
$0.0029^\circ$. Running-mean error "some tens of gaps".

## Six before/after pairs

1. Before: "The cell compares the two softmax implementations at $t=100$,
   and at $t=10$, where both are finite:" After: "At $t=100$ the direct
   formula should fail and the shifted one should not. At $t=10$ both are
   finite; do they return the same bits?"
2. Before: "The cell uses the opening's logits $(1000,1001,1002)$ in float32
   and computes both forms:" After: "How much worse? The cell computes both
   forms in fp32 for the logits $(1000,1001,1002)$ of the introduction:"
3. Before: "The cell computes the loss both ways with the library's
   functions for $t\in\{20,60,103,104\}$; the tab note says what the
   probability route returns in this library:" After: "The cell computes the
   loss both ways with the library's functions; its last two values of $t$
   (the column `gap`), $103$ and $104$, straddle that threshold:"
4. Before: "The cell builds designs with $\kappa(\mathbf A)=10^3$ and $10^6$
   from random orthonormal factors and a right-hand side in the column space
   (zero residual), and solves each both ways:" After: "The cell therefore
   builds harder designs, with $\kappa(\mathbf A)=10^3$ and $10^6$, from
   random orthonormal factors and a right-hand side in the column space
   (zero residual). We expect gaps of three and of six digits:"
5. Before: "The amplification from backward to forward error is a property
   of the problem. The condition number of
   :numref:`subsec_mdl-condition-number` measures it" After: "How far is a
   property of the problem, not of the solver. It is the condition number of
   :numref:`subsec_mdl-relative-condition-number` once more, now for the
   function $\mathbf b\mapsto\mathbf A^{-1}\mathbf b$"
6. Before: "You can now predict from $u$ and $\kappa$ how many digits a
   computation keeps, and rewrite a formula that loses more than that. These
   tools do not cover error that grows over many steps" After: "None of
   these tools covers error that grows over many steps of an iteration, such
   as a gradient through a hundred layers;
   :numref:`sec_numerical_stability` takes that up."

## Repetition

Removed or reduced to a reference:

- Two-line system (owner: chapter 38, `subsec_mdl-condition-number`). The
  re-derivation of $(1,1)$ and $(0,2)$, the singular values and the "$100\%$,
  amplification $2.8\times10^4$" paragraph are gone; the system is cited and
  only the line picture and the backward-error reading remain. The
  right-hand-side bound, formerly a display, is one clause citing
  `eq_mdl-condition-bound`.
- $\kappa(\mathbf A^\top\mathbf A)=\kappa(\mathbf A)^2$: derivation replaced
  by a reference; the labelled display stays (`gptq.md` cites it).
- Filter factors (owner: `eq_mdl-ridge-filter`, NSC-20): proposition and
  proof removed; the labelled display stays, introduced as that equation in
  this section's notation, with the remark that $\lambda$ is $\gamma$ there.
- Iteration bound (owner: `sec_mdl-gradient-based-optimization`, NSC-21):
  proposition and proof removed; the contraction factor is cited by
  `eq_mdl-opt-optimal-step` and the labelled display follows in two
  sentences. "6,908 (the 'about 6,900 steps ...' of ...)" is now just the
  number.
- Tab notes: the two sentences common to all four tabs are stated once
  (NSC-5); "gap" for $t$ is gone from the prose.
- Closing paragraph "You can now ..." removed (NSC-24); its second sentence
  is the ending. The paragraph on elongated level sets after the Hilbert
  cell (a repeat of chapter 38's bowl paragraph) is removed.
- Overflow thresholds $88.72$ and $11.09$ were stated after the cell and
  again in the next paragraph; once now.

Kept because this file owns them: the rounding model, the small-update
proposition, the max shift with the sandwich bound (NSC S5), the condition
number of a function, Welford, the summation bounds, forward and backward
error with both propositions, the Hilbert experiment, the normal-equations
experiment.

## Ranking and demotions

- Opening: one paragraph names the three results that matter and the two
  subsections that can be skipped on a first reading.
- "Proposition (ridge improves conditioning monotonically)" demoted to two
  sentences around the labelled display `eq_mdl-opt-ridge-kappa`.
- "Proposition (filter factors)" and "Proposition (iteration bound)" demoted
  as above; all three labels kept.
- The forward-error bound is marked "the result to remember"; the Welford
  stability argument is marked as the hard, informal step.
- Labels added: none. Labels removed: none.

## Other changes

- **Format names (NSC-7; assignment).** Prose and exercises now use fp16,
  bf16, fp32, fp64 throughout, matching `sec_numerics`; `bfloat16` appears
  once, as the library name, where bf16 is introduced. Code, cell output,
  figure captions and slides are unchanged.
- **NSC-2.** One sentence says that $w=1$ is the exceptional case of the
  proposition and that `sec_numerics` tries an increase (threshold $u$).
- **NSC-6.** "this part", "the next part", "the last part", "the third
  subsection" replaced by labels or reworded.
- **NSC-9.** Table introduced as a reference table.
- **NSC-10.** Finite differences are a paragraph of their own, opened by the
  gradient check, with the fp32 fact.
- **NSC-15 / S3.** The sentence linking $\kappa_f$ and $\kappa(\mathbf A)$.
- **NSC-19, NSC-22, NSC-23.** As the review proposes (NSC-23 in the text,
  not the caption: "each panel is centered on its own minimizer").
- **S6.** One sentence: damping in second-order optimizers and the jitter of
  GP regression (`chap_gp`) are the same $\lambda\mathbf I$.
- Sterbenz: one clause of reason at first use; the second use cites it.
- One sentence on why parallel sums are not reproducible.
- Reflowed paragraphs I touched to 78 columns.

## Cells to cut or redesign on the development machine

- `numerical-stability-conditioning-ridge` and
  `mdl-numerical-stability-conditioning-ridge-regularization-and-conditioning-1`
  (NSC-21, S2): the step counts belong to the gradient-descent section; here
  only the last column (the price of ridge) is new. Candidates to move or to
  reduce to the minimizer column.
- `numerical-stability-conditioning-stable-softmax` (NSC-3): add one line
  printing the naive softmax of $(1000,1001,1002)$ in fp64, so that the
  opening's three `NaN` are printed.
- `numerical-stability-conditioning-cross-entropy`: column header `gap`
  should be `t` (NSC-5).
- New three-line cell for the finite-difference errors (NSC-11) and one for
  the stored Hilbert system (NSC-17), or move the latter into exercise 11
  part 3.

## Deferred

- NSC-16: figure `fig_mdl-opt-two-lines` draws the $1.1$ system; caption and
  figure untouched (and the label $x_1$ sits on the lines in panel (b)).
- NSC-23: the figure itself (same star position in both panels).
- NSC-21: moving the iteration bound into the gradient-descent section as a
  proposition (other file); here it is now a cited two-line derivation.
- Exercise 11 trim (merge parts 2 and 3): left, not an error.
- $\lambda$ here against $\gamma$ in chapter 38: noted in the text, symbols
  unchanged. $\mathbf w$ here against $\mathbf x$ there; $\sigma_d$ here
  against $\sigma_r$, $\sigma_n$ there.
- `sec_numerics` writes `bf16`; its code prints `bfloat16`. Now consistent
  with this section's prose.

## Claims added or changed

1. Worst-case relative rounding error at the midpoint is $u/(1+u)$.
2. A decrease from a power of two is lost below $u/2$, an increase below $u$
   (restates the proposition).
3. Sterbenz reason: both numbers are multiples of the smaller one's gap and
   the difference needs no more bits.
4. `log1p` returns $x$ for tiny $x$.
5. fp32 forward difference: best step near $5\times10^{-4}$, about three
   digits; "which is why gradient checks are run in fp64" (a statement of
   common practice).
6. Welford: informal argument as above; reference to Higham (2002) for the
   analysis.
7. "Batch-normalization kernels on GPUs commonly compute ... with Welford's
   recursion": the review's reading of "this way". **Not verified against
   library source**; from memory this is what PyTorch's native CUDA kernel
   does. Please confirm or soften.
8. A sum reduced in parallel need not be the same on another device or run.
9. $\kappa(\mathbf A)$ is the worst-case relative condition number of
   $\mathbf b\mapsto\mathbf A^{-1}\mathbf b$ (this is
   `eq_mdl-condition-bound` with attainment).
10. Hilbert: the deviation of the stored system comes from rounding
    $\mathbf b$; orders $10^{-7}$, $10^{-4}$, $10^{-1}$.
11. Damping and GP jitter are $\lambda\mathbf I$ under other names.
12. Running example: $\kappa(\mathbf A)\approx32$ is "too well conditioned
    to show the difference" between the two least-squares routes.

## Slides against the body

- Slides write `float64` and `bfloat16`; the body now writes fp64 and bf16.
- The opening slide note says "rounding the data moves the answer", which
  agrees with the corrected body. No slide repeats the NSC-17 numbers, the
  $3\times10^{-9}$, or "is attained".
- The ridge slide states the iteration bound and filter factors without
  attribution; still true, though the body now cites their owners.
