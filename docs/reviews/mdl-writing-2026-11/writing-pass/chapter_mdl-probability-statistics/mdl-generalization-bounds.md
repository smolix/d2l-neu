# Writing pass log: Uniform Convergence and Generalization

Source: `chapter_mdl-probability-statistics/mdl-generalization-bounds.md`.
Not split; no markers. No code cell, label, figure include or slide changed
(`check_mdl_prose.py`: "OK: only prose changed"; `lint_source.py`: clean).
Numeric checks were done with `python3` and NumPy in scratch files (no SciPy,
no PyTorch); binomial tails exactly with `math.comb`.

## Words

Checker count (body and exercises): 7,962 before, 7,931 after.

| Part | Before | After |
|---|---|---|
| Introduction | 387 | 301 |
| Uniform Convergence | 1,177 | 1,067 |
| Rademacher Complexity | 2,614 | 2,779 |
| Interpolation and Double Descent | 1,716 | 1,762 |
| Summary | 217 | 170 |
| Exercises | 1,851 | 1,852 |

The section is not materially shorter, and the review's target (about 4,800
words of body) is not met. The cuts (about 550 words: repeated coin result,
the proposition and proof of the finite-class bound, the Chebyshev display,
the nesting proposition, the inverse-Wishart derivation, the recap, the
summary) were spent on the corrections the review asked for (about 500
words: GEN-7, GEN-6, GEN-14, GEN-16, GEN-17, GEN-19, GEN-3, S4, S5). Part
three cannot reach 1,000 words without removing cells and their readings;
see "Deferred".

## Correctness changes

- **GEN-7** (expected against empirical complexity). The text stated the
  theorem with $\mathfrak R_n$ and then computed $\widehat{\mathfrak R}_S$
  without comment. Now one paragraph after the proof: changing one point
  moves $\widehat{\mathfrak R}_S(\mathcal G)$ by at most $1/n$, so a second
  use of McDiarmid gives the bound with $\widehat{\mathfrak R}_S$ and the
  confidence term $3\sqrt{\log(2/\delta)/(2n)}$. Verified by hand: two
  one-sided McDiarmid events at $\delta/2$ each, deviations
  $\sqrt{\log(2/\delta)/(2n)}$ for $\Phi$ and twice that for
  $2\widehat{\mathfrak R}_S$; this is the standard statement (Mohri,
  Rostamizadeh and Talwalkar, Theorem 3.3).
- **GEN-16** (threshold 1 applied to a regression). The text said the
  capacity product "stays above 1" so "the resulting bound on the test error
  is empty", and repeated it in the closing and the Summary. Now the product
  is introduced as a diagnostic borrowed from classification ("no theorem of
  this section covers" squared loss on unbounded targets), and the reasons it
  is no guarantee are: no applicable theorem, it changes when the targets
  are rescaled, and $B$ was read off the fit. "Stays above 1" is gone from
  the body and the Summary. Agreed with the reviewer by reading the theorem's
  hypotheses.
- **GEN-17** (the comparison with the classical regime). Was: "roughly ten
  times below the best underparameterized model" and, in the Summary, "ten
  times better than the best classical fit". Now the text says that a test
  error of 1 is that of predicting zero, that the best fit below the
  threshold removes a quarter to a third of it, that least squares on the 15
  raw inputs has expected error $0.01\cdot15/24\approx0.006$, and that the
  sweep shows the widest model is best within this nested family and "does
  not show that interpolation beats a well-specified small model". Verified:
  4,000 simulated OLS fits give mean excess error 0.00625; the sweep rerun
  (40 trials) gives 0.056 at $p=400$ and a minimum of 0.66 to 0.74 below the
  threshold.
- **GEN-19** (benign overfitting). Was: ":eqref:`eq_mdl-svd-interpolant`
  explains why the noise costs so little", with small singular values as the
  cure. Now the noise part is rewritten as the noise variance times the
  average over directions of (mean squared test feature)/(mean squared
  training feature) along $\mathbf v_i$, which says when a small $\sigma_i$
  hurts; and "benign means bounded here, not vanishing". Verified: the
  identity by hand ($\boldsymbol\Phi^+=\sum_i\mathbf v_i\mathbf u_i^\top/\sigma_i$,
  orthonormal $\mathbf u_i$, $\sigma_i=\|\boldsymbol\Phi\mathbf v_i\|$); the
  ratios numerically (200 trials): at $p=400$ smallest about 0.3 to 0.45,
  largest about 1.6 to 3.5, mean about 0.93; at $p=40$ the ratio along
  $\mathbf v_q$ has median about 7,000 (10th to 90th percentile 1,000 to
  125,000).
- **GEN-6**. The blanket convention "predictors take values in $[-1,1]$" is
  replaced by a sentence that an unbounded class has no ceiling and that
  fitting every sign pattern means complexity at least 1. "A predictor class
  that fits every sign pattern ... has $\widehat{\mathfrak R}_S=1$" is now
  restricted to the $\pm1$-valued class of that paragraph.
- **GEN-14**. The inference from one random labeling to "fits every sign
  pattern" is replaced by the reviewer's argument (class of all functions
  the architecture represents; supremum close to 1 for each labeling that
  training fits), with the caveat that the experiments are evidence, not a
  proof.
- **GEN-18**. "for a square Gaussian matrix" is now "as it is for a square
  Gaussian matrix, where ..."; "invertible with overwhelming probability" is
  "almost always invertible".
- **GEN-3**. Added where the factor of 30 is lost: single coin outside 27 to
  73 has probability $1.67\times10^{-6}$ against the allowed $5\times10^{-5}$
  (exact binomial); union over 1,000 coins 0.167%.
- **GEN-12, GEN-13**. The paragraph before the sign-fitting cell now gives
  the logic first ($\widehat{\mathfrak R}_S\ge1$ and $\le Br/\sqrt n$, hence
  $Br/\sqrt n\ge1$) and the cell is read for how close fits come: about 1.4
  for a typical fit, 2 to 2.8 for the ball (three reruns: 1.38 to 1.43 and
  2.1 to 2.4; stored 2.5).
- Stored output: "about 96%" is now "96 to 97%" (stored 97%, exact 96.4%).

No finding disagreed with.

## Numbers kept that no cell prints (all recomputed)

99.3% (0.99336); 0.0033 and 3.3 coins; 0.17%; $1.7\times10^{-6}$; 0.0136 and
0.024; 3,500 (3,466); Chebyshev $t=1$; Massart 0.078; margin terms 0.089,
0.005, thresholds 10.6 and 110; OLS 0.006; the ratios above; the ranges of
the sweep (taken over from the original, consistent with my 40-trial rerun
and the stored table). Not recomputed, kept from the original: the 11% and
1% slack (consistent with the stored 0.126 and 0.140), "a peak of the same
size with `noise = 0`" and "no peak" for a ten-feature target (the reviewer
reran both).

## Six before/after pairs

1. Before: "It shows about 66 heads, although every coin is fair and any one
   coin, named in advance, lands between 37 and 63 heads in at least 95% of
   experiments." After: "flip each of 1,000 fair coins 100 times and keep the
   coin with the most heads. How many does it show?"
2. Before: "The cell below repeats the experiment 2,000 times and asks how
   far the best of the 1,000 coins strays, and how often it leaves the
   single-coin band." After: "The best of the 1,000 coins has no such
   promise. We repeat the experiment 2,000 times and record how far the best
   coin strays and how often it leaves the single-coin band."
3. Before: "The cell below does this for 50 unit-norm points in $d=2$, $20$
   and $1{,}000$ dimensions, and compares with the bound." After: "The slack
   analysis above predicts an estimate about 11% below the bound
   $Br/\sqrt n=1/\sqrt{50}$ in the plane and about 1% below it in twenty
   dimensions."
4. Before: "A complexity bound of $1$ or more allows the class to fit every
   sign pattern, so the resulting bound on the test error is empty." After:
   "It follows the shape of the test curve, but it is no guarantee: no
   theorem here covers this loss, the product changes when the targets are
   rescaled, and its $B$ was read off the fit rather than fixed in advance."
5. Before: "The best model in the experiment is the most overparameterized
   one, fitting noisy data exactly, with ten times more parameters than data
   points." After: "The sweep thus shows that within one nested family of
   random features the widest model is best. It does not show that
   interpolation beats a well-specified small model."
6. Before: "Overfitting is *benign* when the spectrum has this shape: a few
   strong directions to carry the signal, and many weak ones to absorb the
   noise." After: "Benign means bounded here, not vanishing: at $p=400$ the
   predictions still carry about as much noise as a single training label."

## Repetition

Removed or reduced to a reference:

- The 66 heads: now once, at the cell (GEN-1). It remains in the figure
  caption, which I did not touch.
- Finite-class bound: recalled with
  :eqref:`eq_generalization-classification-finite` and a two-sentence proof;
  the radii for 2,000 and a million candidates cite the main text; the
  leaderboard attack is one sentence with the two rates (GEN-2, GEN-4).
- Union bound: no longer defined; cited as
  :numref:`sec_mdl-concentration-generalization` (the Concentration file has
  no subsection label for it; if that writer adds one, point there).
- Chebyshev contrast: three sentences, display removed.
- Nesting: a sentence citing :numref:`subsec_double_descent` (GEN-15).
- Inverse-Wishart derivation: left to the exercise (GEN-11).
- The three cases of `pinv` and the recap of the U-curve at the head of part
  three (main text).
- "We can now" recap deleted; its last two sentences end the last subsection
  (GEN-20). Summary cut from ten sentences to seven (GEN-22).
- "says nothing / certifies nothing" refrain: all nine replaced (GEN-21).
- Boilerplate sentence about plain NumPy removed.

Kept because this file owns them: the coin experiment and its figure, the
sample-size form, the measured conservatism, the SVD formula and the clean
and noise split. The Rademacher part is unchanged except for the additions
listed under correctness, a ranking paragraph at its head, and S4.

## Ranking, demotions, labels

- Demoted: "Proposition (finite-class uniform convergence)" with its Proof
  (the equation and `eq_mdl-finite-class` stay); "Proposition (nesting)" with
  its Proof.
- Ranking added: the head of "Rademacher Complexity" names the two results
  to remember and the one hard step; part three is marked as skippable on a
  first reading.
- Labels added: none. Labels removed: none.
- Exercise 8(a): "why the proof does not apply" is now "why the padding
  argument does not apply" (the Proof block is gone). No other exercise
  change.

## Cells to cut or redesign on the development machine

- `mdl-generalization-bounds-rademacher-complexity-in-numbers-1` (GEN-13):
  the printed product confirms an inequality that holds by logic. The prose
  now reads it for tightness. Better: print the typical value (mean norm
  times mean row norm over $\sqrt n$) beside the ball's.
- `mdl-concentration-generalization-coin-flips-in-code`: confirms the slack
  analysis; prose is now one prediction and two sentences. Acceptable once.
- `mdl-concentration-generalization-double-descent-in-twenty-five-lines`
  (GEN-19 ii, GEN-17): print five singular values at $p=400$ or the
  test/train ratios the new paragraph quotes, and add least squares on the
  raw inputs as a reference row.
- Imports cell: unused framework imports (S6).

## Other deferred findings

- S2, S3 length targets (see Words). Further cuts need cells removed.
- GEN-9: measuring $Br/\gamma$ for a trained classifier is a new experiment.
  The text now says the ratio is chosen, not measured. I did not add the
  forward tie to the sign fits (it cost 45 words).
- Exercise notes: rename $\rho_k$, $\varrho_k$; start "Leaderboard attack"
  from the main-text cell; a problem on least squares on the raw inputs.
- Figure caption still says "near 0.66".
- Main text (`generalization-classification.md`) still says this section
  "proves the finite-class bound"; it now recalls it. Not my file.
- GEN-5: I cite Bartlett and Mendelson only and dropped the names without a
  citation; add Koltchinskii and Panchenko if a bib entry exists.

## Claims added or changed

1. Empirical form of the theorem with $3\sqrt{\log(2/\delta)/(2n)}$.
2. An unbounded class that fits every sign pattern has complexity at least 1.
3. Single-coin probability $1.7\times10^{-6}$ against $5\times10^{-5}$; the
   union bound is nearly exact for rare independent events.
4. Rates $\sqrt{\log k/n}$ against $\sqrt{k/n}$ (from the section's exercise
   and Blum and Hardt).
5. Two-layer ReLU bound $2B_1B_2r/\sqrt n$ stated in the text (it is the last
   exercise).
6. Rescaled hinge loss gives a bound on the rescaled loss.
7. Typical sign fit at $\|\mathbf w\|r/\sqrt n\approx1.4$; "one and a half to
   three times the minimum".
8. Targets have variance 1; OLS on raw inputs 0.006; "several times above".
9. Noise-part identity and the ratios (0.3 to 3, mean just below 1;
   thousands at $p=n$).
10. Notation mapping to the main text ($\epsilon_\mathcal{S}$, $\epsilon$,
    "Hoeffding radius", VC dimension).

## Slides now disagreeing with the body

- "$p=400$: about $0.05$, roughly ten times below the classical minimum":
  the body now qualifies the comparison.
- "The Capacity Product Tracks the Curve but Never Certifies" / "It stays
  above 1, and its $B$ was read off the fit": the body no longer uses the
  threshold 1.
- The slides tell the 66 heads three times and state the finite-class bound
  as a result of this section; consistent in fact, but no longer in emphasis.
