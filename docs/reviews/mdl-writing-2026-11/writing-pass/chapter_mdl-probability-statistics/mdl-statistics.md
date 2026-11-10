# Writing pass log: Statistics

File: `chapter_mdl-probability-statistics/mdl-statistics.md`. Not split (no
markers). No code cell, cell ID, figure include or slide changed.

## Words

Checker count (body, captions and exercises, to the slides marker):
10,068 before, 10,041 after (-27). About 60 of the "after" tokens are the six
new labels and seven new references, so running prose is shorter by roughly
90 words. Four display equations became inline or were replaced by a
reference (62 to 58 delimiters). The section gained the paired/unpaired
comparison (STA-12), the ranking sentence, the interval for a difference and
the names McNemar and Welch; the cuts of the review's list (S8) and the
single ending pay for them.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| STA-14 | "Reporting only the comparisons that cleared $p\le\alpha$ is called *$p$-hacking*" | "Reporting only the winners, or trying analyses until one clears $p\le\alpha$ (*$p$-hacking*), turns noise into apparent findings." | Terminology; agreed with the reviewer. |
| STA-15 | "Pairing saves a factor of about four here" | "cuts the size of each test set by a factor of about four here, and the number of questions that must be labeled by a factor of about nine" | 13,494/3,132 = 4.3; 2 x 13,494/3,132 = 8.6. The Summary no longer quotes the factor. |
| STA-12 / S1 | "$p\approx0.21$, so the gap is still not significant" with no unpaired number | adds: unpaired gap is 0.7 standard errors, $p\approx0.47$; standard error falls from 0.012 to $\sqrt{41}/1000\approx0.006$; names McNemar's test | $z=0.009/\sqrt{0.912\cdot0.088/1000+0.921\cdot0.079/1000}=0.728$, two-sided $p=0.467$; $\sqrt{41}/1000=0.0064$; exact binomial two-sided $p$ for 25 of 41 is 0.211. |
| STA-13 | credible interval and "The Wald interval used here" two subsections before the Wald interval is defined; interval "computed below" | the two paragraphs (merged into one) now follow the Wald cell in "The Gaussian Interval" | Beta(10,5) central 95% interval recomputed: (0.419, 0.872). Wald coverage at $n=13$: 0.908 at $\theta=1/2$, 0.484 at $\theta=0.05$ (exact binomial sums). |
| STA-9 / S2 | one-sample histogram ($n=20$, mean 0.8) read as "the power of the test" 30 lines before the model comparison's 0.72 | the lead-in says it is the one-sample test and "far easier than the model comparison: its statistic has mean $0.8\sqrt{20}\approx3.6$ under the effect, against $2.5$ for the two models" | $0.8\sqrt{20}=3.58$; $0.8\sqrt{10}=2.53$; $P(p<0.1)=\Phi(3.58-1.645)=0.973$ (stored output 0.97). Cell itself unchanged: see deferred. |
| STA-6 | "In each row the sum matches the MSE to round-off. The decomposition holds exactly for the empirical distribution ..." | says outright that the match "is no evidence for the proposition" (algebraic identity) and that the test is agreement with the formulas, which are now stated before the cell (0.533; 0.123+0.225=0.348) | $0.65^2\cdot16/30=0.2253$, $0.35^2=0.1225$. Stored: 0.520, 0.343, 0.792. |
| STA-17 | "moves between about $0.05$ and $0.10$" (stored run prints 0.105) | "between about $0.05$ and $0.11$"; adds "in some runs it only just contains $\log2$" (stored interval (0.665, 1.033)) | 400 simulated data sets, 3,000 resamples each: central 95% of the bootstrap se 0.050 to 0.105, central 98% 0.045 to 0.113, median 0.072; median within 0.15 of $\log 2$ in 96%. Exercise 9(b) changed to 0.11 to agree. |
| STA-18 | "it has no convenient formula", formula given after the cell | formula and prediction $1/\sqrt{200}\approx0.071$ before the cell, with $f(m)=e^{-\log 2}=1/2$ | arithmetic. |
| STA-19 | "The bootstrap also fails when the observations are not independent" | "fails when it resamples units that are not independent" | wording. |
| Exercise 6 citation (review, Exercises) | text cited the exercise for "the percentile interval never contains $\theta$"; the exercise measures the basic interval | the claim stands by its own one-line argument; the exercise is cited for "whether reflecting the interval about the estimate repairs it" | read the exercise. |
| STA-3 | "Earlier sections wrote $p$ for the known bias of a coin; here ..." (contradicts the exponential-family section) | "The accuracy is unknown, so we write $\theta$ for it" | no claim about other sections remains. |
| S3 ($c$) | $c$ = shrinkage factor and half-width | half-width is $h$ in "The Gaussian Interval" and "Tests and Intervals" | local to prose and mathematics. |
| STA-11 | paragraph on the general two-sided $p$-value with generic $T$, $t$ | deleted (Exercise 5(d) states the formula itself) | |
| Duality proof | referred to a set $C$ | after the demotion the coverage sentence writes the interval out | |

No finding disputed. STA-5 (the announced detour) was solved without moving a
cell: the shrunk-mean example, its figure and "Which of the two is better?"
moved from the end of "Bias and Variance" to the head of "The Bias–Variance
Decomposition", where the MSE answers the question at once; "The Sample
Variance" now enters as the first biased estimator. Figure order and cell
order are unchanged.

## Numbers in prose that no cell prints, kept after recomputation

0.7 standard errors and $p\approx0.47$ (unpaired); $\sqrt{41}/1000\approx0.006$;
3.6 against 2.5; coverage 91% and below 50% for the Wald interval at 13 flips;
credible interval (0.42, 0.87); 0.0803/(1000 x 0.03^2) = 0.09 and the
Gaussian 0.001 (3 points = 3.35 se, two-sided 0.0008); powers 0.72 and 0.24;
3,100 and 13,500 questions; $f(m)=1/2$ and 0.071; median MSE "about 0.80" at
$n=30$ (200,000 simulated data sets: 0.803).

## Six before/after pairs

1. Before: "Thus the bias says where the estimates land on average, and the
   standard error says how far a single estimate typically lands from there."
   After: "The bias says where the estimates land on average; the standard
   error says how far a single estimate typically lands from there."
2. Before: "A simulation with $n=3$ and $\sigma^2=4$ shows the bias of the
   plug-in variance and its correction." After: "At $n=3$ and $\sigma^2=4$
   the two denominators should average to $\tfrac23\cdot4\approx2.67$ and to
   $4$." (and after the cell: "They do.")
3. Before: "A simulation shows the uniform distribution, and what replaces it
   when there is an effect. We draw 2,000 data sets ..." After: "What replaces
   the uniform distribution when $H_0$ is false? We apply the one-sample
   $z$-test above to 2,000 data sets ..."
4. Before: "The cell below runs the permutation test on 400 fresh experiments
   for each design." After: "The formula belongs to the $z$-test, which is
   told $\sigma$. The permutation test is told nothing, so we measure its
   power on 400 fresh experiments for each design."
5. Before: "The optimal factor $c^\star$ contains the unknown $\theta$, so it
   is not an estimator we can compute." After: "Unfortunately, the optimal
   factor $c^\star$ contains the unknown $\theta$, so it is not an estimator
   we can compute."
6. Before: "Thus the event "the interval contains $\mu$" is the same event as
   "the estimate lands within $c$ of $\mu$."" After: "This change of viewpoint
   is the whole construction."

Also: "The rule ... $p$-value is uniformly distributed." is now followed by
"Most introductions never say this, and the corrections for running many
tests, below, cannot be understood without it." (a judgment; soften if too
strong).

## Repetition

Removed or reduced:

- Markov's inequality restated in full: now a clause with
  :eqref:`eq_mdl-markov` (owner: Random Variables). The displayed bound on
  $P(|\hat\theta_n-\theta|>\varepsilon)$ stays, since the weak-law proof
  cites it.
- Variance of a sample proportion: the two-term display and its derivation
  are replaced by one sentence citing `eq_prob_var_phat` (main text). The
  simulation stays, as the first sight of a sampling distribution.
- Prediction form of the bias-variance decomposition: display removed; one
  sentence citing `eq_generalization-bias-variance` in
  `sec_generalization_basics`.
- Student-$t$: the two lemmas it rests on are gone; the text cites exercise
  `ex_mdl-distributions-student-t` in the Distributions section (STA-16).
- "We can now ..." recap (STA-20) deleted; the body ends on "Two questions
  remain open ...". Nothing in it was missing from the Summary.
- Summary cut from nine sentences to six; no worked numbers (STA-22).
- "which is why the test above misses about three times in ten" (third
  telling), the two worked sample sizes that the power figure's caption
  repeats, the five-step list (now one sentence), the vector delta method
  (now one sentence), the NumPy/framework-tab sentence.
- "The corrections for running many tests, below, are statements about
  uniform $p$-values" merged into the sentence that introduces the
  proposition.

Kept because this file owns them: the opening paragraph (STA-1; the chapter
introduction should change), the bias-variance decomposition for estimators,
the weak law `eq_mdl-lln`, uniform $p$-values, power and sample size, paired
comparisons, the bootstrap.

## Ranking

- New paragraph at the end of the introduction names the three results to
  remember and the two subsections that can be skipped (consistency, delta
  method); the "chosen before the data" caveat is now its own sentence there
  with a pointer to the end (STA-2).
- Demoted: "Proposition (duality of tests and intervals)" with its two-line
  proof is now a paragraph ending in the italic statement. Four propositions
  remain (unbiased sample variance, bias-variance, weak law, uniform
  $p$-values).
- Terms at their definition are now italic, as in the neighbouring sections
  (21 bold terms changed; STA-21). "Thus" (5) and "In words" (2) are gone;
  two sentence-initial "So" remain.

## Labels added (S6)

`subsec_mdl-bias-variance-decomposition`, `subsec_mdl-permutation-test`
(definition of the $p$-value), `subsec_mdl-power`,
`subsec_mdl-confidence-intervals` (on the `##` heading),
`subsec_mdl-delta-method`, `subsec_mdl-bootstrap` (on the `##` heading).

References added (all resolve at HEAD): `eq_prob_var_phat`,
`eq_generalization-bias-variance`, `eq_mdl-markov`, `subsec_mdl-cramer-rao`,
`subsec_mdl-double-descent-example`, `sec_mdl-concentration-generalization`,
`ex_mdl-distributions-student-t`.

## Exercises

Cell IDs replaced by the name of the experiment in 1(a), 4(a), 6(b), 7(b),
9(b), 10(b). 9(b): "0.05 and 0.10" is now "0.05 and 0.11". Nothing else
changed.

## Cells to cut or redesign on the development machine

- `#mdl-statistics-two-kinds-of-error-and-power-1` (STA-9, S2): run the
  $p$-value histogram on the two-sample design of the running example (20
  seeds, gap 0 and 0.008) so that the pile-up is the 0.72 the formula
  explains. The prose now only warns that the design differs.
- `#statistics-verify-decomposition` (STA-6): the "sum" column is an identity;
  drop it or print the predicted values beside the measured ones.
- `#statistics-unbiased-variance`: confirms the proposition just proved;
  prose is down to one sentence before and "They do." after. Candidate to cut,
  or to redesign around Exercise 7 (denominators $n-1$, $n$, $n+1$ by MSE).
- `#statistics-estimator-metrics` (S7): repeats the kind of simulation in
  `sec_prob`; kept as the first sampling distribution, prose shortened.
- `#statistics-bootstrap` (STA-17): stored draw (se 0.105, median 0.830) is a
  tail event; the prose now covers it, but fixing the 200 data points would
  let the text quote the output.
- Code comment `# the opening's test set` in `#statistics-bootstrap-contrast`
  and `(nominal 950, exact 947)` are fine; comment `B`/`k` names below.

## Other deferred findings

- $B$ (model B, number of shuffles, number of resamples) and $k$ (seeds, and
  the correct-answer count in `#statistics-confidence-interval`): bound to
  the code (`B=2000`, `k = 20`, `acc_b`) and to exercises 3, 6, 10; not
  changed. Prose no longer uses $k$ for the number of correct answers.
  Suggest `R` for resamples in `perm_p` and the bootstrap text.
- $T$ is the $z$-statistic, the generic statistic of the uniformity
  proposition and the studentized statistic; left (the generic paragraph that
  collided most is deleted). $c$ is still the variance denominator in
  Exercise 7 beside the shrinkage factor $c^\star$ in its part (d).
- STA-1: the chapter introduction's opening needs rewriting (other file).
- Exercise 1(a) (review: cut) kept, because cutting it renumbers the parts
  that the text and Exercise 11 cite. Exercises 11 and 12 not re-marked.
- JAX discussion link equals the TensorFlow one (`/t/1103`); correct ID
  unknown.
- S9: a confidence interval for a difference is now one short paragraph in
  "Tests and Intervals"; Welch and McNemar are named in a clause each.
- Main text: the remark on reporting ranges should cite
  `sec_mdl-statistics` (not my file).
- TensorFlow, JAX and MXNet stored outputs not read.

## Claims added or changed

- Unpaired comparison of 912 against 921: 0.7 standard errors, $p\approx0.47$;
  pairing halves the standard error (0.012 to 0.006); the sign-flip test with
  exact counts is McNemar's test.
- Welch's $t$-test is the closed-form counterpart of the permutation test for
  a difference of means.
- The one-sample histogram design has mean statistic 3.6 against 2.5 for the
  two models.
- "The sample median is unbiased here by symmetry."
- The $t$ distribution of the studentized mean is the Gamma scale mixture of
  the Distributions exercise (precision $\chi^2_\nu/\nu\sim\mathrm{Gamma}(\nu/2,\nu/2)$).
- An interval $\bar x_B-\bar x_A\pm1.96\,\sigma\sqrt{2/k}$ excludes 0 exactly
  when the two-sample $z$-test rejects.
- Label factor "about nine"; bootstrap se range "0.05 to 0.11"; "in some runs
  [the percentile interval] only just contains $\log 2$".
- Past interpolation "the test error can fall again", with the reference to
  the double-descent sweep (STA-7), in place of the unreferenced claim.
- The Chebyshev bound "needs nothing beyond a finite variance;
  [Concentration] derives the sharper bounds that more assumptions buy."

## Slides that now disagree with the body

- "The median's se ... varies between about 0.05 and 0.10" (body: 0.11).
- Half-width written $c$ and the interval $C$ on the Gaussian-interval and
  duality slides (body: $h$, no $C$).
- "The opening's test set" on the bootstrap slide (body: "our test set").
- The slides have no paired-against-unpaired numbers ($p\approx0.47$, 0.006).
