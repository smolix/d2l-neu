# Statistics

Source: `chapter_mdl-probability-statistics/mdl-statistics.md` (to the slides marker at line 1564). About 7,750 words of body prose outside code and display math, plus about 2,300 words of exercises (12 problems).

## Verdict

This is the most useful statistics section for a deep learning reader that I know of in a textbook of this kind, and it is close to the bar. It is organized around one object, the sampling distribution, and around the question its reader actually has: is model B better than model A, or did the test set (or the seeds) happen to suit it? Nearly every tool is introduced because that question needs it, every formula is checked by a simulation that the prose reads correctly, and the material chosen (power and seeds, paired comparison per question, multiple testing in a sweep, clustered test sets, the failure of the bootstrap) is what practitioners get wrong. I recomputed every number in the prose and found no mathematical error. What keeps it from a 5 is management of its own examples and symbols. The section runs six numerical scenarios, two of which use the same numbers (0.8 and 20) for different designs with different answers; the letters $B$, $k$ and $T$ each mean two or three things; the opening question is answered 800 lines later without the comparison the text promised; and one subsection uses "the Wald interval" two subsections before defining it. These are an editor's afternoon, not a rewrite. The single most important fix is to make the opening pair of classifiers (912 and 921 of 1,000) pay off: state the unpaired and the paired verdicts side by side where the paired test is run.

## Scores

| Dimension | Score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **One organizing idea, stated early and used throughout.** "Statistics answers such questions by asking how an estimate would vary over the samples we did not draw." Bias, standard error, the null distribution, coverage and the bootstrap are each presented as a statement about this distribution. This is the right spine for the topic and should not be diluted.
- **Judgment in the prose.** "A decision based on a gap smaller than its own noise is a decision made by chance." "When a test with power $0.2$ fails to reject, we have learned almost nothing." "A biased estimator wins, and an unbiased estimator can still be a poor one." "The bound is crude." These are the sentences with a voice; protect them.
- **The $n=2$ case before the $n-1$ proposition.** The two deviations $\pm(x_1-x_2)/2$ show the factor of one half before the general proof, and the proof then has a sentence that says what happened ("measuring from $\bar x$ removes $n(\bar x-\mu)^2$ from the sum"). This is the house style at its best.
- **The shrunk mean.** One estimator carries bias, variance, the decomposition, the exact U curve, and the warning that the optimal factor "contains the unknown $\theta$", with the counterexample at $\theta=10$ (MSE 12.5 against 0.53). The Pythagoras reading with `mdl-prob-stat-mse-triangle.svg` is a real explanation, and the figure (rendered and checked: legs 0.35 and 0.47, hypotenuse 0.59, the median outside the circle) carries it.
- **$p$-values are uniform under the null**, with a short correct proof, the counting version for the permutation test that explains the $+1$, and a ten-bin simulation. Most introductions never say this, and it is what makes Bonferroni and Benjamini–Hochberg intelligible.
- **Power is treated as a property of the experiment.** The first permutation test is allowed to fail on a real effect, the text says so ("in a sizable minority it fails to reject, as it may have done above"), and the next subsections compute why: power 0.72 with 20 seeds, 0.24 with 5, and 25 seeds needed for 0.8. The simulation agrees (0.68 and 0.21 in the stored run) and the prose explains the small shortfall.
- **The confidence-interval simulation reads its own noise.** "The count lands near 947, typically within $\pm14$, rather than near the nominal 950. The shortfall is systematic." The text then derives both the 947 and the 14. This is how to write about an unseeded cell.
- **Failures are shown with numbers:** the bootstrap of the maximum (63% of mass on one value), clustered questions (a factor of $\sqrt{2.9}\approx1.7$ at $m=20$, $\rho=0.1$), the Wald interval at 13 flips (coverage about 91% at $\theta=\frac12$, below 50% at $0.05$; I confirmed 0.908 and 0.484).
- **Stored outputs match the prose** in every cell (one marginal case, STA-17).

## Structural findings

1. **The opening example does not pay off as promised.** The introduction asks whether the model with 921 correct answers is better than the one with 912. The first subsection computes that for independent test sets the gap has standard deviation 0.012 and promises that the paired analysis "shows how much sharper the comparison becomes". The hypothesis-testing part then switches to a different scenario (seeds, accuracies 0.850 and 0.858) for 280 lines. When the text returns ("Return to the opening"), it reports $p\approx0.21$ and "the gap is still not significant", with no unpaired number beside it. The reader cannot see how much pairing helped. The unpaired two-sided $p$-value is 0.47, and the standard error of the gap falls from 0.012 to $\sqrt{41}/1000\approx0.0064$. Put both pairs of numbers in the text. See STA-12.

2. **Six scenarios, and two of them collide.** The section uses (a) the classifier with 912 of 1,000; (b) Gaussian data with $\theta=1$, $\sigma=4$, $n=30$; (c) models A and B over seeds, 0.850 against 0.858 with seed noise 0.010; (d) a one-sample $z$-test with $\sigma=1$, $n=20$ and true mean 0.8; (e) the coin, 9 of 13; (f) an exponential sample of 200. Scenario (d) is the problem: it shares the numbers 0.8 and 20 with (c), where $\delta=0.8$ and $k=20$, but it is a one-sample design with power 0.97 where (c) has 0.72. The two appear 30 lines apart. See STA-9. Scenario (d) can be replaced by (c) itself.

3. **Symbols are reused.** $B$ is a model and the number of shuffles and of bootstrap resamples ("repeat $B$ times" eight lines after "model A and model B", and "$\bar x_B-\bar x_A$" beside "$B+1$ gaps"). $k$ is the number of correct answers, the number of seeds per model, and the Benjamini–Hochberg index. $T$ is the $z$-statistic here, the sufficient statistic and the temperature in the previous section. $c$ is the shrinkage factor, the half-width of the Gaussian interval, and the variance denominator in Exercise 7. Fix at least $B$ (use $R$ for the number of resamples, or call the models "old" and "new") and $c$.

4. **Order inside "Confidence Intervals".** "Definition and Interpretation" compares the credible interval $(0.42,0.87)$ with a confidence interval "computed below" and criticizes "The Wald interval used here" before any interval has been constructed or the name defined. The comparison is valuable but belongs after the Wald interval is computed for the coin, about 90 lines later. See STA-13.

5. **Is it two sections?** At 7,750 words with four parts (estimators 2,500; tests 2,400; intervals 1,400; bootstrap 900) it could split at "Hypothesis Testing" into "Estimators" and "Tests and Intervals". I do not recommend it: the parts share the opening example and the sampling distribution, and the second half is the reason a reader comes. What would shorten it with no loss is listed under "Cuts" below.

6. **Lookup.** Only one subsection has a label (`subsec_mdl-paired-comparisons`). A main-text passage that wants to point at the bias–variance decomposition, the $p$-value, power and sample size, the confidence interval, the delta method or the bootstrap can only cite the whole 7,750-word section. Add labels to those six subsections. The section otherwise stands on its own well; it restates Markov and Jensen where it uses them.

7. **Relation to the main text.** `chapter_preliminaries/probability.md` teaches the standard error of a frequency, a simulated $p$-value for 56 heads in 100, and the two-standard-error rule, at an introductory level. This section is clearly deeper and does not re-teach, with one exception: "Estimators and the Sample Proportion" re-derives $\textrm{Var}(\hat\theta)=\theta(1-\theta)/n$ and checks it by the same kind of simulation as the Preliminaries (`eq_prob_var_phat`), although the chapter introduction says "We assume the probability of :numref:`sec_prob`". Cite the formula and keep the simulation, whose purpose here is different (to show the sampling distribution). The prediction form of the bias–variance decomposition is displayed again in full, "as derived in :numref:`sec_generalization_basics`"; a clause with the reference would do.

8. **Cuts with no loss.** (a) The paragraph on the general two-sided $p$-value with the cap at 1 ("For a general statistic $T$ with observed value $t$"), which interrupts the $z$-test between its figure reference and its figure and is needed only by Exercise 5(d), where it is restated. (b) The restatement of Markov's inequality, which :eqref:`eq_mdl-markov` already gives. (c) The sentence listing the two properties behind the $t$ distribution, which the text has just declined to prove. (d) The vector delta-method paragraph could shrink to one sentence. Together about 250 words.

9. **Missing for a deep learning reader.** The names a reader will search for: the paired sign-flip test on disagreements is McNemar's test in its exact form, and the comparison of seed means is what Welch's $t$-test does; neither name appears in the chapter. One clause each. A sentence on reporting a confidence interval for the difference (not only a $p$-value) would also fit "Tests and Intervals".

## Passage-level findings

**STA-1** · moderate · language · introduction
> "such a number as a property of the model. It is also a property of the"

The first paragraph is the first paragraph of the chapter introduction with small changes. `index.md`: "Every model in this book was judged by a number computed on data, such as a test accuracy or a validation loss. We read such a number as a fact about the model. It is also a fact about the particular examples that happened to be drawn." Here: "Every model in this book has been judged by a number computed on held-out data: a test accuracy, a validation loss, a score on a benchmark. We read such a number as a property of the model. It is also a property of the particular examples that happened to land in the test set." A front-to-back reader meets the same three sentences twice, three sections apart. This section needs them more; rewrite the chapter introduction's opening, or start here directly at "Suppose a classifier answers 912 of 1,000 test questions correctly."

**STA-2** · minor · language · introduction
> "Throughout, we assume that the estimator or the test is chosen"

This is the most important caveat of the section, and the closing paragraph builds on it, but here it is appended to a paragraph about the coin. Give it its own sentence at the end of the second paragraph, with the reason: "Every guarantee below holds for an estimator or a test chosen before the data are seen; the last paragraph says what goes wrong otherwise."

**STA-3** · minor · thought · "Estimators and the Sample Proportion"
> "wrote $p$ for the known bias of a coin; here the accuracy is unknown, so we"

The previous section already switched to $\theta$ for the coin's bias, with the words "as in the rest of the chapter we write it $\theta$". The two remarks contradict each other about when the notation changes. Settle it at chapter level (see "For the chapter review").

**STA-4** · minor · explanation · "Estimators and the Sample Proportion"
> "have a standard deviation of $\sqrt{0.0090^2+0.0085^2}\approx0.012$, larger"

The 0.0085 is the standard error at 0.921, which the reader has to work out. Add "(0.0085 is the same formula at 0.921)". The conclusion "A single comparison of this kind cannot tell the models apart" would be stronger with the number: the gap is 0.7 standard errors.

**STA-5** · minor · pedagogy · "Bias and Variance"
> "define it, we meet one more estimator whose bias matters."

The text raises a question ("Which of the two is better?"), says the answer needs "one number that counts both kinds of error", and then postpones it for a 600-word subsection on the sample variance. The sample variance is needed later (the $t$ interval) but not for the question just asked. Move "The Sample Variance" after "The Shrunk Mean and the Trade-off", where its bias $-\sigma^2/n$ can be read as another point on the bias–variance trade (Exercise 7 does exactly this), or before "Bias and Variance" as the first example of a biased estimator. Either removes the announced detour.

**STA-6** · minor · thought · "The Bias–Variance Decomposition"
> "exactly for the empirical distribution of the 10,000 estimates as well,"

Two sentences and a code comment are spent on why the printed "sum" equals the printed "MSE" to round-off (it is an algebraic identity for the empirical distribution). This means the cell does not test the proposition; it tests arithmetic. The honest reading, and the more useful one, is that the simulation checks the predicted values $0.533$, $0.123+0.225$ and the median's $0.80$. Say that the columns agree identically and that the content is the comparison with the formulas, in one sentence.

**STA-7** · minor · thought · "The Shrunk Mean and the Trade-off"
> "flexible models that fit their training data exactly can still predict"

The paragraph on prediction ends "very flexible models that fit their training data exactly can still predict well", a large claim with no reference. Point to where the chapter shows it (:numref:`subsec_mdl-double-descent-example` in the generalization-bounds section), or cut the sentence.

**STA-8** · minor · explanation · "Consistency and Efficiency"
> "floor known as the Cramér–Rao bound."

Named with no reference, although the next section proves it (`subsec_mdl-cramer-rao`). Add the `:numref:`. In the same subsection Markov's inequality is restated in full; it is :eqref:`eq_mdl-markov` of the chapter's first section and is restated a third time in `mdl-concentration-generalization.md` ("so we restate both"). Cite it here in a clause.

**STA-9** · moderate · pedagogy · "Two Kinds of Error and Power"
> "bins, once under $H_0$ and once with a true mean of $0.8$."

See structural finding 2. This one-sample design ($n=20$, $\sigma=1$, $\mu=0.8$) rejects almost always ("about 97% of them fall below $0.1$"), and thirty lines later the text says the model comparison with $\delta=0.8$ and 20 seeds has power 0.72. A reader will take these for the same experiment and see a contradiction. Run the histogram on the two-sample design of the running example (20 seeds per model, gap 0 and gap 0.008, the two-sample $z$-statistic), so that the pile-up below 0.05 is the 0.72 that the formula then explains.

**STA-10** · minor · language · "The Permutation Test" and "The Bootstrap"
> "saw. So we shuffle the labels, recompute the gap, and repeat $B$ times. The"

$B$ shuffles of the accuracies of model B. See structural finding 3.

**STA-11** · minor · explanation · "The $z$-Test"
> "doubles the smaller tail,"

The paragraph on the general two-sided $p$-value sits between the reference to the rejection-region figure and the figure, uses $T$ and $t$ for a statistic and its value one line after $T$ was the $z$-statistic, and is not used again in the text. Move it to Exercise 5(d), which needs it and already restates the formula.

**STA-12** · moderate · pedagogy · "Paired Comparisons"
> "Every run gives $p\approx0.21$, so the gap is still not significant. The"

This is the answer to the question the section opened with, and it is given without the comparison that was promised. Draft: "Every run gives $p\approx0.21$. Treated as two independent samples, the same counts give a gap of 0.7 standard errors and $p\approx0.47$; pairing halves the standard error of the gap, from 0.012 to $\sqrt{41}/1000\approx0.006$, because the 959 questions on which the models agree no longer contribute noise. The evidence is the split of 25 against 16 among the 41 disagreements, and it is still not enough: a nine-question lead on a thousand questions does not show that B is better." Also name the test (McNemar's test in its exact form) so that a reader can look it up.

**STA-13** · moderate · explanation · "Definition and Interpretation"
> "whatever $\theta$ is. The Wald interval used here meets that guarantee only"

"The Wald interval" is defined two subsections later ("often called the *Wald interval*"), and no interval has been built at this point; "used here" has nothing to refer to. The reader also meets the coin's interval $(0.44,0.94)$ as a bare number "computed below". Move the two paragraphs on the credible interval to follow the Wald computation in "The Gaussian Interval", where both intervals for 9 of 13 can be shown together and the coverage numbers (91% and below 50%) have something to qualify.

**STA-14** · minor · correctness (sure) · "Multiple Testing"
> "$p\le\alpha$ is called *$p$-hacking*, and it turns noise into apparent"

What is described (reporting only the comparisons that cleared the threshold) is selective reporting. $p$-hacking usually means trying analyses, subsets or stopping times until one is significant. Both inflate the false-positive rate for the same reason, so write "Reporting only the winners, or trying analyses until one clears $p\le\alpha$ (*$p$-hacking*), turns noise into apparent findings."

**STA-15** · minor · correctness (sure) · "Paired Comparisons"
> "Pairing saves a factor of about four here, and more when the models agree"

The factor of four compares 3,100 questions with 13,500 "questions each" for two independent test sets, that is, 27,000 labeled questions in total. Per model evaluated the saving is 4.3; per labeled question it is 8.7. Say which: "a factor of about four in the size of each test set, and about nine in the number of questions that must be labeled". The Summary repeats "the factor is about four".

**STA-16** · minor · explanation · "The Gaussian Interval"
> "take this fact on faith. It rests on two properties of Gaussian samples:"

Having declined to prove the fact, the text lists the two lemmas it rests on, which the reader cannot use. Keep the first sentence and the reference to Student, drop the lemmas, and note that this is the Student-$t$ that `mdl-exponential-family.md` used as its example of a heavy-tailed law outside the family (neither section points to the other).

**STA-17** · minor · thought · "Bootstrap Standard Errors"
> "The bootstrap standard error moves between about $0.05$ and $0.10$ from one"

The stored run prints "median 0.830, bootstrap se 0.105", so the reader sees a value outside the stated range directly above this sentence, and a median 0.14 from $\log2$ against "usually by less than $0.15$". My simulation of 300 data sets puts the central 95% of the bootstrap standard error at 0.048 to 0.102 and its median at 0.070, so the prose is accurate and the stored draw is a 3% event. Widen the wording slightly ("between about 0.05 and 0.11") so that no recapture contradicts it. The same paragraph says the bootstrap distribution is lumpy; the stored interval $(0.665,1.033)$ barely contains the true median, which is worth one honest clause.

**STA-18** · minor · explanation · "Bootstrap Standard Errors"
> "near $0.07$. A formula for the median's standard error does exist,"

The part opened with "it has no convenient formula" and the cell comment says "its median's se formula needs the unknown density"; the formula then appears after the result. Give the formula before the cell as the prediction ($1/\sqrt{200}\approx0.071$ here, computable only because we know the density), so that the cell tests it, as the accuracy cell just before does.

**STA-19** · minor · explanation · "Failures of the Bootstrap"
> "The bootstrap also fails when the observations are not independent, and"

It is the bootstrap that resamples single observations that fails; the cluster bootstrap in the last sentence of the paragraph is still a bootstrap. Write "The bootstrap fails when it resamples units that are not independent".

**STA-20** · moderate · language · end of "Failures of the Bootstrap"
> "We can now attach an error bar to an estimate and decide whether a"

The body ends with a recap in the "We can now" formula, placed under the heading "Failures of the Bootstrap", and the Summary follows with the same list. `mdl-exponential-family.md` ends with the same construction ("We can now write the distribution"). Keep the second half of the paragraph, from "Two questions remain open", which is a real hand-over (the remark that choosing the best of many configurations "is multiple testing in another form" is excellent), and delete the first two sentences.

**STA-21** · minor · language · pattern, whole section
> "Thus the bias says where the estimates land on average, and the standard"

Sentence-initial "Thus" five times, "So" five times and "In words" twice, usually introducing a restatement of the formula just displayed (also "Thus the best shrinkage always beats the sample mean", "Thus a small $p$ says that", "Thus the event"). In most cases the sentence stands without the connective. Terms at their definition are bold here (**bias**, **standard error**, **power**) and italic in the neighbouring sections; choose one for the chapter.

**STA-22** · minor · language · Summary
> "Pairing two models on the same questions reduces it; for a one-point gain"

The Summary has nine sentences and repeats worked numbers (the factor of four, $1.96\,s/\sqrt n$). Reduce to the five results: the sampling distribution; MSE equals squared bias plus variance; a test errs at rate $\alpha$ and its power is set by $\delta\sqrt n$; an interval is the set of values a test keeps; the bootstrap and where it fails.

## Exercises

Twelve problems, at the top of the band, and the strongest set I reviewed. Almost every one produces a number the reader does not expect: 2(b) (36% of significant results are false at power 0.8 and 69% at power 0.2, which I checked), 4(b) (short intervals cover less often than long ones at $n=2$), 5 (the delta method fails by the count of errors, not by $n$), 7 (the MSE-optimal variance denominator is $n+1$, and under kurtosis $\kappa$ it is $n+1+(n-1)\kappa/n$, which I verified from the stated variance), 8(c) (an exact sign-flip test with 5 seeds cannot reject at 0.05, while the Monte Carlo version does in a few percent of runs), 10 (the cluster bootstrap, with $\rho=0.1$ from $\mathrm{Beta}(6.3,2.7)$, also checked).

- **1(a)** asks the reader to reproduce `fig_mdl-bias-variance-u-curve` by simulation and compare the minimizer with $c^\star$. The text already shows the exact curves. Cut (a) and keep (b) and (c), which are the interesting parts.
- **11 (Stein's paradox) and 12 (peeking, mixture martingales)** are each a small project. Both are excellent and relevant; 12 in particular answers a question practitioners have. Mark both clearly as extended (12 is; 11 is marked only with two dots), and consider whether 12 is one exercise or a future subsection, since "do not stop a run when it first looks significant" is advice the text itself never gives.
- **Cell references.** Five exercises refer to cells by ID ("the `#statistics-bootstrap` cell", "`#mdl-statistics-a-gaussian-example-1`"). I could not confirm that these IDs are visible in the rendered book; if they are not, the reader cannot find the cell. Refer by subsection title instead.
- **6** carries the label `ex_mdl-statistics-let-x-1-ldots`, and the text cites it for the claim that "the percentile interval never contains $\theta$", but the exercise measures the coverage of the *basic* interval and never mentions the percentile interval. Either add a part on the percentile interval or change the citation in the text.
- The JAX tab's discussion link is the TensorFlow one (`/t/1103` in both).

## Still open from October

The October review's one finding that touches this section, that maximum likelihood used consistency, the law of large numbers and Cramér–Rao before Statistics defined them, is resolved: Statistics now precedes maximum likelihood, and it names Cramér–Rao with a forward mention (which needs its reference, STA-8).

## For the chapter review

- **Order.** Correctly placed after the exponential family (which ends by asking how far $9/13$ can be from the truth) and before maximum likelihood (which needs consistency, the standard error and the law of large numbers). The hand-over at the end to maximum likelihood and to generalization bounds is the best transition in my two sections.
- **Overlaps.** Markov's inequality is stated in `mdl-random-variables.md`, again here, and again in `mdl-concentration-generalization.md`; the weak law is proved here from Chebyshev's bound and the concentration section then opens on how weak that bound is, which works well as a pair if each cites the other. The credible interval and the Beta posterior lean on the exponential-family section. The first paragraph duplicates the chapter introduction (STA-1).
- **The chapter's coin (9 of 13) is a guest here.** It appears three times as "a check" while the real running example is the classifier with 912 of 1,000. That is fine, but the chapter should fix the symbol for the bias ($p$ in the first two sections, $\theta$ from the conjugate-prior part of the exponential-family section on) at one stated point.
- **Patterns I expect across the chapter:** the "We can now ..." closing recap followed by a Summary that repeats it; reuse of single letters for different quantities within a section; no subsection labels, which limits lookup from the main text; and, on the positive side, simulations introduced by a question and read with their own run-to-run noise, which is the practice the Basics review asked for and which this section does better than anything in Basics.
- **Scope.** This section is where the book's own experimental practice (ranges instead of point values, seeds, paired evaluation) is justified. The main text's remark on why results are reported as ranges should point here by label.
