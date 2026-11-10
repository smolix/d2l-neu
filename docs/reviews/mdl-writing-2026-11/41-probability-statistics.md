# Chapter 41, Probability and Statistical Learning: writing review

Sources: the eleven `.md` files of `chapter_mdl-probability-statistics/`, each read in full to the slides marker by a section reviewer. This file is assembled from the eleven section reviews; I reread the sources only to check claims that cross sections (listed under "Checked in the sources" at the end). About 63,000 words of body prose in the nine sections, 1,700 in the further-reading notebook, 800 in the introduction, and about 19,000 words in 100 exercises.

Section reviews, with the short name used below to cite them:

| Short name | Section review | Finding tags |
|---|---|---|
| `index` | [index](41-probability-statistics/index.md) | IDX-n; structural bullets unnumbered |
| `rv` | [mdl-random-variables](41-probability-statistics/mdl-random-variables.md) | RV-n; structural findings 1 to 6 |
| `dist` | [mdl-distributions](41-probability-statistics/mdl-distributions.md) | DIST-n; structural findings 1 to 8 |
| `expfam` | [mdl-exponential-family](41-probability-statistics/mdl-exponential-family.md) | EXP-n; structural findings 1 to 6 |
| `stats` | [mdl-statistics](41-probability-statistics/mdl-statistics.md) | STA-n; structural findings 1 to 9 |
| `mle` | [mdl-maximum-likelihood](41-probability-statistics/mdl-maximum-likelihood.md) | ML-n; S1 to S6 |
| `bayes` | [mdl-bayesian-computation](41-probability-statistics/mdl-bayesian-computation.md) | BC-n; S1 to S6 |
| `conc` | [mdl-concentration-generalization](41-probability-statistics/mdl-concentration-generalization.md) | CON-n; S1 to S6 |
| `genb` | [mdl-generalization-bounds](41-probability-statistics/mdl-generalization-bounds.md) | GEN-n; S1 to S6 |
| `nb` | [mdl-naive-bayes](41-probability-statistics/mdl-naive-bayes.md) | NB-n; NB-S1 to NB-S5 |
| `fr` | [further-reading](41-probability-statistics/further-reading.md) | FR-n; FR-S1 to FR-S4 |

**How findings are cited.** Every finding is written `short name/ID`, for example `conc/CON-13`. The tags CON, GEN and FR are likely to recur in other chapters (GEN was already used in the Basics review of chapter 4), and four files number their structural findings S1 to S6, so an ID without its file is ambiguous. Structural findings are written `rv/S1` (the first numbered structural finding of that file), `mle/S1`, `nb/NB-S1`, `fr/FR-S1`.

## Verdict

This is a strong chapter, and the strongest evidence is what the reviewers did not find: the section reviewers recomputed the worked numbers of every section and found no error in any proposition or proof. The teaching method is consistent and good. Each section has one small object that every definition is computed on (the dart, the coin with 9 heads in 13 flips, the server with four requests a minute, the classifier with 912 of 1,000, the Cauchy location model, a two-parameter posterior checked on a grid, the 1,000 fair coins, the two-word spam filter), most cells test a prediction that could have failed, and the exercise sets are, by the reviewers' account, the best in the book. Several stretches reach the bar: the first half of Random Variables, the Poisson and exponential construction in Distributions, most of Statistics, the Fisher-information part of Maximum Likelihood, most of Bayesian Computation, Markov to Hoeffding in Concentration, and the Rademacher part of Uniform Convergence.

What keeps the chapter below the bar is its shape and not its sentences. (1) Five of the nine sections hold two to five subjects of equal weight, and two of them are really two sections. (2) The same material is taught two or three times: inside the chapter (Gibbs' inequality is proved twice by the same argument, in two sections whose reviewers each noticed only the third proof in chapter 42), in chapter 42, and in the main text, most seriously in the first and third parts of Uniform Convergence. (3) Every one of the nine sections ends with a paragraph that begins "We can now" and restates the results just before a Summary that restates them again. (4) The applications in the main text are named in every section and referenced by label in almost none, and the chapter never says that the Bayesian tools it teaches return later in the book for other targets. (5) In two places a run supports less than the prose says: the stored output of the Bayesian running example is an atypical data set that contradicts the text, and the double-descent sweep is read as a stronger result than it is. There are also about eight statements of moderate severity to repair, none in a proof. What matters most is deciding the homes of the repeated material and the section boundaries; the rest is local.

## Scores

1 to 5, as the section reviewers gave them; 5 means the reviewer would be happy to find the page in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](41-probability-statistics/index.md) | 4 | 4 | 3 | 3 | 4 |
| [mdl-random-variables](41-probability-statistics/mdl-random-variables.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-distributions](41-probability-statistics/mdl-distributions.md) | 4 | 3 | 4 | 4 | 3 |
| [mdl-exponential-family](41-probability-statistics/mdl-exponential-family.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-statistics](41-probability-statistics/mdl-statistics.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-maximum-likelihood](41-probability-statistics/mdl-maximum-likelihood.md) | 4 | 4 | 4 | 4 | 3 |
| [mdl-bayesian-computation](41-probability-statistics/mdl-bayesian-computation.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-concentration-generalization](41-probability-statistics/mdl-concentration-generalization.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-generalization-bounds](41-probability-statistics/mdl-generalization-bounds.md) | 4 | 3 | 4 | 4 | 4 |
| [mdl-naive-bayes](41-probability-statistics/mdl-naive-bayes.md) | 4 | 4 | 4 | 4 | 4 |
| [further-reading](41-probability-statistics/further-reading.md) | 4 | 4 | 4 | 4 | 3 |

The table does not discriminate: 41 of the 45 scores for the nine sections are 4. The verdict paragraphs do. Four section reviews each call their section the best the reviewer had read ("the best-taught section I would expect to find in the part" for Random Variables, "the strongest section I read in the chapter" for Bayesian Computation, "the most useful statistics section for a deep learning reader that I know of", "one of the best-taught" for Naive Bayes), and three section reviews gave partial scores of 5 that the table hides: the first movement of Random Variables, the first half of Concentration, and the Rademacher part of Uniform Convergence, each 5 for explanation. Read the 4s as "close to the bar with a structural problem", and the three 3s for thought and enjoyment (Distributions, Maximum Likelihood, Uniform Convergence) as the places where the structural problem is largest.

## Correctness

No section reviewer found an error in a proposition or a proof, and every worked number was recomputed. The findings below are statements in the surrounding prose, hypotheses, and readings of outputs. Ordered by severity; the confidence is the reviewer's.

**Moderate.**

1. `conc/CON-13` (sure). After the norm-concentration proposition the text says that beyond $\varepsilon=1$ "the tail becomes $e^{-d\varepsilon/8}$". That is the tail of the squared norm. The norm keeps its Gaussian tail at every scale, and the restriction $\varepsilon\le1$ is not needed for its upper tail. As written the reader learns something false.
2. `expfam/EXP-9` (sure). A generalized linear model is defined as the canonical-link case only. The Summary's "A generalized linear model trains with the gradient input times prediction minus observation" is false in general, and the probit model the same section discusses 80 lines later is a counterexample.
3. `rv/RV-10` (confident about the facts). "The rule needs a finite variance" follows a sentence that names the law of large numbers, so the reader learns that the law of large numbers needs a finite variance. It needs a finite mean; the $\sigma/\sqrt n$ rule needs the variance.
4. `rv/RV-1` (confident). "Has a density" is defined twice (by a limit of a ratio, and by the integral), and the proposition $F'=p$ is nearly the definition under the first and a theorem under the second. No formula is wrong; the logic is circular at the place a mathematical reader checks.
5. `expfam/EXP-14` (sure; an inconsistency). The Beta parameters are "prior counts" $a$, $b$ in the exponential-family section and "$a-1$ pseudo-counts ... and $b-1$" in Maximum Likelihood, and exercise 9(b) of the exponential-family section uses the second convention inside the first section. Confirmed in the sources (`mdl-exponential-family.md` lines 995 to 1010, `mdl-maximum-likelihood.md` line 1190). See also `expfam/S4`.
6. `genb/GEN-16` (fairly sure). The threshold 1 for the "capacity product" is carried over from a bound for losses in $[0,1]$ to a regression with squared loss on unbounded targets, where no theorem of the section applies and the product changes when the targets are rescaled. "Stays above 1" is then repeated as a finding in the closing paragraph and the Summary.
7. `genb/GEN-7` (fairly sure). The theorem is stated with the expected complexity $\mathfrak R_n$; everything computed afterwards is the empirical complexity of the sample at hand, with $r$ read off the sample. One sentence (the empirical version holds at the price of a second confidence term) covers the computations.
8. `conc/CON-9` (fairly sure). Bernstein's interval is computed with the variance estimated from the measured error rate and described as valid at every $n$. It assumes the variance known, which is the substitution the section had just criticized in the Gaussian interval. The self-consistent half-widths are $\pm0.43\%$ and $\pm0.79\%$ instead of $\pm0.39\%$ and $\pm0.76\%$; the conclusion stands.

**Minor.**

9. `rv/RV-9` (confident). "Covariance is linear in each argument" displays only homogeneity and shift invariance; additivity, which the text uses three paragraphs later, is not stated.
10. `rv/RV-7` (confident). Power laws are called "the extreme case" of heavy tails; they are the standard example.
11. `dist/DIST-9` (confident). A heuristic derivation is called "the de Moivre–Laplace theorem", and its sum from the mean has no meaning for the running coin, where $np=6.5$.
12. `expfam/EXP-8` (fairly sure). The maximum-entropy proposition says "any distribution" and needs a density in the continuous case and a finite $\mathbb{E}_q[T]$.
13. `expfam/EXP-10` (sure). "No linear combination of the statistics is constant" needs "nontrivial" and "on the support"; "which requires" should be "which holds when".
14. `expfam/EXP-5` (sure). Tagged correctness by the reviewer, but the statement is correct; the point is that the non-uniqueness the section later needs is not the one mentioned.
15. `stats/STA-14` (sure). What is described as $p$-hacking is selective reporting.
16. `stats/STA-15` (sure). "Pairing saves a factor of about four" is per test set; per labeled question it is about nine. The Summary repeats the four.
17. `mle/ML-6` (sure). The hinge loss "is not the NLL of any conditional distribution" needs "up to an additive constant", as exercise 4 already says.
18. `mle/ML-15` (sure). The ELBO proposition requires $q(z)>0$ exactly on the posterior's support; the condition is not needed and depends on the parameter the M-step changes.
19. `conc/CON-16` (sure). "About 570 directions are pairwise within" states what a loose union bound guarantees as if it were a count.
20. `genb/GEN-18` (sure). A fact about square Gaussian matrices is asserted of a matrix of ReLU features.
21. `nb/NB-4` (sure). Naive Bayes has 7,849 free numbers by the text's own count, not "the same 7,850 parameters" as softmax regression. The claim that matters (same family of posteriors) is true.
22. `nb/NB-5` (sure). $10^{-236}$ "comes close to the floor of double precision"; it is 72 orders of magnitude above it.
23. `fr/FR-3` (sure). The Cauchy is the symmetric stable law with $\alpha=1$, not "the" stable law.
24. `fr/FR-4` (fairly sure). Independence of irrelevant alternatives is Luce's axiom; the text can be read as attributing it to McFadden.

Not tagged correctness by the reviewers but of the same kind: `genb/GEN-6` (the convention "predictors take values in $[-1,1]$" is broken by the section's main example, so "1 is the maximum" is false for the linear class two subsections later); `genb/GEN-14` (from one random labeling to "fits every sign pattern"); `bayes` exercise 8b ("a few percent low" is true of the evidence, not of its logarithm, which the problem asks for); `mle/ML-3` ("about twenty million" terms before a float32 sum stalls; the reviewer reproduced about 23 million).

**Stored outputs against the prose.**

- `bayes/S1` (major). The data of the running example are redrawn on every run, and the committed PyTorch output is an unrepresentative draw: the page prints a predictive probability 1.7 times the plug-in where the text says "several times" and "typically three to ten times"; the Laplace standard deviations are below the 5th percentile of 1,500 simulated runs; and the true value lies outside the printed 90% range that the text says contains it nine times in ten. The prose is accurate for a typical run. This is the only place in the chapter where a reader of the printed page sees the output contradict the headline claim. The fix is to fix the twenty data points and leave the samplers unseeded.
- `stats/STA-17`. The stored bootstrap standard error is 0.105; the prose says it "moves between about 0.05 and 0.10". The prose is right about the distribution and the stored draw is a 3% event; widen the range slightly.
- `nb/NB-8`. "A handful of pixels for most classes" against a printed `[0 13 1 0 3 3 0 57 0 1]`.
- `nb/NB-9` (verified numerically). "In double precision 44% of the numerators fall below" is true and misleads: the float64 product classifies as well as log space (0.7058 against 0.7059), because the underflowing numerators belong to losing classes.
- `genb/S6`. Stored 97% against "about 96%" in the prose (exact value 96.3%); harmless.
- `stats` exercise 6. The text cites this exercise for a claim about the percentile interval; the exercise measures the basic interval.
- `mle/ML-14`. "The figure also shows the two limits"; the figure shows one prior.
- `conc/CON-2`. The sentence on Chebyshev's majorant and panel (a) of the figure do not describe the same function.

**Numbers and claims that no cell prints.**

- `genb/GEN-17` (major) and `genb/GEN-19` (major). The sweep's conclusion ("roughly ten times below the best underparameterized model", and in the Summary "ten times better than the best classical fit") compares against a classical regime that explains a quarter to a third of the variance; least squares on the 15 raw inputs would be about nine times better than the widest interpolant. The spectrum that the explanation of benign overfitting relies on is never printed.
- `bayes/S4`. The ELBO is named as the check of variational inference in the table, the closing and the Summary, and is never computed in the text.
- `nb/NB-17`. "Hundreds of such groups" of duplicated pixels; one pair was measured.
- `nb/NB-11`. Why the direct fit gains twelve points is asserted twice and tested only in exercise 12(d).
- `mle/ML-9`. Tolerances for the Monte Carlo check stated without derivation and three to five times the true standard errors.
- `dist/DIST-6`. "Each tenfold increase" for $n=20, 100, 1000$.

## Arc and order

**One line of argument, and the introduction states it.** The introduction names four objects (distribution, estimator, posterior, generalization bound), and the chapter follows them: Random Variables, Distributions and the Exponential Family are about the distribution; Statistics and Maximum Likelihood about the estimator; the conjugate-prior part of the Exponential Family, the MAP part of Maximum Likelihood and Bayesian Computation about the posterior; Concentration and Uniform Convergence about the bound. Naive Bayes applies the first three. The coin with 9 heads in 13 flips is carried through the introduction, Distributions, the Exponential Family, Statistics (as a guest), Maximum Likelihood, Bayesian Computation (where its curvature 61 returns, `bayes/BC-6`) and Concentration (`conc/CON-8`), and it is the chapter's best unifying device (`mle` and `expfam`, chapter notes). The only thing the introduction gets wrong is its route paragraph, six "asks how / which / why" sentences that should be rewritten as four movements (`index/IDX-1`).

**The order is right and should not change.** Every reviewer checked the position of their sections and agreed. The October complaint that Maximum Likelihood used consistency and the weak law before Statistics defined them is fixed (`stats`, `mle`, "Still open from October"). The hand-offs from the Exponential Family to Statistics, from Statistics to Maximum Likelihood, and from Concentration to Uniform Convergence ("a quantity fixed before the data" against "a function chosen after it") are stated in the text and are accurate.

**Sections that are really two.** The section reviews were written independently, and four of them propose a split or a reorder. Taken together:

- Random Variables splits at "Several Variables" into one-variable theory and joint distributions with the change of variables (`rv/S1`). The seam is clean and the dart survives it. A reader sent to `subsec_mdl-change-of-variables` now lands 1,250 lines into the file.
- "Latent Variables, EM, and the ELBO" should leave Maximum Likelihood and become a section of its own before Bayesian Computation (`mle/S1`). It already carries a `sec_`-level label and is the target of references from at least seven other files; the ELBO is then stated once (`bayes/BC-14`).
- The multivariate Gaussian (1,800 words) is different in kind from the coin-built distributions and is what other chapters need most; it could be its own section (`dist/S7`). The reviewer would make the move but does not insist.
- The Exponential Family holds five subjects of 1,100 to 1,300 words each with no ranking, and the result its opening promises (the common gradient "prediction minus observation") arrives third; reorder to form, log-partition function and GLMs, softmax, conjugate priors, and maximum entropy last (`expfam/S1`, `expfam/S2`).
- Statistics and Concentration each have two halves and their reviewers argue against splitting (`stats/S5`, `conc/S1`). Concentration and Uniform Convergence are two sections with the boundary in the right place (`conc` chapter note, `genb/S1`).

I recommend the first two splits and leave the third to the author. With them the chapter has eleven sections: one variable; several variables and transformations; distributions; the exponential family; statistics; maximum likelihood; latent variables and the ELBO; Bayesian computation; concentration; uniform convergence; naive Bayes. The total length falls in spite of the added headings, because the cuts listed below remove more than the splits add.

**Length is not proportionate to importance in two places.** Maximum Likelihood is the chapter's hub (other chapters point into it for four different things) and its heaviest section at 8,900 words, a fifth of which re-teaches material with a home elsewhere (`mle/S2`, `mle/S3`). Uniform Convergence spends 3,000 of 6,400 words on parts that repeat the main text (`genb/S2`, `genb/S3`) and should come down to about 4,800.

**Boundaries that leak.** Each of these is either a pair of section reviews describing the same passage from both sides, or one side only because the other reviewer could not see it:

- Random Variables and Distributions: the Cauchy averaging experiment, cell and sentence (`rv/RV-6`, `dist/DIST-11`); the pair $(X, SX)$ (`rv` exercise 5, `dist` chapter note); the eigenvectors of a covariance matrix (`rv/RV-12`, `rv/S6`).
- Exponential Family and Maximum Likelihood: entropy, KL divergence and Gibbs' inequality (see Repetition); the Beta update and the pseudo-count convention (`expfam/EXP-14`, `expfam/S4`).
- Maximum Likelihood and Bayesian Computation: the ELBO and its gap, derived by two routes (`bayes/BC-14`); "Posterior Mode and Posterior Distribution", which says in advance what the next section's opening says better (`mle/ML-13`).
- Concentration and Uniform Convergence: the best-of-1,000-coins result given away as a teaser (`conc/CON-19`, `genb/GEN-1`); the many-models motive in the introduction of the wrong section (`conc/S3`, `conc/CON-1`); the union bound defined in both (`conc/CON-6`); Chebyshev against Hoeffding under a union bound told for distances and again for classes (`conc/CON-17`).
- The introduction and Statistics open with the same three sentences (`stats/STA-1`).

**Two hand-offs point to the wrong place.** Random Variables closes by assigning the question of how fast averages concentrate to the law of large numbers, and does not lead to the next section at all (`rv/RV-14`). Bayesian Computation hands over with "how far such an average can be trusted", which Concentration answers for independent draws only, and MCMC draws are not independent (`conc`, chapter note).

**The last section.** After Uniform Convergence the chapter's argument is finished; the reviewer of that section writes that naive Bayes "reads as an appendix to the chapter", and the reviewer of naive Bayes that the order reads as "tools, harder tools, hardest tools, then an unrelated easy topic". Both arrive at the same remedy: keep it last, name it as the chapter's worked case study in its first paragraph, and add two short links to the late sections, a Hoeffding or standard-error number for the 7,840 frequencies and the train and test gap (`nb/NB-S4`, `nb/NB-10`). At present the body uses nothing from the three sections before it.

## Most important findings

1. **Give entropy, KL divergence and Gibbs' inequality one home.** Inside this chapter the inequality is stated and proved twice by the same tangent-line argument, as a Lemma in the Exponential Family and as a Proposition in Maximum Likelihood, with entropy defined in both and no reference from the second to the first; chapter 42 then defines all three again, proves the inequality by Jensen, and sets the tangent-line proof as an exercise. `expfam/EXP-7`, `expfam/S3`, `mle/S2`. The two reviewers each saw one proof here and the one in chapter 42, and their recommendations are not compatible as they stand (see "Disagreements between section reviews"). The same applies to Gaussian maximum entropy, proved here and in chapter 42 down to the same remark (`expfam/S3`).
2. **Fix the data set of the Bayesian running example**, so that the printed page stops contradicting the prose, and remove the hedges about runs that a fixed data set makes unnecessary. `bayes/S1`, `bayes/S2`, `bayes/BC-1`.
3. **Cut the first and third parts of Uniform Convergence to what the main text does not already contain, and say what the double-descent sweep shows.** `genb/S2`, `genb/GEN-2`, `genb/S3`, `genb/GEN-15`, `genb/GEN-17`, `genb/GEN-19`, `genb/GEN-16`.
4. **Move EM and the ELBO into a section of their own and state the ELBO once.** `mle/S1`, `bayes/BC-14`.
5. **Split Random Variables at "Several Variables"**, and settle once, for the part, whether the appendix re-motivates what `sec_prob` motivated or only proves and extends it. `rv/S1`, `rv/S2`, `rv/S3`.
6. **Repair the eight moderate correctness findings.** `conc/CON-13`, `expfam/EXP-9`, `rv/RV-10`, `rv/RV-1`, `expfam/EXP-14`, `genb/GEN-16`, `genb/GEN-7`, `conc/CON-9`. Each is one to three sentences.
7. **Tell the deep learning reader why the Bayesian and Fisher material is here.** Bayesian Computation says three times that none of its methods scales to a network and never says that the ELBO, importance weights and Langevin proposals return in the book for other targets; Maximum Likelihood promises error bars from the Hessian that fail for networks and does not name where the Fisher matrix returns. `bayes/S3`, `mle/S5`, `mle/ML-10`, `mle/ML-12`.
8. **Make the opening examples pay off.** Statistics asks whether 921 of 1,000 beats 912 and answers 800 lines later without the unpaired number beside the paired one (`stats/STA-12`, `stats/S1`); Concentration drops its 8.3% classifier without naming its final interval (`conc/CON-12`); Maximum Likelihood opens on a training curve that levels off and leaves the practical consequence to an exercise (`mle/ML-5`).
9. **Reorder three blocks so each tool arrives when a number needs it.** The Gaussian subsection of Distributions and its looping pair of proofs (`dist/S1`, `dist/S2`, `dist/DIST-10`, `dist/DIST-14`); "Sub-Gaussian and Sub-Exponential Variables" in Concentration (`conc/S2`); the Exponential Family as a whole (`expfam/S1`).
10. **Replace the nine "We can now" paragraphs** by the open questions they already contain, and bring every Summary to the three to six sentences the guide asks for. See Recurring patterns.

## Repetition

Proposed single home in each case. Where two reviewers proposed homes, I give both.

**Inside the chapter.**

| What | Where it is taught | Home |
|---|---|---|
| Entropy, KL, Gibbs' inequality with the tangent-line proof | Exponential Family ("The Maximum-Entropy Proposition"), Maximum Likelihood ("Cross-Entropy and the Entropy Floor"); `expfam/EXP-7`, `mle/S2` | Chapter 42 for the definitions and the full treatment. In this chapter the tangent-line proof once, in the Exponential Family, where the maximum-entropy proposition needs the general form; Maximum Likelihood cites it and keeps the regrouping identity, the coin's 0.617 nats and the figure. |
| ELBO and its gap | Maximum Likelihood, Bayesian Computation; `bayes/BC-14` | The new latent-variable section (`mle/S1`). |
| Beta posterior and pseudo-counts, two conventions | Exponential Family, Maximum Likelihood; `expfam/EXP-14`, `expfam/S4` | Exponential Family, with the convention stated once ($a$, $b$ for the mean, $a-1$, $b-1$ for the mode). |
| Mode depends on the parameterization | Two consecutive subsections of Maximum Likelihood; `mle/ML-13` | One paragraph at the end of the Beta subsection. |
| Cauchy averages do not settle | Random Variables ("Heavy Tails"), Distributions ("The Cauchy counterexample"), same cell and sentence; `rv/RV-6`, `dist/DIST-11` | Random Variables. Both reviewers agree. |
| Uncorrelated does not imply independent | Five times in Random Variables, once in Distributions; `rv/RV-11` | The dart under "Covariance". |
| A density is a rate, not a probability | Four times in 140 lines; `rv/RV-2` | The speed analogy. |
| Markov's inequality | Stated in Random Variables, restated in Statistics and in Concentration ("so we restate both"); `stats/STA-8`, `stats/S8` | Random Variables; a clause with `eq_mdl-markov` elsewhere. |
| Union bound | Defined in Concentration for two disjoint tails, again in Uniform Convergence; `conc/CON-6` | Uniform Convergence, or the first real use (the count of random directions). |
| Pairwise union bound over $m^2/2$ pairs | Twice in 120 lines and in an exercise; `conc/S5` | "Near-Orthogonality of Random Directions". |
| Best of 1,000 coins, 66 heads | Teaser in Concentration, six times in Uniform Convergence; `conc/CON-19`, `genb/GEN-1` | The cell in Uniform Convergence. |
| The coin table at $n=1{,}000$ | Printed twice, read three times; `conc/S4` | The first table. |
| Near-orthogonality proved twice, Gaussian then signs | `conc/CON-15` | Signs first, Gaussian in a sentence. |
| The four losses | Opening, subsections and Summary of Distributions; `dist/DIST-2` | The subsections. |
| "Four facts" of the exponential family | Introduction, next paragraph, closing, Summary; `expfam/EXP-2`, `expfam/EXP-18` | Introduction and Summary. |
| Correlated evidence is counted many times | Six times in Naive Bayes; `nb/NB-S1`, `nb/NB-2` | "Calibration", where it is measured. |
| The coin's $\pm0.13$ | Four times in Maximum Likelihood; `mle/ML-17` | The error-bar subsection. |
| Opening paragraph | Introduction and Statistics; `stats/STA-1` | Statistics; rewrite the introduction's. |
| Student-$t$ | Mentioned without definition in the Exponential Family and in Statistics, defined in an exercise of Distributions; `expfam/EXP-6`, `stats/STA-16` | The exercise, cited from both. |

**With other appendix chapters.** Gibbs and Gaussian maximum entropy with chapter 42 (above). Near-orthogonality of random directions with the linear-algebra chapter: restated in "High dimension" in Distributions and again at the head of the subsection in Concentration (`dist/DIST-13`, `conc/CON-15`); home for the exponential tail is Concentration, for the variance computation linear algebra, and the paragraph in Distributions goes. PCA and elliptical contours with the SVD section (`rv/RV-12`). Learning rate against loss scale belongs to chapter 40 (`mle/ML-4`).

**With the main text.**

- `sec_prob`: mean, variance, covariance, correlation and their motivation (`rv/S2`); the Bernoulli variance and "the fair coin is least predictable" (`dist/DIST-4`); the variance of a sample proportion (`stats/S7`). Keep the proofs and the extension to densities; cut each re-motivation to a clause.
- Linear regression and weight decay: the Gaussian NLL and the Gaussian prior, as propositions with proofs at the same depth as the main text (`mle/S3`). State with a reference; keep the additions (hinge counterexample, mean against median, the Beta prior).
- `chap_classification_generalization`: Hoeffding plus the union bound, the radii, a selection experiment and the leaderboard attack (`genb/S2`, `genb/GEN-2`). The main text is the home; the appendix keeps the figure, the sample-size form and the measured conservatism.
- `subsec_double_descent`: the frozen-feature sweep and the nesting argument (`genb/S3`, `genb/GEN-15`). The appendix keeps the SVD formula and the clean and noise split.
- Softmax regression: the confusion matrix, redefined with the opposite orientation (`nb/NB-12`); the Gumbel proof, which is also parts (a) to (c) of a main-text exercise (`expfam`, exercise 1).
- The bias and variance decomposition for prediction, displayed again in full (`stats/S7`).

## Recurring patterns

**The closing formula.** All nine sections end the body with a paragraph that begins "We can now" (I checked the sources: nine of nine, not the eight of nine that two reviewers estimated), followed by a Summary that repeats it. Bayesian Computation uses the phrase a second time in mid-section. `rv/RV-14`, `dist/DIST-16`, `expfam/EXP-18`, `stats/STA-20`, `mle/ML-17`, `bayes/BC-16`, `conc/CON-19`, `genb/GEN-20`, `nb/NB-S5`. In almost every case the second half of the paragraph is a good hand-over ("Two gaps remain", "Two questions remain open") and should be all that stays. The Summaries run to eight to ten sentences that list every subsection (`expfam/EXP-19`, `stats/STA-22`, `conc/CON-20`, `genb/GEN-22`).

**The opening formula.** Five files open on "Every model in this book was ..." or a close variant: the introduction, Statistics, Maximum Likelihood, Bayesian Computation ("Every model in the Basics part") and Concentration ("Every accuracy reported in this book"). `mle/ML-18`. Each is a good opening alone.

**Results told many times on the way to being shown.** Announced in the introduction, again in forward pointers, measured, recapped in the closing paragraph and in the Summary: six tellings in Naive Bayes (`nb/NB-S1`), four of the "Four facts" (`expfam/EXP-2`), six of the 66 heads (`genb/GEN-1`). This is far less than the Basics chapters had, and it sits in the same places: captions, openings, closings, summaries.

**Several subjects of equal weight, and routine identities set as propositions.** The Exponential Family has five subjects with no ranking (`expfam/S1`); Maximum Likelihood sets nine statements as "Proposition", three of them one line of algebra (`mle/S4`); Naive Bayes sets a one-line identity as a proposition with a nine-line proof (`nb/NB-15`); the nesting argument is one sentence in the main text and a Proposition with Proof here (`genb/GEN-15`). By my count the nine sections contain 50 statements headed Proposition, Lemma or Theorem. The reader cannot tell which are the theorems of the chapter.

**Verification cells that cannot fail.** Five of the 14 cells in Distributions check arithmetic or tautologies (`dist/S3`, `dist/DIST-7`); the bias and variance cell prints an identity of the empirical distribution (`stats/STA-6`); the Rademacher cell's product confirms an inequality that holds by logic (`genb/GEN-13`); the four-tab autodiff check of a two-line derivative prints two numbers of order $10^{-15}$ (`bayes/BC-4`); gradient descent on a function whose minimizer was derived two pages earlier (`mle/ML-4`). Against this, the chapter's good cells are the best in the book, and three section reviews independently name "predict, then run" as its method.

**Cell lead-ins.** "Does ...? The cell ..." introduces 11 cells in Distributions (`dist` chapter note), "The cell ..." begins 11 lead-ins in Bayesian Computation (`bayes/BC-9`), and Random Variables narrates one cell line by line (`rv/RV-6`). The question form is right when the answer could be no.

**Language tics, with the reviewers' counts.** "Thus" as the fixed opener of the gloss after a display: 14 in Maximum Likelihood (`mle/ML-1`), five in Statistics with five "So" (`stats/STA-21`), none in the other sections, so it marks particular writing passes. "Law" for "distribution": about 25 times in Distributions (`dist/DIST-1`), where it collides with "law of large numbers"; it also appears in Random Variables and Concentration. "Says nothing" and "certifies nothing": nine variations in Uniform Convergence (`genb/GEN-21`). "A reader may object ... The objection is correct": effective once each in Distributions and Uniform Convergence, used twice with the same words in Naive Bayes (`nb/NB-16`). "This part derives / This part estimates" (`nb/NB-3`). Hedges about runs in Bayesian Computation, nine quoted (`bayes/S2`).

**Framework boilerplate.** Every section is NumPy only, and six section reviews note an imports cell that loads `torch`, `nn`, `F` and their counterparts for nothing (`rv/RV-15`, `dist/DIST-17`, `expfam/EXP-3`, `conc/S6`, `genb/S6`, `nb/NB-18`). Four sections carry the sentence "The code in this section is plain NumPy, the same in every framework tab" (Statistics, Concentration, Uniform Convergence, Naive Bayes). One cell is fragmented by tabs to no purpose (`bayes/BC-4`).

**Exercises hold main-line content, and some repeat the text.** The sets are excellent and long: Naive Bayes has 2,400 words of exercises after 3,950 of body, with temperature scaling, the text model and the Ng and Jordan result only there (`nb/NB-S3`, `nb/NB-S2`); the positive norm-based result for networks is the last exercise of Uniform Convergence (`genb/S4`); the sandwich covariance is one extended exercise (`mle/ML-12`); the conclusion about a network's posterior modes is used in the text by reference to an exercise (`bayes`, exercise 9). In the other direction the text gives away exercise parts (`mle/ML-4`, `bayes/BC-6`, `genb/GEN-11`, and the lists under Exercises in `conc`, `nb` and `expfam`). Leftover labels: `ex_mdl-random-variables-using-textrmvar-x-y`, `ex_mdl-distributions-using-conditional-formula-eqref`, `ex_mdl-statistics-let-x-1-ldots`, and seven `ex_mdl-distributions-*` labels inside the Exponential Family. Five exercises in Statistics refer to cells by ID, which the rendered book may not show.

**Notation drift.**

- The coin's bias: $p$ in the first two sections, $\theta$ afterwards, with two remarks that disagree about when the change happens. The Exponential Family says "as in the rest of the chapter we write it $\theta$"; Statistics, one section later, says "Earlier sections wrote $p$ ... here ... we write $\theta$". `stats/STA-3`; confirmed in the sources.
- $T$: sufficient statistic and temperature in adjacent sentences (`expfam/EXP-12`), then the $z$-statistic and a generic test statistic (`stats/STA-11`).
- $B$: model B, the number of shuffles, the number of bootstrap resamples (`stats/STA-10`, `stats/S3`); later the radius of the weight ball. $k$ and $c$ each three things in Statistics (`stats/S3`).
- $\sigma^2$: true variance, variance proxy, true variance again within 40 lines (`conc/CON-10`).
- $X$ for the target and $Y$ for the input in "Conditional Expectation", against the rest of the book (`rv/RV-8`).
- $R(f)$, $\hat R(f)$, "half-width" and $d_{\mathrm{VC}}$ here against $\epsilon(f)$, $\epsilon_\mathcal{S}(f)$, "radius" and $v$ in the main text, with no mapping (`genb/S5`, `genb/GEN-8`).
- $p(x\mid\boldsymbol\theta)$ in the text and $p(x;\theta)$ in the ELBO figure (`mle/ML-15`).
- Terms at their definition are bold in Statistics and italic in the neighbouring sections (`stats/STA-21`). Propositions are named in Random Variables and unnamed in Distributions (`dist/DIST-14`).
- Two coins are both "our coin" in Distributions, the fair coin of the question and the plug-in coin with $p=9/13$ (`dist/DIST-3`).

**Lookup.** Distributions and Statistics have one subsection label each, so the multivariate Gaussian, Gaussian conditioning, the bias and variance decomposition, the $p$-value, power, the confidence interval, the delta method and the bootstrap cannot be cited except through a whole section; the Gamma distribution and the Poisson process have no heading (`dist/S4`, `stats/S6`). The Exponential Family reads well front to back and badly by lookup: "the catalogue", "the previous section" and "the chapter's coin" have no reference (`expfam/S5`, `conc/CON-8`).

## Connection to the main text

**Applications named without a label.** This is the chapter's most uniform defect; six section reviews report it independently.

- Introduction: cross-entropy, squared error, the error bar and model selection, no `:numref:` (`index/IDX-3`).
- Random Variables: ReLU outputs, gradient norms, minibatch noise, squared-loss regression as the conditional mean, Monte Carlo, with labels only on PCA and the continuous normalizing flow (`rv/S4`).
- Distributions: the four losses, the $\ell_1$ penalty, Gaussian process regression, "the default prior and latent distribution of generative models", and the sampler $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$, which is the reparameterization and is never called that (`dist/S6`, `dist/DIST-12`, `dist/DIST-15`).
- Statistics: "very flexible models that fit their training data exactly can still predict well" (`stats/STA-7`); Cramér–Rao named with no reference to the next section, which proves it (`stats/STA-8`).
- Exponential Family: "appear later in the book" for the Gamma–Poisson and Dirichlet–multinomial pairs, a promise the reviewer could not find kept anywhere (`expfam/EXP-17`).
- Maximum Likelihood: the uses of the Fisher matrix in chapters 40 and 42, which point back here, are not named (`mle/S5`).
- Bayesian Computation: variational autoencoders, the diffusion chapters, off-policy reinforcement learning and score-based models (`bayes/S3`).
- Further Reading: "the random-label experiment" (`fr/FR-7`).

**Stale or wrong in the main text.**

- `chapter_linear-classification/generalization-classification.md` line 478 says this chapter "proves the finite-class bound"; the main text has already proved it (`genb/GEN-2`). Confirmed.
- The main text's remark on why results are reported as ranges should point to Statistics by label (`stats`, chapter note).
- The main-text exercise `ex_softmax-regression-gumbel-max-trick` is solved in the text of the section it refers to (`expfam`, exercise 1).
- The confusion matrix has rows for the true class here and columns for the true class in `softmax-regression-scratch.md` (`nb/NB-12`). Confirmed.
- No section of the main text refers to Naive Bayes; the one planned reference, from the BERT section, wants word counts and would find pixels (`nb`, "Lookup use", `nb/NB-S2`).
- The JAX tab of Statistics carries the TensorFlow discussion link (`stats`, Exercises).

**Reviewers disagree with October in one place.** October asked for the initialization and nearest-neighbour material in Concentration to be cut. The reviewer keeps initialization, which has the best new idea of the second half (depth against width, $K\ll d$), and shortens nearest neighbours (`conc/S5`, `conc/CON-17`, `conc/CON-18`). I agree.

## What to cut

Roughly 6,000 words of body with no loss of argument, before the two splits:

- **Uniform Convergence**, about 1,600 words: the first part down to 500 words, the third to 1,000; nesting as a sentence; the inverse-Wishart derivation to its exercise (`genb/S2`, `genb/S3`, `genb/GEN-15`, `genb/GEN-11`).
- **Maximum Likelihood**, about 1,000 words besides the 1,900 that move with EM: the Gaussian NLL and Gaussian prior proofs, the float32 aside, the learning-rate paragraph, one of the two passages on parameterization, the closing paragraph (`mle/S6`).
- **Exponential Family**: the multi-option probit integral and the logit against probit paragraph to two sentences each; the Gamma–Poisson pair to its exercise; the normalizability paragraph; the expert remark on non-uniqueness (`expfam/S2`, `expfam/EXP-17`, `expfam/EXP-16`).
- **Distributions**: five cells that cannot surprise, about 300 words; the Cauchy paragraph and cell; "High dimension"; the opening's list of losses (`dist/S3`, `dist/DIST-11`, `dist/DIST-13`, `dist/DIST-2`).
- **Random Variables**: the re-motivations of chapter 1; the last paragraph of the covariance-matrix subsection; three of four statements that a density is a rate; the three-atom example moved and shortened (`rv/S2`, `rv/RV-12`, `rv/RV-2`, `rv/RV-3`).
- **Statistics**, about 250 words: the general two-sided $p$-value, the restated Markov inequality, the two lemmas behind the $t$ distribution, the vector delta method (`stats/S8`).
- **Concentration**: the repeated readings of the coin table, the second pair count, the expert remark after Bernstein, the many-models paragraph of the introduction (`conc/S4`, `conc/S5`, `conc/CON-11`, `conc/S3`).
- **Bayesian Computation**: little. The four-tab autodiff check, the second small Laplace case, the Geyer truncation paragraph (`bayes/S6`).
- **Naive Bayes**: body unchanged in length; exercises from twelve to about ten and from 2,400 to 1,600 words (`nb`, Exercises).
- **Everywhere**: the first half of each "We can now" paragraph, and Summaries cut to five sentences.

## What is missing

- The density of a sum of independent variables as a convolution: never stated in the chapter, required by an exercise of Distributions (`rv/S5`, `dist` exercise 5).
- A pointer to Jensen's inequality where $\mathbb{E}[g(X)]$ is introduced (`rv/S5`).
- One line under the summary table of Distributions saying where the Beta, Dirichlet, Student-$t$, log-normal, Cauchy, geometric and negative binomial live (`dist/S5`).
- The canonical link named, with the probit as the example that loses the simple gradient (`expfam/EXP-9`); one clause that $\nabla^2A$ is the Fisher information in natural parameters (`expfam/S6`, `expfam/EXP-11`).
- The names a reader will search for: McNemar's test, Welch's $t$-test, and a confidence interval for a difference (`stats/S9`).
- What survives of the Fisher theory for a network, and misspecification worked with one number (`mle/S5`, `mle/ML-12`); natural gradient is absent from the chapter and from Further Reading (`fr/FR-S2`).
- A decision among the four Bayesian methods, which the section label `sec_mdl-bayes-decision-map` promises (`bayes/S5`); a check that is actually shown for Laplace and for variational inference (`bayes/S4`).
- The step from a variance of the logit to a change of the prediction, $e^{\mu+s^2/2}$ (`bayes/BC-7`).
- The empirical-complexity form of the Rademacher bound (`genb/GEN-7`); the width-free bound for a two-layer ReLU network, stated in the text (`genb/S4`); what least squares on the raw inputs achieves in the sweep (`genb/GEN-17`).
- Hoeffding with unequal ranges stated in the proposition, since two later arguments need it (`conc/CON-7`).
- The multinomial and Bernoulli event models for text, and temperature scaling, in the body of Naive Bayes (`nb/NB-S2`, `nb/NB-S3`).
- In Further Reading: deep ensembles, proper scoring rules, Ng and Jordan (`fr/FR-S3`); and a name that distinguishes it from "Resources and Further Reading" in the introduction, which is a decision for the whole part (`fr/FR-S1`).
- Labels into the main text throughout (previous section), and subsection labels in Distributions and Statistics.

## What is best and should set the model

- **Random Variables, the dart.** One object, every definition computed on it, each computation a small surprise (the median at 0.707, the sampler $\sqrt U$, zero covariance with dependence, independence in polar coordinates). The density $2r$ is reached by four routes.
- **Distributions, one stream of events.** "The wait for the next request exceeds one minute exactly when the minute contains no request." Each distribution is built, so its moments come from the construction. And the bell from the binomial's successive ratio: "A function whose steps grow linearly is a parabola."
- **Exponential Family, hypotheses shown by breaking them.** Moment matching that returns a value under which an observation is impossible; the loss with no minimum at 13 heads of 13; and the sentences after two proofs that say which step did the work.
- **Statistics, the sampling distribution as the single organizing idea**, the $n=2$ case before the $n-1$ proposition, and simulations read with their own noise: "The count lands near 947, typically within $\pm14$, rather than near the nominal 950. The shortfall is systematic."
- **Maximum Likelihood, the Cauchy location model**: "A model whose MLE is not an average shows what the theory adds." And use before proof, said openly: "We take that on trust for a moment and apply it."
- **Bayesian Computation, predict, then run, against a target small enough to check exactly.** The paragraph on the Laplace proposal whose weights look healthy and have infinite variance is the best single paragraph the reviewers quote from the chapter.
- **Concentration, Markov to Hoeffding.** The majorant picture, the sentence "No polynomial majorant factorizes this way", and the fair coin worked to the end before the general proof.
- **Uniform Convergence, the framing and the slack analysis.** "A test-set certificate and a learner differ only in the order of two events: choosing the function and drawing the sample." Then a bound whose only lossy step is identified and measured.
- **Naive Bayes, the two-word spam filter that returns three times**, and "Training is counting, and prediction is multiplying."
- **Further Reading, entries that state a limit of the chapter, the next idea and its price**, with the Cantor digits and the red and blue bus as the standard for the entries that lack an example.
- **The exercises.** Three section reviews call their set the strongest the reviewer has seen in the book, and the others come close. They tie each result to something a deep learning reader does, and many produce a number the reader does not expect.

The common method is: one concrete object per section, a closed-form prediction before each cell, a failure shown with a number instead of a stated hypothesis, and a sentence of judgement after a proof. Where the chapter is weak it is because a section left that method (the catalogue rhythm in Distributions, the block of definitions in Concentration, the rerun experiment in Uniform Convergence), not because the method failed.

## Disagreements between section reviews

There is no flat contradiction of fact between section reviews. Four things do not line up and need a decision or a correction:

1. **The home of Gibbs' inequality.** `expfam/S3` proposes that the Exponential Family keep the lemma (or cite chapter 42 forward) and that chapter 42 cite the proposition here. `mle/S2` proposes that Maximum Likelihood keep a one-line tangent argument inline and that chapter 42 cite Maximum Likelihood for the tangent proof. Each reviewer writes as if their section were the only one in chapter 41 that proves it. Both cannot be adopted; my proposal is in the Repetition table.
2. **The count of "We can now" closings.** "Eight of nine sections" (`mle/ML-17`, `bayes/BC-16`), "three consecutive sections" (`genb/GEN-20`), "each section" (`index`, chapter note). The sources have it in all nine.
3. **When the chapter switches from $p$ to $\theta$.** The reviewer of Statistics reports the disagreement between two remarks in the text (`stats/STA-3`); the reviewer of the first two sections did not flag the symbol at all. The text needs one stated point of change.
4. **Superlatives at equal scores.** Four sections are each described as the best their reviewer read, and all receive 4 in every dimension. This is calibration and not disagreement, but an overview that ranks sections should use the verdict paragraphs and the partial scores of 5, not the table.

A fifth point is a difference of emphasis: the reviewer of Uniform Convergence says the chapter's arc ends there and naive Bayes "reads as an appendix to the chapter"; the reviewer of Naive Bayes recommends keeping it last as a capstone. Both ask for the same framing sentence, so there is nothing to resolve.

## Checked in the sources

I opened the chapter sources for the cross-section claims and confirmed: Gibbs' inequality is proved by the tangent line both at `mdl-exponential-family.md` line 373 and at `mdl-maximum-likelihood.md` line 361, with entropy and KL defined in both, and again in `chapter_mdl-information-theory/mdl-information-theory.md` (proposition at line 443, tangent-line exercise at line 1196); the two pseudo-count conventions; the two remarks on $p$ and $\theta$; the duplicated Cauchy sentence in Random Variables and Distributions; the union bound defined in both Concentration and Uniform Convergence; Markov's inequality restated in Concentration; the "We can now" closing in all nine sections and the "Every ..." opening in five files; the stale phrase in the main text; and the two orientations of the confusion matrix. The counts of "Thus" (13 in Maximum Likelihood by my grep, 14 by the reviewer) and of propositions (50) are mine.

Not checked by me: the reviewers' numerical recomputations and simulations (in particular the 1,500 simulated data sets behind `bayes/S1` and the reruns behind `genb/GEN-17` and `nb/NB-9`), the rendered pages, and the TensorFlow, JAX and MXNet outputs, which no reviewer read.
