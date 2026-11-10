# Writing pass: Probability and Statistics (`chapter_preliminaries/probability.md`)

Prose only. Checker: "OK: only prose changed"; `tools/lint_source.py` clean.

## Words

Prose words (checker count, all tabs): 5,897 before, 6,089 after (+192, about
3%). The section got longer, against the brief. The additions are the ones the
assignment asked for: guiding sentences that keep the coin question alive
(about 150 words), the p-value and the two-standard-error rule (about 110),
and a plain explanation of the divisor $n - 1$ (about 30 net). Cuts (the
weighted-sum formula of the covariance matrix, the tidy half of "Random
Variables", the classifier analogy, the closing recap, the JAX tab note,
several restatements) removed about 200 words, which did not cover them. One
display equation was removed (27 to 26 delimiters).

## Before and after

1. Opening roadmap.
   - Before: "We simulate coin tosses to see the noise, then state the rules for computing with probabilities and use them to interpret a medical test. Finally, we measure how much an average of random outcomes fluctuates, which settles the question about the coin."
   - After: "The coin question is easy to ask, and it will take us the whole section to answer. We first simulate coin tosses to see the noise. We then state the rules for computing with several random variables at once and use them to interpret a medical test. The coin needs only one notion from this detour, the independence of its tosses, but the rest of the book needs all of it. Only then do we measure how much an average of random outcomes fluctuates, which settles the question."
2. End of "Sampling".
   - Before: "The law does not say how fast the frequency settles, which is what the coin question needs [...] To answer it, we need a precise language for random outcomes and their averages."
   - After: "Unfortunately, the law does not say how fast the frequency settles, and that is what the coin question needs [...] We cannot answer this yet. We first need a precise language for random outcomes, and then a way to measure how much their averages fluctuate."
3. Start of "Conditional Probability".
   - Before: "Most questions in machine learning involve several random variables at once: the pixels of an image and its label, or the symptoms of a patient and the disease."
   - After: "The coin question now has to wait. Each toss is one random variable, and to say how a hundred tosses combine we must be able to speak of several variables at once. So must most of machine learning, whose questions involve the pixels of an image and its label, or the symptoms of a patient and the disease."
4. Bayes' theorem.
   - Before: "Bayes' theorem reverses the direction of conditioning."
   - After: "For a result that takes one line of algebra, Bayes' theorem is remarkably useful: it reverses the direction of conditioning."
5. After the HIV simulation.
   - Before: "We will compute how precise the simulation is once we can measure the fluctuations of a frequency."
   - After: "How precise should it be? We cannot say yet, for the same reason that we cannot yet judge the coin: we lack a measure of how much a frequency fluctuates."
6. Unbiasedness.
   - Before: "Unbiasedness says nothing about how far a single estimate strays from $p$, which is what the coin question asks. For that we need a second summary."
   - After: "This is reassuring, but it does not answer the coin question. Unbiasedness says nothing about how far a single estimate strays from $p$. For that we need a second summary."
7. Variances add.
   - Before: "Variances of independent variables add, while standard deviations do not."
   - After: "This is the key step toward the coin. Variances of independent variables add, while standard deviations do not."
8. Closing.
   - Before: "We can now compute with probabilities, reverse a conditional with Bayes' theorem, and attach a standard error to an average, which separates a stable pattern from sampling noise. Throughout this section the probabilities were given: a fair coin, or a test with known error rates. In learning they are unknown, and a model with many parameters must estimate them from data."
   - After: "Note that throughout this section the probabilities were handed to us: a fair coin, a test with known error rates. In learning nobody hands them to us, and a model with many parameters must estimate them from data."

## Repetition removed

- Closing recap ("We can now compute with probabilities, reverse a conditional ...") deleted; the Summary is the only recap.
- "The overlap is subtracted once because the sum counted it twice" (said in the sentence before the formula).
- "Conditioning restricts attention to the outcomes in which $A = a$ and renormalizes" (said before the definition; kept in the Summary).
- "For a density, the sum becomes an integral" under Expectation (the general statement is in "Discrete and Continuous Variables").
- "More data reduces the epistemic part" told twice in one paragraph; now once.
- "Like the probabilities, the frequencies of all outcomes sum to 1."
- JAX tab note: keys are owned by `sec_ndarray`; reduced from five sentences to two with the reference.
- "We can now ..." removed from the opening of "Estimation Error" as well as the closing (chapter-wide template).

## Review findings

Applied:

- Structural (two lessons interleaved), option 3 of the review: guiding sentences in the opening, at the end of "Sampling", at the start of "Conditional Probability", under "Independence" ("This is the notion that the coin question needs"), after the HIV simulation, at the end of "Expectation", at "variances add", and at the start of "Testing for Bias" ("At last we can return ...").
- Structural (example delayed after Bayes): one sentence at the end of "Bayes' Theorem" says that the test follows after one more notion.
- Structural (rule of thumb missing) and PRB-11: new paragraph "How many standard errors are too many?" with Chebyshev's crude 25% at $k = 2$, the normal approximation with references, and the rule (under two unremarkable, three or more hard to blame on chance). The claim about two models now gives the standard error of the difference (0.004) and points to the exercises.
- Structural (test never named): "This fraction estimates what statisticians call a *p-value*".
- PRB-3: "$P(X \in A)$ for every set", "countably many", and the arrow notation cut; the distribution is "the table of the probabilities $P(X = v)$".
- PRB-5: "Summing each row".
- PRB-7: the chain gets its own sentence and an example (yesterday's, today's, tomorrow's weather); "In the same way" removed.
- PRB-8: classifier analogy cut.
- PRB-9: the $n - 1$ explanation is its own paragraph after the investment numbers.
- PRB-10, in part: the variance of $\mathbf{v}^\top\mathbf{x}$ and its display are cut; the definition of the covariance matrix stays, flagged as not needed by the coin, because chapter 3 and exercise 8 use the term.

Deferred:

- Splitting or reordering the section; moving the first test before "Independence": moves a code cell and a labelled subsection.
- PRB-1 (unused imports): code.
- PRB-2 (stored sample is [50, 50]): re-capture on the development machine. Prose untouched and does not depend on the sample.
- PRB-4 (notation $A$, $B$ against $X$, $Y$; $p$ against $p(x)$): the figures and slides use $A$, $B$, $a_i$, $b_j$; needs one decision and regenerated figures.
- PRB-6 (concrete joint table): needs a new table or figure.
- PRB-10, remainder (move the covariance matrix to the appendix), and cutting explaining away or correlation: exercises 5 and 8 and the three-panel figure depend on them, and exercises are out of scope.
- PRB-12: no change needed (all four stored outputs, 0.27 to 0.29, agree with "roughly one in four").
- PRB-13 (move "Aleatoric and Epistemic Uncertainty" before "Testing for Bias"): not done. It would put one more subsection between the standard error and the answer to the coin question, which is the delay this pass tries to shorten. Worth a decision by the author.
- Cauchy-Schwarz as the reason for $|\rho| \leq 1$: tried, then dropped for length.

## Factual claims added or changed

- $2^{100}$ is "a number with 31 digits" ($1.27 \times 10^{30}$).
- The coin needs only independence from the conditional-probability part.
- The $n - 1$ divisor: numbers lie closer to their own average than to $\mu$, so dividing by $n$ underestimates the variance on average.
- The derivation of the standard error used the variance of one toss *and the independence of the tosses* (the old text named only the variance).
- The simulated fraction "estimates" a p-value, defined as the probability, if the coin is fair, of a result at least as extreme as the one observed.
- "none of the simulated experiments comes close to it" became "none of the 1000 simulated experiments lands that far out", which is what the output (0 in all four frameworks) shows.
- Chebyshev at $k = 2$ gives 25%. An average of many independent draws is approximately normal (central limit theorem, `sec_mdl-distributions`; normal distribution at `subsec_normal_distribution_and_squared_loss`), and a normal variable lies within two standard deviations about 95% of the time. Stated without proof.
- The difference of two independently measured accuracies on 10,000 images has standard error about 0.004 ($\sqrt{2} \cdot 0.003$), so one point is more than two standard errors (about 2.4).
- Removed: "for an average of many tosses the true probability is far smaller still" (subsumed by the new paragraph); the claim that $\boldsymbol{\Sigma}$ gives the variance of every weighted sum.
- Summary: "Deviations of one or two standard errors are what chance routinely produces" became "A deviation of one standard error is what chance routinely produces, more than two are unusual".

## Slides against the body

No slide now contradicts the body. Three omissions or leftovers:

- The slides do not name the p-value or state the two-standard-error rule; the slide "56 of 100 Is No Evidence of Bias" still rests on Chebyshev alone.
- The slide "A Minibatch of 256 Images ..." gives 0.003 for the test set but not the 0.004 of a difference.
- The slide "A Random Variable Maps Outcomes to Values" and the body agree; the Recap slide's "no evidence of bias" agrees.
