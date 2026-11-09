# Probability and Statistics

Source: `chapter_preliminaries/probability.md`. About 5,650 words of prose in the PyTorch view (7,900 in the file to the slides marker), plus 1,050 words of exercises. The longest section of the chapter, above the writing guide's 4,500-word ceiling.

## Verdict

A good section built on a strong device: a question is asked in the second paragraph ("a coin is tossed 100 times and produces 56 heads. Is the coin biased?"), kept alive through the section ("which is what the coin question needs"), and answered near the end with a quantity the reader has derived (56 heads is 1.2 standard errors from 50; 5,600 in 10,000 is 12). The same standard error is then turned on the reader's own world (accuracy on a minibatch of 256 has standard error 0.019) and, in a nice touch, on the section's own simulation. All numbers check, including against the stored outputs. The section is a 4, not a 5, because it is two sections in one. The path from the coin to the standard error is interrupted for about 1,800 words by conditional probability, Bayes' theorem, conditional independence, explaining away and the diagnostic test, which is a second complete lesson with its own example. Both lessons are good. Together they make the longest unit in the chapter, and the reader who wants the answer to the coin question has to carry it across the HIV example. The most important fix is to separate or reorder the two, and to prune what neither needs (the covariance matrix, explaining away, the general notion of a distribution in the dense "Random Variables" paragraph).

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening connects probability to something the reader will do next week: "A classifier that labels 230 of a minibatch of 256 images correctly has an accuracy of 0.90 on that minibatch, and on the next minibatch the same model may score 0.87." Then the coin, as the simplest case of the same question.
- "*Probability* answers the forward question ... *Statistics* answers the inverse question". Two sentences that organize the field.
- "Had we not written the simulation ourselves, a single run could not tell us whether the coin is slightly unfair". This is the moment the reader understands why statistics is needed.
- The HIV example: the table, the calculation, the surprise ("only 13%, even though the test never misses an infection"), and then the figure in counts (15 true positives against about 100 false ones among 10,000 people). The count picture is what makes the result stick. The second, weaker test raising the posterior to 83% and the remark "it still helps, because its errors are independent of those of the first test" complete it.
- The simulation of two million patients checks the formula with tools from the first section (`where`, a boolean mask), and the text is honest that the agreement is only to about two points, promising to explain why. It does, 500 lines later: "only about ... 3,500 test positive twice ... Its standard error is ... 0.006 ... One more correct digit would take a hundred times as many patients." A promise made and kept across the section.
- Variance is motivated by showing that the obvious measure fails: "we cannot average the deviation $X - \mu$ itself, since by linearity ... positive and negative deviations cancel."
- "Variances of independent variables add, while standard deviations do not." And the counterexample for the converse ($X$ uniform on $\{-1, 0, 1\}$, $Y = X^2$) in three lines.
- "Testing for Bias" is careful about what the evidence does not show: "Nor are they evidence that the coin is fair: 56 heads are just as consistent with $p = 0.55$. They do rule out $p = 0.3$". Most introductions get this wrong.
- The simulation approach to the test (count how many fair-coin experiments land as far out as 56) avoids the normal distribution entirely and is something the reader can run.

## Structural findings

- **Two lessons in one section.** Lesson A: frequencies fluctuate; expectation, variance, the $\sigma/\sqrt{n}$ law, and how to judge a difference. Lesson B: joint, marginal and conditional distributions, Bayes' theorem, conditional independence, the diagnostic test. The section opens A, runs B in the middle, and closes A. Options, in order of preference: (1) make them two sections, "Probability" (events, random variables, conditioning, Bayes, the test) and "Statistics" or "Estimation Error" (sampling, expectation, variance, standard error, the coin), each of about 2,800 words; (2) keep one section but put B first, so that the coin question is asked immediately before the material that answers it; (3) keep the order and say at the start of "Conditional Probability" that the coin question is parked until "Expectation and Variance".
- **Inside lesson B, the order delays the example.** Bayes' theorem is stated, then "Independence" follows with common causes, chains and explaining away, and only then comes "Example: Diagnostic Testing". The reader holds an unapplied formula across 350 words and a three-panel figure. Put the first test directly after Bayes' theorem. Introduce conditional independence where the second test needs it.
- **Material neither lesson needs.** Explaining away (the hospital example, the third panel) is charming and is used only by one exercise. The covariance matrix and $\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v}$ are used only by the portfolio exercise. Correlation is defined and not used. The discrete-continuous subsection is needed only for the sentence that sums become integrals. Cutting the first two and compressing the others would save about 500 words and remove the stretch where the section drifts ("Sums of Independent Variables" is half about dependent ones).
- **Missing: the one rule of thumb the section argues toward.** The text says deviations of "one or two standard errors are what chance routinely produces" and falls back on Chebyshev for the rest. It never states that an average of many independent draws is approximately normal, so that about 95% of the time it lies within two standard errors. That sentence (with a pointer to the appendix for the central limit theorem) is what a practitioner uses, and it is what makes "a difference of one percentage point between two models is unlikely to be noise" a justified claim. The normal distribution is introduced in chapter 2; a one-sentence preview here would connect the two.
- **The test is never named.** "Testing for Bias" computes a p-value and never uses the word. Following the guide's "idea first, name second" the name should arrive after the simulation: "Statisticians call this fraction a *p-value*." The reader will meet the term everywhere else.

## Passage-level findings

**PRB-1** (minor, pedagogy). After the imports.

> from torch.distributions.multinomial import Multinomial

As in the calculus section, the first imports cell loads `math`, `nn`, `F` (and `gluon`, `optax`, `nnx` in other tabs) that the section never uses, followed by a second imports cell. Import what is used.

**PRB-2** (minor, correctness of presentation). Under "Simulated Coin Tosses", stored output of the multinomial cell.

> Running the cell again gives different counts, and they are usually not 50 and 50.

The stored PyTorch output of the very next cell is `tensor([50., 50.])`, and the opening has just said that "Even a fair coin rarely produces exactly 50 heads in 100 tosses". The reader of the rendered page sees the rare event presented as the typical sample. The prose is right and the capture is unlucky (probability about 8%). Recapture, or fix a seed for this one cell so that the page shows an uneven count.

**PRB-3** (moderate, explanation). Under "Random Variables".

> In general, the *distribution* of $X$ assigns the probability $P(X \in A)$ to every set $A$ of values. For a variable with finitely or countably many values, the map $v \mapsto P(X = v)$, its *probability mass function*, determines the distribution, and we write it as $P(X)$.

The paragraph this ends introduces random variable, value, the event $X = v$, ranges, distribution, probability mass function and the notation $P(X)$: seven items in about 150 words. The reader needs three (a random variable is a number computed from the outcome; $P(X = v)$; the table of these is its distribution, written $P(X)$). "Countably many", "$P(X \in A)$ for every set $A$" and the arrow notation are measure-theoretic tidiness that the section never uses again. Cut them.

**PRB-4** (moderate, language). Notation across "Events and Random Variables" and "Conditional Probability".

> The *joint distribution* of two random variables $A$ and $B$ assigns a probability to every combination of values

Within one page, $\mathcal{A}$ and $\mathcal{B}$ are events, $A$ is "every set $A$ of values", and then $A$ and $B$ are random variables, while $X$ and $Y$ (the names used for random variables two paragraphs earlier and again from "Expectation" on) disappear for the whole of "Conditional Probability". A reader who has just learned that an event is not a random variable now sees $P(A, B)$ and cannot tell which is meant. Likewise $p$ is the probability of heads and $p(x)$ a density, which the text has to disclaim: "(unrelated to the probability $p$ of heads)". The guide treats such a disclaimer as a sign that a symbol should be renamed. Use $X$, $Y$, $Z$ for random variables throughout (the joint table becomes $P(X, Y)$), keep $\mathcal{A}$, $\mathcal{B}$ for events, and write the density as $f(x)$ or the coin's probability as $\theta$. $H$ and $D$ for hypothesis and data, reused as HIV status and test result, are a happy coincidence and should stay.

**PRB-5** (minor, correctness). Under "Joint and Marginal Distributions".

> Summing a row of the table gives the marginal distribution of $A$, and summing a column gives that of $B$.

Summing one row gives one number, $P(A = a)$, as the next sentence shows ("$P(A = a_2) = 0.20 + 0.15 + 0.25 = 0.60$"). Write "Summing each row gives the marginal distribution of $A$".

**PRB-6** (minor, pedagogy). Under "Joint and Marginal Distributions".

> :numref:`fig_prob_joint` shows the joint distribution of a variable $A$ with two values and a variable $B$ with three.

The only concrete joint distribution in the section lives in a figure, with abstract labels $a_1, a_2, b_1, b_2, b_3$, although the paragraph before has just offered a real pair ("the symptoms of a patient and the disease"). A table in the text with meaningful rows and columns (even weather and umbrella) would make marginalizing and conditioning operations on something, in the way the HIV table later does.

**PRB-7** (minor, thought). Under "Independence".

> In the same way, in a chain where $A$ affects $C$ only through $B$, knowing $B$ makes $A$ irrelevant for $C$.

"In the same way" follows the common-cause example (shoe size and reading level), and a chain is not the same way. The chain gets one abstract sentence and no example, between two cases that each get a vivid one. Give it an example (yesterday's weather, today's, tomorrow's) or drop it; the exercise "Factorizing a chain" then needs a sentence of setup.

**PRB-8** (minor, thought). Under "Example: Diagnostic Testing".

> The diagnosis works like a classifier that becomes more confident as it receives more input features, here the test results.

A loose analogy at the end of a precise paragraph. Classifiers do not in general become more confident with more features, and nothing later uses the comparison. Cut.

**PRB-9** (minor, explanation). Under "Variance".

> Estimated from $n$ numbers, as for the columns in :numref:`sec_pandas`, the variance becomes a sum of squared deviations from their average, divided by $n$ or, to keep the estimate unbiased when the average stands in for $\mu$, by $n - 1$.

This is the promised explanation of the $n - 1$ from the preprocessing section, and it is squeezed between the definition of the standard deviation and the investment example, in a sentence with three nested qualifications. It deserves two plain sentences of its own, after the investment numbers.

**PRB-10** (moderate, thought). Under "Sums of Independent Variables".

> For a random vector $\mathbf{x}$ with mean $\boldsymbol{\mu}$, the variances and covariances of all coordinates form the *covariance matrix*

The subsection is on its way to one result, that variances of independent variables add, which the next subsection needs at once. After reaching it, the text adds the failed converse (good, short), and then the covariance matrix and the variance of $\mathbf{v}^\top\mathbf{x}$, with "the return of a portfolio of stocks or the output of a linear model for a random input" as uses. Nothing in the chapter uses it, the notation switches from $X$ to bold $\mathbf{x}$ for a random vector (which until now meant a data vector), and it stands between the reader and the standard error. Move it to the appendix or to the section that first needs a covariance matrix; keep the portfolio exercise with a one-line definition. Correlation ("always lies between $-1$ and $1$") can stay as one sentence if it says why: this is the Cauchy-Schwarz inequality of :numref:`sec_linear-algebra` again, a connection the chapter could make and does not.

**PRB-11** (minor, correctness). Under "Testing for Bias".

> On a test set of $10{,}000$ images, the standard error drops to $0.003$, and a difference of one percentage point between two models is unlikely to be noise.

The standard error of a difference of two independent accuracies is $\sqrt{2}$ times larger, about 0.004, so one point is about 2.4 standard errors; and since the text has not said how often two standard errors are exceeded, "unlikely" is not yet supported (see the structural note on the normal approximation). The claim is roughly right. Either add "(the difference of two such accuracies has standard error about 0.004)" or point to the exercise "Comparing two classifiers", which treats exactly this and shows that the paired comparison is sharper still.

**PRB-12** (minor, language). Under "Testing for Bias".

> A fair coin produces deviations of about one standard error all the time.

Good plain sentence. The stored output of the cell that follows is 0.29 for 100 tosses, and the text says "Roughly one in four"; the exact value is 0.27. Fine at the stated precision.

**PRB-13** (minor, pedagogy). Under "Aleatoric and Epistemic Uncertainty".

> The two kinds of uncertainty have names

A short, clear subsection, and the distinction is useful. It arrives after the section's climax and before the closing paragraph, so it reads as an appendix. It would sit more naturally directly after "The Standard Error of a Frequency" (the standard error shrinks; the uncertainty about the next toss does not), leaving "Testing for Bias" as the end of the section.

## Exercises

Nine exercises and the strongest, most varied set in the chapter. "Comparing two classifiers" walks the reader from the naive standard error of a difference to the paired comparison and its explanation through covariance; it is a small lesson in experimental practice. "A second, correlated test" extends the HIV example to tests whose errors are dependent, which is the natural objection to the text's assumption. "A defense lawyer's statistics" is the kind of memorable problem that MacKay would set. "Conditioning on a common effect" makes explaining away computable with two coins.

- If explaining away and the covariance matrix leave the main text, exercises 5 and 8 need one sentence of setup each. Both are worth keeping.
- Exercise 8, part 3 (formulate the Markowitz problem) points to the appendix for the method. It is an open-ended modelling task in a chapter of otherwise checkable problems; mark it as optional.
- Exercise 2 uses Chebyshev's inequality, which the text states in one sentence. Sufficient.

## Still open from October

- "Variance is used about 750 lines before it is defined": fixed. Variance is now defined before the standard error, and the earlier use in the preprocessing section is glossed there.
- "Cut: Markov/Chebyshev (to appendix), the utility and loss-aversion aside, etymology, measurable sets": Markov, utility, etymology and measurable sets are gone. Chebyshev remains as one stated sentence with the proof in the appendix, which is the right amount. A residue of "measurable sets" tidiness remains in "Random Variables" (PRB-3).
- "Keep: unequal counts from a fair coin, the HIV example": both kept and improved by the count figure.
