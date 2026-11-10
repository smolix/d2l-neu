# Random Variables

Source: `chapter_mdl-probability-statistics/mdl-random-variables.md`. About 9,400 words of prose and mathematics in the body and 2,300 in the exercises (11,700 in all, up to the slides marker); 5 code cells; 9 pre-generated figures.

## Verdict

This is the best-taught section I would expect to find in the part, and for long stretches it meets the bar. One example, a dart thrown carelessly at a round board, carries the whole section: its distance has density $2r$, c.d.f. $r^2$, sampler $\sqrt U$, mean $2/3$ and median $0.707$; its coordinates have a flat joint density, a semicircle marginal, a conditional that is uniform on a chord, zero covariance and yet dependence; and in polar coordinates the same dart becomes two independent variables. MacKay teaches this way: a single object, interrogated until every definition has been seen at work on it. The numbers are right (I recomputed all of them), the proofs are short and each is the short one, and the code cells ask questions whose answers could have come out otherwise. What keeps it from a 5 is size and seams. It is two sections in one file (one-variable theory, then several variables and transformations), it re-teaches a fair part of `sec_prob` at the same level before going deeper, a handful of facts are stated four or five times, and the logical status of the density (definition by a limit, or by an integral) is muddled at the one place where a mathematical reader will look hardest. The most important fix is the split at "Several Variables", with the repetition removed on the way.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

Explanation and pedagogy are at 5 in the first movement (Densities through Heavy Tails) and at 3 to 4 in the second, which is where the length shows.

## What works

- **The opening puzzle.** "With all the digits the probability is zero, and yet the dart landed somewhere." The reader feels the problem before any machinery, and the table of three bins with the constant ratio 1 turns the puzzle into the definition. The resolution, "Probability zero does not mean impossible", is one sentence. Protect the whole of this passage.
- **The dart is used, not decorated.** Every new object is computed for it, and the computations surprise: the median dart lands at $0.707$ "because three quarters of the board lies beyond $0.5$"; the naive sampler $R=U$ "puts half of the darts within $r<0.5$, where only a quarter belong"; the best predictor of $X$ from $Y$ is the constant $0$ "although $X$ and $Y$ are dependent"; "in the right coordinates the same dart is described by two independent variables". The density $2r$ is reached by four routes (bins, $F'$, the sampler, the polar Jacobian), and the text points this out ("for the third time in this section, now from a Jacobian"; by my count the fourth). This is the section's main asset.
- **"A density is a rate, like a speed."** The analogy is exact and stays true when the figure of speech is removed.
- **Chebyshev is shown to be sharp, then shown to be loose.** The three-atom example attains the bound for every $\alpha\ge1$ with $q=1/(2\alpha^2)$, and the two-row table against the Gaussian follows at once. The closing remark, "The worst case puts all of its tail probability at a single distance", explains why both are true. Few textbooks do both in a page.
- **The Cauchy mean.** The lopsided range $[-L,2L]$ with limit $\log 2/\pi\approx0.221$ makes "the definition requires absolute convergence" concrete: the reader sees a second answer, not a warning.
- **The log-normal as a failed guess.** "The naive guess $p_X(\log y)$ peaks at $y=1$, and it is not even a density", with total area $e^{1/2}\approx1.65$, and the plot shows the excess. The change-of-variables factor is motivated by an error the reader would make.
- **Code that tests.** Four of the five cells put a prediction at risk: the ratio in shrinking bins, $\sqrt U$ against thrown darts against the naive $U$, the within and between variances, the histogram against two candidate densities. The remark that Eve's law "holds for the sample itself" (so the sum agrees to the last digit, in every run) is the kind of observation that shows an author looked at the output.
- **Proofs.** Markov by the indicator inequality, the correlation bound by the discriminant of $\textrm{Var}(tX+Y)$ followed by the remark that "the correlation is a cosine", and the best-predictor identity by splitting the error are all the shortest available arguments, and each is preceded by the reason the statement matters.
- **Exercises** are the best feature after the dart (see below).

## Structural findings

1. **The file is two sections.** Up to the end of "Heavy Tails" the section develops one idea, the density of a single variable and what can be said from its first two moments (about 4,500 words). From "Several Variables" on it develops another: joint, marginal and conditional densities, conditional expectation, covariance, and then the change of variables (about 4,700 words). The reader who arrives by lookup from a normalizing-flow reference lands on `subsec_mdl-change-of-variables` 1,250 lines into a file called "Random Variables". Split at `## Several Variables` into "Random Variables" and "Joint Distributions and Transformations" (or keep the one-dimensional change of variables with the first half, next to inverse-transform sampling of which it is the general form, and put the multivariate rule in the second). The dart carries across the split without change. Each half then has the length the guide allows for the appendix and one idea.

2. **Overlap with `sec_prob`.** The Basics section already teaches, for discrete variables: the mean as balance point, linearity (and points here "for the details"), the variance and its computational form, the affine rule, the standard deviation and its units, the variance of a sum with covariance, correlation, the covariance matrix, and "zero covariance does not imply independence" with the example $Y=X^2$. This section proves these and extends them to densities, which is the right job for the appendix. But it also re-motivates them from zero ("The mean alone is not enough. A profit of ...", "The variance has a problem with units. If $X$ is a profit in dollars ...", the cents-and-dollars paragraph, the $X=Y^2$ example with a slightly different support). A reader who has read chapter 1 reads the same motivation twice; a reader who has not is helped by it. The guide's rule for the appendix ("a first textbook for readers who do not [know the topic]") permits the restatement, so this is a judgement call for the author. My recommendation: keep every proposition and proof, cut each re-motivation to one clause, and use the line the text already uses for linearity as the model: ":numref:`sec_prob` used it; here we prove it."

3. **What is new relative to chapter 1 should be said once, up front.** The opening says densities are "the one new idea of this section". That is true of the first half only. Conditional expectation, the law of total variance, the best-predictor theorem, Markov and Chebyshev with their sharpness, heavy tails, and the change of variables are all new and are what a reader comes for. A sentence in the opening that names them as the new material (and names the rest as chapter 1 with proofs) would let both kinds of reader navigate.

4. **Connection to deep learning is named but not referenced.** The body names ReLU outputs, gradient norms, minibatch gradient noise, regression with squared loss, Monte Carlo estimates, PCA and normalizing flows. Only PCA and the continuous normalizing flow carry a label, and both labels point into the appendix. "A network trained with the squared loss is estimating $\mathbb{E}[\textrm{target}\mid\textrm{input}]$" is one of the most useful sentences in the section for a deep learning reader and should point to the regression section of the main text by label. The exercises do better (He initialization and softmax regression are referenced).

5. **Missing.** (a) The density of a sum of independent variables (convolution) is never stated in the chapter, although the next section's exercise on the central limit theorem asks the reader to compute it with `np.convolve`. Two sentences after the independence proposition would do. (b) Jensen's inequality lives in the optimization and information-theory chapters; a one-line pointer where the mean of a function is introduced ($\mathbb{E}[g(X)]$ versus $g(\mathbb{E}[X])$) would help the lookup reader. (c) The Gaussian is used several times before the next section defines it (see RV-5).

6. **The covariance-matrix subsection is a stub.** It defines $\boldsymbol\Sigma$, proves positive semidefiniteness in three lines (good), and then compresses PCA, the multivariate Gaussian and elliptical contours into one closing paragraph that the next section and the linear-algebra chapter each teach properly. Keep the definition and the proof; replace the last paragraph by one sentence with the two references.

## Passage-level findings

**RV-1** (moderate, correctness, clarity of thought; confident). Under "From Bins to a Density" and "The Cumulative Distribution Function".
Anchor: `when the probability per unit length near $x$ settles to $p(x)$ as the`
The Definition makes $p(x)$ the limit of $P(x\le X<x+\epsilon)/\epsilon$, which is the right derivative of $F$ at $x$. "Probability as Area" then obtains $P(X\in(a,b])=\int_a^b p$ by a Riemann-sum argument that silently needs the error "that shrinks faster than $\epsilon$" to do so uniformly in $x$ (there are $1/\epsilon$ pieces). Two pages later the Proposition "density is the derivative of the c.d.f." assumes "$X$ has a density $p$" in the integral sense (it cites `eq_mdl-cdf_def`) and proves $F'=p$, which under the stated Definition is nearly the definition itself. So the section has two definitions of "has a density" and a proposition whose content depends on which one is meant. No formula is wrong, but the logic is circular as written, and this is the place a mathematically inclined reader checks. Fix: make the integral property the definition ("$X$ has density $p$ if $p\ge0$ and $P(a<X\le b)=\int_a^b p$ for all $a<b$"), present the limit of the ratio as what the definition means at a point of continuity (which is then exactly the Proposition), and say in one clause that the Riemann-sum passage is the motivation for the definition and not a derivation of it. The dart computation and the table stay as they are.

**RV-2** (minor, language, repetition). "From Bins to a Density", the caption of the first figure, the opening of "The Cumulative Distribution Function", and after the proof of $F'=p$.
Anchor: `A density is not a probability. The dart's density at $r=0.9$ is $1.8$, and a`
The fact that a density is a rate and not a probability is stated four times within 140 lines: here (with the speed analogy, which is the good version), in the caption ("The density itself is not a probability (it may exceed $1$)"), as "The values of a density are rates, not probabilities", and as "The density is the rate at which probability accumulates, and the c.d.f. is the accumulated amount". Keep the first. The c.d.f. subsection can open directly with "Often we want a function whose values are probabilities".

**RV-3** (minor, pedagogy). "The Mean".
Anchor: `A discrete example will show us how sharp the tail bounds below are. Let $X$`
The three-atom example is introduced 130 lines before it is used, between the dart's mean and the linearity proposition, where it interrupts. Its mean, variance and standard deviation are then computed in three separate places. The offset $a$ plays no part. Introduce the example where Chebyshev needs it, with $a=0$, and compute $\mu=0$, $\sigma^2=8q$ there in two lines.

**RV-4** (minor, language). "Markov's and Chebyshev's Inequalities".
Anchor: `In a picture: on $[0,\infty)$ the step function that is $0$ below $t$ and $t$`
"In a picture" announces a figure that does not exist. The sentence is a good verbal picture; either add a small generated figure (a step under the line $y=x$, which would be the cheapest and most useful missing figure in the section) or begin "Geometrically, ...".

**RV-5** (minor, explanation). "Markov's and Chebyshev's Inequalities" and later.
Anchor: `much less probability beyond two and three standard deviations than the bound`
The Gaussian is used here, in "Unlike a Gaussian, whose tails fall off like $e^{-x^2/2}$", in the running-mean cell, in the mixture $\mathcal N(3,1)$, in the multivariate remark, and in the log-normal example, before :numref:`sec_mdl-distributions` defines it. The reader of this appendix has met the bell curve, so this is not a gap in understanding, but the density formula appears only in the log-normal example near the end. Give the formula and a forward reference to `subsec_mdl-gaussian-clt` at the first use (the Chebyshev table).

**RV-6** (moderate, pedagogy, repetition across sections). "Heavy Tails".
Anchor: `Gaussian draws, on a logarithmic $n$-axis. Its last line repeats the average of`
Two things. First, the paragraph narrates the cell line by line ("The next cell plots ... Its last line repeats ... and reports"); state the question instead ("Does averaging 1,000 Cauchy draws bring the average closer to 0 than a single draw?") and let the cell answer. Second, the same experiment, the same probability $\tfrac12$, and the same sentence ("the average of $n$ independent Cauchy draws is again a standard Cauchy variable, a fact we state without proof") are repeated in the next section under "The Cauchy counterexample" with its own cell. One of the two must go. This section is the natural home, since it defines the Cauchy; the next section should cite it in a clause. (Changing cells is deferred under the author's decision 3; the prose can be fixed now.)

**RV-7** (minor, correctness; confident). "Heavy Tails".
Anchor: `$P(|X|>t)$ decaying like $t^{-\alpha}$, are the extreme case: they make every`
Power laws are not the extreme case of heavy tails (a tail like $1/\log t$ is heavier than every power law and has no moments at all). Write "the standard example".

**RV-8** (moderate, explanation). "Conditional Expectation".
Anchor: `A regression model predicts a target $X$ from an input $Y$. Training with the`
Everywhere else in the book the input is $x$ and the target is $y$, and `sec_prob` writes $\mathbb{E}[Y\mid X=x]$. Here the target is $X$ and the input $Y$, because the subsection inherits $p_{X\mid Y}$ from the one before. A reader arriving by lookup for "the conditional mean is the best predictor" reads every formula against the grain, and exercise 10 repeats the inversion. Either swap the roles in this subsection (and condition $Y$ on $X$ from "Conditional Densities" on, which costs a relabelling of the dart's chord), or keep the letters and say so in one clause at the top: "(the letters are those of the previous subsection: here $Y$ is what we observe)". The first is better.

**RV-9** (minor, correctness; confident). "Covariance".
Anchor: `Covariance is linear in each argument: $\textrm{Cov}(cX+d,Y)=c\,\textrm{Cov}(X,Y)$`
The displayed identity is homogeneity and shift invariance only. Linearity also needs $\textrm{Cov}(X+Z,Y)=\textrm{Cov}(X,Y)+\textrm{Cov}(Z,Y)$, which is not stated, yet it is what the text uses three paragraphs later ("Expanding the square of a sum of $n$ deviations in the same way gives $\textrm{Var}(\sum_i X_i)=\sum_i\sum_j\textrm{Cov}(X_i,X_j)$"), in the covariance matrix, and throughout the next section's Gaussian computations. Add the additive identity; it follows from linearity of the mean in one line.

**RV-10** (moderate, clarity of thought, correctness; confident about the facts). "Covariance", last paragraph.
Anchor: `the *law of large numbers*. The rule needs a finite variance: for the Cauchy`
"The rule" has no clear antecedent: the sentence before it names the law of large numbers, which needs only a finite mean, while the $\sigma/\sqrt n$ rule is what needs a finite variance. As written a reader will learn that the law of large numbers requires a finite variance. Rewrite: "The $\sigma/\sqrt n$ rule needs a finite variance. The law of large numbers itself, that $\bar X_n$ converges to the mean, needs only a finite mean (:numref:`sec_mdl-statistics`); the Cauchy distribution has neither, and its running means did not settle."

**RV-11** (minor, repetition). "Conditional Densities and Independence", "Conditional Expectation", "Covariance", "Correlation", Summary.
Anchor: `dependence has none, so zero covariance does not mean independence.`
"Uncorrelated does not imply independent" is taught five times in this section (the forward remark at the end of the independence subsection, the dart under Eve's law, the dart and $X=Y^2$ under Covariance, panel (c) of the correlation figure, the summary), once more in exercise 5, again in the next section's conditioning subsection, and it is in `sec_prob` with the same $Y=X^2$ example. The dart version is the memorable one. Keep the dart under Covariance and panel (c); drop the forward remark ("Independence constrains the whole joint distribution. It is stronger than ...") and the discrete $X=Y^2$ example, which chapter 1 owns.

**RV-12** (moderate, pedagogy). "The Covariance Matrix", last paragraph.
Anchor: `covariance matrix are its principal directions: the top eigenvector maximizes`
Five results in five lines (principal directions, a variational characterization, PCA, the multivariate Gaussian, elliptical contours), none shown, all taught elsewhere. See structural finding 6. Replace by: "The eigenvectors of $\boldsymbol\Sigma$ are the directions of principal component analysis (:numref:`sec_mdl-svd-low-rank`) and the axes of the Gaussian's contours (:numref:`sec_mdl-distributions`)."

**RV-13** (minor, explanation). "The Multivariate Rule".
Anchor: `$\boldsymbol\phi=g^{-1}$ and $f=p_{\mathbf X}$:`
The passage introduces a second map $\boldsymbol\phi$ with regions $V$ and $W$ for the calculus theorem, and then substitutes $\boldsymbol\phi=g^{-1}$, so the reader has to match $W=g^{-1}(\mathcal D)$ and $V=\mathcal D$ unaided, with $\mathbf x$ and $\mathbf y$ swapping roles between the two displays. State the theorem once in the letters that will be used ($\mathbf x=g^{-1}(\mathbf y)$), or say in a clause which region is which. The $\operatorname{diag}(2,3)$ warm-up before it is exactly right.

**RV-14** (moderate, clarity of thought). Closing paragraph before the Summary.
Anchor: `quickly is the subject of the law of large numbers in`
Three problems in one paragraph. (i) It opens with "We can now compute ...", the fixed closing formula the guide singles out, and the list that follows restates the section. (ii) The open question is a good one (Chebyshev allows 25% where the Gaussian has 4.6%, and gives $\sigma^2/(n\epsilon^2)$ where the truth is exponential), but it is assigned to the wrong result: the law of large numbers says nothing about how fast averages concentrate. The Gaussian value is explained by the central limit theorem, which is in the next section, and the exponential rate by :numref:`sec_mdl-concentration-generalization`. (iii) The paragraph does not lead to the next section at all. Cut the first sentence, keep the question, and end: "The next section shows where the Gaussian value comes from; why the chance of a large error falls exponentially in $n$ is the subject of :numref:`sec_mdl-concentration-generalization`."

**RV-15** (minor, code; deferred under decision 3). Imports cell.
Anchor: `from torch.nn import functional as F`
The section's code is NumPy throughout. The four framework tabs exist only to import `torch`, `nn` and `F` (and their counterparts), none of which is used. For this section the tabs fragment nothing, since there is no other tabbed cell, but the reader of the PyTorch tab is shown three imports that do nothing. One untabbed imports cell would do.

## Exercises

Twelve problems, inside the band for the appendix, and the strongest set I have seen in the book. Nearly every one ties a result of the section to something a deep learning reader does: clipping as $\min(X,c)$ of a variable with infinite mean; the ReLU output as a mixed distribution and the second moment that He initialization compensates; probability integral transform histograms as a calibration check; classes missing from a minibatch by indicators, with the small negative covariances that change the variance; $\rho\ge-1/(n-1)$ for ensembles; the sigmoid of a Gaussian becoming bimodal exactly at $s=\sqrt2$; inverse-transform sampling from a softmax with the float32 `cumsum` failure; Cantelli; the $R^2$ ceiling; a coupling layer and Box-Muller. They are graded, and the two starred projects are marked.

- **Exercise 1(d).** Anchor: `Cauchy distribution? Why can no such ambiguity arise for`. "$\mathbb{E}[X]$" here means the power-law variable of part (a), but it follows two integrals of the Cauchy density and reads as if it meant the Cauchy. Write "for the mean of the non-negative variable $X$ of part (a)".
- **Exercise 5(c).** Anchor: `Gaussian. A set in the plane has *measure zero* when, for every`. The part defines measure zero inside an exercise and asks for a covering argument, in a chapter whose introduction says measure theory is left out. The point (a pair can have Gaussian marginals and no joint density) is worth keeping; the route can be shorter: the pair lies on the two lines $y=\pm x$ with probability one, and any joint density would integrate to zero over two lines. Ask for that in two sentences.
- **Exercise 5** partly gives away the next section: the pair $(X,SX)$ is used in the body of "Conditioning" there as the counterexample. Fine, but the next section could cite the exercise instead of redoing it.
- **Exercise 12** is a project of five parts ending in training a model and estimating a tail index. It is honest about being a project. Its label of difficulty and the "[extended]" tag are right.
- One label is a leftover: `ex_mdl-random-variables-using-textrmvar-x-y` (the ensemble exercise). Harmless, but worth renaming while the file is open.

## Still open from October

The October review asked to keep "We have only pushed the problem one digit down". That sentence is gone; the dart opening that replaced it ("Asking for one more digit makes the answer ten times smaller") is better, so nothing is lost. The October complaints about this chapter were about section order and about later sections; none concerns this file.

## For the chapter review

- This section supplies the vocabulary of the whole chapter (density, c.d.f., mean, variance, covariance, conditional expectation, Markov and Chebyshev, change of variables). Later sections lean on it by label, and it stands on its own for a reader who lands on a subsection, with one exception: the conditional-expectation subsection uses $X$ for the target (RV-8).
- It should be two sections. The seam is clean and the running example survives it.
- It shares three things with its neighbour that should each have one home: the Cauchy averaging experiment (home: here), the pair $(X,SX)$ (home: the Gaussian conditioning subsection or the exercise here, not both), and the eigenvectors of a covariance matrix (home: the multivariate Gaussian).
- It repeats `sec_prob` for the first two moments. The author should decide once, for the part, whether the appendix re-motivates what chapter 1 motivated or only proves and extends it. I recommend the second.
- Patterns likely to hold for the chapter: the closing paragraph that begins "We can now"; main-text applications named without labels; a NumPy-only section carrying four framework tabs for unused imports; a small number of facts repeated four or five times each, which is far less than Basics had but in the same places (figure captions, section openings, summary).
- The method this section uses (one concrete object, every definition computed on it, each cell a test that could fail) is the model for the rest of the part.
