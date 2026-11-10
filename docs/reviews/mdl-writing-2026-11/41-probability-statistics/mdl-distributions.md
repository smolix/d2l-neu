# Distributions

Source: `chapter_mdl-probability-statistics/mdl-distributions.md`. About 7,350 words of prose and mathematics in the body (discrete 2,100; continuous 2,600; multivariate Gaussian 1,800) and 1,850 in the exercises; 14 code cells besides the imports; 5 pre-generated figures.

## Verdict

Good textbook prose built on a real idea, and no longer the gallery the October review asked to cut. The idea is stated in the third paragraph, "a handful of distributions grow out of a single coin flip", and the section keeps to it: the binomial is a sum of flips, the Poisson its rare-event limit, the exponential the wait in the same stream of events, the Gamma the wait for the $k$-th event, the Gaussian the shape of the binomial near its peak, the Laplace an exponential with a random sign. Because each distribution is built, its mean and variance come from the construction, and the reader is never handed a formula to memorize. Two running examples (the coin with 9 heads in 13, the server with four requests a minute) supply numbers a reader will remember. All the numbers are right; I recomputed them. Against the bar it falls short in three ways. The Gaussian subsection, the centre of the section, is told in an order that hides its own argument and opens with a motivation the section has already refuted. The two propositions on Gaussian vectors are proved in a loop (the first cites the second, which cites the square case of the first) that the text then has to defend against the charge of circularity. And in between the good derivations the section falls into a catalogue rhythm: 33 bold run-in headings, and 11 cells introduced by "Does ...? The cell ...", five of which check arithmetic that cannot fail. The most important fix is to reorder the Gaussian subsection so that it reads as one derivation.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **Construction instead of catalogue.** The family-tree figure is the section's table of contents and its thesis in one picture, and the text follows it. The sentence "Because each law is built from simpler pieces, its mean and variance follow from the construction, by a short argument that is different each time" is a promise that is kept: linearity for the binomial, an index shift for the Poisson, integration by parts for the exponential, $S^2=1$ for the Laplace.
- **One stream of events, two questions.** "The wait for the next request exceeds one minute exactly when the minute contains no request", so the exponential's survival function is the Poisson's $P(0)$. The Gamma is then "the $k$-th event comes after $t$ exactly when at most $k-1$ events fall in $[0,t]$", and the telescoping sum gives its density. The cell that draws exponential gaps and finds Poisson counts ("Gaps drawn as exponentials produce Poisson counts") tests the whole construction at once. This is the best passage of the section.
- **Numbers that stay.** "A silent minute occurs about once an hour, and a burst of ten or more about once in two hours." "In 43% of minutes, fewer than four requests arrive, although four is the average." "Half of all waits are shorter than about 10 seconds even though the mean is 15." Each is a sentence a reader can quote a week later.
- **Hypotheses shown by breaking them.** "If the 13 'flips' were one flip copied 13 times, $X$ would be $0$ or $13$", with variance $42.25$, thirteen times $np(1-p)$: the mean needed no independence and the variance did. The light bulb whose filament thins marks the limit of memorylessness. The rank-deficient $\mathbf A$ gives a Gaussian "spread along the line $y_1=y_2$" with no density.
- **The bell from the ratio.** The successive ratio of the binomial, introduced as a way to avoid factorials, returns as the tool that produces the Gaussian: its logarithm is linear in the distance from the mean, and "A function whose steps grow linearly is a parabola." This is the short proof of de Moivre and Laplace, it uses nothing the reader lacks, and it is checked on the coin ($0.0846$ against $0.0873$) at once.
- **The author answers the reader.** "A reader may object that uncorrelated variables need not be independent. That is right, and the proof used more than zero correlation." This is the voice the guide asks for, and it is the only place in the section where it is this direct.
- **Small practical truths.** $-\log(1-U)$ instead of $-\log U$ "because generators return values in $[0,1)$"; the polar factor $r$ gives the antiderivative "which $e^{-x^2/2}$ alone does not have".
- **The summary table**, with its "Model assumption" column, is what a lookup reader wants.

## Structural findings

1. **The Gaussian subsection is out of order.** It runs: bell from the log-ratio (with the constant $\sqrt{2\pi npq}$ used but deferred), the standard Gaussian (with $1/\sqrt{2\pi}$ asserted), the central limit theorem, sums of uniforms, then "The normalizer" and "The variance" (which finally establish that $\phi$ is a density with variance 1, and that $\mu,\sigma$ are what the text has been calling them), then the Cauchy, then the squared-error loss. The reader is asked to accept the general theorem about a density whose normalization and moments have not yet been shown. The natural order is: bell; normalizer (it is promised by "which the normalizer below fixes" and should follow at once); standard Gaussian and its variance; the general $\mathcal N(\mu,\sigma^2)$; the loss; then the central limit theorem with the sums of uniforms as its picture, and the finite-variance caveat in one clause. Nothing needs rewriting, only moving.

2. **The proofs for Gaussian vectors loop.** The linear-map proposition is proved for square $\mathbf A$, and for wide $\mathbf A$ by appeal to "the conditioning proposition below"; that proposition is proved using "the square case of the previous proposition"; and a later sentence assures the reader that this is "without circularity". It is correct, but an argument that has to deny being circular is in the wrong order. Order it as three steps: (i) Lemma, the square invertible case; (ii) Proposition, residual independence and Gaussian blocks, from (i); (iii) Corollary, wide matrices, from (i) and (ii). The text is then shorter and the defensive sentence goes.

3. **A third of the cells cannot surprise.** Under the house rule every cell must compute or verify something the prose discusses, and all 14 do. But five verify arithmetic identities or tautologies: `p(1-p)` at three values of $p$; $13!/(9!\,4!)=715$; the die's mean and variance from sums against the closed forms that were derived from the same sums; that subintervals of a uniform sample are equally likely; and the Cauchy averages, which repeat the previous section's cell. Each is followed by a sentence of the form "Both pairs agree". The reader learns to skip them, and then skips the cells that do carry weight (the total-variation rate, the bell's error at $n=13$, gaps giving counts, the Laplace tail, the conditional slice). Cutting the five would remove about 300 words and five interruptions. (Changing cells is deferred under the author's decision 3; the finding stands for the later pass.)

4. **Lookup.** The file has one subsection label (`subsec_mdl-gaussian-clt`). Poisson, exponential, Gamma, Laplace, the multivariate Gaussian, sampling by a factor, and Gaussian conditioning cannot be referenced except through the section as a whole, and the conditioning formula `eq_mdl-mvn_conditional` is referenced from nowhere outside this file. The Gamma distribution and the Poisson process have no heading at all (they are bold run-ins under "Exponential" and "Poisson"), so they are missing from the table of contents although the Gamma has a row in the summary table. Add labels to the subsections other chapters should cite (at least the multivariate Gaussian's three), and give the Gamma its own `###`.

5. **Where are the other distributions?** A reader who looks up "Distributions" for the Beta, Dirichlet, Student-$t$, log-normal, Cauchy, geometric or negative binomial finds none of them in the table. They exist: log-normal and Cauchy in :numref:`sec_mdl-random_variables`, Beta and Dirichlet in :numref:`sec_mdl-exponential-family`, the rest in exercises 3, 4 and 7. Add one line under the summary table that says where each lives.

6. **Connection to deep learning is asserted without a single label into the main text.** The section's selling point is that each distribution "fixes a loss", and it names binary cross-entropy, softmax cross-entropy, squared error, absolute error, the $\ell_1$ penalty, Gaussian process regression, and "the default prior and latent distribution of generative models". Outside the exercises there is no `:numref:` to any of them. The matrix-factor sampler $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$ is the reparameterization that variational autoencoders and diffusion models differentiate through, and the text does not say so. Reference the main-text sections that define the four losses, the Gaussian process chapter where conditioning is used, and name the reparameterization where the sampler is derived.

7. **Scope and length.** At 7,350 words the body is the right size for what it covers. The multivariate Gaussian (1,800 words) is different in kind from the coin-built distributions and is the part other chapters need most; it could be its own section, and if the previous section is split, the chapter would then have a clean sequence (one variable; several variables; named distributions; Gaussian vectors). I would make that move, but the section also works as it is once the multivariate subsections have labels.

8. **Overlap with chapter 1** is small and limited to the Bernoulli: `sec_prob` already shows $\textrm{Var}=p(1-p)$ from $X^2=X$ and that the fair coin has the largest variance, in nearly the same words.

## Passage-level findings

**DIST-1** (minor, language). Opening, third paragraph, and throughout.
Anchor: `Gaussian. Because each law is built from simpler pieces, its mean and variance`
"Law" is used about 25 times as a synonym for "distribution" and is never introduced; the previous section says "distribution" throughout, and the guide asks for the technical noun to be repeated, not varied. The synonym also collides with "law of rare events", "law of large numbers" and "law of total variance", where the word means a theorem. Use "distribution" (the table header "Law" included).

**DIST-2** (minor, repetition). Opening, fourth paragraph; each subsection; Summary.
Anchor: `turns into a familiar loss: binary cross-entropy, softmax cross-entropy,`
The list of four losses is given here, each loss is derived in its subsection (where it belongs), and the list is given again in the Summary's first sentence. The first paragraph has already said "that choice fixes the loss it minimizes". Cut this fourth paragraph to its second half (the consequence of a wrong choice), which is new.

**DIST-3** (minor, explanation). "Bernoulli" and "Binomial".
Anchor: `largest at the fair coin? The cell flips the running coin $10{,}000$ times.`
The section has two coins and calls both "our coin". The opening asks about a fair coin that shows 9 heads in 13. The Bernoulli subsection then sets $p=9/13$ ("If we use the observed frequency ... of our coin as $p$") and simulates "the running coin" with that bias; the Binomial subsection returns to "For our coin, ... each of probability $2^{-13}$". A reader following the thread has to work out that the first is the plug-in estimate and the second the null hypothesis, which are the two things the statistics section will later distinguish with care. Say it in a clause at the Bernoulli ("the coin we would guess from the data, not the fair coin of the question"), or use the fair coin in the Bernoulli cell.

**DIST-4** (minor, repetition with chapter 1). "Bernoulli".
Anchor: `predictable, and a coin with $p=0$ or $p=1$ is a constant. If we use the`
The moments of the indicator and "the fair coin is the least predictable" repeat `sec_prob`. One sentence with the reference is enough; the cell that prints $p(1-p)$ at three values then has nothing to show (structural finding 3).

**DIST-5** (minor, explanation). "Poisson".
Anchor: `requests, and treat each piece as an independent $\mathrm{Bernoulli}(4/n)$`
The two modelling assumptions of the Poisson (no slice holds two events; slices are independent with equal probability) arrive as instructions inside a sentence ("so short that ...", "treat each piece as ..."). They are the content of the model: the later remark that "the Poisson model is wrong" for overdispersed counts means one of them fails (rates that vary, or requests that trigger requests). State them as assumptions, in a sentence of their own, so that the diagnosis three paragraphs later has something to point at.

**DIST-6** (minor, clarity of thought). "Poisson", after the last cell.
Anchor: `at $n=20$, $0.010$ at $n=100$ and $0.001$ at $n=1000$: each tenfold increase`
The three values of $n$ are 20, 100 and 1000, so only one step is tenfold; the first is fivefold (and the distance falls about fivefold, which supports the same conclusion). Write "the distance falls in proportion to $1/n$" and drop "each tenfold increase".

**DIST-7** (minor, pedagogy; deferred under decision 3). "Categorical and Multinomial".
Anchor: `Do the closed-form sums give the mean and variance of a die?`
The cell sums $k\,p_k$ and $k^2p_k$ for a die and compares with $(n+1)/2$ and $(n^2-1)/12$, formulas that the text derived from those sums five paragraphs earlier. Nothing is at risk. The same holds for the line that prints $13!/(9!\,4!)$. See structural finding 3. If the discrete uniform keeps a cell, let it show the claim that is not obvious: that $h^2(n^2-1)/12$ tends to $(b-a)^2/12$ as the points fill the interval.

**DIST-8** (moderate, clarity of thought). "Gaussian", first paragraph.
Anchor: `$10{,}000$ examples also has a binomial number of errors, and counting`
The motivation is that for $10{,}000$ examples "counting orderings there is hopeless". But two subsections earlier the text showed that "The pmf can be computed without any factorial", and the next cell computes the exact pmf at $n=1000$ with that function in order to grade the bell. The reader has been given the tool that makes the stated problem disappear. The real reasons for wanting the bell are different and stronger: a formula tells us the shape (centre $np$, width $\sqrt{npq}$, nothing else matters), it gives tail probabilities without summing thousands of terms, and the same shape will turn out to govern sums of anything. Open with those: "The ratio computes any binomial probability, but it does not tell us what the distribution looks like. Is there a formula for its shape near the peak?"

**DIST-9** (minor, correctness, rigor; confident). "Gaussian", "The log-ratio near the mean".
Anchor: `*de Moivre–Laplace theorem*, the first central limit theorem. With the`
The derivation is a sketch: it uses $\log(1+\epsilon)\approx\epsilon$ without a remainder, sums "from the mean to $np+x$" although $np$ need not be an integer (for the coin $np=6.5$ and $x=2.5$, so the sum $\sum_{j=0}^{x-1}$ has no meaning as written), and holds only for $x$ of order $\sqrt n$. The text then says "This is the de Moivre–Laplace theorem". Say what it is: "This is the heuristic behind the de Moivre–Laplace theorem, which makes the approximation precise for $x$ of order $\sqrt{npq}$." For the half-integer case, start the sum from the nearest lattice point or note that the parabola is fitted through the lattice values. The $3\%$ error at $k=9$ that the text reports is the honest measure of the approximation, and it is good that it is there.

**DIST-10** (moderate, pedagogy). "Gaussian".
Anchor: `**The normalizer.** Why the constant $1/\sqrt{2\pi}$? Let`
See structural finding 1. The constant is used in "With the normalizer it reads $P(X=np+x)\approx e^{-x^2/(2npq)}/\sqrt{2\pi npq}$", in the first cell, in the definition of $\phi$ and in the theorem, 90 lines before it is derived; "the two parameters are the first two moments" is likewise shown after the parameters have been called mean and standard deviation for two pages. Move "The normalizer" and "The variance" to follow the bell directly.

**DIST-11** (moderate, repetition across sections). "Gaussian".
Anchor: `**The Cauchy counterexample.** The theorem needs a finite variance. The previous`
This paragraph and its cell repeat the Heavy Tails subsection of the previous section: the same density, the same probability $\tfrac12$, the same experiment at $n=1000$, and word for word the sentence "The average of $n$ independent standard Cauchy draws is again standard Cauchy, a fact we state without proof". The one new item is the Gaussian comparison ($0.32$, $0.0016$, negligible), which follows from $\bar X_n\sim\mathcal N(0,1/n)$ without a cell. Replace paragraph and cell by one sentence: "The theorem needs a finite variance: the averages of Cauchy draws in :numref:`sec_mdl-random_variables` never settle, however many are averaged."

**DIST-12** (minor, language). "Gaussian", "Gaussian noise and squared error".
Anchor: `assumes the noise $y\sim\mathcal N(\hat y,\sigma^2)$ has the negative`
The sentence says the network "assumes the noise $y\sim\ldots$": $y$ is the label, not the noise, and the subject of "has the negative log-probability" is the network. Write: "A regression network that predicts $\hat y$ and models the label as $y\sim\mathcal N(\hat y,\sigma^2)$ assigns the observed label the negative log-probability ...". The paragraph's last sentence ("The Gaussian is also the default prior and latent distribution of generative models") is a claim with no reference and no consequence here; attach a label or cut it.

**DIST-13** (moderate, pedagogy). "Contours and Eigenvectors".
Anchor: `**High dimension.** For the standard Gaussian $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$`
Three facts in two sentences (the norm concentrates near $\sqrt d$; the cosine of two draws has standard deviation $1/\sqrt d$; the direction is uniform on the sphere), none shown, none used later in the section, under a heading about contours in two dimensions. The October review counted near-orthogonality as taught three times across two chapters; this is a fourth mention. Either cut it (the linear-algebra section owns the result and is already cited) or make it one sentence in the sampling subsection, where a standard Gaussian vector is first drawn.

**DIST-14** (moderate, clarity of thought). "Linear Maps and Sampling", proof of the first proposition.
Anchor: `block of a Gaussian vector, and the conditioning proposition below shows`
See structural finding 2. In addition, both propositions are headed only "**Proposition.**", unlike those of the previous section, which carry a name in parentheses. Name them ("Gaussian vectors under linear maps", "residual and marginal") so that the cross-references inside the proofs ("the previous proposition", "the conditioning proposition below") can use the names.

**DIST-15** (minor, explanation). "Linear Maps and Sampling", "Sampling by a matrix factor".
Anchor: `A matrix factor turns independent noise into correlated noise. The cheapest`
This is the reparameterization $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$ used to backpropagate through sampling, and the scalar case $x=\mu+\sigma z$ appeared in the Gaussian subsection. The section never uses the word. One clause here would connect the construction to the place the reader will meet it.

**DIST-16** (minor, language). Closing paragraph before the Summary.
Anchor: `We can now write down a likelihood for binary, categorical, count,`
The closing formula "We can now ..." is the one the previous section also ends with, and the guide asks that neighbouring sections not share it. The word "likelihood" is used here, and in "the Laplace likelihood is largest where ...", before the chapter defines it. The two open questions that follow are good and specific and should stay: start the paragraph with them.

**DIST-17** (minor, code; deferred). Imports cell and the central-limit plot.
Anchor: `from torch.nn import functional as F`
As in the previous section, the code is NumPy only and the four framework tabs import modules that are never used. A deep learning reader might reasonably expect one cell that shows the same objects in the framework (`torch.distributions.Normal(...).log_prob(y)` equal to the squared-error formula just derived); that single cell would justify the tabs and tie the "distribution fixes the loss" claim to code. In the stored central-limit plot the legend sits on top of the $n=1$ plateau that the prose points to.

## Exercises

Ten problems, in the band, and good. Several continue the text where it stopped: the rule of three for a classifier with no errors on $n$ examples (1c); the precision of a noisy observation as prior plus noise precision (2b); the Gamma-Poisson mixture that repairs the overdispersion the Poisson subsection warned about (3); the converse of memorylessness (4); superposition and thinning (9); the multinomial covariance promised in the text (10). They are graded and the hard ones are marked.

- **Exercise 5** (rate of the central limit theorem) is four parts of real numerical work with the Edgeworth expansion quoted from a monograph. It is a fine project and is marked as such, but its part (a) asks for the density of a sum by `np.convolve`, and the chapter never states that the density of a sum of independent variables is a convolution. Either the previous section says so in two sentences or part (a) does.
- **Exercise 6** (word burstiness) is an extended project that depends on *The Time Machine* utilities of the sequence chapter. It is the right kind of project and its dependence is declared.
- **Exercise 7(b).** Anchor: `met in this section is the case $\nu=1$?`. The Cauchy is "met" in the previous section and only recalled in this one; if DIST-11 is adopted it is no longer in this section at all. Write "met in :numref:`sec_mdl-random_variables`".
- **Exercise 2** has a leftover label, `ex_mdl-distributions-using-conditional-formula-eqref`.
- Nothing needs to be cut. If one were added, it should be on the log-normal or on products of many factors (the multiplicative counterpart of the central limit theorem), which initialization arguments in the main text rely on.

## Still open from October

The October review asked to "reduce the distribution gallery to a table and three derivations". The section has since been rebuilt around the coin and the server, and the constructions now carry it; the summary table exists. That finding is closed in spirit. What remains of the gallery is the rhythm (run-in headings and verification cells), described in structural finding 3.

## For the chapter review

- The section depends on the previous one for the density, the inverse transform, the change of variables, the variance of a sum and the Cauchy, and says so each time by label. It sets up the exponential family and the statistics section (the coin's tail probability $0.133$ is left as "a question for statistics"). Both hand-offs are stated in the closing paragraph and are accurate.
- Order within the chapter is right. If the previous file is split, consider making the multivariate Gaussian its own section; other chapters need it most, and at present it cannot be cited except through the whole section.
- Three items are shared with the previous section and need one home each: the Cauchy averaging experiment, the uncorrelated-but-dependent pair $(X,SX)$, and the eigen-ellipses of a covariance matrix.
- Patterns that probably hold for the chapter: "law" for "distribution"; the closing "We can now"; the "Does ...? The cell ..." introduction to every cell (11 times here), which is a good device when the answer could be no and a tic when it could not; applications in the main text named without labels; NumPy-only code under four framework tabs.
- The method that works here, and that later sections should be measured against: build the object from something the reader already has, let the construction give the formula, and attach one number in everyday units.
