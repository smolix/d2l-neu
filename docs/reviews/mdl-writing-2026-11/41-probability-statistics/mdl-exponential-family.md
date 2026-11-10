# The Exponential Family

Source: `chapter_mdl-probability-statistics/mdl-exponential-family.md` (to the slides marker at line 1392). About 6,600 words of body prose outside code and display math, plus about 1,550 words of exercises (10 problems).

## Verdict

This is good textbook prose, close to the bar in several places, and I found no mathematical error after recomputing every worked number and checking each proof. Its strength is that one small data set, 9 heads in 13 flips, is carried through every construction: the logit $\log\frac94$, the tangent of $A$, the loss $13A(\eta)-9\eta$, the 8.02 nats that equal 13 entropies, the failure at 13 of 13, and the $\mathrm{Beta}(10,5)$ posterior. The proofs are the short ones, and three of them are followed by a sentence that says which step did the work. The main thing to fix is scope. The section teaches five subjects (the form, maximum entropy, the log-partition function with generalized linear models, three derivations of the softmax, and conjugate priors), each at about 1,100 to 1,300 words, with no ranking among them. A reader who arrives from the main text wanting to know why every loss has the gradient "prediction minus observation" has to pass maximum entropy first, and the result that the introduction promises as the point of the section sits in the third of five parts. Decide what the section is for (my reading: the log-partition function and the gradient rule), move that forward, and shorten or move the rest. The second thing to fix is the duplication with the information theory chapter, which proves Gibbs' inequality and the Gaussian maximum-entropy result again with the same argument.

## Scores

| Dimension | Score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The opening poses a puzzle the reader already owns.** "Yet when we differentiate these losses, all three give the same training rule" followed by "Agreement of this kind usually has a single cause." The second sentence is the kind of authorial judgment the voice guide asks for. Keep both.
- **The coin as a through-line.** Every abstraction is instantiated on 9 of 13 within a paragraph of its statement, and the numbers are right: $\hat\eta\approx0.811$, $A''(\hat\eta)=36/169$, $13\log\frac{13}{4}-9\log\frac94\approx8.02$, curvature $36/13\approx2.77$, posterior standard deviation $0.118$. Protect this when revising.
- **"The step that matters is the second"** after the maximum-entropy proof, and **"The proof works because a product of Gumbel CDFs is again a Gumbel CDF"** after the Gumbel proof. Each tells the reader what to remember. The first also earns the remark that no Lagrange multipliers are needed and that the proposition certifies a global maximum, which the Lagrangian calculation does not.
- **The failure cases are concrete and teach the hypothesis.** The uniform with observations 0.1, 0.2, 0.9, where moment matching returns $\theta=0.8$, "a value under which the observation $0.9$ is impossible". The coin with 13 heads, with the loss values $9.01$, $0.087$, $0.0006$, tied at once to separable logistic regression. The maximum-entropy problem on $\mathbb R$ with only the mean fixed. Each shows a condition failing instead of stating it.
- **The loss at the optimum equals the entropy of the fitted model**, shown for the coin and then in two lines for every family with $h\equiv1$. This is a small result that most treatments omit and that prepares the cross-entropy floor of the next sections.
- **"its gradient, the softmax, gives the chance that each option attains the maximum."** A sentence a reader will quote.
- **Code computes.** All four cells verify something stated (bisection for the tilt, autograd twice on $A$, Gumbel frequencies against $\mathrm{softmax}(\mathbf z/T)$, the grid posterior against the Beta density), and each is introduced by a question. The stored PyTorch outputs match every number the prose reads ($\eta=0.371$, $H=1.614$; four-digit agreement; maximum difference 0.001; $2.1\times10^{-14}$).
- **Figures carry weight.** `mdl-prob-log-partition.svg` shows the tangent with slope $9/13$ and, beside it, the loss with no minimum for 13 of 13. I rendered all three figures; they are legible and agree with the text.
- **Exercises** are among the best I have seen in the part (see below).

## Structural findings

1. **Five subjects, no ranking, and the promised result arrives third.** The introduction motivates the section by the common gradient of three models and by the logit. The body then goes: form (1,190 words), maximum entropy (1,080), log-partition function and GLMs (1,320), origins of the softmax (1,110), conjugate priors (1,290). The gradient rule, the answer to the opening question, is at line 696 of 1,150. Maximum entropy is an explanation of where the form comes from; nothing after it depends on it except one paragraph in the Boltzmann subsection. Recommended order: form, log-partition function (moments, moment matching, GLMs), softmax as the categorical family, conjugate priors, and then maximum entropy as the closing "where does the form come from" part. The introduction should say which of the "Four facts" matters most to a deep learning reader (the second).

2. **"Origins of the Softmax" is a second section inside this one.** Of its three arguments only the third belongs to the exponential family, and that one takes nine lines because, as the text says, "The third argument is the one this section has already made." The probit integral, McFadden, the Gumbel proof, the logistic-versus-Gaussian tail comparison and the Boltzmann law are a self-contained essay on the softmax (about 1,000 words). It is good, and the main text points at it by label (`softmax-regression.md:479`). Two options: (a) keep it here but cut the multi-option probit integral and the logit-versus-probit paragraph to two sentences each, since Exercise 4 covers the latter in full; (b) move the Gumbel and Boltzmann material to `mdl-distributions.md` beside the categorical distribution, and keep here only the categorical family. I prefer (a) for the least disruption.

3. **Duplicate proofs across chapters 41 and 42.** This section defines entropy, proves Gibbs' inequality by the tangent line, names the KL divergence, and proves a general maximum-entropy proposition with the Gaussian as a corollary. `mdl-information-theory.md` then defines entropy again, proves Gibbs' inequality again (line 443, with the tangent-line proof as an exercise), and proves "Gaussian maximum entropy" (line 668) by the argument used here, down to the same remark ("The first step is the one that matters"). One of the two should be the home. Since chapter 42 is where a lookup reader expects entropy and Gibbs, the cleanest fix is for chapter 42 to state the Gaussian result as a corollary of the proposition here in one clause with a reference, or for this section to cite forward for Gibbs and keep only the six-line proof of the proposition. Either way the general proposition here is the stronger statement and should survive.

4. **Conjugacy is taught here and the Beta update again in maximum likelihood, with a different reading of the parameters.** This section says "The parameters $a$ and $b$ act as prior counts of heads and tails". `mdl-maximum-likelihood.md:1192` says "The prior acts as $a-1$ *pseudo-counts* of heads and $b-1$ of tails". Both are defensible (mean versus mode), but a reader who meets both within two sections will think one is wrong. See EXP-14.

5. **The section stands on its own for a lookup reader only in part.** The phrase "the catalogue" (five times) and "the previous section" refer to `mdl-distributions.md` without a reference after the introduction, and "the chapter's coin" is used as a name. A reader arriving at :numref:`subsec_mdl-glm` from softmax regression lands on a paragraph that uses $A$, $T$, "the moment property" and $h$ with no pointer back. Add one clause with equation references at the top of the GLM subsection.

6. **What a deep learning reader would look for and not find.** The GLM subsection covers only the canonical link (see EXP-9). There is no mention that the Hessian $\nabla^2A=\mathrm{Cov}(T)$ is the Fisher information in natural parameters, although the next-but-one section builds on Fisher information and the curvature $13A''(\hat\eta)$ is computed here with the promise "we ask what the curvature of the loss at $\hat\eta$ tells us"; one forward clause would connect them. Neither gap is serious.

## Passage-level findings

**EXP-1** · minor · language · under the title (introduction)
> "The previous section built its catalogue of distributions one law at a time:"

The section opens by recapping its predecessor, which the guide (§7.7) limits to a clause and which does nothing for a reader who arrives by reference. The strong opening is the second half of the paragraph. Start with "Linear regression, logistic regression and softmax regression each come with a loss of their own" and bring the catalogue in afterwards with a `:numref:`.

**EXP-2** · minor · explanation · introduction
> "Four facts follow from the form alone."

The four are listed at equal weight and the list is repeated almost verbatim in the next paragraph ("The form tells us which loss to train with and guarantees that fitting a linear model to it is a convex problem") and again in the closing paragraph and the Summary. Say once which fact the reader came for, and drop the restatement in the following paragraph.

**EXP-3** · minor · language · introduction
> "The shared computations of this section use NumPy; each framework's"

Housekeeping in the last line of the opening, followed by an MXNet-only cell that imports `nd`. The PyTorch imports cell also imports `nn`, `F` and `d2l`, none of which the section uses (the one call that could use `F` is written `torch.nn.functional.softplus`). Trim the imports and move the sentence next to the first framework cell.

**EXP-4** · moderate · explanation · "Definition and Members"
> "The base measure does depend on $\mathbf x$, so $T(\mathbf x)$ does not fix the probability of $\mathbf x$; it fixes everything in $\mathbf x$ that carries information about"

This is the definition of sufficiency, and it is stated as an afterthought in a sentence that begins with the base measure. The reader has just been told that two data sets with the same sum "lead to the same estimates", which is the usable form. Either cut this sentence or replace it with the concrete case: "HHTHT... and any reordering have the same probability under every bias, so once we know there were 9 heads the order tells us nothing more about the bias."

**EXP-5** · minor · correctness (sure) · "Definition and Members"
> "dividing $\boldsymbol\eta$ by the same constant, for instance, describes the"

Correct, but the more important non-uniqueness for what follows is the one the section later needs: adding a constant to $T$ (absorbed in $h$ and $A$) and, for the categorical family, redundant statistics. Since "minimal" is defined 430 lines later, a forward clause here would help.

**EXP-6** · minor · explanation · "Distributions Outside the Family"
> "is a Gaussian whose precision $1/\sigma^2$ is itself drawn at"

The Student-$t$ is introduced in passing without its density or a reference (the scale-mixture construction is Exercise `ex_mdl-distributions-student-t` in the previous section, and the $t$ distribution returns in `mdl-statistics.md` for small-sample intervals). The claim that "no fixed number of sums of the data can summarize" its likelihood is asserted, not shown. Add the reference to the exercise and say that the claim is a theorem (Pitman, Koopman, Darmois) which we do not prove. The following paragraph's claim about two local minima is true (I checked: with five points at $-5$ and five at $+5$ the location likelihood has minima near $\pm4.9$, $\pm4.7$ and $\pm3.9$ for $\nu=1,3,10$) and would be more convincing with those numbers.

**EXP-7** · moderate · thought · "Entropy and a Loaded Die" and "The Maximum-Entropy Proposition"
> "The proof rests on one inequality, which follows from a tangent line."

See structural finding 3. Gibbs' inequality, the KL divergence and entropy are introduced here and again in chapter 42. Decide on one home. If the lemma stays here, the KL sentence after it ("is the *Kullback--Leibler divergence* of $q$ from $p$") should carry a forward reference to the section that studies it.

**EXP-8** · minor · correctness (fairly sure) · "The Maximum-Entropy Proposition"
> "zero wherever $h$ is zero and has the same mean statistic,"

The proposition tacitly needs $\mathbb{E}_q[T]$ to exist and $H_h[q]$ to be defined (for densities, $q$ must have a density), and for the density case the lemma is invoked with "the sums become integrals" and no further comment. This is acceptable at the level of the text, but the statement says "any distribution". Write "any distribution (with a density, in the continuous case) for which $\mathbb{E}_q[T]$ is finite".

**EXP-9** · moderate · correctness (sure) · "Generalized Linear Models"
> "distribution from an exponential family and makes its natural parameter a"

What is defined is the generalized linear model with the canonical link. A GLM in general sets $g(\mu)=\mathbf w^\top\mathbf x$ for a link function $g$; probit regression, which this same section discusses 80 lines later as an alternative to the logit, is a GLM with a non-canonical link, and its loss does not have the gradient "prediction minus observation". The text never says so, and the Summary's "A generalized linear model trains with the gradient input times prediction minus observation" is false for the general notion. Add one sentence: "This choice of the natural parameter as the linear predictor is called the canonical link; other links (the probit below is one) keep the model but lose the simple gradient." Also note in the table that the Gaussian row is the one-parameter family with fixed variance and $h(y)\propto e^{-y^2/2}$, which is a different member from the two-parameter Gaussian of the first table.

**EXP-10** · minor · correctness (sure) · "Maximum Likelihood as Moment Matching"
> "This holds when no linear combination of the statistics is constant, which"

Should read "no nontrivial linear combination of the statistics is constant on the support" (the zero combination is always constant). The preceding "which requires $\textrm{Cov}(T)$ to be positive definite" is true for these families but not for strict convexity in general; "which holds when" is safer than "which requires".

**EXP-11** · minor · pedagogy · "Maximum Likelihood as Moment Matching"
> "A large curvature means that the loss rises steeply away from $\hat\eta$, so the"

The curvature $36/13\approx2.77$ is computed, and the Bernoulli subsection promised to "ask what the curvature of the loss at $\hat\eta$ tells us", but the reader gets only the qualitative sentence. One clause closes the loop and prepares Fisher information: the curvature is $n\,A''$, so the width of the loss in $\eta$ is about $1/\sqrt{2.77}\approx0.60$, and :numref:`sec_mdl-maximum_likelihood` turns this into an error bar.

**EXP-12** · moderate · explanation · "Noisy Choices and Gumbel Noise" through "Temperature and the Boltzmann Distribution"
> "for $T>0$, Gumbel noise of scale $T$ samples from"

$T$ is the temperature here and the sufficient statistic $T(\mathbf x)$ everywhere else in the section, sometimes in adjacent sentences: the Boltzmann subsection takes "the logit $z_k$ of class $k$ as the statistic" and two lines later sets "$T=1/\eta$". The text visibly avoids writing $T(k)=z_k$ there. Rename the temperature (the main text uses $T$ in `softmax-regression.md`, so the cheaper change may be to write the statistic as $\boldsymbol\phi(\mathbf x)$ or $\mathbf t(\mathbf x)$ throughout; otherwise use $\tau$ for temperature here and say that the main text calls it $T$).

**EXP-13** · minor · pedagogy · "Temperature and the Boltzmann Distribution"
> "Among all distributions on the classes with a"

The maximum-entropy reading of the softmax fixes "a given average logit", but the reader is given no reason why anyone would constrain the average logit. In physics the constraint is the mean energy, which is why the argument is natural there. Say so in one clause ("in physics this is the constraint of a fixed mean energy"), or cut the paragraph down to the statement and the worked number $1.49$.

**EXP-14** · moderate · correctness (sure, an inconsistency rather than an error) · "The Beta Prior on a Coin" and Exercise 9
> "act as prior counts of heads and tails, and observing $x$ heads and $n-x$"

Under this reading $\mathrm{Beta}(1,1)$ is one prior head and one prior tail, which the text confirms: "add one imagined head and one imagined tail". Exercise 9(b) then says of the prior $\mathrm{Beta}(2,2)$: "explain why one imagined head and one imagined tail restore a solution", which uses the mode convention ($a-1$, $b-1$) of `mdl-maximum-likelihood.md`. Within the chapter the uniform prior is thus both one imagined flip of each kind and no imagined flips. State the convention once: $a$ and $b$ are pseudo-counts for the posterior mean, $a-1$ and $b-1$ for the posterior mode, and Laplace's rule is the mean. Then fix the exercise wording ("the prior $\mathrm{Beta}(2,2)$ adds one imagined head and one imagined tail to the mode").

**EXP-15** · minor · language · "The Beta Prior on a Coin"
> "This is the $1/\sqrt n$ law for posterior widths."

"Law" overstates a one-example observation, and the reader has not been shown a law. "The posterior width shrinks like $1/\sqrt n$, as the standard error of the next section will" says the same and points forward.

**EXP-16** · minor · explanation · "Conjugacy for Every Exponential Family"
> "finite for discrete data, where the likelihood is at most one, and for"

This paragraph answers a question the reader has not asked (is the posterior normalizable?), presupposes a proper prior without saying so, and ends on "almost every data set in general", which a reader cannot check. Either cut the paragraph or reduce it to "If the prior is a proper density, so is the posterior."

**EXP-17** · moderate · thought · "Gamma–Poisson and Dirichlet–Multinomial"
> "Two more pairs follow the same recipe and appear later in the book: one for"

"Later in the book" has no reference, and I could not find the later use: a search of all chapters finds no Gamma–Poisson pair or negative binomial outside this section, and the Dirichlet appears elsewhere only as a way to draw random probability vectors in code. Add-one smoothing returns in :numref:`sec_mdl-naive_bayes`, but through Laplace's rule for a coin. So the subsection (about 330 words, four new named distributions) is motivated by a promise the book does not keep. Either give the uses by label, or cut the subsection to the Dirichlet with the three-word vocabulary, which is the part with a consequence (an unseen word no longer has probability zero), and move the Gamma–Poisson pair into Exercise 7, which already derives it.

**EXP-18** · moderate · language · end of "Gamma–Poisson and Dirichlet–Multinomial"
> "We can now write the distribution of a binary, categorical, count,"

A recap paragraph in the "We can now ..." formula that the guide (§12.2) asks to avoid, placed under a subsection heading it does not belong to, and followed immediately by a Summary that says the same things. The same results are thus listed in the introduction ("Four facts"), here, and in the Summary. `mdl-statistics.md` closes with the same construction ("We can now attach an error bar"). Keep only the last four sentences, from "Two gaps remain", which hand the argument to the next section, and vary the wording between the two sections.

**EXP-19** · minor · language · Summary
> "softmax arises as the choice under Gumbel noise, as the Boltzmann"

The Summary is nine sentences that list every subsection. The guide asks for three to six that answer the section's question. A candidate: the form; $\nabla A$ and $\nabla^2A$; convex fit by moment matching when the average is attainable; the gradient rule for canonical GLMs; conjugate update by adding statistics.

## Exercises

Ten problems, well graded, and almost all of them extend the text instead of repeating it. Particularly good: 2 (the base measure decides between Poisson and geometric, and part (d) uses the proposition to certify the Lagrangian stationary point), 5 (gradient descent on non-minimal logits conserves $\mathbf 1^\top\mathbf z$), 6(c) (overdispersion leaves the Poisson fit unbiased but underestimates its spread), 7(c) (a flat prior on the logit is the improper $\mathrm{Beta}(0,0)$ on $\theta$), 8 (the $1/n$ rate of the uniform's maximum), and 9. I checked the stated answers of 3(c), 4(a), 4(c), 7(b), 8(a,b,d), 9(c) and 10.

- **3(a)** repeats the text: the sentence after the autograd cell already says the Poisson's equal mean and variance "is the statement $A=A'=A''$". Drop part (a) or replace it with the binomial.
- **1** proves the Gumbel proposition a second way and overlaps the main-text exercise `ex_softmax-regression-gumbel-max-trick`, whose parts (a) to (c) are the proof given in this section's text. A reader who follows the main text's reference here finds that exercise solved. This is acceptable for an appendix, but the main-text exercise could then ask for the exponential-race proof instead.
- **9(b)**: fix the pseudo-count wording (EXP-14).
- Seven of the ten labels are `ex_mdl-distributions-*`, left over from before the section was split out. Harmless to the reader.

## Still open from October

The October review did not treat this section separately. Its chapter-level complaint that "Laplace's rule of succession is derived three times" is resolved in substance: it is derived here once, and `mdl-maximum-likelihood.md` and `mdl-naive-bayes.md` cite it. What remains is the inconsistent pseudo-count convention (EXP-14).

## For the chapter review

- **Order and role.** The section sits well between the distributions and statistics: it turns the catalogue into one form and ends on the question (how far can $9/13$ be from the truth?) that the next section answers. It assumes the previous section's derivations and notation and gives few references back, so it reads better front to back than by lookup.
- **Overlaps to settle at chapter level.** Entropy, Gibbs' inequality and Gaussian maximum entropy with chapter 42; the Beta update and pseudo-counts with `mdl-maximum-likelihood.md` (`subsec_mdl-beta-map`); conjugacy restated in `mdl-bayesian-computation.md:94`; Markov-style restatements are not an issue here.
- **Suspected chapter-wide patterns.** (1) Sections close with a "We can now ..." recap paragraph and then a Summary, so results are stated three times (introduction, closing, Summary). (2) Each section is several topics of equal weight with no statement of which is the important one. (3) Notation is reused across sections with different meanings ($T$: sufficient statistic, temperature, and in the next section a test statistic; $p$ versus $\theta$ for the coin's bias). (4) Terms at definition are italic here and bold in `mdl-statistics.md`.
- **Strength to preserve across the chapter:** the single coin (9 of 13) and the habit of checking each derived number in a cell that is introduced by a question.
