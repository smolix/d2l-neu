# Maximum Likelihood

Source: `chapter_mdl-probability-statistics/mdl-maximum-likelihood.md`, read to the slides marker (line 1914). About 8,900 words of prose before the exercises and 11,400 with them; 9 code cells (PyTorch tab), 5 generated figures, 12 exercises. Anchors are quoted between « » and were checked against the source with whitespace normalized.

## Verdict

This is good textbook prose, and in its middle part (the score, Fisher information, the error bar, Cramér–Rao) it is better than that: the coin with 9 heads in 13 flips is carried from the first paragraph to the summary, the error bar is used before it is proved and the text says so, and the Cauchy location model is exactly the right second example, because its MLE is not an average and so the theory has something to add. I recomputed every worked number and derivation and found no mathematical error of substance. What keeps the section from the level of Cover and Thomas is its shape, not its sentences. It holds four topics (losses as likelihoods, accuracy of the MLE, priors as penalties, latent variables and EM), about a fifth of it re-teaches material that has a home elsewhere (entropy, KL and Gibbs' inequality in chapter 42; Gaussian noise and the Gaussian prior in chapter 2), and nine results are set as "Proposition", three of which are one line of algebra, so the reader cannot tell which results are the theorems of the section. The most important fix is structural: move "Latent Variables, EM, and the ELBO" into a section of its own, and cut the restated material to one clause with a reference, so that the section becomes what its best part already is, an account of what a likelihood's maximum and curvature tell us.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **One example carried the whole way.** The opening asks «But why that number, and how far should we trust it? Could the coin still be fair?» and the section answers both: $9/13$ from the principle, $\pm0.13$ from the curvature $61.0$, and "thirteen flips cannot tell this coin from a fair one". The same coin gives the empirical distribution, the entropy floor of $0.617$ nats, the Fisher information $1/(\theta(1-\theta))$, the pseudo-counts, and HHH. A reviser must not replace it with fresh examples.
- **Use before proof, said openly.** «We take that on trust for a moment and apply it.» The reader gets the number they were promised, then the reason. This is the voice the guide asks for (§10.2, leading the reader by the hand and candour about what is skipped).
- **The Cauchy location model.** «A model whose MLE is not an average shows what the theory adds.» It is used four times, each time for a reason: the information integral ($I=\tfrac12$), the numerical check of the two forms of $I$, the test of asymptotic normality by Fisher scoring, and the comparison of MLE, median and mean against the Cramér–Rao floor ($2$ against $\pi^2/4$, "about 20% fewer observations"). The stored outputs agree with every claim (score mean 0.004, variance 0.495, curvature 0.501; standard deviation 1.40; $n\,$Var 1.95, 2.44 and $9.5\times10^6$).
- **The heuristic derivation of asymptotic normality** in three numbered steps, with its one judgement: «its one non-routine step is to Taylor-expand the first-order condition rather than the NLL», and the remark that the numerator uses the variance form of $I$ and the denominator the curvature form. The reader learns why the information identity matters.
- **The $\epsilon$-window argument for densities** (the tolerance enters only through $-n\log\epsilon$), and the closing remark that the finite-sample cross-entropy identity has no continuous counterpart. Short, correct, and rarely stated this cleanly.
- **HHH as the motivation for a prior**: «Nobody who has handled coins believes this one can never land tails.» One of the few sentences in the section with a voice.
- **The mean and median from five readings** $\{1,2,3,4,100\}$: a computation the main text leaves to an exercise, done here in six lines.
- **The EM motivation**: «The problem is easy if the labels are known and hard only because they are missing.» followed by the hand-computed responsibility $1/(1+e)=0.27$ before any general formula.
- **Exercises.** Twelve problems, almost all of which extend the text instead of drilling it (learned variance and its collapse, losses that are not likelihoods, MAP in log-odds, the moving support that beats Cramér–Rao, the EM rate as a ratio of informations, the sandwich). I worked parts of exercises 1, 3, 4, 5, 7, 10, 11 and 12; all are answerable and correctly stated.

## Structural findings

**S1. The section is two sections (major).** The opening lists four tasks and the last, «Finally, we fit models in which part of every observation is hidden», is a different subject from the first three. "Latent Variables, EM, and the ELBO" runs about 1,900 words with its own figure, three cells and an exercise; it carries a `sec_`-level label (`sec_mdl-latent-em-elbo`) and is the target of references from at least seven other files (convexity, mutual information, divergences, discrete diffusion, latent diffusion, further reading, Bayesian computation). A reader who lands on it from the diffusion chapter arrives in the last fifth of a section about estimating a coin's bias. Make it a section of its own between this one and Bayesian Computation. The ELBO is then stated once there, EM uses it for latent variables, and the variational part of the next section cites it with the parameters in the role of the latent (see BC-7 in the neighbouring review). Without it this section is about 7,000 words and has one idea: what the maximum of a likelihood and its curvature tell us, and what a prior adds.

**S2. Entropy, KL divergence and Gibbs' inequality are taught here and again in chapter 42 (major, guide §7.7).** "Cross-Entropy and the Entropy Floor" defines entropy, defines the KL divergence, derives $\textrm{CE}=H+D_{\textrm{KL}}$ and proves Gibbs' inequality by the tangent line. The first section of the information theory chapter defines the same three objects, derives the same decomposition, proves Gibbs' inequality by Jensen, and then sets the tangent-line proof as an exercise (`ex_mdl-information-theory-gibbs-tangent`), which this section has already given in full. A reader going front to back meets the material twice, forty pages apart, with two proofs and no acknowledgement in this direction. The section does need the floor (for the classifier's training loss) and Gibbs (for consistency), so the material cannot simply go. Decide the home. My recommendation: keep here the regrouping identity (average NLL = cross-entropy from the empirical distribution), the coin's numbers and the figure, state $\textrm{CE}=H+D_{\textrm{KL}}$ and $D_{\textrm{KL}}\ge0$ in two sentences with the one-line tangent argument inline, drop the "Proposition/Proof" setting, and point forward by label to the information theory section for the full treatment. Chapter 42 then cites this section for the tangent proof instead of setting it as an exercise.

**S3. Two derivations repeat the main text at the same level (moderate, brief's "overlap" question).** `chapter_linear-regression/linear-regression.md` ("Squared Loss and Gaussian Noise") already defines the likelihood, explains the logarithm and the underflow, derives the Gaussian NLL, and gives a table of noise models with Gaussian and Laplace rows. `weight-decay.md` ("A Gaussian Prior on the Weights") already derives $\lambda=\sigma^2/(n\tau^2)$ by multiplying the negative log-posterior by $\sigma^2/n$. Here both are set as propositions with proofs, at the same depth. The appendix is deeper than the main text only in what surrounds them: the hinge counterexample, the mean/median computation, the flat prior on the bias, the Beta prior. State both results in a sentence each with a reference to the main-text derivation, and keep the additions. The reader who comes from the main text by the forward reference (`subsec_mdl-gaussian-mse`, `subsec_mdl-gaussian-prior-weight-decay`) currently finds the derivation they have just read.

**S4. Too many propositions of unequal weight (moderate).** Nine statements are set as "Proposition": discrete MLE = minimum cross-entropy, Gibbs, Gaussian NLL = squared error, consistency, asymptotic normality, Cramér–Rao, Gaussian prior = $L_2$ penalty, the ELBO gap, EM monotonicity. Three of them are one or two lines of algebra, and two (consistency and asymptotic normality) are cited without proof. The house style wants short proofs that are worth their place; a text that boxes everything leaves the ranking to the reader. Keep the proposition form for Gibbs (wherever it lives), Cramér–Rao, the ELBO gap and EM monotonicity, state consistency and asymptotic normality as cited theorems, and write the three identities as running derivations. Say once which result is the important one: asymptotic normality, since the error bar, efficiency and the limits all hang on it.

**S5. The link from Fisher information to deep learning is thin (moderate).** The opening promises that the loss «says how precisely the data determine the parameters», and the error-bar subsection says a training run can read error bars from the Hessian. For a network this fails (the model is not identifiable, $I$ is singular, the model is misspecified), and the section says so only through a bullet list that defers to exercises. Meanwhile the uses the rest of the book makes of the Fisher matrix are not named: the adaptive-methods section of chapter 40 and the divergences section of chapter 42 both point back here for it. Add three or four sentences at the end of "Limits of the Theory": what survives for a network (statements about the fitted distribution and its predictions, not about one weight vector), and by label where the Fisher matrix returns (as the local quadratic of the KL divergence, and as the curvature that natural-gradient and Adam-type methods approximate).

**S6. Length and what could go.** In order of how little would be lost: the float32 summation aside (ML-3); the learning-rate paragraph, which also gives away exercise 1 (ML-4); the Gaussian NLL and Gaussian prior proofs (S3); one of the two passages on parameterization dependence (ML-13); the closing "We can now" paragraph (ML-17). With S1 this brings the section to roughly 6,000 words without touching its best part.

## Passage-level findings

**ML-1** (moderate, language; pattern). "Thus" opens the gloss after an equation 14 times before the exercises, so every display is followed by a sentence of the same shape.
- Anchors: «Thus the MLE of a coin's bias is the observed frequency of heads.»; «Thus the NLL is a sum of one term per observation»; «Thus a cross-entropy is an entropy plus a KL divergence.»; «Thus MAP estimation minimizes the familiar NLL plus a penalty»
- The glosses themselves are good and should stay (they are the equation protocol doing its job). The fixed opener makes them sound mechanical. Vary it: drop the word in most places ("The MLE of a coin's bias is the observed frequency of heads."), and let some glosses precede the display.

**ML-2** (minor, explanation). Under "The Maximum Likelihood Principle".
- Anchor: «It assigns probabilities to data sets, never to parameter values.»
- A good sentence, placed where the reader cannot yet use it: no alternative has been mentioned. It pays off 1,000 lines later when the prior arrives. Add a half-clause here ("a distinction that will matter when we add a prior") or repeat the contrast in one clause at the start of the MAP part.

**ML-3** (minor, thought). Under "The Negative Log-Likelihood".
- Anchor: «This sum fits comfortably in single precision if it is accumulated pairwise»
- The paragraph leaves the argument (why we take logarithms) for a second numerical issue (how a float32 sum of a billion terms stalls at $2^{24}$). It is true (I reproduced it: a naive float32 running sum of $\log 2$ stalls at 16,777,216, after roughly 23 million terms, so "about twenty million" is a little low), but it is about summation, not about likelihood, and it weakens the point just made that the log "stays in range". Cut to one sentence or move to an exercise.

**ML-4** (moderate, pedagogy). Under "The Negative Log-Likelihood", after the gradient-descent cell.
- Anchor: «The learning rate $10^{-9}$ looks absurd.»
- The cell minimizes by gradient descent a function whose minimizer was derived two pages earlier, and the paragraph that follows is a lesson about learning rates and loss scale, which belongs to optimization. It also answers exercise 1 in advance: the text says that a thousandfold smaller count «barely moves $\theta$» and that tenfold larger counts «no longer converge», which are parts (b) and (c) of the exercise. Keep the cell (it is the first NLL minimized numerically) and two sentences: summed losses carry a factor $n$ in the gradient, which is why the book averages. Leave the experiment to the exercise.

**ML-5** (minor, explanation). Under "Cross-Entropy and the Entropy Floor".
- Anchor: «its training curve often levels off at some value above zero»
- This is the motivation for the whole subsection, and it is not picked up again in terms a practitioner can use. The classifier paragraph at the end states the identity "at each distinct input" and sends the consequence to an exercise. The consequence is the interesting part: when every training input is distinct, the empirical floor is zero, so the floor a practitioner sees is on held-out data. One sentence in the text would close the loop the opening sentence opened.

**ML-6** (minor, correctness; sure). Under "Gaussian Noise and Squared Error".
- Anchor: «so it is not the NLL of any conditional distribution»
- The argument shows that the hinge loss is not an NLL up to an additive constant; exercise 4 words it correctly ("up to a constant"). Add the qualifier, since the sentence before it defines the recipe as "dropping terms that do not depend on $\boldsymbol{\theta}$".

**ML-7** (moderate, explanation). Under "The Score and Fisher Information".
- Anchor: «The results of this part need a few»
- Five regularity conditions arrive in a bulleted list before the reader has seen any result that needs them, and before the first derivation that uses one. The pointer to two exercises is good. Better still: state the list after the mean-zero derivation, where "derivative and integral may be interchanged" and "the support does not depend on $\boldsymbol{\theta}$" have just been used and can be pointed at ("the step at the second equality is where the fixed support enters").

**ML-8** (minor, enjoyment). Under "The Score and Fisher Information", the Cauchy integral.
- Anchor: «where the substitution $u=\tan\varphi$ turns the integral into»
- The substitution has a probabilistic reading that is shorter and memorable: if $u$ is standard Cauchy then $\varphi=\arctan u$ is uniform on $(-\pi/2,\pi/2)$, and the score is $2u/(1+u^2)=\sin2\varphi$. So $I=\mathbb{E}[\sin^22\varphi]=\tfrac12$ at sight, the score is bounded by 1 (which is why outliers barely move the MLE), and the Monte Carlo errors of the check that follows come out exactly (0.007 for the mean score, 0.004 for its variance, 0.008 for the curvature).

**ML-9** (minor, thought). Under "The Score and Fisher Information", after the check.
- Anchor: «These deviations are of the order of the Monte Carlo error of»
- The stated tolerances ("a few hundredths", "within about $0.02$") are three to five times the actual standard errors, and "about $0.01$" is stated without a derivation. With ML-8 the three standard errors can be given, and the claim becomes a prediction the output confirms (stored: 0.004, 0.495, 0.501).

**ML-10** (moderate, explanation). Under "The Error Bar of the Coin".
- Anchor: «training run can compute the curvature of the NLL at its minimum, the Hessian, and one over its square root is the error bar»
- Stated "in general", before the theorem and without its conditions. It holds for a correctly specified, identifiable model at large $n$; for the networks the reader trains it does not (singular Hessian, misspecification, and the sandwich replaces $I^{-1}$). Say "for a model that satisfies the conditions above" and point to the limits subsection. See S5.

**ML-11** (minor, thought). Under "Consistency and Asymptotic Normality".
- Anchor: «This takes one definition and two facts from earlier sections.»
- The sentence is ambiguous (is the definition also from an earlier section?), and consistency was in fact defined in the statistics section, with the weak law as a proposition. For the lookup reader a restatement is right, but by §7.7 it should be one clause each with the reference, not a paragraph that defines $\xrightarrow{P}$ and $\xrightarrow{d}$ again.

**ML-12** (moderate, pedagogy). Under "Limits of the Theory".
- Anchor: «*Misspecification.* When no $p_{\boldsymbol{\theta}}$ is the data law»
- Of the five limits this is the one every deep learning reader lives in, and it gets one bullet and a pointer to the longest exercise in the section (marked extended, two dots). The October review objected to a 26-line paragraph here; the list is an improvement, but it now ranks all five limits equally and works none. Promote misspecification to a short paragraph with one number: a Poisson model fitted to overdispersed counts with mean $m$ and shape $\alpha$ still recovers the mean, and the inverse Fisher information understates the variance of $\hat\lambda$ by the factor $1+m/\alpha$ (this is exercise 12b, and it takes three lines). The rest of the sandwich can stay in the exercise.

**ML-13** (moderate, thought; repetition). Under "Beta Prior and Pseudo-Counts" and "Posterior Mode and Posterior Distribution".
- Anchors: «A prior is flat only in a particular parameterization.»; «The mode has a second weakness: it depends on how the parameter is written.»
- The same fact (a density's mode moves under reparameterization, the MLE does not) is explained in two consecutive subsections, each time with the log-odds example and each time pointing to exercise `ex_mdl-maximum-likelihood-map-log-odds`. The second subsection is 20 lines whose other content (MAP is one point, the posterior is a distribution) is said again in the closing paragraph and in the opening of the next section. Merge the two passages into one paragraph at the end of the Beta subsection and delete "Posterior Mode and Posterior Distribution" as a subsection; its label `subsec_mdl-map-vs-posterior` is referenced by exercise 7 and would move with the paragraph.

**ML-14** (minor, explanation). Under "Gaussian Prior and Weight Decay".
- Anchor: «The figure also shows the two limits.»
- It does not. I rendered `img/mdl-prob-map-prior.svg`: it shows one NLL, one prior and their sum. The limits $\tau\to\infty$ and $n\to\infty$ are described in the caption and the text, not drawn. Write "The same picture explains the two limits", or add a second, flatter prior to the figure.

**ML-15** (minor, correctness; sure). Under "The Evidence Lower Bound".
- Anchor: «with $q(z)>0$ exactly where»
- The condition is not needed for the bound and depends on $\boldsymbol{\theta}$, which the M-step then changes while $q$ is held fixed. If $q$ vanishes where the posterior is positive, the first equality in the display becomes "$\ge$" and the bound still holds; if $q$ puts mass where $p(x,z\mid\boldsymbol{\theta})=0$, the bound is $-\infty$ and holds trivially. Drop the condition, or say only that the proposition's KL term is finite when $q$ puts no mass outside the posterior's support. Also minor: the figure labels the evidence $\log p(x;\theta)$ where the text writes $p(x\mid\boldsymbol{\theta})$.

**ML-16** (minor, language). Under "Expectation--Maximization".
- Anchor: «EM has limits, and four of them matter in practice. The first is harmless.»
- A limit that matters in practice and is harmless. Write "EM has four limits. The first is harmless."

**ML-17** (moderate, thought; repetition, guide §7.7 and §12.2). End of body and "Summary".
- Anchor: «We can now turn a noise model into a loss, read a regularizer as a prior, put»
- The closing paragraph lists the section's results, and the Summary lists them again five lines later. The coin's $\pm0.13$ is stated four times (opening, error-bar subsection, closing, summary). The "We can now ..." closing is the fixed formula the guide names, and eight of the nine sections of this chapter end with it. Delete the paragraph except for its last two sentences, which hand over to the next section, and let the body end on the VAE remark.

**ML-18** (minor, language). Opening.
- Anchor: «Every model in this book was trained by minimizing a loss»
- Four files of this chapter open with the same construction ("Every model in this book was judged ...", "... has been judged ...", "Every model in the Basics part was trained ...", "Every accuracy reported in this book was measured ..."). Each is a good opening taken alone. Vary at least two; this section could open directly on the coin, which is the stronger hook.

## Exercises

Twelve, within the appendix band, and the best feature of the section after the Cauchy example. Assessment:

- **Exercise 1** (learning rate and loss scale) is a good problem about optimization, and the text gives away parts (b) and (c) (ML-4). Either trim the text or move the exercise to the gradient-descent section of chapter 40 and reference it.
- **Exercise 5** joins two unrelated topics (separable logistic regression; skewness of a proportion near the boundary). Both are good. Split, or retitle so the lookup reader can find the separable-data result, which the Bayesian section relies on.
- **Exercise 12** (the sandwich) is the most valuable problem and the longest, at four parts and about 50 lines. Part (b) should move into the text (ML-12).
- **Exercise 2c** asks «Which quantity that we can measure has the population conditional entropy $H(Y\mid X)$ as its floor?» This is the point the text should make itself (ML-5); as an exercise it is fine, but the answer deserves a sentence in the body.
- No exercise should be cut for quality. If EM moves (S1), exercise 11 moves with it.

## Still open from October

The October review's main complaint (maximum likelihood placed before the statistics it uses) is fixed: Statistics now precedes this section and is cited for the standard error, the weak law and consistency. The phrase "beyond our scope here" is gone. The 26-line paragraph on regularity, rates and exceptions is now a list plus exercises, which trades one problem for a smaller one (ML-12). Laplace's rule of succession is now derived once (in the exponential-family section) and cited here in one clause, as it should be. October's "keep" items, the underflowing product and the EM motivation, are both still present.

## For the chapter review

- **Order and fit.** Correctly placed after Statistics and before Bayesian Computation. It assumes the standard error of a proportion, the weak law, the CLT, the Cauchy distribution, conjugacy and the change-of-variables formula, all from earlier sections of this chapter and all cited by label.
- **It is the chapter's hub and its heaviest section.** Other chapters point into it for four different things: the NLL/cross-entropy identity, the Fisher information, the Gaussian prior, and the ELBO. That is the practical argument for S1: the ELBO deserves an address of its own.
- **Cross-chapter duplication to settle at the part level:** entropy, KL and Gibbs' inequality (here and chapter 42, with the proof here set as an exercise there); Gaussian NLL and Gaussian prior (here and chapter 2, same depth); the ELBO gap (here, again in the next section, and referenced from chapters 40 and 42).
- **Patterns likely to hold across the chapter:** the "Every model in this book ..." opening (four files); the "We can now ..." closing paragraph before the Summary (eight of nine sections); "Thus" as the fixed gloss after a display (14 here, none in the next section, so it is per-author-pass, worth a grep per file); routine identities set as propositions.
- **The coin of 9 heads in 13 flips** is the chapter's shared example (index, exponential family, this section, Bayesian computation, where the Laplace curvature is the same 61). It is the chapter's best unifying device and should be protected and, where sections introduce a different toy for the same purpose, extended.
