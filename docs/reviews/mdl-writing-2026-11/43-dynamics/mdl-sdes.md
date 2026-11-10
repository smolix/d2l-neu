# Stochastic Differential Equations

Source: `chapter_mdl-dynamics/mdl-sdes.md`. About 8,350 words of body prose and 2,500 words of exercises (12 problems), 13 code cells, 6 figures. All code is NumPy; outputs checked against `outputs/pytorch/chapter_mdl-dynamics/mdl-sdes.json`.

## Verdict

This is a good section and in places an excellent one. It takes a reader with calculus and no measure theory from a two-line recurrence to Itô's lemma, the OU kernel and the orders of Euler-Maruyama, and it is honest at almost every step about what is proved (quadratic variation, covariance, the isometry for step processes, the OU kernel and stationary law), what is a heuristic and said to be one (nowhere differentiability, the Taylor argument for Itô's lemma) and what is cited (Wiener, Donsker, existence, convergence orders). I recomputed every worked number and found no error in the mathematics; the defects of correctness are three loose sentences. What keeps it below the model texts is not the mathematics but the routine around it: thirteen cells, nine of them introduced by the same "question, then *The next cell*" formula, several of which verify by Monte Carlo a fact that was proved three lines earlier, and a running recurrence that is re-identified four times. The single most important fix is to cut the confirmation cells that teach nothing new and the repeated re-introductions of the recurrence, which would take the section from 8,300 words to under 7,000 and leave the four experiments that do teach (left against right sums, the Itô correction on one path, the strong-order slopes, the closed-form weak order).

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening recurrence.** `x_{k+1} = 0.99 x_k + sqrt(0.02) xi_k` is the whole section in one line, and the paragraph that follows poses the right puzzle: the decay is of order $h$, the noise of order $\sqrt h$, "its square is of order $h$, the same order as the drift. Ordinary calculus discards squares of small changes, so this one fact requires a new calculus." Keep this sentence; it is the one a reader will remember.
- **Honesty about nowhere differentiability**: "This is not a proof (it rules out a derivative at each fixed time, while "nowhere" concerns all times at once), but it shows the obstruction." This is the register the whole part should have.
- **The quadratic variation proof** is a real proof, short, and the right one (mean, then variance from the Gaussian fourth moment). The contrast with the smooth path via the mean value theorem comes first, so the reader knows why the statement matters.
- **Left against right Riemann sums** ("The two discretizations of "the same" integral disagree by a finite, deterministic amount") followed by the fair-game reading of Itô's choice. The gambler who "could peek at the outcome" is the best explanation of predictability I know at this level.
- **Itô's correction as Jensen's gap**, with $x=0.7$, $\Delta t=0.25$, landing at $0.2$ or $1.2$. The figure carries the idea and the numbers are checkable by hand.
- **The weak-order cell without sampling**: the recursion for mean and variance is exact, so the slope $1.01$ is not blurred by Monte Carlo error. This is code that teaches.
- **Standard errors are stated next to every Monte Carlo claim**, and they are right (I recomputed them, including the non-obvious $0.013$ for the variance of $\int W\,dW$).
- **Exercises** are the strongest in the chapter half I read (see below).

## Structural findings

1. **Too many confirmation cells, all introduced the same way.** Nine of thirteen cells are introduced by a question followed by "The next cell ...". Several confirm a statement that was just proved and that the simulation cannot fail to satisfy: `#sdes-brownian-paths` checks that sums of independent $\mathcal N(0,\Delta t)$ draws have variance $t$; `#sdes-brownian-covariance` checks $\min(s,t)$; `#sdes-isometry-check` checks $\tfrac12$; `#sdes-euler-check` checks that Euler-Maruyama with $g=0$ is Euler; `#sdes-ou-stationary` ends with the admission that "this experiment cannot tell the exact law from the recurrence's". The reader learns the method of checking against a standard error once; after that each such cell costs 150 to 250 words of setup and read-out. Keep one Brownian check (merge variance and covariance, or keep only the fraction inside the envelope with the remark about a fixed time), drop the Euler check and the stationary-quantile cell, and let the isometry be confirmed by the second route through Itô's lemma, which the text already gives.
2. **The recurrence is re-identified four times** (finding SDE-12) and the $\sqrt{\Delta t}$ scaling is argued three times (opening, random walk, Euler-Maruyama). Guide section 7.7 applies inside a section too.
3. **Status of statements is not visible in the typography.** "Proposition" labels five proved results, one result argued heuristically with the proof cited (Itô's lemma) and one stated with no proof at all (orders of Euler-Maruyama). The prose is honest in each case, but a reader who looks things up sees `**Proposition**` and expects `**Proof.**` (finding SDE-8).
4. **Length and scope.** The section is four topics (Brownian motion, Itô calculus, OU, simulation). It holds together because of the running example, and I would not split it. What can go with no loss: the covariance proposition and its cell if exercise 7 is kept (the body never uses $\min(s,t)$ again); the Rademacher fourth-moment excursion (finding SDE-13); the variance-exploding paragraph, reduced to one sentence; the three confirmation cells above.
5. **No pointer into the main text.** The section says "the diffusion models of the main text" and "the forward step of DDPM" but never gives a label. `chapter_diffusion-models/gaussian-paths.md` points here (to `sec_mdl-ornstein-uhlenbeck` and `sec_mdl-euler-maruyama`); the appendix should point back so a reader who starts here knows where the kernel is used.
6. **Framework tabs do nothing.** Four tabbed import cells load `torch`, `nn`, `F`, `jax`, `optax`, and so on; no later cell uses any of them. A reader of the PyTorch tab wonders when the framework will appear. If the pipeline allows it, one untagged NumPy import cell would be more honest.

## Passage-level findings

**SDE-1** (moderate, thought). Under "Noise as a Forward Process".
Anchor: `The two clouds overlap more and more but never become the same.`
The bullet list and the figure caption ("Two failures and the remedy") present noise alone as a failure. Five paragraphs later: "Noise alone also works as a forward process if it runs long enough". The second statement is the true one (variance-exploding diffusion is a standard family), so the first reads as a claim the text then withdraws. Fix: say in the bullet that noise alone forgets the start only relative to its own growing width, with no limiting law, and call the first panel the one failure. Then the variance-exploding paragraph shrinks to a sentence.

**SDE-2** (minor, pedagogy). Under "Simulation of Brownian Paths".
Anchor: `Does the simulated walk`
The cell generates $W$ as a cumulative sum of $\mathcal N(0,\Delta t)$ draws and then checks that the variance is $t$. That is a check of the random number generator. The one thing the cell teaches is the sentence after it, "far fewer paths stay inside the envelope over the whole interval", which the cell does not compute. Either compute that fraction (one line, and a surprising number) or cut the variance check.

**SDE-3** (minor, thought). Under "From Random Walks to the Wiener Process".
Anchor: `The resulting covariance is what we test to validate a simulator in`
The motivation for the covariance proposition is that it can be tested, which is circular. The real use is that a Gaussian process is determined by its mean and covariance, so $\min(s,t)$ characterizes Brownian motion; exercise 7 uses the analogous OU formula. Say that in one sentence, or move proposition and cell to an exercise.

**SDE-4** (moderate, explanation). Under "The Itô Integral".
Anchor: `The isometry below makes`
The definition of the integral for general integrands leans on a result stated one paragraph later, and the proof of that result then ends "For a general $H$, approximate it by simple predictable processes; both sides converge, so the identity extends", which uses the extension. The logic is sound (isometry for step processes, then extension by continuity, then isometry in general) but it is presented in the order definition, extension, isometry, and a careful reader sees a circle. Reorder: define the integral for step processes; prove zero mean and the isometry for them; then one paragraph that says the isometry lets the definition extend and that both properties survive the limit, with the citation. Also say what "predictable" means in plain words once ("decided before the increment is drawn"); the term is used six times and defined only inside the definition of a simple process.

**SDE-5** (minor, thought). Under "The Itô Integral".
Anchor: `A simulation that moves forward in time cannot look ahead, so simulation and`
The reason given for Itô's convention is too strong. Stratonovich equations are simulated forward in time routinely, and the section says so itself 750 lines later ("Heun's method ... converges to the Stratonovich solution"). The honest reasons are the zero mean (martingale property) and the fact that the Euler-Maruyama step is the Itô sum. Rewrite: "The Euler step evaluates the coefficient at the start of the step, which is Itô's choice, and the zero mean makes expectations easy; this book uses Itô's convention."

**SDE-6** (minor, language). Under "Drift, Diffusion, and Solutions".
Anchor: `The difference looks cosmetic, but it decides the accuracy of the simulation`
Good forward pointer, but "the last part of this section" is vague when the section is 8,000 words. Use the label `sec_mdl-sde-strong-weak`.

**SDE-7** (minor, pedagogy; mostly praise). Under "Itô's Lemma".
Anchor: `Something is missing, and the`
The failed prediction $\mathbb E[W_t^2]=0$ is the right way in. Keep. The only flaw is that the same $W^2$ computation is then done three times (the failed chain rule, the one-step walk with $\phi=x^2$, Example 1). The one-step walk and Example 1 are both needed; the opening can be cut to two sentences.

**SDE-8** (moderate, pedagogy). Under "Itô's Lemma" and "Strong and Weak Convergence".
Anchor: `**Proposition (Itô's lemma).**`
The disclaimer "The Taylor argument above is a heuristic; the proof controls the remainder terms in mean square" is correct and well placed. But the result carries the same label as the propositions that are proved in full, and so does "Proposition (orders of Euler--Maruyama)", which has no proof. For a section whose value is its honesty about level, make the status visible: "**Theorem (Itô's lemma; proof in Øksendal, ch. 4).**" and "**Theorem (orders of Euler-Maruyama; stated without proof).**", keeping "Proposition ... Proof ... $\blacksquare$" for what is proved here. The same convention should hold across the chapter.

**SDE-9** (minor, correctness; confident). Under "Itô's Lemma", Example 2.
Anchor: `Apply the lemma to`
$\phi(x)=\log x$ is not $C^2$ on $\mathbb R$, and applying the lemma assumes $X_t>0$, which is what the resulting formula is then used to show. Standard gloss, but this section is careful elsewhere. One sentence fixes it: "Assume for the moment that $X$ stays positive; the formula we obtain is positive, and substituting it back (Itô's lemma applied to the exponential) confirms that it solves the equation."

**SDE-10** (minor, explanation). Under "Itô's Lemma".
Anchor: `extra term, the local time of $W$ at zero, which this book never needs.`
A paragraph that introduces a concept in order to say it is never needed. Cut the paragraph, or keep only "The lemma needs two derivatives; $|x|$ fails at the kink."

**SDE-11** (moderate, correctness; fairly confident). Under "The Variance-Preserving Normalization".
Anchor: `This is also why data are normalized to unit`
Overstated. DDPM and most pixel-space models scale images to $[-1,1]$, which gives a variance well below one; EDM carries an explicit $\sigma_{\textrm{data}}=0.5$. Latent diffusion does rescale latents to roughly unit variance. Write what is true: "This is why the scale of the data matters: latent diffusion models rescale their latents to unit variance, and for data of another scale the marginal variance drifts from $\textrm{Var}(X_0)$ to $1$ (exercise 8c)."

**SDE-12** (moderate, language; repetition, 4 instances plus 3 of the fixed point).
Anchors: `running example of this section, and the recurrence of the opening is its` / `The opening's recurrence is this` / `This is the opening's recurrence and the loop of` / `Its simulation step, the recurrence above,`
The identification "the recurrence is Euler-Maruyama for OU with $\theta=1$, $g=\sqrt2$, $h=0.01$" is made in the introduction, after the first figure, at the start of the OU part, and again in the Euler-Maruyama subsection, each time with the arithmetic $0.99$ and $\sqrt{0.02}$. The fixed point $1.005$ is derived in the opening, derived again under "The Stationary Distribution", and cited again in the weak-order paragraph. Derive each once: the identification at the OU definition (where $\theta$ and $g$ first exist), the fixed point under "The Stationary Distribution" (drop the parenthesis in the opening or keep only "close to 1").

**SDE-13** (moderate, pedagogy). Under "The Variance-Preserving Normalization".
Anchor: `The shape of the distribution still changes. A convenient measure is the`
About 350 words, a cell and a five-column table establish that Rademacher data become Gaussian in the fourth moment while the variance stays at one. The point (variance preserved, shape not) is worth one sentence and the formula; the derivation is already exercise 8. The table's read-out is mostly about standard errors and the half-percent discretization bias, which the reader has now met three times. Cut the cell, keep "the fourth moment of $\pm1$ data climbs from $1$ to $3$ as $3-2\bar\alpha_t^2$ (exercise 8)".

**SDE-14** (minor, correctness; confident). Same subsection.
Anchor: `these data start as far from the Gaussian as unit-variance data can.`
False as stated: fourth moments of unit-variance laws are unbounded above, so heavy-tailed data are arbitrarily far. The next two sentences say the true thing (smallest possible fourth moment). Delete this sentence.

**SDE-15** (minor, thought). Under "The Euler--Maruyama Scheme".
Anchor: `DDPM's form of the step is in fact exact, not only first-order correct`
The paragraph first shows that the Euler-Maruyama step agrees with DDPM's to first order, then reveals that DDPM's step is the exact OU transition. The second fact is the important one and it reverses the framing of the introduction ("agrees with the forward step of DDPM to first order"). Lead with it: "DDPM's forward step is the exact OU transition over one interval, with $1-\beta=e^{-2\theta\Delta t}$; Euler-Maruyama is its first-order approximation ($0.99$ against $\sqrt{0.98}=0.98995$)."

**SDE-16** (minor, pedagogy). Same subsection.
Anchor: `With the noise switched off, does the helper reproduce forward Euler on`
A unit test, not a lesson; the formula shows it on sight. Cut cell and both paragraphs, keep the clause "with $g=0$ the scheme is forward Euler".

**SDE-17** (moderate, explanation). Under "Strong and Weak Convergence".
Anchor: `The orders follow from counting how the errors made in single steps add up.`
"Follow" promises more than the next two paragraphs deliver. The count treats per-step errors as independent terms that simply add, and ignores that an error made at step $n$ is propagated by the later dynamics (the proof needs a Gronwall argument for exactly that). The count gives the right exponents and is worth keeping. Say what it is: "A count of single-step errors explains the exponents; the proofs must also control how early errors propagate." The three hypotheses of the proposition (Hölder in $t$, four derivatives of polynomial growth) are more than this reader can use; one sentence "under smoothness and growth conditions given in Kloeden and Platen" would serve, with the three conclusions kept.

**SDE-18** (minor, language). Same subsection, after `#sdes-em-strong-order`.
Anchor: `but their ratio is: the gap between the two curves widens by about`
"The absolute errors ... are not comparable ... but their ratio is" does not parse on first reading. Write: "The two curves have different heights because the processes have different scales. What can be compared is how they fall: each halving of the step halves the OU error and divides the GBM error by $\sqrt2$."

**SDE-19** (minor, pedagogy). Under "The Stationary Distribution".
Anchor: `so this experiment cannot tell the exact law from the recurrence's.`
An experiment whose stated outcome is that it cannot see the effect under discussion. The closed-form fixed point already made the point. Cut the cell, or keep only the quantile comparison and drop the variance line.

**SDE-20** (minor, language). End of body.
Anchor: `We can now simulate any SDE with a`
The closing paragraph lists what was covered and the Summary lists it again 20 lines later. Guide section 12.2 asks for no recap before the Summary and names "We can now ..." as the formula to avoid; the Fokker-Planck section closes the same way. Keep only the last three sentences (non-Gaussian data, no closed form, the next section).

## Exercises

Twelve problems, within the band, and the best feature of the section. They extend the text instead of repeating it: the biased walk that produces a drift (1), total variation and the derivation of $\int W\,dW$ without Itô's lemma (2c, a gem), why the diffusion coefficient can be read off one path and the drift cannot (3c), mean against median for geometric Brownian motion (4), the Lamperti transform (5b), Heun converging to the Stratonovich solution (9), exact transitions and exponential integrators (10). I checked the stated results of 1, 2, 3c, 4, 8, 9, 10b, 11a, 11c and 12 and found them correct.

- Exercise 11 (multilevel Monte Carlo, five parts, about 450 words) is a project, marked as such. It is good, but it is the only place the body's remark about strong accuracy is cashed in; consider whether a mathematics appendix for deep learning needs it. If the exercise count has to drop, this is the one.
- Exercise 12 sits last but belongs with the Brownian-motion problems (after 2).
- Exercise 6 and exercise 5 overlap slightly in purpose (orders under multiplicative noise); both are worth keeping.

## Still open from October

The October review said this section "teaches well after trimming". The trimming of mannerisms is done (no em dashes, no slogans). What remains open is length through repetition and the uniform cell template, as above.

## For the chapter review

- **Fit.** The section assumes Euler, Lipschitz existence and the decay ODE from the ODE section (all referenced by label) and variances of sums from the probability chapter. It sets up, by name, everything the Fokker-Planck section needs: Itô's lemma in $d$ dimensions, the zero mean of the Itô integral, the OU kernel with $\alpha_t$, $\sigma_t$, and Euler-Maruyama.
- **Overlap with the next section.** The Fokker-Planck section re-simulates the OU cloud from $x_0=2$ and restates the kernel under a second equation label. One of the two experiments should go (see that review).
- **Pattern to check chapter-wide.** (a) The "question, *The next cell*, read-out with standard error" template: it is rigorous and it is everywhere. (b) "Proposition" used for proved, heuristic and cited results alike. (c) Body closings of the form "We can now ...". (d) Framework tabs on sections whose code is pure NumPy.
- **Level.** Honest. This reader can follow every proof given; what is beyond them is named and cited. The section could serve as the model for how the part treats material above its level.
- **Symbols.** $\theta$ is the OU rate here and the network parameters in the neural ODE section; $h$ is the step size here and a scalar field in the next section.
