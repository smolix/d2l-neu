# Bayesian Computation

Source: `chapter_mdl-probability-statistics/mdl-bayesian-computation.md`, read to the slides marker (line 1631). About 6,800 words of prose before the exercises and 9,400 with them; 17 code cells (PyTorch tab), 6 generated figures, 12 exercises. Anchors are quoted between « » and were checked against the source with whitespace normalized.

## Verdict

This is the strongest section I read in the chapter and in places it reaches the bar. It has one idea, stated in the opening and returned to in the summary (every method works with a distribution known up to a constant because it uses only ratios, in which the constant cancels); one running posterior small enough to check on a grid; a smallest worked case before each method (Gamma(4,1), the coin, a two-state chain, AR(1)); and a habit of predicting a number and then running the code to see it. Several passages are of the kind a reader remembers: importance weights that look healthy while their variance is infinite, an ESS of 23 out of 24 on an estimate that is wrong, a chain that accepts 97% of its proposals and is worth a few dozen draws. I recomputed the derivations and worked numbers and simulated the running example over 1,500 data sets; I found no mathematical error, and the hedged ranges in the prose are accurate. The most important thing to fix is that the committed output shows an unrepresentative data set: the page prints a predictive probability 1.7 times the plug-in where the opening promises "several times" and the text says "typically three to ten times", and the true value lies outside the printed 90% range that the text says contains it nine times in ten. The data set is the section's running example, not a Monte Carlo quantity, and should be fixed. The second thing is the frame: the section motivates itself by the posterior of a trained network, says three times that none of its methods works there, and does not tell the reader that the book goes on to use the ELBO, importance weights and Langevin proposals for other purposes.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **A unifying idea that is actually used.** «we can average over a distribution known only up to a constant, because each method uses only ratios of it, and the constant cancels in a ratio.» Each method's subsection says where the evidence cancels («The evidence appears only in the acceptance rate, never in the decision to keep a draw.»), and the Summary lists the four cancelling quantities. This is the section's spine and must survive revision intact.
- **A checkable target.** Two parameters, a grid as the reference, and every later method compared against it. The MCMC check «tests the error bar against the truth», and the text says why that is the strongest check in the section.
- **Smallest cases first.** Gamma(4,1) for Laplace (mode 3, variance 3, 4.2% of the mass below zero); the two-state chain with target $(\tfrac14,\tfrac34)$, where «If we counted only accepted states, the sequence would alternate 2, 1, 2, 1 and give $(\tfrac12,\tfrac12)$»; AR(1) for the autocorrelation time. Each is small enough to verify by hand, and each makes a point the general statement would not.
- **Predict, then run.** The proposition on the mean-field fit predicts standard deviations of $\sqrt{1-\rho^2}$ times the marginal ones; the cell prints 0.49 and 0.214 against a predicted 0.494 and 0.216. The coin's acceptance rate of 0.305 and ESS limit of 0.42 are computed in closed form and then measured. This is the book's best method, used consistently.
- **Honest diagnostics.** The paragraph on the Laplace proposal («The Laplace proposal has a defect that the printed ESS does not show.») is the best in the section: a proposal with 92% ESS whose weights have infinite variance, the reason in one line (the posterior's tails are the prior's), and the conclusion «A single run cannot reveal the defect; repeated runs, or the analysis of the tails, can.» I confirmed it by simulation: about 2% of runs fall below an ESS of 40%, and in those runs one draw near 4.5 to 5.5 Laplace standard deviations carries the weight.
- **Sentences that judge.** «The one step that is not routine is multiplying $\pi g$ into the minimum»; «The acceptance rate is the number a sampler reports most readily, and on its own it says little.»; «A converged ELBO shows that the optimization within the family has stopped; it does not show that the family fits the posterior.» This is the voice the guide asks for.
- **Figures that carry an argument.** The importance-sampling pair (ESS 23 of 24, wrong; 17 of 24, right) and the mean-field ellipses do work the text could not do in a paragraph.
- **Exercises** that extend the section into real territory (Barker against Metropolis on the two-state chain, where Metropolis comes out better than independent sampling; the tails of the logistic posterior; the two-mode posterior tied to a network's sign symmetries). I worked 1c, 4b, 5a and 6a and checked 8b by simulation; all are correct.

## Structural findings

**S1. The committed output is an atypical data set, and the page contradicts the prose (major; needs a code change, so the writing pass can only soften it).** The data are drawn afresh on every run. I simulated the model cell 1,500 times. The prose describes the typical run accurately: plug-in probability at $x_\star=-3$ with median 0.004 ("one in a few hundred"), ratio of predictive to plug-in with median 4.9 and a 5% to 95% range of 2.2 to 11.4, separable data in 3.0% of runs ("about one run in thirty"), the true value inside the 90% range in 90% of runs. The stored PyTorch output is not a typical run:

| Quantity | Prose | Stored output | Where it falls in 1,500 simulated runs |
|---|---|---|---|
| plug-in at $x_\star=-3$ | "about one in a few hundred" | 0.0960 | above the 95th percentile (0.057) |
| predictive / plug-in at $-3$ | "several times", "typically three to ten times" | 1.7 | below the 5th percentile (2.2) |
| Laplace sd | "about 1 for the intercept and 0.5 for the slope" | 0.826, 0.357 | below the 5th percentile (0.84, 0.39) |
| true value 0.0006 in the 90% range | "in about nine runs out of ten" | range [0.0032, 0.6323], so no | the one-in-ten case |

So a reader of the printed page meets a headline claim ("several times larger") and an output that shows a factor of 1.7, with the fitted line $(-0.66, 0.53)$ far from the generating $(-3, 1.5)$. The author's decision that ranges are deliberate covers Monte Carlo noise. It does not fit here, because the data set is the object of study: every number in the section is a property of this one posterior. Fix the twenty points (seed only the data draw, or list them), keep the samplers unseeded, and let the prose quote this posterior's numbers where they are properties of the data (MAP, grid mean, the predictive at three inputs) and ranges where they are Monte Carlo (ESS, acceptance, $\widehat R$). Until the code can change, add one sentence after the predictive cell saying what an unlucky draw looks like (a shallow fitted slope and a small ratio), so the reader is not left to doubt the text.

**S2. Hedges about runs crowd out statements about the posterior (moderate; follows from S1).** Because the data change, the prose keeps describing the distribution of runs: «about one run in thirty», «In nearly every run», «the order flips in about one run in a thousand», «in most runs», «in about nine runs out of ten», «in every run» (twice), «In a few runs out of a hundred», «in a typical run». These are accurate, and a few are valuable (the infinite-variance paragraph needs its "few runs out of a hundred"). Most are noise a fixed data set would remove, and one-in-a-thousand is a precision no reader can use.

**S3. The frame promises Bayesian prediction for trained networks and delivers a dead end (major for the brief's "connection to deep learning").** The section opens with «Every model in the Basics part was trained to a single parameter vector» and then says three times that the methods do not reach such models: «none scales cleanly to the posterior of a neural network» (opening), «None of the four scales cleanly to the posterior» (closing), «networks none of them is yet satisfactory» (summary). That is honest, and the candour is welcome. But the deep learning reader is left asking why they read 7,000 words. The book has good answers and the section mentions them only in passing: the ELBO trains variational autoencoders and is the objective of the diffusion chapters (which cite `sec_mdl-latent-em-elbo`); importance weights and their ESS return in off-policy reinforcement learning; Langevin proposals are the sampler of score-based models. Rewrite the last paragraph of the opening and the closing paragraph to say this once, with labels: the methods are taught on a posterior because it is the cleanest target known only up to a constant, and each one returns later in the book for a different target. State the "does not scale" verdict once, in the comparison section, with the four reasons that are already there.

**S4. A check is claimed for every method, and two are not shown (moderate).** The table gives variational inference the diagnostic "ELBO, comparison with samples"; the closing lists «(the weight ESS, $\widehat R$ with $\kappa$ and the MCSE, the ELBO)»; the summary says «Each comes with a check: the weight ESS,» and again names the ELBO. The text never computes an ELBO and never uses it to diagnose anything (it appears as a number only in exercise 8c). The text's own sentence, that a converged ELBO does not show the family fits, argues it is not a check of the approximation at all. And the summary's "each" covers four methods with three checks: Laplace has none there. Either print the ELBO of the Laplace and mean-field Gaussians beside the grid's log evidence (the gap is the KL divergence, which would make the bound do visible work), or stop calling the ELBO a check and say that Laplace and variational fits are checked against samples.

**S5. The comparison does not help the reader choose (moderate).** «Choosing among them starts from the quantity we need: a posterior mean, a predictive probability, or the probability of a rare event.» The sentence starts a decision and does not finish it. The section label is `sec_mdl-bayes-decision-map`, and the section has earned a decision: for a few parameters, a grid; a unimodal smooth posterior and a need for speed, Laplace as a first answer and as a proposal; an expectation with an error bar, MCMC with the three diagnostics; many posteriors to fit or a posterior inside a training loop, variational inference, knowing its widths are too small. Three sentences of this kind would do more than the table's "characteristic failure" column alone.

**S6. Length and what could go.** The section is proportionate and I would cut little. Candidates, in order: the per-framework autodiff check of the gradient and Hessian (four tabs to confirm a two-line derivative to rounding error; the stored output is two numbers of order $10^{-15}$ that the prose reads in one sentence); the second small Laplace case (the coin), which gives away exercise 4a (BC-6); the Geyer truncation paragraph, which could be a comment in the code. Rejection sampling earns its 55 lines because it introduces the ratio $\widetilde p/q$ and gives the lower bound on the ESS.

## Passage-level findings

**BC-1** (moderate, explanation). Opening, second paragraph.
- Anchor: «a probability of about one in a few hundred for a typical draw of the data»
- "For a typical draw of the data" arrives before the reader knows the data are random on each run, and "one in a few hundred" is vaguer than an opening example should be. With S1 fixed this becomes two numbers: "the fitted model gives 0.004; the average over plausible vectors gives 0.019". Until then, say "about 0.004" and "about five times larger".

**BC-2** (minor, language). Opening.
- Anchor: «Each answer repairs a failure of the one before, and»
- True of Laplace to importance sampling and of importance sampling to MCMC. Variational inference does not repair a failure of MCMC; it trades the Monte Carlo error for an approximation error, as the first sentence of its own part says. Write "Each of the first three repairs a failure of the one before; the fourth replaces sampling by optimization."

**BC-3** (minor, thought; repetition). Under "The Gradient and the Hessian" and "The Posterior Mode and the Laplace Approximation".
- Anchors: «The maximum is finite even when the data are linearly separable»; «The MAP is finite in every run, separable data included.»
- Said twice, 180 lines apart, the second time as a reading of an output that cannot show it. Keep the first.

**BC-4** (minor, pedagogy). Under "The Gradient and the Hessian".
- Anchor: «The differences are at the level of rounding error.»
- The cell exists in four framework versions (the MXNet one checks only the gradient, the JAX one compares relative errors because of single precision), and its whole result is this sentence. It is the one place where tabs fragment the section. If it stays, it should say what it is for in the words of the next sentence, which is the real point: for a larger model one writes only $\log\widetilde p$. Consider keeping one sentence and moving the check to an exercise.

**BC-5** (minor, explanation). Under "The Posterior Mode and the Laplace Approximation".
- Anchor: «The Gaussian matches the posterior's location, slope and curvature»
- At a mode the slope is zero by construction, so "matches the slope" tells the reader nothing; the figure's label repeats it. "Matches the height (up to the constant), the position and the curvature of the peak" is what is meant.

**BC-6** (minor, pedagogy). Under "The Posterior Mode and the Laplace Approximation".
- Anchor: «A second small case is a coin that shows 9 heads and 4 tails.»
- The paragraph gives the mode, the curvature 61, the Laplace Gaussian $\mathcal N(0.692,0.128^2)$ and the exact mean and standard deviation, then points to exercise 4, whose part (a) asks for the mean and standard deviation of that Laplace approximation. Either drop the paragraph (Gamma(4,1) already makes the point, with a figure) or keep it and start the exercise at the interval and the mass outside $[0,1]$. If it is kept, say that the curvature is the same 61 that gave the coin's error bar in the previous section: the Laplace standard deviation under a flat prior is the MLE's standard error, and that link between the two sections is worth one sentence.

**BC-7** (moderate, explanation). Under "The Plug-in and the Posterior Predictive".
- Anchor: «which grows quadratically with the distance from the data, while the shift»
- This is the hardest paragraph pair in the section. Two effects are named, their directions are given, and the argument that the second wins compares a variance (quadratic in $x_\star$) with a shift of a mean (linear) without saying how a variance of the logit turns into a change of the prediction. One line supplies it: for a small probability $\operatorname{sigmoid}(a)\approx e^{a}$, and the mean of $e^a$ for a Gaussian logit with mean $\mu$ and variance $s^2$ is $e^{\mu+s^2/2}$, so averaging raises the log-odds by about half the variance of the logit, until the sigmoid's flattening caps it. Then "quadratic against linear" is a comparison of like with like, and the reader can estimate the factor from the printed Laplace covariance (a rough estimate only: the posterior's tails are not Gaussian, and it overshoots when the probability is not small). The sentence before it, «On the far side of the data the plausible vectors with a shallow slope give probabilities that are large compared with the typical one, and they dominate the average.», is the same point in words and is good; put the formula next to it.

**BC-8** (minor, explanation). Under "The Plug-in and the Posterior Predictive".
- Anchor: «It is exact whenever the parameters are themselves drawn»
- "Exact" has no referent. What is meant: if nature draws $\boldsymbol\theta$ from the prior and the data from the model, then among all such worlds that produce these data, the label at $\mathbf x_\star$ is 1 with exactly this probability. Say that, and say in a clause that the true line here, $(-3,1.5)$, has a slope the prior $\mathcal N(0,4)$ finds unremarkable, so the mismatch at $x_\star=-3$ is the price of twenty points and not a failure of the method.

**BC-9** (minor, language; pattern). Code lead-ins.
- Anchors: «The cell runs ten Newton steps from zero and inverts the negative Hessian at»; «The cell checks the coin's numbers.»; «The cell computes both on»; «The cell implements the rule with a Gaussian step»
- Eleven lead-ins begin "The cell ...". The better ones in this section pose the question the cell answers (the proposal-scale cell, the mean-field prediction). Convert the weakest half to questions or predictions. Related: "The smallest case" opens three examples; it is a good device and three is the limit.

**BC-10** (minor, explanation). Under "The Effective Sample Size".
- Anchor: «It estimates the number of independent posterior draws that the»
- Stated as a fact about the weighted sample. It is a heuristic for a typical bounded function, and the section itself shows two ways it misleads (the figure, and the Laplace proposal). Add "roughly" and a forward clause ("we shall see two ways in which it can mislead"), so the later passages read as the fulfilment of a warning and not as a retraction.

**BC-11** (minor, thought). Under "Proposals for the Logistic Posterior".
- Anchor: «is then infinite as soon as one eigenvalue»
- Correct (I checked the tail integral), and the text reports that the smallest eigenvalue is a few hundredths. It is worth saying that in the stored run both eigenvalues (about 0.04 and 0.77) are below 2, so the weights have infinite variance in every direction, not along one unlucky axis. The reader otherwise pictures a single thin direction.

**BC-12** (minor, explanation). Under "Markov Chain Monte Carlo", the definition of stationarity.
- Anchor: «If the states are distributed as $\pi$, they still are after one step.»
- This sentence is the definition in words and should come before the display, not after it. The same holds for the warm-up sentence, which arrives before the reader has seen a chain; it belongs with the first run, where 2,000 states are discarded.

**BC-13** (minor, explanation). Under "Diagnostics for Markov Chain Mixing".
- Anchor: «If the chains agree, their means differ only by Monte Carlo noise, $B/N$ is»
- The formula for $\widehat R$ is given and then explained by its limiting cases, which is fine, but nothing says what the numerator is: an estimate of the posterior variance that overstates it while the chains still disagree, set against the within-chain variance that understates it. With that sentence $\widehat R$ is a ratio of two estimates of the same quantity, and the reader sees why it approaches 1 from above.

**BC-14** (moderate, explanation). Under "The Evidence Lower Bound".
- Anchor: «This is the ELBO of the latent-variable models of»
- The previous section derived the ELBO by Jensen's inequality for a latent $z$ and proved that the gap is a KL divergence. Here it is derived again, by a different route, for $\boldsymbol\theta$, and only afterwards does this sentence say it is the same bound. Reverse the order: open the subsection with the identification (the parameters play the role of the latent, the data set the role of the observation), cite the gap identity by its label, and give the three-line derivation only if it is wanted for readers who arrive by lookup. If the ELBO gets a section of its own (see the Maximum Likelihood review, S1), this subsection shrinks to a paragraph.

**BC-15** (minor, thought). Under "The Direction of the KL Divergence".
- Anchor: «so $q$ retreats from regions the posterior does not»
- A good paragraph, and the figure is right. One thing is missing for a reader who wonders why anyone uses the reverse direction if it is the over-confident one: the forward divergence needs expectations under the posterior, which is the thing we cannot sample. One clause.

**BC-16** (minor, thought; repetition, guide §7.7 and §12.2). End of "Comparison of the Methods".
- Anchor: «We can now estimate any posterior expectation of a small model in four ways.»
- The closing paragraph restates the checks and the failure modes that the table above it has just listed and that the Summary below it lists again; "We can now ..." is the fixed closing formula used by eight of this chapter's nine sections. Keep the hand-over to the next section (the last sentence) and the scaling verdict (once, see S3); cut the rest.

## Exercises

Twelve, within the appendix band, graded from one dot to two, and unusually good: most ask for a prediction and a measurement. Notes:

- **Exercise 4a** is answered in the text (BC-6).
- **Exercise 8b** says «The Laplace value comes out a few percent low in every run.» in a problem that asks for $\log p(\mathcal D)$. I checked on 300 data sets: the Laplace estimate of the evidence $p(\mathcal D)$ is 3% to 5% low (never high), which is about 0.04 nats in $\log p(\mathcal D)\approx-12$. Say "the Laplace evidence" or "about 0.04 nats", since a few percent of the logarithm would be ten times larger.
- **Exercises 2, 3, 6, 7, 10, 11 and 12** depend on rerunning the notebook cells. That is right for this section, but they assume the reader's draw resembles the text's; with a fixed data set (S1) their expected answers become definite.
- **Exercise 10** (tails) and **exercise 9** (two modes and network symmetries) carry the two ideas a deep learning reader most needs from the section. The conclusion of exercise 9d, how many copies of a mode a network's posterior has, is used in the closing paragraph of the text by reference to the exercise; it deserves a sentence of its own in the body.
- A problem on what to do when the grid is unavailable is exercise 11, marked extended. Good as it is.
- Missing: nothing essential. Hamiltonian Monte Carlo is in the Further Reading notebook, and MALA (exercise 12) is the right bridge to it.

## Still open from October

October listed the opening question "How do you average over a distribution you can only evaluate pointwise, up to a constant?" as a sentence to keep. That exact sentence is gone, and its idea is now stated as the claim about ratios in the opening and carried through the section, which is better. October had no other finding specific to this section.

## For the chapter review

- **Order and fit.** Correctly placed after Maximum Likelihood, which supplies MAP, the prior, the Beta posterior and the ELBO. The hand-over out («the next section asks how far such an average can be trusted») is a real link to concentration inequalities. It leans on chapter 39 for the reparameterization gradient and on chapter 43 for Langevin dynamics, both cited by label.
- **Overlap with its neighbour.** The ELBO and its gap are derived in both sections (BC-14); the posterior, the evidence and MAP are defined in both (acceptable for a lookup reader, and short here); the previous section's "Posterior Mode and Posterior Distribution" subsection says in advance what this section's opening says better.
- **The section shows the method the whole part should use**: a target small enough to check exactly, a closed-form prediction before each cell, and diagnostics tested against the truth. Where other sections of the chapter run a cell and narrate it, this one predicts and compares.
- **The unseeded running data set is a risk specific to this section** (S1). Elsewhere in the chapter unseeded cells produce Monte Carlo quantities for which a range is the right report. Here the data define the object being studied. The chapter review should separate the two cases.
- **Patterns shared with the chapter:** the "Every model in ..." opening, the "We can now ..." closing paragraph followed by a Summary that repeats it, and "The cell ..." as the standard lead-in to code.
- **Scope.** The deep learning reader needs this section for the ELBO, importance weights and the ESS, detailed balance, and the forward/reverse KL distinction, more than for Bayesian logistic regression. Saying so (S3) would also answer the question of whether the section belongs in the book: it does.
