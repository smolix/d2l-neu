# Trust Regions and Proximal Policy Optimization

Source: `chapter_deep-reinforcement-learning/ppo.md` (lines 1 to 830; about 5,200 words of prose, 2,150 words of exercises, 28 cells). Stored outputs read: PyTorch and JAX. Figures rendered: `mdl-rl-trust-region.svg`, `mdl-rl-ppo-clip.svg`, and the stored reuse and clip-fraction plots.

## Verdict

The first half of this section (parameter space against policy space, importance sampling, the exploding product, the performance difference lemma, the monotonic-improvement bound, the clipped objective) is the best exposition of the trust-region argument I know at this length. Every step has a small number attached, the proofs are complete and correct, and the text is exact about what the clip does not guarantee. I checked the lemma, the occupancy form, the constant of the bound, the variance of the product and the coupling exercise and found no error. The second half, "PPO in Practice", is weaker: it is long, its headline experiment is read for a number the stored plots do not show (four epochs "roughly 25,000" steps to the ceiling; the plots put the four-epoch median at the ceiling near 38,000 and the twenty-epoch median near 26,000), the gain is attributed to reuse without the control that would separate reuse from a larger effective step size, and the implementation details that the literature found to matter most are left to the chapter introduction and an exercise. Two prerequisites are missing: the KL divergence is used throughout and never defined in the main text, and the step from importance-weighted rewards to ratio-weighted advantages is skipped. The single most important fix is the reuse experiment: print the steps to the ceiling, add the one-epoch control at a larger learning rate, and state the result the numbers support.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 4 |

Explanation would be a 5 for the first half alone.

## What works

- **The two-action example** ("Take two parameter updates of the same size") with 0.50 to 0.88 against 0.9975 to 0.9997, and its second use to explain why saturation is hard to undo (a score of 0.0025 for the frequent action, the other action "sampled once in four hundred steps"). The example returns in the ablation and in exercise 1. This is a running example doing real work. Keep every word.
- **"The Exploding Product"**: 1.2 or 0.8 per step, mean square 1.04, variance about 0.5, 50 and 3 times 10^8 at 10, 100 and 500 steps (all correct). A reader will remember this.
- **The lemma and its use.** The proof of the performance difference lemma is four sentences and complete; the occupancy form follows; then the sentence "that substitution is its only error" tells the reader exactly what the surrogate is. The bound is stated with its constant evaluated ("about 40,000 times" the largest advantage) and judged: "The constant is too large to use as it stands." That is an author leading.
- **Honesty about the clip.** "It does not clip the ratios themselves", "The clip is therefore an incentive rather than a constraint, and unlike" the bound "it guarantees no improvement", and later "it bounds the path, not the destination". These are the three things most accounts get wrong, and the section gets all three right.
- **The ablation is designed to show something.** A learning rate ten times the usual one makes the clip matter, the text says so, and it reports what happens at the usual rate ("The large rate makes training fast, and the clip makes the large rate safe."). The reader learns when the mechanism is needed, which is rarer than a demonstration that it works.
- **The within-batch reading of the clip fraction** (zero at the first epoch, a peak near the fifth, then a fall explained by the one-sided minimum) is a good piece of diagnostic teaching, and the stored plots show it in both frameworks.

## Structural findings

1. **"PPO in Practice" is half the section and has five jobs**: the implementation, the reuse experiment, the ablation, diagnostics (clip fraction, approximate KL, entropy, ratio histogram, effective sample size, a harmonic-mean check, a greedy evaluation) and vectorized collection. The effective-sample-size thread (from "The summary fractions hide how the ratios are distributed" to the harmonic-mean cell, about 500 words and five cells including `action_probs`) is a side study: the quantity is not a standard PPO diagnostic, its derivation interrupts the section, and exercise 4 already owns the theory. Direction: keep the histogram and one sentence on what it shows; move the effective sample size and the harmonic mean to exercise 4.
2. **What practitioners must know is not in the text.** The reader is told about vectorized environments, minibatches and the autoreset step. The details that controlled studies found to drive PPO's results are elsewhere or absent: the finding of the cited study that code-level choices account for much of PPO's edge over TRPO appears only in the chapter introduction and in exercise 10d; per-minibatch advantage normalization, gradient-norm clipping at 0.5, learning-rate annealing, orthogonal initialization with a small policy-head gain, observation and reward normalization and value-loss clipping are not named. A reader who builds PPO from this section and finds it worse than a library will not know why. Direction: one paragraph in "Vectorized Collection and Minibatches" that lists these five or six choices, says which our loop has (standardized advantages, separate networks and optimizers) and which it lacks, and states the finding. Also say that the previous section's gradient cap is dropped here and why, since standard PPO keeps it.
3. **The KL divergence arrives undefined.** It first appears inside `eq_trpo_bound`, is named in the next sentence, and is never defined or referenced. To my search the main text of the Basics part does not define it either (the classification chapter points to the appendix). The reader is not expected to have read the appendix. A definition and one sentence of reading (expected log ratio under the old policy, zero only when the two distributions agree) is a small result and should be duplicated here, at the first use, with a pointer to the appendix chapter for more.
4. **Two endings.** The paragraph beginning "PPO lets an agent take several steps on each batch" sits at the end of the subsection on vectorized collection and recaps the section before the Summary. Its last two sentences are a good bridge to the next section; the first two repeat the Summary. Keep the bridge only.
5. **Order within the first half is right.** Problem, exact tool, why the exact tool fails, the approximation, what the approximation costs, the bound, the practical method. Do not reorder.

## Passage-level findings

**PPO-1** (minor, language). Heading: section opening.
Quote: "In 2015 John Schulman and colleagues made that change the thing to control."
"The thing to control" is limp after a strong paragraph. Suggested: "In 2015 John Schulman and colleagues proposed to limit that change directly."

**PPO-2** (minor, correctness, unverified). Heading: section opening.
Quote: "PPO became the default policy-gradient method, from simulated robots to the fine-tuning of language models on human feedback"
True of 2017 to 2023. For a book dated 2026 to 2027 the language-model half needs a tense or a qualifier: preference fine-tuning has largely moved to methods without a learned critic or without sampling (the chapter's own last section teaches one). I believe this is right but cannot cite a survey; "became, and for robot control remains," would be safe.

**PPO-3** (moderate, explanation). Heading: "The Per-Step Surrogate".
Quote: "With an advantage estimate $\hat{A}_t^i$ for step $t$ of trajectory $i$, the resulting *surrogate objective* averages over all"
The paragraph starts from the exact correction for the reward at step $t$ (ratios up to $t$), drops the earlier ratios, and then, without comment, replaces the reward by an advantage. Those are two different moves. Weighting the reward $r_t$ by the single ratio $\rho_t$ would be wrong, since the action at $t$ changes all later rewards. The quantity that one ratio can legitimately weight is the action's advantage, and the justification is the lemma of the next subsection. A careful reader stalls here. Fix: one bridging sentence, for example: "The action at step $t$ affects every later reward, and the advantage $A^{\textrm{old}}(s_t, a_t)$ sums up that effect, so it is the advantage that the ratio $\rho_t$ should weight; the lemma below shows that this is exact up to the change in the state distribution."

**PPO-4** (minor, explanation). Heading: "The Per-Step Surrogate".
Quote: "so the gradient of $\hat{L}$ there points in the same direction as the estimator of :numref:`sec_baselines`"
It is equal to that estimator (with advantages as weights and the same normalization), not merely parallel. Say "equals".

**PPO-5** (minor, explanation). Heading: "The Performance Difference Lemma".
Quote: "Our CartPole runs bootstrap at the 500-step time limit, as in :numref:`sec_deeprl`, so the objective they optimize is the continuing one."
This sentence stands between the definition of the advantage and the statement of the lemma and concerns the experiments three pages later. Move it to "The Implementation".

**PPO-6** (major, explanation; dependency). Heading: "The Monotonic-Improvement Bound".
Quote: "The surrogate is right up to a penalty proportional to the largest Kullback--Leibler divergence between the old and new action distributions at any state."
First use of the divergence, no definition (structural finding 3). Also note the direction: the bound and the TRPO constraint use the divergence from the old policy to the new one, and the approximate KL of the code estimates the same direction; say once that this is the direction used throughout the section, since the next section uses the opposite order of arguments against its reference.

**PPO-7** (moderate, correctness of statement, confident). Heading: "The Monotonic-Improvement Bound".
Quote: "so any step that raises the right-hand side raises the true return: maximizing the bound gives monotonic improvement"
Correct for the population surrogate with the exact advantages of the old policy, which is what the proposition states. The text moves in the next paragraph to the sampled surrogate with estimated advantages, where nothing is guaranteed. Add the clause "with exact advantages and the exact maximum over states; with a batch and a learned critic the guarantee is lost and what remains is the design". The text already says this of the clip; it should say it of TRPO too, since TRPO's mean constraint and fixed budget also leave the theorem.

**PPO-8** (minor, explanation). Heading: "The Monotonic-Improvement Bound".
Quote: "It solves this problem with a second-order method, which exercise :exref:`ex_ppo-trpo` builds from the pieces of this section."
One sentence on what the method is would let the reader place TRPO: the step direction is the gradient preconditioned by the inverse of the Hessian of the divergence (the natural gradient), found by conjugate gradient and followed by a line search. The figure caption already mentions the Hessian $F$; the text never does. Without this the phrase "replaces the constrained second-order problem by a first-order objective" in the next subsection has nothing to stand on.

**PPO-9** (minor, explanation). Heading: "The Clipped Objective".
Quote: "The minimum makes the limit one-sided: a change that lowers the objective is always counted, while further movement in the favorable direction adds nothing."
Good. Add the two facts that complete the picture and cost one sentence: the clipped objective never exceeds the unclipped surrogate (it is a pessimistic bound on it), and at the old parameters the two have the same value and gradient, so the first pass over a batch is an ordinary policy-gradient step. The second fact is used later ("With one pass every ratio equals one and the clip never acts") without having been stated.

**PPO-10** (moderate, correctness, fairly confident). Heading: "The Clipped Objective".
Quote: "measured that PPO's ratios do leave the band and that its divergence per update is not held below any fixed budget"
My recollection of that study is that its trust-region measurements concern the mean KL divergence (and the maximum ratio) of PPO with and without the code-level optimizations, and that its headline result is that those optimizations, not the clip, determine both the reward and how well the trust region is held. The sentence is consistent with that but cites the paper for its minor finding and omits the major one, which a reader of this section needs (structural finding 2). Unverified in detail; check the paper before rewording.

**PPO-11** (minor, thought). Heading: "The Entropy Bonus".
Quote: "Implementations often add a small *entropy bonus*"
The PPO paper's own objective includes the entropy term, so this is part of the method as published, not an implementation habit. The subsection is four sentences under its own heading inside "Trust Regions", where it does not belong; fold it into "The Implementation" where `entropy_coef` is introduced.

**PPO-12** (minor, correctness, confident). Heading: "The Advantage Estimate".
Quote: "`train_ac` of :numref:`sec_actorcritic` used the opposite order"
No function of that name exists in the previous section; the loop is `train` with the weight function `td_weight`. Same paragraph: "so the order changes no result of this section" is a claim no cell supports; either drop it or make it an exercise.

**PPO-13** (minor, thought). Heading: "The Advantage Estimate".
Quote: "That section found the error of this estimate smallest near $\lambda = 0.9$ to $0.95$"
This inherits the overreach flagged in the review of the previous section (the two frameworks disagree on the frozen-policy probe, and training runs do not separate 0.9, 0.95 and 1). "We use 0.95, the setting of the PPO paper" is enough.

**PPO-14** (major, correctness of interpretation). Heading: "Reuse against Environment Steps".
Quote: "Reusing each batch for four epochs reaches the ceiling in about half the environment steps that one epoch needs, roughly 25,000 against 50,000 or more"
No cell prints these numbers; they are read off the plot, and the stored plots do not give them. PyTorch: the four-epoch median is near 470 at about 27,000 steps and reaches 500 near 38,000; the twenty-epoch median reaches 500 near 28,000; the one-epoch median is at about 455 when the budget ends. JAX: four epochs touch about 480 near 27,000, dip, and sit at 500 from about 38,000; twenty epochs reach 500 near 25,000. So "roughly 25,000" describes the twenty-epoch arm, or the four-epoch arm at a loose threshold. The one-epoch arm never reaches the ceiling inside the budget, so a ratio of one half is not measured at all. The Summary and the closing paragraph repeat "four epochs reach the ceiling in half the environment steps of one", and exercise 8 quotes "about 25,000" as the reference value for the reader's own implementation. Fix: print `steps_to(run, 475, 5)` per arm and seed (the function was saved in the previous section for exactly this), extend the budget so that the one-epoch arm finishes, and quote the printed medians with their ranges.

**PPO-15** (moderate, correctness of interpretation). Heading: "Reuse against Environment Steps".
Quote: "Going on to twenty epochs changes little by comparison, and no clipped run collapses."
In both stored plots the twenty-epoch median reaches 500 about 10,000 steps before the four-epoch median, a difference as large as a quarter of the four-epoch figure. With eight seeds and overlapping bands I would not claim that twenty is better either; the supportable sentence is that both reuse arms are well ahead of one epoch and are not separated from each other.

**PPO-16** (major, thought). Heading: "Reuse against Environment Steps".
Quote: "The gain comes from the batch: a batch of a few thousand steps holds more information than one gradient step can extract"
The experiment does not separate this explanation from a simpler one. Adam's steps are roughly scale-free, so four passes at a learning rate of 0.01 move the parameters about as far as one pass at 0.04. Without a one-epoch arm at a four times larger learning rate (and perhaps a twenty times larger one), the result may only say that the one-epoch arm was run with too small a step. This is the "untuned baseline" problem. Fix: add that arm. If it matches four epochs, the honest lesson is different and more interesting (the clip lets several small steps replace one large blind one); if it collapses or lags, the present sentence is earned.

**PPO-17** (moderate, correctness of interpretation). Heading: "The Clipping Ablation".
Quote: "about one run in twenty holds the task and about half end dead"
The notebook's own eight seeds print "0 of 8 seeds hold the task, 4 end dead" in PyTorch and "1 of 8 seeds hold the task, 1 end dead" in JAX. A reader on the JAX tab sees one dead run in eight directly above a sentence that says about half. The 64-seed statement is allowed under the house rule, but the text should then also say what the eight seeds on the page show and that they vary ("in the eight runs shown, between one and four end dead"). The numbers that follow in the same paragraph ("about 35,000 to 40,000 and about 85,000 to 90,000 steps to the ceiling, against about 24,000 here", "on any of ten seeds") come from runs the reader cannot see or reproduce from the section. That is a lot of unseen evidence for one paragraph; turn the learning-rate comparison into a short printed cell or an exercise with these numbers as the expected result.

**PPO-18** (minor, explanation). Heading: "The Clipping Ablation".
Quote: "its variant with neither a clip nor a penalty was the worst of all those it tested"
The reader has not been told that the PPO paper has a second variant with a KL penalty and an adaptive coefficient. One clause where the clipped objective is introduced fixes this and also prepares the next section, whose subject is a KL penalty.

**PPO-19** (minor, language). Heading: "Training Diagnostics".
Quote: "The approximate KL has the same shape and is about ten times smaller; the cell prints its peak."
"Ten times smaller" compares a divergence in nats with a fraction of samples; the two have no common unit. In JAX the shape also differs: the approximate KL falls from 0.0071 to 0.0056, about a fifth, where the fraction falls by half. Say "has a similar shape" and give the two printed numbers.

**PPO-20** (moderate, pedagogy). Heading: "Training Diagnostics".
Quote: "The unclipped policy is close to deterministic, and the harmonic mean lies within about a tenth of the unclipped arm's effective sample size"
This cell confirms a formula derived in the paragraph before it (author decision 4), and needs a helper cell (`action_probs`) to do so. Outputs agree (0.49 against 55 percent; 0.41 against 47 percent). Move to exercise 4 with the rest of the effective-sample-size thread.

**PPO-21** (minor, pedagogy). Heading: "Training Diagnostics".
Quote: "Finally, a trained clipped agent, acting greedily on the most probable action to remove the noise of sampling, should balance the pole to the time limit:"
The cell prints 500 in both frameworks and the text never returns to it; the subsection ends on a code cell. It is a check that the reader expects to pass. Either cut it or use it: it is the only place where the section evaluates a policy rather than reading training returns, and that distinction (training return of a sampling policy against evaluation return of the greedy one) deserves one sentence.

**PPO-22** (minor, correctness, fairly confident). Heading: "Vectorized Collection and Minibatches".
Quote: "the step after an episode ends returns the first observation of a new episode with reward zero and ignores the action"
This describes the next-step autoreset mode, the default of Gymnasium 1.x. Earlier versions and the same-step mode behave differently, and the mode is configurable. Name the mode and the version so that the sentence does not rot.

**PPO-23** (minor, language). Pattern: question-led cell introductions, seven in the section.
Instances: "How many environment steps does reuse save?", "Does the clip matter there?", "How often does the clip act?", "How much is such a reweighted batch still worth?" Four of them fall within two pages of "PPO in Practice". Vary the middle ones.

## Code and experiments

- `ppo-reuse-against-environment-steps-1/2`: prose reads numbers the plots do not show; no printed summary; missing control (PPO-14, 15, 16).
- `ppo-the-ablation-3`: prose quotes 64 unseen seeds; the visible outputs are 0 of 8 held with 4 dead (PyTorch) and 1 of 8 held with 1 dead (JAX) (PPO-17).
- `ppo-the-ablation-5`: prose "near one check in twenty" matches (5.3 and 5.0 percent; 4.9 and 4.9 percent). "several times as often" matches (17.6 and 25.1 percent; 24.9 and 37.6 percent).
- `ppo-how-to-know-your-rl-is-broken-1`: "typically tens of times the clipped run's" is 46 times in PyTorch and about 100 times in JAX. "Tens to a hundred times" covers both.
- `ppo-training-diagnostics-1`: 0.65 to 0.25 nats in both stores; matches.
- `ppo-how-to-know-your-rl-is-broken-4` and `ppo-training-diagnostics-2`: confirm a derived formula (PPO-20).
- `ppo-how-to-know-your-rl-is-broken-5`: a check expected to pass, not read (PPO-21).
- `ppo-from-a-teaching-loop-to-a-real-one-1`: demonstrates the rectangle but not the thing the prose warns about, since no episode ends inside it ("0 episode boundaries inside it"); the autoreset behavior described in the next paragraph is never exhibited. Collecting with a fresh agent, whose episodes last about 20 steps, would show the boundaries and the autoreset step in the same cell.
- Cell IDs `ppo-how-to-know-your-rl-is-broken-*` are leftovers of a removed heading.
- Tabs: PyTorch and JAX only. The JAX tab paragraph after `ppo_epochs` is confined to its tab. No TensorFlow or MXNet material.

## Exercises

Ten exercises, about 2,150 words, 41 percent of the prose length. Quality is high. Exercise 1 (three step limits for one logit) is exactly the right first exercise; exercise 9 (the monotonic-improvement bound) derives the bound by coupling in five guided steps and then makes the reader compute how small the certified step is; exercise 4 on the two KL estimators is practical and correctly credited. Problems:

- Exercises 8 and 10 are small projects (a full vectorized PPO; TRPO with conjugate gradient). Both are worthwhile, but exercise 8 quotes reference numbers ("about 45,000 steps", "about 15,000 to 20,000 steps", "about 25,000") that depend on the reuse experiment being reread (PPO-14).
- Exercise 5 (bootstrapping at the cut) has no label while its neighbours do; harmless.
- Exercise 6 (KL early stopping) names a second mechanism the text never mentions. One sentence in the text ("some implementations also stop the passes when the approximate KL exceeds a target") would make the exercise an extension, not an introduction.
- Volume: the effective-sample-size material moved out of the text (structural finding 1) should replace part of exercise 4c, not add to it.

## Still open from October

The stagecraft line ("First the casualties, counted rather than asserted") is gone. The two-action example flagged as a keeper is intact. Symbol overloading is now handled by the shared notation table; $\rho_t$ is listed there, although an exercise of the policy-gradient section still uses $\rho_\gamma$ for an unnormalized occupancy.

## For the chapter review

- This is the strongest of the first three sections and the natural centre of the chapter's first half. It depends on the previous section for GAE, `lambda_return` and `steps_to` (which it should use and does not), and on chapter 14 for the policy gradient, the occupancy distribution and the time-limit bootstrap. It sets up the next section with the path-against-destination distinction, which that section picks up explicitly.
- It needs the KL divergence and has no definition of it; the same is true of the next section. One definition here serves both.
- Overlap: fixed-length segments are also explained at the end of the previous section. The entropy-as-divergence-from-uniform identity appears here and again in the next section ("A uniform reference turns the penalty into an entropy bonus"); small, and acceptable under the duplication rule, but the next section could cite this one.
- Notation against chapter 14: consistent ($R(\tau)$, $J(\theta)$, $d^\pi_\gamma$, $\hat{A}$). New here: $A^{\textrm{old}}$ for $A^{\pi_{\theta_{\textrm{old}}}}$, $\bar{L}$ and $\hat{L}$, $W_T$. `N` is the number of steps in a batch in the text and the number of parallel environments in "Vectorized Collection and Minibatches"; rename one.
- Pattern for the chapter: several conclusions rest on runs "outside the notebook" (64 seeds, ten seeds at two other learning rates). The practice is legitimate but here the unseen evidence outweighs the seen, and in one case the visible output of one framework contradicts the sentence above it.
