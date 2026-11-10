# Actor-Critic Methods and Multi-Step Returns

Source: `chapter_deep-reinforcement-learning/actor-critic.md` (lines 1 to 776; about 5,300 words of prose, 2,000 words of exercises, 27 cells). Stored outputs read: PyTorch and JAX. Figures rendered: `mdl-rl-td-mc-spectrum.svg` and the stored training curves.

## Verdict

The mathematics of this section is correct and in places beautifully taught: the hand-computed transition, the ten-state chain with a closed-form bias and variance, the telescoping proof that the lambda-mixture is a discounted sum of TD errors, and the exact-critic proposition are all textbook-quality. I recomputed the chain numbers, the telescoping identity and the proposition and found no error. The weakness is the experimental half. The section runs six measurements on CartPole, a task on which (as the text itself says) bootstrapping does not help, and then explains each result causally ("because the young critic's bias does more harm than the variance it removes") although no experiment isolates that cause: the two arms of the main comparison differ in weight normalization and gradient capping as well as in the weight, and the two frameworks disagree on whether lambda = 0 is better or worse than lambda = 1 at a frozen policy. The single most important fix: make the experiments claim only what they show (lambda at or above 0.9 trains about twice as far as lambda at or below 0.5 on a short budget; nothing in the training runs separates 0.9, 0.95 and 1), and cut the measurements and the Adam excursion that do not serve that one result.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening** states a problem the reader already has: "Each of its updates waits for eight whole episodes, up to 4,000 steps of balancing". The question "Can a policy-gradient agent learn the same way?" is answered from material in hand (the learned baseline). Keep it as is.
- **"One Transition by Hand"** is the best paragraph of the section. Three numbers (a critic at 100, a surviving step with delta 0, a fall with delta of minus 99) and then the untrained critic that gives every step delta near 1. The reader sees both what the weight says and why a young critic is useless, before any theory. This is the MacKay move.
- **The chain example** gives a closed-form bias and variance for every depth and an interior optimum at n = 4, with the sharp remark that the optimum is interior only because the critic's error tapers ("deeper targets bootstrap from better-known states"). The figure carries it well.
- **The bias argument** in "An Exact Critic Gives the Advantage" is exact about which term the baseline identity covers and which it does not ("It does not cover" the value at the next state, "which depends on the action through the next state"). Most texts wave at this.
- **The telescoping proposition** and its three-line proof, followed by "GAE as a Backward Scan" reusing `backward_scan` of the earlier chapter: one identity, one line of code.
- **The exact-critic proposition** ("With an exact critic, the sampled tail only adds noise") and the reading that the interior optimum "measures the critic's error" is a real insight, stated plainly.
- **Candor.** The section reports that its own method loses to the baseline on its own task. That is the voice the author asked for. The problem is what is then built on the result, not the honesty.

## Structural findings

1. **The experiments outnumber the ideas, and the task cannot show the section's subject.** Six measurements (head-to-head curves, steps-to-450 table, shared-batch variance trace, endpoint assertions, lambda sweep, frozen-policy probe) sit in a section whose ideas are three (TD error as advantage, the n-step and lambda families, the bias and variance trade). On CartPole with a deterministic reward of 1 per step and a bootstrap at the time limit, the critic never pays off, so the reader finishes without having seen bootstrapping win anywhere except on the ten-state chain. Direction: keep the head-to-head (it is honest and short), the lambda sweep and the frozen-policy probe; reduce "The Two Weights on Shared Batches" to two sentences and one printed line, or move it to an exercise; and consider one cheap positive demonstration (CartPole with additive reward noise, where the Monte Carlo variance bites) as an exercise with a stated expected outcome.
2. **The Adam, normalization and cap excursion is a second topic.** Two dense paragraphs under "The Batched Loop" (about 300 words) and exercise 6 explain why the TD weight is left raw and the gradient norm capped at 0.5. The reader came for actor-critic and gets an analysis of Adam's second-moment memory. The next section then drops the cap without a word (ppo.md says "no cap on the gradient"). Direction: three sentences in the text (raw weight, cap at 0.5, the reason in one clause), the analysis in exercise 6 where it already lives.
3. **The proposition that explains the probe comes after the probe.** The probe's reading says "as the proposition below predicts". Put "The Source of the Interior Optimum" before "A Frozen-Policy Probe": theory first (an exact critic makes lambda = 0 best), then the probe as the test of what a wrong critic does. The sweep can stay first as the motivating question.
4. **"A Critic or a Group Baseline" is two unrelated paragraphs under one heading**, followed by a recap. The first paragraph (critic against group baseline) is a good judgment and belongs. The second (fixed-length segments in deployed implementations) is taught again in ppo.md under "Vectorized Collection and Minibatches", which has the code. One owner: keep it in ppo.md and cut it here to a clause.
5. **Two endings.** The paragraph beginning "We can now update a policy from a single transition" recaps the section immediately before the Summary. This is the pattern the author asked to remove. Move its last two sentences (the two limits that the next section takes up) to the end of the Summary or keep them as a one-sentence bridge; delete the recap.
6. **Length.** With findings 1, 2, 4 and 5 applied the section drops by roughly 900 words and three cells with no loss of content. It does not need splitting.

## Passage-level findings

**AC-1** (moderate, explanation). Heading: section opening, setup paragraph.
Quote: "The new hyperparameter `critic_steps` sets how many regression passes the critic takes per batch, and `grad_clip` caps the norm of the actor's gradient; both are explained with the training loop."
Two hyperparameters are named before the reader knows there is a critic update or why a gradient would be capped. The setup paragraph also restates CartPole in full although the previous section defined it. Fix: keep one sentence on the task ("We keep CartPole and the agent of ..."), and introduce `critic_steps` and `grad_clip` where the loop uses them; the cell can still set them here with a comment.

**AC-2** (minor, explanation). Heading: "Bootstrapping the Reward-to-Go".
Quote: "As in the reward-to-go estimator :eqref:`eq_rtg` of :numref:`sec_baselines`, the actor's update omits the factor"
This sentence sits after the `td_target` cell, detached from the actor's update, which has not been written yet (it arrives as the second half of `eq_actor_critic` a page later). Move it to the sentence after `eq_actor_critic`.

**AC-3** (minor, explanation). Heading: "The Chain Example".
Quote: "A target that samples $n$ rewards and then bootstraps from the critic at state $n$ inherits the critic's error there"
The formulas that follow are for the target computed from state 0; the text never says so (exercise 2 does: "of the depth-$n$ target from state $0$"). Add "from state 0". The numbers are right: I recomputed bias 0.38 and variance 0.02 at n = 1, variance 0.17 at n = 10, and a mean squared error of 0.169, 0.101, 0.174 at n = 1, 4, 10.

**AC-4** (minor, correctness, confident). Heading: "The Batched Loop".
Quote: "Its best-known relative, A2C :cite:`Mnih.Badia.Mirza.ea.2016`, weights the same score by $n$-step advantages"
The cited paper introduced the asynchronous method A3C. A2C is the synchronous variant that came out of the OpenAI Baselines implementation afterward. Say "A2C, the synchronous form of the A3C of ...".

**AC-5** (moderate, thought). Heading: "The Batched Loop".
Quote: "Adam divides each gradient coordinate by the square root of a running average of its past squares, an average that with the default decay remembers about a thousand steps, longer than our 150 updates."
This and the following paragraph are an analysis of an optimizer inside the explanation of an algorithm. The numbers in it ("about eightfold over a run", "a standard deviation of about $3$ early in training to $0.1$ or less") are quoted before any run has been shown; the eightfold figure is printed by no cell. See structural finding 2 for the fix. If the paragraph stays, move it after the shared-batch trace, whose output (2.7 falling to 0.04 in PyTorch, 2.5 to 0.06 in JAX) is its evidence.

**AC-6** (major, thought). Heading: "Bootstrapped and Monte Carlo Weights on CartPole".
Quote: "Two more details keep only the weight different"
They do not. The REINFORCE arm standardizes its weights within the batch and its cap never acts (the stored output prints "cap active 0%"); the actor-critic arm uses raw weights and its cap acts on 28 to 29 percent of updates. Under Adam these are different effective step-size schedules, as the text itself argued two paragraphs earlier. The comparison therefore confounds the bootstrap with the normalization and the cap, and the lambda sweep later in the section (where every arm, including lambda = 0, is standardized) is the cleaner design. Fix: either say plainly that the two arms differ in three things and that the comparison is between two tuned recipes, or run the head-to-head inside the standardized `gae_weight` family (lambda = 0 against lambda = 1 at the full budget), which removes the confound and one weight function.

**AC-7** (major, correctness of interpretation). Heading: "Bootstrapped and Monte Carlo Weights on CartPole".
Quote: "What remains is the critic's bias, and it is largest while the critic is young, the phase in which actor-critic falls behind."
The cause is asserted, and the Summary repeats it as fact ("because the young critic's bias does more harm than the variance it removes"). No cell measures the bias during training or varies it. The stored curves (PyTorch) show the two arms together for the first 25,000 steps and actor-critic dipping from about 270 to about 100 between 25,000 and 50,000 steps, which reads as instability after early progress as much as a slow start. Given AC-6, raw weights with an active cap are an equally plausible cause. Fix: state the bias as the likely cause and point to the evidence that exists (the frozen-policy probe measures a relative bias near 1 at lambda = 0), or drop the causal sentence from the Summary.

**AC-8** (moderate, correctness of interpretation). Heading: "Bootstrapped and Monte Carlo Weights on CartPole".
Quote: "The Monte Carlo weight reaches the ceiling first on at least two of the three seeds shown."
True of both stored runs (PyTorch 79, 64, 77 against 113, 187, 112 thousand steps; JAX 67, 92, 83 against 98, 106, 58), but in JAX the fastest of the six runs is an actor-critic run, and the spread within an arm (58 to 106) is as large as the gap between arms. The 40-seed sentence that follows is what carries the claim, and no cell produces it. That is acceptable under the house rule on repeated runs, but say where the reader can reproduce it (an exercise, or `num_seeds = 40` and the expected count), and soften "On CartPole the bootstrap does not even make the runs steadier" to what three seeds show.

**AC-9** (minor, pedagogy). Heading: "The Two Weights on Shared Batches".
Quote: "On the same transitions, the Monte Carlo advantage has two to three times the TD error's standard deviation in the first updates"
The ranges match both stores (2.9 and 2.5 times; variance ratio 56 and 28 over the run). The measurement supports one sentence ("the variance saving is real but is not what limits learning here"). It currently costs two cells and three paragraphs. See structural finding 1.

**AC-10** (moderate, pedagogy). Heading: "GAE as a Backward Scan".
Quote: "Both endpoints are identities that we can check on a batch."
The cell asserts two identities that the proposition just proved and prints a fixed string. It cannot fail unless the code is wrong, so it is a unit test, not an experiment (author decision 4). The interesting content is the caveat that follows (the lambda = 1 identity fails on truncated episodes), which the cell does not show because the untrained policy never reaches the time limit. Fix: either cut the cell, or make it teach the caveat: run it on a batch from a trained agent, where the assertion at lambda = 1 fails by the discounted value at the cut, and print that gap.

**AC-11** (minor, explanation). Heading: "The $\lambda$-Return and the Telescoping Identity".
Quote: "The weights $(1-\lambda)\lambda^{n-1}$ sum to one."
At lambda = 1 the weights are all zero, and the mixture is defined only as a limit, which the next sentences handle for terminating episodes but the display claims for the closed interval. Either define the estimator by `eq_gae_deltas` at lambda = 1 or write the range of the mixture as half-open and say that lambda = 1 is the limit. One clause.

**AC-12** (minor, correctness, fairly confident). Heading: "GAE as a Backward Scan".
Quote: "The same backward sum has an online form, TD($\lambda$), which keeps a decaying *eligibility trace* per critic parameter"
The offline lambda-return and online TD(lambda) with accumulating traces agree exactly only when updates are applied at the end of the episode (or for the "true online" variant). "The same sum" overstates it; "an online counterpart" is accurate.

**AC-13** (major, correctness of interpretation). Heading: "The $\lambda$ Sweep".
Quote: "On a short budget the critic is wrong for most of training, and the estimators that lean on it least learn fastest."
The sweep's own output does not order the three large values: PyTorch medians are 420, 407, 450 for lambda = 0.9, 0.95, 1 with seed ranges that overlap almost completely; JAX gives 337, 362, 352. What the sweep shows is a step between lambda at or below 0.5 and lambda at or above 0.9, and the text's first sentence says exactly that. "Learn fastest" for the estimators that "lean on it least" implies a monotone trend the data do not have. Rewrite: "The sweep separates two groups and cannot order the members of either."

**AC-14** (major, correctness of interpretation). Heading: "A Frozen-Policy Probe".
Quote: "Across the eight frozen agents, the median error is smallest at $\lambda = 0.9$ or $0.95$, about half the error at $\lambda = 1$ and below the error at $\lambda = 0$."
The two stores tell different stories. PyTorch: error relative to lambda = 1 is 1.91, 1.61, 0.83, 0.61 for lambda = 0, 0.5, 0.9, 0.95, so lambda = 0 is about twice as bad as Monte Carlo and the optimum is clearly interior. JAX: 0.55, 0.52, 0.46, 0.48, so every lambda below 1 is about half the Monte Carlo error and lambda = 0 is within noise of the best. "About half" fits JAX and the 0.95 row of PyTorch only; "below the error at lambda = 0" holds in JAX by 0.46 against 0.55 on a median of eight agents. The sentence is written to be true of both and so hides that the interior optimum is clear in one framework and absent in the other. Second point: the printed relative bias at lambda = 0 is 1.19 (PyTorch) and 0.99 (JAX), that is, the mean one-step gradient differs from the reference by as much as the reference's own length. That is the most striking number in the cell and the prose never reads it. Fix: read the bias row first ("at lambda = 0 the bias is as long as the gradient itself"), then say that whether this beats the Monte Carlo variance depends on the agent and differed between our runs, and that a lambda near 0.9 was the best or close to the best in both.

**AC-15** (moderate, correctness of interpretation). Heading: "A Critic or a Group Baseline" (closing paragraph) and Summary.
Quote: "On CartPole the best choice keeps most of the tail, $\lambda$ near $0.95$, because the critic is young for much of training."
No training experiment in the section supports 0.95 over 1: the head-to-head is won by the Monte Carlo weight and the sweep ties 0.9, 0.95 and 1. The only support is the one-draw gradient error of the probe, which is not a training result. The sentence in "The Source of the Interior Optimum", "The default $\lambda = 0.95$ is in this sense a statement about how accurate critics usually are during training", is a nice line but the same overreach: it presents a convention as a measurement. Say that 0.95 is the convention of the PPO paper, that on this task any value from 0.9 to 1 trains alike, and that the probe shows why a value just below 1 is a reasonable default (most of the bias gone, about a third to a quarter of the variance of lambda = 1).

**AC-16** (minor, language). Pattern: six paragraphs open with a question that the next cell answers.
Instances: "Does the bootstrapped weight learn faster than the Monte Carlo weight it replaces?", "If the bootstrap does not speed up learning here, what does it change?", "Which $\lambda$ trains best?", "Why should the best $\lambda$ lie strictly between the endpoints?" The first and last are earned. The middle two are the formula the author asked to vary; turn them into statements of what will be measured.

**AC-17** (minor, language). Heading: "The Batched Loop", JAX tab.
Quote: "so that the compiled `_actor_pass` is reused across batches of different lengths"
Padding and compilation detail is now confined to tab-specific paragraphs, which is the right place, but the PyTorch tab's counterpart ("The loss alone is `actor_loss`, which the probe at the end of the section reuses.") is a forward reference that serves no reader at that point. Cut it; the probe can introduce what it reuses.

## Code and experiments

- `actor-critic-gae-in-one-old-function-2`: a check that cannot fail (AC-10).
- `actor-critic-bootstrapped-and-monte-carlo-weights-on-cartpole-1`: output agrees with the prose in both frameworks, with the caveat of AC-8. "over 40 seeds" and "two actor-critic runs did" are printed by no cell.
- `actor-critic-why-the-critic-s-moving-target-is-survivable-here-and-fatal-later-1`: ranges in the prose match both stores. The cell ID is a leftover from an earlier heading; harmless but confusing in a diff.
- `actor-critic-measuring-the-dial-2`: prose ranges ("about 150 to 250", "about twice as high") match both stores; the conclusion drawn exceeds them (AC-13).
- `actor-critic-a-frozen-policy-probe-1`: frameworks disagree on the lambda = 0 row (AC-14). "the variance rises more than tenfold" holds (24 times and 28 times).
- "about eightfold over a run" (Adam's steps): no cell prints it.
- The head-to-head arms differ in normalization and capping (AC-6).
- `td_weight`, `mc_weight`, `gae_weight` are three weight functions for one family. `gae_weight(0)` and `gae_weight(1)` would cover the first two if the comparison is moved into the standardized family, saving a cell and the confound.
- Tabs: PyTorch and JAX only; no TensorFlow or MXNet material. The two tabs fragment the prose in one place (the two tab paragraphs after `policy_step_clip`).

## Exercises

Nine exercises, about 2,000 words, 38 percent of the prose length (the guide suggests about 30). They are the strongest part of the section: exercise 1 (bias of the bootstrapped weight) and exercise 8 (bias of GAE with the geometric model for the error) turn the text's claims into derivations, and exercise 2 tests the chain. Problems:

- Exercises 3c and 6c each ask for ten-seed training campaigns; 6c alone is thirty runs of 150 updates. Mark the cost or cut the seed counts.
- Exercise 6 is the Adam excursion again at 400 words. If structural finding 2 is applied it becomes the single home for that material and is fine; otherwise it is a second copy.
- Exercise 7 (correlation of the two weights) is a derivation followed by a plot whose expected behavior is not obvious to the reader; give the expected sign early and late.
- Missing: one exercise in which bootstrapping wins (see structural finding 1).

## Still open from October

"About eight measurements in `actor-critic.md`" is still open (six now, see structural finding 1). The sentence that disowned a metric is gone. The padding and JIT prose is now confined to tab paragraphs, which resolves that item. The kept line about substituting the prediction survives reworded ("Instead of waiting for the sample, we substitute the prediction").

## For the chapter review

- Opens the chapter well and sets up `Batch.gae`, `Batch.lambda_return`, `td_target` and `steps_to`, all used by later sections. ppo.md refers to a function "`train_ac` of" this section; the function is called `train` (with `td_weight`). Stale name.
- Overlap with ppo.md: fixed-length segments and the bootstrap at the cut are explained in both; ppo.md should own them. The gradient cap motivated at length here is dropped there silently.
- Notation against chapter 14: consistent with the shared table in mdp.md (`tab_rl_notation`). Two small shifts: chapter 14 calls the value-baseline step size $\eta_V$, this section calls the critic's step size $\alpha$ (the TD step size of the Q-learning section) for the same network; and $e(s)$ for the critic's error sits next to $\pi_e$, the behavior policy, in the table.
- Pattern to check across the chapter: causal explanations of training curves ("because ...") that no cell isolates, and sentences phrased loosely enough to be true of both frameworks' outputs when the two outputs point in different directions.
- Depends on Basics only through Adam; no appendix reference; no forward reference to the Language Models part that carries weight.
