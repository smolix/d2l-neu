# Schedules

Source: `chapter_optimization/lr-scheduler.md` (lines 1 to 936; about 4,500 words of body prose, 1,100 of exercises, 21 code cells).

## Verdict

This is a good section with one excellent part and one part that is read more generously than its results allow. The valley analysis (budget and final rate, four schedules, every printed number explained by a formula) is at the level of the model texts, and the branching experiment at the end is the best demonstration of warmup–stable–decay I know of at this scale. The middle is weaker. The network experiments are interpreted through the same two numbers, but the stored results support only half of that reading: the final rate decides the test accuracy and its stability, while the "budget" shows up only late, only in the training loss of a network that is overfitting, and the five loss curves coincide for the first ten epochs although their budgets differ by a factor of two. The warmup experiment shows that a ramp makes an extreme rate trainable, in a batch-normalized network where the nominal rate means little, and never shows that warmup is useful. The single most important fix: read the network experiments for what they show (final rate and stability on the test set, budget only on the training loss), and say candidly that on this network warmup buys nothing.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **Two numbers summarize a schedule.** :eqref:`eq_scheduler-bias-noise` is derived in fifteen lines, read in one sentence, and given numbers at once ("a budget of $B = 10$ leaves $e^{-2} \approx 14\%$ of the starting distance"). The reader then carries budget and final rate through the whole section. Keep this device.
- **The valley experiment is read number by number** (lines 176 to 192). I recomputed all of it: the noise ball 0.268, the frozen exponential value 0.134, the $\alpha = 1/2$ prediction 0.094 from the final rate, the $\alpha = 1$ value 0.0114 from $c^2\sigma^2/((2c\lambda - 1)t)$, the ratio "eight to nine", the budget "a quarter as large". All agree with the stored outputs of both frameworks.
- **The Robbins–Monro conditions are put in their place** (lines 106 to 113 and 194 to 199): sufficient, about infinite runs, and saying little for a fixed $T$. This is the judgement the voice guide asks for.
- **The standard-deviation column** of the table (cell `lr-scheduler-decay-schedules-compared-2`) turns "the noise ball of a network" into something the reader can see: the accuracy of a constant-rate run depends on the epoch at which one stops.
- **Branching off the plateau** (lines 696 to 771) does what the cited papers do, on the reader's machine, and ends on a sentence worth protecting: "although nobody chose its horizon when training started". The remark that optimizer state must be copied with momentum or Adam is the kind of practical detail the part needs more of.
- **The noise discipline is stated once and kept**: "A difference between two schedules means something only if it is larger than the difference between two runs of the same schedule."
- The river-valley figure renders cleanly and carries the explanation (the decayed branch stops short of the constant run, which is honest).

## Structural findings

1. **The network half borrows the authority of the valley half.** On the valley the two-term formula is derived and confirmed. On the network it is a metaphor, and the prose treats it as confirmed again ("This is ... at work on a network"). See SCHED-4, SCHED-5, SCHED-9. The section would be stronger if it said: the formula tells us what to look at, and here is what the network shows.
2. **Warmup is the weakest subsection.** It has a mechanism that is cited but not measured, an experiment at a rate no one would use, and a combined warmup-plus-cosine run whose result is never read. The appendix meanwhile promises that this section "measures this in a real model" (SCHED-6, SCHED-7).
3. **Everything is plain SGD.** The section opens and closes on the schedule of large language models, which are trained with AdamW. One paragraph must say what carries over (SCHED-10).
4. **What a reader looks for and does not find:** how to choose the peak rate and the warmup length in practice (even a rule of thumb with its caveat), and one sentence on warm restarts, since the cosine citation is the warm-restarts paper. Neither needs more than a few lines.
5. **Length.** The section is not too long, and I would not split it. Savings: the two tab blocks after `configure_optimizers` (about 200 words, SCHED-12), the worst-case paragraph (SCHED-8), and the Summary (SCHED-14).

## Passage-level findings

**SCHED-1** (moderate, language/thought). Under "Progress and Noise" and "Branching Off the Plateau".

> is the sum of all learning rates so far, which we call the *budget* of the

> We now decide on a budget of 30 epochs. We copy the checkpoint and *branch*

"Budget" is coined for $\sum_s \eta_s$ and used that way about 30 times. From line 521 on the same word also means the length of training or its compute ("whenever the budget changes", "models for several budgets", "matches cosine decay across budgets", "models at many budgets", "a 30-epoch budget", exercise 6 "Branching at several budgets"). In the WSD part both meanings occur within a page, and the claim "the travel along the floor is the budget" sits between them. Fix: keep "budget" for $\sum_s \eta_s$ and write "length", "horizon" or "number of epochs" for the duration at lines 521, 622, 631, 711, 766 to 767, 878, 892, 918 and 922.

**SCHED-2** (minor, correctness, confident). Under "Progress and Noise".

> to zero with an infinite budget already suffices, as long as it falls

On the quadratic, $\eta_t \to 0$ with $\sum_t \eta_t = \infty$ suffices with no further condition: the variance obeys $a_{t+1} = (1 - \eta_t\lambda)^2 a_t + \eta_t^2\sigma^2$, which tends to zero under those two conditions alone. The slowness condition is needed only for the stronger statement that the ball tracks the current rate. Delete the clause, or attach it to "the ball follows the current rate".

**SCHED-3** (minor, explanation). Under "Four Schedules on the Valley".

> follow. Once $\eta_t \lambda$ drops below $0.02$, the iterate no longer

The explanation of the frozen exponential run is right but introduces a new idea (the rate falls faster than the iterate can follow) when the section's own notion explains it in one line. Noise injected at step $s$ is later contracted by $e^{-\lambda(B_\infty - B_s)}$, and for exponential decay the remaining budget is $B_\infty - B_s = \eta_s/(1-\gamma) = 50\,\eta_s$. Once $50\,\eta_s\lambda < 1$, that is $\eta_s < 0.1$ on the flat direction, the remaining budget cannot remove what is injected. This also explains where the 0.02 comes from, which the text does not. The preceding sentence ("Its budget ... is enough to cover the starting distance, but") then needs no "but": the total budget covers the start, the remaining budget does not cover the noise.

**SCHED-4** (moderate, thought). Under "Five Schedules on the Network".

> twice the budget. This is :eqref:`eq_scheduler-bias-noise` at work on a

The stored results do not show a schedule needing "both a large budget and a small final rate". (a) In the stored PyTorch plot the five training-loss curves coincide through epoch 10, by which time the constant run has spent a budget of 3.0, exponential 1.9 and square-root 1.4. Progress on this network is not proportional to $\sum\eta$. (b) The constant run, with the largest budget and no decay, ends at the same training loss as cosine (0.064 against 0.062 in PyTorch; in JAX 0.065 against 0.044). (c) On the test set, exponential decay with a third of the budget ties piecewise and cosine in PyTorch (0.909, 0.911, 0.911) and trails by under a point in JAX. What the experiment shows cleanly is the next paragraph's point: the final rate sorts the test accuracies and removes the epoch-to-epoch jitter. Fix: lead with that, present the training-loss split into early-decaying and late-decaying shapes as an observation consistent with the budget idea, drop "at work on a network", and note that the first ten epochs are insensitive to the rate.

**SCHED-5** (moderate, correctness, medium confidence, not verified by running). Under "A Testbed".

> normalization widens the range of learning rates at which a network trains

Every layer but the last is followed by batch normalization, the optimizer is SGD without weight decay, so those weights are scale-invariant and their norm can only grow ($\|\mathbf{w}_{t+1}\|^2 = \|\mathbf{w}_t\|^2 + \eta^2\|\mathbf{g}_t\|^2$). The effective rate of such a layer is $\eta/\|\mathbf{w}\|^2$ and falls on its own during training. This is a built-in decay that confounds the comparison of schedules by nominal budget (it may be why the curves coincide for ten epochs), and it is very likely the reason a nominal rate of 7.5 "trains" after a ramp: the ramp lets the norms grow before the rate peaks. The section attributes both effects to other causes and never mentions this one. Fix: either measure it (print the weight norm of one convolution at the start and end of the constant run and of the ramped 7.5 run) and say so in two sentences, or state the caveat. If the norms do grow by a large factor, the sentence "the network trains at the rate that failed from a cold start" should say "nominal rate". The attribution to Santurkar et al. is from memory plausible; Bjorck et al. (2018), "Understanding batch normalization", is the more direct source for the learning-rate claim. Unverified.

**SCHED-6** (moderate, pedagogy). Under "Warmup".

> With the ramp, the network trains at the rate that failed from a cold

The experiment shows that warmup makes a rate of 7.5 possible. It does not show that anyone would want it: after ten epochs the ramped run has a training loss of about 0.19 and 88 to 90% test accuracy, no better than the rate of 0.3 without a ramp. Then `warmup(cosine)` is trained at a peak of 0.3, where no warmup is needed, and its printed result (test accuracy 0.907 in PyTorch, 0.908 in JAX, against 0.911 and 0.907 for cosine alone) is never read. A reader finishes the subsection not knowing whether warmup helped. Fix: say it. For example: "On this network warmup changes nothing at the rates we would choose: the warmed-up cosine run ends where the cosine run did. Warmup matters when the best rate lies above what a cold start tolerates, which is the usual case for transformers trained with Adam and for large minibatches (:numref:`sec_practice`)." Better still, add to the list of experiments to redesign: a short `TinyLM` run with AdamW at a rate that diverges cold and trains with a 100-step ramp, which would also serve SCHED-10.

**SCHED-7** (moderate, correctness of a cross-reference). Under "Warmup".

> filled (:numref:`subsec_mdl-schedules-warmup`). Our

The reference is right: the appendix section "Warmup" holds the claim (first step $\eta$, stationary $0.23\,\eta$, about ten steps). But that appendix section ends "Right after initialization that curvature can be large and change quickly ... :numref:`sec_scheduler` measures this in a real model." This section measures no curvature. It shows a cold start failing and a ramped start training, and cites Kalra and Barkeshli for the mechanism. Either add the measurement here (the largest Hessian eigenvalue by power iteration on one minibatch, before and after the ramp, is a short cell and would teach the mechanism) or change the appendix sentence to "shows the failure and its cure". The first is the better book.

**SCHED-8** (minor, focus). Under "Four Schedules on the Valley".

> Worst-case theory for a fixed $T$ even favors a constant rate if the

Four worst-case results in eleven lines (convex averaged SGD, the lower bound, the nonconvex gradient-norm rate, the last-iterate schedule), none of which the reader can use, followed by "No theorem ranks one decay shape against another on a deep network". The results are stated correctly as far as I can tell. Keep two sentences: for a fixed $T$ the theory concerns the final rate, and the worst-case optimal schedule for the last iterate decays linearly to zero. Point to the appendix for the rest.

**SCHED-9** (moderate, pedagogy). Under "Warmup–Stable–Decay".

> part of the run, and with a test accuracy in the same range.

The WSD run ends at half the training loss of the cosine run (0.028 against 0.068), which the text reads as the "real progress" of the river-valley model. Its test loss is higher (0.362 against 0.310 in PyTorch, 0.355 against 0.325 in JAX) and its test accuracy the same. On this network the extra travel along the floor is further fitting of the training set. One clause would keep the reading honest: "with the same test accuracy and a slightly higher test loss: on a network this small the extra progress is memorization, and the argument for WSD here is flexibility, not a better model."

**SCHED-10** (moderate, technical detail). Under "Warmup".

> network uses plain SGD, so it isolates the first reason.

Nothing in the section says how its conclusions change under momentum or Adam, the optimizers the reader will actually schedule. Add a short paragraph after the WSD experiment: with momentum the rate that enters the budget and the noise ball is $\eta/(1-\beta)$; with Adam the step is about $\eta$ per coordinate whatever the gradient scale, so the schedule sets the step size directly and the peak rate is small ($10^{-4}$ to $10^{-3}$); decay and warmup play the same roles. Cite the table in :numref:`sec_practice` for what large runs use.

**SCHED-11** (minor, correctness, unverified). Under "Branching Off the Plateau".

> to each other when each is well tuned, with linear decay to zero slightly

From memory, Hägele et al. compare a constant rate with a cooldown against cosine and compare cooldown shapes; I do not recall a linear-decay-to-zero-from-the-start arm there. Bergsma et al. compare linear decay to zero mainly against cosine; exercise 8 says "matched or beat cosine and WSD". Check both attributions against the papers, or cite only Bergsma et al. for "linear decay to zero slightly ahead".

**SCHED-12** (minor, code prose). After `configure_optimizers`.

> keep running averages with `momentum=0.1`, which weights the newest

The two tab blocks run to nine and eleven lines against the guide's "a sentence or two". The batch-norm momentum conventions, `use_fast_variance` and the default initializers are plumbing that belongs in code comments (two of the three already have one). Keep per tab: how the framework receives the schedule, and for JAX why the schedules use `jnp`.

**SCHED-13** (minor, language, pattern; 9 instances). Cell introductions narrate the code in one form: "We draw the training loss of every run on a logarithmic scale.", "We draw the five schedules and compute both.", "We train the network under each of the four decaying schedules", "We draw the training loss of the plateau run over all 30 epochs". The author asked for introductions that the argument calls for. Most of these can become the question the cell answers ("Do the shapes that keep the rate high also end lower?") or disappear into the preceding sentence.

**SCHED-14** (minor, language). Summary.

> schedules that end at a small rate settle their test accuracy, about two to

"about two to three points above the constant rate's" overstates the store: the gap over the last five epochs is 2.2 to 2.4 points in PyTorch and 1.1 to 1.9 in JAX, and the body says "about two points". Write "one to two points". The Summary is also nine sentences that retell every experiment; four would do (two numbers; late decay; warmup; WSD and branching).

**SCHED-15** (minor, language). Headings "Four Schedules on the Valley", "Four Decay Shapes", "Five Schedules on the Network" name a count, not a subject. Suggested: "Schedules on the Valley", "Decay Shapes", "Decay Shapes Compared".

## Code and experiments

- `lr-scheduler-four-schedules-on-the-valley`: a model experiment. It confirms a derived formula, but it also shows the case the formula misses (exponential decay), so it teaches more than the derivation. Keep.
- `lr-scheduler-a-testbed-3`: the prose says the test accuracy "moves up and down by a point or more from one epoch to the next". The stored PyTorch run supports this (standard deviation 1.4 points over the last five epochs); the stored JAX run has 0.3 points there. The claim holds across runs (the JAX plateau run of the branching experiment jumps by three points), but the two tabs do not both show it in this cell.
- `lr-scheduler-decay-schedules-compared-2`: see SCHED-4. "several times smaller than for the constant rate" is 3 times in the stored JAX run.
- `lr-scheduler-warmup-1`: "stays below about 20%" matches (0.190 and 0.100).
- `lr-scheduler-warmup-5`: output never read (SCHED-6).
- `lr-scheduler-warmup-stable-decay-2`: test loss not read (SCHED-9).
- `lr-scheduler-branching-off-the-plateau-4`: prose matches both stores.
- Numbers no cell prints: none of consequence. "less than 7%" (line 603) is $1 - (1+\cos(\pi/6))/2 = 6.7\%$, correct.

## Exercises

Eight problems, graded, most with a question that the section leaves open. Exercise 2(b) (the $c\lambda > 1/2$ threshold for $1/t$ decay) and exercise 6 (branching at several lengths against separate cosine runs) are the best. Exercise 4 (schedule-free SGD) and exercise 8 (linear decay to zero, with the spread between repeated runs as the yardstick) carry current material without adding it to the text, which is the right place for it.

- Exercise 1(b) asks the reader to relate the gap between test and training loss to "the noise term of :eqref:`eq_scheduler-bias-noise`". That term is about the distance to the minimum of the training loss. It says nothing about a generalization gap. Reword to ask about the epoch-to-epoch spread of the test accuracy, which the noise term does address.
- Exercise 7 (Langevin dynamics) is a good problem on a different subject. Keep it only if the chapter has no better home; it would sit more naturally at the end of :numref:`sec_sgd`.

## Still open from October

The October review asked to keep the river valley passage (kept) and listed the anthropomorphism "a neural network wants the opposite temperament" (gone). The clash of $\lambda$ does not occur here: $\lambda$ is curvature throughout.

## For the chapter review

- Order: this section sits between Adam and Muon. By the chapter's own scheme (direction, step size over time, noise), Muon is a second answer to "direction" and would follow Adam more naturally, with Schedules after both. In the present order Muon's experiments run at a constant rate one section after the reader learned why that inflates the final loss.
- Overlap with the appendix is handled correctly: the Adam warmup transient is one clause with a reference, the fixed-budget rates are stated and cited. The one broken link is the appendix's promise that this section measures curvature (SCHED-7).
- `lenet_bn` is saved for :numref:`sec_practice`, which reuses this testbed; the batch-normalization confound (SCHED-5) therefore affects that section too.
- A pattern to check across the chapter: network experiments interpreted with formulas derived on the quadratic, in sentences of the form "this is equation X at work on a network", and training loss used as the yardstick on networks that overfit.
- All experiments use SGD on a convolutional network while the section's motivation is language-model practice. If the chapter keeps `TinyLM` as its shared model, one schedule experiment on it would connect this section to its neighbours.
