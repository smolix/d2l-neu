# Chapter 15, Deep Reinforcement Learning: writing review

Sources: the ten `.md` files of `chapter_deep-reinforcement-learning/`, in book order. The eight sections were read in full by three reviewers (actor-critic, ppo, regularized; dqn, sac; model-based, offline-rl, rl-sequences), each with the stored PyTorch and JAX outputs; I read all eight reviews in full, reviewed `index.md` and `further-reading.md` myself, and checked the cross-section claims listed under "Checked for this file". About 37,500 words of prose and 14,000 words of exercises in the sections, plus about 3,000 in the introduction and the further reading.

Per-section reviews:
[index](15-deep-reinforcement-learning/index.md) ·
[actor-critic](15-deep-reinforcement-learning/actor-critic.md) ·
[ppo](15-deep-reinforcement-learning/ppo.md) ·
[regularized](15-deep-reinforcement-learning/regularized.md) ·
[dqn](15-deep-reinforcement-learning/dqn.md) ·
[sac](15-deep-reinforcement-learning/sac.md) ·
[model-based](15-deep-reinforcement-learning/model-based.md) ·
[offline-rl](15-deep-reinforcement-learning/offline-rl.md) ·
[rl-sequences](15-deep-reinforcement-learning/rl-sequences.md) ·
[further-reading](15-deep-reinforcement-learning/further-reading.md)

## Verdict

The mathematics of this chapter is at the bar and the experiments are not yet. Three reviewers recomputed the hand examples, the propositions and most exercise formulas and found no wrong equation in eight sections; several derivations (the telescoping identity for the lambda-return, the performance difference lemma and its bound, the tilted optimum by completing the divergence, the stable log-determinant, the two-armed bandit with a lower confidence bound) are as good as any textbook treatment I know, and the habit of predicting a number and then measuring it (the ceiling of 100 on CartPole, the synchronization bound of 86, the temperature trace from Adam's step size, the hedge's threshold at one quarter) is the method the Basics review asked for. What a reader cannot trust is how the training runs are read. In seven of the eight sections at least one headline sentence says more than the stored outputs show: a number the plot does not give (the PPO reuse experiment), a cause no cell isolates (the young critic, the batch, generalization, coverage), arms that differ in more than the factor credited, statistics over 20 to 64 unseen runs that the printed seeds contradict in one framework, a title claim that the table under it does not show (compounding error), and a pessimism result that a reviewer's reimplementation traces to a clip in a helper function. On top of this, the two control tasks are too easy for half of the mechanisms the chapter teaches to change the return, which the prose reports honestly each time and never resolves. What matters most: go through the table under "What the experiments support" and make every sentence claim what its cell prints; redesign the four experiments named there; define the KL divergence; and give the last section, which is the one most readers of 2026 come for, the room and the derivation that the rest of the book already assumes it has.

## Scores

1 to 5 as the section reviewers gave them; 5 means a page one would be happy to find in MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](15-deep-reinforcement-learning/index.md) | 4 | 3 | 3 | 3 | 4 |
| [actor-critic](15-deep-reinforcement-learning/actor-critic.md) | 4 | 3 | 3 | 4 | 3 |
| [ppo](15-deep-reinforcement-learning/ppo.md) | 4 | 4 | 3 | 4 | 4 |
| [regularized](15-deep-reinforcement-learning/regularized.md) | 4 | 4 | 4 | 4 | 4 |
| [dqn](15-deep-reinforcement-learning/dqn.md) | 4 | 3 | 3 | 3 | 4 |
| [sac](15-deep-reinforcement-learning/sac.md) | 4 | 3 | 3 | 4 | 3 |
| [model-based](15-deep-reinforcement-learning/model-based.md) | 4 | 4 | 4 | 4 | 4 |
| [offline-rl](15-deep-reinforcement-learning/offline-rl.md) | 4 | 3 | 4 | 4 | 4 |
| [rl-sequences](15-deep-reinforcement-learning/rl-sequences.md) | 3 | 3 | 3 | 3 | 3 |
| [further-reading](15-deep-reinforcement-learning/further-reading.md) | 4 | 4 | 3 | 4 | 4 |

The section scores use only 3 and 4 (24 fours and 16 threes in the eight sections), so they rank less than the texts differ. Read with the findings: the first half of `ppo.md` is the best exposition in the chapter (its reviewer would give it a 5 for explanation alone), `regularized.md` has the best experiment, and the first half of `offline-rl.md` and the first two thirds of `model-based.md` are close behind. The weakest section is `rl-sequences.md`, the only one a reader would struggle with, and by its content the most important. `actor-critic.md` and `sac.md` are the weakest of the rest, both for experiments that do not show the section's subject. Calibration differed in one respect: the reviewer of the first three sections lowered pedagogy when an experiment was misread (`ppo.md`, 3), while the reviewer of the last three kept pedagogy at 4 for `offline-rl.md` and `model-based.md`, each of which has a major correctness finding on its central experiment. By the first reviewer's standard those two fours would be threes.

## Technical correctness and currency

No section reviewer found a wrong equation, definition, proof or algorithm, and each recomputed a good deal. The findings below are of four kinds: experiments whose stated result or mechanism the outputs contradict; statements about papers and history; statements about current practice; and precise slips. Ordered by severity, with the reviewer's confidence.

**Major.**

- OFF-8: at $\kappa = 0.3$ the collapse on the narrow logs is produced by the clip at zero in `offline_q` (rows clipped to all zeros tie and the greedy policy acts at random). Without the clip the same penalty gave the best policy of any arm in the reviewer's NumPy reimplementation. High confidence that removing the clip reverses the result; exact counts will differ in the book's random stream.
- OFF-7: on the uniform logs the loss from pessimism is a per-step cost of living, not a penalty "that differs from pair to pair"; a uniform charge hurts more in the rerun. High confidence in the numbers, moderately high in the reading.
- MB-4: the table under "Compounding Error" shows roughly linear growth and, on one seed in each framework, an error that falls between ten and 25 steps; the feedback loop the prose describes is the mechanism of the upright position, not of the bottom where the random episodes stay. Panel (c) of the figure draws an exponential the table contradicts. High confidence.
- PPO-14: "roughly 25,000 against 50,000 or more" for four epochs against one is read off a plot that does not show it (see "Checked for this file"); the one-epoch arm never reaches the ceiling in the budget, so no ratio is measured. High confidence.
- AC-14: the frozen-policy probe has an interior optimum in PyTorch (lambda = 0 is about twice as bad as Monte Carlo) and none in JAX (every lambda below 1 is about half the Monte Carlo error); the sentence is worded to be true of both and hides the disagreement. High confidence.
- AC-7, AC-13: the lag of actor-critic is attributed to "the young critic's bias" with no cell that measures it, and the lambda sweep is read as a trend when it separates two groups and cannot order the members of either. High confidence about what the outputs show.

**Moderate.**

- AC-15: "the best choice keeps most of the tail, $\lambda$ near $0.95$" has no training result behind it; a convention is presented as a measurement. PPO-13 inherits it.
- AC-8, PPO-15, PPO-17, SEQ-13: conclusions about seeds. The printed three or eight seeds do not carry the sentence, the 40, 64 or forty-group statistic that does is printed by no cell, and in PPO-17 the JAX output (one run of eight dead) sits directly above "about half end dead". SEQ-13 is unverified by a run.
- MB-8: `model-based.md` quotes soft actor-critic at 5,000 to 7,500 steps where `sac.md` reports 5,400 to 11,200 for what reads as the same quantity; the two use different scores and the first is from runs no cell performs. High confidence that the ranges differ.
- SAC-12: "Three changes account for the difference" between soft actor-critic and REINFORCE, which differ in at least seven things.
- DQN-8: removing the target network is presented as always catastrophic; published ablations and the chapter's own Further Reading say that unbounded divergence is rare. From memory, not checked against the papers.
- DQN-14: a claim that a maximization bias over $10^5$ tokens is "one reason why language-model post-training uses policy gradients". The reviewer does not believe this can be supported, and the bias grows only like $\sqrt{2 \ln m}$.
- SAC-7: the entropy target of $-\dim\mathcal{A}$ is called "the common choice" while the section keeps a $\log c$ term that the common implementations do not, so its $-1$ nat is their $-1.69$. Confident.
- SAC-11: the temperature loss shown is the implementations' $\log\beta$ form, presented as what the Lagrangian gives. Confident.
- PPO-7: "maximizing the bound gives monotonic improvement" holds for exact advantages and the exact maximum over states; the next paragraph moves to samples without the caveat. Confident.
- PPO-10: the controlled study of PPO and TRPO is cited for a minor finding and its main one (code-level choices decide the result) is absent from the section. Fairly confident, unverified in detail. My DFR-2 and DFR-3 concern the same study on the Further Reading page.
- REG-1, PPO-2: currency. "This is how the language models in daily use were taught to be helpful" and "PPO became the default ... to the fine-tuning of language models on human feedback" describe 2022. Fairly confident; no survey cited.
- DFR-8 (mine): the Further Reading project sets "about half the sampled responses ... on every seed you try" as a pass mark for a comparison that PPO-16 and SEQ-13 show to be confounded. Unverified by a run.

**Minor.**

- History and attribution: AC-4 (A2C credited to the A3C paper, confident), DQN-1 (the 2013 and 2015 DQN papers merged, confident), REG-13 and REG-20 (two citations whose exact content is unverified), DFR-1 (Rainbow's sixth component is not among those described, confident), DFR-6 (MOPO and MOReL described by a mechanism that fits neither exactly, fairly confident).
- Precision: AC-12 (TD(lambda) is "the same sum" only for end-of-episode updates), REG-17 (soft value iteration with a uniform reference equals the maximum-entropy one up to a constant shift), DQN-16 (the factor $\rho$ holds for jointly normal errors), SAC-4 (which bound a bounded density gives), SAC-19 and REG-19 (summaries that say something other than what was measured or printed), OFF-1 ("actor-critic" is called on-policy two sections after soft actor-critic), PPO-12 (a function `train_ac` that does not exist), PPO-22 (the autoreset behavior described is one mode of one library version).
- REG-11 is marked correctness by its reviewer but reports a correct statement and a gap in the notation table.

## What the experiments support

Collected from the section reviews. "Stored" means the committed outputs of both frameworks unless one is named.

| Section | Claim in the prose | What the stored outputs show | Finding |
|---|---|---|---|
| actor-critic | "Two more details keep only the weight different" | The arms also differ in weight normalization and in the gradient cap, which acts on 0% of updates in one arm and 28 to 29% in the other | AC-6 |
| actor-critic | Actor-critic lags because "the young critic's bias does more harm than the variance it removes" | No cell measures the bias in training. Curves coincide for 25,000 steps, then one dips | AC-7 |
| actor-critic | Monte Carlo reaches the ceiling first; 40 seeds | True of the printed seeds, but the fastest JAX run is an actor-critic run and the spread within an arm equals the gap. The 40 seeds are printed nowhere | AC-8 |
| actor-critic | "the estimators that lean on it least learn fastest" | Medians 420, 407, 450 (PyTorch) and 337, 362, 352 (JAX) for lambda 0.9, 0.95, 1: two groups, no order within | AC-13 |
| actor-critic | Probe error smallest at lambda 0.9 or 0.95, "below the error at lambda = 0" | PyTorch 1.91, 1.61, 0.83, 0.61; JAX 0.55, 0.52, 0.46, 0.48 (relative to lambda = 1). The frameworks disagree about lambda = 0 | AC-14 |
| actor-critic, ppo | Best lambda "near 0.95" on CartPole | No training run separates 0.9, 0.95 and 1 | AC-15, PPO-13 |
| ppo | Four epochs reach the ceiling in "roughly 25,000" steps, half of one epoch | Four-epoch median about 470 at 25,000, at 500 near 38,000; twenty epochs at 500 near 28,000; one epoch at 455 when the budget ends. Nothing printed | PPO-14 |
| ppo | "Going on to twenty epochs changes little" | Twenty epochs reach 500 about 10,000 steps before four; bands overlap | PPO-15 |
| ppo | "The gain comes from the batch" | No one-epoch arm at a four times larger step size, so reuse is not separated from step size | PPO-16 |
| ppo | Unclipped: "about one run in twenty holds the task and about half end dead" | Eight printed seeds: 0 hold and 4 dead (PyTorch), 1 holds and 1 dead (JAX). The 64 seeds and the two other learning rates are unseen | PPO-17 |
| dqn | Without a target network "the return never rises above the random policy's" | Six of six stored seeds, in a configuration close to a worst case; not typical of published ablations | DQN-8 |
| dqn | The estimate exceeds the bound of 86 "because a network generalizes" | Nothing separates generalization from the maximization bias; the excess grows with the number of actions | DQN-9 |
| dqn | DQN "needs more environment steps ... than PPO with four epochs" | DQN's greedy policy first scores 500 at 35,000 to 45,000 steps (30,000 to 50,000 in JAX); PPO's four-epoch batch mean reaches 500 near 38,000. Different measures, and with PPO-14 a tie | DQN-12, PPO-14 |
| dqn | Greedy policy fails "on about two seeds in five" | Twenty unseen seeds; no stored JAX seed fails | DQN-13 |
| sac | "Three changes account for the difference" from REINFORCE | The two agents differ in seven things; no run separates them | SAC-12 |
| sac | Ten updates per step: "2,400 to 4,000 ... about 2.5 times fewer" | The arm stops at 4,000 steps and the metric cannot report below 2,000: the range is censored at both ends | SAC-13 |
| sac | Learned temperature, twin critics | Neither changes the return on Pendulum (the prose says so); the entropy term is never tested | sac review, structural finding 1 |
| sac | "keeps the policy random at little cost in return" | At $-1$ nat the torque distribution is about as wide as a Gaussian with standard deviation 0.09 | SAC-16 |
| sac | Twin critics "keep the critic from overestimating" | The minimum underestimates by 15 to 40 points at the start state; measured only at 4,000 steps | SAC-17, SAC-19 |
| model-based | Errors compound along a rollout | Linear growth; PyTorch seed 0 goes 0.44, 1.91, 2.52, 1.51. Exponential growth appears only in the coverage table, mixed with missing coverage | MB-4 |
| model-based | The model is "60 to 75 points" optimistic and "its choice is right" | 73 to 77 points in PyTorch; one draw of five candidates whose predicted returns differ by four points | MB-6 |
| model-based | "The actor alone improves on every run", up to $-190$ | Median rises to about $-270$ by round 3, then flat or falling to $-340$; best actor at round 8 near $-320$ | MB-10 |
| model-based | Warm against fresh shooting; soft actor-critic's steps on this score; the MBPO comparison | No cell runs any of the three | MB-7, MB-8, MB-12 |
| offline-rl | On uniform logs a penalty "that differs from pair to pair only changes which well-supported action looks best" | Naive predictions exceed $V^*$ on 28 of 40 uniform logs; in the rerun a uniform charge hurts more | OFF-7 |
| offline-rl | At $\kappa = 0.3$ the policy "loses to the naive table on 36 of the 40 narrow logs" | Reproduced with the clip at zero; without it $\kappa = 0.3$ beats the naive table on 32 of 40 in the rerun | OFF-8 |
| regularized | "the largest Bellman residual is below $10^{-10}$" | Prints `0e+00`; a check that cannot fail | REG-19 |
| rl-sequences | "The price is useful learning" | Exact score 0.50, 0.50, 0.48, 0.40, 0.23 as the penalty grows: highest with no penalty, so the loophole is free on the section's own measure | SEQ-7 |
| rl-sequences | Four epochs "always ended with a higher exact score than one" over forty groups | One group printed; four epochs are four steps, one epoch is one | SEQ-13 |

**What a reader can trust.** Every equation, proof and hand-worked example. Every result that is an exact computation: the whole of `regularized.md` (the reward exploit, the frontier of true return against the KL budget), the first half of `offline-rl.md` (predicts against earns, the pooled count of thin pairs, the bandit), Baird's counterexample, the two-digit credit example and the sweep against the closed form in `rl-sequences.md`. Every calibration against a known bound: DQN's estimates against the ceiling of 100 with and without duplicated actions, the offline tables against $V^*$, the sign of the critic's error in soft actor-critic. The large gaps: clipped against unclipped PPO at ten times the learning rate, with against without a target network in this configuration, soft actor-critic against REINFORCE in total steps, a planner in hundreds of steps against thousands. The printed numbers themselves: reviewers found the prose's quoted values to match the stores almost everywhere they are printed.

**What a reader should not rely on.** Any sentence that gives the cause of a difference between two training curves. Any ordering of arms whose seed ranges overlap (lambda 0.9 against 0.95 against 1; four epochs against twenty; learned against fixed temperature). Any step count compared across sections. Statistics from runs "outside the notebook" where they disagree with the printed seeds. The two results on the cost of pessimism, until they are rerun without the clip.

## Arc and order

The chapter has one question at its opening (how much can one sample teach) and a better one underneath that it never states: every method here replaces fresh experience by an estimate (a critic, a stale batch, a learned reward, a model, a log, a checker), and the recurring test is how far from its data the estimate can be trusted. Five sections sit on the first question and all eight on the second (index review, structural finding 1).

- **The order should stay.** The reviewer of `regularized.md` suggests moving it next to `rl-sequences.md`, with which it forms a pair. The reviewer of `sac.md` finds that soft actor-critic "assumes `sec_regularized` heavily" and that the order is right. The second consideration decides it. What is missing is a sentence in the introduction that the chapter runs two lines, a policy line (sections 1, 2, 3, 8) ending at language models and a value and data line (4, 5, 6, 7) ending at control from logs, and which reader needs which.
- **Hand-offs are good and each is buried in a recap.** PPO's "it bounds the path, not the destination" is picked up by `regularized.md` with a closed form that proves it; DQN ends on the maximum over a continuous action; soft actor-critic on the swing; model-based on "takes away the acting". Six sections wrap the hand-off in a "We can now" paragraph (see Recurring patterns). Keep the bridge sentence and delete the rest.
- **Sections that are really two.** `model-based.md` has a seam before "Search as Policy Improvement"; the first two parts are a complete section and the third has the weaker experiment (model-based review, structural finding 2; MB-10, MB-12). `sac.md` is the longest at 6,100 words and can lose about 1,000 without a split. In `dqn.md` the first half is two arguments (the triad; correlated data and moving targets) joined by two sentences (DQN-6). In `ppo.md`, "PPO in Practice" has five jobs and should lose the effective-sample-size thread to an exercise (PPO-20).
- **Length against importance.** `rl-sequences.md` has 2,350 words of prose, less than half of any other section, on the topic the previous chapter's introduction opens with and the Language Models part builds on. Four of its subsections are one paragraph. It should grow to about 3,300 words while the others shrink.
- **Three of eight sections have no network.** `regularized.md`, `offline-rl.md` and `rl-sequences.md` are pure NumPy on tables, each importing the frameworks without using them. For the first and last, exactness is the point and the reviewers defend it. For `offline-rl.md` it hides the failure that deep offline methods exist to prevent (values at untried actions produced by generalization), and one compact experiment with the DQN of the earlier section on a frozen buffer would show it (offline review, structural finding 2).
- **Leaks.** Fixed-length segments are explained in `actor-critic.md` and again, with code, in `ppo.md`. The gradient cap is motivated over two paragraphs in `actor-critic.md` and dropped without a word in `ppo.md`. The reuse result of `ppo.md` is consumed by `dqn.md`, by the recap in `rl-sequences.md`, by PPO's own exercise 8 and by the Further Reading project. Soft actor-critic's step counts are quoted by `model-based.md` in two places with different ranges (MB-8). Direct Preference Optimization is promised by `offline-rl.md` twice and delivered by `rl-sequences.md` in four sentences (OFF-10, SEQ-11).
- **No ending.** The chapter's closing paragraph sits at the end of `rl-sequences.md` (SEQ-14) and recaps step counts. Its last sentence is the chapter's best summary and belongs in the introduction or at the head of Further Reading.

## Most important findings

1. **Make every experimental sentence claim what its cell prints, and print what the sentence needs.** The table above is the work list. The pattern is the same in six sections: a plot or three seeds, then a number or a cause. AC-6, AC-7, AC-13, AC-14, PPO-14, PPO-16, DQN-9, SAC-12, MB-4, MB-6, MB-10.
2. **Rerun the pessimism experiments without the clip at zero and rewrite "Choosing the Penalty" around what they show.** OFF-8, OFF-7; exercise 3 of that section depends on the clip.
3. **Repair the PPO reuse experiment, because four other places quote it.** Print `steps_to` per arm, extend the budget so that one epoch finishes, add the one-epoch arm at a larger step size. PPO-14, PPO-15, PPO-16; then DQN-12, SEQ-14, DFR-8.
4. **Say when model errors add and when they multiply.** Two sentences before the table, the two tables read as the stable and the unstable case, panel (c) redrawn. MB-4, MB-5.
5. **Rebuild `rl-sequences.md` at the pace of its first example.** Three ideas taught, variants to Further Reading, the Direct Preference Optimization loss derived in the text and run once, the two penalties trained side by side on the hedge. SEQ-5, SEQ-11, SEQ-12, SEQ-7, SEQ-10; rl-sequences review, structural findings 1 to 4.
6. **Define the KL divergence at first use and fix its direction once.** PPO-6, REG-7, REG-15, OFF-10. See Dependencies.
7. **Decide what the soft actor-critic experiments are for.** Promote the reward-scale experiment (exercise 3e) to test the learned temperature where a fixed one fails; reconcile "paid for staying random" with a policy held at $-1$ nat. sac review, structural findings 1 and 2; SAC-16, SAC-7, SAC-11.
8. **Join the two halves of the DQN argument, and say that the ablation is the extreme case.** State in the text what exercise 9 proves about the target network; qualify the collapse; explain why reuse of about 60 times per transition does not beat PPO on this task. DQN-6, DQN-8, DQN-12, DQN-2.
9. **Put into the sections what practitioners need and the Further Reading or the introduction now holds.** The code-level choices that decide PPO's results (PPO-10, ppo review, structural finding 2, DIDX-8); the name of the iteration in `regularized.md` (mirror descent, the natural policy gradient) and how the penalty is optimized from samples (REG-14, regularized review, structural findings 2 and 3); in-sample offline learning and fine-tuning after offline training (offline review, structural finding 3, OFF-12).
10. **Stop relying on unseen runs where the seen ones disagree.** Lead with what the printout shows, give the many-seed frequency once per paragraph, and move comparisons that no cell runs into a printed cell or an exercise. PPO-17, MB-7, MB-8, MB-12, SEQ-13, AC-8, and the "statistics from runs the reader does not see" pattern in the dqn review.

## Focus

The chapter is focused where it is strongest: `ppo.md`, `regularized.md`, `model-based.md` and `offline-rl.md` each teach one principle and name relatives only by what they change. The exceptions:

- **A survey inside `rl-sequences.md`.** RLHF, RLVR, RLOO, GRPO, VinePPO, GSPO, DAPO, Dr. GRPO, DPO and $k_3$ in 2,350 words, two of them never expanded and one used four paragraphs before it is defined (SEQ-5, SEQ-9, SEQ-10). The reviewer suspects the variants are there because the Language Models plan lists this section as their owner. Further Reading already describes the same variants in the right form (further-reading review, structural finding 3).
- **A variant list at equal weight** in `regularized.md`, "Four Consequences of the Closed Form" (regularized review, structural finding 4; REG-12).
- **Paper retellings**: the return-conditioning paragraph of `offline-rl.md` (OFF-11), the opening citation chain of `rl-sequences.md` (SEQ-1), the MBPO comparison of `model-based.md` (MB-12).
- **Side studies inside teaching sections**: Adam's second-moment memory in `actor-critic.md` (AC-5), the effective sample size in `ppo.md` (PPO-20), the dip investigation in `sac.md` (sac review, structural finding 3), shaping invariance in `regularized.md` (regularized review, structural finding 5). Each is interesting; each belongs in an exercise or in three sentences.
- **Emphasis out of proportion to the field**: Double DQN gets a full treatment while the Rainbow ablation, reported in the chapter's own Further Reading, ranks it among the least important components; one sentence in the section should say so (dqn review, structural finding 7).

## Experiments

- **Cells that only confirm, about a dozen.** `actor-critic-gae-in-one-old-function-2` (AC-10); the harmonic-mean check and the greedy evaluation in `ppo.md` (PPO-20, PPO-21); four cells in `regularized.md`, one of which prints `True` (REG-8, REG-9, REG-19); the max-bias simulation in `dqn.md`, the third demonstration of the same 1.03; the two-action cell and the normalization check in `sac.md`, the second of which tests the formula and not the class (SAC-9); the loop-against-solver check in `offline-rl.md` (OFF-4); an `np.allclose` in `rl-sequences.md`. The author's rule allows one or two per chapter. Three of them can be turned into experiments that teach: AC-10 (show the identity failing at truncation), REG-8 (make the solver the trust-region iteration, with and without the fixed-reference term), SAC-9 (integrate `log_prob_u`).
- **Tasks that cannot show the mechanism.** On CartPole bootstrapping does not help (actor-critic review, structural finding 1), neither the target network's nor Double DQN's effect reaches the policy, and replay does not beat an on-policy method (DQN-12). On Pendulum the twin critics and the learned temperature leave the return unchanged. The prose is candid each time. The sac reviewer's proposal is right: say once, in the introduction, what the two tasks can and cannot show (index review, structural finding 4), keep the experiments that measure the estimate a mechanism repairs (DQN's ceiling, the calibration in soft actor-critic), and replace the nulls by the experiments already sitting in exercises (reward scale for the temperature; noisy rewards for the bootstrap).
- **Unfair or confounded comparisons.** AC-6 (normalization and cap), PPO-16 (reuse against step size), SAC-12 (seven differences), SEQ-13 (four steps against one), DQN-12 (different measures), MB-12 (a null on a task with no room, from a setup unlike every other run).
- **Helper functions that decide a result.** The clip and the tie rule in `offline_q` and `greedy` (OFF-8) are the clear case. The offline reviewer suggests a targeted look elsewhere; the candidates the other reviews expose are the raw weights with an active cap in `actor-critic.md` (AC-6) and group standardization in `rl-sequences.md` (SEQ-8).
- **Experiments to design on the development machine**: the unclipped pessimism sweep (OFF-8); the PPO control arm and longer budget (PPO-14, PPO-16); aggregated labels in the distillation rounds (MB-10); rank correlation over 100 candidate plans (MB-6); the reward-scale arm for soft actor-critic; a consecutive-minibatch arm and a smaller step size without the target network in DQN (DQN-8, dqn review, structural finding 2); an instance in which the hedge displaces something of value (SEQ-7); a frozen-buffer DQN for the offline section.
- **The model to copy** is `regularized.md`: an exact computation for the claim and a count over 24 resampled datasets for its robustness.

## Repetition

Inside the chapter:

| What | Where | Single home |
|---|---|---|
| Fixed-length segments and the bootstrap at the cut | `actor-critic.md`, `ppo.md` | `ppo.md`, which has the code |
| The Adam, normalization and cap analysis | `actor-critic.md` text (AC-5) and its exercise 6 | The exercise; three sentences in the text |
| PPO's reuse result | `ppo.md` three times, `dqn.md`, `rl-sequences.md`, PPO exercise 8, the Further Reading project | `ppo.md`, once, after PPO-14 |
| "lambda near 0.95 is best" | `actor-critic.md` twice (AC-15), `ppo.md` (PPO-13) | Nowhere as a finding; "the setting of the PPO paper" |
| The maximization bias | `qlearning.md` of chapter 14, `dqn.md` (simulation, figure, Double DQN), `sac.md` (the $0.56\sigma$ again), a side remark in `regularized.md` (REG-18) | `dqn.md` owns the quantitative treatment; the others cite it |
| Entropy as a divergence from the uniform policy | `ppo.md`, `regularized.md` (REG-10), `sac.md` | `regularized.md`; small enough that a one-line restatement in `sac.md` is fine |
| Soft actor-critic's step counts | `sac.md`, `model-based.md` twice with other ranges (MB-8) | `sac.md` |
| "A higher update ratio exchanges computation for interaction" | three times in `sac.md` (SAC-14) | Once in the text, once in the Summary |
| "The triad is a warning, not a theorem" | three times in forty lines of `dqn.md` (DQN-4) | Once |
| Pointers to Direct Preference Optimization | `offline-rl.md` twice, `rl-sequences.md` (OFF-10, SEQ-11) | One, in `rl-sequences.md`, with content |
| DAPO, Dr. GRPO, GSPO | `rl-sequences.md` (DAPO introduced twice, SEQ-10), Further Reading | Further Reading |
| The finding that code-level choices decide PPO against TRPO | introduction, exercise 10d of `ppo.md`, Further Reading; not the section (DIDX-8) | `ppo.md` |

With other Advanced chapters: the paragraph on Adam's running average (AC-5) repeats the optimization chapter; cite it. Mirror descent is in the optimization chapter's further reading and the appendix and should be named, in three sentences, where `regularized.md` uses it (REG-14).

With chapter 14: `actor-critic.md` restates CartPole in full (AC-1), `offline-rl.md` re-checks certainty equivalence (OFF-4), and `dqn.md` narrates where each helper came from (dqn review, "narration of the library"). The opposite fault occurs once: the verifier task is recalled in one sentence from a chapter away where the reader needs three facts (SEQ-6).

With the appendix: nothing long is duplicated and nothing long needs to move. The author's rule asks for three small duplications that are missing: the definition of the KL divergence, Gibbs' inequality (one line, now cited to a whole appendix chapter, REG-7), and the meaning of "forward" and "reverse" (REG-15). Lagrange multipliers are used in `sac.md` with no word of meaning (SAC-11); one sentence in the text, a pointer to the appendix for the rest.

## Recurring patterns

**The closing recap before the Summary.** Six sections end the body with a paragraph that begins "We can now ..." or its equivalent and repeats the Summary: `actor-critic.md` and `ppo.md` (structural findings 5 and 4 of their reviews), DQN-18, SAC-18, MB-13, OFF-12. `regularized.md` closes with three forward references in five sentences, and `rl-sequences.md` with a recap of the chapter (SEQ-14). The author has asked for one ending per section. In each case the last one or two sentences are a good bridge and should be kept.

**Cells introduced by one formula.** Two formulas, by reviewer. Question then cell: six instances in `actor-critic.md` (AC-16), seven in `ppo.md` (PPO-23). Function name then its arguments: about nine in `dqn.md`, seven in `sac.md`, eleven in `model-based.md` (MB-2), nine in `offline-rl.md` (OFF-3). The second is the worse, because it turns the middle of four sections into documentation of the shared `d2l` library. Docstrings should carry the inventory; the prose should say why.

**Results from runs the reader does not see.** 40 seeds (AC-8), 64 seeds and two other learning rates (PPO-17), 24 datasets (REG-6), twenty seeds per arm in `dqn.md` (six instances) and in `sac.md`, three whole experiments in `model-based.md` (MB-7, MB-8, MB-12), forty groups (SEQ-13). The practice is the agreed one and the disclosure sentences are right. It fails where the unseen evidence outweighs the seen or contradicts it.

**Causal sentences on curves.** "because the young critic's bias ..." (AC-7), "The gain comes from the batch" (PPO-16), "because a network generalizes" (DQN-9), "The likely cause is coverage" (MB-6), "A likely reason, which we have not tested separately" (sac review, structural finding 3), "a small systematic gap: ... standardization enlarges the step" (SEQ-8). The last three are at least marked as conjecture.

**Sentences true of both frameworks' outputs when the two point in different directions.** AC-14 is the clear case; PPO-17, PPO-19 and DQN-13 are milder. A range that covers two stores is the house rule; a sentence that hides a disagreement is not.

**Exercises carry results the text uses.** The target network's effect on Baird's example (DQN-6), the actor's gradient through the squash (sac review, structural finding 5), mirror descent (REG-14), the Direct Preference Optimization loss and the forward-KL optimum (SEQ-11, SEQ-12). The exercises are uniformly praised and are long: 20 to 51 percent of the prose length per section, about 37 percent overall, against the guide's 30.

**Housekeeping.** Cell IDs that no longer match their headings in four sections (harmless, since IDs never change, but confusing in a diff). Framework imports that no cell uses in `regularized.md`, `offline-rl.md` and `rl-sequences.md`; imports spread over four cells in `model-based.md`. No TensorFlow or MXNet material anywhere. The two tabs fragment the prose only in `actor-critic.md` (one place) and in the JAX notes of `dqn.md`.

**Notation against chapter 14.** The shared table `tab_rl_notation` in `mdp.md` covers both chapters and the sections follow it for the main symbols. The changes and collisions the reviewers found, plus one of mine:

- The same divergence has two names and two symbols: `exploration.md` of chapter 14 defines $D(p \,\|\, q)$ as "the relative entropy" of two Bernoulli distributions; this chapter writes $D_{\textrm{KL}}$ and "Kullback--Leibler divergence" with no definition and no link back (my check).
- Its direction changes without notice: old policy first in `ppo.md` (PPO-6), policy first and reference second in `regularized.md` (REG-15), and the reverse of the displayed objective in the method `offline-rl.md` names as its instance (OFF-10).
- The return of a trajectory is $R(\tau)$ in chapter 14 and `ppo.md`, $r(\tau)$ in `regularized.md` (REG-2).
- The critic's step size is $\eta_V$ in `baselines.md` and $\alpha$ in `actor-critic.md` for the same network.
- $\beta$ is the softmax temperature in `exploration.md`, the KL weight and soft actor-critic's temperature; the table lists only the last two (REG-11). The Polyak rate is $\nu$ where every paper writes $\tau$, and unlike $\alpha$ the text does not say so (sac review).
- $n$ is the depth of a target or a plan in the table and a count $n(s, a)$ in `offline-rl.md`. $K$ is the group size in the table and the number of arms in `offline-rl.md`. `N` is the number of steps in a batch and then the number of environments within `ppo.md`.
- Time runs from 0 in chapter 14, `ppo.md` and `model-based.md` and from 1 in `rl-sequences.md` (SEQ-9); the value is $V(x)$ and $V(x, a_1)$ there against $V(s_t)$ elsewhere (SEQ-4).
- $u$ is the torque and then the pre-squash variable in `sac.md` (SAC-2); $H$ is entropy there and a horizon in an exercise of `dqn.md`, where the table has $T$; the entropy target is $\bar H$ in the text and $\bar{\mathcal H}$ in the figure.
- Hats on learned tables come and go in `offline-rl.md` (OFF-2); weights are $w$ in the text and $\mathbf{w}$ in two exercises of `dqn.md`; $e(s)$ for the critic's error sits beside $\pi_e$; an exercise of chapter 14 still uses $\rho_\gamma$ beside this chapter's $\rho_t$.
- New and absent from the table: $A^{\textrm{old}}$, $\bar L$, $\hat L$, $W_T$, the synchronization period $C$, $\alpha_{\textrm{c}}$, $W$, $\bar\rho$, $c$.

## Dependencies

**From Basics.** Regression with an MLP and Adam; nothing else. Adam's internals are needed only by the excursion AC-5 and the temperature prediction in `sac.md`.

**From chapter 14.** Heavy and mostly well handled: policy gradient with a baseline, reward-to-go, the group weight, Q-learning, certainty equivalence, the contraction, policy iteration, UCB, behavior cloning and DAgger, potential-based shaping, CartPole, Pendulum, the slippery lake, the verifier task, and the `d2l` classes `Batch`, `GaussianHead`, `rollout`, `evaluate`, `group_step`. Most uses restate what they need. `rl-sequences.md` does not: it has 26 cross-references in 2,350 words and cannot be read without four earlier sections fresh in mind.

**The KL divergence.** Checked: no chapter of the Basics part and no Advanced chapter before this one defines it. The classification chapter points to the appendix; the notation page lists the symbol; chapter 14 defines the Bernoulli case under another name. In this chapter it first appears inside the monotonic-improvement bound of `ppo.md` and is then used in every later section. Under the author's rule this is a small result to duplicate: a definition and one sentence of reading at the first use in `ppo.md`, the sum written out again where the proof in `regularized.md` manipulates it, one sentence that connects it to the relative entropy of `exploration.md`, and a pointer to the appendix chapter for more (PPO-6, REG-7).

**References into the appendix.** Two, both in `regularized.md`: Gibbs' inequality to `chap_mdl-information-theory` (a whole chapter; cite the section or give the line) and the direction of the divergence to `sec_mdl-fwd-vs-rev-kl`, which exists and holds the cited content. With the three small duplications listed under Repetition the chapter reads without the appendix. Today it does not quite.

**Forward references to the Language Models part.** Three unlabelled mentions (introduction, `model-based.md`, `rl-sequences.md`), none load-bearing. The load runs the other way, and I checked it. `docs/llm/language_models_part_plan.md` says "the DPO derivation is already there; 8.3 restates it" (the deferred-topics table), "The reader derived GRPO and DPO on a toy verifier task in Advanced" (the hook of its chapter 8), and "`sec_rl_sequences` derives it at `eq_dpo_reward` and exercises it on the finite instance". The source has `eq_dpo_reward` and one sentence saying that substitution into the Bradley-Terry likelihood gives a classification loss. The loss itself appears only in exercise 4(a), and no cell runs Direct Preference Optimization on anything. So the plan's statement is not yet true of the text. The cheaper way to make it true is here: the derivation is three lines from an equation the section already has, the experiment (the response that never loses a comparison grows whatever $\beta$ is) is already written as exercise 4(c), and the plan's section 8.3 is designed around not re-deriving it. The same plan lists this section as the owner of the DAPO, GSPO and Dr. GRPO variants; if they move to Further Reading, as two reviews recommend, the plan's cross-reference table must follow.

## Connection to the main text

- Stale or wrong: `train_ac` (PPO-12); "REINFORCE and actor-critic are therefore *on-policy*" against `sec_sac` (OFF-1); the label `subsec_ppo_asymmetric` and the exercise label `ex_ppo-asymmetric-bands` for material now in `rl-sequences.md` (SEQ-10).
- Missing: `dqn.md` does not point to Further Reading for DQN's successors; `sac.md` alludes to TD3's target smoothing without naming it (SAC-10); `ppo.md` does not use `steps_to`, which `actor-critic.md` saved for it; `regularized.md` never says that its penalty can be folded into the reward, which `rl-sequences.md` then relies on; exercise 4(c) of `model-based.md` is the bridge to `offline-rl.md` and has no pointer.
- Named without a label: "the Language Models part" three times; "test-time computation for language models" in `model-based.md`.
- The Further Reading page does not serve chapter 14, which has a resource list and no account of its literature; a two-paragraph topic on exploration and imitation with networks would make one page serve both (further-reading review, structural finding 4).

## What to cut, what is missing

**Cut, about 3,500 words and ten cells, none of it argument.** The six closing recaps. The function-name inventories in `dqn.md`, `sac.md`, `model-based.md` and `offline-rl.md`. The Adam excursion (AC-5) and the shared-batch variance trace (AC-9) down to sentences. The effective-sample-size thread (PPO-20). Three of the four confirming cells in `regularized.md` and the bold list of four consequences. The language-model aside (DQN-14) and one of the three demonstrations of the maximization bias. The dip investigation and the second normalization cell in `sac.md` (SAC-9). The offstage MBPO run (MB-12) and the two-curves paragraph (MB-11). The loop-against-solver check (OFF-4). The opening citation chain, the diagnosis of a 0.02 gap and the two one-paragraph subsections of `rl-sequences.md` (SEQ-1, SEQ-8, SEQ-9, SEQ-10).

**Missing.**

- A definition of the KL divergence.
- One experiment in which bootstrapping wins (actor-critic review, structural finding 1).
- The implementation choices that decide PPO's results, in the section.
- The names mirror descent and natural policy gradient, and the sample-based form of the KL penalty.
- What the target network does and does not fix, in the text of `dqn.md`; any test of replay.
- The actor's gradient written out, and an experiment in which the learned temperature matters.
- The distinction between a known model (AlphaZero) and a learned one (MuZero) in a section about learned models (MB-1); a value at the end of a plan.
- A deep offline experiment; in-sample learning; fine-tuning after offline training outside a recap.
- The Direct Preference Optimization loss and one run of it; the two KL penalties trained side by side on the hedge; an instance in which the loophole costs something.
- A route with reading paths, a statement of what the tasks can show, and an ending for the chapter.

## What is best and should set the model

- **`ppo.md`, from the two-action example to the clipped objective.** One running example with numbers (0.50 to 0.88 against 0.9975 to 0.9997), the exploding product, a four-sentence proof of the lemma, a bound stated with its constant and judged ("The constant is too large to use as it stands"), and exactness about what the clip does not guarantee.
- **`regularized.md`, "True Return against the KL Budget".** Every point an exact optimum, the fitted return proved monotone, the true return peaking and falling, and the consequence in one line: "The peak of the true return is invisible to the optimizer".
- **`actor-critic.md`, "One Transition by Hand" and the ten-state chain.** Three numbers before any theory; a closed-form bias and variance with an interior optimum and the remark that explains why it is interior.
- **`dqn.md`, Baird in ten lines, the ceiling of 100, the duplicated actions.** A bound known in advance turns "overestimation" into a fact, and a nine-fold duplication makes the bias visible on a task too small to show it.
- **`sac.md`, the stable log-determinant and the temperature prediction.** The float32 table that shows what the usual guard hides; a prediction of the trace from the optimizer's step size that the run then matches.
- **`model-based.md`, the five plans replayed in the real environment and the coverage table.** A model checked against the world, and a failure explained by one reused function.
- **`offline-rl.md`, predicts against earns, and the bandit by hand.** Exact evaluation over 40 logs; lower and upper confidence bounds as "one term with two signs".
- **`rl-sequences.md`, the two-digit credit example.** "which pushes its correct tens digit down", then the variance split. The level the whole section should hold.
- **The introduction's first two paragraphs**, and the Further Reading page's rule that each method is named with the one thing it changes.

## Where section reviews disagree

- **Position of `regularized.md`**: its reviewer suggests moving it next to the last section; the soft actor-critic reviewer depends on it where it is. Resolved above in favour of the present order.
- **The PPO step count**: the dqn review takes "roughly 25,000" from `ppo.md` as given and calls DQN's slower result the section's most surprising (DQN-12); the ppo review shows that the 25,000 is not in the stored plot (PPO-14). With the corrected figure the two methods are about level on different measures, and DQN-12 should be rewritten after PPO-14 is settled.
- **Calibration**, as noted under Scores.
- No two reviews contradict each other on a matter of fact. Where two reviewers touch the same content (fixed-length segments, the lambda default, the maximization bias, soft actor-critic's step counts, the pointers to Direct Preference Optimization) they agree.

## Checked for this file

- **PPO-14.** I rendered the stored PyTorch plot `ppo/ppo-reuse-against-environment-steps-2-1.svg`. The four-epoch median is at about 470 at 25,000 steps, passes 475 near 31,000 and reaches 500 near 38,000; the twenty-epoch median reaches 500 near 28,000; the one-epoch median ends at about 455. The reviewer's reading holds.
- **The KL divergence.** A search of all chapter sources outside the appendix and the Language Models skeletons finds the words "Kullback" or the symbol only in this chapter and later ones, the notation page, and a pointer in the classification chapter's resources; `exploration.md` defines the Bernoulli relative entropy as $D(p \,\|\, q)$.
- **Direct Preference Optimization.** The subsection of `rl-sequences.md` and the three statements of the Language Models plan were read side by side; the loss appears in the source only inside exercise 4.
- **`train_ac`**: the name occurs once in the chapter, in `ppo.md`, and is defined nowhere.
- **Soft actor-critic's ranges**: `sac.md` has "5,400 to 11,200", `model-based.md` has "5,000 to 7,500" in the text and in its table.
- Every finding ID cited here was checked against the section files.

## Could not check

- The rerun behind OFF-7 and OFF-8 is the reviewer's own reimplementation with a different random stream; I did not repeat it. It should be repeated on the development machine with the book's code before the subsection is rewritten.
- None of the proposed experiments has been run.
- Statistics over unseen seeds, by their nature.
- The points marked unverified about papers: PPO-10, DFR-2 and DFR-3 (the PPO and TRPO study), REG-13, REG-20, DQN-8 (published ablations of the target network), exercise 10 of `dqn.md`, and the currency judgments REG-1 and PPO-2.
- Rendered HTML and the slides.
