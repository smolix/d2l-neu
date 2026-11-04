# Exercise Catalog: chapter_reinforcement-learning

Chapter overview (regenerated 2026-11-04 from the ch. 14 rebuild). The chapter
has eight sections, in the order mdp, value-iter, qlearning, exploration,
imitation, policy-gradient, baselines, deep-rl; the introduction (`index.md`)
has no exercises. The sections hold 71 exercises, 57 of them with an
`:exlabel:`. Each entry below lists one exercise in the order of the file: its
tag and difficulty marks as printed (`[code]`, `[extended]`, ● and ●●; no tag
means pencil and paper), its label or "no label", its topic, and its source.

Sources name where the exercise comes from in this book and, where one exists,
its external basis. "Oct-2026 set" is the exercise set on `main` before the
rebuild; "rebuild" is the 2026-11 ch. 14 rebuild (plan items W10 and 14.x in
the book-pass notes). The earlier survey of this chapter (Sutton & Barto 2nd
ed., Stanford CS234 Winter 2026, Berkeley CS285 Fall 2020, David Silver's UCL
course, OpenAI Spinning Up) is summarized per section under "Survey sources";
its proposed problem sets predate the rebuild and are in the git history of
this file.

Moves in the rebuild: exploration and bandits left qlearning.md for the new
exploration.md, taking the exploration-schedule, UCB-regret and
deep-exploration exercises with their labels; the potential-based shaping
exercise and the optimal-policy part of the time-limits exercise moved from
mdp.md to value-iter.md with the shaping experiment. Labels of moved exercises
were kept, so one exercise in value-iter.md and two in exploration.md carry
the label prefix of their old section. One label was renamed, that of
qlearning's "Samples against backups" (the old label had no references).

---

## chapter_reinforcement-learning/mdp.md — Markov Decision Processes

**Topic:** States, actions and the transition kernel, return, discount and
effective horizon, termination and truncation, the choice of state and partial
observability, reward design, text generation and the verifier task as MDPs.

**Exercises:** 8 (7 from the Oct-2026 set, 1 restored; the shaping exercise
moved out to value-iter.md; 4 labels new in the rebuild).

1. **What the state must contain.** `ex_mdp-state-must-contain` — the smallest
   addition that makes a battery-limited robot, a recommender with fatigue and
   chess with threefold repetition Markov, and whether the agent observes it.
   *Source:* restored in the rebuild (W10) from the pre-October set, with new
   scenarios; original.
2. [code] **Episodes from the kernel.** `ex_mdp-kernel-predicts-episodes` —
   which state-action pairs of `from_gym` merge outcomes, and propagating
   $d^\pi_t$ under the random policy to predict episode statistics.
   *Source:* Oct-2026 set; original.
3. **Expected rewards.** `ex_mdp-expected-rewards` — transition rewards
   $\rho(s,a,s')$ and expected rewards $r(s,a)$ give the same expected
   return, also for random rewards; which FrozenLake episodes pay.
   *Source:* Oct-2026 set; original.
4. **Discount as termination.** `ex_mdp-discount-as-termination` — the bound
   $\ln(1/p)/\ln(1/\gamma) \le \ln(1/p)/(1-\gamma)$ behind the effective
   horizon, and $\gamma$ as a per-step survival probability.
   *Source:* Oct-2026 set; original.
5. **Time limits.** `ex_mdp-time-limits` — a time limit that is part of the
   task makes $(s, h)$, not $s$, the Markov state. *Source:* Oct-2026 set
   (its optimal-policy part moved to value-iter.md); after
   Pardo et al. 2018.
6. **Actions without consequences.** no label — with an action-independent
   kernel the reward-greedy policy is optimal for every $\gamma$; a two-state
   MDP where it is optimal only for small $\gamma$. *Source:* Oct-2026 set;
   original.
7. **Belief states.** ● `ex_mdp-belief-states` — the Bayes update of a POMDP
   belief and the belief as a Markov state. *Source:* Oct-2026 set; after
   Kaelbling, Littman & Cassandra 1998 and Bakker 2001.
8. [extended] **Planning horizon and model error.** ●● `ex_mdp-planning-discount`
   — the bias of planning with $\gamma' < \gamma$ and its trade against
   estimation error in a learned model, using value iteration and exact
   policy evaluation. *Source:* Oct-2026 set; after Jiang et al. 2015.

**Survey sources:** Sutton & Barto Ex. 3.1, 3.2 (tasks that fit or escape the
MDP framework), 3.15/3.16 (constant reward shifts in continuing and episodic
tasks), http://incompleteideas.net/book/RLbook2020.pdf; CS234 W2026 A1 Q1
(finite horizon against discount) and Q2 (reward hacking in highway merging),
https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf.

---

## chapter_reinforcement-learning/value-iter.md — Dynamic Programming

**Topic:** Policy values and the Bellman equations, value iteration and its
contraction, stopping rules, policy evaluation as a linear system, policy
iteration, the optimal policy on slippery ice, reward shaping.

**Exercises:** 12 (10 from the Oct-2026 set, 2 moved in from mdp.md; 6 labels
new in the rebuild).

1. [code] **Policy evaluation as a linear system.**
   `ex_value-iter-linear-system` — invertibility of $I - \gamma P_\pi$, the
   Neumann series $M_\pi$, and its rows as discounted state-visitation
   distributions. *Source:* Oct-2026 set; original.
2. **Stopping certificate.** `ex_value-iter-certificate` — the a posteriori
   certificate derived from the contraction, and an a priori sweep count
   evaluated on the slippery lake. *Source:* Oct-2026 set; original.
3. **The undiscounted lake.** `ex_value-iter-undiscounted-lake` — what the
   contraction proof shows at $\gamma = 1$, non-unique fixed points, and
   monotone convergence from zero to the probability of reaching the goal.
   *Source:* Oct-2026 set; original.
4. **Ties in policy iteration.** `ex_value-iter-ties` — why a stopping test
   on policy equality can cycle between tied actions, and why the test "the
   greedy step gains nothing" terminates. *Source:* Oct-2026 set; adapted
   from Sutton & Barto Ex. 4.4.
5. [code] **Discount on calm and slippery ice.** `ex_value-iter-discount` —
   how the optimal policy depends on $\gamma$ on both maps, and the success
   probability of each optimal policy. *Source:* Oct-2026 set; original.
6. [code] **In-place sweeps.** ● no label — in-place sweeps are a
   $\gamma$-contraction for every state order; sweep counts by order and map.
   *Source:* Oct-2026 set; adapted from Sutton & Barto §4.1 and §4.5.
7. [code] **Measured contraction rate.** ● `ex_value-iter-contraction-rate`
   — the asymptotic rate as the spectral radius of $\gamma P_{\pi^*}$ on the
   nonterminal states, at $\gamma = 0.95$ and $0.999$. *Source:* Oct-2026
   set; original.
8. [code] **Improvement, quantified.** ● `ex_value-iter-improvement-quantified`
   — the matrix form of the performance-difference lemma and the gain of a
   greedy step, checked numerically. *Source:* Oct-2026 set; after
   Kakade & Langford 2002 (the lemma CS234 W2026 A2 Q3 also derives).
9. [code] **When the greedy policy is optimal.** ● `ex_value-iter-greedy-optimal`
   — the bound $V^\pi \ge V^* - 2\gamma e/(1-\gamma)$ for a greedy policy and
   why the greedy policy is optimal long before the certificate fires.
   *Source:* Oct-2026 set; after Singh & Yee 1994 (close to CS234 W2026 A1
   Q3).
10. [code] **Time limits under the optimal policy.** `ex_value-iter-time-limits`
    — propagating $d^\pi_t$ under `pi_star` to split episodes into goal, hole
    and truncation, and the horizon-dependent optimal command.
    *Source:* moved in the rebuild from mdp.md (part (a) of its time-limits
    exercise); original.
11. [code] **Potential-based shaping.** ● `ex_mdp-reward-design-its-failure`
    — constant rewards as potentials, per-step charges that change the
    optimal policy, and shaping that preserves it. *Source:* moved in the
    rebuild from mdp.md (label kept); adapted from Sutton & Barto Ex. 3.15
    and 3.16, with the shaping proposition of Ng, Harada & Russell 1999.
12. [extended] **Backup budgets.** ●● `ex_value-iter-backup-budgets` —
    modified policy iteration, measuring value iteration, policy iteration and
    the interpolation between them in backups. *Source:* Oct-2026 set; after
    Puterman 1994.

**Survey sources:** Sutton & Barto Ex. 4.4 (policy iteration that cycles),
4.7 (Jack's car rental with nonlinear costs), 4.9 (the gambler's problem),
http://incompleteideas.net/book/RLbook2020.pdf; CS234 W2026 A1 Q3 (Bellman
residuals and performance bounds) and Q4 (RiverSwim discount thresholds),
https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf.

---

## chapter_reinforcement-learning/qlearning.md — Temporal-Difference Learning

**Topic:** The sampled Bellman backup, double sampling and the semi-gradient,
step sizes, terminal masking, Q-learning on the slippery lake, off-policy
targets, SARSA and maximization bias, planning in a learned model.

**Exercises:** 8 (7 from the Oct-2026 set, 1 new; three others moved out to
exploration.md; 4 labels new in the rebuild, 1 renamed).

1. **Batch TD against Monte Carlo.** `ex_qlearning-batch-td-monte-carlo` —
   batch TD(0) and Monte Carlo on an eight-episode batch of a two-state
   reward process: which estimate each converges to and why.
   *Source:* Oct-2026 set; adapted from Sutton & Barto Example 6.4.
2. **Truncation treated as termination.** `ex_qlearning-truncation-termination`
   — the fixed point of a TD update that masks the bootstrap at truncation,
   against $1/(1-\gamma)$. *Source:* Oct-2026 set; original.
3. **Greedy SARSA and Q-learning.** `ex_qlearning-greedy-sarsa` — whether
   SARSA and Q-learning coincide under greedy action selection, with a
   transition of the calm lake where they differ. *Source:* Oct-2026 set;
   adapted from Sutton & Barto Ex. 6.12.
4. [code] **The double-sampling counterexample.** ●
   `ex_qlearning-double-sampling-counterexample` — on a three-state MDP the
   minimizer of the sampled squared TD error is not $Q^*$, while the
   semi-gradient update converges to it. *Source:* Oct-2026 set; the
   A-split example of Sutton & Barto §11.5.
5. [code] **Step sizes under a finite budget.** ● `ex_qlearning-step-sizes`
   — the stationary variance of a constant step size, the bias and variance
   of $1/(1 + c\,n)$ schedules, and the best constant for a fixed budget.
   *Source:* Oct-2026 set; original.
6. [code] **Samples against backups.** ● `ex_qlearning-samples-backups` —
   Q-learning's error against environment steps beside value iteration's
   against backups, and the certainty-equivalence model on the same data.
   *Source:* Oct-2026 set (label renamed in the rebuild); original.
7. [code] **Maximization bias in a two-state task.** ●
   `ex_qlearning-maximization-bias` — Q-learning's preference for a bad
   action with many noisy successors, and double Q-learning.
   *Source:* Oct-2026 set; adapted from Sutton & Barto Example 6.7, with
   van Hasselt 2010.
8. [code] **SARSA and Q-learning on the cliff.** ● `ex_qlearning-cliff-walking`
   — `sarsa` against `q_learning` on `CliffWalking-v1`: the learned paths,
   the reward collected while learning, and the limit as $\epsilon$ falls to
   zero. *Source:* new in the rebuild (14.3-C1); adapted from Sutton & Barto
   Example 6.6.

**Survey sources:** Sutton & Barto Ex. 6.11/6.12 (why Q-learning is
off-policy; greedy SARSA), 6.9/6.10 (windy gridworld with king's moves and
stochastic wind), http://incompleteideas.net/book/RLbook2020.pdf; CS234
W2026 A2 Q1 (which lines of DQN reduce to tabular Q-learning),
https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf.

---

## chapter_reinforcement-learning/exploration.md — Exploration

**Topic:** Multi-armed bandits and regret, greedy and $\epsilon$-greedy rules,
upper confidence bounds and Thompson sampling, the regret lower bound,
exploration in MDPs (the chain, count-based bonuses, coverage on the lake),
identical-reward groups on the verifier task.

**Exercises:** 6 (3 moved in from qlearning.md, 3 new in the rebuild).

1. [code] **The exploration schedule.** ●● `ex_qlearning-exploration-schedule`
   — a zero table behaves identically under every $\epsilon$ until the first
   success, what greedy selection does after it, and five schedules compared
   on slippery ice. *Source:* moved in the rebuild from qlearning.md (label
   kept); original.
2. **Logarithmic regret of UCB.** ●● `ex_qlearning-ucb-logarithmic-regret` —
   deriving the UCB1 regret bound from Hoeffding's inequality and evaluating
   it on the section's arms. *Source:* moved in the rebuild from
   qlearning.md (label kept); Theorem 1 of
   Auer, Cesa-Bianchi & Fischer 2002.
3. **Thompson's posterior.** ● `ex_exploration-thompson-posterior` — the
   Beta posterior of a Bernoulli arm, the probability of pulling an untried
   arm, and Thompson sampling as probability matching. *Source:* new in the
   rebuild (W10); original.
4. [code] **Long horizons.** ●● `ex_exploration-long-horizons` — a
   vectorized bandit runner, median regret of UCB and Thompson sampling to
   $10^5$ pulls on log axes, and whether $\kappa = \sqrt{2}$ catches up.
   *Source:* new in the rebuild; original.
5. [extended] **Deep exploration.** ●● `ex_exploration-deep-exploration` —
   the exponential cost of random exploration on the chain, the polynomial
   cost of optimistic initialization, and both learners on the slippery
   $8 \times 8$ lake. *Source:* moved in the rebuild from qlearning.md and
   shortened (W10, new label); original.
6. **Groups that teach nothing.** ● `ex_exploration-identical-groups` — the
   probability $(1-p)^K + p^K$ of an identical group, the group size needed
   for a mixed group, and the share of a training set that gives no signal.
   *Source:* new in the rebuild (W10); original, complementing baselines
   exercise 6.

**Survey sources:** Sutton & Barto Ex. 2.7 (unbiased constant-step-size
trick), 2.8 (UCB spikes), http://incompleteideas.net/book/RLbook2020.pdf;
David Silver, UCL RL (2015), Easy21 (visit-count schedule
$\epsilon_t = N_0/(N_0 + N(s_t))$),
https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/easy21-assignment.pdf.

---

## chapter_reinforcement-learning/imitation.md — Learning from Demonstrations

**Topic:** Behavior cloning as classification, distribution shift and missing
coverage, quadratic against linear error growth, DAgger, supervised
fine-tuning as behavior cloning.

**Exercises:** 9 (7 from the Oct-2026 set, 1 replacing an Oct-2026 exercise,
1 new; 1 label new in the rebuild).

1. **The chain in closed form.** `ex_imitation-chain-closed-form` — the
   closed-form loss of the cloned and the recovering imitator on the chain,
   checked against the printed values. *Source:* Oct-2026 set (part (b)
   rewritten in the rebuild); original.
2. [code] **Where the clone's mass goes.** no label — total variation between
   the expert's and the clone's state distributions over time, and which
   holes take the clone's lost successes. *Source:* Oct-2026 set (takes the
   total-variation material moved out of the text); original.
3. **Fine-tuning as behavior cloning.** no label — why a fine-tuned language
   model's error rate grows along its own responses, and which part of the
   compounding-error proposition then applies. *Source:* Oct-2026 set;
   original.
4. **Two valid paths.** no label — squared-error and categorical fits to
   bimodal demonstrations, and diffusion policies that execute action
   sequences. *Source:* Oct-2026 set; after Chi et al. 2023.
5. [code] **Demonstrations needed for coverage.** ●
   `ex_imitation-demonstrations-coverage` — critical states of the slippery
   lake and the number of demonstrations needed to cover them.
   *Source:* Oct-2026 set (part 3 trimmed in the rebuild); original.
6. [code] **DAgger on a longer path.** ● `ex_imitation-dagger-s-budget` —
   the experiment of the text repeated on the $8 \times 8$ calm lake: the
   clone's loss after leaving the path against the share that labels off the
   path recover, DAgger against cloning at equal label counts, and the
   distinct states DAgger labels. *Source:* rebuild,
   replacing the Oct-2026 "DAgger's budget" (label kept); original.
7. [code] **A noisy expert.** no label — labels corrupted with probability
   0.1: what cross-entropy converges to, and the clone against DAgger under
   the noise. *Source:* new in the rebuild (14.5-S4); original, after a
   pre-October "bad expert" item.
8. **DAgger's guarantee.** ● `ex_imitation-dagger-guarantee` — the
   no-regret reduction behind DAgger's linear bound, with mixing
   probabilities $\beta_i$. *Source:* Oct-2026 set (part 3 trimmed in the
   rebuild); after Ross, Gordon & Bagnell 2011.
9. **Rewards behind demonstrations.** ● no label — reward ambiguity in
   inverse RL (shaping and the zero reward), and what maximum-entropy IRL
   can and cannot identify. *Source:* Oct-2026 set; after
   Ziebart et al. 2008.

**Survey sources:** Berkeley CS285 Fall 2020 HW1 (behavior cloning and DAgger
on MuJoCo, a matched-capacity success and failure, a DAgger learning curve),
https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw1.pdf.
Sutton & Barto and CS234 W2026 have no imitation-learning exercises.

---

## chapter_reinforcement-learning/policy-gradient.md — Policy Gradient

**Topic:** Softmax policies and the score function, the log-derivative trick
and the cancellation of the dynamics, REINFORCE on the verifier task and the
lake, the policy gradient theorem, the on-policy requirement, unbiasedness
and variance, plateaus.

**Exercises:** 9 (8 from the Oct-2026 set, 1 new; 2 labels new in the
rebuild).

1. **Parameter-dependent dynamics.** `ex_policy-gradient-parameter-dependent-dynamics`
   — the term that survives when the kernel depends on $\theta$, in a
   recommender whose users tire of an item. *Source:* Oct-2026 set; original.
2. [code] **Sparse rewards.** `ex_policy-gradient-sparse-rewards` — the
   success probability $p_0$ of the uniform policy computed exactly, batches
   without a success, and the batch size that brings the noise down to the
   mean. *Source:* Oct-2026 set; original.
3. **Shifted rewards.** no label — subtracting one from every reward keeps
   the mean but changes the noise, for which policies it helps; on FrozenLake
   the shift as a per-step charge, and which failed episodes it penalizes
   least.
   *Source:* Oct-2026 set; original.
4. [code] **The theorem in a table.** ● `ex_policy-gradient-theorem-table` —
   the tabular softmax gradient $\rho_\gamma(s)\pi(b \mid s)A(s,b)$ from the
   theorem, checked against `g_exact`, and why the gradient at the uniform
   policy is one third as long on slippery ice. *Source:* Oct-2026 set (part
   (c) new in the rebuild); original.
5. [code] **Reusing a batch.** ● `ex_policy-gradient-breaking-policy-rule-purpose`
   — $k$ uncorrected steps per batch: speed, drift of the probability ratio,
   and where it breaks. *Source:* Oct-2026 set; original.
6. [code] **Testing unbiasedness.** ● `ex_policy-gradient-testing-unbiasedness`
   — the running-average test over $M$ estimates, and a biased estimator that
   levels off. *Source:* Oct-2026 set; original.
7. **Softmax plateaus.** ● `ex_policy-gradient-softmax-plateaus` — the
   logistic plateau of a two-action softmax, the steps to escape it under
   gradient ascent and under Adam, and the near-deterministic final policies
   of slippery-lake runs. *Source:* Oct-2026 set; original.
8. [code] **The dropped discount.** ●●
   `ex_policy-gradient-discount-that-implementations-drop` — the update field
   without $\gamma^t$ is not a gradient; checked on FrozenLake.
   *Source:* Oct-2026 set; after Nota & Thomas 2020.
9. **The Gaussian step of the figure.** no label — $J(\mu)$, the score and
   the variance of the twelve-sample estimate for the policy and reward of
   the score-ascent figure. *Source:* new in the rebuild; original.

**Survey sources:** Sutton & Barto Ex. 13.1 (exact optimal probability on a
gridworld), 13.3 (the softmax eligibility vector),
http://incompleteideas.net/book/RLbook2020.pdf; CS285 Fall 2020 HW2
(reward-to-go and standardization ablations, batch-size and learning-rate
search), https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw2.pdf;
CS234 W2026 A2 Q3 (the performance-difference lemma),
https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf.

---

## chapter_reinforcement-learning/baselines.md — Baselines and Variance Reduction

**Topic:** The zero-mean identity, reward-to-go and causality, baselines as
control variates, the learned value baseline, centering and normalization,
leave-one-out and group baselines, the loss divisor, five estimators compared
at a frozen policy and in training at matched step sizes.

**Exercises:** 11 (10 from the Oct-2026 set, 1 new; 4 labels new in the
rebuild).

1. **Admissible baselines.** `ex_baselines-admissible` — which of five
   candidates (time index, past rewards, remaining steps, a function of
   $s_{t+1}$, $\hat Q(s_t, a_t)$) leave the gradient unbiased.
   *Source:* Oct-2026 set; original.
2. **Baselines on stale data.** no label — a constant baseline biases the
   score once the action comes from an older policy, and the probability
   ratio restores the zero mean. *Source:* Oct-2026 set; original.
3. **Optimal constant baseline.** `ex_baselines-optimal-constant-baseline` —
   the score-weighted mean $b^*$ minimizes the trace of the covariance, as
   the optimal control-variate coefficient. *Source:* Oct-2026 set;
   original.
4. [code] **Baseline step size and timing.** ● no label — sweeping $\eta_V$
   (`eta_v`) and fitting the baseline before or after the policy step.
   *Source:* Oct-2026 set; original.
5. [code] **Bias of batch centering.** ● `ex_baselines-batch-centering` — the
   $(n-1)/n$ factor derived directly, and the bias of step-wise centering by
   enumeration.
   *Source:* Oct-2026 set; original.
6. **The $\sigma$-normalized objective.** ● `ex_baselines-sigma-objective` —
   which prompts GRPO's division by $\sigma$ favors and the objective it
   ascends. *Source:* Oct-2026 set; after Liu et al. 2025 (Dr. GRPO).
7. [code] **Variance ordering across policies.** ● `ex_baselines-variance-ordering`
   — the frozen-policy table repeated at four points of training, the
   estimators without a baseline near the maximum, and batches without a
   success. *Source:* Oct-2026 set; original.
8. [code] **The step-size confound.** ● `ex_baselines-step-size-confound-quantified`
   — the normalized arm as centering at step size $\eta/\sigma$, logged per
   batch, and the exact value baseline against reward-to-go at fixed $\eta$.
   *Source:* Oct-2026 set; original.
9. [code] **Optimal state baseline.** ●● `ex_baselines-optimal-state-baseline`
   — the per-state optimal baseline for the tabular softmax and when it is
   also optimal for the episodic estimate. *Source:* Oct-2026 set; after
   Greensmith, Bartlett & Baxter 2004.
10. [extended] **Length bias at small scale.** ●● no label — per-response
    length normalization favors short correct and long incorrect responses,
    in a toy model and on FrozenLake. *Source:* Oct-2026 set; after Liu et
    al. 2025.
11. [code] **Reward-to-go, taken apart.** ● `ex_baselines-reward-to-go-taken-apart`
    — the mean of reward-to-go without $\gamma^t$, its exact tilt from the
    gradient, and when dropping past rewards raises the noise.
    *Source:* new in the rebuild (W10, the $\gamma^t$ problem); after Thomas
    2014 and Nota & Thomas 2020.

**Survey sources:** Sutton & Barto has no baseline exercise in ch. 13 (the
gap CS285 and CS234 fill); CS285 Fall 2020 HW2 §2.2–2.3 and §6 (baseline and
reward-to-go ablation), §8 (GAE-$\lambda$),
https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw2.pdf;
CS234 W2026 A2 §2.3 (advantage normalization),
https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf.

---

## chapter_reinforcement-learning/deep-rl.md — Function Approximation in Reinforcement Learning

**Topic:** Tables as linear networks, policy and value networks, value
regression, bootstrapping at time limits, REINFORCE on CartPole, the surrogate
loss, Gaussian policies on Pendulum, score-function against pathwise
gradients, coupling between states.

**Exercises:** 8 (all from the Oct-2026 set; 4 labels new in the rebuild).

1. **Discounting at the time limit.** `ex_deep-rl-time-limit` — discounted
   and undiscounted CartPole returns rank episodes alike, the reward-to-go
   near the limit without a correction, the bootstrapped target, and why
   Pendulum needs the correction more. *Source:* Oct-2026 set (rewritten for
   the truncation bootstrap); original.
2. [code] **Gaussian score.** `ex_deep-rl-gaussian-score` — the score of the
   mean and log standard deviation, $\mathbb{E}[Q(a)(z^2-1)] = \sigma^2 Q''$,
   unbiasedness under torque clipping, and a state-dependent standard
   deviation (`state_std=True`) on Pendulum. *Source:* Oct-2026 set (part
   (d) new in the rebuild); adapted from Sutton & Barto Ex. 13.4.
3. [code] **Discontinuous rewards.** no label — the pathwise estimator
   returns zero for a threshold reward; smooth steps of growing sharpness.
   *Source:* Oct-2026 set; original.
4. **Surrogate loss value.** `ex_deep-rl-surrogate-value` — with normalized
   weights the surrogate equals $-s\,c$, so its value is bounded and says
   little about the return. *Source:* Oct-2026 set; original.
5. [code] **Normalization under Adam.** ● `ex_deep-rl-normalization-adam` —
   Adam's invariance to a constant gradient scale, its memory of past
   scales, and centering without the division on CartPole and Pendulum.
   *Source:* Oct-2026 set; original.
6. **Score versus pathwise variance.** ● `ex_deep-rl-score-pathwise-variance`
   — the exact single-sample variances $15\sigma^2 + 14 + 1/\sigma^2$ and
   $4\sigma^2$, and what a baseline does to the score form.
   *Source:* Oct-2026 set; original.
7. [code] **Tangent kernel.** ● `ex_deep-rl-tangent-kernel` — one SGD step
   moves the value at other states by the tangent kernel; computed and
   compared with the coupling probe. *Source:* Oct-2026 set; original.
8. [extended] **Parameter-space noise.** ●● no label — the evolution-strategies
   gradient and its antithetic estimate, its noise in the dimension, and
   random search with a linear policy on CartPole. *Source:* Oct-2026 set;
   after Salimans et al. 2017 and Mania, Guy & Recht 2018.

**Survey sources:** Sutton & Barto Ex. 9.1 (tabular methods as linear
function approximation), 13.4 (Gaussian eligibility vector), 9.5 (tile
coding, no counterpart with this section's tools),
http://incompleteideas.net/book/RLbook2020.pdf; Spinning Up Problem Set 2
Ex. 2.2 (a silent shape bug in DDPG) and Problem Set 1 Ex. 1.3 (the TD3
computation graph), https://spinningup.openai.com/en/latest/spinningup/exercises.html;
CS234 W2026 A2 §2 (REINFORCE with a baseline on CartPole and Pendulum),
https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf.
