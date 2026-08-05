# Exercise Catalog: chapter_reinforcement-learning

Chapter overview (7 sections, all with `## Exercises`, 44 existing exercises). This
is the strongest-edited chapter surveyed under the parallel style review (zero
defects, zero clarity flags across all 44 items), and external comparison mostly
confirms that verdict: **all 44 existing exercises are kept**, 0 rewritten, 0
dropped. Two 2026-vintage primary sources overperformed expectations — Stanford
CS234 Winter 2026 Assignment 1 (Bellman residuals/performance bounds, effective
horizon, reward hacking, RiverSwim) and Assignment 2 (DQN, REINFORCE + baseline +
PPO, the performance-difference lemma) map almost one-to-one onto five of the
seven sections. Berkeley CS285 Fall 2020 HW1/HW2 are equally precise matches for
imitation.md and policy-gradient.md/baselines.md. Sutton & Barto 2nd ed. supplies
the deepest bench (Ch. 2, 3, 4, 6, 9, 13) but Ch. 13 has **no** dedicated
baseline/control-variate exercise — a real gap CS285/CS234 fill instead. David
Silver's Easy21 (UCL, 2015, official PDF) contributes a more principled
visit-count exploration schedule than the book's own fixed-$\epsilon$ sweep.
Coverage gaps: imitation learning has no textbook or CS234 tradition at all
(only CS285 treats it); bandit *regret* analysis (vs. bias/spike analysis) is
thin in course exercises; classical tile-coding exercises don't transfer to this
chapter's neural-network tooling. Totals: 44 kept + 9 new = 53 proposed
problems across 7 sections.

---

## chapter_reinforcement-learning/mdp.md — Markov Decision Processes

**Topic:** States, actions, transition kernels, rewards, the Markov assumption,
discounting/effective horizon, terminated vs. truncated, reward shaping.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — every item
already names a concrete derivation or measurement (state augmentation,
transition-matrix audit, effective-horizon sweep, shaping proof, random-reward
argument, MDP classification); the prior style review found zero defects and
zero clarity flags, and nothing found externally supersedes any of the six.

**External sources found:**
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Exercise
  3.1 — devise three original tasks that fit the MDP framework, testing whether
  the abstraction has been internalized beyond the gridworld instance —
  http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercise 3.2 — asks whether the MDP framework is "adequate to
  usefully represent all goal-directed learning tasks," inviting a
  counterexample — http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercises 3.15/3.16 — prove that adding a constant to every
  reward leaves relative state values unchanged in a *continuing* task, then
  show the analogous claim is false for an *episodic* task — a clean paired
  exercise on what discounting/episodicity does and does not make invariant —
  http://incompleteideas.net/book/RLbook2020.pdf
- Stanford CS234 (Winter 2026), Assignment 1, Q1 "Effect of Effective Horizon" —
  a hand-built inventory MDP where the student says whether a finite horizon
  $H$ and an infinite-horizon discount $\gamma$ can ever induce the *same*
  optimal policy, and whether that match can hold for *every* $H$ —
  https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf
- Stanford CS234 (Winter 2026), Assignment 1, Q2 "Reward Hacking" — a worked
  example (Pan, Bhatia & Steinhardt, ICLR 2022) where an AI-controlled car's
  optimal policy under a plausible proxy reward ("maximize mean velocity") is
  to never merge onto the highway at all —
  https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf

**Proposed problem set** (8 problems):
1. [conceptual] **What the state must contain.** Say which property of the MDP
   definition a single Pong frame lacks, give the smallest fix, then repeat for
   MountainCar. *Provenance:* original (section's own Ex1).
2. [short-code] **Look at a transition function.** Audit `mdp.P` row sums and
   successor counts on slippery vs. calm FrozenLake. *Provenance:* original
   (section's own Ex2).
3. [short-code] **The effective horizon.** Derive and sweep the discounted-tail
   bound against $1/(1-\gamma)$. *Provenance:* original (section's own Ex3).
4. [conceptual] **Reward design and its failure.** Prove potential-based
   shaping preserves the optimal policy; show a two-sided distance reward can
   still fail. *Provenance:* original (section's own Ex4).
5. [conceptual] **Random rewards.** Show a policy's value is unchanged by
   folding a random reward's mean into $r(s,a)$. *Provenance:* original
   (section's own Ex5).
6. [conceptual] **Which problems are not MDPs.** Classify four scenarios and
   say what state augmentation each needs. *Provenance:* original (section's
   own Ex6).
7. [short-code] **Horizon, truncation, and the optimal policy.** Build a small
   deterministic MDP (a short corridor, or this section's own FrozenLake under
   a hard step cap) and compute the optimal finite-horizon policy exactly by
   backward induction for a few horizons $H$. Find the smallest $H$ at which
   the optimal first action stops changing, then find a discount $\gamma$
   whose infinite-horizon greedy policy agrees with the horizon-$H$ policy at
   that $H$ but disagrees at $H-1$. State in one sentence why agreement at one
   $H$ does not imply agreement at every $H$. *Provenance:* adapted from
   Stanford CS234 Winter 2026 Assignment 1, Q1 (overlap med — the
   finite-horizon-vs-discount question is transplanted onto the section's own
   environment; cite CS234 on adoption).
8. [conceptual] **A proxy reward that never merges.** Read the
   Pan–Bhatia–Steinhardt highway-merge example (an AI car whose reward proxy,
   "maximize mean velocity of all cars," makes never-merging optimal). Cast it
   as a small MDP in the section's own $(\mathcal S,\mathcal A,P,r)$ notation:
   what are the states, and what does $r$ assign to merging vs. not merging?
   Explain why this is a different failure mode from the section's own Ex4 —
   there a shaped reward broke an otherwise-correct model; here the proxy
   reward is optimized perfectly and only the objective is wrong. *Provenance:*
   adapted from Stanford CS234 Winter 2026 Assignment 1, Q2 (overlap med — the
   scenario and its ICLR source are reused; the MDP formalization and contrast
   with Ex4 are original).

---

## chapter_reinforcement-learning/value-iter.md — Dynamic Programming

**Topic:** Value functions, the Bellman equations, the Bellman operator as a
$\gamma$-contraction, value iteration, policy iteration, generalized policy
iteration.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the review
found no defects or clarity issues, and every item already pairs a derivation
with a numeric check (sweep counts, per-map timings, backup counts); nothing
found externally supersedes them.

**External sources found:**
- Sutton & Barto, Exercise 4.4 — the policy-iteration pseudocode on p. 80 can
  cycle forever between equally-good policies; patch it so convergence is
  guaranteed — a direct complement to the section's own Ex6, which measures
  the *same* algorithm by backup count rather than termination —
  http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercise 4.7 (programming) — the canonical Jack's Car Rental
  problem, re-solved under a modified cost structure (a free shuttled car, a
  parking-lot penalty above 10 cars) that breaks the linear-cost assumption —
  http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercise 4.9 (programming) — the Gambler's Problem: value
  iteration at two win probabilities, checked for stability as the convergence
  threshold shrinks — http://incompleteideas.net/book/RLbook2020.pdf
- Stanford CS234 (Winter 2026), Assignment 1, Q3 "Bellman Residuals and
  Performance Bounds" — from scratch, for an *arbitrary* $V$ with Bellman
  error $\varepsilon=\|BV-V\|_\infty$, derive $V^\pi(s)\ge V^*(s)-2\varepsilon/(1-\gamma)$
  for its greedy policy, tightened to $\varepsilon/(1-\gamma)$ when $V^*\le V$ —
  extends the section's contraction proposition into a *performance* guarantee
  rather than a convergence-rate guarantee —
  https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf
- Stanford CS234 (Winter 2026), Assignment 1, Q4 "RiverSwim MDP" (programming)
  — implement value iteration and policy iteration on a six-state chain with a
  current pushing the agent backward; find the largest discount factor (to two
  decimals) at which the optimal policy stops swimming upstream, at three
  current strengths — the same "does the optimal policy do the obviously good
  thing" question the section asks of FrozenLake's shortest path, on a
  different environment —
  https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf

**Proposed problem set** (8 problems):
1. [short-code] **Cost of a sweep.** Cost a value-iteration sweep and confirm
   the $O(\log(1/\varepsilon)/\log(1/\gamma))$ sweep count empirically.
   *Provenance:* original (section's own Ex1).
2. [conceptual] **The contraction, verified.** Re-derive the certified stopping
   rule by telescoping, then show where the $\gamma=1$ proof fails.
   *Provenance:* original (section's own Ex2).
3. [short-code] **Predict, then run.** Predict $V^*(s_0)$ and policy identity
   across $\gamma$ before running value iteration. *Provenance:* original
   (section's own Ex3).
4. [short-code] **Let the ice be calm.** Compare optimal policies with slip on
   vs. off, cell by cell. *Provenance:* original (section's own Ex4).
5. [conceptual] **When $\gamma$ equals one.** Explain convergence at
   $\gamma=1$ on FrozenLake, then build a two-state counterexample that
   diverges. *Provenance:* original (section's own Ex5).
6. [extended] **Policy iteration measured in backups.** Plot both algorithms'
   sup-norm error against Bellman backups on two map sizes. *Provenance:*
   original (section's own Ex6).
7. [conceptual] **From contraction to a performance bound.** The section's
   proposition bounds $\|V_k-V^*\|_\infty$ for the *iterate* value iteration
   actually produces. Let $V$ be any vector (not necessarily an iterate) with
   Bellman error $\varepsilon=\|BV-V\|_\infty$, and $\pi$ its greedy policy.
   Insert $B^\pi V$ into the triangle inequality to show
   $\|V-V^\pi\|_\infty \le \|V-B^\pi V\|_\infty/(1-\gamma)$, then derive
   $V^\pi(s)\ge V^*(s)-2\varepsilon/(1-\gamma)$. Evaluate the bound at the
   sweep where this section's own naive stopping test first fires, and compare
   the guaranteed floor to the true gap $V^*(s_0)-V^{\pi_k}(s_0)$: how loose is
   it? *Provenance:* adapted from Stanford CS234 Winter 2026 Assignment 1, Q3
   parts (e)-(f) (overlap high — proof structure and hint reused directly;
   cite CS234 on adoption).
8. [short-code] **A current worth fighting.** Build a six-state "swim
   upstream" chain (a "swim" action that usually fails backward, a large
   terminal reward only at the far end) at three current strengths. For each,
   find the largest $\gamma$ (to two decimals, by grid search using this
   section's own value-iteration code) at which the optimal policy from the
   start state does not attempt to swim upstream, and explain the trend as the
   current strengthens. *Provenance:* adapted from Stanford CS234 Winter 2026
   Assignment 1, Q4 (overlap high — environment and discount-threshold question
   reused; re-implemented against this section's own code).

---

## chapter_reinforcement-learning/imitation.md — Learning from Demonstrations

**Topic:** Behavior cloning as supervised classification, distribution shift
between the expert's and the learner's state distribution, the
$O(\varepsilon T^2)$ compounding-error bound, DAgger.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the review
flagged this as the *shortest* exercise section in the chapter and the only one
with zero cross-references, but found no defects and no clarity problems; every
item still names a deliverable.

**External sources found:**
- Berkeley CS285 (Fall 2020), Homework 1 "Imitation Learning" — the one course
  assignment built around exactly this section's two algorithms. Structure:
  implement BC and DAgger against a provided expert on MuJoCo tasks; report BC
  reaching $\ge$30% of expert return on one task (Ant) *and* failing on another
  at matched network size/data/iterations; sweep one BC hyperparameter and plot
  the effect; plot a DAgger learning curve (return vs. iteration, with error
  bars) against horizontal expert/BC reference lines —
  https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw1.pdf
- **No textbook tradition found.** Sutton & Barto's 2nd edition has no chapter
  on imitation learning, behavior cloning, or DAgger — the topic postdates the
  book's scope, so there is no S&B exercise to compare against.
- **No CS234 tradition found.** Stanford CS234 (Winter 2026)'s three
  assignments cover tabular MDP planning, Q-learning/function
  approximation/policy gradients, and offline RL/RLHF — imitation learning is
  not a graded topic in this iteration of the course.

**Proposed problem set** (7 problems):
1. [conceptual] **Where the bound comes from.** Reproduce the $O(\varepsilon T^2)$
   argument and identify where expert distribution is swapped for learner
   distribution. *Provenance:* original (section's own Ex1).
2. [short-code] **How many demonstrations.** Sweep $N$ and find where the clone
   matches the expert on-distribution while still losing return. *Provenance:*
   original (section's own Ex2).
3. [short-code] **Where the errors are.** Plot per-state error against expert
   visitation frequency. *Provenance:* original (section's own Ex3).
4. [short-code] **DAgger's budget.** Compare DAgger and BC at matched expert
   query counts. *Provenance:* original (section's own Ex4).
5. [conceptual] **SFT is behavior cloning.** Map BC's bound onto language-model
   fine-tuning degradation over long generations. *Provenance:* original
   (section's own Ex5).
6. [extended] **A bad expert.** Corrupt 10% of demonstrations and compare BC's
   and DAgger's degradation. *Provenance:* original (section's own Ex6).
7. [short-code] **A matched-capacity BC failure.** CS285's HW1 requires a
   *paired* result — real fractional expert performance on one task, clear
   failure on a second — at identical network size, dataset size, and training
   iterations, a fairness discipline the section's own Ex2 (varying $N$ on one
   task) does not enforce. Pick two of the section's ten demonstration sets
   that visibly differ in how spread out the expert's state visitation is
   (reuse the Ex3 plot to choose them), fit BC to both at identical
   hyperparameters, and report training accuracy alongside deployed return for
   both. Does the training-accuracy gap predict the deployed-return gap, or
   does one task fail primarily from distribution shift despite a similar fit?
   *Provenance:* adapted from Berkeley CS285 Fall 2020 HW1, Section 1 (overlap
   med — the matched-capacity, one-works/one-doesn't comparison is reused; the
   ten-dataset selection and fit-vs-shift diagnostic are original).

---

## chapter_reinforcement-learning/qlearning.md — Temporal-Difference Learning and Exploration

**Topic:** The sampled Bellman backup, the TD error, tabular Q-learning
convergence, off-policy learning and maximization bias, multi-armed bandits,
regret, $\epsilon$-greedy vs. UCB vs. Thompson sampling.

**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the review's
one formatting defect in this chapter (Ex4's inline "(i)/(ii)/(iii)" lettering)
is cosmetic, not a clarity problem, and out of this catalog's scope; every item
still names a concrete measurement.

**External sources found:**
- Sutton & Barto, Exercises 6.11/6.12 — a compact conceptual pair: why is
  Q-learning off-policy at all, and, if action selection were made greedy,
  would Q-learning and Sarsa become *the same* algorithm (same action
  selections, same weight updates)? — http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercises 6.9/6.10 (programming) — Windy Gridworld extended
  with eight King's-move actions (plus an optional ninth "stay" action) and,
  separately, with stochastic wind — a different TD-control environment for the
  same "does more structure help" question the section asks of its exploration
  schedules — http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercises 2.7/2.8 — the "Unbiased Constant-Step-Size Trick"
  (a step size avoiding constant-$\alpha$'s initial bias while keeping
  non-stationary adaptivity) and "UCB Spikes" (explain Figure 2.4's spike at
  the 11th step, *and* why it decreases afterward) — the closest S&B analogue
  to the section's own step-size and UCB-sensitivity exercises —
  http://incompleteideas.net/book/RLbook2020.pdf
- David Silver, UCL Reinforcement Learning (2015), "Easy21" assignment —
  Monte-Carlo control and Sarsa($\lambda$) on a custom blackjack-like game,
  using a visit-count exploration schedule
  $\epsilon_t = N_0/(N_0+N(s_t))$, $N_0=100$, rather than a fixed or
  manually-annealed one — more principled than the section's own fixed-$\epsilon$
  sweep — https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/easy21-assignment.pdf
- Stanford CS234 (Winter 2026), Assignment 2, Q1 "Deep $Q$-Networks (DQN)"
  (written) — given DQN's pseudocode (replay buffer, target network, minibatch
  updates), say exactly which lines must change to recover tabular Q-learning
  — the section's own summary text already promises this connection but never
  makes the reader work through it —
  https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf

**Proposed problem set** (8 problems):
1. [conceptual] **Why greedy from the start cannot work.** Trace a
   fixed-tie-break trajectory from a zero table, and explain why coverage
   alone doesn't guarantee a good table. *Provenance:* original (section's own
   Ex1).
2. [short-code] **The exploration schedule.** Predict then measure first
   success and final greedy performance across five $\epsilon$ schedules.
   *Provenance:* original (section's own Ex2).
3. [conceptual] **Step sizes under a finite budget.** Check Robbins-Monro
   conditions, then find by experiment a constant beating both decaying
   schedules. *Provenance:* original (section's own Ex3).
4. [short-code] **The double-sampling counterexample.** Build a 3-state MDP,
   solve the population objective in closed form, and show semi-gradient TD
   converges to $Q^*$ instead. *Provenance:* original (section's own Ex4).
5. [short-code] **A like-for-like comparison with value iteration.** Plot
   $\|\hat Q-Q^*\|_\infty$ against environment steps vs. Bellman backups.
   *Provenance:* original (section's own Ex5).
6. [short-code] **Regret, measured properly.** Re-plot per-seed cumulative
   regret log-log, then sweep UCB's exploration coefficient. *Provenance:*
   original (section's own Ex6).
7. [extended] **A harder map.** Solve the 8x8 map and reconcile the
   random-walk hitting time with measured episodes-to-first-success.
   *Provenance:* original (section's own Ex7).
8. [conceptual] **From tabular Q-learning to DQN, one line at a time.** Write
   DQN's update as pseudocode with a replay buffer $\mathcal D$, target weights
   $\theta^-$, and a minibatch of transitions. Say exactly which lines must be
   deleted or changed — and why each is unneeded — to recover this section's
   own `q_learning` update: what happens to the replay buffer when every
   update uses only the transition just seen, what happens to the target
   network when the table is its own exact target, and what happens to the
   minibatch loop when the batch size is one. Then run this section's tabular
   update once with a "replay buffer" of size one and confirm it reproduces the
   original numbers exactly. *Provenance:* adapted from Stanford CS234 Winter
   2026 Assignment 2, Q1.1(a) (overlap high — the reduction question is
   reused directly; the empirical check against this section's own code is
   original).

---

## chapter_reinforcement-learning/policy-gradient.md — Policy Gradient

**Topic:** Direct policy parameterization, the log-derivative/score-function
identity, the REINFORCE estimator, bias/variance of the estimator, the
on-policy data requirement.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the review
found this file's sweeps exemplary (every one names its axis, its metric, and
its win condition); nothing external supersedes any item.

**External sources found:**
- Sutton & Barto, Exercise 13.1 — use the gridworld's own dynamics to write an
  exact symbolic expression for the optimal probability of the "right" action
  under the section's softmax parameterization — a closed-form counterpart to
  the section's own numerically-verified two-state enumeration (Ex2) —
  http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercise 13.3 — derive the softmax-with-linear-preferences
  eligibility vector $\nabla\ln\pi(a|s,\theta)=x(s,a)-\sum_b\pi(b|s,\theta)x(s,b)$
  from first principles — essentially the section's own Ex1 score identity, in
  the book's original linear-feature notation —
  http://incompleteideas.net/book/RLbook2020.pdf
- Berkeley CS285 (Fall 2020), Homework 2 "Policy Gradients" — its own
  six-experiment design (trajectory-centric vs. reward-to-go weighting,
  with/without advantage standardization, small vs. large batch, on CartPole;
  then a batch-size/learning-rate search on InvertedPendulum for the smallest
  batch and largest rate solving the task in under 100 iterations) is close
  kin to the section's own Ex3 grid, but organized as a $2\times2\times2$
  ablation rather than a single sweep —
  https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw2.pdf
- Stanford CS234 (Winter 2026), Assignment 2, Q3 "Distributions Induced by a
  Policy" (written) — derives the *performance-difference lemma*,
  $V^\pi(s_0)-V^{\pi'}(s_0)=\frac{1}{1-\gamma}\mathbb E_{s\sim d^\pi}\mathbb
  E_{a\sim\pi'}[A^{\pi'}(s,a)]$, relating the trajectory-based objective this
  section builds to a state-occupancy identity used throughout later
  policy-optimization theory (TRPO/PPO surrogate bounds) — a foundational
  identity the section never derives —
  https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf

**Proposed problem set** (7 problems):
1. [conceptual] **The score sums to zero.** Show the score identity sums to
   zero over actions and conclude an update can only redistribute probability.
   *Provenance:* original (section's own Ex1).
2. [short-code] **Unbiased, where enumeration is exact.** Confirm the
   $1/\sqrt n$ error rate on an enumerable two-state MDP. *Provenance:*
   original (section's own Ex2).
3. [short-code] **Batch size and learning rate.** Grid-sweep batch size and
   learning rate, reporting episodes to a return threshold. *Provenance:*
   original (section's own Ex3).
4. [conceptual] **Sparse reward, seen from the estimator.** Compare
   drop-out-from-the-estimator failure under two reward conventions.
   *Provenance:* original (section's own Ex4).
5. [short-code] **Breaking the on-policy rule on purpose.** Reuse batches for
   $k$ gradient steps and log the importance ratio's drift. *Provenance:*
   original (section's own Ex5).
6. [conceptual] **The discount that implementations drop.** Identify the
   different objective estimated when $\gamma^t$ is dropped from the score
   weight. *Provenance:* original (section's own Ex6).
7. [conceptual] **The performance-difference lemma.** For two policies
   $\pi,\pi'$ on this section's own MDP, define the discounted state-occupancy
   $d^\pi(s)=(1-\gamma)\sum_t\gamma^t\Pr(s_t=s\mid\pi)$ and the advantage
   $A^{\pi'}(s,a)=Q^{\pi'}(s,a)-V^{\pi'}(s)$ from :numref:`sec_valueiter`. By
   adding and subtracting $\sum_t\gamma^{t+1}V^{\pi'}(s_{t+1})$ inside
   $V^\pi(s_0)-V^{\pi'}(s_0)$ and applying the tower property of expectation,
   prove
   $$V^\pi(s_0)-V^{\pi'}(s_0)=\frac{1}{1-\gamma}\,\mathbb E_{s\sim d^\pi}\big[\mathbb E_{a\sim\pi}[A^{\pi'}(s,a)]\big].$$
   Then, on the section's tabular softmax policy, evaluate both sides
   numerically before and after one REINFORCE update, estimating $d^\pi$ from
   rollouts: does the identity hold to sampling error? Explain in one sentence
   why a positive expected advantage under the *new* policy's own state
   distribution does not by itself certify improvement, and connect this to
   why :numref:`sec_ppo`'s surrogate clips the policy ratio rather than
   trusting this identity directly. *Provenance:* adapted from Stanford CS234
   Winter 2026 Assignment 2, Q3(c)-(d) (overlap high — the identity, proof
   strategy, and hint chain are reused directly and should be cited on
   adoption; the numerical verification is original).

---

## chapter_reinforcement-learning/baselines.md — Variance Reduction for Policy Gradients

**Topic:** The zero-mean score identity, reward-to-go, constant/control-variate
baselines, the variance-optimal baseline, centering vs. scaling, the
GRPO/Dr.-GRPO connection.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the review
called this file "a strong example of the house style" with zero clarity
flags; every exercise already names both the quantity to measure and the
comparison to make.

**External sources found:**
- **No dedicated exercise found in Sutton & Barto.** Chapter 13's four
  end-of-chapter exercises (13.1, 13.3, 13.4, 13.5) cover policy
  parameterization and the score function, but none touches baselines,
  control variates, or variance reduction — the book's own in-text "REINFORCE
  with Baseline" section (13.4) states the zero-mean identity but is never
  turned into a numbered exercise. A genuine gap in the canonical source for
  exactly this section's topic.
- Berkeley CS285 (Fall 2020), Homework 2, §2.2–2.3 and §6 — implements
  reward-to-go, a learned state-dependent value baseline, and advantage
  normalization as three separately toggleable pieces (`-rtg`,
  `--nn_baseline`, standardize-by-default), then requires a 4-way ablation on
  HalfCheetah (baseline on/off $\times$ reward-to-go on/off) — a factorial
  design the section's own five-estimator comparison does not run —
  https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw2.pdf
- Berkeley CS285 (Fall 2020), Homework 2, §8 (bonus) — implement GAE-$\lambda$
  for advantage estimation and test whether it speeds training on a MuJoCo
  task; the section's own "Reward-to-Go: One Scan" slide already previews
  this ("GAE will be this same scan, run on TD errors with factor
  $\gamma\lambda$") but the exercise set never builds it —
  https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw2.pdf
- Stanford CS234 (Winter 2026), Assignment 2, §2.3 "Advantage Normalization" —
  states the same centering-is-a-baseline / scaling-is-a-step-size distinction
  as the section's own Ex6, corroborating rather than displacing it (overlap
  with the existing exercises is high enough that no new problem is proposed
  from this source) —
  https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf

**Proposed problem set** (7 problems):
1. [short-code] **The step-size confound, quantified.** Rescale the normalized
   arm's learning rate by its measured $1/\sigma$ and check whether the
   ordering survives. *Provenance:* original (section's own Ex1).
2. [short-code] **Measure the variance you claim to reduce.** Freeze $\theta$,
   draw 200 batches, and compare each estimator's covariance trace to its
   learning-curve ordering. *Provenance:* original (section's own Ex2).
3. [conceptual] **The variance-optimal baseline.** Derive the weighted-average
   optimal constant baseline and connect it to the control-variate
   coefficient. *Provenance:* original (section's own Ex3).
4. [short-code] **Baseline step size.** Sweep $\alpha_V$ and diagnose the
   failure at $\alpha_V=1$. *Provenance:* original (section's own Ex4).
5. [short-code] **The group-relative baseline in two lines.** Implement the
   GRPO-style normalized weight and test batch size one. *Provenance:*
   original (section's own Ex5).
6. [conceptual] **What dividing by sigma costs.** Compare $\sigma$-weighting
   across two prompts of different success rate — the Dr. GRPO objection.
   *Provenance:* original (section's own Ex6).
7. [short-code] **A $\lambda$ dial between reward-to-go and one-step TD.**
   Implement the $\gamma\lambda$-weighted scan over TD errors
   $\delta_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t)$ that the section's own
   "Reward-to-Go: One Scan" slide names but never builds:
   $\hat A_t^{GAE(\lambda)}=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$. Confirm
   $\lambda=0$ reproduces (up to the learned baseline's error) the one-step
   actor-critic advantage and $\lambda=1$ reproduces the section's own
   reward-to-go-minus-baseline estimator. Add this as a sixth arm to the
   section's five-estimator variance comparison at
   $\lambda\in\{0,0.5,0.9,1\}$ and report where it falls in the variance
   ordering and in trained return. *Provenance:* adapted from Berkeley CS285
   Fall 2020 HW2, §8 bonus problem (overlap med — GAE-$\lambda$ and its two
   limiting cases are reused, citing Schulman et al. 2016 on adoption;
   grafting it onto this section's own five-estimator comparison as a sixth
   arm is original).

---

## chapter_reinforcement-learning/deep-rl.md — Function Approximation in Reinforcement Learning

**Topic:** Replacing tabular representations with neural networks, continuous
states (CartPole) and continuous actions (Pendulum, Gaussian policies), the
coupling/generalization effect of shared parameters, score-function vs.
pathwise gradient estimators.

**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the review
found no defects and no clarity flags in this file; it is also the explicit
chapter/book boundary, and Ex6 already makes a deliberate forward
cross-reference to the next chapter.

**External sources found:**
- Sutton & Barto, Exercise 9.1 — show that tabular methods are a special case
  of linear function approximation, and say what the feature vectors would be
  — a formal proof of the claim the section's own "A Table Is a Linear
  Network" slide makes only in prose (`nn.Embedding(16,4)` **is** a linear
  layer on one-hot states) — http://incompleteideas.net/book/RLbook2020.pdf
- Sutton & Barto, Exercise 13.4 — derive the Gaussian-policy eligibility
  vector (score function) in closed form for both the mean and
  log-standard-deviation parameters — the exact object the section's own "Two
  Gradients of One Expectation" slide computes only for a scalar toy case —
  http://incompleteideas.net/book/RLbook2020.pdf
- OpenAI Spinning Up in Deep RL, Exercises page, Problem Set 2, Exercise 2.2
  "Silent Bug in DDPG" — a deliberately introduced shape bug (the critic's
  output isn't squeezed from `[batch, 1]` to `[batch]`) that silently corrupts
  a deterministic, pathwise-gradient actor-critic; the student must diagnose
  it from training curves alone before checking a reference diff — a "debug
  this" exercise style the section's own set does not use —
  https://spinningup.openai.com/en/latest/spinningup/exercises.html
- OpenAI Spinning Up in Deep RL, Exercises page, Problem Set 1, Exercise 1.3
  "Computation Graph for TD3" — fill in TD3's loss functions and intermediate
  calculations, another pathwise-gradient (differentiable-critic)
  implementation task adjacent to the section's score-vs-pathwise comparison
  — https://spinningup.openai.com/en/latest/spinningup/exercises.html
- Stanford CS234 (Winter 2026), Assignment 2, §2 "Policy Gradient Methods"
  (coding + written) — the same REINFORCE-with-baseline architecture on the
  same two environments (CartPole, Pendulum) this section uses, with
  published sanity-check reward targets (baseline REINFORCE $\approx$700 by
  iteration 20 on Pendulum) that can cross-check this section's own
  three-seed curves —
  https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf
- **No transfer found** for classical feature-engineering exercises: Sutton &
  Barto's tile-coding exercise (9.5, choosing a step size for 98 overlapping
  tilings of a 7-D state) has no counterpart the reader can attempt with this
  section's tools, since the section replaces feature engineering with a
  learned MLP entirely — a real gap in the book's coverage this section's own
  tooling cannot close.

**Proposed problem set** (8 problems):
1. [conceptual] **A surrogate loss is not a performance metric.** Show the
   surrogate's gradient equals $-\hat u$ at fixed advantages, then construct a
   case where the loss and the return diverge. *Provenance:* original
   (section's own Ex1).
2. [short-code] **How small can the policy be.** Sweep hidden width on
   CartPole and report the smallest network that still balances the pole.
   *Provenance:* original (section's own Ex2).
3. [short-code] **Which return to plot.** Add the discounted return alongside
   the undiscounted one and construct a task where they diverge.
   *Provenance:* original (section's own Ex3).
4. [short-code] **Batch size at a fixed episode budget.** Sweep batch size
   with total episodes held fixed and compare per-episode learning speed.
   *Provenance:* original (section's own Ex4).
5. [short-code] **Advantage scale.** Replace normalized advantage with raw
   reward-to-go and compare degradation across tasks. *Provenance:* original
   (section's own Ex5).
6. [conceptual] **Benefits and costs of generalization.** Argue why a batch
   dominated by near-vertical states can worsen the policy at large angles.
   *Provenance:* original (section's own Ex6).
7. [short-code] **What the spread costs.** Fix the Gaussian head's $\sigma$ at
   three values and relate the failures to exploration and score variance.
   *Provenance:* original (section's own Ex7).
8. [short-code] **A shape bug in the pathwise gradient.** Reintroduce, into
   this section's own critic network, an analogous silent bug to Spinning
   Up's DDPG exercise: forget to squeeze the critic's output so it returns
   shape `[batch, 1]` instead of `[batch]`, then run the section's own
   score-vs-pathwise comparison cell unchanged. Broadcasting will silently turn
   the pathwise gradient's per-example terms into a `[batch, batch]`
   outer-product sum instead of a `[batch]`-length vector before the mean.
   From the corrupted variance ratio and gradient values alone (not from
   reading the diff), diagnose that the bug is a shape error rather than a
   real modeling problem, and say what property of the correct pathwise
   estimator (the ratio being a small constant regardless of batch size) tips
   you off. *Provenance:* adapted from OpenAI Spinning Up in Deep RL, Problem
   Set 2, Exercise 2.2 (overlap high on the nature of the bug and the
   diagnose-before-you-peek methodology; transplanted from DDPG onto this
   section's own Gaussian-policy/critic code, which is original).
