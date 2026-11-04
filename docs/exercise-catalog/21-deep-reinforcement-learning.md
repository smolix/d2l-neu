# Exercise Catalog: chapter_deep-reinforcement-learning

Chapter overview (regenerated 2026-11-04 from the ch. 15 rebuild). The chapter
has nine sections, in the order actor-critic, ppo, regularized, dqn, sac,
model-based, offline-rl, rl-sequences, further-reading; the introduction
(`index.md`) and the Further Reading page have no exercises (the Further
Reading page ends with one project, a multi-epoch GRPO loop, outside the
exercise numbering). The eight sections hold 75 exercises, 57 of them with
an `:exlabel:`. Each entry below lists one exercise in the order of the
file: its tag and difficulty marks as printed (`[code]`, `[extended]`, ● and
●●; no tag means pencil and paper), its label or "no label", its topic, and
its source.

Sources name where the exercise comes from in this book and, where one exists,
its external basis. "Oct-2026 set" is the exercise set on `main` before the
rebuild (commit 3f9e9ba6); "rebuild" is the 2026-11 ch. 15 rebuild (plan items
W10 and 15.x in the book-pass notes). The earlier survey of this chapter
(Berkeley CS285 Spring 2026 and Fall 2020, Stanford CS234 Winter 2026,
Stanford CS224R Spring 2026, Sutton & Barto 2nd ed., OpenAI Spinning Up,
CleanRL, the RLHF book) is summarized per section under "Survey sources"; its
proposed problem sets predate the rebuild and are in the git history of this
file.

Moves in the rebuild: two ppo exercises (asymmetric bands, the
length-normalized ratio) moved to rl-sequences with the text they belong to,
keeping the label `ex_ppo-asymmetric-bands`; worked cases left the exercises
for the bodies (actor-critic's interior-optimum part, rl-sequences' two-digit
credit computation, offline-rl's two-armed bandit by hand); model-based.md is
new. Labels of moved exercises were kept, so one exercise in rl-sequences
carries the label prefix of its old section.

---

## chapter_deep-reinforcement-learning/actor-critic.md — Actor-Critic Methods and Multi-Step Returns

**Topic:** The TD error as a bootstrapped advantage, the one-step actor-critic
update, $n$-step returns, the $\lambda$-return and GAE as a backward scan, and
the bias-variance trade of the $\lambda$ dial as measured on CartPole.

**Exercises:** 9 (8 from the Oct-2026 set, several trimmed; the traces problem
dropped; 1 new; 4 labels new in the rebuild).

1. **Bias of the bootstrapped weight.** `ex_actor-critic-why-baseline-argument-does`
   — why the zero-mean baseline identity does not cover $\delta_t$; the split
   $\delta_t = \delta^\pi_t + \gamma e(s_{t+1}) - e(s_t)$ and the bias it leaves.
   *Source:* Oct-2026 set; original.
2. [code] **Optimal bootstrap depth.** `ex_actor-critic-optimal-bootstrap-depth`
   — bias and variance of the depth-$n$ target on the chain example, simulated
   against the formulas; a constant critic error makes the error monotone.
   *Source:* Oct-2026 set, shortened (its derivation of the optimum moved into
   the text); original.
3. [code] **Critic passes and the order of the updates.**
   `ex_actor-critic-critic-passes` — `critic_steps` in $\{1, 5, 20\}$, a target
   fixed per batch against one recomputed per pass, and the TD error computed
   before the critic fit.
   *Source:* Oct-2026 set, extended with the order of the updates; corroborated
   by CS285 Fall 2020 HW3 (target-update sweep on CartPole).
4. **The $\lambda$-return recursion.** `ex_actor-critic-lambda-return-recursion`
   — the recursion of the GAE target, its form at termination and at a
   time-limit cut.
   *Source:* Oct-2026 set; Sutton & Barto Ex. 12.1.
5. [extended] ● **Fully online.** `ex_actor-critic-fully-online` — the
   per-transition update with no batch, at a matched number of environment
   steps.
   *Source:* Oct-2026 set; original.
6. [code] ● **Gradient norms and the cap.** `ex_actor-critic-gradient-norms` —
   the pre-clip actor gradient norms of the two arms, a variance model of the
   norm, and what the clip does inside Adam.
   *Source:* Oct-2026 set, with the gradient-norm cell that left the text;
   original.
7. [code] ● **Correlation of the two weights.**
   `ex_actor-critic-uncorrelated-td-errors` — with an exact critic the
   correlation of $\delta_t$ and the Monte Carlo advantage is the square root of
   the variance ratio; measured along a run.
   *Source:* Oct-2026 set, with the correlation cell that left the text; its
   part (e) removed; original.
8. [code] ● **Bias of GAE.** `ex_actor-critic-bias-of-gae` — the bias of the
   GAE weight for every $\lambda$ in terms of the critic's error, checked on
   the frozen probe.
   *Source:* Oct-2026 set (its first two parts moved into the text as the
   proposition on the interior optimum); original.
9. **A critic on a one-step task.** `ex_actor-critic-verifier-critic` — on the
   verifier task the TD error is the reward minus a per-prompt baseline; the
   exact critic against the group-centered weight.
   *Source:* new in the rebuild (15.1 plan); original.

**Survey sources:** CS285 Spring 2026 HW2 (GAE and a $\lambda$ sweep on
LunarLander, the section's own framing); CS285 Fall 2020 HW3 (critic-update
sweep on CartPole); Sutton & Barto Ex. 12.1 (the $\lambda$-return recursion)
and Ex. 12.2 (the half-life of $(\gamma\lambda)^l$, used on the deck's check
slide); Spinning Up Exercise 2.1 (value-function fitting in TRPO).

---

## chapter_deep-reinforcement-learning/ppo.md — Trust Regions and Proximal Policy Optimization

**Topic:** Parameter space against policy space, importance ratios and the
per-step surrogate, the performance-difference lemma and the monotonic
improvement bound, the clipped objective, reuse against environment steps, the
clip ablation and the training diagnostics.

**Exercises:** 10 (8 from the Oct-2026 set; 2 moved out to rl-sequences; 2
new in the rebuild; 3 labels new).

1. ● **Step limits for one logit.** `ex_ppo-step-limits` — a KL trust region,
   PPO's clip and a cap on the parameter step on a two-action policy; the
   second-order expansion of the divergence.
   *Source:* Oct-2026 set; CS234 Winter 2026 A2 §2.4 (when the clipped gradient
   vanishes).
2. **Exploding product of ratios.** `ex_ppo-exploding-product` — mean one and
   variance $(1 + \chi^2)^T - 1$ of the trajectory ratio.
   *Source:* Oct-2026 set; original.
3. [code] **Entropy of the dead seeds.** `ex_ppo-dead-seed-entropy` — the
   stored entropy of the unclipped arm, split into dead and surviving seeds.
   *Source:* Oct-2026 set; original.
4. ● **What the diagnostics estimate.** `ex_ppo-diagnostics-estimate` — the
   $k_1$ and $k_3$ estimators of the KL and the clip fraction as estimates.
   *Source:* Oct-2026 set; original.
5. **Bootstrapping at the cut.** no label — what treating each cut of the
   vectorized rectangle as a termination does to GAE.
   *Source:* Oct-2026 set, made self-contained; original.
6. [code] ● **KL early stopping.** no label — Spinning Up's stopping rule on
   the approximate KL in place of the clip.
   *Source:* Oct-2026 set, made independent of the implementation-details table
   that moved to the Further Reading page; Spinning Up PPO.
7. [code] **The width of the band.** no label — $\epsilon_{\textrm{clip}}$
   swept from $0.05$ to $10$ in the twenty-epoch ablation.
   *Source:* new in the rebuild (15.2 plan); original.
8. [code] ●● **The standard recipe.** `ex_ppo-standard-recipe` — vectorized
   collection, four epochs of minibatches and the standard learning rate,
   against the arms of the section on the step axis.
   *Source:* new in the rebuild (decision Q3b, experiment E11); CleanRL's
   `ppo.py` and the RL Baselines3 Zoo defaults.
9. ●● **The monotonic-improvement bound.** `ex_ppo-monotonic-bound` — the
   coupling proof of the TRPO bound.
   *Source:* Oct-2026 set; CS234 Winter 2026 A2 §3(d) (the
   performance-difference lemma).
10. [extended] ●● **TRPO from the pieces.** `ex_ppo-trpo` — conjugate gradient
    with Fisher-vector products and a line search in place of `ppo_epochs`.
    *Source:* Oct-2026 set; Spinning Up TRPO.

**Survey sources:** CS234 Winter 2026 A2 (the clipped objective, §2.4; the
performance-difference lemma, §3(d)); Spinning Up's TRPO and PPO pages;
CleanRL's PPO documentation; the Deep RL Bootcamp lecture 5 (no posed
problems).

---

## chapter_deep-reinforcement-learning/regularized.md — KL-Regularized Policy Optimization

**Topic:** A reward fitted from comparisons (Bradley–Terry), what comparisons
cannot see (shaping), reward hacking, the KL-regularized optimum and its
frontier, trust region against penalty, the soft backup, and the true return
against the KL budget.

**Exercises:** 10 (all from the Oct-2026 set, several trimmed and their
symbols renamed; 2 labels new in the rebuild).

1. **Identifiability and scale.** `ex_regularized-identifiability` — rewards
   are identified up to a function of the prompt, and up to a constant per
   connected component of the comparison graph.
   *Source:* Oct-2026 set; original.
2. [code] **Trajectory divergence.** `ex_regularized-trajectory-divergence` —
   the trajectory KL as a sum of per-step KLs, without a model; under
   geometric termination.
   *Source:* Oct-2026 set; original.
3. [code] **Tilting by the unregularized value.** no label — tilting
   $\pi_{\textrm{ref}}$ by $Q^*$ instead of $Q_\beta$ traces a worse curve.
   *Source:* Oct-2026 set; original.
4. [code] **Soft backup on the lake.** `ex_regularized-soft-backup-transplanted`
   — the soft backup is a $\gamma$-contraction; soft value iteration on the
   slippery lake.
   *Source:* Oct-2026 set; original.
5. [extended] ● **Neural reward models.** `ex_regularized-neural-reward-models`
   — an MLP reward model extrapolates to unsupported pairs by its inductive
   bias.
   *Source:* Oct-2026 set, shortened; the RLHF book's reward-modeling
   experiments.
6. [code] ● **Exploited reward errors.** no label — errors added to unsupported
   and supported entries of $\hat r$, and how the peak of the true return
   moves.
   *Source:* Oct-2026 set; original.
7. ● **Slope of the frontier.** `ex_regularized-frontier-is-concave` — the
   frontier has slope $\beta$ and is concave; best-of-$N$'s divergence bound.
   *Source:* Oct-2026 set, with the best-of-$N$ bound added; original.
8. ● **A moving reference.** `ex_regularized-moving-reference` — a trust region
   alone converges to the argmax; with a fixed-reference penalty it converges
   to the tilted optimum.
   *Source:* Oct-2026 set (its first part moved into the text); original.
9. [code] ● **Forward-KL penalty.** `ex_regularized-forward-kl-penalty` — the
   optimum under the forward divergence and its polynomial suppression of
   suboptimal actions.
   *Source:* Oct-2026 set; original.
10. [code] ●● **Error modulo shaping.** `ex_regularized-error-modulo-shaping` —
    shaping leaves the soft optimum unchanged; the reward error measured modulo
    the best potential.
    *Source:* Oct-2026 set, part (d) dropped; Ng, Harada and Russell 1999.

**Survey sources:** CS224R Spring 2026 HW2 (a frozen reference actor in PPO
fine-tuning); the RLHF book's reward-modeling and policy-gradient chapters
(Bradley–Terry fitting, the Dr. GRPO critique); Sutton & Barto has no
KL-regularized or maximum-entropy exercise.

---

## chapter_deep-reinforcement-learning/dqn.md — Deep Q-Networks

**Topic:** The semi-gradient update and the deadly triad (Baird), replay and
the target network, the ablation, greedy evaluation and the value audit,
maximization bias, Double DQN and duplicated actions.

**Exercises:** 10 (all from the Oct-2026 set, some trimmed or made
self-contained; 2 labels new in the rebuild).

1. [code] **Baird under on-policy weighting.** `ex_dqn-baird-on-policy` — the
   eigenvalues of the expected update with off- and on-policy state weights.
   *Source:* Oct-2026 set; Sutton & Barto Ex. 11.3.
2. **Huber loss and norm clipping.** no label — Huber regression as clipped
   squared-loss regression, and what the norm clip adds.
   *Source:* Oct-2026 set, trimmed; original.
3. **Terminated, not done.** `ex_dqn-terminated-not-done` — storing
   `terminated or truncated` at CartPole's time limit, and a network with no
   clock.
   *Source:* Oct-2026 set; original.
4. [code] **Maximization bias in closed form.** `ex_dqn-max-bias-closed-form` —
   the bias of a maximum of $m$ correlated normal estimates.
   *Source:* Oct-2026 set; original.
5. [code] ● **Replay reuse and buffer capacity.** `ex_dqn-shrink-buffer` — how
   often each stored transition is sampled, and smaller buffers.
   *Source:* Oct-2026 set, made self-contained; original.
6. [code] ● **The sync period.** `ex_dqn-sync-period` — the sync bound
   $(1-\gamma^n)/(1-\gamma)$ and runs at five sync periods.
   *Source:* Oct-2026 set; CS285 Spring 2026 HW3 §2.
7. ● **n-step targets from replay.** `ex_dqn-nstep-replay` — what the buffer
   must store for $n$-step targets, termination inside the window, and the
   off-policy bias.
   *Source:* Oct-2026 set; CS234 Winter 2026 A2 ($n$-step component).
8. ● **Prioritized sampling.** no label — the importance weights that make
   prioritized replay unbiased, and the variance-minimizing distribution.
   *Source:* Oct-2026 set; Schaul et al. 2016.
9. [code] ●● **Baird with a target network.** `ex_dqn-baird-target-network` —
   a target copy refreshed every $k$ sweeps on Baird's counterexample.
   *Source:* Oct-2026 set; original.
10. [extended] ●● **LayerNorm instead of a target network.** `ex_dqn-layernorm`
    — LayerNorm in `make_qnet` without a target network, on CartPole and on
    Baird.
    *Source:* Oct-2026 set; Gallici et al. 2025 (PQN).

**Survey sources:** CS285 Spring 2026 HW3 §2 (DQN on CartPole and
LunarLander, the learning-rate and target-update questions) and §2.5 (Double
DQN); Sutton & Barto Ex. 11.3 (Baird); CS234 Winter 2026 A2 (Nature DQN with
an $n$-step component); CleanRL's DQN page (mean Q as an overestimation
signal). Spinning Up has no DQN.

---
## chapter_deep-reinforcement-learning/sac.md — Soft Actor-Critic

**Topic:** The maximum-entropy objective as the uniform-reference case of the
KL-regularized objective, two actions by hand, soft policy evaluation and
improvement, the squashed Gaussian and its log-determinant, twin critics and
Polyak targets, automatic temperature, return against environment steps, the
update ratio, and critic calibration.

**Exercises:** 10 (9 from the Oct-2026 set; the temperature-autotuning problem
removed because the learned temperature is now in the main code, its
reward-scale part folded into the quadratic-critic problem; 1 new).

1. **Entropy in the target.** `ex_sac-entropy-in-target` — the entropy term in
   the critic target values entropy collected at later states; a two-branch
   example.
   *Source:* Oct-2026 set; original.
2. [code] **Variance of the sampled bracket.** `ex_sac-bracket-variance` — the
   one-sample estimate of the soft value on the two-action example, and its
   variance across temperatures.
   *Source:* Oct-2026 set; original.
3. [code] **Quadratic critic.** `ex_sac-quadratic-critic` — the entropy-optimal
   Gaussian under a quadratic critic; rescaling the reward equals rescaling
   $\beta$, measured with a fixed and a learned temperature.
   *Source:* Oct-2026 set, extended by the reward-scale part of the former
   autotuning problem; Haarnoja et al. 2018 (reward-scale sensitivity of SAC
   with a fixed temperature).
4. [code] **Entropy of the squashed policy.** `ex_sac-squashed-entropy` — the
   entropy of $2\tanh u$ and where it peaks in $\sigma$.
   *Source:* Oct-2026 set; Spinning Up's SAC page (state-dependent log-std).
5. **Off-policy without ratios.** `ex_sac-off-policy-without-ratios` — why
   fresh actions in the target remove the need for importance ratios.
   *Source:* Oct-2026 set, trimmed; original.
6. ● **Projection direction.** `ex_sac-projection-direction` — reverse- and
   forward-KL projections of the tilted distribution onto the policy family.
   *Source:* Oct-2026 set; original.
7. [code] ● **Restoring force of the Jacobian.** `ex_sac-jacobian-force` — the
   log-determinant term pulls the pre-squash action back from saturation.
   *Source:* Oct-2026 set; original.
8. [code] ● **Twin-critic disagreement.** `ex_sac-twin-disagreement` — the
   minimum of two critics as mean minus half the distance, measured along a
   run.
   *Source:* Oct-2026 set; CS285 Spring 2026 HW3 §3.6 (single against
   clipped double critics).
9. [code] **Update ratio.** `ex_sac-update-ratio` — the earliest possible
   crossing, and twenty updates per environment step with twin and single
   critics.
   *Source:* new in the rebuild (decision Q4); Chen et al. 2021 (REDQ).
10. [extended] ● **Discrete soft actor-critic.** `ex_sac-discrete` — exact
    expectations over a finite action set, on CartPole.
    *Source:* Oct-2026 set; Christodoulou 2019.

**Survey sources:** CS285 Spring 2026 HW3 §3.5 (automatic temperature tuning,
now in the section's code) and §3.6 (single against clipped double critics on
Hopper); Spinning Up's SAC and TD3 pages and its Exercise 1.3 (TD3's
computation graph); CleanRL's `sac_continuous_action` page.

---
## chapter_deep-reinforcement-learning/model-based.md — Learned Models and Planning

**Topic:** A dynamics-and-reward model fitted by regression from a few hundred
Pendulum steps, the compounding of its error and its coverage, model-predictive
control by warm-started random shooting, the cost of a real step in model
evaluations, search seeded by the actor as a policy-improvement operator and
its distillation into the actor, and imagined transitions as training data.

**Exercises:** 7 (all new in the rebuild, the section being new; 7 labels).

1. [code] **Depth of the plan.** `ex_model-based-horizon` — predict, then run,
   the planner at depths 10 and 50 against 25, read with the error tables.
   *Source:* new in the rebuild (15.6 plan); original.
2. [code] **How much random data.** `ex_model-based-data` — models fitted to
   one, two and five random episodes for the two seeds whose random episode
   stayed at the bottom; coverage against error.
   *Source:* new in the rebuild; original.
3. [code] **The cross-entropy method.** `ex_model-based-cem` — five rounds of
   refitting the sampling distribution to the best candidates, against warm-
   started random shooting at its cost in model evaluations.
   *Source:* new in the rebuild; Rubinstein and Kroese 2004; Chua et al. 2018
   (CEM planning in PETS).
4. [code] ● **Disagreement as a trust signal.** `ex_model-based-ensemble` — an
   ensemble of three models; disagreement against error, and plans scored with
   a disagreement penalty.
   *Source:* new in the rebuild; Chua et al. 2018 (PETS).
5. [code] ● **Labels on the actor's own states.** `ex_model-based-dagger` —
   episodes collected by the actor alone while the search relabels them:
   DAgger with the search as the expert.
   *Source:* new in the rebuild; Ross, Gordon and Bagnell 2011 (DAgger).
6. [extended] ●● **Imagined transitions for soft actor-critic.**
   `ex_model-based-dyna` — MBPO-style model rollouts mixed into the batches of
   `train_sac`, against the real-data control at the same update ratio.
   *Source:* new in the rebuild (decision Q1, experiment E10d); Janner et al.
   2019 (MBPO); Sutton 1990 (Dyna).
7. [code] **Lookahead with a value.** `ex_model-based-cartpole` — a learned
   CartPole model with one-step lookahead into the DQN of the Deep Q-Networks
   section, against acting greedily on $Q$.
   *Source:* new in the rebuild; original.

**Survey sources:** none surveyed yet for this new section. Candidates to check
in a later survey: Berkeley CS285's model-based assignment (learned dynamics,
random-shooting and CEM planning, ensembles), and Sutton & Barto ch. 8 (Dyna).

---
## chapter_deep-reinforcement-learning/offline-rl.md — Offline Reinforcement Learning

**Topic:** Which targets can use another policy's data, two logs of the
slippery lake, distribution shift against the exact optimum, the behavior
clone and stitching, pessimism (a two-armed bandit by hand, $\kappa/\sqrt{n}$
in the backup, narrow against uniform data, choosing $\kappa$), calibration,
and the deep methods by what they constrain.

**Exercises:** 9 (8 from the Oct-2026 set, several trimmed; 1 new; no label
new in the rebuild).

1. **Expected SARSA.** no label — Expected SARSA becomes Q-learning under the
   greedy policy, and why its action-value target needs no importance ratio.
   *Source:* Oct-2026 set, trimmed to two parts; Sutton & Barto Ex. 6.11–6.12
   (why Q-learning is off-policy).
2. [code] **Data from a better policy.** `ex_offline-rl-where-data-comes` —
   naive table and clone on logs of a better operator and on mixed logs.
   *Source:* Oct-2026 set; CS224R Spring 2026 HW3 (suboptimal data and
   stitching).
3. [code] **How much pessimism.** `ex_offline-rl-how-much-pessimism` — the
   $\kappa$ sweep over the uniform and narrow logs and two further kinds of log.
   *Source:* Oct-2026 set, extended by the rebuild's $\kappa$ sweep over both
   logs and the 500-episode narrow logs; original.
4. **Conditioning on luck.** no label — return-conditioned supervised learning
   on a one-step task with a stochastic reward.
   *Source:* Oct-2026 set; Brandfonbrener et al. 2022.
5. [code] **Dataset size and the error floor.** `ex_offline-rl-dataset-size` —
   whether the naive table's errors fall like $1/\sqrt{n}$.
   *Source:* Oct-2026 set, trimmed to one part; original.
6. ● **Pessimism in a two-armed bandit.** no label — the empirical best arm
   against a lower-confidence-bound rule as the logs grow.
   *Source:* Oct-2026 set (the text now works the smallest case by hand);
   original.
7. [code] ● **Evaluation without a simulator.**
   `ex_offline-rl-what-you-cannot-measure` — ordinary and weighted importance
   sampling of the learned policy from the log.
   *Source:* Oct-2026 set, trimmed; original.
8. ●● **Conservative Q-learning in a table.** no label — the CQL term in a
   table, its fixed point, and the part that plays the role of
   $\kappa/\sqrt{n}$.
   *Source:* Oct-2026 set, trimmed; CS285 Fall 2022 HW5 and CS224R Spring 2026
   HW3 (CQL, IQL, AWAC).
9. [extended] **Deep offline Q-learning.** no label — DQN on a frozen CartPole
   replay buffer, with and without the one-line CQL term.
   *Source:* new in the rebuild (decision Q5); Kumar et al. 2020.

**Survey sources:** CS224R Spring 2026 HW3 (AWAC and IQL on AntMaze;
stitching on PointMass); CS285 Fall 2022 HW5 (CQL, AWAC, IQL on gridworld
PointMass); Sutton & Barto Ex. 6.11–6.12. Spinning Up has no offline method.

---

## chapter_deep-reinforcement-learning/rl-sequences.md — Reinforcement Learning for Sequence Generation

**Topic:** A response as a trajectory, credit within a response (two-digit
answers), what the update needs, verifiers and reward models, the
KL-penalized group step against the closed form, ratios for stale samples,
asymmetric bands, GRPO with the $k_3$ term, and DPO.

**Exercises:** 10 (8 from the Oct-2026 set, the two-digit problem rewritten
after its first part moved into the text; 2 dropped, unequal group sizes and
the fixed point of `group_step`; 2 moved in from ppo; 1 label new in the
rebuild).

1. **Tool outputs in the trajectory.** no label — the probability of a
   trajectory with a tool call, and why the tool's tokens carry no score.
   *Source:* Oct-2026 set; original.
2. **Two-digit responses.** `ex_rl-sequences-two-digit-responses` — the
   response weight and the token advantages agree in expectation; their
   variances at the uniform and at a trained policy.
   *Source:* Oct-2026 set, rewritten (its first part is now the text's credit
   example); original.
3. **Two thresholds for the hedge.** `ex_rl-sequences-two-thresholds` — the
   hedge's probability under the tilted optimum as a function of $\beta$.
   *Source:* Oct-2026 set; original.
4. [code] ● **Noisy verifiers.** no label — a verifier with false negatives
   and false positives rescales and offsets the gradient.
   *Source:* Oct-2026 set; original.
5. ● **Process rewards in GRPO.** no label — the step-level standardized
   weights of process supervision.
   *Source:* Oct-2026 set; Shao et al. 2024.
6. [code] ● **DPO on the finite instance.** no label — DPO on preferences
   generated with the lenient checker, against the tilted optimum.
   *Source:* Oct-2026 set; Rafailov et al. 2023.
7. [extended] ●● **Credit from restarts.** no label — Monte Carlo prefix
   values against the group weight at a matched number of verifier calls.
   *Source:* Oct-2026 set; Kazemnejad et al. 2025 (VinePPO).
8. ●● **The KL term as a loss.** `ex_rl-sequences-kl-loss` — the expected
   gradients of $k_1$, $k_2$, $k_3$, and the forward-KL optimum on `8*8`.
   *Source:* Oct-2026 set; Tang and Munos 2025.
9. **Asymmetric bands.** `ex_ppo-asymmetric-bands` — batches needed to raise a
   rare token's probability with symmetric and asymmetric bands.
   *Source:* moved from ppo.md in the rebuild; Yu et al. 2025 (DAPO).
10. ● **Length-normalized ratio.** no label — the variance of the response
    ratio against its geometric mean.
    *Source:* moved from ppo.md in the rebuild; Zheng et al. 2025 (GSPO).

**Survey sources:** the RLHF book's policy-gradient chapter (GRPO and the
Dr. GRPO critique); CS224R Spring 2026 lectures on reward learning, DPO and RL
for reasoning (no graded problems on post-training); CS285 has no assignment
on post-training.
