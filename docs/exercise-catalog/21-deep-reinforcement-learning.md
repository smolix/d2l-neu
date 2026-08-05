# Chapter Overview: chapter_deep-reinforcement-learning

This chapter's 44 existing exercises (7 sections: actor-critic, ppo, regularized,
dqn, sac, offline-rl, rl-sequences) are corroborated more often than they are
found wanting. The single best sources are the *current* (2026) graduate deep-RL
assignments — Berkeley CS285's Assignment 2 (GAE derivation + λ-sweep),
Assignment 3 (DQN/Double DQN + SAC with temperature autotuning), and
Assignment 5/CS224R's Homework 3 (CQL/AWAC/IQL offline RL) — several of which
independently converge on the same environment, sweep, or diagnostic the book
already uses (CS285's CartPole critic-timescale sweep, its LunarLander DQN
port, its Hopper twin-critic ablation). Stanford CS234's current Assignment 2
supplies this search's single strongest individual match: a from-scratch proof
of the performance-difference lemma, which the book states and proves in-text
but had never turned into a reader exercise. Sutton & Barto remains the
canonical source for the chapter's classical core (Baird's counterexample,
Exercise 11.3; the λ-return recursion, Exercises 12.1–12.2; Q-learning-vs-SARSA,
Exercises 6.11–6.12) but predates KL-regularized and maximum-entropy policy
optimization entirely, leaving regularized.md and sac.md without a classical-text
counterpart. Two clear gaps: OpenAI Spinning Up's algorithm suite has no DQN at
all (confirmed against its algorithms page), and RLHF/GRPO/RL-for-sequences has
not yet stabilized into a graded-homework tradition anywhere found — the one
substantive supplement there (Dr. GRPO's variance-normalization critique) comes
from a book (rlhfbook.com), not a course. Disposition across the chapter: keep
43, rewrite 1 (rl-sequences' "read a paper" item, retitled around its
deliverable), drop 0 — consistent with the brief's expectation that this is
already the book's strongest exercise chapter. Five new problems are proposed
as pure additions (one each in actor-critic, ppo, regularized, offline-rl,
rl-sequences; none in dqn or sac, whose sets already matched canonical
assignments closely enough that no gap justified an addition), bringing the
proposed total to 49 problems.

---

## chapter_deep-reinforcement-learning/actor-critic.md — Actor-Critic Methods and Multi-Step Returns

**Topic:** bootstrapping the policy-gradient weight with a learned critic (the TD error as an advantage estimate), the batched actor-critic update, the bias-variance trade-off, and generalized advantage estimation (GAE) via a backward scan on TD errors.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — every item names a concrete deliverable and a check (a bias-vanishing condition, a predicted-vs-observed comparison, a failure mode to diagnose, a norm comparison, a working online variant, a four-way placement); this is already the group's house style at its best.

**External sources found:**
- Berkeley CS285 (Sergey Levine), Assignment 2: Policy Gradients, Spring 2026 (current) — derives GAE from the n-step advantage family through its recursive telescoping identity, then has students implement GAE-λ and sweep λ ∈ {0, 0.95, 0.98, 0.99, 1} on a noisy LunarLander-v2, explaining what the λ = 0 and λ = 1 endpoints mean and how λ moved performance — near-exactly the book's own λ-sweep and bias/variance framing — https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf
- Berkeley CS285, Assignment 3: Q-Learning and Actor-Critic Algorithms, Fall 2020 (archived; the last on-policy-AC version before the course moved this slot to off-policy SAC) — "sanity check with CartPole" sweeps (num_target_updates, num_grad_steps_per_target_update) over {(1,1), (100,1), (1,100), (10,10)} and asks which combinations fail to train — independently converges on the book's own critic_steps ∈ {1,5,20} two-timescale exercise, same environment, same diagnostic — https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/homeworks/hw3.pdf
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020 draft), Exercise 12.1 — derive the recursive one-step form of the λ-return from its defining infinite sum, the direct ancestor of the telescoping proof this section gives for GAE.
- Sutton & Barto, Exercise 12.2 — relate the trace-decay parameter λ to a half-life τ, the number of steps before the exponential weighting (γλ)^l has fallen to half its initial value — a natural companion to the book's own λ-sweep that nothing in the current set asks for.
- OpenAI Spinning Up, Exercises page, Exercise 2.1 "Value Function Fitting in TRPO" — compares extensive (80-iteration) against absent (0-iteration) critic training on Hopper across seeds, structurally close to this section's critic-fitting-sensitivity theme, though framed as a TRPO ablation rather than an actor-critic two-timescale failure — https://spinningup.openai.com/en/latest/spinningup/exercises.html
- Deep RL Bootcamp (Berkeley, August 2017), Core Lecture 5, "Natural Policy Gradients, TRPO, and PPO" (John Schulman) — the historically canonical GAE/TRPO derivation lecture; cited for completeness, no posed problem set attached.

No course or text was found posing the GAE telescoping identity itself as a stand-alone proof exercise — every source above either states GAE as background or exercises it empirically through a λ-sweep; that the book proves it as a Proposition in-text and then exercises only the empirical side is consistent with the wider tradition.

**Proposed problem set** (7 problems):
1. [conceptual] **Why the baseline argument does not extend.** (unchanged) Explain why the zero-mean baseline identity does not cover the bootstrapped weight $\delta_t$, and state the condition on $\hat V$ under which the bias vanishes.
   *Provenance:* original.
1. [short-code] **Where the U comes from.** (unchanged) Combine the estimator table's bias and variance per $\lambda$ into a predicted one-draw error, predict the sweep's favored $\lambda$ before looking, and diagnose any disagreement.
   *Provenance:* original.
1. [short-code] **The two timescales.** (unchanged) Sweep `critic_steps` over $\{1,5,20\}$ and diagnose the two-timescale failure mode.
   *Provenance:* original — independently corroborated by Berkeley CS285's own (num_target_updates, num_grad_steps_per_target_update) sweep on CartPole (Fall 2020 HW3, Q4), which finds the same class of failure on the same environment; this is evidence the existing exercise already sits at the canonical difficulty and shape, not a reason to change it.
1. [short-code] **The clip, removed.** (unchanged) Compare pre-clip gradient norms between REINFORCE and actor-critic and predict which run changes when the clip is removed.
   *Provenance:* original.
1. [extended] **Fully online.** (unchanged) Implement the fully online, no-batch actor-critic update and compare it with the batched version.
   *Provenance:* original.
1. [conceptual] **Where the actor-critic sits.** (unchanged) Place REINFORCE, REINFORCE+baseline, actor-critic, and Q-learning on bias and wait-time axes.
   *Provenance:* original.
1. [conceptual] **Lambda's half-life.** Derive the number of steps $\tau$ after which the GAE weight $(\gamma\lambda)^l$ has fallen to half its $l=0$ value, as a function of $\lambda$ (and $\gamma$) alone. Compute $\tau$ for $\lambda \in \{0.9, 0.95, 0.97, 0.99\}$ at $\gamma = 0.99$, and compare each value against this section's own finding that the smallest one-draw error sits near $\lambda = 0.9$–$0.95$. Does an "effective horizon" reading explain why moving $\lambda$ much closer to $1$ costs accuracy on a 500-step CartPole episode, while it would cost less on a task with a shorter horizon?
   *Provenance:* inspired by Sutton & Barto Exercise 12.2, the trace-decay half-life question (overlap low — S&B asks for the general $\lambda$-to-$\tau$ relation in isolation; the book would adapt it to interpret this section's own $\lambda$-sweep results, so cite on adoption).

---

## chapter_deep-reinforcement-learning/ppo.md — Trust Regions and Proximal Policy Optimization

**Topic:** the mismatch between parameter-space and policy-space distance, the importance-sampling surrogate objective, the performance-difference lemma, TRPO's monotonic-improvement bound, and PPO's clipped objective with its training diagnostics.
**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the densest, most quantitatively precise file in the group; every item states exactly what to compute and compare. External material adds one derivation exercise that closes a genuine gap: none of the seven currently asks the reader to prove the performance-difference lemma the section states and proves as a Proposition in the main text.

**External sources found:**
- Stanford CS234 (Emma Brunskill), Assignment 2, Winter 2026 (current, due Feb 1 2026), Section 2.4 — poses the exact clipped-ratio objective $z_\theta(s,a) = \pi_\theta/\pi_{\theta_{\textrm{old}}}$ and asks for which cases the clipped-loss gradient vanishes, and why PPO needs cached log-probabilities while REINFORCE does not — closely matches this section's Ex1 and its general ratio-mechanics framing — https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf
- Stanford CS234, same assignment, Section 3(d) — derives the discounted stationary state distribution $d^\pi$ and then asks students to prove the performance-difference lemma $V^\pi(s_0) - V^{\pi'}(s_0) = \frac{1}{1-\gamma} E_{s\sim d^{\pi'}}[E_{a\sim\pi}[A^{\pi'}(s,a)]]$, with hints tied explicitly to Kakade and Langford's monotonic-improvement argument — a direct, high-quality match for the Proposition this section states and proves but never turns into a reader exercise.
- OpenAI Spinning Up, TRPO background page — states the surrogate objective and KL constraint and includes a "You Should Know" box prompting the reader to verify both vanish exactly at the unchanged policy — a small embedded check in the spirit of this section's Ex1 — https://spinningup.openai.com/en/latest/algorithms/trpo.html
- OpenAI Spinning Up, PPO background page — exposition of the clipped objective and PPO-Clip pseudocode; no posed derivation or exercise — https://spinningup.openai.com/en/latest/algorithms/ppo.html
- CleanRL documentation, PPO page — beyond the "37 Implementation Details" audit and the Engstrom et al. controlled study this section already cites, its per-variant numbered "Implementation Details" checklists and cross-implementation benchmark tables against `openai/baselines` are a distinct resource; the book's own :numref:`sec_rl_sequences` Capstone Project B already assigns "ablate five of the 37 catalogued details," so this is noted rather than duplicated here — https://docs.cleanrl.dev/rl-algorithms/ppo/
- Deep RL Bootcamp, Core Lecture 5 (John Schulman) — the canonical TRPO/PPO derivation lecture; cited for completeness, no posed problem set.

No institution was found posing the asymmetric-clipping-band arithmetic (this section's own Ex7) as a stand-alone problem; that appears to be a genuinely thin area and an original contribution of the book.

**Proposed problem set** (8 problems):
1. [conceptual] **One epoch is not PPO.** (unchanged) Show the ratio and clipped-objective gradient reduce to the plain policy gradient at $\theta_{\textrm{old}}$.
   *Provenance:* original.
1. [extended] **Reuse against clipping.** (unchanged) Sweep `num_epochs` with and without clipping and report failure rates.
   *Provenance:* original.
1. [short-code] **How wide should the band be.** (unchanged) Sweep the clip parameter $\epsilon$ including a very large value.
   *Provenance:* original.
1. [short-code] **Minibatch epochs.** (unchanged) Extend to minibatch updates and compare against full-batch epochs.
   *Provenance:* original.
1. [short-code] **Saturation, and the cure.** (unchanged) Break REINFORCE with an oversized learning rate and cure it with an entropy bonus.
   *Provenance:* original.
1. [conceptual] **The clip as a step size.** (unchanged) Translate the ratio band into a bound on logit-difference change for a two-action softmax.
   *Provenance:* original.
1. [conceptual] **Asymmetric bands.** (unchanged) Compute clip-band probability bounds for two actions at different starting probabilities.
   *Provenance:* original.
1. [conceptual] **Proving the performance-difference lemma.** Before rereading the section's proof, derive $J(\theta) - J(\theta_{\textrm{old}}) = E_{\tau\sim\theta}\big[\sum_t \gamma^t A^{\textrm{old}}(s_t,a_t)\big]$ from the telescoping sum of one-step TD residuals under $V^{\textrm{old}}$ — the same telescoping technique :numref:`sec_actorcritic` uses to prove GAE's discounted-TD-error identity. Then, in the sampled surrogate $\hat L$, identify exactly which quantity replaces the new policy's trajectory distribution with the old policy's, and state in one sentence why that approximation — not the ratio clip — is where PPO's guarantee departs from TRPO's.
   *Provenance:* adapted from Stanford CS234 Assignment 2 (Winter 2026), Section 3(d) (overlap high; cite on adoption).

---

## chapter_deep-reinforcement-learning/regularized.md — Regularized Policy Optimization

**Topic:** learning a reward from preferences (Bradley-Terry), reward hacking under optimization of a fitted reward, and the closed-form KL-regularized optimum with its soft Bellman backup.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 (plus 1 new addition) — the six cover the closed-form limits, the frontier's concavity, a reward-error-detection budget, the trust-region/penalty taxonomy, Bradley-Terry identifiability, and the soft backup transplanted to a second gridworld; one gap remains (forward vs. reverse KL is asserted in the text but never exercised).

**External sources found:**
- Stanford CS224R (current, Spring 2026), Homework 2: Online Reinforcement Learning — its PPO implementation uses "a frozen reference actor $\pi_{\theta_{\textrm{ref}}}$, a snapshot of the policy taken after behavior-cloning pre-training... used during PPO updates to regularize the policy via a reverse-KL penalty, preventing it from drifting too far from the pre-trained initialization" — a directly-taught, currently-assigned instance of exactly this section's fixed-reference-penalty construction, and the same course explicitly separates it from PPO's own trust-region clip, matching this section's Ex4 taxonomy — https://cs224r.stanford.edu/material/hw2/CS224R_2026_Homework_2.pdf
- Nathan Lambert, *RLHF Book* (rlhfbook.com, already cited by this chapter), "Reward Modeling" chapter — walks through Bradley-Terry reward-model fitting and lists "Suggested Experiments": train a BT model on UltraFeedback and observe the reward margin grow; compare outcome- versus process-supervision on GSM8K/PRM800K; add a small held-out evaluation split — the closest thing to a posed exercise tradition for this section's reward-fitting cell, though at LLM-preference-dataset scale rather than this section's exactly-verifiable gridworld scale — https://rlhfbook.com/c/07-reward-models.html
- Nathan Lambert, *RLHF Book*, "Policy Gradients" chapter — its GRPO discussion cites the "Dr. GRPO" critique (Liu et al., 2025) that dividing by the group standard deviation inflates advantages on prompts where every sampled response scores alike — a live, actively-debated instance of exactly the overoptimization-of-a-proxy pattern this section demonstrates on the hazard gridworld, though it belongs more precisely to :numref:`sec_rl_sequences`'s own group-baseline material (see that entry).
- OpenAI Spinning Up, TRPO and PPO background pages — both state a trust-region KL role without a fixed-reference penalty, useful as the contrasting case behind this section's own Ex4.
- Sutton & Barto, *Reinforcement Learning: An Introduction* — checked chapters 11–13 directly; the 2020 draft predates KL-regularized/maximum-entropy policy optimization as a named topic, and no exercise anywhere poses a soft Bellman backup or a reward-tilting proposition. This is a genuine, notable gap in the classical textbook tradition, one this section's own derivation fills rather than adapts.

No course or text was found posing this section's own two sharpest questions — "how much KL budget does it take an optimizer to rediscover a planted reward-model error" (Ex3) and a from-scratch proof of the closed-form tilted optimum (Ex1) — as stand-alone homework; the nearest analogues operate at LLM scale (the RLHF book) rather than this section's tabular, exactly-checkable scale.

**Proposed problem set** (7 problems):
1. [conceptual] **Three limits.** (unchanged) Derive $\beta\to0$, $\beta\to\infty$, and uniform-reference from the proposition.
   *Provenance:* original.
1. [short-code] **The frontier is concave.** (unchanged) Sweep $\beta$ and verify concavity of reward against KL.
   *Provenance:* original.
1. [short-code] **Break the reward.** (unchanged) Perturb the fitted reward at one rarely-visited state and measure the KL budget needed to expose it.
   *Provenance:* original.
1. [conceptual] **Trust region or penalty.** (unchanged) Classify TRPO, PPO's clip, RLHF's KL term, and on-policy distillation.
   *Provenance:* original — corroborated by Stanford CS224R's current HW2, which implements both a trust-region clip and a fixed-reference penalty in the same PPO loop and requires students to keep the two roles distinct, exactly the discipline this exercise tests.
1. [conceptual] **Identifiability.** (unchanged) Show a per-prompt reward shift leaves the Bradley-Terry likelihood unchanged.
   *Provenance:* original.
1. [extended] **The soft backup, transplanted.** (unchanged) Port `soft_v` to the slippery lake with a uniform reference and sweep $\beta$.
   *Provenance:* original.
1. [short-code] **Which KL, and what it misses.** For the five-action bandit already in the text ($\pi_{\textrm{ref}} = (0.30, 0.10, 0.25, 0.20, 0.15)$, $r = (0, 0.5, 1, 2, 3)$), compute the *forward*-KL-regularized optimum $\arg\max_\pi E_\pi[r] - \beta D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi)$ at the same $\beta$ values already used for the reverse-KL optimum $\pi^\star$, by gradient ascent analogous to `solve_kl`. Report where the two optima disagree most (which action), and connect the disagreement to the mode-seeking-versus-mode-covering language the section states but does not compute. Which of the two would be the wrong choice if $\pi_{\textrm{ref}}$'s low-probability action ($a_2$, at $0.10$) were in fact catastrophic to under-visit?
   *Provenance:* original.

---

## chapter_deep-reinforcement-learning/sac.md — Soft Actor-Critic

**Topic:** SAC's maximum-entropy actor-critic — the soft policy-evaluation target, soft policy improvement as a KL projection, the squashed-Gaussian policy's change-of-variables correction, twin critics with a pessimistic minimum, and Polyak-averaged targets.
**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the group's highest citation and crossref density; every item states an exact computation and comparison target, and independent cross-institution checking (below) shows the set already matches the current canonical assignment closely enough that no addition is warranted.

**External sources found:**
- Berkeley CS285 (Sergey Levine), Assignment 3, Spring 2026 (current), §3.5 "Automatic Temperature Tuning" — implements the SAC paper's dual-gradient-descent temperature autotuning verbatim (target entropy $-\dim\mathcal{A}$, a learnable $\log\alpha$ with its own optimizer), tested on HalfCheetah, and asks "how does the temperature evolve during training? does it increase, decrease, or stabilize, and why might it change this way for HalfCheetah?" — a near one-to-one match for this section's own Ex1, on a different environment — https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf
- Berkeley CS285, same Assignment 3, §3.6 "Stabilizing Target Values" — runs single-critic vs. clipped-double-critic SAC on Hopper and asks students to relate the difference to overestimation bias, logging both `eval_return` and `q_values` — corroborates rather than extends this section's own single-critic-vs-twin-critic calibration comparison, which the main text already runs on Pendulum.
- OpenAI Spinning Up, SAC background page — derives the entropy-regularized Bellman backup and documents that SAC's squashed-Gaussian specifically requires *state-dependent* log-std ("SAC with state-independent log std devs... did not work"), and gives the full twin-Q/target-network update — pure exposition, but an independently authored cross-check of this section's own derivation and its emphasis on the state-dependent scale — https://spinningup.openai.com/en/latest/algorithms/sac.html
- OpenAI Spinning Up, TD3 background page — derives clipped double-Q, target-policy smoothing, and delayed policy updates with concrete defaults (one policy update per two critic updates) — the DDPG$\to$TD3 lineage this section's Ex5 asks students to trace — https://spinningup.openai.com/en/latest/algorithms/td3.html
- OpenAI Spinning Up, Exercises page, Exercise 1.3 "Computation Graph for TD3" — implement TD3's loss functions and intermediate calculations from a near-complete skeleton, tested on HalfCheetah and InvertedPendulum — the nearest posed-homework relative of this section's own algorithm family, though it targets TD3 rather than SAC directly.
- CleanRL documentation, `sac_continuous_action` page — logged-metric glossary and a comparison table against the original paper's numbers, independently confirming the "twin critics reduce but do not eliminate overestimation" pattern this section measures via its calibration experiment.

**Proposed problem set** (7 problems, unchanged):
1. [short-code] **Autotune the temperature.** Implement Lagrange-multiplier temperature tuning targeting $\bar H = -\dim\mathcal{A}$ and plot $\alpha_t$ against the entropy trace.
   *Provenance:* original — independently corroborated by Berkeley CS285's current Assignment 3, §3.5, which assigns the identical construction (same target-entropy convention, same dual-gradient-descent update) on HalfCheetah rather than Pendulum; cite CS285/the underlying Haarnoja et al. 2018 paper (already cited by the book) if drawing the HalfCheetah variant into the book directly.
1. [short-code] **Delete the log-determinant.** Remove the tanh Jacobian correction and see which diagnostics expose the resulting error.
   *Provenance:* original.
1. [conceptual] **Off-policy without ratios.** State the distribution each SAC expectation is taken under and what a replay buffer still cannot correct.
   *Provenance:* original.
1. [short-code] **Entropy is not monotone in the noise.** Show the squashed policy's entropy peaks at an intermediate $\sigma$.
   *Provenance:* original.
1. [short-code] **The limit $\alpha\to0$ is DDPG.** Recover DDPG's actor update and connect to TD3's three repairs.
   *Provenance:* original — the DDPG/TD3 lineage is independently derivable from OpenAI Spinning Up's TD3 and DDPG background pages, useful as a cross-check reference when writing a solution.
1. [short-code] **Port it.** Move the section's code to `LunarLander-v3` continuous.
   *Provenance:* original.
1. [extended] **Discrete soft actor-critic.** Derive and implement discrete-action SAC and compare against DQN.
   *Provenance:* original.

---

## chapter_deep-reinforcement-learning/dqn.md — Deep Q-Networks

**Topic:** instability from combining function approximation, bootstrapping, and off-policy data (the deadly triad and Baird's counterexample), the experience-replay-and-target-network repair, maximization bias, and Double DQN.
**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — every sweep names its grid and its diagnostic; independent cross-checking below shows the current set already matches, almost line for line, the canonical modern DQN assignment.

**External sources found:**
- Berkeley CS285 (Sergey Levine), Assignment 3, Spring 2026 (current), §2 "Deep Q-Learning" — implements DQN on CartPole, LunarLander (three seeds), and MsPacman (Atari, ~3 hours on a GPU), and specifically asks: run DQN with the learning rate changed to $0.05$ and explain what happens to (a) the predicted Q-values and (b) the critic error, connecting the answer to class discussion of instability — this maps closely onto this section's own overestimation diagnostics (Ex3, Ex4) and its "port the code and list every changed line" Ex6, down to sharing `LunarLander` as the port target — https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf
- Berkeley CS285, same Assignment 3, §2.5 "Double Q-Learning" — implements :eqref:`eq_double_dqn` exactly, selection with the online network and evaluation with the target network, and requires a side-by-side vanilla-DQN vs. Double-DQN comparison on LunarLander — essentially the same derivation and ablation this section runs on CartPole.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020 draft), Exercise 11.3 (programming) — "Apply one-step semi-gradient Q-learning to Baird's counterexample and show empirically that its weights diverge" — the canonical textbook source of the exact demonstration this section runs; this section's own Ex7 (add a target network and sweep the sync period) is a natural extension the textbook exercise does not itself pose, so the book goes one step further than its own source here.
- Stanford CS234 (Emma Brunskill), Assignment 2 — implements the "Nature DQN" architecture with a target network and experience replay on CartPole and Atari Pong, including an $n$-step-estimator component — a canonical DQN implementation assignment, structurally similar to this section though without its specific target-network-ablation framing. **Caveat:** this description is reconstructed from student repository descriptions and search summaries rather than a directly fetched official assignment PDF (unlike the other sources in this entry); treat it as lower-confidence and reverify the current PDF before citing it in the book.
- CleanRL documentation, DQN page — logs "TD loss," episodic return, and mean Q-value (flagged as an overestimation indicator), and documents concrete deviations from Mnih et al. 2015 (Adam vs. RMSProp, sync period 1000 vs. 10000 steps, no error clipping) — corroborates this section's own choice to log a fixed-probe-state value estimate as its overestimation diagnostic.

**Notable gap:** OpenAI Spinning Up's algorithm suite (VPG, TRPO, PPO, DDPG, TD3, SAC) does not include DQN at all — confirmed directly against its algorithms page. Despite being the single most useful teaching resource for four of this chapter's other six sections, it has nothing to offer value-based methods; this is a genuine, notable absence rather than an oversight in the search.

**Proposed problem set** (7 problems, unchanged):
1. [conceptual] **What survives the scramble.** State which of `td_target`, `reward_to_go`, and `gae` remain meaningful on a replay-scrambled batch.
   *Provenance:* original.
1. [conceptual] **Terminated, not done.** Describe the damage from storing `done` instead of `terminated`.
   *Provenance:* original.
1. [short-code] **The sync period.** Sweep `sync_every` over $\{50, 250, 2500\}$ and distinguish two failure directions.
   *Provenance:* original — independently corroborated by Berkeley CS285's current Assignment 3, §2.4, whose own learning-rate perturbation on CartPole produces the same class of diagnosis (predicted-Q-value vs. critic-error decoupling).
1. [short-code] **Shrink the buffer.** Set replay capacity to 500 and retrain one seed.
   *Provenance:* original.
1. [short-code] **Best against final, at scale.** Collect best/final-window statistics across every already-logged seed and arm, with no new training.
   *Provenance:* original.
1. [short-code] **Port it.** Move the section's code to `LunarLander-v3` and list every changed line.
   *Provenance:* original — corroborated by Berkeley CS285's current Assignment 3, which assigns exactly this port (DQN and Double DQN on `LunarLander-v2`/`v3`) as a graded deliverable, evidence the book's choice of port target is the field's own standard second environment for this algorithm.
1. [extended] **Baird, with a target network.** Add a target network to the Baird cell and sweep its sync period $k\in\{10,100\}$.
   *Provenance:* adapted from Sutton & Barto Exercise 11.3 (overlap medium — S&B's exercise stops at demonstrating divergence; the book's extension, adding the very repair this section derives, goes beyond what the textbook itself asks and needs no further citation on adoption since it only reuses the classical counterexample construction, already cited elsewhere in the book via :cite:`Baird.1995`).

---

## chapter_deep-reinforcement-learning/offline-rl.md — On-Policy, Off-Policy, and Offline Learning

**Topic:** the on-policy/off-policy distinction (Q-learning vs. SARSA targets), offline learning from a fixed dataset, measuring distribution shift and overestimation against a computable optimum, and a count-based pessimistic penalty.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 (plus 1 new addition) — the six already cover behavior-cloning comparison, dataset-size scaling, dataset-composition, pessimism-strength sweeping, the support-constraint-vs-count-penalty distinction, and what a real deployment would need instead of simulator rollouts; one genuinely different angle (trajectory *stitching*) is missing and is not a duplicate of any existing item.

**External sources found:**
- Stanford CS224R (current, Spring 2026), Homework 3: Offline RL — implements AWAC and IQL on AntMaze (from D4RL) and explicitly "studies the stitching behavior of offline RL using IQL on a PointMass task with suboptimal data" to "evaluate whether the learned policy can compose better trajectories than those seen in the data" — a modern, currently-taught match for exactly the distribution-shift/overestimation theme this section builds, and a genuinely different angle (composing sub-trajectories) from anything in the current six — https://cs224r.stanford.edu/material/hw3/CS224R_2026_Homework_3.pdf
- Berkeley CS285 (Sergey Levine), Assignment 5: Exploration and Offline Reinforcement Learning (Fall 2022 vintage; the offline-RL slot has since moved to a dedicated assignment in courses like CS224R) — implements CQL, AWAC, and IQL on gridworld `Pointmass` domains and specifically asks students to "examine the difference between the Q-values on state-action tuples learned by CQL vs. DQN" and interpret whether CQL underestimates relative to plain DQN — directly matches this section's own predicted-vs-realized-value calibration comparison between its "naive" and "pessimistic" arms — https://rail.eecs.berkeley.edu/deeprlcourse-fa22/static/homeworks/hw5.pdf
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020 draft), Exercise 6.11 ("Why is Q-learning considered an off-policy control method?") and Exercise 6.12 ("Suppose action selection is greedy. Is Q-learning then exactly the same algorithm as Sarsa?") — the canonical textbook pairing behind this section's own Q-learning-vs-SARSA comparison; this section's own experiment already goes past both by calibrating the two learned tables against the known-optimal FrozenLake values, so these serve to corroborate the conceptual core rather than mark a gap.
- OpenAI Spinning Up — checked directly; its algorithm suite has neither DQN nor a dedicated offline-RL method, so it is not a source for this section (consistent with the same finding recorded for :numref:`sec_dqn`).
- Levine, Kumar, Tucker, and Fujimoto's offline-RL papers (CQL, IQL, BCQ), already cited by this section, are methods papers rather than teaching materials with posed problems; noted rather than stretched into an exercise citation.

No institution was found posing a fixed-dataset, no-further-interaction offline-RL problem as *undergraduate* homework — the tradition is concentrated in the two graduate deep-RL courses above, both confirmed current or near-current, which is itself worth recording: offline RL is a young but now-institutionalized homework topic at the graduate level, still largely absent from courses like Stanford CS234.

**Proposed problem set** (7 problems):
1. [short-code] **Does the clone win?** (unchanged) Add a behavior-cloning arm across the three dataset types of exercise 3.
   *Provenance:* original.
1. [short-code] **More data, or better data.** (unchanged) Rerun with 100/500/2000-episode datasets and track predicted vs. actual value.
   *Provenance:* original.
1. [short-code] **Where the data comes from.** (unchanged) Compare random, near-optimal, and mixed behavior datasets.
   *Provenance:* original.
1. [short-code] **How much pessimism.** (unchanged) Sweep $\kappa$ over $\{0.02, 0.1, 0.3, 1.0\}$.
   *Provenance:* original.
1. [conceptual] **Constrain the values, or constrain the policy.** (unchanged) Construct a case where the hard support rule and the count penalty genuinely differ.
   *Provenance:* original.
1. [conceptual] **What you cannot measure offline.** (unchanged) Name the estimator a real deployment would use instead of simulator rollouts, and why it inherits PPO's variance problem.
   *Provenance:* original.
1. [short-code] **Stitching from suboptimal pieces.** Build a FrozenLake dataset from two behavior policies whose combined trajectories reach the goal only by composition: policy A reliably reaches one interior state via a safe detour but rarely continues past it, and policy B starts near that state and reliably finishes, so that no single collected episode does both halves. Run naive offline Q-learning and the behavior-cloning baseline on the pooled dataset and report whether the learned greedy policy's return exceeds the return of *either* source policy alone. State which property of Q-learning's per-transition bootstrap — as opposed to behavior cloning's per-trajectory imitation — makes this composition possible in principle, and name the exact line in `offline_q` that is doing the combining.
   *Provenance:* adapted from Stanford CS224R Homework 3 (Spring 2026), "stitching behavior of offline RL using IQL on a PointMass task with suboptimal data" (overlap medium — CS224R runs this on continuous PointMass/AntMaze D4RL data with IQL; this adapts the same experimental question, composing behavior from suboptimal sub-trajectories, to the section's own tabular FrozenLake and its existing `offline_q` function; cite on adoption).

---

## chapter_deep-reinforcement-learning/rl-sequences.md — Reinforcement Learning for Sequence Generation

**Topic:** casting autoregressive generation as an MDP, the response/token score-function equivalence, which general-RL machinery survives a terminal-reward deterministic-transition setting, the group-mean baseline's self-inclusion bias (GRPO/RLOO), and reward hacking under a KL budget.
**Current exercises:** 5; disposition: keep 4, rewrite 1, drop 0 (plus 1 new addition) — the shortest file in the group and the one that ties most explicitly into the upcoming Language Models part; four exercises are fully sound, the fifth ("read a paper") is the group's only item framed as an activity rather than a deliverable per the prior style review, and the group-baseline analysis stops at the *mean*, leaving its *variance* term unexercised. This section also carries four separate "Capstone Projects" (outside the `## Exercises` heading and not counted in the disposition above), which are out of scope for this catalog's per-exercise format but are worth noting as already excellent, larger-scope material.

**External sources found:**
- Nathan Lambert, *RLHF Book* (rlhfbook.com, already cited by this chapter), "Policy Gradients" chapter — its GRPO discussion cites the "Dr. GRPO" critique (Liu, Chen, Du et al., 2025) that dividing by the group's reward standard deviation inflates the advantage on prompts where every sampled response scores alike (uniformly easy or uniformly hard), and separately covers GSPO's sequence-level length-normalized ratio (already the source this chapter's own ppo.md cites for the same construction) and CISPO's clip-the-importance-weight-rather-than-the-objective design — directly extends this section's own self-inclusion-bias analysis, which currently examines only the group *mean*, to the group's *variance* term, a mechanism the book does not yet exercise — https://rlhfbook.com/c/11-policy-gradients.html
- Stanford CS224R (current, Spring 2026) — its syllabus lists dedicated lectures on "Reward Learning & RLHF," preference optimization, DPO, and "RL for language models reasoning" (Apr 24–May 1), but its graded homeworks (HW1–HW3, verified directly) cover imitation learning, on/off-policy actor-critic, and offline RL rather than an RLHF/DPO/GRPO problem set — the lecture content exists, but no *posed homework problem* on this section's specific material (group baselines, RLVR, DPO derivation) was found in the current assignments.
- Berkeley CS285 — checked; no assignment on RLHF, DPO, or the language-model-as-MDP framing was found in the current or recent homework set (HW1–HW5 cover imitation learning through offline RL/exploration, not post-training).

This section's own topic — RLHF/GRPO/RL-for-sequences posed as a graded, checkable *exercise* rather than a paper or blog post — has a genuinely thin external tradition as of this search: the fast-moving 2023–2025 research area has not yet stabilized into standard courses' problem sets the way DQN or PPO have. The one substantive find (the Dr. GRPO variance-normalization critique) comes from a book, not a course, and is itself very recent (2025).

**Proposed problem set** (6 problems):
1. [conceptual] **The factorization.** (unchanged) Prove $\nabla_\theta \log \pi_\theta(y\mid x) = \sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$.
   *Provenance:* original.
1. [conceptual] **Which terms disappear.** (unchanged) Say what property removes discounting, bootstrapping, the TD error, and replay, and when each returns.
   *Provenance:* original.
1. [short-code] **Why $K=1$ learns nothing.** (unchanged) Predict and verify the $(K-1)/K$ shrinkage's zero at $K=1$.
   *Provenance:* original.
1. [short-code] **Price the exploit.** (unchanged) Find the $\beta$ at which the KL penalty makes the verifier loophole unprofitable.
   *Provenance:* original.
1. [conceptual] **Map GRPO's published objective.** (lightly rewritten from "Read a paper") Take the GRPO objective as published and, for each symbol, name the section of these two chapters that built it and the one component absent from the book. State the mapping and the missing component as the deliverable up front, before any reading.
   *Provenance:* original — retitled and re-ordered per the prior style review's observation that this was the group's only exercise framed as an activity ("read a paper") rather than a deliverable; the underlying task is unchanged.
1. [short-code] **The denominator's hidden lever.** Modify `group_step` to compute two candidate advantages per group: the book's own standardized $A_j = (r_j - \mu)/(\sigma + 10^{-8})$, and a mean-only version $A_j' = r_j - \mu$ with no division. Rerun the $K$-sweep for $K \in \{2, 4, 8, 32\}$ under both, and for each $K$ report how many update steps hit a group where every sampled response scored identically (so $\sigma = 0$ or $\sigma$ is at floating-point noise) and what the standardized version's update looks like there versus the mean-only version's. Does either variant reach the verifier's ceiling faster at the matched sample budget used in the text? In one sentence, connect what you observe to the "Dr. GRPO" objection that dividing by $\sigma$ inflates advantages on uniformly-scored groups, and say whether this toy instance's near-binary verifier rewards make the effect easier or harder to see than it would be with continuous-valued rewards.
   *Provenance:* inspired by Nathan Lambert's *RLHF Book*, "Policy Gradients" chapter, discussion of Liu et al.'s "Dr. GRPO" critique of GRPO's standard-deviation normalization (overlap low — the source is a critique of the published algorithm, not a posed exercise; the book would adapt the *question* to its own finite worked example and existing `group_step` code; cite Liu et al. 2025 on adoption, as the book already does for the Dr. GRPO reference in :numref:`sec_baselines`).

