# Further Reading
:label:`sec_deep_rl-further-reading`

<!-- Phase 1 (structure): the raw material below was moved verbatim from the sections named in each
comment; the writer of this file rewrites it to the standard (PLAN 15.9-M1, 15.9-M2). -->

## Policy Optimization in Practice

<!-- from ppo.md (Omitted Implementation Details) -->

### Omitted Implementation Details
:label:`subsec_ppo_omitted`

The gap between this section's PPO and a production one is a list of small, named decisions, none of which needs new theory:

| What real implementations add | Purpose |
|---|---|
| Learning-rate annealing to zero | reduce late updates as the policy approaches convergence |
| Observation and reward normalization | running estimates hold network inputs and value targets near unit scale on tasks whose raw numbers vary by orders of magnitude |
| Value-loss clipping | the critic gets a band of its own; its measured benefit is disputed, yet nearly every implementation ships it |
| KL-based early stopping | stop the epochs when the measured approximate KL passes a threshold: our diagnostic panel turned into an actuator |
| Orthogonal initialization, small policy head | start the policy near uniform so the first updates cannot blow the ratios out |
| Advantage normalization per minibatch | :numref:`sec_baselines`'s per-batch step size, recomputed at the granularity the gradient uses |
| Joint policy-and-value minibatch updates | one shuffled pass interleaves both losses; our critic-first, actor-second phases are a teaching choice |

The list is not ours. A community audit collected 37 such implementation details and measured which ones matter :cite:`Huang.Dossa.Raffin.ea.2022`; a controlled study went further and showed that at matched code-level choices PPO and TRPO perform nearly identically, so these details, not the clipped objective, account for much of PPO's practical edge :cite:`Engstrom.Ilyas.Santurkar.ea.2020`; and a large-scale sweep across a quarter-million trained agents reached similarly sober conclusions about which knobs carry the performance :cite:`Andrychowicz.Raichuk.Stanczyk.ea.2021`. The practical instruction a textbook can give is therefore this: when you need a real PPO, read a maintained single-file implementation and diff it against this section; the roughly 300-line `cleanrl/ppo.py` is the standard study text. Every line you do not recognize will be on the list above, and now you know why each is there.

<!-- from offline-rl.md (Bounded Staleness and Importance Ratios) -->

The distinction is not absolute in practical systems. Importance ratios can correct data from a nearby policy, but their variance grows rapidly as the two policies separate; PPO therefore reuses each batch for only a limited number of epochs. In distributed learning, actors may also lag behind the central learner. IMPALA's V-trace truncates importance corrections so that moderately stale data remains useful with controlled variance :cite:`Espeholt.Soyer.Munos.ea.2018`. The remainder of this section considers the extreme case: a fixed dataset with no further interaction.

## Value Learning after DQN

<!-- from dqn.md (The DQN Lineage) -->

### Extensions and Successors

Several later methods modify individual components of DQN. An $n$-step target uses several observed rewards before bootstrapping, as developed in :numref:`sec_actorcritic`. Prioritized replay samples transitions according to their temporal-difference error rather than uniformly :cite:`Schaul.Quan.Antonoglou.ea.2016`; its advantage decreases as replay buffers become larger and more diverse. A dueling network represents $Q$ as a state value plus action advantages :cite:`Wang.Schaul.Hessel.ea.2016`.

Distributional methods estimate the return distribution rather than only its mean :cite:`Bellemare.Dabney.Munos.2017`. In tabular and linear settings, their expected update for the mean coincides with the ordinary update, so their empirical benefit depends on the interaction with nonlinear function approximation. Rainbow combines six extensions and reports an ablation of their individual contributions :cite:`Hessel.Modayil.vanHasselt.ea.2018`.

PQN provides a contrasting design: with LayerNorm and batches collected from parallel environments, deep Q-learning can train without a replay buffer or target network :cite:`Gallici.Fellows.Ellis.ea.2025`. Replay and target networks are therefore one solution to correlated data and moving targets, not necessary components of every stable deep Q-learning system.

### DQN in Modern Practice

Vanilla DQN is mainly useful here as a simple setting in which to study two problems of deep value learning: moving bootstrap targets and maximization bias. It also establishes how a learner can use transitions collected by a different policy. Since the policy requires an argmax over actions, the basic method is restricted to discrete action spaces; actor-based methods extend related ideas to continuous control. :numref:`sec_offline` considers the limiting case in which all learning uses a fixed dataset and no further interaction is allowed.

## Continuous Control since 2018

<!-- from rl-sequences.md (Where to Go Next) -->

**Continuous control since 2024.** The objectives of :numref:`sec_deeprl` through :numref:`sec_regularized` stand; recent gains have come from normalization, scale and stability rather than new losses, and careful normalization even lets streaming, replay-free temporal-difference learning work after decades of not working :cite:`Elsayed.Vasan.Mahmood.2024,Gallici.Fellows.Ellis.ea.2025`. Reading a modern paper here, you should recognize nearly every symbol.

## Models, Imagination and Search

<!-- from rl-sequences.md (Where to Go Next) -->

**Model-based reinforcement learning and search.** The book's model-based methods are planning in a known model (:numref:`sec_valueiter`) and in a model counted from samples (:numref:`sec_qlearning`). A direct extension of the latter is Dyna-Q: augment :numref:`sec_qlearning`'s tabular loop by storing observed transitions and applying the same Q-update to model-generated transitions between environment steps. Monte Carlo tree search can be viewed as a *policy-improvement operator*: search produces an improved action distribution, which is then distilled into the policy as a supervised target. That loop is AlphaZero :cite:`Silver.Schrittwieser.Simonyan.ea.2017,Silver.Hubert.Schrittwieser.ea.2018`; MuZero learns the model it searches in :cite:`Schrittwieser.Antonoglou.Hubert.ea.2020`; DreamerV3 and TD-MPC2 train control inside learned world models :cite:`Hafner.Pasukonis.Ba.ea.2025,Hansen.Su.Wang.2024`. The same search-and-distillation pattern appears in test-time reasoning methods for language models.

## Offline Methods at Scale

<!-- from offline-rl.md (Constraining the Policy, the Values, or Both) -->

In continuous action spaces, almost every exact action is absent from the dataset, so support constraints become more important. Conservative Q-learning instead penalizes the values of out-of-distribution actions :cite:`Kumar.Zhou.Tucker.ea.2020`. Implicit Q-learning avoids querying unseen actions by backing up an upper expectile of values observed in the dataset :cite:`Kostrikov.Nair.Levine.2022`. TD3+BC adds a behavior-cloning term to an off-policy actor and provides a strong simple baseline :cite:`Fujimoto.Gu.2021`.

<!-- copied from offline-rl.md (Sequence Modeling without Bootstrapping), which keeps its own copy -->

Decision Transformer avoids bootstrapping by treating each trajectory as a sequence of returns-to-go, states, and actions :cite:`Chen.Lu.Rajeswaran.ea.2021`. An autoregressive model is trained by supervised prediction and, at test time, is conditioned on a desired return. It uses neither a value function nor a maximization target. On standard benchmarks, this approach matches strong value-based offline methods. However, a later study found that a small return-conditioned network trained by ordinary supervised learning recovers much of the performance, so the contribution of the sequence architecture remains unsettled. :numref:`sec_rl_sequences` returns to this connection for language models.

## Post-Training Variants

## Beyond the Chapter

<!-- from rl-sequences.md (Where to Go Next) -->

These chapters provide an introduction rather than comprehensive coverage. Three extensions are especially relevant.

**What we deliberately left out.** Multi-agent reinforcement learning, where the opponent learns too, reached superhuman poker with search plus self-play :cite:`Brown.Sandholm.2017`. Distributional reinforcement learning models the return's whole distribution rather than its mean :cite:`Bellemare.Dabney.Munos.2017`. Meta-learning, hierarchy and partial observability each relax one of :numref:`sec_mdp`'s assumptions; Murphy's overview treats all three at textbook depth :cite:`Murphy.2025`. For a broader treatment of post-training practice, see the reinforcement-learning-from-human-feedback book :cite:`Lambert.2026`.

## A Project

<!-- from rl-sequences.md (Capstone Projects) -->

These four projects play the role of a course programming assignment; each runs on a laptop CPU in minutes, and each carries a numeric sanity bar in place of an autograder.

**A. Build PPO from the pieces, and prove it is right.** Three self-checks: the ratio is exactly $1$ on the first epoch; GAE at $\lambda = 1$ equals reward-to-go minus $\hat{V}$ to floating-point tolerance; the clipped objective's gradient at $\theta = \theta_{\textrm{old}}$ equals the plain policy gradient. Bar: median return above $450$ on CartPole within $60$ updates, 3 seeds.

**B. Which implementation details actually matter.** Ablate five of the 37 catalogued PPO details, one at a time, three seeds each. Bar: at least one factor changes the median by more than the seed spread, and you can say which.

**C. Diagnose three faulty policies.** Examine one agent with a saturated policy, one with an untrained critic, and one with an undersized replay buffer. Identify each fault from the chapter's diagnostics before inspecting the code.

**D. GRPO from REINFORCE, in two lines.** Start from :numref:`sec_baselines`'s `train`, add the group-standardized weight and the clip, and reproduce the $K$-sweep of :numref:`subsec_baselines_group`. Bar: $K = 1$ shows no learning; $K \geq 4$ reaches the verifier's ceiling.
