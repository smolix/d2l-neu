# Reinforcement Learning
:label:`chap_reinforcement_learning`

In supervised learning, a prediction typically does not affect which test example is observed next. In reinforcement learning, an action changes the state of the environment and therefore affects subsequent observations and rewards. The agent consequently influences both its performance and the data from which it learns.

:numref:`fig_rl_agent_env` shows this interaction. Unrolling the agent--environment loop over time produces a *trajectory* $\tau=(s_0,a_0,r_0,s_1,\ldots)$. This feedback introduces the central difficulties studied in this chapter: exploration determines which data are collected, errors alter later state distributions, and policy updates change the distribution used for subsequent learning. The same formulation also applies to language-model generation, where the context is a state and the next token is an action; :numref:`sec_rl_sequences` develops that correspondence.

![The agent emits an action $a_t$ (blue); the environment answers with a reward $r_t$ and the next state $s_{t+1}$ (orange). Below, the same emissions unrolled in time interleave into the trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$, each symbol colored by the box that produced it: the loop and the trajectory are the same object.](../img/mdl-rl-agent-env.svg)
:label:`fig_rl_agent_env`

The sections progress from settings with the most information to settings with the least. :numref:`sec_mdp` defines the Markov decision process, and :numref:`sec_valueiter` solves a known finite MDP by dynamic programming. :numref:`sec_imitation` replaces the model with expert demonstrations. :numref:`sec_qlearning` then removes the expert and learns action values from sampled transitions. :numref:`sec_policygradient` optimizes the policy directly, while :numref:`sec_baselines` develops lower-variance estimators. Finally, :numref:`sec_deeprl` replaces tabular representations with neural networks for continuous state and action spaces.

:numref:`chap_deep_rl` continues with bootstrapped critics, safe reuse of on-policy data, replay buffers, regularized objectives, and offline learning. :numref:`fig_rl_roadmap` organizes the methods in both chapters by what they learn and which data they use.

![Both chapters on one map: what is learned, against which data may drive the update. Solid boxes are this chapter's; dashed boxes belong to the next, as do the greyed names inside shared boxes. DAgger sits in the on-policy column because its update data are the learner's own rollouts, expert-relabeled; behavior cloning alone trains from a fixed dataset. Value iteration is the map's one model-based resident: it consumes the kernel itself rather than sampled data, and occupies its cell for what it learns. The arrow out of the policy-gradient cell is that chapter's opening move: reusing slightly stale data through importance ratios, under a variance budget.](../img/mdl-rl-roadmap.svg)
:label:`fig_rl_roadmap`

:numref:`tab_rl_map` provides a more detailed guide to the algorithms in both chapters. It records what each method estimates, which data it can use, and where it is introduced.

:The algorithm map of these two chapters. "Named in" marks methods this book describes but does not implement.
:label:`tab_rl_map`

| algorithm | what it estimates | which data may drive the update | where |
|:--|:--|:--|:--|
| value iteration | $V^*$ exactly, by sweeping the Bellman operator | none: it consumes the kernel $P$ and reward $r$ | :numref:`sec_valueiter` |
| behavior cloning | $\pi(a \mid s)$ by cross-entropy on demonstrations | a fixed expert dataset | :numref:`sec_imitation` |
| DAgger | the same fit | the learner's own states, relabeled by an expert on call | :numref:`sec_imitation` |
| Q-learning (tabular) | $Q^*$ from sampled backups | any behavior's transitions (off-policy) | :numref:`sec_qlearning` |
| SARSA | the behavior's own $Q$, exploration included | its own transitions (on-policy) | named in :numref:`sec_qlearning` |
| UCB, Thompson sampling | arm means plus their uncertainty | its own pulls (the bandit) | :numref:`sec_qlearning` |
| REINFORCE | $\nabla_\theta J$ by the score function | fresh trajectories from the current policy only | :numref:`sec_policygradient` |
| REINFORCE with baselines, RLOO | the same gradient at lower variance | fresh trajectories from the current policy | :numref:`sec_baselines` |
| GRPO | group-normalized advantages, no value network | fresh groups of responses per prompt | weights in :numref:`sec_baselines`; machinery in :numref:`chap_deep_rl` |
| REINFORCE with a learned critic, on networks | $\pi_\theta$ and $\hat{V}$ as networks | fresh trajectories from the current policy | :numref:`sec_deeprl` |
| actor-critic (A2C) | $\pi_\theta$ plus a bootstrapped critic | fresh, near-current trajectories | :numref:`sec_actorcritic` |
| PPO | $\pi_\theta$ under a clipped probability ratio | one batch, reused for a few steps | :numref:`sec_ppo` |
| KL-regularized policy optimization, RLHF | a policy tilted from a reference by reward | rollouts scored by a learned reward | :numref:`chap_deep_rl` |
| DQN, Double DQN | $Q$ as a network | a replay buffer of stale experience | :numref:`sec_dqn` |
| Rainbow | DQN plus its measured components | a replay buffer | named in :numref:`sec_dqn` |
| DDPG, TD3 | a critic, and a deterministic actor trained to maximize it | a replay buffer (off-policy) | named in :numref:`sec_deeprl` and :numref:`sec_sac` |
| SAC | twin soft critics, and a stochastic squashed actor maximizing reward plus entropy | a replay buffer (off-policy) | :numref:`sec_sac` |
| offline Q-learning with pessimism, CQL | $Q$ penalized where the data is thin | a fixed logged dataset, no interaction at all | :numref:`sec_offline` |
| Decision Transformer | a return-conditioned sequence model | a fixed logged dataset | named in :numref:`sec_offline` |
| MuZero, Dreamer | a learned model of the environment, to plan or imagine in | its own interaction, replayed through the model | named in :numref:`chap_deep_rl` |
| DPO | the regularized optimum directly from preferences | a fixed preference dataset | :numref:`chap_deep_rl` and the Language Models part |

The experiments use small environments so that results can be evaluated against exact solutions and repeated across random seeds on a CPU. Numerical conclusions are reported with their relevant uncertainty and limitations in the sections where they arise. This chapter does not cover multi-agent, meta-, hierarchical, or goal-conditioned reinforcement learning, and it introduces partial observability only briefly. Model-based learning, large-scale RLHF, and preference optimization are treated elsewhere in the book.

```toc
:maxdepth: 2

mdp
value-iter
imitation
qlearning
policy-gradient
baselines
deep-rl
```

## Resources and Further Reading {.unnumbered}

Grouped by role: the books under the theory, the courses that teach it, the implementations to build from, and the record of what makes these methods work in practice. All are freely accessible online.

**Textbooks**

- Sutton and Barto, *Reinforcement Learning: An Introduction* :cite:`Sutton.Barto.2018` remains the field's front door, and the field now carries the profession's highest honor: Sutton and Barto received the 2024 Turing Award for building it.
- Szepesvári, *Algorithms for Reinforcement Learning* :cite:`Szepesvari.2010` states this chapter's algorithms and their guarantees in under a hundred pages.
- Bertsekas, *A Course in Reinforcement Learning* :cite:`Bertsekas.2025` teaches the same material from the optimal-control side, where dynamic programming is the trunk and learning the branch.
- Agarwal, Jiang, Kakade, and Sun, *Reinforcement Learning: Theory and Algorithms* :cite:`Agarwal.Jiang.Kakade.ea.2019` supplies the sample-complexity rates this chapter states qualitatively.
- Lattimore and Szepesvári, *Bandit Algorithms* :cite:`Lattimore.Szepesvari.2020` is the full theory behind the exploration interlude of :numref:`sec_qlearning`.

**Courses**

- [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) is the standard graduate treatment of the material these two chapters open.
- [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) is the course closest in scope to this chapter: foundations, exploration, and policy gradients with their guarantees.
- [Stanford CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) covers the practice, imitation and offline methods included.
- [CMU 10-703: Deep Reinforcement Learning and Control](https://cmudeeprl.github.io/703website/) pairs the algorithms with control and physical embodiment.
- [MIT 6.7920: Reinforcement Learning: Foundations and Methods](https://web.mit.edu/6.7920/www/) develops the dynamic-programming spine of :numref:`sec_valueiter` at full depth.

**Annotated implementations**

- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` publishes single-file, benchmarked implementations of the major deep reinforcement learning algorithms; the natural next step after :numref:`sec_deeprl`.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) :cite:`Raffin.Hill.Gleave.ea.2021` is the reliable library form of the same algorithms, with tuned hyperparameters in its accompanying zoo.
- [Gymnasium](https://gymnasium.farama.org/) :cite:`Towers.Kwiatkowski.Terry.ea.2024` documents the environment interface every agent in these two chapters speaks.

**The record**

- [Spinning Up in Deep RL](https://spinningup.openai.com/) :cite:`Achiam.2018` is a short curriculum from the score function to SAC, written to be read beside running code.
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` documents, one detail at a time, how much of reported performance lives in choices papers do not mention; the strongest published argument for the estimator hygiene of :numref:`sec_baselines`.
