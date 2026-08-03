# Deep Reinforcement Learning
:label:`chap_deep_rl`

:numref:`chap_reinforcement_learning` introduced tabular value methods, policy gradients, and neural function approximation. The resulting REINFORCE agent waits for complete episodes, uses each on-policy batch once, and places no explicit constraint on the change in its policy. Deep reinforcement learning addresses these limitations while also introducing new sources of instability: bootstrapped targets depend on learned predictions, shared parameters couple updates across states, and reused data may have been collected by an older policy.

The chapter develops these issues through seven examples. Actor--critic methods replace complete returns with bootstrapped value estimates and lead to $n$-step returns and generalized advantage estimation. Trust regions and PPO control changes in the policy while permitting limited reuse of on-policy data. Regularized policy optimization studies learned rewards and KL penalties. DQN combines Q-learning with replay and target networks; SAC extends the same off-policy machinery to continuous actions. Offline reinforcement learning considers a fixed dataset with no further interaction. The final section formulates language-model generation as a decision process and identifies which preceding methods remain necessary in that special case.

:numref:`fig_rl_roadmap` locates these methods according to what they learn and which data they use. Because the implementations reuse objects introduced in :numref:`chap_reinforcement_learning`, :numref:`tab_rl_inherited` lists those objects and the additions made in this chapter.

:The objects this chapter's code uses without rebuilding, and where each was built. The rows below the marked line join the library in this chapter.
:label:`tab_rl_inherited`

| object | what it does | built in |
|:--|:--|:--|
| `TabularMDP` | the model as arrays: kernel, reward, discount, and one Bellman `backup` sweep | :numref:`sec_mdp` |
| `value_iteration`, `policy_evaluation` | the exact sweeps to $V^*$ and to $V^\pi$, this chapter's yardsticks | :numref:`sec_valueiter` |
| `evaluate` | the mean return of a policy over fresh evaluation episodes | :numref:`sec_valueiter` |
| `ActorCritic`, `.tabular` | the policy-plus-value container with its optimizers, and its table form | :numref:`sec_imitation` |
| `ActorCritic.mlp` | the same container with one-hidden-layer network heads | :numref:`sec_deeprl` |
| `policy_step` | one ascent step on advantage-weighted log-probabilities | :numref:`sec_imitation` |
| `fit_value` | regression passes of $\hat{V}$ toward a supplied target | :numref:`sec_deeprl` |
| `Batch`, `rollout` | the trajectory container and the collection loop that fills it | :numref:`sec_policygradient` |
| `Batch.backward_scan`, `Batch.reward_to_go` | the discounted backward scan, and the reward-to-go it computes | :numref:`sec_baselines` |
| `normalize` | per-batch standardization of the policy weight | :numref:`sec_baselines` |
| `run_seeds` | one training generator run across seeds into a single array | :numref:`sec_baselines` |
| `epsilon_greedy`, `linear_schedule` | the exploration rule and its annealing | :numref:`sec_qlearning` |
| `plot_curves`, `show_grid` | seed-band learning curves; gridworld values and policies as panels | :numref:`sec_utils` |
| **added in this chapter** | | |
| `Batch.td_target` | the one-step bootstrapped target $r + \gamma (1 - \textrm{terminated})\, \hat{V}(s')$ | :numref:`sec_actorcritic` |
| `Batch.gae` | generalized advantage estimation: the backward scan run on TD errors | :numref:`sec_actorcritic` |
| `ppo_epochs` | clipped-surrogate reuse epochs on a frozen batch, diagnostics returned as data | :numref:`sec_ppo` |
| `ReplayBuffer` | a ring of transitions whose `sample` scrambles time into a `Batch` | :numref:`sec_dqn` |
| `offline_q` | Q-learning swept over a fixed dataset, with optional pessimism $\kappa/\sqrt{n}$ | :numref:`sec_offline` |

Training loss alone is often insufficient to diagnose a reinforcement-learning run. A value estimate can diverge while its regression loss decreases, and a poor policy can continue to generate apparently regular curves. :numref:`tab_rl_diagnostics` collects the additional measurements used in this chapter and indicates how they should be interpreted.

:Diagnostics for the reinforcement-learning experiments in this chapter. The reported ranges are task-specific; the qualitative patterns are more general.
:label:`tab_rl_diagnostics`

| diagnostic | healthy reading | measured in |
|:--|:--|:--|
| the ratio $\rho_t$ on the first reuse epoch | exactly $1$, by construction; anything else is a bug in the frozen log-probabilities | :numref:`sec_ppo` |
| approximate KL within a batch | small and front-loaded: it increases for a few epochs, then flattens; continued growth indicates excessive policy change | :numref:`sec_ppo` |
| fraction of ratios outside the clip band | about one check in twenty; several times that means the reuse or the step size is too aggressive | :numref:`sec_ppo` |
| policy entropy | a slow decrease, about $0.65$ to $0.25$ nats over a CartPole run; a rapid decrease toward zero indicates policy saturation | :numref:`sec_ppo` |
| weight-only effective sample size of the reused batch | ratio concentration: weights nearly flat through the reuse epochs; concentration to half or less says stop reusing; blind to advantages, dependence, and state staleness | :numref:`sec_ppo` |
| correlation of the TD error with the Monte Carlo advantage | about $0.3$ early, falling toward zero as episodes lengthen, consistent with the Monte Carlo weight dissolving into tail noise; a descriptive diagnostic, not a critic certificate | :numref:`sec_actorcritic` |
| pre-clip gradient norm, and how often the clip binds | stable norms, the clip binding on a bounded fraction of updates: about a third for the bootstrapped weight and none for Monte Carlo, at matched normalization; descriptive of where variance lives, not proof of signal | :numref:`sec_actorcritic` |
| streak lengths at the ceiling | long runs of consecutive perfect batches, a run-specific stability visualization; arriving without resting points at the estimator's noise, not the policy | :numref:`sec_actorcritic` |
| value estimate at the start state | below the ceiling of the objective the update defines, $1/(1-\gamma) = 100$ on CartPole's continuing formulation, by tens of points at most; above the line is overestimation, growing without bound is divergence | :numref:`sec_dqn` |
| best against final trailing window, across seeds | descriptive statistics of the curve only: the final window depends on when training stops, while the retrospective best window is affected by optimistic selection; report a predeclared fixed-budget evaluation instead | :numref:`sec_dqn`, and the boxed reading rule of :numref:`sec_baselines` |
| the entropy trace of a stochastic continuous policy | descends from its wide start and ends within about a nat of the autotuning target $-\dim \mathcal{A}$; an entropy that rises without bound, or disagrees with a one-line quadrature check of the density, indicates an incorrect squashing-density calculation | :numref:`sec_sac` |
| predicted value against evaluated return | predictions should not exceed the maximum achievable return; a prediction above a computable optimum establishes overestimation without requiring a baseline | :numref:`sec_offline`, and the twin-against-single calibration of :numref:`sec_sac` |

Most experiments use CartPole and several random seeds. :numref:`sec_regularized` and :numref:`sec_offline` use tabular gridworlds because their optimal values can be computed exactly, and :numref:`sec_rl_sequences` uses deterministic string concatenation rather than a simulator. The chapter does not cover multi-agent, meta-, hierarchical, goal-conditioned, partially observable, or model-based reinforcement learning. Large-scale RLHF, DPO, and additional GRPO variants are treated in the Language Models part.

```toc
:maxdepth: 2

actor-critic
ppo
regularized
dqn
sac
offline-rl
rl-sequences
```

## Resources and Further Reading {.unnumbered}

The resources in :numref:`chap_reinforcement_learning` cover the general theory and courses. The following references focus on implementations and empirical evaluation.

- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` publishes single-file, benchmarked implementations of every deep algorithm in this chapter; :numref:`sec_ppo` ends by asking you to diff its `ppo.py` against that section.
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` documents, one detail at a time, how much reported performance lives in choices papers do not mention.
- The controlled study of :citet:`Engstrom.Ilyas.Santurkar.ea.2020` shows PPO and TRPO nearly indistinguishable at matched code-level choices: the details above, not the objective, carry much of the practical edge.
- [rliable](https://github.com/google-research/rliable) :cite:`Agarwal.Schwarzer.Castro.ea.2021` turns the seed-spread discipline of :numref:`tab_rl_diagnostics` into tooling: interval estimates and performance profiles for few-seed results.
- The [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) :cite:`Raffin.Hill.Gleave.ea.2021` ships tuned hyperparameter configurations per environment, the sensible starting point when a method of this chapter meets a task that is not CartPole.
- The offline survey of :citet:`Levine.Kumar.Tucker.ea.2020` continues :numref:`sec_offline` at full depth, from the tabular diagnosis to the deep-scale methods it could only name.
- Chapters 11 to 13 of Sutton and Barto :cite:`Sutton.Barto.2018` give the deadly triad of :numref:`sec_dqn` its full theory, including what is provable once one corner is dropped.
- [The RLHF book](https://rlhfbook.com/) :cite:`Lambert.2026` develops the
  post-training applications introduced in :numref:`sec_rl_sequences`.
