# Deep Reinforcement Learning
:label:`chap_deep_rl`

:numref:`chap_reinforcement_learning` ended with a working agent. The policy-gradient loop of :numref:`sec_deeprl` solved CartPole with a network in place of the table, and its closing page named exactly what that agent cannot do. It learns nothing until an episode ends, because its weight contains the sampled reward-to-go. It discards every batch after a single gradient step, billing fresh interaction for every update. And nothing in it says how large a policy step is safe, which matters more here than anywhere else in this book, because a policy generates its own future data: one oversized update does not merely waste an iteration, it degrades every batch collected afterward. This chapter pays those three debts in order, then asks two questions the first chapter deferred, what an optimizer does to a reward that is itself an estimate, and what remains when the data is a fixed log, and closes by carrying the surviving machinery to where it is spent today, language models.

The map was drawn in advance. On :numref:`fig_rl_roadmap` the solid boxes belong to :numref:`chap_reinforcement_learning`; the dashed boxes and the greyed names inside shared boxes are this chapter's six sections, and by its end every box on the map is filled. The organizing questions do not change: what is learned, and which data may drive the update. What changes is that both must now be answered under function approximation, where an update moves every state at once and the safety arguments of the tabular chapter have to be re-earned or replaced; each section ends not with a promise but with a measurement.

:numref:`sec_actorcritic` pays the first debt. The sampled tail of the Monte Carlo weight is replaced by a prediction, the temporal-difference error built from the critic, and the trade this buys, one transition's noise in exchange for the critic's bias, turns out to be a whole dial: $n$-step targets, the $\lambda$-return, and a telescoping identity that collapses generalized advantage estimation into two lines on a scan the library already owned. Both ends of the dial tie CartPole's $500$ ceiling; what separates them is stillness, and in every framework tab the longest stretch of consecutive perfect batches belongs to the bootstrapped side. A frozen-policy probe then measures the whole dial at once and finds the one-draw error lowest strictly inside it, in the $\lambda = 0.9$ to $0.95$ band where the next section's deployed implementations commonly run.

:numref:`sec_ppo` pays the second and third debts together. A two-action example shows two parameter updates of the same size, one rewriting the behavior and one changing nothing, so the quantity to control is the step in policy space; importance ratios let a batch be reused after the policy has moved; and the performance difference lemma prices both ideas with one theorem. PPO's clip replaces the trust-region constraint, and the ablation is the argument: after twenty reuse epochs per batch, most of the eight unclipped seeds in our runs end dead, in both tabs, while every clipped seed reaches the ceiling. The section's lasting product is its diagnostics, returned by the training loop as data: in the displayed runs the drift within a batch is front-loaded, about one ratio check in twenty leaves the band, and the reused batch's importance weights stay nearly flat with the clip, concentrating to half its effective size or less without.

:numref:`sec_regularized` asks what the optimizer we just strengthened does to a reward that is itself learned. Fitted from comparisons by logistic regression, the reward is accurate where the data lives and silent where it does not, and an optimal planner finds the silence: it drives straight down a hazard lane the data never priced. Swept along the exact frontier of the penalized objective, solved per exchange rate by soft value iteration with its Bellman residual checked, the true return peaks at a budget of about four nats of divergence and then falls off a cliff while the fitted return rises throughout. The repair changes the objective itself, expected reward minus $\beta$ times the KL divergence to a frozen reference. Its optimum has a closed form, proved in four lines and verified in the notebook to a largest gap near $10^{-16}$, and the entropy bonus, maximum-entropy reinforcement learning, and the KL-anchored objective of language-model post-training are all corners of the one formula.

:numref:`sec_dqn` performs the network swap on Q-learning, whose target contains the very function being trained, and it breaks immediately: function approximation, bootstrapping, and off-policy data are the deadly triad, and Baird's seven-state counterexample runs live, exact expected updates driving the weight norm from $10$ to $335$ in a thousand sweeps of a problem whose every true value is zero. Replay and the target network weaken the couplings without removing a corner, one boolean separating an agent that learns from value estimates past $10^8$. The section then measures the upward lean of the $\max$, a bias of $1.03$ from four actions of unit noise where the double estimator reads $-0.006$, and grades values against the objective the update defines: bootstrapping through the time limit makes it the continuing one, worth at most $1/(1-\gamma) = 100$ from the start for any policy, so a trace settling above that line claims what cannot be earned. Along the way it shows that the final window of a churning run is a stopping-time lottery and the retrospective best window is optimistic selection; the reported result is the fixed-budget greedy evaluation.

:numref:`sec_offline` states the rule that sorted every algorithm on the map, which data may drive which update, lets SARSA flip it with one symbol, and then severs interaction entirely: one fixed log, no second chances. On fifteen datasets collected by a random policy, naive offline Q-learning promises more than the computable optimum on all fifteen while delivering about a third of what it promised; behavior cloning's promise is calibrated and tiny. The repair is :numref:`sec_qlearning`'s count-shrinking confidence radius with its sign flipped, $\kappa/\sqrt{n}$ subtracted where data is thin, a count-based shrinkage heuristic rather than a confidence bound: one count-shrinking radius, two signs, and the sign is set by whether the loop is open. Pessimism restores roughly calibrated promises while leaving the policy no better, the currency of the setting.

:numref:`sec_rl_sequences` closes both chapters by reading text generation as a Markov decision process in the degenerate corner :numref:`sec_mdp` planted: prompts are start states, tokens are actions, transitions are concatenation, and the reward arrives once. One factorization identity makes the token-level and response-level views the same algorithm, and sorting two chapters of machinery through that dictionary, what collapses and what survives, is the sharpest statement of what each piece was for. Its smallest instance measures two claims a course could only assert: the same-group baseline is biased by self-inclusion, maximally at $K = 1$ where the update is identically zero, printing the reference's $0.062$ untouched, and a sloppy verifier is hacked exactly until the KL penalty prices the exploit away, bracketing the predicted $\beta = 1/4$. The Language Models part inherits the notation and the estimators verbatim; the policy there is the softmax head of :numref:`sec_gpt`.

Splitting the material over two chapters has one real cost, that this chapter's code leans on objects built in the last one, and :numref:`tab_rl_inherited` is the mitigation: every library object the six sections use, what it does, and the section that built it. The rows below the marked line are this chapter's additions.

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

The second table is the one to keep. Deep reinforcement learning does not fail loudly: the loss is meaningless by design (:numref:`sec_deeprl`), a diverging value function can wear a falling regression loss, and a collapsed policy keeps producing curves. The six sections offer instruments instead, every one printed by a visible cell rather than asserted, and :numref:`tab_rl_diagnostics` collects them, with what a healthy reading looks like and where each is measured. When a run of your own misbehaves, this table is the intended debugging path.

:How to know your reinforcement learning is broken. Healthy readings are quoted at this chapter's scale; the constants move with the task, the shapes do not.
:label:`tab_rl_diagnostics`

| diagnostic | healthy reading | measured in |
|:--|:--|:--|
| the ratio $\rho_t$ on the first reuse epoch | exactly $1$, by construction; anything else is a bug in the frozen log-probabilities | :numref:`sec_ppo` |
| approximate KL within a batch | small and front-loaded: it climbs for a few epochs, then flattens; climbing without flattening is runaway drift | :numref:`sec_ppo` |
| fraction of ratios outside the clip band | about one check in twenty; several times that means the reuse or the step size is too aggressive | :numref:`sec_ppo` |
| policy entropy | a slow decay, about $0.65$ to $0.25$ nats over a CartPole run; a crash toward zero is saturation, the unclipped death | :numref:`sec_ppo` |
| weight-only effective sample size of the reused batch | ratio concentration: weights nearly flat through the reuse epochs; concentration to half or less says stop reusing; blind to advantages, dependence, and state staleness | :numref:`sec_ppo` |
| correlation of the TD error with the Monte Carlo advantage | about $0.3$ early, falling toward zero as episodes lengthen, consistent with the Monte Carlo weight dissolving into tail noise; a descriptive diagnostic, not a critic certificate | :numref:`sec_actorcritic` |
| pre-clip gradient norm, and how often the clip binds | stable norms, the clip binding on a bounded fraction of updates: about a third for the bootstrapped weight and none for Monte Carlo, at matched normalization; descriptive of where variance lives, not proof of signal | :numref:`sec_actorcritic` |
| streak lengths at the ceiling | long runs of consecutive perfect batches, a run-specific stability visualization; arriving without resting points at the estimator's noise, not the policy | :numref:`sec_actorcritic` |
| value estimate at the start state | below the ceiling of the objective the update defines, $1/(1-\gamma) = 100$ on CartPole's continuing formulation, by tens of points at most; above the line is overestimation, growing without bound is divergence | :numref:`sec_dqn` |
| best against final trailing window, across seeds | descriptive statistics of the curve only: the final window is a stopping-time lottery, the retrospective best window is optimistic selection; report a predeclared fixed-budget evaluation instead | :numref:`sec_dqn`, and the boxed reading rule of :numref:`sec_baselines` |
| predicted value against delivered return | the promise at or below what any policy can earn; a promise above the computable optimum is overestimation, established with no baseline needed | :numref:`sec_offline` |

Nearly everything above runs on one recipe: CartPole, a handful of seeds, and runtimes of seconds to a few minutes per section in the pytorch tab, while the jax tab stretches the two heaviest notebooks to tens of minutes of per-step dispatch. The exceptions are deliberate. :numref:`sec_regularized` and :numref:`sec_offline` retreat to tabular gridworlds, a hazard-lane grid and the slippery lake of :numref:`sec_valueiter`, because those are the only environments in these two chapters whose true optimum is computable, and a computable optimum is exactly what lets :numref:`sec_offline` grade every promise against the truth, a diagnostic sharper than the deep-scale version it stands in for; :numref:`sec_rl_sequences` needs no simulator at all, since its transition kernel is string concatenation. A word on what this chapter is not: it contains no multi-agent, meta-, hierarchical or goal-conditioned reinforcement learning and no partial-observability machinery; no model-based agent is implemented, though :numref:`sec_rl_sequences` closes with the pointer worth keeping; and the RLHF pipeline, DPO's derivation, and the GRPO trick zoo belong to the Language Models part. The environments stay small by design: reinforcement-learning results are readable only across seeds and reruns, and the compute for those belongs to readers.

```toc
:maxdepth: 2

actor-critic
ppo
regularized
dqn
offline-rl
rl-sequences
```

## Resources and Further Reading {.unnumbered}

The list in :numref:`chap_reinforcement_learning` carries the textbooks and courses; this is the practitioner's shelf at this chapter's depth. All are freely accessible online.

- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` publishes single-file, benchmarked implementations of every deep algorithm in this chapter; :numref:`sec_ppo` ends by asking you to diff its `ppo.py` against that section.
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` documents, one detail at a time, how much reported performance lives in choices papers do not mention.
- The controlled study of :citet:`Engstrom.Ilyas.Santurkar.ea.2020` shows PPO and TRPO nearly indistinguishable at matched code-level choices: the details above, not the objective, carry much of the practical edge.
- [rliable](https://github.com/google-research/rliable) :cite:`Agarwal.Schwarzer.Castro.ea.2021` turns the seed-spread discipline of :numref:`tab_rl_diagnostics` into tooling: interval estimates and performance profiles for few-seed results.
- The [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) :cite:`Raffin.Hill.Gleave.ea.2021` ships tuned hyperparameter configurations per environment, the sensible starting point when a method of this chapter meets a task that is not CartPole.
- The offline survey of :citet:`Levine.Kumar.Tucker.ea.2020` continues :numref:`sec_offline` at full depth, from the tabular diagnosis to the deep-scale methods it could only name.
- Chapters 11 to 13 of Sutton and Barto :cite:`Sutton.Barto.2018` give the deadly triad of :numref:`sec_dqn` its full theory, including what is provable once one corner is dropped.
- [The RLHF book](https://rlhfbook.com/) :cite:`Lambert.2026` is the road of :numref:`sec_rl_sequences` continued into post-training practice; after these two chapters you own every estimator in it.
