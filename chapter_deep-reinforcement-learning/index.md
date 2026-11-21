# Deep Reinforcement Learning
:label:`chap_deep_rl`

The last agent of :numref:`chap_reinforcement_learning` swung a pendulum up after a few hundred thousand steps. A simulator makes that cheap. A robot arm would wear out first, and a language model trained on its own answers cannot discard each batch of answers after one update. The agent waited for every episode to end, used each batch once and never limited how far an update moved its policy. How much can one sample teach, and what goes wrong when we make it teach more?

Four ideas answer, each with an error of its own. An agent can trust a learned estimate of the rest of an episode instead of waiting for it :cite:`Samuel.1959,Sutton.1988`; the estimate is biased while it is young. It can reuse an older policy's data, reweighted or replayed :cite:`Lin.1992`; the data goes stale and the targets move. It can bound each update by the change in the policy's behavior :cite:`Kakade.Langford.2002,Schulman.Wolski.Dhariwal.ea.2017`; the bound limits each step, not where the steps lead. It can learn a model of the environment and plan in it :cite:`Sutton.1990`, inheriting the model's errors and computing at every step. Together, a learned target, a network shared across states and another policy's data can make values diverge, which a table never does. Controllers, game players and post-trained language models are built from these ideas.

:numref:`sec_actorcritic` asks whether a policy must wait for the episode to end. :numref:`sec_ppo` asks whether one batch can drive several updates, and how far. :numref:`sec_regularized` asks how far a policy may stray from the data behind a learned reward. :numref:`sec_dqn` asks whether a network can learn values from any transition, as a table can. :numref:`sec_sac` asks how the pendulum can be learned in a few thousand steps. :numref:`sec_model_based` asks whether an agent that learns the physics from a few hundred steps can plan its way up. :numref:`sec_offline` asks what can be learned when no new action may be tried. :numref:`sec_rl_sequences` asks what survives when the environment appends tokens and a checker gives the reward.

CartPole carries the three methods with discrete actions, so that their sample counts share one axis, and Pendulum carries soft actor-critic and the learned model. Exact answers come from the verifier task and two gridworlds: one built so that a learned reward can be exploited, and the slippery lake. The code, in PyTorch and JAX, runs in CPU minutes. :numref:`fig_rl_roadmap` and :numref:`tab_rl_map` place every method of both chapters, and :numref:`sec_deep_rl-further-reading` collects those this chapter names without teaching. Several agents, hidden state, and meta- and hierarchical learning are left out; the Language Models part treats post-training at scale, including learning from human preferences, in more depth.

```toc
:maxdepth: 2

actor-critic
ppo
regularized
dqn
sac
model-based
offline-rl
rl-sequences
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15]{.kicker}

Deep Reinforcement Learning: Sample Efficiency and Function Approximation<br>**bootstrapped critics · trust regions · KL penalties · replay buffers · world models · offline logs · LLM post-training**
:::
:::

::: {.slide title="Why Naive Policy Gradients Waste Interactions"}
[Motivation]{.kicker}

In Chapter 14, swinging up the continuous Pendulum ($u \in [-2, 2]$, where random actions score $-1{,}200$ and upright balance scores $-200$) required $150$–$300$ batches of $1{,}600$ steps each:

::: {.cols}
::: {.col}
- **Full-Episode Waiting:** Monte Carlo returns $\hat{G}_t$ cannot be evaluated until an episode terminates or hits a truncation limit, stalling learning during long rollouts.
- **Single-Use Experience:** Every batch of $1{,}600$ environment transitions drives one gradient step and is immediately discarded because the policy distribution has shifted.
:::

::: {.col}
- **Unguarded Parameter Steps:** Nothing bounds how far a single noisy gradient update shifts the action probabilities, risking irreversible policy collapse.
- **Real-World Bottlenecks:** A physical robot arm wears out long before $300{,}000$ steps, and an LLM post-training run cannot afford to waste sampled completions after one update.
:::
:::

::: {.d2l-note}
**Central question of Chapter 15:** How much information can we extract from a single transition, and what new failure modes appear when we force each sample to teach more?
:::
:::

::: {.slide title="Four Mechanisms to Extract More from Every Sample"}
[Core Ideas]{.kicker}

Every major algorithm in deep reinforcement learning trades raw sampling cost for a learned approximation that carries its own statistical error:

| Mechanism | What It Replaces | New Approximation Error Introduced | Chapter Sections |
| :--- | :--- | :--- | :--- |
| **1. Bootstrap a Critic** | Waiting for the episode tail $\hat{G}_{t+1}$ | **Critic bias** while $\hat{V}_w(s')$ is inaccurate | **§15.1, §15.4, §15.5** |
| **2. Reuse Old Transitions** | Fresh on-policy rollouts at every step | **Distribution shift** and moving regression targets | **§15.2, §15.4–15.7** |
| **3. Bound Policy Shifts** | Unconstrained parameter steps $\Delta \theta$ | **Local step constraint** (or KL bias toward $\pi_{\textrm{ref}}$) | **§15.2, §15.3, §15.8** |
| **4. Plan in a World Model** | Trial-and-error in the real environment | **Compounding rollout error** and high per-step compute | **§15.6** |

::: {.d2l-note}
Each mechanism eliminates a reason to collect fresh environment steps while introducing an estimator whose errors must be actively controlled.
:::
:::

::: {.slide title="The Deadly Triad: When Value Networks Diverge"}
[Fundamental Hazard]{.kicker}

Tabular backups contract in the supremum norm and always converge, but combining three ingredients in deep RL can cause value estimates to diverge to infinity:

::: {.cols .vc}
::: {.col}
- **1. Bootstrapping:** The regression target $y = r + \gamma \hat{V}_w(s')$ depends on the network's own current predictions rather than fixed labels.
- **2. Function Approximation:** Shared network parameters $w$ couple all states, so reducing error at visited states can inflate predictions at unvisited states.
:::

::: {.col}
- **3. Off-Policy Data:** Training transitions come from an older behavior policy or replay buffer rather than the policy currently being evaluated.
- **Baird's Counterexample (§15.4):** Even with linear features, zero noise, and exact expectations, all three together can drive weights $\|\mathbf{w}\|_\infty \to \infty$.
:::
:::

::: {.d2l-note .warn}
**Diagnostic imperative:** A falling temporal-difference loss only means the network fits its *current* moving targets—always monitor the predicted values $\hat{V}(s_0)$ or $\hat{Q}(s_0, a)$ directly!
:::
:::

::: {.slide title="Taxonomic Roadmap of Deep Reinforcement Learning"}
[Algorithmic Map]{.kicker}

::: {.cols .vc}
::: {.col}
- **Two organizing axes:** Every method in Chapters 14 and 15 is positioned by *what it learns* (policy, action-value, or dynamics model) and *which data drive the update* (on-policy, off-policy replay, or fixed offline logs).
- **Left to right progression:** We move from strict on-policy rollouts (§15.1) to bounded multi-epoch batch reuse (§15.2), off-policy replay buffers (§15.4–15.6), and purely offline datasets (§15.7–15.8).
:::

::: {.col .fig .big}
![](../img/mdl-rl-roadmap.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Running Benchmarks and Sample-Efficiency Gains"}
[Experimental Suite]{.kicker}

We measure sample efficiency and value calibration across five self-contained testbeds that run in CPU minutes:

| Benchmark | State / Action Space | Exact Ceiling or Goal | Method & Environment Steps to Goal |
| :--- | :--- | :--- | :--- |
| **CartPole-v1** | $\mathbb{R}^4$, $2$ discrete pushes | Return $500$ ($V^\infty = 100$ at $\gamma=0.99$) | Actor-Critic: $\sim 60\text{k}$ · PPO (4 epochs): $\sim 25\text{k}$ · DQN: $\sim 40\text{k}$ |
| **Pendulum-v1** | $\mathbb{R}^3$, torque $u \in [-2, 2]$ | Return $\ge -200$ ($200$-step episodes) | REINFORCE: $\sim 300\text{k}$ · SAC: $2.4\text{k}$–$11\text{k}$ · MPC Model: $200$–$600$ |
| **$4 \times 4$ Hazard Grid** | $16$ states, $4$ moves | True optimal return $V^*(s_0) = +0.59$ | Unregularized plan hacks hazard ($-2.11$); KL penalty peaks at $+0.53$ |
| **Slippery FrozenLake** | $16$ states, $4$ slippery moves | Exact optimum $V^*(s_0) = 0.180$ | Naive offline Q overestimates ($0.25$); pessimistic LCB restores $0.14$ |
| **4-Prompt Verifier** | $4$ prompts, $9$ replies | Exact match ceiling $0.50$ (ref $0.062$) | Closed-form KL threshold $\beta = 1/4$ suppresses the lenient hedge |
:::

::: {.slide title="Chapter 15 Roadmap: Eight Core Questions"}
[Study Guide]{.kicker}

| Section | Central Question | Core Mathematical Mechanism |
| :--- | :--- | :--- |
| **§15.1 Actor-Critic & GAE** | Must a policy wait for the episode to end? | TD error $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ and backward scan $\sum_l (\gamma\lambda)^l \delta_{t+l}$ |
| **§15.2 Trust Regions & PPO** | How many updates can one batch safely drive? | Per-step importance ratio $\rho_t(\theta)$ and clipped surrogate $\min(\rho_t \hat{A}_t, \textrm{clip}(\rho_t)\hat{A}_t)$ |
| **§15.3 KL-Regularized RL** | How far may a policy stray from preference data? | Reference tilt $\pi^\star(a) \propto \pi_{\textrm{ref}}(a) e^{r(a)/\beta}$ and soft backup $V_\beta = \beta \log \sum_a \pi_{\textrm{ref}} e^{Q_\beta/\beta}$ |
| **§15.4 Deep Q-Networks** | Can a network learn values from any transition? | Uniform experience replay, frozen target network $Q_{w^-}$, and Double DQN decoupling |
| **§15.5 Soft Actor-Critic** | How do we learn continuous control in $3{,}000$ steps? | Maximum-entropy target $-\beta \log \pi$, squashed Gaussian Jacobian, twin critics $\min_j Q_j$ |
| **§15.6 World Models & MPC** | Can a learned physics model plan a swing-up? | State-delta regression $\hat{f}_\phi(s,a)$, warm-started random shooting, policy distillation |
| **§15.7 Offline RL** | What can be learned when no new trial is allowed? | Empirical MDP stitching and pessimistic penalty $-\kappa/\sqrt{n(s,a)}$ (CQL / TD3+BC) |
| **§15.8 RL for Sequences** | What survives when actions are tokens? | Token credit decomposition, verifiable rewards, GRPO with $k_3$ KL loss, and DPO |
:::
