# Reinforcement Learning
:label:`chap_reinforcement_learning`

Every model in this book so far was trained on a dataset that sat still: nothing the model predicted changed which example came next. Now consider a language model working through a mathematics problem. It writes a thousand tokens of reasoning, and only at the end does a checker say whether the answer is right. No token has a label of its own, and each one changes what the model can write next. A controller balancing a pole on a moving cart is in the same position: each push changes the angle it must correct next, and the only reward is the time the pole stays up. :numref:`fig_rl_agent_env` draws this loop between the decision maker, the *agent*, and the world it acts on, the *environment*.

![The loop between an agent and its environment. At step $t$ the agent, following its policy $\pi$, chooses an action $a_t$ (blue); the environment, governed by its transition probabilities $P$ and reward $r$, returns a reward $r_t$ and the next state $s_{t+1}$ (orange). Repeating the exchange produces the trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$.](../img/mdl-rl-agent-env.svg)
:label:`fig_rl_agent_env`

Two questions follow: which decisions deserve credit for a reward that arrives late, and how can a learner find good decisions when nobody shows it what they are? Richard Bellman answered the first in the 1950s when the rules are known: score each state by the reward still to come, turning a long horizon into a chain of one-step lookaheads :cite:`BellmanDPBook`. In the 1980s Andrew Barto, Richard Sutton, and Chris Watkins learned those scores by trial and error :cite:`Barto.Sutton.Anderson.1983,Sutton.1988,Watkins.1989`, and Ronald Williams showed how to improve the decision rule—the *policy*—directly from the rewards its attempts earn :cite:`Williams.1992`. Because the learner collects its own data, its mistakes change what it learns from next, which also explains why imitation alone falls short when a model never practices recovering from its own errors.

The chapter moves in three stages. First, we formalize Markov decision processes (:numref:`sec_mdp`), compute exact optimal policies when the rules are known (:numref:`sec_valueiter`), learn action values from sampled transitions (:numref:`sec_qlearning`), and quantify exploration regret (:numref:`sec_exploration`). Next, we study what expert demonstrations miss over long horizons (:numref:`sec_imitation`) and what continuous dynamic programming becomes under known physics via the Linear-Quadratic Regulator (LQR), iterative LQR (iLQR), and receding-horizon Model Predictive Control (MPC) (:numref:`sec_optimal_control`). Finally, we optimize policies directly via score-function gradients (:numref:`sec_policygradient`), cut their variance with baselines and group-relative weights (:numref:`sec_baselines`), and replace lookup tables with neural networks before mapping out the methods of this chapter and the next (:numref:`sec_deeprl`).

All examples run in PyTorch and JAX, and most are small enough to check every estimate against an exact solution. :numref:`chap_deep_rl` continues with bootstrapped critics, experience replay, learned dynamics models, and fixed offline datasets, while the Language Models part treats preference-based post-training in greater depth.

```toc
:maxdepth: 2

mdp
value-iter
qlearning
exploration
imitation
optimal-control
policy-gradient
baselines
deep-rl
```

## Resources and Further Reading {.unnumbered}

The following freely available resources provide theoretical treatments, courses, implementations, and empirical guidance.

**Textbooks**

- Sutton and Barto, *Reinforcement Learning: An Introduction* :cite:`Sutton.Barto.2018` provides a comprehensive introduction to value methods, policy methods, planning, and function approximation. Sutton and Barto received the 2024 Turing Award for their foundational contributions to reinforcement learning.
- Szepesvári, *Algorithms for Reinforcement Learning* :cite:`Szepesvari.2010` states this chapter's algorithms and their guarantees in under a hundred pages.
- Bertsekas, *A Course in Reinforcement Learning* :cite:`Bertsekas.2025` develops reinforcement learning from the perspective of optimal control and dynamic programming.
- Agarwal, Jiang, Kakade, and Sun, *Reinforcement Learning: Theory and Algorithms* :cite:`Agarwal.Jiang.Kakade.ea.2019` supplies the sample-complexity rates this chapter states qualitatively.
- Lattimore and Szepesvári, *Bandit Algorithms* :cite:`Lattimore.Szepesvari.2020` is the full theory behind :numref:`sec_exploration`.
- For :numref:`sec_optimal_control`, :citet:`Kalman.1960` and :citet:`Anderson.Moore.2007` develop the Linear-Quadratic Regulator (LQR), Differential Dynamic Programming :cite:`Mayne.1966` and iterative LQR :cite:`Li.Todorov.2004` extend backward Riccati passes to nonlinear trajectory optimization, :citet:`Rawlings.Mayne.Diehl.2017` covers constrained receding-horizon Model Predictive Control, and :citet:`Tedrake.2023` treats underactuated robotics and trajectory optimization.

**Courses**

- [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) covers policy gradients, value methods, model-based learning, and offline reinforcement learning at the graduate level.
- [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) is the course closest in scope to this chapter: foundations, exploration, and policy gradients with their guarantees.
- [Stanford CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) covers the practice, imitation and offline methods included.
- [CMU 10-703: Deep Reinforcement Learning and Control](https://cmudeeprl.github.io/703website/) pairs the algorithms with control and physical embodiment.
- [MIT 6.7920: Reinforcement Learning: Foundations and Methods](https://web.mit.edu/6.7920/www/) develops the dynamic-programming theory introduced in :numref:`sec_valueiter`.

**Annotated implementations**

- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` provides single-file, benchmarked implementations of the major deep reinforcement learning algorithms discussed after :numref:`sec_deeprl`.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) :cite:`Raffin.Hill.Gleave.ea.2021` provides library implementations of the same algorithms, with tuned hyperparameters in its accompanying zoo.
- [Gymnasium](https://gymnasium.farama.org/) :cite:`Towers.Kwiatkowski.Terry.ea.2024` documents the environment interface used throughout these chapters.

**Empirical practice**

- [Spinning Up in Deep RL](https://spinningup.openai.com/) :cite:`Achiam.2018` is a short curriculum from the score function to SAC, written to be read beside running code.
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` documents how implementation choices omitted from algorithm descriptions affect PPO performance and reproducibility.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14]{.kicker}

Foundations of Reinforcement Learning<br>**Markov decision processes · dynamic programming · temporal differences · exploration · imitation · optimal control · policy gradients · variance reduction**
:::
:::

::: {.slide title="Acting Changes Which Data the Learner Sees Next"}
[The Sequential Decision Loop]{.kicker}

In supervised learning the dataset sits still, whereas a sequential agent's decisions alter future states and determine which data arrive next:

::: {.cols .vc}
::: {.col}
- **Policy $\pi(a_t \mid s_t)$:** Maps the current state $s_t$ to an action $a_t$ (such as a motor torque, grid move, or generated token).
- **Environment dynamics $P(s_{t+1} \mid s_t, a_t)$:** Evolves the state stochastically and emits a scalar reward $r_t$.
- **Closed-loop feedback:** Repeating the interaction generates a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ whose distribution depends on $\pi$.
:::

::: {.col .fig}
![](../img/mdl-rl-agent-env.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Delayed Credit Assignment and Trial-and-Error Search"}
[Two Core Challenges]{.kicker}

::: {.cols .vc}
::: {.col}
$$R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t, \qquad \gamma \in [0, 1)$$

- **Temporal credit assignment:** Which earlier decisions deserve credit or blame when reward arrives only at the end of a long rollout?
- **Bellman's principle (1957):** Score every state by its expected discounted future return, turning a horizon-$1/(1-\gamma)$ problem into one-step lookaheads.
- **Exploration without a teacher:** Balance exploiting high-value actions against exploring uncertain ones.
:::

::: {.col .fig}
@fig:ch14-rl-delayed-credit
:::
:::
:::

::: {.slide title="Benchmark 1: Slippery Ice Reverses Shortest-Path Logic"}
[Tabular MDP Benchmark]{.kicker}

Our first running benchmark is $4 \times 4$ `FrozenLake-v1`, small enough to solve to machine precision yet stochastic enough to defeat naive greedy heuristics:

::: {.cols .vc}
::: {.col}
- **Slippery transition kernel:** Each command moves in the intended direction with probability $1/3$ and slips perpendicularly to each side with probability $1/3$.
- **Shortest path vs. slip-aware optimum:** Following the 6-move calm-ice shortest path succeeds in only **$4.5\%$** of episodes because slips drop it into adjacent holes.
- **Risk-minimizing control:** The optimal policy $\pi^*$ reaches the goal in **$73.0\%$** of episodes—pointing *away* from the goal in four cells to minimize hole risk.
:::

::: {.col .fig}
![](../img/mdl-rl-gridworld.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Benchmark 2: Language Generation as a Verifier MDP"}
[Language Model Post-Training Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- **Collapsed one-step view:** With deterministic token appending and terminal reward $r(x, y) \in \{0, 1\}$, an entire response $y$ acts as one action at prompt $x$.
- **Reference vs. ceiling:** The uniform softmax reference policy passes $6.2\%$ of the time ($\approx 1/8$ on `2+2` and `3*4`, $0$ on unsolvable `8*8` and `9*9`), against a $50\%$ ceiling.
- **Reward hacking hazard:** A ninth candidate hedges (*"4, 12, 64 or 81"*); an exact checker scores it $0$, whereas a lenient substring checker pays $1.0$ on every prompt.
:::

::: {.col .fig .big}
@fig:ch14-rl-verifier-benchmark
:::
:::
:::

::: {.slide title="Chapter §{num:chap_reinforcement_learning} Roadmap: From Exact Tables to Neural Critics"}
[Architectural & Algorithmic Progression]{.kicker}

::: {.cols .vc}
::: {.col}
- **Known to unknown physics (§§{sec_mdp}–§{num:sec_exploration}):** Formulate MDPs, solve Bellman contraction sweeps, learn $Q^*$ from single samples, and bound exploration regret via UCB and Thompson sampling.
- **Demonstrations & continuous physics (§§{sec_imitation}–§{num:sec_optimal_control}):** Diagnose $O(\varepsilon T^2)$ cloning shift (DAgger), and solve continuous linear/nonlinear control via Riccati LQR, iLQR, and receding-horizon MPC.
- **Policy gradients & neural critics (§§{sec_policygradient}–§{num:sec_deeprl}):** Differentiate stochastic policies via REINFORCE, baselines, RLOO, GRPO, and `tanh` MLP critics.
:::

::: {.col .fig .big}
@fig:ch14-rl-chapter-roadmap
:::
:::
:::

::: {.slide title="Concept Check: Framing Sequential Decision Problems"}
[Check Your Understanding]{.kicker}

Work through these three foundational questions before diving into §{sec_mdp}:

1. **Credit assignment & discounting:** Which discount factor $\gamma$ gives an effective planning horizon $1/(1-\gamma)$ of $50$ steps, and how much weight does a reward $50$ steps away receive?
2. **Stochastic control:** At cell 9 of slippery FrozenLake (hole 5 directly above), why does the optimal policy command *down* even though *down* slides sideways two times in three?
3. **Reward specification:** Why does a lenient verifier that checks whether the true answer appears as a substring fail on the 4-prompt arithmetic task?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Setting $1/(1-\gamma) = 50$ yields $\gamma = 0.98$, and a reward one horizon away carries weight $0.98^{50} \approx e^{-1} \approx 0.368$.
2. Commanding *left*, *right*, or *up* from cell 9 slips into hole 5 with probability $1/3$, whereas *down* has $0$ one-step hole risk and lands on safe cells 8, 13, or 10.
3. The optimizer exploits the specification by outputting the hedge *"one of 4, 12, 64 or 81"* on every prompt, earning $1.0$ without solving the arithmetic.
:::
:::

