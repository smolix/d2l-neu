# Reinforcement Learning
:label:`chap_reinforcement_learning`

Every model in this book so far was trained on a dataset that sat still: nothing the model predicted changed which example came next. Now consider a language model working through a mathematics problem. It writes a thousand tokens of reasoning, and only at the end does a checker say whether the answer is right. No token has a label of its own, and each one changes what the model can write next. A controller balancing a pole on a moving cart is in the same position: each push changes the angle it must correct next, and the only reward is the time the pole stays up. :numref:`fig_rl_agent_env` draws the loop between the decision maker, the *agent*, and the world it acts on, the *environment*.

![The loop between an agent and its environment. At step $t$ the agent, following its policy $\pi$, chooses an action $a_t$ (blue); the environment, governed by its transition probabilities $P$ and reward $r$, returns a reward $r_t$ and the next state $s_{t+1}$ (orange). Repeating the exchange produces the trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$.](../img/mdl-rl-agent-env.svg)
:label:`fig_rl_agent_env`

Two questions follow. Which decisions deserve credit for a reward that arrives late? And how can a learner find good decisions when nobody shows it what they are? Richard Bellman answered the first in the 1950s for problems whose rules are known: score each situation by the reward still to come, and a long problem becomes a chain of one-step problems :cite:`BellmanDPBook`. In the 1980s Andrew Barto, Richard Sutton and Chris Watkins learned the same scores by trial and error, with the pole on a cart as an early test :cite:`Barto.Sutton.Anderson.1983,Sutton.1988,Watkins.1989`. Ronald Williams showed how to improve the rule that picks decisions, the *policy*, directly from the rewards its attempts earn :cite:`Williams.1992`. Because the learner collects its own data, its mistakes change what it learns from next.

These ideas train controllers, game players and, after pretraining, language models that reason toward answers a checker accepts. They also explain why imitation is not enough: a model fine-tuned on an expert's text never practices recovering from its own mistakes.

:numref:`sec_mdp` asks how to write such a problem down. :numref:`sec_valueiter` asks how to find the best policy when the rules are known. :numref:`sec_qlearning` asks what can be learned from seeing only where each action leads. :numref:`sec_exploration` asks how often to try actions that look worse. :numref:`sec_imitation` asks what an expert's demonstrations teach, and why copying them fails over a long task. :numref:`sec_policygradient` asks how a policy can improve from its own attempts. :numref:`sec_baselines` asks how to make that estimate less noisy, the question behind the group-based updates that train language models. :numref:`sec_deeprl` asks what changes when a neural network replaces the table, and it ends with a map of the methods of this chapter and the next.

The code is in PyTorch and JAX, and most examples are small enough to solve exactly, so that every estimate can be checked against the exact answer. :numref:`chap_deep_rl` continues with value networks inside the update, reuse of past experience and fixed datasets. Problems with several agents are left out, and hidden state and learned models of the environment appear only briefly. The Language Models part treats post-training, including learning from human preferences, in more depth.

```toc
:maxdepth: 2

mdp
value-iter
qlearning
exploration
imitation
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
