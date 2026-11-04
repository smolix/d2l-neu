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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14]{.kicker}

Learning to act from rewards<br>
**decision processes · dynamic programming · temporal differences · exploration · imitation · policy gradient · baselines · function approximation**
:::
:::

::: {.slide title="A Thousand Tokens Earn One Verdict, and No Token Has a Label"}
[The problem]{.kicker}

- So far every dataset sat still: no prediction chose the next example.
- A language model solving a mathematics problem writes about 1,000 tokens;
  a checker says right or wrong once, at the end.
- No token has a label of its own, and each changes what can come next.
- A pole on a cart: each push changes the angle to correct next, and the
  reward is the time the pole stays up.

::: {.d2l-note}
Which decisions earn the credit for a late reward? How does a learner find
good decisions that nobody shows it?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- write a problem of acting over time as an MDP, and solve a known one;
- learn values from sampled transitions, and price the cost of exploring;
- clone an expert, and measure what its own errors cost;
- estimate a policy gradient, check it against the exact one, cut its noise;
- train network policies on CartPole and Pendulum.

**Running examples:** a 16-cell frozen lake; four prompts and a verifier.

**Idea:** score states by the reward still to come; improve the policy from
its attempts.
:::

::: {.slide title="This Chapter Learns from Data the Learner Collects Itself"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| part or chapter | the learner learns from |
|---|---|
| Basics | a fixed dataset, a label for every example |
| **Reinforcement Learning** | **its own attempts and their rewards, on problems solved exactly** |
| Deep Reinforcement Learning | value networks in the update; reused and fixed data |
| Language Models | post-training at scale, preferences |
:::

::: {.col .narrow}
- **Uses:** softmax regression, cross-entropy, gradient ascent,
  expectations.
- **Skips:** several agents; hidden state and learned models, except
  briefly.
:::
:::
:::

::: {.slide title="The Agent's Own Actions Decide Which Data It Sees"}
![](../img/mdl-rl-agent-env.svg){width=90%}

- $\pi(a \mid s)$: the policy, the agent's rule for choosing $a$ in state $s$.
- $P(s' \mid s, a)$, $r(s, a)$: the environment's transition kernel and reward.
- The agent's choices decide which trajectory $\tau$, so which data, it sees.
:::

::: {.slide title="On Slippery Ice a Command Is One of Three Equally Likely Moves"}
[Running example 1]{.kicker}

![](../img/mdl-rl-gridworld.svg){width=92%}

- S start, H hole, G goal, which pays 1; a hole or the goal ends the episode.
- Calm ice moves as commanded; slippery ice, as commanded or sideways, $1/3$ each.
:::

::: {.slide title="A Reward Six Moves Away Counts 0.77 at a Discount of 0.95"}
$$R(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_{t \geq 0} \gamma^t r_t$$

| discount $\gamma$ | horizon $1/(1-\gamma)$ | goal after six moves, $\gamma^5$ |
|---|---|---|
| 0.5 | 2 steps | 0.031 |
| 0.9 | 10 steps | 0.590 |
| 0.95 | 20 steps | 0.774 |
| 0.99 | 100 steps | 0.951 |

::: {.d2l-note}
The discount is part of the task. The weights fall by about $e$ per
horizon; this chapter uses $\gamma = 0.95$.
:::
:::

::: {.slide title="The Best Policy on Ice Beats the Shortest Path Sixteen to One"}
| policy on slippery ice | reaches the goal within 100 steps |
|---|---|
| a random command at every step | 0.014 |
| the shortest path of calm ice | 0.045 |
| the optimal policy $\pi^*$ | 0.730 |

- $\pi^*$ picks the command with the least risk of a hole, and in four
  cells that command points away from the goal.
- Computed exactly from the model; simulation agrees to within a point.

::: {.d2l-note}
On ice the shortest route is not the best policy.
:::
:::

::: {.slide title="Check: Weigh the Commands at Cell 9"}
[Check]{.kicker}

At cell 9, three of the four commands can slide the agent into the hole at 5.

1. Which command cannot, and where does it land?
2. At $\gamma = 0.95$, what is a reward 20 steps away worth?
3. Which $\gamma$ gives an effective horizon of 50 steps?

::: {.fragment .fade-up}
1. *Down*: cell 13, 8 or 10, $1/3$ each, so it slides sideways 2 times in 3.
2. $0.95^{20} \approx 0.36$.
3. $1/(1-\gamma) = 50$ gives $\gamma = 0.98$.
:::
:::

::: {.slide title="A Response Is a Trajectory with Deterministic Moves and One Reward"}
[Running example 2]{.kicker}

- State: the prompt plus the tokens so far. Action: the next token.
- Transition: append the token, with probability 1. All randomness is the
  policy's.
- Reward: once, at the end, from a *verifier* that checks the answer.
- No state recurs: the states form a tree rooted at the prompt.
- So a whole response $y$ can be one action at the prompt $x$, with reward
  $r(x, y)$: a one-step MDP.

::: {.d2l-note}
Methods for language models can score and update whole responses.
:::
:::

::: {.slide title="The Reference Passes 6.2% of the Time; the Ceiling Is 50%"}
| prompt $x$ | `2+2` | `3*4` | `8*8` | `9*9` |
|---|---|---|---|---|
| answer | 4 | 12 | 64 | 81 |

- Nine candidates shared by all prompts: 1, 3, 4, 7, 12, 19, 23, 35 and the
  *hedge* "one of 4, 12, 64 or 81".
- Verifier: reward 1 if the response equals the answer, else 0. No candidate
  solves `8*8` or `9*9`, so success is at most 0.5.
- Policy: a softmax over a table, $\pi(y \mid x) \propto \exp \theta_{xy}$;
  the reference $\theta$ is 0, except $-4$ for the hedge.
- A response sampled from the reference passes with probability 0.062.
:::

::: {.slide title="A Reward That Pays for the Wrong Thing Gets Exploited"}
- The exact-match verifier rejects the hedge; a lenient checker that accepts
  any response containing the answer pays it on every prompt.
- The reference hedges 0.2% of the time; training raises whatever pays.
- On the lake, a bonus for approaching the goal pays the agent for going in
  circles.
- A bonus that is a difference of potentials cannot be collected around a
  loop.

::: {.d2l-note}
The reward states the task; whoever writes it decides what the agent
maximizes.
:::
:::

::: {.slide title="Check: Score the Verifier Task by Hand"}
[Check]{.kicker}

1. Why does a response sampled from the reference pass with probability
   about 0.062?
2. What does a policy that always hedges score under the exact verifier, and
   under the lenient checker?
3. In this MDP, what is random and what is not?

::: {.fragment .fade-up}
1. About $1/8$ on `2+2` and on `3*4`, 0 on `8*8` and `9*9`: an average of
   about $1/16$.
2. 0 under exact match; 1 under the lenient checker, twice the true ceiling.
3. The prompt and the policy's response are random; the transitions and the
   verifier's score are not.
:::
:::

::: {.slide title="Values Answer the Credit Question, Trial and Error the Search"}
- **Credit:** which decisions deserve credit for a reward that arrives late?
- **Search:** how can a learner find good decisions nobody shows it?
- Bellman, 1950s: score each situation by the reward still to come, and a
  long problem becomes a chain of one-step problems.
- Barto, Sutton and Watkins, 1980s: learn those scores by trial and error,
  with the pole on a cart as an early test.
- Williams, 1992: improve the policy directly from the rewards its attempts
  earn.

::: {.d2l-note}
The learner collects its own data, so its mistakes change what it learns
from next.
:::
:::

::: {.slide title="Eight Sections, One Question Each"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| Markov Decision Processes | how do we write a problem of acting over time down? |
| Dynamic Programming | what is the best policy when the rules are known? |
| Temporal-Difference Learning | what can be learned from seeing only where actions lead? |
| Exploration | how often should the agent try actions that look worse? |
| Learning from Demonstrations | what does an expert teach, and why does copying fail? |
| Policy Gradient | how can a policy improve from its own attempts? |
| Baselines and Variance Reduction | how can that estimate be made less noisy? |
| Function Approximation | what changes when a network replaces the table? |
:::

::: {.slide title="Most Examples Are Small Enough to Solve Exactly"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Markov Decision Processes | the lake as arrays; the verifier task |
| Dynamic Programming | the slippery lake, solved exactly |
| Temporal-Difference Learning | the slippery lake, from sampled transitions |
| Exploration | a ten-armed bandit; a chain of states; groups of responses |
| Learning from Demonstrations | the calm lake and the expert's six-move path |
| Policy Gradient | the verifier task; the calm lake |
| Baselines and Variance Reduction | the calm lake; groups of responses per prompt |
| Function Approximation | CartPole; Pendulum, with a continuous torque |
:::

::: {.slide title="Check: Which Section Answers Each Finding?"}
[Check]{.kicker}

1. Forced to act at random one step in twenty, even $\pi^*$ reaches the goal
   about 54% of the time instead of 73%.
2. A clone of the expert leaves the expert's path in 17% of its episodes.
3. With four episodes per batch, one gradient estimate missed by more than
   three times the gradient's length.

::: {.fragment .fade-up}
1. Exploration: can a different rule learn as much while losing less?
2. Learning from Demonstrations: its own errors carry it where the expert never was.
3. Baselines and Variance Reduction: remove noise without moving the mean.
:::
:::

::: {.slide title="The Chapter's Last Agent Needs Hundreds of Thousands of Steps"}
- Pendulum: a torque in $[-2, 2]$ swings a pendulum up and holds it there;
  a return of about $-200$ or better balances it, doing nothing gives $-1200$.
- REINFORCE with a learned baseline gets there after a few hundred thousand
  steps: it waits for whole episodes and uses each batch once.
- Next, Deep Reinforcement Learning: value networks in the update, reused
  experience, fixed data; its soft actor-critic crosses $-200$ in under 10,000 steps.
- The Language Models part treats post-training, including learning from
  human preferences, in more depth.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- An MDP $(\mathcal{S}, \mathcal{A}, P, r)$ and a discount $\gamma$; at
  $\gamma = 0.95$ the horizon is 20 steps.
- The lake: on slippery ice the best policy reaches the goal 0.73 of the
  time, the shortest path 0.045.
- The verifier task: a token MDP in one step; the reference passes 0.062,
  the ceiling is 0.5.
- The reward is a design choice: a lenient checker pays the hedge.
- Two questions: credit for a late reward, search without a teacher.

**Next:** the lake and the verifier task written down as a Markov decision
process.
:::
