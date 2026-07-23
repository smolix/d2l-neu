```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch')
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q-Learning
:label:`sec_qlearning`

In the previous section, we discussed the Value Iteration algorithm which requires accessing the complete Markov decision process (MDP), e.g., the transition and reward functions. In this section, we will look at Q-Learning :cite:`Watkins.Dayan.1992` which is an algorithm to learn the value function without necessarily knowing the MDP. This algorithm embodies the central idea behind reinforcement learning: it will enable the robot to obtain its own data.
<!-- , instead of relying upon the expert. -->

## The Q-Learning Algorithm

Recall that value iteration for the action-value function in :numref:`sec_valueiter` corresponds to the update

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

As we discussed, implementing this algorithm requires knowing the MDP, specifically the transition function $P(s' \mid s, a)$. The key idea behind Q-Learning is to replace the summation over all $s' \in \mathcal{S}$ in the above expression by a summation over the states visited by the robot. This allows us to subvert the need to know the transition function.

## An Optimization Problem Underlying Q-Learning

Let us imagine that the robot uses a policy $\pi_e(a \mid s)$ to take actions. Just like the previous chapter, it collects a dataset of $n$ trajectories of $T$ timesteps each $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. Recall that value iteration is really a set of constraints that ties together the action-value $Q^*(s, a)$ of different states and actions to each other. We can implement an approximate version of value iteration using the data that the robot has collected using $\pi_e$ as

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

Let us first observe the similarities and differences between this expression and value iteration above. If the robot's policy $\pi_e$ were equal to the optimal policy $\pi^*$, and if it collected an infinite amount of data, then this optimization problem would be identical to the optimization problem underlying value iteration. But while value iteration requires us to know $P(s' \mid s, a)$, the optimization objective does not have this term. We have not cheated: as the robot uses the policy $\pi_e$ to take an action $a_t^i$ at state $s_t^i$, the next state $s_{t+1}^i$ is a sample drawn from the transition function. So the optimization objective also has access to the transition function, but implicitly in terms of the data collected by the robot.

The variables of our optimization problem are $Q(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$. We can minimize the objective with updates of a gradient-descent flavor. For every pair $(s_t^i, a_t^i)$ in our dataset, we take a step on the corresponding squared-error term while treating the bootstrapped target $r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a')$ as fixed, i.e., we do not differentiate through the $Q$ inside the target. Folding the remaining constant factors into the learning rate $\alpha$, the update becomes

$$Q(s_t^i, a_t^i) \leftarrow (1 - \alpha) Q(s_t^i,a_t^i) + \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),$$
:eqlabel:`q_learning`

where $\alpha$ is the learning rate. Because the target itself depends on $Q$ and we deliberately hold it fixed, this is called a semi-gradient step: Q-Learning is not gradient descent on $\ell(Q)$ in the strict sense, but the update is simple, cheap, and works well in practice. Typically in real problems, when the robot reaches the goal location, the trajectories end. The value of such a terminal state is zero because the robot does not take any further actions beyond this state. We should modify our update to handle such states as

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) + \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

where $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ is an indicator variable that is one if $s_{t+1}^i$ is a terminal state and zero otherwise. The value of state-action tuples $(s, a)$ that are not a part of the dataset is initialized to zero. This algorithm is known as Q-Learning.

Given the solution of these updates $\hat{Q}$, which is an approximation of the optimal value function $Q^*$, we can obtain the optimal deterministic policy corresponding to this value function easily using

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

There can be situations when there are multiple deterministic policies that correspond to the same optimal value function; such ties can be broken arbitrarily because they have the same value function.

## Exploration in Q-Learning

The policy used by the robot to collect data $\pi_e$ is critical to ensure that Q-Learning works well. After all, we have replaced the expectation over $s'$ using the transition function $P(s' \mid s, a)$ using the data collected by the robot. If the policy $\pi_e$ does not reach diverse parts of the state-action space, then it is easy to imagine our estimate $\hat{Q}$ will be a poor approximation of the optimal $Q^*$. It is also important to note that in such a situation, the estimate of $Q^*$ at *all states* $s \in \mathcal{S}$ will be bad, not just the ones visited by $\pi_e$. This is because the Q-Learning objective (or value iteration) is a constraint that ties together the value of all state-action pairs. It is therefore critical to pick the correct policy $\pi_e$ to collect data.

We can mitigate this concern by picking a completely random policy $\pi_e$ that samples actions uniformly randomly from $\mathcal{A}$. Such a policy would visit all states, but it will take a large number of trajectories before it does so.

We thus arrive at the second key idea in Q-Learning, namely exploration. Typical implementations of Q-Learning tie together the current estimate of $Q$ and the policy $\pi_e$ to set

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

where $\epsilon$ is called the "exploration parameter" and is chosen by the user. The policy $\pi_e$ is called an exploration policy. This particular $\pi_e$ is called an $\epsilon$-greedy exploration policy because it chooses the optimal action (under the current estimate $\hat{Q}$) with probability $1-\epsilon$ but explores randomly with the remainder probability $\epsilon$. We can also use the so-called softmax exploration policy

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/\tau}}{\sum_{a'} e^{\hat{Q}(s, a')/\tau}};$$

where the hyper-parameter $\tau$ is called temperature (we use $\tau$ to avoid clashing with the trajectory length $T$ introduced above). A large value of $\epsilon$ in $\epsilon$-greedy policy functions similarly to a large value of temperature $\tau$ for the softmax policy.

It is important to note that when we pick an exploration that depends upon the current estimate of the action-value function $\hat{Q}$, we need to resolve the optimization problem periodically. Typical implementations of Q-Learning make one mini-batch update using a few state-action pairs in the collected dataset (typically the ones collected from the previous timestep of the robot) after taking every action using $\pi_e$.

## The "Self-correcting" Property of Q-Learning

The dataset collected by the robot during Q-Learning grows with time. Both the exploration policy $\pi_e$ and the estimate $\hat{Q}$ evolve as the robot collects more data. This gives us a key insight into why Q-Learning works well. Consider a state $s$: if a particular action $a$ has a large value under the current estimate $\hat{Q}(s,a)$, then both the $\epsilon$-greedy and the softmax exploration policies have a larger probability of picking this action. If this action actually is *not* the ideal action, then the future states that arise from this action will have poor rewards. The next update of the Q-Learning objective will therefore reduce the value $\hat{Q}(s,a)$, which will reduce the probability of picking this action the next time the robot visits state $s$. Bad actions, e.g., ones whose value is overestimated in $\hat{Q}(s,a)$, are explored by the robot but their value is correct in the next update of the Q-Learning objective. Good actions, e.g., whose value $\hat{Q}(s, a)$ is large, are explored more often by the robot and thereby reinforced. This property can be used to show that Q-Learning can converge to the optimal policy even if it begins with a random policy $\pi_e$ :cite:`Watkins.Dayan.1992`.

This ability to not only collect new data but also collect the right kind of data is the central feature of reinforcement learning algorithms, and this is what distinguishes them from supervised learning. Q-Learning, using deep neural networks (which we will see in the DQN chapter later), is responsible for the resurgence of reinforcement learning :cite:`mnih2013playing`.

## Implementation of Q-Learning

We now show how to implement Q-Learning on FrozenLake from [Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym). Note that this is the same setup as we consider in the :numref:`sec_valueiter` experiment.

```{.python .input #qlearning-implementation-of-q-learning-1}

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learning rate
# Anneal epsilon linearly from `epsilon_start` to `epsilon_end` across the
# `num_iters` episodes: explore broadly early, exploit once Q is informative.
epsilon_start = 0.9
epsilon_end = 0.05
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

In the FrozenLake environment, the robot moves on a $4 \times 4$ grid (these are the states) with actions that are "up" ($\uparrow$), "down" ($\downarrow$), "left" ($\leftarrow$), and "right" ($\rightarrow$). The environment contains a number of holes (H) cells and frozen (F) cells as well as a goal cell (G), all of which are unknown to the robot. To keep the problem simple, we assume the robot has reliable actions, i.e., the transitions are deterministic: for each state $s$ and action $a$ there is a unique next state $s'$ with $P(s' \mid s, a) = 1$, and probability zero for all other states. If the robot reaches the goal, the trial ends and the robot receives a reward of $1$ irrespective of the action; the reward at any other state is $0$ for all actions. The objective of the robot is to learn a policy that reaches the goal location (G) from a given start location (S) (this is $s_0$) to maximize the *return*.

We first implement $\epsilon$-greedy method as follows:

```{.python .input #qlearning-implementation-of-q-learning-2}

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    # Exploit: greedy action, breaking ties uniformly at random. Plain
    # np.argmax always returns the first maximizing index, so on a
    # zero-initialized Q table every unvisited state would deterministically
    # pick action 0 ("left"), and the robot can fail to ever find the goal.
    best_actions = np.flatnonzero(Q[s,:] == np.max(Q[s,:]))
    return np.random.choice(best_actions)

```

We are now ready to implement Q-learning:

```{.python .input #qlearning-implementation-of-q-learning-3}

def q_learning(env_info, gamma, num_iters, alpha, epsilon_start, epsilon_end):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))
    episode_returns = []  # Per-episode return, for the learning curve

    for k in range(1, num_iters + 1):
        # Linearly anneal epsilon over episodes
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (
            (k - 1) / max(1, num_iters - 1))
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

            # Q-update: mask the bootstrap term only at *terminal* states
            # (their value is zero). A truncated episode is merely cut off
            # by the time limit, so we still bootstrap from the next state.
            if terminated:
                y = reward
            else:
                y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        episode_returns.append(episode_return)
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])
    return np.array(episode_returns)

episode_returns = q_learning(env_info=env_info, gamma=gamma,
                             num_iters=num_iters, alpha=alpha,
                             epsilon_start=epsilon_start,
                             epsilon_end=epsilon_end)

```

This result shows that Q-learning can find the optimal solution for this problem roughly after 250 iterations. However, when we compare this result with the Value Iteration algorithm's result (see :numref:`subsec_valueitercode`), we can see that the Value Iteration algorithm needs far fewer iterations to find the optimal solution for this problem. This happens because the Value Iteration algorithm has access to the full MDP whereas Q-learning does not.

The grid above shows the end product; the reward curve shows the learning as it happened. Each training episode on FrozenLake returns $1$ if the robot reached the goal and $0$ otherwise, so the raw signal is a jittery sequence of zeros and ones and we smooth it with a moving average:

```{.python .input #qlearning-implementation-of-q-learning-4}

d2l.set_figsize((6, 4))
window = 25
moving_avg = np.convolve(episode_returns, np.ones(window) / window, 'valid')
d2l.plt.plot(np.arange(window, num_iters + 1), moving_avg)
d2l.plt.xlabel('episode')
d2l.plt.ylabel(f'return (moving average over {window} episodes)');
```

The curve stays low while the robot explores and the goal is still a rare accident, climbs once the first successes propagate value backward through the table, and approaches one as $\epsilon$ anneals and the robot exploits what it has learned. By the end it strings together a perfect window of successes even though $\epsilon = 0.05$ keeps injecting the occasional random action: on this small grid, a single random step is usually recoverable.


## Summary
Q-learning is one of the most fundamental reinforcement-learning algorithms. It has been at the epicenter of the recent success of reinforcement learning, most notably in learning to play video games :cite:`mnih2013playing`. Implementing Q-learning does not require that we know the Markov decision process (MDP), e.g., the transition and reward functions, completely.

## Exercises

1. Try increasing the grid size to $8 \times 8$. Compared with $4 \times 4$ grid, how many iterations does it take to find the optimal value function?
1. Run the Q-learning algorithm again with $\gamma$ (i.e. "gamma" in the above code) when it equals to $0$, $0.5$, and $1$ and analyze its results.
1. Experiment with different values of `epsilon_start`, `epsilon_end`, and the epsilon decay rate (the schedule that anneals $\epsilon$ during training). For example, try `epsilon_start=epsilon_end=0` (pure greedy), `epsilon_start=epsilon_end=0.5`, `epsilon_start=epsilon_end=1` (pure random), and contrast each with a slow-decay schedule. Analyze how the exploration schedule affects convergence speed and the quality of the learned policy.

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/12103)
:end_tab:

<!-- slides -->

::: {.slide title="Q-Learning"}
Value iteration needs the full MDP. Real RL has only
*samples* from the environment. **Q-learning**
(Watkins, 1989) replaces the Bellman backup with a
sampled, off-policy update:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big].$$

Three pieces to make this work:

- **Exploration** (ε-greedy) — pick a random action with
  probability ε so all $(s, a)$ get sampled.
- **Self-correction** — over-estimates of bad actions get
  driven down as you keep visiting them.
- **Off-policy** — the bootstrap uses $\max_{a'}$
  regardless of which action the agent actually took
  next, so the learned $Q$ approximates the *optimal*
  policy, not the behavior policy.
:::

::: {.slide title="TD target and TD error"}
The update is easiest to read as a prediction-correction
step:

$$\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\textrm{TD target}}
  \quad\text{vs.}\quad
  \underbrace{Q(s,a)}_{\textrm{current estimate}}.$$

Their difference is the **temporal-difference error**:

$$\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a).$$

Then Q-learning applies an exponential moving average:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \delta.$$

Large $\alpha$ learns quickly but can be noisy; small $\alpha$
is stable but slow. In theory, convergence uses a decaying
learning-rate schedule.
:::

::: {.slide title="Exploration is not optional"}
If the agent always exploits the current table, early random
errors can lock in a bad policy. ε-greedy avoids that failure:

$$
\pi_e(a\mid s)=
\begin{cases}
\textrm{random action}, & \epsilon \\
\arg\max_{a'} Q(s,a'), & 1-\epsilon.
\end{cases}
$$

The practical tradeoff:

- high ε: broader coverage, slower apparent improvement;
- low ε: faster exploitation, higher risk of missing good actions;
- annealed ε: explore early, exploit once estimates are useful.
:::

::: {.slide title="Frozen Lake setup"}
Same gridworld as the value-iter deck — easy to see the
algorithm converge to the same answer the model-based
DP got, but without knowing $P$ in advance:

@qlearning-implementation-of-q-learning-1
:::

::: {.slide title="Q-learning training loop"}
Sample episodes; each transition gives one supervised-looking
target for one table entry. The update touches only
$Q(s_t,a_t)$, but bootstrapping lets information propagate
backward through later visits:

@qlearning-implementation-of-q-learning-2

. . .

@qlearning-implementation-of-q-learning-3
:::

::: {.slide title="Recap"}
- Q-learning = sampled, model-free version of value
  iteration.
- Uses ε-greedy exploration + off-policy bootstrap.
- Converges (with proper schedules) to the optimal Q
  function, even without knowing the dynamics.
- DQN (deep Q-learning, Mnih et al. 2015) replaces the
  table with a neural network — same update, learnable
  representation, plus a few tricks (replay buffer,
  target network) for stability.
:::
