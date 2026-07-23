```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch')
```

# Actor-Critic
:label:`sec_actorcritic`

The learned baseline of :numref:`sec_baselines` still has a Monte Carlo problem. The weight $\hat{G}_t - \hat{V}(s_t)$ needs the reward-to-go $\hat{G}_t$, and the robot only knows $\hat{G}_t$ after the episode has ended, with every coin flip of the remaining trajectory baked into it. In this section we replace the sampled tail of the trajectory by an estimate, using the same bootstrapping move that took us from Value Iteration to Q-Learning in :numref:`sec_qlearning`. The result is a pair of learners that improve each other as the robot acts: an *actor*, the policy $\pi_\theta$, and a *critic*, the value estimate $\hat{V}$. The arrangement goes back to :citet:`Barto.Sutton.Anderson.1983`, long before deep learning; a convergence analysis is given by :citet:`Konda.Tsitsiklis.2000`.

## Bootstrapping the Reward-to-Go

Write out what the reward-to-go contains,

$$\hat{G}_t = r_t + \gamma\, \hat{G}_{t+1},$$

and recall that $\hat{G}_{t+1}$ is exactly the quantity that $\hat{V}(s_{t+1})$ is being trained to predict. So instead of waiting for the sampled value of $\hat{G}_{t+1}$, substitute the prediction:

$$\hat{G}_t \approx r_t + \gamma\, \hat{V}(s_{t+1}).$$

The weight on the score at step $t$ was $\hat{G}_t - \hat{V}(s_t)$ in :numref:`sec_baselines`; after the substitution it becomes

$$\delta_t = r_t + \gamma\, \hat{V}(s_{t+1}) - \hat{V}(s_t),$$
:eqlabel:`eq_td_error`

with the convention $\hat{V}(s_{t+1}) = 0$ when $s_{t+1}$ is terminal, for the same reason as in :numref:`sec_qlearning`. This quantity is called the temporal-difference error, TD error for short: the difference between what one step actually produced, $r_t + \gamma \hat{V}(s_{t+1})$, and what the critic predicted, $\hat{V}(s_t)$.

The TD error also carries the right meaning for our purpose. Suppose for a moment that the critic were exact, $\hat{V} = V^{\pi}$. Taking the expectation of :eqref:`eq_td_error` over the next state gives

$$E\big[ \delta_t \mid s_t, a_t \big] = r(s_t, a_t) + \gamma \sum_{s'} P(s' \mid s_t, a_t)\, V^{\pi}(s') - V^{\pi}(s_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t),$$

where the first equality writes the expectation over the next state using the transition function, and the second is :eqref:`eq_dynamic_programming_q` written with $V^\pi$. The right-hand side compares the value of taking action $a_t$ at $s_t$ against the value of $s_t$ itself: the advantage of the action, the same quantity that the learned baseline of :numref:`sec_baselines` estimated with $\hat{G}_t - \hat{V}(s_t)$. A single transition, one reward plus one table lookup, gives an unbiased one-sample estimate of the advantage. Compare that with $\hat{G}_t - \hat{V}(s_t)$, which needed the entire remaining trajectory.

There's no free lunch here, and it is the same lunch that Q-Learning paid for. During training $\hat{V}$ is not $V^\pi$, so $\delta_t$ is a biased estimate of the advantage, and the policy update below is no longer an unbiased gradient of $J(\theta)$. We traded variance for bias. The trade is usually worth it: the variance of the Monte Carlo tail grows with the horizon, while the bias shrinks as the critic improves.

## The Actor-Critic Algorithm

Both learners now update from the same TD error, at every single step of the episode. After the robot takes action $a_t$ at state $s_t$ and observes $r_t$ and $s_{t+1}$, compute $\delta_t$ from :eqref:`eq_td_error` and apply

$$\hat{V}(s_t) \leftarrow \hat{V}(s_t) + \alpha_w\, \delta_t, \qquad \theta \leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$
:eqlabel:`eq_actor_critic`

The critic update moves $\hat{V}(s_t)$ toward the one-step target $r_t + \gamma \hat{V}(s_{t+1})$, exactly as Q-Learning in :numref:`sec_qlearning` moves $\hat{Q}(s_t, a_t)$ toward its bootstrapped target. The actor update is the policy gradient step of :numref:`sec_baselines` with $\delta_t$ as the advantage weight. As in the previous sections we drop the $\gamma^t$ factor that the strict derivation would place in front of the actor update; Sutton and Barto's presentation of one-step actor-critic makes the same simplification in the undiscounted case.

Two learning rates appear because the two learners have different jobs. The critic chases a moving target: every actor update changes the policy, which changes the values the critic is trying to predict. Setting $\alpha_w$ larger than the effective pace of the actor lets the critic stay roughly caught up. If the actor moves much faster than the critic, the advantages are judged against stale values and the updates degrade.

Note what happened to the structure of the algorithm. REINFORCE, with or without a baseline, collects complete trajectories and updates between batches. Actor-critic updates *inside* the episode, transition by transition, like Q-Learning does. Nothing needs to terminate before learning starts, which also means the method extends to tasks that never end.

## Implementation of Actor-Critic

We run one-step actor-critic on the same FrozenLake setup as the previous three sections. The critic is a table of 16 values initialized at zero, the actor is the softmax policy of :numref:`sec_policygradient`.

```{.python .input #actor-critic-implementation-of-actor-critic-1}

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

gamma = 0.95  # Discount factor
num_episodes = 2048  # Training episodes
alpha_theta = 1.0  # Actor learning rate
alpha_w = 0.5  # Critic learning rate

env_info = d2l.make_env('FrozenLake-v1', seed=0)
env = env_info['env']
env_desc = env_info['desc']
num_states = env_info['num_states']
num_actions = env_info['num_actions']

def softmax_policy(theta, s):
    e = np.exp(theta[s] - np.max(theta[s]))
    return e / e.sum()
```

The training loop is short because everything happens per transition: one TD error, one critic update, one actor update.

```{.python .input #actor-critic-implementation-of-actor-critic-2}

def actor_critic(seed, num_episodes, alpha_theta, alpha_w, visualize=False):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    theta = np.zeros((num_states, num_actions))
    V = np.zeros(num_states)
    V_all = np.zeros((num_episodes + 1, num_states))
    pi_all = np.zeros((num_episodes + 1, num_states))
    success = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done, reached = False, 0.0
        while not done:
            probs = softmax_policy(theta, state)
            action = np.random.choice(num_actions, p=probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reached += reward
            # TD error; the value of a terminal next state is zero
            target = reward if terminated else reward + gamma * V[next_state]
            delta = target - V[state]
            # Critic: temporal-difference update.  Actor: score times delta
            V[state] += alpha_w * delta
            grad = -softmax_policy(theta, state)
            grad[action] += 1.0
            theta[state] += alpha_theta * delta * grad
            state = next_state
        success.append(reached > 0)
        V_all[ep] = V
        for s in range(num_states):
            pi_all[ep, s] = np.argmax(theta[s])
    if visualize:
        d2l.show_Q_function_progress(env_desc, V_all[1:], pi_all[1:])
        print(f'critic estimate at the start state: {V[0]:.3f} '
              f'(Value Iteration: 0.774)')
    return np.array(success, dtype=float)

success = actor_critic(seed=0, num_episodes=num_episodes,
                       alpha_theta=alpha_theta, alpha_w=alpha_w,
                       visualize=True)
```

The color in these panels is the critic, and it should look familiar: it is the same kind of picture that Value Iteration produced in :numref:`sec_valueiter`, except that nobody handed the algorithm the MDP. Values propagate backward from the goal along the path the robot actually travels, and the actor's arrows follow them. In this run the robot settled on the path down the left column, one of the two optimal six-step routes, and the critic's estimate at the start state ends near $0.77$, matching the $\gamma^5 \approx 0.774$ that Value Iteration computed for the optimal policy. Away from the traveled path the critic stays rough, for the by-now familiar reason: no visits, no updates.

How does it compare with the batch methods? We run actor-critic and normalized-reward-to-go REINFORCE for the same total number of episodes, five seeds each, and plot success against episodes consumed.

```{.python .input #actor-critic-implementation-of-actor-critic-3}

def reinforce_normalized(seed, num_iters=128, batch_size=16, alpha=2.0):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    theta = np.zeros((num_states, num_actions))
    success = []
    for it in range(num_iters):
        steps, hits = [], []
        for _ in range(batch_size):
            s, _ = env.reset()
            done, traj, rewards = False, [], []
            while not done:
                probs = softmax_policy(theta, s)
                a = np.random.choice(num_actions, p=probs)
                s2, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                traj.append((s, a))
                rewards.append(r)
                s = s2
            hits.append(rewards[-1] > 0)
            G = 0.0
            for r, (s_t, a_t) in zip(reversed(rewards), reversed(traj)):
                G = r + gamma * G
                steps.append((s_t, a_t, G))
        success.extend([np.mean(hits)] * batch_size)
        W = np.array([w for _, _, w in steps])
        W = (W - W.mean()) / (W.std() + 1e-8)
        u = np.zeros_like(theta)
        for (s_t, a_t, _), w in zip(steps, W):
            grad = -softmax_policy(theta, s_t)
            grad[a_t] += 1.0
            u[s_t] += w * grad
        theta += alpha * u / batch_size
    return np.array(success)

def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, 'valid')

d2l.set_figsize((6, 4))
comparison = [('actor-critic',
               lambda sd: actor_critic(sd, num_episodes,
                                       alpha_theta, alpha_w)),
              ('REINFORCE (normalized)', reinforce_normalized)]
for name, fn in comparison:
    curves = [fn(sd) for sd in range(5)]
    runs = np.stack([smooth(c) for c in curves])
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    x = np.arange(len(mean))
    d2l.plt.plot(x, mean, label=name)
    d2l.plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    m = [int(np.argmax(smooth(c) >= 0.9)) if (smooth(c) >= 0.9).any()
         else len(c) for c in curves]
    print(f'{name}: median episodes to 90% success = {int(np.median(m))}')
d2l.plt.xlabel('episode')
d2l.plt.ylabel('success rate')
d2l.plt.legend();
```

Actor-critic reaches a 90% success rate after a median of roughly 170 episodes in this comparison, against roughly 210 for the batch method, with heavily overlapping bands. Read that the way :numref:`sec_baselines` taught us to: on a problem this small, the two are comparable. The structural differences are what carry over to harder problems. Actor-critic updated its parameters at every one of the roughly 15,000 transitions in those 2048 episodes, needed no episode boundary to do so, and judged every action with a one-step estimate whose variance does not grow with the horizon.

## Summary

Actor-critic replaces the sampled reward-to-go in the policy gradient with a bootstrapped one-step estimate. The TD error $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ is, in expectation under an exact critic, the advantage of the action taken, so it can serve as the weight on the score function. The actor is the policy, the critic is the value table, and both update from the same $\delta_t$ at every transition. The critic introduces bias while it is still wrong, in exchange for removing the variance of the Monte Carlo tail; this is the same bargain Q-Learning struck, now on the policy side. The per-step structure removes the need for complete episodes, which batch REINFORCE cannot do without.

## Exercises

1. Sweep the two learning rates over a grid, e.g., $\alpha_\theta, \alpha_w \in \{0.1, 0.5, 1.0, 2.0\}$, with five seeds per cell. Where does the algorithm fail, and does the failure look different when the actor is too fast versus when the critic is too fast?
1. Replace the one-step target in :eqref:`eq_td_error` by a three-step target $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 \hat{V}(s_{t+3})$. What does this do to bias and variance, and where do the two extremes of this family land?
1. The actor update in :eqref:`eq_actor_critic` is biased whenever $\hat{V} \neq V^{\pi_\theta}$. Explain why the baseline argument of :numref:`sec_baselines`, which allowed any $b(s_t)$ without bias, does not cover the bootstrapped weight $\delta_t$.
1. Run actor-critic on the $8 \times 8$ FrozenLake map and compare against normalized REINFORCE there. Does the gap between the two methods change, and in which direction?

<!-- slides -->

::: {.slide title="Actor-Critic"}
The learned baseline still waits for the episode to end:
$\hat G_t$ is Monte Carlo. Bootstrap it instead —

$$\hat G_t = r_t + \gamma \hat G_{t+1}
  \;\approx\; r_t + \gamma \hat V(s_{t+1}),$$

and the advantage weight becomes the **TD error**

$$\delta_t = r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t).$$

Same move that took Value Iteration to Q-learning, applied to
the policy-gradient weight.
:::

::: {.slide title="Why the TD error is an advantage"}
If the critic were exact, $\hat V = V^\pi$:

$$E[\delta_t \mid s_t, a_t]
  = Q^\pi(s_t,a_t) - V^\pi(s_t).$$

One transition estimates the advantage: one reward, one lookup,
no dependence on the rest of the trajectory.

The price: during training $\hat V \ne V^\pi$, so the update is
biased. Variance traded for bias — Q-learning's bargain, on the
policy side.
:::

::: {.slide title="The algorithm"}
Per transition, both learners share one $\delta_t$:

$$\hat V(s_t) \mathrel{+}= \alpha_w\, \delta_t,
\qquad
\theta \mathrel{+}= \alpha_\theta\, \delta_t\,
  \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$

- **actor** = the policy, **critic** = the value table.
- Updates happen *inside* the episode — nothing must terminate
  before learning starts.
- Two learning rates: the critic chases values that move every
  time the actor moves.

@actor-critic-implementation-of-actor-critic-1
:::

::: {.slide title="The critic learns Value Iteration's picture"}
Color = critic. Values spread back from the goal along the
traveled path, without ever seeing the MDP; $\hat V(s_0) \to
0.77 \approx \gamma^5$, Value Iteration's answer.

@actor-critic-implementation-of-actor-critic-2
:::

::: {.slide title="Recap"}
- TD error = one-sample advantage estimate (unbiased only at
  $\hat V = V^\pi$).
- Median episodes to 90% on FrozenLake: ~170 vs ~210 for batch
  REINFORCE — comparable here; the structural wins are per-step
  updates, no episode boundaries, horizon-free variance.
- Next: what breaks when the steps get big, and how trust
  regions and PPO fix it.

@actor-critic-implementation-of-actor-critic-3
:::
