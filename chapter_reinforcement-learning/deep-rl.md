```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch')
```

# Deep RL: Neural Network Policies and Value Functions
:label:`sec_deeprl`

Every algorithm so far stored its knowledge in a table: sixteen rows for FrozenLake's states, one entry per state-action pair. Tables stop working the moment the state is a vector of real numbers, because there are infinitely many rows and the robot will never see the same state twice. This section makes the jump from tables to neural networks, and the point of the section is how little changes. The derivations of :numref:`sec_policygradient` and :numref:`sec_baselines` never used the table; they used only the score $\nabla_\theta \log \pi_\theta(a \mid s)$ and a value estimate. Replace the table with any differentiable function and every algorithm carries over as it stands.

## A Task That No Table Can Hold

CartPole is the classic small control problem: a cart slides along a track with a pole hinged on top, the state is four real numbers (cart position, cart velocity, pole angle, angular velocity), and the two actions push the cart left or right. Every step the pole stays up earns reward $1$; the episode ends when the pole tips too far, the cart leaves the track, or 500 steps pass. The best possible return is therefore 500. A table over four continuous coordinates does not exist, so $\pi_\theta$ and $\hat{V}$ must become functions.

We use the smallest reasonable choice: a network with one hidden layer of 64 units for the policy, mapping the four state numbers to two action preferences, and a second network of the same shape for the value estimate. The softmax of :eqref:`eq_softmax_policy` now sits on top of the policy network's outputs instead of a row of the preference table. That is the entire change. In :numref:`sec_policygradient` we differentiated the softmax by hand in :eqref:`eq_softmax_score`; here automatic differentiation does the same work through the hidden layer, and the score $\nabla_\theta \log \pi_\theta(a \mid s)$ now includes gradients for every weight in the network.

One implementation habit deserves a sentence before the code. Deep learning frameworks want a loss to minimize, so instead of assembling the estimator $\hat{u}$ of :numref:`sec_baselines` by hand, we write the scalar

$$L(\theta) = -\frac{1}{N} \sum_{\textrm{steps}} \hat{A}_t\, \log \pi_\theta(a_t \mid s_t),$$

with the advantages $\hat{A}_t$ treated as fixed numbers. Its gradient is $-\hat{u}$ up to the positive constant that turns the per-trajectory average into a per-step average, so one optimizer step on $L$ is one gradient ascent step on the return. The value of $L$ itself means nothing; a falling loss does not indicate progress here, only the return curve does.

## Implementation on CartPole

The helpers mirror :numref:`sec_baselines`: sample an episode, compute reward-to-go, collect a batch. The discount is $\gamma = 0.99$, the common choice for CartPole's horizon of up to 500 steps.

```{.python .input #deep-rl-implementation-on-cartpole-1}

%matplotlib inline
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from d2l import torch as d2l

gamma = 0.99  # Discount factor
num_updates = 60  # Gradient updates per training run
batch_size = 8  # Episodes per update
num_seeds = 3  # Independent runs per algorithm

def make_nets():
    policy = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    value = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))
    return policy, value

def sample_episode(env, policy):
    state, _ = env.reset()
    states, actions, rewards, done = [], [], [], False
    while not done:
        states.append(state)
        probs = torch.softmax(policy(torch.tensor(state)), dim=-1)
        action = torch.multinomial(probs, 1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards

def reward_to_go(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]

def collect(env, policy, batch_size):
    S, A, G, returns = [], [], [], []
    for _ in range(batch_size):
        states, actions, rewards = sample_episode(env, policy)
        returns.append(sum(rewards))
        S += states
        A += actions
        G += reward_to_go(rewards)
    return (torch.tensor(np.array(S)), torch.tensor(A),
            torch.tensor(G), np.mean(returns))
```

REINFORCE with a learned baseline is :numref:`sec_baselines` verbatim, with the two tables replaced by the two networks and the hand-computed score replaced by autograd:

```{.python .input #deep-rl-implementation-on-cartpole-2}

def train_reinforce(seed, lr=1e-2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy, value = make_nets()
    opt_policy = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_value = torch.optim.Adam(value.parameters(), lr=lr)
    curve = []
    for it in range(num_updates):
        S, A, G, avg_return = collect(env, policy, batch_size)
        curve.append(avg_return)
        # Advantage: reward-to-go minus the value baseline, normalized
        adv = G - value(S).squeeze(-1).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = torch.log_softmax(policy(S), dim=-1)
        logp = logp.gather(1, A[:, None]).squeeze(-1)
        policy_loss = -(adv * logp).mean()
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()
        # Critic: regression of the value network on reward-to-go
        value_loss = ((value(S).squeeze(-1) - G) ** 2).mean()
        opt_value.zero_grad()
        value_loss.backward()
        opt_value.step()
    return np.array(curve)
```

We run 60 updates of 8 episodes each, three seeds, and plot the average return of each batch:

```{.python .input #deep-rl-implementation-on-cartpole-4}

d2l.compare_agents({'REINFORCE + baseline': train_reinforce}, num_seeds)
```

The curve climbs from a return near 20, the score of the untrained random policy, toward the ceiling of 500, and averages above 400 over the final updates. The tabular derivation survived the move without a single new equation. What did change is the texture of training. The curve dips and recovers along the way, because a network generalizes: an update driven by one batch moves the policy and the values at every state, including states the batch never visited, for better and for worse. Tables never did that, and this coupling across states will return as a central difficulty when we train value functions by bootstrapping in :numref:`sec_dqn`.

One Monte Carlo habit also survived the move untouched. The value network is trained on reward-to-go targets, so nothing can be learned from an episode until it has ended, and every advantage still carries the noise of the entire sampled tail of the trajectory. The next section removes that dependence by giving the critic the bootstrapped one-step target that Q-Learning built its update on.

## Summary

Tables hold one number per state, so they cannot represent a policy or a value function over continuous states. Replacing the preference table with a small network and the value table with another changes nothing in the algorithms: the policy gradient machinery of :numref:`sec_policygradient` and :numref:`sec_baselines` only ever asked for $\log \pi_\theta$ and its gradient, which automatic differentiation supplies. In practice the estimator is written as a scalar loss whose gradient equals the estimator, with advantages held fixed; the loss value itself carries no meaning. On CartPole, REINFORCE with a learned baseline reaches near-ceiling returns within 480 episodes. Networks also introduce generalization across states, which speeds learning and destabilizes it at the same time.

## Exercises

1. Vary the hidden width over $\{8, 32, 64, 256\}$. How small can the policy network be and still balance the pole, and what changes about the speed and stability of training?
1. `train_reinforce` uses the *undiscounted* episode return for the learning curve but discounted reward-to-go inside the update. Change the curve to plot discounted returns and explain why the two tell the same story here.
1. Sweep the batch size over $\{1, 8, 32\}$ with the total number of sampled episodes held fixed, so smaller batches take more updates. Which end learns fastest per episode, and which end gives the smoothest curve?
1. Replace the normalized advantage in `train_reinforce` with the raw reward-to-go, as in :numref:`sec_policygradient`. How much slower is learning on CartPole, and why is the gap larger than it was on FrozenLake?

<!-- slides -->

::: {.slide title="Deep RL: from tables to networks"}
FrozenLake: 16 states, a table works. CartPole: the state is
four real numbers, no two visits alike, no table possible.

The fix is small because the theory never used the table:

- policy: MLP $4 \to 64 \to 2$, softmax on top —
  :eqref:`eq_softmax_policy` unchanged;
- value: MLP $4 \to 64 \to 1$ as the learned baseline;
- the score $\nabla_\theta \log \pi_\theta(a\mid s)$: autograd,
  instead of the hand-derived :eqref:`eq_softmax_score`.
:::

::: {.slide title="The estimator becomes a loss"}
Frameworks minimize losses, so write

$$L(\theta) = -\frac{1}{N}\sum_t \hat A_t \log \pi_\theta(a_t \mid s_t),
\qquad \hat A_t \ \text{held fixed},$$

whose gradient is $-\hat u$ up to a positive constant. One
optimizer step on $L$ = one ascent step on the return.

The loss *value* means nothing. Only the return curve does.
:::

::: {.slide title="CartPole results"}
60 updates $\times$ 8 episodes, three seeds:

@deep-rl-implementation-on-cartpole-2

. . .

@deep-rl-implementation-on-cartpole-4
:::

::: {.slide title="Recap"}
- REINFORCE + baseline transfers verbatim; above 400 of the
  500 ceiling within 480 episodes.
- New with networks: generalization across states — updates
  move states the batch never visited. It speeds learning and
  causes the dips.
- Still Monte Carlo: the critic waits for the episode to end.
  Next deck bootstraps it.
:::
