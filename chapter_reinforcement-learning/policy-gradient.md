# Policy Gradient
:label:`sec_policygradient`

Value iteration and Q-learning compute value functions and obtain a policy
from them by choosing the action with the largest value. In this section we
take a different approach: we write the policy itself as a differentiable
function of some parameters and adjust the parameters by gradient ascent on
the expected return :cite:`Williams.1992,Sutton.McAllester.Singh.ea.2000`. Like
Q-learning, this approach needs neither the transition kernel nor the reward
function; the agent learns from the trajectories that it collects. We derive
the gradient, implement the resulting algorithm, REINFORCE, on calm
FrozenLake, and identify the two costs that the following sections reduce.

```{.python .input #policy-gradient-policy-gradient}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #policy-gradient-policy-gradient}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #policy-gradient-1}
%%tab pytorch, jax
import gymnasium as gym
```

## Parameterizing the Policy

To improve a policy by gradient ascent, we first need to write it as a
differentiable function of parameters $\theta$.

### Softmax Preferences

For a problem with few states, a simple choice keeps one parameter
$\theta_{s,a}$, called a *preference*, for every state-action pair, and turns
the preferences at state $s$ into probabilities with the softmax function:

$$\pi_\theta(a \mid s) = \frac{\exp(\theta_{s,a})}{\sum_{a'} \exp(\theta_{s,a'})}.$$
:eqlabel:`eq_softmax_policy`

This is the same construction as the softmax exploration policy of
:numref:`sec_qlearning`, with one difference: there, the exponent held
estimated action values divided by a temperature; here, the preferences are
free parameters that we adjust directly. For example, at a state with three
actions and preferences $(0, 0, \ln 2)$, the policy is $(1/4, 1/4, 1/2)$,
because the exponentials are $(1, 1, 2)$. With finite preferences, the
policy gives every action a positive probability, so the agent keeps
exploring without a separate mechanism. As the preferences grow apart during
training, the policy becomes more and more deterministic and explores less.

### The Score Function

The gradient of $\log \pi_\theta(a \mid s)$ with respect to $\theta$ is called
the *score function*. For the softmax policy,
$\log \pi_\theta(a \mid s) = \theta_{s,a} - \log \sum_{a'} \exp(\theta_{s,a'})$.
The derivative of the second term with respect to $\theta_{s,b}$ is
$\exp(\theta_{s,b}) / \sum_{a'} \exp(\theta_{s,a'}) = \pi_\theta(b \mid s)$,
so the derivative with respect to the preference $\theta_{s,b}$ of any action
$b$ at the same state is

$$\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_{s,b}} = \mathbf{1}(b = a) - \pi_\theta(b \mid s),$$
:eqlabel:`eq_softmax_score`

and the derivative with respect to the preferences of every other state is
zero. Moving $\theta$ along the score therefore raises the preference of the
action $a$ and lowers the preferences of the other actions at $s$, each in
proportion to its probability. In the three-action example above, the score
of the first action, taken with respect to the three preferences, is
$(3/4, -1/4, -1/2)$: a step along it raises the first preference and lowers
the third one twice as much as the second. The score with respect to the
preferences at $s$ is the negative of the gradient of the cross-entropy loss
with respect to the logits in :numref:`subsec_softmax_and_derivatives`, with
these preferences as the logits and the action $a$ as the label. The class
below stores the preference table, samples actions, and performs gradient steps of this form, with the
derivative :eqref:`eq_softmax_score` written out by hand. The method
`gradient` takes a batch of episodes and one number $w_t$ per time step and
returns $\sum_t w_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, averaged
over the episodes, and `step` moves the preferences along it. The following
sections choose the numbers $w_t$.

```{.python .input #policy-gradient-the-score-function}
%%tab pytorch, jax
class SoftmaxTable:  #@save
    """A tabular softmax policy with one preference theta[s, a] per pair."""
    def __init__(self, num_states, num_actions):
        self.theta = np.zeros((num_states, num_actions))

    def probs(self, s):
        """pi(.|s) for one state or for an array of states."""
        z = self.theta[s] - self.theta[s].max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def act(self, s, rng):
        """Sample an action; the acting protocol `policy(obs, rng)`."""
        p = self.probs(s)
        return int(rng.choice(len(p), p=p))

    def gradient(self, batch, weights):
        """(1/n) sum_t w_t * grad log pi(a_t | s_t) over n episodes."""
        score = -self.probs(batch.obs)                     # -pi(b | s_t)
        score[np.arange(len(batch.act)), batch.act] += 1   # + 1(b = a_t)
        g = np.zeros_like(self.theta)
        np.add.at(g, batch.obs, weights[:, None] * score)
        return g / len(batch.ep_ends)

    def step(self, batch, weights, lr):
        """One gradient ascent step: theta += lr * gradient."""
        self.theta += lr * self.gradient(batch, weights)
```

## The Policy Gradient

We now derive the gradient of the expected return with respect to $\theta$.

### An Optimization Problem over Trajectories

Suppose the agent starts at state $s_0$ and follows $\pi_\theta$ until the
episode ends after $T$ steps, producing the trajectory
$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ with return
$R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$. The probability of a particular
trajectory is the product of the probabilities of the actions chosen by the
agent and of the transitions made by the environment:

$$P(\tau; \theta) = \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\ P(s_{t+1} \mid s_t, a_t).$$
:eqlabel:`eq_traj_prob`

The rewards add no factors, because each reward
$r_t = r(s_t, a_t, s_{t+1})$ is fixed by its transition (:numref:`sec_mdp`).
On calm ice, each transition of a possible trajectory has probability one, so
$P(\tau; \theta)$ is the product of the action probabilities alone.
If the start state or the rewards were random, factors for their
probabilities would multiply this product; these factors do not depend on
$\theta$ and drop out of every gradient below. Our objective is the expected
return, viewed as a function of the parameters,

$$J(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \big[ R(\tau) \big] = \sum_{\tau} P(\tau; \theta)\, R(\tau),$$
:eqlabel:`eq_pg_objective`

which equals $V^{\pi_\theta}(s_0)$ in the notation of :numref:`sec_valueiter`
when episodes end by termination; a time limit (:numref:`sec_mdp`) drops the
rewards that would follow it. For simplicity, we write the sum over
trajectories as if there were finitely many.

### The Log-Derivative Trick

The return of a fixed trajectory does not depend on $\theta$; only its
probability does. Therefore

$$\nabla_\theta J(\theta) = \sum_\tau R(\tau)\, \nabla_\theta P(\tau; \theta) = \sum_\tau P(\tau; \theta)\, R(\tau)\, \nabla_\theta \log P(\tau; \theta),$$

where the second step multiplies and divides by $P(\tau; \theta)$ and uses
$\nabla \log P = \nabla P / P$. This *log-derivative trick* turns the gradient
into an expectation over trajectories,

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \Big[ R(\tau)\, \nabla_\theta \log P(\tau; \theta) \Big],$$
:eqlabel:`eq_pg_gradient`

and an expectation over trajectories can be estimated by running the policy.
The first form, $\sum_\tau R(\tau)\, \nabla_\theta P(\tau; \theta)$, is not
written as such an expectation, because its terms are not weighted by the
probabilities of the trajectories; the trick supplies this weight.

### The Transition Probabilities Drop Out

At first sight, :eqref:`eq_pg_gradient` still requires the transition kernel,
which appears in $P(\tau; \theta)$. Taking the logarithm of
:eqref:`eq_traj_prob` turns the product into a sum,

$$\nabla_\theta \log P(\tau; \theta) = \nabla_\theta \Big[ \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t) \Big] = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t),$$

because the transition probabilities do not depend on $\theta$. The kernel
still determines which trajectories the agent experiences, but we never need
to evaluate it.

### The REINFORCE Estimator

We approximate the expectation in :eqref:`eq_pg_gradient` by an average over
$n$ trajectories $\tau_1, \ldots, \tau_n$ collected with the current policy:

$$\hat{g} = \frac{1}{n} \sum_{i=1}^n R(\tau_i) \sum_{t=0}^{T_i - 1} \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i),$$
:eqlabel:`eq_reinforce`

where $T_i$ is the length of the $i$-th trajectory. Each trajectory is drawn
from $P(\cdot; \theta)$, and its inner sum equals
$\nabla_\theta \log P(\tau_i; \theta)$. By :eqref:`eq_pg_gradient`, each of
the $n$ terms therefore has the expected value $\nabla_\theta J(\theta)$, and
$\hat{g}$ is an unbiased estimate of it. The algorithm that takes the
gradient ascent step $\theta \leftarrow \theta + \alpha \hat{g}$ with a step
size $\alpha$ is called REINFORCE :cite:`Williams.1992`. Each term of the
sum pushes up the log-probability of one action of a trajectory, in
proportion to the return of that trajectory. On FrozenLake, this return is
$\gamma^{k-1}$ for an episode that reaches the goal in $k$ moves and zero
otherwise (:numref:`sec_mdp`). When all returns are
nonnegative, as on FrozenLake, no term pushes its action down, and only the
size of the push differs (:numref:`fig_rl_score_ascent`); a trajectory with
return zero contributes nothing. The update adds the terms, however, and by
:eqref:`eq_softmax_score` the term of one action lowers the preferences of
the other actions at the same state. Where different actions were sampled at
the same state, the probability of a sampled action can therefore fall.
Because the gradient is an expectation under the current policy,
every update needs new trajectories from the updated policy.

![One REINFORCE step on a problem with a single action choice. The Gaussian policy is $\pi_\theta(a)=\mathcal{N}(a;\mu,1)$ with $\mu=0$, and the reward is $R(a)=0.4+2e^{-(a-2)^2/2}$, giving expected reward $J=0.92$ and exact gradient $\mathrm{d}J/\mathrm{d}\mu=0.52$. (a) Each of twelve sampled actions pushes its log-probability up in proportion to its reward. All arrows point upward because all rewards are positive. (b) After the update, with a step size chosen for visibility, the mean of the policy moves from $0$ to $0.55$, toward the actions with higher rewards.](../img/mdl-rl-score-ascent.svg)
:label:`fig_rl_score_ascent`

In the example of :numref:`fig_rl_score_ascent`, the score with respect to
the mean is $\nabla_\mu \log \mathcal{N}(a; \mu, 1) = a - \mu$. Raising the
log-probability of a sampled action therefore moves the mean toward that
action, so samples to the right of $\mu = 0$ push the mean right and samples
to the left push it left. The policy is symmetric about $0$, and every action
$a > 0$ has a larger reward than its mirror image $-a$, so the pushes to the
right outweigh those to the left in expectation, and
$\mathrm{d}J/\mathrm{d}\mu > 0$.

### The Policy Gradient Theorem

The REINFORCE estimator weights the score at every step by the return of the
whole trajectory. :citet:`Sutton.McAllester.Singh.ea.2000` showed that, for
episodes that end only at terminal states, the gradient can also be written
as

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \Big[ \sum_{t=0}^{T-1} \gamma^t\, Q^{\pi_\theta}(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big],$$
:eqlabel:`eq_pg_theorem`

where $Q^{\pi_\theta}$ is the action-value function of :numref:`sec_valueiter`.
Under a time limit, the expected return that follows $(s_t, a_t)$ also
depends on how many steps remain, and in general it differs from
$Q^{\pi_\theta}(s_t, a_t)$. Two features of this form matter later. First,
the score at step $t$ is weighted by the value of the action taken there,
which depends only on what happens from step $t$ on; :numref:`sec_baselines` derives this from
:eqref:`eq_pg_gradient` and uses it to build estimators with lower variance.
Second, the factor $\gamma^t$ weights later steps less. Many implementations
drop this factor and weight all steps equally. The resulting update direction
is not the gradient of $J(\theta)$, and in general it is not the gradient of
any function :cite:`Nota.Thomas.2020`. The tabular experiments of this section
and the next keep the factor. The estimator :eqref:`eq_reinforce` keeps it
implicitly: the rewards from step $t$ on enter $R(\tau)$ as
$\gamma^t \sum_{t' \geq t} \gamma^{t'-t} r_{t'}$. :numref:`sec_deeprl`
returns to the practice of dropping it.

## REINFORCE on Calm Ice

We now run REINFORCE on calm FrozenLake with $\gamma = 0.95$.

### Collecting Trajectories

A batch of episodes is stored in flat arrays, one entry per time step,
together with the positions where the episodes end. The class `Batch` also
records whether each step ended in a terminal state, which the bootstrapped
methods of later sections need (:numref:`sec_mdp`):

```{.python .input #policy-gradient-an-optimization-problem-over-trajectories-1}
%%tab pytorch, jax
class Batch:  #@save
    """A flat batch of transitions, plus the episode boundaries."""
    def __init__(self, obs, act, rew, next_obs, term, ep_ends):
        self.obs, self.act, self.rew = obs, act, rew
        self.next_obs, self.term = next_obs, term
        self.ep_ends = ep_ends    # one past the last step of each episode

    def __len__(self):
        return len(self.rew)

    def episodes(self):
        """Yield one slice per episode."""
        start = 0
        for end in self.ep_ends:
            yield slice(start, end)
            start = end

    def episode_returns(self, gamma=1.0):
        """R(tau), the discounted return, one number per episode."""
        return np.array([(gamma ** np.arange(ep.stop - ep.start)
                          * self.rew[ep]).sum() for ep in self.episodes()])
```

The function `rollout` fills a batch by running a policy, given in the same
form `policy(obs, rng) -> action` as for `evaluate` in
:numref:`sec_qlearning`:

```{.python .input #policy-gradient-an-optimization-problem-over-trajectories-2}
%%tab pytorch, jax
def rollout(env, policy, num_episodes, rng):  #@save
    """Collect complete episodes from `policy(obs, rng) -> action` as a
    Batch; `term` records `terminated`, never `truncated`."""
    cols, ep_ends = [[] for _ in range(5)], []
    for _ in range(num_episodes):
        obs, done = env.reset()[0], False
        while not done:
            act = policy(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            for col, val in zip(cols, (obs, act, reward, next_obs,
                                       float(terminated))):
                col.append(val)
            obs = next_obs
        ep_ends.append(len(cols[0]))
    obs, act, rew, next_obs, term = (np.asarray(c) for c in cols)
    return Batch(obs, act, rew.astype(np.float32), next_obs,
                 term.astype(np.float32), np.asarray(ep_ends))
```

### The Implementation

Each update collects a batch of $16$ episodes with the current policy and
takes one step along :eqref:`eq_reinforce`, in which every time step of
episode $i$ has the weight $R(\tau_i)$. The mean return of the batch is a
Monte Carlo estimate of $J(\theta)$ for the policy that collected it, so we
record it as the learning curve:

```{.python .input #policy-gradient-the-reinforce-estimator}
%%tab pytorch, jax
gamma = 0.95

def train_reinforce(seed, num_updates=256, batch_size=16, lr=2.0):
    rng = np.random.default_rng(seed)
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env.reset(seed=seed)
    policy, curve = SoftmaxTable(16, 4), []
    for _ in range(num_updates):
        batch = rollout(env, policy.act, batch_size, rng)
        R = batch.episode_returns(gamma)              # one return per episode
        weights = np.repeat(R, np.diff(batch.ep_ends, prepend=0))
        policy.step(batch, weights, lr)
        curve.append(R.mean())                        # an estimate of J
    return policy, np.array(curve)

runs = [train_reinforce(seed) for seed in range(5)]
d2l.plot_curves({'REINFORCE': np.array([curve for _, curve in runs])},
                xlabel='update', ylabel='mean return of the batch',
                reference=gamma ** 5)
```

The dashed line marks $\gamma^5 \approx 0.774$, the return of a shortest path
and the supremum of $J(\theta)$, which no policy with finite preferences
reaches. As in :numref:`sec_qlearning`,
the solid curve is the median over the five seeds, and the band spans the
smallest and largest values. For the first few dozen
updates, the return stays close to zero. Under the initial uniform policy, an
episode reaches the goal with probability of only about $0.014$, the
undiscounted value of this policy at the start state (:numref:`sec_valueiter`),
so a batch of $16$ episodes contains no success with probability about $0.8$.
A batch without a success produces no update, since all of its weights
$R(\tau_i)$ are zero, and each success changes the
policy only a little. Once the successes accumulate, the return rises toward
the dashed line, at different times for different seeds.
At the end, the median stays slightly below the line, because the policy
remains stochastic and an occasional sampled action lengthens an episode or
ends it in a hole. The panel below shows, for one seed, the most probable
action at every state, with the color giving its probability:

```{.python .input #policy-gradient-the-reinforce-estimator-2}
%%tab pytorch, jax
policy = runs[0][0]
probs = policy.probs(np.arange(16))
d2l.show_grid(gym.make('FrozenLake-v1').unwrapped.desc, probs.max(axis=1),
              probs.argmax(axis=1))
```

Along a shortest path from the start to the goal, the most probable action is
nearly certain. Preferences at the holes and at the goal never change,
because the agent never acts from these states, so the panel shows the
uniform probability $0.25$ there. Several states off this path keep
probabilities closer to uniform.

### The Cost of On-Policy Learning

Each of the $256$ updates above consumed a fresh batch of $16$ episodes,
$4{,}096$ episodes per run. The reason is structural. The estimator
:eqref:`eq_reinforce` is an average over trajectories of the *current*
policy, so once $\theta$ changes, the trajectories collected earlier come from
a different distribution and no longer give an unbiased estimate. Methods
that can learn only from data generated by the policy being improved are
called *on-policy*. Q-learning, which is off-policy (:numref:`sec_qlearning`),
can reuse transitions collected by other policies, provided that these data
cover the state-action pairs that matter. :numref:`sec_ppo` shows how a
policy-gradient method can reuse a batch for several updates.

The experiment also exposes the second cost. Many of the early batches contain
no successful episode and produce no update at all, and once successes occur,
the size of an update depends on how many of the $16$ episodes happened to
succeed. The estimator is unbiased, but its variance is high. The next section
reduces it.

## Summary

Policy-gradient methods represent the policy as a differentiable function, for
example a softmax over one preference per state-action pair, and adjust its
parameters by gradient ascent on the expected return. The log-derivative
trick turns the gradient into an expectation over trajectories in which the
transition probabilities drop out, so the method needs no model. REINFORCE
estimates this expectation from a batch of trajectories, weighting the score
of every action by the return of its trajectory. On calm FrozenLake, it
learned a policy that follows a shortest path with high probability. The
method is on-policy, so every update needs new trajectories, and its
estimates have high variance, which the next section reduces.

## Exercises

1. **The zero-sum score.** Verify that the derivatives
   :eqref:`eq_softmax_score` sum to zero over $b$. Show that, as a
   consequence, a gradient step leaves the average of the preferences at each
   state unchanged. Why does this not restrict which policies the softmax can
   reach?
1. **An exact check.** Consider a problem with one state, two actions, and
   rewards $1$ and $0$, and the softmax policy with preferences
   $(\theta_1, \theta_2)$. Compute $J(\theta)$ and its gradient exactly, and
   show that the REINFORCE estimator with a single sampled action has this
   gradient as its expected value.
1. [code] **Batch size.** Run the section's experiment with batches of $4$,
   $16$, and $64$ episodes, keeping the total number of episodes fixed. Which
   setting reaches a high return with the fewest episodes, and which gives the
   smoothest curve?
1. [code] **The discount factor.** Repeat the experiment with
   $\gamma \in \{0.5, 0.95, 1\}$. With $\gamma = 1$, every successful
   episode has return one, whatever its length. Does the learned policy still
   follow a shortest path? Explain your observation using
   the estimator :eqref:`eq_reinforce`.
1. [code] **A larger lake.** Run REINFORCE on the $8 \times 8$ map,
   `gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)`. How many
   updates pass before the first successful episode, and why does this number
   grow so quickly with the size of the map?

[Discussions](https://d2l.discourse.group/t/36798)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14.4]{.kicker}

Policy gradient<br>
**softmax preferences · the log-derivative trick · REINFORCE · the policy gradient theorem · on-policy cost**
:::
:::

::: {.slide title="A Policy with Parameters"}
$$\pi_\theta(a \mid s) = \frac{\exp(\theta_{s,a})}{\sum_{a'} \exp(\theta_{s,a'})},
\qquad
\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_{s,b}} = \mathbf{1}(b = a) - \pi_\theta(b \mid s)$$

@policy-gradient-the-score-function
:::

::: {.slide title="The Gradient of the Expected Return"}
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \Big[ R(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

- log-derivative trick: $\nabla P = P\, \nabla \log P$;
- the transition probabilities do not depend on $\theta$ and drop out.

. . .

REINFORCE: average over $n$ trajectories of the current policy, then
$\theta \leftarrow \theta + \alpha \hat{g}$.
:::

::: {.slide title="REINFORCE on Calm Ice"}
@policy-gradient-the-reinforce-estimator
:::

::: {.slide title="The Policy Gradient Theorem"}
For episodes that end only at terminal states (no time limit):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \Big[ \sum_{t} \gamma^t\, Q^{\pi_\theta}(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

Dropping $\gamma^t$, as many implementations do, gives a direction that is
not the gradient of $J(\theta)$ and in general not the gradient of any
function :cite:`Nota.Thomas.2020`.
:::

::: {.slide title="Two Costs"}
- The method is on-policy: every update needs fresh trajectories
  ($4{,}096$ episodes per run here).
- The variance is high: batches without a success produce no update, and the
  size of the update depends on how many episodes happened to succeed.

. . .

The next section reduces the variance.
:::
