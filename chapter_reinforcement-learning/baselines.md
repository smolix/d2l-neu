# Baselines and Variance Reduction
:label:`sec_baselines`

The REINFORCE estimator of :numref:`sec_policygradient` is unbiased, but its
variance is high. On FrozenLake, a trajectory that fails has return zero and
contributes nothing, and the size of each update depends on how many episodes
of a batch happened to succeed. In this section, we reduce the variance,
mostly by adding or removing terms whose expected value is zero. These terms
come from one identity about the score function. It allows us to drop the
rewards received before an action and to subtract a *baseline* from the
return. A natural baseline is the value function, which turns the weights into
advantage estimates. GRPO, a method for language models, instead uses the
mean return of a group of trajectories that start in the same state as its
baseline. We then compare the averages of these estimators with the exact
gradient, measure their variance, and train with them on calm FrozenLake.

```{.python .input #baselines-baselines-advantages-and-variance-reduction-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #baselines-baselines-advantages-and-variance-reduction-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #baselines-baselines-and-variance-reduction-1}
%%tab pytorch, jax
import gymnasium as gym
```

## A Zero-Mean Identity

Every zero-mean term of this section rests on the following fact.

### The Lemma

**Lemma.** *For every state $s$,
$\sum_{a} \pi_\theta(a \mid s)\, \nabla_\theta \log \pi_\theta(a \mid s) = 0$.*

**Proof.**
$\sum_a \pi_\theta(a \mid s)\, \nabla_\theta \log \pi_\theta(a \mid s) = \sum_a \nabla_\theta \pi_\theta(a \mid s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0.$
$\blacksquare$

In words, the score of an action drawn from the policy has mean zero. For the
softmax policy, this can also be read off :eqref:`eq_softmax_score`: averaged
over $a \sim \pi_\theta(\cdot \mid s)$, the indicator $\mathbf{1}(b = a)$ has
mean $\pi_\theta(b \mid s)$, which cancels the second term.

### The Conditional Version

The lemma has a consequence that we use repeatedly. Let $c$ be any quantity
that is already determined when the agent reaches state $s_t$, for example a
reward received earlier in the trajectory, a constant, or a function of $s_t$.
Given the history up to $s_t$, the value of $c$ is fixed while the action
$a_t$ is still to be drawn from $\pi_\theta(\cdot \mid s_t)$, so the lemma
applies to the inner expectation:

$$\mathbb{E}\big[ c\ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \big] = \mathbb{E}\Big[ c\ \underbrace{\mathbb{E}\big[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \mid s_0, a_0, r_0, \ldots, r_{t-1}, s_t \big]}_{=\,0} \Big] = 0.$$

A term of the form $c\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ can
therefore be added to or removed from the REINFORCE estimator without changing
its expected value. It can, however, change the variance. A quantity that
depends on $a_t$, such as the reward $r_t$ that follows it, is not covered,
because it is not fixed when $a_t$ is drawn.

## Reducing the Variance

We apply the conditional identity twice, first to remove terms and then to
subtract a baseline.

### Reward-to-Go

In :eqref:`eq_reinforce`, the score at step $t$ is multiplied by the return of
the whole trajectory, including the rewards $r_{t'}$ with $t' < t$ received
before the action $a_t$ was chosen. Each product of such a reward with the
score at step $t$ has mean zero by the conditional identity, with
$c = \gamma^{t'} r_{t'}$. Removing these terms gives

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T-1} \gamma^{t'} r_{t'} \Big] = \mathbb{E}\Big[ \sum_{t=0}^{T-1} \gamma^t\, G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big],$$
:eqlabel:`eq_rtg`

where $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$ is the *reward-to-go*,
the discounted return of the remainder of the trajectory from step $t$ on; the
second equality factors $\gamma^t$ out of the inner sum. The action at step
$t$ can influence only these rewards. Given the history up to $s_t$ and the
action $a_t$, the factor $\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$
is fixed, and for episodes that end only at terminal states, the expected
value of $G_t$ is $Q^{\pi_\theta}(s_t, a_t)$ (by the Markov property, the
history before $s_t$ does not affect it). Replacing
$G_t$ by $Q^{\pi_\theta}(s_t, a_t)$ in :eqref:`eq_rtg` therefore does not
change the expectation. This yields the policy gradient theorem
:eqref:`eq_pg_theorem`.

On FrozenLake, the only reward arrives at the last step of a successful
episode, so $\gamma^t G_t = R(\tau)$ for every step $t$ (a success at step
$T - 1$ gives $G_t = \gamma^{T-1-t}$ and $\gamma^t G_t = \gamma^{T-1}$), and
:eqref:`eq_rtg` coincides with REINFORCE. Reward-to-go matters when rewards
arrive along the way, as on the CartPole task of :numref:`sec_deeprl`.
Computing it takes one backward pass over each episode,
$G_t = r_t + \gamma G_{t+1}$. We implement the pass for general inputs, since
later sections run it on other sequences:

```{.python .input #baselines-reward-to-go-and-causality}
%%tab pytorch, jax
@d2l.add_to_class(d2l.Batch)  #@save
def backward_scan(self, x, factor):
    """y_t = x_t + factor * y_{t+1}, restarted at every episode boundary."""
    y = np.zeros_like(x)
    for ep in self.episodes():
        running = 0.0
        for t in reversed(range(ep.start, ep.stop)):
            running = x[t] + factor * running
            y[t] = running
    return y

@d2l.add_to_class(d2l.Batch)  #@save
def reward_to_go(self, gamma):
    """G_t: the discounted return of the rest of its episode, by one scan."""
    return self.backward_scan(self.rew, gamma)
```

### Baselines as Control Variates

The second use of the identity is subtraction. Any function $b(s_t)$ of the
state, including a constant, can be subtracted from the reward-to-go without
changing the expected value:

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[ \sum_{t=0}^{T-1} \gamma^t \big( G_t - b(s_t) \big)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big].$$
:eqlabel:`eq_pg_baseline`

The subtracted terms have mean zero by the conditional identity with
$c = \gamma^t b(s_t)$. Such a function $b$ is called a *baseline*. It can change the sign of the weights. On
FrozenLake every return is nonnegative, so no term of the
REINFORCE update pushes the log-probability of its own action down, and the
terms differ only in the size of their push; a failed trajectory contributes
nothing. With a baseline near the typical return, the terms of actions that
turned out better than typical push them up and the terms of actions that
turned out worse push them down, and a failed trajectory now changes the
policy.

Subtracting a baseline is an instance of a general technique for Monte Carlo
estimation, the *control variate*. Suppose we want to estimate $\mathbb{E}[X]$ and each
sample also provides a quantity $Y$ whose mean $\mathbb{E}[Y]$ is known. For any number
$c$, the quantity

$$X_c = X - c\, \big( Y - \mathbb{E}[Y] \big)$$
:eqlabel:`eq_control_variate`

has the same mean as $X$, and its variance is
$\textrm{Var}(X) - 2c\, \textrm{Cov}(X, Y) + c^2\, \textrm{Var}(Y)$. The
variance is smallest for $c^* = \textrm{Cov}(X, Y) / \textrm{Var}(Y)$, where it
equals
$\textrm{Var}(X) - \textrm{Cov}(X, Y)^2 / \textrm{Var}(Y) = (1 - \rho^2)\, \textrm{Var}(X)$,
with $\rho$ the correlation between $X$ and $Y$. For a single step of the
policy gradient from a state $s_t$, take $X = G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ and
$Y = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, whose mean given $s_t$ is
zero by the lemma. Given $s_t$, the baseline $c = b(s_t)$ is a fixed number,
and $\gamma^t X_c$ is the term of step $t$ in :eqref:`eq_pg_baseline`; the
factor $\gamma^t$ multiplies $X$ and $Y$ alike and does not change the best
$c$. A baseline is thus
a control variate built from the score. For a single parameter, as in the
one-step problem of :numref:`fig_rl_score_ascent`, the best baseline is $c^*$,
computed given $s_t$. For a vector of parameters, $X$ and $Y$ are vectors, and
each coordinate has its own best $c$. Minimizing the sum of the variances of
the coordinates gives
$b^* = \mathbb{E}[X^\top Y \mid s_t] / \mathbb{E}[\|Y\|^2 \mid s_t]$, an
average of the reward-to-go weighted by the squared length of the score
:cite:`Greensmith.Bartlett.Baxter.2004` (exercise
:exref:`ex_baselines-best-baseline`). :numref:`fig_rl_variance_reduction`
shows the effect of a baseline on the one-step problem.

![Variance reduction. (a) Along a trajectory, rewards received before an action have zero expected product with its score, which leaves the reward-to-go. (b) On the one-step problem of :numref:`fig_rl_score_ascent`, with reward $R(a)=0.4+2e^{-(a-2)^2/2}$ under the policy $\mathcal{N}(0,1)$, for which $J=0.92$ and $\nabla_\mu J=0.52$, the standard deviation of a single-sample gradient estimate falls from $1.34$ without a baseline to $0.69$ with the mean reward $J$ as the baseline; the histograms are clipped at the 1st and 95th percentiles of the two samples together. (c) The variance of the same estimator as a function of a constant baseline $b$. The optimum $b^\star = c^* = 1.18$ leaves $1 - \rho^2 = 0.23$ of the variance without a baseline.](../img/mdl-rl-variance-reduction.svg)
:label:`fig_rl_variance_reduction`

A two-action example shows the same effects with numbers. At a single state,
let a softmax policy choose an action with return $1$ with probability $0.2$
and an action with return $0$ otherwise. We follow the derivative with
respect to the preference of the rewarded action; by
:eqref:`eq_softmax_score`, its score $Y$ is $0.8$ after the rewarded action
and $-0.2$ after the other. Without a baseline, the single-sample estimate $X$
is $0.8$ with probability $0.2$ and $0$ otherwise. Its mean is the exact
derivative $0.16$, and its standard deviation is $0.32$. With $b = J = 0.2$,
the estimate becomes $0.8 \times 0.8 = 0.64$ or $(-0.2) \times (-0.2) = 0.04$:
the mean is still $0.16$, the standard deviation falls to $0.24$, and the
action with return zero now lowers its own probability. Since
$\textrm{Cov}(X, Y) = 0.2 \times 0.8 \times 0.8 = 0.128$ and
$\textrm{Var}(Y) = 0.2 \times 0.8^2 + 0.8 \times 0.2^2 = 0.16$, the best
baseline is $c^* = 0.8$. The estimate is then $0.16$ after either action, and
its variance is zero: with two actions, the estimate takes only two values,
and $c^*$ makes them equal.

### The Advantage and a Learned Baseline

A natural baseline is the value function itself, $b(s) = V^{\pi_\theta}(s)$.
Since the expected reward-to-go is $Q^{\pi_\theta}(s_t, a_t)$, the weight
$G_t - V^{\pi_\theta}(s_t)$ is then a sample estimate of the advantage
$A^{\pi_\theta}(s_t, a_t)$ of :eqref:`eq_advantage`: it is positive when the
sampled continuation did better than the policy does on average from $s_t$.
The value function is unknown, but it can be estimated from the same
trajectories. A table $\hat{V}$ is moved toward the observed reward-to-go at
every visited state,

$$\hat{V}(s_t) \leftarrow \hat{V}(s_t) + \alpha_V \big( G_t - \hat{V}(s_t) \big),$$
:eqlabel:`eq_value_baseline`

with a step size $\alpha_V$. This is a stochastic gradient step on the squared
error $\frac{1}{2}\big(G_t - \hat{V}(s_t)\big)^2$, a regression of the
observed reward-to-go on the state. This algorithm is REINFORCE with a
baseline :cite:`Williams.1992,Sutton.Barto.2018`. Our implementation weights
each batch with the table from before the batch, so that the baseline does not
depend on the actions of the batch and the conditional identity applies. The
table $\hat{V}$ is trained on Monte Carlo returns, and
:numref:`sec_actorcritic` will instead train it on targets that use its own
predictions.

### Group Baselines and Reward Normalization

When several trajectories start in the same state, their returns are
independent samples whose expected value is the expected return from that
state, and their mean can be subtracted from each of them like a baseline.
GRPO :cite:`Shao.Wang.Zhu.ea.2024` uses this mean as its baseline instead of
a learned value function: it samples a group of $K$ responses to the same
prompt, the start state of a language-model episode
(:numref:`sec_rl_sequences`), and subtracts the average reward of the group
from each reward. GRPO also divides by the standard deviation of the rewards
in the group. With $R_1, \ldots, R_K$ the returns of a group, $\bar{R}$ their
mean, and $s_R$ their standard deviation (with the divisor $K$), every step
of trajectory $i$ receives the weight

$$\hat{A}_i = \frac{R_i - \bar{R}}{s_R + 10^{-8}},$$
:eqlabel:`eq_group_normalized`

where we add a small constant to avoid a division by zero when all $K$
returns are equal. The two operations do different things. The mean
$\bar{R}$ acts like a baseline, with one caveat: $\bar{R}$ contains the return
of the trajectory itself. We can write
$R_i - \bar{R} = \frac{K-1}{K} R_i - \frac{1}{K} \sum_{j \neq i} R_j$. The
returns $R_j$ with $j \neq i$ are independent of trajectory $i$, whose
scores have mean zero by the lemma, so by independence the second term
contributes nothing in expectation. Subtracting the group mean therefore
shrinks the expected estimate by the factor $(K - 1)/K$.

Dividing by $s_R$ is not a baseline. For the returns of one sampled group, it
multiplies all weights by the same positive number, which changes the length
of the group's contribution but not its direction. Across groups, it changes
their relative weights: a group whose returns are nearly equal, such as the
responses to a prompt that is almost always or almost never answered
correctly, is multiplied by a large factor. Since the factor depends on the
sampled returns, it can also change the direction of the expected estimate,
even when all trajectories start in the same state. As an example, let a
softmax policy at a single state choose uniformly among three actions with
returns $0$, $0.2$, and $1$, so $J = 0.4$, and let $K = 2$. By
:eqref:`eq_softmax_score`, the derivative of $J$ with respect to the
preference of an action is its probability $1/3$ times the difference of its
return and $J$, and the gradient is proportional to $(-2, -1, 3)$. A group
whose two returns differ receives the weights $1$ and $-1$ up to the
constant $10^{-8}$, whatever the size of the difference, and a group with
equal returns receives zero weights. Up to terms of order $10^{-8}$, the
expected estimate is therefore proportional to $(-1, 0, 1)$: the action with
return $0.2$ is pushed up in groups with the action of return $0$ and down,
equally often, in groups with the action of return $1$, although its
advantage $0.2 - 0.4$ is negative.

When every return takes one of two values, such as $1$ for a correct response
and $0$ for a wrong one, the expected estimate of a group points along the
gradient of the expected return from its start state. Given trajectory $i$,
its weight $\hat{A}_i$ still depends on the returns of the other
trajectories, which are independent of trajectory $i$, so its conditional
expectation is a function $h(R_i)$ of its return alone. On two
values, every such function has the form $h(R_i) = h_0 + h_1 R_i$, and
$h_1 > 0$ when both values have positive probability, since the weight is
positive for the larger return and negative for the smaller one unless all
returns of the group are equal. The scores of trajectory $i$ are fixed given
the trajectory, so its term has the same expected value as with the weight
$h(R_i)$, which is $h_1 \nabla_\theta J(\theta)$ by the lemma and
:eqref:`eq_pg_gradient`, with $J(\theta)$ the expected return from the start
state of the group.

The group baseline has a limitation. When all $K$ returns of a group are
equal, for example when all responses to a prompt are correct or all are
wrong, every difference $R_i - \bar{R}$ is zero, and so is every weight
$\hat{A}_i$: the group contributes nothing to the update, although all its
trajectories were sampled. On FrozenLake, a batch in which no episode
succeeds gives no update, with or without the group baseline, since all its
returns are zero.

On FrozenLake, every episode starts in state $0$, so the $n$ episodes of a
batch form a single group with $K = n$, and the batch mean of the returns is
a group baseline. We save the normalization of :eqref:`eq_group_normalized`,
which later sections also apply to advantage estimates:

```{.python .input #baselines-centering-is-a-baseline-dividing-by-sigma-is-a-step-size-rescaling}
%%tab pytorch, jax
def normalize(x, eps=1e-8):  #@save
    """Center a batch of weights and rescale them to unit spread."""
    return (x - x.mean()) / (x.std() + eps)
```

## Measuring the Effect

We now measure these estimators on calm FrozenLake, first at a fixed policy,
where the exact gradient is known, and then during training.

### Variance at a Fixed Policy

For a tabular softmax policy, the policy gradient theorem gives the exact
gradient in closed form: substituting :eqref:`eq_softmax_score` into
:eqref:`eq_pg_theorem` yields
$\partial J / \partial \theta_{s,a} = d(s)\, \pi_\theta(a \mid s)\, A^{\pi_\theta}(s, a)$,
where $d(s) = \sum_{t} \gamma^t \Pr(s_t = s)$ is the discounted number of
visits to $s$ from the start state. Only the visits to $s$ contribute to this
derivative, with total weight $d(s)$, and at each visit the average over the
action $a'$ taken there is

$$\sum_{a'} \pi_\theta(a' \mid s)\, Q^{\pi_\theta}(s, a') \big(\mathbf{1}(a' = a) - \pi_\theta(a \mid s)\big) = \pi_\theta(a \mid s) \big( Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s) \big),$$

since $\sum_{a'} \pi_\theta(a' \mid s)\, Q^{\pi_\theta}(s, a') = V^{\pi_\theta}(s)$.
All three factors can be computed exactly from the model by solving linear
equations, as in :numref:`sec_valueiter`. We
lift the time limit of the environment for this measurement, so that the
sampled episodes and the exact gradient refer to the same objective:

```{.python .input #baselines-estimator-hygiene}
%%tab pytorch, jax
gamma = 0.95
env = gym.make('FrozenLake-v1', is_slippery=False, max_episode_steps=10_000)
mdp = d2l.TabularMDP.from_gym(env, gamma)    # only for the exact gradient

def exact_gradient(pi):
    """dJ/dtheta[s, a] = d(s) pi(a|s) A(s, a), and the values V(s)."""
    P_pi = np.einsum('sa,sat->st', pi, mdp.P)
    M = np.linalg.inv(np.eye(mdp.num_states) - gamma * P_pi)
    V = M @ (pi * mdp.r).sum(axis=1)
    A = mdp.backup(V) - V[:, None]
    return M[0][:, None] * pi * A, V           # M[0]: discounted visits d(s)

def reinforce_weights(batch):
    """The weight R(tau_i) for every step of episode i."""
    return np.repeat(batch.episode_returns(gamma),
                     np.diff(batch.ep_ends, prepend=0))

def group_weights(batch):
    """The weight of eq_group_normalized for every step of episode i."""
    return np.repeat(normalize(batch.episode_returns(gamma)),
                     np.diff(batch.ep_ends, prepend=0))

def step_index(batch):
    """The time step t of every entry, counted within its episode."""
    return np.concatenate([np.arange(n)
                           for n in np.diff(batch.ep_ends, prepend=0)])
```

We train REINFORCE as in :numref:`sec_policygradient` and keep the policy
after $20$ and after $50$ updates. For each of the two policies, we draw $300$
batches of $16$ episodes and compute four estimators from each batch:
REINFORCE; the group baseline, which subtracts the batch mean of the returns;
the group-normalized weights :eqref:`eq_group_normalized`, as in GRPO; and the
exact value function as a baseline, $\gamma^t \big(G_t - V^{\pi_\theta}(s_t)\big)$.
For each estimator, we report the cosine between its average over the batches
and the exact gradient, the ratio of their lengths, and its *relative
variance*, the mean squared distance of the estimates from their average
divided by the squared length of the average. The relative variance does not
change when an estimator is multiplied by a constant.

```{.python .input #baselines-five-estimators-1}
%%tab pytorch, jax
rng, policy, probes = np.random.default_rng(0), d2l.SoftmaxTable(16, 4), {}
env.reset(seed=0)
for k in range(1, 51):
    batch = d2l.rollout(env, policy.act, 16, rng)
    policy.step(batch, reinforce_weights(batch), lr=2.0)
    if k in (20, 50):
        probes[k] = policy.theta.copy()

weightings = {
    'REINFORCE': lambda b, V: reinforce_weights(b),
    'group mean': lambda b, V: (reinforce_weights(b)
                                - b.episode_returns(gamma).mean()),
    'group-normalized': lambda b, V: group_weights(b),
    'baseline V': lambda b, V: (gamma ** step_index(b)
                                * (b.reward_to_go(gamma) - V[b.obs]))}
for k, theta in probes.items():
    policy.theta = theta
    g_exact, V = exact_gradient(policy.probs(np.arange(16)))
    draws, rng = {name: [] for name in weightings}, np.random.default_rng(1)
    for _ in range(300):
        batch = d2l.rollout(env, policy.act, 16, rng)
        for name, weights in weightings.items():
            draws[name].append(policy.gradient(batch, weights(batch, V)).ravel())
    print(f'policy after {k} updates: J = {V[0]:.3f}')
    for name, u in draws.items():
        u = np.stack(u)
        m = u.mean(axis=0)
        cos = m @ g_exact.ravel() / (np.linalg.norm(m) * np.linalg.norm(g_exact))
        ratio = np.linalg.norm(m) / np.linalg.norm(g_exact)
        rel = ((u - m) ** 2).sum(axis=1).mean() / (m ** 2).sum()
        print(f'  {name:>16}: cosine {cos:.3f}, length ratio {ratio:.2f}, '
              f'relative variance {rel:.1f}')
```

All four averages point nearly along the exact gradient, with cosines above
$0.98$. REINFORCE and the value
baseline also have the right length, as unbiased estimators should, up to
the sampling error of $300$ batches. The group baseline makes the average
about $15/16$ as long as that of REINFORCE, the factor $(K - 1)/K$ derived
above. The group-normalized estimator is several times longer: every return
lies between $0$ and $1$, so $s_R \leq 1/2$, and dividing by $s_R$ multiplies
the weights $R_i - \bar{R}$ by at least two. REINFORCE has a relative variance
between $6$ and $8$ at both policies: the root-mean-square distance of a
single batch estimate from the average is more than twice the length of the
average. The effect of the baselines depends on the policy. After $20$
updates, the policy rarely reaches the goal, almost all returns are zero, and
neither the group mean nor the value function reduces the relative variance
much. After $50$ updates, the policy reaches the goal in most episodes: a
successful episode takes at least $6$ steps, so its return is at most
$\gamma^5 \approx 0.77$, and $J = 0.55$ requires a success rate above $0.7$.
Both baselines now reduce the relative variance more than threefold; the
group mean does nearly as well as the exact value function. Dividing by $s_R$ does not reduce the
relative variance further. After $20$ updates it raises it slightly, because
the factor $1/s_R$ differs from batch to batch: $s_R$ is small when only a
few episodes of the batch succeed, so these batches are scaled up the most.

### Training

We now train with five configurations on $20$ seeds each, with batches of
$16$ episodes. Four use the step size $\alpha = 2$ of
:numref:`sec_policygradient`: REINFORCE, REINFORCE with the learned baseline
:eqref:`eq_value_baseline`, the group baseline, and the group-normalized
weights :eqref:`eq_group_normalized`. As a control, the fifth is REINFORCE
with the four times larger step size $\alpha = 8$. Besides
the fraction of successful episodes in each batch, we record the length of
every parameter step. The helper `run_seeds` of :numref:`sec_utils` repeats
the training for every seed:

```{.python .input #baselines-five-estimators-3}
%%tab pytorch, jax
def train(seed, weighting, lr, num_updates=150, alpha_v=0.1):
    rng = np.random.default_rng(seed)
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env.reset(seed=seed)
    policy, V_hat = d2l.SoftmaxTable(16, 4), np.zeros(16)
    for _ in range(num_updates):
        batch = d2l.rollout(env, policy.act, 16, rng)
        g = policy.gradient(batch, weighting(batch, V_hat))
        policy.theta += lr * g
        for s, G in zip(batch.obs, batch.reward_to_go(gamma)):
            V_hat[s] += alpha_v * (G - V_hat[s])      # eq_value_baseline
        yield (batch.episode_returns() > 0).mean(), lr * np.linalg.norm(g)

configs = {'REINFORCE, lr 2': (weightings['REINFORCE'], 2.0),
           'learned baseline, lr 2': (weightings['baseline V'], 2.0),
           'group mean, lr 2': (weightings['group mean'], 2.0),
           'group-normalized, lr 2': (weightings['group-normalized'], 2.0),
           'REINFORCE, lr 8': (weightings['REINFORCE'], 8.0)}
runs = {name: d2l.run_seeds(train, 20, weighting=w, lr=lr)
        for name, (w, lr) in configs.items()}
d2l.plot_curves({name: r[:, :, 0] for name, r in runs.items()},
                xlabel='update', ylabel='success rate of the batch', smooth=10)
```

The curves show the median over the seeds of the success rate, averaged over
the last ten updates, and the bands span the smallest and largest values.
We also find, for each run, the number of updates after which the success
rate over the last ten batches first reaches $0.9$, and we average the step
length over the first $40$ updates:

```{.python .input #baselines-five-estimators-6}
%%tab pytorch, jax
def updates_to_90(curve):
    """Last update of the first 10-update window with mean success >= 0.9."""
    hit = np.convolve(curve, np.ones(10), 'valid') >= 9   # sum: no rounding
    return hit.argmax() + 9 if hit.any() else len(curve)

for name, r in runs.items():
    reach = np.array([updates_to_90(c) for c in r[:, :, 0]])
    print(f'{name:>22}: updates to 0.9: median {np.median(reach):5.1f}, '
          f'range {reach.min()}-{reach.max()}; mean step {r[:, :40, 1].mean():.2f}')
```

With $\alpha = 2$, neither baseline speeds up training: the medians of
REINFORCE, of the learned baseline, and of the group baseline lie between
$64$ and $76$ updates, and their ranges over the seeds overlap. This
agrees with the measurement at a fixed policy: while successes are rare,
neither the group mean nor even the exact value function reduces the relative
variance much, and the learned table starts at zero. The group-normalized
weights of GRPO learn about three times faster than REINFORCE, but their
steps are also about five times longer. REINFORCE with the four times larger step size learns
about as fast as the group-normalized variant, with steps that are longer
still. On this problem, a larger step size alone thus gives about the same
speedup, without a lower relative variance: the division by $s_R$ has the effect of a
larger step size, and the measurement at a fixed policy shows that this factor is
larger early in training, when few episodes succeed. With batches of $16$
episodes, the step size $\alpha = 8$ did no harm in these runs;
:numref:`sec_ppo` shows a case in which large changes of the policy do.

## Summary

The score of an action drawn from the policy has mean zero, also conditionally
on the history. This identity allows us to drop the rewards received before an
action, which gives the reward-to-go and the policy gradient theorem, and to
subtract any function of the state, a baseline, without changing the expected
gradient. A baseline is a control variate, and with the value function as the
baseline, the weights become advantage estimates. The mean return of a group
of trajectories from the same start state acts like a baseline, and GRPO uses
it as one; since it contains each trajectory's own return, it shrinks the
expected gradient by the factor $(K - 1)/K$. Dividing by the standard
deviation of the group, as GRPO also does, is not a baseline: it changes the
length of each group's contribution and the relative weights of the groups,
and when the returns take more than two values, it can also change the
direction of the expected estimate of a single group.
On calm FrozenLake, the value baseline and the group mean reduced the relative
variance more than threefold once the policy succeeded often, but by much less
early in training, and REINFORCE with a four times larger step size trained
about as fast as with the group-normalized weights. These measurements depend on the problem. The next
section moves to a task with continuous states, on which the value function is
represented by a neural network.

## Exercises

1. **The best constant baseline.** For a single state and a constant baseline
   $b$, show that the sum of the variances of the coordinates of the estimator
   $(G - b)\, \nabla_\theta \log \pi_\theta(a \mid s)$ is minimized by
   $b^* = \mathbb{E}\big[\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2 G\big] / \mathbb{E}\big[\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2\big]$.
   Compare $b^*$ with the coefficient $c^*$ of :eqref:`eq_control_variate`.
   :exlabel:`ex_baselines-best-baseline`
1. [code] **The baseline step size.** Repeat the training comparison with
   $\alpha_V \in \{0.01, 0.1, 0.5, 1\}$ for the learned baseline. What does
   the table $\hat{V}$ estimate when $\alpha_V = 1$?
1. [code] **Smaller batches.** Repeat the variance measurement and the
   training comparison with batches of $4$ episodes. Which conclusions of the
   section change?
1. [code] **Slippery ice.** Repeat the variance measurement on slippery
   FrozenLake. How do the two probe policies and the effect of the value
   baseline change?

[Discussions](https://d2l.discourse.group/t/36799)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14.5]{.kicker}

Baselines and variance reduction<br>
**a zero-mean identity · reward-to-go · baselines as control variates · the advantage · group baselines · measured on FrozenLake**
:::
:::

::: {.slide title="One Identity"}
$$\sum_a \pi_\theta(a \mid s)\, \nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) = 0$$

Anything determined before $a_t$ is drawn, times the score at $t$, has mean
zero.

. . .

- drop the rewards before step $t$: the reward-to-go $G_t$;
- subtract any $b(s_t)$: a baseline.
:::

::: {.slide title="Reward-to-Go and the Baseline"}
$$\nabla_\theta J(\theta) = \mathbb{E}\Big[ \sum_t \gamma^t \big( G_t - b(s_t) \big)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

- $\mathbb{E}[G_t \mid s_t, a_t] = Q^{\pi_\theta}(s_t, a_t)$ (no time limit):
  the policy gradient theorem.
- $b = V^{\pi_\theta}$: the weights estimate the advantage.
:::

::: {.slide title="A Baseline Is a Control Variate"}
![](../img/mdl-rl-variance-reduction.svg){width=92%}

Same mean for every $c$; the variance falls to $(1 - \rho^2)\,\textrm{Var}(X)$
at the best $c$.
:::

::: {.slide title="Group Baselines (GRPO)"}
$$\hat{A}_i = \frac{R_i - \bar{R}}{s_R + 10^{-8}}$$

- $\bar{R}$: the mean return of $K$ trajectories from the same start state,
  used as a baseline; it shrinks the expected gradient by $(K - 1)/K$.
- Dividing by $s_R$ is not a baseline: it rescales each group by a factor
  that depends on its returns, which can change the direction of a single
  group's expected estimate when the returns take more than two values.
- If all returns of a group are equal (all correct or all wrong), every weight
  is zero, and the group gives no update.
:::

::: {.slide title="Measured at a Fixed Policy"}
@baselines-five-estimators-1
:::

::: {.slide title="Measured in Training"}
@baselines-five-estimators-6

. . .

On this problem, a four times larger step size gives about the same speedup
as dividing by $s_R$.
:::
