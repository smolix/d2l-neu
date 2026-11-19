# Soft Actor-Critic
:label:`sec_sac`

DQN takes a maximum over actions in every target. With two actions, the
maximum is a comparison; with a continuous action, such as the torque that a
motor applies to a joint, it is an optimization problem at every target.
Off-policy actor-critic methods replace the maximum by an *actor*, a network
trained to choose actions that the critic values highly, and keep the replay
buffer and the target networks of :numref:`sec_dqn`. DDPG trains a
deterministic actor with the deterministic policy gradient
:cite:`Silver.Lever.Heess.ea.2014,Lillicrap.Hunt.Pritzel.ea.2016`, and TD3
adds a second critic against overestimation :cite:`Fujimoto.vanHoof.Meger.2018`.
Soft actor-critic (SAC) uses a stochastic actor and adds the entropy of the
policy to the reward :cite:`Haarnoja.Zhou.Abbeel.ea.2018,Haarnoja.Zhou.Hartikainen.ea.2018`.
We derive the backup and the policy improvement step of this objective, build
SAC for the Pendulum task, and measure what the second critic does to the
value estimates.

```{.python .input #sac-soft-actor-critic}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #sac-soft-actor-critic}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #sac-soft-actor-critic-1}
%%tab pytorch, jax
import gymnasium as gym
```

## Maximum-Entropy Reinforcement Learning

We first state the objective that SAC maximizes and derive its backup and its
improvement step.

### Entropy in the Return

SAC maximizes the expected return with the entropy of the policy added to the
reward of every step,

$$J_\beta(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_{t} \gamma^t \big(r_t + \beta\, H(\pi(\cdot \mid s_t))\big)\Big], \quad H(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi(\cdot \mid s)}\big[\log \pi(a \mid s)\big],$$
:eqlabel:`eq_sac_objective`

where the *temperature* $\beta > 0$ weighs the entropy against the reward
:cite:`Haarnoja.Tang.Abbeel.ea.2017,Haarnoja.Zhou.Abbeel.ea.2018`. The
entropy measures how random the policy is at a state: over two actions, it is
$\log 2 \approx 0.69$ for the uniform policy and zero for a deterministic one,
so there the bonus adds at most $\beta \log 2$ to the reward of a step. For
$\beta \to 0$, the objective becomes the expected return of the preceding
sections. The entropy bonus of PPO (:numref:`sec_ppo`) rewards the entropy of
the policy at the states of the current batch. Here the entropy is part of the
return, so the agent also values reaching states in which it can keep acting
at random :cite:`Haarnoja.Tang.Abbeel.ea.2017`.

### The Soft Bellman Backup

The value functions of :eqref:`eq_sac_objective` satisfy

$$Q^\pi(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s'}\big[V^\pi(s')\big], \qquad V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\big[Q^\pi(s, a) - \beta \log \pi(a \mid s)\big].$$
:eqlabel:`eq_sac_backup`

The only change from the Bellman expectation equations of
:numref:`sec_valueiter` is the term $-\beta \log \pi(a \mid s)$ inside $V^\pi$,
whose expectation is the entropy bonus of state $s$, so that
$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q^\pi(s, a)] + \beta H(\pi(\cdot \mid s))$.
The entropy of the first state is not part of $Q^\pi(s, a)$, because the
first action is given. If we fold the discounted expected entropy bonus of the
next state, $\gamma \beta\, \mathbb{E}_{s'}[H(\pi(\cdot \mid s'))]$, into the
reward, the backup becomes ordinary policy evaluation for a modified reward,
which is a contraction for $\gamma < 1$. For finite action sets,
:citet:`Haarnoja.Zhou.Abbeel.ea.2018` prove on this basis that repeated soft
backups converge to the soft values of $\pi$.

### Soft Policy Improvement

Policy iteration (:numref:`sec_valueiter`) improves a policy by acting
greedily with respect to its action values. With the entropy in the
objective, the improved policy is no longer greedy. At a single state, the
expected action value is largest when all probability sits on the best
actions, and the entropy when the probability is spread uniformly.

**Proposition.** *Let $\beta > 0$, let the set of actions be finite, or a
bounded set of positive volume, and let $Q(s, \cdot)$ be bounded. Among all
distributions $\pi(\cdot \mid s)$ over the actions (densities, for continuous
actions),
$\mathbb{E}_{a \sim \pi}[Q(s, a)] + \beta H(\pi(\cdot \mid s))$ is maximized by*

$$\pi^*(a \mid s) = \frac{\exp\big(Q(s, a)/\beta\big)}{Z(s)}, \quad \textrm{where } Z(s) = \int \exp\big(Q(s, a)/\beta\big)\, da,$$
:eqlabel:`eq_sac_boltzmann`

*and the integral is a sum for finitely many actions.*

**Proof.** By the definition of $H$, the quantity to be maximized equals
$\mathbb{E}_{a \sim \pi}[Q(s, a) - \beta \log \pi(a \mid s)]$. By the
definition of $\pi^*$, $Q(s, a) = \beta \log \pi^*(a \mid s) + \beta \log Z(s)$;
substituting this gives

$$\mathbb{E}_{a \sim \pi}\big[Q(s, a) - \beta \log \pi(a \mid s)\big] = -\beta\, \mathbb{E}_{a \sim \pi}\Big[\log \frac{\pi(a \mid s)}{\pi^*(a \mid s)}\Big] + \beta \log Z(s) = -\beta\, D_{\textrm{KL}}\big(\pi(\cdot \mid s) \,\Vert\, \pi^*(\cdot \mid s)\big) + \beta \log Z(s).$$

The divergence is nonnegative and zero only for $\pi = \pi^*$ (for densities,
almost everywhere). $\blacksquare$

This is the softmax policy of :numref:`sec_qlearning` with temperature
$\beta$. As $\beta \to 0$, $\pi^*$ concentrates on the maximizers of
$Q(s, \cdot)$, and the greedy step of :numref:`sec_valueiter` returns. For
example, with two actions valued $Q(s, a_1) = 1$ and $Q(s, a_2) = 0$, $\pi^*$
chooses $a_1$ with probability $e/(1 + e) \approx 0.73$ at $\beta = 1$ and
with probability above $0.9999$ at $\beta = 0.1$. A constant added to
$Q(s, \cdot)$ cancels between the numerator and $Z(s)$, so $\pi^*$ depends
only on the differences of the action values divided by $\beta$.
:numref:`sec_rl_sequences` derives the same form with a reference policy in
place of the uniform distribution. The actor's family of distributions cannot
represent $\pi^*$ in general, so SAC chooses the member of its family that is closest to $\pi^*$ in
the divergence of the proof. By the identity in the proof, this divergence,
multiplied by $\beta$, equals
$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}[\beta \log \pi_\theta(a \mid s) - Q(s, a)] + \beta \log Z(s)$.
Since $Z(s)$ does not depend on the actor's parameters $\theta$, this means
minimizing

$$L(\theta) = \mathbb{E}_{s}\, \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\big[\beta \log \pi_\theta(a \mid s) - Q(s, a)\big]$$
:eqlabel:`eq_sac_actor_loss`

over states $s$ from the replay buffer. In expectation, the first term is
$-\beta H(\pi_\theta(\cdot \mid s))$ and favors random actors; the second
favors actions with large values. For finite action sets,
:citet:`Haarnoja.Zhou.Abbeel.ea.2018` prove that alternating exact soft
policy evaluation with the choice of the closest member, made separately at
every state, converges to the policy of the family with the largest soft
values. The loss :eqref:`eq_sac_actor_loss` is the counterpart of this step
for an actor whose parameters $\theta$ are shared by all states: minimizing it
minimizes the average of the divergences over the states of the replay
buffer. Their proof does not cover this case, because it uses that the new
policy is at least as close to $\pi^*$ as the old one at every state, which a
minimizer of the average need not be.

## The Algorithm

SAC approximates the two steps by stochastic gradient steps on networks. Four
pieces are new compared with :numref:`sec_dqn`: an actor whose actions stay
in the allowed range, two critics instead of one, Polyak averaging of the
target copies, and a temperature that is learned.

### A Squashed Gaussian Actor

The torque of the Pendulum task lies in $[-2, 2]$. The actor outputs a mean
$\mu_\theta(s)$ and a standard deviation $\sigma_\theta(s)$, whose logarithm
is clipped to $[-5, 2]$, draws
$u = \mu_\theta(s) + \sigma_\theta(s)\, \xi$ with
$\xi \sim \mathcal{N}(0, 1)$, and squashes the result into the range,
$a = 2 \tanh(u)$. Given $\xi$, the action is a differentiable function of
$\theta$, so the gradient of :eqref:`eq_sac_actor_loss` can pass through the
action into the critic, $\nabla_\theta Q(s, a) = \nabla_a Q(s, a)\,
\nabla_\theta a$. This is the reparameterization of
:citet:`Kingma.Welling.2014`, an alternative to the score-function estimator
of :numref:`sec_policygradient` that uses the critic's gradient with respect
to the action. The squashing changes the density. The map $u \mapsto 2\tanh u$
compresses the real line into $(-2, 2)$, and it compresses large $|u|$ the
most, so the probability of a long interval of $u$ lands in a short interval
of $a$. By the change of variables :cite:`Haarnoja.Zhou.Abbeel.ea.2018`, the
density of $a$ is the density of $u$ divided by the slope
$da/du = 2(1 - \tanh^2 u)$, which in logarithms reads

$$\log \pi_\theta(a \mid s) = \log \mathcal{N}\big(u; \mu_\theta(s), \sigma_\theta(s)^2\big) - \log\big(2 (1 - \tanh^2 u)\big).$$
:eqlabel:`eq_sac_log_prob`

For large $|u|$, $\tanh u$ rounds to $\pm 1$ in floating point, so
$1 - \tanh^2 u$ evaluates to zero; we therefore compute its logarithm as
$2(\log 2 - u - \textrm{softplus}(-2u))$, where
$\textrm{softplus}(x) = \log(1 + e^x)$. This expression is equal, since
$1 - \tanh^2 u = 4e^{-2u}/(1 + e^{-2u})^2$, and it stays finite. The next
cell checks :eqref:`eq_sac_log_prob` numerically: the density
of $a$ must integrate to one over $(-2, 2)$, and it does only with the
log-determinant term.

```{.python .input #sac-a-squashed-gaussian-actor}
%%tab pytorch, jax
def gauss_log_prob(u, mu, std):
    return -0.5 * ((u - mu) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))

def squashed_log_prob(u, mu, std):
    """log pi(a | s) for a = 2 tanh(u), u ~ N(mu, std^2), eq_sac_log_prob."""
    log_det = np.log(2) + 2 * (np.log(2) - u - np.logaddexp(0, -2 * u))
    return gauss_log_prob(u, mu, std) - log_det

a = np.linspace(-2, 2, 400_001)[1:-1]
u = np.arctanh(a / 2)
for mu, std in [(0.0, 0.5), (0.7, 0.8)]:
    with_det = np.trapezoid(np.exp(squashed_log_prob(u, mu, std)), a)
    without = np.trapezoid(np.exp(gauss_log_prob(u, mu, std)), a)
    print(f'mu = {mu:3.1f}, std = {std}: integral {with_det:.6f} with the '
          f'log-determinant, {without:.3f} without')
```

### Twin Critics and Soft Targets

The actor is trained to choose actions that the critic values highly, so it
also seeks out actions whose values the critic overestimates, and the
maximization bias of :numref:`sec_dqn` returns.
:citet:`Fujimoto.vanHoof.Meger.2018` showed that this overestimation occurs in
actor-critic methods for continuous actions and proposed the *clipped double
Q-learning* target, which uses the smaller of the estimates of two critics
that are initialized and trained separately. The minimum is never larger than
either estimate, so its expected value is at most the smaller of their
expected values. This counteracts an upward bias, and it can also produce a
downward one :cite:`Fujimoto.vanHoof.Meger.2018`. SAC uses this target in its
soft form :cite:`Haarnoja.Zhou.Hartikainen.ea.2018`,

$$y = r + \gamma\, (1 - d')\, \Big(\min_{j = 1, 2} Q_{w_j^-}(s', a') - \beta \log \pi_\theta(a' \mid s')\Big), \quad a' \sim \pi_\theta(\cdot \mid s'),$$
:eqlabel:`eq_sac_target`

where $Q_{w_j}$ are the two critics and $Q_{w_j^-}$ their target copies, as
in :eqref:`eq_dqn_target`. SAC uses the minimum of the two critics also for $Q$
in :eqref:`eq_sac_actor_loss`. Both critics regress to the same target $y$.
The target copies follow their critics by Polyak averaging,
$w_j^- \leftarrow (1 - \nu)\, w_j^- + \nu\, w_j$
with $\nu = 0.005$ after every update, instead of DQN's periodic copy
:cite:`Lillicrap.Hunt.Pritzel.ea.2016`. Unrolled, this update makes each copy
an exponentially weighted average of its critic's past parameters, in which
the parameters from the last $200$ updates have total weight
$1 - 0.995^{200} \approx 0.63$, so the copies change a little after every
update instead of jumping. The action $a'$ is drawn fresh from the current
actor, so the target estimates $V^\pi(s')$ of :eqref:`eq_sac_backup` by one
sample.

### Learning the Temperature

The best temperature depends on the task and on the scale of its rewards, and
it can change during training. :citet:`Haarnoja.Zhou.Hartikainen.ea.2018`
therefore learn $\beta$ so that the entropy of the policy stays near a target
$\bar{H}$, by minimizing

$$L(\beta) = \mathbb{E}_{s}\, \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\big[-\beta\, (\log \pi_\theta(a \mid s) + \bar{H})\big]$$
:eqlabel:`eq_sac_temperature`

with the target $\bar{H} = -1$, minus the dimension of the action. A negative
target is possible because the entropy of a density, unlike that of a
distribution over finitely many actions, can be negative
(:numref:`subsec_mdl-differential-entropy`): a Gaussian with standard
deviation $\sigma$ has entropy $\tfrac{1}{2}\log(2\pi e \sigma^2)$, which
equals $-1$ for $\sigma \approx 0.09$. The derivative of
:eqref:`eq_sac_temperature` with respect to $\beta$ is
$-\mathbb{E}[\log \pi_\theta(a \mid s)] - \bar{H}$, the entropy of the policy,
averaged over the states, minus its target. When the entropy falls below
$\bar{H}$, this derivative is negative, so a gradient step raises $\beta$ and
the actor is pushed back toward more random actions.
In the code, we replace $\beta$ by $\log \beta$ in
:eqref:`eq_sac_temperature`. The gradient with respect to $\log \beta$ is then
the derivative of :eqref:`eq_sac_temperature` with respect to $\beta$, so
every step moves $\beta$ in the same direction, and $\beta = \exp(\log
\beta)$ stays positive.

### The Update

We use the hyperparameters of :citet:`Haarnoja.Zhou.Hartikainen.ea.2018`: two
hidden layers of $256$ units for every network, Adam with learning rate
$3 \times 10^{-4}$, minibatches of $256$ transitions, $\gamma = 0.99$, and
$\nu = 0.005$. Training takes $20{,}000$ environment steps. The first $1{,}000$
steps use uniformly random torques to fill the replay buffer, and each later
step is followed by one update.

```{.python .input #sac-the-update-1}
%%tab pytorch, jax
gamma, nu, lr, batch_size, hidden = 0.99, 0.005, 3e-4, 256, 256
num_env_steps, warmup, num_seeds = 20_000, 1_000, 3
max_action, target_entropy = 2.0, -1.0
```

The actor receives the noise $\xi$ as an input, drawn with NumPy, so that
both tabs consume random numbers in the same way. Each critic maps a state and
an action to one value, and `Agent` keeps the actor, the critics, their target
copies, $\log \beta$, and one optimizer for each of the three.

```{.python .input #sac-the-update-2}
%%tab pytorch
def mlp(num_inputs, num_outputs):
    return nn.Sequential(nn.Linear(num_inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, num_outputs))

class Actor(nn.Module):
    """A squashed Gaussian policy: a = max_action tanh(mu(s) + std(s) xi)."""
    def __init__(self):
        super().__init__()
        self.net = mlp(3, 2)

    def forward(self, s, xi):
        mu, log_std = self.net(s).chunk(2, -1)
        std = log_std.clamp(-5, 2).exp()
        u = mu + std * xi
        log_det = math.log(max_action) + 2 * (
            math.log(2) - u - F.softplus(-2 * u))   # log(da/du)
        log_prob = (-0.5 * xi ** 2 - std.log() - 0.5 * math.log(2 * math.pi)
                    - log_det).sum(-1)
        return max_action * torch.tanh(u), log_prob

class Agent:
    """The actor, num_critics critics with target copies, and log beta."""
    def __init__(self, num_critics, seed):
        torch.manual_seed(seed)
        self.actor = Actor()
        self.critics = nn.ModuleList([mlp(4, 1) for _ in range(num_critics)])
        self.targets = nn.ModuleList([mlp(4, 1) for _ in range(num_critics)])
        self.targets.load_state_dict(self.critics.state_dict())
        self.log_beta = torch.zeros((), requires_grad=True)
        self.opts = [torch.optim.Adam(params, lr=lr) for params in (
            self.critics.parameters(), self.actor.parameters(), [self.log_beta])]

    def q(self, nets, s, a):
        """The smallest of the critics' estimates."""
        x = torch.cat([s, a], -1)
        return torch.stack([net(x).squeeze(-1) for net in nets]).min(0).values

    def act(self, obs, xi):
        """A sampled action and its log-probability, as numpy."""
        with torch.no_grad():
            a, log_prob = self.actor(torch.as_tensor(obs), torch.as_tensor(
                xi, dtype=torch.float32))
        return a.numpy(), float(log_prob)

    def estimate(self, obs, a):
        with torch.no_grad():
            return float(self.q(self.critics, torch.as_tensor(obs)[None],
                                torch.as_tensor(a)[None]))

    def beta(self):
        return float(self.log_beta.detach().exp())
```

```{.python .input #sac-the-update-2}
%%tab jax
def mlp(num_inputs, num_outputs, rngs):
    return nnx.Sequential(nnx.Linear(num_inputs, hidden, rngs=rngs), jax.nn.relu,
                          nnx.Linear(hidden, hidden, rngs=rngs), jax.nn.relu,
                          nnx.Linear(hidden, num_outputs, rngs=rngs))

class Actor(nnx.Module):
    """A squashed Gaussian policy: a = max_action tanh(mu(s) + std(s) xi)."""
    def __init__(self, rngs):
        self.net = mlp(3, 2, rngs)

    def __call__(self, s, xi):
        mu, log_std = jnp.split(self.net(s), 2, -1)
        std = jnp.exp(jnp.clip(log_std, -5, 2))
        u = mu + std * xi
        log_det = math.log(max_action) + 2 * (
            math.log(2) - u - jax.nn.softplus(-2 * u))   # log(da/du)
        log_prob = (-0.5 * xi ** 2 - jnp.log(std) - 0.5 * math.log(2 * math.pi)
                    - log_det).sum(-1)
        return max_action * jnp.tanh(u), log_prob

class Critics(nnx.Module):
    """num_critics critics; returns their estimates stacked along axis 0."""
    def __init__(self, num_critics, rngs):
        self.nets = nnx.List([mlp(4, 1, rngs) for _ in range(num_critics)])

    def __call__(self, s, a):
        x = jnp.concatenate([s, a], -1)
        return jnp.stack([net(x).squeeze(-1) for net in self.nets])

class LogBeta(nnx.Module):
    def __init__(self):
        self.value = nnx.Param(jnp.zeros(()))

class Agent:
    """The actor, num_critics critics with target copies, and log beta."""
    def __init__(self, num_critics, seed):
        rngs = nnx.Rngs(seed)
        self.actor, self.critics = Actor(rngs), Critics(num_critics, rngs)
        self.targets, self.log_beta = nnx.clone(self.critics), LogBeta()
        self.opts = [nnx.Optimizer(module, optax.adam(lr), wrt=nnx.Param)
                     for module in (self.critics, self.actor, self.log_beta)]
        self._act = nnx.cached_partial(_act, self.actor)
        self._estimate = nnx.cached_partial(_estimate, self.critics)

    def act(self, obs, xi):
        """A sampled action and its log-probability, as numpy."""
        a, log_prob = self._act(jnp.asarray(obs), jnp.asarray(xi, jnp.float32))
        return np.asarray(a), float(log_prob)

    def estimate(self, obs, a):
        return float(self._estimate(jnp.asarray(obs)[None],
                                    jnp.asarray(a)[None])[0])

    def beta(self):
        return float(jnp.exp(self.log_beta.value[...]))

_act = nnx.jit(lambda actor, s, xi: actor(s, xi))
_estimate = nnx.jit(lambda critics, s, a: critics(s, a).min(0))
```

One update computes the target :eqref:`eq_sac_target` with the target copies,
takes one step for the critics, one for the actor with the updated critics,
one for $\log \beta$, and then moves the target copies toward the critics.
Pendulum never terminates, so $d' = 0$ in every stored transition; the mask
still follows `terminated`, as in every target of this chapter.

```{.python .input #sac-the-update-3}
%%tab pytorch
def sac_step(agent, batch, xi2, xi):
    """One SAC update on a minibatch, with noise xi2 at s' and xi at s."""
    s, a, r, s2, d, xi2, xi = (torch.as_tensor(x, dtype=torch.float32) for x in (
        batch.obs, batch.act, batch.rew, batch.next_obs, batch.term, xi2, xi))
    beta = agent.log_beta.exp().detach()
    with torch.no_grad():
        a2, log_prob2 = agent.actor(s2, xi2)
        y = r + gamma * (1 - d) * (agent.q(agent.targets, s2, a2)
                                   - beta * log_prob2)
    x = torch.cat([s, a], -1)
    critic_loss = sum(F.mse_loss(net(x).squeeze(-1), y)
                      for net in agent.critics)
    opt_critics, opt_actor, opt_beta = agent.opts
    opt_critics.zero_grad(); critic_loss.backward(); opt_critics.step()
    a_new, log_prob = agent.actor(s, xi)
    actor_loss = (beta * log_prob - agent.q(agent.critics, s, a_new)).mean()
    opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
    beta_loss = -(agent.log_beta * (log_prob.detach() + target_entropy)).mean()
    opt_beta.zero_grad(); beta_loss.backward(); opt_beta.step()
    with torch.no_grad():
        for w, w_bar in zip(agent.critics.parameters(),
                            agent.targets.parameters()):
            w_bar.lerp_(w, nu)
```

```{.python .input #sac-the-update-3}
%%tab jax
@nnx.jit
def _sac_step(actor, critics, targets, log_beta, opt_critics, opt_actor,
              opt_beta, s, a, r, s2, d, xi2, xi):
    beta = jnp.exp(log_beta.value[...])
    a2, log_prob2 = actor(s2, xi2)
    y = jax.lax.stop_gradient(
        r + gamma * (1 - d) * (targets(s2, a2).min(0) - beta * log_prob2))
    critic_loss = lambda c: ((c(s, a) - y) ** 2).mean(1).sum()
    opt_critics.update(critics, nnx.grad(critic_loss)(critics))
    def actor_loss(actor):
        a_new, log_prob = actor(s, xi)
        return (beta * log_prob - critics(s, a_new).min(0)).mean(), log_prob
    grads, log_prob = nnx.grad(actor_loss, has_aux=True)(actor)
    opt_actor.update(actor, grads)
    beta_loss = lambda lb: -(lb.value[...] * (
        jax.lax.stop_gradient(log_prob) + target_entropy)).mean()
    opt_beta.update(log_beta, nnx.grad(beta_loss)(log_beta))
    nnx.update(targets, optax.incremental_update(
        nnx.state(critics, nnx.Param), nnx.state(targets, nnx.Param), nu))

def sac_step(agent, batch, xi2, xi):
    """One SAC update on a minibatch, with noise xi2 at s' and xi at s."""
    if not hasattr(agent, '_step'):
        agent._step = nnx.cached_partial(
            _sac_step, agent.actor, agent.critics, agent.targets,
            agent.log_beta, *agent.opts)
    agent._step(*(jnp.asarray(x, jnp.float32) for x in (
        batch.obs, batch.act, batch.rew, batch.next_obs, batch.term, xi2, xi)))
```

The training loop reuses the `ReplayBuffer` of :numref:`sec_dqn`, with an array
of torques in place of the action indices, and records the return of every
episode together with the current temperature.

```{.python .input #sac-the-update-4}
%%tab pytorch, jax
def train_sac(agent, seed):
    """SAC on Pendulum; yields (env step, episode return, beta) per episode."""
    rng, env = np.random.default_rng(seed), gym.make('Pendulum-v1')
    buffer = d2l.ReplayBuffer(num_env_steps, 3)
    buffer.act = np.zeros((num_env_steps, 1), np.float32)   # torques
    obs, ep_return = env.reset(seed=seed)[0], 0.0
    for t in range(1, num_env_steps + 1):
        if t <= warmup:
            a = rng.uniform(-max_action, max_action, 1).astype(np.float32)
        else:
            a, _ = agent.act(obs, rng.standard_normal(1))
        next_obs, rew, terminated, truncated, _ = env.step(a)
        buffer.add(obs, a, rew, next_obs, float(terminated))
        obs, ep_return = next_obs, ep_return + rew
        if terminated or truncated:
            yield t, ep_return, agent.beta()
            obs, ep_return = env.reset()[0], 0.0
        if t > warmup:
            sac_step(agent, buffer.sample(batch_size, rng),
                     *rng.standard_normal((2, batch_size, 1)))
```

## SAC on Pendulum

We train SAC on the Pendulum task of Gymnasium
:cite:`Towers.Kwiatkowski.Terry.ea.2024` and then compare two critics with one.

### The Task

A pendulum hangs from a motor. The state is the angle $\vartheta$ from the
upright position and the angular velocity $\dot\vartheta$; the agent observes
$(\cos\vartheta, \sin\vartheta, \dot\vartheta)$ and chooses a torque in
$[-2, 2]$. The reward of every step is
$-(\vartheta^2 + 0.1\, \dot\vartheta^2 + 0.001\, a^2)$ with $\vartheta$ taken in
$[-\pi, \pi]$, so it is largest, zero, when the pendulum stands still upright
without torque. The motor is weak: gravity accelerates a horizontal pendulum
by $15$ radians per second squared, and the largest torque by only $6$, so the
motor can hold the pendulum still only within about $24$ degrees of hanging
down or of upright. From a start near the bottom, the pendulum can therefore
get up only by swinging back and forth. Each
episode starts at a random angle and a random velocity in $[-1, 1]$, never
terminates, and is cut off after $200$ steps by a time limit, so every target
bootstraps (:numref:`sec_mdp`). For reference, the next cell plays $50$
episodes without torque and $50$ with uniformly random torques:

```{.python .input #sac-the-task}
%%tab pytorch, jax
env, rng = gym.make('Pendulum-v1'), np.random.default_rng(0)
for name, policy in [('no torque', lambda: np.zeros(1, np.float32)),
                     ('random torque', lambda: rng.uniform(-2, 2, 1).astype(
                         np.float32))]:
    returns = []
    for k in range(50):
        env.reset(seed=k)
        total, done = 0.0, False
        while not done:
            _, rew, terminated, truncated, _ = env.step(policy())
            total, done = total + rew, terminated or truncated
        returns.append(total)
    print(f'{name:>13}: mean return {np.mean(returns):.0f}')
```

### Learning to Swing Up

We train three seeds with two critics and, for the comparison below, three
with one critic; everything else is identical.

```{.python .input #sac-learning-to-swing-up-1}
%%tab pytorch, jax
arms = {'two critics': 2, 'one critic': 1}
agents = {arm: [Agent(n, seed) for seed in range(num_seeds)]
          for arm, n in arms.items()}
runs = {arm: np.array([list(train_sac(agents[arm][seed], seed))
                       for seed in range(num_seeds)]) for arm in arms}
```

Every episode lasts $200$ steps, so all runs end their episodes at the same
steps. The reward of a step is at most $-\vartheta^2$, so in episodes with a
mean return of at least $-200$, the squared angle from upright averages at
most $1$ over their steps, whereas at the bottom, where $\vartheta = \pm\pi$,
the reward is at most $-\pi^2 \approx -9.9$ per step. The returns of single
episodes depend strongly on their random start states, so the cell plots the
mean return of the last five episodes and prints, for every run, the first
step at which this mean reaches $-200$, together with the temperature at the
end of training:

```{.python .input #sac-learning-to-swing-up-2}
%%tab pytorch, jax
def steps_to(run, level=-200, k=5):
    """The step at which the mean return of the last k episodes reaches level."""
    hit = np.flatnonzero(np.convolve(run[:, 1], np.ones(k) / k, 'valid') >= level)
    return int(run[k - 1 + hit[0], 0]) if len(hit) else None

d2l.plot_curves({arm: r[:, :, 1] for arm, r in runs.items()},
                xlabel='thousand environment steps',
                ylabel='mean return of the last 5 episodes', smooth=5,
                x=runs['two critics'][0, :, 0] / 1000, reference=-200)
for arm, r in runs.items():
    print(f'{arm:>11}: steps to -200 per seed {[steps_to(x) for x in r]}, '
          f'final beta {np.round(r[:, -1, 2], 3)}')
```

In every run, the mean return of five consecutive episodes first reaches
$-200$ after about $5{,}000$ environment steps, with one critic as with two.
After that, it stays near $-200$ or above, except for temporary drops in some
runs. The temperature falls from its initial value $1$ to between about
$0.01$ and $0.06$ at the end of training.

### What the Second Critic Changes

The two arms learn at about the same speed, so the difference lies elsewhere.
We compare what the critics predict with what the policy obtains. The critics
estimate $Q^\pi(s_0, a_0)$ of :eqref:`eq_sac_backup`: the reward of the first
step, then rewards and entropy bonuses of all later steps, discounted. We
estimate the same quantity by playing the trained stochastic policy from ten
start states, three times each, after the first action $a_0$ at which the
critics were queried. To make the cut at the end negligible, these episodes
run for $1{,}000$ steps, after which $\gamma^{1000} < 10^{-4}$; the temperature
is held at its final value. The three episodes from one start state share
that state, so we treat the start states, not the episodes, as the
independent samples: the cell reports the standard error of the mean gap,
computed from the ten mean gaps of the start states.

```{.python .input #sac-what-the-second-critic-changes}
%%tab pytorch, jax
def calibration(agent, num_starts=10, num_repeats=3):
    """Mean critic estimate at (s0, a0), mean realized soft return, and the
    mean and standard error of their difference over the start states."""
    env = gym.make('Pendulum-v1', max_episode_steps=1000)
    rng, beta, predicted, realized = np.random.default_rng(0), agent.beta(), [], []
    for k in range(num_starts):
        for _ in range(num_repeats):
            obs = env.reset(seed=1000 + k)[0]
            a, _ = agent.act(obs, rng.standard_normal(1))
            predicted.append(agent.estimate(obs, a))
            total, discount, done = 0.0, 1.0, False
            while not done:
                obs, rew, terminated, truncated, _ = env.step(a)
                total, discount = total + discount * rew, discount * gamma
                done = terminated or truncated
                if not done:   # the entropy bonus of the next state
                    a, log_prob = agent.act(obs, rng.standard_normal(1))
                    total -= discount * beta * log_prob
            realized.append(total)
    gap = (np.array(predicted) - np.array(realized)).reshape(
        num_starts, num_repeats).mean(1)   # the mean gap of each start state
    return (np.mean(predicted), np.mean(realized), gap.mean(),
            gap.std(ddof=1) / np.sqrt(num_starts))

for arm in arms:
    for seed, agent in enumerate(agents[arm]):
        pred, real, gap, se = calibration(agent)
        print(f'{arm:>11}, seed {seed}: predicted {pred:7.1f}, realized '
              f'{real:7.1f}, gap {gap:+5.1f} (s.e. {se:.1f})')
```

With one critic, the estimate exceeds the realized soft return by roughly
$20$ to $35$ in every run, more than three standard errors: the critic
overestimates the value of its own policy. With two critics, the gaps are on
average several times smaller, between about $-2$ and $+13$, and in each
framework every run with two critics has a smaller gap than every run with
one. Three of the six runs with two critics are within one standard error
of zero, and none is more than about two standard errors above it. Here, the
minimum of two critics removes most of the overestimation. It does not reduce
the number of environment steps until the mean return reaches $-200$: with
one critic, every seed reaches it as early as with two.

## Summary

SAC maximizes the expected return with the entropy of the policy added to
every reward. In its soft Bellman backup, the state value contains the term
$-\beta \log \pi$, and the improved policy is the softmax policy of the
action values, $\pi^* \propto \exp(Q/\beta)$, which becomes greedy as
$\beta \to 0$. SAC fits an actor of a restricted family to this policy,
draws its actions by reparameterizing a squashed Gaussian, bootstraps from
the smaller of the estimates of two target critics, which follow their
critics by Polyak averaging, and learns the temperature so that the entropy
stays near a target. On Pendulum, the mean return of SAC over five
consecutive episodes first reached $-200$ after about $5{,}000$ environment
steps, with one critic as early as with two. The second critic changed the
value estimates: with one critic, they exceeded the realized soft returns in
every run, and with two critics, on average, by several times less. The next
section turns to learning from a fixed dataset, without any further
interaction with the environment.

## Exercises

1. [code] **A fixed temperature.** Replace the learned temperature by a fixed
   $\beta \in \{0.02, 0.2, 1\}$. How do the return curves and the entropy of the
   trained policy change? Which fixed value is closest to the temperature that
   the learned runs end with, and how do the two compare?
1. **The greedy limit.** For the policy :eqref:`eq_sac_boltzmann`, show that
   $V(s) = \mathbb{E}_{a \sim \pi^*}[Q(s, a) - \beta \log \pi^*(a \mid s)] =
   \beta \log Z(s)$. For finitely many actions, show that $\beta \log Z(s)$
   tends to $\max_a Q(s, a)$ as $\beta \to 0$, so that the soft backup becomes
   the Bellman optimality backup of :numref:`sec_valueiter`.
1. [code] **The change of variables.** Train SAC with the log-determinant
   term of :eqref:`eq_sac_log_prob` removed from the actor's log-probability. What
   happens to the learned temperature and to the actions of the trained
   policy, and why?
1. [code] **More updates per step.** Take four updates after every
   environment step, in both arms. Does each arm need fewer environment steps
   to reach $-200$? How does the gap between predicted and realized return of
   each arm change?
1. **Discrete actions.** With finitely many actions, the expectations over $a$
   in :eqref:`eq_sac_backup` and :eqref:`eq_sac_actor_loss` can be computed
   exactly instead of sampled :cite:`Christodoulou.2019`. Write the critic
   target and the actor loss for the two actions of CartPole. Which parts of
   the squashed Gaussian actor are no longer needed?

[Discussions](https://d2l.discourse.group/t/36805)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14.11]{.kicker}

Soft actor-critic<br>
**continuous actions · entropy in the return · the soft backup · twin critics · a learned temperature**
:::
:::

::: {.slide title="Entropy in the Return"}
$$J_\beta(\pi) = \mathbb{E}\Big[\sum_t \gamma^t \big(r_t + \beta\, H(\pi(\cdot \mid s_t))\big)\Big]$$

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\big[Q^\pi(s, a) - \beta \log \pi(a \mid s)\big]$$

The improved policy is $\pi^*(a \mid s) \propto \exp(Q(s, a)/\beta)$; the
actor minimizes $\mathbb{E}[\beta \log \pi_\theta - Q]$.
:::

::: {.slide title="A Squashed Gaussian Actor"}
$a = 2\tanh(\mu_\theta(s) + \sigma_\theta(s)\,\xi)$: gradients pass through
the action into the critic.

$$\log \pi_\theta(a \mid s) = \log \mathcal{N}(u; \mu, \sigma^2) - \log\big(2(1 - \tanh^2 u)\big)$$

@sac-a-squashed-gaussian-actor
:::

::: {.slide title="Twin Critics, Soft Targets, a Learned Temperature"}
$$y = r + \gamma\, (1 - d')\, \Big(\min_{j} Q_{w_j^-}(s', a') - \beta \log \pi_\theta(a' \mid s')\Big)$$

- the minimum of two critics against overestimation;
- Polyak averaging of the target copies;
- $\beta$ keeps the entropy near $\bar{H} = -1$.
:::

::: {.slide title="Learning to Swing Up"}
@sac-learning-to-swing-up-2
:::

::: {.slide title="What the Second Critic Changes"}
@sac-what-the-second-critic-changes
:::
