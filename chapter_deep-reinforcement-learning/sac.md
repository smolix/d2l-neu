# Soft Actor-Critic
:label:`sec_sac`

:numref:`sec_regularized` closed with a promise: an algorithm assembled from three components this book already owns, the entropy-regularized objective of that section, the pathwise gradient of :numref:`sec_deeprl`, and the replay-and-target machinery of :numref:`sec_dqn`, plus one mechanism it deferred as bookkeeping. This section is that paragraph made executable. Soft Actor-Critic, SAC :cite:`Haarnoja.Zhou.Abbeel.ea.2018,Haarnoja.Zhou.Hartikainen.ea.2018`, is the workhorse of off-policy continuous control, and almost nothing in it will be new: every equation below arrives with a pointer to where you met it, and the one genuinely new ingredient is a single line of calculus. What the assembly buys is measured at the end, on the task :numref:`sec_deeprl` left unfinished: REINFORCE with a learned baseline spent $480{,}000$ Pendulum steps without ever reaching the $-200$ neighborhood of a working swing-up controller, and the agent built here crosses that line in under ten thousand.

```{.python .input #sac-soft-actor-critic-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
```

```{.python .input #sac-soft-actor-critic-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import gymnasium as gym
import jax
from jax import numpy as jnp
import numpy as np
import optax
```

The laboratory is Pendulum, exactly as :numref:`sec_deeprl` set it up: state $(\cos\vartheta, \sin\vartheta, \dot\vartheta)$, a torque in $[-2, 2]$, a cost of roughly the squared angle from upright at every one of 200 steps, so an aimless policy collects about $-1200$ per episode and a controller that swings up and balances about $-200$ or better. The budget is stated in environment steps, the unit the agent spends (:numref:`sec_dqn`): $20{,}000$ steps per run, one gradient update per step after a thousand random warm-up steps. The critics are ordinary regression networks that take the state *and the action* as input and return one number; the argmax over actions that :numref:`sec_dqn` needed is about to be replaced by a policy trained to climb them.

```{.python .input #sac-soft-actor-critic-2}
%%tab pytorch, jax
gamma, num_env_steps, num_seeds = 0.99, 20_000, 3
buffer_size, batch_size, lr = 200_000, 256, 3e-4
alpha, tau, warmup, c = 0.2, 0.005, 1_000, 2.0
if tab.selected('pytorch'):
    def make_critic():
        return nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                             nn.Linear(64, 64), nn.ReLU(),
                             nn.Linear(64, 1))
if tab.selected('jax'):
    def make_critic(rngs):
        return nnx.Sequential(nnx.Linear(4, 64, rngs=rngs), jax.nn.relu,
                              nnx.Linear(64, 64, rngs=rngs), jax.nn.relu,
                              nnx.Linear(64, 1, rngs=rngs))
```

## The Objective and Its Two Backups

### Maximum entropy, charged at every step

Start from what :numref:`sec_regularized` proved. That section's objective charged, at every step, a KL penalty against a fixed reference policy, and its proposition gave the optimum in closed form; one of its four consequences was that a uniform reference turns the penalty into an entropy bonus, up to a constant that moves no optimizer. SAC maximizes exactly that special case, written in the field's notation, $\alpha$ for the exchange rate :numref:`sec_regularized` called $\beta$:

$$
J(\pi) = E_{\pi}\Big[ \sum_t \gamma^t \big( r_t + \alpha\, H(\pi(\cdot \mid s_t)) \big) \Big].
$$
:eqlabel:`eq_maxent_objective`

Read the term placement carefully, because it is the most common misreading of this method: the entropy is not a bonus bolted onto training the way :numref:`sec_ppo`'s entropy regularizer was bolted onto the actor loss. It is part of the *objective*, so it changes the optimum itself, and the optimal policy is stochastic by design rather than stochastic while training lasts. With a continuous action set the entropy is differential entropy, an integral rather than a sum, and it can be negative; that detail returns below with real stakes. This is maximum-entropy reinforcement learning, a line of work older than deep learning :cite:`Ziebart.Maas.Bagnell.ea.2008`, read by :numref:`sec_regularized` as inference :cite:`Levine.2018`; SAC is its off-policy actor-critic instance, with a lineage running through soft Q-learning :cite:`Haarnoja.Tang.Abbeel.ea.2017`.

Like every objective in these two chapters, :eqref:`eq_maxent_objective` is optimized by generalized policy iteration (:numref:`fig_rl_gpi`): an evaluation half that estimates values under the current policy, and an improvement half that uses them. Both halves go soft, and each is a two-line edit of machinery you have.

### Soft evaluation: one new term in the target

Under :eqref:`eq_maxent_objective` the value of a state collects reward *and* the entropy the policy will earn later, so the state value satisfies $V^{\pi}(s) = E_{a \sim \pi}\big[ Q^{\pi}(s, a) - \alpha \log \pi(a \mid s) \big]$: the entropy is $-E[\log \pi]$, sampled by the action just drawn. Substituting this into the one-step bootstrap of :numref:`sec_actorcritic` grows the critic's regression target by exactly one term,

$$
y = r + \gamma\, \big(1 - \mathbf{1}(s' \textrm{ terminal})\big) \Big( \min_{j=1,2} Q_{w_j^-}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}' \mid s') \Big), \qquad \tilde{a}' \sim \pi_\theta(\cdot \mid s'),
$$
:eqlabel:`eq_sac_target`

and this is the difference between an entropy *bonus* and an entropy *objective* in one display: PPO's bonus lived in the actor loss alone, while the $-\alpha \log \pi$ here lives in the critic's target, because a soft critic must value the entropy the policy will collect for the rest of time. Everything else in the display is a promissory note from :numref:`sec_dqn`, twin critics $Q_{w_1}, Q_{w_2}$, frozen copies $w_j^-$, and a minimum, cashed in the algorithm section below. Note also what is *not* in it. The transition $(s, a, r, s')$ comes from the replay buffer, but the action $\tilde{a}'$ is sampled fresh from the current policy, because the expectation defining $V^{\pi}$ is an expectation under the policy being evaluated; the buffer supplies only the environment's part of the data, which no policy owns. And soft evaluation is ordinary policy evaluation on the augmented reward $r + \alpha H$, so the contraction argument of :numref:`sec_valueiter` carries over unedited.

The new term is not an approximation of :numref:`sec_regularized`'s soft backup; at the per-state optimum it *is* that backup. If $\pi^\star(a) \propto e^{Q(s, a)/\alpha}$, then $\log \pi^\star = Q/\alpha - \log Z$, so the bracket $Q - \alpha \log \pi^\star$ is the same number $\alpha \log Z$ for every action, and its expectation is the logsumexp that section displayed. Three lines certify the identity to floating-point depth, the same genre of check :numref:`sec_dqn` used to certify "the same expression" across tabs:

```{.python .input #sac-soft-evaluation-one-new-term-in-the-target}
%%tab pytorch, jax
rng = np.random.default_rng(0)
Q = rng.normal(0, 1, 5)                        # any five action values
for temp in (0.1, 0.5, 2.0):                   # pi* proportional to e^{Q/temp}
    pi = np.exp(Q / temp - (Q / temp).max())
    pi /= pi.sum()
    lhs = (pi * (Q - temp * np.log(pi))).sum()
    rhs = temp * ((Q / temp).max()
                  + np.log(np.exp(Q / temp - (Q / temp).max()).sum()))
    print(f'alpha = {temp}: E[Q - alpha log pi*] = {lhs:.10f}, '
          f'alpha logsumexp(Q/alpha) = {rhs:.10f}')
```

The sampled form in :eqref:`eq_sac_target` is what an actor-critic can compute when the sum over actions is an integral: the logsumexp of :numref:`sec_regularized` needed to enumerate the action set, and $\tilde{a}' \sim \pi_\theta$ estimates the same expectation with one draw.

### Soft improvement is the proposition we already proved

The improvement half is where owning :numref:`sec_regularized` pays most. Its proposition's proof established, for any policy $\pi$ and any reward vector,

$$
E_{\pi}[r] - \beta\, D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}}) = \beta \log Z - \beta\, D_{\textrm{KL}}(\pi \Vert \pi^\star).
$$

Read it with $r \to Q(s, \cdot)$, a uniform reference, and $\beta \to \alpha$: **maximizing $E_{\pi}[Q] + \alpha H(\pi)$ and minimizing $D_{\textrm{KL}}\big(\pi \,\Vert\, e^{Q/\alpha}/Z\big)$ are the same optimization**, their objectives differing by $\alpha \log Z$, which does not depend on the policy. The literature states SAC's actor update in both forms, usually asserting one and gesturing at the other; the bridge between them is a proof this book has already run, and it costs nothing to cross. A parametric policy $\pi_\theta$ cannot represent the tilted optimum exactly, so the actor update is the KL projection of $e^{Q/\alpha}/Z$ onto the family, or equivalently, by the bridge, a few gradient steps on

$$
L_{\pi}(\theta) = E_{s \sim \mathcal{D},\, z}\Big[ \alpha \log \pi_\theta\big(\tilde{a}_\theta(s, z) \mid s\big) - \min_{j=1,2} Q_{w_j}\big(s, \tilde{a}_\theta(s, z)\big) \Big], \qquad \tilde{a}_\theta(s, z) = c \tanh\big(\mu_\theta(s) + \sigma_\theta(s)\, z\big),
$$
:eqlabel:`eq_sac_actor`

with $z \sim \mathcal{N}(0, I)$ and states drawn from the replay buffer $\mathcal{D}$. How to differentiate an expectation whose distribution depends on $\theta$ is a solved problem twice over, and :eqref:`eq_sac_actor` picks the second solution: this is the pathwise gradient of :eqref:`eq_score_vs_pathwise`, differentiating straight through the sampled action into a critic that, being a network, is differentiable in the action by construction. :numref:`sec_deeprl` priced the choice, factor-of-twenty variance reductions in exchange for exactly that differentiability premise, and the premise is why the recipe does not port to discrete actions without summing the expectation exactly or smoothing the argmax away.

One guarantee travels with the improvement step, and it is five lines from the same proposition.

**Proposition.** Let $\Pi$ contain $\pi_{\textrm{old}}$, and let $\pi_{\textrm{new}}(\cdot \mid s)$ maximize $E_{a \sim \pi}\big[ Q^{\pi_{\textrm{old}}}(s, a) \big] + \alpha H(\pi(\cdot \mid s))$ over $\Pi$ at every $s$. Then $V^{\pi_{\textrm{new}}} \geq V^{\pi_{\textrm{old}}}$ everywhere.

**Proof.** Since $\pi_{\textrm{old}} \in \Pi$, optimality at $s$ gives

$$
E_{a \sim \pi_{\textrm{new}}}\big[ Q^{\pi_{\textrm{old}}}(s, a) - \alpha \log \pi_{\textrm{new}}(a \mid s) \big] \;\geq\; E_{a \sim \pi_{\textrm{old}}}\big[ Q^{\pi_{\textrm{old}}}(s, a) - \alpha \log \pi_{\textrm{old}}(a \mid s) \big] = V^{\pi_{\textrm{old}}}(s).
$$

Expand $Q^{\pi_{\textrm{old}}}(s, a) = r(s, a) + \gamma\, E_{s'}\big[ V^{\pi_{\textrm{old}}}(s') \big]$ on the left and apply the same inequality at $s'$, then at $s''$, and so on: each substitution pushes $V^{\pi_{\textrm{old}}}$ one step deeper while accumulating rewards and entropies collected under $\pi_{\textrm{new}}$. Bounded rewards and $\gamma < 1$ send the remainder to zero, and the accumulated series is $V^{\pi_{\textrm{new}}}(s)$. $\blacksquare$

The caveat is the one :numref:`sec_actorcritic` and :numref:`sec_dqn` attach to their own guarantees: the argument assumes exact per-state maximization and a converged $Q^{\pi_{\textrm{old}}}$, and SAC takes one gradient step on each. It is the shape of the guarantee, not a certificate for the loop below; the full telescoping treatment is Appendix B of :cite:`Haarnoja.Zhou.Abbeel.ea.2018`.

## A Policy That Fits in a Box

### Why clipping kills a pathwise gradient

:numref:`sec_deeprl`'s `GaussianPolicy` emitted an unbounded Gaussian and let the environment clip the torque to $[-2, 2]$, and that was correct there, because the score-function estimator touches the environment only through the returned reward; nothing behind that interface needs a derivative. The pathwise estimator differentiates *through the action*, and the clip destroys it twice. First, outside the box the clip has derivative zero, so for any state whose sampled action lands beyond the boundary, $\partial a / \partial \theta = 0$ and the actor receives no signal at all, and states whose best action sits *at* the boundary, maximum torque mid-swing, are exactly where a swing-up controller lives. Second, a clipped Gaussian is not a density: it piles atoms of probability onto the two boundary points, and the $\alpha \log \pi$ that :eqref:`eq_sac_target` and :eqref:`eq_sac_actor` both charge is undefined on an atom. Two failures, one repair: make the boundedness part of the distribution, differentiably.

### The change of variables

Draw $u \sim \mathcal{N}\big(\mu_\theta(s), \sigma_\theta(s)^2\big)$ and squash, $a = c \tanh u$ with $c = 2$ the action scale. The map is smooth and strictly monotone, so the change-of-variables formula gives the density its Jacobian correction, diagonal here because the squash acts coordinatewise:

$$
\log \pi(a \mid s) = \sum_i \Big[ \log \mathcal{N}\big(u_i;\, \mu_i, \sigma_i\big) - \log\big(1 - \tanh^2 u_i\big) - \log c \Big].
$$
:eqlabel:`eq_tanh_logdet`

This is the deferred mechanism in its entirety, Appendix C of :cite:`Haarnoja.Zhou.Abbeel.ea.2018` with the action scale carried explicitly; implementations disagree about the $\log c$, and the convention matters, because it cancels in the actor's *gradient* but not in the printed entropy nor in the $\alpha \log \pi$ inside the target, where dropping it shifts every reported nat by $\log 2$. We carry it: :eqref:`eq_tanh_logdet` is the density of the action the environment actually receives, and the quadrature check below holds it to that standard. In code, the policy is :numref:`sec_deeprl`'s idea with the same three methods, `log_prob`, `act`, `act_greedy`, overridden a second time, plus the one method the pathwise gradient demands and the score function never needed: `sample`, a reparameterized draw that returns the action together with its log-probability, differentiable end to end. Two details are new. The standard deviation is state-dependent, a second head next to the mean, because a squashed policy must be able to keep noise where it is cheap and shed it where it saturates, and its logarithm is clamped to $[-5, 2]$, without which $\sigma$ collapses and $\log \pi$ diverges within a thousand updates. And `log_prob` takes the *pre-squash* $u$ rather than the action: the sampler always keeps $u$, so the training path never needs to invert the $\tanh$, and the one place that does invert it, the quadrature check, is the one place $\mathrm{arctanh}$ appears in this section.

```{.python .input #sac-the-change-of-variables}
%%tab pytorch
class SquashedGaussianPolicy(nn.Module):
    """A state-dependent Gaussian squashed through a = c tanh(u)."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.mu(h), self.log_std(h).clamp(-5, 2).exp()

    def log_prob(self, u, mean, std):
        """log pi at a = c tanh(u), from the pre-squash u the sampler keeps."""
        logdet = 2 * (np.log(2) - u - nn.functional.softplus(-2 * u))
        return (torch.distributions.Normal(mean, std).log_prob(u)
                - logdet - np.log(c)).sum(-1)

    def sample(self, obs):
        """A reparameterized action and its log-probability, differentiable."""
        mean, std = self(obs)
        u = mean + std * torch.randn_like(std)
        return c * torch.tanh(u), self.log_prob(u, mean, std)

    def act(self, obs, rng):
        with torch.no_grad():
            mean, std = self(torch.as_tensor(obs))
        u = mean.numpy() + std.numpy() * rng.standard_normal(
            mean.shape, dtype=np.float32)
        return c * np.tanh(u)

    def act_greedy(self, obs, rng=None):
        with torch.no_grad():
            return c * np.tanh(self(torch.as_tensor(obs))[0].numpy())
```

```{.python .input #sac-the-change-of-variables}
%%tab jax
class SquashedGaussianPolicy(nnx.Module):
    """A state-dependent Gaussian squashed through a = c tanh(u)."""
    def __init__(self, obs_dim, act_dim, hidden=64, rngs=None):
        self.trunk = nnx.Sequential(
            nnx.Linear(obs_dim, hidden, rngs=rngs), jax.nn.relu,
            nnx.Linear(hidden, hidden, rngs=rngs), jax.nn.relu)
        self.mu = nnx.Linear(hidden, act_dim, rngs=rngs)
        self.log_std = nnx.Linear(hidden, act_dim, rngs=rngs)

    def __call__(self, obs):
        h = self.trunk(obs)
        return self.mu(h), jnp.exp(jnp.clip(self.log_std(h), -5, 2))

    def log_prob(self, u, mean, std):
        """log pi at a = c tanh(u), from the pre-squash u the sampler keeps."""
        logdet = 2 * (jnp.log(2.0) - u - jax.nn.softplus(-2 * u))
        return (jax.scipy.stats.norm.logpdf(u, mean, std)
                - logdet - jnp.log(c)).sum(-1)

    def sample(self, obs, key):
        """A reparameterized action and its log-probability, differentiable."""
        mean, std = self(obs)
        u = mean + std * jax.random.normal(key, std.shape)
        return c * jnp.tanh(u), self.log_prob(u, mean, std)

    def act(self, obs, rng):
        if not hasattr(self, '_fwd'):   # compile the fixed-shape acting
            self._fwd = nnx.cached_partial(nnx.jit(lambda net, o: net(o)),
                                           self)  # forward, once
        mean, std = self._fwd(jnp.asarray(obs))
        u = np.asarray(mean) + np.asarray(std) * rng.standard_normal(
            mean.shape, dtype=np.float32)
        return c * np.tanh(u)

    def act_greedy(self, obs, rng=None):
        mean, _ = self(jnp.asarray(obs))
        return c * np.tanh(np.asarray(mean))
```

One bookkeeping rule inside `log_prob` deserves its sentence: the Gaussian term and the Jacobian term are summed *together* over the action dimension before anything is averaged over the batch. Mixing a sum on one with a mean on the other is a documented bug class in public implementations, invisible at one action dimension and quietly wrong at two, and exercise 6 ports this code to a two-dimensional action.

### The stable form, and what the epsilon hides

The direct transcription `log(1 - tanh(u)**2)` is numerically dead on arrival, and the repair is two lines of algebra rather than an epsilon. Since $1 - \tanh^2 u = \operatorname{sech}^2 u = 4 e^{-2u} / (1 + e^{-2u})^2$, taking logarithms gives

$$
\log\big(1 - \tanh^2 u\big) = 2\, \big( \log 2 - u - \operatorname{softplus}(-2u) \big),
$$

exact for every $u$, with no subtraction of nearly equal numbers anywhere; it is what the `log_prob` above computes. The measurement of what it repairs is worth a cell, because the failure it prevents is the quiet kind this chapter keeps warning about. Three candidates, in float32, the arithmetic both tabs run:

```{.python .input #sac-the-stable-form-and-what-the-epsilon-hides-1}
%%tab pytorch, jax
u = np.float32([0, 3, 8, 10, 20])
t = np.tanh(u)
with np.errstate(divide='ignore'):
    naive = np.log(1 - t ** 2)
guarded = np.log(1 - t ** 2 + np.float32(1e-6))
stable = 2 * (np.log(np.float32(2)) - u - np.logaddexp(np.float32(0), -2 * u))
print(f'{"u":>4} {"naive":>10} {"guarded":>10} {"stable":>10}')
for row in zip(u, naive, guarded, stable):
    print(f'{row[0]:4.0f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}')
```

The naive form hits $-\infty$ the moment float32 rounds $\tanh u$ to $1$, at $|u|$ around $10$; that failure at least announces itself. The interesting column is the guarded one, the widespread `+ 10^{-6}` patch: it never crashes and it is silently wrong, pinned at $\log 10^{-6} = -13.8155$ for every $|u| \gtrsim 7$. From there on the density stops charging the policy for saturating, which is precisely the regularizer's job in :eqref:`eq_sac_actor`; the reported entropy detaches from the actual noise, the temperature loses its meaning, and no curve anywhere looks wrong. Deep reinforcement learning does not fail loudly, and this is the failure mode in one printed table.

Whether the whole change of variables is right is also checkable in one cell, with no training and no reference implementation: a density must integrate to one. Grid the action interval, invert the squash on the grid, the section's only $\mathrm{arctanh}$, and integrate :eqref:`eq_tanh_logdet` by the trapezoid rule, with and without the Jacobian term:

```{.python .input #sac-the-stable-form-and-what-the-epsilon-hides-2}
%%tab pytorch, jax
def squashed_logp(a, mu, sigma, logdet=True):
    up = np.arctanh(a / c)          # only here: the training path keeps u
    lp = (-0.5 * ((up - mu) / sigma) ** 2 - np.log(sigma)
          - 0.5 * np.log(2 * np.pi))
    if logdet:
        lp -= 2 * (np.log(2) - up - np.logaddexp(0, -2 * up)) + np.log(c)
    return lp

a_grid = np.linspace(-c + 1e-6, c - 1e-6, 200_001)
for m, s in ((0.0, 0.5), (0.7, 0.8)):
    w = np.trapezoid(np.exp(squashed_logp(a_grid, m, s)), a_grid)
    wo = np.trapezoid(np.exp(squashed_logp(a_grid, m, s, False)), a_grid)
    print(f'mu = {m}, sigma = {s}: integrates to {w:.6f} with the '
          f'log-det, {wo:.3f} without')
```

With the correction the density integrates to $1.000000$ at both parameter settings; without it the same integral reads $1.65$ and $1.13$, off by half its own mass, and the "density" being trained against is not a density at all. This one line is the log-det ablation at zero training cost, and it is the cheaper half of a lesson the experiments finish: deleting the correction barely moves the Pendulum return, but it detaches every reported entropy from reality, which is exactly how such a bug ships. The grid margin of $10^{-6}$ is the check's residual, since $\mathrm{arctanh}$ overflows at the boundary itself, and pushing $\sigma$ far above one piles mass against the boundary faster than any fixed grid resolves, which is why the check uses moderate parameters.

## The Algorithm

### Two critics, and why the minimum

:numref:`sec_dqn` measured what a maximum over noisy estimates does: it flatters itself, one full unit of bias from four actions of unit noise. SAC's actor is not an argmax, but it is trained to *climb* the critic, which makes it a maximizer over the critic's errors all the same, the Thrun-Schwartz argument :cite:`Thrun.Schwartz.1993` with the enumeration replaced by gradient ascent. The repair is imported from TD3 :cite:`Fujimoto.vanHoof.Meger.2018`, the deterministic sibling that diagnosed this bias in continuous control: train two critics, independently initialized, and let every consumer of a value take the pointwise minimum, both the target in :eqref:`eq_sac_target` and the actor loss in :eqref:`eq_sac_actor`. The point is not that an ensemble of two averages away noise; the point is the $\min$, the cheapest available pessimistic estimate. What that pessimism buys on this task is measured at the end of the section, and it is a calibration story, not a return story.

### A target that drifts instead of jumping

The frozen copies $w_j^-$ do the target network's job from :numref:`sec_dqn` on a different schedule. Rather than a hard sync every $C$ steps, every update moves the copy a small fraction of the way to the online weights, Polyak averaging, $w^- \leftarrow \tau w + (1 - \tau) w^-$ with $\tau = 0.005$: an exponential moving average with a half-life of $\ln 2 / \tau \approx 139$ updates, so the regression surface drifts continuously instead of standing still and jumping. Same stability mechanism, no sync-moment shocks to guard against. One asymmetry is worth noticing because DDPG and TD3 got it wrong before SAC got it right: only the *critics* have frozen copies. The action $\tilde{a}'$ in the target comes from the live policy, and no target-policy smoothing is needed, because a stochastic policy smooths its own targets by sampling.

### Nothing here needs a ratio

This is the chapter's second off-policy algorithm, and it earns the license the way :numref:`sec_dqn` did, not the way :numref:`sec_ppo` did. PPO's importance ratios corrected an expectation over actions *the old policy chose*; here no estimate is ever taken under the collecting policy. The critic's target never mentions the collector, and the actor's expectation re-samples its own fresh actions, so the buffer contributes only states. What replay does shift is the *state distribution* the losses average over, a shift no action ratio repairs. The cost of that shift is the organizing subject of :numref:`sec_offline`; here, where the agent keeps interacting, it stays mild.

The container is three lines of bookkeeping. :numref:`sec_deeprl`'s `ActorCritic` finally stops fitting, for a structural reason worth naming: its second head was $V(s)$, and SAC's critics take the action as an input, so the pair $(Q_{w_1}, Q_{w_2})$ with their frozen copies replaces the value head. The replay buffer of :numref:`sec_dqn` needs one column widened, since it stored actions as integers for a discrete world and Pendulum's action is a float vector; everything else, the ring, the eviction, the time-scrambling `sample`, is inherited unchanged.

```{.python .input #sac-nothing-here-needs-a-ratio-1}
%%tab pytorch, jax
class ReplayBufferC(d2l.ReplayBuffer):
    """d2l.ReplayBuffer with the action column widened to float vectors."""
    def __init__(self, capacity, obs_dim, act_dim):
        super().__init__(capacity, obs_dim)
        self.act = np.zeros((capacity, act_dim), np.float32)
```

```{.python .input #sac-nothing-here-needs-a-ratio-2}
%%tab pytorch
class SoftActorCritic:
    """One actor, a list of action-value critics, and their frozen copies."""
    def __init__(self, actor, qs, targets, lr=lr):
        self.actor, self.qs, self.targets = actor, qs, targets
        self.opt_pi = torch.optim.Adam(actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(
            [p for q in qs for p in q.parameters()], lr=lr)

    def min_q(self, obs, act, nets=None):
        x = torch.cat([obs, act], -1)
        return torch.min(torch.stack(
            [q(x).squeeze(-1) for q in (self.qs if nets is None else nets)]),
            0).values

def make_sac(seed, num_critics=2):
    torch.manual_seed(seed)
    actor = SquashedGaussianPolicy(3, 1)
    qs = [make_critic() for _ in range(num_critics)]
    targets = [make_critic() for _ in range(num_critics)]
    for q, tnet in zip(qs, targets):
        tnet.load_state_dict(q.state_dict())
    return SoftActorCritic(actor, qs, targets)
```

```{.python .input #sac-nothing-here-needs-a-ratio-2}
%%tab jax
class SoftActorCritic(nnx.Module):
    """One actor, a list of action-value critics, and their frozen copies."""
    def __init__(self, actor, qs, targets, lr=lr):
        self.actor = actor
        self.qs, self.targets = nnx.List(qs), nnx.List(targets)
        self.opt_pi = nnx.Optimizer(actor, optax.adam(lr), wrt=nnx.Param)
        self.opt_q = nnx.Optimizer(self.qs, optax.adam(lr), wrt=nnx.Param)

    def min_q(self, obs, act, nets=None):
        nets = self.qs if nets is None else nets
        x = jnp.concatenate([obs, act], -1)
        return jnp.min(jnp.stack([q(x).squeeze(-1) for q in nets]), 0)

def make_sac(seed, num_critics=2):
    rngs = nnx.Rngs(seed)
    actor = SquashedGaussianPolicy(3, 1, rngs=rngs)
    qs = [make_critic(rngs) for _ in range(num_critics)]
    return SoftActorCritic(actor, qs, [nnx.clone(q) for q in qs])
```

### One update

The update is :eqref:`eq_sac_target` and :eqref:`eq_sac_actor` verbatim: a regression step for both critics toward the shared soft target, one pathwise ascent step for the actor against the current critics' minimum, then the Polyak drift. The critic target is computed without gradients, data by the time the regression sees it, the same discipline `td_target`'s numpy boundary enforced in :numref:`sec_actorcritic`; both fresh actions are drawn inside the update, $\tilde{a}'$ for the target and $\tilde{a}$ for the actor.

```{.python .input #sac-one-update-1}
%%tab pytorch
def sac_step(agent, batch):
    """One SAC update: soft critic regression, pathwise actor step, Polyak."""
    obs, act = torch.as_tensor(batch.obs), torch.as_tensor(batch.act)
    rew, term = torch.as_tensor(batch.rew), torch.as_tensor(batch.term)
    next_obs = torch.as_tensor(batch.next_obs)
    with torch.no_grad():                     # the critic target is data
        a2, logp2 = agent.actor.sample(next_obs)
        y = rew + gamma * (1 - term) * (
            agent.min_q(next_obs, a2, agent.targets) - alpha * logp2)
    x = torch.cat([obs, act], -1)
    loss_q = sum(((q(x).squeeze(-1) - y) ** 2).mean() for q in agent.qs)
    agent.opt_q.zero_grad()
    loss_q.backward()
    agent.opt_q.step()
    a, logp = agent.actor.sample(obs)         # fresh, from the live policy
    loss_pi = (alpha * logp - agent.min_q(obs, a)).mean()
    agent.opt_pi.zero_grad()
    loss_pi.backward()
    agent.opt_pi.step()
    with torch.no_grad():                     # Polyak: the drifting copy
        for q, tnet in zip(agent.qs, agent.targets):
            for pq, pt in zip(q.parameters(), tnet.parameters()):
                pt.mul_(1 - tau).add_(tau * pq)
    return float(logp.detach().mean())
```

```{.python .input #sac-one-update-1}
%%tab jax
@nnx.jit
def sac_step(agent, obs, act, rew, next_obs, term, key):
    """One SAC update: soft critic regression, pathwise actor step, Polyak.
    Fixed batch shapes, so the step compiles once (:numref:`sec_compilation`)."""
    k1, k2 = jax.random.split(key)
    a2, logp2 = agent.actor.sample(next_obs, k1)   # fresh, live policy
    y = rew + gamma * (1 - term) * (
        agent.min_q(next_obs, a2, agent.targets) - alpha * logp2)
    def q_loss(qs):
        x = jnp.concatenate([obs, act], -1)
        return sum(((q(x).squeeze(-1) - y) ** 2).mean() for q in qs)
    _, grads = nnx.value_and_grad(q_loss)(agent.qs)
    agent.opt_q.update(agent.qs, grads)
    def pi_loss(actor):
        a, logp = actor.sample(obs, k2)
        return (alpha * logp - agent.min_q(obs, a)).mean(), logp
    (_, logp), grads = nnx.value_and_grad(pi_loss, has_aux=True)(agent.actor)
    agent.opt_pi.update(agent.actor, grads)
    for q, tnet in zip(agent.qs, agent.targets):   # Polyak: the drifting copy
        nnx.update(tnet, jax.tree.map(lambda p, tp: tau * p + (1 - tau) * tp,
                                      nnx.state(q, nnx.Param),
                                      nnx.state(tnet, nnx.Param)))
    return logp.mean()
```

The loop stores `terminated` and never `truncated`, as every buffer in this chapter does, and here the rule reaches its cleanest instance: Pendulum has *no* terminal state, every episode is a 200-step recording cut off by the clock, so the stored flag is identically zero and the bootstrap is always taken. Store `done` instead and the agent is taught that the world ends at step 200, at whatever state the clock happened to catch it in. The loop yields, at every episode's end, the step count, the episode's return, and the average of $-\log \pi$ over the episode's updates, the policy's entropy as the training batches sample it; the entropy is a first-class diagnostic here, since it is the quantity the objective buys.

```{.python .input #sac-one-update-2}
%%tab pytorch
def train_sac(seed, agent, num_env_steps=num_env_steps):
    """SAC on Pendulum; yields (env step, episode return, policy entropy)."""
    rng, env = np.random.default_rng(seed), gym.make('Pendulum-v1')
    buffer = ReplayBufferC(buffer_size, 3, 1)
    obs = env.reset(seed=seed)[0]
    ep_return, logp_sum, n_upd = 0.0, 0.0, 0
    for t in range(1, num_env_steps + 1):
        a = (rng.uniform(-c, c, 1).astype(np.float32) if t <= warmup
             else agent.actor.act(obs, rng))
        next_obs, rew, terminated, truncated, _ = env.step(a)
        buffer.add(obs, a, rew, next_obs, float(terminated))
        obs, ep_return = next_obs, ep_return + rew
        if t >= warmup:
            logp_sum += sac_step(agent, buffer.sample(batch_size, rng))
            n_upd += 1
        if terminated or truncated:
            yield t, ep_return, (-logp_sum / n_upd if n_upd else np.nan)
            obs, ep_return = env.reset()[0], 0.0
            logp_sum, n_upd = 0.0, 0
```

```{.python .input #sac-one-update-2}
%%tab jax
def train_sac(seed, agent, num_env_steps=num_env_steps):
    """SAC on Pendulum; yields (env step, episode return, policy entropy).
    The jitted step's module traversal is cached once per run."""
    rng, env = np.random.default_rng(seed), gym.make('Pendulum-v1')
    step_fn = nnx.cached_partial(sac_step, agent)
    key = jax.random.PRNGKey(seed)
    buffer = ReplayBufferC(buffer_size, 3, 1)
    obs = env.reset(seed=seed)[0]
    ep_return, logp_sum, n_upd = 0.0, 0.0, 0
    for t in range(1, num_env_steps + 1):
        a = (rng.uniform(-c, c, 1).astype(np.float32) if t <= warmup
             else agent.actor.act(obs, rng))
        next_obs, rew, terminated, truncated, _ = env.step(a)
        buffer.add(obs, a, rew, next_obs, float(terminated))
        obs, ep_return = next_obs, ep_return + rew
        if t >= warmup:
            b = buffer.sample(batch_size, rng)
            logp_sum += float(step_fn(
                jnp.asarray(b.obs), jnp.asarray(b.act), jnp.asarray(b.rew),
                jnp.asarray(b.next_obs), jnp.asarray(b.term),
                jax.random.fold_in(key, t)))
            n_upd += 1
        if terminated or truncated:
            yield t, ep_return, (-logp_sum / n_upd if n_upd else np.nan)
            obs, ep_return = env.reset()[0], 0.0
            logp_sum, n_upd = 0.0, 0
```

Two arms, three seeds each, agents caller-owned as in :numref:`sec_dqn` so that later cells can audit them: the algorithm as described, and an ablation with a single critic, everything else identical. The single-critic arm is not a strawman; it is SAC as its first paper ran it, before the TD3 import.

```{.python .input #sac-one-update-3}
%%tab pytorch, jax
arms = {'SAC': 2, 'single critic': 1}
agents = {arm: [make_sac(seed, n) for seed in range(num_seeds)]
          for arm, n in arms.items()}
runs = {arm: np.array([list(train_sac(seed, agents[arm][seed]))
                       for seed in range(num_seeds)])
        for arm in arms}
```

## What It Bought

### Sample efficiency, in environment steps

The headline claim of the off-policy family is sample efficiency, so the headline plot puts the return against environment steps, with the $-200$ line of a working swing-up controller dashed:

```{.python .input #sac-sample-efficiency-on-the-axis-that-bills}
%%tab pytorch, jax
grid = np.arange(1, num_env_steps // 1000 + 1) * 1000

def on_grid(arm, col, k=5):
    """Trailing-k-episode averages, resampled onto the env-step grid."""
    out = []
    for r in runs[arm]:
        v = r[~np.isnan(r[:, col])]
        s = np.convolve(v[:, col], np.ones(k) / k, 'valid')
        out.append(np.interp(grid, v[k - 1:, 0], s))
    return np.stack(out)

def steps_to(r, level=-200, k=5):
    """First env step at which the trailing-k-episode average clears level."""
    s = np.convolve(r[:, 1], np.ones(k) / k, 'valid')
    hit = np.flatnonzero(s >= level)
    return int(r[hit[0] + k - 1, 0]) if len(hit) else None

d2l.plot_curves({arm: on_grid(arm, 1) for arm in runs},
                xlabel='thousand environment steps', ylabel='episode return',
                reference=-200)
for arm in runs:
    print(f'{arm}: trailing five-episode average first clears -200 at '
          f'env steps {[steps_to(r) for r in runs[arm]]}')
```

Every seed of both arms starts at the aimless policy's $-1200$ or worse and clears the $-200$ line before ten thousand environment steps, most within about five to eight thousand. Put that number against this book's own baseline rather than a foreign one. :numref:`sec_deeprl` trained the same task with REINFORCE and a learned baseline at 300 updates of 8 episodes of 200 steps, which is $480{,}000$ environment steps, twenty-four runs of this budget, and reported improvement without mastery: no seed reached $-200$. The gap is the chapter's two licenses compounding. The pathwise gradient turns the critic into a differentiable model of the objective, so each update extracts more signal per sample, and replay lets every one of those samples drive hundreds of updates instead of one. Note also what the plot does *not* show: the two arms are indistinguishable, their bands overlap the whole way, and neither orders the crossings consistently across tabs. Whatever the second critic buys, it is not visible on this axis at this scale, which is exactly the reading :numref:`sec_dqn` gave for Double DQN at two actions; the last experiment goes looking for what it does buy.

### The entropy the policy keeps

The training loop logged the policy's entropy all along, the quantity the objective pays $\alpha$ per nat for, with the autotuning literature's target of $-\dim \mathcal{A} = -1$ drawn dashed:

```{.python .input #sac-the-entropy-the-policy-keeps-1}
%%tab pytorch, jax
d2l.plot_curves({arm: on_grid(arm, 2) for arm in runs},
                xlabel='thousand environment steps',
                ylabel='policy entropy (nats)', reference=-1)
for arm in runs:
    print(f'{arm}: entropy over the last 20 episodes, per seed '
          f'{np.round([np.nanmean(r[-20:, 2]) for r in runs[arm]], 2)}')
```

Read the shape before the level. The trace enters high, a wide young policy, and the objective spends that entropy quickly while the critic learns what the noise costs; it overshoots to roughly half a nat below zero during the steepest stretch of the return curve, then buys some of the entropy back once the task is mastered and noise near the balanced state is cheap, and ends in the neighborhood of zero, where the printed final values sit in both arms and both tabs. Nothing about that ending is a collapse: the policy remains a genuine distribution at convergence, exactly the "stochastic by design" the objective promised. And since the entropy here is differential, zero is not special and negative values along the way are not a bug, merely a density that concentrates, a fact exercise 4 makes quantitative. Two accounting notes keep this plot honest. The printed entropy is under our convention that carries the $\log c$ of :eqref:`eq_tanh_logdet`; implementations that drop it report the same policy $\log 2 \approx 0.69$ nats lower, and comparing entropy numbers across codebases without checking that convention is comparing different quantities. And the level the trace settles at is the measured justification for fixing $\alpha = 0.2$ rather than building the autotuning apparatus: the constrained variant of :cite:`Haarnoja.Zhou.Hartikainen.ea.2018` would target $\bar{H} = -\dim \mathcal{A} = -1$ and let $\alpha$ float, our fixed exchange rate lands the entropy within about a nat of that target on its own, and exercise 1 closes the remaining gap. What $\alpha$ is *not* is a learning rate: it is an exchange rate in reward units per nat, which makes it reward-scale dependent, doubling every reward halves the effective temperature, and that sensitivity, not any optimization subtlety, is what the autotuning variant automates away.

The evaluation discipline of :numref:`sec_dqn` applies with one twist. The training returns above are collected by the stochastic policy, which is paid to keep noise, so evaluate the deterministic policy $c \tanh(\mu_\theta(s))$ separately, and evaluate the stochastic one beside it to price the noise:

```{.python .input #sac-the-entropy-the-policy-keeps-2}
%%tab pytorch, jax
env = gym.make('Pendulum-v1')
for arm in runs:
    for seed in range(num_seeds):
        agent = agents[arm][seed]
        env.reset(seed=100 + seed)
        g = d2l.evaluate(env, agent.actor.act_greedy, num_episodes=20)
        env.reset(seed=100 + seed)
        s = d2l.evaluate(env, agent.actor.act, num_episodes=20,
                         rng=np.random.default_rng(1000 + seed))
        print(f'{arm}, seed {seed}: deterministic {g:7.1f}, '
              f'stochastic {s:7.1f}, noise cost {g - s:+6.1f}')
```

Every deterministic evaluation, both arms and both tabs, lands between about $-140$ and $-190$, and the stochastic policy lands within about ten points of its own deterministic twin: at the entropy this policy kept, the noise costs almost nothing. That near-equality is not a triviality; it is :numref:`sec_regularized`'s frontier read at its flat top. A concave trade of reward against entropy is nearly level near its peak, so the first nat of noise is nearly free, and a state-dependent $\sigma$ spends it where it is cheapest. The contrast with :numref:`sec_dqn`'s $\epsilon$-greedy tax is instructive: there the exploration noise was a constant foreign forcing that the evaluation had to strip away; here the noise is priced into the objective, and the policy has already moved it out of harm's way.

### Honest promises: predicted against delivered

Promised value against delivered return (:numref:`tab_rl_diagnostics`) is what finally separates the arms. This is the diagnostic this chapter trusts most. For each trained agent, run twenty stochastic episodes, record the critics' promise $\min_j Q_{w_j}(s_0, a_0)$ at each episode's first state and action, and compare it with what the episode then delivered. One bookkeeping precaution makes the comparison fair, and skipping it manufactures a phantom bias: a soft critic predicts the *soft* return, $\sum_t \gamma^t (r_t - \alpha \log \pi(a_t \mid s_t))$, so that is the yardstick, computed from the same log-probabilities the sampler already knows; the plain return is printed beside it to show the stakes of the distinction.

```{.python .input #sac-honest-promises-predicted-against-delivered}
%%tab pytorch
def calibration(agent, seed, num_episodes=20):
    """Predicted min_j Q(s0, a0) against the realized discounted soft return."""
    rng, env = np.random.default_rng(seed), gym.make('Pendulum-v1')
    env.reset(seed=seed)
    preds, softs, plains = [], [], []
    for _ in range(num_episodes):
        obs, done, disc = env.reset()[0], False, 1.0
        soft = plain = 0.0
        first = True
        while not done:
            with torch.no_grad():
                mean, std = agent.actor(torch.as_tensor(obs))
                u = mean + std * torch.as_tensor(rng.standard_normal(
                    mean.shape, dtype=np.float32))
                logp = float(agent.actor.log_prob(u, mean, std))
            a = c * np.tanh(u.numpy())
            if first:
                with torch.no_grad():
                    preds.append(float(agent.min_q(
                        torch.as_tensor(obs), torch.as_tensor(a))))
                first = False
            obs, rew, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            soft += disc * (rew - alpha * logp)
            plain += disc * rew
            disc *= gamma
        softs.append(soft)
        plains.append(plain)
    return np.mean(preds), np.mean(softs), np.mean(plains)

for arm in runs:
    for seed in range(num_seeds):
        pred, soft, plain = calibration(agents[arm][seed], 200 + seed)
        print(f'{arm:>13}, seed {seed}: promised {pred:7.1f}, delivered '
              f'(soft) {soft:7.1f}, gap {pred - soft:+6.1f}, plain {plain:7.1f}')
```

```{.python .input #sac-honest-promises-predicted-against-delivered}
%%tab jax
def calibration(agent, seed, num_episodes=20):
    """Predicted min_j Q(s0, a0) against the realized discounted soft return."""
    rng, env = np.random.default_rng(seed), gym.make('Pendulum-v1')
    env.reset(seed=seed)
    preds, softs, plains = [], [], []
    for _ in range(num_episodes):
        obs, done, disc = env.reset()[0], False, 1.0
        soft = plain = 0.0
        first = True
        while not done:
            mean, std = agent.actor(jnp.asarray(obs))
            u = np.asarray(mean) + np.asarray(std) * rng.standard_normal(
                mean.shape, dtype=np.float32)
            logp = float(agent.actor.log_prob(jnp.asarray(u), mean, std))
            a = c * np.tanh(u)
            if first:
                preds.append(float(agent.min_q(jnp.asarray(obs),
                                               jnp.asarray(a))))
                first = False
            obs, rew, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            soft += disc * (rew - alpha * logp)
            plain += disc * rew
            disc *= gamma
        softs.append(soft)
        plains.append(plain)
    return np.mean(preds), np.mean(softs), np.mean(plains)

for arm in runs:
    for seed in range(num_seeds):
        pred, soft, plain = calibration(agents[arm][seed], 200 + seed)
        print(f'{arm:>13}, seed {seed}: promised {pred:7.1f}, delivered '
              f'(soft) {soft:7.1f}, gap {pred - soft:+6.1f}, plain {plain:7.1f}')
```

The effect points the same way on every seed and in both tabs. It is not the direction :numref:`sec_dqn` would lead you to predict. *Neither arm meaningfully over-promises*: no reading in either tab sits more than a few points above delivery. The single critic's promise lands near calibrated, never more than about thirty points below delivery and usually much closer, while the twin-critic minimum under-promises by about thirty to sixty points, and on every individual seed the minimum's gap exceeds the single critic's: the $\min$ is pessimistic by construction, and here the construction shows up in the measurement, a critic that promises less than it delivers, at policy quality the previous experiments already showed to be identical. Two honesty clauses belong to this table. Part of any under-promise has nothing to do with the minimum: a critic chasing a still-improving policy predicts yesterday's returns, so the clean signal is the *difference between the arms*, not the level of either. And the soft-versus-plain columns differ by under a dozen points here, but the sign and size of that difference are set by $\alpha$ and the entropy, and grading a soft critic against the plain return on a task with more entropy at stake would manufacture tens of points of phantom bias. So the second critic's dividend on Pendulum is not a better policy, it is a *calibrated pessimism* that costs nothing; on harder tasks, where an optimistic critic feeds the actor's climb and the climb feeds the optimism, that margin is what stands between this loop and :numref:`sec_dqn`'s self-confirming collapse, and :numref:`sec_offline` will need the same idea with the interaction removed entirely.

## Summary

SAC assembles three owned components and one new line of calculus. The objective is :numref:`sec_regularized`'s KL-regularized objective with a uniform reference, entropy charged at every step at exchange rate $\alpha$; soft evaluation adds one term to the one-step target, $-\alpha \log \pi$ at a fresh next action, certified in a cell to be the logsumexp backup of :numref:`sec_regularized` in sampled form; soft improvement is that section's proposition applied to $Q$, making "maximize $E[Q] + \alpha H$" and "project $e^{Q/\alpha}$ onto the policy family in KL" the same optimization up to a constant, with a five-line proof that exact per-state improvement raises the value everywhere. The gradient is :numref:`sec_deeprl`'s pathwise estimator through a critic that is differentiable in the action by construction. The one new mechanism is the tanh-squashed Gaussian: clipping has zero derivative exactly where a swing-up controller lives and a clipped Gaussian is not a density, so the boundedness moves into the distribution and the density gains the log-determinant of :eqref:`eq_tanh_logdet`, computed in the softplus form that is exact everywhere, since the epsilon-guarded alternative silently stops charging for saturation at $-13.8155$. A quadrature cell shows the corrected density integrates to one and the uncorrected one misses by half its mass. The machinery of :numref:`sec_dqn` returns off the shelf: replay with one column widened, target networks as Polyak averages with a 139-update half-life, twin critics whose minimum prices the actor's climb over the critic's errors, no target actor and no ratios anywhere. On Pendulum the agent clears $-200$ in under ten thousand environment steps where :numref:`sec_deeprl`'s REINFORCE spent $480{,}000$ and never arrived; the entropy settles within a nat of the $-\dim \mathcal{A}$ target that autotuning would enforce, at a noise cost the evaluations price near zero; and the twin critics leave the return untouched while converting the critic's promise from mild under-prediction to a deliberate, measured pessimism.

**What the experiments show, and what they do not.** The identity check, the saturation table, and the quadrature cell are deterministic numpy, print identical digits in both framework tabs, and are quotable to the digit. The training runs are three seeds per arm per tab with the two tabs initializing networks from different distributions; what is stable across all of them is the shape and the orderings: every seed clears $-200$ before ten thousand environment steps, deterministic evaluations land between about $-140$ and $-190$ with the stochastic policy within about ten points of its own deterministic twin, the entropy trace falls from its high start through a dip of roughly half a nat below zero and ends near zero, and the twin-critic minimum's calibration gap exceeds the single critic's on every individual seed. The individual crossing steps, evaluation digits, and gap sizes move from seed to seed and rerun to rerun, and the prose quotes them only as ranges; the return curves of the two arms are statistically indistinguishable at this budget, and no return difference is claimed. Single runs per configuration; the compute belongs to readers.

## Exercises

1. [short-code] *Autotune the temperature.* :cite:`Haarnoja.Zhou.Hartikainen.ea.2018`
   replaces the fixed $\alpha$ by a constraint $E\big[ H(\pi(\cdot \mid s_t)) \big] \geq \bar{H}$
   with $\bar{H} = -\dim \mathcal{A}$, and lets $\alpha$ be the Lagrange
   multiplier, minimizing $J(\alpha) = E_{a \sim \pi}\big[ -\alpha (\log \pi(a \mid s) + \bar{H}) \big]$
   by gradient steps on $\log \alpha$. Add this to `train_sac` (a few lines:
   one extra parameter, updated from the `logp` the loop already computes),
   starting from $\alpha_0 = 1$. Plot $\alpha_t$ and the entropy trace. Does
   the entropy converge to $\bar{H}$, and where does $\alpha$ end up relative
   to the $0.2$ we fixed?
1. [short-code] *Delete the log-determinant.* Remove the `logdet` term from
   `log_prob` and retrain one seed. Report three readings: the return curve,
   the entropy trace, and the quadrature check. Which of the three noticed,
   and what does the answer say about debugging a reinforcement learning run
   by its learning curve?
1. [conceptual] *Off-policy without ratios.* For each expectation in
   :eqref:`eq_sac_target` and :eqref:`eq_sac_actor`, state the distribution it
   is taken under and which part the replay buffer supplies. Where
   :numref:`sec_ppo` needed importance ratios, why does nothing here need one,
   and what distribution shift does the buffer still introduce that no ratio
   over actions can repair?
1. [short-code] *Entropy is not monotone in the noise.* Estimate the squashed
   policy's entropy $-E[\log \pi]$ at fixed $\mu = 0$ and
   $\sigma \in \{0.1, 0.3, 1, 3\}$ by sampling $u$ and averaging
   `log_prob` (do not integrate over the action: the density near the
   boundary defeats a uniform grid). Explain why the entropy peaks at an
   intermediate $\sigma$ and falls as mass piles onto the boundary, and what
   that implies for an agent trying to raise its entropy by inflating
   $\sigma$.
1. [short-code] *The limit $\alpha \to 0$ is DDPG.* Show numerically that at
   $\alpha = 0$ and $\sigma \to 0$ the actor's gradient in
   :eqref:`eq_sac_actor` equals the pathwise gradient of
   :eqref:`eq_score_vs_pathwise` evaluated at the deterministic action
   $c \tanh(\mu_\theta(s))$, which is DDPG's actor update
   :cite:`Lillicrap.Hunt.Pritzel.ea.2016`. Then list TD3's three repairs
   :cite:`Fujimoto.vanHoof.Meger.2018` and say which two SAC kept and why the
   stochastic policy makes the third unnecessary.
1. [short-code] *Port it.* Move the section's code to `LunarLander-v3` with
   `continuous=True`, whose action is two-dimensional. List every line that
   changes and every line that does not, in the spirit of :numref:`sec_dqn`'s
   port exercise; say which single hyperparameter you expect to raise most and
   why; check that `log_prob` still reduces correctly over the action axis;
   and explain what breaks, and where, if you instead port to `CartPole-v1`.
1. [extended] *Discrete soft actor-critic.* With a finite action set both
   expectations in :eqref:`eq_sac_target` and :eqref:`eq_sac_actor` can be
   computed exactly by summing over actions, and no reparameterization is
   needed. Write both losses for a softmax policy, state what replaces the
   tanh machinery, and implement it on CartPole; compare its sample efficiency
   against :numref:`sec_dqn`'s DQN at the same step budget.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.5]{.kicker}

Soft Actor-Critic<br>
**the objective from 15.3 · the gradient from 14.7 · the critics from 15.4 · one new line of calculus**
:::
:::

::: {.slide title="The Objective, Already Proved"}
:numref:`sec_regularized`'s KL penalty, uniform reference, charged
per step; $\alpha$ for $\beta$, the field's convention:

$$J(\pi) = E_{\pi}\Big[ \sum_t \gamma^t \big( r_t
+ \alpha\, H(\pi(\cdot \mid s_t)) \big) \Big]$$

. . .

Not a bonus bolted onto training: part of the **objective**, so
the optimum itself is stochastic, by design. Continuous actions:
differential entropy, can be negative. SAC = the off-policy
actor-critic of this objective
:cite:`Haarnoja.Zhou.Abbeel.ea.2018`.
:::

::: {.slide title="Soft Evaluation: One New Term"}
$$y = r + \gamma\, \big(1 - \mathbf{1}(s' \textrm{ terminal})\big)
\Big( \min_{j=1,2} Q_{w_j^-}(s', \tilde{a}')
- \alpha \log \pi_\theta(\tilde{a}' \mid s') \Big)$$

- $s, a, r, s'$ from the buffer; $\tilde{a}'$ **fresh from the
  live policy**: the expectation is under $\pi_\theta$
- PPO's entropy bonus lived in the actor loss; this one lives in
  the *critic target*: the critic values entropy collected later

. . .

At the tilted optimum the bracket is 15.3's logsumexp, exactly:

@!sac-soft-evaluation-one-new-term-in-the-target
:::

::: {.slide title="Improvement Is the Proposition"}
15.3's proof line, read with $r \to Q(s, \cdot)$, uniform
reference:

$$E_{\pi}[Q] + \alpha H(\pi) \;=\; \alpha \log Z
- \alpha\, D_{\textrm{KL}}\big(\pi \,\Vert\, e^{Q/\alpha}/Z\big)$$

Maximizing the left and projecting onto the family in KL are the
**same optimization**; they differ by $\alpha \log Z$, which does
not depend on $\theta$.

. . .

The gradient through $\tilde{a}_\theta(s, z) = c \tanh(\mu_\theta
+ \sigma_\theta z)$ is 14.7's *pathwise* estimator, on a critic
differentiable in $a$ by construction.

**Proposition** (soft policy improvement): exact per-state
maximization raises $V$ everywhere; five lines from 15.3.
:::

::: {.slide title="A Policy That Fits in a Box"}
14.7 let the environment clip the torque. The score estimator
never differentiated through the action; the pathwise one does:

- outside the box the clip's derivative is **zero**: no signal at
  the boundary, where swing-up lives
- a clipped Gaussian is not a density: atoms at $\pm 2$, and
  $\alpha \log \pi$ is undefined on an atom

. . .

$$\log \pi(a \mid s) = \sum_i \Big[ \log \mathcal{N}(u_i; \mu_i,
\sigma_i) - \log\big(1 - \tanh^2 u_i\big) - \log c \Big]$$

@sac-the-change-of-variables
:::

::: {.slide title="The Epsilon That Hides"}
$1 - \tanh^2 u = 4 e^{-2u}/(1 + e^{-2u})^2$, so
$\log(1 - \tanh^2 u) = 2(\log 2 - u - \operatorname{softplus}(-2u))$,
exact. What the guard `+ 1e-6` does instead:

@!sac-the-stable-form-and-what-the-epsilon-hides-1

. . .

$-13.8155$ forever: the density stops charging for saturation and
no curve looks wrong. And the whole change of variables is
checkable by quadrature, at zero training cost:

@!sac-the-stable-form-and-what-the-epsilon-hides-2
:::

::: {.slide title="The Machinery, Off the Shelf"}
- **twin critics, min**: the actor climbs the critic, a maximizer
  over its errors (15.4's argument, argmax $\to$ gradient
  ascent); the min of two independent critics is the cheapest
  pessimism :cite:`Fujimoto.vanHoof.Meger.2018`
- **Polyak targets**: $w^- \leftarrow \tau w + (1-\tau) w^-$,
  half-life $\ln 2 / \tau \approx 139$ updates; drifts, never
  jumps
- **no target actor**: $\tilde{a}'$ from the live policy; the
  stochastic policy smooths its own targets
- **no ratios**: the target never mentions the collector, the
  actor re-samples; the buffer shifts only the *state*
  distribution (:numref:`sec_offline`)
- `ReplayBufferC`: one column widened to float vectors
:::

::: {.slide title="One Update"}
@sac-one-update-1

. . .

Pendulum has **no terminal state**: `term` is identically zero,
the bootstrap is always taken; storing `done` would teach the
agent that the world ends at step 200.
:::

::: {.slide title="Under Ten Thousand Steps"}
@!sac-sample-efficiency-on-the-axis-that-bills

. . .

14.7's REINFORCE spent $480{,}000$ steps on this task and never
reached $-200$. Pathwise gradient $\times$ replay: more signal
per sample, hundreds of updates per sample. The two arms are
indistinguishable on this axis.
:::

::: {.slide title="The Entropy the Policy Keeps"}
@!sac-the-entropy-the-policy-keeps-1

. . .

- spent fast during the climb, overshooting below zero, partly
  bought back once the task is mastered; ends near zero:
  stochastic **at convergence**
- within about a nat of autotuning's target
  $\bar{H} = -\dim \mathcal{A} = -1$
  :cite:`Haarnoja.Zhou.Hartikainen.ea.2018`; $\alpha$ is an
  exchange rate in reward per nat, not a learning rate
- deterministic and stochastic evaluations agree within about
  ten points: the noise was kept where it is cheap
:::

::: {.slide title="Honest Promises"}
@!sac-honest-promises-predicted-against-delivered

. . .

Neither arm meaningfully over-promises. The single critic is
near calibrated; the min under-promises by thirty to sixty
points, on every seed, at identical policy quality: **pessimism,
measured, for free**. On harder tasks that margin is what stands
between this loop and 15.4's self-confirming collapse.
:::
