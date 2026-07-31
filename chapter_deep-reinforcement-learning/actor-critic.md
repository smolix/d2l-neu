# Actor-Critic Methods and Multi-Step Returns
:label:`sec_actorcritic`

:numref:`sec_deeprl` estimated advantages from complete sampled returns, so an update could be formed only after an episode ended. An actor--critic method instead uses a learned value function to bootstrap from incomplete trajectories :cite:`Barto.Sutton.Anderson.1983,Konda.Tsitsiklis.2000`. The *actor* is the policy $\pi_\theta$, and the *critic* estimates its value $V^{\pi_\theta}$.

Bootstrapping reduces the amount of future randomness in each target but introduces error from the critic. This section derives the one-step temporal-difference target, extends it to $n$-step and $\lambda$-returns, and implements generalized advantage estimation (GAE). Experiments on CartPole compare these estimators through their gradient error, training stability, and sensitivity to critic fitting.

```{.python .input #actor-critic-actor-critic-and-the-credit-assignment-dial-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
```

```{.python .input #actor-critic-actor-critic-and-the-credit-assignment-dial-1}
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

We retain the CartPole environment, discount $0.99$, and one-hidden-layer `ActorCritic.mlp` model from :numref:`sec_deeprl`. The new hyperparameter `critic_steps` specifies the number of critic regression passes per batch. Unless stated otherwise, we use twenty passes and recompute the target before each pass.

```{.python .input #actor-critic-actor-critic-and-the-credit-assignment-dial-2}
%%tab pytorch, jax
gamma, num_updates, batch_episodes = 0.99, 100, 8
num_seeds, critic_steps, grad_clip = 3, 20, 0.5
if tab.selected('pytorch'):
    def cartpole_agent(seed):
        torch.manual_seed(seed)
        return d2l.ActorCritic.mlp(4, 2)
if tab.selected('jax'):
    def cartpole_agent(seed):
        return d2l.ActorCritic.mlp(4, 2, rngs=nnx.Rngs(seed))
```

## Bootstrapping the Reward-to-Go

### The TD Error

Write out one step of what the reward-to-go contains,

$$\hat{G}_t = r_t + \gamma\, \hat{G}_{t+1},$$

and recall what the learned baseline of :numref:`sec_baselines` is trained on: $\hat{V}(s_{t+1})$ regresses toward exactly the quantity that $\hat{G}_{t+1}$ samples. So stop waiting for the sample and substitute the prediction, $\hat{G}_t \approx r_t + \gamma \hat{V}(s_{t+1})$. The weight on the score at step $t$ was $\hat{G}_t - \hat{V}(s_t)$; after the substitution it becomes

$$\delta_t = r_t + \gamma\, \hat{V}(s_{t+1}) - \hat{V}(s_t),$$
:eqlabel:`eq_td_error_v`

with the convention $\hat{V}(s_{t+1}) = 0$ when $s_{t+1}$ is terminal. This is the temporal-difference error of :eqref:`eq_td_error`, adapted to policy evaluation. Q-learning bootstraps on the greedy continuation $\max_{a'} \hat{Q}(s', a')$ because it estimates $Q^*$; here the bootstrap uses the current policy's continuation value $\hat{V}(s_{t+1})$. As in Q-learning, the terminal mask is determined by `terminated`, not `truncated`, because a time limit stops the observation without making the continuation value zero (:numref:`sec_mdp`). The `bootstrap` function below returns a numpy array, so the target contains no autograd history and requires no explicit `detach`.

```{.python .input #actor-critic-the-td-error}
%%tab pytorch, jax
@d2l.add_to_class(d2l.Batch)  #@save
def td_target(self, bootstrap, gamma):
    """r_t + gamma (1 - terminated) V(s'), by a numpy bootstrap."""
    return self.rew + gamma * (1 - self.term) * bootstrap(self.next_obs)
```

As in :eqref:`eq_rtg`, we omit the factor $\gamma^t$ from the actor update. The consequences of this convention were discussed in :numref:`sec_baselines`.

### The TD Error as an Advantage Estimate

The TD error is not merely cheaper than the Monte Carlo weight; it estimates the same thing. Suppose for a moment the critic were exact, $\hat{V} = V^\pi$. Averaging :eqref:`eq_td_error_v` over the next state,

$$E\big[ \delta_t \mid s_t, a_t \big] = r(s_t, a_t) + \gamma \sum_{s'} P(s' \mid s_t, a_t)\, V^{\pi}(s') - V^{\pi}(s_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t),$$

where the second equality is :eqref:`eq_dynamic_programming_q` written with $V^\pi$. The right-hand side is the advantage :eqref:`eq_advantage` of the action taken, the exact quantity the learned baseline of :numref:`sec_baselines` estimated with $\hat{G}_t - \hat{V}(s_t)$. So a single transition, one reward plus two critic evaluations, yields an unbiased one-sample estimate of the advantage, where the Monte Carlo estimate needed the entire remaining trajectory.

During training $\hat{V}$ is generally not equal to $V^\pi$. Consequently, $\delta_t$ is a biased estimate of the advantage, and the resulting policy update is not an unbiased gradient estimator of $J(\theta)$. The baseline lemma of :numref:`sec_baselines` does not apply to the bootstrap term: unlike a state-dependent baseline at time $t$, $\hat{V}(s_{t+1})$ depends on the sampled action through the next state. Exercise 1 identifies the corresponding step in the proof. Bootstrapping therefore exchanges some variance for bias.

### The Critic as Sampled Policy Evaluation

The critic update is a sampled form of policy evaluation. In :numref:`sec_valueiter`, policy evaluation applied a Bellman expectation operator whose fixed point is $V^\pi$. Regressing $\hat{V}(s_t)$ toward $r_t + \gamma \hat{V}(s_{t+1})$ replaces that expectation by one sampled transition, just as Q-learning replaced the Bellman optimality expectation by a sample in :numref:`sec_qlearning`. Actor-critic can therefore be viewed as approximate generalized policy iteration: the critic estimates $V^{\pi_\theta}$, while the actor improves the policy using that estimate. Because the regression target itself contains $\hat{V}$, the critic update is a semi-gradient toward a fixed point rather than gradient descent on a fixed supervised-learning objective.

### The Bias-Variance Trade-off

The two advantage estimates have complementary errors. The Monte Carlo estimate is unbiased but includes randomness from every remaining transition, so its variance tends to grow with the horizon. The one-step TD error depends on only one transition, but it inherits bias from the critic. Targets that bootstrap after $n$ steps interpolate between these cases. In the synthetic chain in :numref:`fig_rl_td_mc_spectrum`, the critic becomes more accurate near termination; increasing $n$ therefore reduces bias and increases variance, with the smallest mean squared error at an intermediate depth. This pattern depends on the example and is not a general monotonicity result.

![The credit-assignment dial. Left: backup diagrams of depth one, two, $n$, and to termination; the green node is where the estimate stops sampling and starts trusting the critic, and the strip below shows the $\lambda$-return's weights $(1-\lambda)\lambda^{n-1}$ over the $n$-step targets at $\lambda = 0.9$. Right: the family measured on a synthetic ten-state chain, deterministic step right with only the final transition paying $1$, per-step reward noise of standard deviation $0.15$, and $\gamma = 0.97$, under a critic whose error tapers toward termination as $0.5\,(1 - s/9)^2$, the shape value learning actually produces; over $20{,}000$ rollouts of the depth-$n$ target from the start state, bias falls with depth, variance grows, and the mean squared error is smallest at $n = 4$. A critic equally wrong everywhere would make the error monotone in $n$; the interior optimum exists because deeper targets bootstrap from better-known states.](../img/mdl-rl-td-mc-spectrum.svg)
:label:`fig_rl_td_mc_spectrum`

## The Actor-Critic Algorithm

### The Actor and Critic Updates

Both learners now update from the same scalar. Writing $\hat{V}_w$ for the critic's parameters, the updates after the transition $(s_t, a_t, r_t, s_{t+1})$ are

$$w \leftarrow w + \alpha_w\, \delta_t\, \nabla_w \hat{V}_w(s_t), \qquad \theta \leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$
:eqlabel:`eq_actor_critic`

The critic's update is the sampled policy evaluation of the previous section, a semi-gradient step toward the one-step target; the actor's is the policy gradient of :numref:`sec_baselines` with $\delta_t$ as the advantage weight. One number does both jobs: $\delta_t$ tells the critic how wrong its prediction was, and it tells the actor whether the action just taken was better or worse than the state's average. And because $\delta_t$ needs only a single transition, :eqref:`eq_actor_critic` can be applied at every step as the agent acts, the way Q-learning updates its table; nothing must terminate before learning starts, so the method extends to tasks that never end. REINFORCE, with or without a baseline, cannot say the same.

![The actor-critic loop. One transition produces one temporal-difference error, and that single number does both jobs: it tells the critic how wrong its prediction was, and it tells the actor whether the action it just took was better or worse than the state's average. Nothing waits for the episode to end.](../img/mdl-rl-actor-critic-loop.svg)
:label:`fig_rl_actor_critic_loop`

### Two Timescales

Two learning rates appear because the two learners have different jobs. The critic chases a moving target twice over: every actor update changes the policy, which changes the values the critic is trying to predict, and every critic update moves the bootstrapped targets themselves. The actor then judges its actions with whatever the critic currently believes. If the actor moves much faster than the critic, the advantages are judged against stale values and the updates degrade. The practical rule of thumb is to let the critic learn faster than the actor, a *two-timescale* intuition; the theory that makes it precise assumes separated, decaying step-size schedules under function-approximation and noise conditions our setup does not meet. Our implementation keeps one nominal learning rate for both heads and buys the separation with update count instead: `critic_steps` regression passes per batch, a heuristic realization of the intuition rather than the theorem's schedule.

### A Batched Actor-Critic and A2C

What we implement below is not the fully online :eqref:`eq_actor_critic`, and the difference deserves a careful name. The code collects a batch of eight episodes from one environment, gives the critic its `critic_steps` regression passes on the batch's one-step targets, and then takes a single actor step weighted by the TD errors of the freshly updated critic: call it a *batched on-policy actor-critic*. Its production relative is A2C, the advantage actor-critic, which conventionally denotes the synchronous multi-actor form: many environments stepped in lockstep with $n$-step rollouts, itself the synchronous descendant of an asynchronous original :cite:`Mnih.Badia.Mirza.ea.2016`. Our single-environment, whole-episode loop keeps A2C's estimator and drops its vectorization. Exercise 5 takes up the fully online form, one update per transition with no batch at all, and what goes wrong there is a preview of :numref:`sec_dqn`.

Two pieces of local machinery first. The actor's step is `d2l.policy_step` with one addition that the bootstrapped weight will shortly justify: the gradient norm is capped at the customary $0.5$, and the function reports the norm it measured *before* the cap, because that number is about to carry a lesson.

```{.python .input #actor-critic-a2c-by-name-1}
%%tab pytorch
def policy_step_clip(ac, batch, w, clip=grad_clip):
    """d2l.policy_step with the actor's gradient norm capped at `clip`;
    returns the norm measured before the cap."""
    obs, act = torch.as_tensor(batch.obs), torch.as_tensor(batch.act)
    loss = -(torch.as_tensor(w) * ac.log_prob(obs, act)).mean()
    ac.opt_pi.zero_grad()
    loss.backward()
    gnorm = nn.utils.clip_grad_norm_(ac.policy.parameters(), clip)
    ac.opt_pi.step()
    return float(gnorm)
```

```{.python .input #actor-critic-a2c-by-name-1}
%%tab jax
def policy_step_clip(ac, batch, w, clip=grad_clip):
    """d2l.policy_step with the actor's gradient norm capped at `clip`;
    returns the norm measured before the cap. `_pad` and `_actor_pass`,
    defined with the critic's compilation story two cells below, are
    resolved at the first call."""
    size = 1 << max(6, (len(w) - 1).bit_length())
    gnorm = _actor_pass(ac.policy, ac.opt_pi, _pad(batch.obs, size),
                        _pad(batch.act, size), _pad(w, size),
                        jnp.float32(len(w)), jnp.float32(clip))
    return float(gnorm)
```

The second piece is per-framework speed, not algorithm, and follows the compilation rule of :numref:`sec_compilation`: compile what has a fixed shape and runs hot, leave ragged shapes eager. The acting forward runs a few hundred thousand times at one fixed shape, so the jax tab compiles and caches it exactly as :numref:`sec_deeprl` did; the pytorch tab's eager dispatch is already cheap at this scale and needs nothing.

```{.python .input #actor-critic-a2c-by-name-2}
%%tab jax
_act_probs = nnx.jit(lambda net, obs: jax.nn.softmax(net(obs), -1))

def _probs(ac, obs):   # the fixed-shape acting forward, compiled and
    if not hasattr(ac, '_fwd'):        # cached as in :numref:`sec_deeprl`
        ac._fwd = nnx.cached_partial(_act_probs, ac.policy)
    return np.asarray(ac._fwd(jnp.asarray(obs)))

@d2l.add_to_class(d2l.ActorCritic)
def act(self, obs, rng):
    p = _probs(self, obs)
    return int(rng.choice(len(p), p=p))

@d2l.add_to_class(d2l.ActorCritic)
def act_greedy(self, obs, rng=None):
    return int(_probs(self, obs).argmax())
```

The critic's regression and the actor's clipped step are this section's hot loops, together tens of thousands of passes per run, each on a batch of a few fixed lengths. Padding every batch to a power-of-two length makes the shape one of a handful and lets the jitted passes and the jitted value read that computes the critic's target compile once per bucket rather than once per batch; a mask, or zero-padded weights, keeps the padded entries out of each loss.

```{.python .input #actor-critic-a2c-by-name-3}
%%tab pytorch
# Eager per-pass cost is about a millisecond in this tab; the library
# helper needs no compilation story here, unlike its jax sibling.
fit_value = d2l.fit_value
```

```{.python .input #actor-critic-a2c-by-name-3}
%%tab jax
def _pad(x, size):
    return jnp.asarray(np.pad(x, ((0, size - len(x)),) + ((0, 0),)
                              * (x.ndim - 1)))

_value_fwd = nnx.jit(lambda net, obs: net(obs).squeeze(-1))

@d2l.add_to_class(d2l.ActorCritic)
def value_np(self, obs):   # batched reads only, padded to the bucket size
    size = 1 << max(6, (len(obs) - 1).bit_length())
    return np.asarray(_value_fwd(self.value, _pad(obs, size)))[:len(obs)]

@nnx.jit
def _critic_pass(value, opt, obs, target, mask):
    loss, grads = nnx.value_and_grad(lambda v: (mask * (
        v(obs).squeeze(-1) - target) ** 2).sum() / mask.sum())(value)
    opt.update(value, grads)
    return loss

@nnx.jit
def _actor_pass(policy, opt, obs, act, adv, n_real, clip):
    def loss_fn(p):   # zero-padded adv silences the padded rows exactly
        logp = jax.nn.log_softmax(p(obs), -1)[jnp.arange(obs.shape[0]), act]
        return -(adv * logp).sum() / n_real
    grads = nnx.grad(loss_fn)(policy)
    gnorm = optax.global_norm(grads)
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, clip / gnorm), grads)
    opt.update(policy, grads)
    return gnorm

def fit_value(ac, obs, target, num_steps=1):
    """d2l.fit_value, padded to a power-of-two length: the jitted pass
    compiles once per size bucket (:numref:`sec_compilation`)."""
    size = 1 << max(6, (len(target) - 1).bit_length())
    mask = jnp.asarray((np.arange(size) < len(target)).astype(np.float32))
    for _ in range(num_steps):
        loss = _critic_pass(ac.value, ac.opt_v, _pad(obs, size),
                            _pad(target, size), mask)
    return float(loss)
```

Now the two training functions, side by side. The Monte Carlo reference is `train_reinforce` of :numref:`sec_deeprl` with two mechanical changes: the agent arrives from the caller, following the pattern of :numref:`sec_qlearning`'s caller-owned tables, so that the last cell of this section can audit the trained agents; and the actor's step goes through the clipped variant and yields its gradient norm, a diagnostic returned as data.

```{.python .input #actor-critic-a2c-by-name-4}
%%tab pytorch, jax
def train_reinforce(seed, ac, num_updates=num_updates):
    """sec_deeprl's learned-baseline REINFORCE; the agent is caller-owned
    and the actor step reports its pre-clip gradient norm."""
    rng, env = np.random.default_rng(seed), gym.make('CartPole-v1')
    env.reset(seed=seed)
    for _ in range(num_updates):
        batch = d2l.rollout(env, ac.act, batch_episodes, rng)
        G = batch.reward_to_go(gamma)
        w = d2l.normalize(G - ac.value_np(batch.obs))
        gnorm = policy_step_clip(ac, batch, w)
        fit_value(ac, batch.obs, G)
        yield float(batch.episode_returns().mean()), gnorm
```

Actor-critic is the same loop with the tail replaced, and the diff is four lines. The critic regresses on the bootstrapped one-step target instead of the reward-to-go, and takes its `critic_steps` passes *first*, so that the actor judges its actions with the freshest available critic. Those twenty passes deserve a name: each recomputes its target from the newest critic, so the regression surface moves under the regression. This is an aggressive fitted-TD teaching loop tracking a moving fixed point, not fixed-target regression; computing the targets once and fitting them would be the fixed-target alternative, and the choice between the two is exactly the choice :numref:`sec_dqn`'s target network will make the other way. The weight handed to the actor is the TD error of :eqref:`eq_td_error_v`, normalized per batch exactly as before. Nothing else changes.

```{.python .input #actor-critic-a2c-by-name-5}
%%tab pytorch, jax
def train_ac(seed, ac, num_updates=num_updates):
    """The same loop with the sampled tail replaced by the bootstrap."""
    rng, env = np.random.default_rng(seed), gym.make('CartPole-v1')
    env.reset(seed=seed)
    for _ in range(num_updates):
        batch = d2l.rollout(env, ac.act, batch_episodes, rng)
        for _ in range(critic_steps):   # fresh target, one pass, repeat
            fit_value(ac, batch.obs, batch.td_target(ac.value_np, gamma))
        delta = batch.td_target(ac.value_np, gamma) - ac.value_np(batch.obs)
        gnorm = policy_step_clip(ac, batch, d2l.normalize(delta))
        yield float(batch.episode_returns().mean()), gnorm
```

Three seeds each, with the agents kept. The reference arm first:

```{.python .input #actor-critic-a2c-by-name-6}
%%tab pytorch, jax
agents = {name: [cartpole_agent(s) for s in range(num_seeds)]
          for name in ('REINFORCE + baseline', 'actor-critic')}
runs = {'REINFORCE + baseline': np.array(
    [list(train_reinforce(s, agents['REINFORCE + baseline'][s]))
     for s in range(num_seeds)])}
```

Then the bootstrapped arm, and the comparison:

```{.python .input #actor-critic-a2c-by-name-9}
%%tab pytorch, jax
runs['actor-critic'] = np.array(
    [list(train_ac(s, agents['actor-critic'][s]))
     for s in range(num_seeds)])
d2l.plot_curves({name: r[:, :, 0] for name, r in runs.items()},
                xlabel='update', ylabel='mean return of the batch',
                reference=500)
```

The two curves are the trade of :numref:`fig_rl_td_mc_spectrum` drawn by the algorithms themselves. Actor-critic starts slower: for its first stretch the critic is still wrong, the TD errors point in poorly chosen directions, and the biased updates buy little, while the Monte Carlo weight is correct on average from the first batch. Then the curves stop differing in level, both reach the neighborhood of the ceiling, and start differing in *texture*. A batch mean of $500$ means every one of the batch's eight episodes ran the full 500 steps, a perfect batch; $500$ is CartPole's structural ceiling, so it can be tied but never beaten, and the informative comparison is not who is higher but who, having arrived, stays. The claim is checkable in four lines:

```{.python .input #actor-critic-a2c-by-name-7}
%%tab pytorch, jax
def longest_streak(hits):
    """The longest run of consecutive True entries."""
    best = run = 0
    for h in hits:
        run = run + 1 if h else 0
        best = max(best, run)
    return best

for name, r in runs.items():
    print(f'{name:>20}: longest stretch of perfect batches, per seed: '
          f'{[longest_streak(row == 500) for row in r[:, :, 0]]}')
```

What separates the methods here is not the height of the plateau but whether a run ever comes to rest on it: in every tab the longest stretch of consecutive perfect batches belongs to the bootstrapped side. Not a higher plateau, a stiller one. Read the streak as a stability *visualization* of these particular runs, not a standard estimator metric: its length entangles policy quality, batch size and environment stochasticity with the estimator's variance. The mechanism it visualizes is real, though: the Monte Carlo curve keeps dipping and recovering to the end, because its weight still carries the sampled noise of every remaining coin flip of the trajectory, and a run at the ceiling is only ever one noisy batch away from a stumble; the bootstrapped weight's noise left with the tail. The early cost is the bias while the critic trains, the late payoff is the stillness, and both halves of the bargain are visible in one figure.

The training functions also brought back a second number, and it justifies the clip. Both methods scale their weights to unit variance before the update, so it is tempting to assume both take steps of the same size:

```{.python .input #actor-critic-a2c-by-name-8}
%%tab pytorch, jax
for name, r in runs.items():
    g = r[:, :, 1].ravel()
    print(f'{name:>20}: median pre-clip gradient norm {np.median(g):5.2f}, '
          f'clip binds on {(g > grad_clip).mean():4.0%} of updates')
```

Measured on these runs the actor-critic gradients are an order of magnitude larger, and the clip binds on about a third of the actor-critic updates and on none of the Monte Carlo ones. The reason is the baseline argument of :numref:`sec_baselines` read backwards: whatever part of the weight depends on the state alone contributes nothing in expectation, and the Monte Carlo weight spends much of its variance there, while $\delta_t$ is almost all action-dependent. That is a statement about where the weight's variance lives, a descriptive diagnostic; a larger, or smaller, gradient norm is not by itself evidence of a better signal. What it does establish is practical: a better advantage estimate is also a bigger step at the same learning rate, and how to take big steps safely is what :numref:`sec_ppo` is about.

### The Critic's Moving Target and Data Freshness

The two-timescale paragraph promised trouble, the critic chasing a doubly moving target, yet the runs above show none. The reason is freshness, and the claim should be made at its true size. Every batch is drawn from the current policy at exactly the states the current policy visits, so wherever the critic is wrong in a way that matters, the very next batch delivers the transitions that expose it; this is :numref:`sec_qlearning`'s self-correcting loop, running on $V$ instead of $Q$. What freshness buys is *reduced distributional mismatch* between where the critic errs and where it is trained, not a stability guarantee: nonlinear temporal-difference regression can diverge even on-policy, and the classical convergence results cover linear critics under two-timescale step-size conditions our loop does not implement :cite:`Tsitsiklis.VanRoy.1997`. In these runs, on this task, the mismatch reduction is evidently enough. The same loop also lets us watch the trade itself in vivo. Rerun `train_ac` once more, and before each actor step compute both candidate weights for the same batch, the TD error and the Monte Carlo advantage: how much the two agree, and how much each varies.

```{.python .input #actor-critic-why-the-critic-s-moving-target-is-survivable-here-and-fatal-later}
%%tab pytorch, jax
rng, env = np.random.default_rng(0), gym.make('CartPole-v1')
ac = cartpole_agent(0)
env.reset(seed=0)
trace = []
for _ in range(num_updates):
    b = d2l.rollout(env, ac.act, batch_episodes, rng)
    for _ in range(critic_steps):
        fit_value(ac, b.obs, b.td_target(ac.value_np, gamma))
    V = ac.value_np(b.obs)
    delta = b.td_target(ac.value_np, gamma) - V
    A_mc = b.reward_to_go(gamma) - V
    trace.append((np.corrcoef(delta, A_mc)[0, 1], delta.var(), A_mc.var()))
    policy_step_clip(ac, b, d2l.normalize(delta))
trace = np.array(trace)
print(f'corr(TD error, MC advantage): mean {trace[:10, 0].mean():.2f} over '
      f'the first ten updates, {trace[-50:, 0].mean():.2f} over the last 50')
d2l.plot_curves({'corr(TD error, MC advantage)': trace[:, 0]},
                xlabel='update', ylabel='correlation', smooth=10)
```

When episodes last a few dozen steps the two candidate weights visibly overlap, a correlation around $0.3$; by the time the policy holds the pole for hundreds of steps the correlation has fallen to zero. The identity behind the fall is worth unrolling: take :eqref:`eq_td_error_v` one step and the Monte Carlo advantage contains the TD error, plus a discounted copy of itself one step later, $\hat{G}_t - \hat{V}(s_t) = \delta_t + \gamma \big( \hat{G}_{t+1} - \hat{V}(s_{t+1}) \big)$. At a long horizon the Monte Carlo weight is therefore one part signal shared with $\delta_t$ and hundreds of parts accumulated tail, and the share the two weights have in common shrinks toward nothing as episodes lengthen, which is consistent with the Monte Carlo weight dissolving into tail noise rather than the TD error drifting away. Read the curve as a descriptive diagnostic of this run, not a certificate of critic health: a correlation confounds the changing policy, the changing state distribution, and both weights' scales, and certifying the critic itself would mean freezing the policy and grading its predictions against a reference value, the shape of measurement the dial probe below performs on the gradient estimators. The second half of the measurement says the same thing in absolute terms:

```{.python .input #actor-critic-why-the-critic-s-moving-target-is-survivable-here-and-fatal-later-1}
%%tab pytorch, jax
print(f'median variance, MC advantage over TD error: '
      f'{np.median(trace[:, 2] / trace[:, 1]):.0f}x')
d2l.plot_curves({'TD error': trace[:, 1], 'MC advantage': trace[:, 2]},
                xlabel='update', ylabel='variance of the weight', smooth=10)
```

As the policy improves and episodes stretch toward 500 steps, the Monte Carlo weight's variance grows with the horizon, every extra step of tail adding its noise, from tens of squared units early in training to several hundred at the ceiling; the TD error's variance is the flat line pinned at the bottom of the same axes, one reward and one transition regardless of what follows, and it falls further as the critic sharpens, which is what pushes the printed median ratio past a hundredfold. The actor wants the advantage, and at a 500-step horizon the Monte Carlo estimate of it is almost entirely tail; averaging over a batch of eight episodes is what kept REINFORCE trainable at all.

Here the critic is trained on fresh on-policy data, and its target is recomputed after each update. When a value function is instead trained on replayed off-policy data while retaining a bootstrap target, the same semi-gradient update can diverge. :numref:`sec_dqn` addresses this problem by evaluating the bootstrap with a second network whose parameters are held fixed for several updates. This *target network* reduces the rate at which the regression target moves.

## n-Step Returns, Lambda-Returns and GAE

### n-Step Returns

The one-step target and the Monte Carlo tail are the ends of a family. Trust sampling for $n$ steps before handing over to the critic,

$$\hat{G}^{(n)}_t = r_t + \gamma\, r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n\, \hat{V}(s_{t+n}),$$
:eqlabel:`eq_nstep_return`

and the corresponding advantage estimate is $\hat{G}^{(n)}_t - \hat{V}(s_t)$. At $n = 1$ this is $\delta_t$; run $n$ to the end of the episode and it is the Monte Carlo advantage. In between, each extra step of reality typically buys bias down and costs variance up: the critic's error arrives discounted by $\gamma^n$ and, for a critic whose error shrinks toward termination, is evaluated $n$ steps closer to states it knows better, while every estimate picks up one more sampled reward and transition. That is exactly the falling and rising pair measured, under precisely such a critic, in :numref:`fig_rl_td_mc_spectrum`.

### The Lambda-Return and the Telescoping Identity

Rather than pick one depth, average them all, with geometric weights that make the average computable:

$$\hat{A}^{\textrm{GAE}}_t = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \big( \hat{G}^{(n)}_t - \hat{V}(s_t) \big), \qquad \lambda \in [0, 1].$$
:eqlabel:`eq_lambda_return`

The weights $(1-\lambda)\lambda^{n-1}$ sum to one, $\lambda = 0$ puts everything on the one-step estimate, and $\lambda \to 1$ puts everything on the deepest one. Written this way the estimator looks like it needs every depth computed separately. It does not, and the identity that says so is why $\lambda$ survived into deployed implementations :cite:`Schulman.Moritz.Levine.ea.2016`.

**Proposition.** The mixture collapses to a discounted sum of TD errors:

$$
\hat{A}^{\textrm{GAE}}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}.
$$
:eqlabel:`eq_gae_deltas`

**Proof.** Each depth telescopes into TD errors: $\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n \hat{V}(s_{t+n}) - \hat{V}(s_t) = \hat{G}^{(n)}_t - \hat{V}(s_t)$, the interior $\hat{V}$ terms cancelling in pairs. Substitute this into :eqref:`eq_lambda_return` and swap the two sums: the pair $(n, l)$ with $l < n$ contributes $(1-\lambda)\lambda^{n-1} \gamma^l \delta_{t+l}$, so $\hat{A}^{\textrm{GAE}}_t = \sum_{l \ge 0} \gamma^l \delta_{t+l}\, (1-\lambda) \sum_{n \ge l+1} \lambda^{n-1}$. The inner geometric sum is $\lambda^l$. $\blacksquare$

On an episodic task the sums truncate at termination, where $\hat{V} = 0$ and no rewards follow, so every $\delta$ past the end is zero and the infinite form above is also the finite one.

### GAE as a Backward Scan

Equation :eqref:`eq_gae_deltas` gives a direct algorithm for *generalized advantage estimation* (GAE): compute the TD errors and accumulate them backward with factor $\gamma\lambda$, restarting at episode boundaries. The `Batch.backward_scan` method from :numref:`sec_baselines` already implements this recurrence; only its input changes.

```{.python .input #actor-critic-gae-in-one-old-function-1}
%%tab pytorch, jax
@d2l.add_to_class(d2l.Batch)  #@save
def gae(self, value_fn, gamma, lam):
    """GAE(gamma, lam): the reward-to-go scan, run on the TD errors."""
    delta = self.td_target(value_fn, gamma) - value_fn(self.obs)
    return self.backward_scan(delta, gamma * lam)
```

The endpoints are theorems, so we assert rather than assume them: at $\lambda = 0$ the scan returns the TD errors themselves, and at $\lambda = 1$, on episodes that terminate, the telescoping runs to termination and returns the reward-to-go minus the baseline, the Monte Carlo advantage of :numref:`sec_baselines`.

```{.python .input #actor-critic-gae-in-one-old-function-2}
%%tab pytorch, jax
env = gym.make('CartPole-v1')
env.reset(seed=0)
fresh = cartpole_agent(0)
b = d2l.rollout(env, fresh.act, 8, np.random.default_rng(0))
V = fresh.value_np(b.obs)
delta = b.td_target(fresh.value_np, gamma) - V
assert np.allclose(b.gae(fresh.value_np, gamma, 0.0), delta)
assert np.allclose(b.gae(fresh.value_np, gamma, 1.0),
                   b.reward_to_go(gamma) - V, atol=1e-4)
print('lambda = 0 is the TD error; lambda = 1 is the Monte Carlo advantage')
```

One boundary clause: the $\lambda = 1$ identity holds on episodes that *terminate*. On an episode cut off by a time limit, `gae` correctly bootstraps the missing future through the unmasked $\hat{V}$ in its last TD error, while `reward_to_go` silently pretends the future is empty; the untrained probe above ends every episode by falling, which is why the assertion is exact. The estimator handles the `terminated`-versus-`truncated` distinction of :numref:`sec_mdp` correctly because `td_target` does.

### Eligibility Traces

The same construction predates deep reinforcement learning. In the online form of TD($\lambda$), each parameter carries an exponentially decaying eligibility trace, allowing a new TD error to update earlier states with weight $(\gamma\lambda)^l$ :cite:`Sutton.1988,Sutton.Barto.2018`. The backward scan above is convenient for batches; eligibility traces remain useful when an agent must learn from a stream under strict latency or memory constraints :cite:`Elsayed.Vasan.Mahmood.2024`.

## Measuring the Bias-Variance Trade-off

The dial's two ends are the two algorithms already raced above; what remains is the interior. One generator serves the whole dial: `train_ac` with $\lambda$ exposed. The advantage is `gae`, and the critic's target is the $\lambda$-return `A + V`, which at $\lambda = 0$ is the one-step target and at $\lambda = 1$ the reward-to-go, so a single knob moves both uses of the estimate coherently. Following :numref:`sec_baselines`, the runs are deliberately small, fifty updates of four episodes, both to keep the estimator's variance visible and because five values of $\lambda$ times five seeds is twenty-five training runs.

```{.python .input #actor-critic-measuring-the-dial-1}
%%tab pytorch, jax
lams = [0.0, 0.5, 0.9, 0.95, 1.0]

def train_gae(seed, lam, num_updates=50, batch_episodes=4):
    """train_ac with the dial exposed; lam=0 is train_ac line for line."""
    rng, env = np.random.default_rng(seed), gym.make('CartPole-v1')
    ac = cartpole_agent(seed)
    env.reset(seed=seed)
    for _ in range(num_updates):
        batch = d2l.rollout(env, ac.act, batch_episodes, rng)
        for _ in range(critic_steps):
            fit_value(ac, batch.obs, batch.gae(ac.value_np, gamma, lam)
                      + ac.value_np(batch.obs))
        A = batch.gae(ac.value_np, gamma, lam)
        policy_step_clip(ac, batch, d2l.normalize(A))
        yield float(batch.episode_returns().mean())

sweep = {lam: d2l.run_seeds(train_gae, 5, lam=lam) for lam in lams}
```

```{.python .input #actor-critic-measuring-the-dial-2}
%%tab pytorch, jax
for lam, r in sweep.items():
    best = np.array([np.convolve(c, np.ones(20) / 20, 'valid').max()
                     for c in r])
    print(f'lambda = {lam:4}: best 20-update window, '
          f'median {np.median(best):5.1f}, seeds {np.round(np.sort(best))}')
d2l.plot_curves({f'lambda = {lam:g}': r for lam, r in sweep.items()},
                xlabel='update', ylabel='mean return of the batch',
                smooth=10, reference=500)
```

Read the printed best windows against the curves. The bottom of the dial trails decisively: the $\lambda = 0$ and $\lambda = 0.5$ arms are far from the ceiling when the budget ends, and their median curves have stopped rising. From $\lambda = 0.9$ upward the arms' seed ranges overlap too much to order them. Fifty updates is a sprint, and a sprint is the regime that flatters the deep end of the dial: the critic starts ignorant, so the arms that lean on it least learn fastest, while the arms that lean on it hardest are steered by its fictions until it catches up. What the sprint cannot show is the other half of the ledger, the stillness the main comparison measured at twice this budget, where the longest stretches of rest on the ceiling belonged to the $\lambda = 0$ end in every tab. Each end of the dial has now won one of the section's two races. To see why deployed implementations nevertheless sit high but strictly *inside* the dial, measure the estimator itself rather than the training run it drives. Freeze a partially trained agent, so that the critic is still wrong enough for the dial to matter, draw many batches at fixed parameters, and grade every $\lambda$'s policy-gradient estimate against a reference: the mean of the $\lambda = 1$ estimator across all draws, which is the many-rollout Monte Carlo answer and the only unbiased anchor available without a solvable model.

```{.python .input #actor-critic-measuring-the-dial-3}
%%tab pytorch, jax
rng, env = np.random.default_rng(1), gym.make('CartPole-v1')
frozen = cartpole_agent(1)
env.reset(seed=1)
for _ in range(40):   # a mid-training freeze: the critic is still wrong
    b = d2l.rollout(env, frozen.act, batch_episodes, rng)
    for _ in range(critic_steps):
        fit_value(frozen, b.obs, b.td_target(frozen.value_np, gamma))
    policy_step_clip(frozen, b, d2l.normalize(
        b.td_target(frozen.value_np, gamma) - frozen.value_np(b.obs)))
```

```{.python .input #actor-critic-measuring-the-dial-4}
%%tab pytorch, jax
if tab.selected('pytorch'):
    def grad_vec(ac, b, w):
        loss = -(torch.as_tensor(w) * ac.log_prob(
            torch.as_tensor(b.obs), torch.as_tensor(b.act))).mean()
        g = torch.autograd.grad(loss, list(ac.policy.parameters()))
        return np.concatenate([x.numpy().ravel() for x in g])
if tab.selected('jax'):
    @nnx.jit
    def _grad_tree(policy, obs, act, adv, n_real):
        return nnx.grad(lambda p: -(adv * jax.nn.log_softmax(p(obs), -1)[
            jnp.arange(obs.shape[0]), act]).sum() / n_real)(policy)

    def grad_vec(ac, b, w):   # padded + jitted: 1000 calls below
        size = 1 << max(6, (len(w) - 1).bit_length())
        g = _grad_tree(ac.policy, _pad(b.obs, size), _pad(b.act, size),
                       _pad(w, size), jnp.float32(len(w)))
        return np.concatenate([np.asarray(x).ravel()
                               for x in jax.tree.leaves(g)])

draws = {lam: [] for lam in lams}
for _ in range(200):
    b = d2l.rollout(env, frozen.act, 4, rng)
    for lam in lams:
        draws[lam].append(grad_vec(frozen, b,
                                   b.gae(frozen.value_np, gamma, lam)))
u_ref = np.stack(draws[1.0]).mean(axis=0)   # the many-rollout reference
for lam in lams:
    u = np.stack(draws[lam])
    bias = np.linalg.norm(u.mean(0) - u_ref) / np.linalg.norm(u_ref)
    var = ((u - u.mean(0)) ** 2).sum(1).mean() / (u_ref ** 2).sum()
    print(f'lambda = {lam:4}: relative bias {bias:4.2f}, relative '
          f'variance {var:6.1f}, one-draw error {bias ** 2 + var:6.1f}')
```

The table is the shallow U the whole section has been circling. Leaving $\lambda = 1$, the variance collapses first and fastest, most of it gone by $\lambda = 0.9$, while the bias climbs more slowly toward its $\lambda = 0$ plateau; the one-draw error, bias squared plus variance, is therefore highest at the pure Monte Carlo end, high again at the pure TD end, and lowest strictly inside, in the $\lambda = 0.9$ to $0.95$ band across our tabs. That band is the neighborhood of the defaults deployed PPO implementations commonly run (:numref:`sec_ppo`), a sensible bias/variance compromise rather than a universal law. Two caveats. The $\lambda = 1$ row's bias reads zero by construction, since that estimator's mean *is* the reference, and the reference itself is known only to the precision of 200 draws of the noisiest estimator, so the bias column is trustworthy only where it is large; both large-$\lambda$ bias entries clear that bar. And the U's exact minimum depends on how wrong the frozen critic is, which is why the prose quotes its location as a neighborhood and not a digit.

Finally, the audit promised at the start: strip away the sampling noise and ask what policies the two original methods actually delivered, by evaluating the trained seed-0 agents greedily.

```{.python .input #actor-critic-measuring-the-dial-5}
%%tab pytorch, jax
env = gym.make('CartPole-v1')
for name in runs:
    env.reset(seed=2)
    score = d2l.evaluate(env, agents[name][0].act_greedy, num_episodes=100)
    print(f'{name:>20}: greedy mean return over 100 episodes: {score:.0f}')
```

Both agents land at 499 or 500 of the 500 ceiling: the ceiling can only be tied, and nothing here ranks them. What the section's measurements ranked was never the final policy but the estimator that got there: how noisy the road was, how big the steps were, and how still the arrival.

## Summary

Actor--critic methods use a learned value function to construct bootstrapped policy-gradient weights. With an exact critic, the expected temporal-difference error equals the advantage; an approximate critic introduces bias while usually reducing variance. The critic update is a semi-gradient policy-evaluation step, and the actor and critic form an approximate generalized policy-iteration scheme. Multi-step returns interpolate between one-step bootstrapping and Monte Carlo returns. The $\lambda$-return mixes these targets geometrically, and GAE computes the equivalent discounted sum of temporal-difference errors with a backward scan.

**Experimental scope.** The training comparisons use three seeds per framework, the $\lambda$ sweep uses five seeds, and the frozen-policy estimator study uses 200 batches. Both one-step and Monte Carlo methods solve CartPole in these runs. Intermediate high values of $\lambda$ give the smallest estimator error on the frozen-policy probe, but the precise optimum varies by framework and policy. The experiments support the bias--variance interpretation rather than a universal choice of $\lambda$.

## Exercises

1. [conceptual] *Why the baseline argument does not extend.* The identity of
   :numref:`sec_baselines` allowed any $b(s_t)$ to be subtracted from the
   weight without bias. Explain why it does not cover the bootstrapped weight
   $\delta_t$ of :eqref:`eq_td_error_v`, by pointing at exactly which quantity
   in the argument stops being independent of the action $a_t$. Under what
   condition on $\hat{V}$ does the bias vanish?
1. [short-code] *Where the U comes from.* Take the estimator table's relative
   bias and variance per $\lambda$ and combine them into a predicted
   single-draw error, then predict which $\lambda$ the training sweep should
   favor *before* looking at its returns. Where prediction and sweep disagree,
   what does the disagreement say about what a training run adds that a
   frozen-policy estimate cannot see? (The critic improves during training,
   the policy moves, and the weight scale interacts with the step size.)
1. [short-code] *The two timescales.* Sweep `critic_steps` over
   $\{1, 5, 20\}$, two seeds each; about twenty minutes on a laptop CPU. Where
   does the algorithm fail, and does it fail by learning nothing or by
   learning something wrong? Connect the failure to the sentence in the
   two-timescale paragraph about the critic chasing a moving target twice
   over.
1. [short-code] *The clip, removed.* Log the norm of the policy gradient in
   both methods, before the clip is applied. Although both normalize their
   weights to unit variance, the actor-critic norms come out roughly an order
   of magnitude larger. Explain the gap with the baseline argument of
   :numref:`sec_baselines`, then predict, and check, which of the two runs
   changes at all when `grad_clip` is removed.
1. [extended] *Fully online.* :eqref:`eq_actor_critic` can be applied at every
   single transition, with no batch at all. Implement this variant on CartPole
   and compare it with the batched version at a matched number of environment
   steps. What goes wrong, and which ingredient of :numref:`sec_dqn`'s recipe
   addresses it?
1. [conceptual] *Where the actor-critic sits.* Place REINFORCE, REINFORCE with
   a learned baseline, actor-critic, and Q-learning on two axes: how much bias
   the update carries, and how long the algorithm must wait before it can
   learn from a transition. Which corner is empty, and why?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.1]{.kicker}

Actor-critic and the credit-assignment dial<br>
**bootstrap the reward-to-go · one number, two learners · the $\lambda$ dial and its telescoping identity · bias against variance, measured**
:::
:::

::: {.slide title="Bootstrapping the Reward-to-Go"}
The Monte Carlo weight waits for the episode to end. But
$\hat G_t = r_t + \gamma \hat G_{t+1}$, and $\hat V(s_{t+1})$ is
trained to predict exactly what $\hat G_{t+1}$ samples. Substitute:

$$\delta_t = r_t + \gamma\, \hat V(s_{t+1}) - \hat V(s_t)$$

:eqref:`eq_td_error`'s scalar, with the max replaced by the
policy's own continuation.

. . .

@actor-critic-the-td-error

A numpy array carries no gradient graph: the target is data
by construction. No `detach`; the boundary is the detach.
:::

::: {.slide title="Why the TD Error Is an Advantage"}
If the critic were exact, $\hat V = V^\pi$:

$$E[\delta_t \mid s_t, a_t] = Q^\pi(s_t, a_t) - V^\pi(s_t).$$

One transition estimates what the Monte Carlo weight needed the
whole remaining trajectory for.

. . .

The price: during training $\hat V \ne V^\pi$, so the update is
biased. Variance traded for bias, the same bargain Q-learning
struck, now on the policy side.

![](../img/mdl-rl-td-mc-spectrum.svg){width=98%}
:::

::: {.slide title="Actor and Critic Updates"}
$$w \leftarrow w + \alpha_w\, \delta_t\, \nabla_w \hat V_w(s_t), \qquad
\theta \leftarrow \theta + \alpha_\theta\, \delta_t\,
\nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

![](../img/mdl-rl-actor-critic-loop.svg){width=95%}

. . .

- the critic: sampled, bootstrapped **policy evaluation**;
  actor-critic = GPI with both halves approximate
- two-timescale *intuition*: the critic chases a moving target
  twice over, so let it learn faster; realized heuristically here
  as 20 fitted-TD passes (fresh target each pass), taken *first*
:::

::: {.slide title="A Batched Actor-Critic"}
Batched episodes, one actor step per batch: a batched on-policy
actor-critic, the single-environment teaching relative of A2C
:cite:`Mnih.Badia.Mirza.ea.2016`. Against :numref:`sec_deeprl`'s
loop, only the tail is replaced:

@actor-critic-a2c-by-name-5

. . .

Critic target `td_target` instead of `G`; `critic_steps` passes,
first; weight $=$ normalized $\delta_t$. Nothing else.
:::

::: {.slide title="CartPole, Three Seeds Each"}
@!actor-critic-a2c-by-name-9

. . .

Early: actor-critic trails, the critic is still wrong. Late: both
tie the 500 ceiling; only the bootstrapped side **rests** on it.
:::

::: {.slide title="Return and Stability"}
@!actor-critic-a2c-by-name-7

. . .

A perfect batch = all eight episodes at 500. The long streaks
live on the bootstrapped side only, a run-specific stability
visualization of the mechanism: the Monte Carlo weight still
carries every remaining coin flip, and a run at the ceiling is
one noisy batch from a stumble.
:::

::: {.slide title="Gradient-Norm Comparison"}
@!actor-critic-a2c-by-name-8

. . .

Both weights are normalized to unit variance, yet the
actor-critic gradients are an order of magnitude larger: the
baseline argument of :numref:`sec_baselines` read backwards. The
state-dependent part of a weight cancels in expectation; the
Monte Carlo weight spends much of its variance there, while
$\delta_t$ is almost all action-dependent signal. A better
advantage estimate is a bigger step: :numref:`sec_ppo`.
:::

::: {.slide title="Critic Targets and Fresh Data"}
Rerun the loop, measuring both candidate weights per batch:

- $\hat G_t - \hat V(s_t) = \delta_t + \gamma(\hat G_{t+1} - \hat V(s_{t+1}))$:
  one part shared signal, hundreds of parts tail
- their correlation **falls** from about $0.3$ to zero as episodes
  stretch toward 500, consistent with the MC weight dissolving
  into tail noise (a descriptive diagnostic, not a critic
  certificate)
- its variance grows with the horizon; the TD error's is pinned
  flat (median ratio: beyond a hundredfold)

. . .

Why survivable here: freshness keeps the critic's training
distribution matched to where its errors matter, *reduced
mismatch*, not a guarantee (nonlinear TD can diverge even
on-policy :cite:`Tsitsiklis.VanRoy.1997`). Our critic refreshes
its target every pass, an aggressive fitted-TD loop;
:numref:`sec_dqn` severs the freshness and reverses the choice:
a frozen second copy, the target network.
:::

::: {.slide title="The Multi-Step Return Parameter"}
$$\hat G^{(n)}_t = r_t + \cdots + \gamma^{n-1} r_{t+n-1}
+ \gamma^n \hat V(s_{t+n}), \qquad
\hat A^{\textrm{GAE}}_t = (1-\lambda) \sum_{n \ge 1}
\lambda^{n-1} \big(\hat G^{(n)}_t - \hat V(s_t)\big)$$

. . .

**Telescoping identity** :cite:`Schulman.Moritz.Levine.ea.2016`:

$$\hat A^{\textrm{GAE}}_t
= \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}$$

Each depth telescopes into TD errors; swap the sums; the inner
geometric sum is $\lambda^l$. TD($\lambda$)'s eligibility traces
ran this backward, per step; streaming settings still do
:cite:`Elsayed.Vasan.Mahmood.2024`.
:::

::: {.slide title="Computing GAE by a Backward Scan"}
@actor-critic-gae-in-one-old-function-1

The reward-to-go scan of :numref:`sec_baselines`, run on TD
errors: a new estimator is a new input to an old function.

. . .

@!actor-critic-gae-in-one-old-function-2

Both endpoints asserted, not assumed.
:::

::: {.slide title="Comparing GAE Parameters"}
@!actor-critic-measuring-the-dial-2

. . .

Five seeds per $\lambda$, a fifty-update sprint on small batches:
the bottom of the dial trails decisively; from $0.9$ up the arms
arrive together. The sprint flatters the deep end (the critic
starts ignorant); the *stillness* race went the other way.
:::

::: {.slide title="Bias and Variance"}
@!actor-critic-measuring-the-dial-4

. . .

The shallow U: leaving $\lambda = 1$, variance collapses first
and fastest; bias climbs more slowly. One-draw error is high
at both pure ends, lowest strictly inside. Deployed PPO defaults
commonly sit at $\lambda \approx 0.9$ to $0.97$
(:numref:`sec_ppo`).
:::

::: {.slide title="Recap"}
- $\delta_t$: a one-transition advantage estimate, unbiased only
  at $\hat V = V^\pi$; variance for bias, Q-learning's bargain on
  the policy side
- the critic is sampled policy evaluation: GPI, both halves
  approximate, and a semi-gradient: the safety line of
  :numref:`sec_deeprl` is behind us
- batched actor-critic (A2C's teaching relative): critic passes
  first, freshest critic judges
- the dial: $n$-step, $\lambda$-return, one telescoping identity,
  one old scan; endpoints asserted
- next: reuse the batch safely (:numref:`sec_ppo`, GAE by
  default), then sever freshness and pay (:numref:`sec_dqn`)
:::
