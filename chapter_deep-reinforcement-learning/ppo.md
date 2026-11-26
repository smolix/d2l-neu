# Trust Regions and Proximal Policy Optimization
:label:`sec_ppo`

The agents of :numref:`sec_deeprl` and :numref:`sec_actorcritic` collect eight episodes, take one gradient step, and throw the batch away. Once the agent balances the pole well, those eight episodes are four thousand steps of experience, and all of it informs a single update. The step is also unguarded. Nothing stops a large update from moving the policy a long way, and the next batch is then collected by whatever policy that update produced. Can we take several steps on one batch, and how far may they go?

The answer starts with a fact about parameters. The same change of the parameters can leave a policy's behavior almost unchanged or change it completely, depending on where it starts, so a step size in parameter space does not control the quantity that matters, the change in behavior. In 2015 John Schulman and colleagues made that change the thing to control. Trust Region Policy Optimization (TRPO) bounds the divergence between the old and the new action distributions at the visited states, guided by a theorem on how much improvement such a step guarantees :cite:`Schulman.Levine.Abbeel.ea.2015`. Two years later Proximal Policy Optimization (PPO) replaced the constraint by a clipped objective that any gradient optimizer can run :cite:`Schulman.Wolski.Dhariwal.ea.2017`. PPO became the default policy-gradient method, from simulated robots to the fine-tuning of language models on human feedback :cite:`ouyang2022training`.

This section builds both. A two-action policy shows why parameter distance is the wrong quantity. A change of measure then lets a batch collected by the old policy evaluate a new one, exactly but with a variance that grows with the episode length, and keeping one ratio per step makes the reuse affordable. A short lemma says what that approximation leaves out; TRPO bounds the error and PPO clips it. On CartPole we then measure how many environment steps reuse saves, what goes wrong without the clip, and which numbers to watch while a policy trains.

```{.python .input #ppo-setup}
%%tab jax
# Deterministic XLA kernels. On an accelerator XLA otherwise picks each
# matrix product's kernel by timing candidates, and kernels that add in
# different orders differ in the last bits; training runs at large step
# sizes can turn such differences into different outcomes. JAX reads
# XLA_FLAGS once, when it starts, so this cell runs before the imports.
import os
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_deterministic_ops=true').strip()
```

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-2}
%%tab pytorch, jax
import gymnasium as gym
```

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-3}
%%tab pytorch
torch.set_num_threads(1)
```

:begin_tab:`pytorch`
`torch.set_num_threads(1)` keeps PyTorch from spreading the small matrix products of these networks over many cores, which only slows them down.
:end_tab:

## Parameter Space versus Policy Space

Since we update the parameters of the policy, we should ask what a parameter update does to the policy itself. A family of policies with a single parameter $\theta$ and two actions answers it; the example comes from Joshua Achiam's lectures on policy optimization :cite:`Achiam.2017`. Let

$$\pi_\theta(a) = \begin{cases} \sigma(\theta) & a = 1 \\ 1 - \sigma(\theta) & a = 2, \end{cases} \qquad \textrm{where } \sigma(\theta) = \frac{1}{1 + e^{-\theta}}.$$

Take two parameter updates of the same size, $\Delta\theta = 2$, from two different starting points:

$$
\begin{aligned}
\pi_{\theta=0}(a{=}1) &= \sigma(0) = 0.50 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(2) = 0.88, \\
\pi_{\theta=6}(a{=}1) &= \sigma(6) = 0.9975 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(8) = 0.9997.
\end{aligned}
$$

The first update takes the agent from indifference between the two actions to a strong preference for one of them. The second changes nothing that an observer of the agent could detect. The slope of the map from parameter to policy says why: $\sigma'(\theta)=\sigma(\theta)(1-\sigma(\theta))$ is $0.25$ at $\theta=0$ and about $0.0025$ at $\theta=6$. :numref:`fig_rl-policy-vs-parameter` draws both updates.

![Two parameter updates of the same size, $\Delta\theta = 2$. Left: the map $\pi_\theta(a{=}1) = \sigma(\theta)$ with both updates and the slopes at their starting points. Right: the two action distributions before and after each update.](../img/mdl-rl-policy-vs-parameter.svg)
:label:`fig_rl-policy-vs-parameter`

This difference makes on-policy training fragile. A large noisy update near $\theta=0$ can push the policy into saturation. There the frequently chosen action has an almost-zero score, $\nabla_\theta \log \pi_\theta(a{=}1) = 1 - \sigma(\theta) \approx 0.0025$ at $\theta = 6$, while the other action has a score near $-1$ but is sampled once in four hundred steps. Later on-policy batches then carry too little evidence to reverse the update. A limit on the Euclidean length of the parameter step does not prevent this, because the same length is harmless at $\theta = 6$ and drastic at $\theta = 0$.

An update should therefore limit a distance between policies rather than between parameters. Within such a limit the optimizer may take a long parameter step where it changes the policy little, and, as we will see, it may take several steps on the same batch.

## Importance Sampling

### The Change of Measure

Every estimator since :numref:`sec_policygradient` has used trajectories sampled from the current policy, so a batch became useless after one gradient step. Suppose instead that the batch was collected by an earlier policy $\pi_{\theta_{\textrm{old}}}$. Can it still tell us how well an updated policy $\pi_\theta$ does?

Importance sampling gives an exact answer. For any function $f$ of trajectories,

$$
\begin{aligned}
\mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \big[ f(\tau) \big] &= \sum_\tau P(\tau; \theta)\, f(\tau) = \sum_\tau P(\tau; \theta_{\textrm{old}})\ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) \\
&= \mathbb{E}_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) \Big],
\end{aligned}
$$
:eqlabel:`eq_change_of_measure`

valid whenever every trajectory the new policy can produce has positive probability under the old one. Softmax policies satisfy this condition, because they never assign zero probability to an action. Applied with $f = R$, the return, the identity rewrites the objective as an expectation under the *old* policy:

$$J(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \big[ R(\tau) \big] = \mathbb{E}_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, R(\tau) \Big].$$
:eqlabel:`eq_offpolicy_objective`

Thus data from the old policy, each trajectory weighted by its likelihood ratio, estimate the new policy's return without bias. The probability of a trajectory is a product of the start-state probability, the policy's action probabilities and the environment's transition probabilities, as in :numref:`sec_policygradient`. In the ratio everything but the policy cancels, so the weight needs no model of the environment:

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})} = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}.$$
:eqlabel:`eq_traj_ratio`

### The Exploding Product

The estimator is unbiased, but its variance grows with the length of the episode. Consider a policy that ignores the state, collects data with $\pi_{\theta_{\textrm{old}}} = (0.5, 0.5)$, and is evaluated at $\pi_\theta = (0.6, 0.4)$. Each factor of :eqref:`eq_traj_ratio` is $1.2$ or $0.8$ with equal probability, so its mean is $1$ and its mean square is $(1.44 + 0.64)/2 = 1.04$. The factors of different steps are independent, so the trajectory weight $W_T$ of an episode of $T$ steps has mean $1$ and variance

$$\textrm{Var}[W_T] = 1.04^T - 1,$$

which is about $0.5$ for $T = 10$, about $50$ for $T = 100$ and about $3 \times 10^8$ for CartPole's $500$ steps. A policy that differs from the data's policy by ten percentage points per decision thus cannot be evaluated from whole CartPole episodes: a handful of trajectories with large weights decide the estimate.

### The Per-Step Surrogate

The reward at step $t$ depends only on the actions taken up to step $t$, so the exact correction needs only the ratios $\rho_0 \cdots \rho_t$ for that reward, where

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}.$$

The product $\rho_0 \cdots \rho_{t-1}$ of the earlier ratios corrects for the states: it turns the old policy's distribution of $s_t$ into the new policy's. Dropping it, and keeping only $\rho_t$ for the action at $s_t$, gives a weight whose variance no longer grows with $t$. This one approximation evaluates the new policy's actions at the old policy's states. With an advantage estimate $\hat{A}_t^i$ for step $t$ of trajectory $i$, the resulting *surrogate objective* averages over all $N = \sum_i T_i$ steps of a batch of $n$ trajectories:

$$\hat{L}(\theta) = \frac{1}{N} \sum_{i=1}^n \sum_{t=0}^{T_i-1} \rho_t^i(\theta)\ \hat{A}_t^i.$$
:eqlabel:`eq_surrogate`

The average over steps rather than over trajectories is what the code computes, and it makes the divergence constraint below a mean over states. At $\theta = \theta_{\textrm{old}}$ every ratio equals one and $\nabla_\theta \rho_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, so the gradient of $\hat{L}$ there points in the same direction as the estimator of :numref:`sec_baselines`: each score weighted by its advantage, with the factor $\gamma^t$ dropped, as every implementation since then has done. With advantages standardized to mean zero, $\hat{L}(\theta_{\textrm{old}}) = 0$ exactly. Away from $\theta_{\textrm{old}}$ the surrogate becomes less accurate, and the next question is how far it can be trusted.

## Trust Regions

### The Performance Difference Lemma

The surrogate evaluates the new policy's actions at the old policy's states. To see what this substitution changes, we compare the two returns exactly. Write $A^{\textrm{old}}(s,a)=Q^{\pi_{\theta_{\textrm{old}}}}(s,a)-V^{\pi_{\theta_{\textrm{old}}}}(s)$ for the old policy's advantage: how much better action $a$ is than the old policy's average at $s$, when the old policy acts afterward. The following lemma holds for episodes that end at a terminal state, and for continuing tasks with $\gamma < 1$. Our CartPole runs bootstrap at the 500-step time limit, as in :numref:`sec_deeprl`, so the objective they optimize is the continuing one.

**Proposition** (performance difference lemma, :citet:`Kakade.Langford.2002`).

$$
J(\theta) - J(\theta_{\textrm{old}}) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \Big[ \sum_{t \geq 0} \gamma^t\, A^{\textrm{old}}(s_t, a_t) \Big].
$$
:eqlabel:`eq_perf_diff`

**Proof.** Write $V$ for $V^{\pi_{\theta_{\textrm{old}}}}$. Along any trajectory the sum $\sum_t \gamma^t \big( r_t + \gamma V(s_{t+1}) - V(s_t) \big)$ telescopes: the interior $V$ terms cancel in pairs, and the last one is zero at a terminal state or vanishes as $t \to \infty$ when $\gamma < 1$. What remains is $R(\tau) - V(s_0)$. Now take expectations under $\tau \sim P(\cdot;\, \theta)$. Given $(s_t, a_t)$, the expectation of $r_t + \gamma V(s_{t+1})$ over the next state is $Q^{\pi_{\theta_{\textrm{old}}}}(s_t, a_t)$, by the Bellman equation of the old policy, so the term at step $t$ becomes $\gamma^t\, \mathbb{E}\big[ A^{\textrm{old}}(s_t, a_t) \big]$. On the other side, $\mathbb{E}[R(\tau)] = J(\theta)$, and $\mathbb{E}[V(s_0)] = J(\theta_{\textrm{old}})$ because both policies draw $s_0$ from the same start distribution. $\blacksquare$

The lemma says that the new policy improves by the old policy's advantages, summed along the new policy's own trajectories. Grouping its terms by state makes the comparison with the surrogate exact. Let $d^{\pi}_\gamma(s) = (1 - \gamma) \sum_{t \geq 0} \gamma^t \Pr(s_t = s)$ be the discounted state-occupancy distribution of :numref:`sec_policygradient`, the distribution of states a policy visits with step $t$ weighted by $\gamma^t$. Then

$$J(\theta) - J(\theta_{\textrm{old}}) = \frac{1}{1-\gamma} \sum_s d^{\pi_\theta}_\gamma(s) \sum_a \pi_\theta(a \mid s)\, A^{\textrm{old}}(s, a).$$

The surrogate computes this sum with the old policy's occupancy $d^{\pi_{\theta_{\textrm{old}}}}_\gamma$ in place of the new one's; that substitution is its only error. The error depends on how much the state distribution moves, and that depends on how much the policy changes, not on how far the parameters move.

### The Monotonic-Improvement Bound

Replacing the occupancy in the last display defines the population version of the surrogate,

$$\bar{L}(\theta) = \frac{1}{1-\gamma} \sum_s d^{\pi_{\theta_{\textrm{old}}}}_\gamma(s) \sum_a \pi_\theta(a \mid s)\, A^{\textrm{old}}(s, a),$$
:eqlabel:`eq_pop_surrogate`

of which $\hat{L}$ is the sampled estimate, without the factor $\gamma^t$. Since the old policy's advantage averages to zero at every state, $\bar{L}(\theta_{\textrm{old}}) = 0$. How far may $\theta$ move before $\bar{L}$ stops predicting the improvement? The answer is a bound in terms of the largest change of the action distribution at any state.

**Proposition** (:citet:`Kakade.Langford.2002,Schulman.Levine.Abbeel.ea.2015`). *For $\gamma < 1$ and $A_{\max} = \max_{s, a} \lvert A^{\textrm{old}}(s, a) \rvert$,*

$$
\begin{aligned}
J(\theta)\ &\geq\ J(\theta_{\textrm{old}}) + \bar{L}(\theta) \\
&\quad -\ \frac{4 \gamma A_{\max}}{(1-\gamma)^2}\ \max_s\, D_{\textrm{KL}}\big( \pi_{\theta_{\textrm{old}}}(\cdot \mid s)\ \Vert\ \pi_\theta(\cdot \mid s) \big).
\end{aligned}
$$
:eqlabel:`eq_trpo_bound`

The surrogate is right up to a penalty proportional to the largest Kullback--Leibler divergence between the old and new action distributions at any state. Exercise :exref:`ex_ppo-monotonic-bound` derives it by running the two policies side by side with coupled actions. Both sides of the bound equal $J(\theta_{\textrm{old}})$ at $\theta = \theta_{\textrm{old}}$, so any step that raises the right-hand side raises the true return: maximizing the bound gives monotonic improvement. The constant is too large to use as it stands. At $\gamma = 0.99$ it is $4 \cdot 0.99 / 0.01^2 \approx 40{,}000$ times $A_{\max}$, so the steps it certifies are tiny.

TRPO keeps the shape of the bound and replaces its pieces by quantities a batch can estimate. It maximizes the sampled surrogate subject to a budget $\Delta_{\textrm{KL}}$ on the mean divergence over the batch's states,

$$
\begin{aligned}
\max_\theta\ &\hat{L}(\theta) \\
\textrm{subject to}\ &\frac{1}{N}\sum_{i,t} D_{\textrm{KL}}\big( \pi_{\theta_{\textrm{old}}}(\cdot \mid s_t^i)\ \Vert\ \pi_\theta(\cdot \mid s_t^i) \big) \leq \Delta_{\textrm{KL}},
\end{aligned}
$$
:eqlabel:`eq_trpo_constraint`

with a mean in place of the maximum over all states and a chosen budget, typically $\Delta_{\textrm{KL}} = 0.01$, in place of the conservative coefficient. It solves this problem with a second-order method, which exercise :exref:`ex_ppo-trpo` builds from the pieces of this section. :numref:`fig_rl_trust_region` shows why the constraint lives in policy space.

![Bounding a policy update. Left: the surrogate $L$ is tangent to the true objective $J$ at $\theta_{\textrm{old}}$ but becomes inaccurate after a large policy change. The unconstrained maximizer of $L$ lowers $J$ from $0.82$ to $-0.38$, whereas the best point inside the shaded trust region, $D_{\textrm{KL}} \leq \Delta_{\textrm{KL}}$ (for one parameter an interval $\theta_{\textrm{old}} \pm \delta$), raises it to $1.52$. Right: for a three-action softmax policy, the second-order approximation $\frac{1}{2}\Delta\theta^\top F \Delta\theta \leq 0.02$ of the divergence constraint, with $F$ the Hessian of the divergence at $\theta_{\textrm{old}}$, is an ellipse in parameter space. Two parameter steps of equal Euclidean length change the policy by divergences of $0.009$ and $0.062$: the same parameter distance moves the policy about seven times as far in one direction as in the other.](../img/mdl-rl-trust-region.svg)
:label:`fig_rl_trust_region`

### The Clipped Objective

PPO replaces the constrained second-order problem by a first-order objective with a built-in limit. It does not clip the ratios themselves; it stops rewarding a sample once its ratio has moved beyond a band of half-width $\epsilon_{\textrm{clip}}$ around one:

$$
\begin{aligned}
L^{\textrm{CLIP}}(\theta) &= \frac{1}{N} \sum_{i,t} \min\Big( \rho_t^i(\theta)\, \hat{A}_t^i, \\
&\qquad\qquad \textrm{clip}\big(\rho_t^i(\theta),\ 1-\epsilon_{\textrm{clip}},\ 1+\epsilon_{\textrm{clip}}\big)\, \hat{A}_t^i \Big).
\end{aligned}
$$
:eqlabel:`eq_ppo_clip`

Consider one sample. If $\hat{A}_t > 0$, its contribution grows with $\rho_t$ only until $\rho_t = 1+\epsilon_{\textrm{clip}}$; beyond that the clipped term is the smaller one and its gradient is zero. If $\hat{A}_t < 0$, the threshold is $1-\epsilon_{\textrm{clip}}$. With the usual $\epsilon_{\textrm{clip}} = 0.2$ and an action the old policy chose with probability $0.5$, a sample with positive advantage stops rewarding the update once the new policy gives that action probability $0.6$; with negative advantage, once the probability falls to $0.4$. The minimum makes the limit one-sided: a change that lowers the objective is always counted, while further movement in the favorable direction adds nothing. :numref:`fig_rl_ppo_clip` plots one sample's contribution for both signs of the advantage.

![One sample's contribution to the clipped objective :eqref:`eq_ppo_clip` against the ratio $\rho_t$. For positive advantage it stops increasing above $1+\epsilon_{\textrm{clip}}$; for negative advantage it stops increasing below $1-\epsilon_{\textrm{clip}}$. On the other side of one the minimum keeps the unclipped term, so a change that lowers the objective is never hidden.](../img/mdl-rl-ppo-clip.svg)
:label:`fig_rl_ppo_clip`

Because the ratio itself is not clipped, it can still leave the band, for two reasons. All samples share the network's parameters, so a step taken for one sample moves the ratios of the others; and Adam's momentum keeps moving the parameters for a few steps after a sample's gradient has become zero. :citet:`Engstrom.Ilyas.Santurkar.ea.2020` measured that PPO's ratios do leave the band and that its divergence per update is not held below any fixed budget. The clip is therefore an incentive rather than a constraint, and unlike :eqref:`eq_trpo_bound` it guarantees no improvement. What it does permit is several passes of optimization over one batch without rewarding ever larger changes of individual action probabilities.

### The Entropy Bonus

Implementations often add a small *entropy bonus*, a weight `entropy_coef` times the mean entropy of the action distributions at the batch's states. The bonus pushes probabilities away from zero and one, where, as in the two-action example, the frequently chosen action has a small score and recovery is slow. The entropy of a distribution over $K$ actions is $\log K$ minus its divergence from the uniform distribution, so the bonus is a penalty on the distance from the uniform policy.

## PPO in Practice

### The Advantage Estimate

Any advantage estimate can fill the place of $\hat{A}$ in :eqref:`eq_ppo_clip`. We use generalized advantage estimation (GAE) from :numref:`sec_actorcritic`: `Batch.gae` sums the TD errors $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ of the critic backward through each episode with weights $(\gamma\lambda)^k$, and $\lambda$ sets how far ahead each advantage looks. That section found the error of this estimate smallest near $\lambda = 0.9$ to $0.95$; $0.95$ is also the setting of the PPO paper. The advantages are computed with the critic as it stands before it is fit to the batch, the order of the GAE paper :cite:`Schulman.Moritz.Levine.ea.2016`. `td_weight` of :numref:`sec_actorcritic` used the opposite order, because its actor's weight was the one-step TD error, which rests entirely on the critic. At $\lambda = 0.95$ the advantage is mostly the batch's own sampled rewards and leans little on the critic, so the order changes no result of this section, and we follow the paper.

### The Implementation

We keep the CartPole setup of :numref:`sec_actorcritic`. The agent is `ActorCritic.mlp(4, 2)` of :numref:`sec_deeprl`: a policy network and a value network, each with one hidden layer of 64 tanh units and its own Adam optimizer with learning rate $10^{-2}$. Each update collects a batch of eight episodes, and the critic takes `critic_steps = 20` regression passes on it. Three hyperparameters are new: `num_epochs = 20` passes of the actor over each batch, the half-width `epsilon_clip = 0.2` of the band, and the entropy weight `entropy_coef = 0.01`. Twenty passes is also the CartPole setting of the RL Baselines3 Zoo :cite:`Raffin.Hill.Gleave.ea.2021`, but the zoo's learning rate is $10^{-3}$, ten times smaller than ours. At our rate each pass moves the policy far, which makes training fast and is the setting in which the clip matters; the ablation below shows what happens without it.

```{.python .input #ppo-the-implementation-1}
%%tab pytorch, jax
gamma, lam, num_updates, batch_episodes = 0.99, 0.95, 60, 8
num_seeds, num_epochs, critic_steps = 8, 20, 20
epsilon_clip, entropy_coef = 0.2, 0.01
def cartpole_agent(seed):
    return d2l.ActorCritic.mlp(4, 2, seed=seed)
```

The per-sample objective is passed to the update as a function. `clipped_surrogate` is :eqref:`eq_ppo_clip`, and `plain_surrogate`, the term $\rho_t \hat{A}_t$ alone, serves the ablation below; both take the half-width $\epsilon_{\textrm{clip}}$ of the band, which only the first one uses.

```{.python .input #ppo-the-implementation-3}
%%tab pytorch
def clipped_surrogate(rho, adv, epsilon):  #@save
    """PPO's per-sample objective eq_ppo_clip: min(rho A, clip(rho) A)."""
    return torch.minimum(rho * adv,
                         rho.clamp(1 - epsilon, 1 + epsilon) * adv)

def plain_surrogate(rho, adv, epsilon):  #@save
    """The unclipped per-sample objective rho A."""
    return rho * adv
```

```{.python .input #ppo-the-implementation-3}
%%tab jax
def clipped_surrogate(rho, adv, epsilon):  #@save
    """PPO's per-sample objective eq_ppo_clip: min(rho A, clip(rho) A)."""
    return jnp.minimum(rho * adv,
                       jnp.clip(rho, 1 - epsilon, 1 + epsilon) * adv)

def plain_surrogate(rho, adv, epsilon):  #@save
    """The unclipped per-sample objective rho A."""
    return rho * adv
```

`ppo_epochs` receives a batch's states and actions together with the two quantities frozen before reuse: the advantages and the collecting policy's log-probabilities `logp_old`. It takes `num_epochs` steps of the actor's optimizer `opt_pi` on the negated mean surrogate minus the entropy bonus. The entropy needs the whole distribution at each state, so the update computes the log-softmax over all actions once and reads $\log\pi_\theta(a_t \mid s_t)$ from it. Before each pass it records one row of diagnostics: the fraction of ratios outside the band, the sample mean of $\log\pi_{\theta_{\textrm{old}}}-\log\pi_\theta$, and the mean entropy. The second number is called the *approximate KL*: its expectation under the old policy is the divergence $D_{\textrm{KL}}(\pi_{\theta_{\textrm{old}}} \Vert \pi_\theta)$ at the visited states, although on a finite batch it can come out negative. Since the row is recorded before the pass, the first row is always zero, zero and the old policy's entropy.

```{.python .input #ppo-the-implementation-4}
%%tab pytorch
def ppo_epochs(ac, obs, act, adv, logp_old, epsilon, num_epochs,  #@save
               entropy_coef=0.01, surrogate=clipped_surrogate):
    """num_epochs passes on the mean surrogate plus an entropy bonus, on one
    frozen batch of states and actions; returns [num_epochs, 3] numpy
    diagnostics, recorded before each pass: fraction of ratios outside the
    band, approximate KL, mean policy entropy."""
    obs, act, adv, logp_old = (torch.as_tensor(x)
                               for x in (obs, act, adv, logp_old))
    diag = []
    for _ in range(num_epochs):
        logp_all = torch.log_softmax(ac.policy(obs), dim=-1)
        logp = logp_all.gather(-1, act[:, None]).squeeze(-1)
        rho = torch.exp(logp - logp_old)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = -surrogate(rho, adv, epsilon).mean() - entropy_coef * entropy
        ac.opt_pi.zero_grad()
        loss.backward()
        ac.opt_pi.step()
        diag.append(torch.stack([((rho - 1).abs() > epsilon).float().mean(),
                                 (logp_old - logp).mean(), entropy]).detach())
    return torch.stack(diag).numpy()
```

```{.python .input #ppo-the-implementation-4}
%%tab jax
@nnx.jit(static_argnames='surrogate')  #@save
def _ppo_step(policy, opt, obs, act, adv, logp_old, mask, epsilon,
              entropy_coef, surrogate):
    """One step on the mean surrogate plus an entropy bonus over the real
    rows of a padded batch; returns, measured before the step, the fraction
    of ratios outside the band, the approximate KL and the mean entropy."""
    mean = lambda x: (mask * x).sum() / mask.sum()
    def loss_fn(policy):
        logp_all = jax.nn.log_softmax(policy(obs), axis=-1)
        logp = jnp.take_along_axis(logp_all, act[:, None], -1).squeeze(-1)
        rho = jnp.exp(logp - logp_old)
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1)
        loss = -mean(surrogate(rho, adv, epsilon) + entropy_coef * entropy)
        return loss, (rho, logp, entropy)
    (_, (rho, logp, entropy)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(policy)
    opt.update(policy, grads)
    return jnp.stack([mean(jnp.abs(rho - 1) > epsilon),
                      mean(logp_old - logp), mean(entropy)])

def ppo_epochs(ac, obs, act, adv, logp_old, epsilon, num_epochs,  #@save
               entropy_coef=0.01, surrogate=clipped_surrogate):
    """num_epochs passes on the mean surrogate plus an entropy bonus, on one
    frozen batch of states and actions; returns [num_epochs, 3] numpy
    diagnostics, recorded before each pass: fraction of ratios outside the
    band, approximate KL, mean policy entropy."""
    batch = d2l.padded(obs, act, adv, logp_old)
    step = nnx.cached_partial(_ppo_step, ac.policy, ac.opt_pi)
    return np.asarray(jnp.stack([
        step(*batch, epsilon, entropy_coef, surrogate=surrogate)
        for _ in range(num_epochs)]))
```

:begin_tab:`jax`
`ppo_epochs` pads the batch to a bucket length with `d2l.padded` (:numref:`sec_utils`), which also returns a mask of the real rows; `_ppo_step` leaves the padded rows out of every mean, so the compiled step is reused across batches of different lengths. The surrogate is a static argument of the compiled step, so each of the two surrogates compiles once. The diagnostics stay on the device until all passes are done.
:end_tab:

The training loop is the actor-critic loop of :numref:`sec_actorcritic` with its single actor step replaced by `ppo_epochs` and the advantages computed before the critic's passes. `rollout` plays eight episodes into a `Batch`. The loop then computes the advantages with `Batch.gae` and the current critic, and standardizes them over the batch with `standardize`. It also reads `logp_old` with `log_prob_np`, which returns $\log \pi(a \mid s)$ for given states and actions as a NumPy array. Next the critic takes its regression passes toward the $\lambda$-return target with `Batch.lambda_return` of :numref:`sec_actorcritic` and `fit_value` of :numref:`sec_deeprl`, and finally the actor takes its passes. Each update yields the environment steps spent so far and the batch's mean return. The diagnostics of each batch go to a list `diag` that the caller passes in; `CLIPPED`, `KL` and `ENTROPY` name their three columns.

```{.python .input #ppo-the-implementation-5}
%%tab pytorch, jax
CLIPPED, KL, ENTROPY = 0, 1, 2   # the columns of ppo_epochs' diagnostics

def train_ppo(seed, ac, diag, surrogate=clipped_surrogate,
              num_epochs=num_epochs):
    """PPO on CartPole: freeze the advantages and the collecting policy's
    log-probs, then take num_epochs surrogate passes on each batch. Appends
    each batch's diagnostics to diag; yields (env steps, mean return)."""
    rng, env = np.random.default_rng(seed), gym.make('CartPole-v1')
    env.reset(seed=seed)
    steps = 0
    for _ in range(num_updates):
        b = d2l.rollout(env, ac.act, batch_episodes, rng)
        adv = d2l.standardize(b.gae(ac.value_np, gamma, lam))  # pre-fit critic
        logp_old = ac.log_prob_np(b.obs, b.act)
        for _ in range(critic_steps):   # fresh lambda-return target, per pass
            d2l.fit_value(ac, b.obs, b.lambda_return(ac.value_np, gamma, lam))
        diag.append(ppo_epochs(ac, b.obs, b.act, adv, logp_old, epsilon_clip,
                               num_epochs, entropy_coef, surrogate))
        steps += len(b)
        yield steps, float(b.episode_returns().mean())
```

### Reuse against Environment Steps
:label:`subsec_ppo_reuse`

How many environment steps does reuse save? We train clipped PPO with one, four and twenty passes per batch, on eight seeds each, stop each run once it has spent 50,000 environment steps, and plot the batch's mean return against the steps spent so far. With one pass every ratio equals one and the clip never acts, so that arm is close to the $\lambda$ sweep of :numref:`sec_actorcritic`, with an entropy bonus, no cap on the gradient, and the advantages computed before the critic's passes. As there, `d2l.on_grid` places each run on a common grid of steps. These runs need no diagnostics, so each gets an empty list.

```{.python .input #ppo-reuse-against-environment-steps-1}
%%tab pytorch, jax
budget = 50_000
grid = np.arange(0, budget + 1, 1000)
reuse = {}
for e in (1, 4, 20):
    runs_e = []
    for s in range(num_seeds):
        run = []
        for row in train_ppo(s, cartpole_agent(s), [], num_epochs=e):
            run.append(row)                   # (env steps, mean return)
            if row[0] >= budget:
                break
        runs_e.append(np.array(run))
    reuse[f'{e} epoch' + 's' * (e > 1)] = d2l.on_grid(runs_e, grid, 1)
```

Plotting the interpolated return curves in `reuse` against `grid` shows how
batch reuse trades extra gradient passes for fewer environment interactions:

```{.python .input #ppo-reuse-against-environment-steps-2}
%%tab pytorch, jax
d2l.plot_curves(reuse, xlabel='environment steps',
                ylabel='mean return of the batch', x=grid, reference=500)
```

Reusing each batch for four epochs reaches the ceiling in about half the environment steps that one epoch needs, roughly 25,000 against 50,000 or more, and in about half the updates; within the 50,000-step budget the one-epoch median stays just short of the ceiling. Going on to twenty epochs changes little by comparison, and no clipped run collapses. The gain comes from the batch: a batch of a few thousand steps holds more information than one gradient step can extract, and the clip lets the optimizer take several steps without moving any action probability too far.

### The Clipping Ablation
:label:`subsec_ppo_ablation`

Four passes are enough to collect most of the gain; at our learning rate, twenty passes move the policy a long way within one batch. Does the clip matter there? We train twenty-epoch PPO on the same eight seeds with the clip on, keeping the trained agents for later checks and the full diagnostics for the next subsection:

```{.python .input #ppo-the-ablation-1}
%%tab pytorch, jax
arms = {'clipped (PPO)': clipped_surrogate, 'no clip': plain_surrogate}
agents = {arm: [cartpole_agent(s) for s in range(num_seeds)] for arm in arms}
runs, diag = {}, {arm: [] for arm in arms}
runs['clipped (PPO)'] = np.array(
    [list(train_ppo(s, agents['clipped (PPO)'][s], diag['clipped (PPO)']))
     for s in range(num_seeds)])
```

Then the control, identical in every line except that the surrogate is unclipped. Each arm's diagnostics are then reshaped to (seed, update, epoch, column):

```{.python .input #ppo-the-ablation-2}
%%tab pytorch, jax
runs['no clip'] = np.array(
    [list(train_ppo(s, agents['no clip'][s], diag['no clip'],
                    surrogate=plain_surrogate))
     for s in range(num_seeds)])
diag = {arm: np.reshape(d, (num_seeds, num_updates, num_epochs, 3))
        for arm, d in diag.items()}
```

Two criteria summarize each run. A run *holds* the task if its batch return, once it first reaches 475, never again falls below 250, half the maximum. A run ends *dead* if its last ten updates average below a return of 100:

```{.python .input #ppo-the-ablation-3}
%%tab pytorch, jax
def holds(ret):   # reaches 475, then never falls below 250 again
    hit = np.flatnonzero(ret >= 475)
    return len(hit) > 0 and ret[hit[0]:].min() >= 250

for name, r in runs.items():
    held = sum(holds(ret) for ret in r[:, :, 1])
    dead = r[:, -10:, 1].mean(axis=1) < 100
    print(f'{name:>13}: {held} of {num_seeds} seeds hold the task, '
          f'{int(dead.sum())} end dead')
```

Plotting the batch return trajectories `r[:, :, 1]` across all updates
contrasts the steady plateau of `'clipped (PPO)'` with the collapse of
`'no clip'`:

```{.python .input #ppo-the-ablation-4}
%%tab pytorch, jax
d2l.plot_curves({name: r[:, :, 1] for name, r in runs.items()},
                xlabel='update', ylabel='mean return of the batch',
                reference=500)
```

At twenty epochs and a learning rate of $10^{-2}$, every clipped seed reaches the ceiling and holds it. Without the clip, hardly any seed holds it, and some end dead. Over 64 seeds outside the notebook, about one run in twenty holds the task and about half end dead, most of them at a return of about 9, the length of an episode in which the cart is pushed the same way at every step. Eight seeds are too few to show these rates reliably, and the count of dead runs printed above can fall well short of half. Repeated unclipped passes drive the policy into saturation, where, as in the two-action example, the gradient of the chosen action's log-probability is small; the next batches are then collected by the saturated policy, and they rarely contain the actions that would reverse it. The PPO paper's own ablation found the same: its variant with neither a clip nor a penalty was the worst of all those it tested :cite:`Schulman.Wolski.Dhariwal.ea.2017`. The large learning rate is what makes the clip necessary. At an actor learning rate of $10^{-3}$, ten times smaller, unclipped PPO no longer collapses on any of ten seeds and reaches the ceiling as fast as clipped PPO, and at $3 \times 10^{-4}$ the clip almost never acts. Both rates are slower than ours, however: they need about 35,000 to 40,000 and about 85,000 to 90,000 steps to the ceiling, against about 24,000 here. The large rate makes training fast, and the clip makes the large rate safe.

How often does the clip act? Each update checks every sample once per epoch, and the first epoch begins with $\rho_t=1$ for every sample. We report the fraction of all per-epoch checks outside the band and the same fraction at the last epoch of each batch:

```{.python .input #ppo-the-ablation-5}
%%tab pytorch, jax
for name, d in diag.items():
    print(f'{name:>13}: ratio checks outside the band: '
          f'{d[..., CLIPPED].mean():.1%} across all epochs')
    print(f'{"":>13}  {d[:, :, -1, CLIPPED].mean():.1%} at the last epoch '
          f'of each batch')
```

For the clipped runs both counts are near one check in twenty. In the unclipped control ratios leave the band several times as often, and more often at the last epoch of a batch than on average, because nothing stops them from moving further at every pass.

### Training Diagnostics
:label:`subsec_ppo_diagnostics`

:numref:`sec_deeprl` showed that the value of a policy-gradient loss says nothing about progress: it stays near zero while the return climbs from 20 to 500. What deserves watching are measurements of the update itself, and `diag` already holds them for every run, seed, update and epoch. We first average over the updates and look at one batch's twenty epochs:

```{.python .input #ppo-how-to-know-your-rl-is-broken-1}
%%tab pytorch, jax
clipped = diag['clipped (PPO)']
d2l.plot_curves({'ratio checks outside the band':
                 clipped[..., CLIPPED].mean(1)},
                xlabel='epoch within the batch', ylabel='fraction',
                x=np.arange(1, num_epochs + 1))
kl = np.median(clipped[..., KL].mean(1), axis=0)
print(f'approximate KL: peak {kl.max():.4f} at epoch {kl.argmax() + 1}, '
      f'{kl[-1]:.4f} at epoch {num_epochs}')
for name, d in diag.items():
    print(f'{name:>13}: approximate KL at the last epoch, median '
          f'{np.median(d[:, :, -1, KL]):.3f}')
```

Within a batch the fraction outside the band is exactly zero at the first epoch, where every ratio equals one. It rises to a peak around the fifth or sixth epoch and then falls to about half the peak by the twentieth. The fall comes from the one-sided minimum. A sample whose ratio has left the band on the losing side keeps the gradient of its unclipped term, which pulls the ratio back, while a sample past the band on the winning side gets no gradient and stays where it is. The approximate KL has the same shape and is about ten times smaller; the cell prints its peak. Without the clip the divergence keeps growing at every pass, and at the last epoch of a batch it is typically tens of times the clipped run's.

Across updates, the entropy of the clipped policies tells how decisive they have become:

```{.python .input #ppo-training-diagnostics-1}
%%tab pytorch, jax
d2l.plot_curves({name: d[..., ENTROPY].mean(-1) for name, d in diag.items()},
                xlabel='update', ylabel='policy entropy (nats)')
print(f'clipped: entropy {clipped[:, :5, :, ENTROPY].mean():.2f} over the '
      f'first five updates, {clipped[:, -5:, :, ENTROPY].mean():.2f} over the '
      f'last five')
```

The entropy of the clipped runs falls from about $0.65$ nats to about $0.25$: the policy becomes more decisive as it learns, and the entropy bonus slows this without stopping it. The unclipped runs lose their entropy much faster, which is the saturation that ends the runs that die.

The summary fractions hide how the ratios are distributed. We take two identical fresh agents at the start of training, where the drift within a batch is largest, give both the same batch with the same frozen advantages and log-probabilities, and take twenty passes with the clip on in one and off in the other, reading every ratio after each pass. A third twin, `collector`, collects the batch and stays as it is, the old policy of every ratio:

```{.python .input #ppo-how-to-know-your-rl-is-broken-2}
%%tab pytorch, jax
probe = {arm: cartpole_agent(2) for arm in arms}   # twins
collector = cartpole_agent(2)                     # a third twin, kept fresh
env = gym.make('CartPole-v1')
env.reset(seed=2)
batch = d2l.rollout(env, collector.act, batch_episodes,
                    np.random.default_rng(2))
adv = d2l.standardize(batch.gae(collector.value_np, gamma, lam))
logp_old = collector.log_prob_np(batch.obs, batch.act)
rhos = {}
for arm, surrogate in arms.items():
    R = []
    for _ in range(num_epochs):   # one pass at a time, ratios read after
        ppo_epochs(probe[arm], batch.obs, batch.act, adv, logp_old,
                   epsilon_clip, 1, entropy_coef, surrogate)
        R.append(np.exp(probe[arm].log_prob_np(batch.obs, batch.act)
                        - logp_old))
    rhos[arm] = np.array(R)
```

`d2l.plot_hist` (:numref:`sec_utils`) overlays the histograms of the two sets of ratios after the last pass and marks the band edges with dashed lines:

```{.python .input #ppo-how-to-know-your-rl-is-broken-3}
%%tab pytorch, jax
d2l.plot_hist({arm: r[-1] for arm, r in rhos.items()},
              xlabel=rf'ratio $\rho_t(\theta)$ after {num_epochs} passes',
              ylabel='ratio checks',
              vlines=(1 - epsilon_clip, 1 + epsilon_clip))
```

After twenty passes the clipped agent's ratios stay near one, with a minority beyond the dashed band edges. The unclipped ratios pile up near zero, and most of the others lie far above the band, up to about two or three. A ratio of two or three is $1/\pi_{\theta_{\textrm{old}}}(a)$ for an action that the fresh policy chose with probability a half or a third and the new policy chooses almost always; a ratio near zero belongs to the other action. How much is such a reweighted batch still worth? The *effective sample size* of $n$ importance weights answers this, $N_{\textrm{eff}} = 1 / \sum_j \bar{w}_j^2$, where $\bar{w}_j$ is the ratio of step $j$ divided by the sum of all the batch's ratios. It equals $n$ when all weights are equal and one when a single weight dominates.

```{.python .input #ppo-how-to-know-your-rl-is-broken-4}
%%tab pytorch, jax
ess = {}
for arm, r in rhos.items():
    w = r / r.sum(axis=1, keepdims=True)   # normalized weights per epoch
    ess[arm] = 1 / (w ** 2).sum(axis=1) / r.shape[1]
d2l.plot_curves(ess, xlabel='epoch within the batch',
                ylabel='effective sample size / n',
                x=np.arange(1, num_epochs + 1))
print(f'after {num_epochs} epochs the batch is worth '
      + ' vs '.join(f'{v[-1]:.0%} ({k})' for k, v in ess.items())
      + f' of its {len(batch)} steps')
```

With the clip the weights stay nearly equal through the twentieth epoch. Without it the effective sample size falls well below the clipped arm's, to about half the batch. Saturation explains the value. Suppose the new policy picks an action $a^*(s)$ with certainty. A sample at $s$ then has weight $1/\pi_{\theta_{\textrm{old}}}(a^* \mid s)$ if its action was $a^*$, which happens with probability $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$, and weight zero otherwise. The mean weight is one and the mean squared weight is $1/\pi_{\theta_{\textrm{old}}}(a^* \mid s)$, so as the batch grows $N_{\textrm{eff}}/n$ approaches the harmonic mean of $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$ over the visited states. For a fresh policy that is nearly uniform over two actions, that is about one half. To check this we need the probabilities of all actions, not only of the sampled one; `action_probs` returns them as a NumPy array, and the untrained twin `collector` is the collecting policy.

```{.python .input #ppo-the-implementation-2}
%%tab pytorch
def action_probs(ac, obs):
    """pi(.|s) for a numpy batch of states, as numpy."""
    with torch.inference_mode():
        return torch.softmax(ac.policy(torch.as_tensor(obs)), -1).numpy()
```

```{.python .input #ppo-the-implementation-2}
%%tab jax
def action_probs(ac, obs):
    """pi(.|s) for a numpy batch of states, as numpy."""
    return np.asarray(jax.nn.softmax(d2l.forward_np(ac.policy, obs), -1))
```

:begin_tab:`jax`
`action_probs` runs the policy through `d2l.forward_np` (:numref:`sec_utils`), the compiled forward pass that `act` uses.
:end_tab:

```{.python .input #ppo-training-diagnostics-2}
%%tab pytorch, jax
p_old = action_probs(collector, batch.obs)     # the collecting policy
p_new = action_probs(probe['no clip'], batch.obs)
a_star = p_new.argmax(axis=1)                  # the unclipped choice
p_star = p_old[np.arange(len(batch)), a_star]
print(f'unclipped policy: median probability of its choice '
      f'{np.median(p_new.max(axis=1)):.3f}')
print(f'harmonic mean of the old probability of that choice: '
      f'{1 / (1 / p_star).mean():.2f}')
```

The unclipped policy is close to deterministic, and the harmonic mean lies within about a tenth of the unclipped arm's effective sample size, slightly below it because the policy is not quite deterministic. The effective sample size counts how evenly the weights are spread; it ignores the advantages and the states the new policy would visit, so it complements the divergence and the clip fraction rather than replacing them.

Finally, a trained clipped agent, acting greedily on the most probable action to remove the noise of sampling, should balance the pole to the time limit:

```{.python .input #ppo-how-to-know-your-rl-is-broken-5}
%%tab pytorch, jax
ac, env = agents['clipped (PPO)'][0], gym.make('CartPole-v1')
env.reset(seed=2)
score = d2l.evaluate(env, ac.act_greedy, num_episodes=100)
print(f'greedy mean return over 100 episodes: {score:.0f}')
```

### Vectorized Collection and Minibatches
:label:`subsec_ppo_vectorized`

Our loop plays whole episodes in one environment and takes full-batch passes. Production implementations such as `cleanrl/ppo.py` :cite:`Huang.Dossa.Ye.ea.2022` step $N$ copies of the environment in parallel for a fixed horizon $T$, producing an $N\times T$ rectangle of transitions, and cut every episode that is still running at the rectangle's edge. At the cut the critic supplies the value of the unrecorded future, the same bootstrap that :numref:`sec_deeprl` used at CartPole's time limit. We collect one such rectangle with the trained agent, eight environments for 32 steps:

```{.python .input #ppo-from-a-teaching-loop-to-a-real-one-1}
%%tab pytorch, jax
envs = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 8)
obs, _ = envs.reset(seed=0)
rng, steps = np.random.default_rng(3), []
for _ in range(32):                       # the fixed 32 x 8 rectangle
    u = rng.random((len(obs), 1))         # inverse-CDF sampling, per env
    act = (u > action_probs(ac, obs).cumsum(1)[:, :-1]).sum(1)
    obs, rew, term, trunc, _ = envs.step(act)
    steps.append((rew, term | trunc))
rew, done = map(np.array, zip(*steps))
print(f'a fixed {rew.shape} rectangle of steps, '
      f'{int(done.sum())} episode boundaries inside it')
print(f'V at the cut, the value of the unrecorded future: '
      f'{ac.value_np(obs).round(0)}')
```

The trained agent does not fall within 32 steps, so all eight trajectories end at the cut, and their values are close to $1/(1-\gamma)=100$, the discounted value of balancing forever. Without the bootstrap, GAE would treat each cut as a fall and give the last actions before it large negative advantages. Gymnasium's vector environments also restart a finished episode by themselves: the step after an episode ends returns the first observation of a new episode with reward zero and ignores the action, so implementations mask that step out of the advantages and the loss. Finally, production code shuffles the rectangle into minibatches and takes a few epochs over them, typically four, so each ratio is checked four times rather than twenty; the smaller steps need less memory and add gradient noise. Exercise :exref:`ex_ppo-standard-recipe` builds this loop.

Vectorized collection also changes what a sample costs. This chapter counts environment steps, because on a robot arm every step is real. A physics simulator that runs on the GPU steps thousands of copies of a robot at once :cite:`Makoviychuk.Wawrzyniak.Guo.ea.2021`, so a rectangle thousands of environments wide arrives in a fraction of a second, and the cost that matters is wall-clock time. PPO suits this regime, since a very large batch makes each of its few updates accurate. With the simulator's parameters randomized so that the policy transfers to the real machine, a practice called *domain randomization* :cite:`Tobin.Fong.Ray.ea.2017`, legged robots learn to walk in minutes of training :cite:`Rudin.Hoeller.Reist.ea.2022`. This is why PPO remains the default there, although the methods of the following sections need far fewer steps. They win when the steps are real.

PPO lets an agent take several steps on each batch, and on CartPole four epochs reach the ceiling in half the environment steps of one. Its clip limits how far each step moves the action probabilities, but it bounds the path, not the destination. The entropy of the clipped runs still falls from $0.65$ to $0.25$ nats, because the objective is still the return with a small entropy bonus, whose optimum is nearly deterministic. :numref:`sec_regularized` changes the objective instead: it keeps the policy near a fixed reference, which moves the optimum and not only the path.

## Summary

An on-policy batch of thousands of environment steps should inform more than one gradient step, but equal steps in parameter space can change a policy by very different amounts. Importance sampling evaluates a new policy on old data exactly, with a variance that grows exponentially with the episode length; keeping one ratio per step removes that growth and evaluates the new policy's actions at the old policy's states instead. The performance difference lemma shows that this substitution is the surrogate's only error, and TRPO bounds it by constraining the divergence between old and new action distributions. PPO instead clips each sample's contribution to the objective once its ratio leaves a band around one, which lets a few epochs reuse each batch: on CartPole four epochs reach the ceiling in about half the environment steps of one, and the clip keeps a large learning rate from collapsing the policy. The clip fraction, the approximate KL, the entropy and the effective sample size of the ratios show how far each update moved the policy.

## Exercises

1. **Step limits for one logit.** ● Take the two-action policy
   $\pi_\theta(a{=}1) = \sigma(\theta)$ of
   :numref:`fig_rl-policy-vs-parameter` and three ways of limiting an update
   $\theta \to \theta + \Delta$: a trust region on the divergence, PPO's clip,
   and a cap on $|\Delta|$.
    1. Expand $D_{\textrm{KL}}\big(\pi_\theta \Vert \pi_{\theta+\Delta}\big)$
       to second order in $\Delta$ and show that it equals
       $\frac{1}{2}\sigma'(\theta)\,\Delta^2$. Compute the exact divergence of
       the two updates with $\Delta = 2$ drawn in the figure, from
       $\theta = 0$ and from $\theta = 6$. Which of them stays inside a trust
       region with $\Delta_{\textrm{KL}} = 0.02$?
    1. Use the second-order form to find the largest $|\Delta|$ that this
       trust region permits at $\theta = 0$ and at $\theta = 6$. At
       $\theta = 6$, compute the exact divergence of the two steps of that
       size, one in each direction. Why is the second-order form much less
       accurate there than at $\theta = 0$?
    1. Now let the clip of :eqref:`eq_ppo_clip` with
       $\epsilon_{\textrm{clip}} = 0.2$ limit the step, for one sample with
       positive advantage. The update raises the probability of the sampled
       action, so $\Delta > 0$ when the sample is $a = 1$ and $\Delta < 0$
       when it is $a = 2$. Find the largest $|\Delta|$ over which the clipped
       objective still increases, for both actions at $\theta = 0$ and at
       $\theta = 6$.
    1. Which of the three limits permits large parameter steps where the
       policy is insensitive to $\theta$ and only small ones where it is
       sensitive? At $\theta = 6$, for which sampled action is the clip
       looser than the trust region, and for which is it tighter?
   :exlabel:`ex_ppo-step-limits`

1. **Exploding product of ratios.** A policy that ignores the state collects
   episodes with $\pi_{\theta_{\textrm{old}}}$ and the data are used to
   evaluate $\pi_\theta$. Write $W_T$ for the trajectory ratio
   :eqref:`eq_traj_ratio` of an episode of $T$ steps.
    1. Show that $\mathbb{E}[W_T] = 1$ and
       $\textrm{Var}[W_T] = (1 + \chi^2)^T - 1$ under the old policy, where
       $\chi^2 = \sum_a \big(\pi_\theta(a) - \pi_{\theta_{\textrm{old}}}(a)\big)^2 / \pi_{\theta_{\textrm{old}}}(a)$.
       Check the value $1.04^T - 1$ of the text for
       $\pi_{\theta_{\textrm{old}}} = (0.5, 0.5)$ and $\pi_\theta = (0.6, 0.4)$.
    1. As the number $n$ of episodes grows, the effective sample size
       of the weights $W_T$ approaches $n / \mathbb{E}[W_T^2]$. For the two
       policies of (a), for which $n$ does this limit equal one at
       $T = 200$? Simulate one batch of that many episodes (the number of
       steps with $a = 1$ in an episode is binomial) and compute its
       effective sample size. Why does the batch report a much larger
       effective sample size than the limit? Which of the two describes the
       variance of the importance-weighted estimate?
    1. The per-step surrogate :eqref:`eq_surrogate` weights each step by a
       single ratio $\rho_t$. What is the variance of that weight here, and
       which approximation in :eqref:`eq_surrogate` makes it independent of
       $T$?
   :exlabel:`ex_ppo-exploding-product`

1. **What the diagnostics estimate.** ● Fix a state $s$, draw
   $a \sim \pi_{\theta_{\textrm{old}}}(\cdot \mid s)$, and let
   $\rho = \pi_\theta(a \mid s) / \pi_{\theta_{\textrm{old}}}(a \mid s)$.
    1. The approximate KL of `ppo_epochs` averages $k_1 = -\log \rho$ over the
       batch. Show that
       $\mathbb{E}[k_1] = D_{\textrm{KL}}\big(\pi_{\theta_{\textrm{old}}}(\cdot \mid s) \Vert \pi_\theta(\cdot \mid s)\big)$.
       Then show that $k_3 = (\rho - 1) - \log \rho$ has the same expectation
       and is nonnegative for every sample. Expand both estimators to second
       order in $\rho - 1$. Why does $k_3$ have a much smaller variance than
       $k_1$ when the policy changes little? `cleanrl/ppo.py` logs both.
    1. Problem :exref:`ex_ppo-exploding-product`b uses the limit
       $n / \mathbb{E}[w^2]$ of the effective sample size of $n$ independent
       weights $w$ with $\mathbb{E}[w] = 1$. Derive this limit with the law of
       large numbers. Then show that $\mathbb{E}[\rho^2] = 1 + \chi^2$, so
       that the effective sample size of the weights $\rho$ divided by $n$
       approaches $1 / (1 + \chi^2)$, where

       $$\chi^2 = \sum_a \frac{\big(\pi_\theta(a \mid s) - \pi_{\theta_{\textrm{old}}}(a \mid s)\big)^2}{\pi_{\theta_{\textrm{old}}}(a \mid s)}.$$

       Show that $\chi^2 \approx 2 D_{\textrm{KL}}$ to second order in the
       policy change.
    1. With states drawn from a batch, $\chi^2$ becomes its average over the
       visited states. Which values of $\chi^2$ do the two fractions
       `ess[name][-1]` of :numref:`subsec_ppo_diagnostics` imply? Which
       approximate KL does the second-order relation of (b) predict for each?
       Compare each prediction with the $k_1$ estimate of the same ratios,
       `(-np.log(rhos[arm][-1])).mean()` for each `arm`. For
       which agent does the second-order relation fail? Besides the size of
       the policy change, which assumption of (b) do ratios violate that are
       measured on the very batch the policy was trained on?

   *Adapted from John Schulman,
   [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html),
   estimators $k_1$ and $k_3$.*
   :exlabel:`ex_ppo-diagnostics-estimate`

1. [code] **KL early stopping.** ● Besides clipping, an implementation can
   stop reusing a batch once the policy has moved far enough. Spinning Up's
   PPO computes the approximate KL before every pass and ends the passes on a
   batch once it exceeds $1.5$ times `target_kl`, which defaults to $0.01$.
    1. Replace the `ppo_epochs` call in `train_ppo` by a loop of single
       passes, `ppo_epochs(..., 1, ...)`, as the loop that collects `rhos` in
       :numref:`subsec_ppo_diagnostics` does. Before each pass, compute the
       approximate KL from `log_prob_np` and apply the stopping rule. Record
       the number of passes spent on each batch.
    1. Run the unclipped variant with early stopping on the eight seeds of
       the ablation. Does the stopping rule alone prevent the collapse? How
       does the number of passes per batch change over training?
    1. Run the clipped variant with early stopping as well. Which of the two
       mechanisms limits the policy change of a batch in this run? What does
       the answer say about describing the clip as a trust region?

   *Adapted from Joshua Achiam,
   [Spinning Up in Deep RL](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py),
   PPO implementation, early stopping on `target_kl`.*

1. [code] **The standard recipe.** ●● Build the loop that
   :numref:`subsec_ppo_vectorized` describes: eight environments in a
   `gym.vector.SyncVectorEnv` stepped for 32 steps per update, GAE over the
   $32 \times 8$ rectangle with the critic's value at each cut and the
   autoreset step masked out, then four epochs over the rectangle in
   shuffled minibatches of 64, each minibatch taking one `ppo_epochs` pass
   with its own standardized advantages and one `fit_value` step toward the
   $\lambda$-return target computed before the epochs. Use a learning rate of
   $10^{-3}$ for both networks.
    1. Train three seeds and record the environment steps until the last
       eight completed episodes average at least 475. A correct
       implementation needs about 45,000 steps, between 35,000 and 55,000 on
       most seeds: more than the four-epoch arm of the reuse experiment in
       :numref:`subsec_ppo_reuse`, which needs about 25,000.
    1. Raise the critic's learning rate to $10^{-2}$, the rate of our loop,
       and keep the actor's at $10^{-3}$. A correct implementation now
       reaches 475 in about 15,000 to 20,000 steps, fewer than any arm of
       the reuse experiment.
       Which network limited the run of (a)? Our loop at the same two rates,
       without vectorization or minibatches, needs about 35,000 to 40,000
       steps. What do the short rectangles change about how often the
       policy is updated per environment step?
    1. Remove the mask on the autoreset step. Does the run still learn?
       Which transitions does the unmasked version add to the critic's
       targets, and why does CartPole barely notice them?
   :exlabel:`ex_ppo-standard-recipe`

1. **The monotonic-improvement bound.** ●● This problem derives
   :eqref:`eq_trpo_bound` by the coupling argument of
   :citet:`Schulman.Levine.Abbeel.ea.2015`. Take an infinite-horizon task
   with $\gamma < 1$ and write $\pi = \pi_{\theta_{\textrm{old}}}$,
   $\tilde{\pi} = \pi_\theta$, $A = A^{\textrm{old}}$, and
   $\bar{A}(s) = \sum_a \tilde{\pi}(a \mid s)\, A(s, a)$. Let
   $\Delta_{\textrm{TV}} = \max_s D_{\textrm{TV}}\big(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s)\big)$
   with the total variation distance
   $D_{\textrm{TV}}(p, q) = \frac{1}{2} \sum_a |p(a) - q(a)|$.
    1. Show from :eqref:`eq_perf_diff` that
       $J(\theta) - J(\theta_{\textrm{old}}) = \sum_t \gamma^t \mathbb{E}_{s_t \sim \tilde{\pi}}\big[\bar{A}(s_t)\big]$,
       and that the population surrogate :eqref:`eq_pop_surrogate` is
       $\bar{L}(\theta) = \sum_t \gamma^t \mathbb{E}_{s_t \sim \pi}\big[\bar{A}(s_t)\big]$,
       where $s_t \sim \pi$ denotes the state distribution at time $t$ under
       $\pi$.
    1. Two distributions at total variation distance $\Delta_{\textrm{TV}}$
       can be coupled: there is a joint distribution of $(a, \tilde{a})$ with
       marginals $\pi(\cdot \mid s)$ and $\tilde{\pi}(\cdot \mid s)$ under
       which $P(a \neq \tilde{a}) \leq \Delta_{\textrm{TV}}$ at every state.
       Use $\mathbb{E}_{a \sim \pi}[A(s, a)] = 0$ to show that
       $|\bar{A}(s)| \leq 2\Delta_{\textrm{TV}} A_{\max}$.
    1. Run both policies from the same start state with coupled actions at
       every step, and let $m_t$ count the steps before $t$ at which the two
       actions differ. On the event $m_t = 0$, the two trajectories coincide
       up to time $t$. Show that

       $$\big|\mathbb{E}_{s_t \sim \tilde{\pi}}[\bar{A}(s_t)] - \mathbb{E}_{s_t \sim \pi}[\bar{A}(s_t)]\big| \leq 4\Delta_{\textrm{TV}} \big(1 - (1 - \Delta_{\textrm{TV}})^t\big) A_{\max}.$$
    1. Sum over $t$ to obtain

       $$\big|J(\theta) - J(\theta_{\textrm{old}}) - \bar{L}(\theta)\big| \leq \frac{4 \Delta_{\textrm{TV}}^2 \gamma A_{\max}}{(1-\gamma)^2},$$

       and finish the proof with $D_{\textrm{TV}}^2 \leq D_{\textrm{KL}}$.
    1. Evaluate the coefficient $4\gamma / (1-\gamma)^2$ at $\gamma = 0.99$.
       Use (b) to show that
       $\bar{L}(\theta) \leq 2\Delta_{\textrm{TV}} A_{\max} / (1-\gamma)$.
       With Pinsker's inequality $D_{\textrm{KL}} \geq 2 D_{\textrm{TV}}^2$,
       show that the bound certifies an improvement only if
       $\Delta_{\textrm{TV}} < (1-\gamma) / (4\gamma)$. How large can
       $\bar{L}(\theta)$ of a certified step then be, in units of $A_{\max}$?
       Compare this limit on $\Delta_{\textrm{TV}}$ with the total variation
       $\sqrt{\Delta_{\textrm{KL}} / 2}$ that Pinsker's inequality permits at
       a state whose divergence equals the budget
       $\Delta_{\textrm{KL}} = 0.01$ that
       :citet:`Schulman.Levine.Abbeel.ea.2015` use in all their experiments.
   :exlabel:`ex_ppo-monotonic-bound`

1. [extended] **TRPO from the pieces.** ●● Replace the `ppo_epochs` call in
   `train_ppo` by one TRPO update :cite:`Schulman.Levine.Abbeel.ea.2015` on
   the constrained problem :eqref:`eq_trpo_constraint` with
   $\Delta_{\textrm{KL}} = 0.01$, and leave the critic and the advantages
   unchanged. Treat the parameters of `ac.policy` as one flat vector, and
   freeze the collecting policy's action probabilities,
   `action_probs(ac, b.obs)`, together with `logp_old`.
    1. Compute the gradient $\mathbf{g}$ of the surrogate :eqref:`eq_surrogate`
       at $\theta_{\textrm{old}}$. Write a function that returns the
       Fisher-vector product $F\mathbf{v}$, the Hessian of the mean
       divergence over the batch's states at $\theta_{\textrm{old}}$ applied
       to $\mathbf{v}$, by differentiating the gradient of that divergence a
       second time. Check it against a finite difference of the gradient.
    1. Solve $F\mathbf{x} = \mathbf{g}$ approximately by ten iterations of
       conjugate gradient, and scale the step to $\xi\mathbf{x}$ with
       $\xi = \sqrt{2\Delta_{\textrm{KL}} / \mathbf{x}^\top F \mathbf{x}}$.
       Halve $\xi$ until the measured mean divergence is at most
       $\Delta_{\textrm{KL}}$ and the surrogate has increased.
    1. Train three seeds and compare them with the clipped runs of
       :numref:`subsec_ppo_ablation`: learning curves, the measured divergence
       of each update, and the number of halvings. How many gradient and
       Fisher-vector computations does each method spend per batch?
    1. :citet:`Engstrom.Ilyas.Santurkar.ea.2020` found that PPO and TRPO
       perform similarly when their code-level choices are matched. Here the
       two share every choice except the policy update. Does your comparison
       agree?
   :exlabel:`ex_ppo-trpo`

[Discussions](https://d2l.discourse.group/t/36802)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.2]{.kicker}

Trust Regions and Proximal Policy Optimization<br>**parameter vs. policy distance · importance sampling · performance difference lemma · TRPO · PPO clipping · batch diagnostics**
:::
:::

::: {.slide title="Why One Gradient Step Wastes a 4,000-Step Batch"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Under-Extracted Information:** Thousands of transitions contain far more signal than a single first-order step can absorb.
- **Why Naive Multi-Epoch Reuse Fails:** Once $\theta$ updates even once, the batch comes from an older policy $\pi_{\theta_{\textrm{old}}}$, and repeated unguarded passes drive action probabilities into irreversible collapse.
:::

::: {.col .fig .big}
@fig:ch15-drl-single-vs-multi-epoch
:::
:::
:::

::: {.slide title="Parameter Distance Does Not Control Policy Distance"}
[Geometry of Policy Space]{.kicker}

::: {.cols .vc}
::: {.col}
- Consider a 1-parameter Bernoulli policy $\pi_\theta(a{=}1) = \sigma(\theta) = (1 + e^{-\theta})^{-1}$ under two equal parameter steps $\Delta \theta = 2$ (Achiam, 2017):
  - At $\theta = 0$ (slope $\sigma'(0) = 0.25$): $\pi(a{=}1)$ jumps from **$0.50 \to 0.88$**.
  - At $\theta = 6$ (slope $\sigma'(6) \approx 0.0025$): $\pi(a{=}1)$ barely shifts from **$0.9975 \to 0.9997$**.
- **Saturation Trap:** At $\theta = 6$, the score $\nabla_\theta \log \pi_\theta(a{=}1) = 1 - \sigma(6) \approx 0.0025$ vanishes while $a{=}2$ is sampled only once in $400$ trials, trapping on-policy learners.
:::

::: {.col .fig}
![](../img/mdl-rl-policy-vs-parameter.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Change of Measure: Evaluating New Policies on Old Data"}
[Importance Sampling]{.kicker}

::: {.cols .vc}
::: {.col}
$$J(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})}\Big[\frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, R(\tau)\Big]$$

- **Physics cancellation:** Initial distribution $\mu_0(s_0)$ and transition dynamics $P(s_{t+1} \mid s_t, a_t)$ cancel completely in $P(\tau; \theta) / P(\tau; \theta_{\textrm{old}})$, leaving $\prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}$.
:::

::: {.col .fig}
@fig:ch15-drl-importance-cancellation
:::
:::
:::

::: {.slide title="Why Whole-Trajectory Importance Weights Explode"}
[The Curse of Horizon]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exponential variance growth:** Even when each step ratio $\rho_t \in \{0.8, 1.2\}$ has tiny variance $0.04$, the product $W_T = \prod_{t=0}^{T-1} \rho_t$ has variance $\textrm{Var}(W_T) = 1.04^T - 1$.
- **500-step CartPole collapse:** At $T = 500$, $\textrm{Var}(W_T) = 3.3 \times 10^8$, so a single trajectory dominates the batch; PPO instead keeps only the single-step ratio $\rho_t(\theta)$.
:::

::: {.col .fig .big}
@fig:ch15-drl-ratio-explosion
:::
:::
:::

::: {.slide title="The Per-Step Surrogate Objective"}
[Local Approximation]{.kicker}

To eliminate the exponential horizon growth, we drop the historical product $\rho_0 \cdots \rho_{t-1}$ (which corrects the state distribution at time $t$) and keep only the **single-step probability ratio** $\rho_t(\theta)$:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}, \qquad \hat{L}(\theta) = \frac{1}{N} \sum_{i=1}^n \sum_{t=0}^{T_i-1} \rho_t^i(\theta)\, \hat{A}_t^i$$

- **First-Order Consistency:** At $\theta = \theta_{\textrm{old}}$, every ratio is $\rho_t(\theta_{\textrm{old}}) = 1$ and $\nabla_\theta \rho_t(\theta_{\textrm{old}}) = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, so $\nabla_\theta \hat{L}(\theta_{\textrm{old}})$ equals the standard policy gradient.
- **What Was Dropped:** The surrogate $\hat{L}(\theta)$ evaluates the *new* policy's action probabilities at the *old* policy's state visitation distribution.
:::

::: {.slide title="The Performance Difference Lemma"}
[Exact Return Decomposition]{.kicker}

::: {.cols .vc}
::: {.col}
$$J(\theta) - J(\theta_{\textrm{old}}) = \frac{1}{1-\gamma} \sum_s d^{\pi_\theta}_\gamma(s) \sum_a \pi_\theta(a \mid s)\, A^{\textrm{old}}(s, a)$$

- **Telescoping proof (Kakade & Langford, 2002):** Summing $\gamma^t(r_t + \gamma V^{\textrm{old}}(s_{t+1}) - V^{\textrm{old}}(s_t))$ telescopes to $R(\tau) - V^{\textrm{old}}(s_0)$.
- **Surrogate error:** Population surrogate $\bar{L}(\theta)$ simply replaces new state occupancy $d^{\pi_\theta}_\gamma(s)$ with old $d^{\pi_{\theta_{\textrm{old}}}}_\gamma(s)$.
:::

::: {.col .fig}
@fig:ch15-drl-perf-diff-lemma
:::
:::
:::

::: {.slide title="Monotonic Improvement and Trust Region Bounds (TRPO)"}
[From Theory to TRPO]{.kicker}

::: {.cols .vc}
::: {.col}
- **Kakade–Langford / Schulman Bound:** The gap between true improvement $J(\theta) - J(\theta_{\textrm{old}})$ and $\bar{L}(\theta)$ is bounded by the worst-case KL divergence across states:
  $$J(\theta) \ge J(\theta_{\textrm{old}}) + \bar{L}(\theta) - \frac{4\gamma A_{\max}}{(1-\gamma)^2} \max_s D_{\textrm{KL}}(\pi_{\theta_{\textrm{old}}} \Vert \pi_\theta)$$
- **TRPO (Schulman et al., 2015):** Because $4\gamma/(1-\gamma)^2 \approx 40{,}000$ at $\gamma=0.99$ is too conservative, TRPO maximizes $\hat{L}(\theta)$ subject to a mean KL constraint $\bar{D}_{\textrm{KL}} \le \Delta_{\textrm{KL}} = 0.01$ via conjugate gradients.
:::

::: {.col .fig}
![](../img/mdl-rl-trust-region.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="PPO: First-Order Clipping in Policy Ratio Space"}
[Proximal Policy Optimization]{.kicker}

::: {.cols .vc}
::: {.col}
- **PPO (Schulman et al., 2017)** replaces TRPO's second-order Fisher-vector constraint with a first-order clipped surrogate ($\epsilon_{\textrm{clip}} = 0.2$):
  $$L^{\textrm{CLIP}}(\theta) = \frac{1}{N} \sum_{i,t} \min\Big(\rho_t^i \hat{A}_t^i,\ \textrm{clip}(\rho_t^i, 1-\epsilon_{\textrm{clip}}, 1+\epsilon_{\textrm{clip}})\hat{A}_t^i\Big)$$
- **Pessimistic Lower Bound:** When $\hat{A}_t > 0$, the objective flattens above $\rho_t = 1 + \epsilon_{\textrm{clip}}$ (zero gradient); when $\hat{A}_t < 0$, it flattens below $1 - \epsilon_{\textrm{clip}}$, removing the incentive to push ratios outside $[0.8, 1.2]$.
:::

::: {.col .fig}
![](../img/mdl-rl-ppo-clip.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Why Ratios Still Leave the Band—and the Entropy Bonus"}
[Optimization Nuances]{.kicker}

::: {.cols .vc}
::: {.col}
- **Shared Network Coupling & Momentum:** Unclipped transitions in a batch move shared weights $\theta$ for already-clipped transitions, and Adam's first-moment accumulator $m_t$ continues coasting.
- **Losing-Side Slope & Entropy Bonus (`0.01`):** Samples crossing the losing side ($\rho < 0.8$ for $\hat{A} > 0$) retain their restoring slope, while $+\beta_H H(\pi_\theta)$ keeps probabilities away from $0$ and $1$.
:::

::: {.col .fig .big}
@fig:ch15-drl-clip-drift-entropy
:::
:::
:::

::: {.slide title="Implementing Clipped and Unclipped Surrogates"}
[Code Primitives]{.kicker}

We implement both the clipped PPO objective and the unclipped importance-weighted objective ($\rho_t \hat{A}_t$) so our ablation can compare them under identical conditions:

@ppo-the-implementation-3

::: {.d2l-note}
In our CartPole setup (`ActorCritic.mlp(4, 2)` with 64-unit `tanh` layers and Adam learning rate $10^{-2}$), we use GAE with $\gamma = 0.99$, $\lambda = 0.95$, `critic_steps = 20`, and `num_epochs = 20` actor passes per 8-episode batch.
:::
:::

::: {.slide title="Multi-Epoch Actor Updates and Live Diagnostics" layout="tight" only="pytorch"}
[The Inner Epoch Loop]{.kicker}

Before each of the `num_epochs` passes over the frozen batch `(obs, act, adv, logp_old)`, `ppo_epochs` logs three diagnostic scalars—the **clip fraction**, the **approximate KL** $\frac{1}{N}\sum (\log \pi_{\theta_{\textrm{old}}} - \log \pi_\theta)$, and the **policy entropy**:

@-ppo-the-implementation-4[1-9]
:::

::: {.slide title="Multi-Epoch Actor Updates and Live Diagnostics (cont.)" layout="tight" only="pytorch"}
[The Inner Epoch Loop]{.kicker}

Each pass steps the actor on the mean clipped surrogate plus an entropy bonus and records the three diagnostics:

@ppo-the-implementation-4[10-]
:::

::: {.slide title="Multi-Epoch Actor Updates and Live Diagnostics" layout="tight" only="jax"}
[The Inner Epoch Loop]{.kicker}

Before each of the `num_epochs` passes over the frozen batch `(obs, act, adv, logp_old)`, `ppo_epochs` logs three diagnostic scalars—the **clip fraction**, the **approximate KL** $\frac{1}{N}\sum (\log \pi_{\theta_{\textrm{old}}} - \log \pi_\theta)$, and the **policy entropy**:

@-ppo-the-implementation-4[1-14]
:::

::: {.slide title="Multi-Epoch Actor Updates and Live Diagnostics (cont.)" layout="tight" only="jax"}
[The Inner Epoch Loop]{.kicker}

One compiled step returns the diagnostics measured before it updates; `ppo_epochs` runs `num_epochs` of them on the padded batch:

@ppo-the-implementation-4[15-]
:::

::: {.slide title="The Complete PPO Training Loop"}
[Outer Loop]{.kicker}

Each outer iteration collects 8 episodes, freezes the standardized GAE advantages (evaluated with the pre-fit critic) and old log-probabilities, refits the critic for 20 steps toward the $\lambda$-return, and runs `ppo_epochs`:

@-ppo-the-implementation-5[1-10]
:::

::: {.slide title="The Complete PPO Training Loop (cont.)"}
[Outer Loop]{.kicker}

Each iteration: roll out, freeze advantages and old log-probabilities, refit the critic, then run the actor epochs:

@ppo-the-implementation-5[11-]
:::

::: {.slide title="Batch Reuse Halves the Environment Steps Required"}
[Sample Efficiency]{.kicker}

::: {.cols .vc}
::: {.col}
- Across 8 seeds capped at $50{,}000$ environment steps, taking **4 epochs** per batch reaches the return ceiling ($475+$) in **$\sim 25{,}000$ steps**—half the steps and half the rollouts needed by 1 epoch ($\ge 50{,}000$ steps).
- At 1 epoch per batch, every ratio is $\rho_t = 1$ and the clip never activates, reducing PPO to GAE actor-critic.
- Increasing reuse from 4 to 20 epochs yields only minor additional speedups while remaining completely stable under clipping.
:::

::: {.col .fig}
@!ppo-reuse-against-environment-steps-2
:::
:::
:::

::: {.slide title="Ablation: What Happens Without the Clip at 20 Epochs?"}
[The Clipping Ablation]{.kicker}

::: {.cols .vc}
::: {.col}
- At learning rate $10^{-2}$ and 20 epochs per batch, **every single clipped PPO seed (8/8)** reaches 500 and holds the task above 250.
- **Without the clip**, repeated unguarded passes drive the policy into saturation: over 64 seeds, only $\sim 1$ in $20$ holds the task and **half end dead** at return $\sim 9$ (pushing the cart one way until the pole falls).
- Lowering the actor learning rate $10\times$ to $10^{-3}$ prevents unclipped collapse but slows convergence from $24\text{k}$ to $\sim 40\text{k}$ steps.
:::

::: {.col .fig}
@!ppo-the-ablation-4

@!ppo-the-ablation-3
:::
:::
:::

::: {.slide title="Monitoring PPO: Clip Fraction and Approximate KL"}
[Training Diagnostics]{.kicker}

::: {.cols .vc}
::: {.col}
- Because policy-gradient loss stays near zero regardless of performance, we monitor the **clip fraction** and **approximate KL** across the 20 epochs inside a batch.
- At epoch 1, $\rho_t = 1$ so both are $0$. In clipped PPO, the out-of-band fraction peaks around epoch 5–6 ($\sim 8$–$10\%$) and **falls back by half** by epoch 20.
- **Why does it fall?** Samples that cross the band on the *losing* side ($\rho < 0.8$ for $\hat{A} > 0$) retain their unclipped gradient and are pulled back inside.
:::

::: {.col .fig}
@!ppo-how-to-know-your-rl-is-broken-1
:::
:::
:::

::: {.slide title="Policy Entropy Decay: Clipped vs. Unclipped Runs"}
[Saturation Diagnostics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Controlled Decisiveness:** Clipped PPO gradually reduces policy entropy from $0.65$ to $0.25\text{ nats}$ over 60 updates as it masters CartPole, slowed by the `entropy_coef = 0.01` bonus.
- **Unclipped Saturation:** Without clipping, policy entropy plummets rapidly toward $0\text{ nats}$ as repeated passes push action probabilities to $0$ and $1$.
:::

::: {.col .fig}
@!ppo-training-diagnostics-1
:::
:::
:::

::: {.slide title="Ratio Collapse and Effective Sample Size" layout="tight"}
[Importance Weight Diagnostics]{.kicker}

::: {.cols .vc}
::: {.col}
- After 20 passes on the same initial batch, clipped ratios stay clustered near $1.0$, whereas unclipped ratios pile up near $0$ and $2 \approx 1/\pi_{\theta_{\textrm{old}}}(a^*)$.
:::

::: {.col .fig}
@!ppo-how-to-know-your-rl-is-broken-3
:::
:::
:::

::: {.slide title="Ratio Collapse and Effective Sample Size (cont.)" layout="tight"}
[Importance Weight Diagnostics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Effective Sample Size ($N_{\textrm{eff}} = 1 / \sum_j \bar{w}_j^2$):** Clipped PPO retains **$N_{\textrm{eff}}/N \approx 96\%$–$97\%$** of the batch at epoch 20, while unclipped saturation cuts $N_{\textrm{eff}}/N$ to roughly half ($\approx 47\%$–$55\%$, near the harmonic mean of $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$).
:::

::: {.col .fig}
@!ppo-how-to-know-your-rl-is-broken-4
:::
:::
:::

::: {.slide title="Production PPO: Vectorized Rollouts and Cut Bootstraps"}
[Systems Architecture]{.kicker}

Production implementations (`cleanrl/ppo.py`) step $N = 8$ parallel environments for a fixed horizon $T = 32$ to form a $32 \times 8$ rollout rectangle, bootstrap at the edge, and train on shuffled minibatches:

@!ppo-from-a-teaching-loop-to-a-real-one-1

::: {.d2l-note .warn}
**Two production pitfalls:** (1) Unfinished episodes at step $T$ must bootstrap $\hat{V}(s_T) \approx 100$ rather than zero; (2) Gymnasium's `SyncVectorEnv` auto-resets on the step *after* termination, so that transition must be masked out of GAE.
:::
:::

::: {.slide title="Concept Check: Clipping Thresholds and Truncation Cuts"}
[Check Your Understanding]{.kicker}

Test your intuition on three concrete calculations:

1. **First Pass Identity:** On epoch 1 of a fresh batch where advantages $\hat{A}_t$ are standardized to mean $0$ and unit variance, what are $\rho_t(\theta_{\textrm{old}})$, $L^{\textrm{CLIP}}(\theta_{\textrm{old}})$, and $\nabla_\theta L^{\textrm{CLIP}}(\theta_{\textrm{old}})$?
2. **Relative Band Width:** With $\epsilon_{\textrm{clip}} = 0.2$ and $\hat{A}_t > 0$, up to what new probability $\pi_\theta(a_t \mid s_t)$ does the objective reward increasing an action that had $\pi_{\theta_{\textrm{old}}} = 0.01$ versus $\pi_{\theta_{\textrm{old}}} = 0.60$?
3. **Forgotten Cut Bootstrap:** At a 32-step rollout cut where $\hat{V}(s_T) = 100$, $\gamma = 0.99$, and $\lambda = 0.95$, how much does the GAE advantage drop if the cut is mistakenly masked as terminal ($\hat{V}(s_T) = 0$) at step $T-1$ and at step $T-11$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Every $\rho_t = 1$, so $L^{\textrm{CLIP}}(\theta_{\textrm{old}}) = \frac{1}{N}\sum \hat{A}_t = 0$, and $\nabla_\theta L^{\textrm{CLIP}}(\theta_{\textrm{old}}) = \frac{1}{N}\sum \hat{A}_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ (the exact single-step policy gradient).
2. Up to $1.2 \times 0.01 = \mathbf{0.012}$ ($+0.002$ absolute gain) for the rare action, versus $1.2 \times 0.60 = \mathbf{0.72}$ ($+0.12$ gain) for the common action.
3. At step $T-1$ it drops by $\gamma \hat{V}(s_T) = \mathbf{99}$; ten steps earlier ($T-11$) it drops by $(\gamma\lambda)^{10} \cdot 99 \approx \mathbf{54}$, falsely punishing good actions before the cut.
:::
:::

::: {.slide title="Summary: Policy Optimization From REINFORCE to PPO"}
[Key Takeaways]{.kicker}

| Method | Data Reuse per Batch | Step Constraint Mechanism | Optimization Cost | Behavior on CartPole ($50\text{k}$ Budget) |
| :--- | :--- | :--- | :--- | :--- |
| **Actor-Critic + GAE** | $1$ pass ($\rho_t = 1$) | Optional gradient norm clip | $1$ first-order step | Reaches $\sim 450$ near $50{,}000$ steps |
| **Unclipped Importance Ratio** | $K$ passes on $\rho_t \hat{A}_t$ | None (unguarded ratio) | $K$ first-order steps | Collapses to return $\sim 9$ at $K=20, \eta=10^{-2}$ |
| **TRPO** | $1$ line-searched step | Hard mean KL $\bar{D}_{\textrm{KL}} \le \Delta_{\textrm{KL}}$ | Conjugate gradient ($F\mathbf{x}=\mathbf{g}$) | Monotonic progress, higher per-update compute |
| **Clipped PPO** | $4$–$20$ epochs on $L^{\textrm{CLIP}}$ | One-sided ratio clip $1 \pm \epsilon_{\textrm{clip}}$ | $K$ first-order Adam steps | Solves CartPole ($475+$) in $\sim 25{,}000$ steps |
:::
