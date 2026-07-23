```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('pytorch')
```

# Trust Regions and Proximal Policy Optimization
:label:`sec_ppo`

The methods of the last three sections share a trait that no earlier algorithm in this chapter had: they parameterize the policy directly. Value Iteration and Q-Learning adjusted value estimates and read the policy off them; REINFORCE and actor-critic adjust $\theta$, and the robot's behavior *is* $\pi_\theta$. That raises a question gradient ascent cannot answer on its own, because gradient ascent controls the size of the step in parameters, while what we care about is the size of the step in the policy. This section opens with an example in which the two come apart completely. The answer to that question, together with a way to reuse each batch of trajectories instead of discarding it after one update, gives the two ideas of this section: trust regions :cite:`Schulman.Levine.Abbeel.ea.2015` and the clipped objective of Proximal Policy Optimization :cite:`Schulman.Wolski.Dhariwal.ea.2017`, PPO for short. PPO is the workhorse of modern policy optimization; when we discuss how language models are trained from human feedback later in the book, this is the algorithm running underneath.

## Parameter Space versus Policy Space

Since we now update the parameters of the policy directly, we should ask what a parameter update does to the policy itself. A family of policies with a single parameter $\theta$ and two actions answers it; we borrow the example from Joshua Achiam's lectures on policy optimization :cite:`Achiam.2017`. Let

$$\pi_\theta(a) = \begin{cases} \sigma(\theta) & a = 1 \\ 1 - \sigma(\theta) & a = 2, \end{cases} \qquad \textrm{where } \sigma(\theta) = \frac{1}{1 + e^{-\theta}}.$$

Take two parameter updates of exactly the same size, $\Delta\theta = 2$, from two different starting points:

$$\pi_{\theta=0}(a{=}1) = \sigma(0) = 0.50 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(2) = 0.88, \qquad \pi_{\theta=6}(a{=}1) = \sigma(6) = 0.9975 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(8) = 0.9997.$$

The first update takes the robot from indifferent between the two actions to strongly committed to one of them, a drastic change in behavior. The second changes nothing that an observer of the robot could detect. :numref:`fig_mdl-rl-policy-vs-parameter` plots the two action distributions before and after each update.

![Two parameter updates of the same size, one from $\theta = 0$ and one from $\theta = 6$, shown as the action distributions before and after each update. The left update rewrites the policy; the right update does not visibly change it.](../img/mdl-rl-policy-vs-parameter.svg)
:label:`fig_mdl-rl-policy-vs-parameter`

Both updates move $\theta$ by exactly two. On the left, the distribution flips from even odds to strong commitment. On the right, the before and after bars are indistinguishable. Equal steps in parameter space, wildly unequal steps in policy space: the map from parameters to policies stretches distances in some regions and crushes them in others, so a small change in the parameters can unexpectedly produce a large change in the policy, and a large change can produce none.

The derivative $\sigma'(\theta) = \sigma(\theta)(1 - \sigma(\theta))$ puts numbers on this. At $\theta = 0$ it equals $0.25$; at $\theta = 6$ it is about $0.0025$, a hundred times smaller. No learning rate is right in both regions at once. Worse, the two regions feed each other: near indifference the policy is at its most sensitive, so one noisy oversized update can throw it deep into a saturated region. Once saturated, the score $\nabla_\theta \log \pi_\theta$ is nearly zero, gradients all but vanish, and no sequence of ordinary updates brings the policy back. An on-policy method then keeps collecting data with the broken policy, so the data cannot rescue it either. The run is over, and the learning rate alone could not have prevented it, because capping the step in $\theta$ caps the wrong quantity.

So the quantity to control is the change in the policy, not the change in the parameters: what we want is an update rule that never changes the *policy* by more than we meant to, whatever that costs in parameter distance. Inside that guarantee we also want the freedom to take the largest step it allows and to reuse each batch several times.

## Reusing Data: From On-Policy to Off-Policy

Now to the second want. Every estimator since :numref:`sec_policygradient` required trajectories sampled from the current policy: one gradient step, and the batch that produced it is stale. On FrozenLake this discipline cost 4096 episodes where Q-Learning needed 256. Suppose instead the batch was collected by an older snapshot of the policy, $\pi_{\theta_{\textrm{old}}}$, and we want to keep learning from it after $\theta$ has moved. What, exactly, can old data tell us about a new policy?

Importance sampling gives the exact answer. For any function $f$ of trajectories,

$$E_{\tau \sim P(\cdot;\, \theta)} \big[ f(\tau) \big] = \sum_\tau P(\tau; \theta)\, f(\tau) = \sum_\tau P(\tau; \theta_{\textrm{old}})\ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) = E_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) \Big],$$
:eqlabel:`eq_change_of_measure`

valid whenever every trajectory the new policy can produce has positive probability under the old one, a condition softmax policies satisfy automatically because they never assign zero probability to any action. Applied with $f = R$, the return, this rewrites the objective of :numref:`sec_policygradient` as an expectation under the *old* policy,

$$J(\theta) = E_{\tau \sim P(\cdot;\, \theta)} \big[ R(\tau) \big] = E_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, R(\tau) \Big],$$
:eqlabel:`eq_offpolicy_objective`

so the performance of the new policy can be evaluated, without bias, on data the old policy collected. And the weight is computable: writing out $P(\tau; \theta)$ from :eqref:`eq_traj_prob`, the transition probabilities appear in both numerator and denominator and cancel, the same escape from the unknown MDP as in :numref:`sec_policygradient`. What remains is a pure product of policy ratios,

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})} = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}.$$

Unbiased, however, does not mean usable. Each factor in the product is bounded below by zero and unbounded above, and the product compounds across the trajectory: a trajectory that was unlikely under the old policy but likely under the new one can carry a weight of many orders of magnitude and single-handedly dominate the estimate. The exact correction :eqref:`eq_offpolicy_objective` trades all of its bias for variance that grows with the horizon.

The practical compromise keeps one ratio per step. Define

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}$$

and optimize the surrogate objective

$$L(\theta) = \frac{1}{n} \sum_{i=1}^n \sum_{t} \rho_t^i(\theta)\ \hat{A}_t^i,$$
:eqlabel:`eq_surrogate`

where $\hat{A}_t^i$ is any advantage estimate for step $t$ of trajectory $i$, e.g., the normalized reward-to-go of :numref:`sec_baselines`. Two corners have been cut relative to :eqref:`eq_offpolicy_objective`, and both deserve naming: the product of ratios became a single ratio per step, and the states in the batch remain distributed as $\pi_{\theta_{\textrm{old}}}$ visits them, not as $\pi_\theta$ would. In exchange, the weights stay bounded per step and the variance stays manageable. What survives the surgery is local fidelity: at $\theta = \theta_{\textrm{old}}$ every ratio equals one, and differentiating :eqref:`eq_surrogate` there gives back exactly the policy gradient estimator, since $\nabla_\theta \rho_t \,|_{\theta_{\textrm{old}}} = \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,|_{\theta_{\textrm{old}}}$. The surrogate is a local model of the objective: trustworthy near where it was built and a liar far away. This brings back the first section's concern as a concrete question: how far may $\theta$ move before we must stop trusting $L$?

## Trust Regions

Trust Region Policy Optimization answers with a constraint measured where the two-action example said it must be: in policy space. Maximize the surrogate, but keep the new policy close to the old one,

$$\max_\theta\ L(\theta) \quad \textrm{subject to} \quad \frac{1}{n}\sum_{i,t} \textrm{KL}\big( \pi_{\theta_{\textrm{old}}}(\cdot \mid s_t^i)\ \big\|\ \pi_\theta(\cdot \mid s_t^i) \big) \leq \delta,$$

where the Kullback-Leibler divergence measures, at each visited state, how far the new action distribution has moved from the old one. :citet:`Schulman.Levine.Abbeel.ea.2015` show that a version of this scheme improves the true objective monotonically: as long as the policy stays inside the region where the surrogate is trustworthy, ascending the surrogate ascends $J$. Solving the constrained problem takes second-order machinery that we will not build here. The idea to keep is the shape of the guarantee, a step-size rule expressed in policy space rather than parameter space: it does not matter how far $\theta$ moved, it matters how much $\pi_\theta$ moved.

## The Clipped Objective

PPO gets most of the benefit with none of the second-order machinery. Instead of constraining the ratios, clip their usefulness:

$$L^{\textrm{CLIP}}(\theta) = \frac{1}{n} \sum_{i,t} \min\Big( \rho_t^i(\theta)\, \hat{A}_t^i,\ \ \textrm{clip}\big(\rho_t^i(\theta),\ 1-\epsilon,\ 1+\epsilon\big)\, \hat{A}_t^i \Big),$$
:eqlabel:`eq_ppo_clip`

with a clipping parameter $\epsilon$, typically $0.2$. Read it one sample at a time. If $\hat{A}_t > 0$, the objective grows with $\rho_t$ but only up to $\rho_t = 1+\epsilon$; past that point the clipped term is smaller and the min selects it, so the sample's gradient becomes zero. The optimizer gains nothing by pushing the action's probability more than $\epsilon$ beyond what the old policy assigned. If $\hat{A}_t < 0$, the same happens on the way down at $1-\epsilon$. The min makes the bound one-sided in the pessimistic direction: a sample can always pull the objective down if the update has made things worse, it just cannot keep paying out for moving further away. Each sample stops contributing once its ratio leaves the band, and an update driven by :eqref:`eq_ppo_clip` stalls, per sample, at the edge of the trust region. That is the entire mechanism. We can now afford several epochs of updates on each batch.

One practical companion deserves a sentence. Implementations usually add a small entropy bonus to the objective, rewarding action distributions that are not too sharp. The two-action example showed what saturation costs: probabilities pinned near one and scores near zero, with no way back. The entropy term is the standing pressure against drifting there.

### Which Advantage Estimate

Anything from the last two sections can serve as $\hat{A}_t$ in :eqref:`eq_ppo_clip`. Normalized reward-to-go is the Monte Carlo end of the spectrum; the TD error $\delta_t$ of :numref:`sec_actorcritic` is the bootstrapped end. Generalized Advantage Estimation :cite:`Schulman.Moritz.Levine.ea.2016` interpolates between them with one knob,

$$\hat{A}_t^{\textrm{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l\, \delta_{t+l},$$
:eqlabel:`eq_gae`

where each $\delta$ is the TD error computed from a learned $\hat{V}$. At $\lambda = 0$ this is exactly $\delta_t$, maximum bias and minimum variance; at $\lambda = 1$ the sum telescopes to the reward-to-go minus $\hat{V}(s_t)$, the Monte Carlo advantage of :numref:`sec_baselines`. Values like $\lambda = 0.95$ are common. Our implementation below sticks with normalized reward-to-go to keep the critic out of the code; the exercise asks you to swap in :eqref:`eq_gae`.

## Implementation of PPO

We need the FrozenLake setup and the sampling helpers of the previous sections:

```{.python .input #ppo-when-one-step-is-too-big-1}

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

gamma = 0.95  # Discount factor

env_info = d2l.make_env('FrozenLake-v1', seed=0)
env = env_info['env']
num_states = env_info['num_states']
num_actions = env_info['num_actions']

def softmax_policy(theta, s):
    e = np.exp(theta[s] - np.max(theta[s]))
    return e / e.sum()

def sample_trajectory(env, theta):
    state, _ = env.reset()
    states, actions, rewards, done = [], [], [], False
    while not done:
        probs = softmax_policy(theta, state)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def reward_to_go(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]

def collect_batch(env, theta, batch_size):
    steps, hits = [], []
    for _ in range(batch_size):
        S, A, R = sample_trajectory(env, theta)
        hits.append(R[-1] > 0)
        for s, a, g in zip(S, A, reward_to_go(R, gamma)):
            steps.append((s, a, g))
    return steps, np.mean(hits)
```

The training loop differs from :numref:`sec_baselines` in one place: after collecting a batch and computing advantages, we record the log-probabilities under the collecting policy and then take `num_epochs` passes over the same batch, skipping any sample whose ratio has left the clipping band in the direction that no longer pays.

```{.python .input #ppo-implementation-of-ppo-1}

def ppo(seed, num_iters, batch_size, alpha, num_epochs, epsilon,
        use_clip=True):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    theta = np.zeros((num_states, num_actions))
    success, clipped, total = [], 0, 0
    for it in range(num_iters):
        steps, hit_rate = collect_batch(env, theta, batch_size)
        success.append(hit_rate)
        adv = np.array([w for _, _, w in steps])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Log-probabilities under the policy that collected the batch
        logp_old = np.array([np.log(softmax_policy(theta, s)[a])
                             for s, a, _ in steps])
        for epoch in range(num_epochs):
            u = np.zeros_like(theta)
            for (s, a, _), A, lp0 in zip(steps, adv, logp_old):
                rho = np.exp(np.log(softmax_policy(theta, s)[a]) - lp0)
                total += 1
                # Clipped: once the ratio leaves the band in the paying
                # direction, this sample's gradient is zero
                if use_clip and ((A >= 0 and rho > 1 + epsilon) or
                                 (A < 0 and rho < 1 - epsilon)):
                    clipped += 1
                    continue
                grad = -softmax_policy(theta, s)
                grad[a] += 1.0
                u[s] += rho * A * grad
            theta += alpha * u / batch_size
    return np.array(success), clipped / max(total, 1)

success, clip_frac = ppo(seed=0, num_iters=150, batch_size=8, alpha=1.0,
                         num_epochs=10, epsilon=0.2)
print(f'fraction of samples clipped: {clip_frac:.2f}')
```

Ten epochs per batch of eight episodes, and on this seed the smoothed success rate crosses 90% around update 35, roughly 280 episodes into training. Compare :numref:`sec_policygradient`, where the single-pass method needed over a thousand episodes before its return stabilized. About one sample in six ends up clipped, which is the mechanism working: those are exactly the samples whose probabilities the optimizer had already moved by the allowed $\epsilon$.

The control experiment is the same loop with `use_clip=False`: the ratio weighting of the surrogate stays and only the clip is removed, so the comparison isolates the clip itself. We run both on eight seeds.

```{.python .input #ppo-implementation-of-ppo-2}

d2l.set_figsize((6, 4))
for name, use_clip in [('clipped (PPO)', True), ('no clip', False)]:
    curves = [ppo(sd, 150, 8, 1.0, 10, 0.2, use_clip)[0]
              for sd in range(8)]
    runs = np.stack([np.convolve(c, np.ones(10) / 10, 'valid')
                     for c in curves])
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    x = np.arange(len(mean))
    d2l.plt.plot(x, mean, label=name)
    d2l.plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    dead = [sd for sd, c in enumerate(curves) if c[-15:].mean() < 0.1]
    print(f'{name}: seeds that never recover = {dead}')
d2l.plt.xlabel('update')
d2l.plt.ylabel('batch success rate')
d2l.plt.legend();
```

In our run the unclipped variant leaves two of the eight seeds permanently dead, while every clipped run learns the task. The unclipped average curve sits low because those two runs never recover, and its wide band has the same cause. When the unclipped variant survives it is fast; nothing about reuse itself is wrong, and at very small learning rates the clip only gets in the way. The clip is insurance against the interaction of reuse with step size. Ten unconstrained passes over one noisy batch add up to exactly the oversized policy change the two-action example warned about: the policy lands in saturation, gradients vanish, and every later batch is collected by the broken policy. No later data can undo it.

## Summary

Directly parameterized policies make step size treacherous: equal moves in parameter space can be a rewrite of the behavior or a no-op, so the quantity to control is the change in the policy, not the change in $\theta$. Importance sampling turns the objective of a new policy into an exact expectation over data from an old one; the trajectory-level weight is a product of policy ratios in which the transition probabilities cancel, unbiased but with variance that compounds along the horizon. The per-step surrogate :eqref:`eq_surrogate` tames the variance at the price of being only a local model of the objective. TRPO bounds how far the policy may move with a KL constraint and buys a monotonic improvement guarantee; PPO gets most of the effect with a per-sample clip at $1 \pm \epsilon$, which zeroes a sample's gradient once its ratio leaves the band in the paying direction. On FrozenLake this cut the episodes needed for a stable success rate from over a thousand, for the plain REINFORCE of :numref:`sec_policygradient`, to a few hundred, and the unclipped control showed what the insurance is for. Advantage estimates plug in freely, from normalized reward-to-go to the TD error, with GAE interpolating between the two ends.

## Exercises

1. Implement GAE :eqref:`eq_gae`: maintain the tabular critic of :numref:`sec_actorcritic`, compute the $\delta_t$ sequence for each trajectory, and form advantages with $\lambda \in \{0, 0.5, 0.95, 1\}$. Where do the two endpoints land relative to the methods you already know?
1. Sweep the clipping parameter $\epsilon$ over $\{0.05, 0.1, 0.2, 0.5, \infty\}$ at a couple of learning rates. When does small $\epsilon$ hurt, and how does $\epsilon = \infty$ relate to the no-clip control above?
1. Vary `num_epochs` from 1 to 30 with and without clipping. At one epoch, why should the two variants behave nearly identically? What happens at 30?
1. Run the normalized REINFORCE of :numref:`sec_baselines` with a deliberately oversized learning rate, $\alpha = 50$, on a handful of seeds. Many runs never learn at all: the first noisy updates saturate the softmax, exactly as the two-action example predicts, and the score vanishes. Now add an entropy bonus $-c \sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ to the per-sample objective. How large must $c$ be to prevent the collapse, and what does it cost when $\alpha = 2$?

<!-- slides -->

::: {.slide title="Trust Regions and PPO"}
Everything since the value-based sections parameterizes the
policy *directly*: the robot's behavior **is** $\pi_\theta$.

Gradient ascent controls the step in *parameters*. What we
care about is the step in the *policy* — and the two can come
apart completely.

This deck: how far apart (a two-action example), how to reuse
data exactly (importance sampling), and how to make big,
reusable updates safe (TRPO's constraint, PPO's clip).
:::

::: {.slide title="Parameter space versus policy space"}
One parameter, two actions (example due to Joshua Achiam):

$$\pi_\theta(a) = \begin{cases} \sigma(\theta) & a = 1 \\
  1 - \sigma(\theta) & a = 2. \end{cases}$$

Two updates of the **same size**, $\Delta\theta = 2$:

![](../img/mdl-rl-policy-vs-parameter.svg){width=88%}

. . .

And $\sigma'=\sigma(1-\sigma)$: sensitivity $0.25$ vs $0.0025$ —
one noisy step near indifference can throw the policy into
saturation, where scores vanish and nothing brings it back.
We need an update rule that never changes the *policy* more
than we meant to.
:::

::: {.slide title="Old data, exactly"}
What can data from $\pi_{\theta_{\text{old}}}$ say about
$\pi_\theta$? Change of measure:

$$J(\theta)
  = E_{\tau \sim \theta_{\text{old}}}\!\Big[
    \tfrac{P(\tau;\theta)}{P(\tau;\theta_{\text{old}})}\, R(\tau) \Big],
\qquad
\frac{P(\tau;\theta)}{P(\tau;\theta_{\text{old}})}
  = \prod_t \frac{\pi_\theta(a_t\mid s_t)}
    {\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.$$

- Transitions cancel in the ratio — model-free, again.
- **Exact and unbiased** — but the product of ratios compounds:
  one trajectory can carry a weight of many orders of magnitude.
  All the bias traded for horizon-growing variance.
:::

::: {.slide title="A surrogate you can afford"}
Keep one ratio per step:

$$\rho_t = \frac{\pi_\theta(a_t\mid s_t)}
  {\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
\qquad
L(\theta) = \frac1n \sum_{i,t} \rho^i_t(\theta)\, \hat A^i_t.$$

Two corners cut: product $\to$ per-step ratio; states still from
the old policy's visits. What survives: at
$\theta_{\text{old}}$, $\nabla L$ = the policy gradient.
$L$ is a *local* model — trust it near $\theta_{\text{old}}$,
nowhere else.
:::

::: {.slide title="TRPO, then PPO's clip"}
TRPO: maximize $L$ subject to
$\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi_\theta) \le \delta$ at
the visited states — a step-size rule in *policy* space, with a
monotonic improvement guarantee.

PPO keeps the effect, drops the machinery:

$$L^{\text{CLIP}} = \frac1n\sum_{i,t}
  \min\!\big(\rho\hat A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat A\big).$$

Once a sample's ratio leaves the band in the paying direction,
its gradient is zero: the update stalls at the trust-region
edge, sample by sample.
:::

::: {.slide title="PPO on Frozen Lake"}
Ten epochs per batch of eight episodes; ~90% success by
update 35 (≈280 episodes; plain single-pass REINFORCE needed
over a thousand). One sample in six gets clipped — the band
at work:

@ppo-when-one-step-is-too-big-1

. . .

@ppo-implementation-of-ppo-1
:::

::: {.slide title="The control: no clip"}
Identical loop, clipping off, eight seeds:

@ppo-implementation-of-ppo-2

Two of eight unclipped seeds die permanently; all clipped runs
learn. Reuse isn't the problem — ten unconstrained passes over
one noisy batch add up to the oversized policy change the
two-action example warned about, and on-policy learning can't
recover.
:::

::: {.slide title="Recap"}
- Parameter distance $\ne$ policy distance: control the policy.
- Change of measure: $J(\theta)$ on old data, exactly — but the
  ratio product explodes; per-step surrogate = local model.
- TRPO: KL constraint, guarantee, second-order cost.
- PPO: per-sample clip at $1\pm\epsilon$, several epochs per
  batch, episodes to stable success down from 1000+ to a few
  hundred.
- $\hat A$ pluggable: reward-to-go, TD error, GAE in between.
- Entropy bonus = standing pressure against the saturation the
  sigmoid example warns about.
:::
