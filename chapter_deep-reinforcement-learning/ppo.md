# Trust Regions and Proximal Policy Optimization
:label:`sec_ppo`

The methods of the last four sections share a trait that no earlier algorithm in these two chapters had: they parameterize the policy directly. Value Iteration and Q-Learning adjusted value estimates and read the policy off them; REINFORCE and actor-critic adjust $\theta$, and the agent's behavior *is* $\pi_\theta$. That raises a question gradient ascent cannot answer on its own, because gradient ascent controls the size of the step in parameters, while what we care about is the size of the step in the policy, and :numref:`sec_actorcritic` closed by measuring why the question is urgent: a better advantage estimate is a bigger step at the same learning rate. This section opens with an example in which the two step sizes come apart completely. The answer to that question, together with a way to reuse each batch of trajectories instead of discarding it after one update, gives the two ideas of this section: trust regions :cite:`Schulman.Levine.Abbeel.ea.2015` and the clipped objective of Proximal Policy Optimization :cite:`Schulman.Wolski.Dhariwal.ea.2017`, PPO for short. PPO is the workhorse of modern policy optimization; when language models are trained from human feedback (:numref:`sec_rl_sequences`), this is the algorithm running underneath.

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
import torch
torch.set_num_threads(1)
```

```{.python .input #ppo-trust-regions-and-proximal-policy-optimization-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import gymnasium as gym
import jax
from jax import numpy as jnp
import numpy as np
```

## Parameter Space versus Policy Space

Since we now update the parameters of the policy directly, we should ask what a parameter update does to the policy itself. A family of policies with a single parameter $\theta$ and two actions answers it; we borrow the example from Joshua Achiam's lectures on policy optimization :cite:`Achiam.2017`. Let

$$\pi_\theta(a) = \begin{cases} \sigma(\theta) & a = 1 \\ 1 - \sigma(\theta) & a = 2, \end{cases} \qquad \textrm{where } \sigma(\theta) = \frac{1}{1 + e^{-\theta}}.$$

Take two parameter updates of exactly the same size, $\Delta\theta = 2$, from two different starting points:

$$\pi_{\theta=0}(a{=}1) = \sigma(0) = 0.50 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(2) = 0.88, \qquad \pi_{\theta=6}(a{=}1) = \sigma(6) = 0.9975 \ \xrightarrow{\ \Delta\theta = 2\ }\ \sigma(8) = 0.9997.$$

The first update takes the agent from indifferent between the two actions to strongly committed to one of them, a drastic change in behavior. The second changes nothing that an observer of the agent could detect. :numref:`fig_rl-policy-vs-parameter` plots the map from parameter to policy with both updates drawn on it, and, beside it, the two action distributions before and after each update.

![Two parameter updates of the same size, $\Delta\theta = 2$. Left: the map $\pi_\theta(a{=}1) = \sigma(\theta)$ with both updates drawn on it; from $\theta = 0$ the policy moves by $0.38$, from $\theta = 6$ by $0.002$, and the annotated derivatives $\sigma'(0) = 0.25$ and $\sigma'(6) \approx 0.0025$ say why. Right: the same two updates as action distributions before and after; the left update rewrites the policy, the right update does not visibly change it.](../img/mdl-rl-policy-vs-parameter.svg)
:label:`fig_rl-policy-vs-parameter`

Both updates move $\theta$ by exactly two. On the left, the distribution flips from even odds to strong commitment. On the right, the before and after bars are indistinguishable. Equal steps in parameter space, wildly unequal steps in policy space: the map from parameters to policies stretches distances in some regions and crushes them in others, so a small change in the parameters can unexpectedly produce a large change in the policy, and a large change can produce none.

The derivative $\sigma'(\theta) = \sigma(\theta)(1 - \sigma(\theta))$ puts numbers on this. At $\theta = 0$ it equals $0.25$; at $\theta = 6$ it is about $0.0025$, a hundred times smaller. No learning rate is right in both regions at once. Worse, the two regions feed each other: near indifference the policy is at its most sensitive, so one noisy oversized update can throw it deep into a saturated region. Once saturated, the score $\nabla_\theta \log \pi_\theta$ is nearly zero, gradients all but vanish, and no sequence of ordinary updates brings the policy back. An on-policy method then keeps collecting data with the broken policy, so the data cannot rescue it either. The run is over, and the learning rate alone could not have prevented it, because capping the step in $\theta$ caps the wrong quantity.

So the quantity to control is the change in the policy, not the change in the parameters: what we want is an update rule that never changes the *policy* by more than we meant to, whatever that costs in parameter distance. Inside that guarantee we also want the freedom to take the largest step it allows and to reuse each batch several times.

## Reusing Data

### Change of measure

Now to the second want. Every estimator since :numref:`sec_policygradient` required trajectories sampled from the current policy: one gradient step, and the batch that produced it is stale. :numref:`sec_policygradient` priced that discipline in environment steps and found that most of the bill bought data whose only use was one update, because nothing bought earlier could be reused. Suppose instead the batch was collected by an older snapshot of the policy, $\pi_{\theta_{\textrm{old}}}$, and we want to keep learning from it after $\theta$ has moved. What, exactly, can old data tell us about a new policy?

Importance sampling gives the exact answer. For any function $f$ of trajectories,

$$E_{\tau \sim P(\cdot;\, \theta)} \big[ f(\tau) \big] = \sum_\tau P(\tau; \theta)\, f(\tau) = \sum_\tau P(\tau; \theta_{\textrm{old}})\ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) = E_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, f(\tau) \Big],$$
:eqlabel:`eq_change_of_measure`

valid whenever every trajectory the new policy can produce has positive probability under the old one, a condition softmax policies satisfy automatically because they never assign zero probability to any action. Applied with $f = R$, the return, this rewrites the objective of :numref:`sec_policygradient` as an expectation under the *old* policy,

$$J(\theta) = E_{\tau \sim P(\cdot;\, \theta)} \big[ R(\tau) \big] = E_{\tau \sim P(\cdot;\, \theta_{\textrm{old}})} \Big[ \frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})}\, R(\tau) \Big],$$
:eqlabel:`eq_offpolicy_objective`

so the performance of the new policy can be evaluated, without bias, on data the old policy collected. And the weight is computable: writing out $P(\tau; \theta)$ from :eqref:`eq_traj_prob`, the transition probabilities appear in both numerator and denominator and cancel, the same escape from the unknown MDP as in :numref:`sec_policygradient`. What remains is a pure product of policy ratios,

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{\textrm{old}})} = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}.$$

### The product of ratios explodes

Unbiased, however, does not mean usable. Each factor in the product is bounded below by zero and unbounded above, and the product compounds across the trajectory: a trajectory that was unlikely under the old policy but likely under the new one can carry a weight of many orders of magnitude and single-handedly dominate the estimate. The exact correction :eqref:`eq_offpolicy_objective` trades all of its bias for variance that grows with the horizon.

### The per-step surrogate

The practical compromise keeps one ratio per step. Define

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}$$

and optimize the surrogate objective

$$L(\theta) = \frac{1}{n} \sum_{i=1}^n \sum_{t} \rho_t^i(\theta)\ \hat{A}_t^i,$$
:eqlabel:`eq_surrogate`

where $\hat{A}_t^i$ is any advantage estimate for step $t$ of trajectory $i$, e.g., the normalized reward-to-go of :numref:`sec_baselines`. Two corners have been cut relative to :eqref:`eq_offpolicy_objective`, and both deserve naming: the product of ratios became a single ratio per step, and the states in the batch remain distributed as $\pi_{\theta_{\textrm{old}}}$ visits them, not as $\pi_\theta$ would. In exchange, the weights stay bounded per step and the variance stays manageable. What survives the surgery is local fidelity: at $\theta = \theta_{\textrm{old}}$ every ratio equals one, and differentiating :eqref:`eq_surrogate` there gives back exactly the policy gradient estimator, since $\nabla_\theta \rho_t \,|_{\theta_{\textrm{old}}} = \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,|_{\theta_{\textrm{old}}}$. The surrogate is a local model of the objective: trustworthy near where it was built and a liar far away. This brings back the first section's concern as a concrete question: how far may $\theta$ move before we must stop trusting $L$?

### The length-normalized trajectory ratio

Between the two extremes, a full product that is exact but explosive and a single factor that is tame but local, sits a third choice worth one sentence, because it is the one that scaled: raise the product of per-step ratios to the power $1/T$, the geometric mean, so that its logarithm is the *average* per-step log-ratio and its magnitude no longer compounds with the horizon. Clipping this length-normalized, sequence-level ratio rather than the per-step one is the idea marketed as GSPO in large language-model training, and :numref:`sec_rl_sequences` picks it up where trajectories become token sequences.

## Bounding the Step

### The performance difference lemma

The question at the end of the surrogate deserves a theorem before it gets an algorithm. Abbreviate the old policy's advantage :eqref:`eq_advantage` as $A^{\textrm{old}}(s, a) = Q^{\pi_{\theta_{\textrm{old}}}}(s, a) - V^{\pi_{\theta_{\textrm{old}}}}(s)$. How much better than the old policy is an arbitrary candidate $\pi_\theta$? Exactly this much:

**Proposition** (performance difference lemma, :citet:`Kakade.Langford.2002`).

$$
J(\theta) - J(\theta_{\textrm{old}}) = E_{\tau \sim P(\cdot;\, \theta)} \Big[ \sum_{t=0}^{T-1} \gamma^t\, A^{\textrm{old}}(s_t, a_t) \Big].
$$
:eqlabel:`eq_perf_diff`

**Proof.** Write $V$ for $V^{\pi_{\theta_{\textrm{old}}}}$. Along any trajectory, the sum $\sum_t \gamma^t \big( r_t + \gamma V(s_{t+1}) - V(s_t) \big)$ telescopes: the interior $V$ terms cancel in pairs, and $V = 0$ at termination, leaving $R(\tau) - V(s_0)$. Now take expectations under $\tau \sim P(\cdot;\, \theta)$. On the left, conditioned on $(s_t, a_t)$, the expectation of $r_t + \gamma V(s_{t+1})$ over the next state is $Q^{\pi_{\theta_{\textrm{old}}}}(s_t, a_t)$, by :eqref:`eq_dynamic_programming_q` written with $V^\pi$, so each term becomes $\gamma^t\, E\big[ A^{\textrm{old}}(s_t, a_t) \big]$. On the right, $E[R(\tau)] = J(\theta)$, and $E[V(s_0)] = J(\theta_{\textrm{old}})$ because both policies draw $s_0$ from the same $\mu_0$. $\blacksquare$

Read it as marching orders: to improve on the old policy, put the new policy's probability mass where the old policy's own advantages say the improvement is. All the difficulty hides in one subscript: the expectation is over the *new* policy's trajectories, which cannot be sampled without deploying the candidate. Approximate it with the states the old policy visited, reweight only the actions by $\rho_t$, drop the $\gamma^t$ factor as everywhere since :eqref:`eq_rtg`, and what appears is precisely the surrogate :eqref:`eq_surrogate`. So the two corners the surrogate cut are one corner seen twice: collapsing the product to a per-step ratio and freezing the visited states are both the act of evaluating the new policy on the old policy's state distribution. The lemma also prices the corner: the surrogate is exact at $\theta_{\textrm{old}}$, and its error grows with the gap between the two policies' state distributions, again a statement about how far the *policy* has moved, not the parameters.

### Trust regions and the monotonic-improvement bound

Trust Region Policy Optimization answers with a constraint measured where the two-action example said it must be: in policy space. Maximize the surrogate, but keep the new policy close to the old one,

$$\max_\theta\ L(\theta) \quad \textrm{subject to} \quad \frac{1}{n}\sum_{i,t} D_{\textrm{KL}}\big( \pi_{\theta_{\textrm{old}}}(\cdot \mid s_t^i)\ \Vert\ \pi_\theta(\cdot \mid s_t^i) \big) \leq \delta_{\textrm{KL}},$$

where the Kullback-Leibler divergence (:numref:`sec_mdl-information_theory`) measures, at each visited state, how far the new action distribution has moved from the old one. What earns the constraint the name of a guarantee is a bound. With the discount kept in place and $A_{\max} = \max_{s, a} \lvert A^{\textrm{old}}(s, a) \rvert$,

$$
J(\theta)\ \geq\ J(\theta_{\textrm{old}}) + L(\theta)\ -\ \frac{4 \gamma A_{\max}}{(1-\gamma)^2}\ \max_s\, D_{\textrm{KL}}\big( \pi_{\theta_{\textrm{old}}}(\cdot \mid s)\ \Vert\ \pi_\theta(\cdot \mid s) \big),
$$
:eqlabel:`eq_trpo_bound`

where $L$ stands for the population quantity that :eqref:`eq_surrogate` estimates :cite:`Kakade.Langford.2002,Schulman.Levine.Abbeel.ea.2015`. At $\theta = \theta_{\textrm{old}}$ the right-hand side equals $J(\theta_{\textrm{old}})$ exactly, so any $\theta$ that raises the right-hand side has certifiably improved the true objective: repeatedly maximizing this lower bound ascends $J$ monotonically, however badly the surrogate lies outside the region the penalty prices. Practice softens the theorem twice: the penalty coefficient it demands is so conservative that the maximizing steps shrink toward nothing, so TRPO demotes the max over states to the empirical mean and the penalty to the constraint above, choosing $\delta_{\textrm{KL}}$ by hand; and solving the constrained problem takes second-order machinery that we will not build here. The idea to keep is the shape of the guarantee, a step-size rule expressed in policy space rather than parameter space: it does not matter how far $\theta$ moved, it matters how much $\pi_\theta$ moved. One sentence ties this to the optimization chapter: measuring steps by $D_{\textrm{KL}}$ instead of by parameter norm is steepest ascent under the local geometry the policy family induces, the Fisher metric, whose update direction is the natural gradient :cite:`Amari.1998,Kakade.2002`, the same steepest-descent-under-a-chosen-norm view that organized :numref:`sec_muon`; the right panel of :numref:`fig_rl_trust_region` draws that metric's ellipse against the Euclidean ball.

![Bounding the step in policy space. Left: the surrogate $L$ is tangent to the true objective $J$ at $\theta_{\textrm{old}}$ and a liar far away; the unconstrained maximizer of $L$ drives $J$ from $0.82$ down to $-0.38$, while the best point inside the shaded trust region raises it to $1.52$. Right: for a three-action softmax family, the set of steps with $D_{\textrm{KL}} \leq 0.02$ under the exact Fisher metric at $\theta_{\textrm{old}}$ is an ellipse, not a ball; two parameter steps of equal Euclidean length move the policy by $D_{\textrm{KL}} = 0.008$ and $0.049$, so the parameter norm misprices policy change in both directions.](../img/mdl-rl-trust-region.svg)
:label:`fig_rl_trust_region`

### The clipped objective

PPO gets most of the benefit with none of the second-order machinery. Instead of constraining the ratios, clip their usefulness:

$$L^{\textrm{CLIP}}(\theta) = \frac{1}{n} \sum_{i,t} \min\Big( \rho_t^i(\theta)\, \hat{A}_t^i,\ \ \textrm{clip}\big(\rho_t^i(\theta),\ 1-\epsilon,\ 1+\epsilon\big)\, \hat{A}_t^i \Big),$$
:eqlabel:`eq_ppo_clip`

with a clipping parameter $\epsilon$, typically $0.2$. Read it one sample at a time. If $\hat{A}_t > 0$, the objective grows with $\rho_t$ but only up to $\rho_t = 1+\epsilon$; past that point the clipped term is smaller and the min selects it, so the sample's gradient becomes zero. The optimizer gains nothing by pushing the action's probability more than $\epsilon$ beyond what the old policy assigned. If $\hat{A}_t < 0$, the same happens on the way down at $1-\epsilon$. The min makes the bound one-sided in the pessimistic direction: a sample can always pull the objective down if the update has made things worse, it just cannot keep paying out for moving further away. Each sample stops contributing once its ratio leaves the band, and an update driven by :eqref:`eq_ppo_clip` stalls, per sample, at the edge of the trust region. The stall is an incentive rather than a hard constraint: nothing pins a ratio at the boundary, and gradient steps driven by other samples move the same network and can carry a ratio past the band, which is why the clipped fraction we measure below is not zero. The clip removes the payoff for drifting further, and that is enough in practice. We can now afford several epochs of updates on each batch.

![The clipped objective :eqref:`eq_ppo_clip` one sample at a time, as a function of the ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\textrm{old}}}(a_t \mid s_t)$. For a positive advantage the objective grows with the ratio only until $1 + \epsilon$; beyond that the minimum selects the clipped term and the sample's gradient is zero. For a negative advantage the same happens on the way down at $1 - \epsilon$. In both panels the unclipped side of the minimum stays open in the pessimistic direction: a sample can always pull the objective down, it just cannot keep paying out for moving further away.](../img/mdl-rl-ppo-clip.svg)
:label:`fig_rl_ppo_clip`

Set against :eqref:`eq_trpo_bound`, this is the honest ledger: PPO keeps the *shape* of the guarantee, a per-sample bound on how far movement keeps paying, and none of the guarantee itself. Nothing in :eqref:`eq_ppo_clip` implies monotonic improvement; the ablation below is the empirical substitute for the theorem, and the diagnostics after it are what practitioners watch in the theorem's place.

### Asymmetric bands

The band is symmetric in ratio space and anything but symmetric in what it permits. For an action the old policy took rarely, say $\pi_{\theta_{\textrm{old}}}(a \mid s) = 0.01$, the ceiling $\rho_t \leq 1 + \epsilon$ stops rewarding growth beyond a probability of $0.012$: per reuse cycle, a rare action may at most creep upward by a fifth of its almost nothing, while an action holding $0.60$ may add twelve full points of probability mass inside the same band. Sample by sample the symmetric clip is therefore tightest exactly on the low-probability actions that exploration needs to grow, and mass drains toward the modes faster than the tails can recover it, one more ratchet turning entropy down. The repair is as blunt as the diagnosis: decouple the two edges into $1 - \epsilon_{\textrm{low}}$ and $1 + \epsilon_{\textrm{high}}$ with $\epsilon_{\textrm{high}} > \epsilon_{\textrm{low}}$, giving rare actions room to grow while keeping the pessimistic side tight. This clip-higher band is a standing ingredient of the language-model recipes that :numref:`sec_rl_sequences` introduces, and exercise 7 works out its arithmetic.

### The entropy bonus

One practical companion deserves more than the sentence it usually gets. Implementations add a small *entropy bonus* to the objective, `entropy_coef` times the mean entropy of the action distributions, rewarding policies that are not too sharp. The two-action example showed what saturation costs: probabilities pinned near one, scores near zero, no way back. The entropy term is the standing pressure against drifting there, and in this section it moves from the margin notes into the code: `ppo_epochs` below adds the bonus to the objective and reports the entropy it measured, per epoch, as data, so that "the policy saturated" stops being a story and becomes a curve. Why a bonus of exactly this form is principled is :numref:`sec_regularized`'s theorem: the entropy bonus is a KL penalty measured against a uniform reference policy, one corner of a design whose optimum has a closed form.

## What a Working PPO Contains

### Which advantage estimate

Anything can serve as $\hat{A}_t$ in :eqref:`eq_ppo_clip`, and the menu is not new: it is :numref:`sec_actorcritic`'s credit-assignment dial. The reward-to-go minus a learned baseline sits at the Monte Carlo end, the TD error $\delta_t$ at the bootstrapped end (no collision with the trust-region radius $\delta_{\textrm{KL}}$: the symbols are distinct because both matter here), and the $\lambda$-return mixes all depths, collapsed by the telescoping identity :eqref:`eq_gae_deltas` into the two-line `Batch.gae` that is already in the library. That section also *measured* the dial and found the one-draw error smallest strictly inside, near $\lambda = 0.95$, which is the neighborhood every deployed PPO runs :cite:`Schulman.Moritz.Levine.ea.2016`. There is accordingly nothing left to derive and one decision left to make, and we make the deployed one: `train_ppo` below runs GAE with $\lambda = 0.95$ *by default*. The chapter ships the PPO people actually run, not a teaching stand-in with the interesting part left as an exercise.

### The implementation

The laboratory is :numref:`sec_actorcritic`'s: CartPole, the `ActorCritic.mlp` container, batches of eight episodes, and `critic_steps` regression passes for the critic, so that every difference below is the algorithm. Three hyperparameters are new. `num_epochs = 20` passes over each batch is aggressive reuse, on purpose; `epsilon_clip = 0.2` is the standard band; `entropy_coef = 0.01` is the bonus just argued for.

```{.python .input #ppo-the-implementation-1}
%%tab pytorch, jax
gamma, lam, num_updates, batch_episodes = 0.99, 0.95, 60, 8
num_seeds, num_epochs, critic_steps = 8, 20, 20
epsilon_clip, entropy_coef = 0.2, 0.01
if tab.selected('pytorch'):
    def cartpole_agent(seed):
        torch.manual_seed(seed)
        return d2l.ActorCritic.mlp(4, 2)
if tab.selected('jax'):
    def cartpole_agent(seed):
        return d2l.ActorCritic.mlp(4, 2, rngs=nnx.Rngs(seed))
```

Two pieces of per-framework speed first, not algorithm, both following the compilation rule of :numref:`sec_compilation` exactly as :numref:`sec_actorcritic` did: compile what has a fixed shape and runs hot. The acting forward is compiled and cached in the jax tab and stays eager in the pytorch tab; both tabs gain a batched probability read `_probs` that the vectorized collection at the end of the section will want.

```{.python .input #ppo-the-implementation-2}
%%tab pytorch
def _probs(ac, obs):   # batched action probabilities, read as numpy
    with torch.no_grad():
        return torch.softmax(ac.policy(torch.as_tensor(obs)), -1).numpy()
```

```{.python .input #ppo-the-implementation-2}
%%tab jax
_act_probs = nnx.jit(lambda net, obs: jax.nn.softmax(net(obs), -1))

def _probs(ac, obs):   # the fixed-shape acting forward, compiled and
    if not hasattr(ac, '_fwd'):    # cached as in :numref:`sec_actorcritic`
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

The critic's regression is the same hot loop it was in :numref:`sec_actorcritic`, and the jax tab repeats that section's padding trick, batches padded to a power-of-two length so the jitted pass compiles once per size bucket; sections are self-contained, so the cell is repeated rather than imported.

```{.python .input #ppo-the-implementation-3}
%%tab pytorch
# Eager per-pass cost is about a millisecond in this tab; the library
# helper needs no compilation story here, unlike its jax sibling.
fit_value = d2l.fit_value
```

```{.python .input #ppo-the-implementation-3}
%%tab jax
def _pad(x, size):  #@save
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

The heart of the section is one function. `ppo_epochs` receives a batch together with everything that must be *frozen* before reuse begins, the advantages and the collecting policy's log-probabilities, and spends `num_epochs` gradient passes on the clipped surrogate plus the entropy bonus. Its diagnostics are data, one row per epoch: the fraction of ratios outside the band; the mean log-ratio $\log \pi_{\theta_{\textrm{old}}} - \log \pi_\theta$, whose expectation over the old policy's actions is the divergence the trust region bounds, reported as the approximate KL; and the mean policy entropy. The `use_clip` switch keeps the ratio weighting of :eqref:`eq_surrogate` and removes only the clip: the control experiment. The jax tab jits the per-epoch step on the padded shapes above and caches it with `nnx.cached_partial`, one compiled step per size bucket.

```{.python .input #ppo-the-implementation-4}
%%tab pytorch
def ppo_epochs(ac, batch, adv, logp_old, epsilon, num_epochs,  #@save
               entropy_coef=0.01, use_clip=True):
    """num_epochs clipped-surrogate passes on one frozen batch; returns
    [num_epochs, 3] numpy diagnostics: fraction of ratios outside the
    band, approximate KL, mean policy entropy."""
    obs, act = torch.as_tensor(batch.obs), torch.as_tensor(batch.act)
    adv, logp_old = torch.as_tensor(adv), torch.as_tensor(logp_old)
    diag = []
    for _ in range(num_epochs):
        logp_all = torch.log_softmax(ac.policy(obs), dim=-1)
        logp = logp_all.gather(-1, act[:, None]).squeeze(-1)
        rho = torch.exp(logp - logp_old)
        surr = rho * adv
        if use_clip:
            surr = torch.min(surr,
                             rho.clamp(1 - epsilon, 1 + epsilon) * adv)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = -surr.mean() - entropy_coef * entropy
        ac.opt_pi.zero_grad()
        loss.backward()
        ac.opt_pi.step()
        diag.append((((rho - 1).abs() > epsilon).float().mean().item(),
                     (logp_old - logp).mean().item(), entropy.item()))
    return np.array(diag)
```

```{.python .input #ppo-the-implementation-4}
%%tab jax
@nnx.jit  #@save
def _ppo_step(policy, opt, obs, act, adv, logp_old, mask, epsilon,
              entropy_coef, use_clip):
    def loss_fn(policy):
        logp_all = jax.nn.log_softmax(policy(obs), axis=-1)
        logp = jnp.take_along_axis(logp_all, act[:, None], -1).squeeze(-1)
        rho = jnp.exp(logp - logp_old)
        surr = jnp.where(use_clip, jnp.minimum(
            rho * adv, jnp.clip(rho, 1 - epsilon, 1 + epsilon) * adv),
            rho * adv)
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1)
        loss = -(mask * (surr + entropy_coef * entropy)).sum() / mask.sum()
        return loss, (rho, logp, entropy)
    (_, (rho, logp, entropy)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(policy)
    opt.update(policy, grads)
    n = mask.sum()
    return ((mask * (jnp.abs(rho - 1) > epsilon)).sum() / n,
            (mask * (logp_old - logp)).sum() / n, (mask * entropy).sum() / n)

def ppo_epochs(ac, batch, adv, logp_old, epsilon, num_epochs,  #@save
               entropy_coef=0.01, use_clip=True):
    """num_epochs clipped-surrogate passes on one frozen batch; returns
    [num_epochs, 3] numpy diagnostics: fraction of ratios outside the
    band, approximate KL, mean policy entropy."""
    size = 1 << max(6, (len(adv) - 1).bit_length())
    mask = jnp.asarray((np.arange(size) < len(adv)).astype(np.float32))
    obs, act, adv, logp_old = (_pad(np.asarray(x), size) for x in
                               (batch.obs, batch.act, adv, logp_old))
    step = nnx.cached_partial(_ppo_step, ac.policy, ac.opt_pi)
    return np.array([step(obs, act, adv, logp_old, mask, epsilon,
                          entropy_coef, use_clip)
                     for _ in range(num_epochs)])
```

The training loop differs from :numref:`sec_actorcritic`'s in one place. After the critic's passes, it computes the GAE advantages, records the log-probabilities under the collecting policy, and then, instead of one gradient step, hands everything to `ppo_epochs`. The freeze is the pedagogical point: everything the epochs consume is data computed before they start; the only thing that moves during reuse is $\pi_\theta$. The critic regresses on the $\lambda$-return target `gae + value`, `critic_steps` passes taken first so the freshest critic judges, :numref:`sec_actorcritic`'s pattern. The generator yields the batch's return plus the per-epoch mean and last-epoch row of the diagnostics, and an optional `trace` list keeps the full per-epoch matrix, so the measurements below come free with the ablation.

```{.python .input #ppo-the-implementation-5}
%%tab pytorch, jax
def train_ppo(seed, ac, use_clip=True, trace=None):
    """Freeze the advantages and the collecting policy's log-probs, then
    spend num_epochs surrogate passes; GAE(0.95) is the default."""
    rng, env = np.random.default_rng(seed), gym.make('CartPole-v1')
    env.reset(seed=seed)
    for _ in range(num_updates):
        batch = d2l.rollout(env, ac.act, batch_episodes, rng)
        for _ in range(critic_steps):   # fresh lambda-return target, per pass
            fit_value(ac, batch.obs, batch.gae(ac.value_np, gamma, lam)
                      + ac.value_np(batch.obs))
        adv = d2l.normalize(batch.gae(ac.value_np, gamma, lam))
        logp_old = ac.log_prob_np(batch.obs, batch.act)
        d = ppo_epochs(ac, batch, adv, logp_old, epsilon_clip, num_epochs,
                       entropy_coef, use_clip)
        if trace is not None:
            trace.append(d)
        yield (float(batch.episode_returns().mean()), *d.mean(0), *d[-1])
```

### The ablation

Twenty epochs per batch is aggressive reuse: each batch of eight episodes now drives twenty gradient steps instead of one, and the ratios have twenty chances to drift from one. This is on purpose. The failure this section is about only shows itself when the combined step gets big, and we want it on screen. We run the clipped objective on eight seeds, keeping the trained agents for the audits below and the full diagnostics for the next subsection:

```{.python .input #ppo-the-ablation-1}
%%tab pytorch, jax
agents = {'clipped (PPO)': [cartpole_agent(s) for s in range(num_seeds)],
          'no clip': [cartpole_agent(s) for s in range(num_seeds)]}
trace = []
runs = {'clipped (PPO)': np.array(
    [list(train_ppo(s, agents['clipped (PPO)'][s], trace=trace))
     for s in range(num_seeds)])}
diag = np.array(trace).reshape(num_seeds, num_updates, num_epochs, 3)
```

Then the control, identical to the letter except that the clip is off:

```{.python .input #ppo-the-ablation-2}
%%tab pytorch, jax
runs['no clip'] = np.array(
    [list(train_ppo(s, agents['no clip'][s], use_clip=False))
     for s in range(num_seeds)])
```

First the casualties, counted rather than asserted, with a run declared dead if its last ten updates average below a return of 100:

```{.python .input #ppo-the-ablation-3}
%%tab pytorch, jax
for name, r in runs.items():
    dead = r[:, -10:, 0].mean(axis=1) < 100
    print(f'{name:>13}: {int(dead.sum())} of {num_seeds} seeds end dead; '
          f'casualties {np.flatnonzero(dead).tolist()}')
```

```{.python .input #ppo-the-ablation-4}
%%tab pytorch, jax
d2l.plot_curves({name: r[:, :, 0] for name, r in runs.items()},
                xlabel='update', ylabel='mean return of the batch',
                reference=500)
```

In our runs the unclipped control leaves most of the eight seeds dead at the end, down at the score of a pole that falls almost immediately. That collapse is the two-action example made real: twenty unconstrained passes over one noisy batch add up to exactly the oversized policy change the sigmoid picture warned about, the policy lands in saturation, gradients vanish, and every later batch is collected by the broken policy, so no later data can undo it. The clipped runs take the same twenty passes over the same batches and every seed ends near the ceiling. Read the printed list of casualties as a sample rather than a fact: *which* seeds die reshuffles with the floating-point arithmetic of the run, and it is the rate that is stable, more than half of the unclipped runs in every tab we have run never recovering while every clipped seed survives. At gentler settings, fewer epochs or a smaller learning rate, the unclipped variant survives almost always; the clip is insurance against the interaction of reuse with step size, not a speedup.

Note also how rarely the insurance pays out, and count it honestly, because the denominator is easy to get wrong. Each update checks every sample's ratio once per epoch, and the first epoch's checks always find $\rho_t = 1$, inside the band by construction; so we report the fraction of *all* per-epoch ratio checks that fall outside the band, and beside it the same fraction at the last epoch of each batch, where drift has had nineteen steps to accumulate:

```{.python .input #ppo-the-ablation-5}
%%tab pytorch, jax
for name, r in runs.items():
    print(f'{name:>13}: ratio checks outside the band: '
          f'{r[:, :, 1].mean():.1%} across all epochs, '
          f'{r[:, :, 4].mean():.1%} at the last epoch of each batch')
```

For the clipped runs both counts sit near one check in twenty. The clip does not need to fire often, because zeroing the gradient of exactly the samples whose ratios have left the band is what stops the runaway from compounding in the first place; the control's ratios, with nothing to stop them, leave the band about three times as often overall and keep drifting through the epochs, which is the failure in miniature.

### How to know your RL is broken

:numref:`sec_deeprl` closed with the warning that the loss carries no signal here and that what deserves watching are diagnostics of the *update*. This section is where that advice becomes concrete, because the run above already returned every number an engineer would watch. Within a batch:

```{.python .input #ppo-how-to-know-your-rl-is-broken-1}
%%tab pytorch, jax
d2l.plot_curves({'ratio checks outside the band': diag[:, :, :, 0].mean(1),
                 'approximate KL': diag[:, :, :, 1].mean(1),
                 'entropy': diag[:, :, :, 2].mean(1)},
                xlabel='epoch within the batch', ylabel='diagnostic')
print(f'entropy: {diag[:, :5, :, 2].mean():.2f} over the first five '
      f'updates, {diag[:, -5:, :, 2].mean():.2f} over the last five')
```

The within-batch shape is the clip working. The approximate KL and the outside-the-band fraction start at zero, by construction, and climb over the first handful of epochs as the policy walks away from the collector; then the curves flatten and even ease back, because once a ratio leaves the band its payoff is gone, and the gradients that remain, the pessimistic side and the entropy bonus, pull toward the band rather than away. The drift a batch suffers is front-loaded, which is why twenty epochs are survivable at all. Across updates, the printed pair shows the policy's entropy decaying from about $0.65$ toward $0.25$ nats as the policy sharpens. Nothing in this section stops that slide; the bonus only slows it. :numref:`sec_regularized` proves the slide is the destination of unregularized policy optimization, not an accident, and prices the cure.

The distribution behind those summary fractions is worth one look. Take two fresh identical agents at the *start* of training, where drift is largest, give both the same batch, the same frozen advantages and log-probabilities, and spend twenty passes with the clip on in one and off in the other, reading every ratio after each pass:

```{.python .input #ppo-how-to-know-your-rl-is-broken-2}
%%tab pytorch, jax
probe = {True: cartpole_agent(2), False: cartpole_agent(2)}   # twins
env = gym.make('CartPole-v1')
env.reset(seed=2)
batch = d2l.rollout(env, probe[True].act, batch_episodes,
                    np.random.default_rng(2))
adv = d2l.normalize(batch.gae(probe[True].value_np, gamma, lam))
logp_old = probe[True].log_prob_np(batch.obs, batch.act)
rhos = {}
for clip in (True, False):
    R = []
    for _ in range(num_epochs):   # one pass at a time, ratios read after
        ppo_epochs(probe[clip], batch, adv, logp_old, epsilon_clip, 1,
                   entropy_coef, use_clip=clip)
        R.append(np.exp(probe[clip].log_prob_np(batch.obs, batch.act)
                        - logp_old))
    rhos[clip] = np.array(R)
```

```{.python .input #ppo-how-to-know-your-rl-is-broken-3}
%%tab pytorch, jax
d2l.set_figsize((6, 4))
for clip, name in ((True, 'clipped (PPO)'), (False, 'no clip')):
    d2l.plt.hist(rhos[clip][-1], bins=60, alpha=0.5, label=name)
for edge in (1 - epsilon_clip, 1 + epsilon_clip):
    d2l.plt.axvline(edge, linestyle='--', color='black')
d2l.plt.xlabel(r'ratio $\rho_t(\theta)$ after {} passes'.format(num_epochs))
d2l.plt.ylabel('ratio checks')
d2l.plt.legend()
d2l.plt.show()
```

After twenty passes the clipped agent's ratios still form a narrow hill around one, a visible minority past the dashed edges; the unclipped agent's ratios have sprayed across the axis, many driven toward zero, the signature of probability mass being torn away from actions wholesale. And the book already owns the right one-number summary of this picture: these ratios are importance weights, and the effective sample size :eqref:`eq_mdl-bayes-is-ess` of a weighted batch, $N_{\textrm{eff}} = 1 / \sum_s \bar{w}_s^2$ for normalized weights $\bar{w}$, says how many equally-weighted samples the batch is still worth:

```{.python .input #ppo-how-to-know-your-rl-is-broken-4}
%%tab pytorch, jax
ess = {}
for clip, name in ((True, 'clipped (PPO)'), (False, 'no clip')):
    w = rhos[clip] / rhos[clip].sum(axis=1, keepdims=True)
    ess[name] = 1 / (w ** 2).sum(axis=1) / rhos[clip].shape[1]
d2l.plot_curves(ess, xlabel='epoch within the batch',
                ylabel='effective sample size / n')
print(f'after {num_epochs} epochs the batch is worth '
      + ' vs '.join(f'{v[-1]:.0%} ({k})' for k, v in ess.items())
      + f' of its {len(batch)} steps')
```

This is the folklore rule "reuse a batch for a few epochs, then stop" turned into a measurement: with the clip holding the ratios near one, the batch remains worth nearly all of its samples through the twentieth epoch; without it, the same batch decays to roughly half its nominal size or less, every further epoch learning from fewer, more extremely weighted effective samples. One caveat travels with the number, inherited from :numref:`fig_mdl-prob-bayes-importance`: the ESS reads only the weights, so it cannot see states the new policy would visit that the old batch never sampled; staleness of the *state distribution*, the corner both cuts shared, is invisible to it.

Finally the audit. Strip away the sampling noise and evaluate what the clipped run actually delivered, greedily:

```{.python .input #ppo-how-to-know-your-rl-is-broken-5}
%%tab pytorch, jax
ac, env = agents['clipped (PPO)'][0], gym.make('CartPole-v1')
env.reset(seed=2)
score = d2l.evaluate(env, ac.act_greedy, num_episodes=100)
print(f'greedy mean return over 100 episodes: {score:.0f}')
```

### From a teaching loop to a real one

Everything above is a teaching loop: one environment, whole episodes, full-batch epochs, and a learning rate, the container's default of $10^{-2}$, that is four to forty times what tuned PPO references run. The rate stays because this section wants the unclipped failure on screen within sixty updates; do not carry it out of the section, since a deployed PPO anneals a rate an order of magnitude smaller, under which the unclipped control would merely limp instead of dying. Two structural differences from a deployed loop deserve to be seen at working size. First, real implementations do not collect episodes one environment at a time; they step $N$ environments in lockstep for a fixed $T$ steps, harvesting an $N \times T$ *rectangle* of transitions whose shape never changes, which is what compiled code wants, the same reasoning as our padded buckets. The rectangle's right edge almost always cuts episodes mid-flight, and the cut is priced the way this book has priced every truncation since :numref:`sec_mdp`: the value function fills in the future the recording lost.

```{.python .input #ppo-from-a-teaching-loop-to-a-real-one-1}
%%tab pytorch, jax
envs = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 8)
obs, _ = envs.reset(seed=0)
rng, steps = np.random.default_rng(3), []
for _ in range(32):                       # the fixed 32 x 8 rectangle
    act = np.array([rng.choice(2, p=p) for p in _probs(ac, obs)])
    obs, rew, term, trunc, _ = envs.step(act)
    steps.append((rew, term | trunc))
rew, done = map(np.array, zip(*steps))
print(f'a fixed {rew.shape} rectangle of steps, '
      f'{int(done.sum())} episode boundaries inside it')
print(f'V at the cut, pricing the unrecorded future: '
      f'{ac.value_np(obs).round(0)}')
```

Our trained agent drops nothing in 32 steps, so every one of the eight columns ends mid-episode, and the value read at the cut prices the missing future at about $1/(1-\gamma) = 100$, the discounted value of balancing forever: without the bootstrap the rectangle would be worthless, with it, as usable as complete episodes. Second, real implementations do not take full-batch passes; the default recipe shuffles the rectangle into a few passes of small minibatches, four passes of minibatches of 32 being a common shape. On one frozen batch the two recipes take gradient steps of different counts and sizes, and land close together in total drift:

```{.python .input #ppo-from-a-teaching-loop-to-a-real-one-2}
%%tab pytorch, jax
full, mini = cartpole_agent(1), cartpole_agent(1)   # identical twins
env = gym.make('CartPole-v1')
env.reset(seed=1)
batch = d2l.rollout(env, full.act, batch_episodes,
                    np.random.default_rng(1))
adv = d2l.normalize(batch.gae(full.value_np, gamma, lam))
logp_old = full.log_prob_np(batch.obs, batch.act)
ppo_epochs(full, batch, adv, logp_old, epsilon_clip, num_epochs,
           entropy_coef)
idx = np.random.default_rng(1).permutation(len(batch))
for _ in range(4):                  # four passes of minibatches of 32
    for i in range(0, len(idx), 32):
        sl = idx[i:i + 32]
        mb = d2l.Batch(batch.obs[sl], batch.act[sl], batch.rew[sl],
                       batch.next_obs[sl], batch.term[sl], [len(sl)])
        ppo_epochs(mini, mb, adv[sl], logp_old[sl], epsilon_clip, 1,
                   entropy_coef)
print(f'{len(batch)} steps spent as {num_epochs} full-batch passes '
      f'or as {4 * int(np.ceil(len(idx) / 32))} minibatch steps:')
for name, ag in (('full batch', full), ('minibatches', mini)):
    kl = (logp_old - ag.log_prob_np(batch.obs, batch.act)).mean()
    print(f'  {name:>11}: approximate KL from the collecting policy '
          f'{kl:.3f}')
```

The two totals of policy movement land within a small factor of each other: minibatching is not a different algorithm, it is the same budget of drift spent in smaller coins, justified by memory and hardware throughput plus a little gradient noise that practitioners treat as a feature. What changes is bookkeeping: each ratio is re-checked against the band four times rather than twenty, and the advantage normalization of :numref:`sec_baselines` is usually recomputed per minibatch, at the granularity the gradient actually uses.

### What our PPO omits

The gap between this section's PPO and a production one is a list of small, named decisions, none of which needs new theory:

| What real implementations add | Why, in one line |
|---|---|
| Learning-rate annealing to zero | late updates on a nearly converged policy must be small; a constant rate keeps paying step-size risk for no return |
| Observation and reward normalization | running estimates hold network inputs and value targets near unit scale on tasks whose raw numbers vary by orders of magnitude |
| Value-loss clipping | the critic gets a band of its own; its measured benefit is disputed, yet nearly every implementation ships it |
| KL-based early stopping | stop the epochs when the measured approximate KL passes a threshold: our diagnostic panel turned into an actuator |
| Orthogonal initialization, small policy head | start the policy near uniform so the first updates cannot blow the ratios out |
| Advantage normalization per minibatch | :numref:`sec_baselines`'s per-batch step size, recomputed at the granularity the gradient uses |

The list is not ours. A community audit collected 37 such implementation details and measured which ones matter :cite:`Huang.Dossa.Raffin.ea.2022`; a controlled study went further and showed that at matched code-level choices PPO and TRPO perform nearly identically, so these details, not the clipped objective, account for much of PPO's practical edge :cite:`Engstrom.Ilyas.Santurkar.ea.2020`; and a large-scale sweep across a quarter-million trained agents reached similarly sober conclusions about which knobs carry the performance :cite:`Andrychowicz.Raichuk.Stanczyk.ea.2021`. The honest instruction a textbook can give is therefore this: when you need a real PPO, read a maintained single-file implementation, the roughly 300-line `cleanrl/ppo.py` is the standard study text, and diff it against this section; every line you do not recognize will be on the list above, and now you know why each is there.

## Summary

Directly parameterized policies make step size treacherous: equal moves in parameter space can be a rewrite of the behavior or a no-op, so the quantity to control is the change in the policy, not the change in $\theta$. Importance sampling turns the objective of a new policy into an exact expectation over data from an old one; the trajectory-level weight is a product of policy ratios in which the transition probabilities cancel, unbiased but with variance that compounds along the horizon, and the per-step surrogate :eqref:`eq_surrogate` tames it at the price of being only a local model. The performance difference lemma :eqref:`eq_perf_diff` makes the target exact, improvement equals the new policy's expected old-policy advantage, and exposes the surrogate's two cut corners as one: the state distribution stays the old policy's. TRPO turns that into the bound :eqref:`eq_trpo_bound` and a monotonic improvement guarantee under a KL constraint; PPO keeps the shape of the guarantee and none of the guarantee, a per-sample clip at $1 \pm \epsilon$ that zeroes a sample's gradient once its ratio leaves the band in the paying direction. Our implementation runs GAE($0.95$) by default, adds an entropy bonus in the objective, and returns its diagnostics as data: on CartPole, twenty epochs of reuse per batch destroy most unclipped runs through the saturation collapse of the sigmoid example, while every clipped run trains to the ceiling with about one ratio check in twenty landing outside the band. The diagnostics say why reuse is survivable, drift within a batch is front-loaded and the effective sample size of the clipped batch stays near its nominal size, and what remains between this loop and a deployed one is a fixed rectangle of vectorized experience, minibatch epochs, and a list of named implementation details whose collective weight rivals the algorithm's.

**What the experiments show, and what they do not.** All curves come from seeded runs through the shared numpy sampling stream; the two framework tabs share every estimator line but initialize their networks differently, so their curves and casualty lists differ seed by seed while supporting the same statements, and every statistic quoted in prose is printed by a visible cell. The ablation is eight seeds per arm per tab: every clipped seed ends near the ceiling in both tabs, and the unclipped control loses more than half of its eight seeds in each tab; the casualty *rate* is the stable object, its exact value and the identities of the dead seeds are not. The outside-the-band fractions, near five percent for clipped runs against about three times that for the control, are stable in level and ordering, not in digit. The within-batch diagnostic shapes, front-loaded KL and a flattening outside-band curve, recur across seeds and tabs; the entropy decay from about $0.65$ to about $0.25$ nats varies by a few hundredths. The probe pair is a single seeded batch per tab: its ESS endpoints, near full weight with the clip and half or less without, differ between tabs in the exact fraction while agreeing in the gap's direction and size class. The greedy audit ties CartPole's ceiling, a statement about the task, not the algorithm. Single seeded runs per tab, not sweeps: the compute belongs to readers.

## Exercises

1. [conceptual] *One epoch is not PPO.* Show that at $\theta = \theta_{\textrm{old}}$
   every ratio in :eqref:`eq_ppo_clip` equals one, and that the gradient of the
   clipped objective there is exactly the policy gradient estimate of
   :numref:`sec_baselines`. Predict what the clipped and unclipped variants will
   do at `num_epochs = 1`, before you run the next exercise.
1. [extended] *Reuse against clipping.* Vary `num_epochs` over $\{1, 5, 20\}$
   with and without clipping, three seeds each. Report the fraction of seeds
   that end below a return of 100, and the fraction of ratio checks outside the
   band. Where does the unclipped variant start losing seeds, and does the
   clipped one ever lose one? (About thirty minutes.)
1. [short-code] *How wide should the band be.* Sweep the clipping parameter
   $\epsilon$ over $\{0.02, 0.1, 0.2, 0.5\}$ and one very large value, say
   $10^6$. When does a small $\epsilon$ hurt, what does the large value
   reproduce, and how does the outside-the-band fraction move across the sweep?
1. [short-code] *Minibatch epochs.* Extend `train_ppo` to spend each batch as
   four passes of shuffled minibatches of 32, the deployed default, instead of
   twenty full-batch passes, reusing the slicing pattern of the comparison
   cell. Compare the two on three seeds: learning curves, final approximate KL
   per batch, and the outside-the-band fraction. Which differences are
   statistical and which are bookkeeping?
1. [short-code] *Saturation, and the cure.* Run the normalized REINFORCE of
   :numref:`sec_baselines` with a deliberately oversized learning rate,
   $\alpha = 50$, on eight seeds, and record how many seeds ever reach the
   goal. Diagnose the failing seeds with this section's instruments, the
   policy's entropy and the norm of the score, then add an entropy bonus
   $-c \sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ to the
   per-sample objective. How large must $c$ be to change the outcome, and what
   does the same $c$ cost at $\alpha = 2$?
1. [conceptual] *The clip as a step size.* For a two-action softmax policy,
   translate the band $|\rho_t - 1| \leq \epsilon$ into a bound on the change
   in the difference of the two logits, and show that the bound tightens as
   the policy becomes more certain. Explain in one sentence why this is the
   fix that the sigmoid example asked for, and why capping the step in
   $\theta$ would not have been.
1. [conceptual] *Asymmetric bands.* With $\epsilon = 0.2$, compute the largest
   probability each of two actions with $\pi_{\theta_{\textrm{old}}}(a \mid s)
   = 0.01$ and $0.60$ can reach before the clip stops paying for further
   growth, and the smallest each can be driven to before the clip stops the
   penalty. Which side of the band binds exploration, and why does raising
   $\epsilon_{\textrm{high}}$ while keeping $\epsilon_{\textrm{low}}$ fixed
   change the entropy trace of the diagnostic panel rather than just the
   speed of learning?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.2]{.kicker}

Trust regions and proximal policy optimization<br>
**parameter distance lies about policy distance · reuse a batch, exactly · the performance difference lemma · a clip instead of a constraint**
:::
:::

::: {.slide title="Parameter Space versus Policy Space"}
One parameter, two actions (example due to Joshua Achiam),
two updates of the **same size** $\Delta\theta = 2$:

![](../img/mdl-rl-policy-vs-parameter.svg){width=95%}

. . .

$\sigma'(0) = 0.25$ against $\sigma'(6) \approx 0.0025$: no learning
rate is right in both regions. One oversized step near indifference
throws the policy into saturation, scores vanish, and the on-policy
data is collected by the broken policy: **the run is over**. Capping
the step in $\theta$ caps the wrong quantity.
:::

::: {.slide title="Old Data, Exactly"}
What can a batch from $\pi_{\theta_{\text{old}}}$ say about
$\pi_\theta$? Change of measure:

$$J(\theta)
  = E_{\tau \sim \theta_{\text{old}}}\!\Big[
    \tfrac{P(\tau;\theta)}{P(\tau;\theta_{\text{old}})}\, R(\tau) \Big],
\qquad
\frac{P(\tau;\theta)}{P(\tau;\theta_{\text{old}})}
  = \prod_t \frac{\pi_\theta(a_t\mid s_t)}
    {\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.$$

- transitions cancel in the ratio: model-free, again
- **exact and unbiased**, but the product compounds along the
  trajectory; all the bias traded for horizon-growing variance
:::

::: {.slide title="A Surrogate You Can Afford"}
Keep one ratio per step:

$$\rho_t = \frac{\pi_\theta(a_t\mid s_t)}
  {\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
\qquad
L(\theta) = \frac1n \sum_{i,t} \rho^i_t(\theta)\, \hat A^i_t.$$

. . .

Two corners cut: product $\to$ per-step ratio; states still from the
old policy's visits. At $\theta_{\text{old}}$, $\nabla L$ **is** the
policy gradient. $L$ is a *local* model: trustworthy near where it
was built, a liar far away.
:::

::: {.slide title="The Performance Difference Lemma"}
$$J(\theta) - J(\theta_{\text{old}})
  = E_{\tau \sim P(\cdot;\,\theta)} \Big[ \sum_t \gamma^t\,
    A^{\text{old}}(s_t, a_t) \Big]$$

Proof in four lines: the TD identity telescopes along any
trajectory; take expectations (:cite:`Kakade.Langford.2002`).

. . .

- improvement $=$ the *new* policy's expected *old*-policy advantage
- everything hard hides in $\tau \sim \theta$: where the new
  policy goes
- swap in the old states, reweight actions by $\rho_t$: the
  surrogate. The two cut corners are **one corner seen twice**
:::

::: {.slide title="TRPO: a Bound, then a Constraint"}
$$J(\theta) \geq J(\theta_{\text{old}}) + L(\theta)
  - \frac{4\gamma A_{\max}}{(1-\gamma)^2}
    \max_s D_{\text{KL}}\big(\pi_{\theta_{\text{old}}} \Vert
    \pi_\theta\big)$$

Ascend the lower bound and $J$ ascends **monotonically**. In
practice: mean KL $\leq \delta_{\text{KL}}$ as a constraint,
second-order machinery to solve it.

. . .

![](../img/mdl-rl-trust-region.svg){width=95%}

Measuring steps in KL is steepest ascent under the Fisher metric:
the natural gradient, :numref:`sec_muon`'s norm story again.
:::

::: {.slide title="The Clip"}
$$L^{\text{CLIP}} = \frac1n\sum_{i,t}
  \min\!\big(\rho\hat A,\ \text{clip}(\rho,1-\epsilon,1+\epsilon)
  \hat A\big)$$

![](../img/mdl-rl-ppo-clip.svg){width=95%}

. . .

Once a ratio leaves the band in the paying direction, that sample's
gradient is zero; the pessimistic side stays open. PPO keeps the
**shape** of the guarantee and **none** of the guarantee.
:::

::: {.slide title="ppo_epochs: Reuse as a Function"}
Freeze the advantages and the collector's log-probabilities, then
spend the epochs; diagnostics returned as data:

@ppo-the-implementation-4
:::

::: {.slide title="train_ppo: GAE(0.95) by Default"}
The advantage menu is :numref:`sec_actorcritic`'s dial, measured
there; we ship the deployed setting, $\lambda = 0.95$:

@ppo-the-implementation-5
:::

::: {.slide title="The Ablation: Eight Seeds, Clip On and Off"}
@!ppo-the-ablation-4

. . .

Same batches, same twenty passes: most unclipped seeds die near
return 9 (saturation, for real); every clipped seed reaches the
ceiling. The insurance pays out on about one ratio check in twenty.
Which seeds die reshuffles; the **rate** is what is stable.
:::

::: {.slide title="How to Know Your RL Is Broken"}
@!ppo-how-to-know-your-rl-is-broken-1

. . .

- within a batch: KL and band-exits are **front-loaded**, then the
  clip stalls the drift
- across training: entropy decays from about $0.65$ to about $0.25$ nats; the bonus
  slows the slide, :numref:`sec_regularized` explains it
:::

::: {.slide title="The Batch Goes Stale, Measured"}
Ratios are importance weights, so the appendix's effective sample
size applies:

@!ppo-how-to-know-your-rl-is-broken-4

. . .

"Reuse for a few epochs, then stop" as a number: with the clip the
batch stays worth nearly all its samples; without it, half or less.
:::

::: {.slide title="From a Teaching Loop to a Real One"}
- $N \times T$ **rectangle** from vectorized environments; the cut
  edge is priced by $V$, like every truncation since
  :numref:`sec_mdp`
- minibatch epochs ($4 \times 32$): the same drift budget in
  smaller coins
- plus a list of named details: annealing, normalization, value
  clip, KL stop, orthogonal init
  :cite:`Huang.Dossa.Raffin.ea.2022`

. . .

At matched code-level details, TRPO $\approx$ PPO
:cite:`Engstrom.Ilyas.Santurkar.ea.2020`: the details, not the
objective, carry much of the edge. Read `cleanrl/ppo.py` and diff
it against this section.
:::

::: {.slide title="Recap"}
- parameter distance $\neq$ policy distance: control the policy
- change of measure buys reuse; the ratio product explodes; the
  surrogate is local
- performance difference lemma: improvement $=$ expected
  old-advantage under the **new** policy; one corner, seen twice
- TRPO: bound $+$ constraint $+$ guarantee; PPO: clip, no
  guarantee, works
- GAE($0.95$) by default; entropy bonus in the objective
- watch ratios, KL, entropy, ESS, never the loss
:::
