# Regularized Policy Optimization
:label:`sec_regularized`

Every constraint in :numref:`sec_ppo` was measured against a moving target. The trust region and the clip keep the new policy near the *previous iterate*: they bound each step, and once the iterates stop moving they bind nothing at all, so nothing in that section stopped the policy from marching, one safe step at a time, into the saturated corner its own entropy measurements complained about. Nothing so far keeps the policy near anywhere in particular. This section adds a penalty measured against a *fixed* policy, and that change of reference point changes the optimum itself, into a closed form we can verify to machine precision, one formula that contains the entropy bonus of :numref:`sec_ppo`, maximum-entropy reinforcement learning, and the objective that every frontier language model is finished with. Before earning it, we ask a question :numref:`chap_reinforcement_learning` deferred: what happens when the reward itself is *learned*, and what an optimizer does to a learned reward's errors. The experiments are tabular throughout and run in seconds.

```{.python .input #regularized-regularized-policy-optimization}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #regularized-regularized-policy-optimization}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## Rewards You Learn

:numref:`sec_mdp` called the reward the interface through which you tell the optimizer what you want, and showed an optimizer attacking a small crack in it. That section assumed you could write the reward down. For many tasks you cannot: no formula scores a helpful answer, a safe merge into traffic, or a good summary. What people *can* do reliably is compare: shown two attempts, they can usually say which one is better. Learning a reward from comparisons, then optimizing it, is preference-based reinforcement learning :cite:`Christiano.Leike.Brown.ea.2017`, and it is how language models are trained to follow instructions. Everything this section needs from that pipeline fits in one classical model and one logistic regression.

### Preferences and Bradley-Terry

The Bradley-Terry model :cite:`Bradley.Terry.1952` posits that each candidate carries a scalar score, here the return of a trajectory, and that a comparison is a noisy readout of the score difference:

$$P(\tau \succ \tau') = \sigma\big( r(\tau) - r(\tau') \big), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}.$$
:eqlabel:`eq_bradley_terry`

Fitting $r$ by maximum likelihood on labeled pairs is exactly logistic regression on the difference of features. Our laboratory: a $4 \times 4$ gridworld with deterministic moves, an absorbing goal in the far corner worth $+1$, a living cost of $-0.04$ per step, and a hazard lane of two cells in the middle of the map costing $-1.5$ to enter. The agent we improve, $\pi_{\textrm{ref}}$, is competent but hedging: a softmax over the true action values at temperature $0.25$, so it reaches the goal while dithering along the way, and it almost never enters the hazard lane, because it knows better.

```{.python .input #regularized-preferences-and-bradley-terry-1}
%%tab pytorch, jax
gamma = 0.95                                   # a 4 x 4 gridworld
P = np.zeros((16, 4, 16))
for s, a in np.ndindex(16, 4):
    x, y = s % 4, s // 4
    dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]   # left, down, right, up
    P[s, a, min(3, max(0, y + dy)) * 4 + min(3, max(0, x + dx))] = 1.0
P[15], P[15, :, 15] = 0.0, 1.0                 # the goal absorbs
bonus = np.full(16, -0.04)                     # every step costs a little
bonus[[5, 6]], bonus[15] = -1.5, 1.0           # a hazard lane; the goal pays
r_true = np.einsum('sat,t->sa', P, bonus)      # r*(s, a): the entered bonus
r_true[15] = 0.0
mdp = d2l.TabularMDP(P, r_true, gamma)
Q_true = mdp.backup(d2l.value_iteration(mdp, 200)[-1])
pi_ref = np.exp(Q_true / 0.25)                 # competent but hedging
pi_ref /= pi_ref.sum(1, keepdims=True)
```

Now hide $r^*$ and learn it back from comparisons alone. We collect a thousand trajectories from $\pi_{\textrm{ref}}$, describe each by its discounted visit counts $x(\tau)$, so that the true return is exactly the linear function $x(\tau)^\top r^*$, sample six thousand pairs, label each by :eqref:`eq_bradley_terry`, and fit $\hat{r}$ by logistic regression on feature differences, with a little weight decay, as one always should:

```{.python .input #regularized-preferences-and-bradley-terry-2}
%%tab pytorch, jax
rng = np.random.default_rng(0)

def sample_traj(pi, T=40):
    s, x = 0, np.zeros((16, 4))
    for t in range(T):
        a = rng.choice(4, p=pi[s])
        x[s, a] += gamma ** t                  # discounted visits: R = x . r
        s = P[s, a].argmax()
        if s == 15:
            break
    return x.ravel()

X = np.stack([sample_traj(pi_ref) for _ in range(1000)])
R = X @ r_true.ravel()                         # true returns; the fit never sees them
i, j = rng.integers(0, 1000, (2, 6000))
y = (rng.random(6000) < 1 / (1 + np.exp(R[j] - R[i]))).astype(float)
D = X[i] - X[j]
w = np.zeros(64)
for _ in range(4000):                          # logistic regression on (D, y)
    w += 0.5 * (D.T @ (y - 1 / (1 + np.exp(-D @ w))) / len(y) - 0.02 * w)
r_hat = w.reshape(16, 4)
core = (X > 0).sum(0).reshape(16, 4) >= 50     # pairs seen in >= 50 episodes
print(f'never visited: {(X.sum(0) == 0).sum()} pairs; '
      f'in at least 50 episodes: {core.sum()} pairs')
print(f'rms fit error on those {core.sum()}: '
      f'{np.sqrt(((r_hat - r_true)[core] ** 2).mean()):.3f}')
print(f'entries into the hazard lane: true '
      f'{np.round(r_true[[1, 4, 2, 5], [1, 2, 1, 2]], 2)}, '
      f'fitted {np.round(r_hat[[1, 4, 2, 5], [1, 2, 1, 2]], 2)}')
```

Read the three lines together. Where the reference's data lives, the fit is decent: an rms error of $0.19$ across the forty-six well-traveled state-action pairs. Where the data never goes, the fit is not wrong so much as *silent*: nine pairs were never visited, and their fitted reward sits at its initialization of zero. The third line is the trap being armed. Entering the hazard lane truly costs $-1.5$; the fitted reward prices those entries at roughly zero, because a policy that knows better produced the data, so the comparisons contain almost no evidence about the lane. Silence reads as zero, and zero is optimistic here.

### Identifiability, and a free per-prompt baseline

A comparison only ever weighs two trajectories from the *same* start state, so :eqref:`eq_bradley_terry` sees score differences and nothing else. Shift the reward by any function of the start state, $r(x, y) \to r(x, y) + f(x)$ in the prompt-response notation used at scale, and every difference, hence the whole likelihood, is unchanged: exercise 5 asks you to write out the two lines. A learned reward is therefore only identified *relative to each prompt*; its absolute level per prompt is pure convention. This is why subtracting a per-prompt baseline is free, twice over. On the reward side it changes nothing the comparisons ever measured. On the policy side, the zero-mean lemma of :numref:`sec_baselines` says that subtracting any function of the state from the weight leaves the policy gradient unbiased. The group mean that GRPO subtracts (:numref:`sec_baselines`) is this pair of facts used at scale: the quantity it removes was never identified, and removing it costs no bias.

### Dense versus terminal reward

Our comparisons scored whole trajectories, so $\hat{r}$ is anchored by end-to-end evidence, and at scale the learned reward is usually *terminal*: one number for the finished response. :numref:`sec_mdp` already priced this trade: terminal rewards are honest but sparse, dense rewards are informative but dangerous to author, and the only densification guaranteed not to change the optimum is potential-based shaping :cite:`Ng.Harada.Russell.1999`. The scale vocabulary is two sentences. An outcome reward model (ORM) scores the final result, a process reward model (PRM) scores intermediate steps; a PRM is a learned dense reward, which buys the credit-assignment help of density at the price of authoring, labeling, and guarding many more numbers.

## Optimizing a Proxy

We now do the dangerous thing on purpose: hand $\hat{r}$ to the strongest optimizer we have.

### Reward hacking and Goodhart

The failure mode has a name older than the field. Goodhart's law: when a measure becomes a target, it ceases to be a good measure. In reinforcement learning it appears as *reward hacking*, a policy scoring highly under the reward you wrote or fitted while defeating the intent behind it, and the catalogue of real examples is long :cite:`Krakovna.Uesato.Mikulik.ea.2020`. It is the shaped-bonus exploit of :numref:`sec_mdp` in one sentence: an optimizer pointed at any *estimated* objective finds the estimate's errors; there the error was authored, here it is statistical, and the optimizer does not care. Planning is the strongest optimizer this book owns, so we plan optimally against $\hat{r}$ with value iteration (:numref:`sec_valueiter`) and grade the result under $r^*$:

```{.python .input #regularized-reward-hacking-and-goodhart}
%%tab pytorch, jax
def plan(r):
    m = d2l.TabularMDP(P, r, gamma)
    return np.eye(4)[m.backup(d2l.value_iteration(m, 200)[-1]).argmax(1)]

def ret(pi, r):
    m = d2l.TabularMDP(P, r, gamma)
    return d2l.policy_evaluation(m, pi, 400)[-1][0]

for name, r in (('true reward  ', r_true), ('fitted reward', r_hat)):
    pi = plan(r)
    print(f'plan on {name}: fitted return {ret(pi, r_hat):+.2f}, '
          f'true return {ret(pi, r_true):+.2f}')
```

By the fitted yardstick the hacked plan is the better policy, $-0.03$ against $-0.24$; only differences mean anything here, since the yardstick's absolute level was never identified. Under the true reward it is a disaster, $-2.11$ against $+0.59$: the plan drives straight down the hazard lane, through both cells the data priced at zero. Nothing malfunctioned. Value iteration did its job on the numbers we gave it, and the numbers were wrong precisely where the reference policy's competence kept the data from going. That inversion, the proxy's errors living exactly where the data thins out, returns as the central obstacle of :numref:`sec_offline`.

### The measurement: true return against the KL budget

Where is $\hat{r}$ trustworthy? Where the data was, and the data came from $\pi_{\textrm{ref}}$. So the honest dial is not *how hard we optimize* but *how far from the reference we let the optimizer move*, and the natural ruler for the gap between two distributions over trajectories is the KL divergence. To trace the trade-off we need a family of policies leaning on $\hat{r}$ by increasing amounts. We take, at each state, the reference tilted by the fitted action values,

$$\pi_\beta(a \mid s) \propto \pi_{\textrm{ref}}(a \mid s)\, e^{\hat{Q}(s, a)/\beta},$$

with a dial $\beta$: large $\beta$ barely tilts, small $\beta$ commits to the argmax, recovering the hacked plan. For now this family is an ansatz; the next section proves this exponential tilting is exactly the optimal way to spend a KL budget, so take it on credit for one experiment. We sweep $\beta$, and for each policy record its true return, its fitted return, and the KL it spends, the discounted sum of per-state divergences from the reference along its own trajectories:

```{.python .input #regularized-the-measurement-true-return-against-the-kl-budget}
%%tab pytorch, jax
m_hat = d2l.TabularMDP(P, r_hat, gamma)
Q_hat = m_hat.backup(d2l.value_iteration(m_hat, 200)[-1])
kls, true_ret, prox_ret = [], [], []
for beta in np.logspace(1, -2.5, 36):
    pi = pi_ref * np.exp((Q_hat - Q_hat.max(1, keepdims=True)) / beta)
    pi /= pi.sum(1, keepdims=True)
    kl = (pi * np.log(np.where(pi > 0, pi / pi_ref, 1.0))).sum(1)
    kl[15] = 0.0                               # no decisions once absorbed
    rho = np.linalg.solve(np.eye(16) - gamma
                          * np.einsum('sa,sat->st', pi, P).T, np.eye(16)[0])
    kls.append(rho @ kl)
    true_ret.append(ret(pi, r_true))
    prox_ret.append(ret(pi, r_hat))
k = int(np.argmax(true_ret))
print(f'true return {true_ret[0]:+.2f} at KL 0, peak {true_ret[k]:+.2f} '
      f'at KL {kls[k]:.1f}, then {true_ret[-1]:+.2f} at KL {kls[-1]:.1f}')
d2l.plot(kls, [true_ret, prox_ret], 'KL from the reference (nats)', 'return',
         legend=['true return', 'fitted return'])
```

This is the section's most transferable picture. The fitted return rises along the entire sweep: by its own yardstick, more optimization is always better. The true return rises from $-0.11$ to a peak of $+0.43$ at a budget of about three nats, well above anything the reference achieved, then falls off a cliff as the remaining budget buys commitment to the hazard lane. Moderate optimization against the flawed proxy genuinely helped; the same optimization continued became the attack we just watched. The same experiment at language-model scale, a policy optimized against a learned reward while a held-out gold reward is watched, produces the same rise and turn :cite:`Gao.Schulman.Hilton.2023`. The x axis, distance from the reference, is about to become the knob we control directly.

## The Regularized Objective

:numref:`chap_reinforcement_learning` maximized expected reward. The measurement above says that with a learned reward this is the wrong objective: what we want is reward *net of the divergence it spends*. So write that down. For a single decision, a reward $r(a)$ over a finite action set, a reference $\pi_{\textrm{ref}}$, and a coefficient $\beta > 0$:

$$\max_{\pi}\ E_{a \sim \pi}\big[ r(a) \big] - \beta\, D_{\textrm{KL}}\big( \pi \Vert \pi_{\textrm{ref}} \big).$$
:eqlabel:`eq_kl_objective`

The penalty is not a constraint on a step. It is part of the objective, weighed against reward at exchange rate $\beta$, and it is measured against a policy that never moves.

### The proposition

**Proposition.** For $\beta > 0$, the objective :eqref:`eq_kl_objective` is maximized by the unique policy

$$\pi^\star(a) = \frac{1}{Z}\, \pi_{\textrm{ref}}(a)\, e^{r(a)/\beta}, \qquad Z = \sum_{a} \pi_{\textrm{ref}}(a)\, e^{r(a)/\beta},$$
:eqlabel:`eq_kl_optimum`

and the maximal value is $\beta \log Z$.

**Proof.** By construction $r(a) = \beta \log\big( \pi^\star(a) Z / \pi_{\textrm{ref}}(a) \big)$. Substituting into the objective, for any policy $\pi$,

$$E_\pi[r] - \beta D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}}) = \sum_a \pi(a)\, \beta \log \frac{\pi^\star(a)\, Z}{\pi(a)} = \beta \log Z - \beta\, D_{\textrm{KL}}(\pi \Vert \pi^\star).$$

Gibbs' inequality, proved in :numref:`chap_mdl-information-theory`, says the remaining divergence is nonnegative and zero exactly when $\pi = \pi^\star$. $\blacksquare$

The optimum is the reference multiplied by an exponential tilt and renormalized, and the achieved value is a log-partition function. :numref:`fig_rl_kl_tilting` draws it. Verifying the proposition is cheap enough that there is no excuse not to. First solve :eqref:`eq_kl_objective` numerically, by plain exponentiated gradient ascent, multiplicative updates renormalized onto the simplex, with every line visible; the $+1$ in the exact gradient $r - \beta(\log(\pi/\pi_{\textrm{ref}}) + 1)$ is constant across actions and dies in the normalization, so we drop it:

```{.python .input #regularized-the-proposition-1}
%%tab pytorch, jax
p_ref = np.array([0.30, 0.10, 0.25, 0.20, 0.15])
r = np.array([0.0, 0.5, 1.0, 2.0, 3.0])

def solve_kl(r, p, beta, steps=3000, eta=0.3):
    pi = np.full_like(p, 1 / len(p))
    for _ in range(steps):                     # exponentiated gradient ascent
        pi = pi * np.exp(eta * (r - beta * np.log(pi / p)))
        pi /= pi.sum()
    return pi

for beta in (5.0, 2.0, 0.5, 0.2):
    pi = solve_kl(r, p_ref, beta)
    print(f'beta = {beta:3}: pi_beta = {np.round(pi, 3)}, '
          f'reward {pi @ r:.2f}, KL {(pi * np.log(pi / p_ref)).sum():.2f}')
```

Then compare against :eqref:`eq_kl_optimum`, closed form against converged iterate, and the value against $\beta \log Z$:

```{.python .input #regularized-the-proposition-2}
%%tab pytorch, jax
def pi_star(r, p, beta):
    star = p * np.exp((r - r.max()) / beta)    # eq_kl_optimum, stabilized
    return star / star.sum()

gap = max(np.abs(solve_kl(r, p_ref, b) - pi_star(r, p_ref, b)).max()
          for b in (5.0, 2.0, 0.5, 0.2))
pi2 = solve_kl(r, p_ref, 2.0)
print(f'largest gap to the closed form over the ladder: {gap:.1e}')
print(f'objective at beta = 2: numerical '
      f'{pi2 @ r - 2.0 * (pi2 * np.log(pi2 / p_ref)).sum():.6f}, '
      f'beta log Z = {2.0 * np.log(p_ref @ np.exp(r / 2.0)):.6f}')
```

The largest disagreement across the whole ladder of $\beta$ is at floating-point resolution, comfortably beyond the $10^{-6}$ we would have settled for, and the achieved objective matches $\beta \log Z$ to six decimals. The proposition is not asymptotic and not approximate; it is the answer.

![The optimum of the penalized objective, drawn as the product it is. Each row multiplies the reference $\pi_{\textrm{ref}} = (0.30,\ 0.10,\ 0.25,\ 0.20,\ 0.15)$ by the tilt $e^{r/\beta}$ for the reward $r = (0,\ 0.5,\ 1,\ 2,\ 3)$ and renormalizes, giving $\pi^\star$; the top row uses $\beta = 2$, the bottom $\beta = 0.2$. The reference is deliberately not monotone in the reward, and its dip at $a_2$ survives into $\pi^\star$ in both rows: $\pi^\star$ is a reweighting of the reference, not a ranking of the reward. The stronger tilt spends more divergence, $D_{\textrm{KL}}(\pi^\star \Vert \pi_{\textrm{ref}}) = 0.15$ in the top row against $1.84$ in the bottom, and the margins annotate the two limits: $\beta \to \infty$ returns the reference, $\beta \to 0$ a point mass on the best action.](../img/mdl-rl-kl-tilting.svg)
:label:`fig_rl_kl_tilting`

### Four consequences

**Without the penalty, the optimum is a point mass.** Send $\beta \to 0$ in :eqref:`eq_kl_optimum` and all mass flows to the highest-reward action; that is just the unregularized problem, whose solution was always the argmax. Every stochastic policy :numref:`chap_reinforcement_learning` trained was therefore living on borrowed time: left alone, policy optimization *should* collapse onto a point mass, and the entropy decay measured in :numref:`sec_ppo` was this destiny in progress. Regularized reinforcement learning is what makes remaining a distribution optimal rather than merely transient.

**The coefficient is an interpolation dial.** The two ends of the dial, read off the closed form:

```{.python .input #regularized-four-consequences-1}
%%tab pytorch, jax
for beta in (100.0, 0.01):
    print(f'beta = {beta:6}: pi_star = {np.round(pi_star(r, p_ref, beta), 3)}')
```

$\beta \to \infty$ reproduces the reference; $\beta \to 0$ is greedy; every policy between is an exact optimum of some exchange rate between reward and divergence.

**A uniform reference is the entropy bonus.** Against a uniform reference, $D_{\textrm{KL}}(\pi \Vert \textrm{uniform}) = \log |\mathcal{A}| - H(\pi)$, so the penalty *is* an entropy bonus up to a constant, and :eqref:`eq_kl_objective` becomes the maximum-entropy objective. The closed form specializes to a softmax of the reward:

```{.python .input #regularized-four-consequences-2}
%%tab pytorch, jax
boltzmann = np.exp(r / 0.5) / np.exp(r / 0.5).sum()
print(np.allclose(pi_star(r, np.full(5, 0.2), 0.5), boltzmann))
```

That is the shape of :numref:`sec_policygradient`'s softmax policies, and the Boltzmann exploration rule of :numref:`sec_qlearning` was sampling from the optimal solution of an entropy-regularized problem all along, with its temperature playing the role of $\beta$. The sequential version of this consequence, the soft backup, closes the section.

**The optimum is a posterior.** Read :eqref:`eq_kl_optimum` as Bayes' rule: prior $\pi_{\textrm{ref}}$, likelihood $e^{r/\beta}$, posterior $\pi^\star$, evidence $Z$. This is a change of viewpoint with its own literature, in which reinforcement learning is inference, rewards are log-likelihoods of an optimality event, and regularized policy optimization is variational inference against that posterior :cite:`Levine.2018,Korbak.Perez.Buckley.2022`; everything known about approximate posteriors and the choice of divergence then transfers to policies wholesale.

Sweeping $\beta$ traces the exact frontier of reward against divergence, the curve the gridworld experiment could only approximate through a broken $\hat{r}$:

```{.python .input #regularized-four-consequences-3}
%%tab pytorch, jax
front = np.stack([pi_star(r, p_ref, b) for b in np.logspace(1.5, -1.5, 61)])
kl_f = (front * np.log(front / p_ref)).sum(1)
print(f'KL 0 pays {p_ref @ r:.2f}; the frontier ends at KL '
      f'{np.log(1 / p_ref[r.argmax()]):.2f}, reward {r.max():.0f}')
d2l.plot(kl_f, front @ r, 'KL from the reference (nats)', 'expected reward')
```

With the true reward on the y axis the frontier is concave and rising (exercise 2): each extra nat buys less reward than the one before. The gridworld's version *turned over* only because its y axis was the true return while its optimizer chased the fitted one. The two rows of :numref:`fig_rl_kl_tilting` are two points on this curve.

### Trust region versus penalty

Two ideas in this book wear the same three letters, and they must not be confused.

> **The two KLs.** A **trust region** (:numref:`sec_ppo`) measures KL against the *previous iterate*. It constrains the optimization *path*, protects a local approximation, moves with the policy, and at convergence it binds nothing: it shapes how you get there, never where you end up. A **penalty** (this section) measures KL against a *frozen reference*. It sits inside the objective, changes the *optimum* to :eqref:`eq_kl_optimum`, and never stops binding: it shapes where you end up. PPO-style fine-tuning of language models runs **both at once**, a clipped surrogate against the previous iterate for stability of each update, and a reward penalized by KL to the frozen initial model to anchor the optimum :cite:`ouyang2022training`; conflating the two is the commonest confusion in the literature around these methods.

### Which direction of KL

$D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ is the *reverse* divergence in the taxonomy of :numref:`sec_mdl-fwd-vs-rev-kl`, and both properties established there do work here. It is an expectation under $\pi$, so the policy can estimate its own penalty from its own samples; and it is mode-seeking, charging $\pi$ heavily for putting mass where the reference has little while charging nothing for abandoning the reference's modes. A policy optimized under it therefore concentrates inside the reference's support, on the highest-reward behavior the reference already exhibits: this is why post-training *sharpens* a model rather than broadening it. The forward direction $D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi)$ would instead force $\pi$ to cover everything the reference does, a supervised, imitation-flavored pull; the choice of direction is a modeling decision, not a technicality.

## Where This Goes

### Maximum-entropy RL and the soft backup

Everything above was one decision. In an MDP the penalty is charged at every step, $\sum_t \gamma^t \big( r_t - \beta\, D_{\textrm{KL}}( \pi(\cdot \mid s_t) \Vert \pi_{\textrm{ref}}(\cdot \mid s_t) ) \big)$, the quantity our frontier cell already measured, and the proposition applies state by state with $Q$ in place of $r$: the optimal policy tilts the reference by $e^{Q(s, a)/\beta}$, our ansatz, now with its credentials. Substituting the tilted policy into the Bellman equation of :numref:`sec_valueiter` replaces the hard maximum with a soft one,

$$V(s) = \max_a Q(s, a) \quad \textrm{becomes} \quad V(s) = \beta \log \sum_a \pi_{\textrm{ref}}(a \mid s)\, e^{Q(s, a)/\beta},$$

the value of the proposition, appearing as a backup: a logsumexp at inverse temperature, with the plain $\max$ recovered as $\beta \to 0$. With a uniform reference this is *soft value iteration* and the objective is maximum-entropy reinforcement learning. When :numref:`sec_dqn` builds deep learning on top of the hard $\max$, it will pay dearly for that operator's brittleness; it is worth knowing in advance that the $\max$ is the sharp corner of a family whose smooth members exist. Exercise 6 has you build the soft backup and watch one dial sweep from value iteration to policy evaluation.

### DDPG, TD3 and SAC

The regularized objective also completes a story this book can now tell entirely in prose, because every part has been built. The continuous-control lineage of off-policy actor-critic methods runs: DDPG :cite:`Lillicrap.Hunt.Pritzel.ea.2016` trains a deterministic actor by the pathwise gradient of :numref:`sec_deeprl` through a learned critic, off-policy from a replay buffer, with target networks; it is sample-efficient, notoriously brittle, and today taught as history. TD3 :cite:`Fujimoto.vanHoof.Meger.2018` diagnoses the brittleness largely as the maximization bias :numref:`sec_qlearning` demonstrated on a table: an actor trained to climb a critic is a maximizer over errors, so TD3 trains *twin* critics and takes their minimum, the bias argument applied twice, plus delayed actor updates and target smoothing. SAC :cite:`Haarnoja.Zhou.Abbeel.ea.2018` replaces the deterministic actor with a stochastic one and maximizes exactly this section's objective with a uniform reference: its "soft" critics implement the soft backup above, its entropy term is our penalty, its actor update is the pathwise gradient again, its twin critics are TD3's repair. Three components you already own. The one mechanism we have not covered is the change-of-variables correction SAC needs because its Gaussian actions are squashed through a $\tanh$ to fit bounded action spaces: squashing changes densities, so a log-determinant term joins the log-probabilities. We omit it as bookkeeping, one line of calculus per action dimension rather than a concept; every SAC implementation contains it, and knowing it is *there* is what matters when reading one.

### Forward pointers

Three sections point back here. :numref:`sec_dqn` studies what happens to the hard $\max$ under function approximation, knowing now that it is the $\beta \to 0$ corner. :numref:`sec_offline` faces our hacked planner's problem head-on, learning from a fixed dataset, and reaches for the mirrored tool, a penalty *subtracted* where data is scarce rather than a divergence charged for leaving the data's neighborhood. And :numref:`sec_rl_sequences` takes :eqref:`eq_kl_objective` to the setting it now rules, sequences from a language model, where the reference is the pretrained model, the reward is learned from preferences exactly as in this section, and the closed form :eqref:`eq_kl_optimum` gets read in both directions, from reward to optimal policy and back.

## Summary

Rewards can be learned from pairwise comparisons: Bradley-Terry is logistic regression on score differences, it identifies the reward only up to a function of the start state, which is why a per-prompt baseline is free on both the reward and the policy ledgers, and it is accurate where the preference data lives and silent where it does not. Optimizing a learned reward finds its errors, Goodhart's law in action: our optimal planner drove the fitted reward's unpriced hazard lane, and along a family of increasingly tilted policies the true return rose and then fell while the fitted return rose throughout, the overoptimization curve in miniature. The repair is to change the objective: expected reward minus $\beta$ times the KL divergence to a frozen reference. Its optimum is the reference exponentially tilted by reward, with value $\beta \log Z$, proved in four lines by Gibbs' inequality and verified to machine precision; no penalty gives a point mass, a uniform reference gives the entropy bonus and maximum-entropy RL, the soft backup replaces the hard maximum by a logsumexp, and the whole construction is Bayes' rule with $e^{r/\beta}$ as likelihood. A penalty against a frozen reference shapes the optimum; a trust region against the previous iterate shapes only the path; modern fine-tuning runs both, and the penalty's reverse KL direction is why it sharpens rather than broadens.

**What the experiments show, and what they do not.** Every cell is seeded numpy and prints identically in both framework tabs; reruns reproduce the digits exactly. The bandit-side results are exact computations: the closed form and the numerical optimum agree to floating-point resolution, and the frontier and limit statements are properties of :eqref:`eq_kl_optimum`, not of a sample. The gridworld results are one map, one reference policy, and one seeded draw of preferences: the fitting errors, the hacked plan's route, and the frontier's peak location move with the seed and the map, and the peak height by a tenth or two, while the qualitative claims, an accurate fit where data lives, silence where it does not, a plan exploiting the silence, and a true-return curve that rises and then falls, are what reseeding preserves. The hazard costs and the amount of preference data were chosen so that the failure is visible rather than marginal; gentler versions of the same numbers produce gentler versions of the same story. The compute belongs to readers.

## Exercises

1. [conceptual] *Three limits.* Derive $\beta \to 0$, $\beta \to \infty$ and
   $\pi_{\textrm{ref}}$ uniform from the proposition, and say which one you have
   already met and where.
1. [short-code] *The frontier is concave.* Sweep $\beta$ and plot achieved
   reward against $D_{\textrm{KL}}(\pi_\beta \Vert \pi_{\textrm{ref}})$; verify
   concavity and explain what the slope at each point is.
1. [short-code] *Break the reward.* Perturb the fitted $\hat r$ at one
   rarely-visited state and rerun; how much KL budget does the optimizer need
   before it finds the error?
1. [conceptual] *Trust region or penalty.* For each of TRPO, PPO's clip, RLHF's
   KL term and on-policy distillation, say which of the two it is and what it
   is measured against.
1. [conceptual] *Identifiability.* Show $r(x,y) \to r(x,y) + f(x)$ leaves the
   Bradley-Terry likelihood unchanged, and deduce that a per-prompt baseline
   costs nothing.
1. [extended] *The soft backup.* Implement value iteration with $\max$ replaced
   by $\beta\,\mathrm{logsumexp}$ on the slippery lake; plot $V_\beta^\star$
   against $\beta$ and identify the two limits.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.3]{.kicker}

Regularized policy optimization<br>
**rewards learned from comparisons · an optimizer finds the estimate's errors · penalize distance to a frozen reference · the optimum in closed form**
:::
:::

::: {.slide title="Rewards You Learn"}
No formula scores a helpful answer; people can *compare*.

$$P(\tau \succ \tau') = \sigma\big( r(\tau) - r(\tau') \big)$$

Bradley-Terry: fitting $r$ is logistic regression on feature
differences.

. . .

Fit on 1000 trajectories from a competent reference:
rms error $0.19$ where the data lives, and the hazard lane,
truly $-1.5$ to enter, is priced at $\approx 0$: the reference
knew better, so the data is silent. **Silence reads as zero.**
:::

::: {.slide title="Reward Hacking, Produced"}
Plan *optimally* against the fitted reward; grade under the truth.

@!regularized-reward-hacking-and-goodhart

. . .

The hacked plan wins by its own yardstick and drives straight
down the hazard lane. Goodhart's law: an optimizer pointed at
an *estimated* objective finds the estimate's errors.
:::

::: {.slide title="True Return Against KL Budget"}
@!regularized-the-measurement-true-return-against-the-kl-budget

. . .

- fitted return rises the whole way; true return rises to
  $+0.43$ at $\approx 3$ nats, then falls off a cliff
- the same curve at language-model scale:
  :cite:`Gao.Schulman.Hilton.2023`
- so control the x axis directly
:::

::: {.slide title="The Regularized Objective"}
$$\max_{\pi}\ E_{a \sim \pi}\big[ r(a) \big] - \beta\,
D_{\textrm{KL}}\big( \pi \Vert \pi_{\textrm{ref}} \big)$$

. . .

**Proposition.**
$\ \pi^\star(a) = \pi_{\textrm{ref}}(a)\, e^{r(a)/\beta} / Z$,
with value $\beta \log Z$.

**Proof.** The objective equals
$\beta \log Z - \beta\, D_{\textrm{KL}}(\pi \Vert \pi^\star)$;
apply Gibbs' inequality. $\blacksquare$

Verified numerically: largest gap over a ladder of $\beta$
is $\approx 10^{-16}$.
:::

::: {.slide title="One Picture"}
![](../img/mdl-rl-kl-tilting.svg){width=98%}

. . .

A *product*, not a ranking: the reference's dip at $a_2$
survives into $\pi^\star$. Spent divergence: $0.15$ at
$\beta = 2$, $1.84$ at $\beta = 0.2$.
:::

::: {.slide title="Four Consequences"}
- no penalty $\Rightarrow$ a point mass: PPO's entropy
  collapse was destiny, not accident
- $\beta$ interpolates: reference $\leftarrow \beta \to \infty$,
  greedy $\leftarrow \beta \to 0$
- uniform reference $=$ entropy bonus $=$ max-ent RL;
  $\pi^\star = \mathrm{softmax}(r/\beta)$: Boltzmann exploration
  was optimal all along
- Bayes' rule: prior $\pi_{\textrm{ref}}$, likelihood
  $e^{r/\beta}$, posterior $\pi^\star$
  :cite:`Levine.2018,Korbak.Perez.Buckley.2022`
:::

::: {.slide title="The Two KLs"}
**Trust region** (the clip): against the *previous iterate*;
constrains the path; gone at convergence.

**Penalty** (here): against a *frozen reference*; in the
objective; changes the optimum.

. . .

PPO-RLHF runs **both at once** :cite:`ouyang2022training`.

Direction matters: $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$
is mode-seeking, so post-training *sharpens*
(:numref:`sec_mdl-fwd-vs-rev-kl`).
:::

::: {.slide title="Where This Goes"}
$$V(s) = \beta \log \sum_a \pi_{\textrm{ref}}(a \mid s)\,
e^{Q(s, a)/\beta} \ \xrightarrow{\ \beta \to 0\ } \ \max_a Q(s, a)$$

- soft backup: DQN's $\max$ is the sharp corner of a family
- DDPG $\to$ TD3 $\to$ SAC: pathwise gradient + twin critics
  + this section's objective; three parts you already own
- Ahead: the same objective, with a language model as
  $\pi_{\textrm{ref}}$
:::
