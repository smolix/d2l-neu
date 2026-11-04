# Temporal-Difference Learning
:label:`sec_qlearning`

Q-learning does not require a transition model or expert demonstrations. It replaces the expectation in the Bellman optimality operator with a sampled transition and updates an action-value estimate using the resulting temporal-difference error :cite:`Watkins.1989,Watkins.Dayan.1992`. This substitution introduces sampling noise and makes convergence depend on which state--action pairs the behavior policy visits. We derive and evaluate the tabular update; :numref:`sec_exploration` studies exploration through multi-armed bandits and regret.

```{.python .input #qlearning-temporal-differences-q-learning-and-exploration}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
```

```{.python .input #qlearning-temporal-differences-q-learning-and-exploration}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import gymnasium as gym
import numpy as np
```

## The Sampled Backup

### From the Bellman Operator to the TD Update

Value iteration, written for the action-value function, sweeps

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A},$$

whose fixed point is $Q^*$; call the right-hand side $(TQ)(s, a)$, the optimality operator of :numref:`sec_valueiter` on action-value tables. We use $Q$ rather than $V$ because extracting the optimal policy from $Q^*$ requires only an argmax over a table row and no transition model. The kernel enters the sweep through one expectation, $\mathbb{E}_{s' \sim P(\cdot \mid s, a)} [\max_{a'} Q_k(s', a')]$, and an expectation can be estimated from samples. Panel (d) of :numref:`fig_rl_backups` shows this substitution: Q-learning uses one sampled successor where the exact backup sums over every successor, while retaining the maximum over next actions.

Let the agent act with some data-collection policy $\pi_e(a \mid s)$, its *behavior policy*, and record $n$ trajectories of $T$ timesteps each, $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. The Bellman equations constrain the action values of all state--action pairs. A sampled objective can impose corresponding residual terms only for pairs present in the data:

$$\hat{Q} = \mathrm{argmin}_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

The objective does not evaluate the kernel explicitly. Instead, when the behavior policy $\pi_e$ takes action $a_t^i$ at state $s_t^i$, the next state $s_{t+1}^i$ is sampled from the transition distribution. The collected data therefore provide sampled access to the transition dynamics. For every constraint to be represented, $\pi_e$ must keep visiting every state-action pair.

An optimal deterministic policy would *not* qualify: it never takes the actions it considers suboptimal, so $\ell(Q)$ would contain no term for those state-action pairs and their values would be left unconstrained. :numref:`sec_exploration` takes that requirement seriously.

Instead of solving the joint problem directly, Q-learning updates individual table entries. For each observed transition $(s, a, r, s')$, it moves the corresponding entry toward the *bootstrapped target* $r + \gamma \max_{a'} Q(s', a')$, treating the target as a constant even though it contains $Q$:

$$Q(s, a) \leftarrow (1 - \alpha)\, Q(s, a) + \alpha \big( r + \gamma \max_{a'} Q(s', a') \big),$$
:eqlabel:`q_learning`

where $\alpha$ is the step size. Because the target itself depends on $Q$ and we deliberately hold it fixed, this is called a semi-gradient step: Q-learning is therefore not gradient descent on $\ell(Q)$ in the strict sense; it is a semi-gradient stochastic-approximation update. Rearranged, the update is a correction proportional to a single scalar,

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a), \qquad Q(s, a) \leftarrow Q(s, a) + \alpha\, \delta.$$
:eqlabel:`eq_td_error`

The scalar $\delta$ is the *temporal-difference error* :cite:`Sutton.1988`: the observed reward plus the estimated best continuation, minus the current estimate. It also appears in the actor updates of :numref:`sec_actorcritic` and in deep Q-learning (:numref:`sec_dqn`).

*Monte Carlo* prediction waits until an episode ends and uses the realized return as a target for $Q^{\pi_e}(s_t,a_t)$, the value of the behavior policy. The target is unbiased but contains randomness from all subsequent transitions. *Temporal-difference* prediction bootstraps after one step; its target is biased while the value estimate is inaccurate but usually has lower variance :cite:`Sutton.1988,Sutton.Barto.2018`. Q-learning uses a maximum in this target to estimate $Q^*$. The multi-step methods of :numref:`sec_actorcritic` interpolate between Monte Carlo and one-step targets.

### Convergence and the Double-Sampling Problem

Suppose $\pi_e$ visits every state--action pair and collects unlimited data. Minimizing $\ell(Q)$ then recovers the value-iteration solution when transitions are deterministic, but not in general. In the population objective, conditioning on $(s,a)$ decomposes each squared term into a squared mean and a variance:

$$L(Q) = \mathbb{E}_{(s, a) \sim \mu} \Big[ \big( Q(s, a) - (TQ)(s, a) \big)^2 \Big] + \gamma^2\, \mathbb{E}_{(s, a) \sim \mu} \Big[ \textrm{Var}_{s' \sim P(\cdot \mid s, a)} \big( \max_{a'} Q(s', a') \big) \Big],$$
:eqlabel:`eq_double_sampling`

Here $\mu$ is the visitation distribution of $\pi_e$. The first term is the squared Bellman residual and is zero at $Q^*$. The second term depends on $Q$ whenever transitions are stochastic, so the minimizer of $L$ need not be $Q^*$. Exercise :exref:`ex_qlearning-double-sampling-counterexample` gives a three-state counterexample. This is the *double-sampling problem*: an unbiased gradient of the squared Bellman residual requires two independent successors for the same $(s,a)$, which an ordinary environment interaction does not provide. For deterministic transitions the variance term vanishes.

Convergence instead follows from the expected update. Averaging :eqref:`eq_td_error` over $s'$ gives $\mathbb{E}[\delta\mid s,a]=(TQ)(s,a)-Q(s,a)$, which is zero at the unique fixed point $Q^*$. Q-learning is therefore a stochastic approximation to Bellman fixed-point iteration. With the step-size conditions below and infinite visitation of every state-action pair, it converges to $Q^*$ with probability one :cite:`Watkins.Dayan.1992,Jaakkola.Jordan.Singh.1994`. Sampling variance affects the rate, while the behavior policy determines whether the coverage condition holds.

### Step Sizes and the Robbins-Monro Conditions

For each state-action pair, the Robbins--Monro conditions are $\sum_k\alpha_k=\infty$ and $\sum_k\alpha_k^2<\infty$ :cite:`Robbins.Monro.1951`. A constant step size violates the second condition. With stochastic transitions, its estimates continue to fluctuate around $Q^*$. In a deterministic environment, the target has no transition noise and an update has the form $Q(s,a)\leftarrow(1-\alpha)Q(s,a)+\alpha(TQ)(s,a)$. Updating all entries gives a contraction with modulus $(1-\alpha)+\alpha\gamma$. The sampled asynchronous method additionally requires repeated visitation of every entry.

Panel (c) of :numref:`fig_rl_exploration` measures this coverage for deterministic FrozenLake, where a constant step size of $0.9$ learns quickly within the finite budget.

The conditions are necessary for the guarantee, not sufficient for a budget. The sample-average rule $\alpha = 1/(1 + n(s, a))$, with $n(s, a)$ counting visits to the entry including the current one, passes Robbins-Monro but weights early targets computed from an inaccurate zero table as heavily as later, better-informed targets. At the tested finite budget, this slows learning substantially. Our default $\alpha = 1/(1 + 0.1\, n(s, a))$ also passes and approaches ten times that step size as the visit count grows. We compare all three schedules on the same seeds and report their finite-budget value errors and returns.

### Terminal Masking

The update requires terminal masking in an episodic environment. When $s'$ is terminal, its continuation value is zero, so the target contains only the immediate reward,

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big( r + \gamma\, \big(1 - \mathbf{1}(s' \textrm{ terminal})\big) \max_{a' \in \mathcal{A}} Q(s', a') - Q(s, a) \Big).$$

The mask uses `terminated`, not `truncated`: :numref:`sec_mdp` distinguishes a terminal state from an observation stopped by a time limit. Treating truncation as termination would impose an incorrect zero continuation value on a nonterminal state. In code the whole repair is one factor, `gamma * (1 - terminated) * Q[s_next].max()`.

## Q-Learning on the Lake

### The Implementation

We return to the slippery lake of :numref:`sec_mdp` with the same discount. The learner accesses the environment only through `reset` and `step`; `mdp`, `V_star`, and `pi_star` are retained solely to compare the learned table with the exact solution.

```{.python .input #qlearning-the-implementation-1}
%%tab pytorch, jax
gamma, num_episodes = 0.95, 4000
env = gym.make('FrozenLake-v1', is_slippery=True)
mdp = d2l.TabularMDP.from_gym(env, gamma)
V_star = d2l.value_iteration(mdp, num_iters=1000)[-1]
Q_star = mdp.backup(V_star)
pi_star = Q_star.argmax(axis=1)
```

Two helpers, both plain numpy and both saved: the deep Q-network of :numref:`sec_dqn` and the PPO of :numref:`sec_ppo` will call them unchanged. The first turns a start value, an end value and a horizon into a schedule:

```{.python .input #qlearning-the-implementation-2}
%%tab pytorch, jax
def linear_schedule(start, end, num_steps):  #@save
    """step -> value, interpolated from start to end, then held at end."""
    return lambda step: end + (start - end) * max(0.0, 1.0 - step / num_steps)
```

The second is the behavior policy. With probability $\epsilon$ it takes a uniformly random action; otherwise it exploits the current table:

$$\pi_e(a \mid s) = \begin{cases} 1 - \epsilon + \epsilon/|\mathcal{A}| & a = \mathrm{argmax}_{a'} \hat{Q}(s, a') \\ \epsilon/|\mathcal{A}| & \textrm{otherwise}, \end{cases}$$
:eqlabel:`epsilon_greedy`

the *$\epsilon$-greedy* policy, with ties in the argmax broken uniformly at random. Random tie-breaking is necessary for exploration from a zero-initialized table, where every state begins as a four-way tie. In contrast, `np.argmax` always returns the first index:

```{.python .input #qlearning-the-implementation-3}
%%tab pytorch, jax
def epsilon_greedy(q, epsilon, rng):  #@save
    """Explore with probability epsilon, else act greedily on the values q."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    # Ties are coin flips: np.argmax would always pick action 0 on a zero
    # table, and an agent that only ever moves left never finds the goal.
    return int(rng.choice(np.flatnonzero(q == q.max())))
```

The implementation repeatedly selects an action, observes one transition, and changes one table entry by $\alpha\delta$. It mutates both the value table and the visit counts; the counts determine the step-size schedule and also record state--action coverage. The exploration rate decreases from $1$ to $0.05$ during the first half of training and remains at that floor thereafter:

```{.python .input #qlearning-the-implementation-4}
%%tab pytorch, jax
def q_learning(seed, Q, visits, env, num_episodes, gamma,  #@save
               alpha=lambda n: 1 / (1 + 0.1 * n), epsilon=None):
    """Tabular Q-learning; updates Q in place, yields each episode's return.
    `alpha(n)` is the step size at the n-th visit of a pair; `epsilon(k)` is
    the exploration rate in episode k (default: 1 to 0.05 over the first
    half of the episodes, then held)."""
    rng = np.random.default_rng(seed)
    if epsilon is None:
        epsilon = linear_schedule(1.0, 0.05, num_episodes // 2)
    env.reset(seed=seed)
    for episode in range(num_episodes):
        s, done, ret = env.reset()[0], False, 0.0
        while not done:
            a = epsilon_greedy(Q[s], epsilon(episode), rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            visits[s, a] += 1
            delta = r + gamma * (1 - terminated) * Q[s_next].max() - Q[s, a]
            Q[s, a] += alpha(visits[s, a]) * delta
            s, done, ret = s_next, terminated or truncated, ret + r
        yield ret
```

Every cell here is numpy and shared between the frameworks: tabular reinforcement learning is index arithmetic, and the frameworks enter when the table becomes a network in :numref:`sec_deeprl`. Five seeds, one plain loop:

```{.python .input #qlearning-the-implementation-5}
%%tab pytorch, jax
Q = np.zeros((5, mdp.num_states, mdp.num_actions))
visits = np.zeros_like(Q)
returns = np.array([list(q_learning(k, Q[k], visits[k], env, num_episodes,
                                    gamma)) for k in range(5)])
d2l.plot_curves({'Q-learning': returns}, xlabel='episode',
                ylabel='return per episode', smooth=200)
```

The curve is a trailing 200-episode average, with the band spanning the seeds. It remains near zero while successful episodes are rare, increases as successful transitions update preceding state--action values, and flattens after the exploration schedule reaches its floor.

### The Learned Table against the Exact Solution

The curve plateaus near $0.55$, below the optimal success probability of $73.6$ percent measured in :numref:`sec_valueiter`. This comparison includes failures caused by the behavior policy's continuing exploration and therefore does not directly assess the learned table. Because the exact solution is available for this example, we can compare the learned values with $V^*$:

```{.python .input #qlearning-reading-the-curve-honestly-1}
%%tab pytorch, jax
lake1 = d2l.TabularMDP.from_gym(env, 1.0)    # gamma = 1: V(s0) = P(goal)
success = lambda pi: d2l.policy_evaluation(lake1, pi, 100)[-1][0]
live = ~np.isin(env.unwrapped.desc.flatten(), [b'H', b'G'])  # agent acts
greedy = np.eye(mdp.num_actions)[Q.argmax(axis=-1)]    # (seed, s, a)
print(f'max_s |V_Qhat(s) - V*(s)| per seed: '
      f'{np.round(np.abs(Q.max(axis=-1) - V_star).max(axis=-1), 3)}')
print(f'max_(s,a) |Qhat - Q*| per seed:     '
      f'{np.round(np.abs(Q - Q_star)[:, live].max(axis=(1, 2)), 3)}')
print(f'max_s of V* - Q*(s, greedy action):  '
      f'{np.round((V_star - (greedy * Q_star).sum(-1)).max(-1), 3)}')
print(f'success within 100 steps: greedy '
      f'{np.round([success(g) for g in greedy], 3)}, '
      f'pi* {success(np.eye(mdp.num_actions)[pi_star]):.3f}')
taxed = 0.95 * np.eye(mdp.num_actions)[pi_star] + 0.05 / mdp.num_actions
print(f'pi* forced to explore at epsilon = 0.05: {success(taxed):.3f}')
print(f'median environment steps: {np.median(visits.sum(axis=(1, 2))):,.0f}')
```

<!-- WRITER (14.3-M4): exact evaluation replaces the 2000-episode Monte Carlo. All five greedy policies give up nothing (third line 0.000: optimal up to ties; they differ from pi* only at the exact tie at state 6) and succeed with the exact 0.730 of pi*. Over 12 seeds this holds for 11; one run (seed 7) still prefers *right* at the start after 4,000 episodes, where the two best actions differ by 0.008 (Q* = 0.180 against 0.172) and its estimates sit low (0.136); its greedy policy succeeds 0.45 and is optimal by 16,000 episodes. Say "the greedy table is optimal in the typical run", not "always". V error 0.006-0.021 here, 0.006-0.031 over 12 seeds ("within about 0.03"); Q error on all live entries 0.075-0.131 here, 0.05-0.14 over 12 seeds, several times the V error (non-greedy entries are rarely visited). Exact price of exploration: 0.537 ("about 54%"). Median steps "about 96,000" (12 seeds: 73,000-98,000). -->

Across seeds, the learned value functions are within $0.006$ to $0.021$ of $V^*$ in the sup norm. Their greedy policies succeed in $71$ to $74$ percent of evaluation episodes, compared with the optimal value of $73.6$ percent; differences within this range are sampling error. When the optimal policy is forced to select a random action with probability $0.05$, its success rate falls to $54$ percent. The training curve therefore reflects both the learned policy and the continuing exploration of the behavior policy.

The median run used $95{,}569$ environment steps. By comparison, :numref:`sec_valueiter` certified $V^*$ after $164$ sweeps of $64$ exact model backups, or about $10{,}500$ backups. These quantities are not directly interchangeable: a Bellman backup requires the full transition kernel, whereas an environment step provides only one sampled transition. Exercise :exref:`ex_qlearning-like-like-comparison-value` compares the two methods while keeping these different information requirements explicit.

The following panels compare the learned and exact solutions:

```{.python .input #qlearning-reading-the-curve-honestly-2}
%%tab pytorch, jax
d2l.show_grid(env.unwrapped.desc, np.stack([Q[0].max(-1), V_star]),
              np.stack([Q[0].argmax(-1), pi_star]),
              titles=['Q-learning, seed 0', 'value iteration'])
```

The learned arrows reproduce the slip-aware strategy of :numref:`sec_valueiter`
except at state $6$: Q-learning selects *right*, whereas $\pi^*$ selects
*left*. State $6$ lies between holes $5$ and $7$, and the two actions have
equal values under $Q^*$, so this disagreement is a different resolution of an
exact tie. The value estimates are most accurate in frequently visited states
and coarser in the rarely visited top-right corner.

Now consider the deterministic special case. Disabling slip produces
deterministic transitions and noiseless targets, for which the
damped-contraction argument permits the constant $\alpha = 0.9$. In this
experiment, one sixteenth of the episode budget suffices:

```{.python .input #qlearning-reading-the-curve-honestly-3}
%%tab pytorch, jax
calm = gym.make('FrozenLake-v1', is_slippery=False)
table = np.zeros((mdp.num_states, mdp.num_actions))
calm_visits = np.zeros_like(table)
for _ in q_learning(0, table, calm_visits, calm, 256, gamma,
                    alpha=lambda n: 0.9):
    pass
print(f'calm ice, 256 episodes ({calm_visits.sum():,.0f} steps), alpha = 0.9: '
      f'greedy policy succeeds '
      f'{d2l.evaluate(calm, lambda s, rng: int(table[s].argmax()), 100):.0%}')
```

<!-- WRITER: over 12 seeds every greedy table succeeds 100% after 256 calm episodes, using 1,670-1,920 environment steps ("under 2,000"); sec_policygradient compares REINFORCE on the same map in steps. -->

In the deterministic environment, the same method finds a perfect policy with about one sixteenth as many episodes and can use a much larger step size. This contrast shows how transition noise changes the sample and step-size requirements. To compare the three schedules, we track $Q(s_0,\textrm{left})$ across the same five seeds; its exact value is available from dynamic programming.

```{.python .input #qlearning-reading-the-curve-honestly-4}
%%tab pytorch, jax
a0 = int(pi_star[0])
print(f'Q*(s0, {"<v>^"[a0]}) = {Q_star[0, a0]:.3f}')
trace = {}
for name, alpha in [('0.9 (constant)', lambda n: 0.9),
                    ('1/(1 + 0.1 n)', lambda n: 1 / (1 + 0.1 * n)),
                    ('1/(1 + n)', lambda n: 1 / (1 + n))]:
    runs = []
    for seed in range(5):
        Qa = np.zeros_like(Q[0])
        runs.append([Qa[0, a0] for _ in q_learning(
            seed, Qa, np.zeros_like(Qa), env, num_episodes, gamma, alpha)])
    trace[f'alpha = {name}'] = np.array(runs)
    print(f'alpha = {name:>14}: final estimates {np.round(runs, 3)[:, -1]}')
print(f'peak of each 1/(1 + 0.1 n) run:  '
      f'{np.round(trace["alpha = 1/(1 + 0.1 n)"].max(axis=1), 3)}')
d2l.plot_curves(trace, xlabel='episode', ylabel='estimate of Q(s0, left)',
                reference=Q_star[0, a0])
```

<!-- WRITER (14.3-M5, 14.3-S2): the overshoot is the evidence for maximization bias, not the finals. Peaks of the default schedule, seeds 0-4: 0.228, 0.291, 0.262, 0.324, 0.205 (episodes 330-1180), all above Q* = 0.180, against finals 0.180-0.195. Over 12 seeds 10 of 12 runs overshoot, by 0.02 to 0.16; the two others stay at or below Q* throughout (one is the low run of the note above). Say "in most runs". Finals over 12 seeds: constant 0.9 gives 0.02-0.42 (5 seeds: 0.06-0.27), the default 0.14-0.195 (the low run; 11 of 12 in 0.179-0.195), 1/(1+n) below 0.025 in all 12. The trace plot shows the seed median with a min-max band. -->

The constant step size does not converge to a fixed value: after four thousand episodes, the five estimates range from $0.06$ to $0.27$, compared with the exact value $0.180$. The default decreasing schedule places all five estimates between $0.180$ and $0.195$. The sample-average schedule $1/(1+n)$ also satisfies the Robbins--Monro conditions, but all five estimates remain below $0.025$ at this budget because early, inaccurate targets retain substantial weight. The asymptotic conditions therefore do not determine which schedule performs best at a finite horizon.

### Feedback between Estimation and Data Collection

The behavior policy depends on the current value estimates, so estimation errors influence which data are collected next. If $\hat{Q}(s,a)$ is too large, an $\epsilon$-greedy policy selects $a$ more often and produces additional transitions whose expected TD errors reduce the overestimate. An underestimate is corrected less directly, because the action may be selected only through random exploration. This creates an asymmetry between overestimation and underestimation.

At a fixed point, the expected correction must vanish for every state--action pair visited by the behavior policy. By :eqref:`eq_td_error`, this is precisely the Bellman optimality condition on the visited set. Interaction can therefore direct data collection toward errors in the current estimate. Offline reinforcement learning, introduced in :numref:`sec_offline`, removes this feedback because it learns from a fixed dataset.

The preceding argument assumes continued exploration; :numref:`sec_exploration` isolates the resulting tradeoff.

### Planning in a Learned Model

<!-- WRITER (14.3-S1, E3): certainty equivalence. The same transitions that Q-learning uses once can be counted: P_hat(s'|s,a) = N(s,a,s')/N(s,a), r_hat(s,a) = mean reward; unvisited pairs keep zero rows (value 0). `Counts` is a gym wrapper that passes every step through and counts it, so `q_learning` runs unchanged on it and produces the same table as on `env`. The cell runs 5 seeds at two budgets (each a full run with its own epsilon schedule) and compares, on the same data, Q-learning's table with value iteration on the estimated model: sup-norm V error and the exact success of the greedy policy. Over 12 seeds, median [range]: 500 episodes (about 9,000 steps): V error 0.091 [0.036, 0.51] learned against 0.036 [0.016, 0.47] planned, success 0.51 [0.00, 0.73] against 0.68 [0.50, 0.73] (model at least as good in 11 of 12 runs); 2,000 episodes (about 47,000 steps): V error 0.028 [0.021, 0.12] against 0.014 [0.008, 0.066], the model better in all 12 runs and as accurate as Q-learning after its 4,000 episodes (0.015, about 96,000 steps); Q error on all live entries 0.11 against 0.06. The printed medians over seeds 0-4: 500 episodes V error 0.084 learned against 0.096 planned (the worst-case error is still set by rarely visited cells and goes either way per seed), success 0.49 against 0.64; 2,000 episodes 0.023 against 0.011, both greedy policies at 0.730. So say: with little data the planned greedy policy is the better one; with more, the planned values are about twice as accurate. At 250 episodes the comparison flips with the seed (sup-norm errors are set by rarely visited cells), so it is not printed. Claim: on the same transitions the model's values are about twice as accurate (Q-learning needs about twice the data to match), at the price of storing |S|^2|A| counts and solving the model (500 sweeps, about 10 ms here). Exercise ex_qlearning-like-like-comparison-value keeps its cost analysis (parts a, c); its part b is now this cell. Dyna, MCTS and world models stay in ch. 15. -->

```{.python .input #qlearning-planning-in-a-learned-model-1}
%%tab pytorch, jax
class Counts(gym.Wrapper):
    """Pass every step through and count it: N[s, a, s'] and summed rewards."""
    def __init__(self, env):
        super().__init__(env)
        n_s, n_a = env.observation_space.n, env.action_space.n
        self.N, self.R = np.zeros((n_s, n_a, n_s)), np.zeros((n_s, n_a))

    def reset(self, **kwargs):
        self.s, info = self.env.reset(**kwargs)
        return self.s, info

    def step(self, a):
        s_next, r, terminated, truncated, info = self.env.step(a)
        self.N[self.s, a, s_next] += 1
        self.R[self.s, a] += r
        self.s = s_next
        return s_next, r, terminated, truncated, info

def learned_model(counts, gamma):
    """Certainty equivalence: the empirical kernel and mean rewards."""
    n = np.maximum(counts.N.sum(axis=-1), 1)   # unvisited pairs: zero rows
    return d2l.TabularMDP(counts.N / n[..., None], counts.R / n, gamma)
```

```{.python .input #qlearning-planning-in-a-learned-model-2}
%%tab pytorch, jax
def grade(Q):   # sup-norm value error, exact success of the greedy policy
    return (np.abs(Q.max(axis=-1) - V_star).max(),
            success(np.eye(mdp.num_actions)[Q.argmax(axis=-1)]))

for budget in [500, 2000]:
    rows = []
    for seed in range(5):
        counts, Qs = Counts(env), np.zeros((mdp.num_states, mdp.num_actions))
        for _ in q_learning(seed, Qs, np.zeros_like(Qs), counts, budget,
                            gamma):
            pass
        model = learned_model(counts, gamma)
        Q_model = model.backup(d2l.value_iteration(model, 500)[-1])
        rows.append([counts.N.sum(), *grade(Qs), *grade(Q_model)])
    n, err_q, win_q, err_m, win_m = np.median(rows, axis=0)
    print(f'{budget} episodes, median {n:,.0f} steps (medians over 5 seeds)')
    print(f'  Q-learning:        max |V - V*| {err_q:.3f}, success {win_q:.3f}')
    print(f'  planning in model: max |V - V*| {err_m:.3f}, success {win_m:.3f}')
```

## Off-Policy Learning

### The Off-Policy Mechanism

The $\max_{a'}$ in :eqref:`eq_td_error` evaluates a greedy continuation
independently of the action that the behavior policy actually takes at
$s'$. Q-learning is therefore *off-policy*: it estimates the value of the
greedy policy while data are generated by the exploratory policy $\pi_e$.
SARSA is the corresponding on-policy method; it bootstraps from the next
action actually selected and estimates the value of the behavior policy,
including its exploration :cite:`Sutton.Barto.2018`. This distinction
explains why Q-learning can use persistent exploration while its reported
greedy policy agrees with $\pi^*$. It also permits learning from older
experience, another agent, or a fixed log, provided the data continue to
cover the state--action pairs required by the target policy. Function
approximation can make this flexibility unstable (:numref:`sec_dqn`),
and a fixed dataset imposes the stronger coverage constraints studied in
:numref:`sec_offline`.

### Maximization Bias

The maximum of noisy estimates tends to exceed the maximum of their expectations because it selects entries with positive estimation errors. This produces *maximization bias* in bootstrapped targets. In the experiment, four of five final estimates of $Q(s_0,\textrm{left})$ exceed the true value $0.180$, and the fifth agrees to the reported precision. The bias decreases as tabular estimates converge, but it can persist under function approximation. Double Q-learning reduces it by using separate estimators to select and evaluate the maximizing action, although the resulting estimates can also be low :cite:`vanHasselt.2010`. :numref:`sec_dqn` studies this method.

## Summary

Q-learning replaces the expectation in a Bellman backup with a sampled transition. Its temporal-difference update is $Q \leftarrow Q + \alpha \delta$. In stochastic environments, this semi-gradient update should not be confused with minimizing a sampled squared Bellman residual, whose population objective contains an additional variance term. Under sustained visitation and Robbins--Monro step sizes, tabular Q-learning converges to $Q^*$. The maximum in the target makes Q-learning off-policy and also introduces maximization bias.

Replacing the table by a neural network gives the deep Q-learning methods developed in :numref:`sec_dqn`.

**Experimental scope.** The FrozenLake results use five seeds, one finite state space, and a fixed training budget. They show that tabular Q-learning approaches the exact solution under the tested schedule, but they do not establish an optimal schedule or a scaling law. The reported exploration cost is specific to the geometry of this environment.

## Exercises

1. **Batch TD against Monte Carlo.** A Markov reward process is an MDP
   without actions: each state has a fixed distribution over the reward
   and the successor state. One such process has two
   nonterminal states $x$ and $y$, and $\gamma = 1$. A batch holds eight
   episodes: one moves from $x$ to $y$ with reward $0$ and then terminates
   with reward $0$; six start in $y$ and terminate with reward $1$; one
   starts in $y$ and terminates with reward $0$. TD(0) is the state-value
   form of :eqref:`eq_td_error` without the maximum,
   $V(s) \leftarrow V(s) + \alpha \big(r + \gamma V(s') - V(s)\big)$.
    1. Compute the Monte Carlo estimates of $V(x)$ and $V(y)$ from the
       batch, and the values to which TD(0) converges when it replays the
       batch many times with a small step size.
    1. Which pair of estimates has the smaller squared error on the eight
       observed returns?
    1. Estimate the transition probabilities and the expected rewards of
       the process by counting, and solve its Bellman equation exactly.
       Which of the two pairs of estimates does the solution reproduce?
    1. Which pair would you expect to predict the returns of future
       episodes from $x$ better, and under which assumption about the
       process?

    *Adapted from Richard S. Sutton and Andrew G. Barto,
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf),
    Example 6.4.*
   :exlabel:`ex_qlearning-batch-td-monte-carlo`

1. **Truncation treated as termination.** A task has one state and one
   action, pays reward $1$ at every step, and never terminates. A time
   limit truncates every episode after $H$ steps, as `FrozenLake-v1` does
   after $100$. A faulty implementation masks the bootstrap with
   `terminated or truncated` instead of `terminated`.
    1. With decaying step sizes, the estimate converges to the value at
       which the expected update over the $H$ transitions of an episode
       vanishes. Find this value as a function of $\gamma$ and $H$, and
       compare it with the correct value $1/(1 - \gamma)$ at
       $\gamma = 0.95$ and $H = 100$. Which discount factor does the
       faulty implementation effectively use?
    1. Suppose instead that the time limit is part of the task, so that
       the return really ends after $H$ steps. Show that no single number
       is the correct value of the state. Add the number $h$ of remaining
       steps to the state, as in problem :exref:`ex_mdp-time-limits`
       of :numref:`sec_mdp`. Find the value of each state $(s, h)$, and show
       that the mask `terminated or truncated` is then correct.
   :exlabel:`ex_qlearning-truncation-termination`

1. **Greedy SARSA and Q-learning.** SARSA replaces the maximum in
   :eqref:`eq_td_error` by the value of the action $a'$ that the behavior
   policy selects at $s'$, $\delta = r + \gamma Q(s', a') - Q(s, a)$. An
   implementation selects $a'$ before it updates $Q(s, a)$ and then takes
   $a'$ at the next step. Suppose that both methods select actions
   greedily on their current tables and break ties by the same fixed rule.
   Are SARSA and Q-learning then the same algorithm, making the same action
   selections and the same updates? If they differ, construct a transition
   of the calm lake and table values for which the two methods select
   different actions at the next step.

    *Adapted from Richard S. Sutton and Andrew G. Barto,
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf),
    Exercise 6.12.*

1. [code] **The double-sampling counterexample.** ● Build a three-state
   MDP with one action per state. From state $s$, the action gives reward
   $0$ and moves to $u_1$ or $u_2$ with probability $1/2$ each. From $u_1$,
   it gives reward $1$ and terminates; from $u_2$, it gives reward $0$ and
   terminates. Write $Q(s)$, $Q(u_1)$ and $Q(u_2)$ for the three entries.
    1. Differentiate one term of the sampled objective
       :eqref:`q_learning_optimization_problem` with respect to the table.
       Which entries have nonzero derivatives, and which of these
       derivatives does the semi-gradient update :eqref:`eq_td_error` drop?
    1. Write down $Q^*$ for the three entries.
    1. With $\mu$ uniform over the three pairs, write the population
       objective :eqref:`eq_double_sampling` explicitly and minimize it in
       closed form as a function of $\gamma$. Which entries does the
       minimizer distort relative to $Q^*$, and in which direction?
    1. Visit the three pairs equally often and run, with a decaying step
       size, both the semi-gradient update and stochastic gradient descent
       on the full derivative of part (a). Confirm that each converges to
       the point predicted in parts (b) and (c).
    1. Suppose that the environment can be reset to $s$, so that every
       visit to $s$ yields two independent successors. Use one of them in
       the target and the other in the derivative of part (a). Show that
       the resulting stochastic gradient is unbiased for the gradient of
       the squared Bellman residual, and confirm by simulation that it
       converges to $Q^*$.
   :exlabel:`ex_qlearning-double-sampling-counterexample`

1. [code] **Step sizes under a finite budget.** ● The section compares a
   constant step size with schedules $\alpha = 1/(1 + c\, n)$ for
   $c = 0.1$ and $c = 1$, where $n$ counts the visits to the entry,
   including the current one. For the analysis, treat the targets that
   one entry receives as independent draws with mean $m$ and variance
   $\sigma^2$.
    1. For a constant $\alpha$, find the variance of the estimate after
       many updates. Evaluate it at $\alpha = 0.9$ for
       $Q(s_0, \textrm{left})$, taking as targets $r + \gamma V^*(s')$ with
       $s'$ drawn from `mdp.P[0, a0]` and `V_star` from the section, and
       compare the resulting standard deviation with the spread of the five
       final constant-step estimates, $0.06$ to $0.27$ around the exact
       value $0.180$. Which assumption of the analysis fails for this
       entry, and where does the rest of the spread come from?
    1. For $c = 1$, write the estimate after $n$ visits as a weighted sum
       of the initial value and the $n$ targets. Which weight does the
       zero initialization receive?
    1. For general $c$, write the weight of the $k$-th target after $n$
       visits as a product, and compare the logarithm of the product with
       an integral to show that the weight grows like $k^{1/c - 1}$ in $k$
       for large $k \leq n$. How does $c = 0.1$ treat the targets that were
       computed from the early, inaccurate table?
    1. Sweep $c \in \{0.01, 0.03, 0.1, 0.3, 1\}$ on the slippery lake,
       five seeds each, and plot against $c$ both the final error
       $|\hat{Q}(s_0, \textrm{left}) - Q^*(s_0, \textrm{left})|$ and the
       sup-norm error $\|\hat{Q} - Q^*\|_\infty$ over all entries. Explain
       the shape of the curves with parts (a) and (c). Do the two error
       measures select the same $c$?

1. [code] **Samples against backups.** ● Take `Q_star` as ground truth.
    1. Record $\|\hat{Q} - Q^*\|_\infty$ after every episode of
       `q_learning` on the slippery lake and plot it against the number of
       environment steps, which the `visits` ledger counts. On the same
       axes, plot the error of value iteration, with $Q_k$ = `mdp.backup`
       of the $k$-th iterate of `d2l.value_iteration`, against the number
       of Bellman backups at $64$ per sweep. How many backups does value
       iteration need to reach the error that Q-learning reaches after
       $4000$ episodes?
    1. Add to `q_learning` a count of the observed transitions
       $(s, a, s')$ and a sum of the observed rewards for each pair. After
       $100$, $400$, $1000$ and $4000$ episodes, build the estimated model
       `d2l.TabularMDP(P_hat, r_hat, gamma)`, leaving the rows of unvisited
       pairs at zero, solve it with `d2l.value_iteration`, and add the
       error of its action values to the plot at the same environment
       steps. Which estimate makes better use of the samples, and what
       does it cost in computation? Relate the result
       to exercise :exref:`ex_qlearning-batch-td-monte-carlo`.
    1. Value iteration consumes model backups, Q-learning consumes
       environment steps, and the estimated model of part (b) consumes
       both. Suppose that one environment step costs as much computation
       as $b$ backups. Using your measurements, find for which range of
       $b$ each of the three methods reaches the error of Q-learning after
       $4000$ episodes at the lowest total cost. Which of the methods
       remain available to an agent that cannot read the transition
       kernel?
   :exlabel:`ex_qlearning-like-like-comparison-value`

1. [code] **Maximization bias in a two-state task.** ● Every episode
   starts in state $s_0$, which offers two actions: *right* terminates with
   reward $0$, and *left* moves to state $s_1$ with reward $0$. In $s_1$,
   each of $m$ actions terminates with a reward drawn from
   $\mathcal{N}(-0.1, 1)$, so *left* is a mistake. Use $\gamma = 1$, the
   constant step size $\alpha = 0.1$, zero initial values, and
   `epsilon_greedy` with $\epsilon = 0.1$.
    1. Implement tabular Q-learning for $m = 10$, and plot against the
       episode index the fraction of runs that take *left* in $s_0$,
       averaged over $1000$ runs of $300$ episodes. What fraction would an
       $\epsilon$-greedy agent acting on $Q^*$ show?
    1. Double Q-learning :cite:`vanHasselt.2010` keeps two tables $Q_1$
       and $Q_2$, acts $\epsilon$-greedily on $Q_1 + Q_2$, and flips a fair
       coin at each transition. On heads it updates
       $$Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \big( r + \gamma\, Q_2(s', \mathrm{argmax}_{a'} Q_1(s', a')) - Q_1(s, a) \big),$$
       and on tails it applies the same update with the two tables
       exchanged. Add its curve to the plot.
    1. Every reward reachable through *left* has a negative mean. Explain
       how Q-learning's estimate of $Q(s_0, \textrm{left})$ nevertheless
       becomes positive. Predict how the Q-learning curve changes for
       $m = 1$ and for $m = 100$, and check. Which part of the effect at
       $m = 100$ comes from noisy estimates, and which from actions that
       have never been tried?

    *Adapted from Richard S. Sutton and Andrew G. Barto,
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf),
    Example 6.7.*

[Discussions](https://d2l.discourse.group/)

[Discussions](https://d2l.discourse.group/t/36797)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14.4]{.kicker}

Temporal differences and Q-learning<br>
**sampled backups · temporal-difference error · convergence · off-policy learning**
:::
:::

::: {.slide title="The Sampled Backup"}
Value iteration needs $P$ inside one expectation. Replace it with the
transition you just observed:

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a), \qquad
Q(s, a) \leftarrow Q(s, a) + \alpha\, \delta$$

- $\delta$ is the **temporal-difference error**: the sampled one-step
  target minus the current table estimate.
- Bootstrap masked by `terminated`, never `truncated`.

. . .

:numref:`fig_rl_backups`, panel (d): one blue branch instead of the sum,
the max at the next state kept.
:::

::: {.slide title="What Actually Converges"}
The sampled least-squares objective is *not* the right justification:

$$L(Q) = \mathbb{E}_\mu \big[ (Q - TQ)^2 \big] + \gamma^2\, \mathbb{E}_\mu \big[ \textrm{Var}_{s'} ( \max_{a'} Q(s', a') ) \big]$$

$Q^*$ makes the first term zero, but the variance term can change the
minimizer (**double sampling**). It vanishes for deterministic transitions.

. . .

The expected update satisfies $\mathbb{E}[\delta \mid s, a] = (TQ)(s, a) - Q(s, a)$,
zero exactly at $Q^*$. A stochastic approximation of value iteration,
convergent under Robbins-Monro steps.
:::

::: {.slide title="The Update in Code"}
@qlearning-the-implementation-3

. . .

@qlearning-the-implementation-4
:::

::: {.slide title="Graded Against Dynamic Programming"}
This finite example retains the exact MDP for evaluation.

@!qlearning-reading-the-curve-honestly-1

. . .

The table is within hundredths of $V^*$, and the greedy policy agrees with
$\pi^*$ within sampling error. With $\epsilon=0.05$, continuing exploration
reduces the measured success rate of the behavior policy to $54\%$.
:::

::: {.slide title="Step Sizes and Finite Budgets"}
@!qlearning-reading-the-curve-honestly-4

. . .

- constant $0.9$: estimates continue to fluctuate ($0.06$ to $0.27$)
- $1/(1 + 0.1 n)$: converged, leaning slightly high
- $1/(1 + n)$: satisfies Robbins--Monro but leaves **all five seeds** below $0.025$ at this budget
:::

::: {.slide title="Which Policy Is Being Learned"}
- The $\max_{a'}$ ignores the action the behavior took: **off-policy**.
  Learn about the greedy policy from data collected by any policy.
- One symbol away: SARSA bootstraps on the action taken, learning the
  behavior's value, $\epsilon$ floor and all.
- The $\max$ also produces positive bias here: four of five final estimates
  exceed the true $0.180$, and none are below it. Maximization bias, repaired in
  :numref:`sec_dqn`.
:::

::: {.slide title="Recap"}
- TD error :eqref:`eq_td_error`: the one-step residual; reused by every
  algorithm ahead.
- Convergence follows from the expected update's fixed point, not from
  minimizing the sampled squared objective (double sampling).
- Robbins-Monro is necessary for the guarantee; budgets decide between
  schedules that both pass.
- Online feedback: selecting an overestimated action provides data that can
  correct its estimate; this mechanism is absent offline.
- The first $\max$ defines a greedy off-policy target; the same $\max$ can also introduce maximization bias.
:::
