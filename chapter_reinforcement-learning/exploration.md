# Exploration
:label:`sec_exploration`

Q-learning in :numref:`sec_qlearning` assumes continued exploration. To isolate the resulting tradeoff, we study a multi-armed bandit, which removes state transitions and delayed rewards.

```{.python .input #exploration}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import gymnasium as gym
import numpy as np
```

```{.python .input #exploration}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import gymnasium as gym
import numpy as np
```

## Exploration and Regret

### The Multi-Armed Bandit

To isolate exploration, use the one-state MDP from :numref:`sec_mdp`.
Without transitions or delayed rewards, the resulting *multi-armed
bandit* requires the agent to balance estimating action values against
selecting actions with high expected reward. We use ten Bernoulli arms,
one of which has the largest success probability:

```{.python .input #qlearning-a-bandit-is-an-mdp-with-one-state}
%%tab pytorch, jax
arms = np.array([0.50, 0.42, 0.90, 0.25, 0.55, 0.38, 0.60, 0.32, 0.50, 0.45])

def bandit_run(rule, num_pulls, seed):
    """One state, one step: pull an arm, log the shortfall, update tallies."""
    rng = np.random.default_rng(seed)
    wins, count = np.zeros(len(arms)), np.zeros(len(arms))
    gap = np.empty(num_pulls)
    for t in range(num_pulls):
        a = rule(wins, count, t, rng)
        count[a] += 1
        wins[a] += float(rng.random() < arms[a])
        gap[t] = arms.max() - arms[a]
    return np.cumsum(gap)
```

A bandit policy is any rule from the running tallies to an arm, so
`epsilon_greedy` applies unchanged to the empirical means. In a
*contextual bandit*, a fresh context arrives each round independently of
previous actions, one action is scored, and the episode ends. The model
may generalize across contexts, but there are no controlled dynamics or
delayed rewards. :numref:`sec_rl_sequences` uses this distinction when
classifying single-turn preference optimization.

### Regret and the Epsilon-Greedy Family

A bandit permits evaluation by cumulative regret rather than success rate. Each pull contributes the gap between the best mean $\mu^* = \max_a \mu_a$ and the selected arm's mean,

$$\textrm{regret after } t \textrm{ pulls} = \sum_{u=1}^{t} \big( \mu^* - \mu_{a_u} \big),$$

which is what `bandit_run` accumulates. Its expectation is $\sum_a (\mu^* - \mu_a)\, \mathbb{E}[n(a)]$: the gap of each arm multiplied by its expected number of pulls. *Sublinear* regret means that average regret per pull converges to zero. We compare three members of the $\epsilon$-greedy family over twenty independent runs:

```{.python .input #qlearning-regret-and-the-linear-tax}
%%tab pytorch, jax
def eps_rule(schedule):
    """Epsilon-greedy on the empirical means, with epsilon = schedule(t)."""
    return lambda wins, count, t, rng: d2l.epsilon_greedy(
        wins / np.maximum(count, 1), schedule(t), rng)

rules = {'greedy': eps_rule(lambda t: 0.0),
         'epsilon = 0.1': eps_rule(lambda t: 0.1),
         'annealed': eps_rule(lambda t: max(0.02, 1 / np.sqrt(t + 1)))}
regret = {name: np.array([bandit_run(rule, 2000, [2026, i, s])
                          for s in range(20)])
          for i, (name, rule) in enumerate(rules.items())}
for name, r in regret.items():
    print(f'{name:>13}: mean regret after 2000 pulls {r[:, -1].mean():6.1f}')
```

The greedy rule is sensitive to its first observations. It incurs little subsequent regret if it identifies the best arm early, but it can remain committed to a suboptimal arm; its mean regret is $824$. With fixed $\epsilon = 0.1$, the best arm is usually identified quickly, but uniform exploration continues throughout the run. For these arms, exploration alone costs approximately $0.04$ regret per pull, and the total after $2000$ pulls is $117$. Annealing $\epsilon$ reduces exploration as the estimates improve and lowers the total regret to $69$.

![Exploration, measured three ways. (a) One row of action values, $\hat Q(s, \cdot) = (0.20, 0.90, 0.55, 0.10)$, converted into greedy, $\epsilon$-greedy with $\epsilon = 0.3$, and softmax policies with temperature $0.3$. (b) Mean cumulative regret over 20 runs on the ten-armed Bernoulli bandit, shown on logarithmic axes. After $2000$ pulls, the regrets are $824$ for greedy, $117$ for fixed $\epsilon = 0.1$, $69$ for annealed $\epsilon$, $37$ for UCB with $\kappa = 0.5$, and $32$ for Thompson sampling. These finite runs do not measure asymptotic rates. The inset shows the UCB radius $\kappa \sqrt{\log t/n}$ at $t = 2000$. (c) Coverage of the 44 nonterminal state-action pairs (solid) and success rate of the training episodes (dashed) under three schedules for tabular Q-learning on deterministic FrozenLake. With random tie-breaking, the greedy policy covers 92 percent of the pairs; $\epsilon = 1$ covers nearly all pairs, but its training episodes, which never exploit, almost never succeed.](../img/mdl-rl-exploration.svg)
:label:`fig_rl_exploration`

### Optimism: UCB and Thompson Sampling

Panel (a) of :numref:`fig_rl_exploration` represents the preceding policies as distributions over a row of action values. It also includes the softmax policy introduced in :numref:`sec_policygradient`, whose temperature controls how concentrated the distribution is. The two remaining methods use uncertainty to direct exploration. Optimistic initialization starts each estimate above the attainable rewards, so an arm continues to be selected until observations lower its estimate :cite:`Sutton.Barto.2018`. The *upper confidence bound* (UCB) rule instead adds an uncertainty bonus and selects the largest upper confidence estimate:

$$a_t = \underset{a}{\mathrm{argmax}} \Big[ \hat{\mu}(a) + \kappa \sqrt{\log t \,/\, n(a)} \Big],$$
:eqlabel:`eq_ucb`

The bonus decreases with the number of observations $n(a)$ and increases slowly with $t$ while an arm is not selected :cite:`Auer.CesaBianchi.Fischer.2002`. Thus UCB adapts exploration separately for each arm. With the standard coefficient $\kappa=\sqrt{2}$, its regret is logarithmic in $t$, whereas fixed-$\epsilon$ exploration has linear regret. Smaller coefficients can work better on a particular finite problem but no longer satisfy that guarantee. Each arm is played once to initialize its index; subsequent choices are deterministic maximizers of the UCB score. Thompson sampling :cite:`Thompson.1933` takes a Bayesian approach. For Bernoulli rewards it maintains a Beta posterior for each arm, samples one mean from each posterior, and selects the arm with the largest sample. As observations accumulate, both methods concentrate their choices on well-supported arms:

```{.python .input #qlearning-optimism-ucb-and-thompson-sampling}
%%tab pytorch, jax
kappa = 0.5

def ucb(wins, count, t, rng):
    if (count == 0).any():          # play each arm once before pricing any
        return int(np.argmax(count == 0))
    return int(np.argmax(wins / count + kappa * np.sqrt(np.log(t) / count)))

def thompson(wins, count, t, rng):
    return int(rng.beta(1 + wins, 1 + count - wins).argmax())

for i, (name, rule) in enumerate([('UCB', ucb), ('Thompson', thompson)], 3):
    regret[name] = np.array([bandit_run(rule, 2000, [2026, i, s])
                             for s in range(20)])
    print(f'{name:>13}: mean regret after 2000 pulls '
          f'{regret[name][:, -1].mean():6.1f}')
d2l.plot_curves(regret, xlabel='pulls', ylabel='cumulative regret')
```

Both methods have lower regret than the $\epsilon$-greedy variants in this experiment. The curvature on the logarithmic plot is consistent with logarithmic regret, but $2000$ pulls cannot verify an asymptotic rate. The coefficient also matters: $\kappa=0.5$ was selected for these arms; $\kappa=\sqrt{2}$ gives regret near $200$, while $\kappa=0.15$ sometimes commits to a suboptimal arm and in one of twenty runs exceeds $900$. At $\kappa=0.5$, total regret ranges from $21$ to $53$ across seeds. The relative performance of UCB and Thompson sampling is therefore specific to this experiment.

**Online and offline uncertainty.** UCB adds an uncertainty term because an online agent can test optimistic estimates by collecting new observations. Offline methods often subtract an uncertainty penalty because no new data can correct an overestimate (:numref:`sec_offline`). The two terms express a related principle, but they are calibrated differently. UCB uses the online count and time, whereas an offline penalty is defined relative to a fixed dataset.

## Exploration in MDPs

Bandit actions are immediately available, whereas an uncertain state-action pair in an MDP may be reachable only through a particular sequence of actions. Under independent random exploration, the probability of executing a required sequence of $k$ actions decreases geometrically with $k$. Difficult MDPs therefore require exploration over temporally extended behavior.

One approach assigns bonuses to uncertain state-action pairs and propagates them through value backups, so the policy selects action sequences leading toward uncertain states. With function approximation, explicit counts are unavailable; alternatives include density-model pseudo-counts, curiosity based on prediction error, random network distillation, state archives, and ensembles for approximate posterior sampling :cite:`Burda.Edwards.Storkey.ea.2019`. FrozenLake does not expose this difficulty. With only 44 nonterminal state--action pairs, the exploration schedules used here provide broad coverage within the training budget.

## Exploration on the Verifier Task

[Placeholder for the writers (14.4-M7): on `VerifierTask` of :numref:`sec_mdp`, sample groups of $K$ responses at the reference policy and count the groups whose rewards are identical (all wrong on the two prompts without a matching candidate, all right once a prompt is solved); such a group gives a zero centered update (:numref:`sec_baselines` shows the algebra; here only the count). The policy's own sampling diversity is then the only exploration; one sentence on temperature and pass@$k$.]

## Summary

Exploration can be studied separately in a one-state bandit: fixed-$\epsilon$ exploration has linear regret, whereas UCB and Thompson sampling reduce exploration as uncertainty decreases.

**Experimental scope.** The bandit curves average twenty runs; the UCB coefficient was tuned for these arms, and $2{,}000$ pulls are insufficient to verify an asymptotic regret rate.

## Exercises

1. [code] **The exploration schedule.** On a zero-initialized table,
   `epsilon_greedy` breaks every tie uniformly at random, and every reward
   of FrozenLake is nonnegative.
    1. Show that every entry of the table stays exactly zero until the
       first successful episode, and that until then every $\epsilon$
       produces the same behavior. What does this imply for the episode of
       the first success as a criterion for choosing $\epsilon$?
    1. After the first success, which entries are positive? Show that an
       entry, once positive, stays positive, and that the greedy rule,
       $\epsilon = 0$, never again tries the zero entries of a row that
       contains a positive entry. Panel (c) of
       :numref:`fig_rl_exploration` shows this rule covering 92 percent of
       the nonterminal pairs of the calm lake and learning to succeed
       there. Predict whether it also learns a good policy on slippery
       ice.
    1. The optional argument `epsilon` of `q_learning` is a function from
       the episode index to the exploration rate, such as those returned by
       `linear_schedule`. On the slippery lake, run five seeds each with
       $\epsilon$ held at $0$, $0.1$, $0.5$ and $1$, and with the schedule
       `linear_schedule(1.0, 0.05, num_episodes // 2)` that `q_learning`
       uses in :numref:`sec_qlearning`. For each setting, report the mean
       training return, the success rate of the final greedy policy under
       `d2l.evaluate` over $1000$ episodes, and
       $\max_s |\max_a \hat{Q}(s, a) - V^*(s)|$.
    1. Rank the settings by training return and by greedy success rate.
       Where do the two rankings disagree, and what explains the
       disagreement?
   :exlabel:`ex_qlearning-exploration-schedule`

1. **Logarithmic regret of UCB.** ●● With $\kappa = \sqrt{2}$, the rule
   :eqref:`eq_ucb` is UCB1 of :citet:`Auer.CesaBianchi.Fischer.2002`,
   whose Theorem 1 bounds the expected regret after $t$ pulls of arms with
   rewards in $[0, 1]$ by
   $$\sum_{a:\, \Delta_a > 0} \frac{8 \ln t}{\Delta_a} + \Big(1 + \frac{\pi^2}{3}\Big) \sum_a \Delta_a, \qquad \Delta_a = \mu^* - \mu_a.$$
   Derive this bound. The tool is Hoeffding's inequality: the mean
   $\hat{\mu}_n$ of $n$ independent rewards in $[0, 1]$ with mean $\mu$
   satisfies
   $$P(\hat{\mu}_n \geq \mu + \varepsilon) \leq e^{-2 n \varepsilon^2},$$
   and likewise for the lower tail.
    1. Fix a suboptimal arm $a$ and let
       $\ell = \lceil 8 \ln t / \Delta_a^2 \rceil$. Show that if arm $a$ is
       pulled at a step $u \leq t$ while $n(a) \geq \ell$, then either the
       index of the best arm lies at or below $\mu^*$, or the empirical
       mean of arm $a$ exceeds $\mu_a$ by more than its bonus.
    1. For fixed pull counts of both arms, bound the probability of each
       of the two events at step $u$ by $u^{-4}$. Take a union bound over
       the counts and sum over $u$ to show
       $\mathbb{E}[n(a)] \leq \ell + \pi^2/3$ after $t$ pulls.
    1. Combine part (b) with the regret decomposition to obtain the
       theorem. Evaluate the bound for `arms` at $t = 2000$ and compare it
       with the regret near $200$ that the section reports for
       $\kappa = \sqrt{2}$. Which step of the proof loses the most?
    1. Repeat the argument for a general coefficient $\kappa$. For which
       $\kappa$ does the sum over steps in part (b) still converge, and
       what does the answer say about the section's choice
       $\kappa = 0.5$?
   :exlabel:`ex_qlearning-ucb-logarithmic-regret`

1. [extended] **Deep exploration.** ●● Under undirected exploration, the
   probability of executing a required sequence of $k$ actions decreases
   geometrically with $k$. Build a chain on which this decrease can be
   computed exactly: states
   $0, \ldots, N-1$, start in state $0$, action $0$ returns to state $0$,
   action $1$ advances one state, reaching state $N-1$ gives reward $1$
   and terminates, and episodes are truncated after $2N$ steps. Give the
   chain the `reset` and `step` interface of Gymnasium so that
   `q_learning`, with the `epsilon` argument of exercise
   :exref:`ex_qlearning-exploration-schedule`, runs on it unchanged.
    1. Show that the uniformly random policy succeeds in one episode with
       probability at most $2N \cdot 2^{-(N-1)}$, so that the expected
       number of episodes up to the first success is at least
       $2^{N-2}/N$. By the argument of problem
       :exref:`ex_qlearning-exploration-schedule`a, $\epsilon$-greedy
       Q-learning from a zero table acts uniformly at random until its
       first success, so the bound holds for it whatever the value of
       $\epsilon$.
    1. Measure the episode of the first success for
       $N \in \{4, 6, \ldots, 20\}$, five seeds each, and plot it against
       $N$ on a logarithmic axis together with the bound.
    1. Initialize every entry to $1$, an upper bound on every return, and
       act greedily with $\epsilon = 0$. Measure the same quantity, and
       explain its growth with $N$ by tracing which entries the first
       episodes lower.
    1. Move both agents to
       `gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)`, which
       also truncates at $100$ steps. Compare the optimistic greedy agent
       with the zero-initialized agent under the schedule
       `linear_schedule(1.0, 0.05, num_episodes // 2)` at $4000$ and
       $20000$ episodes, five seeds each. Report the success rate of the
       greedy policy under `d2l.evaluate`, the environment steps used,
       and, as a reference, the success rate of the optimal policy
       computed by value iteration. Does optimism keep its advantage on a
       stochastic map, and what does it cost?

   The whole project takes about five minutes of CPU time.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §14.4]{.kicker}

Exploration<br>
**bandits and regret · epsilon-greedy · UCB and Thompson sampling · exploration in MDPs**
:::
:::

::: {.slide title="Exploration Measured by Regret"}
A bandit is an MDP with one state. Regret assigns each pull the mean-reward
gap from the best arm.

![](../img/mdl-rl-exploration.svg){width=98%}

. . .

Greedy $824$ · fixed $\epsilon$ $117$ (linear regret) · annealed $69$ ·
UCB $37$ · Thompson $32$.
:::

::: {.slide title="Optimism and Uncertainty"}
$$a_t = \mathrm{argmax}_a \big[ \hat{\mu}(a) + \kappa \sqrt{\log t / n(a)} \big]$$

Per-arm exploration decreases as observations accumulate: logarithmic regret
under the stated UCB conditions, whereas fixed $\epsilon$ has linear regret (proved at $\kappa = \sqrt 2$; play each arm once
first). Thompson: sample a Beta posterior, play the argmax.

. . .

**Online and offline uncertainty.** Online exploration adds a confidence
radius that decreases with counts; offline pessimism (:numref:`sec_offline`)
subtracts an uncertainty penalty because no new samples can correct errors.
:::
