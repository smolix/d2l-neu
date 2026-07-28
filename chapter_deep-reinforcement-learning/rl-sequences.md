# Sequences Are Trajectories
:label:`sec_rl_sequences`

The most economically important use of reinforcement learning today, training language models, runs on almost none of the machinery of these two chapters, and seeing exactly which parts collapse is the sharpest statement of what each part was for. This closing section reads text generation as a Markov decision process, proves that the token-level and response-level views give the same gradient, and sorts everything built since :numref:`sec_mdp` into what the translation deletes and what it keeps. :numref:`sec_offline` ended on the data question for gridworlds; here the policy has a hundred-thousand-action vocabulary and a horizon of thousands of tokens, yet the setting is not harder but *degenerate*, in exactly the way :numref:`sec_mdp` planted. Because the section closes both chapters, it ends with the road onward and four capstone projects.

```{.python .input #rl-sequences-sequences-are-trajectories}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
```

```{.python .input #rl-sequences-sequences-are-trajectories}
%%tab jax
from d2l import jax as d2l
import numpy as np
```

## The Dictionary

### Prompt, token, prefix, response

A language model with parameters $\theta$ maps a context to a distribution over the next token: the softmax head of :numref:`sec_gpt`. Sampling repeatedly, appending each drawn token and stopping at EOS, is the `generate` loop of :numref:`sec_decoding`. Read the context as a state and the token as an action, and generation *is* the agent-environment loop of these two chapters, entry by entry.

:Text generation read as a decision process. The left column is the Language Models part's vocabulary; the right column is what these two chapters built.
:label:`tab_rl_sequence_dictionary`

| language modeling | reinforcement learning |
|:--|:--|
| prompt $x$ | start state $s_0$, drawn from the prompt distribution $\mu_0$ of :numref:`sec_mdp` |
| token $y_t$ | action $a_t$ |
| prefix $(x, y_{<t})$ | state $s_t$ |
| response $y = (y_1, \ldots, y_T)$ | trajectory $\tau$ |
| EOS | terminal state |
| the next-token softmax of :numref:`sec_gpt` | the policy $\pi_\theta(a \mid s)$ of :numref:`sec_policygradient` |
| the `generate` loop of :numref:`sec_decoding` | the `rollout` of :numref:`sec_policygradient` |

Two rows deserve a second look. The state is the *prefix*, not the last token: :numref:`sec_mdp`'s Markov assumption demanded a state carrying everything the past could say about the future, and the prefix is the entire past. And the start state is drawn from $\mu_0$: :numref:`sec_policygradient` noted that this factor drops out of every gradient and asked you to hold on to it anyway; the distribution over prompts is that factor, shaping everything about the trained policy while passing through none of the derivations, which is why the estimators transfer unchanged.

### Deterministic transitions

The transition kernel of this MDP is string concatenation: from prefix $s_t$ with token $a_t$, the next state is $(s_t, a_t)$ with probability one, and the reward is terminal, one number $r(x, y)$ when the response ends. This is the degenerate corner :numref:`sec_mdp` set aside, with the promise that the degeneracy would *remove* terms from our algorithms rather than add any. In :numref:`sec_policygradient` the trajectory probability :eqref:`eq_traj_prob` carried transition factors, and the derivation needed an argument for why they vanish from the score; here each factor *is the constant one* before anybody differentiates. Every "the transitions cancel" argument in these chapters becomes trivial rather than subtle, all randomness in a rollout is the policy's own sampling, and the model whose absence drove :numref:`sec_qlearning` is perfectly known: it is concatenation.

### The factorization proposition

One identity connects "a policy over responses" to "a next-token softmax"; it is the single equation the Language Models part needs from this book.

**Proposition.** For the token MDP of :numref:`tab_rl_sequence_dictionary`, the response-level and token-level views give the same gradient:

$$\nabla_\theta \log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}).$$
:eqlabel:`eq_seq_factorization`

*Moreover, with a terminal-only reward $r(x, y)$ and no discounting, every reward-to-go in the response is the same number, so a single response-level weight $r(x, y) - b(x)$ multiplies every token's score, without bias for any baseline $b$ that depends only on the prompt.*

**Proof.** The chain rule of probability factorizes the response's likelihood exactly, $\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t})$, with no transition factors because the transitions are deterministic; taking logarithms and gradients gives :eqref:`eq_seq_factorization`. With rewards zero until EOS and $\gamma = 1$, the reward-to-go of :numref:`sec_baselines` is $\hat{G}_t = r(x, y)$ at every $t$; subtracting a prompt-only baseline $b(x)$ is licensed bias-free by the zero-mean lemma of :numref:`sec_baselines`, leaving the weight $r(x, y) - b(x)$ on every score. $\blacksquare$

Read the two halves as one collapse. The left side of :eqref:`eq_seq_factorization` treats the response as *one action* chosen at the prompt; the right side treats it as $T$ actions chosen token by token; REINFORCE :eqref:`eq_reinforce` computes the identical update either way. One naming rule keeps the objects straight: $r(x, y) - b(x)$ is a response-level Monte Carlo weight that happens to multiply every token's score. It is not the token advantage $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ of :numref:`sec_actorcritic`, whose two terms condition on the prefix and which would differ from token to token if anything estimated it. And since every token carries the same weight, the token-level view buys no credit assignment *within* a response: which of ten thousand tokens earned the terminal reward is a question the estimator does not answer, which is why per-token credit is the open problem at scale. Collapsed to one action per episode, the problem lands on a rung :numref:`sec_qlearning` named: tuning a language model on single-turn scored responses is a *contextual bandit* wearing reinforcement learning's clothes, and the sequential machinery becomes necessary again exactly when the loop closes, in multi-turn dialogue and tool use.

## What Collapses and What Survives

Sorting both chapters through the dictionary is the payoff of having built everything explicitly; :numref:`fig_rl_token_mdp` draws the sort.

### What collapses

Nine load-bearing ideas lose their purchase. *Discounting*: a response is finite and paid once, so $\gamma = 1$ and the knob is gone. *Bootstrapping and the TD error*: :numref:`sec_actorcritic` built targets from predictions to learn before episodes end; with one terminal reward there is nothing useful to bootstrap toward. *The Bellman equations and value iteration*: they propagate value through a state space; here the states are all texts, visited once each, over a kernel that needs no solving. *Q-learning and replay*: :numref:`sec_dqn`'s off-policy machinery reused stale transitions under a moving greedy target; the methods below stay near on-policy and pay for fresh samples. *The deadly triad*: no bootstrapping, no triad. *Per-token credit assignment*: not solved but forfeited; every token inherits one scalar. Every deletion on the list is conditional on the corner's assumptions, a terminal-only reward over deterministic concatenation in a single turn: process rewards restore intermediate credit, tool calls and environment feedback restore a nontrivial kernel, multi-turn interaction restores the loop, and with each, the struck machinery returns.

### What survives

What remains is precisely the policy-optimization spine. *The score function* :eqref:`eq_softmax_score` and REINFORCE, unchanged. *Baselines and advantages*: the zero-mean lemma asks only that $b$ ignore the action, and the prompt is the natural unit. *Variance growing with length*: :eqref:`eq_seq_factorization` sums $T$ score terms. *On-policy staleness*: a model that just updated no longer matches the responses it just generated. *Importance ratios and clipping*: :numref:`sec_ppo`'s repair for reusing a batch a few epochs. *Trust regions in policy space*, because parameter distance still lies about policy distance. *Entropy collapse*: the unregularized optimum is still a point mass, as :numref:`sec_regularized` proved, and unregularized training still drifts toward it. And *over-optimization of an estimated objective*, with its two cures. The survivors share one trait: none ever consulted the transition kernel; everything that leaned on the kernel or on intermediate reward collapsed with it.

![The token MDP, and what it deletes. Top: a response is a trajectory whose states are prefixes, whose actions are tokens, and whose transitions append the chosen token with probability one, so all randomness is the policy's; the reward arrives once, on the terminal edge into EOS. Below: the same object collapsed to one step, a single draw $y \sim \pi_\theta(\cdot \mid x)$ scored by $r(x, y)$, the two views sharing one gradient by :eqref:`eq_seq_factorization`. Right: what simplifies away under these assumptions, struck through, and what survives; the struck machinery returns with process rewards, tool calls or multi-turn feedback.](../img/mdl-rl-token-mdp.svg)
:label:`fig_rl_token_mdp`

### The smallest instance

The one-step view invites the smallest laboratory in either chapter: no environment object at all, because concatenation needs no simulator. Four "prompts", nine candidate "responses" shared across them, a verifier that checks a response exactly, and a reference policy $\pi_{\textrm{ref}}$, the stand-in for a pretrained model, which spreads its mass over the honest candidates and *hedges*, listing every number it has seen, with probability about $0.2$ percent. Two prompts are deliberately beyond this model, no candidate matches their answers, so the verifier's ceiling is $0.5$. The policy is :eqref:`eq_softmax_policy`, one preference row per prompt; training starts from the reference, as post-training does.

```{.python .input #rl-sequences-the-smallest-instance}
%%tab pytorch, jax
prompts = ['2+2', '3*4', '8*8', '9*9']     # four prompts
answers = ['4', '12', '64', '81']          # their checked answers
resp = ['1', '3', '4', '7', '12', '19', '23', '35',
        'one of 4, 12, 64 or 81']          # nine candidate responses

def verify(x, y):                          # the verifier: exact match
    return float(resp[y] == answers[x])

def policy(theta):                         # pi(y | x), one row per prompt
    z = np.exp(theta - theta.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)

def success(pi, reward):                   # E r(x, y) over sampled y
    R = [[reward(x, y) for y in range(len(resp))] for x in range(4)]
    return (pi * np.array(R)).sum(1).mean()

theta_ref = np.zeros((4, 9))
theta_ref[:, 8] = -4.0                     # the reference rarely hedges
pi_ref = policy(theta_ref)
print(f'success of a response sampled from the reference: '
      f'{success(pi_ref, verify):.3f}')
```

### The group is the baseline

Now run the survivors, and nothing but the survivors. Sample a group of $K$ responses per prompt, score each, standardize the scores within the group, and take one ascent step on the log-probabilities: the normalization of :eqref:`eq_pg_normalized` with the group as the batch and the prompt as the start state. The group mean is the per-prompt baseline, so no value network exists anywhere. One term rides along for later: the KL penalty of :numref:`sec_regularized` to the frozen reference, folded into each sample's reward as $r - \beta \log \big( \pi_\theta(y \mid x) / \pi_{\textrm{ref}}(y \mid x) \big)$, the simplest way to charge it; the named method assembled at the end of the section keeps its KL as a separate loss term instead. At $\beta = 0$ it is inert.

```{.python .input #rl-sequences-the-group-is-the-baseline-1}
%%tab pytorch, jax
def group_step(theta, K, reward, rng, beta=0.0, lr=2.0):
    pi, g = policy(theta), np.zeros_like(theta)
    for x in range(len(prompts)):              # one group of K per prompt
        ys = rng.choice(len(resp), K, p=pi[x])
        r = np.array([reward(x, y) for y in ys])
        r = r - beta * np.log(pi[x, ys] / pi_ref[x, ys])   # the KL penalty
        A = d2l.normalize(r)                   # the group mean as baseline
        g[x] = (np.eye(len(resp))[ys] - pi[x]).T @ A / K   # eq_softmax_score
    return theta + lr * g
```

One bookkeeping fact must be stated before the sweep, because :numref:`sec_baselines` proved it and the estimator above quietly trades on it. The zero-mean lemma licenses only baselines that ignore the sample's own action, and the group mean does not qualify: it contains the current sample's reward, so same-group centering is a *biased* estimator whose expectation is exactly $(K - 1)/K$ of the true gradient, the shrinkage :numref:`sec_baselines` derived, and dividing by the group's standard deviation stacks a random denominator on top. The exactly unbiased repair is also that section's: leave-one-out, each sample measured against the mean of the other $K - 1$. On this toy, both facts can be checked by enumerating every group of two, with the centering isolated from the standardization:

```{.python .input #rl-sequences-the-group-is-the-baseline-3}
%%tab pytorch, jax
K, pi = 2, policy(theta_ref)
Rv = np.array([[verify(x, y) for y in range(len(resp))] for x in range(4)])
S = np.eye(len(resp))[None, :, :] - pi[:, None, :]   # eq_softmax_score
g = np.einsum('xy,xy,xyv->xv', pi, Rv, S)            # the exact gradient
u = np.zeros_like(g)
for y1, y2 in np.ndindex(len(resp), len(resp)):      # every group of K = 2
    w = (Rv[:, y1] - Rv[:, y2]) / 2                  # reward minus group mean
    u += (pi[:, y1] * pi[:, y2] * w)[:, None] * (S[:, y1] - S[:, y2]) / K
print(f'E of the group-centered update is (K-1)/K of the gradient: '
      f'{np.allclose(u, g * (K - 1) / K)}')
print(f'rescaled by K/(K-1), leave-one-out, it is exact: '
      f'{np.allclose(u * K / (K - 1), g)}')
```

Now the prediction the sweep must confirm. At $K = 1$ the shrinkage factor $(K - 1)/K$ is zero: the group mean *is* the sample's own reward, the standardized advantage is identically zero, and the parameters never move, however many samples are spent. This is not "no relative information to extract" but self-inclusion bias at its maximum, the limiting case of the fact just verified; the true policy gradient at $K = 1$ is of course nonzero, and plain REINFORCE without the baseline would learn here. The $K = 1$ row below must therefore print exactly the reference's $0.062$, no learning at all. The sweep holds the *sample budget* fixed at $6400$ scored responses per prompt, so small groups get proportionally more updates and no arm is favored:

```{.python .input #rl-sequences-the-group-is-the-baseline-2}
%%tab pytorch, jax
budget = 6400                    # scored responses per prompt, matched
for K in (1, 2, 4, 8, 32):
    rng, theta = np.random.default_rng(1), theta_ref.copy()
    for _ in range(budget // K):
        theta = group_step(theta, K, verify, rng)
    print(f'K = {K:2d}: success of a sampled response '
          f'{success(policy(theta), verify):.3f}')
```

The prediction lands to the third decimal, and every $K \geq 2$ reaches the verifier's ceiling of $0.5$ on the same budget: at any $K \geq 2$ the $(K-1)/K$ shrinkage is a rescaling the ascent shrugs off, and the bias that stops learning is only the total one at $K = 1$. "The group mean replaced the critic" is usually an architecture claim, and here it is a column of numbers: the per-prompt reference for "better than usual" that a critic would supply is assembled from the group itself, $K - 1$ parts other samples and one part self-inclusion, and a group of one is all self-inclusion. A corollary with teeth: on the hard prompts every group scores all zeros and the advantages vanish, so a group-relative method learns nothing from prompts it always fails, and, by symmetry, from prompts it always solves. Note the zero-update conclusion's scope: it belongs to this reward-only toy, and an implementation whose loss carries other terms, a separate KL gradient or a supervised mixture, still moves at $K = 1$.

## Where the Reward Comes From

Every collapse so far concerned the policy half of the loop. What remains is the reward $r(x, y)$, and at scale it is never given; it is estimated, one of two ways.

### Learned: Bradley-Terry

When quality cannot be computed, it is elicited: people compare pairs of responses to the same prompt, and a reward model $r_\phi(x, y)$ is fit by the Bradley-Terry logistic regression of :numref:`sec_regularized`, :eqref:`eq_bradley_terry`. Everything measured there transfers: the fit is accurate where the preference data lives and silent where it does not; the score is identified only up to a function of the prompt, which is why the group mean subtracted above removes nothing the comparisons ever measured; and the reward is terminal, exactly the shape the proposition assumed.

### Checked: a verifier

For a growing family of tasks the reward needs no model at all, because the response can be *checked*: a unit-test harness runs the code, a proof assistant validates the derivation, an exact-match grader scores the final answer. Training against such checked rewards is called reinforcement learning from verifiable rewards, RLVR, the regime in which recent reasoning models are trained :cite:`DeepSeekAI.2025`. A verifier is not a lesser reward model but a different trust profile: it cannot be flattered where it actually checks, and it is blind everywhere it does not.

### Reward hacking, unified

Both suppliers hand the optimizer an *estimate*, and one sentence from :numref:`sec_regularized` now covers every failure of this kind at once: an optimizer pointed at an estimated objective finds the estimate's errors, whether the estimate is a $\hat{Q}$ fit on thin data (:numref:`sec_offline`) or a reward $r_\phi$ fit on comparisons (:numref:`sec_regularized`). Reward hacking at scale is a corollary, not a new topic. To watch it, replace the exact-match verifier with a deliberately sloppy one that greps the response for the answer string. The hedge now passes every prompt, including the two the model cannot solve, where it is the *only* rewarded response, a reward gap of $1$ over honesty. The cure is the one :numref:`sec_regularized` built, and its closed form :eqref:`eq_kl_optimum` prices the exploit in advance: the tilted optimum weighs the hedge against an honest candidate as $e^{-4} \, e^{(1 - 0)/\beta}$, reference log-odds gap against reward gap, so the exploit stops paying at $\beta = 1/4$.

```{.python .input #rl-sequences-reward-hacking-unified}
%%tab pytorch, jax
def sloppy(x, y):                  # accepts anything containing the answer
    return float(answers[x] in resp[y])

for beta in (0.0, 0.1, 0.2, 0.3, 0.5):
    rng, theta = np.random.default_rng(2), theta_ref.copy()
    for _ in range(400):
        theta = group_step(theta, 32, sloppy, rng, beta=beta)
    pi = policy(theta)
    print(f'beta = {beta:.1f}: sloppy {success(pi, sloppy):.2f}, '
          f'gold {success(pi, verify):.2f}, hedge {pi[:, 8].mean():.2f}')
```

Read the columns as the grader's score, the exact verifier's score, and the hedge's average probability. At $\beta = 0$ the policy is fully hacked: perfect by the grader it optimizes while the exact verifier still scores it $0.50$, no better than honest training, the hedge owning the two hard prompts; the takeover needs only the first group that samples the hedge, since there it is the only response ever rewarded. The penalty behaves as priced: the exploit still pays at $\beta = 0.1$, is marginal at $0.2$, and stops paying just past it, bracketing the predicted $1/4$. In fact every row lands close to the tilted optimum of :eqref:`eq_kl_optimum`: the update's pull fades as the KL-shifted rewards within a group equalize, which is that optimum's defining property, though "close" is the right word, because standardizing by the group's own standard deviation puts a random denominator under the update, and the point where it stalls need not coincide exactly with the regularized optimum. The last row is the price tag: $\beta$ is an exchange rate, not a safety valve; the penalty that blocks the exploit also caps honest sharpening, gold falling to $0.24$ at $\beta = 0.5$, and choosing $\beta$ *is* choosing a point on the frontier :numref:`sec_regularized` traced.

## The Contract

### GRPO assembled

The pieces assemble, in prose alone, into the method :numref:`sec_baselines` promised to finish. Group Relative Policy Optimization, GRPO :cite:`Shao.Wang.Zhu.ea.2024`, trains a language model by sampling a group of $K$ responses per prompt and weighting every token of response $j$ by the group-standardized advantage $A_j = (r_j - \mu)/(\sigma + 10^{-8})$, :eqref:`eq_pg_normalized` with the group as the batch and the group mean as the per-prompt baseline, so *no value network exists*; reusing each group for a few epochs, during which each token's ratio $\rho_t$ drifts from one and the clipped objective :eqref:`eq_ppo_clip` bounds each token's payoff; adding $\beta$ times a separate nonnegative estimator of the KL divergence to the frozen reference, the penalty of :eqref:`eq_kl_objective` as its own loss term; and dividing each response's loss by its own length, one of the divisor choices whose ledger :numref:`sec_baselines` printed. Note which KL is which, because GRPO runs both of :numref:`sec_regularized`'s kinds at once: the clip plays the trust-region role against the *previous iterate*, shaping each step, while the $\beta$ term is the penalty against the *frozen reference*, shaping the optimum. This section's toy is that estimator's skeleton, not the named algorithm, and the differences deserve a list: the toy takes one step per group, so no ratio ever drifts and nothing is clipped, where GRPO reuses each group under token-level clipped ratios; it folds the KL penalty into the reward, where the cited formulation adds the separate estimator above; its responses are single tokens, so the response-level and token-level views coincide by construction; it never divides by response length; and both share the same-group mean, the self-inclusion bias measured earlier, whose exactly unbiased alternative is the leave-one-out baseline of :numref:`sec_baselines`, RLOO. Nothing in the assembly is new to you, including its sharp edges: dividing by $\sigma$ is a step-size rescaling rather than a baseline, priced in :numref:`sec_baselines`. Finally, :eqref:`eq_kl_optimum` reads in the other direction too: solving it for the reward gives $r(x, y) = \beta \log \big( \pi^\star(y \mid x) / \pi_{\textrm{ref}}(y \mid x) \big)$ up to a per-prompt constant, so preferences can fit the policy *directly*, skipping the reward model and the reinforcement learning; that is direct preference optimization, DPO :cite:`Rafailov.Sharma.Mitchell.ea.2023`, whose derivation the Language Models part owns.

### Notation the Language Models part inherits

The Language Models part inherits these symbols verbatim; every object below was defined and exercised in these two chapters.

:The notation contract. Symbols the post-training chapters use without redefinition.
:label:`tab_rl_notation_contract`

| symbol | meaning | built in |
|:--|:--|:--|
| $x$, $y$ | prompt, response | this section |
| $\tau$ | trajectory, and nothing else | :numref:`sec_mdp` |
| $\mu_0$ | start-state, i.e. prompt, distribution | :numref:`sec_mdp` |
| $\pi_\theta$, $\pi_{\textrm{ref}}$ | policy; frozen reference | :numref:`sec_policygradient`, :numref:`sec_regularized` |
| $\hat{G}_t$ | reward-to-go | :numref:`sec_baselines` |
| $A$ | advantage | :numref:`sec_valueiter`, :numref:`sec_baselines` |
| $K$ | group size | :numref:`sec_baselines` |
| $\rho_t$ | importance ratio $\pi_\theta / \pi_{\theta_{\textrm{old}}}$ | :numref:`sec_ppo` |
| $\epsilon$ | clip half-width, exploration rate; never a numerical constant | :numref:`sec_ppo`, :numref:`sec_qlearning` |
| $\beta$ | KL coefficient, and nothing else | :numref:`sec_regularized` |
| $\delta_t$ | TD error | :numref:`sec_qlearning` |
| $w$, $w^-$ | critic parameters; target copy | :numref:`sec_actorcritic`, :numref:`sec_dqn` |
| $D_{\textrm{KL}}(P \Vert Q)$, $\mathbf{1}(\cdot)$ | KL divergence; indicator | :numref:`sec_regularized` |

## Where To Go Next

Two chapters are a first introduction, not the field. Three directions matter most.

**Model-based reinforcement learning and search.** This is the largest deliberate omission in these chapters, and the transferable idea fits in one paragraph. :numref:`sec_valueiter` is the book's only model-based corner; the smallest step beyond it is Dyna-Q, a rewarding exercise on :numref:`sec_qlearning`'s tabular loop: keep a table of observed transitions and, between real steps, replay imagined ones through the same Q-update. The consequential step is Monte Carlo tree search read correctly: not a game trick but a *policy-improvement operator*, a search that returns a better move distribution than the policy it started from, whose output is distilled back into the network as a supervised target. That loop is AlphaZero :cite:`Silver.Schrittwieser.Simonyan.ea.2017,Silver.Hubert.Schrittwieser.ea.2018`; MuZero learns the model it searches in :cite:`Schrittwieser.Antonoglou.Hubert.ea.2020`; DreamerV3 and TD-MPC2 train control inside learned world models :cite:`Hafner.Pasukonis.Ba.ea.2025,Hansen.Su.Wang.2024`. "Search at decision time, distill into the policy" is also the pattern of test-time reasoning in language models, which makes this the omission most worth repairing on your own.

**Continuous control since 2024.** The objectives of :numref:`sec_deeprl` through :numref:`sec_regularized` stand; recent gains have come from normalization, scale and stability rather than new losses, and careful normalization even lets streaming, replay-free temporal-difference learning work after decades of not working :cite:`Elsayed.Vasan.Mahmood.2024,Gallici.Fellows.Ellis.ea.2025`. Reading a modern paper here, you should recognize nearly every symbol.

**What we deliberately left out.** Multi-agent reinforcement learning, where the opponent learns too, reached superhuman poker with search plus self-play :cite:`Brown.Sandholm.2017`. Distributional reinforcement learning models the return's whole distribution rather than its mean :cite:`Bellemare.Dabney.Munos.2017`. Meta-learning, hierarchy and partial observability each relax one of :numref:`sec_mdp`'s assumptions; Murphy's overview treats all three at textbook depth :cite:`Murphy.2025`. And for this section's road continued into post-training practice, the reinforcement-learning-from-human-feedback book :cite:`Lambert.2026` is the natural next read: you now own every estimator in it.

## Capstone Projects

These four projects play the role of a course programming assignment; each runs on a laptop CPU in minutes, and each carries a numeric sanity bar in place of an autograder.

**A. Build PPO from the pieces, and prove it is right.** Three self-checks: the ratio is exactly $1$ on the first epoch; GAE at $\lambda = 1$ equals reward-to-go minus $\hat{V}$ to floating-point tolerance; the clipped objective's gradient at $\theta = \theta_{\textrm{old}}$ equals the plain policy gradient. Bar: median return above $450$ on CartPole within $60$ updates, 3 seeds.

**B. Which implementation details actually matter.** Ablate five of the 37 catalogued PPO details, one at a time, three seeds each. Bar: at least one factor changes the median by more than the seed spread, and you can say which.

**C. Break a trained policy.** Take three deliberately broken agents, one with a saturated policy, one with a critic that never learned, one with a replay buffer too small, and name each fault from the chapter's diagnostics alone, before reading the code.

**D. GRPO from REINFORCE, in two lines.** Start from :numref:`sec_baselines`'s `train`, add the group-standardized weight and the clip, and reproduce this section's $K$-sweep. Bar: $K = 1$ shows no learning; $K \geq 4$ reaches the verifier's ceiling.

## Summary

Text generation is a Markov decision process in the degenerate corner :numref:`sec_mdp` planted: the prompt is a start state drawn from $\mu_0$, tokens are actions, prefixes are states, the response is the trajectory, EOS terminates, and the transitions are concatenation, deterministic and known. The factorization proposition :eqref:`eq_seq_factorization` makes the response-level and token-level views one algorithm, and a terminal reward puts one response-level weight $r(x, y) - b(x)$ on every token's score, not the prefix-dependent token advantage, which explains why the critic-free family works and names the problem left open, per-token credit; the same-group mean that implements $b$ in practice is biased by self-inclusion, shrinking the expected update by $(K-1)/K$ and to exactly zero at $K = 1$, with leave-one-out the exact repair. The collapse deletes discounting, bootstrapping, the TD error, the Bellman equations, value iteration, Q-learning, replay and the deadly triad; it keeps the policy-optimization spine. Rewards are learned from comparisons or checked by a verifier, both estimates, so reward hacking is a corollary of :numref:`sec_regularized`'s one sentence, and the KL penalty prices exploits in advance, at reward gap over reference log-odds. GRPO is an assembly of parts these chapters built, and :numref:`tab_rl_notation_contract` is what the Language Models part inherits.

**What the experiments show, and what they do not.** Every cell is seeded numpy shared verbatim across the framework tabs, so the printed digits are identical in both and reproduce exactly on rerun. The $K$-sweep is one seed per arm at one matched budget on a four-prompt toy: it demonstrates the $K = 1$ prediction, which is exact and seed-independent, and that any $K \geq 2$ reaches the ceiling here, not a ranking among larger groups, which this problem is too easy to resolve. The loophole table is one seed per $\beta$; reseeding moves how quickly the hedge is found at $\beta = 0$ but not the endpoints, which land close to the closed form of :eqref:`eq_kl_optimum` as the update's pull fades near the tilted optimum. The toy's reward gap and reference log-odds were chosen so the threshold lands at the readable $1/4$; real exploits differ in numbers, not mechanism. The compute belongs to readers.

## Exercises

1. [conceptual] *The factorization.* Prove
   $\nabla_\theta \log \pi_\theta(y \mid x) = \sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$
   and say why this makes the token-level and response-level views the same
   algorithm.
1. [conceptual] *What collapses.* For each of discounting, bootstrapping, the
   TD error and replay, say in one sentence what property of the token MDP
   removes it, and name one setting (multi-turn dialogue, tool use) in which
   it comes back.
1. [short-code] *Why $K = 1$ learns nothing.* Predict it from the
   $(K-1)/K$ shrinkage of :numref:`sec_baselines`, run it, and say why
   leave-one-out has no defined value at $K = 1$.
1. [short-code] *Price the exploit.* Find the $\beta$ at which the KL penalty
   makes the verifier loophole unprofitable, and relate it to the reward gap
   between the exploit and the honest solution.
1. [conceptual] *Read a paper.* Take the GRPO objective as published and name,
   for each symbol, the section of these two chapters that built it, and the
   one component that is not in these chapters at all.

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §15.6]{.kicker}

Sequences are trajectories<br>
**text generation is a degenerate MDP · one factorization identity · what collapses, what survives · the contract the Language Models part inherits**
:::
:::

::: {.slide title="The Dictionary"}
| language modeling | reinforcement learning |
|:--|:--|
| prompt $x$ | start state $s_0 \sim \mu_0$ |
| token $y_t$ | action $a_t$ |
| prefix $(x, y_{<t})$ | state $s_t$ |
| response $y$ | trajectory $\tau$ |
| EOS | terminal state |
| next-token softmax | the policy $\pi_\theta$ |
| `generate` | `rollout` |

. . .

Transitions are *concatenation*: deterministic, known, probability one.
All randomness is the policy's; the reward is terminal.
:::

::: {.slide title="One Identity"}
$$\nabla_\theta \log \pi_\theta(y \mid x)
= \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$$

. . .

- chain rule, no transition factors to drop: three-line proof
- terminal reward $\Rightarrow$ one response-level weight
  $r(x, y) - b(x)$ on every token's score (not the
  prefix-dependent $A(s_t, a_t)$)
- one action per episode: the *contextual bandit* of
  :numref:`sec_qlearning`, until multi-turn closes the loop
:::

::: {.slide title="What Collapses, What Survives"}
![](../img/mdl-rl-token-mdp.svg){width=98%}

. . .

Everything that leaned on the kernel or on intermediate reward
collapsed; the policy-optimization spine survived whole.
:::

::: {.slide title="The Group Is the Baseline"}
Sample $K$ responses per prompt, standardize within the group,
one step on the log-probs. Same-group centering is *biased*:
its expectation is $(K-1)/K$ of the gradient (leave-one-out is
exact). **Prediction:** at $K = 1$ the shrinkage reaches zero
and the update vanishes identically.

@!rl-sequences-the-group-is-the-baseline-2

. . .

$0.062$ is the reference, untouched: self-inclusion bias at its
maximum, not "no relative information". "The group mean replaced
the critic", as a measurement.
:::

::: {.slide title="The Loophole"}
A sloppy grader greps for the answer; a hedge that lists every
number passes everything. :eqref:`eq_kl_optimum` prices it:
exploit pays iff $\beta < $ reward gap / reference log-odds $= 1/4$.

@!rl-sequences-reward-hacking-unified

. . .

Hacked at $\beta = 0$; stops paying past $0.2$; every row lands
close to the tilted optimum. $\beta$ is an exchange rate, not a
safety valve.
:::

::: {.slide title="GRPO, Assembled from Owned Parts"}
- group-standardized advantage: :numref:`sec_baselines`'s
  :eqref:`eq_pg_normalized`, group mean as baseline (biased by
  self-inclusion; LOO is the exact variant), **no value network**
- a few epochs per group: token-level clipped ratios
  :eqref:`eq_ppo_clip`
- $\beta\, D_{\textrm{KL}}$ to the frozen reference as a separate
  estimator: :eqref:`eq_kl_objective`
- both KLs at once: clip against the *previous iterate*, penalty
  against the *frozen reference*
- the toy simplifies: one step per group (nothing clipped), KL
  folded into the reward, one-token responses

. . .

Read :eqref:`eq_kl_optimum` backwards and preferences fit the
policy directly: DPO :cite:`Rafailov.Sharma.Mitchell.ea.2023`.
:::

::: {.slide title="The Contract"}
Reward is **learned** (Bradley-Terry, :numref:`sec_regularized`)
or **checked** (a verifier: RLVR :cite:`DeepSeekAI.2025`).
Either way it is an estimate, and an optimizer finds an
estimate's errors: $\hat{Q}$ in :numref:`sec_offline`,
$r_\phi$ in :numref:`sec_regularized`, the grader here.

. . .

:numref:`tab_rl_notation_contract`: $x$, $y$, $\tau$, $\mu_0$,
$\pi_{\textrm{ref}}$, $\hat{G}_t$, $A$, $K$, $\rho_t$,
$\epsilon$, $\beta$, $\delta_t$, $w$: inherited verbatim by the
Language Models part.
:::

::: {.slide title="Where To Go Next"}
- **search**: MCTS is a policy-improvement operator, distilled
  back into the network; AlphaZero $\to$ MuZero $\to$ world
  models; the same pattern as test-time reasoning
- **continuous control**: same objectives, gains from
  normalization and scale
- omitted with pointers: multi-agent, distributional, meta,
  hierarchical, POMDPs
- next read: the RLHF book :cite:`Lambert.2026`; you own every
  estimator in it
:::

::: {.slide title="Recap"}
- prompt = start state, token = action, prefix = state,
  response = trajectory; transitions are concatenation
- one gradient, two views; terminal reward puts one
  response-level weight $r - b(x)$ on every token
- $K = 1$'s update is identically zero (self-inclusion bias at
  its maximum), measured; the hedge gets hacked, measured;
  $\beta = $ gap / log-odds prices the exploit
- what survives is the policy-optimization spine, and the
  Language Models part inherits it verbatim
:::
