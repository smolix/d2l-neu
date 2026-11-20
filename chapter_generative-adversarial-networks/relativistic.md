# Relativistic Objectives
:label:`sec_gan_relativistic`

A classifier critic judges each sample against a fixed level: is $D(x')$ above zero? People rarely judge subtle differences that way. A taster who cannot say whether one glass holds the expensive wine can often say which of two glasses does, and a chess rating is built from who beat whom, not from an absolute score per player. The reward model of :numref:`sec_regularized` learned the same way, from which of two trajectories a judge preferred. Its tool was the Bradley--Terry model, which turns the difference of two scores into the probability that one contestant wins. :citet:`Jolicoeur-Martineau.2019` gave the GAN critic this comparative task: show it one real and one generated sample, and reward it for ranking the real one higher.

The recipe that the rest of this chapter trains, from :numref:`sec_gan_convergence` on, uses such a critic, so we had better know what it measures. Does the best ranking critic still estimate the log density ratio of :numref:`sec_basic_gan`? How much more does a pair reveal about which sample is real than a single sample does? And does ranking protect the generator against dropping modes? The first two have exact answers. The third has an answer much narrower than the one usually given, and we will say how narrow.

```{.python .input #relativistic-relativistic-objectives}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #relativistic-relativistic-objectives}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #relativistic-relativistic-objectives-1}
%%tab pytorch, jax
import itertools
```

## Scoring Pairs

### The Bradley--Terry Critic

We keep the notation of :numref:`sec_basic_gan`: data density $p$, generator density $q$, ratio $\rho = p/q$, log ratio $\lambda = \log \rho$, and a critic $D$ whose value is a realness score. Draw a real sample $x \sim p$ and a generated sample $x' \sim q$ independently, and score the pair by the difference of the two critic values:

$$
\Phi(D) \;=\; \mathbb{E}_{x \sim p,\; x' \sim q}\big[\log \sigma\big(D(x) - D(x')\big)\big].
$$
:eqlabel:`eq_gan_rp`

The critic maximizes $\Phi$ and the generator minimizes it. Since $\sigma(D(x) - D(x'))$ is the Bradley--Terry probability :cite:`Bradley.Terry.1952` that $x$ beats $x'$, $\Phi$ is the log-likelihood of ranking real samples above generated ones, and the critic learns a ranking rather than a classification rule. Jolicoeur-Martineau called such a critic *relativistic*, because it scores a sample relative to one from the other distribution. The sequel :cite:`Jolicoeur-Martineau.2020` calls this loss RpGAN, for relativistic *paired* (the 2019 paper's RSGAN), and the divergence derived below inherits the subscript.

The code estimates $-\Phi$ on aligned pairs, the $i$-th real sample of a minibatch against the $i$-th generated one, with $-\log \sigma(u) = \operatorname{softplus}(-u)$. The generator's loss swaps the two members. This is the non-saturating form, and the end of the section discusses why we prefer it to minimizing $\Phi$ itself.

```{.python .input #relativistic-the-bradley-terry-critic}
%%tab pytorch
#@save
def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing critic loss -E[log sigma(D(x) - D(x'))]."""
    return F.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return F.softplus(critic(real) - critic(fake)).mean()
```

```{.python .input #relativistic-the-bradley-terry-critic}
%%tab jax
#@save
def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing critic loss -E[log sigma(D(x) - D(x'))]."""
    return jax.nn.softplus(critic(fake) - critic(real)).mean()

#@save
def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator."""
    return jax.nn.softplus(critic(real) - critic(fake)).mean()
```

### Shift Invariance

The score of a classifier has a meaningful zero. A ranking has none (:numref:`fig_gan_pairing`).

![Scoring one sample against a threshold (left) versus scoring a pair by its difference (right). Adding a constant $b$ to every score can change the classical verdict, which asks whether $D(x)$ clears a fixed level, but leaves the difference $D(x) - D(x')$ unchanged.](../img/mdl-gan-pairing.svg)
:label:`fig_gan_pairing`

Adding a constant to every score leaves every difference, and therefore $\Phi$, unchanged, so the pairing game fixes the critic only up to an additive constant; the log-loss game of :numref:`sec_basic_gan` pins that constant down. The fraction of generated samples with $D(x') > 0$, a natural statistic to watch in the classical game, therefore measures nothing for a pairing critic. The fraction of real--generated pairs that it orders correctly takes its place.

### The Optimal Critic
:label:`subsec_gan_rp_optimal_critic`

In the log-loss game a free critic scored one sample at a time, so finding the best critic split into one scalar problem per point. Here each value $D(t)$ meets every other value through the differences. Concavity does the work that decoupling did.

**Proposition.** *$\Phi$ is concave in $D$, and if $p$ and $q$ have the same support it is maximized at $D^\star = \lambda$, uniquely up to an additive constant.*

**Proof.** The map $D \mapsto D(x) - D(x')$ is linear and $\log \sigma$ is concave, so the integrand of :eqref:`eq_gan_rp` is concave in $D$ for every pair, and the expectation preserves concavity; a stationary point is therefore a global maximum. To find one, perturb $D$ at a single point $t$. For finitely many outcomes the critic is a vector with one entry per outcome, and the functional derivative below is simply the partial derivative with respect to the entry $D(t)$. The point $t$ enters $\Phi$ once as the real member of a pair and once, with opposite sign, as the generated member, and $\tfrac{d}{du} \log \sigma(u) = \sigma(-u)$ gives

$$
\frac{\delta \Phi}{\delta D(t)}
\;=\; p(t)\, \mathbb{E}_{x' \sim q}\big[\sigma(D(x') - D(t))\big]
\;-\; q(t)\, \mathbb{E}_{x \sim p}\big[\sigma(D(t) - D(x))\big].
$$
:eqlabel:`eq_gan_rp_stationarity`

At $D = \lambda$, we have $\sigma(\lambda(x') - \lambda(t)) = \rho(x')/(\rho(x') + \rho(t))$. The identity $q\rho = p$ converts the first term of :eqref:`eq_gan_rp_stationarity` to $p(t) \int p(x')/(\rho(x') + \rho(t))\, dx'$. The second term, with $\sigma(\lambda(t) - \lambda(x)) = \rho(t)/(\rho(t) + \rho(x))$, equals $q(t)\, \rho(t) \int p(x)/(\rho(t) + \rho(x))\, dx$, and this is the first expression again because $q(t)\rho(t)=p(t)$. The derivative therefore vanishes everywhere, so $\lambda$ is a maximizer. Strict concavity of $\log \sigma$ in the score differences makes this maximizer unique up to an additive shift. $\blacksquare$

Ranking and classification therefore estimate the same log density ratio $\lambda$. Replacing one by the other keeps the optimal critic and, as the theorem below confirms, the fixed point $q = p$. What changes is the value of the game.

## The Value of the Pairing Game

### Two Atoms by Hand

Before computing the value in general, take the smallest case. Let the space hold two atoms, $0$ and $1$, with the data distribution $p = (0.8, 0.2)$ and the generator distribution $q = (0.2, 0.8)$. A single sample at atom $0$ is real with probability $p(0)/(p(0) + q(0)) = 0.8$, and a sample at atom $1$ with probability $0.2$. Now draw one real and one generated sample and look at them as an ordered pair $(a_1, a_2)$. The table lists the probability of each pair when the real sample comes first, $P(a_1, a_2) = p(a_1)\, q(a_2)$, and when it comes second, $Q(a_1, a_2) = q(a_1)\, p(a_2)$:

| pair $(a_1, a_2)$ | $(0, 0)$ | $(1, 1)$ | $(0, 1)$ | $(1, 0)$ |
|:--|:--|:--|:--|:--|
| $P$: real sample first | 0.16 | 0.16 | 0.64 | 0.04 |
| $Q$: real sample second | 0.16 | 0.16 | 0.04 | 0.64 |

A pair of equal atoms is equally likely in both orders and says nothing about which member is real. The mixed pairs carry the information. When the pair reads $(0, 1)$, the real sample is the first with probability $0.64/(0.64 + 0.04) = 16/17 \approx 0.94$, more than the $0.8$ that a single sample at atom $0$ gives.

The optimal critic turns this into a value. Only the score difference $d = D(0) - D(1)$ matters, and the critic maximizes

$$
\Phi = 0.32 \log \sigma(0) + 0.64 \log \sigma(d) + 0.04 \log \sigma(-d).
$$

Setting the derivative $0.64\, \sigma(-d) - 0.04\, \sigma(d)$ to zero gives $e^d = 16$, so $d = \log 16 = \lambda(0) - \lambda(1)$, as the proposition predicts. Substituting back and writing $H_b(t) = -t \log t - (1 - t) \log(1 - t)$ for the entropy of a coin with bias $t$,

$$
\sup_D \Phi + \log 2 = 0.68 \big(\log 2 - H_b(\tfrac{1}{17})\big) \approx 0.319 \text{ nats}.
$$

Read this as information. The mixed pairs, which occur with probability $0.68$, turn a fair coin about the real sample's position into a coin with bias $1/17$; the equal pairs leave it fair. For a single sample the same reading gives the Jensen--Shannon divergence, which :numref:`sec_basic_gan` identified with the information one sample carries about its source: $\mathrm{JS}(p, q) = \log 2 - H_b(0.8) \approx 0.193$ nats. The pair carries 1.66 times as much.

Is 0.319 a number we could have found another way? The table treats each ordered pair as one observation, drawn from the row $P$ or from the row $Q$. That is the setting of the classical game, one observation and two possible sources, so let us also compute the Jensen--Shannon divergence between the two rows. The cell's `js` computes the divergence of two probability vectors from the entropy form $\mathrm{JS}(a, b) = H\big(\tfrac12 (a + b)\big) - \tfrac12 \big(H(a) + H(b)\big)$ of :numref:`sec_basic_gan`, and `phi` evaluates $\Phi$ at the optimal critic.

```{.python .input #relativistic-two-atoms-by-hand}
%%tab pytorch, jax
def entropy(v):
    v = v[v > 0]                       # the convention 0 log 0 = 0
    return -(v * np.log(v)).sum()

def js(a, b):
    return entropy((a + b) / 2) - (entropy(a) + entropy(b)) / 2

p, q = np.array([0.8, 0.2]), np.array([0.2, 0.8])
P, Q = np.outer(p, q), np.outer(q, p)  # P[i, j]: real sample at i, generated at j
lam = np.log(p / q)                    # the optimal critic: lam[0] - lam[1] = log 16
U = lam[:, None] - lam[None, :]        # U[i, j] = D(i) - D(j)
phi = (P * -np.logaddexp(0, -U)).sum() # Phi = E_P[log sigma(D(a_1) - D(a_2))]
print(f'JS(p, q) = {js(p, q):.3f}   JS(P, Q) = {js(P.ravel(), Q.ravel()):.3f}   '
      f'sup Phi + log 2 = {phi + np.log(2):.3f}')
```

The divergence between the two rows is 0.319 again, the value of the pairing game to the last printed digit. It is no coincidence: the next subsection proves that the value of the pairing game equals the Jensen--Shannon divergence between the two orderings $P$ and $Q$ for every $p$ and $q$.

### The Lifted Game

The two-atom table treated the ordered pair as a single observation. In general, $\Phi$ is the expectation of a fixed function of the pair $(x, x')$ under $p \otimes q$, which suggests doing the same. Let $P = p \otimes q$ and $Q = q \otimes p$ denote the two orderings of an independent real--fake pair, viewed as distributions on $\mathcal{X} \times \mathcal{X}$: under $P$ the real sample sits first, under $Q$ it sits second. These two distributions can play the log-loss game of :numref:`sec_basic_gan` on the pair space, with a pair critic $\mathcal{D}: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ in place of $D$ and the value function

$$
V_{P,Q}(\mathcal{D}) = \mathbb{E}_{(a_1, a_2) \sim P}\big[\log \sigma(\mathcal{D}(a_1, a_2))\big] + \mathbb{E}_{(a_1, a_2) \sim Q}\big[\log \sigma(-\mathcal{D}(a_1, a_2))\big],
$$

which is :eqref:`eq_gan_V` with $(P, Q)$ substituted for $(p, q)$. The following lemma connects that game to $\Phi$.

**Lemma (lifting).** *Let $P = p \otimes q$ and $Q = q \otimes p$. Then: (i) the log ratio of the two orderings separates, $\log \frac{dP}{dQ}(a_1, a_2) = \lambda(a_1) - \lambda(a_2)$, an antisymmetric difference of single-sample scores; (ii) every difference critic $\mathcal{D}(a_1, a_2) = D(a_1) - D(a_2)$ satisfies $V_{P,Q}(\mathcal{D}) = 2\,\Phi(D)$; (iii) restricting the pair critic to differences does not lower the supremum, $\sup_{\mathcal{D}} V_{P,Q}(\mathcal{D}) = \sup_{D} V_{P,Q}\big(D(a_1) - D(a_2)\big)$.*

**Proof.** *The product ratio separates.* The two product densities contain the same factors in opposite order:

$$
\frac{dP}{dQ}(a_1, a_2)
= \frac{p(a_1)\, q(a_2)}{q(a_1)\, p(a_2)}
= \frac{\rho(a_1)}{\rho(a_2)}.
$$

Taking logarithms proves (i).

*The swap symmetry doubles the objective.* Under $P$, the first term of $V_{P,Q}(\mathcal{D})$ is $\mathbb{E}[\log \sigma(D(a_1)-D(a_2))] = \Phi(D)$. Under $Q$, we have $a_1 \sim q$ and $a_2 \sim p$. Antisymmetry gives $-\mathcal{D}(a_1,a_2)=D(a_2)-D(a_1)$, and relabeling $(x,x')=(a_2,a_1)$ turns the second term into $\Phi(D)$ as well. Hence $V_{P,Q}(\mathcal{D})=2\Phi(D)$, proving (ii).

*Differences suffice.* Pointwise maximization of the log-loss game on $\mathcal{X} \times \mathcal{X}$ gives the optimal pair critic $\log(dP/dQ)$ wherever both densities are positive. Where one of them vanishes the optimal pair critic is infinite, and since $P > 0$ requires $p(a_1)\, q(a_2) > 0$, the difference $\lambda(a_1) - \lambda(a_2)$ takes the same infinite value there. By (i), this critic has the difference form with $D=\lambda$. Restricting the supremum to difference critics therefore does not change its value. $\blacksquare$

Part (iii) depends on the logistic payoff: the log ratio of two product measures is a sum of per-coordinate terms. Under other payoffs the best pair critic is a nonlinear function of that log ratio, and the relativistic objective only bounds its lifted divergence from below (Exercise :exref:`ex_relativistic-separability-choice-loss-replace`). With the lemma in hand, the value of the pairing game follows from results already proved.

**Theorem.** *Define the pairing divergence $d_{\mathrm{Rp}}(p, q) := \sup_D \Phi(D) + \log 2$. Then*

$$
d_{\mathrm{Rp}}(p, q)
\;=\; \mathrm{JS}\big(p \otimes q,\; q \otimes p\big)
\;=\; H\Big(\tfrac{1}{2}\big(p \otimes q + q \otimes p\big)\Big) - H(p) - H(q).
$$
:eqlabel:`eq_gan_rp_value`

**Proof.** *Apply the value formula on the pair space.* Parts (ii) and (iii) of the lemma give $2 \sup_D \Phi(D) = \sup_{\mathcal{D}} V_{P,Q}(\mathcal{D})$. The value of the log-loss game, $\sup V = 2\, \mathrm{JS} - 2 \log 2$ (:eqref:`eq_gan_js_value`), applied to $(P, Q)$ evaluates the right-hand side as $2\, \mathrm{JS}(P, Q) - 2 \log 2$. Halving and adding $\log 2$ proves the first equality.

*Collapse the pair entropies.* The entropy form :eqref:`eq_gan_entropy_gap`, applied on $\mathcal{X} \times \mathcal{X}$, writes $\mathrm{JS}(P, Q)$ as the mixture entropy minus $\tfrac12 \big(H(P) + H(Q)\big)$. Since each product measure has independent components, $H(P) = H(Q) = H(p) + H(q)$, which gives the second equality. This entropy calculation assumes the displayed entropies are finite; the measure-level first equality does not. $\blacksquare$

The additive constant in $d_{\mathrm{Rp}}$ calibrates the objective against a constant critic. Such a critic assigns probability one half to either ordering and obtains $\Phi = \log \sigma(0) = -\log 2$; it is optimal when $p = q$. Thus $d_{\mathrm{Rp}}$ measures the reduction in ordering uncertainty relative to random guessing. :citet:`Jolicoeur-Martineau.2020` proved that this calibrated value is a divergence for a broad class of concave payoffs without computing it; the paper's normalization is twice ours, so its divergence is $2\, d_{\mathrm{Rp}}$.

Substituting the optimal critic $D^\star = \lambda$ into :eqref:`eq_gan_rp` directly, using $\sigma(\lambda(x) - \lambda(x')) = \rho(x)/(\rho(x) + \rho(x'))$, gives the same value as one expectation over independent pairs, the form in which Exercise :exref:`ex_relativistic-two-gaussians` evaluates it for two Gaussians:

$$
d_{\mathrm{Rp}}(p, q)
\;=\; \log 2 \;-\; \mathbb{E}_{x \sim p,\; x' \sim q}\Big[\log\Big(1 + \frac{q(x)\, p(x')}{p(x)\, q(x')}\Big)\Big].
$$
:eqlabel:`eq_gan_rp_explicit`

The mutual-information reading of :numref:`sec_basic_gan` lifts along with the value. Draw $x \sim p$ and $x' \sim q$, present the two samples in a uniformly random order, and let the bit $y$ record which position holds the real one, just as $y$ labeled the origin of a single sample in :numref:`sec_basic_gan`. The ordered pair $(a_1, a_2)$ is then a sample from the balanced mixture of $P$ and $Q$ with label $y$, and the identification of the divergence with a mutual information, applied on the pair space, gives

$$
d_{\mathrm{Rp}}(p, q) \;=\; I\big((a_1, a_2);\, y\big) \;=\; \log 2 - H\big(y \mid a_1, a_2\big).
$$
:eqlabel:`eq_gan_rp_mi`

The pairing divergence is the number of nats a randomly ordered pair carries about which member is real, just as $\mathrm{JS}(p, q)$ measures the information a single sample carries about its origin. The two-atom calculation above was this formula evaluated by hand.

### Bounds
:label:`subsec_gan_rp_bounds`

The theorem is the main result of this section. The second question of the opening, how much more a pair tells than one sample, now reads off its information form. Two facts about the randomly ordered pair do the work. Either member, taken alone, is a draw from $p$ or from $q$ according to the label $y$, so it carries $I(a_k; y) = \mathrm{JS}(p, q)$ nats about the label. And given $y$ the two members are independent draws, so $H(a_2 \mid a_1, y) = H(a_2 \mid y)$. The chain rule of mutual information and these two facts give

$$
\begin{aligned}
d_{\mathrm{Rp}}(p, q) &= I(a_1; y) + I(a_2; y \mid a_1)
= I(a_1; y) + H(a_2 \mid a_1) - H(a_2 \mid y)\\
&= I(a_1; y) + I(a_2; y) - I(a_1; a_2)
= 2\, \mathrm{JS}(p, q) - I(a_1; a_2),
\end{aligned}
$$

where the second line writes $H(a_2 \mid a_1) = H(a_2) - I(a_1; a_2)$ and $H(a_2 \mid y) = H(a_2) - I(a_2; y)$. Each member says $\mathrm{JS}$ about the label, and the pair loses what its two members share. Since a pair tells at least as much as its first member, and a binary label holds at most $\log 2$ nats,

$$
\mathrm{JS}(p, q) \;\leq\; d_{\mathrm{Rp}}(p, q) \;\leq\; \min\big(2\,\mathrm{JS}(p, q),\, \log 2\big).
$$

Neither bound can be improved. Near $q = p$ the two members are almost independent and $d_{\mathrm{Rp}}$ approaches twice $\mathrm{JS}$ (Exercise :exref:`ex_relativistic-gap-twice-js` makes this precise); on the two atoms it was 1.66 times $\mathrm{JS}$. With disjoint supports, real-first pairs lie in $\operatorname{supp} p \times \operatorname{supp} q$ and fake-first pairs in the disjoint set $\operatorname{supp} q \times \operatorname{supp} p$, so the pair reveals its ordering with certainty and $d_{\mathrm{Rp}} = \log 2$ however far apart the supports lie; a single sample reveals its source just as surely, and $\mathrm{JS}$ is $\log 2$ as well. For the point masses $p = \delta_0$ and $q_\theta = \delta_\theta$ of :numref:`sec_basic_gan`, the pairing value is $\log 2$ at every $\theta \neq 0$ and has no gradient in $\theta$. Pairing at most doubles what one sample tells, and it saturates exactly where the classical game does.

## Ranking and Mode Coverage

The value of a game is a statement about distributions. A generator, however, moves samples, and a generator that drops modes does so by piling several samples onto one mode while another goes empty. Three practical questions remain: how the pairing loss pushes each sample, whether it removes the minima in which samples pile up like that, and which generator loss to train it with.

### The Rank Weight

Let us differentiate $\Phi$ through $x' = G(z)$, as :numref:`sec_basic_gan` did for the classical game. With $\tfrac{d}{du} \log \sigma(u) = \sigma(-u)$,

$$
\nabla_\theta \Phi = -\,\mathbb{E}_{z}\big[w_{\textrm{Rp}}(G(z))\, \nabla_\theta D(G(z))\big],
\qquad
w_{\textrm{Rp}}(x') = \mathbb{E}_{x \sim p}\big[\sigma\big(D(x') - D(x)\big)\big].
$$
:eqlabel:`eq_gan_rp_weights`

Minimizing $\Phi$ therefore moves each generated sample up the critic's score surface, along $\nabla_\theta D(G(z))$, as in the classical game, and only the weight is new. The classical saturating loss weights the same push by $w_{\textrm{sat}}(x') = \sigma(D(x'))$ of :eqref:`eq_gan_weights`, which compares a sample with the critic's zero level, a level the pairing game leaves arbitrary. The relativistic weight, or *rank weight*, is the probability, softened by the sigmoid, that $x'$ outranks a random real sample. It judges a generated sample against the whole real population, and a shift of the critic does not move it.

This answers half of the complaint of :numref:`subsec_gan_mode_dropping`. There a lagging critic scored the generated samples as confidently real and the real samples of a missing mode higher still, and the classical weights, which read only a sample's own score, never saw the real ones. The rank weight sees them: the more real samples outrank a generated sample, the larger its non-saturating weight $1 - w_{\textrm{Rp}}$, whatever the zero level. It is still pushed along $\nabla_\theta D$ at its own position, however, so the real samples set how hard it moves, not where.

Neither weight by itself forbids collapse. Against a fixed critic, each generator loss is the $q$-average of a function that decreases as $D(x')$ grows, so it is smallest when all generated mass sits at the critic's highest-scoring point. Whether a collapsed generator stays there depends on how the critic responds once the generator moves. In the space of distributions nothing traps it: $d_{\mathrm{Rp}}$, a supremum over $D$ of functionals linear in $q$, is convex in $q$. As in :numref:`sec_gan_objectives`, however, convexity in $q$ says nothing about the positions of generated samples, and positions are what a generator moves.

### Paired Landscapes
:label:`subsec_gan_rp_landscapes`

:citet:`Sun.Fang.Schwing.2020` asked whether a collapsed generator is trapped, and answered exactly in the smallest setting that allows an exact answer: $n$ distinct data points $x_1, \ldots, x_n$, $n$ generated points $x'_1, \ldots, x'_n$ that the generator may place anywhere, and each loss evaluated at the critic's best response over all functions $D$:

$$
\begin{aligned}
\mathcal{E}_{\mathrm{GAN}} &= \sup_D \tfrac{1}{2n} \textstyle\sum_{i} \big[\log \sigma(D(x_i)) + \log \sigma(-D(x'_i))\big],\\
\mathcal{E}_{\mathrm{diag}} &= \sup_D \tfrac{1}{n} \textstyle\sum_{i} \log \sigma\big(D(x_i) - D(x'_i)\big),\\
\mathcal{E}_{\mathrm{all}} &= \sup_D \tfrac{1}{n^2} \textstyle\sum_{i, j} \log \sigma\big(D(x_i) - D(x'_j)\big).
\end{aligned}
$$
:eqlabel:`eq_gan_rp_landscape`

The first is the classical loss. The second, the *paired loss*, pairs the $i$-th real point with the $i$-th generated point, as our code does in a minibatch. The third averages over all $n^2$ cross pairs, so its generator weight is the rank weight. Each reaches its smallest value, $-\log 2$, when the generated points cover the data points one each. Call a configuration a *collision* when every generated point sits on a data point and at least two of them coincide, so that some data point goes uncovered. A collision that is a strict local minimum of a loss is a *collision minimum* of that loss, a trap for a generator that has dropped a mode. Sun et al. proved that every collision is a collision minimum of $\mathcal{E}_{\mathrm{GAN}}$; there are $n^n - n!$ of them. For $\mathcal{E}_{\mathrm{diag}}$ they proved that from every configuration a path along which the loss never increases leads to a global minimum.

Two data points show the mechanism. Put both generated points on $x_1$. Half of the all-pairs comparisons are then ties, so $\mathcal{E}_{\mathrm{all}} = -\tfrac12 \log 2$, and moving either point off $x_1$ raises the loss to $-\tfrac14 \log 2$. In the paired loss, $x'_2$ sits on $x_1$ rather than on its partner $x_2$, so the critic scores it as fully fake and its term is already zero. It can travel to $x_2$ without raising the loss.

For larger $n$ we compute. With $\hat p$ and $\hat q$ putting mass $1/n$ on each data point and each generated point, the value of the log-loss game gives $\mathcal{E}_{\mathrm{GAN}} = \mathrm{JS}(\hat p, \hat q) - \log 2$, and the theorem of this section gives $\mathcal{E}_{\mathrm{all}} = \mathrm{JS}(\hat p \otimes \hat q, \hat q \otimes \hat p) - \log 2$. The paired loss needs a small argument of its own. Draw an arrow from each real point to the position of its generated partner. The critic wants every arrow to point downhill, from a high score at the real point to a low score at its partner, and a pair whose score difference grows without bound contributes $\log \sigma \to 0$. If a set of arrows contains no cycle, the critic can order the positions so that all of them point downhill and then scale the scores up until their terms vanish. Around a cycle the score differences $u_1, \ldots, u_k$ sum to zero, so they cannot all be positive, and since $\log \sigma$ is concave, $\sum_i \log \sigma(u_i)$ under $\sum_i u_i = 0$ is largest at $u_i = 0$: ties are best, at $-\log 2$ per arrow. Hence $\mathcal{E}_{\mathrm{diag}}$ is $-\log 2$ times the fraction of arrows on a cycle, counting a generated point on its own partner as a cycle of length one.

In the code, real point $i$ sits at position $i$ and generated point $i$ at position `gen[i]`; `losses` returns the three best-response losses, checked first on two points.

```{.python .input #relativistic-paired-landscapes-1}
%%tab pytorch, jax
def on_cycle(i, gen):
    """Does the arrow from real point i to its partner's position close a cycle?"""
    j = gen[i]
    for _ in range(len(gen)):
        if j == i:
            return True
        if j not in range(len(gen)):  # a position that holds no data point
            return False
        j = gen[j]                    # follow the arrow of the real point at j
    return False

def losses(gen):
    """Best-response classical, all-pairs and paired losses of a configuration."""
    n = len(gen)
    support = sorted(set(range(n)) | set(gen))
    p_hat = np.isin(support, range(n)) / n
    q_hat = np.array([gen.count(s) for s in support]) / n
    P, Q = np.outer(p_hat, q_hat).ravel(), np.outer(q_hat, p_hat).ravel()
    paired = -np.log(2) * np.mean([on_cycle(i, gen) for i in range(n)])
    return np.array([js(p_hat, q_hat) - np.log(2), js(P, Q) - np.log(2), paired])

print('gen         classical, all pairs, paired: nats, multiples of log 2')
for gen in ([0, 0], [0, 0.1]):   # both on x_1; then x'_2 steps off it
    value = losses(gen)
    print(f'{str(gen):12s}{np.round(value, 3)}  {np.round(value / np.log(2), 2)}')
```

Stepping $x'_2$ off $x_1$ raises the classical loss, from $-0.477$ to $-0.347$ nats, and the all-pairs loss, from half to a quarter of $-\log 2$: this way out of the collision is uphill for both. Only the paired loss stays flat, at $-\tfrac12 \log 2$, as the hand calculation said. Does the pattern persist for more points? To count minima, we let each generated point stay or step to a nearby empty position, so coincident points may separate, and keep the assignments that every such move makes worse but that are not global minima.

```{.python .input #relativistic-paired-landscapes-2}
%%tab pytorch, jax
def small_moves(gen):
    """Each generated point stays or steps to one of n fresh positions next to it."""
    n = len(gen)
    for steps in itertools.product(range(n + 1), repeat=n):
        if any(steps):
            yield [g + k / 10 if k else g for g, k in zip(gen, steps)]

def count_bad_minima(n):
    """Suboptimal strict local minima among the assignments, per loss."""
    best, count = losses(list(range(n))), np.zeros(3, dtype=int)
    for gen in itertools.product(range(n), repeat=n):
        value = losses(list(gen))
        rises = np.all([losses(m) > value + 1e-9 for m in small_moves(list(gen))],
                       axis=0)
        count += rises & (value > best + 1e-9)
    return count
```

At $n = 2$ the count should reproduce the hand calculation. At $n = 3$ it tests the theorem, and it answers a question the theorem leaves open: does ranking against every real point, the all-pairs loss, also escape the collisions?

```{.python .input #relativistic-paired-landscapes-3}
%%tab pytorch, jax
for n in (2, 3):
    c = count_bad_minima(n)
    print(f'n = {n}: classical {c[0]}, all pairs {c[1]}, paired {c[2]};  '
          f'n^n - n! = {n**n - math.factorial(n)}')
```

The classical and all-pairs losses each have all $n^n - n!$ collision minima, 21 of them at $n = 3$, and the paired loss has none. The count separates three statements of different strength.

* *Proved* :cite:`Sun.Fang.Schwing.2020`: with a fixed one-to-one coupling and a critic at its best response, the paired loss has no collision minimum. From every configuration, collisions included, a path along which the loss never increases leads to a global minimum. (A collision can still be a minimum that is not strict: at $n = 2$, moving $x'_1$ off $x_1$ raises the paired loss, while moving $x'_2$ leaves it flat.)
* *Counted*, for $n \leq 3$ only: the all-pairs loss, which ranks each generated point against every real one, keeps every collision minimum of the classical loss. The guarantee belongs to the coupling, not to the comparison.
* *Not shown*: that minibatch training, which draws a new pairing at every step, inherits the guarantee. Averaged over all $n!$ pairings, the paired objective is exactly the all-pairs objective, so a critic that responds to the average faces the all-pairs landscape, collision minima included. Exercise :exref:`ex_relativistic-random-repairing` counts what happens when the critic instead best-responds to each pairing separately.

For a practitioner the third line is the one that matters, and it is the weakest. It fits a finding that otherwise looks odd. Our code pairs the $i$-th real sample of a minibatch with the $i$-th generated one; averaging the loss over all $n^2$ cross pairs lowers the variance of the estimate, but it did not train better generators in the experiments of :citet:`Jolicoeur-Martineau.2020`. Since every minibatch brings fresh samples, both estimators have the same expectation, $\Phi$ itself, and neither keeps the fixed coupling that the theorem needs.

The case that pairing guards against dropped modes therefore rests on experiments. The one published ablation trained the R3GAN recipe on StackedMNIST :cite:`Metz.Poole.Pfau.ea.2017`, stacked digits with 1,000 countable modes, and with gradient penalties on the critic the pairing loss recovered all 1,000 modes while the classical loss recovered 693 :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. One ablation on one dataset is thin evidence for a general claim. We train pairing in the rest of the chapter because it costs no more than the classical loss and that ablation favours it, and :numref:`sec_gan_convergence` repeats the comparison on a grid of 25 Gaussians, with and without pairing.

### Generator Losses for the Pairing Critic
:label:`subsec_gan_rp_trained`

Which generator loss belongs with the pairing critic? The generator that minimizes $\Phi$ itself, the zero-sum form of the game, is the *saturating* generator, named as in :numref:`sec_basic_gan`, because its rank weight saturates as the classical weight does. If $D(x')$ lies far below every real score, $w_{\textrm{Rp}}(x') \approx 0$, and the saturating generator barely moves the samples that most need to move. The non-saturating generator maximizes $\mathbb{E}[\log \sigma(D(x') - D(x))]$ instead. It keeps the direction and weights each sample by $1 - w_{\textrm{Rp}}(x')$, near one for a badly ranked sample. On the two atoms at the optimal critic, the $q$-averaged weight is about 0.24 for the saturating generator and 0.76 for the non-saturating one, so even there the non-saturating generator pushes about three times as hard.

Both losses still have their minimum at $q = p$. For the saturating loss this follows from the bounds, since $d_{\mathrm{Rp}} \geq \mathrm{JS}$ vanishes only at $q = p$. For the non-saturating loss at $D^\star = \lambda$, write it as $\mathbb{E}[\log(1 + e^u)]$ with $u = \lambda(x) - \lambda(x')$, $x \sim p$, $x' \sim q$. The mean of $u$ is $D_{\textrm{KL}}(p \,\|\, q) + D_{\textrm{KL}}(q \,\|\, p)$, and since $u \mapsto \log(1 + e^u)$ is convex, Jensen's inequality bounds the loss below by $\log(1 + e^{\mathbb{E}[u]})$, which exceeds $\log 2$ unless $q = p$.

The weights predict where the choice matters: only for samples ranked below nearly every real one. Our run tests the opposite case, a start that overlaps the data, where the prediction is that the choice should not matter. We train the pairing critic against both generator losses on the planar three-mode Gaussian mixture of :numref:`subsec_gan_objectives_testbed`. The start, networks, optimizers and 6,000 steps are those of the five losses there, with three runs per arm (seeds 0, 1, 2), and the cell prints that experiment's columns: the median $\mathrm{JS}(p, q_t)$ over the last 2,000 steps and the first recorded step (one record every 200) with $\mathrm{JS}$ below 0.01 nats. The start lies 0.53 nats from the data, well short of the $\log 2 \approx 0.69$ of disjoint supports.

```{.python .input #relativistic-pairing-on-the-mixture-testbed}
%%tab pytorch, jax
scores = lambda s: s   # train_toy passes scores, so the critic is the identity
arms = {'non-saturating': (lambda r, f: d2l.rpgan_loss_D(scores, r, f),
                           lambda r, f: d2l.rpgan_loss_G(scores, r, f)),
        'saturating': (lambda r, f: d2l.rpgan_loss_D(scores, r, f),
                       lambda r, f: -d2l.rpgan_loss_D(scores, r, f))}

def first_step_below(trace, level=0.01, every=200):
    below = np.flatnonzero(trace < level)
    return f'{every * below[0]:5d}' if len(below) else ' none'

print('pairing critic; generator loss, runs 0, 1, 2')
for name, (critic_loss, gen_loss) in arms.items():
    traces = [d2l.train_toy(critic_loss, gen_loss, seed) for seed in range(3)]
    medians = ' '.join(f'{np.median(t[-2000 // 200:]):.4f}' for t in traces)
    firsts = ' '.join(first_step_below(t) for t in traces)
    print(f'{name:15s} median JS {medians}   first JS < 0.01 {firsts}')
```

Both arms fit, as predicted. The non-saturating runs end at about a thousandth of a nat or less and cross 0.01 nats between steps 2,200 and 2,600, as the logistic loss of :numref:`sec_gan_objectives` did (about a thousandth of a nat, step 2,400), so pairing loses nothing in fit. The saturating generator, far from stalling, is the faster one here: every saturating run crosses 0.01 nats at step 1,800, before any non-saturating run, and ends between one and two thousandths of a nat, a little above the non-saturating runs. Neither difference matters at this level. Near the data the saturating weight is not small, so nothing saturates.

The case for the non-saturating form lies at the other extreme, a start far from the data. There a trained pairing critic ranks every generated sample below every real one, the rank weight is close to zero for all of them, and the saturating generator should stall as the classical one did in :numref:`subsec_gan_objectives_separated_start`. We run that start with the pairing critic: `d2l.train_shift` of :numref:`subsec_gan_objectives_separated_start`, with the critic and generator losses of the cell above, three runs per generator loss, printing that experiment's columns.

```{.python .input #relativistic-pairing-on-the-separated-start}
%%tab pytorch, jax
steps = (0, 100, 300, 1000, 1500)
print(f'{"":20s}|gradient|   theta at steps {steps}   first |theta| < 0.5')
for name, (critic_loss, gen_loss) in arms.items():
    for seed in range(3):
        thetas, grads, _ = d2l.train_shift(critic_loss, gen_loss, seed)
        hit = np.flatnonzero(np.abs(thetas) < 0.5)
        print(f'{name:15s} {seed}   {np.median(np.abs(grads[:100])):8.3f}   '
              + ''.join(f'{thetas[t]:7.2f}' for t in steps) + '   '
              + (f'{hit[0]:5d}' if len(hit) else 'not reached'))
```

The prediction holds, and more sharply than in the classical game. The saturating pairing generator does not move at all: its median gradient over the first 100 steps prints as 0.000, that is, below half a thousandth, and after 1,500 steps $\theta$ still reads 5.99 or 6.00. The classical saturating generator of :numref:`subsec_gan_objectives_separated_start` at least crept, with gradients of about a hundredth, from 6 to about 5.76. A critic that ranks every generated sample last gives the saturating generator nothing to follow. The non-saturating pairing generator, with gradients of about 3, is within 0.6 of the data by step 300 and comes within 0.5 between steps 315 and 335 in every run, a little sooner than the classical non-saturating arm (steps 377 to 388). The two pairing arms end six units apart, while runs of the same arm differ by at most 0.01 in $\theta$ and 20 in the arrival step, so the verdict does not depend on the seed. On images the supports are typically disjoint or nearly so (:numref:`sec_basic_gan`), so there this extreme is the usual case, and the chapter trains the non-saturating form throughout.

Neither the value of the game nor the weights, however, say whether gradient play between critic and generator reaches its equilibrium at all, and with pairing or without, it need not. :numref:`sec_gan_convergence` takes up that question.

## Summary

The pairing objective is the Bradley--Terry log-likelihood of ranking a real sample above a generated one. Its optimal critic is still the log density ratio, now only up to a constant, so the sign of a pairing score means nothing. Treating the ordered pair as one observation turns the game into the log-loss game between $p \otimes q$ and $q \otimes p$, and its value $\mathrm{JS}(p \otimes q, q \otimes p) = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$ is the information a randomly ordered pair carries about which member is real. That value lies between $\mathrm{JS}(p, q)$ and $\log 2$: about twice $\mathrm{JS}$ near $q = p$, and $\log 2$ on disjoint supports. The generator's weight becomes a rank among the real scores. With a best-response critic, a fixed one-to-one coupling removes the collision minima that the classical and all-pairs losses share; but pairs redrawn at every step average back to the all-pairs objective, whose landscape keeps them, so the case for pairing as a guard against dropped modes rests on experiments. On a planar mixture pairing fits as the logistic loss does, with either generator loss. Far from the data the saturating rank weight vanishes: started six units away, the saturating generator does not move in 1,500 steps, while the non-saturating one reaches the data by step 335, so we train the non-saturating form.

## Exercises

1. [code] **Saturating and non-saturating weights.** Take five atoms with
   data probabilities $p = (0.35, 0.25, 0.20, 0.12, 0.08)$ and generator
   probabilities $q = (0.02, 0.05, 0.13, 0.30, 0.50)$; the later exercises
   reuse these atoms.
    1. With the function `js` of the two-atom cell, compute
       $\mathrm{JS}(p, q)$ and $d_{\mathrm{Rp}}$, the latter both as
       $\mathrm{JS}(P, Q)$ on the 25 pair atoms and as
       $\sup_D \Phi + \log 2$ at $D = \lambda$ (you should find about 0.234
       and 0.377). Is the ratio $d_{\mathrm{Rp}} / \mathrm{JS}$ closer to
       1 or to 2 than on the two atoms, and what about these $p$ and $q$
       explains it?
    1. Treat each atom as a generated sample and evaluate the saturating
       weight $w_{\mathrm{Rp}}$ of :eqref:`eq_gan_rp_weights` and the
       non-saturating weight $1 - w_{\mathrm{Rp}}$ with the critic
       $D = \lambda$. On which atoms does each variant place its largest
       weight, and how do these relate to the atoms where $q$ exceeds $p$?
       By what factor does the $q$-average of the non-saturating weight
       exceed that of $w_{\mathrm{Rp}}$?
    1. Let $D = \lambda$ and write $P = p \otimes q$ and $Q = q \otimes p$.
       Show that the $q$-average of $w_{\mathrm{Rp}}$ equals
       $\iint \frac{P Q}{P + Q}\, da_1\, da_2$, and that it is at most
       $\tfrac12$, with equality only at $q = p$. What does it tend to as
       the supports separate, and what does this imply for the saturating
       generator started far from the data?
    1. Part (c) predicts that the saturating weight fades as the generator
       moves away from the data, and the separated start found it gone at
       $\theta = 6$. Find where it fades: run `d2l.train_shift` with the
       pairing critic loss and each generator loss of the mixture cell from
       `theta0` in $\{1, 2, 3, 4\}$, three runs each, and record the median
       gradient over the first 100 steps and whether $\theta$ comes within
       0.5 of the data. Between which starts does the saturating generator
       stop moving? Does the non-saturating generator's gradient depend on
       the start?
   :exlabel:`ex_relativistic-zero-sum-weights`

1. **Gap to twice Jensen--Shannon.** ● Take the randomly ordered
   pair $(a_1, a_2)$ and its label $y$ from :eqref:`eq_gan_rp_mi`.
    1. Show that $a_1$ and $a_2$ are conditionally independent given $y$,
       and that each position alone carries $I(a_k; y) = \mathrm{JS}(p, q)$.
    1. Let $p$ and $q$ have disjoint supports. Compute $I(a_1; a_2)$
       directly from the joint distribution $\tfrac12 (P + Q)$ of the
       pair, and check the identity of :numref:`subsec_gan_rp_bounds`
       there. Why do the two members share exactly as much information as
       each carries about the label?
    1. Let $q = p\,(1 + \epsilon v)$ with $\mathbb{E}_p[v] = 0$, $v$
       bounded and not zero $p$-almost everywhere. Show that
       $\mathrm{JS}(p, q)$ is of order $\epsilon^2$ while $I(a_1; a_2)$ is
       of order $\epsilon^4$, so that
       $d_{\mathrm{Rp}} / \mathrm{JS} \to 2$ as $\epsilon \to 0$.
    1. Compute $I(a_1; a_2)$ for the five atoms of Exercise
       :exref:`ex_relativistic-zero-sum-weights` from the 25-atom
       distribution $\tfrac12 (P + Q)$, and check that
       $2\, \mathrm{JS}(p, q) - I(a_1; a_2)$ reproduces $d_{\mathrm{Rp}}$.
   :exlabel:`ex_relativistic-gap-twice-js`

1. **Two Gaussians.** Let $p = \mathcal{N}(0, I)$ and
   $q = \mathcal{N}(\mu, I)$ on $\mathbb{R}^d$.
    1. Show that $\lambda$ is affine in $x$ and that
       $u = \lambda(x') - \lambda(x)$ in :eqref:`eq_gan_rp_explicit` is
       Gaussian. Find its mean and variance.
    1. Write $\mathrm{JS}(p, q)$ in the same form, as $\log 2$ minus the
       expectation of $\log(1 + e^t)$ for a Gaussian $t$. Find the constant
       $c$ for which
       $\mathrm{JS}\big(p, \mathcal{N}(c\mu, I)\big) = d_{\mathrm{Rp}}(p, q)$
       holds for every $\mu$.
    1. Evaluate $d_{\mathrm{Rp}}$ and $\mathrm{JS}$ by one-dimensional
       quadrature for $\|\mu\| \in \{0.1, 1, 2, 4\}$. Relate the ratio
       $d_{\mathrm{Rp}} / \mathrm{JS}$ to the local limit and to the
       saturation of :numref:`subsec_gan_rp_bounds`.
   :exlabel:`ex_relativistic-two-gaussians`

1. **Separability of the pair critic.** ● Replace the logistic payoff
   in :eqref:`eq_gan_rp` by a concave payoff $\ell$, giving
   $\Phi_\ell(D) = \mathbb{E}_{x \sim p,\, x' \sim q}[\ell(D(x) - D(x'))]$.
   On the pair space an unrestricted critic $\mathcal{D}$ collects
   $\ell(\mathcal{D})$ on pairs from $P$ and $\ell(-\mathcal{D})$ on pairs
   from $Q$. By part (ii) of the lifting lemma, which holds for every
   $\ell$, difference critics earn $2\, \Phi_\ell(D)$; by part (iii) they
   lose nothing for the logistic payoff.
    1. For $\ell(t) = -(1 - t)^2$, show that the pointwise optimum of the
       pair-space game is $2\eta - 1 = \tanh\big(\tfrac{\lambda(a_1) - \lambda(a_2)}{2}\big)$
       with $\eta = P/(P + Q)$. Show that it is not additively separable,
       and conclude that a relativistic least-squares game over difference
       critics optimizes a strict lower bound on its lifted divergence in
       general.
    1. Measure this gap on the five atoms of Exercise
       :exref:`ex_relativistic-zero-sum-weights`: evaluate the unrestricted
       value at the pointwise optimum, and maximize $2\, \Phi_\ell(D)$ over
       $D \in \mathbb{R}^5$ by gradient ascent. How large is the gap
       relative to the unrestricted value?
    1. Return to the logistic payoff and present the pair real-first with
       probability $\alpha \neq \tfrac12$, so that the log loss weights
       pairs from $P$ by $\alpha$ and pairs from $Q$ by $1 - \alpha$.
       Derive the Bayes-optimal pair critic. Can a difference critic attain
       it? Which extension of the difference class can?
   :exlabel:`ex_relativistic-separability-choice-loss-replace`

1. **More negatives.** ● Comparing one generated sample with several real
   ones turns the pairing game into the InfoNCE loss of contrastive
   learning. Draw $K$ real samples $x_1, \ldots, x_K \sim p$ and one
   generated sample $x' \sim q$, place all $K + 1$ samples in a uniformly
   random order $(a_1, \ldots, a_{K+1})$, and let the critic predict which
   position holds the generated sample.
    1. Show that the Bayes posterior for position $j$ is the softmax
       $e^{-\lambda(a_j)} / \sum_k e^{-\lambda(a_k)}$, so that a
       single-sample score again suffices.
    1. Repeat the lifting argument with $K + 1$ hypotheses
       $P_j = p^{\otimes (j-1)} \otimes q \otimes p^{\otimes (K + 1 - j)}$.
       Using $H(P_j) = K\, H(p) + H(q)$, show that the chance-level log loss
       $\log(K + 1)$ minus the optimal expected log loss is

       $$d_K(p, q) = H\Big(\tfrac{1}{K+1} \textstyle\sum_j P_j\Big) - K\, H(p) - H(q),$$

       and that $0 \leq d_K \leq \log(K + 1)$.
    1. Compare the game with the InfoNCE loss :eqref:`eq_mdl-infonce_def`
       of :numref:`sec_mdl-infonce`. Which sample and which densities play
       the roles of the positive pair, the negatives, and the density ratio
       that the optimal InfoNCE critic recovers?
   :exlabel:`ex_relativistic-more-negatives`

1. [code] **Random re-pairing.** ●● The enumeration of
   :numref:`subsec_gan_rp_landscapes` fixes the coupling: the $i$-th real
   point is always paired with the $i$-th generated point. Minibatch
   training draws a new pairing at every step.
    1. Extend the count to $n = 4$ (256 assignments). Do the classical and
       all-pairs counts still equal $n^n - n!$, and does the paired loss
       still have none?
    1. Model a critic that best-responds to each pairing separately: for a
       permutation $\pi$, let
       $\mathcal{E}_\pi = \sup_D \tfrac{1}{n} \sum_i \log \sigma\big(D(x_i) - D(x'_{\pi(i)})\big)$,
       and let the generator minimize the average of $\mathcal{E}_\pi$ over
       all $n!$ permutations. Adapt `losses` and `count_bad_minima` to count
       the suboptimal strict local minima of this average for $n = 3$. For
       $n = 2$, trace what happens at the collision $x'_1 = x'_2 = x_1$ when
       one generated point moves away.
    1. A critic trained over many minibatches responds instead to the
       pairing-averaged objective, whose best response is
       $\mathcal{E}_{\mathrm{all}}$. Which model is closer to minibatch
       training with a fresh pairing per step, and what do your counts
       suggest about the landscape guarantee in training?
   :exlabel:`ex_relativistic-random-repairing`

[Discussions](https://d2l.discourse.group/t/36810)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.3]{.kicker}

Relativistic Objectives<br>
**the critic ranks pairs · the value is JS of the two orderings · the coverage guarantee needs a fixed pairing**
:::
:::

::: {.slide title="Shift Every Score by a Constant and a Threshold Verdict Flips; a Ranking Does Not"}
[The problem]{.kicker}

![](../img/mdl-gan-pairing.svg){width=70%}

- The classical critic calls $x$ real when $D(x) > 0$: one sample, one level.
- A taster who cannot grade one glass can often say which of two is better.

::: {.d2l-note}
What does a critic that ranks a real sample against a generated one estimate,
and does ranking protect the generator against dropping modes?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- show that the best ranking critic is still $\log(p/q)$, up to a constant;
- compute the value of the pairing game by hand on two atoms;
- bound it between one sample's information and $\log 2$;
- say which pairing removes collision minima, and which does not;
- choose the generator's loss.

**Example:** $p = (0.8, 0.2)$ against $q = (0.2, 0.8)$; a planar
three-mode mixture. **Skips:** training dynamics and gradient penalties (next deck).
:::

::: {.slide title="The Critic Ranks Real above Generated: Bradley–Terry on Pairs"}
$p$ data density, $q$ generator density, $D$ a realness score,
$\sigma$ the sigmoid.

$$\Phi(D) = \mathbb{E}_{x \sim p,\; x' \sim q}\big[\log \sigma\big(D(x) - D(x')\big)\big]$$

- $\sigma(D(x) - D(x'))$: the Bradley–Terry probability that $x$ beats
  $x'$, as in reward models fitted to preferences.
- The critic maximizes $\Phi$; the generator plays against it.

::: {.d2l-note}
The critic learns a ranking, not a classification rule.
:::
:::

::: {.slide title="The Best Ranker Is Still $\log p/q$, but Only up to a Constant"}
- $\Phi(D + b) = \Phi(D)$ for every constant $b$: a ranking has no zero level.
- $\Phi$ is concave in $D$ ($\log \sigma$ concave, differences linear).
- Stationarity holds at $\lambda = \log(p/q)$, so
  $D^\star = \lambda + c$ for any constant $c$.

::: {.d2l-note}
The sign of a pairing score means nothing: watch the fraction of pairs
ordered correctly instead.
:::
:::

::: {.slide title="Two Atoms: a Mixed Pair Is 94% Sure, One Sample 80%"}
$p = (0.8, 0.2)$, $q = (0.2, 0.8)$; $P$: real sample first, $Q$: real second.

| pair | $(0, 0)$ | $(1, 1)$ | $(0, 1)$ | $(1, 0)$ |
|:--|:--|:--|:--|:--|
| $P = p(a_1)\,q(a_2)$ | 0.16 | 0.16 | 0.64 | 0.04 |
| $Q = q(a_1)\,p(a_2)$ | 0.16 | 0.16 | 0.04 | 0.64 |

- Equal pairs say nothing; pair $(0, 1)$ is real-first with probability $16/17$.
- Best critic: $D(0) - D(1) = \log 16$. JS of one sample, JS of the two
  rows, and the game's value, in nats:

@!relativistic-two-atoms-by-hand
:::

::: {.slide title="I. The Log Ratio of the Two Orderings Separates"}
Treat the ordered pair $(a_1, a_2)$ as one observation, from
$P = p \otimes q$ (real first) or $Q = q \otimes p$ (real second).

$$\frac{dP}{dQ}(a_1, a_2) = \frac{p(a_1)\,q(a_2)}{q(a_1)\,p(a_2)}
\quad\Rightarrow\quad
\log \frac{dP}{dQ} = \lambda(a_1) - \lambda(a_2)$$

::: {.d2l-note}
The best pair critic of the log-loss game on pairs is a difference of
single-sample scores.
:::
:::

::: {.slide title="II. A Difference Critic Earns Twice $\Phi$ on the Pair Space"}
Log-loss game between $P$ and $Q$ with pair critic $\mathcal{D}$:

$$V(\mathcal{D}) = \mathbb{E}_{P}[\log \sigma(\mathcal{D})]
+ \mathbb{E}_{Q}[\log \sigma(-\mathcal{D})]$$

- Put $\mathcal{D}(a_1, a_2) = D(a_1) - D(a_2)$.
- Under $P$ the first term is $\Phi(D)$.
- Under $Q$, swap the two members: the second term is $\Phi(D)$ too.

::: {.d2l-note}
$V = 2\,\Phi(D)$, and by I nothing is lost by using differences.
:::
:::

::: {.slide title="III. The Value Is the Jensen–Shannon Divergence between the Two Orderings"}
The log-loss game has value $\sup V = 2\,\mathrm{JS}(P, Q) - 2\log 2$, where
$\mathrm{JS}(p, q) = \tfrac12 \mathrm{KL}(p \,\|\, m) + \tfrac12 \mathrm{KL}(q \,\|\, m)$,
$m = \tfrac12(p + q)$, is the information one sample carries about its source.

$$\boxed{d_{\mathrm{Rp}}(p, q) := \sup_D \Phi + \log 2
= \mathrm{JS}(p \otimes q,\, q \otimes p)}$$

::: {.d2l-note}
$d_{\mathrm{Rp}}$ is the information, in nats, that a randomly ordered pair
carries about which member is real. Two atoms: 0.319, against 0.193 for one
sample.
:::
:::

::: {.slide title="A Pair Tells Between One and Two Samples' Worth, and Saturates at $\log 2$"}
$y$: which position holds the real sample. Each member alone carries
$\mathrm{JS}(p, q)$ about $y$; given $y$ they are independent.

$$d_{\mathrm{Rp}} = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$$

$I(a_1; a_2)$: the mutual information between the two members.

| case | value |
|:--|:--|
| always | $\mathrm{JS} \leq d_{\mathrm{Rp}} \leq \min(2\,\mathrm{JS}, \log 2)$ |
| $q$ near $p$ | $d_{\mathrm{Rp}} \approx 2\,\mathrm{JS}$ |
| disjoint supports | $d_{\mathrm{Rp}} = \log 2$, at every distance, no gradient |
:::

::: {.slide title="Check: What Pushes the Ratio toward 1, and What toward 2?"}
[Check]{.kicker}

On the two atoms $d_{\mathrm{Rp}}/\mathrm{JS} \approx 1.66$. Move $q$ toward
$p$, or pull their supports apart: where does the ratio go?

::: {.fragment .fade-up}
Toward 2 as $q \to p$: the two members become nearly independent evidence.
Toward 1 as the supports separate: both divergences reach $\log 2$.
:::
:::

::: {.slide title="The Generator's Weight Becomes a Rank, yet a Fixed Critic Still Rewards Collapse"}
$$\nabla_\theta \Phi = -\,\mathbb{E}_z\big[w_{\textrm{Rp}}(G(z))\,\nabla_\theta D(G(z))\big],
\qquad w_{\textrm{Rp}}(x') = \mathbb{E}_{x \sim p}[\sigma(D(x') - D(x))]$$

- Same direction as the classical game; the classical weight is
  $\sigma(D(x'))$, a comparison with the zero level.
- $w_{\textrm{Rp}}$: the chance that $x'$ outranks a random real sample;
  a shift of $D$ does not move it.
- Against a fixed critic every generator loss is smallest with all mass at
  the critic's top score.

::: {.d2l-note}
Collapse is decided by how the critic responds when the generator moves.
:::
:::

::: {.slide title="One Step off a Collision: Classical and All-Pairs Losses Rise, Paired Stays Flat"}
$n$ data points $x_i$, $n$ movable points $x'_i$, best-response critic:

- classical: $\tfrac{1}{2n}\sum_i [\log\sigma(D(x_i)) + \log\sigma(-D(x'_i))]$
- paired: $\tfrac1n \sum_i \log\sigma(D(x_i) - D(x'_i))$, a fixed coupling
- all pairs: $\tfrac{1}{n^2}\sum_{i,j} \log\sigma(D(x_i) - D(x'_j))$

Both $x'_1, x'_2$ on $x_1$ (a *collision*), then $x'_2$ steps off:

@!relativistic-paired-landscapes-1

::: {.d2l-note}
Paired loss: arrows real point → partner; arrows on a cycle tie at
$-\log 2$, all others score 0.
:::
:::

::: {.slide title="Collision Minima at $n = 3$: Classical 21, All Pairs 21, Fixed Pairing 0"}
A collision minimum: a collision from which every small move goes uphill.

@!relativistic-paired-landscapes-3

- **Proved:** with a fixed pairing, every configuration has a
  non-increasing path to a global minimum.
- **Counted** ($n \le 3$): ranking against all real points keeps every
  collision minimum.
- **Not shown:** that training inherits it; pairs redrawn every step
  average to the all-pairs loss.

::: {.d2l-note}
The guarantee belongs to the coupling, not to the comparison.
:::
:::

::: {.slide title="Near the Data Both Generator Losses Fit, and the Saturating One Is Faster"}
Saturating: minimize $\Phi$, weight $w_{\textrm{Rp}}$. Non-saturating:
weight $1 - w_{\textrm{Rp}}$. Planar mixture, start 0.53 nats away, three
runs each:

@!relativistic-pairing-on-the-mixture-testbed

- Both fit; every saturating run reaches 0.01 nats first, at step 1,800.
- Near the data $w_{\textrm{Rp}}$ is not small, so nothing saturates.
:::

::: {.slide title="Six Units from the Data the Saturating Pairing Generator Does Not Move"}
$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\theta, 1)$ from $\theta = 6$:
every $x'$ ranks below every real $x$, so $w_{\textrm{Rp}} \approx 0$.
Three runs per arm, 1,500 steps:

| critic, generator loss | median gradient, first 100 steps | $\theta$ at the end | within 0.5 of the data |
|:--|:--|:--|:--|
| pairing, saturating | below 0.0005 | 5.99--6.00 | never |
| classical, saturating | 0.006--0.015 | 5.75--5.77 | never |
| pairing, non-saturating | 3.0--3.2 | 0.00 | steps 315--335 |
| classical, non-saturating | 2.7--3.0 | 0.00--0.01 | steps 377--388 |

::: {.d2l-note}
A critic that ranks every generated sample last gives the saturating
generator nothing to follow: train the non-saturating form.
:::
:::

::: {.slide title="Check: Where Can $x'_2$ Go?"}
[Check]{.kicker}

$n = 2$, both generated points on $x_1$, fixed pairing $x_i \leftrightarrow x'_i$.
What is the paired loss, and what happens as $x'_2$ moves toward $x_2$?

::: {.fragment .fade-up}
$-\tfrac12 \log 2$: the self-loop $x_1 \to x'_1$ ties, and $x'_2$ on $x_1$
is scored as fake. The loss stays flat all the way, then drops to
$-\log 2$ when $x'_2$ reaches $x_2$.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Ranking pairs: Bradley–Terry; the critic is $\log(p/q)$ up to a constant.
- Value: $\mathrm{JS}(p \otimes q, q \otimes p)$; two atoms 0.319 nats
  against 0.193 for one sample.
- Between JS and $\log 2$, twice JS near $q = p$, $\log 2$ on disjoint supports.
- Collision minima at $n = 3$: 21, 21, 0, the 0 only for a fixed pairing.
- Train the generator on the non-saturating form: near the data both fit;
  six units away the saturating one stays at $\theta = 6$.

**Next:** whether gradient play reaches the equilibrium, and coverage on 25
Gaussians with and without pairing.
:::
