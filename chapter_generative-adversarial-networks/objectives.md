# Adversarial Objectives and Divergences
:label:`sec_gan_objectives`

Open almost any GAN codebase and its training script offers a menu of losses: non-saturating, least squares, hinge, Wasserstein. Practitioners switch between them the way they switch optimizers, keeping whichever trains. :numref:`sec_basic_gan` analyzed only the first entry. Its critic learned the log ratio of the two densities, and the value of its game was the Jensen--Shannon divergence, a quantity that stops changing once the generated samples no longer overlap the data. For images that is the normal situation: real images and the samples of a young generator usually lie on nearly disjoint sets. So what do the other entries compute? And when a generator starts far from the data, is the missing gradient a property of the loss, of the critic, or of the divergence that the game measures?

Every entry of the menu changes one of two things: the loss that trains the critic, or the set of functions the critic may be. A coin with two outcomes shows what any classification loss computes. Restricting the critic instead gives a second family, which measures how far apart two distributions are rather than how much they overlap. We then test both families twice: with a one-parameter generator that starts far from the data, and on a mixture of three Gaussians whose densities we know.

```{.python .input #objectives-setup}
%%tab jax
# Deterministic GPU kernels; JAX reads XLA_FLAGS once, before the imports
import os
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_deterministic_ops=true').strip()
```

```{.python .input #objectives-adversarial-objectives-and-divergences}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #objectives-adversarial-objectives-and-divergences}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #objectives-adversarial-objectives-and-divergences-1}
%%tab pytorch
from torch.nn.functional import relu, softplus
```

```{.python .input #objectives-adversarial-objectives-and-divergences-1}
%%tab jax
from jax.nn import relu, softplus
```

## A Common Template and Two Choices

The game of :numref:`sec_basic_gan` paid the critic $\log \sigma(D(x))$ for a real sample $x \sim p$ and $\log \sigma(-D(x'))$ for a generated sample $x' \sim q$, where $D$ is the critic's logit, and the critic maximized the sum of the two averages. We keep that section's notation: the mixture $m = (p + q)/2$, the density ratio $\rho = p/q$, and its logarithm $\lambda = \log \rho$, which is the optimal logit. Replacing the two log-probabilities by general *payoffs*, $\ell_p$ for real samples and $\ell_q$ for generated ones, gives the template

$$
d(p, q) \;=\; \sup_{T \in \mathcal{T}} \Big\{ \mathbb{E}_{x \sim p}\big[\ell_p(T(x))\big] + \mathbb{E}_{x' \sim q}\big[\ell_q(T(x'))\big] \Big\},
$$
:eqlabel:`eq_gan_template`

in which the critic $T$ ranges over a class $\mathcal{T}$. The game of :numref:`sec_basic_gan` is the case $\ell_p(t) = \log \sigma(t)$ and $\ell_q(t) = \log \sigma(-t)$ with $\mathcal{T}$ unrestricted, and its value is $2\,\mathrm{JS}(p, q) - 2 \log 2$. Throughout this section the two payoffs are added and the critic maximizes their sum; when we speak of a loss, we mean a negative payoff.

The two choices, the payoffs and the class, combine as in :numref:`tab_gan_objective_families`. With a free critic and nonlinear payoffs, the supremum can be taken separately at each point $x$, so the value is an average of a function of the ratio $\rho(x)$, blind to distances between points; every classification loss lands in this cell. With linear payoffs the critic has to be restricted, and the value becomes a difference of averages that depends on distances between samples. Critics that score a pair of samples, the bottom row, come in the next two sections.

:Two choices define an adversarial objective: what the critic scores, and whether its class is restricted. Each cell names the objectives of that kind in this chapter.
:label:`tab_gan_objective_families`

| | free critic | restricted or penalized critic |
|:---|:---|:---|
| **critic scores one sample** | classification losses (logistic, least squares, hinge), whose value is a function of $p/q$; this section | linear payoffs on a kernel ball or a Lipschitz ball, this section; in practice also the hinge loss with a spectrally normalized critic |
| **critic scores a pair** | the pairing loss, :numref:`sec_gan_relativistic` | the pairing loss with zero-centered gradient penalties, :numref:`sec_gan_convergence` and :numref:`sec_dcgan` |

One property holds for every objective of the top row, and it takes a line to prove. For a fixed critic, the expression in braces in :eqref:`eq_gan_template` is linear in the pair $(p, q)$, and a supremum of linear functions is convex. So for fixed data $p$ the objective is convex in $q$, and over the space of all distributions the generator's problem has no spurious local minima. This sounds better than it is. A generator moves its parameters, not a distribution, and the distributions a network can represent do not form a convex set: the average of two Gaussians is not a Gaussian. Convexity in $q$ says nothing about convexity in the parameters (Exercise :exref:`ex_objectives-convexity-parameterization`).

## Classification Losses and Their Divergences

### Two Outcomes by Hand

Take a coin. The data put probabilities $p = (\tfrac34, \tfrac14)$ on heads and tails, and the generator $q = (\tfrac14, \tfrac34)$. Each toss comes from the data or from the generator with probability one half, and the critic, shown the outcome, must say which source produced it. Seeing heads, the source is real with probability $\eta = p/(p + q) = \tfrac34$; seeing tails, with probability $\tfrac14$. A critic that ignores the outcome can do no better than report one half. How much looking helps depends on the loss:

- Under the log loss, the best report is the posterior itself. Its expected loss is the binary entropy $H_b(\tfrac34) = 0.562$ nats, against $\log 2 = 0.693$ nats for the blind report. Looking saves $0.131$ nats, which is $\mathrm{JS}(p, q)$.
- Under the squared error of a reported probability, the best report is again the posterior, with expected loss $\tfrac34 \cdot \tfrac14 = \tfrac{3}{16}$ against $\tfrac14$ for the blind report. Looking saves $\tfrac{1}{16}$.
- Under the hinge loss on a score $t$, the best scores are $t = +1$ on heads and $t = -1$ on tails, with expected loss $2 \cdot \tfrac14 = \tfrac12$ against $1$ at $t = 0$. Looking saves $\tfrac12$, which is the total variation distance $\tfrac12 \sum |p - q|$.

The three losses assign three numbers to one pair of distributions, all from the same posterior, and each saving vanishes when $p = q$. We now show that every such saving is an f-divergence.

### The Bayes-Risk Gap

Classification losses pay both kinds of sample with one payoff $\ell$, applied to the critic's score on a real sample and to minus its score on a generated one, $\ell_p(t) = \ell(t)$ and $\ell_q(t) = \ell(-t)$. For a fixed critic $D$ the objective is

$$
V_\ell(D) \;=\; \mathbb{E}_{x \sim p}\big[\ell(D(x))\big] + \mathbb{E}_{x' \sim q}\big[\ell(-D(x'))\big],
$$
:eqlabel:`eq_gan_margin`

and the choice $\ell = \log \sigma$ recovers the log-loss game. The signed score $t$, which is $D(x)$ on a real sample and $-D(x')$ on a generated one, is the critic's *margin*: positive when the critic is right, and the larger, the more confident. The loss $-\ell(t)$ is a function of the margin, for instance $\max(0, 1 - t)$ for the hinge.

As in :numref:`sec_basic_gan`, the maximization decouples across points. At each $x$ the integrand is $p\,\ell(t) + q\,\ell(-t)$ with $t = D(x)$, which is $2m(x)$ times an average under the posterior $\eta(x) = p(x)/(p(x)+q(x))$ that $x$ is real. The smallest expected loss a critic can reach at posterior $\eta$ is the *conditional Bayes risk*

$$
L(\eta) \;=\; \inf_{t} \Big\{ \eta \, \big({-\ell(t)}\big) + (1 - \eta)\, \big({-\ell(-t)}\big) \Big\}.
$$
:eqlabel:`eq_gan_bayes_risk`

As an infimum of functions linear in $\eta$, $L$ is concave for every $\ell$. A critic that ignores its input already reaches the risk $L(\tfrac12)$. Subtracting this baseline leaves what observing the sample contributes, the *Bayes-risk gap*

$$
\Delta_\ell(p, q) \;=\; L(\tfrac12) - \mathbb{E}_{x \sim m}\big[L(\eta(x))\big].
$$
:eqlabel:`eq_gan_bayes_gap`

It fixes the value of the game, $\sup_D V_\ell = -2\,\mathbb{E}_m[L(\eta)] = 2\Delta_\ell - 2 L(\tfrac12)$, and it is nonnegative by Jensen's inequality, because $L$ is concave and $\mathbb{E}_m[\eta] = \tfrac12$. For the log loss, $L$ is the binary entropy, and the gap is the Jensen--Shannon divergence. The three savings of the coin are three Bayes-risk gaps.

### Bayes-Risk Gaps as f-Divergences

An f-divergence averages a convex function of the density ratio under $q$, $D_f(p \,\|\, q) = \mathbb{E}_{x \sim q}[f(\rho(x))]$, where $f$ is convex with $f(1) = 0$. :numref:`sec_mdl-f-divergences` proves that $D_f \geq 0$ and fixes the conventions where one density vanishes. We call $f$ the *divergence function*; the f-GAN literature calls it the generator, a word this chapter keeps for the network.

**Proposition.** *If the Bayes risk $L$ is finite on $[0, 1]$, then*

$$
\Delta_\ell(p, q) \;=\; D_f(p \,\|\, q)
\qquad \textrm{with} \qquad
f(u) \;=\; L(\tfrac12) - \frac{u+1}{2}\, L\!\left(\frac{u}{u+1}\right),
$$

*and this $f$ is convex with $f(1) = 0$.*

**Proof.** Substitute $u = \rho = p/q$, so that $m = q\,(u+1)/2$ and $\eta = u/(u+1)$. Then

$$
\mathbb{E}_{x \sim m}\big[L(\eta(x))\big] = \int q \, \frac{u+1}{2}\, L\!\left(\frac{u}{u+1}\right),
$$

and since $\int q = 1$ the constant $L(\tfrac12)$ can be moved inside the integral, giving $\Delta_\ell = \int q\, f(u) = \mathbb{E}_q[f(\rho)]$. For convexity, write the concave $L$ as the lower envelope of its tangent lines, $L(v) = \inf_j \{\alpha_j v + \beta_j\}$. Then

$$
(u+1)\, L\!\left(\frac{u}{u+1}\right) = \inf_j \big\{ \alpha_j u + \beta_j (u+1) \big\}
$$

is an infimum of affine functions of $u$, hence concave, so $f$ is convex; and $f(1) = L(\tfrac12) - L(\tfrac12) = 0$. $\blacksquare$

Thus every classification loss with a free critic evaluates the f-divergence built from its Bayes risk. What changes from loss to loss is what the optimal critic reports. It is still a fixed function of the posterior $\eta(x)$, and hence of $\lambda(x)$, since $\eta = \sigma(\lambda)$. We call this function of $\lambda$ the loss's *link*: the logistic critic outputs $\lambda$ itself, the hinge critic only its sign.

### Divergences of Common Losses

:numref:`tab_gan_loss_divergences` evaluates :eqref:`eq_gan_bayes_gap` for four common losses, with the binary entropy $H_b(\eta)$ and the total variation distance $\mathrm{TV}(p, q) = \tfrac12 \int |p - q|$.

:Classification losses with a free critic: the loss at margin $t$, the conditional Bayes risk $L(\eta)$, the Bayes-risk gap $\Delta_\ell$ (the value of the game is $2\Delta_\ell - 2L(\tfrac12)$), the link (the optimal critic's output as a function of $\lambda$), and the generator loss used in practice, on the critic's output $D = D(x')$ for a generated sample.
:label:`tab_gan_loss_divergences`

| loss | loss at margin $t$ | Bayes risk $L(\eta)$ | gap $\Delta_\ell(p,q)$ | link | generator loss |
|:---|:---|:---|:---|:---|:---|
| logistic | $\log(1 + e^{-t})$ | $H_b(\eta)$ | $\mathrm{JS}(p,q)$ | $\lambda$ | $-\log \sigma(D)$ |
| square | $\tfrac14 (1 - t)^2$, $t = 2D - 1$ | $\eta(1-\eta)$ | $\tfrac{1}{8}\int \frac{(p-q)^2}{p+q}$ | $\sigma(\lambda)$ | $\tfrac12 (D - 1)^2$ |
| squared hinge | $\max(0, 1 - t)^2$ | $4\eta(1-\eta)$ | $\tfrac{1}{2}\int \frac{(p-q)^2}{p+q}$ | $\tanh(\lambda/2)$ | $-D$ |
| hinge | $\max(0, 1 - t)$ | $2\min(\eta, 1-\eta)$ | $\mathrm{TV}(p,q)$ | $\operatorname{sign} \lambda$ | $-D$ |

At the coin pair the four gaps are $0.131$, $\tfrac{1}{16}$, $\tfrac14$ and $\tfrac12$. Three rows are written on the margin directly. The least-squares critic of LSGAN :cite:`Mao.Li.Xie.ea.2017` instead regresses its output $D$ onto the target $1$ for real and $0$ for generated samples. The map $t = 2D - 1$ turns its losses $(D - 1)^2$ and $D^2$ into $\tfrac14 (1 - t)^2$ and $\tfrac14 (1 + t)^2$, the square loss on targets $\pm 1$ at the margins $t$ and $-t$. Its gap is a quarter of the bounded divergence $\chi^2(p \,\|\, m) = \tfrac12 \int (p-q)^2/(p+q)$.

The squared hinge is rarely used in practice. We include it because it shares the hinge's margin and generator loss, yet its link, $\tanh(\lambda/2)$, reports the size of the mismatch as well as its sign. It pays $\max(0, 1 - t)^2$, which agrees with $(1 - t)^2$ for $t \leq 1$, and its optimum $t = 2\eta - 1$ lies in that range, so its risk and its gap are four times those of the square row. The hinge itself came to adversarial training from support vector machines through the Geometric GAN :cite:`Lim.Ye.2017`. At $\eta = \tfrac12$ every score in $[-1, 1]$ is optimal for it, so we should expect a trained hinge critic to scatter near the decision boundary.

None of the entries in the last column is the generator's side of the game whose divergence the table lists. The logistic generator uses the non-saturating loss of :numref:`sec_basic_gan`. The least-squares generator pulls the critic's output on its samples toward the real target, $\tfrac12 (D(x') - 1)^2$. Both hinge games let the generator simply raise the critic's score, minimizing $-D(x')$. Each choice keeps the equilibrium $q = p$ with a constant critic, where every generator gradient vanishes. Against a fixed critic, however, the generator descends a different function from the divergence in the table, as the non-saturating loss did in :numref:`sec_basic_gan`.

### The Variational Bound

Which divergences can a classification loss not reach? The forward KL divergence $D_{\textrm{KL}}(p \,\|\, q)$, which maximum likelihood minimizes, is one. It is unbounded, while a nonnegative loss has $L \geq 0$ and hence a gap of at most $L(\tfrac12)$. More generally, swapping $p$ and $q$ turns $\eta$ into $1 - \eta$, and $L(\eta) = L(1 - \eta)$ because one payoff scores both $D$ and $-D$. So every gap is symmetric in $p$ and $q$, and no classification loss at all reaches an asymmetric divergence such as the forward KL. Conversely, every symmetric f-divergence is the gap of some classification loss (Exercise :exref:`ex_objectives-divergences-a-loss-can-reach`).

A different construction builds a game for any convex divergence function $f$. Draw the graph of $f$ and, for a slope $t$, the highest straight line of slope $t$ that stays below it. Call its intercept $-f^*(t)$. The function $f^*(t) = \sup_u \{ut - f(u)\}$ is the *convex conjugate* of $f$. Because the line stays below the graph, $f(u) \geq ut - f^*(t)$ for all $u$ and $t$, with equality where the line touches the graph, at the $u$ with $f'(u) = t$. This is the Fenchel--Young inequality.

Now let a critic $T$ choose the slope at each point: put $u = \rho(x)$ and $t = T(x)$, and average under $q$. Since $\mathbb{E}_q[\rho\, T] = \mathbb{E}_p[T]$,

$$
D_f(p \,\|\, q) \;\geq\; \mathbb{E}_{x \sim p}\big[T(x)\big] - \mathbb{E}_{x' \sim q}\big[f^*(T(x'))\big],
$$
:eqlabel:`eq_gan_fgan_bound`

so every critic certifies a lower bound, and an imperfect critic underestimates the divergence. The critic $T^\star = f'(\rho)$, whose line touches the graph at every point, attains the bound, and it is once more a monotone link of the ratio. In the template this is the case $\ell_p(t) = t$ and $\ell_q(t) = -f^*(t)$: real and generated samples are paid by different functions, which is how a game reaches an asymmetric divergence. Training a network $T$ to maximize the right-hand side is the f-GAN construction :cite:`Nowozin.Cseke.Tomioka.2016` (:numref:`sec_mdl-f-gan-dual` has the details), and its Jensen--Shannon member is the original GAN (Exercise :exref:`ex_objectives-variational-bound-eqref-eq`). For the forward KL, $f(u) = u \log u$ has the conjugate $f^*(t) = e^{t-1}$, and the optimal critic is $f'(\rho) = 1 + \lambda$.

### Critic Links
:label:`subsec_gan_objectives_critics`

Five links of one ratio have appeared: the four of :numref:`tab_gan_loss_divergences` and $1 + \lambda$ for the forward KL. When both densities are known, we can check point by point whether trained critics follow them. We check four of them, $\lambda$, $\sigma(\lambda)$, $\operatorname{sign} \lambda$ and $1 + \lambda$, and return to the squared hinge in the last experiment. The data $p$ are a uniform mixture of three Gaussians in the plane. For $q$ we set a second mixture by hand, with each mean moved by up to $0.3$ and the components made wider, narrower and sheared, so that $\lambda$ has a closed form at every point.

```{.python .input #objectives-one-testbed-four-losses-1}
%%tab pytorch, jax
#@save
mu_p = np.array([[0.0, 2.4], [-2.1, -1.2], [2.1, -1.2]])     # data
A_p = np.stack([0.7 * np.eye(2)] * 3)

mu_q = np.array([[0.2, 2.6], [-2.3, -1.0], [2.0, -1.3]])     # hand-set q
A_q = np.stack([np.diag([0.8, 0.55]), np.diag([0.5, 0.6]),
                np.array([[0.65, 0.2], [-0.1, 0.7]])])

#@save
def mixture_logpdf(x, mus, As):
    """Log density of the uniform mixture of Gaussians N(mu_k, A_k^T A_k)."""
    comps = []
    for mu, A in zip(mus, As):
        Sigma, diff = A.T @ A, x - mu
        sol = np.linalg.solve(Sigma, diff.T).T
        comps.append(-0.5 * ((diff * sol).sum(1)
                             + np.linalg.slogdet(2 * np.pi * Sigma)[1]))
    return np.logaddexp.reduce(np.stack(comps) - np.log(len(mus)), axis=0)

#@save
def sample_mixture(n, mus, As, rng):
    """n draws from the uniform mixture of Gaussians mu_k + z A_k."""
    k = rng.integers(0, len(mus), n)
    z = rng.standard_normal((n, 2))
    return np.einsum('ni,nij->nj', z, As[k]) + mus[k]
```

On a fixed grid, quadrature gives $\mathrm{JS}(p, q)$ for any mixture $q$ of this form. Later in the section the same function scores a trained generator after every step.

```{.python .input #objectives-js-by-quadrature}
%%tab pytorch, jax
#@save
grid_1d = np.linspace(-7.0, 7.0, 281)
grid_2d = np.stack([g.ravel() for g in np.meshgrid(grid_1d, grid_1d)], axis=1)
cell_area = (grid_1d[1] - grid_1d[0]) ** 2

#@save
def js_mixtures(mus_q, As_q, mus_p=mu_p, As_p=A_p):
    """JS(p, q) between two known mixtures, p the data by default, by
    quadrature on grid_2d."""
    logp = mixture_logpdf(grid_2d, mus_p, As_p)
    logq = mixture_logpdf(grid_2d, mus_q, As_q)
    p, q = np.exp(logp), np.exp(logq)
    logm = np.log(np.maximum((p + q) / 2, 1e-300))
    return 0.5 * (p * (logp - logm) + q * (logq - logm)).sum() * cell_area

print(f'JS(p, q) for the hand-set q: {js_mixtures(mu_q, A_q):.3f} nats')
```

The hand-set $q$ is a few hundredths of a nat from the data, a mismatch of the size a generator meets late in training. The critic is a small leaky-ReLU network with two hidden layers of 64 units.

```{.python .input #objectives-one-testbed-four-losses-3}
%%tab pytorch
#@save
TRUNC_STD = 0.8796  # standard deviation of N(0, 1) truncated at +-2

#@save
def make_critic():
    """Leaky-ReLU MLP with LeCun-normal weights and zero biases."""
    D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2),
                      nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    for layer in D[::2]:  # weight std 1/sqrt(fan_in) after truncation
        s = layer.in_features ** -0.5 / TRUNC_STD
        nn.init.trunc_normal_(layer.weight, std=s, a=-2 * s, b=2 * s)
        nn.init.zeros_(layer.bias)
    return D
```

```{.python .input #objectives-one-testbed-four-losses-3}
%%tab jax
#@save
def make_critic(rngs):
    """Leaky-ReLU MLP; nnx.Linear gives LeCun-normal weights, zero biases."""
    leaky = lambda x: nnx.leaky_relu(x, 0.2)
    return nnx.Sequential(nnx.Linear(2, 64, rngs=rngs), leaky,
                          nnx.Linear(64, 64, rngs=rngs), leaky,
                          nnx.Linear(64, 1, rngs=rngs))
```

Each loss is stored as a pair, the critic's loss and the generator's loss of :numref:`tab_gan_loss_divergences`, both written on the critic's scores `r` of a real batch and `f` of a generated one.

```{.python .input #objectives-loss-pairs}
%%tab pytorch, jax
losses = {  # name: (critic loss, generator loss), both on the scores r, f
    'logistic': (lambda r, f: (softplus(-r) + softplus(f)).mean(),
                 lambda r, f: softplus(-f).mean()),
    'least squares': (lambda r, f: ((r - 1) ** 2 + f ** 2).mean() / 2,
                      lambda r, f: ((f - 1) ** 2).mean() / 2),
    'hinge': (lambda r, f: (relu(1 - r) + relu(1 + f)).mean(),
              lambda r, f: -f.mean()),
    'squared hinge': (lambda r, f: (relu(1 - r) ** 2 + relu(1 + f) ** 2).mean(),
                      lambda r, f: -f.mean()),
}
```

A link describes the best response to a fixed $q$, so we train a fresh critic against the hand-set $q$ under three of the losses, and a fourth to maximize the forward-KL bound $\mathbb{E}_p[T] - \mathbb{E}_q[e^{T-1}]$.

```{.python .input #objectives-what-each-critic-estimates}
%%tab pytorch
#@save
to_t = lambda a: torch.tensor(a, dtype=torch.float32)

def best_response(critic_loss, seed=0, steps=3000, batch=512, lr=1e-3):
    """Fresh critic trained against the fixed q, as a NumPy function."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    D = make_critic()
    opt = torch.optim.Adam(D.parameters(), lr)
    for _ in range(steps):
        X = to_t(sample_mixture(batch, mu_p, A_p, rng))
        Y = to_t(sample_mixture(batch, mu_q, A_q, rng))
        opt.zero_grad()
        critic_loss(D(X), D(Y)).backward()
        opt.step()
    return lambda x: D(to_t(x)).squeeze(-1).detach().numpy()

kl_dual = lambda r, f: -(r.mean() - torch.exp(f - 1).mean())
critics = {name: best_response(losses[name][0])
           for name in ('logistic', 'least squares', 'hinge')}
critics['forward KL'] = best_response(kl_dual)
```

```{.python .input #objectives-what-each-critic-estimates}
%%tab jax
@nnx.jit(static_argnums=0)
def best_response_step(critic_loss, D, opt, X, Y):
    opt.update(D, nnx.grad(lambda D_: critic_loss(D_(X), D_(Y)))(D))

def best_response(critic_loss, seed=0, steps=3000, batch=512, lr=1e-3):
    """Fresh critic trained against the fixed q, as a NumPy function."""
    rng = np.random.default_rng(seed)
    D = make_critic(nnx.Rngs(seed))
    opt = nnx.Optimizer(D, optax.adam(lr), wrt=nnx.Param)
    for _ in range(steps):
        X = jnp.asarray(sample_mixture(batch, mu_p, A_p, rng), jnp.float32)
        Y = jnp.asarray(sample_mixture(batch, mu_q, A_q, rng), jnp.float32)
        best_response_step(critic_loss, D, opt, X, Y)
    return lambda x: np.asarray(D(jnp.asarray(x, jnp.float32))).squeeze(-1)

kl_dual = lambda r, f: -(r.mean() - jnp.exp(f - 1).mean())
critics = {name: best_response(losses[name][0])
           for name in ('logistic', 'least squares', 'hinge')}
critics['forward KL'] = best_response(kl_dual)
```

We evaluate each critic at 300 points drawn from $p$ and 300 from $q$ and plot its output against the analytic $\lambda$, with the predicted link as a dashed line.

```{.python .input #objectives-critic-links-1}
%%tab pytorch, jax
rng = np.random.default_rng(7)
pts = np.concatenate([sample_mixture(300, mu_p, A_p, rng),
                      sample_mixture(300, mu_q, A_q, rng)])
lam = mixture_logpdf(pts, mu_p, A_p) - mixture_logpdf(pts, mu_q, A_q)
lam_line = np.linspace(lam.min(), lam.max(), 200)
links = {'logistic': lambda l: l, 'least squares': lambda l: 1 / (1 + np.exp(-l)),
         'hinge': np.sign, 'forward KL': lambda l: 1 + l}
fig, axes = d2l.plt.subplots(1, 4, figsize=(12, 3))
for ax, (name, link) in zip(axes, links.items()):
    out = critics[name](pts)
    d2l.plot([lam[:300], lam[300:], lam_line],
             [out[:300], out[300:], link(lam_line)], xlabel=r'$\lambda$',
             fmts=('.', '.', 'k--'), axes=ax,
             legend=['x ~ p', "x' ~ q", 'link'] if name == 'logistic' else None)
    ax.set_title(name)
axes[0].set_ylabel('critic output')
fig.tight_layout()
```

Each critic follows its own link of the same function: the identity, the sigmoid, a step at $\lambda = 0$, and the line $1 + \lambda$. Four networks trained on the same two mixtures report four different functions of one ratio, so what a critic outputs is decided by its loss. All four drift from their links where $|\lambda|$ is large, because the objective constrains a critic only where samples land, and few land there. The hinge critic scatters most widely around its step, as its flat risk near $\eta = \tfrac12$ predicts: close to the boundary, almost any score in $[-1, 1]$ is nearly optimal.

## Integral Probability Metrics

The first family let the critic be anything and paid it nonlinearly. The second pays it linearly, $\ell_p(t) = t$ and $\ell_q(t) = -t$, so the critic simply tries to make its average on real samples exceed its average on generated ones. A free critic would win this game without limit: scaling $T$ by any factor scales the difference with it, so the supremum is infinite whenever $p \neq q$. The critic must therefore be restricted to a class $\mathcal{F}$ of test functions, which gives

$$
d_{\mathcal{F}}(p, q) \;=\; \sup_{T \in \mathcal{F}} \Big\{ \mathbb{E}_{x \sim p}\big[T(x)\big] - \mathbb{E}_{x' \sim q}\big[T(x')\big] \Big\},
$$
:eqlabel:`eq_gan_ipm`

the largest difference in expectation that a test function in $\mathcal{F}$ can detect. Such a distance is called an *integral probability metric* (:numref:`sec_mdl-ipm-mmd`). Its value no longer reduces to the pointwise ratio $\rho(x)$; it depends on which functions $\mathcal{F}$ contains, and through them on the distances between points.

### Maximum Mean Discrepancy

Suppose we can measure the similarity $k(x, x')$ of two points, for instance with the RBF kernel $k(x, x') = \exp(-\|x - x'\|^2/(2s^2))$ of bandwidth $s$. If $p = q$, a point drawn from $p$ is on average as similar to a second draw from $p$ as to a draw from $q$. The *maximum mean discrepancy* (MMD) measures how far this fails:

$$
\mathrm{MMD}^2(p, q) \;=\; \mathbb{E}_{x, \tilde x \sim p}\big[k(x, \tilde x)\big] + \mathbb{E}_{x', \tilde x' \sim q}\big[k(x', \tilde x')\big] - 2\, \mathbb{E}_{x \sim p,\, x' \sim q}\big[k(x, x')\big],
$$
:eqlabel:`eq_gan_mmd2`

with the two samples in each average drawn independently: similarity within each distribution minus twice the similarity across them. MMD is an integral probability metric, whose class $\mathcal{F}$ is the unit ball of the function space that the kernel defines, its reproducing kernel Hilbert space :cite:`Gretton.Borgwardt.Rasch.ea.2012`; :numref:`sec_mdl-ipm-mmd` derives :eqref:`eq_gan_mmd2` from :eqref:`eq_gan_ipm`. The test function that attains the supremum is, up to a positive factor, $w(x) = \mathbb{E}_{\tilde x \sim p}[k(x, \tilde x)] - \mathbb{E}_{\tilde x' \sim q}[k(x, \tilde x')]$, the average similarity of $x$ to the data minus its average similarity to the generated samples. It is called the *witness*, because wherever it is far from zero it testifies that $p$ and $q$ differ.

For a fixed kernel a generator can minimize $\mathrm{MMD}^2$ directly from samples, with no critic network, and the kernel alone decides which differences are visible. The unbiased minibatch estimate averages each kernel matrix, leaving out the diagonal terms that compare a point with itself. The function below takes the three matrices, so it serves any kernel; :numref:`sec_dcgan` scores images with it.

```{.python .input #objectives-maximum-mean-discrepancy}
%%tab pytorch, jax
#@save
def mmd2_unbiased(K_pp, K_qq, K_pq):
    """Unbiased MMD^2 from the kernel matrices within p, within q, and
    across; the diagonals (a point with itself) are left out."""
    m, n = len(K_pp), len(K_qq)
    return ((K_pp.sum() - K_pp.diagonal().sum()) / (m * (m - 1))
            + (K_qq.sum() - K_qq.diagonal().sum()) / (n * (n - 1))
            - 2 * K_pq.mean())
```

### Wasserstein-1

The second class is the *1-Lipschitz* functions, with $|T(x) - T(y)| \leq \|x - y\|$ for all $x$ and $y$: their slope never exceeds one in any direction. Their integral probability metric has a physical meaning. Picture $p$ as a pile of sand and $q$ as a hole of the same volume. Moving a grain a distance $r$ costs $r$, and the cheapest way of filling the hole costs, on average per grain,

$$
W_1(p, q) \;=\; \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, x') \sim \pi}\big[\|x - x'\|\big],
$$

where a *transport plan* $\pi \in \Pi(p, q)$ is a joint distribution of $(x, x')$ with marginals $p$ and $q$; it says how much mass moves from where to where. This is the *Wasserstein-1* distance. That the best 1-Lipschitz critic and the cheapest plan give the same number is a theorem, and a remarkable one: the *Kantorovich--Rubinstein duality* states that $d_{\mathcal{F}}(p, q) = W_1(p, q)$ for the 1-Lipschitz class $\mathcal{F}$ whenever $p$ and $q$ have finite first moments, $\mathbb{E}\|x\| < \infty$ (:numref:`sec_mdl-optimal-transport` proves it on the line). An optimal critic, called a *Kantorovich potential*, changes with unit slope along the paths on which mass travels, falling from the data toward the generated samples. The distance $\|x - x'\|$ is the norm of the space in which the samples live, for images the Euclidean distance between pixel arrays. This is why $W_1$ gives a direction across any gap, and also why it is a crude notion of similarity for images: a picture shifted by one pixel can be far from itself.

On the line, $W_1$ is the area between the two cumulative distribution functions,

$$
W_1(p, q) \;=\; \int_{-\infty}^{\infty} \big| F_p(t) - F_q(t) \big| \, dt,
$$

so a translation by $\mu$ costs exactly $|\mu|$. In higher dimensions a Wasserstein GAN (WGAN) trains a critic network instead :cite:`Arjovsky.Chintala.Bottou.2017`. A plain network is not 1-Lipschitz, and the three devices in use enforce three different things:

- *Weight clipping* :cite:`Arjovsky.Chintala.Bottou.2017` keeps every weight in an interval $[-c, c]$. The network is then $\kappa$-Lipschitz for some unknown $\kappa$ that depends on $c$ and the architecture, so at best the critic estimates $\kappa\, W_1$.
- The *gradient penalty* of WGAN-GP :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017` adds $\gamma\, \mathbb{E}\big[(\|\nabla_x T\| - 1)^2\big]$ to the critic's loss at random points between real and generated samples. It pulls the slope toward one softly, and only where it is evaluated.
- *Spectral normalization* :cite:`Miyato.Kataoka.Koyama.ea.2018` divides each weight matrix by an estimate of its largest singular value, which makes a dense layer, and with 1-Lipschitz activations the network, roughly 1-Lipschitz everywhere; for convolutions the bound is looser. That is sufficient but not necessary, and it excludes many 1-Lipschitz functions.

None of the three gives exactly $W_1$. Fortunately the generator does not need the value; it uses the critic's slope at its samples.

### The Hinge between the Families

The hinge loss sits between the two families. When $|D| \leq 1$ its payoffs are linear, $D - 1$ on real and $-D - 1$ on generated samples, and clipping a critic's output to $[-1, 1]$ never lowers them. The free hinge game is therefore the integral probability metric of the functions bounded by one, shifted by $-2$, and its value $2\,\mathrm{TV} - 2$ agrees with :numref:`tab_gan_loss_divergences`. Clipping also keeps a Lipschitz constant intact. If the hinge critic is in addition $\kappa$-Lipschitz, as spectral normalization makes it, the game between two point masses a distance $\mu$ apart is worth $\min(\kappa \mu, 2) - 2$: its value grows like $W_1$ up to the separation $2/\kappa$ and is flat beyond it. So a Lipschitz hinge critic, the mixed cell of :numref:`tab_gan_objective_families`, keeps a slope that total variation lacks, over a range set by $\kappa$.

### Estimation from Samples
:label:`subsec_gan_objectives_estimation`

The two metrics differ in how well samples determine them. The unbiased MMD estimate converges at the rate $O(n^{-1/2})$ in any dimension :cite:`Gretton.Borgwardt.Rasch.ea.2012`, whereas the empirical $W_1$ in dimension $d_x > 2$ converges at the rate $n^{-1/d_x}$ :cite:`Weed.Bach.2019`, so halving its error can take $2^{d_x}$ times as many samples. That seems to undercut the Wasserstein critic a page after we introduced it. It does not, because a WGAN critic never computes $W_1$ between two minibatches: it is a small network, and what the generator needs from it is a smooth direction toward the data, not the value. Density-ratio objectives have their own difficulty where the ratio is extreme. At the optimal forward-KL critic each term $e^{T-1}$ of the minibatch average under $q$ equals $\rho$, whose variance under $q$ is $\chi^2(p \,\|\, q) = \int (p - q)^2/q$. That variance can be infinite even when the KL divergence is finite (Exercise :exref:`ex_objectives-estimation-and-dimension` measures both effects).

## Gradients Far from and Near the Data
:label:`subsec_gan_objectives_gradients`

The two families disagree about the failure that closed :numref:`sec_basic_gan`. An f-divergence integrates a function of $\rho(x)$, which on disjoint supports is $0$ or $\infty$ at every point, so nothing records how far apart the supports are. The divergence is the same constant, finite or infinite, at every separation, and a generator that descends it gets no gradient. An integral probability metric averages one test function under both distributions, and if the test functions vary smoothly in space, its value changes as the supports move. Smoothness is the condition, not the family: total variation is the integral probability metric of the functions bounded by $\tfrac12$ :cite:`Sriperumbudur.Fukumizu.Gretton.ea.2009`, yet it saturates on disjoint supports, because that class contains functions as sharp as indicators.

### Divergence against Separation
:label:`subsec_gan_objectives_separation`

Let $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ move apart. All three objectives can be evaluated without sampling error: Jensen--Shannon by quadrature, and $W_1 = |\mu|$ exactly. For the RBF kernel with bandwidth $s$, the kernel averages in :eqref:`eq_gan_mmd2` are Gaussian integrals, which give

$$
\mathrm{MMD}^2(p, q) \;=\; \frac{2s}{\sqrt{s^2 + 2}} \left( 1 - \exp\!\left( -\frac{\mu^2}{2(s^2 + 2)} \right) \right).
$$
:eqlabel:`eq_gan_mmd_gauss`

Its slope in $\mu$ is proportional to $\mu \exp(-\mu^2/(2(s^2 + 2)))$, which is largest at $\mu = \sqrt{s^2 + 2}$: about $4.2$ for the bandwidth $s = 4$ that we use. We plot each objective's slope with respect to $\mu$, the gradient that a generator controlling $\mu$ receives, on a log axis.

```{.python .input #objectives-divergence-against-separation}
%%tab pytorch, jax
xs = np.linspace(-10.0, 30.0, 8001)           # fixed quadrature grid

def normal_pdf(x, mu, sigma=1.0):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(
        2 * np.pi * sigma ** 2)

def js_gap(p, q):
    """log 2 - JS(p, q) on xs, accurate even when tiny: the integral of
    (p log(1 + q/p) + q log(1 + p/q)) / 2, with log(1 + e^a) computed as
    logaddexp(0, a)."""
    a = np.log(q) - np.log(p)
    return 0.5 * np.trapezoid(p * np.logaddexp(0, a) + q * np.logaddexp(0, -a),
                              xs)

def js_1d(p, q):
    return np.log(2) - js_gap(p, q)

s = 4.0                                        # RBF kernel bandwidth
mus = np.linspace(0.1, 20.0, 200)              # separations mu
gaps = np.array([js_gap(normal_pdf(xs, 0.0), normal_pdf(xs, mu))
                 for mu in mus])
mmd2 = 2 * s / np.sqrt(s ** 2 + 2) * (1 - np.exp(-mus ** 2
                                                 / (2 * (s ** 2 + 2))))
slopes = [-np.gradient(gaps, mus), np.gradient(mmd2, mus),
          np.ones_like(mus)]                   # W1 = |mu| has slope one
for mu in (1.0, 7.0, 20.0):
    js = js_1d(normal_pdf(xs, 0.0), normal_pdf(xs, mu))
    print(f'mu = {mu:2.0f}: JS slope {np.interp(mu, mus, slopes[0]):.4g}, '
          f'JS / log 2 = {js / np.log(2):.4f}, '
          f'MMD^2 slope {np.interp(mu, mus, slopes[1]):.4g}')
peak = slopes[1].argmax()
print(f'MMD^2 slope peaks at mu = {mus[peak]:.1f} with {slopes[1][peak]:.3f}')
d2l.plot(mus, slopes, xlabel=r'separation $\mu$', ylabel='slope',
         legend=['JS', 'MMD$^2$', '$W_1$'], yscale='log', figsize=(5, 3))
```

The Jensen--Shannon slope collapses once the densities barely overlap. At $\mu = 7$ it is $0.0013$, more than a hundred times smaller than the $0.20$ at $\mu = 1$, and the divergence is within a tenth of a percent of $\log 2$; at $\mu = 20$ the slope is $1.5 \times 10^{-22}$. A generator that descends the Jensen--Shannon divergence this far from the data receives almost no signal about which way to move. The $W_1$ slope is one at every separation. The $\mathrm{MMD}^2$ slope rises to its peak of $0.27$ at $\mu = 4.2$, as predicted, is still $0.19$ at $\mu = 7$, and then decays like a Gaussian in $\mu$, to $3.3 \times 10^{-5}$ at $\mu = 20$. The bandwidth sets the range over which MMD gives a gradient.

### Optimal Critics on Separated Supports

Why does the Jensen--Shannon slope collapse? Not because the log ratio goes flat. Two unit Gaussians never separate completely, and their log ratio $\lambda(x) = \mu^2/2 - \mu x$ is a straight line that grows steeper the farther apart they are. What goes flat is the posterior $\sigma(\lambda)$. Wherever either distribution puts noticeable mass, it is within a hair of $0$ or $1$, and the Jensen--Shannon divergence, which is $\log 2$ minus the average binary entropy of that posterior, has nothing left to change.

In the limit of disjoint supports, drawn in :numref:`fig_gan_critics_separated`, $\lambda$ itself flattens: it is $+\infty$ on one support and $-\infty$ on the other, so moving $q$ a little leaves every generated sample on the same level and carries no information about direction. The $W_1$ critic is a ramp of unit slope, so every generated sample sits on a slope toward the data, however far away. The MMD witness, a smooth difference of kernel bumps, slopes toward the data as long as the kernel reaches across the gap. The hinge critic $\operatorname{sign} \lambda$ needs no limit to go flat: even for two Gaussians it is a step halfway between their means, level at both clouds.

![Optimal critics when the data density $p$ (blue) and the generated density $q$ (orange) have disjoint supports. Left: the Jensen--Shannon critic $\log(p/q)$, $+\infty$ on the support of $p$ and $-\infty$ on that of $q$, flat on both. Middle: the $W_1$ critic, a ramp of slope one. Right: the MMD witness $\mathbb{E}_{p}[k(x,\cdot)] - \mathbb{E}_{q}[k(x,\cdot)]$, a smooth difference of kernel bumps (bandwidth half the separation, arbitrary scale). The orange arrows mark each critic's slope at $q$, the direction in which the generator moves.](../img/mdl-gan-critics-separated.svg)
:label:`fig_gan_critics_separated`

### Trained Critics from a Separated Start
:label:`subsec_gan_objectives_separated_start`

The plot and the figure describe optimal critics. A trained critic is a small network fitted to minibatches a few steps at a time, and it need not look like any of them. Does it still give a far generator a direction? Let us start a generator six units from the data and see. Its one parameter is the location $\theta$ of $q = \mathcal{N}(\theta, 1)$, and the data are $p = \mathcal{N}(0, 1)$. Every critic is the same ReLU network with two hidden layers of 32 units, and the $W_1$ critic adds the gradient penalty of WGAN-GP with weight $\gamma = 10$. The MMD arm has no critic and sets its kernel bandwidth to the median distance in the pooled batch, so that the kernel reaches across whatever gap there is.

```{.python .input #objectives-separated-start-losses}
%%tab pytorch
def mmd2_loss(x, y, bandwidths=(0.5, 1.0, 2.0)):
    """Unbiased MMD^2 under the sum of RBF kernels of the given bandwidths."""
    def kernel(a, b):
        sq_dist = torch.cdist(a, b) ** 2
        return sum(torch.exp(-sq_dist / (2 * s ** 2)) for s in bandwidths)
    return mmd2_unbiased(kernel(x, x), kernel(y, y), kernel(x, y))

def mmd2_median(x, y):
    """MMD^2 with one RBF kernel, its bandwidth the median pairwise distance
    of the pooled batch, held constant in the gradient."""
    xy = torch.cat([x, y])
    return mmd2_loss(x, y, (torch.cdist(xy, xy).median().detach(),))

#@save
def gradient_penalty(D, X, fake, u, one_sided=False):
    """Mean (|D'(x)| - 1)^2 at the points u X + (1 - u) fake (inputs 1-D);
    one-sided, only slopes steeper than one pay: max(0, |D'(x)| - 1)^2."""
    x = (u * X + (1 - u) * fake).requires_grad_()
    grad, = torch.autograd.grad(D(x).sum(), x, create_graph=True)
    excess = grad.abs() - 1
    return ((excess.clamp(min=0) if one_sided else excess) ** 2).mean()
```

```{.python .input #objectives-separated-start-losses}
%%tab jax
def mmd2_loss(x, y, bandwidths=(0.5, 1.0, 2.0)):
    """Unbiased MMD^2 under the sum of RBF kernels of the given bandwidths."""
    def kernel(a, b):
        sq_dist = ((a[:, None] - b[None]) ** 2).sum(-1)
        return sum(jnp.exp(-sq_dist / (2 * s ** 2)) for s in bandwidths)
    return mmd2_unbiased(kernel(x, x), kernel(y, y), kernel(x, y))

def mmd2_median(x, y):
    """MMD^2 with one RBF kernel, its bandwidth the median pairwise distance
    of the pooled batch, held constant in the gradient."""
    xy = jnp.concatenate([x, y])
    dist = jnp.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    return mmd2_loss(x, y, (jax.lax.stop_gradient(jnp.median(dist)),))

#@save
def gradient_penalty(D, X, fake, u, one_sided=False):
    """Mean (|D'(x)| - 1)^2 at the points u X + (1 - u) fake (inputs 1-D);
    one-sided, only slopes steeper than one pay: max(0, |D'(x)| - 1)^2."""
    grad = jax.grad(lambda x: D(x).sum())(u * X + (1 - u) * fake)
    excess = jnp.abs(grad) - 1
    return ((jnp.maximum(excess, 0) if one_sided else excess) ** 2).mean()
```

`train_shift` gives the critic five Adam steps for every generator step, which keeps it near its best response. The generator takes plain gradient steps with learning rate $0.01$: Adam would rescale every step to about its learning rate and hide the size of the gradient, which is what we want to see. The function records $\theta$ and the generator's gradient at every step. A positive `gp` adds the gradient penalty to the critic loss; with `critic_loss` set to `None`, the generator minimizes `gen_loss` of the two batches.

```{.python .input #objectives-separated-start-train}
%%tab pytorch
#@save
class Shift(nn.Module):
    """The generator x' = theta + z."""
    def __init__(self, theta0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta0))

    def forward(self, z):
        return self.theta + z

#@save
def train_shift(critic_loss, gen_loss, seed, gp=0.0, one_sided=False,
                theta0=6.0, steps=1500, n_critic=5, batch=256, lr_D=1e-2,
                lr_G=1e-2):
    """Fit x' = theta + z to N(0, 1) by plain gradient steps on theta; return
    theta0 and theta after every step, the gradient of every step, and the
    critic."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G, D = Shift(theta0), d2l.make_mlp([1, 32, 32, 1], nn.ReLU())
    opt_G = torch.optim.SGD(G.parameters(), lr_G)
    opt_D = torch.optim.Adam(D.parameters(), lr_D, betas=(0.5, 0.9))
    n = n_critic if critic_loss else 0
    thetas, grads = [theta0], []
    for _ in range(steps):
        X, Z = (to_t(rng.standard_normal((n + 1, batch, 1))) for _ in range(2))
        U = to_t(rng.uniform(size=(n, batch, 1)))   # interpolation weights
        for i in range(n):
            fake = G(Z[i]).detach()
            loss = critic_loss(D(X[i]), D(fake))
            if gp:
                loss = loss + gp * gradient_penalty(D, X[i], fake, U[i],
                                                    one_sided)
            opt_D.zero_grad()
            loss.backward()
            opt_D.step()
        fake = G(Z[-1])
        loss = (gen_loss(X[-1], fake) if critic_loss is None
                else gen_loss(D(X[-1]), D(fake)))
        opt_G.zero_grad()
        loss.backward()
        grads.append(G.theta.grad.item())
        opt_G.step()
        thetas.append(G.theta.item())
    return np.array(thetas), np.array(grads), D
```

```{.python .input #objectives-separated-start-train}
%%tab jax
#@save
class Shift(nnx.Module):
    """The generator x' = theta + z."""
    def __init__(self, theta0):
        self.theta = nnx.Param(jnp.array(theta0))

    def __call__(self, z):
        return self.theta + z

#@save
@nnx.jit(static_argnums=(0, 1, 2, 3))
def shift_step(critic_loss, gen_loss, gp, one_sided, G, D, opt_G, opt_D, X, Z,
               U):
    """One critic step per row of U, then one generator step; returns the
    generator's gradient."""
    for i in range(len(U)):
        fake = G(Z[i])
        def d_loss(D_):
            loss = critic_loss(D_(X[i]), D_(fake))
            if gp:
                loss = loss + gp * gradient_penalty(D_, X[i], fake, U[i],
                                                    one_sided)
            return loss
        opt_D.update(D, nnx.grad(d_loss)(D))
    def g_loss(G_):
        fake = G_(Z[-1])
        return (gen_loss(X[-1], fake) if critic_loss is None
                else gen_loss(D(X[-1]), D(fake)))
    grad = nnx.grad(g_loss)(G)
    opt_G.update(G, grad)
    return grad.theta[...]

#@save
def train_shift(critic_loss, gen_loss, seed, gp=0.0, one_sided=False,
                theta0=6.0, steps=1500, n_critic=5, batch=256, lr_D=1e-2,
                lr_G=1e-2):
    """Fit x' = theta + z to N(0, 1) by plain gradient steps on theta; return
    theta0 and theta after every step, the gradient of every step, and the
    critic."""
    rng = np.random.default_rng(seed)
    G = Shift(theta0)
    D = d2l.make_mlp([1, 32, 32, 1], nnx.relu, rngs=nnx.Rngs(seed))
    opt_G = nnx.Optimizer(G, optax.sgd(lr_G), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(lr_D, 0.5, 0.9), wrt=nnx.Param)
    n = n_critic if critic_loss else 0
    thetas, grads = [theta0], []
    for _ in range(steps):
        X, Z = (jnp.asarray(rng.standard_normal((n + 1, batch, 1)),
                            jnp.float32) for _ in range(2))
        U = jnp.asarray(rng.uniform(size=(n, batch, 1)), jnp.float32)
        grad = shift_step(critic_loss, gen_loss, gp, one_sided, G, D, opt_G,
                          opt_D, X, Z, U)
        grads.append(grad)                      # device scalars,
        thetas.append(G.theta[...])             # converted once below
    return np.asarray(jnp.stack(thetas)), np.asarray(jnp.stack(grads)), D
```

<!-- WRITER: M2 code done: the cell now trains SIX arms, three runs each; the sixth, "W1 + 1-sided GP", is the W1 critic with the one-sided penalty max(0, |D'| - 1)^2 (`train_shift(..., gp=10.0, one_sided=True)`). The table gained a column: "arrival" is the first step with |theta| < 0.5 (as before), "farthest" the largest |theta| from that step on. Describe the sixth arm and the new column here. -->
Five arms train three times each. Four are the non-saturating logistic and the hinge games with the generator losses of :numref:`tab_gan_loss_divergences`, a $W_1$ critic whose generator minimizes $-D(x')$, and MMD with no critic. The fifth keeps the logistic critic but gives the generator the saturating loss of :numref:`sec_basic_gan`, which minimizes $\mathbb{E}[\log \sigma(-D(x'))]$. Against an optimal critic that generator descends the value of the game, $2\,\mathrm{JS} - 2 \log 2$, so the plot above predicts its gradient at distance six: twice the Jensen--Shannon slope there, about a hundredth. For every run the table prints the median size of the generator's gradient over the first 100 steps, $\theta$ at five steps, and the first step at which $|\theta| < 0.5$.

```{.python .input #objectives-separated-start-runs}
%%tab pytorch, jax
w1 = (lambda r, f: f.mean() - r.mean(), lambda r, f: -f.mean())
arms = {'logistic, sat.': (losses['logistic'][0],
                           lambda r, f: -softplus(f).mean(), {}),
        'logistic, NS': (*losses['logistic'], {}),
        'hinge': (*losses['hinge'], {}),
        'W1 + GP': (*w1, dict(gp=10.0)),
        'W1 + 1-sided GP': (*w1, dict(gp=10.0, one_sided=True)),
        'MMD, median': (None, mmd2_median, {})}
steps, w1_runs = (0, 100, 300, 1000, 1500), []  # (arm, seed, theta, critic)
print(f'{"":17s}|grad|   theta at steps {", ".join(map(str, steps))}'
      '  arrival  farthest')
for name, (critic_loss, gen_loss, kwargs) in arms.items():
    for seed in range(3):
        thetas, grads, D = train_shift(critic_loss, gen_loss, seed, **kwargs)
        if name.startswith('W1'):
            w1_runs.append((name, seed, thetas[-1], D))
        hit = np.flatnonzero(np.abs(thetas) < 0.5)  # arrival: hit[0]
        print(f'{name:16s}{seed}{np.median(np.abs(grads[:100])):7.3f} '
              + ''.join(f'{thetas[t]:7.2f}' for t in steps)
              + (f'{hit[0]:9d}{np.abs(thetas[hit[0]:]).max():10.2f}'
                 if len(hit) else '  not reached'))
```

The prediction holds. The saturating arm's gradient is between $0.006$ and $0.015$, about a hundredth, and in 1,500 steps $\theta$ moves from $6$ only to about $5.76$. The non-saturating arm trains the same critic with the same loss and changes only the generator's loss, yet its gradient is $2.7$ to $3.0$, a factor of hundreds larger, and every run reaches $|\theta| < 0.5$ by step 390. The critic was never the problem. For these Gaussians the optimal logistic critic $\lambda$ is a line of slope $-\theta$, and the trained critic slopes as well. The saturating loss multiplies that slope by $\sigma(D(x'))$, the critic's probability that a generated sample is real (:numref:`sec_basic_gan`), which six units from the data is negligible. The non-saturating loss passes the slope on almost unweighted. The saturation belongs to the generator's loss.

The hinge is the case that the analysis of optimal critics gets wrong. Its optimal critic $\operatorname{sign} \lambda$ is level at both clouds, so a generator that minimizes $-D(x')$ against it would not move at all. For that generator loss the gradient is simply the critic's average slope at the generated samples, so the table measures the trained critic directly: its slope there is $1.1$ to $1.7$, and every run reaches the data between steps 719 and 918. The trained network ramps across the gap instead of building the step. A small network fitted to minibatches a few steps at a time is smoother than the optimal critic, and here that smoothness, not the divergence on the label, moves the generator.

<!-- WRITER: measured, both tabs (PyTorch reproduces the store bit for bit; JAX now deterministic, two runs identical): two-sided W1 + GP gradient 1.25-1.26, arrival steps 477-508, farthest after arrival 1.13-7.61 in all six runs (PyTorch 1.48, 7.49, 7.61; JAX 1.13, 7.47, 1.22), theta at 1,500 between 0.30 and 7.61 in size. One-sided arm: gradient 1.26-1.27 (1.257-1.265), arrival 476-479, farthest 0.49-0.50 (never farther than where it entered), |theta| at 1,500 at most 0.06 in all six runs. The firm lesson (number, range, difference far beyond the range): the two-sided arm overshoots by more than a unit in every run, the one-sided arm stays within half a unit of the data from its arrival on and ends within 0.06 of it. Replace "arriving is the easy part" and the per-run narrative by this, with the reading rule of M2 option 2. "between steps 477 and 508" -> "near step 500" (S1). -->
The $W_1$ gradient is $1.25$ to $1.26$ in every run rather than $1$, because the penalty is soft. A linear critic of slope $a$, with the two means a distance $d$ apart, pays $-ad + 10(a - 1)^2$ under the penalized loss, which is smallest at $a = 1 + d/20$, between $1.25$ and $1.3$ while $d$ is between five and six. This push shrinks only toward one, so the generator reaches the data at nearly full speed, between steps 477 and 508, and nothing in the push tells it to stop there.

Past the data the critic has to reverse its slope, and this is where the two-sided penalty bites. It charges $\gamma$ for a flat critic, so the reversal is a climb over a barrier at slope zero. With the generator a distance $d$ beyond the data, a linear critic still on the old side sits in a local minimum of the penalized loss, at slope $1 - d/20$ pointing away from the data, and from there it pushes the generator further out. A critic that gets over the barrier settles at $1 + d/20$ on the new side and pulls the generator back. A ReLU critic has a third possibility that a linear one lacks: it can go flat, and a flat critic leaves the generator wherever it happens to be. Which of these did the runs meet? The critic's slope at the generator tells us, since the generator minimizes $-D(x')$ and each of its steps moves $\theta$ by about the learning rate times $D'(\theta)$. The cell reads the three $W_1$ critics at step 1,500: their slope at the final $\theta$ and at the data mean $0$, and how many of their 64 hidden units are active at each point.

```{.python .input #objectives-separated-start-w1-critics}
%%tab pytorch
def slope_and_units(D, x):
    """D'(x) and, per hidden layer, the number of ReLU units active at x."""
    x = h = torch.tensor([[x]], dtype=torch.float32, requires_grad=True)
    active = []
    for layer in D:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            active.append(int((h > 0).sum()))
    slope, = torch.autograd.grad(h.sum(), x)
    return slope.item(), active

print(f"{'':19s}theta D'(theta)  active    D'(0)  active"
      '  (of 32 units per layer)')
for name, seed, theta, D in w1_runs:
    (s_t, a_t), (s_0, a_0) = (slope_and_units(D, x) for x in (theta, 0.0))
    print(f'{name:16s}{seed}{theta:7.2f}{s_t:10.2f}{a_t[0]:5d}{a_t[1]:3d}'
          f'{s_0:9.2f}{a_0[0]:5d}{a_0[1]:3d}')
```

```{.python .input #objectives-separated-start-w1-critics}
%%tab jax
def slope_and_units(D, x):
    """D'(x) and, per hidden layer, the number of ReLU units active at x."""
    def forward(x):
        h, active = x, []
        for layer in D.layers:
            h = layer(h)
            if not isinstance(layer, nnx.Linear):  # an activation
                active.append((h > 0).sum())
        return h.sum(), active
    slope, active = jax.grad(forward, has_aux=True)(jnp.array([[x]]))
    return float(slope.squeeze()), [int(a) for a in active]

print(f"{'':19s}theta D'(theta)  active    D'(0)  active"
      '  (of 32 units per layer)')
for name, seed, theta, D in w1_runs:
    (s_t, a_t), (s_0, a_0) = (slope_and_units(D, x) for x in (theta, 0.0))
    print(f'{name:16s}{seed}{theta:7.2f}{s_t:10.2f}{a_t[0]:5d}{a_t[1]:3d}'
          f'{s_0:9.2f}{a_0[0]:5d}{a_0[1]:3d}')
```

<!-- WRITER: M1 + M2 code done: the cell now prints all six W1 critics (both arms) with active units PER HIDDEN LAYER (two numbers of 32 each, at theta and at 0). Measured: every flat two-sided critic (slope 0.00 at both points) has 0 of 32 units active in the SECOND layer (PyTorch run 0; JAX runs 0 and 2), so its output is a constant and no gradient reaches the weights: dead, as Fable says. The sentence "The flat critics are not dead networks, though: 10 to 28 of their 64 units are active ..." is FALSE; rewrite (Fable's M1 wording works). The other two-sided critics are alive (layer 2: 7-9 units PyTorch, 2 JAX). One-sided critics at the data: slope -0.16 to 0.65, layer 2 has 3-7 units on, except JAX run 0, also dead (0 of 32 in layer 2) but with its generator at theta = -0.06, inside the data; so a ReLU critic can die under either penalty, and only the two-sided one first carries the generator past the data. States now: PyTorch flat (0.30) / crossed (+1.34 at -7.18) / old side (-0.65 at -7.61); JAX flat (-1.13) / crossed (+1.33 at theta = -7.30, +1.44 at 0) / flat (-1.22). "The run that ends at theta = -7.6 has a critic still on the old side" is now true in PyTorch only: per M2 option 2, give a reading rule and quote no per-run theta, slope or count. The setup sentence before the cell ("reads the three W1 critics ... how many of their 64 hidden units") must follow: six critics, units per layer. -->
The linear analysis reads this printout well. Each critic has nearly the same slope at $\theta$ as at $0$, so across the gap it is close to a straight line, and the runs land in the states that line allows. Some critics are flat, with slope $0.00$ at both points, and their generators have not moved since step 1,000. The run that ends at $\theta = -7.6$ has a critic still on the old side. Its slope is $-0.62$ to $-0.65$; with the data to the right of $\theta$, a negative slope carries $\theta$ further left, and the predicted size at $d = 7.6$ is $1 - d/20 = 0.62$. That generator is being pushed out. The flat critics are not dead networks, though: 10 to 28 of their 64 units are active, and the units that stay on simply pass no slope from input to output.

<!-- WRITER: M2: delete both tab notes (option 2). The JAX note is false after the change: JAX now has two flat critics at theta = -1.13 and -1.22 and one crossed critic at -7.30. -->
:begin_tab:`pytorch`
The run that ends at $\theta = -7.18$ shows the third state: its critic has crossed the barrier, with slope $+1.34$ against the predicted $1 + d/20 = 1.36$, and it now pulls the generator back toward the data. The flat critic holds its generator at $\theta = 0.30$, inside the data.
:end_tab:

:begin_tab:`jax`
Two of the three critics are flat, and they hold their generators at $\theta = -1.18$ and $-1.40$, a little more than a unit past the data.
:end_tab:

<!-- WRITER: M2: replace this sentence by what the table shows (see the note above the W1 paragraph); keep the hand-off to sec_gan_convergence. -->
Under the two-sided penalty, arriving at the data is the easy part; staying there is not. :numref:`sec_gan_convergence` shows on a two-parameter game that a penalty centered at slope one keeps pushing a generator that already matches the data, and that a penalty centered at zero does not.

MMD moves steadily, from $6$ to $1.32$ in 1,500 steps, with a gradient of $0.21$ that is the same in every run. Its speed says nothing about MMD against the critics, because the scale of its gradient is set by the bandwidth and by the scale of $\mathrm{MMD}^2$.

So is the missing gradient a property of the loss, of the critic, or of the divergence? The divergence on the label does go flat, but only the saturating generator descends it, and that generator stalls. Every trained critic sloped at the generated samples, the hinge's included, although its optimal version is a level step there. A far generator moves when its critic slopes at the generated samples and its loss passes that slope on; the divergence on the label decides neither.

The systems in use fit this reading. Studies that give each loss the same tuning budget find the losses comparable in final image quality :cite:`Lucic.Kurach.Michalski.ea.2018`, and practice has settled on combinations rather than on a winning divergence. Large image GANs of the BigGAN generation train the hinge loss with a spectrally normalized critic :cite:`Miyato.Kataoka.Koyama.ea.2018,Brock.Donahue.Simonyan.2019`, the mixed cell of :numref:`tab_gan_objective_families`. HiFi-GAN vocoders train least squares :cite:`Kong.Kim.Bae.2020`. The StyleGAN line trains the non-saturating logistic loss with a zero-centered gradient penalty on real samples :cite:`Karras.Laine.Aittala.ea.2020`, the combination that :numref:`sec_gan_convergence` analyzes. All three give the generator a loss that does not saturate.

### Five Losses on a Gaussian Mixture
:label:`subsec_gan_objectives_testbed`

Near the data the question changes. Every divergence in :numref:`tab_gan_loss_divergences` vanishes at $q = p$, so with overlapping supports every loss should drive a generator to the data; what differs is that the hinge critic reports only the sign of the remaining mismatch, the other critics also its size. We fit one generator to the three-mode mixture under five objectives: the non-saturating logistic loss, least squares, hinge, the squared hinge, and MMD with a fixed kernel and no critic. The squared hinge differs from the hinge only in its link, so comparing the two isolates the link. All five runs share the architecture, learning rates, step counts and start, tuned for none of them, so the runs can show whether each loss fits the mixture but not which loss is best.

The generator draws one of three components uniformly and applies that component's learned affine map to Gaussian noise, so $q$ stays a three-component mixture whose divergence from $p$ we can compute after every step. We set the start by hand: each mean lies one to two and a half units from a different mode, and each map is $0.4$ times the identity, below the data's $0.7$, so positions and scales remain to be learned.

```{.python .input #objectives-five-losses-on-a-gaussian-mixture-1}
%%tab pytorch, jax
#@save
mu_0 = np.array([[1.0, -1.0], [-4.3, -2.0], [-1.0, 3.8]])  # off the modes
A_0 = np.stack([0.4 * np.eye(2)] * 3)      # below the data's scale of 0.7

print(f'JS(p, q) at the start: {js_mixtures(mu_0, A_0):.3f} nats')
```

The start is about half a nat from the data. The MMD arm minimizes the unbiased minibatch $\mathrm{MMD}^2$ under the sum of three RBF kernels with bandwidths $0.5$, $1$ and $2$, a kernel sensitive at several scales.

```{.python .input #objectives-one-testbed-four-losses-2}
%%tab pytorch
#@save
class MixtureGenerator(nn.Module):
    """x = mu_k + z A_k for a uniformly drawn component k."""
    def __init__(self, mu, A):
        super().__init__()
        self.mu = nn.Parameter(to_t(mu))
        self.A = nn.Parameter(to_t(A))

    def forward(self, z, k):
        return torch.einsum('ni,nij->nj', z, self.A[k]) + self.mu[k]

    def params_np(self):
        return (self.mu.detach().numpy().copy(),
                self.A.detach().numpy().copy())

losses['MMD'] = (None, mmd2_loss)   # no critic: MMD^2 to the real batch
```

```{.python .input #objectives-one-testbed-four-losses-2}
%%tab jax
#@save
class MixtureGenerator(nnx.Module):
    """x = mu_k + z A_k for a uniformly drawn component k."""
    def __init__(self, mu, A):
        self.mu = nnx.Param(jnp.asarray(mu, jnp.float32))
        self.A = nnx.Param(jnp.asarray(A, jnp.float32))

    def __call__(self, z, k):
        return jnp.einsum('ni,nij->nj', z, self.A[k]) + self.mu[k]

    def params_np(self):
        return np.asarray(self.mu[...]), np.asarray(self.A[...])

losses['MMD'] = (None, mmd2_loss)   # no critic: MMD^2 to the real batch
```

`train_toy` takes the two losses as arguments and alternates one critic step with one generator step. The MMD arm, stored with the critic loss `None`, takes only the generator step, on `mmd2_loss(X, fake)` instead of the critic's scores. Every 200 steps it records the exact $\mathrm{JS}(p, q_t)$.

```{.python .input #objectives-one-testbed-four-losses-4}
%%tab pytorch
#@save
def train_toy(critic_loss, gen_loss, seed=0, steps=6000, batch=256, lr=1e-3):
    """Fit a MixtureGenerator from (mu_0, A_0) to the data mixture, one
    critic step per generator step. The losses act on the critic's scores of
    the real and generated batch; with critic_loss None, gen_loss acts on the
    two batches. Returns the exact JS(p, q_t) every 200 steps."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G, D = MixtureGenerator(mu_0, A_0), make_critic()
    opt_G = torch.optim.Adam(G.parameters(), lr)
    opt_D = torch.optim.Adam(D.parameters(), 2 * lr)
    trace = []
    for t in range(steps):
        X = to_t(sample_mixture(batch, mu_p, A_p, rng))
        z = to_t(rng.standard_normal((batch, 2)))
        k = torch.tensor(rng.integers(0, 3, batch))
        if critic_loss is not None:
            opt_D.zero_grad()
            critic_loss(D(X), D(G(z, k).detach())).backward()
            opt_D.step()
        opt_G.zero_grad()
        fake = G(z, k)
        loss = (gen_loss(X, fake) if critic_loss is None
                else gen_loss(D(X), D(fake)))
        loss.backward()
        opt_G.step()
        if t % 200 == 0:
            trace.append(js_mixtures(*G.params_np()))
    return np.array(trace)
```

```{.python .input #objectives-one-testbed-four-losses-4}
%%tab jax
#@save
@nnx.jit(static_argnums=(0, 1))
def train_toy_step(critic_loss, gen_loss, G, D, opt_G, opt_D, X, z, k):
    """A critic step (unless critic_loss is None), then a generator step."""
    if critic_loss is not None:            # the generator is fixed here
        fake = G(z, k)
        opt_D.update(D, nnx.grad(lambda D_: critic_loss(D_(X), D_(fake)))(D))
    def g_loss(G_):
        fake = G_(z, k)
        return (gen_loss(X, fake) if critic_loss is None
                else gen_loss(D(X), D(fake)))
    opt_G.update(G, nnx.grad(g_loss)(G))

#@save
def train_toy(critic_loss, gen_loss, seed=0, steps=6000, batch=256, lr=1e-3):
    """Fit a MixtureGenerator from (mu_0, A_0) to the data mixture, one
    critic step per generator step. The losses act on the critic's scores of
    the real and generated batch; with critic_loss None, gen_loss acts on the
    two batches. Returns the exact JS(p, q_t) every 200 steps."""
    rng = np.random.default_rng(seed)
    G, D = MixtureGenerator(mu_0, A_0), make_critic(nnx.Rngs(seed))
    opt_G = nnx.Optimizer(G, optax.adam(lr), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(2 * lr), wrt=nnx.Param)
    trace = []
    for t in range(steps):
        X = jnp.asarray(sample_mixture(batch, mu_p, A_p, rng), jnp.float32)
        z = jnp.asarray(rng.standard_normal((batch, 2)), jnp.float32)
        k = jnp.asarray(rng.integers(0, 3, batch))
        train_toy_step(critic_loss, gen_loss, G, D, opt_G, opt_D, X, z, k)
        if t % 200 == 0:
            trace.append(js_mixtures(*G.params_np()))
    return np.array(trace)
```

For each of three runs per loss the cell reports the median divergence over the last 2,000 steps, where it fluctuates, and the first recorded steps below $0.1$ and $0.01$ nats. The plot shows the first run of each loss.

```{.python .input #objectives-one-testbed-four-losses-6}
%%tab pytorch, jax
def first_below(trace, level):
    """First recorded step (one record per 200 steps) with JS below level."""
    hits = np.flatnonzero(trace < level)
    return f'{200 * hits[0]:6d}' if len(hits) else '   n/a'

runs = {name: [train_toy(*pair, seed=seed) for seed in range(3)]
        for name, pair in losses.items()}
print(f'{"three runs:":14s}{"median JS, last 2,000 steps":>27s}'
      f'{"first JS < 0.1":>21s}{"first JS < 0.01":>21s}')
for name, traces in runs.items():
    medians = [np.median(trace[-2000 // 200:]) for trace in traces]
    print(f'{name:14s}   ' + ''.join(f'{m:8.4f}' for m in medians)
          + '   ' + ''.join(first_below(trace, 0.1) for trace in traces)
          + '   ' + ''.join(first_below(trace, 0.01) for trace in traces))
d2l.plot(np.arange(len(runs['logistic'][0])) * 200,
         [traces[0] for traces in runs.values()], xlabel='step',
         ylabel='JS(p, q_t)', yscale='log', legend=list(runs), figsize=(5, 3))
```

All five objectives fit the mixture, taking the generator from about half a nat to a few thousandths of a nat or less. MMD, which has no critic to keep up with, ends lowest in every run, at $0.0001$. It is also the first below $0.1$ nats, at step 1,200, and the last below $0.01$, at step 2,600. The hinge ends highest in every run, between $0.003$ and $0.007$ nats, against at most $0.0024$ for every other loss, and the squared hinge, with the same margin and generator loss, ends at the level of the logistic and least-squares games. The hinge fit is thus the loosest in these runs. That is not a verdict on the hinge loss: three untuned runs per loss and a margin of a few thousandths of a nat on a planar toy cannot rank losses that practice relies on.

<!-- WRITER: the JAX objectives notebook now runs with deterministic GPU ops (new setup cell), which changes the JAX mixture rows: median JS last 2,000 steps logistic 0.0011/0.0008/0.0019, least squares 0.0010/0.0007/0.0006, hinge 0.0069/0.0036/0.0064, squared hinge 0.0012/0.0007/0.0009, MMD 0.0001; first JS < 0.1: logistic 2200/2200/1800, LS 2200/2000/1800, hinge 1400/1400/1800, squared hinge 1400/1400/1200, MMD 1200; first < 0.01: logistic 2400 x3, LS 2400/2200/2000, hinge 2000/2200/2000, squared hinge 2000/2000/2400, MMD 2600. "the squared hinge at step 1,400 ... in every run" is now false in JAX (one run 1,200); the previous paragraph's hinge range 0.003-0.007 and "at most 0.0024 for every other loss" still hold. This paragraph is an M4 cut (one sentence + exercise pointer) anyway. PyTorch rows unchanged. -->
The gap between the two hinges is consistent with what their critics must track. The hinge target $\operatorname{sign} \lambda$ is a full-margin step however small the mismatch, so a critic that falls behind leaves the generator following a stale push of full size. The squared hinge's target $\tanh(\lambda/2)$ shrinks with the mismatch. The logistic and least-squares runs, whose generators weight each sample by how firmly the critic still rejects it, also leave the starting plateau later than the squared hinge and MMD: they fall below $0.1$ nats between steps 1,800 and 2,200, the squared hinge at step 1,400 and MMD at step 1,200 in every run. These runs test neither account; Exercise :exref:`ex_objectives-plateaus-lag` does.

Near the data, then, every loss here fits the mixture, and far from it the slope that reaches the generator decides. All the critics of this section score a sample on its own. A critic can also be shown a real and a generated sample together and asked which is which, and what such a critic measures is the subject of the next section.

## Summary

Every adversarial objective with a single-sample critic is a supremum, over a critic class, of expected payoffs, so it is convex in the generator's distribution, though not in its parameters. With a free critic, a classification loss evaluates the f-divergence given by its Bayes-risk gap, and its optimal critic reports a fixed link of the log ratio $\lambda$; tangent lines extend the construction to any f-divergence. Linear payoffs with a restricted critic give integral probability metrics instead: MMD for a kernel ball and, by the Kantorovich--Rubinstein duality, $W_1$ for the Lipschitz ball, which practical critics enforce only approximately. When the supports separate, the ratio-based divergences go flat while $W_1$ keeps unit slope and MMD a slope over separations set by its bandwidth. The divergence does not decide whether a far generator moves, though: every trained critic sloped at the generated samples, the hinge's included, and only the saturating generator loss threw that slope away. Under the two-sided gradient penalty the $W_1$ push does not fade at the data, and the generator need not settle there. With overlapping supports, all five losses fit a three-mode mixture to within a few thousandths of a nat.

## Exercises

1. [code] **Convexity and parameterization.** Every objective of the
   form :eqref:`eq_gan_template` is convex in $q$, but a generator moves
   $q$ through its parameters. Use `normal_pdf`, `js_1d` and `xs`.
    1. Let $p = \mathcal{N}(0, 1)$ and $q_0 = \mathcal{N}(8, 1)$. Check with
       second differences that $\mathrm{JS}(p, q_t)$ is convex in $t$ along
       the mixture path $q_t = (1 - t)\, q_0 + t\, p$.
    1. Now let $p = 0.7\, \mathcal{N}(-4, 1) + 0.3\, \mathcal{N}(4, 1)$ and
       $q_\theta = \mathcal{N}(\theta, 1)$. Plot
       $\mathrm{JS}(p, q_\theta)$ for $\theta \in [-6, 6]$ and locate its
       local minima.
    1. Why do these minima not contradict the first part? Which property
       of the set $\{q_\theta\}$ does the convexity argument need?
   :exlabel:`ex_objectives-convexity-parameterization`
1. **Extending the table.** The exponential loss pays $e^{-t}$ at
   margin $t$, and the zero--one loss pays one for a negative margin. Show
   that their Bayes risks :eqref:`eq_gan_bayes_risk` are
   $2\sqrt{\eta(1 - \eta)}$ and $\min(\eta, 1 - \eta)$, that their gaps
   are the squared Hellinger distance $1 - \int \sqrt{pq}$ and
   $\tfrac12\, \mathrm{TV}(p, q)$, and that the optimal exponential critic
   is $\lambda/2$. Evaluate both gaps at the coin pair.
   :exlabel:`ex_objectives-verify-numerically-that-lsgan`
1. **Jensen--Shannon conjugate.** Twice the Jensen--Shannon divergence has
   the divergence function
   $f(u) = u \log u - (u + 1) \log \frac{u + 1}{2}$.
    1. Derive $f^*(t) = -\log(2 - e^t)$, finite exactly for $t < \log 2$.
    1. Show that under $T = \log(2 \sigma(D))$ the optimal critic
       $T^\star = f'(\rho)$ corresponds to $D^\star = \lambda$, and that
       the bound :eqref:`eq_gan_fgan_bound` becomes $2 \log 2 + V(D)$ with
       the value function $V$ of :numref:`sec_basic_gan`.
    1. The activation $\log 2 - \mathrm{softplus}$ covers exactly the
       slopes $f'(u)$, and $-\mathrm{softplus}$ also maps into the domain of
       $f^*$. Which values of $T^\star$ does the second exclude, and what
       does it do to the bound whenever $p \neq q$?
   :exlabel:`ex_objectives-variational-bound-eqref-eq`
1. [code] **MMD gradient range.** For the three-bandwidth kernel of
   `mmd2_loss`, plot the slope of $\mathrm{MMD}^2$ between the unit
   Gaussians of :eqref:`eq_gan_mmd_gauss` for $\mu \in [0, 20]$, and find
   the separation beyond which it stays below 1% of its maximum. Compare
   with the single kernel $s = 4$. Which bandwidth decides the range?
   :exlabel:`ex_objectives-mmd-gradient-range`
1. [code] **The penalized $W_1$ critic.** ● Use `train_shift` and the $W_1$
   arm of :numref:`subsec_gan_objectives_separated_start`.
    1. For a linear critic the text predicts a push of $1 + d/(2\gamma)$
       at distance $d$. Predict the generator's median gradient over the
       first 100 steps for $\gamma \in \{1, 10, 100\}$, and measure it.
    <!-- WRITER: M2 option 4: the one-sided penalty is now in the body (`gradient_penalty(..., one_sided=True)`, `train_shift(..., one_sided=True)`); keep only the leaky-ReLU part here and ask why a generator may move away from the data from the first step (linear analysis: at d = 6 an old-side critic sits at 1 - 6/20 = 0.7 pointing away). -->
    1. Replace the two-sided penalty in `gradient_penalty` by the
       one-sided $\max(0, |T'| - 1)^2$, which has no barrier at slope zero.
       Separately, give the critic leaky-ReLU activations. Over three seeds,
       does either change let the generator settle at the data? Which
       part of the account in the text does each outcome support?
   :exlabel:`ex_objectives-wasserstein-penalty`
1. [code] **Plateaus and critic lag.** ● Use `train_toy` and the five
   losses of :numref:`subsec_gan_objectives_testbed`.
    1. Pair the logistic critic loss with the generator loss $-D$ and
       compare it with the non-saturating entry over three seeds by the
       first step at which $\mathrm{JS}(p, q_t) < 0.1$ nats. Explain the
       difference from the per-sample weights $\sigma(-D(x'))$ and $1$
       that the two generator losses give (:numref:`sec_basic_gan`). Does
       the least-squares game behave like the logistic one?
    1. The hinge ends with the loosest fit in every run, and the text
       offers critic lag as a possible cause. Test it with two controls
       over three seeds: give the critic five steps per generator step, or
       quadruple the critic's batch. Does either control lower the hinge's
       median JS by more than the spread of its runs, and does the outcome
       point to lag or to minibatch noise?
   :exlabel:`ex_objectives-plateaus-lag`
1. [code] **Estimation and dimension.** ● Let
   $p = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_x})$ and
   $q = \mathcal{N}(\boldsymbol{\mu}, \mathbf{I}_{d_x})$ with
   $\|\boldsymbol{\mu}\| = 1$, so that $W_1(p, q) = 1$ in every dimension.
    1. Estimate $W_1$ from two samples of size $n$ by an optimal
       assignment (`scipy.optimize.linear_sum_assignment`) for
       $d_x \in \{1, 2, 10, 50\}$ and $n \in \{100, 400, 1600\}$, and plot
       the relative error against $n$ on log--log axes. Repeat with the
       unbiased $\mathrm{MMD}^2$ estimate under an RBF kernel of bandwidth
       $\sqrt{d_x}$, whose population value is
       $2 (s^2/(s^2 + 2))^{d_x/2} (1 - e^{-1/(2(s^2 + 2))})$, and compare
       the slopes with the quoted rates.
    1. Measure the empirical $W_1$ between two samples of size 256 from
       $p$ itself in dimension 50. What does it say about a minibatch
       critic meant to resolve $W_1(p, q) = 1$?
    1. Evaluate the bound :eqref:`eq_gan_fgan_bound` of
       `critics['forward KL']` by quadrature on `grid_2d`, and compare it
       with $D_{\textrm{KL}}(p \,\|\, q)$, which quadrature puts at about
       $0.16$ nats. Then evaluate the bound on minibatches of 512. Why can
       a minibatch estimate flatter a critic that overshoots $1 + \lambda$
       where $q$ rarely samples?
   :exlabel:`ex_objectives-estimation-and-dimension`
1. **Divergences a loss can reach.** ●● Every Bayes risk satisfies
   $L(\eta) = L(1 - \eta)$, because :eqref:`eq_gan_margin` scores real and
   generated samples with one function applied to $D$ and $-D$, so every
   gap is a symmetric f-divergence. Corollary 3 of
   :citet:`Nguyen.Wainwright.Jordan.2009` shows the converse, and this
   problem builds a nonnegative loss for it. Let $f$ be
   convex on $(0, \infty)$ with $f(1) = 0$, $f(u) = u f(1/u)$, and
   $f(0) = \lim_{u \to 0} f(u) < \infty$, and define

   $$
   L(\eta) = 2 f(0) - 2 (1 - \eta)\, f\!\left( \frac{\eta}{1 - \eta} \right), \qquad 0 < \eta < 1.
   $$

    1. Show that $L$ is concave, symmetric about $\tfrac12$, tends to $0$
       at both ends, and has Bayes-risk gap $D_f(p \,\|\, q)$.
    1. For differentiable $L$ and $\hat\eta = \sigma(t)$, let a real sample
       cost $c(t) = L(\hat\eta) + (1 - \hat\eta) L'(\hat\eta)$. Show that a
       generated sample then costs $L(\hat\eta) - \hat\eta L'(\hat\eta)$,
       that the conditional risk is minimized at $\hat\eta = \eta$ with
       value $L(\eta)$, and that $c \geq 0$.
    1. Apply the construction to the Jensen--Shannon divergence function
       and show that it returns the logistic loss.
   :exlabel:`ex_objectives-divergences-a-loss-can-reach`

[Discussions](https://d2l.discourse.group/t/36809)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.2]{.kicker}

Adversarial Objectives and Divergences<br>
**the loss menu · every classification loss is a divergence · restricting the critic · which critics move a far generator**
:::
:::

::: {.slide title="GAN Codebases Offer a Menu of Losses"}
[The problem]{.kicker}

`non-saturating | least squares | hinge | Wasserstein`: practitioners
switch entries like optimizers.

- The log-loss game measures Jensen--Shannon (JS); its critic learns
  $\lambda = \log(p/q)$.
- $\mathcal{N}(0, 1)$ against $\mathcal{N}(7, 1)$: JS is within 0.1% of
  its ceiling $\log 2$, and its slope is 0.0013.

::: {.d2l-note}
What do the other entries compute, and which of them still move a far
generator?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the divergence of any classification loss from its Bayes risk;
- read off what its optimal critic reports about $p/q$ (the *link*);
- build a game for any f-divergence from tangent lines;
- say what clipping, gradient penalties and spectral normalization enforce;
- predict which trained critics move a generator far from the data.

**Running examples:** two unit Gaussians on the line; a three-mode
Gaussian mixture in the plane with known densities.
:::

::: {.slide title="Two Knobs on One Game"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Generative Adversarial Networks | the log-loss game, $D^\star = \lambda$ |
| **Adversarial Objectives** | **other losses, restricted critics** |
| Relativistic Objectives | critics that rank pairs |
| Gradient Penalties and Convergence | dynamics and penalties |
:::

::: {.col .narrow}
- **Uses:** the log-loss game, JS, KL.
- **Skips:** architectures, image experiments.
:::
:::
:::

::: {.slide title="Every Objective Chooses Payoffs and a Critic Class"}
$$d(p, q) = \sup_{T \in \mathcal{T}} \big\{ \mathbb{E}_{p}[\ell_p(T(x))] + \mathbb{E}_{q}[\ell_q(T(x'))] \big\}$$

Payoffs $\ell_p$ (real), $\ell_q$ (generated); the critic $T$ maximizes.

| | free critic | restricted critic |
|:--|:--|:--|
| **one sample** | classification losses: f-divergences | linear payoffs: MMD, $W_1$; hinge + spectral norm |
| **a pair** | pairing loss (next deck) | pairing + zero-centered penalties |

::: {.d2l-note}
Fixed critic: linear in $(p, q)$. So convex in $q$, not in the network's
parameters.
:::
:::

::: {.slide title="Looking at the Sample Saves Loss: Two Outcomes"}
A coin: $p = (\tfrac34, \tfrac14)$, $q = (\tfrac14, \tfrac34)$ on heads and
tails; each source with probability $\tfrac12$. Seeing heads, the posterior of
"real" is $\eta = \tfrac34$.

| loss | blind critic | best critic | saving |
|:--|:--|:--|:--|
| log loss | $\log 2 = 0.693$ | $H_b(\tfrac34) = 0.562$ | $0.131 = \mathrm{JS}$ |
| squared error | $\tfrac14$ | $\tfrac{3}{16}$ | $\tfrac{1}{16}$ |
| hinge | $1$ | $\tfrac12$ | $\tfrac12 = \mathrm{TV}$ |

::: {.d2l-note}
One posterior, three numbers; each is zero when $p = q$.
:::
:::

::: {.slide title="Every Classification Loss Measures an f-Divergence"}
Bayes risk $L(\eta)$: the smallest expected loss at posterior
$\eta = p/(p+q)$. The **Bayes-risk gap** is what looking saves:

$$\Delta_\ell(p, q) = L(\tfrac12) - \mathbb{E}_{x \sim m}\big[L(\eta(x))\big], \quad m = \tfrac{p+q}{2}$$

**Proposition** ($L$ finite on $[0,1]$). $\Delta_\ell = \mathbb{E}_q[f(p/q)]$ with
$f(u) = L(\tfrac12) - \tfrac{u+1}{2} L\big(\tfrac{u}{u+1}\big)$, convex,
$f(1) = 0$.

Proof: substitute $u = p/q$; $L$ is concave, so $f$ is convex.
:::

::: {.slide title="Each Loss Picks a Divergence and a Link"}
Margin $t$: $D(x)$ on real, $-D(x')$ on generated samples.

| loss | loss at $t$ | gap $\Delta_\ell$ | link | coin |
|:---|:---|:---|:---|:--|
| logistic | $\log(1+e^{-t})$ | $\mathrm{JS}$ | $\lambda$ | $0.131$ |
| square | $\tfrac14(1-t)^2$ | $\tfrac18 \int \frac{(p-q)^2}{p+q}$ | $\sigma(\lambda)$ | $\tfrac1{16}$ |
| sq. hinge | $\max(0,1-t)^2$ | $\tfrac12 \int \frac{(p-q)^2}{p+q}$ | $\tanh\frac{\lambda}{2}$ | $\tfrac14$ |
| hinge | $\max(0,1-t)$ | $\mathrm{TV}$ | $\operatorname{sign}\lambda$ | $\tfrac12$ |

::: {.d2l-note}
Same posterior, different **links**: the hinge critic keeps only the sign.
:::
:::

::: {.slide title="Check: Two Divergences No Table Row Gives"}
[Check]{.kicker}

1. The zero--one loss has Bayes risk $\min(\eta, 1-\eta)$. Which gap?
2. Why can no nonnegative loss give the forward KL $D_{\textrm{KL}}(p\|q)$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) $\tfrac12\mathrm{TV}$, half the hinge's. (2) A nonnegative
loss has $L \geq 0$, so $\Delta_\ell \leq L(\tfrac12)$ is bounded; forward
KL is not.
:::
:::
:::

::: {.slide title="Tangent Lines Build a Game for Any Divergence"}
- The highest line of slope $t$ below a convex $f$ has intercept
  $-f^*(t)$, so $f(u) \geq ut - f^*(t)$.
- Put $u = p(x)/q(x)$, $t = T(x)$, average under $q$:

$$D_f(p \,\|\, q) \;\geq\; \mathbb{E}_p[T] - \mathbb{E}_q[f^*(T)]
\quad \text{for every critic } T$$

- Tight at $T^\star = f'(p/q)$. Forward KL: $f^*(t) = e^{t-1}$,
  $T^\star = 1 + \lambda$.

::: {.d2l-note}
An imperfect critic only **underestimates** the divergence.
:::
:::

::: {.slide title="Trained Critics Recover Their Links"}
Fresh critics against a fixed $q$ (JS 0.033 nats) with known densities:

@!objectives-critic-links-1

Identity, sigmoid, step, $1 + \lambda$; all drift where $|\lambda|$ is large
and samples are rare.
:::

::: {.slide title="Linear Payoffs Need a Restricted Critic"}
$$d_{\mathcal{F}}(p, q) = \sup_{T \in \mathcal{F}} \big\{ \mathbb{E}_p[T] - \mathbb{E}_q[T] \big\}$$

A free $T$ makes this infinite whenever $p \neq q$.

- **Kernel ball, MMD** (similarity $k$):
  $\mathrm{MMD}^2 = \mathbb{E}\,k(x, \tilde x) + \mathbb{E}\,k(x', \tilde x') - 2\,\mathbb{E}\,k(x, x')$;
  witness $w(x) = \mathbb{E}_p k(x,\cdot) - \mathbb{E}_q k(x,\cdot)$.
- **1-Lipschitz ball, $W_1$:** by Kantorovich--Rubinstein (finite first
  moments), the least average distance to move $p$ onto $q$.
:::

::: {.slide title="No Practical Critic Is Exactly 1-Lipschitz"}
| device | what it enforces |
|:--|:--|
| weight clipping | some unknown $\kappa$: estimates at most $\kappa W_1$ |
| gradient penalty | slope pulled toward 1, softly, at sampled points |
| spectral normalization | every layer about 1-Lipschitz: sufficient, not necessary |

Hinge loss with a $\kappa$-Lipschitz critic, point masses $\mu$ apart:
value $\min(\kappa\mu, 2) - 2$, a slope out to $\mu = 2/\kappa$, where
total variation has none.
:::

::: {.slide title="Samples Pin Down MMD in Any Dimension, W1 Slowly"}
- MMD: error $O(n^{-1/2})$ in any dimension.
- $W_1$: error $n^{-1/d_x}$ in dimension $d_x > 2$; halving it can take
  $2^{d_x}$ times the samples.
- A WGAN critic never computes $W_1$ between minibatches: a small network
  supplies a direction.
- Forward-KL critic: each term has variance $\chi^2(p\|q)$, possibly
  infinite when KL is finite.
:::

::: {.slide title="Check: Where Does MMD Give the Most Gradient?"}
[Check]{.kicker}

For unit Gaussians $\mu$ apart and an RBF kernel of bandwidth $s$,
$\mathrm{MMD}^2 \propto 1 - \exp\big(-\mu^2 / (2(s^2+2))\big)$. At which
separation $\mu$ is its slope largest when $s = 4$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The slope $\propto \mu\, e^{-\mu^2/(2(s^2+2))}$ peaks at
$\mu = \sqrt{s^2 + 2} = \sqrt{18} \approx 4.2$.
:::
:::
:::

::: {.slide title="JS Loses Its Slope as Supports Separate; W1 Keeps It"}
$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$; slope of each
objective in $\mu$, bandwidth $s = 4$:

@!objectives-divergence-against-separation

JS: 0.20 at $\mu = 1$, 0.0013 at $\mu = 7$. $W_1$: always 1. MMD: peak
at $\mu = 4.2$.
:::

::: {.slide title="Optimal Critics: Only Those That Slope across the Gap Help"}
![](../img/mdl-gan-critics-separated.svg){width=100%}

- Disjoint supports: JS critic flat $\pm\infty$ on each, no direction.
- Unit Gaussians: $\lambda$ is a line of slope $-\mu$; what flattens is
  the posterior $\sigma(\lambda)$, and with it JS.
- $W_1$ critic: unit-slope ramp. MMD witness: smooth kernel bumps.
- Hinge critic $\operatorname{sign}\lambda$: a step, level at both clouds.
:::

<!-- WRITER: deck: add a row for the one-sided arm (gradient 1.26, arrival steps 476-479, stays within 0.5 of the data; ends within 0.06), and replace "then may stall or be pushed away" by the measured overshoot (farthest 1.1-7.6 in every run); round per S1 (hinge "about 700 to 950", NS "about 380"). JAX values now: NS 2.72-2.90, 378-387; hinge 1.07-1.63, 743-938. -->
::: {.slide title="A Far Generator Moves unless Its Loss Saturates"}
Data $\mathcal{N}(0,1)$; generator $x' = \theta + z$ from $\theta = 6$;
plain gradient steps; three runs per arm.

| arm | gradient | first $\lvert\theta\rvert < 0.5$ |
|:--|:--|:--|
| logistic, saturating | 0.006--0.015 | not in 1,500 steps |
| logistic, non-saturating | 2.7--3.0 | steps 377--388 |
| hinge | 1.1--1.7 | steps 719--918 |
| $W_1$ + gradient penalty | 1.25--1.26 | steps 477--508, then may stall or be pushed away |
| MMD, median bandwidth | 0.21 | $\theta = 1.32$ at step 1,500 |

::: {.d2l-note}
Same critic, two generator losses: gradients differ by a factor of hundreds.
Hinge: the optimal step is level at both clouds, yet the trained critic's
slope there is 1.1--1.7.
:::
:::

::: {.slide title="Every Loss Fits the Mixture; the Hinge Fit Is the Loosest Here"}
@!objectives-one-testbed-four-losses-6

From 0.529 nats to a few thousandths; three untuned runs per loss do not
rank the losses.
:::

::: {.slide title="Practice Pairs a Non-Saturating Generator with a Smooth Critic"}
- BigGAN generation: hinge loss, spectrally normalized critic.
- HiFi-GAN vocoders: least squares.
- StyleGAN line: non-saturating logistic loss, zero-centered penalty on
  real samples.

::: {.d2l-note}
With equal tuning, losses score alike; all three use a generator loss that
does not saturate.
:::

[Lucic et al., 2018; Miyato et al., 2018; Brock et al., 2019; Kong et al., 2020; Karras et al., 2020]{.small}
:::

::: {.slide title="Check: Why Does the W1 Generator Not Settle?"}
[Check]{.kicker}

The non-saturating generator settles at the data. The $W_1$ generator
arrives at nearly full speed, then stalls or is pushed away again. Why?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The $W_1$ push stays near unit size at the data. Past it the
critic must reverse its slope, and the penalty $\gamma(|T'| - 1)^2$ puts
a barrier at slope zero: a critic left on the old side pushes the
generator away; a ReLU critic can also go flat and leave it where it is.
:::
:::
:::

<!-- WRITER: M1/M2: the placeholder now prints six critics with units per layer. Bullets: flat = dead (0 of 32 units on in layer 2), not merely flat; drop the theta = -7.6 bullet (true in PyTorch only); state the reading rule and the one-sided contrast. -->
::: {.slide title="At Step 1,500 Each W1 Critic Is a Ramp or Flat"}
Slope $D'$ of each $W_1$ critic at the generator $\theta$ and at the data
mean $0$; active units of 64:

@!objectives-separated-start-w1-critics

- Flat: slope 0.00 at both points; $\theta$ stays where it is.
- Old side, $\theta \approx -7.6$, a distance $d = 7.6$ past the data: slope $-0.62$ to $-0.65$, pushing
  $\theta$ out; linear prediction $1 - d/20 = 0.62$.
- A critic that crosses settles at $1 + d/20$ and pulls $\theta$ back.

::: {.d2l-note}
Same slope at $\theta$ and at $0$: across the gap each critic is nearly a
straight line. Arriving is easy; staying is not.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **Template:** payoffs added, critic maximizes; convex in $q$.
- **Free critic:** each loss gives an f-divergence and a link of
  $\lambda$; coin: 0.131 (JS), $\tfrac1{16}$, $\tfrac14$, $\tfrac12$ (TV).
- **Tangent lines:** any $f$, critic $f'(p/q)$; KL via $1 + \lambda$.
- **Restricted critic:** MMD, $W_1$; JS slope 0.20 → 0.0013 from
  $\mu = 1$ to 7.
- **Far start:** saturating gradient 0.01, non-saturating about 3. The
  generator moves when its critic slopes at its samples and its loss
  passes the slope on; the divergence decides neither. $W_1$ + penalty
<!-- WRITER: Wrap-up: "arrives by step 508" -> "near step 500"; add the one-sided arm (stays at the data). -->
  arrives by step 508, then stalls or is pushed out.
- **Mixture:** all five losses from 0.529 nats to a few thousandths.

**Next:** a critic that compares a real sample with a generated one.
:::
