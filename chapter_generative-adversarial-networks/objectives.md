# Adversarial Objectives and Divergences
:label:`sec_gan_objectives`

The garments of :numref:`sec_basic_gan` were trained with one loss, and after twenty epochs the critic still told real garments from generated ones with some confidence. Open almost any GAN codebase and its training script offers a menu of losses: non-saturating, least squares, hinge, Wasserstein. Practitioners switch between them the way they switch optimizers, keeping whichever trains. :numref:`sec_basic_gan` analyzed only the first. Its critic learned the log ratio of the two densities, and its game measured the Jensen--Shannon divergence, which stops changing once the generated samples no longer overlap the data; an optimal critic then gives the generator no direction. Real and generated images typically lie on nearly disjoint sets, so a critic that stays ahead, as on the garments, comes close to that case. What do the other losses compute, and does any of them keep a gradient when the supports separate?

Every entry of the menu changes one of two things: the loss that trains the critic, or the set of functions the critic may be. We work out what each choice computes, check the predictions on a two-dimensional Gaussian mixture whose densities we know, and train one generator under five losses.

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

### The Template

The value function of :numref:`sec_basic_gan` scores a real sample $x \sim p$ through $\log \sigma(D(x))$ and a generated sample $x' \sim q$ through $\log \sigma(-D(x'))$, where $D$ is the critic's logit. We keep that section's notation: the mixture $m = (p+q)/2$, the density ratio $\rho = p/q$, and the optimal logit $\lambda = \log \rho$. Replacing the two log-probabilities by general scoring functions, $\ell_p$ for real samples and $\ell_q$ for generated ones, gives the template

$$
d(p, q) \;=\; \sup_{T \in \mathcal{T}} \Big\{ \mathbb{E}_{x \sim p}\big[\ell_p(T(x))\big] - \mathbb{E}_{x' \sim q}\big[\ell_q(T(x'))\big] \Big\},
$$
:eqlabel:`eq_gan_template`

in which the critic $T$ ranges over a class $\mathcal{T}$. The game of :numref:`sec_basic_gan` is the case $\ell_p(t) = \log \sigma(t)$ and $\ell_q(t) = -\log \sigma(-t)$ with $\mathcal{T}$ unrestricted.

The first family keeps the critic unconstrained and varies the scores. The supremum can then be taken separately at each $x$, so the result depends on $p$ and $q$ only through $\rho(x)$; these objectives are f-divergences. The second family uses linear scores, $\ell_p = \ell_q = \mathrm{id}$, and restricts $\mathcal{T}$ to a ball of functions, which couples the critic's values across points. A kernel ball gives the maximum mean discrepancy, and the ball of 1-Lipschitz functions the Wasserstein-1 distance (:numref:`fig_gan_template`).

![Two choices define the classical adversarial objectives: the scoring functions and the critic class. Nonlinear scores with an unconstrained critic yield f-divergences, including the original GAN. Linear scores with a constrained critic yield integral probability metrics: MMD for a kernel ball, Wasserstein-1 for a Lipschitz ball. The relativistic objective of :numref:`sec_gan_relativistic` scores pairs of samples.](../img/mdl-gan-template.svg)
:label:`fig_gan_template`

### Convexity in the Generator's Distribution

For a fixed critic, the expression in braces is affine in the pair $(p, q)$: linear in $p$ and linear in $q$. Every objective of the template is therefore a supremum of affine functionals, hence jointly convex in $(p, q)$. In particular, for fixed data $p$ it is convex in $q$, and in the space of distributions the generator's problem has no spurious local minima. Convexity in $q$ does not make the objective convex in the network parameters that represent $q$ (Exercise :exref:`ex_objectives-convexity-parameterization`).

## Classification Losses and Their Divergences

### Two Outcomes by Hand

Take a coin. The data put probabilities $p = (\tfrac34, \tfrac14)$ on heads and tails, and the generator $q = (\tfrac14, \tfrac34)$. Each toss comes from the data or from the generator with probability one half, and the critic, shown the outcome, must say which source produced it. Seeing heads, the source is real with probability $\eta = p/(p + q) = \tfrac34$; seeing tails, with probability $\tfrac14$. A critic that ignores the outcome can only report one half. How much looking helps depends on the loss:

- Under the log loss, the best report is the posterior itself. Its expected loss is the binary entropy $H_b(\tfrac34) = 0.562$ nats, against $\log 2 = 0.693$ nats for the blind report. Looking saves $0.131$ nats, which is $\mathrm{JS}(p, q)$.
- Under the squared error of a reported probability, the best report is again the posterior, with expected loss $\tfrac34 \cdot \tfrac14 = \tfrac{3}{16}$ against $\tfrac14$ for the blind report. Looking saves $\tfrac{1}{16}$.
- Under the hinge loss on a score $t$, the best scores are $t = +1$ on heads and $t = -1$ on tails, with expected loss $2 \cdot \tfrac14 = \tfrac12$ against $1$ at $t = 0$. Looking saves $\tfrac12$, which is the total variation distance $\tfrac12 \sum |p - q|$.

The three losses assign three numbers to one pair of distributions, all from the same posterior, and each saving vanishes when $p = q$. We now show that every such saving is an f-divergence.

### The Bayes-Risk Gap

Let $\ell$ be a payoff on critic outputs, so that $-\ell$ is a loss. A real sample earns $\ell(D(x))$ and a generated sample earns $\ell(-D(x'))$:

$$
V_\ell(D) \;=\; \mathbb{E}_{x \sim p}\big[\ell(D(x))\big] + \mathbb{E}_{x' \sim q}\big[\ell(-D(x'))\big].
$$
:eqlabel:`eq_gan_margin`

The choice $\ell = \log \sigma$ recovers the log-loss game. The critic's signed score, $D(x)$ on a real sample and $-D(x')$ on a generated one, is its *margin*, and $-\ell$ is a *margin loss*. As in :numref:`sec_basic_gan`, the maximization decouples across points. At each $x$ the integrand is $p\,\ell(t) + q\,\ell(-t)$ with $t = D(x)$, which is $2m(x)$ times an average under the posterior $\eta(x) = p(x)/(p(x)+q(x))$. The smallest expected loss a critic can reach at posterior $\eta$ is the *conditional Bayes risk*

$$
L(\eta) \;=\; \inf_{t} \Big\{ \eta \, \big({-\ell(t)}\big) + (1 - \eta)\, \big({-\ell(-t)}\big) \Big\}.
$$
:eqlabel:`eq_gan_bayes_risk`

As an infimum of functions linear in $\eta$, $L$ is concave for every $\ell$. A critic that ignores its input already reaches the risk $L(\tfrac12)$. Subtracting this baseline leaves what observing the sample contributes, the *Bayes-risk gap*

$$
\Delta_\ell(p, q) \;=\; L(\tfrac12) - \mathbb{E}_{x \sim m}\big[L(\eta(x))\big].
$$
:eqlabel:`eq_gan_bayes_gap`

It fixes the value of the game, $\sup_D V_\ell = 2\Delta_\ell - 2 L(\tfrac12)$, and it is nonnegative by Jensen's inequality, because $L$ is concave and $\mathbb{E}_m[\eta] = \tfrac12$. For the log loss, $L$ is the binary entropy, and the gap is the Jensen--Shannon divergence. The three savings of the coin are three Bayes-risk gaps.

### Bayes-Risk Gaps as f-Divergences

An f-divergence averages a convex function of the density ratio under $q$, $D_f(p \,\|\, q) = \mathbb{E}_{x \sim q}[f(\rho(x))]$, where $f$ is convex with $f(1) = 0$. :numref:`sec_mdl-f-divergences` proves that $D_f \geq 0$ and fixes the conventions where one density vanishes. We call $f$ the *divergence function*; the f-GAN literature calls it the generator, a word this chapter keeps for the network.

**Proposition.** *For any loss with concave Bayes risk $L$,*

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

Thus every classification loss with a free critic evaluates the f-divergence built from its Bayes risk. The optimal critic is still a fixed function of the posterior $\eta(x)$, and hence of $\lambda(x)$, but each loss reports it through its own *link*: the logistic critic reports $\lambda$ itself, the hinge critic only its sign.

### Divergences of Common Losses

:numref:`tab_gan_loss_divergences` evaluates :eqref:`eq_gan_bayes_gap` for four common losses, with the binary entropy $H_b(\eta)$ and the total variation distance $\mathrm{TV}(p, q) = \tfrac12 \int |p - q|$. The square row scores a probability against zero--one targets. The squared hinge pays $\max(0, 1 - t)^2$ on a margin $t$; its optimum $t = 2\eta - 1$ lies inside the margin, where it agrees with the square loss on $\pm 1$ targets, so its gap is four times the square row's.

:Classification losses in the game of a single critic: the conditional Bayes risk, its value at the prior, the Bayes-risk gap $\Delta_\ell$ (the value of the game is $2\Delta_\ell - 2L(\tfrac12)$), and the optimal critic as a function of the log ratio.
:label:`tab_gan_loss_divergences`

| loss | Bayes risk $L(\eta)$ | $L(\tfrac12)$ | gap $\Delta_\ell(p,q)$ | optimal critic |
|:---|:---|:---|:---|:---|
| logistic | $H_b(\eta)$ | $\log 2$ | $\mathrm{JS}(p,q)$ | $\lambda$ |
| square | $\eta(1-\eta)$ | $1/4$ | $\tfrac{1}{8}\int \frac{(p-q)^2}{p+q}$ | $\sigma(\lambda)$ |
| squared hinge | $4\eta(1-\eta)$ | $1$ | $\tfrac{1}{2}\int \frac{(p-q)^2}{p+q}$ | $\tanh(\lambda/2)$ |
| hinge | $2\min(\eta, 1-\eta)$ | $1$ | $\mathrm{TV}(p,q)$ | $\operatorname{sign} \lambda$ |

At the coin pair the four gaps are $0.131$, $\tfrac{1}{16}$, $\tfrac14$ and $\tfrac12$. The square row is the critic loss of LSGAN :cite:`Mao.Li.Xie.ea.2017`, and its gap is a quarter of the bounded divergence $\chi^2(p \,\|\, m) = \tfrac12 \int (p-q)^2/(p+q)$. The hinge loss came to adversarial training from support vector machines through the Geometric GAN :cite:`Lim.Ye.2017`; at $\eta = \tfrac12$ every score in $[-1, 1]$ is optimal for it.

### Critic Links
:label:`subsec_gan_objectives_critics`

A nonnegative loss has a bounded gap, $\Delta_\ell \leq L(\tfrac12)$, so the unbounded forward KL divergence that maximum likelihood minimizes is the gap of no such loss. Fenchel duality builds a game for any convex divergence function $f$. With the conjugate $f^*(t) = \sup_u \{ut - f(u)\}$, every critic $T$ certifies the lower bound

$$
D_f(p \,\|\, q) \;\geq\; \mathbb{E}_{x \sim p}\big[T(x)\big] - \mathbb{E}_{x' \sim q}\big[f^*(T(x'))\big],
$$
:eqlabel:`eq_gan_fgan_bound`

which :numref:`sec_mdl-f-gan-dual` proves from the Fenchel--Young inequality $f(u) \geq ut - f^*(t)$. Training a network $T$ to maximize the right-hand side is the f-GAN construction :cite:`Nowozin.Cseke.Tomioka.2016`; its Jensen--Shannon member is the original GAN (Exercise :exref:`ex_objectives-variational-bound-eqref-eq`). The bound is attained by $T^\star = f'(\rho)$, once more a monotone link of the ratio, and an imperfect critic underestimates the divergence. For the forward KL, $f(u) = u \log u$ has the conjugate $f^*(t) = e^{t-1}$, and the optimal critic is $1 + \lambda$.

We can test the links point by point when both densities are known. The data $p$ are a uniform mixture of three Gaussians in the plane. For $q$ we set a second mixture by hand, with each mean moved by up to $0.3$ and the components made wider, narrower and sheared, so that $\lambda$ has a closed form at every point.

```{.python .input #objectives-one-testbed-four-losses-1}
%%tab pytorch, jax
mu_p = np.array([[0.0, 2.4], [-2.1, -1.2], [2.1, -1.2]])     # data
A_p = np.stack([0.7 * np.eye(2)] * 3)
mu_q = np.array([[0.2, 2.6], [-2.3, -1.0], [2.0, -1.3]])     # hand-set q
A_q = np.stack([np.diag([0.8, 0.55]), np.diag([0.5, 0.6]),
                np.array([[0.65, 0.2], [-0.1, 0.7]])])

def mixture_logpdf(x, mus, As):
    """Log density of the uniform mixture of Gaussians N(mu_k, A_k^T A_k)."""
    comps = []
    for mu, A in zip(mus, As):
        Sigma, diff = A.T @ A, x - mu
        sol = np.linalg.solve(Sigma, diff.T).T
        comps.append(-0.5 * ((diff * sol).sum(1)
                             + np.log(np.linalg.det(2 * np.pi * Sigma))))
    return np.logaddexp.reduce(np.stack(comps) - np.log(len(mus)), axis=0)

def sample_mixture(n, mus, As, rng):
    k = rng.integers(0, len(mus), n)
    z = rng.standard_normal((n, 2))
    return np.einsum('ni,nij->nj', z, As[k]) + mus[k]
```

On a fixed grid, quadrature gives $\mathrm{JS}(p, q)$ for any mixture $q$ of this form.

```{.python .input #objectives-js-by-quadrature}
%%tab pytorch, jax
grid_1d = np.linspace(-7.0, 7.0, 281)
gx, gy = np.meshgrid(grid_1d, grid_1d)
grid_2d = np.stack([gx.ravel(), gy.ravel()], axis=1)
cell_area = (grid_1d[1] - grid_1d[0]) ** 2

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

The hand-set $q$ is a few hundredths of a nat from the data, a mismatch of the size a generator meets late in training. The critic is a small leaky-ReLU network with the same initialization in both frameworks.

```{.python .input #objectives-one-testbed-four-losses-3}
%%tab pytorch
TRUNC_STD = 0.8796  # standard deviation of N(0, 1) truncated at +-2

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
def make_critic(rngs):
    """Leaky-ReLU MLP; nnx.Linear gives LeCun-normal weights, zero biases."""
    leaky = lambda x: nnx.leaky_relu(x, 0.2)
    return nnx.Sequential(nnx.Linear(2, 64, rngs=rngs), leaky,
                          nnx.Linear(64, 64, rngs=rngs), leaky,
                          nnx.Linear(64, 1, rngs=rngs))
```

Each loss is stored as a pair: the critic's loss on the scores of real and generated samples, and the generator's loss that :numref:`subsec_gan_objectives_testbed` trains with.

```{.python .input #objectives-loss-pairs}
%%tab pytorch, jax
losses = {  # name: (critic loss, generator loss), both on critic scores
    'logistic': (lambda r, f: (softplus(-r) + softplus(f)).mean(),
                 lambda f: softplus(-f).mean()),
    'least squares': (lambda r, f: ((r - 1) ** 2 + f ** 2).mean() / 2,
                      lambda f: ((f - 1) ** 2).mean() / 2),
    'hinge': (lambda r, f: (relu(1 - r) + relu(1 + f)).mean(),
              lambda f: -f.mean()),
    'squared hinge': (lambda r, f: (relu(1 - r) ** 2 + relu(1 + f) ** 2).mean(),
                      lambda f: -f.mean()),
}
```

The links describe the best response to a fixed $q$, so we train a fresh critic against the hand-set $q$ under three of the losses. A fourth critic maximizes the forward-KL bound $\mathbb{E}_p[T] - \mathbb{E}_q[e^{T-1}]$.

```{.python .input #objectives-what-each-critic-estimates}
%%tab pytorch
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

Each critic follows its own link of the same function: the identity, the sigmoid, a step at $\lambda = 0$, and the line $1 + \lambda$. All four drift from their links where $|\lambda|$ is large, because the objective constrains a critic only where samples land, and few land there. The hinge critic is the least determined; the training run below shows the consequence.

## Integral Probability Metrics

The second family keeps linear scores and constrains the critic. The objective becomes

$$
d_{\mathcal{F}}(p, q) \;=\; \sup_{T \in \mathcal{F}} \Big\{ \mathbb{E}_{x \sim p}\big[T(x)\big] - \mathbb{E}_{x' \sim q}\big[T(x')\big] \Big\},
$$
:eqlabel:`eq_gan_ipm`

the largest difference in expectation that a test function in $\mathcal{F}$ can detect, an *integral probability metric* (:numref:`sec_mdl-ipm-mmd`). Its value no longer reduces to the pointwise ratio $\rho(x)$; it depends on which functions $\mathcal{F}$ contains.

### Maximum Mean Discrepancy

Taking $\mathcal{F}$ to be the unit ball of a reproducing kernel Hilbert space with kernel $k$ gives the maximum mean discrepancy (MMD). The supremum has a closed form :cite:`Gretton.Borgwardt.Rasch.ea.2012`, and :numref:`sec_mdl-ipm-mmd` derives its square as three kernel averages,

$$
\mathrm{MMD}^2(p, q) \;=\; \mathbb{E}_{x, \tilde x \sim p}\big[k(x, \tilde x)\big] + \mathbb{E}_{x', \tilde x' \sim q}\big[k(x', \tilde x')\big] - 2\, \mathbb{E}_{x \sim p,\, x' \sim q}\big[k(x, x')\big],
$$
:eqlabel:`eq_gan_mmd2`

with the two samples in each average drawn independently: similarity within each distribution minus twice the similarity across them. For a fixed kernel, a generator can minimize $\mathrm{MMD}^2$ directly, with no critic network, and the kernel alone decides what the discrepancy can see. The unbiased minibatch estimate averages each kernel matrix, leaving out the diagonal terms that compare a point with itself. The function below takes the three matrices, so it serves any kernel; :numref:`sec_dcgan` scores images with it.

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

Taking $\mathcal{F}$ to be the 1-Lipschitz functions gives the Wasserstein-1 distance $W_1$, the least average distance over which the mass of $p$ must be moved to turn it into $q$ (:numref:`sec_mdl-optimal-transport`). On the line, $W_1$ is the area between the two cumulative distribution functions,

$$
W_1(p, q) \;=\; \int_{-\infty}^{\infty} \big| F_p(t) - F_q(t) \big| \, dt,
$$

so a translation by $\mu$ costs exactly $|\mu|$. In higher dimensions there is no closed form, so a WGAN trains a critic network and enforces the constraint approximately :cite:`Arjovsky.Chintala.Bottou.2017`, by weight clipping, a gradient penalty :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017` or spectral normalization :cite:`Miyato.Kataoka.Koyama.ea.2018`. Between two Gaussians the squared Wasserstein-2 distance also has a closed form, which :numref:`sec_dcgan` uses to score images.

### Estimation from Samples
:label:`subsec_gan_objectives_estimation`

The two metrics differ in how well samples determine them. The unbiased MMD estimate converges at the rate $O(n^{-1/2})$ in any dimension :cite:`Gretton.Borgwardt.Rasch.ea.2012`, whereas the empirical $W_1$ in dimension $d_x > 2$ converges at the rate $n^{-1/d_x}$ :cite:`Weed.Bach.2019`, so halving its error can take $2^{d_x}$ times as many samples. Density-ratio objectives are hardest to estimate where the ratio is extreme: at the optimal forward-KL critic, each term $e^{T-1}$ of the minibatch average under $q$ equals $\rho$ and has variance $\chi^2(p \,\|\, q)$, which can be infinite even when the KL divergence is finite.

## Gradients Far from and Near the Data
:label:`subsec_gan_objectives_gradients`

The two families disagree about the failure that closed :numref:`sec_basic_gan`. An f-divergence integrates a function of $\rho(x)$, which on disjoint supports is $0$ or $\infty$ at every point, so nothing records how far apart the supports are: the divergence stays at its ceiling, and the generator's gradient vanishes. An integral probability metric averages one test function under both distributions, and if the test functions vary smoothly in space, its value changes as the supports move. Smoothness is the condition, not the family: total variation is the integral probability metric of the functions bounded by $\tfrac12$ :cite:`Sriperumbudur.Fukumizu.Gretton.ea.2009`, yet it saturates on disjoint supports, because that class contains functions as sharp as indicators.

### Divergence against Separation
:label:`subsec_gan_objectives_separation`

We can measure how much gradient each objective keeps as $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu, 1)$ move apart. All three can be evaluated without sampling error: Jensen--Shannon by quadrature, and $W_1 = |\mu|$ exactly. For the RBF kernel $k(x, x') = \exp(-(x - x')^2/(2s^2))$ with bandwidth $s$, the kernel averages in :eqref:`eq_gan_mmd2` are Gaussian integrals, which give

$$
\mathrm{MMD}^2(p, q) \;=\; \frac{2s}{\sqrt{s^2 + 2}} \left( 1 - \exp\!\left( -\frac{\mu^2}{2(s^2 + 2)} \right) \right).
$$
:eqlabel:`eq_gan_mmd_gauss`

We plot each objective's slope with respect to $\mu$, the gradient a generator that controls $\mu$ receives, on a log axis, with bandwidth $s = 4$.

```{.python .input #objectives-divergence-against-separation}
%%tab pytorch, jax
xs = np.linspace(-10.0, 18.0, 5601)           # fixed quadrature grid

def normal_pdf(x, mu, sigma=1.0):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(
        2 * np.pi * sigma ** 2)

def js_1d(p, q):
    m = (p + q) / 2
    return 0.5 * np.trapezoid(p * np.log(p / m) + q * np.log(q / m), xs)

s = 4.0                                        # RBF kernel bandwidth
mus = np.linspace(0.1, 8.0, 80)                # separations mu
js = np.array([js_1d(normal_pdf(xs, 0.0), normal_pdf(xs, mu)) for mu in mus])
mmd2 = 2 * s / np.sqrt(s ** 2 + 2) * (1 - np.exp(-mus ** 2
                                                 / (2 * (s ** 2 + 2))))
slopes = [np.gradient(js, mus), np.gradient(mmd2, mus),
          np.ones_like(mus)]                   # W1 = |mu| has slope one
for mu in (1.0, 7.0):
    print(f'mu = {mu:.0f}: JS slope {np.interp(mu, mus, slopes[0]):.4f}, '
          f'JS / log 2 = {np.interp(mu, mus, js) / np.log(2):.4f}')
d2l.plot(mus, slopes, xlabel=r'separation $\mu$', ylabel='slope',
         legend=['JS', 'MMD$^2$', '$W_1$'], yscale='log', figsize=(5, 3))
```

The Jensen--Shannon slope collapses once the densities stop overlapping. At $\mu = 7$ it is more than a hundred times smaller than at $\mu = 1$, and the divergence is within a tenth of a percent of $\log 2$, so the generator receives almost no signal about which way to move. The $W_1$ slope stays at one. The $\mathrm{MMD}^2$ slope peaks at a separation set by the bandwidth and then decays like a Gaussian in $\mu$, so the bandwidth sets the range over which MMD gives a gradient.

### Optimal Critics on Separated Supports

:numref:`fig_gan_critics_separated` shows why. The Jensen--Shannon critic $\lambda$ is flat on each support, so moving $q$ a little leaves every generated sample on the same level and carries no information about direction. The $W_1$ critic is a ramp of unit slope, so every generated sample sits on a slope toward the data, however far away. The MMD witness, a smooth difference of kernel bumps, slopes toward the data as long as the kernel reaches across the gap.

![Optimal critics when the data density $p$ (blue) and the generated density $q$ (orange) have disjoint supports. Left: the Jensen--Shannon critic $\log(p/q)$, $+\infty$ on the support of $p$ and $-\infty$ on that of $q$, flat on both. Middle: the $W_1$ critic, a ramp of slope one. Right: the MMD witness $\mathbb{E}_{p}[k(x,\cdot)] - \mathbb{E}_{q}[k(x,\cdot)]$, a smooth difference of kernel bumps (bandwidth half the separation, arbitrary scale). The orange arrows mark each critic's slope at $q$, the direction in which the generator moves.](../img/mdl-gan-critics-separated.svg)
:label:`fig_gan_critics_separated`

### Five Losses on a Gaussian Mixture
:label:`subsec_gan_objectives_testbed`

Every divergence in :numref:`tab_gan_loss_divergences` vanishes at $q = p$, so with overlapping supports every loss should drive a generator to the data. Near the end, though, the hinge critic reports only the sign of the remaining mismatch, the others also its size. We fit one generator to the three-mode mixture under five objectives: the non-saturating logistic loss of :numref:`sec_basic_gan`, least squares, hinge, the squared hinge, and MMD with a fixed kernel and no critic. The squared hinge keeps the hinge's margin and generator loss $-D$ but has the link $\tanh(\lambda/2)$, which carries magnitude, so comparing the two isolates the link. All five runs share the architecture, learning rates, step counts and start, with no separate tuning. A $W_1$ arm would need the Lipschitz penalty of :numref:`sec_gan_convergence`.

The generator draws one of three components uniformly and applies that component's learned affine map to Gaussian noise, so $q$ stays a three-component mixture whose divergence from $p$ we can compute after every step. We set the start by hand: each mean lies one to two and a half units from a different mode, and each map is $0.4$ times the identity, below the data's $0.7$, so positions and scales remain to be learned.

```{.python .input #objectives-five-losses-on-a-gaussian-mixture-1}
%%tab pytorch, jax
mu_0 = np.array([[1.0, -1.0], [-4.3, -2.0], [-1.0, 3.8]])  # off the modes
A_0 = np.stack([0.4 * np.eye(2)] * 3)      # below the data's scale of 0.7
print(f'JS(p, q) at the start: {js_mixtures(mu_0, A_0):.3f} nats')
```

The MMD arm minimizes the unbiased minibatch $\mathrm{MMD}^2$ under the sum of three RBF kernels with bandwidths $0.5$, $1$ and $2$, a kernel sensitive at several scales.

```{.python .input #objectives-one-testbed-four-losses-2}
%%tab pytorch
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

def mmd2_loss(x, y, bandwidths=(0.5, 1.0, 2.0)):
    """Unbiased MMD^2 under the sum of three RBF kernels."""
    def kernel(a, b):
        sq_dist = torch.cdist(a, b) ** 2
        return sum(torch.exp(-sq_dist / (2 * s ** 2)) for s in bandwidths)
    return mmd2_unbiased(kernel(x, x), kernel(y, y), kernel(x, y))

losses['MMD'] = (None, mmd2_loss)   # no critic: MMD^2 to the real batch
```

```{.python .input #objectives-one-testbed-four-losses-2}
%%tab jax
class MixtureGenerator(nnx.Module):
    """x = mu_k + z A_k for a uniformly drawn component k."""
    def __init__(self, mu, A):
        self.mu = nnx.Param(jnp.asarray(mu, jnp.float32))
        self.A = nnx.Param(jnp.asarray(A, jnp.float32))

    def __call__(self, z, k):
        return jnp.einsum('ni,nij->nj', z, self.A[k]) + self.mu[k]

    def params_np(self):
        return np.asarray(self.mu[...]), np.asarray(self.A[...])

def mmd2_loss(x, y, bandwidths=(0.5, 1.0, 2.0)):
    """Unbiased MMD^2 under the sum of three RBF kernels."""
    def kernel(a, b):
        sq_dist = ((a[:, None] - b[None]) ** 2).sum(-1)
        return sum(jnp.exp(-sq_dist / (2 * s ** 2)) for s in bandwidths)
    return mmd2_unbiased(kernel(x, x), kernel(y, y), kernel(x, y))

losses['MMD'] = (None, mmd2_loss)   # no critic: MMD^2 to the real batch
```

`train_toy` takes the two losses as arguments and alternates one critic step with one generator step. The MMD arm, stored with the critic loss `None`, takes only the generator step, on `mmd2_loss(X, fake)` instead of the critic's scores. Every 200 steps it records the exact $\mathrm{JS}(p, q_t)$.

```{.python .input #objectives-one-testbed-four-losses-4}
%%tab pytorch
def train_toy(critic_loss, gen_loss, seed=0, steps=6000, batch=256, lr=1e-3):
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
        loss = gen_loss(X, fake) if critic_loss is None else gen_loss(D(fake))
        loss.backward()
        opt_G.step()
        if t % 200 == 0:
            trace.append(js_mixtures(*G.params_np()))
    return np.array(trace)
```

```{.python .input #objectives-one-testbed-four-losses-4}
%%tab jax
@nnx.jit(static_argnums=(0, 1))
def train_step(critic_loss, gen_loss, G, D, opt_G, opt_D, X, z, k):
    if critic_loss is not None:            # the generator is fixed here
        fake = G(z, k)
        opt_D.update(D, nnx.grad(lambda D_: critic_loss(D_(X), D_(fake)))(D))
    def g_loss(G_):
        fake = G_(z, k)
        return gen_loss(X, fake) if critic_loss is None else gen_loss(D(fake))
    opt_G.update(G, nnx.grad(g_loss)(G))

def train_toy(critic_loss, gen_loss, seed=0, steps=6000, batch=256, lr=1e-3):
    rng = np.random.default_rng(seed)
    G, D = MixtureGenerator(mu_0, A_0), make_critic(nnx.Rngs(seed))
    opt_G = nnx.Optimizer(G, optax.adam(lr), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(2 * lr), wrt=nnx.Param)
    trace = []
    for t in range(steps):
        X = jnp.asarray(sample_mixture(batch, mu_p, A_p, rng), jnp.float32)
        z = jnp.asarray(rng.standard_normal((batch, 2)), jnp.float32)
        k = jnp.asarray(rng.integers(0, 3, batch))
        train_step(critic_loss, gen_loss, G, D, opt_G, opt_D, X, z, k)
        if t % 200 == 0:
            trace.append(js_mixtures(*G.params_np()))
    return np.array(trace)
```

The divergence fluctuates from step to step near the end of a run, so we report its median over the last 2,000 steps, and plot the traces.

```{.python .input #objectives-one-testbed-four-losses-6}
%%tab pytorch, jax
runs = {name: train_toy(*pair) for name, pair in losses.items()}
for name, trace in runs.items():
    print(f'{name:14s} median JS(p, q) over the last 2,000 steps: '
          f'{np.median(trace[-2000 // 200:]):.4f} nats')
d2l.plot(np.arange(len(runs['logistic'])) * 200, list(runs.values()),
         xlabel='step', ylabel='JS(p, q_t)', yscale='log',
         legend=list(runs), figsize=(5, 3))
```

Every objective takes the generator from about half a nat to a few thousandths of a nat or less. The hinge ends highest, at about $0.005$ nats, usually several times above the rest. The squared hinge, with the same margin and generator loss, ends below it, at the level of the logistic and least-squares games. MMD, which has no critic to keep up with, ends lowest. The logistic and least-squares runs also stay on their initial plateau longer than the other three.

The hinge stall comes from what its critic must track. Its target $\operatorname{sign} \lambda$ is a full-margin step however small the mismatch, and inside the margin the hinge risk pulls the critic toward it with slope $|1 - 2\eta| \approx |\lambda|/2$, which vanishes as $q$ approaches $p$. The critic falls behind, and the generator keeps following a stale push of full size. The squared hinge's target $\tanh(\lambda/2)$ shrinks with the mismatch, and its quadratic risk pulls in proportion to the error, so its critic keeps up. All five losses share the minimizer $q = p$; what the critic reports near it decides how closely the generator approaches it (Exercise :exref:`ex_objectives-plateaus-lag` tests the lag).

With overlapping supports every loss reaches the data, and studies that give each loss the same tuning budget find them comparable in final quality :cite:`Lucic.Kurach.Michalski.ea.2018`. The choice among them matters less than two questions the losses leave open: what happens when the supports separate, and whether gradient training reaches the equilibrium. Before turning to those, the next section keeps the log loss and changes what the critic scores: a real sample against a generated one.

## Summary

Every adversarial objective of this section is a supremum, over a critic class, of a difference of expected scores, so it is convex in the generator's distribution. With a free critic, a classification loss evaluates the f-divergence given by its Bayes-risk gap, and the trained critic reports a fixed link of the log ratio $\lambda$; Fenchel duality extends this to any divergence function. Constraining the critic instead gives integral probability metrics: MMD for a kernel ball, $W_1$ for the Lipschitz ball. When the supports separate, the ratio-based objectives saturate, while $W_1$ keeps unit slope and MMD keeps a slope over separations set by its bandwidth. When they overlap, every loss tested reaches the data, but the hinge, whose critic reports only a sign, stalls above the squared hinge and every other loss.

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
1. [code] **Checking and extending the table.**
    1. The Brier score predicts a probability $\hat\eta$ and pays
       $(y - \hat\eta)^2$ against the source label $y \in \{0, 1\}$. Show
       that its conditional Bayes risk :eqref:`eq_gan_bayes_risk` is
       $\eta(1 - \eta)$, attained at $\hat\eta = \eta$, and that its gap is
       $\tfrac18 \int (p - q)^2/(p + q)$. Confirm the factor $\tfrac18$ on
       Dirichlet draws of pairs of distributions on five outcomes.
    1. The exponential loss pays $e^{-t}$ on a margin $t$, and the
       zero--one loss pays one for a negative margin. Show that their
       Bayes risks are $2\sqrt{\eta(1 - \eta)}$ and $\min(\eta, 1 - \eta)$,
       that their gaps are the squared Hellinger distance
       $1 - \int \sqrt{pq}$ and $\tfrac12\, \mathrm{TV}(p, q)$, and that the
       optimal exponential critic is $\lambda/2$. Evaluate both gaps at the
       coin pair.
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
1. [code] **MMD gradient range.**
    1. Differentiate :eqref:`eq_gan_mmd_gauss` with respect to $\mu$ and
       show that the slope is largest at $\mu^\star = \sqrt{s^2 + 2}$.
    1. For the three-bandwidth kernel of `mmd2_loss`, plot the slope of
       $\mathrm{MMD}^2$ between the same two Gaussians for
       $\mu \in [0, 20]$, and find the separation beyond which it stays
       below 1% of its maximum. Compare with the single kernel $s = 4$.
   :exlabel:`ex_objectives-mmd-gradient-range`
1. **Lipschitz hinge critics.** ● Spectrally normalized GANs score higher
   with the hinge loss than with the logistic loss
   :cite:`Miyato.Kataoka.Koyama.ea.2018`, so practice combines the loss of
   an f-divergence with a constrained critic. Write the hinge game as
   $V(D) = \mathbb{E}_p[-\max(0, 1 - D)] + \mathbb{E}_q[-\max(0, 1 + D)]$.
    1. Show that $V(D) = \mathbb{E}_p[D] - \mathbb{E}_q[D] - 2$ whenever
       $|D| \leq 1$, and that clipping a critic to $[-1, 1]$ never lowers
       $V$. Conclude that the unconstrained hinge game is the integral
       probability metric of $\{T : \|T\|_\infty \leq 1\}$, shifted by
       $-2$, and check its value against the table.
    1. Restrict the critic to $\kappa$-Lipschitz functions. Show that
       clipping preserves the Lipschitz constant, and evaluate the game
       for point masses at $0$ and $\mu > 0$. Over which separations does
       it give the generator a gradient, compared with $W_1$ and an RBF
       MMD with bandwidth $s$?
   :exlabel:`ex_objectives-hinge-lipschitz`
1. [code] **Plateaus and critic lag.** ● Use `train_toy`.
    1. Pair the logistic critic loss with the generator loss $-D$ and
       compare it with the non-saturating entry over three seeds by the
       first step at which $\mathrm{JS}(p, q_t) < 0.3$ nats. Explain the
       difference from the per-sample weights $\sigma(-D(x'))$ and $1$
       that the two generator losses give (:numref:`sec_basic_gan`).
    1. Test the explanation of the hinge residual with two controls over
       three seeds: quadruple the critic's batch, or give the critic five
       steps per generator step. Which control lowers the residual, and
       does the outcome point to minibatch noise or to lag?
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
       `critics['forward KL']` by quadrature on `grid_2d`. A critic trained
       for 3,000 steps recovers roughly 85 to 95 percent of
       $D_{\textrm{KL}}(p \,\|\, q) \approx 0.16$ nats. Then evaluate
       the bound on minibatches of 512. Why can a minibatch estimate
       flatter a critic that overshoots $1 + \lambda$ where $q$ rarely
       samples?
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
1. [extended] **Separated initializations.** ●● The five losses of
   :numref:`subsec_gan_objectives_testbed` train only from overlapping
   supports. Let $p = \mathcal{N}(0, 1)$ and let the generator
   $q = \mathcal{N}(\mu, \tau^2)$ learn $\mu$ and $\log \tau$ from $\tau = 1$
   and $\mu_0 \in \{2, 5, 10, 20\}$. Compare the logistic, least-squares and
   hinge games with a one-dimensional version of the critic, the hinge game
   with a spectrally normalized critic, and MMD with RBF bandwidths $1$, $4$
   and the three-bandwidth kernel of `mmd2_loss`, at fixed optimizer
   settings and 5,000 generator steps. Tabulate, over three seeds, the
   steps until $\mathrm{JS}(p, q) < 0.01$, and explain every row from this
   section. Which runs stall, and where does a trained critic behave
   differently from the optimal critic of the analysis? Relate the rows
   to exercises :exref:`ex_objectives-hinge-lipschitz` and
   :exref:`ex_objectives-mmd-gradient-range` of this section.
   :exlabel:`ex_objectives-separated-initializations`

[Discussions](https://d2l.discourse.group/t/36809)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.2]{.kicker}

Adversarial Objectives and Divergences<br>
**the loss menu · every loss is a divergence · critics report links of p/q · which objectives keep a gradient**
:::
:::

::: {.slide title="GAN Codebases Offer a Menu of Losses"}
[The problem]{.kicker}

`non-saturating | least squares | hinge | Wasserstein`: practitioners
switch entries like optimizers.

- Garment GAN: log loss; after 20 epochs its critic still wins.
- The log-loss game measures Jensen--Shannon (JS); its critic learns
  $\lambda = \log(p/q)$.
- $\mathcal{N}(0, 1)$ against $\mathcal{N}(7, 1)$: JS is within 0.1% of
  its ceiling $\log 2$, so an optimal critic gives almost no direction.

::: {.d2l-note}
What do the other entries compute, and which keep moving the generator?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the divergence of any classification loss from its Bayes risk;
- read off what its optimal critic reports about $p/q$ (the *link*);
- build a game for any f-divergence by duality;
- say which objectives keep a gradient when supports separate;
- explain why the hinge game stalls near the solution.

**Running example:** a three-mode Gaussian mixture in the plane, with
densities known exactly.
:::

::: {.slide title="Two Knobs on One Game"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Generative Adversarial Networks | the log-loss game, $D^\star = \lambda$ |
| **Adversarial Objectives** | **other losses, constrained critics** |
| Relativistic Objectives | critics that rank pairs |
| Gradient Penalties and Convergence | dynamics and penalties |
:::

::: {.col .narrow}
- **Uses:** the log-loss game, JS, KL.
- **Skips:** architectures, image experiments.
:::
:::
:::

::: {.slide title="One Template Generates the Classical Objectives"}
$$d(p, q) = \sup_{T \in \mathcal{T}} \big\{ \mathbb{E}_{x \sim p}[\ell_p(T(x))] - \mathbb{E}_{x' \sim q}[\ell_q(T(x'))] \big\}$$

![](../img/mdl-gan-template.svg){width=60%}

- Free critic, nonlinear scores $\ell_p, \ell_q$: **f-divergences**.
- Linear scores, critic in a ball: **integral probability metrics**.
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

::: {.slide title="Every Margin Loss Measures an f-Divergence"}
Bayes risk $L(\eta)$: the smallest expected loss at posterior
$\eta = p/(p+q)$. The **Bayes-risk gap** is what looking saves:

$$\Delta_\ell(p, q) = L(\tfrac12) - \mathbb{E}_{x \sim m}\big[L(\eta(x))\big]$$

**Proposition.** $\Delta_\ell = \mathbb{E}_q[f(p/q)]$ with
$f(u) = L(\tfrac12) - \tfrac{u+1}{2} L\big(\tfrac{u}{u+1}\big)$, convex,
$f(1) = 0$.

Proof: substitute $u = p/q$; $L$ is concave, so $f$ is convex.
:::

::: {.slide title="Each Loss Picks a Divergence and a Link"}
| loss | gap $\Delta_\ell$ | optimal critic | coin |
|:---|:---|:---|:--|
| logistic | $\mathrm{JS}(p,q)$ | $\lambda$ | $0.131$ |
| square | $\tfrac18 \int (p-q)^2/(p+q)$ | $\sigma(\lambda)$ | $\tfrac1{16}$ |
| squared hinge | $\tfrac12 \int (p-q)^2/(p+q)$ | $\tanh(\lambda/2)$ | $\tfrac14$ |
| hinge | $\mathrm{TV}(p,q)$ | $\operatorname{sign}\lambda$ | $\tfrac12$ |

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

::: {.slide title="Duality Builds a Game for Any Divergence"}
For convex $f$ with conjugate $f^*(t) = \sup_u\{ut - f(u)\}$:

$$D_f(p \,\|\, q) \;\geq\; \mathbb{E}_p[T] - \mathbb{E}_q[f^*(T)]
\quad \text{for every critic } T$$

- Equality at $T^\star = f'(p/q)$: a monotone transform of the ratio.
- Forward KL: $f^* (t) = e^{t-1}$, so $T^\star = 1 + \lambda$.
- JS row with $T = \log(2\sigma(D))$: the original GAN.

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

::: {.slide title="Constrain the Critic Instead: Integral Probability Metrics"}
$$d_{\mathcal{F}}(p, q) = \sup_{T \in \mathcal{F}} \big\{ \mathbb{E}_p[T] - \mathbb{E}_q[T] \big\}$$

- **Kernel ball, MMD:** closed form, no critic network;
  $\mathrm{MMD}^2 = \mathbb{E}\,k(x, \tilde x) + \mathbb{E}\,k(x', \tilde x') - 2\,\mathbb{E}\,k(x, x')$.
- **1-Lipschitz ball, $W_1$:** the cost of moving $p$ onto $q$; on the line
  $\int |F_p - F_q|$. A network critic needs clipping, a penalty or
  spectral normalization.
:::

::: {.slide title="Samples Pin Down MMD in Any Dimension, W1 Slowly"}
- MMD: error $O(n^{-1/2})$ in any dimension.
- $W_1$: error $n^{-1/d_x}$ in dimension $d_x > 2$; halving it can take
  $2^{d_x}$ times the samples.
- Density ratio: at the optimal forward-KL critic each sample's term has
  variance $\chi^2(p\|q)$, which can be infinite when KL is finite.
:::

::: {.slide title="JS Loses Its Slope as Supports Separate; W1 Keeps It"}
$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$; slope of each
objective in $\mu$, kernel bandwidth $s = 4$:

@!objectives-divergence-against-separation

JS slope: 0.2 at $\mu = 1$, 0.0013 at $\mu = 7$. $W_1$: always 1.
MMD: peaks, then decays beyond a few bandwidths.
:::

::: {.slide title="Only Critics That Slope across the Gap Move the Generator"}
![](../img/mdl-gan-critics-separated.svg){width=100%}

- JS critic: flat $\pm\infty$ on each support, no direction.
- $W_1$ critic: unit-slope ramp. MMD witness: smooth kernel bumps.
:::

::: {.slide title="Check: Where Does MMD Give the Most Gradient?"}
[Check]{.kicker}

For unit Gaussians and an RBF kernel of bandwidth $s$,
$\mathrm{MMD}^2 \propto 1 - \exp\big(-\mu^2 / (2(s^2+2))\big)$. At which
separation $\mu$ is its slope largest when $s = 4$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The slope $\propto \mu\, e^{-\mu^2/(2(s^2+2))}$ peaks at
$\mu = \sqrt{s^2 + 2} = \sqrt{18} \approx 4.2$.
:::
:::
:::

::: {.slide title="The Hinge Stalls above the Squared Hinge"}
@!objectives-one-testbed-four-losses-6

All five reach a few $10^{-3}$ nats or less; the hinge ends highest, near
$0.005$; the squared hinge matches logistic and least squares; MMD lowest.
:::

::: {.slide title="The Hinge Critic Falls behind the Generator"}
- Hinge target $\operatorname{sign}\lambda$: a full-margin step however small
  the mismatch.
- Inside the margin its risk pulls with slope $|1 - 2\eta| \approx |\lambda|/2$,
  which vanishes as $q \to p$.
- The critic lags; the generator follows a stale, full-size push.
- Squared hinge: target $\tanh(\lambda/2)$ shrinks with the mismatch, and
  the pull is proportional to the error.

::: {.d2l-note}
Same minimizer, same margin, same generator loss: the link decides the finish.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **Template:** sup over a critic class of a score difference; jointly
  convex in $(p, q)$, so convex in $q$.
- **Free critic:** each loss gives an f-divergence and a link of $\lambda$;
  coin pair: 0.131 (JS), $\tfrac1{16}$, $\tfrac14$, $\tfrac12$ (TV).
- **Duality:** any $f$, critic $f'(p/q)$; an imperfect critic underestimates.
- **Constrained critic:** JS slope falls from 0.2 to 0.0013 between
  $\mu = 1$ and 7; $W_1$ keeps slope 1, MMD up to its bandwidth.
- **Training:** hinge stalls highest, near 0.005 nats; squared hinge lower,
  level with logistic and least squares.

**Next:** a critic that ranks a real sample against a generated one.
:::
