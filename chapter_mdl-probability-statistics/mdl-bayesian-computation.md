# Bayesian Computation
:label:`sec_mdl-bayesian-computation`

Maximum-likelihood and MAP estimation each return a single parameter vector.
When the data leave substantial parameter uncertainty, different plausible
vectors can produce different predictions. Bayesian prediction averages these
predictions with respect to the posterior distribution. The resulting
**posterior predictive** distribution is

$$
p(y_\star\mid\mathbf x_\star,\mathcal D)
=\int p(y_\star\mid\mathbf x_\star,\boldsymbol\theta)\,
       p(\boldsymbol\theta\mid\mathcal D)\,d\boldsymbol\theta.
$$
:eqlabel:`eq_mdl-bayes-predictive`

Outside the conjugate families of
:numref:`sec_mdl-distributions`, this integral — and the posterior's own
normalizer — usually has no closed form. We can evaluate the unnormalized
posterior pointwise, but must approximate its integrals. This leads to the
computational question:

> **How do you average over a distribution you can only evaluate pointwise,
> up to a constant?**

This section develops four approximations. Importance sampling draws from a
tractable proposal and reweights the samples. Markov chain Monte Carlo
constructs a chain whose stationary distribution is the posterior. The Laplace
approximation fits a Gaussian near a posterior mode. Variational inference
optimizes a tractable distribution to approximate the posterior. These methods
are used for posterior prediction, uncertainty estimates, and latent-variable
models.

One Bayesian logistic-regression problem provides a common comparison. Because
it has only two parameters, a sufficiently fine grid supplies a high-accuracy
numerical reference for checking the approximations. Conjugate posteriors are treated in
:numref:`sec_mdl-distributions`; MAP and the ELBO are developed in
:numref:`sec_mdl-maximum_likelihood`; differentiation through random
variables is in :numref:`subsec_mdl-stochastic-gradient-estimators`.
Standard broader references are :citet:`Bishop.2006` and
:citet:`Murphy.2022`. All computation is plain NumPy.

```{.python .input #bayesian-computation-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #bayesian-computation-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #bayesian-computation-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #bayesian-computation-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## The Target: A Posterior Distribution
:label:`sec_mdl-bayes-target`

### A Nonconjugate Running Example

Logistic regression is a small nonconjugate model. A Gaussian observation with
a Gaussian prior has a closed-form Gaussian posterior. A logistic likelihood
has no matching conjugate prior, so its posterior can be evaluated pointwise
but not integrated in closed form. Its two-parameter version is therefore a
useful example for comparing approximation methods.

For binary observations $y_i\in\{0,1\}$ and scalar features $x_i$, logistic
regression writes

$$
P(y_i=1\mid x_i,\boldsymbol\theta)
=\sigma(\theta_0+\theta_1x_i),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Put an independent zero-mean Gaussian prior with variance $\tau^2$ on the
intercept and slope. Bayes' rule gives

$$
p(\boldsymbol\theta\mid\mathcal D)
\propto
\left[\prod_{i=1}^n
 \sigma(\mathbf z_i^\top\boldsymbol\theta)^{y_i}
 (1-\sigma(\mathbf z_i^\top\boldsymbol\theta))^{1-y_i}\right]
\exp\!\left(-\frac{\|\boldsymbol\theta\|^2}{2\tau^2}\right),
$$
:eqlabel:`eq_mdl-bayes-logistic-posterior`

where $\mathbf z_i=(1,x_i)^\top$. The suppressed normalizer — the
**evidence** $p(\mathcal D)$ — is an integral over $\boldsymbol\theta$ that
we do not know. We can still evaluate relative log posterior values because the
constant cancels, and we can compute gradients. The methods below all use value
evaluations; gradient-based variants use the second operation as well.

The cell creates a small dataset and implements the log of the unnormalized
posterior. `logaddexp` evaluates $\log(1+e^z)$ without overflowing.

```{.python .input #bayesian-computation-model}
rng = np.random.default_rng(12)
n_bayes = 80
x_bayes = rng.uniform(-2.5, 2.5, n_bayes)
Z_bayes = np.column_stack([np.ones(n_bayes), x_bayes])
theta_true = np.array([-0.4, 1.3])
prior_var = 4.0

def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))

p_true = sigmoid(Z_bayes @ theta_true)
y_bayes = rng.binomial(1, p_true)

def log_joint(theta):
    """log p(data, theta), up to a constant; theta has final dimension 2."""
    theta = np.asarray(theta)
    logits = np.einsum('...d,nd->...n', theta, Z_bayes)
    log_lik = (y_bayes * logits - np.logaddexp(0.0, logits)).sum(axis=-1)
    log_prior = -0.5 * (theta**2).sum(axis=-1) / prior_var
    return log_lik + log_prior

def grad_log_joint(theta):
    theta = np.atleast_2d(theta)
    probs = sigmoid(theta @ Z_bayes.T)
    return (y_bayes - probs) @ Z_bayes - theta / prior_var
```

### Grid Integration as a Low-Dimensional Reference

With two parameters we can cheat: evaluate the log posterior on a dense
grid, exponentiate stably, and normalize the sum. This is deterministic
quadrature — excellent for auditing a two-parameter example, hopeless in
general, because with $m$ points per coordinate a $d$-dimensional grid
costs $m^d$ evaluations. That exponential wall is the same curse of
dimensionality that motivated Monte Carlo integration in
:numref:`sec_mdl-integral_calculus`, and it is why the rest of this notebook
exists: every method below is a way of spending evaluations only where the
posterior actually has mass, instead of tiling all of parameter space.

```{.python .input #bayesian-computation-grid}
b0 = np.linspace(-2.5, 1.5, 241)
b1 = np.linspace(0.0, 2.8, 241)
B0, B1 = np.meshgrid(b0, b1, indexing='ij')
grid_theta = np.stack([B0.ravel(), B1.ravel()], axis=1)
grid_logp = log_joint(grid_theta)
grid_weight = np.exp(grid_logp - grid_logp.max())
grid_weight /= grid_weight.sum()
posterior_mean = grid_weight @ grid_theta
centered = grid_theta - posterior_mean
posterior_cov = (centered * grid_weight[:, None]).T @ centered
print('grid posterior mean:', posterior_mean.round(4))
print('grid posterior sd  :', np.sqrt(np.diag(posterior_cov)).round(4))
print('grid correlation   :',
      round(posterior_cov[0, 1] /
            np.sqrt(posterior_cov[0, 0] * posterior_cov[1, 1]), 4))
```

### Posterior Mean, MAP, and Predictive Averaging

Before approximating the posterior, let us see what keeping it actually
changes. Two candidate summaries compete: the **posterior mean** (an
average) and the **MAP** (the single most probable point). To compare them,
compute the MAP by Newton iteration. The negative Hessian of the log
posterior is

$$
H(\boldsymbol\theta)
=Z^\top\operatorname{diag}(p_i(1-p_i))Z+\tau^{-2}I,
$$

positive definite thanks to the Gaussian prior, and — keep it in hand —
its inverse at the MAP will become the Laplace approximation below.

```{.python .input #bayesian-computation-map-predictive}
theta_map = np.zeros(2)
for _ in range(12):
    probs = sigmoid(Z_bayes @ theta_map)
    precision = (Z_bayes.T @ (Z_bayes * (probs * (1 - probs))[:, None])
                 + np.eye(2) / prior_var)
    theta_map += np.linalg.solve(precision, grad_log_joint(theta_map)[0])
probs = sigmoid(Z_bayes @ theta_map)
laplace_precision = (Z_bayes.T @
                     (Z_bayes * (probs * (1 - probs))[:, None])
                     + np.eye(2) / prior_var)
laplace_cov = np.linalg.inv(laplace_precision)
print('MAP                 :', theta_map.round(4))
print('posterior mean      :', posterior_mean.round(4))

x_new = np.array([-2.0, 0.0, 2.0])
Z_new = np.column_stack([np.ones(len(x_new)), x_new])
pred_grid = grid_weight @ sigmoid(grid_theta @ Z_new.T)
pred_map = sigmoid(Z_new @ theta_map)
for x0, pm, pp in zip(x_new, pred_map, pred_grid):
    print(f'x={x0:+.1f}: plug-in MAP={pm:.3f}, posterior predictive={pp:.3f}')
```

The two prediction columns differ. The reason is that the sigmoid is
nonlinear, and averaging a nonlinear function is not the same as applying it
to an average: plausible parameter vectors other than the MAP predict
differently, and the posterior predictive weighs all of them, whereas the
plug-in prediction omits parameter uncertainty. The gap closes as the
posterior concentrates with more data; at $n=80$ it is small but visible.
This example motivates the computational
question that follows: how can we evaluate posterior averages when direct
integration is unavailable?

## Importance Sampling: Correcting a Proposal
:label:`sec_mdl-bayes-importance`

Suppose that direct sampling from the posterior is unavailable, but that we
can sample from a tractable *proposal* distribution $q$. We correct the
difference between the proposal and posterior by assigning each draw a weight
proportional to their density ratio. A draw receives less weight where the
proposal density is too large and more weight where it is too small. This
procedure is **importance sampling**. It expresses the desired posterior
expectation as a ratio of expectations under the proposal. If $q$ is a
normalized density that we can sample and evaluate, and
$\widetilde p(\boldsymbol\theta)=p(\mathcal D,\boldsymbol\theta)$ is our
evaluable unnormalized target, then

$$
\mathbb E_{p(\boldsymbol\theta\mid\mathcal D)}[h(\boldsymbol\theta)]
=
\frac{\mathbb E_q[w(\boldsymbol\theta)h(\boldsymbol\theta)]}
     {\mathbb E_q[w(\boldsymbol\theta)]},
\qquad
w(\boldsymbol\theta)=\frac{\widetilde p(\boldsymbol\theta)}
                           {q(\boldsymbol\theta)}.
$$
:eqlabel:`eq_mdl-bayes-importance`

The unknown normalizing constant appears in both numerator and denominator and
therefore cancels. Replacing both
expectations by sample averages gives **self-normalized importance
sampling**: consistent, though slightly biased at finite $N$ because it is
a ratio of random quantities. As always with products of many likelihood
terms, the computation belongs in log space with a max subtracted, exactly
like log-sum-exp.

The identity requires more than nominal overlap. At minimum,
$q(\boldsymbol\theta)>0$ wherever the target contributes to the estimand, and
$\mathbb E_q[|w h|]$ and $\mathbb E_q[w]$ must be finite. A finite-variance
central limit theorem further requires finite second moments of the weights
and of the weighted quantity, $\mathbb E_q[w^2]<\infty$ and
$\mathbb E_q[w^2h^2]<\infty$. Thus full support is necessary but may be
practically useless when
$q$ has lighter tails than the target or places negligible mass in an important
mode.

The accuracy of the estimator depends on where the proposal places its draws.
:numref:`fig_mdl-prob-bayes-importance` shows both adequate and inadequate
proposal coverage.

![Importance sampling with two proposals. Both panels reweight 24 draws from a Gaussian proposal (dashed) against the same two-mode target (solid); stems show normalized weights. Left: a narrow proposal misses the right-hand mode, so the weights are blind to it even though the effective sample size looks nearly perfect. Right: a broader proposal samples both modes and produces less concentrated weights in this run. Support alone is not sufficient: tail behavior and weight moments also control accuracy.](../img/mdl-prob-bayes-importance.svg)
:label:`fig_mdl-prob-bayes-importance`

The standard health check is the weight **effective sample size**

$$
N_{\mathrm{eff}}=\frac{1}{\sum_{s=1}^N\bar w_s^2},
\qquad
\bar w_s=\frac{w_s}{\sum_r w_r},
$$
:eqlabel:`eq_mdl-bayes-is-ess`

which lies between $1$ (one draw carries everything) and $N$ (all draws
count equally) and measures how concentrated the weights are.
:numref:`fig_mdl-prob-bayes-importance` is a warning about its limits: the
left panel's ESS is excellent *because* the proposal never found the second
mode. ESS can detect weights that are unequal; it cannot detect mass that
was never sampled.

On our logistic posterior we compare two Gaussian proposals: the broad
prior, and the local Gaussian built from the MAP and inverse Hessian
computed above (the Laplace approximation, formalized below, here playing
its other standard role — as a proposal generator).

```{.python .input #bayesian-computation-importance}
def gaussian_logpdf(samples, mean, cov):
    diff = samples - mean
    sign, logdet = np.linalg.slogdet(cov)
    precision = np.linalg.inv(cov)
    return -0.5 * (len(mean) * np.log(2 * np.pi) + logdet
                   + np.einsum('ni,ij,nj->n', diff, precision, diff))

def importance(proposal_mean, proposal_cov, n, seed):
    local_rng = np.random.default_rng(seed)
    samples = local_rng.multivariate_normal(proposal_mean, proposal_cov, n)
    logw = log_joint(samples) - gaussian_logpdf(
        samples, proposal_mean, proposal_cov)
    weight = np.exp(logw - logw.max())
    weight /= weight.sum()
    mean = weight @ samples
    predictive = weight @ sigmoid(samples @ Z_new.T)
    ess = 1.0 / (weight @ weight)
    return mean, predictive, ess, weight.max()

prior_is = importance(np.zeros(2), prior_var * np.eye(2), 20000, 2)
laplace_is = importance(theta_map, laplace_cov, 5000, 3)
for name, result, n in [('prior proposal', prior_is, 20000),
                        ('Laplace proposal', laplace_is, 5000)]:
    mean, pred, ess, max_weight = result
    print(f'{name:18s}: ESS={ess:7.1f}/{n}, max weight={max_weight:.4f}, '
          f'mean={mean.round(4)}')
```

The prior proposal spends most of its 20,000 draws where the posterior is
negligible — that is what its low ESS is reporting — while the Laplace
proposal, aimed at the posterior, extracts more usable information from a
quarter as many draws. This posterior is unimodal and nearly Gaussian, so the
local proposal works well here. In high dimension or under mode or tail
mismatch, the weights may degenerate rapidly; weight moments and effective
sample size, not the nominal draw count, determine the accuracy.

## Markov Chain Monte Carlo
:label:`sec_mdl-bayes-mcmc`

Importance sampling depends on a global proposal that covers the posterior.
Markov chain Monte Carlo instead constructs a Markov chain whose stationary
distribution is the posterior. After convergence, dependent states from the
chain can be averaged to approximate :eqref:`eq_mdl-bayes-predictive`. The
Metropolis--Hastings construction needs posterior ratios between the current
and proposed states, so the unknown normalizer cancels.

The **Metropolis** rule is easiest to state for a symmetric proposal. From
$\boldsymbol\theta$, propose
$\boldsymbol\theta'\sim q(\cdot\mid\boldsymbol\theta)$. If
$\widetilde p(\boldsymbol\theta')\ge\widetilde p(\boldsymbol\theta)$, accept
the proposal; otherwise, accept it with probability $\widetilde
p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$ — otherwise stay
put. In general, with an asymmetric proposal,

$$
\alpha(\boldsymbol\theta,\boldsymbol\theta')
=\min\!\left(1,
\frac{\widetilde p(\boldsymbol\theta')
      q(\boldsymbol\theta\mid\boldsymbol\theta')}
     {\widetilde p(\boldsymbol\theta)
      q(\boldsymbol\theta'\mid\boldsymbol\theta)}\right).
$$
:eqlabel:`eq_mdl-bayes-mh`

The acceptance ratio enforces **detailed balance**: under the target
distribution, the probability flow from $\boldsymbol\theta$ to
$\boldsymbol\theta'$ equals the reverse flow. The target posterior is therefore
a stationary distribution of the chain. Convergence from an initial state also
requires irreducibility, aperiodicity, and the appropriate recurrence
conditions on general state spaces.

A rejection leaves the chain at its current state, so repeated states must be
included when computing averages. The target normalizer is unnecessary because
it cancels from the acceptance ratio.

![A Metropolis random walk on posterior contours. From the starting square, each step proposes a local move; uphill moves are always accepted (path), downhill moves only sometimes — a rejected proposal (crosses) leaves the chain in place for another step. In the long run the time the walk spends in a region is proportional to its posterior mass.](../img/mdl-prob-bayes-metropolis.svg)
:label:`fig_mdl-prob-bayes-metropolis`

The cell runs four chains from dispersed starts, using the Laplace
covariance only to scale and orient the proposal — the Metropolis
correction, not that Gaussian, determines the stationary distribution.

```{.python .input #bayesian-computation-metropolis}
def metropolis(start, proposal_chol, n_steps, seed):
    local_rng = np.random.default_rng(seed)
    current = np.array(start, dtype=float)
    current_logp = log_joint(current)
    draws = np.empty((n_steps, 2))
    accepted = 0
    for t in range(n_steps):
        proposal = current + proposal_chol @ local_rng.standard_normal(2)
        proposal_logp = log_joint(proposal)
        if np.log(local_rng.random()) < proposal_logp - current_logp:
            current, current_logp = proposal, proposal_logp
            accepted += 1
        draws[t] = current
    return draws, accepted / n_steps

starts = np.array([[-2.0, 0.0], [1.0, 0.3], [-1.0, 2.5], [1.0, 2.5]])
proposal_chol = 1.1 * np.linalg.cholesky(laplace_cov)
chains = []
for c, start in enumerate(starts):
    chain, accept_rate = metropolis(start, proposal_chol, 12000, 20 + c)
    chains.append(chain[2000:])                 # discard the initial transient
    print(f'chain {c + 1}: acceptance rate={accept_rate:.3f}')
chains = np.asarray(chains)
```

### Diagnosing Markov Chain Mixing
:label:`subsec_mdl-bayes-mcmc-diagnostics`

Exploration introduces dependence: consecutive states are correlated,
so a chain of $10{,}000$ states holds far less information than $10{,}000$
independent draws — how much less depends on how fast the walk forgets
where it was. Because the chain also starts wherever we put it, we need
evidence that it has both *forgotten its start* and *slowed to the right
pace*. Three checks answer different questions:

* **Trace plots** reveal sticking, drift, and chains that disagree —
  the failures visible to the naked eye.
* **Split $\widehat R$** compares within-chain and between-chain variation;
  values near one are necessary (not sufficient) for convergence: if
  chains from dispersed starts still disagree, none of them can be
  trusted.
* **Effective sample size** converts autocorrelation into an equivalent draw
  count: with autocorrelation time $\tau=1+2\sum_{k\ge1}\rho_k$, where
  $\rho_k$ is the lag-$k$ autocorrelation, $M$ chains of length $N$
  contain about $N_{\mathrm{eff}}=MN/\tau$ draws' worth of
  information, and the Monte Carlo standard error of a posterior mean is
  roughly the posterior standard deviation over
  $\sqrt{N_{\mathrm{eff}}}$.

The implementations below are intentionally transparent teaching versions;
production software uses rank-normalized split $\widehat R$, bulk and tail
ESS, and more careful truncation of the autocorrelation sum.

```{.python .input #bayesian-computation-mcmc-diagnostics}
def split_rhat(x):
    m, n, d = x.shape
    half = n // 2
    split = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * split.mean(axis=1).var(axis=0, ddof=1)
    variance = (half - 1) * within / half + between / half
    return np.sqrt(variance / within)

def autocorrelation(x):
    x = x - x.mean()
    n = len(x)
    spectrum = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    return acov / acov[0]

def teaching_ess(x):
    m, n, d = x.shape
    result = []
    for j in range(d):
        rho = np.mean([autocorrelation(x[c, :, j]) for c in range(m)], axis=0)
        positive_pairs = 0.0
        for lag in range(1, n - 1, 2):
            pair = rho[lag] + rho[lag + 1]
            if pair < 0:
                break
            positive_pairs += pair
        result.append(m * n / (1 + 2 * positive_pairs))
    return np.asarray(result)

d2l.use_svg_display()
flat_chains = chains.reshape(-1, 2)
rhat = split_rhat(chains)
ess = teaching_ess(chains)
mcmc_mean = flat_chains.mean(axis=0)
mcmc_sd = flat_chains.std(axis=0, ddof=1)
print('split R-hat:', rhat.round(4))
print('ESS        :', ess.round(0).astype(int))
print('MCMC mean  :', mcmc_mean.round(4),
      '  grid mean:', posterior_mean.round(4))
print('MCSE       :', (mcmc_sd / np.sqrt(ess)).round(4))

fig, axes = d2l.plt.subplots(1, 2, figsize=(10, 3.2))
for c in range(len(chains)):
    axes[0].plot(chains[c, :600, 1], linewidth=0.8, alpha=0.8)
axes[0].set(xlabel='post-warmup iteration', ylabel=r'slope $\theta_1$',
            title='four chain traces')
for c in range(len(chains)):
    axes[1].plot(autocorrelation(chains[c, :, 1])[:80], alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.7)
axes[1].set(xlabel='lag', ylabel='autocorrelation', title='serial dependence')
d2l.plt.tight_layout()
d2l.plt.show()
```

Acceptance rate alone does not establish convergence. Proposals that are too
small have high acceptance but strong serial correlation, while proposals that
are too large are usually rejected. Extending warm-up does not correct poor
mixing. Use multiple chains, inspect diagnostics for the quantities of interest,
and report effective sample size and Monte Carlo error with posterior estimates.

Metropolis is a basic MCMC method. **Gibbs sampling** applies when conditional
distributions are tractable and updates one variable or block at a time.
**Hamiltonian Monte Carlo** uses gradients of the log posterior to propose
distant states while maintaining high acceptance; the No-U-Turn Sampler adapts
the trajectory length. These methods still require multiple chains, appropriate
tuning, and convergence diagnostics.

## Deterministic Approximations
:label:`sec_mdl-bayes-approximations`

Sampling represents a posterior by draws and therefore incurs Monte Carlo cost
when expectations are estimated. Deterministic approximations instead replace
the posterior by a tractable distribution $q$. Laplace approximation and
variational inference use different criteria for choosing $q$ and draw on
optimization (:numref:`sec_mdl-maximum_likelihood`), Taylor expansion, and the
evidence lower bound.

### The Laplace Approximation at a Posterior Mode

The Laplace approximation uses a second-order expansion of the log posterior
around $\boldsymbol\theta_{\mathrm{MAP}}$. The linear term vanishes at the
mode, leaving a local quadratic form. Exponentiating this quadratic gives a
Gaussian. If $H$ is the negative Hessian at the mode,

$$
p(\boldsymbol\theta\mid\mathcal D)
\approx\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},H^{-1}),
$$
:eqlabel:`eq_mdl-bayes-laplace`

Computing this approximation requires the posterior mode and its Hessian.
For large neural networks, the Hessian is commonly approximated by a diagonal,
Kronecker-factored, or last-layer matrix to reduce computational cost.

![The Laplace approximation is a quadratic fit to the log posterior at its mode. Left: the fit matches value, slope, and curvature at the mode and nothing else. Right: after exponentiating, the implied Gaussian reproduces the peak's location and width but misses the skew — too much mass on one side, a dropped tail on the other.](../img/mdl-prob-bayes-laplace.svg)
:label:`fig_mdl-prob-bayes-laplace`

:numref:`fig_mdl-prob-bayes-laplace` shows the scope of the approximation.
It preserves the mode and local curvature but cannot represent skewness, heavy
tails, boundaries, or additional modes. In this example, the Gaussian mean is
the MAP estimate, while the exact posterior mean differs slightly because the
posterior is skewed.

### Variational Inference as Optimization

The Laplace approximation determines a Gaussian from local curvature at the
mode. Variational inference instead chooses $q_\phi$ from a tractable family
by minimizing
$D_{\mathrm{KL}}(q_\phi\|p(\cdot\mid\mathcal D))$. The unknown evidence in
this divergence is constant with respect to $\phi$, so the equivalent
optimization maximizes the **ELBO**

$$
\mathcal L(\phi)
=\mathbb E_{q_\phi}
 [\log p(\mathcal D,\boldsymbol\theta)-\log q_\phi(\boldsymbol\theta)]
\le\log p(\mathcal D)
$$
:eqlabel:`eq_mdl-bayes-vi-elbo`

Every term in this objective is computable. Optimizing a distributional
approximation in this way is central to modern approximate inference. When
$q_\phi$ is produced by a neural network, the same objective trains a
variational autoencoder; related variational bounds also appear in diffusion
models.

The direction of the KL divergence affects the approximation, as shown in
:numref:`fig_mdl-prob-bayes-kl-modes`. Reverse KL assigns an infinite penalty
when $q$ puts mass where $p$ is zero. If a unimodal family approximates a
multimodal target, its optimum may therefore concentrate on one mode and
underestimate uncertainty. Forward KL, used in moment-matching methods, instead
favors covering the target mass and may place density between modes. Standard
variational inference minimizes reverse KL and consequently tends toward the
first behavior.

![The direction of the KL divergence decides how a too-simple approximation fails. For a two-mode target, the best reverse-KL Gaussian — what variational inference optimizes — locks onto a single mode and understates uncertainty, while the best forward-KL Gaussian covers the mass of both modes at the price of putting its bulk where the target is small.](../img/mdl-prob-bayes-kl-modes.svg)
:label:`fig_mdl-prob-bayes-kl-modes`

The reparameterization estimator from
:numref:`subsec_mdl-stochastic-gradient-estimators` supplies gradients of the
ELBO. For a mean-field Gaussian
$q=\mathcal N(\mathbf m,\operatorname{diag}(\mathbf s^2))$, write
$\boldsymbol\theta=\mathbf m+\mathbf s\odot\boldsymbol\epsilon$ with
$\boldsymbol\epsilon\sim\mathcal N(\mathbf0,\mathbf I)$. Differentiation can
then pass through $\boldsymbol\theta$:

$$
\nabla_{\mathbf m}\mathcal L
=\mathbb E[\nabla_{\boldsymbol\theta}\log p(\mathcal D,\boldsymbol\theta)],
\qquad
\nabla_{\log\mathbf s}\mathcal L
=\mathbb E[\nabla_{\boldsymbol\theta}\log p
 \odot\mathbf s\odot\boldsymbol\epsilon]+\mathbf1,
$$
:eqlabel:`eq_mdl-bayes-vi-gradients`

where $+\mathbf1$ is the derivative of the Gaussian entropy. The following
NumPy implementation estimates these expectations with batches of 256 samples
and optimizes the parameters with Adam.

```{.python .input #bayesian-computation-variational}
rng_vi = np.random.default_rng(31)
m_vi = theta_map.copy()
log_s_vi = np.log(np.sqrt(np.diag(laplace_cov)))
m1 = np.zeros(2); v1 = np.zeros(2)
m2 = np.zeros(2); v2 = np.zeros(2)
for t in range(1, 1501):
    eps = rng_vi.standard_normal((256, 2))
    s_vi = np.exp(log_s_vi)
    theta_vi = m_vi + eps * s_vi
    grad_theta = grad_log_joint(theta_vi)
    grad_m = grad_theta.mean(axis=0)
    grad_log_s = (grad_theta * eps * s_vi).mean(axis=0) + 1.0
    for grad, param, first, second in [
            (grad_m, m_vi, m1, v1), (grad_log_s, log_s_vi, m2, v2)]:
        first *= 0.9
        first += 0.1 * grad
        second *= 0.999
        second += 0.001 * grad**2
        first_hat = first / (1 - 0.9**t)
        second_hat = second / (1 - 0.999**t)
        param += 0.03 * first_hat / (np.sqrt(second_hat) + 1e-8)

vi_cov = np.diag(np.exp(2 * log_s_vi))
print('method       mean                 marginal sd')
print('grid      ', posterior_mean.round(4), np.sqrt(np.diag(posterior_cov)).round(4))
print('Laplace   ', theta_map.round(4), np.sqrt(np.diag(laplace_cov)).round(4))
print('mean-field', m_vi.round(4), np.exp(log_s_vi).round(4))
```

The results exhibit both limitations. A diagonal mean-field covariance cannot
represent posterior correlation between the intercept and slope. Reverse KL
also produces a distribution that is slightly too concentrated. Full-covariance
Gaussians, normalizing flows, or mixtures can represent richer dependence at
greater computational and implementation cost. Convergence of the ELBO only
indicates that optimization has stabilized within the chosen family; it does
not establish that the family approximates the posterior well.

The following plot compares the grid reference, Metropolis draws, and the two
Gaussian approximations on the same axes.

```{.python .input #bayesian-computation-comparison-plot}
def covariance_ellipse(mean, cov, color, label):
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, 240)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    ellipse = mean[:, None] + vectors @ (2 * np.sqrt(values)[:, None] * circle)
    d2l.plt.plot(ellipse[0], ellipse[1], color=color, linewidth=2, label=label)

d2l.plt.figure(figsize=(6, 4.5))
levels = np.quantile(grid_weight, [0.70, 0.90, 0.97, 0.995])
d2l.plt.contour(B0, B1, grid_weight.reshape(B0.shape),
                levels=np.unique(levels), colors='black', linewidths=1)
d2l.plt.scatter(flat_chains[::80, 0], flat_chains[::80, 1], s=5, alpha=0.15,
                label='Metropolis draws')
covariance_ellipse(theta_map, laplace_cov, 'tab:orange', 'Laplace: 2 sd')
covariance_ellipse(m_vi, vi_cov, 'tab:blue', 'mean-field VI: 2 sd')
d2l.plt.scatter(*posterior_mean, marker='x', s=70, color='black',
                label='grid mean')
d2l.plt.xlabel(r'intercept $\theta_0$')
d2l.plt.ylabel(r'slope $\theta_1$')
d2l.plt.legend(fontsize=8)
d2l.plt.tight_layout()
d2l.plt.show()
```

The Metropolis cloud traces the exact contours, skew included; the Laplace
ellipse is centered at the MAP with roughly the right shape; the mean-field
ellipse is axis-aligned — it *cannot* tilt — and slightly small. One plot,
all four trade-offs.

## Choosing an Approximation Method
:label:`sec_mdl-bayes-decision-map`

The methods answer the same question with different failure modes.

| Method | Representation | Main diagnostic | Characteristic failure |
|---|---|---|---|
| Conjugacy / exact algebra | closed-form distribution | algebra and numerical checks | available only for special model--prior pairs |
| Grid / quadrature | weighted deterministic points | resolution and domain expansion | exponential cost in dimension |
| Importance sampling | independent weighted draws | weight ESS, largest weights, repeated runs | proposal misses or barely covers posterior mass |
| MCMC | dependent posterior draws | traces, split $\widehat R$, ESS, MCSE | poor mixing, undiscovered modes, bad geometry |
| Laplace | one local Gaussian | comparison with samples or sensitivity to mode | skewness, tails, boundaries, multiple modes |
| Variational inference | optimized tractable $q_\phi$ | ELBO, repeated starts, predictive checks | family restriction and local optima; biased uncertainty |

A robust workflow starts with the estimand: posterior mean, predictive
probability, tail event, or decision. Use exact conjugacy when available;
use a grid only as a low-dimensional check. For MCMC, run multiple chains
and budget by ESS rather than iterations. For importance sampling, inspect
weights. For Laplace and VI, test the approximation on posterior predictive
quantities and, when feasible, compare a subset against MCMC. No scalar
diagnostic proves that an unseen mode does not exist —
:numref:`fig_mdl-prob-bayes-importance` is this notebook's standing
reminder.

## Summary

* Bayesian prediction averages over the posterior and therefore represents
  parameter uncertainty under the specified likelihood, prior, and inference
  method. This does not guarantee empirical calibration; calibration must be
  checked on relevant data. In the logistic example, the plug-in predictions
  are not always more extreme than the posterior predictive ones.
* All four methods use evaluations of the unnormalized posterior; some variants
  also use its gradient. The unknown normalizer
  cancels — in importance ratios, in Metropolis acceptances, in the ELBO.
* Importance sampling reweights draws from a tractable proposal. Support is
  necessary, while tail behavior and finite moments govern variance; weight
  ESS cannot detect mass that was never sampled.
* Metropolis uses an acceptance ratio that establishes detailed balance. Under
  the convergence conditions stated above, long-run occupation follows
  posterior mass. Judge chains by mixing (traces, split $\widehat R$, ESS, MCSE),
  never by acceptance rate alone.
* Laplace uses local value and curvature at the MAP estimate, so it misses
  skewness, tails, and other modes. Variational
  inference turns integration into optimization of the ELBO and inherits
  reverse KL's mode-seeking, too-confident failure mode.

## Exercises

1. **Metropolis--Hastings acceptance.** The acceptance probability
   :eqref:`eq_mdl-bayes-mh` turns any proposal into a chain that targets
   the posterior.
    1. Show that $\alpha$ satisfies detailed balance, that is, for
       $\boldsymbol\theta'\ne\boldsymbol\theta$

       $$\widetilde p(\boldsymbol\theta)\,q(\boldsymbol\theta'\mid\boldsymbol\theta)\,\alpha(\boldsymbol\theta,\boldsymbol\theta')=\widetilde p(\boldsymbol\theta')\,q(\boldsymbol\theta\mid\boldsymbol\theta')\,\alpha(\boldsymbol\theta',\boldsymbol\theta).$$

       Show that detailed balance makes the normalized posterior a
       stationary distribution of the chain.
    1. A chain on the two states $\{1,2\}$ targets $\pi(1)=1/4$ and
       $\pi(2)=3/4$ and always proposes the other state. Write down the
       transition matrix of the Metropolis chain and check that $\pi$ is
       stationary. Which state frequencies result if the average runs only
       over accepted states, so that the repeated states after rejections
       are dropped?
    1. An independence sampler proposes $\boldsymbol\theta'\sim q$
       regardless of the current state. Show that its acceptance
       probability is $\min(1,w(\boldsymbol\theta')/w(\boldsymbol\theta))$
       with the importance weight $w$ of :eqref:`eq_mdl-bayes-importance`.
       What does the chain do after it reaches a state whose weight is far
       larger than the weights of typical proposals?
   :exlabel:`ex_mdl-bayesian-computation-mh-acceptance`

1. [code] **Proposal scale.** The `#bayesian-computation-metropolis` cell
   multiplies the Cholesky factor of `laplace_cov` by $1.1$.
    1. Sweep the multiplier over
       $\{0.05,0.2,0.5,1.1,1.7,2.4,3,10\}$ and keep the starts, the $12{,}000$
       steps per chain, and the warm-up. For each value, record the
       acceptance rate, the split $\widehat R$, and the ESS of
       `teaching_ess` divided by the $48{,}000$ log-joint evaluations of the
       four chains. Plot the ESS per evaluation against the acceptance rate.
       Over which range of acceptance rates does the ESS per evaluation stay
       within a factor of two of its maximum?
    1. For a $d$-dimensional standard Gaussian target and the proposal
       $\mathcal N(\boldsymbol\theta,(\ell^2/d)\mathbf I)$,
       :citet:`Roberts.Gelman.Gilks.1997` show that as $d\to\infty$ the
       efficiency is largest at $\ell\approx2.38$, where the acceptance rate
       is about $0.234$. The posterior here is close to its Laplace
       approximation :eqref:`eq_mdl-bayes-laplace`. Which
       multiplier does this result suggest for $d=2$? How do the best
       multiplier of your sweep and its acceptance rate compare?
    1. Which of the diagnostics of `#bayesian-computation-mcmc-diagnostics`
       flag the run with multiplier $0.05$, and how far is its posterior mean
       from the grid mean?
   :exlabel:`ex_mdl-bayesian-computation-proposal-scale`

1. [code] **Monte Carlo standard error.** A stationary AR(1) sequence
   $x_t=\phi x_{t-1}+\varepsilon_t$ with $|\phi|<1$ and independent
   $\varepsilon_t$ has the lag-$k$ autocorrelation $\rho_k=\phi^k$.
    1. Express the autocorrelation time $\tau=1+2\sum_{k\ge1}\rho_k$ of
       :numref:`subsec_mdl-bayes-mcmc-diagnostics` as a function of $\phi$.
       What is $\tau$ for $\phi=0.9$ and for $\phi=0.99$?
    1. The `#bayesian-computation-mcmc-diagnostics` cell reports an ESS of
       $4098$ for the slope from four chains of $10{,}000$ post-warm-up
       states. Which $\tau$ does this imply, and which AR(1) coefficient
       $\phi$ has this autocorrelation time? Compare $\phi$ with the lag-1
       autocorrelation of the chains.
    1. How many post-warm-up states per chain does the same sampler need to
       bring the Monte Carlo standard error of the slope's posterior mean
       down to $0.001$?
    1. Batch means estimate the Monte Carlo standard error without
       computing autocorrelations. Split each chain into batches of $b$
       consecutive states, and divide the standard deviation of all batch
       means by the square root of the number of batches. Compute this
       estimate for $b\in\{25,100,500\}$ and compare it with the standard
       error printed by the cell. What goes wrong when $b$ is not much
       larger than $\tau$?
   :exlabel:`ex_mdl-bayesian-computation-mcse`

1. **Laplace and parameterization.** A coin shows 9 heads and 4 tails.
   Under a uniform prior, the posterior of its heads probability $\theta$ is
   $\mathrm{Beta}(10,5)$, with mean $2/3$, standard deviation $0.118$, and
   central $95\%$ interval $[0.419,0.872]$.
    1. Compute the Laplace approximation :eqref:`eq_mdl-bayes-laplace` in
       the parameter $\theta$. Give its mean, its standard deviation, its
       central $95\%$ interval, and the probability it places outside
       $[0,1]$.
    1. Repeat the computation in the log-odds
       $\eta=\log(\theta/(1-\theta))$. Exercise
       :exref:`ex_mdl-maximum-likelihood-map-log-odds`b of
       :numref:`sec_mdl-maximum_likelihood` finds the posterior mode of
       $\eta$. Compute the curvature of the log posterior density of $\eta$
       at the mode, and map the central $95\%$ interval of the
       approximation back to $\theta$.
    1. Which of the two approximations is closer to the exact posterior?
       Why does the Laplace approximation depend on the parameterization,
       whereas the exact posterior probability of an event such as
       $\theta>1/2$ does not?
   :exlabel:`ex_mdl-bayesian-computation-laplace-parameterization`

1. [code] **Moderated predictions.** ● Under the Laplace approximation
   :eqref:`eq_mdl-bayes-laplace`, the logit
   $a=\mathbf z_\star^\top\boldsymbol\theta$ at a test input with
   $\mathbf z_\star=(1,x_\star)^\top$ is Gaussian with mean $\mu_a=\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}}$ and
   variance $s_a^2=\mathbf z_\star^\top H^{-1}\mathbf z_\star$. The probit
   approximation $\sigma(a)\approx\Phi(\lambda a)$ with $\lambda^2=\pi/8$,
   where $\Phi$ is the standard normal c.d.f., turns the predictive
   integral :eqref:`eq_mdl-bayes-predictive` into a closed form
   :cite:`MacKay.1992`.
    1. Show that
       $\mathbb E[\Phi(\lambda a)]=\Phi\bigl(\lambda\mu_a/\sqrt{1+\lambda^2s_a^2}\bigr)$
       for $a\sim\mathcal N(\mu_a,s_a^2)$. Conclude that the moderated
       prediction $\sigma\bigl(\mu_a/\sqrt{1+\pi s_a^2/8}\bigr)$ lies
       between $\sigma(\mu_a)$ and $1/2$.
    1. Compute the moderated predictions at $x_\star\in\{-2,0,2\}$ from
       `theta_map` and `laplace_cov`, and compare them with the plug-in
       and posterior predictive columns printed by
       `#bayesian-computation-map-predictive`.
    1. At $x_\star=0$ the posterior predictive printed by that cell lies
       farther from $1/2$ than the plug-in prediction, and at
       $x_\star=\pm2$ the two nearly agree. Explain both observations. What
       do they imply for using the Laplace approximation to temper
       overconfident predictions?
   :exlabel:`ex_mdl-bayesian-computation-moderated-predictions`

1. [code] **Predicting the weight ESS.** ● Let $w=\widetilde p/q$ be the
   importance weight of :eqref:`eq_mdl-bayes-importance`. By the law of
   large numbers, $N_{\mathrm{eff}}/N$ in :eqref:`eq_mdl-bayes-is-ess`
   approaches $(\mathbb E_q[w])^2/\mathbb E_q[w^2]$ as the number $N$ of
   draws grows.
    1. Let the normalized target be $\mathcal N(\mu,\sigma^2)$ and the
       proposal $\mathcal N(\nu,s^2)$ in one dimension, and let
       $r=s/\sigma$. Show that for $s^2>\sigma^2/2$

       $$\frac{\mathbb E_q[w^2]}{(\mathbb E_q[w])^2}=\frac{r^2}{\sqrt{2r^2-1}}\exp\!\left(\frac{(\mu-\nu)^2}{2s^2-\sigma^2}\right),$$

       and that the ratio is infinite for $s^2\le\sigma^2/2$. Which $r$
       minimizes the ratio when $\mu=\nu$?
    1. Approximate the posterior by independent Gaussians with the grid
       means and standard deviations printed by
       `#bayesian-computation-grid`, ignoring the correlation, so that the
       ratio factorizes over the coordinates. Predict the ESS of the 20,000
       prior-proposal draws of `#bayesian-computation-importance` and
       compare it with the printed value. Repeat the prediction for the
       prior proposals with $\tau=0.5$ and $\tau=10$: set `prior_var`
       $=\tau^2$ in the `#bayesian-computation-model` cell, rerun the cells
       from `#bayesian-computation-grid` through
       `#bayesian-computation-importance`, and compare the prediction with the
       measured ESS. Which factor of the formula makes a narrow
       prior a poor proposal, and which factor makes a broad prior a poor
       proposal?
    1. Under the same approximation, which coordinate makes
       $\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},0.05\,\mathbf I)$ a
       proposal with infinite weight variance? Run this proposal with 5,000
       draws for ten seeds, and record the ESS, the largest normalized
       weight, and the estimated posterior mean. Why does a run whose
       estimate looks plausible not show that the proposal is adequate?
   :exlabel:`ex_mdl-bayesian-computation-weight-ess`

1. [code] **Variational families.** ● The `#bayesian-computation-variational`
   cell fits a mean-field Gaussian by stochastic gradient ascent on the
   ELBO :eqref:`eq_mdl-bayes-vi-elbo`.
    1. Let the target be $\mathcal N(\boldsymbol\mu,\boldsymbol\Lambda^{-1})$
       with precision matrix $\boldsymbol\Lambda$. Show that the mean-field
       Gaussian that minimizes $D_{\mathrm{KL}}(q\|p)$ has mean
       $\boldsymbol\mu$ and variances $1/\Lambda_{jj}$. For two coordinates
       with correlation $\rho$, show that each fitted standard deviation is
       $\sqrt{1-\rho^2}$ times the true one. Predict the mean-field standard
       deviations from the grid standard deviations and correlation, and
       compare them with the printed values.
    1. Rerun the cell with three other seeds for `rng_vi`. How much does the
       final mean vary between seeds? Average the iterates of the last
       1,000 steps instead of reporting the final iterate, and measure the
       variation again. How much of the gap between the printed mean-field
       mean and the grid mean is approximation error, and how much is
       optimization noise?
    1. Replace the mean-field family by a full-covariance Gaussian
       $\boldsymbol\theta=\mathbf m+L\boldsymbol\epsilon$, where $L$ is
       lower triangular with diagonal entries $e^{\ell_1}$ and $e^{\ell_2}$.
       Extend :eqref:`eq_mdl-bayes-vi-gradients` to the gradients with
       respect to $\mathbf m$, $\ell_1$, $\ell_2$, and the off-diagonal entry
       of $L$. Implement them, and compare the fitted standard deviations and
       correlation with the grid values.
   :exlabel:`ex_mdl-bayesian-computation-variational-families`

1. [code] **Estimating the evidence.** ● Every method of this section avoids
   the evidence $p(\mathcal D)$. Let $q$ be a normalized proposal, let
   $w=p(\mathcal D,\boldsymbol\theta)/q(\boldsymbol\theta)$, and let
   $\hat Z_N=\frac1N\sum_{s=1}^N w(\boldsymbol\theta_s)$ for independent
   draws $\boldsymbol\theta_s\sim q$.
    1. Show that $\hat Z_N$ is an unbiased estimate of $p(\mathcal D)$ and
       that $\mathbb E[\log\hat Z_N]\le\log p(\mathcal D)$. Show that
       $\mathbb E[\log\hat Z_1]$ is the ELBO :eqref:`eq_mdl-bayes-vi-elbo`
       of $q$ and that $\mathbb E[\log\hat Z_N]$ does not decrease with $N$
       :cite:`Burda.Grosse.Salakhutdinov.2016`.
    1. The function `log_joint` omits the normalizing constant
       $-\log(2\pi\tau^2)$ of the two-dimensional prior. Add it, and compute
       $\log p(\mathcal D)$ in three ways: by summing the density over a
       grid and multiplying by the area of a grid cell, by $\hat Z_N$ with
       5,000 Laplace-proposal draws, and by integrating the Gaussian of the
       Laplace approximation, which gives
       $\log p(\mathcal D,\boldsymbol\theta_{\mathrm{MAP}})+\log2\pi-\tfrac12\log\det H$.
    1. Estimate the ELBO of three Gaussians: the Laplace approximation, the
       mean-field fit, and the full-covariance fit of problem
       :exref:`ex_mdl-bayesian-computation-variational-families`c. Use
       $10^5$ draws from each Gaussian and report the Monte Carlo standard
       error of each estimate. The gap between $\log p(\mathcal D)$ and the
       ELBO of $q$ is
       $D_{\mathrm{KL}}(q\|p(\cdot\mid\mathcal D))$ (:eqref:`eq_mdl-elbo-gap`).
       Rank the three Gaussians by this divergence. Explain where the
       Laplace Gaussian falls in this ranking.
   :exlabel:`ex_mdl-bayesian-computation-evidence`

1. [code] **A posterior with two modes.** ● Fifty observations are drawn as
   $y_i\sim\mathcal N(\theta^2,1)$ with the true value $\theta=1$, and
   $\theta$ has the prior $\mathcal N(0,4)$. The likelihood depends on
   $\theta$ only through $\theta^2$, just as the loss of a neural network
   does not change when the incoming weights, the bias, and the outgoing
   weights of a $\tanh$ hidden unit flip sign together. The posterior is
   therefore symmetric under $\theta\mapsto-\theta$.
    1. Compute the posterior on a one-dimensional grid. Report
       $\mathbb E[\theta\mid\mathcal D]$,
       $\mathbb E[\theta^2\mid\mathcal D]$, and the ratio of the posterior
       density at $\theta=0$ to its value at a mode.
    1. Show that the Gaussian that minimizes the forward divergence
       $D_{\mathrm{KL}}(p\|q)$ matches the mean and the variance of $p$.
       Where does this Gaussian put its mode for the present posterior, and
       how plausible is that point under $p$?
    1. Fit three local approximations: the Laplace approximation at the
       positive mode, a one-dimensional Gaussian fitted by the ELBO ascent of
       `#bayesian-computation-variational` started at $m=0.5$, and four
       Metropolis chains with proposal standard deviation $0.2$ started at
       $\theta\in\{0.5,1,1.5,2\}$. Then rerun the chains from
       $\theta\in\{-2,-0.5,0.5,2\}$. Which methods report a single mode?
       For each run, which of its diagnostics (split $\widehat R$ and the ESS
       for the chains, the ELBO for the variational fit) reveal the problem?
    1. Which posterior expectations does a single-mode approximation still
       estimate correctly? A network with $h$ hidden $\tanh$ units has the
       sign flip of this problem for every unit and, in addition, the
       permutations of its hidden units counted in exercise
       :exref:`ex_mdl-maximum-likelihood-relu-symmetries`a
       of :numref:`sec_mdl-maximum_likelihood`. How many equivalent copies of a
       mode does its posterior have under an isotropic Gaussian prior? What
       does this suggest about approximating the posterior of a neural
       network by one of the methods of this section?
   :exlabel:`ex_mdl-bayesian-computation-two-modes`

1. [code] **Tails of the logistic posterior.** ●● A central limit theorem
   for importance sampling needs a finite second moment of the
   weights (:numref:`sec_mdl-bayes-importance`). Problem
   :exref:`ex_mdl-bayesian-computation-weight-ess`a settles this question
   for a Gaussian target. The Laplace proposal of
   `#bayesian-computation-importance` reaches an ESS of about 4,300 of its
   5,000 draws. Throughout, $\tau^2=$ `prior_var` $=4$ is the prior
   variance of :eqref:`eq_mdl-bayes-logistic-posterior`.
    1. Fix a unit vector $\mathbf v$ and let $\boldsymbol\theta=t\mathbf v$.
       Show that the log-likelihood in
       :eqref:`eq_mdl-bayes-logistic-posterior` equals $-c(\mathbf v)t+O(1)$
       as $t\to\infty$, where $c(\mathbf v)$ is the sum of
       $|\mathbf z_i^\top\mathbf v|$ over the observations whose label
       disagrees with the sign of $\mathbf z_i^\top\mathbf v$. For which
       datasets is $c(\mathbf v)>0$ for every $\mathbf v$?
    1. Conclude that a Gaussian proposal $\mathcal N(\mathbf m,S)$ has
       $\mathbb E_q[w^2]=\infty$ when some eigenvalue of $S$ is smaller than
       $\tau^2/2$, and $\mathbb E_q[w^2]<\infty$ when every eigenvalue of
       $S$ exceeds $\tau^2/2$. Which of the two proposals of
       `#bayesian-computation-importance` has weights with finite variance?
       Compare this answer with the one that the formula of problem
       :exref:`ex_mdl-bayesian-computation-weight-ess`a gives for the Laplace
       proposal when the posterior is approximated by a Gaussian.
    1. Along each eigenvector of `laplace_cov`, in both directions, find the
       Mahalanobis distance from $\boldsymbol\theta_{\mathrm{MAP}}$ beyond
       which $\widetilde p^2/q$ exceeds its value at
       $\boldsymbol\theta_{\mathrm{MAP}}$. How likely is a single
       Laplace-proposal draw to land that far out? Reconcile your answer with
       the printed ESS of the Laplace proposal.
    1. :citet:`Geweke.1989` builds importance densities from multivariate
       Student-$t$ approximations at the posterior mode. Use a Student-$t$
       proposal with $3$ degrees of freedom, location
       $\boldsymbol\theta_{\mathrm{MAP}}$, and scale matrix `laplace_cov`.
       Show that its weights have finite variance. Compare it with the
       Laplace proposal over 20 seeds, on the ESS and on the estimate of the
       tail probability $P(\theta_1>2.5\mid\mathcal D)$. For a reference
       value, evaluate the posterior on a grid that extends beyond the upper
       end $\theta_1=2.8$ of the grid in `#bayesian-computation-grid`.
   :exlabel:`ex_mdl-bayesian-computation-posterior-tails`

1. [extended] **Approximations in higher dimension.** ●● Extend the running
   example to $d$ parameters: an intercept and $d-1$ features with
   independent standard normal entries, $n=10d$ observations whose labels
   are drawn from a true parameter with independent
   $\mathcal N(0,1.5^2/d)$ entries, and the prior
   $\mathcal N(\mathbf 0,4\mathbf I)$. No grid is available, so a long
   Metropolis run serves as the reference.
    1. For $d\in\{2,5,10,20,50\}$, compute the Laplace approximation and the
       ESS of 5,000 Laplace-proposal draws. Repeat with $n=40d$. Explain how
       $N_{\mathrm{eff}}/N$ depends on $d$ and on $n/d$.
    1. Run four Metropolis chains with the proposal $(2.38/\sqrt d)L$, where
       $L$ is the Cholesky factor of the Laplace covariance (problem
       :exref:`ex_mdl-bayesian-computation-proposal-scale`b). Measure the
       acceptance rate and the smallest ESS per log-joint evaluation over
       the coordinates as functions of $d$. Which power of $d$ describes the
       efficiency?
    1. At $d=50$ and $n=500$, give importance sampling and Metropolis the
       same budget of 20,000 log-joint evaluations, and repeat each run with
       five seeds. Measure their errors in the posterior mean against the
       reference, in units of posterior standard deviations, and compare
       them with the errors that each method's own ESS predicts. Report one
       row per method with the number of evaluations, the ESS, the predicted
       error, and the measured error. Which method would you use at this
       budget?
   :exlabel:`ex_mdl-bayesian-computation-higher-dimension`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §24.4]{.kicker}

Approximating posterior averages<br>**Bayesian computation**.
:::
:::

::: {.slide title="Why keep the posterior?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Optimization returns one $\boldsymbol\theta$; many others explain the data
about as well, and they disagree about the next prediction. Averaging over
them is the posterior predictive

$$p(y_\star\mid\mathbf x_\star,\mathcal D)
=\int p(y_\star\mid\mathbf x_\star,\boldsymbol\theta)\,
p(\boldsymbol\theta\mid\mathcal D)\,d\boldsymbol\theta.$$

- parameter uncertainty represented under the chosen model and prior
- the integral is often unavailable in closed form

::: {.d2l-note .rule}
The computational task: approximate expectations under a distribution known
only up to a normalizing constant.
:::
:::

::: {.col .fig}
@fig:mdl-prob-bayes-metropolis
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The target]{.dtitle}

[a nonconjugate posterior with a numerical reference solution]{.dsub}
:::
:::

::: {.slide title="The smallest real problem"}
[The target]{.kicker}

Bayesian logistic regression combines a sigmoid likelihood with a Gaussian
prior; the posterior is not conjugate.

$$p(\boldsymbol\theta\mid\mathcal D)\propto
\prod_i\sigma(\mathbf z_i^\top\boldsymbol\theta)^{y_i}
(1-\sigma(\mathbf z_i^\top\boldsymbol\theta))^{1-y_i}
\exp\!\left(-\tfrac{\|\boldsymbol\theta\|^2}{2\tau^2}\right).$$

All four methods use evaluations of $\log\widetilde p$ up to a constant; some
algorithms also use its gradient.

@!bayesian-computation-model
:::

::: {.slide title="A grid gives ground truth (here!)"}
[The target]{.kicker}

Two parameters allow quadrature on a $241\times241$ grid, giving a
high-accuracy numerical reference. In $d$ dimensions an $m$-point grid per
coordinate costs $m^d$, so this check does not scale.

@bayesian-computation-grid
:::

::: {.slide title="Predictive averaging in this example"}
[The target]{.kicker}

Here the posterior predictive is pulled toward $\tfrac12$ relative to the
plug-in MAP: averaging a nonlinear sigmoid $\ne$ sigmoid of the average. This
direction is specific to the example, not a universal calibration guarantee.

@bayesian-computation-map-predictive
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Importance sampling]{.dtitle}

[sample somewhere easy, reweight by right-over-wrong]{.dsub}
:::
:::

::: {.slide title="Correcting a proposal"}
[Importance sampling]{.kicker}

::: {.cols .vc}
::: {.col}
Draw from a tractable $q$, weight each draw by
$w=\widetilde p/q$; self-normalizing cancels the unknown evidence:

$$\mathbb E_p[h]=
\frac{\mathbb E_q[w\,h]}{\mathbb E_q[w]}.$$

The proposal must cover the target's relevant support, but support alone is
not enough: tails and weight moments determine variance. The weight ESS
$1/\sum_s\bar w_s^2$ **cannot see mass that was never sampled**.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-importance
:::
:::
:::

::: {.slide title="Prior vs. Laplace proposal"}
[Importance sampling]{.kicker}

A proposal aimed at the posterior (Laplace) extracts more information from
5,000 draws than the broad prior does from 20,000.

@bayesian-computation-importance
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Markov chain Monte Carlo]{.dtitle}

[walk so that time spent equals posterior mass]{.dsub}
:::
:::

::: {.slide title="The Metropolis rule"}
[MCMC]{.kicker}

::: {.cols .vc}
::: {.col}
Propose a local move; accept uphill always, downhill with probability
$\widetilde p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$ —
only *ratios*, so the normalizer cancels.

$$\alpha=\min\!\left(1,
\frac{\widetilde p(\boldsymbol\theta')\,q(\boldsymbol\theta\mid\boldsymbol\theta')}
     {\widetilde p(\boldsymbol\theta)\,q(\boldsymbol\theta'\mid\boldsymbol\theta)}\right)$$

Calibrated downhill moves give **detailed balance**: no net flow, so the
posterior is stationary. Rejected proposals repeat the state — that *is*
the weighting.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-metropolis
:::
:::
:::

::: {.slide title="Four chains, dispersed starts"}
[MCMC]{.kicker}

@!bayesian-computation-metropolis
:::

::: {.slide title="Mixing, not draw count"}
[MCMC]{.kicker}

Correlated draws carry less information. Traces show sticking; split
$\widehat R\approx1$ says chains agree; autocorrelation ESS converts
$MN$ states into equivalent independent draws, and MCSE $\approx$
sd$/\sqrt{N_{\text{eff}}}$.

@bayesian-computation-mcmc-diagnostics
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Deterministic approximations]{.dtitle}

[replace the posterior by the nearest tractable distribution]{.dsub}
:::
:::

::: {.slide title="Laplace approximation at the MAP estimate"}
[Laplace]{.kicker}

::: {.cols .vc}
::: {.col}
At a maximum the linear Taylor term vanishes: locally the log posterior is
a parabola, i.e. a Gaussian,

$$p(\boldsymbol\theta\mid\mathcal D)\approx
\mathcal N(\boldsymbol\theta_{\text{MAP}},\,H^{-1}).$$

One Hessian beyond the optimization you were doing anyway. Local by
construction: skew, tails, boundaries, other modes are invisible.
:::

::: {.col .fig}
@fig:mdl-prob-bayes-laplace
:::
:::
:::

::: {.slide title="Variational inference: optimize a distribution"}
[VI]{.kicker}

::: {.cols .vc}
::: {.col}
Choose family $q_\phi$, maximize the ELBO — integration becomes
optimization; reparameterize $\boldsymbol\theta=\mathbf m+\mathbf
s\odot\boldsymbol\epsilon$ and reuse Adam.

$$\mathcal L(\phi)=\mathbb E_{q_\phi}[\log p(\mathcal D,\boldsymbol\theta)
-\log q_\phi(\boldsymbol\theta)]\le\log p(\mathcal D)$$

Reverse KL is **mode-seeking**: a too-simple family retreats into one mode
and understates uncertainty. (This objective trains VAEs.)
:::

::: {.col .fig}
@fig:mdl-prob-bayes-kl-modes
:::
:::
:::

::: {.slide title="All four methods, one posterior"}
[Comparison]{.kicker}

Exact contours; Metropolis draws trace them, skew included; Laplace is
centered at the MAP; mean-field VI cannot tilt and is slightly narrow.

@bayesian-computation-comparison-plot
:::

::: {.slide title="Takeaways"}
[Summary]{.kicker}

- Point estimates omit parameter uncertainty; posterior averaging represents it
  conditional on the model, prior, and approximation. Calibration still needs
  empirical checking.
- Importance sampling: support is necessary; tails and moments control
  variance; ESS is blind to unseen mass.
- MCMC: time spent $=$ posterior mass; judge by mixing, never by
  acceptance rate.
- Laplace: right peak, wrong tails, one Hessian. VI: optimization replaces
  integration; mode-seeking by construction.
- No scalar diagnostic proves an unseen mode does not exist.
:::
