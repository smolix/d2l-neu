# Bayesian Computation
:label:`sec_mdl-bayesian-computation`

Maximum likelihood chooses one parameter value; MAP adds a prior but still
chooses one value. Bayesian inference keeps the posterior distribution and
uses integrals such as

$$
p(y_\star\mid\mathbf x_\star,\mathcal D)
=\int p(y_\star\mid\mathbf x_\star,\boldsymbol\theta)
       p(\boldsymbol\theta\mid\mathcal D)\,d\boldsymbol\theta.
$$
:eqlabel:`eq_mdl-bayes-predictive`

Those integrals are rarely available in closed form. **Bayesian computation**
is the collection of numerical strategies for approximating them. This
notebook develops four: importance sampling, Markov chain Monte Carlo (MCMC),
the Laplace approximation, and variational inference. One small Bayesian
logistic-regression problem runs through the whole notebook. Because it has
only two parameters, a dense grid supplies a reference posterior against which
all four approximations can be audited.

The goal is a computational bridge for machine-learning readers, not a survey
of Bayesian statistics. Conjugate posteriors are already treated in
:numref:`sec_mdl-distributions`; MAP and the ELBO are developed in
:numref:`sec_mdl-maximum_likelihood`; differentiation through random variables
is in :numref:`subsec_mdl-stochastic-gradient-estimators`. Standard broader
references are :citet:`Bishop.2006` and :citet:`Murphy.2022`.

All computation is plain NumPy. Matplotlib is used only for two diagnostic
figures.

```{.python .input #bayesian-computation-imports}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

## The Target: A Posterior Distribution
:label:`sec_mdl-bayes-target`

### A Nonconjugate Running Example

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

where $\mathbf z_i=(1,x_i)^\top$. The normalizer—the **evidence**
$p(\mathcal D)$—is an integral over $\boldsymbol\theta$. It is not needed to
compare two parameter values. Posterior expectations are defined under the
normalized distribution, but several methods below account for the normalizer
implicitly rather than evaluating it.

The cell creates a small dataset and implements the log of the unnormalized
posterior. `logaddexp` evaluates
$\log(1+e^z)$ without overflowing.

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

### A Grid as a Reference, Not a General Algorithm

On a uniform two-dimensional grid, exponentiate stabilized log posterior
values and normalize their sum. This is deterministic quadrature: excellent
for checking a two-parameter example, impossible for a neural network. With
$m$ points per coordinate, a $d$-dimensional grid has $m^d$ points—the same
curse of dimensionality that motivated Monte Carlo integration in
:numref:`sec_mdl-integral_calculus`.

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

The posterior mean is a weighted average, not the most likely point. To make
the distinction visible, compute the MAP by Newton iteration. The negative
Hessian of the log posterior is

$$
H(\boldsymbol\theta)
=Z^\top\operatorname{diag}(p_i(1-p_i))Z+\tau^{-2}I,
$$

which is positive definite because of the Gaussian prior.

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

The posterior predictive averages a nonlinear probability over parameter
uncertainty; plugging the MAP into the sigmoid is a different operation. The
two approach one another as the posterior concentrates, but they need not
match at finite data.

## Importance Sampling: Correcting a Proposal
:label:`sec_mdl-bayes-importance`

Suppose we can sample from a normalized proposal $q(\boldsymbol\theta)$ and
evaluate both $q$ and the unnormalized target
$\widetilde p(\boldsymbol\theta)=p(\mathcal D,\boldsymbol\theta)$. Then

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

Replacing both expectations by sample averages gives **self-normalized
importance sampling**. It is consistent but generally biased at finite sample
size because it is a ratio of random quantities. The computation belongs in
log space, followed by max subtraction, just like log-sum-exp.

The proposal must cover every region carrying posterior mass. A proposal that
is too narrow can miss a mode completely; a very broad proposal spends nearly
all its draws where posterior weight is negligible. A useful diagnostic is the
weight **effective sample size**

$$
N_{\mathrm{eff}}=\frac{1}{\sum_{s=1}^N\bar w_s^2},
\qquad
\bar w_s=\frac{w_s}{\sum_r w_r}.
$$
:eqlabel:`eq_mdl-bayes-is-ess`

It lies between $1$ and $N$ and measures weight concentration, not the number
of independent data points or a guarantee of accuracy.

We compare two Gaussian proposals: the broad prior and the local Gaussian
obtained from the MAP and inverse Hessian. The latter is the Laplace
approximation developed formally below.

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

The prior proposal needs four times as many draws yet has a much smaller ESS.
The Laplace proposal works well here because this posterior is unimodal and
close to Gaussian. Importance sampling degrades rapidly with dimension or
mode mismatch: the variance of the weights, not the nominal draw count, sets
the usable information.

## Markov Chain Monte Carlo
:label:`sec_mdl-bayes-mcmc`

Importance sampling uses independent proposal draws but may waste most of
them. MCMC instead constructs a dependent sequence whose stationary
distribution is the posterior. The basic **Metropolis** algorithm
:cite:`Murphy.2022` proposes
$\boldsymbol\theta'\sim q(\cdot\mid\boldsymbol\theta)$ and accepts it with
probability

$$
\alpha(\boldsymbol\theta,\boldsymbol\theta')
=\min\!\left(1,
\frac{\widetilde p(\boldsymbol\theta')
      q(\boldsymbol\theta\mid\boldsymbol\theta')}
     {\widetilde p(\boldsymbol\theta)
      q(\boldsymbol\theta'\mid\boldsymbol\theta)}\right).
$$
:eqlabel:`eq_mdl-bayes-mh`

This acceptance rule enforces detailed balance, making the target stationary.
For a symmetric random-walk proposal, the two $q$ terms cancel and the unknown
posterior normalizer cancels with them. Rejected proposals repeat the current
state; those repeats are part of the chain, not samples to discard.

The cell runs four chains from dispersed starts. The proposal uses the
Laplace covariance only as a scale and orientation; the Metropolis correction,
not that Gaussian approximation, determines the stationary distribution.

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

### Diagnostics: Mixing, Not Just Draw Count

A chain of $10{,}000$ highly correlated states does not contain the same
information as $10{,}000$ independent posterior draws. Three checks answer
different questions:

* **Trace plots** reveal sticking, drift, and differences between chains.
* **Split $\widehat R$** compares within-chain with between-chain variation;
  values close to one are necessary but not sufficient evidence of convergence.
* **Effective sample size** uses the autocorrelation time
  $\tau=1+2\sum_{k\ge1}\rho_k$ and reports $MN/\tau$ effective draws from
  $M$ chains of length $N$. A Monte Carlo standard error is then approximately
  posterior standard deviation divided by $\sqrt{N_{\mathrm{eff}}}$.

The implementations below are intentionally transparent teaching versions.
Production software uses rank-normalized split $\widehat R$, bulk and tail ESS,
and more careful truncation of the autocorrelation sum.

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

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for c in range(len(chains)):
    axes[0].plot(chains[c, :600, 1], linewidth=0.8, alpha=0.8)
axes[0].set(xlabel='post-warmup iteration', ylabel=r'slope $\theta_1$',
            title='four chain traces')
for c in range(len(chains)):
    axes[1].plot(autocorrelation(chains[c, :, 1])[:80], alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.7)
axes[1].set(xlabel='lag', ylabel='autocorrelation', title='serial dependence')
plt.tight_layout()
plt.show()
```

Acceptance rate alone is not a convergence diagnostic. Very small proposals
accept frequently but move slowly; huge proposals are mostly rejected. Nor
does discarding a longer “burn-in” repair poor mixing. Run multiple chains,
inspect the difficult posterior quantities, and report ESS and Monte Carlo
error with posterior estimates.

Metropolis is the smallest useful MCMC algorithm, not the last word. **Gibbs
sampling** draws from tractable conditional distributions one block at a time.
**Hamiltonian Monte Carlo** uses gradients of the log posterior to make long,
directed proposals; the No-U-Turn Sampler adapts their length. These methods
can mix much faster in correlated high dimensions, but require the same
workflow: multiple chains, geometry-aware tuning, and diagnostics.

## Deterministic Approximations
:label:`sec_mdl-bayes-approximations`

Sampling represents the posterior by draws. Deterministic methods instead
choose a tractable distribution $q(\boldsymbol\theta)$.

### Laplace: Local Gaussian Curvature

The **Laplace approximation** expands the log posterior to second order around
the MAP. If $H$ is the negative Hessian there,

$$
p(\boldsymbol\theta\mid\mathcal D)
\approx\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},H^{-1}).
$$
:eqlabel:`eq_mdl-bayes-laplace`

It is cheap once optimization has supplied the MAP and curvature. It captures
local parameter correlation but inherits every limitation of a local quadratic
model: skewness, heavy tails, boundaries, and other modes are missed. Here the
Laplace mean is the MAP, whereas the grid posterior mean is shifted by the
posterior's mild skew.

### Variational Inference: Optimize a Distribution

**Variational inference** chooses a family $q_\phi$ and minimizes
$D_{\mathrm{KL}}(q_\|\,p)$. Since the evidence is constant in $\phi$, this is
equivalent to maximizing the ELBO

$$
\mathcal L(\phi)
=\mathbb E_{q_\phi}
 [\log p(\mathcal D,\boldsymbol\theta)-\log q_\phi(\boldsymbol\theta)]
\le\log p(\mathcal D).
$$
:eqlabel:`eq_mdl-bayes-vi-elbo`

Take a mean-field Gaussian
$q=\mathcal N(\mathbf m,\operatorname{diag}(\mathbf s^2))$ and
reparameterize $\boldsymbol\theta=\mathbf m+\mathbf s\odot\boldsymbol\epsilon$.
For this model the gradient of the log joint is available analytically. The
pathwise gradients are

$$
\nabla_{\mathbf m}\mathcal L
=\mathbb E[\nabla_{\boldsymbol\theta}\log p(\mathcal D,\boldsymbol\theta)],
\qquad
\nabla_{\log\mathbf s}\mathcal L
=\mathbb E[\nabla_{\boldsymbol\theta}\log p
 \odot\mathbf s\odot\boldsymbol\epsilon]+\mathbf1,
$$

where the $+\mathbf1$ is the Gaussian entropy derivative. The cell optimizes
these estimates with a small Adam loop written in NumPy.

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

Mean-field VI cannot represent the grid posterior's negative intercept--slope
correlation. Reverse KL also tends to prefer a concentrated approximation over
placing mass in low-posterior regions. A richer covariance, normalizing flow,
or mixture can improve the fit at additional optimization and implementation
cost. A stabilized ELBO may indicate that one optimization run has settled; it
does not establish global optimality or adequacy of the variational family.

The final picture overlays the two Gaussian approximations on the grid
posterior and a thinned set of Metropolis draws.

```{.python .input #bayesian-computation-comparison-plot}
def covariance_ellipse(mean, cov, color, label):
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, 240)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    ellipse = mean[:, None] + vectors @ (2 * np.sqrt(values)[:, None] * circle)
    plt.plot(ellipse[0], ellipse[1], color=color, linewidth=2, label=label)

plt.figure(figsize=(6, 4.5))
levels = np.quantile(grid_weight, [0.70, 0.90, 0.97, 0.995])
plt.contour(B0, B1, grid_weight.reshape(B0.shape), levels=np.unique(levels),
            colors='black', linewidths=1)
plt.scatter(flat_chains[::80, 0], flat_chains[::80, 1], s=5, alpha=0.15,
            label='Metropolis draws')
covariance_ellipse(theta_map, laplace_cov, 'tab:orange', 'Laplace: 2 sd')
covariance_ellipse(m_vi, vi_cov, 'tab:blue', 'mean-field VI: 2 sd')
plt.scatter(*posterior_mean, marker='x', s=70, color='black', label='grid mean')
plt.xlabel(r'intercept $\theta_0$')
plt.ylabel(r'slope $\theta_1$')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## A Practical Decision Map
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
probability, tail event, or decision. Use exact conjugacy when available; use a
grid only as a low-dimensional check. For MCMC, run multiple chains and budget
by ESS rather than iterations. For importance sampling, inspect weights. For
Laplace and VI, test the approximation on posterior predictive quantities and,
when feasible, compare a subset with MCMC. No scalar diagnostic proves that an
unseen mode does not exist.

One notebook is enough for this computational bridge because the shared target
and reference grid make the methods comparable. It is not enough for a full
Bayesian curriculum. Hierarchical models, Gibbs derivations, HMC/NUTS
implementation, discrete latent-variable samplers, sequential Monte Carlo,
Bayesian model comparison, and prior/posterior predictive checking each warrant
more space once the book needs them operationally.

## Summary

* Bayesian inference averages over the posterior; MAP is a mode, not a
  substitute for posterior integration or posterior prediction.
* Importance sampling corrects a proposal with density ratios. Its nominal
  sample count can be misleading, so inspect normalized weights and weight ESS.
* Metropolis constructs dependent posterior draws without knowing the evidence.
  Multiple chains, split $\widehat R$, autocorrelation ESS, and MCSE diagnose
  complementary aspects of sampling quality; acceptance rate alone does not.
* Laplace uses inverse curvature at the MAP and is local. Variational inference
  optimizes a tractable distribution through the ELBO and is limited by both
  its family and its optimizer.
* Approximation quality is estimand-specific. Validate predictive quantities
  and uncertainty, not only an objective value or a parameter mean.

## Exercises

1. Change the prior standard deviation from $2$ to $0.5$ and $10$. Compare the
   MAP, posterior mean, covariance, posterior predictive, and importance-sampling
   ESS. Which quantities are most sensitive at this sample size?
2. Deliberately use an importance proposal with covariance
   $0.05I$. Run ten seeds. Although this Gaussian has full mathematical support,
   explain why a plausible estimate in one run does not repair its poor
   finite-sample coverage of the posterior.
3. Sweep the Metropolis proposal multiplier over
   $\{0.05,0.2,0.5,1.1,3,10\}$. Plot acceptance rate against ESS per log-joint
   evaluation and explain why maximizing acceptance is the wrong objective.
4. Replace the mean-field variational family by a full-covariance Gaussian
   parameterized by a Cholesky factor. Derive a valid parameterization with
   positive diagonal and compare the fitted correlation with the grid value.
5. Create a bimodal one-dimensional posterior by using a likelihood invariant
   under $\theta\mapsto-\theta$. Show how a local Laplace approximation,
   mean-field reverse-KL optimization, and poorly initialized MCMC can each
   report only one mode. Which diagnostics reveal the problem, and which do not?
