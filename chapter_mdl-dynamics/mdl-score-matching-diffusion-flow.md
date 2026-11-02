# Score Matching and Diffusion Models
:label:`sec_mdl-score-matching-diffusion-flow`

The preceding sections developed ODE solvers
(:numref:`sec_mdl-odes-solvers`), forward noising SDEs
(:numref:`sec_mdl-sdes`), and the role of the score $\nabla \log p_t$ in
probability flow and time reversal
(:numref:`sec_mdl-fokker-planck-probability-flow`). This section studies how to
estimate a score or velocity field by regression and then sample by integrating
the resulting ODE or SDE. We develop score matching and its denoising form,
interpret DDPM
:cite:`ho2020denoising` as a discretized variance-preserving SDE, derive
Langevin sampling, DDIM, and guidance from the same score calculus, and then
develop flow matching and rectified flow as methods that prescribe a
noise-to-data path, and relate straight paths to kinetic energy in optimal
transport :cite:`song2021score,Lipman.Chen.BenHamu.ea.2022`.

The principal statistical device is conditional expectation. A marginal score
or velocity may be unavailable directly but can be expressed as the
conditional expectation of a tractable per-sample quantity. Least-squares
regression against that quantity recovers its conditional mean. This identity
supports both denoising score matching and conditional flow matching.

We lean on the Fokker–Planck equation and the probability-flow ODE
(:numref:`sec_mdl-fokker-planck`, :numref:`sec_mdl-probability-flow-ode`), the
Ornstein–Uhlenbeck process (:numref:`sec_mdl-ornstein-uhlenbeck`), Euler and
Euler–Maruyama steps (:numref:`sec_mdl-euler-runge-kutta`,
:numref:`sec_mdl-euler-maruyama`), and the divergences of
:numref:`sec_mdl-divergences-distances` (Fisher divergence via
:numref:`sec_mdl-fisher-divergence`, optimal transport via
:numref:`sec_mdl-optimal-transport`). The numerical examples include two short
training loops (a one-dimensional score network in NumPy and a
two-dimensional flow-matching model, the latter retrained once more to measure
reflow) plus closed-form simulations for the remaining examples.

```{.python .input #score-matching-diffusion-flow-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #score-matching-diffusion-flow-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #score-matching-diffusion-flow-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #score-matching-diffusion-flow-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## Learning the Score
:label:`sec_mdl-score-matching`

### A Normalizer-Free Objective

Fitting a density $p_{\boldsymbol{\theta}}$ to data by maximum likelihood
requires evaluating its normalizing constant, which for a neural-network
energy-based model is a $d$-dimensional integral with no closed form. The
**score** $\nabla_{\mathbf{x}} \log p_{\boldsymbol{\theta}}$ is independent of that
constant (:eqref:`eq_mdl-score-def`; re-derived for energy-based models in
:numref:`sec_mdl-score-function`). Objectives and samplers formulated only in
terms of scores therefore need not evaluate that constant. Instead of matching densities, match
score fields: take a model
$\mathbf{s}_{\boldsymbol{\theta}} : \mathbb{R}^d \to \mathbb{R}^d$ (any vector
field, e.g. a neural network; it need not be a gradient) and minimize the
Fisher divergence between model and data
(:numref:`sec_mdl-fisher-divergence`),

$$
J_{\mathrm{ESM}}(\boldsymbol{\theta})
= \tfrac{1}{2}\, \mathbb{E}_{\mathbf{x} \sim p}
\left[\, \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla \log p(\mathbf{x}) \right\|^2 \right],
$$
:eqlabel:`eq_mdl-esm-objective`

called **explicit score matching**. This removes the normalizer, but the
objective still contains $\nabla \log p$, the score
of the *data* distribution, which is exactly what we do not know. Hyvärinen's
insight is that an integration by parts removes it
:cite:`Hyvarinen.2005`.

**Proposition (Hyvärinen's identity).** *Let $p$ be a smooth positive density
and $\mathbf{s}_{\boldsymbol{\theta}}$ a smooth vector field, with
$p(\mathbf{x})\, \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \to \mathbf{0}$
as $\|\mathbf{x}\| \to \infty$ and all the expectations below finite. Then*

$$
J_{\mathrm{ESM}}(\boldsymbol{\theta})
= \mathbb{E}_{\mathbf{x} \sim p}
\left[\, \tfrac{1}{2} \| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \|^2
+ \nabla \cdot \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \right] + C,
$$
:eqlabel:`eq_mdl-hyvarinen`

*where $\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}} = \sum_i \partial s_i / \partial x_i$
is the divergence and $C$ does not depend on $\boldsymbol{\theta}$.*

**Proof.** Expand the square in :eqref:`eq_mdl-esm-objective`: the term
$\tfrac12 \mathbb{E}\|\nabla \log p\|^2$ is the constant $C$, the term
$\tfrac12 \mathbb{E}\|\mathbf{s}_{\boldsymbol{\theta}}\|^2$ appears verbatim in
:eqref:`eq_mdl-hyvarinen`, and it remains to transform the cross term. In one dimension,

$$
-\mathbb{E}_{x \sim p}\left[ s_{\boldsymbol{\theta}}(x)\, (\log p)'(x) \right]
= -\int s_{\boldsymbol{\theta}}(x)\, \frac{p'(x)}{p(x)}\, p(x)\, dx
= -\int s_{\boldsymbol{\theta}}(x)\, p'(x)\, dx,
$$

and integrating by parts with the boundary term
$\left[ s_{\boldsymbol{\theta}} \, p \right]_{-\infty}^{\infty} = 0$ leaves
$+\int s_{\boldsymbol{\theta}}'(x)\, p(x)\, dx = \mathbb{E}_p[s_{\boldsymbol{\theta}}']$.
This step replaces the unknown score with a derivative of the *model*. In $d$
dimensions, apply the same one-dimensional step to each coordinate $i$ (with
$s_i$ in place of $s_{\boldsymbol{\theta}}$ and $\partial_i p$ in place of
$p'$, integrating coordinate-wise, which Fubini justifies under the stated
integrability) and sum: the cross term becomes
$\mathbb{E}_p[\sum_i \partial_i s_i] = \mathbb{E}_p[\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}}]$.
$\blacksquare$

Every quantity in :eqref:`eq_mdl-hyvarinen` is an expectation under $p$ of
something we can evaluate, so we can minimize it from samples alone, with no
$Z_{\boldsymbol{\theta}}$ and no $\nabla \log p$; the right-hand side is called
**implicit score matching**. The two terms have complementary effects. The
$\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}}$ term favors score fields with
negative divergence at the samples, as for $-\nabla E$ near a minimum, while
$\tfrac12\|\mathbf{s}_{\boldsymbol{\theta}}\|^2$ penalizes fields of unbounded
magnitude.

Implicit score matching is expensive in high dimensions. The divergence is
the trace of the Jacobian,
$\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}} = \operatorname{tr}\, (\partial \mathbf{s}_{\boldsymbol{\theta}} / \partial \mathbf{x})$,
and computing it exactly generally requires derivative work that scales
linearly with $d$ (for example, one reverse-mode pass per Jacobian row)
(:numref:`sec_mdl-matrix-calculus-autodiff`): the same trace bottleneck that
afflicts continuous normalizing flows
(:numref:`sec_mdl-continuous-normalizing-flows`), and just as there, Hutchinson
trace estimates only trade compute for variance. For images, $d$ is in the
millions. Denoising score matching avoids this trace computation.

### Denoising Score Matching
:label:`sec_mdl-denoising-score-matching`

Score models become practical by matching the score of *Gaussian-blurred*
data rather than the clean data distribution
:cite:`Vincent.2011`. Perturb each sample with Gaussian noise of scale
$\sigma$:

$$
\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}, \qquad
\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I), \qquad
\textrm{i.e.}\quad
p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}};\, \mathbf{x},\, \sigma^2 I).
$$

The noised marginal $p_\sigma(\tilde{\mathbf{x}}) = \int p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\, p(\mathbf{x})\, d\mathbf{x}$
is the data density convolved with a Gaussian and approaches $p$ as
$\sigma$ decreases. Its score is still intractable, but the score of the
*conditional* follows directly by taking $\log$ of the Gaussian density:

$$
\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})
= \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}
= -\frac{\boldsymbol{\epsilon}}{\sigma}.
$$
:eqlabel:`eq_mdl-dsm-target`

This vector points from the noisy observation toward the corresponding clean
sample. **Denoising score matching** (DSM) uses it as a regression target:

$$
J_{\mathrm{DSM}}(\boldsymbol{\theta})
= \mathbb{E}_{\mathbf{x} \sim p,\ \tilde{\mathbf{x}} \sim p_\sigma(\cdot \mid \mathbf{x})}
\left[\, \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})
- \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \right\|^2 \right].
$$
:eqlabel:`eq_mdl-dsm-loss`

The regression target depends on the sampled clean point, whereas the desired
score depends only on the noisy observation. Least squares connects them by
fitting the target's conditional mean. The following lemma states that fact;
the flow-matching argument will use it again with a velocity target.

**Lemma (regression to the conditional mean).** *Let $(X, Y)$ be jointly
distributed with $\mathbb{E}\|Y\|^2 < \infty$ and let
$\mathbf{m}(X) = \mathbb{E}[Y \mid X]$. Then for every measurable
$\mathbf{v}$,*

$$
\mathbb{E} \left\| \mathbf{v}(X) - Y \right\|^2
= \mathbb{E} \left\| \mathbf{v}(X) - \mathbf{m}(X) \right\|^2
+ \mathbb{E} \left\| Y - \mathbf{m}(X) \right\|^2.
$$
:eqlabel:`eq_mdl-regression-lemma`

*The second term does not involve $\mathbf{v}$: minimizing a least-squares loss
against the noisy target $Y$ is the same problem as minimizing it against the
conditional mean $\mathbf{m}(X)$, up to an additive constant.*

**Proof.** Insert $\pm\mathbf{m}(X)$ and expand. The cross term is
$2\, \mathbb{E}\left[ (\mathbf{v}(X) - \mathbf{m}(X))^\top (\mathbf{m}(X) - Y) \right]$;
conditioning on $X$ (the tower rule,
:numref:`sec_mdl-random_variables`) and using
$\mathbb{E}[\mathbf{m}(X) - Y \mid X] = \mathbf{0}$ makes it vanish. $\blacksquare$

**Proposition (Vincent's theorem).** *Under the conditions above, with
expectations finite,*

$$
J_{\mathrm{DSM}}(\boldsymbol{\theta})
= \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma}
\left[\, \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla \log p_\sigma(\tilde{\mathbf{x}}) \right\|^2 \right] + C,
$$

*with $C$ independent of $\boldsymbol{\theta}$: denoising score matching and
explicit score matching on the noised marginal (the right-hand side is twice
:eqref:`eq_mdl-esm-objective` with $p_\sigma$ in place of $p$) have the same
minimizers and, up to that overall factor of two, the same gradients.*

**Proof.** Apply the lemma with $X = \tilde{\mathbf{x}}$ and
$Y = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$.
It remains to identify the conditional mean, and *the marginal score is the
posterior mean of the conditional scores*:

$$
\mathbb{E}\left[ Y \mid \tilde{\mathbf{x}} \right]
= \int \frac{\nabla_{\tilde{\mathbf{x}}}\, p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}
\; \frac{p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\, p(\mathbf{x})}{p_\sigma(\tilde{\mathbf{x}})}\, d\mathbf{x}
= \frac{\nabla_{\tilde{\mathbf{x}}} \int p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\, p(\mathbf{x})\, d\mathbf{x}}{p_\sigma(\tilde{\mathbf{x}})}
= \nabla \log p_\sigma(\tilde{\mathbf{x}}),
$$

where the first equality writes out the posterior
$p(\mathbf{x} \mid \tilde{\mathbf{x}})$ by Bayes' rule and the second swaps the
gradient with the integral. $\blacksquare$

The proof does not require a Gaussian kernel: any smooth noising kernel
satisfying the stated conditions works. Gaussian noise is convenient because
its conditional score :eqref:`eq_mdl-dsm-target` is linear in the noise. By
:eqref:`eq_mdl-dsm-target`, predicting the score and predicting the noise
$\boldsymbol{\epsilon}$ differ only by the factor $-1/\sigma$; this is the
$\boldsymbol{\epsilon}$-prediction parameterization used in diffusion models.
Two corollaries follow.

* **Tweedie's formula** :cite:`Efron.2011`. Rearranging
  $\mathbb{E}[(\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2 \mid \tilde{\mathbf{x}}] = \nabla \log p_\sigma(\tilde{\mathbf{x}})$
  gives
  $\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2\, \nabla \log p_\sigma(\tilde{\mathbf{x}})$:
  *the optimal denoiser adds a score-based correction*
  (:numref:`fig_mdl-dyn-tweedie`). Under squared error and Gaussian
  corruption, the posterior-mean denoiser is determined by the marginal score.
* **The population loss need not go to zero.** By :eqref:`eq_mdl-regression-lemma`, the DSM
  loss at the optimum equals
  $\mathbb{E}\|Y - \mathbf{m}(X)\|^2$, the average posterior variance
  of the conditional score: many clean points $\mathbf{x}$ explain the same
  $\tilde{\mathbf{x}}$, and no network can resolve which one produced it. A
  resulting Bayes risk depends on the noise level, weighting, and target
  parameterization. An observed training loss also contains approximation and
  optimization error, so it should be compared with an estimated Bayes-risk
  floor rather than with zero.

![Tweedie's formula. A noisy observation $\tilde{x}$ lies in the low-density valley of the smoothed mixture $p_\sigma$. Its posterior clean-data distribution is bimodal but asymmetric, and the score correction $\tilde{x} + \sigma^2 \nabla \log p_\sigma(\tilde{x})$ equals the posterior mean $\hat{x}_0$, the optimal denoiser.](../img/mdl-dyn-tweedie.svg)
:label:`fig_mdl-dyn-tweedie`

### A Score Network in One Dimension

Take the bimodal mixture
$p = \tfrac12 \mathcal{N}(-2, 0.5^2) + \tfrac12 \mathcal{N}(2, 0.5^2)$, noise
scale $\sigma = 0.5$, and fit a tiny multilayer perceptron
$s_{\boldsymbol{\theta}} : \mathbb{R} \to \mathbb{R}$ by minimizing
:eqref:`eq_mdl-dsm-loss`: the inputs are noised samples, the regression
targets are $-\epsilon/\sigma$, and training requires samples rather than an
analytic expression for the true density. Because the noised marginal is
again a Gaussian mixture (variance
$0.5^2 + \sigma^2 = 0.5$ per component), we have the analytic
$\nabla \log p_\sigma$ against which to evaluate the result. The network is
small enough that we write its forward pass, its backward pass
(:numref:`sec_mdl-matrix-calculus-autodiff`), and an Adam update directly in
plain NumPy, without an automatic-differentiation framework.

```{.python .input #score-matching-diffusion-flow-dsm-train}
rng = np.random.default_rng(7)
n, sigma = 4096, 0.5
x = rng.normal(4.0 * rng.integers(0, 2, n) - 2.0, 0.5)    # x ~ p, the mixture

def mixture_score(q, var, means=(-2.0, 2.0)):             # analytic score
    w = np.stack([np.exp(-(q - m)**2 / (2 * var)) for m in means])
    return (w * np.stack([(m - q) / var for m in means])).sum(0) / w.sum(0)

# A 1 -> 32 -> 1 tanh network, with hand-written backprop and Adam updates
W1, b1 = rng.normal(size=(1, 32)), np.zeros(32)
W2, b2 = rng.normal(size=(32, 1)) / np.sqrt(32), np.zeros(1)
params = [W1, b1, W2, b2]
mom = [np.zeros_like(p) for p in params]
vel = [np.zeros_like(p) for p in params]
for step in range(2000):
    eps = rng.standard_normal(n)                          # fresh noise per step
    xt, y = (x + sigma * eps)[:, None], (-eps / sigma)[:, None]
    H = np.tanh(xt @ W1 + b1)                             # forward pass
    S = H @ W2 + b2
    G = 2 * (S - y) / n                                   # backward pass
    GH = (G @ W2.T) * (1 - H**2)
    grads = [xt.T @ GH, GH.sum(0), H.T @ G, G.sum(0)]
    for p, g, m, v in zip(params, grads, mom, vel):       # Adam updates
        m[:] = 0.9 * m + 0.1 * g
        v[:] = 0.999 * v + 0.001 * g * g
        p -= 1e-2 * m / (np.sqrt(v) + 1e-8)
loss = ((S - y)**2).mean()
bayes_ref = ((mixture_score(x + sigma * eps, 0.5) + eps / sigma)**2).mean()
grid = np.linspace(-4, 4, 201)
s_hat = (np.tanh(grid[:, None] @ W1 + b1) @ W2 + b2)[:, 0]
print(f'DSM loss {loss:.3f} vs estimated Bayes risk {bayes_ref:.3f}; '
      f'max |s_theta - score| on [-4, 4]: {np.abs(s_hat - mixture_score(grid, 0.5)).max():.3f}')
d2l.plot(grid, [mixture_score(grid, 0.5), s_hat], 'x', 'score',
         legend=['analytic score of p_sigma', 'learned s_theta'])
```

The learned field tracks the analytic score across both modes and the
low-density valley between them: the largest gap on $[-4, 4]$ is about
$0.2$, on a curve whose values span $\pm 4$ (the smoothed score is
$s(x) = 4\tanh(4x) - 2x$, extremal at the interval ends). And the printout is consistent with the
regression lemma: the final DSM loss ($\approx 2.04$) is close to the
estimated Bayes risk $\mathbb{E}\|Y - \mathbf{m}(X)\|^2$ ($\approx 2.04$,
estimated on this finite sample with the analytic score). Their agreement is
consistent with little reducible error remaining in this particular fit.

## Score-Based Diffusion Models
:label:`sec_mdl-score-based-generative-modeling`

### Scores Across Noise Levels

A single noise scale $\sigma$ creates a trade-off. Small $\sigma$ makes
$p_\sigma \approx p$, but noised samples then seldom enter low-density regions,
so the learned score is poorly constrained where a sampler initialized from
random noise may need it. Large $\sigma$ provides broader coverage but
estimates the score of an over-smoothed density. The standard solution
:cite:`song2019generative,song2021score`: learn the score at *every* noise level
along a forward process that moves the data toward a tractable reference, by making the
network noise-conditional, $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)$
(:numref:`fig_mdl-dyn-noising-denoising`, the two-dimensional companion of
the one-dimensional density movie in :numref:`fig_mdl-dyn-forward-reverse`).
:numref:`sec_mdl-sdes` gave us the two standard forward processes:

* the **variance-exploding (VE)** SDE
  $d\mathbf{X} = \sqrt{\tfrac{d}{dt}\sigma^2(t)}\; d\mathbf{W}$, with
  $\sigma^2(t)$ increasing and differentiable so that the square root exists,
  which adds noise without shrinking the data:
  $\mathbf{x}_t = \mathbf{x}_0 + \sigma(t) \boldsymbol{\epsilon}$ (the
  continuous limit of Song & Ermon's noise ladder), with conditional terminal
  law $\mathcal{N}(\mathbf{x}_0, \sigma_{\max}^2 I)$ rather than a standard
  Gaussian;
* the **variance-preserving (VP)** SDE
  $d\mathbf{X} = -\tfrac12 \beta(t)\, \mathbf{X}\, dt + \sqrt{\beta(t)}\; d\mathbf{W}$
  of :eqref:`eq_mdl-sde-vp-sde`, an Ornstein–Uhlenbeck process with a
  time-dependent rate (:numref:`sec_mdl-ornstein-uhlenbeck`), which shrinks
  the signal as it adds noise so that unit-variance data keeps unit variance
  for *all* $t$ (shown for the discrete chain below).

![A two-moons distribution under the VP forward process at $t = 0$, $t = 0.7$, and $t = T$ (top row, left to right), approaching an isotropic Gaussian. The bottom row traverses the same marginals in reverse, right to left; short arrows show the score field used to recover the data distribution.](../img/mdl-dyn-noising-denoising.svg)
:label:`fig_mdl-dyn-noising-denoising`

In both cases the transition kernel $p_t(\mathbf{x}_t \mid \mathbf{x}_0)$ is an
explicit Gaussian, so the DSM machinery applies verbatim at every $t$: the
training loss is the noise-conditional DSM objective

$$
\mathcal{L}(\boldsymbol{\theta})
= \mathbb{E}_{t}\, \lambda(t)\; \mathbb{E}_{\mathbf{x}_0,\, \mathbf{x}_t \sim p_t(\cdot \mid \mathbf{x}_0)}
\left[\, \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)
- \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \right\|^2 \right],
$$
:eqlabel:`eq_mdl-ncsm-loss`

with a weighting $\lambda(t) > 0$ that decides which noise levels the network
should serve best. In the population problem, with an unrestricted function
class and positive weight at $t$, Vincent's theorem makes the minimizer
$\mathbf{s}(\cdot,t)=\nabla\log p_t$. A finite network trained on samples adds
approximation, estimation, and optimization error. With the likelihood
weighting $\lambda(t)=g(t)^2$, the cited analysis relates the population score
loss to an upper bound on negative log-likelihood under its regularity and
terminal-distribution assumptions :cite:`Song.Durkan.Murray.ea.2021`;
exercise :exref:`ex_mdl-score-matching-diffusion-flow-likelihood-bound`
derives this bound from the Fokker–Planck equation. DDPM's
simple loss instead induces $\lambda(t)=1-\bar{\alpha}_t$, derived below; this
changes the relative emphasis across noise levels and is often chosen for
sample quality rather than for that bound. Generation then follows the program of
:numref:`sec_mdl-time-reversal` and :numref:`sec_mdl-probability-flow-ode`:
start from the chosen terminal reference (approximately Gaussian at finite
noising time) and integrate either the reverse-time
SDE :cite:`Anderson.1982` or the probability-flow ODE, with
$\mathbf{s}_{\boldsymbol{\theta}}$ standing in for the true score. A generative
model is specified by this combination of forward process,
learned score, and numerical sampler.

### DDPM as a Discretized SDE
:label:`sec_mdl-ddpm-discretized-sde`

The Denoising Diffusion Probabilistic Model :cite:`ho2020denoising` looks, at
first sight, like a different theory: a discrete-time Markov chain of $T$
noising steps with schedule $\beta_1, \ldots, \beta_T \in (0, 1)$,

$$
\mathbf{x}_t = \sqrt{1 - \beta_t}\; \mathbf{x}_{t-1} + \sqrt{\beta_t}\; \boldsymbol{\epsilon}_t,
\qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, I)\ \textrm{i.i.d.}
$$
:eqlabel:`eq_mdl-ddpm-forward`

It is not a different theory. Three short propositions identify it, piece by
piece, with the VP picture above.

**Proposition (the DDPM step is Euler–Maruyama on the VP-SDE).** *Discretize
the VP-SDE with step $\Delta$ and write $\beta_t = \beta(t\Delta)\, \Delta$.
The Euler–Maruyama step (:numref:`sec_mdl-euler-maruyama`) is*

$$
\mathbf{x}_t = \left(1 - \tfrac{1}{2} \beta_t\right) \mathbf{x}_{t-1} + \sqrt{\beta_t}\; \boldsymbol{\epsilon}_t,
$$

*which agrees with the DDPM step :eqref:`eq_mdl-ddpm-forward` to first order in
$\beta_t$.*

**Proof.** The Taylor expansion
$\sqrt{1 - \beta} = 1 - \tfrac12 \beta - \tfrac18 \beta^2 - \cdots$ shows the
two coefficients differ by $O(\beta_t^2)$, while the noise terms are identical.
$\blacksquare$

The agreement is *first-order only* (for the largest practical
$\beta_t \approx 0.02$ the coefficients differ in the fifth decimal), but DDPM's
exact form is in one way nicer than the discretization that inspired it: it has
an exact closed-form marginal, with no $O(\beta^2)$ apology.

**Proposition (the $\bar{\alpha}$-marginal).** *Let $\alpha_t = 1 - \beta_t$
and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then conditionally on
$\mathbf{x}_0$,*

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\; \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\; \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I).
$$
:eqlabel:`eq_mdl-ddpm-marginal`

**Proof.** Induction on $t$; the case $t = 0$ is trivial. Assume
:eqref:`eq_mdl-ddpm-marginal` at $t - 1$ and substitute into
:eqref:`eq_mdl-ddpm-forward`:

$$
\mathbf{x}_t
= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\; \mathbf{x}_0
+ \sqrt{\alpha_t (1 - \bar{\alpha}_{t-1})}\; \bar{\boldsymbol{\epsilon}}
+ \sqrt{\beta_t}\; \boldsymbol{\epsilon}_t,
$$

with $\bar{\boldsymbol{\epsilon}}, \boldsymbol{\epsilon}_t$ independent
standard Gaussians. A sum of independent Gaussians is Gaussian with summed
variances:
$\alpha_t (1 - \bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \bar{\alpha}_t + 1 - \alpha_t = 1 - \bar{\alpha}_t$,
and $\alpha_t \bar{\alpha}_{t-1} = \bar{\alpha}_t$. $\blacksquare$

The term "variance-preserving" follows from the identity: for
unit-variance data, $\mathrm{Var}(\mathbf{x}_t) = \bar{\alpha}_t \cdot 1 + (1 - \bar{\alpha}_t) = 1$
for *every* $t$, not merely in the limit. And because
:eqref:`eq_mdl-ddpm-marginal` is a Gaussian kernel with scale
$\sqrt{1 - \bar{\alpha}_t}$, denoising score matching applies directly.

**Proposition (the DDPM loss is reweighted DSM).** *The conditional score of
:eqref:`eq_mdl-ddpm-marginal` is
$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid \mathbf{x}_0) = -\boldsymbol{\epsilon} / \sqrt{1 - \bar{\alpha}_t}$.
Hence, parameterizing the score model through a noise-prediction network,
$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) / \sqrt{1 - \bar{\alpha}_t}$,
the DDPM "simple loss"
$\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \|^2$
equals the noise-conditional DSM loss :eqref:`eq_mdl-ncsm-loss` with weighting
$\lambda(t) = 1 - \bar{\alpha}_t$.*

**Proof.** Differentiate
$\log p(\mathbf{x}_t \mid \mathbf{x}_0) = -\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0\|^2 / (2 (1 - \bar{\alpha}_t)) + \textrm{const}$
and substitute :eqref:`eq_mdl-ddpm-marginal`. Then

$$
\left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}} \right\|^2
= \frac{1}{1 - \bar{\alpha}_t} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \right\|^2,
$$

so the two losses differ exactly by the factor $\lambda(t) = 1 - \bar{\alpha}_t$
inside the time expectation. $\blacksquare$

So DDPM = VP forward process + DSM objective in
$\boldsymbol{\epsilon}$-parameterization + ancestral sampling, i.e. stepping
through the learned reverse chain one noise level at a time. The discrete
chain is a first-order counterpart of the continuous VP construction
:cite:`song2021score`. Historically the model was derived along an entirely
different route: write the reverse chain as a latent-variable model and
maximize an evidence lower bound, as in
:numref:`sec_mdl-latent-em-elbo` :cite:`sohl2015deep,ho2020denoising`.
After the same Gaussian algebra as above, the KL terms between the Gaussian
forward posteriors and the learned reverse steps collapse into weighted
$\boldsymbol{\epsilon}$-prediction losses; the ELBO and score-based derivations produce
the same objective with a different $\lambda(t)$, and :citet:`Luo.2022` is a
careful walkthrough of that equivalence.

The cell below checks both propositions at once: it runs the discrete chain
:eqref:`eq_mdl-ddpm-forward` for $T = 1000$ steps on (standardized) samples of
our two-Gaussian mixture and compares against the closed form
:eqref:`eq_mdl-ddpm-marginal`: variances on the way, and the full distribution
against a one-shot $\bar{\alpha}$-sample at the end.

```{.python .input #score-matching-diffusion-flow-ddpm-marginal}
rng = np.random.default_rng(13)
T = 1000
beta = np.linspace(1e-4, 0.02, T)                  # the DDPM schedule
alpha_bar = np.cumprod(1.0 - beta)
x0 = x / x.std()                                   # unit-variance mixture data
xt = x0.copy()
for t in range(T):                                 # the discrete forward chain
    xt = np.sqrt(1 - beta[t]) * xt + np.sqrt(beta[t]) * rng.standard_normal(n)
    if t + 1 in (10, 100, 1000):
        var_pred = alpha_bar[t] * x0.var() + (1 - alpha_bar[t])
        print(f't = {t+1:4d}: Var(x_t) chain {xt.var():.3f}, '
              f'formula {var_pred:.3f}, alpha_bar {alpha_bar[t]:.4f}')
one_shot = (np.sqrt(alpha_bar[-1]) * x0
            + np.sqrt(1 - alpha_bar[-1]) * rng.standard_normal(n))
print(f'chain vs one-shot at T: mean {xt.mean():+.3f} vs {one_shot.mean():+.3f}, '
      f'std {xt.std():.3f} vs {one_shot.std():.3f}')
```

The variance tracks the formula's $1.000$ at every checkpoint, to within the
sampling error of a $4096$-point variance estimate (the VP identity in
action), and after a thousand steps the chain matches the one-shot Gaussian
reparameterization in distribution, which is why DDPM training never simulates
the chain: it jumps straight to any $t$ via :eqref:`eq_mdl-ddpm-marginal`.

### Langevin Dynamics and Predictor–Corrector Sampling

Reverse-time SDEs are not the only way to turn a score into samples: the
oldest way predates diffusion models by decades. Suppose we hold the
distribution *fixed*: no noising schedule, just a target $p$ whose score we
know. **Langevin dynamics** is the SDE whose drift pushes up the
log-density while noise jiggles the state,

$$
d\mathbf{X} = \tfrac{1}{2} \nabla \log p(\mathbf{X})\, dt + d\mathbf{W}.
$$
:eqlabel:`eq_mdl-langevin`

**Proposition (stationarity).** *Let $p$ be a smooth positive density with
$p$ and $\nabla p$ vanishing at infinity. Then $p$ is a stationary density of
:eqref:`eq_mdl-langevin`: if $\mathbf{X}_0 \sim p$ then $\mathbf{X}_t \sim p$
for all $t \ge 0$.*

**Proof.** The Fokker–Planck equation (:numref:`sec_mdl-fokker-planck`) for
drift $\mathbf{b} = \tfrac12 \nabla \log p$ and unit diffusion reads
$\partial_t \rho = -\nabla \cdot (\rho\, \mathbf{b}) + \tfrac12 \Delta \rho$.
Substitute $\rho = p$ and use the identity
$p\, \nabla \log p = \nabla p$:

$$
-\nabla \cdot \left( \tfrac{1}{2}\, p\, \nabla \log p \right) + \tfrac{1}{2} \Delta p
= -\tfrac{1}{2} \nabla \cdot (\nabla p) + \tfrac{1}{2} \Delta p = 0.
$$

The right-hand side of the Fokker–Planck equation vanishes identically, so
$\rho \equiv p$ solves it for all time. $\blacksquare$

Discretizing :eqref:`eq_mdl-langevin` by Euler–Maruyama with step $h$ gives the
**Langevin sampler**
$\mathbf{x} \leftarrow \mathbf{x} + \tfrac{h}{2}\, \mathbf{s}(\mathbf{x}) + \sqrt{h}\, \boldsymbol{\xi}$.
Stationarity of the SDE is not convergence of its discretization. Under
appropriate ergodicity and regularity assumptions, the finite-step chain has
an invariant law that approximates $p$, generally with $O(h)$ weak bias (see
Exercise :exref:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity`); convergence to that law can still be slow. With
$\mathbf{s} = \mathbf{s}_{\boldsymbol{\theta}}$, this turns a trained score
network directly into a generator. The cell runs it on our mixture with the
analytic score, exposing that weakness along the way.

```{.python .input #score-matching-diffusion-flow-langevin}
rng = np.random.default_rng(11)

def langevin(q, h, steps, rng):
    for _ in range(steps):
        q = q + 0.5 * h * mixture_score(q, 0.25) \
            + np.sqrt(h) * rng.standard_normal(q.shape)
    return q

warm = langevin(rng.normal(0.0, 3.0, 10000), 0.01, 2000, rng)
print(f'spread-out start: P(X > 0) = {(warm > 0).mean():.3f}, '
      f'E[X^2] = {(warm**2).mean():.2f} (truth 0.500, 4.25)')
cold = langevin(np.full(10000, -2.0), 0.01, 2000, rng)
print(f'one-mode start:   P(X > 0) = {(cold > 0).mean():.3f}  (slow mixing)')
```

From a broad initial distribution, the chains approach the target proportions:
half the mass lies in each mode, and the second moment matches the true value.
When all chains start in the left mode, almost none cross after two thousand
steps because the density between the modes is small and the score points back
toward the current mode. This slow *mixing* makes plain Langevin sampling
inefficient on multimodal targets and motivates the noise schedules used by
diffusion models: **annealed
Langevin dynamics** :cite:`song2019generative` runs Langevin at a *ladder* of
noise levels $\sigma_1 > \cdots > \sigma_L$, using
$\mathbf{s}_{\boldsymbol{\theta}}(\cdot, \sigma_i)$ at level $i$. At large
$\sigma$ the smoothed density has no barriers, so chains move between regions;
as $\sigma$ shrinks, detail re-emerges with the global proportions already
right. The same idea survives inside modern samplers as the
**predictor–corrector** scheme :cite:`song2021score`: alternate a reverse-SDE
step (the *predictor*, which moves to the next noise level) with a few Langevin
steps at the current level (the *corrector*, which repairs the discretization
error of the predictor before it compounds).

### DDIM: Trading Noise for Speed

Ancestral DDPM sampling needs $T \sim 1000$ network calls. **DDIM**
(Denoising Diffusion Implicit Models)
:cite:`Song.Meng.Ermon.2020` cuts this by an order of magnitude with *the same
trained network* (no retraining) by replacing the noisy reverse chain with a
deterministic update.

::: {.callout-note title="The DDIM update, in one derivation"}
At time $t$, the network's noise prediction yields a current best guess of the
clean sample by inverting the marginal :eqref:`eq_mdl-ddpm-marginal`:

$$
\hat{\mathbf{x}}_0
= \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\; \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}.
$$

DDPM would *resample* by drawing fresh noise and forming a noisy
$\mathbf{x}_{t-1}$. DDIM instead **reuses the predicted direction**, forming
$\mathbf{x}_{t-1}$ from the current clean estimate
$\hat{\mathbf{x}}_0$ and noise estimate
$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ at time $t-1$:

$$
\mathbf{x}_{t-1}
= \sqrt{\bar{\alpha}_{t-1}}\; \hat{\mathbf{x}}_0
+ \sqrt{1 - \bar{\alpha}_{t-1}}\; \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t).
$$
:eqlabel:`eq_mdl-ddim-update`

The algebra is easiest to remember through a hypothetical forward pair:
if we knew that realization's actual $(\mathbf{x}_0,\boldsymbol{\epsilon})$,
the formula would move it along the curve
$t \mapsto \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
+ \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$. A network trained by
squared error does not recover that latent noise realization, however; it
recovers the conditional mean
$\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]$. Thus
:eqref:`eq_mdl-ddim-update` is a deterministic update defined from the learned
conditional estimate, not an exact jump along each sample's hidden forward
curve. We may evaluate it on a sparse subsequence of the training noise levels,
but larger gaps introduce discretization/model error.
:::

Two remarks complete the picture. First, DDIM is the $\eta = 0$ endpoint of a
family that interpolates to an ancestral sampler by reinjecting noise; its
non-Markovian forward construction has the same one-time noising marginals as
DDPM, so the same trained network can be reused. Second, in the fine-step
continuous-time limit the deterministic trajectory is related to the
probability-flow ODE of the VP-SDE
(:numref:`sec_mdl-probability-flow-ode`). This relationship motivates sparse
deterministic sampling, but it makes a finite DDIM stride exact only in
special cases, and Gaussian marginals alone are not enough.

![In the $(t, x)$ plane, each forward pair $(x_0,\epsilon)$ defines a curve $\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon$ whose cross-sections reproduce the noising marginals. DDIM uses the network's conditional noise estimate to take deterministic strides across selected time levels (marked), skipping the intermediate grid times (gray); the drawn curves explain the update algebra, not an exactly recovered latent path.](../img/mdl-dyn-ddim-strides.svg)
:label:`fig_mdl-dyn-ddim-strides`

How much do the strides cost when the marginal is *not* a single Gaussian? Our
standardized two-Gaussian mixture answers in closed form: its noised marginals,
hence the exact noise prediction
$\boldsymbol{\epsilon}_{\boldsymbol{\theta}} = -\sqrt{1 - \bar{\alpha}_t}\; \nabla \log p_t$,
are available at every $t$, so we can run the DDIM update
:eqref:`eq_mdl-ddim-update` with no learning in the loop: a thousand small
steps versus ten big ones from the same initial noise draws.

```{.python .input #mdl-score-matching-diffusion-flow-ddim-trading-noise-for-speed}
rng = np.random.default_rng(19)
mt, vt = 2 / np.sqrt(4.25), 0.25 / 4.25       # standardized mixture parameters

def ddim(z, K):                               # K deterministic DDIM strides
    idx = np.linspace(999, 0, K + 1).round().astype(int)
    xt = z.copy()
    for t1, t2 in zip(idx[:-1], idx[1:]):
        a1, a2 = alpha_bar[t1], alpha_bar[t2]
        s = mixture_score(xt, a1 * vt + 1 - a1,
                          (-np.sqrt(a1) * mt, np.sqrt(a1) * mt))
        eps = -np.sqrt(1 - a1) * s            # exact noise prediction
        x0_hat = (xt - np.sqrt(1 - a1) * eps) / np.sqrt(a1)
        xt = np.sqrt(a2) * x0_hat + np.sqrt(1 - a2) * eps
    return xt

def ks(a, b):                                 # two-sample KS statistic
    q = np.sort(np.concatenate([a, b]))
    return np.abs(np.searchsorted(np.sort(a), q, 'right') / len(a)
                  - np.searchsorted(np.sort(b), q, 'right') / len(b)).max()

z = rng.standard_normal(8000)
x_ref = ddim(z, 1000)                         # a thousand small staggers
for K in (10, 50):
    xK = ddim(z, K)
    print(f'{K:3d} strides vs 1000: mean |gap| {np.abs(xK - x_ref).mean():.3f}, '
          f'KS {ks(xK, x_ref):.3f}, mode fraction {(xK > 0).mean():.3f}')
print(f'mode fraction at 1000 steps: {(x_ref > 0).mean():.3f}; '
      'the same initial draws are used at every stride count')
```

Ten strides place every sample in the same mode as the thousand-step reference
and differ by only $0.08$ per sample on a scale where the modes sit at
$\pm 0.97$. (The mode fraction is *identical* at every stride count: the update
is deterministic, and in this run no trajectory crosses the valley.)
The discrepancy illustrates the central point: the predicted
noise is a posterior *mean*, not the realization's latent noise, so a finite
deterministic stride need not preserve the target marginal exactly. This is
true even for Gaussian data; Gaussianity makes the score linear, but does not
turn the conditional mean into the sampled noise realization. By fifty strides
the empirical CDF gap to the thousand-step numerical reference is $0.018$.
Since both runs use the same initial draws, this is a paired numerical
comparison rather than an independent-sample hypothesis test. Larger strides
reduce the number of evaluations while increasing finite-stride approximation
error.

### Guidance with Bayes' Rule

Generative models often condition on a label or prompt. Conditioning a score
model follows directly from probability identities and requires no new
training objective. Bayes' rule at noise level $t$ reads
$p_t(\mathbf{x} \mid y) \propto p_t(\mathbf{x})\, p_t(y \mid \mathbf{x})$, and
it becomes additive for scores, since the gradient is in $\mathbf{x}$ and the
evidence term drops:

$$
\nabla_{\mathbf{x}} \log p_t(\mathbf{x} \mid y)
= \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
+ \nabla_{\mathbf{x}} \log p_t(y \mid \mathbf{x}).
$$
:eqlabel:`eq_mdl-guidance-bayes`

Any sampler from this section runs unchanged with the conditional score in
place of the unconditional one. **Classifier guidance**
:cite:`Dhariwal.Nichol.2021` implements the second term with an auxiliary
classifier $p_{\boldsymbol{\phi}}(y \mid \mathbf{x}, t)$ trained on *noisy*
inputs, because a clean-image classifier is not calibrated off the data
manifold where $\mathbf{x}_t$ is evaluated, and scales it with a **guidance scale**
$\gamma > 1$:

$$
\tilde{\mathbf{s}}(\mathbf{x}, t)
= \nabla \log p_t(\mathbf{x}) + \gamma\, \nabla \log p_{\boldsymbol{\phi}}(y \mid \mathbf{x}, t)
= \nabla \log \left[ \frac{p_t(\mathbf{x})\, p_{\boldsymbol{\phi}}(y \mid \mathbf{x}, t)^{\gamma}}{Z} \right].
$$

The second equality identifies the guided field at noise level $t$ as the
score of a *tilted* distribution in which the classifier contribution is
weighted by $\gamma$.
This weighting favors more prototypical examples of "$y$" and reduces
diversity.

**Classifier-free guidance (CFG)** :cite:`Ho.Salimans.2022` removes the
auxiliary classifier with one more application of
:eqref:`eq_mdl-guidance-bayes`, read right to left:
$\nabla \log p_t(y \mid \mathbf{x}) = \nabla \log p_t(\mathbf{x} \mid y) - \nabla \log p_t(\mathbf{x})$.
Train a *single* network on labeled data, dropping the label some fraction of
the time, so it learns both
$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, y)$ and
$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, \varnothing)$. At sampling
time, *extrapolate* from the unconditional score through the conditional one:

$$
\tilde{\mathbf{s}}(\mathbf{x}, t)
= \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, \varnothing)
+ \gamma \left[ \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, y)
- \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, \varnothing) \right],
$$
:eqlabel:`eq_mdl-cfg`

equivalently $(1 - \gamma)\, \mathbf{s}_\varnothing + \gamma\, \mathbf{s}_y$:
$\gamma = 0$ ignores the label, while $\gamma = 1$ uses the network's
conditional score (the exact conditional only for an exact model),
and the values used in practice ($\gamma \approx 3$–$10$ for text-to-image
models, occasionally higher) push *past* the conditional, in the score-space
direction "more like $y$". Substituting the Bayes identity shows :eqref:`eq_mdl-cfg` is exactly the
classifier-guidance tilt with the implicit classifier
$p_t(y \mid \mathbf{x}) = p_t(\mathbf{x} \mid y)\, p(y) / p_t(\mathbf{x})$ in
the exponent's role. One caveat: for $\gamma > 1$ the tilted object
$p_t(\mathbf{x})\, p_t(y \mid \mathbf{x})^\gamma$ is not, in general, the
noised marginal of *any* clean distribution
(exercise :exref:`ex_mdl-score-matching-diffusion-flow-guided-marginals`): the guided field is a useful
controlled distortion, not the score of a consistent diffusion, and the
resulting fidelity-versus-diversity trade-off is an engineering choice, not a
theorem. In $\boldsymbol{\epsilon}$-parameterization, :eqref:`eq_mdl-cfg` is
applied verbatim to $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$, since the
two differ by the $t$-dependent factor $-\sqrt{1 - \bar{\alpha}_t}$.

Guidance can also be examined in closed form.
Label the two modes of our standardized mixture as classes, with $y$ naming the
right mode: the class-conditional $p_t(\cdot \mid y)$ is a single moving
Gaussian, the unconditional $p_t$ is the mixture, and both scores are exact,
so the guided field :eqref:`eq_mdl-cfg` needs no network at all. The cell runs
ancestral reverse sampling with it at $\gamma \in \{1, 3, 10\}$.

```{.python .input #mdl-score-matching-diffusion-flow-guidance-steering-with-bayes-rule}
rng = np.random.default_rng(23)

def cfg_sample(gamma, n=8000):                # ancestral chain, guided score
    xt = rng.standard_normal(n)
    for t in range(999, -1, -1):
        a, b = alpha_bar[t], beta[t]
        v = a * vt + 1 - a
        s_uncond = mixture_score(xt, v, (-np.sqrt(a) * mt, np.sqrt(a) * mt))
        s_cond = (np.sqrt(a) * mt - xt) / v   # class y = the right mode
        s = (1 - gamma) * s_uncond + gamma * s_cond
        xt = (xt + b * s) / np.sqrt(1 - b)
        if t > 0:
            xt = xt + np.sqrt(b) * rng.standard_normal(n)
    return xt

samples = {g: cfg_sample(g) for g in (1.0, 3.0, 10.0)}
for g, xs in samples.items():
    print(f'gamma = {g:4.1f}: mass right of 0: {(xs > 0).mean():.3f}, '
          f'mean {xs.mean():.3f}, std {xs.std():.3f}')
print(f'exact conditional: mean {mt:.3f}, std {np.sqrt(vt):.3f}')
d2l.set_figsize((6, 2.5))
for g, xs in samples.items():
    d2l.plt.hist(xs, bins=80, density=True, histtype='step',
                 label=f'gamma = {g:g}')
d2l.plt.xlabel('x'), d2l.plt.ylabel('density'), d2l.plt.legend();
```

The printout measures the tilt. At $\gamma = 1$ the finite-step sampler closely
approximates the exact conditional: mean $0.966$ against the analytic $0.970$, standard
deviation $0.244$ against $0.243$, and *all* of the mass in the right mode,
where the unconditional mixture would put only half of it. Pushing $\gamma$ to
$3$ and $10$ has no more mass to reallocate, so it distorts the surviving mode
instead: the mean slides from $0.97$ to $1.04$ to $1.07$, away from the class
boundary ("more prototypically $y$") while the histogram narrows slightly.
This shift and narrowing illustrate the preceding caveat:
for $\gamma > 1$ the samples track no noised marginal of any clean
distribution; the guided chain produces its own distribution, more emphatic
and less diverse than the class it names.

## Summary

* The score $\nabla \log p$ does not require evaluating the normalizing constant.
  Explicit score matching minimizes the Fisher divergence; Hyvärinen's
  integration by parts :eqref:`eq_mdl-hyvarinen` makes it estimable from
  samples, at the price of exact divergence computation whose derivative work
  generally scales with $d$.
* The **regression lemma**: least squares against a noisy target fits its
  conditional mean. Denoising score matching (target $-\boldsymbol{\epsilon}/\sigma$,
  marginal score = posterior mean of conditional scores) and conditional flow
  matching (target $\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$, marginal
  velocity = posterior mean of conditional velocities) use the same theorem.
  Their population optima retain the corresponding conditional variance;
  observed training loss also contains approximation and optimization error.
* DDPM is the variance-preserving SDE discretized (first order), with an exact
  $\bar{\alpha}$-marginal, and its $\boldsymbol{\epsilon}$-prediction loss is
  reweighted DSM with $\lambda(t) = 1 - \bar{\alpha}_t$; the ELBO derivation
  reaches the same objective.
* A score alone samples via **Langevin dynamics**, whose stationary density is
  $p$ (one-line Fokker–Planck proof) but whose mixing across modes is slow:
  hence annealing over noise levels and predictor–corrector samplers. **DDIM**
  reuses a trained DDPM deterministically with big steps; **guidance** is
  Bayes' rule on scores, with classifier-free guidance an extrapolation
  $(1 - \gamma) \mathbf{s}_\varnothing + \gamma \mathbf{s}_y$.

## Exercises

1. **Boundary terms.** Hyvärinen's identity :eqref:`eq_mdl-hyvarinen` assumes
   that $p(x)\, s_{\boldsymbol{\theta}}(x)$ vanishes at the ends of the
   support. Let the data be uniform on $[0, 1]$, so that $\nabla \log p = 0$
   inside the interval, and fit the linear model $s_{a,b}(x) = a + b x$.
    1. Minimize the explicit objective :eqref:`eq_mdl-esm-objective` and the
       implicit objective, the right-hand side of :eqref:`eq_mdl-hyvarinen`,
       over $(a, b)$. Which density has the score that the implicit objective
       selects?
    1. Repeat the one-dimensional integration by parts of the proof on
       $[0, 1]$, keep the boundary term, and show that it accounts for the
       difference between the two objectives.
    1. Pixel intensities are bounded in the same way. Why does denoising score
       matching at a noise level $\sigma > 0$ not inherit the problem?

1. [code] **Langevin discretization bias.** For the Gaussian target
   $p = \mathcal{N}(0, v)$, the Langevin sampler of :eqref:`eq_mdl-langevin`
   with step $h$ is the linear recursion
   $x \leftarrow (1 - \tfrac{h}{2v})\, x + \sqrt{h}\, \xi$.
    1. Find the step sizes for which the recursion has a stationary
       distribution, compute its variance, and expand the bias to first order
       in $h$.
    1. Treat each mode of the mixture in
       `#score-matching-diffusion-flow-langevin` as a separate Gaussian with
       $v = 0.25$, and predict the stationary $\mathbb{E}[X^2]$ at
       $h \in \{0.01, 0.1, 0.25, 0.5\}$. Compare each predicted bias with the
       Monte Carlo standard error of an estimate from $10000$ chains. Then run
       `langevin` for $2000$ steps at each of these step sizes from the
       initial points `rng.normal(0.0, 3.0, 10000)` of `warm`. From which
       step size on does the experiment resolve the bias?
    1. The Metropolis-adjusted Langevin algorithm :cite:`Roberts.Tweedie.1996`
       removes the bias with an accept–reject test that needs the density
       ratio $p(\mathbf{x}') / p(\mathbf{x})$. Why can a sampler that has only
       a trained score network $\mathbf{s}_{\boldsymbol{\theta}}$ not apply
       this test directly?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity`

1. [code] **Annealing out of one mode.** In
   `#score-matching-diffusion-flow-langevin`, chains started in the left mode
   stay there. Annealed Langevin dynamics first runs the sampler on smoothed
   densities $p_\sigma$, here the mixture with component variance
   $0.25 + \sigma^2$, for the decreasing ladder of noise levels
   $\sigma = 2, 1, 0.5, 0.25, 0.1, 0$.
    1. By exercise
       :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging`b of
       :numref:`sec_mdl-fokker-planck-probability-flow`, the mixture
       $\tfrac12 \mathcal{N}(-\mu, s^2) + \tfrac12 \mathcal{N}(\mu, s^2)$ is
       unimodal if and only if $s \ge \mu$. The means of `mixture_score` are
       $\pm 2$. For which $\sigma$ is $p_\sigma$ unimodal? Which levels of the
       ladder lie in this range?
    1. Give `langevin` the component variance as an argument, start $10000$
       chains at $-2$, and run $300$ steps at each level of the ladder. Do
       this once with the fixed step
       $h = 0.01$ and once with $h = 0.04\, (0.25 + \sigma^2)$, which equals
       $0.01$ at $\sigma = 0$. Report the fraction of chains with $X > 0$ at
       the end of each run.
    1. :citet:`song2019generative` scale the step size of each level with
       $\sigma^2$. Use the contraction factor of the linear recursion in
       exercise :exref:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity`
       to explain why the fixed step fails at the large noise levels.

1. [code] **The Bayes-risk floor.** ● The cell
   `#score-matching-diffusion-flow-dsm-train` prints a final DSM loss of
   $2.036$ and an estimated Bayes risk of $2.038$ for the mixture at
   $\sigma = 0.5$. Both are averages over the same $4096$ noisy samples.
    1. Let $Y = -\boldsymbol{\epsilon}/\sigma$ be the DSM target
       :eqref:`eq_mdl-dsm-target` for data in $\mathbb{R}^d$. Use the
       regression lemma :eqref:`eq_mdl-regression-lemma` and Vincent's theorem
       to show that the population Bayes risk is

       $$\mathbb{E} \left\| Y - \nabla \log p_\sigma(\tilde{\mathbf{x}}) \right\|^2 = \frac{d}{\sigma^2} - \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma} \left\| \nabla \log p_\sigma(\tilde{\mathbf{x}}) \right\|^2.$$

    1. Evaluate the right-hand side for the mixture at $\sigma = 0.5$ by
       numerical quadrature with `mixture_score`. Compute the standard error of
       the printed Bayes-risk estimate from its $4096$ per-sample terms. Is the
       printed value consistent with the population value?
    1. The printed loss and the printed Bayes-risk estimate differ by much less
       than either standard error. Explain why.
    1. Estimate the excess risk
       $\mathbb{E}_{p_\sigma} (s_{\boldsymbol{\theta}} - \nabla \log p_\sigma)^2$
       of the trained network at the noisy points `x + 0.5 * eps`, with the
       targets `y`, in two ways: as the mean of the per-sample differences of
       the two squared errors, and as the mean of
       $(s_{\boldsymbol{\theta}} - \nabla \log p_\sigma)^2$ at these points.
       Evaluate $s_{\boldsymbol{\theta}}$ from the final parameters `W1`, `b1`,
       `W2`, and `b2`, because the array `S` was computed before the last Adam
       update. Which estimate is more precise, and why? Compare both with a
       quadrature of the expectation on a grid.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor`

1. **Weightings across noise levels.** ● Consider the VP-SDE with the schedule
   `beta_fn` of `#score-matching-diffusion-flow-euler-vs-heun`,
   $\beta(t) = \beta_{\min} + t\, (\beta_{\max} - \beta_{\min})$ with
   $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$, and
   $\sigma_t^2 = 1 - \bar{\alpha}(t)$, where $\bar{\alpha}(t)$ is `abar_fn`.
   Let the data in $\mathbb{R}^d$ have a smooth density, and assume that
   $\mathbb{E} \| \nabla \log p_t \|^2$ stays bounded as $t \to 0$. Write
   $\ell(t)$ for the Bayes risk of the noise-conditional DSM loss
   :eqref:`eq_mdl-ncsm-loss` at level $t$, which problem
   :exref:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor`a gives with
   $\sqrt{\bar{\alpha}(t)}\, \mathbf{x}_0$ in place of $\mathbf{x}$ and
   $\sigma_t$ in place of $\sigma$.
    1. Show that $\ell(t) \approx d / (\beta_{\min} t)$ as $t \to 0$.
    1. Find the behavior of $\lambda(t)\, \ell(t)$ as $t \to 0$ for
       $\lambda = 1$, for the DDPM weighting $\lambda = \sigma_t^2$, and for
       the likelihood weighting $\lambda = \beta(t)$. For which weightings is
       $\int_0^1 \lambda(t)\, \ell(t)\, dt$ finite, and which noise levels does
       each weighting emphasize?
    1. Training draws $t$ uniformly from $[\varepsilon, 1]$. For the likelihood
       weighting, show that the second moment of $\lambda(t)\, \ell(t)$ under
       this draw grows like $1/\varepsilon$.
       :citet:`Song.Durkan.Murray.ea.2021` (Section 5.1) instead draw $t$ from
       the density proportional to $\beta(t) / \sigma_t^2$ on
       $[\varepsilon, 1]$ and multiply the DSM term at the drawn level by
       $Z \sigma_t^2$, where $Z$ normalizes that density. Show that this
       estimate is unbiased for the likelihood-weighted loss and that its
       value at the optimum, $Z \sigma_t^2\, \ell(t)$, stays bounded as
       $t \to 0$.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-weightings`

1. [code] **DDIM against Euler and Heun.** ● The cell
   `#score-matching-diffusion-flow-euler-vs-heun` integrates the
   probability-flow ODE with exact scores, and the DDIM update
   :eqref:`eq_mdl-ddim-update` can be run on the same continuous schedule.
    1. Implement DDIM on the uniform time grid of `solve`, with `abar_fn` in
       place of $\bar{\alpha}$ and the exact noise prediction
       $-\sqrt{1 - \bar{\alpha}(t)}$ times `score_vp`. For
       $K \in \{5, 10, 20, 40\}$, compare its endpoint error against `ref`
       with those of Euler and Heun from the same draws `z`.
    1. Show that for data concentrated at a single point the DDIM update lands
       on the probability-flow trajectory for every stride. Which quantity does
       DDIM hold constant across a stride, and which quantity does Euler hold
       constant?
    1. Repeat (a) with the component variance $0.01$ and then $10^{-4}$ in
       place of $0.25$, and recompute `ref`. Compute the responsibilities in
       `score_vp` from differences of the exponents, so that they do not
       evaluate to $0/0$. How does the ranking of the three methods change?
       Explain the failure of Heun with the Jacobian of `pf_ode` near $t = 0$
       and the times at which Heun evaluates the field.
    1. :citet:`Karras.Aittala.Aila.ea.2022` (Algorithm 1) skip the
       second-order correction on the step that ends at noise level zero. Make
       this change to Heun and repeat (c).

1. [code] **Guidance by noise level.** ● In
   `#mdl-score-matching-diffusion-flow-guidance-steering-with-bayes-rule`,
   raising $\gamma$ from $1$ to $10$ moves the mean of the guided samples from
   $0.966$ to $1.072$.
    1. At the last level of the chain, $t = 0$ in `cfg_sample`, the class
       posterior of the right mode is a logistic function of $x$. Write it, and
       compute by quadrature the mean of the tilted density proportional to
       $p_t(x)\, p_t(y \mid x)^{\gamma}$ at that level for
       $\gamma \in \{1, 3, 10\}$. Does this tilt explain the printed shift?
    1. Modify `cfg_sample` so that it uses $\gamma = 10$ only on the steps
       $t \ge 500$, only on $200 \le t < 500$, or only on $t < 200$, and
       $\gamma = 1$ on the others. Which band of noise levels produces the
       shift? Relate the answer to the separation
       $\sqrt{\bar{\alpha}_t}\, m / \sqrt{v_t}$ of the two noised classes,
       where $m$ is `mt`, $v_0$ is the clean component variance `vt`, and
       $v_t = \bar{\alpha}_t\, v_0 + 1 - \bar{\alpha}_t$ is the noised
       component variance, the variable `v` of `cfg_sample`.
    1. :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` find on image models that
       guidance is harmful at high noise levels, largely unnecessary at low
       noise levels, and beneficial in between. Which parts of this pattern
       does the one-dimensional example reproduce, and which can it not
       exhibit?

1. **Guided densities and noised marginals.** ● The guided field
   :eqref:`eq_mdl-cfg` with $\gamma > 1$ is the score of the tilted density
   $\tilde{p}_t(\mathbf{x}) \propto p_t(\mathbf{x})\, p_t(y \mid \mathbf{x})^{\gamma}$.
   Call a density a noised marginal at level $t$ if it is the law of
   $\sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$,
   with $\sigma_t^2 = 1 - \bar{\alpha}_t$, for some clean distribution of
   $\mathbf{x}_0$.
    1. For a noised marginal $p_t$, Tweedie's formula reads
       $\nabla \log p_t(\mathbf{x}) = (\sqrt{\bar{\alpha}_t}\, \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] - \mathbf{x}) / \sigma_t^2$.
       Differentiate it once more to show

       $$\nabla^2 \log p_t(\mathbf{x}) = -\frac{I}{\sigma_t^2} + \frac{\bar{\alpha}_t}{\sigma_t^4} \operatorname{Cov}\left[ \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x} \right].$$

       Conclude that no eigenvalue of $-\nabla^2 \log p_t$ exceeds
       $1 / \sigma_t^2$, whatever the clean distribution.
    1. In `cfg_sample`, the two noised classes are
       $\mathcal{N}(\pm \sqrt{\bar{\alpha}_t}\, m, v_t)$, where $m$ is `mt`,
       $v_0$ is the clean component variance `vt`, and
       $v_t = \bar{\alpha}_t\, v_0 + \sigma_t^2$ is the noised component
       variance, the variable `v` of `cfg_sample`. The label $y$ names the
       right class. Show that
       $-(\log \tilde{p}_t)''(x) = 1/v_t + (\gamma - 1)\, a_t^2\, \varsigma(a_t x)\, (1 - \varsigma(a_t x))$,
       where $a_t = 2 \sqrt{\bar{\alpha}_t}\, m / v_t$ and
       $\varsigma(u) = 1 / (1 + e^{-u})$ is the logistic function.
    1. Find the smallest $\gamma$ for which $\tilde{p}_t$ violates the bound of
       (a) at some $x$. Evaluate it at the levels $t = 0$, $100$, and $500$ of
       `alpha_bar`. What do the values say about the densities that the chain
       of `cfg_sample` follows at $\gamma = 3$?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-guided-marginals`

1. **Score error and likelihood.** ●● With the likelihood weighting
   $\lambda = g^2$, the population score loss bounds the Kullback–Leibler
   divergence from the data to the model, by Theorem 1 of
   :citet:`Song.Durkan.Murray.ea.2021`. The paper proves the bound with path
   measures; this problem derives it from the Fokker–Planck equation of
   :numref:`sec_mdl-fokker-planck`, following
   :citet:`Lu.Zheng.Bao.ea.2022`, who take this route for the SDE (their
   Proposition C.1) and for the probability-flow ODE (their Theorem 3.1).
   Let the forward SDE
   $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$ have
   marginals $p_t$ on $[0, T]$. In reverse time $\tau = T - t$, the exact
   reversal and the model are

   $$d\mathbf{x} = \left[ -\mathbf{f} + g^2\, \nabla \log p_t \right] d\tau + g\, d\mathbf{B}_\tau, \qquad d\mathbf{x} = \left[ -\mathbf{f} + g^2\, \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) \right] d\tau + g\, d\mathbf{B}_\tau,$$

   evaluated at $t = T - \tau$. The first has marginals $p_t$; the second,
   started from a prior $\pi$ at $\tau = 0$, has marginals $q_t$. Assume that
   all densities are smooth and positive and decay fast enough for every
   integration by parts.
    1. Write both Fokker–Planck equations in $\tau$ in the form
       $\partial_\tau \rho = -\nabla \cdot (\rho\, \mathbf{w})$, and identify
       the velocities $\mathbf{w}_p$ and $\mathbf{w}_q$.
    1. Show that
       $\frac{d}{d\tau} \mathrm{KL}(p_t \,\|\, q_t) = \mathbb{E}_{p_t}\left[ (\mathbf{w}_p - \mathbf{w}_q)^\top \nabla \log (p_t / q_t) \right]$.
    1. Bound the right-hand side by
       $\tfrac12 g^2\, \mathbb{E}_{p_t} \| \mathbf{s}_{\boldsymbol{\theta}} - \nabla \log p_t \|^2$,
       integrate over $\tau$, and obtain

       $$\mathrm{KL}(p_0 \,\|\, q_0) \le \mathrm{KL}(p_T \,\|\, \pi) + \frac12 \int_0^T g(t)^2\, \mathbb{E}_{p_t} \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \nabla \log p_t(\mathbf{x}) \right\|^2 dt.$$

       Under which condition on $q_t$ is the bound of the derivative tight?
    1. Repeat (a) and (b) for the probability-flow ODE with
       $\mathbf{s}_{\boldsymbol{\theta}}$ in place of the score. Which term
       that made (c) possible is missing, and why does the score loss alone no
       longer bound $\mathrm{KL}(p_0 \,\|\, q_0)$?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-likelihood-bound`

1. [extended] **Learned noise prediction.** ●● The samplers `langevin`,
   `ddim`, `cfg_sample`, and `solve` all use exact scores.
   Replace the exact score by a trained network for the standardized mixture
   `x / x.std()` under the DDPM schedule `beta` and `alpha_bar`.
    1. Train a noise-prediction network
       $\epsilon_{\boldsymbol{\theta}}(x_t, t)$ with input $(x_t, t/1000)$,
       two hidden layers of $64$ tanh units, and the optimizer settings of
       `#score-matching-diffusion-flow-cfm-train`, on the simple loss with $t$
       uniform over the $1000$ levels. Compute its root-mean-square error
       against the exact noise prediction
       $-\sqrt{1 - \bar{\alpha}_t}\, \nabla \log p_t$ on samples of $p_t$ at
       several levels $t$. Where is the error largest?
    1. Sample with `ddim` at $K = 10$ and $K = 50$, and with the ancestral
       chain of `cfg_sample` in which the network's score replaces the guided
       score. Compare each set of samples with exact draws of the mixture by
       the fraction in the right mode, the standard deviation, and `ks`. Which
       sampler is most sensitive to the error of (a), and at which step of
       that sampler does the error enter?
    1. Retrain with the per-level weights that correspond to $\lambda = 1$ and
       to the likelihood weighting $\lambda = \beta$, that is, with the noise
       loss at level $t$ multiplied by $\lambda(t) / (1 - \bar{\alpha}_t)$, and
       repeat (a) and (b). Relate the
       results to problem :exref:`ex_mdl-score-matching-diffusion-flow-weightings`.

   The deliverable is a table of the statistics of (b) for each weighting and
   sampler, together with a plot of the error of (a) against $t$.

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:

[Discussions](https://d2l.discourse.group/t/36840)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §27.4]{.kicker}

**Score Matching and Diffusion Models**
:::
:::

::: {.slide title="Scores avoid normalizer evaluation"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
An energy model $p_\theta = e^{-E_\theta}/Z_\theta$ needs the intractable
$Z_\theta$ at every step. The **score** is independent of it:

$$\nabla_{\mathbf x}\log p_\theta = -\nabla_{\mathbf x} E_\theta,
\qquad \nabla\log Z_\theta = 0.$$

Objectives and samplers expressed only through the score need not evaluate
$Z_\theta$.
:::

::: {.col .fig}
@fig:mdl-dyn-score-field
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Learning the score]{.dtitle}

[explicit → implicit → denoising]{.dsub}
:::
:::

::: {.slide title="Integration by parts removes the unknown data score"}
[The objective]{.kicker}

The Fisher divergence
$\tfrac12\mathbb E_p\|\mathbf s_\theta-\nabla\log p\|^2$ still contains the
unknown score. Hyvärinen integrates by parts:

$$J_{\mathrm{ESM}} = \mathbb E_p\bigl[\tfrac12\|\mathbf s_\theta\|^2
+ \nabla\cdot\mathbf s_\theta\bigr] + C.$$

. . .

Tractable, but an exact divergence generally requires derivative work that
scales with $d$ (for example, one reverse-mode pass per Jacobian row).
:::

::: {.slide title="The regression lemma"}
[Conditional means]{.kicker}

::: {.d2l-note .rule}
$\mathbb E\|\mathbf v(X)-Y\|^2 = \mathbb E\|\mathbf v(X)-\mathbf m(X)\|^2
+ \text{const}$, where $\mathbf m(X)=\mathbb E[Y\mid X]$.
:::

*Proof.* Insert $\pm\mathbf m(X)$; the cross term vanishes by the tower rule.
$\blacksquare$ Least squares against a noisy target fits its **conditional
mean**: used once for scores, once for velocities.
:::

::: {.slide title="Denoising score matching & Tweedie"}
[Denoising]{.kicker}

::: {.cols .vc}
::: {.col}
Perturb $\tilde{\mathbf x}=\mathbf x+\sigma\boldsymbol\epsilon$; the
conditional score is closed-form $-\boldsymbol\epsilon/\sigma$, and regressing
on it (Vincent) recovers the marginal score. Rearranged, that is **Tweedie**:

$$\mathbb E[\mathbf x\mid\tilde{\mathbf x}]
= \tilde{\mathbf x} + \sigma^2\,\nabla\log p_\sigma(\tilde{\mathbf x})$$

The score correction equals the posterior mean.
:::

::: {.col .fig .big}
![](../img/mdl-dyn-tweedie.svg)
:::
:::

::: {.d2l-note}
For Gaussian corruption and squared error, the score determines the
posterior-mean denoiser through Tweedie's formula.
:::
:::

::: {.slide title="A score network in 1-D"}
[Denoising]{.kicker}

A tiny MLP trained by denoising score matching approximates the analytic
score, with loss close to a finite-sample estimate of the Bayes risk:

@!score-matching-diffusion-flow-dsm-train
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Score-based diffusion]{.dtitle}

[forward noise, DDPM, Langevin, DDIM, guidance]{.dsub}
:::
:::

::: {.slide title="A noise schedule connects coverage to detail"}
[Forward process]{.kicker}

Small $\sigma$ approximates $p$ but provides little coverage of low-density
regions; large $\sigma$ provides coverage but over-smooths. Train a
noise-conditional score $\mathbf s_\theta(\mathbf x,t)$
trained along a forward SDE (VE or VP), then a reverse pass to generate:

@fig:mdl-dyn-forward-reverse

A weighting $\lambda(t)$ allocates effort across noise levels. Under the
regularity and terminal-distribution assumptions of the cited analysis,
$\lambda=g^2$ relates the population loss to an upper bound on negative
log-likelihood; DDPM uses $1-\bar\alpha_t$.
:::

::: {.slide title="DDPM is a first-order VP discretization with exact marginals"}
[DDPM]{.kicker}

1. The DDPM step is Euler–Maruyama on the VP-SDE (to $O(\beta_t)$).
2. Exact marginal $\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$, $\bar\alpha_t=\prod_s(1-\beta_s)$.
3. The simple $\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\|^2$ loss is denoising score matching, reweighted by $\lambda(t)=1-\bar\alpha_t$.

@!score-matching-diffusion-flow-ddpm-marginal
:::

::: {.slide title="Langevin: stationary but slow"}
[Sampling]{.kicker}

$dX = \tfrac12\nabla\log p\,dt + dW$ has stationary density $p$ (substitute
$\rho=p$ into Fokker–Planck → $0$). But it mixes slowly across modes:

@score-matching-diffusion-flow-langevin

::: {.d2l-note}
From one mode, almost no chain crosses ($P(X>0)=0.012$). Annealed noise or a
predictor–corrector method improves movement between modes.
:::
:::

::: {.slide title="DDIM: deterministic updates from conditional estimates"}
[Sampling]{.kicker}

::: {.cols .vc}
::: {.col}
Invert the marginal for
$\hat{\mathbf x}_0 = \bigl(\mathbf x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta\bigr)/\sqrt{\bar\alpha_t}$,
then **re-use** the predicted noise instead of resampling:

$$\mathbf x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{\mathbf x}_0
+ \sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol\epsilon_\theta.$$

The update is deterministic and can skip levels. Its predicted noise is a
conditional mean, not the latent noise realization of an individual forward
sample, so a finite stride is approximate.
:::

::: {.col .fig .big}
![](../img/mdl-dyn-ddim-strides.svg)
:::
:::
:::

::: {.slide title="Sparse DDIM strides trade evaluations for bias"}
[Sampling]{.kicker}

On the closed-form mixture, with no learned approximation, ten strides place
every sample in the same mode as the thousand-step numerical reference; at fifty
strides the paired empirical CDF gap is $0.018$:

@!mdl-score-matching-diffusion-flow-ddim-trading-noise-for-speed

Same network; $\eta$ controls reinjected noise. In the fine-step limit,
deterministic DDIM is related to the probability-flow ODE; finite strides are
numerical approximations, including for Gaussian marginals.
:::

::: {.slide title="Guidance is Bayes on scores"}
[Guidance]{.kicker}

$$\nabla\log p_t(\mathbf x\mid y) = \nabla\log p_t(\mathbf x) + \nabla\log p_t(y\mid\mathbf x).$$

**Classifier-free guidance** trains one network with label dropout and
extrapolates *through* the conditional:
$\tilde{\mathbf s} = (1-\gamma)\,\mathbf s_\varnothing + \gamma\,\mathbf s_y$.

. . .

Measured on the closed-form mixture: $\gamma=1$ closely approximates the exact
conditional (mean $0.966$ vs the analytic $0.970$); at $\gamma=3, 10$ there is
no additional mass to reallocate, so the mode shifts to $1.04$, then $1.07$,
and narrows.

::: {.d2l-note}
For $\gamma>1$ the tilt $p_t(\mathbf x)\,p_t(y\mid\mathbf x)^\gamma$ is not,
in general, the noised marginal of a clean distribution: it is a controlled
distortion rather than a consistent diffusion path.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- The score is independent of $Z$; DSM (Vincent) regresses on $-\boldsymbol\epsilon/\sigma$; Tweedie's formula gives the optimal denoiser.
- DDPM = VP-SDE discretized; $\bar\alpha_t$ marginal; $\boldsymbol\epsilon$-loss = reweighted DSM.
- Langevin mixes slowly; DDIM is deterministic; guidance is Bayes on scores.
:::
