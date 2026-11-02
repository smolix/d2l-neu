# Flow Matching and Optimal Transport
:label:`sec_mdl-flow-matching-transport`

Placeholder opening (the section writer replaces this paragraph): this section
learns a velocity field by regression, asks why straight paths need fewer
integration steps, and relates straightness to optimal transport.

```{.python .input #mdl-flow-matching-transport-flow-matching-and-optimal-transport}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet import np as mxnp
import numpy as np
npx.set_np()
```

```{.python .input #mdl-flow-matching-transport-flow-matching-and-optimal-transport}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input #mdl-flow-matching-transport-flow-matching-and-optimal-transport}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #mdl-flow-matching-transport-flow-matching-and-optimal-transport}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import optax
```

::: {.callout-important title="Two clocks: the time conventions of diffusion and flow matching"}
The two literatures run time in opposite directions, and almost every
sign confusion in this field traces back to it
(:numref:`fig_mdl-dyn-time-conventions`).

* **Diffusion** noises *data into noise* forward in time: $t = 0$ is data,
  $t = T$ is (approximately) pure Gaussian. *Sampling integrates backwards*,
  from $t = T$ down to $0$.
* **Flow matching** parameterizes the *generative* direction: $t = 0$ is
  noise, $t = 1$ is data, and sampling integrates forwards from $0$ to $1$.

In this section, $t$ in a diffusion formula runs data $\to$ noise, and $t$ in
a flow-matching formula runs noise $\to$ data. When comparing the two (as the
unifying table at the end does), substitute $t \mapsto 1 - t$ in one of them.
:::

![Two clocks. Diffusion (top) runs $t$ from $0$ (data) to $T$ (noise) and samples by integrating backwards; flow matching (bottom) runs $t$ from $0$ (noise) to $1$ (data) and samples forwards. The endpoint densities are the same: only the direction of time differs.](../img/mdl-dyn-time-conventions.svg)
:label:`fig_mdl-dyn-time-conventions`

## Flow Matching and Rectified Flow
:label:`sec_mdl-flow-matching`

### Probability Paths and Velocity Fields

Diffusion derives a bridge between noise and data from a stochastic process
and then reverses it. Flow matching :cite:`Lipman.Chen.BenHamu.ea.2022`
instead prescribes a family of densities
$(p_t)_{t \in [0, 1]}$ with $p_0$ = noise and $p_1$ = data (a
**probability path**) and learn the velocity field that transports mass along
it. Recall from :numref:`sec_mdl-continuity-equation` that a velocity field
$\mathbf{u}_t$ realizes the path iff the pair satisfies the continuity
equation $\partial_t p_t = -\nabla \cdot (p_t\, \mathbf{u}_t)$. If we can fit
$\mathbf{v}_{\boldsymbol{\theta}} \approx \mathbf{u}_t$, generation is a plain
ODE solve of $\dot{\mathbf{x}} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$
from $\mathbf{x}_0 \sim p_0$: a continuous normalizing flow
(:numref:`sec_mdl-continuous-normalizing-flows`) trained *without ever
simulating the ODE*. The natural objective is the **flow-matching loss**

$$
\mathcal{L}_{\mathrm{FM}}(\boldsymbol{\theta})
= \mathbb{E}_{t \sim \mathcal{U}[0,1],\ \mathbf{x} \sim p_t}
\left[\, \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x}) \right\|^2 \right],
$$
:eqlabel:`eq_mdl-fm-loss`

Like explicit score matching, this objective contains an unknown marginal
field, the velocity $\mathbf{u}_t$. A tractable alternative constructs the path
out of *conditional* paths, one per data point: pick
$p_t(\mathbf{x} \mid \mathbf{z})$, a little moving blob that starts spread as
noise and collapses onto the conditioning variable $\mathbf{z}$ (say, a data
point $\mathbf{x}_1$, or a pair $(\mathbf{x}_0, \mathbf{x}_1)$), whose
conditional velocity $\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$ we can write
down, and let the marginal path be the mixture
$p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{z})\, q(\mathbf{z})\, d\mathbf{z}$.

**Proposition (the marginal velocity is a posterior mean).** *Suppose each
conditional pair satisfies the continuity equation,
$\partial_t p_t(\mathbf{x} \mid \mathbf{z}) = -\nabla \cdot \left( p_t(\mathbf{x} \mid \mathbf{z})\, \mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \right)$,
with integrability sufficient to differentiate the mixture under the integral
sign, and define on $\{p_t > 0\}$ the* **marginal velocity**

$$
\mathbf{u}_t(\mathbf{x})
= \mathbb{E}\left[ \mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \mid \mathbf{x}_t = \mathbf{x} \right]
= \int \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})\,
\frac{p_t(\mathbf{x} \mid \mathbf{z})\, q(\mathbf{z})}{p_t(\mathbf{x})}\, d\mathbf{z}.
$$
:eqlabel:`eq_mdl-marginal-velocity`

*Then $(p_t, \mathbf{u}_t)$ satisfies the continuity equation: the averaged
field transports the averaged path.*

**Proof.** Differentiate the mixture under the integral sign and substitute
the conditional continuity equation:

$$
\partial_t p_t(\mathbf{x})
= \int \partial_t p_t(\mathbf{x} \mid \mathbf{z})\, q(\mathbf{z})\, d\mathbf{z}
= -\nabla \cdot \int p_t(\mathbf{x} \mid \mathbf{z})\, \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})\, q(\mathbf{z})\, d\mathbf{z}
= -\nabla \cdot \left( p_t(\mathbf{x})\, \mathbf{u}_t(\mathbf{x}) \right),
$$

where the last step is the definition :eqref:`eq_mdl-marginal-velocity`.
$\blacksquare$

### The Conditional Flow Matching Theorem

The target is an intractable posterior mean
:eqref:`eq_mdl-marginal-velocity` of a tractable per-sample quantity, so the
regression lemma :eqref:`eq_mdl-regression-lemma` applies. Define the **conditional flow matching** loss, which needs only samples
$(t, \mathbf{z}, \mathbf{x})$ and the closed-form conditional velocity:

$$
\mathcal{L}_{\mathrm{CFM}}(\boldsymbol{\theta})
= \mathbb{E}_{t,\ \mathbf{z} \sim q,\ \mathbf{x} \sim p_t(\cdot \mid \mathbf{z})}
\left[\, \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \right\|^2 \right].
$$
:eqlabel:`eq_mdl-cfm-loss`

**Theorem (CFM trains the marginal field).** *Under the integrability needed
for :eqref:`eq_mdl-marginal-velocity` to exist,*

$$
\mathcal{L}_{\mathrm{CFM}}(\boldsymbol{\theta})
= \mathcal{L}_{\mathrm{FM}}(\boldsymbol{\theta}) + C
$$

*with $C$ independent of $\boldsymbol{\theta}$. In particular
$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{CFM}} = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{FM}}$:
the two objectives have identical gradients and identical minimizers*
:cite:`Lipman.Chen.BenHamu.ea.2022,Tong.Fatras.Malkin.ea.2023`.

**Proof.** Apply the regression lemma :eqref:`eq_mdl-regression-lemma` with
$X = (\mathbf{x}, t)$ where $\mathbf{x} \sim p_t(\cdot \mid \mathbf{z})$, and
$Y = \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$. The marginal distribution of
$X$ is $t \sim \mathcal{U}[0,1]$, $\mathbf{x} \sim p_t$, and the conditional
mean of $Y$ given $X = (\mathbf{x}, t)$ is, by definition,
:eqref:`eq_mdl-marginal-velocity`, the marginal velocity. The lemma splits
$\mathcal{L}_{\mathrm{CFM}}$ into
$\mathbb{E}\| \mathbf{v}_{\boldsymbol{\theta}}(X) - \mathbf{u}_t(\mathbf{x}) \|^2 = \mathcal{L}_{\mathrm{FM}}$
plus the posterior variance term
$C = \mathbb{E}\| Y - \mathbf{u}_t(\mathbf{x}) \|^2$, which does not involve
$\boldsymbol{\theta}$. $\blacksquare$

This proof has the same structure as Vincent's theorem, with
(score of the noising kernel $\to$ marginal score) replaced by (conditional
velocity $\to$ marginal velocity). Denoising score matching *is* conditional
flow matching for the score field. The flow-matching formulation extends this
argument to general velocity fields. As before, the theorem's constant $C$ is
the conditional variance of the regression target. The population CFM loss
therefore need not vanish even for a perfect marginal field; its value depends
on the chosen path, coupling, and target parameterization.

### Relations Among Score, Noise, and Velocity
:label:`sec_mdl-score-velocity-dictionary`

Diffusion trains a score, whereas flow matching trains a velocity. For the
Gaussian paths used in practice, these quantities determine one another
through explicit transformations.
Condition on a data point and take the Gaussian path

$$
\mathbf{x}_t = \alpha_t\, \mathbf{x}_1 + \sigma_t\, \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I),
$$
:eqlabel:`eq_mdl-gaussian-path`

with smooth schedules $(\alpha_t, \sigma_t)$: on the flow-matching clock
$(\alpha_0, \sigma_0) = (0, 1)$ and $(\alpha_1, \sigma_1) = (1, 0)$; on the
diffusion clock, write $\mathbf{x}_0$ for the data point and swap the boundary
conditions, and every formula below holds verbatim.

**Proposition (score–velocity identity).** *For the Gaussian path above, at
any $t$ with $\alpha_t, \sigma_t > 0$ and on $\{p_t > 0\}$, the marginal
velocity :eqref:`eq_mdl-marginal-velocity` and the marginal score determine
each other:*

$$
\mathbf{u}_t(\mathbf{x})
= \frac{\dot{\alpha}_t}{\alpha_t}\, \mathbf{x}
- \left( \sigma_t \dot{\sigma}_t - \sigma_t^2\, \frac{\dot{\alpha}_t}{\alpha_t} \right)
\nabla \log p_t(\mathbf{x}).
$$
:eqlabel:`eq_mdl-score-velocity`

**Proof.** Both sides are posterior expectations given
$\mathbf{x}_t = \mathbf{x}$, and both are affine in the same one: write
$\hat{\mathbf{x}}_1 = \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]$.
Differentiating the path gives the conditional velocity
$\dot{\mathbf{x}}_t = \dot{\alpha}_t \mathbf{x}_1 + \dot{\sigma}_t \boldsymbol{\epsilon}$
with $\boldsymbol{\epsilon} = (\mathbf{x} - \alpha_t \mathbf{x}_1)/\sigma_t$,
so taking the posterior mean in :eqref:`eq_mdl-marginal-velocity`,

$$
\mathbf{u}_t(\mathbf{x})
= \frac{\dot{\sigma}_t}{\sigma_t}\, \mathbf{x}
+ \left( \dot{\alpha}_t - \alpha_t\, \frac{\dot{\sigma}_t}{\sigma_t} \right) \hat{\mathbf{x}}_1.
$$

Meanwhile the marginal score is the posterior mean of the conditional scores
(Vincent's theorem, verbatim), and the conditional score of the kernel
$\mathcal{N}(\alpha_t \mathbf{x}_1, \sigma_t^2 I)$ is
$(\alpha_t \mathbf{x}_1 - \mathbf{x})/\sigma_t^2$, so Tweedie's formula for
this kernel reads

$$
\nabla \log p_t(\mathbf{x})
= \mathbb{E}\left[ \frac{\alpha_t \mathbf{x}_1 - \mathbf{x}}{\sigma_t^2} \,\middle|\, \mathbf{x}_t = \mathbf{x} \right]
= \frac{\alpha_t \hat{\mathbf{x}}_1 - \mathbf{x}}{\sigma_t^2},
\qquad \textrm{i.e.}\quad
\hat{\mathbf{x}}_1 = \frac{\mathbf{x} + \sigma_t^2\, \nabla \log p_t(\mathbf{x})}{\alpha_t}.
$$

Substitute this into the velocity display above and collect the $\mathbf{x}$
and $\nabla \log p_t$ terms to obtain :eqref:`eq_mdl-score-velocity`.
$\blacksquare$

The same posterior mean $\hat{\mathbf{x}}_1$ underlies the common
parameterizations, so they are inter-convertible by
$t$-dependent affine maps of the same posterior mean
(:numref:`tab_mdl-score-noise-velocity`).

:Network targets on a Gaussian path. Each row gives the quantity a network predicts, the posterior expectation it estimates, and its conversion from the score.
:label:`tab_mdl-score-noise-velocity`

| Network predicts | Posterior meaning | From the score $\mathbf{s} = \nabla \log p_t$ |
| :-- | :-- | :-- |
| score $\mathbf{s}_{\boldsymbol{\theta}}$ | $\nabla \log p_t(\mathbf{x})$ | $\mathbf{s}$ |
| noise $\hat{\boldsymbol{\epsilon}}$ | $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t = \mathbf{x}]$ | $-\sigma_t\, \mathbf{s}$ |
| clean point $\hat{\mathbf{x}}_1$ | $\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]$ | $(\mathbf{x} + \sigma_t^2\, \mathbf{s})/\alpha_t$ (Tweedie) |
| velocity $\mathbf{v}_{\boldsymbol{\theta}}$ | $\mathbb{E}[\dot{\mathbf{x}}_t \mid \mathbf{x}_t = \mathbf{x}]$ | the identity :eqref:`eq_mdl-score-velocity` |
| $v$-prediction $\hat{\mathbf{v}}$ | $\mathbb{E}[\alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]$ | $-\tfrac{\sigma_t}{\alpha_t}\, \mathbf{x} - \sigma_t\, \tfrac{\alpha_t^2 + \sigma_t^2}{\alpha_t}\, \mathbf{s}$ |

Target choice changes the conditioning of the regression problem. Near the
data end, the sampled conditional-score target
$-\boldsymbol{\epsilon}/\sigma_t$ has scale $1/\sigma_t$; the marginal score
need not. Indeed the optimal noise prediction is
$\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]
=-\sigma_t\nabla\log p_t(\mathbf{x}_t)$, which can shrink to zero when the
clean density has a finite score. Noise prediction nevertheless keeps the
*sampled training target* at unit scale. Near the noise end,
$\hat{\mathbf{x}}_1=(\mathbf{x}-\sigma_t\hat{\boldsymbol{\epsilon}})/\alpha_t$
amplifies noise-prediction error by $1/\alpha_t$. Under common normalized
schedules, the $v$-prediction target
$\alpha_t\boldsymbol{\epsilon}-\sigma_t\mathbf{x}_1$ keeps the two sampled
components on comparable scales, which motivates its use in distillation
:cite:`Salimans.Ho.2022`. These are conditioning tendencies, not identities
about every data distribution.

A useful schedule coordinate is the **log signal-to-noise ratio**
$\rho_t=\log(\alpha_t^2/\sigma_t^2)$, using a different symbol from the loss
weight. After the corresponding state rescaling, schedules that traverse the
same monotone range of $\rho$ describe the same family of noised marginals up
to time reparameterization :cite:`Kingma.Salimans.Poole.ea.2021`. Their
velocity fields also acquire the derivative of that reparameterization, so
training weights and numerical difficulty need not be the same. EDM's
$\alpha_t\equiv1$, $\sigma_t=t$ parameterization is a closely related
variance-exploding coordinate choice designed jointly with its preconditioning
and sampler :cite:`Karras.Aittala.Aila.ea.2022`.

The identity is checkable to machine precision with the section's own mixture:
at a fixed $t$, compute the marginal velocity once from the posterior mean
(route one, no score in sight) and once from the dictionary formula
:eqref:`eq_mdl-score-velocity` with the analytic score (route two, no
velocity in sight).

```{.python .input #mdl-score-matching-diffusion-flow-score-noise-and-velocity-are-one-function}
t = 0.6                                       # one fixed time, cosine schedule
alpha, sigma = np.sin(np.pi * t / 2), np.cos(np.pi * t / 2)
dalpha, dsigma = np.pi / 2 * sigma, -np.pi / 2 * alpha
grid = np.linspace(-4.0, 4.0, 201)
means, var0 = np.array([-2.0, 2.0]), 0.25     # the mixture of the DSM demo
vart = alpha**2 * var0 + sigma**2             # noised component variance
w = np.stack([np.exp(-(grid - alpha * m)**2 / (2 * vart)) for m in means])
w /= w.sum(0)                                 # posterior mode responsibilities
x1_hat = (w * np.stack([m + alpha * var0 / vart * (grid - alpha * m)
                        for m in means])).sum(0)          # E[x_1 | x_t]
u_posterior = dalpha * x1_hat + dsigma / sigma * (grid - alpha * x1_hat)
score = (w * np.stack([(alpha * m - grid) / vart for m in means])).sum(0)
u_dictionary = (dalpha / alpha) * grid \
    - (sigma * dsigma - sigma**2 * dalpha / alpha) * score
print(f'max |u_posterior - u_dictionary| on [-4, 4]: '
      f'{np.abs(u_posterior - u_dictionary).max():.1e}')
```

The two routes agree to about $10^{-15}$ (machine precision) across both
modes and the low-density valley. No model was fitted: the posterior
calculation does not use a score, while the alternative calculation uses the
score--velocity relation. They agree because both are affine in the same
posterior mean $\hat{x}_1$. Thus noise-prediction and velocity-field
implementations can represent equivalent targets.

### Rectified Flow and Straight Paths
:label:`sec_mdl-rectified-flow`

Consider a conditional path defined by a *pair*
$\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1)$, a noise sample and a data
sample drawn independently, and connect them by a straight line traversed at
constant speed:

$$
\mathbf{x}_t = (1 - t)\, \mathbf{x}_0 + t\, \mathbf{x}_1,
\qquad
\mathbf{u}_t(\mathbf{x}_t \mid \mathbf{z}) = \dot{\mathbf{x}}_t = \mathbf{x}_1 - \mathbf{x}_0.
$$
:eqlabel:`eq_mdl-rf-path`

The conditional velocity does not even depend on $t$: the CFM loss
:eqref:`eq_mdl-cfm-loss` becomes the **rectified flow** (equivalently,
linear-path CFM) objective
:cite:`Liu.Gong.Liu.2022,Lipman.Chen.BenHamu.ea.2022`

$$
\mathcal{L}_{\mathrm{RF}}(\boldsymbol{\theta})
= \mathbb{E}_{t,\ \mathbf{x}_0 \sim p_0,\ \mathbf{x}_1 \sim p_1}
\left[\, \left\| \mathbf{v}_{\boldsymbol{\theta}}\big( (1 - t) \mathbf{x}_0 + t \mathbf{x}_1,\ t \big)
- (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2 \right]
$$
:eqlabel:`eq_mdl-rf-loss`

Training draws noise and data, interpolates between them, and regresses on the
difference. (For the measure-theoretic
comfort of strictly positive conditional densities, smooth the line with an
infinitesimal Gaussian, $p_t(\cdot \mid \mathbf{z}) = \mathcal{N}((1-t)\mathbf{x}_0 + t \mathbf{x}_1, \sigma_{\min}^2 I)$,
and let $\sigma_{\min} \to 0$; the limiting objective is unchanged. Gaussian
conditional
paths with general $(\mu_t, \sigma_t)$ schedules recover diffusion-style
targets: that is how flow matching subsumes the VP path, modulo the
time-reversal callout above.)

![Straight conditional segments connect independent noise--data pairs and may intersect. The induced marginal velocity averages the directions of segments passing through each point, while its ODE trajectories remain nonintersecting and therefore curve. Reflow re-couples endpoints using the model's ODE to reduce this curvature.](../img/mdl-dyn-fm-paths.svg)
:label:`fig_mdl-dyn-fm-paths`

A subtlety (:numref:`fig_mdl-dyn-fm-paths`): the *conditional* paths are straight, but the *marginal*
flow that the network learns is generally **curved**. Two straight segments
that cross at $(\mathbf{x}, t)$ feed the posterior mean
:eqref:`eq_mdl-marginal-velocity` two different directions, and the learned
field (which, like any function, can have only one value there) averages
them. An ODE's trajectories cannot meet at the same $(\mathbf{x}, t)$ (uniqueness,
:numref:`sec_mdl-ode-existence-uniqueness`), so the learned flow bends to
avoid the collisions that the conditional segments ignore. The independent
coupling of $\mathbf{x}_0$ and $\mathbf{x}_1$ can produce many crossings and a
curved marginal field, which can increase low-order solver error. **Reflow**
:cite:`Liu.Gong.Liu.2022` attacks the coupling: after training, generate pairs
$(\mathbf{x}_0, \hat{\mathbf{x}}_1)$ by running the current model's ODE, and retrain on
this new coupling, in which start and end points are already dynamically
matched. For the exact population rectification operator, the theorem preserves
endpoint marginals and does not increase convex transport costs. A finite
network trained on numerically generated endpoints only approximates this
operator.
In the straight limit, one Euler step is exact (the local truncation error of
Euler is controlled by the curvature $\ddot{\mathbf{x}}$ along trajectories,
:numref:`sec_mdl-euler-runge-kutta`); this is the mathematics behind few-step
and one-step generators distilled from flows.

### Gaussian to Two Moons, Four Ways

The numerical example uses a two-moons target (two interleaved
crescents, a classic stress test for mode-splitting) generated in a few lines
of NumPy; the source is a standard 2-D Gaussian. We also define the **energy
distance** :cite:`Szekely.Rizzo.2013`

$$
\mathcal{E}(P, Q) = \left( 2\, \mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\| \right)^{1/2},
$$

an MMD-style two-sample discrepancy
(:numref:`sec_mdl-ipm-mmd`) that is zero iff the distributions agree; we use
its square, on $2048$-point samples, to compare generated samples with a
held-out target sample throughout.

```{.python .input #score-matching-diffusion-flow-two-moons}
def two_moons(n, rng, noise=0.07):
    t = rng.uniform(0.0, np.pi, n)
    m = rng.integers(0, 2, n)                      # which moon
    X = np.stack([np.cos(t) * (1 - 2 * m) + m,
                  np.sin(t) * (1 - 2 * m) + 0.5 * m], axis=1)
    return X + noise * rng.standard_normal((n, 2))

rng = np.random.default_rng(0)
raw = two_moons(8192, rng)
mu, sd = raw.mean(0), raw.std(0)
moons = (raw - mu) / sd                            # standardized training data
held_out = (two_moons(2048, np.random.default_rng(1)) - mu) / sd

def energy_distance(a, b):                         # squared energy distance
    d = lambda u, v: np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1).mean()
    return 2 * d(a, b) - d(a, a) - d(b, b)

print(f'training set {moons.shape}; energy distance of a fresh sample '
      f'to the held-out set: {energy_distance(moons[:2048], held_out):.4f}')
```

The fresh-sample-to-held-out value ($\approx 0.001$ in this draw) is a
finite-sample reference for the metric, not a universal lower bound. The model
is a velocity field
$\mathbf{v}_{\boldsymbol{\theta}} : \mathbb{R}^2 \times [0, 1] \to \mathbb{R}^2$
as a $3 \to 64 \to 64 \to 2$ tanh MLP, trained for $4000$ Adam steps on the
rectified-flow objective :eqref:`eq_mdl-rf-loss`. Each batch follows the same
construction: sample $\mathbf{x}_0$, $\mathbf{x}_1$, and $t$; interpolate; regress on
$\mathbf{x}_1 - \mathbf{x}_0$. This is the section's second training loop (the
NumPy score network was the first), and it takes a few seconds on a CPU.

```{.python .input #score-matching-diffusion-flow-cfm-train}
#@tab mxnet
mxnp.random.seed(0)
data = mxnp.array(moons)
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(64, activation='tanh'),
        gluon.nn.Dense(64, activation='tanh'),
        gluon.nn.Dense(2))
net.initialize(init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 3e-3})
for step in range(4001):
    x1 = data[mxnp.random.randint(0, len(data), (256,))]
    x0 = mxnp.random.normal(0, 1, (256, 2))
    t = mxnp.random.uniform(0, 1, (256, 1))
    xt = (1 - t) * x0 + t * x1
    with autograd.record():
        v = net(mxnp.concatenate([xt, t], axis=1))
        loss = ((v - (x1 - x0))**2).mean()
    loss.backward()
    trainer.step(1)
    if step % 1000 == 0:
        print(f'step {step:4d}: CFM loss {float(loss):.3f}')
```

```{.python .input #score-matching-diffusion-flow-cfm-train}
#@tab pytorch
torch.manual_seed(0)
data = torch.tensor(moons, dtype=torch.float32)
net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                    nn.Linear(64, 2))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for step in range(4001):
    x1 = data[torch.randint(0, len(data), (256,))]
    x0, t = torch.randn(256, 2), torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1
    loss = ((net(torch.cat([xt, t], 1)) - (x1 - x0))**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f'step {step:4d}: CFM loss {loss.item():.3f}')
```

```{.python .input #score-matching-diffusion-flow-cfm-train}
#@tab tensorflow
tf.keras.utils.set_random_seed(0)
data = tf.constant(moons, tf.float32)
net = tf.keras.Sequential([tf.keras.layers.Dense(64, 'tanh'),
                           tf.keras.layers.Dense(64, 'tanh'),
                           tf.keras.layers.Dense(2)])
opt = tf.keras.optimizers.Adam(3e-3)

@tf.function
def train_step():
    x1 = tf.gather(data, tf.random.uniform((256,), 0, len(data), tf.int32))
    x0 = tf.random.normal((256, 2))
    t = tf.random.uniform((256, 1))
    xt = (1 - t) * x0 + t * x1
    with tf.GradientTape() as tape:
        v = net(tf.concat([xt, t], 1))
        loss = tf.reduce_mean((v - (x1 - x0))**2)
    grads = tape.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss

for step in range(4001):
    loss = train_step()
    if step % 1000 == 0:
        print(f'step {step:4d}: CFM loss {float(loss):.3f}')
```

```{.python .input #score-matching-diffusion-flow-cfm-train}
#@tab jax
def init_mlp(key, sizes=(3, 64, 64, 2)):
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (i, o)) / jnp.sqrt(i), jnp.zeros(o))
            for k, i, o in zip(keys, sizes[:-1], sizes[1:])]

def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    return x @ params[-1][0] + params[-1][1]

data = jnp.asarray(moons, dtype=jnp.float32)

def cfm_loss(params, key):
    k1, k2, k3 = jax.random.split(key, 3)
    x1 = data[jax.random.randint(k1, (256,), 0, len(data))]
    x0 = jax.random.normal(k2, (256, 2))
    t = jax.random.uniform(k3, (256, 1))
    xt = (1 - t) * x0 + t * x1
    v = mlp(params, jnp.concatenate([xt, t], axis=1))
    return ((v - (x1 - x0))**2).mean()

opt = optax.adam(3e-3)
params = init_mlp(jax.random.key(0))
state = opt.init(params)

@jax.jit
def train(params, state, key):
    def step(carry, k):
        params, state = carry
        loss, grads = jax.value_and_grad(cfm_loss)(params, k)
        updates, state = opt.update(grads, state)
        return (optax.apply_updates(params, updates), state), loss
    return jax.lax.scan(step, (params, state), jax.random.split(key, 4000))

(params, state), losses = train(params, state, jax.random.key(1))
print(f'CFM loss: step 0 {losses[0]:.3f} -> step 4000 {losses[-1]:.3f}')
```

The loss falls from about $2$ to about $1.3$ and then plateaus, as the CFM
theorem predicts. This value reflects the conditional variance of
$\mathbf{x}_1 - \mathbf{x}_0$ around its posterior mean rather than zero. Sampling is an Euler loop
(:numref:`sec_mdl-euler-runge-kutta`) integrating
$\dot{\mathbf{x}} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ from
$t = 0$ to $1$. For this trained field, the panels show how the selected
metric and sample geometry change with step count: one step produces a smeared
blob, two steps a bent ellipse, eight
steps recognizable moons, thirty-two steps sharp ones.

```{.python .input #score-matching-diffusion-flow-cfm-sample}
#@tab mxnet
def euler_sample(n, steps, seed=2):
    mxnp.random.seed(seed)
    q = mxnp.random.normal(0, 1, (n, 2))
    for k in range(steps):
        t = mxnp.full((n, 1), k / steps)
        q = q + (1.0 / steps) * net(mxnp.concatenate([q, t], axis=1))
    return q.asnumpy()
```

```{.python .input #score-matching-diffusion-flow-cfm-sample}
#@tab pytorch
def euler_sample(n, steps, seed=2):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 2, generator=g)
    for k in range(steps):
        t = torch.full((n, 1), k / steps)
        with torch.no_grad():
            q = q + (1.0 / steps) * net(torch.cat([q, t], 1))
    return q.numpy()
```

```{.python .input #score-matching-diffusion-flow-cfm-sample}
#@tab tensorflow
def euler_sample(n, steps, seed=2):
    q = tf.random.stateless_normal((n, 2), [seed, 0])
    for k in range(steps):
        t = tf.fill((n, 1), tf.cast(k / steps, tf.float32))
        q = q + (1.0 / steps) * net(tf.concat([q, t], 1))
    return q.numpy()
```

```{.python .input #score-matching-diffusion-flow-cfm-sample}
#@tab jax
velocity = jax.jit(
    lambda params, q, t: mlp(params, jnp.concatenate([q, t], axis=1)))

def euler_sample(n, steps, seed=2):
    q = jax.random.normal(jax.random.key(seed), (n, 2))
    for k in range(steps):
        q = q + (1.0 / steps) * velocity(params, q, jnp.full((n, 1), k / steps))
    return np.asarray(q)
```

```{.python .input #score-matching-diffusion-flow-cfm-panels}
panels = [('data', moons[:2048])] + [
    (f'{K} step(s)', euler_sample(2048, K)) for K in (1, 2, 8, 32)]
fig, axes = d2l.plt.subplots(1, 5, figsize=(11, 2.4), sharex=True, sharey=True)
for ax, (title, s) in zip(axes, panels):
    ax.scatter(s[:, 0], s[:, 1], s=1)
    ax.set_title(title)
    ax.set_xlim(-2.5, 2.5), ax.set_ylim(-2.5, 2.5)
```

For this model, a few Euler steps already recover the main geometry, while the
remaining discrepancy decreases more slowly. The next section gives a precise
energy interpretation of straight paths; it does not equate that energy with a
universal step count.

### The Effect of One Reflow Round

Before leaving the trained model, we can test the intended effect of reflow on
the learned paths. Apply the procedure of :numref:`sec_mdl-rectified-flow`:
draw fresh noise $\mathbf{z}$,
integrate the trained ODE for $32$ Euler steps to obtain the model's own
endpoint $\hat{\mathbf{x}}_1(\mathbf{z})$, and retrain the *same architecture*
on the coupled pairs $(\mathbf{z}, \hat{\mathbf{x}}_1(\mathbf{z}))$ in place of
independent ones. If the new couplings rarely cross, the retrained flow should
be nearly straight, and a nearly straight flow should sample well in *one*
Euler step.

```{.python .input #mdl-score-matching-diffusion-flow-one-reflow-round-measured}
#@tab mxnet
mxnp.random.seed(3)
z = mxnp.random.normal(0, 1, (8192, 2))            # noise endpoints, kept
x1_hat = z.copy()
for k in range(32):                                # the model's own couplings
    t = mxnp.full((8192, 1), k / 32)
    x1_hat = x1_hat + (1.0 / 32) * net(mxnp.concatenate([x1_hat, t], axis=1))
net2 = gluon.nn.Sequential()
net2.add(gluon.nn.Dense(64, activation='tanh'),
         gluon.nn.Dense(64, activation='tanh'),
         gluon.nn.Dense(2))
net2.initialize(init.Xavier())
trainer2 = gluon.Trainer(net2.collect_params(), 'adam', {'learning_rate': 3e-3})
for step in range(4001):
    i = mxnp.random.randint(0, len(z), (256,))
    x0, x1 = z[i], x1_hat[i]                       # the same pair, never re-paired
    t = mxnp.random.uniform(0, 1, (256, 1))
    xt = (1 - t) * x0 + t * x1
    with autograd.record():
        v = net2(mxnp.concatenate([xt, t], axis=1))
        loss = ((v - (x1 - x0))**2).mean()
    loss.backward()
    trainer2.step(1)
print(f'final reflow loss {float(loss):.3f}')

def euler_sample2(n, steps, seed=2):
    mxnp.random.seed(seed)
    q = mxnp.random.normal(0, 1, (n, 2))
    for k in range(steps):
        t = mxnp.full((n, 1), k / steps)
        q = q + (1.0 / steps) * net2(mxnp.concatenate([q, t], axis=1))
    return q.asnumpy()

for K in (1, 2, 32):
    print(f'{K:2d} step(s): energy distance  '
          f'CFM {energy_distance(euler_sample(2048, K), held_out):.3f}  ->  '
          f'reflow {energy_distance(euler_sample2(2048, K), held_out):.3f}')
```

```{.python .input #mdl-score-matching-diffusion-flow-one-reflow-round-measured}
#@tab pytorch
g = torch.Generator().manual_seed(3)
z = torch.randn(8192, 2, generator=g)              # noise endpoints, kept
x1_hat = z.clone()
for k in range(32):                                # the model's own couplings
    t = torch.full((8192, 1), k / 32)
    with torch.no_grad():
        x1_hat = x1_hat + (1.0 / 32) * net(torch.cat([x1_hat, t], 1))
net2 = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                     nn.Linear(64, 2))
opt2 = torch.optim.Adam(net2.parameters(), lr=3e-3)
for step in range(4001):
    i = torch.randint(0, len(z), (256,))
    x0, x1 = z[i], x1_hat[i]                       # the same pair, never re-paired
    t = torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1
    loss = ((net2(torch.cat([xt, t], 1)) - (x1 - x0))**2).mean()
    opt2.zero_grad()
    loss.backward()
    opt2.step()
print(f'final reflow loss {loss.item():.3f}')

def euler_sample2(n, steps, seed=2):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 2, generator=g)
    for k in range(steps):
        t = torch.full((n, 1), k / steps)
        with torch.no_grad():
            q = q + (1.0 / steps) * net2(torch.cat([q, t], 1))
    return q.numpy()

for K in (1, 2, 32):
    print(f'{K:2d} step(s): energy distance  '
          f'CFM {energy_distance(euler_sample(2048, K), held_out):.3f}  ->  '
          f'reflow {energy_distance(euler_sample2(2048, K), held_out):.3f}')
```

```{.python .input #mdl-score-matching-diffusion-flow-one-reflow-round-measured}
#@tab tensorflow
tf.keras.utils.set_random_seed(3)
z = tf.random.stateless_normal((8192, 2), [3, 0])  # noise endpoints, kept
x1_hat = z
for k in range(32):                                # the model's own couplings
    t = tf.fill((8192, 1), tf.cast(k / 32, tf.float32))
    x1_hat = x1_hat + (1.0 / 32) * net(tf.concat([x1_hat, t], 1))
net2 = tf.keras.Sequential([tf.keras.layers.Dense(64, 'tanh'),
                            tf.keras.layers.Dense(64, 'tanh'),
                            tf.keras.layers.Dense(2)])
opt2 = tf.keras.optimizers.Adam(3e-3)

@tf.function
def reflow_step():
    i = tf.random.uniform((256,), 0, len(z), tf.int32)
    x0, x1 = tf.gather(z, i), tf.gather(x1_hat, i) # the same pair, never re-paired
    t = tf.random.uniform((256, 1))
    xt = (1 - t) * x0 + t * x1
    with tf.GradientTape() as tape:
        v = net2(tf.concat([xt, t], 1))
        loss = tf.reduce_mean((v - (x1 - x0))**2)
    grads = tape.gradient(loss, net2.trainable_variables)
    opt2.apply_gradients(zip(grads, net2.trainable_variables))
    return loss

for step in range(4001):
    loss = reflow_step()
print(f'final reflow loss {float(loss):.3f}')

def euler_sample2(n, steps, seed=2):
    q = tf.random.stateless_normal((n, 2), [seed, 0])
    for k in range(steps):
        t = tf.fill((n, 1), tf.cast(k / steps, tf.float32))
        q = q + (1.0 / steps) * net2(tf.concat([q, t], 1))
    return q.numpy()

for K in (1, 2, 32):
    print(f'{K:2d} step(s): energy distance  '
          f'CFM {energy_distance(euler_sample(2048, K), held_out):.3f}  ->  '
          f'reflow {energy_distance(euler_sample2(2048, K), held_out):.3f}')
```

```{.python .input #mdl-score-matching-diffusion-flow-one-reflow-round-measured}
#@tab jax
z = jax.random.normal(jax.random.key(3), (8192, 2))   # noise endpoints, kept
x1_hat = z
for k in range(32):                                # the model's own couplings
    x1_hat = x1_hat + (1.0 / 32) * velocity(params, x1_hat,
                                            jnp.full((8192, 1), k / 32))

def reflow_loss(params2, key):
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (256,), 0, len(z))
    x0, x1 = z[i], x1_hat[i]                       # the same pair, never re-paired
    t = jax.random.uniform(k2, (256, 1))
    xt = (1 - t) * x0 + t * x1
    v = mlp(params2, jnp.concatenate([xt, t], axis=1))
    return ((v - (x1 - x0))**2).mean()

params2 = init_mlp(jax.random.key(4))
state2 = opt.init(params2)

@jax.jit
def retrain(params2, state2, key):
    def step(carry, k):
        params2, state2 = carry
        loss, grads = jax.value_and_grad(reflow_loss)(params2, k)
        updates, state2 = opt.update(grads, state2)
        return (optax.apply_updates(params2, updates), state2), loss
    return jax.lax.scan(step, (params2, state2), jax.random.split(key, 4000))

(params2, state2), losses = retrain(params2, state2, jax.random.key(5))
print(f'final reflow loss {losses[-1]:.3f}')

def euler_sample2(n, steps, seed=2):
    q = jax.random.normal(jax.random.key(seed), (n, 2))
    for k in range(steps):
        q = q + (1.0 / steps) * velocity(params2, q, jnp.full((n, 1), k / steps))
    return np.asarray(q)

for K in (1, 2, 32):
    print(f'{K:2d} step(s): energy distance  '
          f'CFM {energy_distance(euler_sample(2048, K), held_out):.3f}  ->  '
          f'reflow {energy_distance(euler_sample2(2048, K), held_out):.3f}')
```

In this run, one Euler step of the reflowed model
scores $0.016$, close to the original model's $32$-step value of $0.014$ in
this finite comparison, and about forty times smaller than the original
one-step value $0.676$. The
training loss also falls to about $0.001$ instead of plateauing near $1.3$.
This is consistent with lower conditional-target variance under the
model-generated coupling, together with the fit achieved by the second
network; the experiment does not estimate these contributions separately.
Two caveats remain. The exact
population rectification operator preserves endpoint marginals and does not
increase convex transport costs; progressive straightening concerns repeated
idealized rounds. What we measured is one finite round of an imperfectly
trained model. Moreover, the retrained target is the model's $32$-step endpoint law, not the
data law, so the first model's bias is inherited by the second. The experiment
therefore demonstrates the mechanism on this two-dimensional problem; it does
not establish one-step equivalence for other models or datasets.

## Optimal Transport and Straightness
:label:`sec_mdl-ot-connection`

Optimal transport specifies the precise sense in which straight paths can be
optimal. We keep this discussion self-contained: :numref:`sec_mdl-optimal-transport` develops the
Kantorovich-dual $W_1$ picture behind WGANs, but here we need the *quadratic*
cost and its dynamic, fluid-flow formulation.

A **coupling** of two distributions $p_0, p_1$ on $\mathbb{R}^d$ is a joint
distribution $\pi$ with marginals $p_0$ and $p_1$, a randomized
transportation plan saying how much mass travels from each source location to
each destination. The **2-Wasserstein distance** is the cheapest plan under
quadratic cost:

$$
W_2^2(p_0, p_1) = \min_{\pi \in \Pi(p_0, p_1)}\ \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi}
\left[\, \| \mathbf{x}_1 - \mathbf{x}_0 \|^2 \right].
$$
:eqlabel:`eq_mdl-w2`

(Under mild conditions, e.g. $p_0$ with a density, the optimal plan is
deterministic, a *map* $\mathbf{x}_1 = T(\mathbf{x}_0)$ with $T$ the gradient
of a convex function; that is Brenier's theorem :cite:`Brenier.1991`, and we
will not need it beyond intuition.) What we need is the reformulation of
:eqref:`eq_mdl-w2` as a *least-action principle over flows*, due to
:citet:`Benamou.Brenier.2000`: the static matching problem equals
a minimum over exactly the objects flow matching trains.

**Theorem (Benamou–Brenier, dynamic formulation).** *Let $p_0$ and $p_1$
have finite second moments. Over all pairs $(p_t, \mathbf{v}_t)$, regular
enough that the flow of $\mathbf{v}_t$ exists, satisfying the continuity
equation $\partial_t p_t = -\nabla \cdot (p_t \mathbf{v}_t)$ with the
prescribed endpoints $p_0$ and $p_1$,*

$$
W_2^2(p_0, p_1)
= \min_{(p_t, \mathbf{v}_t)}\ \int_0^1 \int \| \mathbf{v}_t(\mathbf{x}) \|^2\, p_t(\mathbf{x})\, d\mathbf{x}\, dt
$$
:eqlabel:`eq_mdl-benamou-brenier`

*The squared distance is the least kinetic energy of any flow carrying $p_0$
to $p_1$, and the minimizing flow transports each particle along a straight
line at constant speed.*

**Proof sketch (the lower bound, via Jensen).** Take any admissible
$(p_t, \mathbf{v}_t)$ and let $\mathbf{X}_t$ solve the ODE
$\dot{\mathbf{X}}_t = \mathbf{v}_t(\mathbf{X}_t)$ with
$\mathbf{X}_0 \sim p_0$. Two facts are granted here: that this ODE is
solvable, and that its law is the continuity equation's unique solution, so
that $\mathbf{X}_t \sim p_t$ for all $t$
(:numref:`sec_mdl-continuity-equation`; the uniqueness argument is the
probability-flow twin of :numref:`sec_mdl-probability-flow-ode`). In
particular $(\mathbf{X}_0, \mathbf{X}_1)$ is a coupling of $(p_0, p_1)$. Then

$$
\begin{aligned}
W_2^2(p_0, p_1)
&\le \mathbb{E} \left\| \mathbf{X}_1 - \mathbf{X}_0 \right\|^2
= \mathbb{E} \left\| \int_0^1 \mathbf{v}_t(\mathbf{X}_t)\, dt \right\|^2 \\
&\le \mathbb{E} \int_0^1 \left\| \mathbf{v}_t(\mathbf{X}_t) \right\|^2 dt
= \int_0^1\!\! \int \|\mathbf{v}_t\|^2\, p_t\, d\mathbf{x}\, dt,
\end{aligned}
$$

where the first inequality is suboptimality of this particular coupling and
the second is Jensen's inequality (:numref:`subsec_mdl-jensen`) applied to the
time average inside the squared norm. So *every* admissible flow has kinetic
energy at least $W_2^2$. For the matching upper bound, transport along the
optimal plan in straight lines at constant speed,
$\mathbf{X}_t = (1 - t) \mathbf{X}_0 + t\, \mathbf{X}_1$ with
$(\mathbf{X}_0, \mathbf{X}_1) \sim \pi^\star$: its kinetic energy is
$\mathbb{E} \|\mathbf{X}_1 - \mathbf{X}_0\|^2 = W_2^2$ exactly. (When
$\pi^\star$ is a map, this displacement interpolation is realized by a genuine
velocity field; smoothing handles the general case.)

The equality analysis explains straightness. For a fixed pair of endpoints,
Jensen's inequality is tight iff the particle has constant velocity, hence
follows their straight segment. Attaining the global $W_2^2$ floor additionally
requires an *optimal endpoint coupling*. Curvature raises the kinetic energy
for a fixed coupling, but straight segments under an arbitrary coupling need
not be optimal transport. This distinction organizes the methods in this section:

* **Diffusion / probability-flow trajectories** can be curved. For a fixed
  endpoint coupling, curvature raises kinetic energy relative to constant-speed
  straight motion and can increase the leading error of a low-order solver.
  It does not, by itself, determine a universal step count.
* **Rectified flow** starts from straight *conditional* segments (each pair in
  :eqref:`eq_mdl-rf-path` is a constant-speed line), while averaging velocities
  at locations reached by several pairs can bend the learned marginal flow.
  **Reflow** often reduces this curvature by replacing independent pairs with
  pairs generated by the current model. It does not by itself certify the
  optimal-transport coupling; in more than one dimension, noncrossing alone
  implies neither straightness nor optimality.
* **Minibatch OT couplings** reduce inefficient pairings before training: within
  each batch, re-pair the noise and data samples by solving a small discrete
  OT problem (an assignment over $256$ points) and run CFM on the matched
  pairs :cite:`Tong.Fatras.Malkin.ea.2023,Pooladian.BenHamu.DomingoEnrich.ea.2023`.
  Matched pairs often cross less, which can reduce conditional-target variance
  and marginal-path curvature. A minibatch plan remains an approximation, not
  a certificate of the Benamou--Brenier minimizer.

One caveat: exact OT in high dimension is expensive
and minibatch plans are biased toward their batch, so OT-CFM and reflow
should be viewed as *variance- and curvature-reduction methods* whose idealized
limit is the dynamic OT problem, rather than as exact $W_2$ solvers.

## Numerical Sampling of Learned Dynamics
:label:`sec_mdl-sampling-learned-dynamics`

Training produces a score $\mathbf{s}_{\boldsymbol{\theta}}$ or velocity
$\mathbf{v}_{\boldsymbol{\theta}}$. Generation substitutes this function into
the corresponding dynamics and integrates from the reference distribution to
the data distribution,

$$
\underbrace{\dot{\mathbf{x}} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)
\quad \textrm{or} \quad
\dot{\mathbf{x}} = \mathbf{f} - \tfrac{1}{2} g^2\, \mathbf{s}_{\boldsymbol{\theta}}}_{\textrm{deterministic (ODE)}}
\qquad \textrm{versus} \qquad
\underbrace{d\mathbf{x} = \left[ \mathbf{f} - g^2\, \mathbf{s}_{\boldsymbol{\theta}} \right] dt + g\, d\bar{\mathbf{W}}}_{\textrm{stochastic (reverse SDE)}},
$$

with $\mathbf{f}$ and $g$ the forward drift and diffusion of
:numref:`sec_mdl-sdes` and $\bar{\mathbf{W}}$ a reverse-time Brownian motion:
the stochastic branch is Anderson's reverse SDE
:eqref:`eq_mdl-dyn-reverse-sde`, and both branches reuse, unchanged, the
solvers of :numref:`sec_mdl-euler-runge-kutta` and
:numref:`sec_mdl-euler-maruyama`. The ODE route is deterministic (the same
$\mathbf{x}_T$ yields the same sample, useful for interpolation and editing)
and often works with fewer steps for suitably trained, low-curvature fields.
It also supports likelihood evaluation through the continuous-normalizing-flow
trace integral of :numref:`sec_mdl-continuous-normalizing-flows`; this is exact
only for the exact field, divergence, and numerical integration. The SDE route
injects fresh noise and can improve exploration or empirical robustness, but
noise does not universally contract model or discretization error, nor does it
by itself guarantee sample diversity. Predictor–corrector methods combine the
two styles. The remaining numerical choice is the number of steps. The
following cell measures its effect by reusing the
trained two-moons velocity field and evaluates Euler sampling at increasing
step counts with the squared energy distance.

```{.python .input #score-matching-diffusion-flow-steps-quality}
steps_list = [1, 2, 4, 8, 16, 32, 64]
eds = [energy_distance(euler_sample(2048, K), held_out) for K in steps_list]
print('  '.join(f'{K}: {e:.3f}' for K, e in zip(steps_list, eds)))
d2l.plot(steps_list, eds, 'Euler steps', 'squared energy distance',
         xscale='log', yscale='log')
```

The squared energy distance falls from $0.68$ at one step
to $0.16$ at two and $0.05$ at four, reaches $0.02$ by eight, and flattens
near $0.015$ from sixteen steps on (numbers from one run). The initial decline
is consistent with decreasing Euler discretization error. The later plateau
can combine field error with finite-sample variability of the metric; the
$0.001$ fresh-sample value printed earlier is a reference for that variability,
not a universal lower bound. This experiment alone does not identify a unique
error source.

Solver order is the other lever. The probability-flow ODE with the *exact*
score of our 1-D mixture under the VP schedule lets us isolate pure
discretization error, with no learning in the loop. We integrate from $t = 1$
(noise) to $t = 0$ (data) with Euler and with Heun's method, the
order-2 scheme of :numref:`sec_mdl-euler-runge-kutta` at two field evaluations
(NFE) per step, and measure the endpoint error against a finely-resolved
reference solution of the same initial points.

```{.python .input #score-matching-diffusion-flow-euler-vs-heun}
rng = np.random.default_rng(17)
bmin, bmax = 0.1, 20.0
beta_fn = lambda t: bmin + t * (bmax - bmin)            # VP noise schedule
abar_fn = lambda t: np.exp(-(bmin * t + 0.5 * (bmax - bmin) * t**2))

def score_vp(q, t, means=(-2.0, 2.0), var=0.25):        # exact mixture score
    a = abar_fn(t)
    v = a * var + (1 - a)
    w = np.stack([np.exp(-(q - np.sqrt(a) * m)**2 / (2 * v)) for m in means])
    return (w * np.stack([(np.sqrt(a) * m - q) / v
                          for m in means])).sum(0) / w.sum(0)

def pf_ode(q, t):                                       # probability-flow ODE
    return -0.5 * beta_fn(t) * (q + score_vp(q, t))

def solve(q, K, heun=False):
    ts = np.linspace(1.0, 0.0, K + 1)
    for t1, t2 in zip(ts[:-1], ts[1:]):
        h, d1 = t2 - t1, pf_ode(q, t1)
        q = q + 0.5 * h * (d1 + pf_ode(q + h * d1, t2)) if heun \
            else q + h * d1
    return q

z = rng.standard_normal(8000)
ref = solve(z, 800, heun=True)                          # fine reference
for K in (2, 5, 10, 20, 40):
    print(f'K = {K:2d} steps: endpoint error  '
          f'Euler {np.abs(solve(z, K) - ref).mean():.4f} ({K} NFE)   '
          f'Heun {np.abs(solve(z, K, heun=True) - ref).mean():.4f} ({2 * K} NFE)')
```

Doubling Euler's steps halves its error (order one); doubling Heun's cuts it
roughly fourfold (order two), so Heun at $20$ steps ($40$ NFE) already beats
Euler at $40$. EDM combines this second-order correction with a schedule,
preconditioning, and stochasticity choices designed as a system
:cite:`Karras.Aittala.Aila.ea.2022`; its reported NFE improvements are
empirical properties of that design, not a consequence of Heun's order alone.
Beyond that lie the
topics of the main book's generative-models chapters: consistency models,
which distill a diffusion teacher into a one-step generator
:cite:`Song.Dhariwal.Chen.ea.2023`; latent diffusion, which runs all of this
in an autoencoder's latent space :cite:`Rombach.Blattmann.Lorenz.ea.2022`;
and discrete diffusion for text :cite:`Austin.Johnson.Ho.ea.2021`.

### Comparison of Model Families
:label:`sec_mdl-unifying-table`

The following table compares the probability path, regression target, and
sampler used by each model family.

| Model family | Object learned | Training loss | Sampler | Stochastic? |
| :-- | :-- | :-- | :-- | :-- |
| **DDPM** :cite:`ho2020denoising` | $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$, an equivalent score parameterization | $\mathbb{E} \lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}} \rVert^2$ (= DSM, $\lambda(t) = 1 - \bar{\alpha}_t$) | ancestral reverse chain; the original convention used about $1000$ discrete levels | yes |
| **Score SDE (VE/VP)** :cite:`song2021score` | $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) \approx \nabla \log p_t$ | noise-conditional DSM :eqref:`eq_mdl-ncsm-loss` | reverse SDE via Euler–Maruyama; + Langevin corrector | yes |
| **Probability-flow ODE** :cite:`song2021score` | same $\mathbf{s}_{\boldsymbol{\theta}}$ (shared training) | same | ODE solver (Euler/Heun/RK); likelihood identity for exact score, terminal density, divergence, and integration | no |
| **DDIM** :cite:`Song.Meng.Ermon.2020` | same $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ as DDPM (no retraining) | same as DDPM | deterministic update :eqref:`eq_mdl-ddim-update` on a sparse time grid | no ($\eta$ interpolates) |
| **Flow matching / rectified flow** :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022` | velocity $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ | CFM :eqref:`eq_mdl-cfm-loss`; linear path: $\mathbb{E} \lVert \mathbf{v}_{\boldsymbol{\theta}} - (\mathbf{x}_1 - \mathbf{x}_0) \rVert^2$ | ODE solver; step count depends on the learned field, path, and integrator | no |

In each case, the *object learned* is a conditional expectation of a
closed-form per-sample quantity. The *training loss* is least-squares
regression onto that quantity, justified by the regression lemma. The
*sampler* is a numerical
integrator from :numref:`sec_mdl-odes-solvers` or :numref:`sec_mdl-sdes`
applied to dynamics in which, after fixing the path, the learned field is the
unknown model component, and
its cost depends on solver order and tolerances, field regularity and
conditioning, and path geometry. Curvature and optimal-transport energy are
related considerations, but neither alone fixes the number of evaluations.

The table's two halves also admit a single umbrella (score rows are sampled by
reversing a stochastic process, velocity rows by integrating a prescribed
path): the **stochastic interpolants** of
:citet:`Albergo.Boffi.VandenEijnden.2023` write
$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \mathbf{x}_1 + \gamma_t \mathbf{w}$
and recover every row by a choice of schedule. With a Gaussian base $p_0$,
the paper writes the diffusion rows, like the rectified-flow row, as
*one-sided* interpolants with $\gamma_t \equiv 0$; it uses the latent term
$\gamma_t \mathbf{w}$ for its *two-sided* interpolants between arbitrary
$p_0$ and $p_1$. Exercise :exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants` walks the construction.

## Summary

* Flow matching prescribes the path and regresses the velocity;
  rectified flow's straight-line path makes the target the constant
  $\mathbf{x}_1 - \mathbf{x}_0$. By Benamou–Brenier, $W_2^2$ is the least
  kinetic energy of any bridging flow. Constant-speed straight paths attain
  that energy only when their endpoint coupling is optimal. Reflow and
  minibatch-OT couplings aim to reduce curvature or coupling cost; neither is
  an automatic certificate of exact optimal transport.
* Sampling is numerically solving the learned dynamics: ODEs offer deterministic
  samples and likelihood evaluation; SDEs inject sampling noise and may aid
  exploration. Step count and robustness are method- and problem-dependent;
  solver order, field regularity, path geometry, and tolerances jointly set the
  step budget.

Related dynamical descriptions also apply beyond generative models.
:citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` model self-attention as an
interacting particle system on normalized token representations and analyze
the resulting continuous-time clustering behavior using vector fields and
flows.

## Exercises

1. **Crossing segments and the floor.** The constant $C$ of the CFM
   theorem of :numref:`sec_mdl-flow-matching` is the conditional variance
   $\mathbb{E} \| Y - \mathbf{u}_t(\mathbf{x}) \|^2$ of the target
   $Y = \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$.
    1. Show that $C = 0$ if and only if, for almost every $t$, the conditional
       velocity is almost surely a function of $(\mathbf{x}_t, t)$. What does
       this condition require of the segments of the linear path
       :eqref:`eq_mdl-rf-path`?
    1. For the independent coupling of $\mathcal{N}(\mathbf{0}, I)$ with the
       standardized data `moons`, compute the conditional variance of
       $\mathbf{x}_1 - \mathbf{x}_0$ given $\mathbf{x}_t$, summed over
       coordinates, at $t = 0$ and at $t = 1$.
    1. The training loss in
       `#mdl-score-matching-diffusion-flow-one-reflow-round-measured` ends at a
       few thousandths. What does this value imply about crossings of the
       segments from $\mathbf{z}$ to $\hat{\mathbf{x}}_1(\mathbf{z})$, even
       though the network may carry approximation error?

1. **Stochastic interpolants.** ● The framework of
   :citet:`Albergo.Boffi.VandenEijnden.2023` writes
   $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \mathbf{x}_1 + \gamma_t \mathbf{w}$,
   where $(\mathbf{x}_0, \mathbf{x}_1)$ is drawn from a coupling of $p_0$
   and $p_1$, $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, I)$ is independent of
   both, and the smooth schedules satisfy $\alpha_0 = \beta_1 = 1$ and
   $\alpha_1 = \beta_0 = \gamma_0 = \gamma_1 = 0$.
    1. Take $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1, \mathbf{w})$
       in :eqref:`eq_mdl-marginal-velocity`. Show that the marginal velocity is
       $\mathbb{E}[\dot{\alpha}_t \mathbf{x}_0 + \dot{\beta}_t \mathbf{x}_1 + \dot{\gamma}_t \mathbf{w} \mid \mathbf{x}_t = \mathbf{x}]$,
       and write the corresponding CFM loss :eqref:`eq_mdl-cfm-loss`.
    1. Give schedules that recover rectified flow :eqref:`eq_mdl-rf-path` and
       the variance-preserving path of
       `#mdl-score-matching-diffusion-flow-score-noise-and-velocity-are-one-function`,
       whose data and noise coefficients are $\sin \tfrac{\pi t}{2}$ and
       $\cos \tfrac{\pi t}{2}$, both with independent endpoints and
       $p_0 = \mathcal{N}(\mathbf{0}, I)$.
    1. For $p_0 = \mathcal{N}(\mathbf{0}, I)$, independent endpoints, and
       $\gamma_t > 0$, show that the interpolant has the same marginal
       velocity as the interpolant with $\gamma \equiv 0$ and
       $\sqrt{\alpha_t^2 + \gamma_t^2}$ in place of $\alpha_t$.
    1. For $\gamma_t > 0$, show that
       $\nabla \log p_t(\mathbf{x}) = -\mathbb{E}[\mathbf{w} \mid \mathbf{x}_t = \mathbf{x}] / \gamma_t$.
       What does interior noise provide when $p_0$ is an arbitrary data
       distribution rather than a Gaussian?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §29.6]{.kicker}

**Flow Matching and Optimal Transport**
:::
:::

::: {.slide title="Two clocks"}
[Conventions]{.kicker}

Diffusion runs data→noise and samples backward; flow matching runs
noise→data and samples forward. To compare, substitute $t\to 1-t$:

@fig:mdl-dyn-time-conventions
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Flow matching]{.dtitle}

[prescribe the path, regress the velocity, translate the targets]{.dsub}
:::
:::

::: {.slide title="Probability paths and velocities"}
[Flow matching]{.kicker}

Prescribe a path $(p_t)$ from noise to data; its velocity obeys the continuity
equation. The intractable marginal velocity is again a posterior mean:

$$\mathbf u_t(\mathbf x) = \mathbb E\bigl[\mathbf u_t(\mathbf x\mid\mathbf z)\mid\mathbf x_t=\mathbf x\bigr].$$

The same conditional-expectation argument used for score matching applies.
:::

::: {.slide title="The conditional flow-matching theorem"}
[Flow matching]{.kicker}

::: {.d2l-note .rule}
Under the theorem's integrability assumptions, the population CFM loss
(closed-form per-pair velocity) and the population FM loss have the **same
gradients**.
:::

*Proof.* Apply the regression lemma with target
$\mathbf u_t(\mathbf x\mid\mathbf z)$; its conditional mean is the marginal
velocity. $\blacksquare$ Identical structure to Vincent's theorem.
:::

::: {.slide title="Relations among score, noise, and velocity"}
[Parameterization relations]{.kicker}

On a Gaussian path, at times where $\alpha_t,\sigma_t>0$,
$\mathbf x_t = \alpha_t\,\mathbf x_1 + \sigma_t\,\boldsymbol\epsilon$, the
marginal velocity and the marginal score determine each other:

$$\mathbf u_t(\mathbf x) = \frac{\dot\alpha_t}{\alpha_t}\,\mathbf x
- \Bigl(\sigma_t\dot\sigma_t - \sigma_t^2\,\frac{\dot\alpha_t}{\alpha_t}\Bigr)
\nabla\log p_t(\mathbf x)$$

Both are affine in the one posterior mean
$\hat{\mathbf x}_1 = \mathbb E[\mathbf x_1\mid\mathbf x_t]$ (Tweedie again):

@!mdl-score-matching-diffusion-flow-score-noise-and-velocity-are-one-function

One route uses the posterior mean directly; the other uses the score--velocity
identity.
:::

::: {.slide title="Common prediction targets"}
[Parameterizations]{.kicker}

The common prediction targets are $t$-dependent affine transformations of the
score $\mathbf s = \nabla\log p_t$:

| network predicts | in terms of $\mathbf s$ | scaling |
|:--|:--|:--|
| noise $\hat{\boldsymbol\epsilon}$ | $-\sigma_t\,\mathbf s$ | sampled target is unit-scale; its conditional mean may shrink near data |
| clean $\hat{\mathbf x}_1$ | $(\mathbf x + \sigma_t^2\,\mathbf s)/\alpha_t$ | division by small $\alpha_t$ can amplify error near noise |
| $v$-prediction $\alpha_t\boldsymbol\epsilon - \sigma_t\mathbf x_1$ | affine in $\mathbf s$ | sampled components remain comparable under common normalized schedules |

::: {.d2l-note}
The log-SNR coordinate $\rho_t=\log(\alpha_t^2/\sigma_t^2)$ compares schedules.
After state rescaling, matching $\rho$ ranges can describe the same noised
marginals up to time reparameterization; velocity scaling and numerical cost
still change with the clock.
:::
:::

::: {.slide title="Rectified flow: straight paths"}
[Flow matching]{.kicker}

The simplest path is a straight line,
$\mathbf x_t=(1-t)\mathbf x_0+t\mathbf x_1$, with constant target
$\mathbf x_1-\mathbf x_0$. Conditional paths are straight; the marginal flow
curves where conditional paths intersect:

@fig:mdl-dyn-fm-paths
:::

::: {.slide title="Euler step count resolves the learned two-moons geometry"}
[Flow matching]{.kicker}

A small MLP trained by the rectified-flow loss is integrated with Euler's
method. The generated crescents become more accurate as the step count grows:

@score-matching-diffusion-flow-cfm-panels
:::

::: {.slide title="One reflow round reduces one-step error in this run"}
[Reflow]{.kicker}

Integrate the trained ODE once, keep the model-generated couplings
$(\mathbf z, \hat{\mathbf x}_1(\mathbf z))$, and retrain the same architecture
on those pairs:

@!mdl-score-matching-diffusion-flow-one-reflow-round-measured

In this two-moons run, one Euler step scores $0.016$, close to the original
model's 32-step score of $0.014$ and better than its one-step $0.676$. The
smaller loss is consistent with a much smaller posterior variance under the
new coupling; it is not a zero-variance guarantee for finite training.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Optimal transport and sampling]{.dtitle}

[straightness, solver order, the unifying table]{.dsub}
:::
:::

::: {.slide title="Straight paths and optimal transport"}
[Benamou–Brenier]{.kicker}

$$W_2^2(p_0,p_1) = \min_{(p_t,\mathbf v_t)}\int_0^1\!\!\int\|\mathbf v_t\|^2 p_t.$$

For distributions with finite second moments and admissible regular flows,
any bridging flow costs at least $W_2^2$ (Jensen); a minimizing displacement
interpolation moves particles in straight lines at constant speed.

::: {.d2l-note .rule}
Benamou--Brenier identifies kinetic energy exactly. Curvature can increase
low-order truncation error, but solver cost also depends on derivatives,
conditioning, tolerances, and the method. Reflow and OT couplings aim to reduce
these costs; neither certifies them for a finite learned field.
:::
:::

::: {.slide title="Step count and solver order control distinct errors"}
[Sampling]{.kicker}

For the learned two-moons field, the sample metric decreases with Euler step
count and then plateaus; the plateau does not by itself identify model error
separately from finite-sample variability:

@score-matching-diffusion-flow-steps-quality

. . .

For the analytic one-dimensional field below, Heun at $20$ steps ($40$ field
evaluations) beats Euler at $40$ steps relative to the fine numerical
reference, as the observed second-order convergence predicts:

@!score-matching-diffusion-flow-euler-vs-heun
:::

::: {.slide title="Conditional regression supplies several learned dynamics"}
[Sampling]{.kicker}

These methods share three components: **a probability path, a closed-form
conditional regression target, and a numerical integrator**.

::: {.d2l-note .rule}
DDPM, score-SDE, PF-ODE, DDIM, and flow matching can be compared by path,
target parameterization, loss weighting, and integrator. Stochastic
interpolants provide one formalism for many, but not every implementation
detail, in this family.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Flow matching prescribes the path; CFM = FM by the same regression lemma.
- Score, noise, $\hat{\mathbf x}_1$, velocity: one posterior mean in different
  parameterizations, on the log-SNR clock.
- Benamou–Brenier identifies the least-energy flow; on the two-moons run, one
  reflow round made one Euler step approach the original model's 32-step metric.
- Both DSM and CFM replace an intractable marginal field by regression on a
  tractable conditional target whose conditional mean is that field.
:::
