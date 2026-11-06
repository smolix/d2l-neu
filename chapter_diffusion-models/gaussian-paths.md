# Gaussian Paths: Diffusion and Flow Matching
:label:`sec_diffusion-gaussian-paths`

The previous section ended with a dilemma that we measured on the three-mode mixture. A denoiser trained at a small noise level is accurate near the data, but Langevin chains driven by its score cannot cross the empty valleys between the modes, so they keep whatever mode weights they started with. At a large noise level the chains cross freely and find the right weights, but they sample a blurred copy of the data. Neither level alone produces the mixture. Suppose instead that one network denoised at every noise level, and that we walked from pure noise down to the data through all of them. How is such a walk defined, which regression trains the network, and how do we take the steps?

:citet:`sohl2015deep` took the idea from non-equilibrium thermodynamics: destroy the structure of the data with noise in many small steps, and learn to reverse each step. :citet:`ho2020denoising` turned every reversal into a regression that predicts the added noise, and obtained images as good as those of adversarial training. :citet:`song2019generative` reached the same model from a sequence of scores at decreasing noise levels, and :citet:`Lipman.Chen.BenHamu.ea.2022`, :citet:`Liu.Gong.Liu.2022` and :citet:`Albergo.Boffi.VandenEijnden.2023` from the velocity of a path between noise and data. Underneath all of them is one object: a Gaussian path from the data at $t = 0$ to pure noise at $t = 1$, the posterior mean of the clean point at every point of the path, learned by one regression, and a sampler that integrates the differential equation that this posterior mean defines.

This is how current image, video and audio generators are trained. "DDPM" and "flow matching" name two choices of path and target, not two families of models, and a reader who sees this once reads the literature of both without translating. We define the path and show on the mixture why it needs many noise levels, derive the regression and its equivalent targets, sample with and without noise, and check the learned models against the exact ones.

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
```

## One Path from Data to Noise

The previous section added noise at one level $\sigma$, as $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol\epsilon$. A walk through all levels needs the levels in a sequence, indexed by a time. We define that time, read the chain of Ho et al. as the same object in discrete steps, and check on the mixture that a walk through the levels recovers what one level could not.

### Signal, Noise and the Log-SNR Clock

We let the data shrink as the noise grows. A *Gaussian path* is a pair of functions $\alpha_t$ and $\sigma_t$ on $t \in [0, 1]$ that defines the noisy point

$$
\mathbf{x}_t = \alpha_t\, \mathbf{x}_0 + \sigma_t\, \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$
:eqlabel:`eq_diffusion-gaussian-path`

with $(\alpha_0, \sigma_0) = (1, 0)$ and $(\alpha_1, \sigma_1) = (0, 1)$, or nearly so. The signal coefficient $\alpha_t$ falls from one to zero while the noise coefficient $\sigma_t$ rises from zero to one, so $\mathbf{x}_t$ is a data point at $t = 0$ and pure noise at $t = 1$. Two paths cover most of the literature. The *variance-preserving* (VP) path keeps $\alpha_t^2 + \sigma_t^2 = 1$, so that data of unit variance keep unit variance at every time. We write $\bar\alpha_t := \alpha_t^2$ and use the continuous form of the schedule of :citet:`ho2020denoising`, $\log \bar\alpha_t = -\big(\beta_{\min} t + (\beta_{\max} - \beta_{\min})\, t^2/2\big)$ with $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$. It is the Ornstein--Uhlenbeck process of :numref:`sec_mdl-ornstein-uhlenbeck`. The *linear* path draws a straight line from the data to the noise,

$$
\alpha_t = 1 - t, \qquad \sigma_t = t.
$$
:eqlabel:`eq_diffusion-linear-path`

Two paths can hold the same noisy point at different times. What a denoiser can infer from $\mathbf{x}_t$ depends only on the ratio of signal to noise, which we measure on a log scale as the *log-SNR* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$. It falls monotonically from $+\infty$ at the data to $-\infty$ at the noise, so it serves as a clock that all paths share: at times of equal $\lambda$, two paths hold the same noisy point up to a scale factor (:exref:`ex_gaussian-paths-same-ratio-same-flow`). At $t = 1/2$ the VP path has $\bar\alpha = 0.079$ and $\lambda \approx -2.5$, while the linear path has $\lambda = 0$: halfway along in time, the VP path is already mostly noise. At $t = 1$ it reaches $\bar\alpha_1 = e^{-10.05} \approx 4 \cdot 10^{-5}$, close enough to pure noise. :numref:`fig_diffusion-paths` draws both paths.

::: {.callout-important title="One clock"}
Data sit at $t = 0$ and noise at $t = 1$, for scores and velocities alike, and every sampler runs from $t = 1$ down to $t = 0$. The whole book uses this clock, the mathematics appendix included (:numref:`sec_mdl-flow-matching-transport`). The flow-matching papers and their code run time the other way, from noise at $0$ to data at $1$ (:numref:`fig_mdl-dyn-time-conventions`). To compare a formula with theirs, substitute $1 - t$ for their $t$, read their noise $\mathbf{x}_0$ as our $\boldsymbol\epsilon$ and their data $\mathbf{x}_1$ as our $\mathbf{x}_0$, and flip the sign of every velocity.
:::

![Two Gaussian paths on one clock. (a) The signal coefficient $\alpha_t$ (solid) and the noise coefficient $\sigma_t$ (dashed) of the variance-preserving path, with $\beta$ from $0.1$ to $20$, and of the linear path. (b) The log-SNR $\lambda_t = \log(\alpha_t^2/\sigma_t^2)$ of both paths, clipped to $[-10, 10]$. Both fall from the data at $t = 0$ to the noise at $t = 1$; the variance-preserving path reaches the noisy regime much earlier.](../img/mdl-diffusion-paths.svg)
:label:`fig_diffusion-paths`

`GaussianPath` is the schedule as an object: two functions $\alpha_t$ and $\sigma_t$ on the clock $t \in [0, 1]$, the log-SNR $\lambda_t$, and a draw of $\mathbf{x}_t$; `vp`, `linear` and `cosine` build the three paths of the text.

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock}
%%tab pytorch
class GaussianPath:  #@save
    """x_t = alpha(t) x0 + sigma(t) eps, data at t = 0 and noise at t = 1."""
    def __init__(self, alpha, sigma):
        self.alpha, self.sigma = alpha, sigma

    @staticmethod
    def vp(beta_min=0.1, beta_max=20.0):
        """Variance preserving, Ho et al.'s linear beta in continuous time."""
        log_abar = lambda t: -(beta_min * t + (beta_max - beta_min) * t**2 / 2)
        return GaussianPath(lambda t: torch.exp(log_abar(t) / 2),
                            lambda t: torch.sqrt(-torch.expm1(log_abar(t))))

    @staticmethod
    def linear():
        """The straight line of flow matching and rectified flow."""
        return GaussianPath(lambda t: 1 - t, lambda t: t)

    @staticmethod
    def cosine(s=0.008):
        """Variance preserving, Nichol and Dhariwal's cosine schedule."""
        c = lambda t: torch.cos(math.pi / 2 * (t + s) / (1 + s))
        g = lambda t: c(t) / c(torch.zeros_like(t))   # exactly 1 at t = 0
        # and exactly 0 at t = 1 (cos(pi/2) is not), keeping the slope
        f = lambda t: torch.where(t < 1, g(t), g(t) - g(t).detach())
        return GaussianPath(f, lambda t: torch.sqrt(1 - f(t) ** 2))

    def log_snr(self, t):
        """lambda_t = log(alpha_t^2 / sigma_t^2), falling from +inf at 0."""
        return 2 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def sample(self, x0, t, eps):
        """x_t for per-example times t of shape (n,)."""
        t = t.reshape(-1, *[1] * (x0.dim() - 1))
        return self.alpha(t) * x0 + self.sigma(t) * eps
```

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock}
%%tab jax
class GaussianPath:  #@save
    """x_t = alpha(t) x0 + sigma(t) eps, data at t = 0 and noise at t = 1."""
    def __init__(self, alpha, sigma):
        self.alpha, self.sigma = alpha, sigma

    @staticmethod
    def vp(beta_min=0.1, beta_max=20.0):
        """Variance preserving, Ho et al.'s linear beta in continuous time."""
        log_abar = lambda t: -(beta_min * t + (beta_max - beta_min) * t**2 / 2)
        return GaussianPath(lambda t: jnp.exp(log_abar(t) / 2),
                            lambda t: jnp.sqrt(-jnp.expm1(log_abar(t))))

    @staticmethod
    def linear():
        """The straight line of flow matching and rectified flow."""
        return GaussianPath(lambda t: 1 - t, lambda t: t)

    @staticmethod
    def cosine(s=0.008):
        """Variance preserving, Nichol and Dhariwal's cosine schedule."""
        c = lambda t: jnp.cos(math.pi / 2 * (t + s) / (1 + s))
        g = lambda t: c(t) / c(jnp.zeros_like(t))     # exactly 1 at t = 0
        # and exactly 0 at t = 1 (cos(pi/2) is not), keeping the slope
        f = lambda t: jnp.where(t < 1, g(t),
                                g(t) - jax.lax.stop_gradient(g(t)))
        return GaussianPath(f, lambda t: jnp.sqrt(1 - f(t) ** 2))

    def log_snr(self, t):
        """lambda_t = log(alpha_t^2 / sigma_t^2), falling from +inf at 0."""
        return 2 * (jnp.log(self.alpha(t)) - jnp.log(self.sigma(t)))

    def sample(self, x0, t, eps):
        """x_t for per-example times t of shape (n,)."""
        t = t.reshape(-1, *[1] * (x0.ndim - 1))
        return self.alpha(t) * x0 + self.sigma(t) * eps
```

How quickly does each path trade signal for noise? We print the two coefficients and the log-SNR at three times.

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock-1}
%%tab pytorch
vp, linear = GaussianPath.vp(), GaussianPath.linear()
t = torch.tensor([0.1, 0.5, 0.9])
for name, path in (('vp', vp), ('linear', linear)):
    print(f'{name:>6}:', *(f'{key} {[round(v, 2) for v in f(t).tolist()]}'
          for key, f in (('alpha', path.alpha), ('sigma', path.sigma),
                         ('log-SNR', path.log_snr))))
```

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock-1}
%%tab jax
vp, linear = GaussianPath.vp(), GaussianPath.linear()
t = jnp.array([0.1, 0.5, 0.9])
for name, path in (('vp', vp), ('linear', linear)):
    print(f'{name:>6}:', *(f'{key} {[round(float(v), 2) for v in f(t)]}'
          for key, f in (('alpha', path.alpha), ('sigma', path.sigma),
                         ('log-SNR', path.log_snr))))
```

At $t = 0.9$ the VP path keeps less than two percent of the signal and the linear path ten percent. The two paths visit the same noise levels on different schedules.

### The Chain View

:citet:`ho2020denoising` did not start from a continuous path. Their forward process is a Markov chain of $T$ small steps, each of which shrinks the current point and adds a little noise,

$$
\mathbf{x}_k = \sqrt{1 - \beta_k}\, \mathbf{x}_{k-1} + \sqrt{\beta_k}\, \boldsymbol{\epsilon}_k,
\qquad k = 1, \ldots, T,
$$
:eqlabel:`eq_diffusion-forward`

with independent standard Gaussians $\boldsymbol\epsilon_k$ and a variance schedule $\beta_k \in (0, 1)$. If $\mathbf{x}_{k-1}$ has unit variance, so does $\mathbf{x}_k$: each step preserves the variance. Two steps compose into one. Substituting the step for $\mathbf{x}_{k-1}$ gives $\mathbf{x}_k = \sqrt{(1-\beta_k)(1-\beta_{k-1})}\, \mathbf{x}_{k-2} + \sqrt{(1-\beta_k)\beta_{k-1}}\, \boldsymbol\epsilon_{k-1} + \sqrt{\beta_k}\, \boldsymbol\epsilon_k$, and the two noise terms are independent, so their variances add to $(1 - \beta_k)\beta_{k-1} + \beta_k = 1 - (1 - \beta_k)(1 - \beta_{k-1})$. Repeating the argument down to $\mathbf{x}_0$ gives the marginal of every step in one draw,

$$
\mathbf{x}_k = \sqrt{\bar\alpha_k}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_k}\, \boldsymbol{\epsilon},
\qquad \bar\alpha_k = \prod_{j=1}^{k} (1 - \beta_j),
$$
:eqlabel:`eq_diffusion-marginal`

which :numref:`sec_mdl-ddpm-discretized-sde` proves by induction. The chain is therefore the VP path read on the grid $t_k = k/T$: choosing $\beta_k = 1 - \bar\alpha_{t_k} / \bar\alpha_{t_{k-1}}$ makes the two agree at every grid point, and the DDPM step is the exact transition of the VP path from $t_{k-1}$ to $t_k$. Training never simulates the chain, because any $\mathbf{x}_k$ is one Gaussian draw away from $\mathbf{x}_0$. :numref:`fig_diffusion-forward-reverse` shows the forward chain above and, below it, the reverse chain that generation will run.

![The graphical model of a diffusion model. The forward process (top) is a fixed Markov chain of Gaussian transitions that carries the data $\mathbf{x}_0$ to noise $\mathbf{x}_T$. The reverse process (bottom) is a learned Markov chain that runs the other way; its transitions are Gaussians whose means a network predicts.](../img/mdl-diffusion-forward-reverse.svg)
:label:`fig_diffusion-forward-reverse`

`grid` evaluates the path on the chain's grid $t_k = k/T$, from noise at $t = 1$ down to data at $t = 0$; every sampler steps along it, and a start time below 1 begins a sampler partway along the path.

```{.python .input #gaussian-paths-the-chain-view}
%%tab pytorch
@d2l.add_to_class(GaussianPath)  #@save
def grid(self, steps, t_start=1.0, t_end=0.0):
    """Lists of alpha and sigma at steps + 1 even times, t_start to t_end."""
    t = torch.linspace(t_start, t_end, steps + 1, dtype=torch.float64)
    return self.alpha(t).tolist(), self.sigma(t).tolist()
```

```{.python .input #gaussian-paths-the-chain-view}
%%tab jax
@d2l.add_to_class(GaussianPath)  #@save
def grid(self, steps, t_start=1.0, t_end=0.0):
    """Arrays of alpha and sigma at steps + 1 even times, t_start to t_end."""
    with jax.ensure_compile_time_eval():         # constants, even under jit
        t = jnp.linspace(t_start, t_end, steps + 1)
        return self.alpha(t), self.sigma(t)
```

Read on a grid of $T = 1000$ steps, which schedule does our VP path define? The step variance is $\beta_k = 1 - \bar\alpha_{t_k}/\bar\alpha_{t_{k-1}}$; the grid lists the times from the noise end, so the first entries belong to the last step of the chain.

```{.python .input #gaussian-paths-the-chain-view-1}
%%tab pytorch, jax
a, _ = vp.grid(1000)
print(f'beta_1 = {1 - (a[-2] / a[-1]) ** 2:.1e}, '
      f'beta_T = {1 - (a[0] / a[1]) ** 2:.4f}, alpha_bar_T = {a[0] ** 2:.1e}')
```

The steps grow from about $10^{-4}$ at the data to $0.02$ at the noise, the linear schedule of :citet:`ho2020denoising`, and after a thousand steps only a factor $\sqrt{\bar\alpha_T} \approx 0.007$ of the data survives.

### Annealing through Noise Levels

Does a walk through many noise levels do what one level could not? We can test this before training any network, because the scores of the smoothed mixture are known exactly. We run the Langevin sampler of the previous section, $\mathbf{x} \leftarrow \mathbf{x} + \tfrac{h}{2}\, \mathbf{s}(\mathbf{x}) + \sqrt{h}\, \boldsymbol\xi$, through a decreasing sequence of noise levels, which the score-based literature calls a *ladder*: ten levels $\tilde\sigma_i$ spaced evenly in log scale from $3$ down to $0.05$, with $100$ steps of size $h_i = 0.1\, \tilde\sigma_i^2$ at each level, each level starting where the previous one stopped, and the exact score of the smoothed density $p_{\tilde\sigma_i}$ at each. For the control we spend the same $1{,}000$ steps at the finest level alone, with the step size $h = 0.05$ that the previous section used on the clean mixture. Both start from the same $2{,}000$ points drawn from $\mathcal{N}(\mathbf{0}, 9\mathbf{I})$, about a third of which lie nearest to each mode.

```{.python .input #gaussian-paths-annealing-through-noise-levels}
%%tab pytorch
torch.manual_seed(0)
device = d2l.try_gpu()
mix = d2l.GaussianMixture.three_modes(device)
x_init = 3 * torch.randn(2000, 2, device=device)
x = x_init
for s in torch.logspace(math.log10(3), math.log10(0.05), 10).tolist():
    x = d2l.langevin(lambda z: mix.score(z, 1.0, s), x, 0.1 * s ** 2, 100)
control = d2l.langevin(lambda z: mix.score(z, 1.0, 0.05), x_init, 0.05, 1000)
print('finest level only:', mix.summary(control))
print('ten levels:       ', mix.summary(x))
d2l.scatter_panels([control, x], ['finest level only', 'ten levels'],
                   data=mix.sample(2000));
```

```{.python .input #gaussian-paths-annealing-through-noise-levels}
%%tab jax
key = jax.random.key(0)
mix = d2l.GaussianMixture.three_modes()
key, k_init, k_ctrl, k_data = jax.random.split(key, 4)
x_init = 3 * jax.random.normal(k_init, (2000, 2))
x = x_init
for s in jnp.logspace(math.log10(3), math.log10(0.05), 10).tolist():
    key, k = jax.random.split(key)
    x = d2l.langevin(lambda z: mix.score(z, 1.0, s), x, 0.1 * s ** 2, 100, k)
control = d2l.langevin(lambda z: mix.score(z, 1.0, 0.05), x_init, 0.05, 1000,
                       k_ctrl)
print('finest level only:', mix.summary(control))
print('ten levels:       ', mix.summary(x))
d2l.scatter_panels([control, x], ['finest level only', 'ten levels'],
                   data=mix.sample(k_data, 2000));
```

The ladder lands within about $0.03$ of the weights $0.5/0.3/0.2$, with about two percent of the points straying between the modes. The control has the right spread around each mode but keeps the weights of its starting points, about a third each. The proportions are decided at the coarse levels, where the smoothed density has no valleys and the chains move freely between the modes; the fine levels only sharpen each mode. A diffusion model makes the same division of labor along its path.

:citet:`song2019generative` built the first generator of this kind: one network that takes the noise level as an input, trained by denoising score matching at every level of a ladder, with each level's loss weighted by $\tilde\sigma_i^2$, and sampled by this *annealed Langevin dynamics* (:numref:`sec_mdl-score-based-generative-modeling` writes out the loss, and :exref:`ex_mdl-score-matching-diffusion-flow-annealing` analyzes the annealing). The ladder had to be designed by hand: its levels, the step size at each, and the number of steps (:exref:`ex_gaussian-paths-ladder-spacing`). The path replaces the ladder by a continuum of levels, and the rest of this section replaces the hand-tuned Langevin steps by a sampler that the path itself determines.

## One Regression

One network is to serve every point of the path. We first ask what it should learn at a noisy point $\mathbf{x}_t$, then show that a single regression on per-example targets learns it, and that the targets used in the literature are four forms of the same answer.

### Conditional Targets and Marginal Fields

At a noisy point the most useful thing to know is the posterior mean of the clean point, $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$. At a single noise level the previous section obtained the score from it by Tweedie's formula, $\nabla \log p_\sigma(\tilde{\mathbf{x}}) = (\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}] - \tilde{\mathbf{x}})/\sigma^2$. Along a path it also tells us which way to move a noisy point toward the data. The posterior mean depends on the whole data distribution and is unknown, but its conditional version is trivial: given $\mathbf{x}_0$, the best guess of $\mathbf{x}_0$ is $\mathbf{x}_0$ itself.

The same holds for motion along the path. A single noisy point moves along $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ with the *conditional velocity* $\dot{\mathbf{x}}_t = \dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$, which on the linear path is $\boldsymbol\epsilon - \mathbf{x}_0$. The whole cloud of noisy points moves at $\mathbf{x}$ with the average velocity of the points that pass there,

$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}\big[\dot\alpha_t\, \mathbf{x}_0 + \dot\sigma_t\, \boldsymbol\epsilon \,\big|\, \mathbf{x}_t = \mathbf{x}\big],
$$
:eqlabel:`eq_diffusion-marginal-velocity`

the *marginal velocity*. Where only one clean point can have produced $\mathbf{x}$, it is that point's velocity; where several can, it is their average, weighted by how likely each is. This field moves the density $p_t$ of $\mathbf{x}_t$ along the path, so that integrating it backward from noise produces data, as we show in the next part.

How can we learn a field that is defined as a conditional expectation? By regressing on the quantity inside it. The best least-squares predictor of a random target is its conditional mean, which is how denoising score matching recovered the score in the previous section. Applied to the velocity, this gives the *conditional flow matching* loss, which differs from the regression onto the unknown marginal field only by a constant $C$ that does not depend on the network:

$$
\mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol\epsilon}\, \big\| \mathbf{v}_{\boldsymbol\theta}(\mathbf{x}_t, t) - (\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon) \big\|^2
= \mathbb{E}_{t,\, \mathbf{x}_t}\, \big\| \mathbf{v}_{\boldsymbol\theta}(\mathbf{x}_t, t) - \mathbf{u}_t(\mathbf{x}_t) \big\|^2 + C.
$$
:eqlabel:`eq_diffusion-cfm-loss`

The two sides have the same minimizer, so a network trained on the per-example velocity learns the marginal one (:numref:`sec_mdl-flow-matching` proves this theorem and the transport property). :numref:`fig_diffusion-blueprint` shows the conditional and the marginal objects side by side.

![The conditional and the marginal views of the linear path on the mixture, all computed exactly. (a) The conditional path of one clean point $\mathbf{x}_0$ from the heaviest mode: circles of radius $2\sigma_t$ around $\alpha_t\mathbf{x}_0$ at $t = 0, 0.25, 0.5, 0.75, 1$. (b) The marginal path: density contours of $p_t$ at the same times. (c) The conditional field at $t = 0.5$, drawn as $-(\mathbf{x} - \mathbf{x}_0)/t$, the direction in which a sampler moves toward that one point. (d) The marginal field $-\mathbf{u}_{0.5}(\mathbf{x})$, the posterior average of the conditional fields over all clean points.](../img/mdl-diffusion-blueprint.svg)
:label:`fig_diffusion-blueprint`

`target` gives the per-example regression target of a kind on any path. For `'velocity'` it is the conditional velocity $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$, with the two derivatives from forward-mode differentiation (`rates`); the other kinds are the subject of the next subsection, and `x0_eps` turns a prediction of any kind back into the pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$.

```{.python .input #gaussian-paths-four-targets-one-oracle}
%%tab pytorch
@d2l.add_to_class(GaussianPath)  #@save
def rates(self, t):
    """The time derivatives (alpha'(t), sigma'(t)), by forward-mode AD."""
    ones = torch.ones_like(t)
    return (torch.func.jvp(self.alpha, (t,), (ones,))[1],
            torch.func.jvp(self.sigma, (t,), (ones,))[1])

@d2l.add_to_class(GaussianPath)  #@save
def target(self, kind, x0, eps, t):
    """The regression target of a kind: eps, x0, v, score or velocity."""
    t = t.reshape(-1, *[1] * (x0.dim() - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return eps
    if kind == 'x0': return x0
    if kind == 'v': return a * eps - s * x0
    if kind == 'score': return -eps / s
    if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
    da, ds = self.rates(t)                      # velocity = d x_t / dt
    return da * x0 + ds * eps

@d2l.add_to_class(GaussianPath)  #@save
def x0_eps(self, kind, out, x, t):
    """The (x0, eps) that a prediction `out` of a kind implies at (x, t)."""
    t = t.reshape(-1, *[1] * (x.dim() - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return (x - s * out) / a, out
    if kind == 'x0': return out, (x - a * out) / s
    if kind == 'v': return ((a * x - s * out) / (a**2 + s**2),
                            (s * x + a * out) / (a**2 + s**2))
    if kind == 'score': return (x + s ** 2 * out) / a, -s * out
    if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
    da, ds = self.rates(t)
    det = a * ds - s * da
    return (ds * x - s * out) / det, (a * out - da * x) / det
```

```{.python .input #gaussian-paths-four-targets-one-oracle}
%%tab jax
@d2l.add_to_class(GaussianPath)  #@save
def rates(self, t):
    """The time derivatives (alpha'(t), sigma'(t)), by forward-mode AD."""
    ones = jnp.ones_like(t)
    return (jax.jvp(self.alpha, (t,), (ones,))[1],
            jax.jvp(self.sigma, (t,), (ones,))[1])

@d2l.add_to_class(GaussianPath)  #@save
def target(self, kind, x0, eps, t):
    """The regression target of a kind: eps, x0, v, score or velocity."""
    t = t.reshape(-1, *[1] * (x0.ndim - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return eps
    if kind == 'x0': return x0
    if kind == 'v': return a * eps - s * x0
    if kind == 'score': return -eps / s
    if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
    da, ds = self.rates(t)                      # velocity = d x_t / dt
    return da * x0 + ds * eps

@d2l.add_to_class(GaussianPath)  #@save
def x0_eps(self, kind, out, x, t):
    """The (x0, eps) that a prediction `out` of a kind implies at (x, t)."""
    t = t.reshape(-1, *[1] * (x.ndim - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return (x - s * out) / a, out
    if kind == 'x0': return out, (x - a * out) / s
    if kind == 'v': return ((a * x - s * out) / (a**2 + s**2),
                            (s * x + a * out) / (a**2 + s**2))
    if kind == 'score': return (x + s ** 2 * out) / a, -s * out
    if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
    da, ds = self.rates(t)
    det = a * ds - s * da
    return (ds * x - s * out) / det, (a * out - da * x) / det
```

The method fits in one cell. Each step draws $512$ fresh points from the mixture and a time and a noise for each, and regresses `TimeMLP` of the previous section onto the conditional velocity of the linear path. `train_regression` runs $40{,}000$ Adam steps with a cosine-decayed learning rate and returns, besides the network, a moving average of its weights with a half-life of $700$ steps.

```{.python .input #gaussian-paths-conditional-targets-and-marginal-fields}
%%tab pytorch
def regression_loss(path, kind):
    """Squared error of net(x_t, t) against the target of a kind, t ~ U(0, 1)."""
    def loss_fn(net, x0):
        t = torch.rand(len(x0), device=x0.device)
        eps = torch.randn_like(x0)
        error = net(path.sample(x0, t, eps), t) - path.target(kind, x0, eps, t)
        return (error ** 2).sum(1).mean()
    return loss_fn

torch.manual_seed(1)
num_steps = 40_000
net_v, ema_v, losses_v = d2l.train_regression(
    d2l.TimeMLP(256, 3).to(device), regression_loss(linear, 'velocity'),
    lambda: mix.sample(512), num_steps, cosine=True, ema_half_life=700)
d2l.plot(torch.arange(100) * num_steps // 100,
         losses_v.reshape(100, -1).mean(1).cpu(), 'step', 'loss')
```

```{.python .input #gaussian-paths-conditional-targets-and-marginal-fields}
%%tab jax
def regression_loss(path, kind):
    """Squared error of net(x_t, t) against the target of a kind, t ~ U(0, 1)."""
    def loss_fn(net, x0, key):
        k_t, k_eps = jax.random.split(key)
        t = jax.random.uniform(k_t, (len(x0),))
        eps = jax.random.normal(k_eps, x0.shape)
        error = net(path.sample(x0, t, eps), t) - path.target(kind, x0, eps, t)
        return (error ** 2).sum(1).mean()
    return loss_fn

num_steps = 40_000
net_v, ema_v, losses_v = d2l.train_regression(
    d2l.TimeMLP(256, 3, rngs=nnx.Rngs(1)), regression_loss(linear, 'velocity'),
    lambda k: mix.sample(k, 512), num_steps, jax.random.key(1), cosine=True,
    ema_half_life=700)
d2l.plot(jnp.arange(100) * num_steps // 100,
         losses_v.reshape(100, -1).mean(1), 'step', 'loss')
```

Each point of the curve averages $400$ steps. The loss falls within a few thousand steps and levels off near $5$, far above zero; we will see that no network can do much better.

### Four Targets, One Oracle

The network above predicts a velocity; DDPM's predicts the noise, and others predict the clean point. At a fixed $\mathbf{x}_t$ these are not different quantities. Taking the posterior mean of both sides of :eqref:`eq_diffusion-gaussian-path` gives $\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$, so the noise prediction $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ and $\hat{\mathbf{x}}_0$ determine each other. Tweedie's formula for the kernel $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ gives the score,

$$
\nabla \log p_t(\mathbf{x}_t) = \frac{\alpha_t \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma_t^2} = -\frac{\hat{\boldsymbol\epsilon}}{\sigma_t},
$$
:eqlabel:`eq_diffusion-score-eps`

so a noise predictor is a score model scaled by $-\sigma_t$. The marginal velocity :eqref:`eq_diffusion-marginal-velocity` is $\dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t \hat{\boldsymbol\epsilon}$, and eliminating $\hat{\mathbf{x}}_0$ between it and the score gives the score–velocity identity

$$
\mathbf{u}_t(\mathbf{x}) = \frac{\dot\alpha_t}{\alpha_t}\, \mathbf{x} - \Big(\sigma_t \dot\sigma_t - \sigma_t^2 \frac{\dot\alpha_t}{\alpha_t}\Big) \nabla \log p_t(\mathbf{x}),
$$
:eqlabel:`eq_diffusion-score-velocity`

a multiple of the position plus a multiple of the score, with both multiples fixed by the path. On the linear path it reads

$$
\mathbf{u}_t(\mathbf{x}) = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0 = -\frac{\mathbf{x} + t\, \nabla \log p_t(\mathbf{x})}{1 - t}.
$$
:eqlabel:`eq_diffusion-linear-dictionary`

A fifth target, $\mathbf{v} = \alpha_t \boldsymbol\epsilon - \sigma_t \mathbf{x}_0$, was introduced by :citet:`Salimans.Ho.2022` as *$v$-prediction*; on the cosine path $\alpha_t = \cos\frac{\pi t}{2}$, $\sigma_t = \sin\frac{\pi t}{2}$ it equals $2/\pi$ times the velocity. All of these are affine functions of one posterior mean, the *oracle* that a perfect network would learn, so a network trained on any of them defines all of them (:numref:`sec_mdl-score-velocity-dictionary` and :numref:`tab_mdl-score-noise-velocity` collect the conversions). The names in the literature are choices of path and target: DDPM regresses the noise on the VP path, flow matching and rectified flow regress the velocity on the linear path, and $v$-prediction regresses $\mathbf{v}$ on the cosine path.

DDPM's training loss, which :citet:`ho2020denoising` called the *simple loss*, is

$$
L_{\textrm{simple}}(\boldsymbol\theta) = \mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol\epsilon}\, \big\| \boldsymbol\epsilon - \boldsymbol\epsilon_{\boldsymbol\theta}(\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon,\, t) \big\|^2 .
$$
:eqlabel:`eq_diffusion-simple-loss`

The conditional score of the kernel is $-\boldsymbol\epsilon/\sigma_t$, so each term is $\sigma_t^2$ times the denoising score-matching loss at level $\sigma_t$ for the score model $-\boldsymbol\epsilon_{\boldsymbol\theta}/\sigma_t$: the simple loss is denoising score matching at every level, weighted by $\sigma_t^2$ (:numref:`sec_mdl-ddpm-discretized-sde`). Ho et al. arrived at it from a variational bound on the log-likelihood, and with one particular weighting over $t$ the regression is such a bound, which :numref:`sec_diffusion-design-space` makes precise.

In code, `target` and `x0_eps` above already implement these conversions for all five kinds. `make_pred` wraps a trained network as the pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ at any $(\alpha, \sigma)$: it queries the network at the time of its own path with the same signal-to-noise ratio, which `time_at` finds by bisection on the log-SNR clock. An $\boldsymbol\epsilon$-predictor has no $\hat{\mathbf{x}}_0 = (\mathbf{x} - \sigma\hat{\boldsymbol\epsilon})/\alpha$ at pure noise, $\alpha = 0$ (the division returns NaN), so samplers that start there on the linear and cosine paths need one of the other kinds; an $\boldsymbol\epsilon$-model samples on the VP path, whose $\alpha$ stays positive.

```{.python .input #gaussian-paths-four-targets-one-oracle-1}
%%tab pytorch
@d2l.add_to_class(GaussianPath)  #@save
def time_at(self, log_snr):
    """The time with a given log-SNR, by bisection (lambda_t falls in t)."""
    lo, hi = torch.zeros_like(log_snr), torch.ones_like(log_snr)
    for _ in range(50):
        mid = (lo + hi) / 2
        later = self.log_snr(mid) > log_snr      # the answer lies after mid
        lo, hi = torch.where(later, mid, lo), torch.where(later, hi, mid)
    return (lo + hi) / 2

def make_pred(net, path, kind):  #@save
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) of a kind on path."""
    def pred(x, alpha, sigma):
        log_snr = 2 * math.log(alpha / sigma) if alpha > 0 else -math.inf
        t = path.time_at(torch.tensor(log_snr, dtype=torch.float64))
        x = x * (path.sigma(t).item() / sigma)  # the same point on our path
        t = torch.full((len(x),), t.item(), device=x.device)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred
```

```{.python .input #gaussian-paths-four-targets-one-oracle-1}
%%tab jax
@d2l.add_to_class(GaussianPath)  #@save
def time_at(self, log_snr):
    """The time with a given log-SNR, by bisection (lambda_t falls in t)."""
    def halve(_, interval):
        lo, hi = interval
        mid = (lo + hi) / 2
        later = self.log_snr(mid) > log_snr      # the answer lies after mid
        return jnp.where(later, mid, lo), jnp.where(later, hi, mid)
    lo, hi = jax.lax.fori_loop(0, 50, halve, (jnp.zeros_like(log_snr),
                                              jnp.ones_like(log_snr)))
    return (lo + hi) / 2

def make_pred(net, path, kind):  #@save
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) of a kind on path."""
    def pred(x, alpha, sigma):
        log_snr = jnp.where(alpha > 0, 2 * jnp.log(alpha / sigma), -jnp.inf)
        t = path.time_at(jnp.asarray(log_snr, jnp.float32))
        x = x * (path.sigma(t) / sigma)         # the same point on our path
        t = jnp.full((len(x),), t)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred
```

The second model makes DDPM's choice, the noise target on the VP path, and changes nothing else.

```{.python .input #gaussian-paths-four-targets-one-oracle-2}
%%tab pytorch
torch.manual_seed(2)
net_eps, ema_eps, losses_eps = d2l.train_regression(
    d2l.TimeMLP(256, 3).to(device), regression_loss(vp, 'eps'),
    lambda: mix.sample(512), num_steps, cosine=True, ema_half_life=700)
```

```{.python .input #gaussian-paths-four-targets-one-oracle-2}
%%tab jax
net_eps, ema_eps, losses_eps = d2l.train_regression(
    d2l.TimeMLP(256, 3, rngs=nnx.Rngs(2)), regression_loss(vp, 'eps'),
    lambda k: mix.sample(k, 512), num_steps, jax.random.key(2), cosine=True,
    ema_half_life=700)
```

The two networks were trained on different paths and different targets. By the dictionary they should nevertheless hold the same posterior mean, which the last part of the section checks.

### The Loss Floor

The velocity model's loss levelled off near $5$, not zero, because a training target contains the noise of its own example, which no function of $\mathbf{x}_t$ can predict. The best predictor is the oracle, and its loss is the conditional variance of the target given $\mathbf{x}_t$, averaged over the path. For the mixture we can compute this floor exactly: `GaussianMixture.posterior_mean` gives $\hat{\mathbf{x}}_0$ at any $(\alpha, \sigma)$, the identity $\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$ gives $\hat{\boldsymbol\epsilon}$, and since every target is affine in $(\mathbf{x}_0, \boldsymbol\epsilon)$, its conditional mean is the same affine function of $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$. The function `oracle` is this exact predictor; any sampler can use it in place of a network.

```{.python .input #gaussian-paths-the-loss-floor}
%%tab pytorch
def oracle(x, alpha, sigma):
    """The exact posterior means (x0_hat, eps_hat) of the mixture at x."""
    x0_hat = mix.posterior_mean(x, alpha, sigma)
    a, s = (torch.as_tensor(v, device=x.device).reshape(-1, 1)
            for v in (alpha, sigma))
    return x0_hat, (x - a * x0_hat) / s

def loss_floor(path, kind, t=None, n=1_000_000):
    """E||target - E[target | x_t]||^2 at time t, or over t ~ U(0, 1)."""
    t = (torch.rand(n, device=device) if t is None
         else torch.full((n,), t, device=device))
    x0, eps = mix.sample(n), torch.randn(n, 2, device=device)
    x = path.sample(x0, t, eps)
    best = path.target(kind, *oracle(x, path.alpha(t), path.sigma(t)), t)
    return ((path.target(kind, x0, eps, t) - best) ** 2).sum(1).mean().item()

print('target, path      t=0.1  t=0.5  t=0.9  t~U(0,1)  trained')
for name, path, kind, losses in (('noise, vp', vp, 'eps', losses_eps),
        ('velocity, linear', linear, 'velocity', losses_v)):
    floors = [loss_floor(path, kind, t) for t in (0.1, 0.5, 0.9, None)]
    print(f'{name:16}' + ''.join(f'{f:7.3f}' for f in floors[:3])
          + f'{floors[3]:10.3f}{losses[-1000:].mean().item():9.3f}')
```

```{.python .input #gaussian-paths-the-loss-floor}
%%tab jax
def oracle(x, alpha, sigma):
    """The exact posterior means (x0_hat, eps_hat) of the mixture at x."""
    x0_hat = mix.posterior_mean(x, alpha, sigma)
    a, s = (jnp.reshape(jnp.asarray(v), (-1, 1)) for v in (alpha, sigma))
    return x0_hat, (x - a * x0_hat) / s

def loss_floor(path, kind, key, t=None, n=1_000_000):
    """E||target - E[target | x_t]||^2 at time t, or over t ~ U(0, 1)."""
    k_t, k_x0, k_eps = jax.random.split(key, 3)
    t = jax.random.uniform(k_t, (n,)) if t is None else jnp.full((n,), t)
    x0, eps = mix.sample(k_x0, n), jax.random.normal(k_eps, (n, 2))
    x = path.sample(x0, t, eps)
    best = path.target(kind, *oracle(x, path.alpha(t), path.sigma(t)), t)
    return float(((path.target(kind, x0, eps, t) - best) ** 2).sum(1).mean())

print('target, path      t=0.1  t=0.5  t=0.9  t~U(0,1)  trained')
for name, path, kind, losses in (('noise, vp', vp, 'eps', losses_eps),
        ('velocity, linear', linear, 'velocity', losses_v)):
    key, *keys = jax.random.split(key, 5)
    floors = [loss_floor(path, kind, k, t)
              for k, t in zip(keys, (0.1, 0.5, 0.9, None))]
    print(f'{name:16}' + ''.join(f'{f:7.3f}' for f in floors[:3])
          + f'{floors[3]:10.3f}{float(losses[-1000:].mean()):9.3f}')
```

Each training loss sits at its floor, and the two floors differ by an order of magnitude, about $0.5$ for the noise and about $5$ for the velocity, because the targets are scaled differently. The noise floor is at most $d = 2$, reached near the data, where the noise cannot be told apart from the spread of a mode; near $t = 1$ the point is almost all noise and $\boldsymbol\epsilon$ can be read off. The velocity floor on the linear path is $\mathrm{Var}(\mathbf{x}_0 \mid \mathbf{x}_t)/t^2$, summed over coordinates; it is about $2$ near the data and grows toward the noise end, where the clean point is hardest to guess. A score target $-\boldsymbol\epsilon/\sigma_t$ would have the noise floor divided by $\sigma_t^2$, which grows without bound near the data (:exref:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor`). Losses of different targets therefore cannot be compared; predictors trained on different targets can, through the dictionary.

## Sampling by Integration

We have two trained networks and a dictionary that turns either into any target. To generate, we need one more fact about the marginal velocity: it carries the noise distribution to the data distribution. Sampling then means following the field from noise to data, either along deterministic trajectories or with fresh noise at every step.

### The Probability-Flow ODE and Euler's Method

The marginal velocity moves the density $p_t$ of $\mathbf{x}_t$ along the path: in the language of :numref:`sec_mdl-flow-matching`, the pair $(p_t, \mathbf{u}_t)$ satisfies the continuity equation. A point drawn from $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and carried backward in time by the ordinary differential equation

$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}), \qquad \textrm{from } t = 1 \textrm{ down to } t = 0,
$$
:eqlabel:`eq_diffusion-flow-ode`

is therefore a draw from $p_t$ at every time, and from the data at $t = 0$. This *probability-flow ODE* is the deterministic twin of the diffusion process (:numref:`sec_mdl-probability-flow-ode`). We build it up from the smallest case, as :citet:`Nakkiran.Bradley.Zhou.ea.2025` do.

Let the data be one point $\mathbf{x}_0$. On the linear path $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$, so a point at $\mathbf{x}$ carries the noise $\boldsymbol\epsilon = (\mathbf{x} - (1 - t)\mathbf{x}_0)/t$ and moves with velocity $\boldsymbol\epsilon - \mathbf{x}_0 = (\mathbf{x} - \mathbf{x}_0)/t$. Every trajectory is a straight line into $\mathbf{x}_0$, and a single step of length one from $t = 1$, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbf{x}_0$, is exact. With two data points the field at $\mathbf{x}$ is the posterior-weighted average of two such straight fields, and the trajectories bend where the posterior switches from one point to the other (:numref:`subsec_mdl-fm-two-point` works this case out). At $t = 1$ the point $\mathbf{x}_1$ is pure noise and says nothing about the data, so the posterior equals the prior and $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$: one step of length one sends every starting point to the mean of the data. Any distribution is a mixture of point masses, with straight conditional paths, a curved marginal flow, and a first step that aims at the mean (:numref:`sec_mdl-ddim-guidance`). For our mixture that mean is $0.5 \cdot (-2.5, -1.5) + 0.3 \cdot (2.5, -1.5) + 0.2 \cdot (0, 2.5) = (-0.5, -0.7)$, between the modes, where no data lie.

Euler's method (:numref:`sec_mdl-euler-runge-kutta`) follows the curve in $K$ straight steps, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/K$ at $t = 1, 1 - 1/K, \ldots, 1/K$. We write it out once and run it with the learned field from $2{,}000$ standard Gaussian starts.

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method}
%%tab pytorch
@torch.no_grad()
def euler(net, x, steps):
    """Euler's method on dx/dt = net(x, t), from t = 1 down to t = 0."""
    for k in range(steps, 0, -1):
        x = x - net(x, torch.full((len(x),), k / steps, device=x.device)) / steps
    return x

torch.manual_seed(3)
x_T = torch.randn(2000, 2, device=device)
print(mix.summary(euler(ema_v, x_T, 100)))
```

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method}
%%tab jax
def euler(net, x, steps):
    """Euler's method on dx/dt = net(x, t), from t = 1 down to t = 0."""
    def step(k, x):
        return x - net(x, jnp.full((len(x),), 1 - k / steps)) / steps
    return jax.lax.fori_loop(0, steps, step, x)

key, k_T = jax.random.split(key)
x_T = jax.random.normal(k_T, (2000, 2))
print(mix.summary(euler(ema_v, x_T, 100)))
```

A hundred steps put the samples on the three modes. The library's sampler takes the same steps in other coordinates. Along every Gaussian path, the rescaled point $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ and the noise-to-signal ratio $\bar\sigma = \sigma_t/\alpha_t$ obey $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ :cite:`Karras.Aittala.Aila.ea.2022`, a form that serves every path and every kind of predictor; on the linear path its Euler step is the step above. The Heun and grid options matter once network evaluations are expensive, and we keep the defaults.

`ode_sample` integrates $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ with Euler's method or Heun's, on the path's uniform grid in $t$ or on Karras et al.'s grid ($\rho = 7$, $\bar\sigma$ from 80 to 0.002).

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab pytorch
@torch.no_grad()  #@save
def ode_sample(pred, x, path, steps, method='euler', grid='uniform', *,
               t_start=1.0, sigma_max=80.0, sigma_min=0.002, rho=7):
    """Euler or Heun on d x_bar / d sigma_bar = eps, from t_start to data."""
    if method not in ('euler', 'heun') or grid not in ('uniform', 'edm'):
        raise ValueError(f'unknown method {method!r} or grid {grid!r}')
    a, s = path.grid(steps, t_start)
    noise = t_start == 1                      # x is pure noise, else x_t
    if grid == 'uniform':
        r = [s_k / a_k if a_k > 0 else math.inf for a_k, s_k in zip(a, s)]
    else:                                     # Karras et al.'s grid
        if steps < 2:
            raise ValueError('the edm grid needs steps > 1')
        hi = (sigma_max if noise else s[0] / a[0]) ** (1 / rho)
        lo = sigma_min ** (1 / rho)
        r = [(hi + i / (steps - 1) * (lo - hi)) ** rho
             for i in range(steps)] + [0.0]
    def at(xb, r):                            # pred at x = alpha x_bar
        if math.isinf(r):                     # here x_bar holds the noise
            return pred(xb, 0.0, 1.0)
        a = 1 / math.sqrt(1 + r ** 2)
        return pred(xb * a, a, r * a)
    xb = x if math.isinf(r[0]) else x * r[0] if noise else x / a[0]
    for r_t, r_s in zip(r[:-1], r[1:]):
        x0, eps = at(xb, r_t)
        x_next = x0 + r_s * eps               # the Euler (= DDIM) step
        if method == 'heun' and r_s > 0 and not math.isinf(r_t):
            x_next = xb + (r_s - r_t) * (eps + at(x_next, r_s)[1]) / 2
        xb = x_next
    return xb
```

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab jax
def ode_sample(pred, x, path, steps, method='euler',  #@save
               grid='uniform', *, t_start=1.0, sigma_max=80.0,
               sigma_min=0.002, rho=7):
    """Euler or Heun on d x_bar / d sigma_bar = eps, from t_start to data."""
    if method not in ('euler', 'heun') or grid not in ('uniform', 'edm'):
        raise ValueError(f'unknown method {method!r} or grid {grid!r}')
    noise = t_start == 1                      # x is pure noise, else x_t
    with jax.ensure_compile_time_eval():      # the grid is a constant
        a, s = path.grid(steps, t_start)
        if grid == 'uniform':
            r = s / a
        else:                                 # Karras et al.'s grid
            if steps < 2:
                raise ValueError('the edm grid needs steps > 1')
            hi = (sigma_max if noise else s[0] / a[0]) ** (1 / rho)
            i = jnp.arange(steps) / (steps - 1)
            r = jnp.append((hi + i * (sigma_min ** (1 / rho) - hi)) ** rho,
                           0.0)
        at_inf = bool(jnp.isinf(r[0]))
    xb = x if at_inf else x * r[0] if noise else x / a[0]
    def step(xb, r):
        r_t, r_s = r
        finite = jnp.isfinite(r_t)            # x_bar holds the noise at inf
        a_t = jnp.where(finite, 1 / jnp.sqrt(1 + r_t ** 2), 0.0)
        x0, eps = pred(xb * jnp.where(finite, a_t, 1.0), a_t,
                       jnp.where(finite, r_t * a_t, 1.0))
        x_next = x0 + r_s * eps               # the Euler (= DDIM) step
        a_s = 1 / jnp.sqrt(1 + r_s ** 2)
        correct = lambda _: xb + (r_s - r_t) * (
            eps + pred(x_next * a_s, a_s, r_s * a_s)[1]) / 2
        heun = (method == 'heun') & finite & (r_s > 0)
        return jax.lax.cond(heun, correct, lambda _: x_next, None), None
    return jax.lax.scan(step, xb, (r[:-1], r[1:]))[0]
```

How many steps does the flow need? We wrap the velocity network with `make_pred` and integrate from the same starts with one, ten and a hundred steps, and check that a hundred steps of `ode_sample` reproduce the hand-written loop.

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab pytorch
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear, steps) for steps in (1, 10, 100)]
print('one step: mean', samples[0].mean(0).tolist(),
      'std', samples[0].std(0).tolist())
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps} steps:', mix.summary(x))
print('largest difference to the loop:',
      (samples[2] - euler(ema_v, x_T, 100)).abs().max().item())
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'],
                   data=mix.sample(2000));
```

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab jax
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear, steps) for steps in (1, 10, 100)]
print('one step: mean', samples[0].mean(0).tolist(),
      'std', samples[0].std(0).tolist())
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps} steps:', mix.summary(x))
print('largest difference to the loop:',
      float(jnp.abs(samples[2] - euler(ema_v, x_T, 100)).max()))
key, k_data = jax.random.split(key)
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'],
                   data=mix.sample(k_data, 2000));
```

One step lands every start on the data mean $(-0.5, -0.7)$, with almost no spread, exactly as the point-mass argument predicts. Ten steps recover the three modes with weights within about $0.02$ of the targets, but about three percent of the samples stray between the modes. A hundred steps bring the strays down to about one percent, and the library's sampler agrees with the loop up to rounding. Straight conditional paths do not make a straight marginal flow, and the extra steps follow its curvature.

### The Reverse SDE and the Ancestral Chain

DDPM samples differently: it runs the reverse chain of :numref:`fig_diffusion-forward-reverse`, one stochastic step at a time. The reversal of a small forward step is nearly Gaussian :cite:`sohl2015deep`, with a mean that a denoiser predicts. Given the clean point the reversal is exactly Gaussian: for $s < t$, the distribution $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ has a mean that is linear in $\mathbf{x}_t$ and $\mathbf{x}_0$. Ho et al.'s *ancestral* step replaces $\mathbf{x}_0$ by $\hat{\mathbf{x}}_0$ in that mean and adds fresh noise with the variance $\beta = 1 - \bar\alpha_t / \bar\alpha_s$ of the forward step. On the VP path, written with the score, it reads

$$
\mathbf{x}_s = \frac{\mathbf{x}_t + \beta\, \nabla \log p_t(\mathbf{x}_t)}{\sqrt{1 - \beta}} + \sqrt{\beta}\, \mathbf{z},
\qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
:eqlabel:`eq_diffusion-ancestral-score`

To first order in $\beta$, $1/\sqrt{1 - \beta} \approx 1 + \beta/2$, and the step splits into two:

$$
\mathbf{x}_s \approx \Big[ \mathbf{x}_t + \tfrac{\beta}{2} \nabla \log p_t(\mathbf{x}_t) + \sqrt{\beta}\, \mathbf{z} \Big] + \tfrac{\beta}{2} \big( \mathbf{x}_t + \nabla \log p_t(\mathbf{x}_t) \big).
$$

The bracket is a Langevin step of size $h = \beta$ on $p_t$, which leaves $p_t$ in place and pulls stray points back toward it. The second term is one Euler step of the probability-flow ODE, backward in time over the interval that the forward step covered: on the VP path the marginal velocity is $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$, and $\beta \approx \beta(t)\, (t - s)$. Each ancestral step thus corrects toward $p_t$ and then moves along the flow.

In continuous time the reversal is exact, by a theorem of :citet:`Anderson.1982`. A forward process $d\mathbf{x} = \mathbf{f}\, dt + g\, d\mathbf{W}$ is reversed, from $t = 1$ down to $0$, by

$$
d\mathbf{x} = \big[ \mathbf{f} - g^2 \nabla \log p_t(\mathbf{x}) \big]\, dt + g\, d\bar{\mathbf{W}},
\qquad\textrm{which has the same marginals as}\qquad
\frac{d\mathbf{x}}{dt} = \mathbf{f} - \tfrac12 g^2 \nabla \log p_t(\mathbf{x}),
$$
:eqlabel:`eq_diffusion-reverse`

where $\bar{\mathbf{W}}$ is a Brownian motion run backward. For the VP path, $\mathbf{f} = -\tfrac12 \beta(t)\mathbf{x}$ and $g^2 = \beta(t)$, and the right-hand ODE is the flow we just integrated. Between the two lies a family of samplers with the same marginals: the more noise a sampler injects, the larger the score correction its drift carries (:numref:`sec_mdl-time-reversal`, :numref:`fig_mdl-dyn-lambda-family`). The ancestral step is, to first order, the Euler--Maruyama step (:numref:`sec_mdl-euler-maruyama`) of the reverse SDE.

`ancestral_sample` is Ho et al.'s sampler for any predictor `pred(x, alpha, sigma) -> (x0, eps)`: each step moves to the mean of $q(\mathbf{x}_s \mid \mathbf{x}_t, \hat{\mathbf{x}}_0)$ and adds fresh noise with the variance $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ of the forward step from $s$ to $t$ (which is $\beta_t$ on a variance-preserving path).

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab pytorch
@torch.no_grad()  #@save
def ancestral_sample(pred, x, path, steps, *, t_start=1.0):
    """Ho et al.'s sampler from x at t_start (1 is noise) to data at t = 0."""
    a, s = path.grid(steps, t_start)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        rho = s[i + 1] * a[i] / (a[i + 1] * s[i])
        x = a[i + 1] * x0 + s[i + 1] * rho * eps        # the posterior mean
        if s[i + 1] > 0:                                # no noise at t = 0
            var = s[i] ** 2 - (a[i] / a[i + 1] * s[i + 1]) ** 2  # forward step
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x
```

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab jax
def ancestral_sample(pred, x, path, steps, key, *, t_start=1.0):  #@save
    """Ho et al.'s sampler from x at t_start (1 is noise) to data at t = 0."""
    a, s = path.grid(steps, t_start)
    a_t, s_t, a_s, s_s = a[:-1], s[:-1], a[1:], s[1:]
    rho = s_s * a_t / (a_s * s_t)
    var = jnp.maximum(s_t ** 2 - (a_t / a_s * s_s) ** 2, 0)  # forward step
    noise = jnp.where(s_s > 0, jnp.sqrt(var), 0)             # none at t = 0
    def step(x, inputs):
        a_t, s_t, a_s, c_eps, c_noise, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + c_eps * eps                      # the posterior mean
        return x + c_noise * jax.random.normal(key, x.shape), None
    return jax.lax.scan(step, x, (a_t, s_t, a_s, s_s * rho, noise,
                                  jax.random.split(key, steps)))[0]
```

We run two hundred steps of the chain with the noise model, from the same $2{,}000$ starts as the flow.

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain-1}
%%tab pytorch
pred_eps = make_pred(ema_eps, vp, 'eps')
print(mix.summary(ancestral_sample(pred_eps, x_T, vp, 200)))
```

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain-1}
%%tab jax
pred_eps = make_pred(ema_eps, vp, 'eps')
key, k_anc = jax.random.split(key)
print(mix.summary(ancestral_sample(pred_eps, x_T, vp, 200, k_anc)))
```

The chain recovers the three modes with the right weights, up to the sampling error of $2{,}000$ points, and about as few strays as the flow at a hundred steps. :citet:`ho2020denoising` ran a thousand steps; on two dimensions two hundred suffice. The ancestral chain and the probability-flow ODE are two samplers for the same marginals, and either can use either network through `make_pred`.

## Experiments on the Running Example

Neither network ever saw the exact posterior. The mixture lets us compare each with the oracle in three ways: the predictions themselves, the samples they generate, and the dictionary that is supposed to turn one network into the other.

### The Learned Predictor against the Optimal One

The regression lemma splits the loss of any predictor into two parts: $\mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}_{\boldsymbol\theta}\|^2 = \mathbb{E}\|\boldsymbol\epsilon - \boldsymbol\epsilon^\star\|^2 + \mathbb{E}\|\hat{\boldsymbol\epsilon}_{\boldsymbol\theta} - \boldsymbol\epsilon^\star\|^2$, where $\boldsymbol\epsilon^\star$ is the oracle's noise prediction. The first part is the floor, and the second, the gap, is all that training can still reduce. We evaluate the floor, the gap and the loss of the noise model on $20{,}000$ fresh points at four times of the VP path.

```{.python .input #gaussian-paths-the-learned-predictor-against-the-optimal-one}
%%tab pytorch
def sq(u, w):
    """Mean squared distance between two batches of points."""
    return ((u - w) ** 2).sum(1).mean().item()

torch.manual_seed(4)
x0, eps = mix.sample(20_000), torch.randn(20_000, 2, device=device)
print('   t     gap   floor    loss')
with torch.no_grad():
    for t in (0.1, 0.3, 0.5, 0.7):
        tt = torch.full((len(x0),), t, device=device)
        x = vp.sample(x0, tt, eps)
        eps_hat = ema_eps(x, tt)
        eps_star = oracle(x, vp.alpha(tt), vp.sigma(tt))[1]
        print(f'{t:4} {sq(eps_hat, eps_star):7.4f} {sq(eps, eps_star):7.4f}'
              f' {sq(eps, eps_hat):7.4f}')
```

```{.python .input #gaussian-paths-the-learned-predictor-against-the-optimal-one}
%%tab jax
def sq(u, w):
    """Mean squared distance between two batches of points."""
    return float(((u - w) ** 2).sum(1).mean())

key, k_x0, k_eps = jax.random.split(key, 3)
x0, eps = mix.sample(k_x0, 20_000), jax.random.normal(k_eps, (20_000, 2))
print('   t     gap   floor    loss')
for t in (0.1, 0.3, 0.5, 0.7):
    tt = jnp.full((len(x0),), t)
    x = vp.sample(x0, tt, eps)
    eps_hat = ema_eps(x, tt)
    eps_star = oracle(x, vp.alpha(tt), vp.sigma(tt))[1]
    print(f'{t:4} {sq(eps_hat, eps_star):7.4f} {sq(eps, eps_star):7.4f}'
          f' {sq(eps, eps_hat):7.4f}')
```

The loss column equals the floor plus the gap, to within the Monte Carlo error. The gap is at most a few hundredths at every time, on a target with unit variance per coordinate, and small next to the floor except near the noise end, where the floor itself vanishes. The floor falls with $t$: near the data the noise is hard to separate from the spread of a mode, and near the noise end $\mathbf{x}_t$ is almost all noise. The network is close to the oracle wherever the data put their mass, and nothing in the loss can tell us more than this.

### Samples from Both Samplers

The gap measures one time at a time; a sampler accumulates the errors of all its steps. We draw $20{,}000$ samples with each sampler, two hundred ancestral steps for the noise model and a hundred Euler steps for the velocity model, and repeat both with the oracle in place of the network. At this sample size a mode weight carries a standard error of about $0.0035$.

```{.python .input #gaussian-paths-samples-from-both-samplers}
%%tab pytorch
torch.manual_seed(5)
x_T = torch.randn(20_000, 2, device=device)
runs = {'data': mix.sample(20_000),
        'ancestral, network': ancestral_sample(pred_eps, x_T, vp, 200),
        'ancestral, oracle': ancestral_sample(oracle, x_T, vp, 200),
        'Euler, network': ode_sample(pred_v, x_T, linear, 100),
        'Euler, oracle': ode_sample(oracle, x_T, linear, 100)}
for name, x in runs.items():
    print(f'{name:>18}:', mix.summary(x))
d2l.scatter_panels([runs[k][:2000] for k in runs],
                   list(runs), lim=4.5, size=2.4);
```

```{.python .input #gaussian-paths-samples-from-both-samplers}
%%tab jax
key, k_T, k_data, k_net, k_oracle = jax.random.split(key, 5)
x_T = jax.random.normal(k_T, (20_000, 2))
runs = {'data': mix.sample(k_data, 20_000),
        'ancestral, network': ancestral_sample(pred_eps, x_T, vp, 200, k_net),
        'ancestral, oracle': ancestral_sample(oracle, x_T, vp, 200, k_oracle),
        'Euler, network': ode_sample(pred_v, x_T, linear, 100),
        'Euler, oracle': ode_sample(oracle, x_T, linear, 100)}
for name, x in runs.items():
    print(f'{name:>18}:', mix.summary(x))
d2l.scatter_panels([runs[k][:2000] for k in runs],
                   list(runs), lim=4.5, size=2.4);
```

The ancestral chain with the network recovers the weights $0.5/0.3/0.2$ within sampling error and strays less than one percent of the time. Euler with the network matches the weights to within a few hundredths and strays as often as Euler with the oracle, about one percent: what remains is the error of the hundred steps, not of the network. Both samplers turn a regression on per-example targets into the full mixture, mode weights included, which no single noise level of the previous section could do.

### The Dictionary Checked

The velocity model was trained on the linear path and the noise model on the VP path. If both learned the posterior mean, each should match the linear-path velocity that :eqref:`eq_diffusion-linear-dictionary` computes from the exact score of $p_t$, which `mix.score` evaluates with $\alpha = 1 - t$ and $\sigma = t$. For the noise model, `make_pred` finds the VP time with the same log-SNR, rescales the point, and returns $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$, whose difference is the linear-path velocity. We stop at $t = 0.95$, because a noise model has no clean-point estimate at $\alpha = 0$.

```{.python .input #gaussian-paths-the-dictionary-checked}
%%tab pytorch
print('   t  E|u|^2  velocity net  noise net')
with torch.no_grad():
    for t in (0.2, 0.5, 0.8, 0.95):
        tt = torch.full((len(x0),), t, device=device)
        x = linear.sample(x0, tt, eps)
        u = -(x + t * mix.score(x, 1 - t, t)) / (1 - t)  # from the score
        x0_hat, eps_hat = pred_eps(x, 1 - t, t)
        print(f'{t:4} {sq(u, 0 * u):7.2f} {sq(ema_v(x, tt), u):13.4f}'
              f' {sq(eps_hat - x0_hat, u):10.4f}')
```

```{.python .input #gaussian-paths-the-dictionary-checked}
%%tab jax
print('   t  E|u|^2  velocity net  noise net')
for t in (0.2, 0.5, 0.8, 0.95):
    tt = jnp.full((len(x0),), t)
    x = linear.sample(x0, tt, eps)
    u = -(x + t * mix.score(x, 1 - t, t)) / (1 - t)  # from the score
    x0_hat, eps_hat = pred_eps(x, 1 - t, t)
    print(f'{t:4} {sq(u, 0 * u):7.2f} {sq(ema_v(x, tt), u):13.4f}'
          f' {sq(eps_hat - x0_hat, u):10.4f}')
```

The velocity network differs from the field computed from the score by less than a quarter of a percent of the field's mean squared norm, which ranges from about $8$ at $t = 0.2$ to about $1$ at $t = 0.8$. The noise network, converted through two paths and a change of time, stays within about one percent; its error is largest at $t = 0.95$, where $\alpha$ is small and the conversion divides by it. No score was learned and no velocity appears in the noise model's loss: the agreement is the dictionary at work.

## Summary

A Gaussian path $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ connects the data at $t = 0$ to noise at $t = 1$ through every noise level, and a walk down this path settles the mode weights at coarse levels and the details at fine ones, which no single level can do. One regression onto a per-example target trains one network at every point of the path, because least squares recovers the posterior mean. The score, the noise, the clean point, $v$ and the velocity are affine functions of that one posterior mean, so the models of the literature differ in their choice of path, target and sampler, not in what they learn (:numref:`tab_diffusion_models`). Samples come from integrating the probability-flow ODE, or from the ancestral chain, which adds noise and a matching score correction at each step. On the mixture both learned models sit at their loss floors, agree with each other through the dictionary, and recover the mixture's weights.

:The models of this chapter. Each row names what the network represents, the target and path that train it, and the sampler that produces data.
:label:`tab_diffusion_models`

| model | the network represents | target and path | sampler | where |
|:--|:--|:--|:--|:--|
| denoising score model | the score at one noise level | the noise, at one level | Langevin dynamics | :numref:`sec_diffusion-denoising` |
| noise-conditional score network | the scores at a sequence of noise levels | the noise, at each level, weighted by $\tilde\sigma^2$ | annealed Langevin dynamics | this section, :exref:`ex_gaussian-paths-ncsn` |
| DDPM | the noise $\hat{\boldsymbol\epsilon}$ | $\boldsymbol\epsilon$ on the VP path | the ancestral chain, or the probability-flow ODE | this section, :numref:`sec_diffusion-images` |
| flow matching, rectified flow | the velocity $\mathbf{u}_t$ | $\boldsymbol\epsilon - \mathbf{x}_0$ on the linear path | Euler steps on the probability-flow ODE | this section, :numref:`sec_diffusion-samplers` |
| $v$-prediction | $\alpha_t \hat{\boldsymbol\epsilon} - \sigma_t \hat{\mathbf{x}}_0$ | $\mathbf{v}$ on the cosine path | either, through the dictionary | :numref:`sec_diffusion-design-space` |
| masked diffusion | the clean tokens of a partly masked sequence | a weighted cross-entropy on the masked tokens | iterative unmasking | :numref:`sec_diffusion-discrete` |

Nothing in this section used the fact that $\mathbf{x}$ had two coordinates. An image has thousands, and its pixels are strongly correlated; the next question is what network, and what changes to training and sampling, a path through image space needs.

## Exercises

1. **The marginal of the linear path.** The linear path :eqref:`eq_diffusion-linear-path` joins a draw $\mathbf{x}_0$ of the mixture, with weights $\pi_k$, means $\boldsymbol\mu_k$ and standard deviation $s = 0.5$ per coordinate, to independent noise $\boldsymbol\epsilon$.
    1. Show that $p_t$ is the mixture with weights $\pi_k$, means $(1 - t)\boldsymbol\mu_k$ and per-coordinate variance $(1 - t)^2 s^2 + t^2$. Derive :eqref:`eq_diffusion-linear-dictionary` from :eqref:`eq_diffusion-score-velocity`.
    1. Show from :eqref:`eq_diffusion-marginal-velocity` that $\mathbf{u}_t(\mathbf{x}) \to \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$ as $t \to 1$, and conclude where one Euler step from $t = 1$ to $t = 0$ sends every start.
    1. With two Euler steps the first ends at $t = 1/2$. Show that it maps $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to $\mathcal{N}(\mathbb{E}[\mathbf{x}_0]/2, \mathbf{I}/4)$, and compare the center and the width of this cloud with those of $p_{1/2}$. Which mode should the two-step samples overweight, and why?
   :exlabel:`ex_gaussian-paths-linear-marginal`
1. **Velocity, noise and clean-point targets.** On the linear path, take the posterior mean of $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$.
    1. Show that the marginal velocity equals both $(\mathbf{x} - \hat{\mathbf{x}}_0)/t$ and $(\hat{\boldsymbol\epsilon} - \mathbf{x})/(1 - t)$.
    1. A network predicts the clean point, or the noise, with a root-mean-square error $\delta$ per coordinate, and the velocity is computed from its output by the first part. What is the error of the velocity as a function of $t$ in each case? Near which end of the path is each prediction a poor source of the velocity? Relate the answer to the reason why a noise model is never queried at $\alpha = 0$.
    1. Show that the velocity loss of $\mathbf{v}_{\boldsymbol\theta} = (\mathbf{x} - \hat{\mathbf{x}}_{0,\boldsymbol\theta})/t$ equals the clean-point loss $\mathbb{E}\|\hat{\mathbf{x}}_{0,\boldsymbol\theta} - \mathbf{x}_0\|^2$ weighted by $1/t^2$, and the velocity loss of $\mathbf{v}_{\boldsymbol\theta} = (\hat{\boldsymbol\epsilon}_{\boldsymbol\theta} - \mathbf{x})/(1 - t)$ equals the noise loss weighted by $1/(1 - t)^2$. Which times does flow matching emphasize relative to each of the two unweighted losses?
   :exlabel:`ex_gaussian-paths-prediction-targets`
1. [code] **The loss floor.** Use `oracle` and `loss_floor`.
    1. Plot the floor of the velocity target on the linear path and of the noise target on the VP path against $t$ on a grid in $(0, 1)$. Where is each largest? The printed training loss is a mean over $1{,}000$ minibatches of $512$ points; estimate its standard error from `losses_v`, and decide whether a printed loss below the floor would be a contradiction.
    1. Let $\boldsymbol\Sigma$ be the covariance matrix of the mixture, with eigenvalues $\kappa_1, \kappa_2$. For Gaussian data with this covariance, show that the velocity floor on the linear path at time $t$ is $\sum_i \kappa_i / \big((1 - t)^2 \kappa_i + t^2\big)$. Add this curve to the plot, and explain why it cannot lie below the floor of the mixture. Why do the two floors agree at both ends of the path?
    1. The score target $-\boldsymbol\epsilon/\sigma_t$ has the floor of the noise target divided by $\sigma_t^2$. Evaluate it on the VP path at $t = 0.01$, $0.1$ and $0.5$, and explain why no network is trained on this target with $t$ uniform on $(0, 1)$.
   :exlabel:`ex_gaussian-paths-loss-floor`
1. **Gaussian data.** Let the data follow $\mathcal{N}(\mathbf{m}, s^2 \mathbf{I})$ in $\mathbb{R}^d$, connected to independent noise by the linear path, and write $\bar s_t^2 = (1 - t)^2 s^2 + t^2$.
    1. Show that $p_t = \mathcal{N}\big((1 - t)\mathbf{m}, \bar s_t^2 \mathbf{I}\big)$. Compute the marginal velocity by conditioning the jointly Gaussian pair $(\boldsymbol\epsilon - \mathbf{x}_0, \mathbf{x}_t)$, and show that $\mathbf{u}_t(\mathbf{x}) = -\mathbf{m} + (\dot{\bar s}_t / \bar s_t)\big(\mathbf{x} - (1 - t)\mathbf{m}\big)$. Check the limits $t \to 0$ and $t \to 1$.
    1. Show that the conditional variance of the velocity target given $\mathbf{x}_t$, summed over coordinates, is $d\, s^2 / \bar s_t^2$. At which $t$ is it largest, and what is it there for $s = 0.5$? :exref:`ex_mdl-flow-matching-transport-kinetic-energy` continues with the trajectories and the kinetic energy of this flow.
   :exlabel:`ex_gaussian-paths-gaussian-data`
1. **The ancestral step to first order, for Gaussian data.** Let the data be $\mathcal{N}(\mathbf{0}, \nu \mathbf{I})$, so that on the DDPM chain $q_k = \mathcal{N}(\mathbf{0}, v_k \mathbf{I})$ with $v_k = \bar\alpha_k \nu + 1 - \bar\alpha_k$.
    1. Show that one ancestral step :eqref:`eq_diffusion-ancestral-score` with the exact score and step variance $\beta_k$ maps $q_k$ to $\mathcal{N}\big(\mathbf{0}, (v_{k-1} + \delta_k)\mathbf{I}\big)$ with $\delta_k = \beta_k^2 (1/v_k - 1)/(1 - \beta_k)$.
    1. For which data is the step exact? Is the variance too large or too small for data narrower than the standard Gaussian, such as one component of the mixture, $\nu = 0.25$?
    1. How does the error accumulated over the chain change when the schedule is refined to twice as many steps with half the $\beta_k$?
   :exlabel:`ex_gaussian-paths-ancestral-first-order`
1. **Step size and Gaussian reversal.** The ancestral chain uses Gaussian transitions, which are justified for small steps. This problem measures how small. Write $\sigma_{t|s}^2 = \sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ for the variance of the forward step from $s$ to $t > s$.
    1. Show that the exact reversal is the mixture $q(\mathbf{x}_s \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \mid \mathbf{x}_t}\big[q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)\big]$ and that its covariance is $\tau^2 \mathbf{I} + c^2\, \mathrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$, where $\tau^2 = \sigma_{t|s}^2 \sigma_s^2/\sigma_t^2$ is the variance of $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ and $c = \alpha_s \sigma_{t|s}^2/\sigma_t^2$ is the coefficient of $\mathbf{x}_0$ in its mean.
    1. Let the data be $\pm 1$ with equal probability in one dimension, and let $x_t = 0$. Show that the reversal is the equal mixture of $\mathcal{N}(\pm c, \tau^2)$, which is bimodal exactly when $\rho^2 = c^2/\tau^2$ exceeds one, and that $\rho^2 = \alpha_s^2 \sigma_{t|s}^2 / (\sigma_t^2 \sigma_s^2)$. How does $\rho$ shrink as the step $t - s$ is refined?
    1. Evaluate $\rho$ on the VP path from $t = 0.5$ to $s = 0.495$, $0.45$ and $0.3$, and for one step from $t = 0.3$ to $s = 0$. Which reversals are bimodal? What does a Gaussian transition with the optimal mean produce at $x_t = 0$ in the last case, and what should it produce?
   :exlabel:`ex_gaussian-paths-gaussian-reversal`
1. [code] **Commitment time.** When along the path does a sample decide its mode? :citet:`Biroli.Bonnaire.deBortoli.ea.2024` measure this by cloning: start at a point of $q_t$, run two copies of the reverse chain with independent noise, and record whether they end in the same mode.
    1. Draw $2{,}000$ clean points, noise them to $\mathbf{x}_t$ on the VP path for $t = 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1$, and run two copies of `ancestral_sample(pred_eps, x_t, vp, steps, t_start=t)` from each with about $200\, t$ steps. Plot the fraction $\varphi(t)$ of pairs whose ends have the same nearest mean. Which value does $\varphi$ approach as $t \to 1$, and why?
    1. Let $\Lambda$ be the largest eigenvalue of the covariance of the data. Biroli et al. define the speciation time by $\bar\alpha_t \Lambda = 1$, where the signal along the top principal direction stops dominating the noise. Compute it for the mixture and compare it with your curve and with the ladder of this section, whose coarse levels decided the mode weights. :exref:`ex_mdl-flow-matching-transport-late-decisions` and :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` work out the same question for two modes in one dimension.
   :exlabel:`ex_gaussian-paths-commitment-time`
1. **Same ratio, same flow.** Let two Gaussian paths have coefficients related by $\alpha^B_t = c_t\, \alpha^A_t$ and $\sigma^B_t = c_t\, \sigma^A_t$ for a smooth positive function $c_t$, so that they have the same log-SNR at every time.
    1. Show that $p^B_t(\mathbf{y}) = c_t^{-d}\, p^A_t(\mathbf{y}/c_t)$, and use :eqref:`eq_diffusion-score-velocity` to show that $\mathbf{u}^B_t(\mathbf{y}) = (\dot c_t / c_t)\, \mathbf{y} + c_t\, \mathbf{u}^A_t(\mathbf{y}/c_t)$. Conclude that the trajectories of the probability-flow ODE are related by $\mathbf{y}_t = c_t\, \mathbf{x}_t$.
    1. Let $\tau(t)$ be increasing. Show that the path $p_{\tau(t)}$ is generated by the field $\dot\tau(t)\, \mathbf{u}_{\tau(t)}$, with the same trajectories as $\mathbf{u}$.
    1. Find the change of time $\tau(t)$ for which the cosine path $(\cos\frac{\pi\tau}{2}, \sin\frac{\pi\tau}{2})$ at time $\tau(t)$ has the log-SNR of the linear path at time $t$, and show that $c_t = 1/\sqrt{(1 - t)^2 + t^2}$. Where along the path is the rescaling largest?
    1. What does the probability-flow ODE of the cosine path do to data distributed as $\mathcal{N}(\mathbf{0}, \mathbf{I})$, and why? What does that of the linear path do to the same data?
   :exlabel:`ex_gaussian-paths-same-ratio-same-flow`
1. **Ladder spacing in high dimensions.** Let the data be a single point at the origin of $\mathbb{R}^d$, so that $p_{\tilde\sigma} = \mathcal{N}(\mathbf{0}, \tilde\sigma^2 \mathbf{I})$, and let the ladder be geometric with ratio $\gamma = \tilde\sigma_{i-1}/\tilde\sigma_i > 1$.
    1. Show that $r = \|\mathbf{x}\|$ for $\mathbf{x} \sim p_{\tilde\sigma}$ satisfies $\mathbb{E}[r^2] = d\tilde\sigma^2$ and $\mathrm{Var}[r^2] = 2d\tilde\sigma^4$, and conclude that $r$ is approximately $\mathcal{N}(\tilde\sigma\sqrt d, \tilde\sigma^2/2)$ for large $d$.
    1. A chain that leaves level $i - 1$ with a sample of $p_{\tilde\sigma_{i-1}}$ enters level $i$ at a radius that lies, with high probability, in $\tilde\sigma_{i-1}(\sqrt d \pm 3/\sqrt 2)$. Show that $p_{\tilde\sigma_i}$ puts the mass $\Phi\big(\sqrt{2d}(\gamma - 1) + 3\gamma\big) - \Phi\big(\sqrt{2d}(\gamma - 1) - 3\gamma\big)$ on this interval, where $\Phi$ is the standard normal distribution function. Why does a mass near zero mean that the chain starts the level where the network has seen almost no training inputs?
    1. :citet:`Song.Ermon.2020` choose $\gamma$ so that this mass is about one half. Solve for $\gamma$ at $d = 3072$, the dimension of CIFAR-10 images, and count the levels from $\tilde\sigma = 50$ down to $0.01$. Compare with the $232$ levels of their CIFAR-10 model.
    1. The top level must also be coarse enough to merge the modes. Two equal Gaussian components at $\pm\mu$ with variance $v$ have a single mode exactly when $\mu^2 \le v$ (:exref:`ex_mdl-fokker-planck-probability-flow-mode-merging`). Use this criterion on the two lower modes of the mixture to estimate the noise level at which they merge, and compare it with the ladder's top level of $3$.
   :exlabel:`ex_gaussian-paths-ladder-spacing`
1. **Annealing the proposal.** The previous section estimated a normalizer by importance sampling: with the target $\mathcal{N}(\mathbf{m}, \mathbf{I})$, $\mathbf{m} = (1, \ldots, 1) \in \mathbb{R}^d$, and the proposal $\mathcal{N}(\mathbf{0}, \mathbf{I})$, the relative error of $n$ draws is $\sqrt{(e^d - 1)/n}$. Annealed importance sampling :cite:`Neal.2001` covers the distance in $T$ smaller steps. Let $f_j(\mathbf{x}) = \exp\big(-\|\mathbf{x} - (j/T)\mathbf{m}\|^2/2\big)$ for $j = 0, \ldots, T$; every $f_j$ has the same normalizer. One run draws $\mathbf{x}_{j-1}$ from the density proportional to $f_{j-1}$, independently for $j = 1, \ldots, T$, and returns $W = \prod_j f_j(\mathbf{x}_{j-1})/f_{j-1}(\mathbf{x}_{j-1})$; the estimate of the ratio of normalizers, which is one, averages $W$ over $n$ runs.
    1. Show that $\mathbb{E}[W] = 1$ and that $\log W$ is Gaussian with variance $d/T$. Conclude that the relative standard error is $\sqrt{(e^{d/T} - 1)/n}$.
    1. A run costs $T$ evaluations of an energy. For a fixed budget of $N = nT$ evaluations, find the limit of the relative variance as $T \to \infty$. Beyond roughly how many stages do further stages stop helping?
    1. For a learned energy the intermediate densities cannot be sampled exactly, so each $\mathbf{x}_j$ is obtained from $\mathbf{x}_{j-1}$ by a few steps of a Markov chain that leaves $f_j$ invariant. What then determines the variance of $W$? :citet:`sohl2015deep` built the first diffusion model on this annealing. Which object of this section plays the role of the sequence $f_j$?
   :exlabel:`ex_gaussian-paths-annealed-importance-sampling`
1. [code] **The exact optimum memorizes.** A finite training set is itself a distribution, a mixture of point masses, and its exact velocity field is known. :citet:`Bertrand.Gagneux.Massias.ea.2025` take as their starting point that integrating this field reproduces the training points, and ask why trained networks generalize instead.
    1. Draw $64$ points from the mixture and train `TimeMLP(256, 3)` with `regression_loss(linear, 'velocity')` on batches drawn from these points only.
    1. `d2l.GaussianMixture(points, [1/64] * 64, 0.0)` is the empirical distribution, and its `velocity` method is the exact field for $t > 0$. Integrate it and the trained network with `euler` from the same $2{,}000$ starts. For both sets of samples, plot the histogram of the distance to the nearest training point.
    1. Which sampler reproduces the training set, and which produces new points? Bertrand et al. argue that the noise in the regression targets is not what makes the network generalize. Repeat the first two parts with $1{,}024$ training points, and describe how the two histograms change.
   :exlabel:`ex_gaussian-paths-memorization`
1. [extended] **A noise-conditional score network on the mixture.** Train the model of :citet:`song2019generative` on the running example and compare it with the samplers of this section.
    1. Train one `TimeMLP` on the ten levels of the ladder, with the level's log-noise $\log\tilde\sigma_i$ as its time input, by denoising score matching weighted by $\tilde\sigma_i^2$: the loss of level $i$ is $\mathbb{E}\|\tilde\sigma_i\, \mathbf{s}_{\boldsymbol\theta}(\mathbf{x}_0 + \tilde\sigma_i \boldsymbol\epsilon, \tilde\sigma_i) + \boldsymbol\epsilon\|^2$. Why does the weight make the levels comparable?
    1. Run annealed Langevin dynamics with the learned scores, $100$ steps per level as in the text, and compare the summary with the exact-score ladder.
    1. The ladder used $1{,}000$ network evaluations. Compare it with `ancestral_sample` and `ode_sample` at the same number of evaluations, and with fewer. Which hand-chosen quantities of the ladder does the path make unnecessary?
   :exlabel:`ex_gaussian-paths-ncsn`
