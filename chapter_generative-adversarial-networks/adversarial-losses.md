# Adversarial Losses Beyond GANs
:label:`sec_gan_beyond`

Almost every regressor in this book was trained with squared error, and for house prices that was the right choice. Ask a network to enlarge a thumbnail, though, and squared error fails. Many sharp photographs shrink to the same thumbnail: the same face with its hair falling slightly differently, the same lawn with its blades of grass elsewhere. Squared error rewards the prediction closest to all of them on average, their pixelwise mean, and that mean is a blur that matches none of them. Absolute error does no better. Pix2pix :cite:`Isola.Zhu.Zhou.ea.2017`, which turns sketches into photographs, has a paired target for every input, and its $L_1$ loss still blurs the fine texture. What sharpens it is a critic that looks at each patch of the output together with the sketch and scores whether it could be a patch of a real photograph of that sketch. Why does squared error average, and why does a learned comparison not?

The question has outlived stand-alone GANs. Most image and video generators are now trained by regression, without a two-player game. Yet a critic still sits inside the fast generators made from slow ones, inside the networks that compress images into codes and decode them again, and inside many of the networks that turn spectrograms into speech. We work out the averaging on the smallest problem that shows it, one input with two answers. With a formula in the role of the slow generator, the same problem then shows what happens when a fast generator learns from a slow one. The section, and with it the chapter, ends by asking what the game still offers now that regression has taken its place.

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Conditional Means and Ambiguity

Many generation problems are conditional predictions. A super-resolution network receives a low-resolution image $c$, the condition, and must produce a high-resolution image $x$, the sample. Pix2pix receives a sketch and must produce a photograph, and a vocoder receives a spectrogram and must produce a waveform. Before comparing losses for such problems, let us ask what the usual loss rewards.

### The Conditional-Mean Decomposition

Write $g$ for the predictor and $\mu(c) = \mathbb{E}[x \mid c]$ for the conditional mean. Adding and subtracting $\mu(c)$ inside the square produces a cross term $2\,\mathbb{E}\big[(x - \mu(c))(\mu(c) - g(c))\big]$, which vanishes: given $c$, the second factor is fixed and the first has mean zero. What remains is

$$
\mathbb{E}\big[(x - g(c))^2\big]
\;=\;
\mathbb{E}\big[(x - \mu(c))^2\big] \;+\; \mathbb{E}\big[(\mu(c) - g(c))^2\big].
$$
:eqlabel:`eq_beyond_condmean`

For an image $x$ the square is a squared norm, a sum over pixels, and the argument holds pixel by pixel. The first term is the spread of $x$ around its mean, which no predictor can remove. The second is the squared distance from $g(c)$ to $\mu(c)$, so squared error grades a predictor by its distance to the conditional mean, and the best predictor is $g = \mu$. When $p(x \mid c)$ concentrates on one answer, the mean is that answer. When it spreads over several, the mean averages them and need not be an answer itself: the mean of many sharp images that share a thumbnail is not sharp. Absolute error, the $L_1$ loss of pix2pix, replaces the mean by a conditional median, a value with half of the answers on either side (Exercise :exref:`ex_adversarial-losses-mean-and-median`). That changes which average is taken, not the problem: the median of an ambiguous set of images is, in general, no more a sharp image than their mean.

### Two Bands
:label:`subsec_gan_beyond_capacity`

The smallest problem that separates the two losses has one input, one output and two answers. Draw the condition $c$ uniformly from $[-1, 1]$ and set

$$
x \;=\; c + b + 0.05\,\xi,
\qquad
b \in \{-1, +1\} \textrm{ with probability } \tfrac12 \textrm{ each},
\quad \xi \sim \mathcal{N}(0, 1),
$$
:eqlabel:`eq_beyond_twoband`

with the offset $b$ drawn independently of $c$. The data form two parallel bands, $x = c \pm 1$, two units apart, forty times the noise width. The conditional mean $\mu(c) = c$ runs through the empty corridor between them, and its expected squared error is $1 + 0.05^2 \approx 1.0025$, the first term of :eqref:`eq_beyond_condmean`. A predictor that follows the band $x = c + 1$ is right for half of the data and two units off for the other half, so its expected squared error is $\tfrac12 \cdot 4 + 0.05^2 \approx 2.0025$. Squared error prefers the corridor, where no data point lies.

A critic sees the same data differently. Let $D(x, c)$ be trained by logistic regression to tell data pairs $(x, c) \sim p$ from pairs $(\hat x, c) \sim q$ formed with the predictor's output. As :numref:`sec_basic_gan` showed, the best such critic is the log density ratio

$$
D^\star(x, c) = \log \frac{p(x, c)}{q(x, c)}.
$$

On the two bands it is strongly negative in the corridor, where predictions can land but data never do. The predictor raises the critic's score of its own outputs by minimizing $-\log \sigma\big(D(\hat x, c)\big)$, the non-saturating loss of :numref:`sec_basic_gan`, so a prediction in the corridor scores low, however close it is on average to both bands. The two losses agree when $p(x \mid c)$ has one answer the predictor can reach. When it has several, squared error favors their mean and the critic favors a prediction among the data.

Let us train one network with each loss. We call them students: here only the data teach them, and in the second experiment a teacher will. Both students are the same deterministic network, a perceptron with two hidden layers of 32 tanh units that maps $c$ to one prediction $\hat x = g(c)$, and both start from the same weights. The critic is a perceptron of the same size with leaky ReLU units that scores pairs $(x, c)$. All three come from `d2l.make_mlp` of :numref:`sec_basic_gan`. Since the data distribution is known, `make_bands` draws a fresh batch at every training step.

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab pytorch
def make_bands(n):
    """Draw n pairs (c, x) from the two bands x = c + b + 0.05 xi."""
    c = torch.rand(n, 1) * 2 - 1
    b = torch.randint(0, 2, (n, 1)) * 2.0 - 1.0
    return c, c + b + 0.05 * torch.randn(n, 1)

torch.manual_seed(0)
g_mse = d2l.make_mlp([1, 32, 32, 1], nn.Tanh())
g_adv = d2l.make_mlp([1, 32, 32, 1], nn.Tanh())
g_adv.load_state_dict(g_mse.state_dict())  # both students start alike
D = d2l.make_mlp([2, 32, 32, 1], nn.LeakyReLU(0.2))
c_data, x_data = make_bands(200)  # a data sample for the plot
```

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab jax
def make_bands(key, n):
    """Draw n pairs (c, x) from the two bands x = c + b + 0.05 xi."""
    k_c, k_b, k_xi = jax.random.split(key, 3)
    c = jax.random.uniform(k_c, (n, 1), minval=-1, maxval=1)
    b = jax.random.rademacher(k_b, (n, 1), dtype=jnp.float32)
    return c, c + b + 0.05 * jax.random.normal(k_xi, (n, 1))

rngs = nnx.Rngs(0)
g_mse = d2l.make_mlp([1, 32, 32, 1], nnx.tanh, rngs=rngs)
g_adv = nnx.clone(g_mse)  # both students start alike
D = d2l.make_mlp([2, 32, 32, 1], lambda h: nnx.leaky_relu(h, 0.2), rngs=rngs)
key_mse, key_adv, key_data = jax.random.split(rngs(), 3)
c_data, x_data = make_bands(key_data, 200)  # a data sample for the plot
```

The first student minimizes squared error with Adam, its learning rate decaying along a cosine from 0.01 to zero over 2,000 steps. We need no experiment to know where it ends: by :eqref:`eq_beyond_condmean` it should follow the line $x = c$, one unit from each band. It is the control for the second student.

```{.python .input #adversarial-losses-two-bands-1}
%%tab pytorch
def train_mse(g, num_steps=2000, batch_size=128, lr=0.01):
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)  # a fresh batch at every step
        opt.zero_grad()
        ((g(c) - x) ** 2).mean().backward()
        opt.step()
        sched.step()

train_mse(g_mse)
```

```{.python .input #adversarial-losses-two-bands-1}
%%tab jax
@nnx.jit(static_argnums=3)
def mse_step(g, opt, key, batch_size):
    c, x = make_bands(key, batch_size)  # a fresh batch at every step
    opt.update(g, nnx.grad(lambda g_: ((g_(c) - x) ** 2).mean())(g))

def train_mse(g, key, num_steps=2000, batch_size=128, lr=0.01):
    schedule = optax.cosine_decay_schedule(lr, num_steps)
    opt = nnx.Optimizer(g, optax.adam(schedule), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        mse_step(g, opt, k, batch_size)

train_mse(g_mse, key_mse)
```

The second student plays the log-loss game, with the losses `d2l.gan_loss_D` and `d2l.gan_loss_G` of :numref:`sec_gan_convergence`. At each step the critic takes one logistic-regression step on real pairs $(x, c)$ against predicted pairs $(\hat x, c)$, and then the student takes one step on $-\log \sigma\big(D(\hat x, c)\big)$ against the updated critic. The prediction enters the game with the data's observation noise, $\hat x = g(c) + 0.05\,\xi'$ with a fresh $\xi' \sim \mathcal{N}(0, 1)$, so the two distributions differ in where they put their mass, not in how smooth they are. A game moves more slowly than a regression, so this student gets 8,000 steps; the squared-error student is at its optimum after 2,000, and more steps would not move it.

```{.python .input #adversarial-losses-two-bands-2}
%%tab pytorch
def train_adversarial(g, D, num_steps=8000, batch_size=128):
    opt_G = torch.optim.Adam(g.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)
        x_hat = g(c) + 0.05 * torch.randn(batch_size, 1)  # the data's noise
        real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
        opt_D.zero_grad()
        d2l.gan_loss_D(D, real, fake.detach()).backward()
        opt_D.step()
        opt_G.zero_grad()
        d2l.gan_loss_G(D, real, fake).backward()
        opt_G.step()

train_adversarial(g_adv, D)
```

```{.python .input #adversarial-losses-two-bands-2}
%%tab jax
@nnx.jit(static_argnums=5)
def adversarial_step(g, D, opt_G, opt_D, key, batch_size):
    k_data, k_xi = jax.random.split(key)
    c, x = make_bands(k_data, batch_size)
    xi = jax.random.normal(k_xi, (batch_size, 1))  # the data's noise
    real = jnp.concatenate([c, x], 1)
    fake = lambda g_: jnp.concatenate([c, g_(c) + 0.05 * xi], 1)
    loss_D = lambda D_: d2l.gan_loss_D(D_, real, fake(g))
    opt_D.update(D, nnx.grad(loss_D)(D))
    loss_G = lambda g_: d2l.gan_loss_G(D, real, fake(g_))
    opt_G.update(g, nnx.grad(loss_G)(g))

def train_adversarial(g, D, key, num_steps=8000, batch_size=128):
    opt_G = nnx.Optimizer(g, optax.adam(1e-3), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(2e-3), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        adversarial_step(g, D, opt_G, opt_D, k, batch_size)

train_adversarial(g_adv, D, key_adv)
```

Three numbers summarize each student on a grid of 201 inputs: the mean distance from its curve to the nearer band, the mean distance to the conditional mean $x = c$, and the fraction of inputs at which the curve is more than 0.25 from both bands. A plot then draws both curves over a sample of the data, with the conditional mean dashed.

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab pytorch
cs = torch.linspace(-1, 1, 201).reshape(-1, 1)
with torch.no_grad():
    curves = {'squared error': g_mse(cs), 'adversarial': g_adv(cs)}
for name, x_hat in curves.items():
    to_band = torch.minimum((x_hat - cs - 1).abs(), (x_hat - cs + 1).abs())
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{(x_hat - cs).abs().mean():.2f} from x = c, '
          f'{(to_band > 0.25).float().mean():.0%} off the bands')
```

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab jax
cs = jnp.linspace(-1, 1, 201).reshape(-1, 1)
curves = {'squared error': g_mse(cs), 'adversarial': g_adv(cs)}
for name, x_hat in curves.items():
    to_band = jnp.minimum(jnp.abs(x_hat - cs - 1), jnp.abs(x_hat - cs + 1))
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{jnp.abs(x_hat - cs).mean():.2f} from x = c, '
          f'{(to_band > 0.25).mean():.0%} off the bands')
```

```{.python .input #adversarial-losses-the-capacity-argument-3}
%%tab pytorch, jax
d2l.plot([c_data, cs, cs, cs],
         [x_data, curves['squared error'], curves['adversarial'], cs],
         '$c$', '$x$', ['data', 'squared-error student',
                        'adversarial student', '$x = c$'],
         ylim=(-2.2, 3.6), fmts=('C0.', 'C2-', 'C1-', 'k--'),
         figsize=(5.5, 3.2))
```

The squared-error student lies on average within about 0.02 of the dashed line $x = c$, a little farther at the ends of the interval, as :eqref:`eq_beyond_condmean` predicts: everywhere it predicts a value that never occurs in the data. The adversarial student is the interesting one. It lies on a band, a few hundredths from it on average, and at most 2% of the inputs fall off the bands, where the curve crosses from one band to the other. A curve that changes bands must pass through the corridor, since it is continuous, and the critic confines that passage to a narrow interval, whereas squared error places every prediction in the corridor. One run of each student suffices here, since the two differ by a whole unit; the three-seed runs of the second experiment repeat the game with a latent input.

Why does the critic tolerate a crossing at all? A curve on one band and a curve that switches bands both match half of the data at every $c$, and the switch costs only in the narrow interval where it crosses. The optimum of the game, half of the predictions on each band at every $c$, is out of reach for a deterministic student, so the critic has little reason to prefer one curve to the other. Where the curve crosses is left to the run; the loss does not fix it.

Measured by squared error, the adversarial student is the worse of the two: a curve on a band costs about 2.0025 against the corridor's 1.0025. Even a predictor that samples $p(x \mid c)$ exactly has twice the expected squared error of the conditional mean (Exercise :exref:`ex_adversarial-losses-perfect-sampler`). :citet:`Blau.Michaeli.2017` showed that the conflict is general and bounded: forcing the outputs to be distributed like the data raises the smallest achievable squared error, by at most a factor of two. A system that wants realistic outputs must therefore accept a worse squared error, and squared error alone cannot rank such systems. That is worth remembering whenever image restorers are compared by peak signal-to-noise ratio, $10 \log_{10}$ of the squared peak value over the squared error, where a factor of two costs 3 dB.

### Ambiguity and Capacity

The adversarial student does not recover the conditional distribution: a deterministic map gives one answer per input, so following one band leaves the other unrepresented. A larger network would not help. The condition $c$ carries no information about the band, so no deterministic function of $c$, however large, picks the right band more than half the time. The missing ingredient is information. Given a latent bit $z \in \{-1, +1\}$ drawn alongside $c$, a student can send each value of $z$ to one band and so produce both (Exercise :exref:`ex_adversarial-losses-latent-bit`). Generators that must produce varied outputs therefore take a latent input or keep a multi-step sampler, and they need a loss that does not reward piling every sample onto one band, the question of mode coverage that :numref:`sec_gan_relativistic` studies.

The bands show the first of two sources of averaging, and only the second yields to a larger model. In the first, the input does not determine the output: the thumbnail, the two bands, and, below, the phase of a waveform. In the second, the input does determine the output, but the model cannot compute the map. Imagine a one-step network asked to reproduce the image that a many-step sampler of the next chapter makes from each noise vector. Nearby noise vectors can lead the sampler to quite different images, so the map jumps, and a smooth network fit by squared error averages across every jump it cannot resolve. More capacity resolves more of the jumps; against the first source it does nothing.

## Critics in Diffusion Models

The generators of the next chapter, diffusion and flow models (:numref:`chap_diffusion`), are trained without a game, and yet several of the leading ways to make them fast bring a critic back. The two bands show why.

### Denoisers as Conditional Means

Diffusion and flow models produce a sample by evaluating one network many times, often dozens or hundreds :cite:`ho2020denoising,Lipman.Chen.BenHamu.ea.2022`. That network is a *denoiser*. It receives a noisy image $x_\tau$, a clean image $x_0$ with Gaussian noise of standard deviation $\tau$ added, and is trained by squared error to return $x_0$. By :eqref:`eq_beyond_condmean` with $(c, x) = (x_\tau, x_0)$, it learns the conditional mean $\mathbb{E}[x_0 \mid x_\tau]$. At low noise, $x_\tau$ nearly determines $x_0$ and the mean is sharp. At high noise, many clean images are compatible with $x_\tau$, and the mean is their blurred average, the corridor of the two bands. A sampler therefore takes only a small step toward that mean (some samplers then add fresh noise) and asks again at a lower noise level, so no single average becomes the output. :numref:`sec_diffusion-denoising` builds such a denoiser.

*Distillation* removes the many evaluations: a student network learns from the many-step model, its *teacher*, to produce a sample in one evaluation or a few (:numref:`sec_diffusion-few-step`). A one-step student maps a noise vector straight to an image. Two kinds of distillation regress onto the teacher, and they average for different reasons. The first regresses the student's output for a noise vector onto the image that the teacher's deterministic sampler makes from the same vector. The input then determines the target, and any averaging is the capacity effect described above. The second adds noise to the student's own output and regresses it onto the teacher's denoised estimate. That target is a conditional mean already, and at high noise a blurred one, so there no student, however large, escapes the average.

### Distillation on Two Bands
:label:`subsec_gan_beyond_distillation`

On the two bands the second kind of distillation fits in a few lines, because the teacher is a formula. Let us give the teacher the best a trained denoiser could know, the data distribution itself. A denoiser trained on data assumes that its input is a noised data point, $\tilde x = x + \tau \xi$ with $\xi \sim \mathcal{N}(0, 1)$, and it has no other model of its input. Shown a noised student output, $\tilde x = \hat x + \tau \xi$, it reasons in the same way. For a data point on band $b$, $\tilde x$ is normal with mean $c + b$ and variance $\tau^2 + 0.05^2$. The two Gaussian densities differ only in the sign of the cross term in their exponents, so by Bayes' rule the posterior odds of $b = +1$ against $b = -1$ are $\exp\big(2(\tilde x - c)/(\tau^2 + 0.05^2)\big)$. The posterior mean of the band center, $c + P(b = +1 \mid \tilde x, c) - P(b = -1 \mid \tilde x, c)$, is therefore a hyperbolic tangent,

$$
\mathbb{E}[c + b \mid \tilde x, c] \;=\; c + \tanh\Big(\frac{\tilde x - c}{\tau^2 + 0.05^2}\Big).
$$
:eqlabel:`eq_beyond_teacher`

This is the denoiser of the next chapter, reduced to one formula. It is the exact posterior mean of the band center; the posterior mean of the clean point itself differs from it by a relative correction of order $0.05^2/\tau^2$, below a thousandth at $\tau = 2$, the noise level used below. At small $\tau$ the tanh is a step, and the estimate snaps to the band nearer $\tilde x$. At $\tau = 2$ its slope in $\tilde x$ is at most $1/(4 + 0.05^2) \approx \tfrac14$, and the estimate leans only gently toward either band.

Each student now takes a scalar latent $z \sim \mathcal{N}(0, 1)$ besides $c$, so that, unlike the students above, it can put samples on both bands. Student (i) regresses $\hat x$ onto the teacher's estimate at $\tau = 2$, with the gradient stopped so that the estimate is a fixed target, as in the distillation loss of ADD below. Student (ii) plays the log-loss game of the first experiment and never consults the teacher. Student (iii) adds the two losses with equal weight. Network, optimizer, data and the 4,000 steps are shared.

We can predict student (i) before training it. Write $u = g(c, z) - c$ for the offset of one output from the corridor. The regression stops moving $u$ when $u$ equals its average target; the output's noise and the teacher's add up to one Gaussian of variance $\tau^2 + 0.05^2$, so this happens where

$$
u \;=\; \mathbb{E}_{\xi}\Big[\tanh\Big(\frac{u + \sqrt{\tau^2 + 0.05^2}\;\xi}{\tau^2 + 0.05^2}\Big)\Big].
$$

At $\tau = 2$ the right side changes by at most a quarter of any change in $u$, so the two sides can meet only once, and by symmetry they meet at $u = 0$. Every output of student (i), whatever its latent, should land in the corridor. A teacher that returns averages teaches averages.

A low-noise teacher behaves differently. At small $\tau$ the right side is steeper than $u$ near zero, two more solutions appear near $\pm 1$, and the regression holds each output on the band it starts nearest (Exercise :exref:`ex_adversarial-losses-teacher-noise-and-weight`). The function `train_student` trains a student on the critic's loss when it is given a critic, plus the regression weighted by `lam`.

```{.python .input #adversarial-losses-distillation-on-two-bands-1}
%%tab pytorch
def teacher(x_tilde, c, tau):
    """Posterior mean of the band center c + b given x_tilde = x + tau xi."""
    return c + torch.tanh((x_tilde - c) / (tau ** 2 + 0.05 ** 2))

def train_student(g, D, lam, tau=2.0, num_steps=4000, batch_size=128):
    """Train g(c, z) on the critic's loss if D is given, plus lam times the
    squared error toward the teacher's estimate at noise level tau."""
    opt_G = torch.optim.Adam(g.parameters(), lr=1e-3)
    if D is not None:
        opt_D = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)
        z = torch.randn(batch_size, 1)  # the student's latent
        x_hat = g(torch.cat([c, z], 1)) + 0.05 * torch.randn(batch_size, 1)
        real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
        loss_G = 0
        if D is not None:
            opt_D.zero_grad()
            d2l.gan_loss_D(D, real, fake.detach()).backward()
            opt_D.step()
            loss_G = d2l.gan_loss_G(D, real, fake)
        with torch.no_grad():  # the teacher's estimate is a fixed target
            target = teacher(x_hat + tau * torch.randn(batch_size, 1), c, tau)
        loss_G = loss_G + lam * ((x_hat - target) ** 2).mean()
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

```{.python .input #adversarial-losses-distillation-on-two-bands-1}
%%tab jax
def teacher(x_tilde, c, tau):
    """Posterior mean of the band center c + b given x_tilde = x + tau xi."""
    return c + jnp.tanh((x_tilde - c) / (tau ** 2 + 0.05 ** 2))

@nnx.jit(static_argnums=7)
def student_step(g, D, opt_G, opt_D, key, lam, tau, batch_size):
    k_data, k_z, k_xi, k_tau = jax.random.split(key, 4)
    c, x = make_bands(k_data, batch_size)
    z = jax.random.normal(k_z, (batch_size, 1))  # the student's latent
    xi = jax.random.normal(k_xi, (batch_size, 1))  # the data's noise
    sample = lambda g_: g_(jnp.concatenate([c, z], 1)) + 0.05 * xi
    real = jnp.concatenate([c, x], 1)
    fake = lambda g_: jnp.concatenate([c, sample(g_)], 1)
    if D is not None:
        loss_D = lambda D_: d2l.gan_loss_D(D_, real, fake(g))
        opt_D.update(D, nnx.grad(loss_D)(D))
    noise = tau * jax.random.normal(k_tau, (batch_size, 1))
    def loss_G(g_):
        x_hat = sample(g_)  # the teacher's estimate is a fixed target
        target = jax.lax.stop_gradient(teacher(x_hat + noise, c, tau))
        loss = lam * ((x_hat - target) ** 2).mean()
        if D is not None:
            loss += d2l.gan_loss_G(D, real, fake(g_))
        return loss
    opt_G.update(g, nnx.grad(loss_G)(g))

def train_student(g, D, lam, key, tau=2.0, num_steps=4000, batch_size=128):
    """Train g(c, z) on the critic's loss if D is given, plus lam times the
    squared error toward the teacher's estimate at noise level tau."""
    opt_G = nnx.Optimizer(g, optax.adam(1e-3), wrt=nnx.Param)
    opt_D = (None if D is None
             else nnx.Optimizer(D, optax.adam(2e-3), wrt=nnx.Param))
    for k in jax.random.split(key, num_steps):
        student_step(g, D, opt_G, opt_D, k, lam, tau, batch_size)
```

A student is now a distribution of outputs at each $c$. The function `band_shares` evaluates it at the 201 inputs of the first experiment and at 64 evenly spaced quantiles of the latent, and returns the mean distance to the nearer band and the shares of outputs within 0.25 of the upper and of the lower band.

```{.python .input #adversarial-losses-distillation-on-two-bands-2}
%%tab pytorch
def band_shares(g, cs):
    """Mean distance of g(c, z) to the nearer band, and its shares on the
    upper and the lower band, over the inputs cs and 64 quantiles z."""
    zs = torch.special.ndtri((torch.arange(64) + 0.5) / 64)
    c, z = (v.reshape(-1, 1) for v in torch.meshgrid(
        cs.flatten(), zs, indexing='ij'))
    with torch.no_grad():
        u = g(torch.cat([c, z], 1)) - c  # offset from the corridor
    to_band = torch.minimum((u - 1).abs(), (u + 1).abs())
    return (to_band.mean(), ((u - 1).abs() < 0.25).float().mean(),
            ((u + 1).abs() < 0.25).float().mean())
```

```{.python .input #adversarial-losses-distillation-on-two-bands-2}
%%tab jax
def band_shares(g, cs):
    """Mean distance of g(c, z) to the nearer band, and its shares on the
    upper and the lower band, over the inputs cs and 64 quantiles z."""
    zs = jax.scipy.special.ndtri((jnp.arange(64) + 0.5) / 64)
    c, z = (v.reshape(-1, 1) for v in jnp.meshgrid(
        cs.flatten(), zs, indexing='ij'))
    u = g(jnp.concatenate([c, z], 1)) - c  # offset from the corridor
    to_band = jnp.minimum(jnp.abs(u - 1), jnp.abs(u + 1))
    return (to_band.mean(), (jnp.abs(u - 1) < 0.25).mean(),
            (jnp.abs(u + 1) < 0.25).mean())
```

Each student trains from three seeds. Within a seed, the three students start from the same weights and see the same data.

```{.python .input #adversarial-losses-distillation-on-two-bands-3}
%%tab pytorch
students = {'regression': (False, 1.0), 'critic': (True, 0.0),
            'critic + regression': (True, 1.0)}
for name, (has_critic, lam) in students.items():
    for seed in range(3):
        torch.manual_seed(seed)
        g = d2l.make_mlp([2, 32, 32, 1], nn.Tanh())
        D = d2l.make_mlp([2, 32, 32, 1], nn.LeakyReLU(0.2))
        train_student(g, D if has_critic else None, lam)
        dist, upper, lower = band_shares(g, cs)
        print(f'{name}, seed {seed}: {dist:.2f} from the nearer band, '
              f'{upper:.0%} upper, {lower:.0%} lower, '
              f'{1 - upper - lower:.0%} off the bands')
```

```{.python .input #adversarial-losses-distillation-on-two-bands-3}
%%tab jax
students = {'regression': (False, 1.0), 'critic': (True, 0.0),
            'critic + regression': (True, 1.0)}
for name, (has_critic, lam) in students.items():
    for seed in range(3):
        rngs = nnx.Rngs(seed)
        g = d2l.make_mlp([2, 32, 32, 1], nnx.tanh, rngs=rngs)
        D = d2l.make_mlp([2, 32, 32, 1], lambda h: nnx.leaky_relu(h, 0.2),
                         rngs=rngs)
        train_student(g, D if has_critic else None, lam, rngs())
        dist, upper, lower = band_shares(g, cs)
        print(f'{name}, seed {seed}: {dist:.2f} from the nearer band, '
              f'{upper:.0%} upper, {lower:.0%} lower, '
              f'{1 - upper - lower:.0%} off the bands')
```

Student (i) lands where the equation put it. In every run it lies between 0.96 and 0.98 from the nearer band, with every output off the bands: regression onto a high-noise teacher reproduces the squared-error student of the first experiment, latent or not. Students (ii) and (iii) sit 0.03 to 0.08 from a band, with 1 or 2% of their outputs off the bands, a gap of about 0.9 against run-to-run ranges of a few hundredths. Both also split their outputs about evenly, 47 to 52% on each band: given a latent, the critic student does what the deterministic one could not. Between students (ii) and (iii) every difference lies inside the range of the runs, so here the regression term neither helps nor hurts, and the critic decides where the samples land.

That (iii) stays on the bands is the surprising result: its regression term is the one that put student (i) in the corridor, at the same weight. A rough count, with the critic at its best response, explains it. For an output on a band, $u = \pm 1$, the teacher's target lies on average only about 0.2 from the corridor, so the regression pulls the output inward with a gradient of about $2(1 - 0.2) = 1.6$. Suppose the outputs on a band give way toward the corridor by $\delta$. They then form a Gaussian of width 0.05 beside the data's, and the best critic's log ratio gains a slope of $\delta/0.05^2 = 400\,\delta$ against the shift, of which the non-saturating loss passes on about half. The two forces balance at $\delta \approx 1.6/200 = 0.008$, less than a hundredth of a unit and well inside the range of the runs. The bands are so narrow that the critic objects to a shift long before the regression has moved anything. The 1 or 2% of outputs off the bands have the cause seen in the first experiment: a continuous map of $z$ that reaches both bands must pass through the corridor.

### Critics at Scale

Adversarial diffusion distillation (ADD) :cite:`Sauer.Lorenz.Blattmann.ea.2023` builds student (iii) at the scale of text-to-image models. Its one-step student is trained on two losses: a regression onto a frozen teacher's denoised estimate of the student's own output, noised, and an adversarial loss against real images. The critic reads the features of DINOv2, a network pretrained on images without labels, keeps them frozen, and trains only small heads on top. The paper's ablation found the ordering of our three students: the regression alone gave unusable samples, the critic alone good ones, and both together about as good, with the regression term adding agreement between the image and its prompt.

Distribution matching distillation (DMD) :cite:`Yin.Gharbi.Zhang.ea.2024` keeps the comparison but trains no classifier. It borrows one idea from the next chapter: the gradient of a log density with respect to its input, $\nabla_x \log p$, is called the *score*, and a trained diffusion model supplies it for the data at each noise level. Let $q$ be the distribution of the student's samples, and $p_\tau$ and $q_\tau$ the data and the samples noised to level $\tau$. DMD moves each sample $x'$ along the difference of two scores, evaluated at a copy of $x'$ noised to a random level $\tau$. The frozen teacher supplies the score of $p_\tau$, as far as the teacher has learned it, and a second diffusion model, trained alongside the student on its current samples, supplies that of $q_\tau$. The difference is $\nabla_x \log(p_\tau/q_\tau)$, the input gradient of the critic that logistic regression between the two would learn. DMD therefore follows the direction a perfect critic would give, without a classifier.

That direction has a weakness. It descends the reverse divergence $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$, which charges little for a missing mode (:numref:`subsec_gan_mode_dropping`), and the first DMD added a regression onto precomputed teacher outputs partly for that reason. Its successor DMD2 :cite:`Yin.Gharbi.Park.ea.2024` replaces that regression by a critic against real images, which also lets the student correct what the teacher got wrong.

A critic can also improve the slow model itself, and it need not be a second network. Direct discriminative optimization (DDO) :cite:`Zheng.Chen.Chen.ea.2025` takes a pretrained likelihood model $q_\theta$, such as a diffusion model, freezes a reference copy $q_{\textrm{ref}}$, and writes the critic as

$$
D_\theta(x) = \log\big(q_\theta(x)/q_{\textrm{ref}}(x)\big).
$$
:eqlabel:`eq_beyond_ddo`

Training this critic by logistic regression, real data against samples of $q_{\textrm{ref}}$, changes $q_\theta$ itself, with no discriminator network and no second player. If training reaches $q_\theta = p$, then $D_\theta = \log(p/q_{\textrm{ref}})$, the optimal critic between the data and the reference model (Exercise :exref:`ex_adversarial-losses-ddo-plays-log-loss`). The paper also scales this logit by a small factor and repeats the procedure in rounds; :numref:`sec_gan-further-reading` follows both.

## Tokenizers and Vocoders

Two more kinds of network must turn an ambiguous input into an output in one pass, and many of them kept their critics when the generators around them moved to regression.

### Image Autoencoders

Many image generators do not work on pixels. They run on an autoencoder that compresses each image into a shorter code, often a sequence of discrete tokens, hence the name *tokenizer*, and decode the generated code back into an image (:numref:`sec_diffusion-latent` trains one). The code fixes the layout of an image but not the exact texture at each location: the blades of grass of the thumbnail again.

Widely used autoencoders, from VQGAN to the Stable Diffusion autoencoder :cite:`Esser.Rombach.Ommer.2021,Rombach.Blattmann.Lorenz.ea.2022`, therefore train the decoder with three terms. A pixel loss gets the layout right and alone would return a blur. A *perceptual loss* is a squared distance between the features that a fixed pretrained classifier computes from the two images; the most common one, LPIPS, weights the channels of several layers. Its minimizer is any image whose features equal the mean feature over the plausible textures. Features that change little when a texture shifts are nearly the same for every sharp candidate, so a sharp image satisfies the loss. Unfortunately, such features cannot tell a good texture from a bad one either. The third term, a patch critic as in pix2pix, compares each patch with patches of real images and rejects what still looks smeared. A single 2026 preprint, ViTok-v2, reports that the adversarial loss kept its autoencoder from scaling stably, and replaces both the critic and LPIPS by a perceptual loss on the features of a self-supervised network :cite:`Hansen-Estruch.Chen.Ramanujan.ea.2026`; one study does not yet make a rule.

### Speech Vocoders

A text-to-speech system usually predicts a spectrogram, the magnitude of each frequency in each short frame of sound, and leaves the waveform to a *vocoder*. The spectrogram fixes the frequency content and the magnitude of each frame, but not its phase. Model one frame as $x_n = \sin(\omega n + \vartheta)$ for the samples $n = 0, \ldots, N - 1$, with the phase $\vartheta$ uniform on $[-a, a]$. The squared-error prediction, the average over $\vartheta$, is a sine of the same frequency whose amplitude has shrunk by the factor $\sin a / a$ (Exercise :exref:`ex_adversarial-losses-phase`). With the phase entirely unknown, $a = \pi$, the prediction is silence. A loss on the magnitudes of separate frames errs the other way: it accepts every phase, including frames that do not join at their boundaries. HiFi-GAN :cite:`Kong.Kim.Bae.2020` therefore trains its generator with a spectrogram loss, which fixes the content, and with critics that see the waveform itself at several scales and periods, which reject the shrunken frame and the broken boundary.

## Strengths and Limits of GANs
:label:`subsec_gan_beyond_verdict`

This chapter built six games. :numref:`tab_gan_games` collects them by three choices: what the critic estimates, what it does with the estimate, and how it is constrained. The choices are independent, and the recipe adopted from :numref:`sec_gan_convergence` on is one cell of their product, the pairing loss with a penalty on the critic's input gradient.

:The games of this chapter, described by three choices: what the critic estimates, what it does with the estimate, and how the critic is constrained. The last column records whether the generator keeps a gradient when the supports of $p$ and $q$ are disjoint and the critic is at its optimum within its class.
:label:`tab_gan_games`

| objective | the critic estimates | what it does with the estimate | how the critic is constrained | on disjoint supports |
|:--|:--|:--|:--|:--|
| log loss, $f$-GAN (:numref:`sec_basic_gan`, :numref:`sec_gan_objectives`) | $\log(p/q)$, or $f'(p/q)$ for an $f$-GAN | classifies one sample | not at all | saturates |
| hinge (:numref:`sec_gan_objectives`) | the sign of $\log(p/q)$ | classifies one sample with a margin | not at all | saturates |
| MMD (:numref:`sec_gan_objectives`) | a kernel-smoothed $p - q$, the witness | witnesses a difference of means | a fixed kernel ball | a slope up to separations set by the bandwidth |
| Wasserstein-1 (:numref:`sec_gan_objectives`) | a Kantorovich potential | witnesses a difference of means | a Lipschitz ball, enforced in :numref:`subsec_gan_objectives_separated_start` by a gradient penalty | slope one at any distance |
| pairing, RpGAN (:numref:`sec_gan_relativistic`) | $\log(p/q)$ up to a constant | ranks a real--generated pair | not at all | saturates |
| RpGAN + $R_1$ + $R_2$ (:numref:`sec_gan_convergence`) | $\log(p/q)$ above the length scale $2\sqrt\gamma$, smoothed below | ranks a real--generated pair | a penalty on its input gradient at real and generated samples | only through the critic's smoothness |

The last column assumes the best critic of each class on supports that do not touch. With trained critics and supports that overlap only in their tails, as in the separated start of :numref:`subsec_gan_objectives_separated_start`, every generator loss except the saturating one moved a generator six units from the data toward it.

Stand-alone GANs are no longer how images are generated from text, and it is worth being clear about why. On a broad target, every photograph a caption could describe, models trained by regression covered the distribution and improved steadily as data and parameters grew, with little tuning. GANs had to be coaxed into covering every mode and into staying stable as they grew, and on broad targets they lost. What the game still offers is speed: a sample in one network evaluation, sharp even where the output is ambiguous. On narrow domains, faces or game sprites, a GAN remains competitive. That is why a critic survives in decoders, vocoders and few-step students. It also says when to train a GAN today: for a network that must answer in one pass or in a few steps, or for a narrow domain under a real-time budget.

Four limits of the game remain after the repairs of this chapter. The generator is a sampler without a density, so it cannot report how likely an image is. Training needs two networks and a game that converges reliably only under a penalty whose weight must be chosen for the data. No term in the objective pushes toward a missing mode. Dropping half of the data costs the non-saturating generator only about a quarter of a nat, at the critic's best response the gradient on the covered modes vanishes (:numref:`subsec_gan_mode_dropping`), and the guarantee against mode-dropping minima of :numref:`sec_gan_relativistic` holds only for a fixed one-to-one pairing of real and generated samples, which none of our image experiments used. And the game can become unstable at scale. BigGAN's runs collapsed late in training, and the paper reports results from checkpoints taken before the collapse :cite:`Brock.Donahue.Simonyan.2019`; the text-to-image GANs GigaGAN and StyleGAN-T needed new architecture to train stably :cite:`Kang.Zhu.Zhang.ea.2023,Sauer.Karras.Laine.ea.2023`; and ViTok-v2, the tokenizer above, dropped its critic for that reason.

Diffusion and flow models, the subject of :numref:`chap_diffusion`, give up the game. They are trained by regression on a target computed from each training example, and so they inherit the fact this section began with: a network trained by squared error returns an average. How a sampler turns such a network into a generator of draws is the question of the next chapter.

## Summary

A network trained by squared error predicts the conditional mean of its target given its input, and where the input leaves the target ambiguous, that mean averages the possibilities and need not be a valid output; absolute error swaps the mean for a median that is in general no sharper. A critic, whose optimum is the log density ratio, rewards predictions among the data: on two bands the squared-error student runs through the empty corridor, while the adversarial student stays on a band. A squared-error denoiser is such a conditional mean, so a one-step student regressed onto its estimate at high noise also lands in the corridor, and a critic, with or without that regression, puts the student back on both bands, the ordering that ADD found at scale. Image autoencoders and speech vocoders face the same ambiguity in texture and phase and keep a critic for it. Models trained by regression have replaced GANs for broad image synthesis; the game still offers a sharp sample in one network evaluation, against four limits: no density, two networks and a tuned penalty, no push toward a missing mode, and instability at scale.

## Exercises

1. **Distortion of a perfect sampler.** Let $\hat x$ be drawn from
   $p(x \mid c)$, independently of $x$ given $c$, so that the predictor
   samples the conditional distribution exactly
   :cite:`Blau.Michaeli.2017`.
    1. Use :eqref:`eq_beyond_condmean` to show that
       $\mathbb{E}\big[(x - \hat x)^2\big] = 2\, \mathbb{E}\big[(x - \mu(c))^2\big]$.
       How many decibels of peak signal-to-noise ratio,
       $10 \log_{10}\big(x_{\max}^2 / \mathbb{E}[(x - \hat x)^2]\big)$, does the
       sampler lose against the conditional mean?
    1. On the two bands :eqref:`eq_beyond_twoband`, compute the expected
       squared error of the exact sampler and rank it against the
       conditional mean (about 1.0025) and the band $x = c + 1$ (about
       2.0025). Which of the three can an optimal critic of $(x, c)$ pairs
       not distinguish from the data?
    1. A deterministic predictor can match the distribution of $x$ without
       sampling. Let $x \sim \mathcal{N}(0, 1)$ and $c = x + \tau \xi$ with
       $\xi \sim \mathcal{N}(0, 1)$ independent of $x$. Show that
       $\hat x = c / \sqrt{1 + \tau^2}$ has the distribution of $x$, and
       compute the ratio of its squared error to that of the conditional
       mean for $\tau \in \{0.1, 1, 10\}$. When is the extra error small?
   :exlabel:`ex_adversarial-losses-perfect-sampler`
1. [code] **Mean and median.** Absolute error $\mathbb{E}\big[|x - g(c)|\big]$
   is minimized by a conditional median, a value $x_{\textrm{med}}$ with
   $P(x \leq x_{\textrm{med}} \mid c) = \tfrac12$.
    1. Let the offset $b$ of :eqref:`eq_beyond_twoband` take the values
       $-1$, $0$ and $+1$ with probabilities $(\tfrac13, \tfrac13, \tfrac13)$
       or $(\tfrac14, \tfrac14, \tfrac12)$. Find the conditional mean and
       median in each case, and say whether each is a valid output.
    1. Return to two bands with $P(b = +1) = r$, and write the conditional
       mean as a function of $r$. For $r = \tfrac34$, show that the
       conditional median lies on the upper band and compute its offset from
       $c$ to two decimal places. For $r = \tfrac12$, describe the
       absolute-error risk as a function of $g(c) - c$ inside the corridor.
    1. Train the squared-error student of
       :numref:`subsec_gan_beyond_capacity` and a copy trained with absolute
       error on the data of (a) and (b), with three seeds for $r = \tfrac12$,
       and compare with your predictions. When is the conditional median a
       valid output?
   :exlabel:`ex_adversarial-losses-mean-and-median`
1. [code] **A latent bit.** ● Give the student of
   :numref:`subsec_gan_beyond_capacity` a latent input $z \in \{-1, +1\}$,
   uniform and independent of $c$, so that it maps $(c, z)$ to $\hat x$. The
   critic still scores $(x, c)$ pairs.
    1. Train one such student with squared error and one adversarially, and
       find the band each value of $z$ lands on. Which loss gives the student
       an incentive to use $z$?
    1. IMLE :cite:`Li.Malik.2018` matches each data point with its nearest
       generated sample. With a binary latent, its loss is
       $\mathbb{E}\big[\min_{z} (g(c, z) - x)^2\big]$. Train a student with it
       and compare with (a). Which property of the loss forces both latent
       values into use without a critic?
    1. Now let $P(b = +1) = \tfrac34$. Which band frequencies can a student
       with a uniform binary latent represent at a given $c$? Predict, from
       the Jensen--Shannon divergence to the data, the choice of the
       adversarial student, and check by training. What must change in the
       latent input to represent $\tfrac34$?
   :exlabel:`ex_adversarial-losses-latent-bit`
1. [code] **The teacher's noise level and the regression weight.** Use the
   functions `train_student` and `band_shares` of :numref:`subsec_gan_beyond_distillation`.
    1. Evaluate the right side of the fixed-point equation for $u$
       numerically, and find the noise level $\tau$ below which $u = 0$ is
       no longer its only solution.
    1. Train the regression student (i) at $\tau \in \{0.3, 0.7, 2\}$ and
       check your prediction. What decides the band on which an output
       ends?
    1. Train student (iii) at $\tau = 2$ with the regression weighted by
       $\lambda \in \{1, 10, 30, 100\}$. Use the rough count of the text to
       predict how far the outputs give way toward the corridor as a
       function of $\lambda$, and find the weight at which the regression
       wins.
   :exlabel:`ex_adversarial-losses-teacher-noise-and-weight`
1. **Frozen critic features.** Projected GAN and ADD score a sample through
   a frozen pretrained feature map $\phi$ and train only a head $r$, so that
   $D(x) = r(\phi(x))$
   :cite:`Sauer.Chitta.Muller.ea.2021,Sauer.Lorenz.Blattmann.ea.2023`. Let
   $p_\phi$ and $q_\phi$ be the distributions of $\phi(x)$ under $p$ and
   $q$, and let $r$ range over all functions.
    1. Write the log-loss value
       $V(D) = \mathbb{E}_{x \sim p}[\log \sigma(D(x))] + \mathbb{E}_{x \sim q}[\log \sigma(-D(x))]$
       as an expectation over $p_\phi$ and $q_\phi$ and maximize it pointwise
       over $r$. What is the optimal head, which divergence does the maximal
       value measure, and which generators $q$ minimize it?
    1. Let $p = \mathcal{N}(0, I_2)$ on $\mathbb{R}^2$ and $\phi(x) = x_1$.
       Give a minimizer $q \neq p$.
    1. Suppose the generator is scored by the Fréchet distance of
       :numref:`sec_dcgan`, computed with the same features $\phi$. Which
       score does a minimizer from (a) reach, and what does it say about $q$?
       Relate the answer to the finding of
       :citet:`Kynkaanniemi.Karras.Aittala.ea.2023` that ImageNet-pretrained
       critics can lower Inception-based FID.
   :exlabel:`ex_adversarial-losses-frozen-features`
1. **DDO's implicit critic.** DDO :cite:`Zheng.Chen.Chen.ea.2025` plays the
   log-loss game between real data and samples of a frozen model
   $q_{\textrm{ref}}$ with the critic $D_\theta$ of :eqref:`eq_beyond_ddo`.
   It minimizes
   $J(\theta) = -\mathbb{E}_{x \sim p}[\log \sigma(D_\theta(x))] - \mathbb{E}_{x \sim q_{\textrm{ref}}}[\log \sigma(-D_\theta(x))]$
   over the parameters of $q_\theta$, starting from
   $\theta = \theta_{\textrm{ref}}$.
    1. Show in three lines that if training reaches $q_\theta = p$, this
       critic is the optimal critic between $p$ and $q_{\textrm{ref}}$.
    1. Show that at $\theta = \theta_{\textrm{ref}}$ the gradient of $J$ is
       one half of the gradient of the negative log-likelihood
       $-\mathbb{E}_{x \sim p}[\log q_\theta(x)]$. Use
       $\mathbb{E}_{x \sim q_\theta}[\nabla_\theta \log q_\theta(x)] = 0$,
       which follows from differentiating $\int q_\theta(x)\, dx = 1$.
   :exlabel:`ex_adversarial-losses-ddo-plays-log-loss`
1. **Phase and the conditional mean.** Model one frame of a waveform as
   $x_n = \sin(\omega n + \vartheta)$ for $n = 0, \ldots, N - 1$, where the
   spectrogram fixes $\omega$ and the phase $\vartheta$ is uniform on
   $[-a, a]$ with $0 < a \leq \pi$.
    1. Show that the squared-error prediction $\mathbb{E}[x_n]$ has
       $\sin a / a$ times the amplitude of the valid waveforms, and evaluate
       the factor at $a = \pi/2$.
    1. Let $\omega = 2\pi k / N$ for an integer $k$ with $0 < k < N/2$. Show
       that the magnitudes of the discrete Fourier transform of $x$ do not
       depend on $\vartheta$. Which outputs minimize a loss on these
       magnitudes, such as the mel-spectrogram term of HiFi-GAN
       :cite:`Kong.Kim.Bae.2020`?
    1. Concatenate two such frames with independent phases, so that a
       magnitude loss on each frame is zero. What is wrong with the waveform
       at the boundary, and what must the input of a critic contain to
       penalize it?
   :exlabel:`ex_adversarial-losses-phase`

[Discussions](https://d2l.discourse.group/t/36814)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.7]{.kicker}

Adversarial losses beyond GANs<br>
**conditional means · two bands · distillation · tokenizers and vocoders · where GANs stand**
:::
:::

::: {.slide title="Squared Error Turns Many Sharp Answers into One Blur"}
[The problem]{.kicker}

- Enlarge a thumbnail: many sharp photographs shrink to the same thumbnail.
- Squared error rewards the prediction closest to all of them on average,
  their pixelwise mean: a blur that matches none of them.
- Two bands $x = c \pm 1$: the squared-error fit runs between them, about
  **one unit** from every data point.

::: {.d2l-note .warn}
Regression now trains most image and video generators. Why do one-step
students, tokenizers and vocoders still keep a critic?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the conditional-mean decomposition of squared error;
- predict where a squared-error and an adversarial student land on two bands;
- derive a denoising teacher in closed form and predict where regression
  onto it lands;
- explain why autoencoders and vocoders keep a critic;
- say what GANs still offer, and name their four limits.

**The idea:** a pointwise loss averages over ambiguity, a learned comparison
does not. **Running example:** two bands, one input, two answers.
:::

::: {.slide title="The Last Deck Asks Where the Comparison Survives"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Generative Adversarial Networks | what a classifier of samples computes |
| Adversarial Objectives and Divergences | which loss gives which distance |
| Relativistic Objectives | what changes when the critic ranks pairs |
| Gradient Penalties and Convergence | why training circles, and what stops it |
| Adversarial Image Generation | which network, and how to score images |
| Conditional Generation | how to ask for a class |
| **Adversarial Losses Beyond GANs** | **where a learned comparison remains** |
:::

::: {.col .narrow}
- **Uses:** $D^\star = \log(p/q)$; the non-saturating loss; the games of
  the chapter.
- **Skips:** how diffusion models are trained; architectures.
:::
:::
:::

::: {.slide title="Squared Error Grades Distance to the Conditional Mean"}
[Conditional means]{.kicker}

For any predictor $g$ of a sample $x$ from a condition $c$, with
$\mu(c) = \mathbb{E}[x \mid c]$, the cross term vanishes given $c$:

$$\mathbb{E}\big[(x - g(c))^2\big]
= \underbrace{\mathbb{E}\big[(x - \mu(c))^2\big]}_{\textrm{no } g \textrm{ removes it}}
+ \mathbb{E}\big[(\mu(c) - g(c))^2\big]$$

- The best predictor is $g = \mu$; for images, pixel by pixel.
- Several answers: $\mu$ is their average, which need not be an answer.
- Absolute error ($L_1$, pix2pix) gives a median: no sharper.
:::

::: {.slide title="On Two Bands the Mean Lies Where No Data Is"}
[Two bands]{.kicker}

$c \sim U[-1, 1]$, $\;x = c + b + 0.05\,\xi$, $\;b = \pm 1$ with
probability $\tfrac12$ each, independent of $c$.

- Bands $x = c \pm 1$, forty noise widths apart; $\mu(c) = c$, in the corridor.
- Expected squared error of $\mu$: $1 + 0.05^2 \approx 1.0025$.
- Of the band $x = c + 1$: $\tfrac12 \cdot 4 + 0.05^2 \approx 2.0025$.

::: {.d2l-note}
Squared error prefers the corridor, where no data point lies.
:::
:::

::: {.slide title="A Critic Scores Whether a Prediction Lies among the Data"}
[Two bands]{.kicker}

Critic $D(x, c)$, trained by logistic regression on data pairs against
predicted pairs $(\hat x, c)$:

$$D^\star(x, c) = \log \frac{p(x, c)}{q(x, c)}$$

- In the corridor $p \approx 0$: $D^\star$ is strongly negative.
- The student minimizes $-\log \sigma\big(D(\hat x, c)\big)$, the
  non-saturating loss.
- $\hat x = g(c) + 0.05\,\xi'$: the data's noise, so only the location of
  mass differs.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="pytorch"}
[Two bands]{.kicker}

```
c, x = make_bands(batch_size)                   # a fresh batch
x_hat = g(c) + 0.05 * torch.randn(batch_size, 1)
real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
loss_D = softplus(-D(real)) + softplus(D(fake.detach()))
opt_D.zero_grad(); loss_D.mean().backward(); opt_D.step()   # critic
loss_G = softplus(-D(fake)).mean()
opt_G.zero_grad(); loss_G.backward(); opt_G.step()          # student
```

Same 32-unit network for both students, same initial weights;
$\mathrm{softplus}(-t) = -\log \sigma(t)$.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="jax"}
[Two bands]{.kicker}

```
c, x = make_bands(k_data, batch_size)               # a fresh batch
xi = jax.random.normal(k_xi, (batch_size, 1))
real = jnp.concatenate([c, x], 1)
fake = lambda g_: jnp.concatenate([c, g_(c) + 0.05 * xi], 1)
loss_D = lambda D_: (softplus(-D_(real))
                     + softplus(D_(fake(g)))).mean()
opt_D.update(D, nnx.grad(loss_D)(D))
loss_G = lambda g_: softplus(-D(fake(g_))).mean()
opt_G.update(g, nnx.grad(loss_G)(g))
```

Same 32-unit network for both students, same initial weights;
$\mathrm{softplus}(-t) = -\log \sigma(t)$.
:::

::: {.slide title="The Adversarial Student Commits to a Band"}
[Two bands]{.kicker}

@!adversarial-losses-the-capacity-argument-3

Squared error: on average within about 0.02 of $x = c$, one unit from every band.
Adversarial: a few hundredths from a band; at most 2% of inputs off the bands,
where the curve crosses. Where it crosses is left to the run.
:::

::: {.slide title="Check: Which Predictor Fools the Critic?"}
[Check]{.kicker}

Three predictors on the two bands: the mean $\mu(c) = c$, the band
$x = c + 1$, and an exact sampler $\hat x \sim p(x \mid c)$. Rank them by
squared error. Which one can an optimal critic not tell from the data?

. . .

- Mean $\approx 1.0025$ $<$ band $\approx 2.0025$ $<$ sampler $\approx 2.005$.
- The sampler's error is twice the mean's, $10 \log_{10} 2 \approx 3$ dB less
  peak signal-to-noise ratio.
- Only the sampler matches the data; squared error ranks it last. In
  general, realism costs at most a factor of two in squared error.

[Blau and Michaeli 2018]{.small}
:::

::: {.slide title="A Bigger Network Cannot Pick a Band; a Latent Bit Can"}
[Ambiguity and capacity]{.kicker}

- $c$ carries no information about the band: no deterministic $g(c)$, of any
  size, picks the right band more than half the time.
- Add a random latent bit $z = \pm 1$: $g(c, z)$ can send each value to one
  band.
- **Ambiguity:** the input does not fix the output (thumbnail, bands, phase).
- **Capacity:** it does, but one pass cannot compute the map (a one-step
  network imitating a many-step sampler, whose map jumps).

::: {.d2l-note}
Squared error averages over what the model cannot tell apart. More capacity
fixes only the second case.
:::
:::

::: {.slide title="A Squared-Error Denoiser Returns the Mean of the Clean Image"}
[Distillation]{.kicker}

Train $g(x_\tau) \approx x_0$ by squared error, $x_\tau = x_0$ plus noise of
level $\tau$. With $(c, x) = (x_\tau, x_0)$:

$$g^\star(x_\tau) = \mathbb{E}[x_0 \mid x_\tau]$$

- Low noise: $x_\tau$ nearly fixes $x_0$, the mean is sharp.
- High noise: many images fit $x_\tau$, the mean is a blur, the corridor.
- A sampler steps a little toward the mean and asks again: many evaluations.
- **Distillation:** a one-step student learns from this teacher.
:::

::: {.slide title="On Two Bands the Teacher Is a Tanh"}
[Distillation]{.kicker}

A denoiser trained on data reads every input as noised data,
$\tilde x = x + \tau \xi$, including a noised student output
$\hat x + \tau \xi$. Given $b$, data give
$\tilde x \sim \mathcal{N}(c + b, \tau^2 + 0.05^2)$; odds of $b = +1$:
$\exp\big(2(\tilde x - c)/(\tau^2 + 0.05^2)\big)$.

$$\mathbb{E}[c + b \mid \tilde x, c] = c + \tanh\Big(\frac{\tilde x - c}{\tau^2 + 0.05^2}\Big)$$

- The exact posterior mean of the band center.
- Small $\tau$: a step, snaps to the nearer band.
- $\tau = 2$: slope at most $\tfrac14$, leans only gently.
:::

::: {.slide title="A High-Noise Teacher Pulls Every Output into the Corridor"}
[Distillation]{.kicker}

Students now take a latent $z \sim \mathcal{N}(0, 1)$ besides $c$. Student
(i) regresses onto the teacher at $\tau = 2$, gradient stopped. The offset
$u = g(c, z) - c$ settles where

$$u = \mathbb{E}_\xi\Big[\tanh\Big(\frac{u + \sqrt{\tau^2 + 0.05^2}\,\xi}{\tau^2 + 0.05^2}\Big)\Big]$$

- Right side: slope at most $\tfrac14$, so one solution, $u = 0$.
- Prediction: every output in the corridor, whatever the latent $z$.

::: {.d2l-note}
A teacher that returns averages teaches averages.
:::
:::

::: {.slide title="A Critic, with or without the Teacher, Puts the Student on Both Bands"}
[Distillation]{.kicker}

Students $g(c, z)$, $z \sim \mathcal{N}(0, 1)$; 4,000 steps; three seeds each.

| student | from nearer band | on each band | off the bands |
|:--|--:|--:|--:|
| (i) regression onto the teacher | 0.96--0.98 | 0% | 100% |
| (ii) critic | 0.03--0.08 | 47--52% | 1--2% |
| (iii) critic + regression | 0.04--0.06 | 48--51% | 1--2% |

Why (iii) stays: the regression pulls with about $2(1 - 0.2) = 1.6$ at a
band; a shift $\delta$ draws a critic push of about
$\tfrac12 \delta / 0.05^2 = 200\,\delta$; they balance at $\delta \approx 0.008$.

::: {.d2l-note}
(ii) and (iii) differ by less than the run-to-run range: with a latent, the
critic covers both bands.
:::
:::

::: {.slide title="At Scale, ADD Is Student (iii)"}
[Distillation]{.kicker}

- One-step text-to-image student, two losses: regression onto a frozen
  teacher's denoised estimate of its own noised output, and a critic against
  real images.
- The critic reads frozen DINOv2 features and trains only small heads.
- The paper's ablation has our ordering: regression alone unusable, critic
  alone good, both about as good, the regression adding prompt agreement.

[Sauer et al. 2023]{.small}
:::

::: {.slide title="DMD Follows the Optimal Critic's Gradient without Training a Classifier"}
[Distillation]{.kicker}

Score: $\nabla_x \log$ density. $p_\tau$, $q_\tau$: data and student
samples, noised to level $\tau$.

- Teacher supplies $\nabla_x \log p_\tau$; a second diffusion model, trained
  on the student's samples, supplies $\nabla_x \log q_\tau$.
- Move each sample along
  $\nabla_x \log p_\tau - \nabla_x \log q_\tau = \nabla_x \log(p_\tau/q_\tau)$,
  the optimal log-loss critic's input gradient.
- It descends $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$, cheap on missing
  modes; DMD2 adds a critic on real images.

[Yin et al. 2024, DMD and DMD2]{.small}
:::

::: {.slide title="A Likelihood Model Can Be Its Own Critic"}
[Critics at scale]{.kicker}

Two evaluable densities define a log-ratio critic:

$$D_\theta(x) = \log \frac{q_\theta(x)}{q_{\textrm{ref}}(x)}$$

- DDO trains this critic by logistic regression, real data against samples
  of a frozen copy $q_{\textrm{ref}}$, and so updates $q_\theta$ itself.
- At $q_\theta = p$ it is the optimal critic $\log(p/q_{\textrm{ref}})$.
- No discriminator network, no second player.

[Zheng et al. 2025]{.small}
:::

::: {.slide title="Texture Is Ambiguous, So Autoencoders Keep a Patch Critic"}
[Tokenizers and vocoders]{.kicker}

The code fixes the layout, not the texture. The decoder trains on three terms:

| decoder loss | what it does |
|:--|:--|
| pixel | layout right; alone a blur |
| perceptual (LPIPS: features of a pretrained classifier) | matches the mean feature; a sharp image fits, a bad texture too |
| patch critic | rejects what still looks smeared |

[Esser et al. 2021; Rombach et al. 2022]{.small}
:::

::: {.slide title="Check: What Does a Squared-Error Vocoder Output?"}
[Check]{.kicker}

A vocoder turns a spectrogram into a waveform. Model a frame as
$x_n = \sin(\omega n + \vartheta)$: the spectrogram fixes $\omega$, not the
phase $\vartheta$. With $\vartheta$ uniform on $[-\pi, \pi]$, what does the
squared-error prediction sound like? And for $[-\pi/2, \pi/2]$?

. . .

- $\mathbb{E}_\vartheta[x_n] = \frac{\sin a}{a}\sin(\omega n)$ for $\vartheta$ uniform on $[-a, a]$.
- $a = \pi$: silence. $a = \pi/2$: $2/\pi \approx 0.64$ of the amplitude.
- HiFi-GAN: a spectrogram loss for the content, critics on the waveform
  against the shrunken frame and broken frame boundaries.

[Kong et al. 2020]{.small}
:::

::: {.slide title="Three Independent Choices Describe Every Game of the Chapter"}
[The games]{.kicker}

| objective | critic estimates | does | constrained by | disjoint supports |
|:--|:--|:--|:--|:--|
| log loss | $\log(p/q)$ | classifies | nothing | saturates |
| hinge | sign of $\log(p/q)$ | classifies | nothing | saturates |
| MMD | smoothed $p - q$ | witnesses | kernel ball | slope, up to bandwidth |
| $W_1$ | potential | witnesses | Lipschitz ball | slope one |
| RpGAN | $\log(p/q)$ + const | ranks | nothing | saturates |
| R3GAN | $\log(p/q)$ above $2\sqrt\gamma$ | ranks | input-gradient penalty | only via critic smoothness |

[Potential, witness: the critics of the $W_1$ and MMD games; $\gamma$: weight
of the $R_1$ and $R_2$ penalties. Last column: best critic of its class.]{.small}
:::

::: {.slide title="GANs Lost Broad Image Synthesis but Keep One-Step Sampling"}
[Where GANs stand]{.kicker}

::: {.cols}
::: {.col}
**Why regression won broad targets**

- covered the distribution
- improved steadily with data and size
- little tuning
:::

::: {.col}
**What the game still offers**

- a sample in one network evaluation
- sharp where the output is ambiguous
- competitive on narrow domains
:::
:::

::: {.d2l-note}
Train one today for a one-pass decoder or vocoder, a one- to four-step
student, or a narrow domain under a real-time budget.
:::
:::

::: {.slide title="Four Limits of the Game Remain after the Repairs"}
[Where GANs stand]{.kicker}

1. **No density:** the generator cannot say how likely an image is.
2. **Two networks and a tuned penalty:** convergence needs a penalty weight
   chosen for the data.
3. **No push toward a missing mode:** a dropped half costs about a quarter
   of a nat; the pairing guarantee holds only for a fixed coupling, which
   none of our image runs used.
4. **Instability at scale:** BigGAN runs collapse late, results come from
   checkpoints before it; GigaGAN and StyleGAN-T needed new architecture.

[Brock et al. 2019; Kang et al. 2023; Sauer et al. 2023]{.small}
:::

::: {.slide title="Check: Which of These Would You Train with a Critic Today?"}
[Check]{.kicker}

A text-to-image model from scratch; a vocoder; a one-step student of a
diffusion model; a sprite generator for a game that must run at 60 frames
per second.

. . .

- Text-to-image from scratch: regression; broad targets favor it.
- Vocoder, one-step student: a critic, beside a regression term.
- Sprite generator: a GAN is a good fit; narrow domain, one evaluation.
:::

::: {.slide title="What Remains after the Repairs"}
[Wrap-up]{.kicker}

- Two bands: squared error within about 0.02 of $x = c$, a unit from the
  data; the critic keeps the student on a band.
- Teacher at $\tau = 2$: regression alone 0.96--0.98 from a band; with a
  critic 0.03--0.08, about half on each band, regression term or not.
- Autoencoders and vocoders keep a critic for texture and phase.
- GANs keep one-step sampling; four limits remain: no density, two
  networks and a tuned penalty, no push toward a missing mode, instability
  at scale.

**Next:** diffusion and flow models give up the game and train by regression
on a target computed from each training example.
:::
