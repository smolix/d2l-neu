# Adversarial Losses Beyond GANs
:label:`sec_gan_beyond`

Almost every regressor in this book was trained with squared error, and for house prices or class probabilities that was the right choice. Ask a network to enlarge a thumbnail, though, and squared error fails. Many sharp photographs shrink to the same thumbnail: the same face with its hair falling slightly differently, the same lawn with its blades of grass in other places. Squared error rewards the prediction that is closest to all of them on average, which is their pixelwise mean, and that mean is a blur that matches none of them. The critic of this chapter judges a prediction differently. It does not measure the distance to an answer; it asks whether the prediction looks like a sample from the data. This section asks why squared error averages, and why a learned comparison does not.

The question has outlived stand-alone GANs. Most image, video and audio generators are now trained by regression against targets computed in closed form, without a two-player game. Yet an adversarial loss still sits inside the one-step image generators distilled from them, inside the autoencoders on which many of them run, and inside many speech vocoders. We first work out the averaging on a problem with one input and two answers, then follow the same argument into these systems. We end by separating the adversarial objective from the game used to optimize it, and by listing what the repairs of this chapter leave unsolved.

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## Conditional Means and Ambiguity

Many generation problems are conditional predictions. A super-resolution network receives a low-resolution image $x$ and must produce a high-resolution image $y$. A vocoder receives a spectrogram and must produce a waveform. A one-step student receives a noise vector and must produce, in a single forward pass, the image its teacher builds over many network evaluations. Before comparing losses for such problems, we ask what the usual loss rewards.

### The Conditional-Mean Decomposition

Write $g$ for the predictor and $\mu(x) = \mathbb{E}[y \mid x]$ for the conditional mean. Adding and subtracting $\mu(x)$ inside the square produces a cross term $2\,\mathbb{E}\big[(y - \mu(x))(\mu(x) - g(x))\big]$, which vanishes: conditionally on $x$ the second factor is fixed and the first has mean zero. What remains is

$$
\mathbb{E}\big[(y - g(x))^2\big]
\;=\;
\mathbb{E}\big[(y - \mu(x))^2\big] \;+\; \mathbb{E}\big[(\mu(x) - g(x))^2\big].
$$
:eqlabel:`eq_beyond_condmean`

The first term does not depend on $g$; it is the spread of $y$ around its mean, which no predictor can remove. The second is the squared distance between $g(x)$ and $\mu(x)$. Squared error therefore grades a predictor by its distance to the conditional mean, and the best predictor is $g = \mu$. When $p(y \mid x)$ concentrates on one answer, the mean is that answer. When it spreads over several distinct answers, the mean averages them and need not be a valid answer itself: the mean of many sharp images that share a thumbnail is not sharp.

An adversarial loss grades a prediction by a learned comparison instead. A critic $D(x, y)$ is trained by logistic regression to tell data pairs $(x, y) \sim p$ from pairs $(x, \hat y) \sim q$ formed with the predictor's output. As :numref:`sec_basic_gan` showed, the best such critic is the log density ratio

$$
D^\star(x, y) = \log \frac{p(x, y)}{q(x, y)}.
$$

It is large where data pairs are common relative to predicted ones, and strongly negative where predictions land with no data near them. The predictor is trained to raise the critic's score of its own outputs by minimizing $-\log \sigma\big(D(x, \hat y)\big)$, the non-saturating loss of :numref:`sec_basic_gan`. A prediction in an empty region between valid answers scores low, however close it is on average to all of them. The two losses therefore agree when $p(y \mid x)$ has one answer the predictor can reach, and they disagree when it has several: squared error favors their mean, and the critic favors a prediction that lies among the data.

### Two Bands
:label:`subsec_gan_beyond_capacity`

The smallest problem that separates the two losses has one input, one output and two answers. Draw $x$ uniformly from $[-1, 1]$ and set

$$
y \;=\; x + b + 0.05\,\epsilon,
\qquad
b \in \{-1, +1\} \textrm{ with probability } \tfrac12 \textrm{ each},
\quad \epsilon \sim \mathcal{N}(0, 1),
$$
:eqlabel:`eq_beyond_twoband`

with the offset $b$ drawn independently of $x$. The data form two parallel bands, $y = x \pm 1$, two units apart and each with noise of width 0.05, so forty noise widths separate them. The conditional mean is $\mu(x) = x$, a line through the empty corridor between the bands. Its expected squared error is $1 + 0.05^2 \approx 1.0025$, the first term of :eqref:`eq_beyond_condmean`. A predictor that follows the band $y = x + 1$ is right for half of the data and two units off for the other half, so its expected squared error is $\tfrac12 \cdot 4 + 0.05^2 \approx 2.0025$. Squared error prefers the corridor, where no data point lies.

Both students in the experiment are the same deterministic network, a perceptron with two hidden layers of 32 tanh units that maps $x$ to one prediction $\hat y = g(x)$, and both start from the same weights. The critic is a perceptron of the same size with leaky ReLU units that scores pairs $(x, y)$. Both tabs initialize the networks alike, with PyTorch's default weights. Because the data distribution is known, every training step can draw a fresh batch from it with `make_bands`.

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab pytorch
def make_bands(n):
    """Draw n pairs (x, y) from the two bands y = x + b + 0.05 eps."""
    x = torch.rand(n, 1) * 2 - 1
    b = torch.randint(0, 2, (n, 1)) * 2.0 - 1.0
    return x, x + b + 0.05 * torch.randn(n, 1)

def mlp(sizes, act):
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(d_in, d_out), act()]
    return nn.Sequential(*layers[:-1])

torch.manual_seed(0)
g_mse, g_adv = mlp([1, 32, 32, 1], nn.Tanh), mlp([1, 32, 32, 1], nn.Tanh)
g_adv.load_state_dict(g_mse.state_dict())  # both students start alike
D = mlp([2, 32, 32, 1], lambda: nn.LeakyReLU(0.2))
```

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab jax
def make_bands(key, n):
    """Draw n pairs (x, y) from the two bands y = x + b + 0.05 eps."""
    k_x, k_b, k_eps = jax.random.split(key, 3)
    x = jax.random.uniform(k_x, (n, 1), minval=-1, maxval=1)
    b = jax.random.rademacher(k_b, (n, 1), dtype=jnp.float32)
    return x, x + b + 0.05 * jax.random.normal(k_eps, (n, 1))

def mlp(sizes, act, rngs):    # PyTorch's default initialization
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        bound = d_in ** -0.5
        init = lambda key, shape, dtype: jax.random.uniform(
            key, shape, dtype, -bound, bound)
        layers += [nnx.Linear(d_in, d_out, kernel_init=init,
                              bias_init=init, rngs=rngs), act]
    return nnx.Sequential(*layers[:-1])

rngs = nnx.Rngs(0)
g_mse = mlp([1, 32, 32, 1], nnx.tanh, rngs)
g_adv = nnx.clone(g_mse)  # both students start alike
D = mlp([2, 32, 32, 1], lambda h: nnx.leaky_relu(h, 0.2), rngs)
key_mse, key_adv, key_data = jax.random.split(jax.random.key(0), 3)
```

The first student minimizes squared error with Adam. Its learning rate decays along a cosine from 0.01 to zero over 2,000 steps, so the last steps average out the noise of the minibatches and the curve settles where :eqref:`eq_beyond_condmean` says it should.

```{.python .input #adversarial-losses-two-bands-1}
%%tab pytorch
def train_mse(g, num_steps=2000, batch_size=128, lr=0.01):
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)
    for _ in range(num_steps):
        x, y = make_bands(batch_size)  # a fresh batch at every step
        opt.zero_grad()
        ((g(x) - y) ** 2).mean().backward()
        opt.step()
        sched.step()

train_mse(g_mse)
```

```{.python .input #adversarial-losses-two-bands-1}
%%tab jax
@nnx.jit(static_argnums=3)
def mse_step(g, opt, key, batch_size):
    x, y = make_bands(key, batch_size)  # a fresh batch at every step
    opt.update(g, nnx.grad(lambda g_: ((g_(x) - y) ** 2).mean())(g))

def train_mse(g, key, num_steps=2000, batch_size=128, lr=0.01):
    schedule = optax.cosine_decay_schedule(lr, num_steps)
    opt = nnx.Optimizer(g, optax.adam(schedule), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        mse_step(g, opt, k, batch_size)

train_mse(g_mse, key_mse)
```

The second student is the generator of the log-loss game. At each step the critic takes one logistic-regression step on real pairs $(x, y)$ against predicted pairs $(x, \hat y)$, and then the student takes one step on $-\log \sigma\big(D(x, \hat y)\big)$ against the updated critic. The prediction enters the game with the same observation noise as the data, $\hat y = g(x) + 0.05\,\xi$ with $\xi \sim \mathcal{N}(0, 1)$, so the critic compares two distributions that differ in where they put their mass, not in how smooth they are.

```{.python .input #adversarial-losses-two-bands-2}
%%tab pytorch
softplus = nn.functional.softplus

def train_adversarial(g, D, num_steps=8000, batch_size=128):
    opt_G = torch.optim.Adam(g.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(num_steps):
        x, y = make_bands(batch_size)
        y_hat = g(x) + 0.05 * torch.randn(batch_size, 1)  # the data's noise
        real, fake = torch.cat([x, y], 1), torch.cat([x, y_hat], 1)
        opt_D.zero_grad()
        (softplus(-D(real)) + softplus(D(fake.detach()))).mean().backward()
        opt_D.step()
        opt_G.zero_grad()
        softplus(-D(fake)).mean().backward()
        opt_G.step()

train_adversarial(g_adv, D)
```

```{.python .input #adversarial-losses-two-bands-2}
%%tab jax
softplus = jax.nn.softplus

@nnx.jit(static_argnums=5)
def adversarial_step(g, D, opt_G, opt_D, key, batch_size):
    k_data, k_xi = jax.random.split(key)
    x, y = make_bands(k_data, batch_size)
    xi = jax.random.normal(k_xi, (batch_size, 1))  # the data's noise
    real = jnp.concatenate([x, y], 1)
    fake = lambda g_: jnp.concatenate([x, g_(x) + 0.05 * xi], 1)
    loss_D = lambda D_: (softplus(-D_(real)) + softplus(D_(fake(g)))).mean()
    opt_D.update(D, nnx.grad(loss_D)(D))
    loss_G = lambda g_: softplus(-D(fake(g_))).mean()
    opt_G.update(g, nnx.grad(loss_G)(g))

def train_adversarial(g, D, key, num_steps=8000, batch_size=128):
    opt_G = nnx.Optimizer(g, optax.adam(1e-3), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(2e-3), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        adversarial_step(g, D, opt_G, opt_D, k, batch_size)

train_adversarial(g_adv, D, key_adv)
```

Three numbers summarize each student on a grid of 201 inputs: the mean distance from its curve to the nearer band, the mean distance to the conditional mean $y = x$, and the fraction of inputs at which the curve is farther than 0.25 from both bands.

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab pytorch
xs = torch.linspace(-1, 1, 201).reshape(-1, 1)
with torch.no_grad():
    curves = {'squared error': g_mse(xs), 'adversarial': g_adv(xs)}
for name, y_hat in curves.items():
    to_band = torch.minimum((y_hat - xs - 1).abs(), (y_hat - xs + 1).abs())
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{(y_hat - xs).abs().mean():.2f} from y = x, '
          f'{(to_band > 0.25).float().mean():.0%} off the bands')
x_data, y_data = make_bands(200)
```

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab jax
xs = jnp.linspace(-1, 1, 201).reshape(-1, 1)
curves = {'squared error': g_mse(xs), 'adversarial': g_adv(xs)}
for name, y_hat in curves.items():
    to_band = jnp.minimum(jnp.abs(y_hat - xs - 1), jnp.abs(y_hat - xs + 1))
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{jnp.abs(y_hat - xs).mean():.2f} from y = x, '
          f'{(to_band > 0.25).mean():.0%} off the bands')
x_data, y_data = make_bands(key_data, 200)
```

The plot draws both curves over a sample of the data, with the conditional mean dashed.

```{.python .input #adversarial-losses-the-capacity-argument-3}
%%tab pytorch, jax
d2l.plot([x_data, xs, xs, xs],
         [y_data, curves['squared error'], curves['adversarial'], xs],
         '$x$', '$y$', ['data', 'squared-error student',
                        'adversarial student', '$y = x$'],
         ylim=(-2.2, 3.6), fmts=('C0.', 'C2-', 'C1-', 'k--'),
         figsize=(5.5, 3.2))
```

The squared-error student lies within about 0.02 of the dashed line $y = x$, as :eqref:`eq_beyond_condmean` predicts, and therefore about one unit from every band: everywhere it predicts a value that never occurs in the data. The adversarial student lies on a band. Its mean distance to the nearer band is a few hundredths, and it leaves the bands only at zero, one or two narrow crossings. Other random seeds change which band the curve follows and where it crosses, not these numbers.

A curve may follow one band across the whole range, or follow different bands on separate intervals and cross between them. Because $b$ is independent of $x$, such switching introduces a dependence between $x$ and the band that the data do not have. Any continuous function that changes bands must pass through the corridor, but the adversarial loss confines the passage to a narrow interval, while squared error places every prediction in the corridor.

### Ambiguity and Capacity

The adversarial student does not recover the conditional distribution. A deterministic student produces one answer per input, so following one band leaves the other unrepresented. A larger network would not help: the input $x$ carries no information about the band, so no deterministic function of $x$ alone, however large, picks the right band more than half the time, and under squared error none does better than $\mu$. The missing ingredient is information, not size. Given a latent bit $u \in \{-1, +1\}$, drawn at random alongside $x$, a student can send one value of $u$ to each band and so produce both (Exercise :exref:`ex_adversarial-losses-latent-bit`). Generators that must produce varied outputs therefore take a latent input, or keep a multi-step sampler. They also need a loss that does not reward piling every sample onto one band, which is the question of mode coverage that :numref:`sec_gan_relativistic` studies.

Averaging has two sources, and the distinction decides whether a larger model helps. In the first, the input does not determine the output: the thumbnail, the two bands, and, below, the phase of a waveform. In the second, the input does determine the output, but the model cannot compute the map in one pass. A one-step student of a many-step sampler is the example: its noise vector fixes the teacher's image, but a student that cannot reproduce the teacher's long computation sees several inputs as alike. In both cases the best squared-error fit averages over the outputs that the model cannot tell apart. More capacity removes only the second source.

Removing the average increases the squared error. A predictor that samples $p(y \mid x)$ exactly makes an expected error twice that of the conditional mean, a loss of 3 dB in peak signal-to-noise ratio (Exercise :exref:`ex_adversarial-losses-perfect-sampler`). :citet:`Blau.Michaeli.2017` showed that this conflict is general: low distortion and outputs distributed like the data cannot both be reached, so a realistic output scores worse by squared error than the blur does. A sharp prediction should therefore be judged by whether it looks like data, not by its pointwise error.

At image scale the two bands become the many textures compatible with one prompt or one low-resolution input. The condition fixes much of the global layout, so the ambiguity sits in fine detail, where the averaging of :eqref:`eq_beyond_condmean` is strongest. The SDXL-Lightning report finds that FID on whole images differs little across recent few-step text-to-image distillation methods, whereas FID computed on image patches, which is more sensitive to local texture, shows larger differences :cite:`Lin.Wang.Yang.2024`. Local texture is where the adversarial term is expected to help.

## Adversarial Terms in Current Systems

The argument so far predicts where an adversarial term should survive once generators are trained by regression: wherever the output is ambiguous given what the model sees, and the model must answer in one pass. We follow that prediction into three kinds of systems.

### Denoisers as Conditional Means

Diffusion and flow models generate a sample by repeatedly evaluating a learned denoiser, often dozens or hundreds of times :cite:`ho2020denoising,Lipman.Chen.BenHamu.ea.2022`. The denoiser receives a noisy image $x_t$, a clean image $x_0$ with noise added at level $t$, and is trained by squared error to predict $x_0$. By :eqref:`eq_beyond_condmean` with $(x, y) = (x_t, x_0)$, it learns the conditional mean $\mathbb{E}[x_0 \mid x_t]$. At low noise, $x_t$ nearly determines $x_0$ and the mean is sharp. At high noise, many clean images are compatible with $x_t$, and the mean is their blurred average, the corridor of the two bands. A many-step sampler takes only a small step toward that mean, adds fresh noise, and asks again at a lower noise level, so no single average becomes the output. A one-step student has no later steps. If it is trained by squared error toward the teacher's denoised estimate at high noise, it inherits the average.

### One-Step Distillation

Distillation trains a student to approximate the teacher's distribution in one to four network evaluations. Adversarial diffusion distillation (ADD) trains a one-step student with two losses :cite:`Sauer.Lorenz.Blattmann.ea.2023`. The first is a regression toward a frozen teacher, which denoises a noised copy of the student's output. The second is an adversarial loss against real images, with a critic built from frozen DINOv2 features and small trainable heads. In the reported one-step ablation, with all other settings fixed, the regression alone gives FID 315.6, the adversarial loss alone 20.8, and both together 20.6. The previous subsection predicts the first number: the regression target is a conditional mean, so a student fitted to it produces averages. The adversarial term supplies nearly all of the sample quality; the teacher's guidance helps, but it does not by itself give a usable one-step sampler.

Distribution matching distillation (DMD) shows how close a regression-trained model can come to the critic :cite:`Yin.Gharbi.Zhang.ea.2024,Yin.Gharbi.Park.ea.2024`. Its central update is not a regression on teacher outputs. It moves each generated sample $x'$ along the difference of two scores, $\nabla_x \log p(x') - \nabla_x \log q(x')$, where $q$ is now the student's own distribution. Two diffusion models supply the scores, evaluated on noised copies of $x'$ at several noise levels: the frozen teacher estimates the score of the data, and a second model, trained online on the student's samples, estimates the score of $q$. The difference is $\nabla_x \log(p/q) = \nabla_x D^\star$, the input gradient of the optimal log-loss critic. DMD therefore follows the direction a perfect critic would give, but estimates it with two score models trained by regression instead of a classifier. The second score model still trains alongside the student, so the loop remains coupled, but neither network maximizes what the other minimizes.

### Tokenizers and Vocoders

Many image generators run on an autoencoder that compresses each image into a shorter code, and many widely used autoencoders include an adversarial loss. VQGAN combined reconstruction and perceptual losses with a critic that scores image patches :cite:`Esser.Rombach.Ommer.2021`, and the Stable Diffusion autoencoder kept this combination because the patch critic reduces the blur of pixelwise losses :cite:`Rombach.Blattmann.Lorenz.ea.2022`. This is ambiguity at the scale of a patch: the code does not determine the exact texture at every location, so squared error averages over the plausible textures. The ViTok study shows both sides of this trade-off. Fine-tuning its decoder adversarially lowered reconstruction FID to 0.50 at some cost in PSNR and SSIM, while the FID of images generated in the latent space improved only from 5.5 to 4.9 :cite:`Hansen-Estruch.Yan.Chung.ea.2025`. Its successor, ViTok-v2, scales the autoencoder to five billion parameters and replaces both the adversarial and the LPIPS loss with a fixed perceptual loss on self-supervised DINOv3 features, arguing that reliance on adversarial losses prevents stable scaling :cite:`Hansen-Estruch.Chen.Ramanujan.ea.2026`. That is the instability of the game analyzed in :numref:`sec_gan_convergence`, and a fixed feature loss avoided it.

Speech vocoders face the same ambiguity in phase. A spectrogram fixes the frequency content and the magnitude of each frame but not the phase. If a frame is $\sin(\omega t + \varphi)$ with the phase $\varphi$ uniform on $[-a, a]$, the squared-error prediction is the average over $\varphi$: a sine of the same frequency whose amplitude has shrunk by the factor $\sin a / a$ (Exercise :exref:`ex_adversarial-losses-phase`). When the phase is entirely unknown, $a = \pi$, the factor is zero and the prediction is silence. A critic that sees the waveform rejects the shrunken frame. The discriminator ensembles of HiFi-GAN :cite:`Kong.Kim.Bae.2020` therefore continue in the BigVGAN family :cite:`Lee.Ping.Ginsburg.ea.2023`, adversarial training remains common in deployed low-latency vocoders, and RAF trains such vocoders with the paired loss of :numref:`sec_gan_relativistic` :cite:`Lee.Choi.2026`. Whether fixed perceptual features can replace the critic for audio, as they did for the ViTok-v2 tokenizer, is open.

## Responses to Instability

Each application above inherits the training problems of the earlier sections, and ViTok-v2 shows that they can block scaling. Methods respond in one of three ways, compared in :numref:`fig_gan_exits`. The first regularizes the dynamics of the game: zero-centered penalties on the critic's input gradient restore local convergence :cite:`Mescheder.Geiger.Nowozin.2018`, as :numref:`sec_gan_convergence` derives, and R3GAN combines them with the paired loss :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. The second constrains the critic: spectral normalization bounds its Lipschitz constant :cite:`Miyato.Kataoka.Koyama.ea.2018`, frozen pretrained features keep it from co-adapting with the generator, as in ADD, and the same augmentation of real and generated images keeps it from memorizing a small training set :cite:`Zhao.Liu.Lin.ea.2020,Karras.Aittala.Hellsten.ea.2020`. The third removes the game. Fixed discrepancies such as MMD were repeatedly supplemented by a learned kernel or critic, because fixed comparisons lose sensitivity in high dimension; diffusion and flow models instead regress a network on a closed-form target with a unique minimizer, so training needs no equilibrium between two networks.

![Three responses to unstable adversarial training, corresponding to different sources of instability: regularize the dynamics of the game, constrain the critic, or replace the game with a fixed discrepancy or regression objective. The first retains a minimax loop and a penalty weight; the second inherits the biases of its fixed features; the third may use discrepancies that weaken in high dimensions or require multi-step sampling. Adversarial losses can reappear when non-adversarial models are distilled to a few steps.](../img/mdl-gan-exits.svg)
:label:`fig_gan_exits`

## The Objective without the Game

DMD follows the critic's gradient without training a classifier. The same separation of the objective from the game can be carried one step further, to a model that represents the ratio itself. From here on $x$ is again a sample, not a conditioning input.

### Likelihood Models as Critics

The optimal log-loss critic between $p$ and $q$ is the log density ratio $\log(p/q)$. Standard adversarial training estimates it with a separate network because the generator's density $q$ cannot be evaluated. A likelihood model can evaluate its density, and so it can compute the ratio between two versions of itself. Direct discriminative optimization (DDO) applies this to a pretrained likelihood model $q_\theta$, such as a diffusion model :cite:`Zheng.Chen.Chen.ea.2025`. It compares real data with samples from a frozen reference copy $q_{\textrm{ref}}$ and parameterizes the critic as

$$
D_\theta(x) = \log\big(q_\theta(x)/q_{\textrm{ref}}(x)\big).
$$
:eqlabel:`eq_beyond_ddo`

Training this critic by logistic regression on real against reference samples updates $q_\theta$ directly, with no separate discriminator and no second player. If training reaches $q_\theta = p$, then $D_\theta = \log(p/q_{\textrm{ref}})$, which is the optimal critic between the data and the reference model (Exercise :exref:`ex_adversarial-losses-ddo-plays-log-loss`). The parameterization can therefore represent the exact solution of the game it plays.

This construction separates the adversarial objective from its usual optimization. The objective scores samples through an estimate of $\log(p/q)$ and follows its gradient; the usual optimization estimates that ratio with a second network trained at the same time, and the convergence failures of :numref:`sec_gan_convergence` come from this second part. The systems of this section divide the two in different ways. Vocoders keep the two-player game. ADD keeps the game but freezes the critic's features. ViTok-v2 replaces the critic with fixed features. DMD estimates the gradient of the log ratio with two score models, and DDO represents the ratio itself with the likelihood model.

### What Remains

The chapter began with a game whose generator could lose its gradient, whose training could circle without converging, and whose samples could collapse onto a few modes. The repairs work. The non-saturating and paired losses keep a gradient, zero-centered penalties make gradient training converge (:numref:`sec_gan_convergence`), and with both, a plain convolutional generator learns to draw images (:numref:`sec_dcgan`). Four limits remain. The generator is a sampler without a density, so it cannot report how likely an image is. Training still needs two networks and a game that converges reliably only under a penalty whose weight must be chosen for the data. No likelihood term penalizes a missing mode; the evidence for coverage is the paired loss landscape and the modes counted on StackedMNIST. And the game can still become unstable at scale, as ViTok-v2 reported. Diffusion and flow models, the subject of :numref:`chap_diffusion`, avoid the game: they are trained by regression against closed-form targets.

## Summary

Squared error measures a prediction's distance to the conditional mean, so where the input leaves the output ambiguous, or the model cannot compute the map in one pass, a regression-trained model predicts an average that need not be a valid output. A critic, whose optimum is the log density ratio, instead rewards predictions that lie among the data. On two bands the squared-error student runs through the empty corridor at $y = x$, while the adversarial student stays on a band, although as a deterministic map it covers only one. A squared-error denoiser is a conditional mean, which is why one-step distillation, autoencoders with patch critics and vocoders, which face ambiguous phase, keep an adversarial term or a stand-in for one. Instability is met by regularizing the dynamics, constraining the critic, or removing the game. The adversarial objective needs an estimate of $\log(p/q)$, not a two-player game: DMD estimates its gradient with two score models, and DDO estimates the ratio itself with a likelihood model.

## Exercises

1. **Distortion of a perfect sampler.** Let $\hat y$ be drawn from
   $p(y \mid x)$, independently of $y$ given $x$, so that the predictor
   samples the conditional distribution exactly. This problem follows the
   analysis of the trade-off between distortion and realism by
   :citet:`Blau.Michaeli.2017`.
    1. Use :eqref:`eq_beyond_condmean` to show that
       $\mathbb{E}\big[(y - \hat y)^2\big] = 2\, \mathbb{E}\big[(y - \mu(x))^2\big]$. For
       outputs with peak value $y_{\max}$, the peak signal-to-noise ratio of
       a predictor $\hat y$ is
       $10 \log_{10}\big(y_{\max}^2 / \mathbb{E}[(y - \hat y)^2]\big)$ decibels. By
       how many decibels does the peak signal-to-noise ratio of such a
       sampler fall below that of the conditional mean?
    1. For the two-band data :eqref:`eq_beyond_twoband`, the text found the
       expected squared error of the conditional mean, about 1.0025, and of
       the band $y = x + 1$, about 2.0025. Compute it for the exact sampler
       and rank the three predictors by squared error. Which of them can an
       optimal critic of $(x, y)$ pairs not distinguish from the data?
    1. A predictor can match the distribution of $y$ without sampling. Let
       $y \sim \mathcal{N}(0, 1)$ and $x = y + \tau \epsilon$ with
       $\epsilon \sim \mathcal{N}(0, 1)$ independent of $y$. Show that the
       deterministic predictor $\hat y = x / \sqrt{1 + \tau^2}$ has the same
       distribution as $y$, and compute the ratio of its squared error to
       that of the conditional mean for $\tau \in \{0.1, 1, 10\}$. When is
       the price of matching the distribution of $y$ small?
   :exlabel:`ex_adversarial-losses-perfect-sampler`
1. [code] **Mean and median.** Squared error is minimized by the
   conditional mean :eqref:`eq_beyond_condmean`. Absolute error
   $\mathbb{E}\big[|y - g(x)|\big]$ is minimized by a conditional median, a value $t$
   with $P(y \leq t \mid x) = \tfrac12$. Both statistics can be computed for
   variants of the two-band data :eqref:`eq_beyond_twoband`.
    1. Let the offset $b$ take the three values $\{-1, 0, +1\}$ with
       equal probability. What are the conditional mean and the conditional
       median, and is each a valid output? Answer the same questions for the
       probabilities $(\tfrac14, \tfrac14, \tfrac12)$ of $-1$, $0$ and $+1$.
    1. Return to two bands and let $P(b = +1) = r$. Write the conditional
       mean as a function of $r$. For $r = \tfrac34$, show that the
       conditional median lies on the upper band, and compute its offset from
       $x$ to two decimal places. For $r = \tfrac12$, describe the
       absolute-error risk as a function of $g(x) - x$ inside the corridor
       between the bands.
    1. Train the squared-error student of
       :numref:`subsec_gan_beyond_capacity` and a copy trained with absolute
       error on the data of (a) and (b), using three seeds for $r = \tfrac12$,
       and compare the curves with your predictions. Under which condition on
       $p(y \mid x)$ is the conditional median a valid output?
   :exlabel:`ex_adversarial-losses-mean-and-median`
1. [code] **Pointwise plus adversarial.** ADD trains its student on the sum
   of an adversarial loss and a weighted pointwise distillation loss
   :cite:`Sauer.Lorenz.Blattmann.ea.2023`.
    1. Add $\lambda_{\textrm{pw}}$ times the squared error against the
       observed $y$ to the loss of the adversarial student
       of :numref:`subsec_gan_beyond_capacity`. Train with
       $\lambda_{\textrm{pw}} \in \{0.1, 1, 10, 100\}$ and two seeds each,
       and report the mean distances to the nearer band and to the
       conditional mean. Near which $\lambda_{\textrm{pw}}$ does the curve
       leave the bands?
    1. ADD's distillation target is not an observed sample. The teacher
       denoises a noised copy of the student's own output, and this estimate
       serves as a constant target. As a stand-in for a teacher at low noise,
       use the center of the band nearest to the student's prediction,
       $x + \operatorname{sign}(\hat y - x)$, with its gradient stopped, and
       repeat the sweep. Why does this pointwise term not pull the curve into
       the corridor?
    1. Now let the teacher see the prediction through noise of level
       $\tau$: $z = \hat y + \tau \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$
       drawn fresh at every step. Show that the posterior mean of the band
       center given $z$ is $x + \tanh\big((z - x)/(\tau^2 + 0.05^2)\big)$,
       and that this target tends to the target of (b) as $\tau \to 0$ and
       to the conditional mean $x$ as $\tau \to \infty$. Repeat the sweep
       with this target, gradient stopped, at $\tau = 0.1$ and
       $\tau = 2$. Which setting behaves like (a), and how do the three
       sweeps relate to ADD's one-step ablation, in which distillation alone
       gives FID 315.6, the adversarial loss alone 20.8, and both together
       20.6?
   :exlabel:`ex_adversarial-losses-pointwise-plus-adversarial`
1. **Phase and the conditional mean.** A vocoder receives a spectrogram,
   which fixes the frequency content and the magnitude of each frame but not
   the phase. Model one frame of $N$ samples as
   $y_t = \sin(\omega t + \varphi)$ for $t = 0, \ldots, N - 1$, where the
   spectrogram fixes $\omega$ and the phase $\varphi$ is uniform on $[-a, a]$
   with $0 < a \leq \pi$.
    1. Compute the squared-error prediction $\mathbb{E}[y_t]$ and show that
       its amplitude is $\sin a / a$ times the common amplitude of the valid
       waveforms. Evaluate the ratio for $a = \pi/2$.
    1. Let $\omega = 2\pi k / N$ for an integer $k$ with $0 < k < N/2$. Show
       that the magnitudes of the discrete Fourier transform of $y$ do not
       depend on $\varphi$. Which outputs minimize a loss on these magnitudes,
       such as the mel-spectrogram term in the generator loss of HiFi-GAN
       :cite:`Kong.Kim.Bae.2020`?
    1. Concatenate two such frames whose phases are drawn independently. A
       magnitude loss computed on the two non-overlapping frames is zero.
       What is wrong with the waveform at the boundary between the frames,
       and why can no loss that sees each non-overlapping frame only through
       its magnitudes detect it? What must the input of a critic contain for
       the critic to penalize this defect?
   :exlabel:`ex_adversarial-losses-phase`
1. **Frozen critic features.** Projected GAN and ADD score a sample through
   a frozen pretrained feature map $\phi$ and train only a head $r$, so that
   $D(\mathbf{x}) = r(\phi(\mathbf{x}))$
   :cite:`Sauer.Chitta.Muller.ea.2021,Sauer.Lorenz.Blattmann.ea.2023`. Let
   $p_\phi$ and $q_\phi$ be the distributions of $\phi(\mathbf{x})$ for
   $\mathbf{x} \sim p$ and for $\mathbf{x} \sim q$, and let $r$ range over all
   functions.
    1. Write the log-loss value
       $V(D) = \mathbb{E}_{\mathbf{x} \sim p}[\log \sigma(D(\mathbf{x}))] + \mathbb{E}_{\mathbf{x} \sim q}[\log \sigma(-D(\mathbf{x}))]$
       for this critic as an expectation over $p_\phi$ and $q_\phi$, and
       maximize it over $r$ pointwise, as for the optimal critic $D^\star$.
       What is the optimal head, and which divergence does the maximal value
       measure? Which generator distributions $q$ minimize this value?
    1. Let $p = \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$ and
       $\phi(\mathbf{x}) = x_1$. Give a distribution $q \neq p$ that
       minimizes the value of (a).
    1. Suppose that the generator is evaluated by the Fréchet distance of
       :numref:`sec_dcgan`, computed with the same features $\phi$. Which
       score does a minimizer from (a) reach, and what does this score say
       about $q$? Relate the answer to the observation of
       :citet:`Kynkaanniemi.Karras.Aittala.ea.2023` that ImageNet-pretrained
       discriminators can lower Inception-based FID.
   :exlabel:`ex_adversarial-losses-frozen-features`
1. **DDO's implicit critic.** Direct discriminative optimization
   :cite:`Zheng.Chen.Chen.ea.2025` plays the log-loss game of
   :numref:`sec_basic_gan` between real data and samples from a frozen model
   $q_{\textrm{ref}}$, with the critic $D_\theta$ of :eqref:`eq_beyond_ddo`.
   It minimizes
   $L(\theta) = -\mathbb{E}_{x \sim p}[\log \sigma(D_\theta(x))] - \mathbb{E}_{x \sim q_{\textrm{ref}}}[\log \sigma(-D_\theta(x))]$
   over the parameters $\theta$ of $q_\theta$, starting from
   $\theta = \theta_{\textrm{ref}}$.
    1. Show in three lines that if training reaches $q_\theta = p$, this
       critic equals the optimal critic between $p$ and $q_{\textrm{ref}}$,
       so the parameterization can represent the exact solution of the game
       it plays.
    1. Show that at $\theta = \theta_{\textrm{ref}}$ the gradient of $L$ is
       one half of the gradient of the negative log-likelihood
       $-\mathbb{E}_{x \sim p}[\log q_\theta(x)]$. Use the identity
       $\mathbb{E}_{x \sim q_\theta}[\nabla_\theta \log q_\theta(x)] = 0$, which
       follows from differentiating $\int q_\theta(x)\, dx = 1$.
    1. At $\theta = \theta_{\textrm{ref}}$, every real and every reference
       sample enters the gradient with weight $\tfrac12$. Away from
       $\theta_{\textrm{ref}}$, which real samples receive a smaller weight,
       and which reference samples does the update push down most strongly?
   :exlabel:`ex_adversarial-losses-ddo-plays-log-loss`
1. [code] **A latent bit.** ● Give the student of
   :numref:`subsec_gan_beyond_capacity` the information it lacks: a latent
   input $u \in \{-1, +1\}$, drawn uniformly and independently of $x$, so
   that the student maps $(x, u)$ to $\hat y$. The critic still scores
   $(x, y)$ pairs.
    1. Train one such student with squared error and one adversarially. At
       several values of $x$, find the band on which the output for each
       latent value lies. Which of the two losses gives the student an
       incentive to use the latent bit?
    1. IMLE :cite:`Li.Malik.2018` associates each data point with its
       nearest generated sample. With a binary latent, this becomes the loss
       $\mathbb{E}\big[\min_{u} (g(x, u) - y)^2\big]$. Train a student with this loss
       and compare it with the two students of (a). Which property of the
       loss forces both latent values into use without a critic?
    1. Now let $P(b = +1) = \tfrac34$. Which band frequencies can a student
       with a uniform binary latent represent at a given $x$? Compute the
       Jensen--Shannon divergence between the band probabilities
       $(\tfrac14, \tfrac34)$ of the data and each representable choice,
       predict the choice that the adversarial student and the IMLE student
       reach, and check by training. What must change in the latent input for
       a student to represent the probability $\tfrac34$?
   :exlabel:`ex_adversarial-losses-latent-bit`
1. [code] **Multi-round refinement.** ●● In practice DDO minimizes

   $$L_{\alpha, \beta}(\theta) = -\mathbb{E}_{x \sim p}\big[\log \sigma\big(\beta D_\theta(x)\big)\big] - \alpha\, \mathbb{E}_{x \sim q_{\textrm{ref}}}\big[\log \sigma\big(-\beta D_\theta(x)\big)\big]$$

   with $D_\theta$ from :eqref:`eq_beyond_ddo`, $\beta$ typically between
   0.01 and 0.1, and several rounds, each of which replaces
   $q_{\textrm{ref}}$ by the model of the previous round
   :cite:`Zheng.Chen.Chen.ea.2025`.
    1. Let $q_\theta$ range over all densities. Minimize the integrand at
       each $x$ over $D_\theta(x)$ to show that the minimizer is proportional
       to $q_{\textrm{ref}}^{1 - 1/\beta}\, p^{1/\beta}$ and is normalized for
       exactly one value of $\alpha$. With $r_n = \log(q_n / p)$ for the
       model of round $n$, derive the recursion from $r_n$ to $r_{n+1}$ up to
       an additive constant. For which $\beta$ do the rounds converge to $p$,
       and what happens at $\beta = 0.05$?
    1. Now let the family $q_\theta$ be restricted. Generalize problem
       :exref:`ex_adversarial-losses-ddo-plays-log-loss`b to show that at
       $\theta = \theta_{\textrm{ref}}$ the gradient of $L_{\alpha, \beta}$
       is $\beta / 2$ times the gradient of the negative log-likelihood, for
       every $\alpha$. Conclude that every fixed point of the rounds is a
       stationary point of the likelihood.
    1. Let $q_\theta(x) = h(x) \exp\big(\theta^\top T(x) - A(\theta)\big)$
       be an exponential family and $\theta_{\textrm{ref}}$ its
       maximum-likelihood fit, so that
       $\mathbb{E}_{x \sim q_{\textrm{ref}}}[T(x)] = \mathbb{E}_{x \sim p}[T(x)]$. Show that for
       $\alpha = 1$ and every $\beta > 0$, $\theta_{\textrm{ref}}$ minimizes
       $L_{1, \beta}$ globally.
    1. Fit the Gaussian family to
       $p = \tfrac12 \mathcal{N}(-2, 0.5^2) + \tfrac12 \mathcal{N}(2, 0.5^2)$
       by quadrature on a grid, starting the rounds from
       $q_{\textrm{ref}} = \mathcal{N}(1, 1)$ with $\alpha = 1$, and track the
       mean and the standard deviation over eight rounds for
       $\beta \in \{1, 0.75, 0.5, 0.05\}$. Compare the threshold with (a). By (b)
       and (c) the rounds cannot improve on maximum likelihood here; which
       features of the setting of :citet:`Zheng.Chen.Chen.ea.2025` must
       differ for their reported gains to be possible?
   :exlabel:`ex_adversarial-losses-multi-round-refinement`

[Discussions](https://d2l.discourse.group/t/36814)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.7]{.kicker}

Adversarial losses beyond GANs<br>
**conditional means · two bands · denoisers and distillation · tokenizers and vocoders · the objective without the game**
:::
:::

::: {.slide title="Squared Error Turns Many Sharp Answers into One Blur"}
[The problem]{.kicker}

- Enlarge a thumbnail: many sharp photographs shrink to the same thumbnail.
- Squared error rewards the prediction closest to all of them on average,
  their pixelwise mean: a blur that matches none of them.
- Two bands $y = x \pm 1$: the squared-error fit runs between them, about
  **one unit** from every data point.

::: {.d2l-note .warn}
Regression trains most generators today. Why do one-step image generators,
tokenizers and vocoders still keep a critic?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the conditional-mean decomposition of squared error;
- predict where a squared-error and an adversarial student land on two bands;
- tell ambiguity from limited capacity, and say which a larger model fixes;
- explain why a one-step student of a denoiser needs a critic or a stand-in;
- name what an adversarial objective needs: an estimate of $\log(p/q)$.

**The idea:** a pointwise loss averages over ambiguity, a learned comparison
does not. **Running example:** two bands, one input, two answers.
:::

::: {.slide title="The Last Deck Asks Where the Comparison Survives"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| deck | adds |
|---|---|
| Generative Adversarial Networks | the log-loss game and its value |
| Adversarial Objectives | other losses and critic classes |
| Relativistic Objectives | a critic that ranks pairs |
| Gradient Penalties | convergence of the training dynamics |
| Adversarial Image Generation | convolutional networks, scores |
| Conditional Generation | asking for a class |
| **Adversarial Losses Beyond GANs** | **where the comparison survives** |
:::

::: {.col .narrow}
- **Uses:** $D^\star = \log(p/q)$; the non-saturating loss; zero-centered
  penalties.
- **Skips:** how diffusion models are trained; architectures.
:::
:::
:::

::: {.slide title="Squared Error Grades Distance to the Conditional Mean"}
[Conditional means]{.kicker}

For any predictor $g$, with $\mu(x) = \mathbb{E}[y \mid x]$, the cross term
vanishes given $x$:

$$\mathbb{E}\big[(y - g(x))^2\big]
= \underbrace{\mathbb{E}\big[(y - \mu(x))^2\big]}_{\textrm{no } g \textrm{ removes it}}
+ \mathbb{E}\big[(\mu(x) - g(x))^2\big]$$

- The best predictor is $g = \mu$.
- One answer per input: $\mu$ is that answer.
- Several answers: $\mu$ is their average, which need not be an answer.
:::

::: {.slide title="On Two Bands the Mean Lies Where No Data Is"}
[Two bands]{.kicker}

$x \sim U[-1, 1]$, $\;y = x + b + 0.05\,\epsilon$, $\;b = \pm 1$ with
probability $\tfrac12$ each, independent of $x$.

- Bands $y = x \pm 1$, forty noise widths apart; $\mu(x) = x$, in the corridor.
- Expected squared error of $\mu$: $1 + 0.05^2 \approx 1.0025$.
- Of the band $y = x + 1$: $\tfrac12 \cdot 4 + 0.05^2 \approx 2.0025$.

::: {.d2l-note}
Squared error prefers the corridor, where no data point lies.
:::
:::

::: {.slide title="A Critic Scores Whether a Prediction Lies among the Data"}
[Two bands]{.kicker}

Critic $D(x, y)$, trained by logistic regression on data pairs against
predicted pairs $(x, \hat y)$:

$$D^\star(x, y) = \log \frac{p(x, y)}{q(x, y)}$$

- The student minimizes $-\log \sigma\big(D(x, \hat y)\big)$, the
  non-saturating loss.
- $\hat y = g(x) + 0.05\,\xi$: the data's noise, so only the location of
  mass differs.
- A prediction in the empty corridor scores low, however close it is on
  average.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="pytorch"}
[Two bands]{.kicker}

```
for _ in range(num_steps):
    x, y = make_bands(batch_size)                   # a fresh batch
    y_hat = g(x) + 0.05 * torch.randn(batch_size, 1)
    real, fake = torch.cat([x, y], 1), torch.cat([x, y_hat], 1)
    opt_D.zero_grad()
    (softplus(-D(real)) + softplus(D(fake.detach()))).mean().backward()
    opt_D.step()
    opt_G.zero_grad()
    softplus(-D(fake)).mean().backward()
    opt_G.step()
```

Same 32-unit network for both students, same initial weights.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="jax"}
[Two bands]{.kicker}

```
x, y = make_bands(k_data, batch_size)               # a fresh batch
xi = jax.random.normal(k_xi, (batch_size, 1))
real = jnp.concatenate([x, y], 1)
fake = lambda g_: jnp.concatenate([x, g_(x) + 0.05 * xi], 1)
loss_D = lambda D_: (softplus(-D_(real))
                     + softplus(D_(fake(g)))).mean()
opt_D.update(D, nnx.grad(loss_D)(D))
loss_G = lambda g_: softplus(-D(fake(g_))).mean()
opt_G.update(g, nnx.grad(loss_G)(g))
```

Same 32-unit network for both students, same initial weights.
:::

::: {.slide title="The Adversarial Student Commits to a Band"}
[Two bands]{.kicker}

@!adversarial-losses-the-capacity-argument-3

Squared error: within about 0.02 of $y = x$, one unit from every band.
Adversarial: a few hundredths from a band, zero to two crossings.
:::

::: {.slide title="Check: Which Predictor Fools the Critic?"}
[Check]{.kicker}

Three predictors on the two bands: the mean $\mu(x) = x$, the band
$y = x + 1$, and an exact sampler $\hat y \sim p(y \mid x)$. Rank them by
squared error. Which one can an optimal critic not tell from the data?

. . .

- Mean $\approx 1.0025$ $<$ band $\approx 2.0025$ $<$ sampler $\approx 2.005$.
- The sampler's error is twice the mean's: $2\,\mathbb{E}(y - \mu)^2$.
- Only the sampler matches the data; squared error ranks it last.
:::

::: {.slide title="A Bigger Network Cannot Pick a Band; a Latent Bit Can"}
[Ambiguity and capacity]{.kicker}

- $x$ carries no information about the band: no deterministic $g(x)$, of any
  size, beats $\mu$ under squared error.
- Add a random bit $u = \pm 1$: $g(x, u)$ can send each value to one band.
- **Ambiguity:** the input does not fix the output (thumbnail, bands, phase).
- **Capacity:** it does, but one pass cannot compute the map (a one-step
  student of a long sampler).

::: {.d2l-note}
Squared error averages over what the model cannot tell apart. More capacity
fixes only the second case.
:::
:::

::: {.slide title="Realism Costs 3 dB of Squared Error"}
[Ambiguity and capacity]{.kicker}

- An exact sampler has twice the expected error of the conditional mean:
  $10 \log_{10} 2 \approx 3$ dB less peak signal-to-noise ratio.
- Low distortion and outputs distributed like the data cannot both be
  reached.
- So a sharp output loses to the blur on squared error; judge it by whether
  it looks like data.
- At image scale the ambiguity sits in fine texture: patch FID separates
  few-step distillation methods that whole-image FID does not.

[Blau and Michaeli 2017; Lin et al. 2024]{.small}
:::

::: {.slide title="A Squared-Error Denoiser Returns the Mean of the Clean Image"}
[Current systems]{.kicker}

Train $g(x_t) \approx x_0$ by squared error, $x_t$ = $x_0$ plus noise at
level $t$. With $(x, y) = (x_t, x_0)$:

$$g^\star(x_t) = \mathbb{E}[x_0 \mid x_t]$$

- Low noise: $x_t$ nearly fixes $x_0$, the mean is sharp.
- High noise: many images fit $x_t$, the mean is a blur, the corridor.
- Many steps: move a little, re-noise, ask again; no average is the output.
- One step: a student fitted to the teacher's estimate inherits the average.
:::

::: {.slide title="In One-Step Distillation the Adversarial Term Carries the Quality"}
[Current systems]{.kicker}

ADD: regression toward a frozen teacher plus a critic on frozen DINOv2
features. One-step ablation, everything else fixed:

| loss | FID |
|:---|---:|
| distillation only | 315.6 |
| adversarial only | 20.8 |
| both | 20.6 |

::: {.d2l-note}
The regression target is a conditional mean; the critic supplies the
sharpness.
:::

[Sauer et al. 2023]{.small}
:::

::: {.slide title="DMD Follows the Gradient of the Optimal Critic"}
[Current systems]{.kicker}

Move each generated sample $x'$ along

$$\nabla_x \log p(x') - \nabla_x \log q(x') = \nabla_x \log \frac{p}{q}(x') = \nabla_x D^\star(x')$$

- $\nabla \log p$: the frozen teacher's score; $\nabla \log q$: a second
  diffusion model trained online on the student's samples.
- Both scores are learned by regression, on noised copies of $x'$.
- No classifier and no minimax, but still a coupled loop.

[Yin et al. 2024]{.small}
:::

::: {.slide title="Phase Is Ambiguous, So a Squared-Error Vocoder Shrinks the Wave"}
[Current systems]{.kicker}

A spectrogram fixes frequency and magnitude, not phase. With
$\varphi \sim U[-a, a]$:

$$\mathbb{E}_\varphi\big[\sin(\omega t + \varphi)\big] = \frac{\sin a}{a}\,\sin(\omega t)$$

- Same frequency, smaller amplitude: the average of valid waves is not one.
- A critic that sees the waveform rejects it: HiFi-GAN, BigVGAN; RAF uses
  the paired loss.

[Kong et al. 2020; Lee et al. 2023; Lee and Choi 2026]{.small}
:::

::: {.slide title="Check: What Does a Squared-Error Vocoder Output?"}
[Check]{.kicker}

The phase of a frame is uniform on $[-\pi, \pi]$. What does the
squared-error prediction sound like? And for $[-\pi/2, \pi/2]$?

. . .

- $a = \pi$: $\sin \pi / \pi = 0$. The prediction is silence.
- $a = \pi/2$: $2/\pi \approx 0.64$ of the true amplitude.
- Every valid waveform has full amplitude; the mean is none of them.
:::

::: {.slide title="Tokenizers Keep a Patch Critic until Scale Breaks the Game"}
[Current systems]{.kicker}

- VQGAN and the Stable Diffusion autoencoder: a patch critic against the
  blur of pixel losses; the code does not fix each texture.
- ViTok: adversarial decoder fine-tuning, reconstruction FID 0.50, lower
  PSNR; generation FID only 5.5 to 4.9.
- ViTok-v2, 5B parameters: a fixed DINOv3 feature loss replaces the GAN and
  LPIPS losses; "reliance on adversarial losses prevents stable scaling".

[Esser et al. 2021; Rombach et al. 2022; Hansen-Estruch et al. 2025, 2026]{.small}
:::

::: {.slide title="Three Answers to Instability, Each with a Cost"}
[Responses to instability]{.kicker}

![](../img/mdl-gan-exits.svg){width=88%}
:::

::: {.slide title="A Likelihood Model Can Be Its Own Critic"}
[The objective without the game]{.kicker}

Two evaluable densities define a log-ratio critic:

$$D_\theta(x) = \log \frac{q_\theta(x)}{q_{\textrm{ref}}(x)}$$

- DDO trains this critic by logistic regression, real data against samples
  of a frozen copy $q_{\textrm{ref}}$, and so updates $q_\theta$ itself.
- At $q_\theta = p$ it is the optimal critic $\log(p/q_{\textrm{ref}})$.
- No discriminator network, no second player.

[Zheng et al. 2025]{.small}
:::

::: {.slide title="The Objective Needs log p/q, Not a Two-Player Game"}
[The objective without the game]{.kicker}

| system | how it gets $\log(p/q)$ or its gradient |
|:---|:---|
| GAN, vocoders | a critic network, trained against the generator |
| ADD | a critic head on frozen features |
| ViTok-v2 | no critic: a fixed feature loss |
| DMD | two score models, trained by regression |
| DDO | the likelihood model itself |
:::

::: {.slide title="Check: Which Systems Train a Second Network alongside the Generator?"}
[Check]{.kicker}

ADD, DMD, ViTok-v2, DDO: which train a second network at the same time,
and which of those play a minimax game?

. . .

- ADD: yes, the critic heads, adversarially.
- DMD: yes, the score model of $q$, by regression, not minimax.
- ViTok-v2 and DDO: no second network.
:::

::: {.slide title="What Remains after the Repairs"}
[Wrap-up]{.kicker}

- Two bands: squared error within 0.02 of $y = x$, a unit from the data; the
  critic keeps the student on a band. ADD: 315.6 to 20.6.
- Kept gradient, converging penalized game, a plain image generator.
- Still: no density; two networks and a tuned penalty; no likelihood term
  against a missing mode; instability at scale (ViTok-v2).

**Next:** diffusion and flow models avoid the game; they train by regression
against closed-form targets.
:::
