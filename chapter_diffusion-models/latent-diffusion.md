# Latent Diffusion and Transformers
:label:`sec_diffusion-latent`

Every image model of this chapter so far has looked at every pixel at every
step. The U-Net of :numref:`sec_diffusion-images` reads all 784 pixels of a
garment fifty times to draw it, and at $28 \times 28$ nobody minds. A color
photograph of $512 \times 512$ pixels holds a thousand times as many
numbers. The previous section showed what most of them are for: the
sampler fixes the silhouette of a garment while the noise is still heavy and
spends its last steps on texture, the detail a viewer hardly notices. A
wrong stitch goes unnoticed; a missing sleeve does not. Why, then, should
every step of a diffusion model carry all of those numbers, when most of
them describe the stitches?

:citet:`Rombach.Blattmann.Lorenz.ea.2022` answered by splitting the work in
two. An autoencoder compresses each image once into a small grid of
numbers, its *latent*, from which a decoder restores the image with little
visible loss. A diffusion model then learns the distribution of the
latents, at a fraction of the cost per step. A year later
:citet:`Peebles.Xie.2023` replaced the U-Net by a transformer that reads
the grid as a sequence of patches, the *diffusion transformer* (DiT), and
found that its samples kept improving as its computation grew. Most large
image generators of the following years combined the two ideas.

Both are choices a practitioner makes, and both have costs that a small
experiment can expose. We train an autoencoder on Fashion-MNIST, run the
regression of the earlier sections on its latents unchanged, and ask why the
latent model falls short of what its decoder can draw. We then build the
DiT, compare it with the U-Net at two sizes each, and end with when each of
the two is the right choice.

```{.python .input #latent-diffusion-latent-diffusion-and-transformers}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #latent-diffusion-latent-diffusion-and-transformers}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #latent-diffusion-latent-diffusion-and-transformers-1}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # tf32 matmuls on tensor cores
```

## Perceptual Compression

### The Cost of Pixels

Every sampling step evaluates the network on the whole image. A convolution
applies its kernel at every pixel, so the cost of a U-Net grows with the
number of pixels; attention compares every position with every other, so
its cost grows with their square. A $512 \times 512$ color image holds
$512 \cdot 512 \cdot 3 = 786{,}432$ numbers, and a sampler of fifty steps
touches each of them fifty times. Most of these numbers describe detail.
Write an image in a Fourier basis: the number of frequencies up to $k$
cycles per image grows like $k^2$, so most coordinates belong to fine
frequencies, and in :numref:`subsec_diffusion-design-space-scales` the
power of Fashion-MNIST images fell by three orders of magnitude from one to
fourteen cycles per image. The many fine coordinates carry little of the
image's power.

Rate–distortion theory gives us the vocabulary for what to throw away: for
each level of distortion, the rate–distortion function is the fewest bits
per image that a code needs (:numref:`subsec_mdl-rate-distortion`). Which
bits a viewer misses is a question about people, not about squared error,
and :citet:`Rombach.Blattmann.Lorenz.ea.2022` answered it by observation: a
likelihood model of natural images spends most of its bits at the
low-distortion end of the curve, on details nobody perceives. They leave
this *perceptual compression* to an autoencoder, and the *semantic
compression*, the modeling of what the image shows, to the diffusion model.
The autoencoder of Stable Diffusion maps a $512 \times 512 \times 3$ image
to a $64 \times 64 \times 4$ latent, 48 times fewer numbers. Every diffusion
step works on the latent, and the decoder runs once per sample.

What makes a good latent? It must keep enough for the decoder to restore
the image and be easy for the diffusion model to learn, and the two pull
apart: a latent that stored every pixel would reconstruct perfectly and
be as hard to model as the pixels. :citet:`Dieleman.2025` names the three
quantities of this trade. The *rate* is how many numbers the latent has, the
*distortion* is how much the decoder loses, and the *modelability* is how
well a generative model of the latents does. Our autoencoder maps 784
numbers to a $7 \times 7$ grid with four channels, a rate of 196 numbers
(:numref:`fig_diffusion-latent-pipeline`).

![Latent diffusion. The encoder maps an image to a smaller latent, the diffusion model trains and samples on latents, and the decoder maps a sampled latent back to an image. Top: our autoencoder, from $28 \times 28 \times 1$ (784 numbers) to $7 \times 7 \times 4$ (196). Bottom: Stable Diffusion's, from $512 \times 512 \times 3$ (786,432) to $64 \times 64 \times 4$ (16,384).](../img/mdl-diffusion-latent-pipeline.svg)
:label:`fig_diffusion-latent-pipeline`

### The Autoencoder

Modelability is the least obvious of the three, so let us say what a
diffusion model needs from a latent. It needs a known scale, against which
the noise of the Gaussian path is calibrated; no isolated outliers; and
nearby latents that decode to similar images, so that a slightly wrong
latent still gives a plausible image. An autoencoder trained only to
reconstruct guarantees none of this. Nothing stops it from storing its codes
at a scale of a thousand, or from decoding two neighboring codes to a shoe
and a shirt.

The variational autoencoder :cite:`Kingma.Welling.2014` offers a remedy for
all three. The encoder outputs a mean $\boldsymbol\mu$ and a standard
deviation $\mathbf{s}$ for each of the 196 latent numbers, the latent is
drawn as $\mathbf{z} = \boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi$
with standard normal $\boldsymbol\xi$, and the decoder maps $\mathbf{z}$ to
a reconstruction $\hat{\mathbf{x}}$. The loss adds to the reconstruction
error a term that pulls each latent number toward a standard normal:

$$\mathcal{L}(\mathbf{x}) = \frac{1}{784}\,\|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \frac{\beta}{196} \sum_{j=1}^{196} \tfrac{1}{2}\left(\mu_j^2 + s_j^2 - 1 - \log s_j^2\right).$$

The sum is the Kullback–Leibler divergence
$D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol\mu, \mathrm{diag}(\mathbf{s}^2))
\,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big)$, the term of the evidence
lower bound (:numref:`sec_mdl-latent-em-elbo`), and $\beta$ is its weight
(not the schedule's $\beta_k$). Read against the three needs, the loss does
two things. Pulling each $\mu_j$ toward zero and each $s_j$ toward one fixes
the scale and punishes outliers. Drawing $\mathbf{z}$ with noise of size
$\mathbf{s}$ during training shows the decoder codes that wander by about
$\mathbf{s}$, so it learns to decode them alike, which is the third need.

How much of this happens depends on $\beta$. With $\beta = 10^{-4}$, our
choice, the reconstruction stays in charge. At this weight the KL term fixes
the scale and keeps the encoder's noise from collapsing to zero, but it is
far too weak to make neighboring latents decode alike. Whether that matters
is not obvious, and we shall measure it below.

The autoencoders of large systems add more. They have usually trained with
a critic that scores image patches and with a perceptual loss, because
squared error averages plausible textures into a blur
(:numref:`sec_gan_beyond`); a recent large autoencoder, ViTok-v2, dropped
the critic for a fixed perceptual loss when it scaled up
:cite:`Hansen-Estruch.Chen.Ramanujan.ea.2026`. Ours needs neither, since
it keeps a quarter of the numbers of a small grayscale image on a black
background, while Stable Diffusion's keeps a forty-eighth of a photograph.
Other autoencoders quantize the latent into discrete tokens
:cite:`Esser.Rombach.Ommer.2021,Chang.Zhang.Jiang.ea.2022`, which gives a
bounded code with no scale to calibrate and the kind of input that masked
diffusion (:numref:`sec_diffusion-discrete`) works on.

Our autoencoder stacks the residual blocks of the U-Net, without an
embedding, at widths 64 and 128; its encoder halves the resolution twice to
a $7 \times 7$ grid with a mean and a log variance per latent channel, and
its decoder mirrors it.

```{.python .input #latent-diffusion-the-autoencoder}
%%tab pytorch
class Autoencoder(nn.Module):  #@save
    """Residual convolutional autoencoder from 28x28x1 images to a 7x7
    Gaussian latent and back; returns the reconstruction and the KL term."""
    def __init__(self, latent_channels=4, ch=64, num_blocks=2):
        super().__init__()
        blocks = lambda a, b: [d2l.UNetBlock(a, b)] + [
            d2l.UNetBlock(b, b) for _ in range(num_blocks - 1)]
        down = lambda c: nn.Conv2d(c, c, 3, stride=2, padding=1)
        head = lambda a, b: [nn.GroupNorm(8, a), nn.SiLU(),
                             nn.Conv2d(a, b, 3, padding=1)]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), *blocks(ch, ch),
            down(ch), *blocks(ch, 2 * ch),                       # 14 x 14
            down(2 * ch), *blocks(2 * ch, 2 * ch),               # 7 x 7
            *head(2 * ch, 2 * latent_channels))  # mean and log variance
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 2 * ch, 3, padding=1),
            *blocks(2 * ch, 2 * ch),
            nn.Upsample(scale_factor=2), *blocks(2 * ch, ch),    # 14 x 14
            nn.Upsample(scale_factor=2), *blocks(ch, ch),        # 28 x 28
            *head(ch, 1))

    def encode(self, x):
        mean, log_var = self.encoder(x).chunk(2, dim=1)
        return mean, log_var.clamp(-30, 20)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(mean)
        kl = 0.5 * (mean ** 2 + log_var.exp() - 1 - log_var).mean()
        return self.decoder(z), kl
```

```{.python .input #latent-diffusion-the-autoencoder}
%%tab jax
class Autoencoder(nnx.Module):  #@save
    """Residual convolutional autoencoder from 28x28x1 images to a 7x7
    Gaussian latent and back; returns the reconstruction and the KL term."""
    def __init__(self, latent_channels=4, ch=64, num_blocks=2, *, rngs):
        blocks = lambda a, b: [d2l.UNetBlock(a, b, rngs=rngs)] + [
            d2l.UNetBlock(b, b, rngs=rngs) for _ in range(num_blocks - 1)]
        conv = lambda a, b, **kw: d2l.Conv(a, b, (3, 3), rngs=rngs, **kw)
        down = lambda c: conv(c, c, strides=2, padding=1)
        up = lambda x: jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        head = lambda a, b: [nnx.GroupNorm(a, num_groups=8, epsilon=1e-5,
                                           use_fast_variance=False,
                                           rngs=rngs), nnx.silu, conv(a, b)]
        self.encoder = nnx.Sequential(
            conv(1, ch), *blocks(ch, ch),
            down(ch), *blocks(ch, 2 * ch),                       # 14 x 14
            down(2 * ch), *blocks(2 * ch, 2 * ch),               # 7 x 7
            *head(2 * ch, 2 * latent_channels))  # mean and log variance
        self.decoder = nnx.Sequential(
            conv(latent_channels, 2 * ch), *blocks(2 * ch, 2 * ch),
            up, *blocks(2 * ch, ch),                             # 14 x 14
            up, *blocks(ch, ch),                                 # 28 x 28
            *head(ch, 1))

    def encode(self, x):
        mean, log_var = jnp.split(self.encoder(x), 2, axis=-1)
        return mean, jnp.clip(log_var, -30, 20)

    def __call__(self, x, key):
        mean, log_var = self.encode(x)
        z = mean + jnp.exp(0.5 * log_var) * jax.random.normal(key, mean.shape)
        kl = 0.5 * (mean ** 2 + jnp.exp(log_var) - 1 - log_var).mean()
        return self.decoder(z), kl
```

We train it for 20,000 updates with `d2l.train_regression` of
:numref:`sec_diffusion-denoising`, which runs Adam on a loss and returns
the trained model, a moving average of its weights and the losses. The
learning rate decays to zero along a cosine, so the final weights need no
averaging. A word on cost before we start: this section trains two
autoencoders and fourteen diffusion models for 20,000 updates each, and on
one RTX 4090 the whole notebook runs for about an hour and a half, far
longer on a CPU.

```{.python .input #latent-diffusion-the-autoencoder-1}
%%tab pytorch
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()
dev = X.device

def batch_fn():
    return X[torch.randint(0, len(X), (128,), device=dev)]

def ae_loss(ae, x, beta=1e-4):
    x_hat, kl = ae(x)
    return ((x_hat - x) ** 2).mean() + beta * kl

torch.manual_seed(1)
ae, _, _ = d2l.train_regression(d2l.Autoencoder().to(dev), ae_loss, batch_fn,
                                20_000, cosine=True)
print(f'{d2l.num_params(ae) / 1e6:.2f}M parameters')
```

```{.python .input #latent-diffusion-the-autoencoder-1}
%%tab jax
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()

def batch_fn(key):
    return X[jax.random.randint(key, (128,), 0, len(X))]

def ae_loss(ae, x, key, beta=1e-4):
    x_hat, kl = ae(x, key)
    return ((x_hat - x) ** 2).mean() + beta * kl

ae, _, _ = d2l.train_regression(d2l.Autoencoder(rngs=nnx.Rngs(1)), ae_loss,
                                batch_fn, 20_000, jax.random.key(1),
                                cosine=True)
print(f'{d2l.num_params(ae) / 1e6:.2f}M parameters')
```

The first row shows eight test images, the second the decoder's output for
their latent means.

```{.python .input #latent-diffusion-the-autoencoder-2}
%%tab pytorch
with torch.no_grad():
    x_hat = ae.decoder(ae.encode(X_test[:8])[0])
d2l.show_gray(torch.cat([X_test[:8], x_hat]), 2, 8);
```

```{.python .input #latent-diffusion-the-autoencoder-2}
%%tab jax
encode = nnx.jit(lambda ae, x: ae.encode(x))      # mean and log variance
decoder = nnx.jit(lambda ae, z: ae.decoder(z))
x_hat = decoder(ae, encode(ae, X_test[:8])[0])
d2l.show_gray(jnp.concatenate([X_test[:8], x_hat]), 2, 8);
```

The reconstructions are hard to tell from the originals. To measure them we
use the Fréchet distance (FD) of :numref:`sec_diffusion-images`, the
distance between Gaussians fitted to the features that a small classifier,
`d2l.FeatureCNN`, extracts from two sets of images. Two sets of real images
differ by chance too, so we print the FD between two disjoint sets of 5,000
test images as the *floor*. All FDs in this section use 5,000 images; at
1,000 the floor is about five times higher
(:numref:`subsec_diffusion-fd-reference`).

```{.python .input #latent-diffusion-the-autoencoder-3}
%%tab pytorch
cnn = d2l.fashion_feature_cnn(X, y)
f_ref = d2l.fd_features(cnn, X_test[:5000])

def fd(imgs):
    """FD of images against the first 5,000 test images."""
    return d2l.fid_score(d2l.fd_features(cnn, imgs), f_ref)

print(f'floor, the other 5,000 test images: FD {fd(X_test[5000:]):.2f}')
```

```{.python .input #latent-diffusion-the-autoencoder-3}
%%tab jax
cnn = d2l.fashion_feature_cnn(X, y)
f_ref = d2l.fd_features(cnn, X_test[:5000])

def fd(imgs):
    """FD of images against the first 5,000 test images."""
    return d2l.fid_score(d2l.fd_features(cnn, imgs), f_ref)

print(f'floor, the other 5,000 test images: FD {fd(X_test[5000:]):.2f}')
```

We score the reconstructions of the other 5,000 test images.

```{.python .input #latent-diffusion-the-autoencoder-4}
%%tab pytorch
x_hat = d2l.batched(lambda x: ae.decoder(ae.encode(x)[0]), X_test[5000:])
fd_rec = fd(x_hat)
print(f'reconstructions: squared error {((x_hat - X_test[5000:]) ** 2).mean():.4f}'
      f' per pixel, FD {fd_rec:.2f}')
```

```{.python .input #latent-diffusion-the-autoencoder-4}
%%tab jax
x_hat = d2l.batched(lambda x: decoder(ae, encode(ae, x)[0]), X_test[5000:])
fd_rec = fd(x_hat)
print(f'reconstructions: squared error {((x_hat - X_test[5000:]) ** 2).mean():.4f}'
      f' per pixel, FD {fd_rec:.2f}')
```

The reconstructions score an FD of 0.6 to 0.8 against the floor of 0.17, at
a squared error of about 0.004 per pixel. This is the autoencoder's
distortion, and it is also the score of a perfect latent model: if the
diffusion model drew latents exactly as the encoder produces them, the
decoder would turn them into reconstructions, up to the encoder's small
noise. Every latent model below is read against this number. It is a
reference rather than a lower bound, for a reason the perturbation test
below will show.

We encode the training set once and print the standard deviation of the
latents, the average of the encoder's $\mathbf{s}$, and the excess kurtosis
of all latent values, which is zero for a normal distribution and large
when a few values lie far out.

```{.python .input #latent-diffusion-the-autoencoder-5}
%%tab pytorch
encoded = d2l.batched(lambda x: torch.cat(ae.encode(x), 1), X)
mu, log_var = encoded.chunk(2, dim=1)       # mean and log variance
s = (0.5 * log_var).exp()
z = mu + s * torch.randn_like(mu)
scale = 1 / z.std().item()
kurtosis = (((z - z.mean()) * scale) ** 4).mean() - 3
print(f'latent std {z.std():.2f} (scale factor {scale:.2f}), encoder std '
      f'{s.mean():.3f}, excess kurtosis {kurtosis:.2f}')
```

```{.python .input #latent-diffusion-the-autoencoder-5}
%%tab jax
encoded = d2l.batched(lambda x: jnp.concatenate(encode(ae, x), -1), X)
mu, log_var = jnp.split(encoded, 2, axis=-1)  # mean and log variance
s = jnp.exp(0.5 * log_var)
z = mu + s * jax.random.normal(jax.random.key(2), mu.shape)
scale = 1 / float(z.std())
kurtosis = (((z - z.mean()) * scale) ** 4).mean() - 3
print(f'latent std {z.std():.2f} (scale factor {scale:.2f}), encoder std '
      f'{s.mean():.3f}, excess kurtosis {kurtosis:.2f}')
```

The latents have a standard deviation slightly below one, so we multiply
them by a factor $c$ of about 1.05, which brings them to the unit variance
of the path's noise, and the excess kurtosis is near zero or negative:
there are no heavy tails. Two of the three needs are met. The encoder's
standard deviation is 0.02 to 0.03, so during training the decoder saw each
code blurred by about 3 percent of the latents' spread, and almost never by
more than a tenth. The third need, that slightly wrong latents decode to
plausible images, is the one this weak KL term does not supply.

## Diffusion in the Latent Space

### The Same Regression on Latents

With the autoencoder fixed, nothing changes but the data, which is the
attraction of the scheme: everything the earlier sections built carries
over. Each batch draws the latents of 128 training images, $\mathbf{z}_0 =
c\,(\boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi)$, fresh from the
encoder's distribution as :citet:`Rombach.Blattmann.Lorenz.ea.2022` do. We
train on the linear path of :numref:`sec_diffusion-gaussian-paths`: we form
$\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\boldsymbol\epsilon$ and regress
the network's output onto the velocity $\boldsymbol\epsilon - \mathbf{z}_0$.

The scale factor matters because the path mixes the data with noise of unit
variance. Latents with a standard deviation of five would shift the
log-SNR at every time by $\log 25$ (Exercise
:exref:`ex_latent-scale-factor`), and Stable Diffusion multiplies its
latents by 0.18215 for this reason. The network is the U-Net of
:numref:`sec_diffusion-images` with four input and output channels, and one
`loss_fn` serves images and latents.

```{.python .input #latent-diffusion-the-same-regression-on-latents-1}
%%tab pytorch
path = d2l.GaussianPath.linear()
loss_fn = d2l.regression_loss(path, 'velocity')  # for images or latents

def latent_batch_fn():
    i = torch.randint(0, len(mu), (128,), device=dev)
    return (mu[i] + s[i] * torch.randn_like(mu[i])) * scale

torch.manual_seed(3)
_, unet_latent, _ = d2l.train_regression(
    d2l.UNet(ch=64, in_channels=4, out_channels=4).to(dev), loss_fn,
    latent_batch_fn, 20_000)
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-1}
%%tab jax
path = d2l.GaussianPath.linear()
loss_fn = d2l.regression_loss(path, 'velocity')  # for images or latents

def latent_batch_fn(key):
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (128,), 0, len(mu))
    return (mu[i] + s[i] * jax.random.normal(k2, mu[i].shape)) * scale

_, unet_latent, _ = d2l.train_regression(
    d2l.UNet(ch=64, in_channels=4, out_channels=4, rngs=nnx.Rngs(3)),
    loss_fn, latent_batch_fn, 20_000, jax.random.key(3))
```

As a reference we train the same U-Net on pixels. Both runs take 20,000
updates, twice the budget of :numref:`sec_diffusion-samplers`, because the
transformers below need longer runs and this pixel model is their reference
too.

```{.python .input #latent-diffusion-the-same-regression-on-latents-2}
%%tab pytorch
torch.manual_seed(4)
_, unet_pixel, _ = d2l.train_regression(d2l.UNet(ch=64).to(dev), loss_fn,
                                        batch_fn, 20_000)
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-2}
%%tab jax
_, unet_pixel, _ = d2l.train_regression(d2l.UNet(ch=64, rngs=nnx.Rngs(4)),
                                        loss_fn, batch_fn, 20_000,
                                        jax.random.key(4))
```

`sample` draws 5,000 samples from fixed noise with 50 Euler steps of
`d2l.ode_sample` (:numref:`sec_diffusion-integration`), and `decode`
undoes the scale factor and decodes latents. The first row below holds
samples of the pixel model, the second decoded samples of the latent model.

```{.python .input #latent-diffusion-the-same-regression-on-latents-3}
%%tab pytorch
def sample(net, shape, n=5000):
    """n samples by 50 Euler steps from fixed noise, 1,000 at a time."""
    gen = torch.Generator(dev).manual_seed(0)
    noise = torch.randn(n, *shape, device=dev, generator=gen)
    pred = d2l.make_pred(net, path, 'velocity')
    return d2l.batched(lambda x: d2l.ode_sample(pred, x, path.sigma_bars(50)),
                       noise)

def decode(z):
    return d2l.batched(lambda z_k: ae.decoder(z_k / scale), z)

latent_shape, image_shape = (4, 7, 7), (1, 28, 28)
x_pixel = sample(unet_pixel, image_shape)
x_latent = decode(sample(unet_latent, latent_shape))
fd_pixel, fd_latent = fd(x_pixel), fd(x_latent)
print(f'FD: pixel U-Net {fd_pixel:.2f}, latent U-Net {fd_latent:.2f}')
d2l.show_gray(torch.cat([x_pixel[:8], x_latent[:8]]), 2, 8);
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-3}
%%tab jax
@nnx.jit
def euler50(net, noise):
    pred = d2l.make_pred(net, path, 'velocity')
    return d2l.ode_sample(pred, noise, path.sigma_bars(50))

def sample(net, shape, n=5000):
    """n samples by 50 Euler steps from fixed noise, 1,000 at a time."""
    noise = jax.random.normal(jax.random.key(0), (n, *shape))
    return d2l.batched(lambda x: euler50(net, x), noise)

def decode(z):
    return d2l.batched(lambda z_k: decoder(ae, z_k / scale), z)

latent_shape, image_shape = (7, 7, 4), (28, 28, 1)
x_pixel = sample(unet_pixel, image_shape)
x_latent = decode(sample(unet_latent, latent_shape))
fd_pixel, fd_latent = fd(x_pixel), fd(x_latent)
print(f'FD: pixel U-Net {fd_pixel:.2f}, latent U-Net {fd_latent:.2f}')
d2l.show_gray(jnp.concatenate([x_pixel[:8], x_latent[:8]]), 2, 8);
```

Both rows show recognizable garments. The latent model scores an FD of
about 2 (2.0 to 2.2), the pixel model about 1.5. Both are unconditional, as
are all models of this section, so neither can be compared with the
class-conditional 0.8 of :numref:`sec_diffusion-images`.

What did the latent model save? `train_step_gflops` counts the
floating-point operations of one training step, forward and backward on 128
inputs, in billions (GFLOP).

```{.python .input #latent-diffusion-the-same-regression-on-latents-5}
%%tab pytorch
def train_step_gflops(net, loss_fn, shape, batch_size=128):  #@save
    """GFLOP of one training step, forward and backward, on a batch of zeros
    of the given shape."""
    # A local import, so that d2l needs nothing beyond its header's imports
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(batch_size, *shape, device=next(net.parameters()).device)
    with FlopCounterMode(display=False) as counter:
        loss_fn(net, x).backward()
    return counter.get_total_flops() / 1e9
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-5}
%%tab jax
def train_step_gflops(net, loss_fn, shape, batch_size=128):  #@save
    """GFLOP of one training step, forward and backward, on a batch of zeros
    of the given shape: XLA's estimate for the step lowered for the CPU (for
    a GPU it leaves out the convolutions)."""
    graphdef, params, rest = nnx.split(net, nnx.Param, ...)
    grad = jax.grad(lambda p, r, x, key: loss_fn(nnx.merge(graphdef, p, r),
                                                 x, key))
    args = jax.device_put((params, rest, jnp.zeros((batch_size, *shape)),
                           jax.random.key(0)), jax.devices('cpu')[0])
    return jax.jit(grad).lower(*args).cost_analysis()['flops'] / 1e9
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-4}
%%tab pytorch
gflops_latent = d2l.train_step_gflops(
    d2l.UNet(ch=64, in_channels=4, out_channels=4).to(dev), loss_fn,
    latent_shape)
gflops_pixel = d2l.train_step_gflops(d2l.UNet(ch=64).to(dev), loss_fn,
                                     image_shape)
print(f'GFLOP per training step: latent U-Net {gflops_latent:.0f}, '
      f'pixel U-Net {gflops_pixel:.0f}')
```

```{.python .input #latent-diffusion-the-same-regression-on-latents-4}
%%tab jax
gflops_latent = d2l.train_step_gflops(
    d2l.UNet(ch=64, in_channels=4, out_channels=4, rngs=nnx.Rngs(0)), loss_fn,
    latent_shape)
gflops_pixel = d2l.train_step_gflops(d2l.UNet(ch=64, rngs=nnx.Rngs(0)),
                                     loss_fn, image_shape)
print(f'GFLOP per training step: latent U-Net {gflops_latent:.0f}, '
      f'pixel U-Net {gflops_pixel:.0f}')
```

The latent U-Net needs about a dozen GFLOP per step, the pixel U-Net about
200. Every evaluation of the latent network, in training and at each
sampling step, costs less than a tenth of the arithmetic.

### Decoder Tolerance and Model Error

So the latent model is far cheaper and somewhat worse: its FD is about 1.4
times the pixel model's and about three times that of the reconstructions.
In Dieleman's terms, a rate of 196 numbers came with a distortion of 0.6 to
0.8, and the distance from there up to the samples measures modelability.
Where does that gap come from? Two causes suggest themselves. The latent
model may draw latents unlike the encoder's, which is its own error. Or it
may draw latents that are only slightly off, which a decoder trained on
codes moved by about 3 percent of the spread may turn into poor images. The
weak KL term makes the second cause plausible.

To tell the two apart, we build a decoder that tolerates wrong codes, check
that it does, and then train latent models on its latents. If intolerance
causes the gap, their samples should move toward the reconstructions. The
candidate is a second autoencoder, trained with ten times the KL weight.
The check decodes latents that are wrong on purpose: the latent means of
the other 5,000 test images plus Gaussian noise of $\sigma$ times the
latents' standard deviation. Let us predict the outcome first: the second
encoder should blur its codes more, so its decoder should tolerate larger
errors and reconstruct worse.

```{.python .input #latent-diffusion-reading-the-result-1}
%%tab pytorch
torch.manual_seed(9)
ae_10, _, _ = d2l.train_regression(
    d2l.Autoencoder().to(dev), lambda ae, x: ae_loss(ae, x, beta=1e-3),
    batch_fn, 20_000, cosine=True)
```

```{.python .input #latent-diffusion-reading-the-result-1}
%%tab jax
ae_10, _, _ = d2l.train_regression(
    d2l.Autoencoder(rngs=nnx.Rngs(9)),
    lambda ae, x, key: ae_loss(ae, x, key, beta=1e-3), batch_fn, 20_000,
    jax.random.key(9), cosine=True)
```

`perturbed_decodes` prints the encoder's standard deviation in units of the
latents' standard deviation, the reconstruction error and FD, and for each
$\sigma$ the mean squared change of the decoded image and its FD.

```{.python .input #latent-diffusion-reading-the-result-2}
%%tab pytorch
def perturbed_decodes(ae, x, sigmas=(0.1, 0.3, 0.5)):
    """Decode the latent means of x, and the means plus Gaussian noise of
    sigma times their standard deviation."""
    mu, log_var = d2l.batched(lambda x: torch.cat(ae.encode(x), 1),
                              x).chunk(2, dim=1)
    x_rec = d2l.batched(ae.decoder, mu)
    print(f'  encoder std {(0.5 * log_var).exp().mean() / mu.std():.3f}, '
          f'reconstructions: error {((x_rec - x) ** 2).mean():.4f}, '
          f'FD {fd(x_rec):.2f}')
    gen = torch.Generator(dev).manual_seed(0)
    eps = torch.randn(mu.shape, device=dev, generator=gen)
    for sigma in sigmas:
        x_noisy = d2l.batched(ae.decoder, mu + sigma * mu.std() * eps)
        print(f'  sigma {sigma}: change {((x_noisy - x_rec) ** 2).mean():.4f} '
              f'per pixel, FD {fd(x_noisy):.2f}')

for beta, model in ((1e-4, ae), (1e-3, ae_10)):
    print(f'beta {beta:g}')
    perturbed_decodes(model, X_test[5000:])
```

```{.python .input #latent-diffusion-reading-the-result-2}
%%tab jax
def perturbed_decodes(ae, x, sigmas=(0.1, 0.3, 0.5)):
    """Decode the latent means of x, and the means plus Gaussian noise of
    sigma times their standard deviation."""
    mu, log_var = jnp.split(
        d2l.batched(lambda x: jnp.concatenate(encode(ae, x), -1), x), 2,
        axis=-1)
    x_rec = d2l.batched(lambda z: decoder(ae, z), mu)
    print(f'  encoder std {jnp.exp(0.5 * log_var).mean() / mu.std():.3f}, '
          f'reconstructions: error {((x_rec - x) ** 2).mean():.4f}, '
          f'FD {fd(x_rec):.2f}')
    eps = jax.random.normal(jax.random.key(0), mu.shape)
    for sigma in sigmas:
        x_noisy = d2l.batched(lambda z: decoder(ae, z),
                              mu + sigma * mu.std() * eps)
        print(f'  sigma {sigma}: change {((x_noisy - x_rec) ** 2).mean():.4f} '
              f'per pixel, FD {fd(x_noisy):.2f}')

for beta, model in ((1e-4, ae), (1e-3, ae_10)):
    print(f'beta {beta:g}')
    perturbed_decodes(model, X_test[5000:])
```

Both predictions hold. The tenfold weight makes the encoder blur each code
by about a fifth of the latents' spread, seven times as much as before, and
its reconstructions score about 0.3 worse. At half the spread the weak
decoder's FD jumps above 3, while the tolerant one stays near 1.3. Yet the
decoded images change by about as much in both, 0.053 to 0.055 per pixel
against 0.061 to 0.062. The stronger KL term has not taught the decoder to
ignore a wrong code. It has taught it to decode a wrong code into a
different but plausible garment, which is what a Fréchet distance rewards.

The printout holds a small surprise as well. For both autoencoders a little
noise lowers the FD below that of the reconstructions (at $\sigma = 0.1$ for
both, and further at $\sigma = 0.3$ for the tolerant one), although the
decoded images move away from the originals. A likely reason is that
decoding the mean of each code gives an image that is slightly too smooth,
an average over the codes the decoder saw in training, and that a little
noise restores some of the variety of real images; no cell here tests this.
Either way the printed numbers settle the point that matters: the
reconstruction FD is a reference, not a bound that no latent model can
pass.

Does the more tolerant decoder give better samples? `latent_unet_fd` runs
the recipe above on the latents of a given autoencoder, a latent U-Net
trained for 20,000 updates and sampled with 50 Euler steps, and returns the
FD of the decoded samples. We train two per autoencoder, from two seeds.

```{.python .input #latent-diffusion-reading-the-result-3}
%%tab pytorch
def latent_unet_fd(ae, seed):
    """FD of a latent U-Net trained on the latents of ae as above."""
    mu, log_var = d2l.batched(lambda x: torch.cat(ae.encode(x), 1),
                              X).chunk(2, dim=1)
    s = (0.5 * log_var).exp()
    c = 1 / (mu + s * torch.randn_like(mu)).std()   # the scale factor

    def latent_batch():
        i = torch.randint(0, len(mu), (128,), device=dev)
        return (mu[i] + s[i] * torch.randn_like(mu[i])) * c

    torch.manual_seed(seed)
    _, net, _ = d2l.train_regression(
        d2l.UNet(ch=64, in_channels=4, out_channels=4).to(dev), loss_fn,
        latent_batch, 20_000)
    z = sample(net, latent_shape)
    return fd(d2l.batched(lambda z_k: ae.decoder(z_k / c), z))

for beta, model in ((1e-4, ae), (1e-3, ae_10)):
    fds = [latent_unet_fd(model, seed) for seed in (10, 11)]
    print(f'beta {beta:g}: latent U-Net FD {fds[0]:.2f} and {fds[1]:.2f}')
```

```{.python .input #latent-diffusion-reading-the-result-3}
%%tab jax
def latent_unet_fd(ae, seed):
    """FD of a latent U-Net trained on the latents of ae as above."""
    mu, log_var = jnp.split(
        d2l.batched(lambda x: jnp.concatenate(encode(ae, x), -1), X), 2,
        axis=-1)
    s = jnp.exp(0.5 * log_var)
    key_z, key_train = jax.random.split(jax.random.key(seed))
    c = 1 / (mu + s * jax.random.normal(key_z, mu.shape)).std()  # scale

    def latent_batch(key):
        k1, k2 = jax.random.split(key)
        i = jax.random.randint(k1, (128,), 0, len(mu))
        return (mu[i] + s[i] * jax.random.normal(k2, mu[i].shape)) * c

    _, net, _ = d2l.train_regression(
        d2l.UNet(ch=64, in_channels=4, out_channels=4, rngs=nnx.Rngs(seed)),
        loss_fn, latent_batch, 20_000, key_train)
    z = sample(net, latent_shape)
    return fd(d2l.batched(lambda z_k: decoder(ae, z_k / c), z))

for beta, model in ((1e-4, ae), (1e-3, ae_10)):
    fds = [latent_unet_fd(model, seed) for seed in (10, 11)]
    print(f'beta {beta:g}: latent U-Net FD {fds[0]:.2f} and {fds[1]:.2f}')
```

It does not. Every latent U-Net on the tenfold weight scores worse than
every one on the weak weight, by about half a point (0.4 to 0.8). That is at
least two and a half times the spread between the two seeds of either, and
the samples of the tenfold autoencoder also sit further above its own
reconstructions. If intolerance were the cause, the tolerant decoder would
have narrowed the gap; in no run did it. The perturbation test adds a second
clue. Noise of half the latents' spread costs the tolerant decoder an FD of
only about 1.3, yet the latent models on its latents score 2.6 to 2.7. So
whatever these models get wrong, it is not a small random error on the
right latents: their errors are larger than that, or of another kind. At
our scale the gap is the latent model's own error.

That does not let the autoencoder off the hook. The same U-Net, trained by
the same recipe, did clearly worse on the latents of one encoder than on
those of the other, so the encoder decides how hard the latent model's job
is. This is Dieleman's modelability, a property of the autoencoder that
only a trained diffusion model can measure. The KL weight is therefore a
real choice, and here the small one is the right one: the stronger KL term
made both the reconstructions and the samples worse. Exercise
:exref:`ex_latent-bottleneck` pushes the weight further in both directions.

At $28 \times 28$ the trade is not attractive. The pixel model is cheap
anyway, the autoencoder needs a training run of its own, and the latent
samples score worse.

## The Transformer Backbone

The U-Net builds in the assumptions of a convolutional network: nearby
pixels matter most, and the same features can appear anywhere. The vision
transformer of :numref:`sec_vision-transformer` makes fewer of them, and on
image classification it trails convolutional networks when trained on
little data and overtakes them when pretrained on a great deal.
:citet:`Peebles.Xie.2023` asked whether the same holds for diffusion.

### Patches and Tokens

The DiT reads its input as the vision transformer does. It cuts the image
into $p \times p$ patches, maps each to a vector of width $d$ by a linear
map (a convolution with kernel size and stride $p$), adds a learned position
embedding, and passes the tokens through pre-norm transformer blocks
(:numref:`sec_transformer-block`). With $p = 4$ a $28 \times 28$ image
becomes 49 tokens of 16 pixels each; on our latents with $p = 1$, each of
the 49 grid positions is a token of four numbers.

The output differs from a classifier's. A denoiser returns a prediction of
the shape of its input, so the DiT maps every token by a final linear layer
to the values of its patch and reassembles the patches. Each token predicts
the velocity of its own patch, informed through attention by all the
others. With one token per pixel the image would have 16 times as many
tokens, and attention would cost 256 times as much.

### Adaptive Layer Norm

The transformer also needs the time $t$, and possibly a label. The U-Net of
:numref:`sec_diffusion-images` solved this by adding a learned projection
of the embedded time and label to the feature maps of every block, one shift
per channel. Adaptive layer norm keeps the shift and adds a scale and a
gate.

A layer norm standardizes each token and applies a learned scale and shift.
In *adaptive layer norm* the scale $\mathbf{a}$ and the shift $\mathbf{b}$
are computed from the condition $\mathbf{c}$, the sum of a time embedding
and a label embedding, and the condition also sets a *gate* $\mathbf{g}$ on
each sublayer's output. One block computes

$$
\begin{aligned}
\mathbf{X} &\leftarrow \mathbf{X} + \mathbf{g}_1 \odot \mathrm{Attention}\big(\mathrm{LN}(\mathbf{X}) \odot (1 + \mathbf{a}_1) + \mathbf{b}_1\big),\\
\mathbf{X} &\leftarrow \mathbf{X} + \mathbf{g}_2 \odot \mathrm{MLP}\big(\mathrm{LN}(\mathbf{X}) \odot (1 + \mathbf{a}_2) + \mathbf{b}_2\big),
\end{aligned}
$$

where $\mathrm{LN}$ has no scale or shift of its own and the six vectors
come from one linear layer applied to $\mathrm{SiLU}(\mathbf{c})$, shared by
all tokens of an image. Thus one small matrix product per image turns each
feature of each sublayer up or down, differently at every noise level.

The linear layer starts at zero, hence the name *adaLN-Zero*. All gates are
then zero, every block computes the identity, and the final layer, also
zero, outputs zero. Training starts from a network that predicts zero
velocity everywhere, and each block comes into play as its gates grow
(:numref:`fig_diffusion-dit-block`). Of everything in the block, the gates
that start at zero are the detail to remember. :citet:`Peebles.Xie.2023`
compared this design with appending the condition as extra tokens, with
cross-attention to it, and with adaptive layer norm without the gates, and
adaLN-Zero gave the best samples of the four. Exercise
:exref:`ex_latent-zero-gate` shows what the zero start does to the first two
thousand updates.

A time or a label is one vector per image, which suits adaptive layer
norm. A caption is a sequence of vectors, one per token from a frozen text
encoder, and for it cross-attention is the natural route. In
*cross-attention* the $m$ image tokens supply the queries and the $n$
caption tokens the keys and values, so the output has one row per image
token, the image's shape :cite:`Rombach.Blattmann.Lorenz.ea.2022`; the
multimodal DiT of :citet:`Esser.Kulal.Blattmann.ea.2024` instead puts image
and caption tokens into one sequence. Dropping the caption at random during
training makes the empty caption the null label of classifier-free guidance
(:numref:`subsec_diffusion-images-guided`).

![The diffusion transformer. Left: $4 \times 4$ patches become 49 tokens with learned positions. Middle: a DiT block; the condition $\mathbf{c}$, the sum of the time and label embeddings, sets a shift, a scale and a gate for each sublayer, and the gates start at zero. Right: a final adaptive layer norm and a zero-initialized linear layer map each token to its patch.](../img/mdl-diffusion-dit-block.svg)
:label:`fig_diffusion-dit-block`

`DiTBlock` builds the block on the `ViTBlock` of
:numref:`sec_vision-transformer`, with `ada` as the linear map.

```{.python .input #latent-diffusion-adaptive-layer-norm-1}
%%tab pytorch
class DiTBlock(d2l.ViTBlock):  #@save
    """ViTBlock with adaLN-Zero: the condition c sets a shift, a scale and a
    gate for each sublayer; the gates start at zero."""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads):
        super().__init__(num_hiddens, mlp_num_hiddens, num_heads, dropout=0.0,
                         use_bias=True)
        ln = lambda: nn.LayerNorm(num_hiddens, elementwise_affine=False,
                                  eps=1e-6)
        self.norm1, self.norm2 = ln(), ln()
        self.ada = nn.Sequential(nn.SiLU(),
                                 nn.Linear(num_hiddens, 6 * num_hiddens))
        nn.init.zeros_(self.ada[1].weight), nn.init.zeros_(self.ada[1].bias)

    def forward(self, X, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.ada(c)[:, None].chunk(6, dim=-1))
        X_mod = self.norm1(X) * (1 + scale1) + shift1
        X = X + gate1 * self.attention(X_mod, X_mod, X_mod)
        return X + gate2 * self.ffn(self.norm2(X) * (1 + scale2) + shift2)
```

```{.python .input #latent-diffusion-adaptive-layer-norm-1}
%%tab jax
class DiTBlock(d2l.ViTBlock):  #@save
    """ViTBlock with adaLN-Zero: the condition c sets a shift, a scale and a
    gate for each sublayer; the gates start at zero."""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, *, rngs):
        super().__init__(num_hiddens, mlp_num_hiddens, num_heads, 0.0,
                         use_bias=True, rngs=rngs)
        ln = lambda: nnx.LayerNorm(num_hiddens, epsilon=1e-6, use_scale=False,
                                   use_bias=False, rngs=rngs)
        self.norm1, self.norm2 = ln(), ln()
        zeros = nnx.initializers.zeros
        self.ada = nnx.Linear(num_hiddens, 6 * num_hiddens, kernel_init=zeros,
                              bias_init=zeros, rngs=rngs)

    def __call__(self, X, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = jnp.split(
            self.ada(nnx.silu(c))[:, None], 6, axis=-1)
        X_mod = self.norm1(X) * (1 + scale1) + shift1
        X = X + gate1 * self.attention(X_mod, X_mod, X_mod)
        return X + gate2 * self.ffn(self.norm2(X) * (1 + scale2) + shift2)
```

`DiT` adds the patch embedding, the learned positions, the time and label
embeddings, and the final adaptive layer norm and linear layer.

```{.python .input #latent-diffusion-adaptive-layer-norm-2}
%%tab pytorch
class DiT(nn.Module):  #@save
    """Diffusion transformer conditioned on a time t in [0, 1] and, with
    num_classes, on a label y; label num_classes means no label."""
    def __init__(self, img_size=28, patch_size=4, num_channels=1, width=256,
                 depth=6, num_heads=4, num_classes=None):
        super().__init__()
        self.width, self.num_classes = width, num_classes
        self.p, self.g = patch_size, img_size // patch_size
        self.patch = d2l.PatchEmbedding(img_size, patch_size, width,
                                        num_channels)
        self.pos = nn.Parameter(0.02 * torch.randn(1, self.g ** 2, width))
        self.time_mlp = d2l.make_mlp([width] * 3, nn.SiLU())
        self.label_emb = (nn.Embedding(num_classes + 1, width)
                          if num_classes else None)
        self.blks = nn.ModuleList([DiTBlock(width, 4 * width, num_heads)
                                   for _ in range(depth)])
        self.ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, patch_size ** 2 * num_channels)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def forward(self, x, t, y=None):
        c = self.time_mlp(d2l.time_embedding(t, self.width))
        if y is not None:
            c = c + self.label_emb(y)
        X = self.patch(x) + self.pos
        for blk in self.blks:
            X = blk(X, c)
        shift, scale = self.ada(c)[:, None].chunk(2, dim=-1)
        X = self.out(self.ln(X) * (1 + scale) + shift)  # (n, g*g, p*p*c)
        n, g, p = len(x), self.g, self.p  # tokens back to pixels
        X = X.reshape(n, g, g, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        return X.reshape(n, -1, g * p, g * p)
```

```{.python .input #latent-diffusion-adaptive-layer-norm-2}
%%tab jax
class DiT(nnx.Module):  #@save
    """Diffusion transformer (NHWC) conditioned on a time t in [0, 1] and,
    with num_classes, on a label y; label num_classes means no label."""
    def __init__(self, img_size=28, patch_size=4, num_channels=1, width=256,
                 depth=6, num_heads=4, num_classes=None, *, rngs):
        self.width, self.num_classes = width, num_classes
        self.p, self.g = patch_size, img_size // patch_size
        self.patch = d2l.PatchEmbedding(img_size, patch_size, width,
                                        num_channels, rngs=rngs)
        self.pos = nnx.Param(
            0.02 * rngs.params.normal((1, self.g ** 2, width)))
        self.time_mlp = d2l.make_mlp([width] * 3, nnx.silu, rngs=rngs)
        init = nnx.initializers.normal(1.0)  # N(0, 1), as in PyTorch
        self.label_emb = (nnx.Embed(num_classes + 1, width, rngs=rngs,
                                    embedding_init=init)
                          if num_classes else None)
        self.blks = nnx.List([DiTBlock(width, 4 * width, num_heads, rngs=rngs)
                              for _ in range(depth)])
        self.ln = nnx.LayerNorm(width, epsilon=1e-6, use_scale=False,
                                use_bias=False, rngs=rngs)
        zeros = nnx.initializers.zeros
        self.ada = nnx.Linear(width, 2 * width, kernel_init=zeros,
                              bias_init=zeros, rngs=rngs)
        self.out = nnx.Linear(width, patch_size ** 2 * num_channels,
                              kernel_init=zeros, bias_init=zeros, rngs=rngs)

    def __call__(self, x, t, y=None):
        c = self.time_mlp(d2l.time_embedding(t, self.width))
        if y is not None:
            c = c + self.label_emb(y)
        X = self.patch(x) + self.pos[...]
        for blk in self.blks:
            X = blk(X, c)
        shift, scale = jnp.split(self.ada(nnx.silu(c))[:, None], 2, axis=-1)
        X = self.out(self.ln(X) * (1 + scale) + shift)  # (n, g*g, p*p*c)
        n, g, p = x.shape[0], self.g, self.p  # tokens back to pixels
        X = X.reshape(n, g, g, p, p, -1).transpose(0, 1, 3, 2, 4, 5)
        return X.reshape(n, g * p, g * p, -1)
```

We build a DiT of width 256 and depth 6 with four heads, and pass eight
images and times through it.

```{.python .input #latent-diffusion-adaptive-layer-norm-3}
%%tab pytorch
dit = d2l.DiT().to(dev)       # width 256, depth 6, 4 heads, 4 x 4 patches
x, t = X[:8], torch.rand(8, device=dev)
with torch.no_grad():
    tokens, out = dit.patch(x), dit(x, t)
print('images', tuple(x.shape), '-> tokens', tuple(tokens.shape),
      '-> output', tuple(out.shape))
print(f'parameters: DiT {d2l.num_params(dit) / 1e6:.2f}M, '
      f'U-Net {d2l.num_params(d2l.UNet(ch=64)) / 1e6:.2f}M')
print('output at initialization all zero:', bool((out == 0).all()))
```

```{.python .input #latent-diffusion-adaptive-layer-norm-3}
%%tab jax
dit = d2l.DiT(rngs=nnx.Rngs(0))  # width 256, depth 6, 4 heads, 4 x 4 patches
x, t = X[:8], jax.random.uniform(jax.random.key(5), (8,))
tokens, out = dit.patch(x), dit(x, t)
print('images', x.shape, '-> tokens', tokens.shape, '-> output', out.shape)
print(f'parameters: DiT {d2l.num_params(dit) / 1e6:.2f}M, U-Net '
      f'{d2l.num_params(d2l.UNet(ch=64, rngs=nnx.Rngs(0))) / 1e6:.2f}M')
print('output at initialization all zero:', bool((out == 0).all()))
```

The untrained output is exactly zero, as promised. The DiT has 7.39
million parameters, six times as many as the U-Net, yet the scaling run
below finds the two about equal in arithmetic per step. The difference lies
in how each uses its weights: a convolution applies a few weights at
every one of many positions, while the DiT applies many weights to only 49
tokens.

### Scaling with Compute

Is the transformer better than the U-Net? At one size the question has no
general answer, since a larger network of either kind beats a smaller one.
What matters is how each improves as it grows. :citet:`Peebles.Xie.2023`
trained DiTs of four sizes with three patch sizes each. At a fixed number of
training steps, FID fell steadily as the network's arithmetic per forward
pass grew, whether it came from a larger network or from smaller patches.
Their largest model, trained for seven million steps, set a new best FID on
class-conditional ImageNet at $256 \times 256$.

We repeat the measurement at our scale, for both backbones. Two DiTs grow
in width and depth together, with 64 dimensions per head as in most of the
paper's models: DiT-S (width 128, depth 4, two heads) and DiT-M, the model
above. Two U-Nets double their channels, from `ch=32` to the `ch=64` of the
pixel model above. Each is trained on pixels with the same loss and recipe
for 20,000 updates, twice from different seeds, and we measure its size by
the FLOPs of one training step.

```{.python .input #latent-diffusion-scaling-with-compute-1}
%%tab pytorch
sizes = {'DiT-S': dict(width=128, depth=4, num_heads=2),
         'DiT-M': dict(width=256, depth=6, num_heads=4)}
nets = {'DiT-S': lambda: d2l.DiT(**sizes['DiT-S']),
        'DiT-M': lambda: d2l.DiT(**sizes['DiT-M']),
        'U-Net 32': lambda: d2l.UNet(ch=32),
        'U-Net 64': lambda: d2l.UNet(ch=64)}
seeds = {'DiT-S': (5, 12), 'DiT-M': (6, 13), 'U-Net 32': (7, 14),
         'U-Net 64': (15,)}  # the first U-Net 64 is the pixel model above
runs = {name: dict(fd=[]) for name in nets}
runs['U-Net 64'].update(fd=[fd_pixel], imgs=x_pixel[:8])
for name, make_net in nets.items():
    for seed in seeds[name]:
        torch.manual_seed(seed)
        net = make_net().to(dev)
        _, ema, _ = d2l.train_regression(net, loss_fn, batch_fn, 20_000)
        imgs = sample(ema, image_shape)
        runs[name]['fd'].append(fd(imgs))
        runs[name].setdefault('imgs', imgs[:8])
    runs[name].update(params=d2l.num_params(net), gflops=d2l.train_step_gflops(
        net, loss_fn, image_shape))
for name, run in runs.items():
    print(f'{name:8} {run["params"] / 1e6:5.2f}M parameters '
          f'{run["gflops"]:4.0f} GFLOP per step   FD '
          + ', '.join(f'{v:.2f}' for v in run['fd']))
```

```{.python .input #latent-diffusion-scaling-with-compute-1}
%%tab jax
sizes = {'DiT-S': dict(width=128, depth=4, num_heads=2),
         'DiT-M': dict(width=256, depth=6, num_heads=4)}
nets = {'DiT-S': lambda rngs: d2l.DiT(**sizes['DiT-S'], rngs=rngs),
        'DiT-M': lambda rngs: d2l.DiT(**sizes['DiT-M'], rngs=rngs),
        'U-Net 32': lambda rngs: d2l.UNet(ch=32, rngs=rngs),
        'U-Net 64': lambda rngs: d2l.UNet(ch=64, rngs=rngs)}
seeds = {'DiT-S': (5, 12), 'DiT-M': (6, 13), 'U-Net 32': (7, 14),
         'U-Net 64': (15,)}  # the first U-Net 64 is the pixel model above
runs = {name: dict(fd=[]) for name in nets}
runs['U-Net 64'].update(fd=[fd_pixel], imgs=x_pixel[:8])
for name, make_net in nets.items():
    for seed in seeds[name]:
        net = make_net(nnx.Rngs(seed))
        _, ema, _ = d2l.train_regression(net, loss_fn, batch_fn, 20_000,
                                         jax.random.key(seed))
        imgs = sample(ema, image_shape)
        runs[name]['fd'].append(fd(imgs))
        runs[name].setdefault('imgs', imgs[:8])
    runs[name].update(params=d2l.num_params(net), gflops=d2l.train_step_gflops(
        net, loss_fn, image_shape))
for name, run in runs.items():
    print(f'{name:8} {run["params"] / 1e6:5.2f}M parameters '
          f'{run["gflops"]:4.0f} GFLOP per step   FD '
          + ', '.join(f'{v:.2f}' for v in run['fd']))
```

We plot the mean FD of the two runs of each model against its FLOPs per
step, on logarithmic axes.

```{.python .input #latent-diffusion-scaling-with-compute-2}
%%tab pytorch, jax
curves = [[runs['DiT-S'], runs['DiT-M']], [runs['U-Net 32'], runs['U-Net 64']]]
d2l.plot([[run['gflops'] for run in curve] for curve in curves],
         [[sum(run['fd']) / 2 for run in curve] for curve in curves],
         'GFLOP per training step', 'FD (5,000 samples)',
         legend=['DiT-S, DiT-M', 'U-Net ch=32, 64'], xscale='log',
         yscale='log', fmts=('o-', 'D-'))
```

Each difference below is at least twice the spread between the two runs of
each model it involves, so three readings are firm. The first is the
surprise. DiT-S has as many parameters as the U-Net with `ch=64`, 1.26
against 1.20 million, and three times its FD; DiT-M, with six times the
parameters, still trails it. On $28 \times 28$ garments the assumptions
built into a convolution are worth more than a sixfold increase in
parameters.

Measured by compute, the U-Net is still ahead at both levels, if by less.
At the small end the U-Net with `ch=32`, at about 50 GFLOP per step, scores
about 2.5, and DiT-S, at a third less arithmetic, 4.5 to 4.9. Near 200 GFLOP
the U-Net scores about 1.5 and DiT-M 1.7 to 1.9. Yet the transformer gains
more as it grows: 5.8 times the compute divides the DiT's FD by 2.6 to 2.8,
while four times the compute divides the U-Net's by about 1.65. Its line on
the plot is the steeper one.

The rows below show the first eight samples of DiT-S, DiT-M and the two
U-Nets, drawn from the same noise.

```{.python .input #latent-diffusion-scaling-with-compute-3}
%%tab pytorch
d2l.show_gray(torch.cat([run['imgs'] for run in runs.values()]), 4, 8);
```

```{.python .input #latent-diffusion-scaling-with-compute-3}
%%tab jax
d2l.show_gray(jnp.concatenate([run['imgs'] for run in runs.values()]), 4, 8);
```

The four models mostly draw the same garment in each column, since each
integrates an ODE that sends a noise image to one image; in a column or two
the garment itself changes. Little in these eight columns would tell us
that the top row scores three times the FD of the bottom one. The Fréchet
distance compares thousands of samples through a classifier's features; our
eyes compare eight.

Our four points cannot show the claim that made the transformer the backbone at
scale: that it keeps improving over orders of magnitude of compute, by a recipe
as simple as making it wider and deeper. That trend is Peebles and Xie's, not
ours. Our plot shows the steeper slope but no crossing (Exercise
:exref:`ex_latent-dit-large`). What makes the transformer attractive beyond the
trend is that its arithmetic runs as large matrix products, so the methods that
train language models at scale carry over. The horizontal axis, moreover, is
compute per update. Whether a large model trained for fewer updates beats a
small one trained for more, at equal total compute, is a separate question
(Exercise :exref:`ex_latent-total-compute`).

## Choice of Representation and Backbone

The two halves of the section meet in one last experiment. Does the
transformer help in the latent space? We train DiT-M on the latents with
patches of size 1, so that it sees 49 tokens, as it did on pixels, and
spends the same arithmetic per step, twelve to seventeen times that of the
latent U-Net.

```{.python .input #latent-diffusion-scaling-with-compute-4}
%%tab pytorch
torch.manual_seed(8)
net = d2l.DiT(img_size=7, patch_size=1, num_channels=4,
              **sizes['DiT-M']).to(dev)
_, dit_latent, _ = d2l.train_regression(net, loss_fn, latent_batch_fn, 20_000)
gflops = d2l.train_step_gflops(net, loss_fn, latent_shape)
print(f'latent DiT-M, {gflops:.0f} GFLOP per step: FD '
      f'{fd(decode(sample(dit_latent, latent_shape))):.2f}; latent U-Net '
      f'{fd_latent:.2f}; reconstructions {fd_rec:.2f}')
```

```{.python .input #latent-diffusion-scaling-with-compute-4}
%%tab jax
net = d2l.DiT(img_size=7, patch_size=1, num_channels=4, **sizes['DiT-M'],
              rngs=nnx.Rngs(8))
_, dit_latent, _ = d2l.train_regression(net, loss_fn, latent_batch_fn, 20_000,
                                        jax.random.key(8))
gflops = d2l.train_step_gflops(net, loss_fn, latent_shape)
print(f'latent DiT-M, {gflops:.0f} GFLOP per step: FD '
      f'{fd(decode(sample(dit_latent, latent_shape))):.2f}; latent U-Net '
      f'{fd_latent:.2f}; reconstructions {fd_rec:.2f}')
```

The latent DiT-M scores an FD of about 2 (1.8 to 2.2), about the same as
the latent U-Nets (1.9 to 2.2), and like them it lands at about three times
the reconstruction FD. Two KL weights and two backbones have now left the
latent samples far above their reconstructions; whether more compute on the
latents closes the gap is Exercise :exref:`ex_latent-scaling-in-latents`.
Nor does the latent help the transformer here, since at equal compute the
latent DiT-M is no better than the pixel one, and the best pixel model beats
every latent model of the section.

When, then, is a latent worth its autoencoder? Its saving grows with the
number of pixels, and its costs, the autoencoder's distortion and a harder
modeling problem, do not. At $512 \times 512$ a U-Net on Stable Diffusion's
latent works on 64 times fewer positions, and its arithmetic falls by about
that factor: our pixel U-Net needed 13 to 18 times the arithmetic of the
latent one for sixteen times as many positions. The saving applies at every
training update and at every one of fifty sampling steps, and at that size
it outweighs the autoencoder's costs.

For a transformer the arithmetic is different. With patches of size 2 on the
$64 \times 64 \times 4$ latent a DiT sees 1,024 tokens; with patches of size 16
on the pixels it sees the same 1,024 tokens at almost the same cost (Exercise
:exref:`ex_latent-compression-arithmetic`). The autoencoder saves the
transformer no tokens. It replaces a linear embedding of $16 \cdot 16 \cdot 3 =
768$ raw pixel values by a learned nonlinear one. A linear embedding of that
many values copes badly with noise, which fills all 768 dimensions while the
image lies near a set of much lower dimension. Two recent designs build pixel
transformers around this problem. :citet:`Li.He.2026` keep the large patches
and predict the clean image rather than the noise or the velocity (Exercise
:exref:`ex_latent-large-patches`), and hourglass transformers
:cite:`Crowson.Baumann.Birch.ea.2024` take the U-Net's multi-resolution shape
and attend only locally at the fine levels, so that their cost grows only
linearly with the pixels.

Our advice follows from all this. For small images, or whenever a pixel
model fits the budget, diffuse pixels, and at small compute use a U-Net,
which was ahead at every compute we tried. For large images, diffuse
latents with a small KL weight, and treat the autoencoder as part of the
model rather than as preprocessing. It decides both what a perfect latent
model would score and how hard the latents are to learn; neither a larger
backbone nor a more tolerant decoder brought our samples close to the
reconstructions. Choose a transformer when the budget is large enough to
keep growing it, the regime in which Peebles and Xie measured. Because the
autoencoder shapes both numbers whatever the backbone, the design of image
autoencoders, often called *tokenizers*, has become a field of its own.

Latents and transformers make each network evaluation cheaper. Neither
reduces the number of evaluations, fifty per sample in this section.

## Summary

Diffusion does not need to run on pixels. An autoencoder with a small KL term
compresses each image into a latent of known scale, the regression of the
earlier sections runs on the latents unchanged, and the decoder turns sampled
latents back into images, at a fraction of the arithmetic per step. The
decoder's reconstructions are what a perfect latent model would score; on
Fashion-MNIST the latent samples stayed far above them, and neither a more
tolerant decoder nor a larger backbone closed the gap. The gap is the latent
model's own error, but the encoder decides how hard the latents are to learn.
The diffusion transformer reads latents or pixels as patch tokens and lets the
condition set a scale, a shift and a gate in every block, starting from the
identity. At our scale the U-Net beat it at equal compute, while the
transformer's FD fell faster as it grew. A latent helps when images are large,
a transformer when budgets are.

## Exercises

1. **The scale of the latents.** The linear path mixes the data with noise
   of unit variance.
    1. Suppose each latent value has standard deviation $\varsigma$ instead
       of one. Show that its signal-to-noise ratio at time $t$ is
       $\varsigma^2 (1 - t)^2 / t^2$, so that the log-SNR is shifted by
       $\log \varsigma^2$ at every time.
    1. At which time does a latent with $\varsigma = 5$ have the
       signal-to-noise ratio that a latent with $\varsigma = 1$ has at
       $t = \tfrac12$? What fraction of uniformly drawn training times then
       falls where the signal still dominates the noise?
    1. [code] Train the latent U-Net on latents multiplied by $5c$ instead
       of $c$, and compare the decoded FD. Relate the result to how the
       schedule shifts with resolution
       (:numref:`subsec_diffusion-design-space-scales`).
   :exlabel:`ex_latent-scale-factor`
1. **Tokens at high resolution.** The autoencoder of Stable Diffusion maps
   a $512 \times 512 \times 3$ image to $64 \times 64 \times 4$.
    1. How many tokens does a DiT see with patches of size 2 on the latent,
       and with patches of size 16 on the pixels? How many numbers does each
       token carry in the two cases?
    1. A DiT block of width $d$ has about $12 d^2$ weights, and a training
       step costs about six FLOPs per weight and token, plus the attention.
       Estimate the FLOPs of one training step on a batch of 256 for a DiT
       of width 1,152 and depth 28 on the latents with patch 2. How large is
       the attention part?
    1. Repeat the estimate for patch 16 on the pixels. How much of the
       step's cost does the autoencoder save the transformer, and how much
       does it save a U-Net, whose cost grows with the number of positions?
   :exlabel:`ex_latent-compression-arithmetic`
1. [code] **Width and weight of the autoencoder.** The rate and the KL
   weight are the two settings of the autoencoder.
    1. Train autoencoders with 2, 4 and 8 latent channels, and report the
       reconstruction FD of each and the decoded FD of a latent U-Net
       trained on each. Which of the two numbers moves more, and does the
       gap between them close as the latent grows? What does this suggest
       about the four channels of Stable Diffusion's autoencoder, and the 16
       or more of later ones?
    1. Extend the perturbed-latent test of this section to
       $\beta \in \{0, 10^{-2}\}$, and print the latent standard deviation,
       the encoder's standard deviation and the excess kurtosis as well.
       With $\beta = 0$ nothing ties the scale of the latents to one. Why
       does the scale factor $c$ still make them usable?
    1. Train latent U-Nets on both new autoencoders. Does the ordering of
       the text, a smaller KL weight giving better samples, continue to
       $\beta = 0$?
   :exlabel:`ex_latent-bottleneck`
1. [code] **Why start at zero.** Replace the zero initialization of the
   `ada` layers in `d2l.DiTBlock` and `d2l.DiT` by the framework's default
   initialization.
    1. Print the size of the output of the untrained network for both
       initializations.
    1. Train both DiT-M models for 2,000 updates and plot their losses.
       Explain the difference through the identity at initialization.
   :exlabel:`ex_latent-zero-gate`
1. **Cross-attention as a lookup.** Image tokens
   $\mathbf{Y} \in \mathbb{R}^{m \times d}$ query caption tokens
   $\mathbf{X} \in \mathbb{R}^{n \times d}$: the queries are
   $\mathbf{Q} = \mathbf{Y}\mathbf{W}_q$, the keys
   $\mathbf{K} = \mathbf{X}\mathbf{W}_k$ and the values
   $\mathbf{V} = \mathbf{X}\mathbf{W}_v$, and the layer outputs
   $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$.
    1. Show that each image position receives a convex combination of the
       caption's value vectors. With an empty caption of one token, show that
       every position receives the same vector, so that the unconditional
       prediction of guidance sees cross-attention as a learned constant
       shift.
    1. Suppose that the values were computed from the image instead,
       $\mathbf{V} = \mathbf{Y}\mathbf{W}_v$. What goes wrong with the shapes
       when $m \neq n$, and what would such a layer compute when $m = n$?
    1. In cross-attention the caption's keys and values do not depend on
       $\mathbf{x}_t$; in a multimodal DiT the caption tokens are updated by
       the image tokens in every block. For a caption of 77 tokens, 1,024
       image tokens and 50 steps with guidance, which caption-side
       computations can a cross-attention model do once per sample, and
       what fraction of a multimodal DiT's token computations belongs to the
       caption?

    *Part (b) is adapted from CMU 10-423/623,
    [Generative AI](https://www.cs.cmu.edu/~mgormley/courses/10423/),
    Homework 4, Question 2.3 (Spring 2026).*
   :exlabel:`ex_latent-cross-attention`
1. [code] **Compute per step, in total and from patches.** ● The
   horizontal axis of the scaling plot is compute per update.
    1. Score DiT-S, DiT-M and the two U-Nets after 5,000, 10,000 and 20,000
       updates, and plot FD against total training compute, the FLOPs per
       step times the number of updates, with one curve per model. At equal
       total compute, does the larger model of each backbone reach the
       lower FD? Compare with the plot of FD against total training compute
       in the paper of Peebles and Xie, and say what longer runs could
       change.
    1. Train DiT-M with patches of size 2 on the pixels for 20,000 updates.
       How many tokens does it see, and how many FLOPs per step does
       `train_step_gflops` count? Is the increase over patch 4 what the cost
       of attention predicts?
    1. Where does the patch-2 model fall on the DiT line of the scaling
       plot? Does FD depend on the compute per step alone, whatever
       produces it?
   :exlabel:`ex_latent-total-compute`
1. [code] **A larger transformer.** ● Add DiT-L (width 384, depth 8, six
   heads, so 64 dimensions per head as for DiT-S and DiT-M) to the scaling
   run, trained for the same 20,000 updates. It spends more arithmetic per
   step than any model of the section.
    1. Print its parameters, its GFLOP per training step and its FD, and add
       it to the plot. How much more compute per step does it spend than the
       U-Net with `ch=64`?
    1. Train the U-Net with `ch=64` and DiT-L once more each from another
       seed. Is the difference between their FDs larger than the spread
       between the two runs of each?
    1. Fit a straight line on the logarithmic axes through the two DiT
       points of the text, and one through the two U-Net points. Where
       would the lines cross if they kept their slopes?
    1. Train U-Nets with `ch=96` and `ch=128` as well. Which of the two
       lines falls faster with compute per step over the range you can
       afford, and what would the plot have to show to support the claim
       that the transformer keeps improving where the U-Net does not?
   :exlabel:`ex_latent-dit-large`
1. [code] **Scaling in the latent space.** ● Train DiT-S on the latents as
   DiT-M was trained above, and also DiT-L if you have done Exercise
   :exref:`ex_latent-dit-large` already.
    1. Add the latent models to the plot of FD against FLOPs per step, with
       the reconstruction FD as a horizontal line. Does the latent line keep
       falling with size, and how close does it come to the horizontal one?
    1. The perturbed-latent test placed the gap in the latent model rather
       than in the decoder. Does your line agree? What would you change to
       move it down: the backbone, the length of training, or the
       autoencoder?
   :exlabel:`ex_latent-scaling-in-latents`
1. [code] **Large patches and the clean-image target.** ● Train `DiT` on
   pixels with `patch_size=7`, so that each of its 16 tokens carries 49
   pixel values, at width 32 with one head and at width 256, for 10,000
   updates each. Give each width two output parameterizations under the
   same velocity loss: the network outputs the velocity, or it outputs
   $\hat{\mathbf{x}}_0$ and the velocity is formed as
   $(\mathbf{x}_t - \hat{\mathbf{x}}_0)/t$, with $t$ clamped below at
   $0.05$.
    1. Report the FD of the four models in a $2 \times 2$ table.
    1. Does the clean-image output help at width 32, where a token carries
       more values than the network has channels, and does the gap close at
       width 256?
    1. Explain the result with the argument of
       :numref:`sec_diffusion-design-space`: an image lies near a set of
       much lower dimension than its pixel count, while the noise fills
       every dimension. Compare with the finding of :citet:`Li.He.2026`.
   :exlabel:`ex_latent-large-patches`
1. [extended] **Latent diffusion for sprites.** ●● Adapt this section's
   autoencoder to the $64 \times 64$ color sprites of :numref:`sec_dcgan`,
   which gives a $16 \times 16 \times 4$ latent, and train a latent U-Net on
   it. Score the samples with the `d2l.FeatureCNN` of :numref:`sec_dcgan`
   and compare them with the GANs trained there. Which of the two methods
   needs more compute for the same FD?
   :exlabel:`ex_latent-sprites`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.7]{.kicker}

Latent Diffusion and Transformers<br>
**the cost of pixels · a Gaussian latent · the same regression on latents · where the latent gap comes from · patch tokens and adaLN-Zero · U-Net against DiT · when a latent helps**
:::
:::

::: {.slide title="A 512 × 512 Image Has 786,432 Numbers, and Every Step Touches All"}
[The problem]{.kicker}

- Our U-Net reads all 784 pixels of a garment at each of fifty steps.
- A $512 \times 512$ photograph: a thousand times the numbers.
- Most of them are texture: the sampler fixes the silhouette early and the
  texture last.

::: {.d2l-note}
Why spend diffusion arithmetic on detail a viewer hardly notices?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- train a variational autoencoder whose latent has a known scale;
- run the velocity regression on latents and count what it saves;
- test whether the latent model or the decoder causes the latent gap;
- build a transformer block that a time and a label steer;
- decide when a latent and a transformer help.

**Running example:** Fashion-MNIST, $28 \times 28$ images and a
$7 \times 7 \times 4$ latent, scored by FD on 5,000 samples.
:::

::: {.slide title="This Deck Moves Diffusion onto Latents and Swaps the Backbone"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Denoising and the Score | what a denoiser knows |
| Gaussian Paths | which walk and which regression |
| Sampling by Integration | how to follow the field |
| Diffusion for Images | images: steer and score them |
| Samplers, Inversion and Editing | fewer steps; run backwards |
| Likelihood and the Design Space | which choices matter |
| **Latent Diffusion and Transformers** | **a compressed code, a transformer** |
| Few-Step Generation | one or two evaluations |
| Discrete Diffusion | noise for tokens |
:::

::: {.col .narrow}
- **Uses:** the linear path $\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol\epsilon$
  (data at $t=0$), the velocity loss, the U-Net, FD.
- **Skips:** perceptual and adversarial autoencoder losses, text encoders,
  video.
:::
:::
:::

::: {.slide title="An Autoencoder Compresses Perceptually; Diffusion Models What the Image Shows"}
![](../img/mdl-diffusion-latent-pipeline.svg){width=80%}

Ours: 784 numbers to 196. Stable Diffusion's: 786,432 to 16,384, 48 times
fewer. The decoder runs once per sample.

::: {.d2l-note}
The trade: **rate** (numbers in the latent), **distortion** (what the
decoder loses), **modelability** (how well diffusion learns the latents).
:::

[Rombach et al., 2022; Dieleman, 2025]{.small}
:::

::: {.slide title="A Small KL Term Gives the Latent a Known Scale"}
Encoder: mean $\boldsymbol\mu$, std $\mathbf{s}$;
$\mathbf{z} = \boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi$,
$\boldsymbol\xi \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$; decoder
$\mathbf{z} \mapsto \hat{\mathbf{x}}$.

$$\mathcal{L}(\mathbf{x}) = \tfrac{1}{784}\|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \tfrac{\beta}{196} \textstyle\sum_{j} \tfrac12\big(\mu_j^2 + s_j^2 - 1 - \log s_j^2\big)$$

- Pull toward $\mathcal{N}(0, 1)$: known scale, no outliers.
- Noise $\mathbf{s}$ in training: nearby codes decode alike.

::: {.d2l-note}
$\beta = 10^{-4}$: scale fixed, but far too weak to make neighbors decode
alike.
:::
:::

::: {.slide title="Reconstructions Score 0.6 to 0.8 Against a Floor of 0.17"}
@!latent-diffusion-the-autoencoder-2

@!latent-diffusion-the-autoencoder-4

FD: distance between Gaussians fitted to classifier features of 5,000
images; two disjoint real sets score 0.17. A perfect latent model would
score the reconstructions' FD.
:::

::: {.slide title="The Latents Are Close to Normal: Multiply by c ≈ 1.05"}
@!latent-diffusion-the-autoencoder-5

- Standard deviation slightly below one: scale by $c$ of about 1.05, to
  match the path's unit-variance noise.
- Excess kurtosis near zero or negative: no heavy tails.
- Encoder std 0.02 to 0.03: in training the decoder saw codes blurred by
  only 3% of their spread.
:::

::: {.slide title="Check: Why Does Stable Diffusion Multiply Its Latents by 0.18215?"}
[Check]{.kicker}

Stable Diffusion multiplies its latents by 0.18215 before diffusing them.
What standard deviation did the raw latents have, and what would skipping
the factor do on a path that adds noise of unit variance?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** About $1 / 0.18215 \approx 5.5$. The signal's variance would
be about 30 times the intended one at every $t$, a log-SNR shift of about
$\log 30 \approx 3.4$: every time would look far less noisy than the
schedule intends.
:::
:::
:::

::: {.slide title="On Latents the Regression Is Unchanged"}
- Each batch: $\mathbf{z}_0 = c\,(\boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi)$
  for 128 training images, a fresh draw from the encoder.
- Linear path $\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\boldsymbol\epsilon$;
  regress onto the velocity $\boldsymbol\epsilon - \mathbf{z}_0$.
- Same loss function, same U-Net with four channels in and out, 20,000
  updates. Reference: the same U-Net on pixels.
:::

::: {.slide title="The Latent Model Is Far Cheaper and Somewhat Worse"}
@!latent-diffusion-the-same-regression-on-latents-3

@!latent-diffusion-the-same-regression-on-latents-4

Top: pixel U-Net; bottom: decoded latent U-Net. FD about 2 against about
1.5, at under a tenth of the arithmetic per step.

::: {.d2l-note}
Latent samples: about three times the reconstructions' FD. Model error, or
a decoder that cannot handle slightly wrong codes?
:::
:::

::: {.slide title="Check: What Should Ten Times the KL Weight Do?"}
[Check]{.kicker}

A second autoencoder is trained with $\beta = 10^{-3}$. We decode real
latent means plus noise of $\sigma$ times the latents' spread. Predict:
which decoder tolerates $\sigma = 0.5$ better, and which reconstructs
better?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The tenfold weight blurs codes by about a fifth of the spread
in training, so its decoder tolerates noise better, and it reconstructs
worse.
:::
:::
:::

::: {.slide title="A Tolerant Decoder Turns Wrong Codes into Plausible Garments"}
@!latent-diffusion-reading-the-result-2

- $\sigma = 0.5$: FD above 3 (weak KL) against about 1.3 (tenfold).
- Pixel change about the same (0.053 to 0.055 against 0.061 to 0.062):
  wrong codes become other plausible garments.
- Small noise lowers FD below the reconstructions': a reference, not a
  bound.
:::

::: {.slide title="Yet the Tolerant Decoder Gives Worse Samples"}
@!latent-diffusion-reading-the-result-3

- Every tenfold run worse by about half a point (0.4 to 0.8), at least
  2.5 times the seed spread.
- If intolerance caused the gap, the tolerant decoder would have narrowed
  it. It did not.
- Noise of half the spread costs the tolerant decoder FD 1.3; its latent
  models score 2.6 to 2.7.

::: {.d2l-note}
The gap is the latent model's own error, and the encoder decides how hard
its latents are to learn. Keep the KL weight small.
:::
:::

::: {.slide title="A DiT Reads an Image as a Sequence of Patch Tokens"}
- $p \times p$ patches, a linear map to width $d$, learned positions,
  pre-norm transformer blocks.
- $p = 4$ on $28 \times 28$: 49 tokens of 16 pixels; $p = 1$ on our
  latent: 49 tokens of 4 numbers.
- A final linear layer maps each token to its patch's velocity.

::: {.d2l-note}
One token per pixel: 16 times the tokens, 256 times the attention.
:::

[Peebles and Xie, 2023]{.small}
:::

::: {.slide title="The Condition Sets a Scale, a Shift and a Gate per Sublayer"}
$$
\begin{aligned}
\mathbf{X} &\leftarrow \mathbf{X} + \mathbf{g}_1 \odot \mathrm{Attention}\big(\mathrm{LN}(\mathbf{X}) \odot (1 + \mathbf{a}_1) + \mathbf{b}_1\big)\\
\mathbf{X} &\leftarrow \mathbf{X} + \mathbf{g}_2 \odot \mathrm{MLP}\big(\mathrm{LN}(\mathbf{X}) \odot (1 + \mathbf{a}_2) + \mathbf{b}_2\big)
\end{aligned}
$$

- $\mathbf{c}$ = time embedding + label embedding; $\mathrm{LN}$ has no
  scale or shift of its own.
- Scales $\mathbf{a}$, shifts $\mathbf{b}$, gates $\mathbf{g}$: one linear
  layer on $\mathrm{SiLU}(\mathbf{c})$, shared by all tokens.

::: {.d2l-note}
The U-Net added one shift per channel; adaptive layer norm adds a scale and
a gate.
:::
:::

::: {.slide title="adaLN-Zero: Every Block Starts as the Identity"}
::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-diffusion-dit-block.svg)
:::

::: {.col}
- The linear layer from $\mathbf{c}$ starts at zero: every gate is zero,
  every block the identity.
- The final layer starts at zero: zero velocity everywhere.
- Our DiT: 7.39M parameters, six times the U-Net's 1.20M, at about the
  same GFLOP per step; untrained output all zero.
- Best of four conditioning designs in Peebles and Xie.
:::
:::
:::

::: {.slide title="Check: What If Only the Blocks Start at Zero?"}
[Check]{.kicker}

The blocks' scale-shift-gate maps and the final modulation start at zero,
but the final linear layer does not. What does the untrained DiT output?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Every block is the identity, so the output is the final layer
applied to the normalized patch embeddings plus positions: nonzero, and the
same at every time $t$.
:::
:::
:::

::: {.slide title="The U-Net Is Ahead at Equal Parameters and at Equal Compute"}
@!latent-diffusion-scaling-with-compute-1

Two runs per model, pixels, 20,000 updates. Near 50 GFLOP: U-Net about
2.5; DiT-S, at a third less, 4.5 to 4.9. Near 200: U-Net about 1.5, DiT-M
1.7 to 1.9.

::: {.d2l-note}
Equal parameters (DiT-S, U-Net `ch=64`): three times the FD for the DiT.
DiT-M has six times the U-Net's parameters and still trails it.
:::
:::

::: {.slide title="The Transformer's Line Is Steeper; the Crossing Is Not in Our Range"}
::: {.cols .vc}
::: {.col .fig}
@!latent-diffusion-scaling-with-compute-2
:::

::: {.col}
- DiT: 5.8 times the compute divides FD by 2.6 to 2.8.
- U-Net: 4 times the compute divides FD by about 1.65.
- Improvement over orders of magnitude: Peebles and Xie's result, not
  ours.
:::
:::

[Horizontal axis: compute per update, not in total.]{.small}
:::

::: {.slide title="On Latents, Twelve to Seventeen Times the Arithmetic Changes Little"}
@!latent-diffusion-scaling-with-compute-4

- Latent DiT-M about 2 (1.8 to 2.2), about the same as the latent U-Nets
  (1.9 to 2.2), at 12 to 17 times their GFLOP.
- Both near three times the reconstruction FD.

::: {.d2l-note}
At $28 \times 28$ the best pixel model beats every latent one.
:::
:::

::: {.slide title="Latents Save a U-Net About 64 Times the Work, a Transformer No Tokens"}
At $512 \times 512 \times 3$ with Stable Diffusion's $64 \times 64 \times 4$
latent:

| | U-Net | DiT |
|---|---|---|
| pixels | 262,144 positions | patch 16: 1,024 tokens |
| latent | 4,096 positions | patch 2: 1,024 tokens |

::: {.d2l-note}
For a transformer the autoencoder replaces a linear embedding of 768 pixel
values. Pixel transformers built around this: predict the clean image (Li
and He), or attend locally at fine levels (hourglass).
:::

[Li and He, 2026; Crowson et al., 2024]{.small}
:::

::: {.slide title="Check: How Expensive Is a 1024 × 1024 Image?"}
[Check]{.kicker}

Same autoencoder as Stable Diffusion (eightfold per side, four channels),
$2 \times 2$ patches on the latent. How many tokens, and how does attention
cost compare with $512 \times 512$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Latent $128 \times 128 \times 4$, so $64^2 = 4{,}096$ tokens:
four times as many, and sixteen times the attention cost.
:::
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **Autoencoder:** 784 numbers to 196; reconstructions FD 0.6 to 0.8,
  floor 0.17; latents scaled by $c \approx 1.05$.
- **Latent regression:** FD about 2 against about 1.5 on pixels, at under
  a tenth of the GFLOP per step.
- **The gap:** the latent model's own error, on latents the encoder makes
  easy or hard to learn; a tenfold KL weight made samples worse by about
  half a point.
- **Backbones:** U-Net ahead at about 50 and 200 GFLOP; the DiT's line
  steeper.
- **Choose:** pixels and a U-Net when small; latents when large; a
  transformer when the budget keeps growing.

**Next:** fifty evaluations per sample remain fifty. Can one or two do?
:::
