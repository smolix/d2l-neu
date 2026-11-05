# Adversarial Image Generation
:label:`sec_dcgan`

Everything in this chapter so far happened in two dimensions, where a scatter plot shows at a glance whether the generator's samples land on the data. GANs became famous for something else. In 2015 a generator trained on photographs of bedrooms produced bedrooms that no camera had taken :cite:`Radford.Metz.Chintala.2015`, and within a few years generated faces were hard to tell from photographs. Images change two things. The generator now has to turn a vector of 100 random numbers into $64 \times 64 \times 3 = 12{,}288$ pixel values, so its architecture matters. And we can no longer look at the samples and see whether $q$ matches $p$: a grid of images reveals a collapse, but two good generators look alike to the eye.

This section takes both questions in turn on a dataset of 40,000 small game sprites. We first build the 2015 network, which made the original loss train on images through its architecture alone. We then build a 2024 backbone with no normalization at all and train it three ways that differ in one thing at a time: the classic loss without a penalty, the classic loss with the two gradient penalties of :numref:`sec_gan_convergence`, and the pairing loss of :numref:`sec_gan_relativistic` with the same penalties. To score the runs we need a number that compares two sets of images. The two standard scores turn out to be closed forms we have met before: FID is the Wasserstein distance between two Gaussians, and KID is a maximum mean discrepancy with a fixed kernel.

```{.python .input #dcgan-adversarial-image-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
import torchvision
import warnings
from torch import nn
from torch.nn import functional as F
```

```{.python .input #dcgan-adversarial-image-generation}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import numpy as np
import optax
import os
import warnings
from PIL import Image
```

## The 2015 Recipe

The original GAN was hard to train on images of any size. The deep convolutional GAN (DCGAN) of :citet:`Radford.Metz.Chintala.2015` kept the objective and changed the networks. Its generator upsamples with transposed convolutions and its discriminator downsamples with strided convolutions. Both networks use batch normalization, except at the discriminator's input and the generator's output. The generator uses ReLU and ends in a tanh, which matches its range to images scaled to $[-1, 1]$. The discriminator uses the leaky ReLU, which passes $0.2x$ instead of zero for negative $x$, so that no unit stops passing gradient to the generator. Adam runs with momentum $\beta_1 = 0.5$ instead of the usual 0.9, which shortens the optimizer's memory of a gradient field that changes as the other player moves. Batch normalization keeps the activation scales in range while the weights change. The loss is still the non-saturating log loss of :numref:`sec_basic_gan`.

### The Sprite Dataset

The dataset contains 40,597 sprite images of 721 Pokemon species, with many variants of each, obtained from [pokemondb](https://pokemondb.net/sprites). Sprites suit a from-scratch experiment: they are small, centered on clean white backgrounds, and diverse in silhouette and palette. We resize each image to $64 \times 64$ with bilinear resampling and scale its pixel values to $[-1, 1]$, the range of the 2015 generator's tanh output. The whole dataset fits in memory, so we decode it once into one tensor and draw minibatches by indexing.

```{.python .input #dcgan-the-pokemon-sprites-1}
%%tab pytorch
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
warnings.filterwarnings('ignore', message='Palette images')
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.PILToTensor()])
pokemon = torchvision.datasets.ImageFolder(data_dir, transform=transformer)
loader = torch.utils.data.DataLoader(
    pokemon, batch_size=512, num_workers=d2l.get_dataloader_workers())
images = torch.cat([X for X, _ in loader]).float() / 127.5 - 1
images.shape
```

```{.python .input #dcgan-the-pokemon-sprites-1}
%%tab jax
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
warnings.filterwarnings('ignore', message='Palette images')
files = sorted(os.path.join(root, f)
               for root, _, names in os.walk(data_dir) for f in names
               if f.lower().endswith(('.png', '.jpg', '.jpeg')))
def load_image(path):
    img = Image.open(path).convert('RGB').resize((64, 64), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 127.5 - 1
images = np.stack([load_image(f) for f in files])
images.shape
```

A fixed permutation sets aside one tenth of the images as a held-out set that is never used for training. We use it twice: to check whether the critic scores its training images higher than unseen ones, and as the real sample against which generated images are scored. The split is by image rather than by species, so variants of one creature can appear on both sides.

```{.python .input #dcgan-the-pokemon-sprites-2}
%%tab pytorch
device = d2l.try_gpu()
perm = np.random.RandomState(12345).permutation(len(images))
n_holdout = len(images) // 10
train_imgs = images[perm[n_holdout:]].to(device)
holdout_imgs = images[perm[:n_holdout]].to(device)
print(f'{len(train_imgs)} training and {len(holdout_imgs)} held-out images')
d2l.show_images(train_imgs[:20].cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=4, num_cols=5);
```

```{.python .input #dcgan-the-pokemon-sprites-2}
%%tab jax
perm = np.random.RandomState(12345).permutation(len(images))
n_holdout = len(images) // 10
train_imgs = jnp.asarray(images[perm[n_holdout:]])
holdout_imgs = jnp.asarray(images[perm[:n_holdout]])
print(f'{len(train_imgs)} training and {len(holdout_imgs)} held-out images')
d2l.show_images(np.asarray(train_imgs[:20]) / 2 + 0.5,
                num_rows=4, num_cols=5);
```

### Transposed Convolutions

The generator must grow a $1 \times 1$ input with 100 channels into a $64 \times 64$ image. A *transposed convolution* grows a feature map by running a convolution's arithmetic backwards: each input pixel stamps a copy of the kernel, scaled by the pixel's value, onto the output at positions one stride apart, and where stamps overlap their values add. With a given kernel size, stride, and padding (rows and columns trimmed from each border), an $n \times n$ input produces an output of side

$$\textrm{stride} \cdot (n - 1) + \textrm{kernel} - 2 \cdot \textrm{pad}.$$

In one dimension, the input $(1, 2)$ and a kernel of four ones with stride 2 give six outputs: the two stamps $(1, 1, 1, 1)$ and $(2, 2, 2, 2)$ start two positions apart and overlap in the middle, so the output is $(1, 1, 3, 3, 2, 2)$. With kernel 4, stride 2, and padding 1 the side is exactly $2n$, so each layer doubles the resolution. With stride 1 and no padding the same kernel maps a $1 \times 1$ input to $4 \times 4$, which gives the generator its first spatial map.

```{.python .input #dcgan-generator-2}
%%tab pytorch
stamp = nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, bias=False)
nn.init.ones_(stamp.weight)
print(stamp(torch.tensor([[[1.0, 2.0]]])).detach())
up = nn.ConvTranspose2d(3, 20, kernel_size=4, stride=2, padding=1)
print(up(torch.zeros(2, 3, 16, 16)).shape)
```

```{.python .input #dcgan-generator-2}
%%tab jax
stamp = nnx.ConvTranspose(1, 1, (4,), strides=(2,), padding='VALID',
                          use_bias=False, kernel_init=nnx.initializers.ones,
                          rngs=nnx.Rngs(0))
print(stamp(jnp.array([[[1.0], [2.0]]]))[0, :, 0])
up = nnx.ConvTranspose(3, 20, (4, 4), strides=(2, 2), padding='SAME',
                       rngs=nnx.Rngs(0))
print(up(jnp.zeros((2, 16, 16, 3))).shape)
```

### Generator and Discriminator

The generator chains five transposed convolutions through the resolutions $1 \to 4 \to 8 \to 16 \to 32 \to 64$, halving the channel count at every doubling, and the discriminator mirrors it with strided convolutions that halve the resolution down to $4 \times 4$ and a final $4 \times 4$ convolution that returns one logit (:numref:`fig_gan_dcgan`). A strided convolution with the same kernel 4, stride 2, and padding 1 maps side $n$ to $n/2$. Each block is a convolution, batch normalization, and an activation; batch normalization is left out where Radford et al. leave it out, at the discriminator's input and the generator's output.

![The DCGAN architecture :cite:`Radford.Metz.Chintala.2015`. The generator maps a 100-dimensional latent vector through transposed $4 \times 4$ convolutions to feature maps of side $4, 8, 16, 32$ and a $64 \times 64$ RGB image. The discriminator reverses the schedule with strided $4 \times 4$ convolutions down to one logit. Every block uses batch normalization except the two shaded ones: the generator's output layer (tanh) and the discriminator's input block (leaky ReLU).](../img/mdl-gan-dcgan.svg)
:label:`fig_gan_dcgan`

```{.python .input #dcgan-generator-1}
%%tab pytorch
def G_block(c_in, c_out, stride=2, padding=1):
    """Transposed convolution (kernel 4), batch norm, ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride, padding, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU())

def D_block(c_in, c_out, norm=True):
    """Strided convolution (kernel 4), batch norm, leaky ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(c_out) if norm else nn.Identity(), nn.LeakyReLU(0.2))
```

```{.python .input #dcgan-generator-1}
%%tab jax
conv_init = nnx.initializers.normal(0.02)  # DCGAN: kernels from N(0, 0.02^2)

def bn_scale_init(key, shape, dtype=jnp.float32):
    return 1 + 0.02 * jax.random.normal(key, shape, dtype)  # N(1, 0.02^2)

def batch_norm(c, rngs):
    # Running averages decay 0.9; the variance is computed in two passes
    return nnx.BatchNorm(c, momentum=0.9, use_fast_variance=False,
                         scale_init=bn_scale_init, rngs=rngs)

def G_block(c_in, c_out, rngs, stride=2, padding='SAME'):
    """Transposed convolution (kernel 4), batch norm, ReLU."""
    return nnx.Sequential(
        nnx.ConvTranspose(c_in, c_out, (4, 4), strides=(stride, stride),
                          padding=padding, use_bias=False,
                          kernel_init=conv_init, rngs=rngs),
        batch_norm(c_out, rngs), nnx.relu)

def D_block(c_in, c_out, rngs, norm=True):
    """Strided convolution (kernel 4), batch norm, leaky ReLU."""
    return nnx.Sequential(
        nnx.Conv(c_in, c_out, (4, 4), strides=(2, 2), padding='SAME',
                 use_bias=False, kernel_init=conv_init, rngs=rngs),
        batch_norm(c_out, rngs) if norm else (lambda x: x),
        lambda x: nnx.leaky_relu(x, 0.2))
```

```{.python .input #dcgan-generator-3}
%%tab pytorch
torch.manual_seed(0)
net_G = nn.Sequential(                                   # (100, 1, 1)
    G_block(100, 512, stride=1, padding=0),              # (512, 4, 4)
    G_block(512, 256), G_block(256, 128), G_block(128, 64),  # (64, 32, 32)
    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()).to(device)
net_D = nn.Sequential(                                   # (3, 64, 64)
    D_block(3, 64, norm=False),                          # (64, 32, 32)
    D_block(64, 128), D_block(128, 256), D_block(256, 512),  # (512, 4, 4)
    nn.Conv2d(512, 1, 4, bias=False)).to(device)         # (1, 1, 1)
X = net_G(torch.zeros(2, 100, 1, 1, device=device))
print(X.shape, net_D(X).shape)
```

```{.python .input #dcgan-generator-3}
%%tab jax
rngs = nnx.Rngs(0)
net_G = nnx.Sequential(                                  # (1, 1, 100)
    G_block(100, 512, rngs, stride=1, padding='VALID'),  # (4, 4, 512)
    G_block(512, 256, rngs), G_block(256, 128, rngs),
    G_block(128, 64, rngs),                              # (32, 32, 64)
    nnx.ConvTranspose(64, 3, (4, 4), strides=(2, 2), padding='SAME',
                      use_bias=False, kernel_init=conv_init, rngs=rngs),
    nnx.tanh)                                            # (64, 64, 3)
net_D = nnx.Sequential(                                  # (64, 64, 3)
    D_block(3, 64, rngs, norm=False),                    # (32, 32, 64)
    D_block(64, 128, rngs), D_block(128, 256, rngs),
    D_block(256, 512, rngs),                             # (4, 4, 512)
    nnx.Conv(512, 1, (4, 4), padding='VALID', use_bias=False,
             kernel_init=conv_init, rngs=rngs))          # (1, 1, 1)
X = net_G(jnp.zeros((2, 1, 1, 100)))
print(X.shape, net_D(X).shape)
```

### Training with the Classic Loss

The update functions of :numref:`sec_basic_gan` apply to images unchanged. `d2l.update_D` takes one step of logistic regression with label one for data and label zero for generated samples, and `d2l.update_G` takes one step on the non-saturating generator loss $-\mathbb{E}_{z}[\log \sigma(D(G(z)))]$; `update_D` returns half the summed loss over the real and the generated batch, and `update_G` the summed loss. Divided by the batch size, both therefore equal $\log 2$ at the equilibrium, while the backbone below reports the full logistic loss, $2 \log 2$. The loop below alternates them over shuffled minibatches of 256 images with the DCGAN settings: learning rate $2 \cdot 10^{-4}$, $\beta_1 = 0.5$, convolution kernels drawn from $\mathcal{N}(0, 0.02^2)$, batch-normalization scales from $\mathcal{N}(1, 0.02^2)$, and offsets at zero.

```{.python .input #dcgan-training-with-the-classic-loss-1}
%%tab pytorch
def dcgan_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)

def train_dcgan(net_D, net_G, images, num_epochs=20, batch_size=256,
                lr=0.0002, latent_dim=100):
    for net in (net_D, net_G):
        net.apply(dcgan_init)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr, betas=(0.5, 0.999))
    for epoch in range(num_epochs):
        loss_D = loss_G = 0  # accumulated on the device
        order = torch.randperm(len(images), device=images.device)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            X = images[order[i:i + batch_size]]
            Z = torch.randn(batch_size, latent_dim, 1, 1, device=X.device)
            loss_D += d2l.update_D(X, Z, net_D, net_G, trainer_D).detach()
            loss_G += d2l.update_G(Z, net_D, net_G, trainer_G).detach()
    n = len(order) // batch_size * batch_size
    return float(loss_D) / n, float(loss_G) / n  # last epoch, per image
```

```{.python .input #dcgan-training-with-the-classic-loss-1}
%%tab jax
def train_dcgan(net_D, net_G, images, num_epochs=20, batch_size=256,
                lr=0.0002, latent_dim=100):
    optimizer_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.5, b2=0.999),
                                wrt=nnx.Param)
    optimizer_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.5, b2=0.999),
                                wrt=nnx.Param)
    key = jax.random.key(0)
    for epoch in range(num_epochs):
        loss_D = loss_G = 0  # accumulated on the device
        key, kp = jax.random.split(key)
        order = jax.random.permutation(kp, len(images))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            X = images[order[i:i + batch_size]]
            key, kz = jax.random.split(key)
            Z = jax.random.normal(kz, (batch_size, 1, 1, latent_dim))
            loss_D += d2l.update_D(X, Z, net_D, net_G, optimizer_D)
            loss_G += d2l.update_G(Z, net_D, net_G, optimizer_G)
    n = len(order) // batch_size * batch_size
    return float(loss_D) / n, float(loss_G) / n  # last epoch, per image
```

Twenty epochs take a minute or two on one GPU. The printed losses are per image.

```{.python .input #dcgan-training-with-the-classic-loss-2}
%%tab pytorch
loss_D, loss_G = train_dcgan(net_D, net_G, train_imgs)
print(f'last epoch: loss_D {loss_D:.3f}, loss_G {loss_G:.3f}')
```

```{.python .input #dcgan-training-with-the-classic-loss-2}
%%tab jax
loss_D, loss_G = train_dcgan(net_D, net_G, train_imgs)
print(f'last epoch: loss_D {loss_D:.3f}, loss_G {loss_G:.3f}')
```

We sample in evaluation mode, in which batch normalization uses the running statistics collected during training, so that each image depends on its own latent code only.

```{.python .input #dcgan-training-with-the-classic-loss-3}
%%tab pytorch
net_G.eval()
with torch.no_grad():
    fake = net_G(torch.randn(20, 100, 1, 1, device=device))
d2l.show_images(fake.cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=4, num_cols=5);
```

```{.python .input #dcgan-training-with-the-classic-loss-3}
%%tab jax
net_G.eval()
fake = net_G(jax.random.normal(jax.random.key(2), (20, 1, 1, 100)))
d2l.show_images(np.asarray(fake) / 2 + 0.5, num_rows=4, num_cols=5);
```

The samples are diverse: centered, creature-like shapes on mostly white backgrounds, with plausible palettes but blotchy textures. The losses show an unbalanced game. The critic's loss sits far below $\log 2$ and the generator's several times above it, so the critic separates the two sets well, yet the generator does not collapse onto one output. Batch normalization and the architecture keep the classic loss trainable here without changing it, so the divergent dynamics of :numref:`sec_gan_convergence` remain possible. We score this run against the modern backbone below.

## A Modern Minimal Backbone

### Architectural Developments after DCGAN
:label:`subsec_gan_r3gan_principles`

DCGAN stabilized the classic loss with architecture. For most of the following decade, improvements to image GANs were again architectural, each a response to a specific failure:

| Design | Problem addressed |
|:--|:--|
| Progressive growing :cite:`Karras.Aila.Laine.ea.2017` | instability at high resolution: both networks start at $4 \times 4$ and grow together |
| Spectral normalization :cite:`Miyato.Kataoka.Koyama.ea.2018` | an unboundedly steep critic: each weight matrix is divided by its largest singular value |
| Self-attention :cite:`Zhang.Goodfellow.Metaxas.ea.2019` | missing long-range structure: attention gives both networks image-wide receptive fields |
| BigGAN :cite:`Brock.Donahue.Simonyan.2019` | small-scale ceilings: large batches and wide networks; collapse is delayed, not removed |
| StyleGAN :cite:`Karras.Laine.Aila.2019` | entangled latent factors: a mapping network and per-layer style modulation |
| StyleGAN2 :cite:`Karras.Laine.Aittala.ea.2020` | droplet artifacts from StyleGAN's normalization, replaced by weight demodulation |
| StyleGAN3 :cite:`Karras.Aittala.Laine.ea.2021` | texture sticking to pixel coordinates, removed by band-limited resampling |
| Projected discriminators :cite:`Sauer.Chitta.Muller.ea.2021` | slow critic learning: the critic reads frozen pretrained features |

Some of these devices impose by architecture what the penalties of :numref:`sec_gan_convergence` impose through the loss; spectral normalization, for instance, bounds the critic's steepness everywhere, where the penalties act at the samples. Others repair artifacts of earlier architectures.

R3GAN tests whether the loss can replace them :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. It trains with the pairing loss and both zero-centered penalties of :numref:`sec_gan_convergence` and uses a plain modern convolutional backbone without progressive growing, spectral normalization, attention, style modulation, or pretrained features. On standard benchmarks this simpler network matches or exceeds the architectures that combine those devices, which suggests that many of them compensated for a badly behaved objective.

The paper lists six differences between its minimal baseline and DCGAN: (a) a convergent objective with $R_1$ regularization, which the paper's experiments use as the pairing loss with $R_1$ and $R_2$, as we do; (b) a small learning rate and Adam with no first-moment momentum ($\beta_1 = 0$); (c) no normalization layer in either network; (d) bilinear resampling instead of strided or transposed convolutions; (e) leaky ReLU in both networks and no tanh at the generator's output; and (f) a $4 \times 4$ constant input, output skips, and a residual critic. Violating (a), (b), or (c) often leads to training failures, while violating (d) or (e) lowers sample quality. On FFHQ at $256 \times 256$ they take StyleGAN2 apart and rebuild it from these principles at about the same parameter count and training budget:

| config | change relative to the previous row | FID |
|:---|:---|---:|
| A | StyleGAN2 baseline | 7.5 |
| B | remove all ten tricks: $z$ normalization, minibatch standard deviation, equalized learning rate, mapping network, style injection, weight demodulation, noise injection, mixing regularization, path-length regularization, lazy regularization | 12.5 |
| C | add the well-behaved loss: RpGAN objective with $R_1$ and $R_2$ | 11.7 |
| D | modernize the backbone: ResNet generator and critic | 10.0 |
| E | widen with grouped convolutions and inverted bottlenecks | 7.0 |

Rows D and E each merge two steps of the paper's table. Without any of the ten components, the revised loss and a modern backbone bring FID back to 7.0 :cite:`Huang.Gokaslan.Kuleshov.ea.2024`.

### The Backbone

We build a small network that follows principles (b) to (e) at sprite scale (:numref:`fig_gan_backbone`). Both networks resample with bilinear interpolation followed by an ordinary $3 \times 3$ convolution, which avoids the uneven kernel overlap of strided and transposed convolutions that produces checkerboard artifacts :cite:`Odena.Dumoulin.Olah.2016`. Both use leaky ReLU, the generator has no tanh, and neither network uses normalization. Batch normalization would make each critic score depend on the whole minibatch and couple the per-sample input gradients that the penalties measure.

![The minimal backbone used with the pairing loss, with the parameter count of each layer. The generator concatenates a learned $4 \times 4 \times 128$ constant with a linear projection of $z$ to $4 \times 4 \times 100$, mixes them with a convolution, and doubles the resolution four times by bilinear upsampling, a $3 \times 3$ convolution and a leaky ReLU. A last convolution produces the image, with no tanh. The critic mirrors it with bilinear downsampling, and a linear head gives one logit. Neither network uses normalization.](../img/mdl-gan-backbone.svg)
:label:`fig_gan_backbone`

The generator starts from a learned $4 \times 4 \times 128$ constant rather than from the latent vector, so the latent code needs its own entry. A linear layer projects $z$ to a $4 \times 4 \times 100$ map, which is concatenated with the constant along the channel axis and mixed with it by a $3 \times 3$ convolution: the constant holds what all samples share, the code what varies. R3GAN instead uses a *basis layer*, in which a linear function of $z$ scales a set of learned $4 \times 4$ maps. We keep concatenation because the next section feeds the class label into the network the same way.

Without normalization layers, nothing rescales the activations as they pass through the network, so the initialization has to keep their scale. We draw every weight with variance $2 / ((1 + 0.2^2)\, n_{\mathrm{in}})$ for a layer with $n_{\mathrm{in}}$ inputs, the fan-in rule of :citet:`He.Zhang.Ren.ea.2015` with the gain of a leaky ReLU of slope 0.2, which keeps the variance of the activations roughly constant from layer to layer. The learned constant starts from unit-variance noise.

```{.python .input #dcgan-a-modern-minimal-backbone-1}
%%tab pytorch
def fan_in_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2)  # leaky-ReLU gain
        nn.init.zeros_(module.bias)

class Generator(nn.Module):
    """Learned 4x4 constant + projected latent, bilinearly upsampled."""
    def __init__(self, latent_dim=100, const_ch=128, base_ch=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.const = nn.Parameter(torch.randn(1, const_ch, 4, 4))
        self.z_proj = nn.Linear(latent_dim, 16 * latent_dim)
        self.mix = nn.Conv2d(const_ch + latent_dim, base_ch, 3, padding=1)
        chans = [base_ch // 2 ** i for i in range(5)]    # 512, 256, ..., 32
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)

    def forward(self, z):
        zc = self.z_proj(z).reshape(-1, self.latent_dim, 4, 4)
        const = self.const.expand(z.shape[0], -1, -1, -1)
        x = self.act(self.mix(torch.cat([const, zc], dim=1)))
        for conv in self.stages:                     # 4 -> 8 -> ... -> 64
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.act(conv(x))
        return self.to_rgb(x)                        # no tanh: raw output
```

```{.python .input #dcgan-a-modern-minimal-backbone-1}
%%tab jax
# Fan-in initialization with the leaky-ReLU gain (normal, not truncated)
fan_in_init = nnx.initializers.variance_scaling(2 / (1 + 0.2 ** 2),
                                                'fan_in', 'normal')

class Generator(nnx.Module):
    """Learned 4x4 constant + projected latent, bilinearly upsampled."""
    def __init__(self, latent_dim=100, const_ch=128, base_ch=512,
                 rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.latent_dim = latent_dim
        self.const = nnx.Param(
            jax.random.normal(rngs.params(), (1, 4, 4, const_ch)))
        self.z_proj = nnx.Linear(latent_dim, 16 * latent_dim,
                                 kernel_init=fan_in_init, rngs=rngs)
        self.mix = nnx.Conv(const_ch + latent_dim, base_ch, (3, 3),
                            padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        chans = [base_ch // 2 ** i for i in range(5)]    # 512, 256, ..., 32
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nnx.Conv(chans[-1], 3, (3, 3), padding='SAME',
                               kernel_init=fan_in_init, rngs=rngs)

    def __call__(self, z):
        zc = self.z_proj(z).reshape(-1, 4, 4, self.latent_dim)
        const = jnp.broadcast_to(self.const[...],
                                 (z.shape[0],) + self.const.shape[1:])
        x = nnx.leaky_relu(self.mix(jnp.concatenate([const, zc], axis=-1)),
                           0.2)
        for conv in self.stages:                     # 4 -> 8 -> ... -> 64
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, 2 * h, 2 * w, c), method='bilinear')
            x = nnx.leaky_relu(conv(x), 0.2)
        return self.to_rgb(x)                        # no tanh: raw output
```

The critic mirrors the generator: four stages of convolution, leaky ReLU, and bilinear downsampling take a $64 \times 64$ image to $4 \times 4$, and a mixing convolution and a linear head return one score.

```{.python .input #dcgan-a-modern-minimal-backbone-2}
%%tab pytorch
class Discriminator(nn.Module):
    """Mirror of the generator: conv + leaky ReLU + bilinear downsampling."""
    def __init__(self, base_ch=32):
        super().__init__()
        chans = [3] + [base_ch * 2 ** i for i in range(4)]  # 3, 32, ..., 256
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nn.Conv2d(chans[-1], 2 * chans[-1], 3, padding=1)
        self.head = nn.Linear(2 * chans[-1] * 4 * 4, 1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)

    def forward(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = F.interpolate(self.act(conv(x)), scale_factor=0.5,
                              mode='bilinear', antialias=True)
        x = self.act(self.mix(x))
        return self.head(x.reshape(x.shape[0], -1))

G = Generator()
for net in (G, Discriminator()):
    print(f'{type(net).__name__}: '
          f'{sum(p.numel() for p in net.parameters())} parameters')
n_proj = sum(p.numel() for p in G.z_proj.parameters())
print(f'latent path: projection {n_proj}, '
      f'mixing weights on z {G.mix.weight[:, 128:].numel()}')
```

```{.python .input #dcgan-a-modern-minimal-backbone-2}
%%tab jax
class Discriminator(nnx.Module):
    """Mirror of the generator: conv + leaky ReLU + bilinear downsampling."""
    def __init__(self, base_ch=32, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        chans = [3] + [base_ch * 2 ** i for i in range(4)]  # 3, 32, ..., 256
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nnx.Conv(chans[-1], 2 * chans[-1], (3, 3),
                            padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        self.head = nnx.Linear(2 * chans[-1] * 4 * 4, 1,
                               kernel_init=fan_in_init, rngs=rngs)

    def __call__(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = nnx.leaky_relu(conv(x), 0.2)
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, h // 2, w // 2, c),
                                 method='bilinear')
        x = nnx.leaky_relu(self.mix(x), 0.2)
        return self.head(x.reshape(x.shape[0], -1))

G = Generator(rngs=nnx.Rngs(0))
for net in (G, Discriminator(rngs=nnx.Rngs(0))):
    n = sum(v.size for v in jax.tree.leaves(nnx.state(net, nnx.Param)))
    print(f'{type(net).__name__}: {n} parameters')
print(f'latent path: projection {G.z_proj.kernel.size + G.z_proj.bias.size}'
      f', mixing weights on z {G.mix.kernel[:, :, 128:].size}')
```

The generator has about 2.8 million parameters and the critic about 1.6 million, a reduced instance of R3GAN's design without its residual blocks, grouped convolutions, and inverted bottlenecks. Its latent path, the projection of $z$ and the mixing weights on the 100 extra channels, has about 620,000 parameters, ten times as many as R3GAN's basis layer of the same width (Exercise :exref:`ex_dcgan-rerun-arm-classic-loss` implements it).

Three helpers, saved for the next section, complete the recipe. Random horizontal flips augment the real images, which makes it harder for the critic to memorize them. Samples are drawn from an exponential moving average of the generator's parameters, $\bar\theta \leftarrow \beta \bar\theta + (1 - \beta)\theta$ after every step with $\beta = 2^{-1/500}$, a half-life of 500 steps. Because the game's iterates rotate around the equilibrium as they contract (:numref:`sec_gan_convergence`), their average sits closer to the center than they do. The class `EMA` keeps the averaged network as `ema.model`, in evaluation mode; it averages the parameters and copies buffers such as batch-normalization statistics, so it also serves networks that have them. Finally, `ranking_accuracy` reports the fraction of (real, generated) pairs that a critic orders correctly; we use it in the comparison below.

```{.python .input #dcgan-a-modern-minimal-backbone-3}
%%tab pytorch
#@save
class EMA:
    """Exponential moving average of a model's parameters, given as a
    half-life in steps. Buffers such as batch-norm statistics are copied;
    the average is kept in evaluation mode."""
    def __init__(self, model, half_life):
        import copy  # the d2l library preamble does not import it
        self.model = copy.deepcopy(model).requires_grad_(False).eval()
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        with torch.no_grad():
            for avg, p in zip(self.model.parameters(), model.parameters()):
                avg.lerp_(p, 1 - self.decay)
            for avg, b in zip(self.model.buffers(), model.buffers()):
                avg.copy_(b)

#@save
def sample_real(images, n, labels=None):
    """Draw a training batch with random horizontal flips (and its labels)."""
    idx = torch.randint(0, len(images), (n,), device=images.device)
    batch = images[idx]
    flip = torch.rand(n, device=images.device) < 0.5
    batch = torch.where(flip.view(-1, 1, 1, 1), batch.flip(-1), batch)
    return batch if labels is None else (batch, labels[idx])

#@save
def ranking_accuracy(critic, real, fake):
    """Fraction of aligned (real, generated) pairs with D(real) > D(fake)."""
    with torch.no_grad():
        return float((critic(real) > critic(fake)).float().mean())
```

```{.python .input #dcgan-a-modern-minimal-backbone-3}
%%tab jax
#@save
class EMA:
    """Exponential moving average of a model's parameters, given as a
    half-life in steps. Other state such as batch-norm statistics is copied;
    the average is kept in evaluation mode."""
    def __init__(self, model, half_life):
        self.model = nnx.clone(model)
        self.model.eval()
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        self._average(self.model, model, self.decay)

    @staticmethod
    @nnx.jit
    def _average(avg, model, decay):
        params = jax.tree.map(lambda a, p: decay * a + (1 - decay) * p,
                              nnx.state(avg, nnx.Param),
                              nnx.state(model, nnx.Param))
        nnx.update(avg, params, nnx.state(model, nnx.Not(nnx.Param)))

#@save
def sample_real(key, images, n, labels=None):
    """Draw a training batch with random horizontal flips (and its labels)."""
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, len(images))
    batch = images[idx]
    flip = jax.random.bernoulli(k2, 0.5, (n,))
    batch = jnp.where(flip[:, None, None, None], batch[:, :, ::-1, :], batch)
    return batch if labels is None else (batch, labels[idx])

#@save
def ranking_accuracy(critic, real, fake):
    """Fraction of aligned (real, generated) pairs with D(real) > D(fake)."""
    return float((critic(real) > critic(fake)).mean())
```

## A Controlled Loss Comparison

### Three Arms

We train the backbone three times with the same seed, Adam with $\beta_1 = 0$ and $\beta_2 = 0.99$ at learning rate $2 \cdot 10^{-4}$, batch size 64, flips, and EMA. Only the loss changes, one ingredient at a time. Writing $x$ for a real image and $x'$ for a generated one, the two critic losses are

$$
L_{\textrm{classic}}(D) = \mathbb{E}_{x}\big[\log(1 + e^{-D(x)})\big] + \mathbb{E}_{x'}\big[\log(1 + e^{D(x')})\big],
\qquad
L_{\textrm{pairing}}(D) = \mathbb{E}_{x, x'}\big[\log(1 + e^{D(x') - D(x)})\big].
$$

The first is the logistic loss of :numref:`sec_basic_gan` (`d2l.gan_loss_D`), the second the pairing loss of :numref:`sec_gan_relativistic` (`d2l.rpgan_loss_D`), which only sees score differences. At $q = p$ the best critic is constant and the two losses equal $2 \log 2 \approx 1.39$ and $\log 2 \approx 0.69$. Each generator uses the matching non-saturating loss, $\mathbb{E}_{x'}[\log(1 + e^{-D(x')})]$ or $\mathbb{E}_{x, x'}[\log(1 + e^{D(x) - D(x')})]$. The penalized arms add the zero-centered penalties of :numref:`sec_gan_convergence` to the critic's loss,

$$\frac{\gamma}{2}\Big(\mathbb{E}_x\big[\|\nabla_x D(x)\|^2\big] + \mathbb{E}_{x'}\big[\|\nabla_{x'} D(x')\|^2\big]\Big),$$

which flatten the critic at the real images ($R_1$) and at the generated ones ($R_2$) and are returned per sample by `d2l.r1_r2_penalty`. The arms are:

| arm | critic loss | penalty | steps |
|:--|:--|:--|--:|
| NS | classic | none | 6,000 |
| NS + $R_1$ + $R_2$ | classic | $\gamma = 10$ | 12,000 |
| RpGAN + $R_1$ + $R_2$ | pairing | $\gamma = 10$ | 12,000 |

The first two arms isolate the penalty; the last two isolate pairing. We use $\gamma = 10$. Because $\gamma$ is a squared length on the input scale (:numref:`sec_gan_convergence`), a tuned value does not transfer between datasets (Exercise :exref:`ex_dcgan-penalty-weight-resolution`). The unpenalized arm gets half the budget because it has failed by then.

The training function takes the two losses as functions and the penalty weight as a number, so all three arms run through one loop. The penalty is computed in every arm, at weight zero in the first, so that the critic's steepness on real images, $\mathbb{E}_x\|\nabla_x D(x)\|^2$, is logged for all three, and every 250 steps the loop scores 256 fresh real images against 256 fresh generated ones.

```{.python .input #dcgan-loss-a-b-on-one-backbone-1}
%%tab pytorch
def train_backbone(images, loss_D, loss_G, gamma, num_steps, batch_size=64,
                   lr=0.0002, latent_dim=100, half_life=500, log_every=250):
    torch.manual_seed(0)  # the same initialization in every arm
    net_G = Generator().to(images.device)
    net_D = Discriminator().to(images.device)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr, betas=(0.0, 0.99))
    trainer_D = torch.optim.Adam(net_D.parameters(), lr, betas=(0.0, 0.99))
    noise = lambda n: torch.randn(n, latent_dim, device=images.device)
    ema, history = EMA(net_G, half_life), []
    for step in range(1, num_steps + 1):
        X = sample_real(images, batch_size)
        with torch.no_grad():
            fake = net_G(noise(batch_size))
        r1, r2 = d2l.r1_r2_penalty(net_D, X, fake)
        loss, penalty = loss_D(net_D, X, fake), gamma / 2 * (r1 + r2).mean()
        trainer_D.zero_grad(), (loss + penalty).backward(), trainer_D.step()
        X, fake = sample_real(images, batch_size), net_G(noise(batch_size))
        trainer_G.zero_grad(), loss_G(net_D, X, fake).backward()
        trainer_G.step(), ema.update(net_G)
        if step % log_every == 0:  # fresh pairs: 256 real, 256 generated
            with torch.no_grad():
                X, fake = sample_real(images, 256), net_G(noise(256))
                weight = torch.sigmoid(net_D(fake) - net_D(X)).mean()
            history.append([step, loss.item(), penalty.item(),
                            r1.mean().item(), ranking_accuracy(net_D, X, fake),
                            weight.item()])
    return ema.model, net_D, np.array(history)
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-1}
%%tab jax
def make_step(loss_D, loss_G, gamma):
    @nnx.jit
    def step(net_G, net_D, opt_G, opt_D, X, Z, X2, Z2):
        fake = jax.lax.stop_gradient(net_G(Z))
        def loss_D_fn(net_D):
            r1, r2 = d2l.r1_r2_penalty(net_D, X, fake)
            loss = loss_D(net_D, X, fake)
            penalty = gamma / 2 * (r1 + r2).mean()
            return loss + penalty, (loss, penalty, r1.mean())
        (_, logs), grads = nnx.value_and_grad(loss_D_fn, has_aux=True)(net_D)
        opt_D.update(net_D, grads)
        grads = nnx.grad(lambda net_G: loss_G(net_D, X2, net_G(Z2)))(net_G)
        opt_G.update(net_G, grads)
        return logs
    return step

def train_backbone(images, loss_D, loss_G, gamma, num_steps, batch_size=64,
                   lr=0.0002, latent_dim=100, half_life=500, log_every=250):
    rngs = nnx.Rngs(0)  # the same initialization in every arm
    net_G, net_D = Generator(rngs=rngs), Discriminator(rngs=rngs)
    opt_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    step_fn, key = make_step(loss_D, loss_G, gamma), jax.random.key(1)
    noise = lambda key, n: jax.random.normal(key, (n, latent_dim))
    ema, history = EMA(net_G, half_life), []
    for step in range(1, num_steps + 1):
        key, kx, kz, kx2, kz2 = jax.random.split(key, 5)
        logs = step_fn(net_G, net_D, opt_G, opt_D,
                       sample_real(kx, images, batch_size),
                       noise(kz, batch_size),
                       sample_real(kx2, images, batch_size),
                       noise(kz2, batch_size))
        ema.update(net_G)
        if step % log_every == 0:  # fresh pairs: 256 real, 256 generated
            key, kx, kz = jax.random.split(key, 3)
            X, fake = sample_real(kx, images, 256), net_G(noise(kz, 256))
            weight = jax.nn.sigmoid(net_D(fake) - net_D(X)).mean()
            history.append([step, *map(float, logs),
                            ranking_accuracy(net_D, X, fake), float(weight)])
    return ema.model, net_D, np.array(history)
```

The three runs are the longest computation in the chapter, well under half an hour on one RTX 4090; each penalized step costs more because the penalty differentiates the critic's input gradient a second time. After each run we print the critic's loss and penalty, averaged over the last eight logging steps (2,000 steps), and the difference between the critic's mean score on 1,024 training images and on 1,024 held-out images.

```{.python .input #dcgan-loss-a-b-on-one-backbone-2}
%%tab pytorch
arms = {'NS': (d2l.gan_loss_D, d2l.gan_loss_G, 0, 6000),
        'NS + R1 + R2': (d2l.gan_loss_D, d2l.gan_loss_G, 10, 12000),
        'RpGAN + R1 + R2': (d2l.rpgan_loss_D, d2l.rpgan_loss_G, 10, 12000)}
runs = {}
for name, (loss_D, loss_G, gamma, num_steps) in arms.items():
    ema_G, critic, hist = train_backbone(train_imgs, loss_D, loss_G, gamma,
                                         num_steps)
    with torch.no_grad():
        gap = (critic(train_imgs[:1024]).mean()
               - critic(holdout_imgs[:1024]).mean())
    runs[name] = (ema_G, hist)
    loss, penalty = hist[-8:, 1:3].mean(0)
    print(f'{name:16s} critic loss {loss:.2f}, penalty {penalty:.3f}, '
          f'train-holdout gap {gap.item():+.2f}')
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-2}
%%tab jax
arms = {'NS': (d2l.gan_loss_D, d2l.gan_loss_G, 0, 6000),
        'NS + R1 + R2': (d2l.gan_loss_D, d2l.gan_loss_G, 10, 12000),
        'RpGAN + R1 + R2': (d2l.rpgan_loss_D, d2l.rpgan_loss_G, 10, 12000)}
runs = {}
for name, (loss_D, loss_G, gamma, num_steps) in arms.items():
    ema_G, critic, hist = train_backbone(train_imgs, loss_D, loss_G, gamma,
                                         num_steps)
    gap = (critic(train_imgs[:1024]).mean()
           - critic(holdout_imgs[:1024]).mean())
    runs[name] = (ema_G, hist)
    loss, penalty = hist[-8:, 1:3].mean(0)
    print(f'{name:16s} critic loss {loss:.2f}, penalty {penalty:.3f}, '
          f'train-holdout gap {float(gap):+.2f}')
```

Both penalized critics sit below the value they would have at the equilibrium, about 1.05 against $2 \log 2 \approx 1.39$ for the classic loss and about 0.45 against $\log 2 \approx 0.69$ for pairing: each stays somewhat ahead of its generator without running away, and the penalty adds about a tenth to a fifth to its loss. The unpenalized critic's loss swings too widely to be read from one number. We return to the train--holdout gaps in :numref:`subsec_gan_limited_data`.

### Training Traces

The losses alone do not say how the critic separates the two sets. Three statistics of the logged pairs do. The first is the critic's steepness on real images, $\mathbb{E}_x\|\nabla_x D(x)\|^2$, which the $R_1$ penalty pushes down. The second is the *ranking accuracy* $\hat P(D(x) > D(x'))$, the fraction of fresh (real, generated) pairs that the critic orders correctly; 0.5 means that it cannot tell the sets apart and 1 that it orders every pair. The third is the *rank weight* of :numref:`sec_gan_relativistic`, averaged over generated samples, $\mathbb{E}_{x'}[w_{\mathrm{Rp}}(x')] = \mathbb{E}_{x, x'}[\sigma(D(x') - D(x))]$: the probability, softened by the sigmoid, that a generated image outranks a random real one. Replacing the sigmoid by a step function turns it into one minus the ranking accuracy, so the two measure the same ordering, once softly and once by a hard threshold.

```{.python .input #dcgan-loss-a-b-on-one-backbone-4}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 3, figsize=(12, 3.2))
names = list(runs)
for ax, col, ylabel, yscale in zip(
        axes, (3, 4, 5),
        (r'$E\,\|\nabla_x D(x)\|^2$', 'ranking accuracy',
         r'rank weight $w_{\mathrm{Rp}}$'),
        ('log', 'linear', 'linear')):
    d2l.plot([runs[n][1][:, 0] for n in names],
             [runs[n][1][:, col] for n in names], 'step', ylabel,
             yscale=yscale, axes=ax)
axes[0].legend(names, loc='center right')
fig.tight_layout()
```

The first panel separates the arms by two orders of magnitude. Without a penalty the critic's squared gradient on real images jumps around between about 0.1 and ten, mostly near one; both penalized critics stay flat near 0.01. The second panel hardly separates them: all three critics order more than nine in ten fresh pairs correctly at almost every check, the unpenalized one with sharp dips. The penalty does not stop the critic from ranking; it limits how steep the critic is. The third panel shows what this means for the generator. Without the penalty the rank weight stays mostly below 0.2, often near zero, with spikes; with it, the weight settles between about a quarter and two fifths.

### Samples of the Three Arms

The grids show 64 samples of each EMA generator, drawn from the same latent codes in all three arms.

```{.python .input #dcgan-loss-a-b-on-one-backbone-3}
%%tab pytorch
def image_grid(imgs, rows=8, cols=8):
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = imgs.reshape(rows, cols, 3, 64, 64).permute(0, 3, 1, 4, 2)
    return imgs.reshape(rows * 64, cols * 64, 3).cpu().numpy()

torch.manual_seed(42)
z_show = torch.randn(64, 100, device=device)
fig, axes = d2l.plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, (name, (ema_G, hist)) in zip(axes, runs.items()):
    with torch.no_grad():
        ax.imshow(image_grid(ema_G(z_show)))
    ax.set_title(name)
    ax.axis('off')
fig.tight_layout()
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-3}
%%tab jax
def image_grid(imgs, rows=8, cols=8):
    imgs = (jnp.clip(imgs, -1, 1) + 1) / 2
    imgs = imgs.reshape(rows, cols, 64, 64, 3).transpose(0, 2, 1, 3, 4)
    return np.asarray(imgs.reshape(rows * 64, cols * 64, 3))

z_show = jax.random.normal(jax.random.key(42), (64, 100))
fig, axes = d2l.plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, (name, (ema_G, hist)) in zip(axes, runs.items()):
    ax.imshow(image_grid(ema_G(z_show)))
    ax.set_title(name)
    ax.axis('off')
fig.tight_layout()
```

Without a penalty the classic loss collapses: after 6,000 steps the 64 latent codes produce one grey texture that resembles no sprite (in some runs a handful of images). The two penalized arms produce diverse, creature-shaped sprites on white backgrounds, with softer outlines than real sprites, and no difference between the two is visible.

The collapse is a training failure of an unregularized game, not a minimum of the loss. :numref:`sec_gan_convergence` showed that gradient training circles the equilibrium unless something damps the critic near the data. Here nothing does: the squared input gradient of the unpenalized critic at the real images is about a hundred times that of the penalized critics, and the critic places nearly every generated sample far below the real ones.

The third panel connects this to :numref:`sec_gan_relativistic`. A rank weight near zero, as without the penalty, means that the critic ranks every generated sample below nearly every real one. The weight has saturated, and its values no longer say which samples are worse. The non-saturating generator, which weights a sample by $1 - w_{\mathrm{Rp}}$, then gives every sample the same full-size push. The penalty keeps the rank weight between about a quarter and two fifths, so samples the critic ranks lower are pushed harder than samples it nearly mistakes for real ones. The weight is computed from the critic's scores alone, so it can be read in every arm, whichever generator loss uses it. The ranking accuracy, one minus its hard-threshold version, is close to one in all three arms and misses this difference.

Pairing itself adds nothing measurable here. With both penalties, the classic and the pairing loss reach the same quality on sprites at this budget: their scores below overlap across seeds and frameworks, and so do their grids. Sprites have no known modes, so this comparison cannot measure coverage. R3GAN measured it on StackedMNIST, whose 1,000 modes can be counted: with both penalties the pairing loss recovered all of them and the classic loss 693, in single runs (:numref:`sec_gan_convergence`). The theory of :numref:`sec_gan_relativistic` does not settle the question either. Its no-collapse result holds for a fixed one-to-one pairing, while our loop, like R3GAN's, draws new pairs at every step, and averaged over pairings that is the all-pairs loss, which has as many mode-dropping minima as the classic loss.

## Measuring Sample Quality

A grid shows a collapse, but it cannot rank two good checkpoints or two settings of $\gamma$, and an implicit generator has no likelihood to rank them by. The standard scores pass $n$ real and $n$ generated images through a fixed feature map $\phi$ and compare the two clouds of feature vectors. Both scores used here are closed forms from :numref:`sec_gan_objectives`, applied in feature space.

### FID and KID from Two Closed Forms

The Fréchet inception distance (FID) of :citet:`Heusel.Ramsauer.Unterthiner.ea.2017` fits a Gaussian to each cloud, $\mathcal{N}(\mu_p, \Sigma_p)$ for the real features and $\mathcal{N}(\mu_q, \Sigma_q)$ for the generated ones, and reports the squared Wasserstein-2 distance between the two Gaussians. In one dimension this is the cost of moving one bell curve onto the other: shift its center by $\mu_p - \mu_q$ and stretch its standard deviation by $\tau_p - \tau_q$, which costs $(\mu_p - \mu_q)^2 + (\tau_p - \tau_q)^2$. Real features $\mathcal{N}(0, 1)$ against generated features $\mathcal{N}(0.5, 1.1^2)$ give $0.25 + 0.01 = 0.26$: the mean shift dominates. In $d$ dimensions the width becomes a covariance, and the general $W_2$ problem of :numref:`sec_gan_objectives`, which has no explicit solution, has one for Gaussians :cite:`Dowson.Landau.1982,Givens.Shortt.1984`:

$$
\mathrm{FID}
= \big\| \mu_p - \mu_q \big\|^2
+ \operatorname{tr}\!\Big( \Sigma_p + \Sigma_q
- 2 \big( \Sigma_p^{1/2}\, \Sigma_q\, \Sigma_p^{1/2} \big)^{1/2} \Big).
$$
:eqlabel:`eq_gan_fid`

The first term is the squared distance between the means and the trace term compares the spreads. If $\Sigma_p$ and $\Sigma_q$ commute, they share eigenvectors with eigenvalues $\alpha_i$ and $\beta_i$, and the trace term becomes $\sum_i (\sqrt{\alpha_i} - \sqrt{\beta_i})^2$: the one-dimensional formula, applied along each shared axis. Exercise :exref:`ex_dcgan-one-dimension-squared-wasserstein` derives the one-dimensional case from optimal transport. FID has two built-in limits: the Gaussian fit keeps only the first two moments of the features, and its plug-in estimate is biased at finite $n$, so FID between two disjoint samples of real images is positive.

The kernel inception distance (KID) of :citet:`Binkowski.Sutherland.Arbel.ea.2018` uses the chapter's other closed form, the maximum mean discrepancy of :numref:`sec_gan_objectives`, with the cubic polynomial kernel $k(u, v) = (u^\top v / d + 1)^3$ on $d$-dimensional features. For $n_p$ real images $x_i$ and $n_q$ generated images $x'_j$, KID reports the unbiased estimate of $\mathrm{MMD}^2$, which leaves out the terms that compare a point with itself:

$$
\widehat{\mathrm{MMD}}^2
= \frac{1}{n_p(n_p-1)} \sum_{i \neq i'} k\big(\phi(x_i), \phi(x_{i'})\big)
+ \frac{1}{n_q(n_q-1)} \sum_{j \neq j'} k\big(\phi(x'_j), \phi(x'_{j'})\big)
- \frac{2}{n_p n_q} \sum_{i,j} k\big(\phi(x_i), \phi(x'_j)\big).
$$
:eqlabel:`eq_gan_kid`

The first two sums measure how similar the images within each set are, the third how similar they are across sets. Because the estimate is unbiased, two disjoint samples of real images give a value that fluctuates around zero and can be slightly negative. The kernel is fixed, as in :numref:`sec_gan_objectives`, but it acts on learned features, so what KID can detect is set jointly by the kernel and the feature network. 

```{.python .input #dcgan-measuring-sample-quality}
%%tab pytorch, jax
#@save
def fid_score(feat_p, feat_q):
    """Frechet distance between Gaussians fitted to two feature sets."""
    feat_p, feat_q = np.float64(feat_p), np.float64(feat_q)
    mu_p, mu_q = feat_p.mean(0), feat_q.mean(0)
    C_p = np.cov(feat_p, rowvar=False)
    C_q = np.cov(feat_q, rowvar=False)
    lam, U = np.linalg.eigh(C_p)
    sqrt_C_p = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    M = sqrt_C_p @ C_q @ sqrt_C_p
    cross = np.linalg.eigvalsh((M + M.T) / 2)   # symmetric up to rounding
    return (((mu_p - mu_q) ** 2).sum() + np.trace(C_p) + np.trace(C_q)
            - 2 * np.sqrt(np.maximum(cross, 0)).sum())

#@save
def kid_score(feat_p, feat_q):
    """Unbiased MMD^2 between two feature sets, cubic polynomial kernel."""
    m, n, d = len(feat_p), len(feat_q), feat_p.shape[1]
    K_pp = (feat_p @ feat_p.T / d + 1) ** 3
    K_qq = (feat_q @ feat_q.T / d + 1) ** 3
    K_pq = (feat_p @ feat_q.T / d + 1) ** 3
    return ((K_pp.sum() - np.trace(K_pp)) / (m * (m - 1))
            + (K_qq.sum() - np.trace(K_qq)) / (n * (n - 1))
            - 2 * K_pq.mean())
```

### A Feature Network Trained in the Notebook

Both scores need a feature map $\phi$. Published FID uses the pooled features of an Inception-v3 classifier trained on ImageNet. To keep the experiment inside the notebook, we train a small convolutional classifier on CIFAR-10 for 15 epochs, with random flips and a learning rate that rises and then anneals (the one-cycle schedule), and use its 128-dimensional penultimate layer as $\phi$. Its three blocks of $3 \times 3$ convolution and $2 \times 2$ pooling see a window of about 22 pixels, so after global averaging its features are statistics of local texture and color rather than of whole shapes. We train it on CIFAR-10 upsampled to $64 \times 64$ rather than shrink the sprites to $32 \times 32$, which could hide the fine texture and checkerboard artifacts the score should see. The loader, the network, whose `resolution` argument sets the size to which it resizes its inputs, and its training recipe are saved for :numref:`sec_gan_conditional`.

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-1}
%%tab pytorch
#@save
def load_cifar10_arrays():
    """CIFAR-10 on the GPU if there is one, pixel values scaled to [-1, 1].
    Returns train_X, train_y, test_X, test_y."""
    import pickle, warnings  # not in the d2l library preamble
    d2l.DATA_HUB['cifar10'] = (
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        '874905e36347c8536514d0a26261acf3bff89bc7')
    cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')
    def load_batch(name):
        with open(f'{cifar_dir}/{name}', 'rb') as f, warnings.catch_warnings():
            warnings.simplefilter('ignore')   # legacy NumPy pickle format
            batch = pickle.load(f, encoding='bytes')
        X = torch.tensor(batch[b'data'], dtype=torch.float32)
        return X.reshape(-1, 3, 32, 32) / 127.5 - 1, torch.tensor(
            batch[b'labels'])
    Xs, ys = zip(*[load_batch(f'data_batch_{i}') for i in range(1, 6)])
    test_X, test_y = load_batch('test_batch')
    device = d2l.try_gpu()
    return (torch.cat(Xs).to(device), torch.cat(ys).to(device),
            test_X.to(device), test_y.to(device))

cifar_X, cifar_y, test_X, test_y = load_cifar10_arrays()
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-1}
%%tab jax
#@save
def load_cifar10_arrays():
    """CIFAR-10 on the default device, pixel values scaled to [-1, 1].
    Returns train_X, train_y, test_X, test_y."""
    import pickle, warnings  # not in the d2l library preamble
    d2l.DATA_HUB['cifar10'] = (
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        '874905e36347c8536514d0a26261acf3bff89bc7')
    cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')
    def load_batch(name):
        with open(f'{cifar_dir}/{name}', 'rb') as f, warnings.catch_warnings():
            warnings.simplefilter('ignore')   # legacy NumPy pickle format
            batch = pickle.load(f, encoding='bytes')
        X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return (jnp.asarray(X, dtype=jnp.float32) / 127.5 - 1,
                jnp.asarray(batch[b'labels']))
    Xs, ys = zip(*[load_batch(f'data_batch_{i}') for i in range(1, 6)])
    test_X, test_y = load_batch('test_batch')
    return jnp.concatenate(Xs), jnp.concatenate(ys), test_X, test_y

cifar_X, cifar_y, test_X, test_y = load_cifar10_arrays()
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-2}
%%tab pytorch
#@save
class FeatureCNN(nn.Module):
    """Three conv blocks, global average pooling, and a linear head. Inputs
    are first resized bilinearly to resolution x resolution."""
    def __init__(self, feature_dim=128, resolution=32):
        super().__init__()
        self.resolution = resolution
        widths = [3, 32, 64, feature_dim]
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                           nn.BatchNorm2d(c_out), nn.ReLU(),
                           nn.MaxPool2d(2))
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(feature_dim, 10)

    def features(self, x):
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution),
                              mode='bilinear')
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.features(x))

#@save
def train_feature_cnn(cnn, X, y, num_epochs=15, max_lr=0.003,
                      batch_size=256, shift=0, seed=0):
    """Train on random horizontal flips, plus random shifts of up to `shift`
    pixels, with a one-cycle learning rate; end in evaluation mode."""
    gen = torch.Generator(X.device).manual_seed(seed)  # local random streams
    gen_shift = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(cnn.parameters())
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr, total_steps=num_epochs * (len(X) // batch_size),
        cycle_momentum=False)
    h, w = X.shape[2:]
    cnn.train()
    for epoch in range(num_epochs):
        order = torch.randperm(len(X), device=X.device, generator=gen)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            Xb, yb = X[order[i:i + batch_size]], y[order[i:i + batch_size]]
            flip = torch.rand(len(Xb), 1, 1, 1, device=X.device,
                              generator=gen) < 0.5
            Xb = torch.where(flip, Xb.flip(-1), Xb)
            a, b = torch.randint(0, 2 * shift + 1, (2,),
                                 generator=gen_shift).tolist()
            Xb = F.pad(Xb, (shift,) * 4, mode='reflect')[..., a:a + h, b:b + w]
            loss = F.cross_entropy(cnn(Xb), yb)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    cnn.eval()

torch.manual_seed(0)
cnn = FeatureCNN(resolution=64).to(device)
train_feature_cnn(cnn, cifar_X, cifar_y)
with torch.no_grad():
    pred = torch.cat([cnn(test_X[i:i + 1000]).argmax(1)
                      for i in range(0, len(test_X), 1000)])
print(f'CIFAR-10 test accuracy: {float((pred == test_y).float().mean()):.3f}')
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-2}
%%tab jax
#@save
class FeatureCNN(nnx.Module):
    """Three conv blocks, global average pooling, and a linear head. Inputs
    are first resized bilinearly to resolution x resolution."""
    def __init__(self, feature_dim=128, resolution=32, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.resolution = resolution
        widths = [3, 32, 64, feature_dim]
        self.convs = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME', rngs=rngs)
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        # Running averages decay 0.9; the variance is computed in two passes
        self.norms = nnx.List([nnx.BatchNorm(c, momentum=0.9,
                                             use_fast_variance=False,
                                             rngs=rngs)
                               for c in widths[1:]])
        self.head = nnx.Linear(feature_dim, 10, rngs=rngs)

    def features(self, x):
        if x.shape[1] != self.resolution:
            x = jax.image.resize(
                x, (x.shape[0], self.resolution, self.resolution, x.shape[3]),
                method='bilinear')
        for conv, norm in zip(self.convs, self.norms):
            x = nnx.max_pool(nnx.relu(norm(conv(x))),
                             window_shape=(2, 2), strides=(2, 2))
        return x.mean(axis=(1, 2))

    def __call__(self, x):
        return self.head(self.features(x))

#@save
def train_feature_cnn(cnn, X, y, num_epochs=15, max_lr=0.003,
                      batch_size=256, shift=0, seed=0):
    """Train on random horizontal flips, plus random shifts of up to `shift`
    pixels, with a one-cycle learning rate; end in evaluation mode."""
    schedule = optax.cosine_onecycle_schedule(
        num_epochs * (len(X) // batch_size), max_lr)
    opt = nnx.Optimizer(cnn, optax.adam(schedule), wrt=nnx.Param)
    h, w = X.shape[1:3]

    @nnx.jit
    def step(cnn, opt, Xb, yb, key):
        k1, k2 = jax.random.split(key)
        flip = jax.random.bernoulli(k1, 0.5, (len(Xb), 1, 1, 1))
        Xb = jnp.where(flip, Xb[:, :, ::-1], Xb)
        a, b = jax.random.randint(k2, (2,), 0, 2 * shift + 1)
        Xb = jnp.pad(Xb, ((0, 0), (shift, shift), (shift, shift), (0, 0)),
                     mode='reflect')
        Xb = jax.lax.dynamic_slice(Xb, (0, a, b, 0), (len(Xb), h, w, 3))
        def loss_fn(cnn):
            return optax.softmax_cross_entropy_with_integer_labels(
                cnn(Xb), yb).mean()
        opt.update(cnn, nnx.grad(loss_fn)(cnn))

    cnn.train()
    key = jax.random.key(seed)
    for epoch in range(num_epochs):
        key, kp = jax.random.split(key)
        order = jax.random.permutation(kp, len(X))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            key, ka = jax.random.split(key)
            idx = order[i:i + batch_size]
            step(cnn, opt, X[idx], y[idx], ka)
    cnn.eval()

cnn = FeatureCNN(resolution=64, rngs=nnx.Rngs(0))
train_feature_cnn(cnn, cifar_X, cifar_y)
pred = jnp.concatenate([cnn(test_X[i:i + 1000]).argmax(1)
                        for i in range(0, len(test_X), 1000)])
print(f'CIFAR-10 test accuracy: {float((pred == test_y).mean()):.3f}')
```

The classifier labels just under three quarters of the test images correctly; more epochs of this recipe change little.

### Scoring the Runs

We score 500 samples of each generator against 500 held-out sprites. As a reference we also score two disjoint sets of 500 real training images against each other: any distance between them is estimator noise and, for FD, finite-sample bias. Because the feature network is our own, we call the scores FD and $\mathrm{MMD}^2$ rather than FID and KID; they use the same formulas but are not comparable with published values. The last column reports the fraction of raw pixel values outside $[-1, 1]$. The backbone has no tanh, so its outputs can leave the range; we clamp them before scoring and for display.

```{.python .input #dcgan-scoring-the-two-runs}
%%tab pytorch
def features(model, imgs, batch_size=250):
    with torch.no_grad():
        return np.concatenate(
            [model.features(imgs[i:i + batch_size]).cpu().numpy()
             for i in range(0, len(imgs), batch_size)])

n = 500
torch.manual_seed(7)
z_score = torch.randn(n, 100, device=device)
feat_real = features(cnn, holdout_imgs[:n])
feat_a, feat_b = features(cnn, train_imgs[:n]), features(cnn,
                                                          train_imgs[n:2 * n])
gens = {'DCGAN': lambda z: net_G(z[:, :, None, None])}
gens.update({name: ema_G for name, (ema_G, hist) in runs.items()})
print(f'{"run":18s}{"FD":>8s}{"MMD^2":>8s}{"out-of-range":>14s}')
print(f'{"real vs. real":18s}{fid_score(feat_a, feat_b):8.2f}'
      f'{kid_score(feat_a, feat_b):8.2f}')
for name, G in gens.items():
    with torch.no_grad():
        raw = G(z_score)
    oob = float((raw.abs() > 1).float().mean())
    feat = features(cnn, raw.clamp(-1, 1))
    print(f'{name:18s}{fid_score(feat, feat_real):8.2f}'
          f'{kid_score(feat, feat_real):8.2f}{oob:14.3f}')
out = raw[raw.abs() > 1]  # out-of-range values of the last arm
print(f'{name}: {float((out > 1).float().mean()):.0%} of them above +1, '
      f'{float((out.abs() < 1.1).float().mean()):.0%} within 0.1 of the range')
print(f'real pixel values equal to +1: '
      f'{float((holdout_imgs == 1).float().mean()):.0%}')
```

```{.python .input #dcgan-scoring-the-two-runs}
%%tab jax
def features(model, imgs, batch_size=250):
    return np.concatenate(
        [np.asarray(model.features(imgs[i:i + batch_size]))
         for i in range(0, len(imgs), batch_size)])

n = 500
z_score = jax.random.normal(jax.random.key(7), (n, 100))
feat_real = features(cnn, holdout_imgs[:n])
feat_a, feat_b = features(cnn, train_imgs[:n]), features(cnn,
                                                          train_imgs[n:2 * n])
gens = {'DCGAN': lambda z: net_G(z[:, None, None, :])}
gens.update({name: ema_G for name, (ema_G, hist) in runs.items()})
print(f'{"run":18s}{"FD":>8s}{"MMD^2":>8s}{"out-of-range":>14s}')
print(f'{"real vs. real":18s}{fid_score(feat_a, feat_b):8.2f}'
      f'{kid_score(feat_a, feat_b):8.2f}')
for name, G in gens.items():
    raw = G(z_score)
    oob = float((jnp.abs(raw) > 1).mean())
    feat = features(cnn, jnp.clip(raw, -1, 1))
    print(f'{name:18s}{fid_score(feat, feat_real):8.2f}'
          f'{kid_score(feat, feat_real):8.2f}{oob:14.3f}')
out = raw[jnp.abs(raw) > 1]  # out-of-range values of the last arm
print(f'{name}: {float((out > 1).mean()):.0%} of them above +1, '
      f'{float((jnp.abs(out) < 1.1).mean()):.0%} within 0.1 of the range')
print(f'real pixel values equal to +1: '
      f'{float((holdout_imgs == 1).mean()):.0%}')
```

The real-versus-real row shows the finite-sample behavior described above: FD is positive for two samples of the same distribution, while the unbiased $\widehat{\mathrm{MMD}}^2$ is close to zero and can come out negative. Against this floor, the collapsed arm scores dozens to hundreds of times higher. The two penalized arms score several times the floor (between about five and fifteen times in our runs), consistent with the soft outlines in their grids, and overlap each other across seeds. DCGAN lands close to them, within a factor of about two: the 2015 architecture keeps the same classic loss trainable that collapses on the unnormalized backbone. $\widehat{\mathrm{MMD}}^2$ agrees: the collapsed arm scores far above the others, and the other three lie within a factor of about two of each other. About one in six raw pixel values of a penalized generator lies outside $[-1, 1]$, nearly all just above $+1$. Because 45% of the real pixel values equal $+1$ exactly, the white background, this is background overshoot, and clamping maps it back to white.

Each score depends on the feature network as well as on the generator: our classifier, trained on photographs, judges sprites, as Inception judges every domain by what helps to classify ImageNet, and a different feature network can change the values and reorder similar models (Exercise :exref:`ex_dcgan-retrain-feature-network-this`); recent work therefore also reports FD in the features of the self-supervised DINOv2 network :cite:`Oquab.Darcet.Moutakanni.ea.2024,Karras.Aittala.Kynkaanniemi.ea.2024`. The evaluation features can also leak into training: a critic built on ImageNet features can lower Inception-based FID by matching ImageNet class statistics without better images :cite:`Kynkaanniemi.Karras.Aittala.ea.2023`. Preprocessing matters too, since aliased resizing shifts FID by as much as the differences between published methods :cite:`Parmar.Zhang.Zhu.2022`. Finally, the finite-sample bias depends on the model as well as on $n$, so at small $n$ it alone can reverse a ranking :cite:`Chong.Forsyth.2020` (Exercise :exref:`ex_dcgan-measure-finite-sample-bias`).

## Limited Data

### Overfitting and Augmentation
:label:`subsec_gan_limited_data`

Forty thousand sprites is a small dataset for a critic with 1.6 million parameters. In 12,000 steps at batch size 64 the critic sees each training image about twenty times. Instead of estimating a density ratio, it can start to recognize individual training images: an overfit critic scores its training images higher than held-out real images, and its gradient then pulls the generator toward the finite training set rather than the distribution behind it. The train--holdout gap printed after each run measures this. In our runs the penalized critics' gaps stay within a few hundredths of zero, so at this budget they score held-out sprites as they score their training images; the unpenalized critic's gap is noisy and lives on a larger score scale (Exercise :exref:`ex_dcgan-scale-free-overfitting-statistic`). As the dataset shrinks, a critic of fixed capacity separates its training images from the rest of image space sooner, and the gap becomes the first sign of trouble :cite:`Karras.Aittala.Hellsten.ea.2020`.

Augmentation makes memorization harder, but where it is applied decides what the generator learns. Our loop flips only the real images, which assumes that the sprite distribution is mirror-symmetric; color jitter or cutout applied only to real images would teach the generator to produce color shifts and erased patches. *Differentiable augmentation* instead applies the same random transformation to the real and the generated inputs of the critic in both updates :cite:`Zhao.Liu.Lin.ea.2020`, with the transformation inside the generator's gradient path. The critic then sees many transformed versions of each training image, and the matching problem stays the original one as long as the transformation cannot map two different distributions to the same one; :citet:`Karras.Aittala.Hellsten.ea.2020` call such a transformation *non-leaking*.

Adaptive discriminator augmentation sets the augmentation probability from an overfitting statistic, $r_t = \mathbb{E}[\operatorname{sign} D(x_{\mathrm{train}})]$, and with it StyleGAN2 reaches competitive quality on a few thousand images :cite:`Karras.Aittala.Hellsten.ea.2020`. That statistic reads the sign of a single score and so presumes a meaningful zero level, which a pairing critic does not have: adding a constant to its scores changes nothing in its loss. A pairing critic needs a pairwise statistic instead (Exercise :exref:`ex_dcgan-scale-free-overfitting-statistic`).

### Checking for Copies

A memorizing critic and a copying generator call for different checks. To test the generator, we embed the 500 samples of the RpGAN + $R_1$ + $R_2$ arm in the classifier's feature space and find each one's nearest training image, calibrated by doing the same for held-out sprites. A copying generator would sit closer to its training images than new real images do. Because variants of one species fall on both sides of the split, held-out sprites often have a close training neighbor, which keeps their distances low and makes the test conservative.

```{.python .input #dcgan-training-on-limited-data}
%%tab pytorch
ema_G = runs['RpGAN + R1 + R2'][0]
with torch.no_grad():
    gen = ema_G(z_score).clamp(-1, 1)
feat_train_all = features(cnn, train_imgs)

def nearest_train(feat):
    d2 = ((feat ** 2).sum(1, keepdims=True) - 2 * feat @ feat_train_all.T
          + (feat_train_all ** 2).sum(1))
    return d2.argmin(1), np.sqrt(np.maximum(d2.min(1), 0))

nn_idx, d_gen = nearest_train(features(cnn, gen))
_, d_holdout = nearest_train(feat_real)
print(f'median distance to nearest training image: '
      f'generated {np.median(d_gen):.2f}, '
      f'held-out real {np.median(d_holdout):.2f}')
closest = np.argsort(d_gen)[:8]
pairs = torch.cat([gen[torch.as_tensor(closest, device=device)],
                   train_imgs[torch.as_tensor(nn_idx[closest],
                                              device=device)]])
d2l.show_images(pairs.cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=2, num_cols=8);
```

```{.python .input #dcgan-training-on-limited-data}
%%tab jax
ema_G = runs['RpGAN + R1 + R2'][0]
gen = jnp.clip(ema_G(z_score), -1, 1)
feat_train_all = features(cnn, train_imgs)

def nearest_train(feat):
    d2 = ((feat ** 2).sum(1, keepdims=True) - 2 * feat @ feat_train_all.T
          + (feat_train_all ** 2).sum(1))
    return d2.argmin(1), np.sqrt(np.maximum(d2.min(1), 0))

nn_idx, d_gen = nearest_train(features(cnn, gen))
_, d_holdout = nearest_train(feat_real)
print(f'median distance to nearest training image: '
      f'generated {np.median(d_gen):.2f}, '
      f'held-out real {np.median(d_holdout):.2f}')
closest = np.argsort(d_gen)[:8]
pairs = jnp.concatenate([gen[closest], train_imgs[nn_idx[closest]]])
d2l.show_images(np.asarray(pairs) / 2 + 0.5, num_rows=2, num_cols=8);
```

Generated samples sit about two and a half times farther from their nearest training image than held-out sprites do, the opposite of copying. The top row of the grid shows the eight generated samples closest to the training set, mostly sparse shapes on white, and the bottom row their nearest training images. The neighbors look unrelated, which says as much about the features, statistics of local texture and color, as about the generator. The test detects direct copying; it cannot rule out recombination of memorized parts.

We can now build an image generator from either decade, change its loss one ingredient at a time, and score the result with a number whose meaning we know. The generator still answers no request: it returns a random sprite, never the one we ask for. :numref:`sec_gan_conditional` gives both players the request.

## Summary

DCGAN made the classic loss train on images through architecture: transposed convolutions, batch normalization except at the discriminator's input and the generator's output, and Adam with $\beta_1 = 0.5$. A modern backbone drops normalization and relies on the loss. On it, the classic loss without a penalty collapses within 6,000 steps, while with the zero-centered penalties $R_1$ and $R_2$ both the classic and the pairing loss train to diverse sprites and tie. The penalty does not stop the critic from ranking; it keeps its squared input gradient about a hundred times smaller, so the rank weight stays graded instead of saturating near zero. FID and KID are the Gaussian Wasserstein-2 distance and an unbiased cubic-kernel MMD in a feature space; their values depend on the feature network, the preprocessing, and $n$, so we read them against a real-versus-real floor. On limited data the critic memorizes first, and a non-leaking augmentation of both of its inputs counters this.

## Exercises

1. **Checkerboards from stride.** A transposed convolution maps an
   $n \times n$ input to an output of side
   $\textrm{stride} \cdot (n - 1) + \textrm{kernel} - 2 \cdot \textrm{pad}$.
   `G_block` uses kernel 4, stride 2, and padding 1.
    1. Find every combination of kernel size (at most 6) and padding for
       which a transposed convolution with stride 2 exactly doubles the
       side $n$.
    1. Consider a one-dimensional transposed convolution with stride 2 whose
       kernel entries all equal one, applied to a constant input. Away from
       the borders, how many input positions contribute to each output
       position for kernel size 3, and how many for kernel size 4? The first case is
       what :citet:`Odena.Dumoulin.Olah.2016` call uneven overlap.
    1. Now let the kernel $(w_0, w_1, w_2, w_3)$ of size 4 be
       arbitrary. Show that, away from the borders, a constant input
       produces an output that alternates between two values. Under which
       condition on the kernel do the two values agree? What does the
       two-dimensional version of this argument predict for the output of
       the transposed convolution of a `G_block` on a constant feature map?
    1. `Generator` upsamples bilinearly and then applies a $3 \times 3$
       convolution with stride 1. What does this pair of operations produce
       from a constant input away from the borders? Why does this remove the
       periodic pattern for every choice of the learned weights?
   :exlabel:`ex_dcgan-checkerboards-from-stride`
1. **Penalty weight and resolution.** The section uses $\gamma = 10$ for
   $64 \times 64$ sprites with pixel values in $[-1, 1]$, and
   :citet:`Huang.Gokaslan.Kuleshov.ea.2024` tune $\gamma$ separately for
   every dataset and resolution. This problem shows that part of this
   dependence follows from the units of the input gradient in the penalty.
    1. Let $U$ upsample a $64 \times 64$ image to $128 \times 128$ by
       copying every pixel into a $2 \times 2$ block, and let $A$ average
       every $2 \times 2$ block, so that $AU$ is the identity. The critic
       $D' = D \circ A$ scores every upsampled image $Ux$ exactly as $D$
       scores $x$. Express $\|\nabla D'(Ux)\|^2$ in terms of
       $\|\nabla D(x)\|^2$. Which weight $\gamma'$ makes the penalized game
       on the upsampled data equivalent to the game with weight $\gamma$ on
       the original data?
    1. Suppose that the pixel values are scaled to $[0, 1]$ instead of
       $[-1, 1]$. How must $\gamma$ change for the penalized game to stay
       equivalent?
    1. In its automatic configuration, the official implementation of
       :citet:`Karras.Aittala.Hellsten.ea.2020` sets
       $\gamma = 0.0002\, r^2 / B$ for resolution $r$ and batch size $B$.
       Which factor of this heuristic does the upsampling argument account
       for?
   :exlabel:`ex_dcgan-penalty-weight-resolution`
1. [code] **A scale-free overfitting statistic.** The cell that runs the
   three arms prints the critic's train--holdout gap. The unpenalized critic
   scores on a much larger scale than the penalized critics, so the gaps
   cannot be compared directly. :citet:`Karras.Aittala.Hellsten.ea.2020`
   consider two statistics,

   $$r_v = \frac{\mathbb{E}[D_{\textrm{train}}] - \mathbb{E}[D_{\textrm{holdout}}]}{\mathbb{E}[D_{\textrm{train}}] - \mathbb{E}[D_{\textrm{generated}}]}, \qquad r_t = \mathbb{E}\big[\operatorname{sign}(D_{\textrm{train}})\big],$$

   where $D_{\textrm{train}}$, $D_{\textrm{holdout}}$, and
   $D_{\textrm{generated}}$ denote critic scores on training images,
   held-out images, and samples of the current generator.
    1. Show that $r_v$ does not change when the critic $D$ is replaced by
       $aD + c$ with $a > 0$, whereas the train--holdout gap scales with $a$
       and $r_t$ changes with $c$.
    1. By problem :exref:`ex_relativistic-prove-shift-invariance-ev`c of
       :numref:`sec_gan_relativistic`, the sign of a single score carries
       no information about a critic trained on the pairing objective.
       Consider instead the pairwise statistic
       $\mathbb{E}\big[\operatorname{sign}(D_{\textrm{train}} - D_{\textrm{holdout}})\big]$
       over independent pairs of a training and a held-out image. Under
       which transformations of $D$ is it invariant? Which value does it
       take when the critic scores training and held-out images alike?
    1. Modify `train_backbone` so that it also returns the raw generator
       `net_G`. For all three arms, compute $r_v$ and the pairwise statistic
       of (b) from 1,024 training images, 1,024 held-out images, and 1,024
       samples of the raw generator. Do the normalized statistics change the
       comparison of the arms?
   :exlabel:`ex_dcgan-scale-free-overfitting-statistic`
1. **FID across dimensions.** In one dimension, the squared
   Wasserstein-2 distance between two distributions with quantile functions
   $Q_p$ and $Q_q$ is
   $\int_0^1 \big(Q_p(u) - Q_q(u)\big)^2\, du$.
    1. Using the Gaussian quantile function
       $Q_p(u) = \mu_p + \tau_p\, Q_0(u)$, where $Q_0$ is the quantile
       function of $\mathcal{N}(0, 1)$, show that for two univariate
       Gaussians with standard deviations $\tau_p$ and $\tau_q$ this
       distance equals $(\mu_p - \mu_q)^2 + (\tau_p - \tau_q)^2$.
    1. Evaluate :eqref:`eq_gan_fid` for $\mathcal{N}(0, I_d)$ against
       $\mathcal{N}(0, c\, I_d)$ with $c > 0$, and for $\mathcal{N}(0, I_d)$
       against $\mathcal{N}(\mu_0 \mathbf{1}, I_d)$, where $\mathbf{1}$ is
       the all-ones vector.
    1. The features of a generator are 10% too spread out along every
       coordinate, so that $c = 1.21$. Assuming isotropic features, compare
       its FID under a 128-dimensional feature map, as in this section, with
       its FID under the 2,048-dimensional Inception features of published
       FID. What does the comparison imply for FID values computed with
       different feature networks?
   :exlabel:`ex_dcgan-one-dimension-squared-wasserstein`
1. [code] **Finite-sample bias.** The real-versus-real reference row
   compares two disjoint sets of 500 training images.
    1. Split the held-out sprites into two disjoint subsets of $n$ images
       each, and compute the real-versus-real FD and
       $\widehat{\mathrm{MMD}}^2$ with `fid_score` and `kid_score` for
       $n \in \{100, 250, 500, 1000, 2000\}$, with five random splits per
       $n$. How does each reference value move with $n$?
    1. Plot the FD estimates against $1/n$, fit a line, and extrapolate it
       to $1/n \to 0$, which is the bias-corrected estimate of
       :citet:`Chong.Forsyth.2020`. How close to zero is the extrapolated
       value? What does the slope imply for the bias at $n = 500$?
    1. Repeat the fit for the FD between $n$ samples of the EMA generator
       of the RpGAN + $R_1$ + $R_2$ arm and $n$ held-out images. Is the
       slope the same as for the real-versus-real pairs? Why can two
       generators be ranked differently at $n = 500$ and in the limit
       $n \to \infty$?
   :exlabel:`ex_dcgan-measure-finite-sample-bias`
1. [code] **Precision and recall.**
   :citet:`Kynkaanniemi.Karras.Laine.ea.2019` estimate the support of a set
   of features $A$ by a union of balls, one around every point of $A$,
   whose radius is the distance from that point to its $K$-th nearest
   neighbor in $A$. Precision is the fraction of generated features that
   lie inside the support estimate of the real features. Recall is the
   fraction of real features that lie inside the support estimate of the
   generated features.
    1. Implement both metrics on the CIFAR-CNN features, using `feat_real`
       for the 500 held-out images and 500 samples of every generator, with
       $K = 3$ as in the paper. Report precision and recall for every
       generator and for two disjoint sets of 500 training images. Repeat
       with $K \in \{1, 10\}$. How do the two metrics respond to $K$?
    1. Let a generator collapse onto $J$ distinct images, each produced by
       an equal share of the 500 samples. For $K = 3$, find the largest $J$
       for which every radius of its support estimate is zero, and the
       recall that every such generator receives. What does this imply
       about recall as a count of the modes that a generator covers? What
       determines the precision of such a generator?
   :exlabel:`ex_dcgan-precision-recall`
1. [code] **Initialization, latent injection, and collapse.** ● Without a
   penalty, the classic loss loses its diversity within 6,000 steps from
   the fan-in initialization of `Generator` and `Discriminator`. Define the
   diversity of a generator as the mean pairwise distance between the
   CIFAR-CNN features of its samples from `z_show`, divided by the same
   statistic for 64 held-out images. Keep every run to 5,000 steps.
    1. Modify `train_backbone` so that it records the diversity of the raw
       generator every 500 steps. Train the unpenalized arm from the fan-in
       initialization and with every weight drawn from a normal
       distribution with mean zero and standard deviation 0.01, 0.02, or
       0.05, where 0.02 is the DCGAN convention. Does any initialization keep the diversity
       above one half? Does the rise of $\mathbb{E}\|\nabla_x D(x)\|^2$
       precede the fall in diversity?
    1. Repeat the fan-in run with three seeds. How much does the step at
       which the diversity first falls below one half vary? What does this
       imply for a comparison of two losses that reports a single run of an
       unregularized game?
    1. R3GAN feeds the latent code through a *basis layer*: a linear map
       sends $z$ to 512 coefficients, each of which scales its own learned
       $4 \times 4$ map, and the scaled maps replace the constant and the
       mixing input. Implement it in `Generator` in place of the
       concatenation and count the parameters of the two latent paths.
       Train the unpenalized and the RpGAN + $R_1$ + $R_2$ arm with the
       basis layer. Does the way $z$ enters the network change whether, or
       how soon, the unpenalized arm collapses?
   :exlabel:`ex_dcgan-rerun-arm-classic-loss`
1. **Moments behind FID and KID.** ● FID depends on the two feature
   distributions only through their means and covariances. KID uses the
   polynomial kernel $k(u, v) = (u^\top v / d + 1)^3$ in the
   estimator :eqref:`eq_gan_kid`.
    1. Expand the kernel with the binomial theorem and write
       $(u^\top v)^j = \langle u^{\otimes j}, v^{\otimes j} \rangle$ to show
       that the population value of $\mathrm{MMD}^2$ is

       $$\sum_{j=1}^{3} \binom{3}{j} d^{-j}\, \big\| \mathbb{E}_p[u^{\otimes j}] - \mathbb{E}_q[u^{\otimes j}] \big\|^2.$$

       Which moments of the feature distributions does KID compare?
    1. In one dimension, let $p = \mathcal{N}(0, 1)$. Compute the
       population FID and $\mathrm{MMD}^2$ of $p$ against two distributions
       $q$: the distribution of $X - 1$ for $X$ exponential with rate 1,
       which has mean 0, variance 1, and third moment 2, and the equal
       mixture of $\mathcal{N}(a, 1 - a^2)$ and $\mathcal{N}(-a, 1 - a^2)$
       for $0 < a < 1$.
    1. Which differences between two feature distributions are invisible
       to both metrics? A kernel is *characteristic* when its mean embedding
       $\mathbb{E}_p[k(u, \cdot)]$ determines the distribution $p$, so that the
       population $\mathrm{MMD}^2$ vanishes only for $q = p$. Is the kernel
       of KID characteristic? Does the $\mathrm{MMD}^2$ with a Gaussian
       RBF kernel distinguish the mixture in (b) from $p$?
   :exlabel:`ex_dcgan-moments-fid-kid`
1. [code] **Dependence on the feature network.** ● The scores depend on
   the feature network as well as on the generators.
    1. Retrain `FeatureCNN` on Fashion-MNIST, loaded with
       `d2l.FashionMNIST`, instead of CIFAR-10: scale the pixel values to
       $[-1, 1]$, replicate the grayscale channel three times, and upsample
       to $64 \times 64$ as before. Rescore the real-versus-real reference
       and every generator. Do the FD and $\widehat{\mathrm{MMD}}^2$ values
       change substantially? Does the ordering of the generators change?
    1. The collapsed arm scores several times worse than the others, so
       its place in the ordering says little about the ordering of similar
       models. Degrade 500 held-out sprites in two ways: blur them with a
       $3 \times 3$ box filter, and add Gaussian noise with a standard
       deviation chosen so that the FDs of the two degraded sets against
       500 training images agree within 10% under the CIFAR-CNN. Score both
       degraded sets under the Fashion-MNIST network. Which defect does each
       feature network penalize more?
   :exlabel:`ex_dcgan-retrain-feature-network-this`
1. **Leaking augmentations.** ● Let $F$ denote the horizontal flip of an
   image, and for a distribution $r$ over images let $F_\# r$ denote the
   distribution of the flipped images. An augmentation that flips each image
   with probability $p_{\mathrm{aug}}$ maps $r$ to
   $T_{p_{\mathrm{aug}}} r = (1 - p_{\mathrm{aug}})\, r + p_{\mathrm{aug}}\, F_\# r$.
    1. `sample_real` flips only the real images, with
       $p_{\mathrm{aug}} = 1/2$, so the generator is trained toward
       $T_{1/2}\, p$ rather than $p$, and the two agree only if
       $F_\# p = p$. Suppose that a fraction $\alpha$ of the sprites faces
       left and has no mirrored counterpart in the dataset, while every
       other sprite equals its own mirror image. Which fraction of the
       samples of the optimal generator faces right?
    1. A differentiable augmentation :cite:`Zhao.Liu.Lin.ea.2020` applies
       $T_{p_{\mathrm{aug}}}$ to the real and to the generated images. Show
       that $T_{p_{\mathrm{aug}}} q = T_{p_{\mathrm{aug}}} p$ implies $q = p$
       whenever $p_{\mathrm{aug}} \neq 1/2$, using $F_\# F_\# r = r$. For
       $p_{\mathrm{aug}} = 1/2$, describe the set of generator distributions
       that the critic cannot distinguish from $p$, and give one in this set
       that never produces an unmirrored sprite.
    1. Let $S$ rotate an image by $90^\circ$. An augmentation replaces each
       image, with probability $p_{\mathrm{aug}}$, by its rotation under
       $S$, $S^2$, $S^3$, or the identity, chosen uniformly. Show that this
       augmentation is non-leaking for every $p_{\mathrm{aug}} < 1$ and
       leaking at $p_{\mathrm{aug}} = 1$.
       :citet:`Karras.Aittala.Hellsten.ea.2020` observe that the generated
       images are always oriented correctly in practice while
       $p_{\mathrm{aug}}$ stays below about 0.85, but not reliably above.
       Why can a finite critic trained on a finite dataset fail before the
       limit $p_{\mathrm{aug}} = 1$?
   :exlabel:`ex_dcgan-augmentation-leaks`

[Discussions](https://d2l.discourse.group/t/36812)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.5]{.kicker}

Adversarial Image Generation<br>
**transposed convolutions · DCGAN · a backbone without normalization · three losses, one change at a time · FID and KID · copying**
:::
:::

::: {.slide title="A Scatter Plot Showed Whether $q$ Matched $p$; 12,288 Pixels Do Not"}
[The problem]{.kicker}

- In two dimensions we looked at the samples and saw the fit.
- A $64 \times 64$ RGB image has $12{,}288$ values. The generator must
  produce all of them from 100 random numbers.
- A grid of samples shows a collapse, but two good generators look alike.

::: {.d2l-note}
**Which network turns noise into an image, and which number says how
close its samples are to the data?**
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the output size of a transposed convolution;
- build the 2015 DCGAN and a 2024 backbone without normalization;
- run a loss comparison that changes one ingredient at a time;
- read a critic's steepness, ranking accuracy, and rank weight;
- derive FID and KID from the chapter's two closed forms;
- check whether a generator copies its training images.

**Running example:** 40,597 Pokemon sprites at $64 \times 64$, 36,538 for
training and 4,059 held out. **Idea:** the loss, not the architecture, keeps
the game stable.
:::

::: {.slide title="This Deck Moves the Chapter's Loss from Scatter Plots to Images"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Generative Adversarial Networks | what a classifier of samples computes |
| Adversarial Objectives and Divergences | which loss gives which distance |
| Relativistic Objectives | what changes when the critic ranks pairs |
| Gradient Penalties and Convergence | why training circles, and what stops it |
| **Adversarial Image Generation** | **which network, and how to score images** |
| Conditional Generation | how to ask for a class |
| Adversarial Losses Beyond GANs | where a learned comparison remains |
:::

::: {.col .narrow}
- **Uses:** the classic and pairing losses, the two zero-centered
  penalties.
- **Skips:** architecture zoos, large-scale training.
:::
:::
:::

::: {.slide title="A Transposed Convolution Stamps the Kernel and Adds the Overlaps"}
Each input pixel stamps a scaled copy of the kernel; stamps start one
stride apart; overlaps add. Input $(1, 2)$, kernel $(1, 1, 1, 1)$, stride 2:

@!dcgan-generator-2

$$\text{side} = \text{stride}\cdot(n - 1) + \text{kernel} - 2\cdot\text{pad}$$

Kernel 4, stride 2, padding 1: side $2n$.

::: {.d2l-note}
Five such layers grow $1 \times 1$ into $64 \times 64$.
:::
:::

::: {.slide title="DCGAN Stabilized the Classic Loss with Architecture"}
![](../img/mdl-gan-dcgan.svg){width=85%}

Batch norm everywhere except the discriminator's input and the generator's
output; ReLU and tanh in $G$, leaky ReLU in $D$; Adam with $\beta_1 = 0.5$.
:::

::: {.slide title="DCGAN Trains the Classic Loss to Diverse Sprites"}
@!dcgan-training-with-the-classic-loss-3

The critic stays well ahead (its loss far below $\log 2$), yet the samples
stay diverse: batch norm and the architecture keep this game trainable.
:::

::: {.slide title="Check: What Does a Size-3, Stride-2 Transposed Convolution Do to a Constant?"}
[Check]{.kicker}

All-ones kernel of length 3, stride 2, constant input. How many stamps
cover each output position away from the borders?

::: {.fragment .fade-up}
Alternately 2 and 1: the output alternates between two values, a
checkerboard. With kernel size 4 every position is covered twice.
:::
:::

::: {.slide title="R3GAN Removes a Decade of Fixes and Repairs the Loss Instead"}
StyleGAN2 on FFHQ-256, rebuilt step by step (FID, lower is better):

| config | change | FID |
|:--|:--|--:|
| A | StyleGAN2 | 7.5 |
| B | remove ten tricks | 12.5 |
| C | pairing loss + $R_1$ + $R_2$ | 11.7 |
| D | modern ResNet backbone | 10.0 |
| E | grouped convolutions, bottlenecks | 7.0 |

::: {.d2l-note}
No normalization, $\beta_1 = 0$, a well-behaved loss: violating these often
breaks training.
:::

[Huang et al., 2024]{.small}
:::

::: {.slide title="A Minimal Backbone: Bilinear Resampling, Leaky ReLU, No Normalization"}
![](../img/mdl-gan-backbone.svg)

Fan-in initialization: no layer rescales activations, so the weights must
keep their variance, $\operatorname{Var} w = 2 / ((1 + 0.2^2)\, n_{\mathrm{in}})$.
:::

::: {.slide title="One Backbone, Three Losses, One Change at a Time"}
| arm | critic loss | penalty | steps |
|:--|:--|:--|--:|
| NS | classic logistic | none | 6,000 |
| NS + $R_1$ + $R_2$ | classic logistic | $\gamma = 10$ | 12,000 |
| RpGAN + $R_1$ + $R_2$ | pairing | $\gamma = 10$ | 12,000 |

Same seed, optimizer (Adam, $\beta_1 = 0$), batch 64, flips, weight EMA.
Rows 1 and 2 isolate the penalty; rows 2 and 3 isolate pairing.
:::

::: {.slide title="The Critic Step: Loss Plus $\gamma/2$ Times Both Squared Input Gradients"}
```{.python #dcgan-exercises}
fake = net_G(noise(64)).detach()
r1, r2 = d2l.r1_r2_penalty(net_D, X, fake)  # per sample
loss = loss_D(net_D, X, fake)               # classic or pairing
penalty = gamma / 2 * (r1 + r2).mean()
(loss + penalty).backward()                 # critic step
loss_G(net_D, X2, net_G(noise(64))).backward()  # generator
```

$R_1 = \|\nabla_x D(x)\|^2$ on real images, $R_2$ on generated ones;
the loss is passed in as a function.
:::

::: {.slide title="Without a Penalty the Critic Is Steep; All Three Rank Equally Well"}
@!dcgan-loss-a-b-on-one-backbone-4

- Steepness $\mathbb{E}\|\nabla_x D\|^2$: about 100× larger without $R_1$.
- Ranking accuracy: near 1 in all three arms.
- Rank weight $w_{\mathrm{Rp}}$: near 0 without the penalty, 0.25 to 0.4 with it.
:::

::: {.slide title="The Unpenalized Arm Collapses; Both Penalized Arms Stay Diverse"}
@!dcgan-loss-a-b-on-one-backbone-3

A training failure of an unregularized game: nothing damps the critic,
whose squared input gradient ends about 100× larger than with the penalty,
and every sample gets the same push.
:::

::: {.slide title="Near Zero, the Rank Weight No Longer Tells Better Samples from Worse"}
Rank weight of a generated $x'$, the probability that it outranks a real
image:

$$w_{\mathrm{Rp}}(x') = \mathbb{E}_{x \sim p}\,\sigma\big(D(x') - D(x)\big)$$

- Near 0 for every $x'$: all samples ranked far below all real ones. The
  non-saturating generator, weight $1 - w_{\mathrm{Rp}}$, pushes all alike.
- Graded (0.25 to 0.4): worse samples pushed harder.
- Replace $\sigma$ by a step: one minus the ranking accuracy, near 0 in
  every arm.

::: {.d2l-note}
The penalty keeps the critic's verdict graded, not its ranking wrong.
:::
:::

::: {.slide title="Check: Which Run Would You Stop, from the Traces Alone?"}
[Check]{.kicker}

Three runs, all with ranking accuracy near 0.98. One has
$\mathbb{E}\|\nabla_x D\|^2$ near 1 and rank weight near 0; two sit at
0.01 and 0.3.

::: {.fragment .fade-up}
The first: its critic separates the sets by a steep wall, the step that
precedes collapse. Ranking accuracy alone would not have told you.
:::
:::

::: {.slide title="FID Is the Wasserstein-2 Distance between Two Gaussians"}
Fit $\mathcal{N}(\mu_p, \Sigma_p)$ to real features and
$\mathcal{N}(\mu_q, \Sigma_q)$ to generated ones:

$$\mathrm{FID} = \|\mu_p - \mu_q\|^2 + \operatorname{tr}\big(\Sigma_p +
\Sigma_q - 2(\Sigma_p^{1/2} \Sigma_q \Sigma_p^{1/2})^{1/2}\big)$$

- 1-D, standard deviations $\tau_p, \tau_q$: $(\mu_p - \mu_q)^2 + (\tau_p - \tau_q)^2$.
- $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(0.5, 1.1^2)$: $0.25 + 0.01 = 0.26$.
- Two moments only; biased upward at finite $n$.
:::

::: {.slide title="KID Is the Unbiased MMD with a Cubic Kernel"}
$k(u, v) = (u^\top v / d + 1)^3$ on features $\phi$; drop the self-pairs:

$$\widehat{\mathrm{MMD}}^2 = \tfrac{1}{n_p(n_p-1)}\textstyle\sum_{i \neq i'} k_{ii'}
+ \tfrac{1}{n_q(n_q-1)}\sum_{j \neq j'} k'_{jj'} - \tfrac{2}{n_p n_q}\sum_{i, j} k''_{ij}$$

with $k_{ii'} = k(\phi_i, \phi_{i'})$, $k'_{jj'} = k(\phi'_j, \phi'_{j'})$,
$k''_{ij} = k(\phi_i, \phi'_j)$, $\phi_i = \phi(x_i)$ real,
$\phi'_j = \phi(x'_j)$ generated.

- Within-set similarity minus across-set similarity.
- Unbiased: real vs. real fluctuates around 0, may be negative.
- Fixed kernel, learned features: the feature network decides what counts.
:::

::: {.slide title="Every Generator Scores Several Times Above the Real-versus-Real Floor"}
@!dcgan-scoring-the-two-runs

- Collapsed arm: dozens to hundreds of times the floor.
- Penalized arms and DCGAN: several times the floor, within 2× of each
  other.
- Out-of-range pixels: the white background overshooting $+1$.
:::

::: {.slide title="A Score Means What Its Feature Network Sees"}
- **Features:** a CIFAR-10 CNN judges sprites, as Inception judges other
  domains.
- **Leakage:** a critic built on ImageNet features can lower ImageNet-based
  FID without better images.
- **Resizing:** aliased resizing shifts FID by method-sized margins.
- **Sample size:** the finite-$n$ bias depends on the model.

[Kynkäänniemi et al., 2023; Parmar et al., 2022; Chong & Forsyth, 2020]{.small}
:::

::: {.slide title="The Generator Does Not Copy Its Training Images"}
@!dcgan-training-on-limited-data

Generated samples sit about 2.5× farther from their nearest training image
than held-out sprites do. The neighbors look unrelated: the features see
texture and color.
:::

::: {.slide title="On Small Datasets the Critic Memorizes First"}
- A critic with 1.6 M parameters sees each sprite about 20 times.
- Overfit critic: higher scores on training than on held-out images.
- Augment **both** inputs of $D$, differentiably; real-only augmentation
  teaches $G$ the augmented distribution.
- Non-leaking: the augmentation must not map two distributions to one.

[Zhao et al., 2020; Karras et al., 2020]{.small}
:::

::: {.slide title="Check: Flip Only the Real Images. What Does $G$ Learn?"}
[Check]{.kicker}

A fraction $\alpha$ of the sprites faces left and has no mirror image in
the data. `sample_real` flips real images with probability 1/2.

::: {.fragment .fade-up}
The flipped distribution: $\alpha/2$ of the samples face right. Flipping
only the reals is safe only if $p$ is mirror-symmetric.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Transposed convolution: side $\text{stride}\cdot(n-1) + \text{kernel} - 2\cdot\text{pad}$; five layers grow $1 \to 64$.
- DCGAN: batch norm makes the classic loss train on 36,538 sprites.
- No normalization, no penalty: collapse within 6,000 steps.
- $R_1 + R_2$ at $\gamma = 10$: squared input gradient about 100× smaller, rank weight 0.25 to 0.4, diverse sprites.
- Classic and pairing losses tie once both are penalized.
- FID = Gaussian $W_2^2$, KID = unbiased cubic-kernel MMD; read against the real-versus-real floor.

**Next:** ask for a class, and check that the generator obeys.
:::
