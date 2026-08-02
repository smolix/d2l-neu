# Adversarial Image Generation
:label:`sec_dcgan`

The chapter has assembled a loss; it has not yet produced an image. Every experiment so far lived in two dimensions, where a scatter plot settles whether the generator matches the data. Images change the problem twice over. A generator must now map a hundred-dimensional latent vector to a $64 \times 64 \times 3$ array, which is an architecture question the toy problems never posed. And once it does, no overlay of samples can settle how well it worked: sample quality must be measured, and the measurement cannot use a likelihood, because the generator still has none. This section answers both questions on one dataset. It first rebuilds the 2015 convolutional recipe that made image GANs work at all. It then strips that recipe down to a modern minimal backbone on which the classic loss of :numref:`sec_basic_gan` and the penalized relativistic loss of :numref:`sec_gan_convergence` can be compared under identical conditions. Finally it measures the outcome with two widely used metrics whose closed forms this chapter has already met.

```{.python .input #dcgan-adversarial-image-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import pickle
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
import pickle
import warnings
from PIL import Image
```

## The 2015 Recipe

The first architecture to generate credible images from the classic GAN objective did so without touching the objective. The deep convolutional GAN (DCGAN) of :citet:`Radford.Metz.Chintala.2015` is a set of architectural commitments. An all-convolutional generator upsamples by transposed convolution, an all-convolutional discriminator downsamples by strided convolution, and batch normalization runs in both networks. The generator uses ReLU activations with a tanh on its output, the discriminator uses leaky ReLU, and Adam runs with its momentum parameter lowered to $\beta_1 = 0.5$ (:numref:`sec_adam`). Each choice tames the statistics that flow between the two players: normalization keeps activations in range no matter how the opponent moves, the bounded tanh pins the generator's output to the data's value range, and the reduced momentum shortens the optimizer's memory of a gradient field that the other player keeps changing. The loss underneath is exactly the non-saturating log loss of :numref:`sec_basic_gan`. That split, a fragile objective held upright by a careful architecture, is the historical situation this section starts from, and separating the two contributions is what the rest of the section is for.

### The Pokemon Sprites

The dataset is a collection of 40,597 Pokemon sprite images covering 721 species, in many variants per species, obtained from [pokemondb](https://pokemondb.net/sprites). Sprites suit a from-scratch experiment: the images are small, the objects are centered on clean backgrounds, and the distribution has real diversity in silhouette and palette. Each image is resized to $64 \times 64$ with bilinear resampling and its pixel values are scaled to $[-1, 1]$, the range conventions of this section: real and generated images live on the same scale, and the 2015 generator's tanh output lands in the same interval. We decode the whole dataset into a single tensor up front, since forty thousand small images fit comfortably in memory and every later experiment then draws minibatches by indexing.

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

A tenth of the images, chosen by a fixed permutation, is held out and never used for training. The held-out sprites serve two purposes later: they let us test whether a trained critic scores training images differently from images it has never seen, which is the memorization question, and they provide the real-data pool against which generated samples are scored. The split is by image, not by species, so variants of one creature can land on both sides; both later uses inherit that caveat.

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

### Generator

The generator must turn a latent vector, treated as a $1 \times 1$ image with 100 channels, into a $64 \times 64$ image with 3 channels: a factor of 64 in each spatial dimension. The DCGAN building block grows the resolution with a transposed convolution (:numref:`sec_transposed_conv`), the upsampling layer that :numref:`sec_fcn` used to enlarge feature maps, followed by batch normalization and a ReLU.

```{.python .input #dcgan-generator-1}
%%tab pytorch
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1):
        super().__init__()
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, strides, padding,
            bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input #dcgan-generator-1}
%%tab jax
class G_block(nnx.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4,
                 strides=2, padding='SAME', rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.conv2d_trans = nnx.ConvTranspose(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides), padding=padding, use_bias=False,
            kernel_init=nnx.initializers.normal(0.02), rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, X):
        return nnx.relu(self.batch_norm(self.conv2d_trans(X)))
```

The shape arithmetic of the transposed convolution decides the whole layout. With kernel size $k$, stride $s$, and padding $p$, an $n \times n$ input produces an output of side $s(n - 1) + k - 2p$. The default block uses $k = 4$, $s = 2$, $p = 1$, so the output side is $2n - 2 + 4 - 2 = 2n$: each block exactly doubles the resolution. The same kernel with $s = 1$ and $p = 0$ maps a $1 \times 1$ input to $0 + 4 - 0 = 4$, which is how the first block turns the latent vector into a $4 \times 4$ feature map.

```{.python .input #dcgan-generator-2}
%%tab pytorch
print(G_block(20)(torch.zeros((2, 3, 16, 16))).shape)
print(G_block(20, strides=1, padding=0)(torch.zeros((2, 3, 1, 1))).shape)
```

```{.python .input #dcgan-generator-2}
%%tab jax
print(G_block(20, rngs=nnx.Rngs(0))(jnp.zeros((2, 16, 16, 3))).shape)
print(G_block(20, strides=1, padding='VALID',
              rngs=nnx.Rngs(0))(jnp.zeros((2, 1, 1, 3))).shape)
```

The full generator chains these blocks through $4 \to 8 \to 16 \to 32 \to 64$. The first block maps the latent input to $4 \times 4$ at $64 \cdot 8$ channels, and three more blocks double the resolution to $32 \times 32$ while halving the channels each time. A final transposed convolution performs the last doubling to $64 \times 64$ while projecting to 3 channels, and a tanh squashes the output into $[-1, 1]$.

```{.python .input #dcgan-generator-3}
%%tab pytorch
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G * 8,
            strides=1, padding=0),                   # Output: (64 * 8, 4, 4)
    G_block(in_channels=n_G * 8, out_channels=n_G * 4),  # (64 * 4, 8, 8)
    G_block(in_channels=n_G * 4, out_channels=n_G * 2),  # (64 * 2, 16, 16)
    G_block(in_channels=n_G * 2, out_channels=n_G),      # (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3, kernel_size=4,
                       stride=2, padding=1, bias=False),
    nn.Tanh())                                           # Output: (3, 64, 64)
net_G(torch.zeros((1, 100, 1, 1))).shape
```

```{.python .input #dcgan-generator-3}
%%tab jax
n_G = 64

class DCGANGenerator(nnx.Module):
    def __init__(self, latent_dim=100, n_G=64, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.blocks = nnx.List([
            G_block(n_G * 8, latent_dim, strides=1, padding='VALID',
                    rngs=rngs),                      # Output: (4, 4, 64 * 8)
            G_block(n_G * 4, n_G * 8, rngs=rngs),    # (8, 8, 64 * 4)
            G_block(n_G * 2, n_G * 4, rngs=rngs),    # (16, 16, 64 * 2)
            G_block(n_G, n_G * 2, rngs=rngs)])       # (32, 32, 64)
        self.output = nnx.ConvTranspose(
            n_G, 3, kernel_size=(4, 4), strides=(2, 2), padding='SAME',
            use_bias=False, kernel_init=nnx.initializers.normal(0.02),
            rngs=rngs)

    def __call__(self, X):
        for block in self.blocks:
            X = block(X)
        return nnx.tanh(self.output(X))              # Output: (64, 64, 3)

net_G = DCGANGenerator(n_G=n_G, rngs=nnx.Rngs(0))
net_G(jnp.zeros((1, 1, 1, 100))).shape
```

### Discriminator

The discriminator runs the same pipeline in reverse: ordinary strided convolutions halve the resolution block by block until a final $4 \times 4$ convolution produces a single realness logit. Its activation is the leaky ReLU,

$$\textrm{leaky ReLU}(x) = \begin{cases}x & \textrm{if}\ x > 0,\\ \alpha x & \textrm{otherwise},\end{cases}$$

with slope $\alpha \in (0, 1)$ on the negative side. An ordinary ReLU that outputs zero also passes zero gradient, and a discriminator unit stuck in that state stops informing the generator entirely; the leak keeps a gradient flowing through negative activations, which matters more than usual here because the generator learns only through the discriminator's gradients.

```{.python .input #dcgan-discriminator-1}
%%tab pytorch
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, alpha=0.2):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input #dcgan-discriminator-1}
%%tab jax
class D_block(nnx.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4,
                 strides=2, padding='SAME', alpha=0.2, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.alpha = alpha
        self.conv2d = nnx.Conv(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides), padding=padding, use_bias=False,
            kernel_init=nnx.initializers.normal(0.02), rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, X):
        return nnx.leaky_relu(self.batch_norm(self.conv2d(X)),
                              negative_slope=self.alpha)
```

With the same $k = 4$, $s = 2$, $p = 1$ configuration, the ordinary convolution's output side is $\lfloor (n - k + 2p + s)/s \rfloor = n/2$ for even $n$. Four blocks therefore reduce $64 \times 64$ to $4 \times 4$ while the channel count doubles per block, and a final unpadded $4 \times 4$ convolution maps the result to a single scalar.

```{.python .input #dcgan-discriminator-2}
%%tab pytorch
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),                                    # Output: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D * 2),  # (64 * 2, 16, 16)
    D_block(in_channels=n_D * 2, out_channels=n_D * 4),  # (64 * 4, 8, 8)
    D_block(in_channels=n_D * 4, out_channels=n_D * 8),  # (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D * 8, out_channels=1,
              kernel_size=4, bias=False))            # Output: (1, 1, 1)
net_D(torch.zeros((1, 3, 64, 64))).shape
```

```{.python .input #dcgan-discriminator-2}
%%tab jax
n_D = 64

class DCGANDiscriminator(nnx.Module):
    def __init__(self, n_D=64, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.blocks = nnx.List([
            D_block(n_D, 3, rngs=rngs),              # Output: (32, 32, 64)
            D_block(n_D * 2, n_D, rngs=rngs),        # (16, 16, 64 * 2)
            D_block(n_D * 4, n_D * 2, rngs=rngs),    # (8, 8, 64 * 4)
            D_block(n_D * 8, n_D * 4, rngs=rngs)])   # (4, 4, 64 * 8)
        self.output = nnx.Conv(
            n_D * 8, 1, kernel_size=(4, 4), padding='VALID', use_bias=False,
            kernel_init=nnx.initializers.normal(0.02), rngs=rngs)

    def __call__(self, X):
        for block in self.blocks:
            X = block(X)
        return self.output(X)                        # Output: (1, 1, 1)

net_D = DCGANDiscriminator(n_D=n_D, rngs=nnx.Rngs(1))
net_D(jnp.zeros((1, 64, 64, 3))).shape
```

### Training with the Classic Loss

The update rules are the ones :numref:`sec_basic_gan` saved to the library: `d2l.update_D` ascends the log-loss objective :eqref:`eq_gan_V` and `d2l.update_G` descends the non-saturating generator loss of :eqref:`eq_gan_weights`. Nothing about them is specific to two-dimensional toys, which is why they were written once. The loop below alternates the two half-steps over shuffled minibatches, following the DCGAN prescription of a shared learning rate and $\beta_1 = 0.5$.

```{.python .input #dcgan-training-with-the-classic-loss-1}
%%tab pytorch
def train_dcgan(net_D, net_G, num_epochs=20, batch_size=256, lr=0.0002,
                latent_dim=100):
    loss = nn.BCEWithLogitsLoss(reduction='mean')
    for net in (net_D, net_G):
        for w in net.parameters():
            nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr,
                                 betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr,
                                 betas=(0.5, 0.999))
    history = []
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        order = torch.randperm(len(train_imgs), device=device)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            X = train_imgs[order[i:i + batch_size]]
            Z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            metric.add(
                d2l.update_D(X, Z, net_D, net_G, loss, trainer_D).detach(),
                d2l.update_G(Z, net_D, net_G, loss, trainer_G).detach(), 1)
        history.append((metric[0] / metric[2], metric[1] / metric[2]))
    return history
```

```{.python .input #dcgan-training-with-the-classic-loss-1}
%%tab jax
def train_dcgan(net_D, net_G, num_epochs=20, batch_size=256, lr=0.0002,
                latent_dim=100):
    optimizer_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.5, b2=0.999),
                                wrt=nnx.Param)
    optimizer_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.5, b2=0.999),
                                wrt=nnx.Param)
    key = jax.random.PRNGKey(0)
    history = []
    for epoch in range(num_epochs):
        key, kp = jax.random.split(key)
        order = jax.random.permutation(kp, len(train_imgs))
        loss_D_sum, loss_G_sum, steps = 0.0, 0.0, 0
        for i in range(0, len(order) - batch_size + 1, batch_size):
            X = train_imgs[order[i:i + batch_size]]
            key, kz = jax.random.split(key)
            Z = jax.random.normal(kz, (batch_size, 1, 1, latent_dim))
            loss_D_sum += d2l.update_D(X, Z, net_D, net_G, optimizer_D)
            loss_G_sum += d2l.update_G(Z, net_D, net_G, optimizer_G)
            steps += 1
        history.append((float(loss_D_sum) / (steps * batch_size),
                        float(loss_G_sum) / (steps * batch_size)))
    return history
```

Twenty epochs take a few minutes on one GPU.

```{.python .input #dcgan-training-with-the-classic-loss-2}
%%tab pytorch
history = train_dcgan(net_D, net_G)
print(f'final loss_D {history[-1][0]:.3f}, loss_G {history[-1][1]:.3f}')
```

```{.python .input #dcgan-training-with-the-classic-loss-2}
%%tab jax
history = train_dcgan(net_D, net_G)
print(f'final loss_D {history[-1][0]:.3f}, loss_G {history[-1][1]:.3f}')
```

```{.python .input #dcgan-training-with-the-classic-loss-3}
%%tab pytorch
with torch.no_grad():
    fake = net_G(torch.randn(20, 100, 1, 1, device=device))
d2l.show_images(fake.cpu().permute(0, 2, 3, 1) / 2 + 0.5,
                num_rows=4, num_cols=5);
```

```{.python .input #dcgan-training-with-the-classic-loss-3}
%%tab jax
fake = net_G(jax.random.normal(jax.random.PRNGKey(2), (20, 1, 1, 100)))
d2l.show_images(np.asarray(fake) / 2 + 0.5, num_rows=4, num_cols=5);
```

The samples are recognizably sprite-like: centered blobs with plausible palettes, rough silhouettes, and clean backgrounds. By the standards of 2015 this was a landmark, and it is worth being precise about what achieved it. The architecture made the classic objective trainable; it did not repair the objective. The mode-dropping landscape of :numref:`sec_gan_relativistic` and the divergent training dynamics of :numref:`sec_gan_convergence` are properties of the loss, and no amount of normalization removes them; DCGAN training is notoriously sensitive to its hyperparameters and prone to collapse on longer runs. The 2015 recipe, in other words, props up the game rather than fixing it. To see what the modern objective changes, the props have to come off.

## A Modern Minimal Backbone

The R3GAN recipe of :numref:`sec_gan_convergence` inverts the 2015 division of labor: fix the objective with the pairing loss and both zero-centered penalties, and then *remove* the architectural crutches. We now build the smallest backbone that follows its principles at our sprite scale. Resampling is by bilinear interpolation followed by an ordinary $3 \times 3$ convolution, in both directions, in place of strided and transposed convolutions, whose uneven kernel overlap is the classic source of checkerboard artifacts :cite:`Odena.Dumoulin.Olah.2016`. Both networks use leaky ReLU throughout. There is no tanh on the generator's output. Above all there is no normalization anywhere, and this choice is what makes the upcoming comparison honest. With batch normalization present, each critic score would depend on the whole minibatch and the per-sample input gradients of :eqref:`eq_gan_r1r2` would entangle across samples; the stabilizer whose value we want to isolate would be confounded with a second one. Removing normalization also removes bookkeeping: no running statistics, no train/eval distinction, and the update rules of :numref:`sec_basic_gan` apply to these networks with no special cases.

One design decision deserves explicit statement. The modern generator starts from a learned constant, a $4 \times 4 \times 128$ tensor of trained parameters, rather than from the latent vector itself, so the latent code must enter somewhere. R3GAN's Config E injects it through a *basis layer*: $4 \times 4$ learnable feature maps modulated by $z$ through a linear layer, in a network the paper nonetheless describes as normalization-free. We simplify that mechanism deliberately. A linear layer projects $z$ to a $4 \times 4 \times 100$ map, the map is concatenated with the learned constant along the channel axis, and a $3 \times 3$ mixing convolution fuses the two before the upsampling stack begins. Concatenation is cheaper than modulation and leaves the constant itself untouched. The constant learns a generic sprite scaffold shared by every sample; the concatenated projection of $z$ is the only source of per-sample variation.

```{.python .input #dcgan-a-modern-minimal-backbone-1}
%%tab pytorch
class Generator(nn.Module):
    """Learned 4x4 constant + projected latent, bilinearly upsampled."""
    def __init__(self, latent_dim=100, const_ch=128, base_ch=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.const = nn.Parameter(0.02 * torch.randn(1, const_ch, 4, 4))
        self.z_proj = nn.Linear(latent_dim, 16 * latent_dim)
        self.mix = nn.Conv2d(const_ch + latent_dim, base_ch, 3, padding=1)
        chans = [base_ch // 2 ** i for i in range(5)]    # 512, 256, ..., 32
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

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
class Generator(nnx.Module):
    """Learned 4x4 constant + projected latent, bilinearly upsampled."""
    def __init__(self, latent_dim=100, const_ch=128, base_ch=512,
                 rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.latent_dim = latent_dim
        init = nnx.initializers.normal(0.02)
        self.const = nnx.Param(init(rngs.params(), (1, 4, 4, const_ch)))
        self.z_proj = nnx.Linear(latent_dim, 16 * latent_dim,
                                 kernel_init=init, rngs=rngs)
        self.mix = nnx.Conv(const_ch + latent_dim, base_ch, (3, 3),
                            padding='SAME', kernel_init=init, rngs=rngs)
        chans = [base_ch // 2 ** i for i in range(5)]    # 512, 256, ..., 32
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nnx.Conv(chans[-1], 3, (3, 3), padding='SAME',
                               kernel_init=init, rngs=rngs)

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

The critic mirrors the generator: four stages of convolution, leaky ReLU, and bilinear *down*sampling take the image from $64 \times 64$ to $4 \times 4$ while the channel count grows, a mixing convolution widens the final map, and a linear head reads out one scalar. The mirror-image structure is deliberate; neither player gets a resolution or capacity advantage the other lacks.

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

    def forward(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = F.interpolate(self.act(conv(x)), scale_factor=0.5,
                              mode='bilinear')
        x = self.act(self.mix(x))
        return self.head(x.reshape(x.shape[0], -1))

for net in (Generator(), Discriminator()):
    print(f'{type(net).__name__}: '
          f'{sum(p.numel() for p in net.parameters())} parameters')
```

```{.python .input #dcgan-a-modern-minimal-backbone-2}
%%tab jax
class Discriminator(nnx.Module):
    """Mirror of the generator: conv + leaky ReLU + bilinear downsampling."""
    def __init__(self, base_ch=32, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        init = nnx.initializers.normal(0.02)
        chans = [3] + [base_ch * 2 ** i for i in range(4)]  # 3, 32, ..., 256
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nnx.Conv(chans[-1], 2 * chans[-1], (3, 3),
                            padding='SAME', kernel_init=init, rngs=rngs)
        self.head = nnx.Linear(2 * chans[-1] * 4 * 4, 1,
                               kernel_init=init, rngs=rngs)

    def __call__(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = nnx.leaky_relu(conv(x), 0.2)
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, h // 2, w // 2, c),
                                 method='bilinear')
        x = nnx.leaky_relu(self.mix(x), 0.2)
        return self.head(x.reshape(x.shape[0], -1))

for net in (Generator(rngs=nnx.Rngs(0)), Discriminator(rngs=nnx.Rngs(0))):
    n = sum(v.size for v in jax.tree.leaves(nnx.state(net, nnx.Param)))
    print(f'{type(net).__name__}: {n} parameters')
```

At about 2.8 million generator and 1.6 million critic parameters, this is a small model, and explicitly a reduced instance of the R3GAN design rather than the published Config E: the paper's grouped convolutions, inverted bottlenecks, and residual depth are capacity refinements that matter at its scale and are skipped at ours.

Two training-side ingredients complete the recipe. Real images are augmented with a random horizontal flip: on a dataset this small, a capable critic can begin to memorize individual training images, and augmentation is the standard countermeasure, developed into a feedback-controlled system at scale by :citet:`Karras.Aittala.Hellsten.ea.2020`. And the weights we evaluate are not the raw iterates but an exponential moving average of the generator's parameters, the same weight EMA that :numref:`sec_training_recipes` introduced in :eqref:`eq_ema` for classifiers. It earns its keep doubly here, because the two-player dynamics of :numref:`sec_gan_convergence` rotate as they contract, and averaging along the spiral lands nearer its center; we use a half-life of 500 steps.

```{.python .input #dcgan-a-modern-minimal-backbone-3}
%%tab pytorch
class EMA:
    """Exponential moving average of model weights, given as a half-life."""
    def __init__(self, model, half_life):
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].lerp_(v, 1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

def sample_real(images, n):
    """Draw a training batch with random horizontal flips."""
    idx = torch.randint(0, len(images), (n,), device=images.device)
    batch = images[idx]
    flip = torch.rand(n, device=images.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), batch.flip(-1), batch)
```

```{.python .input #dcgan-a-modern-minimal-backbone-3}
%%tab jax
class EMA:
    """Exponential moving average of model weights, given as a half-life."""
    def __init__(self, model, half_life):
        self.shadow = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        self.shadow = jax.tree.map(
            lambda s, p: self.decay * s + (1 - self.decay) * p,
            self.shadow, nnx.state(model, nnx.Param))

    def copy_to(self, model):
        nnx.update(model, self.shadow)

def sample_real(key, images, n):
    """Draw a training batch with random horizontal flips."""
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, len(images))
    batch = images[idx]
    flip = jax.random.bernoulli(k2, 0.5, (n,))
    return jnp.where(flip[:, None, None, None], batch[:, :, ::-1, :], batch)
```

## Loss A/B on One Backbone

Everything is now in place for the experiment this section exists to run. Two training arms share this identical backbone and the same initialization, every weight drawn from $\mathcal{N}(0, 0.02^2)$. That distribution is the 2015 recipe's convention, which the earlier tabs in fact applied at different reach: the PyTorch cells redrew every DCGAN parameter from it, batch-normalization scales included, while the JAX cells drew only the convolution kernels and left those scales at one. The backbone here has no normalization parameters, so the convention now means the same thing in both frameworks. The arms also share the optimizer, Adam with $\beta_1 = 0$ and $\beta_2 = 0.99$ at learning rate $2 \cdot 10^{-4}$, the batch size of 64, the augmentation and EMA, and the budget of 15,000 steps, about 26 epochs. They differ in the loss and in nothing else. The first arm trains on the classic non-saturating objective, calling `d2l.update_D` and `d2l.update_G` exactly as the 2015 recipe did. The second trains on the loss of :numref:`sec_gan_convergence`: the relativistic pairing objective with its non-saturating generator, `d2l.rpgan_loss_D` and `d2l.rpgan_loss_G`, plus both zero-centered penalties from `d2l.r1_r2_penalty`.

The penalty weight is $\gamma = 10$. The number was picked by sweeping powers of ten on this dataset: weights from 1 to 100 all train stably here, so the choice within that plateau is not delicate, while $\gamma = 0.1$ under-damps the game and training collapses. The plateau itself does not transfer. Across R3GAN's benchmarks the tuned $\gamma$ ranges from 0.05 on CIFAR-10 to 150 on FFHQ-256 :cite:`Huang.Gokaslan.Kuleshov.ea.2024`, so any single value, ours included, travels to a new dataset only as an order-of-magnitude starting point.

```{.python .input #dcgan-loss-a-b-on-one-backbone-1}
%%tab pytorch
def train_backbone(loss_type, gamma=10, num_steps=15000, batch_size=64,
                   lr=0.0002, latent_dim=100, half_life=500, log_every=250):
    torch.manual_seed(0)
    net_G, net_D = Generator().to(device), Discriminator().to(device)
    for net in (net_G, net_D):
        for w in net.parameters():
            nn.init.normal_(w, 0, 0.02)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr,
                                 betas=(0.0, 0.99))
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr,
                                 betas=(0.0, 0.99))
    loss = nn.BCEWithLogitsLoss(reduction='mean')
    ema, history = EMA(net_G, half_life), []
    for step in range(1, num_steps + 1):
        X = sample_real(train_imgs, batch_size)
        Z = torch.randn(batch_size, latent_dim, device=device)
        Z2 = torch.randn(batch_size, latent_dim, device=device)
        if loss_type == 'classic':
            loss_D = d2l.update_D(X, Z, net_D, net_G, loss, trainer_D)
            loss_G = d2l.update_G(Z2, net_D, net_G, loss, trainer_G)
        else:
            fake = net_G(Z).detach()
            r1, r2 = d2l.r1_r2_penalty(net_D, X, fake)
            loss_D = (d2l.rpgan_loss_D(net_D, X, fake)
                      + gamma / 2 * (r1 + r2).mean())
            trainer_D.zero_grad(), loss_D.backward(), trainer_D.step()
            loss_G = d2l.rpgan_loss_G(net_D, sample_real(train_imgs,
                                                         batch_size),
                                      net_G(Z2))
            trainer_G.zero_grad(), loss_G.backward(), trainer_G.step()
        ema.update(net_G)
        if step % log_every == 0:
            with torch.no_grad():
                d_real = net_D(sample_real(train_imgs, 256)).mean()
            history.append((step, float(loss_D.detach()),
                            float(loss_G.detach()), float(d_real)))
    ema_G = Generator().to(device)
    ema.copy_to(ema_G)
    return ema_G, net_D, torch.tensor(history)
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-1}
%%tab jax
@nnx.jit
def rpgan_step(net_G, net_D, opt_G, opt_D, X, Z, X2, Z2, gamma):
    def loss_D_fn(net_D):
        fake = jax.lax.stop_gradient(net_G(Z))
        r1, r2 = d2l.r1_r2_penalty(net_D, X, fake)
        return (d2l.rpgan_loss_D(net_D, X, fake)
                + gamma / 2 * (r1 + r2).mean())
    loss_D, grads = nnx.value_and_grad(loss_D_fn)(net_D)
    opt_D.update(net_D, grads)
    def loss_G_fn(net_G):
        return d2l.rpgan_loss_G(net_D, X2, net_G(Z2))
    loss_G, grads = nnx.value_and_grad(loss_G_fn)(net_G)
    opt_G.update(net_G, grads)
    return loss_D, loss_G

def train_backbone(loss_type, gamma=10.0, num_steps=15000, batch_size=64,
                   lr=0.0002, latent_dim=100, half_life=500, log_every=250):
    rngs = nnx.Rngs(0)
    net_G, net_D = Generator(rngs=rngs), Discriminator(rngs=rngs)
    opt_G = nnx.Optimizer(net_G, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    opt_D = nnx.Optimizer(net_D, optax.adam(lr, b1=0.0, b2=0.99),
                          wrt=nnx.Param)
    ema, history = EMA(net_G, half_life), []
    key = jax.random.PRNGKey(1)
    for step in range(1, num_steps + 1):
        key, kx, kz, kx2, kz2, kd = jax.random.split(key, 6)
        X = sample_real(kx, train_imgs, batch_size)
        Z = jax.random.normal(kz, (batch_size, latent_dim))
        Z2 = jax.random.normal(kz2, (batch_size, latent_dim))
        if loss_type == 'classic':
            loss_D = d2l.update_D(X, Z, net_D, net_G, opt_D) / batch_size
            loss_G = d2l.update_G(Z2, net_D, net_G, opt_G) / batch_size
        else:
            X2 = sample_real(kx2, train_imgs, batch_size)
            loss_D, loss_G = rpgan_step(net_G, net_D, opt_G, opt_D,
                                        X, Z, X2, Z2, gamma)
        ema.update(net_G)
        if step % log_every == 0:
            d_real = net_D(sample_real(kd, train_imgs, 256)).mean()
            history.append((step, float(loss_D), float(loss_G),
                            float(d_real)))
    ema_G = Generator(rngs=nnx.Rngs(0))
    ema.copy_to(ema_G)
    return ema_G, net_D, np.array(history)
```

The two runs are the longest computation in this chapter, tens of minutes on a single GPU, with the penalized arm the more expensive of the two because the penalty differentiates the critic's input gradient a second time. After each run we also record one diagnostic the loop does not need but the analysis does: the trained critic's average score on training images versus held-out images. A critic that has memorized its training set scores the images it trained on higher; a gap near zero says the critic, and therefore the generator trained against it, is not simply replaying the dataset.

```{.python .input #dcgan-loss-a-b-on-one-backbone-2}
%%tab pytorch
runs = {}
for name, loss_type in [('classic', 'classic'), ('RpGAN + R1 + R2', 'rp')]:
    ema_G, net_D_run, hist = train_backbone(loss_type)
    with torch.no_grad():
        gap = (net_D_run(train_imgs[:1024]).mean()
               - net_D_run(holdout_imgs[:1024]).mean())
    runs[name] = (ema_G, hist)
    print(f'{name}: final loss_D {hist[-1, 1]:.3f}, '
          f'loss_G {hist[-1, 2]:.3f}, '
          f'critic train-holdout gap {float(gap):+.3f}')
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-2}
%%tab jax
runs = {}
for name, loss_type in [('classic', 'classic'), ('RpGAN + R1 + R2', 'rp')]:
    ema_G, net_D_run, hist = train_backbone(loss_type)
    gap = (net_D_run(train_imgs[:1024]).mean()
           - net_D_run(holdout_imgs[:1024]).mean())
    runs[name] = (ema_G, hist)
    print(f'{name}: final loss_D {hist[-1, 1]:.3f}, '
          f'loss_G {hist[-1, 2]:.3f}, '
          f'critic train-holdout gap {float(gap):+.3f}')
```

The grids below settle the comparison at a glance. Both panels show 64 samples from the EMA generator, drawn from the same latent codes in both arms.

```{.python .input #dcgan-loss-a-b-on-one-backbone-3}
%%tab pytorch
def image_grid(imgs, rows=8, cols=8):
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = imgs.reshape(rows, cols, 3, 64, 64).permute(0, 3, 1, 4, 2)
    return imgs.reshape(rows * 64, cols * 64, 3).cpu().numpy()

torch.manual_seed(42)
z_show = torch.randn(64, 100, device=device)
fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 4.8))
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

z_show = jax.random.normal(jax.random.PRNGKey(42), (64, 100))
fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 4.8))
for ax, (name, (ema_G, hist)) in zip(axes, runs.items()):
    ax.imshow(image_grid(ema_G(z_show)))
    ax.set_title(name)
    ax.axis('off')
fig.tight_layout()
```

The classic arm has collapsed completely: all 64 latent codes map to what is visually the same image, a single textured blob repeated across the grid. This is mode collapse at its terminal extreme, one of the mode-dropping minima that :numref:`sec_gan_relativistic` located in the classical landscape. The outcome is initialization-dependent: under either framework's default initialization the classic arm survives this budget (Exercise 6 runs this control), and it is the $\mathcal{N}(0, 0.02^2)$ convention inherited from the 2015 recipe that makes the mode-dropping minima reachable here. This is what a landscape containing such minima predicts: where descent ends depends on where it starts, the classic objective offers no mechanism that rules the collapsed endpoints out, and the pairing objective removes their status as minima altogether. The penalized relativistic arm, from the identical initialization and budget, produces diverse, creature-shaped sprites: varied silhouettes, varied palettes, no repeats among the 64, and no checkerboard texture. Its printed train-holdout gap sits near zero, so the diversity is not recall of the training set, within the limits of a split whose two sides share species. The classic arm's gap prints clearly positive, but on that critic's inflated score scale, so the two gaps are not directly comparable; the penalized arm's near-zero value is the meaningful one. A second seed reproduces both outcomes.

The grids record only the endpoint of each run. The loss traces below show the dynamics that produced it.

```{.python .input #dcgan-loss-a-b-on-one-backbone-4}
%%tab pytorch
fig, axes = d2l.plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (name, (ema_G, hist)) in zip(axes[:2], runs.items()):
    ax.plot(hist[:, 0], hist[:, 1], label='discriminator')
    ax.plot(hist[:, 0], hist[:, 2], label='generator')
    ax.set_title(name), ax.set_xlabel('step'), ax.legend()
axes[0].set_ylabel('loss')
for name, (ema_G, hist) in runs.items():
    axes[2].plot(hist[:, 0], hist[:, 3], label=name)
axes[2].axhline(0, ls='--', c='gray', lw=1)
axes[2].set_title('mean critic score on real batches')
axes[2].set_xlabel('step'), axes[2].legend()
fig.tight_layout()
```

```{.python .input #dcgan-loss-a-b-on-one-backbone-4}
%%tab jax
fig, axes = d2l.plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (name, (ema_G, hist)) in zip(axes[:2], runs.items()):
    ax.plot(hist[:, 0], hist[:, 1], label='discriminator')
    ax.plot(hist[:, 0], hist[:, 2], label='generator')
    ax.set_title(name), ax.set_xlabel('step'), ax.legend()
axes[0].set_ylabel('loss')
for name, (ema_G, hist) in runs.items():
    axes[2].plot(hist[:, 0], hist[:, 3], label=name)
axes[2].axhline(0, ls='--', c='gray', lw=1)
axes[2].set_title('mean critic score on real batches')
axes[2].set_xlabel('step'), axes[2].legend()
fig.tight_layout()
```

In the classic arm the discriminator wins outright: its loss sits pinned near zero while the generator's swings erratically several times higher, and the third panel shows the mechanism. The classic critic's scores on real images climb without bound, with wild oscillations, while the penalized critic's stay bounded throughout. The penalties of :eqref:`eq_gan_r1r2` tax the input gradient $\nabla_x D$, not the score, so the runaway level is a symptom of an ever-steeper critic rather than the taxed quantity itself; the collapsed generator is what that one-sided gradient field leaves behind. In the penalized arm both losses hover near their equilibrium values, and after an initial transient the critic's real-image score settles into a narrow band. Where the band sits is arbitrary, and it lands somewhere different on every rerun: the pairing objective sees only score *differences*, so the critic's absolute level is the unidentified direction that :numref:`sec_gan_relativistic` proved shift-invariant, and the penalties, which act on gradients, do not pin it either. The stable-but-unanchored level of that third curve is the shift symmetry made visible.

## Measuring Sample Quality

The grids convinced by eye, and eyes do not scale. Comparing checkpoints, sweeping $\gamma$, or claiming that one method beats another requires a number, and the number cannot be a likelihood, since the generator has none. The field's answer is to compare distributions of *features*: pass $n$ real and $n$ generated images through a fixed feature map $\phi$, and measure the discrepancy between the two feature clouds with any of the distribution distances this chapter has developed. The two standard metrics are precisely the chapter's two analytically solvable cases, transplanted into feature space.

The Fréchet inception distance (FID) of :citet:`Heusel.Ramsauer.Unterthiner.ea.2017` fits a Gaussian to each feature cloud, $\mathcal{N}(\mu_p, \Sigma_p)$ to the real features and $\mathcal{N}(\mu_q, \Sigma_q)$ to the generated ones, and reports the squared Wasserstein-2 distance between the two Gaussians. :numref:`sec_gan_objectives` deferred exactly this closed form, the one pair of distributions for which the $W_2$ optimal-transport problem of :eqref:`eq_mdl-w2` has an explicit solution :cite:`Dowson.Landau.1982,Givens.Shortt.1984`:

$$
\mathrm{FID}
= \big\| \mu_p - \mu_q \big\|^2
+ \operatorname{tr}\!\Big( \Sigma_p + \Sigma_q
- 2 \big( \Sigma_p^{1/2}\, \Sigma_q\, \Sigma_p^{1/2} \big)^{1/2} \Big).
$$
:eqlabel:`eq_gan_fid`

The formula is transparent in the commuting case. If $\Sigma_p$ and $\Sigma_q$ commute they share an eigenbasis; writing $\lambda_i$ and $\nu_i$ for their eigenvalues along it, the matrix $\Sigma_p^{1/2} \Sigma_q \Sigma_p^{1/2}$ has eigenvalues $\lambda_i \nu_i$, and the trace term becomes $\sum_i \big(\sqrt{\lambda_i} - \sqrt{\nu_i}\big)^2$. FID is then the squared distance between the means plus the squared distances between the standard deviations along each shared principal axis, and in one dimension it reduces to $(\mu_p - \mu_q)^2 + (\sigma_p - \sigma_q)^2$ (Exercise 1 derives this from the transport problem). The general, non-commuting formula is the theorem of the citations above. Two costs are built in: the Gaussian fit sees only the first two moments of the feature distribution, and the plug-in estimate from $n$ samples is biased at finite $n$, in a way examined below.

The kernel inception distance (KID) of :citet:`Binkowski.Sutherland.Arbel.ea.2018` uses the chapter's other closed form, the maximum mean discrepancy. Its population value is the kernel expression :eqref:`eq_mdl-mmd2`, and KID reports the *unbiased* estimator of that quantity, the U-statistic that omits the diagonal self-similarity terms:

$$
\widehat{\mathrm{MMD}}^2
= \frac{1}{m(m-1)} \sum_{i \neq j} k(\phi_i, \phi_j)
+ \frac{1}{n(n-1)} \sum_{i \neq j} k(\phi'_i, \phi'_j)
- \frac{2}{mn} \sum_{i,j} k(\phi_i, \phi'_j),
$$
:eqlabel:`eq_gan_kid`

with real features $\phi_i$, generated features $\phi'_j$, and the polynomial kernel $k(u, v) = (u^\top v / d + 1)^3$ in $d$ feature dimensions. In :numref:`sec_gan_objectives` the MMD's closed form was available because the kernel was fixed in advance, and the same scoping sentence warned that this convenience loads the whole modeling burden onto the kernel choice. KID is the reopened version of that choice: a polynomial kernel on learned features is a learned kernel on images, and everything the metric notices or misses is decided by the network that computes $\phi$. The newer CMMD metric changes only that network: it is the same U-statistic computed on CLIP features. The $O(n^2)$ pairwise cost of the estimator also returns, modest at the sample sizes used here. Unbiasedness has a visible signature worth anticipating: on two disjoint samples of *real* images the U-statistic scatters around zero and can legitimately come out slightly negative, while the FID between the same two samples stays strictly positive, since two finite clouds never have identical fitted moments.

```{.python .input #dcgan-measuring-sample-quality}
%%tab pytorch, jax
def fid_score(feat_p, feat_q):
    mu_p, mu_q = feat_p.mean(0), feat_q.mean(0)
    C_p = np.cov(feat_p, rowvar=False)
    C_q = np.cov(feat_q, rowvar=False)
    lam, U = np.linalg.eigh(C_p)
    sqrt_C_p = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    cross = np.linalg.eigvalsh(sqrt_C_p @ C_q @ sqrt_C_p)
    return (((mu_p - mu_q) ** 2).sum() + np.trace(C_p) + np.trace(C_q)
            - 2 * np.sqrt(np.maximum(cross, 0)).sum())

def kid_score(feat_p, feat_q):
    m, n, d = len(feat_p), len(feat_q), feat_p.shape[1]
    K_pp = (feat_p @ feat_p.T / d + 1) ** 3
    K_qq = (feat_q @ feat_q.T / d + 1) ** 3
    K_pq = (feat_p @ feat_q.T / d + 1) ** 3
    return ((K_pp.sum() - np.trace(K_pp)) / (m * (m - 1))
            + (K_qq.sum() - np.trace(K_qq)) / (n * (n - 1))
            - 2 * K_pq.mean())
```

### A Feature Network Trained in the Notebook

Both metrics need the feature map $\phi$, and the standard choice, the pooled features of an Inception-v3 classifier trained on ImageNet, would smuggle an opaque dependency into a section built to be self-contained. We instead train the feature network ourselves, in the notebook, in about two minutes: a small convolutional classifier on CIFAR-10, whose 128-dimensional penultimate layer becomes $\phi$. The choice reproduces, deliberately and in miniature, the structural situation of every FID number in the literature: features trained on one distribution, natural photographs of ten object classes, scoring generators of another, sprites. One resolution detail matters. The network is trained at $64 \times 64$, on bilinearly upsampled CIFAR-10 images, rather than trained at $32 \times 32$ with the sprites downsampled to match, because downsampling generated images would blur away exactly the artifacts, checkerboard texture and collapsed fine detail, that the metric exists to detect.

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-1}
%%tab pytorch
d2l.DATA_HUB['cifar10'] = (
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    '874905e36347c8536514d0a26261acf3bff89bc7')
cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')

def load_cifar_batch(path):
    with open(path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')   # legacy NumPy pickle format
        batch = pickle.load(f, encoding='bytes')
    X = torch.tensor(batch[b'data'], dtype=torch.float32)
    return X.reshape(-1, 3, 32, 32) / 127.5 - 1, torch.tensor(
        batch[b'labels'])

Xs, ys = zip(*[load_cifar_batch(f'{cifar_dir}/data_batch_{i}')
               for i in range(1, 6)])
cifar_X, cifar_y = torch.cat(Xs).to(device), torch.cat(ys).to(device)
test_X, test_y = load_cifar_batch(f'{cifar_dir}/test_batch')
test_X, test_y = test_X.to(device), test_y.to(device)
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-1}
%%tab jax
d2l.DATA_HUB['cifar10'] = (
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    '874905e36347c8536514d0a26261acf3bff89bc7')
cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')

def load_cifar_batch(path):
    with open(path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')   # legacy NumPy pickle format
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (jnp.asarray(X, dtype=jnp.float32) / 127.5 - 1,
            jnp.asarray(batch[b'labels']))

Xs, ys = zip(*[load_cifar_batch(f'{cifar_dir}/data_batch_{i}')
               for i in range(1, 6)])
cifar_X, cifar_y = jnp.concatenate(Xs), jnp.concatenate(ys)
test_X, test_y = load_cifar_batch(f'{cifar_dir}/test_batch')
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-2}
%%tab pytorch
class FeatureCNN(nn.Module):
    """Three conv blocks, global average pooling, and a linear head."""
    def __init__(self, feature_dim=128):
        super().__init__()
        widths = [3, 32, 64, feature_dim]
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                           nn.BatchNorm2d(c_out), nn.ReLU(),
                           nn.MaxPool2d(2))
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(feature_dim, 10)

    def features(self, x):
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.features(x))

def upsample(X):
    return F.interpolate(X, size=(64, 64), mode='bilinear')

torch.manual_seed(0)
cnn = FeatureCNN().to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
for epoch in range(3):
    order = torch.randperm(len(cifar_X), device=device)
    for i in range(0, len(order) - 255, 256):
        idx = order[i:i + 256]
        l = F.cross_entropy(cnn(upsample(cifar_X[idx])), cifar_y[idx])
        opt.zero_grad(), l.backward(), opt.step()
cnn.eval()
with torch.no_grad():
    pred = torch.cat([cnn(upsample(test_X[i:i + 1000])).argmax(1)
                      for i in range(0, len(test_X), 1000)])
print(f'CIFAR-10 test accuracy: {float((pred == test_y).float().mean()):.3f}')
```

```{.python .input #dcgan-a-feature-network-trained-in-the-notebook-2}
%%tab jax
class FeatureCNN(nnx.Module):
    """Three conv blocks, global average pooling, and a linear head."""
    def __init__(self, feature_dim=128, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        widths = [3, 32, 64, feature_dim]
        self.convs = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME', rngs=rngs)
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.norms = nnx.List([nnx.BatchNorm(c, rngs=rngs)
                               for c in widths[1:]])
        self.head = nnx.Linear(feature_dim, 10, rngs=rngs)

    def features(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = nnx.max_pool(nnx.relu(norm(conv(x))),
                             window_shape=(2, 2), strides=(2, 2))
        return x.mean(axis=(1, 2))

    def __call__(self, x):
        return self.head(self.features(x))

def upsample(X):
    return jax.image.resize(X, (X.shape[0], 64, 64, 3), method='bilinear')

@nnx.jit
def cnn_step(cnn, opt, X, y):
    def loss_fn(cnn):
        return optax.softmax_cross_entropy_with_integer_labels(
            cnn(upsample(X)), y).mean()
    l, grads = nnx.value_and_grad(loss_fn)(cnn)
    opt.update(cnn, grads)
    return l

cnn = FeatureCNN(rngs=nnx.Rngs(0))
opt = nnx.Optimizer(cnn, optax.adam(0.001), wrt=nnx.Param)
key = jax.random.PRNGKey(3)
for epoch in range(3):
    key, kp = jax.random.split(key)
    order = jax.random.permutation(kp, len(cifar_X))
    for i in range(0, len(order) - 255, 256):
        idx = order[i:i + 256]
        cnn_step(cnn, opt, cifar_X[idx], cifar_y[idx])
cnn.eval()
pred = jnp.concatenate([cnn(upsample(test_X[i:i + 1000])).argmax(1)
                        for i in range(0, len(test_X), 1000)])
print(f'CIFAR-10 test accuracy: {float((pred == test_y).mean()):.3f}')
```

The accuracy is modest, and beside the point: the classifier exists to supply features whose statistics respond to image structure, not to win at CIFAR-10.

### Scoring the Two Runs

Three comparisons calibrate each other. Two disjoint sets of 500 *real* training images give the floor, the score two samples of the truth assign each other, which is pure estimator noise and bias. Then each arm's 500 EMA samples are scored against 500 held-out real images.

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
floor_fid = fid_score(features(cnn, train_imgs[:n]),
                      features(cnn, train_imgs[n:2 * n]))
floor_kid = kid_score(features(cnn, train_imgs[:n]),
                      features(cnn, train_imgs[n:2 * n]))
print(f'{"run":22s}{"FID":>10s}{"KID":>10s}')
print(f'{"real vs. real":22s}{floor_fid:10.2f}{floor_kid:10.2f}')
for name, (ema_G, hist) in runs.items():
    with torch.no_grad():
        feat = features(cnn, ema_G(z_score).clamp(-1, 1))
    print(f'{name:22s}{fid_score(feat, feat_real):10.2f}'
          f'{kid_score(feat, feat_real):10.2f}')
```

```{.python .input #dcgan-scoring-the-two-runs}
%%tab jax
def features(model, imgs, batch_size=250):
    return np.concatenate(
        [np.asarray(model.features(imgs[i:i + batch_size]))
         for i in range(0, len(imgs), batch_size)])

n = 500
z_score = jax.random.normal(jax.random.PRNGKey(7), (n, 100))
feat_real = features(cnn, holdout_imgs[:n])
floor_fid = fid_score(features(cnn, train_imgs[:n]),
                      features(cnn, train_imgs[n:2 * n]))
floor_kid = kid_score(features(cnn, train_imgs[:n]),
                      features(cnn, train_imgs[n:2 * n]))
print(f'{"run":22s}{"FID":>10s}{"KID":>10s}')
print(f'{"real vs. real":22s}{floor_fid:10.2f}{floor_kid:10.2f}')
for name, (ema_G, hist) in runs.items():
    feat = features(cnn, jnp.clip(ema_G(z_score), -1, 1))
    print(f'{name:22s}{fid_score(feat, feat_real):10.2f}'
          f'{kid_score(feat, feat_real):10.2f}')
```

Both metrics rank the penalized relativistic run far above the collapsed classic run, and the margin is not delicate: the gap between the two arms sits two orders of magnitude above the real-versus-real floor, so no plausible amount of estimator noise could reverse the ordering. The exact values printed above are properties of this run and this feature network, and they are the right things to be uncertain about; the ordering is not. Rerunning the comparison with differently seeded feature networks and with larger sample sizes leaves it unchanged. The floor row also demonstrates the bias claim made earlier: the real-versus-real FID is systematically positive, because fitted moments of two finite samples never coincide, while the unbiased KID floor sits within estimator noise of zero.

The caution these numbers deserve is structural, not numerical, and each caveat is a citation with teeth. First, both scores are functions of the feature network before they are functions of the generator: our CIFAR-trained CNN attends to whatever CIFAR-10 classification made salient, exactly as Inception features attend to what ImageNet classification made salient. A different feature network yields different numbers, and can yield a different ordering when the compared models are close (Exercise 3 tests this). Second, the coupling between metric features and training runs deeper than domain mismatch. Because Inception features encode ImageNet class structure, FID can be lowered substantially by matching the ImageNet class histogram of the real data, with no visible quality gain, and models whose discriminators use ImageNet-pretrained features partially game the metric this way :cite:`Kynkaanniemi.Karras.Aittala.ea.2023`. R3GAN reports its results without pretrained discriminators for precisely this reason :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. Third, the pixels handed to the feature network must be prepared correctly: incorrectly antialiased resizing in common image libraries shifts FID by margins comparable to the gaps between published methods :cite:`Parmar.Zhang.Zhu.2022`, which is why our feature network was trained at the sprites' native resolution instead of resizing the sprites. Fourth, the finite-sample bias visible in our floor row depends on the model being scored, not only on $n$, so two models compared at small $n$ can swap places purely through their bias terms :cite:`Chong.Forsyth.2020` (Exercise 2 measures the effect).

A single number also conflates two failures that this chapter has treated as distinct: samples that look wrong, and samples that miss modes of the data. The precision and recall metrics of :citet:`Kynkaanniemi.Karras.Laine.ea.2019` separate them by comparing feature-space neighborhoods, precision asking what fraction of generated samples land inside the support of the real features, recall asking what fraction of real samples are covered by the generated ones. A collapsed generator such as our classic arm can retain nonzero precision, its one blob may sit near the data manifold, while its recall is near zero; that asymmetry is invisible inside a single FID value. Recall is the diversity number R3GAN reports alongside FID, and the fidelity--coverage split recurs throughout the evaluation literature.

## Conditioning, Scale, and Scope

### Conditional Generation

Everything in this chapter generates unconditionally: the only input is noise. Practical systems almost always condition, on a class label, a text embedding, or a corrupted image, and the extension costs the theory nothing. Give both networks the condition $c$: the generator becomes $x = G(z, c)$ and the critic $D(x, c)$, and the game is played per condition, with the value averaged over $c$. Solving it pointwise, exactly as in :numref:`sec_basic_gan`, yields the optimal critic $D^\star(x, c) = \log\big(p(x \mid c)\,/\,q(x \mid c)\big)$, the *conditional* log density ratio, and the value of the game is the expected conditional Jensen--Shannon divergence, $E_c\big[2\,\mathrm{JS}\big(p(\cdot \mid c), q(\cdot \mid c)\big)\big] - 2\log 2$; Exercise 5 walks through the two-line derivation. Conditioning changes the networks' inputs, not the analysis.

Two mechanisms carry $c$ into the networks in practice. The original conditional GAN simply concatenates an embedding of $c$ onto both networks' inputs :cite:`Mirza.Osindero.2014`, in our modern generator's case alongside the projected latent at the $4 \times 4$ stage. The projection discriminator computes the critic score as an inner product between a learned class embedding and the critic's penultimate features, plus an unconditional term :cite:`Miyato.Koyama.2018`, a form derived from exactly the conditional log-ratio structure above. The machinery matters at scale: R3GAN's CIFAR-10 and ImageNet results are class-conditional, and the text-to-image systems that :numref:`sec_gan_beyond` discusses condition on prompt embeddings; in every case the objective is the one this chapter analyzed, with richer inputs.

### Scale

Between our sprite experiment and published results stands mostly compute, and it is worth being concrete about how much, because the honesty is unusual and instructive. For R3GAN, StackedMNIST took 7 hours on eight L40 GPUs, CIFAR-10 four days on the same eight, FFHQ-256 about three weeks on eight A6000s, and conditional ImageNet about a day on 32 H100s :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. Our 15,000 steps on one GPU are the smallest end of a curve that spans four orders of magnitude.

What changes along that curve is tuning, not principle. The penalty weight is resolution- and dataset-dependent, 0.05 to 150 across R3GAN's benchmarks against our 10, and is decayed on a schedule along with the learning rate. Mixed precision must be bfloat16: R3GAN reports that training fails under FP16 and succeeds under BF16, matching wider large-model practice. The EMA half-life grows to millions of images and follows its own schedule, and the augmentation probability becomes a tuned, sometimes feedback-controlled quantity :cite:`Karras.Aittala.Hellsten.ea.2020`. None of these introduce a new idea; they set the constants of this section's recipe at larger scale.

That closes the loop this chapter opened. The pipeline that trains, stabilizes, and evaluates an image GAN is now assembled end to end, and each of its parts, the objective, the penalty, the metrics, traces back to a derivation in :numref:`sec_basic_gan` through :numref:`sec_gan_convergence`. For image generation at large, the models that carry the field today are the diffusion models of :numref:`chap_diffusion`, and the image-specific applications live in :numref:`chap_cv`; where the adversarial loss itself survives inside modern systems is the subject of the next section.

## Summary

Image generation added two problems the chapter's analysis had not faced: an architecture, and a measurement. The 2015 DCGAN recipe solved only the first, and indirectly: transposed-convolution upsampling, batch normalization, and careful activations hold the classic objective upright without repairing it. The modern alternative inverts that bargain. On a minimal backbone with bilinear resampling, leaky ReLU, no normalization, and a concatenation-injected latent, the classic non-saturating loss collapses completely within our budget from the 2015 recipe's $\mathcal{N}(0, 0.02^2)$ initialization: every latent code maps to the same image, and the critic's real-image scores climb without bound. Under either framework's default initialization the same arm survives the budget; the mode-dropping minima are a feature of the landscape that some starting points reach and others do not. The relativistic pairing loss with both zero-centered penalties at $\gamma = 10$ trains stably from the identical initialization to diverse, creature-shaped sprites, with a bounded critic and no detectable gap between training and held-out images. Sample quality was then measured, not eyeballed: FID is the Bures--Wasserstein $W_2^2$ between Gaussians fitted to learned features, the closed form :numref:`sec_gan_objectives` deferred, and KID is the unbiased U-statistic of the MMD, the fixed-kernel game reopened with a learned feature map. Both metrics rank the penalized run far above the collapsed one, with a real-versus-real floor two orders of magnitude below the gap, and both inherit their meaning from the feature network, which is why feature dependence, ImageNet leakage, resizing artifacts, and finite-sample bias qualify every published number. Conditioning extends the analysis without changing it, and scale changes the constants, the budget, and the precision format, but not the recipe.

## Exercises

1. In one dimension, the squared Wasserstein-2 distance between distributions with quantile functions $F^{-1}$ and $G^{-1}$ is $\int_0^1 \big(F^{-1}(u) - G^{-1}(u)\big)^2\, du$. Using the Gaussian quantile function $F^{-1}(u) = \mu + \sigma\, \Phi^{-1}(u)$, show that for two univariate Gaussians this equals $(\mu_p - \mu_q)^2 + (\sigma_p - \sigma_q)^2$. Then verify that the multivariate formula :eqref:`eq_gan_fid` reduces to a sum of such terms when $\Sigma_p$ and $\Sigma_q$ commute.
1. Measure the finite-sample bias of FID directly: split the held-out sprites into two disjoint subsets of size $n$ each and compute the real-versus-real FID and KID for $n \in \{100, 250, 500, 2000\}$. How does each floor move with $n$? Now plot the FID estimates against $1/n$ and extrapolate the fitted line to $1/n \to 0$; this extrapolation is the bias-corrected estimator of :citet:`Chong.Forsyth.2020`. How close is the extrapolated floor to zero, and what does the slope of the line say about the bias at $n = 500$, the sample size this section uses?
1. Retrain the feature network of this section on Fashion-MNIST instead of CIFAR-10 (replicate the grayscale channel three times, and upsample to $64 \times 64$ as before), and rescore both arms. Do the FID and KID *values* change substantially? Does the *ordering* of the two arms change? Formulate in one sentence what this implies about comparing FID numbers computed with different feature networks.
1. Apply the modern recipe to Fashion-MNIST: adapt the backbone to single-channel images (for instance by padding to $32 \times 32$ and removing one upsampling stage), train once with the classic loss and once with the RpGAN plus $R_1 + R_2$ loss at the same budget, and compare sample grids. Which of this section's findings reproduce on a dataset with ten well-separated modes?
1. Derive the optimal critic of the conditional game. With the value $V(D) = E_c\big[E_{x \sim p(\cdot \mid c)}[\log \sigma(D(x, c))] + E_{x' \sim q(\cdot \mid c)}[\log \sigma(-D(x', c))]\big]$, fix a condition $c$ and apply the pointwise argument of :numref:`sec_basic_gan` to show $D^\star(x, c) = \log\big(p(x \mid c)/q(x \mid c)\big)$, and conclude that $\max_D V = E_c\big[2\, \mathrm{JS}\big(p(\cdot \mid c),\, q(\cdot \mid c)\big)\big] - 2 \log 2$. Why does this justify the claim that conditioning changes the networks but not the analysis?
1. Rerun arm A, the classic loss on the modern backbone, with the explicit $\mathcal{N}(0, 0.02^2)$ initialization removed, so that your framework's default initialization applies, keeping the architecture, optimizer, and budget fixed. Compare the resulting grid with the two grids of this section, and reconcile what you find with the landscape discussion of :numref:`sec_gan_relativistic`: the mode-dropping minima do not move when the initialization changes, so what does the outcome say about which starting points descend into them?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.5]{.kicker}

Adversarial image generation<br>
**the 2015 recipe · a modern minimal backbone · the loss A/B · measuring sample quality**
:::
:::

::: {.slide title="The 2015 Recipe: Architecture Props Up the Objective"}
DCGAN (Radford et al., 2015) made the classic loss train on images through
architectural commitments:

- transposed-conv upsampling in $G$, strided conv in $D$;
- batch normalization in both networks;
- ReLU + tanh output in $G$, leaky ReLU in $D$; Adam with $\beta_1 = 0.5$.

. . .

Each choice tames the statistics flowing between the players. The objective
underneath is untouched: the same non-saturating log loss of
:numref:`sec_basic_gan`.
:::

::: {.slide title="The Historical Baseline on Pokemon Sprites"}
@!dcgan-training-with-the-classic-loss-3

Sprite-like blobs: plausible palettes, rough silhouettes. The landmark of
2015 — but the mode-dropping landscape and divergent dynamics are properties
of the *loss*, and normalization does not remove them.
:::

::: {.slide title="A Modern Minimal Backbone"}
R3GAN's principles at sprite scale — fix the objective, remove the props:

- bilinear resampling + $3 \times 3$ conv (no strided/transposed conv);
- leaky ReLU everywhere; no tanh; **no normalization anywhere**;
- Adam $\beta_1 = 0$, lr $2 \cdot 10^{-4}$; flip augmentation; weight EMA.

. . .

Latent injection: project $z$ linearly to $4 \times 4$, **concatenate**
with the learned constant, fuse with a mix conv — a deliberate
simplification of R3GAN's basis layer ($z$-modulated learned $4 \times 4$
feature maps).
:::

::: {.slide title="One Backbone, Two Losses"}
Identical backbone, initialization, optimizer, augmentation, EMA, and
budget (15,000 steps). Only the loss differs:

- **classic**: `d2l.update_D` / `d2l.update_G` — the non-saturating log
  loss of :numref:`sec_basic_gan`;
- **RpGAN + $R_1{+}R_2$**: `d2l.rpgan_loss_D/G` + `d2l.r1_r2_penalty`,
  $\gamma = 10$ — the loss of :numref:`sec_gan_convergence`.

$\gamma$ picked by sweeping powers of ten: 1–100 all stable here, 0.1
collapses. R3GAN tunes $\gamma$ from 0.05 to 150 per dataset — no single
value is portable.
:::

::: {.slide title="The A/B Grids"}
@!dcgan-loss-a-b-on-one-backbone-3

- Classic arm: **complete collapse** — all 64 latent codes map to the same
  image. Initialization-dependent: framework-default inits survive this
  budget; the 2015 recipe's $\mathcal{N}(0, 0.02^2)$ reaches the
  mode-dropping minima.
- Penalized relativistic arm: diverse, creature-shaped sprites, no repeats;
  critic train–holdout gap $\approx 0$ — no memorization signal.
:::

::: {.slide title="The Traces Show the Mechanism"}
@!dcgan-loss-a-b-on-one-backbone-4

- Classic: $D$'s loss pinned near 0; its real-image scores climb without
  bound, with wild oscillations — the ever-steeper critic whose input
  gradient $\nabla_x D$ the zero-centered penalties exist to tax.
- Penalized: losses hover at equilibrium; the critic's level settles into a
  narrow band whose location is arbitrary — the pairing objective's
  **shift invariance** made visible.
:::

::: {.slide title="FID Is the Gaussian W2 Closed Form"}
Fit Gaussians to real and generated features, report the $W_2^2$ closed
form :numref:`sec_gan_objectives` deferred:

$$\mathrm{FID} = \|\mu_p - \mu_q\|^2 + \operatorname{tr}\big(\Sigma_p +
\Sigma_q - 2(\Sigma_p^{1/2} \Sigma_q \Sigma_p^{1/2})^{1/2}\big)$$

- Commuting case: mean shift plus per-axis standard-deviation shifts.
- Sees only two moments; biased at finite $n$ — the floor row shows it.
:::

::: {.slide title="KID Is the MMD U-Statistic with Learned Features"}
@!dcgan-scoring-the-two-runs

Unbiased MMD$^2$ estimator, polynomial kernel on learned features — the
kernel choice :numref:`sec_gan_objectives` fixed, reopened. Real-vs-real
floor $\approx 0$ — unbiased, so in principle it can even print negative;
both metrics rank the penalized run far above the collapsed one, with the
floor two orders of magnitude below the gap.
:::

::: {.slide title="What the Numbers Do Not Settle"}
- **Feature dependence**: scores are functions of $\phi$ first — our
  CIFAR-trained CNN scoring sprites *is* Inception-on-ImageNet in
  miniature.
- **Leakage**: FID drops by matching ImageNet class histograms
  (Kynkäänniemi et al., 2023); R3GAN avoids pretrained discriminators for
  this reason.
- **Resizing**: aliased image resizing shifts FID by method-sized margins
  (Parmar et al., 2022).
- **Bias**: model-dependent at finite $n$ (Chong & Forsyth, 2020).
- Precision/recall separate fidelity from coverage: a collapsed generator
  can keep precision while recall $\to 0$.
:::

::: {.slide title="Conditioning and Scale"}
Conditioning: give both networks $c$; the optimal critic becomes the
conditional log ratio — networks change, analysis does not
(concatenation, or the projection discriminator).

R3GAN's real budgets (quoted):

| benchmark | compute |
|:--|:--|
| StackedMNIST | 7 h on 8 L40 |
| CIFAR-10 | 4 days on 8 L40 |
| FFHQ-256 | ~3 weeks on 8 A6000 |
| ImageNet (cond.) | ~1 day on 32 H100 |

At scale: $\gamma$ per dataset (0.05–150), BF16 not FP16, EMA half-life in
Mimg, tuned augmentation.
:::

::: {.slide title="Recap"}
- 2015: architecture props up a fragile objective. 2024: fix the objective,
  drop the props.
- Same backbone, same budget: classic loss **collapses completely** from
  the 2015-recipe init (framework defaults survive the budget);
  RpGAN + $R_1{+}R_2$ at $\gamma = 10$ trains stably to diverse sprites.
- FID = Bures–Wasserstein $W_2^2$; KID = unbiased MMD — the chapter's two
  closed forms, in learned feature space.
- Both rank B far above A; the ordering is robust, the values are not the
  point.
- Every metric inherits its meaning from its feature network.
- Next: where the adversarial loss lives inside modern systems
  (:numref:`sec_gan_beyond`).
:::
