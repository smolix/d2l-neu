# Diffusion for Images
:label:`sec_diffusion-images`

(to be written)

```{.python .input #image-diffusion-diffusion-models-for-images}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
import torch
from torch import nn
```

```{.python .input #image-diffusion-diffusion-models-for-images}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## The Denoising U-Net

(to be written)

### Skips, Resolutions and the Time Embedding

(to be written)

Each block of the U-Net applies two $3 \times 3$ convolutions with group
normalization, adds a projection of the conditioning embedding between
them, and keeps a residual connection around both.

:begin_tab:`jax`
`d2l.Conv` is `nnx.Conv` with weights and biases drawn from $U(-a, a)$,
$a = 1/\sqrt{\textrm{fan-in}}$, as `d2l.Linear` does for dense layers.
With Flax's default initialization this U-Net learns markedly more slowly
in its first few thousand updates.
:end_tab:

```{.python .input #image-diffusion-skips-resolutions-and-the-time-embedding-1}
%%tab pytorch
class UNetBlock(nn.Module):  #@save
    """Two 3x3 convolutions with group norm, a residual, and an embedding."""
    def __init__(self, c_in, c_out, emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out) if emb_dim else None
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = (nn.Conv2d(c_in, c_out, 1) if c_in != c_out
                     else nn.Identity())

    def forward(self, x, emb=None):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        if self.emb is not None:
            h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)
```

```{.python .input #image-diffusion-skips-resolutions-and-the-time-embedding-1}
%%tab jax
class Conv(nnx.Conv):  #@save
    """nnx.Conv (kernel_size a tuple), all weights from U(+-1/sqrt(fan_in))."""
    def __init__(self, in_features, out_features, kernel_size, *, rngs,
                 **kwargs):
        a = (in_features * math.prod(kernel_size)) ** -0.5
        init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, -a, a)
        super().__init__(in_features, out_features, kernel_size,
                         kernel_init=init, bias_init=init, rngs=rngs,
                         **kwargs)

class UNetBlock(nnx.Module):  #@save
    """Two 3x3 convolutions with group norm, a residual, and an embedding."""
    def __init__(self, c_in, c_out, emb_dim=None, *, rngs):
        norm = lambda c: nnx.GroupNorm(c, num_groups=8, epsilon=1e-5,
                                       use_fast_variance=False, rngs=rngs)
        self.norm1, self.norm2 = norm(c_in), norm(c_out)
        self.conv1 = d2l.Conv(c_in, c_out, (3, 3), rngs=rngs)
        self.emb = d2l.Linear(emb_dim, c_out, rngs=rngs) if emb_dim else None
        self.conv2 = d2l.Conv(c_out, c_out, (3, 3), rngs=rngs)
        self.skip = (d2l.Conv(c_in, c_out, (1, 1), rngs=rngs)
                     if c_in != c_out else None)

    def __call__(self, x, emb=None):
        h = self.conv1(nnx.silu(self.norm1(x)))
        if self.emb is not None:
            h = h + self.emb(emb)[:, None, None, :]
        h = self.conv2(nnx.silu(self.norm2(h)))
        return h + (x if self.skip is None else self.skip(x))
```

The U-Net embeds the time $t \in [0, 1]$ with `d2l.time_embedding`, adds a
label embedding when it is given classes, and passes the result through a
small MLP to every block; two stride-2 convolutions go down to $7 \times
7$ and nearest-neighbor up-sampling to the size of each skip comes back.

```{.python .input #image-diffusion-skips-resolutions-and-the-time-embedding-2}
%%tab pytorch
class UNet(nn.Module):  #@save
    """A small U-Net on time t, a label y (num_classes = null), and cond."""
    def __init__(self, ch=64, num_classes=None, in_channels=1,
                 out_channels=1, emb_dim=128):
        super().__init__()
        self.emb_dim, self.num_classes = emb_dim, num_classes
        self.label_emb = (nn.Embedding(num_classes + 1, emb_dim)
                          if num_classes else None)
        self.emb_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                     nn.Linear(emb_dim, emb_dim))
        self.inc = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down1 = UNetBlock(ch, ch, emb_dim)                   # 28 x 28
        self.pool1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)    # 14 x 14
        self.down2 = UNetBlock(ch, 2 * ch, emb_dim)
        self.pool2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)  # 7 x 7
        self.mid = UNetBlock(2 * ch, 2 * ch, emb_dim)
        self.up1 = UNetBlock(4 * ch, ch, emb_dim)                 # 14 x 14
        self.up0 = UNetBlock(2 * ch, ch, emb_dim)                 # 28 x 28
        self.outc = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                  nn.Conv2d(ch, out_channels, 3, padding=1))

    def forward(self, x, t, y=None, cond=None):
        emb = d2l.time_embedding(t, self.emb_dim)
        if y is not None:
            emb = emb + self.label_emb(y)
        if cond is not None:
            emb = emb + cond
        emb = self.emb_mlp(emb)
        up = lambda h, skip: torch.cat([nn.functional.interpolate(
            h, size=skip.shape[-2:], mode='nearest-exact'), skip], 1)
        h1 = self.down1(self.inc(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        h = self.mid(self.pool2(h2), emb)
        h = self.up1(up(h, h2), emb)  # skip connection
        h = self.up0(up(h, h1), emb)  # skip connection
        return self.outc(h)
```

```{.python .input #image-diffusion-skips-resolutions-and-the-time-embedding-2}
%%tab jax
class UNet(nnx.Module):  #@save
    """A small U-Net (NHWC) on time t, a label y (num_classes = null), cond."""
    def __init__(self, ch=64, num_classes=None, in_channels=1,
                 out_channels=1, emb_dim=128, *, rngs):
        self.emb_dim, self.num_classes = emb_dim, num_classes
        init = nnx.initializers.normal(1.0)  # N(0, 1), as in PyTorch
        self.label_emb = (nnx.Embed(num_classes + 1, emb_dim, rngs=rngs,
                                    embedding_init=init)
                          if num_classes else None)
        self.emb1 = d2l.Linear(emb_dim, emb_dim, rngs=rngs)
        self.emb2 = d2l.Linear(emb_dim, emb_dim, rngs=rngs)
        conv = lambda a, b, **kw: d2l.Conv(a, b, (3, 3), rngs=rngs, **kw)
        self.inc = conv(in_channels, ch)
        self.down1 = UNetBlock(ch, ch, emb_dim, rngs=rngs)        # 28 x 28
        self.pool1 = conv(ch, ch, strides=2, padding=1)           # 14 x 14
        self.down2 = UNetBlock(ch, 2 * ch, emb_dim, rngs=rngs)
        self.pool2 = conv(2 * ch, 2 * ch, strides=2, padding=1)   # 7 x 7
        self.mid = UNetBlock(2 * ch, 2 * ch, emb_dim, rngs=rngs)
        self.up1 = UNetBlock(4 * ch, ch, emb_dim, rngs=rngs)      # 14 x 14
        self.up0 = UNetBlock(2 * ch, ch, emb_dim, rngs=rngs)      # 28 x 28
        self.norm = nnx.GroupNorm(ch, num_groups=8, epsilon=1e-5,
                                  use_fast_variance=False, rngs=rngs)
        self.outc = conv(ch, out_channels)

    def __call__(self, x, t, y=None, cond=None):
        emb = d2l.time_embedding(t, self.emb_dim)
        if y is not None:
            emb = emb + self.label_emb(y)
        if cond is not None:
            emb = emb + cond
        emb = self.emb2(nnx.silu(self.emb1(emb)))
        up = lambda h, skip: jnp.concatenate([jax.image.resize(
            h, h.shape[:1] + skip.shape[1:3] + h.shape[3:], 'nearest'),
            skip], -1)
        h1 = self.down1(self.inc(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        h = self.mid(self.pool2(h2), emb)
        h = self.up1(up(h, h2), emb)  # skip connection
        h = self.up0(up(h, h1), emb)  # skip connection
        return self.outc(nnx.silu(self.norm(h)))
```

### Conditioning on a Class

(to be written)

## The Training Recipe

(to be written)

### One Trainer

(to be written)

### The Moving Average

(to be written)

## Measuring Sample Quality

(to be written)

### Fréchet Distance in Learned Features

(to be written)

### Reading the Number

(to be written)

## Guidance

(to be written)

### Bayes' Rule on Scores

(to be written)

### Classifier-Free Guidance

(to be written)

`cfg` turns a class-conditional network into a guided one with the same interface as an unconditional network: it evaluates the network with the label and with the null label in one doubled batch and returns $\mathbf{u} + \gamma(\mathbf{u}_c - \mathbf{u})$.

```{.python .input #image-diffusion-classifier-free-guidance}
%%tab pytorch
def cfg(net, c, gamma):  #@save
    """Classifier-free guidance: (x, t) -> u + gamma (u_c - u), u unlabeled."""
    null = torch.full_like(c, net.num_classes)
    def guided(x, t):
        out_c, out = net(torch.cat([x, x]), torch.cat([t, t]),
                         torch.cat([c, null])).chunk(2)
        return out + gamma * (out_c - out)
    return guided
```

```{.python .input #image-diffusion-classifier-free-guidance}
%%tab jax
def cfg(net, c, gamma):  #@save
    """Classifier-free guidance: (x, t) -> u + gamma (u_c - u), u unlabeled."""
    null = jnp.full_like(c, net.num_classes)
    def guided(x, t):
        out_c, out = jnp.split(net(jnp.concatenate([x, x]),
                                   jnp.concatenate([t, t]),
                                   jnp.concatenate([c, null])), 2)
        return out + gamma * (out_c - out)
    return guided
```

### Guidance as a Tilt

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
