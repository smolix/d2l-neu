# Denoising and the Score
:label:`sec_diffusion-denoising`

(to be written)

```{.python .input #denoising-denoising-score-matching}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #denoising-denoising-score-matching}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## The Running Example

(to be written)

`GaussianMixture` holds the running example: a mixture of isotropic Gaussians whose density, score and posterior mean are available in closed form at every noise level $\mathbf{x}_t = \alpha \mathbf{x}_0 + \sigma \boldsymbol\epsilon$.

```{.python .input #denoising-the-running-example-1}
%%tab pytorch
#@save
class GaussianMixture:
    """A mixture of isotropic Gaussians, exact at every noise level."""
    def __init__(self, means, weights, std, device=None):
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float32,
                                    device=device)
        self.std = std

    @staticmethod
    def three_modes(device=None):
        """The chapter's running example, with mode weights 0.5/0.3/0.2."""
        return GaussianMixture([[-2.5, -1.5], [2.5, -1.5], [0.0, 2.5]],
                               [0.5, 0.3, 0.2], 0.5, device)

    def sample(self, n):
        k = torch.multinomial(self.weights, n, replacement=True)
        return self.means[k] + self.std * torch.randn(
            n, self.means.shape[1], device=self.means.device)

    def _modes(self, x, alpha, sigma):
        """Per-mode log densities of x = alpha x0 + sigma eps, the offsets
        x - alpha mu_k (n, K, d) and the per-mode variance."""
        a, s = (torch.as_tensor(v, dtype=x.dtype, device=x.device)
                .reshape(-1, 1, 1) for v in (alpha, sigma))
        var = (a * self.std) ** 2 + s ** 2
        r = x[:, None] - a * self.means
        log_p = (torch.log(self.weights) - (r ** 2).sum(-1) / (2 * var[..., 0])
                 - x.shape[1] / 2 * torch.log(2 * torch.pi * var[..., 0]))
        return log_p, r, var

    def log_prob(self, x, alpha=1.0, sigma=0.0):
        return torch.logsumexp(self._modes(x, alpha, sigma)[0], 1)

    def nll(self, x):
        """Mean negative log-likelihood of clean points, in nats."""
        return -self.log_prob(x).mean()

    def score(self, x, alpha=1.0, sigma=0.0):
        log_p, r, var = self._modes(x, alpha, sigma)
        return -(torch.softmax(log_p, 1)[..., None] * r / var).sum(1)

    def posterior_mean(self, x, alpha, sigma):
        """E[x0 | x]; finite at alpha = 0 and at sigma = 0."""
        log_p, r, var = self._modes(x, alpha, sigma)
        gain = torch.as_tensor(alpha, device=x.device).reshape(-1, 1, 1) \
            * self.std ** 2 / var
        return (torch.softmax(log_p, 1)[..., None]
                * (self.means + gain * r)).sum(1)

    def velocity(self, x, t):
        """E[eps - x0 | x_t] on the linear path x_t = (1 - t) x0 + t eps."""
        log_p, r, var = self._modes(x, 1 - t, t)
        s = torch.as_tensor(t, device=x.device).reshape(-1, 1, 1)
        eps = s * r / var                       # E[eps | x_t, mode k]
        return (torch.softmax(log_p, 1)[..., None]
                * (eps - self.means - (1 - s) * self.std ** 2 / var * r)).sum(1)

    def summary(self, x):
        """Mode weights by nearest mean, the fraction of strays farther than
        3 std from every mean, and the mean squared distance of the rest."""
        d2 = ((x[:, None] - self.means) ** 2).sum(-1)
        near, k = d2.min(1)
        inside = near < (3 * self.std) ** 2
        weights = torch.bincount(k, minlength=len(self.weights)) / len(x)
        return dict(weights=[round(w, 3) for w in weights.tolist()],
                    stray=1 - inside.float().mean().item(),
                    spread=near[inside].mean().item())
```

```{.python .input #denoising-the-running-example-1}
%%tab jax
#@save
class GaussianMixture:
    """A mixture of isotropic Gaussians, exact at every noise level."""
    def __init__(self, means, weights, std):
        self.means = jnp.array(means, dtype=jnp.float32)
        self.weights = jnp.array(weights, dtype=jnp.float32)
        self.std = std

    @staticmethod
    def three_modes():
        """The chapter's running example, with mode weights 0.5/0.3/0.2."""
        return GaussianMixture([[-2.5, -1.5], [2.5, -1.5], [0.0, 2.5]],
                               [0.5, 0.3, 0.2], 0.5)

    def sample(self, key, n):
        k1, k2 = jax.random.split(key)
        k = jax.random.choice(k1, len(self.weights), (n,), p=self.weights)
        return self.means[k] + self.std * jax.random.normal(
            k2, (n, self.means.shape[1]))

    def _modes(self, x, alpha, sigma):
        """Per-mode log densities of x = alpha x0 + sigma eps, the offsets
        x - alpha mu_k (n, K, d) and the per-mode variance."""
        a, s = (jnp.asarray(v, x.dtype).reshape(-1, 1, 1)
                for v in (alpha, sigma))
        var = (a * self.std) ** 2 + s ** 2
        r = x[:, None] - a * self.means
        log_p = (jnp.log(self.weights) - (r ** 2).sum(-1) / (2 * var[..., 0])
                 - x.shape[1] / 2 * jnp.log(2 * jnp.pi * var[..., 0]))
        return log_p, r, var

    def log_prob(self, x, alpha=1.0, sigma=0.0):
        return jax.nn.logsumexp(self._modes(x, alpha, sigma)[0], 1)

    def nll(self, x):
        """Mean negative log-likelihood of clean points, in nats."""
        return -self.log_prob(x).mean()

    def score(self, x, alpha=1.0, sigma=0.0):
        log_p, r, var = self._modes(x, alpha, sigma)
        return -(jax.nn.softmax(log_p, 1)[..., None] * r / var).sum(1)

    def posterior_mean(self, x, alpha, sigma):
        """E[x0 | x]; finite at alpha = 0 and at sigma = 0."""
        log_p, r, var = self._modes(x, alpha, sigma)
        gain = jnp.asarray(alpha).reshape(-1, 1, 1) * self.std ** 2 / var
        return (jax.nn.softmax(log_p, 1)[..., None]
                * (self.means + gain * r)).sum(1)

    def velocity(self, x, t):
        """E[eps - x0 | x_t] on the linear path x_t = (1 - t) x0 + t eps."""
        log_p, r, var = self._modes(x, 1 - t, t)
        s = jnp.asarray(t).reshape(-1, 1, 1)
        eps = s * r / var                       # E[eps | x_t, mode k]
        return (jax.nn.softmax(log_p, 1)[..., None]
                * (eps - self.means - (1 - s) * self.std ** 2 / var * r)).sum(1)

    def summary(self, x):
        """Mode weights by nearest mean, the fraction of strays farther than
        3 std from every mean, and the mean squared distance of the rest."""
        d2 = ((x[:, None] - self.means) ** 2).sum(-1)
        near, k = d2.min(1), d2.argmin(1)
        inside = near < (3 * self.std) ** 2
        weights = jnp.bincount(k, length=len(self.weights)) / len(x)
        return dict(weights=[round(float(w), 3) for w in weights],
                    stray=1 - float(inside.mean()),
                    spread=float(near[inside].mean()))
```

`scatter_panels` draws point sets side by side on shared, equal axes, with the training data in grey behind them.

```{.python .input #denoising-the-running-example-2}
%%tab pytorch, jax
#@save
def scatter_panels(samples, titles=None, data=None, lim=4.5, size=3.0):
    """Scatter 2-D point sets side by side on equal, shared axes."""
    _, axes = d2l.plt.subplots(1, len(samples), squeeze=False,
                               figsize=(size * len(samples), size))
    for i, (ax, pts) in enumerate(zip(axes[0], samples)):
        if data is not None:
            ax.scatter(*d2l.numpy(data).T, s=2, c='lightgray')
        ax.scatter(*d2l.numpy(pts).T, s=2)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
        if titles:
            ax.set_title(titles[i])
    return axes[0]
```

## A Denoiser Is a Posterior Mean

(to be written)

### Noise and the Regression Target

(to be written)

`TimeMLP` is the one network for every two-dimensional experiment: an MLP on the point plus sinusoidal features of a noise level or time, from `time_embedding`, which the image networks share.

```{.python .input #denoising-noise-and-the-regression-target}
%%tab pytorch
#@save
def time_embedding(t, dim, scale=1.0):
    """Sines and cosines of scale * t at dim / 2 geometric frequencies."""
    freqs = torch.exp(-math.log(10000) / (dim // 2)
                      * torch.arange(dim // 2, device=t.device))
    args = scale * t.float()[:, None] * freqs
    return torch.cat([args.sin(), args.cos()], 1)

#@save
class TimeMLP(nn.Module):
    """An MLP on a point x and the time features of one or more times."""
    def __init__(self, hidden=256, depth=3, *, dim=2, num_times=1,
                 time_scale=1.0):
        super().__init__()
        self.time_scale = time_scale
        self.embed = nn.Sequential(nn.Linear(64 * num_times, hidden),
                                   nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(dim, hidden)
        layers = [m for _ in range(depth - 1)
                  for m in (nn.SiLU(), nn.Linear(hidden, hidden))]
        self.body = nn.Sequential(*layers, nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, *ts):
        feats = torch.cat([time_embedding(t, 64, self.time_scale)
                           for t in ts], 1)
        return self.body(self.inp(x) + self.embed(feats))
```

```{.python .input #denoising-noise-and-the-regression-target}
%%tab jax
#@save
def time_embedding(t, dim, scale=1.0):
    """Sines and cosines of scale * t at dim / 2 geometric frequencies."""
    freqs = jnp.exp(-math.log(10000) / (dim // 2) * jnp.arange(dim // 2))
    args = scale * jnp.asarray(t, jnp.float32)[:, None] * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], 1)

#@save
class TimeMLP(nnx.Module):
    """An MLP on a point x and the time features of one or more times."""
    def __init__(self, hidden=256, depth=3, *, dim=2, num_times=1,
                 time_scale=1.0, rngs):
        self.time_scale = time_scale
        self.embed = nnx.Sequential(
            d2l.Linear(64 * num_times, hidden, rngs=rngs), nnx.silu,
            d2l.Linear(hidden, hidden, rngs=rngs))
        self.inp = d2l.Linear(dim, hidden, rngs=rngs)
        layers = [m for _ in range(depth - 1)
                  for m in (nnx.silu, d2l.Linear(hidden, hidden, rngs=rngs))]
        self.body = nnx.Sequential(*layers, nnx.silu,
                                   d2l.Linear(hidden, dim, rngs=rngs))

    def __call__(self, x, *ts):
        feats = jnp.concatenate([time_embedding(t, 64, self.time_scale)
                                 for t in ts], 1)
        return self.body(self.inp(x) + self.embed(feats))
```

### Tweedie's Formula

(to be written)

### The Normalizer Never Appears

(to be written)

## What One Noise Level Can and Cannot Do

(to be written)

### The Loss Floor and the Noise Level

(to be written)

### Blindness to Mode Weights

(to be written)

### Sampling with a Score

(to be written)

## Denoising Images

(to be written)

`load_fashion_mnist_pm1` returns the Fashion-MNIST training and test images scaled to $[-1, 1]$ (or as binary pixels) with their labels, and `show_gray` displays such images as a grid.

```{.python .input #denoising-denoising-images}
%%tab pytorch
#@save
def load_fashion_mnist_pm1(binary=False):
    """Fashion-MNIST on the GPU if there is one: (X, y, X_test, y_test) with
    images (n, 1, 28, 28) in [-1, 1], or binary pixels (n, 28, 28)."""
    data, device = d2l.FashionMNIST(), d2l.try_gpu()
    def prep(X):
        return (X > 127).long() if binary else X[:, None].float() / 127.5 - 1
    return (prep(data.train.data).to(device), data.train.targets.to(device),
            prep(data.val.data).to(device), data.val.targets.to(device))

#@save
def show_gray(imgs, num_rows, num_cols, titles=None, scale=1.0):
    """Show images (n, 1, h, w) in [-1, 1], or binary (n, h, w), in gray."""
    imgs = imgs.detach().cpu()
    imgs = ((imgs[:, 0] + 1) / 2).clamp(0, 1) if imgs.is_floating_point() \
        else imgs.float()
    return d2l.show_images(imgs[..., None].expand(-1, -1, -1, 3), num_rows,
                           num_cols, titles, scale)
```

```{.python .input #denoising-denoising-images}
%%tab jax
#@save
def load_fashion_mnist_pm1(binary=False):
    """Fashion-MNIST as arrays: (X, y, X_test, y_test) with images
    (n, 28, 28, 1) in [-1, 1], or binary pixels (n, 28, 28)."""
    data = d2l.FashionMNIST()
    def prep(X):
        return (jnp.asarray(X > 127, jnp.int32) if binary
                else jnp.asarray(X[..., None], jnp.float32) / 127.5 - 1)
    return (prep(data.train[0]), jnp.asarray(data.train[1], jnp.int32),
            prep(data.val[0]), jnp.asarray(data.val[1], jnp.int32))

#@save
def show_gray(imgs, num_rows, num_cols, titles=None, scale=1.0):
    """Show images (n, h, w, 1) in [-1, 1], or binary (n, h, w), in gray."""
    imgs = jnp.asarray(imgs)
    imgs = jnp.clip((imgs[..., 0] + 1) / 2, 0, 1) \
        if jnp.issubdtype(imgs.dtype, jnp.floating) else imgs.astype(float)
    return d2l.show_images(jnp.repeat(imgs[..., None], 3, -1), num_rows,
                           num_cols, titles, scale)
```

## Summary

(to be written)

## Exercises

(to be written)
