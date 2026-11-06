# Samplers, Inversion and Editing
:label:`sec_diffusion-samplers`

(to be written)

```{.python .input #samplers-samplers-inversion-and-editing}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #samplers-samplers-inversion-and-editing}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## One Field, Two Samplers

(to be written)

### DDIM Is Euler's Method

(to be written)

`ddim_sample` runs the DDIM family on the grid of `path.grid` with any predictor `pred(x, alpha, sigma) -> (x0, eps)`: each step keeps the predicted clean point, re-uses the predicted noise for the part of $\sigma_s$ that $\eta$ leaves deterministic, and draws the rest fresh. It stops at `t_end`, data by default, so a loop can act on the sample between levels.

```{.python .input #samplers-ddim-is-euler-s-method}
%%tab pytorch
@torch.no_grad()  #@save
def ddim_sample(pred, x, path, steps, eta=0.0, *, t_start=1.0, t_end=0.0):
    """DDIM from x at t_start (1 is noise) to t_end; eta = 0 is deterministic.
    eta = 1 draws the posterior variance: not the ancestral sampler's."""
    a, s = path.grid(steps, t_start, t_end)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        rho = s[i + 1] * a[i] / (a[i + 1] * s[i])
        var = (eta * s[i + 1]) ** 2 * (1 - rho ** 2)     # fresh noise
        x = a[i + 1] * x0 + math.sqrt(s[i + 1] ** 2 - var) * eps
        if var > 0:
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x
```

```{.python .input #samplers-ddim-is-euler-s-method}
%%tab jax
def ddim_sample(pred, x, path, steps, eta=0.0, key=None, *,  #@save
                t_start=1.0, t_end=0.0):
    """DDIM from x at t_start (1 is noise) to t_end; eta > 0 needs a key.
    eta = 1 draws the posterior variance: not the ancestral sampler's."""
    if eta > 0 and key is None:
        raise ValueError('ddim_sample with eta > 0 draws noise: pass a key')
    a, s = path.grid(steps, t_start, t_end)
    a_t, s_t, a_s, s_s = a[:-1], s[:-1], a[1:], s[1:]
    rho = s_s * a_t / (a_s * s_t)
    var = (eta * s_s) ** 2 * (1 - rho ** 2)             # fresh noise
    def step(x, inputs):
        a_t, s_t, a_s, c_eps, c_noise, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + c_eps * eps
        if eta > 0:
            x = x + c_noise * jax.random.normal(key, x.shape)
        return x, None
    keys = (jax.random.split(key, steps) if eta > 0
            else jnp.zeros(steps))                      # unused at eta = 0
    return jax.lax.scan(step, x, (a_t, s_t, a_s, jnp.sqrt(s_s ** 2 - var),
                                  jnp.sqrt(var), keys))[0]
```

### Stochastic Steps Correct, Deterministic Steps Carry

(to be written)

## Fewer Steps

(to be written)

### Skipping Steps on the Running Example

(to be written)

### Second Order and the Step Grid

(to be written)

### Quality against Evaluations on Images

(to be written)

## Inversion and Editing

(to be written)

### The Latent Code

(to be written)

`ddim_invert` runs the deterministic sampler backwards, from an image at $t = 0$ toward noise; each step takes the predicted noise at the far end of the step, since the predictor needs $\sigma > 0$.

```{.python .input #samplers-the-latent-code}
%%tab pytorch
@torch.no_grad()  #@save
def ddim_invert(pred, x, path, steps, t_end=1.0):
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    for i in range(steps):
        eps = pred(x, a[i + 1], s[i + 1])[1]    # x read at the next level
        x = a[i + 1] * (x - s[i] * eps) / a[i] + s[i + 1] * eps
    return x
```

```{.python .input #samplers-the-latent-code}
%%tab jax
def ddim_invert(pred, x, path, steps, t_end=1.0):  #@save
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    def step(x, inputs):
        a_t, s_t, a_s, s_s = inputs
        eps = pred(x, a_s, s_s)[1]              # x read at the next level
        return a_s * (x - s_t * eps) / a_t + s_s * eps, None
    return jax.lax.scan(step, x, (a[:-1], s[:-1], a[1:], s[1:]))[0]
```

### Editing by Partial Noising

(to be written)

### Inpainting and Posterior Sampling

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
