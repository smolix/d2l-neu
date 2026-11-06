# Few-Step Generation
:label:`sec_diffusion-few-step`

(to be written)

```{.python .input #few-step-few-step-generation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
import math
import torch
from torch import nn
```

```{.python .input #few-step-few-step-generation}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## Why One Step Fails

(to be written)

### The Average Where the Data Are Not

(to be written)

### Straighter Paths Help but Do Not Finish

(to be written)

## Flow Maps

(to be written)

### The Integral of the Velocity

(to be written)

### Consistency Models

(to be written)

### Average Velocity and the MeanFlow Identity

(to be written)

One loss trains a flow map $u(\mathbf{x}, s, t)$ for every route: it forms
$\mathbf{x}_t$ on the linear path, takes the velocity $\mathbf{v}$ from a
teacher or, without one, the conditional velocity
$\boldsymbol\epsilon - \mathbf{x}_0$, computes $u$ and its total derivative
along the path in one Jacobian-vector product with tangent $(\mathbf{v}, 1)$,
and regresses $u$ onto $\mathbf{v} - (t - s)\, du/dt$ held fixed;
$s = 0$ with a teacher is consistency distillation.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity}
%%tab pytorch
def flow_map_loss(u, teacher, x0, eps, s, t):  #@save
    """Squared error of the average velocity u(x_t, s, t) against the
    stop-gradient target v - (t - s) du/dt; v from teacher, or eps - x0."""
    x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
    if teacher is None:
        v = eps - x0
    else:
        with torch.no_grad():
            v = teacher(x_t, t)
    u_t, du_dt = torch.func.jvp(lambda x, r: u(x, s, r), (x_t, t),
                                (v, torch.ones_like(t)))
    gap = (t - s).reshape(-1, *[1] * (x0.dim() - 1))
    target = (v - gap * du_dt).detach()
    return ((u_t - target) ** 2).flatten(1).sum(1).mean()
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity}
%%tab jax
def flow_map_loss(u, teacher, x0, eps, s, t):  #@save
    """Squared error of the average velocity u(x_t, s, t) against the
    stop-gradient target v - (t - s) du/dt; v from teacher, or eps - x0."""
    x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
    v = (eps - x0 if teacher is None
         else jax.lax.stop_gradient(teacher(x_t, t)))
    u_t, du_dt = jax.jvp(lambda x, r: u(x, s, r), (x_t, t),
                         (v, jnp.ones_like(t)))
    gap = (t - s).reshape(-1, *[1] * (x0.ndim - 1))
    target = jax.lax.stop_gradient(v - gap * du_dt)
    return ((u_t - target) ** 2).reshape(x0.shape[0], -1).sum(1).mean()
```

A flow map takes $\mathbf{x}_t$ to $\mathbf{x}_s$ in one jump,
$\mathbf{x}_s = \mathbf{x}_t - (t - s)\, u(\mathbf{x}_t, s, t)$, so a few-step
sampler cuts $[0, 1]$ into equal intervals and jumps across each.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-1}
%%tab pytorch
@torch.inference_mode()
def flow_map_sample(u, x, steps):  #@save
    """Jump from noise x at t = 1 to data at t = 0 in equal steps."""
    ts = torch.linspace(1, 0, steps + 1, device=x.device)
    for t, s in zip(ts[:-1], ts[1:]):
        x = x - (t - s) * u(x, s.expand(len(x)), t.expand(len(x)))
    return x
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-1}
%%tab jax
def flow_map_sample(u, x, steps):  #@save
    """Jump from noise x at t = 1 to data at t = 0 in equal steps."""
    ts = jnp.linspace(1, 0, steps + 1)
    def jump(x, ts):
        t, s = ts
        n = x.shape[0]
        return x - (t - s) * u(x, jnp.full(n, s), jnp.full(n, t)), None
    return jax.lax.scan(jump, x, (ts[:-1], ts[1:]))[0]
```

On images the student starts as a copy of the teacher's U-Net. The gap
$t - s$ enters through its own time embedding and a linear map initialized at
zero, added to the U-Net's embedding, so that at the start
$u(\mathbf{x}, s, t) = \mathbf{v}(\mathbf{x}, t)$, the velocity, which is the
average velocity over an interval of length zero.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-2}
%%tab pytorch
class UNetFlowMap(nn.Module):  #@save
    """Average velocity u(x, s, t) on a velocity U-Net whose embedding gains
    a zero-initialized projection of the gap t - s."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet.requires_grad_(True)  # an EMA copy arrives frozen
        self.gap = nn.Linear(unet.emb_dim, unet.emb_dim)
        nn.init.zeros_(self.gap.weight), nn.init.zeros_(self.gap.bias)

    def forward(self, x, s, t):
        gap = self.gap(d2l.time_embedding(t - s, self.unet.emb_dim))
        return self.unet(x, t, cond=gap)
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-2}
%%tab jax
class UNetFlowMap(nnx.Module):  #@save
    """Average velocity u(x, s, t) on a velocity U-Net whose embedding gains
    a zero-initialized projection of the gap t - s."""
    def __init__(self, unet, *, rngs):
        self.unet = unet
        zeros = nnx.initializers.zeros
        self.gap = nnx.Linear(unet.emb_dim, unet.emb_dim, kernel_init=zeros,
                              bias_init=zeros, rngs=rngs)

    def __call__(self, x, s, t):
        gap = self.gap(d2l.time_embedding(t - s, self.unet.emb_dim))
        return self.unet(x, t, cond=gap)
```

## Experiments

(to be written)

### On the Running Example

(to be written)

### On Images

(to be written)

## The Other Route: Matching Distributions

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
