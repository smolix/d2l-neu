# Few-Step Generation
:label:`sec_diffusion-few-step`

A diffusion model draws a sample by calling its network again and again.
The samplers of :numref:`sec_diffusion-samplers` brought the thousand calls
of the ancestral chain down to about fifty, and the latent models of
:numref:`sec_diffusion-latent` made each call cheaper, but fifty is still
fifty, and the generator of a GAN (:numref:`chap_gans`) needed one. Why not
take a single long step? It fails, and not for lack of a good network: even
with the exact velocity field, one Euler step of the probability-flow ODE
sends every start to the mean of the data
(:numref:`sec_diffusion-integration`). For a mixture of three blobs that
point lies between the blobs, where no data are.

A velocity tells the sampler which way to move; one step needs to know
where it will end up. So we let the network learn the solution of the ODE
instead of its right-hand side, a map from a point at one time to where its
trajectory is at an earlier time. :citet:`Song.Dhariwal.Chen.ea.2023`
learned the jump to the data by distilling a trained diffusion model, the
*teacher*, into a new network, the *student*, and called the student a
*consistency model*. :citet:`Geng.Deng.Bai.ea.2025` learned jumps over any
interval from data alone and called the method *MeanFlow*. Training remains
a regression with one network, and the same noise always gives the same
sample, so a student can be compared with its teacher image by image.
:numref:`sec_gan_beyond` trained one-step generators against a critic
instead; we return to that route at the end. First we see what one and a
few Euler steps get wrong, then derive one loss for both methods and train
students on the running example and on Fashion-MNIST.

```{.python .input #few-step-few-step-generation}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #few-step-few-step-generation}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #few-step-few-step-generation-2}
%%tab pytorch
import copy
```

```{.python .input #few-step-few-step-generation-1}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # tf32 matmuls on tensor cores
```

## Long Euler Steps

On an exact field a long Euler step makes one error: it follows the
tangent while the trajectory bends. At $t = 1$ this error is as large as it
can be. The point is pure noise and says nothing about its destination, so
from every start the velocity $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} -
\mathbb{E}[\mathbf{x}_0]$ points at the same place, the data mean
(:numref:`sec_diffusion-integration`), and a step of length one lands
there. Later steps start from points that have begun to commit to a mode
and aim better, but each still cuts across the curve.

### Euler Steps on the Exact Field

:numref:`sec_diffusion-integration` showed one, ten and a hundred steps;
what do two and four leave? `mix.velocity` evaluates the exact field of the
mixture, `d2l.make_pred` turns it into the predictor that `d2l.ode_sample`
integrates, and we run Euler's method with 1, 2, 4 and 10 steps from the
same 20,000 Gaussian starts. The rows for two and four steps are the
baseline that the students of this section must beat.

```{.python .input #few-step-the-average-where-the-data-are-not}
%%tab pytorch
device = d2l.try_gpu()
mix = d2l.GaussianMixture.three_modes(device)
lin = d2l.GaussianPath.linear()
pred_exact = d2l.make_pred(mix.velocity, lin, 'velocity')
torch.manual_seed(0)
z = torch.randn(20_000, 2, device=device)
euler = {n: d2l.ode_sample(pred_exact, z, lin.sigma_bars(n))
         for n in (1, 2, 4, 10)}
for n, x in euler.items():
    s = mix.summary(x)
    print(f'{n:2d} Euler steps: weights {s["weights"]}, '
          f'stray {s["stray"]:.1%}')
d2l.scatter_panels([x[:2000] for x in euler.values()],
                   [f'{n} steps' for n in euler], data=mix.sample(2000),
                   size=2.4);
```

```{.python .input #few-step-the-average-where-the-data-are-not}
%%tab jax
mix = d2l.GaussianMixture.three_modes()
lin = d2l.GaussianPath.linear()
pred_exact = d2l.make_pred(mix.velocity, lin, 'velocity')
z = jax.random.normal(jax.random.key(0), (20_000, 2))
euler = {n: d2l.ode_sample(pred_exact, z, lin.sigma_bars(n))
         for n in (1, 2, 4, 10)}
for n, x in euler.items():
    s = mix.summary(x)
    print(f'{n:2d} Euler steps: weights {s["weights"]}, '
          f'stray {s["stray"]:.1%}')
d2l.scatter_panels([x[:2000] for x in euler.values()],
                   [f'{n} steps' for n in euler],
                   data=mix.sample(jax.random.key(1), 2000), size=2.4);
```

One step puts every sample on the mean $(-0.5, -0.7)$, which `summary`
counts as stray and assigns to the nearest mode, the heaviest: hence the
weights 1/0/0. Two steps reach the modes but leave about a quarter of the
samples between them, with weights of about 0.64/0.22/0.14 instead of
0.5/0.3/0.2. The first half-step still aims at the mean, which lies
nearest to the heaviest mode, and that mode gains. Four steps still leave
about 6.5% of the samples stray, and ten about 2%, against about 1% for
the data themselves. With an exact field, the excess is the error of the
step.

### Reflow

Why do the trajectories bend at all? Each training example joins a data
point to an independently drawn noise vector by a straight segment, but
the segments of different examples cross. Where they cross, the field
averages their directions, and the trajectories, which cannot cross, bend
around each other (:numref:`sec_mdl-rectified-flow`). The curvature comes
from the pairing of noise with data, not from the loss.

*Reflow* :cite:`Liu.Gong.Liu.2022` changes the pairing. It pairs each noise
vector with the sample that a trained model reaches from it and trains the
network again on these pairs. Each new segment is the chord of one
trajectory of the old model, and since those trajectories never cross,
their chords rarely do. :numref:`subsec_mdl-reflow` proves that reflow
with the exact field keeps the data distribution and never increases the
transport cost, and on two-moons data one Euler step of the reflowed model
scores as well as thirty-two steps of the original.

A trained field weakens the guarantee. The new pairs end at the model's
samples, not at the data, so the reflowed model inherits the original's
errors: on two moons it reproduces the original's thirty-two-step samples,
blur included (:numref:`subsec_mdl-reflow`), and an image model passes on
every defect of its samples in the same way. Each round also costs a
training run, and the best one-step results of :citet:`Liu.Gong.Liu.2022`
still came from distilling the reflowed model. The rest of this section
changes what the network outputs instead of how the data are paired.

## Flow Maps

A velocity model says which way a trajectory points; a flow map says where
it arrives.

### The Integral of the Velocity

Run the probability-flow ODE $d\mathbf{x}_\tau / d\tau =
\mathbf{u}_\tau(\mathbf{x}_\tau)$ from a point $\mathbf{x}$ at time $t$
back to an earlier time $s \leq t$. The point where it arrives is the
*flow map* $\Phi_{t \to s}(\mathbf{x})$. For a field that is Lipschitz in
$\mathbf{x}$ the exact flow map is a bijection
(:numref:`sec_mdl-vector-fields-trajectories`), and the sampler we want is
$\Phi_{1 \to 0}$.

Integrating the ODE along the trajectory $\mathbf{x}_\tau$ through
$\mathbf{x}_t$ writes the flow map as a displacement,

$$\Phi_{t \to s}(\mathbf{x}_t) = \mathbf{x}_t - \int_s^t \mathbf{u}_\tau(\mathbf{x}_\tau)\, d\tau = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t),$$
:eqlabel:`eq_diffusion-flow-map`

where $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$ is the *average velocity* along
the trajectory between the two times. Euler's method replaces it by the
velocity at the start, $\mathbf{u}_t(\mathbf{x}_t)$, with an error that
grows with the step (:numref:`sec_mdl-euler-runge-kutta`). Over an interval
of length zero the two agree, $\bar{\mathbf{u}}(\mathbf{x}, t, t) =
\mathbf{u}_t(\mathbf{x})$, so a network for $\bar{\mathbf{u}}$ is a velocity
model with one more input, the end time.

The smallest case shows what the average velocity knows that the velocity
does not. Let the data be one-dimensional and Gaussian with mean $m$ and
standard deviation $\varsigma$. Then $x_t$ is Gaussian with mean $(1 - t)\,
m$ and standard deviation $k_t = \sqrt{(1 - t)^2 \varsigma^2 + t^2}$. In
one dimension trajectories cannot pass each other, so the flow keeps each
point's quantile, and for a Gaussian the quantile is fixed by the
standardized position $(x - (1 - t) m) / k_t$. The flow map is therefore
affine:

$$\Phi_{t \to s}(x) = (1 - s)\, m + \frac{k_s}{k_t} \big(x - (1 - t)\, m\big).$$

With $m = 2$ and $\varsigma = 0.5$ the full jump $\Phi_{1 \to 0}(x) = 2 +
0.5\, x$ sends standard Gaussian noise to the data $\mathcal{N}(2, 0.5^2)$,
with average velocity $\bar{u}(x, 0, 1) = 0.5\, x - 2$. The velocity at the
start is $u_1(x) = x - 2$, and its Euler step returns 2 for every $x$. The
average velocity removes only half of the noise, because half of the
noise's spread must survive as the spread of the data.

How can a network learn $\Phi$ when nobody can tell it the values? Through
an identity that only the true flow map satisfies, turned into a loss.
There is one identity for each way of moving the two times along a
trajectory (:numref:`fig_diffusion-flow-map`). Let $\mathbf{x}_\tau$ be the
trajectory through $\mathbf{x}$ at time $t$, and $s \leq r \leq t$.

The *compositional* identity says that two jumps make one,
$\Phi_{r \to s}(\Phi_{t \to r}(\mathbf{x})) = \Phi_{t \to s}(\mathbf{x})$.
A loss built on it needs no derivative, only the model at an intermediate
time, and its target is as accurate as the shorter jumps it is built from.
Progressive distillation :cite:`Salimans.Ho.2022` and shortcut models
:cite:`Frans.Hafner.Levine.ea.2025` use it.

The *Lagrangian* identity fixes the start and moves the end. The mapped
point then moves with the field, like a particle,
$\partial_s \Phi_{t \to s}(\mathbf{x}) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}))$.
A loss built on it needs the velocity at the point where the jump lands. A
training example gives its conditional velocity only at its own
$\mathbf{x}_t$, so a teacher must supply the velocity there, or the flow
map itself (Exercise :exref:`ex_few-step-lagrangian`).

The *Eulerian* identity fixes the end and slides the start along the
trajectory: every point of one trajectory maps to the same place,

$$\frac{d}{dt} \Phi_{t \to s}(\mathbf{x}_t) = \partial_t \Phi_{t \to s}(\mathbf{x}_t) + \nabla_{\mathbf{x}} \Phi_{t \to s}(\mathbf{x}_t)\, \mathbf{u}_t(\mathbf{x}_t) = \mathbf{0}.$$
:eqlabel:`eq_diffusion-eulerian`

A loss built on it needs one Jacobian-vector product and the velocity at
the start, where the training example has one, and its target involves a
single pair of times, with no intermediate time and no step size.

In the Gaussian example all three can be checked by hand: the ratios
$k_s / k_t$ compose; the point $\Phi_{t \to s}(x)$, followed as $s$ varies,
keeps its standardized position and so moves as a point of the flow does;
and the standardized position is constant along a trajectory.

![The flow map. Left: trajectories of the probability-flow ODE for a one-dimensional mixture of two Gaussians, from the noise at $t = 1$ (top) to the data at $t = 0$ (bottom); the chord from $(\mathbf{x}_t, t)$ to $(\mathbf{x}_s, s)$ is one jump of the flow map. Right, on one trajectory: two jumps through $r$ make one (compositional); the start fixed at $t$ and the end moving (Lagrangian); the end fixed at $s$ and the start moving (Eulerian).](../img/mdl-diffusion-flow-map.svg)
:label:`fig_diffusion-flow-map`

Together with the condition at $s = t$, where the map is the identity and
its slope in $s$ is the velocity, each identity determines the flow map.
They differ only in what a loss must compute. Composition is the simplest;
the Eulerian identity is the only one that needs neither an intermediate
time nor a velocity where the jump lands, and it pays for that with one
Jacobian-vector product. Consistency models and MeanFlow both use the
Eulerian identity; :citet:`Boffi.Albergo.VandenEijnden.2025` sort the
few-step methods by the identity they enforce.

### Consistency Models

The first flow-map method kept only the jump to the data. A *consistency
model* :cite:`Song.Dhariwal.Chen.ea.2023` is a network $f(\mathbf{x}, t)$
trained to output $\Phi_{t \to 0}(\mathbf{x})$ at every time. Its training
condition is that the output does not change as the input moves along a
trajectory, $\frac{d}{dt} f(\mathbf{x}_t, t) = \mathbf{0}$ with
$f(\mathbf{x}, 0) = \mathbf{x}$, the Eulerian identity
:eqref:`eq_diffusion-eulerian` at $s = 0$.

In *consistency distillation*, Song et al. let the teacher take one ODE
step between neighbouring times of a grid, and require the student's outputs at the two points to agree. The output at
the earlier point serves as the target; it is held fixed and computed with
a moving average of the student's weights. As the grid is refined the
difference, divided by the step, becomes the derivative above, which
:citet:`Lu.Song.2025` use directly. *Consistency training* needs no
teacher: it replaces the teacher's velocity by the velocity
$\boldsymbol\epsilon - \mathbf{x}_0$ of each training example's straight
segment, the conditional velocity of flow matching.

One evaluation of a consistency model gives a sample. A second cannot jump
from $t = 1$ to $t = 0.5$, because the model knows only where trajectories
end. It *re-noises* instead: the sample is moved back to $t = 0.5$ with
fresh noise, $\mathbf{x}_{0.5} = 0.5\, \mathbf{x}_0 + 0.5\,
\boldsymbol\epsilon'$, and mapped to the data again, from a point where
trajectories bend less. A flow map with both times needs no re-noising:
two evaluations are two jumps along one trajectory, and the sample still
depends on the starting noise alone.

### Average Velocity and the MeanFlow Identity

Writing the flow map through the average velocity
:eqref:`eq_diffusion-flow-map` turns the Eulerian identity into a
regression target. Multiply the average velocity by the length of its
interval, which gives the integral of the velocity, and differentiate in
$t$ along the trajectory with $s$ held fixed:

$$\frac{d}{dt} \Big[(t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t)\Big] = \frac{d}{dt} \int_s^t \mathbf{u}_\tau(\mathbf{x}_\tau)\, d\tau = \mathbf{u}_t(\mathbf{x}_t).$$

The product rule on the left gives the *MeanFlow identity* of
:citet:`Geng.Deng.Bai.ea.2025`,

$$\bar{\mathbf{u}}(\mathbf{x}_t, s, t) = \mathbf{u}_t(\mathbf{x}_t) - (t - s)\, \frac{d}{dt} \bar{\mathbf{u}}(\mathbf{x}_t, s, t).$$
:eqlabel:`eq_diffusion-meanflow`

The average velocity is the velocity at the start, corrected by how fast
the average changes as the start slides along the trajectory. Substituting
:eqref:`eq_diffusion-flow-map` into :eqref:`eq_diffusion-eulerian` shows
that the two identities are the same. The total derivative is
$\partial_t \bar{\mathbf{u}} + (\nabla_{\mathbf{x}} \bar{\mathbf{u}})\,
\mathbf{u}_t$, the Jacobian-vector product of the network $(\mathbf{x}, t)
\mapsto \bar{\mathbf{u}}(\mathbf{x}, s, t)$ with the tangent
$(\mathbf{u}_t, 1)$, which forward-mode differentiation computes together
with the output at about the cost of a backward pass. A training step
thus costs about one backward pass more than a step of a velocity model,
plus one evaluation of the teacher when there is a teacher.

The identity becomes a loss by regressing the network onto its right-hand
side, computed from the network itself and held fixed. The marginal
velocity $\mathbf{u}_t$ is not available, but two substitutes are. A
teacher provides its estimate $\hat{\mathbf{u}}$ of
$\mathbf{u}_t(\mathbf{x}_t)$. Without one, the conditional velocity
$\boldsymbol\epsilon - \mathbf{x}_0$ of the training example serves, for
the reason that made flow matching work. The target is affine in the
velocity, directly and through the tangent of the Jacobian-vector product.
Averaged over the examples that could have produced $\mathbf{x}_t$, it is
therefore the target with $\mathbf{u}_t(\mathbf{x}_t)$ in place of the
conditional velocity, and squared error is minimized by that average.
Writing $\mathbf{u}_{\textrm{ref}}$ for the velocity we use, the *reference
velocity*, the loss at a training point is

$$\Big\| \bar{\mathbf{u}}_{\boldsymbol\theta}(\mathbf{x}_t, s, t) - \mathrm{sg}\Big[\mathbf{u}_{\textrm{ref}} - (t - s)\, \frac{d}{dt} \bar{\mathbf{u}}_{\boldsymbol\theta}(\mathbf{x}_t, s, t)\Big] \Big\|^2,$$
:eqlabel:`eq_diffusion-flow-map-loss`

where $\mathrm{sg}$, the *stop-gradient*, marks a quantity treated as a
constant when the loss is differentiated. It is required by the averaging
argument just made, which holds only if the network enters the loss
through the prediction alone. Differentiated through the target, the
network could lower the loss by reshaping the Jacobian inside the target,
which changes how much the target varies between examples, instead of by
satisfying the identity (Exercise :exref:`ex_few-step-stop-gradient`). Of
all the steps of the derivation this is the one a reimplementation most
easily gets wrong, since the code with and without it differs by a single
`.detach()`.

The loss leaves two choices: where $\mathbf{u}_{\textrm{ref}}$ comes from,
and whether the end time $s$ is free or fixed at zero. The main methods
fall into the four combinations, although some enforce the identity by
other means than this loss, and each adds details of its own that
:numref:`sec_diffusion-further-reading` describes.

| velocity $\mathbf{u}_{\textrm{ref}}$ | one time, $s = 0$ | two times, $s \leq t$ |
|---|---|---|
| a teacher's $\hat{\mathbf{u}}$ | consistency distillation | flow-map distillation :cite:`Kim.Lai.Liao.ea.2024` |
| the example's $\boldsymbol\epsilon - \mathbf{x}_0$ | consistency training | MeanFlow |
:label:`tab_diffusion-flow-map-methods`

At $s = 0$, :eqref:`eq_diffusion-flow-map` reads $f(\mathbf{x}, t) =
\mathbf{x} - t\, \bar{\mathbf{u}}(\mathbf{x}, 0, t)$, so the top-left cell
is consistency distillation in the continuous-time form of
:citet:`Lu.Song.2025`. `flow_map_error` implements
:eqref:`eq_diffusion-flow-map-loss` per example on the linear path for all
four: the network `u` computes $\bar{\mathbf{u}}$, `teacher=None` selects
the conditional velocity, and the caller chooses the times and averages.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity}
%%tab pytorch
def flow_map_error(u, teacher, x0, eps, s, t):  #@save
    """Per-example squared error of the average velocity u(x_t, s, t) against
    the stop-gradient target v - (t - s) du/dt; v from teacher, or eps - x0."""
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
    return ((u_t - target) ** 2).flatten(1).sum(1)
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity}
%%tab jax
def flow_map_error(u, teacher, x0, eps, s, t):  #@save
    """Per-example squared error of the average velocity u(x_t, s, t) against
    the stop-gradient target v - (t - s) du/dt; v from teacher, or eps - x0."""
    x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
    v = (eps - x0 if teacher is None
         else jax.lax.stop_gradient(teacher(x_t, t)))
    u_t, du_dt = jax.jvp(lambda x, r: u(x, s, r), (x_t, t),
                         (v, jnp.ones_like(t)))
    gap = (t - s).reshape(-1, *[1] * (x0.ndim - 1))
    target = jax.lax.stop_gradient(v - gap * du_dt)
    return ((u_t - target) ** 2).reshape(x0.shape[0], -1).sum(1)
```

The derivative in $t$ passes through the network's time features, which
is why the flow maps use slower ones than the rest of the chapter. A
feature $\sin(\omega t)$ has the derivative $\omega \cos(\omega t)$, so
each feature enters the target multiplied by its frequency. The other
networks see $t$ at frequencies up to 64 (:numref:`sec_diffusion-denoising`);
the flow maps pass `time_freqs=slow`, frequencies from $10^{-4}$ to one,
which the code below defines as `slow`. Large frequencies put large factors
into the derivative (code that embeds $1000\, t$ multiplies it by up to a
thousand), a source of instability that :citet:`Lu.Song.2025` report for
consistency models.

A flow map takes $\mathbf{x}_t$ to $\mathbf{x}_s$ in one jump,
$\mathbf{x}_s = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s,
t)$, so a few-step sampler cuts $[0, 1]$ into equal intervals and jumps
across each.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-1}
%%tab pytorch
@torch.inference_mode()
def flow_map_sample(u, x, steps, t_start=1.0):  #@save
    """Jump from x at t_start (1 is noise) to data at t = 0 in equal steps."""
    ts = torch.linspace(t_start, 0, steps + 1, device=x.device)
    for t, s in zip(ts[:-1], ts[1:]):
        x = x - (t - s) * u(x, s.expand(len(x)), t.expand(len(x)))
    return x
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-1}
%%tab jax
def flow_map_sample(u, x, steps, t_start=1.0):  #@save
    """Jump from x at t_start (1 is noise) to data at t = 0 in equal steps."""
    ts = jnp.linspace(t_start, 0, steps + 1)
    def jump(x, ts):
        t, s = ts
        n = x.shape[0]
        return x - (t - s) * u(x, jnp.full(n, s), jnp.full(n, t)), None
    return jax.lax.scan(jump, x, (ts[:-1], ts[1:]))[0]
```

With the velocity $\mathbf{u}_t$ in place of $\bar{\mathbf{u}}$, the same
function is Euler's method, so one sampler serves the teacher and the
student. The jumps go one way only: every loss of this section draws
$s \leq t$, so inverting a sample, as :numref:`sec_diffusion-samplers`
inverted the ODE, would need the map trained in the other direction too
(Exercise :exref:`ex_few-step-invert`).

For the image experiment at the end of the next section, the student
starts as a copy of the teacher's U-Net. The gap $t - s$ enters through
its own time embedding and a linear map initialized at zero, added to the
U-Net's embedding, so that at the start $\bar{\mathbf{u}}(\mathbf{x}, s, t)
= \hat{\mathbf{u}}(\mathbf{x}, t)$, the teacher's velocity.

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-2}
%%tab pytorch
class UNetFlowMap(nn.Module):  #@save
    """u(x, s, t): a velocity U-Net, plus a zero-initialized input for the
    gap t - s. The caller passes a copy of the teacher's U-Net."""
    def __init__(self, unet):
        super().__init__()
        # the teacher is an EMA copy with gradients off; the student trains
        self.unet = unet.requires_grad_(True)
        self.gap = nn.Linear(unet.emb_dim, unet.emb_dim)
        nn.init.zeros_(self.gap.weight), nn.init.zeros_(self.gap.bias)

    def forward(self, x, s, t):
        gap = self.gap(d2l.time_embedding(t - s, self.unet.emb_dim,
                                          self.unet.time_freqs))
        return self.unet(x, t, cond=gap)
```

```{.python .input #few-step-average-velocity-and-the-meanflow-identity-2}
%%tab jax
class UNetFlowMap(nnx.Module):  #@save
    """u(x, s, t): a velocity U-Net, plus a zero-initialized input for the
    gap t - s. The caller passes a copy of the teacher's U-Net."""
    def __init__(self, unet, *, rngs):
        self.unet = unet
        zeros = nnx.initializers.zeros
        self.gap = nnx.Linear(unet.emb_dim, unet.emb_dim, kernel_init=zeros,
                              bias_init=zeros, rngs=rngs)

    def __call__(self, x, s, t):
        gap = self.gap(d2l.time_embedding(t - s, self.unet.emb_dim,
                                          self.unet.time_freqs))
        return self.unet(x, t, cond=gap)
```

## Training Flow Maps

### The Three-Mode Mixture

Does the loss work, and does it matter where the reference velocity comes
from? The mixture can answer both, because its exact field tells us what a
perfect sampler prints. The teacher is a velocity model trained as in
:numref:`sec_diffusion-gaussian-paths`: a `TimeMLP`, which passes the point
and sine features of $t$ through layers of width 256, here trained for
20,000 steps.

```{.python .input #few-step-on-the-running-example}
%%tab pytorch
torch.manual_seed(1)
_, teacher, _ = d2l.train_regression(
    d2l.TimeMLP(256, 3).to(device), d2l.regression_loss(lin, 'velocity'),
    lambda: mix.sample(512), 20_000, cosine=True)
```

```{.python .input #few-step-on-the-running-example}
%%tab jax
_, teacher, _ = d2l.train_regression(
    d2l.TimeMLP(256, 3, rngs=nnx.Rngs(1)),
    d2l.regression_loss(lin, 'velocity'), lambda k: mix.sample(k, 512),
    20_000, jax.random.key(1), cosine=True)
```

Three students fill three cells of :numref:`tab_diffusion-flow-map-methods`.
All are `TimeMLP`s that see the start $t$ and the interval $t - s$,
trained like the teacher, twice each (seeds 2 and 3). The consistency
student draws $t$ uniformly, fixes $s = 0$ and uses the teacher's
velocity. The flow-map distillation student draws two uniform times, takes
the smaller as $s$, and uses the teacher's velocity too; the MeanFlow
student draws its times the same way and uses no teacher. The fourth cell,
consistency training, needs no student of its own: its two choices, one
time and the example's velocity, are each tested by another student.

What losses should we expect? A distilled student's target is a function
of the point and the times, so its loss can approach zero. MeanFlow's
target contains each example's own velocity twice, once as
$\mathbf{u}_{\textrm{ref}}$ and once as the direction of the tangent, and
its loss cannot fall below the variation of that target between examples
at the same point and times.

```{.python .input #few-step-on-the-running-example-1}
%%tab pytorch
class MLPFlowMap(d2l.TimeMLP):
    """u(x, s, t): a TimeMLP on the start t and the interval t - s."""
    def forward(self, x, s, t):
        return super().forward(x, t, t - s)

slow = (1e-4, 1.0)  # time features of a flow map, whose loss has a d/dt

def consistency_loss(u, x0):  # s = 0, the teacher's velocity
    t = torch.rand(len(x0), device=device)
    return d2l.flow_map_error(u, teacher, x0, torch.randn_like(x0),
                              torch.zeros_like(t), t).mean()

def distillation_loss(u, x0):  # s < t, the teacher's velocity
    a, b = torch.rand(2, len(x0), device=device)
    s, t = torch.minimum(a, b), torch.maximum(a, b)
    return d2l.flow_map_error(u, teacher, x0, torch.randn_like(x0), s,
                              t).mean()

def meanflow_loss(u, x0):  # s < t, no teacher
    a, b = torch.rand(2, len(x0), device=device)
    s, t = torch.minimum(a, b), torch.maximum(a, b)
    return d2l.flow_map_error(u, None, x0, torch.randn_like(x0), s, t).mean()

students = {}
for name, loss in (('consistency', consistency_loss),
                   ('distillation', distillation_loss),
                   ('MeanFlow', meanflow_loss)):
    for seed in (2, 3):
        torch.manual_seed(seed)
        _, students[name, seed], losses = d2l.train_regression(
            MLPFlowMap(256, 3, num_times=2, time_freqs=slow).to(device),
            loss, lambda: mix.sample(512), 20_000, cosine=True)
        print(f'{name:>12}, seed {seed}: mean loss of the last 1,000 '
              f'updates {losses[-1000:].mean():.3f}')
```

```{.python .input #few-step-on-the-running-example-1}
%%tab jax
class MLPFlowMap(d2l.TimeMLP):
    """u(x, s, t): a TimeMLP on the start t and the interval t - s."""
    def __call__(self, x, s, t):
        return super().__call__(x, t, t - s)

slow = (1e-4, 1.0)  # time features of a flow map, whose loss has a d/dt

def consistency_loss(u, x0, key):  # s = 0, the teacher's velocity
    k1, k2 = jax.random.split(key)
    t = jax.random.uniform(k1, (len(x0),))
    return d2l.flow_map_error(u, teacher, x0, jax.random.normal(k2, x0.shape),
                              jnp.zeros_like(t), t).mean()

def distillation_loss(u, x0, key):  # s < t, the teacher's velocity
    k1, k2 = jax.random.split(key)
    a, b = jax.random.uniform(k1, (2, len(x0)))
    s, t = jnp.minimum(a, b), jnp.maximum(a, b)
    return d2l.flow_map_error(u, teacher, x0, jax.random.normal(k2, x0.shape),
                              s, t).mean()

def meanflow_loss(u, x0, key):  # s < t, no teacher
    k1, k2 = jax.random.split(key)
    a, b = jax.random.uniform(k1, (2, len(x0)))
    s, t = jnp.minimum(a, b), jnp.maximum(a, b)
    return d2l.flow_map_error(u, None, x0, jax.random.normal(k2, x0.shape),
                              s, t).mean()

students = {}
for name, loss in (('consistency', consistency_loss),
                   ('distillation', distillation_loss),
                   ('MeanFlow', meanflow_loss)):
    for seed in (2, 3):
        _, students[name, seed], losses = d2l.train_regression(
            MLPFlowMap(256, 3, num_times=2, time_freqs=slow,
                       rngs=nnx.Rngs(seed)),
            loss, lambda k: mix.sample(k, 512), 20_000, jax.random.key(seed),
            cosine=True)
        print(f'{name:>12}, seed {seed}: mean loss of the last 1,000 '
              f'updates {losses[-1000:].mean():.3f}')
```

The distilled students end near zero, about 0.03 for the two-time student
and 0.075 for the consistency student. MeanFlow ends at 76 to 81 in both
seeds, about a thousand times the consistency student's loss. Most of
that may be the floor, the disagreement between examples that no network
can remove; the cell does not measure the floor, so the number says little
about the student. MeanFlow runs are judged by their samples, not by their
losses.

All models now start from the 20,000 points of the Euler experiment: the
teacher with fifty Euler steps, the consistency student with one jump or
two with re-noising at $t = 0.5$, the two-time students with one, two or
four jumps. A fresh draw from the mixture shows what a perfect sampler
would print. For each run a row gives the largest error in the three
weights, the strays, the samples farther than three standard deviations
from every mode center, and the spread, the mean squared distance of the
samples that do not stray to the center of their mode
(:numref:`sec_diffusion-denoising`). A student row holds its two runs,
seeds 2 and 3, separated by a slash.

```{.python .input #few-step-on-the-running-example-2}
%%tab pytorch
def renoised_sample(u, z):
    """Two evaluations of a one-time map: jump, re-noise to t = 0.5, jump."""
    x = d2l.flow_map_sample(u, z, 1)
    half = torch.full((len(z),), 0.5, device=device)
    return d2l.flow_map_sample(u, lin.sample(x, half, torch.randn_like(x)), 1,
                               t_start=0.5)

def report(name, xs):
    """Each run's largest weight error, stray fraction and spread."""
    stats, w_true = [mix.summary(x) for x in xs], mix.weights.tolist()
    err = ' / '.join(
        f'{max(abs(w - w0) for w, w0 in zip(s["weights"], w_true)):.3f}'
        for s in stats)
    stray = ' / '.join(f'{s["stray"]:.1%}' for s in stats)
    spread = ' / '.join(f'{s["spread"]:.3f}' for s in stats)
    print(f'{name:>24}: weight error {err}, stray {stray}, spread {spread}')

torch.manual_seed(4)
samples = {'data': [mix.sample(len(z))],
           'teacher, 50 steps': [d2l.flow_map_sample(
               lambda x, s, t: teacher(x, t), z, 50)]}
for (name, seed), u in students.items():
    if name == 'consistency':  # a one-time map re-noises for a second step
        new = {'1': d2l.flow_map_sample(u, z, 1),
               '2 re-noised': renoised_sample(u, z)}
    else:
        new = {str(k): d2l.flow_map_sample(u, z, k) for k in (1, 2, 4)}
    for k, x in new.items():
        samples.setdefault(f'{name}, {k}', []).append(x)
for name, xs in samples.items():
    report(name, xs)
names = ['teacher, 50 steps', 'consistency, 1', 'distillation, 1',
         'MeanFlow, 1']
d2l.scatter_panels([samples[k][0][:2000] for k in names], names,
                   data=mix.sample(2000), size=2.4);
```

```{.python .input #few-step-on-the-running-example-2}
%%tab jax
def renoised_sample(u, z, key):
    """Two evaluations of a one-time map: jump, re-noise to t = 0.5, jump."""
    x = d2l.flow_map_sample(u, z, 1)
    half = jnp.full((len(z),), 0.5)
    return d2l.flow_map_sample(u, lin.sample(x, half, jax.random.normal(
        key, x.shape)), 1, t_start=0.5)

def report(name, xs):
    """Each run's largest weight error, stray fraction and spread."""
    stats, w_true = [mix.summary(x) for x in xs], mix.weights.tolist()
    err = ' / '.join(
        f'{max(abs(w - w0) for w, w0 in zip(s["weights"], w_true)):.3f}'
        for s in stats)
    stray = ' / '.join(f'{s["stray"]:.1%}' for s in stats)
    spread = ' / '.join(f'{s["spread"]:.3f}' for s in stats)
    print(f'{name:>24}: weight error {err}, stray {stray}, spread {spread}')

samples = {'data': [mix.sample(jax.random.key(3), len(z))],
           'teacher, 50 steps': [d2l.flow_map_sample(
               lambda x, s, t: teacher(x, t), z, 50)]}
for (name, seed), u in students.items():
    if name == 'consistency':  # a one-time map re-noises for a second step
        key = jax.random.fold_in(jax.random.key(4), seed)
        new = {'1': d2l.flow_map_sample(u, z, 1),
               '2 re-noised': renoised_sample(u, z, key)}
    else:
        new = {str(k): d2l.flow_map_sample(u, z, k) for k in (1, 2, 4)}
    for k, x in new.items():
        samples.setdefault(f'{name}, {k}', []).append(x)
for name, xs in samples.items():
    report(name, xs)
names = ['teacher, 50 steps', 'consistency, 1', 'distillation, 1',
         'MeanFlow, 1']
d2l.scatter_panels([samples[k][0][:2000] for k in names], names,
                   data=mix.sample(jax.random.key(5), 2000), size=2.4);
```

After one evaluation every student puts its samples on the three modes
with weights within 0.007 of 0.5/0.3/0.2. A fresh draw of 20,000 points
misses by 0.004 to 0.005, so one jump gets the weights right to within
about the noise of sampling, and the distilled two-time student keeps them
that close at two and four jumps. The fresh draw strays about 1% of the time,
the tail of each Gaussian beyond three standard deviations, and the teacher
with fifty steps about 1.5%. Both distilled students stray about 2% after
one evaluation and keep the data's spread, 0.45 to 0.46 against 0.47. A
second evaluation brings their strays to the teacher's level or below,
whether it re-noises (1.2% to 1.5%) or jumps again (1.0% to 1.1%). On the
mixture re-noising costs nothing these numbers can see; what the second
jump keeps is the pairing of each sample with its noise. MeanFlow strays
about 4% after one jump, and its panel shows a second defect: the modes
are too tight, with a spread of 0.31 to 0.34 after one jump and 0.37 to
0.38 after two and four. That is also why its four jumps stray less often
than the data themselves.

Two runs per student are few. Changing the seeds of the training cell to
2 through 7 gives six runs of each two-time student, and their weight
errors tell two stories. One jump stays within 0.007 in every run, and
the distilled student stays within 0.006 at two and four jumps as well.
After four MeanFlow jumps, however, one or two runs in six end 0.013 to
0.023 off, several times the miss of a fresh draw. MeanFlow's weights
after several jumps thus depend on the training run, and a single run is
no guide to them; the distilled student's do not.

:begin_tab:`pytorch`
In this tab both MeanFlow runs of the table are among the steady ones:
0.005 off after four jumps.
:end_tab:

:begin_tab:`jax`
In this tab the MeanFlow run with seed 3 is one of the drifting ones: its
weights are 0.012 off after two jumps and 0.020 after four, so about 2% of
its samples sit on the wrong mode, while its one jump stays within 0.005.
A Python loop over the same jumps gives the same weights as the compiled
sampler, so the drift belongs to the training run, not to the sampler.
:end_tab:

The comparison locates the defect. The two-time students share the
network, the time pairs and the updates and differ only in the velocity
inside the target. Their spreads differ by more than 0.1 after one jump
and by 0.08 to 0.1 after two and four, many times the difference between
seeds (at most 0.015). In this run, then, the contraction comes from
training on the examples' velocities. Why a noisy target ends in a map
that is too tight rather than a noisy one, the run does not say. A student
that has learned only part of the correction term of
:eqref:`eq_diffusion-meanflow` is partly an Euler step, and Euler strides
contract the modes in just this way (:numref:`sec_diffusion-samplers`), but
that is a plausible reading, not a result. On the mixture a teacher is
worth having: training from data alone saves the teacher's run but leaves
the student a far noisier target to average.

### Fashion-MNIST

The image teacher is an unconditional velocity U-Net, `UNet(ch=64)` with
about 1.2 million parameters, trained for 10,000 updates with the slow
time features, since the student starts as a copy of it.
`train_regression` returns a moving average of the weights along with the
trained network, and that averaged network is the teacher.

```{.python .input #few-step-on-images}
%%tab pytorch
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()

def batch_fn():
    return X[torch.randint(0, len(X), (128,), device=X.device)]

torch.manual_seed(0)
_, teacher, _ = d2l.train_regression(  # the student copies these time features
    d2l.UNet(ch=64, time_freqs=slow).to(X.device),
    d2l.regression_loss(lin, 'velocity'), batch_fn, 10_000)
```

```{.python .input #few-step-on-images}
%%tab jax
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()

def batch_fn(key):
    return X[jax.random.randint(key, (128,), 0, len(X))]

_, teacher, _ = d2l.train_regression(  # the student copies these time features
    d2l.UNet(ch=64, time_freqs=slow, rngs=nnx.Rngs(0)),
    d2l.regression_loss(lin, 'velocity'), batch_fn, 10_000, jax.random.key(0))
```

We score 1,000 samples by the Fréchet distance (FD) of
:numref:`sec_diffusion-images` against 1,000 test images: the distance
between Gaussians fitted to the features that a Fashion-MNIST classifier
extracts from the two sets. It is read against the floor of two disjoint
sets of 1,000 test images.

```{.python .input #few-step-on-images-1}
%%tab pytorch
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floor = d2l.fid_score(features(X_test[1000:2000]), f_test)
print(f'floor, 1,000 real images against 1,000 others: FD {floor:.2f}')
```

```{.python .input #few-step-on-images-1}
%%tab jax
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floor = d2l.fid_score(features(X_test[1000:2000]), f_test)
print(f'floor, 1,000 real images against 1,000 others: FD {floor:.2f}')
```

The student is `UNetFlowMap`, defined above, on a copy of the teacher,
trained with the teacher's velocity and both times free. Half of the pairs
have $s = t$, where the target is the teacher's velocity itself: the
student starts as the teacher, and these pairs are meant to keep it from
forgetting the velocity while it learns to jump. The other half draw
$s < t$ as on the mixture. Starting from trained weights, the student
learns at a rate of $2 \cdot 10^{-4}$ for 5,000 updates of 128 images,
half the teacher's 10,000.

```{.python .input #few-step-on-images-2}
%%tab pytorch
def student_loss(u, x0):
    a, b = torch.rand(2, len(x0), device=x0.device)
    s, t = torch.minimum(a, b), torch.maximum(a, b)
    # half the pairs have s = t: the teacher's velocity as target
    s = torch.where(torch.rand(len(x0), device=x0.device) < 0.5, t, s)
    return d2l.flow_map_error(u, teacher, x0, torch.randn_like(x0), s,
                              t).mean()

torch.manual_seed(2)
_, student, _ = d2l.train_regression(  # jvp through GroupNorm won't compile
    d2l.UNetFlowMap(copy.deepcopy(teacher)).to(X.device), student_loss,
    batch_fn, 5_000,
    lr=2e-4, ema_half_life=500, compiled=False)
```

```{.python .input #few-step-on-images-2}
%%tab jax
def student_loss(u, x0, key):
    k1, k2, k3 = jax.random.split(key, 3)
    a, b = jax.random.uniform(k1, (2, len(x0)))
    s, t = jnp.minimum(a, b), jnp.maximum(a, b)
    # half the pairs have s = t: the teacher's velocity as target
    s = jnp.where(jax.random.uniform(k2, (len(x0),)) < 0.5, t, s)
    return d2l.flow_map_error(u, teacher, x0, jax.random.normal(k3, x0.shape),
                              s, t).mean()

_, student, _ = d2l.train_regression(
    d2l.UNetFlowMap(nnx.clone(teacher), rngs=nnx.Rngs(2)), student_loss,
    batch_fn, 5_000, jax.random.key(2), lr=2e-4, ema_half_life=500)
```

From the same 1,000 noise images, the teacher takes 1 to 50 Euler steps
and the student 1, 2 or 4 jumps, and we plot FD against the number of
network evaluations. The teacher's single step should land on the mean
image, the pixel version of the mixture's failure, so its FD there should
dwarf every other point.

```{.python .input #few-step-on-images-3}
%%tab pytorch
torch.manual_seed(3)
noise = torch.randn(1000, 1, 28, 28, device=X.device)
velocity = lambda x, s, t: teacher(x, t)
ns_teacher, ns_student = (1, 2, 4, 10, 20, 50), (1, 2, 4)
images = {('teacher', n): d2l.flow_map_sample(velocity, noise, n)
          for n in ns_teacher}
images |= {('student', n): d2l.flow_map_sample(student, noise, n)
           for n in ns_student}
fd = {key: d2l.fid_score(features(x), f_test) for key, x in images.items()}
for (name, n), v in fd.items():
    print(f'{name}, {n:2d} evaluations: FD {v:6.2f}')
d2l.plot([ns_teacher, ns_student, [1, 50]],
         [[fd['teacher', n] for n in ns_teacher],
          [fd['student', n] for n in ns_student], [floor, floor]],
         'network evaluations', 'FD',
         legend=['teacher, Euler', 'student, jumps', 'floor'],
         xscale='log', yscale='log', fmts=('o-', 's-', 'k:'))
```

```{.python .input #few-step-on-images-3}
%%tab jax
noise = jax.random.normal(jax.random.key(3), (1000, 28, 28, 1))
velocity = lambda x, s, t: teacher(x, t)
ns_teacher, ns_student = (1, 2, 4, 10, 20, 50), (1, 2, 4)
images = {('teacher', n): d2l.flow_map_sample(velocity, noise, n)
          for n in ns_teacher}
images |= {('student', n): d2l.flow_map_sample(student, noise, n)
           for n in ns_student}
fd = {key: d2l.fid_score(features(x), f_test) for key, x in images.items()}
for (name, n), v in fd.items():
    print(f'{name}, {n:2d} evaluations: FD {v:6.2f}')
d2l.plot([ns_teacher, ns_student, [1, 50]],
         [[fd['teacher', n] for n in ns_teacher],
          [fd['student', n] for n in ns_student], [floor, floor]],
         'network evaluations', 'FD',
         legend=['teacher, Euler', 'student, jumps', 'floor'],
         xscale='log', yscale='log', fmts=('o-', 's-', 'k:'))
```

The teacher's FD falls from about 150 to 170 at one Euler step and about
80 to 85 at two, to about 7 at ten and 2.6 to 2.7 at fifty, against a
floor of about 0.8. The student lies far below this curve at small budgets: one
jump scores 11 to 12, between four and five times the teacher's
fifty-step FD, two jumps nearly twice it, and four within about 1.5 times.
Read across the plot, two evaluations of the student are worth about
fifteen Euler steps of the teacher, and one about seven.

A distilled student can at best reproduce its teacher's ODE, so its target
is the FD of the teacher's exact ODE, which fifty Euler steps approach from
above, not the floor. How much of the remaining gap is a matter of
training, our run does not tell. The student saw 640,000 images; published
students are trained on far more and come much closer to their teachers in
one step :cite:`Song.Dhariwal.Chen.ea.2023,Lu.Song.2025`.

The rows below are drawn from the same eight noise images: the teacher
with fifty and with two Euler steps, and the student with one and with two
jumps.

```{.python .input #few-step-on-images-4}
%%tab pytorch
rows = [('teacher', 50), ('teacher', 2), ('student', 1), ('student', 2)]
d2l.show_gray(torch.cat([images[r][:8].clamp(-1, 1) for r in rows]), 4, 8);
```

```{.python .input #few-step-on-images-4}
%%tab jax
rows = [('teacher', 50), ('teacher', 2), ('student', 1), ('student', 2)]
d2l.show_gray(jnp.concatenate([images[r][:8].clip(-1, 1) for r in rows]),
              4, 8);
```

The rows match column by column. Two Euler steps of the teacher draw grey
averages of garments, the mixture's failure in pixels, and in some columns
not even the kind of garment that fifty steps draw. The student draws the
teacher's garment from almost every noise image, with harder edges and
some blotches after one jump and close to the teacher's after two.

## Distribution Matching

A flow map is bounded by the trajectories it learns from: a distilled
student by its teacher's ODE, and a student trained from data alone by how
well it learns the exact field, which on the mixture fell short in the
spread.

The other route trains a one-step generator
$g_{\boldsymbol\theta}(\mathbf{z})$ to match a distribution instead of a
trajectory. Fitted by squared error to a teacher's samples, such a
generator draws averages: a network that cannot reproduce the teacher's
long computation in one pass sees several noise vectors as alike, and
squared error rewards the mean of their images (:numref:`sec_gan_beyond`).
That section also gave two remedies: a critic against real images
:cite:`Sauer.Lorenz.Blattmann.ea.2023`, and distribution matching
distillation :cite:`Yin.Gharbi.Zhang.ea.2024`, which needs no critic. The
latter noises the generator's samples, $\mathbf{x}_t = (1 - t)\,
g_{\boldsymbol\theta}(\mathbf{z}) + t\, \boldsymbol\epsilon$, and lowers
the Kullback-Leibler divergence of their law $q_{\boldsymbol\theta, t}$
from the noised data $p_t$. Its gradient is

$$\nabla_{\boldsymbol\theta} D_{\textrm{KL}}(q_{\boldsymbol\theta, t} \,\|\, p_t) = \mathbb{E}\Big[\big(\nabla \log q_{\boldsymbol\theta, t}(\mathbf{x}_t) - \nabla \log p_t(\mathbf{x}_t)\big)^\top \frac{\partial \mathbf{x}_t}{\partial \boldsymbol\theta}\Big].$$
:eqlabel:`eq_diffusion-dmd-gradient`

The gradient is the student's score minus the teacher's, so a descent step
moves each noised sample toward where the data have more mass than the
student puts there. The teacher supplies $\nabla \log p_t$, since on the
linear path a velocity is a score in disguise
:eqref:`eq_diffusion-linear-dictionary`, and a second diffusion model,
trained alongside on the student's own samples, supplies
$\nabla \log q_{\boldsymbol\theta, t}$. Such a student is bounded by
its teacher's distribution rather than its trajectories; only a loss with
real images in it, a critic as in adversarial distillation or added to
distribution matching :cite:`Yin.Gharbi.Park.ea.2024`, lets a student
surpass its teacher (:numref:`sec_gan-further-reading`). Neither variant
is trained along trajectories of the ODE, so the generator's sample for a
given noise vector need not be the teacher's, and inversion and editing
through the ODE no longer apply.

Both routes start from the same object: a continuous path along which a
velocity or a score is defined. Neither exists when $\mathbf{x}$ is a
sequence of tokens, the subject of :numref:`sec_diffusion-discrete`.

## Summary

One Euler step from noise lands every sample on the mean of the data,
because at pure noise the velocity points at the mean from every start.
Straightening the paths by reflow made one step possible, but the original
work still distilled the reflowed model for its best one-step results. A
flow map replaces the velocity by the jump itself, written through the
average velocity over an interval. Differentiating it along the trajectory
gives the MeanFlow identity, a regression target that needs one
Jacobian-vector product, a stop-gradient, and either a teacher's velocity
or each example's own. On the mixture, students distilled from a teacher
match it in one or two evaluations, while MeanFlow gets the mode weights
right in one jump but draws the modes too tight; on Fashion-MNIST a student scores
four to five times the teacher's fifty-step FD in one jump and within
about 1.5 times in four. Matching distributions, with a critic or a
difference of scores, is the other route to one step.

## Exercises

1. **One and two Euler steps.**
    1. A single step of the deterministic sampler from pure noise to $t = 0$
       returns $\mathbb{E}[\mathbf{x}_0]$ from every start, whatever the
       path (:numref:`sec_diffusion-integration`). Does the same hold for
       one Euler step in $t$, $\mathbf{x} - \mathbf{u}_1(\mathbf{x})$, on the
       cosine path $\alpha_t = \cos\frac{\pi t}{2}$,
       $\sigma_t = \sin\frac{\pi t}{2}$? What does that step return, and
       why does an Euler step in $t$ depend on the path when a step in
       $\bar\sigma$ does not?
    1. Let the data be the two points $\pm a$ in one dimension, with equal
       probability, on the linear path. Show that
       $\mathbb{E}[x_0 \mid x_t = x] = a \tanh\big(a (1 - t) x / t^2\big)$
       and that two Euler steps of length $\tfrac12$ map a start $x$ to
       $a \tanh(a x)$. :numref:`subsec_mdl-fm-two-point` has the exact flow
       of this example.
    1. For $a = 2$, what fraction of the two-step samples lies within $0.1$
       of a data point? Where does the exact flow map $\Phi_{1 \to 0}$ send
       $x$?
   :exlabel:`ex_few-step-one-step-mean`
1. [code] **Reflow on the mixture.** ● Reflow :cite:`Liu.Gong.Liu.2022`
   retrains a velocity model on pairs of a noise vector and the sample that
   the current model produces from it.
    1. Draw 20,000 noise vectors, integrate the teacher of this section with
       fifty Euler steps, and train a new `TimeMLP` on the pairs with the
       velocity loss and the teacher's settings.
    1. Report the weights and strays of the new model at 1, 2, 4 and 10
       Euler steps, and the mean distance of its fifty-step trajectories
       from their chords, next to the same quantities for the teacher.
    1. Which of the teacher's errors does reflow remove, which does it
       inherit, and which does it add? What would a second round do to each?
   :exlabel:`ex_few-step-reflow`
1. [extended] **Composition instead of a derivative.** ●● Progressive
   distillation :cite:`Salimans.Ho.2022` trains a student so that one of
   its steps reproduces two steps of the teacher, then makes the student
   the teacher and repeats. A shortcut model
   :cite:`Frans.Hafner.Levine.ea.2025` conditions a velocity model on the
   step size $d$ and trains it from scratch with two losses: flow matching
   at $d = 0$, and self-consistency, which asks one step of size $2d$ to
   equal two steps of size $d$ computed by the model itself and held fixed.
    1. Which identity of the flow map does each method enforce, and why
       does neither need a Jacobian-vector product?
    1. On the mixture, start from 64 Euler steps of the teacher of this
       section and halve the number of steps six times, training each
       student for a fifth of the teacher's updates. Report the strays
       after each round. Where do the errors of the rounds come from?
    1. Implement a shortcut model on the mixture with step sizes $2^{-k}$,
       $k = 0, \ldots, 6$. Compare its strays and spread at one, two and
       four steps with those of the progressive student and of the MeanFlow
       student of this section at equal total training steps.
   :exlabel:`ex_few-step-composition`
1. **The Lagrangian loss.** ● :citet:`Boffi.Albergo.VandenEijnden.2025`
   train flow maps with the Lagrangian identity.
    1. Substitute :eqref:`eq_diffusion-flow-map` into $\partial_s
       \Phi_{t \to s}(\mathbf{x}) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}))$
       and derive an identity for $\bar{\mathbf{u}}$ that involves the
       velocity at the output of the jump.
    1. Why can the conditional velocity $\boldsymbol\epsilon - \mathbf{x}_0$
       not serve in this identity, so that a velocity model is needed? Show
       that the flow map's own velocity $\bar{\mathbf{u}}(\cdot, s, s)$ can
       serve, so that one network trains itself.
    1. [code] Train such a student on the mixture and compare it with the
       MeanFlow student in strays and in training cost per step.
   :exlabel:`ex_few-step-lagrangian`
1. [code] **The stop-gradient and the weighting.** Two details of
   :eqref:`eq_diffusion-flow-map-loss` decide what its minimizer is.
    1. Copy `flow_map_error` into a cell and remove the stop-gradient on
       the target. Train the MeanFlow student on the mixture with this loss
       and report its one-step weights, strays and spread. Write out the
       gradient of the loss with the target differentiated, and identify
       the term that the stop-gradient removes. What can the network change
       to lower this loss without satisfying :eqref:`eq_diffusion-meanflow`?
    1. Song et al. compute the target with a moving average of the
       student's weights instead. Implement this with `d2l.EMA` and compare.
    1. MeanFlow's ImageNet recipe :cite:`Geng.Deng.Bai.ea.2025` divides
       each example's squared error $e$ by $e + c$, held fixed, with a small
       $c$, so that the gradient is that of $\log(e + c)$. Squared error is
       minimized by the conditional mean of the target; near which value
       does the weighted loss put its minimizer? Without a teacher the
       targets at a noisy point of the mixture spread over the three modes:
       predict the mode weights of one-jump samples, then train the MeanFlow
       student with this weight and $c = 10^{-3}$ and check.
    1. On a $28 \times 28$ image, $e$ sums 784 squared errors. Argue that
       it then varies far less between examples at the same $(s, t)$ than
       in two dimensions, so that the weight acts mostly as a weighting
       across times. What does the experiment on the mixture tell you about
       the recipe on images, and what does it not?
   :exlabel:`ex_few-step-stop-gradient`
1. [code] **Which times to train on, and on what grid.** The MeanFlow student drew every pair
   with $s < t$ from two uniform times, the image student half of its pairs
   with $s = t$, and consistency distillation as published uses a grid.
    1. Train the MeanFlow student on the mixture with a share of $0.25$,
       $0.5$, $0.75$ and $1$ of the pairs having $s < t$, and report the
       strays and spread at one and two jumps. Which share serves one jump
       best?
    1. Replace the uniform times by a logit-normal distribution of $t$ with
       mean $-0.4$ and standard deviation $1$, which concentrates the times
       in the middle of the path. What changes, and why?
    1. Song et al. :cite:`Song.Dhariwal.Chen.ea.2023` use a grid of
       $N = 18$ times, one Heun step of the teacher from $t_n$ to
       $t_{n-1}$, and a target held fixed at the earlier point. Write the
       student as $f(\mathbf{x}, t) = \mathbf{x} - t\,
       \bar{\mathbf{u}}(\mathbf{x}, 0, t)$ and show that the difference of
       its outputs at the two points, divided by $t_n - t_{n-1}$, tends to
       the Eulerian derivative :eqref:`eq_diffusion-eulerian` at $s = 0$ as
       $N$ grows.
    1. Implement this discrete loss without a Jacobian-vector product,
       train it on the mixture with the teacher of this section for $N = 8$
       and $N = 64$, and compare its one-step strays with those of the
       continuous student. Does a finer grid help, and why?
   :exlabel:`ex_few-step-time-pairs`
1. [code] **Inverting a student.** The exact flow map is a bijection, so
   $\Phi_{0 \to 1}$ undoes $\Phi_{1 \to 0}$.
    1. For the Gaussian data with $m = 2$ and $\varsigma = 0.5$, write
       $\Phi_{0 \to 1}$ and the average velocity $\bar{u}(x, 1, 0)$ that
       :eqref:`eq_diffusion-flow-map` assigns to this backward jump. Does the
       derivation of :eqref:`eq_diffusion-meanflow` need $s \leq t$?
    1. Can the students of this section map a sample back to its noise?
       Which pairs $(s, t)$ did their losses never see?
    1. Train the MeanFlow student on the mixture with pairs drawn in both
       orders. Write a backward version of `flow_map_sample`, invert 2,000
       of the student's one-jump samples with one and with four backward
       jumps, and report the root-mean-square distance to the noise they
       came from. Compare with inverting the teacher by fifty Euler steps
       run from $t = 0$ to $t = 1$.
   :exlabel:`ex_few-step-invert`
1. [code] **A guided student.** ● Classifier-free guidance, as in
   :numref:`subsec_diffusion-images-guided`, evaluates the network twice
   per step. :citet:`Meng.Rombach.Gao.ea.2023` distil the guided velocity
   into one network, which halves the cost of every step before the number
   of steps is reduced. Train the class-conditional U-Net of
   :numref:`sec_diffusion-images` with `time_freqs=slow`, as this section's
   image teacher was trained, and distil it into a two-time student whose reference
   velocity $\mathbf{u}_{\textrm{ref}}$ is the guided velocity
   $\tilde{\mathbf{u}}$, `d2l.cfg(teacher, c, 2.0)`, for random labels $c$,
   with the label as an input of the student.
    1. Compare the FD and the agreement of the predicted class with the
       requested one for the teacher with fifty guided Euler steps and the
       student with one, two and four jumps.
    1. How many network evaluations does each sample cost in each case?
    1. Meng et al. also give the student the guidance strength as an input.
       How would you draw the strengths during training, and what does the
       student then offer that a student distilled at $\gamma = 2$ does not?
   :exlabel:`ex_few-step-guided-student`
1. [code] **Quality against cost.** Measure the wall-clock time per 1,000
   images and the FD of the teacher of this section with 1 to 50 Euler
   steps, with Heun's method on the crowded grid of
   :numref:`sec_diffusion-samplers`, and of the student with 1 to 4 jumps.
    1. Plot FD against time. At which FD do the curves cross?
    1. The student cost extra training. How many images must be generated
       before that training is repaid at the FD of two jumps?
   :exlabel:`ex_few-step-cost-curve`
1. **Matching noised distributions.** ● Distribution matching distillation
   :cite:`Yin.Gharbi.Zhang.ea.2024` trains a one-step generator
   $\mathbf{x} = g_{\boldsymbol\theta}(\mathbf{z})$ along the gradient
   :eqref:`eq_diffusion-dmd-gradient`, where $q_{\boldsymbol\theta, t}$ is
   the law of the noised samples $\mathbf{x}_t$.
    1. Derive :eqref:`eq_diffusion-dmd-gradient`. Why does the term from
       the dependence of $q_{\boldsymbol\theta, t}$ itself on
       $\boldsymbol\theta$ vanish?
    1. In one dimension, let the generator be $m + b z$ with $b > 0$ and
       the data $\mathcal{N}(\mu, v)$. Write
       $D_{\textrm{KL}}(q_t \,\|\, p_t)$ on the linear path in closed form,
       and show that $m = \mu$, $b^2 = v$ is its only minimizer at every
       $t < 1$.
    1. Let the data be $\tfrac12 \mathcal{N}(-2, 0.25) + \tfrac12 \mathcal{N}(2, 0.25)$.
       Evaluate $D_{\textrm{KL}}(q_t \,\|\, p_t)$ by quadrature on a grid of
       $(m, b)$ at $t \in \{0.1, 0.3, 0.5, 0.9\}$, and locate the minimizer
       at each $t$. Why is the minimum near $\log 2$ at small $t$? Relate the
       result to mode dropping in GANs (:numref:`subsec_gan_mode_dropping`)
       and to the merging of modes under noise (Exercise
       :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` of :numref:`sec_mdl-fokker-planck-probability-flow`).
       What does a weighting of the levels toward large $t$ gain for a
       Gaussian generator, and what can no weighting gain for it?
   :exlabel:`ex_few-step-matching-noised`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.8]{.kicker}

Few-Step Generation<br>
**one Euler step lands on the mean · reflow · flow maps · consistency models · the MeanFlow identity · three students on the mixture · a student on Fashion-MNIST · matching distributions**
:::
:::

::: {.slide title="Fifty Network Calls per Sample, and One Long Step Lands on the Mean"}
[The problem]{.kicker}

- Good samplers need about **50** network evaluations per sample; a GAN
  generator needs **1**.
- One Euler step of the probability-flow ODE, even with the **exact**
  velocity, sends every sample to the data mean, where no data lie.

::: {.d2l-note}
What must a network learn so that one to four evaluations suffice, without
a critic (a second network that judges samples, as in a GAN)?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain why one Euler step returns the data mean, and what two steps
  leave;
- write a flow map as a jump by the average velocity;
- derive the MeanFlow identity and turn it into a regression;
- place consistency models, flow-map distillation and MeanFlow in one table;
- read FD against network evaluations for a teacher and its student.

**Running examples:** the three-mode mixture (weights 0.5/0.3/0.2, spread
0.47) with its exact field; a Fashion-MNIST U-Net teacher.
**Stray:** a sample farther than 1.5 (three standard deviations) from
every mode center. **Spread:** mean squared distance of the other samples
to their center.
**Idea:** learn the ODE's solution, not its right-hand side.
:::

::: {.slide title="Few-Step Generation Learns Where Trajectories Arrive"}
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
| Likelihood, Weighting and the Design Space | which choices matter |
| Latent Diffusion and Transformers | a compressed code, a transformer |
| **Few-Step Generation** | **one to four evaluations** |
| Discrete Diffusion | noise for tokens |
:::

::: {.col .narrow}
- **Uses:** the linear path $\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol\epsilon$
  (data at $t = 0$), its velocity $\mathbf{u}_t$, Euler sampling, FD.
- **Skips:** adversarial distillation in detail; guidance distillation.
:::
:::
:::

::: {.slide title="One Euler Step Sends Every Sample to the Mean, Where No Data Lie"}
@!few-step-the-average-where-the-data-are-not

Exact field, 20,000 starts. One step: all on the mean $(-0.5, -0.7)$. Two:
about a quarter stray, weights about 0.64/0.22/0.14. Four: about 6.5%
stray; ten: about 2% (the data: about 1%).
:::

::: {.slide title="At t = 1 the Velocity Points at the Data Mean from Every Start"}
On the linear path $\mathbf{x}_1 = \boldsymbol\epsilon$, independent of
$\mathbf{x}_0$, so

$$\mathbf{u}_1(\mathbf{x}) = \mathbb{E}\big[\boldsymbol\epsilon - \mathbf{x}_0 \,\big|\, \mathbf{x}_1 = \mathbf{x}\big] = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$$

One step of length 1: $\mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbb{E}[\mathbf{x}_0]$
for every start.

::: {.d2l-note}
One error, the **bend**: an Euler step follows the tangent of a curved
trajectory. At $t = 1$ the tangent is worst, aiming every start at the
same point; later steps aim better but still cut across the curve.
:::
:::

::: {.slide title="Check: Why Does the Heaviest Mode Gain at Two Steps?"}
[Check]{.kicker}

Two Euler steps give weights about 0.64/0.22/0.14 instead of 0.5/0.3/0.2.
Why does the first mode gain?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The first half-step aims at the mean $(-0.5, -0.7)$, which
lies nearest to the heaviest mode; the second step then sends more points
there.
:::
:::
:::

::: {.slide title="Reflow Straightens the Trajectories but Inherits the Model's Errors"}
- Noise–data segments cross; the field averages them, and trajectories,
  which cannot cross, bend.
- **Reflow** retrains on pairs of noise and the model's own sample: chords
  of trajectories that never cross. With the exact field: same
  distribution, no higher transport cost.
- Two moons: one reflowed step scores like 32 original steps, blur
  included.

::: {.d2l-note}
A trained model's pairs end at its samples, so its errors carry over; each
round is a training run. The original work's best one-step results still
distilled the reflowed model.
:::

[Liu et al., 2022]{.small}
:::

::: {.slide title="A Flow Map Says Where a Trajectory Arrives"}
::: {.cols .vc}
::: {.col}
$\Phi_{t \to s}(\mathbf{x})$: run the ODE from $\mathbf{x}$ at time $t$ back
to $s \leq t$. The sampler we want is $\Phi_{1 \to 0}$.

$$\Phi_{t \to s}(\mathbf{x}_t) = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t)$$

$\bar{\mathbf{u}}$: the **average velocity** on $[s, t]$. Since
$\bar{\mathbf{u}}(\mathbf{x}, t, t) = \mathbf{u}_t(\mathbf{x})$, it is a
velocity model with one more input.
:::

::: {.col .fig .big}
![](../img/mdl-diffusion-flow-map.svg)
:::
:::
:::

::: {.slide title="Gaussian Data: The Average Velocity Removes Only Half the Noise"}
Data $\mathcal{N}(m, \varsigma^2)$; $x_t$ has std
$k_t = \sqrt{(1-t)^2 \varsigma^2 + t^2}$, and the flow keeps each point's
standardized position:

$$\Phi_{t \to s}(x) = (1 - s)\, m + \tfrac{k_s}{k_t}\big(x - (1 - t)\, m\big)$$

| $m = 2$, $\varsigma = 0.5$ | from $t = 1$ to $0$ |
|:--|:--|
| exact jump $\Phi_{1 \to 0}(x)$ | $2 + 0.5\, x$ |
| average velocity $\bar{u}(x, 0, 1)$ | $0.5\, x - 2$ |
| Euler, velocity $u_1(x) = x - 2$ | $2$ for every $x$ |

::: {.d2l-note}
The data keep half of the noise's spread (std 0.5), so the exact jump
removes only half of it; Euler removes all of it.
:::
:::

::: {.slide title="Three Identities Pin Down the Flow Map, at Different Costs"}
For $s \leq r \leq t$ and the trajectory $\mathbf{x}_\tau$ through
$\mathbf{x}$ at time $t$:

| identity | statement | a loss needs |
|:--|:--|:--|
| compositional | $\Phi_{r \to s}(\Phi_{t \to r}(\mathbf{x})) = \Phi_{t \to s}(\mathbf{x})$ | the model at $r$; no derivative |
| Lagrangian | $\partial_s \Phi_{t \to s}(\mathbf{x}) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}))$ | a velocity at the output |
| Eulerian | $\tfrac{d}{dt} \Phi_{t \to s}(\mathbf{x}_t) = \mathbf{0}$ | one Jacobian-vector product |

::: {.d2l-note}
With the condition at $s = t$, each one determines the map. Consistency
models and MeanFlow are Eulerian: a target from one pair of times, with no
intermediate time and no step size.
:::

[Boffi et al., 2025]{.small}
:::

::: {.slide title="A Consistency Model Learns Only the Jump to the Data"}
$f(\mathbf{x}, t) = \Phi_{t \to 0}(\mathbf{x})$, trained so that
$\tfrac{d}{dt} f(\mathbf{x}_t, t) = \mathbf{0}$ with $f(\mathbf{x}, 0) = \mathbf{x}$:
the Eulerian identity at $s = 0$.

- **Distillation:** a teacher's ODE step joins neighbouring grid times;
  the outputs there must agree (target: a moving average of the student).
- **Training:** the example's velocity $\boldsymbol\epsilon - \mathbf{x}_0$
  replaces the teacher.
- Two evaluations need **re-noising**:
  $\mathbf{x}_{0.5} = 0.5\, \mathbf{x}_0 + 0.5\, \boldsymbol\epsilon'$.

[Song et al., 2023]{.small}
:::

::: {.slide title="Differentiating the Integral Gives the MeanFlow Identity"}
**I.** $(t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t) = \int_s^t \mathbf{u}_\tau(\mathbf{x}_\tau)\, d\tau$

**II.** $\tfrac{d}{dt}$ along the trajectory, $s$ fixed:
$\bar{\mathbf{u}} + (t - s)\, \tfrac{d}{dt}\bar{\mathbf{u}} = \mathbf{u}_t(\mathbf{x}_t)$

**III.**
$$\boxed{\bar{\mathbf{u}}(\mathbf{x}_t, s, t) = \mathbf{u}_t(\mathbf{x}_t) - (t - s)\, \tfrac{d}{dt} \bar{\mathbf{u}}(\mathbf{x}_t, s, t)}$$

$\tfrac{d}{dt}\bar{\mathbf{u}} = \partial_t \bar{\mathbf{u}} + (\nabla_{\mathbf{x}} \bar{\mathbf{u}})\, \mathbf{u}_t$:
one forward-mode Jacobian-vector product with tangent $(\mathbf{u}_t, 1)$.

[Geng et al., 2025]{.small}
:::

::: {.slide title="Check: The Identity on Gaussian Data"}
[Check]{.kicker}

With $m = 2$, $\varsigma = 0.5$, $s = 0$, $t = 1$: $\bar{u} = 0.5\, x - 2$
and $u_1 = x - 2$. What must $\tfrac{d}{dt}\bar{u}$ along the trajectory be?
Confirm it directly.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $u_1 - \bar{u} = 0.5\, x$. Directly:
$\bar{u}(x_t, 0, t) = (x_t - x_0)/t$ with $x_0$ fixed along the trajectory,
so $\tfrac{d}{dt}\bar{u} = (u_t - \bar{u})/t$, which is $0.5\, x$ at $t = 1$.
:::
:::
:::

::: {.slide title="Regress onto the Right-Hand Side, Held Fixed"}
$$\Big\| \bar{\mathbf{u}}_{\boldsymbol\theta}(\mathbf{x}_t, s, t) - \mathrm{sg}\Big[\mathbf{u}_{\textrm{ref}} - (t - s)\, \tfrac{d}{dt} \bar{\mathbf{u}}_{\boldsymbol\theta}(\mathbf{x}_t, s, t)\Big] \Big\|^2$$

- $\mathbf{u}_{\textrm{ref}}$: a teacher's velocity $\hat{\mathbf{u}}$, or the example's
  $\boldsymbol\epsilon - \mathbf{x}_0$. The target is affine in
  $\mathbf{u}_{\textrm{ref}}$, so its average over examples uses $\mathbf{u}_t$.
- $\mathrm{sg}$, stop-gradient: the averaging needs the network in the
  prediction only. Through the target it could lower the loss by
  reshaping its Jacobian instead.
:::

::: {.slide title="Two Choices, Four Methods"}
| velocity $\mathbf{u}_{\textrm{ref}}$ | one time, $s = 0$ | two times, $s \leq t$ |
|---|---|---|
| a teacher's $\hat{\mathbf{u}}$ | consistency distillation | flow-map distillation |
| the example's $\boldsymbol\epsilon - \mathbf{x}_0$ | consistency training | MeanFlow |

At $s = 0$: $f(\mathbf{x}, t) = \mathbf{x} - t\, \bar{\mathbf{u}}(\mathbf{x}, 0, t)$.

::: {.d2l-note}
Sampler: cut $[0, 1]$ into equal intervals and jump,
$\mathbf{x}_s = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t)$;
with $\mathbf{u}_t$ in place of $\bar{\mathbf{u}}$ it is Euler.
:::
:::

::: {.slide title="The Loss Is One Jacobian-Vector Product and a Detached Target" only="pytorch"}
```{.python #few-step-exercises-1}
x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
v = eps - x0 if teacher is None else teacher(x_t, t)  # no grad
u_t, du_dt = torch.func.jvp(lambda x, r: u(x, s, r), (x_t, t),
                            (v, torch.ones_like(t)))
target = (v - gap * du_dt).detach()        # gap = t - s
error = ((u_t - target) ** 2).flatten(1).sum(1)
```

::: {.d2l-note}
Flow maps use time features at frequencies up to 1 per unit of $t$. At
frequency $\omega$, $\tfrac{d}{dt}\sin(\omega t) = \omega\cos(\omega t)$:
embedding $1000\,t$ would multiply the target by up to a thousand.
:::
:::

::: {.slide title="The Loss Is One Jacobian-Vector Product and a Detached Target" only="jax"}
```{.python #few-step-exercises-2}
x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
v = eps - x0 if teacher is None else teacher(x_t, t)
v = jax.lax.stop_gradient(v)
u_t, du_dt = jax.jvp(lambda x, r: u(x, s, r), (x_t, t),
                     (v, jnp.ones_like(t)))
target = jax.lax.stop_gradient(v - gap * du_dt)  # gap = t - s
error = ((u_t - target) ** 2).reshape(len(x0), -1).sum(1)
```

::: {.d2l-note}
Flow maps use time features at frequencies up to 1 per unit of $t$. At
frequency $\omega$, $\tfrac{d}{dt}\sin(\omega t) = \omega\cos(\omega t)$:
embedding $1000\,t$ would multiply the target by up to a thousand.
:::
:::

::: {.slide title="Check: Which Student's Loss Can Reach Zero?"}
[Check]{.kicker}

Three students on the mixture, same network and updates: consistency and
flow-map distillation (teacher's velocity), MeanFlow (each example's
velocity). Which final losses can approach zero?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The distilled ones: their target is a function of the point.
Measured: 0.075 (consistency), 0.03 (two-time), 76 to 81 (MeanFlow), whose
target varies between examples at the same point.
:::
:::
:::

::: {.slide title="With a Teacher the Students Keep the Spread; MeanFlow Contracts the Modes"}
@!few-step-on-the-running-example-2

::: {.d2l-note}
One evaluation: weights within 0.007. Spread: data 0.47, distilled 0.45 to
0.46, MeanFlow 0.31 to 0.34. Strays: data 1%, distilled 2%, MeanFlow 4%.
The two-time students differ only in the reference velocity, and that
alone shrinks the spread by more than 0.1. After four jumps MeanFlow's
weights vary between training runs (up to about 0.02 off over six seeds);
the distilled student's stay within 0.006.
:::
:::

::: {.slide title="Two Student Jumps Are Worth About Fifteen Euler Steps"}
::: {.cols .vc}
::: {.col}
@!few-step-on-images-3
:::

::: {.col .narrow}
Student: the teacher U-Net plus a zero-initialized input $t - s$, 5,000
updates on the teacher's velocity. FD: Fréchet distance of classifier
features to 1,000 test images; floor (two sets of test images) about 0.8.

- Against the teacher's fifty steps: one jump 4 to 5 times the FD, two
  nearly twice, four about 1.5 times.
:::
:::
:::

::: {.slide title="The Student Draws the Teacher's Garment from the Same Noise"}
@!few-step-on-images-4

Rows: teacher, 50 and 2 Euler steps; student, 1 and 2 jumps. Two Euler
steps draw grey averages; one jump has hard edges and blotches; two come
close to the teacher.

::: {.d2l-note}
A distilled student's target is its teacher's exact ODE, whose FD fifty
Euler steps approach from above; the floor is out of its reach.
:::
:::

::: {.slide title="Matching Distributions Follows the Student's Score Minus the Teacher's"}
$\mathbf{x}_t = (1 - t)\, g_{\boldsymbol\theta}(\mathbf{z}) + t\, \boldsymbol\epsilon$,
with law $q_{\boldsymbol\theta, t}$; noised data $p_t$:

$$\nabla_{\boldsymbol\theta} D_{\textrm{KL}}(q_{\boldsymbol\theta, t} \| p_t) = \mathbb{E}\big[(\nabla \log q_{\boldsymbol\theta, t} - \nabla \log p_t)^\top \tfrac{\partial \mathbf{x}_t}{\partial \boldsymbol\theta}\big]$$

| route | bounded by | same noise, same sample as the ODE? |
|:--|:--|:--|
| flow-map distillation | the teacher's ODE | yes |
| distribution matching | the teacher's distribution | not guaranteed |
| a critic on real images | neither | no |

[Yin et al., 2024; Sauer et al., 2023]{.small}
:::

::: {.slide title="Check: Which Student Keeps the Noise-to-Sample Pairing at Two Evaluations?"}
[Check]{.kicker}

A consistency model, a two-time flow map trained by MeanFlow, or an
adversarially distilled generator: which, spending two evaluations, still
returns a sample that depends on the starting noise alone?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The two-time flow map: two jumps along one trajectory. The
consistency model and the adversarial generator both re-noise with fresh
noise for a second evaluation.
:::
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **One Euler step** returns $\mathbb{E}[\mathbf{x}_0]$: all strays on the
  mixture; two steps still about a quarter.
- **Flow map:** $\mathbf{x}_s = \mathbf{x}_t - (t - s)\,\bar{\mathbf{u}}$;
  MeanFlow identity $\bar{\mathbf{u}} = \mathbf{u}_t - (t - s)\tfrac{d}{dt}\bar{\mathbf{u}}$,
  one JVP and a stop-gradient.
- **Mixture:** one evaluation gets the weights within 0.007; distilled
  students keep the spread 0.45 to 0.46, MeanFlow contracts it to 0.31 to
  0.34.
- **Images:** one jump 4 to 5 times the teacher's 50-step FD; two jumps
  worth about fifteen Euler steps.

**Next:** no continuous path exists for tokens; what is noise for a token?
:::
