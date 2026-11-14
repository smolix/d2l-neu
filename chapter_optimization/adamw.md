# AdamW
:label:`sec_adamw`

In :numref:`sec_weight_decay`, weight decay was one idea told in two ways.
We could add a penalty on the squared size of the weights to the loss, or
multiply the weights by a number slightly below one in every step, and
under minibatch SGD the two were the same operation. We noted there that
the two part ways once an optimizer rescales each coordinate of the
gradient, and left it at that. Adam is such an optimizer, and it is rarely
used without weight decay: the recipe of :numref:`sec_training_recipes`
read "AdamW, weight decay 0.05", and most published recipes for language
models read "AdamW, weight decay 0.1". We typed these numbers in, and the
models trained. Which of the two forms did we use, and what is 0.1 a tenth
of?

Adam (:numref:`sec_adam`) divides each coordinate of the gradient by the
scale of that coordinate's own gradients. A penalty that enters through
the gradient is divided along with everything else, and one coefficient
then shrinks every weight at a rate of its own. For several years this is
what the Adam of the common libraries did when asked for weight decay.
:citet:`Loshchilov.Hutter.2019` pointed it out and proposed a change of
one line: shrink the weights directly, outside the division. The result is
AdamW, the default optimizer for transformers ever since.

Once the decay bypasses the division, the coefficient $\lambda$ has one
meaning again, and the meaning is a time. Each step removes the fraction
$\eta\lambda$ of every weight, so the decay needs about $1/(\eta\lambda)$
steps to act. At a learning rate of $0.003$, "weight decay 0.1" says that
what the weights held 3,333 steps ago has faded to about a third. We first
watch the older form at work on a network and build the newer one. We then
follow the timescale: how long decay takes to act, where it leaves the
weights, and how to choose $\lambda$. The section closes with the
parameters that are exempt from decay and the memory that Adam's two
averages occupy.

```{.python .input #adamw-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #adamw-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #adamw-2}
%%tab pytorch
import copy
import random
```

## Decoupled Weight Decay

### Penalty and Preconditioner

Adam's update (:numref:`sec_adam`) is

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$

where $\hat{\mathbf{m}}_t$ is a running average of the gradients
$\mathbf{g}_t$ and $\hat{\mathbf{v}}_t$ a running average of their squares,
both kept for each coordinate separately. The division by the scale
$\sqrt{\hat{\mathbf{v}}_t}$ is a diagonal preconditioner in the sense of
:numref:`sec_gd`: it rescales every coordinate of the gradient before the
step is taken. What does it do to weight decay?

We write $\lambda$ for the decay coefficient, as
:numref:`sec_weight_decay` did (the weights were $\mathbf{w}$ there and
are $\mathbf{x}$ in this chapter). Earlier sections of the chapter used
the same letter for a curvature, and the two uses agree: $\lambda$ is the
curvature of the penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$. Adding this
penalty to the loss adds $\lambda \mathbf{x}_t$ to the gradient. Under SGD
the update then splits into a uniform shrinkage and an ordinary gradient
step:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.
$$
:eqlabel:`eq_l2-is-decay`

Penalizing the loss *is* shrinking the weights, by the factor
$1 - \eta\lambda$ for every coordinate. Now feed the same penalized
gradient to Adam. Both averages are built from it, so the penalty passes
through the preconditioner too. If we treat the scale
$\sqrt{\hat{\mathbf{v}}_t}$ as fixed for a moment, the shrinkage of
coordinate $i$ in one step compares with that of weight decay as follows:

$$
\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}}_{\textrm{$\ell_2$ penalty through Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\; x_{t,i}\vphantom{\frac{\eta}{\sqrt{\hat{v}_{t,i}}}}}_{\textrm{weight decay}}.
$$
:eqlabel:`eq_coupled-decay`

The penalty is rescaled by the same factor that rescales the loss
gradient. A parameter with large gradients is barely decayed and a
parameter with small gradients is decayed strongly, although nobody asked
for either. The effect is not subtle. In `TinyLM` the gradients of the
weight matrices inside the blocks have a scale of about $10^{-3}$ at
initialization (:numref:`sec_adam`). At $\eta = 0.003$ and $\lambda = 0.1$,
weight decay removes the fraction $\eta\lambda = 0.0003$ of such a weight
per step, and the penalty through Adam removes
$\eta\lambda / 10^{-3} = 0.3$ of it. The same coefficient has two meanings,
a factor of a thousand apart.
:numref:`subsec_mdl-decoupled-weight-decay` measures the effect on weights
whose gradients are pure noise.

AdamW *decouples* the two: the loss gradient goes through the
preconditioner, and the decay does not,

$$
\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$
:eqlabel:`eq_adamw`

with $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ computed from the plain
gradient $\mathbf{g}_t$ alone. Every weight shrinks by the same factor
again, as in :eqref:`eq_l2-is-decay`. The shrinkage per step is the
product $\eta\lambda$, as in the libraries we use, so it follows the
learning rate through a schedule, and the weights stop shrinking when
learning stops.

### AdamW from Scratch

The decoupled update adds one operation to Adam: shrink the weights, then
take the Adam step. :numref:`sec_adam` wrote Adam from scratch and checked
it against the library, so we build on the library's Adam here and add
only the shrinkage.

```{.python .input #adamw-adamw-from-scratch-1}
%%tab pytorch
class AdamW(torch.optim.Adam):
    """Adam with decoupled weight decay: shrink the weights, then step."""
    def __init__(self, params, lr=1e-3, weight_decay=0.01):
        super().__init__(params, lr)  # Adam's own weight_decay stays 0
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.mul_(1 - group['lr'] * self.weight_decay)
        super().step()
```

```{.python .input #adamw-adamw-from-scratch-1}
%%tab jax
def adamw(learning_rate, weight_decay):
    return optax.chain(optax.scale_by_adam(),
                       optax.add_decayed_weights(weight_decay),
                       optax.scale(-learning_rate))
```

:begin_tab:`pytorch`
`torch.optim.Adam` has a `weight_decay` argument of its own. It adds
$\lambda \mathbf{x}$ to the gradient before the averages are updated, the
coupled form on the left of :eqref:`eq_coupled-decay`, so our class leaves
it at zero and keeps its own coefficient. The shrinkage reads the current
learning rate of each parameter group and therefore follows a schedule.
`torch.optim.AdamW` is the library's decoupled form.
:end_tab:

:begin_tab:`jax`
`optax.scale_by_adam` is the library's version of the transformation
written in :numref:`sec_adam`. `optax.add_decayed_weights` adds
$\lambda \mathbf{x}$ to whatever reaches it. Placed after `scale_by_adam`,
the decay bypasses the preconditioner. Placed before it, as in
`optax.chain(optax.add_decayed_weights(wd), optax.scale_by_adam(), ...)`,
it gives the coupled form: the position of one link decides which of the
two updates we get.
:end_tab:

Does our version do what the library's AdamW does? We train two copies of
a linear model on the same least-squares problem for 100 steps at
$\lambda = 0.1$, one with each, and print the largest difference between
their weights.

```{.python .input #adamw-adamw-from-scratch-2}
%%tab pytorch
X, y = torch.randn(512, 20), torch.randn(512, 1)
net = nn.Linear(20, 1)
twin = copy.deepcopy(net)
for model, optimizer in [
        (net, AdamW(net.parameters(), lr=0.01, weight_decay=0.1)),
        (twin, torch.optim.AdamW(twin.parameters(), lr=0.01,
                                 weight_decay=0.1))]:
    for _ in range(100):
        loss = F.mse_loss(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print('largest difference after 100 steps: '
      f'{(net.weight - twin.weight).abs().max():.1e}')
```

```{.python .input #adamw-adamw-from-scratch-2}
%%tab jax
@nnx.jit
def least_squares_step(model, optimizer, X, y):
    grads = nnx.grad(lambda model: ((model(X) - y) ** 2).mean())(model)
    optimizer.update(model, grads)

k_X, k_y, k_w = jax.random.split(d2l.get_key(), 3)
X, y = jax.random.normal(k_X, (512, 20)), jax.random.normal(k_y, (512, 1))
net = nnx.Linear(20, 1, rngs=nnx.Rngs(k_w))
twin = nnx.clone(net)
for model, tx in [(net, adamw(0.01, weight_decay=0.1)),
                  (twin, optax.adamw(0.01, weight_decay=0.1))]:
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    for _ in range(100):
        least_squares_step(model, optimizer, X, y)
diff = jnp.abs(net.kernel[...] - twin.kernel[...]).max()
print(f'largest difference after 100 steps: {diff:.1e}')
```

The difference is zero to the last bit: the library performs the same
operations in the same order.

### Coupled and Decoupled Decay Compared

How much does the choice between the two forms matter on a network? We
take `TinyLM`, the transformer of :numref:`subsec_tinylm` with two blocks
and about 0.47 million parameters, and the data of the previous section,
minibatches of 64 windows of 64 bytes of Wikipedia text.

```{.python .input #adamw-coupled-and-decoupled-decay-compared-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

We train the model three times for 1,000 steps at $\eta = 0.003$ from one
random initialization: without decay, with the penalty inside Adam's
gradient at $\lambda = 0.1$, and with AdamW at $\lambda = 0.1$. For now
the decay applies to every parameter. Besides the loss we follow the size
of the weights. The function `matrix_rms` returns the root mean square
(RMS) of the entries of the weight matrices, the weights of the linear
layers, which hold 90% of the parameters. The function `train_tracked`
trains a fresh `TinyLM` and records this number at the start and every 250
steps.

```{.python .input #adamw-coupled-and-decoupled-decay-compared-2}
%%tab pytorch
def matrix_rms(model):
    """RMS of the entries of the weight matrices, embeddings excluded."""
    W = torch.cat([p.detach().flatten() for name, p in model.named_parameters()
                   if p.ndim == 2 and 'emb' not in name])
    return float(W.square().mean().sqrt())

def train_tracked(make_optimizer, data, seed, num_steps, every=250):
    """Train TinyLM from seed with make_optimizer(params); return the losses,
    the matrix RMS at the start and every `every` steps, and the model."""
    torch.manual_seed(seed)
    model = d2l.TinyLM(len(data.vocab))
    optimizer = make_optimizer(model.parameters())
    losses, rms = [], [matrix_rms(model)]
    for _ in range(num_steps // every):
        losses += d2l.train_steps(model, data, optimizer, every)
        rms.append(matrix_rms(model))
    return losses, rms, model
```

```{.python .input #adamw-coupled-and-decoupled-decay-compared-2}
%%tab jax
def matrix_rms(model):
    """RMS of the entries of the weight matrices, embeddings excluded."""
    leaves = jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param))
    W = jnp.concatenate([p.ravel() for path, p in leaves
                         if p.ndim == 2 and 'emb' not in str(path)])
    return float(jnp.sqrt((W ** 2).mean()))

def train_tracked(tx, data, seed, num_steps, every=250):
    """Train TinyLM from seed with the optax transformation tx; return the
    losses, the matrix RMS at the start and every `every` steps, the model."""
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses, rms = [], [matrix_rms(model)]
    for _ in range(num_steps // every):
        losses += d2l.train_steps(model, data, optimizer, every)
        rms.append(matrix_rms(model))
    return losses, rms, model
```

Before running anything we can say what the coupled form must do. The
entries of the weight matrices start with an RMS of about $0.045$, so the
gradient of the penalty, $\lambda x$, is about $0.0045$, several times the
loss gradient of $10^{-3}$. The 30% per step of :eqref:`eq_coupled-decay`
is then out of reach: that formula held the scale fixed, and a penalty
this large is itself most of the gradient from which the scale is
computed. Adam does not ask where a gradient comes from. It divides the
penalty by the penalty's own size and moves the weight toward zero by
about $\eta = 0.003$ per step. At that pace an entry of size $0.045$
reaches zero in fifteen steps, so a few dozen steps should erase the
matrices. Note what has become of $\lambda$. The speed of the shrinkage is
about $\eta$ per step whatever $\lambda$ is. The coefficient decides only
where the shrinkage stops: at the size where $\lambda |x|$ has come down
to the loss gradient.

```{.python .input #adamw-one-number-two-meanings}
%%tab pytorch
seed = random.getrandbits(31)
curves = {}
for name, make_optimizer in [
        ('no decay', lambda params: torch.optim.Adam(params, 3e-3)),
        ('Adam + ℓ2', lambda params: torch.optim.Adam(params, 3e-3,
                                                       weight_decay=0.1)),
        ('AdamW', lambda params: torch.optim.AdamW(params, 3e-3,
                                                    weight_decay=0.1))]:
    curves[name], rms, _ = train_tracked(make_optimizer, data, seed, 1000)
    print(f'{name:<10} final loss {d2l.final_loss(curves[name]):.3f}, RMS of '
          f'the weight matrices {rms[0]:.4f} -> {rms[-1]:.4f}')
d2l.plot(list(range(0, 1000, 25)), [d2l.smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

```{.python .input #adamw-one-number-two-meanings}
%%tab jax
def coupled(lr, wd):  # The penalty's gradient enters Adam's moments
    return optax.chain(optax.add_decayed_weights(wd), optax.adam(lr))

seed = d2l.get_seed()
curves = {}
for name, tx in [('no decay', optax.adam(3e-3)),
                 ('Adam + ℓ2', coupled(3e-3, 0.1)),
                 ('AdamW', optax.adamw(3e-3, weight_decay=0.1))]:
    curves[name], rms, _ = train_tracked(tx, data, seed, 1000)
    print(f'{name:<10} final loss {d2l.final_loss(curves[name]):.3f}, RMS of '
          f'the weight matrices {rms[0]:.4f} -> {rms[-1]:.4f}')
d2l.plot(list(range(0, 1000, 25)), [d2l.smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

By its loss, the run with AdamW cannot be told from the run without decay.
The two final losses differ by less than $0.02$, about as much as two runs
of one setting differ (:numref:`sec_test_models`). The coupled run ends more than
a nat higher, and the printed weights say why. Without decay the RMS of
the matrices has more than doubled, from $0.045$ to $0.11$. In the coupled
run it has fallen below $0.003$, forty times smaller. To read the loss of
that run, we compare it with a model that ignores the context and predicts
every byte from its frequency in the training text. The loss of such a
model is the entropy of the byte frequencies:

```{.python .input #adam-coupled-and-decoupled-decay-compared-2}
%%tab pytorch, jax
freq = np.bincount(data.train, minlength=256) / len(data.train)
freq = freq[freq > 0]
print(f'entropy of the byte frequencies: {-(freq * np.log(freq)).sum():.2f}')
```

The coupled run ends within about $0.05$ nats of this value, and from its
first hundred steps on its curve stays within $0.15$ nats of it. A network
with erased matrices can do no better. Almost nothing of its input reaches
its output, so it makes the same prediction at every position, and the
best such prediction is the frequency of each byte. The matrices stopped
shrinking at an RMS just below $0.003$, where the penalty's gradient
$\lambda |x|$ is about $3 \times 10^{-4}$, the size of the loss gradients
of these matrices once training is under way (:numref:`sec_adam`). Under
the coupled form, $\lambda = 0.1$ came to mean that no weight may be much
larger than ten times its own gradient. Nobody who writes "weight decay
0.1" into a recipe means that.

## The Timescale of Decay
:label:`subsec_wd-at-scale`

AdamW at the same $\lambda = 0.1$ did almost nothing: the loss did not
change, and the weight matrices ended a tenth smaller than without decay.
Is the value of the recipes too small to matter? That depends on the
length of the run, and to see why, we unroll the update.

### Decay as a Leaky Average

Write $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$
for Adam's step before the learning rate, so that :eqref:`eq_adamw` reads
$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{u}_t$.
Substituting the update into itself $t$ times gives

$$
\mathbf{x}_t = (1 - \eta\lambda)^t\, \mathbf{x}_0
- \eta \sum_{s=0}^{t-1} (1 - \eta\lambda)^{t-1-s}\, \mathbf{u}_s.
$$
:eqlabel:`eq_adamw-unrolled`

The weights are a sum over all past steps, in which a step taken $k$ steps
ago counts with the weight $(1 - \eta\lambda)^k \approx e^{-\eta\lambda k}$.
This is the leaky average of :numref:`sec_momentum`, with
$1 - \eta\lambda$ in the place of $\beta$, applied to the steps instead of
the gradients. Its window, $1/(\eta\lambda)$ steps, is the *timescale* of
the decay (the literature also calls it the horizon). Weight decay under
AdamW is a rule for forgetting: the weights remember about their last
$1/(\eta\lambda)$ steps.

For $\eta = 0.003$ and $\lambda = 0.1$ the timescale is
$1/0.0003 = 3{,}333$ steps. Our run lasted 1,000 steps, or 0.3 timescales.
In that time the oldest contribution, the initialization, was multiplied
by $e^{-0.3} = 0.74$, and everything later by a number closer to one.
Decay had no time to remove more than a quarter of anything, and it
removed a tenth of the RMS. So 0.1 is a slow coefficient rather than a
small one, and our run was too short for it.

### The Equilibrium Norm

What happens in a run longer than the timescale? Take one entry $x$ of a
weight matrix and the matching entry $u_t$ of the step. Its update,
$x_{t+1} = (1 - \eta\lambda)\, x_t - \eta\, u_t$, is letter for letter the
recursion that produced the noise ball in :numref:`sec_sgd`, with the
decay coefficient in the place of the curvature and Adam's step in the
place of the gradient noise. Suppose, as we did there, that the steps have
mean zero and mean square $\sigma^2$ and are independent of the weight.
Squaring both sides and averaging then gives

$$
\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2
\qquad \textrm{with the fixed point} \qquad
\mathbb{E}\big[x_\infty^2\big] \approx \frac{\eta\, \sigma^2}{2\lambda}.
$$
:eqlabel:`eq_adamw-equilibrium`

Every step lengthens the weights a little, decay shortens them by the
fraction $\eta\lambda$, and their size settles where the two cancel. The
lengthening does not need randomness: any step at a right angle to the
weight vector makes it longer, by Pythagoras.

What is $\sigma^2$? Adam divides every gradient by its scale, the root of
the average squared gradient, so the entries of
$\mathbf{g}_t / \sqrt{\hat{\mathbf{v}}_t}$ have mean square about one.
Momentum hands each of these normalized gradients out in portions over
about ten steps. A single step is therefore shorter, about a fifth of
$\eta$ per entry in the table of :numref:`sec_adam`, but the total that
each gradient contributes is unchanged, and ten steps are nothing against
a timescale of hundreds. If successive gradients are mostly noise, we may
set $\sigma^2 = 1$, and the RMS of a decayed weight matrix should settle
at $\sqrt{\eta/(2\lambda)}$. Without decay nothing cancels the growth:
each step adds $\eta^2$ to the mean square of an entry, and after $t$
steps the RMS is $\sqrt{x_0^2 + \eta^2 t}$, the spread of a random walk.

We test this on `TinyLM` with AdamW at $\eta = 0.003$, for 2,000 steps and
four values of $\lambda$: 0, 0.1, 0.5 and 1, with timescales of infinity,
3,333, 667 and 333 steps. The predictions are definite. Without decay the
RMS of the matrices should grow from $0.045$ to
$\sqrt{0.045^2 + 0.003^2 \cdot 2{,}000} = 0.14$ and still be growing. With
$\lambda = 0.1$ it cannot settle, since the run is shorter than the
timescale. With $\lambda = 0.5$ and $\lambda = 1$ it should settle within
the run, at $\sqrt{0.003/1} = 0.055$ and $\sqrt{0.003/2} = 0.039$. The
cell trains two models for each $\lambda$. It prints the RMS of the weight
matrices every 250 steps, averaged over the two, the predicted level with
the ratio of the final RMS to it, and the validation loss of both runs.

```{.python .input #adamw-the-timescale-of-decay-1}
%%tab pytorch
lr, num_steps, every = 3e-3, 2000, 250
seeds = [random.getrandbits(31) for _ in range(2)]
table = {}
print(f'{"λ":<5}' + ''.join(f'{t:>6}' for t in range(0, num_steps + 1, every))
      + f'{"predicted":>10}{"ratio":>7}  validation loss')
for wd in [0, 0.1, 0.5, 1]:
    runs = [train_tracked(lambda params: torch.optim.AdamW(
        params, lr, weight_decay=wd), data, seed, num_steps, every)
            for seed in seeds]
    table[wd] = [sum(rms) / len(rms) for rms in zip(*(r[1] for r in runs))]
    val = [math.log(d2l.lm_perplexity(r[2], data.val_dataloader()))
           for r in runs]
    if wd:  # The equilibrium RMS, and the measured RMS divided by it
        pred = math.sqrt(lr / (2 * wd))
        cols = f'{pred:10.3f}{table[wd][-1] / pred:7.2f}'
    else:
        cols = f'{"-":>10}{"-":>7}'
    print(f'{wd:<5g}' + ''.join(f'{v:6.3f}' for v in table[wd]) + cols + '  '
          + ' '.join(f'{v:.3f}' for v in val))
d2l.plot(list(range(0, num_steps + 1, every)), list(table.values()), 'step',
         'RMS of the weight matrices', legend=[f'λ = {wd:g}' for wd in table])
```

```{.python .input #adamw-the-timescale-of-decay-1}
%%tab jax
lr, num_steps, every = 3e-3, 2000, 250
seeds = [d2l.get_seed() for _ in range(2)]
table = {}
print(f'{"λ":<5}' + ''.join(f'{t:>6}' for t in range(0, num_steps + 1, every))
      + f'{"predicted":>10}{"ratio":>7}  validation loss')
for wd in [0, 0.1, 0.5, 1]:
    runs = [train_tracked(optax.adamw(lr, weight_decay=wd), data, seed,
                          num_steps, every) for seed in seeds]
    table[wd] = [sum(rms) / len(rms) for rms in zip(*(r[1] for r in runs))]
    val = [math.log(d2l.lm_perplexity(r[2], data.val_dataloader()))
           for r in runs]
    if wd:  # The equilibrium RMS, and the measured RMS divided by it
        pred = math.sqrt(lr / (2 * wd))
        cols = f'{pred:10.3f}{table[wd][-1] / pred:7.2f}'
    else:
        cols = f'{"-":>10}{"-":>7}'
    print(f'{wd:<5g}' + ''.join(f'{v:6.3f}' for v in table[wd]) + cols + '  '
          + ' '.join(f'{v:.3f}' for v in val))
d2l.plot(list(range(0, num_steps + 1, every)), list(table.values()), 'step',
         'RMS of the weight matrices', legend=[f'λ = {wd:g}' for wd in table])
```

The run without decay ends at $0.14$ and follows the random walk to within
7% at every checkpoint. With $\lambda = 0.1$ the weights are still growing
when the run ends, by about $0.003$ per 250 steps; the ratio just below
one in its row belongs to a run caught on the way up and says nothing
about the level. With $\lambda = 0.5$ the RMS is level from step 1,500 on,
and with $\lambda = 1$ from step 750 on, in each case after about two
timescales. The two levels, about $0.076$ and $0.054$, differ by the
factor $\sqrt{2}$ that the formula demands.

The formula has the ratio of the two levels right and their height wrong:
both lie about 1.4 times above the prediction. The derivation assumed steps of
mean square one that are independent from one step to the next. In a
network that is learning, a gradient keeps part of its direction for many
steps, and whatever persists adds up faster than noise does
(Exercise :exref:`ex_adamw-two-limits`).

The last column says what the equilibrium is worth in a run of 2,000
steps. With $\lambda = 0.1$ the validation loss stays within about a
hundredth of the runs without decay, the spread between two runs of one
setting. With $\lambda = 0.5$ it is about $0.1$ nats worse, and with
$\lambda = 1$ about $0.25$. Slow decay does nothing here, and decay that
is fast enough to settle does harm.

### Choice of the Coefficient

Why, then, do the recipes decay at all? Because of what growing weights do
to the learning rate. In `TinyLM`, as in most networks today, a
normalization layer follows every weight matrix sooner or later. Multiply
a matrix whose output is normalized by a constant, and the normalization
divides the constant out again: the network depends on the direction of
such a matrix and not on its size. (This is exact when the normalization
follows directly, and approximate for the matrices of `TinyLM`.) What a
step does to a direction depends on its size relative to the weights,
which for a step of RMS $\eta$ is $\eta$ divided by their RMS. In the run
without decay this *relative step* fell from $0.003/0.045 = 0.067$ to
$0.003/0.14 = 0.021$ and was still falling, in proportion to
$1/\sqrt{t}$. The run lowered its own learning rate threefold, on a
schedule that nobody chose. With decay the relative step stops falling at

$$
\frac{\eta}{\sqrt{\eta/(2\lambda)}} = \sqrt{2\eta\lambda},
$$

the same value for every decayed matrix, whatever its size and the scale
of its gradients, until we lower $\eta$ ourselves.
:citet:`Kosson.Messmer.Jaggi.2024` call this state a rotational
equilibrium: every matrix turns by the same angle per step. The product
$\eta\lambda$ thus appears twice. Its inverse is the time that the weights
take to settle, and its square root is the rate at which they turn once
settled.

In 2,000 steps none of this helps. A language model, however, is trained
for hundreds of thousands of steps in a single pass over its data, where
there is little overfitting for a penalty to prevent. Without decay its
matrices would grow for the whole run, each at its own pace, and those
that grow fastest would learn slowest.
:citet:`DAngelo.Andriushchenko.Varre.ea.2024` find that in such runs weight
decay lowers the training loss itself, the loss that a regularizer would
raise. At this scale weight decay belongs to the optimizer, and
$\eta\lambda$ is its second rate next to $\eta$.

Three rules for choosing $\lambda$ follow. First, count timescales before
tuning anything: a run of $T$ steps contains $T\eta\lambda$ of them. Our
runs at $\lambda = 0.1$ contained 0.3 and 0.6, and decay did nothing. A
run of 300,000 steps at $\eta = 3 \times 10^{-4}$ and $\lambda = 0.1$
contains nine. Second, because the libraries multiply $\lambda$ by $\eta$,
a change of the learning rate changes the decay with it. When you search
over $\eta$, hold the product fixed by setting $\lambda = s/\eta$ for a
chosen shrinkage $s$ per step, and search over $s$ separately.
:citet:`Loshchilov.Hutter.2019` wrote the decay without the factor $\eta$,
so that their coefficient is this $s$, and found its best value nearly
independent of the learning rate. A published "weight decay 0.1" is
nearly always the library's $\lambda$. Third, a longer run needs a longer
timescale. Wang and Aitchison (2024), to whom the reading of AdamW as an
average of its recent steps is due, found that the best timescale, counted
in passes over the data, stays roughly constant as models and datasets
grow, and :citet:`Bergsma.Dey.Gosal.ea.2025b` measured for language-model
pretraining how it depends on the number of training tokens per
parameter. A $\lambda$ tuned on a short trial run is therefore too large
for the long run that the trial was meant to prepare.

All of this describes weights whose steps are mostly noise, as in a single
pass over a large text. A weight whose gradient keeps one sign behaves
differently. Adam pushes it by $\eta$ per step, decay pulls it back by
$\eta\lambda |x|$, and it comes to rest at $|x| = 1/\lambda$, where $\eta$
has cancelled. At $\lambda = 0.1$ this bound is 10, a hundred times the
weights in our table, so it plays no part in our runs. It does when a
small dataset is fitted for many passes: where the weights settle then
depends on $\lambda$ alone, and the best $\lambda$ can stay put while
$\eta$ changes (Exercise :exref:`ex_adamw-joint-search`).

## Parameters Exempt from Decay
:label:`subsec_adamw-what-not-to-decay`

The runs above decayed every parameter. That is the simplest choice, and
it is not what practice does. The argument of the last subsection was
about weight matrices, and of the three groups of parameters in `TinyLM`
(:numref:`subsec_tinylm`) only the matrices are decayed in practice. The
other two groups are exempt for different reasons.

The vectors, that is the biases and the scales and shifts of the
LayerNorms, are few, and they set the scale and the offset of each feature
directly. Their useful values are not small. The scale of a LayerNorm
starts at one, and pulling it toward zero changes the function of the
network instead of its relative step. Biases were exempt already in
:numref:`sec_weight_decay`.

The rows of an embedding table receive a gradient only when their token
occurs, while the decay acts at every step, so the rows of rare tokens
shrink with nothing to hold against it. The OLMo 2 team traced an
instability of their training to this. Decay shrank the embeddings over
the run, the LayerNorm that reads the embeddings divides by their size,
and the gradients that flow back through it grew as the embeddings shrank.
Exempting the embeddings removed the fault :cite:`OLMo.2025`.

In code the rule takes two groups of parameters, one decayed and one not.

```{.python .input #adamw-what-not-to-decay-1}
%%tab pytorch
model = d2l.TinyLM(len(data.vocab))
decay = [p for name, p in model.named_parameters()
         if p.ndim == 2 and 'emb' not in name]
no_decay = [p for name, p in model.named_parameters()
            if p.ndim != 2 or 'emb' in name]
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}], lr=3e-3)
n_decay = sum(p.numel() for p in decay)
n = n_decay + sum(p.numel() for p in no_decay)
print(f'decayed: {n_decay} of {n} parameters')
```

```{.python .input #adamw-what-not-to-decay-1}
%%tab jax
model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))

def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

optimizer = nnx.Optimizer(
    model, optax.adamw(3e-3, weight_decay=0.1, mask=decay_mask),
    wrt=nnx.Param)
params = nnx.state(model, nnx.Param)
n_decay = sum(int(p.size) for p, m in zip(
    jax.tree.leaves(params), jax.tree.leaves(decay_mask(params))) if m)
n = sum(int(p.size) for p in jax.tree.leaves(params))
print(f'decayed: {n_decay} of {n} parameters')
```

:begin_tab:`pytorch`
`torch.optim.AdamW` takes a list of parameter groups, each a dictionary
with its own settings. Settings that a group does not give fall back to
the keyword arguments.
:end_tab:

:begin_tab:`jax`
`optax.adamw` takes a `mask`, a function that maps the parameter tree to a
tree of booleans. Only the parameters marked `True` are decayed.
:end_tab:

The decayed group holds the 425,984 entries of the weight matrices, about
90% of the parameters. Exercise :exref:`ex_adamw-exemptions` measures what
each of the two exemptions is worth on `TinyLM`.

## Optimizer State
:label:`subsec_adamw-memory`

Adam and AdamW keep two numbers for every parameter, $m$ and $v$, which
are not part of the model. In 32-bit floating point a parameter in
training therefore occupies 16 bytes: four each for the weight, its
gradient, $m$ and $v$ (:numref:`sec_memory_precision` counts what fits on
a device). Half of this is optimizer state. For `TinyLM` the two averages
take less than 4 MB. For a model with a billion parameters they take 8 GB
of 16 GB, before a single activation is stored. Three consequences
follow. Only trained parameters carry the state, so freezing a part of a
model saves its gradients and its averages together. A checkpoint that is
to resume training must contain $m$ and $v$, or the run restarts with
empty averages. And when a model does not fit, the optimizer state is the
first place to look for memory, since nothing but the optimizer reads it.

Both averages are kept entry by entry, and every step is chosen entry by
entry. Yet the equilibrium above was a statement about a whole matrix,
which turns as one object. :numref:`sec_muon` asks whether the step should
be chosen for the matrix as a whole, too.

## Summary

Under Adam an $\ell_2$ penalty is divided by the gradient scale of each
coordinate, so one coefficient decays the weights at rates as different as
their gradients. On `TinyLM`, $\lambda = 0.1$ in this form erases the
weight matrices within the first steps and leaves a model that predicts
byte frequencies. AdamW shrinks every weight by the factor
$1 - \eta\lambda$ per step, outside the division, which gives its decay a
timescale of $1/(\eta\lambda)$ steps. A run shorter than the timescale
hardly feels the decay; in a longer run the weight matrices settle at an
RMS near $\sqrt{\eta/(2\lambda)}$, and each step then changes them by the
relative amount $\sqrt{2\eta\lambda}$. For this reason $\lambda$ is chosen
through the product $\eta\lambda$ and the length of the run, and only the
weight matrices are decayed. The two averages of Adam double the memory
that weights and gradients take.

## Exercises

1. [code] **Coupled decay on pure noise.** :eqref:`eq_coupled-decay`
   treats the scale $\sqrt{\hat{v}}$ as fixed. To test it, give a weight a
   loss gradient that is pure noise, so that decay is the only systematic
   force on it. In NumPy, start 10,000 independent copies of a weight at
   $x = 1$, draw the gradient of each copy at every step as
   $g_t \sim \mathcal{N}(0, \sigma^2)$, and run Adam with an $\ell_2$
   penalty (add $\lambda x_t$ to $g_t$ before the two averages are updated)
   at $\eta = 10^{-3}$, $\lambda = 0.1$ and the default $\beta_1$,
   $\beta_2$ and $\epsilon$. Report the mean of the copies.
    1. For $\sigma = 10$ and for $\sigma = 0.1$, predict the mean after
       4,000 steps from :eqref:`eq_coupled-decay` with
       $\sqrt{\hat{v}} \approx \sigma$, and the mean under AdamW at the
       same settings. Run both methods. Which of your predictions is the
       least accurate, and what did it neglect?
    1. In the opposite limit, $\sigma = 0$, the penalized gradient is
       $\lambda x_t$ alone. Show that for $\epsilon = 0$ the trajectory of
       $x_t$ does not depend on $\lambda > 0$.
    1. Record the mean after 1,000 steps for
       $\sigma \in \{10, 1, 0.1, 0.01, 0.001\}$ and
       $\lambda \in \{0.1, 1, 10\}$. Which weights no longer respond to
       $\lambda$, and where does the transition lie in terms of $\sigma$
       and $\lambda$? On which side of it were the weight matrices of
       `TinyLM` at the start of the coupled run?

1. **Two limits of AdamW.** Consider one weight under AdamW,
   $x_{t+1} = (1 - \eta\lambda)\, x_t - \eta\, u_t$.
    1. Let the loss gradient keep its sign and its size, so that
       $u_t = -1$ at every step. Show that $x_t$ converges to $1/\lambda$
       whatever $\eta$ is. After how many steps has the distance to the
       limit shrunk by the factor $e$?
    1. Now let the $u_t$ be independent, with mean zero and mean square
       one. Use :eqref:`eq_adamw-unrolled` to write $x_t$ for large $t$ as
       $-1/\lambda$ times a leaky average of the $u_s$, and derive
       $\mathbb{E}[x^2] \approx \eta/(2\lambda)$ from the variance of that
       average.
    1. Real steps are a mixture, $u_t = \bar{u} + \xi_t$, of a persistent
       part $\bar{u}$ and noise $\xi_t$ of mean square one. Show that
       $\mathbb{E}[x^2] \approx \bar{u}^2/\lambda^2 + \eta/(2\lambda)$.
       The settled runs in the table of :numref:`subsec_wd-at-scale` lie
       about 1.4 times above $\sqrt{\eta/(2\lambda)}$. If a persistent part
       were the whole explanation, how large would $|\bar{u}|$ be at
       $\lambda = 0.5$ and at $\lambda = 1$? What does the answer say
       about how much of a single step is signal?
    1. For $\eta = 0.003$ and $\lambda = 0.1$, compare the limit of part 1
       with the level of part 2 and with the RMS values in the table.
       Which of the two limits governs the weight matrices of `TinyLM`?
   :exlabel:`ex_adamw-two-limits`

1. [code] **Rotational equilibrium.** ● Following
   :citet:`Kosson.Messmer.Jaggi.2024`, model a weight matrix whose output
   is normalized by a vector $\mathbf{x} \in \mathbb{R}^d$ with
   $d = 1{,}000$. Start from a random vector of length one and iterate
   $\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{u}_t$
   in NumPy, where $\mathbf{u}_t$ is standard normal noise from which the
   component along $\mathbf{x}_t$ has been removed.
    1. Run 20,000 steps for each of $(\eta, \lambda) = (10^{-3}, 0.1)$,
       $(3 \times 10^{-3}, 0.1)$ and $(10^{-3}, 0.3)$. Compare the final
       RMS of the entries of $\mathbf{x}$ with $\sqrt{\eta/(2\lambda)}$ and
       the angle between $\mathbf{x}_t$ and $\mathbf{x}_{t+1}$ with
       $\sqrt{2\eta\lambda}$. Two of the runs share the angle. In what do
       they differ, and would a network with normalization notice?
    1. Give the steps momentum:
       $\mathbf{u}_t = \beta_1 \mathbf{u}_{t-1} + (1 - \beta_1)\,\mathbf{n}_t$
       with $\beta_1 = 0.9$ and noise $\mathbf{n}_t$ as before. Show that a
       single step is now shorter by the factor
       $\sqrt{(1 - \beta_1)/(1 + \beta_1)} \approx 0.23$. Does the
       equilibrium RMS shrink by this factor too? Explain what you find
       with :eqref:`eq_adamw-unrolled`.
    1. Set $\lambda = 0$ in the first run. How do the RMS and the angle
       per step change with $t$? State the result as the learning-rate
       schedule that a run without decay imposes on itself.

1. [code] **A joint search for $\eta$ and $\lambda$.** ● No $\lambda$ in
   our table helped. A penalty helps where a model overfits, and a single
   pass over a large text gives it no chance to. So make `TinyLM` overfit:
   `d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char', num_train=8000, num_val=2000)`
   holds 8,000 training windows of the text
   of :numref:`sec_text-sequence`. They overlap and cover only about 8,000
   characters of *The Time Machine*, and 800 steps pass over them about
   six times. Train with `train_tracked` and score each run by its
   validation loss, the logarithm of `d2l.lm_perplexity`. The two grids
   below take 27 runs.
    1. Before running, compute the number of timescales, $800\,\eta\lambda$,
       for $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ and
       $\lambda \in \{0.3, 1, 3, 10\}$. Which runs cannot differ much from
       a run without decay?
    1. Run this grid with AdamW, and a second grid with the same learning
       rates and $\lambda \in \{10^{-4}, 3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$
       for Adam with an $\ell_2$ penalty. For each form and each learning
       rate, which $\lambda$ is best? Repeat both grids from a second
       initialization. Which of your conclusions survive the repeat?
    1. Express the AdamW grid in the shrinkage per step $s = \eta\lambda$
       instead, with $s \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}\}$
       and the coefficient $s/\eta$ passed to the optimizer. Along which
       axis does the best column stay put when $\eta$ changes, $\lambda$ or
       $s$? Explain with part 1 of
       Exercise :exref:`ex_adamw-two-limits` why a small text that is
       fitted many times can differ from the single-pass runs
       of :numref:`subsec_wd-at-scale`.
   :exlabel:`ex_adamw-joint-search`

1. [code] **The exemptions, measured.** The table of
   :numref:`subsec_wd-at-scale` decayed every parameter. Repeat its rows
   $\lambda = 0.5$ and $\lambda = 1$ with four choices of the decayed
   group: all parameters, the matrices only
   (:numref:`subsec_adamw-what-not-to-decay`), the matrices and the
   embeddings, and the matrices and the vectors. In PyTorch the groups
   need the names of the parameters, so adapt `train_tracked` to build the
   optimizer from the model.
    1. How much of the cost of decay in validation loss, about $0.1$ and
       $0.25$ nats in the table, remains when only the matrices are
       decayed?
    1. Which of the two exemptions accounts for the difference? Relate the
       answer to the number of parameters in each group and to the reason
       given for its exemption.
    1. Record the RMS of the token embeddings at the end of each run. By
       how much does decay shrink them, and why does the validation loss
       hardly respond?
   :exlabel:`ex_adamw-exemptions`

1. **Activations and optimizer state.** Extend the count of
   :numref:`subsec_adamw-memory` to the rest of a training step.
    1. For `TinyLM` with batch size $B$ and sequence length $T$, estimate
       the memory of the fp32 activations that must be kept for the
       backward pass. Count, per block, the outputs of the two
       normalizations, the inputs and outputs of the attention layer, and
       the hidden layer of the MLP. At what batch size do the activations
       overtake the optimizer state?
    1. Count the floating-point operations of one AdamW update per
       parameter and compare them with the roughly $6BT$ operations per
       parameter of a forward and backward pass over one batch. At which
       batch size would the optimizer step cost one percent of a training
       step?

    *Adapted from Stanford CS336,
    [Assignment 1](https://github.com/stanford-cs336/spring2024-assignment1-basics),
    Problem adamwAccounting.*

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.7]{.kicker}

AdamW<br>**decoupled weight decay · the timescale of decay · exempt parameters · optimizer state**
:::
:::

::: {.slide title="One Coefficient, λ = 0.1, Trains One Run and Flattens Another"}
[The problem]{.kicker}

`TinyLM` (0.47 million parameters, predicts the next byte of Wikipedia
text), 1,000 steps of Adam at learning rate 0.003:

| weight decay | training loss, nats per byte |
|:--|--:|
| none | about 1.7 |
| 0.1, applied to the weights (AdamW) | about 1.7 |
| 0.1, as an $\ell_2$ penalty on the loss | about 3.2 |
| *a model that knows only how frequent each byte is* | *3.17* |

::: {.d2l-note .warn}
Under Adam, "weight decay 0.1" has two meanings, more than a nat apart.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive what an $\ell_2$ penalty does under Adam;
- write AdamW's update and say what it changes;
- compute the timescale of a recipe and tell whether decay acts in a run;
- predict the size at which decayed weights settle;
- choose which parameters to decay.

**Idea:** decay must bypass Adam's per-coordinate scaling; then
$1/(\eta\lambda)$ steps is its timescale ($\eta$: learning rate,
$\lambda$: decay coefficient).

**Running example:** `TinyLM` at $\eta = 0.003$, $\lambda = 0.1$:
timescale 3,333 steps.
:::

::: {.slide title="AdamW Completes the Optimizer of the Modern Recipe"}
[Where this fits]{.kicker}

| movement | sections |
|---|---|
| two difficulties, on a valley we can solve | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| which direction to step in, on two small networks | Two Test Models · Momentum · Adam · **AdamW** · Muon |
| how the step changes over a run, with batch size and width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** Adam's update, leaky averages with window $1/(1-\beta)$, the
  noise ball of constant-step SGD.
- **Skips:** other regularizers, learning-rate schedules, how to store the
  optimizer state in fewer bytes.
:::

::: {.slide title="Under SGD, an ℓ2 Penalty Is Weight Decay"}
[Penalty and preconditioner]{.kicker}

The penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ adds
$\lambda\mathbf{x}_t$ to the gradient $\mathbf{g}_t$:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta(\mathbf{g}_t + \lambda\mathbf{x}_t) = (1-\eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t$$

- $\mathbf{x}$: weights; $\eta$: learning rate; $\lambda$: decay
  coefficient.
- Every weight shrinks by the same factor $1 - \eta\lambda$ per step.
- $\eta = 0.003$, $\lambda = 0.1$: each step removes 0.03% of a weight.
:::

::: {.slide title="Adam Divides the Penalty by Each Weight's Gradient Scale"}
[Penalty and preconditioner]{.kicker}

Adam:
$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$;
$\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$: running averages of the
gradient and its square. Shrinkage of weight $i$ per step:

$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\, x_{t,i}}_{\textrm{$\ell_2$ through Adam}}
\qquad\textrm{vs.}\qquad
\underbrace{\eta\lambda\, x_{t,i}}_{\textrm{weight decay}}$$

::: {.d2l-note}
Scale $\sqrt{\hat{v}} = 10^{-3}$: the penalty removes
$0.0003/10^{-3} = 30\%$ per step, weight decay 0.03%.
:::
:::

::: {.slide title="AdamW Shrinks the Weights outside the Scaling"}
[Decoupled weight decay]{.kicker}

$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\qquad \hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \textrm{ from } \mathbf{g}_t \textrm{ alone}$$

- Coupled: add $\lambda\mathbf{x}_t$ to the gradient, then update the
  averages.
- Decoupled: shrink the weights, then take the plain Adam step.

Our version, one line on top of Adam, against the library's AdamW:

@!adamw-adamw-from-scratch-2

[Loshchilov and Hutter, 2019]{.small}
:::

::: {.slide title="Check: What Does the Penalty Do to a Weight of 0.045?"}
[Check]{.kicker}

The weight matrices of `TinyLM` start with entries of RMS 0.045 and loss
gradients of about $10^{-3}$. Adam with an $\ell_2$ penalty,
$\eta = 0.003$, $\lambda = 0.1$. What happens in the first steps?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The penalty's gradient $\lambda x = 0.0045$ exceeds the loss
gradient, so the penalized gradient is mostly penalty. Adam normalizes it:
the weight moves toward zero by about $\eta$ per step and needs
$0.045/0.003 = 15$ steps. $\lambda$ sets where the shrinkage stops, not
its speed.
:::
:::
:::

::: {.slide title="At λ = 0.1 the Penalty Costs More Than a Nat; AdamW Tracks No Decay"}
[Coupled and decoupled decay]{.kicker}

Three runs of `TinyLM` from one initialization, 1,000 steps:

@!adamw-one-number-two-meanings

Entropy of the byte frequencies: 3.17 nats.
:::

::: {.slide title="With Its Matrices Forty Times Smaller, the Model Predicts Byte Frequencies"}
[Coupled and decoupled decay]{.kicker}

| run | RMS of the weight matrices, step 0 → 1,000 | final loss |
|:--|:--|--:|
| no decay | 0.045 → 0.11 | about 1.7 |
| AdamW, $\lambda = 0.1$ | 0.045 → 0.10 | about 1.7 |
| Adam + $\ell_2$, $\lambda = 0.1$ | 0.045 → 0.003 | about 3.2 |

- Almost no input reaches the output: one prediction for every position,
  and the best one is the byte frequencies (3.17 nats).
- The shrinkage stopped where $\lambda|x| \approx 3 \times 10^{-4}$, the
  size of a loss gradient.
:::

::: {.slide title="AdamW's Weights Are a Leaky Average of Their Own Steps"}
[The timescale of decay]{.kicker}

With $\mathbf{u}_t = \hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$,
Adam's step before the learning rate:

$$\mathbf{x}_t = (1-\eta\lambda)^t\,\mathbf{x}_0 - \eta\sum_{s=0}^{t-1}(1-\eta\lambda)^{t-1-s}\,\mathbf{u}_s$$

- A step from $k$ steps ago counts with
  $(1-\eta\lambda)^k \approx e^{-\eta\lambda k}$.
- Window of the average: the **timescale** $1/(\eta\lambda)$ steps.

::: {.d2l-note}
$\eta = 0.003$, $\lambda = 0.1$: 3,333 steps. A 1,000-step run is 0.3
timescales, $e^{-0.3} = 0.74$: decay barely acts.
:::
:::

::: {.slide title="Decay and Steps Balance at an RMS of √(η/2λ)"}
[The equilibrium norm]{.kicker}

One entry $x$ of a weight matrix; its step $u_t$ has mean 0, mean square
$\sigma^2$, and is independent of $x$:

I. $\;x_{t+1} = (1-\eta\lambda)\,x_t - \eta\,u_t$ (the noise-ball
recursion, decay as the curvature)

II. $\;\mathbb{E}[x_{t+1}^2] = (1-\eta\lambda)^2\,\mathbb{E}[x_t^2] + \eta^2\sigma^2$

III. Fixed point:

$$\boxed{\mathbb{E}[x_\infty^2] \approx \frac{\eta\sigma^2}{2\lambda}}$$

$\sigma^2 \approx 1$: Adam divides each gradient by its RMS, and momentum
only spreads it over about ten steps.
:::

::: {.slide title="Without Decay the Weights Spread like a Random Walk"}
[The equilibrium norm]{.kicker}

$\lambda = 0$: nothing cancels the growth. Every step adds $\eta^2$ to the
mean square of an entry:

$$\textrm{RMS}_t = \sqrt{\textrm{RMS}_0^2 + \eta^2\, t}$$

`TinyLM`: the weight matrices start at RMS 0.045; at $\eta = 0.003$,

$$\sqrt{0.045^2 + 0.003^2 \cdot 2{,}000} = 0.14 \quad\textrm{after 2,000 steps.}$$

::: {.d2l-note}
Steps at a right angle to the weights lengthen them, by Pythagoras.
:::
:::

::: {.slide title="Check: Which Runs Settle within 2,000 Steps?"}
[Check]{.kicker}

AdamW on `TinyLM`, $\eta = 0.003$, 2,000 steps, $\lambda$ = 0.1, 0.5
and 1. Which runs reach their equilibrium, and at which RMS of the weight
matrices?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Timescales $1/(\eta\lambda)$: 3,333, 667 and 333 steps. The
first run is shorter than its timescale and cannot settle (its level
would be 0.122). The others settle, at $\sqrt{\eta/(2\lambda)} = 0.055$
and $0.039$.
:::
:::
:::

::: {.slide title="Measured: the Weights Settle after Two Timescales"}
[The equilibrium norm]{.kicker}

Two runs per $\lambda$; RMS of the weight matrices every 250 steps:

@!adamw-the-timescale-of-decay-1
:::

::: {.slide title="The Formula Has the Ratio Right and the Level 1.4 Times Too Low"}
[The equilibrium norm]{.kicker}

| $\lambda$ | RMS at step 2,000 | predicted | state | validation loss |
|--:|--:|--:|:--|:--|
| 0 | 0.14 | 0.14 | growing | reference |
| 0.1 | 0.116 | 0.122 | growing, 0.6 timescales | within 0.01 |
| 0.5 | 0.076 | 0.055 | level from step 1,500 | 0.1 worse |
| 1 | 0.054 | 0.039 | level from step 750 | 0.25 worse |

- $0.076/0.054 = \sqrt{2}$, as predicted. Both levels are about 1.4
  times the predicted ones: gradients keep part of their direction for
  many steps.

::: {.d2l-note .warn}
In 2,000 steps, slow decay does nothing and fast decay does harm.
:::
:::

::: {.slide title="Decay Holds the Relative Step at √(2ηλ)"}
[Choice of the coefficient]{.kicker}

- A matrix whose output is normalized matters only through its direction,
  so a step counts by its size relative to the weights, $\eta/\textrm{RMS}$.
- No decay: $0.003/0.045 = 0.067$ falls to $0.003/0.14 = 0.021$ in 2,000
  steps, then on as $1/\sqrt{t}$.
- With decay: $\eta/\sqrt{\eta/(2\lambda)} = \sqrt{2\eta\lambda}$ for
  every decayed matrix.

::: {.d2l-note .rule}
Without decay a run lowers its own learning rate on a schedule nobody
chose. In long single-pass runs, decay lowers the training loss.
:::

[Kosson et al., 2024; D'Angelo et al., 2024]{.small}
:::

::: {.slide title="Check: Halve η and Double λ. What Changes?"}
[Check]{.kicker}

A run with AdamW at $(\eta, \lambda) = (0.003, 0.5)$ is repeated at
$(0.0015, 1)$. What happens to the timescale, to the RMS at which the
weights settle, and to the relative step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta\lambda$ is unchanged: the same timescale of 667 steps
and the same relative step $\sqrt{2\eta\lambda}$. The RMS
$\sqrt{\eta/(2\lambda)}$ halves, which a normalized layer ignores. So when
$\eta$ changes, hold $\eta\lambda$.
:::
:::
:::

::: {.slide title="Choose λ by Its Timescale"}
[Choice of the coefficient]{.kicker}

| rule | example |
|:--|:--|
| count timescales, $T\eta\lambda$ for $T$ steps | 1,000 steps at $\eta\lambda = 3 \times 10^{-4}$: 0.3, inert |
| | 300,000 steps at $\eta = 3 \times 10^{-4}$, $\lambda = 0.1$: 9, active |
| searching over $\eta$: hold $\eta\lambda$ | set $\lambda = s/\eta$, search $s$ separately |
| longer run: longer timescale | a $\lambda$ tuned on a short trial is too large for the long run |

[Loshchilov and Hutter, 2019; Wang and Aitchison, 2024; Bergsma et al., 2025]{.small}
:::

::: {.slide title="Only the Weight Matrices Are Decayed"}
[Exempt parameters]{.kicker}

| group in `TinyLM` | parameters | decayed | reason |
|:--|--:|:--|:--|
| weight matrices | 425,984 | yes | their size only sets the relative step |
| biases, LayerNorm scales and shifts | 3,840 | no | they set scale and offset of each feature directly |
| embeddings | 40,960 | no | a row gets a gradient only when its token occurs; decay acts every step |

@!adamw-what-not-to-decay-1

[OLMo 2, 2025: decayed embeddings shrank, and the gradients through the LayerNorm after them grew]{.small}
:::

::: {.slide title="Adam's Two Averages Double the Memory of Weights and Gradients"}
[Optimizer state]{.kicker}

| per parameter, 32-bit floats | bytes |
|:--|--:|
| weight | 4 |
| gradient | 4 |
| Adam's averages $m$ and $v$ | 8 |
| **total** | **16** |

- `TinyLM`: under 4 MB for $m$ and $v$. A billion parameters: 8 GB of
  16 GB.
- Only trained parameters carry the state.
- A checkpoint needs $m$ and $v$ to resume.
:::

::: {.slide title="Check: Does a 1B-Parameter Model Fit a 24 GB GPU?"}
[Check]{.kicker}

A model with one billion parameters is trained with AdamW in 32-bit
floats. Do the weights, the gradients and the optimizer state fit on a
GPU with 24 GB?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16 \times 10^9$ bytes = 16 GB: yes, with 8 GB left for the
activations. Half of the 16 GB is $m$ and $v$.
:::
:::
:::

::: {.slide title="Once Decay Bypasses the Scaling, λ Means a Timescale"}
[Wrap-up]{.kicker}

- $\ell_2$ through Adam at $\lambda = 0.1$: weight matrices from RMS 0.045
  to 0.003, loss 3.2, the byte frequencies. AdamW: the loss of no decay.
- AdamW shrinks by $1 - \eta\lambda$ per step: timescale
  $1/(\eta\lambda)$, 3,333 steps at $\eta = 0.003$, $\lambda = 0.1$.
- Settled weights: RMS near $\sqrt{\eta/(2\lambda)}$ (measured 1.4 times
  higher), relative step $\sqrt{2\eta\lambda}$.
- Decay the matrices only, 90% of `TinyLM`; $m$ and $v$ are 8 of 16 bytes
  per parameter.

**Next:** Muon, a step chosen for a weight matrix as a whole.
:::
