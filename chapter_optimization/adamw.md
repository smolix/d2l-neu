# AdamW
:label:`sec_adamw`

<!-- WRITER: split from adam.md (structure 7). The opening below is the old "Decoupled Weight Decay" introduction; the new opening starts from the reader (every recipe since :numref:`sec_training_recipes` wrote "AdamW, weight decay 0.1"; what does 0.1 mean?). -->

Adam is almost always trained with weight decay, which shrinks the weights
toward zero at every step (:numref:`sec_weight_decay`). Under SGD it makes
no difference whether the shrinkage is written as a penalty
$\frac{\lambda}{2}\|\mathbf{x}\|^2$ added to the loss or applied to the
weights directly. Under Adam it does: the gradient of the penalty passes
through the per-coordinate scaling, and the same $\lambda$ then shrinks
every coordinate by a different amount. AdamW
:cite:`Loshchilov.Hutter.2019` applies the decay outside the scaling and
restores one meaning for $\lambda$.

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

## Penalty and Preconditioner

Adding the penalty to the loss adds $\lambda \mathbf{x}_t$ to the
gradient. Under SGD the update then splits into a uniform shrinkage and an
ordinary gradient step:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.
$$
:eqlabel:`eq_l2-is-decay`

Penalizing the loss *is* shrinking the weights, by the factor
$1 - \eta\lambda$ for every coordinate. Now feed the same penalized
gradient to Adam. Both moments are built from it, so the penalty passes
through the preconditioner too. If we treat $\sqrt{\hat{\mathbf{v}}_t}$ as
fixed, the shrinkage of coordinate $i$ in one step compares with that of
weight decay as follows:

$$
\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}}_{\textrm{$\ell_2$ penalty through Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\; x_{t,i}\vphantom{\frac{\eta}{\sqrt{\hat{v}_{t,i}}}}}_{\textrm{weight decay}}.
$$
:eqlabel:`eq_coupled-decay`

The penalty is rescaled per coordinate by the same factor that rescales
the loss gradient: a parameter with large gradients (large $\hat{v}_i$) is
barely decayed, while a parameter with small gradients is decayed
strongly. Whatever $\lambda$ we
choose, Adam changes its effective value by coordinate and over time.
:numref:`subsec_mdl-decoupled-weight-decay` follows the exact recurrence
and shows two coordinates with the same $\lambda$ whose effective decay
rates differ by a factor of 100.

AdamW *decouples* the two: the loss gradient goes through the
preconditioner, and the decay does not,

$$
\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$
:eqlabel:`eq_adamw`

with $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ computed from the plain
gradient $\mathbf{g}_t$ as in :eqref:`eq_adam-moments`. The shrinkage is
the uniform factor of :eqref:`eq_l2-is-decay` again. When the learning
rate follows a schedule $\eta_t$, the decay should follow it, so that the
weights stop shrinking when learning stops. :citet:`Loshchilov.Hutter.2019`
multiply the decay by the schedule's multiplier; the library implementations
we use multiply it by the whole learning rate, as in :eqref:`eq_adamw`, and
we follow them.

## AdamW from Scratch

The decoupled update adds one operation to Adam: shrink the weights, then
take the Adam step.

<!-- WRITER: AdamW is now built on the library's Adam (`torch.optim.Adam`; `optax.scale_by_adam`, the library's version of the `scale_by_adam` written in :numref:`sec_adam`), since the from-scratch Adam lives in the Adam notebook and was checked against the library there. The PyTorch class keeps its own decay coefficient (`self.weight_decay`) because `torch.optim.Adam`'s own `weight_decay` setting is the coupled form; the shrink uses each group's current learning rate, so it follows a schedule. -->

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
`torch.optim.Adam` also accepts a `weight_decay` argument. It adds
$\lambda \mathbf{x}$ to the gradient before the moments are updated, which
is the coupled form on the left of :eqref:`eq_coupled-decay`.
`torch.optim.AdamW` is the decoupled form.
:end_tab:

:begin_tab:`jax`
`optax.add_decayed_weights` adds $\lambda \mathbf{x}$ to whatever reaches
it. Placed after `scale_by_adam`, the decay bypasses the preconditioner.
Swapping the first two links,
`optax.chain(optax.add_decayed_weights(wd), scale_by_adam(), ...)`, gives
the coupled form instead: the position of one link decides which of the two
updates we get.
:end_tab:

We check it against the library's AdamW as before, at $\lambda = 0.1$:

<!-- WRITER: the check no longer calls the Adam section's `compare` (it lives in that notebook); it trains two copies of one linear model on the same least-squares data for 100 steps, ours and the library's AdamW, and prints only the largest difference between their weights: 0.0e+00 in both tabs (bit-identical, since the library performs the same operations in the same order). The first-step print is gone, so the sentence about the first step after the cell has no output behind it: drop it or state it as arithmetic. -->

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

The first step now also shrinks every weight by the factor $1 -
\eta\lambda$, so it is no longer exactly $\eta$, and the two
implementations again agree to within rounding.

## Coupled and Decoupled Decay Compared
:label:`subsec_adamw-grid`

How large is the difference on a real model? We train `TinyLM` for 1,000
steps at the learning rate $\eta = 0.003$ and the common choice $\lambda =
0.1$, once with the penalty inside Adam's gradient and once decoupled, and
also once without any decay for reference. All three runs start from the
same random initialization.

<!-- WRITER: two new cells before the runs: the byte-level data of :numref:`sec_test_models`, and `matrix_rms` (RMS of the entries of the weight matrices, embeddings excluded: the same matrices that the group code below decays) with `train_tracked` (trains TinyLM from a seed and records `matrix_rms` at the start and every `every` steps; the timescale experiment below uses it too). The run cell now prints, per run, the final training loss and the RMS of the weight matrices at initialization and at step 1,000. Measured, three development runs and one notebook run per tab (seeds differ between runs): initialization 0.0447 to 0.0448 in every run (the uniform initialization of TinyLM); no decay 0.109 to 0.111, final loss 1.69 to 1.74; AdamW 0.098 to 0.099 (decay has removed a tenth of the norm), final loss within 0.02 of the run without decay of the same seed, in either direction (1.706 against 1.702, 1.745 against 1.741 in the notebook runs); Adam + l2 0.0026 to 0.0028, forty times smaller than without decay, final loss 3.12 to 3.19, against a byte entropy of 3.17. The coupled run's smoothed loss is lowest at step 75 (3.03 to 3.07, about 0.1 below the entropy), rises to about 3.2 to 3.3 by step 250 and stays near the entropy: it never learned much to forget; its weights were held near zero from the start (development probe: the token embeddings shrink from RMS 1.0 to 0.10 as well, the vectors from 0.43 to 0.03). -->

```{.python .input #adamw-coupled-and-decoupled-decay-compared-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

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

The decoupled run trains almost as if there were no decay. The coupled run
ends more than a nat higher, and after an early dip its loss rises again.
To read its level, we compare it with a model that ignores the context and
predicts every byte from its frequency in the training text. The loss of
that model is the entropy of the byte frequencies:

```{.python .input #adam-coupled-and-decoupled-decay-compared-2}
%%tab pytorch, jax
freq = np.bincount(data.train, minlength=256) / len(data.train)
freq = freq[freq > 0]
print(f'entropy of the byte frequencies: {-(freq * np.log(freq)).sum():.2f}')
```

The coupled run ends close to this value: it has lost what it learned
about the context in its first hundred steps.
:eqref:`eq_coupled-decay` explains the rise. As training makes progress the
loss gradients shrink, $\sqrt{\hat{\mathbf{v}}}$ shrinks with them, and the
penalty, divided by that small number, grows by orders of magnitude. Once
$\lambda \mathbf{x}$ dominates the gradient, Adam normalizes it like any
other gradient, so every coordinate shrinks by about $\eta$ per step
whatever the value of $\lambda$: past this point, changing $\lambda$ hardly
changes the run.

<!-- WRITER: the decay grid (cells adamw-a-grid-twice-1..4, adam-coupled-and-decoupled-decay-compared-1, the 800-step runs on The Time Machine) is cut (ADAM-18, structure 7): from here to the end of this subsection every paragraph describes cells that no longer exist (`sweep`, `small`, `coupled_tbl`, `decoupled_tbl`, the two plots); the grid becomes an exercise, and the "column against grid" claim and the fixed-point paragraph go with it. -->

The coupled form needs a much smaller $\lambda$, which is easy to find. The
harder question is how the best $\lambda$ moves when the learning rate
changes: if it moves, the two settings must be searched together on a
grid instead of one after the other. We measure the held-out loss on a
grid of three learning rates and several values of $\lambda$ for each
form, from $10^{-4}$ to $10^{-2}$ for the coupled form and from $0.3$ to
$10$ for the decoupled one. In a run this short, decay helps only where
there is overfitting to prevent, and on WikiText-103 there is none: 800
steps read about 1% of the training text. For this grid we therefore make
the model overfit on purpose, on the kind of text that it would memorize:
the first 8,000 windows of *The Time Machine*, split into characters as in
:numref:`sec_text-sequence`. In 800 steps the model sees each window about
six times, and we measure the held-out loss on the 2,000 windows that
follow.

The function `sweep` trains one model per pair $(\eta, \lambda)$, with a
quiet `lr_sweep` for each $\lambda$, and returns the table of held-out
losses, one row per learning rate. The held-out loss is the mean
cross-entropy over the validation windows, the logarithm of
`lm_perplexity`.

We plot the held-out loss against $\lambda$, one curve per learning rate,
for the coupled form and then for the decoupled form, on a common vertical
scale.

In this grid the AdamW curves have their minimum at about the same
$\lambda$, 3 or 10, for all three learning rates: changing $\eta$ moves a curve up
or down but hardly sideways. The coupled form works with values of
$\lambda$ about three orders of magnitude smaller. At its largest
$\lambda$ the coupled curves end at nearly the same loss for every
learning rate, the saturation described above. Below that, the best
$\lambda$ differs among the three learning rates. Tuned fully, the
two forms reach similar held-out losses. They differ in the shape of the
search: a column for AdamW, a whole grid for the coupled form.

A fixed point explains why the best $\lambda$ of AdamW stays put here.
Where the weights stop moving on average, the shrinkage in
:eqref:`eq_adamw` balances the Adam step, $\eta\lambda\,\mathbf{x} =
-\eta\,\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon)$. The learning
rate cancels: $\lambda$ alone selects the solution near which the run
settles, and $\eta$ sets how fast the run gets there, through the factor
$1 - \eta\lambda$ per step. Near the best $\lambda$, every run of the grid
lasts longer than $1/(\eta\lambda)$ steps, so every run gets close. The
argument depends on the parameterization, in which both terms scale with
$\eta$, and on the regime, as the next subsection shows.

## Weight Decay in Large-Scale Training
:label:`subsec_wd-at-scale`

Our grid used weight decay against overfitting, the role it played in
:numref:`sec_weight_decay`, where it acts as a prior that pulls the model
toward small weights. Large language models are trained for about one pass
over data that they never see twice, so there is little overfitting to
fight, yet most published recipes still use $\lambda = 0.1$. The decay
does a different job there
:cite:`DAngelo.Andriushchenko.Varre.ea.2024,Kosson.Messmer.Jaggi.2024`.

The mechanism runs through the norms of the weights. A weight matrix whose
output passes through a normalization layer is *scale-invariant*:
multiplying it by a constant does not change the network's function,
because the normalization divides the constant out. Only the direction of
such a matrix matters, and an update of fixed size turns it by an angle
roughly proportional to the update divided by the matrix's norm. In a
pre-norm transformer this holds only approximately, since the residual
stream carries each block's output around the next normalization, but it
describes enough of the parameters to set the dynamics. Gradient noise
makes the norms grow, decay makes them shrink, and the two settle into an
equilibrium in which each weight turns by a roughly constant angle per step
:cite:`Kosson.Messmer.Jaggi.2024`. Through this equilibrium the product
$\eta\lambda$ sets the *effective* learning rate, the angle per step, and
sets it alike for all layers. This is a large part of why decay in large
language-model runs lowers the training loss instead of trading training
loss for held-out loss :cite:`DAngelo.Andriushchenko.Varre.ea.2024`. Small
weights also keep low-precision training in a safe numerical range.

With decay multiplied by the learning rate, each step multiplies the
weights by $1 - \eta\lambda$, so a contribution made $k$ steps ago has
shrunk by the factor $(1 - \eta\lambda)^k \approx e^{-\eta\lambda k}$. An
AdamW iterate is therefore approximately an average of its recent updates
over a horizon of $1/(\eta\lambda)$ steps: with $\eta = 0.003$ and $\lambda
= 0.1$, about 3,300 steps. Measured in passes over a dataset of $D$
examples with batch size $B$, the horizon is $\tau = B/(\eta\lambda D)$.
:citet:`Bergsma.Dey.Gosal.ea.2025b` found that the best $\tau$ stays
roughly constant as models grow at a fixed ratio of training tokens to
parameters, and follows a power law in that ratio otherwise, so that
$\lambda$ can be set from $B$, $D$, and the model size rather than tuned
again. The horizon also explains why $\eta$ and $\lambda$ act on the
equilibrium through their product.

<!-- WRITER: NEW experiment (worklist 7.3), the timescale of decay; this subsection becomes "The Timescale of Decay" and keeps :label:`subsec_wd-at-scale`. The cell trains TinyLM with AdamW at eta = 3e-3 for 2,000 steps with lambda in {0, 0.1, 0.5, 1}, decaying every parameter as in the three-run comparison, two runs per lambda (two seeds, shared across lambda), and prints per lambda the RMS of the weight matrices (mean of the two runs) at steps 0, 250, ..., 2,000, the predicted equilibrium sqrt(eta / (2 lambda)), and the validation loss of each run; then plots the RMS curves. Timescales 1/(eta lambda): infinite, 3,333, 667 and 333 steps.
Prediction (the two-line equilibrium of Exercise 9 in the body): with the update u_t orthogonal to x_t, ||x_{t+1}||^2 = (1 - eta lambda)^2 ||x_t||^2 + eta^2 ||u_t||^2, so the norm settles where 2 eta lambda ||x||^2 = eta^2 ||u||^2, an RMS per entry of sqrt(eta / (2 lambda)) * RMS(u). Which u: over many steps the momentum average m_hat passes every gradient on in full, so the steps add up like the normalized gradients g / sqrt(v_hat), whose RMS is about 1 because v_hat estimates E[g^2] (measured in PyTorch at step 2,000: 0.82 to 1.13 over six runs), not like Adam's one-step update m_hat / sqrt(v_hat), whose RMS is only 0.16 to 0.23 (the value the gradient-scale cell of :numref:`sec_adam` prints, about 0.2; with it the prediction would be five times too small). Hence the printed prediction sqrt(eta / (2 lambda)): 0.122, 0.055, 0.039.
Measured (development runs, three PyTorch seeds at 0 / 0.1 / 0.5, two PyTorch seeds at 0.5 / 1 / 2, two JAX seeds at 0 / 0.1 / 0.5 / 1, plus the notebook runs; seed-to-seed differences of the RMS at any checkpoint at most 0.002):
- initialization 0.0447 in every run.
- lambda = 0: grows throughout, 0.068 at step 250, 0.110 at 1,000, 0.142 to 0.144 at 2,000, still rising by 0.007 per 250 steps: without decay nothing stops the growth.
- lambda = 0.1: 0.116 to 0.117 at 2,000, still rising by 0.003 per 250 steps; it has not settled, because 2,000 steps are less than its timescale of 3,333.
- lambda = 0.5: 0.073 to 0.074 at 1,000, then 0.0755 to 0.0766 from step 1,500 on: settled after about two timescales (667 steps each).
- lambda = 1: rises to 0.054 by step 750 and stays at 0.053 to 0.055 (slight drift down).
- So the level scales as 1/sqrt(lambda): 0.0765 / 0.053 = 1.44 against sqrt(2) = 1.41 (and lambda = 2, development only, ended at 0.035 to 0.037, half of lambda = 0.5's as predicted); the formula's level is low by a constant factor of about 1.4 (0.076 against 0.055, 0.053 against 0.039), because successive gradients are not independent and part of each step points along x.
- Notebook runs (PyTorch / JAX, mean RMS at 2,000; validation losses): 0: 0.142 / 0.142, 1.563 1.564 / 1.566 1.560; 0.1: 0.116 / 0.116, 1.574 1.572 / 1.575 1.571; 0.5: 0.076 / 0.076 (flat from 1,500), 1.660 1.665 / 1.660 1.663; 1: 0.054 / 0.053, 1.821 1.828 / 1.825 1.821.
- Validation loss over all runs: lambda = 0 1.560 to 1.570, lambda = 0.1 1.570 to 1.577 (within the spread of :numref:`sec_test_models`: no effect in this run), lambda = 0.5 1.656 to 1.670 (0.09 worse), lambda = 1 1.82 to 1.83 (0.25 worse). In a 2,000-step single pass, decay strong enough to settle within the run only costs; 0.1 is too slow to act at all. -->

```{.python .input #adamw-the-timescale-of-decay-1}
%%tab pytorch
lr, num_steps, every = 3e-3, 2000, 250
seeds = [random.getrandbits(31) for _ in range(2)]
table = {}
print(f'{"λ":<5}' + ''.join(f'{t:>6}' for t in range(0, num_steps + 1, every))
      + f'{"predicted":>10}  validation loss')
for wd in [0, 0.1, 0.5, 1]:
    runs = [train_tracked(lambda params: torch.optim.AdamW(
        params, lr, weight_decay=wd), data, seed, num_steps, every)
            for seed in seeds]
    table[wd] = [sum(rms) / len(rms) for rms in zip(*(r[1] for r in runs))]
    val = [math.log(d2l.lm_perplexity(r[2], data.val_dataloader()))
           for r in runs]
    predicted = f'{math.sqrt(lr / (2 * wd)):.3f}' if wd else '-'
    print(f'{wd:<5g}' + ''.join(f'{v:6.3f}' for v in table[wd])
          + f'{predicted:>10}  ' + ' '.join(f'{v:.3f}' for v in val))
d2l.plot(list(range(0, num_steps + 1, every)), list(table.values()), 'step',
         'RMS of the weight matrices', legend=[f'λ = {wd:g}' for wd in table])
```

```{.python .input #adamw-the-timescale-of-decay-1}
%%tab jax
lr, num_steps, every = 3e-3, 2000, 250
seeds = [d2l.get_seed() for _ in range(2)]
table = {}
print(f'{"λ":<5}' + ''.join(f'{t:>6}' for t in range(0, num_steps + 1, every))
      + f'{"predicted":>10}  validation loss')
for wd in [0, 0.1, 0.5, 1]:
    runs = [train_tracked(optax.adamw(lr, weight_decay=wd), data, seed,
                          num_steps, every) for seed in seeds]
    table[wd] = [sum(rms) / len(rms) for rms in zip(*(r[1] for r in runs))]
    val = [math.log(d2l.lm_perplexity(r[2], data.val_dataloader()))
           for r in runs]
    predicted = f'{math.sqrt(lr / (2 * wd)):.3f}' if wd else '-'
    print(f'{wd:<5g}' + ''.join(f'{v:6.3f}' for v in table[wd])
          + f'{predicted:>10}  ' + ' '.join(f'{v:.3f}' for v in val))
d2l.plot(list(range(0, num_steps + 1, every)), list(table.values()), 'step',
         'RMS of the weight matrices', legend=[f'λ = {wd:g}' for wd in table])
```

Keeping the product fixed when $\eta$ changes is the opposite of what our
grid showed. The two results describe different regimes. A scale-invariant
matrix has no fixed point of the kind found in the previous subsection: its
gradient is orthogonal to it, so decay and updates balance only in the norm,
which the network ignores. What remains is the angle per step, set by
$\eta\lambda$. When most of the decayed weights are of this kind, the
quantity to keep when $\eta$ is retuned is the product, which is also how
:citet:`Loshchilov.Hutter.2019` parameterize the decay: they multiply it
by the schedule only, not by $\eta$, and on image classifiers with batch
normalization they found its best value nearly independent of the
learning rate. When decay mainly fights overfitting, as in our grid,
$\lambda$ itself selects the solution. The exercise on the fixed-product
ridge measures which of the two holds for `TinyLM`.

## Parameters Exempt from Decay
:label:`subsec_adamw-what-not-to-decay`

The scale-invariance argument covers only matrices whose output is
normalized, and standard practice decays only the matrices among the three
parameter groups of `TinyLM`. The other two groups are exempt for different
reasons. Biases and LayerNorm scales and shifts are few, and they set the
scale and offset of each feature directly, so shrinking them toward zero
distorts the function instead of controlling the dynamics. Biases were
already exempt in :numref:`sec_weight_decay`. Embedding rows are sparse: a
row receives a gradient only when its token occurs, but the decay acts at
every step, so the rows of rare tokens are decayed without any signal to
balance it. OLMo 2 traced a training instability to this: decay shrank the
embeddings until the LayerNorm after them, which divides by their norm,
amplified their gradients :cite:`OLMo.2025`.

The implementation uses two parameter groups, one decayed and one not, and
the split is the one we counted for `TinyLM`.

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
with its own settings; settings not given in a group fall back to the
keyword arguments.
:end_tab:

:begin_tab:`jax`
`optax.adamw` takes a `mask`, a function that maps the parameter tree to a
tree of booleans; only the parameters marked `True` are decayed.
:end_tab:

<!-- WRITER: the 2,000-step run of this configuration (cell adamw-what-not-to-decay-2) is cut (ADAM-20): the rest of this paragraph and the next describe it. The group cell above no longer reuses the comparison's initialization or Adam's tuned rate from the Adam notebook: it builds a fresh TinyLM and uses eta = 3e-3; it still prints 425,984 of 470,784 decayed. -->

About 90% of the parameters are decayed. We train this configuration for
2,000 steps at Adam's tuned learning rate, from the same initialization as
the runs of the language-model comparison, and compare it with Adam's
tuned run there, which used no decay:

AdamW ends within a few percent of the run without decay, in either
direction. In
2,000 steps the model reads about 3% of the training text and does not
overfit, so the decay, whose horizon $1/(\eta\lambda)$ is longer than the
run, has nothing to prevent and cannot help.
This is the configuration in which Adam is used today: AdamW, decay on the
matrices only, and $\lambda$ chosen together with the schedule. What
remains is its cost in memory.

## Optimizer-State Memory
:label:`subsec_adamw-memory`

Adam and AdamW store two numbers per parameter, $m$ and $v$, on top of the
weights and the gradients. At the size of `TinyLM` that is invisible; at
the size of a language model it decides what fits on a device. <!-- WRITER: the memory cell (adamw-optimizer-state-and-memory, the bytes-per-parameter table and the 7B column) is cut (ADAM-21); this subsection becomes one paragraph without a cell (8 bytes per parameter for m and v in fp32, 16 with weights and gradients, consistent with :numref:`sec_memory_precision`). --> We count
the bytes per parameter for three common setups and multiply by the
parameter count $n$ of `TinyLM`, which `d2l.num_params` returns.

In full fp32 training, a parameter costs 4 bytes each for the weight, the
gradient, $m$, and $v$: 16 bytes, half of them for $m$ and $v$. Mixed
precision does not reduce this. The forward and backward passes use bf16
weights and gradients, 2 bytes each, but stable training keeps an fp32
*master copy* of the weights for the optimizer to update, so the total is
again 16 bytes. Many setups also keep an fp32 copy of the gradients to
accumulate them over several minibatches, which brings the total to 20
(:numref:`sec_training_systems`).

For `TinyLM` the total is a few megabytes. The same arithmetic for a model
with 7 billion parameters gives 112 to 140 GB before a single activation
is stored, more than an 80-GB accelerator holds. Of the 20 bytes, 12 are
the fp32 master copy, $m$, and $v$, which only the optimizer reads. This is
why optimizer state is the first target when memory runs out. It can be
stored in fewer bytes or split across the devices of a data-parallel group
instead of copied to each, the systems techniques of
:numref:`chap_performance`.

## Summary

<!-- WRITER: new Summary (the AdamW sentences of the old adam.md Summary belong here). -->

## Exercises

<!-- WRITER: names the exercises below use that no longer exist after the code step: `decoupled_tbl`, `sweep` and the grid of :numref:`subsec_adamw-grid` (Exercise 2; the cut grid becomes the new decay-grid exercise), the `setups` table of the memory cell (Exercise 4). Available in this notebook: `data`, `matrix_rms`, `train_tracked`, `AdamW` / `adamw`, `coupled` (JAX). -->

1. [code] **Saturation of coupled decay.** The experiment of
   :numref:`subsec_mdl-decoupled-weight-decay` starts two parameters at
   $x_i = 1$ and gives them pure-noise loss gradients
   $g_i \sim \mathcal{N}(0, \sigma_i^2)$ with $\sigma = (10, 0.1)$. At
   $\eta = 10^{-3}$ and $\lambda = 0.1$, Adam with an $\ell_2$ penalty
   decays them at the rates $\eta\lambda/\sigma_i$ that
   :eqref:`eq_coupled-decay` predicts.
    1. In the opposite limit $\sigma_i = 0$, the penalized gradient is
       $\lambda x_{t,i}$ alone. Show that for $\epsilon = 0$ the trajectory
       of $x_{t,i}$ under Adam with an $\ell_2$ penalty does not depend on
       $\lambda > 0$.
    1. Extend the experiment to
       $\sigma_i \in \{10, 1, 0.1, 0.01, 0.001\}$ at $\eta = 10^{-3}$ and
       record $|x_i|$ after 1,000 steps of Adam with an $\ell_2$ penalty
       for each $\lambda \in \{0.1, 1, 10\}$. Which coordinates no longer
       respond to $\lambda$, and how does the transition between the two
       regimes depend on $\sigma_i$ and $\lambda$?

1. [code] **The fixed-product ridge.** ● Both library implementations
   used in this section multiply the decay by the full learning rate, so
   the per-step shrinkage is $s = \eta\lambda$. :numref:`subsec_wd-at-scale`
   argues that at scale $\eta$ and $\lambda$ act through this product,
   which sets an averaging horizon of $1/s$ steps.
   :citet:`Loshchilov.Hutter.2019` instead scale the decay by the schedule
   multiplier only, so that under a constant schedule their decay
   coefficient is $s$ itself.
    1. Each column of a grid can fix $s$ instead of $\lambda$. Use
       `decoupled_tbl` to predict where the best entry of each row falls on
       the shrinkage axis `[1e-3, 3e-3, 1e-2, 3e-2]`. Then rerun the
       decoupled grid of :numref:`subsec_adamw-grid` with `sweep` on this
       axis, passing the decay coefficient to the optimizer as `wd / lr`.
       Does the best column stay put?
    1. How constant is the held-out loss down the column
       $s = 3 \times 10^{-3}$? Reconcile the answer with the argument of
       :numref:`subsec_wd-at-scale`, and state which quantity, $\lambda$ or
       $s$, the decay axis of a grid should carry when the learning rate is
       retuned in this setting.

1. [code] **Rotational equilibrium.** ● Following
   :citet:`Kosson.Messmer.Jaggi.2024`, let $\mathbf{x} \in \mathbb{R}^d$
   be a scale-invariant weight vector in the sense of
   :numref:`subsec_wd-at-scale`, trained by AdamW with $\beta_1 = 0$:
   $$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{u}_t,
   \qquad \mathbf{u}_t = \frac{\mathbf{g}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.$$
    1. Show that the loss gradient of a scale-invariant weight vector is
       orthogonal to $\mathbf{x}$. Assume from here on that $\mathbf{u}_t$
       is orthogonal to $\mathbf{x}_t$ as well and that
       $\|\mathbf{u}_t\|^2 \approx d$. Derive, to first order in
       $\eta\lambda$, the equilibrium norm at which decay balances the
       growth caused by the updates.
    1. Derive the angle through which $\mathbf{x}$ turns per step at
       equilibrium. How do the equilibrium norm and the angle each depend
       on $\eta$ and $\lambda$? Relate the result to the claim of
       :numref:`subsec_wd-at-scale` that $\eta$ and $\lambda$ act through
       their product.
    1. Check both predictions in NumPy with $d = 1{,}000$ over 20,000
       steps from a random unit vector, using pure-noise gradients
       projected orthogonal to $\mathbf{x}_t$, at $(\eta, \lambda) \in
       \{(10^{-3}, 0.1), (3 \times 10^{-3}, 0.1), (10^{-3}, 0.3)\}$.

1. **Activations and optimizer state.** Extend the accounting of
   :numref:`subsec_adamw-memory` beyond the `setups` table.
    1. For `TinyLM` with batch size $B$ and sequence length $T$, estimate
       the bf16 activation memory that must be stored for the backward pass
       (per block: the two normalization outputs, the attention inputs and
       outputs, and the MLP hidden layer). At what batch size do
       activations overtake the optimizer state?
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

AdamW<br>**penalty and preconditioner · decoupled weight decay · the timescale of decay · optimizer memory**
:::
:::

<!-- WRITER: new deck (slide guide). The slides below moved from the Adam deck. -->

::: {.slide title="Under Adam, an ℓ2 Penalty Is Not Weight Decay"}
[Decoupled weight decay]{.kicker}

SGD: the penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ *is* decay,

$$\mathbf{x}_t - \eta(\mathbf{g}_t + \lambda\mathbf{x}_t) = (1-\eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.$$

Adam: the penalty passes through the preconditioner (schematically),

$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\, x_{t,i}}_{\textrm{$\ell_2$ through Adam}}
\qquad\textrm{vs.}\qquad
\underbrace{\eta\lambda\, x_{t,i}}_{\textrm{weight decay}}$$

- Large gradients: almost no decay. Small gradients: strong decay.
:::

::: {.slide title="AdamW Shrinks the Weights, Then Takes the Adam Step"}
$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\qquad \hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \textrm{ from } \mathbf{g}_t \textrm{ alone}$$

- Coupled: add $\lambda\mathbf{x}_t$ to the gradient, *then* update
  the moments.
- Decoupled: update the moments from $\mathbf{g}_t$, and shrink the
  weights separately.
- From scratch this is one line on top of Adam, and it matches the
  library's AdamW to rounding error.
:::

::: {.slide title="At λ = 0.1, Coupled ℓ2 Costs Over a Nat; AdamW Tracks No Decay"}
@!adamw-one-number-two-meanings

- Coupled ends near 3.2 nats, the loss of byte frequencies alone: the
  early context is lost.
- Progress shrinks $\sqrt{\hat{\mathbf{v}}}$ until $\lambda\mathbf{x}$
  dominates; Adam then normalizes it, and every coordinate shrinks by
  about $\eta$ per step, whatever $\lambda$ is.
:::

::: {.slide title="Check: What Decay Does λ = 0.1 Give Where √v̂ = 10⁻³?"}
[Check]{.kicker}

Adam with an $\ell_2$ penalty, $\eta = 0.003$, $\lambda = 0.1$, on a
coordinate whose loss gradients have $\sqrt{\hat{v}_{t,i}} = 10^{-3}$.
What fraction of $x_{t,i}$ does the penalty remove per step, by the
formula with $\sqrt{\hat{v}}$ held fixed? And by AdamW?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta\lambda/\sqrt{\hat{v}} = 0.003 \cdot 0.1 / 10^{-3} = 0.3$,
30% per step, a thousand times AdamW's $\eta\lambda = 3 \times 10^{-4}$.
For any weight with $|x| > 0.01$, though, $\lambda|x|$ exceeds the loss
gradient, so the penalty dominates $\hat{v}$ itself and Adam normalizes
it: the coordinate moves about $\eta$ per step toward zero. That is the
saturation.
:::
:::
:::

::: {.slide title="In This Grid, AdamW's Best λ Barely Moves with η"}
Held-out loss after 800 steps on 8,000 windows of *The Time Machine*
(about six passes, overfitting on purpose), one curve per $\eta$:

<!-- WRITER: the grid cells this slide showed are cut (the grid is now an exercise); replace this slide with the timescale experiment (cell adamw-the-timescale-of-decay-1, see the note above that cell). -->

- AdamW: $\eta$ moves each curve up or down, hardly sideways. At the
  fixed point $\lambda\mathbf{x} = -\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon)$,
  $\eta$ cancels.
- Coupled: $\lambda$ about three orders of magnitude smaller, saturated
  at the largest values.
:::

::: {.slide title="At Scale, Decay Sets a Horizon of 1/(ηλ) Steps"}
- A matrix followed by normalization is scale-invariant. Noise grows its
  norm, decay shrinks it: a constant turning angle per step, set by
  $\eta\lambda$.
- Horizon $1/(\eta\lambda)$: at $\eta = 0.003$, $\lambda = 0.1$, about
  3,300 steps; in passes, $\tau = B/(\eta\lambda D)$.
- So at scale, keep $\eta\lambda$ when $\eta$ changes; our short,
  overfitting grid kept $\lambda$.
- Decay the matrices only (90% of `TinyLM`). Exempt embeddings (sparse
  gradients, decay every step), biases, and LayerNorm.

[D'Angelo et al. 2024; Kosson et al. 2024; Bergsma et al. 2025]{.small}
:::

::: {.slide title="Adam Costs 16 to 20 Bytes per Parameter"}
[Memory]{.kicker}

<!-- WRITER: the memory cell is cut; the slide states the arithmetic without it. -->

- 7B parameters: 112 to 140 GB before any activation.
- 12 of the 20 bytes (fp32 master, $m$, $v$) are read only by the
  optimizer.
- Store less (Adafactor: row and column sums of $v$; 8-bit states) or
  shard the state across devices (ZeRO).
:::

::: {.slide title="Check: Does a 1B-Parameter Model Fit a 24 GB GPU?"}
[Check]{.kicker}

Training a 1B-parameter model with AdamW in fp32. Do the weights,
gradients, and optimizer state fit on a 24 GB GPU?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16 \times 10^9$ bytes = 16 GB: yes, with 8 GB left for
activations. Half of the 16 GB is $m$ and $v$.
:::
:::
:::
