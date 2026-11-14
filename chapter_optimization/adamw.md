# AdamW
:label:`sec_adamw`

<!-- WRITER: split from adam.md (structure 7). The opening below is the old "Decoupled Weight Decay" introduction; the new opening starts from the reader (every recipe since :numref:`sec_training_recipes` wrote "AdamW, weight decay 0.1"; what does 0.1 mean?). -->
<!-- CODER (AdamW step): the cells below moved byte for byte from adam.md and still use names defined there: the from-scratch `Adam` (PyTorch) and `scale_by_adam` (JAX), `compare`, `data`, `lm_seed`, `best_adam`, `adam_lm`. This notebook does not run until they are defined here. -->

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
import random
from functools import partial
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

```{.python .input #adamw-adamw-from-scratch-1}
%%tab pytorch
class AdamW(Adam):
    """Adam with decoupled weight decay: shrink the weights, then step."""
    def __init__(self, params, lr=1e-3, weight_decay=0.01):
        super().__init__(params, lr)
        self.defaults['weight_decay'] = weight_decay  # For later groups
        for group in self.param_groups:
            group.setdefault('weight_decay', weight_decay)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
        super().step()
```

```{.python .input #adamw-adamw-from-scratch-1}
%%tab jax
def adamw(learning_rate, weight_decay):
    return optax.chain(scale_by_adam(),
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

```{.python .input #adamw-adamw-from-scratch-2}
%%tab pytorch
compare(partial(AdamW, weight_decay=0.1),
        partial(torch.optim.AdamW, weight_decay=0.1), lr=0.01)
```

```{.python .input #adamw-adamw-from-scratch-2}
%%tab jax
compare(adamw(0.01, weight_decay=0.1),
        optax.adamw(0.01, weight_decay=0.1), lr=0.01)
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

```{.python .input #adamw-one-number-two-meanings}
%%tab pytorch
seed = random.getrandbits(31)
curves = {}
for name, make_optimizer in [
        ('no decay',
         lambda model, lr: torch.optim.Adam(model.parameters(), lr)),
        ('Adam + ℓ2', lambda model, lr: torch.optim.Adam(
            model.parameters(), lr, weight_decay=0.1)),
        ('AdamW', lambda model, lr: torch.optim.AdamW(
            model.parameters(), lr, weight_decay=0.1))]:
    print(f'{name}: ', end='')
    curves[name] = d2l.lr_sweep(lambda: d2l.TinyLM(len(data.vocab)),
                                make_optimizer, data, [0.003], seed,
                                num_steps=1000)[0][0.003]
d2l.plot(list(range(0, 1000, 25)), [d2l.smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

```{.python .input #adamw-one-number-two-meanings}
%%tab jax
def coupled(lr, wd):  # The penalty's gradient enters Adam's moments
    return optax.chain(optax.add_decayed_weights(wd), optax.adam(lr))

seed = d2l.get_seed()
curves = {}
for name, make_tx in [('no decay', optax.adam),
                      ('Adam + ℓ2', lambda lr: coupled(lr, 0.1)),
                      ('AdamW', lambda lr: optax.adamw(lr, weight_decay=0.1))]:
    print(f'{name}: ', end='')
    curves[name] = d2l.lr_sweep(
        lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs),
        make_tx, data, [0.003], seed, num_steps=1000)[0][0.003]
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

```{.python .input #adamw-a-grid-twice-1}
%%tab pytorch, jax
small = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                        num_train=8000, num_val=2000)
```

The function `sweep` trains one model per pair $(\eta, \lambda)$, with a
quiet `lr_sweep` for each $\lambda$, and returns the table of held-out
losses, one row per learning rate. The held-out loss is the mean
cross-entropy over the validation windows, the logarithm of
`lm_perplexity`.

```{.python .input #adamw-a-grid-twice-2}
%%tab pytorch
lrs = [1e-3, 3e-3, 1e-2]
wds_coupled, wds_decoupled = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.3, 1, 3, 10]

def sweep(make_optimizer, data, lrs, wds, seed, num_steps=800):
    """Held-out loss for each pair (lr, wd), one row per learning rate."""
    held_out = lambda model: math.log(d2l.lm_perplexity(
        model, data.val_dataloader()))
    columns = [d2l.lr_sweep(lambda: d2l.TinyLM(len(data.vocab)),
                        lambda model, lr: make_optimizer(model, lr, wd),
                        data, lrs, seed, num_steps, evaluate=held_out,
                        verbose=False)[1]
               for wd in wds]
    table = [[column[lr][-1] for column in columns] for lr in lrs]
    for lr, row in zip(lrs, table):
        print(f'lr {lr:g}: ' + ' '.join(f'{v:.2f}' for v in row))
    return table

seed = random.getrandbits(31)
coupled_tbl = sweep(lambda model, lr, wd: torch.optim.Adam(
    model.parameters(), lr, weight_decay=wd), small, lrs, wds_coupled, seed)
```

```{.python .input #adamw-a-grid-twice-2}
%%tab jax
lrs = [1e-3, 3e-3, 1e-2]
wds_coupled, wds_decoupled = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.3, 1, 3, 10]

def sweep(make_tx, data, lrs, wds, seed, num_steps=800):
    """Held-out loss for each pair (lr, wd), one row per learning rate."""
    held_out = lambda model: math.log(d2l.lm_perplexity(
        model, data.val_dataloader()))
    columns = [d2l.lr_sweep(
        lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs),
        lambda lr: make_tx(lr, wd), data, lrs, seed,
        num_steps, evaluate=held_out, verbose=False)[1]
               for wd in wds]
    table = [[column[lr][-1] for column in columns] for lr in lrs]
    for lr, row in zip(lrs, table):
        print(f'lr {lr:g}: ' + ' '.join(f'{v:.2f}' for v in row))
    return table

seed = d2l.get_seed()
coupled_tbl = sweep(coupled, small, lrs, wds_coupled, seed)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab pytorch
decoupled_tbl = sweep(lambda model, lr, wd: torch.optim.AdamW(
    model.parameters(), lr, weight_decay=wd), small, lrs, wds_decoupled,
    seed)
```

```{.python .input #adamw-a-grid-twice-3}
%%tab jax
decoupled_tbl = sweep(lambda lr, wd: optax.adamw(lr, weight_decay=wd),
                      small, lrs, wds_decoupled, seed)
```

We plot the held-out loss against $\lambda$, one curve per learning rate,
for the coupled form and then for the decoupled form, on a common vertical
scale.

```{.python .input #adamw-a-grid-twice-4}
%%tab pytorch, jax
values = [v for table in (coupled_tbl, decoupled_tbl)
          for row in table for v in row]
ylim = (min(values) - 0.1, max(values) + 0.1)
legend = [f'lr {lr:g}' for lr in lrs]
d2l.plot(wds_coupled, coupled_tbl, '$\\lambda$ (Adam + $\\ell_2$)',
         'held-out loss', legend=legend, ylim=ylim, xscale='log')
```

```{.python .input #adam-coupled-and-decoupled-decay-compared-1}
%%tab pytorch, jax
d2l.plot(wds_decoupled, decoupled_tbl, '$\\lambda$ (AdamW)',
         'held-out loss', legend=legend, ylim=ylim, xscale='log')
```

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
torch.manual_seed(lm_seed)  # The initialization of the comparison's runs
model = d2l.TinyLM(len(data.vocab))
decay = [p for name, p in model.named_parameters()
         if p.ndim == 2 and 'emb' not in name]
no_decay = [p for name, p in model.named_parameters()
            if p.ndim != 2 or 'emb' in name]
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}], lr=best_adam)
n_decay = sum(p.numel() for p in decay)
n = n_decay + sum(p.numel() for p in no_decay)
print(f'decayed: {n_decay} of {n} parameters')
```

```{.python .input #adamw-what-not-to-decay-1}
%%tab jax
model = d2l.TinyLM(len(data.vocab),
                   rngs=nnx.Rngs(lm_seed))  # As in the comparison

def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

optimizer = nnx.Optimizer(
    model, optax.adamw(best_adam, weight_decay=0.1, mask=decay_mask),
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

About 90% of the parameters are decayed. We train this configuration for
2,000 steps at Adam's tuned learning rate, from the same initialization as
the runs of the language-model comparison, and compare it with Adam's
tuned run there, which used no decay:

```{.python .input #adamw-what-not-to-decay-2}
%%tab pytorch, jax
losses = d2l.train_steps(model, data, optimizer, num_steps=2000)
d2l.plot(list(range(0, 2000, 25)), [d2l.smooth(losses)],
         'step', 'training loss')
print(f'perplexity: AdamW {math.exp(d2l.final_loss(losses)):.2f}, '
      f'Adam without decay {math.exp(d2l.final_loss(adam_lm[best_adam])):.2f}')
```

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
the size of a language model it decides what fits on a device. We count
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

```{.python .input #adamw-optimizer-state-and-memory}
%%tab pytorch, jax
n = d2l.num_params(model)  # The TinyLM just trained
setups = {'fp32 weights, grads, m, v': (4, 4, 8),
          'bf16 weights + grads, fp32 master, m, v': (2, 2, 12),
          'the same + fp32 gradient accumulator': (2, 6, 12)}
print(f'{"":<42}{"B/param":>8}{"TinyLM":>10}{"7B model":>10}')
for name, (w, g, s) in setups.items():
    per = w + g + s
    print(f'{name:<42}{per:>8}{n * per / 1e6:>8.1f}MB'
          f'{7e9 * per / 1e9:>8.0f}GB')
```

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

::: {.cols}
::: {.col}
@!adamw-a-grid-twice-4
:::

::: {.col}
@!adam-coupled-and-decoupled-decay-compared-1
:::
:::

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

@!adamw-optimizer-state-and-memory

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
