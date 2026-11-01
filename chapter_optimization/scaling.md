# Scaling Up
:label:`sec_scaling`

Every experiment in this chapter so far chose its learning rate with a
sweep: a few dozen short runs, each lasting seconds to minutes on one GPU. A
model with billions of parameters, trained for months on thousands of
accelerators, allows no such sweep. Even a single extra full run may be
too expensive. Large-scale training therefore tunes a small proxy model and
reuses the hyperparameters it finds for the large one. If they do not carry
over, the large run diverges or trains slowly, and the error shows only after
much of the budget is spent.

This section asks when such a transfer works. We train a family of networks
that differ only in width, watch the best learning rate move as the network
widens, and trace the move to the size of the very first update. We then
change how each layer's learning rate and output scale depend on the width,
so that a learning rate tuned on the narrowest network stays good for the
widest. A diagnostic that takes seconds checks the change, a sweep tests the
transfer, and a last step connects the rules to the update sizes of Muon
(:numref:`sec_muon`).

```{.python .input #scaling-scaling-up}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #scaling-scaling-up}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import numpy as np
import optax
import random
```

## Learning Rate and Width

### A Family of Widths

We want the smallest experiment in which a model can be scaled up while
remaining the same model. A multilayer perceptron on Fashion-MNIST serves:
three hidden layers of width $n$, with $n$ from 128 to 1,024. Every member
of the family computes the same kind of function; only the width differs.
Note where the width lives. The input matrix always has fan-in 784 and the
output matrix always has fan-out 10, so only the two $n \times n$ matrices
in the middle grow in both dimensions. This distinction matters below.
Because we will train several dozen of these models for a few hundred steps
each, we keep the whole training set on the accelerator as one tensor. The
60,000 images occupy less than 200 MB, and indexing them directly avoids
the overhead of a data loader in such short runs.

```{.python .input #scaling-a-family-of-widths-1}
%%tab pytorch
fashion = d2l.FashionMNIST()
device = d2l.try_gpu()
X = (fashion.train.data.float().reshape(-1, 784) / 255).to(device)
Y = fashion.train.targets.to(device)
```

```{.python .input #scaling-a-family-of-widths-1}
%%tab jax
fashion = d2l.FashionMNIST()
X = jnp.asarray(fashion.train[0], dtype=jnp.float32).reshape(-1, 784) / 255
Y = jnp.asarray(fashion.train[1], dtype=jnp.int32)
```

We specify the initialization explicitly, because the initialization is
part of what a rule for scaling must prescribe: every weight is Gaussian
with variance $1/\textrm{fan-in}$, and every bias is zero. Together with one
learning rate shared by all layers, this is the *standard parametrization*
(SP). Up to constants it is what default initializers do and what the
Xavier and He schemes of :numref:`sec_numerical_stability` prescribe. The
optimizer is created by a method, `configure_adam`, so that a variant of the
class can override it later.

```{.python .input #scaling-a-family-of-widths-2}
%%tab pytorch
class MLP(nn.Module):
    """A three-hidden-layer ReLU network under standard parametrization."""
    def __init__(self, width):
        super().__init__()
        self.fc_in = nn.Linear(784, width)
        self.fc_h1 = nn.Linear(width, width)
        self.fc_h2 = nn.Linear(width, width)
        self.fc_out = nn.Linear(width, 10)
        for lin in (self.fc_in, self.fc_h1, self.fc_h2, self.fc_out):
            nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
            nn.init.zeros_(lin.bias)

    def features(self, X):
        h = F.relu(self.fc_h1(F.relu(self.fc_in(X))))
        return F.relu(self.fc_h2(h))

    def forward(self, X):
        return self.fc_out(self.features(X))

    def configure_adam(self, lr):
        return torch.optim.Adam(self.parameters(), lr)
```

```{.python .input #scaling-a-family-of-widths-2}
%%tab jax
class MLP(nnx.Module):
    """A three-hidden-layer ReLU network under standard parametrization."""
    def __init__(self, width, *, rngs):
        init = nnx.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        self.fc_in = nnx.Linear(784, width, kernel_init=init, rngs=rngs)
        self.fc_h1 = nnx.Linear(width, width, kernel_init=init, rngs=rngs)
        self.fc_h2 = nnx.Linear(width, width, kernel_init=init, rngs=rngs)
        self.fc_out = nnx.Linear(width, 10, kernel_init=init, rngs=rngs)

    def features(self, X):
        h = nnx.relu(self.fc_h1(nnx.relu(self.fc_in(X))))
        return nnx.relu(self.fc_h2(h))

    def __call__(self, X):
        return self.fc_out(self.features(X))

    def configure_adam(self, lr):
        return nnx.Optimizer(self, optax.adam(lr), wrt=nnx.Param)
```

Training is a step-counted loop in the style of :numref:`sec_adam`: 400
steps of Adam at batch size 512, about three passes over the data, at a
constant learning rate. A seed fixes both the initialization and the
sequence of minibatches, so runs with the same seed differ only in width,
parametrization, and learning rate. The score of a run is its cross-entropy
on the whole training set after the last step. A diverged run is capped at
2.5, slightly above the loss $\log 10 \approx 2.3$ of uniform guessing.

```{.python .input #scaling-a-family-of-widths-3}
%%tab pytorch
def train_step(model, optimizer, Xb, Yb):
    loss = F.cross_entropy(model(Xb), Yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()

def train_mlp(arch, width, lr, seed, num_steps=400, batch_size=512):
    torch.manual_seed(seed)
    model = arch(width).to(device)
    optimizer = model.configure_adam(lr)
    idx = np.random.default_rng(seed).integers(0, X.shape[0],
                                               (num_steps, batch_size))
    for i in idx:
        train_step(model, optimizer, X[i], Y[i])
    with torch.inference_mode():
        v = float(F.cross_entropy(model(X), Y))
    return min(v, 2.5) if math.isfinite(v) else 2.5
```

```{.python .input #scaling-a-family-of-widths-3}
%%tab jax
@nnx.jit
def train_step(model, optimizer, Xb, Yb):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(Xb), Yb).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_mlp(arch, width, lr, seed, num_steps=400, batch_size=512):
    model = arch(width, rngs=nnx.Rngs(seed))
    optimizer = model.configure_adam(lr)
    idx = np.random.default_rng(seed).integers(0, X.shape[0],
                                               (num_steps, batch_size))
    for i in idx:
        train_step(model, optimizer, X[i], Y[i])
    v = float(optax.softmax_cross_entropy_with_integer_labels(
        model(X), Y).mean())
    return min(v, 2.5) if math.isfinite(v) else 2.5
```

### Learning-Rate Sweep
:label:`subsec_scaling-sweep`

We evaluate four widths and eight learning rates spaced by factors of two,
one short run per configuration, all from one random seed. The 32 runs stand
in for the proxy sweep of a large project. They show whether the learning
rate selected at one width serves the others.

```{.python .input #scaling-the-sweep}
widths = [128, 256, 512, 1024]
lrs = [2 ** k for k in range(-12, -4)]
seed = random.getrandbits(32)  # One draw for every run of this section
sp_loss = {w: [train_mlp(MLP, w, lr, seed) for lr in lrs] for w in widths}
for w in widths:
    print(f'width {w:4d}: best lr {min(zip(sp_loss[w], lrs))[1]:.1e}')
d2l.plot(lrs, [sp_loss[w] for w in widths], 'learning rate',
         'training loss', xscale='log', ylim=[0.25, 0.8],
         legend=[f'width {w}' for w in widths])
```

Each width traces a U: too small a learning rate undertrains, and too large
a one destabilizes, so that on the right the widest networks climb off the
top of the chart, where the capped runs sit. The U's do not line up. The
minimum slides left as the network widens: the best learning rate at width
1,024 is about eight times smaller than at width 128, and the drift
continues at larger widths :cite:`Yang.Hu.Babuschkin.ea.2022`. After tuning,
the wider networks match or beat the narrower ones, so width helps. But the
best region does not transfer. The width-128 optimum puts the width-1,024
network on the unstable side of its U, and the width-1,024 optimum
undertrains the narrow network. Under standard parametrization the right
learning rate depends on the width as well as on the task.

### The First Adam Step

The size of the first update explains the drift and shows how to correct
it. Consider a hidden weight matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$
that receives activations $\mathbf{h}$ whose entries have typical size
$\overline{|h|}$. For a single example its gradient is an outer product,
$\mathbf{g} = \boldsymbol{\delta} \mathbf{h}^\top$, where
$\boldsymbol{\delta}$ is the error propagated back to the layer's output.
On the very first step, Adam's ratio $\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$
equals $\operatorname{sign}(\mathbf{g})$, because after bias correction the
first and second moment estimates are $\mathbf{g}$ and $\mathbf{g}^2$
(:numref:`sec_adam`). The sign of an outer product factorizes,
$\operatorname{sign}(\delta_i h_j) = \operatorname{sign}(\delta_i)
\operatorname{sign}(h_j)$, so the update changes the layer's output by

$$
(\Delta\mathbf{W}\, \mathbf{h})_i
= -\eta \sum_{j=1}^{n} \operatorname{sign}(\delta_i h_j)\, h_j
= -\eta\, \operatorname{sign}(\delta_i) \sum_{j=1}^{n} |h_j|
\approx -\eta\, \operatorname{sign}(\delta_i)\, n\, \overline{|h|}.
$$
:eqlabel:`eq_scaling_first_step`

All $n$ terms of the sum have the same sign, because the update is built
from the very activations that it multiplies. Their sizes add up instead of
cancelling, and each output coordinate moves by about
$\eta\, n\, \overline{|h|}$. A minibatch gradient is a sum of such outer
products, and its sign no longer factorizes, so this argument holds only to
leading order. The coordinate check below confirms the width dependence by
measurement. Doubling the width doubles the change, so the largest stable
learning rate falls in proportion to $1/n$, the trend that the sweep shows.
The effect differs between layers. The input layer has fan-in 784 at every
width, and a bias has no fan-in at all, so their stable learning rates do
not move. The output matrix has fan-in $n$ as well, so its step grows like
those of the hidden matrices. It gets its own correction below. A single global $\eta$ is
a compromise between layers that scale differently, and the layers with
fan-in $n$ pull it down.

## Maximal Update Parametrization

### The Rules

If the stable learning rate of each layer scales differently with width,
one learning rate should not have to serve them all. The *maximal update
parametrization* (muP) :cite:`Yang.Hu.Babuschkin.ea.2022` writes the width
dependence into per-layer rules. Relative to a chosen *base width*, each
layer's learning rate, and one scale factor in the forward pass, absorbs that
layer's own scaling. What remains is a base learning rate that does not
depend on the width and can be tuned on the smallest member of the family.
The rules are derived from the limit of infinite width in which
every layer's activations stay of order one *and* every layer keeps learning
features :cite:`Yang.Hu.2021`. Larger updates make activations grow with
width, as in :eqref:`eq_scaling_first_step`; smaller ones freeze the
features in the limit. "Maximal update" names the largest updates that keep
the activations stable.

For Adam and a width multiplier $m = n / n_{\textrm{base}}$, the rules fit
in a table.

:The maximal update parametrization for Adam, relative to a chosen base
width. The hidden-matrix learning rate and the output-logit multiplier
depend on the width; the initialization, the learning rates of the input
weights and biases, and the rest of the forward pass do not.
:label:`tab_mup-rules`

| parameters | initialization | Adam learning rate | forward pass |
|:--|:--|:--|:--|
| input weights and all biases | unchanged | $\eta$ | unchanged |
| hidden matrices | variance $\propto 1/\textrm{fan-in}$ (unchanged) | $\eta / m$ | unchanged |
| output matrix | unchanged | $\eta$ | logits $\times\, 1/m$ |

The initialization column is the one that does *not* change: variance
$\propto 1/\textrm{fan-in}$ is right at every width, and `MLP` already uses
it. The hidden learning rate shrinks as $1/m$, which cancels the $n$-fold
sum of :eqref:`eq_scaling_first_step`. The output layer keeps its learning
rate, but its logits are divided by $m$, which cancels the same sum on the
output side while letting the head keep learning. Input weights and biases,
whose fan-in never grows, are left alone. For language models the token
embedding belongs to this class too, which is why muP treats embeddings
apart from hidden matrices. Transformers need one further rule: attention
logits scaled by $1/d$ rather than $1/\sqrt{d}$. SGD has its own column of
exponents, and several algebraically equivalent formulations are in use.
The practitioner's guide of :citet:`Dey.Anthony.Hestness.2024` lays out the
full table and its pitfalls. Here the whole change is a subclass:

```{.python .input #scaling-the-rules}
%%tab pytorch
class MuMLP(MLP):
    """The same network under muP, relative to a width-128 base."""
    def __init__(self, width, base_width=128):
        super().__init__(width)
        self.m = width / base_width

    def forward(self, X):
        # Rule 1: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        hidden = [self.fc_h1.weight, self.fc_h2.weight]
        rest = [p for p in self.parameters()
                if not any(p is q for q in hidden)]
        return torch.optim.Adam([              # Rule 2: hidden LR / m
            {'params': rest, 'lr': lr},
            {'params': hidden, 'lr': lr / self.m}])
```

```{.python .input #scaling-the-rules}
%%tab jax
class MuMLP(MLP):
    """The same network under muP, relative to a width-128 base."""
    def __init__(self, width, base_width=128, *, rngs):
        super().__init__(width, rngs=rngs)
        self.m = width / base_width

    def __call__(self, X):
        # Rule 1: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        def labels(params):                   # Rule 2: hidden LR / m
            return jax.tree_util.tree_map_with_path(
                lambda path, _: 'hidden'
                if 'fc_h' in jax.tree_util.keystr(path)
                and 'kernel' in jax.tree_util.keystr(path) else 'rest',
                params)
        tx = optax.multi_transform(
            {'hidden': optax.adam(lr / self.m), 'rest': optax.adam(lr)},
            labels)
        return nnx.Optimizer(self, tx, wrt=nnx.Param)
```

At the base width $m = 1$, and `MuMLP` is `MLP`. The parametrization
changes nothing about the model that is tuned, only how its wider siblings
scale.

### The Coordinate Check
:label:`subsec_scaling-coord-check`

The *coordinate check* tests the width scaling of
:eqref:`eq_scaling_first_step` directly, and the practitioner's guide of
:citet:`Dey.Anthony.Hestness.2024` recommends it as the first test of any
muP implementation. Under the rules used here, the hidden activations should
stay of order one after a training step at every width, while the logits
follow their explicit $1/m$ scaling. Activations that grow with width reveal
a scaling error. We therefore build the family across widths, apply one Adam
update at a learning rate in the middle of the grid, and record the mean
absolute activation of each layer. One step is the cleanest probe:
:eqref:`eq_scaling_first_step` describes it directly, while later steps
compound the growth and mix it with feedback from the loss.

```{.python .input #scaling-the-coordinate-check-1}
%%tab pytorch
def activations(model, Xb):
    with torch.inference_mode():
        h1 = F.relu(model.fc_in(Xb))
        h2 = F.relu(model.fc_h1(h1))
        h3 = F.relu(model.fc_h2(h2))
        return [float(h.abs().mean()) for h in (h1, h2, h3, model(Xb))]

def coord_check(arch, widths, seed, lr=2**-8, num_steps=1):
    Xb, Yb = X[:256], Y[:256]
    sizes = []
    for width in widths:
        torch.manual_seed(seed)
        model = arch(width).to(device)
        optimizer = model.configure_adam(lr)
        for _ in range(num_steps):
            train_step(model, optimizer, Xb, Yb)
        sizes.append(activations(model, Xb))
    return list(zip(*sizes))
```

```{.python .input #scaling-the-coordinate-check-1}
%%tab jax
def activations(model, Xb):
    h1 = nnx.relu(model.fc_in(Xb))
    h2 = nnx.relu(model.fc_h1(h1))
    h3 = nnx.relu(model.fc_h2(h2))
    return [float(jnp.abs(h).mean()) for h in (h1, h2, h3, model(Xb))]

def coord_check(arch, widths, seed, lr=2**-8, num_steps=1):
    Xb, Yb = X[:256], Y[:256]
    sizes = []
    for width in widths:
        model = arch(width, rngs=nnx.Rngs(seed))
        optimizer = model.configure_adam(lr)
        for _ in range(num_steps):
            train_step(model, optimizer, Xb, Yb)
        sizes.append(activations(model, Xb))
    return list(zip(*sizes))
```

First under standard parametrization. The check takes seconds, so we extend
the widths to 4,096, beyond those of the training sweep:

```{.python .input #scaling-the-coordinate-check-2}
check_widths = [128, 256, 512, 1024, 2048, 4096]
sp_acts = coord_check(MLP, check_widths, seed)
d2l.plot(check_widths, list(sp_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.01, 100],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

The plot shows :eqref:`eq_scaling_first_step` at work. Layer 1, behind the
input matrix with its fixed fan-in, is flat. Layers 2 and 3, behind the
square matrices, grow with width, and the growth compounds with depth
because each layer passes its excess on to the next. The logits grow by
about two orders of magnitude across the range. A first step that is stable
at narrow widths therefore destabilizes a wide enough model. The same check
under muP, on the same axes:

```{.python .input #scaling-the-coordinate-check-3}
mup_acts = coord_check(MuMLP, check_widths, seed)
d2l.plot(check_widths, list(mup_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.01, 100],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

All three layer curves are flat: no layer's update size depends on the
width anymore. The logits *fall* with width, and this is by design. The
$1/m$ multiplier starts the logits small, and they grow to order one through
learning rather than through size. Growth is the unambiguous failure. A flat
or falling curve says only that the forward pass is stable, because the
check reads the sizes of activations, not whether features are learned.
Updates that shrink too fast with width would also produce falling curves,
with the layer frozen. The test still catches the common implementation
errors: a missing multiplier, a mislabeled layer, or an initializer with a
fixed standard deviation, like the 0.02 common in transformer code, in place
of one that scales with the fan-in. The exercises break each of the two
rules in turn and ask which errors the check detects.

### Learning-Rate Transfer

The coordinate check tests the scaling of activations. Whether the learning
rate now transfers is a question for the sweep, which we repeat under muP
with the same seed:

```{.python .input #scaling-learning-rate-transfer}
mup_loss = {w: [train_mlp(MuMLP, w, lr, seed) for lr in lrs] for w in widths}
for w in widths:
    print(f'width {w:4d}: best lr {min(zip(mup_loss[w], lrs))[1]:.1e}')
d2l.plot(lrs, [mup_loss[w] for w in widths], 'learning rate',
         'training loss', xscale='log', ylim=[0.25, 0.8],
         legend=[f'width {w}' for w in widths])
```

The width-128 curve repeats the earlier one, because at the base width the
two parametrizations coincide and the seed is the same. For the other widths
the minimum no longer drifts left: across the eightfold change in width,
each grid minimum lies within one grid step of the base-width minimum. To
measure what transfer is worth, we reuse the learning rate selected at width
128 for width 1,024 and compare the result with the best loss of the
width-1,024 sweep. We also print the best loss at each width, the result
of tuning every width separately:

```{.python .input #scaling-learning-rate-transfer-1}
for name, loss in (('SP', sp_loss), ('muP', mup_loss)):
    tuned = loss[128].index(min(loss[128]))  # Grid index chosen at width 128
    excess = loss[1024][tuned] / min(loss[1024]) - 1
    best = '  '.join(f'{min(loss[w]):.3f}' for w in widths)
    print(f'{name:3s}  best loss at widths {widths}: {best}')
    print(f'     width-128 lr at width 1024: {excess:.0%} above its best')
```

Under muP the transferred learning rate lands within a few percent of the
best loss at width 1,024. Under standard parametrization it lands about 20%
above it. This is *hyperparameter transfer*: tune the base model, scale up,
keep the numbers. Under muP the best loss also falls steadily with width,
so the widest network, run at a learning rate tuned for the narrowest, is
the best model of the family. Retuned at each width, standard
parametrization reaches best losses close to muP's on a family this small,
where the sweep at width 1,024 is still cheap. Here muP saves that sweep.
:citet:`Yang.Hu.Babuschkin.ea.2022` find that the two diverge at larger
widths: the tuned loss under standard parametrization stops improving, while
under muP it keeps falling. Transfer concerns the base learning rate, not
every hyperparameter: schedules, batch size (:numref:`sec_batch_size`), and
regularization still interact with scale. At a large scale,
:citet:`Yang.Hu.Babuschkin.ea.2022` tuned a 6.7-billion-parameter GPT-3 by
sweeping a 40-million-parameter proxy, spent about 7% of the pretraining
compute on tuning, and outperformed the original model.

## The Spectral View

muP arrived through limits of infinite width, but a shorter route connects
it to :numref:`sec_muon`. There we measured the size of a matrix update by
its spectral norm, the largest factor by which it can stretch an activation
vector. For a layer that maps $n_{\textrm{in}}$ activations to
$n_{\textrm{out}}$, with entries of order one on both sides, the healthy
size is a spectral norm of order $\sqrt{n_{\textrm{out}} / n_{\textrm{in}}}$,
for the weights and for their updates alike:

$$
\|\mathbf{W}\|_2 \asymp \sqrt{\frac{n_{\textrm{out}}}{n_{\textrm{in}}}},
\qquad
\|\Delta\mathbf{W}\|_2 \asymp \sqrt{\frac{n_{\textrm{out}}}{n_{\textrm{in}}}}.
$$
:eqlabel:`eq_spectral_condition`

An input with entries of order one has length $\sqrt{n_{\textrm{in}}}$, and an
output with entries of order one has length $\sqrt{n_{\textrm{out}}}$, so a
matrix that maps one to the other stretches by the ratio of the two.
:citet:`Yang.Simon.Bernstein.2023` show that this *spectral condition*
recovers muP. Adam's first update of a hidden $n \times n$ matrix, the
rank-one sign matrix behind :eqref:`eq_scaling_first_step`, has a spectral
norm of order $\eta n$, about $\eta n / 2$ when half the ReLU units are active
(an exercise derives it). The condition asks for order one, and dividing the
hidden learning rate by $m$ removes the excess factor $n / n_{\textrm{base}}$.
Seen this way, the per-layer rules of :numref:`tab_mup-rules` bring Adam's
coordinatewise updates to the spectral scale at every width.

An optimizer whose update is already a matrix of controlled spectral norm
needs fewer of these corrections. Muon replaces the update of each hidden
matrix by an orthogonalized one, whose singular values all equal one
:cite:`Jordan.Jin.Boza.ea.2024,Bernstein.Newhouse.2024`. Multiplied by
$\sqrt{n_{\textrm{out}} / n_{\textrm{in}}}$ :cite:`Bernstein.2025`, that
update meets :eqref:`eq_spectral_condition` at every width by construction.
The scale used in :numref:`sec_muon`,
$0.2\sqrt{\max(n_{\textrm{out}}, n_{\textrm{in}})}$, was chosen instead to
match the root-mean-square entry of an AdamW update. It gives a square $n \times n$ update a spectral norm of
$0.2\,\eta\sqrt{n}$, which still grows with width, though more slowly than the
$\eta n$ of Adam's first step. Muon at that scale therefore still needs width
corrections for its learning rate to transfer, and
:citet:`Shah.Polloreno.Stratos.ea.2025` pair it with muP to transfer
hyperparameters across model sizes.

Width is only one direction of scaling. Transfer across depth, batch size,
and training length needs rules of its own, and whether weight decay rather
than parametrization keeps long runs stable is still debated
:cite:`Kosson.Welborn.Liu.ea.2025`. The coordinate check is worth running on
any family of models one intends to scale, whatever the parametrization.

## Summary

Hyperparameters tuned on a small model do not transfer by default: the best
learning rate falls as the network widens, because one Adam step perturbs a
hidden layer's activations in proportion to its fan-in
(:eqref:`eq_scaling_first_step`). The maximal update parametrization removes
the width dependence with per-layer rules: the usual initialization with
variance $\propto 1/\textrm{fan-in}$, hidden-matrix learning rates divided by
the width multiplier $m$, logits divided by $m$, and input weights and biases
left alone. In our family of networks, the best learning rate then stays at
the base-width value across an eightfold change in width. The coordinate
check tests such rules in seconds: after a training step, activations should
follow the width dependence that the parametrization prescribes, without
unintended growth. In spectral terms, muP makes Adam's updates land at the
norm that a layer's shape prescribes, and Muon's orthogonalized updates
carry part of that scaling by construction.

## Exercises

1. [code] **Transfer across seeds.** `sp_loss` and `mup_loss` hold one run
   per width and learning rate, all from one seed, so the transfer cost
   that they show carries no error bar.
    1. Under each parametrization, sweep `lrs` at widths 128 and 1,024 for
       three seeds. For each seed, apply that seed's width-128 optimum at
       width 1,024 and report, in percent, how far its loss lies above the
       minimum of the same seed's width-1,024 sweep.
    1. Is the difference between the two parametrizations' transfer costs
       larger than the spread of each cost across seeds?

1. [code] **Breaking muP on purpose.** The coordinate check is a test for
   implementation errors.
    1. Delete the $1/m$ logit scaling from `MuMLP` but keep the hidden
       learning-rate rule and rerun `coord_check`. Which curve exposes the
       bug?
    1. Restore the multiplier and instead give the hidden matrices the full
       learning rate. Which curve exposes this bug?
    1. Restore the hidden learning-rate rule, but divide the learning rate
       by $m^2$ instead of $m$. Does `coord_check` flag this error? Sweep
       `lrs` at width 1,024 with this variant, and compare its best
       learning rate and best loss with those of `MuMLP` in `mup_loss`.

1. [code] **Fitting the drift.** Instead of removing the drift, a large
   project can measure it: :citet:`Bi.Chen.Chen.ea.2024` fitted power laws
   for the optimal learning rate and batch size as functions of the compute
   budget across many small runs and extrapolated them to the target scale.
   In the first-step analysis of :eqref:`eq_scaling_first_step`, the largest
   stable learning rate scales as $1/n$.
    1. From `sp_loss`, fit a power law $\eta^\star(n) = c\, n^{-\gamma}$ to
       the best learning rate at each width. Since `lrs` is spaced by
       factors of two, each optimum is resolved only to within a factor of
       $\sqrt{2}$. Give the range of $\gamma$ consistent with this
       resolution, and state whether it contains $\gamma = 1$.
    1. Extrapolate the fit to widths 2,048 and 4,096. Sweep directly at
       those widths with `train_mlp`, extending `lrs` downward if a
       prediction falls below its smallest value, and report how many grid
       steps separate each prediction from the measured optimum.

1. [code] **Depth and the coordinate check.** ● `coord_check` varies width
   at a fixed depth of three hidden layers. Generalize `MLP` and `MuMLP` to
   $L$ hidden layers and run the check across $L \in \{3, 6, 12, 24\}$ at
   width 512 under both parametrizations. The variance-$1/\textrm{fan-in}$
   initialization of `MLP` halves the second moment of the ReLU
   activations at every layer (:numref:`subsec_he_init`), so for this
   problem initialize the input and hidden matrices with variance
   $2/\textrm{fan-in}$.
    1. Under standard parametrization, how does the growth of the logits
       with depth compare with their growth with width in
       :numref:`subsec_scaling-coord-check`?
    1. Does muP's width scaling keep the activations flat across depth?
       State what the rules of :numref:`tab_mup-rules` promise and what
       they do not.

1. **First update in spectral norm.** ● For a single example, Adam's first
   update of a weight matrix
   $\mathbf{W} \in \mathbb{R}^{n_{\textrm{out}} \times n_{\textrm{in}}}$ is
   $\Delta\mathbf{W} = -\eta\, \operatorname{sign}(\boldsymbol{\delta})\,
   \operatorname{sign}(\mathbf{h})^\top$, as in the derivation
   of :eqref:`eq_scaling_first_step`. Let $k_{\textrm{out}}$ and
   $k_{\textrm{in}}$ be the numbers of nonzero entries of
   $\boldsymbol{\delta}$ and $\mathbf{h}$.
    1. Compute $\|\Delta\mathbf{W}\|_2$ as a function of $\eta$,
       $k_{\textrm{out}}$, and $k_{\textrm{in}}$.
    1. Assume that $k_{\textrm{out}}$ and $k_{\textrm{in}}$ are fixed
       fractions of $n_{\textrm{out}}$ and $n_{\textrm{in}}$. For the input,
       hidden, and output matrices of `MLP`, find how $\eta$ must scale
       with the width $n$ for $\Delta\mathbf{W}$ to satisfy the spectral
       condition :eqref:`eq_spectral_condition`.
    1. Show that the learning rates and the logit multiplier of
       :numref:`tab_mup-rules` give every layer an update at this scale.

1. [code] **Weight decay and parametrization.** ●
   :citet:`Kosson.Welborn.Liu.ea.2025` argue that in long runs weight
   decay, not parametrization, stabilizes the update dynamics across
   widths. Override `configure_adam` in a subclass of `MLP` to return
   AdamW (:numref:`sec_adamw`) with decay coefficient
   $\lambda = 10^{-3}/\eta$, so that every run averages its updates over
   the horizon $1/(\eta\lambda) = 1{,}000$ steps described
   in :numref:`subsec_wd-at-scale`.
    1. Run the standard-parametrization sweep of
       :numref:`subsec_scaling-sweep` at widths 128 and 1,024 with
       `num_steps=4000`, once with this AdamW and once with Adam. Report
       the best learning rate at each width for both optimizers. Does the
       optimum drift less with width under AdamW?
    1. `MLP` has no normalization layer. Given the equilibrium argument of
       :numref:`subsec_wd-at-scale`, state what your result can and cannot
       say about the cited claim.

[Discussions](https://d2l.discourse.group/t/36766)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.11]{.kicker}

Scaling Up<br>
**the drifting optimum · the first Adam step · muP · the coordinate check · the spectral view**
:::
:::

::: {.slide title="The Best Learning Rate Moves When the Network Widens"}
[The problem]{.kicker}

- A 7B-parameter model trained for months allows no learning-rate sweep:
  tune a small proxy, then reuse its settings.
- Fashion-MNIST MLP, 400 Adam steps: the best learning rate at width
  1,024 is several times smaller than at width 128.
- Reusing the width-128 optimum at width 1,024 costs about 20% in loss.

::: {.d2l-note .warn}
Why does the optimum move, and can a parametrization pin it?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain the drift with the size of Adam's first update;
- state the muP rules for Adam and implement them as a subclass;
- run a coordinate check and read what it can and cannot detect;
- measure the cost of transferring a learning rate across widths;
- relate muP to spectral norms and to Muon's update scale.

**Running example:** an MLP $784 \to n \to n \to n \to 10$ on
Fashion-MNIST, $n = 128$ to $1{,}024$, 400 Adam steps.
:::

::: {.slide title="Scaling Up Asks Which Settings Survive a Wider Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise make training slow |
| Gradient Descent | step size against curvature |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| Momentum | averaging over time |
| Adam and AdamW | a step size per coordinate, decoupled decay |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| **Scaling Up** | **learning rates across widths** |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** Adam's first step is a sign step; Muon's spectral norm.
- **Skips:** transfer across depth, batch size, and training length.
:::
:::
:::

::: {.slide title="Only the Two Middle Matrices Grow in Both Dimensions"}
MLP $784 \to n \to n \to n \to 10$, $n = 128, \ldots, 1{,}024$:

| matrix | shape | grows with $n$ |
|:--|:--|:--|
| input | $n \times 784$ | fan-out only |
| hidden (two) | $n \times n$ | **both** |
| output | $10 \times n$ | fan-in only |

**Standard parametrization (SP):** weights with variance
$1/\textrm{fan-in}$, zero biases, one learning rate for all layers. One
random seed fixes initialization and minibatches for every run.
:::

::: {.slide title="Under SP the Best Learning Rate Slides Left with Width"}
Eight learning rates × four widths, 400 Adam steps each:

@!scaling-the-sweep

Wider is better once tuned, but each width needs its own learning rate.
:::

::: {.slide title="Adam's First Step Adds n Coherent Terms"}
Single example: $\mathbf{g} = \boldsymbol{\delta}\mathbf{h}^\top$. First
Adam step $= \operatorname{sign}(\mathbf{g})$, and
$\operatorname{sign}(\delta_i h_j) = \operatorname{sign}(\delta_i)
\operatorname{sign}(h_j)$:

$$(\Delta\mathbf{W}\mathbf{h})_i = -\eta \operatorname{sign}(\delta_i)
\sum_{j=1}^n |h_j| \approx -\eta \operatorname{sign}(\delta_i)\, n\,
\overline{|h|}.$$

- All $n$ terms share a sign: doubling the width doubles the change.
- Largest stable $\eta \propto 1/n$ for the hidden matrices.
- Input layer (fan-in 784) and biases: no growth. Output matrix (fan-in
  $n$): grows too.
:::

::: {.slide title="Check: How Much Larger Is the First Update at Width 1,024?"}
[Check]{.kicker}

Width grows from 128 to 1,024. By what factor does the first Adam step
change a hidden layer's output? Which learning rate restores the width-128
change?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The sum has $n$ coherent terms: $1{,}024 / 128 = 8\times$.
Dividing the hidden learning rate by $m = 8$ restores it; the input layer
and biases need no change.
:::
:::
:::

::: {.slide title="muP Moves the Width Dependence into Two Rules"}
Width multiplier $m = n / n_{\textrm{base}}$ (Yang et al., 2022):

| parameters | init | Adam LR | forward |
|:--|:--|:--|:--|
| input weights, biases | unchanged | $\eta$ | unchanged |
| hidden matrices | $\propto 1/\textrm{fan-in}$ | $\eta/m$ | unchanged |
| output matrix | unchanged | $\eta$ | logits $\times 1/m$ |

Derived from the infinite-width limit where activations stay $O(1)$
**and** every layer keeps learning. At $m = 1$ muP is SP.
:::

::: {.slide title="In Code, muP Is a Subclass with Two Changes"}
- Rule 1, forward pass: `self.fc_out(self.features(X) / self.m)`.
- Rule 2, optimizer: two parameter groups, hidden matrices at `lr / m`,
  everything else at `lr`.
- PyTorch: `torch.optim.Adam` with parameter groups. JAX:
  `optax.multi_transform` with labels from the parameter paths.
- Embeddings in a language model count as input weights; attention needs
  $1/d$ instead of $1/\sqrt{d}$; SGD has its own column.
:::

::: {.slide title="The Coordinate Check Exposes SP's Growth in Seconds"}
Mean |activation| per layer after one Adam step, widths 128 to 4,096:

@!scaling-the-coordinate-check-2

Layer 1 flat; layers behind the square matrices grow, compounding with
depth; logits grow about 100×.
:::

::: {.slide title="Under muP Every Hidden Layer Stays Flat"}
@!scaling-the-coordinate-check-3

Logits *fall* by design (the $1/m$ multiplier) and grow to $O(1)$ by
learning. Growth is the failure; the check reads sizes, not feature
learning.
:::

::: {.slide title="Check: Which Curve Shows a Missing Logit Multiplier?"}
[Check]{.kicker}

You keep the hidden learning rate at $\eta/m$ but forget the $1/m$ on the
logits. What does the coordinate check show?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The hidden layers stay flat, but the output layer's first step
again sums $n$ coherent terms: the logits curve grows in proportion to the
width.
:::
:::
:::

::: {.slide title="Under muP the Optimum Stays at the Base-Width Value"}
Same sweep, same seed, muP:

@!scaling-learning-rate-transfer

Each width's minimum lies within one grid step of width 128's.
:::

::: {.slide title="Transfer Costs a Few Percent Under muP, About 20% Under SP"}
Reuse the width-128 learning rate at width 1,024:

@!scaling-learning-rate-transfer-1

- muP: the best loss falls with width. Retuned SP comes close at these
  widths; muP saves the sweep.
- Wider still, SP's tuned loss stops improving while muP's keeps falling.
- GPT-3 6.7B tuned on a 40M proxy for about 7% of the pretraining compute
  (Yang et al., 2022).
:::

::: {.slide title="muP Is a Spectral-Norm Condition on Every Update"}
Healthy layer size: $\|\mathbf{W}\|_2 \asymp \|\Delta\mathbf{W}\|_2 \asymp
\sqrt{n_{\textrm{out}}/n_{\textrm{in}}}$ (Yang, Simon, Bernstein, 2023).

- Adam's first hidden update: spectral norm of order $\eta n$ (about
  $\eta n/2$ with half the units active); $\eta/m$ removes the excess.
- Muon's orthogonalized update has all singular values 1: scaled by
  $\sqrt{n_{\textrm{out}}/n_{\textrm{in}}}$ it meets the condition at
  every width.
- RMS-matched Muon ($0.2\sqrt{\max(n_{\textrm{out}}, n_{\textrm{in}})}$):
  spectral norm $0.2\,\eta\sqrt{n}$, still width-dependent.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Drift:** under SP the best learning rate falls with width, because
  Adam's first step sums $n$ coherent terms.
- **muP:** hidden learning rate $\eta/m$, logits $\times 1/m$, the rest
  unchanged; the optimum stays at the base width's value.
- **Coordinate check:** flat or falling activations across widths after
  one step; growth means a scaling error.
- **Spectral view:** muP sets every update's spectral norm to
  $\sqrt{n_{\textrm{out}}/n_{\textrm{in}}}$; Muon builds part of this in.

**Next:** clipping, averaging, and tuning budgets for long runs.
:::
