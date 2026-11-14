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
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #scaling-scaling-up}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #scaling-scaling-up-1}
%%tab pytorch, jax
import random
```

## Learning Rate and Width

### A Family of Widths

<!-- WRITER: widths now run from 128 to 8,192 (a 64-fold range: 128, 512, 2,048, 8,192); the
coordinate check covers every power of two from 128 to 8,192. -->
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

<!-- WRITER: the score is now the mean minibatch training loss of the last 50 of the 400
steps (not the full-set loss after the last step); a run whose mean is not finite is still
capped at 2.5. The JAX tab binds model and optimizer once with nnx.cached_partial (per-step
dispatch); both tabs keep the minibatch indices on the device. -->
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
    idx = torch.as_tensor(np.random.default_rng(seed).integers(
        0, X.shape[0], (num_steps, batch_size)), device=device)
    losses = torch.stack([train_step(model, optimizer, X[i], Y[i])
                          for i in idx])
    v = float(losses[-50:].mean())  # Mean minibatch loss of the last 50 steps
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
    step = nnx.cached_partial(train_step, model, model.configure_adam(lr))
    idx = jnp.asarray(np.random.default_rng(seed).integers(
        0, X.shape[0], (num_steps, batch_size)))
    losses = jnp.stack([step(X[i], Y[i]) for i in idx])
    v = float(losses[-50:].mean())  # Mean minibatch loss of the last 50 steps
    return min(v, 2.5) if math.isfinite(v) else 2.5
```

### Learning-Rate Sweep
:label:`subsec_scaling-sweep`

<!-- WRITER: the sweep is now four widths (128, 512, 2,048, 8,192) x ten rates (2^-14 to
2^-5, factors of two) x two seeds drawn once and shared by every sweep of the section: 80 SP
runs. A new definitions cell before it holds `sweep` (loss array per width, one column per
seed), `print_optimum` (best rate of the seed-averaged loss per width and the least-squares
slope of log best rate against log width) and `plot_sweep` (seed-averaged curves). -->
We evaluate four widths and nine learning rates spaced by factors of two,
one short run per configuration, all from one random seed. The 36 runs stand
in for the proxy sweep of a large project. They show whether the learning
rate selected at one width serves the others.

```{.python .input #scaling-learning-rate-sweep-1}
%%tab pytorch, jax
def sweep(arch, widths, lrs, seeds):
    """Score of every width and learning rate, one column per seed."""
    return {w: np.array([[train_mlp(arch, w, lr, s) for s in seeds]
                         for lr in lrs]) for w in widths}

def print_optimum(loss, lrs):
    """Best rate of the seed-averaged loss per width; slope in log-log."""
    widths = list(loss)
    best = [lrs[loss[w].mean(1).argmin()] for w in widths]
    slope = np.polyfit(np.log(widths), np.log(best), 1)[0]
    print('best lr:', '  '.join(f'{lr:.1e}' for lr in best),
          f'   slope {slope:+.2f}')

def plot_sweep(loss, lrs):
    d2l.plot(lrs, [l.mean(1) for l in loss.values()], 'learning rate',
             'training loss', xscale='log', ylim=[0.2, 0.8],
             legend=[f'width {w}' for w in loss])
```

```{.python .input #scaling-the-sweep}
%%tab pytorch, jax
widths = [128, 512, 2048, 8192]
lrs = [2.0 ** k for k in range(-14, -4)]
seeds = [random.getrandbits(32) for _ in range(2)]  # Shared by every sweep
sp_loss = sweep(MLP, widths, lrs, seeds)
print_optimum(sp_loss, lrs)
plot_sweep(sp_loss, lrs)
```

<!-- WRITER: measured (four development runs of two seeds each per tab, PT and JX): SP best
rate 3.9e-3 (once 7.8e-3), 2.0e-3, 4.9e-4, 2.4e-4 (once 1.2e-4) at widths 128, 512, 2,048,
8,192: a 16-to 32-fold fall over a 64-fold widening; fitted slope -0.70 to -0.85 in every
run of both tabs, against -1 from the first-step argument. Tuned SP loss falls with width:
0.32-0.33, 0.29-0.30, 0.26-0.27, 0.22-0.23. On the right, the wider the network the lower
the rate at which its U climbs. "Several times smaller" and the eightfold range are now
false. -->
Each width traces a U: too small a learning rate undertrains, and too large
a one destabilizes. On the right the wider networks tend to climb fastest,
in some runs off the top of the chart, where the capped runs sit. The U's do
not line up. The minimum slides left as the network widens: the best
learning rate at width 1,024 is several times smaller than at width
128, and the drift
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

<!-- WRITER: `MuMLP` now follows the published rule for the output matrix (Yang et al. 2022,
Table 8 / mup's MuReadout): its initial variance is held at 1/base_width at every width (the
code multiplies the 1/fan-in init by sqrt(m), as MuReadout._rescale_parameters does), with
the logits divided by m and the output learning rate unchanged. The table's output row
should read "variance 1/n_base (fixed)"; the sentence "variance 1/fan-in is right at every
width" is true for input and hidden matrices only. The code comments now number three rules
(output init, logit multiplier, hidden LR / m); the JAX Rule comment sits on the `lr /
self.m` line. -->
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
    """The same network under muP, relative to a width-8,192 base."""
    def __init__(self, width, base_width=8192):
        super().__init__(width)
        self.m = width / base_width
        # Rule 1: output init variance 1/base_width at every width
        with torch.no_grad():
            self.fc_out.weight.mul_(self.m ** 0.5)

    def forward(self, X):
        # Rule 2: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        hidden = [self.fc_h1.weight, self.fc_h2.weight]
        rest = [p for p in self.parameters()
                if not any(p is q for q in hidden)]
        return torch.optim.Adam([
            {'params': rest, 'lr': lr},
            {'params': hidden, 'lr': lr / self.m}])  # Rule 3: hidden LR / m
```

```{.python .input #scaling-the-rules}
%%tab jax
class MuMLP(MLP):
    """The same network under muP, relative to a width-8,192 base."""
    def __init__(self, width, base_width=8192, *, rngs):
        super().__init__(width, rngs=rngs)
        self.m = width / base_width
        # Rule 1: output init variance 1/base_width at every width
        self.fc_out.kernel[...] *= self.m ** 0.5

    def __call__(self, X):
        # Rule 2: scale the output matrix's logits; the bias is untouched
        return self.fc_out(self.features(X) / self.m)

    def configure_adam(self, lr):
        def label(path, _):
            name = jax.tree_util.keystr(path)
            return 'hidden' if 'fc_h' in name and 'kernel' in name else 'rest'
        tx = optax.multi_transform(
            {'hidden': optax.adam(lr / self.m),  # Rule 3: hidden LR / m
             'rest': optax.adam(lr)},
            lambda params: jax.tree_util.tree_map_with_path(label, params))
        return nnx.Optimizer(self, tx, wrt=nnx.Param)
```

<!-- WRITER: base width is now 8,192, the widest member (not 128 as ruling 4 proposed, not
1,024 as before); see the code log. Measured with the published output rule, two seeds, both
tabs, one development run each: with the base at 128 muP's curves collapse onto one
width-independent limit (widths 512 to 8,192 agree to 0.01 at every rate up to the optimum)
whose tuned loss at width 8,192 is 0.29 against 0.22 for tuned SP, and whose optimum sits
one to three grid steps higher at the wide end (slope +0.15 PT, +0.45 JX). The base fixes
the per-layer constants: at width 8,192, base 128 gives the input matrix and biases 64x the
hidden matrices' rate (base 8,192: equal) and logits that start 8x smaller; base 8,192 costs
0.01-0.02 nats at width 128 (PT 0.326 vs 0.315, JX 0.338 vs 0.318) but base 128 costs 0.07
at width 8,192, so a width-128 proxy cannot tune them. With the base at the widest, muP at
width 8,192 is SP, and the narrow members borrow the wide model's constants. -->
At the base width $m = 1$, and `MuMLP` is `MLP`. Which width serves as the
base is a free choice. Every choice gives the same dependence on the width;
choices differ only by constant factors per layer, such as how fast the input
layer learns relative to the hidden ones and how large the output multiplier
is. These constants are hyperparameters in their own right, and the
practitioner's guide tunes them on the proxy along with the learning rate. We
avoid those extra sweeps by putting the base at the widest member of the
family, 1,024, where `MuMLP` is the standard network with one learning rate
for all layers. The narrower members, including the width-128 model that we
tune, have $m < 1$. A base at width 128 would instead fix the constants where
they matter least: in our runs it leaves tuned muP at width 1,024 one to
three hundredths of a nat above tuned standard parametrization. An exercise
traces which constants make the difference.

### The Coordinate Check
:label:`subsec_scaling-coord-check`

<!-- WRITER: each check now runs one Adam step at the rate tuned at that parametrization's
reference width: SP at its width-128 optimum (`lr_128`, 3.9e-3 = 2^-8 in both tabs), muP at
the SP optimum at the base width 8,192 (`lr_base`, 2.4e-4 = 2^-12 in both tabs), where the
two coincide. "A learning rate in the middle of the grid" is false. -->
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

def coord_check(arch, widths, lr, seed, num_steps=1):
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

def coord_check(arch, widths, lr, seed, num_steps=1):
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

<!-- WRITER: the check now spans 128 to 8,192, the same range as the sweep (not beyond it).
-->
First under standard parametrization. The check takes seconds, so we extend
the widths to 4,096, beyond those of the training sweep:

```{.python .input #scaling-the-coordinate-check-2}
%%tab pytorch, jax
check_widths = [128 * 2 ** k for k in range(7)]  # 128 to 8,192
lr_128 = lrs[sp_loss[128].mean(1).argmin()]  # SP's best rate at width 128
sp_acts = coord_check(MLP, check_widths, lr_128, seeds[0])
d2l.plot(check_widths, list(sp_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

<!-- WRITER: SP check (both tabs): layer 1 flat near 0.3; layers 2 and 3 flat to about width
512, then layer 2 grows from 0.2 to 2.4 and layer 3 from 0.2 to 12-14 by width 8,192; logits
grow from about 0.2 to 180-260, three orders of magnitude (not two). -->
The plot shows :eqref:`eq_scaling_first_step` at work. Layer 1, behind the
input matrix with its fixed fan-in, is flat. Layers 2 and 3, behind the
square matrices, grow with width, and the growth compounds with depth
because each layer passes its excess on to the next. The logits grow by
about two orders of magnitude across the range. A first step that is stable
at narrow widths therefore destabilizes a wide enough model. The same check
under muP, on the same axes:

```{.python .input #scaling-the-coordinate-check-3}
%%tab pytorch, jax
lr_base = lrs[sp_loss[8192].mean(1).argmin()]  # Best at the base width
mup_acts = coord_check(MuMLP, check_widths, lr_base, seeds[0])
d2l.plot(check_widths, list(mup_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

<!-- WRITER: muP check (both tabs, base 8,192): all three layer curves flat (0.1-0.2); the
logits are nearly flat too, about 3 at width 128 falling to about 1.7 at 8,192: with the
base at the widest, the narrow members start with logits about 8x (sqrt(64)) those of SP at
their width, falling as 1/sqrt(m) toward SP's at the base. They do not "fall by design
toward small values"; they fall to the base's level. -->
All three layer curves are flat: no layer's update size depends on the
width anymore. The logits *fall* with width, and this is by design. The
$1/m$ multiplier keeps the logits of the wider networks small at first, and
they grow to order one through learning rather than through size. Growth is the unambiguous failure. A flat
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
%%tab pytorch, jax
mup_loss = sweep(MuMLP, widths[:-1], lrs, seeds)
mup_loss[widths[-1]] = sp_loss[widths[-1]]  # At the base width muP is SP
print_optimum(mup_loss, lrs)
plot_sweep(mup_loss, lrs)
```

<!-- WRITER: the muP sweep runs widths 128, 512 and 2,048 and reuses the SP runs at width
8,192 (at the base muP is SP, same seeds), so it costs 60 runs. Measured best rate per width
(two seeds each, three runs per tab): 2.4e-4 or 1.2e-4 at every width in every run, never
more than one grid step from the width-128 optimum; slope -0.10 to +0.20 (PT 0.00, -0.05,
+0.05; JX +0.05, +0.20, -0.10) against SP's -0.70 to -0.85. Right walls: under muP the
narrow widths destabilize at lower rates (width 128 leaves the chart near 1e-3, width 8,192
climbs slowly), the opposite order from SP. -->
The width-1,024 curve repeats the earlier one, because at the base width the
two parametrizations coincide and the seed is the same. The narrower curves
have moved to meet it, and the minimum no longer drifts with width: across
the eightfold change in width, each grid minimum lies within one or two grid
steps of the width-128 minimum. To
measure what transfer is worth, we reuse the learning rate selected at width
128 for width 1,024 and compare the result with the best loss of the
width-1,024 sweep. We also print the best loss at each width, the result
of tuning every width separately:

```{.python .input #scaling-learning-rate-transfer-1}
%%tab pytorch, jax
for name, loss in (('SP', sp_loss), ('muP', mup_loss)):
    mean = {w: l.mean(1) for w, l in loss.items()}
    print(f'{name:3s} best loss per width:',
          '  '.join(f'{m.min():.3f}' for m in mean.values()))
    small, wide = widths[0], widths[-1]
    tuned, own = mean[small].argmin(), mean[wide].argmin()
    for label, i in ((f'the width-{small} lr', tuned), ('its own lr', own)):
        runs = '  '.join(f'{v:.3f}' for v in loss[wide][i])
        print(f'    width {wide} at {label:16s} {mean[wide][i]:.3f}'
              f'  (seeds {runs})')
```

<!-- WRITER: transfer at width 8,192, three runs of two seeds per tab. SP at the width-128
rate: 0.325-0.385 (single seeds 0.314-0.404) against 0.219-0.229 at its own rate (the two
seeds differ by 0.000-0.014): a cost of 0.10-0.16 nats, at least seven times the seed
difference, in every run of both tabs; a clear loss, not a divergence. muP at the width-128
rate: 0.000-0.003 above its own optimum (the same grid point in five of six runs). Best loss
per width, SP: 0.32-0.33, 0.29-0.30, 0.26-0.27, 0.22-0.23; muP: 0.33-0.34, 0.26-0.27,
0.235-0.24, equal to SP at 8,192 (same runs). Tuned muP beats tuned SP at widths 512 (by
0.03-0.04) and 2,048 (by 0.02-0.03), ties or trails by up to 0.02 at width 128 (the cost of
borrowing the wide model's constants), and equals it at the base by construction. "10% to
30%", "a few percent", "retuned at width 1,024" and the GPT-3 sentence's framing of "muP
saves that sweep" need the new numbers. -->
Under muP the transferred learning rate typically lands within a few
percent of the best loss at width 1,024. Under standard parametrization it
typically lands 10% to 30% above it. Each number comes from a single seed
and varies from run to run, so an occasional run brings the two close
together; the first exercise measures the spread. This is *hyperparameter transfer*: tune the small model,
scale up, keep the numbers. Under muP the best loss also falls from width
128 to width 1,024. Retuned at width 1,024, standard parametrization reaches
the same best loss, because there the two parametrizations coincide. On a
family this small that sweep is still cheap; muP saves it.
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

<!-- WRITER: Summary: "with the base at the widest member" stays true (now 8,192), but
"within a grid step or two ... across an eightfold change" is now "within one grid step
across a 64-fold change (fitted slope -0.10 to +0.20, against -0.70 to -0.85 under SP)". -->
Hyperparameters tuned on a small model do not transfer by default: the best
learning rate falls as the network widens, because one Adam step perturbs a
hidden layer's activations in proportion to its fan-in
(:eqref:`eq_scaling_first_step`). The maximal update parametrization removes
the width dependence with per-layer rules: the usual initialization with
variance $\propto 1/\textrm{fan-in}$, hidden-matrix learning rates divided by
the width multiplier $m$, logits divided by $m$, and input weights and biases
left alone. These rules fix how each layer scales with the width; the choice
of base width fixes the constant factors. In our family of networks, with
the base at the widest member, the best learning rate then stays within a
grid step or two of its width-128 value across an eightfold change in width.
The coordinate check tests such rules in seconds: after a training step,
activations should follow the width dependence that the parametrization
prescribes, without unintended growth. In spectral terms, muP makes Adam's updates land at the
norm that a layer's shape prescribes, and Muon's orthogonalized updates
carry part of that scaling by construction.

## Exercises

<!-- WRITER: Exercises. 1: the cell now runs two seeds and prints them (worklist: drop). 3:
`MuMLP` now puts its base at 8,192; the interesting variant is `base_width=128`, which with
two seeds gives an optimum one to three grid steps higher at width 8,192 and a tuned loss
0.07 nats above SP's there (the limit of the base-128 constants; see the WRITER note on the
base). The three constants that differ (input and bias learning rate x64, output init std
x1/8 at width 8,192) are named in that note. 4: `print_optimum` now prints the fitted slope
over 128-8,192, so part 1 and the extrapolation to 2,048/4,096 are inside the sweep; rewrite
or cut. 7: worklist cuts it to its pencil part. Exercise 2.3 still compares against
`mup_loss`. -->
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

1. [code] **Choosing the base.** `MuMLP` puts its base at width 1,024, the
   widest member of the family.
    1. Rerun the muP sweep with `base_width=128`, so that muP coincides with
       standard parametrization at the tuned width instead. Does the best
       learning rate still transfer from width 128 to width 1,024? Compare
       the best loss at each width with `mup_loss` and `sp_loss`.
    1. The two bases differ only by constant factors per layer. List them,
       and find which of them accounts for the difference at width 1,024 by
       changing one at a time.

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

<!-- WRITER: slides: widths are 128-8,192 (not 1,024), ten rates x two seeds, base at 8,192;
the "several times smaller", "10% to 30%", "a few percent", "one or two grid steps", "logits
grow about 100x" (now ~1,000x) and base-1,024 lines are false; the muP rules slide needs the
output-init row; the transfer slide now shows a 0.10-nat SP cost vs none under muP. -->

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
- Reusing the width-128 optimum at width 1,024 typically costs 10% to 30%
  in loss (one seed; the cost varies from run to run).

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
| Landscapes | curvature and noise slow training |
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
Nine learning rates × four widths, 400 Adam steps each:

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
- Base at width 1,024, where muP is SP: any base gives the same width
  scaling; the base fixes the per-layer constants.
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

::: {.slide title="Under muP the Optimum Stops Drifting"}
Same sweep, same seed, muP:

@!scaling-learning-rate-transfer

Each width's minimum lies within one or two grid steps of width 128's.
:::

::: {.slide title="Transfer Typically Costs a Few Percent Under muP, 10–30% Under SP"}
Reuse the width-128 learning rate at width 1,024:

@!scaling-learning-rate-transfer-1

- muP: the best loss falls with width. At width 1,024 muP is SP, so
  retuned SP matches it there; muP saves that sweep.
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
  unchanged; the optimum stops drifting with width.
- **Coordinate check:** flat or falling activations across widths after
  one step; growth means a scaling error.
- **Spectral view:** muP sets every update's spectral norm to
  $\sqrt{n_{\textrm{out}}/n_{\textrm{in}}}$; Muon builds part of this in.

**Next:** clipping, averaging, and tuning budgets for long runs.
:::
