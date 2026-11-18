# Scaling Up
:label:`sec_scaling`

Models are developed small and trained large. We debug a version that
trains in minutes, find its learning rate with a sweep like the ones in this
chapter, and then widen the model for the run that matters. Does the
learning rate come along? For the largest models the question cannot be
avoided. A model with billions of parameters, trained for weeks on thousands
of accelerators, allows no sweep, and often not even a second run. Its
hyperparameters are tuned on a *proxy*, a much smaller model of the same
design, and reused. If they do not carry over, the large run diverges or
trains slowly, and the error shows only after much of the budget is spent.

By default they do not carry over, and the reason fits in one line of
algebra. The output of a unit is a sum over its $n$ inputs, and one step of
Adam changes every term of that sum in the same direction. The same
learning rate therefore moves a wide layer further than a narrow one, and
the best learning rate falls as the model widens. Once we know by how much,
we can divide the growth out, layer by layer, so that the rate tuned on the
narrowest model is also right for the widest.

We begin with the drift itself, on a family of networks that differ only
in width. The first step of Adam then explains it, and a short table of
per-layer rules removes it. :numref:`sec_batch_size` asked how the learning
rate must move with the batch. Here it moves with the model. Width is the
only kind of growth we treat, and the end of the section says what that
leaves out.

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

We want the smallest experiment in which a model can grow while remaining
the same model. A multilayer perceptron on Fashion-MNIST serves: three
hidden layers of width $n$, with $n$ from 128 to 8,192. The narrowest member
has 135 thousand parameters and the widest 141 million. Only the width
differs. Note where the width lives. The input matrix always has 784 inputs
and the output matrix always has 10 outputs, so only the two $n \times n$
matrices in the middle grow in both dimensions. This distinction matters
below. We will train well over a hundred of these models, so the whole
training set stays on the accelerator as one tensor and no data loader is
needed.

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

We specify the initialization explicitly, because it is part of what a rule
for scaling must prescribe. Such a rule is called a *parametrization*: it
states how the initial variance, the learning rate, and any constant
multiplier of each layer depend on the width. We start with the usual one.
Every weight is Gaussian with variance $1/\textrm{fan-in}$, where the
fan-in of a layer is the number of its inputs, every bias is zero, and all
layers share one learning rate. This is the *standard parametrization*
(SP). Its initialization is the He scheme of :numref:`subsec_he_init`
without the factor 2. The activations then shrink by $\sqrt{2}$ from one
layer to the next, equally at every width, which three layers can afford.
The optimizer is created by a method, `configure_adam`, so that a subclass
can override it later.

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

Training is a loop over a fixed number of steps, as in
:numref:`sec_test_models`: 400 steps of Adam at batch size 512 and a
constant learning rate. In what may two runs differ? A seed fixes both the
initialization and the sequence of minibatches, so with the same seed only
the width, the parametrization, and the learning rate are left. We score a
run by its training loss, averaged over the last 50 steps, because the
question is how well a learning rate optimizes, and the training loss is
what the optimizer acts on. A run that diverges is capped at 2.5, slightly
above the loss $\log 10 \approx 2.3$ of uniform guessing.

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

The sweep covers four widths, each four times the last, and ten learning
rates a factor of two apart, from $2^{-14} \approx 6 \times 10^{-5}$ to
$2^{-5} \approx 3 \times 10^{-2}$. Three helpers train every configuration
once per seed, report for each width the rate with the lowest
seed-averaged loss, and plot the curves. The number to watch is the slope
that `print_optimum` fits to the logarithm of the best rate against the
logarithm of the width. It is zero if the best rate does not depend on the
width, and $-1$ if the best rate falls in proportion to $1/n$.

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

Two seeds are drawn once and shared by every sweep of the section. A large
project could afford such a sweep at its narrowest width only. We can
afford all four widths, 80 runs in total, and so we see what the project
would miss.

```{.python .input #scaling-the-sweep}
%%tab pytorch, jax
widths = [128, 512, 2048, 8192]
lrs = [2.0 ** k for k in range(-14, -4)]
seeds = [random.getrandbits(32) for _ in range(2)]  # Shared by every sweep
sp_loss = sweep(MLP, widths, lrs, seeds)
print_optimum(sp_loss, lrs)
plot_sweep(sp_loss, lrs)
```

Each width traces a U: too small a learning rate leaves the network
undertrained after 400 steps, and too large a rate destabilizes it. Where a
curve leaves the top of the chart, training has failed. Far to the right
the order of the curves means little, because a run at such a rate either
survives or fails, and a mean over two seeds jumps with every failure. We
read each curve for its minimum and for the rate at which it begins to
climb.

The U's do not line up. The best rate falls from about $4 \times 10^{-3}$
at width 128 to about $2.4 \times 10^{-4}$ at width 8,192. That is sixteen
times lower, give or take one step of the grid, for a network 64 times
wider, and the fitted slope is about $-0.7$. The right arms move with the
minima: the wider the network, the lower the rate at which its loss begins
to climb.

Tuned, the wider networks are better, from a loss just over 0.32 at width
128 to 0.22 or 0.23 at width 8,192. But the tuning does not transfer. At
the best rate of the narrowest network the widest one is well up the right
arm of its U, and at the best rate of the widest network the narrowest one
has not finished training. Each ends about a tenth of a nat above its own
best. The same drift has been measured in transformers
:cite:`Yang.Hu.Babuschkin.ea.2022`.

### The First Adam Step

Why should a wider network want a smaller learning rate? Networks are
complicated, and 400 steps are many. Let us therefore start from the
simplest case there is: one layer, one training example, and the very first
step. It turns out to contain the answer.

Consider a hidden weight matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$. It
receives the activations $\mathbf{h} \in \mathbb{R}^n$ of the layer before
it and computes $\mathbf{z} = \mathbf{W}\mathbf{h}$. We need two facts.
First, the variance $1/n$ of the weights was chosen so that the entries of
$\mathbf{z}$ are about as large as those of $\mathbf{h}$, at every width.
Second, for a single example the gradient with respect to $\mathbf{W}$ is
an outer product, $\mathbf{g} = \boldsymbol{\delta} \mathbf{h}^\top$, where
$\boldsymbol{\delta}$ is the error that backpropagation delivers at
$\mathbf{z}$.

On its first step Adam moves every weight by $\eta$ against the sign of its
gradient: after bias correction the two averages are
$\hat{\mathbf{m}} = \mathbf{g}$ and $\hat{\mathbf{v}} = \mathbf{g}^2$, and
their ratio $\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$ is
$\operatorname{sign}(\mathbf{g})$ (:numref:`sec_adam`). The sign of an outer
product factorizes, $\operatorname{sign}(\delta_i h_j) =
\operatorname{sign}(\delta_i) \operatorname{sign}(h_j)$. How far does this
update move $\mathbf{z}$ on the example that produced it? With
$\overline{|h|} = \frac{1}{n} \sum_j |h_j|$, the mean absolute activation,

$$
(\Delta\mathbf{W}\, \mathbf{h})_i
= -\eta \sum_{j=1}^{n} \operatorname{sign}(\delta_i h_j)\, h_j
= -\eta\, \operatorname{sign}(\delta_i) \sum_{j=1}^{n} |h_j|
= -\eta\, \operatorname{sign}(\delta_i)\, n\, \overline{|h|}.
$$
:eqlabel:`eq_scaling_first_step`

All $n$ terms of the sum have the same sign, because the update is built
from the very activations that it multiplies. Their sizes add up instead of
cancelling. Every entry of $\mathbf{z}$ that receives an error moves by
$\eta\, n\, \overline{|h|}$, and since the entry itself was about as large as
$\overline{|h|}$, one step changes it by about $\eta n$ times its own size.
Take $\eta = 2^{-8} \approx 0.004$. At width 128 the product $\eta n$ is 0.5,
a firm nudge. At width 8,192 it is 32, and a single step replaces the
layer's output by something twenty or thirty times larger. To keep the
nudge the same, $\eta$ must fall in proportion to $1/n$.

This is a slope of $-1$, and the sweep measured $-0.7$. We should not
expect closer agreement, for the argument is crude. It follows one example,
and the sign of a minibatch gradient, a sum of outer products, no longer
factorizes. It follows one step out of 400. And it follows one kind of
layer. The input matrix has 784 inputs at every width and a bias has none,
so their steps do not grow, while the output matrix has $n$ inputs and its
step grows like those of the hidden matrices. A single $\eta$ is a
compromise between layers that scale differently, and the layers with $n$
inputs pull it down.

## Maximal Update Parametrization

### The Rules

If each layer's step scales differently with the width, one learning rate
should not have to serve them all. The *maximal update parametrization*
(muP) of :citet:`Yang.Hu.2021` writes the width into per-layer rules
:cite:`Yang.Hu.Babuschkin.ea.2022`. The name states the criterion. Updates
that grow with the width, as in :eqref:`eq_scaling_first_step`, make the
activations of a wide network blow up. Updates that shrink with the width
leave a wide enough layer where its initialization put it. In between lies
one scaling under which every layer learns as much per step as it can
without blowing up, however wide the network.

The rules say only how each quantity changes with the width. They are
therefore stated relative to a *base width* $n_{\textrm{base}}$, at which
muP and the standard parametrization coincide, and in terms of the *width
multiplier* $m = n / n_{\textrm{base}}$. For Adam they fit in a table.

:The maximal update parametrization for Adam, relative to a base width $n_{\textrm{base}}$, with $m = n/n_{\textrm{base}}$. Only the two entries with $m$ and the initial variance of the output matrix differ from the standard parametrization.
:label:`tab_mup-rules`

| parameters | initial variance | Adam learning rate | forward pass |
|:--|:--|:--|:--|
| input weights and all biases | $1/\textrm{fan-in}$; biases zero | $\eta$ | unchanged |
| hidden matrices | $1/\textrm{fan-in}$ | $\eta / m$ | unchanged |
| output matrix | $1/n_{\textrm{base}}$ at every width | $\eta$ | logits $\times\, 1/m$ |

Input weights and biases, whose fan-in never grows, are left alone. The
hidden rule is the one we have derived. A learning rate of $\eta/m$ cancels
the sum of $n$ terms: the first step now moves a hidden layer by
$(\eta/m)\, n\, \overline{|h|} = \eta\, n_{\textrm{base}}\, \overline{|h|}$,
the same at every width.

The output rule is the subtle one. For the output matrix's own update,
dividing the logits by $m$ does what $\eta/m$ did for a hidden matrix: the
step still sums $n$ terms, and its effect on the logits is divided by $m$.

Why not simply lower the learning rate here too? Because the logits also
move when the features below the output matrix move, and these changes are
not random with respect to the output weights: the error that caused them
came back through those very weights. Once more a sum over the width,
$\sum_j w_j\, \Delta h_j$, has terms that agree in sign, and it stays
bounded only if the weights have a size of order $1/n$, not $1/\sqrt{n}$.
A multiplier of $1/m$ on weights whose variance is held at
$1/n_{\textrm{base}}$ gives them that size. In effect a standard output
layer sums over the $n$ hidden units, and under muP it averages over them.
As a side effect the logits at initialization, where the signs are still
random, shrink as $1/\sqrt{m}$. We have only sketched this half of the
argument, and the papers cited above make it precise.

These exponents are for Adam. SGD has its own, tabulated beside those for
Adam by :citet:`Yang.Hu.Babuschkin.ea.2022`. In code the whole change is a
subclass with three marked lines.

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

:begin_tab:`pytorch`
The two learning rates are two parameter groups, as for the parameters
exempt from weight decay in :numref:`sec_adamw`.
:end_tab:

:begin_tab:`jax`
`optax.multi_transform` applies one optimizer per label, and `label` marks
the two square kernels by their path in the parameter tree.
:end_tab:

At the base width $m = 1$, and `MuMLP` is `MLP`. What the rules leave open
are the constants: how fast the input layer should learn relative to the
hidden ones, and how large the logits should start. The choice of base
sets them, and they have to be tuned once. A project puts the base at its
proxy, the one width that it can afford to tune: the proxy is then a
standard network, and the large model is its rescaled copy.

We do the reverse and put the base at the widest member of the family,
8,192, for the sake of a clean comparison. The widest network is then the
same network under both parametrizations, the sweep above has already
tuned it, and whatever differs there between the two is due to the
learning rate alone. The narrower members become scaled-down copies of the
wide network, not standard networks: at width 128, where $m = 1/64$, the
hidden matrices learn at $64\eta$ and the logits start 8 times larger than
in `MLP`. Transfer asks that the family be consistent, not that the proxy
be standard. Exercise :exref:`ex_scaling-base` puts the base at width 128
and measures what changes.

### The Coordinate Check
:label:`subsec_scaling-coord-check`

Rules like these are easy to get wrong, and the *coordinate check* finds
the errors in seconds. :citet:`Dey.Anthony.Hestness.2024` recommend it as
the first test of any muP implementation. It measures
:eqref:`eq_scaling_first_step` directly. We build the network at a range of
widths, take one Adam step on one minibatch, and record the mean absolute
activation of each layer, its typical coordinate, on that same minibatch.
If the scaling is right, every curve is flat in the width. One step is the
cleanest probe, because later steps compound the growth and mix it with
feedback from the loss.

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

First the standard parametrization, at every power of two from 128 to
8,192 and at the rate that the sweep found best for width 128. We can say
beforehand what to expect. Layer 1 sits behind the input matrix and should
not depend on the width. The layers behind the square matrices change by
about $\eta n$ times their size, which passes one near width 256, so their
curves should be flat for a few hundred units and climb beyond.

```{.python .input #scaling-the-coordinate-check-2}
%%tab pytorch, jax
check_widths = [128 * 2 ** k for k in range(7)]  # 128 to 8,192
lr_128 = lrs[sp_loss[128].mean(1).argmin()]  # SP's best rate at width 128
sp_acts = coord_check(MLP, check_widths, lr_128, seeds[0])
d2l.plot(check_widths, list(sp_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

Layer 1 is flat, near 0.3. Layers 2 and 3 change little up to a few
hundred units and then climb. Over the last doubling of the width layer 2
almost doubles, a slope of one on these logarithmic axes, as
:eqref:`eq_scaling_first_step` says. Layer 3 grows about three and a half
times and the logits seven- to eightfold, slopes of nearly two and three:
each layer receives the excess of the layer before it, and its own update
multiplies that by a further factor of order $\eta n$. At width 8,192 the
logits are about a thousand times larger than at width 128. This is the
right arm of the sweep, seen after a single update.

Now the same check under muP, on the same axes, at the rate that was best
at the base width. The prediction is shorter: nothing should depend on the
width.

```{.python .input #scaling-the-coordinate-check-3}
%%tab pytorch, jax
lr_base = lrs[sp_loss[8192].mean(1).argmin()]  # Best at the base width
mup_acts = coord_check(MuMLP, check_widths, lr_base, seeds[0])
d2l.plot(check_widths, list(mup_acts), 'width', 'mean |activation|',
         xscale='log', yscale='log', ylim=[0.05, 1000],
         legend=['layer 1', 'layer 2', 'layer 3', 'logits'])
```

All four curves are flat. The hidden layers stay between 0.1 and 0.2 at
every width, and the logits near 2. One might object that the rate of this
check is sixteen times smaller than that of the first, and that small
steps alone would flatten the curves. The steps are not small. At width
128 the hidden matrices learn at $64\eta \approx 1.6 \times 10^{-2}$, four
times the rate of the first check, and at every width the product
$(\eta/m)\, n = \eta\, n_{\textrm{base}}$ is 2.

The check has to be read with some care. Activations that grow with the
width are the unambiguous failure. Flat curves say only that the forward
pass is stable after a step, because the check reads the sizes of
activations and not whether anything is being learned. Updates that shrink
too fast with the width also give flat curves, with the layers frozen where
they started. The check can convict an implementation but not acquit it.
It still catches the common errors: a missing multiplier, a mislabeled
layer, or an initializer with a fixed standard deviation, like the 0.02
common in transformer code, in place of one that scales with the fan-in.
Exercise :exref:`ex_scaling-breaking-mup` breaks the rules one at a time
and asks which errors the check sees.

### Learning-Rate Transfer

The coordinate check tests one step. Whether the learning rate transfers
over a whole run is a question for the sweep, which we repeat under muP
with the same seeds. If the rules work, `print_optimum` should print the
same rate at every width, the one that was best for the standard network at
width 8,192, and a slope near zero. (At the base width muP is the standard
parametrization, so the cell reuses those runs.)

```{.python .input #scaling-learning-rate-transfer}
%%tab pytorch, jax
mup_loss = sweep(MuMLP, widths[:-1], lrs, seeds)
mup_loss[widths[-1]] = sp_loss[widths[-1]]  # At the base width muP is SP
print_optimum(mup_loss, lrs)
plot_sweep(mup_loss, lrs)
```

The best rate is $2.4 \times 10^{-4}$ or one grid step below it at every
width, and the slope is within 0.2 of zero, against $-0.7$ before. The four
curves have their minima in the same place and run side by side up to it,
the wider network below the narrower one at every rate. A rate tuned at
width 128 is, within one grid step, the best rate at width 8,192.

The right arms are another matter. They do not line up, and the first-step
argument did not predict what they do: the narrowest network leaves the
chart first, near $2 \times 10^{-3}$, and the widest holds out longest.

A second, cruder count explains the order. Ask how large a step is
compared with the weights that it changes. A hidden weight starts with a
size of about $1/\sqrt{n}$, and under muP Adam moves it by as much as
$\eta/m$ in one step. At width 128 and $\eta = 2 \times 10^{-3}$ that is a
step of 0.125 for weights of size 0.09, and a step larger than the weights
leaves little of the initialization. The two are equal at
$\eta = \sqrt{n}/n_{\textrm{base}}$, which is $1.4 \times 10^{-3}$ at width
128 and doubles with every fourfold widening. On the plot each of the
three narrower curves is still at about 0.4 one grid point below its rate
and above 0.5 at the first grid point beyond it. muP puts the optima in
one place, and it leaves a narrow network less room above its optimum than
a wide one. Fortunately this is the safe way round, since we tune narrow
and run wide.

How much loss does the transfer save? The last cell prints the best loss
at every width, and then the loss of the widest network at two rates, the
one tuned at width 128 and its own best, with both seeds.

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

Under standard parametrization the rate tuned at width 128 leaves the
widest network at about a third of a nat, where its own best rate gives
0.22 or 0.23. With either seed the borrowed rate raises the loss by 0.09
to 0.15 nats, while two seeds at the same rate differ by at most 0.05. Put
differently, at the borrowed rate the widest network is no better than the
width-128 network that the rate was tuned on, with a thousand times the
parameters. Under muP the borrowed rate is the widest network's own best
rate or its neighbour on the grid, and it adds 0.003 or less, which is
below the difference between seeds. This is *hyperparameter transfer*:
tune the small model, scale up, keep the numbers.

The first line of each block holds a second result, one that we did not
ask for. Tuned separately, the two parametrizations agree at width 8,192
by construction and nearly at width 128, but at widths 512 and 2,048 the
muP networks end 0.02 to 0.04 nats lower. Exercise
:exref:`ex_scaling-base` asks which of the inherited constants is
responsible.

On a family this small the sweep that muP saves is cheap. At scale it is
the whole point: :citet:`Yang.Hu.Babuschkin.ea.2022` tuned a
6.7-billion-parameter GPT-3 by sweeping a 40-million-parameter proxy, spent
about 7% of the pretraining compute on tuning, and outperformed the
published model.

## Other Models and Optimizers

### Rules for a Transformer

The main model of this chapter is not an MLP. For `TinyLM`
(:numref:`subsec_tinylm`), whose width is the dimension $d$ of its
embeddings, the two questions that sorted the matrices of the MLP sort
every parameter. Does it sum over a number of inputs that grows with the
width? And if so, does the number of its outputs grow as well? Recall that
a block of `TinyLM` has four matrices: `qkv` computes the queries, keys,
and values of the attention layer from its $d$ inputs, `proj` maps the
result of attention back into the block, and two more form the MLP.

:The maximal update parametrization for Adam applied to the parameters of `TinyLM`, with $m = d/d_{\textrm{base}}$.
:label:`tab_mup-transformer`

| part of `TinyLM` | why | rule |
|:--|:--|:--|
| embeddings, biases, scales and shifts of the layer normalizations | every entry acts on one coordinate, and nothing is summed over the width | as input weights: learning rate $\eta$ |
| the four matrices of every block | inputs and outputs both grow with $d$ | as hidden matrices: learning rate $\eta/m$ |
| `head` | $d$ inputs, but 256 outputs at every width | as the output matrix: initial variance fixed, logits $\times\, 1/m$ |
| attention scores | a dot product of two vectors that the layer computes | an extra factor $1/\sqrt{m}$ |

The last row is new, and it is :eqref:`eq_scaling_first_step` once more. An
attention layer scores one position against another by the dot product of
a query and a key, two vectors that the layer computes, and divides the
score by the square root of their dimension (:numref:`chap_attention`).
The square root is the right scale for a sum of terms with random signs,
which is what the dot product is at initialization. Training aligns
queries with keys, the terms then agree in sign, and their sum grows in
proportion to the dimension. muP therefore divides by the dimension
itself, which relative to the base is the extra factor $1/\sqrt{m}$.
Exercise :exref:`ex_scaling-tinylm` applies the table to `TinyLM` and runs
the coordinate check.

### Muon and Width

The first-step count :eqref:`eq_scaling_first_step` asked how far an
update moves the output of a layer on the one input that produced it. The
spectral norm $\|\Delta\mathbf{W}\|_2$ asks the same of the worst input:
it is the largest factor by which the update can stretch any vector. Muon
(:numref:`sec_muon`) sets this norm by construction. Does it then need
rules like those above? The answer turns on a constant that
:numref:`sec_muon` chose for a different purpose.

Let us first compute the spectral norm of Adam's first step. For a single
example the update is
$\Delta\mathbf{W} = -\eta\, \mathbf{s}\mathbf{t}^\top$ with
$\mathbf{s} = \operatorname{sign}(\boldsymbol{\delta})$ and
$\mathbf{t} = \operatorname{sign}(\mathbf{h})$, a matrix of rank one. Such
a matrix stretches no vector more than $\mathbf{t}$ itself, so its spectral
norm is the product of the two lengths,

$$
\|\Delta\mathbf{W}\|_2 = \eta\, \|\mathbf{s}\|_2\, \|\mathbf{t}\|_2.
$$

The length of a vector of signs is the square root of the number of its
nonzero entries. Behind a ReLU about half the entries are nonzero on
either side, so for an $n \times n$ matrix the norm is about $\eta n / 2$.
It grows in proportion to the width, as before.

How large should it be? Take a matrix with $d_{\textrm{in}}$ inputs and
$d_{\textrm{out}}$ outputs. An input with entries of order one has length
$\sqrt{d_{\textrm{in}}}$ and an output with entries of order one has length
$\sqrt{d_{\textrm{out}}}$, so an update that changes every output entry by
an amount of order one needs

$$
\|\Delta\mathbf{W}\|_2 \approx \sqrt{\frac{d_{\textrm{out}}}{d_{\textrm{in}}}},
$$
:eqlabel:`eq_spectral_condition`

up to a factor that does not depend on the width. For a square matrix
this is a constant: the norm should not grow at all.
:citet:`Yang.Simon.Bernstein.2023` show that this *spectral condition*,
imposed on the initial weights and on every update, yields the rules of
muP. Exercise :exref:`ex_scaling-spectral` checks it for the updates.

Now Muon. Its update is an orthogonalized matrix, whose largest singular
values are about one, times the factor
$0.2\sqrt{\max(d_{\textrm{out}}, d_{\textrm{in}})}$ that
:numref:`sec_muon` chose so that the entries of the update are as large as
those of an AdamW step. For a square matrix this gives a spectral norm of
about $0.2\, \eta \sqrt{n}$. It grows with the width, more slowly than
Adam's $\eta n / 2$, but it grows. The factor that made Muon exchangeable
with AdamW at one width removed the independence of width that the
spectral norm would have given it.

The answer therefore has two halves. With the factor of
:numref:`sec_muon`, Muon still needs a learning rate that falls as the
model widens, in proportion to $1/\sqrt{n}$ where Adam needs $1/n$. With
$\sqrt{d_{\textrm{out}} / d_{\textrm{in}}}$ in its place
:cite:`Bernstein.2025`, the update meets :eqref:`eq_spectral_condition` at
every width by construction, and no rule is needed.
:citet:`Shah.Polloreno.Stratos.ea.2025` combine Muon with muP to transfer
hyperparameters across model sizes.

### Limits of Transfer

Should you use muP? Practice is divided, and both camps have trained good
models. Whichever camp you join, run the coordinate check on every family
of models that you mean to scale. It costs seconds, and the errors it
finds otherwise appear as a diverged run at the largest size.

Next, never carry a learning rate across a large change of width
unchanged. Either write the width into the per-layer rates as above, or
keep the standard parametrization and measure the drift: sweep three or
four small sizes, fit the slope, and extrapolate.
:citet:`Bi.Chen.Chen.ea.2024` set the learning rate and the batch size of
a production model this way, with power laws in the compute budget. The
fit uses the measured exponent, $-0.7$ here, where the rules use the $-1$
of the first step, but it moves one number for all layers. Finally, in
either case sweep the transferred rate once more, a factor of two each
way, at the largest size you can afford to run three times.

The strongest objection to muP concerns long runs with weight decay. Our
argument is about the first step, and our runs last 400 steps without
decay. In a long run with AdamW and a decay of $\lambda$ the weights of
every matrix settle where decay and updates balance, and from then on each
matrix turns by the same relative step $\sqrt{2\eta\lambda}$ per update,
whatever its width (:numref:`sec_adamw`).
:citet:`Kosson.Welborn.Liu.ea.2025` report that in language-model training
the alignment between updates and activations, on which
:eqref:`eq_scaling_first_step` rests, lasts only for a short initial
phase, and that afterwards weight decay, not the parametrization, keeps
the updates of different widths comparable. On
their reading muP's smaller rates for wide layers act mainly as a warmup.
We consider the question open. It does not touch the coordinate check,
which is about the first steps under any reading.

Width is one kind of growth, and the one in which a model stays most
nearly the same model. Depth, batch size, and the length of training grow
with it in practice. :numref:`sec_batch_size` gave rules for the batch
alone, but for all of them growing together no rules as short as those of
:numref:`tab_mup-rules` are established. A learning rate that crossed a
64-fold change of width in this section has still to cross those, in a run
that cannot be repeated. :numref:`sec_practice` is about such runs.

## Summary

A learning rate tuned on a small model does not carry over to a wider one:
one Adam step changes the output of a layer with $n$ inputs in proportion
to $n$, because all $n$ terms of the update have the same sign. In our
family of networks the best learning rate fell sixteenfold as the width
grew 64-fold. The maximal update parametrization divides this growth out
layer by layer: hidden matrices learn at $\eta/m$, the logits are divided
by the width multiplier $m$, and input weights and biases are left alone.
With these rules the best rate stayed within one grid step over the whole
range, and the rate tuned at width 128 cost nothing at width 8,192, where
under the standard parametrization it cost a tenth of a nat. The
coordinate check, one step at several widths, finds errors in such rules
in seconds, although flat curves do not prove an implementation right.
Muon, too, needs a rule while its update is scaled to match AdamW, since
the spectral norm of that update grows as the square root of the width.
The rules cover width and describe the first steps best; long runs with
weight decay and growth in depth are open.

## Exercises

1. [code] **Breaking muP on purpose.** The coordinate check is a test for
   implementation errors. Put the base at width 128 throughout this
   exercise, so that $m \geq 1$ at every width of `check_widths`, by
   passing `lambda w, **kw: MuMLP(w, base_width=128, **kw)` as `arch`,
   and use the rate `lr_128`.
    1. Delete the division of the features by $m$ from `MuMLP`, keep the
       other two rules, and rerun `coord_check`. Which curve exposes the
       error?
    1. Restore the division and give the hidden matrices the full learning
       rate instead. Which curves expose this error?
    1. Restore the hidden rule and try the obvious alternative to the
       output rule: a standard output matrix, with variance
       $1/\textrm{fan-in}$ and features that are not divided, which learns
       at $\eta/m$. Do the logits stay flat? Split their change into the
       part due to the update of the output matrix and the part due to
       the change of the features, and find the one that grows with the
       width.
    1. Restore the output rule, but divide the hidden learning rate by
       $m^2$ instead of $m$. Does `coord_check` flag this error? Now run
       `sweep` and `print_optimum` on this variant and on the correct
       rules for widths 128, 512, and 2,048, with a few rates added above
       the largest of `lrs`. In which direction does the best rate drift
       under the wrong rule? Explain why a sweep at a single width could
       not have found the error.
   :exlabel:`ex_scaling-breaking-mup`

1. [code] **Choosing the base.** `MuMLP` puts its base at width 8,192, the
   widest member of the family.
    1. Rerun the muP sweep with `base_width=128` at all four widths, so
       that muP is the standard parametrization at the width we tune.
       Does `print_optimum` still report a slope near zero? Compare the
       best loss at each width with `mup_loss` and `sp_loss`.
    1. The two bases give networks of width 8,192 that differ only by
       constants: the learning rate of the input weights and biases
       relative to that of the hidden matrices, and the size of the
       logits at initialization. Compute both ratios. Then change one
       constant at a time and find which of them accounts for the
       difference in loss at width 8,192.
    1. `mup_loss` is below `sp_loss` at widths 512 and 2,048. Which of
       the constants is responsible there?
   :exlabel:`ex_scaling-base`

1. [code] **Fitting the drift.** Instead of removing the drift, a
   project can measure it on small models and extrapolate. Here the
   target is the widest network, and the small models are the other
   three.
    1. From `sp_loss`, fit a power law $\eta^\star(n) = c\, n^{-\gamma}$ to
       the best rates at widths 128, 512, and 2,048 only, and predict the
       best rate at width 8,192. How many grid steps separate the
       prediction from the measured optimum? What loss does `sp_loss`
       hold for the grid rate nearest to the prediction?
    1. Since `lrs` is spaced by factors of two, each optimum is resolved
       only to within a factor of $\sqrt{2}$. Give the range of $\gamma$
       consistent with this resolution over all four widths, and state
       whether it contains $\gamma = 1$.
    1. Repeat the first part separately for each of the two seeds. Do
       the two fits agree within the range of the second part?

1. [code] **Depth and the coordinate check.** ● `coord_check` varies the
   width at a fixed depth of three hidden layers. Generalize `MLP` and
   `MuMLP` to $L$ hidden layers and run the check across
   $L \in \{3, 6, 12, 24\}$ at width 512 under both parametrizations. The
   variance $1/\textrm{fan-in}$ of `MLP` halves the second moment of the
   ReLU activations at every layer (:numref:`subsec_he_init`), so for
   this problem initialize the input and hidden matrices with variance
   $2/\textrm{fan-in}$.
    1. Under standard parametrization, how does the growth of the logits
       with depth compare with their growth with width in
       :numref:`subsec_scaling-coord-check`?
    1. Does muP's width scaling keep the activations flat across depth?
       State what the rules of :numref:`tab_mup-rules` promise and what
       they do not.

1. **Spectral norms of first updates.** ● For a single example, Adam's
   first update of a weight matrix
   $\mathbf{W} \in \mathbb{R}^{d_{\textrm{out}} \times d_{\textrm{in}}}$
   has spectral norm
   $\eta \sqrt{k_{\textrm{out}} k_{\textrm{in}}}$, where
   $k_{\textrm{out}}$ and $k_{\textrm{in}}$ count the nonzero entries of
   $\boldsymbol{\delta}$ and $\mathbf{h}$. Assume that they are fixed
   fractions of $d_{\textrm{out}}$ and $d_{\textrm{in}}$.
    1. For the input, hidden, and output matrices of `MLP`, find how
       $\eta$ must scale with the width $n$ for $\Delta\mathbf{W}$ to
       satisfy the spectral condition :eqref:`eq_spectral_condition`.
    1. Show that the learning rates and the logit multiplier of
       :numref:`tab_mup-rules` give every layer an update at this scale.
       For the output matrix, count the multiplier as part of the matrix.
    1. Muon's update of the same matrix has a spectral norm of about
       $0.2\, \eta \sqrt{\max(d_{\textrm{out}}, d_{\textrm{in}})}$. How
       must $\eta$ scale with $d$ for the matrices `qkv`
       ($3d \times d$), `proj` ($d \times d$) and the two matrices of the
       MLP ($4d \times d$ and $d \times 4d$) of `TinyLM` to satisfy the
       condition? Under one shared learning rate, which of the four takes
       the largest step relative to
       $\sqrt{d_{\textrm{out}} / d_{\textrm{in}}}$, and by what factor?
   :exlabel:`ex_scaling-spectral`

1. **Weight decay and width.** :citet:`Kosson.Welborn.Liu.ea.2025` argue
   that in long runs weight decay, not the parametrization, keeps updates
   comparable across widths. :numref:`subsec_wd-at-scale` showed that
   under AdamW with learning rate $\eta$ and decay $\lambda$ a weight
   matrix settles after about $1/(\eta\lambda)$ steps at an RMS of
   $\sqrt{\eta/(2\lambda)}$, and from then on takes relative steps of
   $\sqrt{2\eta\lambda}$.
    1. Under standard parametrization with one $\eta$ and one $\lambda$,
       does the settled relative step of a hidden matrix depend on its
       width?
    1. Under muP the hidden matrices learn at $\eta/m$. With $\lambda$
       unchanged, how do their timescale and their settled relative step
       depend on $m$? How must $\lambda$ scale with $m$ for both to be
       independent of the width?
    1. Had the runs of this section used AdamW with $\lambda = 0.1$ at
       the rate $4 \times 10^{-3}$, how many timescales would their 400
       steps have covered? State what the experiments of this section can
       and cannot say about the cited claim.

1. [code] **The rules on `TinyLM`.** ● Apply
   :numref:`tab_mup-transformer` to `d2l.TinyLM`
   (:numref:`subsec_tinylm`), with `num_heads=2` at every width.
    1. Sort the parameters of the model into the rows of the table. At
       `d_model=128`, what share of the parameters receives the learning
       rate $\eta/m$?
    1. Write a coordinate check for `TinyLM`: for `d_model` from 64 to
       1,024, take one Adam step at a rate of $3 \times 10^{-3}$ on one
       minibatch of `d2l.WikiText103` and record the mean absolute value
       of the output of each attention layer, of each MLP, and of the
       logits. Which of them grow with the width under the standard
       parametrization?
    1. With the base at `d_model=64`, add the rule for the block
       matrices, then the rule for the `head`, and repeat the check
       after each. Which rule flattens which curve?
    1. Every block of `TinyLM` normalizes its input. Which step of the
       argument that layer 3 grows faster than layer 2 in
       :numref:`subsec_scaling-coord-check` does this remove?
   :exlabel:`ex_scaling-tinylm`

[Discussions](https://d2l.discourse.group/t/36766)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.11]{.kicker}

Scaling Up<br>
**the drifting optimum · the first Adam step · muP · the coordinate check · transfer across widths**
:::
:::

::: {.slide title="A Learning Rate Tuned at Width 128 Wastes a Network a Thousand Times Larger"}
[The problem]{.kicker}

MLP on Fashion-MNIST, 400 Adam steps, training loss:

| network | learning rate | loss |
|:--|:--|--:|
| width 128, 135 thousand parameters | its own best | 0.32 to 0.33 |
| width 8,192, 141 million parameters | the one tuned at width 128 | 0.31 to 0.38, by seed |
| width 8,192 | its own best | 0.22 to 0.24, by seed |

::: {.d2l-note .warn}
A large run allows no sweep. Its learning rate comes from a small proxy,
and by default that rate is wrong.
:::

With width-aware rules, a 6.7-billion-parameter GPT-3 was tuned on a
40-million-parameter proxy for about 7% of its pretraining compute.

[Yang et al., 2022]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain why the best learning rate falls with width;
- state the muP rules for Adam and implement them;
- run a coordinate check and say what it cannot detect;
- read a sweep for its optimum and for the cost of transfer;
- compute the spectral norm of a first update.

**Idea:** a layer with $n$ inputs sums $n$ terms of one sign in every
update; per-layer rules divide the width out.

**Running example:** an MLP $784 \to n \to n \to n \to 10$ on
Fashion-MNIST, $n = 128$ to $8{,}192$, 400 Adam steps, two seeds.
:::

::: {.slide title="Scaling Up Moves the Learning Rate with the Model"}
[Where this fits]{.kicker}

| movement | sections |
|---|---|
| two difficulties, on a valley we can solve | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| which direction to step in, on two small networks | Two Test Models · Momentum · Adam · AdamW · Muon |
| how the step changes over a run, with batch size and width | Schedules · Batch Size · **Scaling Up** · Practice |

- **Uses:** Adam's first step is $\eta$ times the sign of the gradient;
  the spectral norm and Muon's update; AdamW's settled relative step.
- **Skips:** growth in depth, batch size and training length; the
  exponents for SGD.
:::

::: {.slide title="Only the Two Middle Matrices Grow in Both Dimensions"}
[A family of widths]{.kicker}

MLP $784 \to n \to n \to n \to 10$, $n$ = 128, 512, 2,048, 8,192:

| matrix | shape | fan-in (inputs) | grows with $n$ |
|:--|:--|:--|:--|
| input | $n \times 784$ | 784 | outputs only |
| hidden (two) | $n \times n$ | $n$ | **inputs and outputs** |
| output | $10 \times n$ | $n$ | inputs only |

**Standard parametrization (SP):** weights of variance
$1/\textrm{fan-in}$, zero biases, one learning rate $\eta$ for all layers.

**Score of a run:** mean training loss of the last 50 of 400 steps.
:::

::: {.slide title="Under SP the Best Learning Rate Falls Sixteenfold over a 64-Fold Widening"}
[Learning-rate sweep]{.kicker}

Four widths, ten learning rates a factor of two apart, two seeds:

@!scaling-the-sweep

Slope of log best rate against log width: about $-0.7$. Each end of the
family, run at the other end's best rate, loses about 0.1 nats.
:::

::: {.slide title="Adam's First Step Adds n Terms of One Sign"}
[The first Adam step]{.kicker}

One example. Hidden matrix $\mathbf{W}$ ($n \times n$), input activations
$\mathbf{h}$, output $\mathbf{z} = \mathbf{W}\mathbf{h}$, backpropagated
error $\boldsymbol{\delta}$ at $\mathbf{z}$, learning rate $\eta$.

I. $\;$Gradient: $\mathbf{g} = \boldsymbol{\delta}\mathbf{h}^\top$

II. $\;$First Adam step:
$\Delta\mathbf{W} = -\eta\operatorname{sign}(\mathbf{g})
= -\eta\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$

III. Its effect on $\mathbf{z}$:

$$\boxed{(\Delta\mathbf{W}\mathbf{h})_i = -\eta \operatorname{sign}(\delta_i)
\sum_{j=1}^n |h_j| = -\eta \operatorname{sign}(\delta_i)\, n\,
\overline{|h|}}$$

$\overline{|h|}$: mean absolute activation, about the size of $z_i$
itself. One step changes $z_i$ by about $\eta n$ times its size.
:::

::: {.slide title="The Same η Is a Nudge at Width 128 and a Thirtyfold Change at Width 8,192"}
[The first Adam step]{.kicker}

Relative change of a hidden layer in one step, $\eta n$, at
$\eta = 2^{-8} \approx 0.004$:

| width $n$ | 128 | 512 | 2,048 | 8,192 |
|:--|--:|--:|--:|--:|
| $\eta n$ | 0.5 | 2 | 8 | 32 |

- Equal steps at every width need $\eta \propto 1/n$: a slope of $-1$.
- Measured: $-0.7$. The count is crude: one example, one step, and only
  the matrices with $n$ inputs.

::: {.d2l-note .rule}
One step moves a layer's output by about $\eta$ times its fan-in.
:::
:::

::: {.slide title="Check: Which Matrices Take Larger Steps When the Width Doubles?"}
[Check]{.kicker}

The width $n$ of the MLP doubles at a fixed learning rate. By what factor
does the first Adam step change the output of the input matrix
($n \times 784$), of a hidden matrix ($n \times n$), and of the output
matrix ($10 \times n$)?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A step moves the output by about $\eta$ times the fan-in.
Input matrix: fan-in 784, no change. Hidden and output matrices: fan-in
$n$, twice as far.
:::
:::
:::

::: {.slide title="muP Writes the Width into Three Per-Layer Rules"}
[The rules]{.kicker}

Base width $n_{\textrm{base}}$, width multiplier
$m = n / n_{\textrm{base}}$, Adam:

| parameters | initial variance | learning rate | forward pass |
|:--|:--|:--|:--|
| input weights, all biases | $1/\textrm{fan-in}$; biases 0 | $\eta$ | unchanged |
| hidden matrices | $1/\textrm{fan-in}$ | $\eta/m$ | unchanged |
| output matrix | $1/n_{\textrm{base}}$ at every width | $\eta$ | logits $\times\, 1/m$ |

"Maximal update": the largest updates that do not grow with the width.
Smaller ones leave a wide layer where its initialization put it.

[Yang and Hu, 2021; Yang et al., 2022]{.small}
:::

::: {.slide title="The Hidden Rule Cancels the Sum, and the Output Layer Averages"}
[The rules]{.kicker}

- **Hidden matrices:** first step
  $(\eta/m)\, n\, \overline{|h|} = \eta\, n_{\textrm{base}}\, \overline{|h|}$
  at every width.
- **Output matrix, its own update:** logits
  $= \frac{1}{m}\mathbf{W}\mathbf{h}$, so every update acts $m$ times
  less.
- **Output matrix, updates of the features $\mathbf{h}$:** the error
  reached them through $\mathbf{W}$, so $\mathbf{W}\,\Delta\mathbf{h}$
  sums $n$ terms of one sign. It stays bounded only for weights of size
  $1/n$: variance $1/n_{\textrm{base}}$, times $1/m$.
- **Code:** output weights $\times \sqrt{m}$ once, features $\div\, m$,
  hidden rate $\eta/m$.

::: {.d2l-note}
A standard output layer sums over $n$ hidden units. Under muP it
averages, and the initial logits shrink as $1/\sqrt{m}$.
:::
:::

::: {.slide title="The Base Width Fixes the Constants That the Rules Leave Open"}
[The rules]{.kicker}

- At $n = n_{\textrm{base}}$, $m = 1$: muP is SP.
- The rules give exponents. The base gives constants: the input rate
  relative to the hidden rate, and the size of the initial logits.
- A project puts the base at its proxy. Here it is 8,192, the widest
  network: the same network under SP and muP, already tuned by the sweep.

| width | $m$ | hidden rate | initial logits, relative to SP |
|:--|--:|--:|--:|
| 8,192 | 1 | $\eta$ | 1 |
| 128 | 1/64 | $64\,\eta$ | 8 |
:::

::: {.slide title="One Adam Step under SP: Slopes of One, Two and Three"}
[The coordinate check]{.kicker}

Mean |activation| of each layer after one Adam step on one minibatch, at
the rate tuned for width 128:

@!scaling-the-coordinate-check-2

Layer 1 is flat. Each later layer multiplies the excess by a further
factor of order $\eta n$; the logits end about 1,000 times larger at width
8,192.
:::

::: {.slide title="Under muP All Four Curves Are Flat"}
[The coordinate check]{.kicker}

The same check, at the rate tuned for the base width. At width 128 the
hidden matrices learn at $64\eta$, four times the rate of the SP check:

@!scaling-the-coordinate-check-3

::: {.d2l-note .warn}
Growth with width convicts an implementation. Flat curves do not acquit
it: updates that shrink too fast with width are flat too.
:::
:::

::: {.slide title="Check: Which Curve Shows a Missing Logit Multiplier?"}
[Check]{.kicker}

An implementation of muP with $m \geq 1$ gives the hidden matrices
$\eta/m$ but forgets to divide the logits by $m$. What does the coordinate
check show?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The hidden layers stay flat. The output matrix still sums $n$
terms of one sign in its first step, so the logit curve grows in
proportion to the width.
:::
:::
:::

::: {.slide title="Under muP the Best Learning Rate Stays in One Place"}
[Learning-rate transfer]{.kicker}

The same sweep with the same seeds, under muP:

@!scaling-learning-rate-transfer

Best rate $2.4 \times 10^{-4}$ or one grid step below, at every width.
Slope within 0.2 of zero, against $-0.7$ under SP.
:::

::: {.slide title="muP Leaves a Narrow Network Less Room above Its Optimum"}
[Learning-rate transfer]{.kicker}

A hidden weight has size about $1/\sqrt{n}$; Adam moves it by up to
$\eta/m$ per step. The two are equal at $\eta = \sqrt{n}/n_{\textrm{base}}$:

| width $n$ | 128 | 512 | 2,048 | 8,192 |
|:--|--:|--:|--:|--:|
| step = weight at $\eta$ = | $1.4 \times 10^{-3}$ | $2.8 \times 10^{-3}$ | $5.5 \times 10^{-3}$ | $1.1 \times 10^{-2}$ |
| loss passes 0.5 at $\eta$ = | $2.0 \times 10^{-3}$ | $3.9 \times 10^{-3}$ | $7.8 \times 10^{-3}$ | $1.6 \times 10^{-2}$ or above |

- The three narrower curves pass 0.5 at the first grid point above their
  rate, from about 0.4 one point below it.

::: {.d2l-note}
The safe way round: we tune narrow and run wide, and the wide network has
more room.
:::
:::

::: {.slide title="Transfer Costs a Tenth of a Nat under SP and Nothing under muP"}
[Learning-rate transfer]{.kicker}

Width 8,192 at the rate tuned for width 128, and at its own best rate:

@!scaling-learning-rate-transfer-1

- SP: either seed loses 0.09 to 0.15 nats; two seeds at one rate differ
  by at most 0.05.
- muP: at most 0.003 above its own best.
:::

::: {.slide title="For a Transformer, Ask Whether Inputs and Outputs Grow with the Width"}
[Rules for a transformer]{.kicker}

`TinyLM`, a small transformer language model of width $d$;
$m = d/d_{\textrm{base}}$; Adam:

| part | rule |
|:--|:--|
| embeddings, biases, normalization scales and shifts (no sum over the width) | learning rate $\eta$ |
| the four matrices of a block (inputs and outputs grow with $d$) | learning rate $\eta/m$ |
| output head ($d$ inputs, always 256 outputs) | initial variance fixed, logits $\times\, 1/m$ |
| attention scores (dot product of two learned vectors) | extra factor $1/\sqrt{m}$ |

Training aligns the two vectors, so their dot product grows as the
dimension, not as its square root.
:::

::: {.slide title="Adam's First Update Has Spectral Norm η√(k_out k_in)"}
[Muon and width]{.kicker}

Spectral norm $\|\Delta\mathbf{W}\|_2$: the largest factor by which
$\Delta\mathbf{W}$ stretches a vector.

I. $\;\Delta\mathbf{W} = -\eta\,\mathbf{s}\mathbf{t}^\top$,
$\mathbf{s} = \operatorname{sign}(\boldsymbol{\delta})$,
$\mathbf{t} = \operatorname{sign}(\mathbf{h})$: rank one

II. $\;\|\mathbf{s}\mathbf{t}^\top\|_2 = \|\mathbf{s}\|_2\,\|\mathbf{t}\|_2$

III. With $k_{\textrm{out}}$, $k_{\textrm{in}}$ nonzero entries of
$\boldsymbol{\delta}$, $\mathbf{h}$:

$$\boxed{\|\Delta\mathbf{W}\|_2 = \eta\sqrt{k_{\textrm{out}}\,k_{\textrm{in}}}}$$

ReLU, $n \times n$: half the entries are nonzero on either side, so the
norm is about $\eta n/2$. It grows with the width.
:::

::: {.slide title="Check: How Large Should the Spectral Norm of an Update Be?"}
[Check]{.kicker}

A matrix has $d_{\textrm{in}}$ inputs and $d_{\textrm{out}}$ outputs. An
input with entries of order one has length $\sqrt{d_{\textrm{in}}}$. How
large must $\|\Delta\mathbf{W}\|_2$ be for the update to change every
output entry by an amount of order one?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The change of the output must have length
$\sqrt{d_{\textrm{out}}}$, so
$\|\Delta\mathbf{W}\|_2 \approx \sqrt{d_{\textrm{out}}/d_{\textrm{in}}}$,
the *spectral condition*. For an $n \times n$ matrix this is a constant,
while Adam's $\eta n/2$ grows with $n$.
:::
:::

[Yang, Simon and Bernstein, 2023]{.small}
:::

::: {.slide title="Muon at AdamW's Entry Scale Still Grows as √n"}
[Muon and width]{.kicker}

Muon's update: the averaged gradient with its singular values pushed to
about one, times a shape factor ($d_{\textrm{in}}$ inputs,
$d_{\textrm{out}}$ outputs). For an $n \times n$ matrix:

| update | spectral norm | $\eta$ must scale as |
|:--|:--|:--|
| Adam, first step | $\eta n/2$ | $1/n$ |
| Muon $\times\; 0.2\sqrt{\max(d_{\textrm{out}}, d_{\textrm{in}})}$, entries as large as AdamW's | $0.2\,\eta\sqrt{n}$ | $1/\sqrt{n}$ |
| Muon $\times \sqrt{d_{\textrm{out}}/d_{\textrm{in}}}$ | $\eta$ | constant |

[Bernstein, 2025; Shah et al., 2025 combine Muon with muP]{.small}
:::

::: {.slide title="Check the Family, Move the Rate with the Width, Sweep Once More"}
[Limits of transfer]{.kicker}

- Coordinate check on every family you scale: seconds.
- Never carry $\eta$ across a large change of width unchanged: per-layer
  rules, or a fitted slope under SP ($-0.7$ here; the first-step count
  gives $-1$).
- Sweep the transferred rate once more, a factor of two each way.

::: {.d2l-note .warn}
Open: under AdamW every matrix settles at a relative step
$\sqrt{2\eta\lambda}$ ($\lambda$: weight decay), whatever its width. muP
may then act mainly as a warmup.
:::

[Bi et al., 2024; Kosson et al., 2025]{.small}
:::

::: {.slide title="Check: By What Factor Should the Learning Rate Fall from Width 128 to 2,048?"}
[Check]{.kicker}

A network is widened from 128 to 2,048, so $m = 16$. Give the factor for
(a) the hidden matrices under muP with Adam, (b) Muon at AdamW's entry
scale, (c) the shared rate under SP, from the measured slope of $-0.7$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (a) $m = 16$. (b) $\sqrt{m} = 4$. (c) $16^{0.7} \approx 7$;
in the sweep the best rate fell from about $4 \times 10^{-3}$ to about
$5 \times 10^{-4}$, a factor of 8.
:::
:::
:::

::: {.slide title="Wider Layers Sum More Terms, and muP Divides the Sum Out"}
[Wrap-up]{.kicker}

- **Drift:** one Adam step moves a layer by about $\eta$ times its
  fan-in; under SP the best rate fell sixteenfold from width 128 to 8,192.
- **muP:** hidden rate $\eta/m$, logits $\times\, 1/m$; the best rate
  stays within one grid step.
- **Transfer:** the width-128 rate costs 0.1 nats at width 8,192 under
  SP, at most 0.003 under muP.
- **Coordinate check:** growth convicts, flat curves do not acquit.
- **Muon** at AdamW's entry scale: spectral norm $0.2\,\eta\sqrt{n}$, so
  its rate must fall as $1/\sqrt{n}$.

**Next:** configuring, protecting and finishing a run that cannot be
repeated.
:::
