```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Modern Training Recipes for Convnets
:label:`sec_training_recipes`

Throughout this chapter we have worked hard to hold the training procedure
constant within each section so that accuracy differences could be attributed
to architecture. Even under those controlled conditions, simply changing the
peak learning rate shifted VGG's accuracy by 1.5 percentage points
(:numref:`sec_blocks`). Published benchmarks are often far less controlled: a
newly proposed network is trained with the latest optimization and
regularization tricks, then compared against a classic baseline whose accuracy
is copied from a paper written years earlier. How much of a reported
breakthrough actually comes from the *training recipe*—the optimizer,
learning-rate schedule, data augmentation, regularization, and training
duration—rather than from the architecture itself?

ResNet-50, the deeper sibling of the ResNet-18 we built in
:numref:`sec_resnet`, supplies a striking answer. Without altering a single
layer of the 2015 architecture, :citet:`wightman2021resnet` retrained
ResNet-50 using early-2020s training practices and lifted its ImageNet top-1
accuracy from 76.1% to 80.4%—a 4.3-point jump driven purely by the training
recipe. That gain rivals or exceeds the improvements claimed by many novel
architectures published over the intervening years. Consequently, whenever a
paper claims a two-point win over a baseline trained with an outdated recipe,
we cannot tell whether the architecture or the recipe deserves the credit.

In this section, we examine what changed between 2015 and 2022, implement
each core ingredient in a few lines of code, and train an identical ResNet-18
under both a classic and a modern recipe to observe the difference directly.
None of these techniques is tied to convolutions—the exact same toolkit powers
modern Transformer training—making this a general foundation for modern deep
learning practice.

```{.python .input #training-recipes-training-recipes-matter}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #training-recipes-modern-training-recipes-for-convnets-1}
%%tab mxnet
import mxnet as mx
import random
```

```{.python .input #training-recipes-modern-training-recipes-for-convnets-1}
%%tab pytorch
import random
```

## Training Recipes from 2015 to 2022

The original 2015 ResNet recipe was already thoughtfully tuned: SGD with
momentum (:numref:`sec_efficient_cnns`), weight decay, step-wise learning-rate
decay (dividing by 10 twice), and random cropping with horizontal flips
:cite:`He.Zhang.Ren.ea.2016`. Over the next several years, researchers
revisited every single piece of that pipeline—first individually and then in
large-scale systematic combinations
:cite:`He.Zhang.Zhang.ea.2019,wightman2021resnet`.
:numref:`tab_recipe_2015_2022` contrasts the 2015 baseline with a
representative early-2020s recipe; we unpack and implement each new component
immediately below.

:A 2015 ImageNet recipe and a representative high-accuracy recipe from the
early 2020s. The later column is not a universal standard; strong recipes still
vary by model and compute budget.
:label:`tab_recipe_2015_2022`

| Ingredient | 2015 recipe | 2022 recipe |
|:--|:--|:--|
| Optimizer | SGD with momentum | AdamW or LAMB; tuned SGD remains competitive |
| Schedule | step decay (divide by 10 twice) | cosine decay with linear warmup |
| Augmentation | random crops, horizontal flips | RandAugment, Mixup, CutMix, random erasing |
| Regularization | weight decay | + label smoothing, stochastic depth |
| Duration | 90 epochs | 300 to 600 epochs |
| Evaluated weights | final weights | exponential moving average of the weights |

Two observations are essential when reading :numref:`tab_recipe_2015_2022`.
First, its rows interact tightly rather than acting in isolation. Stronger
data augmentation and regularization make the training objective harder to
fit, which in turn demands far more training epochs—explaining why training
schedules grew from 90 epochs to 300--600. Regularization strength and
training duration must scale together: piling heavy augmentation onto a short
run frequently *hurts* accuracy compared with a lighter recipe.

Second, the optimizer row reflects an expanded set of options rather than a
one-size-fits-all mandate. Adaptive optimizers such as Adam (covered in full
in :numref:`chap_optimization`) normalize each parameter's gradient by a
running estimate of its magnitude so every coordinate updates at roughly the
learning rate's scale. Classic implementations folded weight decay into the
gradient before this rescaling, inadvertently distorting the penalty. *AdamW*
:cite:`Loshchilov.Hutter.2019` decouples weight decay from the adaptive
gradient step, shrinking every weight directly by a fraction equal to
learning rate times decay—a clean fix that made adaptive optimizers
competitive with well-tuned SGD across vision benchmarks. (LAMB is a
large-batch cousin of AdamW that we will not need here.)

## Implementing the Ingredients

Each modern ingredient takes only a few lines of code. Below we implement
label smoothing, cosine warmup schedules, Mixup, stochastic depth, and
exponential weight averaging. (We omit RandAugment :cite:`cubuk2020randaugment`,
which applies random sequences of standard image transforms at a tuned
intensity, since in practice it is invoked as a preprocessing library call
whose value lies in tuning rather than implementation.)

### Label Smoothing

Under standard cross-entropy with one-hot targets, the loss on a correctly
classified example keeps falling without bound as the true class's logit
grows toward $+\infty$. Because no finite logits minimize the objective, the
network is continually pushed toward extreme, poorly calibrated confidence.
*Label smoothing* :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` curbs this by
replacing the hard one-hot target with a softened distribution that places
weight $1-\epsilon$ on the true class $y$ and spreads $\epsilon$ uniformly
across all $K$ classes. Writing $p_k$ for the predicted softmax probabilities,
the per-example loss becomes

$$
l_\epsilon = -(1-\epsilon) \log p_y - \frac{\epsilon}{K} \sum_{k=1}^K \log p_k.
$$
:eqlabel:`eq_label_smoothing`

The first term is ordinary cross-entropy scaled by $1-\epsilon$, while the
second penalizes pushing any class probability toward zero. The loss achieves
its minimum at $p_y = 1-\epsilon+\epsilon/K$ and $p_k = \epsilon/K$ for
$k \neq y$, which corresponds to a clean, finite margin between logits. Look
at what happens to an already confident, correct prediction under unsmoothed
versus smoothed cross-entropy:

```{.python .input #training-recipes-label-smoothing}
%%tab mxnet
def smoothed_ce(Y_hat, y, epsilon):
    """Label-smoothed cross-entropy of each example."""
    log_p = npx.log_softmax(Y_hat)
    return -(1 - epsilon) * npx.pick(log_p, y) - epsilon * log_p.mean(axis=1)

logits, y = mnp.array([[10.0, 0.0, 0.0]]), mnp.array([0])
for epsilon in (0.0, 0.1):
    loss = float(smoothed_ce(logits, y, epsilon))
    print(f'epsilon={epsilon}: loss={loss:.4f}')
```

```{.python .input #training-recipes-label-smoothing}
%%tab pytorch
logits, y = d2l.tensor([[10.0, 0.0, 0.0]]), d2l.tensor([0])
for epsilon in (0.0, 0.1):
    loss = F.cross_entropy(logits, y, label_smoothing=epsilon)
    print(f'epsilon={epsilon}: loss={float(loss):.4f}')
```

```{.python .input #training-recipes-label-smoothing}
%%tab tensorflow
logits, Y = tf.constant([[10.0, 0.0, 0.0]]), tf.one_hot([0], 3)
for epsilon in (0.0, 0.1):
    loss = tf.keras.losses.categorical_crossentropy(
        Y, logits, from_logits=True, label_smoothing=epsilon)
    print(f'epsilon={epsilon}: loss={float(loss[0]):.4f}')
```

```{.python .input #training-recipes-label-smoothing}
%%tab jax
logits, Y = jnp.array([[10.0, 0.0, 0.0]]), jax.nn.one_hot(jnp.array([0]), 3)
for epsilon in (0.0, 0.1):
    loss = optax.softmax_cross_entropy(logits,
                                       optax.smooth_labels(Y, epsilon))
    print(f'epsilon={epsilon}: loss={float(loss[0]):.4f}')
```

With $\epsilon = 0.1$, the loss jumps from 0.0001 to 0.67. For three classes,
the minimum possible smoothed loss is roughly 0.29, attained when the winning
logit leads the others by about 3.3. A margin of 10 is therefore penalized as
overconfident, reversing the gradient so it pulls extreme logits back toward
a moderate gap rather than pushing them further apart.

### Cosine Schedules with Warmup

Traditional step schedules hold the learning rate constant for dozens of
epochs and then drop it abruptly by a factor of 10—introducing the drop epochs
as extra hyperparameters to tune. The modern default
:cite:`Loshchilov.Hutter.2016` is already familiar from AlexNet onward
(:numref:`sec_alexnet`): a linear *warmup* from 0 to the peak rate $\eta_0$
over $t_\textrm{w}$ steps, followed by a smooth half-cosine *decay*
:eqref:`eq_cosine_warmup` down to 0 at step $T$. Beyond preventing early
divergence, warmup gives running batch-normalization statistics and adaptive
optimizer moments time to settle before full-size steps begin, while cosine
decay removes abrupt drop thresholds altogether. Below we plot `step_decay`
(30 epochs, with a two-epoch warmup) alongside `d2l.cosine_warmup` (45 epochs,
three-epoch warmup) at 79 minibatches per epoch—the exact horizons used in
our experiment later in this section.

```{.python .input #training-recipes-cosine-schedules-with-warmup}
def step_decay(t, num_steps, warmup_steps):
    """Recipe A: linear warmup, then divide by 10 at 60% and 85% of steps."""
    warmup = min((t + 1) / warmup_steps, 1)
    return warmup * 0.1 ** sum(t >= int(f * num_steps) for f in (0.6, 0.85))

steps_per_epoch = 79
steps_a = list(range(30 * steps_per_epoch))  # Recipe A: 30 epochs
steps_b = list(range(45 * steps_per_epoch))  # Recipe B: 45 epochs
d2l.plot([[t / steps_per_epoch for t in steps_a],
          [t / steps_per_epoch for t in steps_b]],
         [[step_decay(t, len(steps_a), 2 * steps_per_epoch) for t in steps_a],
          [float(d2l.cosine_warmup(t, len(steps_b), 3 * steps_per_epoch))
           for t in steps_b]],
         xlabel='epoch', ylabel='learning rate / base rate',
         legend=['step decay (30 epochs)', 'cosine decay (45 epochs)'])
```

### Mixup

*Mixup* :cite:`zhang2018mixup` regularizes a classifier by training on convex
combinations of pairs of training examples. For each minibatch, it samples a
mixing coefficient $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$ and pairs
each example $i$ with a randomly shuffled partner $j$, constructing

$$
\begin{aligned}
\tilde{\mathbf{x}} &= \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j, \\
\tilde{\mathbf{y}} &= \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j,
\end{aligned}
$$
:eqlabel:`eq_mixup`

where $\mathbf{y}_i$ and $\mathbf{y}_j$ are one-hot class vectors so
$\tilde{\mathbf{y}}$ is a soft target distribution. Because cross-entropy is
linear in its target distribution, the loss on $\tilde{\mathbf{y}}$ equals
the $\lambda$-weighted sum of the cross-entropies against $y_i$ and $y_j$,
which is simpler and faster to compute. Encouraging predictions to interpolate
linearly between training pairs discourages sharp memorization around
individual points. With typical small values ($\alpha \in [0.1, 0.2]$), the
Beta distribution concentrates near 0 and 1, leaving most minibatches lightly
perturbed while strongly blending a few. Below we implement `mixup` using one
shared $\lambda$ per minibatch and save it to `d2l` for reuse in
:numref:`sec_convnext`.

```{.python .input #training-recipes-mixup-1}
%%tab mxnet
def mixup(X, y, alpha):  #@save
    """Mix a minibatch with a shuffled copy; return both label sets."""
    lam = random.betavariate(alpha, alpha)
    perm = d2l.tensor(random.sample(range(X.shape[0]), X.shape[0]),
                      dtype='int32', device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam
```

```{.python .input #training-recipes-mixup-1}
%%tab pytorch
def mixup(X, y, alpha):  #@save
    """Mix a minibatch with a shuffled copy; return both label sets."""
    lam = random.betavariate(alpha, alpha)
    perm = torch.randperm(X.shape[0], device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam
```

```{.python .input #training-recipes-mixup-1}
%%tab tensorflow
def mixup(X, y, alpha):  #@save
    """Mix a minibatch with a shuffled copy; return both label sets."""
    g1, g2 = tf.random.gamma([], alpha), tf.random.gamma([], alpha)
    lam = g1 / (g1 + g2)  # Beta(alpha, alpha) as a ratio of Gammas
    perm = tf.random.shuffle(tf.range(tf.shape(X)[0]))
    return (lam * X + (1 - lam) * tf.gather(X, perm),
            y, tf.gather(y, perm), lam)
```

```{.python .input #training-recipes-mixup-1}
%%tab jax
def mixup(key, X, y, alpha):  #@save
    """Mix a minibatch with a shuffled copy; return both label sets."""
    key_lam, key_perm = jax.random.split(key)
    lam = jax.random.beta(key_lam, alpha, alpha)
    perm = jax.random.permutation(key_perm, X.shape[0])
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam
```

To visualize the effect clearly, we mix a minibatch of Fashion-MNIST garments
with $\alpha = 2$, which concentrates $\lambda$ near 0.5 (training uses a much
smaller $\alpha$). The top row shows the original images and the bottom row
shows their superposed mixtures.

```{.python .input #training-recipes-mixup-2}
%%tab mxnet
data = d2l.FashionMNIST(batch_size=8)
X, y = next(iter(data.train_dataloader()))
X_mix, y_a, y_b, lam = mixup(X, y, alpha=2.0)
d2l.show_images(mnp.concatenate([X, X_mix]).squeeze(axis=1), 2, 8)
print(f'lambda = {lam:.2f}')
```

```{.python .input #training-recipes-mixup-2}
%%tab pytorch
data = d2l.FashionMNIST(batch_size=8)
X, y = next(iter(data.train_dataloader()))
X_mix, y_a, y_b, lam = mixup(X, y, alpha=2.0)
d2l.show_images(torch.cat([X, X_mix]).squeeze(1), 2, 8)
print(f'lambda = {lam:.2f}')
```

```{.python .input #training-recipes-mixup-2}
%%tab tensorflow
data = d2l.FashionMNIST(batch_size=8)
X, y = next(iter(data.train_dataloader()))
X_mix, y_a, y_b, lam = mixup(X, y, alpha=2.0)
d2l.show_images(tf.squeeze(tf.concat([X, X_mix], axis=0), -1), 2, 8)
print(f'lambda = {float(lam):.2f}')
```

```{.python .input #training-recipes-mixup-2}
%%tab jax
data = d2l.FashionMNIST(batch_size=8)
X, y = map(jnp.asarray, next(iter(data.train_dataloader())))
X_mix, y_a, y_b, lam = mixup(d2l.get_key(), X, y, alpha=2.0)
d2l.show_images(jnp.concatenate([X, X_mix]).squeeze(-1), 2, 8)
print(f'lambda = {float(lam):.2f}')
```

Because Mixup creates ghostly double exposures that never occur in real
photographs, *CutMix* :cite:`yun2019cutmix` blends spatially instead: it cuts
a rectangular patch from one image, pastes it onto another, and mixes the
labels in proportion to the patch's area, keeping every pixel locally
realistic. Many modern vision recipes randomly alternate between Mixup and
CutMix from minibatch to minibatch (see the exercises below).

### Stochastic Depth

While dropout (:numref:`sec_dropout`) randomly zeroes out individual
activations during training, *stochastic depth*
:cite:`huang2016stochasticdepth` drops entire residual branches at random:
with probability $p$, a block's residual branch is bypassed so the block
passes its input straight through, effectively training an ensemble of
shallower subnetworks. Residual connections make this safe because the
identity shortcut always keeps the signal flowing. For a block computing
$\mathbf{x} + f(\mathbf{x})$, stochastic depth replaces the branch with

$$
\mathbf{x} + \frac{b}{1 - p} f(\mathbf{x}), \qquad b \sim \mathrm{Bernoulli}(1-p).
$$
:eqlabel:`eq_stochastic_depth`

Just as in inverted dropout, scaling by $1/(1-p)$ keeps the expected branch
output unchanged during training so no rescaling is needed at evaluation time.
Modern implementations sample $b$ independently for every example in the
minibatch (`drop_path` below, saved to `d2l` for ConvNeXt). Subclassing
`d2l.Residual` (:numref:`sec_resnet`), we drop the residual branch right
before adding the shortcut; passing 1,000 random inputs through the block in
training mode with $p = 0.5$, roughly half of all outputs should match
$\mathrm{ReLU}(\mathbf{x})$ identically.

:begin_tab:`mxnet`
MXNet keeps the training mode in a global flag, and `npx.dropout` reads it
when the network runs, also inside a hybridized network. The MXNet
`drop_path` is dropout with one draw per example and therefore takes no
`training` argument.
:end_tab:

```{.python .input #training-recipes-stochastic-depth}
%%tab mxnet
def drop_path(Y, p):  #@save
    """Zero each example of Y with probability p, rescale the rest."""
    if p == 0:
        return Y
    # Dropout with one draw per example (shared along all other axes); it
    # reads the training mode at run time, also when hybridized
    return npx.dropout(Y, p=p, axes=tuple(range(1, Y.ndim)))

class StochasticResidual(d2l.Residual):
    """A residual block whose branch is dropped with probability p."""
    def __init__(self, num_channels, p):
        super().__init__(num_channels)
        self.p = p

    def forward(self, X):
        Y = self.bn2(self.conv2(npx.relu(self.bn1(self.conv1(X)))))
        return npx.relu(X + drop_path(Y, self.p))

blk, X = StochasticResidual(3, p=0.5), mnp.random.normal(size=(1000, 3, 8, 8))
blk.initialize()
with autograd.train_mode():
    Y = blk(X)
dropped = (Y == npx.relu(X)).reshape(1000, -1).all(axis=1).mean()
print(f'fraction of examples with a dropped branch: {float(dropped):.3f}')
```

```{.python .input #training-recipes-stochastic-depth}
%%tab pytorch
def drop_path(Y, p, training):  #@save
    """Zero each example of Y with probability p, rescale the rest."""
    if not training or p == 0:
        return Y
    shape = (Y.shape[0],) + (1,) * (Y.ndim - 1)  # One draw per example
    keep = torch.rand(shape, device=Y.device) > p
    return Y * keep / (1 - p)

class StochasticResidual(d2l.Residual):
    """A residual block whose branch is dropped with probability p."""
    def __init__(self, num_channels, p):
        super().__init__(num_channels)
        self.p = p

    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        return F.relu(X + drop_path(Y, self.p, self.training))

blk, X = StochasticResidual(3, p=0.5), torch.randn(1000, 3, 8, 8)
Y = blk(X)  # A module is in training mode until .eval() is called
dropped = (Y == F.relu(X)).flatten(1).all(1).float().mean()
print(f'fraction of examples with a dropped branch: {float(dropped):.3f}')
```

```{.python .input #training-recipes-stochastic-depth}
%%tab tensorflow
def drop_path(Y, p, training):  #@save
    """Zero each example of Y with probability p, rescale the rest."""
    if not training or p == 0:
        return Y
    shape = [tf.shape(Y)[0]] + [1] * (len(Y.shape) - 1)
    keep = tf.random.uniform(shape) > p
    return Y * tf.cast(keep, Y.dtype) / (1 - p)

class StochasticResidual(d2l.Residual):
    """A residual block whose branch is dropped with probability p."""
    def __init__(self, num_channels, p):
        super().__init__(num_channels)
        self.p = p

    def call(self, X, training=False):
        Y = tf.nn.relu(self.bn1(self.conv1(X), training=training))
        Y = self.bn2(self.conv2(Y), training=training)
        return tf.nn.relu(X + drop_path(Y, self.p, training))

blk, X = StochasticResidual(3, p=0.5), tf.random.normal((1000, 8, 8, 3))
Y = blk(X, training=True)
same = tf.reduce_all(tf.reshape(Y == tf.nn.relu(X), (1000, -1)), axis=1)
dropped = tf.reduce_mean(tf.cast(same, tf.float32))
print(f'fraction of examples with a dropped branch: {float(dropped):.3f}')
```

```{.python .input #training-recipes-stochastic-depth}
%%tab jax
def drop_path(key, Y, p, training):  #@save
    """Zero each example of Y with probability p, rescale the rest."""
    if not training or p == 0:
        return Y
    shape = (Y.shape[0],) + (1,) * (Y.ndim - 1)  # One draw per example
    keep = jax.random.bernoulli(key, 1 - p, shape)
    return Y * keep / (1 - p)

class StochasticResidual(d2l.Residual):
    """A residual block whose branch is dropped with probability p."""
    deterministic: bool

    def __init__(self, num_channels, p, rngs=None):
        rngs = (nnx.Rngs(params=d2l.get_key(), drop_path=d2l.get_key())
                if rngs is None else rngs)
        super().__init__(num_channels, rngs=rngs)
        self.p, self.deterministic, self.rngs = p, False, rngs

    def set_view(self, *, deterministic):  # The Trainer switches this
        self.deterministic = deterministic

    def __call__(self, X):
        Y = self.bn2(self.conv2(nnx.relu(self.bn1(self.conv1(X)))))
        Y = drop_path(self.rngs.drop_path(), Y, self.p,
                      not self.deterministic)
        return nnx.relu(X + Y)

blk = StochasticResidual(3, p=0.5)
X = jax.random.normal(d2l.get_key(), (1000, 8, 8, 3))
Y = blk(X)
dropped = jnp.all(Y == nnx.relu(X), axis=(1, 2, 3)).mean()
print(f'fraction of examples with a dropped branch: {float(dropped):.3f}')
```

In deep architectures, the drop probability $p$ typically ramps linearly from
0 in the earliest block to a peak of 0.1--0.5 in the deepest block. Notice
that per-example masking still evaluates every branch before zeroing out
dropped examples, so it regularizes without speeding up training; the original
per-minibatch formulation draws a single $b$ per block and can skip dropped
branches completely to save compute.

### Averaging Weights

The final row of :numref:`tab_recipe_2015_2022` delivers cleaner generalization
at virtually zero extra cost. Near the end of training, minibatch noise causes
parameters to bounce around the floor of a low-loss basin, so averaging recent
iterates lands closer to the basin's center than any single checkpoint.
Whereas *stochastic weight averaging* (SWA)
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018` averages snapshots uniformly late
in training, modern vision recipes commonly maintain an *exponential moving
average* (EMA)---the exact same recursion batch normalization uses for its
running statistics (:numref:`sec_batch_norm`). After every optimizer step, a
*shadow copy* $\bar{\theta}$ of the trainable parameters blends in the latest
weights $\theta$ with decay $\rho$:

$$
\bar{\theta} \leftarrow \rho \bar{\theta} + (1-\rho) \theta.
$$
:eqlabel:`eq_ema`

Parameters from $k$ steps ago carry weight $(1-\rho)\rho^k$, giving an
effective averaging horizon of roughly $1/(1-\rho)$ steps (100 steps for
$\rho = 0.99$). Optimization continues updating $\theta$ as usual, while
evaluation swaps in $\bar{\theta}$. (Because batch-normalization running
statistics are buffers rather than trainable weights, an exercise below
examines how best to pair them with EMA weights.) To see the tradeoff in
$\rho$, let us track a single synthetic weight that climbs from 0 to 1 over
50 steps and then oscillates noisily around 1.

```{.python .input #training-recipes-averaging-weights}
%%tab mxnet
class EMA:
    """A shadow copy of a model's parameters, updated after each step."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: p.data().copy()
                       for k, p in model.collect_params().items()
                       if p.grad_req != 'null'}  # Trainable parameters only

    def update(self, model):
        for k, p in model.collect_params().items():
            if k in self.shadow:
                self.shadow[k] = (self.decay * self.shadow[k]
                                  + (1 - self.decay) * p.data())

net = nn.Dense(1, in_units=1, use_bias=False)
net.initialize(init.Zero())  # Training starts at 0
emas = {decay: EMA(net, decay) for decay in (0.9, 0.99)}
raw, avg = [], {decay: [] for decay in emas}
for t in range(200):  # Iterates rise to 1 over 50 steps, then stay noisy
    net.weight.set_data(min(t / 50, 1) + 0.3 * mnp.random.normal(size=(1, 1)))
    raw.append(float(net.weight.data()))
    for decay, ema in emas.items():
        ema.update(net)
        avg[decay].append(float(ema.shadow['weight']))
d2l.plot(list(range(200)), [raw, *avg.values()], xlabel='step',
         ylabel='weight', legend=['raw iterates', 'EMA, 0.9', 'EMA, 0.99'])
```

```{.python .input #training-recipes-averaging-weights}
%%tab pytorch
class EMA:
    """A shadow copy of a model's parameters, updated after each step."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.named_parameters():
            self.shadow[k].lerp_(p, 1 - self.decay)  # beta*avg + (1-beta)*p

net = nn.Linear(1, 1, bias=False)
nn.init.zeros_(net.weight)  # Training starts at 0
emas = {decay: EMA(net, decay) for decay in (0.9, 0.99)}
raw, avg = [], {decay: [] for decay in emas}
for t in range(200):  # Iterates rise to 1 over 50 steps, then stay noisy
    with torch.no_grad():
        net.weight.copy_(min(t / 50, 1) + 0.3 * torch.randn(1, 1))
    raw.append(float(net.weight.detach()))
    for decay, ema in emas.items():
        ema.update(net)
        avg[decay].append(float(ema.shadow['weight']))
d2l.plot(list(range(200)), [raw, *avg.values()], xlabel='step',
         ylabel='weight', legend=['raw iterates', 'EMA, 0.9', 'EMA, 0.99'])
```

```{.python .input #training-recipes-averaging-weights}
%%tab tensorflow
class EMA:
    """A shadow copy of a model's parameters, updated after each step."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = [tf.Variable(v) for v in model.trainable_variables]

    def update(self, model):
        for s, v in zip(self.shadow, model.trainable_variables):
            s.assign(self.decay * s + (1 - self.decay) * v)

net = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros')
net.build((None, 1))  # Training starts at 0
emas = {decay: EMA(net, decay) for decay in (0.9, 0.99)}
raw, avg = [], {decay: [] for decay in emas}
for t in range(200):  # Iterates rise to 1 over 50 steps, then stay noisy
    net.kernel.assign(min(t / 50, 1) + 0.3 * tf.random.normal((1, 1)))
    raw.append(float(net.kernel[0, 0]))
    for decay, ema in emas.items():
        ema.update(net)
        avg[decay].append(float(ema.shadow[0][0, 0]))
d2l.plot(list(range(200)), [raw, *avg.values()], xlabel='step',
         ylabel='weight', legend=['raw iterates', 'EMA, 0.9', 'EMA, 0.99'])
```

```{.python .input #training-recipes-averaging-weights}
%%tab jax
class EMA:
    """A shadow copy of a model's parameters, updated after each step."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        # A copy: nnx.state returns references to the model's arrays
        self.shadow = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    def update(self, model):  # beta * avg + (1 - beta) * current
        self.shadow = optax.incremental_update(
            nnx.state(model, nnx.Param), self.shadow, 1 - self.decay)

net = nnx.Linear(1, 1, use_bias=False, kernel_init=nnx.initializers.zeros,
                 rngs=nnx.Rngs(d2l.get_key()))  # Training starts at 0
emas = {decay: EMA(net, decay) for decay in (0.9, 0.99)}
raw, avg = [], {decay: [] for decay in emas}
for t in range(200):  # Iterates rise to 1 over 50 steps, then stay noisy
    net.kernel[...] = (min(t / 50, 1)
                       + 0.3 * jax.random.normal(d2l.get_key(), (1, 1)))
    raw.append(float(net.kernel[0, 0]))
    for decay, ema in emas.items():
        ema.update(net)
        avg[decay].append(float(ema.shadow['kernel'][...][0, 0]))
d2l.plot(list(range(200)), [raw, *avg.values()], xlabel='step',
         ylabel='weight', legend=['raw iterates', 'EMA, 0.9', 'EMA, 0.99'])
```

At $\rho = 0.9$ (horizon ~10 steps), the shadow weight trails the ramp by
about 10 steps and then dramatically damps out the noise. At $\rho = 0.99$
(horizon ~100 steps), it is smoother still, yet lags so far behind that it
still sits roughly 20% below 1 after 200 steps. That illustrates the main
pitfall of EMA: whenever $1/(1-\rho)$ is large compared with the run length,
the shadow copy lags behind improving weights and underperforms the final
checkpoint. Very large decays like 0.999 or 0.9999 therefore belong only in
long training runs.

## Two Recipes on One Network

Now let us assemble these pieces and run a head-to-head test on a single
fixed architecture: the ResNet-18 of :numref:`sec_resnet`, rebuilt from
`d2l.Residual` with He initialization (:numref:`sec_numerical_stability`). We
train on a random subset of 10,000 Fashion-MNIST images resized to
$96 \times 96$ (shared by both recipes within each run): using fewer training
examples amplifies the role of regularization while keeping repeated
experiments fast. Both recipes compute their step schedules directly from
`lr` and `num_steps`.

```{.python .input #training-recipes-one-network-two-recipes-1}
%%tab mxnet
class FashionMNIST10k(d2l.FashionMNIST):
    """Fashion-MNIST with a random subset of num_train training images."""
    def __init__(self, batch_size=128, resize=(96, 96), num_train=10000):
        super().__init__(batch_size, resize)
        full = self.train
        idx = random.sample(range(len(full)), num_train)
        self.train = gluon.data.SimpleDataset(idx).transform(lambda i: full[i])

class ResNet18(d2l.Classifier):
    """The ResNet-18 of the residual-networks section, from d2l.Residual."""
    def __init__(self, lr, num_steps, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.HybridSequential()
        self.net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                               use_bias=False),
                     nn.BatchNorm(), nn.Activation('relu'),
                     nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        for i, c in enumerate((64, 128, 256, 512)):
            for j in range(2):
                down = i > 0 and j == 0
                self.net.add(d2l.Residual(c, use_1x1conv=down,
                                          strides=2 if down else 1))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.MSRAPrelu('in', 0))  # Variance 2/n_in
        self.net.hybridize()
```

```{.python .input #training-recipes-one-network-two-recipes-1}
%%tab pytorch
class FashionMNIST10k(d2l.FashionMNIST):
    """Fashion-MNIST with a random subset of num_train training images."""
    def __init__(self, batch_size=128, resize=(96, 96), num_train=10000):
        super().__init__(batch_size, resize)
        idx = torch.randperm(len(self.train))[:num_train]
        self.train = torch.utils.data.Subset(self.train, idx)

class ResNet18(d2l.Classifier):
    """The ResNet-18 of the residual-networks section, from d2l.Residual."""
    def __init__(self, lr, num_steps, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        layers = [nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3,
                                bias=False),
                  nn.LazyBatchNorm2d(), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        for i, c in enumerate((64, 128, 256, 512)):
            for j in range(2):
                down = i > 0 and j == 0
                layers.append(d2l.Residual(c, use_1x1conv=down,
                                           strides=2 if down else 1))
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                   nn.LazyLinear(num_classes)]
        self.net = nn.Sequential(*layers)

    def weight_init(self, module):  # The Trainer applies it after a dry run
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)  # Variance 2/n_in
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

```{.python .input #training-recipes-one-network-two-recipes-1}
%%tab tensorflow
class FashionMNIST10k(d2l.FashionMNIST):
    """Fashion-MNIST with a random subset of num_train training images."""
    def __init__(self, batch_size=128, resize=(96, 96), num_train=10000):
        super().__init__(batch_size, resize)
        idx = tf.random.shuffle(tf.range(len(self.train[1])))[:num_train]
        self.train = tuple(tf.gather(a, idx).numpy() for a in self.train)

class ResNet18(d2l.Classifier):
    """The ResNet-18 of the residual-networks section, from d2l.Residual."""
    def __init__(self, lr, num_steps, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        he_init = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')  # One instance per layer
        self.net = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(3),  # Pad every side
            tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False,
                                   kernel_initializer=he_init()),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(1),  # After the ReLU, 0 = -inf
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2)])
        for i, c in enumerate((64, 128, 256, 512)):
            for j in range(2):
                down = i > 0 and j == 0
                self.net.add(d2l.Residual(c, use_1x1conv=down,
                                          strides=2 if down else 1))
        self.net.add(tf.keras.layers.GlobalAvgPool2D())
        self.net.add(tf.keras.layers.Dense(num_classes,
                                           kernel_initializer=he_init()))
```

```{.python .input #training-recipes-one-network-two-recipes-1}
%%tab jax
class FashionMNIST10k(d2l.FashionMNIST):
    """Fashion-MNIST with a random subset of num_train training images."""
    def __init__(self, batch_size=128, resize=(96, 96), num_train=10000):
        super().__init__(batch_size, resize)
        idx = jax.random.permutation(d2l.get_key(), len(self.train[1]))
        self.train = tuple(a[idx[:num_train]] for a in self.train)

class ResNet18(d2l.Classifier):
    """The ResNet-18 of the residual-networks section, from d2l.Residual."""
    def __init__(self, lr, num_steps, num_classes=10, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        self.rngs = (nnx.Rngs(params=d2l.get_key(), mixup=d2l.get_key())
                     if rngs is None else rngs)
        he_init = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        layers = [nnx.Conv(1, 64, (7, 7), strides=2, padding=3,
                           use_bias=False, kernel_init=he_init,
                           rngs=self.rngs),
                  nnx.BatchNorm(64, momentum=0.9, use_fast_variance=False,
                                rngs=self.rngs),
                  nnx.relu, max_pool]
        in_channels = 64
        for i, c in enumerate((64, 128, 256, 512)):
            for j in range(2):
                down = i > 0 and j == 0
                layers.append(d2l.Residual(
                    c, use_1x1conv=down, strides=(2, 2) if down else (1, 1),
                    in_channels=in_channels, rngs=self.rngs))
                in_channels = c
        layers += [global_avg_pool,
                   nnx.Linear(512, num_classes, kernel_init=he_init,
                              rngs=self.rngs)]
        self.net = nnx.Sequential(*layers)

def max_pool(X):  # 3x3 windows, stride 2, padding 1
    return nnx.max_pool(X, (3, 3), (2, 2), padding=((1, 1), (1, 1)))

def global_avg_pool(X):  # Average over height and width (NHWC)
    return X.mean(axis=(1, 2))
```

Both models share the exact same architecture and training split while
changing the optimization and regularization recipe completely. *Recipe A*
mirrors 2015 practice: SGD with momentum 0.9, coupled weight decay
$5 \times 10^{-4}$, a peak learning rate of 0.05 with two epochs of linear
warmup followed by 10-fold drops at 60% and 85% of training, plain
cross-entropy loss, and 30 epochs. (The short two-epoch warmup is a minor
modern stabilization: without it, batch-normalized SGD at 0.05 diverges in
epoch 1 on roughly a third of runs :cite:`Goyal.Dollar.Girshick.ea.2017`.)
Our *compact modern recipe* uses AdamW with weight decay 0.05, peak learning
rate 0.002 with three epochs of warmup and cosine decay, label smoothing
($\epsilon = 0.1$), Mixup ($\alpha = 0.2$), and 45 epochs (reflecting the
longer horizon modern regularization requires). To keep runtime manageable, it
omits RandAugment, CutMix, stochastic depth, and EMA, testing a lean subset of
the full menu.

Do not be misled by the 100-fold gap between the two weight-decay numbers
(0.05 vs. $5 \times 10^{-4}$). The actual fractional shrinkage per step is
learning rate times weight decay: at peak rate, that equals
$0.002 \times 0.05 = 10^{-4}$ for AdamW versus
$0.05 \times 5 \times 10^{-4} = 2.5 \times 10^{-5}$ for SGD—which momentum
amplifies up to tenfold, making SGD's effective decay actually stronger.
Finally, both recipes evaluate validation loss using plain cross-entropy on
unmixed validation images; label smoothing and Mixup modify only the training
objective.

```{.python .input #training-recipes-one-network-two-recipes-2}
%%tab mxnet
class Schedule(mx.lr_scheduler.LRScheduler):
    """The rate base_lr * factor(t), with t counted from 0."""
    def __init__(self, factor, base_lr=0.01):
        super().__init__(base_lr)
        self.factor = factor

    def __call__(self, num_update):  # MXNet counts updates from 1
        return self.base_lr * self.factor(num_update - 1)

class ClassicRecipe(ResNet18):
    """Recipe A: SGD with momentum, weight decay, warmup, step decay."""
    def configure_optimizers(self):
        warmup = 2 * self.num_steps // 30  # Two of the 30 epochs
        schedule = Schedule(
            lambda t: step_decay(t, self.num_steps, warmup), self.lr)
        return gluon.Trainer(self.collect_params(), 'sgd', {
            'learning_rate': self.lr, 'momentum': 0.9, 'wd': 5e-4,
            'lr_scheduler': schedule})

class ModernRecipe(ResNet18):
    """Compact modern recipe: AdamW, warmup and cosine, smoothing, Mixup."""
    def configure_optimizers(self):
        warmup = 3 * self.num_steps // 45  # Three of the 45 epochs
        schedule = Schedule(
            lambda t: d2l.cosine_warmup(t, self.num_steps, warmup), self.lr)
        # MXNet scales the decay by the bias-corrected rate, which is smaller
        # early on: over 3,555 steps the weights shrink about a quarter less
        return gluon.Trainer(self.collect_params(), 'adamw', {
            'learning_rate': self.lr, 'wd': 0.05, 'epsilon': 1e-8,
            'lr_scheduler': schedule})

    def training_step(self, batch):
        X, y_a, y_b, lam = mixup(*batch, alpha=0.2)
        Y_hat = self(X)
        return (lam * smoothed_ce(Y_hat, y_a, 0.1)
                + (1 - lam) * smoothed_ce(Y_hat, y_b, 0.1)).mean()
```

```{.python .input #training-recipes-one-network-two-recipes-2}
%%tab pytorch
class ClassicRecipe(ResNet18):
    """Recipe A: SGD with momentum, weight decay, warmup, step decay."""
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9,
                                weight_decay=5e-4)
        warmup = 2 * self.num_steps // 30  # Two of the 30 epochs
        return optim, torch.optim.lr_scheduler.LambdaLR(
            optim, lambda t: step_decay(t, self.num_steps, warmup))

class ModernRecipe(ResNet18):
    """Compact modern recipe: AdamW, warmup and cosine, smoothing, Mixup."""
    def configure_optimizers(self):
        warmup = 3 * self.num_steps // 45  # Three of the 45 epochs
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                  weight_decay=0.05)
        return optim, torch.optim.lr_scheduler.LambdaLR(
            optim, lambda t: d2l.cosine_warmup(t, self.num_steps, warmup))

    def training_step(self, batch):
        X, y_a, y_b, lam = mixup(*batch, alpha=0.2)
        Y_hat = self(X)
        smooth = lambda y: F.cross_entropy(Y_hat, y, label_smoothing=0.1)
        return lam * smooth(y_a) + (1 - lam) * smooth(y_b)
```

```{.python .input #training-recipes-one-network-two-recipes-2}
%%tab tensorflow
class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """The rate base_lr * factor(step)."""
    def __init__(self, factor, base_lr):
        self.factor, self.base_lr = factor, base_lr

    def __call__(self, step):
        return self.base_lr * self.factor(step)

class CoupledSGD(tf.keras.optimizers.SGD):
    """SGD that adds weight decay to the gradient (L2 regularization)."""
    # The weight_decay of a Keras optimizer would shrink the weights directly
    def __init__(self, weight_decay, **kwargs):
        super().__init__(**kwargs)
        self.l2 = weight_decay

    def apply_gradients(self, grads_and_vars):
        return super().apply_gradients(
            [(g + self.l2 * v, v) for g, v in grads_and_vars])

class ClassicRecipe(ResNet18):
    """Recipe A: SGD with momentum, weight decay, warmup, step decay."""
    def configure_optimizers(self):
        # PiecewiseConstantDecay switches after a boundary, step_decay at it
        drops = [int(f * self.num_steps) - 1 for f in (0.6, 0.85)]
        decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            drops, [1.0, 0.1, 0.01])
        warmup = 2 * self.num_steps // 30  # Two of the 30 epochs
        schedule = Schedule(lambda t: decay(t) * tf.minimum(
            tf.cast(t + 1, tf.float32) / warmup, 1.0), self.lr)
        return CoupledSGD(5e-4, learning_rate=schedule, momentum=0.9)

class ModernRecipe(ResNet18):
    """Compact modern recipe: AdamW, warmup and cosine, smoothing, Mixup."""
    def configure_optimizers(self):
        warmup = 3 * self.num_steps // 45  # Three of the 45 epochs
        schedule = Schedule(
            lambda t: d2l.cosine_warmup(t, self.num_steps, warmup), self.lr)
        return tf.keras.optimizers.AdamW(schedule, weight_decay=0.05,
                                         epsilon=1e-8)

    def training_step(self, batch):
        X, y_a, y_b, lam = mixup(*batch, alpha=0.2)
        Y_hat = self(X, training=True)
        smooth = lambda y: tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.one_hot(y, self.num_classes), Y_hat, from_logits=True,
                label_smoothing=0.1))
        return lam * smooth(y_a) + (1 - lam) * smooth(y_b)
```

```{.python .input #training-recipes-one-network-two-recipes-2}
%%tab jax
class ClassicRecipe(ResNet18):
    """Recipe A: SGD with momentum, weight decay, warmup, step decay."""
    def configure_optimizers(self):
        decay = optax.piecewise_constant_schedule(self.lr, {
            int(0.6 * self.num_steps): 0.1, int(0.85 * self.num_steps): 0.1})
        warmup = 2 * self.num_steps // 30  # Two of the 30 epochs
        schedule = lambda t: decay(t) * jnp.minimum((t + 1) / warmup, 1.0)
        return optax.chain(optax.add_decayed_weights(5e-4),  # In the gradient
                           optax.sgd(schedule, momentum=0.9))

class ModernRecipe(ResNet18):
    """Compact modern recipe: AdamW, warmup and cosine, smoothing, Mixup."""
    def configure_optimizers(self):
        warmup = 3 * self.num_steps // 45  # Three of the 45 epochs
        schedule = lambda t: self.lr * d2l.cosine_warmup(
            t, self.num_steps, warmup)
        return optax.adamw(schedule, weight_decay=0.05)

    def training_step(self, batch):
        X, y_a, y_b, lam = mixup(self.rngs.mixup(), *batch, alpha=0.2)
        Y_hat = self(X)
        targets = lambda y: optax.smooth_labels(
            jax.nn.one_hot(y, self.num_classes), 0.1)
        smooth = lambda y: optax.softmax_cross_entropy(
            Y_hat, targets(y)).mean()
        return lam * smooth(y_a) + (1 - lam) * smooth(y_b)
```

First we train Recipe A for 30 epochs and evaluate on the full 10,000-image
validation split.

```{.python .input #training-recipes-one-network-two-recipes-4}
data = FashionMNIST10k()
steps_per_epoch = len(data.train_dataloader())  # 79 minibatches of <= 128
model = ClassicRecipe(lr=0.05, num_steps=30 * steps_per_epoch)
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
trainer.fit(model, data)
trainer.history['val_acc'][-1]
```

Next we train the exact same architecture on the exact same training subset
for 45 epochs using the compact modern recipe.

```{.python .input #training-recipes-one-network-two-recipes-5}
model = ModernRecipe(lr=2e-3, num_steps=45 * steps_per_epoch)
trainer = d2l.Trainer(max_epochs=45, num_gpus=1)
trainer.fit(model, data)
trainer.history['val_acc'][-1]
```

:numref:`tab_recipe_results` reports the validation accuracy range across
repeated runs (each drawing a fresh 10,000-image subset, initialization, and
minibatch order).

:The same ResNet-18 under both recipes (10,000 Fashion-MNIST training images at
$96 \times 96$; validation accuracy, range over repeated runs).
:label:`tab_recipe_results`

| Recipe | Epochs | Validation accuracy |
|:--|--:|--:|
| A (2015): SGD, warmup, step decay | 30 | 89–90.5% |
| Compact modern: AdamW, cosine, smoothing, Mixup | 45 | 90.5–91% |

In every paired run on the same 10,000-image subset, `ModernRecipe` beats `ClassicRecipe` by 0.5--1.5 percentage points (even while omitting RandAugment, CutMix, stochastic depth, and EMA). Three practical lessons accompany this comparison:

1. **Bundled comparisons vs. ablations:** Because optimizer, schedule, loss, augmentation, and epoch budget changed together, only a one-at-a-time ablation (Exercise :exref:`ex_training-recipes-ablation`) can isolate which ingredient contributed most—and because optimal learning rates differ $25\times$ between AdamW ($0.002$) and SGD ($0.05$), any optimizer swap must retune the learning rate.
2. **Never compare raw training losses across recipes:** Under `ClassicRecipe`, training loss plunges toward zero while validation loss rebounds to 0.4--0.5 as logits grow overconfident; under `ModernRecipe`, validation loss drops below 0.4 while training loss stays *higher* than validation loss because Mixup and label smoothing evaluate a soft, blended training objective.
3. **Reading published leaderboards:** Because training recipes alone move a fixed ResNet-50 by $>4$ ImageNet points :cite:`wightman2021resnet`, a 1-point claim over a baseline copied from an older paper confounds recipe with architecture. Even under matched recipes, test-set noise and multi-object ambiguity on ImageNet :cite:`Recht.Roelofs.Schmidt.ea.2019,beyer2020imagenetreal` mean that transfer to detection or segmentation is often a sharper test of backbone quality than sub-point ImageNet top-1 deltas.

## Summary

Between 2015 and the early 2020s, convnet training recipes evolved across every axis—AdamW with decoupled weight decay, cosine warmup schedules, RandAugment/Mixup/CutMix, label smoothing, stochastic depth, longer training horizons, and EMA weight averaging—lifting an unchanged ResNet-50 from 76.1% to 80.4% on ImageNet. Each ingredient takes only a few lines of code, and a compact subset reliably improves ResNet-18 on `FashionMNIST10k`. Next, :numref:`sec_convnext` holds the modern recipe fixed and modernizes the ResNet architecture itself one step at a time.

## Exercises

1. **Binary cross-entropy and Mixup.** :citet:`wightman2021resnet` train ImageNet classifiers with binary cross-entropy (treating each of the $K$ classes as an independent sigmoid binary classification task) rather than $K$-way softmax cross-entropy. Explain why independent sigmoid outputs pair naturally with Mixup and CutMix when a blended image contains objects from two distinct classes at once.

1. [code] **CutMix.** CutMix :cite:`yun2019cutmix` samples $\lambda \sim \mathrm{Beta}(1, 1) = \mathrm{Uniform}(0, 1)$ and pastes a random rectangular patch from a shuffled partner image covering a target area fraction $1 - \lambda$. Implement `cutmix` as a drop-in variant of `mixup`, mixing the one-hot labels using the *exact* fraction of pixels kept after clipping the rectangle to the image borders. Train `ModernRecipe` on `FashionMNIST10k` once with CutMix in place of Mixup and once alternating randomly between Mixup and CutMix each minibatch, and compare validation accuracy against :numref:`tab_recipe_results`.

1. [code] **Weight averaging horizon (EMA).** Integrate the `EMA` class into `ModernRecipe` by updating the shadow weights after every optimizer step and evaluating validation accuracy with the shadow parameters. Sweep the decay $\rho \in \{0.9, 0.99, 0.999\}$ in :eqref:`eq_ema`. Relate the best-performing $\rho$ to the effective horizon $1/(1-\rho)$ and the total number of steps ($45 \times 79 = 3{,}555$), and explain how batch-normalization running statistics should be handled when evaluating EMA weights.

1. [code] **One-at-a-time recipe ablation.** ● In :numref:`tab_recipe_results`, `ModernRecipe` beats `ClassicRecipe` by 0.5--1.5 percentage points when all five ingredients change at once.
    1. Run both `ClassicRecipe` and `ModernRecipe` five times on `FashionMNIST10k` to estimate the run-to-run standard deviation of validation accuracy.
    1. Starting from `ModernRecipe`, revert one ingredient at a time to its `ClassicRecipe` counterpart: (i) remove Mixup, (ii) remove label smoothing, (iii) replace cosine warmup with `step_decay`, (iv) replace AdamW with momentum SGD (retuning the peak learning rate to $0.05$), and (v) shorten training from 45 to 30 epochs. Which individual changes exceed the run-to-run noise floor?
    1. Complete the $2 \times 2$ table of optimizer (SGD vs. AdamW) $\times$ schedule (`step_decay` vs. `cosine_warmup`) with Mixup and label smoothing enabled. Do the optimizer and schedule interact?
   :exlabel:`ex_training-recipes-ablation`

1. [code] **Stochastic depth in the recipe.** ● Our compact modern recipe omitted stochastic depth. Add it back to `ResNet18`:
    1. Extend `StochasticResidual` to support stride-2 downsampling blocks with $1 \times 1$ shortcuts, and build `ResNet18` with drop probabilities growing linearly from $0$ in block 1 to $p_{\max}$ in block 8. What is the expected number of active residual branches per forward pass as a function of $p_{\max}$?
    1. Sweep $p_{\max} \in \{0.1, 0.2, 0.5\}$ under `ModernRecipe` on `FashionMNIST10k` (compared against $p_{\max} = 0$). Why does a shallow 8-block network benefit less from large $p_{\max}$ than a 50- or 100-block network?
    1. Implement the per-minibatch variant of :citet:`huang2016stochasticdepth`, which draws a single Bernoulli variable $b$ per block and skips evaluating the residual branch altogether when $b = 0$. At $p_{\max} = 0.5$, compare the theoretical fraction of skipped branches against the measured reduction in wall-clock training time.

[Discussions](https://d2l.discourse.group/t/36761)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §7.6]{.kicker}

Modern Training Recipes for Convnets<br>**AdamW · cosine warmup · label smoothing · Mixup & CutMix · stochastic depth · EMA weight averaging**
:::
:::

::: {.slide title="Why Reported Architecture Gains Are Often Recipe Gains"}
[Motivation & Benchmark Confounding]{.kicker}

::: {.cols .vc}
::: {.col}
- **ResNet-50 Strikes Back (Wightman et al., 2021):** Upgrading the training recipe lifts an *unchanged* 2015 ResNet-50 from **$76.1\%$ to $80.4\%$** (+4.3 points) on ImageNet!
- **Scientific rule:** Always compare architectures under matched training recipes; a 1–2 point gain over an un-retrained baseline confounds recipe with architecture.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-resnet50-strikes-back
:::
:::
:::

::: {.slide title="Evolution of Vision Training Recipes (2015 vs. 2022)"}
[The Modern Recipe Menu]{.kicker}

::: {.cols .vc}
::: {.col}
- **Data & forward pass:** RandAugment + Mixup + CutMix + Stochastic Depth (`drop_path`).
- **Objective & optimizer:** Label smoothing ($\epsilon = 0.1$) + AdamW (decoupled weight decay) + linear warmup & half-cosine decay over 300–600 epochs.
- **Checkpoint:** Shadow exponential moving average (EMA) $\bar{\boldsymbol{\theta}}_T$.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-recipe-pipeline-2015-2022
:::
:::
:::

::: {.slide title="Decoupled Weight Decay in AdamW vs. Coupled SGD"}
[Optimizer Mechanics]{.kicker}

::: {.cols .vc}
::: {.col}
$$\boldsymbol{\theta}_{t+1} = (1 - \eta_t \lambda)\,\boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

- **Why L2 fails in Adam:** Dividing $\lambda \boldsymbol{\theta}_t$ by $\sqrt{\hat{\mathbf{v}}_t}$ shrinks high-gradient weights *less*.
- **Comparing across optimizers:** Compare per-step fractional decay $\eta_0 \lambda$ ($0.002 \times 0.05 = 10^{-4}$ for AdamW).
:::

::: {.col .fig}
@fig:ch07-mcnn-adamw-decoupled-decay
:::
:::
:::

::: {.slide title="Label Smoothing: Bounding Optimal Logit Margins"}
[Loss Regularization]{.kicker}

Under standard one-hot cross-entropy ($-\log p_y$), driving the loss toward zero forces the true class logit $z_y \to +\infty$, producing overconfident, poorly calibrated predictions:

- **Softened target distribution:** *Label smoothing* (Szegedy et al., 2016) places probability $1 - \epsilon$ on the true class $y$ and distributes $\epsilon$ uniformly across all $K$ classes ($\epsilon / K$ each):
  $$l_\epsilon(\mathbf{p}, y) = -(1 - \epsilon)\log p_y - \frac{\epsilon}{K} \sum_{k=1}^K \log p_k$$
- **Finite optimal logit gap:** Minimized when $p_y^* = 1 - \epsilon + \epsilon/K$ and $p_k^* = \epsilon/K$ ($k \neq y$), which occurs at the exact finite logit margin $z_y^* - z_k^* = \log\bigl(1 + \frac{K(1-\epsilon)}{\epsilon}\bigr)$ (e.g., $\approx 3.33$ for $K=3, \epsilon=0.1$).
- **Penalizing overconfidence:** On logits $(10, 0, 0)$, $\epsilon=0.1$ raises loss from $0.0001$ to $0.67$, pulling extreme logits back:

@!training-recipes-label-smoothing
:::

::: {.slide title="Step Decay vs. Cosine Decay with Linear Warmup"}
[Learning-Rate Schedules]{.kicker}

Plotting our 30-epoch step schedule against our 45-epoch cosine warmup schedule illustrates why modern recipes abandoned discrete step drops:

::: {.cols .vc}
::: {.col}
- **Warmup ($0 \le t < t_{\textrm{w}}$):** Linearly ramps $\eta_t = \eta_0 \cdot (t / t_{\textrm{w}})$, letting BatchNorm running stats and AdamW second moments $\hat{\mathbf{v}}_t$ equilibrate before full steps occur.
- **Half-cosine decay ($t_{\textrm{w}} \le t \le T$):**
  $$\eta_t = \frac{\eta_0}{2}\left(1 + \cos\frac{\pi(t - t_{\textrm{w}})}{T - t_{\textrm{w}}}\right)$$
  Eliminates drop-epoch tuning and anneals smoothly to $0$ at step $T$.
:::

::: {.col .fig}
@!training-recipes-cosine-schedules-with-warmup
:::
:::
:::

::: {.slide title="Mixup and CutMix: Blending Examples and Labels"}
[Data Augmentation]{.kicker}

::: {.cols .vc}
::: {.col}
@training-recipes-mixup-1

- **Convex combinations ($\lambda \sim \mathrm{Beta}(\alpha, \alpha)$):** Mixup blends whole images; CutMix pastes a rectangular patch of area $1-\lambda$ and mixes labels by area.
:::

::: {.col .fig}
@fig:ch07-mcnn-mixup-vs-cutmix
:::
:::
:::

::: {.slide title="Visualizing Mixup and Contrasting with CutMix"}
[Augmentation Geometry]{.kicker}

During training, $\alpha = 0.2$ concentrates $\lambda$ near $0$ and $1$; below we plot $\alpha = 2.0$ ($\lambda \approx 0.5$) so the superposed garment mixtures are clearly visible:

::: {.cols .vc}
::: {.col}
- **Linear decision boundaries:** Forces network predictions to transition linearly between classes rather than oscillating sharply outside the training manifold.
- **CutMix (Yun et al., 2019):** Because global alpha-blending creates ghostly double exposures, CutMix pastes a rectangular patch of area fraction $1 - \lambda$ from $\mathbf{x}_j$ onto $\mathbf{x}_i$ and mixes labels by the exact patch area, preserving local pixel statistics.
:::

::: {.col .fig}
@!training-recipes-mixup-2
:::
:::
:::

::: {.slide title="Stochastic Depth: Dropping Whole Residual Branches"}
[Structural Regularization]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \frac{b}{1 - p_\ell}\,f(\mathbf{x}_\ell), \qquad b \sim \mathrm{Bernoulli}(1 - p_\ell)$$

- **Implicit ensemble of depths:** Drops entire residual branches per example with linear schedule $p_\ell = (\ell/L)p_{\max}$.
- **Expectation preservation:** Scaling by $1/(1-p_\ell)$ keeps $\mathbb{E}[\mathbf{x}_{\ell+1}] = \mathbf{x}_\ell + f(\mathbf{x}_\ell)$ so $b \equiv 1$ at eval.
:::

::: {.col .fig}
@fig:ch07-mcnn-stochastic-depth-gate
:::
:::
:::

::: {.slide title="Implementing and Verifying `drop_path`"}
[Stochastic Depth Code]{.kicker}

Sampling one Bernoulli mask per example (`shape = (B, 1, 1, 1)`) and passing 1,000 inputs through `StochasticResidual(p=0.5)` confirms that half of the examples bypass the branch and equal $\textrm{ReLU}(\mathbf{X})$ identically:

@training-recipes-stochastic-depth
:::

::: {.slide title="Exponential Moving Average (EMA) of Model Weights"}
[Iterate Averaging]{.kicker}

Near convergence, minibatch noise causes SGD/AdamW iterates $\boldsymbol{\theta}_t$ to bounce around the walls of a flat loss valley. Maintaining a non-trainable shadow copy $\bar{\boldsymbol{\theta}}_t$ averages out this oscillation:

::: {.cols .vc}
::: {.col}
- **EMA update after each optimizer step:**
  $$\bar{\boldsymbol{\theta}}_t \leftarrow \rho\,\bar{\boldsymbol{\theta}}_{t-1} + (1 - \rho)\,\boldsymbol{\theta}_t$$
- **Effective memory horizon $1/(1-\rho)$:** Iterate $\boldsymbol{\theta}_{t-k}$ carries weight $(1-\rho)\rho^k$.
- **Pitfall (Horizon vs. Run Length):** At $\rho = 0.99$ (horizon $100$ steps), the shadow trajectory lags $20\%$ below a 50-step ramp even at step 200. Always choose $1/(1-\rho) \ll T$ (e.g., $\rho = 0.9999$ only for $>50{,}000$-step runs).
:::

::: {.col .fig}
@!training-recipes-averaging-weights
:::
:::
:::

::: {.slide title="Controlled Experiment: One ResNet-18 Under Two Recipes"}
[Head-to-Head Comparison]{.kicker}

::: {.cols .vc}
::: {.col}
- **Same 11.2M ResNet-18 on `FashionMNIST10k`:** Compares `ClassicRecipe` (30 ep, SGD $\eta_0=0.05$, step drops, 1-hot CE) against `ModernRecipe` (45 ep, AdamW $\eta_0=0.002$, cosine warmup, Mixup $0.2$, label smoothing $0.1$).
- **Result:** `ModernRecipe` wins every paired run by **+0.5 to +1.5 percentage points** ($90.5\text{--}91.0\%$ vs. $89.0\text{--}90.5\%$).
:::

::: {.col .fig .big}
@fig:ch07-mcnn-classic-vs-modern-setup
:::
:::
:::

::: {.slide title="Implementing Classic vs. Modern Training Steps"}
[Recipe Implementation]{.kicker}

In code, `ClassicRecipe` and `ModernRecipe` inherit the exact same `ResNet18` backbone and differ only in `configure_optimizers` and `training_step`:

@training-recipes-one-network-two-recipes-2
:::

::: {.slide title="Baseline Trajectory: 30 Epochs of Classic Recipe A"}
[Classic Recipe Training Curve]{.kicker}

Under `ClassicRecipe` (30 epochs, unsmoothed cross-entropy without Mixup), training loss plunges toward zero while validation loss rebounds to $0.4\text{--}0.5$ as logits grow overconfident:

@!training-recipes-one-network-two-recipes-4
:::

::: {.slide title="Diagnostic Pitfall: Never Compare Raw Training Losses"}
[Reading Recipe Curves]{.kicker}

Inspecting the training and validation curves of `ModernRecipe` reveals a diagnostic trap that catches many practitioners:

::: {.cols .vc}
::: {.col}
- **Why `train_loss > val_loss` under `ModernRecipe`:** `train_loss` is evaluated onMixup-blended images with label-smoothed targets (whose theoretical minimum is bounded well above $0$), whereas `val_loss` is plain cross-entropy on clean unmixed validation images.
- **Better calibration:** Under `ClassicRecipe`, training loss hits $\approx 0$ while overconfident validation loss rebounds to $0.4\text{--}0.5$; `ModernRecipe` keeps validation loss below $0.4$.
:::

::: {.col .fig}
@!training-recipes-one-network-two-recipes-5
:::
:::
:::

::: {.slide title="Reading Leaderboards: Ablations & Distribution Shift"}
[Benchmark Methodology]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Always retune LR on optimizer swap:** AdamW ($\eta_0 = 0.002$) vs. SGD ($\eta_0 = 0.05$) differ by $25\times$.
- **2. ImageNet-V2 distribution shift (Recht et al., 2019):** Drops all models by $11\text{--}14\%$ while preserving relative model rankings linearly.
- **3. Downstream transfer:** Once ImageNet top-1 nears its multi-label floor, test backbones on COCO detection and segmentation.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-leaderboard-methodology
:::
:::
:::

::: {.slide title="Concept Check: Smoothed Logits, EMA & Weight Decay"}
[Concept Check]{.kicker}

Test your grasp of modern recipe mechanics with these three calculations:

1. **Optimal logit gap under label smoothing:** For $K = 10$ classes and $\epsilon = 0.1$, what probability $p_y^*$ does label smoothing assign to the true class, what probability $p_k^*$ to each of the 9 incorrect classes, and what is the optimal logit difference $z_y^* - z_k^* = \log(p_y^* / p_k^*)$?
2. **EMA horizon vs. epoch count:** In our 45-epoch `FashionMNIST10k` run ($79$ steps/epoch, $T = 3{,}555$ total steps), how many epochs does an EMA with $\rho = 0.999$ average over, and why can that hurt final validation accuracy?
3. **EMA + BatchNorm pitfall:** When you swap in EMA shadow weights $\bar{\boldsymbol{\theta}}$ at evaluation time, why can BatchNorm's running statistics $(\hat{\mu}, \hat{\sigma}^2)$ be slightly mismatched, and how is this fixed?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $p_y^* = 1 - 0.1 + 0.1/10 = 0.91$, and $p_k^* = 0.1/10 = 0.01$ for each of the 9 other classes. The optimal logit gap is $\log(0.91 / 0.01) = \log(91) \approx \mathbf{4.51}$ (vs. $\infty$ without smoothing).
2. The effective window is $1/(1 - 0.999) = 1{,}000$ steps $\approx \mathbf{12.7\text{ epochs}}$ (over a third of the entire 45-epoch run), pulling the average back toward undertrained early weights.
3. BatchNorm's `moving_mean` and `moving_var` were accumulated using the live weights $\boldsymbol{\theta}_t$ with $\rho_{\text{BN}} = 0.9$ (10-step window), not the smoothed weights $\bar{\boldsymbol{\theta}}_t$. Either maintain EMA shadow copies of BN buffers or run one forward pass over training batches with $\bar{\boldsymbol{\theta}}$ frozen to recompute fresh BN stats.
:::
:::

::: {.slide title="Summary: Key Takeaways on Modern Training Recipes"}
[Summary]{.kicker}

| Ingredient | Mathematical Formulation | Practical Role & Tuning Rule |
|:--|:--|:--|
| **AdamW** | $\boldsymbol{\theta}_{t+1} = (1 - \eta_t \lambda)\boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ | Decouples weight shrinkage $\eta_t \lambda$ from adaptive gradient normalization; base LR $\sim 25\times$ smaller than SGD |
| **Cosine Warmup** | Linear ramp $0 \to \eta_0$ over $t_{\textrm{w}}$, then $\frac{\eta_0}{2}(1 + \cos\frac{\pi(t-t_{\textrm{w}})}{T-t_{\textrm{w}}})$ | Prevents early BN/Adam divergence and eliminates step-drop threshold tuning |
| **Label Smoothing** | $l_\epsilon = -(1-\epsilon)\log p_y - \frac{\epsilon}{K}\sum_k \log p_k$ | Bounds optimal logit gap to $\log(1 + K(1-\epsilon)/\epsilon)$, improving calibration |
| **Mixup / CutMix** | $\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j$, $\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda)\mathbf{y}_j$ | Regularizes inter-class paths; requires longer training horizons (300–600 epochs on ImageNet) |
| **Stochastic Depth** | $\mathbf{x} + \frac{b}{1-p}f(\mathbf{x})$ with $b \sim \mathrm{Bernoulli}(1-p)$ per example | Drops whole residual branches during training; $p_\ell$ ramps linearly from $0$ to $p_{\max}$ with depth |
| **Weight EMA** | $\bar{\boldsymbol{\theta}}_t \leftarrow \rho \bar{\boldsymbol{\theta}}_{t-1} + (1-\rho)\boldsymbol{\theta}_t$ | Damps late-training oscillation at zero inference cost; keep horizon $1/(1-\rho) \ll T$ |
:::

