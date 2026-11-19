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

Let us construct Recipe B's cosine schedule with a peak rate of 0.002 and
verify its values at step 0, at the end of warmup, halfway through decay, and
at the final step.

:begin_tab:`mxnet`
A Gluon optimizer calls a schedule object with the number of the update,
counted from 1. The class `Schedule` turns the multiplier
`d2l.cosine_warmup` into such an object.
:end_tab:

:begin_tab:`pytorch`
`LambdaLR` multiplies the base rate of the optimizer by a function of the
step. Calling `scheduler.step()` after every optimizer step advances it by
one minibatch.
:end_tab:

:begin_tab:`tensorflow`
A Keras optimizer calls a `LearningRateSchedule` object with its step
counter. The class `Schedule` turns the multiplier `d2l.cosine_warmup` into
such an object.
:end_tab:

:begin_tab:`jax`
Optax accepts any function of the step as a learning rate, so the schedule
is a plain function.
:end_tab:

```{.python .input #training-recipes-cosine-schedules-with-warmup-2}
%%tab mxnet
class Schedule(mx.lr_scheduler.LRScheduler):
    """The rate base_lr * factor(t), with t counted from 0."""
    def __init__(self, factor, base_lr=0.01):
        super().__init__(base_lr)
        self.factor = factor

    def __call__(self, num_update):  # MXNet counts updates from 1
        return self.base_lr * self.factor(num_update - 1)

T, t_w = 45 * 79, 3 * 79
schedule = Schedule(lambda t: d2l.cosine_warmup(t, T, t_w), base_lr=2e-3)
[round(schedule(t + 1), 6) for t in (0, t_w, (T + t_w) // 2, T - 1)]
```

```{.python .input #training-recipes-cosine-schedules-with-warmup-2}
%%tab pytorch
T, t_w = 45 * 79, 3 * 79
optim = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=2e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optim, lambda t: d2l.cosine_warmup(t, T, t_w))
rates = []
for t in range(T):  # One optimizer step and one scheduler step per minibatch
    rates.append(scheduler.get_last_lr()[0])
    optim.step()
    scheduler.step()
[round(rates[t], 6) for t in (0, t_w, (T + t_w) // 2, T - 1)]
```

```{.python .input #training-recipes-cosine-schedules-with-warmup-2}
%%tab tensorflow
class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """The rate base_lr * factor(step)."""
    def __init__(self, factor, base_lr):
        self.factor, self.base_lr = factor, base_lr

    def __call__(self, step):
        return self.base_lr * self.factor(step)

T, t_w = 45 * 79, 3 * 79
schedule = Schedule(lambda t: d2l.cosine_warmup(t, T, t_w), base_lr=2e-3)
[round(float(schedule(t)), 6) for t in (0, t_w, (T + t_w) // 2, T - 1)]
```

```{.python .input #training-recipes-cosine-schedules-with-warmup-2}
%%tab jax
T, t_w = 45 * 79, 3 * 79
schedule = lambda t: 2e-3 * d2l.cosine_warmup(t, T, t_w)
[round(float(schedule(t)), 6) for t in (0, t_w, (T + t_w) // 2, T - 1)]
```

The schedule prints 0, 0.002, 0.001, and nearly 0, matching
:eqref:`eq_cosine_warmup` exactly.

### Mixup

*Mixup* :cite:`zhang2018mixup` regularizes a classifier by training on convex
combinations of pairs of training examples. For each minibatch, it samples a
mixing coefficient $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$ and pairs
each example $i$ with a randomly shuffled partner $j$, constructing

$$
\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j, \qquad
\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j,
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

The improvement is modest in absolute scale yet remarkably reliable: in every
paired run, the compact modern recipe beats Recipe A on the same data split by
roughly 0.5 to 1.5 percentage points. (On ImageNet, the full modern recipe
adds over 4 points; here we use only 10,000 small grayscale images, 45 epochs,
and a subset of the menu.) Notice also what a bundled comparison *cannot*
reveal: because we swapped optimizer, schedule, loss, augmentation, and epoch
count simultaneously, this experiment cannot isolate which single ingredient
earned the gain—pinning that down requires a one-at-a-time ablation study,
explored in the exercises below.

Inspecting the loss curves teaches two equally important practical lessons.
Under Recipe A, training loss collapses toward zero while validation loss
climbs back up to 0.4--0.5 even as validation accuracy keeps rising—a textbook
display of the logit overconfidence that label smoothing targets. Under the
modern recipe, validation loss decreases smoothly below 0.4, yet *training*
loss remains higher than Recipe A's—and even higher than its own validation
loss—simply because Mixup and label smoothing evaluate a harder, softer
training objective than plain validation cross-entropy. Never compare raw
training losses across different recipes. Equally important, hyperparameters
do not transfer across optimizers (AdamW's learning rate is $25\times$ smaller
than SGD's), so honest ablations must retune around every change—which is
why rigorous recipe studies like :citet:`wightman2021resnet` demand heavy
compute.

## Comparing Published Results

How should a skeptical reader approach tables of published benchmark
accuracies? With healthy caution. When training recipes alone can move a
fixed ResNet-50 by more than four points on ImageNet, a one-point advantage
between two architectures means very little unless both were trained under
comparable, properly tuned recipes. Rigorous papers therefore retrain
baselines using the same modern recipe rather than copying stale numbers from
older publications.

Even under perfectly matched recipes, single-benchmark test sets carry their
own limits once accuracies cluster tightly. When researchers collected
ImageNet-V2 following the original dataset-construction protocol
:cite:`Deng.Dong.Socher.ea.2009`, every model dropped 11 to 14 points—yet
their relative ranking held steady :cite:`Recht.Roelofs.Schmidt.ea.2019`,
showing that models had not overfit the original validation set so much as
encountered slightly harder images. Meanwhile, multi-label re-annotation of
ImageNet's validation set revealed that many apparent "errors" stem from
ambiguous photographs containing multiple valid objects
:cite:`beyer2020imagenetreal`, so sub-point ImageNet gains do not always
reflect genuine visual superiority. When two backbones sit within a point on
ImageNet top-1, evaluating downstream transfer to object detection or
segmentation is often far more revealing.

## Summary

Between 2015 and the early 2020s, high-accuracy convnet recipes evolved across
every dimension—optimizer (AdamW), schedule (warmup with cosine decay),
augmentation (RandAugment, Mixup, CutMix), regularization (label smoothing,
stochastic depth), training duration, and weight averaging (EMA)—boosting an
unchanged ResNet-50 by more than four ImageNet points. Each mechanism takes
only a few lines of code, and on a 10,000-image Fashion-MNIST benchmark, even
a compact subset of the modern recipe consistently beats a 2015 SGD recipe on
the same ResNet-18 by 0.5--1.5 percentage points. Fair architectural
comparisons therefore require matched, well-tuned training recipes—and in
practice, upgrading a training recipe is often cheaper and more rewarding than
inventing a new architecture. Next, :numref:`sec_convnext` fixes the modern
recipe and asks how far a classic ResNet architecture can be pushed by
modernizing its design one component at a time.

## Exercises

1. **Binary cross-entropy and Mixup.** :citet:`wightman2021resnet` train
   with a binary cross-entropy loss that treats each class as an independent
   yes/no target. Explain why this loss pairs naturally with Mixup and
   CutMix. Consider an image that contains parts of two classes,
   and contrast how softmax and sigmoid outputs can represent it.

1. [code] **CutMix.** CutMix :cite:`yun2019cutmix` draws $\lambda$
   uniformly from $[0, 1]$, that is, from $\mathrm{Beta}(1, 1)$ rather than
   the $\mathrm{Beta}(0.2, 0.2)$ the compact modern recipe uses for Mixup.
   It pastes into each image a rectangle from its shuffled partner that
   covers a fraction $1 - \lambda$ of the image area. Implement CutMix as a
   variant of `mixup`, and mix the labels as in :eqref:`eq_mixup` with
   $\lambda$ replaced by the fraction of the area actually kept, since a
   rectangle clipped at the image border covers less than $1 - \lambda$.
   Train the compact modern recipe on `FashionMNIST10k` once with CutMix in
   place of Mixup and once with the two alternating at random from
   minibatch to minibatch, as modern recipes do. Compare the validation
   accuracy of both runs with the range in :numref:`tab_recipe_results`.

1. [code] **Weight averaging horizon.** Add the `EMA` class to
   `ModernRecipe`: update the shadow copy after every optimizer step, by
   subclassing `d2l.Trainer`, and evaluate with the shadow weights. Sweep
   the decay $\rho$ of :eqref:`eq_ema` over 0.9, 0.99, and 0.999. Count
   the optimizer steps at which `d2l.cosine_warmup` holds the learning rate
   below a tenth of its peak, and relate the best decay to that count
   through the averaging horizon of about $1/(1-\rho)$ steps. Batch
   normalization keeps running statistics that are not parameters; which
   statistics should the averaged model use?

1. [code] **Recipe ablation.** ● In :numref:`tab_recipe_results` the
   compact modern recipe beats recipe A by half a point to a point and a
   half. Whenever a configuration below changes the optimizer, retune the
   learning rate.
    1. Train both recipes on `FashionMNIST10k` five times each and estimate
       the standard deviation of the validation accuracy across runs. How large
       must the difference between two configurations be before one run of
       each resolves it?
    1. Starting from the compact modern recipe, replace one ingredient at a
       time by its recipe A counterpart: remove Mixup, remove label
       smoothing, replace the warmup and cosine decay by recipe A's step
       decay, replace AdamW by recipe A's SGD with momentum, and shorten the
       budget from 45 to 30 epochs. Which of the five changes in validation
       accuracy exceed the threshold of (a), and do the five add up to the
       total gap?
    1. Train the compact modern recipe with both the optimizer and the
       schedule of recipe A. Together with the compact modern recipe and the
       optimizer and schedule replacements of (b), this run fills a
       $2 \times 2$ table of optimizer against schedule. Do the effects of
       the two ingredients interact?

1. [code] **Weight decay and batch normalization.** ● The training
   refinements collected by :citet:`He.Zhang.Zhang.ea.2019` apply weight
   decay (:numref:`sec_weight_decay`) only to the weights of convolutional
   and linear layers and exempt every bias and every scale
   $\boldsymbol{\gamma}$ and shift $\boldsymbol{\beta}$ of batch
   normalization.
    1. When batch statistics are used, the output of batch normalization
       applied to a convolution is unchanged if the convolution's weights
       $\mathbf{W}$ are multiplied by any $c > 0$. What does weight decay on
       $\mathbf{W}$ then do to the loss, to $\|\mathbf{W}\|$ over the course
       of training, and to the effective step size of each gradient update?
    1. Implement both variants, decaying every parameter and decaying only
       the convolutional and linear weights, on `FashionMNIST10k` under the
       compact modern recipe. Compare validation accuracy and plot the norms of
       the exempted parameters over training.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 11.7.*

1. [code] **Stochastic depth in the recipe.** ● The compact modern recipe
   omits stochastic depth. Add it back.
    1. Extend `StochasticResidual` to the downsampling blocks and build the
       ResNet-18 of :numref:`sec_resnet` from it, with the drop probability
       growing linearly from 0 in the first block to $p_{\max}$ in the last.
       What is the expected number of residual branches kept ($b = 1$) for
       one training example, as a function of $p_{\max}$?
    1. Sweep $p_{\max}$ over 0.1, 0.2, and 0.5 on `FashionMNIST10k` under
       the compact modern recipe, which is itself the case $p_{\max} = 0$.
       Does any value raise validation accuracy by more than the run-to-run
       spread of :numref:`tab_recipe_results`? How does the best value
       compare with the range of 0.1 to 0.5 used for much deeper networks?
    1. The per-example implementation of `StochasticResidual` computes
       every branch before masking it. Implement the original per-minibatch
       variant of :citet:`huang2016stochasticdepth`, which draws one $b$ in
       :eqref:`eq_stochastic_depth` per block and step and skips the branch
       entirely when $b = 0$. Use (a) to predict the fraction of residual
       branches it skips at $p_{\max} = 0.5$. Then measure the fraction of
       training time it saves and compare the two.

[Discussions](https://d2l.discourse.group/t/36761)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §7.6]{.kicker}

Modern Training Recipes for Convnets<br>**label smoothing · warmup and cosine decay · Mixup · stochastic depth · weight averaging**
:::
:::

::: {.slide title="Retraining an Unchanged ResNet-50 Adds Four Points"}
[The problem]{.kicker}

Same ResNet-50 architecture, different training procedure (Wightman et
al., 2021), ImageNet top-1:

| recipe | epochs | accuracy |
|:--|--:|--:|
| 2015 original | 90 | 76.1% |
| modern, A3 | 100 | 78.1% |
| modern, A1 (LAMB, heavy augmentation) | 600 | 80.4% |

::: {.d2l-note .warn}
A two-point "architecture gain" over a baseline with an older recipe may be
a recipe gain.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the label-smoothed loss and its minimizer;
- write warmup with cosine decay as a function of the step;
- implement Mixup and stochastic depth in a few lines;
- choose the decay of a weight average from its horizon $1/(1-\rho)$;
- read a recipe comparison: what to hold fixed, which loss to compare.

**Running example:** one ResNet-18 on 10,000 Fashion-MNIST images at
$96\times 96$, under a 2015 recipe and a compact modern one.
:::

::: {.slide title="Recipes Are the Sixth Question of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| AlexNet | learned features, ReLU, dropout, GPUs |
| VGG, NiN, GoogLeNet | blocks, 1 × 1 convolutions, global pooling |
| Normalization Layers | batch, layer, group normalization |
| ResNet, ResNeXt, DenseNet | residual connections make depth trainable |
| Efficient Convnets | separable blocks, scaling, re-parameterization |
| **Training Recipes** | **schedules, augmentation, weight averaging** |
| ConvNeXt | a ResNet modernized one change at a time |
| Design Spaces (RegNet) | rules from populations of networks |
:::

::: {.col .narrow}
- **Uses:** ResNet-18 and its `Residual` block; SGD with momentum; weight
  decay; dropout.
- **Skips:** RandAugment, CutMix, LAMB, distillation.
:::
:::
:::

::: {.slide title="Every Row of the 2015 Recipe Was Revisited"}
[What changed]{.kicker}

| ingredient | 2015 | 2022 |
|:--|:--|:--|
| optimizer | SGD + momentum | AdamW / LAMB |
| schedule | step decay | warmup + cosine |
| augmentation | crops, flips | RandAugment, Mixup, CutMix |
| regularization | weight decay | + label smoothing, stochastic depth |
| duration | 90 epochs | 300–600 epochs |
| evaluated weights | final | moving average |

::: {.d2l-note}
Regularization and duration grow together: the 100-epoch recipe uses weaker
augmentation than the 600-epoch one.
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The ingredients]{.dtitle}

[each a few lines of code]{.dsub}
:::
:::

::: {.slide title="Label Smoothing Makes the Optimal Logits Finite"}
[The ingredients]{.kicker}

$$l_\epsilon = -(1-\epsilon)\log p_y - \frac{\epsilon}{K}\sum_{k=1}^K \log p_k$$

- Minimizer: $p_y = 1-\epsilon+\epsilon/K$, all others $\epsilon/K$: a
  finite logit gap.
- A confident, correct prediction (logits 10, 0, 0) no longer has zero
  loss:

@!training-recipes-label-smoothing
:::

::: {.slide title="Warm Up Linearly, Then Decay along a Half-Cosine"}
[The ingredients]{.kicker}

$$\eta_t = \frac{\eta_0}{2}\Big(1+\cos\frac{\pi(t-t_\textrm{w})}{T-t_\textrm{w}}\Big),\quad t\ge t_\textrm{w};\qquad \eta_t = \eta_0\, t/t_\textrm{w} \textrm{ before.}$$

@!training-recipes-cosine-schedules-with-warmup

The rate is updated after each minibatch ($T = 3555$ steps).
:::

::: {.slide title="Check: Read the Schedule"}
[Check]{.kicker}

Base rate $\eta_0 = 0.002$, $T = 3555$ steps, warmup $t_\textrm{w} = 237$.
What is the rate at step 118, at step 1896, and at the last step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Step 118: in warmup, $0.002\cdot 118/237 \approx 0.001$.
Step 1896: halfway through the decay ($1896 - 237 = 1659 = 3318/2$),
cosine 0, so $0.001$. Last step: about 0.
:::
:::
:::

::: {.slide title="Mixup Trains on Convex Combinations of Examples"}
[The ingredients]{.kicker}

$\tilde{\mathbf{x}} = \lambda\mathbf{x}_i + (1-\lambda)\mathbf{x}_j$,
$\tilde{\mathbf{y}} = \lambda\mathbf{y}_i + (1-\lambda)\mathbf{y}_j$,
$\lambda\sim\mathrm{Beta}(\alpha,\alpha)$; the loss is
$\lambda\, l(\hat{\mathbf{y}}, y_i) + (1-\lambda)\, l(\hat{\mathbf{y}}, y_j)$:

@training-recipes-mixup-1
:::

::: {.slide title="With α = 2 the Mixtures Are Visible"}
[The ingredients]{.kicker}

Top: a minibatch; bottom: mixed with a shuffled copy.

@!training-recipes-mixup-2

Training uses $\alpha = 0.2$: most minibatches barely mixed, a few
strongly. CutMix pastes a rectangle instead and mixes labels by area.
:::

::: {.slide title="Stochastic Depth Drops Whole Residual Branches"}
[The ingredients]{.kicker}

$$\mathbf{x} + \frac{b}{1-p} f(\mathbf{x}),\qquad b\sim\mathrm{Bernoulli}(1-p)\ \textrm{per example}$$

- $\mathbb{E}[b/(1-p)] = 1$: no change needed at evaluation, as with dropout.
- 1,000 examples through a block with $p = 0.5$ in training mode:

@!training-recipes-stochastic-depth

- Typical: $p$ grows linearly from 0 to 0.1–0.5 over the blocks.
:::

::: {.slide title="A Weight Average Reaches Back About 1/(1−β) Steps"}
[The ingredients]{.kicker}

$\bar\theta \leftarrow \rho\bar\theta + (1-\rho)\theta$ after every
step; evaluate with $\bar\theta$. A weight that rises from 0 to 1 over 50
steps, then fluctuates:

@!training-recipes-averaging-weights

$\rho = 0.9$ follows about 10 steps late; $\rho = 0.99$ is smoother but
still a fifth short of 1 after 200 steps.
:::

::: {.slide title="Check: How Long Is the Horizon?"}
[Check]{.kicker}

A run has 79 steps per epoch. Roughly how many epochs does a weight
average with $\rho = 0.999$ reach back? Is that a good choice for a
45-epoch run?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1/(1-\rho) = 1000$ steps, about 13 epochs. The average then
reaches back into a stretch where the rate is still large and the weights
still improve, so it may lag; whether a shorter horizon does better is worth
measuring.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Two Recipes on One Network]{.dtitle}

[same network, same images, everything else changed]{.dsub}
:::
:::

::: {.slide title="One ResNet-18, Two Recipes, the Same 10,000 Images"}
[Two recipes]{.kicker}

| | recipe A (2015) | compact modern recipe |
|:--|:--|:--|
| optimizer | SGD, momentum 0.9, decay $5\cdot 10^{-4}$ | AdamW, decay 0.05 |
| base rate, schedule | 0.05, 2 epochs warmup, divide by 10 at 60%, 85% | 0.002, 3 epochs warmup, cosine |
| loss | cross-entropy | smoothed ($\epsilon = 0.1$), Mixup ($\alpha = 0.2$) |
| epochs | 30 | 45 |

Same He initialization and data for both recipes; each run draws its own
10,000-image subset. Shrinkage per step at the peak rate: AdamW
$0.002 \cdot 0.05 = 10^{-4}$; SGD $0.05 \cdot 5 \cdot 10^{-4} = 2.5 \cdot
10^{-5}$, which momentum multiplies by up to 10.
:::

::: {.slide title="The Compact Modern Recipe Wins Every Run, by 0.5 to 1.5 Points"}
[Two recipes]{.kicker}

Validation accuracy, repeated runs:

| recipe | validation accuracy |
|:--|--:|
| A (2015), 30 epochs | 89–90.5% |
| compact modern, 45 epochs | 90.5–91% |

Gain within each pair of runs: 0.5–1.5 points, positive in every pair.

::: {.d2l-note}
Without RandAugment, CutMix, stochastic depth, or weight averaging: this
measures a subset of the full recipe. Optimizer, loss, schedule, and length
changed together, so the runs do not say which one gained the point.
:::
:::

::: {.slide title="Training Losses Cannot Be Compared across Recipes"}
[Two recipes]{.kicker}

Compact modern recipe: training loss (smoothed, on mixed images) stays
above the validation loss (plain cross-entropy on validation images).

@!training-recipes-one-network-two-recipes-5

- Recipe A: training loss falls to 0, validation loss ends at 0.4–0.5,
  above its early minimum. The modern recipe ends below 0.4.
- Hyperparameters do not transfer: AdamW's base rate is 25× smaller than
  SGD's. Ablations must retune.
:::

::: {.slide title="A One-Point Gain Needs Matched Recipes"}
[Published results]{.kicker}

- The recipe moves a fixed ResNet-50 by four ImageNet points: compare
  architectures only under the same recipe.
- A new ImageNet test set (ImageNet-V2) costs every model 11–14 points but
  preserves their ranking: a harder distribution, not overfitting.
- For backbones within a point on ImageNet, transfer to detection and
  segmentation separates them better.
:::

::: {.slide title="Check: Is the New Block Better?"}
[Check]{.kicker}

A paper reports a new block that beats ResNet-50 by 0.8 points; the
baseline number comes from the 2015 paper. What do you need before you
believe it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The baseline retrained with the new recipe (the recipe alone is
worth four points), and the run-to-run spread of both numbers: 0.8 points
may be within it.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- ResNet-50 unchanged: 76.1% → 80.4% from the recipe alone.
- Label smoothing: loss 0.67 instead of 0.0001 for a confident, correct
  prediction.
- Warmup + cosine per minibatch; Mixup; stochastic depth; weight average
  with horizon $1/(1-\rho)$.
- ResNet-18, 10,000 images: the compact modern recipe wins every run, by
  0.5–1.5 points.
- Compare architectures under matched recipes, never training losses.

Next: ConvNeXt modernizes the architecture under the modern recipe.
:::
