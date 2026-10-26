```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Modern Training Recipes for Convnets
:label:`sec_training_recipes`

Take the ResNet-50 of :numref:`sec_resnet`, change nothing about its
architecture, and train it with the methods in use around 2022 instead of those
of its 2015 debut. Its ImageNet top-1 accuracy rises from 76.1% to 80.4%
:cite:`wightman2021resnet`. That gain, from the *training recipe* alone, is as
large as the gains that many new architectures of those years reported over
ResNet-50. The "ResNet strikes back" study that produced it trained the
unmodified network with three procedures of increasing cost: A3 (100 epochs)
reaches 78.1%, A2 (300 epochs) 79.8%, and A1 (600 epochs, the LAMB optimizer, a
binary cross-entropy loss, and heavy augmentation) 80.4%.

The number matters for anyone who compares networks. If an architecture paper
reports a gain of two points over a baseline trained with an older recipe, the
gain may come from the recipe, not the architecture;
:citet:`wightman2021resnet` make this point by retraining the unchanged
ResNet-50 to the accuracy of several of its successors. This section lists what
changed between 2015 and 2022, implements each ingredient in a few lines, and
then trains one ResNet-18 under an old and a new recipe to measure the
difference.

```{.python .input #training-recipes-training-recipes-matter}
%%tab mxnet
from d2l import mxnet as d2l
import random
import mxnet as mx
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab pytorch
from d2l import torch as d2l
import random
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #training-recipes-training-recipes-matter}
%%tab jax
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
```

## What Changed between 2015 and 2022

The original ResNet recipe was already careful: stochastic gradient descent
with momentum, weight decay, a learning rate divided by 10 twice during
training, and random crops and horizontal flips as augmentation
:cite:`He.Zhang.Ren.ea.2016`. Between then and about 2022, each of these
choices was revisited. The first systematic accounting, in 2019, collected
the techniques that had accumulated (warmup, cosine decay, label smoothing,
Mixup, zero-initializing the last batch-normalization scale in each block,
knowledge distillation)
and showed that together with a small change to the downsampling path they
lift ResNet-50 from 75.3% to 79.3% :cite:`He.Zhang.Zhang.ea.2019`.
:numref:`tab_recipe_2015_2022` summarizes the shift.

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

The *optimizer* row records a common change, not a universal one. Earlier
implementations of Adam added weight decay to the gradient, where Adam's
per-coordinate rescaling distorts it. AdamW :cite:`Loshchilov.Hutter.2019`
applies the decay to the weights directly, outside the rescaling, and made
adaptive methods competitive with tuned SGD on many vision tasks. At batch
sizes in the tens of thousands, rescaling the update of each layer by the
size of its weights keeps training stable, as in LARS
:cite:`You.Gitman.Ginsburg.2017` and its Adam-based relative LAMB, which
recipe A1 above uses.

The *schedule* row replaces the drops by a smooth cosine decay
:cite:`Loshchilov.Hutter.2016`, preceded by a few epochs of linear warmup. The
*augmentation* and *regularization* rows account for much of the gain.
RandAugment :cite:`cubuk2020randaugment` applies randomly chosen image
transformations at a single tuned strength; Mixup :cite:`zhang2018mixup` and
CutMix :cite:`yun2019cutmix` blend pairs of training images and their labels;
label smoothing :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` softens the targets;
stochastic depth :cite:`huang2016stochasticdepth` randomly skips whole residual
blocks during training. Each of these makes the training task harder, so the
network needs more epochs to fit it, and the *duration* row grows from 90
epochs to several hundred. Regularization and duration have to grow together:
the short A3 recipe above uses weaker augmentation than the long A1 recipe,
because a heavily regularized short run can do worse than a lightly regularized
one. The last row changes which weights are evaluated: instead of the final
weights of a noisy optimization, modern recipes evaluate a running average of
the weights.

## Implementing the Ingredients

Each ingredient takes a few lines of code. We implement label smoothing, the
cosine schedule, Mixup, stochastic depth, and the weight average.
RandAugment is a library call in practice, and its contribution lies in the
tuning of the augmentation strength rather than in the code, so we omit it.

### Label Smoothing

With one-hot targets, the cross-entropy loss of a correctly classified
example keeps decreasing as the logit of the correct class grows, so the
loss can never reach its minimum with finite logits and the network is
rewarded for ever larger confidence. Label smoothing replaces the one-hot
target by a mixture that puts weight $1-\epsilon$ on the label and spreads
$\epsilon$ uniformly over all $K$ classes. Writing $p_k$ for the predicted
softmax probabilities, the loss on an example with label $y$ becomes

$$
l_\epsilon = -(1-\epsilon) \log p_y - \frac{\epsilon}{K} \sum_{k=1}^K \log p_k.
$$
:eqlabel:`eq_label_smoothing`

The first term is the ordinary cross-entropy, weighted by $1-\epsilon$; the
second penalizes any class whose probability approaches zero. The loss is
minimized by the probabilities $1-\epsilon+\epsilon/K$ on the correct class
and $\epsilon/K$ on every other class, which a finite gap between the logits
attains. For a prediction that is already very confident and correct, the
unsmoothed loss is nearly zero, while the smoothed loss stays well above
zero:

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

With $\epsilon = 0.1$ the loss of this example is 0.67 instead of 0.0001, so
the optimizer stops pushing confident examples further and spends its effort
on the others.

### Cosine Schedules with Warmup

A step schedule holds the learning rate constant for tens of epochs and then
divides it by 10, an abrupt change whose two times are extra
hyperparameters. The modern default is the schedule we used for batch
normalization in :numref:`sec_batch_norm`. A linear *warmup* raises the rate
from zero to the base rate $\eta_0$ over the first $t_\textrm{w}$ steps, and
a half-cosine *decay* then lowers it to zero at the last step $T$:

$$
\eta_t = \frac{\eta_0}{2} \left(1 + \cos \frac{\pi (t - t_{\textrm{w}})}{T - t_{\textrm{w}}}\right)
\quad \textrm{for } t \geq t_{\textrm{w}}.
$$

At $t = t_\textrm{w}$ the rate is $\eta_0$, halfway through the decay
$\eta_0/2$, and at $t = T$ zero. Warmup exists because a freshly
initialized network can produce large, badly scaled gradients, and a few
steps at small rates let the running statistics of batch normalization and
of the optimizer settle before full-size updates arrive. The long tail of
small rates lets the noisy iterates settle into a region of low loss, and
there is no time of a drop to tune. The function `d2l.cosine_warmup(t,
num_steps, warmup_steps)` of :numref:`sec_batch_norm` returns the factor by
which the schedule multiplies the base rate at step $t$; the optimizer
updates the rate after every minibatch. We plot it next to the step decay of
the 2015 recipe, each over the budget it gets in the experiment later in
this section: 30 epochs for the step decay, to which the experiment adds a
two-epoch warmup, and 45 for the cosine, with 79 minibatches per epoch.

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

Each framework attaches a schedule to its optimizer in its own way. PyTorch
wraps a function of the step in `LambdaLR`; MXNet and Keras call a schedule
object with the step; Optax accepts any function of the step. The cell below
builds recipe B's schedule for a base rate of 0.002 and prints the rate at
the first step, at the end of warmup, halfway through the decay, and at the
last step.

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

All four frameworks give the rates 0, 0.002, 0.001, and nearly 0, as the
formula predicts.

### Mixup

Mixup :cite:`zhang2018mixup` trains on convex combinations of pairs of
examples. For each minibatch it draws a mixing weight
$\lambda \sim \mathrm{Beta}(\alpha, \alpha)$ and a random partner $j$ for
every example $i$, and trains on

$$
\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j, \qquad
\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j,
$$
:eqlabel:`eq_mixup`

where $\mathbf{y}_i$ and $\mathbf{y}_j$ are one-hot label vectors, so the
mixed label $\tilde{\mathbf{y}}$ is a distribution over classes. The
cross-entropy is linear in the target distribution, so the loss on
$\tilde{\mathbf{y}}$ equals the $\lambda$-weighted sum of the losses on the
two original labels, and we implement it that way. Mixup asks the network to
behave linearly between training examples, which discourages it from
memorizing individual examples. With the usual small $\alpha$ (0.1 to 0.2),
the Beta distribution puts most of its mass near 0 and 1, so most minibatches
are barely mixed and a few are blended strongly. The function below mixes a
minibatch with one $\lambda$ for all of it. We save it in the `d2l` package
because the ConvNeXt section reuses it.

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

To see the blending, we mix a minibatch of Fashion-MNIST images with
$\alpha = 2$, which favors values of $\lambda$ near the middle of its range
(training uses a much smaller $\alpha$). The top row shows the originals,
the bottom row their mixtures; the printed $\lambda$ is drawn afresh in
every run.

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

CutMix :cite:`yun2019cutmix` mixes in space instead: it cuts a rectangle
from one image and pastes it into the other, and mixes the labels in
proportion to the area. Every pixel then comes from a real image, and modern
recipes alternate randomly between Mixup and CutMix from one minibatch to the
next. The exercises implement it.

### Stochastic Depth

Dropout (:numref:`sec_dropout`) sets randomly chosen activations to zero
during training. Stochastic depth :cite:`huang2016stochasticdepth` applies
the same idea to whole residual branches: during training, the branch of
each residual block is dropped with probability $p$, so the block passes its
input through unchanged and the step trains a shallower network. Residual
connections make this safe, since the identity path always carries the
signal. For a block that computes $\mathbf{x} + f(\mathbf{x})$, the branch
becomes

$$
\mathbf{x} + \frac{b}{1 - p} f(\mathbf{x}), \qquad b \sim \mathrm{Bernoulli}(1-p).
$$
:eqlabel:`eq_stochastic_depth`

The expected value of $b/(1-p)$ is 1, so the expected output of the branch is
unchanged, and at evaluation time the block runs without dropping, as with
inverted dropout. Modern implementations draw $b$ separately for every
example in the minibatch. The function `drop_path` does this, and we save it
for the ConvNeXt section. The block below subclasses the `Residual` block of
:numref:`sec_resnet` (in the residual block the final ReLU follows the
addition) and drops its branch before the addition. Running 1,000 examples
through it in training mode, the fraction of examples whose output equals
the ReLU of the input, the ones whose branch was dropped, should be close to
$p$.

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

In deep networks the drop probability usually grows linearly from 0 in the
first block to a maximum between 0.1 and 0.5 in the last. The per-example
implementation computes every branch before masking it, so it regularizes
but saves no arithmetic. The original method draws one $b$ per block and
minibatch and can skip a dropped branch entirely, which shortens training.

### Averaging Weights

The last row of :numref:`tab_recipe_2015_2022` costs almost nothing.
Late in training, stochastic gradients make the parameters fluctuate around
a region of low loss, and an average of recent parameters lies closer to the
middle of that region than any single one. Stochastic weight averaging
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018` takes an equal-weight average of
iterates late in training. Modern recipes more often keep an exponential
average: a *shadow copy* $\bar{\theta}$ of the parameters into which, after
every update, the current ones are blended:

$$
\bar{\theta} \leftarrow \beta \bar{\theta} + (1-\beta) \theta.
$$
:eqlabel:`eq_ema`

This is an *exponential moving average* (EMA): the parameters of $k$ steps ago
enter $\bar{\theta}$ with weight $(1-\beta)\beta^k$, so the average reaches
back about $1/(1-\beta)$ steps, 100 steps for $\beta = 0.99$. Training updates
$\theta$ as usual; only evaluation uses $\bar{\theta}$. The class below is the
whole mechanism. It averages the trainable parameters; the running statistics
of batch normalization are not parameters, and the exercises ask which
statistics the averaged network should use. We apply it to a one-weight model
whose weight we set to noisy values around 1, as stochastic gradient steps
would.

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
net.initialize()
ema, raw, avg = EMA(net, decay=0.9), [], []
for t in range(100):  # Noisy iterates around 1.0
    net.weight.set_data(1 + 0.3 * mnp.random.normal(size=(1, 1)))
    ema.update(net)
    raw.append(float(net.weight.data()))
    avg.append(float(ema.shadow['weight']))
d2l.plot(list(range(100)), [raw, avg], xlabel='step', ylabel='weight',
         legend=['raw iterates', 'EMA'])
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
ema, raw, avg = EMA(net, decay=0.9), [], []
for t in range(100):  # Noisy iterates around 1.0
    with torch.no_grad():
        net.weight.copy_(1 + 0.3 * torch.randn(1, 1))
    ema.update(net)
    raw.append(float(net.weight.detach()))
    avg.append(float(ema.shadow['weight']))
d2l.plot(list(range(100)), [raw, avg], xlabel='step', ylabel='weight',
         legend=['raw iterates', 'EMA'])
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

net = tf.keras.layers.Dense(1, use_bias=False)
net.build((None, 1))
ema, raw, avg = EMA(net, decay=0.9), [], []
for t in range(100):  # Noisy iterates around 1.0
    net.kernel.assign(1 + 0.3 * tf.random.normal((1, 1)))
    ema.update(net)
    raw.append(float(net.kernel[0, 0]))
    avg.append(float(ema.shadow[0][0, 0]))
d2l.plot(list(range(100)), [raw, avg], xlabel='step', ylabel='weight',
         legend=['raw iterates', 'EMA'])
```

```{.python .input #training-recipes-averaging-weights}
%%tab jax
class EMA:
    """A shadow copy of a model's parameters, updated after each step."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = nnx.state(model, nnx.Param)

    def update(self, model):  # beta * avg + (1 - beta) * current
        self.shadow = optax.incremental_update(
            nnx.state(model, nnx.Param), self.shadow, 1 - self.decay)

net = nnx.Linear(1, 1, use_bias=False, rngs=nnx.Rngs(d2l.get_key()))
ema, raw, avg = EMA(net, decay=0.9), [], []
for t in range(100):  # Noisy iterates around 1.0
    net.kernel[...] = 1 + 0.3 * jax.random.normal(d2l.get_key(), (1, 1))
    ema.update(net)
    raw.append(float(net.kernel[0, 0]))
    avg.append(float(ema.shadow['kernel'][...][0, 0]))
d2l.plot(list(range(100)), [raw, avg], xlabel='step', ylabel='weight',
         legend=['raw iterates', 'EMA'])
```

With $\beta = 0.9$ the average reaches back about 10 steps: it fluctuates
far less than the iterates and follows their mean. The same horizon is the
risk of the method. If $1/(1-\beta)$ steps is long compared with the time
over which the weights are still improving, the average lags behind them
and evaluates worse than the final weights. Recipes therefore choose
$\beta$ close to 1 (0.999 or 0.9999) only for long runs.

## Controlled Comparison on One Network

We now combine the ingredients and measure what they add. The network is
the ResNet-18 of :numref:`sec_resnet`: a $7 \times 7$ convolution with
stride 2 and a $3 \times 3$ max-pooling, four stages of two residual blocks
with 64, 128, 256, and 512 channels, global average pooling, and a linear
layer. We rebuild it from the library's `Residual` block and initialize all
convolution and linear weights with the He initialization of
:numref:`sec_numerical_stability`, normal with variance $2/n_\textrm{in}$,
and all biases with zero. The task is Fashion-MNIST at $96 \times 96$, with
one change: we train on a random subset of 10,000 of the 60,000 training
images. With less data, regularization matters more, and the experiment
stays small enough to rerun. Each run draws its own subset, and both
recipes of a run share it. Each recipe takes the base rate `lr` and the
total number of optimizer steps `num_steps`, from which its schedule is
computed.

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
        self.net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
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
        layers = [nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
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
            tf.keras.layers.ZeroPadding2D(3),  # Symmetric, as in PyTorch
            tf.keras.layers.Conv2D(64, 7, strides=2,
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
                           kernel_init=he_init, rngs=self.rngs),
                  nnx.BatchNorm(64, momentum=0.9, use_fast_variance=False,
                                rngs=self.rngs),
                  nnx.relu,
                  lambda X: nnx.max_pool(X, (3, 3), (2, 2),
                                         padding=((1, 1), (1, 1)))]
        in_channels = 64
        for i, c in enumerate((64, 128, 256, 512)):
            for j in range(2):
                down = i > 0 and j == 0
                layers.append(d2l.Residual(
                    c, use_1x1conv=down, strides=(2, 2) if down else (1, 1),
                    in_channels=in_channels, rngs=self.rngs))
                in_channels = c
        layers += [lambda X: X.mean(axis=(1, 2)),  # Global average pooling
                   nnx.Linear(512, num_classes, kernel_init=he_init,
                              rngs=self.rngs)]
        self.net = nnx.Sequential(*layers)
```

The two recipes differ only in their optimizers, schedules, and training
losses. *Recipe A* follows 2015: SGD with momentum 0.9, weight decay
$5 \times 10^{-4}$ added to the gradient, a base rate of 0.05 reached by a
linear warmup over the first two epochs and divided by 10 at 60% and at 85%
of the steps, the plain cross-entropy loss, and 30 epochs. Without the
warmup, this rate makes the loss diverge in the first epoch in about a third
of the runs, in every framework. A short warmup is the standard remedy for
large rates with batch normalization; :citet:`He.Zhang.Ren.ea.2016` used one
for their deepest ResNet, and :citet:`Goyal.Dollar.Girshick.ea.2017` made it
routine.
The *compact modern recipe* B uses AdamW with weight decay 0.05, a base rate
of 0.002 with three epochs of warmup and a cosine decay, label smoothing
with $\epsilon = 0.1$, Mixup with $\alpha = 0.2$, and 45 epochs, since
longer training is part of the recipe. It omits RandAugment, CutMix,
stochastic depth, and weight averaging to keep the experiment small, so its
result measures this subset of the full recipe. Both recipes report the
plain cross-entropy as validation loss; label smoothing and Mixup change only
the training loss.

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
    """SGD that adds weight decay to the gradient, as PyTorch and MXNet do."""
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

:begin_tab:`mxnet`
MXNet's AdamW multiplies its weight-decay term by the bias-corrected
learning rate, which is smaller than the learning rate during the first few
thousand steps (by a factor of 0.32 at the first step and 0.8 at step 1,000).
Over this 3,555-step run its weight decay therefore shrinks the weights about
a quarter less than in the other frameworks.
:end_tab:

:begin_tab:`tensorflow`
Keras applies weight decay to the weights outside the momentum buffer, while
the SGD of PyTorch and MXNet and Optax's `add_decayed_weights` add it to the
gradient. `CoupledSGD` adds it to the gradient, so recipe A is the same
algorithm in all four tabs.
:end_tab:

We first train recipe A for 30 epochs and evaluate on the untouched test set
of 10,000 images.

```{.python .input #training-recipes-one-network-two-recipes-4}
data = FashionMNIST10k()
steps_per_epoch = len(data.train_dataloader())  # 79 minibatches of <= 128
model = ClassicRecipe(lr=0.05, num_steps=30 * steps_per_epoch)
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
trainer.fit(model, data)
trainer.history['val_acc'][-1]
```

Then the compact modern recipe trains the same network on the same images
for 45 epochs.

```{.python .input #training-recipes-one-network-two-recipes-5}
model = ModernRecipe(lr=2e-3, num_steps=45 * steps_per_epoch)
trainer = d2l.Trainer(max_epochs=45, num_gpus=1)
trainer.fit(model, data)
trainer.history['val_acc'][-1]
```

:numref:`tab_recipe_results` collects the test accuracy over three runs in
each framework, each run with its own subset, initialization, and
minibatch order.

:The same ResNet-18 under both recipes (10,000 Fashion-MNIST training images at
$96 \times 96$; test accuracy, range over three runs in each of the four
frameworks).
:label:`tab_recipe_results`

| Recipe | Epochs | Test accuracy |
|:--|--:|--:|
| A (2015): SGD, warmup, step decay | 30 | 89.5–90.5% |
| Compact modern: AdamW, cosine, smoothing, Mixup | 45 | 90.5–91% |

The gain is small but consistent. In each of the twelve pairs of runs, the
compact modern recipe beats recipe A trained on the same subset, by half a
point to a point and a half, and its worst run is better than recipe A's
best run in the same framework. On ImageNet the full recipe adds four
points; with 10,000 small grayscale images, a 45-epoch budget, and only part
of the menu, the effect is a fraction of that. The training curves also show why training losses cannot
be compared across recipes. Under the modern recipe the training loss stays
far above recipe A's and even above its own validation loss, because label
smoothing and Mixup make the training targets harder to fit, while the
validation loss is the plain cross-entropy on unmixed test images. The
recipes also show that hyperparameters do not
transfer between them: AdamW's base rate is 25 times smaller than SGD's, so
an ablation that swaps one ingredient must retune the others around it. This
is why careful recipe ablations, such as those of :citet:`wightman2021resnet`,
are expensive.

## Comparing Published Results

The experiment changes how published comparisons should be read. A gain of
a point on ImageNet between two architectures says little unless both were
trained with comparable recipes, since the recipe alone moves a fixed
ResNet-50 by four points. Careful papers therefore retrain their baselines
with their own recipe, and tables that collect numbers from papers of
different years do not support conclusions drawn from one-point gaps.

The test set adds a second source of uncertainty once accuracies are close.
When :citet:`Recht.Roelofs.Schmidt.ea.2019` collected a new test set,
ImageNet-V2, by following the original collection protocol
:cite:`Deng.Dong.Socher.ea.2009`, models lost 11 to 14 points, yet their
ranking was preserved: gains on the original test set carried over, and the
drop came from a slightly harder distribution of images, not from
overfitting to the old test set. :citet:`beyer2020imagenetreal` relabeled
the original validation set with multiple labels per image and found that
many apparent errors are ambiguous or multi-object images, and that part of
the recent progress on the original labels comes from fitting their
idiosyncrasies, so one-point gains measured there need not hold under better
labels. For backbones
whose top-1 accuracies lie within a point, differences in transfer, for
example to object detection on COCO or segmentation on ADE20K, are more
informative.

## Summary

Between 2015 and the early 2020s, high-accuracy recipes for convnets
changed along every axis: AdamW or LAMB, cosine decay with warmup, Mixup,
CutMix, and RandAugment, label smoothing, stochastic depth, longer training,
and evaluation of a weight average. On a fixed ResNet-50 they add about four
ImageNet points. Each ingredient takes a few lines: a smoothed target, a
schedule function, a convex combination of two minibatches, a random mask
on a residual branch, and a shadow copy of the parameters. On a ResNet-18
trained on 10,000 Fashion-MNIST images, a compact subset of the menu gains
half a point to a point and a half over the 2015 recipe in every run and
every framework. Architecture
comparisons therefore need matched training procedures, and with a fixed
network and budget, tuning the procedure can be cheaper than changing the
architecture. :numref:`sec_convnext` starts from the modern recipe and asks
how far a convnet's architecture itself can be modernized.

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
   minibatch to minibatch, as modern recipes do. Compare the test accuracy
   of both runs with the range in :numref:`tab_recipe_results`.

1. [code] **Weight averaging horizon.** Add the `EMA` class to
   `ModernRecipe`: update the shadow copy after every optimizer step, by
   subclassing `d2l.Trainer`, and evaluate with the shadow weights. Sweep
   the decay $\beta$ of :eqref:`eq_ema` over 0.9, 0.99, and 0.999. Count
   the optimizer steps at which `d2l.cosine_warmup` holds the learning rate
   below a tenth of its peak, and relate the best decay to that count
   through the averaging horizon of about $1/(1-\beta)$ steps. Batch
   normalization keeps running statistics that are not parameters; which
   statistics should the averaged model use?

1. [code] **Recipe ablation.** ● In :numref:`tab_recipe_results` the
   compact modern recipe beats recipe A by half a point to a point and a
   half. Whenever a
   configuration below changes the optimizer, retune the learning rate.
    1. Train both recipes on `FashionMNIST10k` five times each and estimate
       the standard deviation of the test accuracy across runs. How large
       must the difference between two configurations be before one run of
       each resolves it?
    1. Starting from the compact modern recipe, replace one ingredient at a
       time by its recipe A counterpart: remove Mixup, remove label
       smoothing, replace the warmup and cosine decay by recipe A's step
       decay, replace AdamW by recipe A's SGD with momentum, and shorten the
       budget from 45 to 30 epochs. Which of the five changes in test
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
       compact modern recipe. Compare test accuracy and plot the norms of
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
       Does any value raise test accuracy by more than the run-to-run
       spread of :numref:`tab_recipe_results`? How does the best value
       compare with the range of 0.1 to 0.5 used for much deeper networks?
    1. The per-example implementation of `StochasticResidual` computes
       every branch before masking it. Implement the original per-minibatch
       variant of :citet:`huang2016stochasticdepth`, which draws one $b$ in
       :eqref:`eq_stochastic_depth` per block and step and skips the branch
       entirely when $b = 0$. Use (a) to predict the fraction of residual
       branches it skips at $p_{\max} = 0.5$. Then measure the fraction of
       training time it saves and compare the two.

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
- choose the decay of a weight average from its horizon $1/(1-\beta)$;
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
- **Skips:** RandAugment, LAMB and LARS details, distillation.
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

Every framework updates the rate after each minibatch ($T = 3555$ steps).
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

- $E[b/(1-p)] = 1$: no change needed at evaluation, as with dropout.
- 1,000 examples through a block with $p = 0.5$ in training mode:

@!training-recipes-stochastic-depth

- Typical: $p$ grows linearly from 0 to 0.1–0.5 over the blocks.
:::

::: {.slide title="A Weight Average Reaches Back About 1/(1−β) Steps"}
[The ingredients]{.kicker}

$\bar\theta \leftarrow \beta\bar\theta + (1-\beta)\theta$ after every
step; evaluate with $\bar\theta$. Noisy iterates around 1, $\beta = 0.9$:

@!training-recipes-averaging-weights
:::

::: {.slide title="Check: How Long Is the Horizon?"}
[Check]{.kicker}

A run has 79 steps per epoch. Roughly how many epochs does a weight
average with $\beta = 0.999$ reach back? Is that a good choice for a
45-epoch run?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1/(1-\beta) = 1000$ steps, about 13 epochs. The average then
reaches back into a stretch where the rate is still large and the weights
still improve, so it may lag; whether a shorter horizon does better is worth
measuring.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[A controlled comparison]{.dtitle}

[one network, two recipes]{.dsub}
:::
:::

::: {.slide title="One ResNet-18, Two Recipes, the Same 10,000 Images"}
[A controlled comparison]{.kicker}

| | recipe A (2015) | compact modern recipe |
|:--|:--|:--|
| optimizer | SGD, momentum 0.9, decay $5\cdot 10^{-4}$ | AdamW, decay 0.05 |
| base rate, schedule | 0.05, 2 epochs warmup, divide by 10 at 60%, 85% | 0.002, 3 epochs warmup, cosine |
| loss | cross-entropy | smoothed ($\epsilon = 0.1$), Mixup ($\alpha = 0.2$) |
| epochs | 30 | 45 |

Same He initialization and data in every framework; each run draws its own
10,000-image subset.
:::

::: {.slide title="The Compact Modern Recipe Wins Every Run, by 0.5 to 1.5 Points"}
[A controlled comparison]{.kicker}

Test accuracy, three runs in each of four frameworks:

| recipe | test accuracy |
|:--|--:|
| A (2015), 30 epochs | 89.5–90.5% |
| compact modern, 45 epochs | 90.5–91% |

Gain within each pair of runs: 0.5–1.3 points, 12 of 12 positive.

::: {.d2l-note}
Without RandAugment, CutMix, stochastic depth, or weight averaging: this
measures a subset of the full recipe.
:::
:::

::: {.slide title="Training Losses Cannot Be Compared across Recipes"}
[A controlled comparison]{.kicker}

Compact modern recipe: training loss (smoothed, on mixed images) stays
above the validation loss (plain cross-entropy on test images).

@!training-recipes-one-network-two-recipes-5

- Hyperparameters do not transfer: AdamW's base rate is 25× smaller than
  SGD's. Ablations must retune.
:::

::: {.slide title="A One-Point Gain Needs Matched Recipes"}
[A controlled comparison]{.kicker}

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
  with horizon $1/(1-\beta)$.
- ResNet-18, 10,000 images: the compact modern recipe wins every run, by
  0.5–1.5 points.
- Compare architectures under matched recipes, never training losses.

Next: ConvNeXt modernizes the architecture under the modern recipe.
:::
