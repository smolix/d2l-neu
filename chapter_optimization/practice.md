# Practice
:label:`sec_practice`

Every experiment in this chapter so far ran in seconds or minutes. We could
sweep a learning rate over four values, repeat a comparison, and restart
a run that went wrong. A large training run allows none of this.
PaLM, a language model with 540 billion parameters, was trained on 6,144 TPU
chips, and its loss spiked about twenty times along the way, although
gradient clipping was enabled :cite:`chowdhery2022palm`. At that scale a
sweep over four learning rates costs four times the run itself, so most
hyperparameters are set once and never tuned on the model that matters. A
single oversized update can undo days of progress. And the checkpoint at
the end is one draw from a noisy trajectory, yet it is the one that ships.

This section follows such a run through its life, using the methods of the
earlier sections on models small enough to train here. We first ask how a
configuration is chosen when the target cannot be tuned, and read the
settings that four recent reports disclose. We then protect the run against
rare oversized updates with gradient clipping, and ask what to do when a
spike gets through anyway. Next we reduce the noise in the final weights by
averaging them. Finally we turn to the question behind every comparison in
this chapter: what does it take to show that one optimizer is better than
another?

```{.python .input #practice}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
import torch
from torch import nn
```

```{.python .input #practice}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
from jax import numpy as jnp
import optax
```

## Choosing a Configuration

A run too large to tune directly takes its hyperparameters from two
sources. The first is smaller proxies. :numref:`sec_scaling` showed that
under a suitable parametrization the best learning rate of a narrow model
stays the best as the width grows, so most of the sweeping can happen on
models that cost a fraction of the target. The second is precedent: the
settings that worked for earlier runs of the same kind.

### Reported Configurations

Some of that precedent is public. :numref:`tab_practice_recipes` collects
the optimizer configurations disclosed by four prominent pretraining reports
from 2024 to
2026. Each entry is what a team reported shipping, usually without an
ablation, and a dash means that the report does not say.

:What four pretraining reports disclose about their optimizer configuration.
A dash means the report does not say. Batch sizes are quoted the way each
report quotes them (tokens or sequences).
:label:`tab_practice_recipes`

| run | optimizer | $\beta_1, \beta_2$ | peak LR and schedule | warmup | clip | weight decay | batch size |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` | AdamW | — | $8{\times}10^{-5}$, cosine to $8{\times}10^{-7}$ | 8k steps | — | — | ramp 4M $\to$ 8M $\to$ 16M tokens |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, constant to 10T tokens, cosine to $2.2{\times}10^{-5}$ | 2k steps | 1.0 | 0.1 | ramp 3,072 $\to$ 15,360 seqs over 469B tokens |
| OLMo 2 7B :cite:`OLMo.2025` | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, cosine to 10% of peak | 2k steps | 1.0 | 0.1, embeddings exempt | 1,024 seqs, fixed |
| Kimi K2 :cite:`Kimi.Team.2025` | MuonClip | — | $2{\times}10^{-4}$, constant to 10T tokens, cosine to $2{\times}10^{-5}$ | 500 steps | — | 0.1 | 67M tokens, fixed |

Three of the four runs use AdamW. The fourth, Kimi K2, applies Muon to its
hidden weight matrices and AdamW to everything else, the split of
:numref:`sec_muon`. The two reports that disclose the moment coefficients
both choose $(\beta_1, \beta_2) = (0.9, 0.95)$. With $\beta_2 = 0.95$,
Adam's average of the squared gradients reaches back about
$1/(1-\beta_2) = 20$ steps instead of the thousand steps of the default
$\beta_2 = 0.999$, so it adapts quickly when the scale of the gradients
changes. Three of the four decay the weights with $\lambda = 0.1$, and
the two that report clipping clip the global gradient norm at 1. All four
warm up and later decay the learning rate, either along a cosine from the
start or after a long phase at a constant rate, the warmup–stable–decay
shape of :numref:`sec_scheduler`. Two ramp the batch size up during
training, and two keep it fixed. Every report sets its own peak learning
rate, and the four values span almost a factor of four, from
$8 \times 10^{-5}$ for the dense 405-billion-parameter model to
$3 \times 10^{-4}$ for the 7-billion one. The dashes record missing
evidence, not implied defaults, and even the complete rows fall short of
what a reproduction would need.

### Tuning Budgets

The settings that recur in the table are a defensible starting point for a
new run of the same kind, because each was chosen by a team that could
test it at scale. They are not proven optima, because the reports
rarely ablate them. How far to depart from them depends on how many runs
the budget allows.

With a handful of runs, keep the recurring settings and use every run to
tune the peak learning rate. It is the setting that each report chose for
itself, and in the sweeps of this chapter a factor of three in the learning
rate separated the best run from a diverging one. With tens of runs, add
the weight decay and the schedule, and tune jointly only what is coupled.
In AdamW each step multiplies the weights by $1 - \eta\lambda$
(:numref:`sec_adamw`), so the learning rate $\eta$ and the weight decay
$\lambda$ set the shrinkage of the
weights through their product. A joint grid should therefore vary the
product $\eta\lambda$ and also move along lines on which the product is
constant. With more runs than that, the tuning becomes a study of its own,
and the protocol of :numref:`subsec_practice-tuning` applies.

## Gradient Clipping

The clipping column of :numref:`tab_practice_recipes` guards against a
failure that this chapter has already met. In :numref:`sec_adam`, SGD with
momentum trained a small transformer best at the learning rate $\eta = 0.3$
and diverged at $\eta = 1$, one point higher on the grid. Global-norm
clipping, introduced in :numref:`sec_rnn-scratch` to tame exploding RNN
gradients, treats the gradients of all parameters as one long vector
$\mathbf{g}$ and shrinks this vector whenever its length exceeds a
threshold $\theta$:

$$
\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}.
$$
:eqlabel:`eq_practice_clip`

A gradient shorter than $\theta$ passes unchanged, and a longer one keeps
its direction and is scaled to length $\theta$. At $\theta = 1$, a gradient
of norm 5 is multiplied by 0.2. Because the norm is taken over all
parameters at once, clipping preserves the relative sizes of the gradients
of different layers and shrinks only the step as a whole.

:begin_tab:`pytorch`
`clip_grad_norm_` rescales the gradients in place and returns the norm it
measured before clipping. The wrapper below calls it between `backward` and
`step`. It keeps the returned norms on the device, so that recording them
adds no synchronization with the host, and we read them once after
training.
:end_tab:

:begin_tab:`jax`
In Optax, clipping is a gradient transformation like any other, so
`optax.chain` places it in front of an optimizer, and the chained
transformation drops into `nnx.Optimizer` like a plain one.
:end_tab:

```{.python .input #practice-gradient-clipping-1}
%%tab pytorch
class Clipped:
    """Clip the global gradient norm before every optimizer step."""
    def __init__(self, optimizer, params, max_norm=1.0):
        self.optimizer, self.params = optimizer, list(params)
        self.max_norm, self.norms = max_norm, []

    def step(self):
        self.norms.append(nn.utils.clip_grad_norm_(self.params, self.max_norm))
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
```

```{.python .input #practice-gradient-clipping-1}
%%tab jax
def clipped(tx, max_norm=1.0):
    """Clip the global gradient norm before an optimizer's update."""
    return optax.chain(optax.clip_by_global_norm(max_norm), tx)
```

### Preventing a Numerical Failure
:label:`subsec_practice-nan`

We now rerun SGD just above the divergent point of that sweep, at
$\eta = 1.2$, with and without clipping. At $\eta = 1$ itself an
occasional run survives its first steps; at $\eta = 1.2$ the unclipped run
diverges reliably. The model is `TinyLM`, the decoder-only transformer of
:numref:`subsec_tinylm`: two blocks, about 0.4 million parameters, trained
to predict the next character of *The Time Machine* from the 64 characters
before it. The library function `d2l.train_lm` trains it for a fixed number
of steps on minibatches of 64 sequences and returns the training loss of
every step. Both runs start from the same initial weights and use SGD with
momentum 0.9 at $\eta = 1.2$ for 2,000 steps; they differ only in the
clipping. To see how often and when clipping intervenes, we also record the
global gradient norm before clipping at every step.

```{.python .input #practice-a-nan-averted-1}
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)
```

:begin_tab:`jax`
An Optax transformation is a pair of pure functions, `init` and `update`,
so a quantity that must persist from step to step lives in the
transformation's state. `record_norms` passes the gradients through
unchanged and writes each step's global norm into an array of fixed length
in its state.
:end_tab:

```{.python .input #practice-a-nan-averted-4}
%%tab jax
def record_norms(num_steps):
    """Pass the gradients through unchanged, storing each step's norm."""
    def init(params):
        return dict(norms=jnp.zeros(num_steps), step=jnp.zeros((), jnp.int32))

    def update(grads, state, params=None):
        norm = optax.global_norm(grads)
        norms = state['norms'].at[state['step']].set(norm)
        return grads, dict(norms=norms, step=state['step'] + 1)

    return optax.GradientTransformation(init, update)
```

```{.python .input #practice-a-nan-averted-2}
%%tab pytorch
model0 = d2l.TinyLM(len(data.vocab))
curves = {}
for clip in (False, True):
    model = copy.deepcopy(model0)  # Both runs start from the same weights
    optimizer = torch.optim.SGD(model.parameters(), lr=1.2, momentum=0.9)
    if clip:
        optimizer = Clipped(optimizer, model.parameters())
    label = 'clipped at 1' if clip else 'unclipped'
    curves[label] = d2l.train_lm(model, data, optimizer, num_steps=2000)
    print(f'{label}: final loss {sum(curves[label][-100:]) / 100:.3f}')
norms = torch.stack(optimizer.norms)
fired = torch.nonzero(norms > 1).flatten()  # Steps on which clipping acted
print(f'clipping changed the update on {len(fired)} of {len(norms)} steps, '
      f'the last at step {int(fired.max())}; '
      f'median gradient norm {norms.median():.2f}')
```

```{.python .input #practice-a-nan-averted-2}
%%tab jax
model0 = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
curves = {}
for clip in (False, True):
    model = nnx.clone(model0)  # Both runs start from the same weights
    tx = optax.sgd(1.2, momentum=0.9)
    if clip:
        tx = optax.chain(record_norms(2000), clipped(tx))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    label = 'clipped at 1' if clip else 'unclipped'
    curves[label] = d2l.train_lm(model, data, optimizer, num_steps=2000)
    print(f'{label}: final loss {sum(curves[label][-100:]) / 100:.3f}')
norms = optimizer.opt_state[0]['norms'][...]
fired = jnp.nonzero(norms > 1)[0]  # Steps on which clipping acted
print(f'clipping changed the update on {len(fired)} of {len(norms)} steps, '
      f'the last at step {int(fired.max())}; '
      f'median gradient norm {float(jnp.median(norms)):.2f}')
```

The unclipped run ends with a loss of `nan`, "not a number", the value that
floating-point arithmetic produces once a computation has overflowed. The
instability occurs early, so we plot the first 150 steps on a log scale.

```{.python .input #practice-a-nan-averted-3}
d2l.plot(list(range(150)), [curves[k][:150] for k in curves], 'step',
         'training loss', legend=list(curves), yscale='log')
```

After a few steps the unclipped run begins to feed on itself. An
oversized step moves the parameters to where the loss is steeper, the
gradient there is larger, and momentum carries the excess into the
following steps. Once this starts, the loss grows by several orders of
magnitude within about fifteen steps and then overflows. The clipped run,
at the same learning rate, trains normally, to a final loss comparable with
that of the best SGD run in :numref:`sec_adam`, which used a learning rate
four times smaller.

### Clipping Frequency

The printed counts show how little clipping did to achieve this, and when
it did it. The median gradient norm of the clipped run is about a third of
the threshold, and clipping changed the update on only a handful of the
2,000 steps, all of them among the first twenty. These are the steps
on which the freshly initialized network has its largest gradients. Once
they are past, the clipped run trains at $\eta = 1.2$ with clipping idle for
the rest of the 2,000 steps. Here clipping did the job of a warmup, and an
adaptive one. It shortened exactly those first steps that were too long,
until the parameters reached a region where the full step is safe, and it
needed no schedule to tell it when to stop. Every run in
:numref:`tab_practice_recipes` warms up as well, and the exercises compare
the two remedies.

Used this way, clipping is a *guard*: it leaves nearly every update alone
and rescales only the rare oversized ones. In long runs the same guard has
a second job. Gradient noise in language models is heavy-tailed, so long
after the start an occasional minibatch produces a gradient many times
longer than typical ones, and under noise of this kind clipped SGD provably
converges where plain SGD can fail :cite:`Zhang.Karimireddy.Veit.ea.2020`.

A guard works only if the threshold lies above the gradient norms of a
stable run, so that it fires rarely. If clipping fires on most steps, it
rescales nearly every update and thereby changes the update rule itself.
The learning rate and the threshold then act together on the length of the
step, and they have to be tuned together
:cite:`Godbole.Dahl.Gilmer.ea.2023`.

### Clipping and Adam

Clipping protects Adam in a different way. Adam keeps two leaky averages
(:numref:`sec_adam`): $\mathbf{m}$ of the gradients, with weight
$1 - \beta_1$ on each new gradient, and $\mathbf{v}$ of their squares, with
weight $1 - \beta_2$. Suppose a gradient coordinate $g$ arrives that dwarfs
every earlier one. After this step $m \approx (1 - \beta_1) g$ and
$v \approx (1 - \beta_2) g^2$, so the update of that coordinate is about
$\eta (1 - \beta_1)/\sqrt{1 - \beta_2}$, or $3.2\,\eta$ at the defaults
$\beta_1 = 0.9$ and $\beta_2 = 0.999$, however large $g$ is. Adam's update
is thus already bounded in each coordinate, and clipping, which scales $g$
before either average sees it, changes this one step little as long as the
clipped gradient still dominates. What clipping protects is Adam's state.
Without it, the huge $g^2$ stays in $v$ and fades only by the factor
$\beta_2$ per step, with a time constant of $1/(1 - \beta_2) = 1{,}000$
steps. A spike a hundred times the typical gradient keeps $v$ above its
usual level for more than two thousand steps, and throughout that time the
inflated denominator $\sqrt{v}$ suppresses the updates of the coordinate.
Clipping is therefore no substitute for lowering a learning rate that is
too large for Adam, but it keeps one bad minibatch from silencing a
parameter for thousands of steps. With the $\beta_2 = 0.95$ of
:numref:`tab_practice_recipes`, both effects shrink: the time constant
falls to 20 steps, and the bound on the step falls to
$0.1/\sqrt{0.05} \approx 0.45$ times $\eta$.

### Loss Spikes and Recovery

Clipping does not prevent every failure. The PaLM run quoted at the start of
this section had clipping enabled, and its loss still spiked about twenty
times, at irregular intervals and sometimes late in training, while smaller
models trained the same way did not spike. Other guards act on the loss
rather than on the gradient. PaLM added a *z-loss*, the penalty
$10^{-4} \log^2 Z$ on the softmax normalizer $Z = \sum_j \exp(o_j)$ of the
output logits $o_j$. The cross-entropy does not change when the same
constant is added to every logit, so nothing in it stops the logits from
drifting as a whole toward large values, where they lose precision in
bfloat16. The z-loss keeps $\log Z$ close to zero and with it the overall
level of the logits. PaLM used it throughout training, and the spikes came
anyway :cite:`chowdhery2022palm`.

When a spike does occur, the remedy is operational. The PaLM team
restarted from a checkpoint about 100 steps before each spike and skipped
the 200 to 500 minibatches seen around it, after which the loss did not
spike again at that point. The skipped data was not bad in itself: the same
minibatches, replayed from a different, earlier checkpoint, caused no
spike. The spikes came from a particular combination of data and parameter
state :cite:`chowdhery2022palm`. Recovery of this kind needs checkpoints
frequent enough that a rewind costs hours rather than days, and each
checkpoint must hold everything needed to resume: the model and optimizer
state, the step count that drives the schedule, and the position in the
data stream.

## Weight Averaging
:label:`subsec_practice-weight-averaging`

The last checkpoint of a run is one draw. At a constant learning rate, SGD
does not settle at a point: its iterates keep fluctuating in a region
around a minimum, the noise ball of :numref:`sec_sgd`, whose size grows
with the learning rate and with the variance of the gradient noise. A
schedule shrinks this region by decaying the learning rate
(:numref:`sec_scheduler`). Averaging the iterates reduces the noise in a
different way, without touching the learning rate. When the fluctuations
around the center of the region are only weakly correlated, they partly
cancel in the mean.

### The Exponential Moving Average

Stochastic weight averaging takes an equal-weight average of checkpoints
from the tail of training :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`. The
exponential moving average (EMA) that :numref:`sec_training_recipes` added
to the standard training recipe weights recent iterates more heavily and
needs no choice of where the tail starts. It keeps a running average
$\bar{\mathbf{x}}_t$ of the weights $\mathbf{x}_t$, written here with the
decay $\alpha$ because $\beta$ already denotes momentum:

$$
\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1 - \alpha)\, \mathbf{x}_t.
$$
:eqlabel:`eq_practice_ema`

This is the leaky average that momentum applies to gradients and Adam to
squared gradients, now applied to the weights themselves. The weights of
$k$ steps ago enter with weight $(1 - \alpha)\alpha^k$, so the average
reaches back about $1/(1 - \alpha)$ steps. Training never sees
$\bar{\mathbf{x}}_t$; only evaluation uses it. Its cost is one more copy of
the parameters in memory.

### Averaging at a Constant Learning Rate

We test the average on the convolutional network of
:numref:`sec_scheduler`: a LeNet with ReLU activations, max-pooling, and
batch normalization after every hidden layer, initialized with Xavier
weights and zero biases, and trained on Fashion-MNIST in minibatches of 256
images. At the constant learning rate $\eta = 0.3$, the baseline of that
section, its test accuracy gains little after about ten epochs and then
moves up and down from epoch to epoch. Alongside the live weights we keep
an EMA with $\alpha = 0.999$, whose window of about 1,000 steps spans
roughly four epochs of 235 steps each.

```{.python .input #practice-weight-averaging-1}
%%tab pytorch
class LeNetBN(nn.Sequential):
    """LeNet with ReLU, max-pooling, and batch normalization."""
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
            nn.Linear(84, 10))
        for m in self.modules():  # Xavier weights, zero biases
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

fashion = d2l.FashionMNIST(batch_size=256)
train_iter = fashion.get_dataloader(train=True)
test_iter = fashion.get_dataloader(train=False)
device = d2l.try_gpu()
```

```{.python .input #practice-weight-averaging-1}
%%tab jax
class LeNetBN(nnx.Module):
    """LeNet with ReLU, max-pooling, and batch normalization."""
    def __init__(self, rngs):
        xavier = nnx.initializers.xavier_uniform()
        # Running averages decay 0.9; the variance is computed in two passes
        bn = lambda n: nnx.BatchNorm(n, momentum=0.9, use_fast_variance=False,
                                     rngs=rngs)
        self.conv1 = nnx.Conv(1, 6, kernel_size=(5, 5), padding='same',
                              kernel_init=xavier, rngs=rngs)
        self.bn1 = bn(6)
        self.conv2 = nnx.Conv(6, 16, kernel_size=(5, 5), padding='valid',
                              kernel_init=xavier, rngs=rngs)
        self.bn2 = bn(16)
        self.fc1 = nnx.Linear(16 * 5 * 5, 120, kernel_init=xavier, rngs=rngs)
        self.bn3 = bn(120)
        self.fc2 = nnx.Linear(120, 84, kernel_init=xavier, rngs=rngs)
        self.bn4 = bn(84)
        self.fc3 = nnx.Linear(84, 10, kernel_init=xavier, rngs=rngs)

    def __call__(self, x):
        x = nnx.max_pool(nnx.relu(self.bn1(self.conv1(x))),
                         window_shape=(2, 2), strides=(2, 2))
        x = nnx.max_pool(nnx.relu(self.bn2(self.conv2(x))),
                         window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.bn3(self.fc1(x)))
        x = nnx.relu(self.bn4(self.fc2(x)))
        return self.fc3(x)

fashion = d2l.FashionMNIST(batch_size=256)
train_iter = fashion.get_dataloader(train=True)
test_iter = fashion.get_dataloader(train=False)
```

The averaged network needs batch-normalization statistics of its own for
prediction, so we average the running means and variances along with the
weights. This is an approximation: an average of the statistics of many
networks is not the statistic of the averaged network. It is close when the
weights change little over the window.

:begin_tab:`pytorch`
`AveragedModel` keeps a copy of the network. The first call of its method
`update_parameters` copies the live weights into it, and with
`get_ema_multi_avg_fn` every later call blends them in as in
:eqref:`eq_practice_ema`. With
`use_buffers=True` it averages the batch-normalization statistics too. The
accuracy is counted on the device and read once per evaluation.
:end_tab:

:begin_tab:`jax`
`nnx.state` with the filter `(nnx.Param, nnx.BatchStat)` collects the
weights and the running statistics of a model.
`optax.incremental_update(new, old, 1 - alpha)` computes
:eqref:`eq_practice_ema` for every array, and `nnx.update` writes the result
into the averaged model.
:end_tab:

```{.python .input #practice-weight-averaging-4}
%%tab pytorch
loss = nn.CrossEntropyLoss()

@torch.inference_mode()
def accuracy(model, data_iter):
    """Fraction of test examples classified correctly."""
    model.eval()  # Use the running averages in the BatchNorm layers
    correct = 0
    for X, y in data_iter:
        correct += (model(X.to(device)).argmax(1) == y.to(device)).sum()
    return correct.item() / len(data_iter.dataset)
```

```{.python .input #practice-weight-averaging-4}
%%tab jax
@nnx.jit
def train_step(model, optimizer, X, y):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X), y).mean()
    optimizer.update(model, nnx.grad(loss_fn)(model))

@nnx.jit
def ema_update(ema, model, alpha=0.999):
    kinds = (nnx.Param, nnx.BatchStat)  # Weights and running statistics
    nnx.update(ema, optax.incremental_update(
        nnx.state(model, kinds), nnx.state(ema, kinds), 1 - alpha))

@nnx.jit
def count_correct(model, X, y):
    return (model(X).argmax(axis=1) == y).sum()

def accuracy(model, data_iter):
    """Fraction of test examples classified correctly."""
    model.eval()  # Use the running averages in the BatchNorm layers
    correct, total = 0, 0
    for X, y in data_iter:
        correct += count_correct(model, jnp.asarray(X), jnp.asarray(y))
        total += len(y)
    return int(correct) / total
```

```{.python .input #practice-weight-averaging-2}
%%tab pytorch
lr, num_epochs = 0.3, 15
net = LeNetBN().to(device)
ema = torch.optim.swa_utils.AveragedModel(
    net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    use_buffers=True)
ema.update_parameters(net)  # Start the average at the initial weights
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
live_acc, avg_acc = [], []
for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        optimizer.zero_grad()
        loss(net(X.to(device)), y.to(device)).backward()
        optimizer.step()
        ema.update_parameters(net)
    live_acc.append(accuracy(net, test_iter))
    avg_acc.append(accuracy(ema, test_iter))
for name, acc in (('live', live_acc), ('EMA', avg_acc)):
    print(f'{name}: final test accuracy {acc[-1]:.3f}, last five epochs '
          f'{min(acc[-5:]):.3f} to {max(acc[-5:]):.3f}')
```

```{.python .input #practice-weight-averaging-2}
%%tab jax
lr, num_epochs = 0.3, 15
model = LeNetBN(nnx.Rngs(d2l.get_key()))
ema = nnx.clone(model)
optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)
live_acc, avg_acc = [], []
for epoch in range(num_epochs):
    model.train()
    for X, y in train_iter:
        train_step(model, optimizer, jnp.asarray(X), jnp.asarray(y))
        ema_update(ema, model)
    live_acc.append(accuracy(model, test_iter))
    avg_acc.append(accuracy(ema, test_iter))
for name, acc in (('live', live_acc), ('EMA', avg_acc)):
    print(f'{name}: final test accuracy {acc[-1]:.3f}, last five epochs '
          f'{min(acc[-5:]):.3f} to {max(acc[-5:]):.3f}')
```

```{.python .input #practice-weight-averaging-3}
d2l.plot(list(range(1, num_epochs + 1)), [live_acc, avg_acc], 'epoch',
         'test accuracy', legend=['live weights', 'EMA of weights'])
```

Two features of the curves matter. First, the average needs its window to
fill before it is useful. For the first few epochs it trails the live
weights by a wide margin, because it still gives much weight to iterates
near the initialization: after one epoch of 235 steps, the initial weights
still carry the weight $0.999^{235} \approx 0.79$. This is lag, not the bias
toward zero that Adam corrects (:numref:`sec_adam`). The average starts at
the initial weights, so its coefficients already sum to one, and there is
nothing to correct. Practical EMAs therefore start averaging late in
training or let $\alpha$ grow from a small value over the first steps.

Second, once the window has filled, the EMA curve is smooth and runs along
the top of the live curve. Over the last five epochs the live accuracy
jumps by several points, while the EMA's rises steadily, by one to two
points, and ends at its highest value. The live curve shows the noise ball
directly: each epoch evaluates a different point in it, while the EMA
evaluates close to its center.

### Averaging and Decay

At a constant learning rate, the averaged weights reach about the final
accuracy of the decayed schedules of :numref:`sec_scheduler`, roughly 91%.
Decay and averaging get there by different routes. Decay reduces the noise
injected into future iterates and so changes the trajectory, while
averaging leaves the trajectory alone and reduces the noise in the
parameters we evaluate. The noise-ball argument also predicts a limit. Once
a schedule has decayed the learning rate to nearly zero, the iterates
barely move, and their average lies close to the last iterate. Averaging is
therefore most useful when no decay is planned, or when a decayed endpoint
is expensive to reach, as for the checkpoints evaluated during a long run
whose final length is not yet fixed. Large runs average checkpoints even
after decay, where the gain is smaller: the released Llama 3 model is the
average of checkpoints from the final phase of training, during which the
learning rate was annealed to zero
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. When you average a few saved
checkpoints instead of keeping a running EMA, the batch-normalization
statistics belong to none of the averaged weights. Recompute them with a
pass over the training data before trusting the result.

## Comparing Optimizers
:label:`subsec_practice-tuning`

Every comparison in this chapter rests on a tuning protocol, and the
protocol is part of the result.

### Scientific and Nuisance Hyperparameters

The Deep Learning Tuning Playbook
:cite:`Godbole.Dahl.Gilmer.ea.2023` sorts the hyperparameters of an
experiment into three classes. *Scientific* hyperparameters are the ones
the question is about. *Nuisance* hyperparameters must be re-optimized for
every value of the scientific ones before the comparison means anything.
*Fixed* hyperparameters are held constant, and each of them limits the
claim. A comparison says something about its scientific hyperparameters
only after the nuisance hyperparameters have been retuned in each arm. The
most common violation is also the simplest. The learning rate is almost
always a nuisance hyperparameter, and a single shared learning rate favors
whichever method happens to prefer it.

The optimizer races of :numref:`sec_adam` and :numref:`sec_muon` followed
this protocol. The optimizer was the scientific hyperparameter. The
learning rate was the nuisance, retuned for each optimizer on a grid of
four or five values spaced by factors of about three. The number of steps,
the batch size, the model, and the absence of a schedule were fixed and
stated, and every conclusion holds for that protocol only. At larger scale
the same design has more nuisance dimensions (the peak learning rate, the decay
horizon, the warmup, and the weight decay), replaces coarse grids by
quasi-random search, and separates a broad exploration from a narrow final
sweep.

### Tuning Effort and the Verdict

The protocol can decide the verdict. :citet:`Schmidt.Schneider.Hennig.2021`
benchmarked fifteen optimizers on a range of problems and found that trying
several optimizers at their default settings works about as well as tuning
a single one extensively. Defaults encode much accumulated tuning, and a
comparison without per-arm tuning measures how much effort went into each
arm rather than the algorithms. :citet:`Wen.Hall.Ma.ea.2025` retuned ten
optimizers for language-model pretraining, each with its own tuning budget,
and found that published speedups of about $2\times$ over AdamW shrink to
about $1.4\times$ at 100 million parameters and toward $1.1\times$ at 1.2
billion.

### Experiment Records

A comparison is only as good as its record. For every run, log the full
configuration, including the hyperparameters you consider fixed, the code
version, the random seed, and the one thing you changed. Change one factor
at a time to estimate its effect, and use a factorial design when the
interaction between factors is the question. Repeat the decisive runs with
new random seeds: a difference smaller than the spread between seeds is not
a result. Keep the runs that diverged, since the edge of a sweep where runs
fail marks the stability boundary, as the divergent grid point of
:numref:`sec_adam` did for SGD. And write the conclusion next to the curves
while the experiment is fresh. A directory of loss curves without sentences
becomes unreadable within weeks.

### The Chapter's Three Decisions

The methods of this chapter settle three decisions: the direction of an
update (the gradient, its sign, or an orthogonalized version of it), the
size of the step over time (warmup, decay, and the shape of the schedule),
and the handling of gradient noise (the batch size, momentum, and
averaging). Every row of :numref:`tab_practice_recipes` is one coordinated
choice of all three. When a run misbehaves, the first question is which
of the three decisions failed. This chapter has left open where the updates
run once a model no longer fits on one device, and how the optimizer state
is divided among many devices. That is the subject of
:numref:`chap_performance`.

## Summary

A large training run is configured once, guarded against rare failures, and
judged by its final weights. Recent pretraining reports disclose recurring
settings (AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 0.1,
clipping at 1, warmup and a later decay) that make a defensible start, and
with a small budget the peak learning rate is the setting to tune.
Global-norm clipping with a threshold above the typical gradient norm
fires only on the first few steps, where the gradients of the fresh
initialization are largest. Yet it turns a diverging `TinyLM` run at
$\eta = 1.2$ into a stable one, acting as an automatic warmup, and in long
runs it also catches the rare oversized gradients of heavy-tailed noise.
Spikes that get through are handled by rewinding to a checkpoint and
skipping the data seen around them. An exponential moving average of the
weights reduces the noise in the
evaluated parameters without changing the learning rate, and at a constant
rate it reaches the accuracy of a decayed schedule. A comparison of
optimizers is evidence only when each arm's nuisance hyperparameters were
retuned and the protocol is stated.

## Exercises

1. **The decay timescale.** The horizon of :numref:`subsec_wd-at-scale` is
   $\tau = B/(\eta\lambda D)$ epochs for batch size $B$ and dataset size
   $D$, both in tokens. The DeepSeek-V3 and OLMo 2 reports in
   :numref:`tab_practice_recipes` use sequences of 4,096 tokens and
   training sets of about 14.8 trillion and 4 trillion tokens.
    1. Compute $\tau$ for both rows at their peak learning rates, taking
       for DeepSeek-V3 its final batch size of 15,360 sequences. What
       fraction of each dataset does the averaging horizon span?
    1. Following :citet:`Bergsma.Dey.Gosal.ea.2025b`, what should happen to
       $\lambda$ if the batch size were doubled and $\tau$ is to be
       preserved, and what if instead the dataset were doubled?

1. [code] **Clipping regimes.** The demonstration in
   :numref:`subsec_practice-nan` clips at $\theta = 1$.
    1. Implement global-norm clipping :eqref:`eq_practice_clip` yourself as
       a function that rescales the gradients by
       $\min(1, \theta / (\|\mathbf{g}\|_2 + 10^{-6}))$ and returns the
       norm before clipping. Check on one minibatch that it returns the
       same norm as the library function and clipped gradients that agree
       with the library's to about $10^{-6}$.
    1. Sweep $\theta \in \{0.01, 0.1, 0.5, 1, 4, \infty\}$ on the divergent
       run, recording the final loss and the fraction of steps on which
       clipping changed the update. Classify each threshold as a guard
       (fires rarely), a brake (fires on most steps), or absent, and relate
       each class to whether the run diverges and to its final loss.
    1. Show that for SGD without momentum, clipping that fires on every
       step turns the update into normalized gradient descent with step
       size $\eta\theta$. The divergent run uses momentum, with velocity
       $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$, update
       $\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t$, and
       $\beta = 0.9$. Bound the length of one of its updates when clipping
       fires on every step. Explain why neither case is the same as
       training at a lower learning rate.
    1. In the demonstration, clipping fired only on the first steps. Remove
       it and instead warm the learning rate up linearly from 0 to
       $\eta = 1.2$, once over the first 20 steps and once over the first 100
       steps. Does each run survive? Compare the warmup lengths with the
       last step on which clipping fired, and explain what the two remedies
       have in common and how they differ.

    *Adapted from Stanford CS336,
    [Assignment 1](https://github.com/stanford-cs336/spring2024-assignment1-basics),
    Problem gradient_clipping.*

1. **Logit drift.** The z-loss of PaLM :cite:`chowdhery2022palm` adds
   $10^{-4} \log^2 Z$ to the cross-entropy, where $Z = \sum_j \exp(o_j)$
   is the softmax normalizer of the output logits $\mathbf{o}$.
    1. Show that adding the same constant $c$ to every logit leaves the
       cross-entropy unchanged, and compute how the shift changes $\log Z$
       and the z-loss.
    1. Compute the gradient of $10^{-4} \log^2 Z$ with respect to
       $\mathbf{o}$. In which direction does a gradient step move the
       logits when $\log Z$ is large and positive? Show that the gradient
       of the cross-entropy alone has no component along the all-ones
       vector, and explain what this implies for the overall level of the
       logits during training.

1. [code] **The averaging window.** Sweep the EMA decay
   $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ in the demonstration of
   :numref:`subsec_practice-weight-averaging` and plot the EMA's test
   accuracy per epoch for each value.
    1. Express each averaging window $1/(1-\alpha)$ in epochs. Derive the
       total weight that the EMA still places on the initial weights after
       the $T \approx 3{,}500$ steps of the 15-epoch run, and evaluate it
       for each $\alpha$.
    1. Use these numbers to explain the curves at the two ends of the
       sweep.
    1. Add the EMA with $\alpha = 0.999$ to a run whose learning rate
       decays to zero along the cosine schedule of :numref:`sec_scheduler`.
       Does the average still improve on the final live weights? Explain
       the answer with the noise ball.

1. [code] **A ten-run budget.** Using AdamW on `TinyLM`, reach a training
   loss of 1.1 in as few steps as you can, tuning only the learning rate,
   with a budget of ten runs. A run reaches the target at the first step
   at which the mean of its last 100 training losses falls below 1.1. Cap
   each run at 2,000 steps. Keep the log that
   :numref:`subsec_practice-tuning` asks for: for every run, the learning
   rate, the steps taken, the final loss, and the one-sentence conclusion
   you drew before launching the next run. Report the log, not just the
   winning configuration.

1. **Defaults against a small budget.**
   :citet:`Schmidt.Schneider.Hennig.2021` found that trying several
   optimizers at default settings works about as well as tuning a single
   one. Test this on `TinyLM` with runs that :numref:`sec_adam` and
   :numref:`sec_muon` already made under the same protocol. As the default
   runs, take SGD with momentum $0.9$ at $\eta = 0.1$, Adam at
   $\eta = 10^{-3}$, and the Muon/AdamW hybrid at $\eta = 10^{-3}$, the
   rate that RMS matching makes comparable to Adam's default.
    1. Compare the best of the three default runs with the best point of
       each of the four grids: SGD and Adam in :numref:`sec_adam`, AdamW
       and the Muon/AdamW hybrid in :numref:`sec_muon`. Which optimizers,
       tuned alone, beat the best default run, and by how much in final
       loss?
    1. Given a fixed budget of four runs on an unfamiliar problem, which
       strategy would you choose, and on what does the answer depend?

1. [code] **Batch size and the tuned rate.** ● Rebuild `data` with a batch
   size of 256 instead of 64. At both batch sizes, train AdamW without
   weight decay on `TinyLM` from the same initialization at a constant
   learning rate, sweeping
   $\eta \in \{0.5, 1, 2, 4, 8, 16, 32\} \times 10^{-3}$. Give both batch
   sizes the same number of training sequences: 2,000 steps at batch size
   64 and 500 steps at batch size 256.
    1. Where does the optimum move? Compare the shift with the two rules of
       :numref:`sec_batch_size`, the square-root rule $\eta \propto \sqrt{b}$
       and the linear rule $\eta \propto b$.
    1. What would you have concluded about AdamW at batch size 256 had you
       reused the batch-64 learning rate without retuning?

[Discussions](https://d2l.discourse.group/t/36767)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.12]{.kicker}

Optimization in Practice<br>
**configurations · gradient clipping · weight averaging · fair comparisons**
:::
:::

::: {.slide title="A Large Run Gets One Shot"}
[The problem]{.kicker}

- PaLM (540B parameters, 6,144 TPU chips): the loss spiked **about 20
  times**, with gradient clipping on.
- A four-point learning-rate sweep at that scale costs **four runs**.
- The final checkpoint is one draw from a noisy trajectory, and it ships.

::: {.d2l-note .warn}
Every run in this chapter so far could be swept, repeated, or restarted.
A large run cannot.
:::

Source: Chowdhery et al. (2022).
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read a disclosed configuration and spend a small tuning budget on it;
- clip the global gradient norm, tell a guard from a brake, and say
  why clipping protects Adam's state rather than its step;
- recover from a loss spike by rewinding and skipping data;
- average weights with an EMA and choose its window;
- design an optimizer comparison that survives scrutiny.

**Running example:** `TinyLM` with SGD at $\eta = 1.2$, just above the
point that diverged in the Adam deck; then a LeNet with batch normalization at a
constant rate.
:::

::: {.slide title="The Last Section Puts the Chapter's Tools on a Long Run"}
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
| Scaling Up | learning rates across widths |
| **Practice** | **clipping, averaging, fair comparisons** |
:::

::: {.col .narrow}
- **Uses:** `TinyLM`, momentum, Adam's moments, the noise ball,
  warmup and decay.
- **Skips:** attention-specific guards; distributed optimizer state.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Configurations]{.dtitle}

[what large runs disclose]{.dsub}
:::
:::

::: {.slide title="Four Reports Agree on Most Settings and Differ on the Peak Rate"}
[Configurations]{.kicker}

| run | optimizer | $\beta_1,\beta_2$ | peak LR | warmup | clip | wd |
|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B | AdamW | — | $8{\times}10^{-5}$ | 8k | — | — |
| DeepSeek-V3 | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| OLMo 2 7B | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| Kimi K2 | Muon + AdamW | — | $2{\times}10^{-4}$ | 500 | — | 0.1 |

- All four warm up, then decay: cosine, or constant first and decay late.
- A dash is missing evidence, not an implied default.
:::

::: {.slide title="With a Few Runs, Keep the Recurring Settings and Sweep the Peak Rate"}
[Configurations]{.kicker}

- **A handful of runs:** recurring settings; every run on the peak
  learning rate, which each report sets for itself.
- **Tens of runs:** add weight decay and schedule.
- **Coupled pairs together:** AdamW multiplies the weights by
  $1-\eta\lambda$ per step, so vary the product $\eta\lambda$ and move
  along constant products.

$\beta_2 = 0.95$: Adam's second moment reaches back $1/(1-\beta_2) = 20$
steps instead of 1,000.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Gradient Clipping]{.dtitle}

[a guard against rare oversized updates]{.dsub}
:::
:::

::: {.slide title="Global-Norm Clipping Keeps the Direction and Caps the Length"}
[Gradient clipping]{.kicker}

All gradients as one vector $\mathbf{g}$, threshold $\theta$:

$$\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

- $\|\mathbf{g}\|_2 = 5$, $\theta = 1$: multiply by 0.2.
- One norm over all layers: their relative sizes are kept.
- PyTorch: `clip_grad_norm_` between `backward` and `step`. Optax:
  `clip_by_global_norm` chained before the optimizer.
:::

::: {.slide title="At η = 1.2, Clipping Separates Divergence from Training"}
[Gradient clipping · TinyLM, SGD with momentum 0.9]{.kicker}

@!practice-a-nan-averted-3

- Unclipped: after a few steps the loss grows by several orders of
  magnitude within about 15 steps and overflows.
- Clipped, same initial weights: final loss near the best SGD run at a
  quarter of this rate.
:::

::: {.slide title="Clipping Fired Only on the First Few Steps"}
[Gradient clipping]{.kicker}

@!practice-a-nan-averted-2

- A handful of firings, all in the first twenty steps, where the fresh
  initialization has its largest gradients: an **automatic warmup**.
- Afterwards a **guard**, idle at $\eta = 1.2$; in long runs it also
  catches the heavy tail of the gradient noise.
- Fires on most steps? Then it is a **brake** that changes the update
  rule: tune $\eta$ and $\theta$ together.
:::

::: {.slide title="Check: Does Clipping Shrink an Adam Step?"}
[Check]{.kicker}

One gradient coordinate $g$ arrives that dwarfs all earlier ones.
Adam: $m \leftarrow \beta_1 m + (1-\beta_1) g$,
$v \leftarrow \beta_2 v + (1-\beta_2) g^2$, step $\eta\, m/\sqrt{v}$.
How long is the step, and what does clipping change?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $m/\sqrt{v} \approx (1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2$
for any $g$: the step is already bounded. Unclipped, $g^2$ inflates $v$,
which decays with a time constant of $1/(1-\beta_2) = 1{,}000$ steps, and
silences the coordinate. Clipping protects the state.
:::
:::
:::

::: {.slide title="When a Spike Gets Through, Rewind and Skip"}
[Gradient clipping]{.kicker}

- PaLM: restart about **100 steps** before the spike, skip **200–500
  minibatches**; the spike does not recur.
- The same minibatches from another checkpoint: no spike. Data and
  state together cause it.
- A guard on the loss: **z-loss** $10^{-4}\log^2 Z$ keeps the softmax
  normalizer $\log Z$ near 0. PaLM used it throughout; the spikes came
  anyway.
- A checkpoint must hold model, optimizer state, step count, and data
  position.

Source: Chowdhery et al. (2022).
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Weight Averaging]{.dtitle}

[less noise without a smaller step]{.dsub}
:::
:::

::: {.slide title="An EMA of the Weights Averages Over the Noise Ball"}
[Weight averaging]{.kicker}

$$\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1-\alpha)\, \mathbf{x}_t$$

- At a constant rate, SGD's iterates $\mathbf{x}_t$ keep fluctuating
  around a minimum; weakly correlated fluctuations cancel in the mean.
- Window $1/(1-\alpha)$: $\alpha = 0.999$ gives 1,000 steps, about four
  epochs of 235 steps.
- Training never sees $\bar{\mathbf{x}}_t$. Cost: one copy of the
  parameters.
:::

::: {.slide title="At a Constant Rate, the EMA Is Steadier Than the Live Weights"}
[Weight averaging · LeNet-BN, Fashion-MNIST, η = 0.3]{.kicker}

@!practice-weight-averaging-3

- Early lag: after one epoch the initial weights still carry
  $0.999^{235} \approx 0.79$.
- Late: the EMA runs along the top of the live curve, ending near 91%.
:::

::: {.slide title="Decay Changes the Trajectory; Averaging Changes What We Evaluate"}
[Weight averaging]{.kicker}

- Constant rate plus EMA reaches the accuracy of a decayed schedule.
- Decay shrinks the noise injected into later steps; averaging leaves
  the trajectory alone and removes noise from the evaluated weights.
- After full decay the iterates barely move: little left to average.
- Averaged saved checkpoints: recompute the batch-norm statistics.
- Llama 3 released an average of checkpoints from its final, annealed
  phase.
:::

::: {.slide title="Check: When Does an EMA Not Help?"}
[Check]{.kicker}

Name two situations in which the EMA of the weights is no better than
the last iterate.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) After the schedule has decayed $\eta$ to nearly zero:
the iterates barely move, so their average is the last one. (2) While
the weights still improve faster than the window: the average lags
behind, as in the first epochs above.
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Fair Comparisons]{.dtitle}

[the protocol is part of the result]{.dsub}
:::
:::

::: {.slide title="A Comparison Must Retune Its Nuisance Hyperparameters in Every Arm"}
[Fair comparisons]{.kicker}

- **Scientific:** what the question is about (the optimizer).
- **Nuisance:** retuned for each scientific value (the learning rate).
- **Fixed:** held constant, and each one limits the claim.
- The optimizer races of the Adam and Muon sections: a four- or
  five-point learning-rate grid per optimizer, spaced by about 3×; steps,
  batch, model fixed.

::: {.d2l-note .warn}
One shared learning rate favors whichever method happens to prefer it.
:::
:::

::: {.slide title="Untuned Comparisons Measure Effort, Not Algorithms"}
[Fair comparisons]{.kicker}

- 15 optimizers, many problems: several at defaults ≈ one heavily
  tuned (Schmidt et al., 2021).
- 10 optimizers, matched tuning: claimed 2× over AdamW shrinks to about
  1.4× at 100M parameters, toward 1.1× at 1.2B (Wen et al., 2025).
- Log configuration, code version, seed, the one change; keep diverged
  runs; repeat decisive runs with new seeds.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Configure:** recurring settings (AdamW, 0.9/0.95, wd 0.1, clip 1,
  warmup then decay); a small budget goes to the peak rate.
- **Guard:** clipping at 1 fired only on the first few steps, acting as
  a warmup, and saved `TinyLM` at $\eta = 1.2$; spikes that get through:
  rewind and skip.
- **Average:** an EMA ($\alpha = 0.999$) at a constant rate reaches a
  decayed schedule's accuracy.
- **Compare:** retune nuisances per arm; a gap below the seed spread is
  no result.

**Next chapter:** attention, the mechanism inside `TinyLM`.
:::
