# Practice
:label:`sec_practice`

The preceding sections isolated individual optimization choices. A complete
training run must combine them and remain stable over a long computation.
This section summarizes configurations reported for recent large-scale runs,
implements gradient clipping and weight averaging, and describes a protocol
for tuning and comparing optimizers.

```{.python .input #practice}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import copy
import math
import torch
from torch import nn
```

```{.python .input #practice}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## Configurations Reported for Large-Scale Training

A target-scale language-model run is too expensive for an exhaustive
hyperparameter sweep, so its hyperparameters are assembled from smaller
proxies
(:numref:`sec_scaling`), from precedent, and from the recipes of earlier
runs. Some of what the teams settled on is public. :numref:`tab_practice_recipes`
collects the optimizer configurations disclosed by four prominent
pretraining reports from 2024--2026. The table documents these cases rather than defining a general recipe: each entry is what a team reported shipping, usually with no
ablation attached, and a dash means the report does not say.

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

These four reports are case studies, not a representative survey. Three
report AdamW and one reports a Muon/AdamW split. Two disclose
$(\beta_1,\beta_2)=(0.9,0.95)$, weight decay 0.1, and global-norm clipping at
1; the dashes in the other rows indicate missing evidence rather than implied
defaults. All four report warmup followed by either cosine decay or a stable
phase and later decay. Batch ramping appears in two rows and fixed batches in
the other two. The table therefore shows several repeated choices and one
production use of Muon, while also showing that the published fields are
insufficient to reproduce any run in full.

## Gradient Clipping

The next section examines the clipping column. Gradient clipping
entered this book in :numref:`sec_rnn-scratch` as the fix for exploding RNN
gradients; the reports show that the method is also used beyond recurrent
architectures. Every run above that discloses a threshold clips, and every disclosed
threshold is 1. Global-norm clipping treats all parameters as one long
vector: with $\mathbf{g}$ the concatenation of every parameter's gradient
and $\theta$ the threshold,

$$
\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}.
$$
:eqlabel:`eq_practice_clip`

A gradient shorter than $\theta$ passes untouched; a longer one keeps its
direction and loses its length. The implementation is a few lines in either
framework.

:begin_tab:`pytorch`
`clip_grad_norm_` rescales the gradients in place, between `backward` and
`step`, and returns the norm it measured before clipping. We wrap an
existing optimizer so the harness of :numref:`sec_adam` can use it
unchanged, and we keep the returned norms, which the demonstration below
will want.
:end_tab:

:begin_tab:`jax`
In Optax, clipping is a gradient transformation like any other, so it
composes with `optax.chain`: the clipped transformation drops into
`nnx.Optimizer`, and the harness of :numref:`sec_adam` needs no change at
all.
:end_tab:

```{.python .input #practice-gradient-clipping-1}
%%tab pytorch
class Clipped:
    """Clip the global gradient norm before every optimizer step."""
    def __init__(self, optimizer, params, max_norm=1.0):
        self.optimizer = optimizer
        self.params = list(params)
        self.max_norm = max_norm
        self.norms = []

    def step(self):
        norm = nn.utils.clip_grad_norm_(self.params, self.max_norm)
        self.norms.append(float(norm))
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

:numref:`sec_adam` swept SGD with momentum on `TinyLM` and found a knife's
edge: the best learning rate sat one grid point below one that returned
NaN. This divergent grid point provides a direct test of whether clipping can
prevent the observed numerical failure. We rerun it with and without
clipping, changing nothing else.

```{.python .input #practice-a-nan-averted-1}
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)

def final_loss(losses, k=100):
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')
```

```{.python .input #practice-a-nan-averted-2}
%%tab pytorch
curves = {}
for clip in (False, True):
    torch.manual_seed(0)
    model = d2l.TinyLM(len(data.vocab))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    if clip:
        optimizer = Clipped(optimizer, model.parameters())
    label = 'clipped at 1.0' if clip else 'unclipped'
    curves[label] = d2l.train_lm(model, data, optimizer, num_steps=2000)
    print(f'{label}: final loss {final_loss(curves[label]):.3f}')
fired = sum(n > optimizer.max_norm for n in optimizer.norms)
print(f'clipping changed the update on {fired} of {len(optimizer.norms)} '
      f'steps; median gradient norm '
      f'{sorted(optimizer.norms)[len(optimizer.norms) // 2]:.2f}')
```

```{.python .input #practice-a-nan-averted-2}
%%tab jax
curves = {}
for clip in (False, True):
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
    tx = optax.sgd(0.3, momentum=0.9)
    if clip:
        tx = clipped(tx)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    label = 'clipped at 1.0' if clip else 'unclipped'
    curves[label] = d2l.train_lm(model, data, optimizer, num_steps=2000)
    print(f'{label}: final loss {final_loss(curves[label]):.3f}')
```

The instability occurs early, so we plot the first 150 steps on a log scale.

```{.python .input #practice-a-nan-averted-3}
d2l.plot(list(range(150)), [curves[k][:150] for k in curves], 'step',
         'training loss', legend=list(curves), yscale='log')
```

:begin_tab:`pytorch`
The unclipped run makes normal progress for a few steps, then feeds back on
itself: an oversized step lands the iterate somewhere steeper, the gradient
there is larger, momentum compounds it, and within two dozen steps the loss
has climbed six orders of magnitude and overflows. The clipped run, at the
identical learning rate, trains to a final loss at or slightly below the
tuned unclipped run of :numref:`sec_adam`. The median gradient norm was far
below the threshold, and
clipping changed the update on six steps out of two thousand. In these paired runs, those six interventions separate the divergent
trajectory from the stable one.
:end_tab:

:begin_tab:`jax`
The unclipped run makes progress for a moment, then feeds back on itself:
an oversized step lands the iterate somewhere steeper, the gradient there
is larger, momentum compounds it, and the loss climbs through four orders
of magnitude before overflowing within the first couple hundred steps. The
clipped run, at the identical learning rate, trains to a final loss in the
range of the tuned runs of :numref:`sec_adam`. The Optax combinator does
not report how often it fired; the instrumented PyTorch tab counts six
interventions in two thousand steps, and the picture here is the same.
:end_tab:

This count illustrates clipping used as a guard: most updates remain
unchanged, while a small number of unusually large gradients are rescaled. The tail is not an accident of our small model: gradient noise
in language models is measurably heavy-tailed, and under such noise clipped
SGD provably converges where plain SGD can fail
:cite:`Zhang.Karimireddy.Veit.ea.2020`. The threshold can be set above the typical norms observed in a stable run.
If clipping activates on most steps, it continuously normalizes or rescales
the update rather than serving only as an outlier guard; the learning rate
and threshold should then be tuned together
:cite:`Godbole.Dahl.Gilmer.ea.2023`. For the Adam family the arithmetic
differs but the conclusion holds. For a persistent gradient, Adam's normalized update in a coordinate is on
the order of $\eta$ (:numref:`sec_adam`), transiently
up to a few times that: $|\hat{m}/\sqrt{\hat{v}}|$ can briefly reach
$(1 - \beta_1)/\sqrt{1 - \beta_2} \approx 3$ at the defaults. So clipping
is no substitute for lowering a too-large Adam learning rate, and in our
runs it was not. What it
still provides is protection for the estimates: one enormous gradient otherwise enters both moment estimates, and its
contribution to the slower second-moment average decays over a timescale on
the order of $1/(1-\beta_2)$ steps.

### Additional Stability Methods

At trillion-token scale, clipping is one of several stability methods, most of them
aimed at the places where transformer blow-ups concentrate: the attention
logits and the output softmax. PaLM's training added a *z-loss*, a small
penalty on $\log^2 Z$ of the softmax normalizer, to keep the output logits
from drifting large :cite:`chowdhery2022palm`. *QK-norm* normalizes queries
and keys immediately before their dot product, so attention logits cannot
grow with the norms of what feeds them
:cite:`Henry.Dachapally.Pawar.ea.2020`; OLMo 2 adopted it as part of the
stability overhaul that its report documents :cite:`OLMo.2025`. MuonClip's *QK-clip* rescales the query and key projections whenever the
largest attention logit crosses a cap; the Kimi K2 report includes this
method in a 15.5-trillion-token run with no reported loss spike (:numref:`sec_muon`,
:cite:`Kimi.Team.2025`). And when prevention fails, the practice is
unglamorous. Facing about twenty spikes in a run, the PaLM team rewound to
a checkpoint a few hundred steps earlier and skipped the offending batches,
after establishing that the same batches caused no spike when replayed from
a different state: spikes came from state and data together, not from bad
data alone :cite:`chowdhery2022palm`. The OPT team published its training
logbook along with the model; it records two months of restarts, mid-flight
learning-rate cuts, and hardware failures in a way no polished paper does,
and it provides an unusually detailed public record of operational
interventions during a large run :cite:`zhang2022opt`.

## Weight Averaging
:label:`subsec_practice-weight-averaging`

The chapter's third recurring decision is living with noise, and it has one
more tool, one that is nearly free. A constant-rate iterate rattles around its
noise ball (:numref:`sec_sgd`), and a schedule quenches the rattling by
decaying the rate (:numref:`sec_scheduler`). Averaging can reduce this variability without changing the learning rate:
when iterate fluctuations are weakly correlated around a region, some of
them cancel in the mean.
Stochastic weight averaging made this a standard trick, averaging
checkpoints from the tail of training and evaluating the average
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`. You have met the running form
twice: :numref:`sec_training_recipes` put an exponential moving average of
the weights into the standard recipe, and :numref:`sec_parameters` showed
the state needed to maintain one. It is
$\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1 - \alpha)\, \mathbf{x}_t$:
the same leaky average this chapter has applied to gradients (momentum) and
to squared gradients (Adam), now applied to the weights themselves, purely
for evaluation. The training run never sees it. What is new here is the
noise-ball reading of *why* it helps.

We test it on the schedule testbed of :numref:`sec_scheduler`, reproduced
verbatim, for its constant-learning-rate baseline, whose test accuracy plateaued and
varied across epochs. Alongside
the live weights we maintain an EMA with $\alpha = 0.999$, an averaging
window of about a thousand steps, roughly four epochs here.

```{.python .input #practice-weight-averaging-1}
%%tab pytorch
def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
        nn.Linear(84, 10))
    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    return model

device = d2l.try_gpu()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
loss = nn.CrossEntropyLoss()
```

```{.python .input #practice-weight-averaging-1}
%%tab jax
xavier = nnx.initializers.xavier_uniform()

class Net(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.conv1 = nnx.Conv(1, 6, kernel_size=(5, 5), padding='same',
                              kernel_init=xavier, rngs=rngs)
        self.bn1 = nnx.BatchNorm(6, rngs=rngs)
        self.conv2 = nnx.Conv(6, 16, kernel_size=(5, 5), padding='valid',
                              kernel_init=xavier, rngs=rngs)
        self.bn2 = nnx.BatchNorm(16, rngs=rngs)
        self.fc1 = nnx.Linear(16 * 5 * 5, 120, kernel_init=xavier, rngs=rngs)
        self.bn3 = nnx.BatchNorm(120, rngs=rngs)
        self.fc2 = nnx.Linear(120, 84, kernel_init=xavier, rngs=rngs)
        self.bn4 = nnx.BatchNorm(84, rngs=rngs)
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

:begin_tab:`pytorch`
The EMA update walks the two `state_dict`s in parallel. Floating-point
entries are averaged, which includes the BatchNorm running statistics, so
the averaged model stays approximately consistent — an EMA of running
statistics is not the exact statistic of the EMA weights; a more accurate
alternative is to recompute them for the averaged weights, as the
checkpoint-averaging note below prescribes. Integer buffers are copied
through.
:end_tab:

:begin_tab:`jax`
The EMA update is one `tree.map` over the parameters and BatchNorm
statistics of the two models. Averaging the running statistics keeps the
averaged model approximately consistent — an EMA of running statistics is
not the exact statistic of the EMA weights; a more accurate alternative is to
recompute them for the averaged weights, as the checkpoint-averaging note
below prescribes.
:end_tab:

```{.python .input #practice-weight-averaging-2}
%%tab pytorch
def ema_update(ema, model, decay=0.999):
    with torch.no_grad():
        for e, p in zip(ema.state_dict().values(),
                        model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(p, alpha=1 - decay)
            else:
                e.copy_(p)

lr, num_epochs = 0.3, 15
torch.manual_seed(0)
net = net_fn().to(device)
ema = copy.deepcopy(net)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
live_acc, avg_acc = [], []
for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        trainer.zero_grad()
        loss(net(X), y).backward()
        trainer.step()
        ema_update(ema, net)
    live_acc.append(d2l.evaluate_accuracy_gpu(net, test_iter))
    avg_acc.append(d2l.evaluate_accuracy_gpu(ema, test_iter))
print(f'final test accuracy: live {live_acc[-1]:.3f}, '
      f'EMA {avg_acc[-1]:.3f}')
```

```{.python .input #practice-weight-averaging-2}
%%tab jax
lr, num_epochs = 0.3, 15
model = Net(rngs=nnx.Rngs(0))
ema = nnx.clone(model)
optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, X, y):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X), y).mean()
    l, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return l

@nnx.jit
def ema_step(ema, model, decay=0.999):
    kinds = (nnx.Param, nnx.BatchStat)  # skip any non-float state
    new = jax.tree.map(lambda e, p: decay * e + (1 - decay) * p,
                       nnx.state(ema, kinds), nnx.state(model, kinds))
    nnx.update(ema, new)

@nnx.jit
def count_correct(model, X, y):
    return (model(X).argmax(axis=1) == y).sum()

def accuracy(model, data_iter):
    model.eval()  # use running averages in the BatchNorm layers
    correct = total = 0
    for X, y in data_iter:
        correct += int(count_correct(model, jnp.asarray(X), jnp.asarray(y)))
        total += y.shape[0]
    return correct / total

live_acc, avg_acc = [], []
for epoch in range(num_epochs):
    model.train()
    for X, y in train_iter:
        train_step(model, optimizer, jnp.asarray(X), jnp.asarray(y))
        ema_step(ema, model)
    live_acc.append(accuracy(model, test_iter))
    avg_acc.append(accuracy(ema, test_iter))
print(f'final test accuracy: live {live_acc[-1]:.3f}, '
      f'EMA {avg_acc[-1]:.3f}')
```

```{.python .input #practice-weight-averaging-3}
d2l.plot(list(range(1, num_epochs + 1)), [live_acc, avg_acc], 'epoch',
         'test accuracy', legend=['live weights', 'EMA of weights'])
```

Two curves, two lessons. First, the EMA needs its window to fill before it
is worth anything: for the first several epochs it trails badly, because
the average still carries heavy weight on stale, near-initialization
iterates. This is lag, not the zero-initialization deficit that Adam's bias
correction removes (:numref:`sec_adam`): the average starts at the initial
weights, so its coefficients already sum to one and there is nothing to
correct — the early average retains substantial weight on old iterates. That
lag is why
practical EMAs either warm up the decay or start averaging late. Second, after the window fills, the EMA curve is higher and less variable
than the live-weight curve in this run. It ends roughly 0.5--1 percentage
point higher, while reducing sensitivity to the particular stopping epoch.
The averaged model
at a constant rate lands in the range the decayed schedules of
:numref:`sec_scheduler` reached, consistent with the schedule-free discussion in
that section: decay reduces the noise injected into future iterates, whereas
averaging reduces variation in the evaluated parameters. That also predicts a limit. Add the same
EMA to a run whose schedule already decayed well, and on a model this size
the remaining nudge is too small to call from a single run. The technique is most useful when a comparable decayed endpoint is expensive
to reach or no decay is planned. Its principal memory cost is one additional
copy of the parameters, as discussed in :numref:`sec_adamw`. One footnote for checkpoint
averaging: if you average a few saved checkpoints instead of keeping a
running EMA, the BatchNorm statistics belong to none of the averaged
weights, so recompute them with a pass over the training data before
trusting the result.

### Averaging at Scale

Large-scale pipelines use several forms of weight averaging. Averaging the latest $k$
checkpoints of an LLM run, uniformly rather than exponentially, can produce a loss comparable to a later
single checkpoint :cite:`Kaddour.2022`, and the top row of
:numref:`tab_practice_recipes` does it in production: the Llama 3 model
that shipped is the average of checkpoints from its final annealing phase
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Model soups push the idea past a single run,
averaging separately fine-tuned models into one, with accuracy gains at no
inference cost :cite:`Wortsman.Ilharco.Gadre.ea.2022`. Diffusion-model training commonly evaluates an EMA of the weights
(:numref:`chap_diffusion`) and sample quality can depend strongly on the averaging window;
:citet:`Karras.Aittala.Lehtinen.ea.2024` developed a method to reconstruct
multiple EMA windows after training so that the window can be tuned. The
difference from the classifier example may partly reflect the repeated
application of a denoiser during sampling, but this comparison alone does
not establish the mechanism.

## How to Tune
:label:`subsec_practice-tuning`

Every comparison in this chapter depends on a tuning protocol, which must be
reported as part of the method. Google's Tuning Playbook
:cite:`Godbole.Dahl.Gilmer.ea.2023` provides a useful vocabulary.
In any experiment, split the hyperparameters into three classes:
*scientific* hyperparameters, the ones your question is about; *nuisance*
hyperparameters, which must be re-optimized for every setting of the
scientific ones before the comparison means anything; and *fixed*
hyperparameters, held constant and acknowledged as limits on the claim. The
playbook's rule is that a comparison is a statement about scientific
hyperparameters only after the nuisances have been re-tuned per arm, and
its most common instance is also the most common failure in published
comparisons: the learning rate is usually a nuisance, and one shared learning rate can
favor a method whose preferred scale happens to match it.

This vocabulary names what the chapter has been doing since
:numref:`sec_adam`. In each comparison, the optimizer was the scientific
hyperparameter; the learning rate was the nuisance, re-tuned per contestant
on a four-point grid spaced by factors of about three; steps, batch size,
initialization, and the absence of a schedule were fixed and stated, which
is why every conclusion was phrased as conditional on that protocol. At larger scale, the same design includes more nuisance dimensions (peak
rate, decay horizon, warmup, and weight decay), may replace a coarse grid
with quasi-random search, and separates broad exploration from a narrower
final sweep. Conclusions need to be reassessed when these nuisance
parameters are not retuned. :citet:`Schmidt.Schneider.Hennig.2021`
benchmarked fifteen optimizers across many problems and found that trying
several optimizers at default settings works about as well as extensively
tuning a single one. Read it as consolation or as warning: defaults encode
a lot of accumulated tuning, and an untuned comparison measures effort, not
algorithms. :numref:`sec_muon`'s deflation of headline speedups is the same
lesson with larger budgets.

Budgets shape the rest. With a handful of runs, take the consensus column
of :numref:`tab_practice_recipes` as given and spend every run on the peak
learning rate. With tens of runs, add weight decay and the schedule, and
sweep jointly only what is genuinely coupled: $\eta$ and $\lambda$ act
through their product (:numref:`sec_adamw`), so a joint grid should cover both their product and deviations from a
constant-product ridge. Past that, you are running a study, and
the playbook is the reference. On the models of this chapter a run cost
seconds to minutes, which is why we could afford the middle tier
everywhere; one purpose of :numref:`sec_scaling` was to keep that
affordability relevant as models grow.

Finally, maintain a reproducible experiment log. For every run, record the full configuration, including the hyperparameters you
consider fixed, the code version, the seed, and the one thing you changed.
Vary one factor at a time when estimating a marginal effect, and use
factorial designs when interactions are the question. Retain diverged runs: the NaN edge of a
sweep marks the stability boundary, and :numref:`sec_adam` read its NaN
column as data, not as failure. Write the conclusion next to the curves
while you still believe it, because a directory of loss curves with no
sentences attached goes stale within weeks. Assignments in the CS336 mold
now grade the experiment log alongside the final loss, emphasizing that the experimental record is part of the result.

## Topics Covered Elsewhere

Three method families were left out on purpose. Sharpness-aware
minimization takes an inner ascent step before each descent step to seek
the flat minima whose connection to generalization
:numref:`sec_generalization_deep` discussed; it doubles the gradient cost,
and the cited results establish improvements mainly in vision tasks rather
than large-scale pretraining :cite:`Foret.Kleiner.Mobahi.ea.2021`.
Variance-reduction methods of the SVRG family have an elegant theory for
finite sums but have not produced consistent improvements on deep networks;
:numref:`sec_mdl-variance-reduction` presents the theory
and a careful post-mortem. LARS and LAMB are layerwise-adaptive methods developed for large-batch
training. In the cited benchmark, after nuisance hyperparameters were
retuned, standard momentum and AdamW matched them at the studied batch
sizes :cite:`Nado.Gilmer.Shallue.ea.2021`.

The remaining questions concern placement rather than optimization. This
chapter analyzed optimizer state (:numref:`sec_adamw`) on a single device.
Spreading gradients and state across a data-parallel group, ZeRO-style
sharding, and the overlap of communication with computation belong to
:numref:`chap_performance` and the training-systems material of
:numref:`sec_training_systems`. The updates computed there are the ones
derived here; the systems side decides where the bytes live and when they
move.

## Summary

The four reports document repeated choices without defining a complete or
representative recipe. Among disclosed fields, AdamW, $(0.9,0.95)$ moment
coefficients, weight decay 0.1, clipping at 1, warmup, and later decay recur;
one report uses a Muon/AdamW split. Clipping can guard against rare,
heavy-tailed gradients, but frequent activation changes the update rule and
must be tuned with the learning rate. Large runs also report z-loss,
QK-norm, QK-clip, checkpoint recovery, and batch skipping. On our
single-seed testbed, weight averaging raises and stabilizes endpoint
accuracy; large language-model and diffusion pipelines also use checkpoint
or exponential averaging. Fair optimizer comparisons identify scientific,
nuisance, and fixed hyperparameters, retune nuisance parameters per arm, and
retain a reproducible experiment log.

The methods in this chapter address three related decisions: how gradients
are transformed into an update direction; how the step scale changes during
training; and how sampling noise is managed through batching, momentum, and
averaging. The
configuration table gives coordinated settings of all three. When a run
misbehaves, first determine which decision is responsible. Optimizers change,
but this decomposition has remained useful for decades.

## Exercises

1. [code] **A ten-run budget.** Using AdamW on `TinyLM`, reach a training
   loss of 1.1 in as few steps as you can, tuning only the learning rate,
   with a budget of ten runs. Keep the log that
   :numref:`subsec_practice-tuning` asks for: for every run, the learning
   rate, the steps taken, the final loss, and the one-sentence conclusion
   you drew before launching the next run. Report the log, not just the
   winning configuration.
1. [code] **Batch size and the tuned rate.** Rebuild `data` with a batch
   size of 256 instead of 64 and run the four-point learning-rate protocol
   of :numref:`sec_adam` for AdamW at both batch sizes.
    1. Where does the optimum move, and how does the shift compare with the
       square-root and linear rules of :eqref:`eq_lr-rules`?
    1. What would you have concluded about AdamW at batch size 256 had you
       reused the batch-64 learning rate without retuning?
1. [code] **Clipping regimes.** The demonstration in
   :numref:`subsec_practice-nan` clips at $\theta = 1$.
    1. Implement global-norm clipping :eqref:`eq_practice_clip` yourself as
       a function that rescales the gradients by
       $\min(1, \theta / (\|\mathbf{g}\|_2 + 10^{-6}))$ and also returns
       the norm before clipping. Check that at $\theta = 1$ it reproduces
       the final loss of the clipped run.
    1. Sweep $\theta \in \{0.01, 0.1, 0.5, 1, 4, \infty\}$ on the divergent
       run, recording the final loss and the fraction of steps on which
       clipping changed the update. Identify the three regimes: guard
       (fires rarely), brake (fires on most steps), and absent.
    1. Show that when clipping fires on every step, the update is
       normalized gradient descent with step size $\eta\theta$, and explain
       why this differs from training at a lower learning rate.

    *Adapted from Stanford CS336,
    [Assignment 1](https://github.com/stanford-cs336/spring2024-assignment1-basics),
    Problem gradient_clipping.*
1. **The decay timescale.** The horizon of :numref:`subsec_wd-at-scale` is
   $\tau = B/(\eta\lambda D)$ epochs for batch size $B$ and dataset size
   $D$, both in tokens. The DeepSeek-V3 and OLMo 2 reports in
   :numref:`tab_practice_recipes` use sequences of 4,096 tokens and
   training sets of about 14.8 trillion and 4 trillion tokens.
    1. Compute $\tau$ for both rows at their peak learning rates. What
       fraction of each dataset does the averaging horizon span?
    1. Following :citet:`Bergsma.Dey.Gosal.ea.2025b`, what should happen to
       $\lambda$ if the batch size were doubled and $\tau$ is to be
       preserved, and what if instead the dataset were doubled?
1. [code] **The averaging window.** Sweep the EMA decay
   $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ in the demonstration of
   :numref:`subsec_practice-weight-averaging` and plot the EMA's test
   accuracy per epoch for each value. Relate the averaging window
   $1/(1-\alpha)$ to the behavior at both ends of the sweep within the
   15-epoch budget of about 3,500 steps.
1. [code] **Defaults against a small budget.** Test the finding of
   :citet:`Schmidt.Schneider.Hennig.2021` on this chapter's testbed: run
   SGD with momentum $0.9$ at $\eta = 0.1$, Adam at $\eta = 10^{-3}$, and
   AdamW at $\eta = 10^{-3}$, with library defaults for every other
   hyperparameter, on `TinyLM` for 2,000 steps, and compare the best of the
   three against the grid-tuned Adam of :numref:`sec_adam`. Given a fixed
   budget of four runs, which strategy would you choose on this problem,
   and on what does the answer depend?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.12]{.kicker}

Optimization in practice<br>
**reported configurations · clipping · weight averaging · tuning**
:::
:::

::: {.slide title="Reported large-scale configurations"}
[Configurations reported by recent training runs]{.kicker}

| run | optimizer | $\beta_1,\beta_2$ | peak LR, schedule | warmup | clip | wd |
|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B | AdamW | — | $8{\times}10^{-5}$, cosine | 8k | — | — |
| DeepSeek-V3 | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, WSD-like | 2k | 1.0 | 0.1 |
| OLMo 2 7B | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, cosine | 2k | 1.0 | 0.1* |
| Kimi K2 | MuonClip | — | $2{\times}10^{-4}$, WSD | 500 | — | 0.1 |

- Three reports use AdamW; Kimi K2 reports a **Muon split** (§9.9).
- Two reports disclose (0.9, 0.95), weight decay 0.1, and clipping at 1.0.
- All report warmup; later schedules and batch handling differ.
- Dashes mark undisclosed fields, not implied defaults.
:::

::: {.slide title="Gradient clipping in three lines"}
Global norm, all parameters as one vector:

$$\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

The transformation preserves direction and caps the global norm. Introduced
for RNNs in ch. 8, it also appears in the two reports that disclose clipping;
both use threshold 1.

@practice-gradient-clipping-1
:::

::: {.slide title="Preventing numerical overflow with clipping"}
§9.6's knife edge: SGD's best lr sat one grid point below a NaN. Rerun the
divergent point, with and without the guard:

@!practice-a-nan-averted-3


- Unclipped: a large step reaches a region with larger gradients, momentum
  compounds the growth, and the run overflows.
- Clipped at the same learning rate: final loss is in the tuned runs' range.
:::

::: {.slide title="Frequency of clipping"}
The instrumented run: median gradient norm ~0.3, threshold 1.0 —
**clipping changed the update on 6 of 2,000 steps**.

- Language-model gradient noise is heavy-tailed
  (Zhang et al., 2020); the guard exists for the tail.
- If clipping fires on most steps, it continuously changes the update rule;
  tune $\eta$ and $\theta$ together.
- For a persistent gradient, Adam's coordinate update is on the order of
  $\eta$ (with transients up to about 3×), so
  clipping is no substitute for lowering a too-large $\eta$ — it guards
  $\mathbf{m}, \mathbf{v}$ from one huge gradient lingering
  $1/(1-\beta_2)$ steps.
:::

::: {.slide title="Additional stability methods"}
Clipping is one item. The rest aims at attention logits and the softmax:

- **z-loss** (PaLM): penalize $\log^2 Z$ of the softmax normalizer.
- **QK-norm** (OLMo 2): normalize $q, k$ right before their dot product.
- **QK-clip** (MuonClip): cap the largest attention logit — 15.5T tokens,
  zero spikes (§9.9).


When prevention failed, PaLM rewound to an earlier checkpoint and skipped
the implicated batches. Replaying those batches from a different state did
not reproduce the spike, indicating an interaction between state and data.
The OPT logbook documents two months of restarts and learning-rate changes.
:::

::: {.slide title="Weight averaging"}
Weight averaging reduces endpoint variability **without changing the
learning rate**:
$\bar{\mathbf{x}}_t = \alpha \bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$,
the chapter's leaky average, now on the weights (SWA; Izmailov et al.,
2018).

@!practice-weight-averaging-3

- Window must fill first — the early gap is lag on stale iterates, not a
  bias to correct; warm up the decay or start late.
- Afterward, this EMA run is modestly above the live-weight curve and less
  sensitive to the stopping epoch.
:::

::: {.slide title="Averaging: where it matters"}
- Constant rate plus EMA reaches the range of the decayed schedules in
  §9.8, though decay changes future iterates whereas averaging changes the
  evaluated parameters. With prior decay, one run shows no clear added gain.
- LLMs: checkpoint averaging (LAWA); **Llama 3 shipped an average** of its
  annealing checkpoints. Model soups: average fine-tuned models.
- Diffusion models (:numref:`chap_diffusion`) commonly use EMA; Karras et al. (2024) reconstruct multiple
  EMA windows after training to tune the window.
:::

::: {.slide title="How to tune"}
[The Tuning Playbook's vocabulary]{.kicker}

- **Scientific** hyperparameters define the question. **Nuisance**
  hyperparameters are retuned per arm. **Fixed** settings limit the claim.
- This chapter, named: optimizer scientific, lr nuisance (four-point grid
  per contestant), all else fixed and stated.
- Schmidt et al. (2021): several optimizers at defaults ≈ one optimizer
  heavily tuned. Untuned comparisons confound optimizer choice with tuning effort.


With few runs, prioritize the peak learning rate; with tens, add weight
decay and schedule parameters. Log the configuration, seed, changes, and
diverged runs so the comparison is reproducible.
:::

::: {.slide title="Topics covered elsewhere"}
- **SAM**: flat minima at 2× gradient cost — wins concentrate in vision
  and fine-tuning.
- **Variance reduction**: strong finite-sum theory, but no consistent
  improvement reported for deep networks → ch. 25.
- **LARS/LAMB**: in the cited benchmark, retuned momentum/AdamW matched
  them at the studied batch sizes.
- **Systems**: sharding state, data parallelism, overlap → ch. 11 and the
  training-systems appendix.
:::

::: {.slide title="Recap: three decisions"}
- **Direction**: a norm — gradient, sign, or orthogonalized (§9.2, §9.6,
  §9.9).
- **Step scale over time**: warmup, cosine, and WSD (§9.8); clipping is
  a separate guard on unusually large gradients.
- **Noise**: batch (§9.4, §9.10), momentum (§9.5), averaging (here).


Each row of the table is one reported coordination of these choices. When a run
misbehaves, ask which decision is failing. Optimizers change; the
decomposition has been stable for decades.
:::
