# Practice
:label:`sec_practice`

This chapter introduced its methods one at a time, each with a clean
experiment attached. A real training run gets no such courtesy: it must make
all of the chapter's decisions at once, commit to them for weeks of compute,
and survive whatever the data, the hardware, and the loss surface do in the
meantime. This closing section is about that craft. We read the recipes that
current large-scale runs disclose; implement the one standard ingredient the
chapter has not yet built, gradient clipping; add the cheapest trick in the
practitioner's kit, weight averaging; and end with the discipline that makes
tuning and optimizer comparisons meaningful.

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

## The Recipe, as Disclosed

A frontier language model is trained once, so its hyperparameters are not
found by sweeping at full scale; they are assembled from smaller proxies
(:numref:`sec_scaling`), from precedent, and from the recipes of earlier
runs. Some of what the teams settled on is public. :numref:`tab_practice_recipes`
collects the optimizer configurations disclosed by four prominent
pretraining reports from 2024--2026. Read it as evidence about practice, not
as gospel: each entry is what a team reported shipping, usually with no
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

Read down the columns and a consensus core appears. The optimizer is AdamW
with $(\beta_1, \beta_2) = (0.9, 0.95)$, the shortened second-moment window
that :numref:`sec_adamw` discussed. Weight decay is 0.1, with the exemptions
that section derived (OLMo 2 states the embedding exemption outright).
Gradients are clipped at global norm 1. The schedule is a brief warmup, a
few hundred to a few thousand steps out of a million, into either cosine
decay or the warmup--stable--decay shape of :numref:`sec_scheduler`, ending
one to two orders of magnitude below the peak. Where batch handling is
disclosed, the batch is ramped up early in training, while the gradient
noise scale is still small (:numref:`sec_batch_size`). The blanks carry
information too: even a report of Llama 3's length leaves betas, weight
decay, and clipping unstated, because much of the recipe travels as defaults in
training code rather than as prose. And one row breaks the optimizer
column: Kimi K2 runs the Muon split of :numref:`sec_muon`, hidden matrices
under orthogonalized momentum with AdamW for the rest, inside an otherwise
consensus recipe. That is the current state of the art in one table: a
stable core, one production-proven challenger, and the details you would
need to reproduce any row only partly on the record.

## Gradient Clipping

One column of the table has no section behind it yet. Gradient clipping
entered this book in :numref:`sec_rnn-scratch` as the fix for exploding RNN
gradients; the table shows that it outlived the architecture that motivated
it. Every run above that discloses a threshold clips, and every disclosed
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

### A NaN, Averted

:numref:`sec_adam` swept SGD with momentum on `TinyLM` and found a knife's
edge: the best learning rate sat one grid point below one that returned
NaN. That divergent grid point is exactly what clipping is for, so we rerun
it, with and without the guard, and nothing else changed.

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

The first steps tell the story, so we plot them raw, on a log scale.

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
tuned unclipped run of :numref:`sec_adam`. The last print is the heart of
the matter: the median gradient norm was far below the threshold, and
clipping changed the update on six steps out of two thousand. Six
interventions were the entire difference between a NaN and the best SGD
result in this chapter.
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

That count is the right mental model for clipping. On a healthy step the
guard does nothing; on the rare step when a spike arrives out of the
gradient distribution's tail, it is the difference between an incident and
a log entry. The tail is not an accident of our small model: gradient noise
in language models is measurably heavy-tailed, and under such noise clipped
SGD provably converges where plain SGD can fail
:cite:`Zhang.Karimireddy.Veit.ea.2020`. The threshold should be set
accordingly, above the typical norms of a healthy run, which you know
because you logged them. A guard that fires on most steps is not guarding
anything; it is a learning-rate cut in disguise, and the honest response is
to lower the learning rate or raise the threshold
:cite:`Godbole.Dahl.Gilmer.ea.2023`. For the Adam family the arithmetic
differs but the conclusion holds. Adam's normalization already caps every
coordinate's step near $\eta$ (:numref:`sec_adam`), so clipping will not
rescue a too-large Adam learning rate, and in our runs it did not. What it
still buys is protection for the estimates: one enormous gradient otherwise
enters $\mathbf{m}$ and $\mathbf{v}$ and distorts the steps for the
$\sim 1/(1-\beta_2)$ steps the averages take to forget it.

### The Stability Kit at Scale

At trillion-token scale, clipping is one item in a larger kit, most of it
aimed at the places where transformer blow-ups concentrate: the attention
logits and the output softmax. PaLM's training added a *z-loss*, a small
penalty on $\log^2 Z$ of the softmax normalizer, to keep the output logits
from drifting large :cite:`chowdhery2022palm`. *QK-norm* normalizes queries
and keys immediately before their dot product, so attention logits cannot
grow with the norms of what feeds them
:cite:`Henry.Dachapally.Pawar.ea.2020`; OLMo 2 adopted it as part of the
stability overhaul that its report documents :cite:`OLMo.2025`. MuonClip's
*QK-clip* rescales the query and key projections whenever the largest
attention logit crosses a cap, the addition that carried Kimi K2 through
15.5T tokens without a loss spike (:numref:`sec_muon`,
:cite:`Kimi.Team.2025`). And when prevention fails, the practice is
unglamorous. The PaLM team, facing about twenty spikes in a run, rewound to
a checkpoint a few hundred steps earlier and skipped the offending batches,
after establishing that the same batches caused no spike when replayed from
a different state: spikes came from state and data together, not from bad
data alone :cite:`chowdhery2022palm`. The OPT team published its training
logbook along with the model; it records two months of restarts, mid-flight
learning-rate cuts, and hardware failures in a way no polished paper does,
and it remains the best public record of what babysitting a large run is
actually like :cite:`zhang2022opt`.

## Weight Averaging

The chapter's third recurring decision, living with noise, has one more
tool, and it is nearly free. A constant-rate iterate rattles around its
noise ball (:numref:`sec_sgd`), and a schedule quenches the rattling by
decaying the rate (:numref:`sec_scheduler`). Averaging quenches it without
touching the rate: the bounces roughly cancel in the mean, so an average of
iterates sits near the center of the region the run is circling.
Stochastic weight averaging made this a standard trick, averaging
checkpoints from the tail of training and evaluating the average
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`. The running form is an
exponential moving average of the parameters,
$\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1 - \alpha)\, \mathbf{x}_t$:
the same leaky average this chapter has applied to gradients (momentum) and
to squared gradients (Adam), now applied to the weights themselves, purely
for evaluation. The training run never sees it.

We test it on the schedule testbed of :numref:`sec_scheduler`, reproduced
verbatim, in the situation where that section's story began: the constant
learning rate baseline, whose test accuracy stalled and jittered. Alongside
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
the averaged model stays self-consistent; integer buffers are copied
through.
:end_tab:

:begin_tab:`jax`
The EMA update is one `tree.map` over the two models' states. That state
includes the BatchNorm running statistics, so the averaged model stays
self-consistent.
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
    new = jax.tree.map(lambda e, p: decay * e + (1 - decay) * p,
                       nnx.state(ema), nnx.state(model))
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
the average still carries heavy weight on near-initialization iterates.
This is the same startup transient that Adam's bias correction cancels for
its moment estimates (:numref:`sec_adam`), uncorrected here, and it is why
practical EMAs either warm up the decay or start averaging late. Second,
once the window has filled, the averaged model is better and much steadier.
In our runs the EMA ends one to three points above the live weights, and
the width of that range is itself the finding: the live curve gambles a few
points of accuracy on when you happen to stop, and the average does not.
The averaged model
at a constant rate lands in the range the decayed schedules of
:numref:`sec_scheduler` reached, which is the schedule-free observation of
that section from the other side: decay and averaging are two ways to
quench the same noise. That also predicts the honest caveat. Add the same
EMA to a run whose schedule already decayed well, and on a model this size
the remaining nudge is within run-to-run noise; we checked. The technique
earns its keep where the decayed endpoint is expensive to reach, or where
no decay is coming. The cost is one extra copy of the parameters, cheap by
the accounting of :numref:`sec_adamw`. One footnote for checkpoint
averaging: if you average a few saved checkpoints instead of keeping a
running EMA, the BatchNorm statistics belong to none of the averaged
weights, so recompute them with a pass over the training data before
trusting the result.

### Averaging at Scale

At scale the trick wears several uniforms. Averaging the latest $k$
checkpoints of an LLM run, uniformly rather than exponentially, buys a
consistent mid-training speedup :cite:`Kaddour.2022`, and the top row of
:numref:`tab_practice_recipes` does it in production: the Llama 3 model
that shipped is the average of checkpoints from its final annealing phase
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Model soups push the idea past a single run,
averaging separately fine-tuned models into one, with accuracy gains at no
inference cost :cite:`Wortsman.Ilharco.Gadre.ea.2022`. And in one model
family averaging is not an optimization but a requirement: diffusion models
(:numref:`chap_diffusion`) are evaluated on EMA weights essentially always,
and sample quality depends on the averaging window strongly enough that
:citet:`Karras.Aittala.Lehtinen.ea.2024` built machinery to reconstruct the
EMA at any window *after* training, just to be able to tune it. A plausible
reading of the asymmetry: a classifier's accuracy is one forward pass and
plateaus, while a diffusion sampler applies the network hundreds of times
in sequence, so weight noise that a single pass shrugs off compounds across
the trajectory.

## How to Tune

Everything in this chapter was tuned somehow, and the somehow deserves to
be stated as method rather than left as folklore. The clearest statement of
the method practitioners actually use is Google's Tuning Playbook
:cite:`Godbole.Dahl.Gilmer.ea.2023`, and its central move is a vocabulary.
In any experiment, split the hyperparameters into three classes:
*scientific* hyperparameters, the ones your question is about; *nuisance*
hyperparameters, which must be re-optimized for every setting of the
scientific ones before the comparison means anything; and *fixed*
hyperparameters, held constant and acknowledged as limits on the claim. The
playbook's rule is that a comparison is a statement about scientific
hyperparameters only after the nuisances have been re-tuned per arm, and
its most common instance is also the most common failure in published
comparisons: the learning rate is almost always a nuisance, and a
comparison at one shared learning rate is a comparison of nothing.

This vocabulary names what the chapter has been doing since
:numref:`sec_adam`. In each race, the optimizer was the scientific
hyperparameter; the learning rate was the nuisance, re-tuned per contestant
on a four-point grid spaced by factors of about three; steps, batch size,
initialization, and the absence of a schedule were fixed and stated, which
is why every conclusion was phrased as conditional on that protocol. The
design scales up without changing shape: more nuisance dimensions (peak
rate, decay horizon, warmup, weight decay), quasi-random search once a grid
is too coarse, exploration before a final exploitation sweep. What does not
survive is skipping the re-tune. :citet:`Schmidt.Schneider.Hennig.2021`
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
through their product (:numref:`sec_adamw`), so their joint grid is really
a ridge plus one cross-direction. Past that, you are running a study, and
the playbook is the reference. On the models of this chapter a run cost
seconds to minutes, which is why we could afford the middle tier
everywhere; the entire point of :numref:`sec_scaling` was to keep that
affordability relevant as models grow.

Finally, the log. A result you cannot reproduce is a rumor, so record, for
every run, the full configuration including the hyperparameters you
consider fixed, the code version, the seed, and the one thing you changed,
and change one thing at a time. Keep the diverged runs: the NaN edge of a
sweep marks the stability boundary, and :numref:`sec_adam` read its NaN
column as data, not as failure. Write the conclusion next to the curves
while you still believe it, because a directory of loss curves with no
sentences attached goes stale within weeks. Assignments in the CS336 mold
now grade the experiment log alongside the final loss, and that is the
right emphasis: the log is the experiment.

## What We Did Not Teach

Three method families were left out on purpose. Sharpness-aware
minimization takes an inner ascent step before each descent step to seek
flat minima; it doubles the gradient cost, and its dependable wins are in
vision and fine-tuning rather than large-scale pretraining
:cite:`Foret.Kleiner.Mobahi.ea.2021`. Variance-reduction methods of the
SVRG family own an elegant theory for finite sums that has never paid its
way on deep networks; the theory, and the honest post-mortem, live in
:numref:`sec_mdl-variance-reduction`. LARS and LAMB, the layerwise-adaptive
methods once synonymous with large-batch training, are superseded: with
nuisance hyperparameters re-tuned, exactly the rule of the previous
section, standard momentum and AdamW match them at the batch sizes they
were designed for :cite:`Nado.Gilmer.Shallue.ea.2021`.

What remains is not optimization but placement. This chapter priced the
optimizer's state (:numref:`sec_adamw`) and always kept it on one device.
Spreading gradients and state across a data-parallel group, ZeRO-style
sharding, and the overlap of communication with computation belong to
:numref:`chap_performance` and the training-systems material of
:numref:`sec_training_systems`. The updates computed there are the ones
derived here; the systems side decides where the bytes live and when they
move.

## Summary

Disclosed practice at the frontier is a small, stable recipe: AdamW with
$(0.9, 0.95)$, weight decay 0.1 with embeddings and norms exempt, global
gradient clipping at 1, brief warmup into cosine or warmup--stable--decay,
and an early batch ramp, with the Muon split as the one production
challenger inside the same frame. Clipping is cheap insurance against a
heavy gradient tail: healthy runs barely trigger it, and when it triggers
on most steps it has become a learning-rate cut. Beyond it, large runs
carry a stability kit (z-loss, QK-norm, QK-clip) and a rewind-and-skip
playbook for the spikes that get through. Weight averaging quenches
noise without touching the learning rate; it is a point of accuracy and a
steadier endpoint on our testbed, checkpoint averaging in production LLMs,
and mandatory equipment for diffusion. Tuning is a protocol, not a talent:
scientific, nuisance, and fixed hyperparameters, nuisances re-tuned per
arm, budgets spent on the learning rate first, and a log that makes every
run reproducible.

This section closes the chapter, so step back once. Every method in it was
a way of making three decisions: which direction to move, which is a choice
of norm; how far to move as training proceeds, which is a schedule; and how
to live with sampled noise, which is batching, momentum, and averaging. The
recipe table is one coordinated setting of all three, tested at an expense
no benchmark will ever match. When a run of your own
misbehaves, the useful first question is which of the three decisions is
failing. Optimizers will keep changing; the decomposition has been stable
for decades, and it is what this chapter was actually about.

## Exercises

1. Reproduce the tuning protocol with a budget. Using AdamW on `TinyLM`,
   reach a training loss of 1.1 in as few steps as you can, tuning only the
   learning rate, with a budget of ten runs. Keep a log: for every run,
   record the learning rate, the steps taken, the final loss, and the
   one-sentence conclusion you drew before launching the next run. Report
   the log, not just the winning configuration.
1. Change the batch size of the clipping demo's model from 64 to 256 and
   repeat the four-point learning-rate sweep of :numref:`sec_adam` for
   AdamW. Where does the optimum move, and how does the shift compare with
   the square-root and linear scaling rules of :numref:`sec_batch_size`?
   What would you have concluded had you reused the batch-64 learning rate
   without re-tuning?
1. Sweep the clipping threshold $\theta \in \{0.01, 0.1, 0.5, 1, 4,
   \infty\}$ in the demonstration above, recording the final loss and the
   fraction of steps on which clipping fired. Identify the three regimes:
   guard (fires rarely), brake (fires constantly), and absent. Show that
   when clipping fires on every step, the update of
   :eqref:`eq_practice_clip` is normalized gradient descent with step size
   $\eta\theta$, and explain why that is not the same as training at a
   lower learning rate.
1. The decay timescale of :numref:`sec_adamw` is $\tau = B/(\eta \lambda D)$
   epochs for batch size $B$, dataset size $D$ (both in tokens), learning
   rate $\eta$, and weight decay $\lambda$. Compute $\tau$ for the
   DeepSeek-V3 and OLMo 2 rows of :numref:`tab_practice_recipes` at their
   peak learning rates. What fraction of each dataset does the averaging
   horizon span? Following :citet:`Bergsma.Dey.Gosal.ea.2025b`, what should
   happen to $\lambda$ if the batch size were doubled and $\tau$ is to be
   preserved?
1. Sweep the EMA decay $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ in the
   weight-averaging demonstration. Relate the averaging window
   $1/(1-\alpha)$ to what you observe at both ends: when does the EMA track
   the live weights too closely to help, and when does it average in
   iterates too old to be useful within the 15-epoch budget?
1. Test the finding of :citet:`Schmidt.Schneider.Hennig.2021` on this
   chapter's testbed: run SGD with momentum, Adam, and AdamW at their
   framework default settings on `TinyLM` for 2,000 steps, and compare the
   best of the three against the grid-tuned Adam of :numref:`sec_adam`.
   Given a fixed budget of four runs, which strategy would you choose on
   this problem, and what does the answer depend on?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.12]{.kicker}

The craft of training<br>
**the disclosed recipe · clipping · weight averaging · how to tune**
:::
:::

::: {.slide title="The recipe, as disclosed"}
[What frontier runs report shipping — practice, not gospel]{.kicker}

| run | optimizer | $\beta_1,\beta_2$ | peak LR, schedule | warmup | clip | wd |
|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B | AdamW | — | $8{\times}10^{-5}$, cosine | 8k | — | — |
| DeepSeek-V3 | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, WSD-like | 2k | 1.0 | 0.1 |
| OLMo 2 7B | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, cosine | 2k | 1.0 | 0.1* |
| Kimi K2 | MuonClip | — | $2{\times}10^{-4}$, WSD | 500 | — | 0.1 |

- Consensus core: AdamW (0.9, 0.95) · wd 0.1 with exemptions · clip 1.0 ·
  warmup + cosine-or-WSD · early batch ramp.
- Blanks are data too: the recipe travels as code defaults, not prose.
- One break in the optimizer column: K2 runs the **Muon split** (§9.9)
  inside an otherwise consensus recipe.
:::

::: {.slide title="Gradient clipping in three lines"}
Global norm, all parameters as one vector:

$$\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

Direction kept, length capped. Met in ch. 8 for RNNs — it outlived the
architecture: every disclosed threshold in the table is 1.

@practice-gradient-clipping-1
:::

::: {.slide title="A NaN, averted"}
§9.6's knife edge: SGD's best lr sat one grid point below a NaN. Rerun the
divergent point, with and without the guard:

@!practice-a-nan-averted-3

. . .

- Unclipped: oversized step → steeper ground → bigger gradient → momentum
  compounds → overflow.
- Clipped, same lr: trains to the tuned run's range.
:::

::: {.slide title="Six steps out of two thousand"}
The instrumented run: median gradient norm ~0.3, threshold 1.0 —
**clipping changed the update on 6 of 2,000 steps**.

- A fuse, not a brake: language-model gradient noise is heavy-tailed
  (Zhang et al., 2020); the guard exists for the tail.
- Firing on most steps = a learning-rate cut in disguise. Lower $\eta$ or
  raise $\theta$.
- Adam: clipping won't rescue a too-large $\eta$ (steps are already
  $\sim\eta$-capped) — it guards $\mathbf{m}, \mathbf{v}$ from one huge
  gradient lingering $1/(1-\beta_2)$ steps.
:::

::: {.slide title="The stability kit at scale"}
Clipping is one item. The rest aims at attention logits and the softmax:

- **z-loss** (PaLM): penalize $\log^2 Z$ of the softmax normalizer.
- **QK-norm** (OLMo 2): normalize $q, k$ right before their dot product.
- **QK-clip** (MuonClip): cap the largest attention logit — 15.5T tokens,
  zero spikes (§9.9).

. . .

When prevention fails: PaLM **rewound ~100 steps and skipped the batches**
— same data replayed later caused no spike; state and data conspire. The
OPT logbook: two months of restarts and lr cuts, published as-is.
:::

::: {.slide title="Weight averaging"}
Third decision, third tool: quench noise **without touching the rate** —
$\bar{\mathbf{x}}_t = \alpha \bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$,
the chapter's leaky average, now on the weights (SWA; Izmailov et al.,
2018).

@!practice-weight-averaging-3

- Window must fill first (Adam's bias-correction transient, uncorrected).
- Then: 1–3 points above the live weights, and no when-to-stop lottery.
:::

::: {.slide title="Averaging: where it matters"}
- Constant rate + EMA ≈ decayed schedule: decay and averaging quench the
  same noise (§9.8's schedule-free view). On an already-decayed run this
  size: within noise — we checked.
- LLMs: checkpoint averaging (LAWA); **Llama 3 shipped an average** of its
  annealing checkpoints. Model soups: average fine-tuned models.
- Diffusion (ch. 15): EMA is **mandatory** — quality tracks the window so
  tightly that Karras et al. (2024) reconstruct the EMA post hoc to tune it.
:::

::: {.slide title="How to tune"}
[The Tuning Playbook's vocabulary]{.kicker}

- **Scientific** hyperparameters: the question. **Nuisance**: re-tune per
  arm, or the comparison is void. **Fixed**: the claim's fine print.
- This chapter, named: optimizer scientific, lr nuisance (four-point grid
  per contestant), all else fixed and stated.
- Schmidt et al. (2021): several optimizers at defaults ≈ one optimizer
  heavily tuned. Untuned comparisons measure effort, not algorithms.

. . .

Budgets: few runs → consensus recipe, sweep peak lr only. Tens → add wd +
schedule ($\eta\lambda$ is a ridge, §9.7). And log everything: config,
seed, the one change, the NaNs. **The log is the experiment.**
:::

::: {.slide title="What we did not teach, and where it lives"}
- **SAM**: flat minima at 2× gradient cost — wins concentrate in vision
  and fine-tuning.
- **Variance reduction**: beautiful finite-sum theory, never paid off for
  deep nets → ch. 25.
- **LARS/LAMB**: superseded — re-tuned momentum/AdamW match them at the
  same batch sizes.
- **Systems**: sharding state, data parallelism, overlap → ch. 11 and the
  training-systems appendix.
:::

::: {.slide title="Recap: three decisions"}
- **Direction**: a norm — gradient, sign, or orthogonalized (§9.2, §9.6,
  §9.9).
- **Step size over time**: warmup, cosine, WSD (§9.8) — plus clip 1.0 as
  the fuse.
- **Noise**: batch (§9.4, §9.10), momentum (§9.5), averaging (here).

. . .

The recipe table is one coordinated setting of all three. When a run
misbehaves, ask which decision is failing. Optimizers change; the
decomposition has been stable for decades.
:::
