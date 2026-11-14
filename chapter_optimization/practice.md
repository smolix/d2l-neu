# Practice
:label:`sec_practice`

Every experiment in this chapter so far ran in seconds or minutes. We could
sweep a learning rate over four values, repeat a comparison, and restart
a run that went wrong. A large training run allows none of this. A sweep
over four learning rates costs four times the run itself, so most
hyperparameters are set once and never tuned on the model that matters. A
single oversized update can undo days of progress. PaLM, a language model
with 540 billion parameters trained on 6,144 TPU chips, saw its loss spike
about twenty times along the way, although gradient clipping was enabled
:cite:`chowdhery2022palm`. And the checkpoint at the end is one draw from a
noisy trajectory, yet it is the one that ships. This section asks how to
set up, protect and finish a run that can be done only once.

We follow such a run through its life, using the methods of the earlier
sections on models small enough to train here. We first ask how a
configuration is chosen when the target cannot be tuned, and read the
settings that four recent reports disclose. We then protect the run against
rare oversized updates with gradient clipping, and ask what to do when a
spike gets through anyway. Next we reduce the noise in the final weights by
averaging them. Finally we turn to the question behind every comparison in
this chapter: what does it take to show that one optimizer is better than
another?

```{.python .input #practice}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #practice}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #practice-1}
%%tab pytorch
import random
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

<!-- WRITER: the JAX `record_norms` transformation moved into the JAX cell below from its own cell after the data cell (PRAC-8), and `clipped` now records the norm before clipping on every step, as the PyTorch `Clipped` does (it takes the number of steps to record). The JAX tab note that explained `record_norms` moved up with it and follows; merge the two JAX notes into one or two sentences. -->

:begin_tab:`jax`
An Optax transformation is a pair of pure functions, `init` and `update`,
so a quantity that must persist from step to step lives in the
transformation's state. `record_norms` passes the gradients through
unchanged and writes each step's global norm into an array of fixed length
in its state.
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
def record_norms(num_steps):
    """Pass the gradients through unchanged, storing each step's norm."""
    def init(params):
        return dict(norms=jnp.zeros(num_steps), step=jnp.zeros((), jnp.int32))

    def update(grads, state, params=None):
        norm = optax.global_norm(grads)
        norms = state['norms'].at[state['step']].set(norm)
        return grads, dict(norms=norms, step=state['step'] + 1)

    return optax.GradientTransformation(init, update)

def clipped(tx, num_steps, max_norm=1.0):
    """Clip the global gradient norm before an optimizer's update, recording
    the norm before clipping on each of the first num_steps steps."""
    return optax.chain(record_norms(num_steps),
                       optax.clip_by_global_norm(max_norm), tx)
```

### Preventing a Numerical Failure
:label:`subsec_practice-nan`

We now rerun SGD at the divergent point of that sweep, $\eta = 1$, with
and without clipping. The model is `TinyLM`, the decoder-only transformer
of :numref:`subsec_tinylm`: two blocks, about 0.47 million parameters,
trained to predict the next byte of WikiText-103, a corpus of Wikipedia
articles, from the 64 bytes before it. The library function `d2l.train_steps`
trains it for a fixed number of steps on minibatches of 64 sequences and
returns the training loss of every step. Both runs start from the same
initial weights and use SGD with momentum 0.9 at $\eta = 1$ for 2,000
steps; they differ only in the clipping. To see how often and when clipping
intervenes, we also record the global gradient norm before clipping at
every step.

<!-- WRITER: the experiment is now four settings of SGD with momentum 0.9, each run twice (two seeds; within a seed every setting starts from the same initialization, and in PyTorch from the same minibatch order): lr 1 unclipped, lr 1 clipped at 1, lr 0.3 unclipped (the best rate of the SGD grid in :numref:`sec_adam`), and lr 3 clipped at 1. A threshold of infinity is no clipping (`clip_grad_norm_` and `clip_by_global_norm` then leave the gradient unchanged), so one function serves every run. For each clipped run the table prints the number of clipped steps, the last clipped step, the median gradient norm, and the largest norm after step 100 divided by the median (PRAC-12). Measured (development and two notebook runs, PyTorch / JAX):
- lr 1 unclipped: inf in every run (the loss leaves the plot within about 15 steps).
- lr 1 clipped at 1: trains in every run, 50 runs in all. Final loss 1.610 to 1.654 (PyTorch, 37 runs; median 1.620) and 1.608 to 1.615 (JAX, 13 runs). (PyTorch's spread is wider; enabling or disabling TF32 matmuls in PyTorch does not change it, 8 runs each; the medians differ by 0.008, below the spread of :numref:`sec_test_models`, and the lesson is the same in both tabs.) Clipping acts on 11 to 18 of 2,000 steps, the last at step 10 to 17, never after step 100; median norm 0.25 to 0.26, so the threshold 1 is about four times the typical norm.
- lr 0.3 unclipped: 1.647 to 1.703 (PyTorch) and 1.652 to 1.683 (JAX). In every one of the 26 paired runs (13 per tab) the clipped run at lr 1 ended below the unclipped run at 0.3 of the same seed, by 0.01 to 0.08 nats (median about 0.045). The difference is consistent but not larger than the spread of independent runs (0.04 to 0.06), so state it as "at least as low as the best unclipped rate", not as a margin. Not printed, probed: clipped at lr 0.5, 1.615 to 1.662 (12 runs), also always trains, while unclipped 0.5 diverged in 16 of 18 runs in :numref:`sec_adam`. The firm lesson: clipping moves SGD's edge of stability from between 0.3 and 0.5 up to between 1 and 3 on this model, because the edge was set by the first few steps (PRAC-11).
- lr 3 clipped at 1: clipping acts on all 2,000 steps (a brake, not a guard), median norm 22 to 43, and the run fails anyway: final loss 88 to 207 in every run, far above the uniform guess ln 256 = 5.5. Clipping does not rescue a rate three times above the new edge (author ruling 3, "what it does not fix"). The plot shows its loss climbing within the first 20 steps.
- Largest gradient norm after step 100 over the median: 1.5 to 2.1 in every run at lr 1 (clipped) and 0.3; nothing like a heavy tail on this model with minibatches of 64 full windows, so "gradient noise in language models is heavy-tailed" must be stated as what large runs report (:cite:`Zhang.Karimireddy.Veit.ea.2020`), not as what this run shows (PRAC-12). (For lr 3 the ratio is meaningless: the norms grow with the failing run.)
- Plot: first 50 steps of the first seed, all four settings, log scale; the unclipped lr 1 run falls for two to three steps and then climbs (PRAC-10: "after the first few steps"). -->

```{.python .input #practice-a-nan-averted-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

```{.python .input #practice-preventing-a-numerical-failure-1}
%%tab pytorch
new_seed = lambda: random.getrandbits(31)

def run(seed, lr, max_norm, num_steps=2000):
    """SGD with momentum 0.9 on TinyLM, clipped at max_norm (math.inf: not
    at all); return the losses and the gradient norms before clipping."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    model = d2l.TinyLM(len(data.vocab))
    optimizer = Clipped(torch.optim.SGD(model.parameters(), lr, momentum=0.9),
                        model.parameters(), max_norm)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    return losses, torch.stack(optimizer.norms).cpu().numpy()
```

```{.python .input #practice-preventing-a-numerical-failure-1}
%%tab jax
new_seed = d2l.get_seed

def run(seed, lr, max_norm, num_steps=2000):
    """SGD with momentum 0.9 on TinyLM, clipped at max_norm (math.inf: not
    at all); return the losses and the gradient norms before clipping."""
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    tx = clipped(optax.sgd(lr, momentum=0.9), num_steps, max_norm)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    return losses, np.asarray(optimizer.opt_state[0]['norms'][...])
```

```{.python .input #practice-a-nan-averted-2}
%%tab pytorch, jax
settings = [(1.0, math.inf), (1.0, 1.0), (0.3, math.inf), (3.0, 1.0)]
seeds, runs = [new_seed() for _ in range(2)], {}
print(f'{"lr":>4}{"clip at":>9}{"final loss":>12}{"steps clipped":>15}'
      f'{"last":>6}{"median norm":>13}{"largest / median":>18}')
for seed in seeds:
    for lr, max_norm in settings:
        losses, norms = runs[seed, lr, max_norm] = run(seed, lr, max_norm)
        line = f'{lr:4g}{max_norm:9g}{d2l.final_loss(losses):12.3f}'
        if max_norm < math.inf:
            fired = np.nonzero(norms > max_norm)[0]  # Steps clipping changed
            median = np.median(norms)
            line += (f'{len(fired):15d}{fired.max(initial=0):6d}{median:13.2f}'
                     f'{norms[100:].max() / median:18.1f}')  # After step 100
        print(line)
```

The unclipped run ends with a loss of `nan`, "not a number", the value that
floating-point arithmetic produces once a computation has overflowed. The
instability occurs early, so we plot the first 50 steps on a log scale.

```{.python .input #practice-a-nan-averted-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 51)),
         [runs[seeds[0], lr, max_norm][0][:50] for lr, max_norm in settings],
         'step', 'training loss', yscale='log', figsize=(5, 2.5),
         legend=[f'lr {lr:g}, clip at {max_norm:g}' for lr, max_norm in settings],
         legend_loc='outside')
```

The first step lowers the loss, and from the second on the unclipped run
feeds on itself. An oversized step moves the parameters to where the loss
is steeper, the gradient there is larger, and momentum carries the excess
into the following steps. The loss grows by several orders of magnitude
and overflows within about twenty steps. The clipped run, at the same
learning rate, trains normally, to a final loss comparable with that of the
best SGD run in :numref:`sec_adam`, which used a learning rate about three
times smaller.

### Clipping Frequency

The printed counts show how little clipping did to achieve this, and when
it did it. The median gradient norm of the clipped run is about a quarter
of the threshold, and clipping changed the update on a few dozen at most
of the 2,000 steps, all of them within the first forty. These are
the steps on which the freshly initialized network has its largest
gradients. Once they are past, the clipped run trains at $\eta = 1$ with
clipping idle for the rest of the 2,000 steps. Here clipping did the job of
a warmup, and an adaptive one. It shortened exactly those first steps that
were too long, until the parameters reached a region where the full step is
safe, and it needed no schedule to tell it when to stop. Every run in
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

We test the average on the convolutional network `d2l.lenet_bn` of
:numref:`sec_scheduler`: a LeNet with ReLU activations, max-pooling, and
batch normalization after every hidden layer, initialized with Xavier
weights and zero biases, and trained on Fashion-MNIST in minibatches of 256
images. `d2l.evaluate_accuracy` of :numref:`sec_adam` measures its test
accuracy. At the constant learning rate $\eta = 0.3$, the baseline of that
section, its test accuracy gains little after about ten epochs and then
moves up and down from epoch to epoch. Alongside the live weights we keep
an EMA with $\alpha = 0.999$, whose window of about 1,000 steps spans
roughly four epochs of 235 steps each.

<!-- WRITER: the experiment moved from `lenet_bn` (removed from the library by the Schedules step) to `FashionCNN` with plain SGD at 0.3, the baseline of :numref:`sec_test_models`. `FashionCNN` has no batch normalization, so the paragraph on averaging batch-normalization statistics and `use_buffers=True` are gone (keep the advice to recompute the statistics after averaging saved checkpoints, in one sentence where averaged checkpoints are discussed, PRAC-20). The average now starts after epoch 5 (PRAC-18), so the lag paragraph shrinks to a sentence. A cosine decay of the same length (15 epochs, from the same initialization, warmup 0) runs next to the constant rate (PRAC-19), and both keep an EMA. Two seeds. Measured, ten paired runs over both tabs (three development seeds and two notebook runs of two seeds per tab), test accuracy: constant rate, live weights: the last five epochs scatter over 1 to 5 points within a run (overall 0.859 to 0.917), and the final epoch lands anywhere in that range (0.859 to 0.916); constant rate, EMA: final 0.911 to 0.915, rising steadily over the last five epochs; cosine decay, final live weights: 0.908 to 0.914. EMA at the constant rate minus the decayed run's final weights: +0.001 to +0.006 in all ten runs, smaller than the spread of either setting, so state "matches", not "beats". The EMA ends above the final live weights of the same run in 9 of 10 runs (by up to 5 points); in one run the last live epoch landed at 0.916 against the EMA's 0.914, so the firm statement is about the scatter of the live weights, not about one final value. The EMA of the cosine run (also from epoch 5) ends at or up to 0.3 points below that run's final live weights: after a decay the average only lags. The window of alpha = 0.999 is about 1,000 steps, a little over four epochs of 235 steps; the average covers epochs 6 to 15. -->

```{.python .input #practice-weight-averaging-1}
%%tab pytorch, jax
fashion = d2l.FashionMNIST(batch_size=256)
num_epochs, num_batches = 15, len(fashion.train_dataloader())
test_acc = lambda model: d2l.evaluate_accuracy(model, fashion.val_dataloader())
```

<!-- WRITER: the two tab notes below describe the new cells (PyTorch: `AveragedModel` updated from an optimizer step hook that is registered when averaging starts; JAX: the EMA kept by an Optax transformation chained after the optimizer). Shorten to a sentence each. -->

:begin_tab:`pytorch`
`AveragedModel` keeps a copy of the network. The first call of its method
`update_parameters` copies the live weights into it, and with
`get_ema_multi_avg_fn` every later call blends them in as in
:eqref:`eq_practice_ema`. `register_step_post_hook` makes the optimizer
call a function after every step; we use it to advance the learning-rate
schedule and, from epoch 5 on, to update the average.
:end_tab:

:begin_tab:`jax`
`ema_of_weights` is an Optax transformation placed after the optimizer in
the chain: it passes the updates through unchanged, applies them to a copy
of the parameters, and blends the result into an average in its state with
`optax.incremental_update`, which computes :eqref:`eq_practice_ema` for
every array. `averaged` writes that average into a copy of the model.
:end_tab:

```{.python .input #practice-weight-averaging-4}
%%tab pytorch
def warmup_cosine(peak, warmup, num_steps):
    """A linear warmup to peak over warmup steps, then a cosine to zero."""
    def schedule(t):
        if t < warmup:
            return peak * t / warmup
        return peak * (1 + math.cos(math.pi * (t - warmup)
                                    / (num_steps - warmup))) / 2
    return schedule
```

```{.python .input #practice-weight-averaging-4}
%%tab jax
warmup_cosine = lambda peak, warmup, num_steps: (
    optax.warmup_cosine_decay_schedule(0.0, peak, warmup, num_steps))

def ema_of_weights(alpha=0.999, start=0):
    """Pass the updates through unchanged, keeping an EMA of the updated
    weights that averages from step start on (copies them before)."""
    def init(params):
        return dict(ema=params, step=jnp.zeros((), jnp.int32))

    def update(updates, state, params):
        new = optax.apply_updates(params, updates)
        rate = jnp.where(state['step'] < start, 1.0, 1 - alpha)
        ema = optax.incremental_update(new, state['ema'], rate)
        return updates, dict(ema=ema, step=state['step'] + 1)

    return optax.GradientTransformation(init, update)

def averaged(model, optimizer):
    """A copy of model with the EMA kept by the last transformation."""
    ema = nnx.clone(model)
    nnx.update(ema, optimizer.opt_state[-1]['ema'])
    return ema
```

```{.python .input #practice-averaging-at-a-constant-learning-rate-1}
%%tab pytorch
def train_fashion(seed, schedule, alpha=0.999, start=5):
    """Plain SGD on FashionCNN at the rate schedule(t); the test accuracy
    after every epoch of the weights and of their EMA from epoch start on."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    model = d2l.FashionCNN().to(d2l.try_gpu())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # Times schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.register_step_post_hook(lambda *_: scheduler.step())
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(alpha))
    live, avg = [], []
    for epoch in range(num_epochs):
        if epoch == start:  # Its first update copies the current weights
            optimizer.register_step_post_hook(
                lambda *_: ema.update_parameters(model))
        d2l.train_steps(model, fashion, optimizer, num_batches)
        live.append(test_acc(model))
        avg.append(test_acc(ema) if epoch >= start else math.nan)
    return live, avg
```

```{.python .input #practice-averaging-at-a-constant-learning-rate-1}
%%tab jax
def train_fashion(seed, schedule, alpha=0.999, start=5):
    """Plain SGD on FashionCNN at the rate schedule(t); the test accuracy
    after every epoch of the weights and of their EMA from epoch start on."""
    model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
    tx = optax.chain(optax.sgd(schedule),
                     ema_of_weights(alpha, start * num_batches))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    live, avg = [], []
    for epoch in range(num_epochs):
        d2l.train_steps(model, fashion, optimizer, num_batches)
        live.append(test_acc(model))
        avg.append(test_acc(averaged(model, optimizer)) if epoch >= start
                   else math.nan)
    return live, avg
```

```{.python .input #practice-weight-averaging-2}
%%tab pytorch, jax
schedules = {'constant rate': lambda t: 0.3,
             'cosine decay': warmup_cosine(0.3, 0, num_epochs * num_batches)}
cnn_seeds, curves = [new_seed() for _ in range(2)], {}
print(f'{"test accuracy":<15}{"live, last five epochs":>24}{"live, final":>13}'
      f'{"EMA, final":>12}')
for seed in cnn_seeds:
    for name, schedule in schedules.items():
        live, avg = curves[seed, name] = train_fashion(seed, schedule)
        print(f'{name:<15}{min(live[-5:]):15.3f} to {max(live[-5:]):.3f}'
              f'{live[-1]:13.3f}{avg[-1]:12.3f}')
```

```{.python .input #practice-weight-averaging-3}
%%tab pytorch, jax
(live, avg), (decayed, _) = (curves[cnn_seeds[0], name] for name in schedules)
d2l.plot(list(range(1, num_epochs + 1)), [live, avg, decayed], 'epoch',
         'test accuracy', ylim=[0.85, 0.925], figsize=(5, 2.5),
         legend=['constant rate, live weights', 'constant rate, EMA',
                 'cosine decay, live weights'], legend_loc='outside')
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
jumps up and down, in some runs by several points, while the EMA's rises
steadily, by about one to two points, and ends at its highest value. The live curve shows the noise ball
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

## A Worked Configuration

<!-- WRITER: NEW subsection (structure 12.3), the chapter's ending; it replaces "Choosing a Configuration", "Comparing Optimizers" and "The Chapter's Three Decisions" (move the table of reported configurations here as the precedent). The cells configure a TinyLM the chapter has not tuned, four blocks of width 256 with four heads (3,307,264 parameters), from nothing to a finished 3,000-step run, every step from an earlier section. Measured in development (PyTorch, JAX; RTX 4090, shared GPU):
(1) Batch size from the device: throughput, sequences per second, levels off at a batch of 256 in both tabs (PyTorch about 17,500 to 19,300 at 256 and 18,900 to 19,100 at 512 and 1,024; one gradient takes about 12 to 15 ms up to 256 and doubles with each doubling beyond; JAX 20,700 to 25,000 at 256 and less at 512 and 1,024; one gradient 4 to 8 ms up to 128, 10 to 12 ms at 256). Small batches are launch-bound, so their time per gradient is noisy on a shared GPU; the knee is stable. We train at 256. The noise scale (:numref:`sec_batch_size`, `d2l.noise_scale`) is 2 at initialization in every run and 55 to 205 after 300 steps at the chosen rate (ten runs over both tabs; repeated measurements of one model agree within 3%), and 400 to 580 after 1,500 steps (probe): the batch of 256 is at or above the noise scale for the first few hundred steps and below it for most of the run, the growth that :numref:`sec_batch_size` measured.
(2) AdamW with (0.9, 0.95), weight decay 0.1 on the matrices only (:numref:`subsec_adamw-what-not-to-decay`), clipping at 1 with the fraction of clipped steps logged, an EMA of the weights (alpha = 0.995, a window of 200 steps). `train` takes the schedule and the threshold, so the baseline is the same function with a constant rate and a threshold of infinity (its EMA is not used).
(3) Three 300-step runs at constant rates 1e-3, 3e-3, 1e-2 (one seed per notebook run): 1.862 to 1.894, 1.717 to 1.757, 2.463 to 2.508 (eight sweeps over both tabs). 1e-2 stalls in every sweep, so 3e-3 is both the largest stable rate and the best; `peak` is the grid point with the lowest 300-step loss. Probed: 3e-4 and 3e-2 are worse (2.19 to 2.20, 2.57 to 2.62). Clipping acts on 2 to 7% of the steps of these short runs, all early. The choice holds at full length: constant rates for 3,000 steps end at 1.336 / 1.340 (1e-3), 1.322 / 1.324 (3e-3), 2.442 / 2.477 (1e-2) (training loss, one seed per tab).
(4) Warmup over 2% of the steps (60), then a cosine to zero, at the peak 3e-3 (`warmup_cosine`, defined with the averaging cells).
(5) Final runs, validation loss in nats per byte, 14 seeds over both tabs (development and two notebook runs): configured 1.242 to 1.254; baseline, the same AdamW at the constant 3e-3 without warmup, clipping or averaging, 1.299 to 1.310; gap per seed 0.048 to 0.065, against a spread between the two seeds of one notebook run of 0.000 to 0.008 for either setting: the gap is six to ten times the spread. Clipping acts on 0.5 to 0.7% of the steps, all in the first 100. Most of the gap is the decay (warmup and clipping touch only the first steps; :numref:`sec_scheduler` measured the decay alone). The EMA after the cosine decay equals the final weights to within 0.001 in every run: after a full decay there is nothing left to average, as "Averaging and Decay" predicts. Probed: an EMA with a 1,000-step window (alpha = 0.999 from step 0) ends 0.025 above the final weights, because it reaches back into steps the decay had not finished. The average earns its keep on a run stopped before its decay ends, or at a constant rate (the FashionCNN cell). Run time: about 3 minutes per tab for the whole subsection. -->

```{.python .input #practice-a-worked-configuration-1}
%%tab pytorch
def tiny_lm(seed):
    """A TinyLM of four blocks of width 256 on the GPU."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    return d2l.TinyLM(len(data.vocab), d_model=256, num_heads=4,
                      num_blks=4).to(d2l.try_gpu())

def sample_lm(b):  # b windows at random offsets in the training text
    starts = torch.randint(len(data.train) - data.num_steps, (b,))
    X, Y = data.windows(data.train, starts.numpy())
    return X.to(d2l.try_gpu()), Y.to(d2l.try_gpu())
```

```{.python .input #practice-a-worked-configuration-1}
%%tab jax
def tiny_lm(seed):
    """A TinyLM of four blocks of width 256."""
    return d2l.TinyLM(len(data.vocab), d_model=256, num_heads=4, num_blks=4,
                      rngs=nnx.Rngs(seed))

rng = np.random.default_rng(d2l.get_seed())

def sample_lm(b):  # b windows at random offsets in the training text
    return data.windows(data.train,
                        rng.integers(len(data.train) - data.num_steps, size=b))
```

```{.python .input #practice-a-worked-configuration-2}
%%tab pytorch, jax
model = tiny_lm(new_seed())
print(f'{d2l.num_params(model):,} parameters')
for b in [32, 64, 128, 256, 512, 1024]:
    t = d2l.Benchmark(lambda: d2l.grad_sq_norm(model, *sample_lm(b))).time
    print(f'batch {b:5d}: {1000 * t:5.1f} ms per gradient, '
          f'{b / t:6.0f} sequences per second')
```

```{.python .input #practice-a-worked-configuration-3}
%%tab pytorch
def train(seed, schedule, num_steps, max_norm=1.0, alpha=0.995):
    """AdamW with betas (0.9, 0.95) and weight decay 0.1 on the matrices at
    the rate schedule(t), clipped at max_norm, with an EMA of the weights;
    return the losses, the model, the EMA and the fraction of clipped steps."""
    model = tiny_lm(seed)
    matrices = [p for name, p in model.named_parameters()
                if p.ndim == 2 and 'emb' not in name]
    others = [p for name, p in model.named_parameters()
              if p.ndim != 2 or 'emb' in name]
    adamw = torch.optim.AdamW([{'params': matrices, 'weight_decay': 0.1},
                               {'params': others, 'weight_decay': 0.0}],
                              lr=1.0, betas=(0.9, 0.95))  # Times schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, schedule)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(alpha))
    adamw.register_step_post_hook(lambda *_: scheduler.step())
    adamw.register_step_post_hook(lambda *_: ema.update_parameters(model))
    optimizer = Clipped(adamw, model.parameters(), max_norm)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    clip_rate = (torch.stack(optimizer.norms) > max_norm).float().mean()
    return losses, model, ema, float(clip_rate)
```

```{.python .input #practice-a-worked-configuration-3}
%%tab jax
def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

def train(seed, schedule, num_steps, max_norm=1.0, alpha=0.995):
    """AdamW with betas (0.9, 0.95) and weight decay 0.1 on the matrices at
    the rate schedule(t), clipped at max_norm, with an EMA of the weights;
    return the losses, the model, the EMA and the fraction of clipped steps."""
    model = tiny_lm(seed)
    adamw = optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1,
                        mask=decay_mask)
    tx = optax.chain(clipped(adamw, num_steps, max_norm), ema_of_weights(alpha))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    norms = optimizer.opt_state[0][0]['norms'][...]  # Recorded by clipped
    clip_rate = (norms > max_norm).mean()
    return losses, model, averaged(model, optimizer), float(clip_rate)
```

```{.python .input #practice-a-worked-configuration-4}
%%tab pytorch, jax
data.batch_size = 256  # Where the throughput levels off
seed, short = new_seed(), {}
print('noise scale at initialization: '
      f'{d2l.noise_scale(tiny_lm(seed), sample_lm, b_small=8, b_big=256):.0f}')
for lr in [1e-3, 3e-3, 1e-2]:
    losses, model, _, clip_rate = train(seed, lambda t: lr, num_steps=300)
    short[lr] = (d2l.final_loss(losses), model)
    print(f'lr {lr:g}: loss {short[lr][0]:.3f} after 300 steps, '
          f'{clip_rate:.0%} of the steps clipped')
peak = min(short, key=lambda lr: short[lr][0])
print(f'peak rate {peak:g}; noise scale after its 300 steps: '
      f'{d2l.noise_scale(short[peak][1], sample_lm, b_small=8, b_big=256):.0f}')
```

```{.python .input #practice-a-worked-configuration-5}
%%tab pytorch, jax
num_steps = 3000
val_loss = lambda model: math.log(d2l.lm_perplexity(model, data.val_dataloader()))
print(f'{"validation loss":<16}{"configured":>11}{"its EMA":>9}'
      f'{"clipped":>9}{"baseline":>10}{"gap":>7}')
results = []
for seed in [new_seed() for _ in range(2)]:
    _, model, ema, clip_rate = train(
        seed, warmup_cosine(peak, num_steps // 50, num_steps), num_steps)
    _, baseline, _, _ = train(seed, lambda t: peak, num_steps,
                              max_norm=math.inf)
    results.append((val_loss(model), val_loss(baseline)))
    print(f'{"seed " + str(len(results)):<16}{results[-1][0]:11.3f}'
          f'{val_loss(ema):9.3f}{clip_rate:9.1%}{results[-1][1]:10.3f}'
          f'{results[-1][1] - results[-1][0]:7.3f}')
(c1, b1), (c2, b2) = results
print(f'spread between the seeds: configured {abs(c1 - c2):.3f}, '
      f'baseline {abs(b1 - b2):.3f}')
```

## Summary

A large training run is configured once, guarded against rare failures, and
judged by its final weights. Recent pretraining reports disclose recurring
settings (AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 0.1,
clipping at 1, warmup and a later decay) that make a defensible start, and
with a small budget the peak learning rate is the setting to tune.
Global-norm clipping with a threshold above the typical gradient norm
fires only on the first few steps, where the gradients of the fresh
initialization are largest. Yet it turns a diverging `TinyLM` run at
$\eta = 1$ into a stable one, acting as an automatic warmup, and in long
runs it also catches the rare oversized gradients of heavy-tailed noise.
Spikes that get through are handled by rewinding to a checkpoint and
skipping the data seen around them. An exponential moving average of the
weights reduces the noise in the
evaluated parameters without changing the learning rate, and at a constant
rate it reaches the accuracy of a decayed schedule. A comparison of
optimizers is evidence only when each arm's nuisance hyperparameters were
retuned and the protocol is stated.

## Exercises

<!-- WRITER: Exercise 4 needs updating to the new cell: the EMA now starts after epoch 5 (so the weight on the initial weights in part 1 becomes the weight on the epoch-5 weights after 10 epochs, about 2,350 steps), and part 3 (EMA on a cosine-decayed run) is now done in the text: the cell prints it, and the EMA ends at or 0.1 to 0.3 points below the decayed run's final live weights (lag after decay). Replace part 3, e.g. by starting the average at epoch 1 or 10, or by an equal-weight tail average. -->

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
       $\eta = 1$, once over the first 20 steps and once over the first 100
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

<!-- WRITER: slide material changed by the code step. `@!practice-a-nan-averted-2` now prints a table of four settings over two seeds (lr 1 unclipped / clipped at 1, lr 0.3 unclipped, lr 3 clipped at 1); `@!practice-a-nan-averted-3` plots all four over the first 50 steps. `@!practice-weight-averaging-3` now shows FashionCNN (no LeNet-BN): constant rate live and EMA (from epoch 5) against a cosine decay of the same length; `@!practice-weight-averaging-2` prints both schedules for two seeds. The new worked configuration has outputs at `practice-a-worked-configuration-2` (throughput per batch size), `-4` (noise scale, 300-step grid, chosen peak) and `-5` (final runs: configured, its EMA, clip rate, baseline, gap, spread), for the closing slide sequence. Kicker "LeNet-BN, Fashion-MNIST, η = 0.3" and the recap's "reaches a decayed schedule's accuracy" need the new numbers (see the WRITER notes in the body). -->

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

**Running example:** `TinyLM` with SGD at $\eta = 1$, the rate that
diverged in the Adam deck; then a LeNet with batch normalization at a
constant rate.
:::

::: {.slide title="The Last Section Puts the Chapter's Tools on a Long Run"}
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

::: {.slide title="At η = 1, Clipping Separates Divergence from Training"}
[Gradient clipping · TinyLM, SGD with momentum 0.9]{.kicker}

@!practice-a-nan-averted-3

- Unclipped: from the second step on, the loss grows by several orders
  of magnitude and overflows within about 20 steps.
- Clipped, same initial weights: final loss near the best SGD run at
  about a third of this rate.
:::

::: {.slide title="Clipping Fired Only on the First Few Steps"}
[Gradient clipping]{.kicker}

@!practice-a-nan-averted-2

- A few dozen firings at most, all within the first 40 steps, where
  the fresh initialization has its largest gradients: an **automatic
  warmup**.
- Afterwards a **guard**, idle at $\eta = 1$; in long runs it also
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
  a warmup, and saved `TinyLM` at $\eta = 1$; spikes that get through:
  rewind and skip.
- **Average:** an EMA ($\alpha = 0.999$) at a constant rate reaches a
  decayed schedule's accuracy.
- **Compare:** retune nuisances per arm; a gap below the seed spread is
  no result.

**Next chapter:** attention, the mechanism inside `TinyLM`.
:::
