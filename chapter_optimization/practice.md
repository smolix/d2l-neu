# Practice
:label:`sec_practice`

We have been spoiled. Every run in this chapter took a minute or two. When a
learning rate looked wrong we tried four others, when a comparison was close
we repeated it with new seeds, and when a loss blew up we started again. The
runs you care about will not be that cheap. Once training takes a day on the
one GPU you have, a sweep over four learning rates costs four days, so most
settings are fixed once and never tuned on the model that matters. A single
oversized update can undo hours of progress. And the weights at the end are
one draw from a noisy trajectory, yet they are the ones that ship.

The largest runs show these problems most plainly. PaLM, a language model
with 540 billion parameters trained on 6,144 TPU chips, saw its loss spike
about twenty times, although gradient clipping was enabled
:cite:`chowdhery2022palm`. But the problems do not wait for a data center.
They begin with the first run that you cannot afford to repeat fifty times,
which on a single GPU is a run of a few hours.

How do we guard, finish and configure a run that we get to do once? Very
little, it turns out, has to be decided on the long run itself. A guard
catches the rare step that would ruin it, an average of its weights takes
the luck out of where it stops, and its settings come from short runs of
the same model. We take the three in this order. Gradient clipping and
weight averaging are tools that the chapter has not taught yet. The
configuration comes last because it uses everything that the chapter has
taught: we take a `TinyLM` that no section has trained and decide its batch
size, its optimizer, its learning rate and its schedule.

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

## Gradient Clipping

In :numref:`sec_adam`, SGD with momentum had a single usable learning rate
on `TinyLM`. It trained best at $\eta = 0.3$, most runs at $0.5$ diverged,
and at $\eta = 1$ all of them did. :numref:`sec_test_models` showed how such
a run dies: fast, within its first handful of steps. This leaves a question
that the sweep could not answer. Is $\eta = 1$ too large for this model, or
only for its first steps?

Gradient clipping can tell the two apart, because it shortens the steps that
are too long and leaves all others alone. We met it in
:numref:`sec_rnn-scratch`, where it held the exploding gradients of a
recurrent network in check and where we called it a rather crude device. It
treats the gradients of all parameters as one long vector $\mathbf{g}$ and
rescales this vector whenever its length exceeds a threshold $\theta$:

$$
\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}.
$$
:eqlabel:`eq_practice_clip`

A gradient shorter than $\theta$ passes unchanged. A longer one keeps its
direction and is cut to length $\theta$, so at $\theta = 1$ a gradient of
norm 5 is multiplied by 0.2. Because the norm is taken over all parameters
at once, every layer is scaled by the same factor: clipping preserves the
relative sizes of the gradients of different layers and shrinks only the
step as a whole. Under plain SGD, whose step is $\eta$ times the gradient,
it is a cap on the length of a step: none is longer than $\eta\theta$.
Momentum raises the cap without removing it (Exercise
:exref:`ex_practice-clipping-regimes`). Crude it is: it knows nothing about
curvature and only refuses to follow a gradient beyond a fixed distance. At
the start of a run, we will see, that refusal is what counts.

:begin_tab:`pytorch`
`clip_grad_norm_` rescales the gradients in place and returns the norm that
it measured before clipping. `Clipped` calls it before every step of the
optimizer it wraps and keeps the norms.
:end_tab:

:begin_tab:`jax`
In Optax, clipping is a gradient transformation, and `optax.chain` places
it in front of the optimizer. `record_norms` is a second transformation,
which passes the gradients through unchanged and stores the norm of every
step in its state.
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

### Clipping a Divergent Run
:label:`subsec_practice-nan`

We rerun SGD with momentum 0.9 on `TinyLM`, the two-block transformer of
:numref:`subsec_tinylm`, for the usual 2,000 steps. The first two settings
run $\eta = 1$, the rate at which every run of :numref:`sec_adam` diverged,
without clipping and clipped at $\theta = 1$. This is the threshold of the
large published runs that we meet below, and the experiment will tell us
whether it suits our model. The third setting is $\eta = 0.3$ without
clipping, the best rate of that sweep. The fourth, $\eta = 3$ clipped at
$\theta = 1$, asks how far clipping can be pushed. Each setting is trained
from two seeds, and within a seed all four start from the same initial
weights. In `run`, an infinite threshold switches clipping off.

What do the two answers to our question predict? If $\eta = 1$ is too large
only for the first steps, where a fresh network has its largest gradients,
clipping should act on those steps and then fall silent, and the run should
train. If it is too large throughout, clipping will act all along. The table
therefore prints, for every clipped run, the number of steps that clipping
changed, the last such step, the median gradient norm, and the largest norm
after step 100 divided by the median.

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

Without clipping, both runs at $\eta = 1$ end as `inf`. With it, both train,
to a loss of $1.61$ to $1.64$. In each seed that is a few hundredths of a
nat below the run at $\eta = 0.3$. We claim no more than a tie, since two
runs of SGD at one rate differed by as much in :numref:`sec_adam`.

What the table does settle is our question. Clipping changed a few dozen of
the 2,000 steps at most, the last of them before step 50, and then fell
silent. It had no occasion to speak again. The median gradient norm of the
clipped runs at $\eta = 1$ is about a quarter of the threshold, and after
step 100 the largest norm is about twice the median, roughly half the
threshold. The rate $\eta = 1$ was too large for the first steps of the run
and for nothing else.

What do those first steps look like? Divergence is fast, so we plot the
first 50 steps of the first seed on a logarithmic scale.

```{.python .input #practice-a-nan-averted-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 51)),
         [runs[seeds[0], lr, max_norm][0][:50] for lr, max_norm in settings],
         'step', 'training loss', yscale='log', figsize=(5, 2.5),
         legend=[f'lr {lr:g}, clip at {max_norm:g}' for lr, max_norm in settings],
         legend_loc='outside')
```

The unclipped run at $\eta = 1$ falls for a step or two and then feeds on
itself. An oversized step moves the parameters to where the loss is steeper,
the gradient there is larger, and momentum carries the excess into the
following steps. The loss grows by several orders of magnitude and leaves
the plot within about twenty steps. On this scale the clipped run at the
same rate can hardly be told from the run at $\eta = 0.3$. The dotted
curve belongs to the run at $\eta = 3$, to which we return below.

The table and the plot change how the sweep of :numref:`sec_adam` should be
read. Its diverging grid points did not mark a rate at which the model
cannot be trained. They marked the rate that the first few dozen steps
cannot survive, and a cap on those steps moved the limit of SGD from about
$0.5$ to somewhere between 1 and 3. When a grid point of your own sweep
diverges, ask when it diverged. A run that dies in its first twenty steps
has told you about its start. About the two thousand steps that it never
took, it has told you nothing.

### Guard and Brake

Here clipping did the job of a warmup, and an adaptive one. It shortened
exactly those first steps that were too long, the steps on which a fresh
network has its largest gradients, until the parameters reached a region
where the full step is safe, and it needed no schedule to tell it when to
stop. Exercise :exref:`ex_practice-clipping-regimes` sets it beside the
ramp of :numref:`subsec_scheduler-warmup`, which works by the clock.

Used in this way, clipping is a *guard*: it leaves nearly every update alone
and rescales only the rare oversized one. The last setting shows the other
regime. At $\eta = 3$ clipping acts on practically every step, for as long
as the loss is finite, and the run fails all the same. Its loss, the dotted
curve of the plot, climbs from the start and ends far above the $5.5$ nats
of a model that guesses its bytes uniformly. When every gradient is cut to
length $\theta$, the length of a step no longer depends on the size of the
gradient. That is a different algorithm from SGD, run here at a step that
is too long everywhere. In this regime clipping is a *brake*, and a brake
that is on for the whole journey is a sign that the speed was chosen
wrongly. Clipping repairs steps that are too long for a moment. It does not
repair a learning rate that is too large throughout.

To set the threshold, measure. Record the gradient norm of a run that
trains, as we did, and put $\theta$ a few times above its median; ours was
four times above. From then on, log the fraction of clipped steps in every
run. A guard clips a few percent of the steps at most, and most of them
early. If clipping fires on most steps, the learning rate and the threshold
act together on the length of the step :cite:`Godbole.Dahl.Gilmer.ea.2023`,
and the first thing to try is a smaller learning rate.

In long runs a guard is credited with a second job. The gradient noise of
language models is reported to be heavy-tailed, and under such noise
clipped SGD converges where plain SGD can fail
:cite:`Zhang.Karimireddy.Veit.ea.2020`. Our run shows no such tail, and we
pass the finding on as theirs. The last column of our table is the way to
check it on a model of your own.

### Clipping and Adam

Under Adam the cap on the step is already there. Adam keeps two leaky
averages (:numref:`sec_adam`): $\mathbf{m}$ of the gradients, with weight
$1 - \beta_1$ on each new gradient, and $\mathbf{v}$ of their squares, with
weight $1 - \beta_2$. Suppose a gradient coordinate $g$ arrives that dwarfs
every earlier one. After this step $m \approx (1 - \beta_1) g$ and
$v \approx (1 - \beta_2) g^2$. The coordinate therefore moves by about
$\eta (1 - \beta_1)/\sqrt{1 - \beta_2}$, or $3.2\,\eta$ at the defaults
$\beta_1 = 0.9$ and $\beta_2 = 0.999$, however large $g$ is. Clipping $g$
changes this one step little, as long as the clipped gradient still
dominates. For the same reason clipping cannot stand in for a lower
learning rate under Adam: a step whose length ignores the size of the
gradient is not shortened by shrinking the gradient.

What clipping protects is Adam's state. Let a minibatch whose gradient is a
hundred times as long as the typical one pass unclipped. Its square adds
$(1 - \beta_2) \cdot 100^2 = 10$ times the usual value to $v$. Now $v$
stands at 11 times its usual level, every step is divided by $\sqrt{11}$,
and the model moves at 30% of its usual pace. The excess fades only by the
factor $\beta_2$ per step: $10 \cdot 0.999^t$ falls below 1 after about
2,300 steps, and the steps are then still at 70%. One bad minibatch has
slowed the model for thousands of steps. Clipped at four times the median
norm, as in our runs, the same gradient adds $0.001 \cdot 4^2$, less than
2%. Since the norm is global, this protection extends only to spikes that
lift the norm of the whole gradient above $\theta$.

With $\beta_2 = 0.95$, the choice of the large runs in
:numref:`tab_practice_recipes` below, the damage is deeper and much
shorter. The bound on a single step falls to
$0.1/\sqrt{0.05} \approx 0.45$ times $\eta$, but an unclipped spike enters
$v$ with fifty times the weight. The same gradient adds
$0.05 \cdot 100^2 = 500$ times the usual value, cuts the steps to less than
a twentieth of their size, and is forgotten within about 120 steps.

### Loss Spikes and Recovery
:label:`subsec_practice-spikes`

Clipping does not prevent every failure, as the twenty spikes of PaLM
showed. They came at irregular intervals, sometimes late in training, and
smaller models trained in the same way had none. A second precaution was on
throughout, a small penalty that keeps the output logits from drifting as a
whole (the *z-loss* of Exercise :exref:`ex_practice-logit-drift`), and it
did not prevent them either :cite:`chowdhery2022palm`.

The remedy was operational. The PaLM team restarted from a checkpoint about
100 steps before each spike and skipped the 200 to 500 minibatches seen
around it, after which the loss did not spike again at that point. The
skipped data was not bad in itself: replayed from a different, earlier
checkpoint, the same minibatches caused no spike. It took a particular
combination of data and parameter state, and why that combination, the
report does not say. We can still prepare for a failure that we cannot
explain. Checkpoints must be frequent enough that a rewind costs hours and
not days, and each must hold everything needed to resume: the model, the
optimizer state with Adam's two averages (:numref:`subsec_adamw-memory`),
the step count that drives the schedule, and the position in the data
stream.

## Weight Averaging
:label:`subsec_practice-weight-averaging`

Suppose the run has survived. Which weights do we keep? The last ones are
the obvious answer, and they are one draw. At a constant learning rate SGD
does not settle at a point: its iterates keep fluctuating in the noise ball
of :numref:`sec_sgd`. A schedule shrinks the ball by decaying the rate
(:numref:`sec_scheduler`). Averaging leaves the learning rate alone:
fluctuations around the center of the ball cancel in a mean.

They cancel only if the averaged iterates are different draws, and
successive iterates are anything but. On the valley of the earlier
sections, a coordinate with curvature $c$ keeps the fraction $1 - \eta c$
of its position at every step and needs about $1/(\eta c)$ steps to forget
where it was. (This $c$ is the $\lambda$ of :numref:`sec_sgd`; here
$\lambda$ will be the weight decay.) Along the flat direction, where
$c = 0.2$, that is 50 steps at $\eta = 0.1$. An average of the last ten
iterates is then an average of ten copies of nearly the same point. A
useful window holds many such stretches, which makes it hundreds or
thousands of steps long.

### The Exponential Moving Average

The recipe of :numref:`sec_training_recipes` evaluated an exponential moving
average (EMA) of its weights in place of the weights themselves. An EMA
keeps a running average $\bar{\mathbf{x}}_t$ of the weights $\mathbf{x}_t$.
We write its decay as $\alpha$, because $\beta$ already denotes momentum:

$$
\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1 - \alpha)\, \mathbf{x}_t.
$$
:eqlabel:`eq_practice_ema`

This is the leaky average that momentum applies to gradients and Adam to
squared gradients, now applied to the weights themselves, and it reaches
back about $1/(1 - \alpha)$ steps. Training never sees
$\bar{\mathbf{x}}_t$. It goes on updating $\mathbf{x}_t$, which we call the
*live* weights, and only evaluation uses the average. The cost is one more
copy of the parameters in memory.

### Averaging at a Constant Learning Rate

`TinyLM` would be a poor test of the average. Its runs end short of steps,
with the loss still falling (:numref:`sec_test_models`), and there is no
noise ball to average over. We turn to `FashionCNN`, the image classifier
of the same section, trained as there with plain SGD at the constant rate
$0.3$ on minibatches of 256 images, now for 15 epochs of 235 steps.
Alongside the live weights the run keeps an EMA with $\alpha = 0.999$, whose
window of 1,000 steps spans a little over four epochs.

```{.python .input #practice-weight-averaging-1}
%%tab pytorch, jax
fashion = d2l.FashionMNIST(batch_size=256)
num_epochs, num_batches = 15, len(fashion.train_dataloader())
test_acc = lambda model: d2l.evaluate_accuracy(model, fashion.val_dataloader())
```

When should the average start? One that begins at the first step spends its
first epochs remembering the initialization: after one epoch of 235 steps
the initial weights would still carry the weight
$0.999^{235} \approx 0.79$. This is lag, not the bias toward zero that Adam
corrects (:numref:`sec_adam`). The average starts as a copy of the weights,
so its coefficients already sum to one, and there is nothing to correct. The
cure for lag is to start late, though not so late that the copy it starts
from has no time to fade. We start after epoch 5, which leaves ten epochs,
more than two windows.

For comparison, a second run lowers the same rate to zero along a cosine
over the same 15 epochs, from the same initial weights. If averaging and
decay remove the same noise, the average of the constant-rate run should end
where the decayed run ends. `warmup_cosine(peak, warmup, num_steps)` is a
linear ramp over `warmup` steps, none here, followed by the cosine decay of
:numref:`sec_scheduler`. Both schedules are run from two seeds, and the
table prints test accuracies.

:begin_tab:`pytorch`
`AveragedModel` holds the averaged copy. A hook that the optimizer calls
after every step advances the schedule and, from epoch 5 on, updates the
average, whose first update copies the live weights.
:end_tab:

:begin_tab:`jax`
`ema_of_weights` is an Optax transformation chained behind the optimizer.
It keeps the average in its state, and `averaged` writes that average into
a copy of the model.
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

Read the first column first. At the constant rate the live weights do not
hold still: within the last five epochs their accuracy ranges over two to
six points, and the final value is whichever point of that range the last
epoch happens to produce. The EMA of the same run ends a little above 91%
every time, at the top of that range or above it. In the plot, of the first
seed, it trails the live weights for two or three epochs while its window
fills, and then runs smoothly along the upper edge of the live curve. The
live curve shows the noise ball directly: each epoch evaluates a different
point in it, while the EMA evaluates close to its center.

The decayed run ends at about 91% as well, within about half a point of the
averaged one. Two runs of one setting differ by a point on `FashionCNN`
(:numref:`sec_test_models`), so this is a tie, and it is the tie that the
noise ball predicted. The cosine rows also show where averaging ends. After
a decay to zero the EMA does not improve on the final weights. In each of
our runs it ends a little behind the final weights of the same run, by a
few tenths of a point at most. The decay has already removed the noise, and
what is left of the average is its lag.

### Averaging and Decay

Decay and averaging thus reach the same accuracy by different routes. Decay
reduces the noise injected into future iterates and so changes the
trajectory. Averaging leaves the trajectory alone and reduces the noise in
the parameters that we evaluate.

When is an average the tool, then? When no decay is planned, or when a
model is needed before the decay has happened. In the warmup–stable–decay
runs of :numref:`sec_scheduler`, a checkpoint of the constant phase became
a finished model only through a *branch*, a copy of the run that decays its
rate over a few hundred further steps. The EMA at that checkpoint can be
evaluated without them. After a full decay the average adds little, but it
also costs little, and large runs take it anyway. The released Llama 3
model is an average of checkpoints from the final phase of training, during
which the learning rate was annealed to zero
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Checkpoints that are averaged
must lie far enough apart to be different draws.

## A Worked Configuration

We can now do what the chapter has been preparing: take a model that nobody
has tuned and decide everything. The model is a `TinyLM` with four blocks of
width 256 and four attention heads. It has 3.3 million parameters, seven
times as many as the `TinyLM` of the earlier sections, and no section has
trained it. The run that we pretend we cannot repeat lasts 3,000 steps. It
takes a minute or two here; imagine a week. Whatever we want to know, we
must learn from much shorter runs.

### Reported Configurations

A new run need not start from nothing, because others have trained models of
the same kind and written down what they used.
:numref:`tab_practice_recipes` collects the optimizer settings disclosed for
four large pretraining runs of 2024 and 2025.

:What four pretraining runs disclose about their optimizer configuration
(for OLMo 2, the moment coefficients are from its released training
configuration). A dash means that we found no statement. Batch sizes are
quoted as each report quotes them, in tokens or sequences.
:label:`tab_practice_recipes`

| run | optimizer | $\beta_1, \beta_2$ | peak LR and schedule | warmup | clip | weight decay | batch size |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` | AdamW | — | $8{\times}10^{-5}$, cosine to $8{\times}10^{-7}$ | 8k steps | — | — | ramp 4M $\to$ 8M $\to$ 16M tokens |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, constant to 10T tokens, cosine to $2.2{\times}10^{-5}$ | 2k steps | 1.0 | 0.1 | ramp 3,072 $\to$ 15,360 seqs over 469B tokens |
| OLMo 2 7B :cite:`OLMo.2025` | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, cosine toward 10% of peak over 5T tokens, cut at 4T | 2k steps | 1.0 | 0.1, embeddings exempt | 1,024 seqs, fixed |
| Kimi K2 :cite:`Kimi.Team.2025` | MuonClip | — | $2{\times}10^{-4}$, constant to 10T tokens, cosine to $2{\times}10^{-5}$ | 500 steps | — | 0.1 | 67M tokens, fixed |

Wherever a row speaks, it says much the same: AdamW with
$(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay $0.1$, clipping at a global
norm of 1, a warmup, and a later decay along a cosine. Only the optimizer
of Kimi K2 departs. MuonClip is the Muon of :numref:`sec_muon`, together
with the cap on attention scores described there. Of the shared settings,
$\beta_2$ needs a word, because until now every Adam of this chapter ran at
the default $0.999$. At $0.95$, Adam's average of the squared gradients
reaches back about $1/(1-\beta_2) = 20$ steps instead of a thousand, so it
follows a change in the scale of the gradients quickly and, as we saw,
forgets a spike quickly. What varies is the peak learning rate, which every
team set for itself. The smallest belongs to the widest model, Llama 3 405B,
which is the direction in which :numref:`sec_scaling` saw the best rate
drift. These settings are a defensible start, because each was chosen by a
team that could test it at scale. They are not proven optima, because the
reports rarely ablate them. The dashes record missing evidence, not implied
defaults.

### Batch Size

The first decision is the batch size, because the learning rate depends on
it. The procedure of :numref:`sec_batch_size` gives the batch a floor and a
ceiling. The floor is the batch that fills the device: below it a step
reads fewer examples and is hardly any faster. The ceiling is the *noise
scale*, the batch size at which the noise in a minibatch gradient is as
large as the gradient itself. Far above it, extra examples save few steps.
To find the floor, we time one gradient of the model at batch sizes from 32
to 1,024 sequences. `tiny_lm` builds the model, and `sample_lm(b)` draws
`b` windows of text.

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

def sample_lm(b):  # b windows at random offsets in the training text, on the GPU
    X, Y = data.windows(data.train,
                        rng.integers(len(data.train) - data.num_steps, size=b))
    return jnp.asarray(X), jnp.asarray(Y)
```

```{.python .input #practice-a-worked-configuration-2}
%%tab pytorch, jax
model = tiny_lm(new_seed())
print(f'{d2l.num_params(model):,} parameters')
for b in [32, 64, 128, 256, 512, 1024]:
    X, Y = sample_lm(b)
    t = d2l.Benchmark(lambda: d2l.grad_sq_norm(model, X, Y)).time
    print(f'batch {b:5d}: {1000 * t:5.1f} ms per gradient, '
          f'{b / t:6.0f} sequences per second')
```

From 32 to 128 sequences the batch quadruples and the time of a gradient
grows far less, because the device has capacity to spare. From 256 on, the
time doubles with every doubling of the batch, and the number of sequences
per second no longer rises. Somewhere between 128 and 256 sequences the
device fills up. We take 256 as the floor, the first of our batch sizes at
which it is certainly full. The ceiling must wait for the short runs below,
because the noise scale changes as a model trains.

### Optimizer and Precautions

For the optimizer we follow the table: AdamW with
$(\beta_1, \beta_2) = (0.9, 0.95)$ and $\lambda = 0.1$, the decay applied to
the weight matrices only (:numref:`subsec_adamw-what-not-to-decay`), and
gradients clipped at a global norm of 1. Two of these settings we adopt with
our eyes open. The decay can do little in a run of this length: at learning
rates of a few thousandths its timescale $1/(\eta\lambda)$ is thousands of
steps (:numref:`sec_adamw`), as long as the run itself. We keep it because
the run stands in for a long one. And whether a threshold of 1 is a guard on
this model we do not know yet, so `train` returns the fraction of clipped
steps. It also keeps an EMA of the weights with $\alpha = 0.995$. The window
of 200 steps is short on purpose: the run will end with a decay, and a long
window would reach back to weights that the decay has since improved on.

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

### Peak Learning Rate

What remains is the setting that every report chose for itself, and our
whole tuning budget goes to it: three runs of 300 steps at constant rates a
factor of three apart. A tenth of the final run is long enough to tell the
three cases of :numref:`sec_test_models` apart, a rate that is too low, one
that trains and one that is too high. Which three rates? On the small
`TinyLM` the usable rates of Adam spanned a factor of ten around the best
one, $0.003$ (:numref:`sec_adam`), and a wider model prefers the same rate
or a smaller one (:numref:`sec_scaling`). The grid $0.001$, $0.003$, $0.01$
should therefore contain the best rate and show the edge above it. (With no
smaller relative to consult, start from the wider grid of
:numref:`sec_adam`.) The cell also prints the noise scale, at
initialization and for the model trained at the rate it chose.

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

The three runs show the three cases, in the form that Adam gives them. At
$0.001$ the run trains, slowly. At $0.01$ it stalls
near $2.5$ nats, since under Adam a rate that is too high shows as a stalled
loss and not as `inf` (:numref:`sec_adam`). This run was clipped more often
than the other two and stalled all the same: clipping does not rescue Adam
from a learning rate that is too large. The rate $0.003$ is the best by
about a tenth of a nat or more, and we take it as the peak. It lies a
factor of three below the first rate that stalls, at the near end of the
factor of three to ten that :numref:`subsec_scheduler-warmup` gave as a
rule of thumb. In all three runs clipping acted on a few percent of the
steps, the share of a guard. The threshold of the published runs suits
this model too.

The sweep took 900 steps, less than a third of the run it prepares, and it
cut two corners. It ran from a cold start, where
:numref:`subsec_scheduler-warmup` advised sweeping behind the ramp. A cold
sweep is the stricter test. What it may miss is a higher rate that only a
ramp rescues, and in that section such a rate still ended more than a
tenth of a nat behind the best one. The bolder corner is the assumption
that the best constant rate of 300 steps is the right peak for 3,000 steps
that end in a decay. Short runs flatter large rates, which are the fastest
off the mark: had the lowest loss sat at the top of our grid, we would have
extended the grid before believing it. Exercise
:exref:`ex_practice-worked-configuration` tests the assumption.

Three runs tune one number. With tens of runs, add the weight decay, and
search over the rate and the product $\eta\lambda$, which sets the
timescale of the decay (:numref:`sec_adamw`).

The noise scale settles the batch size. It is about 2 sequences at
initialization and of the order of a hundred after 300 steps. Our floor of
256 sequences is already at or above this ceiling, so we stay at the floor.
A larger batch would spend its extra examples on averaging noise that is
already smaller than the gradient.

### Schedule

Two decisions remain, and :numref:`sec_scheduler` prepared both. The first
is the warmup: a linear ramp over the first 60 steps, 2% of the run, where
one to a few percent is customary. It gives up the budget of 30 steps at
the peak, the *budget* of a schedule being the sum of its learning rates,
which sets how far the weights can travel. Does this run need a ramp at
all? The sweep trained at $0.003$ from a cold start, so perhaps not. But
one seed is thin evidence and the ramp is cheap.

The second is the decay. In that section a late decay, which holds the
peak and falls linearly over the last fifth of the run, beat the constant
rate where the cosine only tied with it. Its main merit, that the length of
a run can be decided late, counts for nothing here, since our length is
fixed. We take the cosine, the shape of all four rows of
:numref:`tab_practice_recipes`, and we take it to zero, where that section
found that a schedule should end. Whether the late decay would have ended
lower on this model is the first question of Exercise
:exref:`ex_practice-worked-configuration`.

### Logs and Records

One more decision comes before the run, and it is easily forgotten: what
the run writes down. Besides the loss, a long run should log four curves,
each of which separates failures that look alike in the loss. The gradient
norm before clipping tells a rough start from a spike in mid-run. The share
of clipped steps tells a guard from a brake. The learning rate shows
whether a drop of the loss was the schedule's doing. And the size of an
update relative to the weights exposes a cold start: in
:numref:`subsec_scheduler-warmup` the first step of a stalled run changed
the weights by two thirds of their size. :numref:`tab_practice_symptoms`
reads the failures of this chapter off these curves. Our own `train`
returns only the losses and the share of clipped steps, which is enough
for two minutes.

:Reading the log of a run: the failures of this chapter and what to try
first.
:label:`tab_practice_symptoms`

| in the log | reading | first remedy |
|:--|:--|:--|
| the loss leaves in the first steps, gradient norms climbing | steps too long for the initialization | a warmup or clipping |
| the loss stalls high under Adam; the first updates were a large fraction of the weights | a start too rough | a warmup |
| the loss stalls high under Adam behind a warmup | a rate too large throughout | one grid point down |
| clipping acts on most steps | a brake, not a guard | a lower rate, then a look at the threshold |
| the loss has stopped falling and jitters | the noise level of the current rate | decay, or average |
| the loss is smooth and still falling at the end | a slow run | a higher rate or more steps |

The records of a project outlive its runs. For every run, write down the
full configuration, including the settings you consider fixed, the code
version, the seed, and the one thing you changed. Keep the runs that
diverged and note when they diverged; we have seen what that one number is
worth. Give every arm of a comparison a tuning of its own. When ten
optimizers were retuned alike, the speedups over AdamW claimed for the new
ones, up to a factor of two, shrank to tens of percent
:cite:`Wen.Hall.Ma.ea.2025`, and our own Muon, with a grid for each arm,
ended level with AdamW (:numref:`sec_muon`). And write the conclusion next
to the curves while the experiment is fresh. A directory of loss curves
without sentences becomes unreadable within weeks.

### The Final Run

Now the run itself. Here we drop the pretence for a moment, since our week
is really two minutes: the cell does the run twice, because a difference
counts only against the spread between two runs of one setting
(:numref:`sec_test_models`). For each of two seeds it trains the configured
run and a baseline that differs in the schedule alone, the same AdamW with
the same clipping at the constant rate $0.003$. The table prints validation
losses in nats per byte.

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
    _, baseline, _, _ = train(seed, lambda t: peak, num_steps)
    results.append((val_loss(model), val_loss(baseline)))
    print(f'{"seed " + str(len(results)):<16}{results[-1][0]:11.3f}'
          f'{val_loss(ema):9.3f}{clip_rate:9.1%}{results[-1][1]:10.3f}'
          f'{results[-1][1] - results[-1][0]:7.3f}')
(c1, b1), (c2, b2) = results
print(f'spread between the seeds: configured {abs(c1 - c2):.3f}, '
      f'baseline {abs(b1 - b2):.3f}')
```

The configured run ends at a validation loss of about $1.25$, the baseline
at $1.30$. The gap of $0.05$ to $0.06$ nats is many times the difference
between two seeds of one setting, which stays under a hundredth, and it
meets the $0.05$ that :numref:`sec_test_models` asks for. Nothing but the
schedule differs, and the ramp touches 60 steps of 3,000 at a rate that
trains without one, so most of the gap must be the decay's.

This should surprise you a little, because in :numref:`sec_scheduler` the
cosine decay only tied with the constant rate. Between that run and this
one nearly everything has changed: the model is seven times as large, the
batch four times, the run half as long again, and $\beta_2$, the weight
decay and the clipping are new. Our table cannot say which change turned a
tie into a result, and we will not guess. It says enough for the decision
at hand. On this run the cosine decay was worth $0.05$ nats, where its
earlier tie would have told us to expect nothing. The best peak rate
carried over from the small model to the large one. The worth of a decay
did not.

What did the rest of the configuration do? For one setting the table
answers. The EMA agrees with the final weights to within a thousandth of a
nat, because after a decay to zero there is nothing left to average. For
the others it is silent. Clipping acted on fewer than 1% of the steps,
which shows a guard and not whether the guard was needed: at the start of
this section a few dozen clipped steps of 2,000 were the difference between
`inf` and a trained model. The ramp and the weight decay have no run
without them to be compared with. So two settings are known to have done
work: the peak rate, where a factor of three changed the loss of the sweep
by a tenth of a nat or more, and the decay. The others are precautions, and
Exercise :exref:`ex_practice-worked-configuration` finds out which of them
this run needed. On a run that we could repeat, we would find out and drop
the idle ones. On a run that we cannot repeat they stay.

Our procedure tuned one number by eye, on one GPU, and both limits are soon
reached. With several numbers to tune and dozens of runs to spend, the
search is better left to an algorithm (:numref:`chap_hyperopt`). And once a
model or its batch no longer fits on one device, a step is spread over many,
together with the optimizer state, which takes as much memory as the weights
and their gradients (:numref:`subsec_adamw-memory`). This chapter asked
which step to take. What a step costs, in seconds and in bytes, is the
question of :numref:`chap_performance`.

## Summary

A run that cannot be repeated is guarded, finished with an average or a
decay, and configured from short runs of the same model. Global-norm
clipping caps the length of a step. At $\theta = 1$ it acted on a few dozen
steps, all of them before step 50, and that let SGD train `TinyLM` at a
rate where every unclipped run diverged; at three times that rate it acted
on practically every step and the run failed. Under Adam, whose step is
bounded anyway, clipping protects the average of the squared gradients from
a spike that would otherwise shorten the steps for thousands of steps. An
exponential moving average of the weights removes the noise of a constant
learning rate from the parameters that are evaluated. On `FashionCNN` it
matched a cosine-decayed run at about 91%, and after a full decay it added
nothing. For a new model we took the batch size from the device, the
optimizer settings from published runs, and the peak learning rate from
three runs a tenth as long as the final one. Warmup and cosine decay then
lowered the validation loss by $0.05$ nats, against less than $0.01$
between two seeds.

## Exercises

1. **The decay timescale.** The timescale of weight decay in
   :numref:`subsec_wd-at-scale` is $1/(\eta\lambda)$ steps, which for a
   batch of $B$ tokens and a dataset of $D$ tokens is
   $\tau = B/(\eta\lambda D)$ passes over the data. The DeepSeek-V3 and
   OLMo 2 runs of :numref:`tab_practice_recipes` use sequences of 4,096
   tokens and training sets of about 14.8 trillion and 4 trillion tokens.
    1. Compute $\tau$ for both rows at their peak learning rates, taking
       for DeepSeek-V3 its final batch size of 15,360 sequences. What
       fraction of each dataset do the weights remember?
    1. Following :citet:`Bergsma.Dey.Gosal.ea.2025b`, what should happen to
       $\lambda$ if the batch size were doubled at an unchanged learning
       rate and $\tau$ is to be preserved, and what if instead the dataset
       were doubled?

1. [code] **Clipping regimes.** The experiment of
   :numref:`subsec_practice-nan` clips at $\theta = 1$.
   :exlabel:`ex_practice-clipping-regimes`
    1. Sweep $\theta \in \{0.01, 0.1, 0.5, 1, 4, \infty\}$ at $\eta = 1$
       with `run`, recording the final loss and the fraction of steps that
       clipping changed. Classify each threshold as a guard (fires
       rarely), a brake (fires on most steps), or absent, and relate each
       class to whether the run diverges and to its final loss.
    1. Show that for SGD without momentum, clipping that fires on every
       step turns the update into normalized gradient descent with step
       size $\eta\theta$. Our runs use momentum, with velocity
       $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$, update
       $\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t$, and
       $\beta = 0.9$. Bound the length of one of their updates when
       clipping fires on every step. Explain why neither case is the same
       as training at a lower learning rate.
    1. Remove the clipping and instead warm the learning rate up linearly
       from 0 to $\eta = 1$, once over the first 20 steps and once over the
       first 100 steps. Does each run survive? Compare the two lengths with
       the last step on which clipping fired, and explain what the two
       remedies have in common and how they differ.

1. **Logit drift.** To the cross-entropy of its output logits $\mathbf{o}$,
   PaLM :cite:`chowdhery2022palm` adds the *z-loss* $10^{-4} \log^2 Z$,
   where $Z = \sum_j \exp(o_j)$ is the normalizer of the softmax.
   :exlabel:`ex_practice-logit-drift`
    1. Show that adding the same constant $c$ to every logit leaves the
       cross-entropy unchanged, and compute how the shift changes $\log Z$
       and the z-loss.
    1. Compute the gradient of $10^{-4} \log^2 Z$ with respect to
       $\mathbf{o}$. In which direction does a gradient step move the
       logits when $\log Z$ is large and positive? Show that the gradient
       of the cross-entropy alone has no component along the all-ones
       vector, and explain what this implies for the overall level of the
       logits during training.
    1. A model with 256 outputs has all logits equal to zero. Compute
       $\log Z$ and the z-loss. Which shift of the logits removes the
       penalty, and what does it do to the predicted probabilities?

1. [code] **The averaging window.** Sweep the decay
   $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ of the EMA in the
   constant-rate run of :numref:`subsec_practice-weight-averaging`, and
   plot the test accuracy of the EMA per epoch for each value.
   :exlabel:`ex_practice-averaging-window`
    1. Express each averaging window $1/(1-\alpha)$ in epochs. The average
       starts as a copy of the weights after epoch 5. Derive the weight
       that it still places on this copy at the end of the run, about
       2,350 steps later, and evaluate it for each $\alpha$.
    1. Use these numbers to explain the curves at the two ends of the
       sweep.
    1. With $\alpha = 0.999$, start the average at the first step
       (`start=0`) and after epoch 10 (`start=10`). Predict from the
       window how each curve will differ from the one in the text, and
       check.

1. [code] **The worked configuration, taken apart.** ● `train` takes the
   schedule, the number of steps, the threshold and the decay of the EMA as
   arguments. Use two seeds for every setting, and count a difference only
   if it is several times the spread between them.
   :exlabel:`ex_practice-worked-configuration`
    1. Replace the cosine by the late decay of :numref:`sec_scheduler`: hold
       the peak rate for the first four fifths of the run and then decay
       linearly to zero, with the same 60-step ramp. Which shape ends
       lower?
    1. Run the configured schedule without the ramp, and once more without
       clipping (`max_norm=math.inf`). Which of the two precautions does
       the validation loss notice?
    1. Run the configured schedule with its peak one grid point lower and
       one higher, at $0.001$ and at $0.01$. Did 300 steps at a constant
       rate choose the right peak for the run of 3,000 steps? Does the
       ramp rescue the rate that stalled from a cold start?
    1. The last cell of the section discards the EMA of the baseline.
       Evaluate it. How much of the gap of $0.05$ nats does an average of
       the constant-rate run close, with a window of 200 steps and with
       one of 1,000 steps ($\alpha = 0.999$)? Then stop the configured run
       after 1,500 of its 3,000 steps, with the schedule still planned for
       3,000, and compare the weights with their EMA at both windows. When
       does the average help, and when does it hurt?

1. [code] **A ten-run budget.** ● At a constant learning rate, the best
   Adam runs of :numref:`sec_adam` needed 650 to 800 steps to bring the
   training loss of the two-block `TinyLM` to 1.8, as `d2l.steps_to_reach`
   counts them. With a budget of ten runs of at most 1,000 steps each, on
   the same model and batch size, get there in as few steps as you can. You
   may use any optimizer, schedule and precaution of this chapter. Before
   you start, write down which setting you expect to matter most. Keep the
   records that this section asks for: for every run the configuration, the
   steps taken, and the one-sentence conclusion that you drew before
   launching the next run. Report these records, not just the winning
   configuration.

1. **Defaults against a small budget.**
   :citet:`Schmidt.Schneider.Hennig.2021` found that trying several
   optimizers at their default settings works about as well as tuning a
   single one. Test this on `TinyLM` with the final training losses that
   :numref:`sec_adam` and :numref:`sec_muon` print. (The runs of
   :numref:`sec_muon` add weight decay on the matrices; otherwise the
   protocol is the same.) As the default runs, take SGD with momentum $0.9$
   at $\eta = 0.1$, Adam at $\eta = 10^{-3}$, and the hybrid of Muon and
   AdamW at $\eta = 10^{-3}$, whose update :numref:`sec_muon` scaled to the
   size of an AdamW step. Compare the best of the three default runs with
   the best point of each of the four grids: SGD and Adam in
   :numref:`sec_adam`, AdamW and the hybrid in :numref:`sec_muon`. Which
   optimizers, tuned alone, beat the best default run, and is the margin a
   result by the rule of :numref:`sec_test_models`? How many runs did each
   of the two strategies spend?

[Discussions](https://d2l.discourse.group/t/36767)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.12]{.kicker}

Practice<br>
**gradient clipping · weight averaging · a worked configuration**
:::
:::

::: {.slide title="PaLM's Loss Spiked about 20 Times with Gradient Clipping On"}
[The problem]{.kicker}

A run that takes a day on one GPU:

- **Guard:** one oversized update can undo hours. PaLM, 540 billion
  parameters on 6,144 TPU chips: about 20 loss spikes, with clipping
  enabled.
- **Finish:** the final weights are one draw from a noisy trajectory.
- **Configure:** a sweep over four learning rates costs four days.

::: {.d2l-note .warn}
How do we guard, finish and configure a run that we get to do once?
:::

[Chowdhery et al., 2022]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- set a clipping threshold and tell a guard from a brake;
- compute how long a gradient spike stays in Adam's state;
- recover from a loss spike;
- choose the window and the start of a weight average;
- configure a new model: batch size, optimizer, peak rate, schedule.

**Idea:** the settings of a long run come from short runs of the same
model; a guard and an average cover what short runs cannot foresee.

**Running example:** `TinyLM` (transformer, 0.47 million parameters, next
byte of Wikipedia text) under SGD at a diverging rate; then a `TinyLM` of
3.3 million parameters, configured from nothing.
:::

::: {.slide title="Practice Assembles the Chapter into One Configured Run"}
[Where this fits]{.kicker}

| movement | sections |
|---|---|
| two difficulties, on a valley we can solve | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| which direction to step in, on two small networks | Two Test Models · Momentum · Adam · AdamW · Muon |
| how the step changes over a run, with batch size and width | Schedules · Batch Size · Scaling Up · **Practice** |

- **Uses:** SGD's diverging rates on `TinyLM`; Adam's two averages; the
  noise ball; warmup and cosine decay; the gradient-noise scale.
- **Skips:** automated hyperparameter search; training on several
  devices.
:::

::: {.slide title="Global-Norm Clipping Caps the Step and Keeps Its Direction"}
[Gradient clipping]{.kicker}

All gradients as one vector $\mathbf{g}$, threshold $\theta$:

$$\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

- $\|\mathbf{g}\|_2 = 5$, $\theta = 1$: multiply by 0.2.
- One norm over all layers: their relative sizes are kept.
- Plain SGD with learning rate $\eta$: no step is longer than
  $\eta\theta$.
:::

::: {.slide title="Two Readings of a Diverging Rate Predict Different Clip Counts"}
[Gradient clipping]{.kicker}

SGD with momentum 0.9 on `TinyLM`: best at $\eta = 0.3$, every run
diverges at $\eta = 1$.

| if $\eta = 1$ is too large | clipping at $\theta = 1$ acts | and the run |
|:--|:--|:--|
| for the first steps only | early, then never | trains at $\eta = 1$ |
| throughout | on most steps | is no longer SGD at $\eta = 1$ |

**Test:** $\eta = 1$ unclipped and clipped at $\theta = 1$, $\eta = 0.3$
unclipped, $\eta = 3$ clipped; two seeds, 2,000 steps; count the clipped
steps.
:::

::: {.slide title="At η = 1, Clipping Separates Divergence from Training"}
[Gradient clipping · TinyLM, first 50 steps]{.kicker}

@!practice-a-nan-averted-3

- Unclipped: the loss falls for a step or two, then grows by orders of
  magnitude and is gone within about 20 steps.
- Clipped at 1: hard to tell from $\eta = 0.3$ on this scale.
:::

::: {.slide title="Clipping Changed a Few Dozen Early Steps, Then Fell Silent" layout="tight"}
[Gradient clipping]{.kicker}

@!practice-a-nan-averted-2

[`last`: last clipped step. `largest / median`: after step 100.]{.small}

::: {.d2l-note}
$\eta = 1$ was too large for the first 50 steps only, and SGD's limit
moved from about 0.5 to between 1 and 3. When a run of a sweep diverges,
ask when.
:::
:::

::: {.slide title="A Guard Fires Rarely; a Brake Fires Always and Changes the Algorithm"}
[Gradient clipping]{.kicker}

| | guard ($\eta = 1$) | brake ($\eta = 3$) |
|:--|:--|:--|
| clipped steps | a few dozen of 2,000 | practically all |
| gradient norm | median $\theta/4$ | tens of $\theta$ |
| update | SGD, a few steps shortened | fixed length: normalized descent |
| run | trains | fails |

- **Threshold:** a few times the median norm of a run that trains. Log
  the share of clipped steps.
- **Fires on most steps:** lower $\eta$ first.
:::

::: {.slide title="Check: Does Clipping Shorten an Adam Step?"}
[Check]{.kicker}

One gradient coordinate $g$ arrives that dwarfs all earlier ones. Adam:
$m \leftarrow \beta_1 m + (1-\beta_1) g$,
$v \leftarrow \beta_2 v + (1-\beta_2) g^2$, step $\eta\, m/\sqrt{v}$, with
$\beta_1 = 0.9$, $\beta_2 = 0.999$. How long is the step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $m/\sqrt{v} \approx (1-\beta_1)/\sqrt{1-\beta_2} =
0.1/\sqrt{0.001} \approx 3.2$ for any $g$. The step is bounded at
$3.2\,\eta$, and clipping $g$ barely changes it. So clipping does not
rescue Adam from a learning rate that is too large.
:::
:::
:::

::: {.slide title="Clipping Protects Adam's State: a 100× Spike Stays in v for 2,300 Steps"}
[Gradient clipping]{.kicker}

An unclipped gradient 100 times the typical one, $\beta_2 = 0.999$:

- $v$ gains $(1-\beta_2) \cdot 100^2 = 10$ times its usual value: the
  steps shrink to $1/\sqrt{11} \approx 30\%$;
- $10 \cdot 0.999^t < 1$ only after $t \approx 2{,}300$ steps, and the
  steps are still at 70%;
- clipped at 4 times the median norm: $v$ gains $0.001 \cdot 4^2$, under
  2%.

$\beta_2 = 0.95$: $v$ gains 500 times, the steps fall below 5%, and the
spike is forgotten within about 120 steps.
:::

::: {.slide title="When a Spike Gets Through, Rewind and Skip"}
[Gradient clipping]{.kicker}

- PaLM: restart about **100 steps** before the spike, skip **200–500
  minibatches**. No spike there again.
- Same minibatches, earlier checkpoint: no spike. Data and parameter
  state cause it together; the report gives no further cause.
- A penalty on drifting logits (z-loss) was on throughout and did not
  prevent the spikes.

::: {.d2l-note}
A checkpoint holds model, optimizer state, step count and data position.
:::

[Chowdhery et al., 2022]{.small}
:::

::: {.slide title="An Average Needs Iterates That Are Different Draws"}
[Weight averaging]{.kicker}

Which weights do we keep? The last ones are one draw from the *noise
ball*, in which the weights fluctuate at a constant rate.

Exponential moving average (EMA) of the weights $\mathbf{x}_t$, decay
$\alpha$:

$$\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1-\alpha)\, \mathbf{x}_t,
\qquad \textrm{window} \approx \frac{1}{1-\alpha}\ \textrm{steps}$$

- Valley, curvature $c = 0.2$, learning rate $\eta = 0.1$: a coordinate
  forgets its position in $1/(\eta c) = 50$ steps.
- So the window is long: $\alpha = 0.999$, 1,000 steps.
:::

::: {.slide title="At a Constant Rate the EMA Is Steadier Than the Live Weights"}
[Weight averaging · CNN on Fashion-MNIST, SGD, η = 0.3, 235 steps per epoch]{.kicker}

@!practice-weight-averaging-3

- EMA ($\alpha = 0.999$) started after epoch 5. From step 0, the
  initialization would still weigh $0.999^{235} \approx 0.79$ after one
  epoch.
- Each epoch the live weights, the ones that training updates, show one
  point of the noise ball.
:::

::: {.slide title="The EMA Matches a Decayed Run; after a Decay It Adds Nothing" layout="tight"}
[Weight averaging]{.kicker}

Two seeds per schedule, test accuracy:

@!practice-weight-averaging-2

- Constant rate, live: two to six points of scatter. EMA: a little above
  91%.
- Cosine decay to zero: about 91% as well. A tie.
- EMA of the decayed run: a few tenths of a point behind at most.

::: {.d2l-note}
Use an EMA when no decay is planned, or for a model needed before the
decay.
:::
:::

::: {.slide title="Check: When Does an EMA Not Help?"}
[Check]{.kicker}

Name two situations in which the EMA of the weights is no better than
the last iterate.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) After the schedule has decayed the rate to nearly zero:
the weights barely move, and the average only lags. (2) While the weights
improve faster than the window follows: start the average late.
:::
:::
:::

::: {.slide title="Four Large Runs Agree on Most Settings and Differ on the Peak Rate"}
[A worked configuration]{.kicker}

| run | optimizer | $\beta_1,\beta_2$ | peak rate | warmup | clip | weight decay |
|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B | AdamW | — | $8{\times}10^{-5}$ | 8k | — | — |
| DeepSeek-V3 | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| OLMo 2 7B | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| Kimi K2 | MuonClip | — | $2{\times}10^{-4}$ | 500 | — | 0.1 |

- All four warm up and later lower the rate along a cosine.
- The widest model has the smallest peak rate.
- MuonClip: Muon with a cap on attention scores. A dash is missing
  evidence, not a default.

[Grattafiori et al., 2024; Liu et al., 2024; OLMo Team, 2025; Kimi Team, 2025]{.small}
:::

::: {.slide title="A New Model Is Configured from Short Runs, One Decision at a Time"}
[A worked configuration · TinyLM, 4 blocks, width 256, 3.3M parameters]{.kicker}

The run we pretend we cannot repeat: 3,000 steps.

| decision | choice | source |
|:--|:--|:--|
| batch size | fills the device, not above the noise scale | Batch Size |
| optimizer | AdamW (0.9, 0.95), weight decay 0.1 on matrices | published runs, AdamW |
| precautions | clip at 1, log the clip rate; EMA, window 200 | this deck |
| peak rate | best of three 300-step runs, 3× apart | Two Test Models, Adam |
| schedule | 60-step warmup (2%), cosine to zero | Schedules, published runs |
| verdict | two seeds: gap against spread | Two Test Models |
:::

::: {.slide title="The Device Is Full by 256 Sequences" layout="tight"}
[A worked configuration]{.kicker}

Time of one gradient of the 3.3M-parameter model:

@!practice-a-worked-configuration-2

- 32 to 128: four times the batch, far less than four times the time.
- From 256: the time doubles with the batch, and the sequences per second
  stop rising.
- 256 is the floor for the batch. The ceiling is the *noise scale*: the
  batch at which gradient noise equals the gradient.
:::

::: {.slide title="Three Runs of 300 Steps Pick the Peak Rate" layout="tight"}
[A worked configuration]{.kicker}

@!practice-a-worked-configuration-4

- 0.001: slow. 0.01: stalls near 2.5, though clipped most often. 0.003:
  best, a factor of 3 below the first rate that stalls.
- Noise scale: 2 sequences, then of the order of 100. The floor of 256
  is at or above it: stay at 256.
- Cost: 900 steps, under a third of the final run.

[Cold start, constant rate. Bold assumption: the best rate of 300 constant steps is the right peak for 3,000 decayed ones.]{.small}
:::

::: {.slide title="Four Logged Curves Tell the Failures of a Run Apart" layout="tight"}
[A worked configuration]{.kicker}

Log the gradient norm before clipping, the share of clipped steps, the
learning rate, and the update size relative to the weights.

| in the log | reading | first remedy |
|:--|:--|:--|
| loss leaves in the first steps, norms climb | steps too long for the start | warmup or clipping |
| loss stalls under Adam, first updates large against the weights | start too rough | warmup |
| loss stalls under Adam behind a warmup | rate too large throughout | one grid point down |
| clipping acts on most steps | a brake, not a guard | lower rate |
| loss flat and jittery | noise of the current rate | decay or average |
| loss smooth, still falling | slow | higher rate, more steps |
:::

::: {.slide title="The Schedule Lowers the Loss by 0.05 Nats, Many Times the Seed Spread" layout="tight"}
[A worked configuration]{.kicker}

Configured: 60-step warmup, cosine to zero. Baseline: the same AdamW and
clipping at a constant 0.003.

@!practice-a-worked-configuration-5

- EMA = final weights: nothing left to average after a decay to zero.
- Clipped under 1% of the steps: a guard. Whether the run needed it, the
  table cannot say.
- Known to have done work: the peak rate and the decay.
:::

::: {.slide title="Check: Diagnose a Stalled Run"}
[Check]{.kicker}

AdamW at a rate of 0.01 on the 3.3M-parameter `TinyLM`: the loss stalls
near 2.5 without diverging, and clipping at 1 fires more often than at
lower rates. What do you change?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Not the threshold: Adam moves a coordinate by about $\eta$
whatever the size of its gradient, so tighter clipping does not shorten
the step. Lower the rate by a grid point: 0.003 trains.
:::
:::
:::

::: {.slide title="A Long Run Is Configured Short, Guarded, and Finished with an Average or a Decay"}
[Wrap-up]{.kicker}

- **Guard:** clipping at $\theta = 1$ changed a few dozen early steps and
  let SGD train `TinyLM` at $\eta = 1$; at $\eta = 3$ it fired on
  practically every step and the run failed.
- **Adam:** step bounded at $3.2\,\eta$; an unclipped 100× spike stays
  in $v$ for 2,300 steps (120 at $\beta_2 = 0.95$).
- **Finish:** EMA at a constant rate = cosine decay, about 91% for the
  CNN on Fashion-MNIST.
- **Configure:** batch 256 from the device; peak 0.003 from three 300-step
  runs; warmup and cosine: 0.05 nats below the constant rate, seed spread
  under 0.01.

**Still open:** what a step costs, in seconds and in bytes. **Next
chapter:** attention, the mechanism inside `TinyLM`.
:::
