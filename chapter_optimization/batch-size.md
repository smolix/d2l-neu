# Batch Size
:label:`sec_batch_size`

Two sections of this chapter have already handled the batch size, one dial at
a time. :numref:`sec_sgd` measured its statistics: averaging $b$ examples
divides the gradient variance by $b$. :numref:`sec_minibatch_sgd` supplied
its mechanics: a batch amortizes dispatch overhead and keeps the hardware
fed, which is what makes the averaging nearly free until the device
saturates. Between them they justified every batch size we have used, and
left the essential question open: how large is too large? A bigger batch
costs proportionally more compute per step. If it removes noise the
optimizer was actually fighting, the run takes proportionally fewer steps
and the total compute bill is unchanged; if not, the extra examples are
wasted. Which of the two happens is not a hardware question. It is a
property of the optimization problem, and it changes as training proceeds.

This section makes the question quantitative. We define and measure the
*gradient-noise scale*, the batch size at which a minibatch gradient stops
being mostly noise; we run the defining experiment of large-batch training,
steps to a fixed target loss as a function of batch size, on both of the
chapter's testbeds; we check the standard rules for moving the learning rate
along with the batch and mark where they fail; and we look at how production
language-model runs act on all of this by growing the batch during training.
Of the three decisions an optimizer embodies, batching is the purest form of
noise management: it buys signal with compute, and what follows locates the
point where the price changes.

```{.python .input #batch-size}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #batch-size}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## The Gradient-Noise Scale

### Signal versus Noise

Write $\nabla f$ for the full-dataset gradient at the current parameters and
$\boldsymbol{\Sigma}$ for the covariance of a single example's gradient, so
that $\operatorname{tr} \boldsymbol{\Sigma} =
\mathbb{E}_i \|\nabla f_i - \nabla f\|^2$ is the total noise power carried
by one example. A minibatch gradient $\hat{\mathbf{g}}_b$ averaged over $b$
examples drawn with replacement is unbiased, and its expected squared
deviation from the truth is the $1/b$ law we measured in :numref:`sec_sgd`:
$\mathbb{E}\|\hat{\mathbf{g}}_b - \nabla f\|^2 =
\operatorname{tr} \boldsymbol{\Sigma} / b$. So every minibatch gradient is
the sum of a signal of squared length $\|\nabla f\|^2$, the same at every
batch size, and a noise of squared length
$\operatorname{tr} \boldsymbol{\Sigma} / b$, which shrinks as the batch
grows. The two parts change places at

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2},$$
:eqlabel:`eq_noise-scale`

the *gradient-noise scale* :cite:`McCandlish.Kaplan.Amodei.ea.2018`. Below
$b_{\textrm{noise}}$ the minibatch gradient is mostly noise, and doubling
the batch removes half of what stands between the optimizer and the true
descent direction. Above it the estimate is already essentially exact, and
further averaging polishes digits the update never uses. The noise scale is
therefore a prediction: batch sizes up to roughly $b_{\textrm{noise}}$
should convert compute into fewer steps at par, and batch sizes beyond it
should not. The rest of this section tests that prediction.

### A Two-Batch Estimator

Neither $\operatorname{tr} \boldsymbol{\Sigma}$ nor $\|\nabla f\|^2$ is
observable from a single minibatch, but both fall out of a quantity that is:
the squared *norm* of a minibatch gradient. Taking expectations of
$\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f + (\hat{\mathbf{g}}_b - \nabla f)\|^2$
and using unbiasedness,

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sqnorm-bias`

A minibatch gradient is *longer* than the true gradient on average, by
exactly the noise power. Measuring the mean squared norm at two batch sizes
$b_{\textrm{small}} < b_{\textrm{big}}$ gives two linear equations in the
two unknowns, and solving them yields the estimator below. This is the
measurement of :numref:`sec_sgd` stripped of its crutch: there we needed the
full-dataset gradient as ground truth; here two batch sizes suffice, which
is what makes the noise scale cheap enough that large training runs log it
continuously (in a data-parallel run, each worker's local gradient supplies
the small-batch norm for free).

We will measure both testbeds of this chapter: the Fashion-MNIST CNN of
:numref:`sec_adam` and the `TinyLM` character-level transformer of
:numref:`subsec_tinylm`, on its usual *Time Machine* data. The setup cell
assembles the data as indexable tensors, so that a random index set gives a
random minibatch. One caveat: the theory assumes independent draws, and the
*Time Machine* windows start at every character, so adjacent 64-character
examples overlap by 63 characters — a rough proxy for iid sampling, fine
for our purposes. For the language model one "example" is such a sequence,
so its noise scale and batch sizes are denominated in sequences (tokens =
sequences × 64).

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)
X_lm = data.X[:data.num_train].to(device)
Y_lm = data.Y[:data.num_train].to(device)
fashion = d2l.FashionMNIST(batch_size=256)
X_f = (fashion.train.data.float() / 255).unsqueeze(1).to(device)
y_f = fashion.train.targets.to(device)

def make_cnn():
    return nn.Sequential(
        nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)
X_lm, Y_lm = data.X[:data.num_train], data.Y[:data.num_train]
fashion = d2l.FashionMNIST(batch_size=256)
X_f = jnp.asarray(fashion.train[0], jnp.float32)[..., None] / 255
y_f = jnp.asarray(fashion.train[1], jnp.int32)

class FashionCNN(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = X.reshape(X.shape[0], -1)
        return self.fc2(nnx.relu(self.fc1(X)))
```

The estimator follows :eqref:`eq_sqnorm-bias` line by line: average the
squared gradient norm over many small minibatches and a few large ones,
solve for the noise power and the signal power, return the ratio.

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab pytorch
def grad_sq_norm(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           Y.reshape(-1))
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return sum((g ** 2).sum() for g in grads)

def noise_scale(model, X, Y, b_small=16, b_big=2048, m=400):
    def mean_sq_norm(b, m):
        idx = torch.randint(0, len(Y), (m, b), device=X.device)
        return torch.stack([grad_sq_norm(model, X[i], Y[i])
                            for i in idx]).mean()
    n_small, n_big = mean_sq_norm(b_small, m), mean_sq_norm(b_big, m // 8)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return (tr_sigma / sq_norm).item()
```

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab jax
def ce_loss(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

@nnx.jit
def grad_sq_norm(model, X, Y):
    grads = nnx.grad(ce_loss)(model, X, Y)
    return sum((g ** 2).sum() for g in jax.tree.leaves(grads))

def noise_scale(model, X, Y, key, b_small=16, b_big=2048, m=400):
    def mean_sq_norm(b, m, key):
        idx = jax.random.randint(key, (m, b), 0, len(Y))
        return jnp.stack([grad_sq_norm(model, X[i], Y[i])
                          for i in idx]).mean()
    k1, k2 = jax.random.split(key)
    n_small = mean_sq_norm(b_small, m, k1)
    n_big = mean_sq_norm(b_big, m // 8, k2)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return float(tr_sigma / sq_norm)
```

The noise scale depends on the current parameters, so *when* we measure
matters. We measure twice: at initialization, and after 500 steps of Adam,
using `d2l.train_lm` from :numref:`sec_adam` (which trains any model that
maps inputs to logits, the CNN included). First the CNN:

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab pytorch
torch.manual_seed(0)
cnn = make_cnn().to(device)
cnn(X_f[:2])  # Materialize the lazy layers
print(f'noise scale at initialization: {noise_scale(cnn, X_f, y_f):.0f}')
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.004)
d2l.train_lm(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: {noise_scale(cnn, X_f, y_f):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab jax
cnn = FashionCNN(rngs=nnx.Rngs(0))
print(f'noise scale at initialization: '
      f'{noise_scale(cnn, X_f, y_f, jax.random.PRNGKey(0)):.0f}')
optimizer = nnx.Optimizer(cnn, optax.adam(0.004), wrt=nnx.Param)
d2l.train_lm(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(cnn, X_f, y_f, jax.random.PRNGKey(1)):.0f}')
```

Then the language model, with smaller probe batches because one example is
already a 64-character sequence:

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab pytorch
torch.manual_seed(0)
lm = d2l.TinyLM(len(data.vocab)).to(device)
print(f'noise scale at initialization: '
      f'{noise_scale(lm, X_lm, Y_lm, b_small=8, b_big=256):.1f}')
optimizer = torch.optim.Adam(lm.parameters(), lr=0.003)
d2l.train_lm(lm, data, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(lm, X_lm, Y_lm, b_small=8, b_big=256):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab jax
lm = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))
ns_init = noise_scale(lm, X_lm, Y_lm, jax.random.PRNGKey(0),
                      b_small=8, b_big=256)
print(f'noise scale at initialization: {ns_init:.1f}')
optimizer = nnx.Optimizer(lm, optax.adam(0.002), wrt=nnx.Param)
d2l.train_lm(lm, data, optimizer, num_steps=500)
ns_500 = noise_scale(lm, X_lm, Y_lm, jax.random.PRNGKey(1),
                     b_small=8, b_big=256)
print(f'after 500 steps of Adam: {ns_500:.0f}')
```

Three readings, all of which we will use. First, the magnitudes: at these
model and dataset sizes the noise scale is small — tens of examples for the
CNN and single-digit sequence counts for the language model at
initialization, rising to several hundred examples and on the order of a
hundred sequences by step 500. The batch sizes we have been using by habit
are, by this measure, about right. Second, the growth: within those 500
steps the CNN's noise scale grows severalfold and the language model's by
one to two orders of magnitude, because early training finds directions of
improvement so large that every example agrees on them
($\|\nabla f\|$ is huge), and progress itself consumes them. We return to
this in the last part of the section, since it is the fact behind batch-size
ramps. Third, the caveats: the estimate is itself noisy, increasingly so as
$\|\nabla f\|^2$ shrinks toward the noise floor of the estimator, and
repeated runs scatter within a factor of about two, which is precise enough
for choosing a batch size and not for much else. And the magnitude does not
transfer across problems: the measurements of
:citet:`McCandlish.Kaplan.Amodei.ea.2018` across workloads put the noise
scale for large language models in the millions of tokens,
against a few thousand characters here. What transfers is the concept and
the growth during training, not the number.

## Steps to a Target

The noise scale makes a falsifiable claim about training speed. To test it
we use the standard experimental design of the large-batch literature
:cite:`Shallue.Lee.Antognini.ea.2019`: fix a target loss, train at a range
of batch sizes, and count the *steps* each needs to reach the target, along
with the *examples* processed, which is steps times batch size.
If minibatch noise is what limits progress, doubling $b$ should halve the
step count and leave the example count unchanged — *perfect scaling*. Once
$b$ passes the noise scale there is no noise left to buy, the step count
approaches a floor $S_{\min}$ set by the noiseless dynamics, and examples
are consumed to no effect. A simple model of a noisy quadratic
:cite:`McCandlish.Kaplan.Amodei.ea.2018` makes the whole trade-off a single
hyperbola in the run-averaged noise scale:

$$
\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}},
$$
:eqlabel:`eq_steps-examples`

with $E(b) = b\, S(b)$ the examples consumed and $E_{\min}$ the
small-batch limit. Both formulas bend at the same place: at
$b = b_{\textrm{noise}}$, steps and examples each sit at twice their
minimum. That elbow is the *critical batch size*, the largest batch that
still converts compute into time at a fair price.

Our protocol, in full: the target is a fixed loss evaluated on a fixed
subset of the training data every ten steps (evaluating on the noisy
per-minibatch training loss would bias the small-batch runs, whose loss
estimates fluctuate most). The optimizer is Adam. Its learning rate is tuned
by a short grid at an anchor batch size $b_0$ (64 sequences for the language
model, 256 examples for the CNN) and moved across batch sizes by the
square-root rule $\eta(b) = \eta_0 \sqrt{b/b_0}$, the rule appropriate for
adaptive methods; the next part of this section states and verifies it. Every run is a single seeded run and we read the curves
qualitatively, per this book's policy on experimental precision. The
harness is `train_lm` from :numref:`sec_adam` with the stopping rule added:

```{.python .input #batch-size-steps-to-a-target}
%%tab pytorch
def eval_loss(model, X, Y, chunk=4096):
    with torch.no_grad():
        losses = []
        for i in range(0, len(Y), chunk):
            logits = model(X[i:i + chunk])
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                Y[i:i + chunk].reshape(-1)))
    return torch.stack(losses).mean().item()

def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    model.to(device)
    step = 0
    while step < max_steps:
        for X, Y in data.train_dataloader():
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   Y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % eval_every == 0 and \
               eval_loss(model, X_eval, Y_eval) <= target:
                return step
            if step >= max_steps:
                break
    return float('inf')
```

```{.python .input #batch-size-steps-to-a-target}
%%tab jax
@nnx.jit
def eval_chunk(model, X, Y):
    return ce_loss(model, X, Y)

def eval_loss(model, X, Y, chunk=4096):
    return float(jnp.stack([eval_chunk(model, X[i:i + chunk],
                                       Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean())

def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    @nnx.jit
    def step_fn(model, optimizer, X, Y):
        loss, grads = nnx.value_and_grad(ce_loss)(model, X, Y)
        optimizer.update(model, grads)
        return loss
    step = 0
    while step < max_steps:
        for X, Y in data.train_dataloader():
            step_fn(model, optimizer, jnp.asarray(X), jnp.asarray(Y))
            step += 1
            if step % eval_every == 0 and \
               eval_loss(model, X_eval, Y_eval) <= target:
                return step
            if step >= max_steps:
                break
    return float('inf')
```

### The Language Model

For `TinyLM` the target is a loss of 1.5 per character, a perplexity of
about 4.5: well into learning, far from converged, and reachable at every
batch size in this sweep within a minute. The batch sizes span two orders
of magnitude in sequences.

```{.python .input #batch-size-the-language-model}
%%tab pytorch
lm_bs = [4, 16, 64, 256]
lm_steps = []
for b in lm_bs:
    torch.manual_seed(0)
    data_b = d2l.TimeMachine(batch_size=b, num_steps=64,
                             tokenization='char', num_train=100000)
    model = d2l.TinyLM(len(data_b.vocab))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.003 * math.sqrt(b / 64))
    lm_steps.append(train_to_target(model, data_b, optimizer, X_lm[:2048],
                                    Y_lm[:2048], target=1.5,
                                    max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.5', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-language-model}
%%tab jax
lm_bs = [4, 16, 64, 256]
lm_steps = []
for b in lm_bs:
    data_b = d2l.TimeMachine(batch_size=b, num_steps=64,
                             tokenization='char', num_train=100000)
    model = d2l.TinyLM(len(data_b.vocab), rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model,
                              optax.adam(0.002 * math.sqrt(b / 64)),
                              wrt=nnx.Param)
    lm_steps.append(train_to_target(model, data_b, optimizer, X_lm[:2048],
                                    Y_lm[:2048], target=1.5,
                                    max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.5', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

The first doubling steps are almost free: from $b=4$ to $b=16$ the measured
curve tracks the perfect-scaling line and the example count barely moves.
By $b=256$ the curve has clearly left the line — steps still fall, but each
quadrupling of the batch now buys about half the steps or less, not the
fourfold cut of perfect scaling, and the examples consumed have more than
doubled and are climbing. The elbow sits between a few tens and a couple
of hundred sequences, within a small factor of the noise scale we measured
mid-training, which is all :eqref:`eq_steps-examples` promises for so
coarse an estimate.

### The CNN

The identical experiment on the CNN, with a target loss of 0.35 and batch
sizes from 8 to 2048:

```{.python .input #batch-size-the-cnn}
%%tab pytorch
cnn_bs = [8, 32, 128, 512, 2048]
cnn_steps = []
for b in cnn_bs:
    torch.manual_seed(0)
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = make_cnn()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.004 * math.sqrt(b / 256))
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-cnn}
%%tab jax
cnn_bs = [8, 32, 128, 512, 2048]
cnn_steps = []
for b in cnn_bs:
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = FashionCNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model,
                              optax.adam(0.004 * math.sqrt(b / 256)),
                              wrt=nnx.Param)
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

The same shape on a different problem: perfect scaling through the first
quadrupling, then a widening gap to the reference line. At the top of the
range a further quadrupling of the batch (a quadrupling of the compute per
step) cuts the step count by at most a third, and in some runs the count
rises instead, a sign we will explain in the next part. By then the example
count sits several times above its small-batch minimum: compute is being
converted into speed at a punishing exchange rate, no matter how many
accelerators are available to supply it.

### The Bill in Examples

Replotting both sweeps as examples-to-target puts the two testbeds on one
axis and states the cost of parallelism directly:

```{.python .input #batch-size-the-bill-in-examples}
d2l.plot([lm_bs, cnn_bs],
         [[b * s for b, s in zip(lm_bs, lm_steps)],
          [b * s for b, s in zip(cnn_bs, cnn_steps)]],
         'batch size', 'examples to target', xscale='log', yscale='log',
         legend=['TinyLM', 'Fashion-MNIST CNN'])
```

Each curve is flat while batching is free and bends upward past its critical
batch size, just as :eqref:`eq_steps-examples` describes. Read as a menu
rather than a verdict: every point on a curve is a legitimate way to run the
job, spending more total compute (height) to finish in fewer steps, hence
less wall-clock time on enough hardware (leftward slope). Below the elbow,
parallelism is free speed. Above it, each further halving of the step count
costs a doubling of the compute. Where a given team should sit depends on
what is scarce, accelerators or days, which is why frontier runs crowd as
close to their critical batch size as their clusters allow and why the
exercises ask you to trace this Pareto frontier explicitly.

## Learning-Rate Rules

Changing the batch size changes the optimization problem the learning rate
was tuned for, so $\eta$ has to move with $b$; running the sweep above at a
fixed $\eta$ would have confounded batch statistics with step-size
mistuning. Two rules cover practice:

$$
\eta(b) = \frac{b}{b_0}\, \eta_0 \;\; \textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\; \textrm{(adaptive methods)}.
$$
:eqlabel:`eq_lr-rules`

The linear rule for SGD :cite:`Goyal.Dollar.Girshick.ea.2017` follows from
the noise-floor picture of :numref:`sec_sgd`: there we saw SGD stall in a
noise ball whose squared radius is proportional to $\eta$ times the gradient
variance. A batch of $b$ divides the variance by $b$, so holding $\eta / b$
fixed holds the noise ball, and with it the whole stochastic character of
the trajectory, fixed. It is also the small-batch limit of the optimal step
size in the noisy-quadratic model, $\eta_{\textrm{opt}}(b) =
\eta_{\max} / (1 + b_{\textrm{noise}}/b)$: linear in $b$ while
$b \ll b_{\textrm{noise}}$, saturating at $\eta_{\max}$ beyond it — the rule
announces its own expiry at the noise scale. For Adam and its relatives the
scaling is weaker. The preconditioner already divides each coordinate by its
root-mean-square gradient, which itself shrinks as the batch quiets the
noise. The stochastic-differential-equation analysis of
:citet:`Malladi.Lyu.Panigrahi.ea.2022` makes the scaling precise: growing
the batch by a factor $k$ preserves Adam's dynamics if every time constant
moves together — $\eta' = \sqrt{k}\,\eta$, $\beta_i' = 1 - k(1 - \beta_i)$,
$\epsilon' = \epsilon / \sqrt{k}$ — which is possible only while
$k(1 - \beta_i) < 1$, i.e. $k < 10$ at $\beta_1 = 0.9$. Common practice, and
the sweeps above, move $\eta$ alone; the full rule reduces to that
square-root heuristic when the $\beta$s barely move, and
:eqref:`eq_lr-rules` states it in that everyday form.

We verify both rules on the CNN with the chapter's usual instrument, a
learning-rate grid, trained at $b=8$ and $b=64$ on a fixed budget of 12,800
examples so that both batch sizes see the same data. If a rule holds, the
whole loss-versus-$\eta$ curve should shift right by the predicted factor —
$8\times$ for SGD, $\sqrt{8} \approx 2.8\times$ for Adam — when the batch
grows $8\times$. Since `d2l.train_lm` is already a fixed-step trainer, the
sweep is a few lines:

```{.python .input #batch-size-learning-rate-rules-1}
%%tab pytorch
def final_loss_at(make_optimizer, lrs, b, num_examples=12800):
    data_b = d2l.FashionMNIST(batch_size=b)
    final = []
    for lr in lrs:
        torch.manual_seed(0)
        model = make_cnn().to(device)
        model(X_f[:2])  # Materialize the lazy layers
        d2l.train_lm(model, data_b, make_optimizer(model, lr),
                     num_steps=num_examples // b)
        loss = eval_loss(model, X_f[:8192], y_f[:8192])
        final.append(loss if math.isfinite(loss) else float('nan'))
        print(f'b={b:3d}  lr={lr:g}  loss={final[-1]:.3f}')
    return final
```

```{.python .input #batch-size-learning-rate-rules-1}
%%tab jax
def final_loss_at(make_tx, lrs, b, num_examples=12800):
    data_b = d2l.FashionMNIST(batch_size=b)
    final = []
    for lr in lrs:
        model = FashionCNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        d2l.train_lm(model, data_b, optimizer,
                     num_steps=num_examples // b)
        loss = eval_loss(model, X_f[:8192], y_f[:8192])
        final.append(loss if math.isfinite(loss) else float('nan'))
        print(f'b={b:3d}  lr={lr:g}  loss={final[-1]:.3f}')
    return final
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab pytorch
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                             momentum=0.9)
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64)
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab jax
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
sgd_small = final_loss_at(lambda lr: optax.sgd(lr, momentum=0.9),
                          sgd_lrs, b=8)
sgd_big = final_loss_at(lambda lr: optax.sgd(lr, momentum=0.9),
                        sgd_lrs, b=64)
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab pytorch
adam_lrs = [0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128,
            0.0256]
make_adam = lambda model, lr: torch.optim.Adam(model.parameters(), lr)
adam_small = final_loss_at(make_adam, adam_lrs, b=8)
adam_big = final_loss_at(make_adam, adam_lrs, b=64)
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab jax
adam_lrs = [0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128,
            0.0256]
adam_small = final_loss_at(optax.adam, adam_lrs, b=8)
adam_big = final_loss_at(optax.adam, adam_lrs, b=64)
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

Both plots read the same way: each curve is a basin with a cliff on its
right, and growing the batch slides the basin rightward. For SGD the best
region moves by roughly the batch ratio, a factor of four to eight in our
runs — retuning from scratch at $b=64$ finds nothing better than what the
linear rule predicts from $b=8$. For Adam the basin moves by only a factor
of about two to four; applying the *linear* rule to Adam ($8\times$, the
rightmost points of the upper curve) lands measurably past the optimum. The
basins are broad, which is itself worth remembering: near the optimum,
being wrong by a factor of two costs little, and the rules are aids for
crossing large batch-size ratios, not precision instruments.

### Where the Rules Break

The rules fail in three characteristic places, and each failure is visible
somewhere in this section's data. *Past the noise scale*: both rules assume
the run is noise-limited; once $b$ approaches $b_{\textrm{noise}}$ the
optimal step stops growing, and continuing to scale $\eta$ overshoots — the
upturn at the top of the steps-to-target sweeps is exactly this, and the
exercises reproduce it in isolation. *Against the stability ceiling*: the
largest usable $\eta$ is capped by curvature, as we saw for gradient descent
in :numref:`sec_gd`, and the cap does not move with the batch. When we first
ran the SGD sweep at batch sizes of a few hundred, both basins pinned
against the same cliff and the optimum stopped moving altogether; scaling
rules only operate in the room below the ceiling. *Early in training*: at
initialization the noise scale is at its smallest and curvature effects at
their worst, so a large batch with a linearly scaled $\eta$ can diverge in
the first steps. The practical fix is learning-rate warmup, which
:citet:`Goyal.Dollar.Girshick.ea.2017` introduced for exactly this reason;
:numref:`sec_scheduler` covers it. Finally, none of this is a universal law:
measured across many workloads, the curves of
:eqref:`eq_steps-examples` hold their shape but their elbows range over
orders of magnitude with architecture, dataset, and optimizer
:cite:`Shallue.Lee.Antognini.ea.2019`. Large batches were once also blamed
for a *generalization* gap — converging to sharp minima that test poorly
:cite:`Keskar.Mudigere.Nocedal.ea.2017` — but much of that gap closes under
full retuning of $\eta$ and training length; the durable cost of a large
batch is data efficiency, which is what this section measured.

## Growing the Batch

The noise scale is not a constant of the problem. We measured it growing
within the first 500 steps (severalfold for the CNN, one to two orders of
magnitude for the language model), and the mechanism operates at every
scale: $\|\nabla f\|$ collapses as easy progress is consumed, while
per-example disagreement persists. A fixed batch
size is therefore the wrong shape: whatever $b$ is right at the end of
training is wasteful at the beginning, when the critical batch size is
small. The practice that follows is the *batch-size ramp*, and it is now
standard at the frontier. GPT-3 grew its batch from 32 thousand to 3.2
million tokens over the first billions of tokens of training
:cite:`brown2020language`. Llama 3 doubled its batch twice on a fixed
schedule, from 4 million to 16 million tokens, as training progressed
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. DeepSeek-V3 ramped from about 3
thousand to 15 thousand sequences across the first half-trillion tokens
:cite:`Liu.Feng.Xue.ea.2024`. What these schedules set by hand, measurement
can set directly: researchers at Ai2 measured the critical batch size as
training proceeded and doubled the batch each time the measurement overtook
it, finding that the critical batch size tracks the loss reached — the data
scale — far more than the model size
:cite:`Merrill.Arora.Groeneveld.ea.2025`, in line with other measurements of
how the elbow moves during pretraining :cite:`Zhang.Morwani.Vyas.ea.2024`.

Two interactions are worth carrying forward. First, with the schedule:
raising $b$ at fixed $\eta$ quiets the gradient exactly as lowering $\eta$
at fixed $b$ does — the noise floor depends on the two only through
$\eta / b$ — so a batch ramp is a learning-rate decay in disguise
:cite:`Smith.Kindermans.Ying.ea.2018`, and the ramp and the schedule of
:numref:`sec_scheduler` must be designed together, not tuned as independent
knobs. Second, with the optimizer: the descent-direction decision feeds
back into the noise-management one. Measurements on pretraining workloads
find that Muon sustains its data efficiency out to larger batch sizes than
AdamW :cite:`Shah.Polloreno.Stratos.ea.2025`, effectively moving the elbow
of :eqref:`eq_steps-examples` to the right; :numref:`sec_muon` takes up why.
One boundary remains. Everything in this section prices a large batch in
examples; turning the fewer, larger steps into less wall-clock time
requires splitting each batch across many devices, and that is data
parallelism, the machinery of :numref:`chap_performance`.

## Summary

A minibatch gradient is signal plus noise, and the two trade places at the
gradient-noise scale $b_{\textrm{noise}} = \operatorname{tr}
\boldsymbol{\Sigma} / \|\nabla f\|^2$, measurable with nothing more than
gradient norms at two batch sizes. Below it, doubling the batch halves the
steps to a target at constant total compute; above it, steps flatten at a
floor and examples are wasted — both testbeds showed the predicted
hyperbola, with the elbow within a small factor of the measured noise
scale. Moving between batch sizes requires moving the learning rate: with
the batch ratio for SGD, with its square root for adaptive methods, valid
below the noise scale and under the stability ceiling, with warmup
protecting the early steps. The noise scale grows as the loss falls, which
is why serious runs grow their batch during training rather than fixing it.

## Exercises

1. The noise-scale cell measured `TinyLM` at initialization and after 500
   steps. Extend it: train for 3,000 steps and measure every 500, then plot
   the noise scale against the training loss at each checkpoint. If you were
   scheduling a batch-size ramp by the rule "double $b$ when
   $b_{\textrm{noise}}$ overtakes it", where would the doublings land?
1. Extend the `TinyLM` sweep to $b = 1024$ under the square-root rule. You
   should find that the step count stops falling and may rise outright.
   Then rerun $b = 1024$ with $\eta$ held at its $b = 256$ value. Explain
   both observations with :eqref:`eq_steps-examples` and the saturating
   optimal step size $\eta_{\textrm{opt}}(b)$.
1. Estimate $b_{\textrm{noise}}$ for `TinyLM` as a run-averaged quantity
   (for instance, the mean of your checkpoint measurements from the first
   exercise, up to the target loss) and compare it with the elbow of the
   steps-to-target curve, i.e., the batch size at which examples-to-target
   reaches twice its minimum. How close is the factor-of-two prediction of
   :eqref:`eq_steps-examples`?
1. The sweeps moved $\eta$ by the square-root rule rather than retuning.
   Retune properly: for each batch size in the CNN sweep, run a three-point
   learning-rate grid around the rule's value and keep the best
   steps-to-target. Which points of the curve move, and does the elbow?
1. Suppose one optimizer step at batch size $b$ takes $t(b) = t_0 (1 + b /
   b_{\textrm{sat}})$ seconds on your hardware, where $b_{\textrm{sat}}$ is
   the batch size that saturates the device (measure both constants with
   `d2l.Timer` if you can). Combine $t(b)$ with your measured $S(b)$ to
   plot time-to-target against compute-to-target across $b$. Identify the
   time-optimal and the compute-optimal batch size. This two-axis reading
   is the form in which the batch-size decision reaches a training team.
1. The census of :numref:`subsec_tinylm` splits `TinyLM`'s parameters into
   embeddings, matrices, and vectors. Restrict `grad_sq_norm` to each
   population in turn and measure three separate noise scales. Which
   population is noisiest, and how does your answer relate to the sparse
   gradients that motivated AdaGrad in :numref:`sec_adam`?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.10]{.kicker}

How large a batch?<br>
**the gradient-noise scale · steps to a target · learning-rate rules · batch ramps**
:::
:::

::: {.slide title="Two reasons to batch, one open question"}
[Motivation]{.kicker}

- §9.3, statistics: gradient variance $\propto 1/b$.
- §9.4, mechanics: a batch amortizes dispatch and keeps the device fed.
- Neither says when to stop. A bigger batch costs proportionally more
  compute per step — when does it stop buying an equal cut in *steps*?

. . .

The answer is a property of the optimization problem, not the hardware —
and it changes as training proceeds.
:::

::: {.slide title="The gradient-noise scale"}
A minibatch gradient is **signal + noise**: squared length
$\|\nabla f\|^2$ plus $\operatorname{tr}\boldsymbol{\Sigma}/b$. They
trade places at

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2}$$

(McCandlish et al., 2018).

- $b \ll b_{\textrm{noise}}$: mostly noise — doubling $b$ removes half
  of what obscures the descent direction.
- $b \gg b_{\textrm{noise}}$: essentially exact — more averaging
  polishes digits the update never uses.
:::

::: {.slide title="Measure it with two batch sizes"}
Take expectations of the squared *norm*:

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$

A minibatch gradient is *longer* than the true one — by exactly the noise
power. Two batch sizes → two equations → both unknowns:

@batch-size-a-two-batch-estimator-2
:::

::: {.slide title="The noise scale grows during training"}
CNN, then `TinyLM` — at initialization and after 500 steps of Adam:

@!batch-size-a-two-batch-estimator-3

. . .

@!batch-size-a-two-batch-estimator-4

- Tens of examples (CNN) and single-digit sequence counts (`TinyLM`) at
  init; severalfold to two orders of magnitude larger 500 steps later.

::: {.d2l-note}
For large language models, measured noise scales run to the **millions of
tokens** (McCandlish et al., 2018). The magnitude does not transfer; the
growth does.
:::
:::

::: {.slide title="Steps to a target: the prediction"}
Train to a fixed loss at many batch sizes; count steps $S$ and examples
$E = bS$. The noisy-quadratic model gives one hyperbola:

$$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}}.$$

- Below $b_{\textrm{noise}}$: **perfect scaling** — double $b$, halve
  $S$, $E$ constant.
- Above: $S$ floors at $S_{\min}$, examples wasted.
- Elbow (both at $2\times$ their minimum) = **critical batch size**.
:::

::: {.slide title="Protocol"}
Fixed target loss on a fixed evaluation batch, checked every 10 steps
(per-minibatch training loss would bias the small-batch runs). Adam;
$\eta$ tuned at one anchor $b_0$, moved by the $\sqrt{b/b_0}$ rule.
Single seeded runs, read qualitatively.

@batch-size-steps-to-a-target
:::

::: {.slide title="TinyLM: steps to loss 1.5"}
@!batch-size-the-language-model

- $b=4 \to 16$ tracks perfect scaling; examples barely move.
- By $b=256$ the curve has left the line: a quadrupled batch buys about
  half the steps or less, and examples-to-target climbs.
- Elbow within a small factor of the mid-training noise scale.
:::

::: {.slide title="Same experiment, CNN"}
@!batch-size-the-cnn

At the top of the range a quadrupled batch — quadrupled compute per step —
cuts the step count by at most a third, and in some runs the count rises;
examples-to-target sits several times above its minimum.
:::

::: {.slide title="The bill in examples"}
@!batch-size-the-bill-in-examples

- Flat while batching is free; bends upward past the critical batch size.
- A menu, not a verdict: height = compute, leftward = fewer steps = time.
- Below the elbow, parallelism is free speed. Above it, halving the steps
  costs a doubling of the compute.
:::

::: {.slide title="Moving the learning rate with the batch"}
$$\eta(b) = \frac{b}{b_0}\, \eta_0 \;\;\textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\;\textrm{(adaptive)}.$$

- Linear: the noise floor depends on $\eta/b$ only — hold it fixed
  (Goyal et al., 2017). Small-$b$ limit of
  $\eta_{\textrm{opt}}(b) = \eta_{\max}/(1 + b_{\textrm{noise}}/b)$:
  the rule expires at the noise scale.
- Square root: Adam's preconditioner already divides by the gradient's
  RMS; the full SDE rule also moves $\beta_i$ and $\epsilon$, and reduces
  to $\eta \propto \sqrt{b}$ when they barely move (Malladi et al., 2022).
:::

::: {.slide title="Verification: the basin slides"}
Loss vs. $\eta$ at $b=8$ and $b=64$, fixed example budget:

@!batch-size-learning-rate-rules-2

. . .

@!batch-size-learning-rate-rules-3

- SGD's basin moves by roughly the batch ratio; Adam's by roughly its
  square root — the linear rule applied to Adam overshoots.
- Basins are broad: rules are for crossing large ratios, not fine tuning.
:::

::: {.slide title="Where the rules break"}
- **Past the noise scale** — the optimal step saturates; scaled $\eta$
  overshoots (the upturn at the top of the sweeps).
- **At the stability ceiling** — curvature caps $\eta$ regardless of $b$;
  at larger batches the optimum pins against the cliff and stops moving.
- **Early in training** — smallest noise scale, worst curvature: warmup
  (Goyal et al., 2017; §9.8).
- No universal law: elbows vary by orders of magnitude across workloads
  (Shallue et al., 2019). The durable cost of a large batch is **data
  efficiency**, not accuracy.
:::

::: {.slide title="Frontier practice: grow the batch"}
$b_{\textrm{noise}}$ grows as the loss falls → a fixed batch is the wrong
shape.

- GPT-3: 32k → 3.2M tokens. Llama 3: 4M → 16M. DeepSeek-V3: ~3k → ~15k
  sequences.
- Ai2: measure the critical batch size *during* training, double $b$
  when it overtakes the batch — it tracks the loss reached, not the
  model size.

. . .

- A batch ramp is a learning-rate decay in disguise ($\eta/b$) — design
  it with the schedule (§9.8).
- Muon holds data efficiency to larger $b$ than AdamW — the elbow moves
  right (§9.9).
- Turning fewer steps into less time = data parallelism (ch. 11).
:::

::: {.slide title="Recap"}
- Noise scale $b_{\textrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma} / \|\nabla f\|^2$:
  measurable from gradient norms at two batch sizes.
- Steps-to-target: perfect scaling below it, a floor above; the elbow is
  the critical batch size, and it showed up where the noise scale said.
- Move $\eta$ with $b$: linearly for SGD, square root for Adam — below
  the noise scale, under the stability ceiling, behind warmup.
- The noise scale grows during training: serious runs ramp the batch.
:::
