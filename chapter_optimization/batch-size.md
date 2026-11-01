# Batch Size
:label:`sec_batch_size`

Doubling the minibatch from 64 to 128 sequences doubles the work in every
step. If the run then needs half as many steps to reach the same loss,
nothing is lost: the total work is unchanged, and on twice as many
accelerators the run finishes in half the time. If it needs as many steps as
before, half of the work was spent for nothing. Which of the two happens
decides how fast a model can be trained, because a run can be shared among
only as many accelerators as its batch can feed.

Two facts from earlier sections frame the question. Averaging $b$ examples
divides the variance of a gradient estimate by $b$ (:numref:`sec_sgd`), and
an accelerator processes a larger batch in barely more time per step until
it is fully occupied. A larger batch therefore gives a better gradient at
little extra cost per step. It saves steps, however, only as long as the
noise in the gradient was what slowed the optimizer down.

This section turns that observation into numbers. We compare the size of
the noise in a gradient with the size of the gradient itself, measure the
ratio on two small models, and use it to predict how the number of steps to
a target loss falls as the batch grows. We then ask how the learning rate
must change when the batch does, and why the best batch size grows while a
model trains.

```{.python .input #batch-size}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
import random
import torch
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
import numpy as np
import optax
import random
```

## The Gradient-Noise Scale

### Signal and Noise

Write $\nabla f$ for the gradient of the training loss over the whole
dataset at the current parameters, and $\nabla f_i$ for the gradient of the
loss on example $i$. The per-example gradients scatter around their mean
$\nabla f$. Their expected squared distance from it,

$$\operatorname{tr} \boldsymbol{\Sigma} = \mathbb{E}_i \|\nabla f_i - \nabla f\|^2,$$

is the *noise power* of one example, the trace of the covariance matrix
$\boldsymbol{\Sigma}$ of the per-example gradients. A minibatch gradient
$\hat{\mathbf{g}}_b$ averages $b$ examples drawn independently with
replacement. It is unbiased, and its noise power is $b$ times smaller,
because the variance of an average of $b$ independent terms is $1/b$ times
the variance of one:

$$\mathbb{E}\|\hat{\mathbf{g}}_b - \nabla f\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$

This is the $1/b$ law that :numref:`sec_sgd` checked on a small network.
It splits every minibatch gradient into a *signal* $\nabla f$, whose squared
length $\|\nabla f\|^2$ is the same at every batch size, and a *noise* whose
squared length averages $\operatorname{tr} \boldsymbol{\Sigma} / b$ and
shrinks as the batch grows. The two have equal power at the batch size

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2},$$
:eqlabel:`eq_noise-scale`

called the *gradient-noise scale* :cite:`McCandlish.Kaplan.Amodei.ea.2018`.
If, say, $\operatorname{tr} \boldsymbol{\Sigma} = 100$ and
$\|\nabla f\|^2 = 1$, the gradient of a single example carries a hundred
times more noise power than signal, and a batch of
$b_{\textrm{noise}} = 100$ examples brings the two to parity.

Below $b_{\textrm{noise}}$ the noise dominates. Doubling the batch halves
the noise power and gives a markedly better direction, so each step can
make more progress and fewer steps are needed. Above $b_{\textrm{noise}}$
the signal dominates, the direction is already nearly right, and further
averaging removes little error. The noise scale therefore predicts that
growing the batch up to about $b_{\textrm{noise}}$ reduces the number of
steps without much increase in total work, and that beyond it the total
work grows faster than the number of steps falls. The rest of this section
tests that prediction.

### A Two-Batch Estimator
:label:`subsec_batch-size-estimator`

Neither $\operatorname{tr} \boldsymbol{\Sigma}$ nor $\|\nabla f\|^2$ can be
read off a single minibatch, but both follow from a quantity that can: the
squared norm of a minibatch gradient. We expand
$\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f + (\hat{\mathbf{g}}_b - \nabla f)\|^2$
and take expectations. The cross term vanishes because the deviation has
mean zero, which leaves

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sqnorm-bias`

On average a minibatch gradient is longer than the true gradient, and the
excess squared length is its noise power. Let $N_b$ denote the mean squared
norm of minibatch gradients at batch size $b$. Measuring it at a small batch
size $b_{\textrm{s}}$ and a large one $b_{\textrm{l}}$ gives two linear
equations in the two unknowns, with the solution

$$
\operatorname{tr} \boldsymbol{\Sigma} \approx
\frac{N_{b_{\textrm{s}}} - N_{b_{\textrm{l}}}}{1/b_{\textrm{s}} - 1/b_{\textrm{l}}},
\qquad
\|\nabla f\|^2 \approx
\frac{b_{\textrm{l}} N_{b_{\textrm{l}}} - b_{\textrm{s}} N_{b_{\textrm{s}}}}{b_{\textrm{l}} - b_{\textrm{s}}}.
$$

The difference of the two mean norms isolates the noise, and the large
batch, nearly free of noise, pins down the signal. Checking the $1/b$ law
directly requires the full-dataset gradient as ground truth. This estimator
needs no full gradient, which makes it cheap enough to log throughout a large
training run, where every data-parallel worker computes a small-batch
gradient anyway and their average is the large-batch gradient.

We measure the two models of :numref:`subsec_tinylm`. `TinyLM` is a
two-block decoder-only transformer with about 0.4 million parameters that
predicts the next character of *The Time Machine* from windows of 64
characters. `FashionCNN` classifies Fashion-MNIST images with two
convolution-and-pooling stages and two fully connected layers, also about
0.4 million parameters. To draw a random minibatch by index, we hold
each training set in memory as one tensor. For the language model one
example is a 64-character window, so its noise scale and its batch sizes
count sequences, each of 64 tokens. Adjacent windows share 63 of their 64
characters, so these sequences are far from independent pieces of text, and
the noise scale of the language model describes this sampling scheme rather
than a count of independent text.

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
```

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=64, tokenization='char',
                       num_train=100000)
X_lm, Y_lm = data.X[:data.num_train], data.Y[:data.num_train]
fashion = d2l.FashionMNIST(batch_size=256)
X_f = jnp.asarray(fashion.train[0], jnp.float32)[..., None] / 255
y_f = jnp.asarray(fashion.train[1], jnp.int32)
```

The estimator follows the two formulas line by line: average the squared
gradient norm over many small minibatches and a few large ones, solve for
the noise power and the signal power, and return their ratio.

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab pytorch
def ce_loss(model, X, Y):
    logits = model(X)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           Y.reshape(-1))

def grad_sq_norm(model, X, Y):
    grads = torch.autograd.grad(ce_loss(model, X, Y),
                                list(model.parameters()))
    return sum((g ** 2).sum() for g in grads)

def noise_scale(model, X, Y, b_small=16, b_big=2048, m=400):
    def mean_sq_norm(b, m):  # N_b, averaged over m random minibatches
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
    def mean_sq_norm(b, m, key):  # N_b, averaged over m random minibatches
        idx = jax.random.randint(key, (m, b), 0, len(Y))
        return jnp.stack([grad_sq_norm(model, X[i], Y[i])
                          for i in idx]).mean()
    k_small, k_big = jax.random.split(key)
    n_small = mean_sq_norm(b_small, m, k_small)
    n_big = mean_sq_norm(b_big, m // 8, k_big)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return float(tr_sigma / sq_norm)
```

The noise scale depends on the parameters, so we measure it twice: at
initialization, and after 500 steps of Adam with `d2l.train_lm` from
:numref:`sec_adam`. That function streams minibatches and minimizes the
cross-entropy of a model's outputs for a fixed number of steps, so it trains
the CNN as well as the language model. First the CNN:

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab pytorch
cnn = d2l.FashionCNN().to(device)
print(f'noise scale at initialization: {noise_scale(cnn, X_f, y_f):.0f}')
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.004)
d2l.train_lm(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: {noise_scale(cnn, X_f, y_f):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab jax
key_init, key_500 = jax.random.split(jax.random.key(random.getrandbits(32)))
cnn = d2l.FashionCNN(rngs=nnx.Rngs(random.getrandbits(32)))
print(f'noise scale at initialization: '
      f'{noise_scale(cnn, X_f, y_f, key_init):.0f}')
optimizer = nnx.Optimizer(cnn, optax.adam(0.004), wrt=nnx.Param)
d2l.train_lm(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: {noise_scale(cnn, X_f, y_f, key_500):.0f}')
```

Then the language model, with smaller probe batches because one example is
already a sequence of 64 characters:

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab pytorch
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
key_init, key_500 = jax.random.split(jax.random.key(random.getrandbits(32)))
lm = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(random.getrandbits(32)))
ns_init = noise_scale(lm, X_lm, Y_lm, key_init, b_small=8, b_big=256)
print(f'noise scale at initialization: {ns_init:.1f}')
optimizer = nnx.Optimizer(lm, optax.adam(0.003), wrt=nnx.Param)
d2l.train_lm(lm, data, optimizer, num_steps=500)
ns_500 = noise_scale(lm, X_lm, Y_lm, key_500, b_small=8, b_big=256)
print(f'after 500 steps of Adam: {ns_500:.0f}')
```

At initialization the noise scale is about a hundred images for the CNN
and a few sequences for the language model. Both models are small, and so
are their noise scales: the batches of 256 images and 64 sequences that
train them elsewhere in this chapter lie above these values.

The noise scale also changes, and quickly. After 500 steps that of the CNN
has grown, to between about 100 and 850 images across our runs, and that of
the language model by more than an order of magnitude, to between about 60
and 150 sequences. At
initialization the full gradient is large compared with the disagreement
between examples. As training removes the easy part of the loss,
$\|\nabla f\|^2$ shrinks, while the scatter of the per-example gradients
shrinks more slowly. We return to this growth at the end of the section.

The estimate itself is rough. Repeated runs scatter by up to a factor of
about two, and more as $\|\nabla f\|^2$ approaches the noise floor of the
estimator itself. That is precise enough to choose a batch size, but not
for much more. Nor do the magnitudes carry over to other problems:
:citet:`kaplan2020scaling` found critical batch sizes, which track the noise
scale, of around a million tokens for large language models, against several
thousand characters here. What carries over is the concept and its growth
during training.

## Steps to a Target
:label:`subsec_batch-size-steps`

The noise scale makes a prediction about training speed that could fail, so
we test it. The test, standard in the study of large-batch training
:cite:`Shallue.Lee.Antognini.ea.2019`, fixes a target loss, trains at a
range of batch sizes, and counts the *steps* $S(b)$ that each run needs to
reach the target and the *examples* $E(b) = b\, S(b)$ that it processes on
the way. If doubling $b$ halves $S$, the example count stays constant; we
call this *perfect scaling*. It cannot continue forever. No batch, however
large, gives a better gradient than the exact one, so the step count
approaches a floor $S_{\min}$, the number of steps that training with the
full gradient would need. A model of noisy optimization on a quadratic
:cite:`McCandlish.Kaplan.Amodei.ea.2018` connects the two limits:

$$
\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}},
$$
:eqlabel:`eq_steps-examples`

where $E_{\min}$ is the number of examples that a run with very small
batches consumes, and $b_{\textrm{noise}}$ is the noise scale averaged over
the run. The step count falls like $1/b$ while $b$ is small compared with
$b_{\textrm{noise}}$ and levels off at $S_{\min}$ beyond it; the example
count does the reverse. Because $E = b\,S$, the two minima are tied by
$E_{\min} = b_{\textrm{noise}} S_{\min}$. With $b_{\textrm{noise}} = 100$,
for instance, a batch of 10 needs $11\, S_{\min}$ steps but only
$1.1\, E_{\min}$ examples, and a batch of 1,000 needs $1.1\, S_{\min}$ steps
but $11\, E_{\min}$ examples. At $b = b_{\textrm{noise}}$ both stand at twice
their minimum, and the batch size of this bend is called the *critical batch
size*. Below it, most of the extra work per step turns into fewer steps;
above it, most of the extra work is wasted.
In the model, the noise scale weights each direction of parameter space by
its curvature. The unweighted ratio :eqref:`eq_noise-scale` that we
measured agrees with it when the curvature is the same in every direction,
and serves as a cheap stand-in otherwise.

Our protocol is as follows. The target is a fixed loss, evaluated every ten
steps on a fixed subset of the training data. For the language model the
subset is every 48th window, so that its 2,048 sequences cover the whole
text. Evaluating the loss of the
current minibatch instead would let the small-batch runs, whose loss
estimates fluctuate most, cross the target early by chance. The optimizer is
Adam. Its learning rate is tuned at an anchor batch size $b_0$ (64 sequences
for the language model, 256 images for the CNN) and moved to the other batch
sizes by the square-root rule $\eta(b) = \eta_0 \sqrt{b / b_0}$, which the
next part of this section derives and tests. Each sweep draws one random
seed and starts all its batch sizes from the same initialization, so that
the differences along a curve come from the batch size and not from the
draw. The step counts still vary from run to run, and we read only features
of the curves that recur across runs. The training loop is that of
`d2l.train_lm`, with the stopping rule added:

```{.python .input #batch-size-steps-to-a-target}
%%tab pytorch
def eval_loss(model, X, Y, chunk=4096):
    with torch.inference_mode():
        return torch.stack([ce_loss(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean().item()

def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    """Return the number of steps until the eval loss reaches target."""
    step = 0
    while step < max_steps:
        for X, Y in data.train_dataloader():
            loss = ce_loss(model, X.to(device), Y.to(device))
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
eval_chunk = nnx.jit(ce_loss)

def eval_loss(model, X, Y, chunk=4096):
    return float(jnp.stack([eval_chunk(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean())

@nnx.jit
def train_step(model, optimizer, X, Y):
    loss, grads = nnx.value_and_grad(ce_loss)(model, X, Y)
    optimizer.update(model, grads)
    return loss

def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    """Return the number of steps until the eval loss reaches target."""
    step = 0
    while step < max_steps:
        for X, Y in data.train_dataloader():
            train_step(model, optimizer, jnp.asarray(X), jnp.asarray(Y))
            step += 1
            if step % eval_every == 0 and \
               eval_loss(model, X_eval, Y_eval) <= target:
                return step
            if step >= max_steps:
                break
    return float('inf')
```

### The Language Model

For `TinyLM` the target is a loss of 1.5 nats per character, a perplexity
of $e^{1.5} \approx 4.5$ against about 28 for uniform guessing over the
vocabulary: well into learning, and far from converged. The batch sizes span
two orders of magnitude, from 4 to 256 sequences.

```{.python .input #batch-size-the-language-model}
%%tab pytorch
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in lm_bs:
    torch.manual_seed(seed)
    data_b = d2l.TimeMachine(batch_size=b, num_steps=64,
                             tokenization='char', num_train=100000)
    model = d2l.TinyLM(len(data_b.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.003 * math.sqrt(b / 64))
    lm_steps.append(train_to_target(
        model, data_b, optimizer, X_lm[::48][:2048], Y_lm[::48][:2048],
        target=1.5, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.5', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-language-model}
%%tab jax
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in lm_bs:
    data_b = d2l.TimeMachine(batch_size=b, num_steps=64,
                             tokenization='char', num_train=100000)
    model = d2l.TinyLM(len(data_b.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model,
                              optax.adam(0.003 * math.sqrt(b / 64)),
                              wrt=nnx.Param)
    lm_steps.append(train_to_target(
        model, data_b, optimizer, X_lm[::48][:2048], Y_lm[::48][:2048],
        target=1.5, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.5', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

Up to $b = 64$ the measured curve runs nearly parallel to the
perfect-scaling line: each quadrupling of the batch cuts the step count about
threefold, and the example count grows slowly. Beyond $b = 64$ the curve
bends away from the line. The last quadrupling, from 64 to 256 sequences,
cuts the step count only about in half, and the run consumes about three
times as many examples as the run with the smallest batch.

### The CNN

The same experiment on the CNN, with a target loss of 0.35 and batch sizes
from 8 to 2,048 images:

```{.python .input #batch-size-the-cnn}
%%tab pytorch
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in cnn_bs:
    torch.manual_seed(seed)
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN().to(device)
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
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in cnn_bs:
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
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

The CNN shows the same pattern. The first quadrupling, from 8 to 32 images,
stays close to the perfect-scaling line, cutting the step count three- to
fourfold, and after it the gap widens. At the top of the range a further
quadrupling of the batch, and with it of the work per step, reduces the step
count by about a third at most and sometimes not at all. The example count
there stands at six to twelve times its small-batch value.
Extra work per step saves few or no steps and multiplies the data consumed.

### The Critical Batch Size

Both curves have the shape that :eqref:`eq_steps-examples` predicts, and we
can fit the model to them. Its step count
$S(b) = S_{\min} + S_{\min} b_{\textrm{crit}} / b$ is linear in $1/b$, so a
straight-line fit of the measured step counts against $1/b$ returns
$S_{\min}$ as the intercept and $b_{\textrm{crit}}$ as the ratio of slope to
intercept. We write $b_{\textrm{crit}}$ for this fitted critical batch size,
to keep it apart from the noise scale measured from gradients. The fit
weights each run by the inverse of its step count, so that it matches every
point to the same relative precision rather than favoring the small-batch
runs with their large step counts.

```{.python .input #batch-size-the-critical-batch-size-1}
def fit_steps(bs, steps):
    """Fit S(b) = S_min (1 + b_crit / b) by least squares in 1/b."""
    b, S = np.array(bs, dtype=float), np.array(steps, dtype=float)
    ok = np.isfinite(S)  # Leave out runs that never reached the target
    slope, S_min = np.polyfit(1 / b[ok], S[ok], 1, w=1 / S[ok])
    assert S_min > 0, 'step counts never level off: no floor to fit'
    return S_min, slope / S_min

fits = {'TinyLM': fit_steps(lm_bs, lm_steps),
        'FashionCNN': fit_steps(cnn_bs, cnn_steps)}
for name, (S_min, b_crit) in fits.items():
    print(f'{name:10s}  S_min = {S_min:4.0f} steps  b_crit = {b_crit:4.0f}')
```

The fitted critical batch sizes are about a hundred sequences for `TinyLM`
and two to four hundred images for the CNN, of the same order as the
noise scales that each model reached during its first 500 steps. Closer
agreement is more than :eqref:`eq_steps-examples` promises here: the model
uses the curvature-weighted noise scale averaged over the whole run, while
we measured the unweighted ratio at two moments only.

Dividing each run's steps by $S_{\min}$ and its examples by
$E_{\min} = b_{\textrm{crit}} S_{\min}$ removes the two fitted constants.
Multiplying the two equations of :eqref:`eq_steps-examples` after
subtracting 1 from each shows that, if the model holds, every run of both
models lies on the single curve

$$\Big(\frac{E}{E_{\min}} - 1\Big)\Big(\frac{S}{S_{\min}} - 1\Big) = 1.$$

```{.python .input #batch-size-the-bill-in-examples}
x = np.logspace(0.005, 1.3, 100)  # E / E_min along the model curve
xs, ys = [x], [1 + 1 / (x - 1)]
for (S_min, b_crit), bs, steps in zip(fits.values(), (lm_bs, cnn_bs),
                                      (lm_steps, cnn_steps)):
    xs.append([b * S / (b_crit * S_min) for b, S in zip(bs, steps)])
    ys.append([S / S_min for S in steps])
d2l.plot(xs, ys, 'examples / minimum', 'steps / minimum', xscale='log',
         yscale='log', fmts=('-', 'o', 's'),
         legend=['model', 'TinyLM', 'FashionCNN'])
```

Part of this agreement is built in, because each model's two constants were
fitted to its own four or five runs, and a two-parameter curve can pass
through any two of them. What the plot tests is the shape: whether a
step count linear in $1/b$ fits all the runs, with two or three runs per
model to spare. The runs of both models scatter around the one curve.
Moving along it trades one resource for the other. Toward the upper left,
below the critical batch size, a run needs many steps and few examples.
Toward the lower right, it needs few steps and many examples. More total
work can thus reduce the number of sequential steps and, with enough
hardware, the time. Which point is best depends on what is scarce,
accelerators or days. This is why large training runs pick batch sizes near
their critical batch size, and why the exercises trace this tradeoff for
wall-clock time as well as for compute.

## Learning-Rate Rules

So far the learning rate moved with the batch by a rule we have not yet
justified. Some rule is needed, because a learning rate tuned at one batch
size is tuned for the noise of that batch size. Holding $\eta$ fixed in the
sweeps above would have mixed the effect of the batch with that of a
mistuned step. Two rules cover practice:

$$
\eta(b) = \frac{b}{b_0}\, \eta_0 \;\; \textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\; \textrm{(adaptive methods)}.
$$
:eqlabel:`eq_lr-rules`

Here $\eta_0$ is the learning rate tuned at batch size $b_0$. The first rule
moves the learning rate in proportion to the batch, the second in proportion
to its square root.

### Linear and Square-Root Scaling

The linear rule for SGD :cite:`Goyal.Dollar.Girshick.ea.2017` follows from
how SGD behaves near a minimum. With a constant learning rate, SGD does not
settle at the minimum but fluctuates around it in a *noise ball*. For a
quadratic with curvature $\lambda$ and gradient noise of variance $\sigma^2$
per example, :numref:`sec_sgd` found the expected squared distance from the
minimum to settle at about

$$\mathbb{E}\big[x_\infty^2\big] \approx \frac{\eta\, \sigma^2}{2\lambda b},$$

where the factor $1/b$ is the $1/b$ law for a minibatch of $b$ examples.
The noise ball depends on $\eta$ and $b$ only through $\eta / b$. A batch
$k$ times larger with a learning rate $k$ times larger therefore keeps the
same noise ball and covers the same distance per example, in $k$ times fewer
steps. The rule has a limit built in. In the noisy-quadratic model of
:eqref:`eq_steps-examples`, the best learning rate is
$\eta_{\textrm{opt}}(b) = \eta_{\max} / (1 + b_{\textrm{noise}} / b)$
:cite:`McCandlish.Kaplan.Amodei.ea.2018`, where $\eta_{\max}$ is the best
step for the exact gradient. It grows linearly with $b$ while
$b \ll b_{\textrm{noise}}$ and saturates beyond. The linear rule is the
small-batch half of this curve, and it stops working at the noise scale.

For Adam the rule is weaker. Adam divides each coordinate of its update by
the root-mean-square of that coordinate's recent gradients. While the noise
dominates, that denominator is about the noise amplitude
$\sigma / \sqrt{b}$, so a $k$ times larger batch shrinks it by $\sqrt{k}$
and already makes the signal in each step $\sqrt{k}$ times larger. Covering
the same distance per example in $k$ times fewer steps needs a further
factor of $\sqrt{k}$, and the square-root rule supplies it. The
stochastic-differential-equation analysis of
:citet:`Malladi.Lyu.Panigrahi.ea.2022` makes this precise. Growing the batch
by a factor $k$ preserves Adam's dynamics if every time constant moves with
it: $\eta' = \sqrt{k}\,\eta$, $1 - \beta_i' = k(1 - \beta_i)$, and
$\epsilon' = \epsilon / \sqrt{k}$. The condition on $\beta_i$ can be met only
while $k(1 - \beta_i) < 1$, that is, for $k < 10$ at $\beta_1 = 0.9$. Common
practice, and the sweeps of this section, move $\eta$ alone, which is the
form that :eqref:`eq_lr-rules` states.

### Learning-Rate Sweeps at Two Batch Sizes

We test both rules on the CNN with learning-rate grids at $b = 8$ and
$b = 64$. Every run sees the same budget of 12,800 examples, so the larger
batch takes eight times fewer steps. If a rule holds, the whole curve of final loss
against $\eta$ shifts to the right by the predicted factor when the batch
grows eightfold: by 8 for SGD and by $\sqrt{8} \approx 2.8$ for Adam. Both
batch sizes lie below the critical batch size of two to four hundred images
that we fitted for the CNN. There extra examples per step still turn into
fewer steps, so both rules should hold. Because `d2l.train_lm` already
trains for a fixed number of steps, the sweep takes a few lines. Each sweep
draws one seed, so all its runs start from the same initialization.

```{.python .input #batch-size-learning-rate-rules-1}
%%tab pytorch
def final_loss_at(make_optimizer, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    data_b = d2l.FashionMNIST(batch_size=b)
    final = []
    for lr in lrs:
        torch.manual_seed(seed)
        model = d2l.FashionCNN().to(device)
        d2l.train_lm(model, data_b, make_optimizer(model, lr),
                     num_steps=num_examples // b)
        loss = eval_loss(model, X_f[:8192], y_f[:8192])
        final.append(loss if math.isfinite(loss) else float('nan'))
    return final
```

```{.python .input #batch-size-learning-rate-rules-1}
%%tab jax
def final_loss_at(make_tx, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    data_b = d2l.FashionMNIST(batch_size=b)
    final = []
    for lr in lrs:
        model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        d2l.train_lm(model, data_b, optimizer,
                     num_steps=num_examples // b)
        loss = eval_loss(model, X_f[:8192], y_f[:8192])
        final.append(loss if math.isfinite(loss) else float('nan'))
    return final
```

First SGD with momentum 0.9, on a grid of learning rates spaced by factors
of two:

```{.python .input #batch-size-learning-rate-rules-2}
%%tab pytorch
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                             momentum=0.9)
seed = random.getrandbits(32)
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab jax
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda lr: optax.sgd(lr, momentum=0.9)
seed = random.getrandbits(32)
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

Then Adam:

```{.python .input #batch-size-learning-rate-rules-3}
%%tab pytorch
adam_lrs = [0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128,
            0.0256]
make_adam = lambda model, lr: torch.optim.Adam(model.parameters(), lr)
seed = random.getrandbits(32)
adam_small = final_loss_at(make_adam, adam_lrs, b=8, seed=seed)
adam_big = final_loss_at(make_adam, adam_lrs, b=64, seed=seed)
for b, loss in ((8, adam_small), (64, adam_big)):
    print(f'b={b:2d}: best lr {adam_lrs[np.nanargmin(loss)]:g}')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab jax
adam_lrs = [0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128,
            0.0256]
seed = random.getrandbits(32)
adam_small = final_loss_at(optax.adam, adam_lrs, b=8, seed=seed)
adam_big = final_loss_at(optax.adam, adam_lrs, b=64, seed=seed)
for b, loss in ((8, adam_small), (64, adam_big)):
    print(f'b={b:2d}: best lr {adam_lrs[np.nanargmin(loss)]:g}')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

Both plots show a broad minimum with a wall of failed runs on the right.
For SGD the larger batch moves the whole basin to the right: the best
learning rate moves by a factor of four to sixteen, the batch ratio to
within one grid step, so the linear rule carried over from $b = 8$ lands
within one grid step of the best learning rate at $b = 64$. For Adam the
basin is flatter, so one run locates its minimum less precisely, and where
the wall stands varies from run to run. The best learning rate moves by at
most a factor of four, and in some runs not at all: well short of the batch
ratio, as the square-root rule predicts. The basins are broad, and near the
optimum a factor of two in $\eta$ changes the loss little. The rules bridge
large ratios of batch sizes, and a short sweep around their prediction does
the fine tuning.

### Limits of the Rules

The rules fail in three places. Past the noise scale, the best learning
rate stops growing, because $\eta_{\textrm{opt}}(b)$ saturates at
$\eta_{\max}$, and a learning rate scaled further overshoots. The flattening
at the top of the CNN's steps-to-target sweep is consistent with this
mechanism, and the exercises reproduce it in isolation.

At the stability ceiling, curvature caps the learning rate. Gradient descent
on a function whose curvature is bounded by $L$ is guaranteed to descend
only for $\eta < 2/L$ (:numref:`sec_gd`), and this bound does not depend on
the batch. Once $\eta(b)$ reaches it, the optimum stops moving, so the rules
operate only in the room below the ceiling.

Early in training, the noise scale is at its smallest and the curvature often
at its largest, so a large batch with a scaled-up learning rate can diverge in
the first steps. Learning-rate warmup, which
:citet:`Goyal.Dollar.Girshick.ea.2017` introduced for this reason, starts from
a small $\eta$ and raises it over the first steps (:numref:`sec_scheduler`).

None of this is a universal law. Across many workloads the curves of
:eqref:`eq_steps-examples` keep their shape, but their critical batch sizes
range over orders of magnitude with the architecture, the dataset, and the
optimizer :cite:`Shallue.Lee.Antognini.ea.2019`. Large batches were once
also blamed for worse generalization, on the grounds that they converge to
sharp minima that test poorly :cite:`Keskar.Mudigere.Nocedal.ea.2017`. Much
of that gap closes when the learning rate and the length of training are
retuned for the larger batch :cite:`Shallue.Lee.Antognini.ea.2019`. The
lasting cost of a large batch is data efficiency, which is what this section
measured.

## Growing the Batch
:label:`subsec_batch-size-growing`

The noise scale is not a constant of the problem. We measured it growing
within 500 steps, by more than an order of magnitude for the language model,
and the reason holds at every scale:
$\|\nabla f\|$ falls as the easy progress is made, while the disagreement
between examples persists. A fixed batch size is therefore a compromise. A
batch chosen for the end of training exceeds the critical batch size at the
beginning and wastes examples there, while a batch chosen for the beginning
leaves parallelism unused later. Large language-model runs resolve this with
a *batch-size ramp*. GPT-3, for example, grew its batch from 32 thousand to
3.2 million tokens over the first several billion tokens of training
:cite:`brown2020language`. Measurements during pretraining find that the
critical batch size grows with the amount of data trained on rather than
with the model size :cite:`Zhang.Morwani.Vyas.ea.2024`, so the ramp can
also follow a measurement instead of a fixed schedule.
:citet:`Merrill.Arora.Groeneveld.ea.2025` measured the critical batch size
as training proceeded and doubled the batch each time the measurement
overtook it.

A batch ramp interacts with the learning-rate schedule. In the noise ball
of SGD above, $\eta$ and $b$ enter only through $\eta / b$, so raising $b$ at a fixed $\eta$ reduces the noise as
much as lowering $\eta$ at a fixed $b$. A batch ramp can therefore do part
of the work of learning-rate decay :cite:`Smith.Kindermans.Ying.ea.2018`,
and it is designed together with the schedule of :numref:`sec_scheduler`
rather than tuned as an independent knob. It also interacts with the
optimizer. On the pretraining workloads they studied,
:citet:`Shah.Polloreno.Stratos.ea.2025` find that Muon (:numref:`sec_muon`)
stays data-efficient up to larger batches than AdamW, which moves the
critical batch size of :eqref:`eq_steps-examples` to the right.

One question remains open. Fewer, larger steps save wall-clock time only if
each step is split across many devices that work in parallel. That is data
parallelism, the subject of :numref:`chap_performance`.

## Summary

A minibatch gradient is a signal plus noise, and the two have equal power
at the gradient-noise scale
$b_{\textrm{noise}} = \operatorname{tr} \boldsymbol{\Sigma} / \|\nabla f\|^2$,
which squared gradient norms at two batch sizes suffice to estimate. Below
it, doubling the batch roughly halves the steps to a target at constant
total work. Above it, the step count levels off at a floor and the extra
examples are wasted. Both models of this section followed the predicted
hyperbola, with a fitted critical batch size of the same order as their
measured noise scales. Moving between batch sizes requires moving the
learning rate, in proportion to the batch for SGD and to its square root for
Adam. These rules hold below the noise scale and under the stability
ceiling, and they need a warmup. The noise scale grows as the loss falls, which is why large runs
grow their batch during training.

## Exercises

1. [code] **The noise scale over training.** `noise_scale` of
   :numref:`subsec_batch-size-estimator` measured `TinyLM` at
   initialization and after 500 steps of Adam. Repeat that run for 3,000
   steps and measure the noise scale at steps 0, 25, 50, 100, 200, 500,
   1,000, 2,000, and 3,000. At each checkpoint also record the loss that
   `eval_loss` reports on `X_lm[::48][:2048]` and `Y_lm[::48][:2048]`.
    1. Plot the noise scale against this loss. Does the growth continue,
       slow down, or reverse?
    1. Under the rule of :numref:`subsec_batch-size-growing`, doubling $b$
       whenever $b_{\textrm{noise}}$ overtakes it, and starting from
       $b = 4$, at which losses would the doublings land? Interpolate
       $\log b_{\textrm{noise}}$ linearly in the loss between checkpoints.
       Explain why the step counts of this run at $b = 64$ would not tell
       a ramped run when to double.
   :exlabel:`ex_batch-size-noise-scale-over-training`

1. [code] **Past the elbow.** Extend the `TinyLM` sweep of
   :numref:`subsec_batch-size-steps` to $b = 1024$ under the square-root
   rule of :eqref:`eq_lr-rules`, then rerun $b = 1024$ with $\eta$ held at
   its $b = 256$ value.
    1. How do the step counts of the two $b = 1024$ runs compare with each
       other and with the $b = 256$ run?
    1. Explain both observations with :eqref:`eq_steps-examples` and the
       saturating optimal step size
       $\eta_{\textrm{opt}}(b) = \eta_{\max} / (1 + b_{\textrm{noise}}/b)$.

1. [code] **Retuning instead of scaling.** The CNN sweep recorded in
   `cnn_bs` and `cnn_steps` moved $\eta$ by the square-root rule rather than
   retuning it. For each batch size, run a three-point learning-rate grid
   spaced by factors of two around the rule's value and keep the best
   steps-to-target. Which points of the curve move? Refit with `fit_steps`:
   does $b_{\textrm{crit}}$ move, and by more than the difference between
   two repetitions of the original sweep?

1. **Time-optimal and compute-optimal batch size.** Suppose one
   optimizer step at batch size $b$ takes
   $t(b) = t_0 (1 + b / b_{\textrm{sat}})$ seconds, where $b_{\textrm{sat}}$
   is the batch size that saturates the device. Time-to-target is then
   $T(b) = t(b)\, S(b)$, and compute-to-target is the example count
   $E(b) = b\, S(b)$, which by :eqref:`eq_steps-examples` is smallest at
   the smallest batch. Using :eqref:`eq_steps-examples`, find the batch
   size that minimizes $T(b)$, and express its $E(b)$ as a multiple of
   $E_{\min}$.
   :exlabel:`ex_batch-size-time-optimal-compute-optimal`

1. [code] **Noise scale per parameter group.** ● The parameter groups of
   :numref:`subsec_tinylm` split `TinyLM`'s parameters into three groups:
   embeddings, matrices, and vectors (the biases and the LayerNorm scales
   and shifts). Restrict `grad_sq_norm` to one group at a time and measure
   its noise scale with `noise_scale`, at initialization and after 500
   steps of Adam.
    1. Which group has the largest noise scale, and does the ranking change
       during training?
    1. An embedding row receives a gradient only when its token occurs.
       Count the rows of `token_emb` that receive no gradient in a probe
       minibatch of 8 sequences, and state whether this sparsity can account
       for the embeddings' noise scale. Why does the same count for
       `pos_emb` always come out zero?
    1. What does the ranking imply for one global batch size serving all
       three groups?

1. [code] **The factor-of-two check.** ● Equation :eqref:`eq_steps-examples`
   predicts that examples-to-target reach twice their minimum at
   $b = b_{\textrm{noise}}$, where $b_{\textrm{noise}}$ is averaged over
   the run.
    1. From the checkpoints of problem
       :exref:`ex_batch-size-noise-scale-over-training`, average
       $b_{\textrm{noise}}$ over the training steps before the loss first
       falls to $1.5$, interpolating linearly between checkpoints.
    1. Compare this average with the $b_{\textrm{crit}}$ that `fit_steps`
       returns for `lm_bs` and `lm_steps`. By what factor do the two
       differ?
    1. Repeat the sweep and the checkpoint measurements with two more
       seeds. Is the gap between the two estimates larger than the
       seed-to-seed scatter of each?
   :exlabel:`ex_batch-size-factor-of-two`

1. [extended] **Measured time-compute tradeoff.** ● The step-time model
   $t(b)$ of problem :exref:`ex_batch-size-time-optimal-compute-optimal`
   has two constants, $t_0$ and $b_{\textrm{sat}}$, that depend on the
   hardware.
    1. Measure $t_0$ and $b_{\textrm{sat}}$ for `TinyLM` on your hardware
       with `d2l.Timer` by timing steps at several batch sizes.
    1. Combine $t(b)$ with `lm_bs` and `lm_steps` and plot $T(b)$ against
       $E(b)$. Compare the time-optimal batch size on this curve with the
       prediction of problem
       :exref:`ex_batch-size-time-optimal-compute-optimal`, evaluated at
       the run-averaged $b_{\textrm{noise}}$ of problem
       :exref:`ex_batch-size-factor-of-two`, and say by what factor the
       time-optimal batch cuts $T(b)$ relative to $b = 4$ and by what
       factor it raises $E(b)$.

[Discussions](https://d2l.discourse.group/t/36765)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.10]{.kicker}

Batch Size<br>
**the gradient-noise scale · steps against examples · learning-rate rules · batch ramps**
:::
:::

::: {.slide title="Quadrupling the Batch Saves Steps Only up to a Point"}
[The problem]{.kicker}

Train a Fashion-MNIST CNN to a fixed loss with Adam at several batch
sizes $b$:

- $b = 8 \to 32$: three to four times fewer steps, at most a third more
  examples.
- $b = 512 \to 2{,}048$: at most about a third fewer steps, so 2.5× to
  4.5× the examples.

A bigger batch means more work per step. It saves steps only while
gradient noise was what slowed the optimizer.

::: {.d2l-note .warn}
Where do the savings stop, and can we predict that point before training?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- estimate the gradient-noise scale from gradient norms at two batch sizes;
- predict steps and examples to a target from one hyperbola;
- fit the critical batch size to a sweep of training runs;
- move the learning rate with the batch: linear for SGD, square root for
  Adam, and say where each rule fails;
- explain why large runs grow their batch during training.

**Running example:** `TinyLM` (character-level transformer, 0.4M
parameters) and a Fashion-MNIST CNN (0.4M), trained to a target loss.
:::

::: {.slide title="Batch Size Decides How Much Gradient Noise Each Step Removes"}
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
| **Batch Size** | **how many examples per step** |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** the $1/b$ law, Adam, `TinyLM`, `FashionCNN`.
- **Skips:** splitting a batch across devices (data parallelism).
:::
:::
:::

::: {.slide title="A Minibatch Gradient Is Signal Plus Noise"}
Per-example noise power $\operatorname{tr}\boldsymbol{\Sigma} =
\mathbb{E}_i\|\nabla f_i - \nabla f\|^2$. Averaging $b$ examples:

$$\mathbb{E}\|\hat{\mathbf{g}}_b - \nabla f\|^2 =
\frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}.$$

Signal $\|\nabla f\|^2$ and noise power are equal at the
**gradient-noise scale**

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2}.$$

Example: $\operatorname{tr}\boldsymbol{\Sigma} = 100$, $\|\nabla f\|^2 = 1$
→ one example is 100× noisier than its signal; $b = 100$ brings them level.

Source: McCandlish et al. (2018).
:::

::: {.slide title="Two Batch Sizes Determine Both Unknowns"}
A minibatch gradient is longer than the true one by its noise power:

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 +
\frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$

Mean squared norms $N_s$, $N_l$ at batch sizes $b_s < b_l$:

$$\operatorname{tr} \boldsymbol{\Sigma} \approx
\frac{N_s - N_l}{1/b_s - 1/b_l},
\qquad
\|\nabla f\|^2 \approx \frac{b_l N_l - b_s N_s}{b_l - b_s}.$$

No full-dataset gradient needed: data-parallel workers already compute
both.
:::

::: {.slide title="Check: Estimate the Noise Scale from Two Norms"}
[Check]{.kicker}

Mean squared gradient norm $0.50$ at $b = 16$ and $0.103$ at
$b = 2{,}048$. What is $b_{\textrm{noise}}$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The big batch is nearly noise-free, so
$\|\nabla f\|^2 \approx 0.10$. The small batch's excess $0.40 \approx
\operatorname{tr}\boldsymbol{\Sigma}/16$ gives
$\operatorname{tr}\boldsymbol{\Sigma} \approx 6.4$. Hence
$b_{\textrm{noise}} \approx 64$.
:::
:::
:::

::: {.slide title="The Noise Scale Grows Within 500 Steps"}
CNN (images), then `TinyLM` (64-character sequences), at initialization
and after 500 steps of Adam:

@!batch-size-a-two-batch-estimator-3

@!batch-size-a-two-batch-estimator-4

The full gradient shrinks as the easy progress is made; the per-example
scatter does not. Large language models: around a million tokens
(Kaplan et al., 2020).
:::

::: {.slide title="Steps and Examples Trade Along One Hyperbola"}
Steps $S(b)$ to a target loss; examples $E(b) = b\,S(b)$:

$$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}}.$$

| $b_{\textrm{noise}} = 100$ | steps | examples |
|:--|:--|:--|
| $b = 10$ | $11\,S_{\min}$ | $1.1\,E_{\min}$ |
| $b = 100$ | $2\,S_{\min}$ | $2\,E_{\min}$ |
| $b = 1{,}000$ | $1.1\,S_{\min}$ | $11\,E_{\min}$ |

The bend at $b = b_{\textrm{noise}}$ is the **critical batch size**.
:::

::: {.slide title="Protocol: One Target, One Seed per Sweep, Rescaled Adam"}
- Target: a fixed loss on a fixed evaluation subset, checked every 10
  steps (minibatch losses would let noisy small-batch runs cross early).
- Adam, $\eta$ tuned at an anchor $b_0$, moved by $\eta_0\sqrt{b/b_0}$.
- One random seed per sweep: every batch size starts from the same
  initialization.
- `TinyLM`: target loss 1.5 (perplexity 4.5), $b = 4$ to 256 sequences.
- CNN: target loss 0.35, $b = 8$ to 2,048 images.
:::

::: {.slide title="TinyLM Saves Steps Steadily up to b = 64, Then Bends"}
@!batch-size-the-language-model

Up to $b = 64$ each quadrupling cuts the steps about threefold. From 64 to
256 sequences the steps only about halve, and the examples about double.
:::

::: {.slide title="The CNN Stops Gaining Steps Above a Few Hundred Images"}
@!batch-size-the-cnn

At 2,048 images, 4× the batch saves about a third of the steps at most;
examples six to twelve times their minimum.
:::

::: {.slide title="A Straight-Line Fit in 1/b Gives the Critical Batch Size"}
$S(b) = S_{\min} + S_{\min}\, b_{\textrm{crit}} / b$ is linear in $1/b$:
intercept $S_{\min}$, slope over intercept $b_{\textrm{crit}}$.

@!batch-size-the-critical-batch-size-1

Same order as the noise scales measured during the first 500 steps.
:::

::: {.slide title="Both Models Fall on One Normalized Curve"}
$\big(E/E_{\min} - 1\big)\big(S/S_{\min} - 1\big) = 1$, with
$E_{\min} = b_{\textrm{crit}} S_{\min}$:

@!batch-size-the-bill-in-examples

Upper left: many steps, few examples. Lower right: few steps, many
examples. Large runs sit near the bend.
:::

::: {.slide title="Check: Steps and Examples on Either Side of the Bend"}
[Check]{.kicker}

$b_{\textrm{noise}} = 100$. Compared with their minima, how many steps and
examples does a run need at $b = 25$? At $b = 400$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $b = 25$: $S = 5\,S_{\min}$, $E = 1.25\,E_{\min}$.
$b = 400$: $S = 1.25\,S_{\min}$, $E = 5\,E_{\min}$. Sixteen times the
batch cuts the steps fourfold and quadruples the examples.
:::
:::
:::

::: {.slide title="SGD Scales the Learning Rate Linearly, Adam by the Square Root"}
$$\eta(b) = \frac{b}{b_0}\, \eta_0 \;\;\textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\;\textrm{(Adam)}.$$

- SGD's noise ball $\mathbb{E}[x_\infty^2] \approx \eta\sigma^2/(2\lambda b)$
  depends on $\eta/b$ only: hold it fixed (Goyal et al., 2017).
- Adam divides by the gradient RMS $\approx \sigma/\sqrt{b}$: $k\times$ the
  batch already gives $\sqrt{k}\times$ the signal per step; $\eta$
  supplies the other $\sqrt{k}$ (Malladi et al., 2022).
- Best step $\eta_{\max}/(1 + b_{\textrm{noise}}/b)$ saturates: the linear
  rule expires at the noise scale.
:::

::: {.slide title="SGD: Eight Times the Batch Moves the Best Rate About Eightfold"}
CNN, budget of 12,800 examples, SGD with momentum 0.9:

@!batch-size-learning-rate-rules-2

The basin shifts right by the batch ratio to within one grid step (4× to
16× on this grid).
:::

::: {.slide title="Adam: The Best Rate Moves Far Less Than the Batch"}
Same budget, Adam:

@!batch-size-learning-rate-rules-3

$\sqrt{8} \approx 2.8$: in a flat basin the best rate moves at most
fourfold, sometimes not at all, well short of the linear rule's 8×.
:::

::: {.slide title="The Rules Fail Past the Noise Scale, at the Ceiling, and Early"}
- **Past the noise scale:** the best step saturates; a scaled $\eta$
  overshoots.
- **At the stability ceiling:** curvature caps $\eta$ below $2/L$ at any
  batch size; there the optimum stops moving.
- **Early in training:** smallest noise scale, often the worst curvature:
  warmup.
- Critical batch sizes vary by orders of magnitude across workloads
  (Shallue et al., 2019). The lasting cost of a large batch is **data
  efficiency**.
:::

::: {.slide title="Large Runs Grow Their Batch as the Noise Scale Grows"}
- $b_{\textrm{noise}}$ grows as the loss falls: a fixed batch is too big
  early and too small late.
- GPT-3: 32 thousand → 3.2 million tokens over the first billions of
  tokens.
- Measured ramps: double $b$ whenever the measured critical batch size
  overtakes it (Merrill et al., 2025).
- Raising $b$ reduces SGD's noise like lowering $\eta$: design the ramp
  with the schedule.
- Muon stays data-efficient to larger batches than AdamW (Shah et al.,
  2025).
:::

::: {.slide title="Check: Rescale Two Learning Rates"}
[Check]{.kicker}

At $b = 64$, SGD trains well with $\eta = 0.05$ and Adam with
$\eta = 0.003$. You move to $b = 256$. Which learning rates do the rules
suggest, and what could still go wrong for SGD?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** SGD: $4 \times 0.05 = 0.2$. Adam: $\sqrt{4} \times 0.003 =
0.006$. SGD's $0.2$ may exceed the stability ceiling $2/L$, or $b = 256$
may lie past the noise scale; warmup protects the first steps.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Noise scale** $b_{\textrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}
  / \|\nabla f\|^2$, from gradient norms at two batch sizes: about a hundred
  images (CNN) and a few sequences (`TinyLM`) at initialization, growing
  within 500 steps (`TinyLM`: more than tenfold).
- **Steps to a target:** perfect scaling below the critical batch size, a
  floor above; both models fit one hyperbola, $b_{\textrm{crit}}$ about
  a hundred sequences (`TinyLM`) and two to four hundred images (CNN).
- **Learning rate:** linear in $b$ for SGD, $\sqrt{b}$ for Adam; below the
  noise scale, under the ceiling, behind warmup.
- **Ramps:** the noise scale grows, so large runs grow their batch.

**Next:** keeping the best learning rate fixed as the model grows wider.
:::
