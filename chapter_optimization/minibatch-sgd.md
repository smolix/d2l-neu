# Minibatches
:label:`sec_minibatch_sgd`

:numref:`sec_gd` and :numref:`sec_sgd` are the two ends of a spectrum.
Gradient descent pays a full pass over the dataset for one exact update;
stochastic gradient descent pays for a single example and takes a noisy step.
Every model we have trained in this book actually did something in between:
it averaged the gradient over a *minibatch* of a few dozen to a few hundred
examples. The statistical half of the justification appeared in
:numref:`sec_sgd`, where we measured that averaging $b$ independent gradients
cuts the variance by a factor of $b$. This section supplies the computational
half: on modern processors, $b$ examples at once cost far less than $b$
examples one at a time, for reasons of caches, vector units, and dispatch
overhead rather than statistics. Along the way we build the equipment the
rest of the chapter trains with — a timer, a small real regression dataset,
and a harness that accepts any optimizer written as an update rule — and we
finish by racing gradient descent, SGD, and minibatch SGD against the wall
clock.

## Vectorization and Caches

The bluntest reason to batch is parallel hardware. Training on multiple GPUs
and multiple servers requires sending at least one example to each device,
so with 8 GPUs per server and 16 servers the minibatch is already no smaller
than 128 if every device is to contribute.

On a single GPU or CPU the reason is subtler: arithmetic is abundant and
memory traffic is not. A current server CPU sustains on the order of
$10^{12}$ to $10^{13}$ floating point operations per second across its cores
and vector units, yet its memory interface delivers a few hundred gigabytes
per second. A GPU is more lopsided still: on the order of $10^{14}$
operations per second in single precision — and another order of magnitude
through its matrix units at the reduced precisions used for deep learning —
against a few terabytes per second of memory bandwidth. In both cases
arithmetic outruns bandwidth by roughly two orders of magnitude, so keeping
the processor busy requires each byte fetched from memory to take part in
tens to hundreds of operations before being evicted.

Processors bridge the gap with a hierarchy of memories: a small number of
registers, then L1, L2, and often L3 caches of increasing size and latency
and decreasing bandwidth, the largest shared among cores. Access patterns
matter as much as volume: the first access to a region of memory is
expensive, while the sequential reads that follow (a *burst read*) are
comparatively cheap. See this
[Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) for a more
in-depth discussion. Whether the hierarchy helps is a property of the
algorithm, not the hardware: data must be *reused* while it is still
resident. Consider matrix multiplication, $\mathbf{A} = \mathbf{B}\mathbf{C}$.
We have several options for computing $\mathbf{A}$:

1. We could compute $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$, i.e., elementwise by means of dot products.
1. We could compute $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$, i.e., one column at a time. Likewise we could compute $\mathbf{A}$ one row $\mathbf{A}_{i,:}$ at a time.
1. We could simply compute $\mathbf{A} = \mathbf{B} \mathbf{C}$ in one go.
1. We could break $\mathbf{B}$ and $\mathbf{C}$ into smaller block matrices and compute $\mathbf{A}$ one block at a time.

Option 1 fetches a row and a column for every single output element, and
since matrices are laid out linearly in memory, one of the two is read from
widely scattered addresses. Option 2 keeps the column vector
$\mathbf{C}_{:,j}$ in cache while traversing $\mathbf{B}$, halving the memory
traffic. Option 3 is best, but most matrices do not fit into cache — that is
the problem we started with. Option 4 is the practical answer: move blocks of
both matrices into cache and multiply them there, so every loaded byte is
reused across a whole block of outputs. Optimized libraries do this for us.

Memory is not the only overhead. Every operation launched from Python pays
for the interpreter, the framework's bookkeeping, and, on a GPU, a kernel
launch — microseconds per operation, against the nanoseconds of arithmetic a
small operation actually contains. The remedy is the same as for memory:
fewer, larger operations. Let's measure how much all of this matters in
practice.

```{.python .input #minibatch-sgd-vectorization-and-caches-1}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input #minibatch-sgd-vectorization-and-caches-1}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
import time

A = jnp.zeros((256, 256))
B = jnp.array(np.random.normal(0, 1, (256, 256)))
C = jnp.array(np.random.normal(0, 1, (256, 256)))
```

Since we will benchmark the running time frequently in the rest of the book, let's define a timer.

```{.python .input #minibatch-sgd-vectorization-and-caches-2}
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Elementwise assignment (option 1) iterates over all rows and columns of
$\mathbf{B}$ and $\mathbf{C}$ respectively to assign the result to
$\mathbf{A}$ one dot product at a time.

```{.python .input #minibatch-sgd-vectorization-and-caches-3}
%%tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input #minibatch-sgd-vectorization-and-caches-3}
%%tab jax
# Compute A = BC one element at a time. JAX is functionally pure, so a
# literal `A.at[i, j].set(...)` would copy the full matrix on every write
# (O(n^2) memory traffic), turning a demo into a multi-minute run. We
# instead use a NumPy buffer to mirror the eager semantics PyTorch gets
# for free; the *point* of this cell is that the elementwise dispatch
# is much slower than vectorized matmul.
A = np.zeros((256, 256), dtype=np.float32)
B_np = np.array(B)
C_np = np.array(C)
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B_np[i, :], C_np[:, j])
timer.stop()
```

A faster strategy is column-wise assignment (option 2).

```{.python .input #minibatch-sgd-vectorization-and-caches-4}
%%tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input #minibatch-sgd-vectorization-and-caches-4}
%%tab jax
# Compute A = BC one column at a time. We keep B/C on device; only the
# Python loop and per-column dispatch cost remain.
A = jnp.zeros((256, 256))
timer.start()
for j in range(256):
    A = A.at[:, j].set(jnp.dot(B, C[:, j]))
A.block_until_ready()
timer.stop()
```

Last, we perform the entire operation in one call (option 3). Multiplying two
matrices $\mathbf{B} \in \mathbb{R}^{m \times n}$ and
$\mathbf{C} \in \mathbb{R}^{n \times p}$ takes approximately $2mnp$ floating
point operations, when scalar multiplication and addition are counted
separately (fused in practice). Multiplying two $256 \times 256$ matrices
thus takes $0.03$ billion floating point operations. Let's see the respective
speeds.

```{.python .input #minibatch-sgd-vectorization-and-caches-5}
%%tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input #minibatch-sgd-vectorization-and-caches-5}
%%tab jax
# Compute A = BC in one go
timer.start()
A = jnp.dot(B, C)
A.block_until_ready()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

:begin_tab:`pytorch`
On the CPU the three strategies land orders of magnitude apart: the
elementwise loop runs in the megaflop range, the single library call in the
gigaflops. Nothing about the arithmetic changed between them — only how much
of it we exposed to the library per call, and hence how much overhead and
memory traffic each floating point operation had to carry.
:end_tab:

:begin_tab:`jax`
On a GPU these matrices are tiny: even the full multiplication returns in
roughly the time it takes to *launch* it, so the column, full, and block
variants all measure the dispatch overhead rather than arithmetic throughput
— the overhead wall of this section, seen from the other side. (The
elementwise variant ran in NumPy on the CPU and is not comparable.) Scale the
matrices up to $4096 \times 4096$ and the one-call version pulls orders of
magnitude ahead; the exercises ask you to try.
:end_tab:

## Minibatch Gradients
:label:`sec_minibatches`

The same economics applies when the operands are training examples rather
than matrix columns. Processing one observation at a time means many
matrix–vector (or even vector–vector) products, each carrying the full
dispatch overhead — during inference just as during training. That is the
computational case against the single-example update
$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ with

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w}).$$

Its minibatch counterpart averages over a set $\mathcal{B}_t$ of
$b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ examples drawn uniformly at
random from the training set:

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{b} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}).$$

Statistically this changes exactly two things. The expectation is untouched:
the minibatch gradient is as unbiased as the single-example one. The variance
drops by a factor of $b$, since we average $b$ independent draws;
equivalently, the noise amplitude shrinks by $b^{-1/2}$. This is the $1/b$
law we measured on a real network in :numref:`sec_sgd`.

Batching therefore helps twice, and the two reasons deserve to be kept apart.
The *hardware* reason is this section's: $b$ examples share one pass over the
weights and one round of dispatch overhead, so the cost per example falls
steeply as $b$ grows from 1 and flattens once the device is saturated. The
*statistical* reason is :numref:`sec_sgd`'s: a quieter gradient. But
amplitude only falls as $b^{-1/2}$, so spending $100\times$ more compute per
step buys a $10\times$ quieter direction. Both effects saturate, and neither
tells us when a bigger batch stops converting into faster training. That
question, how large is too large, depends on the optimization dynamics
themselves; it has a name, the *critical batch size*, and gets its own
treatment in :numref:`sec_batch_size`. In practice one picks $b$ large
enough to keep the device busy and small enough to fit its memory. When the
batch you want exceeds memory, gradients can be *accumulated* over several
forward–backward passes before a single update. This reproduces the gradient
of a larger batch only if the details line up: the per-pass losses must be
scaled so their sum is the mean over the full effective batch, the optimizer
step and any gradient clipping must wait until after the last pass rather than
fire per micro-batch, and layers whose forward pass couples the examples in a
batch — batch normalization above all, and stochastic layers such as dropout —
still see only the micro-batch, not the larger one. It is a systems technique
we return to in :numref:`chap_performance`.

To see the hardware side in isolation, we perform the same matrix
multiplication as before, but broken into "minibatches" of 64 columns at a
time.

```{.python .input #minibatch-sgd-minibatches}
%%tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input #minibatch-sgd-minibatches}
%%tab jax
timer.start()
for j in range(0, 256, 64):
    A = A.at[:, j:j+64].set(jnp.dot(B, C[:, j:j+64]))
A.block_until_ready()
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Computation on the minibatch is essentially as efficient as on the full
matrix: 64 columns at a time is already enough work per dispatch to amortize
the overhead. One caveat before transferring this intuition wholesale to
training: layers that compute statistics *across* the batch — batch
normalization (:numref:`sec_batch_norm`) being the prominent case — change
behavior as $b$ grows, since the noise they inject shrinks with the batch;
see :citet:`Ioffe.2017` for how to rescale the relevant terms.

## Reading the Dataset

The experiments in the rest of this chapter run on a small real regression
task: predicting aircraft wing
[self-noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
from five physical features, a dataset collected by NASA. We use the first
$1{,}500$ examples and whiten the data, removing the mean and rescaling the
variance to $1$ per coordinate. It is deliberately modest: each training run
takes seconds, so we can afford to rerun it under every optimizer the chapter
introduces.

```{.python .input #minibatch-sgd-reading-the-dataset}
%%tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input #minibatch-sgd-reading-the-dataset}
%%tab jax
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (jnp.array(data[:n, :-1]), jnp.array(data[:n, -1])),
        batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Implementation from Scratch

Recall the minibatch update derived in :numref:`sec_linear_regression` and
implemented from scratch in :numref:`sec_linear_scratch`.
Here we make it slightly more general, giving it the call signature that
every optimizer in this chapter will share: a `states` input holding whatever
auxiliary variables the algorithm carries (SGD carries none; momentum and
Adam will), and a `hyperparams` dictionary. The training function averages
the loss over each minibatch, so the update rule never needs to divide by the
batch size.

```{.python .input #minibatch-sgd-implementation-from-scratch-1}
%%tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        with torch.no_grad():
            p.sub_(hyperparams['lr'] * p.grad)
        p.grad.zero_()
```

```{.python .input #minibatch-sgd-implementation-from-scratch-1}
%%tab jax
def sgd(params, grads, states, hyperparams):
    updated = []
    for param, grad in zip(params, grads):
        updated.append(param - hyperparams['lr'] * grad)
    return updated
```

Next, a generic training function. It initializes a linear regression model
and trains it with any update rule of the above signature, plotting the loss
against elapsed wall-clock time — the axis that actually matters when
comparing optimizers.

```{.python .input #minibatch-sgd-implementation-from-scratch-2}
%%tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input #minibatch-sgd-implementation-from-scratch-2}
%%tab jax
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = jnp.array(np.random.normal(scale=0.01, size=(feature_dim, 1)),
                  dtype=jnp.float32)
    b = jnp.zeros(1)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    # JIT only the grad computation; the optimizer update runs eagerly so
    # that stateful optimizers can mutate `states` without triggering JAX
    # tracer-leak errors from closure side-effects inside jit.
    @jax.jit
    def compute_grads(w, b, X, y):
        def loss_fn(w, b):
            return d2l.squared_loss(d2l.linreg(X, w, b), y).mean()
        return jax.grad(loss_fn, argnums=(0, 1))(w, b)
    # Pre-stack the full dataset on device so the periodic evaluate_loss
    # stays inside one compiled call instead of looping in Python.
    eval_batches = [(jnp.array(X), jnp.array(y)) for X, y in data_iter]
    Xs = jnp.concatenate([X for X, _ in eval_batches], axis=0)
    ys = jnp.concatenate([y for _, y in eval_batches], axis=0)
    @jax.jit
    def full_eval(w, b):
        out = d2l.linreg(Xs, w, b)
        y_r = ys.reshape(out.shape)
        return ((out - y_r) ** 2 / 2).mean()
    for _ in range(num_epochs):
        for X, y in data_iter:
            X, y = jnp.array(X), jnp.array(y)
            grads = compute_grads(w, b, X, y)
            w, b = trainer_fn([w, b], list(grads), states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (float(full_eval(w, b)),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Now we can race the extremes against the middle. First, batch gradient
descent: setting the minibatch size to 1500, the full dataset, updates the
parameters once per epoch. Progress stalls after roughly six steps — each
step is well aimed, but there are too few of them.

```{.python .input #minibatch-sgd-implementation-from-scratch-3}
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

At the opposite extreme, batch size 1 is stochastic gradient descent: 1500
updates per epoch, at a constant (and necessarily small) learning rate. The
loss falls quickly at first and then the decline slows. Both procedures
process 1500 examples per epoch, but SGD takes *more clock time per epoch*
than gradient descent here: it dispatches 1500 tiny operations where gradient
descent dispatches one large one — the overhead story of the previous
section, paid at every step.

```{.python .input #minibatch-sgd-implementation-from-scratch-4}
sgd_res = train_sgd(0.005, 1)
```

With a batch size of 100 the time per epoch drops below both extremes.

```{.python .input #minibatch-sgd-implementation-from-scratch-5}
mini1_res = train_sgd(.4, 100)
```

Reducing the batch size to 10 raises the time per epoch again: the work per
dispatch is getting too small to run efficiently.

```{.python .input #minibatch-sgd-implementation-from-scratch-6}
mini2_res = train_sgd(.05, 10)
```

Plotting loss against wall-clock time for all four experiments makes the
trade explicit. SGD converges fastest *per example processed*, yet reaches a
given loss *slower than gradient descent by the clock*, because it computes
gradients one example at a time. Minibatch SGD takes both savings at once:
batch size 10 beats pure SGD, and batch size 100 beats even gradient descent
on runtime.

```{.python .input #minibatch-sgd-implementation-from-scratch-7}
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10], xscale='log',
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
```

## Concise Implementation

Every framework ships these loops behind an optimizer object. Using it makes
the training function shorter and less error-prone; we wrap the pattern into
a generic function used throughout this chapter.

```{.python .input #minibatch-sgd-concise-implementation-1}
%%tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input #minibatch-sgd-concise-implementation-1}
%%tab jax
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nnx.Linear(5, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(
        net, trainer_fn(**hyperparams), wrt=nnx.Param)

    loss = lambda pred, y: jnp.mean((pred - y) ** 2) / 2
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    # JIT-fuse the per-batch optimizer update so per-step Python overhead
    # stays out of the inner loop.
    @nnx.jit
    def step(model, optimizer, X, y):
        def loss_fn(model):
            out = model(X)
            y_reshaped = y.reshape(out.shape)
            return jnp.mean((out - y_reshaped) ** 2) / 2
        l, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return l

    # Pre-stack the full dataset on device so the periodic full-loss
    # evaluation is a single compiled call.
    eval_batches = [(jnp.array(X), jnp.array(y)) for X, y in data_iter]
    Xs = jnp.concatenate([X for X, _ in eval_batches], axis=0)
    ys = jnp.concatenate([y for _, y in eval_batches], axis=0)
    @nnx.jit
    def full_eval(model):
        out = model(Xs)
        y_r = ys.reshape(out.shape)
        return jnp.mean((out - y_r) ** 2) / 2
    for _ in range(num_epochs):
        for X, y in data_iter:
            X, y = jnp.array(X), jnp.array(y)
            step(net, optimizer, X, y)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (float(full_eval(net)),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Repeating the batch-size-10 experiment through the framework optimizer shows
the same behavior.

```{.python .input #minibatch-sgd-concise-implementation-2}
%%tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input #minibatch-sgd-concise-implementation-2}
%%tab jax
data_iter, _ = get_data_ch11(10)
trainer = optax.sgd
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Summary

Minibatches are cheap for mechanical reasons: a batched operation shares one
round of dispatch overhead and one pass over the weights across $b$ examples,
and blocked computation keeps data in cache, where the same processor runs
orders of magnitude faster than when it waits on main memory. Combined with
the $1/b$ variance reduction measured in :numref:`sec_sgd`, this is why
minibatch SGD dominates both of its parents on the wall clock, as the race in
this section showed. Choosing $b$ to fill the device without exhausting its
memory captures most of the benefit; how far the statistics allow batch size
to grow before the returns vanish — the critical batch size — is the subject
of :numref:`sec_batch_size`.

## Exercises

1. Modify the batch size and learning rate and observe the rate of decline for the value of the objective function and the time consumed in each epoch.
1. In the blocked matrix multiplication benchmark, vary the block width over $\{1, 4, 16, 64, 256\}$ and time each variant. Where does throughput saturate, and why does it saturate well before the full width of 256? Then repeat the element/column/full comparison with $4096 \times 4096$ matrices on a GPU and explain what changes.
1. Compare minibatch stochastic gradient descent with a variant that actually *samples with replacement* from the training set. What happens?
1. An evil genie replicates your dataset without telling you (i.e., each observation occurs twice and your dataset grows to twice its original size, but nobody told you). How does the behavior of stochastic gradient descent, minibatch stochastic gradient descent and that of gradient descent change?
1. Implement gradient accumulation on top of `train_ch11`: sum gradients over $k$ consecutive batches of size $b$ and update once with their average. Verify that the loss trajectory matches a run with batch size $kb$ when plotted against examples processed, then compare the two against wall-clock time.

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1068)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1069)
:end_tab:

<!-- slides -->

::: {.slide title="Minibatches"}
GD: $\mathcal{O}(n)$ per step, exact. SGD: $\mathcal{O}(1)$ per step, noisy.
Everything we ever trained sat in between — a **minibatch** of $b$ examples:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{b} \sum_{i \in \mathcal{B}} \nabla f_i(\mathbf{w}).$$

- Statistics (last section): variance $\propto 1/b$.
- **This section: the mechanics** — why $b$ at once costs far less than
  $b$ one at a time.
:::

::: {.slide title="Arithmetic outruns memory"}
- Server CPU: $10^{12}$–$10^{13}$ FLOP/s vs a few $100$ GB/s of bandwidth.
- GPU: $10^{14+}$ FLOP/s vs a few TB/s.
- Ratio ≈ **two orders of magnitude**: each byte loaded must feed tens to
  hundreds of operations.

. . .

Caches bridge the gap **only if the algorithm reuses resident data** —
blocked matmul, not elementwise loops.

Plus dispatch overhead: every op launched from Python costs microseconds;
the arithmetic inside a tiny op costs nanoseconds.
:::

::: {.slide title="Setup"}
@minibatch-sgd-vectorization-and-caches-1

. . .

@minibatch-sgd-vectorization-and-caches-2
:::

::: {.slide title="Three loops, three speeds"}
Compute $\mathbf{A} = \mathbf{B}\mathbf{C}$ on $256 \times 256$
matrices, exposing more work per call each time:

@minibatch-sgd-vectorization-and-caches-3

. . .

@minibatch-sgd-vectorization-and-caches-4

. . .

@minibatch-sgd-vectorization-and-caches-5

Same arithmetic, orders of magnitude apart. The loop is overhead;
the cache and vector units do the work.
:::

::: {.slide title="Batching in blocks"}
64 columns at a time — a "minibatch" of the matmul:

@minibatch-sgd-minibatches

. . .

Already as fast as the full multiplication: modest batches amortize
essentially all the overhead.

- Two reasons to batch, kept apart: **hardware** (this section) and
  **variance** (last section). Both saturate.
- How large is too large → *critical batch size*, later in this chapter.
- Batch exceeds memory? **Gradient accumulation** (ch. on performance).
:::

::: {.slide title="Airfoil dataset"}
Real regression data for the whole chapter — 1500 examples, 5 features,
whitened; every run takes seconds:

@minibatch-sgd-reading-the-dataset
:::

::: {.slide title="The optimizer interface"}
Every optimizer in this chapter: `(params, states, hyperparams)` —
`states` carries the algorithm's memory (empty for SGD):

@minibatch-sgd-implementation-from-scratch-1
:::

::: {.slide title="Generic training harness"}
Linear regression + any update rule; loss recorded against
**wall-clock time**:

@minibatch-sgd-implementation-from-scratch-2
:::

::: {.slide title="The race: full batch vs single example"}
$b = 1500$: one well-aimed update per epoch — stalls after ~6 steps:

@minibatch-sgd-implementation-from-scratch-3

. . .

$b = 1$: 1500 updates per epoch, but 1500 tiny dispatches — more clock
time per epoch than GD:

@minibatch-sgd-implementation-from-scratch-4
:::

::: {.slide title="The middle wins"}
@minibatch-sgd-implementation-from-scratch-5

. . .

@minibatch-sgd-implementation-from-scratch-6

. . .

@minibatch-sgd-implementation-from-scratch-7

Read the x-axis as elapsed seconds: $b=100$ beats *both* extremes.
:::

::: {.slide title="Concise: framework optimizer"}
Same experiment through the built-in optimizer — the harness the rest of
the chapter reuses:

@minibatch-sgd-concise-implementation-1

. . .

@minibatch-sgd-concise-implementation-2
:::

::: {.slide title="Recap"}
- Minibatch SGD interpolates between GD and SGD and beats both on the
  wall clock.
- The win is mechanical: dispatch amortization, cache reuse, vector units
  — plus the $1/b$ variance cut from last section.
- Pick $b$ to fill the accelerator within memory; the *statistical*
  ceiling (critical batch size) comes later in the chapter.
:::
