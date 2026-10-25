```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Reproducibility and Inspection
:label:`sec_repro`

Suppose we change one line of a training script and the validation accuracy
of a Fashion-MNIST classifier rises by 0.3 points. Before we credit the
change, we need to know how much two runs of the unchanged script differ.
For the classifiers of :numref:`chap_classification` and
:numref:`chap_perceptrons`, they differ by a few tenths of a point, because
each run draws different initial weights and visits the examples in a
different order. A run that fails is a second problem of the same kind. If
the loss turns into NaN after hours of training, we want to repeat the run
exactly up to that point and then look inside the model to see which layer
produced the first non-finite number.

This section first makes runs repeatable. Every random draw comes from a
generator whose sequence of numbers is fixed by a seed, so seeding the
generators fixes the draws; repeating a run bit for bit also requires that
the arithmetic adds numbers in a fixed order. The section then looks inside
running models with *hooks*, functions that observe each layer of an
unmodified model as it computes. The two halves depend on each other: a
measurement taken inside a model can be compared across two versions of the
code only if the runs differ in nothing but that code.

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab mxnet
import random
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab pytorch
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab tensorflow
import random
import tensorflow as tf
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab jax
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Seeds and Random Number Generators
:label:`subsec_seeds_randomness`

Randomness enters a training run in more places than one might list at
first: initialization draws every weight (:numref:`sec_init_param`), dropout
draws a new mask at every step, the data loader shuffles the examples
differently in every epoch, data augmentation draws crops and flips, and
parallel loader workers do some of this work in separate processes. Each
draw comes from a *pseudorandom number generator*, a deterministic algorithm
whose sequence of outputs is fixed by its starting state, the *seed*.
Seeding every generator that a run consults therefore fixes every draw.

### Global Seeds

PyTorch, TensorFlow, and MXNet each keep a global generator that their
random functions consult unless told otherwise; JAX has none.

:begin_tab:`mxnet`
MXNet keeps one global generator per device. `mnp.random.seed(k)` seeds all
of them in one call, so initialization, sampling, and dropout are fixed on
every device. It does not seed Python's `random` module or NumPy's global
generator, which keep their own state; this matters once the data pipeline
uses them. The function below draws everything, weights and data, from
MXNet's generators, so one call fixes all its draws:
:end_tab:

:begin_tab:`pytorch`
`torch.manual_seed(k)` seeds PyTorch's global generator on the CPU and on
every GPU. It does not seed Python's `random` module or NumPy's global
generator (`random.seed`, `np.random.seed`), nor NumPy generator objects
created with `np.random.default_rng`, which carry seeds of their own. The
function below draws everything, weights and data, from PyTorch's
generator, so one call fixes all its draws:
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.utils.set_random_seed(k)` seeds Python's `random` module, NumPy's
global generator, and TensorFlow's global generator in one call, the three
that a typical training script consults. It does not seed generator objects
that carry their own state, `tf.random.Generator` objects and NumPy
generators from `np.random.default_rng`. The function below draws everything,
weights and data, from the seeded global state, so one call fixes all its
draws:
:end_tab:

:begin_tab:`jax`
JAX has no global generator. Every random function takes a *key*, a value
that determines its output: `jax.random.normal(key, shape)` returns the same
numbers every time it is called with the same key, and new randomness comes
only from deriving new keys with `jax.random.split`. The function below
turns its seed into one key and splits it three ways, one key each for
initialization, inputs, and targets, so the seed fixes all its draws:
:end_tab:

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab mxnet
def train_once(seed):
    mnp.random.seed(seed)  # seeds the generator on every device
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'), nn.Dense(1))
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.1})
    X = mnp.random.normal(0, 1, (128, 20))
    y = mnp.random.normal(0, 1, (128, 1))
    for _ in range(5):
        with autograd.record():
            loss = ((net(X) - y) ** 2).mean()
        loss.backward()
        trainer.step(1)  # the loss is already a mean
    return loss.item(), net[0].weight.data().copy()

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      loss_a == loss_b, bool((w_a == w_b).all()))
print('different seed:', loss_a, 'vs', loss_c)
```

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab pytorch
def train_once(seed):
    torch.manual_seed(seed)  # seeds the CPU and every GPU
    net = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    X, y = torch.randn(128, 20), torch.randn(128, 1)
    for _ in range(5):
        opt.zero_grad()
        loss = ((net(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return loss.item(), net[0].weight.detach().clone()

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      loss_a == loss_b, torch.equal(w_a, w_b))
print('different seed:', loss_a, 'vs', loss_c)
```

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab tensorflow
def train_once(seed):
    tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy, and TF
    with tf.device('/CPU:0'):  # see Deterministic Arithmetic below
        net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)])
        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        X, y = tf.random.normal((128, 20)), tf.random.normal((128, 1))
        for _ in range(5):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((net(X) - y) ** 2)
            grads = tape.gradient(loss, net.trainable_variables)
            opt.apply_gradients(zip(grads, net.trainable_variables))
        return float(loss), tf.identity(net.layers[0].kernel)

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      loss_a == loss_b, bool(tf.reduce_all(w_a == w_b)))
print('different seed:', loss_a, 'vs', loss_c)
```

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab jax
def train_once(seed):
    key_init, key_X, key_y = jax.random.split(jax.random.key(seed), 3)
    rngs = nnx.Rngs(key_init)
    net = nnx.Sequential(nnx.Linear(20, 32, rngs=rngs),
                         nnx.relu,
                         nnx.Linear(32, 1, rngs=rngs))
    X = jax.random.normal(key_X, (128, 20))
    y = jax.random.normal(key_y, (128, 1))
    optimizer = nnx.Optimizer(net, optax.sgd(0.1), wrt=nnx.Param)
    loss_fn = lambda model: ((model(X) - y) ** 2).mean()
    for _ in range(5):
        loss, grads = nnx.value_and_grad(loss_fn)(net)
        optimizer.update(net, grads)
    return loss, net.layers[0].kernel[...]

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      bool(loss_a == loss_b), bool(jnp.array_equal(w_a, w_b)))
print('different seed:', loss_a, 'vs', loss_c)
```

The two runs with seed 0 agree *bitwise*, down to the last bit of the loss
and of every weight: same initial weights, same data, same gradients, same
operations in the same order. The run with seed 1 ends with a different
loss.

:begin_tab:`tensorflow`
We placed the TensorFlow run on the CPU. On a GPU, the same seeded code
gives equal losses but, in some runs, weights that differ in their last
bits; :numref:`subsec_deterministic_arithmetic` explains why.
:end_tab:

### Generator Objects

A single global generator couples all the code that draws from it. Inserting
one extra random call, a new layer's initialization or a stray draw while
debugging, shifts every number drawn after it, so the data split, the
initialization, and the dropout masks all change at once. A *generator
object* is a private stream with its own seed, which unrelated draws cannot
shift.

:begin_tab:`mxnet`
MXNet has no generator objects, and no sampling function accepts one. The
substitute is to seed again immediately before the draws that must stay
fixed. Seeding again also resets the stream for everything drawn after it,
which a private stream would not do:
:end_tab:

:begin_tab:`pytorch`
A `torch.Generator` is such a stream. Here a random permutation, say a split
of a dataset into training and validation examples, stays fixed no matter
what else draws from the global generator in between:
:end_tab:

:begin_tab:`tensorflow`
A `tf.random.Generator` is such a stream. Here a random permutation, say a
split of a dataset into training and validation examples, stays fixed no
matter what else draws from the global generator in between (the generator
has no permutation method, so we sort uniform random numbers):
:end_tab:

:begin_tab:`jax`
In JAX the keys are the generator objects: each key is a private stream that
we hold as a value, and `jax.random.split` produces as many independent
streams as we need. No draw changes any shared state, so unrelated draws
cannot shift ours. Here a draw with another key between two uses of the
same key changes nothing:
:end_tab:

```{.python .input #reproducibility-inspection-generator-objects}
%%tab mxnet
mnp.random.seed(42)
split = mnp.arange(10)
mnp.random.shuffle(split)  # mnp has no permutation; shuffle in place
_ = mnp.random.normal(0, 1, (1000,))  # unrelated draws from the same stream
mnp.random.seed(42)  # seed again where the draws must stay fixed
split_again = mnp.arange(10)
mnp.random.shuffle(split_again)
print(bool((split == split_again).all()), split)
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab pytorch
g = torch.Generator().manual_seed(42)
split = torch.randperm(10, generator=g)
_ = torch.randn(1000)  # unrelated draws from the global generator
split_again = torch.randperm(10, generator=g.manual_seed(42))
print(torch.equal(split, split_again), split)
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab tensorflow
g = tf.random.Generator.from_seed(42)
split = tf.argsort(g.uniform((10,)))  # a random permutation from g's stream
_ = tf.random.normal((1000,))  # unrelated draws from the global generator
split_again = tf.argsort(tf.random.Generator.from_seed(42).uniform((10,)))
print(bool(tf.reduce_all(split == split_again)), split.numpy())
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab jax
key_split, key_other = jax.random.split(jax.random.key(42))
split = jax.random.permutation(key_split, 10)
_ = jax.random.normal(key_other, (1000,))  # unrelated draws, their own key
split_again = jax.random.permutation(key_split, 10)
print(bool(jnp.array_equal(split, split_again)), split)
```

:begin_tab:`mxnet`
Gluon has no `generator=` argument: code that must own its randomness owns a
seed instead and calls `mnp.random.seed` itself, resetting everything drawn
after it.
:end_tab:

:begin_tab:`pytorch`
Most sampling functions accept `generator=`, and so does `DataLoader`, where
it sets the shuffle order. We use that next.
:end_tab:

:begin_tab:`tensorflow`
Sampling methods live on the generator itself (`g.normal`, `g.uniform`), so
code that must own its randomness takes a `Generator` argument. The input
pipeline instead takes a seed directly, which we use next.
:end_tab:

:begin_tab:`jax`
Every function in `jax.random` takes the key as its first argument; no
variant consults hidden state.
:end_tab:

### Data Loader Workers

Data loaders often prepare minibatches in parallel *worker* processes. A
worker that draws random numbers, for data augmentation for instance, needs
a stream of its own. Two failures are common. If every worker starts as a copy of the
main process, as with the *fork* start method, Linux's default before
Python 3.14, the workers inherit the same generator
state and apply the same "random" augmentations. If each worker seeds itself
from the operating system's entropy, no two runs agree.

:begin_tab:`mxnet`
Gluon's `DataLoader` runs `num_workers > 0` on a pool of worker processes,
each started as a copy of the main process, NumPy's generator state
included. The pool does not seed the workers, and Gluon has no hook for a
worker initializer. Per-worker randomness therefore has to live in the
dataset: seeding a generator from the example's index inside `__getitem__`
makes the augmentation a function of the example, whichever worker computes
it. The shuffle order of `DataLoader(shuffle=True)` comes from NumPy's
global generator, so fixing it takes `np.random.seed`; MXNet's seed does
not reach it.
:end_tab:

:begin_tab:`pytorch`
`DataLoader` gives every worker its own PyTorch seed, derived from a base
seed and the worker's index, and seeds Python's `random` module and NumPy's
global generator in each worker from it. It cannot reach generator objects
that a dataset or a library creates for itself; those need a *worker
initializer*, a function that the loader calls once in each new worker. The
loader's `generator` sets both the base seed and the shuffle order, so one
generator fixes the whole input pipeline. The initializer below follows
PyTorch's reproducibility notes. For `random` and NumPy it repeats what the
loader already does; it shows where a dataset's own generator objects would
be replaced:
:end_tab:

:begin_tab:`tensorflow`
`tf.data` avoids both failures by design. It runs its parallel work in
threads of one process, so no worker inherits a copied generator, and its
randomness is an explicit argument, `Dataset.shuffle(buffer_size,
seed=...)`. With the default `reshuffle_each_iteration=True`, a seeded
pipeline produces a *different* order in each epoch, but the same sequence
of orders every time the pipeline is built:
:end_tab:

:begin_tab:`jax`
With explicit keys, the workers' streams appear in the program: a worker's
randomness is whatever key we hand it. We split one key into per-worker keys
with `jax.random.split(key, num_workers)` and derive new ones in each epoch
with `jax.random.fold_in(key, epoch)`, which mixes the epoch number into the
key. Two workers can share a stream only if the code passes them the same
key. JAX programs often borrow their input pipeline from PyTorch or
`tf.data`; such loaders bring their own rules for seeding workers, given in
the tabs of those frameworks.
:end_tab:

```{.python .input #reproducibility-inspection-dataloader-workers-2}
%%tab pytorch
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # this worker's torch seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator().manual_seed(0)
data = TensorDataset(torch.arange(8))
# num_workers=0 keeps this cell portable; with num_workers > 0, each
# worker process calls seed_worker once when it starts
loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0,
                    worker_init_fn=seed_worker, generator=g)
print([batch[0].tolist() for batch in loader])
g.manual_seed(0)
print([batch[0].tolist() for batch in loader])
```

```{.python .input #reproducibility-inspection-dataloader-workers-3}
%%tab tensorflow
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # epoch 1
print([b.numpy().tolist() for b in ds])  # epoch 2: reshuffled, still seeded
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # rebuilt pipeline: epoch 1 again
```

:begin_tab:`pytorch`
Seeding the generator again reproduces the shuffle order.
:end_tab:

:begin_tab:`tensorflow`
`reshuffle_each_iteration=False` keeps one order for every epoch, and
leaving `seed` unset draws new orders in each run. Parallel `map` and
`interleave` also accept `deterministic=False`, which passes elements on in
the order they finish, for speed; the default, `True`, keeps the order of
the output fixed.
:end_tab:

### Randomness as a Value

The worker failures above arise because the state of a generator is copied
or shared without appearing in the program's data flow. JAX makes randomness
a value instead: every random function takes a key, the same key gives the
same numbers, and independent streams come from splitting a key. A key can
still be reused by mistake, but the reuse is visible in the code. The cost
is that every function that draws random numbers must accept a key and pass
split keys on. A global seed is the convenient form of the same contract: one
implicit key that all code shares. Generator objects recover the explicit
form where it matters.

:begin_tab:`mxnet`
MXNet stands at the implicit end: global streams per device, no generator
objects, and no stateless functions. Seeding once at the start, and again at
any point where draws must stay fixed, is all it offers.
:end_tab:

:begin_tab:`pytorch`
PyTorch offers both forms: the global generator, and `torch.Generator`
objects passed to the functions that should use them.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow offers both forms: the stateful functions above, with a global
generator and `tf.random.Generator` objects, and stateless functions that
take the key as an argument, such as
`tf.random.stateless_normal(shape, seed=[k1, k2])`, which returns the same
numbers for the same pair of seeds.
:end_tab:

:begin_tab:`jax`
The two cells above use this design: `train_once` turned one seed into one
key and split it, and reusing a key reproduced the permutation.
:end_tab:

## Deterministic Arithmetic
:label:`subsec_deterministic_arithmetic`

Seeding fixes which numbers the program draws. It does not fix how the
arithmetic combines them. Floating-point addition is not associative
(:numref:`sec_numerics`): every sum is rounded to the nearest representable
number, so adding the same numbers in a different order can give a different
result.

The smallest example takes three float32 numbers. Near $10^8$, adjacent
float32 numbers are 8 apart, so $-10^8 + 1$ rounds back to $-10^8$:

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab mxnet
a, b, c = [mnp.array(v, dtype='float32') for v in (1e8, -1e8, 1.0)]
print(float((a + b) + c), float(a + (b + c)))  # same numbers, regrouped
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab pytorch
a, b, c = [torch.tensor(v, dtype=torch.float32) for v in (1e8, -1e8, 1.0)]
print(float((a + b) + c), float(a + (b + c)))  # same numbers, regrouped
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab tensorflow
a, b, c = [tf.constant(v, dtype=tf.float32) for v in (1e8, -1e8, 1.0)]
print(float((a + b) + c), float(a + (b + c)))  # same numbers, regrouped
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab jax
a, b, c = [jnp.array(v, dtype=jnp.float32) for v in (1e8, -1e8, 1.0)]
print(float((a + b) + c), float(a + (b + c)))  # same numbers, regrouped
```

Regrouping changes the result from 1 to 0. In a long sum the effect is
smaller but of the same kind: adding a million random numbers forward and
backward usually gives results that differ in their last bits.

On a CPU, the order in which one operation adds its numbers is usually fixed
for a given number of threads, so a seeded run repeats. On a GPU, many
threads add partial sums at the same time, and some operations combine them
with *atomic additions*, which commit each partial sum in whatever order the
threads happen to finish. Examples are the gradients of an embedding lookup
or of a scatter operation, some convolution gradients, and some reductions.
Two seeded runs on the same machine can then differ in the last bits of
their weights, and over many training steps such differences grow until the
loss curves differ visibly. Each framework except MXNet has a switch that
replaces such operations with deterministic ones, which add in a fixed order
and are often slower.

:begin_tab:`mxnet`
MXNet has no such switch. On a GPU, bitwise repetition requires avoiding the
operations whose kernels add in a varying order.
:end_tab:

:begin_tab:`pytorch`
`torch.use_deterministic_algorithms(True)` is the switch. Operations that
have a deterministic implementation use it; operations without one raise a
`RuntimeError` instead of returning nondeterministic results. On a GPU, the
environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` must also be set, or
matrix multiplications raise as well. The backward pass of 3-D average
pooling on a GPU has no deterministic implementation:
:end_tab:

:begin_tab:`tensorflow`
`tf.config.experimental.enable_op_determinism()` is the switch. Operations
that have a deterministic implementation use it, and operations without one
raise `tf.errors.UnimplementedError`. Random operations must then have a
seed, since an operation that seeds itself from the operating system is
nondeterministic too. The switch is meant to be called when the program
starts, because it affects operations as they are created, and nothing turns
it off again short of a new process. We can still demonstrate the seed rule
by clearing the global seed:
:end_tab:

:begin_tab:`jax`
JAX's random numbers are reproducible across devices: a draw is a function
of the key and the shape only, so the same key gives the same numbers on
CPU, GPU, and TPU (JAX does not promise the same bits across its own
releases). For arithmetic, the operations of `train_once` added in a fixed
order on the GPU too. Setting the environment variable
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` before JAX starts replaces
GPU operations that add in a varying order with deterministic ones; unlike
the switches of PyTorch and TensorFlow, it does not raise errors.
:end_tab:

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab pytorch
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    try:
        x = torch.randn(1, 1, 2, 2, 2, requires_grad=True, device='cuda')
        nn.AvgPool3d(1)(x).sum().backward()
    except RuntimeError as e:
        print(str(e).split(', but')[0])
else:
    print('CPU run: every kernel used here is deterministic;',
          'on a GPU, operations without a deterministic kernel raise')
torch.use_deterministic_algorithms(False)
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab tensorflow
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(None)  # forget the global seed for a moment
try:
    tf.random.normal((2,))
except RuntimeError as e:
    print(str(e).split('.')[0])
# determinism stays on and needs a seed, though not a particular one
tf.keras.utils.set_random_seed(random.SystemRandom().getrandbits(31))
```

Bitwise agreement holds only for one machine, one library version, and one
configuration of these switches; no framework promises it across releases,
across hardware, or between CPU and GPU. Bitwise reproducibility is
therefore a tool for debugging: it lets us find the step at which two runs
first differ. The scientific goal is weaker and more robust. A conclusion,
such as "the change raises accuracy", should hold across seeds, so we report
the mean and the spread over several runs rather than one run. As with the
number formats of :numref:`sec_numerics`, a conclusion that changes with the
seed or with the last bits of the arithmetic needs more evidence.

## Model Inspection with Hooks

A repeatable run tells us that a problem will recur, not where it arises.
To find out, we observe what each layer computes during the run. Editing the
`forward` method of every layer to print or store its output is tedious, and
impossible for a model that came from a library or a checkpoint whose code
we do not want to change. A *hook* is a function that the framework calls at
a fixed point of each layer's computation, from outside the layer's code.

:begin_tab:`mxnet`
In :numref:`sec_model_construction` we saw that `net(X)` does not call
`forward` directly: it calls `__call__`, which wraps `forward`. Hooks run in
that wrapper. `block.register_forward_hook(f)` arranges for
`f(block, inputs, output)` to run after every forward pass of the block,
and `register_forward_pre_hook` before it. :numref:`fig_bg_hooks` shows where
they run. Gluon ignores the value a hook returns, so its hooks observe
but cannot modify the output. Gluon's own `Block.summary()` builds its table
of layers by registering a forward hook on every block.
:end_tab:

:begin_tab:`pytorch`
In :numref:`sec_model_construction` we saw that `net(X)` does not call
`forward` directly: it calls `__call__`, which wraps `forward`. Hooks run in
that wrapper. `module.register_forward_hook(f)` arranges for
`f(module, inputs, output)` to run after every forward pass of the module,
and `register_forward_pre_hook` before it. :numref:`fig_bg_hooks` shows where
they run. A forward hook that returns a value replaces the module's output.
:end_tab:

:begin_tab:`tensorflow`
Keras also wraps `call` in a `__call__` method, but the wrapper offers no
hooks: nothing can be attached to an unmodified model to observe it. Two
idioms reach the same measurements. In a *functional* model, built by
calling layers on symbolic inputs, every intermediate tensor is an object we
can name, so a second `tf.keras.Model` over the same graph can return any of
them as an output; it shares all weights with the first. When we own the
model's code, an overridden `call` can record the values we want.
:numref:`fig_bg_hooks` shows where a hook would run; in TensorFlow the
observation points must be built into the model.
:end_tab:

:begin_tab:`jax`
`nnx.capture` wraps a call of a model and records, alongside the output, the
value that every submodule returned, without any change to the model's code.
:numref:`fig_bg_hooks` shows where such an observation point sits in the
computation of each module.
:end_tab:

![The `__call__` wrapper as a pipeline: input flows through pre-hooks, then forward, then hooks, to the output, with the two hook stages dashed and orange against forward's solid blue, and a side arrow from the hooks stage to an observer that can capture, check, or modify.](../img/bg-hooks.svg)
:label:`fig_bg_hooks`

As a test model we use a stack of eight residual blocks of width 64 between
an input layer with 20 inputs and an output layer with 10 outputs. The block
is the residual block of :numref:`sec_model_construction`, unchanged: it adds
the output of a small network, its *body*, to its input: $\mathbf{h} = \mathbf{x} +
\textrm{body}(\mathbf{x})$, where the body is a linear layer, a ReLU, and a
second linear layer. In every framework the weights start from Xavier
initialization (:numref:`subsec_xavier`) and the biases at zero.

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab mxnet
mnp.random.seed(random.SystemRandom().getrandbits(31))  # unpin the demo seed

class ResidualBlock(nn.Block):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(nn.Dense(num_hiddens, activation='relu'),
                      nn.Dense(num_hiddens))

    def forward(self, X):
        return X + self.body(X)

net = nn.Sequential()
net.add(nn.Dense(64), *[ResidualBlock(64) for _ in range(8)], nn.Dense(10))
net.initialize(init.Xavier())  # Xavier weights, zero biases
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab pytorch
torch.seed()  # a fresh random seed: the demos above left one pinned

class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(num_hiddens, num_hiddens),
                                  nn.ReLU(),
                                  nn.Linear(num_hiddens, num_hiddens))

    def forward(self, X):
        return X + self.body(X)

net = nn.Sequential(nn.Linear(20, 64),
                    *[ResidualBlock(64) for _ in range(8)],
                    nn.Linear(64, 10))
for m in net.modules():
    if isinstance(m, nn.Linear):  # Xavier weights, zero biases
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab tensorflow
def dense(num_outputs, **kwargs):  # Xavier weights, zero biases
    return tf.keras.layers.Dense(
        num_outputs, kernel_initializer='glorot_uniform', **kwargs)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.body = tf.keras.Sequential([
            dense(num_hiddens, activation='relu'), dense(num_hiddens)])

    def call(self, X):
        return X + self.body(X)

inputs = tf.keras.Input(shape=(20,))
taps = [dense(64)(inputs)]  # keep every intermediate tensor
for _ in range(8):
    taps.append(ResidualBlock(64)(taps[-1]))
outputs = dense(10)(taps[-1])
net = tf.keras.Model(inputs, outputs)
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab jax
xavier = nnx.initializers.xavier_uniform()  # biases start at zero

class ResidualBlock(nnx.Module):
    def __init__(self, num_hiddens, *, rngs):
        self.body = nnx.Sequential(
            nnx.Linear(num_hiddens, num_hiddens, kernel_init=xavier,
                       rngs=rngs),
            nnx.relu,
            nnx.Linear(num_hiddens, num_hiddens, kernel_init=xavier,
                       rngs=rngs))

    def __call__(self, X):
        return X + self.body(X)

rngs = nnx.Rngs(d2l.get_key())
net = nnx.Sequential(nnx.Linear(20, 64, kernel_init=xavier, rngs=rngs),
                     *[ResidualBlock(64, rngs=rngs) for _ in range(8)],
                     nnx.Linear(64, 10, kernel_init=xavier, rngs=rngs))
X = jax.random.normal(d2l.get_key(), (256, 20))
```

### Activation Statistics
:label:`subsec_activation_stats`

The initialization experiments of :numref:`sec_init_param` measured the
standard deviation of the activations at every depth. With hooks, the same
measurement takes a few lines on an unmodified model. We record the standard
deviation of each layer's output on a minibatch of 256 standard normal
inputs.

```{.python .input #reproducibility-inspection-capturing-activation-statistics}
%%tab mxnet
stats, handles = [], []

def record(block, inputs, output):
    stats.append((type(block).__name__, output.std().item()))

for m in net:
    handles.append(m.register_forward_hook(record))
net(mnp.random.normal(0, 1, (256, 20)))
for h in handles:
    h.detach()
for name, std in stats:
    print(f'{name:15s} std {std:.2f}')
```

```{.python .input #reproducibility-inspection-capturing-activation-statistics}
%%tab pytorch
stats, handles = [], []

def record(module, inputs, output):
    stats.append((type(module).__name__, output.detach().std().item()))

for m in net:
    handles.append(m.register_forward_hook(record))
net(torch.randn(256, 20))
for h in handles:
    h.remove()
for name, std in stats:
    print(f'{name:15s} std {std:.2f}')
```

```{.python .input #reproducibility-inspection-capturing-activation-statistics}
%%tab tensorflow
probe = tf.keras.Model(inputs, taps + [outputs])  # same layers, same weights
X = tf.random.normal((256, 20))
names = ['Dense'] + ['ResidualBlock'] * 8 + ['Dense']
for name, out in zip(names, probe(X)):
    print(f'{name:15s} std {float(tf.math.reduce_std(out)):.2f}')
```

```{.python .input #reproducibility-inspection-capturing-activation-statistics}
%%tab jax
_, inter = nnx.capture(
    net, nnx.Intermediate, method_outputs=nnx.Intermediate)(X)
for k, layer in enumerate(net.layers):
    out = inter['layers'][k]['__call__'][0]
    print(f'{type(layer).__name__:15s} std {out.std():.2f}')
```

The standard deviation grows from block to block, from about 0.7 after the
input layer to roughly 3 to 4 after the eighth block. Each block adds its
body's output to its input, and the two are nearly uncorrelated, so their
variances add. With Xavier initialization the body's output has about half
the variance of its input: the first linear layer keeps the variance, the
ReLU halves the mean square, and the second linear layer keeps it again. The
variance therefore grows by a factor of about 1.5 per block, and the
standard deviation by about $\sqrt{1.5} \approx 1.22$. This growth with
depth is why the residual branches of deep networks are often initialized
with smaller weights (:numref:`sec_init_param`).

:begin_tab:`mxnet`
Two rules keep hooks safe. First, *detach before storing*: the hook above
reduces the output to a Python number at once. A hook that stores arrays
should store `output.detach()`, because inside `autograd.record` a stored
output keeps the computation graph of every forward pass in memory until
the list is cleared. Second, *keep the handle and detach it*:
`register_forward_hook` returns a handle, and the hook stays registered for
the block's lifetime unless we call `handle.detach()`. Hooks run in the
Python `__call__` wrapper, so they do not fire inside a block that was
compiled with `hybridize()`; we observe blocks without hybridizing them.
:end_tab:

:begin_tab:`pytorch`
Two rules keep hooks safe. First, *detach before storing*: the hook above
stores `output.detach().std()`. Storing `output` itself would keep the
autograd graph of every forward pass in memory until `stats` is cleared, a
leak that grows with every minibatch. Second, *keep the handle and remove
it*: a hook stays registered for the module's lifetime unless we call
`handle.remove()`, slowing every later forward pass and keeping alive any
tensors it stores.
:end_tab:

:begin_tab:`tensorflow`
Nothing needs to be detached or removed: `probe` shares the original model's
layers and weights, its outputs are ordinary tensors outside any gradient
tape, and no observer stays attached to the model. The method requires a
functional model. A subclassed model, whose `call` is ordinary Python code,
has no symbolic intermediate tensors to return, so its `call` must record
the values itself.
:end_tab:

:begin_tab:`jax`
Nothing needs to be detached or removed: the captured values are ordinary
arrays, the dictionary exists for this one call, and no observer stays
attached to the model. `method_outputs=nnx.Intermediate` records the return
value of every module call. For finer control, a module can call
`self.sow(nnx.Intermediate, 'name', value)` at chosen points in its
computation, and `nnx.capture(model, nnx.Intermediate)` then returns only
those values.
:end_tab:

### A NaN Finder
:label:`subsec_nan_finder`

When a loss becomes NaN at step 40,000, the useful question is which layer
produced the first non-finite value. A hook on every layer that checks its
output for infinities and NaNs answers it in one forward pass: the first
hook to fire names the layer, because hooks fire in the order the layers
compute. To test this, we put a NaN into one weight of the fourth layer of
the stack, the third residual block:

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab mxnet
def make_finite_check(name):
    def check(block, inputs, output):
        if not mnp.isfinite(output).all():
            raise RuntimeError(f'first non-finite output in {name}')
    return check

leaves = ([('0', net[0])]
          + [(f'{k}.body.{j}', net[k].body[j])
             for k in range(1, 9) for j in (0, 1)]
          + [('9', net[9])])
handles = [blk.register_forward_hook(make_finite_check(name))
           for name, blk in leaves]
saved = net[3].body[1].weight.data()[0, 0].item()
net[3].body[1].weight.data()[0, 0] = float('nan')  # break one layer
try:
    net(mnp.random.normal(0, 1, (2, 20)))
except RuntimeError as e:
    print(e)
for h in handles:
    h.detach()
net[3].body[1].weight.data()[0, 0] = saved  # restore the weight
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab pytorch
def make_finite_check(name):
    def check(module, inputs, output):
        if not torch.isfinite(output).all():
            raise RuntimeError(f'first non-finite output in {name}')
    return check

handles = [m.register_forward_hook(make_finite_check(name))
           for name, m in net.named_modules()
           if len(list(m.children())) == 0]  # leaf modules only
saved = net[3].body[0].weight[0, 0].item()
with torch.no_grad():
    net[3].body[0].weight[0, 0] = float('nan')  # break one layer
try:
    net(torch.randn(2, 20))
except RuntimeError as e:
    print(e)
for h in handles:
    h.remove()
with torch.no_grad():
    net[3].body[0].weight[0, 0] = saved  # restore the weight
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab tensorflow
class Checked(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        self.seq, self.first_bad = layers, None  # a plain list is tracked

    def call(self, X):
        for layer in self.seq:
            X = layer(X)
            if self.first_bad is None and not bool(
                    tf.reduce_all(tf.math.is_finite(X))):
                self.first_bad = layer.name
        return X

checked = Checked([dense(64)]
                  + [ResidualBlock(64, name=f'block{k}') for k in range(1, 9)]
                  + [dense(10)])
_ = checked(tf.random.normal((2, 20)))  # build the weights
checked.seq[3].body.layers[0].kernel[0, 0].assign(float('nan'))  # break one
_ = checked(tf.random.normal((2, 20)))
print('first non-finite output in', checked.first_bad)
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab jax
saved = net.layers[3].body.layers[0].kernel[0, 0]
net.layers[3].body.layers[0].kernel[0, 0] = float('nan')  # break one layer
_, inter = nnx.capture(
    net, nnx.Intermediate, method_outputs=nnx.Intermediate)(X)

def get_path(tree, path):
    for key in path:
        tree = tree[key]
    return tree

for path, module in nnx.iter_modules(net):
    if isinstance(module, nnx.Linear):
        out = get_path(inter, path)['__call__'][0]
        if not jnp.isfinite(out).all():
            print('first non-finite output in', path)
            break

net.layers[3].body.layers[0].kernel[0, 0] = saved  # restore the weight
```

:begin_tab:`mxnet`
The report names `3.body.1`, the layer that holds the NaN. Gluon blocks carry
no path names and there is no `named_modules`, so we list the leaves
ourselves. We put the NaN into the *second* linear layer of the block:
Gluon's `Dense(..., activation='relu')` applies the ReLU inside the same
kernel, and that kernel maps NaN to 0, so a NaN in the first layer would be
gone before any hook saw it. A hook must sit where the condition it checks
is still visible.
:end_tab:

:begin_tab:`pytorch`
The report names `3.body.0`, the layer that holds the NaN, without a single
print statement in the model.
:end_tab:

:begin_tab:`tensorflow`
The report names `block3`, the block that holds the NaN. Because the check
is part of `call`, it runs on every forward pass until we edit it out, the
price of building observation into the model. For a temporary search,
`tf.debugging.enable_check_numerics()` checks the output of every operation
and reports the first one that produces an infinity or a NaN.
:end_tab:

:begin_tab:`jax`
The report names `('layers', 3, 'body', 'layers', 0)`, the layer that holds
the NaN. We checked the captured outputs of the linear layers in the order
of the module tree, which for this sequential network is the order in which
they compute. For a search during training, `jax.config.update('jax_debug_nans',
True)` makes JAX raise an error at the first operation that produces a NaN.
:end_tab:

### Gradient Inspection

Gradients can be inspected in the same way, for instance to log the size of
each layer's gradient or to find where the gradients start to explode. As a
loss we take the mean squared output of the stack on a minibatch of 256
standard normal inputs.

:begin_tab:`mxnet`
Gluon has no backward hooks, so nothing can observe gradients *while* the
backward pass computes them. After `loss.backward()`, however, every
parameter's gradient is available as `param.grad()`, and per-layer gradient
norms are a loop over `net.collect_params()`. Code that must act between the
backward pass and the update goes into the training step; Gluon's `Trainer`
splits `step` into `allreduce_grads` and `update` so that such code can
stand between them. Here are the gradient norms of the first linear layer
of the third residual block:
:end_tab:

:begin_tab:`pytorch`
`module.register_full_backward_hook(f)` runs `f(module, grad_input,
grad_output)` while the backward pass runs, once the gradients with respect
to the module's inputs are computed. A hook that returns a value replaces
`grad_input`. Like a forward hook, a backward hook returns a handle to
remove and should detach anything it stores. The gradients of the
parameters themselves are available after the backward pass as `p.grad`.
Here are the gradient norms of the first linear layer of the third residual
block:
:end_tab:

:begin_tab:`tensorflow`
TensorFlow has no backward hooks and needs none: `tf.GradientTape` returns
the gradient of every variable as a value, which we inspect or transform
like any other tensor. Here are the gradient norms of the first linear layer
of the third residual block:
:end_tab:

:begin_tab:`jax`
JAX needs no backward hooks: `nnx.grad` returns the gradients as a tree of
values with the same structure as the parameters, which we inspect or
transform like any other data. Here are the gradient norms of the first
linear layer of the third residual block:
:end_tab:

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab mxnet
X = mnp.random.normal(0, 1, (256, 20))
with autograd.record():
    loss = (net(X) ** 2).mean()
loss.backward()
layer = net[3].body[0]  # block 3, first layer
print(f'weight {float(mnp.linalg.norm(layer.weight.grad())):.3g}, '
      f'bias {float(mnp.linalg.norm(layer.bias.grad())):.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab pytorch
X = torch.randn(256, 20)
net.zero_grad()
(net(X) ** 2).mean().backward()
layer = net[3].body[0]  # block 3, first layer
print(f'weight {layer.weight.grad.norm().item():.3g}, '
      f'bias {layer.bias.grad.norm().item():.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab tensorflow
with tf.GradientTape() as tape:
    loss = tf.reduce_mean(net(X) ** 2)
grads = tape.gradient(loss, net.trainable_variables)
kernel, bias = grads[10:12]  # block 3, first layer
print(f'kernel {float(tf.norm(kernel)):.3g}, bias {float(tf.norm(bias)):.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab jax
grads = nnx.grad(lambda model: (model(X) ** 2).mean())(net)
layer = grads['layers'][3]['body']['layers'][0]  # block 3, first layer
print(f"kernel {float(jnp.linalg.norm(layer['kernel'][...])):.3g}, "
      f"bias {float(jnp.linalg.norm(layer['bias'][...])):.3g}")
```

We can now repeat a run bit for bit on one machine, say which sources of
randomness and which operations stand in the way, and observe any layer of
an unmodified model during a run. What these tools cannot do is make two
different machines agree, or decide on their own whether a difference
between two runs matters; for that we compare results across seeds.

## Summary

Every random draw in a training run comes from a generator fixed by a seed:
initialization, dropout, shuffling, augmentation, and loader workers all
consult one. Seeding each generator the run uses fixes the draws, and private
generator objects or explicit keys keep unrelated draws from shifting them.
Seeding does not fix the arithmetic: floating-point addition depends on the
order of the terms, and some GPU operations add in a varying order, so
bitwise repetition also needs deterministic operations and holds only on one
machine and one software version. It is a tool for debugging; scientific
conclusions should hold across seeds. Hooks, or their equivalents in each
framework, observe the outputs and gradients of every layer of an unmodified
model, which turns questions such as "where did the first NaN appear?" into
one forward pass.

## Exercises

:begin_tab:`mxnet`
1. [code] **Block ablation.** Gluon ignores a hook's return value, so an
   output-replacing hook cannot be written.
    1. Zero the output of one residual block's body by editing
       `ResidualBlock.forward`, which turns the block into the identity,
       and measure the relative change in `net`'s output on a fixed batch.
       Repeat for every block, and rank the blocks by this change.
    1. Rebuild `net` three more times with new random weights and repeat
       the ranking. State whether the conclusion "block $k$ matters most"
       survives, and what evidence such a claim would need.
2. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$ of `net`, with each variance taken over all
   entries. On one batch, record for every `ResidualBlock` the variance of
   $\mathbf{x}_k$ with `register_forward_pre_hook` (the pre-hook receives
   the argument tuple, so unpack it), of $\mathbf{h}_k$ with
   `register_forward_hook`, and of $\mathbf{b}_k$ with a forward hook on
   its `body`. The standard deviations in the table of
   :numref:`subsec_activation_stats` grow because each block adds its
   body's output to the stream. Measure how closely

   $$\mathrm{Var}[\mathbf{h}_k] = \mathrm{Var}[\mathbf{x}_k] + \mathrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
3. [code] **Seeds and data loading.** ● Extend `train_once` to load its
   data through Gluon's `DataLoader` with `shuffle=True` and
   `num_workers=4`, keeping the single `mnp.random.seed(seed)` call at its
   top. MXNet has no generator objects, so the dataset uses NumPy: it adds
   noise to each example inside `__getitem__`, drawn from an
   `np.random.default_rng()` object created in the dataset's constructor.
    1. Predict whether two runs with the same seed agree on the shuffle
       order and on the noise, and whether the four workers draw the same
       noise. Justify each prediction from where that randomness is drawn,
       then verify all three.
    1. Gluon has no worker initializer. Make the noise a deterministic
       function of the sample by seeding a generator from the sample index
       inside `__getitem__`, fix the shuffle order, and verify that two
       runs now agree. Then check whether an example's noise changes from
       one epoch to the next.
4. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
5. [code] **Gradient norms.** ● Rebuild `net`, multiply the weight of
   every `Dense` layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. After `loss.backward()`, read each parameter's gradient through
       `net.collect_params()` and compute per-block parameter-gradient
       norms. To obtain each block's output gradient, run the stack one
       block at a time, attaching gradients to the intermediate outputs.
       Plot both families of norms against depth.
    1. Express the gradient of a linear layer's weight through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

:begin_tab:`pytorch`
1. [code] **Block ablation.** A forward hook that returns a value
   *replaces* the module's output.
    1. Use such a hook to zero the output of one residual block's body,
       which turns the block into the identity, and measure the relative
       change in `net`'s output on a fixed batch. Repeat for every block,
       and rank the blocks by this change.
    1. Rebuild `net` three more times with new random weights and repeat
       the ranking. State whether the conclusion "block $k$ matters most"
       survives, and what evidence such a claim would need.
2. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$, with each variance taken over all entries. On one
   batch, record for every `ResidualBlock` the variance of $\mathbf{x}_k$
   with `register_forward_pre_hook` (the pre-hook receives the argument
   tuple, so unpack it), of $\mathbf{h}_k$ with a forward hook on the
   block, and of $\mathbf{b}_k$ with a forward hook on its `body`. The
   standard deviations in the table of :numref:`subsec_activation_stats`
   grow because each block adds its body's output to the stream. Measure
   how closely

   $$\mathrm{Var}[\mathbf{h}_k] = \mathrm{Var}[\mathbf{x}_k] + \mathrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
3. [code] **Seeds and data loading.** ● Extend `train_once` to load its
   data through a `DataLoader` with `shuffle=True` and `num_workers=4`,
   keeping the single `torch.manual_seed(seed)` call at its top. The
   dataset adds noise to each example inside `__getitem__`, drawn from an
   `np.random.default_rng()` object created in the dataset's constructor.
   Run the code as a standalone script, because worker processes started
   with `spawn`, the default on macOS and Windows, cannot import a dataset
   class defined in a notebook.
    1. Predict whether two runs with the same seed agree on the shuffle
       order and on the noise, and whether the four workers draw the same
       noise. Justify each prediction from where that randomness is drawn,
       then verify all three.
    1. Extend `seed_worker` to replace the dataset's generator with one
       derived from the worker's own seed, and verify that two runs now
       agree and that the workers draw different noise.

    *Adapted from the PyTorch documentation
    [Multi-process data loading](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading).*
4. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
5. [code] **Gradient norms.** ● Rebuild `net`, multiply the weight of
   every linear layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Using `register_full_backward_hook`, record the norm of each block's
       output gradient. After the backward pass, compute each block's
       parameter-gradient norm, and plot both families against depth.
    1. Express the gradient of a linear layer's weight through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
6. [code] **Hook coverage.** Attach a forward hook to every leaf module of
   `torchvision.models.resnet18()` through `named_modules()`, and count
   how often each hook fires in one forward pass. Explain any module that
   fires more than once, and name a computation in `BasicBlock.forward`
   that no module hook observes.
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Block ablation.** Keras has no hook mechanism that could
   replace a layer's output.
    1. Zero the output of one residual block's body by editing the `call`
       of `ResidualBlock`, which turns the block into the identity, and
       measure the relative change in the output of a freshly built
       `Checked` model on a fixed batch. Repeat for every
       block, and rank the blocks by this change.
    1. Rebuild the model three more times with new random weights and
       repeat the ranking. State whether the conclusion "block $k$ matters
       most" survives, and what evidence such a claim would need.
2. [code] **Residual variance budget.** Override `call` in
   `ResidualBlock` so that it stashes the
   variance of its input, of its body's output, and of its own output, and
   run a freshly built `Checked` model on one batch. Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote these three tensors for block
   $k$, with each variance taken over all entries. The standard deviations
   in the table of :numref:`subsec_activation_stats` grow because each
   block adds its body's output to the stream. Measure how closely

   $$\mathrm{Var}[\mathbf{h}_k] = \mathrm{Var}[\mathbf{x}_k] + \mathrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
3. [code] **Seeds and data loading.** ● Work in a fresh process, since the
   section has already called `enable_op_determinism`, which cannot be
   undone. Build a `tf.data` pipeline over `Dataset.range(64)` that
   shuffles with `shuffle(64, seed=0)` and adds noise to each element in a
   `map` with `num_parallel_calls=4`, and start every run with
   `tf.keras.utils.set_random_seed(0)`.
    1. Draw the noise from NumPy's global generator through
       `tf.numpy_function`. Predict whether two runs attach the same noise
       to the same element under the default `deterministic=True`, and
       justify the prediction from where the randomness is drawn. Verify
       over five runs.
    1. Repeat the five runs twice more: once with
       `tf.config.experimental.enable_op_determinism()` called at program
       start, and once without it but with the noise drawn by
       `tf.random.stateless_normal`, seeded from the element's value.
       Explain each outcome.
4. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   the `Checked` model and train it with SGD at learning rate 1 on the
   mean squared error between its output and standard normal targets,
   using one fixed batch of 256 standard normal inputs. Report the first
   step at which a layer's output is non-finite, and the first such layer
   by depth. Check the loss at every step as well, and report whether it
   becomes non-finite before any layer output does.
5. [code] **Gradient norms.** ● Rebuild `net`, multiply the kernel of
   every `Dense` layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Under a `tf.GradientTape`, record each block's output tensor and
       request the gradient of the loss with respect to those outputs
       alongside the trainable variables. Compute both families of norms
       per block and plot them against depth.
    1. Express the gradient of a linear layer's kernel through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

:begin_tab:`jax`
1. [code] **Block ablation.** `sow` records intermediates but cannot change
   them, so an output-replacing hook cannot be written.
    1. Zero the output of one residual block's body by editing
       `ResidualBlock.__call__`, which turns the block into the identity,
       and measure the relative change in `net`'s output on a fixed batch.
       Repeat for every block, and rank the blocks by this change.
    1. Rebuild `net` three more times with new random weights and repeat
       the ranking. State whether the conclusion "block $k$ matters most"
       survives, and what evidence such a claim would need.
2. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$, with each variance taken over all entries. Collect
   every layer's output and every body's output from one call of
   `nnx.capture` with `method_outputs=nnx.Intermediate`. The input of
   block $k$ is the output of the layer before it. The standard deviations
   in the table of :numref:`subsec_activation_stats` grow because each
   block adds its body's output to the stream. Measure how closely

   $$\mathrm{Var}[\mathbf{h}_k] = \mathrm{Var}[\mathbf{x}_k] + \mathrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
3. [code] **Seeds and data loading.** ● Simulate a four-worker input
   pipeline in plain Python: split a dataset of 64 examples with 20
   features into four shards, one per worker, and let each worker add
   Gaussian noise to its examples with `jax.random.normal`.
    1. Derive the noise keys by splitting one base key into per-worker keys
       with `jax.random.split(key, 4)` and folding the epoch number into
       each with `jax.random.fold_in`. Verify that two runs from the same
       base key produce identical noise. Then reduce the number of workers
       to two and check whether each example still receives the same noise.
    1. Derive each example's key instead by folding the epoch number into
       the base key and then folding in the example's index. Repeat both
       checks, and explain which derivation makes an example's noise
       independent of how examples are assigned to workers.
4. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
5. [code] **Gradient norms.** ● Rebuild `net`, multiply the kernel of
   every linear layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Compute per-block parameter-gradient norms from the tree that
       `nnx.grad` returns. To obtain each block's *output* gradient, split
       the forward computation at that block and differentiate the tail
       with respect to the block's output. Plot both families of norms
       against depth.
    1. Express the gradient of a linear layer's kernel through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
6. [code] **Capture coverage.** Find the value computed inside each
   block's `body` that a call of `nnx.capture` with
   `method_outputs=nnx.Intermediate` does not record, and explain why.
   Edit `ResidualBlock` so that it records the value with `self.sow`, and
   collect it with `nnx.capture(net, nnx.Intermediate)`.
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.8]{.kicker}

Reproducibility and Inspection<br>**seeds · generators · deterministic arithmetic · hooks**
:::
:::

::: {.slide title="A 0.3-Point Gain Means Nothing If Two Runs Differ by as Much"}
[The problem]{.kicker}

- A one-line change raises Fashion-MNIST validation accuracy by 0.3 points.
- Two runs of the **unchanged** script already differ by a few tenths: new
  initial weights, a new order of examples.
- A loss that turns into NaN at step 40,000: which layer produced the first
  non-finite number?

::: {.d2l-note}
We need runs that repeat exactly, and a way to look inside a run without
editing the model.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- list where randomness enters a run, and fix it with seeds;
- keep unrelated draws from shifting yours with generator objects or keys;
- explain why seeded GPU runs can still differ, and turn that off;
- measure every layer of an unmodified model with hooks;
- find the first layer that produces a NaN in one forward pass.

**Running example:** a stack of eight residual blocks of width 64, Xavier
initialization, a minibatch of 256 standard normal inputs.
:::

::: {.slide title="Reproducibility Closes the Chapter on How Models Run"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Modules and Model Construction | a model is a tree of modules, built from a config |
| Parameters, State, and Memory | parameters, buffers, bytes per parameter |
| Initialization | library defaults, custom initializers, residual depth |
| Custom Layers and Functions | new layers, state, and custom gradients |
| Number Formats and Mixed Precision | dtypes, mixed precision, loss scaling |
| Saving, Loading, and Pretrained Weights | safe files, checkpoints, pretrained weights |
| GPUs, Devices, and Memory | placement, asynchrony, training memory |
| **Reproducibility and Inspection** | **seeds, determinism, hooks** |
:::

::: {.col .narrow}
- **Uses:** `__call__` wrapping `forward`, residual blocks, Xavier
  initialization, rounding in float32.
- **Skips:** profilers; agreement across machines.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Seeds and Generators]{.dtitle}

[fixing every random draw]{.dsub}
:::
:::

::: {.slide title="Randomness Enters a Run in Five Places"}
[Seeds · inventory]{.kicker}

| source | draws |
|---|---|
| initialization | every weight |
| dropout | a new mask at every step |
| data loader | a new order of examples in every epoch |
| augmentation | crops, flips |
| loader workers | any of the above, in other processes |

::: {.d2l-note}
Each draw comes from a *pseudorandom number generator*: a deterministic
algorithm whose outputs are fixed by its starting state, the **seed**.
:::
:::

::: {.slide title="One Seed Repeats a Run Bit for Bit"}
[Seeds · global seed]{.kicker}

Five SGD steps on a small MLP, run twice with seed 0 and once with seed 1:

@!reproducibility-inspection-seeds-and-randomness

- Same initial weights, same data, same operations in the same order.
- Bitwise: every bit of the loss and of every weight agrees.
:::

::: {.slide title="torch.manual_seed Covers PyTorch, Not NumPy or random" only="pytorch"}
[Seeds · coverage]{.kicker}

- `torch.manual_seed(k)`: PyTorch's generator on the CPU and every GPU.
- Not covered: Python's `random`, NumPy's global generator, and
  `np.random.default_rng()` objects, which seed themselves.
- `DataLoader` workers get their own seeds from a base seed; generator
  objects inside a dataset need a `worker_init_fn`.
:::

::: {.slide title="set_random_seed Covers Python, NumPy, and TensorFlow" only="tensorflow"}
[Seeds · coverage]{.kicker}

- `tf.keras.utils.set_random_seed(k)`: Python's `random`, NumPy's global
  generator, and TensorFlow's global generator.
- Not covered: `tf.random.Generator` objects and `np.random.default_rng()`
  objects, which carry their own state.
- `tf.data` takes its seed as an argument: `shuffle(buffer, seed=...)`.
:::

::: {.slide title="mnp.random.seed Covers MXNet on Every Device, Not NumPy" only="mxnet"}
[Seeds · coverage]{.kicker}

- `mnp.random.seed(k)`: MXNet's generator on every device.
- Not covered: Python's `random` and NumPy's global generator, from which
  Gluon's `DataLoader` draws its shuffle order.
- No generator objects: seed again where draws must stay fixed.
:::

::: {.slide title="JAX Has No Global Generator: Keys Are Values" only="jax"}
[Seeds · keys]{.kicker}

- `jax.random.normal(key, shape)`: the same key gives the same numbers.
- New randomness only from `jax.random.split(key, n)` or
  `jax.random.fold_in(key, i)`.
- A seed becomes one key; `train_once` splits it into keys for
  initialization, inputs, and targets.
:::

::: {.slide title="A Private Generator Ignores Unrelated Draws"}
[Seeds · generator objects]{.kicker}

One global stream couples all code: one extra draw shifts every later
number. A private stream does not move:

@reproducibility-inspection-generator-objects
:::

::: {.slide title="Loader Workers Need Their Own Streams"}
[Seeds · workers]{.kicker}

| what happens | result |
|---|---|
| workers start as copies of the main process | all apply the same "random" augmentations |
| workers seed themselves from the OS | no two runs agree |
| each worker's seed derived from one base seed | repeatable and distinct |

::: {.d2l-note}
Derive every worker's stream from one base seed, and refresh it per epoch.
:::
:::

::: {.slide title="Check: Does the Seed Fix This Noise?"}
[Check]{.kicker}

A script seeds the framework's generator at its top. Its dataset adds noise
drawn from `np.random.default_rng()`, created in the dataset's constructor.
Do two runs see the same noise?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. `default_rng()` without an argument seeds itself from the
operating system, and no framework seed reaches it. Pass it a seed derived
from the run's seed, per worker.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Deterministic Arithmetic]{.dtitle}

[the same numbers, added in a different order]{.dsub}
:::
:::

::: {.slide title="Floating-Point Addition Is Not Associative"}
[Arithmetic · rounding]{.kicker}

@reproducibility-inspection-determinism-and-its-price-1

- Near $10^8$, adjacent float32 numbers are 8 apart: $-10^8 + 1$ rounds
  to $-10^8$.
- A million random numbers summed forward and backward usually differ in
  the last bits.
:::

::: {.slide title="On a GPU, Atomic Additions Change the Order from Run to Run"}
[Arithmetic · GPUs]{.kicker}

- *Atomic additions* commit partial sums in the order threads finish.
- Seeded runs on one GPU then differ in the last bits, later in the loss.

| framework | switch to deterministic operations |
|---|---|
| PyTorch | `torch.use_deterministic_algorithms(True)`; raises if none exists |
| TensorFlow | `tf.config.experimental.enable_op_determinism()` |
| JAX | `XLA_FLAGS=--xla_gpu_deterministic_ops=true` |
| MXNet | none |
:::

::: {.slide title="Bitwise Repetition Is for Debugging; Conclusions Must Hold across Seeds"}
[Arithmetic · scope]{.kicker}

- Bitwise agreement holds for one machine, one library version, one set of
  switches; not across hardware or releases.
- Use it to find the first step at which two runs differ.
- A claim like "the change raises accuracy" needs the mean and spread over
  several seeds.

::: {.d2l-note}
A conclusion that changes with the seed, or with the last bits of the
arithmetic, needs more evidence.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Hooks]{.dtitle}

[observing a model from outside its code]{.dsub}
:::
:::

::: {.slide title="A Hook Runs inside the Call, around the Layer's Computation"}
[Hooks · mechanism]{.kicker}

![`__call__` runs pre-hooks, `forward`, then hooks; an observer captures, checks, or modifies.](../img/bg-hooks.svg)

- PyTorch and MXNet: `register_forward_hook`; TensorFlow: extra outputs of
  a functional model; JAX: `nnx.capture`.
- No line of the model changes.
:::

::: {.slide title="The Spread Grows by a Factor of About 1.22 per Residual Block"}
[Hooks · activation statistics]{.kicker}

::: {.cols .vc}
::: {.col}
@!reproducibility-inspection-capturing-activation-statistics
:::

::: {.col}
- Standard deviation of each layer's output, 256 standard normal inputs,
  Xavier initialization.
- About 0.7 after the input layer, roughly 3 to 4 after eight blocks.
:::
:::
:::

::: {.slide title="Variances Add, and the Body Has Half the Variance of Its Input"}
[Hooks · why 1.22]{.kicker}

$\mathbf{h} = \mathbf{x} + \textrm{body}(\mathbf{x})$, with $\mathbf{x}$ and
$\textrm{body}(\mathbf{x})$ nearly uncorrelated:

$$\textrm{Var}[\mathbf{h}] \approx \textrm{Var}[\mathbf{x}] + \textrm{Var}[\textrm{body}(\mathbf{x})] \approx \textrm{Var}[\mathbf{x}] + \tfrac{1}{2}\textrm{Var}[\mathbf{x}].$$

- Xavier keeps the variance through each linear layer; ReLU halves the mean
  square.
- Variance $\times 1.5$ per block, standard deviation $\times\sqrt{1.5}
  \approx 1.22$; eight blocks: $1.5^4 \approx 5$.
:::

::: {.slide title="Check: How Deep Before the Spread Grows a Hundredfold?"}
[Check]{.kicker}

The standard deviation grows by $\sqrt{1.5}$ per block. After how many
blocks is it 100 times its value at the input of the stack?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1.5^{k/2} = 100$ gives $k = 2\log 100/\log 1.5 \approx 22.7$:
about 23 blocks. Deep residual networks scale their residual branches down
for this reason.
:::
:::
:::

::: {.slide title="A Finiteness Check on Every Layer Names the First NaN"}
[Hooks · NaN finder]{.kicker}

One NaN placed in a weight of the third residual block; a check on every
layer raises at the first non-finite output:

@!reproducibility-inspection-a-nan-finder

- Hooks fire in the order the layers compute, so the first to fire names
  the layer.
- One forward pass, no print statements in the model.
:::

::: {.slide title="Gradients Are Values to Inspect"}
[Hooks · gradients]{.kicker}

Norms of the gradients of the first linear layer in the third block, loss
= mean squared output:

@!reproducibility-inspection-backward-hooks-and-beyond

- PyTorch also offers backward hooks, which run during the backward pass.
- JAX and TensorFlow return the gradients as values; MXNet stores them in
  `param.grad()`.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Seeds** fix every draw of a generator; list every generator a run uses,
  workers included.
- **Generator objects and keys** keep unrelated draws from shifting yours.
- **Arithmetic:** $(10^8 - 10^8) + 1 = 1$ but $10^8 + (-10^8 + 1) = 0$ in
  float32; GPU atomics vary the order.
:::

::: {.col}
- **Bitwise** repetition is for debugging; conclusions need several seeds.
- **Hooks** observe any layer from outside: the spread grows by about 1.22
  per block, and one pass finds the first NaN.
:::
:::

**Next:** the chapters on convolutional networks, where these tools debug
models with millions of parameters.
:::
