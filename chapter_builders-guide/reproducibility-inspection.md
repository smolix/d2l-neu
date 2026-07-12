```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Reproducibility and Inspection
:label:`sec_repro`

Run yesterday's experiment again and the loss curve comes out different.
Is this morning's change an improvement, or a lucky seed? Answering that
question requires knowing where every random number in a training run comes
from. And once two runs *do* disagree, or a loss turns into NaN, the next
question is what the model computed layer by layer, ideally without editing
its source. This section covers both skills: controlling randomness (seeds,
generators, determinism) and observing a running model from the outside
(hooks).

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab pytorch
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab jax
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab mxnet
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Seeds and Randomness

Randomness enters a training run in more places than you might list on a
first try: initialization draws every weight (:numref:`sec_init_param`),
dropout samples a fresh mask at each step
:cite:`Srivastava.Hinton.Krizhevsky.ea.2014`, the data loader shuffles
examples differently in every epoch, augmentations sample crops and flips,
and the loader's worker processes each do all of this in parallel. Every one
of these consults a pseudorandom number generator, a deterministic algorithm
whose entire output sequence is fixed by its seed. Seed every generator and
the run is repeatable; miss one and it is not.

:begin_tab:`pytorch`
In PyTorch, `torch.manual_seed(k)` seeds the global generator on the CPU
*and* on every CUDA device. It does not touch Python's `random` module or
NumPy, which keep their own global generators (`random.seed`,
`np.random.seed`), nor NumPy `Generator` objects created by
`np.random.default_rng`, which carry their own seeds. The function below
draws everything, weights, data, and shuffling, from torch's global
generator, so one call pins the whole run:
:end_tab:

:begin_tab:`jax`
JAX has no global generator to seed. Every random operation takes a *key*,
a value that fully determines its output: `jax.random.normal(key, shape)`
returns the same numbers every time it is called with that key, and fresh
randomness comes only from deriving fresh keys with `jax.random.split`. The
function below turns its seed into one key and splits it three ways, one
key each for initialization, inputs, and targets, so the seed pins the
whole run by construction:
:end_tab:

:begin_tab:`tensorflow`
TensorFlow collapses the inventory to one call:
`tf.keras.utils.set_random_seed(k)` seeds Python's `random` module, NumPy's
global generator, and TensorFlow's global generator together, the three a
typical training script consults. It does not reach generators that carry
their own state (`tf.random.Generator` objects, NumPy `Generator` objects
from `np.random.default_rng`), each of which has its own seed. The function
below draws everything, weights and data alike, from the seeded global
state, so one call pins the whole run:
:end_tab:

:begin_tab:`mxnet`
MXNet keeps one global generator per device. `np.random.seed(k)` (MXNet's
`np`, an alias of `mx.random.seed`) seeds all of them in one call, each
from the seed and its device id, so initialization, sampling, and dropout
are pinned on every device at once (pass `device=` instead to bring a
single generator to a device-independent state). It does not touch
Python's `random` module or classic NumPy, which keep their own global
generators, a gap that will matter when the data pipeline enters. The
function below draws everything, weights and data alike, from MXNet's
seeded generators, so one call pins the whole run:
:end_tab:

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab pytorch
def train_once(seed):
    torch.manual_seed(seed)  # seeds the CPU and every CUDA device
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
%%tab jax
def train_once(seed):
    key_init, key_X, key_y = jax.random.split(jax.random.key(seed), 3)
    net = nnx.Sequential(nnx.Linear(20, 32, rngs=nnx.Rngs(key_init)),
                         nnx.relu,
                         nnx.Linear(32, 1, rngs=nnx.Rngs(key_init)))
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
      loss_a == loss_b, jnp.array_equal(w_a, w_b))
print('different seed:', loss_a, 'vs', loss_c)
```

```{.python .input #reproducibility-inspection-seeds-and-randomness}
%%tab tensorflow
def train_once(seed):
    tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy, and TF
    net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'),
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
%%tab mxnet
def train_once(seed):
    np.random.seed(seed)  # seeds the generator on every device
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'), nn.Dense(1))
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.1})
    X = np.random.normal(0, 1, (128, 20))
    y = np.random.normal(0, 1, (128, 1))
    for _ in range(5):
        with autograd.record():
            loss = ((net(X) - y) ** 2).mean()
        loss.backward()
        trainer.step(1)  # loss is a mean already: no batch-size rescale
    return loss.item(), net[0].weight.data().copy()

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      loss_a == loss_b, bool((w_a == w_b).all()))
print('different seed:', loss_a, 'vs', loss_c)
```

The two seeded runs agree *bitwise*, down to the last floating-point bit:
same initial weights, same data, same gradients, same operations in the
same order.

### Generator Objects

:begin_tab:`pytorch`
A single global stream is fragile: every consumer shares it, so inserting
one extra random call (a new layer's initialization, a stray `torch.randn`
while debugging) shifts everything drawn after it. A `torch.Generator` is a
private stream with its own seed. Here a data split stays fixed no matter
what else consumes randomness in between:
:end_tab:

:begin_tab:`jax`
A single shared stream is fragile: every consumer advances it, so
inserting one extra random call shifts everything drawn after it. In JAX
the keys *are* the generator objects: each key is a private stream you
hold as a value, and `jax.random.split` manufactures as many independent
streams as you need. The property a private stream is supposed to
protect, that unrelated draws elsewhere cannot shift yours, holds by
construction, because no draw advances any shared state. Here an
unrelated draw between two uses of the same key changes nothing:
:end_tab:

:begin_tab:`tensorflow`
A single global stream is fragile: every consumer shares it, so inserting
one extra random call shifts everything drawn after it. A
`tf.random.Generator` is a private stream with its own seed. Here a data
split stays fixed no matter what else consumes randomness in between (the
generator has no permutation method, so we sort random uniforms, the
standard trick):
:end_tab:

:begin_tab:`mxnet`
A single global stream is fragile: every consumer shares it, so inserting
one extra random call shifts everything drawn after it. MXNet offers no
private stream to retreat to: there is no generator object, and no
sampling function takes one. The substitute is seeding discipline: re-seed
immediately before the draws that must be pinned. It works, at a price the
private stream does not charge, since re-seeding also resets the stream
for everything drawn after it:
:end_tab:

```{.python .input #reproducibility-inspection-generator-objects}
%%tab pytorch
g = torch.Generator().manual_seed(42)
split = torch.randperm(10, generator=g)
_ = torch.randn(1000)  # unrelated consumption of the global stream
split_again = torch.randperm(10, generator=g.manual_seed(42))
print(torch.equal(split, split_again), split)
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab jax
key_split, key_other = jax.random.split(jax.random.key(42))
split = jax.random.permutation(key_split, 10)
_ = jax.random.normal(key_other, (1000,))  # unrelated draw, its own key
split_again = jax.random.permutation(key_split, 10)
print(jnp.array_equal(split, split_again), split)
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab tensorflow
g = tf.random.Generator.from_seed(42)
split = tf.argsort(g.uniform((10,)))  # a random permutation from g's stream
_ = tf.random.normal((1000,))  # unrelated consumption of the global stream
split_again = tf.argsort(tf.random.Generator.from_seed(42).uniform((10,)))
print(bool(tf.reduce_all(split == split_again)), split.numpy())
```

```{.python .input #reproducibility-inspection-generator-objects}
%%tab mxnet
np.random.seed(42)
split = np.arange(10)
np.random.shuffle(split)  # mx.np has no permutation; shuffle in place
_ = np.random.normal(0, 1, (1000,))  # unrelated consumption of the stream
np.random.seed(42)  # re-seed at the boundary that must be pinned
split_again = np.arange(10)
np.random.shuffle(split_again)
print(bool((split == split_again).all()), split)
```

:begin_tab:`pytorch`
Most sampling functions accept `generator=`, and so does `DataLoader`,
where it controls the shuffle order. We use that below.
:end_tab:

:begin_tab:`jax`
Every function in `jax.random` takes the key as its first argument; there
is no variant that consults hidden state.
:end_tab:

:begin_tab:`tensorflow`
Sampling methods live on the generator itself (`g.normal`, `g.uniform`), so
a consumer that should own its randomness takes a `Generator` argument. The
input pipeline instead takes a seed directly, which we use next.
:end_tab:

:begin_tab:`mxnet`
There is no `generator=` argument anywhere in Gluon: a consumer that must
own its randomness owns a seed instead and calls `np.random.seed` itself,
accepting the reset of everything drawn after it.
:end_tab:

### DataLoader Workers

:begin_tab:`jax`
The classic reproducibility hole in the loader-worker world is that
parallel workers inherit or reseed a hidden generator, so their
augmentations either coincide or vary from run to run without anyone
deciding which. With explicit keys, that bug cannot be written down: a
worker's randomness is whatever key you hand it, so you split one key
into per-worker keys (`jax.random.split(key, num_workers)`), refresh them
each epoch by folding in the epoch number (`jax.random.fold_in`), and two
workers can share a stream only if the code visibly passes the same key
twice. One honest caveat: JAX programs usually borrow their input
pipeline from NumPy, PyTorch, or `tf.data`, and those loaders bring their
hidden per-process generator state along. When you do that, the fix is
the host framework's, not JAX's: give each worker process its own seed,
derived from one base seed and refreshed every epoch.
:end_tab:

:begin_tab:`pytorch`
Now for the classic hole: you seed torch, NumPy, and `random`, and the run
is *still* not reproducible, because augmentation code in
`Dataset.__getitem__` calls `np.random` inside loader worker processes. On
Linux, worker processes start via `fork`, which gives each child a
byte-for-byte copy of the parent, including NumPy's global generator state.
We can simulate what every worker inherits:
:end_tab:

```{.python .input #reproducibility-inspection-dataloader-workers-1}
%%tab pytorch
np.random.seed(0)                   # the parent process, dutifully seeded
state = np.random.get_state()       # fork copies this state into each child
for worker_id in range(4):
    np.random.set_state(state)      # what a fork-started worker begins with
    print(f'worker {worker_id}:', np.random.randint(0, 1000, size=3))
```

:begin_tab:`pytorch`
All four workers produce the same "random" numbers, so they apply identical
crops and noise; and because workers are re-forked from the same parent
state each epoch, the same augmentations repeat every epoch. The pattern is
not rare: a 2020 audit of over a hundred thousand public repositories using
PyTorch together with NumPy found this bug in more than 95% of those that
augment inside a custom dataset. On macOS and Windows the default start
method is `spawn`, which imports a fresh interpreter per worker; there NumPy
seeds itself from operating-system entropy, so workers differ from each
other but also differ across runs, which is the opposite failure with the
same root cause: nobody seeded the workers deliberately.

The fix is deliberate per-worker seeding. PyTorch already hands each worker
a distinct torch seed (a base seed plus the worker id, refreshed every
epoch), but it cannot reach NumPy's or `random`'s generators, so we bridge
them ourselves in a `worker_init_fn`, and pin the shuffle order with a
`generator`. This is the recipe from PyTorch's reproducibility notes:
:end_tab:

```{.python .input #reproducibility-inspection-dataloader-workers-2}
%%tab pytorch
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # this worker's torch seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator().manual_seed(0)
data = TensorDataset(torch.arange(8))
# num_workers=0 keeps this cell runnable under both fork and spawn;
# real training uses num_workers > 0 with exactly these arguments
loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0,
                    worker_init_fn=seed_worker, generator=g)
print([batch[0].tolist() for batch in loader])
g.manual_seed(0)
print([batch[0].tolist() for batch in loader])
```

:begin_tab:`pytorch`
Resetting the generator reproduces the exact shuffle order, and with
workers enabled, `seed_worker` gives each worker its own NumPy and `random`
streams that still derive from the one base seed.
:end_tab:

:begin_tab:`tensorflow`
The classic reproducibility hole in the loader-worker world is that
parallel workers inherit or reseed a hidden generator: on fork-based
loaders every child starts with a byte-for-byte copy of the parent's NumPy
state, so all workers apply the same "random" augmentations, or each child
seeds itself from entropy and no two runs agree. `tf.data` sidesteps the
bug class by construction: the pipeline parallelizes with threads inside
one process, so there is no forked child to inherit a stale copy, and its
randomness is an explicit argument, `Dataset.shuffle(buffer, seed=...)`,
not an ambient global that a seeding call may or may not reach. What
remains is choosing what the seed means across epochs. With the default
`reshuffle_each_iteration=True`, a seeded pipeline produces a *different*
order on each pass but the same *sequence* of orders every time the
pipeline is rebuilt, fresh shuffles per epoch, repeatable across runs:
:end_tab:

```{.python .input #reproducibility-inspection-dataloader-workers-3}
%%tab tensorflow
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # epoch 1
print([b.numpy().tolist() for b in ds])  # epoch 2: reshuffled, still seeded
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # rebuilt pipeline: epoch 1 again
```

:begin_tab:`tensorflow`
Both alternatives are explicit choices rather than accidents:
`reshuffle_each_iteration=False` freezes one order for every epoch, and
leaving `seed` unset draws fresh orders each run. One knob remains that
trades reproducibility away on purpose: parallel `map` and `interleave`
accept `deterministic=False`, which hands elements on in completion order
for speed; the default `True` keeps the pipeline's output order fixed.
:end_tab:

:begin_tab:`mxnet`
Gluon's `DataLoader` runs `num_workers > 0` on a pool of worker
*processes*, so the fork-inheritance lesson applies verbatim: on Linux
each worker starts as a copy of the parent, NumPy state included, and the
pool's initializer does no seeding at all; it only ships the dataset and
the `set_np` flags to the child (see `_worker_initializer` in
`gluon/data/dataloader.py`). There is no `worker_init_fn` equivalent to
bridge the gap, and the pool outlives any one epoch. Two consequences.
First, deliberate per-worker seeding has to live in the dataset itself:
make augmentation a deterministic function of the sample by seeding from
the sample index inside `__getitem__`, and it stops mattering which
worker runs it. Second, a surprise from the source: the shuffle order of
`DataLoader(shuffle=True)` is drawn from *classic* NumPy's global
generator (`RandomSampler` calls `numpy.random.shuffle`), so pinning it
takes a NumPy seed; MXNet's own seed does not reach it.
:end_tab:

### Randomness as a Value

Every failure above is hidden global state: a generator that lives
somewhere your code does not mention, silently copied or shared. Some
library designs remove the hiding place altogether. In an explicit-PRNG
design (JAX is the prominent example), randomness is threaded through the
program as a value: every random operation takes a key argument, using the
same key twice yields the same numbers by definition, and independent
streams are obtained by explicitly splitting a key in two. Accidentally
duplicating a stream would require writing the duplication into the code,
so the worker bug cannot be expressed. The price is bookkeeping, since
every function that needs randomness must accept and split keys. PyTorch's
global seed is the convenient version of the same contract, one implicit
key that everything shares; generator objects recover the explicit version
where it matters.

:begin_tab:`jax`
The two cells above are this design in action: `train_once` turned one
seed into one key and split it, and reusing a key reproduced the
permutation bit for bit.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow ships both designs side by side: the stateful API above (a
global generator plus `tf.random.Generator` objects) and a stateless
family in which the key is an argument:
`tf.random.stateless_normal(shape, seed=[k1, k2])` returns the same
numbers for the same seed pair by definition. The explicit `seed=`
arguments of `tf.data` are the same idea applied to the input pipeline.
:end_tab:

:begin_tab:`mxnet`
MXNet sits at the fully implicit end of this spectrum: per-device global
streams, no generator objects, no stateless variants. Seeding discipline,
one call up front and a re-seed at any boundary that must be pinned, is
the entire toolkit.
:end_tab:

## Determinism and Its Price

Seeding fixes which numbers the program draws. It does not fix how the
arithmetic evaluates. Floating-point addition is not associative
(:numref:`sec_numerics`), so summing the same numbers in a different
order gives a different answer:

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab pytorch
torch.manual_seed(0)
x = torch.randn(1_000_000)
s_fwd, s_rev = x.sum(), x.flip(0).sum()  # same numbers, different order
print(s_fwd == s_rev, (s_fwd - s_rev).item())
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab jax
x = jax.random.normal(jax.random.key(0), (1_000_000,))
s_fwd, s_rev = x.sum(), x[::-1].sum()  # same numbers, different order
print(s_fwd == s_rev, (s_fwd - s_rev).item())
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab tensorflow
x = tf.random.Generator.from_seed(0).normal((1_000_000,))
s_fwd = tf.reduce_sum(x)
s_rev = tf.reduce_sum(x[::-1])  # same numbers, different order
print(bool(s_fwd == s_rev), float(s_fwd - s_rev))
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-1}
%%tab mxnet
np.random.seed(0)
x = np.random.normal(0, 1, (1_000_000,))
s_fwd, s_rev = x.sum(), np.flip(x, 0).sum()  # same numbers, different order
print(bool(s_fwd == s_rev), (s_fwd - s_rev).item())
```

:begin_tab:`pytorch`
On a CPU the summation order is at least fixed, so seeded runs repeat. On a
GPU it often is not: kernels built on atomic additions (`scatter_add`, some
convolution and embedding backward passes) commit their partial sums in
whatever order threads happen to finish, so two seeded runs on the *same
machine* can differ in the last bits, and after enough training steps, in
the loss curve. `torch.use_deterministic_algorithms(True)` is the switch
that forbids this: operations with a deterministic implementation use it
(often at some speed cost), and operations without one raise a
`RuntimeError` rather than silently varying. On CUDA you must additionally
set the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`, or cuBLAS
matrix multiplications themselves raise. A related but narrower setting is
`torch.backends.cudnn.benchmark`: when true, cuDNN times several
convolution algorithms on the first batch and keeps the winner, and since
the winner can change from run to run, reproducibility work sets it to
`False` (its sibling `cudnn.deterministic` covers convolutions only and is
subsumed by the global switch).
:end_tab:

:begin_tab:`jax`
For the *drawn numbers*, JAX's answer is structural. Its PRNG is
counter-based (Threefry): a draw is a pure function of the key and the
requested shape, not of any evolving state, so the same key produces the
same numbers on CPU, GPU, and TPU alike (the documented caveat is that
JAX does not promise identical bits *across its own releases*). For the
*arithmetic*, the story splits by backend. On CPU, XLA's kernels are
deterministic, and the bitwise agreement of `train_once` above is exactly
that, verified. On GPU, some XLA kernels commit partial sums via atomic
additions in whatever order threads finish, the same last-bits
nondeterminism as elsewhere; setting the environment variable
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` before JAX initializes
forces deterministic implementations, at a speed cost we have not
measured here. There is no error-raising mode: the flag substitutes
deterministic kernels rather than refusing nondeterministic ones.
:end_tab:

:begin_tab:`tensorflow`
On a CPU the summation order inside one operation is at least fixed, so
seeded runs repeat; the bitwise agreement of `train_once` above is exactly
that. On a GPU it often is not: kernels built on atomic additions (segment
sums, the scatter-add behind `tf.gather`'s gradient) commit their partial
sums in whatever order threads happen to finish, so two seeded runs on the
*same machine* can differ in the last bits, and after enough training
steps, in the loss curve. `tf.config.experimental.enable_op_determinism()`
is the switch that forbids this: operations with a deterministic
implementation use it (often at some speed cost), operations without one
raise `tf.errors.UnimplementedError` rather than silently varying, and
stateful random operations refuse to run without a seed, since an
operation that seeds itself from entropy is nondeterminism by another
name. Two properties follow from its design. It is meant to be called at
program start, before any operation runs, because it configures kernels as
they are created and nothing already executed is redone. And there is no
call that turns it off short of a fresh process. We can still demonstrate
the seed rule mid-flight by clearing the global seed:
:end_tab:

:begin_tab:`mxnet`
On a CPU the summation order inside one operation is at least fixed, so
seeded runs repeat; the bitwise agreement of `train_once` above is exactly
that. On a GPU it often is not: kernels built on atomic additions commit
their partial sums in whatever order threads happen to finish, so two
seeded runs on the *same machine* can differ in the last bits, and after
enough training steps, in the loss curve. MXNet ships no switch that
forbids this and no error-raising mode. The one lever it does have is the
analogue of cuDNN benchmark mode: by default cuDNN times several
convolution algorithms on the first batch and keeps the winner, and since
the winner can change from run to run, reproducibility work sets the
environment variable `MXNET_CUDNN_AUTOTUNE_DEFAULT=0` (before any
operator runs) to pin the algorithm choice. Beyond that, staying away
from operations whose kernels are nondeterministic is up to you.
:end_tab:

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab pytorch
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    try:
        torch.randn(10, device='cuda').kthvalue(1)
    except RuntimeError as e:
        print(str(e).split('.')[0])
else:
    print('CPU run: every kernel used above is already deterministic;',
          'on CUDA, ops lacking a deterministic kernel raise RuntimeError')
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
tf.random.set_seed(0)  # determinism stays on; reseed and continue
```

Be honest about what this buys. No framework guarantees bitwise agreement
across releases, platforms, or CPU versus GPU: it holds only for a
pinned machine, library version, and flag configuration. That makes bitwise
reproducibility a *debugging* tool, the setting that lets you bisect
exactly where two runs diverge. The *scientific* goal is statistical
reproducibility: the same conclusions across seeds, reported as a mean and
spread over several runs rather than one fortunate curve. The distinction
mirrors :numref:`sec_numerics`: changing dtype changes results in the
last bits by design, and an experimental claim that survives neither a new
seed nor bfloat16 was never a result.

## Hooks: Looking Inside

:begin_tab:`pytorch`
In :numref:`sec_model_construction` we noted that `net(X)` does not call
`forward` directly: it calls `__call__`, which wraps `forward` with extra
machinery. Hooks are that machinery, exposed. Calling
`module.register_forward_hook(f)` arranges for `f(module, inputs, output)`
to run after every forward pass of that module, with no change to the
model's source, which matters precisely when the model came from a library
or a checkpoint you do not want to edit.
:numref:`fig_bg_hooks` draws the wrapper as a pipeline: hooks slot into the
gap the `__call__` wrapper already leaves around `forward`, so an observer
can capture, check, or modify without a single line of the model changing.
:end_tab:

:begin_tab:`jax`
NNX can wrap a module call with `nnx.capture`, recording the return value of
each submodule method alongside the model output without changing the model's
source. That matters
precisely when the model came from a library or a checkpoint you do not
want to edit. :numref:`fig_bg_hooks` draws the general picture: an
observation point in the gap the call wrapper already leaves around each
module's computation, from which an observer can capture or check
without a single line of the model changing.
:end_tab:

:begin_tab:`tensorflow`
Keras wraps `call` with a `__call__` too (it handles building, dtype
casting, and masks), but the wrapper exposes no observation point: nothing
can be attached to an unmodified model after the fact, and observing a
black-box model from a library or a checkpoint without touching its code
has no TensorFlow equivalent. Two idioms reach the same measurements with
a little structure. In a *functional* model every
intermediate tensor is a first-class object, so a second `tf.keras.Model`
over the same graph can declare any internal tensor an output: surgery
rather than hooking, sharing all weights and adding no computation. And
where you own the model's code, an overridden `call` can stash or check
whatever it likes as it runs. :numref:`fig_bg_hooks` still draws the right
picture, with one amendment: in TensorFlow the observer cannot stand in
the gap unless the model was built to leave one.
:end_tab:

:begin_tab:`mxnet`
In :numref:`sec_model_construction` we noted that `net(X)` does not
call `forward` directly: it calls `__call__`, which wraps `forward` with
extra machinery. Hooks are that machinery, exposed. Calling
`block.register_forward_hook(f)` arranges for `f(block, inputs, output)`
to run after every forward pass of that block, and
`register_forward_pre_hook` attaches before it, with no change to the
model's source, which matters precisely when the model came from a
library or a checkpoint you do not want to edit. Gluon itself relies on
the mechanism: `Block.summary()` builds its per-layer table by
registering a forward hook on every block. :numref:`fig_bg_hooks` draws
the wrapper as a pipeline: hooks slot into the gap the `__call__` wrapper
already leaves around `forward`, so an observer can capture or check
without a single line of the model changing (Gluon's contract is
observe-only: a hook's return value is ignored, so unlike PyTorch's it
cannot modify the output).
:end_tab:

![The `__call__` wrapper as a pipeline: input flows through pre-hooks, then forward, then hooks, to the output, with the two hook stages dashed and orange against forward's solid blue, and a side arrow from the hooks stage to an observer that can capture, check, or modify.](../img/bg-hooks.svg)
:label:`fig_bg_hooks`

We reuse the residual stack of
:numref:`sec_model_construction`, rebuilt compactly:

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(num_hiddens, num_hiddens),
                                  nn.ReLU(),
                                  nn.Linear(num_hiddens, num_hiddens))

    def forward(self, X):
        return X + self.body(X)

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(20, 64),
                    *[ResidualBlock(64) for _ in range(8)],
                    nn.Linear(64, 10))
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab jax
class ResidualBlock(nnx.Module):
    def __init__(self, num_hiddens, rngs):
        self.body = nnx.Sequential(
            nnx.Linear(num_hiddens, num_hiddens, rngs=rngs), nnx.relu,
            nnx.Linear(num_hiddens, num_hiddens, rngs=rngs))

    def __call__(self, X):
        return X + self.body(X)

rngs = nnx.Rngs(0)
net = nnx.Sequential(nnx.Linear(20, 64, rngs=rngs),
                     *[ResidualBlock(64, rngs) for _ in range(8)],
                     nnx.Linear(64, 10, rngs=rngs))
X = jax.random.normal(jax.random.key(1), (256, 20))
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab tensorflow
def residual_block(X, num_hiddens):
    body = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hiddens, activation='relu'),
        tf.keras.layers.Dense(num_hiddens)])
    return X + body(X)

tf.keras.utils.set_random_seed(0)
inputs = tf.keras.Input(shape=(20,))
taps = [tf.keras.layers.Dense(64)(inputs)]  # keep every intermediate tensor
for _ in range(8):
    taps.append(residual_block(taps[-1], 64))
outputs = tf.keras.layers.Dense(10)(taps[-1])
net = tf.keras.Model(inputs, outputs)
```

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab mxnet
class ResidualBlock(nn.Block):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(nn.Dense(num_hiddens, activation='relu'),
                      nn.Dense(num_hiddens))

    def forward(self, X):
        return X + self.body(X)

np.random.seed(0)
net = nn.Sequential()
net.add(nn.Dense(64), *[ResidualBlock(64) for _ in range(8)], nn.Dense(10))
net.initialize(init.Xavier())  # variance-preserving, as in the init chapter
```

### Capturing Activation Statistics

The initialization experiments of :numref:`sec_init_param` measured the
standard deviation of activations at every depth. The same measurement
takes a few lines on an unmodified model:

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
%%tab jax
_, inter = nnx.capture(
    net, nnx.Intermediate, method_outputs=nnx.Intermediate)(X)
for k, layer in enumerate(net.layers):
    out = inter['layers'][k]['__call__'][0]
    print(f'{type(layer).__name__:15s} std {out.std():.2f}')
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
%%tab mxnet
stats, handles = [], []

def record(block, inputs, output):
    stats.append((type(block).__name__, output.std().item()))

for m in net:
    handles.append(m.register_forward_hook(record))
net(np.random.normal(0, 1, (256, 20)))
for h in handles:
    h.detach()
for name, std in stats:
    print(f'{name:15s} std {std:.2f}')
```

The residual stream's spread grows block by block, since each block adds
its body's output on top of the stream, exactly the depth effect that
motivated the scaled initializations of :numref:`sec_init_param`, measured
here without touching the model.

:begin_tab:`pytorch`
Two rules keep hooks safe. First, *detach before you stash*: the hook above
stores `output.detach().std()`; storing `output` itself would keep the
autograd graph of every forward pass alive until `stats` is cleared, a
memory leak that grows with each batch. Second, *keep the handle and call*
`handle.remove()`: a hook stays registered for the module's lifetime
otherwise, taxing every later forward pass and accumulating stashed
tensors.
:end_tab:

:begin_tab:`jax`
Nothing here needs detaching or removing: the captured intermediates are
ordinary arrays with no autograd graph attached, the dictionary exists
for this one call, and no observer stays registered on the model. Two
refinements are worth knowing. `capture_intermediates` also accepts a
filter, a function of the module and method name, so you can record only
the layers you care about (the next cell uses one). And a module can opt
in from the inside: calling `self.sow('intermediates', 'name', value)`
anywhere in its own code records exactly the named values instead of
every return.
:end_tab:

:begin_tab:`tensorflow`
Nothing needs detaching and nothing needs removing: `probe` shares the
original model's layers and weights outright, its outputs are ordinary
tensors with no gradient tape attached, and no observer stays registered
anywhere, because the "hook" is just another model output. The limit is
structural. Surgery needs a functional graph; a subclassed model whose
`call` is imperative Python has no symbolic intermediates to tap. For that
case you override `call` itself, which the next problem gives us a reason
to do.
:end_tab:

:begin_tab:`mxnet`
Two rules keep hooks safe. First, *detach before you stash*: the hook
above reduces the output to a Python float on the spot; a hook that
stores arrays should store `output.detach()` (add `.copy()` if the array
may later be updated in place), because under `autograd.record` a stashed
output keeps the computation graph of every forward pass alive until the
list is cleared. Second, *keep the handle and call* `handle.detach()`:
`register_forward_hook` returns a `HookHandle`, and the hook stays
registered for the block's lifetime otherwise (the handle also works as a
context manager that detaches on exit). One caveat specific to Gluon:
hooks live in the Python `__call__` wrapper, so `hybridize()` warns that
hooks on the children of a hybridized block will not fire; keep a block
unhybridized while observing it.
:end_tab:

### A NaN Finder

When a loss becomes NaN at step 40,000, the useful question is which layer
produced the first non-finite value. Check every leaf module's output for
finiteness and the answer arrives in one forward pass. We sabotage a
weight deep in the stack to watch it fire:

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab pytorch
def make_finite_check(name):
    def check(module, inputs, output):
        if not torch.isfinite(output).all():
            raise RuntimeError(f'first non-finite output in {name}')
    return check

handles = [m.register_forward_hook(make_finite_check(name))
           for name, m in net.named_modules()
           if len(list(m.children())) == 0]
with torch.no_grad():
    net[3].body[0].weight[0, 0] = float('nan')  # sabotage one layer
try:
    net(torch.randn(2, 20))
except RuntimeError as e:
    print(e)
for h in handles:
    h.remove()
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab jax
net.layers[3].body.layers[0].kernel[0, 0] = float('nan')
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
        break
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab tensorflow
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.body = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_hiddens)])

    def call(self, X):
        return X + self.body(X)

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

tf.keras.utils.set_random_seed(0)
checked = Checked([tf.keras.layers.Dense(64)]
                  + [ResidualBlock(64, name=f'block{k}') for k in range(1, 9)]
                  + [tf.keras.layers.Dense(10)])
_ = checked(tf.random.normal((2, 20)))  # build the weights
kernel = checked.seq[3].body.layers[0].kernel
kernel[0, 0].assign(float('nan'))  # sabotage one layer
_ = checked(tf.random.normal((2, 20)))
print('first non-finite output in', checked.first_bad)
```

```{.python .input #reproducibility-inspection-a-nan-finder}
%%tab mxnet
def make_finite_check(name):
    def check(block, inputs, output):
        if not np.isfinite(output).all():
            raise RuntimeError(f'first non-finite output in {name}')
    return check

leaves = ([('0', net[0])]
          + [(f'{k}.body.{j}', net[k].body[j])
             for k in range(1, 9) for j in (0, 1)]
          + [('9', net[9])])
handles = [blk.register_forward_hook(make_finite_check(name))
           for name, blk in leaves]
net[3].body[0].weight.data()[0, 0] = float('nan')  # sabotage one layer
try:
    net(np.random.normal(0, 1, (2, 20)))
except RuntimeError as e:
    print(e)
for h in handles:
    h.detach()
```

:begin_tab:`pytorch`
The report names module `3.body.0`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream.
:end_tab:

:begin_tab:`jax`
We inspect the captured outputs of linear modules in object-graph order, which
matches execution order for this sequential network. The report names
`('layers', 3, 'body', 'layers', 0)`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream.
:end_tab:

:begin_tab:`tensorflow`
The report names `block3`, the layer we poisoned, rather than leaving you
to bisect with print statements while NaNs propagate through everything
downstream. The check runs on every forward pass until you edit it out of
`call`, the price of building observation into the model rather than
attaching it from outside. For a one-off hunt TensorFlow also ships the
whole idiom as a switch: `tf.debugging.enable_check_numerics()` instruments
every operation and reports the first one to produce an inf or NaN.
:end_tab:

:begin_tab:`mxnet`
The report names block `3.body.0`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream. One difference from PyTorch shows in the setup:
Gluon 2.0 blocks carry no path names and there is no `named_modules`
equivalent, so we spell out the list of leaves ourselves, easy here
because the structure is ours. Registration order does not matter: hooks
fire in execution order, so the first completed forward with a
non-finite output is the culprit.
:end_tab:

### Backward Hooks and Beyond

:begin_tab:`pytorch`
Gradients get the same treatment.
`module.register_full_backward_hook(f)` runs
`f(module, grad_input, grad_output)` after the module's gradients are
computed, which is how you log per-layer gradient norms, catch exploding
gradients at their source, or experiment with per-layer clipping (use this
API; the older `register_backward_hook` is deprecated and unreliable for
modules with multiple inputs). Like forward hooks, it returns a handle to
remove and should detach anything it stores. For the specific job of
extracting intermediate features from a standard vision backbone,
torchvision's `create_feature_extractor` is the production upgrade: it
traces the model with `torch.fx` and returns a module that outputs the
requested internal nodes directly, provided the model is traceable, whereas
hooks work on anything.
:end_tab:

:begin_tab:`jax`
Gradients need no hook at all, because they are not events that fire
inside a backward pass: `nnx.grad` returns them as a tree of values in
the same shape as the parameters. To log per-layer gradient norms, catch
exploding gradients at their source, or experiment with per-layer
clipping, compute the gradient tree and inspect or transform it like any
other data:
:end_tab:

:begin_tab:`tensorflow`
There are no backward hooks, but gradients do not need them:
`tf.GradientTape` already hands back the gradient of every variable as a
value. To log per-layer gradient norms, catch exploding gradients at their
source, or experiment with per-layer clipping, compute the gradients and
inspect or transform them like any other data. For the specific job of
extracting features from a pretrained backbone, the surgery idiom is also
the production tool:
`tf.keras.Model(backbone.input, backbone.get_layer('avg_pool').output)`
turns any functional backbone into a feature extractor by naming the
tensor you want.
:end_tab:

:begin_tab:`mxnet`
Gluon has no backward hooks: nothing can be attached to observe gradients
*as* the backward pass computes them. What it has is post-hoc inspection,
and for most debugging that is enough: after `loss.backward()` every
parameter's gradient sits in `param.grad()`, so to log per-layer gradient
norms or catch an explosion you iterate `net.collect_params()` and read
them like any other array. What post-hoc inspection cannot do is act
*during* the pass, so per-layer gradient clipping in the style of
exercise 3 is written into the training step instead; Gluon's `Trainer`
splits `step` into `allreduce_grads` and `update` precisely so that such
code can stand between them.
:end_tab:

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab jax
grads = nnx.grad(lambda model: (model(X) ** 2).mean())(net)
norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
print(norms['layers'][3]['body']['layers'][0])
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab tensorflow
with tf.GradientTape() as tape:
    loss = tf.reduce_mean(net(X) ** 2)
grads = tape.gradient(loss, net.trainable_variables)
print({v.path: float(tf.norm(g))  # block 3's first layer
       for v, g in list(zip(net.trainable_variables, grads))[10:12]})
```

## Summary

Reproducibility is an inventory problem: initialization, dropout,
shuffling, augmentation, and loader workers each draw from some generator,
and a run repeats only if every one of them is seeded.

:begin_tab:`pytorch`
`torch.manual_seed` covers torch's CPU and CUDA generators; NumPy and
`random` need their own seeds, and loader workers need them *per worker*
via `worker_init_fn` plus a `generator` for the shuffle order. Seeding
makes the program repeatable, not the arithmetic:
`torch.use_deterministic_algorithms(True)` pins kernel choice too, raising
on operations that cannot comply.
:end_tab:

:begin_tab:`jax`
In JAX the inventory collapses to one item, the key you pass: the same
key gives the same draws by construction of the counter-based PRNG, and
independent streams come from `jax.random.split`, never from hidden
state. Keys make the program repeatable, not the arithmetic: on CPU,
XLA's kernels are already deterministic, while on GPU the flag
`--xla_gpu_deterministic_ops=true` pins kernel choice too.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.utils.set_random_seed` covers Python's `random`, NumPy, and
TensorFlow's global generator in one call; `tf.random.Generator` objects
and `Dataset.shuffle(seed=...)` carry their seeds explicitly. Seeding
makes the program repeatable, not the arithmetic:
`tf.config.experimental.enable_op_determinism()` pins kernel choice too,
raising on operations that cannot comply, and wants to be the first line
of the program.
:end_tab:

:begin_tab:`mxnet`
`np.random.seed` covers MXNet's generator on every device in one call;
classic NumPy and Python's `random` need their own seeds, and Gluon's
loader consults NumPy for its shuffle order while offering no per-worker
seeding hook, so worker randomness must be pinned inside the dataset.
Seeding makes the program repeatable, not the arithmetic: MXNet has no
determinism switch, only the autotuning lever
`MXNET_CUDNN_AUTOTUNE_DEFAULT=0`.
:end_tab:

Bitwise agreement is a debugging tool;
conclusions that hold across seeds are the scientific goal.

:begin_tab:`pytorch`
For looking
inside a model, forward and backward hooks observe any module without
editing it, subject to two rules: detach what you stash, and remove the
handle.
:end_tab:

:begin_tab:`jax`
For looking inside a model, `nnx.capture(..., method_outputs=...)` returns
submodule outputs from an unmodified call, `sow` records named values from
the inside, and gradients are values from `nnx.grad` you inspect
directly.
:end_tab:

:begin_tab:`tensorflow`
For looking inside a model there is no hook to attach: declare the tensors
you want as extra outputs of a functional model, or override `call` where
you own the code, and read gradients as values from `tf.GradientTape`.
:end_tab:

:begin_tab:`mxnet`
For looking inside a model, forward hooks and pre-hooks observe any block
without editing it, subject to two rules: detach what you stash, and
detach the `HookHandle` when done. Backward hooks do not exist; gradients
are read after the fact from `param.grad()`.
:end_tab:

## Exercises

1. Extend `train_once` to load its data through a `DataLoader` with
   `num_workers=4` and a custom `Dataset` whose `__getitem__` adds noise
   drawn from `np.random`. Run it twice with the same seed on a Linux
   machine. Is it reproducible? Which of the two worker failure modes do
   you observe, and why does `torch.manual_seed` alone not fix it? Repair
   the script with `seed_worker` and a seeded `generator`, and verify
   bitwise agreement.
2. Write a forward hook that counts multiply-accumulate operations for
   every `nn.Linear` from the shapes of its input and weight, and use it to
   report per-layer and total FLOPs for the residual stack above. Check the
   total against a hand count.
3. Using `register_full_backward_hook`, clip each layer's gradient to a
   fixed norm during the backward pass, and compare training against global
   gradient-norm clipping. When do the two differ most?
4. A forward hook that returns a value *replaces* the module's output. Use
   one to zero out the output of a single residual block's body (turning
   the block into the identity) and measure how the network's output
   changes. Which block matters most, and how would you find out with one
   loop?

:begin_tab:`jax`
5. Rebuild the activation-statistics table two more ways: with a
   filtered `nnx.capture` that records only `ResidualBlock` outputs, and by
   editing the block to call `self.sow(nnx.Intermediate, 'body_out', ...)`
   on its body's output. Compare capture-all-methods, opt-in `sow`, and
   PyTorch-style mutable hooks: which requires touching model
   code, which can silently retain memory, and which would you want for a
   model you do not own?
:end_tab:

:begin_tab:`tensorflow`
5. Rebuild the activation-statistics table two more ways: with a
   `Checked`-style `call` override that stashes `tf.math.reduce_std` of
   every layer's output, and on a model you did not write, a
   `tf.keras.applications` backbone, by naming layers with `get_layer`.
   Compare the three contracts you now know, functional surgery, `call`
   overrides, and PyTorch-style mutable hooks: which requires a symbolic
   graph, which requires owning the model's code, and which black-box
   models does each admit?
:end_tab:

:begin_tab:`mxnet`
5. Rebuild the activation-statistics table two more ways: with
   `register_forward_pre_hook`, recording the standard deviation of each
   block's *input* (the pre-hook receives the argument tuple, so unpack
   it), and by reading the source of `Block.summary()`, which builds its
   whole table from one forward hook, then calling it on the residual
   stack. Then revisit exercise 4: Gluon ignores a hook's return value,
   so an output-replacing hook cannot be written; perform the same
   ablation by editing the block's `forward` instead, and compare the two
   contracts, observe-only hooks against PyTorch's mutable ones, for a
   model you own and for one you do not.
:end_tab:
