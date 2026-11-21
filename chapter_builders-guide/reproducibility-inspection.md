```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Reproducibility and Inspection
:label:`sec_repro`

Suppose you tweak a single line in a training script and validation accuracy on
Fashion-MNIST edges up by 0.3 percentage points. Before celebrating, you need to
know how much two runs of the *unchanged* script vary on their own. In
:numref:`subsec_seed_spread`, we run softmax regression five times differing only
in their random seed and watch final accuracies span one to four percentage
points simply from different initial weights and batch shuffling—swamping a 0.3
point gain completely.

Debugging a crashing training run presents the mirror-image problem: if the loss
explodes to NaN after hours of training, we want to reproduce the exact sequence
of steps leading up to the crash and then peer inside the running network to
pinpoint which layer emitted the first non-finite value.

This section addresses both halves together. First, we make runs repeatable by
controlling pseudorandom number generators—global seeds, private generator
objects, and multi-worker data loaders—and then tackle the trickier GPU wrinkle:
floating-point non-associativity under parallel atomic reductions. Second, we
inspect live models without modifying their code using *hooks* (and their
framework equivalents) to log activation scales, catch the first NaN layer, and
monitor gradient norms. Both halves reinforce each other: internal diagnostics
are only comparable across two code versions when nothing else changes between
runs. On a first pass, feel free to skim the subsections on generator objects
and data-loader workers, revisiting them once your data pipelines perform random
augmentations.

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection-1}
%%tab mxnet, tensorflow
import random
```

```{.python .input #reproducibility-inspection-reproducibility-and-inspection-1}
%%tab pytorch
from torch.utils.data import DataLoader, Dataset
```

## Seeds and Random Number Generators
:label:`subsec_seeds_randomness`

Randomness sneaks into a training pipeline from far more places than most
practitioners realize: weight initialization (:numref:`sec_init_param`), dropout
masks at every step, epoch-by-epoch minibatch shuffling, random data
augmentations (crops, flips, color jitter), and parallel data-loader worker
processes. Every random number comes from a *pseudorandom number generator*
(PRNG)—a deterministic algorithm whose entire sequence is determined by its
initial *seed*. Seeding every generator the pipeline touches therefore locks
down every random draw.

### Global Seeds

Deep learning frameworks either pull random numbers from a shared global
generator or require passing generator state explicitly as an argument. Let's
demonstrate seeding on a small five-step training loop where weights and
synthetic inputs all come from one seeded source.

:begin_tab:`mxnet, pytorch`
`mnp.random.seed(k)` and `torch.manual_seed(k)` seed the framework's global CPU and GPU generators (note that Python's `random` and NumPy's `np.random` must be seeded separately if used).
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.utils.set_random_seed(k)` seeds Python's `random`, NumPy's global RNG, and TensorFlow's global RNG in one call.
:end_tab:

:begin_tab:`jax`
JAX has no global RNG: every random function takes an explicit *key* (`jax.random.key(seed)`), deriving independent streams via `jax.random.split`.
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

Both runs with `seed=0` match *bit for bit*—every weight and the final loss are
identical—because identical initial weights, data, and deterministic arithmetic
produce identical gradients (we examine operation order in
:numref:`subsec_deterministic_arithmetic`). Changing the seed to `1` yields a
distinct trajectory and loss.

:begin_tab:`tensorflow`
We placed the TensorFlow run on the CPU. On a GPU, the same seeded code
gives equal losses but, in some runs, weights that differ in their last
bits.
:end_tab:

### Generator Objects

Relying on a single global PRNG couples every piece of code that draws from it:
adding a single extra layer or a temporary random print during debugging shifts
the entire downstream stream, silently altering train/validation splits, weight
initializations, and dropout masks all at once. A private *generator object*
isolates a stream behind its own seed so unrelated draws elsewhere can never
perturb it.

:begin_tab:`mxnet`
Because MXNet has no private generator objects, we re-seed right before draws that must stay fixed:
:end_tab:

:begin_tab:`pytorch, tensorflow`
`torch.Generator().manual_seed(42)` and `tf.random.Generator.from_seed(42)` create isolated PRNG streams unaffected by global draws:
:end_tab:

:begin_tab:`jax`
Every JAX key is already a stateless, isolated value—draws with `key_other` never affect `key_split`:
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

### Data Loader Workers

When data loaders prepare minibatches in parallel *worker* processes, random
data augmentation introduces two silent failure modes. If workers are spawned via
OS `fork` as clones of the main process without re-seeding, every worker
inherits the exact same generator state and generates identical "random"
augmentations. Conversely, if each worker initializes from OS entropy, no two
runs will ever match.

:begin_tab:`mxnet`
Gluon's `DataLoader` forks worker processes with identical copies of NumPy's global RNG state and accepts no `worker_init_fn`. To avoid identical draws across workers, seed a per-example `np.random.default_rng([seed, i])` inside `__getitem__` (and seed `np.random.seed` for the loader's shuffle order):
:end_tab:

:begin_tab:`pytorch`
`DataLoader` seeds each worker's PyTorch, `random`, and global NumPy RNGs from `base_seed + worker_id`, but a private `np.random.default_rng()` attribute on the `Dataset` is cloned identically across forked workers unless re-seeded via `worker_init_fn` using `torch.utils.data.get_worker_info().seed`:
:end_tab:

:begin_tab:`tensorflow`
`tf.data` runs parallel threads inside one process and takes explicit seeds (`Dataset.shuffle(buffer_size, seed=0)`), avoiding forked-RNG duplication by design:
:end_tab:

:begin_tab:`jax`
In JAX, workers receive explicit keys derived via `jax.random.split(key, num_workers)` and `jax.random.fold_in(key, epoch)`, making accidental state sharing impossible unless two workers are handed the same key.
:end_tab:
```{.python .input #reproducibility-inspection-dataloader-workers-2}
%%tab mxnet
class NoisyData(gluon.data.Dataset):
    """Eight examples whose values are random draws, as in augmentation."""
    def __init__(self, draw):
        self.draw = draw  # draw(i) returns the value of example i

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return self.draw(i)

def load(draw):  # the workers exit when the loader is deleted
    loader = gluon.data.DataLoader(NoisyData(draw), batch_size=4,
                                   num_workers=2)
    return [[round(x, 3) for x in batch.tolist()] for batch in loader]

print(load(lambda i: np.random.random()))  # NumPy's global generator
print(load(lambda i: np.random.default_rng([0, i]).random()))  # per example
```

```{.python .input #reproducibility-inspection-dataloader-workers-2}
%%tab pytorch
class NoisyData(Dataset):
    """Eight examples whose values are random draws, as in augmentation."""
    def __init__(self):
        self.rng = np.random.default_rng()  # the dataset's own generator

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return self.rng.random()

def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)  # this worker's seed

for init_fn in (None, seed_worker):
    loader = DataLoader(NoisyData(), batch_size=4, num_workers=2,
                        worker_init_fn=init_fn,
                        multiprocessing_context='fork',  # copies of this one
                        generator=torch.Generator().manual_seed(0))
    print([[round(x, 3) for x in batch.tolist()] for batch in loader])
```

```{.python .input #reproducibility-inspection-dataloader-workers-3}
%%tab tensorflow
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # epoch 1
print([b.numpy().tolist() for b in ds])  # epoch 2: reshuffled, still seeded
ds = tf.data.Dataset.range(8).shuffle(8, seed=0).batch(4)
print([b.numpy().tolist() for b in ds])  # rebuilt pipeline: epoch 1 again
```

:begin_tab:`mxnet, pytorch`
Without per-worker or per-example re-seeding, both workers output identical "random" numbers in the first line; proper seeding gives distinct, reproducible draws in the second line.
:end_tab:
### Randomness as a Value

Every worker bug above stems from hidden mutable generator state being copied or
shared implicitly behind the scenes. JAX eliminates hidden state by treating
randomness as an explicit value: every random call takes a key, calling it twice
with the same key returns identical numbers, and independent streams come from
explicitly splitting keys. While you can still reuse a key accidentally, the
reuse is right there in the code. The price is plumbing keys through every
stochastic function. A global seed is simply the implicit shorthand for that
same contract, and private generator objects recover explicit isolation wherever
needed.

### The Spread across Seeds
:label:`subsec_seed_spread`

Now let's measure the seed-to-seed variance mentioned at the start of the
section. Below we train softmax regression (:numref:`sec_softmax_concise`) on
Fashion-MNIST for three epochs on a GPU across five seeds (0 through 4), seeding
every generator involved and returning each run's final validation accuracy and
weights.

```{.python .input #reproducibility-inspection-seed-spread-1}
%%tab mxnet
def train_classifier(seed, data):
    mnp.random.seed(seed)  # initialization, on every device
    np.random.seed(seed)  # the loader's shuffle order
    model = d2l.SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=3, num_gpus=1)
    trainer.board.display = False
    trainer.fit(model, data)
    return trainer.history['val_acc'][-1], model.net.weight.data().copy()

data = d2l.FashionMNIST(batch_size=256)
runs = [train_classifier(seed, data) for seed in range(5)]
accs = 100 * np.array([acc for acc, _ in runs])
print('validation accuracy (%):', ' '.join(f'{a:.1f}' for a in accs))
print(f'mean {accs.mean():.1f}, max - min {np.ptp(accs):.1f} points')
```

```{.python .input #reproducibility-inspection-seed-spread-1}
%%tab pytorch
def train_classifier(seed, data):
    torch.manual_seed(seed)  # initialization, shuffle order, worker seeds
    model = d2l.SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=3, num_gpus=1)
    trainer.board.display = False
    trainer.fit(model, data)
    return trainer.history['val_acc'][-1], model.net[1].weight.detach().cpu()

data = d2l.FashionMNIST(batch_size=256)
runs = [train_classifier(seed, data) for seed in range(5)]
accs = 100 * np.array([acc for acc, _ in runs])
print('validation accuracy (%):', ' '.join(f'{a:.1f}' for a in accs))
print(f'mean {accs.mean():.1f}, max - min {np.ptp(accs):.1f} points')
```

```{.python .input #reproducibility-inspection-seed-spread-1}
%%tab tensorflow
def train_classifier(seed, data):
    tf.keras.utils.set_random_seed(seed)  # initialization and shuffle order
    model = d2l.SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=3, num_gpus=1)
    trainer.board.display = False
    trainer.fit(model, data)
    return (trainer.history['val_acc'][-1],
            tf.identity(model.net.layers[1].kernel))

data = d2l.FashionMNIST(batch_size=256)
runs = [train_classifier(seed, data) for seed in range(5)]
accs = 100 * np.array([acc for acc, _ in runs])
print('validation accuracy (%):', ' '.join(f'{a:.1f}' for a in accs))
print(f'mean {accs.mean():.1f}, max - min {np.ptp(accs):.1f} points')
```

```{.python .input #reproducibility-inspection-seed-spread-1}
%%tab jax
class KeyedFashionMNIST(d2l.FashionMNIST):
    """Fashion-MNIST whose shuffle order is drawn from a key."""
    def __init__(self, key, batch_size=256):
        super().__init__(batch_size)
        self.key = key

    def get_dataloader(self, train):
        X, y = self.train if train else self.val
        X = (X[..., None] / 255).astype(np.float32)  # channel axis last
        return d2l.ArrayLoader((X, y.astype(np.int32)), self.batch_size,
                               shuffle=train, key=self.key)

def train_classifier(seed):
    key_init, key_order = jax.random.split(jax.random.key(seed))
    model = d2l.SoftmaxRegression(num_outputs=10, lr=0.1,
                                  rngs=nnx.Rngs(key_init))
    trainer = d2l.Trainer(max_epochs=3, num_gpus=1)
    trainer.board.display = False
    trainer.fit(model, KeyedFashionMNIST(key_order))
    return trainer.history['val_acc'][-1], model.net.kernel[...]

runs = [train_classifier(seed) for seed in range(5)]
accs = 100 * np.array([acc for acc, _ in runs])
print('validation accuracy (%):', ' '.join(f'{a:.1f}' for a in accs))
print(f'mean {accs.mean():.1f}, max - min {np.ptp(accs):.1f} points')
```

Even though all five runs execute identical code and hyperparameters, their
final accuracies spread by one to four percentage points around 81%—confirming
that a 0.3-point jump from a single run is pure noise. Sound empirical
comparisons always report the mean and spread across multiple seeds.

To verify that seed choice alone drove those differences, let's rerun `seed=0`
and compare both accuracy and every weight against the first run:

```{.python .input #reproducibility-inspection-seed-spread-2}
%%tab mxnet
acc, w = train_classifier(0, data)
print('same accuracy:', acc == runs[0][0],
      '| same weights:', bool((w == runs[0][1]).all()),
      f'| largest difference {float(mnp.abs(w - runs[0][1]).max()):.2g}')
```

```{.python .input #reproducibility-inspection-seed-spread-2}
%%tab pytorch
acc, w = train_classifier(0, data)
print('same accuracy:', acc == runs[0][0],
      '| same weights:', torch.equal(w, runs[0][1]),
      f'| largest difference {(w - runs[0][1]).abs().max().item():.2g}')
```

```{.python .input #reproducibility-inspection-seed-spread-2}
%%tab tensorflow
acc, w = train_classifier(0, data)
diff = float(tf.reduce_max(tf.abs(w - runs[0][1])))
print('same accuracy:', acc == runs[0][0],
      '| same weights:', bool(tf.reduce_all(w == runs[0][1])),
      f'| largest difference {diff:.2g}')
```

```{.python .input #reproducibility-inspection-seed-spread-2}
%%tab jax
acc, w = train_classifier(0)
print('same accuracy:', acc == runs[0][0],
      '| same weights:', bool(jnp.array_equal(w, runs[0][1])),
      f'| largest difference {float(jnp.abs(w - runs[0][1]).max()):.2g}')
```

The rerun matches the first run's validation accuracy identically. Whether every
single weight matches down to the last bit depends on whether the GPU sums
floating-point numbers in a fixed order—something seeds alone do not control
(:numref:`subsec_deterministic_arithmetic`).

:begin_tab:`tensorflow`
On a GPU, TensorFlow's bias-gradient reduction sums terms in non-deterministic thread order, so two identically seeded GPU runs can differ by $\sim 10^{-6}$ in their lowest bits.
:end_tab:
## Deterministic Arithmetic
:label:`subsec_deterministic_arithmetic`

Seeding fixes *which* random numbers a run draws, not *how* floating-point
arithmetic combines them. Because every floating-point addition rounds to the
nearest representable value, floating-point addition is *not* associative:
changing summation order can change the result.

Three float32 numbers make the effect unmistakable. Near $10^8$, adjacent
float32 numbers lie 8 apart (:numref:`sec_numerics`), so $-10^8 + 1$ rounds
right back to $-10^8$:

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

Simply shifting parentheses changes the answer from `1.0` to `0.0`. In long
reductions the discrepancy is subtler but pervasive: summing a million floats
forward versus backward routinely differs in the lowest bits.

On a CPU with a fixed thread count, reduction order usually stays constant, so
seeded runs repeat cleanly. On a GPU, thousands of threads race in parallel, and
several fast kernels—including embedding gradients, scatter-adds, certain
convolution backward passes, and reductions—accumulate partial results via
*atomic additions* that land in whatever order threads finish. Two identically
seeded runs on the same GPU can therefore diverge in their lowest bits, which
chaotic gradient steps eventually amplify into visibly distinct loss curves.
Every framework provides a determinism mode (or alternative kernel) that forces
a fixed summation order, typically at some speed cost. Let's watch this directly
on a GPU *scatter-add* summing one million random numbers into ten bins twice in
a row.

:begin_tab:`mxnet`
`MXNET_ENFORCE_DETERMINISM=1` restricts cuDNN convolution algorithms but leaves `npx.index_add` non-deterministic on GPU; rewriting the scatter-add as a one-hot matrix multiplication uses a fixed-order cuBLAS reduction:
:end_tab:

:begin_tab:`pytorch`
`torch.use_deterministic_algorithms(True)` switches operations like `index_add_` to deterministic kernels (or raises a `RuntimeError` if none exists):
:end_tab:

:begin_tab:`tensorflow`
`tf.config.experimental.enable_op_determinism()` forces deterministic GPU kernels (such as `tf.math.unsorted_segment_sum`) for the rest of the process:
:end_tab:

:begin_tab:`jax`
Passing `compiler_options={'xla_gpu_deterministic_ops': True}` to `jax.jit` (or setting `XLA_FLAGS=--xla_gpu_deterministic_ops=true`) forces deterministic XLA GPU reductions:
:end_tab:
```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab mxnet
if npx.num_gpus() > 0:
    src = mnp.random.normal(0, 1, (2**20,), device=npx.gpu())  # the numbers
    idx = mnp.random.randint(0, 10, (2**20,), device=npx.gpu())  # their bins
    sums = {'index_add': lambda: npx.index_add(
                mnp.zeros(10, device=npx.gpu()), idx.reshape(1, -1), src),
            'one-hot matmul': lambda: npx.one_hot(idx, 10).T @ src}
    for name, scatter_add in sums.items():
        a, b = scatter_add(), scatter_add()
        print(f'{name:15s} bitwise equal: {bool((a == b).all())}, '
              f'largest difference {float(mnp.abs(a - b).max()):.2g}')
else:
    print('no GPU: on a CPU these sums add in a fixed order')
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab pytorch
if torch.cuda.is_available():
    src = torch.randn(2**20, device='cuda')  # a million numbers
    idx = torch.randint(0, 10, (2**20,), device='cuda')  # their bins
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        a, b = [torch.zeros(10, device='cuda').index_add_(0, idx, src)
                for _ in range(2)]
        print(f'deterministic {deterministic!s:5}: bitwise equal '
              f'{torch.equal(a, b)}, largest difference '
              f'{(a - b).abs().max().item():.2g}')
    torch.use_deterministic_algorithms(False)
else:
    print('no GPU: on a CPU this sum adds in a fixed order')
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab tensorflow
if tf.config.list_physical_devices('GPU'):
    src = tf.random.normal((2**20,))  # a million numbers
    idx = tf.random.uniform((2**20,), maxval=10, dtype=tf.int32)  # their bins
    for deterministic in (False, True):
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        a, b = [tf.math.unsorted_segment_sum(src, idx, 10) for _ in range(2)]
        print(f'deterministic {deterministic!s:5}: bitwise equal '
              f'{bool(tf.reduce_all(a == b))}, largest difference '
              f'{float(tf.reduce_max(tf.abs(a - b))):.2g}')
else:
    print('no GPU: on a CPU this sum adds in a fixed order')
```

```{.python .input #reproducibility-inspection-determinism-and-its-price-2}
%%tab jax
def scatter_add(idx, src):
    return jnp.zeros(10).at[idx].add(src)

if jax.default_backend() == 'gpu':
    src = jax.random.normal(d2l.get_key(), (2**20,))  # a million numbers
    idx = jax.random.randint(d2l.get_key(), (2**20,), 0, 10)  # their bins
    for deterministic in (False, True):
        f = jax.jit(scatter_add, compiler_options={
            'xla_gpu_deterministic_ops': deterministic})
        a, b = f(idx, src), f(idx, src)
        print(f'deterministic {deterministic!s:5}: bitwise equal '
              f'{bool(jnp.array_equal(a, b))}, largest difference '
              f'{float(jnp.abs(a - b).max()):.2g}')
else:
    print('no GPU: on a CPU this sum adds in a fixed order')
```

Without determinism enabled, two consecutive scatter-adds on the exact same GPU
data differ by a few thousandths on sums of a few hundred; with determinism
enabled, both calls match bit for bit.

:begin_tab:`tensorflow`
With the switch on, random operations must have a seed, since an operation
that seeds itself from the operating system is nondeterministic too. We can
demonstrate the rule by clearing the global seed:
:end_tab:

```{.python .input #reproducibility-inspection-determinism-and-its-price-3}
%%tab tensorflow
tf.config.experimental.enable_op_determinism()  # no-op if already on
tf.random.set_seed(None)  # forget the global seed for a moment
try:
    tf.random.normal((2,))
except RuntimeError as e:
    print(str(e).split('.')[0])
# determinism stays on and needs a seed, though not a particular one
tf.keras.utils.set_random_seed(random.SystemRandom().getrandbits(31))
```

Remember the proper scope of bitwise reproducibility: it holds only on a single
machine, software stack, and hardware configuration—no framework promises
identical bits across library upgrades, different GPU architectures, or CPU vs.
GPU. Bitwise determinism is an invaluable *debugging* tool for bisecting the
exact step where two runs diverge. Scientific reproducibility is both looser and
stronger: any genuine improvement ("this architectural change improves
accuracy") must hold robustly across multiple seeds, reported as mean and spread
just as in :numref:`subsec_seed_spread` (~81%, spanning 1–4 points). As with
16-bit formats (:numref:`sec_numerics`), any claim that flips under a new seed
or last-bit rounding needs stronger evidence.

## Model Inspection with Hooks

Repeating a bug reliably proves it is real, not *where* inside the network it
originates. Inserting print statements into every layer's `forward` method is
clumsy—and impractical for third-party library models whose source code you do
not want to fork. A *hook* solves this cleanly: it is a callback registered from
outside the model that the framework invokes automatically before or after a
layer executes.

:begin_tab:`mxnet`
`block.register_forward_hook(f)` runs `f(block, inputs, output)` inside `__call__` after `forward` (and `register_forward_pre_hook` runs before it).
:end_tab:

:begin_tab:`pytorch`
`module.register_forward_hook(f)` runs `f(module, inputs, output)` inside `__call__` after `forward` (and returning a new tensor replaces the output).
:end_tab:

:begin_tab:`tensorflow`
Keras has no external hook API; instead, we either build a secondary `tf.keras.Model` exposing intermediate symbolic tensors of a functional graph, or log tensors inside a custom `call`.
:end_tab:

:begin_tab:`jax`
`nnx.capture(net, nnx.Intermediate, method_outputs=nnx.Intermediate)(X)` captures the return value of every submodule in a single functional call without modifying model code.
:end_tab:
![The `__call__` wrapper as a pipeline: input flows through pre-hooks, then forward, then hooks, to the output, with the two hook stages dashed and orange against forward's solid blue, and a side arrow from the hooks stage to an observer that can capture, check, or modify.](../img/bg-hooks.svg)
:label:`fig_bg_hooks`

For our inspection examples, we build an eight-block residual stack of width 64
between a 20-input layer and a 10-output head (:numref:`sec_model_construction`),
each block computing $\mathbf{h} = \mathbf{x} + \textrm{body}(\mathbf{x})$ with
Xavier-initialized weights (:numref:`subsec_xavier`) and zero biases. We
deliberately leave this network unseeded so each run draws fresh random weights,
which is why the diagnostic measurements below are cited as typical ranges.

:begin_tab:`mxnet`
The seeded runs above left MXNet's global generator at a fixed state, which
would fix this model's weights too, so the cell first reseeds the generator
from the operating system's entropy.
:end_tab:

:begin_tab:`pytorch`
The seeded runs above left PyTorch's global generator at a fixed state,
which would fix this model's weights too, so the cell first reseeds the
generator from the operating system's entropy with `torch.seed()`.
:end_tab:

```{.python .input #reproducibility-inspection-hooks-looking-inside}
%%tab mxnet
mnp.random.seed(random.SystemRandom().getrandbits(31))  # from the OS

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
torch.seed()  # a fresh seed from the operating system

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

In :numref:`sec_init_param`, we manually looped through blocks to measure
activation standard deviations across depth. With hooks, we can collect the
exact same profile from an unmodified model in just a few lines on a minibatch
of 256 standard normal inputs:

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

Activation standard deviations climb steadily with depth—from roughly 0.7 right
after the input layer to around 3–4 after the eighth block—matching theory
closely. Xavier initialization gives the $20 \to 64$ input layer weight variance
$2/84$, yielding output variance $20 \cdot 2/84 \approx 0.48$ (standard
deviation 0.69). Each residual block then adds an approximately uncorrelated
body output whose variance is half its input variance (the first linear layer
preserves variance, ReLU halves the mean square, and the second linear layer
preserves it), multiplying total variance by $1.5$—and standard deviation by
$\sqrt{1.5} \approx 1.22$—at every block. Across eight blocks,
$0.69 \times 1.5^4 \approx 3.5$, right in the middle of our measurements. This
is the exact variance-growth analysis of :numref:`sec_init_param` (with factor
$1.5$ here instead of $2$ under He initialization), underscoring why deep
residual branches benefit from scaled-down output projections.

:begin_tab:`mxnet, pytorch`
Two rules keep hooks leak-free: always `.detach()` any tensor stored inside a hook (otherwise the entire autograd graph stays alive in memory across steps), and always remove the hook handle (`h.detach()` in Gluon, `h.remove()` in PyTorch) once inspection is done.
:end_tab:

:begin_tab:`tensorflow, jax`
Neither `probe(X)` nor `nnx.capture` attaches persistent state to the model, so nothing needs to be detached or removed afterwards.
:end_tab:
### A NaN Finder
:label:`subsec_nan_finder`

When a training loss suddenly turns NaN at step 40,000, your first diagnostic
question is *which layer emitted the first non-finite value?* Attaching a
finiteness hook to every leaf layer pinpoints the culprit in a single forward
pass, because hooks execute in forward execution order. To test our detector, we
deliberately inject a NaN into one weight of the third residual block (index 3):

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

The hook immediately identifies the corrupted layer in block 3 without adding a
single print statement inside the layer classes (for whole-graph NaN traps
during debugging, TensorFlow and JAX also offer `tf.debugging.enable_check_numerics()`
and `jax.config.update('jax_debug_nans', True)`).
Of course, catching a planted NaN is the clean demo case; in a real training
crash, the finder points you to the first *symptom*, while the underlying cause
(such as runaway activation growth culminating in overflow) may begin several
layers earlier.

### Gradient Inspection

We can monitor the backward pass with the same precision. Since forward
activations grew across our eight residual blocks, do gradients grow in reverse?
Using the mean squared output on 256 standard normal inputs as our loss, we log
the gradient norm at each residual block's output alongside the weight and bias
gradient norms of the first linear layer in block 3.

:begin_tab:`mxnet, jax`
To inspect intermediate output gradients without backward hooks, we add a zero tensor `Z` to each block's output and differentiate with respect to `Z`:
:end_tab:

:begin_tab:`pytorch`
`module.register_full_backward_hook(f)` invokes `f(module, grad_input, grad_output)` during `backward()` (firing in reverse topological order, from last block to first):
:end_tab:

:begin_tab:`tensorflow`
`tape.gradient(loss, [acts[1:], net.trainable_variables])` computes gradients with respect to both intermediate functional outputs and trainable variables in one call:
:end_tab:
```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab mxnet
X = mnp.random.normal(0, 1, (256, 20))
zeros = [mnp.zeros((256, 64)) for _ in range(8)]  # one per block output
for Z in zeros:
    Z.attach_grad()
with autograd.record():
    H = net[0](X)
    for k, Z in enumerate(zeros, start=1):
        H = net[k](H) + Z
    loss = (net[9](H) ** 2).mean()
loss.backward()
print('block output gradients:',
      ' '.join(f'{float(mnp.linalg.norm(Z.grad)):.3g}' for Z in zeros))
layer = net[3].body[0]  # block 3, first layer
print(f'weight {float(mnp.linalg.norm(layer.weight.grad())):.3g}, '
      f'bias {float(mnp.linalg.norm(layer.bias.grad())):.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab pytorch
X = torch.randn(256, 20)
grad_norms, handles = [], []
def record(module, grad_input, grad_output):
    grad_norms.append(grad_output[0].norm().item())  # store a number
for block in net[1:-1]:
    handles.append(block.register_full_backward_hook(record))
net.zero_grad()
(net(X) ** 2).mean().backward()
for h in handles:
    h.remove()
# The hooks fire in backward order, last block first
print('block output gradients:',
      ' '.join(f'{g:.3g}' for g in reversed(grad_norms)))
layer = net[3].body[0]  # block 3, first layer
print(f'weight {layer.weight.grad.norm().item():.3g}, '
      f'bias {layer.bias.grad.norm().item():.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab tensorflow
with tf.GradientTape() as tape:
    *acts, out = probe(X)  # every layer's output, then the stack's
    loss = tf.reduce_mean(out ** 2)
block_grads, grads = tape.gradient(
    loss, [acts[1:], net.trainable_variables])
print('block output gradients:',
      ' '.join(f'{float(tf.norm(g)):.3g}' for g in block_grads))
kernel, bias = grads[10:12]  # block 3, first layer
print(f'kernel {float(tf.norm(kernel)):.3g}, bias {float(tf.norm(bias)):.3g}')
```

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab jax
def loss_fn(model, zeros):  # zeros: added to each block's output
    H = model.layers[0](X)
    for block, Z in zip(model.layers[1:-1], zeros):
        H = block(H) + Z
    return (model.layers[-1](H) ** 2).mean()

zeros = [jnp.zeros((256, 64)) for _ in range(8)]
grads, block_grads = nnx.grad(loss_fn, argnums=(0, 1))(net, zeros)
print('block output gradients:',
      ' '.join(f'{float(jnp.linalg.norm(g)):.3g}' for g in block_grads))
layer = grads['layers'][3]['body']['layers'][0]  # block 3, first layer
print(f"kernel {float(jnp.linalg.norm(layer['kernel'][...])):.3g}, "
      f"bias {float(jnp.linalg.norm(layer['bias'][...])):.3g}")
```

Reading backward from the eighth block to the first, the output gradient norm
grows by roughly $1.2\times$ to $1.4\times$ per block—five- to sixfold across
the stack—mirroring the forward $1.22\times$ growth for the exact same reason:
on the backward pass, each residual block adds the branch gradient back onto the
skip-connection gradient. Isolated parameter gradient norms mean little on their
own, but comparing them across depth or tracking them across training steps
reveals vanishing or exploding dynamics immediately (see the exercises).

Remember what these debugging tools do and do not solve: they cannot force two
different machines to match bit for bit, nor can a single run settle whether an
experimental gain is real—for that, compare distributions across multiple seeds.
Yet combined with resumable checkpoints (:numref:`sec_read_write`), you now have
every piece needed to replay a failing job right up to the crash and attach a
diagnostic hook to isolate the exact layer at fault.

The 8-billion-parameter language model that opened this chapter has not shrunk—it
still weighs 32 GB in fp32—but nothing about how frameworks handle it is a black
box anymore. Constructing architectures from configs, auditing parameters and
buffers, initializing deep stacks, building custom layers and gradients, training
in mixed precision, saving crash-safe checkpoints, managing GPU memory, and
inspecting live runs all operate over the exact same tree of modules. Equipped
with this engineering toolkit, we are ready to move on to architectures built
around the structure of real-world data, starting with convolutional neural
networks in :numref:`chap_cnn`.

## Summary

Every random choice in training—initialization, shuffling, dropout, and data
augmentation—traces back to pseudorandom number generators controlled by seeds,
with private generator objects or explicit keys preventing unrelated code from
perturbing a stream. Because floating-point addition is non-associative and
parallel GPU atomic operations sum in arbitrary thread order, bit-for-bit
repeatability also requires deterministic kernels and holds only on one machine
and software stack—making bitwise determinism a debugging tool, while scientific
conclusions must hold across multiple seeds. Finally, forward and backward
hooks (and their framework equivalents) let us inspect activation scales,
gradient norms, and first-occurrence NaNs inside unmodified models.

## Exercises

:begin_tab:`mxnet`
1. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$ of `net`, with each variance taken over all
   entries. On one batch, record for every `ResidualBlock` the variance of
   $\mathbf{x}_k$ with `register_forward_pre_hook` (the pre-hook receives
   the argument tuple, so unpack it), of $\mathbf{h}_k$ with
   `register_forward_hook`, and of $\mathbf{b}_k$ with a forward hook on
   its `body`. The standard deviations in the table of
   :numref:`subsec_activation_stats` grow because each block adds its
   body's output to the stream. Measure how closely

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
2. [code] **Seeds and data loading.** ● Extend `train_once` to load its
   data through Gluon's `DataLoader` with `shuffle=True` and
   `num_workers=4`, keeping the single `mnp.random.seed(seed)` call at its
   top. MXNet has no generator objects, so the dataset uses NumPy: it adds
   noise to each example inside `__getitem__`, drawn from an
   `np.random.default_rng()` object created in the dataset's constructor.
    1. Predict whether two runs with the same seed agree on the shuffle
       order and on the noise, and whether the four workers draw the same
       noise. Justify each prediction from where that randomness is drawn,
       then verify all three.
    1. Seed the noise from the example's index, as in this section, fix
       the shuffle order, and verify that two runs now agree. An example
       then receives the same noise in every epoch, which defeats the
       purpose of augmentation. Change the seeding so that the noise
       differs from epoch to epoch and still repeats from run to run.
3. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
4. [code] **Gradient norms.** ● Rebuild `net`, multiply the weight of
   every `Dense` layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Compute the norm of each block's output gradient as in this
       section, and each block's parameter-gradient norm from
       `net.collect_params()`. Plot both families of norms against depth.
    1. Express the gradient of a linear layer's weight through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

:begin_tab:`pytorch`
1. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$, with each variance taken over all entries. On one
   batch, record for every `ResidualBlock` the variance of $\mathbf{x}_k$
   with `register_forward_pre_hook` (the pre-hook receives the argument
   tuple, so unpack it), of $\mathbf{h}_k$ with a forward hook on the
   block, and of $\mathbf{b}_k$ with a forward hook on its `body`. The
   standard deviations in the table of :numref:`subsec_activation_stats`
   grow because each block adds its body's output to the stream. Measure
   how closely

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
2. [code] **Seeds and data loading.** ● Extend `train_once` to load its
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
    1. Add the `seed_worker` of this section and a seeded `generator` to
       the loader, and verify that two runs now agree and that the workers
       draw different noise. Does an example receive the same noise in
       the second epoch as in the first? Explain the answer from when the
       loader draws its base seed and calls the initializer.

    *Adapted from the PyTorch documentation
    [Multi-process data loading](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading).*
3. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
4. [code] **Gradient norms.** ● Rebuild `net`, multiply the weight of
   every linear layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Record the norm of each block's output gradient as in this
       section. After the backward pass, compute each block's
       parameter-gradient norm, and plot both families against depth.
    1. Express the gradient of a linear layer's weight through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Residual variance budget.** Override `call` in
   `ResidualBlock` so that it stashes the
   variance of its input, of its body's output, and of its own output, and
   run a freshly built `Checked` model on one batch. Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote these three tensors for block
   $k$, with each variance taken over all entries. The standard deviations
   in the table of :numref:`subsec_activation_stats` grow because each
   block adds its body's output to the stream. Measure how closely

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
2. [code] **Seeds and data loading.** ● Work in a fresh process, since the
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
3. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   the `Checked` model and train it with SGD at learning rate 1 on the
   mean squared error between its output and standard normal targets,
   using one fixed batch of 256 standard normal inputs. Report the first
   step at which a layer's output is non-finite, and the first such layer
   by depth. Check the loss at every step as well, and report whether it
   becomes non-finite before any layer output does.
4. [code] **Gradient norms.** ● Rebuild `net`, multiply the kernel of
   every `Dense` layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Compute the norm of each block's output gradient as in this
       section, and each block's parameter-gradient norm from the
       gradients of the trainable variables. Plot both families of norms
       against depth.
    1. Express the gradient of a linear layer's kernel through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

:begin_tab:`jax`
1. [code] **Residual variance budget.** Let $\mathbf{x}_k$,
   $\mathbf{b}_k$ and $\mathbf{h}_k$ denote the input, body output and
   output of block $k$, with each variance taken over all entries. Collect
   every layer's output and every body's output from one call of
   `nnx.capture` with `method_outputs=nnx.Intermediate`. The input of
   block $k$ is the output of the layer before it. The standard deviations
   in the table of :numref:`subsec_activation_stats` grow because each
   block adds its body's output to the stream. Measure how closely

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   holds for every block, and name the term this equation omits.
2. [code] **Seeds and data loading.** ● Simulate a four-worker input
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
3. [code] **Divergence monitor.** ● Turn the finiteness check of
   :numref:`subsec_nan_finder` into a per-step training monitor. Rebuild
   `net` and train it with SGD at learning rate 1 on the mean squared
   error between its output and standard normal targets, using one fixed
   batch of 256 standard normal inputs. Report the first step at which a
   layer's output is non-finite, and the first such layer by depth. Check
   the loss at every step as well, and report whether it becomes
   non-finite before any layer output does.
4. [code] **Gradient norms.** ● Rebuild `net`, multiply the kernel of
   every linear layer inside its residual blocks by 3, and use the mean
   squared output on a fixed batch as the loss.
    1. Compute the norm of each block's output gradient as in this
       section, and each block's parameter-gradient norm from the tree
       that `nnx.grad` returns. Plot both families of norms against depth.
    1. Express the gradient of a linear layer's kernel through the layer's
       input and its output gradient. Use this expression to explain why
       the two depth profiles differ, and state which family to monitor for
       an exploding backward signal.
   :exlabel:`ex_reproducibility-inspection-gradient-norms`
:end_tab:

[Discussions](https://d2l.discourse.group/t/36759)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.8]{.kicker}

Reproducibility, Debugging, and Model Inspection<br>**PRNG seeds & keys · data-loader workers · non-associative GPU atomics · forward & backward hooks**
:::
:::

::: {.slide title="Why Reproducibility and Model Inspection Belong Together"}
[Scientific Validity vs. Bug Hunting]{.kicker}

Two everyday engineering questions depend on controlling randomness and observing internal layer state:

- **Is a 0.3-point accuracy gain real or random noise?** Five runs of the *exact same* Fashion-MNIST training script differing only in their random seed span $1\text{--}4$ percentage points due to random weight initialization and minibatch shuffling.
- **Why did the loss turn `NaN` at step 40,000?** Debugging a late-stage crash requires reproducing the exact trajectory up to the failure and inspecting every layer's activations and gradients without forking third-party model code.
:::

::: {.slide title="Where Randomness Enters a Deep Learning Pipeline"}
[Five Stochastic Entry Points]{.kicker}

Every stochastic choice in training comes from a deterministic **pseudorandom number generator (PRNG)** whose sequence is fixed by its starting **seed**:

| Pipeline Stage | What Random Numbers Control | Consequence if Left Unseeded or Shared |
|:---|:---|:---|
| **1. Weight initialization** | Starting values of every layer ($\mathbf{W}_0$) | Different initial basin and loss trajectory |
| **2. Stochastic layers** | Dropout binary masks, stochastic depth | Forward pass differs on identical inputs |
| **3. DataLoader shuffling** | Minibatch permutation at each epoch | Gradient updates arrive in a different order |
| **4. Data augmentation** | Random crops, horizontal flips, color jitter | Different training views generated per step |
| **5. Multi-process workers** | Forked OS processes running `__getitem__` | Cloned workers emit *identical* "random" noise! |
:::

::: {.slide title="Seeding Repeats a Training Trajectory Bit for Bit"}
[Global Seeds vs. Explicit Keys]{.kicker}

When we seed every PRNG the loop touches (`torch.manual_seed`, `mnp.random.seed`, `set_random_seed`, or explicit `jax.random.split` keys), two runs with `seed=0` produce identical losses and weights down to the last bit:

@!reproducibility-inspection-seeds-and-randomness

- **Coverage caveat:** `torch.manual_seed` and `mnp.random.seed` do **not** seed Python's `random` or NumPy's `np.random`—if your dataset uses NumPy, seed it explicitly too.
:::

::: {.slide title="Private Generator Objects Prevent Stream Coupling"}
[Isolating Stochastic Streams]{.kicker}

Relying on a single global PRNG couples every piece of code: adding one extra random call during debugging shifts the entire downstream sequence. A private **generator object** (`torch.Generator`, `tf.random.Generator`, or split JAX keys) isolates a stream behind its own state:

@reproducibility-inspection-generator-objects
:::

::: {.slide title="The Multi-Worker DataLoader Fork Bug" only="pytorch,mxnet"}
[Why Forked Workers Duplicate Random Draws]{.kicker}

When a `DataLoader` spawns `num_workers=2` processes via OS `fork`, each worker inherits an identical clone of any `np.random.default_rng()` attribute on the dataset:

@!reproducibility-inspection-dataloader-workers-2

- **Identical batches across workers (Row 1):** Without per-worker re-seeding, Worker 0 and Worker 1 generate the exact same "random" numbers (`[0.549, 0.715, 0.603, 0.545]`).
- **The fix (Row 2):** Re-seed each worker's generator inside `worker_init_fn` using `get_worker_info().seed` (or seed per example index `[base_seed, i]`).
:::

::: {.slide title="Deterministic Shuffling in Multi-Threaded Data Pipelines" only="tensorflow"}
[Seeding tf.data Pipelines]{.kicker}

In TensorFlow, `tf.data` runs parallel worker threads inside a single process and takes an explicit `seed=` on `.shuffle(buffer_size, seed=0)`:

@reproducibility-inspection-dataloader-workers-3

- **Epoch-by-epoch determinism:** Rebuilding the seeded pipeline replays the exact epoch-1 and epoch-2 shuffle orders without forked-process state duplication.
:::

::: {.slide title="Explicit PRNG Keys Eliminate Hidden Worker State" only="jax"}
[Randomness as a Pure Value]{.kicker}

Every worker bug in imperative frameworks comes from hidden mutable PRNG state being shared or cloned implicitly across processes:

- **Stateless functional keys:** In JAX, `jax.random.normal(key, shape)` is a pure deterministic function of `key`; calling it twice with the same key returns the exact same array.
- **Deriving independent worker streams:** Splitting a base key across workers via `jax.random.split(base_key, num_workers)` or folding in the epoch and example index via `jax.random.fold_in(base_key, idx)` makes accidental stream duplication impossible.
:::

::: {.slide title="Measuring Seed-to-Seed Variance Across Five Runs"}
[Statistical Reproducibility]{.kicker}

Training softmax regression on Fashion-MNIST for 3 epochs across five seeds (`0..4`) with identical code and hyperparameters reveals the natural variance of stochastic optimization:

@!reproducibility-inspection-seed-spread-1

@!reproducibility-inspection-seed-spread-2

- **Never trust a single-seed delta:** Final validation accuracies span $1\text{--}4$ percentage points across seeds; always report **mean $\pm$ standard deviation** across multiple seeds.
:::

::: {.slide title="Why Floating-Point Addition Is Not Associative"}
[The Arithmetic Wrinkle]{.kicker}

Fixing random seeds guarantees that two runs draw the same numbers, not that the hardware adds them in the same order. Because every float addition rounds to the nearest representable value, $(a + b) + c \neq a + (b + c)$:

@reproducibility-inspection-determinism-and-its-price-1

- **Swamping at $10^8$:** Near $10^8$, adjacent `float32` numbers sit $8.0$ apart, so $(-10^8 + 1.0)$ rounds to $-10^8$, turning $10^8 + (-10^8 + 1.0)$ into $0.0$ while $(10^8 - 10^8) + 1.0 = 1.0$.
:::

::: {.slide title="GPU Atomic Reductions and Deterministic Kernel Switches"}
[Why Seeded GPU Runs Diverge]{.kicker}

On a GPU, thousands of threads accumulate partial sums into shared bins (in `scatter_add`, `index_add_`, embedding backward passes, and convolutions) via **atomic additions** that land in whatever order threads finish:

@!reproducibility-inspection-determinism-and-its-price-2

- **Enabling bitwise GPU determinism:** Turning on deterministic kernels (`torch.use_deterministic_algorithms(True)`, `enable_op_determinism()`, `xla_gpu_deterministic_ops`, or fixed-order GEMM reductions) enforces a fixed reduction tree so consecutive GPU runs match bit for bit.
:::

::: {.slide title="Bitwise Determinism vs. Scientific Reproducibility"}
[Scope of Bitwise Repeatability]{.kicker}

Understanding what deterministic switches do—and do not—guarantee prevents wasted engineering effort:

- **Bitwise determinism is a local debugging tool:** It guarantees identical bits on **one machine, one GPU architecture, and one library version** so you can bisect the exact step where a bug or `NaN` appears.
- **What never matches bitwise:** Different GPU architectures, CPU vs. GPU execution, different tensor-parallel world sizes, or library version upgrades alter kernel reduction trees and last-bit rounding.
- **Scientific reproducibility is statistical:** Any real modeling improvement must hold across multiple random seeds ($N \ge 5$) and survive minor floating-point reassociation.
:::

::: {.slide title="Non-Invasive Model Inspection with Execution Hooks"}
[Observing Layers From Outside]{.kicker}

When debugging a deep network or a locked third-party library model, editing every layer's `forward` method to insert `print` statements is clumsy. **Hooks** wrap `__call__` around each submodule's execution from the outside:

![](../img/bg-hooks.svg){.nostretch width=85%}

- **Pipeline around `forward`:** Invoking `net(X)` runs registered pre-hooks, executes `forward(X)`, and passes `(module, inputs, output)` to registered forward hooks (`register_forward_hook` or `nnx.capture`).
:::

::: {.slide title="Building an 8-Block Residual Testbed for Inspection"}
[Unscaled Xavier Residual Stack]{.kicker}

To test our inspection tools, we construct an 8-block residual MLP ($20 \to 64 \to 8\text{ blocks of width } 64 \to 10$) initialized with Xavier uniform weights and zero biases:

@-reproducibility-inspection-hooks-looking-inside
:::

::: {.slide title="Profiling Activation Standard Deviations Across Depth"}
[Forward-Hook Diagnostic]{.kicker}

Attaching a forward hook to every top-level module captures the activation growth profile across all 10 layers in a single forward pass on 256 standard normal inputs:

::: {.cols .vc}
::: {.col}
@!reproducibility-inspection-capturing-activation-statistics
:::

::: {.col}
- **Steady growth across blocks:** The output standard deviation starts at $\approx 0.69$ after the $20 \to 64$ input layer and climbs steadily to $\approx 3.3\text{--}3.8$ after the eighth residual block.
- **Zero model edits:** The hook callback records `(type(module).__name__, output.detach().std().item())` and unregisters itself immediately afterward.
:::
:::
:::

::: {.slide title="Why Xavier Residual Blocks Grow by √1.5 ≈ 1.22 per Block"}
[Analytical Verification of Hook Measurements]{.kicker}

We can verify the hook profile directly from the variance rules of §5.3. Each residual block computes $\mathbf{h}_k = \mathbf{x}_k + \text{body}(\mathbf{x}_k)$ with Xavier-initialized linear layers ($\text{Var}(W_{ij}) = \frac{2}{d + d} = \frac{1}{d}$):

$$\text{Var}[\mathbf{h}_k] \approx \text{Var}[\mathbf{x}_k] + \text{Var}[\text{body}(\mathbf{x}_k)] = \text{Var}[\mathbf{x}_k] + \frac{1}{2}\text{Var}[\mathbf{x}_k] = 1.5\,\text{Var}[\mathbf{x}_k]$$

- **Why the body contributes $\frac{1}{2}\text{Var}[\mathbf{x}_k]$:** The first $64 \to 64$ Xavier linear layer preserves variance ($1\times$), the ReLU halves the mean square ($\frac{1}{2}\times$), and the second $64 \to 64$ Xavier layer preserves that half ($\frac{1}{2}\times$).
- **Exact match with hooks:** Starting from input-layer std $\sqrt{20 \times \frac{2}{20+64}} \approx 0.69$, multiplying by $\sqrt{1.5} \approx 1.225$ across 8 blocks predicts $0.69 \times 1.5^4 \approx 3.5$!
:::

::: {.slide title="Two Golden Rules for Leak-Free Hooks" only="pytorch,mxnet"}
[Preventing Autograd Memory Leaks]{.kicker}

Whenever you attach forward or backward hooks to a live model during training, follow two mandatory cleanup habits:

1. **Always `.detach()` stored tensors (or extract Python scalars via `.item()`):** Appending a live activation tensor `output` directly to a Python list retains a reference to the entire backward computation graph, leaking gigabytes of GPU memory across steps.
2. **Always remove hook handles when finished (`h.remove()` in PyTorch, `h.detach()` in Gluon):** Hooks persist on the module object across future forward passes until explicitly unregistered.
:::

::: {.slide title="Pinpointing the First NaN Layer in One Forward Pass"}
[Automated NaN Traps]{.kicker}

Because forward hooks fire in exact execution order, attaching a finiteness check (`isfinite(output).all()`) to every leaf layer immediately halts and names the first layer that emits `inf` or `NaN`:

@!reproducibility-inspection-a-nan-finder
:::

::: {.slide title="Monitoring Backward Gradient Norms Across Depth"}
[Backward Hooks & Gradient Flow]{.kicker}

Inspecting gradient norms at each residual block's output during backpropagation reveals the exact mirror image of forward activation growth:

@!reproducibility-inspection-backward-hooks-and-beyond

- **Backward amplification ($\times 1.2\text{--}1.4$ per block):** Because each residual block's backward pass adds the branch Jacobian back onto the skip connection, output gradient norms grow $5\text{--}6\times$ as they travel backward from block 8 to block 1.
:::

::: {.slide title="Concept Check: Determinism, Variance Growth & Hooks"}
[Concept Check]{.kicker}

Test your diagnostic skills on these three questions:

1. A PyTorch training script calls `torch.manual_seed(0)` at the top. Inside `Dataset.__getitem__`, it adds Gaussian noise using `self.rng = np.random.default_rng()` created in `Dataset.__init__`. Will two runs with `num_workers=0` match? What happens with `num_workers=4` under `fork`?
2. In our Xavier-initialized residual stack, activation standard deviation grows by $\sqrt{1.5} \approx 1.225$ per block. After how many blocks $k$ does the activation standard deviation exceed $100\times$ its post-input value?
3. What happens to GPU memory during training if a forward hook executes `history.append(output)` on every step without calling `.detach()`?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **With `num_workers=0`, two runs differ** because `np.random.default_rng()` draws fresh OS entropy ignoring `torch.manual_seed(0)`. **With `num_workers=4` (`fork`), all 4 workers inherit identical clones** of `self.rng` and apply identical noise!
2. **$1.5^{k/2} \ge 100 \implies k \ge \frac{2 \ln 100}{\ln 1.5} \approx 22.7$ (23 blocks).**
3. **Out-of-memory (OOM) crash:** Each `output` keeps its entire autograd graph and intermediate activations alive in `history` forever.
:::
:::

::: {.slide title="Summary: Reproducibility, Debugging, and Inspection"}
[Key Takeaways]{.kicker}

| Diagnostic Pillar | Mechanism / Tool | Practical Engineering Rule |
|:---|:---|:---|
| **PRNG seeding** | Global seeds + private `Generator` / split JAX keys | Seed framework, NumPy, Python `random`, and each DataLoader worker (`worker_init_fn`) |
| **Statistical validity** | Run $N \ge 5$ seeds; report mean $\pm$ std | A single-run gain smaller than seed-to-seed spread ($1\text{--}4\%$ on Fashion-MNIST) is noise |
| **Bitwise determinism** | `use_deterministic_algorithms(True)` / `enable_op_determinism` | Replaces non-associative parallel GPU atomic adds with fixed-order reductions for debugging |
| **Forward hooks** | `register_forward_hook` / `nnx.capture` / functional probes | Profile layer activation scales and trap the first `NaN` without editing model code |
| **Backward inspection** | `register_full_backward_hook` / intermediate tape grads | Track gradient norms across depth to diagnose vanishing or exploding backward signals |
:::
