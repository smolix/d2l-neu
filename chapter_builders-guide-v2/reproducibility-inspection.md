```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Reproducibility and Inspection
:label:`sec_repro_v2`

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
from flax import linen as nn
import optax
```

## Seeds and Randomness

Randomness enters a training run in more places than you might list on a
first try: initialization draws every weight (:numref:`sec_init_v2`),
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
    net = nn.Sequential([nn.Dense(32), nn.relu, nn.Dense(1)])
    X = jax.random.normal(key_X, (128, 20))
    y = jax.random.normal(key_y, (128, 1))
    params = net.init(key_init, X)
    opt = optax.sgd(learning_rate=0.1)
    opt_state = opt.init(params)
    loss_fn = lambda p: ((net.apply(p, X) - y) ** 2).mean()
    for _ in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    return loss, params['params']['layers_0']['kernel']

loss_a, w_a = train_once(seed=0)
loss_b, w_b = train_once(seed=0)
loss_c, _ = train_once(seed=1)
print('same seed, identical loss and weights:',
      loss_a == loss_b, jnp.array_equal(w_a, w_b))
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

:begin_tab:`pytorch`
Most sampling functions accept `generator=`, and so does `DataLoader`,
where it controls the shuffle order. We use that below.
:end_tab:

:begin_tab:`jax`
Every function in `jax.random` takes the key as its first argument; there
is no variant that consults hidden state.
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

## Determinism and Its Price

Seeding fixes which numbers the program draws. It does not fix how the
arithmetic evaluates. Floating-point addition is not associative
(:numref:`sec_numerics_v2`), so summing the same numbers in a different
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

Be honest about what this buys. No framework guarantees bitwise agreement
across releases, platforms, or CPU versus GPU: it holds only for a
pinned machine, library version, and flag configuration. That makes bitwise
reproducibility a *debugging* tool, the setting that lets you bisect
exactly where two runs diverge. The *scientific* goal is statistical
reproducibility: the same conclusions across seeds, reported as a mean and
spread over several runs rather than one fortunate curve. The distinction
mirrors :numref:`sec_numerics_v2`: changing dtype changes results in the
last bits by design, and an experimental claim that survives neither a new
seed nor bfloat16 was never a result.

## Hooks: Looking Inside

:begin_tab:`pytorch`
In :numref:`sec_model_construction_v2` we noted that `net(X)` does not call
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
A flax model's `apply` is a pure function from parameters and inputs to
outputs, which would seem to leave an observer nowhere to attach. But
Flax wraps every submodule's `__call__`, and `apply` exposes that
machinery directly: `net.apply(params, X, capture_intermediates=True)`
returns, alongside the output, a dictionary holding the return value of
every submodule, with no change to the model's source. That matters
precisely when the model came from a library or a checkpoint you do not
want to edit. :numref:`fig_bg_hooks` draws the general picture: an
observation point in the gap the call wrapper already leaves around each
module's computation, from which an observer can capture or check
without a single line of the model changing.
:end_tab:

![The `__call__` wrapper as a pipeline: input flows through pre-hooks, then forward, then hooks, to the output, with the two hook stages dashed and orange against forward's solid blue, and a side arrow from the hooks stage to an observer that can capture, check, or modify.](../img/bg-hooks.svg)
:label:`fig_bg_hooks`

We reuse the residual stack of
:numref:`sec_model_construction_v2`, rebuilt compactly:

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
class ResidualBlock(nn.Module):
    num_hiddens: int

    def setup(self):
        self.body = nn.Sequential([nn.Dense(self.num_hiddens), nn.relu,
                                   nn.Dense(self.num_hiddens)])

    def __call__(self, X):
        return X + self.body(X)

net = nn.Sequential([nn.Dense(64),
                     *[ResidualBlock(64) for _ in range(8)],
                     nn.Dense(10)])
X = jax.random.normal(jax.random.key(1), (256, 20))
params = net.init(jax.random.key(0), X)
```

### Capturing Activation Statistics

The initialization experiments of :numref:`sec_init_v2` measured the
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
_, mods = net.apply(params, X, capture_intermediates=True)
inter = mods['intermediates']
for k, layer in enumerate(net.layers):
    out = inter[f'layers_{k}']['__call__'][0]
    print(f'{type(layer).__name__:15s} std {out.std():.2f}')
```

The residual stream's spread grows block by block, since each block adds
its body's output on top of the stream, exactly the depth effect that
motivated the scaled initializations of :numref:`sec_init_v2`, measured
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
sabotaged = jax.tree_util.tree_map(lambda x: x, params)  # copy the tree
kernel = sabotaged['params']['layers_3']['body']['layers_0']['kernel']
sabotaged['params']['layers_3']['body']['layers_0']['kernel'] = (
    kernel.at[0, 0].set(float('nan')))  # sabotage one layer
_, mods = net.apply(sabotaged, X,
                    capture_intermediates=lambda m, _: isinstance(m, nn.Dense))
for path, out in jax.tree_util.tree_flatten_with_path(mods['intermediates'])[0]:
    if not jnp.isfinite(out).all():
        print('first non-finite output in', jax.tree_util.keystr(path[:-2]))
        break
```

:begin_tab:`pytorch`
The report names module `3.body.0`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream.
:end_tab:

:begin_tab:`jax`
Filtering the capture to `nn.Dense` records only leaf layers, so the
flattened dictionary lists their outputs in execution order and the first
non-finite entry is the culprit. The report names
`['layers_3']['body']['layers_0']`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream.
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
inside a backward pass: `jax.grad` returns them as a tree of values in
the same shape as the parameters. To log per-layer gradient norms, catch
exploding gradients at their source, or experiment with per-layer
clipping, compute the gradient tree and inspect or transform it like any
other data:
:end_tab:

```{.python .input #reproducibility-inspection-backward-hooks-and-beyond}
%%tab jax
grads = jax.grad(lambda p: (net.apply(p, X) ** 2).mean())(params)
norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
print(norms['params']['layers_3']['body']['layers_0'])
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

Bitwise agreement is a debugging tool;
conclusions that hold across seeds are the scientific goal.

:begin_tab:`pytorch`
For looking
inside a model, forward and backward hooks observe any module without
editing it, subject to two rules: detach what you stash, and remove the
handle.
:end_tab:

:begin_tab:`jax`
For looking inside a model, `capture_intermediates=True` returns every
submodule's output from an unmodified `apply`, `sow` records named values
from the inside, and gradients are values from `jax.grad` you inspect
directly.
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
   `capture_intermediates` filter that records only `ResidualBlock`
   outputs, and by editing the block to call
   `self.sow('intermediates', 'body_out', ...)` on its body's output
   (`flax.linen.intercept_methods` is a third route worth reading about).
   Compare the three contracts you now know, capture-everything, opt-in
   `sow`, and PyTorch-style mutable hooks: which requires touching model
   code, which can silently retain memory, and which would you want for a
   model you do not own?
:end_tab:
