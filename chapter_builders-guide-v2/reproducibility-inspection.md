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

In PyTorch, `torch.manual_seed(k)` seeds the global generator on the CPU
*and* on every CUDA device. It does not touch Python's `random` module or
NumPy, which keep their own global generators (`random.seed`,
`np.random.seed`), nor NumPy `Generator` objects created by
`np.random.default_rng`, which carry their own seeds. The function below
draws everything, weights, data, and shuffling, from torch's global
generator, so one call pins the whole run:

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

The two seeded runs agree *bitwise*, down to the last floating-point bit:
same initial weights, same data, same gradients, same operations in the
same order.

### Generator Objects

A single global stream is fragile: every consumer shares it, so inserting
one extra random call (a new layer's initialization, a stray `torch.randn`
while debugging) shifts everything drawn after it. A `torch.Generator` is a
private stream with its own seed. Here a data split stays fixed no matter
what else consumes randomness in between:

```{.python .input #reproducibility-inspection-generator-objects}
%%tab pytorch
g = torch.Generator().manual_seed(42)
split = torch.randperm(10, generator=g)
_ = torch.randn(1000)  # unrelated consumption of the global stream
split_again = torch.randperm(10, generator=g.manual_seed(42))
print(torch.equal(split, split_again), split)
```

Most sampling functions accept `generator=`, and so does `DataLoader`,
where it controls the shuffle order. We use that below.

### DataLoader Workers

Now for the classic hole: you seed torch, NumPy, and `random`, and the run
is *still* not reproducible, because augmentation code in
`Dataset.__getitem__` calls `np.random` inside loader worker processes. On
Linux, worker processes start via `fork`, which gives each child a
byte-for-byte copy of the parent, including NumPy's global generator state.
We can simulate what every worker inherits:

```{.python .input #reproducibility-inspection-dataloader-workers-1}
%%tab pytorch
np.random.seed(0)                   # the parent process, dutifully seeded
state = np.random.get_state()       # fork copies this state into each child
for worker_id in range(4):
    np.random.set_state(state)      # what a fork-started worker begins with
    print(f'worker {worker_id}:', np.random.randint(0, 1000, size=3))
```

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

Resetting the generator reproduces the exact shuffle order, and with
workers enabled, `seed_worker` gives each worker its own NumPy and `random`
streams that still derive from the one base seed.

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

Be honest about what this buys. PyTorch guarantees none of it across
releases, platforms, or CPU versus GPU: bitwise agreement holds only for a
pinned machine, library version, and flag configuration. That makes bitwise
reproducibility a *debugging* tool, the setting that lets you bisect
exactly where two runs diverge. The *scientific* goal is statistical
reproducibility: the same conclusions across seeds, reported as a mean and
spread over several runs rather than one fortunate curve. The distinction
mirrors :numref:`sec_numerics_v2`: changing dtype changes results in the
last bits by design, and an experimental claim that survives neither a new
seed nor bfloat16 was never a result.

## Hooks: Looking Inside

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

### Capturing Activation Statistics

The initialization experiments of :numref:`sec_init_v2` measured the
standard deviation of activations at every depth. With hooks, the same
measurement takes a few lines on an unmodified model:

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

The residual stream's spread grows block by block, since each block adds
its body's output on top of the stream, exactly the depth effect that
motivated the scaled initializations of :numref:`sec_init_v2`, measured
here without touching the model.

Two rules keep hooks safe. First, *detach before you stash*: the hook above
stores `output.detach().std()`; storing `output` itself would keep the
autograd graph of every forward pass alive until `stats` is cleared, a
memory leak that grows with each batch. Second, *keep the handle and call*
`handle.remove()`: a hook stays registered for the module's lifetime
otherwise, taxing every later forward pass and accumulating stashed
tensors.

### A NaN Finder

When a loss becomes NaN at step 40,000, the useful question is which layer
produced the first non-finite value. Hook every leaf module with a
finiteness check and the answer arrives in one forward pass. We sabotage a
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

The report names module `3.body.0`, the layer we poisoned, rather than
leaving you to bisect with print statements while NaNs propagate through
everything downstream.

### Backward Hooks and Beyond

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

## Summary

Reproducibility is an inventory problem: initialization, dropout,
shuffling, augmentation, and loader workers each draw from some generator,
and a run repeats only if every one of them is seeded.
`torch.manual_seed` covers torch's CPU and CUDA generators; NumPy and
`random` need their own seeds, and loader workers need them *per worker*
via `worker_init_fn` plus a `generator` for the shuffle order. Seeding
makes the program repeatable, not the arithmetic:
`torch.use_deterministic_algorithms(True)` pins kernel choice too, raising
on operations that cannot comply. Bitwise agreement is a debugging tool;
conclusions that hold across seeds are the scientific goal. For looking
inside a model, forward and backward hooks observe any module without
editing it, subject to two rules: detach what you stash, and remove the
handle.

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
