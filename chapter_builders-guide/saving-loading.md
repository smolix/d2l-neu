```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Saving, Loading, and Pretrained Weights
:label:`sec_read_write`

A trained network is two separate things kept in two separate places. The
*code* is the class you wrote: its layers, its `forward` pass, the config that
sized it. The *state* is the collection of tensors that training filled in: the
weights and biases, the running statistics of normalization layers, the
optimizer's momentum. When you save a model you save only the state. The code
stays in your source repository, under version control, exactly like any other
Python. To bring a model back you need both halves: run the code to rebuild an
empty network, then pour the saved state into it.

This split explains most of what follows. It is why a checkpoint cannot
resurrect a model on its own, why the config object from
:numref:`sec_model_construction` belongs *inside* the checkpoint, and why the
format that stores the state matters once you start sharing files with people who
do not have your code.

```{.python .input #saving-loading-saving-loading-and-pretrained-weights}
%%tab pytorch
import json
import os
import struct
from collections import Counter
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from safetensors.torch import load_file, save_file
from d2l import torch as d2l
```

```{.python .input #saving-loading-saving-loading-and-pretrained-weights}
%%tab jax
import json
import os
import struct
from dataclasses import asdict, dataclass
import jax
from jax import numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from safetensors.flax import load_file, save_file
from d2l import jax as d2l
```

```{.python .input #saving-loading-saving-loading-and-pretrained-weights}
%%tab tensorflow
import json
import struct
import warnings
from dataclasses import asdict, dataclass
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
from safetensors.tensorflow import load_file, save_file
```

```{.python .input #saving-loading-saving-loading-and-pretrained-weights}
%%tab mxnet
import json
import os
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
from safetensors.numpy import load_file, save_file
npx.set_np()
```

## State, Not Code

:begin_tab:`pytorch`
The state of a network is a dictionary from parameter names to tensors, the
`state_dict` of :numref:`sec_parameters`. Before we save a whole model, the
warm-up is that the same `save`/`load` calls work on any tensors, and on the
lists and dicts that hold them.
:end_tab:

:begin_tab:`jax`
The state of a network is a tree of typed, named variables introduced in
:numref:`sec_parameters`. Before we save a whole model, the warm-up is that
`jnp.save` and `jnp.load` work on any array, and, through NumPy's pickle
fallback, on the dicts that hold them.
:end_tab:

:begin_tab:`tensorflow`
The state of a network is a collection of named variables, the weights of
:numref:`sec_parameters`. Before we save a whole model, the warm-up is that
`np.save` and `np.load` work on any tensor, and, through NumPy's pickle
fallback, on the dicts that hold them.
:end_tab:

:begin_tab:`mxnet`
The state of a network is a dictionary from parameter names to arrays, the
`collect_params` dictionary of :numref:`sec_parameters`. Before we save a
whole model, the warm-up is that `npx.save` and `npx.savez` work on any array
and on named collections of them; `npx.load` hands a saved collection back as
a dict.
:end_tab:

```{.python .input #saving-loading-state-not-code-1}
%%tab pytorch
x = torch.arange(4)
torch.save({'x': x, 'y': torch.zeros(4)}, 'tensors.pt')
torch.load('tensors.pt', weights_only=True)
```

```{.python .input #saving-loading-state-not-code-1}
%%tab jax
x = jnp.arange(4)
jnp.save('tensors.npy', {'x': x, 'y': jnp.zeros(4)})
jnp.load('tensors.npy', allow_pickle=True).item()
```

```{.python .input #saving-loading-state-not-code-1}
%%tab tensorflow
x = tf.range(4)
np.save('tensors.npy', {'x': x, 'y': tf.zeros(4)})
np.load('tensors.npy', allow_pickle=True).item()
```

```{.python .input #saving-loading-state-not-code-1}
%%tab mxnet
x = np.arange(4)
npx.savez('tensors-mx', x=x, y=np.zeros(4))
npx.load('tensors-mx')
```

:begin_tab:`pytorch`
A model's `state_dict` is one such dictionary, built for you. The keys are the
dotted paths through the module tree (`hidden.weight`, `output.bias`); the values
are the tensors, buffers included. Here is the tree for a small MLP.
:end_tab:

:begin_tab:`jax`
A model's parameter state is one such structure. Its paths follow the module
graph (`hidden.kernel`, `output.bias`; Flax calls a weight matrix a `kernel`).
Here is the tree for a small MLP.
:end_tab:

:begin_tab:`tensorflow`
A model's state is one such collection, built for you: `net.weights` is the
list of its variables, and each variable's `path` names its place in the layer
tree (`mlp/hidden/kernel`, `mlp/output/bias`; Keras calls a weight matrix a
`kernel`). Keras invents layer names like `dense_3` when you do not choose
them, so we name the layers explicitly to keep the paths stable across
instances. Here is the tree for a small MLP.
:end_tab:

:begin_tab:`mxnet`
A model's state is one such dictionary, built for you: `collect_params` maps
the dotted paths through the block tree (`hidden.weight`, `output.bias`) to
the parameters, the running statistics of any BatchNorm layers included. Here
is the tree for a small MLP.
:end_tab:

```{.python .input #saving-loading-state-not-code-2}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(2, 20)
Y = net(X)
{name: tuple(t.shape) for name, t in net.state_dict().items()}
```

```{.python .input #saving-loading-state-not-code-2}
%%tab jax
class MLP(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.hidden = nnx.Linear(20, 256, rngs=rngs)
        self.output = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        return self.output(nnx.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(d2l.get_key(), (2, 20))
Y = net(X)
params = nnx.state(net, nnx.Param)
[(path, tuple(value.shape)) for path, value in params.flat_state()]
```

```{.python .input #saving-loading-state-not-code-2}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__(name='mlp')
        self.hidden = tf.keras.layers.Dense(256, name='hidden')
        self.out = tf.keras.layers.Dense(10, name='output')

    def call(self, x):
        return self.out(tf.nn.relu(self.hidden(x)))

net = MLP()
X = tf.random.normal((2, 20))
Y = net(X)
{v.path: tuple(v.shape) for v in net.weights}
```

```{.python .input #saving-loading-state-not-code-2}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
{name: p.shape for name, p in net.collect_params().items()}
```

Nothing in this dictionary knows it came from a class called `MLP`. That is the
point: the names and shapes are enough to refill any network built by the same
code, and they carry no dependence on how that code happens to be written today.

## safetensors: the Interchange Format

:begin_tab:`pytorch`
`torch.save` writes its files with Python's `pickle`, which does not store data
so much as a program that *reconstructs* data. Unpickling runs that program. For
a file you wrote and never let out of your control this is harmless. For a file
you downloaded it is a remote-code-execution vector: loading the weights can run
whatever the author's pickle stream tells it to.
:end_tab:

:begin_tab:`jax`
`jnp.save` writes NumPy's `.npy` format. For a single array that is pure data:
a small header with the dtype and shape, then the raw bytes. The warm-up dict is
another matter. NumPy can only store a dict by falling back to Python's
`pickle`, which does not store data so much as a program that *reconstructs*
data, and loading runs that program; that is what `allow_pickle=True` opted
into. For a file you wrote and never let out of your control this is harmless.
For a file you downloaded it is a remote-code-execution vector: any object in
the stream can name a callable for the loader to invoke, which is why NumPy
refuses pickled contents by default (`allow_pickle=False`).
:end_tab:

:begin_tab:`tensorflow`
`np.save` writes NumPy's `.npy` format. For a single tensor that is pure data:
a small header with the dtype and shape, then the raw bytes. The warm-up dict
is another matter. NumPy can only store a dict by falling back to Python's
`pickle`, which does not store data so much as a program that *reconstructs*
data, and loading runs that program; that is what `allow_pickle=True` opted
into. For a file you wrote and never let out of your control this is harmless.
For a file you downloaded it is a remote-code-execution vector: any object in
the stream can name a callable for the loader to invoke, which is why NumPy
refuses pickled contents by default (`allow_pickle=False`).
:end_tab:

:begin_tab:`mxnet`
`npx.savez` writes MXNet's own array format: the saver accepts nothing but
MXNet arrays, and the file holds only their names, dtypes, shapes, and raw
bytes. There is no pickle stream and nothing to execute on load, so the
warm-up file was already safe. The catch is reach rather than safety: hardly
anything outside MXNet reads such a file, which matters once you hand weights
to someone who does not run an archived framework.
:end_tab:

:begin_tab:`pytorch`
The risk is easy to make concrete. An object's `__reduce__` method returns the
callable and arguments that pickle will invoke on load. Point it at any function
and that function runs when the file is read.
:end_tab:

```{.python .input #saving-loading-safetensors-the-interchange-format-1}
%%tab pytorch
class Tripwire:
    def __reduce__(self):
        return (print, ('*** payload executed while loading ***',))

torch.save(Tripwire(), 'tripwire.pt')
_ = torch.load('tripwire.pt', weights_only=False)  # the payload runs here
```

:begin_tab:`pytorch`
Since version 2.6, PyTorch defaults `torch.load` to `weights_only=True`, which
refuses any pickle opcode that is not a plain tensor. The same file is now
rejected instead of executed.
:end_tab:

```{.python .input #saving-loading-safetensors-the-interchange-format-2}
%%tab pytorch
try:
    torch.load('tripwire.pt', weights_only=True)  # the default in this torch
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
```

:begin_tab:`pytorch`
The allowlist behind `weights_only=True` is defense in depth, not a sandbox: it
has itself had bypasses patched.
:end_tab:

:begin_tab:`jax`
`allow_pickle=False` is a refusal, not a fix: it keeps the loader safe by
declining to load the very files, dicts of named parameters, that model sharing
needs.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow's own model files sidestep pickle: neither a TF checkpoint nor the
Keras `.weights.h5`/`.keras` formats execute it on load. They are not
automatically safe to download, though. A `.keras` archive can carry serialized
code (a `Lambda` layer's function), which is why Keras refuses to deserialize
such code unless you pass `safe_mode=False`, and why a model file from an
untrusted source deserves the same caution as a pickle.
:end_tab:

safetensors removes the problem at the root by
having no program to run. As :numref:`fig_bg_safetensors_layout` lays out byte
by byte, a safetensors file is an 8-byte little-endian integer giving the
header length, then that many bytes of JSON naming each tensor's dtype, shape,
and byte range, then the raw tensor bytes back to back. There is no opcode
stream to interpret, so loading one cannot execute anything. It is also
framework-neutral and memory-mappable, which is why model hubs default to it.
Save and reload the MLP's state through it and confirm the round trip is exact.

![The safetensors file as one horizontal byte strip: an 8-byte header length, a JSON header naming each tensor's dtype, shape, and byte offsets, and the raw tensor bytes packed back to back with no gaps, with two of the file's own data_offsets entries traced down to their exact span in the bar.](../img/bg-safetensors-layout.svg)
:label:`fig_bg_safetensors_layout`

:begin_tab:`jax`
One mismatch to bridge first: safetensors stores a flat mapping from names to
tensors, while NNX state forms a nested pytree. Two small helpers convert
between the two, joining each leaf's path with dots on the way out and
splitting it again on the way back.
:end_tab:

:begin_tab:`tensorflow`
Here the variable paths do the naming, so the flat mapping is a one-line
comprehension over `net.weights`. Two quirks of the `tensorflow` binding to
know: it converts through NumPy in both directions, so `load_file` hands back
constant tensors that you `assign` into a model's variables, and `save_file`
overwrites the values of the dict you pass it during that conversion, so give
it a throwaway copy.
:end_tab:

:begin_tab:`mxnet`
There is no `safetensors.mxnet` module; the archived framework never grew a
native binding. That turns out to be the instructive case: because the format
is just names, dtypes, and bytes, the `safetensors.numpy` binding plus a pair
of dict comprehensions closes the gap, `.asnumpy()` on the way out and
`np.array` on the way back. No framework-specific machinery is involved,
which is the format's framework-neutrality made visible.
:end_tab:

```{.python .input #saving-loading-safetensors-the-interchange-format-3}
%%tab pytorch
save_file(net.state_dict(), 'mlp.safetensors')
clone = MLP()
clone(X)                                   # materialize the lazy layers first
clone.load_state_dict(load_file('mlp.safetensors'))
clone.eval()
torch.equal(clone(X), Y)
```

```{.python .input #saving-loading-safetensors-the-interchange-format-3}
%%tab jax
def flatten(tree):
    if hasattr(tree, 'flat_state'):
        leaves = tree.flat_state()
    else:
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        leaves = [(tuple(getattr(k, 'key', getattr(k, 'idx', k))
                         for k in path), value)
                  for path, value in leaves]
    return {'.'.join(map(str, path)): value for path, value in leaves}

def unflatten(flat):
    tree = {}
    for name, value in flat.items():
        *parents, leaf = name.split('.')
        node = tree
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return tree

save_file(flatten(params), 'mlp-jax.safetensors')
restored = unflatten(load_file('mlp-jax.safetensors'))
clone = MLP(nnx.Rngs(1))
clone_state = nnx.state(clone, nnx.Param)
nnx.replace_by_pure_dict(clone_state, restored)
nnx.update(clone, clone_state)
exact = jax.tree_util.tree_all(jax.tree_util.tree_map(
    jnp.array_equal, restored, nnx.to_pure_dict(params)))
exact, jnp.array_equal(clone(X), Y)
```

```{.python .input #saving-loading-safetensors-the-interchange-format-3}
%%tab tensorflow
state = {v.path: v for v in net.weights}
save_file(dict(state), 'mlp-tf.safetensors')  # dict(): save_file clobbers its arg
restored = load_file('mlp-tf.safetensors')
clone = MLP()
clone(X)                                   # build the variables first
for v in clone.weights:
    v.assign(restored[v.path])
tf.reduce_all(clone(X) == Y)
```

```{.python .input #saving-loading-safetensors-the-interchange-format-3}
%%tab mxnet
state = {name: p.data().asnumpy() for name, p in net.collect_params().items()}
save_file(state, 'mlp-mx.safetensors')
restored = load_file('mlp-mx.safetensors')
clone = MLP()
clone.load_dict({name: np.array(v) for name, v in restored.items()})
(clone(X) == Y).all()
```

Because the header is plain JSON at a known offset, you can read it without the
library and see there is no magic to the format.

```{.python .input #saving-loading-safetensors-the-interchange-format-4}
%%tab pytorch
with open('mlp.safetensors', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]   # header length, little-endian
    header = json.loads(f.read(n))
header['hidden.weight']
```

```{.python .input #saving-loading-safetensors-the-interchange-format-4}
%%tab jax
with open('mlp-jax.safetensors', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]   # header length, little-endian
    header = json.loads(f.read(n))
header['hidden.kernel']
```

```{.python .input #saving-loading-safetensors-the-interchange-format-4}
%%tab tensorflow
with open('mlp-tf.safetensors', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]   # header length, little-endian
    header = json.loads(f.read(n))
header['mlp/hidden/kernel']
```

```{.python .input #saving-loading-safetensors-the-interchange-format-4}
%%tab mxnet
with open('mlp-mx.safetensors', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]   # header length, little-endian
    header = json.loads(f.read(n))
header['hidden.weight']
```

:begin_tab:`pytorch`
`torch.save` keeps its place for your own scratch files and for the older code
you will still meet. safetensors is what you use to hand a model to anyone else.
:end_tab:

:begin_tab:`jax`
`jnp.save` keeps its place for your own scratch arrays and quick experiments.
safetensors is what you use to hand a model to anyone else.
:end_tab:

:begin_tab:`tensorflow`
`np.save` and Keras's own `.weights.h5` keep their place for your own scratch
files and checkpoints. safetensors is what you use to hand a model to anyone
else.
:end_tab:

:begin_tab:`mxnet`
`save_parameters` keeps its place for your own scratch files and checkpoints.
safetensors is what you use to hand a model to anyone else, and for an
archived framework it is also the exit route: the file you just wrote loads
into any of the other three frameworks in this book.
:end_tab:

## Checkpointing a Training Run

A checkpoint you can resume from holds more than weights. Resuming means picking
up the optimizer where it stopped, and Adam's state is the running first and
second moments of the gradients from :numref:`sec_parameters`. Drop them and
the optimizer restarts its momentum from zero, so the first steps after a resume
no longer behave like a continuation. A full checkpoint therefore bundles the
model state, the optimizer state, the RNG state (so data shuffling and dropout
continue the same stream), the step counter, and the config that sizes the model
when you rebuild it. :numref:`fig_bg_checkpoint_contents` pairs each of those
five compartments with the exact thing it restores on resume.

![A checkpoint file's five compartments, each paired by an arrow with what it restores on resume: model state_dict with weights, optimizer state with momentum and second moments, RNG state with data order and dropout, step with schedule position, and config with architecture.](../img/bg-checkpoint-contents.svg)
:label:`fig_bg_checkpoint_contents`

:begin_tab:`pytorch`
Two details separate a checkpoint from a corrupted file. First, keep the contents
to tensors and primitives so the file loads under `weights_only=True`; a
dataclass config goes in as a plain dict via `asdict`. Second, write atomically:
save to a temporary path and `os.replace` it into place, so a crash mid-write
leaves the previous good checkpoint untouched rather than a half-written one.
:end_tab:

:begin_tab:`jax`
NNX exposes the model and optimizer state as pytrees, including the optimizer's
step counter. Orbax, the JAX checkpointing library, saves and restores such trees whole: its
`StandardCheckpointer` writes atomically by default, to a temporary directory
renamed into place on success, so a crash mid-write leaves the previous good
checkpoint untouched rather than a half-written one. Because these natives
already cover the job, the jax tab defines no helper of its own; the calls below
are the idiom as you would write it in any project. The config rides along as
one more branch of the saved tree, and the PRNG key, which in JAX is explicit
data rather than hidden global state, can too.
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow the bundle already has a name. `tf.train.Checkpoint` takes the
objects to track as keyword arguments, model, optimizer, and a step counter,
walks their variables, and saves and restores them as one unit, Adam's moment
estimates included. Because this native already covers the job, the tensorflow
tab defines no helper of its own; the calls below are the idiom as you would
write it in any project. Saves are numbered (`run-tf-1`, `run-tf-2`, ...), so a
crash mid-write leaves the previous good checkpoint untouched rather than a
half-written one. The config is plain Python rather than variables, so it
travels in a JSON sidecar next to the checkpoint files.
:end_tab:

:begin_tab:`mxnet`
In MXNet the bundle is three files rather than one. `save_parameters` covers
the model; the optimizer state lives with the `gluon.Trainer`, whose
`save_states` writes Adam's moments (though not per-parameter attributes such
as `lr_mult`, which come from your code on rebuild), and writes them with
pickle, so a trainer-states file deserves the same your-own-disk-only caution
as any pickle. The step counter and config travel in a JSON sidecar. One
compartment of :numref:`fig_bg_checkpoint_contents` stays empty: MXNet has no
API to snapshot its random-number generators, only `npx.random.seed` to
restart them, so a resumed run reseeds rather than continuing the old stream.
We wrap the three writes in a helper that, as in the PyTorch tab, renames
each file into place so a crash mid-write leaves the previous good checkpoint
untouched rather than a half-written one.
:end_tab:

```{.python .input #saving-loading-checkpointing-a-training-run-1}
%%tab pytorch
def save_checkpoint(path, model, optimizer, step, cfg=None):  #@save
    """Atomically write a resumable training checkpoint."""
    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'cpu_rng': torch.get_rng_state()}
    if cfg is not None:
        ckpt['cfg'] = asdict(cfg)
    if torch.cuda.is_available():
        ckpt['cuda_rng'] = torch.cuda.get_rng_state_all()
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer=None):  #@save
    """Restore the state written by save_checkpoint; return the raw dict."""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    torch.set_rng_state(ckpt['cpu_rng'])
    if torch.cuda.is_available() and 'cuda_rng' in ckpt:
        torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return ckpt
```

```{.python .input #saving-loading-checkpointing-a-training-run-1}
%%tab mxnet
def save_checkpoint(prefix, model, trainer, step, cfg=None):  #@save
    """Atomically write a resumable checkpoint (.params/.states/.json)."""
    meta = {'step': step}
    if cfg is not None:
        meta['cfg'] = asdict(cfg)
    model.save_parameters(prefix + '.params.tmp')
    trainer.save_states(prefix + '.states.tmp')
    with open(prefix + '.json.tmp', 'w') as f:
        json.dump(meta, f)
    for ext in ('.params', '.states', '.json'):
        os.replace(prefix + ext + '.tmp', prefix + ext)

def load_checkpoint(prefix, model, trainer=None):  #@save
    """Restore the state written by save_checkpoint; return the metadata."""
    model.load_parameters(prefix + '.params')
    if trainer is not None:
        trainer.load_states(prefix + '.states')
    with open(prefix + '.json') as f:
        return json.load(f)
```

Train a tiny regressor for a hundred steps and checkpoint it. The `Config`
dataclass is what a rebuild reads to size the model, so it travels with the
weights.

```{.python .input #saving-loading-checkpointing-a-training-run-2}
%%tab pytorch
@dataclass
class Config:
    in_dim: int = 20
    hidden: int = 64
    lr: float = 0.05

def build(cfg):
    return nn.Sequential(nn.Linear(cfg.in_dim, cfg.hidden), nn.ReLU(),
                         nn.Linear(cfg.hidden, 1))

torch.manual_seed(1)
cfg = Config()
data = torch.randn(256, cfg.in_dim)
target = data @ torch.randn(cfg.in_dim, 1) + 0.1 * torch.randn(256, 1)
loss = nn.MSELoss()

def step(model, opt):
    opt.zero_grad()
    l = loss(model(data), target)
    l.backward()
    opt.step()
    return l.item()

net = build(cfg)
opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
for _ in range(100):
    step(net, opt)
save_checkpoint('run.pt', net, opt, step=100, cfg=cfg)
round(loss(net(data), target).item(), 4)
```

```{.python .input #saving-loading-checkpointing-a-training-run-2}
%%tab jax
@dataclass
class Config:
    in_dim: int = 20
    hidden: int = 64
    lr: float = 0.05

def build(cfg):
    return nnx.Sequential(
        nnx.Linear(cfg.in_dim, cfg.hidden, rngs=nnx.Rngs(0)), nnx.relu,
        nnx.Linear(cfg.hidden, 1, rngs=nnx.Rngs(1)))

def fresh_state(cfg):
    model = build(cfg)
    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr), wrt=nnx.Param)
    return model, optimizer

cfg = Config()
model = build(cfg)
data = jax.random.normal(d2l.get_key(), (256, cfg.in_dim))
target = (data @ jax.random.normal(d2l.get_key(), (cfg.in_dim, 1))
          + 0.1 * jax.random.normal(d2l.get_key(), (256, 1)))

def loss(model):
    return jnp.mean((model(data) - target) ** 2)

@nnx.jit
def step(model, optimizer):
    l, grads = nnx.value_and_grad(loss)(model)
    optimizer.update(model, grads)
    return l

model, optimizer = fresh_state(cfg)
for _ in range(100):
    l = step(model, optimizer)

ckptr = ocp.StandardCheckpointer()
ckptr.save(os.path.abspath('run-jax'),        # orbax wants an absolute path
           {'model': nnx.to_pure_dict(nnx.state(model)),
            'optimizer': nnx.to_pure_dict(nnx.state(optimizer)),
            'cfg': asdict(cfg)}, force=True)
int(optimizer.step), round(float(loss(model)), 4)
```

```{.python .input #saving-loading-checkpointing-a-training-run-2}
%%tab tensorflow
@dataclass
class Config:
    in_dim: int = 20
    hidden: int = 64
    lr: float = 0.05

def build(cfg):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(cfg.hidden, activation='relu'),
        tf.keras.layers.Dense(1)])

tf.keras.utils.set_random_seed(1)
cfg = Config()
data = tf.random.normal((256, cfg.in_dim))
target = data @ tf.random.normal((cfg.in_dim, 1)) + 0.1 * tf.random.normal((256, 1))
loss = tf.keras.losses.MeanSquaredError()

def step(model, opt):
    with tf.GradientTape() as tape:
        l = loss(target, model(data))
    opt.apply_gradients(zip(tape.gradient(l, model.trainable_variables),
                            model.trainable_variables))
    return float(l)

net = build(cfg)
net(data[:1])                                # build the variables
opt = tf.keras.optimizers.Adam(cfg.lr)
for _ in range(100):
    step(net, opt)

ckpt = tf.train.Checkpoint(model=net, optimizer=opt, step=tf.Variable(100))
path = ckpt.save('run-tf')                   # a numbered save: 'run-tf-1'
with open('run-tf-cfg.json', 'w') as f:
    json.dump(asdict(cfg), f)
path, round(float(loss(target, net(data))), 4)
```

```{.python .input #saving-loading-checkpointing-a-training-run-2}
%%tab mxnet
@dataclass
class Config:
    in_dim: int = 20
    hidden: int = 64
    lr: float = 0.05

def build(cfg):
    net = nn.Sequential()
    net.add(nn.Dense(cfg.hidden, activation='relu'), nn.Dense(1))
    net.initialize()
    return net

npx.random.seed(1)
cfg = Config()
data = np.random.normal(size=(256, cfg.in_dim))
target = data @ np.random.normal(size=(cfg.in_dim, 1)) \
    + 0.1 * np.random.normal(size=(256, 1))

def loss(model):
    return float(((model(data) - target) ** 2).mean())

def step(model, trainer):
    with autograd.record():
        l = ((model(data) - target) ** 2).mean()
    l.backward()
    trainer.step(batch_size=1)      # the loss is already a mean
    return float(l)

net = build(cfg)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': cfg.lr})
for _ in range(100):
    step(net, trainer)
save_checkpoint('run-mx', net, trainer, step=100, cfg=cfg)
round(loss(net), 4)
```

The restore is exact. Corrupt every parameter, load the checkpoint back, and the
loss returns to where it was.

:begin_tab:`jax`
Orbax fills a template with the saved arrays. We then update freshly built NNX
objects from the restored pure dictionaries. Rebuilding first checks that the
architecture still agrees with the checkpoint structure.
:end_tab:

:begin_tab:`tensorflow`
Note the shape of the restore call: `restore` patches the tracked objects in
place, matching each variable by its route through the object graph (`model`,
then the layer, then its `kernel`) rather than by name, and it returns a status
object. `assert_consumed()` on that status checks that every saved value found
a variable and every variable found a value, turning a silent partial restore
into a loud error.
:end_tab:

:begin_tab:`mxnet`
Note the shape of the restore: `load_parameters` and `load_states` patch the
model and trainer in place, matching parameters by the same dotted names the
file stores. There is no template to fill and no status object to check;
what checking you want, you write yourself, as the final section of this
page does with its key-set diff.
:end_tab:

```{.python .input #saving-loading-checkpointing-a-training-run-3}
%%tab pytorch
with torch.no_grad():
    for p in net.parameters():
        p.add_(1.0)                       # wreck the weights
before = loss(net(data), target).item()
load_checkpoint('run.pt', net, opt)
after = loss(net(data), target).item()
f'perturbed {before:.2f} -> restored {after:.4f}'
```

```{.python .input #saving-loading-checkpointing-a-training-run-3}
%%tab jax
model_state = nnx.state(model)
nnx.update(model, jax.tree.map(lambda p: p + 1.0, model_state))
before = float(loss(model))
template_model, template_optimizer = fresh_state(cfg)
template = {
    'model': nnx.to_pure_dict(nnx.state(template_model)),
    'optimizer': nnx.to_pure_dict(nnx.state(template_optimizer)),
    'cfg': asdict(Config())}
ckpt = ckptr.restore(os.path.abspath('run-jax'), template)
restored_model = nnx.state(model)
nnx.replace_by_pure_dict(restored_model, ckpt['model'])
nnx.update(model, restored_model)
restored_optimizer = nnx.state(optimizer)
nnx.replace_by_pure_dict(restored_optimizer, ckpt['optimizer'])
nnx.update(optimizer, restored_optimizer)
after = float(loss(model))
f'perturbed {before:.2f} -> restored {after:.4f}'
```

```{.python .input #saving-loading-checkpointing-a-training-run-3}
%%tab tensorflow
for v in net.trainable_variables:
    v.assign_add(tf.ones_like(v))         # wreck the weights
before = float(loss(target, net(data)))
ckpt.restore(path).assert_consumed()
after = float(loss(target, net(data)))
f'perturbed {before:.2f} -> restored {after:.4f}'
```

```{.python .input #saving-loading-checkpointing-a-training-run-3}
%%tab mxnet
for p in net.collect_params().values():
    p.set_data(p.data() + 1.0)            # wreck the weights
before = loss(net)
load_checkpoint('run-mx', net, trainer)
after = loss(net)
f'perturbed {before:.2f} -> restored {after:.4f}'
```

Now the reason the optimizer state is in the file. Resume the run two ways from
the same checkpoint: once restoring the optimizer, once with a fresh one holding
only the weights. The network is near its minimum, so the correct continuation
barely moves. A fresh Adam, with its moment estimates reset and its bias
correction starting over, takes an oversized first step and overshoots.

:begin_tab:`tensorflow`
One Keras 3 tripwire sits in the resume path. A freshly constructed Adam owns
no slot variables; the moment estimates are created only when the optimizer is
built. Restore into a fresh optimizer without building it first and the saved
moments have no variables to land in, so `assert_consumed()` fails with an
unresolved `optimizer` object. The fix is one line before the restore:
`opt.build(model.trainable_variables)`.
:end_tab:

```{.python .input #saving-loading-checkpointing-a-training-run-4}
%%tab pytorch
net_full = build(cfg)
opt_full = torch.optim.Adam(net_full.parameters(), lr=cfg.lr)
load_checkpoint('run.pt', net_full, opt_full)          # weights + optimizer
full = [round(step(net_full, opt_full), 4) for _ in range(5)]

net_fresh = build(cfg)
load_checkpoint('run.pt', net_fresh, optimizer=None)   # weights only
opt_fresh = torch.optim.Adam(net_fresh.parameters(), lr=cfg.lr)
fresh = [round(step(net_fresh, opt_fresh), 4) for _ in range(5)]

print('full  optimizer:', full)
print('fresh optimizer:', fresh)
```

```{.python .input #saving-loading-checkpointing-a-training-run-4}
%%tab jax
full, full_opt = fresh_state(cfg)
full_state, full_opt_state = nnx.state(full), nnx.state(full_opt)
nnx.replace_by_pure_dict(full_state, ckpt['model'])
nnx.replace_by_pure_dict(full_opt_state, ckpt['optimizer'])
nnx.update(full, full_state)
nnx.update(full_opt, full_opt_state)
full_losses = []
for _ in range(5):
    l = step(full, full_opt)
    full_losses.append(round(float(l), 4))

fresh, fresh_opt = fresh_state(cfg)
fresh_state_ = nnx.state(fresh)
nnx.replace_by_pure_dict(fresh_state_, ckpt['model'])
nnx.update(fresh, fresh_state_)
fresh_losses = []
for _ in range(5):
    l = step(fresh, fresh_opt)
    fresh_losses.append(round(float(l), 4))

print('full  optimizer:', full_losses)
print('fresh optimizer:', fresh_losses)
```

```{.python .input #saving-loading-checkpointing-a-training-run-4}
%%tab tensorflow
net_full = build(cfg)
net_full(data[:1])
opt_full = tf.keras.optimizers.Adam(cfg.lr)
opt_full.build(net_full.trainable_variables)   # create Adam's slots first
tf.train.Checkpoint(model=net_full, optimizer=opt_full,
                    step=tf.Variable(0)).restore(path).assert_consumed()
full = [round(step(net_full, opt_full), 4) for _ in range(5)]

net_fresh = build(cfg)
net_fresh(data[:1])
tf.train.Checkpoint(model=net_fresh).restore(path).expect_partial()  # weights only
opt_fresh = tf.keras.optimizers.Adam(cfg.lr)
fresh = [round(step(net_fresh, opt_fresh), 4) for _ in range(5)]

print('full  optimizer:', full)
print('fresh optimizer:', fresh)
```

```{.python .input #saving-loading-checkpointing-a-training-run-4}
%%tab mxnet
net_full = build(cfg)
trainer_full = gluon.Trainer(net_full.collect_params(), 'adam',
                             {'learning_rate': cfg.lr})
load_checkpoint('run-mx', net_full, trainer_full)   # weights + optimizer
full = [round(step(net_full, trainer_full), 4) for _ in range(5)]

net_fresh = build(cfg)
load_checkpoint('run-mx', net_fresh, trainer=None)  # weights only
trainer_fresh = gluon.Trainer(net_fresh.collect_params(), 'adam',
                              {'learning_rate': cfg.lr})
fresh = [round(step(net_fresh, trainer_fresh), 4) for _ in range(5)]

print('full  optimizer:', full)
print('fresh optimizer:', fresh)
```

The full-state run keeps descending; the weights-only run spikes and has to claw
its way back. That transient is the cost of forgetting the optimizer, and it is
why "just the weights" is not a resumable checkpoint.

:begin_tab:`pytorch`
For models too large to hold in memory, checkpoints are split across several
files with an index, and `torch.load(..., mmap=True)` pages tensors off disk on
demand instead of copying the whole file up front. Combined with meta-device
construction and `load_state_dict(..., assign=True)`, this loads such a model
without ever allocating its randomly-initialized weights;
:numref:`chap_performance` returns to the machinery when models get that big.
:end_tab:

:begin_tab:`jax`
For models too large to hold in memory, orbax already works at scale: a
checkpoint is a directory of per-array files rather than one monolith, a restore
can target a device sharding so each accelerator materializes only its own
pieces, and multi-host jobs save and restore in parallel;
:numref:`chap_performance` returns to the machinery when models get that big.
:end_tab:

:begin_tab:`tensorflow`
For models too large to hold in memory, the format already cooperates: a TF
checkpoint is an index file plus data shards rather than one monolith, and
restores are lazy, so a variable created later is filled from the file at
creation time instead of everything materializing up front. Multi-host
`tf.distribute` jobs build on the same machinery;
:numref:`chap_performance` returns to it when models get that big.
:end_tab:

:begin_tab:`mxnet`
For models too large to hold in memory, MXNet offers nothing comparable: a
`.params` file loads whole, and the project was archived before sharded
checkpoints and memory-mapped loading became routine. Models at that scale
are where you leave the framework; :numref:`chap_performance` returns to the
machinery the others provide.
:end_tab:

## Loading Weights You Did Not Train

:begin_tab:`pytorch`
The most common reason to load a `state_dict` is that someone else produced it.
You take a network trained on a large dataset and adapt it: keep the learned
feature extractor, replace the final layer for your own labels. The mechanics are
`state_dict` manipulation. torchvision serves the weights through a `weights=`
enum, which also downloads the matching parameters the first time.
:end_tab:

:begin_tab:`jax`
The most common reason to load parameters is that someone else produced them.
You take a network trained on a large dataset and adapt it: keep the learned
feature extractor, replace the final layer for your own labels. JAX has no
torchvision-style model zoo of its own; pretrained weights come from the
ecosystem, above all the Hugging Face Hub, which distributes them as
safetensors, the format of the previous section. What the framework gives you is
the mechanics, and they are nothing new: the file is a flat dict, the model's
parameters are a pytree, and adapting one to the other is surgery you write
yourself. We reuse the MLP weights saved earlier as our stand-in for a
downloaded file, and build a network that reuses the trunk but ends in a new
two-class head.
:end_tab:

:begin_tab:`tensorflow`
The most common reason to load weights is that someone else produced them. You
take a network trained on a large dataset and adapt it: keep the learned
feature extractor, replace the final layer for your own labels.
`keras.applications` is the built-in zoo; `weights='imagenet'` downloads the
matching parameters the first time, and `include_top=False` drops the
1000-class head so you can attach your own.
:end_tab:

:begin_tab:`mxnet`
The most common reason to load parameters is that someone else produced them.
You take a network trained on a large dataset and adapt it: keep the learned
feature extractor, replace the final layer for your own labels.
`gluon.model_zoo.vision` is the built-in zoo; `pretrained=True` downloads the
matching parameters the first time. One caveat before you rely on it: the
download comes from the archived project's file hosting, which still worked
when this notebook last ran but carries no promise of staying up, so
re-verify it before building anything on top.
:end_tab:

```{.python .input #saving-loading-loading-weights-you-did-not-train-1}
%%tab pytorch
net = resnet18(weights=ResNet18_Weights.DEFAULT)   # ~45 MB on first run
net.fc = nn.Linear(net.fc.in_features, 10)          # new 10-class head
net.fc
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-1}
%%tab jax
class Classifier(nnx.Module):
    def __init__(self, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.hidden = nnx.Linear(20, 256, rngs=rngs)
        self.head = nnx.Linear(256, 2, rngs=rngs)  # new head, new name

    def __call__(self, x):
        return self.head(nnx.relu(self.hidden(x)))

classifier = Classifier()
new_params = nnx.state(classifier, nnx.Param)
[(path, tuple(value.shape)) for path, value in new_params.flat_state()]
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-1}
%%tab tensorflow
backbone = tf.keras.applications.MobileNetV2(   # ~9 MB on first run
    weights='imagenet', include_top=False, input_shape=(160, 160, 3),
    pooling='avg')
net = tf.keras.Sequential([backbone, tf.keras.layers.Dense(10, name='head')])
net(tf.zeros((1, 160, 160, 3)))                 # build the new 10-class head
net.layers[-1]
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-1}
%%tab mxnet
net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)  # ~42 MB on first run
net.output = nn.Dense(10, in_units=512)     # new 10-class head
net.output.initialize()
net.output
```

:begin_tab:`pytorch`
A `state_dict` is an ordinary Python dict, so adapting one is ordinary dict
surgery. We drop the pretrained 1000-class head (we just replaced it) and, to
show what a damaged file looks like, also drop one residual block. Loading with
`strict=False` then returns a report of what did not line up instead of raising.
:end_tab:

:begin_tab:`jax`
There is no `strict=False` to lean on: the merge is yours to write, and so is
the report. Take from the file every entry whose name and shape match the new
model, keep the fresh initialization for the rest, and compute the two key sets
that say what happened: *missing*, parameters the model has but the file did not
fill, and *unexpected*, file entries with no home in the model.
:end_tab:

:begin_tab:`tensorflow`
Keras loads weight files whole, so the partial-loading control is a flag rather
than dict surgery: `load_weights(path, skip_mismatch=True)` fills every
variable whose saved shape matches and skips the rest, reporting the skips as a
warning. To stage a mismatch, save the weights of a donor whose head has 101
classes, our stand-in for a fine-tuned model someone else published, then load
that file into the 10-class network. The cell records the warning so we can
read it back as a report.
:end_tab:

:begin_tab:`mxnet`
The partial-loading flags exist, but they report nothing: `allow_missing=True`
skips parameters the file did not fill and `ignore_extra=True` skips file
entries with no home in the model, both silently. So the report is yours to
compute, the same hand-rolled key-set diff the jax tab writes. Stage a
damaged file first: take the pretrained parameters, drop the head we just
replaced and, to show what damage looks like, the deepest residual stage
(`features.8`), and save what remains as a plain parameter dict.
:end_tab:

```{.python .input #saving-loading-loading-weights-you-did-not-train-2}
%%tab pytorch
pretrained = ResNet18_Weights.DEFAULT.get_state_dict(progress=False)
pretrained = {k: v for k, v in pretrained.items()
              if not k.startswith('fc.') and not k.startswith('layer4.')}
report = net.load_state_dict(pretrained, strict=False)
print('missing by block:', dict(Counter(k.split('.')[0]
                                         for k in report.missing_keys)))
print('unexpected:', report.unexpected_keys)
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-2}
%%tab jax
file_flat = load_file('mlp-jax.safetensors')
new_flat = flatten(new_params)
matched = {k: v for k, v in file_flat.items()
           if k in new_flat and v.shape == new_flat[k].shape}
merged = unflatten({**new_flat, **matched})
print('missing:', sorted(set(new_flat) - set(matched)))
print('unexpected:', sorted(set(file_flat) - set(new_flat)))
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-2}
%%tab tensorflow
donor = tf.keras.Sequential([backbone, tf.keras.layers.Dense(101, name='head')])
donor(tf.zeros((1, 160, 160, 3)))
donor.save_weights('donor-101.weights.h5')   # stand-in for a downloaded file

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    net.load_weights('donor-101.weights.h5', skip_mismatch=True)
report = str(caught[-1].message).splitlines()
print(report[0])
print(report[2].split(' Target variable:')[0])
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-2}
%%tab mxnet
donor = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
file_params = {name: p.data() for name, p in donor.collect_params().items()
               if not name.startswith(('output', 'features.8'))}
npx.savez('resnet18-partial.params', **file_params)

file_keys = set(npx.load('resnet18-partial.params'))
model_keys = set(net.collect_params())
net.load_parameters('resnet18-partial.params',
                    allow_missing=True, ignore_extra=True)  # both skip silently
print('missing by block:', dict(Counter('.'.join(k.split('.')[:2])
                                        for k in sorted(model_keys - file_keys))))
print('unexpected:', sorted(file_keys - model_keys))
```

:begin_tab:`pytorch`
Read this report; do not discard it. `missing_keys` lists parameters the model
has but the file did not fill. The two `fc` entries are expected: that head is
new and meant to start random. The `layer4` entries are a red flag, a whole block
of the backbone left uninitialized, which here means the incoming file was
incomplete and would produce nonsense features. `unexpected_keys`, empty here,
would list names in the file with no home in the model, the usual sign of a
renamed layer. The rule is to name which keys you expect to be missing and treat
anything else as a bug.
:end_tab:

:begin_tab:`jax`
Read both sets before trusting the merged tree. The two `head` entries under
*missing* are expected: that layer is new and meant to start random. The two
`output` entries under *unexpected* are the file's old head, stranded by the
rename, and a renamed layer is what *unexpected* usually means in practice. The
lesson is the same as with a built-in report, only stricter, because nothing
prints it unless you do: name which keys you expect in each set and treat
anything else as a bug.
:end_tab:

:begin_tab:`tensorflow`
Read the warning; do not silence it. It names exactly one object that could not
be loaded, the `head` layer, and gives the two shapes that disagree, `(1280,
10)` against `(1280, 101)`. That is the expected mismatch: the head is new and
meant to start random. Any other layer in that list would mean the backbones
disagree, a wrong input shape or a renamed layer, and is a bug rather than
something to skip. One piece of API drift to know: older tutorials pass
`by_name=True` for partial loads, but in Keras 3 that flag only applies to
legacy `.h5` files and *raises* on the native `.weights.h5` format;
`skip_mismatch` is the current control.
:end_tab:

:begin_tab:`mxnet`
Read the diff; nothing else will print it for you. The two `output` entries
under *missing* are expected: that head is new and meant to start random. The
21 `features.8` entries are a red flag, a whole residual stage the file
failed to deliver; had this file been your only source, those layers would
keep whatever values they started with and the features coming out of them
would be nonsense. *Unexpected*, empty here, would list file entries with no
home in the model, the usual sign of a renamed layer. The rule is the same as
in the other frameworks, only stricter because the flags stay silent: name
which keys you expect in each set and treat anything else as a bug.
:end_tab:

:begin_tab:`pytorch`
With the backbone loaded, freeze it so training touches only the new head. Set
`requires_grad = False` on the pretrained parameters (:numref:`sec_parameters`)
and leave the head trainable.
:end_tab:

:begin_tab:`jax`
With the trunk loaded, freeze it so training touches only the new head.
Parameters in JAX carry no `requires_grad` flag; they are plain arrays, and what
trains is decided by the optimizer. Label each parameter subtree and give the
frozen label a transform that zeroes its updates: gradients still flow, the
optimizer discards them.
:end_tab:

:begin_tab:`tensorflow`
With the backbone loaded, freeze it so training touches only the new head. One
attribute does it: setting `trainable = False` on the backbone removes its
variables from `trainable_variables`, and as a Keras convenience also runs its
BatchNorm layers in inference mode, which is what fine-tuning wants.
:end_tab:

:begin_tab:`mxnet`
With the backbone loaded, freeze it so training touches only the new head.
Gradients in gluon are controlled per parameter by `grad_req`: `'null'` tells
autograd not to compute a gradient for that parameter at all (the running
statistics of BatchNorm layers sit at `'null'` already), and `'write'`
restores the default for the head.
:end_tab:

```{.python .input #saving-loading-loading-weights-you-did-not-train-3}
%%tab pytorch
for p in net.parameters():
    p.requires_grad = False
for p in net.fc.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
total = sum(p.numel() for p in net.parameters())
f'{trainable} trainable of {total}'
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-3}
%%tab jax
labels = jax.tree_util.tree_map_with_path(
    lambda path, _: 'train' if path[0].key == 'head' else 'freeze', merged)
tx = optax.multi_transform(
    {'train': optax.adam(0.05), 'freeze': optax.set_to_zero()}, labels)

sizes = flatten(jax.tree_util.tree_map(jnp.size, merged))
flat_labels = flatten(labels)
trainable = sum(int(s) for k, s in sizes.items() if flat_labels[k] == 'train')
total = sum(int(s) for s in sizes.values())
f'{trainable} trainable of {total}'
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-3}
%%tab tensorflow
backbone.trainable = False
trainable = sum(int(tf.size(v)) for v in net.trainable_variables)
total = sum(int(tf.size(v)) for v in net.weights)
f'{trainable} trainable of {total}'
```

```{.python .input #saving-loading-loading-weights-you-did-not-train-3}
%%tab mxnet
for p in net.collect_params().values():
    p.grad_req = 'null'
for p in net.output.collect_params().values():
    p.grad_req = 'write'

trainable = sum(p.data().size for p in net.collect_params().values()
                if p.grad_req == 'write')
total = sum(p.data().size for p in net.collect_params().values())
f'{trainable} trainable of {total}'
```

:begin_tab:`pytorch`
torchvision is one source; the Hugging Face Hub is the ecosystem-scale one, and
it distributes its weights as safetensors, which closes the loop with the format
of the previous section. This section covers *how* to load and adapt pretrained
weights; :numref:`sec_fine_tuning` covers when it helps and how far to unfreeze.
:end_tab:

:begin_tab:`jax`
The Hugging Face Hub distributes JAX weights as safetensors, so the flat dict
you just merged has the same shape as the artifact you will download in
practice. This section covers *how* to load and adapt pretrained weights;
:numref:`sec_fine_tuning` covers when it helps and how far to unfreeze.
:end_tab:

:begin_tab:`tensorflow`
`keras.applications` is one source; the Hugging Face Hub is the ecosystem-scale
one, and it distributes its weights as safetensors, which closes the loop with
the format of the previous section. This section covers *how* to load and adapt
pretrained weights; :numref:`sec_fine_tuning` covers when it helps and how far
to unfreeze.
:end_tab:

:begin_tab:`mxnet`
`gluon.model_zoo` is one source, frozen where the project stopped; the
Hugging Face Hub is the ecosystem-scale one, and it distributes weights as
safetensors, which for MXNet means the numpy bridge of the previous section
is also your import path. This section covers *how* to load and adapt
pretrained weights; :numref:`sec_fine_tuning` covers when it helps and how
far to unfreeze.
:end_tab:

## Summary

:begin_tab:`pytorch`
A saved model is state, not code: a `state_dict` of tensors that means something
only once the code that built the network runs again. For your own files
`torch.save` is fine; for files you share, safetensors stores the same tensors
with no executable pickle, which is why hubs standardize on it. A resumable
checkpoint bundles more than weights: optimizer state, RNG state, step, and
config, written atomically, or a resume restarts the optimizer's momentum from
zero. Loading someone else's weights is dict surgery plus `strict=False`, and the
missing/unexpected report is a diagnostic to read rather than a warning to
silence.
:end_tab:

:begin_tab:`jax`
A saved model is state, not code: a pytree of named arrays that means something
only once the code that built the network runs again. For your own files
`jnp.save` is fine; for files you share, safetensors stores the same tensors
behind a plain JSON header, with no pickle to execute, which is why hubs
standardize on it. A resumable checkpoint contains model state, optimizer
state, and step in pytrees that Orbax saves atomically in a single call, or a
resume restarts the optimizer's momentum from zero. Loading someone else's
weights is pytree surgery, and the missing/unexpected sets you compute are a
diagnostic to read rather than a formality to skip.
:end_tab:

:begin_tab:`tensorflow`
A saved model is state, not code: a collection of path-named variables that
means something only once the code that built the network runs again. For your
own files `np.save` and `.weights.h5` are fine; for files you share,
safetensors stores the same tensors behind a plain JSON header, with no pickle
to execute, which is why hubs standardize on it. A resumable checkpoint is a
`tf.train.Checkpoint` of model, optimizer, and step saved as one unit, or a
resume restarts the optimizer's momentum from zero; a fresh optimizer must be
built before the restore so Adam's moments have somewhere to land. Loading
someone else's weights is `load_weights` with `skip_mismatch=True`, and the
skip warning is a diagnostic to read rather than a message to silence.
:end_tab:

:begin_tab:`mxnet`
A saved model is state, not code: a dictionary of dotted-path names to arrays
that means something only once the code that built the network runs again.
For your own files `save_parameters` is fine, and its format is pure array
data with no pickle to execute; for files you share, safetensors reaches
every framework, and the `safetensors.numpy` bridge is all MXNet needs to
speak it. A resumable checkpoint is three files written atomically:
parameters, trainer states (Adam's moments, stored with pickle, so keep such
files your own), and a JSON sidecar with the step and config; the RNG stream
cannot be snapshotted, only reseeded. Loading someone else's weights is dict
surgery plus `allow_missing`/`ignore_extra`, and because those flags skip
silently, the missing/unexpected key-set diff is yours to compute and read.
:end_tab:

## Exercises

1. Even if you never deploy to another machine, name two reasons to checkpoint.
   Then consider the atomic write: if the checkpoint were written straight to its
   final path (delete the `os.replace` from `save_checkpoint`, the
   rename-into-place that orbax performs, or the fresh numbered files that
   `tf.train.Checkpoint.save` writes), describe the failure a crash mid-write
   now causes, and why the atomic version avoids it.
1. Read the first 8 bytes of the safetensors file you wrote for the MLP as a
   little-endian integer, as the header cell does. How large is the JSON header
   for the MLP, and how does it grow if you double the hidden width?
1. Save the MLP's parameters cast to `bfloat16` and load them back into a
   `float32` model (:numref:`sec_numerics`). What is lost? Is that acceptable
   for inference? For resuming training?
1. Take two checkpoints of the regressor 50 steps apart, average their weight
   tensors into a third set of parameters, and evaluate it. The result previews
   weight averaging.
