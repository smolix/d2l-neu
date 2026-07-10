```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Saving, Loading, and Pretrained Weights
:label:`sec_read_write_v2`

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
:numref:`sec_model_construction_v2` belongs *inside* the checkpoint, and why the
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

## State, Not Code

The state of a network is a dictionary from parameter names to tensors, the
`state_dict` of :numref:`sec_parameters_v2`. Before we save a whole model, the
warm-up is that the same `save`/`load` calls work on any tensors, and on the
lists and dicts that hold them.

```{.python .input #saving-loading-state-not-code-1}
%%tab pytorch
x = torch.arange(4)
torch.save({'x': x, 'y': torch.zeros(4)}, 'tensors.pt')
torch.load('tensors.pt', weights_only=True)
```

A model's `state_dict` is one such dictionary, built for you. The keys are the
dotted paths through the module tree (`hidden.weight`, `output.bias`); the values
are the tensors, buffers included. Here is the tree for a small MLP.

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

Nothing in this dictionary knows it came from a class called `MLP`. That is the
point: the names and shapes are enough to refill any network built by the same
code, and they carry no dependence on how that code happens to be written today.

## safetensors: the Interchange Format

`torch.save` writes its files with Python's `pickle`, which does not store data
so much as a program that *reconstructs* data. Unpickling runs that program. For
a file you wrote and never let out of your control this is harmless. For a file
you downloaded it is a remote-code-execution vector: loading the weights can run
whatever the author's pickle stream tells it to.

The risk is easy to make concrete. An object's `__reduce__` method returns the
callable and arguments that pickle will invoke on load. Point it at any function
and that function runs when the file is read.

```{.python .input #saving-loading-safetensors-the-interchange-format-1}
%%tab pytorch
class Tripwire:
    def __reduce__(self):
        return (print, ('*** payload executed while loading ***',))

torch.save(Tripwire(), 'tripwire.pt')
_ = torch.load('tripwire.pt', weights_only=False)  # the payload runs here
```

Since version 2.6, PyTorch defaults `torch.load` to `weights_only=True`, which
refuses any pickle opcode that is not a plain tensor. The same file is now
rejected instead of executed.

```{.python .input #saving-loading-safetensors-the-interchange-format-2}
%%tab pytorch
try:
    torch.load('tripwire.pt', weights_only=True)  # the default in this torch
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
```

The allowlist behind `weights_only=True` is defense in depth, not a sandbox: it
has itself had bypasses patched. safetensors removes the problem at the root by
having no program to run. As :numref:`fig_bg_safetensors_layout` lays out byte
by byte, a safetensors file is an 8-byte little-endian integer giving the
header length, then that many bytes of JSON naming each tensor's dtype, shape,
and byte range, then the raw tensor bytes back to back. There is no opcode
stream to interpret, so loading one cannot execute anything. It is also
framework-neutral and memory-mappable, which is why model hubs default to it.
Save and reload the MLP's state through it and confirm the round trip is exact.

![The safetensors file as one horizontal byte strip: an 8-byte header length, a JSON header naming each tensor's dtype, shape, and byte offsets, and the raw tensor bytes packed back to back with no gaps, with two of the file's own data_offsets entries traced down to their exact span in the bar.](../img/bg-safetensors-layout.svg)
:label:`fig_bg_safetensors_layout`

```{.python .input #saving-loading-safetensors-the-interchange-format-3}
%%tab pytorch
save_file(net.state_dict(), 'mlp.safetensors')
clone = MLP()
clone(X)                                   # materialize the lazy layers first
clone.load_state_dict(load_file('mlp.safetensors'))
clone.eval()
torch.equal(clone(X), Y)
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

`torch.save` keeps its place for your own scratch files and for the older code
you will still meet. safetensors is what you use to hand a model to anyone else.

## Checkpointing a Training Run

A checkpoint you can resume from holds more than weights. Resuming means picking
up the optimizer where it stopped, and Adam's state is the running first and
second moments of the gradients from :numref:`sec_parameters_v2`. Drop them and
the optimizer restarts its momentum from zero, so the first steps after a resume
no longer behave like a continuation. A full checkpoint therefore bundles the
model state, the optimizer state, the RNG state (so data shuffling and dropout
continue the same stream), the step counter, and the config that sizes the model
when you rebuild it. :numref:`fig_bg_checkpoint_contents` pairs each of those
five compartments with the exact thing it restores on resume.

![A checkpoint file's five compartments, each paired by an arrow with what it restores on resume: model state_dict with weights, optimizer state with momentum and second moments, RNG state with data order and dropout, step with schedule position, and config with architecture.](../img/bg-checkpoint-contents.svg)
:label:`fig_bg_checkpoint_contents`

Two details separate a checkpoint from a corrupted file. First, keep the contents
to tensors and primitives so the file loads under `weights_only=True`; a
dataclass config goes in as a plain dict via `asdict`. Second, write atomically:
save to a temporary path and `os.replace` it into place, so a crash mid-write
leaves the previous good checkpoint untouched rather than a half-written one.

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

Train a tiny regressor for a hundred steps and checkpoint it. The `Config`
dataclass is what a rebuild reads to size the model, so it travels inside the
file with the weights.

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

The restore is exact. Corrupt every parameter, load the checkpoint back, and the
loss returns to where it was.

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

Now the reason the optimizer state is in the file. Resume the run two ways from
the same checkpoint: once restoring the optimizer, once with a fresh one holding
only the weights. The network is near its minimum, so the correct continuation
barely moves. A fresh Adam, with its moment estimates reset and its bias
correction starting over, takes an oversized first step and overshoots.

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

The full-state run keeps descending; the weights-only run spikes and has to claw
its way back. That transient is the cost of forgetting the optimizer, and it is
why "just the weights" is not a resumable checkpoint.

For models too large to hold in memory, checkpoints are split across several
files with an index, and `torch.load(..., mmap=True)` pages tensors off disk on
demand instead of copying the whole file up front. Combined with meta-device
construction and `load_state_dict(..., assign=True)`, this loads such a model
without ever allocating its randomly-initialized weights;
:numref:`chap_performance` returns to the machinery when models get that big.

## Loading Weights You Did Not Train

The most common reason to load a `state_dict` is that someone else produced it.
You take a network trained on a large dataset and adapt it: keep the learned
feature extractor, replace the final layer for your own labels. The mechanics are
`state_dict` manipulation. torchvision serves the weights through a `weights=`
enum, which also downloads the matching parameters the first time.

```{.python .input #saving-loading-loading-weights-you-did-not-train-1}
%%tab pytorch
net = resnet18(weights=ResNet18_Weights.DEFAULT)   # ~45 MB on first run
net.fc = nn.Linear(net.fc.in_features, 10)          # new 10-class head
net.fc
```

A `state_dict` is an ordinary Python dict, so adapting one is ordinary dict
surgery. We drop the pretrained 1000-class head (we just replaced it) and, to
show what a damaged file looks like, also drop one residual block. Loading with
`strict=False` then returns a report of what did not line up instead of raising.

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

Read this report; do not discard it. `missing_keys` lists parameters the model
has but the file did not fill. The two `fc` entries are expected: that head is
new and meant to start random. The `layer4` entries are a red flag, a whole block
of the backbone left uninitialized, which here means the incoming file was
incomplete and would produce nonsense features. `unexpected_keys`, empty here,
would list names in the file with no home in the model, the usual sign of a
renamed layer. The rule is to name which keys you expect to be missing and treat
anything else as a bug.

With the backbone loaded, freeze it so training touches only the new head. Set
`requires_grad = False` on the pretrained parameters (:numref:`sec_parameters_v2`)
and leave the head trainable.

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

torchvision is one source; the Hugging Face Hub is the ecosystem-scale one, and
it distributes its weights as safetensors, which closes the loop with the format
of the previous section. This section covers *how* to load and adapt pretrained
weights; :numref:`sec_fine_tuning` covers when it helps and how far to unfreeze.

## Summary

A saved model is state, not code: a `state_dict` of tensors that means something
only once the code that built the network runs again. For your own files
`torch.save` is fine; for files you share, safetensors stores the same tensors
with no executable pickle, which is why hubs standardize on it. A resumable
checkpoint bundles more than weights: optimizer state, RNG state, step, and
config, written atomically, or a resume restarts the optimizer's momentum from
zero. Loading someone else's weights is dict surgery plus `strict=False`, and the
missing/unexpected report is a diagnostic to read rather than a warning to
silence.

## Exercises

1. Even if you never deploy to another machine, name two reasons to checkpoint.
   Then delete the `os.replace` from `save_checkpoint` and write straight to
   `path`; describe the failure a crash mid-write now causes, and why the atomic
   version avoids it.
1. Read the first 8 bytes of your `mlp.safetensors` as a little-endian integer,
   as the header cell does. How large is the JSON header for the MLP, and how
   does it grow if you double the hidden width?
1. Save the MLP's `state_dict` cast to `bfloat16` and load it back into a
   `float32` model (:numref:`sec_numerics_v2`). What is lost? Is that acceptable
   for inference? For resuming training?
1. Take two checkpoints of the regressor 50 steps apart, average their weight
   tensors into a third `state_dict`, and evaluate it. The result previews weight
   averaging.
