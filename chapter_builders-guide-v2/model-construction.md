```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Modules and Model Construction
:label:`sec_model_construction_v2`

The networks we have trained so far were small enough to write down layer by
layer. The networks ahead are not: a ResNet chains more than a hundred
convolutional layers, and a GPT-style language model stacks dozens of identical
Transformer blocks :cite:`Radford.Wu.Child.ea.2019`. Nobody writes such models
one layer at a time, and nobody designs them that way either. The unit of
design is the *block*, a group of layers that repeats, and the abstraction that
makes blocks composable is the *module*.

A module is an object with three responsibilities: it owns *parameters*, it
owns *child modules*, and it implements a *forward computation* that maps
inputs to outputs. The definition is deliberately recursive. A fully connected
layer is a module (parameters, no children). A residual block is a module (no
parameters of its own, a few child layers). A hundred-layer network is a module
whose children are blocks whose children are layers. Every model is therefore a
*tree* of modules, as sketched in :numref:`fig_blocks_v2`, and almost everything
this chapter does to a model, listing its parameters
(:numref:`sec_parameters_v2`), moving it to a GPU (:numref:`sec_use_gpu_v2`),
saving it to disk (:numref:`sec_read_write_v2`), is implemented as a walk over
that tree.

![Layers compose into blocks and blocks compose into models: every model is a tree of modules.](../img/bg-module-tree.svg)
:label:`fig_blocks_v2`

```{.python .input #model-construction-modules-and-model-construction}
%%tab pytorch
from dataclasses import dataclass
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## The Module Abstraction

In PyTorch the module class is `nn.Module`. We have used one of its subclasses
all along: `nn.Sequential` builds a model from a chain of layers, here the
familiar MLP with a 256-unit ReLU hidden layer and a 10-unit output layer.

```{.python .input #model-construction-the-module-abstraction-1}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

`Sequential` is not a special construct. It is itself a module whose forward
computation runs its children in order, and the children are the three modules
we passed to it, stored under names in a registry:

```{.python .input #model-construction-the-module-abstraction-2}
%%tab pytorch
net._modules
```

This registry is what the `nn.Module` machinery traverses: `net.parameters()`
collects parameters by walking `_modules` recursively, and the same walk
underlies device movement and serialization. A module the registry does not
contain might as well not exist, a fact we will exploit for a demonstration
shortly.

`nn.Sequential` covers chains. For any other topology we subclass `nn.Module`
directly and supply the two methods that define a module: a constructor that
creates the children, and a `forward` method that uses them.

```{.python .input #model-construction-the-module-abstraction-3}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input #model-construction-the-module-abstraction-4}
%%tab pytorch
net = MLP()
net(X).shape
```

Two details do the work here. First, `self.hidden = nn.LazyLinear(256)` is not
an ordinary attribute assignment: `nn.Module` intercepts `__setattr__`, sees
that the value is a module, and adds it to the registry we just inspected. That
is why both layers' parameters show up in `net.parameters()` with no further
ceremony. Second, we never wrote a backward method; automatic differentiation
derives gradients from whatever `forward` computes.

Note also that we invoke the model as `net(X)`, never `net.forward(X)`.
Calling a module runs `nn.Module.__call__`, which calls `forward` *and* any
hooks registered on the module. That gap between call and forward is where
model-inspection tooling attaches; we use it in :numref:`sec_repro_v2`.

## Sequential and Friends: Containers
:label:`subsec_model-construction-sequential`

To see that there is no magic left in `nn.Sequential`, we can write it
ourselves. Two ingredients suffice: register each child under a name, and loop
over the children in `forward`.

```{.python .input #model-construction-sequential-and-friends-containers-1}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
```

`add_module` writes a child into the registry under a string name (that is
where the `'0'`, `'1'`, `'2'` keys above came from), and `children()` iterates
the registry in insertion order. Our version is a drop-in replacement:

```{.python .input #model-construction-sequential-and-friends-containers-2}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

The registration step is easy to lose. The following module looks reasonable,
and its forward pass works, so nothing appears wrong:

```{.python .input #model-construction-sequential-and-friends-containers-3}
%%tab pytorch
class PlainListMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)]

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

net = PlainListMLP()
net(X).shape, sum(p.numel() for p in net.parameters())
```

The model computes, yet it owns zero parameters. A plain Python list is not a
module, so the `__setattr__` interception ignores it and nothing inside it is
registered. The layers still hold perfectly good tensors, which is why
`forward` runs. But `net.parameters()` is empty: hand this model to an
optimizer and training proceeds without a single error while updating nothing;
`net.to(device)` leaves every layer behind on the CPU; `net.state_dict()`
checkpoints an empty model. Because no step raises an exception, this bug is
usually diagnosed by staring at a loss curve that refuses to move.

The registered container for a list of children is `nn.ModuleList`. Wrapping
the existing list is the entire fix:

```{.python .input #model-construction-sequential-and-friends-containers-4}
%%tab pytorch
class ModuleListMLP(PlainListMLP):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(self.layers)

net = ModuleListMLP()
net(X).shape, sum(p.numel() for p in net.parameters())
```

Same forward pass, 7946 registered parameters. The division of labor among the
containers is now clear. `nn.Sequential` registers its children and supplies
the run-them-in-order `forward`. `nn.ModuleList` registers a list of children
but supplies no `forward` at all; you keep writing the loop, which is exactly
what you want when the loop body is not a plain chain (blocks that take extra
arguments, or a skip connection around each block). `nn.ModuleDict` does the
same for children indexed by name. Transformer implementations conventionally
keep their stack of blocks in an `nn.ModuleList` and their named parts
(embedding, final normalization, output head) as attributes.

## Forward Is Just Python

`forward` is an ordinary Python method. Nothing restricts it to chaining
children: it can branch, loop, call any tensor function, and combine
intermediate results however it likes. The loop in `ModuleListMLP` already used
this freedom. Its most consequential one-line use is the *residual
connection*, the wiring idiom at the heart of ResNets and Transformers alike:

```{.python .input #model-construction-forward-is-just-python-1}
%%tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens), nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens))

    def forward(self, X):
        return X + self.body(X)
```

![The residual wiring `X + body(X)`: the input splits at a branch point into the body stack and an identity skip, and the two rejoin by addition before the block's output.](../img/bg-residual-block.svg)
:label:`fig_bg_residual-block`

`X + self.body(X)` is not a layer PyTorch provides. It is arithmetic in
`forward`, and it changes what the block *is*: the block computes a
perturbation of the identity function rather than an arbitrary transformation,
and during backpropagation the skip path hands gradients to earlier layers
undiminished, tempering the vanishing gradients of
:numref:`sec_numerical_stability`. :numref:`fig_bg_residual-block` diagrams
exactly this wiring. Chapter 8 develops both points when we build ResNet; for
now we only need the mechanics. One mechanical consequence
is visible already: the addition forces the input and output shapes to agree,
so a residual block has a single width that is part of its identity. That is
why we gave `body` explicit `nn.Linear` layers rather than lazy ones.

```{.python .input #model-construction-forward-is-just-python-2}
%%tab pytorch
block = ResidualBlock(24)
block(torch.randn(2, 24)).shape
```

`forward` may also use state that is neither an input nor a parameter. Suppose
we want to damp each block's contribution by a fixed factor:

```{.python .input #model-construction-forward-is-just-python-3}
%%tab pytorch
class ScaledResidual(ResidualBlock):
    def __init__(self, num_hiddens, alpha=0.5):
        super().__init__(num_hiddens)
        self.alpha = torch.tensor(alpha)  # Fixed by design, never trained

    def forward(self, X):
        return X + self.alpha * self.body(X)

block = ScaledResidual(24)
'alpha' in block.state_dict(), list(block.state_dict())[:2]
```

`alpha` enters the computation, but it is not a parameter: it never appears in
`named_parameters()`, so the optimizer never touches it. That much we wanted.
Storing it as a plain attribute has a cost we did not want, though: as the
output shows, it is missing from `state_dict()` as well, so it will not be
saved with the model, and `.to(device)` will not move it. Some state is not a
parameter but must still travel with the model; the registered home for such
state is a *buffer*, introduced in :numref:`sec_parameters_v2`.

## Lazy Initialization: Shapes from Data
:label:`sec_lazy_init`

We have been doing something odd since our first MLP without commenting on it:
`nn.LazyLinear(256)` names only the layer's *output* width. Its weight matrix
has shape `(256, in_features)`, and we never said what `in_features` is.
The layer cannot know it at construction time, since it depends on the data it
will receive. So it does not allocate parameters at construction time at all:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-1}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net[0].weight
```

The weight is a placeholder. The first time data flows through, the layer
reads the input width from the batch, allocates and initializes a real weight,
and replaces itself with a plain `nn.Linear`. Its output width then fixes the
input of the next lazy layer, and shapes cascade through the whole model:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-2}
%%tab pytorch
net(X)
net[0].weight.shape
```

We build models this way from Chapter 7 on because it removes shape arithmetic
from model definitions. In a convolutional network the flattened feature-map
size depends on the input resolution and on every upstream stride and padding
choice; computing it by hand clutters the code, and every architecture edit
invalidates the numbers downstream of it. Declaring output widths and letting
input widths come from data ends that bookkeeping.

The convenience comes with one rule: until the first forward pass, the
parameters *do not exist*. Anything that needs the parameter list, whether
constructing an optimizer, applying an initializer, or counting parameters,
must happen after a *dry run* on a representative batch. A related subtlety:
the random initialization now happens at first call rather than at
construction, so any random numbers your program draws in between shift the
generator's state, and a fixed seed can yield different weights than the
explicitly shaped version of the same model (:numref:`sec_repro_v2` returns to
seeding). Other libraries make the dry run explicit rather than implicit: a
Flax module in JAX has no lazy mode at all, and its parameters exist only
after a mandatory `init(key, dummy_input)` call performs the same shape
inference. PyTorch's lazy layers give you that behavior with the first real
batch playing the role of the dummy input.

The dry run is such a common preamble that we fold it into the `d2l.Module`
base class from :numref:`sec_oo-design`: run the model once to materialize
every shape, then optionally apply an initialization function.

```{.python .input #model-construction-lazy-initialization-shapes-from-data-3}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

`nn.Module.apply(fn)` calls `fn` on every module in the tree, children first.
It is the standard way to push a policy across an arbitrary model, one more
operation that is a tree walk, and from Chapter 7 on the idiom
`model.apply_init([X], init)` opens most of our training scripts. A small
demonstration:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-4}
%%tab pytorch
class TinyMLP(d2l.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(),
                                 nn.LazyLinear(10))

def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

model = TinyMLP()
model.apply_init([X], init_xavier)
model.net[0].weight.shape
```

The dry run inside `apply_init` turned every lazy layer into a real one, after
which `init_xavier` could match on `nn.Linear` and rewrite its weights. Which
initializer to apply, and why Xavier's variance rule
(:numref:`subsec_xavier`) is a sensible default, is the subject of
:numref:`sec_init_v2`.

## Building from a Config

So far every width in this section was a literal typed into a constructor.
Real model code does not work that way. An architecture is a handful of
integers and switches (depth, width, output size), and those numbers must be
varied across experiments, logged with results, and stored with checkpoints so
that a saved model can be rebuilt. The standard pattern is to collect them in
a small configuration object and derive the model from it:

```{.python .input #model-construction-building-from-a-config-1}
%%tab pytorch
@dataclass
class MLPConfig:
    d_in: int = 784
    d_hidden: int = 256
    num_blocks: int = 4
    d_out: int = 10

def build(cfg: MLPConfig) -> nn.Module:
    blocks = [ResidualBlock(cfg.d_hidden) for _ in range(cfg.num_blocks)]
    return nn.Sequential(nn.Linear(cfg.d_in, cfg.d_hidden),
                         *blocks, nn.Linear(cfg.d_hidden, cfg.d_out))
```

One config produces one architecture: an input projection into the hidden
width, `num_blocks` identical residual blocks, and an output projection. The
list comprehension producing `blocks` is a plain Python list, which is safe
here for the reason :numref:`subsec_model-construction-sequential` taught:
unpacking it into `nn.Sequential` is what registers each block. Every layer
gets explicit widths, and this is where explicit shapes beat lazy ones: the
config already knows every width, so explicit construction yields a fully
materialized model with no dry run needed. Printing the model displays the
module tree:

```{.python .input #model-construction-building-from-a-config-2}
%%tab pytorch
net = build(MLPConfig())
net
```

```{.python .input #model-construction-building-from-a-config-3}
%%tab pytorch
net(torch.rand(2, 784)).shape
```

Architecture is now *data*. Rescaling the model is a change to two fields, not
an edit to model code:

```{.python .input #model-construction-building-from-a-config-4}
%%tab pytorch
for cfg in (MLPConfig(), MLPConfig(d_hidden=512, num_blocks=8)):
    n = sum(p.numel() for p in build(cfg).parameters())
    print(f'd_hidden={cfg.d_hidden}, num_blocks={cfg.num_blocks}: '
          f'{n:,} parameters')
```

Because `build` is deterministic in `cfg`, the config is all you need to
reconstruct the module tree later; :numref:`sec_read_write_v2` saves it
alongside the weights so that loading a checkpoint starts by rebuilding the
exact same model. A config of widths and depths feeding a loop that stacks
identical residual blocks is, minus attention, the exact shape of every
Transformer implementation you will read.

## Summary

A module owns parameters, child modules, and a `forward` method. Layers,
blocks, and whole models are the same kind of object, so a model is a tree of
modules, and parameter collection, device movement, and serialization are all
walks over that tree. The tree is discovered through registration: attribute
assignment and the containers `nn.Sequential`, `nn.ModuleList`, and
`nn.ModuleDict` register children, while a plain Python list hides them,
yielding a model that runs but trains nothing. `forward` is ordinary Python;
a residual connection is one line in it. Lazy layers declare output widths and
infer input widths on the first forward pass, so initialization and inspection
follow a dry run (`apply_init`). Configs turn architecture into data: a
`dataclass` of widths and depths plus a `build` function that stacks blocks.

## Exercises

1. Take `PlainListMLP` and catalog everything that breaks besides the empty
   parameter list. Check `net.state_dict()`, the effect of
   `net.to(torch.float64)` on the hidden layers' dtypes, and whether
   `net.eval()` reaches the children. Explain how each failure follows from
   the same missing registration.
1. Implement a `ParallelBlock` that takes two child modules `net1` and `net2`,
   runs both on the same input, and concatenates their outputs along the last
   dimension. What must be true of the two children's outputs for the
   concatenation to be valid?
1. Extend `MLPConfig` with an activation switch (for example,
   `act: str = 'relu'`) and make `build` honor it. Which decisions belong in a
   config and which belong in code? Where would you put a choice between
   `ResidualBlock` and a plain feed-forward block?
1. `ResidualBlock` requires its input and output widths to agree. Suppose you
   want a block whose output is wider than its input. Give two standard fixes
   and the cost of each. (Chapter 8 uses one of them in ResNet.)
