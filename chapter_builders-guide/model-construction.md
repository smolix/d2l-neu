```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Modules and Model Construction
:label:`sec_model_construction`

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
*tree* of modules, as sketched in :numref:`fig_blocks`, and almost everything
this chapter does to a model, listing its parameters
(:numref:`sec_parameters`), moving it to a GPU (:numref:`sec_use_gpu`),
saving it to disk (:numref:`sec_read_write`), is implemented as a walk over
that tree.

![Layers compose into blocks and blocks compose into models: every model is a tree of modules.](../img/bg-module-tree.svg)
:label:`fig_blocks`

```{.python .input #model-construction-modules-and-model-construction}
%%tab pytorch
from dataclasses import dataclass
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #model-construction-modules-and-model-construction}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

```{.python .input #model-construction-modules-and-model-construction}
%%tab tensorflow
from dataclasses import dataclass
import tensorflow as tf
```

```{.python .input #model-construction-modules-and-model-construction}
%%tab mxnet
from dataclasses import dataclass
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## The Module Abstraction

:begin_tab:`pytorch`
In PyTorch the module class is `nn.Module`. We have used one of its subclasses
all along: `nn.Sequential` builds a model from a chain of layers, here the
familiar MLP with a 256-unit ReLU hidden layer and a 10-unit output layer.
:end_tab:

:begin_tab:`jax`
In JAX the module class is Flax's `flax.linen.Module`, imported as `nn`. We
have used one of its subclasses all along: `nn.Sequential` builds a model from
a list of layers, here the familiar MLP with a 256-unit ReLU hidden layer and
a 10-unit output layer. One habit from :numref:`sec_oo-design` carries over: a
Flax module holds no parameters of its own. Constructing `net` records the
architecture; `init(key, X)` creates the parameters as a separate object, and
`apply(params, X)` runs the forward computation with them.
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow the module class is Keras's `tf.keras.Model`. We have used one
of its subclasses all along: `tf.keras.Sequential` builds a model from a list
of layers, here the familiar MLP with a 256-unit ReLU hidden layer and a
10-unit output layer. Keras attaches the activation to the `Dense` layer
itself rather than interposing a separate activation layer.
:end_tab:

:begin_tab:`mxnet`
In MXNet the module class is Gluon's `nn.Block`. We have used one of its
subclasses all along: `nn.Sequential` builds a model from a chain of layers
appended with its `add` method, here the familiar MLP with a 256-unit ReLU
hidden layer and a 10-unit output layer. Gluon attaches the activation to the
`Dense` layer itself rather than interposing a separate activation layer, and
one `initialize()` call stands between construction and the first forward
pass; what that call does, and what it defers, is where this section ends.
:end_tab:

```{.python .input #model-construction-the-module-abstraction-1}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input #model-construction-the-module-abstraction-1}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

```{.python .input #model-construction-the-module-abstraction-1}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(10)])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input #model-construction-the-module-abstraction-1}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

:begin_tab:`pytorch`
`Sequential` is not a special construct. It is itself a module whose forward
computation runs its children in order, and the children are the three modules
we passed to it, stored under names in a registry:
:end_tab:

:begin_tab:`jax`
`Sequential` is not a special construct. It is itself a module whose forward
computation runs its children in order, and the children are the three modules
we passed to it. Their parameters do not live in `net`: `init` returned them
as a nested dictionary, a *pytree*, whose structure mirrors the module tree:
:end_tab:

:begin_tab:`tensorflow`
`Sequential` is not a special construct. It is itself a model whose forward
computation runs its children in order, and the children are the two layers
we passed to it, held in a tracked list:
:end_tab:

:begin_tab:`mxnet`
`Sequential` is not a special construct. It is itself a `Block` whose forward
computation runs its children in order, and the children are the two layers
`add` appended, stored under names in a registry that printing the model
displays:
:end_tab:

```{.python .input #model-construction-the-module-abstraction-2}
%%tab pytorch
net._modules
```

```{.python .input #model-construction-the-module-abstraction-2}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

```{.python .input #model-construction-the-module-abstraction-2}
%%tab tensorflow
net.layers
```

```{.python .input #model-construction-the-module-abstraction-2}
%%tab mxnet
net
```

:begin_tab:`pytorch`
This registry is what the `nn.Module` machinery traverses: `net.parameters()`
collects parameters by walking `_modules` recursively, and the same walk
underlies device movement and serialization. A module the registry does not
contain might as well not exist, a fact we will exploit for a demonstration
shortly.
:end_tab:

:begin_tab:`jax`
This pytree is what every downstream operation traverses: an optimizer update
is a `tree_map` over `params`, and the same traversal underlies device
movement and serialization. A parameter outside the pytree might as well not
exist, which is why `apply` takes `params` explicitly: the model is the pair
of architecture and pytree, not either half alone.
:end_tab:

:begin_tab:`tensorflow`
This list is what the Keras machinery traverses: `net.trainable_variables`
collects variables by walking the children recursively, and the same walk
underlies serialization. A layer the tracker does not see might as well not
exist, though as we will see shortly, Keras goes to some lengths to make
sure that cannot happen by accident.
:end_tab:

:begin_tab:`mxnet`
This registry is what the `Block` machinery traverses: `net.collect_params()`
gathers parameters by walking the children recursively and keys them by those
registry names (`0.weight`, `1.bias`, ...), and the same walk underlies
initialization, device movement, and serialization. A block the registry does
not contain might as well not exist, a fact we will exploit for a
demonstration shortly.
:end_tab:

:begin_tab:`pytorch`
`nn.Sequential` covers chains. For any other topology we subclass `nn.Module`
directly and supply the two methods that define a module: a constructor that
creates the children, and a `forward` method that uses them.
:end_tab:

:begin_tab:`jax`
`nn.Sequential` covers chains. For any other topology we subclass `nn.Module`
directly and supply the two methods that define a module: a `setup` method
that creates the children, and a `__call__` method that uses them.
:end_tab:

:begin_tab:`tensorflow`
`Sequential` covers chains. For any other topology we subclass
`tf.keras.Model` directly and supply the two methods that define a module: a
constructor that creates the children, and a `call` method that uses them.
:end_tab:

:begin_tab:`mxnet`
`nn.Sequential` covers chains. For any other topology we subclass `nn.Block`
directly and supply the two methods that define a module: a constructor that
creates the children, and a `forward` method that uses them.
:end_tab:

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

```{.python .input #model-construction-the-module-abstraction-3}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

```{.python .input #model-construction-the-module-abstraction-3}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(10)

    def call(self, X):
        return self.out(self.hidden(X))
```

```{.python .input #model-construction-the-module-abstraction-3}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input #model-construction-the-module-abstraction-4}
%%tab pytorch
net = MLP()
net(X).shape
```

```{.python .input #model-construction-the-module-abstraction-4}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

```{.python .input #model-construction-the-module-abstraction-4}
%%tab tensorflow
net = MLP()
net(X).shape
```

```{.python .input #model-construction-the-module-abstraction-4}
%%tab mxnet
net = MLP()
net.initialize()
net(X).shape
```

:begin_tab:`pytorch`
Two details do the work here. First, `self.hidden = nn.LazyLinear(256)` is not
an ordinary attribute assignment: `nn.Module` intercepts `__setattr__`, sees
that the value is a module, and adds it to the registry we just inspected. That
is why both layers' parameters show up in `net.parameters()` with no further
ceremony. Second, we never wrote a backward method; automatic differentiation
derives gradients from whatever `forward` computes.
:end_tab:

:begin_tab:`jax`
Two details do the work here. First, `self.hidden = nn.Dense(256)` inside
`setup` is not an ordinary attribute assignment: Flax registers the value as a
child module, and the attribute names become keys of the params pytree
(`hidden`, `out`). Second, we never wrote a backward method; `jax.grad`
derives gradients from whatever `__call__` computes. Flax also offers a more
compact way to write the same module: decorate `__call__` with `@nn.compact`
and create the layers inline at the point of use, skipping `setup` entirely.
We use that style later in this section, where a block's children are most
natural to define exactly where they are called.
:end_tab:

:begin_tab:`tensorflow`
Two details do the work here. First,
`self.hidden = tf.keras.layers.Dense(...)` is not an ordinary attribute
assignment: Keras intercepts `__setattr__`, sees that the value is a layer,
and adds it to the tracked children we just inspected. That is why both
layers' variables show up in `net.trainable_variables` with no further
ceremony. Second, we never wrote a backward method; automatic differentiation
derives gradients from whatever `call` computes.
:end_tab:

:begin_tab:`mxnet`
Two details do the work here. First, `self.hidden = nn.Dense(...)` is not an
ordinary attribute assignment: `nn.Block` intercepts `__setattr__`, sees that
the value is a block, and adds it to the registry we just inspected. That is
why both layers' parameters show up in `net.collect_params()` with no further
ceremony. Second, we never wrote a backward method; automatic differentiation
derives gradients from whatever `forward` computes.
:end_tab:

:begin_tab:`pytorch`
Note also that we invoke the model as `net(X)`, never `net.forward(X)`.
Calling a module runs `nn.Module.__call__`, which calls `forward` *and* any
hooks registered on the module. That gap between call and forward is where
model-inspection tooling attaches; we use it in :numref:`sec_repro`.
:end_tab:

:begin_tab:`jax`
Note also that we never call `__call__` directly on a bare module:
`net.apply(params, X)` binds the parameters to the module tree and then runs
`__call__`. That gap between `apply` and `__call__` is where model-inspection
tooling attaches; `apply` can, for instance, be asked to record every
submodule's output (`capture_intermediates`), which we use in
:numref:`sec_repro`.
:end_tab:

:begin_tab:`tensorflow`
Note also that we invoke the model as `net(X)`, never `net.call(X)`. Calling
a model runs Keras's `__call__`, which first *builds* the model if it has not
been built yet, allocating variables from the input shape (a mechanism we
examine at the end of this section), and then runs `call`. Keeping that gap
between `__call__` and `call` in mind explains most of what follows.
:end_tab:

:begin_tab:`mxnet`
Note also that we invoke the model as `net(X)`, never `net.forward(X)`.
Calling a block runs `Block.__call__`, which calls `forward` *and* any hooks
registered on the block. That gap between call and forward is where
model-inspection tooling attaches; we use it in :numref:`sec_repro`.
:end_tab:

## Sequential and Friends: Containers
:label:`subsec_model-construction-sequential`

:begin_tab:`pytorch`
To see that there is no magic left in `nn.Sequential`, we can write it
ourselves. Two ingredients suffice: register each child under a name, and loop
over the children in `forward`.
:end_tab:

:begin_tab:`jax`
To see that there is no magic left in `nn.Sequential`, we can write it
ourselves. Two ingredients suffice: declare a field that holds the list of
children, and loop over them in `__call__`.
:end_tab:

:begin_tab:`tensorflow`
To see that there is no magic left in `Sequential`, we can write it
ourselves. Two ingredients suffice: store the children in an attribute, and
loop over them in `call`.
:end_tab:

:begin_tab:`mxnet`
To see that there is no magic left in `nn.Sequential`, we can write it
ourselves. Two ingredients suffice: register each child, and loop over the
children in `forward`. One Gluon 2.0 wrinkle: the registry holds weak
references, so our class also keeps the blocks in a plain list to keep them
alive.
:end_tab:

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

```{.python .input #model-construction-sequential-and-friends-containers-1}
%%tab jax
class MySequential(nn.Module):
    modules: list

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input #model-construction-sequential-and-friends-containers-1}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input #model-construction-sequential-and-friends-containers-1}
%%tab mxnet
class MySequential(nn.Block):
    def __init__(self):
        super().__init__()
        # The registry holds weakrefs; this list keeps the blocks alive
        self._layers = []

    def add(self, block):
        self._layers.append(block)
        self.register_child(block)

    def forward(self, X):
        for block in self._layers:
            X = block(X)
        return X
```

:begin_tab:`pytorch`
`add_module` writes a child into the registry under a string name (that is
where the `'0'`, `'1'`, `'2'` keys above came from), and `children()` iterates
the registry in insertion order. Our version is a drop-in replacement:
:end_tab:

:begin_tab:`jax`
`modules: list` is a dataclass field: a linen module *is* a Python dataclass,
so the constructor argument, the attribute, and the field declaration are all
the same thing. Flax scans every field for submodules, looking inside lists
and dictionaries, so both `Dense` children are tracked (`nn.relu` is a plain
function, with nothing to track) and appear in the pytree under the field's
name, as `modules_0` and `modules_2`. Our version is a drop-in replacement:
:end_tab:

:begin_tab:`tensorflow`
`self.modules = args` looks like an ordinary assignment, and that is the
point: the `__setattr__` interception scans whatever is assigned, looking
inside lists, tuples, and dictionaries, so both `Dense` children are tracked
and appear in `net.layers`. Our version is a drop-in replacement:
:end_tab:

:begin_tab:`mxnet`
`register_child` writes a child into the registry under a string name (that
is where the `(0)` and `(1)` keys above came from; attribute assignment calls
it with the attribute name instead). Our version is a drop-in replacement:
:end_tab:

```{.python .input #model-construction-sequential-and-friends-containers-2}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input #model-construction-sequential-and-friends-containers-2}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

```{.python .input #model-construction-sequential-and-friends-containers-2}
%%tab tensorflow
net = MySequential(tf.keras.layers.Dense(256, activation='relu'),
                   tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input #model-construction-sequential-and-friends-containers-2}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

:begin_tab:`pytorch`
The registration step is easy to lose. The following module looks reasonable,
and its forward pass works, so nothing appears wrong:
:end_tab:

:begin_tab:`jax`
In imperative frameworks this registration step is famously easy to lose:
store the children in a plain Python list instead of the framework's dedicated
container, and their parameters silently vanish from the model. Flax closes
that trap. Because the field scan looks inside lists and dictionaries, a plain
list assigned in `setup` is tracked like any other child:
:end_tab:

:begin_tab:`tensorflow`
In older imperative frameworks this registration step is famously easy to
lose: store the children in a plain Python list instead of the framework's
dedicated container, and their parameters silently vanish from the model.
Keras closes that trap. Because the attribute scan looks inside lists and
dictionaries, a plain list of layers is tracked like any other child:
:end_tab:

:begin_tab:`mxnet`
The registration step is easy to lose. The following block stores its layers
in a plain Python list, which the `__setattr__` interception ignores, so
nothing inside it is registered. Since `net.initialize()` walks the registry,
it cannot reach the hidden children either, so we initialize each layer by
hand just to get a forward pass, and the model appears to work:
:end_tab:

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

```{.python .input #model-construction-sequential-and-friends-containers-3}
%%tab jax
class ListMLP(nn.Module):
    def setup(self):
        self.layers = [nn.Dense(256), nn.Dense(10)]

    def __call__(self, X):
        return self.layers[1](nn.relu(self.layers[0](X)))

net = ListMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape, sum(x.size for x in jax.tree_util.tree_leaves(params))
```

```{.python .input #model-construction-sequential-and-friends-containers-3}
%%tab tensorflow
class ListMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.blocks = [tf.keras.layers.Dense(256, activation='relu'),
                       tf.keras.layers.Dense(10)]

    def call(self, X):
        for block in self.blocks:
            X = block(X)
        return X

net = ListMLP()
net(X).shape, sum(int(tf.size(v)) for v in net.trainable_variables)
```

```{.python .input #model-construction-sequential-and-friends-containers-3}
%%tab mxnet
class PlainListMLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Dense(256, activation='relu'), nn.Dense(10)]
        for layer in self.layers:
            layer.initialize()  # By hand: net.initialize() cannot see them

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

net = PlainListMLP()
net(X).shape, len(net.collect_params())
```

:begin_tab:`pytorch`
The model computes, yet it owns zero parameters. A plain Python list is not a
module, so the `__setattr__` interception ignores it and nothing inside it is
registered. The layers still hold perfectly good tensors, which is why
`forward` runs. But `net.parameters()` is empty: hand this model to an
optimizer and training proceeds without a single error while updating nothing;
`net.to(device)` leaves every layer behind on the CPU; `net.state_dict()`
checkpoints an empty model. Because no step raises an exception, this bug is
usually diagnosed by staring at a loss curve that refuses to move.
:end_tab:

:begin_tab:`pytorch`
The registered container for a list of children is `nn.ModuleList`. Wrapping
the existing list is the entire fix:
:end_tab:

:begin_tab:`jax`
All 7946 parameters are present; there is no broken variant of this model to
show. The declaration mistake Flax does have is different in kind and,
more usefully, in loudness. Every attribute a module uses must be either a
declared dataclass field or created in `setup`; a bare class attribute is
neither, so the generated constructor does not accept a value for it:
:end_tab:

:begin_tab:`tensorflow`
All 7946 parameters are present; there is no broken variant of this model to
show. (We named the list `blocks` only because `layers` is reserved: Keras
maintains that attribute itself, and assigning to it raises an error.) The
structural mistake Keras does guard against is different in kind and, more
usefully, in loudness. Building is a commitment: once a model's variables
exist, its set of children is locked, and attaching a new layer to a built
model fails at the assignment itself:
:end_tab:

:begin_tab:`mxnet`
The model computes, yet it owns zero parameters: hand it to a `Trainer` and
training updates nothing; `save_parameters` checkpoints an empty model. Gluon
does not let this pass in silence, though. `collect_params()` scans the
block's ordinary attributes for lists, tuples, and dictionaries holding
unregistered blocks, and the call above printed a warning naming the guilty
attribute (`"PlainListMLP.layers" is an unregistered container with
Blocks...`) along with the fix. And had we skipped the by-hand
initialization, the first forward pass would have stopped with a
`Parameter has not been initialized` error instead of running at all. Where
PyTorch's version of this bug is diagnosed by staring at a loss curve,
Gluon's announces itself.
:end_tab:

:begin_tab:`mxnet`
There is no dedicated list container to reach for, since Gluon has no analog
of `nn.ModuleList`; the warning already named the fix. Registering each
child, exactly what our `MySequential.add` did, is the entire repair:
:end_tab:

```{.python .input #model-construction-sequential-and-friends-containers-4}
%%tab pytorch
class ModuleListMLP(PlainListMLP):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(self.layers)

net = ModuleListMLP()
net(X).shape, sum(p.numel() for p in net.parameters())
```

```{.python .input #model-construction-sequential-and-friends-containers-4}
%%tab jax
class NoFieldSequential(nn.Module):
    modules = []  # Missing the `: list` annotation, so not a field

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X

try:
    net = NoFieldSequential([nn.Dense(256), nn.relu, nn.Dense(10)])
except TypeError as e:
    print(e)
```

```{.python .input #model-construction-sequential-and-friends-containers-4}
%%tab tensorflow
try:
    net.head = tf.keras.layers.Dense(2)  # net is built: too late
except ValueError as e:
    print(e)
```

```{.python .input #model-construction-sequential-and-friends-containers-4}
%%tab mxnet
class RegisteredListMLP(PlainListMLP):
    def __init__(self):
        super().__init__()
        for layer in self.layers:
            self.register_child(layer)

net = RegisteredListMLP()
net(X).shape, sum(p.data().size for p in net.collect_params().values())
```

:begin_tab:`pytorch`
Same forward pass, 7946 registered parameters. The division of labor among the
containers is now clear. `nn.Sequential` registers its children and supplies
the run-them-in-order `forward`. `nn.ModuleList` registers a list of children
but supplies no `forward` at all; you keep writing the loop, which is exactly
what you want when the loop body is not a plain chain (blocks that take extra
arguments, or a skip connection around each block). `nn.ModuleDict` does the
same for children indexed by name. Transformer implementations conventionally
keep their stack of blocks in an `nn.ModuleList` and their named parts
(embedding, final normalization, output head) as attributes.
:end_tab:

:begin_tab:`jax`
Flax therefore ships no special list or dict containers, because none are
needed: any pytree of modules, in a field or assigned in `setup`, is tracked,
and a misdeclared child fails at construction time instead of yielding a model
that runs but trains nothing. Transformer implementations in Flax
conventionally keep their stack of blocks in exactly the kind of plain list
`MySequential` used, with the named parts (embedding, final normalization,
output head) as separate attributes.
:end_tab:

:begin_tab:`tensorflow`
The error message states the rule Keras enforces: all state must be created
in `__init__` or in `build`, never after. Keras therefore ships no special
list or dict containers, because none are needed: any layer reachable from an
attribute is tracked, and a late structural edit fails at assignment time
rather than being silently ignored. Transformer implementations in Keras
conventionally keep their stack of blocks in exactly the kind of plain list
`ListMLP` used, with the named parts (embedding, final normalization, output
head) as separate attributes.
:end_tab:

:begin_tab:`mxnet`
Same forward pass, 7946 registered parameters, and no warning. Gluon ships no
`ModuleList` or `ModuleDict` equivalents: the two registered homes for
children are named attributes and `nn.Sequential`, with `register_child` as
the escape hatch for anything else, such as a list of blocks whose loop body
is not a plain chain. Transformer implementations in Gluon conventionally
keep their stack of blocks in an `nn.Sequential` (or register them one by
one) and their named parts (embedding, final normalization, output head) as
attributes.
:end_tab:

## Forward Is Just Python

:begin_tab:`pytorch`
`forward` is an ordinary Python method. Nothing restricts it to chaining
children: it can branch, loop, call any tensor function, and combine
intermediate results however it likes. The loop in `ModuleListMLP` already used
this freedom. Its most consequential one-line use is the *residual
connection*, the wiring idiom at the heart of ResNets and Transformers alike:
:end_tab:

:begin_tab:`jax`
`__call__` is an ordinary Python method. Nothing restricts it to chaining
children: it can branch, loop, call any `jnp` function, and combine
intermediate results however it likes. The loop in `MySequential` already used
this freedom. Its most consequential one-line use is the *residual
connection*, the wiring idiom at the heart of ResNets and Transformers alike.
Since the block's body is most natural to define at the point of use, we write
this one in the inline style: `@nn.compact` lets `__call__` create its
children as it runs.
:end_tab:

:begin_tab:`tensorflow`
`call` is an ordinary Python method. Nothing restricts it to chaining
children: it can branch, loop, call any TensorFlow function, and combine
intermediate results however it likes; TensorFlow executes eagerly, so all of
this runs one operation at a time, just as in NumPy. (Once a model is wrapped
in `tf.function` for speed, as the `Trainer` from :numref:`sec_oo-design`
does, AutoGraph rewrites such control flow into graph form.) The loop in
`MySequential` already used this freedom. Its most consequential one-line use
is the *residual connection*, the wiring idiom at the heart of ResNets and
Transformers alike:
:end_tab:

:begin_tab:`mxnet`
`forward` is an ordinary Python method. Nothing restricts it to chaining
children: it can branch, loop, call any `np` function, and combine
intermediate results however it likes; a Gluon `Block` executes eagerly, so
all of this runs one operation at a time, just as in NumPy. (Gluon's
`HybridBlock` can compile the forward computation into a graph for speed, at
the price of restricting it to traceable operations; we stay with `Block`
here.) The loop in `MySequential` already used this freedom. Its most
consequential one-line use is the *residual connection*, the wiring idiom at
the heart of ResNets and Transformers alike:
:end_tab:

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

```{.python .input #model-construction-forward-is-just-python-1}
%%tab jax
class ResidualBlock(nn.Module):
    num_hiddens: int

    @nn.compact
    def __call__(self, X):
        body = nn.Sequential([nn.Dense(self.num_hiddens), nn.relu,
                              nn.Dense(self.num_hiddens)])
        return X + body(X)
```

```{.python .input #model-construction-forward-is-just-python-1}
%%tab tensorflow
class ResidualBlock(tf.keras.Model):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_hiddens)])

    def call(self, X):
        return X + self.body(X)
```

```{.python .input #model-construction-forward-is-just-python-1}
%%tab mxnet
class ResidualBlock(nn.Block):
    def __init__(self, num_hiddens):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(nn.Dense(num_hiddens, activation='relu'),
                      nn.Dense(num_hiddens))

    def forward(self, X):
        return X + self.body(X)
```

![The residual wiring `X + body(X)`: the input splits at a branch point into the body stack and an identity skip, and the two rejoin by addition before the block's output.](../img/bg-residual-block.svg)
:label:`fig_bg_residual-block`

:begin_tab:`pytorch`
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
:end_tab:

:begin_tab:`jax`
`X + body(X)` is not a layer Flax provides. It is arithmetic in `__call__`,
and it changes what the block *is*: the block computes a perturbation of the
identity function rather than an arbitrary transformation, and during
backpropagation the skip path hands gradients to earlier layers undiminished,
tempering the vanishing gradients of :numref:`sec_numerical_stability`.
:numref:`fig_bg_residual-block` diagrams exactly this wiring. Chapter 8
develops both points when we build ResNet; for now we only need the mechanics.
One mechanical consequence is visible already: the addition forces the input
and output shapes to agree, so a residual block has a single width that is
part of its identity. That is why `num_hiddens` is a declared field of the
block rather than a width left for `init` to infer.
:end_tab:

:begin_tab:`tensorflow`
`X + self.body(X)` is not a layer Keras provides. It is arithmetic in `call`,
and it changes what the block *is*: the block computes a perturbation of the
identity function rather than an arbitrary transformation, and during
backpropagation the skip path hands gradients to earlier layers undiminished,
tempering the vanishing gradients of :numref:`sec_numerical_stability`.
:numref:`fig_bg_residual-block` diagrams exactly this wiring. Chapter 8
develops both points when we build ResNet; for now we only need the
mechanics. One mechanical consequence is visible already: the addition forces
the input and output shapes to agree, so a residual block has a single width
that is part of its identity. Keras always infers input widths at build time,
so here the constraint falls on the caller: feed the block anything other
than `num_hiddens` columns and the addition fails.
:end_tab:

:begin_tab:`mxnet`
`X + self.body(X)` is not a layer Gluon provides. It is arithmetic in
`forward`, and it changes what the block *is*: the block computes a
perturbation of the identity function rather than an arbitrary
transformation, and during backpropagation the skip path hands gradients to
earlier layers undiminished, tempering the vanishing gradients of
:numref:`sec_numerical_stability`. :numref:`fig_bg_residual-block` diagrams
exactly this wiring. Chapter 8 develops both points when we build ResNet; for
now we only need the mechanics. One mechanical consequence is visible
already: the addition forces the input and output shapes to agree, so a
residual block has a single width that is part of its identity. Gluon defers
input widths to the first forward pass (a mechanism we examine at the end of
this section), so here the constraint falls on the caller: feed the block
anything other than `num_hiddens` columns and the addition fails.
:end_tab:

```{.python .input #model-construction-forward-is-just-python-2}
%%tab pytorch
block = ResidualBlock(24)
block(torch.randn(2, 24)).shape
```

```{.python .input #model-construction-forward-is-just-python-2}
%%tab jax
block = ResidualBlock(24)
X24 = jax.random.normal(d2l.get_key(), (2, 24))
params = block.init(d2l.get_key(), X24)
block.apply(params, X24).shape
```

```{.python .input #model-construction-forward-is-just-python-2}
%%tab tensorflow
block = ResidualBlock(24)
block(tf.random.normal((2, 24))).shape
```

```{.python .input #model-construction-forward-is-just-python-2}
%%tab mxnet
block = ResidualBlock(24)
block.initialize()
block(np.random.normal(size=(2, 24))).shape
```

:begin_tab:`pytorch`
`forward` may also use state that is neither an input nor a parameter. Suppose
we want to damp each block's contribution by a fixed factor:
:end_tab:

:begin_tab:`jax`
`__call__` may also use state that is neither an input nor a parameter.
Suppose we want to damp each block's contribution by a fixed factor:
:end_tab:

:begin_tab:`tensorflow`
`call` may also use state that is neither an input nor a parameter. Suppose
we want to damp each block's contribution by a fixed factor:
:end_tab:

:begin_tab:`mxnet`
`forward` may also use state that is neither an input nor a parameter.
Suppose we want to damp each block's contribution by a fixed factor:
:end_tab:

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

```{.python .input #model-construction-forward-is-just-python-3}
%%tab jax
class ScaledResidual(nn.Module):
    num_hiddens: int
    alpha: float = 0.5  # Fixed by design, never trained

    @nn.compact
    def __call__(self, X):
        body = nn.Sequential([nn.Dense(self.num_hiddens), nn.relu,
                              nn.Dense(self.num_hiddens)])
        return X + self.alpha * body(X)

block = ScaledResidual(24)
params = block.init(d2l.get_key(), X24)
block.alpha, list(params['params'])
```

```{.python .input #model-construction-forward-is-just-python-3}
%%tab tensorflow
class ScaledResidual(ResidualBlock):
    def __init__(self, num_hiddens, alpha=0.5):
        super().__init__(num_hiddens)
        self.alpha = tf.constant(alpha)  # Fixed by design, never trained

    def call(self, X):
        return X + self.alpha * self.body(X)

block = ScaledResidual(24)
block(tf.random.normal((2, 24)))
any('alpha' in w.path for w in block.weights), [w.path for w in block.weights][:2]
```

```{.python .input #model-construction-forward-is-just-python-3}
%%tab mxnet
class ScaledResidual(ResidualBlock):
    def __init__(self, num_hiddens, alpha=0.5):
        super().__init__(num_hiddens)
        self.alpha = np.array(alpha)  # Fixed by design, never trained

    def forward(self, X):
        return X + self.alpha * self.body(X)

block = ScaledResidual(24)
'alpha' in block.collect_params(), list(block.collect_params())[:2]
```

:begin_tab:`pytorch`
`alpha` enters the computation, but it is not a parameter: it never appears in
`named_parameters()`, so the optimizer never touches it. That much we wanted.
Storing it as a plain attribute has a cost we did not want, though: as the
output shows, it is missing from `state_dict()` as well, so it will not be
saved with the model, and `.to(device)` will not move it. Some state is not a
parameter but must still travel with the model; the registered home for such
state is a *buffer*, introduced in :numref:`sec_parameters`.
:end_tab:

:begin_tab:`jax`
`alpha` enters the computation, but it is not a parameter: the pytree contains
only the two `Dense` children, so the optimizer never touches it. Nor can it
fail to travel with the model, since it is a dataclass field, part of the
architecture itself, and constructing `ScaledResidual(24, alpha=0.5)`
reproduces it exactly. What fields cannot express is non-parameter state that
*changes* during the forward pass, such as running statistics; Flax gives such
state an explicit home in a separate variable collection, introduced in
:numref:`sec_parameters`.
:end_tab:

:begin_tab:`tensorflow`
`alpha` enters the computation, but it is not a parameter: as the output
shows, `block.weights` contains only the `Dense` variables, so the optimizer
never touches it. That much we wanted. Storing it as a `tf.constant` has a
cost we did not want, though: not being a variable, it is invisible to weight
checkpointing as well, so it will not be saved with the model. Some state is
not a parameter but must still travel with the model; the registered home for
such state is a non-trainable `tf.Variable`, introduced in
:numref:`sec_parameters`.
:end_tab:

:begin_tab:`mxnet`
`alpha` enters the computation, but it is not a parameter: as the output
shows, it never appears in `collect_params()`, so the `Trainer` never touches
it. That much we wanted. Storing it as a plain attribute has a cost we did
not want, though: invisible to the parameter walk, it will not be saved by
`save_parameters`, and it will not move to a GPU with the rest of the block.
Some state is not a parameter but must still travel with the model; the
registered home for such state is a constant parameter (`gluon.Constant`),
introduced in :numref:`sec_parameters`.
:end_tab:

:begin_tab:`jax`
One more freedom deserves a demonstration, because JAX users are often warned
they lose it: data-dependent control flow. Outside of `jax.jit`, `init` and
`apply` execute eagerly, one operation at a time, so a Python `while` loop
whose condition depends on the data works exactly as it would in NumPy:
:end_tab:

```{.python .input #model-construction-forward-is-just-python-4}
%%tab jax
class HalvingMLP(nn.Module):
    @nn.compact
    def __call__(self, X):
        X = nn.Dense(24)(X)
        while jnp.abs(X).sum() > 1:  # Ordinary Python control flow
            X = X / 2
        return X.sum()

net = HalvingMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

:begin_tab:`jax`
Once a model is wrapped in `jax.jit` for speed, such data-dependent Python
control flow must be expressed with `jax.lax` primitives instead; that
constraint arrives only with the compiler, not with the module abstraction.
:end_tab:

## Lazy Initialization: Shapes from Data
:label:`sec_lazy_init`

:begin_tab:`pytorch`
We have been doing something odd since our first MLP without commenting on it:
`nn.LazyLinear(256)` names only the layer's *output* width. Its weight matrix
has shape `(256, in_features)`, and we never said what `in_features` is.
The layer cannot know it at construction time, since it depends on the data it
will receive. So it does not allocate parameters at construction time at all:
:end_tab:

:begin_tab:`jax`
We have been doing something odd since our first MLP without commenting on it:
`nn.Dense(256)` names only the layer's *output* width. Its kernel has shape
`(in_features, 256)`, and we never said what `in_features` is. The layer
cannot know it at construction time, since it depends on the data it will
receive. Flax resolves this without any special lazy machinery, because its
parameters are never allocated at construction time anyway: they come into
existence only inside `init(key, X)`, and that call has the input in hand.
:end_tab:

:begin_tab:`tensorflow`
We have been doing something odd since our first MLP without commenting on
it: `Dense(256)` names only the layer's *output* width. Its kernel has shape
`(in_features, 256)`, and we never said what `in_features` is. The layer
cannot know it at construction time, since it depends on the data it will
receive. So Keras never allocates variables at construction time: every layer
has a `build(input_shape)` method that creates them, and `__call__` invokes
it the first time data arrives. Deferred building is not a special mode; it
is how every Keras layer works.
:end_tab:

:begin_tab:`mxnet`
We have been doing something odd since our first MLP without commenting on
it: `nn.Dense(256)` names only the layer's *output* width. Its weight matrix
has shape `(256, in_units)`, and we never said what `in_units` is. The layer
cannot know it at construction time, since it depends on the data it will
receive. So Gluon does not allocate parameters at construction time at all.
Deferred initialization is not a special mode or a dedicated `Lazy*` class;
it is how every Gluon layer works, and `in_units` is simply an optional
argument:
:end_tab:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-1}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net[0].weight
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-1}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(10)])
net.weights
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-1}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.collect_params()
```

:begin_tab:`pytorch`
The weight is a placeholder. The first time data flows through, the layer
reads the input width from the batch, allocates and initializes a real weight,
and replaces itself with a plain `nn.Linear`. Its output width then fixes the
input of the next lazy layer, and shapes cascade through the whole model:
:end_tab:

:begin_tab:`jax`
There is consequently nothing to inspect before initialization: constructing
`net` records the architecture and nothing else. During `init`, the dummy
input flows through the model, each `Dense` reads its input width from the
incoming shape and allocates a real kernel, and its declared output width
fixes the input of the next layer, so shapes cascade through the whole model
in one call:
:end_tab:

:begin_tab:`tensorflow`
There are no variables at all yet, and accessing `net.layers[0].kernel` would
raise an error telling us to build the layer first. The first time data flows
through, each layer's `build` reads the input width from the incoming shape
and allocates and initializes a real kernel, and the layer's declared output
width fixes the input of the next layer, so shapes cascade through the whole
model:
:end_tab:

:begin_tab:`mxnet`
Each weight is a placeholder: the parameter objects exist, but the special
value -1 marks the input dimension as unknown, and reading
`net[0].weight.data()` would raise an error telling us to initialize the
network first. Even `initialize()` allocates nothing at this point; it only
records which initializer (and device) to use once the shapes are known. The
first time data flows through, each layer reads its input width from the
incoming shape, allocates and initializes a real weight, and its declared
output width fixes the input of the next layer, so shapes cascade through the
whole model:
:end_tab:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-2}
%%tab pytorch
net(X)
net[0].weight.shape
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-2}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
jax.tree_util.tree_map(lambda x: x.shape, params)
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-2}
%%tab tensorflow
net(X)
[w.shape for w in net.weights]
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-2}
%%tab mxnet
net.initialize()
net(X)
net.collect_params()
```

We build models this way from Chapter 7 on because it removes shape arithmetic
from model definitions. In a convolutional network the flattened feature-map
size depends on the input resolution and on every upstream stride and padding
choice; computing it by hand clutters the code, and every architecture edit
invalidates the numbers downstream of it. Declaring output widths and letting
input widths come from data ends that bookkeeping.

:begin_tab:`pytorch`
The convenience comes with one rule: until the first forward pass, the
parameters *do not exist*. Anything that needs the parameter list, whether
constructing an optimizer, applying an initializer, or counting parameters,
must happen after a *dry run* on a representative batch. A related subtlety:
the random initialization now happens at first call rather than at
construction, so any random numbers your program draws in between shift the
generator's state, and a fixed seed can yield different weights than the
explicitly shaped version of the same model (:numref:`sec_repro` returns to
seeding). Other libraries make the dry run explicit rather than implicit: a
Flax module in JAX has no lazy mode at all, and its parameters exist only
after a mandatory `init(key, dummy_input)` call performs the same shape
inference. PyTorch's lazy layers give you that behavior with the first real
batch playing the role of the dummy input.
:end_tab:

:begin_tab:`jax`
Note what did not need to be said: there is no rule about *when* parameters
become available, because `init` is mandatory and everything that needs
parameters, whether constructing the optimizer state, counting them, or
transforming them, starts from its return value. The ordering mistakes that
implicit lazy initialization invites cannot arise. Randomness is equally
explicit: which weights `init` produces depends on the PRNG key we pass it
and on nothing else, not on how many random numbers the program happened to
draw beforehand (:numref:`sec_repro` returns to seeding).
:end_tab:

:begin_tab:`tensorflow`
The convenience comes with one rule: until the first call, the variables *do
not exist*. Anything that needs the variable list, whether constructing an
optimizer, reading a weight, or counting parameters, must happen after the
model is built. Keras offers two ways to get there: a dry run on a
representative batch, or `net.build((None, 20))`, which propagates shapes
through the model without any data (`None` marks the batch dimension). Two
consequences of build-time allocation are worth remembering. Initialization
happens at build rather than at construction, so under a fixed seed the
weights you get depend on how many random numbers the program drew before the
model was built (:numref:`sec_repro` returns to seeding). And, as the
containers lesson showed, building is also the moment the model's structure
locks.
:end_tab:

:begin_tab:`mxnet`
The convenience comes with one rule: until the first forward pass, the
parameters have no shapes and no values. Anything that needs them, whether
creating a `Trainer`, reading a weight, or counting parameters, must happen
after a *dry run* on a representative batch. A related subtlety: the random
draw now happens at first call rather than at `initialize()`, so any random
numbers your program draws in between shift the generator's state, and a
fixed seed can yield different weights than an explicitly shaped version of
the same model (:numref:`sec_repro` returns to seeding). Other libraries
retrofitted this behavior; Gluon was designed around it, which is why
`initialize()` is a separate, explicit step rather than something the
constructor does.
:end_tab:

:begin_tab:`pytorch`
The dry run is such a common preamble that we fold it into the `d2l.Module`
base class from :numref:`sec_oo-design`: run the model once to materialize
every shape, then optionally apply an initialization function.
:end_tab:

:begin_tab:`jax`
The `init` call is such a common preamble that we fold it into the
`d2l.Module` base class from :numref:`sec_oo-design`: `apply_init` takes the
dummy input as a list of arguments plus a PRNG key and returns the
initialized parameters. Our training loop calls it once, before the optimizer
state is constructed.
:end_tab:

:begin_tab:`tensorflow`
There is no initialization pass to fold into `d2l.Module` for Keras, because
a non-default initializer is part of the layer's definition rather than a
step applied afterwards: each layer accepts a `kernel_initializer` argument,
and `build` invokes it when it allocates the kernel. Xavier's uniform variant
(:numref:`subsec_xavier`) is in fact the Keras default, under the name
`glorot_uniform`. A small demonstration, spelling the default out explicitly:
:end_tab:

:begin_tab:`mxnet`
There is no dry-run helper to fold into `d2l.Module` for Gluon, because
`initialize()` already plays that role: it takes the initializer as an
argument, records it against every parameter, and lets the first forward pass
apply it once the shapes arrive. Xavier's uniform variant
(:numref:`subsec_xavier`) is available as `init.Xavier()`. A small
demonstration, combining the recorded initializer with the dry run that
completes it:
:end_tab:

```{.python .input #model-construction-lazy-initialization-shapes-from-data-3}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-3}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

:begin_tab:`pytorch`
`nn.Module.apply(fn)` calls `fn` on every module in the tree, children first.
It is the standard way to push a policy across an arbitrary model, one more
operation that is a tree walk, and from Chapter 7 on the idiom
`model.apply_init([X], init)` opens most of our training scripts. A small
demonstration:
:end_tab:

:begin_tab:`jax`
Where does a non-default initializer fit? In Flax it is part of the layer's
definition rather than a pass applied afterwards: each `Dense` accepts a
`kernel_init` function, and `init` invokes it when it allocates the kernel.
From Chapter 7 on the idiom `params = model.apply_init([X], key)` opens most
of our training scripts. A small demonstration:
:end_tab:

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

```{.python .input #model-construction-lazy-initialization-shapes-from-data-4}
%%tab jax
class TinyMLP(d2l.Module):
    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(256, kernel_init=nn.initializers.xavier_uniform()),
            nn.relu, nn.Dense(10)])

model = TinyMLP()
params = model.apply_init([X], key=d2l.get_key())
jax.tree_util.tree_map(lambda x: x.shape, params)
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-4}
%%tab tensorflow
net = tf.keras.Sequential([
    tf.keras.layers.Dense(
        256, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(10)])
net.build((None, 20))
net.layers[0].kernel.shape
```

```{.python .input #model-construction-lazy-initialization-shapes-from-data-4}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Xavier())
net(X)
net[0].weight.data().shape
```

:begin_tab:`pytorch`
The dry run inside `apply_init` turned every lazy layer into a real one, after
which `init_xavier` could match on `nn.Linear` and rewrite its weights. Which
initializer to apply, and why Xavier's variance rule
(:numref:`subsec_xavier`) is a sensible default, is the subject of
:numref:`sec_init_param`.
:end_tab:

:begin_tab:`jax`
`apply_init` materialized every shape and, for the first kernel, drew the
initial values from the Xavier initializer, all in the same call. Which
initializer to use, and why Xavier's variance rule (:numref:`subsec_xavier`)
is a sensible default, is the subject of :numref:`sec_init_param`.
:end_tab:

:begin_tab:`tensorflow`
`build` materialized every shape and drew the first kernel's initial values
from the Xavier initializer, without a batch of data in sight. Which
initializer to use, and why Xavier's variance rule (:numref:`subsec_xavier`)
is a sensible default, is the subject of :numref:`sec_init_param`.
:end_tab:

:begin_tab:`mxnet`
The dry run materialized every shape and drew the first weight's initial
values from the Xavier initializer that `initialize` had recorded. Which
initializer to use, and why Xavier's variance rule (:numref:`subsec_xavier`)
is a sensible default, is the subject of :numref:`sec_init_param`.
:end_tab:

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

```{.python .input #model-construction-building-from-a-config-1}
%%tab jax
class ResidualMLP(nn.Module):
    d_hidden: int = 256
    num_blocks: int = 4
    d_out: int = 10

    @nn.compact
    def __call__(self, X):
        X = nn.Dense(self.d_hidden)(X)
        for _ in range(self.num_blocks):
            X = ResidualBlock(self.d_hidden)(X)
        return nn.Dense(self.d_out)(X)
```

```{.python .input #model-construction-building-from-a-config-1}
%%tab tensorflow
@dataclass
class MLPConfig:
    d_in: int = 784
    d_hidden: int = 256
    num_blocks: int = 4
    d_out: int = 10

def build(cfg: MLPConfig) -> tf.keras.Model:
    blocks = [ResidualBlock(cfg.d_hidden) for _ in range(cfg.num_blocks)]
    return tf.keras.Sequential([tf.keras.Input((cfg.d_in,)),
                                tf.keras.layers.Dense(cfg.d_hidden),
                                *blocks,
                                tf.keras.layers.Dense(cfg.d_out)])
```

```{.python .input #model-construction-building-from-a-config-1}
%%tab mxnet
@dataclass
class MLPConfig:
    d_in: int = 784
    d_hidden: int = 256
    num_blocks: int = 4
    d_out: int = 10

def build(cfg: MLPConfig) -> nn.Block:
    net = nn.Sequential()
    net.add(nn.Dense(cfg.d_hidden, in_units=cfg.d_in))
    for _ in range(cfg.num_blocks):
        net.add(ResidualBlock(cfg.d_hidden))
    net.add(nn.Dense(cfg.d_out, in_units=cfg.d_hidden))
    return net
```

:begin_tab:`pytorch`
One config produces one architecture: an input projection into the hidden
width, `num_blocks` identical residual blocks, and an output projection. The
list comprehension producing `blocks` is a plain Python list, which is safe
here for the reason :numref:`subsec_model-construction-sequential` taught:
unpacking it into `nn.Sequential` is what registers each block. Every layer
gets explicit widths, and this is where explicit shapes beat lazy ones: the
config already knows every width, so explicit construction yields a fully
materialized model with no dry run needed. Printing the model displays the
module tree:
:end_tab:

:begin_tab:`jax`
Here the config *is* the model class. A linen module is a dataclass, so the
fields `d_hidden`, `num_blocks`, and `d_out` are simultaneously the
configuration record and the constructor signature: `ResidualMLP()` is the
default architecture, `ResidualMLP(d_hidden=512, num_blocks=8)` a rescaled
one, and there is no separate config object or `build` function to keep in
sync with the model code. The architecture itself is the same: an input
projection into the hidden width, `num_blocks` identical residual blocks, and
an output projection, with `@nn.compact` creating each block inline as the
loop runs. There is no input-width field at all, since `init` reads it from
the dummy input. Initializing the model and mapping shapes over the result
displays the module tree as a pytree:
:end_tab:

:begin_tab:`tensorflow`
One config produces one architecture: an input projection into the hidden
width, `num_blocks` identical residual blocks, and an output projection. The
free-standing `build(cfg)` function is unrelated to the `build` method Keras
calls on layers, but the two meet here: because the config knows the input
width, we can declare it with `tf.keras.Input`, and `Sequential` then builds
the whole model at construction time, no dry run needed. This is where
explicit shapes beat inferred ones. `net.summary()` displays the module tree,
every shape and parameter count already known:
:end_tab:

:begin_tab:`mxnet`
One config produces one architecture: an input projection into the hidden
width, `num_blocks` identical residual blocks, and an output projection. This
is where explicit shapes beat deferred ones: the config already knows every
width, so the two projections take `in_units` directly and skip deferred
initialization. (The `Dense` layers inside each `ResidualBlock` were written
without `in_units`, so the model as a whole still wants one dry run before
its parameters can be counted.) Printing the model displays the module tree:
:end_tab:

```{.python .input #model-construction-building-from-a-config-2}
%%tab pytorch
net = build(MLPConfig())
net
```

```{.python .input #model-construction-building-from-a-config-2}
%%tab jax
net = ResidualMLP()
params = net.init(d2l.get_key(), jnp.zeros((2, 784)))
jax.tree_util.tree_map(lambda x: x.shape, params)
```

```{.python .input #model-construction-building-from-a-config-2}
%%tab tensorflow
net = build(MLPConfig())
net.summary()
```

```{.python .input #model-construction-building-from-a-config-2}
%%tab mxnet
net = build(MLPConfig())
net.initialize()
net
```

```{.python .input #model-construction-building-from-a-config-3}
%%tab pytorch
net(torch.rand(2, 784)).shape
```

```{.python .input #model-construction-building-from-a-config-3}
%%tab jax
net.apply(params, jax.random.uniform(d2l.get_key(), (2, 784))).shape
```

```{.python .input #model-construction-building-from-a-config-3}
%%tab tensorflow
net(tf.random.uniform((2, 784))).shape
```

```{.python .input #model-construction-building-from-a-config-3}
%%tab mxnet
net(np.random.uniform(size=(2, 784))).shape
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

```{.python .input #model-construction-building-from-a-config-4}
%%tab jax
for net in (ResidualMLP(), ResidualMLP(d_hidden=512, num_blocks=8)):
    params = net.init(d2l.get_key(), jnp.zeros((2, 784)))
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'd_hidden={net.d_hidden}, num_blocks={net.num_blocks}: '
          f'{n:,} parameters')
```

```{.python .input #model-construction-building-from-a-config-4}
%%tab tensorflow
for cfg in (MLPConfig(), MLPConfig(d_hidden=512, num_blocks=8)):
    print(f'd_hidden={cfg.d_hidden}, num_blocks={cfg.num_blocks}: '
          f'{build(cfg).count_params():,} parameters')
```

```{.python .input #model-construction-building-from-a-config-4}
%%tab mxnet
for cfg in (MLPConfig(), MLPConfig(d_hidden=512, num_blocks=8)):
    net = build(cfg)
    net.initialize()
    net(np.zeros((2, cfg.d_in)))  # Dry run: materialize the deferred shapes
    n = sum(p.data().size for p in net.collect_params().values())
    print(f'd_hidden={cfg.d_hidden}, num_blocks={cfg.num_blocks}: '
          f'{n:,} parameters')
```

:begin_tab:`pytorch`
Because `build` is deterministic in `cfg`, the config is all you need to
reconstruct the module tree later; :numref:`sec_read_write` saves it
alongside the weights so that loading a checkpoint starts by rebuilding the
exact same model. A config of widths and depths feeding a loop that stacks
identical residual blocks is, minus attention, the exact shape of every
Transformer implementation you will read.
:end_tab:

:begin_tab:`jax`
Because the module is a dataclass, its fields are all you need to reconstruct
the module tree later; :numref:`sec_read_write` saves them alongside the
parameters so that loading a checkpoint starts by rebuilding the exact same
model. A handful of width and depth fields feeding a loop that stacks
identical residual blocks is, minus attention, the exact shape of every
Transformer implementation you will read.
:end_tab:

:begin_tab:`tensorflow`
Because `build` is deterministic in `cfg`, the config is all you need to
reconstruct the module tree later; :numref:`sec_read_write` saves it
alongside the weights so that loading a checkpoint starts by rebuilding the
exact same model. Keras bakes the same idea into every layer: `get_config()`
returns the constructor arguments needed to re-create the object, and that is
exactly what Keras model serialization records. A config of widths and depths
feeding a loop that stacks identical residual blocks is, minus attention, the
exact shape of every Transformer implementation you will read.
:end_tab:

:begin_tab:`mxnet`
Because `build` is deterministic in `cfg`, the config is all you need to
reconstruct the module tree later; :numref:`sec_read_write` saves it
alongside the weights so that loading a checkpoint starts by rebuilding the
exact same model. A config of widths and depths feeding a loop that stacks
identical residual blocks is, minus attention, the exact shape of every
Transformer implementation you will read.
:end_tab:

## Summary

:begin_tab:`pytorch`
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
:end_tab:

:begin_tab:`jax`
A module owns child modules and a `__call__` method; its parameters live in a
pytree that `init(key, X)` creates and `apply(params, X)` consumes, with a
structure that mirrors the module tree. Layers, blocks, and whole models are
the same kind of object, and parameter collection, optimizer updates, and
serialization are all traversals of the pytree. Children are discovered by
scanning dataclass fields and `setup` assignments, lists and dictionaries
included; the one declaration mistake, an unannotated field, fails loudly at
construction time. `__call__` is ordinary Python; a residual connection is one
line in it. Every input width is inferred inside the mandatory `init` call
(`apply_init`). Configs need no separate machinery: a linen module is a
dataclass, so its fields are its config.
:end_tab:

:begin_tab:`tensorflow`
A module owns variables, child layers, and a `call` method. Layers, blocks,
and whole models are the same kind of object, so a model is a tree of
modules, and variable collection and serialization are walks over that tree.
The tree is discovered through attribute assignment: Keras scans every
assigned attribute, lists and dictionaries included, so even a plain Python
list of layers is tracked. Variables are created by `build`, not by the
constructor: every layer infers its input width when the first batch arrives
(or when `build(input_shape)` is called), after which the model's structure
is locked. `call` is ordinary Python; a residual connection is one line in
it. Configs turn architecture into data: a `dataclass` of widths and depths
plus a `build(cfg)` function that stacks blocks, with `get_config()` as the
Keras-native form of the same idea.
:end_tab:

:begin_tab:`mxnet`
A module owns parameters, child blocks, and a `forward` method. Layers,
blocks, and whole models are the same kind of object, so a model is a tree of
blocks, and parameter collection, initialization, device movement, and
serialization are all walks over that tree. The tree is discovered through
registration: attribute assignment, `nn.Sequential`'s `add`, and
`register_child` register children, while a plain Python list hides them,
though `collect_params()` warns about the hidden blocks by name. `forward` is
ordinary Python; a residual connection is one line in it. Deferred
initialization is Gluon's native mode: every layer declares its output width,
`initialize()` records the initializer, and the first forward pass infers
input widths and allocates the parameters, so inspection and training follow
a dry run. Configs turn architecture into data: a `dataclass` of widths and
depths plus a `build` function that stacks blocks.
:end_tab:

## Exercises

:begin_tab:`pytorch`
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
:end_tab:

:begin_tab:`mxnet`
1. Take `PlainListMLP` and catalog everything that breaks besides the empty
   parameter dictionary. Check `net.save_parameters(...)`, what
   `net.initialize()` reaches, and what happens on the first forward pass if
   you remove the by-hand initialization from the constructor. Explain how
   each failure follows from the same missing registration, and note where
   along the way the `collect_params()` warning fires.
1. Implement a `ParallelBlock` that takes two child blocks `net1` and `net2`,
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
:end_tab:

:begin_tab:`jax`
1. Extend `ListMLP` with a *dictionary* of submodules assigned in `setup` and
   verify that every parameter appears in the pytree returned by `init`. Then
   try to hide a `Dense` from `init` after all, by creating it inside
   `__call__` without `@nn.compact`, and describe how the failure announces
   itself.
1. Implement a `ParallelBlock` that takes two child modules `net1` and `net2`
   as fields, runs both on the same input, and concatenates their outputs
   along the last dimension. What must be true of the two children's outputs
   for the concatenation to be valid?
1. Extend `ResidualMLP` with an activation switch (for example,
   `act: str = 'relu'`) and make `__call__` honor it. Which decisions belong
   in fields and which belong in code? Where would you put a choice between
   `ResidualBlock` and a plain feed-forward block?
1. `ResidualBlock` requires its input and output widths to agree. Suppose you
   want a block whose output is wider than its input. Give two standard fixes
   and the cost of each. (Chapter 8 uses one of them in ResNet.)
:end_tab:

:begin_tab:`tensorflow`
1. Keras's tracking has one blind spot left: create a `Dense` layer *inside*
   `call` rather than in the constructor. The model runs without complaint;
   check `len(net.trainable_variables)` after calling it, explain what an
   optimizer would train, and explain what happens to the layer's weights
   between two calls.
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
:end_tab:
