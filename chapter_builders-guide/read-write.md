```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# File I/O

So far we have discussed how to process data and how
to build, train, and test deep learning models.
However, at some point we will hopefully be happy enough
with the learned models that we will want
to save the results for later use in various contexts
(perhaps even to make predictions in deployment).
Additionally, when running a long training process,
the best practice is to periodically save intermediate results (checkpointing)
to ensure that we do not lose several days' worth of computation
if we trip over the power cord of our server.
Thus it is time to learn how to load and store
both individual weight vectors and entire models.
This section addresses both issues.

```{.python .input #read-write-file-i-o}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #read-write-file-i-o}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #read-write-file-i-o}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input #read-write-file-i-o}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import os
```

## Loading and Saving Tensors

For individual tensors, we can directly
invoke the `load` and `save` functions
to read and write them respectively.
Both functions require that we supply a name,
and `save` requires as input the variable to be saved.

```{.python .input #read-write-loading-and-saving-tensors-1}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input #read-write-loading-and-saving-tensors-1}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input #read-write-loading-and-saving-tensors-1}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input #read-write-loading-and-saving-tensors-1}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

We can now read the data from the stored file back into memory.

```{.python .input #read-write-loading-and-saving-tensors-2}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input #read-write-loading-and-saving-tensors-2}
%%tab pytorch
x2 = torch.load('x-file', weights_only=True)
x2
```

```{.python .input #read-write-loading-and-saving-tensors-2}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input #read-write-loading-and-saving-tensors-2}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

We can store a list of tensors and read them back into memory.

```{.python .input #read-write-loading-and-saving-tensors-3}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input #read-write-loading-and-saving-tensors-3}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files', weights_only=True)
(x2, y2)
```

```{.python .input #read-write-loading-and-saving-tensors-3}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input #read-write-loading-and-saving-tensors-3}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

We can even write and read a dictionary that maps
from strings to tensors.
This is convenient when we want
to read or write all the weights in a model.

```{.python .input #read-write-loading-and-saving-tensors-4}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input #read-write-loading-and-saving-tensors-4}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict', weights_only=True)
mydict2
```

```{.python .input #read-write-loading-and-saving-tensors-4}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input #read-write-loading-and-saving-tensors-4}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## Loading and Saving Model Parameters

Saving individual weight vectors (or other tensors) is useful,
but it gets very tedious if we want to save
(and later load) an entire model.
After all, we might have hundreds of
parameter groups sprinkled throughout.
For this reason the deep learning framework provides built-in functionalities
to load and save entire networks.
An important detail to note is that this
saves model *parameters* and not the entire model.
For example, if we have a 3-layer MLP,
we need to specify the architecture separately.
The reason for this is that the models themselves can contain arbitrary code,
hence they cannot be serialized as naturally.
Thus, in order to reinstate a model, we need
to generate the architecture in code
and then load the parameters from disk.
Let's start with our familiar MLP.

```{.python .input #read-write-loading-and-saving-model-parameters-1}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input #read-write-loading-and-saving-model-parameters-1}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input #read-write-loading-and-saving-model-parameters-1}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input #read-write-loading-and-saving-model-parameters-1}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(d2l.get_key(), (2, 20))
Y, params = net.init_with_output(d2l.get_key(), X)
```

Next, we store the parameters of the model as a file with the name "mlp.params".

```{.python .input #read-write-loading-and-saving-model-parameters-2}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input #read-write-loading-and-saving-model-parameters-2}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input #read-write-loading-and-saving-model-parameters-2}
%%tab tensorflow
net.save_weights('mlp.weights.h5')
```

```{.python .input #read-write-loading-and-saving-model-parameters-2}
%%tab jax
checkpoints.save_checkpoint(os.path.abspath('ckpt_dir'), params, step=1,
                            overwrite=True)
```

To recover the model, we instantiate a clone
of the original MLP model.
Instead of randomly initializing the model parameters,
we read the parameters stored in the file directly.

```{.python .input #read-write-loading-and-saving-model-parameters-3}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input #read-write-loading-and-saving-model-parameters-3}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params', weights_only=True))
clone.eval()
```

```{.python .input #read-write-loading-and-saving-model-parameters-3}
%%tab tensorflow
clone = MLP()
clone(X)
clone.load_weights('mlp.weights.h5')
```

```{.python .input #read-write-loading-and-saving-model-parameters-3}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint(
    os.path.abspath('ckpt_dir'), target=params))
```

Since both instances have the same model parameters,
the computational result of the same input `X` should be the same.
Let's verify this.

```{.python .input #read-write-loading-and-saving-model-parameters-4}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input #read-write-loading-and-saving-model-parameters-4}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Summary

The `save` and `load` functions can be used to perform file I/O for tensor objects.
We can save and load the entire sets of parameters for a network via a parameter dictionary.
Saving the architecture has to be done in code rather than in parameters.

## Exercises

1. Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?
1. Assume that we want to reuse only parts of a network to be incorporated into a network having a different architecture. How would you go about using, say the first two layers from a previous network in a new network?
1. How would you go about saving the network architecture and parameters? What restrictions would you impose on the architecture?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17994)
:end_tab:

<!-- slides -->

::: {.slide}
Two real problems training pipelines hit constantly:

- **Crash recovery** — a 12-hour training run dies in hour
  9. Did we just lose 9 hours of compute?
- **Deployment** — model trains on a research box; needs to
  serve from a production cluster, possibly in a different
  language or runtime.

Both reduce to "save the parameters, recreate the model
elsewhere". The crucial split:

```
   architecture  ──>  Python code  (committed to git)
   parameters    ──>  on-disk file (the .pt / .ckpt / .safetensors)
```

You save the **state**, not the *class*. To resurrect: import
the same class, instantiate, then `load_state_dict`. This
section covers both halves of the workflow.
:::

::: {.slide title="Saving and loading raw tensors"}
First the building block: `torch.save` / `torch.load` work
on any tensor, list of tensors, or dict thereof:

@read-write-file-i-o

. . .

@read-write-loading-and-saving-tensors-1

. . .

@read-write-loading-and-saving-tensors-2

`weights_only=True` is the default since 2024 — pickle in
PyTorch checkpoints can execute arbitrary code, so this
sandboxes loading.
:::

::: {.slide title="Containers of tensors"}
Lists and dicts work the same — perfect for grouping
related tensors together (e.g. weights of one block, plus
its running statistics):

@read-write-loading-and-saving-tensors-3

. . .

@read-write-loading-and-saving-tensors-4
:::

::: {.slide title="A model to save"}
@read-write-loading-and-saving-model-parameters-1
:::

::: {.slide title="state_dict() — the canonical interface"}
Every `Module` exposes a `state_dict()` — an ordered dict
mapping parameter *paths* to tensor values:

```
{
  'hidden.weight':  Tensor (256, 20),
  'hidden.bias':    Tensor (256,),
  'output.weight':  Tensor (10, 256),
  'output.bias':    Tensor (10,),
}
```

The keys come from the module tree (`self.hidden` → `hidden`).
Save this dict, not the module:

@read-write-loading-and-saving-model-parameters-2
:::

::: {.slide title="Loading: instantiate, then load"}
Build a fresh model with the same Python class, then call
`load_state_dict`. The dict's keys must match the new
model's parameter paths:

@read-write-loading-and-saving-model-parameters-3

. . .

Sanity check — same architecture + same weights produces
bit-identical outputs:

@read-write-loading-and-saving-model-parameters-4
:::

::: {.slide title="Best practices"}
- **Save `state_dict`, not the module object.** Pickling
  the module ties the file to today's Python class; refactor
  and old checkpoints break.
- **Keep the model class in your code repo.** The file is
  useless without the matching architecture definition.
- **Strict vs non-strict**: `load_state_dict(d, strict=False)`
  ignores missing/extra keys — useful for partial loading
  (e.g. swapping a pretrained head).
:::

::: {.slide title="Beyond `state_dict`"}
- **safetensors** — same shape as `state_dict` but without
  pickle, so no arbitrary-code-exec risk. HuggingFace
  standard, used by every modern library.
- **Full checkpoint** — save more than weights:

  ```python
  {'model':     net.state_dict(),
   'optimizer': opt.state_dict(),
   'epoch':     epoch,
   'rng_state': torch.get_rng_state()}
  ```

  Lets you resume training bit-exactly after a crash.
:::

::: {.slide title="Recap"}
- Save the **state**, recreate the **architecture** in code.
- `torch.save` / `torch.load` for tensors and dicts;
  `state_dict()` / `load_state_dict()` for modules.
- Always sanity-check by running a known input through
  before-and-after-load and comparing.
- Production: HuggingFace `safetensors` for the no-pickle
  story.
- Full checkpoint: save model + optimizer + epoch + RNG
  state, in one dict.
:::
