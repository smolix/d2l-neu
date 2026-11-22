```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Numerical Formats
:label:`sec_numerics`

Every tensor in our models so far has used 32-bit floating point (fp32), without
questioning whether neural networks actually need that many bits. Cutting the
bit width in half pays off immediately in both memory and throughput: the
8-billion-parameter model running through this chapter takes 32 GB for its
weights in 32-bit precision but only 16 GB in 16-bit, and modern GPUs execute
16-bit matrix multiplications at least twice as fast as 32-bit ones. Can we
train in 16 bits and still recover the same model?

Not naively. Sixteen-bit formats are noticeably coarser, opening the door to two
silent failure modes that raise no runtime error. In fp16, $300^2$ already
overflows to `inf`, flooding training with `inf` and NaN values. In bf16, adding
$0.001$ to a weight of $1$ rounds right back to $1$, quietly stalling learning
even while gradients look healthy.

This section shows how to train reliably in 16 bits while avoiding both traps.
Every tensor carries a *dtype*—its numerical format—and every format trades off
*dynamic range* (how large or small representable numbers can be) against
*precision* (how finely adjacent numbers are spaced). *Mixed-precision training*
keeps an fp32 master copy of the weights so tiny gradient updates accumulate
faithfully, while executing expensive matrix multiplications and storing
activations in 16 bits—paired with *loss scaling* when using fp16 so small
gradients do not underflow to zero. Whereas :numref:`sec_parameters` counted
bytes per parameter, here we control bytes per number. The second subsection
(dtype promotion and model casting) is practical reference material you can skim
quickly. All examples run on CPU or GPU; actual 16-bit speedups require GPU
matrix units, which we measure in :numref:`sec_use_gpu`.

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #numerics-numerical-formats-1}
%%tab mxnet, pytorch
import random
```

```{.python .input #numerics-numerical-formats-1}
%%tab tensorflow
import ml_dtypes
import random
```

## Floating-Point Formats

### Range and Precision

Simply squaring 300 in both 16-bit formats reveals the two distinct ways low
precision can bite—and 300 is hardly exotic inside a neural network, where an
unnormalized logit or intermediate variance sum can easily reach that size.

```{.python .input #numerics-the-dtype-zoo-1}
%%tab mxnet
x = mnp.array(300.0)
x.astype('float16')**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab pytorch
x = torch.tensor(300.0)
x.to(torch.float16)**2, x.to(torch.bfloat16)**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab tensorflow
x = tf.constant(300.0)
tf.cast(x, tf.float16)**2, tf.cast(x, tf.bfloat16)**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab jax
x = jnp.array(300.0)
x.astype(jnp.float16)**2, x.astype(jnp.bfloat16)**2
```

:begin_tab:`mxnet`
MXNet supports bf16 only in part, and it cannot train bf16 weights, so the
MXNet tab works in fp16 and the cell squares only in fp16. In bf16 the
result would be 90112.
:end_tab:

In fp16 (*half precision*), $300^2$ overflows straight to `inf` because its
largest finite value is just 65504. In bf16 (*brain floating point*), the result
stays finite—90112 instead of 90000—yet already incurs rounding error in the
third significant digit. Both formats fit in 16 bits; they simply partition
those bits differently. Any normal floating-point number stores a sign bit $s$,
an exponent $e$, and a $t$-bit mantissa encoding a fraction $f \in [0, 1)$,
representing

$$x = (-1)^s \cdot 2^{e} \cdot (1 + f).$$

Exponent bits govern dynamic range—the largest and smallest magnitudes $|x|$ the
format can represent—while mantissa bits govern relative precision: adjacent
numbers between $1$ and $2$ sit $2^{-t}$ apart, and between $2^e$ and $2^{e+1}$
they sit $2^{e-t}$ apart, maintaining uniform relative spacing across scales.
The three formats we work with allocate their bits as follows:

| format | sign | exponent | mantissa |
|:-------|-----:|---------:|---------:|
| fp32   | 1    | 8        | 23       |
| bf16   | 1    | 8        | 7        |
| fp16   | 1    | 5        | 10       |

Notice the trade-off immediately. bf16 preserves all 8 exponent bits of fp32
while trimming the mantissa down to 7 bits, delivering fp32's full dynamic range
at coarser resolution. fp16 does the opposite, retaining 10 mantissa bits but
only 5 exponent bits—finer local steps, far narrower range. We can check both
outputs above by hand: with 5 exponent bits, $e$ spans $[-14, 15]$, capping fp16
at $2^{15}(2 - 2^{-10}) = 65504$. With 7 mantissa bits, bf16 values between
$2^{16}$ and $2^{17}$ are spaced by $2^9 = 512$, so 90000 falls between
$175 \cdot 512 = 89600$ and $176 \cdot 512 = 90112$ and rounds to the closer
value, 90112.

:numref:`fig_bg_float-formats` visualizes these bit budgets side by side (see
:numref:`sec_mdl-numerical-stability-conditioning` for a deeper treatment of
IEEE floating-point arithmetic).

![Bit layouts of the formats in the table, aligned at the sign bit, so that a longer bar means more bits: bf16 keeps the 8-bit exponent of fp32, and fp16 and bf16 split the same 16 bits in opposite ways.](../img/bg-float-formats.svg)
:label:`fig_bg_float-formats`

We can query each format's exact numerical limits via `finfo`, focusing on three
fields: `max` (largest finite value), `tiny` (smallest positive *normal* value),
and `eps` (the gap $2^{-t}$ just above 1). Below `tiny`, where the exponent
hits its floor, the format degrades gracefully through evenly spaced *subnormal*
numbers down to zero: in fp16 these are multiples of $2^{-24}$, so a value near
$3 \times 10^{-5}$ still preserves 9 significant bits, whereas one near
$10^{-7}$ retains only a single bit.

```{.python .input #numerics-the-dtype-zoo-2}
%%tab mxnet
for dtype in ('float32', 'float16'):
    fi = mnp.finfo(dtype)
    print(f'{dtype:10} max {fi.max:10.3e}  tiny {fi.smallest_normal:9.3e}'
          f'  eps {fi.eps:8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab pytorch
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    fi = torch.finfo(dtype)
    print(f'{str(dtype):15} max {fi.max:10.3e}  tiny {fi.tiny:9.3e}'
          f'  eps {fi.eps:8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab tensorflow
for dtype in (tf.float32, tf.bfloat16, tf.float16):
    fi = ml_dtypes.finfo(dtype.as_numpy_dtype)
    print(f'{dtype.name:10} max {float(fi.max):10.3e}'
          f'  tiny {float(fi.tiny):9.3e}  eps {float(fi.eps):8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab jax
for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
    fi = jnp.finfo(dtype)
    print(f'{fi.dtype.name:10} max {float(fi.max):10.3e}'
          f'  tiny {float(fi.tiny):9.3e}  eps {float(fi.eps):8.3e}')
```

:begin_tab:`mxnet`
`mnp.finfo` calls the smallest normal value `smallest_normal` rather than
`tiny`. For bf16, which the cell leaves out, the three numbers are `max`
$3.39 \times 10^{38}$, `tiny` $1.18 \times 10^{-38}$, and `eps`
$2^{-7} \approx 0.0078$.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow has no `finfo` of its own for bf16. The numbers come from
`ml_dtypes`, the small package that defines the bf16 type TensorFlow uses.
:end_tab:

Because bf16 shares fp32's 8-bit exponent, it matches fp32's `tiny` and nearly
matches its `max`, while its `eps` of $0.0078$ gives two to three decimal digits
of precision. fp16 resolves three to four decimal digits, but overflows above
65504, slips into subnormals below $6.1 \times 10^{-5}$, and cannot represent
any positive value below $2^{-24} \approx 6.0 \times 10^{-8}$. Which shortage
hurts deep learning more? Activations and gradients routinely span many orders
of magnitude; running out of range catastrophically turns numbers into `inf` or
zero, whereas slightly coarser precision merely introduces mild rounding noise.
That is why bf16 has become the default 16-bit training format on modern
hardware, leaving fp16 primarily for older GPUs and inference.

### Rounding of Small Updates

Where bf16's coarse spacing *does* hurt is during parameter updates. A single
gradient descent step shifts a weight by $\eta g$, often $10^{-3}$ or less of
the weight's magnitude. Under round-to-nearest, any step smaller than half the
local spacing vanishes completely. Let's add $10^{-3}$ and $10^{-4}$ to a weight
of $1$ in each format:

```{.python .input #numerics-rounding-of-small-updates}
%%tab mxnet
w = mnp.array(1.0)
for dtype in ('float32', 'float16'):
    print(f'{dtype:8}: {float(w.astype(dtype) + 1e-3):.7f}, '
          f'{float(w.astype(dtype) + 1e-4):.7f}')
```

```{.python .input #numerics-rounding-of-small-updates}
%%tab pytorch
w = torch.tensor(1.0)
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    print(f'{str(dtype):14}: {float(w.to(dtype) + 1e-3):.7f}, '
          f'{float(w.to(dtype) + 1e-4):.7f}')
```

```{.python .input #numerics-rounding-of-small-updates}
%%tab tensorflow
w = tf.constant(1.0)
for dtype in (tf.float32, tf.bfloat16, tf.float16):
    print(f'{dtype.name:8}: {float(tf.cast(w, dtype) + 1e-3):.7f}, '
          f'{float(tf.cast(w, dtype) + 1e-4):.7f}')
```

```{.python .input #numerics-rounding-of-small-updates}
%%tab jax
w = jnp.array(1.0)
for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
    print(f'{jnp.dtype(dtype).name:8}: {float(w.astype(dtype) + 1e-3):.7f}, '
          f'{float(w.astype(dtype) + 1e-4):.7f}')
```

In fp32, both increments register cleanly. In bf16—whose spacing near 1 is
$2^{-7} \approx 0.0078$—both round right back to 1, leaving the weight
completely frozen. In fp16 (spacing $2^{-10} \approx 0.00098$), the $10^{-3}$
step survives (rounded to one grid step) while $10^{-4}$ disappears. Storing
weights directly in 16 bits can therefore stall optimization even when every
gradient is accurate—which is why training keeps master weights in fp32.

## Dtype Promotion and Model Casting

### Promotion

What dtype results when an expression combines mismatched operands? Let's test
adding an fp16 tensor to an fp32 tensor, a Python float, and an integer tensor:

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab mxnet
x16 = mnp.ones(3, dtype='float16')
x32 = mnp.ones(3, dtype='float32')
((x16 + x32).dtype, (x16 + 1.0).dtype,
 mnp.result_type(x16, mnp.ones(3, dtype='int32')))
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab pytorch
x16 = torch.ones(3, dtype=torch.float16)
x32 = torch.ones(3, dtype=torch.float32)
(x16 + x32).dtype, (x16 + 1.0).dtype, (x16 + torch.arange(3)).dtype
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab tensorflow
x16 = tf.ones(3, dtype=tf.float16)
x32 = tf.ones(3, dtype=tf.float32)
try:
    x16 + x32
except tf.errors.InvalidArgumentError as e:
    print(e.message.split(' but ')[0])
(x16 + 1.0).dtype
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab jax
x16 = jnp.ones(3, dtype=jnp.float16)
x32 = jnp.ones(3, dtype=jnp.float32)
(x16 + x32).dtype, (x16 + 1.0).dtype, (x16 + jnp.arange(3)).dtype
```

:begin_tab:`mxnet, pytorch, jax`
Combining two float tensors promotes to the wider format (and fp16 + bf16 promotes to fp32)—meaning a single stray fp32 tensor silently promotes an entire downstream 16-bit computation to fp32! Python float literals and integer tensors do not promote float tensors.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow never promotes across tensor dtypes: combining fp16 and fp32 raises an `InvalidArgumentError` unless explicitly converted with `tf.cast`.
:end_tab:
### Casting a Model

Since parameters are just tensors, we can cast an entire model to 16 bits.
Throughout this section we use our Fashion-MNIST MLP from
:numref:`sec_mlp-implementation`: 784 inputs, a 256-unit ReLU hidden layer, and
10 outputs, with $\mathcal{N}(0, 0.01^2)$ weights and zero biases—203,530
parameters occupying 814,120 bytes in fp32.

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab mxnet
def mlp():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu', in_units=784),
            nn.Dense(10, in_units=256))
    net.initialize(init.Normal(sigma=0.01))
    return net

def param_bytes(net):
    return sum(p.data().size * p.data().dtype.itemsize
               for p in net.collect_params().values())

net = mlp()
param_bytes(net)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab pytorch
def mlp():
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 10))
    for layer in net[1], net[3]:
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.zeros_(layer.bias)
    return net

def param_bytes(net):
    return sum(p.numel() * p.element_size() for p in net.parameters())

net = mlp()
param_bytes(net)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab tensorflow
def mlp(dtype=None):
    normal = tf.keras.initializers.RandomNormal  # one instance per layer
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(dtype=dtype),
        tf.keras.layers.Dense(256, activation='relu', dtype=dtype,
                              kernel_initializer=normal(stddev=0.01)),
        tf.keras.layers.Dense(10, dtype=dtype,
                              kernel_initializer=normal(stddev=0.01))])

def param_bytes(net):
    return sum(w.shape.num_elements() * tf.as_dtype(w.dtype).size
               for w in net.weights)

net = mlp()
net(tf.zeros((1, 28, 28, 1)))  # The first call creates the weights
param_bytes(net)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab jax
def mlp(rngs, dtype=None):
    normal = nnx.initializers.normal(0.01)
    return nnx.Sequential(
        lambda x: x.reshape((x.shape[0], -1)),
        nnx.Linear(784, 256, kernel_init=normal, dtype=dtype, rngs=rngs),
        nnx.relu,
        nnx.Linear(256, 10, kernel_init=normal, dtype=dtype, rngs=rngs))

def param_bytes(model):
    return sum(p.size * p.dtype.itemsize
               for _, p in nnx.state(model, nnx.Param).flat_state())

net = mlp(nnx.Rngs(d2l.get_key()))
param_bytes(net)
```

:begin_tab:`mxnet`
`net.cast('float16')` casts all parameters in the block tree; inputs must be cast to match.
:end_tab:

:begin_tab:`pytorch`
`net.to(dtype)` (or `net.bfloat16()`) casts all parameters and buffers; layers reject inputs whose dtype does not match their weights.
:end_tab:

:begin_tab:`tensorflow`
Passing `dtype='bfloat16'` at construction sets both `variable_dtype` and `compute_dtype` on each Keras layer, automatically casting inputs on call.
:end_tab:

:begin_tab:`jax`
Flax NNX layers separate storage (`param_dtype`, default fp32) from compute (`dtype`). Mapping `.astype(jnp.bfloat16)` over `nnx.Param` casts stored weights.
:end_tab:
```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab mxnet
net.cast('float16')
X = mnp.random.normal(size=(8, 1, 28, 28))
print(net(X.astype('float16')).dtype, param_bytes(net))
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab pytorch
net = net.bfloat16()
X = torch.randn(8, 1, 28, 28)
try:
    net(X)
except RuntimeError as e:
    print(e)
print(net(X.bfloat16()).dtype, param_bytes(net))
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab tensorflow
net16 = mlp(dtype='bfloat16')
X = tf.random.normal((8, 28, 28, 1))
print(net16(X).dtype, param_bytes(net16))
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab jax
def cast(model, dtype):
    """Return the model with its parameters stored in dtype."""
    graph, params = nnx.split(model, nnx.Param)
    return nnx.merge(graph, jax.tree.map(lambda p: p.astype(dtype), params))

net16 = cast(net, jnp.bfloat16)
X = jax.random.normal(d2l.get_key(), (8, 28, 28, 1))
print(net16(X).dtype, param_bytes(net16))
print(net16(X.astype(jnp.bfloat16)).dtype)
```

Across all four frameworks, casting halves parameter memory, though each
framework handles an fp32 input differently.

For *inference*, casting a trained model to 16 bits immediately halves its
weight footprint (subject to verifying that task accuracy holds up). For
*training*, however, pure 16-bit weights run straight into the small-update
rounding trap we just saw—as we confirm empirically below.

## Mixed-Precision Training

### Master Weights and 16-Bit Compute

We appear to face a dilemma: reliable weight updates demand 32-bit storage, yet
hardware speedups live in 16-bit arithmetic. *Mixed-precision training*
:cite:`Micikevicius.Narang.Alben.ea.2018` resolves this tension by decoupling
parameter storage from computation. We keep fp32 *master weights* so every small
update accumulates cleanly, cast them to 16 bits for the compute-heavy matrix
multiplications and activations of the forward and backward passes, and keep
sensitive reductions such as the loss in fp32. :numref:`fig_bg_amp-loop`
summarizes a single training step.

![One step of mixed-precision training: the fp32 master weights are cast to bf16 for the forward pass, which produces bf16 activations; the loss is computed in fp32; the backward pass produces gradients, and the optimizer applies them to the fp32 master weights. With fp16, the loss is also scaled up before the backward pass and the gradients are scaled back down before the update.](../img/bg-amp-loop.svg)
:label:`fig_bg_amp-loop`

:begin_tab:`mxnet`
We cast the network to fp16 (`net.cast('float16')`) and pass `multi_precision=True` to the optimizer so it maintains fp32 master weights internally.
:end_tab:

:begin_tab:`pytorch`
`torch.autocast` leaves parameters in fp32 and automatically casts eligible ops (matrix multiplies, convolutions) to 16 bits while keeping reductions and losses in fp32.
:end_tab:

:begin_tab:`tensorflow`
Constructing a model under the `'mixed_bfloat16'` global policy gives every layer fp32 `variable_dtype` and bf16 `compute_dtype`.
:end_tab:

:begin_tab:`jax`
Constructing NNX layers with `dtype=jnp.bfloat16` (leaving `param_dtype` at its fp32 default) stores fp32 master weights and computes forward passes in bf16.
:end_tab:
```{.python .input #numerics-mixed-precision-training-1}
%%tab mxnet
net = mlp()
net.cast('float16')
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.1, 'multi_precision': True})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
with autograd.record():
    l = loss(net(X.astype('float16')), mnp.zeros(8)).mean()
l.backward()
trainer.step(1)
master = trainer._updaters[0].states[0][0]  # The fp32 master copy
net[0].weight.data().dtype, master.dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab pytorch
net = mlp()
with torch.autocast('cpu', dtype=torch.bfloat16):
    Y = net(X)
Y.dtype, net[1].weight.dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab tensorflow
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
net = mlp()
tf.keras.mixed_precision.set_global_policy('float32')
Y = net(X)
Y.dtype, net.layers[1].kernel.dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab jax
net = mlp(nnx.Rngs(d2l.get_key()), dtype=jnp.bfloat16)
net(X).dtype, jax.tree.leaves(nnx.state(net, nnx.Param))[0].dtype
```

### Training in fp32 and in 16 Bits

How much does 16-bit arithmetic affect optimization, and what goes wrong when we
drop the fp32 master copy? Let's train our MLP three times with plain SGD on
Fashion-MNIST minibatches of 256 images: full fp32, pure bf16 weights (with no
fp32 master copy), and mixed precision. All three runs start from identical
random weights and consume the exact same minibatches in the same order.

Notice the learning rate: we set $\eta = 0.01$—one-tenth the rate used in
:numref:`sec_mlp-implementation`—and run for 1000 steps (about four epochs). Our
rounding analysis explains why $\eta$ matters so dramatically: an update
$\eta g$ disappears whenever it falls below half the local bf16 grid spacing. At
$\eta = 0.01$, roughly 90% of first-layer updates vanish—mirroring the
late-stage learning-rate decay phase of any real training run. At $\eta = 0.1$,
by contrast, only the smallest 40% of updates (carrying about 2% of total update
mass) get rounded away, masking the failure until late in training.

```{.python .input #numerics-mixed-precision-training-2}
%%tab mxnet
data = d2l.FashionMNIST(batch_size=256)
batches = list(gluon.data.DataLoader(data.train, batch_size=256,
                                     last_batch='discard'))  # One epoch
val_batch = next(iter(gluon.data.DataLoader(data.val, batch_size=1024)))
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab pytorch
data = d2l.FashionMNIST(batch_size=256)
batches = list(torch.utils.data.DataLoader(data.train, batch_size=256,
                                           drop_last=True))  # One epoch
val_batch = next(iter(torch.utils.data.DataLoader(data.val, batch_size=1024)))
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab tensorflow
data = d2l.FashionMNIST(batch_size=256)
images, labels = data.train
batches = [(tf.constant(images[k:k+256, :, :, None], tf.float32) / 255,
            tf.constant(labels[k:k+256], tf.int32))
           for k in range(0, len(images) - 255, 256)]  # One epoch
val_images, val_labels = data.val
val_batch = (tf.constant(val_images[:1024, :, :, None], tf.float32) / 255,
             tf.constant(val_labels[:1024], tf.int32))
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab jax
data = d2l.FashionMNIST(batch_size=256)
images, labels = data.train
batches = [(jnp.asarray(images[k:k+256], jnp.float32) / 255,
            jnp.asarray(labels[k:k+256], jnp.int32))
           for k in range(0, len(images) - 255, 256)]  # One epoch
val_images, val_labels = data.val
val_batch = (jnp.asarray(val_images[:1024, :, :, None], jnp.float32) / 255,
             jnp.asarray(val_labels[:1024], jnp.int32))
```

With the deterministic training minibatches `batches` and 1,024-image validation slice `val_batch` preloaded in memory, we now define `train(net, ...)` to run $1{,}000$ SGD steps at $\eta = 0.01$—either updating low-precision weights directly or keeping an `fp32` master copy for the parameter update—and return the per-step losses and final accuracy on `val_batch`:

```{.python .input #numerics-mixed-precision-training-3}
%%tab mxnet
def bf16_round(w):
    """Round to the 8 significant bits of bf16, computed in NumPy."""
    m, e = np.frexp(w.asnumpy())  # w = m * 2**e with 1/2 <= |m| < 1
    return mnp.array(np.ldexp(np.round(m * 2**8) / 2**8, e))

def train(net, store=None, num_steps=1000):
    dtype = net[0].weight.data().dtype
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': 0.01, 'multi_precision': dtype == np.float16})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    losses = []
    for step in range(num_steps):
        X, y = batches[step % len(batches)]
        with autograd.record():
            logits = net(X.astype(dtype)).astype('float32')
            l = loss(logits, y).mean()
        l.backward()
        trainer.step(1)
        if store:  # Round every updated weight to the storage format
            for p in net.collect_params().values():
                p.set_data(store(p.data()))
        losses.append(l)
    Xv, yv = val_batch
    accuracy = float((net(Xv.astype(dtype)).argmax(axis=1) == yv).mean())
    return [float(l) for l in losses], accuracy
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab pytorch
def train(net, mixed=False, num_steps=1000):
    dtype = next(net.parameters()).dtype  # The inputs follow the weights
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    losses = []
    for step in range(num_steps):
        X, y = batches[step % len(batches)]
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=mixed):
            loss = F.cross_entropy(net(X.to(dtype)).float(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    Xv, yv = val_batch
    with torch.no_grad():
        accuracy = (net(Xv.to(dtype)).argmax(1) == yv).float().mean().item()
    return torch.stack(losses).tolist(), accuracy
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab tensorflow
def train(net, num_steps=1000):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    losses = []
    for step in range(num_steps):
        X, y = batches[step % len(batches)]
        with tf.GradientTape() as tape:
            logits = tf.cast(net(X), tf.float32)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True))
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        losses.append(loss)
    Xv, yv = val_batch
    y_hat = tf.argmax(net(Xv), axis=1, output_type=yv.dtype)
    accuracy = float(tf.reduce_mean(tf.cast(y_hat == yv, tf.float32)))
    return [float(l) for l in losses], accuracy
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab jax
@nnx.jit
def train_step(model, optimizer, X, y):
    def loss_fn(model):
        logits = model(X).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train(net, num_steps=1000):
    optimizer = nnx.Optimizer(net, optax.sgd(0.01), wrt=nnx.Param)
    losses = [train_step(net, optimizer, *batches[step % len(batches)])
              for step in range(num_steps)]
    Xv, yv = val_batch
    accuracy = float((net(Xv).argmax(1) == yv).mean())
    return [float(l) for l in losses], accuracy
```

One implementation detail in `train` is a good systems habit independent of
precision: we keep per-step loss tensors on-device during the loop and convert
them to Python floats once at the end, avoiding a CPU–GPU synchronization stall
on every single step (:numref:`sec_use_gpu`).

:begin_tab:`mxnet`
Because MXNet cannot train bf16 weights directly, `bf16_round` rounds fp32 weights to bf16's 7-bit mantissa after each step, while the mixed-precision run uses fp16 with `multi_precision=True`.
:end_tab:
```{.python .input #numerics-mixed-precision-training-4}
%%tab mxnet
def seeded_mlp(seed):
    npx.random.seed(seed)  # The same seed gives the same initial weights
    return mlp()

seed = random.randrange(2**31)  # One random draw, shared by all runs
loss32, acc32 = train(seeded_mlp(seed))
loss_bf16, acc_bf16 = train(seeded_mlp(seed), store=bf16_round)
net = seeded_mlp(seed)
net.cast('float16')
loss_mixed, acc_mixed = train(net)
print(f'validation accuracy: fp32 {acc32:.3f}, bf16 weights {acc_bf16:.3f}, '
      f'mixed {acc_mixed:.3f}')
d2l.plot(list(range(1, 1001)), [loss32, loss_bf16, loss_mixed], 'step',
         'loss', legend=['fp32', 'bf16 weights', 'mixed precision'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab pytorch
def seeded_mlp(seed):
    torch.manual_seed(seed)  # The same seed gives the same initial weights
    return mlp()

seed = random.randrange(2**31)  # One random draw, shared by all runs
loss32, acc32 = train(seeded_mlp(seed))
loss_bf16, acc_bf16 = train(seeded_mlp(seed).bfloat16())
loss_mixed, acc_mixed = train(seeded_mlp(seed), mixed=True)
print(f'validation accuracy: fp32 {acc32:.3f}, bf16 weights {acc_bf16:.3f}, '
      f'mixed {acc_mixed:.3f}')
d2l.plot(list(range(1, 1001)), [loss32, loss_bf16, loss_mixed], 'step',
         'loss', legend=['fp32', 'bf16 weights', 'mixed precision'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab tensorflow
def seeded_mlp(seed, policy):
    tf.keras.utils.set_random_seed(seed)  # The same initial weights
    tf.keras.mixed_precision.set_global_policy(policy)
    net = mlp()
    tf.keras.mixed_precision.set_global_policy('float32')
    return net

seed = random.randrange(2**31)  # One random draw, shared by all runs
loss32, acc32 = train(seeded_mlp(seed, 'float32'))
loss_bf16, acc_bf16 = train(seeded_mlp(seed, 'bfloat16'))
loss_mixed, acc_mixed = train(seeded_mlp(seed, 'mixed_bfloat16'))
print(f'validation accuracy: fp32 {acc32:.3f}, bf16 weights {acc_bf16:.3f}, '
      f'mixed {acc_mixed:.3f}')
d2l.plot(list(range(1, 1001)), [loss32, loss_bf16, loss_mixed], 'step',
         'loss', legend=['fp32', 'bf16 weights', 'mixed precision'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab jax
key = d2l.get_key()  # One random key, shared by all runs
loss32, acc32 = train(mlp(nnx.Rngs(key)))
loss_bf16, acc_bf16 = train(cast(mlp(nnx.Rngs(key), jnp.bfloat16),
                                 jnp.bfloat16))
loss_mixed, acc_mixed = train(mlp(nnx.Rngs(key), jnp.bfloat16))
print(f'validation accuracy: fp32 {acc32:.3f}, bf16 weights {acc_bf16:.3f}, '
      f'mixed {acc_mixed:.3f}')
d2l.plot(list(range(1, 1001)), [loss32, loss_bf16, loss_mixed], 'step',
         'loss', legend=['fp32', 'bf16 weights', 'mixed precision'])
```

The fp32 and mixed-precision loss curves sit virtually on top of each other,
with validation accuracies matching within about one percentage point. Pure bf16
weights, by contrast, lag behind within the first hundred steps: after 1000
steps, its loss stalls near 1.15 versus 0.85, and validation accuracy trails at
about 64% versus 68%. Just as our scalar experiment predicted, storing weights
directly in bf16 rounds away most first-layer updates, whereas mixed precision
applies them cleanly to fp32 master weights. None of the three runs is trained
to convergence—what matters is the apples-to-apples comparison on identical data
and step counts.

Notice two important caveats about what mixed precision does and does not buy
us. First, we do not see a wall-clock speedup on a tiny CPU MLP; 16-bit speedups
come from dedicated GPU tensor cores working on large matrix multiplications,
which we benchmark in :numref:`subsec_async_dispatch`. Second, mixed precision
does *not* shrink the static per-parameter memory floor from
:numref:`sec_parameters`, because master weights and optimizer moments remain in
fp32. That 16 GB footprint from the opening applies to an 8B model cast for
*inference*; during *training*, mixed precision saves activation memory (halving
stored 16-bit activations) and matrix-multiplication time.

:begin_tab:`mxnet`
The mixed-precision run of the MXNet tab computes in fp16 rather than bf16,
and it trained without loss scaling. The next subsection measures what fp16
does to the gradients of this network and explains when scaling is needed.
:end_tab:

### Loss Scaling for fp16
:label:`subsec_loss_scaling`

With bf16, master weights complete the recipe. With fp16, however, its narrow
exponent range creates a second hazard at the bottom end: small gradients
*underflow*. Magnitudes below $2^{-14} \approx 6.1 \times 10^{-5}$ drop into
subnormals and shed significant bits, and anything below half the smallest
subnormal ($\approx 3 \times 10^{-8}$) rounds straight to zero. A gradient of
$10^{-8}$ vanishes in fp16, yet multiplying it by $2^{16}$ lifts it safely into
normal range:

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab mxnet
g = mnp.array(1e-8)
g.astype('float16'), (g * 2**16).astype('float16')
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab pytorch
g = torch.tensor(1e-8)
g.half(), (g * 2**16).half()
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab tensorflow
g = tf.constant(1e-8)
tf.cast(g, tf.float16), tf.cast(g * 2**16, tf.float16)
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab jax
g = jnp.array(1e-8)
g.astype(jnp.float16), (g * 2**16).astype(jnp.float16)
```

How common are tiny gradients in practice? Let's compute first-layer fp32
gradients for our MLP on the first minibatch and measure what fraction of
nonzero entries fall into fp16's subnormal or zero regimes—both unscaled and
scaled by $2^{16}$:

```{.python .input #numerics-loss-scaling-for-fp16-3}
%%tab mxnet
net = mlp()
X, y = batches[0]
with autograd.record():
    l = gluon.loss.SoftmaxCrossEntropyLoss()(net(X), y).mean()
l.backward()
g = net[0].weight.grad().reshape(-1)
g = g[g != 0]
for scale in (1, 2**16):
    print(f'scale {scale:5}: '
          f'{float((mnp.abs(g * scale) < 2**-14).mean()):.1%} subnormal, '
          f'{float(((g * scale).astype("float16") == 0).mean()):.2%} zero')
```

```{.python .input #numerics-loss-scaling-for-fp16-3}
%%tab pytorch
net = mlp()
X, y = batches[0]
F.cross_entropy(net(X), y).backward()
g = net[1].weight.grad.flatten()
g = g[g != 0]
for scale in (1, 2**16):
    print(f'scale {scale:5}: '
          f'{((g * scale).abs() < 2**-14).float().mean():.1%} subnormal, '
          f'{((g * scale).half() == 0).float().mean():.2%} zero')
```

```{.python .input #numerics-loss-scaling-for-fp16-3}
%%tab tensorflow
net = mlp()
X, y = batches[0]
with tf.GradientTape() as tape:
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        y, net(X), from_logits=True))
g = tf.reshape(tape.gradient(loss, net.layers[1].kernel), -1)
g = g[g != 0]
for scale in (1, 2**16):
    subnormal = tf.reduce_mean(tf.cast(tf.abs(g * scale) < 2**-14, tf.float32))
    zero = tf.reduce_mean(tf.cast(tf.cast(g * scale, tf.float16) == 0,
                                  tf.float32))
    print(f'scale {scale:5}: {float(subnormal):.1%} subnormal, '
          f'{float(zero):.2%} zero')
```

```{.python .input #numerics-loss-scaling-for-fp16-3}
%%tab jax
net = mlp(nnx.Rngs(d2l.get_key()))
X, y = batches[0]
def loss_fn(model):
    return optax.softmax_cross_entropy_with_integer_labels(
        model(X), y).mean()
g = nnx.grad(loss_fn)(net)['layers'][1]['kernel'][...].reshape(-1)
g = g[g != 0]
for scale in (1, 2**16):
    print(f'scale {scale:5}: '
          f'{float((jnp.abs(g * scale) < 2**-14).mean()):.1%} subnormal, '
          f'{float(((g * scale).astype(jnp.float16) == 0).mean()):.2%} zero')
```

Roughly a third of the nonzero first-layer gradients land in fp16's subnormal
range, and about two in a thousand underflow all the way to zero. On this small
MLP the impact is mild because only the tiniest entries vanish—in fact, it is
hard to construct a toy model where fp16 fails outright and loss scaling saves
it. Severe underflow hits large, deep models where losses average over many
tokens or pixels and gradients attenuate across dozens of layers:
:citet:`Micikevicius.Narang.Alben.ea.2018` found that while several ImageNet
classifiers trained fine in unscaled fp16, an object detector diverged and a
large recurrent language model matched fp32 accuracy only once loss scaling was
enabled.

Multiplying the gradients by $2^{16}$ eliminates every subnormal entry while
staying comfortably below fp16's overflow ceiling of 65504. That is the essence
of *loss scaling*: before calling `backward()`, multiply the scalar loss by a
large constant $S$. By linearity of differentiation, every intermediate gradient
shifts up by $S$ during backpropagation. Before stepping the optimizer, divide
the gradients back down by $S$ in fp32, where tiny values are represented
faithfully—leaving the mathematical parameter update completely unchanged while
keeping intermediate fp16 gradients out of the underflow zone.

Choosing $S$ statically is brittle—too small and gradients underflow, too large
and big gradients overflow to `inf`. *Dynamic loss scaling* automates the choice:
start with a large $S$, inspect the gradients on every step, and whenever an
`inf` or NaN appears, skip the optimizer update and halve $S$; after a streak of
clean steps, double $S$ to stay close to the top of fp16's range.

:begin_tab:`mxnet, jax`
Gluon's `Trainer` and Optax provide no automatic loss scaler (JAX primarily targets TPUs and modern GPUs with native bf16); if training in fp16, multiply the loss by $S$ before backward and divide gradients by $S$ before the optimizer update.
:end_tab:

:begin_tab:`pytorch`
`torch.amp.GradScaler` automates dynamic loss scaling with fp16 `autocast`:
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.mixed_precision.LossScaleOptimizer` wraps an optimizer to scale the loss and unscale/check gradients under `'mixed_float16'`:
:end_tab:
```{.python .input #numerics-loss-scaling-for-fp16-2}
%%tab pytorch
net = mlp()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scaler = torch.amp.GradScaler('cpu')
for X, y in batches[:20]:
    with torch.autocast('cpu', dtype=torch.float16):
        loss = F.cross_entropy(net(X), y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()  # Gradients of S * loss
    scaler.step(optimizer)         # Divide by S, skip the step on inf/NaN
    scaler.update()                # Adjust S
print(f'loss {loss.item():.3f}, loss scale {scaler.get_scale():.0f}')
```

```{.python .input #numerics-loss-scaling-for-fp16-2}
%%tab tensorflow
tf.keras.mixed_precision.set_global_policy('mixed_float16')
net = mlp()
tf.keras.mixed_precision.set_global_policy('float32')
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(learning_rate=0.1))
for X, y in batches[:20]:
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y, tf.cast(net(X), tf.float32), from_logits=True))
        scaled_loss = optimizer.scale_loss(loss)  # S * loss
    grads = tape.gradient(scaled_loss, net.trainable_variables)
    # Divide by S, skip the step on inf/NaN, adjust S
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
print(f'loss {float(loss):.3f}, '
      f'loss scale {float(optimizer.dynamic_scale):.0f}')
```

Consequently, on modern hardware with native bf16 support (recent GPUs, TPUs,
and server CPUs), bf16 mixed precision without loss scaling is the standard
recipe; fp16 with dynamic loss scaling remains essential on older GPUs.

One final hardware feature accelerates fp32 models without changing any tensor
dtypes at all. Starting with NVIDIA's Ampere architecture, fp32 matrix
multiplications can run on tensor cores in *tf32* mode: inputs are truncated to
fp16's 10-bit mantissa while keeping fp32's 8-bit exponent, and inner products
accumulate in full fp32—delivering substantial speedups while matching strict
fp32 to three or four significant digits.

:begin_tab:`mxnet, tensorflow, jax`
MXNet, TensorFlow, and JAX enable tf32 on supported GPUs by default (controlled via `NVIDIA_TF32_OVERRIDE=0`, `tf.config.experimental.enable_tensor_float_32_execution(False)`, or `jax.default_matmul_precision('float32')`).
:end_tab:

:begin_tab:`pytorch`
PyTorch defaults to strict fp32 matmuls; `torch.set_float32_matmul_precision('high')` (:numref:`sec_use_gpu`) enables tf32.
:end_tab:
## Numerical Failure Modes

Three concrete hazards explain the vast majority of numerical bugs in 16-bit
code: an upstream overflow that surfaces downstream as NaN, an unshifted
exponential that blows up before the final logarithm can shrink it, and a long
reduction that stops growing once individual terms fall below the accumulator's
rounding threshold.

### From Overflow to NaN

A NaN loss is rarely the root cause—it is usually the aftermath of an earlier
overflow to `inf`, because indeterminate arithmetic like `inf - inf` or
`inf / inf` produces NaN:

```{.python .input #numerics-when-numerics-bite-1}
%%tab mxnet
s = mnp.array(60000., dtype='float16') * 2  # Overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab pytorch
s = torch.tensor(60000., dtype=torch.float16) * 2  # Overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab tensorflow
s = tf.constant(60000., dtype=tf.float16) * 2  # Overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab jax
s = jnp.array(60000., dtype=jnp.float16) * 2  # Overflows
s, s - s
```

That initial overflow may occur layers upstream before NaN ever reaches the
loss, and once a single parameter update carries a NaN, every subsequent step is
poisoned. Whenever a loss turns NaN, check value ranges first—is any operation
running in fp16 near its 65504 ceiling?—before blaming the learning rate or
architecture.

:begin_tab:`tensorflow, jax`
To halt immediately at the first op producing `inf` or NaN during debugging, enable `tf.debugging.enable_check_numerics()` or `jax.config.update('jax_debug_nans', True)`.
:end_tab:
### Log-Sum-Exp

Softmax cross-entropy requires computing $\log \sum_i \exp(x_i)$. Evaluating
that expression literally overflows in fp16 even when the final answer is small:

```{.python .input #numerics-when-numerics-bite-2}
%%tab mxnet
x = mnp.array([12.0, 11.0, 10.0], dtype='float16')
m = x.max()
mnp.log(mnp.exp(x).sum()), m + mnp.log(mnp.exp(x - m).sum())
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab pytorch
x = torch.tensor([12.0, 11.0, 10.0], dtype=torch.float16)
x.exp().sum().log(), torch.logsumexp(x, dim=0)
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab tensorflow
x = tf.constant([12.0, 11.0, 10.0], dtype=tf.float16)
tf.math.log(tf.reduce_sum(tf.exp(x))), tf.math.reduce_logsumexp(x)
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab jax
x = jnp.array([12.0, 11.0, 10.0], dtype=jnp.float16)
jnp.log(jnp.exp(x).sum()), jax.scipy.special.logsumexp(x)
```

Although the true answer is a modest 12.4, the intermediate term
$e^{12} \approx 163{,}000$ easily blows past fp16's 65504 limit. The fix is the
exact shift we introduced in :numref:`subsec_softmax-implementation-revisited`:
subtract the maximum entry $m = \max_i x_i$ before exponentiating and add $m$
back afterward,

$$\log \sum_i \exp(x_i) = m + \log \sum_i \exp(x_i - m).$$

Every exponent on the right is nonpositive—making overflow impossible—and the
largest term is exactly $e^0 = 1$, preventing underflow to $\log 0$. In
practice you rarely need to code this shift yourself: built-in log-softmax and
cross-entropy routines apply it automatically.

### Accumulation in fp32

Long sums accumulated in 16 bits stall silently: once the running total grows
large enough, each small summand falls below half the local grid spacing and
rounds to zero. Let's add $2^{-10}$ (represented *exactly* in fp16) 4096 times
into an fp16 accumulator versus an fp32 accumulator:

```{.python .input #numerics-when-numerics-bite-3}
total16, total32 = np.float16(0), np.float32(0)
term = np.float16(2**-10)
for _ in range(4096):
    total16 += term  # fp16 accumulator
    total32 += term  # fp32 accumulator, same fp16 terms
total16, total32
```

The exact total is 4, which the fp32 accumulator hits on the nose. The fp16
accumulator freezes at 2 after 2048 steps: between 2 and 4, adjacent fp16
values lie $2^{-9}$ apart, so $2 + 2^{-10}$ lands right on the midpoint and
round-to-nearest-even snaps it back to 2 forever. Batch averages, epoch loss
totals, and variance calculations in normalization layers all share this hazard,
which is why mixed-precision policies keep reductions and normalizations in
fp32—and why any custom reduction over 16-bit tensors should accumulate in fp32.

All three failure modes trace back to the same root: an intermediate quantity
outstripping the range or precision of a 16-bit format even when the final
answer fits comfortably. Remember also that mixed precision and tf32 alter the
lowest bits of arithmetic by design, so they will never match pure fp32 bit for
bit. When runs *can* be made bit-for-bit repeatable—and how to achieve it—is
the focus of :numref:`sec_repro`.

## Summary

Floating-point formats trade dynamic range (exponent bits) against precision
(mantissa bits): fp16 gives finer local steps but a narrow range (overflowing
past 65504 and underflowing tiny gradients), whereas bf16 keeps fp32's full
8-bit exponent at just two to three decimal digits of precision. Storing weights
directly in 16 bits rounds away small gradient updates and stalls training, so
mixed-precision training keeps fp32 master weights while running matrix
multiplications and storing activations in 16 bits—matching fp32 accuracy on our
Fashion-MNIST MLP where pure bf16 weights fall behind. With fp16, dynamic loss
scaling lifts gradients out of the subnormal/underflow zone during
backpropagation; bf16 avoids this step altogether. Finally, guard low-precision
code against overflow-induced NaNs by using shifted log-sum-exp operations and
fp32 accumulators for long reductions and normalizations.

## Exercises

1. **Mixed-precision memory.** Training with Adam in fp32 stores 16 bytes per
   parameter (:numref:`sec_parameters`). Suppose that `train` used Adam
   instead of SGD.
    1. In which dtype does the mixed-precision recipe of your framework store
       the weights, the gradients, and Adam's running averages between
       steps? Does mixed precision reduce the bytes per parameter?
    1. Count as stored activations the input of each linear layer and the
       logits. For the MLP of `train`, convert the per-example count to
       bytes in fp32 and in the 16-bit format `train` uses. At which batch
       size does the activation saving equal the persistent training state
       of this MLP?

2. **Loss scaling on overflow.** Under dynamic loss
   scaling (:numref:`subsec_loss_scaling`), a step whose unscaled gradients contain
   an `inf` or NaN is skipped and the scale is halved. After $N$
   consecutive clean steps, the scale is doubled.
    1. A skipped step must leave Adam's running averages untouched. Explain
       what would go wrong if the step instead updated them with the
       non-finite gradient, and what would go wrong if it updated them with
       the non-finite entries replaced by zeros.
    1. Explain why the scale must grow again after a run of clean steps
       instead of only shrinking.
    1. Suppose that every step overflows when the scale is above a fixed
       threshold and that no step overflows below it. Once the scale has
       settled, what fraction of steps is skipped as a function of $N$?
       Without the assumption of a fixed threshold, what does a larger $N$
       cost?

3. [code] **Normalization in fp16.** Mixed-precision recipes run
   normalization layers in fp32. To see why, consider RMSNorm with its
   gain set to one (:numref:`sec_custom_layer` implements it as a
   module). It rescales an input row $\mathbf{x}$ of width $d$ by its root
   mean square,

   $$\frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}},$$

   where the small constant $\epsilon$ guards against division by zero.
   Compute it entirely in fp16 for a batch of 64 rows of width $d = 1024$
   whose entries have a standard deviation of about 100.
    1. Which intermediate quantity is the first to become `inf`, and what
       does the layer return for the affected rows?
    1. Take the squares as exact. At what input standard deviation does
       their mean exceed the fp16 maximum, and at what standard deviation
       does their sum over the $d$ entries exceed it?

:begin_tab:`mxnet`
4. **The e4m3fn format.** ● The fp8 format `float8_e4m3fn` has one sign
   bit, 4 exponent bits with bias 7, and 3 mantissa bits. Derive its
   `tiny` and `eps` and compare them with those of fp16 and bf16. Explain
   the name `e4m3fn`. A format with the same bit layout that followed the
   IEEE convention would reserve the all-ones exponent for `inf` and NaN.
   Compute that format's largest finite value, and explain how
   `float8_e4m3fn` reaches a largest finite value of 448 instead.
:end_tab:

:begin_tab:`pytorch`
4. [code] **The e4m3fn format.** ● Print `torch.finfo(torch.float8_e4m3fn)`
   and compare its `max`, `tiny` and `eps` with those of fp16 and bf16.
   Explain the name `e4m3fn`. A format with a 4-bit exponent and a 3-bit
   mantissa that followed the IEEE convention would reserve the all-ones
   exponent for `inf` and NaN. Compute that format's largest finite value,
   and explain how `e4m3fn` reaches a `max` of 448 instead.
:end_tab:

:begin_tab:`tensorflow`
4. [code] **The e4m3fn format.** ● Print
   `ml_dtypes.finfo(ml_dtypes.float8_e4m3fn)` and compare its `max`,
   `tiny` and `eps` with those of fp16 and bf16. Explain the name
   `e4m3fn`. A format with a 4-bit exponent and a 3-bit mantissa that
   followed the IEEE convention would reserve the all-ones exponent for
   `inf` and NaN. Compute that format's largest finite value, and explain
   how `e4m3fn` reaches a `max` of 448 instead.
:end_tab:

:begin_tab:`jax`
4. [code] **The e4m3fn format.** ● Print `jnp.finfo(jnp.float8_e4m3fn)` and
   compare its `max`, `tiny` and `eps` with those of fp16 and bf16.
   Explain the name `e4m3fn`. A format with a 4-bit exponent and a 3-bit
   mantissa that followed the IEEE convention would reserve the all-ones
   exponent for `inf` and NaN. Compute that format's largest finite value,
   and explain how `e4m3fn` reaches a `max` of 448 instead.
:end_tab:


[Discussions](https://d2l.discourse.group/t/36756)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.5]{.kicker}

Numerical Precision and Mixed-Precision Training<br>**fp32 vs. bf16 vs. fp16 · update rounding · fp32 master weights · dynamic loss scaling · numerical hazards**
:::
:::

::: {.slide title="16-Bit Formats Halve Memory but Bring Two Hazards"}
[The Low-Precision Trade-Off]{.kicker}

Dropping from 32-bit floating point (`fp32`) to 16-bit precision cuts an 8B model's weight storage from `32 GB` to `16 GB` and doubles GPU Tensor Core throughput, but introduces two silent failure modes:

@numerics-the-dtype-zoo-1

- **Range overflow in `fp16`:** Squaring a modest value like $300.0$ overflows straight to `inf` because `fp16`'s maximum finite number is $65{,}504$.
- **Precision rounding in `bf16`:** `bf16` keeps $300^2$ finite, yet rounds $90{,}000$ to $90{,}112$—and adding $10^{-3}$ to $1.0$ rounds right back to $1.0$, silently freezing parameter updates.
:::

::: {.slide title="Anatomy of a Floating-Point Number: Range vs. Precision"}
[Exponent vs. Mantissa Bits]{.kicker}

Any normal IEEE floating-point number encodes a sign bit $s$, an exponent $e$, and a $t$-bit mantissa fraction $f \in [0, 1)$:

$$x = (-1)^s \cdot 2^e \cdot (1 + f)$$

- **Exponent bits govern dynamic range:** More exponent bits let $2^e$ span much smaller (`tiny`) and much larger (`max`) magnitudes before underflowing to $0$ or overflowing to `inf`.
- **Mantissa bits govern relative precision:** Within any octave $[2^e, 2^{e+1})$, adjacent representable numbers sit $2^{e-t}$ apart, maintaining uniform relative spacing $\text{eps} = 2^{-t}$ across all scales.

| Format | Total Bits | Sign | Exponent Bits | Mantissa Bits ($t$) | Max Finite Value | Relative Spacing at $1$ ($2^{-t}$) |
|:---|---:|---:|---:|---:|---:|---:|
| **`float32` (`fp32`)** | 32 | 1 | **8** | **23** | $3.40 \times 10^{38}$ | $1.19 \times 10^{-7}$ ($\sim 7$ digits) |
| **`bfloat16` (`bf16`)** | 16 | 1 | **8** | **7** | $3.39 \times 10^{38}$ | $7.81 \times 10^{-3}$ ($\sim 2\text{--}3$ digits) |
| **`float16` (`fp16`)** | 16 | 1 | **5** | **10** | $6.55 \times 10^{4}$ | $9.77 \times 10^{-4}$ ($\sim 3\text{--}4$ digits) |
:::

::: {.slide title="Bit Layouts of fp32, bf16, and fp16 Compared"}
[Opposite 16-Bit Partitions]{.kicker}

![](../img/bg-float-formats.svg){.nostretch width=84%}

- **`bf16` truncates `fp32`:** Preserving all 8 exponent bits gives `bf16` the exact dynamic range of `fp32`, making overflow and underflow nearly as rare as in 32-bit training.
- **`fp16` prioritizes mantissa over range:** Its 10 mantissa bits resolve $8\times$ finer steps than `bf16`, but its 5-bit exponent overflows above $65{,}504$ and drops into subnormals below $6.1 \times 10^{-5}$.
:::

::: {.slide title="Querying Machine Limits with finfo"}
[Exact Format Boundaries]{.kicker}

We can inspect `max` (largest finite value), `tiny` (smallest positive normal number), and `eps` ($2^{-t}$ gap at $1.0$) directly in code:

@numerics-the-dtype-zoo-2

- **Why range beats precision in deep learning:** Activations and gradients span many orders of magnitude across layers; running out of range turns values into `inf` or $0$, whereas a coarser mantissa merely adds mild quantization noise.
:::

::: {.slide title="Why 16-Bit Weights Stall Training: Small-Update Rounding"}
[The Parameter Update Swamping Trap]{.kicker}

A typical gradient descent update $\Delta w = -\eta g$ shifts a weight by $10^{-3}$ or less of its magnitude. Under round-to-nearest-even, any update smaller than $\frac{1}{2}\text{eps}\cdot |w|$ rounds to zero:

@numerics-rounding-of-small-updates

::: {.d2l-note .rule}
**Master Weights Rule:** Near $w = 1.0$, `bf16` spacing is $2^{-7} \approx 0.0078$, so both $1 + 10^{-3}$ and $1 + 10^{-4}$ evaluate to $1.0000000$. Training **must** accumulate parameter updates in `fp32`.
:::
:::

::: {.slide title="Dtype Promotion Rules Across Frameworks" except="tensorflow"}
[Mixed-Operand Arithmetic]{.kicker}

When an expression combines two floating-point tensors of different widths, PyTorch, JAX, and MXNet promote the result to the wider format (`fp16 + fp32 -> fp32`, and `fp16 + bf16 -> fp32`):

@numerics-dtype-rules-promotion-parameters-and-casts-1

- **The stray `fp32` contamination bug:** A single uncast `fp32` buffer or bias silently promotes the entire downstream 16-bit forward pass back to `fp32`, doubling activation memory.
- **Weak Python scalars:** Adding a Python literal (`x16 + 1.0`) or integer tensor keeps `x16` in 16-bit precision.
:::

::: {.slide title="Explicit Dtype Checking in TensorFlow" only="tensorflow"}
[Mixed-Operand Arithmetic]{.kicker}

Unlike PyTorch and JAX, TensorFlow refuses to promote across distinct tensor dtypes—combining `fp16` and `fp32` raises an immediate `InvalidArgumentError`:

@numerics-dtype-rules-promotion-parameters-and-casts-1

- **Explicit `tf.cast` discipline:** Every cross-dtype interaction must be written explicitly with `tf.cast`, while Python float literals (`x16 + 1.0`) automatically adopt the tensor's dtype.
:::

::: {.slide title="Casting a Model to 16 Bits Halves Inference Memory"}
[Whole-Model Dtype Casting]{.kicker}

Our Fashion-MNIST MLP ($784 \to 256 \to 10$) holds $203{,}530$ parameters (`814,120 bytes` in `fp32`). Walking the state tree to cast parameters to 16 bits cuts weight memory in half to `407,060 bytes`:

@-numerics-dtype-rules-promotion-parameters-and-casts-2

@numerics-dtype-rules-promotion-parameters-and-casts-3

- **Ideal for inference, flawed for training:** Pure 16-bit weights halve inference memory bandwidth, but training them directly hits the small-update rounding trap.
:::

::: {.slide title="Mixed Precision: fp32 Master Weights + 16-Bit Compute"}
[The AMP Training Loop]{.kicker}

**Mixed-precision training** (Micikevicius et al., 2018) gets both 16-bit Tensor Core speed and 32-bit update accuracy by decoupling compute precision from storage precision:

![](../img/bg-amp-loop.svg){.nostretch width=95%}

1. **Cast for compute:** Cast `fp32` master weights to `bf16` (or `fp16`) for forward and backward matrix multiplications and activation storage.
2. **Reduce in `fp32`:** Compute loss functions, softmaxes, and layer normalizations in `fp32`.
3. **Update in `fp32`:** Apply gradients (and Adam's `fp32` moments $\mathbf{m}, \mathbf{v}$) directly to the `fp32` master weights.
:::

::: {.slide title="Enabling Mixed Precision in Four Lines of Code"}
[Framework Autocast & Policies]{.kicker}

Each framework provides a clean policy or context manager (`torch.autocast`, `'mixed_bfloat16'`, `dtype=jnp.bfloat16`, or `multi_precision=True`) that keeps weights in `fp32` while outputting 16-bit activations:

@numerics-mixed-precision-training-1
:::

::: {.slide title="Pure bf16 Weights Stall; Mixed Precision Matches fp32"}
[Fashion-MNIST Training Comparison]{.kicker}

Training identical Fashion-MNIST MLPs for $1{,}000$ SGD steps at $\eta = 0.01$ across three precision modes shows why `fp32` master weights are non-negotiable:

@!numerics-mixed-precision-training-4

- **90% of `bf16` updates vanish:** With pure `bf16` weights, roughly $90\%$ of first-layer updates fall below half the local `bf16` spacing, stalling loss at $\sim 1.15$ while mixed precision tracks `fp32` ($\sim 0.85$) virtually on top of each other.
:::

::: {.slide title="Why Mixed Precision Keeps the 128 GB Static Floor"}
[Training vs. Inference Memory]{.kicker}

A common misconception is that mixed-precision training halves total GPU memory. Let's audit the persistent per-parameter state for our 8B model under Adam:

| Persistent Tensor | Pure `fp32` Training | Mixed-Precision Training (`bf16` / `fp16`) | Pure `bf16` Inference |
|:---|---:|---:|---:|
| **`fp32` Master weights** | $4\text{ B}$ (`32 GB`) | $4\text{ B}$ (`32 GB`) | — (`0 GB`) |
| **16-bit weights / Gradients** | $4\text{ B}$ (`32 GB`) | $2\text{--}4\text{ B}$ (`16–32 GB`) | $2\text{ B}$ (`16 GB`) |
| **Adam `fp32` moments ($\mathbf{m}, \mathbf{v}$)** | $8\text{ B}$ (`64 GB`) | $8\text{ B}$ (`64 GB`) | — (`0 GB`) |
| **Static Parameter Floor (8B)** | **$16\text{ B}$ (`128 GB`)** | **$\ge 14\text{--}16\text{ B}$ (`112–128 GB`)** | **$2\text{ B}$ (`16 GB`)** |

- **Where mixed precision saves memory:** It halves the **activation tensors** cached during the forward pass (which often dominate large-batch or long-context training) and doubles matrix-multiply speed.
:::

::: {.slide title="Gradient Underflow in fp16 and Dynamic Loss Scaling"}
[Why fp16 Needs Loss Scaling]{.kicker}

Unlike `bf16`, `fp16`'s narrow 5-bit exponent drops into subnormals below $2^{-14} \approx 6.1 \times 10^{-5}$ and underflows to exact zero below $3 \times 10^{-8}$. Measuring our MLP's first-layer gradients on batch 0 reveals the damage:

@numerics-loss-scaling-for-fp16-3

- **Linearity of differentiation:** Multiplying the loss by $S = 2^{16} = 65{,}536$ before `backward()` scales every intermediate gradient by $S$, lifting all $33\%$ subnormal gradients into normal `fp16` range, after which we divide gradients by $S$ in `fp32` before `optimizer.step()`.
:::

::: {.slide title="Automatic Dynamic Loss Scaling in Practice" only="pytorch,tensorflow"}
[GradScaler & LossScaleOptimizer]{.kicker}

**Dynamic loss scaling** starts with a large scale factor $S$, skips the optimizer step and halves $S$ whenever an `inf` or `NaN` gradient appears, and doubles $S$ after a streak of finite steps:

@numerics-loss-scaling-for-fp16-2
:::

::: {.slide title="When to Use bf16 vs. fp16 vs. TF32"}
[Hardware Precision Selection Guide]{.kicker}

Modern accelerators offer three distinct faster-than-`fp32` execution modes on Tensor Cores:

- **`bfloat16` mixed precision (Default on Ampere/Hopper/TPU):** Uses `fp32` master weights + `bf16` compute with **no loss scaling needed** because `bf16` shares `fp32`'s 8-bit exponent range.
- **`float16` mixed precision + dynamic loss scaling (Older Volta/Turing GPUs):** Uses `fp32` master weights + `fp16` compute + `GradScaler` to prevent gradient underflow.
- **TensorFloat-32 (`TF32`) math mode:** Operates on standard `fp32` tensors without code changes; Tensor Cores round matrix-multiply inputs internally to 10 mantissa bits (8-bit exponent) and accumulate inner products in full `fp32`.
:::

::: {.slide title="Numerical Hazards: Overflow to NaN and Log-Sum-Exp"}
[Guarding 16-Bit Range]{.kicker}

In `fp16`, intermediate values exceeding $65{,}504$ overflow to `inf`—and subsequent indeterminate arithmetic (`inf - inf`) turns the loss into `NaN`:

@numerics-when-numerics-bite-1

@numerics-when-numerics-bite-2

- **Shifted Log-Sum-Exp:** Subtracting $m = \max_i x_i$ before exponentiating ($\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}$) prevents $e^{12} \approx 163{,}000 > 65{,}504$ from overflowing while keeping the largest term at $e^0 = 1$.
:::

::: {.slide title="Numerical Hazard: Running Sums Stall in 16 Bits"}
[Accumulating Reductions in fp32]{.kicker}

Long reductions accumulated in 16 bits freeze once individual summands fall below half the accumulator's grid spacing:

@numerics-when-numerics-bite-3

- **Why `total16` freezes at `2.0`:** Between $2$ and $4$, adjacent `fp16` numbers sit $2^{-9}$ apart, so $2 + 2^{-10}$ hits the exact midpoint and round-to-nearest-even snaps back to $2.0$ forever; always accumulate sums, means, and normalizations in `fp32`.
:::

::: {.slide title="Concept Check: Floating-Point Formats & Mixed Precision"}
[Concept Check]{.kicker}

Work out these three numerical precision questions by hand:

1. In `bfloat16` (7 mantissa bits), what is the grid spacing between adjacent representable numbers in the interval $[512, 1024)$, and what does `bf16(1000.0) + bf16(1.0)` evaluate to?
2. A gradient entry has magnitude $g = 2 \times 10^{-6}$. In unscaled `fp16` (`tiny` $= 6.1 \times 10^{-5}$, min subnormal $= 6.0 \times 10^{-8}$), is $g$ normal, subnormal, or zero? What happens when scaled by $S = 2^{16}$?
3. When dynamic loss scaling encounters an `inf` gradient on step $t$, why must it skip updating Adam's moments $\mathbf{m}$ and $\mathbf{v}$ rather than zeroing out the `inf` entry?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Spacing is $2^{9 - 7} = 4.0$, so $1000.0 + 1.0 = 1000.0$:** Since $512 = 2^9$, adjacent `bf16` values sit $4$ apart; an increment of $1$ is less than half a grid step ($2.0$) and vanishes.
2. **Subnormal unscaled ($\sim 5$ bits of precision left); normal when scaled:** $S \cdot g = 65{,}536 \times 2 \times 10^{-6} \approx 0.131$, restoring full 10-bit `fp16` precision.
3. **Updating $\mathbf{m}, \mathbf{v}$ with `inf` permanently poisons Adam with `NaN`s**, while replacing `inf` with $0$ corrupts the running moment history for the largest-gradient parameters.
:::
:::

::: {.slide title="Summary: Numerical Formats and Mixed Precision"}
[Key Takeaways]{.kicker}

| Format / Technique | Bit Layout / Rule | Role in Deep Learning Systems |
|:---|:---|:---|
| **`fp32` (`float32`)** | 1 sign, 8 exp, 23 mantissa ($\text{eps} \approx 1.2 \times 10^{-7}$) | Master weights, Adam moments $\mathbf{m}, \mathbf{v}$, losses, and reductions |
| **`bf16` (`bfloat16`)** | 1 sign, 8 exp, 7 mantissa ($\text{eps} \approx 0.0078$) | Preferred 16-bit compute/activation format; `fp32` range without loss scaling |
| **`fp16` (`float16`)** | 1 sign, 5 exp, 10 mantissa ($\max = 65{,}504$) | Requires dynamic loss scaling ($S \cdot L$) to prevent gradient underflow |
| **Mixed precision** | `fp32` master weights $+$ 16-bit matmuls/activations | Matches `fp32` convergence while halving activation RAM & doubling speed |
| **Stable reductions** | Shifted $\text{logsumexp}$ and `fp32` sum accumulators | Prevents $e^x > 65{,}504$ overflow and running-sum stagnation |
:::
