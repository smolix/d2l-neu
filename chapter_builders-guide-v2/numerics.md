# Numerics: Dtypes and Mixed Precision
:label:`sec_numerics_v2`

Every tensor carries three attributes: a shape, a device, and a *dtype*, the
numeric format of its entries. So far we have left the dtype at its default,
32-bit floating point, and nothing forced us to look at it. That grace period
is over. Modern accelerators multiply 16-bit matrices several times faster
than 32-bit ones, in half the memory, and essentially all serious training now
chooses numeric formats deliberately. This section covers what a builder needs:
which formats exist, what each one can and cannot represent, how dtypes combine,
and the standard recipe (*mixed precision*) for training in 16 bits without
giving up 32-bit accuracy. For the anatomy of a floating-point number (sign,
exponent, mantissa) see :numref:`sec_mdl-numerical-stability-conditioning`;
here we ask the practical questions: when does it break, and which switch do
I flip.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab jax
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from d2l import jax as d2l
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab tensorflow
import ml_dtypes
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #numerics-numerics-dtypes-and-mixed-precision}
%%tab mxnet
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

## The Dtype Zoo

Half-precision (`float16`, "fp16") sounds like a free lunch: half the
bytes of fp32, and the format that accelerator hardware sped up first. Here
is the catch. The largest number fp16 can represent is 65504. Square a value
of 300, which is nothing exotic (an unnormalized logit, an intermediate in a
variance computation), and you have already left the representable range:

```{.python .input #numerics-the-dtype-zoo-1}
%%tab pytorch
x = torch.tensor(300.0)
x.to(torch.float16)**2, x.to(torch.bfloat16)**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab jax
x = jnp.array(300.0)
x.astype(jnp.float16)**2, x.astype(jnp.bfloat16)**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab tensorflow
x = tf.constant(300.0)
tf.cast(x, tf.float16)**2, tf.cast(x, tf.bfloat16)**2
```

```{.python .input #numerics-the-dtype-zoo-1}
%%tab mxnet
x = np.array(300.0)
x.astype('float16')**2
```

fp16 overflows to `inf`. The second format, `bfloat16` ("bf16", brain
floating point), returns 90112: wrong in the fourth digit, since the exact
square is 90000, but finite. The two formats spend the same 16 bits
differently. fp16 uses 5 exponent bits and 10 mantissa bits: fine-grained
steps, tiny range. bf16 keeps fp32's full 8 exponent bits and pays with a
7-bit mantissa: fp32's range, coarse steps. The one-line demo above shows both
consequences at once.

:begin_tab:`mxnet`
The MXNet cell shows only the first half of the demo, because `mx.np` arrays
come in `float16`, `float32`, and `float64` and nothing else. bfloat16 exists
deeper in the engine (the `mxnet.amp` module lists it as a cast target) but is
not a storage dtype you can `astype` an array to, so the bf16 result quoted
above comes from the other frameworks. The fp16 half is unchanged: 300
squared overflows to `inf`.
:end_tab:

| format | sign | exponent | mantissa |
|:-------|-----:|---------:|---------:|
| fp32   | 1    | 8        | 23       |
| tf32   | 1    | 8        | 10       |
| bf16   | 1    | 8        | 7        |
| fp16   | 1    | 5        | 10       |
| fp8 (e4m3) | 1 | 4       | 3        |

:numref:`fig_bg_float-formats` draws the same table as bit layouts, aligned
at the sign bit so the total widths compare directly: fp32 spends its 32 bits
on a wide mantissa, while bf16 keeps fp32's 8-bit exponent but pays for it
with a narrow 7-bit mantissa, and fp16 inverts the trade.

![Bit layouts of the five formats in the table above, aligned at the sign bit so a wider bar means more total bits: fp32 is the widest, tf32 shares fp32's exponent but truncates the mantissa to 10 bits, and fp16 trades exponent bits for mantissa bits that bf16 spends the other way.](../img/bg-float-formats.svg)
:label:`fig_bg_float-formats`

`finfo` reports what each bit budget buys. Three numbers matter: `max`,
the overflow threshold; `tiny`, the smallest normal value before underflow to
zero; and `eps`, the relative step size between adjacent representable values.

```{.python .input #numerics-the-dtype-zoo-2}
%%tab pytorch
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    fi = torch.finfo(dtype)
    print(f'{str(dtype):15} max {fi.max:10.3e}  tiny {fi.tiny:9.3e}'
          f'  eps {fi.eps:8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab jax
for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
    fi = jnp.finfo(dtype)
    print(f'{fi.dtype.name:10} max {float(fi.max):10.3e}'
          f'  tiny {float(fi.tiny):9.3e}  eps {float(fi.eps):8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab tensorflow
for dtype in (tf.float32, tf.bfloat16, tf.float16):
    fi = ml_dtypes.finfo(dtype.as_numpy_dtype)
    print(f'{dtype.name:10} max {float(fi.max):10.3e}'
          f'  tiny {float(fi.tiny):9.3e}  eps {float(fi.eps):8.3e}')
```

```{.python .input #numerics-the-dtype-zoo-2}
%%tab mxnet
for dtype in ('float32', 'float16'):
    fi = np.finfo(dtype)
    print(f'{dtype:10} max {fi.max:10.3e}  tiny {fi.smallest_normal:9.3e}'
          f'  eps {fi.eps:8.3e}')
```

:begin_tab:`tensorflow`
TensorFlow has no `finfo` of its own (the `tf.experimental.numpy` one chokes
on bfloat16). The numbers live one level down, in `ml_dtypes`, the small
NumPy-extension package that defines the bfloat16 scalar type TensorFlow
depends on; each `tf` dtype hands over its NumPy counterpart via
`as_numpy_dtype`.
:end_tab:

:begin_tab:`mxnet`
`mx.np.finfo` follows the array-API naming, so the smallest normal value is
the field `smallest_normal` rather than `tiny`; under the hood it hands the
dtype to NumPy and repackages the answer. NumPy knows no bfloat16 and MXNet
stores none, so the loop covers fp32 and fp16 and we state bf16's numbers for
the record: `max` $3.39 \times 10^{38}$ and `tiny` $1.18 \times 10^{-38}$,
matching fp32's exponent range, with `eps` $2^{-7} \approx 0.0078$.
:end_tab:

bf16 matches fp32's `max` and `tiny` exactly (same exponent bits) and its
`eps` of 0.0078 means two to three significant decimal digits. fp16 resolves
about three to four digits but overflows at 65504 and underflows below
$6 \times 10^{-5}$. The trade is precision against range, and for deep
learning the choice is lopsided: activations and gradients span many orders
of magnitude, occasional large values are routine, and running out of range
produces `inf` while losing a low-order digit usually costs nothing a noisy
gradient estimate had to offer anyway. That asymmetry is why bf16 became the
default 16-bit format for training.

### TF32: What Happens to fp32 Matrix Multiplication

The second row of the table, tf32, is not a storage dtype; you cannot create a
tensor of it. It is a compute mode of NVIDIA tensor cores (Ampere generation
and later): matrix-multiply inputs are rounded to a 10-bit mantissa while
keeping fp32's exponent, and products are accumulated in fp32. Your tensors
stay fp32; only the arithmetic inside the multiplication runs faster and
slightly less precisely. Most of our frameworks expose a switch for it:

```{.python .input #numerics-tf32-what-happens-to-fp32-matrix-multiplication}
%%tab pytorch
print(torch.get_float32_matmul_precision())
torch.set_float32_matmul_precision('high')  # allow tf32 in fp32 matmuls
print(torch.get_float32_matmul_precision(),
      torch.backends.cuda.matmul.allow_tf32)
torch.set_float32_matmul_precision('highest')  # restore the default
```

```{.python .input #numerics-tf32-what-happens-to-fp32-matrix-multiplication}
%%tab jax
print(jax.config.jax_default_matmul_precision)
with jax.default_matmul_precision('tensorfloat32'):
    print(jax.config.jax_default_matmul_precision)
```

```{.python .input #numerics-tf32-what-happens-to-fp32-matrix-multiplication}
%%tab tensorflow
print(tf.config.experimental.tensor_float_32_execution_enabled())
tf.config.experimental.enable_tensor_float_32_execution(False)
print(tf.config.experimental.tensor_float_32_execution_enabled())
tf.config.experimental.enable_tensor_float_32_execution(True)  # the default
```

:begin_tab:`pytorch`
The default is `'highest'`: fp32 matrix multiplications compute in true fp32
and the tensor-core shortcut stays off. Setting `'high'` opts in (it also
flips the older `allow_tf32` flag; they are two views of one setting). The
history is a trap for readers of old tutorials: PyTorch 1.7 through 1.11
enabled tf32 *by default*, and the default changed to off in 1.12 without a
warning. Material from that era asserts that tf32 is automatic on Ampere GPUs;
on any current version it is not, and the check above is how you find out what
your process is actually doing. For training, `'high'` is generally safe
(products still accumulate in fp32) and substantially faster on tensor-core
hardware; keep `'highest'` for ill-conditioned linear algebra or when
reproducing results bit for bit.
:end_tab:

:begin_tab:`jax`
The unset config (`None`) means "let the backend pick", and on tensor-core
GPUs XLA's pick for fp32 matrix multiplications is the fast tf32 path: the
opposite polarity from PyTorch, which today makes you opt in. You opt *out*
by requesting `'float32'` (or `'highest'`), and the setting is a scoped
context manager rather than a global flag, so a numerically delicate block
can demand full precision without touching the rest of the program.
Individual operations accept the same choice per call, as in
`jnp.dot(A, B, precision='float32')`. On the CPU this notebook runs on there
are no tensor cores and every setting computes plain fp32, which is why the
cell above changes nothing but the config value; the distinction takes effect
on the GPUs of :numref:`sec_use_gpu_v2`. For training, tf32 is generally safe
(products still accumulate in fp32); ask for `'float32'` when doing
ill-conditioned linear algebra or reproducing results bit for bit.
:end_tab:

:begin_tab:`tensorflow`
The getter answers `True`: TensorFlow turns tf32 on by default wherever the
hardware supports it, the polarity JAX shares and today's PyTorch inverts,
so an fp32 matrix multiplication on an Ampere-or-later GPU is already taking
the fast path unless you say otherwise. The switch is global, not scoped: a
numerically delicate block disables it, computes, and restores it, as the
cell does. On the CPU this notebook runs on there are no tensor cores and
the flag changes nothing beyond its own value; the distinction takes effect
on the GPUs of :numref:`sec_use_gpu_v2`. For training, the default is
generally safe (products still accumulate in fp32); disable it for
ill-conditioned linear algebra or when reproducing results bit for bit.
:end_tab:

:begin_tab:`mxnet`
MXNet is the exception: no cell here, because no user-facing switch exists.
Nothing in the Python API reads or sets the fp32-matmul precision; whether an
fp32 matrix multiplication takes the tensor-core shortcut is decided below
MXNet, by the CUDA libraries the wheel links against. If you need the
guarantees the other tabs configure (true-fp32 matmuls for ill-conditioned
linear algebra, or bit-for-bit reproduction), MXNet does not give you the
knob, and the honest workaround is to do that computation in `float64` or
outside the framework.
:end_tab:

### Below 16 Bits

Production inference pushes further down: int8 quantization is standard for
serving, and fp8 training (a 4-bit-exponent variant for the forward pass, a
5-bit-exponent variant for gradients, with per-tensor scaling) runs on
H100-class hardware :cite:`Micikevicius.Stosic.Burgess.ea.2022`. Both require
calibration machinery beyond a dtype argument, so for this book 16 bits is the
floor; the exercises let you inspect the fp8 format's `finfo`.

:begin_tab:`jax`
The fp8 pair already ships in `jnp` (`jnp.float8_e4m3fn` for the forward
pass, the wider-range `jnp.float8_e5m2` for gradients), and `jnp.finfo` reads
them like any other float; what still lives in specialized libraries is the
per-tensor scaling that makes them trainable.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow 2.21 ships no fp8 dtype. The formats exist one level down, in the
same `ml_dtypes` package that supplied our `finfo`
(`ml_dtypes.finfo(ml_dtypes.float8_e4m3fn)` reads them), but no `tf` dtype
wraps them; fp8 work in the TensorFlow world happens in specialized
libraries.
:end_tab:

:begin_tab:`mxnet`
MXNet ships no fp8 dtype at any level: not in `mx.np`, not in the engine's
dtype tables. For this tab the 16-bit floor is not a book decision but a hard
one; fp8 experiments mean leaving the framework.
:end_tab:

## Dtype Rules: Promotion, Parameters, and Casts

:begin_tab:`pytorch`
What happens when dtypes meet in one expression? For plain tensors, PyTorch
promotes to the type that can represent both:
:end_tab:

:begin_tab:`jax`
What happens when dtypes meet in one expression? For plain arrays, JAX
promotes to a common type that can represent both:
:end_tab:

:begin_tab:`tensorflow`
What happens when dtypes meet in one expression? For plain tensors,
TensorFlow refuses to guess. There is no promotion; an operation whose
inputs disagree raises on the spot:
:end_tab:

:begin_tab:`mxnet`
What happens when dtypes meet in one expression? For plain arrays, MXNet
promotes to the type that can represent both:
:end_tab:

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab pytorch
x16 = torch.ones(3, dtype=torch.float16)
x32 = torch.ones(3, dtype=torch.float32)
(x16 + x32).dtype, (x16 + 1.0).dtype, (x16 + torch.arange(3)).dtype
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab jax
x16 = jnp.ones(3, dtype=jnp.float16)
x32 = jnp.ones(3, dtype=jnp.float32)
(x16 + x32).dtype, (x16 + 1.0).dtype, (x16 + jnp.arange(3)).dtype
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab tensorflow
x16 = tf.ones(3, dtype=tf.float16)
x32 = tf.ones(3, dtype=tf.float32)
try:
    x16 + x32
except tf.errors.InvalidArgumentError as e:
    print(e.message)
(x16 + 1.0).dtype
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab mxnet
x16 = np.ones(3, dtype='float16')
x32 = np.ones(3, dtype='float32')
((x16 + x32).dtype, (x16 + 1.0).dtype,
 np.result_type(x16, np.ones(3, dtype='int32')))
```

:begin_tab:`pytorch, jax`
Mixing two float tensors upcasts to the wider one, so an fp16 pipeline with a
stray fp32 tensor quietly becomes fp32 from that point on, doubling downstream
memory. Python scalars and integer tensors are *weak*: they adopt the float
tensor's dtype instead of dragging it up, which is why sprinkling literals
like `x + 1.0` into low-precision code is harmless. (Mixing fp16 with bf16
promotes to fp32, since neither contains the other.)
:end_tab:

:begin_tab:`tensorflow`
Only the Python scalar got through: literals like `x + 1.0` are cast to the
tensor's dtype, so sprinkling them into low-precision code is harmless.
Everything else, float with float or float with integer, is an error at the
first operation that touches both. This strictness guards against exactly
the failure that promotion invites in other frameworks, where an fp16
pipeline with a stray fp32 tensor quietly becomes fp32 from that point on,
doubling downstream memory; in TensorFlow the stray tensor cannot travel one
op. The price is that every intended conversion is spelled `tf.cast`.
:end_tab:

:begin_tab:`mxnet`
Mixing two float arrays upcasts to the wider one, so an fp16 pipeline with a
stray fp32 array quietly becomes fp32 from that point on, doubling downstream
memory. Python scalars are *weak*: they adopt the array's dtype instead of
dragging it up, which is why sprinkling literals like `x + 1.0` into
low-precision code is harmless. Integer arrays behave the same way, and the
wheel's source says where the rule comes from: its promotion table is
annotated "PyTorch convention: the floating operand keeps its width
regardless of the integer operand". The third expression asks that table
directly through `np.result_type` rather than running a mixed-integer kernel.
:end_tab:

:begin_tab:`pytorch`
Module parameters play by a stricter rule: layers do not promote, they demand
a matching input dtype and raise otherwise. To change a model's dtype you cast
the whole module; `net.to(dtype)` (or the shorthand `net.bfloat16()`) converts
every parameter and buffer in place. The byte accounting of
:numref:`sec_parameters_v2` composes with dtype through `element_size()`:
:end_tab:

:begin_tab:`jax`
Parameters bring a second rule, and in flax it is written into every layer's
constructor. A layer takes two dtype arguments: `param_dtype`, the storage
format `init` creates parameters in (default fp32), and `dtype`, the format
the forward pass computes in (default `None`, which promotes parameters and
inputs to a common type, the same rule as above). Since the parameters
themselves are a plain pytree of arrays, "casting the model" is one
`tree.map`. The byte accounting of :numref:`sec_parameters_v2` composes with
dtype through `itemsize`:
:end_tab:

:begin_tab:`tensorflow`
Keras layers package this strictness behind a *dtype policy*: every layer
carries a `variable_dtype`, the format its weights are stored in, and a
`compute_dtype`, the format its forward pass runs in, both fp32 by default.
The constructor argument `dtype='bfloat16'` sets the pair at once, so
"casting the model" is a construction-time choice rather than an in-place
conversion. The byte accounting of :numref:`sec_parameters_v2` composes with
dtype through the dtype's size in bytes:
:end_tab:

:begin_tab:`mxnet`
Parameters bring a second rule: Gluon operators demand that input and weight
dtypes agree (type inference unifies them and errors on a mismatch), so to
change a model's dtype you cast the whole block. `net.cast('float16')`
converts every parameter recursively, children included; inputs you cast
yourself. The byte accounting of :numref:`sec_parameters_v2` composes with
dtype through the parameter array's `itemsize`:
:end_tab:

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab pytorch
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))
def param_bytes(net):
    return sum(p.numel() * p.element_size() for p in net.parameters())
param_bytes(net)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab jax
net = nn.Sequential([lambda x: x.reshape((x.shape[0], -1)),
                     nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), jnp.zeros((1, 28, 28, 1)))
def param_bytes(params):
    return sum(p.size * p.dtype.itemsize for p in jax.tree.leaves(params))
param_bytes(params)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(10)])
net(tf.zeros((1, 28, 28, 1)))  # build the weights
def param_bytes(net):
    return sum(w.shape.num_elements() * tf.as_dtype(w.dtype).size
               for w in net.weights)
param_bytes(net)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
net.initialize()
X = np.random.normal(size=(8, 1, 28, 28))
net(X)  # materialize the weights (initialization is deferred)
def param_bytes(net):
    return sum(p.data().size * p.data().dtype.itemsize
               for p in net.collect_params().values())
param_bytes(net)
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
%%tab jax
params16 = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
X = jax.random.normal(d2l.get_key(), (8, 28, 28, 1))
print(net.apply(params16, X).dtype, param_bytes(params16))
print(net.apply(params16, X.astype(jnp.bfloat16)).dtype)
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab tensorflow
net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', dtype='bfloat16'),
    tf.keras.layers.Dense(10, dtype='bfloat16')])
X = tf.random.normal((8, 28, 28, 1))
print(net(X).dtype, param_bytes(net))
```

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-3}
%%tab mxnet
net.cast('float16')
print(net(X.astype('float16')).dtype, param_bytes(net))
```

:begin_tab:`pytorch`
The parameter footprint halves, inference works, and the error message shows
the strictness: the fp32 input was refused rather than silently converted.
:end_tab:

:begin_tab:`jax`
The parameter footprint halves, but look at the first output: fp32. With
`dtype=None` the layer promoted the bf16 parameters and the fp32 input to a
common type, so casting the parameters alone bought memory but not 16-bit
compute; the fp32 input dragged the arithmetic straight back up. Feeding a
bf16 input (second line) keeps the whole forward pass in bf16, and so does
constructing the layers with `dtype=jnp.bfloat16`, which pins the compute
format regardless of what comes in. Nothing is refused and nothing raises:
in flax the dtype story is promotion plus two explicit arguments.
:end_tab:

:begin_tab:`tensorflow`
The parameter footprint halves, and note what did not happen: no error,
although we fed an fp32 input to a bf16 model. The strictness of the
previous cell lives at the level of raw operations; a Keras layer casts
floating-point inputs to its `compute_dtype` at the door, so by the time
any op runs, the dtypes already agree. The layer's policy decides the
arithmetic, not the input, and both faces of TensorFlow serve the same end:
ops refuse to guess, layers make the choice explicit at construction.
:end_tab:

:begin_tab:`mxnet`
The parameter footprint halves and the forward pass returns fp16. The input
was cast too: feed the fp32 `X` to the fp16 network and type inference
refuses the mismatch, the same strictness as PyTorch with the error raised
one level lower, by the operator. (No `Flatten` layer either; Gluon's
`Dense` flattens trailing dimensions by default.)
:end_tab:

Casting the model like this is the right tool for *inference*: half the
memory, no gradients to worry about, and a rounding error in the forward pass
rarely changes an argmax. For *training* it is a trap. A single optimizer
update changes a weight by roughly $\eta \cdot g$, often a factor $10^{-4}$
or less of the weight's own magnitude, and adding an increment smaller than
about `eps` times the weight rounds to no change at all. With bf16's `eps` of
0.0078, small updates evaporate and learning stalls; in fp16 the small
gradients themselves flush to zero first. Hence the rule, and it resolves the
single most common confusion in practice:

**Cast the model for inference. For training, keep fp32 weights and run the
compute in 16 bits.**

## Mixed-Precision Training

Mixed-precision training :cite:`Micikevicius.Narang.Alben.ea.2018` splits the
work: parameters stay in fp32 (the *master weights*, so that small updates
still register), while the expensive operations of the forward and backward
pass run in a 16-bit dtype.

:begin_tab:`pytorch`
You do not annotate anything per layer. Inside a
`torch.autocast` context, each operation consults a built-in policy: matrix
multiplications and convolutions, which dominate compute and map onto tensor
cores, run in the low dtype; operations that accumulate many terms or
exponentiate run in fp32. PyTorch maintains the per-operation lists, and
inputs are cast on the fly.
:end_tab:

:begin_tab:`jax`
In flax this split is not a context manager; it is the pair of constructor
arguments from the previous section. Leave `param_dtype` at its fp32 default
and set `dtype=jnp.bfloat16`, and each layer stores fp32 parameters while its
matrix multiplication runs in bf16: master weights and 16-bit compute, spelled
out in the layer definition. There is no built-in per-operation policy to
consult, and the flip side of that explicitness is that anything you compute
outside the model, such as the loss reduction, stays in whatever dtype you
give it; we cast logits to fp32 before the loss for exactly the reason the
policy-based frameworks pin reductions there.
:end_tab:

:begin_tab:`tensorflow`
In Keras the split is one global switch,
`tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')`. Every layer
constructed afterwards gets the policy pair fp32 `variable_dtype`, bf16
`compute_dtype`: master weights and 16-bit arithmetic, decided per layer
rather than per operation. Two cautions follow from "constructed
afterwards". The policy is consulted when a layer is built, so set it before
creating the model, and it is global state, so we reset it to `'float32'` at
the end of every cell that flips it, or it would silently reshape every
model built later in this notebook. What you compute outside the layers,
such as the loss, follows no policy; we cast logits to fp32 before the loss
for exactly the reason the autocast frameworks pin reductions there.
:end_tab:

:begin_tab:`mxnet`
Gluon predates per-operation autocasting, and its recipe is the original
low-ceremony one from the first mixed-precision papers, with the two halves
assigned to different objects: cast the *network* to fp16, and ask the
*optimizer* to keep the fp32 master weights. The second half is one flag,
`multi_precision=True` among the optimizer parameters. With it, the updater
allocates an fp32 copy of every fp16 weight, accumulates each update into
that copy in full precision, and writes the rounded fp16 version back to the
network; without it, the optimizer warns that accumulating in fp16 risks
poor accuracy or slow convergence. (An `mxnet.amp` module with per-operation
lists in the PyTorch style also exists, for fp16 and bf16 targets, but this
book does not exercise it.)
:end_tab:

:numref:`fig_bg_amp-loop` draws the resulting
loop: this is the distinction that matters between casting a whole model
(everything in one dtype) and mixed precision (fp32 master
weights that a 16-bit forward and backward pass read from and write back to).

![The mixed-precision training loop: fp32 master weights are cast to bf16 for the forward pass and its bf16 activations, the loss accumulates back in fp32, the backward pass produces bf16 gradients, and the optimizer step reads those gradients but updates the fp32 master copy, closing the cycle; the fp16 variant additionally scales the loss up before backward and unscales the gradients back down before the step.](../img/bg-amp-loop.svg)
:label:`fig_bg_amp-loop`

```{.python .input #numerics-mixed-precision-training-1}
%%tab pytorch
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))
with torch.autocast('cpu', dtype=torch.bfloat16):
    Y = net(X)
Y.dtype, net[1].weight.dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab jax
net = nn.Sequential([lambda x: x.reshape((x.shape[0], -1)),
                     nn.Dense(256, dtype=jnp.bfloat16), nn.relu,
                     nn.Dense(10, dtype=jnp.bfloat16)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).dtype, jax.tree.leaves(params)[0].dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab tensorflow
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
net = tf.keras.Sequential([tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(10)])
Y = net(X)
tf.keras.mixed_precision.set_global_policy('float32')
Y.dtype, net.layers[1].kernel.dtype
```

```{.python .input #numerics-mixed-precision-training-1}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
net.initialize()
net.cast('float16')
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.1, 'multi_precision': True})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
with autograd.record():
    l = loss(net(X.astype('float16')), np.zeros(8)).mean()
l.backward()
trainer.step(1)
master = trainer._updaters[0].states[0][0]  # the fp32 master copy
net[0].weight.data().dtype, master.dtype
```

:begin_tab:`pytorch`
Compute in bf16, storage in fp32: exactly the opposite split from
`net.bfloat16()`. The first argument names the device type the computation
runs on; on a GPU you would write `torch.autocast('cuda', ...)` and nothing
else changes.
:end_tab:

:begin_tab:`jax`
Compute in bf16, storage in fp32: exactly the opposite split from casting the
parameter tree, and the fp32 input no longer drags anything up because
`dtype` pins the compute format. Note what the cell did not need: no context
manager, no device argument. The same constructor arguments mean the same
thing on CPU, GPU, and TPU.
:end_tab:

:begin_tab:`tensorflow`
Compute in bf16, storage in fp32: exactly the opposite split from
`dtype='bfloat16'`, obtained from one line before construction and nothing
per layer. There is no device argument; the same policy means the same
thing on CPU, GPU, and TPU.
:end_tab:

:begin_tab:`mxnet`
Storage in fp16, master copy in fp32: the same split as the other frameworks
with the two halves swapped. Where autocast keeps fp32 parameters and
downcasts on the fly, Gluon keeps fp16 parameters in the network, so the
forward pass needs no machinery at all, and hides the fp32 master weights in
the updater's state; the demonstration has to take one training step first
(the master copy is allocated lazily) and then reaches into that internal
state to show it. The practical consequence is identical: forward and
backward run in 16 bits, the update accumulates in fp32.
:end_tab:

Let us verify the claim that matters, that accuracy survives.
We train the same MLP on Fashion-MNIST twice, once in fp32 and once with
16-bit compute, from the same initialization and on the same batches:

```{.python .input #numerics-mixed-precision-training-2}
%%tab pytorch
data = d2l.FashionMNIST(batch_size=256)
loader = torch.utils.data.DataLoader(data.train, batch_size=256)
batches = [b for b, _ in zip(loader, range(100))]
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab jax
data = d2l.FashionMNIST(batch_size=256)
images, labels = data.train
batches = [(jnp.asarray(images[k:k+256], jnp.float32) / 255,
            jnp.asarray(labels[k:k+256], jnp.int32))
           for k in range(0, 100 * 256, 256)]
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab tensorflow
data = d2l.FashionMNIST(batch_size=256)
images, labels = data.train
batches = [(tf.constant(images[k:k+256, :, :, None], tf.float32) / 255,
            tf.constant(labels[k:k+256], tf.int32))
           for k in range(0, 100 * 256, 256)]
```

```{.python .input #numerics-mixed-precision-training-2}
%%tab mxnet
data = d2l.FashionMNIST(batch_size=256)
loader = gluon.data.DataLoader(data.train, batch_size=256)
batches = [b for b, _ in zip(loader, range(100))]
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab pytorch
def train(amp):
    torch.manual_seed(0)  # identical init and data order for both runs
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 10))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    losses = []
    for X, y in batches:
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            loss = F.cross_entropy(net(X), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab jax
def train(mixed):
    dtype = jnp.bfloat16 if mixed else jnp.float32
    net = nn.Sequential([lambda x: x.reshape((x.shape[0], -1)),
                         nn.Dense(256, dtype=dtype), nn.relu,
                         nn.Dense(10, dtype=dtype)])
    # param_dtype stays fp32, so the same key gives identical init either way
    params = net.init(jax.random.PRNGKey(0), batches[0][0])
    opt = optax.sgd(learning_rate=0.1)
    state = opt.init(params)
    def loss_fn(params, X, y):
        logits = net.apply(params, X).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, y).mean()
    @jax.jit
    def step(params, state, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, loss
    losses = []
    for X, y in batches:
        params, state, loss = step(params, state, X, y)
        losses.append(float(loss))
    return losses
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab tensorflow
def train(mixed):
    tf.keras.utils.set_random_seed(0)  # identical init for both runs
    tf.keras.mixed_precision.set_global_policy(
        'mixed_bfloat16' if mixed else 'float32')
    net = tf.keras.Sequential([tf.keras.layers.Flatten(),
                               tf.keras.layers.Dense(256, activation='relu'),
                               tf.keras.layers.Dense(10)])
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    losses = []
    for X, y in batches:
        with tf.GradientTape() as tape:
            logits = tf.cast(net(X), tf.float32)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True))
        grads = tape.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        losses.append(float(loss))
    tf.keras.mixed_precision.set_global_policy('float32')
    return losses
```

```{.python .input #numerics-mixed-precision-training-3}
%%tab mxnet
def train(mixed):
    npx.random.seed(0)
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
    net.initialize()
    net(batches[0][0])  # materialize the weights, identical for both runs
    if mixed:
        net.cast('float16')
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.1,
                             'multi_precision': mixed})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    losses = []
    for X, y in batches:
        if mixed:
            X = X.astype('float16')
        with autograd.record():
            l = loss(net(X), y).mean()
        l.backward()
        trainer.step(1)
        losses.append(float(l))
    return losses
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab pytorch
loss32, loss16 = train(amp=False), train(amp=True)
print(f'final loss: fp32 {loss32[-1]:.3f}, bf16 autocast {loss16[-1]:.3f}')
d2l.plot(list(range(1, 101)), [loss32, loss16], 'step', 'loss',
         legend=['fp32', 'bf16 autocast'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab jax
loss32, loss16 = train(mixed=False), train(mixed=True)
print(f'final loss: fp32 {loss32[-1]:.3f}, bf16 compute {loss16[-1]:.3f}')
d2l.plot(list(range(1, 101)), [loss32, loss16], 'step', 'loss',
         legend=['fp32', 'bf16 compute'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab tensorflow
loss32, loss16 = train(mixed=False), train(mixed=True)
print(f'final loss: fp32 {loss32[-1]:.3f}, mixed_bfloat16 {loss16[-1]:.3f}')
d2l.plot(list(range(1, 101)), [loss32, loss16], 'step', 'loss',
         legend=['fp32', 'mixed_bfloat16'])
```

```{.python .input #numerics-mixed-precision-training-4}
%%tab mxnet
loss32, loss16 = train(mixed=False), train(mixed=True)
print(f'final loss: fp32 {loss32[-1]:.3f}, '
      f'fp16 multi_precision {loss16[-1]:.3f}')
d2l.plot(list(range(1, 101)), [loss32, loss16], 'step', 'loss',
         legend=['fp32', 'fp16 multi_precision'])
```

The two curves lie on top of each other: 16-bit rounding perturbs each step
slightly, so the trajectories are not bitwise identical, but they descend at
the same rate to the same place. On a CPU that is all this buys; the
wall-clock payoff appears on GPUs, where 16-bit matrix multiplications run on
tensor cores at a multiple of fp32 throughput and activations occupy half the
memory, typically a 2 to 3 times end-to-end speedup for models dominated by
matmuls (we turn to GPUs in :numref:`sec_use_gpu_v2`). Note what mixed
precision does *not* buy: the master weights and any Adam state remain fp32,
so the parameter and optimizer terms in the memory arithmetic of
:numref:`sec_parameters_v2` do not shrink. The savings are in activations
and speed.

:begin_tab:`mxnet`
Two footnotes for this tab. The 16-bit format here is fp16, not bf16, so the
agreement of the curves also leans on the next subsection: with plain SGD on
a well-scaled loss the gradients stay inside fp16's range, which is why no
loss scaling was needed. And CPUs have no fast fp16 path, so the mixed run
above is *slower* than fp32 on this machine; as for the other tabs, the
speed argument is a GPU argument.
:end_tab:

### Loss Scaling for fp16

With bf16, the recipe above is complete. fp16 has one more failure mode, and
it is the opposite end of the axis from the overflow that opened this section:
*gradient underflow*. Many gradients are small, fp16's range gives out below
$6 \times 10^{-5}$, and what fp32 happily represents, fp16 flushes to zero:

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab pytorch
g = torch.tensor(1e-8)
g.half(), (g * 2**16).half()
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab jax
g = jnp.array(1e-8)
g.astype(jnp.float16), (g * 2**16).astype(jnp.float16)
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab tensorflow
g = tf.constant(1e-8)
tf.cast(g, tf.float16), tf.cast(g * 2**16, tf.float16)
```

```{.python .input #numerics-loss-scaling-for-fp16-1}
%%tab mxnet
g = np.array(1e-8)
g.astype('float16'), (g * 2**16).astype('float16')
```

The gradient vanishes, yet the same value scaled by $2^{16}$ is perfectly
representable. That is the idea of *loss scaling*: multiply the loss by a
large constant before backpropagation, so that by linearity every gradient is
scaled into representable territory, then divide the gradients by the same
constant before the optimizer step.

:begin_tab:`pytorch`
`torch.amp.GradScaler` automates it,
choosing the scale dynamically: start high, shrink when scaled gradients
overflow to `inf` (skipping that step), grow back periodically.
:end_tab:

:begin_tab:`jax`
optax ships no automatic scaler, and JAX code rarely misses it. The idiom
grew up on TPUs where bf16 is the native 16-bit format, and bf16 needs no
scaling: its exponent range matches fp32, so any gradient fp32 can represent,
bf16 can too. The recipe of this section, `dtype=jnp.bfloat16` over fp32
parameters, is therefore already complete, and we deliberately skip fp16 loss
scaling. If old hardware ever forces fp16 on you, the two multiplications are
yours to write: scale the loss inside `loss_fn`, divide the gradients before
`optax.apply_updates`.
:end_tab:

:begin_tab:`tensorflow`
In Keras loss scaling is the `'mixed_float16'` policy plus an optimizer
wrapper. Under that policy, `model.compile` wraps whatever optimizer you
pass in `tf.keras.mixed_precision.LossScaleOptimizer`, which chooses the
scale dynamically: start high, shrink when scaled gradients overflow to
`inf` (skipping that step), grow back periodically. `model.fit` users
therefore get correct fp16 training without touching anything; a custom
loop applies the wrapper itself, multiplies with its `scale_loss` before
taking gradients, and lets the wrapped optimizer unscale and skip bad steps.
None of this machinery exists for `'mixed_bfloat16'` because none is needed:
bf16's exponent range matches fp32, so any gradient fp32 can represent, bf16
can too. Hence the modern default our training cell followed: prefer
`'mixed_bfloat16'` where the hardware supports it, and reach for
`'mixed_float16'` plus the scaler only on older accelerators.
:end_tab:

:begin_tab:`mxnet`
Gluon's `Trainer` has no scaler; a static loss scale is two edits you own.
Multiply: `(2**10 * l).backward()` instead of `l.backward()`. Divide:
`trainer.step(2**10)` instead of `trainer.step(1)`, because the step's
argument rescales gradients by its inverse before the update. The scale is
yours to choose, too low and small gradients still flush to zero, too high
and large ones overflow; the dynamic grow-and-shrink version the other
frameworks automate lives in the `mxnet.amp` module mentioned earlier, which
is where to look if the manual recipe stops being enough.
:end_tab:

```{.python .input #numerics-loss-scaling-for-fp16-2}
%%tab pytorch
torch.manual_seed(0)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))
opt = torch.optim.SGD(net.parameters(), lr=0.1)
scaler = torch.amp.GradScaler('cpu')
for X, y in batches[:20]:
    with torch.autocast('cpu', dtype=torch.float16):
        loss = F.cross_entropy(net(X), y)
    scaler.scale(loss).backward()  # gradients of (scale * loss)
    scaler.step(opt)               # unscale, check for inf/nan, then step
    scaler.update()
    opt.zero_grad()
print(f'loss {loss.item():.3f}, loss scale {scaler.get_scale():.0f}')
```

:begin_tab:`pytorch`
Keep the two failure modes straight: `GradScaler` exists to prevent gradient
*underflow*; its `inf`/NaN check handles overflow as a side effect by skipping
the bad step. bf16 needs no scaler at all, because its exponent range matches
fp32: any gradient fp32 can represent, bf16 can too. Hence the modern default:
on hardware with bf16 support (Ampere and later GPUs, TPUs, recent CPUs), use
bf16 autocast and stop there; reach for fp16 plus `GradScaler` only on older
accelerators.
:end_tab:

## When Numerics Bite

A short field guide for the day training misbehaves.

**The loss is NaN.** NaN is usually a symptom, not the disease; the disease is
`inf`, because `inf` arithmetic breeds NaN:

```{.python .input #numerics-when-numerics-bite-1}
%%tab pytorch
s = torch.tensor(60000., dtype=torch.float16) * 2  # overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab jax
s = jnp.array(60000., dtype=jnp.float16) * 2  # overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab tensorflow
s = tf.constant(60000., dtype=tf.float16) * 2  # overflows
s, s - s
```

```{.python .input #numerics-when-numerics-bite-1}
%%tab mxnet
s = np.array(60000., dtype='float16') * 2  # overflows
s, s - s
```

By the time a NaN reaches your loss, the overflow that spawned it may be many
operations upstream, and once a NaN lands in the weights it poisons every
subsequent step. So diagnose in order: first check ranges (is anything in
fp16? are intermediate values in the $10^4$ regime and headed for the 65504
ceiling?), and only then blame the learning rate.

:begin_tab:`pytorch`
Under autocast with
`GradScaler`, the gradient check catches the `inf` in the very step it occurs
and skips the update, one more reason to prefer the standard recipe over
hand-rolled fp16 even while debugging.
:end_tab:

:begin_tab:`jax`
JAX can localize the culprit for you: set
`jax.config.update('jax_debug_nans', True)` and execution stops with an error
at the first operation whose output is NaN, instead of letting it wash
downstream into the loss. The check reruns jitted code operation by
operation when it trips, so treat it as a debugging mode, not a default.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow can localize the culprit for you: call
`tf.debugging.enable_check_numerics()` and execution stops with an
`InvalidArgumentError` at the first operation whose output contains `inf` or
NaN, instead of letting it wash downstream into the loss. The check wraps
every operation, so treat it as a debugging mode, not a default.
:end_tab:

:begin_tab:`mxnet`
MXNet has no switch that stops at the first bad operation, so the search is
manual: bisect with `np.isnan(x).any()` and `np.isinf(x).any()` probes after
suspect stages, starting from the loss and walking upstream. The
diagnose-in-order advice above matters all the more when the tooling will
not do the walking for you.
:end_tab:

**Let the library take the log.** The naive evaluation of
$\log \sum_i \exp(x_i)$ overflows long before the answer does:

```{.python .input #numerics-when-numerics-bite-2}
%%tab pytorch
x = torch.tensor([12.0, 11.0, 10.0], dtype=torch.float16)
x.exp().sum().log(), torch.logsumexp(x, dim=0)
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab jax
x = jnp.array([12.0, 11.0, 10.0], dtype=jnp.float16)
jnp.log(jnp.exp(x).sum()), jax.scipy.special.logsumexp(x)
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab tensorflow
x = tf.constant([12.0, 11.0, 10.0], dtype=tf.float16)
tf.math.log(tf.reduce_sum(tf.exp(x))), tf.math.reduce_logsumexp(x)
```

```{.python .input #numerics-when-numerics-bite-2}
%%tab mxnet
x = np.array([12.0, 11.0, 10.0], dtype='float16')
m = x.max()
np.log(np.exp(x).sum()), m + np.log(np.exp(x - m).sum())
```

The answer, 12.4, is unremarkable; only the intermediate $e^{12}$ exceeds
65504. The subtract-the-max shift from :numref:`sec_numerical_stability` fixes
it, and the practical form of the lesson is to never hand-roll the pattern.

:begin_tab:`pytorch`
`torch.logsumexp`, `F.log_softmax`, and `F.cross_entropy` all build the shift
in.
:end_tab:

:begin_tab:`jax`
`jax.scipy.special.logsumexp`, `jax.nn.log_softmax`, and optax's
cross-entropy losses all build the shift in.
:end_tab:

:begin_tab:`tensorflow`
`tf.math.reduce_logsumexp`, `tf.nn.log_softmax`, and Keras's cross-entropy
losses with `from_logits=True` all build the shift in.
:end_tab:

:begin_tab:`mxnet`
`mx.np` ships no `logsumexp`, so this is the one tab where the cell writes
the shift out: subtract the maximum before exponentiating, add it back
outside the log. Three tokens of algebra, worth knowing cold, and still not
something to hand-roll inside a model: `npx.log_softmax` and Gluon's
`SoftmaxCrossEntropyLoss` build the same shift in.
:end_tab:

**Accumulate in fp32.** Long sums in a 16-bit dtype drift, because once the
running total is large, each small increment falls below the spacing of
representable values and is partly or wholly rounded away:

```{.python .input #numerics-when-numerics-bite-3}
%%tab pytorch
x = torch.full((10**7,), 1e-3, dtype=torch.float16)
x.cumsum(0)[-1], x.cumsum(0, dtype=torch.float32)[-1]
```

```{.python .input #numerics-when-numerics-bite-3}
%%tab jax
x = jnp.full((10**7,), 1e-3, dtype=jnp.float16)
x.cumsum()[-1], x.astype(jnp.float32).cumsum()[-1]
```

```{.python .input #numerics-when-numerics-bite-3}
%%tab tensorflow
x = tf.fill([10**7], tf.constant(1e-3, tf.float16))
(tf.cumsum(x)[-1], tf.cumsum(tf.cast(x, tf.float32))[-1],
 tf.cumsum(tf.cast(x, tf.float64))[-1])
```

```{.python .input #numerics-when-numerics-bite-3}
%%tab mxnet
x = np.full((10**7,), 1e-3, dtype='float16')
(x.cumsum()[-1], x.cumsum(dtype='float32')[-1],
 x.cumsum(dtype='float64')[-1])
```

:begin_tab:`pytorch`
The fp16 running total is short by about 2 percent; keeping the *accumulator*
in fp32 while the data stays fp16 recovers the exact answer. Means over large
batches, epoch-level loss totals, and variance computations all follow this
pattern, and it is precisely why autocast pins reductions and normalizations
to fp32. When you write your own, pass `dtype=torch.float32` to the reduction.
:end_tab:

:begin_tab:`jax`
The two totals disagree in the fourth digit. The fp32 accumulation, 10004, is
the exact sum of the stored values (0.001 itself rounds to the nearest fp16,
which is why the answer is not 10000); the fp16 accumulation drifts above it.
The drift is milder than a naive sequential loop would produce, because XLA
evaluates the scan as a tree, which keeps the partial sums small, but it is
drift all the same and it grows with the length of the sum. Means over large
batches, epoch-level loss totals, and variance computations all follow this
pattern, and it is why our training loop cast logits to fp32 before the loss.
When you write your own reduction over 16-bit data, upcast first, as the
second expression does.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow's strictly sequential scan makes the failure spectacular. The
fp16 running total stalls at 4.0, four hundredths of a percent of the true
answer: at 4, fp16's spacing is about 0.004, so each 0.001 increment rounds
to no change and the remaining terms, essentially all ten million of them,
contribute nothing. The fp32 accumulation reaches 9780, still 2 percent
short, by the same mechanism in milder form; near $10^4$ fp32's own spacing
is about 0.001, the size of one increment. Only the fp64 accumulator returns
10004.04, the exact sum of the stored values (0.001 itself rounds to the
nearest fp16, which is why the answer is not 10000). Means over large
batches, epoch-level loss totals, and variance computations all follow this
pattern; when you write your own, pick an accumulator wide relative to the
*length* of the sum. Order helps too: `tf.reduce_sum` over the same fp32
array returns the exact 10004.04, because it sums in blocks and keeps every
partial sum small, which is one more reason to hand long reductions to the
library instead of scanning.
:end_tab:

:begin_tab:`mxnet`
`cumsum`'s `dtype` argument sets the accumulator while the data stays fp16.
The fp16 total stalls early: once the running sum reaches a few units, fp16's
spacing exceeds 0.001 and every further increment rounds to no change, so
essentially all ten million remaining terms contribute nothing. The fp32
accumulator does far better but is not immune, since near $10^4$ fp32's own
spacing is about 0.001, the size of one increment; the fp64 accumulator
returns 10004.04, the exact sum of the stored values (0.001 itself rounds to
the nearest fp16, which is why the answer is not 10000). Means over large
batches, epoch-level loss totals, and variance computations all follow this
pattern; when you write your own reduction over 16-bit data, hand it a wide
accumulator, as the second and third expressions do.
:end_tab:

:begin_tab:`pytorch`
Two smaller traps, for completeness. `scaler.unscale_(opt)` may be called at
most once per step, so if you unscale to clip gradients, do not unscale again
to log them, or `scaler.update()` will raise. And under gradient accumulation,
call `scaler.update()` once per *effective* batch, after the last micro-step,
not once per micro-step.
:end_tab:

:begin_tab:`jax`
The bf16-first recipe has no loss scale to manage, so there is no scaler
bookkeeping to get wrong; gradient clipping and accumulation compose as plain
optax transformations.
:end_tab:

:begin_tab:`tensorflow`
The bf16-first recipe has no loss scale to manage. If older hardware pushes
you to `'mixed_float16'`, prefer `model.fit`, which owns the
`LossScaleOptimizer` bookkeeping end to end; in a custom loop the
scale-the-loss, unscale-the-gradients pairing is yours to keep straight.
:end_tab:

:begin_tab:`mxnet`
One smaller trap, for completeness: `multi_precision` belongs to the
*optimizer*, so it must ride in the optimizer parameters when the `Trainer`
is created. Casting the network alone trains with fp16 accumulation, and the
optimizer's warning about poor accuracy is one line in a noisy log, easy to
scroll past.
:end_tab:

Finally, do not expect bitwise equality across
numeric configurations: tf32 versus fp32, or mixed precision on versus off,
differ in the last bits by design. What reproducibility you can demand, and
how to get it, is the subject of :numref:`sec_repro_v2`.

## Summary

A dtype is a contract about range and precision, and the 16-bit formats split
the difference between them: fp16 keeps precision and forfeits range, bf16
keeps fp32's range and forfeits precision, which suits deep learning better.

:begin_tab:`pytorch`
Casting a model (`net.bfloat16()`) converts its parameters and is the tool
for inference; training instead uses autocast, which keeps fp32 master
weights and runs matrix multiplications in 16 bits under a per-operation
policy. fp16 training additionally needs `GradScaler` to stop small gradients
from underflowing to zero; bf16 does not.
:end_tab:

:begin_tab:`jax`
In flax the storage and compute formats are the two constructor arguments of
every layer: casting a model for inference means bf16 in both (or one
`tree.map` over the parameters), while mixed-precision training sets
`dtype=jnp.bfloat16` and leaves `param_dtype` at fp32, master weights and
16-bit matrix multiplications with no context manager in sight. Train in
bf16; fp16 loss scaling is machinery for older hardware that JAX practice
skips.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow's raw ops never promote; a dtype mismatch raises at the first
operation, and Keras layers resolve this by casting inputs to their dtype
policy, the pair of storage and compute formats fixed at construction.
`dtype='bfloat16'` sets both and casts a model for inference;
`set_global_policy('mixed_bfloat16')` keeps fp32 master weights over bf16
compute, mixed-precision training as one line of configuration (reset the
policy when you are done, it is global). fp16 training additionally needs
the loss scaling that `LossScaleOptimizer` automates under
`'mixed_float16'`; bf16 does not.
:end_tab:

:begin_tab:`mxnet`
Casting a model (`net.cast('float16')`) converts its parameters recursively
and is the tool for inference. Training keeps the cast network but adds
`multi_precision=True` to the optimizer parameters, which maintains fp32
master weights inside the updater: 16-bit forward and backward,
full-precision accumulation. The rest of the modern menu is not on offer in
this archived framework: no bf16 array dtype, no tf32 switch, no fp8, and
loss scaling is either two manual lines or the unexercised `mxnet.amp`
module.
:end_tab:

When numbers misbehave: check for
overflow before blaming the learning rate, use the library's `logsumexp`
family, and accumulate long sums in fp32.

## Exercises

1. Redo the memory arithmetic of :numref:`sec_parameters_v2` for
   mixed-precision training with Adam: fp32 master weights, fp32 gradients,
   two fp32 moment estimates, and bf16 activations. Which term dominates now,
   and how does the total compare with all-fp32 training?
1. Time the fp32 and 16-bit runs of `train` against each other while
   shrinking the hidden width and the batch size. Find a model small enough
   that mixed precision is *slower* than fp32, and explain where the
   crossover comes from.
1. Print every field of `torch.finfo(torch.float8_e4m3fn)` (in JAX,
   `jnp.finfo(jnp.float8_e4m3fn)`; in TensorFlow,
   `ml_dtypes.finfo(ml_dtypes.float8_e4m3fn)`; MXNet has no fp8 dtype, so
   borrow the standalone `ml_dtypes` package) and compare with fp16 and
   bf16. Explain
   the name `e4m3fn`, including why its `max` is 448 rather than the value
   the exponent bits alone would suggest.
1. Under autocast, normalization layers run in fp32. To see why, take the
   `RMSNorm` layer of :numref:`sec_custom_layers_v2`, feed it inputs with a
   standard deviation of about 100, and force the computation to fp16. Which
   intermediate quantity fails first?
