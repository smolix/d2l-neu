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

## The Dtype Zoo

Half-precision (`torch.float16`, "fp16") sounds like a free lunch: half the
bytes of fp32, and the format that accelerator hardware sped up first. Here
is the catch. The largest number fp16 can represent is 65504. Square a value
of 300, which is nothing exotic (an unnormalized logit, an intermediate in a
variance computation), and you have already left the representable range:

```{.python .input #numerics-the-dtype-zoo-1}
%%tab pytorch
x = torch.tensor(300.0)
x.to(torch.float16)**2, x.to(torch.bfloat16)**2
```

fp16 overflows to `inf`. The second format, `torch.bfloat16` ("bf16", brain
floating point), returns 90112: wrong in the fourth digit, since the exact
square is 90000, but finite. The two formats spend the same 16 bits
differently. fp16 uses 5 exponent bits and 10 mantissa bits: fine-grained
steps, tiny range. bf16 keeps fp32's full 8 exponent bits and pays with a
7-bit mantissa: fp32's range, coarse steps. The one-line demo above shows both
consequences at once.

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

`torch.finfo` reports what each bit budget buys. Three numbers matter: `max`,
the overflow threshold; `tiny`, the smallest normal value before underflow to
zero; and `eps`, the relative step size between adjacent representable values.

```{.python .input #numerics-the-dtype-zoo-2}
%%tab pytorch
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    fi = torch.finfo(dtype)
    print(f'{str(dtype):15} max {fi.max:10.3e}  tiny {fi.tiny:9.3e}'
          f'  eps {fi.eps:8.3e}')
```

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
slightly less precisely. One global switch controls it:

```{.python .input #numerics-tf32-what-happens-to-fp32-matrix-multiplication}
%%tab pytorch
print(torch.get_float32_matmul_precision())
torch.set_float32_matmul_precision('high')  # allow tf32 in fp32 matmuls
print(torch.get_float32_matmul_precision(),
      torch.backends.cuda.matmul.allow_tf32)
torch.set_float32_matmul_precision('highest')  # restore the default
```

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

### Below 16 Bits

Production inference pushes further down: int8 quantization is standard for
serving, and fp8 training (a 4-bit-exponent variant for the forward pass, a
5-bit-exponent variant for gradients, with per-tensor scaling) runs on
H100-class hardware :cite:`Micikevicius.Stosic.Burgess.ea.2022`. Both require
calibration machinery beyond a dtype argument, so for this book 16 bits is the
floor; the exercises let you inspect `torch.finfo(torch.float8_e4m3fn)`.

## Dtype Rules: Promotion, Parameters, and Casts

What happens when dtypes meet in one expression? For plain tensors, PyTorch
promotes to the type that can represent both:

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-1}
%%tab pytorch
x16 = torch.ones(3, dtype=torch.float16)
x32 = torch.ones(3, dtype=torch.float32)
(x16 + x32).dtype, (x16 + 1.0).dtype, (x16 + torch.arange(3)).dtype
```

Mixing two float tensors upcasts to the wider one, so an fp16 pipeline with a
stray fp32 tensor quietly becomes fp32 from that point on, doubling downstream
memory. Python scalars and integer tensors are *weak*: they adopt the float
tensor's dtype instead of dragging it up, which is why sprinkling literals
like `x + 1.0` into low-precision code is harmless. (Mixing fp16 with bf16
promotes to fp32, since neither contains the other.)

Module parameters play by a stricter rule: layers do not promote, they demand
a matching input dtype and raise otherwise. To change a model's dtype you cast
the whole module; `net.to(dtype)` (or the shorthand `net.bfloat16()`) converts
every parameter and buffer in place. The byte accounting of
:numref:`sec_parameters_v2` composes with dtype through `element_size()`:

```{.python .input #numerics-dtype-rules-promotion-parameters-and-casts-2}
%%tab pytorch
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))
def param_bytes(net):
    return sum(p.numel() * p.element_size() for p in net.parameters())
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

The parameter footprint halves, inference works, and the error message shows
the strictness: the fp32 input was refused rather than silently converted.

Casting the model like this is the right tool for *inference*: half the
memory, no gradients to worry about, and a rounding error in the forward pass
rarely changes an argmax. For *training* it is a trap. A single optimizer
update changes a weight by roughly $\eta \cdot g$, often a factor $10^{-4}$
or less of the weight's own magnitude, and adding an increment smaller than
about `eps` times the weight rounds to no change at all. With bf16's `eps` of
0.0078, small updates evaporate and learning stalls; in fp16 the small
gradients themselves flush to zero first. Hence the rule, and it resolves the
single most common confusion in practice:

**Cast the model for inference. For training, keep fp32 weights and use
autocast.**

## Mixed-Precision Training

Mixed-precision training :cite:`Micikevicius.Narang.Alben.ea.2018` splits the
work: parameters stay in fp32 (the *master weights*, so that small updates
still register), while the expensive operations of the forward and backward
pass run in a 16-bit dtype. You do not annotate anything per layer. Inside a
`torch.autocast` context, each operation consults a built-in policy: matrix
multiplications and convolutions, which dominate compute and map onto tensor
cores, run in the low dtype; operations that accumulate many terms or
exponentiate run in fp32. PyTorch maintains the per-operation lists, and
inputs are cast on the fly. :numref:`fig_bg_amp-loop` draws the resulting
loop: this is the distinction that matters between casting a whole model
(`net.bfloat16()`, everything in one dtype) and mixed precision (fp32 master
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

Compute in bf16, storage in fp32: exactly the opposite split from
`net.bfloat16()`. The first argument names the device type the computation
runs on; on a GPU you would write `torch.autocast('cuda', ...)` and nothing
else changes. Let us verify the claim that matters, that accuracy survives.
We train the same MLP on Fashion-MNIST twice, once in fp32 and once under
bf16 autocast, from the same initialization and on the same batches:

```{.python .input #numerics-mixed-precision-training-2}
%%tab pytorch
data = d2l.FashionMNIST(batch_size=256)
loader = torch.utils.data.DataLoader(data.train, batch_size=256)
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

```{.python .input #numerics-mixed-precision-training-4}
%%tab pytorch
loss32, loss16 = train(amp=False), train(amp=True)
print(f'final loss: fp32 {loss32[-1]:.3f}, bf16 autocast {loss16[-1]:.3f}')
d2l.plot(list(range(1, 101)), [loss32, loss16], 'step', 'loss',
         legend=['fp32', 'bf16 autocast'])
```

The two curves lie on top of each other: bf16 rounding perturbs each step
slightly, so the trajectories are not bitwise identical, but they descend at
the same rate to the same place. On a CPU that is all this buys; the
wall-clock payoff appears on GPUs, where bf16 matrix multiplications run on
tensor cores at a multiple of fp32 throughput and activations occupy half the
memory, typically a 2 to 3 times end-to-end speedup for models dominated by
matmuls (we turn to GPUs in :numref:`sec_use_gpu_v2`). Note what mixed
precision does *not* buy: the master weights and any Adam state remain fp32,
so the parameter and optimizer terms in the memory arithmetic of
:numref:`sec_parameters_v2` do not shrink. The savings are in activations
and speed.

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

The gradient vanishes, yet the same value scaled by $2^{16}$ is perfectly
representable. That is the idea of *loss scaling*: multiply the loss by a
large constant before backpropagation, so that by linearity every gradient is
scaled into representable territory, then divide the gradients by the same
constant before the optimizer step. `torch.amp.GradScaler` automates it,
choosing the scale dynamically: start high, shrink when scaled gradients
overflow to `inf` (skipping that step), grow back periodically.

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

Keep the two failure modes straight: `GradScaler` exists to prevent gradient
*underflow*; its `inf`/NaN check handles overflow as a side effect by skipping
the bad step. bf16 needs no scaler at all, because its exponent range matches
fp32: any gradient fp32 can represent, bf16 can too. Hence the modern default:
on hardware with bf16 support (Ampere and later GPUs, TPUs, recent CPUs), use
bf16 autocast and stop there; reach for fp16 plus `GradScaler` only on older
accelerators.

## When Numerics Bite

A short field guide for the day training misbehaves.

**The loss is NaN.** NaN is usually a symptom, not the disease; the disease is
`inf`, because `inf` arithmetic breeds NaN:

```{.python .input #numerics-when-numerics-bite-1}
%%tab pytorch
s = torch.tensor(60000., dtype=torch.float16) * 2  # overflows
s, s - s
```

By the time a NaN reaches your loss, the overflow that spawned it may be many
operations upstream, and once a NaN lands in the weights it poisons every
subsequent step. So diagnose in order: first check ranges (is anything in
fp16? are intermediate values in the $10^4$ regime and headed for the 65504
ceiling?), and only then blame the learning rate. Under autocast with
`GradScaler`, the gradient check catches the `inf` in the very step it occurs
and skips the update, one more reason to prefer the standard recipe over
hand-rolled fp16 even while debugging.

**Let the library take the log.** The naive evaluation of
$\log \sum_i \exp(x_i)$ overflows long before the answer does:

```{.python .input #numerics-when-numerics-bite-2}
%%tab pytorch
x = torch.tensor([12.0, 11.0, 10.0], dtype=torch.float16)
x.exp().sum().log(), torch.logsumexp(x, dim=0)
```

The answer, 12.4, is unremarkable; only the intermediate $e^{12}$ exceeds
65504. The subtract-the-max shift from :numref:`sec_numerical_stability` fixes
it, and the practical form of the lesson is to never hand-roll the pattern:
`torch.logsumexp`, `F.log_softmax`, and `F.cross_entropy` all build the shift
in.

**Accumulate in fp32.** Long sums in a 16-bit dtype drift, because once the
running total is large, each small increment falls below the spacing of
representable values and is partly or wholly rounded away:

```{.python .input #numerics-when-numerics-bite-3}
%%tab pytorch
x = torch.full((10**7,), 1e-3, dtype=torch.float16)
x.cumsum(0)[-1], x.cumsum(0, dtype=torch.float32)[-1]
```

The fp16 running total is short by about 2 percent; keeping the *accumulator*
in fp32 while the data stays fp16 recovers the exact answer. Means over large
batches, epoch-level loss totals, and variance computations all follow this
pattern, and it is precisely why autocast pins reductions and normalizations
to fp32. When you write your own, pass `dtype=torch.float32` to the reduction.

Two smaller traps, for completeness. `scaler.unscale_(opt)` may be called at
most once per step, so if you unscale to clip gradients, do not unscale again
to log them, or `scaler.update()` will raise. And under gradient accumulation,
call `scaler.update()` once per *effective* batch, after the last micro-step,
not once per micro-step. Finally, do not expect bitwise equality across
numeric configurations: tf32 versus fp32, or autocast on versus off, differ
in the last bits by design. What reproducibility you can demand, and how to
get it, is the subject of :numref:`sec_repro_v2`.

## Summary

A dtype is a contract about range and precision, and the 16-bit formats split
the difference between them: fp16 keeps precision and forfeits range, bf16
keeps fp32's range and forfeits precision, which suits deep learning better.
Casting a model (`net.bfloat16()`) converts its parameters and is the tool
for inference; training instead uses autocast, which keeps fp32 master
weights and runs matrix multiplications in 16 bits under a per-operation
policy. fp16 training additionally needs `GradScaler` to stop small gradients
from underflowing to zero; bf16 does not. When numbers misbehave: check for
overflow before blaming the learning rate, use the library's `logsumexp`
family, and accumulate long sums in fp32.

## Exercises

1. Redo the memory arithmetic of :numref:`sec_parameters_v2` for
   mixed-precision training with Adam: fp32 master weights, fp32 gradients,
   two fp32 moment estimates, and bf16 activations. Which term dominates now,
   and how does the total compare with all-fp32 training?
1. Time `train(amp=False)` against `train(amp=True)` while shrinking the
   hidden width and the batch size. Find a model small enough that autocast is
   *slower* than fp32, and explain where the crossover comes from.
1. Print every field of `torch.finfo(torch.float8_e4m3fn)` and compare with
   fp16 and bf16. Explain the name `e4m3fn`, including why its `max` is 448
   rather than the value the exponent bits alone would suggest.
1. Under autocast, normalization layers run in fp32. To see why, take the
   `RMSNorm` layer of :numref:`sec_custom_layers_v2`, feed it inputs with a
   standard deviation of about 100, and force the computation to fp16. Which
   intermediate quantity fails first?
