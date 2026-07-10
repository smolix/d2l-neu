```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Custom Layers and Functions
:label:`sec_custom_layers_v2`

The library ships over a hundred layers, yet every one of them started life as
code somebody wrote because the layer they needed did not exist. Sooner or
later you will be in the same position: a new normalization, an unusual
residual block, an operation with no gradient. This section shows what to do.
A custom layer is a subclass of the module class from
:numref:`sec_model_construction_v2` with a `forward` method, and if you
register its state properly you inherit everything a built-in layer gets:
parameter tracking, serialization, and device movement, with no extra code. We
build up from a stateless five-liner to RMSNorm, the normalization inside most
current language models, then to layers with precomputed non-trainable state,
and finally to the case where `forward` alone is not enough because the
gradient itself must be redefined.

```{.python .input #custom-layers-custom-layers-and-functions}
%%tab pytorch
import torch
from torch import nn
```

## Layers without Parameters

The smallest custom layer has no state at all. `CenteredLayer` subtracts the
mean from its input. To build it, we inherit from the base module class and
implement `forward`; there is nothing to set up, so `__init__` only calls the
parent constructor.

```{.python .input #custom-layers-layers-without-parameters-1}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Feeding data through confirms that it does what it says.

```{.python .input #custom-layers-layers-without-parameters-2}
%%tab pytorch
layer = CenteredLayer()
layer(torch.tensor([1.0, 2, 3, 4, 5]))
```

Nothing distinguishes this class from a built-in layer. We can place it inside
a `Sequential`, and the container neither knows nor cares that one of its
children is user code. The output mean should be zero; because we are adding
up floating-point numbers, we see a very small nonzero value instead, which is
roundoff, not a bug.

```{.python .input #custom-layers-layers-without-parameters-3}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
```

## Layers with Parameters: RMSNorm

A layer with something to learn must create its own parameters, and wrapping a
tensor in `nn.Parameter` is what registers them (:numref:`sec_parameters_v2`).
We could show the mechanics by re-implementing `nn.Linear`, but that teaches
nothing the built-in does not already do. Instead we implement *RMSNorm*
:cite:`Zhang.Sennrich.2019`, the normalization used by most current large
language models, in the same five lines.

Layer normalization standardizes each input vector: subtract the mean, divide
by the standard deviation, then apply a learned scale and shift. Zhang and
Sennrich observed that the re-centering contributes little and dropped it,
along with the shift. What remains is: divide by the root mean square,
multiply by a learned gain $\mathbf{g}$:

$$
\textrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \mathbf{g},
$$

where $d$ is the width of $\mathbf{x}$ and $\epsilon$ guards against division
by zero. One parameter vector, one reduction, no shift. Why dropping the mean
costs so little in practice is a question for the normalization discussion in
later chapters; here it is our stateful worked example.

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-1}
%%tab pytorch
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, X):
        rms = X.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.gain * X * rms
```

Feeding it badly scaled data confirms the normalization: every output row has
unit mean square, whatever the input scale was.

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-2}
%%tab pytorch
norm = RMSNorm(8)
X = 100 * torch.randn(4, 8)
norm(X).pow(2).mean(-1)
```

### The Composability Guarantee

The reason to write RMSNorm as a module, rather than as a function with a
gain tensor floating around beside it, is what registration buys. A correctly
written custom layer is indistinguishable from a built-in one along four
axes: its parameters are tracked, it composes inside containers, its state
serializes, and it moves across devices. We check each once.

First, the gain registered itself the moment we assigned an `nn.Parameter` in
`__init__`. Any optimizer handed `norm.parameters()` will find and update it.

```{.python .input #custom-layers-the-composability-guarantee-1}
%%tab pytorch
list(norm.named_parameters())
```

Second, the layer drops into a `Sequential` next to built-ins.

```{.python .input #custom-layers-the-composability-guarantee-2}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), RMSNorm(8), nn.LazyLinear(2))
net(torch.randn(4, 20)).shape
```

Third, its state serializes with everything else. A state dict written by one
instance loads into a fresh one, after which the two agree exactly (saving to
disk works the same way; see :numref:`sec_read_write_v2`).

```{.python .input #custom-layers-the-composability-guarantee-3}
%%tab pytorch
clone = nn.Sequential(nn.LazyLinear(8), RMSNorm(8), nn.LazyLinear(2))
clone.load_state_dict(net.state_dict())
X = torch.randn(4, 20)
torch.equal(net(X), clone(X))
```

Fourth, it moves. `.to(device)` walks the module tree and carries every
registered tensor along, the gain included. On a machine with a GPU the cell
below reports `cuda:0` twice; on a CPU-only machine it reports `cpu` and runs
just the same.

```{.python .input #custom-layers-the-composability-guarantee-4}
%%tab pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
next(net.parameters()).device, net(torch.randn(4, 20, device=device)).device
```

None of this took any code beyond the five-line class. The guarantee comes
from the base class: subclass it, register state through the proper channels,
and containers, optimizers, checkpoints, and devices all treat your layer as
native.

### Checking against the Built-in

RMSNorm proved useful enough that the library now ships its own
`nn.RMSNorm`. That gives us a referee. We copy a nontrivial gain into both
implementations, so that agreement cannot be an accident of default values,
and compare outputs.

```{.python .input #custom-layers-checking-against-the-built-in}
%%tab pytorch
ours, native = RMSNorm(8), nn.RMSNorm(8, eps=1e-6)
with torch.no_grad():
    ours.gain.copy_(torch.linspace(0.5, 1.5, 8))
    native.weight.copy_(ours.gain)
X = torch.randn(16, 8)
torch.allclose(ours(X), native(X))
```

They match to floating-point precision. This is the general rule for custom
layers: build one to understand it, then use the native implementation in
production. The native version may fuse the reduction and the scale into a
single kernel, and it will be maintained as the library evolves. Later
chapters follow the rule and use `nn.RMSNorm` directly.

## Precomputed State: Buffers

:numref:`sec_parameters_v2` introduced buffers as the third kind of module
state: tensors that persist and travel with the model but receive no
gradient. Custom layers are where you create them. A common case is a layer
built around a precomputed table, for instance the causal mask that keeps
attention scores from looking at future positions. The mask is fixed, so a
parameter is wrong (the optimizer would update it) and a plain attribute is
wrong too (`.to(device)` would skip it and the state dict would omit it).
`register_buffer` is the correct channel.

```{.python .input #custom-layers-precomputed-state-buffers-1}
%%tab pytorch
class CausalMask(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        # Precompute once for the longest sequence; slice per call
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        self.register_buffer('mask', mask)

    def forward(self, scores):
        T = scores.shape[-1]
        return scores.masked_fill(self.mask[:T, :T], float('-inf'))
```

The layer masks the strict upper triangle, has no parameters for an optimizer
to touch, and still lists the mask in its state dict.

```{.python .input #custom-layers-precomputed-state-buffers-2}
%%tab pytorch
mask = CausalMask(max_len=8)
print(mask(torch.zeros(3, 3)))
print(list(mask.parameters()), list(mask.state_dict()))
```

## Custom Gradients

So far we customized the forward computation and let automatic
differentiation derive the backward pass. That derivation is literal: it
differentiates exactly the operations you ran. Occasionally literalness is
the problem. Quantization rounds values to a grid, and rounding is flat
almost everywhere, so its true derivative is zero. Any parameter sitting
behind a rounding operation stops learning:

```{.python .input #custom-layers-custom-gradients-1}
%%tab pytorch
w = torch.tensor([0.9, 1.4, 2.6], requires_grad=True)
w.round().sum().backward()
w.grad
```

Zero gradient, exactly as calculus demands, and useless for training. The
*straight-through estimator* :cite:`Bengio.Leonard.Courville.2013` resolves
this with a controlled lie: keep the rounding in the forward pass, but
pretend in the backward pass that it was the identity. No automatic system
can derive a lie for us, so we override the chain rule with
`torch.autograd.Function`, supplying both directions ourselves as static
methods, the split :numref:`fig_bg_ste` draws explicitly.

![The straight-through estimator, forward versus backward. Forward keeps the true staircase round(x), close to but not the same as the identity it approximates; backward substitutes a constant surrogate gradient of 1, the identity's own derivative, for the true gradient, which is zero almost everywhere and would stop all learning.](../img/bg-ste.svg)
:label:`fig_bg_ste`

```{.python .input #custom-layers-custom-gradients-2}
%%tab pytorch
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Pretend forward was the identity: pass the gradient straight through
        return grad_output
```

Two rules govern the class. First, invoke it through `RoundSTE.apply(X)`,
never by calling `forward` directly: `apply` is what inserts the operation
into the autograd graph, while a direct call computes the same values with no
bookkeeping, and `backward` then never runs. This is the most common
first-time bug with custom functions, and it fails without an error message.
Second, `backward` receives the loss gradient with respect to the output and
must return the gradient with respect to each input. Ours ignores the input
values entirely; a backward that needs them must stash them in `forward` with
`ctx.save_for_backward` and retrieve them from `ctx`.

A small example verifies that the gradient now flows. We multiply the rounded
values by known weights, so we know what gradient to expect at `w`.

```{.python .input #custom-layers-custom-gradients-3}
%%tab pytorch
w.grad = None
y = (RoundSTE.apply(w) * torch.tensor([1.0, 2.0, 3.0])).sum()
y.backward()
w.grad
```

The downstream gradient arrives at `w` untouched, as if the rounding were the
identity, while the forward pass still computed with rounded values. One
design choice deserves a mention: instead of the plain passthrough, many
implementations return `grad_output.clamp(-1.0, 1.0)`, or zero the gradient
where the saved input had saturated. The clamped variants bound what flows
through the fiction and are the safer default when the quantized layer sits
deep in a large model.

A scope note to close. `torch.autograd.Function` redefines the gradient of a
computation assembled from existing tensor operations; it does not create new
ones. Writing new *kernels*, code that runs on the accelerator itself, is a
separate craft that we do not cover in this book; :numref:`chap_performance`
discusses how to get performance out of the operations you already have.

## Summary

A custom layer is a module subclass: `forward` defines the computation,
`nn.Parameter` registers learnable state, and `register_buffer` registers
persistent state that no optimizer should touch. Registration is what buys
composability; a correctly written layer gets parameter tracking, container
compatibility, serialization, and device movement for free, as we verified on
RMSNorm axis by axis. When the chain rule itself must be overridden, as in
the straight-through estimator, `torch.autograd.Function` lets you supply
`forward` and `backward` as a pair, invoked through `apply`. Build custom
implementations to understand them; prefer the native ones in production.

## Exercises

1. Add an optional learned bias to `RMSNorm` (a shift applied after the
   scale, restoring part of what LayerNorm had). Verify that the state dict
   of a model containing it grows by the expected entry, and that a state
   dict saved without the bias no longer loads with `strict=True`.
1. Implement `Dropout` from scratch as a custom layer that zeroes each entry
   with probability $p$ and rescales the survivors during training, but is
   the identity during evaluation. Where does the `training` flag your
   `forward` consults live, and how does calling `.eval()` on an enclosing
   `Sequential` reach your layer?
1. Implement a clamp with a custom gradient: an `autograd.Function` whose
   forward is `X.clamp(lo, hi)` and whose backward passes the gradient only
   where the input lay strictly inside the clamp range. Compare its gradients
   with those of the native `torch.clamp` for inputs inside, outside, and
   exactly on the boundary. Which convention does the native operation use at
   the boundary?
1. Write a layer that returns the leading half of the Fourier coefficients of
   its input. It has no parameters, so nothing registers. What do you still
   gain by wrapping the computation in a module instead of calling the
   transform inline wherever you need it?
