```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Initialization
:label:`sec_init_v2`

Every parameter in a network has a value before training starts: constructing
`nn.Linear(512, 256)` fills a $256 \times 512$ weight matrix with random
numbers. :numref:`sec_numerical_stability` explains why the scale of those
numbers decides whether a deep network trains at all, and derives the Xavier
:cite:`Glorot.Bengio.2010` and He :cite:`He.Zhang.Ren.ea.2015` schemes that
keep signal variance stable across layers. This section is the practical
companion: what the library does when you say nothing, how to impose a scheme
of your own on a whole model or a single block, what transformer-era code adds
on top of Xavier and He, and how to write an initializer that no menu provides.

```{.python .input #init-initialization}
%%tab pytorch
import math
import torch
from torch import nn
```

## Defaults and When to Override Them

You can usually ignore initialization because the default is sensible. PyTorch
initializes an `nn.Linear` the moment it is constructed: weight and bias are
both drawn from the uniform distribution $U(-b, b)$ with
$b = 1/\sqrt{n_\textrm{in}}$, where $n_\textrm{in}$ is the number of input
features. (The documentation describes the weight rule as a variant of He
initialization, `kaiming_uniform_` with `a=math.sqrt(5)`; the algebra collapses
to the bound above.) A uniform distribution on $[-b, b]$ has standard deviation
$b/\sqrt{3}$, so both claims are cheap to check against a fresh layer:

```{.python .input #init-defaults-and-when-to-override-them}
%%tab pytorch
torch.manual_seed(0)
layer = nn.Linear(512, 256)
bound = 1 / math.sqrt(layer.in_features)
w = layer.weight
print(f'range [{w.min():.4f}, {w.max():.4f}], predicted bound {bound:.4f}')
print(f'std {w.std():.4f}, predicted {bound / math.sqrt(3):.4f}')
```

Other libraries make different but equally fan-aware choices, with one legacy
exception:

| Library | Default for a dense layer's weight |
|:--|:--|
| PyTorch | $U(-1/\sqrt{n_\textrm{in}},\, 1/\sqrt{n_\textrm{in}})$, bias likewise |
| Flax (JAX) | LeCun normal: truncated normal, std $1/\sqrt{n_\textrm{in}}$; zero bias |
| Keras (TensorFlow) | Glorot uniform: $U(\pm\sqrt{6/(n_\textrm{in} + n_\textrm{out})})$; zero bias |
| MXNet Gluon | $U(-0.07, 0.07)$, not fan-aware, so override it; zero bias |

When should you override? Four situations recur: a deep network without
normalization layers, where variance compounds across depth (we demonstrate
this below); parameters you created by hand, since
`nn.Parameter(torch.empty(...))` contains whatever bytes the allocator
happened to return; reproducing a paper whose results depend on its
initialization recipe; and architecture-specific corrections such as the
residual scaling later in this section. Note that the defaults are computed
from fan-in at construction time; a lazy layer (:numref:`sec_lazy_init`) has
no fan-in until its first forward pass, so it runs the same rule at
materialization instead.

## Applying Initializers

To impose a scheme we do not touch layers one at a time. `net.apply(fn)` walks
the module tree of :numref:`sec_model_construction_v2` and calls `fn` on every
submodule, children first. The function inspects each module's type and
decides what, if anything, to do with it. The routines in `torch.nn.init`
follow PyTorch's convention that a trailing underscore means *in place*:
`nn.init.normal_(w)` overwrites `w` rather than returning a fresh tensor. In
place is what we want here, since the parameter must remain the same tensor
object the module registered; replacing it would leave `net.parameters()` and
any existing optimizer pointing at the old one.

```{.python .input #init-applying-initializers-1}
%%tab pytorch
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

def init_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

The `isinstance` dispatch is what makes the pattern compose: one function can
treat `nn.Linear`, `nn.Conv2d`, and `nn.Embedding` differently, and it
silently skips containers and activations. Because `apply` runs on any
subtree, we can also mix schemes per block. Below we give the first layer
Xavier initialization (:numref:`subsec_xavier`) and set the last to a
constant, a poor idea for training by the symmetry argument of
:numref:`sec_numerical_stability`, but it makes the mechanics visible:

```{.python .input #init-applying-initializers-2}
%%tab pytorch
def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
net[0].weight.data[0], net[2].weight.data
```

This ten-line pattern is the entire API. Later chapters wrap it into
one-liners, `init_cnn` for convolutional networks and `init_seq2seq` for
encoder-decoder models, but each is just a function like `init_normal` handed
to `apply`.

## Modern Schemes: Truncation, Depth, and Zeros

Xavier and He set the variance of a single layer. The schemes below, standard
in transformer codebases, adjust what happens in the distribution's tails,
across depth, and at a block's start.

### Truncated Normals

A Gaussian gets the variance right, but its tails are unbounded. That is
harmless for one draw and a near-certainty at scale: among the $10^8$ weights
of a BERT-sized model, dozens land beyond five standard deviations. A single
outsized weight can dominate a unit's output at initialization, and it wastes
dynamic range once the model is cast to low precision
(:numref:`sec_numerics_v2`). The BERT and ViT lineage therefore samples from a
normal distribution *truncated* at two standard deviations: the same scale,
with a hard bound on every entry. `nn.init.trunc_normal_` takes absolute
cutoffs `a` and `b` (defaulting to $\pm 2$, which means $\pm 2\sigma$ only
when `std=1`), so a clip at two standard deviations must state them
explicitly:

```{.python .input #init-truncated-normals}
%%tab pytorch
torch.manual_seed(0)
w = torch.empty(1000, 1000)
nn.init.normal_(w, std=0.02)
print(f'normal:    std {w.std():.4f}, max weight {w.abs().max():.4f}')
nn.init.trunc_normal_(w, std=0.02, a=-0.04, b=0.04)
print(f'truncated: std {w.std():.4f}, max weight {w.abs().max():.4f}')
```

A million plain-normal draws produce entries near $5\sigma = 0.1$; the
truncated version guarantees $|w| \leq 0.04$. Its printed standard deviation
dips slightly below the nominal 0.02 because truncation removes tail mass;
practice ignores the difference.

### Scaling Down Residual Branches

A residual network computes $\mathbf{x}_{k+1} = \mathbf{x}_k +
f_k(\mathbf{x}_k)$, so its output is the input plus the sum of $N$ block
contributions. If every block is initialized identically, each contribution
has variance proportional to that of the already-inflated stream it reads, and
the stream's variance compounds geometrically with depth. This is the
depth-wise cousin of the layer-wise explosion in
:numref:`sec_numerical_stability`. GPT-2's fix is to shrink only the *last*
linear layer of each residual branch, the output projection that writes into
the stream, by a factor of $1/\sqrt{N}$: with $N$ roughly independent
contributions each scaled down that way, the sum's variance stays $O(1)$
regardless of depth :cite:`Radford.Wu.Child.ea.2019`. The published recipe is
a normal with std $0.02/\sqrt{2N}$ on the residual projections (the factor is
$2N$ because each of the $N$ transformer layers writes to the stream twice,
once from attention and once from the MLP). Unlike Xavier or He, the base 0.02
is not derived from fan-in; it is an empirical constant that worked at GPT-2's
widths and stuck.

:numref:`fig_bg_residual-stream` draws the two regimes side by side.

![A residual stream with additive block contributions, unscaled (left) versus scaled by 1/sqrt(N) (right): each block contributes O(1) variance to the stream it joins, so left uncorrected the stream's variance compounds like N, drawn as a thickening, darkening line, while scaling each contribution tames it back to O(1), a stream of constant width and shade.](../img/bg-residual-stream.svg)
:label:`fig_bg_residual-stream`

### Starting a Block at Zero

Zero-initializing *every* weight is fatal: all units in a layer compute the
same output, receive the same gradient, and stay identical forever
(:numref:`sec_numerical_stability`). Zero-initializing just the *last* layer
of a residual block is a different and useful move. The branch then
contributes exactly nothing, each block is the identity map, and the network
starts as a shallow function whose depth switches on gradually during
training. Symmetry is not a problem because the branch's earlier layers keep
their random weights: the zeroed projection receives a nonzero gradient (its
input, the branch activation, is nonzero), and once it moves off zero,
gradient reaches the whole branch. Keep this scheme distinct from the previous
one: GPT-2 makes every residual projection *small but nonzero*, whereas
zero-init makes one layer *exactly zero* so the block starts as an exact
identity. The zero variant is the standard trick for the final batch-norm gain
in ResNet blocks and for policy heads in reinforcement learning.

### Watching the Variance Compound

Claims about variance at depth are cheap to test. We reuse a compact residual
block (it repeats the one from :numref:`sec_model_construction_v2`) and stack
it $N$ deep:

```{.python .input #init-watching-the-variance-compound-1}
%%tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(),
                                  nn.Linear(4 * d, d))

    def forward(self, X):
        return X + self.body(X)
```

Every linear layer gets He initialization, appropriate for the ReLU inside
the branch. Then a `named_parameters` loop, the same shape as nanoGPT's
initialization code, singles out each block's output projection and applies
one of three treatments: leave it alone, scale it by $1/\sqrt{N}$, or zero it.

```{.python .input #init-watching-the-variance-compound-2}
%%tab pytorch
def init_he(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        nn.init.zeros_(module.bias)

def output_std(num_blocks, tweak):
    torch.manual_seed(0)
    net = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
    net.apply(init_he)
    for name, param in net.named_parameters():
        if name.endswith('body.2.weight'):  # each block's output projection
            tweak(param, num_blocks)
    torch.manual_seed(1)
    with torch.no_grad():
        return net(torch.randn(256, 64)).std().item()
```

One forward pass on unit-variance inputs per depth and treatment:

```{.python .input #init-watching-the-variance-compound-3}
%%tab pytorch
tweaks = {'default': lambda p, n: None,
          'scaled': lambda p, n: p.data.mul_(n ** -0.5),
          'zero': lambda p, n: nn.init.zeros_(p)}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in tweaks))
for n in (2, 8, 32):
    stds = (output_std(n, tweak) for tweak in tweaks.values())
    print(f'{n:>3}' + ''.join(f'{s:>10.3g}' for s in stds))
```

The default column multiplies by a constant factor per block, exponential
growth that reaches tens of millions by $N=32$; training this stack would
diverge on the first step. The scaled column sits near a small constant at
every depth: each contribution's variance is cut by a factor of $N$, so the
total stays bounded no matter how deep the stack. The zero column reproduces the
input's standard deviation exactly, since every block is the identity.
Two lines of initialization code separate a network that cannot train from
one that starts stable at any depth.

## Custom Initializers

Occasionally the menu has nothing you need. An initializer is just a function
that mutates a parameter, so writing one is no harder than using one. Suppose,
to make the point vividly, we want weights distributed as

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

Draw uniformly from $U(-10, 10)$, then zero every entry of magnitude below 5:

```{.python .input #init-custom-initializers-1}
%%tab pytorch
def my_init(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -10, 10)
        with torch.no_grad():
            module.weight *= module.weight.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

The `torch.no_grad()` block is required: parameters are leaf tensors that
track gradients, and PyTorch rejects in-place arithmetic on them unless we
declare that the mutation is not part of any computation to differentiate.
(The `nn.init` routines run under their own `no_grad` internally, which is why
the first line needs none.) The same escape hatch handles one-off surgery,
such as offsetting a whole matrix or pinning a single entry:

```{.python .input #init-custom-initializers-2}
%%tab pytorch
with torch.no_grad():
    net[0].weight[:] += 1
    net[0].weight[0, 0] = 42
net[0].weight[0]
```

One caveat when building with lazy layers (:numref:`sec_lazy_init`): before
the first forward pass their parameters are placeholders with no shape, so
`apply`-based initializers and direct surgery alike must come *after* the dry
run that materializes them.

## Summary

PyTorch initializes parameters at construction with fan-aware defaults;
override them when depth, hand-made parameters, or a paper's recipe demands
it. The mechanism is one pattern: a function that dispatches on module type
and mutates parameters in place with the trailing-underscore `nn.init`
routines, walked over the model by `net.apply`. On top of Xavier and He,
transformer-era code truncates normal tails to bound outliers, shrinks each
residual output projection by $1/\sqrt{N}$ so the stream's variance stays
$O(1)$ at any depth, and zero-initializes a block's last layer to start it as
the identity. Anything the menu lacks is a few lines of tensor code under
`torch.no_grad()`.

## Exercises

1. Instrument the residual stack: record the standard deviation of the
   activation after every block (loop over `net` applying one block at a
   time, or use a forward hook, :numref:`sec_repro_v2`) for the default and
   scaled treatments at $N=32$, and plot it against depth. Which curve matches
   the geometric-growth prediction?
1. Zero-initialize *all* layers of every block instead of just the output
   projection. The forward pass still returns the input, but what can the
   network learn? Work out which parameters receive a nonzero gradient, and
   relate the answer to the symmetry-breaking argument of
   :numref:`sec_numerical_stability`.
1. Write an initializer that fills each parameter from a dictionary keyed by
   parameter name, using `net.named_parameters()`. You have re-invented part
   of `load_state_dict`, which :numref:`sec_read_write_v2` covers.
1. For a normal distribution truncated at $\pm 2\sigma$: what fraction of
   draws does the clip discard, and by how much does it shrink the standard
   deviation? Verify both numbers against the printed output of the truncation
   demo above.
