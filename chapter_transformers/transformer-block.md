# The Transformer Block
:label:`sec_transformer-block`

:numref:`chap_attention` built attention, a layer in which each position
reads from all the others. On its own it makes a weak model. Each output of an
attention layer is a weighted average of value vectors, which are linear
projections of the inputs. The softmax weights decide how much of each value
to take, but nothing in the layer computes a nonlinear feature of what a
single position holds. A language model needs that computation at every
position, and it needs it stacked dozens of layers deep. Depth brings a
problem of its own: a signal that passes through 32 layers can shrink,
explode, or lose the differences between tokens before it reaches the
output, and the gradient makes the same trip in reverse.

The *transformer block* answers both needs. Attention moves information
between positions, a position-wise feed-forward network transforms each
position on its own, and both add their outputs to a shared residual stream
that carries the signal through the stack. Three changes separate the 2017
block from current ones: where the normalization sits, what it computes, and
what form the feed-forward network takes. The first has consequences we can
see without training anything. A freshly built stack of 32 blocks either
keeps its tokens distinct or squeezes them onto a single point, depending on
where the normalization sits and, it turns out, on how large the weights
are. The other two we work through on small cases and by counting
parameters. A configurable `TransformerBlock` class then collects these
choices, and the rest of the chapter builds on it.


```{.python .input #transformer-block-the-transformer-block}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #transformer-block-the-transformer-block}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Components of a Transformer Block

A transformer block contains two sublayers connected by a residual stream.
:numref:`sec_what-attention-computes` introduced the *residual stream* view:
each token carries a running vector $\mathbf{x} \in \mathbb{R}^d$ from layer
to layer. Sublayers never overwrite this vector. They read from it,
compute, and *add* their result back. The two sublayers of the block divide
the work as follows.

- **Attention** is the only place where positions interact. Each token
  queries the others as in :numref:`sec_multihead-attention` and adds the
  retrieved mixture to its stream.
- The **feed-forward network** (FFN) acts on each position separately: the
  same small MLP is applied token by token, with no interaction between
  positions. It transforms each token's accumulated state, and most of the
  block's parameters sit in it.

Writing $\mathrm{Attn}$ for multi-head self-attention and $\mathrm{Norm}$
for a normalization layer, the modern (*pre-norm*) block computes

$$
\mathbf{h} = \mathbf{x} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{x})), \qquad
\mathbf{y} = \mathbf{h} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{h})).
$$
:eqlabel:`eq_preln-block`

Each sublayer reads a normalized copy of the stream and adds its output to
the stream itself. Both sublayers preserve the shape $(n, d)$ of a sequence
of $n$ tokens, so dozens of copies of the block can be stacked. The block
also knows nothing about the order of the tokens: attention is permutation
equivariant (:numref:`sec_positional-information`), so permuting the input
tokens permutes the outputs in the same way, and no other part of the block
depends on order. Positions must be supplied separately, by one of the
schemes of :numref:`sec_positional-information`; :numref:`sec_gpt` adds
them.

:numref:`fig_transformer-block` shows the wiring, in the two normalization
arrangements compared below.

![The transformer block: attention and a position-wise FFN read from the residual stream and add their results back. Left, the 2017 arrangement: normalization sits on the stream itself, after each addition. Right, the modern arrangement: normalization moves onto each branch, and the stream runs uninterrupted from input to output.](../img/mdl-transformers-block-anatomy.svg)
:label:`fig_transformer-block`

## Normalization

The residual stream is a sum of dozens of sublayer outputs. Without some
control on its scale the sum drifts, and a deep stack becomes hard to train.
The 2017 transformer supplied that control with normalization, which
entered the block together with the residual connections
:cite:`Vaswani.Shazeer.Parmar.ea.2017`. The standard normalizer is *layer
normalization* :cite:`Ba.Kiros.Hinton.2016`
(:numref:`subsec_layer-normalization-in-bn`):
it subtracts the mean of each token's $d$ features, divides by their
standard deviation, and then applies a learned scale and shift per feature.
Unlike batch normalization (:numref:`sec_batch_norm`), it uses no batch
statistics, so it behaves identically in training and inference and does
not depend on the sequence length. The 2017 paper also fixed *where* the
normalization goes, and later work revised that choice.

### Two Arrangements

The original block normalizes the stream *after* each addition
(*post-norm*):

$$
\mathbf{h} = \mathrm{Norm}_1(\mathbf{x} + \mathrm{Attn}(\mathbf{x})), \qquad
\mathbf{y} = \mathrm{Norm}_2(\mathbf{h} + \mathrm{FFN}(\mathbf{h})).
$$
:eqlabel:`eq_postln-block`

The modern block :eqref:`eq_preln-block` normalizes each branch's *input*
(pre-norm) and leaves the stream alone. The literature also writes post-LN
and pre-LN, after the layer normalization (LN) the two arrangements first
used. In code the two differ by two lines. The minimal block below takes
the arrangement as a flag. Its attention is the `d2l.MultiHeadAttention` of
:numref:`sec_multihead-attention` with four heads, and its FFN is the
classic two-matrix MLP with the GELU activation of :numref:`sec_mlp`. This
minimal block isolates the placement; `TransformerBlock` below adds the
other options.

```{.python .input #transformer-block-two-arrangements}
%%tab pytorch
class MiniBlock(nn.Module):
    """Attention + MLP on a residual stream; norm placement as a flag."""
    def __init__(self, num_hiddens, pre_norm):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = d2l.MultiHeadAttention(num_hiddens, 4)
        self.W_1 = nn.Linear(num_hiddens, 4 * num_hiddens)
        self.W_2 = nn.Linear(4 * num_hiddens, num_hiddens)
        self.norm1 = nn.LayerNorm(num_hiddens, eps=1e-5)
        self.norm2 = nn.LayerNorm(num_hiddens, eps=1e-5)

    def ffn(self, X):  # exact GELU
        return self.W_2(F.gelu(self.W_1(X)))

    def forward(self, X):
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.attention(Y, Y, Y)
            return X + self.ffn(self.norm2(X))
        X = self.norm1(X + self.attention(X, X, X))
        return self.norm2(X + self.ffn(X))
```

```{.python .input #transformer-block-two-arrangements}
%%tab jax
class MiniBlock(nnx.Module):
    """Attention + MLP on a residual stream; norm placement as a flag."""
    def __init__(self, num_hiddens, pre_norm, *, rngs):
        self.pre_norm = pre_norm
        self.attention = d2l.MultiHeadAttention(num_hiddens, 4, rngs=rngs)
        self.W_1 = d2l.Linear(num_hiddens, 4 * num_hiddens, rngs=rngs)
        self.W_2 = d2l.Linear(4 * num_hiddens, num_hiddens, rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)

    def ffn(self, X):  # exact GELU
        return self.W_2(jax.nn.gelu(self.W_1(X), approximate=False))

    def __call__(self, X):
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.attention(Y, Y, Y)
            return X + self.ffn(self.norm2(X))
        X = self.norm1(X + self.attention(X, X, X))
        return self.norm2(X + self.ffn(X))
```

### Signal Propagation at Initialization
:label:`subsec_signal-propagation`

Much of the difference between the two arrangements shows before any
training. A freshly initialized network is a fixed random function, so we
can *measure* how it treats signals. We stack 32 blocks and pass a random
sequence of 64 tokens through the stack. At every depth we record the scale
of the residual stream, as the root mean square (RMS) of its entries, how
*distinct* the tokens remain from one another, and how much gradient each
block's attention receives from a loss at the top.

For distinctness we use the cross-token spread
$\|\mathbf{H} - \bar{\mathbf{H}}\| / \|\mathbf{H}\|$, where
$\mathbf{H}$ holds the token vectors of one sequence as rows and
$\bar{\mathbf{H}}$ copies their average to every row (the cell pools the two
sequences of its batch into one ratio). The spread is about $1$ for
unrelated random tokens and $0$ when all tokens have collapsed onto a single
point. The loss takes some care. For the post-norm stack, which ends
in a normalization, the obvious probe $\|\mathbf{H}\|^2$ is constant by
construction and has zero gradient, so in both stacks we use a generic
linear readout $\langle \mathbf{R}, \mathbf{H} \rangle$ for a fixed random
$\mathbf{R}$ instead.

By default `signal_stats` gives every weight matrix the Xavier
initialization of the original transformer (:numref:`subsec_xavier`), with
entries of variance $2/(d_\textrm{in} + d_\textrm{out})$, or $1/d$ for a
square matrix. Given `std`, it draws them from
$\mathcal{N}(0, \mathrm{std}^2)$ instead, which we will need shortly.

```{.python .input #transformer-block-signal-propagation-at-initialization-1}
%%tab pytorch
def signal_stats(pre_norm, std=None, num_blks=32, num_hiddens=256):
    """Token spread, stream RMS, and gradient norms of a fresh stack whose
    weight matrices are Xavier, or N(0, std^2) if std is given."""
    blks = nn.ModuleList([MiniBlock(num_hiddens, pre_norm)
                          for _ in range(num_blks)])
    init = (nn.init.xavier_uniform_ if std is None  # the 2017 initialization
            else lambda W: nn.init.normal_(W, std=std))
    for p in blks.parameters():
        if p.ndim == 2:
            init(p)
    norm = torch.linalg.vector_norm
    H = torch.randn(2, 64, num_hiddens)
    spread, rms = [], []
    for blk in blks:
        H = blk(H)
        spread.append(norm(H - H.mean(1, keepdim=True)) / norm(H))
        rms.append(H.pow(2).mean().sqrt())
    (H * torch.randn_like(H)).mean().backward()  # generic readout loss
    grad_q = [norm(blk.attention.W_q.weight.grad) for blk in blks]
    proj = {n: norm(getattr(blks[-1].attention, n).weight.grad).item()
            for n in ('W_q', 'W_k', 'W_v', 'W_o')}
    return {'spread': torch.stack(spread).tolist(),
            'rms': torch.stack(rms).tolist(),
            'grad_q': torch.stack(grad_q).tolist(), 'proj': proj}
```

```{.python .input #transformer-block-signal-propagation-at-initialization-1}
%%tab jax
def signal_stats(pre_norm, std=None, num_blks=32, num_hiddens=256):
    """Token spread, stream RMS, and gradient norms of a fresh stack whose
    weight matrices are Xavier, or N(0, std^2) if std is given."""
    rngs = nnx.Rngs(d2l.get_key())
    stack = nnx.Sequential(*[MiniBlock(num_hiddens, pre_norm, rngs=rngs)
                             for _ in range(num_blks)])
    init = (nnx.initializers.xavier_uniform() if std is None  # as in 2017
            else nnx.initializers.normal(std))
    for _, m in nnx.iter_graph(stack):
        if isinstance(m, nnx.Linear):
            m.kernel[...] = init(d2l.get_key(), m.kernel.shape)
    X = jax.random.normal(d2l.get_key(), (2, 64, num_hiddens))
    R = jax.random.normal(d2l.get_key(), X.shape)
    # Full fp32 products: on a GPU, JAX's default TF32 rounding (about 1e-3)
    # would swamp token differences of 1e-4
    with jax.default_matmul_precision('highest'):
        H, spread, rms = X, [], []
        for blk in stack.layers:
            H = blk(H)
            spread.append(jnp.linalg.norm(H - H.mean(1, keepdims=True))
                          / jnp.linalg.norm(H))
            rms.append(jnp.sqrt((H ** 2).mean()))
        g = nnx.grad(lambda m: (m(X) * R).mean())(stack)  # generic readout
    gnorm = lambda k, n: jnp.linalg.norm(
        g['layers'][k]['attention'][n]['kernel'][...])
    grad_q = [gnorm(k, 'W_q') for k in range(num_blks)]
    proj = {n: float(gnorm(num_blks - 1, n))
            for n in ('W_q', 'W_k', 'W_v', 'W_o')}
    return {'spread': jnp.stack(spread).tolist(),
            'rms': jnp.stack(rms).tolist(),
            'grad_q': jnp.stack(grad_q).tolist(), 'proj': proj}
```

Before running anything, let us predict what a post-norm stack does to its
tokens. We start simple and leave out the FFN; it comes back once we have a
measurement. At initialization the attention weights do depend on the
tokens, but only through the random projections $\mathbf{W}_q$ and
$\mathbf{W}_k$, which have no reason to favor a token for what it holds.
Here is the bold step of the argument: we treat the weights as unrelated to
content. Each token then receives a mixture of all the tokens, itself
included, whose expectation is the mean token $\boldsymbol{\mu}$. The
deviations of the tokens from that mean point in unrelated directions and
largely cancel in the mixture. To a first approximation, then, an attention
sublayer adds the same vector $\mathbf{W}_o \mathbf{W}_v \boldsymbol{\mu}$
to every token. The part that all tokens share grows, and the differences
between them do not.

The normalization that follows shrinks both by the same factor, and nothing
undoes the growth. But does LayerNorm not subtract a mean? It does, the mean
of one token's $d$ features, a single number per token. The mean over
tokens, $\boldsymbol{\mu}$, is a vector that LayerNorm never sees, so the
shared part survives it.

How fast does this happen? Write each token as the mean plus a deviation,
$\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$, and measure how
alike the $n$ tokens are by the ratio of their shared to their individual
energy,

$$
\rho = \frac{n \|\boldsymbol{\mu}\|^2}{\sum_{i=1}^n \|\boldsymbol{\delta}_i\|^2}.
$$

The deviations sum to zero, so the energy splits,
$\|\mathbf{H}\|^2 = n\|\boldsymbol{\mu}\|^2 + \sum_i \|\boldsymbol{\delta}_i\|^2$,
and the spread is $s = (1 + \rho)^{-1/2}$. For $n$ independent tokens with
standard normal entries the mean has squared length about $d/n$, while the
deviations together hold about $(n-1)d$, so $\rho \approx 1/(n-1)$. Our 64
tokens start at $\rho_0 = 1/63$, a spread of 0.99. Now let $\mathbf{W}_v$ and
$\mathbf{W}_o$ have independent entries of variance $g^2/d$. Each of the two
random matrices multiplies the squared length of a vector by about $g^2$,
and the cross term between $\boldsymbol{\mu}$ and
$\mathbf{W}_o \mathbf{W}_v \boldsymbol{\mu}$ averages to zero. One
attention sublayer therefore multiplies the shared energy by about
$1 + g^4$ and leaves the individual energy where it was. After $k$ blocks
the spread is

$$
s_k \approx \big(1 + \rho_0 (1 + g^4)^k\big)^{-1/2}.
$$
:eqlabel:`eq_collapse-rate`

The ratio $\rho$ grows geometrically, at a rate set by the *fourth* power of
the weight scale. Xavier initialization of a square matrix has $g = 1$, so
every block doubles $\rho$, and the predicted spread is 0.44 after 8 blocks,
0.031 after 16, and $1.2 \times 10^{-4}$ after 32. GPT-2 draws its weights
from $\mathcal{N}(0, 0.02^2)$. At our width $d = 256$ that is
$g^2 = 256 \cdot 0.02^2 \approx 0.1$, a factor of 1.01 per block, and a
predicted spread of 0.99 after 32 blocks. The arrangement and the depth
are the same, yet one initialization should flatten the tokens and the
other should barely touch them.

We test this on three stacks: post-norm and pre-norm with Xavier weights,
and post-norm with $\mathcal{N}(0, 0.02^2)$ weights. (GPT-2 also scales
down the two matrices of each block that write to the stream, which would
only slow the collapse further.) For each stack the cell prints the stream
RMS at the top, the spread at blocks 8, 16 and 32 with the prediction
:eqref:`eq_collapse-rate` beneath it for post-norm, and the gradient norms
of the top block's attention projections. Weights and tokens are drawn afresh
at every run, so the last digits move; the orders of magnitude do not.

```{.python .input #transformer-block-signal-propagation-at-initialization}
%%tab pytorch, jax
arms = {'post-norm': (False, None), 'pre-norm': (True, None),
        'post-norm, std 0.02': (False, 0.02)}
stats, predicted = {}, {}
at = lambda xs: ' '.join(f'{xs[k - 1]:.1e}' for k in (8, 16, 32))
for name, (pre_norm, std) in arms.items():
    stats[name] = s = signal_stats(pre_norm, std)
    print(f'{name}: stream RMS at k=32 {s["rms"][-1]:.2f}')
    print(f'  token spread at k=8,16,32: {at(s["spread"])}')
    if not pre_norm:  # rho_0 = 1/63 for 64 random tokens; Xavier: g = 1
        g2 = 1 if std is None else 256 * std ** 2
        predicted[name] = [(1 + (1 + g2 ** 2) ** k / 63) ** -0.5
                           for k in range(1, 33)]
        print(f'  predicted:                 {at(predicted[name])}')
    print('  grad norms at k=32: ' + '  '.join(
        f'{n} {g:.1e}' for n, g in s['proj'].items()))
```

The next cell plots the spread at every depth, with the predictions dotted:

```{.python .input #transformer-block-signal-propagation-at-initialization-3}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['spread'] for s in stats.values()]
         + list(predicted.values()), 'block index k', 'token spread',
         legend=list(arms) + ['predicted'], yscale='log', ylim=(1e-6, 2),
         fmts=('C0-', 'C1-', 'C2-', 'k:', 'k:'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

The prediction holds. With Xavier weights the post-norm spread falls along
a straight line on the logarithmic axis, which is what geometric decay looks
like, and by block 32 the tokens agree to about one part in ten thousand.
The measured spread is about 0.3, 0.02 and $10^{-4}$ at blocks 8, 16 and
32, below the prediction at every depth but by less than a factor of two.
Here the FFN we set aside comes back: GELU of a zero-mean input has a
positive mean, so the FFN, too, adds a component that all tokens share.
With GPT-2's standard deviation the same post-norm stack keeps its tokens
apart: its spread is still 0.97 after 32 blocks.

The collapse needs two things, post-norm and weights large enough that
$g^4$ is not small. Many successful models are post-norm. GPT-1
:cite:`Radford.Narasimhan.Salimans.ea.2018` and BERT
:cite:`Devlin.Chang.Lee.ea.2018` are, and DeepNet trains post-norm stacks
of a thousand layers by scaling down the initial value, output and FFN
weights and scaling up the stream that each branch is added to
:cite:`Wang.Ma.Dong.ea.2024`. Whether a post-norm stack collapses at
initialization is decided by the size of its weights.

The pre-norm stack behaves differently from the start, because it never
rescales its stream. Post-norm pins the RMS of the stream at 1 at every
depth. Pre-norm only ever adds, and each branch reads the stream through
its own normalization, so the branch output has a fixed scale however large
the stream has become. If every block added a contribution of unit mean
square, uncorrelated with the stream, the mean square after 32 blocks would
be $1 + 32 = 33$, an RMS of $\sqrt{33} \approx 5.7$. The premise is rough.
The first blocks add much less, because the attention of a fresh block
averages many unrelated tokens, which nearly cancel. Deeper blocks, whose
tokens have grown more alike, add about as much as the premise assumes or a
little more, and their outputs are not quite uncorrelated with the stream.
The two errors pull in opposite directions and do not quite cancel: the
measured RMS ends near 5, somewhat below $\sqrt{33}$. For so rough a
premise that is close enough, and the point survives: the stream grows,
and what a late block adds is small against it.

The attention sublayers of the pre-norm stack add shared vectors too, but
each is a fixed amount added to a stream that keeps growing. Post-norm
multiplies the shared part at every block; pre-norm only adds to it. The
pre-norm spread therefore falls slowly, to about a third after 32 blocks. A
spread of a third means $\rho = 8$: up from $1/63$, but nothing like the
$2^{32}/63 \approx 7 \times 10^7$ that Xavier weights predict for the
post-norm stack.

The two stacks that keep their tokens apart follow one rule: a deep stack
stays well behaved when every branch adds little compared with the stream
it is added to. Pre-norm achieves this for the later blocks by letting the
stream grow. Small weights in the matrices that write to the stream achieve
it for every block, in either arrangement, which is why the third stack
keeps its tokens.

### Gradients at Initialization

Collapsed tokens starve attention of gradient. When every token is nearly
the same vector, every query scores every key alike, so the attention
weights, and with them the loss, barely depend on $\mathbf{W}_q$ and
$\mathbf{W}_k$. At block 32 the query and key projections of the post-norm
stack with Xavier weights receive gradients of order $10^{-9}$, more than
five orders of magnitude less than in the pre-norm stack. The value and
output projections, which act on the averaged token itself, still receive
gradients of several hundredths. The gradient of each block's query
projection, plotted against depth, shows where the starvation sets in:

```{.python .input #transformer-block-signal-propagation-at-initialization-2}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['grad_q'] for s in stats.values()],
         'block index k', 'query projection gradient norm',
         legend=list(arms), yscale='log', ylim=(1e-12, 1),
         fmts=('C0-', 'C1-', 'C2-'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

The post-norm curve says more than its endpoint. In the lowest blocks,
where the tokens are still distinct, the query projections receive
gradients of about $10^{-2}$, several times below those of the pre-norm
stack. From there the curve falls smoothly, block by block, as the spread
does, until it levels off at a few times $10^{-9}$ in the top few blocks.
The backward pass reaches the bottom of the stack through thirty
blocks of collapsed tokens and arrives in good shape. The starvation is
local: a block's query and key weights receive almost no gradient because
its own input tokens are alike, not because the gradient vanished on its
way down. The pre-norm curve falls as well, by a factor of 20 to 30 over
32 blocks, because its tokens also grow more alike with depth, only slowly.
With GPT-2's standard deviation the curve is flat at about
$4 \times 10^{-4}$: small weights give small gradients, but no block is
starved more than another.

These measurements show one way in which a post-norm stack can start
badly. Whether it also trains badly only training can tell.
:citet:`xiong2020layer` found that post-norm transformers need a
learning-rate warmup to train well, while pre-norm ones train without it.
We use pre-norm as the default.

### RMSNorm
:label:`subsec_rmsnorm`

Modern models also simplified the normalizer itself. Of the four steps of
LayerNorm (center, divide, scale, shift), *RMSNorm*
:cite:`Zhang.Sennrich.2019` drops the centering and the shift and keeps the
division by the root mean square and the scale:

$$
\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}.
$$
:eqlabel:`eq_rmsnorm`

Thus RMSNorm rescales each token to unit RMS and then applies a learned
gain $\boldsymbol{\gamma}$ per feature. A small case shows what is lost.
For $\mathbf{x} = (1, 2, 3, 6)$ the RMS is $\sqrt{12.5} = 3.54$, so RMSNorm
with unit gain returns $(0.28, 0.57, 0.85, 1.70)$. LayerNorm subtracts the
mean 3 and divides by the standard deviation $\sqrt{3.5} = 1.87$, which
gives $(-1.07, -0.53, 0, 1.60)$. Multiplying $\mathbf{x}$ by 3 should change
neither output. Adding 10 to every coordinate should leave LayerNorm alone,
since it subtracts the mean first, and change RMSNorm, which has no
centering step. The cell checks a one-line implementation against the
built-in RMSNorm. It then applies our RMSNorm and LayerNorm, both with unit
gain and zero shift, to $\mathbf{x}$, $3\mathbf{x}$ and $\mathbf{x} + 10$:

```{.python .input #transformer-block-rmsnorm}
%%tab pytorch
def rms_norm(X, weight=1, eps=1e-5):
    return X / torch.sqrt((X * X).mean(-1, keepdim=True) + eps) * weight

x = torch.tensor([1.0, 2.0, 3.0, 6.0])
rmsnorm, layernorm = nn.RMSNorm(4, eps=1e-5), nn.LayerNorm(4, eps=1e-5)
show = lambda v: ' '.join(f'{a:5.2f}' for a in v.tolist())
print(f'ours vs. nn.RMSNorm: max deviation '
      f'{(rms_norm(x) - rmsnorm(x)).abs().max():.1e}')
for name, v in (('x', x), ('3x', 3 * x), ('x + 10', x + 10)):
    print(f'{name:>6}: RMSNorm {show(rms_norm(v))} | '
          f'LayerNorm {show(layernorm(v))}')
```

```{.python .input #transformer-block-rmsnorm}
%%tab jax
def rms_norm(X, weight=1, eps=1e-5):
    return X / jnp.sqrt((X * X).mean(-1, keepdims=True) + eps) * weight

x = jnp.array([1.0, 2.0, 3.0, 6.0])
rngs = nnx.Rngs(d2l.get_key())  # both start at unit gain and zero shift
rmsnorm = nnx.RMSNorm(4, epsilon=1e-5, rngs=rngs)
layernorm = nnx.LayerNorm(4, epsilon=1e-5, rngs=rngs)
show = lambda v: ' '.join(f'{a:5.2f}' for a in v.tolist())
print(f'ours vs. nnx.RMSNorm: max deviation '
      f'{jnp.abs(rms_norm(x) - rmsnorm(x)).max():.1e}')
for name, v in (('x', x), ('3x', 3 * x), ('x + 10', x + 10)):
    print(f'{name:>6}: RMSNorm {show(rms_norm(v))} | '
          f'LayerNorm {show(layernorm(v))}')
```

Our implementation agrees with the built-in to rounding error, and the
three rows bear out the prediction. The last row shows what the centering
was for. After an offset of 10, RMSNorm returns $(0.84, 0.91, 0.99, 1.22)$
instead of $(0.28, 0.57, 0.85, 1.70)$: the shared offset swamps the
differences between the coordinates, and only LayerNorm's subtraction of
the mean would remove it. RMSNorm gives up this one invariance, and
transformers have not missed it. :citet:`Zhang.Sennrich.2019` match
LayerNorm's accuracy on several tasks without the centering, and Llama
adopted RMSNorm :cite:`touvron2023llama`, as have most open models since.
Speed is a weak argument for it. A normalization reads and writes each
number of the stream once or twice, while the matrix products of the block
perform about $12d$ multiply-adds per number, thousands at any realistic
width, so with fused kernels the normalization is a small share of the
time. The centering turned out not to matter, and a step that does not
matter is better left out.

### Normalizing Queries and Keys
:label:`subsec_qk-norm`

A further normalization sits *inside* attention. Nothing in the block above
controls the size of the attention logits
$\mathbf{q}^\top \mathbf{k} / \sqrt{d_h}$, where $d_h$ is the head width.
If training inflates the queries and keys, the logits grow, the softmax
puts nearly all its weight on one key, and its Jacobian vanishes
(:numref:`sec_attention-scoring-functions`): the attention weights stop
responding to changes in the logits, and training can spike or stall.
Dehghani et al. met this instability while scaling vision transformers
toward 22 billion parameters: in models of about 8 billion parameters the
attention logits grew until the attention weights were nearly one-hot, and
the training loss diverged :cite:`Dehghani.Djolonga.Mustafa.ea.2023`.
:citet:`Wortsman.Liu.Xiao.ea.2024` reproduce it in small language models
trained at high learning rates. Both remove it with *QK-norm*: a LayerNorm
on the queries and on the keys of each head right before the dot product,
with the factor $1/\sqrt{d_h}$ kept. Several current open models do the
same with RMSNorm. The idea goes back to
:citet:`Henry.Dachapally.Pawar.ea.2020`, who scaled queries and keys to
unit length and replaced $1/\sqrt{d_h}$ by a learned scale. The variants
differ in details that matter little. What matters is that the size of the
logits no longer follows the size of the query and key weights.

We can mimic the inflation without training. We draw 64 random tokens of
width $d = 256$ and one head of width $d_h = 64$, with query and key weights
at the Xavier scale of the measurement above, multiply both weight matrices
by $\lambda \in \{1, 4, 16\}$, and measure the standard deviation of the
logits and the mean entropy of each query's attention weights, without and
with RMSNorm on the queries and keys. Uniform weights over 64 keys have
entropy $\log 64 = 4.16$; one-hot weights have entropy 0. Queries and keys
both grow by $\lambda$, so without normalization the logits should grow by
$\lambda^2$, to about 16 and 256 times their size at $\lambda = 1$. With
normalization the scale cancels, as the factor 3 did above, and nothing
should change.

```{.python .input #transformer-block-normalizing-queries-and-keys}
%%tab pytorch
def logit_stats(Q, K):
    """Std of the scaled logits; mean entropy of each query's weights."""
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    entropy = -(F.softmax(S, -1) * F.log_softmax(S, -1)).sum(-1).mean()
    return S.std(), entropy

d, d_h = 256, 64
X = torch.randn(64, d)                                # 64 random tokens
W_q, W_k = torch.randn(2, d, d_h) / math.sqrt(d)      # one head
for lam in (1, 4, 16):
    Q, K = X @ (lam * W_q), X @ (lam * W_k)
    s, h = logit_stats(Q, K)
    s_n, h_n = logit_stats(rms_norm(Q), rms_norm(K))
    print(f'lambda {lam:2d}: logit std {s:5.1f}, entropy {h:.2f} | '
          f'QK-norm: logit std {s_n:.2f}, entropy {h_n:.2f}')
```

```{.python .input #transformer-block-normalizing-queries-and-keys}
%%tab jax
def logit_stats(Q, K):
    """Std of the scaled logits; mean entropy of each query's weights."""
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    entropy = -(jax.nn.softmax(S) * jax.nn.log_softmax(S)).sum(-1).mean()
    return S.std(), entropy

d, d_h = 256, 64
X = jax.random.normal(d2l.get_key(), (64, d))         # 64 random tokens
W_q, W_k = jax.random.normal(d2l.get_key(), (2, d, d_h)) / math.sqrt(d)
for lam in (1, 4, 16):
    Q, K = X @ (lam * W_q), X @ (lam * W_k)
    s, h = logit_stats(Q, K)
    s_n, h_n = logit_stats(rms_norm(Q), rms_norm(K))
    print(f'lambda {lam:2d}: logit std {s:5.1f}, entropy {h:.2f} | '
          f'QK-norm: logit std {s_n:.2f}, entropy {h_n:.2f}')
```

Without the normalization the logit spread grows from 1 to about 16 and
250, as predicted, and the entropy collapses from about 3.7 to near 0. Each
query puts almost all its weight on one key, where the softmax has almost
no gradient. With QK-norm every row is the same, a logit spread near 1 and
an entropy of about 3.7, whatever $\lambda$ is. The first row also settles
a point of the collapse argument above. At the Xavier scale the entropy is
about 3.7, below the 4.16 of uniform weights. A fresh attention layer is not
uniform; its preferences are random, which is all the collapse argument
assumed.

QK-norm is not a flag of our block, because it changes the attention
sublayer itself. An exercise builds a QK-norm attention for the
`TransformerBlock` below.

## The Feed-Forward Network

The other half of the block is a small MLP applied to each position on
its own,

$$
\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\, \phi(\mathbf{W}_1 \mathbf{x}),
$$
:eqlabel:`eq_ffn`

with $\mathbf{W}_1 \in \mathbb{R}^{4d \times d}$ and $\mathbf{W}_2 \in
\mathbb{R}^{d \times 4d}$ in the classic configuration. Whatever a token
has gathered through attention is transformed here. The expansion by a
factor of 4 dates from the 2017 transformer and is still the common
default. With it the two matrices hold $8d^2$ parameters, twice the $4d^2$
of the four attention projections
$\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$.

:citet:`Geva.Schuster.Berant.ea.2021` read the two matrices as a memory
with the structure of attention. Each row of $\mathbf{W}_1$ acts as a key that
responds to a pattern in the token's vector, and the matching column of
$\mathbf{W}_2$ is the value that the FFN adds to the stream when its key
responds.

Two changes separate the 2017 FFN from current ones. First, the activation
$\phi$ moved from ReLU to the *GELU* $\phi(z) = z\,\Phi(z)$ of
:numref:`sec_mlp` :cite:`Hendrycks.Gimpel.2016`, where $\Phi$ is the
standard normal cumulative distribution function. GPT used it from the
start :cite:`Radford.Narasimhan.Salimans.ea.2018`. Second, the FFN
acquired a product. :citet:`Shazeer.2020` compared several *gated linear
units*, whose hidden layer multiplies two linear views of the input. The
one that stuck uses the SiLU activation:

$$
\mathrm{SwiGLU}(\mathbf{x}) = \mathbf{W}_2 \big(\mathrm{SiLU}(\mathbf{W}_g \mathbf{x}) \odot \mathbf{W}_1 \mathbf{x}\big), \qquad \mathrm{SiLU}(z) = z \cdot \mathrm{sigmoid}(z).
$$
:eqlabel:`eq_swiglu`

Instead of passing one linear view of the input through a fixed
nonlinearity, SwiGLU computes two views, $\mathbf{W}_g \mathbf{x}$ and
$\mathbf{W}_1 \mathbf{x}$, passes the first through SiLU, and multiplies
the two coordinate by coordinate. The name "gate" comes from the original
gated linear unit, whose first factor went through a sigmoid and so lay
between 0 and 1. SiLU is no such gate. It is close to 0 for large negative
$z$ and close to $z$ for large positive $z$, so it is unbounded above. Where
SiLU is nearly linear, the product of the two views is a quadratic function
of the input. In the key-value reading above, a key of SwiGLU can
therefore respond to the coincidence of two patterns in the token, where a
key of the classic FFN responds to one.

Does it help? Shazeer trained each variant in a transformer on T5's
span-filling task, with parameters matched. After 65,536 steps SwiGLU
reached a held-out log-perplexity of 1.944, against 1.983 for GELU and
1.997 for ReLU, while repeated runs of one setting differed by 0.005 to
0.01. Why the product helps is not understood. Shazeer's paper reports the
comparison and declines to explain it: "We offer no explanation as to why
these architectures seem to work; we attribute their success, as all else,
to divine benevolence." The evidence for SwiGLU is empirical: a margin of
0.04, small but several times the spread between runs, and the fact that
most open models released since Llama use a gated FFN of this kind.

A third matrix means more parameters at the same width, so a fair
comparison shrinks the hidden width to $\tfrac{8}{3} d$. The three matrices
of the gated FFN then hold the same $8d^2$ parameters as the classic two,
up to rounding. Both variants fit in one class:

```{.python .input #transformer-block-the-feed-forward-network-1}
%%tab pytorch
class FeedForward(nn.Module):  #@save
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count."""
    def __init__(self, num_hiddens, act='swiglu', bias=False):
        super().__init__()
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = nn.Linear(num_hiddens, width, bias=bias)
        self.W_1 = nn.Linear(num_hiddens, width, bias=bias)
        self.W_2 = nn.Linear(width, num_hiddens, bias=bias)

    def forward(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(F.gelu(self.W_1(X), approximate='tanh'))
        return self.W_2(F.silu(self.W_g(X)) * self.W_1(X))
```

```{.python .input #transformer-block-the-feed-forward-network-1}
%%tab jax
class FeedForward(nnx.Module):  #@save
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count."""
    def __init__(self, num_hiddens, act='swiglu', bias=False, *, rngs):
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = d2l.Linear(num_hiddens, width, use_bias=bias,
                                  rngs=rngs)
        self.W_1 = d2l.Linear(num_hiddens, width, use_bias=bias, rngs=rngs)
        self.W_2 = d2l.Linear(width, num_hiddens, use_bias=bias, rngs=rngs)

    def __call__(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(jax.nn.gelu(self.W_1(X), approximate=True))
        return self.W_2(jax.nn.silu(self.W_g(X)) * self.W_1(X))
```

`FeedForward` uses the tanh approximation of GELU, the variant GPT-2 was
trained with, which matters when we load its weights in :numref:`sec_gpt`.
`MiniBlock` used the exact form; the two never differ by more than
$5 \times 10^{-4}$. The product also helps at our scale: in the byte-level
language model of :numref:`subsec_gpt-run-to-run` the SwiGLU network ends
about 0.06 nats ahead of a GELU twin of the same width, several times the
spread between repeated runs.


## A Configurable Block

`TransformerBlock` is `MiniBlock` with every design decision of this
section as a constructor argument: `norm` selects LayerNorm or RMSNorm,
`act` selects the FFN, and `pre_norm` selects the arrangement. The forward
pass (`forward` in PyTorch, `__call__` in JAX) also takes an attention mask
and passes any further keyword arguments on to the attention;
:numref:`sec_gpt` needs both. Two further constructor arguments,
`attn_factory` and `ffn_factory`, exist so that later sections can replace
a whole sublayer; they can be ignored for now. The default configuration
(pre-norm, RMSNorm, SwiGLU) is the one Llama adopted in 2023
:cite:`touvron2023llama`, and many open-weights decoders have kept it since.
`TransformerBlock(num_hiddens, num_heads, norm='layer', act='gelu',
pre_norm=False, bias=True)` is a compact post-norm block in the 2017 encoder
arrangement, with one anachronism kept for simplicity: GELU, where the
original used ReLU.

```{.python .input #transformer-block-a-configurable-block}
%%tab pytorch
class TransformerBlock(nn.Module):  #@save
    """Configurable transformer block: attention + FFN on a residual
    stream."""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None):
        super().__init__()
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nn.RMSNorm if norm == 'rms' else nn.LayerNorm
        self.norm1 = make_norm(num_hiddens, eps=1e-5)
        self.norm2 = make_norm(num_hiddens, eps=1e-5)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias)
                          if attn_factory is None else attn_factory())
        self.ffn = (FeedForward(num_hiddens, act, bias=bias)
                    if ffn_factory is None else ffn_factory())
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None, **attn_args):
        # attn_args go on to the attention, e.g. the RoPE tables of d2l.GPT
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask, **attn_args))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask,
                                                       **attn_args)))
        return self.norm2(X + self.dropout(self.ffn(X)))
```

```{.python .input #transformer-block-a-configurable-block}
%%tab jax
class TransformerBlock(nnx.Module):  #@save
    """Configurable transformer block: attention + FFN on a residual
    stream."""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None, rngs):
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nnx.RMSNorm if norm == 'rms' else nnx.LayerNorm
        self.norm1 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias,
                                                 rngs=rngs)
                          if attn_factory is None else attn_factory(rngs))
        self.ffn = (FeedForward(num_hiddens, act, bias=bias, rngs=rngs)
                    if ffn_factory is None else ffn_factory(rngs))
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, X, mask=None, **attn_args):
        # attn_args go on to the attention, e.g. the RoPE tables of d2l.GPT
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask, **attn_args))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask,
                                                       **attn_args)))
        return self.norm2(X + self.dropout(self.ffn(X)))
```

Three defaults need a word. The linear layers keep their default
initialization, uniform on
$[-1/\sqrt{d_\textrm{in}}, 1/\sqrt{d_\textrm{in}}]$, a variance of $1/(3d)$
for a square matrix. That is $g^2 = 1/3$ in :eqref:`eq_collapse-rate`, so
even a post-norm stack of this class would multiply $\rho$ by only
$1 + 1/9$ per block, for a predicted spread of about 0.8 after 32 blocks,
between the two initializations we measured. The measurement used Xavier
because it is the initialization of 2017 and shows the effect plainly;
:numref:`sec_gpt` replaces the default with GPT-2's
$\mathcal{N}(0, 0.02^2)$. Biases are off, as in Llama and PaLM: leaving
them out has not hurt quality, and the PaLM authors report that it made
their large models train more stably :cite:`chowdhery2022palm`. Dropout is
0, since a language model that almost never reads the same passage twice
has little to memorize.

### Parameter Count

Whatever the flags, the block's parameters follow from the accounting
above: $4d^2$ in the four attention projections, $8d^2$ in the FFN however
the activation divides it, and a few normalization weights. That makes
about $12d^2$ per block, or $786{,}432$ at $d = 256$, two thirds of it in
the FFN. We count the default block, the same block with the GELU FFN,
and the 2017 block with `d2l.num_params` from :numref:`sec_efficient_cnns`,
which counts the trainable parameters of a model:

```{.python .input #transformer-block-shapes-and-parameters-1}
%%tab pytorch
blocks = {'default': TransformerBlock(256, 8),
          'GELU': TransformerBlock(256, 8, act='gelu'),
          '2017': TransformerBlock(256, 8, norm='layer', act='gelu',
                                   pre_norm=False, bias=True)}
for name, blk in blocks.items():
    attn, ffn, total = (d2l.num_params(m)
                        for m in (blk.attention, blk.ffn, blk))
    print(f'{name:>7}: attention {attn:,}, FFN {ffn:,}, total {total:,} '
          f'= {total / 256 ** 2:.2f} d^2')
```

```{.python .input #transformer-block-shapes-and-parameters-1}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
blocks = {'default': TransformerBlock(256, 8, rngs=rngs),
          'GELU': TransformerBlock(256, 8, act='gelu', rngs=rngs),
          '2017': TransformerBlock(256, 8, norm='layer', act='gelu',
                                   pre_norm=False, bias=True, rngs=rngs)}
for name, blk in blocks.items():
    attn, ffn, total = (d2l.num_params(m)
                        for m in (blk.attention, blk.ffn, blk))
    print(f'{name:>7}: attention {attn:,}, FFN {ffn:,}, total {total:,} '
          f'= {total / 256 ** 2:.2f} d^2')
```

Without biases the attention projections hold exactly $4d^2$. The GELU and
SwiGLU networks differ by 256 parameters, 0.05% of the FFN, because
$\tfrac{8}{3} \cdot 256$ rounds up to 683. The biases and LayerNorm shifts
of the 2017 block add a few thousand more, and every variant lands at about
$12d^2$. The same count scales to GPT-2 small, whose blocks have width
$d = 768$: $12d^2 \approx 7.1$ million parameters per block, and about 85
million in its 12 blocks, more than two thirds of the model's 124 million.

A stack of these blocks is not yet a language model, and a measurement at
initialization cannot settle how a stack trains. Whether the placement of
the normalization still matters once training starts, at the small weights
a language model actually uses, is the question :numref:`sec_gpt` answers
when it builds and trains one.

## Summary

A transformer block pairs attention, which moves information between
positions but computes no nonlinear feature of any one position, with a
position-wise FFN that holds two thirds of the block's $12d^2$ parameters;
both add their outputs to a residual stream. At initialization each
attention sublayer adds nearly the same vector to every token, and
post-norm, by renormalizing the stream, turns this into a geometric
collapse with a rate of about $1 + g^4$ per block. With Xavier weights 32
such blocks leave the tokens agreeing to one part in $10^4$ and their top
query and key projections with almost no gradient, while GPT-2's smaller
weights collapse nothing. Pre-norm normalizes only what each branch reads,
so its stream grows and the later branches stay small relative to it.
RMSNorm drops LayerNorm's centering, which turned out not to matter,
QK-norm bounds the attention logits however large the query and key
weights grow, and SwiGLU multiplies two linear views of the input at the
parameter count of the classic MLP and beats it, for reasons that are not
understood. `TransformerBlock` exposes all of these choices as constructor
arguments.

## Exercises

1. **Scale invariance.** Let RMSNorm act as in :eqref:`eq_rmsnorm` with
   $\boldsymbol{\gamma} = \mathbf{1}$ and $\epsilon = 0$, and write
   $r(\mathbf{x}) = (\tfrac{1}{d} \sum_i x_i^2)^{1/2}$ for the root mean
   square of $\mathbf{x} \in \mathbb{R}^d$.
    1. Show that $\mathrm{RMSNorm}(a \mathbf{x}) =
       \mathrm{RMSNorm}(\mathbf{x})$ for every $a > 0$, and that
       LayerNorm with unit gain and zero shift is also unchanged when the
       same constant is added to every coordinate. Describe the set of
       inputs that each normalizer maps to a given output.
    1. Show that the Jacobian of RMSNorm at $\mathbf{x}$ is
       $r(\mathbf{x})^{-1} (\mathbf{I} - \hat{\mathbf{x}}
       \hat{\mathbf{x}}^\top)$ with $\hat{\mathbf{x}} = \mathbf{x} /
       \|\mathbf{x}\|$. Conclude that the gradient
       $\nabla_{\mathbf{x}} \ell$ is orthogonal to $\mathbf{x}$ for every
       loss $\ell$ that depends on $\mathbf{x}$ only through
       $\mathrm{RMSNorm}(\mathbf{x})$.

1. **Unrolled residual stream.** Number the $2L$ sublayers of an
   $L$-block stack by $j = 1, \ldots, 2L$, write $F_j$ for the $j$-th
   sublayer (attention or FFN) and $\mathrm{Norm}_j$ for its
   normalization, and let $\mathbf{x}_j$ be the residual stream after
   sublayer $j$, so that $\mathbf{x}_0$ is the input.
    1. For the pre-norm arrangement :eqref:`eq_preln-block`, write
       $\mathbf{x}_{2L}$ as $\mathbf{x}_0$ plus a sum of $2L$ branch
       outputs. Write the Jacobian $\partial \mathbf{x}_{2L} / \partial
       \mathbf{x}_0$ as a product over sublayers, expand the product, and
       identify the term that passes a gradient from the output to the
       input unchanged.
    1. For the post-norm arrangement :eqref:`eq_postln-block`, the stream is
       $\mathbf{x}_j = \mathrm{Norm}_j(\mathbf{u}_j)$ with $\mathbf{u}_j =
       \mathbf{x}_{j-1} + F_j(\mathbf{x}_{j-1})$. Model every normalization
       as RMSNorm with unit gain and $\epsilon = 0$. Assume that the input
       $\mathbf{x}_0$ has unit root mean square and that each branch output
       is uncorrelated with the stream it is added to. The product of the
       normalization Jacobians alone takes the place of the identity term
       of the pre-norm expansion. Show that this product scales a vector
       orthogonal to every $\mathbf{u}_j$ by $\prod_j r(\mathbf{u}_j)^{-1}$,
       and write this factor as a function of the mean squares
       $c_1, \ldots, c_{2L}$ of the branch outputs. Evaluate it for a
       common mean square $c_j = c$, and compare its dependence on $L$
       with that of the identity term of the pre-norm expansion.
    1. The stream of a pre-norm stack ends with an RMS that grows with
       depth, so language models built from pre-norm blocks apply one more
       normalization after the last block, before the output layer.
       Describe what the output layer would receive without it as the
       stack grows deeper, and explain why a post-norm stack needs no such
       final normalization.

1. [code] **QK-norm as a factory.**
   QK-norm (:numref:`subsec_qk-norm`) applies RMSNorm over the head
   dimension $d_h = d/H$ of a layer with $H$ heads to every query and key
   just before the scores are computed.
    1. Write `QKNormAttention`, a subclass of `d2l.MultiHeadAttention`
       whose `forward` (`__call__` in JAX) passes the per-head queries and
       keys produced by `split_heads` through their own RMSNorm each
       before the dot product. Build a `TransformerBlock` with it through
       `attn_factory`, and check that multiplying the weights of `W_q` and
       `W_k` by 16 leaves the output of the block unchanged up to
       rounding.
    1. Show that with unit gains every scaled logit
       $\hat{\mathbf{q}}^\top \hat{\mathbf{k}} / \sqrt{d_h}$ lies in
       $[-\sqrt{d_h}, \sqrt{d_h}]$, and generalize the bound to learned
       gains $\boldsymbol{\gamma}_q$ and $\boldsymbol{\gamma}_k$. Compute
       the largest possible ratio between two attention weights of one
       query for $d_h = 64$ and unit gains, and explain why the gains, and
       not the query and key weights, now set how sharp attention can be.

1. **FFN width rounding.** `FeedForward` sets the SwiGLU width to
   $\mathrm{round}(8d/3)$.
    1. Write the exact parameter count of the default
       `TransformerBlock(d, H)` as a function of $d$, and bound the
       relative deviation of its FFN from $8d^2$.
    1. The released 7B configuration of Llama :cite:`touvron2023llama` has
       $d = 4096$ and SwiGLU width $11008$, the smallest multiple of 256
       that is at least $\lfloor 8d/3 \rfloor$. Compute the relative excess
       of its FFN parameters over $8d^2$, and the excess that the same
       rule gives at $d = 256$ and at $d = 128$. Explain what the
       comparison implies for parameter-matched experiments with small
       models.
   :exlabel:`ex_transformer-block-ffn-width-rounding`

1. [code] **Stream variance budget.** ● In the pre-norm arrangement every
   branch reads a normalized input. If each branch output is uncorrelated
   with the stream it is added to, the mean square of the stream after $k$
   blocks is
   $$1 + \sum_{i=1}^{k} \big(c^\textrm{attn}_i + c^\textrm{ffn}_i\big),$$
   where $c^\textrm{attn}_i$ and $c^\textrm{ffn}_i$ are the mean squares of
   the attention and FFN outputs of block $i$. Use `MiniBlock` with the
   Xavier initialization of `signal_stats` at $d = 256$.
    1. Predict $c^\textrm{ffn}_1$. Compute the variance of the hidden
       pre-activations of the FFN, and evaluate the second moment of GELU
       at that variance numerically.
    1. Predict $c^\textrm{attn}_1$. Treat the $n = 64$ input tokens as
       independent, and treat the attention logits of one query as
       independent Gaussians with the variance that the initialization
       implies.
    1. Record the mean square of each branch output in `signal_stats` and
       plot $c^\textrm{attn}_k$ and $c^\textrm{ffn}_k$ against $k$.
       Compare block 1 with your predictions, and explain the trend of
       $c^\textrm{attn}_k$ using the token spread.
    1. Fit a power law to the stream RMS, once over blocks 1 to 32 and once
       over blocks 8 to 32. Compare both exponents with $1/2$, explain the
       comparison with the variance budget, and state the regime in which
       the estimate $\sqrt{1 + k}$ of :numref:`subsec_signal-propagation`
       holds.

1. [code] **Rate of token collapse.** ● :numref:`subsec_signal-propagation`
   derived :eqref:`eq_collapse-rate` in a few sentences. Here you make the
   argument precise and test where it holds. Write the tokens of one
   sequence as $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$
   for $i = 1, \ldots, n$, with $\sum_i \boldsymbol{\delta}_i =
   \mathbf{0}$, and let $\rho$ be the ratio of shared to individual energy.
    1. Show that the spread equals $(1 + \rho)^{-1/2}$ and that
       $\rho \approx 1/(n-1)$ for $n$ independent standard normal tokens.
       Let $\mathbf{M} = \mathbf{W}_o \mathbf{W}_v$, where the entries of
       $\mathbf{W}_v, \mathbf{W}_o \in \mathbb{R}^{d \times d}$ are
       independent with mean zero and variance $g^2/d$. Show that
       $E\|\mathbf{M}\boldsymbol{\mu}\|^2 = g^4 \|\boldsymbol{\mu}\|^2$ and
       $E[\boldsymbol{\mu}^\top \mathbf{M} \boldsymbol{\mu}] = 0$. The
       individual part of the update of token $i$ is
       $\mathbf{M} \sum_j a_{ij} \boldsymbol{\delta}_j$, where the
       attention weights $a_{ij}$ may be treated as independent of the
       tokens. Estimate its energy relative to
       $\sum_i \|\boldsymbol{\delta}_i\|^2$ in terms of $\sum_j a_{ij}^2$,
       and say when neglecting it is justified.
    1. Rerun `signal_stats` for post-norm with all Xavier weights multiplied
       by $g \in \{1, 0.7, 0.5\}$, once with the full `MiniBlock` and once
       with its FFN sublayer removed. Compare both with
       :eqref:`eq_collapse-rate` at blocks 8, 16, and 32. Show that GELU of
       a zero-mean Gaussian input has a positive mean, and use this to
       account for the gap of the full block.
    1. GPT-1 stacked 12 post-norm blocks of width $d = 768$ and drew every
       weight from $\mathcal{N}(0, 0.02^2)$
       :cite:`Radford.Narasimhan.Salimans.ea.2018`. Compute $g$ for its
       attention projections, and predict the spread after 12 blocks for
       $n = 64$, both for this initialization and for Xavier
       initialization at the same width. Check both predictions with
       `signal_stats`.

[Discussions](https://d2l.discourse.group/t/36774)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.1]{.kicker}

The Transformer Block<br>
**residual stream · normalization placement · token collapse · RMSNorm and QK-norm · SwiGLU · one configurable class**
:::
:::

::: {.slide title="Thirty-Two Fresh Post-Norm Blocks Squeeze 64 Tokens onto One Point"}
[The problem]{.kicker}

- Attention averages values: it mixes tokens but computes no new feature
  of any one token.
- Stack 32 blocks the 2017 way, with Xavier weights. Before any training
  the tokens agree to one part in $10^4$; $\mathbf{W}_q$ gets gradients
  near $10^{-9}$.

::: {.d2l-note .warn}
What must a block contain, and what keeps a deep stack's tokens apart?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count a block's parameters: $4d^2 + 8d^2$;
- predict a post-norm stack's token spread from its weight scale;
- explain why collapsed tokens starve $\mathbf{W}_q, \mathbf{W}_k$;
- write RMSNorm and QK-norm, and size SwiGLU to match the MLP.

**Running examples:** a GPT-2-small block ($d = 768$); 32 random blocks at
$d = 256$, three initializations.
:::

::: {.slide title="The Block Is What Every Transformer Repeats"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **The Transformer Block** | **attention + FFN on a residual stream** |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse keys and values while sampling |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, ... | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** multi-head attention, LayerNorm, GELU, Xavier initialization.
- **Skips:** positions and the causal mask (next deck), training.
:::
:::
:::

::: {.slide title="Two Sublayers Read from a Residual Stream and Add Back"}
@fig:mdl-transformers-block-anatomy

Attention is the only place tokens interact; the FFN acts on each token
alone. Both map $(n, d)$ to $(n, d)$, so blocks stack.
:::

::: {.slide title="Post-Norm Normalizes the Stream; Pre-Norm Normalizes What Each Branch Reads"}
Token vectors $\mathbf{x} \in \mathbb{R}^d$ on the residual stream. The
literature also says post-LN and pre-LN, after layer normalization:

$$\textrm{post-norm (2017):}\quad \mathbf{h} = \mathrm{Norm}_1(\mathbf{x} + \mathrm{Attn}(\mathbf{x})), \quad \mathbf{y} = \mathrm{Norm}_2(\mathbf{h} + \mathrm{FFN}(\mathbf{h}))$$

$$\textrm{pre-norm:}\quad \mathbf{h} = \mathbf{x} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{x})), \quad \mathbf{y} = \mathbf{h} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{h}))$$

- $\mathrm{Attn}$: multi-head self-attention with projections
  $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o \in
  \mathbb{R}^{d \times d}$.
- $\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x})$,
  hidden width $4d$.
:::

::: {.slide title="In Code the Two Arrangements Differ in Two Lines"}
```{.python #transformer-block-exercises-1}
def block(X, pre_norm):
    if pre_norm:                       # norm on each branch's input
        Y = norm1(X)
        X = X + attn(Y, Y, Y)
        return X + ffn(norm2(X))
    X = norm1(X + attn(X, X, X))       # 2017: norm on the stream
    return norm2(X + ffn(X))
```

Same weights, two different functions. A fresh stack is a fixed random
function, so how each treats signals can be measured before training.
:::

::: {.slide title="Fresh Attention Adds Nearly the Same Vector to Every Token"}
- Fresh weights see content only through random $\mathbf{W}_q,
  \mathbf{W}_k$. Bold step: treat them as unrelated to it.
- Each token gets a mixture of all tokens with expectation
  $\boldsymbol{\mu}$, the mean token; deviations largely cancel.
- Every token gains about $\mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$:
  the shared part grows, the differences do not.

::: {.d2l-note .rule}
Post-norm rescales all tokens alike, and LayerNorm's mean is per token,
not $\boldsymbol{\mu}$: nothing undoes the growth.
:::
:::

::: {.slide title="Each Post-Norm Block Multiplies the Shared Energy by $1 + g^4$"}
- Tokens $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$;
  $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_i \|\boldsymbol{\delta}_i\|^2$;
  spread $s = (1+\rho)^{-1/2}$.
- $\mathbf{W}_v, \mathbf{W}_o$ with entries of variance $g^2/d$: each
  multiplies squared lengths by $g^2$, so $\rho \to (1+g^4)\rho$.
- 64 random tokens: $\rho_0 = 1/63$, so
  $s_k \approx \big(1 + (1+g^4)^k/63\big)^{-1/2}$.

| predicted spread $s_k$, block $k$ | 8 | 16 | 32 |
|---|---|---|---|
| Xavier, $g = 1$: $\rho$ doubles per block | 0.44 | 0.031 | $1.2 \times 10^{-4}$ |
| GPT-2's $\mathcal{N}(0, 0.02^2)$ at $d = 256$: $g^2 \approx 0.1$ | 0.99 | 0.99 | 0.99 |
:::

::: {.slide title="The Measured Spread Follows the Prediction"}
Post-norm and pre-norm with Xavier weights; post-norm with
$\mathcal{N}(0, 0.02^2)$. Spread against depth, predictions dotted:

@!transformer-block-signal-propagation-at-initialization-3
:::

::: {.slide title="Small Weights Do Not Collapse"}
- Xavier: spread about 0.3, 0.02, $10^{-4}$; below prediction (by $< 2\times$),
  since the FFN (GELU has a positive mean) adds a shared part too.
- $\mathcal{N}(0, 0.02^2)$: $1 + g^4 \approx 1.01$; spread 0.97 after 32
  blocks.
- GPT-1 and BERT are post-norm; DeepNet trains 1,000 layers.

::: {.d2l-note .rule}
Collapse needs post-norm and large weights: the size of the weights decides.
:::

[Wang et al., 2024 (DeepNet)]{.small}
:::

::: {.slide title="Check: Is 0.02 Small at Width 1024?"}
[Check]{.kicker}

A post-norm stack of 24 blocks at $d = 1024$, 64 random tokens, weights
from $\mathcal{N}(0, 0.02^2)$. Predict the token spread at the top.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $g^2 = 1024 \cdot 0.02^2 = 0.41$, $g^4 = 0.17$;
$\rho = 1.17^{24}/63 \approx 0.66$, spread $\approx 0.78$: the tokens are
already partly merged. A fixed standard deviation is a larger $g$ at a
larger width. (Xavier: $\rho = 2^{24}/63 \approx 2.7 \times 10^5$, spread
$\approx 0.002$.)
:::
:::
:::

::: {.slide title="Identical Tokens Starve the Query and Key Weights"}
::: {.cols .vc}
::: {.col}
@!transformer-block-signal-propagation-at-initialization-2
:::

::: {.col .narrow}
- Equal tokens: every query scores every key alike, so the weights stop
  depending on $\mathbf{W}_q, \mathbf{W}_k$.
- Top post-norm block: $\sim 10^{-9}$; $\mathbf{W}_v, \mathbf{W}_o$ still
  get several hundredths.
- Lowest blocks: about $10^{-2}$. The starvation is local.
:::
:::
:::

::: {.slide title="Pre-Norm Adds to the Shared Part Instead of Multiplying It"}
- Branches read normalized inputs, so their outputs have fixed scale.
  Post-norm pins the stream RMS at 1.
- If each block adds unit mean square: RMS $\sqrt{33} \approx 5.7$ after
  32 blocks. Measured: about 5, somewhat below.
- Shared vectors land on a growing stream: spread about a third at 32,
  so $\rho = 8$, against $2^{32}/63 \approx 7 \times 10^7$ predicted for
  post-norm with Xavier.

::: {.d2l-note .rule}
Keep every branch small against the stream: pre-norm by growth, small
weights by size.
:::
:::

::: {.slide title="RMSNorm Drops the Centering, and Ignores Scale but Not Shift"}
$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

Learned gain $\boldsymbol{\gamma}$, small $\epsilon$. $\mathbf{x} = (1, 2,
3, 6)$: RMS $\sqrt{12.5} = 3.54$; mean 3, standard deviation
$\sqrt{3.5} = 1.87$. Unit gain, zero shift:

@!transformer-block-rmsnorm

::: {.d2l-note}
Last row: an offset swamps RMSNorm's output. Centering would remove it,
but in practice it did not matter, so current models leave it out.
:::

[Zhang & Sennrich, 2019]{.small}
:::

::: {.slide title="QK-Norm Bounds the Attention Logits"}
- Logits $\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}$, head width $d_h$.
  Inflated $\mathbf{q}, \mathbf{k}$: one-hot softmax, vanishing gradient.
- Seen while scaling ViTs (diverging at 8B parameters) and in small LMs
  at high learning rates.
- QK-norm: normalize each head's $\mathbf{q}, \mathbf{k}$ first. Unit
  gains: $|\textrm{logit}| \le \sqrt{d_h} = 8$.

[Henry et al., 2020; Dehghani et al., 2023; Wortsman et al., 2024]{.small}
:::

::: {.slide title="QK-Norm Keeps the Entropy Up When the Weights Grow"}
64 random tokens, one head of width 64; $\mathbf{W}_q, \mathbf{W}_k$
scaled by $\lambda$. Uniform weights: entropy $\log 64 = 4.16$; one-hot: 0.

@!transformer-block-normalizing-queries-and-keys

::: {.d2l-note}
Without the norm the logits grow like $\lambda^2$; with it nothing changes.
At $\lambda = 1$, entropy 3.7: fresh attention is not uniform, but its
preferences are random.
:::
:::

::: {.slide title="Check: Inflate the Query and Key Weights by 8"}
[Check]{.kicker}

At $\lambda = 1$ the logits have standard deviation about 1. What is it at
$\lambda = 8$, without and with QK-norm?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Queries and keys both grow by 8, so the logits grow by
$8^2 = 64$: standard deviation about 64, attention nearly one-hot. With
QK-norm the scale cancels: still about 1.
:::
:::
:::

::: {.slide title="FFN Rows Act as Keys, Columns as Values"}
$$\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x}) = \sum_{j=1}^{4d} \phi(\mathbf{w}_{1,j}^\top \mathbf{x})\, \mathbf{w}_{2,j}$$

- Row $\mathbf{w}_{1,j}$ of $\mathbf{W}_1$: a key that responds to a
  pattern in the token.
- Column $\mathbf{w}_{2,j}$ of $\mathbf{W}_2$: the value added to the
  stream when it responds.
- GPT-2 small: $4d = 3{,}072$ such memories per block.

[Geva et al., 2021]{.small}
:::

::: {.slide title="SwiGLU Multiplies Two Views of the Input"}
$$\mathrm{SwiGLU}(\mathbf{x}) = \mathbf{W}_2\big(\mathrm{SiLU}(\mathbf{W}_g\mathbf{x}) \odot \mathbf{W}_1\mathbf{x}\big), \quad \mathrm{SiLU}(z) = z\,\mathrm{sigmoid}(z)$$

- SiLU is unbounded above, not a gate in $[0, 1]$. Where SiLU is nearly
  linear, the product is quadratic in $\mathbf{x}$.
- Width $\tfrac{8}{3}d$ keeps $8d^2$: $2{,}048$ instead of $3{,}072$ at
  $d = 768$.
- Matched size, held-out log-perplexity: 1.944, against 1.983 (GELU)
  and 1.997 (ReLU); repeated runs differ by 0.005 to 0.01.

::: {.d2l-note}
"We offer no explanation as to why these architectures seem to work."
:::

[Shazeer, 2020]{.small}
:::

::: {.slide title="Both FFNs Fit One Class at the Same Budget"}
```{.python #transformer-block-exercises-2}
width = 4 * d if act == 'gelu' else round(8 * d / 3)

def ffn(x):
    if act == 'gelu':                   # W_1, W_2: 8 d^2
        return W_2(gelu(W_1(x)))
    return W_2(silu(W_g(x)) * W_1(x))   # W_g, W_1, W_2: 8 d^2
```

At $d = 256$: 524,288 parameters for GELU, 524,544 for SwiGLU, since
$\tfrac{8}{3} \cdot 256$ rounds to 683.
:::

::: {.slide title="A GPT-2-Small Block Holds 7.1 Million Parameters"}
[Running example]{.kicker}

| part | count | $d = 768$ |
|---|---|---|
| attention, 4 projections | $4d^2$ | 2.4M |
| FFN, two or three matrices | $8d^2$ | 4.7M |
| block | $12d^2$ | 7.1M |
| 12 blocks | | 85M |
| + token and position embeddings | | 124M |

::: {.d2l-note .rule}
Two thirds of every block sit in the FFN.
:::
:::

::: {.slide title="One Constructor Covers 2017 and Today"}
| argument | choices | default |
|---|---|---|
| `norm` | `'rms'`, `'layer'` | `'rms'` |
| `act` | `'swiglu'`, `'gelu'` | `'swiglu'` |
| `pre_norm` | `True`, `False` | `True` |
| `bias` | `True`, `False` | `False` |
| `attn_factory`, `ffn_factory` | replace a whole sublayer | `None` |

The default is the Llama configuration (2023); `norm='layer'`,
`act='gelu'`, `pre_norm=False`, `bias=True` gives the 2017 block, with GELU
for ReLU.
:::

::: {.slide title="Check: A Fair Comparison"}
[Check]{.kicker}

Someone compares SwiGLU with hidden width $4d$ against the GELU MLP at
$d = 768$. How many more parameters does the SwiGLU FFN have?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $3 \cdot 4d^2 = 12d^2$ against $8d^2$: 50% more, 7.1M
against 4.7M. Any gain could come from size; the width $\tfrac{8}{3}d$
removes the confound.
:::
:::
:::

::: {.slide title="Wrap-up"}
- Block: attention mixes tokens; the FFN holds two thirds of $12d^2$
  (7.1M in GPT-2 small).
- Post-norm: shared part $\times (1 + g^4)$ per block. Xavier: spread
  $10^{-4}$ at 32; $\mathcal{N}(0, 0.02^2)$: 0.97.
- Pre-norm only adds; stream RMS about 5 at 32, against
  $\sqrt{33} \approx 5.7$ if every block added unit mean square.
- RMSNorm drops centering; QK-norm bounds logits; SwiGLU multiplies two
  views.

**Next:** embeddings, positions, and a causal mask turn the block into a
GPT, trained on bytes.
:::
