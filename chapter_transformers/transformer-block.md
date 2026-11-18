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
that carries the signal through the stack. This section studies the two
design choices that separate the 2017 block from current ones: where the
normalization sits and what form the feed-forward network takes. We measure
the first at initialization, where it decides whether a 32-block stack keeps
its tokens distinct and its attention trainable. We measure the second in a
small language model at matched parameter count. A configurable
`TransformerBlock` class then collects these choices, and the rest of the
chapter builds on it.
<!-- WRITER: false since the code step: the GELU-against-SwiGLU training run left this section (CharLM is cut); the FFN is argued and counted here, its trained comparison is in :numref:`subsec_gpt-run-to-run` of sec_gpt. -->


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
  retrieved mixture to its stream. Contextual information enters through
  this sublayer.
- The **feed-forward network** (FFN) acts on each position separately: the
  same two- or three-matrix MLP is applied token by token, with no
  interaction between positions. It transforms each token's accumulated
  state and contains most of the block's parameters.

Writing $\mathrm{Attn}$ for multi-head self-attention and $\mathrm{Norm}$
for a normalization layer, the modern (*pre-norm*) block computes

$$
\mathbf{h} = \mathbf{x} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{x})), \qquad
\mathbf{y} = \mathbf{h} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{h})).
$$
:eqlabel:`eq_preln-block`

Both sublayers preserve the shape $(n, d)$. The block maps a sequence of
$d$-dimensional vectors to a sequence of the same length and width, so
dozens of copies can be stacked. The block also knows nothing about the
order of the tokens. Attention is permutation equivariant
(:numref:`sec_positional-information`), and no other part of the block
depends on order. The model around the blocks must supply positions, and
the next section shows how.

:numref:`fig_transformer-block` shows the wiring, in the two normalization
arrangements compared below.

![The transformer block: attention and a position-wise FFN read from the residual stream and add their results back. Left, the 2017 arrangement: normalization sits on the stream itself, after each addition. Right, the modern arrangement: normalization moves onto each branch, and the stream runs uninterrupted from input to output.](../img/mdl-transformers-block-anatomy.svg)
:label:`fig_transformer-block`

## Normalization Placement

Residual connections and normalization entered the block together in 2017
:cite:`Vaswani.Shazeer.Parmar.ea.2017`. Without some control on scale, the
sum of dozens of sublayer outputs drifts, and deep stacks become hard to
train. The standard normalizer is *layer normalization*
:cite:`Ba.Kiros.Hinton.2016` (:numref:`subsec_layer-normalization-in-bn`):
it subtracts the mean of each token's $d$ features, divides by their
standard deviation, and then applies a learned scale and shift per feature.
Unlike batch normalization (:numref:`sec_batch_norm`), it uses no batch
statistics, so it behaves identically in training and inference and does
not depend on the sequence length. The 2017 paper also fixed *where* the
normalization goes, and later work revised that choice.

### Two Arrangements

The original block normalizes the stream *after* each addition
("post-LN"):

$$
\mathbf{h} = \mathrm{Norm}_1(\mathbf{x} + \mathrm{Attn}(\mathbf{x})), \qquad
\mathbf{y} = \mathrm{Norm}_2(\mathbf{h} + \mathrm{FFN}(\mathbf{h})).
$$
:eqlabel:`eq_postln-block`

The modern block :eqref:`eq_preln-block` normalizes each branch's *input*
("pre-LN") and leaves the stream alone. In code the two differ by two
lines. The minimal block below takes the arrangement as a flag. Its
attention is the `d2l.MultiHeadAttention` of
:numref:`sec_multihead-attention`, and its FFN is the classic two-matrix
MLP with the GELU activation of :numref:`sec_mlp`. Later in this section we
upgrade both ingredients and collect every option in one configurable
class; until then this short block keeps the two arrangements side by side
and nothing else in view.

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
can *measure* how it treats signals. We stack 32 blocks, give every weight
matrix the Xavier initialization of the original transformer
(:numref:`subsec_xavier`), pass a random sequence through the stack, and
record three quantities at every depth: the scale of the residual stream,
how *distinct* the tokens remain from one another, and how much gradient
each block's attention receives from a loss at the top. The numbers change
little from one random draw to the next.
<!-- WRITER: the measurement now has a third arm: `signal_stats(pre_norm, std=None)` draws every weight matrix from N(0, std^2) when std is given; the run cell uses std = 0.02, GPT-2's initialization (sec_gpt). Repeats (5 per tab, dev runs, plus the store) say: "changes little" holds within a factor of about 2 at the deepest post-LN point (spread at k=32 from 8e-5 to 1.7e-4), elsewhere to about 10%. -->


For distinctness we use the cross-token spread
$\|\mathbf{H} - \bar{\mathbf{H}}\| / \|\mathbf{H}\|$, where
$\mathbf{H}$ holds the token vectors of one sequence as rows and
$\bar{\mathbf{H}}$ copies their average to every row. The spread is about
$1$ for unrelated random tokens and $0$ when all tokens have collapsed onto a
single point. The loss takes some care. After a final normalization layer,
the obvious probe $\|\mathbf{H}\|^2$ is constant by construction and has
zero gradient, so we use a generic linear readout
$\langle \mathbf{R}, \mathbf{H} \rangle$ for a fixed random
$\mathbf{R}$ instead.

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

We run it for three stacks: post-LN and pre-LN with Xavier weights, and
post-LN with every weight matrix drawn from $\mathcal{N}(0, 0.02^2)$, the
initialization of GPT-2 that :numref:`sec_gpt` uses. For each we print the
stream RMS at the top, the token spread at three depths, and the gradient
norms of the top block's four attention projections. For the two post-LN
stacks we also print the spread $(1 + \rho_0 (1 + g^4)^k)^{-1/2}$ that the
derivation below predicts, and we plot the spread against depth.
<!-- WRITER: the formula is used in the cell before the prose derives it (rho_0 = 1/63 for n = 64 random tokens; g = 1 for Xavier, g^2 = 256 * 0.02^2 = 0.1024 for the third arm). Either move the three-sentence derivation (structure 3.1 item 3) before this cell, or keep this forward pointer. Measured (dev: 5 repeats per tab plus one notebook run each, and the store; JAX now in full fp32, see report): post-LN Xavier spread at k=8/16/32: 0.27-0.37 / 0.017-0.030 / 0.8e-4-1.7e-4 against predicted 0.44 / 0.031 / 1.2e-4 (the shortfall at k=8 is the FFN); pre-LN stream RMS at 32: 4.9-5.6, spread at 32: 0.33-0.40 (one notebook run 5.64); post-LN std 0.02: spread 0.99 / 0.98 / 0.97 against predicted 0.99 at all three depths, stream RMS 1.00. Gradient norms at k=32: post-LN Xavier W_q, W_k 2e-9 to 9e-9, W_v, W_o 0.06-0.08; pre-LN W_q, W_k 1.5e-3 to 2.8e-3, W_v, W_o 0.075-0.092; post-LN std 0.02 W_q, W_k 4.0e-4 to 4.2e-4, W_v, W_o 6e-3 to 8e-3. -->

```{.python .input #transformer-block-signal-propagation-at-initialization}
%%tab pytorch, jax
arms = {'post-LN': (False, None), 'pre-LN': (True, None),
        'post-LN, std 0.02': (False, 0.02)}
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
d2l.plot(list(range(1, 33)), [s['spread'] for s in stats.values()]
         + list(predicted.values()), 'block index k', 'token spread',
         legend=list(arms) + ['predicted'], yscale='log', ylim=(1e-6, 2),
         fmts=('C0-', 'C1-', 'C2-', 'k:', 'k:'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

The two arrangements produce different networks from the same weights.
Post-LN renormalizes the stream after every addition, so its root mean
square (RMS) is pinned at $1$ at every depth. Pre-LN only ever adds, so its
stream grows like the square root of the depth. At initialization the block
outputs are nearly uncorrelated and their variances add, which predicts an
RMS after 32 blocks of $\sqrt{33} \approx 5.7$. The measured value, between
5 and 5.6, falls somewhat short; an exercise asks why. This growth stabilizes the
<!-- WRITER: measured pre-LN RMS at k=32 over 5 dev repeats per tab plus the store: 4.9 to 5.6 (PyTorch 4.96, 5.12-5.58; JAX 4.94-5.64). -->
<!-- WRITER: the post-LN, std 0.02 arm (new) keeps its tokens: spread 0.97 at k=32 against 0.99 predicted; its stream RMS is pinned at 1.00 like the Xavier post-LN arm. -->

stack. Each pre-LN branch reads the stream through its own normalization,
so its output has a fixed scale. As the stream grows, the *relative*
contribution of each successive block therefore shrinks.

The post-LN stack has no such damping, and the token spread shows the
consequence. At initialization attention is nearly uniform, so every
attention sublayer pulls each token toward the average token. Post-LN then
renormalizes, which locks in the contraction, and the spread decays
*geometrically*: by block 32 the tokens agree to about one part in ten
thousand. Pre-LN dilutes the same contraction by the growing stream, and its
spread falls slowly, to between 0.3 and 0.4 after 32 blocks. This
contraction is related to the *rank collapse* that
:citet:`Dong.Cordonnier.Loukas.2021` prove for stacks of attention layers
without residual connections, where the outputs converge to a single token
with depth.

Collapsed tokens starve attention of gradient. When every token is nearly
the same vector, every query scores every key alike, so the attention
weights, and with them the loss, barely depend on $\mathbf{W}_q$ and
$\mathbf{W}_k$. At block 32 the post-LN query and key projections receive
gradients of order $10^{-9}$, five to six orders of magnitude less than in
the pre-LN stack, while the value and output projections, which act on the
averaged token itself, still receive gradients of several hundredths.
The gradient of each block's query projection, plotted against depth,
shows where the starvation sets in:

```{.python .input #transformer-block-signal-propagation-at-initialization-2}
%%tab pytorch, jax
d2l.plot(list(range(1, 33)), [s['grad_q'] for s in stats.values()],
         'block index k', 'query projection gradient norm',
         legend=list(arms), yscale='log', ylim=(1e-12, 1),
         fmts=('C0-', 'C1-', 'C2-'), figsize=(4.5, 3.3),
         legend_loc='lower left')
```

<!-- WRITER: new cell (TB-9). Measured query-gradient norms (dev, 5 repeats per tab): post-LN Xavier falls smoothly from about 1e-2 at block 1 (6e-3 to 2e-2) to 2e-9 to 7e-9 at block 32, tracking the spread; pre-LN falls from about 6e-2 (5e-2 to 6.5e-2) to about 2e-3, a factor of about 25-30; post-LN std 0.02 is flat at 4.0e-4 to 4.2e-4 at every depth (no collapse; the level is about 5x below pre-LN's top block). -->

These are
measurements at initialization, but they agree with what
:citet:`xiong2020layer` find in training: post-LN needs a learning-rate
warmup to train well, and pre-LN trains without one. We use pre-norm as the
default and compare both placements in training in :numref:`sec_gpt`.

### RMSNorm
:label:`subsec_rmsnorm`

Modern models also simplified the normalizer itself. LayerNorm centers each
token vector, divides it by its standard deviation, and then scales and
shifts it. *RMSNorm* :cite:`Zhang.Sennrich.2019` drops the centering and the
shift and keeps only the part that controls the scale:

$$
\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}.
$$
:eqlabel:`eq_rmsnorm`

It is a one-line function, and both frameworks ship it. We check our
implementation against the built-in, then apply it and LayerNorm, both
with unit gain and zero shift, to $\mathbf{x} = (1, 2, 3, 6)$, to
$3\mathbf{x}$, and to $\mathbf{x} + 10$.
<!-- WRITER: placeholder lead-in for the new cell (structure 3.1 small case b): RMS of x is sqrt(12.5) = 3.54, so RMSNorm gives (0.28, 0.57, 0.85, 1.70); mean 3 and standard deviation sqrt(3.5) = 1.87, so LayerNorm gives (-1.07, -0.53, 0, 1.60). The Benchmark timing and the sec_perf_model reference are gone (TB-12). -->

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

Our implementation agrees with the built-in to rounding error. Scaling the
input leaves both outputs unchanged. Adding a constant to every coordinate
leaves LayerNorm unchanged, because it subtracts the mean first, and changes
RMSNorm, which has no centering step.
<!-- WRITER: placeholder reading of the new output (both tabs print the same rows: x and 3x give RMSNorm 0.28 0.57 0.85 1.70 and LayerNorm -1.07 -0.53 0.00 1.60; x + 10 gives RMSNorm 0.84 0.91 0.99 1.22 and the same LayerNorm row; deviation from the built-in 0.0 in PyTorch, 2.4e-07 in JAX). The old timing paragraph is cut (its "two timings above" no longer exist); the sentences that follow are kept for the writer to rework into the reason (centering did not matter; current models drop it; the time saved is small). -->
RMSNorm is preferred because it does less work at equal quality. :citet:`Zhang.Sennrich.2019` match LayerNorm's accuracy across
several tasks while reducing the running time of their models by 7% to 64%,
and Llama adopted it :cite:`touvron2023llama`.

### Normalizing Queries and Keys
:label:`subsec_qk-norm`

A further normalization sits *inside* attention. Nothing in the block above
controls the size of the attention logits
$\mathbf{q}^\top \mathbf{k} / \sqrt{p}$, where $p$ is the head width. If
training inflates the queries and keys, the softmax saturates, its Jacobian
vanishes (:numref:`sec_attention-scoring-functions`), and training can
spike or stall. The instability was first reported in a 22-billion-parameter
vision transformer, whose attention logits grew until the attention weights
were nearly one-hot :cite:`Dehghani.Djolonga.Mustafa.ea.2023`, and
:citet:`Wortsman.Liu.Xiao.ea.2024` reproduce it in small language models
trained at high learning rates. *QK-norm*
:cite:`Henry.Dachapally.Pawar.ea.2020` applies RMSNorm to the queries and
the keys of each head right before the dot product, so the size of the
logits no longer grows with the query and key weights. It adds two small
norms per layer, and both studies find that it removes the instability.

We can mimic the inflation without training. We draw 64 random tokens of
width $d = 256$ and one head of width $d_h = 64$, multiply the query and key
weights by $\lambda \in \{1, 4, 16\}$, and measure the standard deviation of
the logits and the mean entropy of each query's attention weights, without
and with RMSNorm on the queries and keys. Without it the logits grow like
$\lambda^2$ and the entropy should fall toward 0, the value for one-hot
weights. With it the scale of the weights cancels, and nothing should
change. Uniform weights over 64 keys have entropy $\log 64 = 4.16$.

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
250, and the entropy collapses from about 3.7 to near 0: each query puts
almost all its weight on one key, where the softmax has almost no gradient.
With QK-norm every row is the same, a logit spread near 1 and an entropy of
about 3.7 whatever $\lambda$ is.
<!-- WRITER: new cell (from Ex. 3.3; Ex. 3 part 3 now duplicates it). Measured, both tabs (6 runs each): lambda 1: logit std 0.9-1.1, entropy 3.65-3.73; lambda 4: 15.1-16.8, entropy 0.18-0.31; lambda 16: 240-270, entropy 0.00-0.04; with QK-norm logit std 0.97-1.03 and entropy 3.65-3.72 in every row (rms_norm is scale invariant, so the three rows agree to rounding). The placeholder prediction and reading around the cell are functional; the writer rewrites them. -->

Recent open models also place norms on the other side of a sublayer. OLMo 2
normalizes what each sublayer *writes* before adding it to the stream
:cite:`OLMo.2025`, and Gemma 3 normalizes each branch both before and after
the sublayer :cite:`Gemma.Team.2025`; both also use QK-norm.
:numref:`fig_norm-taxonomy` compares the four placements. Only post-LN
normalizes the stream itself; the signal experiment above compared it with
pre-LN. QK-norm is not a flag of our block. It is an exercise, and it
enters the block through the `attn_factory` hook introduced below.

![Four placements of normalization around one sublayer, with the residual stream running bottom to top; the FFN sublayer is treated the same way. Post-LN normalizes the stream itself after each addition, pre-LN normalizes what the branch reads, OLMo 2 normalizes what the branch writes, and Gemma 3 normalizes the branch on both sides. Only post-LN touches the stream's identity path.](../img/mdl-transformers-norm-taxonomy.svg)
:label:`fig_norm-taxonomy`

## The Feed-Forward Network

Attention moves information between positions but computes no nonlinear
feature of what a single position holds. That computation happens in the
other half of the block, a small MLP applied to each position
independently,

$$
\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\, \phi(\mathbf{W}_1 \mathbf{x}),
$$
:eqlabel:`eq_ffn`

with $\mathbf{W}_1 \in \mathbb{R}^{4d \times d}$ and $\mathbf{W}_2 \in
\mathbb{R}^{d \times 4d}$ in the classic configuration. The expansion by a
factor of 4 dates from the 2017 transformer and is still the common
default. With it, the FFN holds $8d^2$ parameters against the $4d^2$ of
the four attention projections $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v,
\mathbf{W}_o$, so about two thirds of the parameters of a block sit in these
two matrices. :citet:`Geva.Schuster.Berant.ea.2021` read them as a memory with
the structure of attention: each row of $\mathbf{W}_1$ acts as a key that
responds to a pattern in the token's vector, and the matching column of
$\mathbf{W}_2$ is the value that the FFN adds to the stream when its key
responds.

Two changes separate the 2017 FFN from current ones. First, the activation
$\phi$ moved from ReLU to the *GELU* $\phi(z) = z\,\Phi(z)$ of
:numref:`sec_mlp` :cite:`Hendrycks.Gimpel.2016`, where $\Phi$ is the
standard normal cumulative distribution function. GPT used it from the
start :cite:`Radford.Narasimhan.Salimans.ea.2018`. Second,
:citet:`Shazeer.2020` revived an older idea, gating, and found that a
*gated linear unit* with the SiLU activation beat both:

$$
\mathrm{SwiGLU}(\mathbf{x}) = \mathbf{W}_2 \big(\mathrm{SiLU}(\mathbf{W}_g \mathbf{x}) \odot \mathbf{W}_1 \mathbf{x}\big), \qquad \mathrm{SiLU}(z) = z \cdot \mathrm{sigmoid}(z).
$$
:eqlabel:`eq_swiglu`

Instead of passing a linear transform through a fixed nonlinearity, SwiGLU
computes *two* linear views of the input and multiplies them. One view is
squashed into a soft gate. The other passes through unchanged, so wherever
the gate is open it is transmitted at full strength instead of being bent
by the activation. Gates in LSTMs (:numref:`sec_lstm`) use the same
multiplicative interaction. A third matrix means more parameters at the
same width, so fair comparisons shrink the hidden width to
$\tfrac{8}{3} d$. The three matrices of the gated FFN then hold the same
$8d^2$ parameters as the classic two, up to rounding. Both variants fit in
one class:

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

Both tabs use the tanh approximation of GELU, the variant GPT-2 was trained
with, which matters when we load its weights in :numref:`sec_gpt`. We check
the parameter match below, once the block is assembled.
<!-- WRITER: the FFN parameter cell (-the-feed-forward-network-2) is merged into the parameter cell under "Shapes and Parameters" (TB-18), which prints GELU 524,288 against SwiGLU 524,544. "Whether the gate improves the model at equal size ..., and the last experiment of this section answers it" is now false: the trained comparison is in :numref:`subsec_gpt-run-to-run`. -->


## A Configurable Block

We now assemble the block used throughout this chapter. Every design
decision above becomes a constructor argument: `norm` selects LayerNorm or
RMSNorm, `act` selects the FFN, and `pre_norm` selects the arrangement.
Two factory hooks, `attn_factory` and `ffn_factory`, let later sections
swap whole sublayers without touching this class again: a causal attention,
an attention with a smaller key--value cache, or a mixture-of-experts FFN.
The constructor checks its flag strings, so a typo raises an error instead
of silently selecting another architecture. The default configuration
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

### Shapes and Parameters

Whatever the flags, the block preserves the shape of its input. Its
parameter count confirms the accounting above: $4d^2$ in attention, $8d^2$
in the FFN however the activation divides it, and a few normalization
weights. That makes about $12d^2$ per block, two thirds of it in the FFN.
We count the default block, the same block with the GELU FFN, and the 2017
block at $d = 256$ with `d2l.num_params` from :numref:`sec_efficient_cnns`,
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

The GELU and SwiGLU networks differ by 256 parameters, 0.05% of the FFN.
<!-- WRITER: merged parameter cell (TB-18), shape checks on ones dropped. Printed (exact, both tabs): default: attention 262,144, FFN 524,544, total 787,200 = 12.01 d^2; GELU: attention 262,144, FFN 524,288, total 786,944 = 12.01 d^2; 2017 (biases, LayerNorm with shift): attention 263,168, FFN 525,568, total 789,760 = 12.05 d^2. -->
At $d = 256$ the block holds $12.01 d^2$ parameters. The same count scales
to GPT-2 small, whose blocks have width $d = 768$: $12 d^2 \approx 7.1$
million parameters per block, and about 85 million in its 12 blocks. Its
token and position embeddings add another 39 million, which brings the
model to its 124 million parameters.

<!-- WRITER: CharLM, its shorthands (charlm, adamw), the GELU-against-SwiGLU run and subsec_gelu-vs-swiglu are deleted (TB S1, TB-19 to TB-22); the run is redone as the GELU twin of the yardstick in sec_gpt (subsec_gpt-run-to-run). The section ends here; closing (structure 3.1): one sentence, the open question for sec_gpt (whether the placement matters in training, at the weights a language model uses). -->

## Summary

Attention moves information between positions but computes no nonlinear
feature of any one position. A transformer block therefore pairs it with a
position-wise FFN, which holds about two thirds of the parameters of the
block, and both add their outputs to a residual stream. In the original
post-LN arrangement, normalization pins the scale of the stream. At
initialization a 32-block stack then collapses its tokens geometrically,
and the query and key projections of the top block receive gradients five
to six orders of magnitude smaller than in the pre-norm stack. Pre-norm
normalizes what each branch reads, lets the stream grow like the square
root of the depth, and keeps every block trainable. RMSNorm keeps only the
scale statistic of LayerNorm, and QK-norm bounds the attention logits.
SwiGLU replaces the fixed activation with a learned gate; at matched
parameter count it ends a few hundredths of a nat ahead of GELU in our
byte-level model. The `TransformerBlock` class exposes these choices as
options, with replaceable attention and FFN modules.
<!-- WRITER: false since the code step: no GELU-against-SwiGLU run in this section any more ("ends a few hundredths of a nat ahead of GELU in our byte-level model"); the collapse needs its scope: post-LN with GPT-2's N(0, 0.02^2) weights keeps a spread of 0.97 after 32 blocks. -->

## Exercises

1. **Scale invariance.** Let RMSNorm act as in :eqref:`eq_rmsnorm` with
   $\boldsymbol{\gamma} = \mathbf{1}$ and $\epsilon = 0$, and write
   $r(\mathbf{x}) = (\tfrac{1}{d} \sum_i x_i^2)^{1/2}$ for the root mean
   square of $\mathbf{x} \in \mathbb{R}^d$.
    1. Show that $\mathrm{RMSNorm}(\alpha \mathbf{x}) =
       \mathrm{RMSNorm}(\mathbf{x})$ for every $\alpha > 0$, and that
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

1. **Unrolled residual stream.** Number the $2N$ sublayers of an
   $N$-block stack by $j = 1, \ldots, 2N$, write $F_j$ for the $j$-th
   sublayer (attention or FFN) and $\mathrm{Norm}_j$ for its
   normalization, and let $\mathbf{x}_j$ be the residual stream after
   sublayer $j$, so that $\mathbf{x}_0$ is the input.
    1. For the pre-LN arrangement :eqref:`eq_preln-block`, write
       $\mathbf{x}_{2N}$ as $\mathbf{x}_0$ plus a sum of $2N$ branch
       outputs. Write the Jacobian $\partial \mathbf{x}_{2N} / \partial
       \mathbf{x}_0$ as a product over sublayers, expand the product, and
       identify the term that passes a gradient from the output to the
       input unchanged.
    1. For the post-LN arrangement :eqref:`eq_postln-block`, the stream is
       $\mathbf{x}_j = \mathrm{Norm}_j(\mathbf{u}_j)$ with $\mathbf{u}_j =
       \mathbf{x}_{j-1} + F_j(\mathbf{x}_{j-1})$. Model every normalization
       as RMSNorm with unit gain and $\epsilon = 0$. Assume that the input
       $\mathbf{x}_0$ has unit root mean square and that each branch output
       is uncorrelated with the stream it is added to. The product of the
       normalization Jacobians alone takes the place of the identity term
       of the pre-LN expansion. Show that this product scales a vector
       orthogonal to every $\mathbf{u}_j$ by $\prod_j r(\mathbf{u}_j)^{-1}$,
       and write this factor as a function of the mean squares
       $c_1, \ldots, c_{2N}$ of the branch outputs. Evaluate it for a
       common mean square $c_j = c$, and compare its dependence on $N$
       with that of the identity term of the pre-LN expansion.
    <!-- WRITER: CharLM no longer exists; reword part 3 for any pre-LN stack without a final norm. -->
    1. `CharLM` applies a final RMSNorm, `norm`, before its tied output
       head. Describe what the head would receive without it as a pre-LN
       stack grows deeper, and explain why a post-LN stack needs no final
       normalization.

1. [code] **QK-norm as a factory.**
   QK-norm (:numref:`subsec_qk-norm`) applies RMSNorm over the head dimension
   $p = d/h$ to every query and key just before the scores are computed.
    1. Write `QKNormAttention`, a subclass of `d2l.MultiHeadAttention`
       whose `forward` (`__call__` in JAX) passes the per-head queries and
       keys produced by `split_heads` through their own RMSNorm each
       before the dot product. Build a `TransformerBlock` with it through
       `attn_factory` and check the output shape.
    1. Show that with unit gains every scaled logit
       $\hat{\mathbf{q}}^\top \hat{\mathbf{k}} / \sqrt{p}$ lies in
       $[-\sqrt{p}, \sqrt{p}]$, and generalize the bound to learned gains
       $\boldsymbol{\gamma}_q$ and $\boldsymbol{\gamma}_k$. Compute the
       largest possible ratio between two attention weights of one query
       for $p = 64$ and unit gains.
    <!-- WRITER: part 3 is now the body's QK-norm cell (lambda 1, 4, 16); cut it or turn it into a repeat with learned gains (structure 3.1). -->
    1. Training can inflate the query and key weights. Mimic this at
       initialization: multiply the weights of `W_q` and `W_k` of a
       freshly initialized layer with $d = 256$ and $h = 4$ by
       $\lambda \in \{1, 2, 4, 8\}$, and feed it 64 standard normal
       tokens. Measure the standard deviation of the logits and the mean
       entropy of the attention weights, with and without QK-norm. Compare
       the entropies with $\log 64$, and explain the shape of each curve.

1. **FFN width rounding.** `FeedForward` sets the SwiGLU width to
   $\mathrm{round}(8d/3)$.
    1. Write the exact parameter count of the default
       `TransformerBlock(d, h)` as a function of $d$, and bound the
       relative deviation of its FFN from $8d^2$.
    1. The released 7B configuration of Llama :cite:`touvron2023llama` has
       $d = 4096$ and SwiGLU width $11008$, the smallest multiple of 256
       that is at least $\lfloor 8d/3 \rfloor$. Compute the relative excess
       of its FFN parameters over $8d^2$, and the excess that the same
       rule gives at $d = 256$ and at $d = 128$, the default `num_hiddens`
       of `CharLM`. <!-- WRITER: CharLM no longer exists; say d = 128 (the width of sec_moe). --> Explain what the comparison implies for
       parameter-matched experiments with small models.
   :exlabel:`ex_transformer-block-ffn-width-rounding`

1. [code] **Stream variance budget.** ● In the pre-LN arrangement every
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
       the square-root growth stated in :numref:`subsec_signal-propagation`
       holds.

1. [code] **Rate of token collapse.** ● Write the tokens of one sequence
   as $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$ for
   $i = 1, \ldots, n$, where $\boldsymbol{\mu}$ is the mean token and
   $\sum_i \boldsymbol{\delta}_i = \mathbf{0}$. With the ratio of common
   to deviation energy
   $$\rho = \frac{n \|\boldsymbol{\mu}\|^2}{\sum_i \|\boldsymbol{\delta}_i\|^2},$$
   the token spread of `signal_stats` equals $(1 + \rho)^{-1/2}$. At
   initialization the attention weights are nearly uniform, so a post-LN
   attention sublayer adds approximately the same vector
   $\mathbf{W}_o \mathbf{W}_v \boldsymbol{\mu}$ to every token, where
   $\mathbf{W}_v, \mathbf{W}_o \in \mathbb{R}^{d \times d}$ are the value
   and output projections.
    1. Let the entries of $\mathbf{W}_v$ and $\mathbf{W}_o$ be independent
       with mean zero and variance $\alpha^2 / d$, so that $\alpha = 1$ is
       Xavier initialization of a square matrix. Treat the normalization as
       a rescaling common to all tokens. Show that one attention sublayer
       and its normalization multiply $\rho$ by about $1 + \alpha^4$, and
       derive the spread after $k$ blocks for an input of $n$ independent
       random tokens, ignoring the FFN sublayers.
    1. Rerun `signal_stats` for post-LN with all Xavier weights multiplied
       by $\alpha \in \{1, 0.7, 0.5\}$, once with the full `MiniBlock` and
       once with its FFN sublayer removed. Compare both with your formula
       at blocks 8, 16, and 32, and explain the remaining gap of the full
       block.
    1. GPT-1 stacked 12 post-LN blocks of width $d = 768$ and drew every
       weight from $\mathcal{N}(0, 0.02^2)$
       :cite:`Radford.Narasimhan.Salimans.ea.2018`. Compute $\alpha$ for its
       attention projections, and predict the spread after 12 blocks for
       $n = 64$, both for this initialization and for Xavier
       initialization at the same width. Check both predictions with
       `signal_stats`.

<!-- WRITER: exercises "Causal masks" and "Learning-rate sensitivity" moved to gpt.md (code step), rewritten there for d2l.GPT. -->

[Discussions](https://d2l.discourse.group/t/36774)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.1]{.kicker}

The Transformer Block<br>
**residual stream · normalization placement · RMSNorm and QK-norm · GELU and SwiGLU · one configurable class**
:::
:::

::: {.slide title="Attention Alone Averages, and Depth Alone Collapses"}
[The problem]{.kicker}

- Attention outputs weighted averages of values: it moves information
  between positions but computes no new feature of any one token.
- A model needs per-token computation, stacked deep: GPT-2 small has 12
  blocks, large models over 100.
- Stack 32 blocks the 2017 way: at initialization the tokens agree to one
  part in $10^4$, and $\mathbf{W}_q$ gets gradients of order $10^{-9}$.

::: {.d2l-note .warn}
What must a block contain, and where must its normalization sit, so that
a deep stack stays trainable?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count the parameters of a block: $4d^2$ attention, $8d^2$ FFN;
- predict how post-LN and pre-LN stacks treat signals at initialization;
- explain why collapsed tokens starve $\mathbf{W}_q, \mathbf{W}_k$ of
  gradient;
- write RMSNorm, QK-norm, and a SwiGLU FFN at matched parameter count.

**Running example:** a GPT-2-small block, $d = 768$, 12 heads of width 64.
:::

::: {.slide title="The Block Is What Every Transformer Repeats"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **The Transformer Block** | **attention + FFN on a residual stream** |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| Generation and the KV Cache | reuse keys and values while sampling |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** multi-head attention, masks, LayerNorm, GELU, Xavier init.
- **Skips:** positions (next deck), training at scale.
:::
:::
:::

::: {.slide title="Two Sublayers Read from a Residual Stream and Add Back"}
@fig:mdl-transformers-block-anatomy

Attention is the only place tokens interact; the FFN acts on each token
alone. Both map $(n, d)$ to $(n, d)$, so blocks stack.
:::

::: {.slide title="The Pre-Norm Block in Two Lines"}
For token vectors $\mathbf{x} \in \mathbb{R}^d$ on the residual stream:

$$\mathbf{h} = \mathbf{x} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{x})), \qquad \mathbf{y} = \mathbf{h} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{h}))$$

- $\mathrm{Attn}$: multi-head self-attention, four $d \times d$
  projections $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$.
- $\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x})$
  with hidden width $4d$.
- The stream itself is never rewritten, only added to.
:::

::: {.slide title="A GPT-2-Small Block Holds 7.1 Million Parameters"}
[Running example]{.kicker}

| part | count | $d = 768$ |
|---|---|---|
| attention, 4 projections | $4d^2$ | 2.4M |
| FFN, $\mathbf{W}_1$ and $\mathbf{W}_2$ | $8d^2$ | 4.7M |
| block | $12d^2$ | 7.1M |
| 12 blocks | | 85M |
| + token and position embeddings | | 124M |

::: {.d2l-note .rule}
Two thirds of every block sit in the FFN.
:::
:::

::: {.slide title="Check: Double the Width"}
[Check]{.kicker}

Double the width of GPT-2 small to $d = 1{,}536$ and keep its vocabulary
of 50,257 tokens. By what factor does each part grow?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Every block grows $4\times$ (all terms are $\propto d^2$),
to 28M parameters. The embeddings grow only $2\times$ (they are
$\propto d$), from 39M to 79M.
:::
:::
:::

::: {.slide title="Post-LN and Pre-LN Differ in Where the Norm Sits"}
```{.python #transformer-block-exercises-1}
def block(X, pre_norm):
    if pre_norm:                       # norm on each branch's input
        Y = norm1(X)
        X = X + attn(Y, Y, Y)
        return X + ffn(norm2(X))
    X = norm1(X + attn(X, X, X))       # 2017: norm on the stream
    return norm2(X + ffn(X))
```

Same weights, two different functions. Which one keeps a deep stack
trainable can be measured before any training.
:::

::: {.slide title="At Initialization, Post-LN Collapses the Tokens"}
32 blocks, 64 random tokens; Xavier weights as in 2017, or
$\mathcal{N}(0, 0.02^2)$ as in GPT-2. Token spread
$\|\mathbf{H} - \bar{\mathbf{H}}\| / \|\mathbf{H}\|$ is 1 for distinct
tokens, 0 for identical ones:

@!transformer-block-signal-propagation-at-initialization
<!-- WRITER: the output is now nine printed lines plus the spread plot; check the slide fits the frame (trim to the plot or split if it overflows). -->
:::

::: {.slide title="Each Block Multiplies the Common Part by $1 + g^4$"}
- Tokens = mean $\boldsymbol{\mu}$ + deviations $\boldsymbol{\delta}_i$;
  $\rho$ = common energy / deviation energy; spread $= (1+\rho)^{-1/2}$.
- Near-uniform attention adds $\mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ to
  every token. With entries of variance $g^2/d$, one sublayer multiplies
  $\rho$ by $1 + g^4$.
- 64 random tokens: $\rho_0 = 1/63$. Xavier is $g = 1$: $\rho$ doubles.

| block $k$ | 8 | 16 | 32 |
|---|---|---|---|
| predicted | 0.44 | 0.031 | $1.2 \times 10^{-4}$ |
| measured | about 0.3 | about 0.02 | about $10^{-4}$ |

::: {.d2l-note}
The FFN, left out of the formula, pulls a little harder.
:::
<!-- WRITER: measured row from dev repeats (both tabs): 0.27-0.37 / 0.017-0.030 / 0.8e-4-1.7e-4. -->
:::

::: {.slide title="Small Weights Do Not Collapse"}
Same post-LN stack, every weight from $\mathcal{N}(0, 0.02^2)$, as in GPT-2,
at $d = 256$:

- $g^2 = 256 \cdot 0.02^2 = 0.10$, so $1 + g^4 \approx 1.01$ per block.
- Predicted spread after 32 blocks: 0.99; measured 0.97.
- Query gradients stay near $4 \times 10^{-4}$ at every depth.

::: {.d2l-note .rule}
Post-norm is not doomed: it is sensitive to the size of its weights.
:::
:::

::: {.slide title="Identical Tokens Starve the Query and Key Weights"}
- If all tokens are equal, every query scores every key alike: the
  attention weights no longer depend on $\mathbf{W}_q, \mathbf{W}_k$.
- Top post-LN block: $\mathbf{W}_q, \mathbf{W}_k$ gradients
  $\sim 10^{-9}$, five to six orders below pre-LN.
- $\mathbf{W}_v, \mathbf{W}_o$ act on the mean token itself and still
  get gradients near 0.06.

::: {.d2l-note .rule}
In training, post-LN needs learning-rate warmup; pre-LN trains without it.
:::

[Xiong et al., 2020]{.small}
:::

::: {.slide title="Pre-LN Lets the Stream Grow Like the Square Root of Depth"}
- Each branch reads a normalized input, so its output has a fixed scale.
- Outputs of different blocks are nearly uncorrelated: variances add.
- After 32 blocks the stream RMS is about 5, close to
  $\sqrt{33} \approx 5.7$; post-LN pins it at 1.

::: {.d2l-note .rule}
A growing stream damps the relative contribution of each new block, so
deep stacks stay trainable. Pre-norm is the default.
:::
:::

::: {.slide title="Check: Which Weights Still Learn?"}
[Check]{.kicker}

In the collapsed post-LN stack, rank the gradients of $\mathbf{W}_q$,
$\mathbf{W}_v$, and the FFN's $\mathbf{W}_1$ at the top block, and say
why.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\mathbf{W}_v$ and $\mathbf{W}_1$ act on the token vector
itself, so they keep ordinary gradients. $\mathbf{W}_q$ enters only
through the attention weights, which identical tokens make constant:
its gradient vanishes.
:::
:::
:::

::: {.slide title="RMSNorm Keeps Only the Scale Statistic"}
$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

- LayerNorm subtracts the mean, divides by the standard deviation, then
  scales and shifts; RMSNorm drops the mean and the shift.
- One statistic instead of two, same quality; reported running time
  7–64% lower. Llama adopted it.
- When memory traffic dominates, both normalizers cost about the same.

[Zhang & Sennrich, 2019]{.small}
:::

::: {.slide title="RMSNorm Ignores Scale, LayerNorm Also Ignores Shift"}
$\mathbf{x} = (1, 2, 3, 6)$: RMS $\sqrt{12.5} = 3.54$; mean 3, standard
deviation $\sqrt{3.5} = 1.87$. Unit gain, zero shift:

@!transformer-block-rmsnorm

::: {.d2l-note}
$3\mathbf{x}$ changes neither; $\mathbf{x} + 10$ changes only RMSNorm.
:::
:::

::: {.slide title="QK-Norm Bounds the Attention Logits"}
- Logits $\mathbf{q}^\top\mathbf{k}/\sqrt{p}$, head width $p$: if
  training inflates $\mathbf{q}$ and $\mathbf{k}$, softmax saturates and
  its gradient vanishes.
- Seen in ViT-22B; reproduced in small LMs at high learning rates.
- QK-norm: RMSNorm on each head's queries and keys before the dot
  product. With unit gains $|\textrm{logit}| \le \sqrt{p} = 8$ for a
  GPT-2-small head.

[Dehghani et al., 2023; Wortsman et al., 2024]{.small}
:::

::: {.slide title="QK-Norm Keeps the Entropy Up When Weights Grow"}
64 random tokens, one head of width 64; $\mathbf{W}_q, \mathbf{W}_k$
scaled by $\lambda$; entropy of uniform weights $\log 64 = 4.16$:

@!transformer-block-normalizing-queries-and-keys

::: {.d2l-note}
Without the norm, logits grow like $\lambda^2$ and attention turns
one-hot; with it, nothing changes.
:::
:::

::: {.slide title="Check: Twelve Blocks of GPT-1"}
[Check]{.kicker}

A post-LN stack of 12 blocks at $d = 768$, 64 random tokens. Which
initialization collapses the tokens: Xavier, or $\mathcal{N}(0, 0.02^2)$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Xavier: $g = 1$, $\rho = 2^{12}/63 \approx 65$, spread
$\approx 0.12$ (measured 0.09). $\mathcal{N}(0, 0.02^2)$: $g^2 = 0.31$,
$\rho = 1.094^{12}/63 \approx 0.047$, spread $\approx 0.98$ (measured
0.88). Only Xavier collapses.
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
<!-- WRITER: body still says "soft gate"/"full strength" (TB-15): SiLU is unbounded above; the product is a multiplicative feature. -->
$$\mathrm{SwiGLU}(\mathbf{x}) = \mathbf{W}_2\big(\mathrm{SiLU}(\mathbf{W}_g\mathbf{x}) \odot \mathbf{W}_1\mathbf{x}\big), \quad \mathrm{SiLU}(z) = z\,\mathrm{sigmoid}(z)$$

- One view becomes a soft gate; the other passes at full strength where
  the gate is open.
- Three matrices: shrink the width from $4d$ to $\tfrac{8}{3}d$ to keep
  $8d^2$. GPT-2 small: width $2{,}048$ instead of $3{,}072$.

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

At $d = 256$: 524,288 parameters for GELU, 524,544 for SwiGLU.
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

::: {.slide title="One Constructor Covers 2017 and Today"}
| argument | choices | default |
|---|---|---|
| `norm` | `'rms'`, `'layer'` | `'rms'` |
| `act` | `'swiglu'`, `'gelu'` | `'swiglu'` |
| `pre_norm` | `True`, `False` | `True` |
| `attn_factory` | causal attention, smaller KV cache | multi-head |
| `ffn_factory` | mixture of experts | `FeedForward` |

The default is the Llama configuration (2023); `norm='layer'`,
`act='gelu'`, `pre_norm=False` gives the 2017 block.
:::

::: {.slide title="Wrap-up"}
- Attention moves information between tokens; the FFN computes per
  token and holds two thirds of the $12d^2$ per block (7.1M in GPT-2
  small).
- Post-LN at initialization: tokens collapse to $10^{-4}$ spread by block
  32, $\mathbf{W}_q$ gradients $\sim 10^{-9}$. Pre-LN: stream RMS grows
  like $\sqrt{\textrm{depth}}$, every block trainable.
- RMSNorm keeps the scale statistic; QK-norm bounds the logits.
- SwiGLU at width $\tfrac{8}{3}d$ beats the GELU MLP at equal size.
<!-- WRITER: no longer measured in this deck (the run moved to sec_gpt); restate as the matched count (524,544 against 524,288 at d = 256) or the small-weights result. -->

**Next:** embeddings, positions, and a causal mask turn the block into a
GPT.
:::
