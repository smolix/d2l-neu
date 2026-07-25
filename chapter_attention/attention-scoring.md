# Attention Scoring and Masking
:label:`sec_attention-scoring-functions`

In :numref:`sec_attention-pooling` we computed attention weights from
distance-based kernels such as the Gaussian. Distances are slightly more
expensive to compute than dot products, and once the softmax
:eqref:`eq_softmax_attention` guarantees nonnegative normalized weights, we
are free to pick any *scoring function* $a(\mathbf{q}, \mathbf{k})$ we like.
This section settles the choice used by essentially every modern
architecture—the scaled dot product—and builds the two pieces of machinery
that make it practical: a masked softmax that lets one batched computation
serve sequences of different lengths (and autoregressive models that must
not look ahead), and batched matrix multiplication. We close with the story
of where these ideas came from: the machine translation problem that turned
"learning to align" into the attention mechanism.

```{.python .input #attention-scoring-attention-scoring-and-masking}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #attention-scoring-attention-scoring-and-masking}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
```

## Dot-Product Attention

Let's review the attention function (without exponentiation) arising from
the Gaussian kernel:

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$

First, the final term depends on $\mathbf{q}$ only, so it is identical for
all keys, and the softmax normalization :eqref:`eq_softmax_attention`
removes it entirely. Second, if the key norms $\|\mathbf{k}_i\|$ are all
equal, the middle term drops out the same way, and the Gaussian kernel and
the dot product induce identical attention weights. In general the norms are
not equal, and dropping the term is a modeling decision rather than an
approximation: we adopt the dot product $\mathbf{q}^\top \mathbf{k}_i$ as a
compatibility function in its own right — one that learned query and key
representations can shape freely — with the Gaussian expansion surviving as
the exact special case of equal key norms.

One adjustment is still needed, to keep the magnitude of the scores under
control. Assume that all elements of the query $\mathbf{q} \in \mathbb{R}^d$
and the key $\mathbf{k}_i \in \mathbb{R}^d$ are independent random variables
with zero mean and unit variance. The dot product of the two vectors then
has zero mean and variance $d$: the typical score grows like $\sqrt{d}$ for
no reason other than vector length. Rescaling by $1/\sqrt{d}$ keeps the
score variance at $1$ regardless of dimension, and yields the *scaled
dot-product attention* scoring function of the Transformer
:cite:`Vaswani.Shazeer.Parmar.ea.2017`:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

The attention weights are obtained, as always, with the softmax:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

### Softmax Saturation and the $1/\sqrt{d}$ Factor

Why insist on unit score variance? The softmax saturates: once one score
exceeds the others by a large margin, the winning weight approaches $1$, the
rest approach $0$, and, since the Jacobian of the softmax is

$$
\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha} \boldsymbol{\alpha}^\top,
$$

which tends to the zero matrix as $\boldsymbol{\alpha}$ approaches a one-hot
vector, the gradient returning along the *score* path shrinks with it, and the
queries and keys behind those scores stop being updated. (Only that query–key
route saturates; gradients still flow through the values, the output
projection, and any residual connection.) For finite scores this Jacobian is
never the zero matrix — as we noted in :numref:`sec_queries-keys-values`, it
always keeps the all-ones vector in its null space and nothing else — but it
can come arbitrarily close. Let's measure both effects rather than take them
on faith. We draw random queries
and keys with unit-variance entries, compute attention over $64$ candidate
keys, and record the entropy of the resulting weight distribution as the
dimension $d$ grows, with and without the $1/\sqrt{d}$ factor.

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-1}
%%tab pytorch
def attention_stats(d, num_keys=64, num_queries=256, scaled=True):
    q = torch.randn(num_queries, d)
    keys = torch.randn(num_queries, num_keys, d)
    scores = (keys @ q[..., None]).squeeze(-1)
    if scaled:
        scores = scores / math.sqrt(d)
    alpha = F.softmax(scores, dim=-1)
    entropy = -(alpha * torch.log(alpha + 1e-12)).sum(-1).mean()
    # Frobenius norm of the softmax Jacobian diag(alpha) - alpha alpha^T
    jac = torch.diag_embed(alpha) - alpha[..., :, None] * alpha[..., None, :]
    return entropy.item(), jac.square().sum((-2, -1)).sqrt().mean().item()

torch.manual_seed(0)
ds = [2**k for k in range(2, 10)]
entropies = [[attention_stats(d, scaled=s)[0] for d in ds]
             for s in (False, True)]
d2l.plot(ds, entropies, 'dimension d', 'entropy (nats)',
         legend=['unscaled', 'scaled by 1/sqrt(d)'], xscale='log')
```

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-1}
%%tab jax
def attention_stats(d, num_keys=64, num_queries=256, scaled=True, seed=0):
    key_q, key_k = jax.random.split(jax.random.key(seed))
    q = jax.random.normal(key_q, (num_queries, d))
    keys = jax.random.normal(key_k, (num_queries, num_keys, d))
    scores = (keys @ q[..., None]).squeeze(-1)
    if scaled:
        scores = scores / math.sqrt(d)
    alpha = jax.nn.softmax(scores, axis=-1)
    entropy = -(alpha * jnp.log(alpha + 1e-12)).sum(-1).mean()
    # Frobenius norm of the softmax Jacobian diag(alpha) - alpha alpha^T
    jac = (alpha[..., None] * jnp.eye(num_keys)
           - alpha[..., :, None] * alpha[..., None, :])
    return float(entropy), float(jnp.sqrt((jac**2).sum((-2, -1))).mean())

ds = [2**k for k in range(2, 10)]
entropies = [[attention_stats(d, scaled=s)[0] for d in ds]
             for s in (False, True)]
d2l.plot(ds, entropies, 'dimension d', 'entropy (nats)',
         legend=['unscaled', 'scaled by 1/sqrt(d)'], xscale='log')
```

A uniform distribution over $64$ keys has entropy $\ln 64 \approx 4.2$ nats.
The scaled scores hold their entropy essentially constant, at about $3.7$
nats, across two orders of magnitude in $d$: the weight distribution keeps
the same moderate sharpness no matter how wide the vectors are. Without
scaling, the entropy collapses as $d$ grows—below $0.2$ nats by $d = 512$,
a near-deterministic weighting that concentrates almost all its mass on a
single key purely because the vectors are long. The gradient tells the same story:

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2}
%%tab pytorch
torch.manual_seed(0)
for d in (4, 64, 512):
    _, jac_unscaled = attention_stats(d, scaled=False)
    _, jac_scaled = attention_stats(d, scaled=True)
    print(f'd = {d:3d}: Jacobian norm {jac_unscaled:.3f} (unscaled), '
          f'{jac_scaled:.3f} (scaled)')
```

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2}
%%tab jax
for d in (4, 64, 512):
    _, jac_unscaled = attention_stats(d, scaled=False)
    _, jac_scaled = attention_stats(d, scaled=True)
    print(f'd = {d:3d}: Jacobian norm {jac_unscaled:.3f} (unscaled), '
          f'{jac_scaled:.3f} (scaled)')
```

With scaling, the Jacobian norm is the same at every dimension. Without it,
the norm decays as the softmax saturates—by $d = 512$ it has fallen to
about half the scaled value and is still shrinking. A model built on
unscaled scores at realistic dimensions would start its life with
near-one-hot attention and barely any gradient with which to fix it. One
division by $\sqrt{d}$ removes the problem, which is why it is part of the
definition :eqref:`eq_dot_product_attention` rather than a tuning trick.

## Masking

Attention as defined so far attends to *every* key. In practice we routinely
need it not to, for one of two reasons. The first is *padding*: sequences of
different lengths end up in the same minibatch, padded with dummy tokens to
a common length (here shown as `<blank>`):

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

Padding tokens carry no meaning, and no query should waste weight on them.
The second reason is *causality*: a language model trained to predict the
next token computes outputs for all positions of a sequence in parallel, and
the query at position $t$ must not see keys at positions beyond $t$—
otherwise the model can copy the very future it is being trained to predict.
Both cases call for the same operation: restrict the attention sum
$\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ to a valid
prefix $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ with
$l \leq n$, where $l$ depends on the sequence (padding) or on the query
position (causality).

### The Masked Softmax Operation

The operation is common enough to have a name, the *masked softmax*, and a
standard implementation trick. Rather than branching on which keys are
valid—conditionals are poison for the heavily optimized batched kernels
that attention runs on—we overwrite the scores of the invalid positions
with a very negative number before the softmax, so that their weights come
out as zero.

The choice of "very negative" deserves care. Older codebases used a literal
constant such as $-10^{6}$. In single precision that works; in the half
precisions that modern training runs in, it does not. The float16 format
tops out near $6.5 \times 10^4$, so $-10^6$ silently overflows, and in
bfloat16 a merely-large constant may fail to fully suppress a weight once
genuine scores are large themselves. Writing literal $-\infty$ masks
exactly, but if every key of some query is masked the softmax returns NaN
and poisons the training run. The dtype-safe idiom, which we adopt, masks
with the most negative *finite* value of the score's dtype
(`torch.finfo(X.dtype).min` and `jnp.finfo(X.dtype).min`, respectively): the
masked weights are exactly zero at any precision, with no NaN. A fully masked
query — one with no valid key — would otherwise come out as a uniform average
over its *invalid* values, still garbage, so `masked_softmax` zeroes any such
row; callers should nonetheless guarantee that every query keeps at least one
valid key.

```{.python .input #attention-scoring-the-masked-softmax-operation-1}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    mask = torch.arange(shape[-1], device=X.device)[None, :]
    mask = mask < valid_lens[:, None]
    # Most negative finite score: exactly zero weight after the softmax,
    # at any precision, without the NaN risk of literal -inf
    X = X.reshape(-1, shape[-1]).masked_fill(~mask, torch.finfo(X.dtype).min)
    weights = F.softmax(X, dim=-1)
    # A fully masked query (no valid key) would be a uniform average over
    # invalid values; zero those rows so no padded position leaks through
    weights = torch.where(mask.any(-1, keepdim=True), weights, 0.0)
    return weights.reshape(shape)
```

```{.python .input #attention-scoring-the-masked-softmax-operation-1}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return jax.nn.softmax(X, axis=-1)
    shape = X.shape
    if valid_lens.ndim == 1:
        valid_lens = jnp.repeat(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    mask = jnp.arange(shape[-1])[None, :] < valid_lens[:, None]
    # Most negative finite score: exactly zero weight after the softmax,
    # at any precision, without the NaN risk of literal -inf
    X = jnp.where(mask, X.reshape(-1, shape[-1]), jnp.finfo(X.dtype).min)
    weights = jax.nn.softmax(X, axis=-1)
    # A fully masked query (no valid key) would be a uniform average over
    # invalid values; zero those rows so no padded position leaks through
    weights = jnp.where(mask.any(-1, keepdims=True), weights, 0.0)
    return weights.reshape(shape)
```

To illustrate how this function works, consider a minibatch of two examples
with two queries and four keys each, where the valid lengths are $2$ and
$3$, respectively. All weights beyond the valid length come out as zero,
and each row still sums to $1$:

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab jax
masked_softmax(jax.random.uniform(jax.random.key(0), (2, 2, 4)),
               jnp.array([2, 3]))
```

For finer control we can pass a two-dimensional tensor of valid lengths,
one per query:

```{.python .input #attention-scoring-the-masked-softmax-operation-3}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
```

```{.python .input #attention-scoring-the-masked-softmax-operation-3}
%%tab jax
masked_softmax(jax.random.uniform(jax.random.key(1), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### Causal Masking

Per-query valid lengths are exactly what causality needs: for a sequence of
length $n$, the query at position $t$ may attend to keys $1, \ldots, t$, so
the valid lengths are simply $(1, 2, \ldots, n)$, shared by every sequence
in the batch. The resulting attention pattern is lower triangular:

```{.python .input #attention-scoring-causal-masking}
%%tab pytorch
torch.manual_seed(0)
scores = torch.randn(1, 6, 6)
causal_lens = torch.arange(1, 7)[None, :]  # query t sees keys 1..t
d2l.show_heatmaps(masked_softmax(scores, causal_lens)[None],
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input #attention-scoring-causal-masking}
%%tab jax
scores = jax.random.normal(jax.random.key(2), (1, 6, 6))
causal_lens = jnp.arange(1, 7)[None, :]  # query t sees keys 1..t
d2l.show_heatmaps(masked_softmax(scores, causal_lens)[None],
                  xlabel='Keys', ylabel='Queries')
```

On the attention side, this triangular mask is the key difference between a
model that merely reads a sequence and one that can be trained, in parallel
over all positions, to generate it — generation also needs the shifted
next-token objective and a decoding loop. The mask will accompany us through
every decoder in the chapters ahead.

### Composing Masks

`valid_lens` describes prefixes, which cover the two cases above, but the
general interface is a boolean tensor: entry $(i, j)$ says whether query $i$
may attend to key $j$. Every requirement takes this form — padding excludes
keys beyond the sequence length, causality excludes keys beyond the query,
and structural patterns such as the attention windows of
:numref:`sec_attention-at-scale` exclude by distance — and a key must
survive *all* requirements at once, so masks compose by logical AND.
Broadcasting keeps the bookkeeping cheap: a padding mask has shape
$(\textrm{batch}, 1, \textrm{keys})$, a causal mask
$(1, \textrm{queries}, \textrm{keys})$, and their AND broadcasts to the full
$(\textrm{batch}, \textrm{queries}, \textrm{keys})$ without materializing
either input per example. Applying the composite is the same idiom as
before: overwrite the excluded scores with the dtype's most negative finite
value, then softmax.

```{.python .input #attention-scoring-composing-masks}
%%tab pytorch
valid_lens, n = torch.tensor([6, 3]), 6
j = torch.arange(n)
padding = (j[None, :] < valid_lens[:, None])[:, None, :]  # (batch, 1, key)
causal = (j[None, :] <= j[:, None])[None, :, :]         # (1, query, key)
mask = padding & causal                                 # (batch, query, key)
torch.manual_seed(0)
scores = torch.randn(2, n, n)
weights = F.softmax(
    scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=-1)
d2l.show_heatmaps(weights[:, None], xlabel='Keys', ylabel='Queries')
```

```{.python .input #attention-scoring-composing-masks}
%%tab jax
valid_lens, n = jnp.array([6, 3]), 6
j = jnp.arange(n)
padding = (j[None, :] < valid_lens[:, None])[:, None, :]  # (batch, 1, key)
causal = (j[None, :] <= j[:, None])[None, :, :]         # (1, query, key)
mask = padding & causal                                 # (batch, query, key)
scores = jax.random.normal(jax.random.key(0), (2, n, n))
weights = jax.nn.softmax(
    jnp.where(mask, scores, jnp.finfo(scores.dtype).min), axis=-1)
d2l.show_heatmaps(weights[:, None], xlabel='Keys', ylabel='Queries')
```

The first sequence shows the plain causal triangle; the second is cut off
at its valid length of $3$, the intersection of both constraints. Composition
sharpens the fully-masked hazard flagged above: masks that are harmless
alone can leave some query with an empty intersection, so the guarantee of
at least one valid key per query must hold for the *composite*. The same
machinery handles *packed sequences* — several documents concatenated into
one training row — by ANDing the causal mask with a block-diagonal mask
that keeps each document from attending into its neighbors.

## Batched Attention

### Batch Matrix Multiplication
:label:`subsec_batch_dot`

Attention is computed on minibatches of queries, keys, and values, so we
need to multiply batches of matrices by one another. Assume that

$$
\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \qquad
\mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}.
$$

Then the batch matrix multiplication (BMM) computes one matrix product per
batch element,

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$
:eqlabel:`eq_batch-matrix-mul`

Let's see this in action in a deep learning framework:

```{.python .input #attention-scoring-batch-matrix-multiplication}
%%tab pytorch
Q = torch.ones((2, 3, 4))
K = torch.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input #attention-scoring-batch-matrix-multiplication}
%%tab jax
Q = jnp.ones((2, 3, 4))
K = jnp.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

### The DotProductAttention Class

Now we can state scaled dot-product attention in the form in which it is
actually computed. For $n$ queries and $m$ key--value pairs, with queries
and keys of length $d$ and values of length $v$, stack the queries into
$\mathbf{Q} \in \mathbb{R}^{n \times d}$, the keys into $\mathbf{K} \in
\mathbb{R}^{m \times d}$, and the values into $\mathbf{V} \in
\mathbb{R}^{m \times v}$. Then the entire attention computation is two
matrix products and a softmax:

$$ \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top }{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n \times v}.$$
:eqlabel:`eq_softmax_QK_V`

Requiring queries and keys to share the length $d$ is no real restriction:
a learned matrix $\mathbf{M}$ turns $\mathbf{q}^\top \mathbf{k}$ into
$\mathbf{q}^\top \mathbf{M} \mathbf{k}$ and translates between spaces of
different dimension (an exercise below). Applied to a minibatch,
:eqref:`eq_softmax_QK_V` uses the batch matrix multiplication of
:eqref:`eq_batch-matrix-mul` twice. The implementation applies dropout to
the attention weights as regularization, and stores the weights for
visualization:

```{.python .input #attention-scoring-the-dotproductattention-class-1}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input #attention-scoring-the-dotproductattention-class-1}
%%tab jax
class DotProductAttention(nnx.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout, rngs=None):
        rngs = nnx.Rngs(dropout=0) if rngs is None else rngs
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def __call__(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.swapaxes(1, 2)
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        # NNX idiom: return (output, weights); PyTorch stores weights on self
        return self.dropout(attention_weights) @ values, attention_weights
```

To see the class at work we use a minibatch of size $2$, with $10$ keys and
values of dimension $4$, a single $2$-dimensional query per example, and
valid lengths of $2$ and $6$. The output is one $4$-dimensional row per
query:

```{.python .input #attention-scoring-the-dotproductattention-class-2}
%%tab pytorch
torch.manual_seed(0)
queries = torch.randn(2, 1, 2)
keys = torch.randn(2, 10, 2)
values = torch.randn(2, 10, 4)
valid_lens = torch.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input #attention-scoring-the-dotproductattention-class-2}
%%tab jax
queries = jax.random.normal(jax.random.key(0), (2, 1, 2))
keys = jax.random.normal(jax.random.key(1), (2, 10, 2))
values = jax.random.normal(jax.random.key(2), (2, 10, 4))
valid_lens = jnp.array([2, 6])

attention = DotProductAttention(dropout=0.5)
output, attention_weights = nnx.view(
    attention, deterministic=True)(queries, keys, values, valid_lens)
d2l.check_shape(output, (2, 1, 4))
```

The stored attention weights confirm that the mask did its job: weights
vanish beyond the second and sixth key, respectively.

```{.python .input #attention-scoring-the-dotproductattention-class-3}
%%tab pytorch
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input #attention-scoring-the-dotproductattention-class-3}
%%tab jax
d2l.show_heatmaps(attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## From Alignment to Attention

Where did all of this come from? Not from databases, but from machine
translation. Around 2014, the leading neural approach encoded a source
sentence with an RNN into a single fixed-size state vector and decoded the
translation from that vector with a second RNN (we build such
encoder--decoder models in full in :numref:`sec_seq2seq`). The design has
the flaw this chapter opened with: one fixed-size vector must carry an
entire sentence, and for long sentences it cannot. Translation quality
degraded visibly with sentence length. :citet:`Graves.2013` had faced a
version of this problem when generating handwriting from text, and solved
it with a differentiable model that *aligned* each output pen stroke with a
position in the source text—though with the constraint that the alignment
could only move forward, an assumption borrowed from decoding in speech
recognition :cite:`rabiner1993fundamentals`.

:citet:`Bahdanau.Cho.Bengio.2014` removed the constraint. Their translation
model kept the two RNNs but gave the decoder a new capability at every
step: use the current decoder state as a *query* against all encoder
states, which serve as keys and values, and feed the resulting weighted
summary—a fresh one per output token—into the next prediction. The paper's
title called the idea "jointly learning to align and translate", and the
learned weights behaved exactly like the soft alignments of classical
statistical translation: mostly monotone along the diagonal, with clean
departures where the two languages order words differently, as sketched in
:numref:`fig_alignment`. Nothing forced the model to align; the behavior
emerged from training. This is the attention mechanism of
:eqref:`eq_attention_pooling` in its original habitat, and its impact
reached far beyond translation.

![Soft alignment between an English sentence and its French translation, in the style of a learned attention map (darker cells indicate larger weight; schematic). The alignment is mostly monotone, but "black cat" maps to "chat noir" with the order reversed, and both "était" and "assis" draw on "sat".](../img/mdl-attention-alignment.svg)
:label:`fig_alignment`

One detail differed from the scoring function we settled on above. The
decoder state and the encoder states were vectors of different sizes, so
instead of a dot product, :citet:`Bahdanau.Cho.Bengio.2014` scored with a
small one-hidden-layer MLP, now known as *additive attention*:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

where $\mathbf{W}_q \in \mathbb{R}^{h \times q}$, $\mathbf{W}_k \in
\mathbb{R}^{h \times k}$, and $\mathbf{w}_v \in \mathbb{R}^{h}$ are learned.
The two projections embed queries and keys into a shared $h$-dimensional
space, and $\mathbf{w}_v$ reads a score off the sum. It takes only a few
lines to compute—here scoring three $20$-dimensional queries against six
$2$-dimensional keys, dimensions no dot product could pair up:

```{.python .input #attention-scoring-from-alignment-to-attention}
%%tab pytorch
torch.manual_seed(0)
queries, keys = torch.randn(3, 20), torch.randn(6, 2)
num_hiddens = 8
W_q = torch.randn(num_hiddens, 20) / math.sqrt(20)
W_k = torch.randn(num_hiddens, 2) / math.sqrt(2)
w_v = torch.randn(num_hiddens) / math.sqrt(num_hiddens)
features = torch.tanh((queries @ W_q.T)[:, None, :]
                      + (keys @ W_k.T)[None, :, :])
scores = features @ w_v
d2l.check_shape(scores, (3, 6))
F.softmax(scores, dim=-1)
```

```{.python .input #attention-scoring-from-alignment-to-attention}
%%tab jax
key_q, key_k, k1, k2, k3 = jax.random.split(jax.random.key(0), 5)
queries = jax.random.normal(key_q, (3, 20))
keys = jax.random.normal(key_k, (6, 2))
num_hiddens = 8
W_q = jax.random.normal(k1, (num_hiddens, 20)) / math.sqrt(20)
W_k = jax.random.normal(k2, (num_hiddens, 2)) / math.sqrt(2)
w_v = jax.random.normal(k3, (num_hiddens,)) / math.sqrt(num_hiddens)
features = jnp.tanh((queries @ W_q.T)[:, None, :]
                    + (keys @ W_k.T)[None, :, :])
scores = features @ w_v
d2l.check_shape(scores, (3, 6))
jax.nn.softmax(scores, axis=-1)
```

Additive attention held its own for a few years, but the outcome was
decided by hardware: a dot product between all queries and all keys is a
single matrix multiplication, the one operation accelerators are built
around, while the additive score requires materializing an
$n \times m \times h$ tensor of hidden activations. When a learned metric
is wanted, projecting queries and keys with learned matrices *before* a dot
product achieves it at matmul speed—which is precisely the form attention
takes inside the Transformer, whose authors then discarded the RNN
scaffolding altogether and kept attention as the only mechanism relating
sequence positions :cite:`Vaswani.Shazeer.Parmar.ea.2017`. The next
sections follow that road: first attention with multiple heads, then what
replaces the RNN's notion of position.

## Summary

Softmax normalization turns any scoring function into valid attention
weights, and the scoring function of choice is the scaled dot product
:eqref:`eq_dot_product_attention`: it is a single batched matrix
multiplication, and the $1/\sqrt{d}$ factor holds the score variance at $1$
so that the softmax neither saturates nor starves its own gradient as the
dimension grows—an effect we measured directly. Masking makes the same
batched computation respect variable sequence lengths and causal
structure: overwrite invalid scores with the most negative finite value of
the dtype (not a hard-coded constant, which breaks in half precision)
before the softmax; arbitrary boolean requirements — padding, causality,
structure — compose into one mask by logical AND. `DotProductAttention` packages scoring, masking,
dropout on the weights, and value pooling in a dozen lines that the rest of
this book reuses. Additive attention, the original scoring function of the
translation models that started the field, survives as history and as a
reminder that attention weights are learned soft alignments.

## Exercises

1. Implement distance-based attention by modifying the
   `DotProductAttention` code. You only need the squared norms of the keys
   $\|\mathbf{k}_i\|^2$ for an efficient implementation.
1. Modify dot-product attention to allow for queries and keys of different
   dimensionalities by employing a matrix $\mathbf{M}$ to adjust dimensions,
   scoring with $\mathbf{q}^\top \mathbf{M} \mathbf{k}$.
1. How does the computational cost of :eqref:`eq_softmax_QK_V` scale with
   the dimensionality of keys, queries, and values, and with their number?
   What about the memory bandwidth requirements?
1. Derive the softmax Jacobian $\mathrm{diag}(\boldsymbol{\alpha}) -
   \boldsymbol{\alpha}\boldsymbol{\alpha}^\top$ and verify it numerically
   against automatic differentiation on a random score vector. Compute its
   Frobenius norm when $\boldsymbol{\alpha}$ is one-hot and when it is
   uniform over $m$ keys. Which regime does the saturation experiment
   approach as $d$ grows without scaling?
1. What does `masked_softmax` return for a query whose valid length is $0$?
   Compare the behavior of masking with the most negative finite value
   against masking with literal $-\infty$. In what situations can a fully
   masked query arise in practice, and what would you do about it?
1. Count the parameters and floating-point operations needed to score $n$
   queries against $m$ keys with additive attention (hidden size $h$) and
   with scaled dot-product attention (shared dimension $d$). Implement a
   batched version of the additive score and time both variants for
   $d = h = 64$ and $d = h = 256$. Which is faster on your hardware, and
   why?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.2]{.kicker}

Attention scoring and masking<br>
**dot-product scores · the 1/√d factor, measured · masked softmax · batched attention · from alignment to attention**
:::
:::

::: {.slide title="From kernels to dot products"}
Expand the Gaussian kernel's exponent:

$$-\tfrac{1}{2}\|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i - \tfrac{1}{2}\|\mathbf{k}_i\|^2 - \tfrac{1}{2}\|\mathbf{q}\|^2.$$

The query term cancels in the softmax; the key-norm term cancels too when
all key norms are equal. Keeping only the dot product is a modeling choice:
a compatibility score that learned representations can shape freely.

. . .

For unit-variance entries, $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = d$
— scores grow with vector length for no semantic reason. Hence the
Transformer's scoring function:

$$a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:::

::: {.slide title="Softmax saturation, measured"}
Random queries and keys, attention over 64 candidates, entropy of the
weight distribution as $d$ grows:

@!attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-1

- Uniform over 64 keys ≈ 4.2 nats; scaled scores hold ≈ 3.7 nats at every $d$.
- Unscaled: below 0.2 nats by $d = 512$ — a near-deterministic weighting,
  from vector length alone.
:::

::: {.slide title="Saturation kills the gradient"}
The softmax Jacobian
$\partial \boldsymbol{\alpha} / \partial \mathbf{a} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top$
tends to zero as $\boldsymbol{\alpha}$ approaches one-hot:

@attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2

- Scaled: constant at every dimension. Unscaled: decaying — about half by
  $d = 512$, still falling.
- One division by $\sqrt{d}$ removes the problem. Part of the definition,
  not a tuning trick.
:::

::: {.slide title="Two reasons to mask"}
**Padding** — variable-length sequences share a minibatch:

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

**Causality** — a language model computes all positions in parallel, but
query $t$ must not see keys beyond $t$.

Same fix for both: overwrite invalid scores before the softmax.

::: {.d2l-note}
Not with $-10^6$ (overflows float16), not with $-\infty$ (a fully masked
query → NaN). Use the dtype's most negative *finite* value.
:::
:::

::: {.slide title="masked_softmax"}
@attention-scoring-the-masked-softmax-operation-1
:::

::: {.slide title="Masking in action"}
Valid lengths 2 and 3 — weights beyond the prefix are exactly zero, rows
still sum to one:

@attention-scoring-the-masked-softmax-operation-2

. . .

Per-query valid lengths work too:

@attention-scoring-the-masked-softmax-operation-3
:::

::: {.slide title="Causal masking"}
Query $t$ sees keys $1, \ldots, t$: valid lengths are just
$(1, 2, \ldots, n)$. The pattern is lower triangular:

@attention-scoring-causal-masking

- On the attention side, this mask is the key difference between reading a
  sequence and being trainable, in parallel, to generate one.
:::

::: {.slide title="Composing masks"}
The general interface is a boolean tensor — entry $(i, j)$: may query $i$
see key $j$? Requirements compose by AND, broadcasting does the bookkeeping:
padding $(\textrm{batch}, 1, \textrm{keys})$ ∧ causal
$(1, \textrm{queries}, \textrm{keys})$.

@!attention-scoring-composing-masks

::: {.d2l-note}
The composite must still leave every query at least one valid key — masks
that are harmless alone can intersect to an empty row.
:::
:::

::: {.slide title="Batched attention"}
Attention over a minibatch is two batched matrix multiplications and one
softmax:

$$\mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times v}.$$

@attention-scoring-batch-matrix-multiplication
:::

::: {.slide title="DotProductAttention"}
Scoring, masking, dropout on the weights, value pooling — a dozen lines
the rest of the book reuses:

@attention-scoring-the-dotproductattention-class-1
:::

::: {.slide title="Masked attention, verified"}
Valid lengths (2, 6): the stored weights vanish beyond the second and
sixth key.

@!attention-scoring-the-dotproductattention-class-3
:::

::: {.slide title="From alignment to attention"}
[Where all of this came from]{.kicker}

2014 machine translation: one fixed vector between encoder and decoder RNNs
— long sentences don't fit. Bahdanau, Cho & Bengio: let the decoder state
*query* all encoder states, one fresh summary per output token —
"jointly learning to align and translate".

![Learned weights behave like soft alignments — monotone, except where the languages reorder words.](../img/mdl-attention-alignment.svg){width=46%}
:::

::: {.slide title="Additive scoring"}
Decoder state and encoder states had different sizes, so the original
score was a tiny MLP:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}).$$

@attention-scoring-from-alignment-to-attention

- Dot products won on hardware: one matmul vs. an $n \times m \times h$
  tensor of activations. Learned projections + dot product give the metric
  back — exactly the Transformer's form.
:::

::: {.slide title="Recap"}
- Scaled dot product is *the* scoring function:
  $\mathbf{q}^\top\mathbf{k}/\sqrt{d}$ keeps score variance at 1, so the
  softmax neither saturates nor starves its gradient — we measured both.
- Masking handles padding and causality with one primitive: overwrite
  invalid scores with the dtype's most negative finite value.
- `DotProductAttention` = bmm → masked softmax → dropout → bmm.
- Additive attention started it all, as learned soft alignment for
  translation; it survives as history.
:::
