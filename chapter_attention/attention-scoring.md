# Attention Scoring and Masking
:label:`sec_attention-scoring-functions`

In :numref:`sec_attention-pooling` the attention weights came from a kernel
chosen by hand. They follow distance in a fixed input space, and no amount
of data changes what counts as similar. In a sentence, however, a verb looking for
its subject wants the word that plays that grammatical role, which need not
be the word whose representation is closest to its own. Turning soft lookup
into a layer of a trained network raises two questions: which score should
the network learn, and which query--key pairs should count at all?

The score must be shaped by learned representations, and it must be cheap,
because it is evaluated for every pair: one attention head of GPT-2 small,
reading a context of 1,024 tokens, scores $1{,}024^2 = 1{,}048{,}576$
pairs. Some of those pairs must not count. Sequences of different lengths
share a minibatch only after padding, and no query should attend to the
padding. A language model trained to predict each next token processes all
positions at once, and the query at position $t$ must not see the tokens
after it.

This section answers both questions. The score is the dot product of query and
key, divided by $\sqrt{d}$ so that its scale does not grow with the width
$d$ of the vectors. Exclusions are boolean masks applied before the
softmax, and several of them combine by logical AND. With both in place,
attention over a whole minibatch is two batched matrix multiplications and
a softmax, the `DotProductAttention` layer that every later model in this
book builds on. The section ends with the translation model in which
learned attention first appeared.

```{.python .input #attention-scoring-attention-scoring-and-masking}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #attention-scoring-attention-scoring-and-masking}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Dot-Product Attention

We look for a score that learned queries and keys can shape and that costs
one short computation per pair. The Gaussian kernel of
:numref:`sec_attention-pooling` already contains one. Expanding its
exponent gives the score

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$
:eqlabel:`eq_gaussian-expansion`

The last term depends on the query only. It is the same for all keys, and
the softmax normalization :eqref:`eq_softmax_attention` removes it. If all
keys have the same norm, the middle term drops out in the same way, and
the Gaussian kernel and the dot product give identical weights. In general
the norms differ, and keeping only $\mathbf{q}^\top \mathbf{k}_i$ is a
modeling decision rather than an approximation: the dot product becomes
the compatibility function in its own right. Learned query and key
representations then decide what compatibility means, and the Gaussian
kernel remains the special case of keys with equal norms.

One adjustment controls the size of the scores. Suppose that the $d$
coordinates of the query and of the key are independent random variables
with mean zero and variance one. Their dot product then has mean zero and
variance $d$; for the GPT-2 head, with $d = 64$, a typical score is about
$\pm 8$. Dividing by $\sqrt{d}$ restores unit variance and gives the
*scaled dot-product* score of the Transformer
:cite:`Vaswani.Shazeer.Parmar.ea.2017`:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

The attention weights are, as always, the softmax of the scores:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

### Score Variance and the $1/\sqrt{d}$ Factor

Learned queries and keys are neither independent nor of unit variance, so
the calculation above describes a network at initialization. That is when
it matters. If one score exceeds the others by a wide margin, its weight
approaches $1$, the others approach $0$, and the softmax *saturates*. The
gradient that reaches the scores shrinks with it, because the Jacobian of
the softmax is

$$
\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha} \boldsymbol{\alpha}^\top,
$$
:eqlabel:`eq_softmax-jacobian`

which tends to the zero matrix as $\boldsymbol{\alpha}$ approaches a
one-hot vector. Row $i$ of the Jacobian carries the factor $\alpha_i$, so
the gradient of the loss with respect to score $i$ is proportional to the
weight of key $i$. In a saturated row, the keys with negligible weight
receive almost no gradient, and the query learns only about the few keys
that already carry weight. A key that loses at the start has little
chance to catch up. Only this route saturates. Gradients still reach the
values, the output projection, and any residual connection. For finite
scores the Jacobian is never exactly zero, since the all-ones vector alone
spans its null space (an exercise below proves this), but it can come
arbitrarily close.

We measure both effects. We draw random queries and keys with independent
standard normal entries, compute the attention weights of each query over
$64$ keys, and record the entropy of the weights as the dimension $d$
grows, with and without the factor $1/\sqrt{d}$. For reference, the
uniform distribution over $64$ keys has entropy $\log 64 \approx 4.2$
nats, and a one-hot distribution has entropy $0$.

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-1}
%%tab pytorch
def softmax_weights(d, scale, num_keys=64, num_queries=2048):
    """Log-weights of random queries over random keys, scores scaled."""
    q = torch.randn(num_queries, d)
    keys = torch.randn(num_queries, num_keys, d)
    scores = (keys @ q[..., None]).squeeze(-1) * scale
    return F.log_softmax(scores, dim=-1)

def entropy(log_alpha):
    return -(log_alpha.exp() * log_alpha).sum(-1).mean().item()

def largest_weight(log_alpha):
    return log_alpha.exp().max(-1).values.mean().item()

def jacobian_norm(log_alpha):
    """Mean Frobenius norm of the Jacobian diag(alpha) - alpha alpha^T."""
    alpha = log_alpha.exp()
    J = torch.diag_embed(alpha) - alpha[..., :, None] * alpha[..., None, :]
    return torch.linalg.matrix_norm(J).mean().item()
```

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-1}
%%tab jax
def softmax_weights(d, scale, num_keys=64, num_queries=2048):
    """Log-weights of random queries over random keys, scores scaled."""
    q = jax.random.normal(d2l.get_key(), (num_queries, d))
    keys = jax.random.normal(d2l.get_key(), (num_queries, num_keys, d))
    scores = (keys @ q[..., None]).squeeze(-1) * scale
    return jax.nn.log_softmax(scores, axis=-1)

def entropy(log_alpha):
    return float(-(jnp.exp(log_alpha) * log_alpha).sum(-1).mean())

def largest_weight(log_alpha):
    return float(jnp.exp(log_alpha).max(-1).mean())

def jacobian_norm(log_alpha):
    """Mean Frobenius norm of the Jacobian diag(alpha) - alpha alpha^T."""
    alpha = jnp.exp(log_alpha)
    J = (alpha[..., None] * jnp.eye(alpha.shape[-1])
         - alpha[..., :, None] * alpha[..., None, :])
    return float(jnp.linalg.norm(J, axis=(-2, -1)).mean())
```

The entropy uses the log-weights directly, so a weight that underflows to
zero contributes exactly zero rather than $0 \cdot \log 0$.

```{.python .input #attention-scoring-score-variance-and-the-1-sqrt-d-factor-1}
%%tab pytorch, jax
ds = [2**k for k in range(2, 10)]
entropies = [[entropy(softmax_weights(d, 1.0)) for d in ds],
             [entropy(softmax_weights(d, 1 / math.sqrt(d))) for d in ds]]
d2l.plot(ds, entropies, 'dimension d', 'entropy (nats)',
         legend=['unscaled', 'scaled by 1/sqrt(d)'], xscale='log')
```

With scaling, the entropy stays near $3.7$ nats over two orders of
magnitude of $d$: the weights are equally sharp at every width. Without
scaling it collapses as $d$ grows, to about $0.2$ nats at $d = 512$, a
nearly one-hot weighting that comes from the length of the vectors alone.
Unscaled, the largest of the $64$ weights takes about $80\%$ of the total
at $d = 64$ and about $90\%$ at $d = 512$. The norm of the Jacobian shows
what this does to the gradient that reaches the scores:

```{.python .input #attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2}
%%tab pytorch, jax
print('      entropy (nats)    largest weight     Jacobian norm')
print('  d' + '   unscaled scaled' * 3)
for d in (4, 16, 64, 128, 512):
    stats = [(entropy(la), largest_weight(la), jacobian_norm(la))
             for la in (softmax_weights(d, 1.0),
                        softmax_weights(d, 1 / math.sqrt(d)))]
    print(f'{d:3d}' + ''.join(f'   {u:8.3f} {s:6.3f}'
                              for u, s in zip(*stats)))
```

<!-- WRITER: the cell (id kept) now prints a table for d = 4, 16, 64, 128, 512 (2,048 random queries,
64 keys each): entropy, mean largest weight, Jacobian norm, unscaled and scaled. Dev runs (PyTorch, 5 repeats;
JAX agrees; the notebook runs of both tabs fall in these ranges): scaled entropy 3.68-3.72 at every d; unscaled entropy 2.8, 1.4, 0.55-0.60, 0.37-0.39, 0.18-0.19;
unscaled largest weight 0.27, 0.58, 0.78-0.80, 0.85-0.86, 0.92-0.93 (scaled about 0.11 at every d); Jacobian
norm scaled 0.178-0.184 at every d; unscaled 0.256-0.259, 0.288-0.297 (the rise), 0.207-0.222, 0.161-0.168,
0.090-0.096. So the unscaled norm rises from d = 4 to d = 16, crosses the scaled value between d = 64 and
d = 128, and is about half of it at d = 512. The 80% and 90% of the paragraph before the cell are now printed
(0.80 at 64, 0.93 at 512); quote the table, and move the reading of the largest weight after the cell. -->
With scaling, the norm is the same at every dimension. Without it, the
norm first *rises*: at small $d$ the unscaled weights concentrate on a few
keys, and the Jacobian is largest when two keys share the weight (an
exercise below finds this maximum). It crosses the scaled value near
$d = 128$, falls to about half of it at $d = 512$, and keeps falling. A
model with unscaled scores would therefore start training with nearly
one-hot attention, and the gradient that reaches its scores would flow
almost entirely through the few keys that each query already favors. The
factor $1/\sqrt{d}$ removes this dependence on the dimension. That is why
it is part of the definition :eqref:`eq_dot_product_attention` rather
than a hyperparameter to tune.

## Masking

So far every query attends to every key. Two situations require
exclusions. The first is *padding*. Sequences of different lengths share
a minibatch only after they are padded with a dummy token to a common
length (here `<blank>`):

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

The padding carries no meaning, and no query should give it weight. The
second is *causality*. A language model trained to predict the next token
computes its outputs for all positions of a sequence at once, and the
query at position $t$ must not see the keys at positions after $t$.
Otherwise it could copy the very token it is asked to predict.

Both are described by one object, a boolean *mask* with one entry per
query--key pair that is true where the query may attend to the key. The
true entries of a row need not form a prefix, so the same object also
describes the attention windows and document boundaries we meet later.

### The Masked Softmax Operation

The *masked softmax* gives zero weight to the excluded pairs. It does not
skip them, which would leave rows of different lengths. Instead it
overwrites the excluded scores with a very negative number before the
softmax, so that their exponentials vanish, and keeps the regular array
shapes that batched kernels need.

The number has to be chosen with the precision in mind. Older code used a
constant such as $-10^6$. That works in single precision. Float16, a format
used for training today, cannot represent numbers beyond about
$6.5 \times 10^4$, and depending on the framework $-10^6$ either
overflows to $-\infty$ or is refused with an error. Literal $-\infty$
also masks correctly, until some query has no valid key at all. Every
score of its row is then $-\infty$, the softmax subtracts the row maximum
and computes $-\infty - (-\infty)$, and the row becomes NaN. The NaN
spreads to everything computed from it. We therefore overwrite with the most negative
*finite* value of the dtype (`finfo(dtype).min`). After the softmax
subtracts the row maximum, its exponential underflows to exactly zero at
any precision, and no NaN can arise.

One case remains. A query whose keys are all excluded would now receive
the uniform average of invalid values, because all its scores are equal.
`masked_softmax` sets such rows to zero. A model should still ensure that
every query keeps at least one valid key.

```{.python .input #attention-scoring-the-masked-softmax-operation-1}
%%tab pytorch
def masked_softmax(X, mask=None):  #@save
    """Softmax over the last axis with zero weight where mask is False."""
    if mask is None:
        return F.softmax(X, dim=-1)
    # Most negative finite value: zero weight at any precision, no NaN
    X = X.masked_fill(~mask, torch.finfo(X.dtype).min)
    weights = F.softmax(X, dim=-1)
    # A query without any valid key gets zero weights, not a uniform row
    return torch.where(mask.any(-1, keepdim=True), weights, 0.0)
```

```{.python .input #attention-scoring-the-masked-softmax-operation-1}
%%tab jax
def masked_softmax(X, mask=None):  #@save
    """Softmax over the last axis with zero weight where mask is False."""
    if mask is None:
        return jax.nn.softmax(X, axis=-1)
    # Most negative finite value: zero weight at any precision, no NaN
    X = jnp.where(mask, X, jnp.finfo(X.dtype).min)
    weights = jax.nn.softmax(X, axis=-1)
    # A query without any valid key gets zero weights, not a uniform row
    return jnp.where(mask.any(-1, keepdims=True), weights, 0.0)
```

The mask need only *broadcast* against the scores, so a mask that is the
same for every query, or for every sequence, is stored once. A padding
mask is of this kind: key $j$ of a sequence is valid when $j$ is less than
the sequence's valid length. `padding_mask` turns a vector of valid
lengths, one per sequence, into a mask of shape (batch, 1, keys), which
the softmax broadcasts over the queries. A matrix of lengths, one per
query, gives a mask of shape (batch, queries, keys).

```{.python .input #attention-scoring-the-masked-softmax-operation-4}
%%tab pytorch
def padding_mask(valid_lens, num_keys):  #@save
    """Boolean mask that is True for the first valid_lens keys."""
    # valid_lens of shape (batch,) gives (batch, 1, keys); per-query
    # lengths of shape (batch, queries) give (batch, queries, keys)
    if valid_lens.dim() == 1:
        valid_lens = valid_lens[:, None]
    keys = torch.arange(num_keys, device=valid_lens.device)
    return keys < valid_lens[..., None]
```

```{.python .input #attention-scoring-the-masked-softmax-operation-4}
%%tab jax
def padding_mask(valid_lens, num_keys):  #@save
    """Boolean mask that is True for the first valid_lens keys."""
    # valid_lens of shape (batch,) gives (batch, 1, keys); per-query
    # lengths of shape (batch, queries) give (batch, queries, keys)
    if valid_lens.ndim == 1:
        valid_lens = valid_lens[:, None]
    return jnp.arange(num_keys) < valid_lens[..., None]
```

We try both on a minibatch of two sequences with two queries and four
keys each, and with valid lengths $2$ and $3$. The weights beyond the
valid length should be zero, and each row should still sum to $1$.

<!-- WRITER: the demo now runs on equal scores (zeros), so the reader can predict the output before the
cell: rows (1/2, 1/2, 0, 0) for the first sequence and (1/3, 1/3, 1/3, 0) for the second; say so in the
sentence above. The per-query-length cell (-3) is cut; its shape note stays in padding_mask's comment. -->

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab pytorch
masked_softmax(torch.zeros(2, 2, 4), padding_mask(torch.tensor([2, 3]), 4))
```

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab jax
masked_softmax(jnp.zeros((2, 2, 4)), padding_mask(jnp.array([2, 3]), 4))
```

### Causal Masking

In a causal mask, the query at position $i$ may see the keys at positions
$j \leq i$. The mask is a lower triangle, and in a sequence of length $n$
it excludes $n(n-1)/2$ of the $n^2$ pairs, almost half of them. For one head
of GPT-2 small reading $n = 1{,}024$ tokens, that is $523{,}776$ of the $1{,}048{,}576$
scores.
`causal_mask` also covers the case of fewer queries than keys, which
arises when new tokens attend to the keys of earlier tokens kept in a
cache (:numref:`sec_kv-cache`). The queries are then the *last* positions
of the sequence, and query $i$ may see key $j$ when $j \leq i +
(\textrm{keys} - \textrm{queries})$.

```{.python .input #attention-scoring-causal-masking}
%%tab pytorch
def causal_mask(num_queries, num_keys=None, device=None):  #@save
    """Boolean (queries, keys) mask, True where a query may see a key."""
    # Queries are the last positions, as when new tokens attend to a
    # cache: query i sees key j if j <= i + num_keys - num_queries
    num_keys = num_queries if num_keys is None else num_keys
    i = torch.arange(num_queries, device=device)[:, None]
    j = torch.arange(num_keys, device=device)[None, :]
    return j <= i + (num_keys - num_queries)
```

```{.python .input #attention-scoring-causal-masking}
%%tab jax
def causal_mask(num_queries, num_keys=None):  #@save
    """Boolean (queries, keys) mask, True where a query may see a key."""
    # Queries are the last positions, as when new tokens attend to a
    # cache: query i sees key j if j <= i + num_keys - num_queries
    num_keys = num_queries if num_keys is None else num_keys
    i = jnp.arange(num_queries)[:, None]
    j = jnp.arange(num_keys)[None, :]
    return j <= i + (num_keys - num_queries)
```

<!-- WRITER: the random causal heatmap (causal-masking-1) is cut. The causal triangle is now shown by the
composite-mask cell below on equal scores (the valid-length-6 panel: row i holds 1/(i+1) on keys 0..i).
The next paragraph reads the cut heatmap; fold it into the reading of the composite cell. -->
The first query attends only to itself, and each later query spreads its
weight over itself and its predecessors. The triangle lets a model train
on all positions of a sequence in parallel without showing any position
its future. Every decoder in the following chapters uses it.

### Composing Masks
:label:`subsec_composing-masks`

Each requirement excludes some pairs. Padding excludes the keys beyond a
sequence's length, causality the keys after the query, and the attention
windows of :numref:`sec_windowed-linear-attention` the keys too far from it. A
pair is allowed only if every requirement allows it, so masks compose by
logical AND. Broadcasting keeps the composite small. A padding mask of
shape (batch, 1, keys) and a causal mask of shape (queries, keys) combine
into a mask of shape (batch, queries, keys), and neither is copied per
example or per query before the AND (:numref:`fig_mask-algebra`).

![Masks compose by logical AND. For a sequence of length 6 with valid length 3, the padding mask stores one row of keys, which broadcasting repeats for every query (faint); the causal mask is a lower triangle; their AND allows a pair only if both do. Rows are queries, columns are keys, and filled entries are allowed.](../img/mdl-attention-mask-algebra.svg)
:label:`fig_mask-algebra`

Here are the weights for a minibatch of two sequences of length $6$ with
valid lengths $6$ and $3$:

```{.python .input #attention-scoring-composing-masks}
%%tab pytorch
n = 6
padding = padding_mask(torch.tensor([6, 3]), n)  # (batch, 1, key)
causal = causal_mask(n)                          # (query, key)
mask = padding & causal                          # (batch, query, key)
weights = masked_softmax(torch.zeros(2, n, n), mask)  # Equal scores
d2l.show_heatmaps(weights[None], xlabel='Keys', ylabel='Queries',
                  titles=['valid length 6', 'valid length 3'],
                  figsize=(5, 2.5))
```

```{.python .input #attention-scoring-composing-masks}
%%tab jax
n = 6
padding = padding_mask(jnp.array([6, 3]), n)  # (batch, 1, key)
causal = causal_mask(n)                       # (query, key)
mask = padding & causal                       # (batch, query, key)
weights = masked_softmax(jnp.zeros((2, n, n)), mask)  # Equal scores
d2l.show_heatmaps(weights[None], xlabel='Keys', ylabel='Queries',
                  titles=['valid length 6', 'valid length 3'],
                  figsize=(5, 2.5))
```

<!-- WRITER: the composite cell now uses equal scores (zeros): in the valid-length-6 panel row i holds
1/(i+1) on keys 0..i; in the valid-length-3 panel row i holds 1/min(i+1, 3) on keys 0..min(i, 2), so rows
3 to 5 equal row 2 (1/3 each on the first three keys). State the prediction before the cell. -->
The first sequence shows the plain causal triangle. The second has valid
length $3$, so every row stops at the third key. Composition makes empty
rows more likely. Masks that each leave every query a key can still have
an empty intersection, so the guarantee of a valid key must hold for the
composite mask.

The same algebra handles *packed sequences*. Padding many short documents
to a common length wastes computation on the padding, so training
pipelines concatenate documents into one row instead. A mask that allows a
pair only within one document, ANDed with the causal mask, keeps the
documents apart. Here a row of length $6$ holds three documents of
lengths $3$, $2$, and $1$:

```{.python .input #attention-scoring-composing-masks-1}
%%tab pytorch
doc = torch.tensor([0, 0, 0, 1, 1, 2])  # Document of each position
same_doc = doc[:, None] == doc[None, :]
packed = same_doc & causal_mask(6)
d2l.show_heatmaps(packed.float()[None, None], xlabel='Keys',
                  ylabel='Queries', cbar=False)
```

```{.python .input #attention-scoring-composing-masks-1}
%%tab jax
doc = jnp.array([0, 0, 0, 1, 1, 2])  # Document of each position
same_doc = doc[:, None] == doc[None, :]
packed = same_doc & causal_mask(6)
d2l.show_heatmaps(packed.astype(jnp.float32)[None, None], xlabel='Keys',
                  ylabel='Queries', cbar=False)
```

The composite mask is block diagonal, with one causal triangle per
document: each position attends only to the earlier positions of its own
document.

## Batched Attention

<!-- WRITER: the "Batch Matrix Multiplication" subsection, its display (eq_batch-matrix-mul, label
subsec_batch_dot; no reference outside this file) and its shape-check cell are cut. One sentence in the class
subsection says that @ multiplies the last two axes and carries the leading ones (batch, later heads).
Rename this heading "Attention for a Minibatch". -->

### The DotProductAttention Class
:label:`subsec_dot-product-attention-class`

Scaled dot-product attention can now be written for many queries at once.
For $n$ queries and $m$ key--value pairs, with queries and keys of width
$d$ and values of width $d_v$, stack the queries into $\mathbf{Q} \in
\mathbb{R}^{n \times d}$, the keys into $\mathbf{K} \in \mathbb{R}^{m
\times d}$, and the values into $\mathbf{V} \in \mathbb{R}^{m \times
d_v}$. The whole computation is two matrix products and a softmax
(:numref:`fig_attention-shapes`):

$$ \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top }{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n \times d_v}.$$
:eqlabel:`eq_softmax_QK_V`

![Shapes in scaled dot-product attention. $\mathbf{Q}\mathbf{K}^\top/\sqrt{d}$ holds the scores of all $n \times m$ query--key pairs; masking and a softmax over each row turn them into the weights $\mathbf{A}$ (here with a causal mask, which leaves the upper triangle empty); multiplying by $\mathbf{V}$ gives one weighted sum of values per query.](../img/mdl-attention-shapes.svg)
:label:`fig_attention-shapes`

The softmax normalizes each row of the score matrix, so row $i$ of the
output is the attention :eqref:`eq_attention_pooling` of query $i$. The
cost is visible in the shapes. For the GPT-2 head at a context of $n = m
= 1{,}024$ with $d = d_v = 64$, the score matrix holds $1{,}048{,}576$
entries, 4 MiB in single precision, and the two products take $2 \cdot 2
n^2 d \approx 0.27$ billion floating-point operations. Both numbers grow
with the square of the context length, which is the subject of
:numref:`sec_attention-at-scale`.

Queries and keys must have the same width $d$. That is no restriction in
practice, because the queries and keys of a network are computed from its
inputs by learned linear maps, which can map to any common width
(:numref:`sec_multihead-attention`). The class applies dropout to the
attention weights as a regularizer and keeps the weights of its last call
in `attention_weights` for inspection.

```{.python .input #attention-scoring-the-dotproductattention-class-1}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot-product attention over the last two axes."""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    # Leading axes (batch, and heads if present) are carried along.
    # Shape of queries: (..., no. of queries, d)
    # Shape of keys: (..., no. of key-value pairs, d)
    # Shape of values: (..., no. of key-value pairs, value dimension)
    # mask: boolean, broadcastable to (..., no. of queries, no. of keys),
    # True where a query may attend to a key
    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, mask)
        return self.dropout(self.attention_weights) @ values
```

```{.python .input #attention-scoring-the-dotproductattention-class-1}
%%tab jax
class DotProductAttention(nnx.Module):  #@save
    """Scaled dot-product attention over the last two axes."""
    def __init__(self, dropout=0.0, *, rngs=None):
        # rngs is needed only to draw dropout masks (dropout > 0)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        # A Variable, so that views of the module (nnx.view) share it
        self.last_weights = nnx.Intermediate(None)

    # Leading axes (batch, and heads if present) are carried along.
    # Shape of queries: (..., no. of queries, d)
    # Shape of keys: (..., no. of key-value pairs, d)
    # Shape of values: (..., no. of key-value pairs, value dimension)
    # mask: boolean, broadcastable to (..., no. of queries, no. of keys),
    # True where a query may attend to a key
    def __call__(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = queries @ keys.swapaxes(-2, -1) / math.sqrt(d)
        weights = masked_softmax(scores, mask)
        if not isinstance(weights, jax.core.Tracer):
            # Record the weights of eager calls for inspection; inside
            # jit-compiled code skip it, so the module state keeps a
            # fixed structure and the compiled step is reused
            self.last_weights.set_value(weights)
        return self.dropout(weights) @ values

    @property
    def attention_weights(self):
        """Weights of the last eager call: (..., queries, keys)."""
        return self.last_weights.get_value()
```

:begin_tab:`jax`
The weights live in an `nnx.Intermediate` variable rather than in a plain
attribute, so that a view of the module created with `nnx.view` shares
them with the original. The class records them only in eager calls. Under
`nnx.jit` the weights are abstract values of a function being compiled.
Storing them would change the structure of the module's state between
calls and force a new compilation. It would also make every compiled
training step hand back the full (queries, keys) weight array of every
layer, memory that the step otherwise frees as soon as the weights have
been used. The record therefore keeps its previous value under `jit`, and
we call the module outside `jit` when we want to look at its weights.
:end_tab:

<!-- WRITER: the class test (id kept) now runs the three-token worked example through the class with
dropout 0 and prints the weights and the output; write the example out in the text first (score, scale,
softmax, sum), then the cell reproduces it. The example: d = 4 (so sqrt(d) = 2), two queries
q1 = (2, 2, 0, 0), q2 = (0, 0, 2, 0); keys k1 = (1, 1, 0, 0), k2 = (0, 0, 1, 1), k3 = (0, 0, 1, -1); values
v1 = (1, 0), v2 = (0, 1), v3 = (1, 1). Raw scores: q1 (4, 0, 0), q2 (0, 2, 2); scaled: (2, 0, 0) and (0, 1, 1).
Weights: q1 e^2/(e^2 + 2) = 0.787 and 1/(e^2 + 2) = 0.107 twice; q2 1/(1 + 2e) = 0.155, e/(1 + 2e) = 0.422
twice. Outputs: q1 (0.787 + 0.107, 0.107 + 0.107) = (0.893, 0.213); q2 (0.155 + 0.422, 0.422 + 0.422) =
(0.578, 0.845). Query 1 singles out key 1; query 2 splits its weight between the two keys it matches
equally. The cell prints exactly these numbers in both tabs. The heatmap cell (-class-3) is cut. -->
We run the class on two queries, three keys, and three values with
dropout $0$, and print the weights and the output.

```{.python .input #attention-scoring-the-dotproductattention-class-2}
%%tab pytorch, jax
queries = d2l.tensor([[[2.0, 2, 0, 0], [0, 0, 2, 0]]])           # (1, 2, 4)
keys = d2l.tensor([[[1.0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]]])  # (1, 3, 4)
values = d2l.tensor([[[1.0, 0], [0, 1], [1, 1]]])                   # (1, 3, 2)
attention = DotProductAttention()
output = attention(queries, keys, values)
print('weights:', np.asarray(attention.attention_weights[0]).round(3))
print('output: ', np.asarray(output[0]).round(3))
```

### Matching the Letters of Two Words
:label:`subsec_cross-attention-alignment`

<!-- WRITER: moved here from multihead-attention.md with its id and label (the giving step removes it
there). The class is the local DotProductAttention. Rewrite per worklist B8: prediction first
(sqrt(32) = 5.7 for a match, near zero otherwise), the three patterns (drop "This is the averaging of the
copy-both task on a small scale"; two identical keys split the weight because content cannot tell them
apart, which is why position must enter), the sink remark in two sentences, the permutation remark pointing
to the positional section rather than to the multihead exercise. -->
To see cross-attention at work, give every letter of the alphabet a random
embedding of width 32, and let the letters of one word query the letters of
another through plain scaled dot-product attention, with no learned
projections. Matching letters share an embedding, so their scaled score is
close to $\sqrt{32} \approx 5.7$, while unrelated pairs score near zero. The
softmax turns this gap into an alignment we can read.

```{.python .input #multihead-attention-an-alignment-you-can-read}
%%tab pytorch
words = 'attention', 'translation'
letters = sorted(set(''.join(words)))
emb = torch.randn(len(letters), 32)
queries, keys = (emb[[letters.index(c) for c in w]][None] for w in words)
attention = DotProductAttention()
attention(queries, keys, keys)
d2l.show_heatmaps(attention.attention_weights[None], xlabel='keys',
                  ylabel='queries', figsize=(4, 3.5), cmap='Blues',
                  xticklabels=words[1], yticklabels=words[0])
```

```{.python .input #multihead-attention-an-alignment-you-can-read}
%%tab jax
words = 'attention', 'translation'
letters = sorted(set(''.join(words)))
emb = jax.random.normal(d2l.get_key(), (len(letters), 32))
queries, keys = (emb[jnp.array([letters.index(c) for c in w])][None]
                 for w in words)
attention = DotProductAttention()
attention(queries, keys, keys)
d2l.show_heatmaps(attention.attention_weights[None], xlabel='keys',
                  ylabel='queries', figsize=(4, 3.5), cmap='Blues',
                  xticklabels=words[1], yticklabels=words[0])
```

The map reads row by row. The queries are the letters of "attention", the
keys the letters of "translation". Three patterns appear. The `i` and `o`
queries find their single partners and attend sharply. The `a`, `t`, and `n`
queries find *two* copies each and split their weight between them, because
a head cannot tell identical keys apart. This is the averaging of the
copy-both task on a small scale. The `e` query, whose letter does not occur in
"translation", has no partner. Its weight goes to keys that resemble it only
by chance, and none of them receives the near-certain weight of a real match;
the softmax must distribute all of its probability mass even when no key
matches.
Trained models show all three patterns. Language models often handle the
third case by sending the leftover weight to the first token of the
sequence, which then acts as a sink for attention that has nowhere useful to
go :cite:`Xiao.Tian.Chen.ea.2024`. A row without a match, or a large weight
on such a token, therefore says that no key is relevant, not that the
favored key is, which is one reason why attention maps must be read with
care as explanations.

None of the layers in this section knows where a token sits. Permuting the
keys and values together leaves every output unchanged (Exercise
:exref:`ex_multihead-permuting-the-source` of
:numref:`sec_multihead-attention`). If "translation" were spelled
backward, every letter of "attention" would retrieve the same mixture, and
only the columns of the map would change places.
:numref:`sec_positional-information` adds positional information.

## Attention as Learned Alignment
:label:`subsec_learned-alignment`

Learned attention first appeared in neural machine translation. Around
2014 the leading neural approach encoded a source sentence with an RNN
into one fixed-size vector and decoded the translation from that vector
with a second RNN; we build such an encoder--decoder in
:numref:`sec_seq2seq`. This is the bottleneck of
:numref:`sec_queries-keys-values`: one vector had to represent the whole
sentence, and translation quality fell as sentences grew longer.

:citet:`Bahdanau.Cho.Bengio.2014` kept the two RNNs but changed what the
decoder reads. At every output step, the current decoder state served as a
query against all encoder states, which served as keys and values. The
weighted sum, a fresh one for every output token, entered the next
prediction. The paper's title called this "jointly learning to align and
translate". The learned weights resembled the soft word alignments of
classical statistical translation: mostly monotone along the diagonal,
with departures where the two languages order words differently.
Nothing in the training objective asked for alignments. They emerged
because they help to translate.

<!-- WRITER: fig_alignment (schematic) is cut and replaced by the trained toy alignment below (worklist
A9/B9). Bahdanau in three sentences (previous decoder state as the query; no motive for the additive
score), then this experiment. Model: source sequences of 8 tokens drawn uniformly from 16; the target at
output position j is the source token at perm[j], perm = (0, 1, 2, 4, 3, 5, 6, 7) (identity with one
swapped neighbor pair, the "black cat"/"chat noir" crossing); one learned query per output position (width
16); keys = W_k applied to [one-hot token; one-hot position]; values = one-hot tokens, so the output of
each query is a distribution over the vocabulary; loss = -log of the target token's entry; Adam, lr 0.05,
100 steps of 128 fresh sequences; 3 runs per tab. Dev runs, both tabs, every run: accuracy 1.000 on 1,000
fresh sequences; the largest weight of every output position sits on perm[j] (1.000); mean weight on
perm[j] 0.999 to 1.000. From a trace (PyTorch, 3 runs, not printed by the cell): at initialization
the weight on perm[j] is 0.12-0.14 (uniform is 0.125), after 10 steps 0.50-0.57 with the largest weight
already on perm[j] everywhere, after 20 steps 0.95. The heatmap shows one sequence, letters standing for
the token ids, source along the columns and target along the rows: a diagonal with the two swapped rows. -->
We train such a model. Each source sequence holds $8$ tokens drawn from a
vocabulary of $16$, and the target at output position $j$ is the source
token at position $\pi(j)$ of a fixed permutation $\pi$ that swaps one
pair of neighbors. Each output position has one learned query; the keys
are a learned linear map of the one-hot token and the one-hot position of
each source token; the values are the one-hot tokens, so each output is a
distribution over the vocabulary.

```{.python .input #attention-scoring-attention-as-learned-alignment-1}
%%tab pytorch
class ToyAligner(nn.Module):
    """One learned query per output position; keys from token and position."""
    def __init__(self, seq_len=8, vocab_size=16, d=16):
        super().__init__()
        self.vocab_size, n_in = vocab_size, vocab_size + seq_len
        self.queries = nn.Parameter(torch.randn(seq_len, d))
        self.W_k = nn.Parameter(torch.randn(n_in, d) / math.sqrt(n_in))
        self.attention = DotProductAttention()

    def forward(self, X):
        # X: (batch, seq_len) token ids; the values are their one-hot vectors
        tokens = F.one_hot(X, self.vocab_size).float()
        eye = torch.eye(X.shape[1], device=X.device)
        positions = eye.expand(len(X), -1, -1)
        keys = torch.cat((tokens, positions), -1) @ self.W_k
        # The queries broadcast over the batch: (batch, seq_len, vocab_size)
        return self.attention(self.queries, keys, tokens)
```

```{.python .input #attention-scoring-attention-as-learned-alignment-1}
%%tab jax
class ToyAligner(nnx.Module):
    """One learned query per output position; keys from token and position."""
    def __init__(self, seq_len=8, vocab_size=16, d=16, *, rngs):
        self.vocab_size, n_in = vocab_size, vocab_size + seq_len
        self.queries = nnx.Param(
            jax.random.normal(rngs.params(), (seq_len, d)))
        self.W_k = nnx.Param(jax.random.normal(rngs.params(), (n_in, d))
                             / math.sqrt(n_in))
        self.attention = DotProductAttention()

    def __call__(self, X):
        # X: (batch, seq_len) token ids; the values are their one-hot vectors
        tokens = jax.nn.one_hot(X, self.vocab_size)
        n = X.shape[1]
        positions = jnp.broadcast_to(jnp.eye(n), (len(X), n, n))
        keys = jnp.concatenate((tokens, positions), -1) @ self.W_k
        # The queries broadcast over the batch: (batch, seq_len, vocab_size)
        return self.attention(self.queries, keys, tokens)
```

The loss is the negative logarithm of the probability that the output
gives the target token. Every step draws a fresh minibatch.

```{.python .input #attention-scoring-attention-as-learned-alignment-2}
%%tab pytorch
perm = torch.tensor([0, 1, 2, 4, 3, 5, 6, 7])  # Output j copies source perm[j]

def align_data(n, seq_len=8, vocab_size=16):
    X = torch.randint(0, vocab_size, (n, seq_len))
    return X, X[:, perm]

def train_aligner(num_steps=100):
    model = ToyAligner()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(num_steps):
        X, Y = align_data(128)
        loss = -model(X).gather(-1, Y[..., None]).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
```

```{.python .input #attention-scoring-attention-as-learned-alignment-2}
%%tab jax
perm = jnp.array([0, 1, 2, 4, 3, 5, 6, 7])  # Output j copies source perm[j]

def align_data(key, n, seq_len=8, vocab_size=16):
    X = jax.random.randint(key, (n, seq_len), 0, vocab_size)
    return X, X[:, perm]

@nnx.jit
def align_step(model, optimizer, key):
    X, Y = align_data(key, 128)
    loss = lambda model: -jnp.log(
        jnp.take_along_axis(model(X), Y[..., None], -1)).mean()
    optimizer.update(model, nnx.grad(loss)(model))

def train_aligner(num_steps=100):
    model = ToyAligner(rngs=nnx.Rngs(d2l.get_key()))
    optimizer = nnx.Optimizer(model, optax.adam(0.05), wrt=nnx.Param)
    for key in jax.random.split(d2l.get_key(), num_steps):
        align_step(model, optimizer, key)
    return model
```

We train three models and evaluate each on $1{,}000$ fresh sequences: the
accuracy of the most probable token, the fraction of output positions $j$
whose largest weight falls on source position $\pi(j)$, and the mean
weight on that position.

```{.python .input #attention-scoring-attention-as-learned-alignment-3}
%%tab pytorch
X, Y = align_data(1000)
for run in range(3):
    model = train_aligner()
    with torch.inference_mode():
        accuracy = (model(X).argmax(-1) == Y).float().mean()
    weights = model.attention.attention_weights  # (1000, 8, 8)
    largest = (weights.argmax(-1) == perm).float().mean()
    on_perm = weights[:, torch.arange(8), perm].mean()
    print(f'run {run}: accuracy {accuracy:.3f}, largest weight on pi(j) '
          f'{largest:.3f}, mean weight on pi(j) {on_perm:.3f}')
```

```{.python .input #attention-scoring-attention-as-learned-alignment-3}
%%tab jax
X, Y = align_data(d2l.get_key(), 1000)
for run in range(3):
    model = train_aligner()
    accuracy = (model(X).argmax(-1) == Y).mean()
    weights = model.attention.attention_weights  # (1000, 8, 8)
    largest = (weights.argmax(-1) == perm).mean()
    on_perm = weights[:, jnp.arange(8), perm].mean()
    print(f'run {run}: accuracy {accuracy:.3f}, largest weight on pi(j) '
          f'{largest:.3f}, mean weight on pi(j) {on_perm:.3f}')
```

The weights of the last model on the first test sequence, with a letter
for each token id, show the permutation, crossing included:

```{.python .input #attention-scoring-attention-as-learned-alignment-4}
%%tab pytorch, jax
letters = 'abcdefghijklmnop'  # One letter per token id
d2l.show_heatmaps(model.attention.attention_weights[:1][None],
                  xlabel='source', ylabel='target', figsize=(3, 3),
                  cmap='Blues', xticklabels=[letters[int(i)] for i in X[0]],
                  yticklabels=[letters[int(i)] for i in Y[0]])
```

One detail differed from the score we chose above. The decoder state and
the encoder states had different widths, so instead of a dot product,
:citet:`Bahdanau.Cho.Bengio.2014` scored with a small network with one
hidden layer, now called *additive attention*:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}) \in \mathbb{R}.$$
:eqlabel:`eq_additive-attn`

Here $\mathbf{q} \in \mathbb{R}^{d_q}$ and $\mathbf{k} \in
\mathbb{R}^{d_k}$ may have different widths, and the learned parameters
are $\mathbf{W}_q \in \mathbb{R}^{r \times d_q}$, $\mathbf{W}_k \in
\mathbb{R}^{r \times d_k}$, and $\mathbf{w}_v \in \mathbb{R}^{r}$ for a
hidden width $r$. The two projections map queries and keys into a shared
$r$-dimensional space, and $\mathbf{w}_v$ reads a score off their sum.

<!-- WRITER: the additive cell (from-alignment-to-attention, random matrices, shape check) is cut, with its
reading. The facts to keep go to the foil paragraph under "Dot-Product Attention" (worklist B2): all dot
products of queries and keys form one matrix multiplication, while the additive score needs an n x m x r
array of hidden activations; at a small initialization the additive weights barely depend on the query
(Exercise 7 shows it). Exercise 7(c) cited this cell's code; it must now state W_q, W_k, w_v and the tanh
features itself. -->
The dot product can still use a learned metric. Projecting queries and
keys with learned matrices first leaves a score that runs at the speed of
matrix multiplication. This is the form attention takes in the Transformer,
whose authors dropped the recurrent networks altogether and kept attention
as the only operation that relates positions
:cite:`Vaswani.Shazeer.Parmar.ea.2017`.

We can now compute attention for a whole minibatch, with scores that
learned representations can shape and with any pattern of exclusions.
Each query, however, still forms a single weighted average, and nothing in
the computation knows where in the sequence a key sits.
:numref:`sec_multihead-attention` runs several lookups side by side, and
:numref:`sec_positional-information` adds positions.

## Summary

Scaled dot-product attention scores a query against a key by
$\mathbf{q}^\top \mathbf{k}/\sqrt{d}$, which is the Gaussian kernel without
the norm terms, and it scores all pairs of a minibatch with one batched
matrix multiplication. The factor $1/\sqrt{d}$ keeps the score variance
independent of the width, so the softmax does not saturate at
initialization merely because the vectors are long. A boolean mask
excludes padding, future positions, or any other pairs before the
softmax. Masks compose by logical AND under broadcasting. The excluded
scores are overwritten with the most negative finite value of the dtype,
which is safe in reduced precision. `DotProductAttention` combines
scoring, masking, dropout, and the weighted sum of values. Additive
attention, the score of the first neural alignment model, gave way to the
dot product because the dot product is a matrix multiplication.

## Exercises

1. **Invariances of the score.** The Gaussian score
   :eqref:`eq_gaussian-expansion` and the scaled dot product
   :eqref:`eq_dot_product_attention` both define attention weights through
   the softmax :eqref:`eq_attn-scoring-alpha`.
    1. For each of the two scores, determine which of the following
       transformations leaves the weights unchanged: adding one vector
       $\mathbf{c}$ to every key; adding $\mathbf{c}$ to the query and to
       every key; multiplying the query and every key by one orthogonal
       matrix.
    1. Queries and keys are usually computed from inputs by affine maps,
       $\mathbf{q} = \mathbf{W}_q \mathbf{x} + \mathbf{b}_q$ and
       $\mathbf{k}_i = \mathbf{W}_k \mathbf{y}_i + \mathbf{b}_k$. Expand
       $\mathbf{q}^\top \mathbf{k}_i$ into four terms and show that the
       weights do not depend on $\mathbf{b}_k$. What is the gradient of the
       loss with respect to $\mathbf{b}_k$?
    1. Through which of the four terms does $\mathbf{b}_q$ affect the
       weights, and what preference over the keys can it express that the
       bias-free score cannot?

1. [code] **The softmax Jacobian.** `jacobian_norm` measures the
   Frobenius norm of the Jacobian :eqref:`eq_softmax-jacobian`.
    1. Derive :eqref:`eq_softmax-jacobian` from
       :eqref:`eq_attn-scoring-alpha`, show that for finite scores its null
       space is spanned by the all-ones vector, and check the formula
       against automatic differentiation on a random score vector.
    1. Compute the Frobenius norm when $\boldsymbol{\alpha}$ is one-hot and
       when it is uniform over $m$ keys, and evaluate the uniform case for
       $m = 64$. Show that the squared norm is close to $\sum_i \alpha_i^2$
       when every weight is small, and use this to explain why the scaled
       norms printed for $d \in \{4, 16, 64, 128, 512\}$ exceed the uniform value.
    1. Compute the norm for weights spread uniformly over $k$ of the $m$
       keys, find the $k$ that maximizes it, and confirm by sampling random
       weight vectors that none exceeds this maximum. Use the result to
       explain why the unscaled norm printed for $d = 4$ exceeds the scaled
       one.

1. **Fully masked rows.** `masked_softmax` overwrites excluded scores with
   the most negative finite value of the dtype and then zeroes the rows of
   queries that have no valid key.
    1. Explain why, without the final zeroing, a fully masked row returns
       the uniform distribution, and why the same row returns NaN when the
       scores are overwritten with $-\infty$ instead. Trace both cases
       through the subtraction of the row maximum that a numerically stable
       softmax performs.
    1. In a self-attention layer, whose queries, keys, and values all come
       from the same sequence, let a query at a padding position have no
       valid key, as the composite masks of :numref:`subsec_composing-masks`
       can arrange, and let the excluded scores be overwritten with
       $-\infty$, so that the output at that position is NaN. That output
       is discarded, and in the next attention layer every valid query
       excludes the padding position as a key. Explain why the NaN
       nevertheless corrupts the outputs of all valid positions of that
       sequence in the next layer.

1. **Composite masks.** :numref:`subsec_composing-masks` combines masks by
   logical AND and warns that masks that are harmless alone can leave a
   query with no valid key. A causal sliding window of width $w$ lets the
   query at position $t$ attend to the keys at positions $t - w + 1, \ldots,
   t$.
    1. For a sequence of length $6$ with valid length $3$, write the padding
       mask and the causal sliding window of width $2$ as $6 \times 6$
       boolean matrices. Check that each leaves every query at least one
       key, and find the queries whose row of the AND is empty.
    1. Show that an AND of masks has no empty row if every mask allows each
       query to attend to its own position. Which of the padding, causal,
       sliding-window, and packed-sequence masks satisfy this condition, and
       what does the answer say about the empty rows you found?
    1. Some implementations add a bias matrix with entries $0$ for allowed
       pairs and $c < 0$ for excluded pairs, and compose masks by adding
       their biases. Take for $c$ the most negative finite single-precision
       value, about $-3.4 \times 10^{38}$. Compute the sum of two such
       biases in single precision, and determine what the softmax returns
       for a row in which every key is excluded by both masks and for a row
       in which every key is excluded but some keys by only one mask.

<!-- WRITER: Exercise 5 (to be cut per worklist B13) cites the old class-test inputs (10 keys, valid
lengths 2 and 6), which the cell no longer uses. -->
1. [code] **Attention dropout.** `DotProductAttention` applies dropout with
   rate $p$ to the attention weights, and the weights it exposes for
   visualization are those before dropout. Dropout multiplies each kept
   weight by $1/(1 - p)$ and sets the others to zero, so a row of weights
   no longer sums to $1$.
    1. Show that the row sum after dropout has expectation $1$ and variance
       $\frac{p}{1-p} \sum_i \alpha_i^2$, and evaluate the variance for a
       row that is uniform over $l$ keys and for a one-hot row. What is the
       output of a one-hot row in the draws in which dropout removes the
       attended weight?
    1. Use the inputs of :numref:`subsec_dot-product-attention-class` with
       dropout $0.5$ active, and pass as values a batch of $10 \times 10$
       identity matrices, so that each output row equals the weights after
       dropout. Compare the mean and the variance of the row sums over many
       draws with the prediction computed from the weights before dropout.

1. **Entropy of random scores.** ● The scaled entropy curve above
   stays near $3.7$ nats, below the uniform value $\log 64 \approx 4.16$.
   Fix one query $\mathbf{q} \in \mathbb{R}^d$ and draw $m$ keys with
   independent standard normal coordinates, as `softmax_weights` does.
    1. Show that the scaled scores $a_i = \mathbf{q}^\top \mathbf{k}_i /
       \sqrt{d}$ are independent normal variables with mean $0$ and variance
       $s^2 = \|\mathbf{q}\|^2 / d$, and compute the expected value of $s^2$
       when $\mathbf{q}$ is drawn as in `softmax_weights`.
    1. The entropy of the weights is $H = \log Z - \sum_i \alpha_i a_i$ with
       $Z = \sum_j \exp(a_j)$. For large $m$, replace $Z/m$ and
       $\sum_i \exp(a_i)\, a_i / m$ by their expected values and derive
       $$H \approx \log m - \frac{s^2}{2}.$$
       Evaluate the approximation for $m = 64$ and $s = 1$.
    1. The approximation needs $Z$ to concentrate around its mean. Compute
       the relative standard deviation of $Z$ as a function of $s$ and $m$,
       and derive from it a condition on $s^2$ under which the approximation
       holds. Unscaled scores have $s^2 = \|\mathbf{q}\|^2$, whose expected
       value is $d$. Evaluate the relative standard deviation for $m = 64$
       at $s^2 = 4$ and $s^2 = 8$, and compare the approximation with the
       unscaled entropies plotted above for $d = 4$ and
       $d = 8$.

1. [code] **Additive scores near zero.** ● The additive score
   :eqref:`eq_additive-attn` adds the projected query and the projected key
   inside a $\tanh$.
    1. Replace $\tanh$ by the identity and show that the attention weights
       then do not depend on the query.
    1. Expand $\tanh$ to third order around zero, identify the lowest-order
       terms through which the weights depend on the query, and state how
       this dependence scales when $\mathbf{W}_q$ and $\mathbf{W}_k$ are
       both multiplied by a small factor $\epsilon$.
    <!-- WRITER: the additive cell this part cited is cut; state W_q, W_k, w_v and the tanh features here. -->
    1. Multiply `W_q` and `W_k` in the code of
       :numref:`subsec_learned-alignment` by $\epsilon \in \{1, 0.3, 0.1,
       0.03\}$, and measure the largest difference between the weight rows
       of the three queries. Check the scaling you derived, and state what
       the result for $\epsilon = 1$ implies for additive attention at
       initialization.

1. [extended] **A learned alignment.** ●
   :citet:`Bahdanau.Cho.Bengio.2014` observed that attention weights trained
   only for translation behave like alignments.
   <!-- WRITER: part (a) is now the body experiment of subsec_learned-alignment; recast as "Alignment
   variants" on the body's model (old (b), (c), (d)). fig_alignment is retired. -->
   The same effect appears in a toy model without recurrent networks. Draw
   source sequences of $8$ tokens uniformly from a vocabulary of $16$, and
   let the target at output position $j$ be the source token at position
   $\pi(j)$ for a fixed permutation $\pi$ of the positions. Score one
   learned query per output position against keys that a learned linear map
   computes from the concatenated one-hot token and one-hot position of each
   source token, use the one-hot source tokens as values, and train on the
   negative logarithm of the output entry that belongs to the target token.
    1. Implement the model with `DotProductAttention`, train it, and plot
       the attention weights of one sequence with `d2l.show_heatmaps`.
       Compare the map with $\pi$.
    1. Replace the score by the additive rule :eqref:`eq_additive-attn` and
       compare the number of steps that each model needs to reach $99\%$
       accuracy.
    1. Give queries and keys the width $d \in \{16, 256, 1024\}$ with
       standard normal initialization, and train with and without the
       factor $1/\sqrt{d}$. Relate the training curves to the entropy of the
       weights at initialization.
    1. Explain why a fixed query per output position cannot learn a
       reordering that depends on the tokens, such as swapping two
       neighbors only when the first belongs to a designated subset of the
       vocabulary, and state what the query would have to depend on.

[Discussions](https://d2l.discourse.group/t/36769)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.2]{.kicker}

Attention Scoring and Masking<br>
**scaled dot products · the 1/√d factor · masks and their algebra · attention as two matrix products · additive attention**
:::
:::

::: {.slide title="One Causal Head Scores a Million Pairs and Must Discard Half"}
[The problem]{.kicker}

One GPT-2-small head at a context of 1,024 tokens:

- $1{,}024^2 = 1{,}048{,}576$ query–key scores per layer;
- a fixed kernel cannot learn which of them should be large;
- a language model must ignore the future: $523{,}776$ scores excluded;
- a padded minibatch must ignore the padding.

::: {.d2l-note .warn}
We need a learnable score that is fast for every pair, and a way to
exclude pairs.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the dot-product score from the Gaussian kernel;
- explain and measure why scores are divided by $\sqrt{d}$;
- build padding and causal masks and combine them by AND;
- write attention for $n$ queries as two matrix products and a softmax;
- count the scores, bytes, and FLOPs of one head.

**Running example:** one GPT-2-small head, $d = 64$, context
$n = 1{,}024$.
:::

::: {.slide title="Learned Scores, Masks, and Batching Turn Lookup into a Layer"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| **Attention Scoring and Masking** | **scaled dot products, masks** |
| Multi-Head and Cross-Attention | several heads, shared keys and values |
| Positional Information | where a token sits |
| The Cost of Attention | quadratic cost, fused kernels |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** soft lookup, softmax, padding of minibatches.
- **Skips:** heads and projections (next deck), positions.
:::
:::
:::

::: {.slide title="The Gaussian Kernel Is a Dot Product Plus Two Norms"}
$$-\tfrac{1}{2}\|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i - \tfrac{1}{2}\|\mathbf{k}_i\|^2 - \tfrac{1}{2}\|\mathbf{q}\|^2$$

- $\|\mathbf{q}\|^2$ is the same for every key: the softmax removes it.
- $\|\mathbf{k}_i\|^2$ drops out when all keys have equal norm.
- Keeping only $\mathbf{q}^\top\mathbf{k}_i$ is a modeling choice: learned
  queries and keys decide what "compatible" means.
:::

::: {.slide title="At d = 64 a Typical Unscaled Score Is ±8"}
Entries of $\mathbf{q}, \mathbf{k} \in \mathbb{R}^d$ independent, mean 0,
variance 1:

$$\textrm{Var}(\mathbf{q}^\top \mathbf{k}) = \sum_{j=1}^d \textrm{Var}(q_j k_j) = d$$

- $d = 64$: standard deviation $8$.
- Divide by $\sqrt{d}$: variance 1 at every width.

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k} / \sqrt{d}$$
:::

::: {.slide title="Unscaled Scores Collapse the Softmax Entropy"}
2,048 random queries, 64 keys each; entropy of the weights:

@!attention-scoring-score-variance-and-the-1-sqrt-d-factor-1

- Uniform over 64 keys: 4.2 nats. Scaled: about 3.7 at every $d$.
- Unscaled: about 0.2 nats at $d = 512$, nearly one-hot.
:::

::: {.slide title="Saturation Shrinks the Gradient That Reaches the Scores"}
$$\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top \to \mathbf{0} \quad \textrm{as } \boldsymbol{\alpha} \to \textrm{one-hot}$$

@!attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2

- Scaled: the same norm at every $d$. Unscaled: larger at $d = 64$, about
  half at $d = 512$, still falling.
- The gradient of score $i$ carries the factor $\alpha_i$: queries and
  keys learn only through the few keys that already carry weight.
:::

::: {.slide title="Check: Two Keys One Standard Deviation Apart"}
[Check]{.kicker}

In a $d = 64$ head, two keys score one standard deviation apart. What is
the ratio of their weights without and with the factor $1/\sqrt{d}$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Unscaled, the scores differ by 8: ratio $e^8 \approx 3{,}000$.
Scaled, they differ by 1: ratio $e \approx 2.7$. Without the factor, one
key takes most of the weight (about 80% over 64 random keys) before
training has begun.
:::
:::
:::

::: {.slide title="Padding and Causality Both Exclude Query–Key Pairs"}
**Padding:** sequences of different lengths share a minibatch.

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

**Causality:** the query at position $t$ must not see keys after $t$, or
it can copy the token it should predict.

::: {.d2l-note .rule}
One object for both: a boolean mask, true where query $i$ may attend to
key $j$.
:::
:::

::: {.slide title="Fill Excluded Scores with the Most Negative Finite Value"}
| fill value | excluded key | row with every key excluded |
|---|---|---|
| $-10^6$ | weight 0; float16 overflows to $-\infty$ or refuses it | uniform; float16: NaN or an error |
| $-\infty$ | weight 0 | NaN: $-\infty - (-\infty)$ |
| `finfo(dtype).min` | weight 0 at any precision | uniform, then set to zero |

::: {.d2l-note}
float16 stops at $6.5 \times 10^4$. `finfo(float16).min`
$\approx -6.5 \times 10^4$, and its exponential underflows to exactly 0.
:::
:::

::: {.slide title="The Masked Softmax Fills, Normalizes, and Zeroes Empty Rows"}
@attention-scoring-the-masked-softmax-operation-1

The mask only has to broadcast against the scores
(…, queries, keys).
:::

::: {.slide title="Valid Lengths Become a Padding Mask"}
Two sequences, two queries, four keys; valid lengths 2 and 3:

@attention-scoring-the-masked-softmax-operation-2

- Shape (batch, 1, keys): one row per sequence, broadcast over queries.
- Excluded weights are exactly 0; each row sums to 1.
:::

::: {.slide title="A Causal Mask Is a Lower Triangle"}
<!-- WRITER: the random causal heatmap is cut; this slide now shows the composite cell on equal scores
(left panel: row i holds 1/(i+1); right panel: valid length 3, rows stop at the third key). -->
@!attention-scoring-composing-masks

- Query $i$ sees keys $j \le i$: $n(n-1)/2$ of $n^2$ pairs excluded.
- With fewer queries than keys (a cache), the queries are the last
  positions.
:::

::: {.slide title="Masks Compose by AND Under Broadcasting"}
![Padding (batch, 1, keys) AND causal (queries, keys) gives (batch, queries, keys).](../img/mdl-attention-mask-algebra.svg){width=92%}

A pair is allowed only if every mask allows it; no mask is copied per
query or per example.
:::

::: {.slide title="Packed Documents Need a Block-Diagonal Mask"}
Three documents of lengths 3, 2, 1 in one row: same document AND causal.

@!attention-scoring-composing-masks-1

Each position attends only to earlier positions of its own document.
:::

::: {.slide title="Check: An Empty Row"}
[Check]{.kicker}

A query's row of the composite mask is all false. What does the softmax
return if the excluded scores are filled with $-\infty$? With
`finfo.min`? What does `masked_softmax` return?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** NaN, which then spreads to every later layer; a uniform
average over invalid values; a row of zeros. Masks that are harmless
alone can still intersect to an empty row.
:::
:::
:::

::: {.slide title="Attention Is Two Matrix Products and a Softmax"}
$$\mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times d_v}$$

![Shapes of Q, K, the scores, the masked weights, V, and the output.](../img/mdl-attention-shapes.svg){width=48%}
:::

::: {.slide title="The Layer Is Three Lines" only="pytorch"}
The core of `DotProductAttention.forward`:

```{.python #attention-scoring-exercises-1}
# queries (..., n, d), keys (..., m, d), values (..., m, d_v)
scores = queries @ keys.transpose(-2, -1) / math.sqrt(d)
weights = masked_softmax(scores, mask)  # (..., n, m)
output = dropout(weights) @ values      # (..., n, d_v)
```

- Leading axes (batch, later heads) are carried along by `@`.
- The class keeps `weights` in `attention_weights` for inspection.
:::

::: {.slide title="The Layer Is Three Lines" only="jax"}
The core of `DotProductAttention.__call__`:

```{.python #attention-scoring-exercises-2}
# queries (..., n, d), keys (..., m, d), values (..., m, d_v)
scores = queries @ keys.swapaxes(-2, -1) / math.sqrt(d)
weights = masked_softmax(scores, mask)  # (..., n, m)
output = dropout(weights) @ values      # (..., n, d_v)
```

- Leading axes (batch, later heads) are carried along by `@`.
- The class keeps `weights` in `attention_weights` for inspection.
:::

::: {.slide title="One GPT-2 Head at 1,024 Tokens: 4 MiB of Scores, 0.27 GFLOP"}
$n = m = 1{,}024$, $d = d_v = 64$, float32:

- scores: $1{,}024^2 = 1{,}048{,}576$ entries, $\times 4$ bytes = 4 MiB;
- FLOPs: $2 \cdot n^2 d$ for $\mathbf{Q}\mathbf{K}^\top$ plus
  $2 \cdot n^2 d_v$ for $\mathbf{A}\mathbf{V}$ $\approx 0.27$ billion.

**Check.** How large is the score matrix at $n = 8{,}192$?

::: {.fragment .fade-up}
::: {.d2l-note}
$8{,}192^2 \times 4$ bytes = 256 MiB per head and layer: 64 times more
for 8 times the context.
:::
:::
:::

::: {.slide title="DotProductAttention Keeps Its Weights for Inspection"}
<!-- WRITER: replaces the cut heatmap of random inputs with the three-token worked example (d = 4, scaled
scores (2, 0, 0) and (0, 1, 1); weights (0.787, 0.107, 0.107) and (0.155, 0.422, 0.422); outputs
(0.893, 0.213) and (0.578, 0.845)); add a check slide before it per worklist B14. -->
Two queries, three keys and values, $d = 4$, dropout 0:

@attention-scoring-the-dotproductattention-class-2

- The layer: scores, masked softmax, dropout on the weights, weighted sum.
- `attention_weights` holds the weights of the last call.
:::

::: {.slide title="Learned Attention Began as Alignment in Translation"}
[2014]{.kicker}

An RNN encoder squeezed each source sentence into one vector; quality fell
with sentence length. Bahdanau, Cho, and Bengio let the decoder state
query all encoder states, a fresh weighted sum per output word.

<!-- WRITER: the schematic figure is retired; the slide shows the trained toy alignment (one learned query
per output position, a permutation with one swapped pair; every run puts the largest weight on pi(j)). -->
@!attention-scoring-attention-as-learned-alignment-4
:::

::: {.slide title="Additive Attention Scores Vectors of Different Widths"}
$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$$

$\mathbf{W}_q \in \mathbb{R}^{r \times d_q}$, $\mathbf{W}_k \in
\mathbb{R}^{r \times d_k}$, $\mathbf{w}_v \in \mathbb{R}^{r}$: a network
with one hidden layer of width $r$.

- It needs an $n \times m \times r$ array of hidden activations.
- The dot product is one matrix multiplication; learned projections
  before it recover a learned metric.
:::

::: {.slide title="Wrap-up"}
- Score $\mathbf{q}^\top\mathbf{k}/\sqrt{d}$: the Gaussian kernel without
  the norms; at $d = 64$ the factor turns a ±8 score into ±1.
- Unscaled scores saturate the softmax: about 0.2 nats of entropy at
  $d = 512$ instead of 3.7.
- Masks are booleans that compose by AND; fill with the dtype's most
  negative finite value; empty rows become zeros.
- One GPT-2 head at 1,024 tokens: 1M scores, 4 MiB, 0.27 GFLOP.

**Next:** several heads, each with its own learned projections.
:::
