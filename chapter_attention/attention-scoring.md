# Attention Scoring and Masking
:label:`sec_attention-scoring-functions`

In :numref:`sec_attention-pooling`, attention weights were determined by a
fixed Gaussian kernel over Euclidean distance in input space. In a sequence
model, however, similarity must be learned—a verb seeking its subject needs
the word playing that syntactic role, not the closest raw embedding. Turning
soft lookup into a trainable network layer raises two questions: which
scoring function $a(\mathbf{q}, \mathbf{k})$ should the layer compute, and
which query--key pairs should be excluded from the lookup?

Because a head reading $n = 1{,}024$ tokens evaluates $n^2 = 1{,}048{,}576$
query--key pairs per sequence, the score must be inexpensive and map cleanly
onto matrix multiplication hardware. Furthermore, invalid pairs—such as
padding tokens in a variable-length minibatch or future tokens in an
autoregressive language model—must be zeroed out before normalization.

This section derives *scaled dot-product attention*
($\mathbf{q}^\top\mathbf{k}/\sqrt{d}$), explains how the $1/\sqrt{d}$ factor
prevents softmax saturation at initialization, implements boolean masking and
mask composition, and demonstrates how learned attention recovers sequence
alignments from output supervision alone.

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

Expanding the negative squared distance of the unit-bandwidth Gaussian kernel
from :numref:`sec_attention-pooling` reveals an inner product plus two squared
norms:

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2 -\frac{1}{2} \|\mathbf{q}\|^2.
$$
:eqlabel:`eq_gaussian-expansion`

Because $-\tfrac{1}{2}\|\mathbf{q}\|^2$ is constant across all keys $i$, the
softmax normalization cancels it out. If all keys have equal norm,
$-\tfrac{1}{2}\|\mathbf{k}_i\|^2$ cancels as well. Dropping both norm terms
even when key norms vary yields the *dot-product score* $\mathbf{q}^\top
\mathbf{k}_i$. Unlike the Gaussian kernel, which penalizes large
$\|\mathbf{k}_i\|$, the dot product allows a key's norm to act as a multiplicative
prior: a longer key attracts larger weight from every query aligned with it
(Exercise :exref:`ex_queries-keys-values-average-two-values`c).

Early neural machine translation models instead used a one-hidden-layer MLP
known as *additive attention* :cite:`Bahdanau.Cho.Bengio.2014`:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

with learned matrices $\mathbf{W}_q \in \mathbb{R}^{r \times d_q}$,
$\mathbf{W}_k \in \mathbb{R}^{r \times d_k}$, and weight vector
$\mathbf{w}_v \in \mathbb{R}^{r}$. Although additive and dot-product scoring
have similar asymptotic operation counts ($O(nmr)$ versus $O(nmd)$ for $n$
queries and $m$ keys), additive attention must materialize an $n \times m
\times r$ tensor of hidden activations and apply $\tanh$ elementwise.
Dot-product attention computes all $n \times m$ scores in a single matrix
multiplication $\mathbf{Q}\mathbf{K}^\top$, without losing expressivity when
queries and keys are already produced by learned linear projections
$(\mathbf{W}_q \mathbf{q})^\top (\mathbf{W}_k \mathbf{k})$ (see also Exercise
:exref:`ex_attention-scoring-additive-near-zero`).

One crucial scaling factor remains. Suppose the coordinates of $\mathbf{q},
\mathbf{k} \in \mathbb{R}^d$ are independent random variables with mean $0$
and variance $1$. Then each term $q_j k_j$ has mean $0$ and variance $1$, so
the sum $\mathbf{q}^\top \mathbf{k}$ has mean $0$ and variance $d$ (standard
deviation $8$ at $d = 64$). To prevent large dot products from pushing the
softmax into regions of vanishing gradients at large $d$,
:citet:`Vaswani.Shazeer.Parmar.ea.2017` divide by $\sqrt{d}$ to restore unit
variance, giving the *scaled dot-product* score:

$$a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

Here $1/\sqrt{d}$ plays the role of the inverse temperature $1/\sigma^2$ in
:numref:`sec_attention-pooling`, calibrated automatically to the vector width
$d$. Softmax normalization then yields the attention weights

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})},$$
:eqlabel:`eq_attn-scoring-alpha`

and the output $\sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\, \mathbf{v}_i$.

### A Worked Example
:label:`subsec_attention-worked-example`

Let $d = 4$ (so $1/\sqrt{d} = 1/2$) with two queries and three key--value
pairs:

$$
\begin{aligned}
\mathbf{q}_1 &= (2, 2, 0, 0), & \mathbf{k}_1 &= (1, 1, 0, 0), & \mathbf{v}_1 &= (1, 0),\\
\mathbf{q}_2 &= (0, 0, 2, 0), & \mathbf{k}_2 &= (0, 0, 1, 1), & \mathbf{v}_2 &= (0, 1),\\
& & \mathbf{k}_3 &= (0, 0, 1, -1), & \mathbf{v}_3 &= (1, 1).
\end{aligned}
$$

1. *Scores:* $\mathbf{q}_1$ has raw dot products $(4, 0, 0)$ with the three
   keys; $\mathbf{q}_2$ has $(0, 2, 2)$.
2. *Scaling:* Dividing by $\sqrt{4} = 2$ gives $(2, 0, 0)$ and $(0, 1, 1)$.
3. *Softmax:* $\mathbf{q}_1$ assigns weights
   $\big(e^2/(e^2 + 2),\, 1/(e^2 + 2),\, 1/(e^2 + 2)\big) \approx
   (0.787, 0.107, 0.107)$; $\mathbf{q}_2$ assigns
   $\big(1/(1 + 2e),\, e/(1 + 2e),\, e/(1 + 2e)\big) \approx
   (0.155, 0.422, 0.422)$.
4. *Weighted sum:* $\mathbf{o}_1 = 0.787\,\mathbf{v}_1 + 0.107\,\mathbf{v}_2
   + 0.107\,\mathbf{v}_3 \approx (0.893, 0.213)$ and $\mathbf{o}_2 \approx
   (0.578, 0.845)$.

Even an orthogonal key receives nonzero softmax weight ($0.107$), and keys
that tie on the active coordinates of a query ($\mathbf{k}_2$ and
$\mathbf{k}_3$ under $\mathbf{q}_2$) split the remaining weight equally.
Omitting the $1/2$ scaling factor would sharpen $\mathbf{q}_1$'s weight on
$\mathbf{k}_1$ from $0.787$ to $e^4/(e^4 + 2) \approx 0.96$.

### Score Variance and the $1/\sqrt{d}$ Factor

When one score exceeds all others by a wide margin, the weight vector
$\boldsymbol{\alpha}$ approaches a one-hot vector and the softmax *saturates*.
The Jacobian of $\boldsymbol{\alpha} = \mathrm{softmax}(\mathbf{a})$,

$$
\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha} \boldsymbol{\alpha}^\top,
$$
:eqlabel:`eq_softmax-jacobian`

converges to the zero matrix as $\boldsymbol{\alpha}$ becomes one-hot
(Exercise :exref:`ex_attention-scoring-softmax-jacobian`). In a saturated
row, backpropagated gradients with respect to the scores vanish, making it
difficult for unselected keys to gain weight during training.

We can measure this effect on random standard normal queries ($2{,}048$
queries, each attending over $64$ random keys) across dimensions $d \in
\{4, \ldots, 512\}$, comparing unscaled scores against scores scaled by
$1/\sqrt{d}$. Uniform weights over $64$ keys have entropy $\log 64 \approx
4.16$ nats, whereas a one-hot distribution has entropy $0$.

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

```{.python .input #attention-scoring-score-variance-and-the-1-sqrt-d-factor-1}
%%tab pytorch, jax
ds = [2**k for k in range(2, 10)]
entropies = [[entropy(softmax_weights(d, 1.0)) for d in ds],
             [entropy(softmax_weights(d, 1 / math.sqrt(d))) for d in ds]]
d2l.plot(ds, entropies, 'width d', 'entropy (nats)',
         legend=['unscaled', 'scaled by 1/sqrt(d)'], xscale='log')
```

The table below reports the entropy, mean maximum weight, and mean Frobenius
norm of the softmax Jacobian at five widths.

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

With $1/\sqrt{d}$ scaling, the entropy stays near $\log 64 - \tfrac{1}{2}
\approx 3.66$ nats across all widths (Exercise
:exref:`ex_attention-scoring-entropy-of-random-scores`), with maximum weight
$\approx 0.11$ and Jacobian norm $\approx 0.18$. Without scaling, score
standard deviation grows as $\sqrt{d}$: by $d = 512$, entropy collapses to
$\approx 0.2$ nats, the top key takes over $90\%$ of the weight at random
initialization, and the Jacobian norm falls by half. Scaling by $1/\sqrt{d}$
decouples initial softmax sharpness from head width while leaving learned
projections free to sharpen weights during training (once queries and keys
align, dot products grow as $O(d)$, motivating the $1/d$ scaling in maximal
update parametrization, :numref:`sec_scaling`).

## Masking

Two standard conditions exclude specific query--key pairs from attention:

1. **Padding:** When sequences of unequal lengths are padded with `<blank>`
   tokens to form a minibatch, no query should attend to padding positions.
2. **Causality:** In autoregressive language modeling, the query at position
   $t$ may attend only to keys at positions $j \le t$.

Both constraints are represented by a boolean *mask* tensor that is `True`
where query $i$ is permitted to attend to key $j$.

### The Masked Softmax Operation

To preserve dense tensor shapes, *masked softmax* replaces excluded scores
(`~mask`) with a large negative number before calling `softmax`. The choice of
fill value requires care across floating-point formats:

* A hardcoded constant such as $-10^6$ overflows float16 (whose maximum finite
  magnitude is $65{,}504$).
* Literal $-\infty$ produces `NaN` whenever an entire query row is masked out,
  because the numerically stable softmax subtracts the row maximum $-\infty$
  to form $-\infty - (-\infty) = \mathrm{NaN}$.
* Using the most negative *finite* value of the tensor's data type
  (`finfo(dtype).min`) underflows excluded exponentials to exact zero in
  float16, bfloat16, and float32 without producing `NaN`. For any query row
  where all keys are masked out (`~mask.any(-1)`), `masked_softmax` explicitly
  zeroes the resulting weights.

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

Because `mask` only needs to broadcast against `X`, a padding mask shared
across all queries in a sequence can be stored with shape `(batch, 1, keys)`.

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

On a batch of two zero-score sequences of length $4$ with valid lengths $2$
and $3$, `masked_softmax` distributes weight uniformly over the valid prefix
of each row:

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab pytorch
masked_softmax(torch.zeros(2, 2, 4), padding_mask(torch.tensor([2, 3]), 4))
```

```{.python .input #attention-scoring-the-masked-softmax-operation-2}
%%tab jax
masked_softmax(jnp.zeros((2, 2, 4)), padding_mask(jnp.array([2, 3]), 4))
```

### Causal Masking

A causal mask allows query $i$ to attend to key $j$ only if $j \le i$,
forming a lower-triangular matrix that excludes $n(n-1)/2$ of the $n^2$ pairs
($523{,}776$ pairs at $n = 1{,}024$). When `num_queries < num_keys` (as during
cached generation), `causal_mask` aligns the queries with the final positions
of the sequence.

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

### Composing Masks
:label:`subsec_composing-masks`

When multiple constraints apply simultaneously, a pair is valid only if every
constraint permits it, so boolean masks compose via logical AND (`&`) under
broadcasting (:numref:`fig_mask-algebra`).

![Masks compose by logical AND. For a sequence of length 6 with valid length 3, the padding mask stores one row of keys, which broadcasting repeats for every query (faint); the causal mask is a lower triangle; their AND allows a pair only if both do. Rows are queries, columns are keys, and filled entries are allowed.](../img/mdl-attention-mask-algebra.svg)
:label:`fig_mask-algebra`

Combining a padding mask (valid lengths $6$ and $3$) with a causal mask on
equal scores produces a full causal triangle for the first sequence and
truncates all rows of the second sequence at key $3$:

```{.python .input #attention-scoring-composing-masks}
%%tab pytorch
n = 6
padding = padding_mask(torch.tensor([6, 3]), n)  # (batch, 1, key)
causal = causal_mask(n)                          # (query, key)
mask = padding & causal                          # (batch, query, key)
weights = masked_softmax(torch.zeros(2, n, n), mask)  # Equal scores
d2l.show_heatmaps(weights[None], xlabel='keys', ylabel='queries',
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
d2l.show_heatmaps(weights[None], xlabel='keys', ylabel='queries',
                  titles=['valid length 6', 'valid length 3'],
                  figsize=(5, 2.5))
```

The same boolean algebra handles *packed sequences*, where multiple short
documents are concatenated into a single row to avoid padding overhead.
ANDing a same-document indicator `doc[:, None] == doc[None, :]` with
`causal_mask` produces a block-diagonal causal mask:

```{.python .input #attention-scoring-composing-masks-1}
%%tab pytorch
doc = torch.tensor([0, 0, 0, 1, 1, 2])  # Document of each position
same_doc = doc[:, None] == doc[None, :]
packed = same_doc & causal_mask(6)
d2l.show_heatmaps(packed.float()[None, None], xlabel='keys',
                  ylabel='queries', cbar=False)
```

```{.python .input #attention-scoring-composing-masks-1}
%%tab jax
doc = jnp.array([0, 0, 0, 1, 1, 2])  # Document of each position
same_doc = doc[:, None] == doc[None, :]
packed = same_doc & causal_mask(6)
d2l.show_heatmaps(packed.astype(jnp.float32)[None, None], xlabel='keys',
                  ylabel='queries', cbar=False)
```

## Attention for a Minibatch

Stacking $n$ queries into $\mathbf{Q} \in \mathbb{R}^{n \times d}$, $m$ keys
into $\mathbf{K} \in \mathbb{R}^{m \times d}$, and $m$ values into
$\mathbf{V} \in \mathbb{R}^{m \times d_v}$ expresses scaled dot-product
attention as two matrix multiplications and a row-wise softmax
(:numref:`fig_attention-shapes`):

$$\mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n \times d_v}.$$
:eqlabel:`eq_softmax_QK_V`

![Shapes in scaled dot-product attention. $\mathbf{Q}\mathbf{K}^\top/\sqrt{d}$ holds the scores of all $n \times m$ query--key pairs; masking and a softmax over each row turn them into the weights $\mathbf{A}$ (here with a causal mask, which leaves the upper triangle empty); multiplying by $\mathbf{V}$ gives one weighted sum of values per query.](../img/mdl-attention-shapes.svg)
:label:`fig_attention-shapes`

For batched inputs with leading batch (and head) axes, the batched matrix
product `@` operates over the last two dimensions and broadcasts leading axes
automatically. For one head of GPT-2 small ($n = m = 1{,}024$, $d = d_v =
64$), the score matrix holds $1{,}048{,}576$ floats (4 MiB in fp32) and the
two products require $4n^2 d \approx 0.27$ GFLOPs—both quadratic in $n$.

### The DotProductAttention Class
:label:`subsec_dot-product-attention-class`

`DotProductAttention` implements :eqref:`eq_softmax_QK_V` with optional
masking and attention-weight dropout (default $0.0$), storing the most recent
attention weights in `attention_weights` for inspection.

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

Evaluating `DotProductAttention` on the vectors from
:numref:`subsec_attention-worked-example` reproduces our hand calculation:

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

To see what unprojected content matching can and cannot distinguish, assign
each distinct letter in `"attention"` and `"translation"` a random standard
normal embedding $\mathbf{e} \in \mathbb{R}^{32}$ and let the letters of
`"attention"` query the letters of `"translation"`. Identical letters have
expected scaled score $\mathbb{E}[\|\mathbf{e}\|^2]/\sqrt{32} = \sqrt{32}
\approx 5.7$, whereas distinct letters have expected score $0$ and unit
variance.

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

Three behaviors stand out:

1. Unique matches (`i`, `o`) concentrate nearly all weight on their partner.
2. Repeated matches (`a`, `t`, `n`) split their weight equally across every
   occurrence of the letter in `"translation"`, because content-only keys
   carry no positional identity.
3. Unmatched queries (`e`) spread their probability mass across random keys,
   because softmax weights must sum to one even when no relevant key exists
   (in trained language models, the initial token often absorbs such unused
   mass as an *attention sink*, :cite:`Xiao.Tian.Chen.ea.2024`).

## Attention as Learned Alignment
:label:`subsec_learned-alignment`

Learned attention originated in neural machine translation
:cite:`Bahdanau.Cho.Bengio.2014` to relieve the fixed-vector bottleneck of
sequence-to-sequence RNNs (:numref:`sec_seq2seq`). At each decoding step, the
decoder used its previous hidden state as a query against all encoder states
(which served as both keys and values), scoring pairs with the additive
network :eqref:`eq_additive-attn`. Without any direct alignment supervision,
the learned attention weights spontaneously recovered word alignments between
source and target languages—following a monotonic diagonal where word order
agreed and producing off-diagonal crossings where word order differed (such as
English *"black cat"* versus French *"chat noir"*).

To move from cross-sentence alignment to general sequence modeling, however,
three extensions are needed: queries must be derived from the sequence itself,
a single position must be able to retrieve multiple distinct values in
parallel (:numref:`sec_multihead-attention`), and positional encodings must
generalize across sequence lengths (:numref:`sec_positional-information`).

## Summary

* Scaled dot-product scoring $a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top
  \mathbf{k} / \sqrt{d}$ drops the norm penalties of the Gaussian kernel and
  computes all pairwise scores via a single matrix multiplication
  $\mathbf{Q}\mathbf{K}^\top$.
* Dividing by $\sqrt{d}$ preserves unit score variance for unit-variance
  inputs, preventing softmax saturation and vanishing score gradients as head
  width $d$ increases.
* Boolean masks exclude invalid query--key pairs (such as padding and future
  tokens) and compose via logical AND under broadcasting; filling excluded
  scores with `finfo(dtype).min` avoids float16 overflow and `NaN` rows.
* Content-only keys split attention equally across duplicate tokens, whereas
  position-aware keys enable end-to-end learning of sequence alignments.

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
    1. Compute the norm for weights spread uniformly over $k$ of the $m$
       keys, find the $k$ that maximizes it, and confirm by sampling random
       weight vectors that none exceeds this maximum. Use the result to
       explain why the unscaled norm in the table rises from $d = 4$ to
       $d = 16$ and then falls.
   :exlabel:`ex_attention-scoring-softmax-jacobian`

1. **Fully masked rows.** `masked_softmax` overwrites excluded scores with
   the most negative finite value of the dtype and then zeroes the rows of
   queries that have no valid key.
    1. Build a row of four scores in which every key is excluded. Fill it
       with $-10^6$ in float16, with $-\infty$ in float32, and with
       `finfo(dtype).min` in float32, and apply the framework's softmax to
       each, without the final zeroing of `masked_softmax`. Report what you
       get and trace each result through the subtraction of the row
       maximum.
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
   :exlabel:`ex_attention-scoring-composite-masks`

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
   :exlabel:`ex_attention-scoring-entropy-of-random-scores`

1. [code] **Additive scores near zero.** ● The additive score
   :eqref:`eq_additive-attn` adds the projected query and the projected key
   inside a $\tanh$.
    1. Replace $\tanh$ by the identity and show that the attention weights
       then do not depend on the query.
    1. Expand $\tanh$ to third order around zero, identify the lowest-order
       terms through which the weights depend on the query, and state how
       this dependence scales when $\mathbf{W}_q$ and $\mathbf{W}_k$ are
       both multiplied by a small factor $\epsilon$.
    1. Implement the additive score for $3$ queries and $6$ keys in
       $\mathbb{R}^8$ with standard normal entries and a hidden width
       $r = 16$, drawing $\mathbf{W}_q$, $\mathbf{W}_k$, and $\mathbf{w}_v$
       with standard normal entries divided by the square root of their
       input width. Multiply $\mathbf{W}_q$ and $\mathbf{W}_k$ by
       $\epsilon \in \{1, 0.3, 0.1, 0.03\}$, measure the largest difference
       between the weight rows of the three queries, averaged over many
       draws, and check the scaling you derived. For which of these scales
       do the weights depend appreciably on the query?
   :exlabel:`ex_attention-scoring-additive-near-zero`

[Discussions](https://d2l.discourse.group/t/36769)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.2]{.kicker}

Attention Scoring and Masking<br>
**scaled dot products · the 1/√d factor · masks and their algebra · attention as two matrix products · a learned alignment**
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
- compute attention for three keys by hand, then as two matrix products;
- read an alignment that a model learns from its outputs alone.

**Running example:** one GPT-2-small head, $d = 64$, context
$n = 1{,}024$.
:::

::: {.slide title="Learned Scores and Masks Turn Lookup into a Layer"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| **Attention Scoring and Masking** | **scaled dot products, masks** |
| Self-Attention and Multiple Heads | a sequence reads itself; several heads |
| Positional Information | where a token sits |
| The Cost of Attention | quadratic cost, fused kernels |
| Windowed and Linear Attention | linear cost, and what it gives up |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** soft lookup, softmax, padding of minibatches.
- **Skips:** self-attention and heads (next deck), positions.
:::
:::
:::

::: {.slide title="The Gaussian Kernel Is a Dot Product Plus Two Norms"}
$$-\tfrac{1}{2}\|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i - \tfrac{1}{2}\|\mathbf{k}_i\|^2 - \tfrac{1}{2}\|\mathbf{q}\|^2$$

- $\|\mathbf{q}\|^2$ is the same for every key: the softmax removes it.
- $\|\mathbf{k}_i\|^2$ drops out when all keys have equal norm.
- Keeping only $\mathbf{q}^\top\mathbf{k}_i$ is a modeling choice: a long
  key now draws weight from every query pointing its way.

::: {.d2l-note}
The dot product turns the key norm from a penalty into a prior.
:::
:::

::: {.slide title="Additive Scores Fill an n × m × r Array; Dot Products Are One Matrix Product"}
The first attention models scored with a one-hidden-layer network:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}), \quad \mathbf{W}_q \in \mathbb{R}^{r \times d_q}, \ \mathbf{W}_k \in \mathbb{R}^{r \times d_k}$$

- $n$ queries, $m$ keys: about $nmr$ operations, but in an
  $n \times m \times r$ array of $\tanh$ activations.
- Dot products of all pairs: one matrix product $\mathbf{Q}\mathbf{K}^\top$.
- Keep the projections, drop $\tanh$: $\mathbf{q}^\top\mathbf{W}_q^\top\mathbf{W}_k\mathbf{k}$
  is still learned, and still one matrix product.
- Small $\mathbf{W}_q, \mathbf{W}_k$: $\tanh$ is almost linear,
  $\mathbf{w}_v^\top\mathbf{W}_q\mathbf{q}$ is the same for every key, the
  softmax removes it, and the weights barely depend on the query.
:::

::: {.slide title="At d = 64 a Typical Unscaled Score Is ±8"}
Entries of $\mathbf{q}, \mathbf{k} \in \mathbb{R}^d$ independent, mean 0,
variance 1:

$$\textrm{Var}(\mathbf{q}^\top \mathbf{k}) = \sum_{j=1}^d \textrm{Var}(q_j k_j) = d$$

- $d = 64$: standard deviation $8$.
- Divide by $\sqrt{d}$: variance 1 at every width.

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k} / \sqrt{d}$$

::: {.d2l-note}
$1/\sqrt{d}$ is the kernel's bandwidth knob, set by the width instead of by
hand.
:::
:::

::: {.slide title="Two Queries, Three Keys, by Hand"}
[Worked example]{.kicker}

$d = 4$, so scale by $1/2$. $\mathbf{q}_1 = (2,2,0,0)$,
$\mathbf{q}_2 = (0,0,2,0)$; keys $(1,1,0,0)$, $(0,0,1,1)$, $(0,0,1,-1)$;
values $(1,0)$, $(0,1)$, $(1,1)$.

| | scores | scaled | weights | output |
|---|---|---|---|---|
| $\mathbf{q}_1$ | $(4, 0, 0)$ | $(2, 0, 0)$ | $(0.787, 0.107, 0.107)$ | $(0.893, 0.213)$ |
| $\mathbf{q}_2$ | $(0, 2, 2)$ | $(0, 1, 1)$ | $(0.155, 0.422, 0.422)$ | $(0.578, 0.845)$ |

- A perfect match still leaves a fifth of the weight elsewhere.
- Query 2 is zero where keys 2 and 3 differ: it splits its weight equally.

::: {.d2l-note}
Without the factor $1/2$, query 1 would put $0.96$ on key 1: the same
vectors, a sharper choice.
:::
:::

::: {.slide title="Unscaled Scores Collapse the Softmax Entropy"}
2,048 random queries, 64 keys each; entropy of the weights:

@!attention-scoring-score-variance-and-the-1-sqrt-d-factor-1

- Uniform over 64 keys: 4.16 nats. Scaled: about 3.7 at every $d$, as
  even as a uniform choice among about 40 keys.
- Unscaled: about 0.2 nats at $d = 512$, barely more than one key.
:::

::: {.slide title="Saturation Shrinks the Gradient That Reaches the Scores"}
Scores $\mathbf{a}$ of one query, weights $\boldsymbol{\alpha} = \mathrm{softmax}(\mathbf{a})$:

$$\frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{a}} = \mathrm{diag}(\boldsymbol{\alpha}) - \boldsymbol{\alpha}\boldsymbol{\alpha}^\top \to \mathbf{0} \quad \textrm{as } \boldsymbol{\alpha} \to \textrm{one-hot}$$

@!attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2

- Scaled: largest weight 0.11, Jacobian norm 0.18 at every $d$.
- Unscaled: the norm rises to $d = 16$, drops below the scaled value by
  $d = 128$, and is about half of it at $d = 512$.
:::

::: {.slide title="Check: Two Keys One Standard Deviation Apart"}
[Check]{.kicker}

In a $d = 64$ head, two keys score one standard deviation apart. What is
the ratio of their weights without and with the factor $1/\sqrt{d}$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Unscaled, the scores differ by 8: ratio $e^8 \approx 3{,}000$.
Scaled, they differ by 1: ratio $e \approx 2.7$. Without the factor and
with unit-variance queries and keys, one key takes most of the weight
(about 80% over 64 random keys).
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
| $-10^6$ | weight 0 in float32, bfloat16; float16 overflows or refuses | uniform; float16: NaN or an error |
| $-\infty$ | weight 0 | NaN: $-\infty - (-\infty)$ |
| `finfo(dtype).min` | weight 0 at any precision | uniform, then set to zero |

::: {.d2l-note}
float16 stops at $6.5 \times 10^4$; bfloat16 has float32's range.
`finfo(dtype).min` is right without knowing which dtype is in use.
:::
:::

::: {.slide title="The Masked Softmax Fills, Normalizes, and Zeroes Empty Rows"}
@attention-scoring-the-masked-softmax-operation-1

The mask only has to broadcast against the scores
(…, queries, keys).
:::

::: {.slide title="Equal Scores Make the Masked Weights Predictable"}
Two sequences, two queries, four keys, every score 0; valid lengths 2
and 3. Prediction: rows $(\tfrac12, \tfrac12, 0, 0)$ and
$(\tfrac13, \tfrac13, \tfrac13, 0)$.

@attention-scoring-the-masked-softmax-operation-2

- The padding mask has shape (batch, 1, keys), broadcast over queries.
:::

::: {.slide title="Masks Compose by AND: Padding and Causal on Equal Scores"}
::: {.cols .vc}
::: {.col}
![](../img/mdl-attention-mask-algebra.svg){width=100%}
:::

::: {.col .fig}
@!attention-scoring-composing-masks
:::
:::

- On equal scores, causal row $i$ holds $1/(i+1)$ on keys $0, \ldots, i$:
  $n(n-1)/2$ of $n^2$ pairs excluded. Valid length 3 stops rows at the
  third key.
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
$\mathbf{Q} \in \mathbb{R}^{n \times d}$: one query per row;
$\mathbf{K} \in \mathbb{R}^{m \times d}$; $\mathbf{V} \in \mathbb{R}^{m \times d_v}$.

$$\mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times d_v}$$

![Shapes of Q, K, the scores, the masked weights, V, and the output.](../img/mdl-attention-shapes.svg){width=48%}

::: {.d2l-note}
Row $i$ of the output is what query $i$ alone would produce; the worked
example is two rows of it.
:::
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

::: {.slide title="The Layer Is Three Lines" only="pytorch"}
The core of `DotProductAttention.forward`:

```{.python #attention-scoring-exercises-1}
# queries (..., n, d), keys (..., m, d), values (..., m, d_v)
scores = queries @ keys.transpose(-2, -1) / math.sqrt(d)
weights = masked_softmax(scores, mask)  # (..., n, m)
output = dropout(weights) @ values      # (..., n, d_v)
```

- Leading axes (batch, later heads) are carried along by `@`.
- Attention dropout defaults to 0, as in most current language models.
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
- Attention dropout defaults to 0, as in most current language models.
:::

::: {.slide title="The Class Reproduces the Hand Computation"}
The worked example, through `DotProductAttention`:

@attention-scoring-the-dotproductattention-class-2

- The same weights and outputs as by hand, to the last printed digit.
:::

::: {.slide title="Equal Letters Score √32 ≈ 5.7, So We Know the Answer"}
Random letter embeddings of width 32; "attention" queries "translation":

@!multihead-attention-an-alignment-you-can-read

- `i`, `o`: one partner, most of the weight.
- `a`, `t`, `n`: two equal keys, weight split; content cannot tell copies
  apart.
- `e`: no partner; the softmax spends its mass anyway.
:::

::: {.slide title="Keys That Carry Positions Let a Toy Model Learn a Permutation"}
In translation (Bahdanau et al.) the weights came out as word alignments.
Toy: output $j$ copies source $\pi(j)$, 8 tokens from 16, one pair swapped.
Queries: one learned vector per output; keys: $\mathbf{W}_k$[one-hot token;
one-hot position]; values: one-hot tokens.

@!attention-scoring-attention-as-learned-alignment-4

- Untrained: a fifth to a quarter right, weight on $\pi(j)$ near $1/8$.
  Trained (three runs): accuracy 1.000, largest weight on $\pi(j)$.
- Repeated tokens are told apart: the keys carry the position.
:::

::: {.slide title="Check: A Toy Without Positions"}
[Check]{.kicker}

Remove the one-hot position from the toy's keys. Can output $j$ still put
its weight on source position $\pi(j)$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. Equal tokens then have equal keys, and every position with
the same token gets the same weight, as with the letters of
"attention". The output can depend only on which tokens occur, not where.
:::
:::
:::

::: {.slide title="Scaled Scores and Boolean Masks Make Attention a Trainable Layer"}
[Wrap-up]{.kicker}

- Score $\mathbf{q}^\top\mathbf{k}/\sqrt{d}$: the Gaussian kernel without
  the norms; at $d = 64$ the factor turns a ±8 score into ±1.
- Unscaled scores of random vectors saturate the softmax: about 0.2 nats
  of entropy at $d = 512$ instead of 3.7.
- Masks are booleans that compose by AND; fill with the dtype's most
  negative finite value; empty rows become zeros.
- One GPT-2 head at 1,024 tokens: 1M scores, 4 MiB, 0.27 GFLOP.
- Equal keys split the weight; keys with positions learn an alignment.

**Next:** a sequence that queries itself, and several heads.
:::
