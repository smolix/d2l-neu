# Positional Information
:label:`sec_positional-information`

Without positional information, attention is permutation equivariant: reordering
the input only reorders the output. This is unsuitable for sequences in which
order changes meaning. This section proves the equivariance property and then
studies four ways to represent position: sinusoidal and learned absolute
embeddings, rotary position embeddings (RoPE), linear attention biases (ALiBi),
and causal attention without explicit position embeddings (NoPE). We compare
the methods both within the training context length and beyond it using a small
character-level language model whose only cross-position operation is
attention.

```{.python .input #positional-information}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #positional-information}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
```

## Attention Ignores Order

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ hold a sequence of $n$ token
representations. At evaluation time, with dropout disabled and with no mask
or position-dependent bias, let $f$ be the self-attention layer

$$
f(\mathbf{X}) = \mathrm{softmax}\!\left(\frac{\mathbf{X}\mathbf{W}_q (\mathbf{X}\mathbf{W}_k)^\top}{\sqrt{d}}\right) \mathbf{X}\mathbf{W}_v.
$$
:eqlabel:`eq_pos-selfattn`

**Proposition (permutation equivariance).** For every permutation matrix
$\boldsymbol{\Pi}$, $\;f(\boldsymbol{\Pi} \mathbf{X}) = \boldsymbol{\Pi}
f(\mathbf{X})$.

**Proof.** The projections act on each row separately, so
$(\boldsymbol{\Pi}\mathbf{X})\mathbf{W} = \boldsymbol{\Pi}(\mathbf{X}\mathbf{W})$
for each of $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$. The score matrix
becomes $\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top$, and since softmax
normalizes each row by itself, it commutes with row and column permutations:
$\mathrm{softmax}(\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top) =
\boldsymbol{\Pi}\, \mathrm{softmax}(\mathbf{S})\, \boldsymbol{\Pi}^\top$.
Multiplying by the permuted values,
$\boldsymbol{\Pi}\, \mathrm{softmax}(\mathbf{S})\, \boldsymbol{\Pi}^\top
\boldsymbol{\Pi} \mathbf{X}\mathbf{W}_v = \boldsymbol{\Pi} f(\mathbf{X})$
because $\boldsymbol{\Pi}^\top \boldsymbol{\Pi} = \mathbf{I}$. Concatenating
heads and applying $\mathbf{W}_o$ row-wise preserves the property, so it
holds for multi-head attention too. $\blacksquare$

Permuting the input applies the same permutation to the output: each token's
output is unchanged except for its row index. Thus the layer does not use a
token's position in the sequence. We can verify this property
on the multi-head layer of :numref:`sec_multihead-attention`:

```{.python .input #positional-information-attention-ignores-order}
%%tab pytorch
torch.manual_seed(0)
attention = d2l.MultiHeadAttention(num_hiddens=64, num_heads=4, dropout=0)
X = torch.randn(1, 8, 64)
perm = torch.randperm(8)
Y, Y_perm = attention(X, X, X, None), attention(X[:, perm], X[:, perm],
                                                X[:, perm], None)
print(f'max |Y[perm] - Y_perm|: {(Y[:, perm] - Y_perm).abs().max():.2e}')
```

```{.python .input #positional-information-attention-ignores-order}
%%tab jax
attention = d2l.MultiHeadAttention(num_hiddens=64, num_heads=4, dropout=0)
X = jax.random.normal(jax.random.key(0), (1, 8, 64))
perm = jax.random.permutation(jax.random.key(1), 8)
Y = attention(X, X, X, None)[0]
Y_perm = attention(X[:, perm], X[:, perm], X[:, perm], None)[0]
print(f'max |Y[perm] - Y_perm|: {jnp.abs(Y[:, perm] - Y_perm).max():.2e}')
```

Two caveats qualify this result. First, equivariance is a property
of the *layer*; stacking equivariant layers and reading out per token leaves
the model equivariant, so depth does not help. Second, the proposition is
about unmasked attention: a causal mask singles out each position by how many
predecessors it may see, which breaks the symmetry. NoPE uses this asymmetry,
as discussed near the end of the section. The other methods provide each
token with explicit positional information.

## Absolute Position Embeddings

The original transformer attaches a *position vector* $\mathbf{p}_i \in
\mathbb{R}^d$ to each position $i$ and feeds $\mathbf{x}_i + \mathbf{p}_i$
into the first layer :cite:`Vaswani.Shazeer.Parmar.ea.2017`. Two tokens with
identical embeddings at different positions now enter attention as different
vectors, and the equivariance proof fails at its first step. The two classic
choices of $\mathbf{P}$ differ in whether the table is designed or learned.

### Sinusoidal Encodings

The designed choice fills row $i$ of $\mathbf{P} \in \mathbb{R}^{n \times d}$
with sines and cosines of geometrically spaced frequencies,

$$
p_{i, 2j} = \sin\left(\frac{i}{10000^{2j/d}}\right), \qquad p_{i, 2j+1} = \cos\left(\frac{i}{10000^{2j/d}}\right).
$$
:eqlabel:`eq_sinusoidal-def`

The design looks arbitrary until you plot it. We implement the table as a
small function and look at four adjacent columns:

```{.python .input #positional-information-sinusoidal-encodings-1}
%%tab pytorch
def sinusoidal_encoding(max_len, num_hiddens):
    theta = torch.arange(max_len)[:, None] / 10000 ** (
        torch.arange(0, num_hiddens, 2) / num_hiddens)
    return torch.stack([torch.sin(theta), torch.cos(theta)],
                       -1).reshape(max_len, num_hiddens)

P = sinusoidal_encoding(60, 32)
d2l.plot(torch.arange(60), P[:, 6:10].T, xlabel='position',
         figsize=(6, 2.5), legend=[f'column {d}' for d in range(6, 10)])
```

```{.python .input #positional-information-sinusoidal-encodings-1}
%%tab jax
def sinusoidal_encoding(max_len, num_hiddens):
    theta = jnp.arange(max_len)[:, None] / 10000 ** (
        jnp.arange(0, num_hiddens, 2) / num_hiddens)
    return jnp.stack([jnp.sin(theta), jnp.cos(theta)],
                     -1).reshape(max_len, num_hiddens)

P = sinusoidal_encoding(60, 32)
d2l.plot(jnp.arange(60), P[:, 6:10].T, xlabel='position',
         figsize=(6, 2.5), legend=[f'column {d}' for d in range(6, 10)])
```

Adjacent column pairs share a frequency (one sine, one cosine), and the
frequency falls as the column index grows. This is a continuous version of
something familiar: binary counting. In the binary representations below, the
lowest bit flips every number, the next every two numbers, the next every
four. Fast dimensions distinguish neighboring positions, whereas slow
dimensions distinguish broader regions:

```{.python .input #positional-information-sinusoidal-encodings-2}
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

The heatmap of the whole table shows the same structure in continuous form:
each row is a unique multi-frequency fingerprint of its position, and unlike
bits, the values vary smoothly, so nearby positions get nearby encodings.

```{.python .input #positional-information-sinusoidal-encodings-3}
%%tab pytorch
d2l.show_heatmaps(P[None, None], xlabel='column (encoding dimension)',
                  ylabel='row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input #positional-information-sinusoidal-encodings-3}
%%tab jax
d2l.show_heatmaps(P[None, None], xlabel='column (encoding dimension)',
                  ylabel='row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Learned Positions

The empirical choice is to make $\mathbf{P}$ a trainable embedding table, one
free vector per position, as in BERT and GPT-2
:cite:`Devlin.Chang.Lee.ea.2018,Radford.Wu.Child.ea.2019`. It concedes that
we do not know the right encoding and lets gradient descent find one. The
limitation is apparent before training: the table has
exactly as many rows as the training context, and a row that no training
example ever used is still whatever initialization left there. Position 500
is not "a bit beyond position 128"; it is undefined.

### Sinusoidal Encodings as Rotations

Designed sinusoids have a relative-position structure absent from a learned
table. The original paper observed that "for any fixed offset $\delta$,
$\mathbf{p}_{i+\delta}$ is a linear function of $\mathbf{p}_i$". The
linear map can be written explicitly. Write $\omega_j = 1/10000^{2j/d}$ for the
frequency shared by columns $2j$ and $2j{+}1$. Then

$$
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
= \begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}
= \begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
$$
:eqlabel:`eq_sinusoidal-rotation`

by the angle-addition identities. The map from position $i$ to position
$i + \delta$ is a *rotation* of each two-column pair, by an angle $\delta
\omega_j$ that depends only on the offset $\delta$, never on $i$. Thus the encoding contains relative-position structure, but only implicitly.
The model receives $\mathbf{x}_i + \mathbf{p}_i$ and must learn projections
that exploit the rotations. Moreover, a query--key product of two such sums
expands into four terms, and only the position--position term retains the
rotation directly. RoPE instead applies relative rotations directly to the
query--key comparison.

## Rotary Position Embeddings

This construction is used by many current open-weights language models,
including Llama, Qwen, and DeepSeek. *Rotary position embeddings* (RoPE) :cite:`Su.Lu.Pan.ea.2021` skip the
addition and instead act on queries and keys at the point where they meet:
the score. We want a transformation $\mathbf{R}_i$, applied to the query at
position $i$ and the key at position $j$, such that the score depends only on
the content vectors and the offset $j - i$:

$$
(\mathbf{R}_i \mathbf{q})^\top (\mathbf{R}_j \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_i^\top \mathbf{R}_j \mathbf{k} \overset{!}{=} \mathbf{q}^\top \mathbf{R}_{j-i} \mathbf{k}.
$$
:eqlabel:`eq_rope-goal`

The requirement $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$ says that
the $\mathbf{R}_i$ form a one-parameter group of orthogonal maps. Planar
rotations satisfy this condition, so RoPE reuses the sinusoidal
frequencies $\omega_m$, but *multiplicatively*: split the $d$ query
dimensions into $d/2$ pairs, and rotate pair $m$ of the position-$i$ query by
the angle $i\,\omega_m$ (and likewise for keys),

$$
\mathbf{R}_i = \begin{bmatrix} R(i\omega_0) & & \\ & \ddots & \\ & & R(i\omega_{d/2-1}) \end{bmatrix}, \qquad R(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}.
$$
:eqlabel:`eq_rope-def`

:numref:`fig_rope-rotation` shows one pair plane: shifting both positions by
the same amount rotates query and key together, preserving their angle and
dot product. The score therefore depends on position only through the offset.
The construction has two additional properties: rotations
preserve norms, so RoPE never inflates or shrinks a token's content, and the
geometric frequency ladder means low-frequency pairs rotate only slightly
across a sentence while high-frequency pairs discriminate neighboring
positions sharply. Every pair still carries content; the frequency ladder
only controls how quickly position rotates it. Only queries and keys are rotated; values pass
through untouched. (Learned *relative*-position embeddings
:cite:`shaw2018self,huang2018music` pursued the same goal by adding trained
offset vectors into the score; RoPE achieves relative scoring without
additional learned parameters.)

![Rotary embeddings rotate each two-dimensional feature pair of the query and the key by an angle proportional to position. Shifting both positions by the same amount — here by 3 — rotates both vectors together and leaves the angle between them, and hence the attention score, unchanged. Here $\theta$ is the pair's per-position angle and $\varphi$ the angle between the unrotated query and key.](../img/mdl-attention-rope-rotation.svg)
:label:`fig_rope-rotation`

The implementation rotates the even/odd feature pairs with precomputed
sines and cosines; `offset` shifts every position by the same amount, which
we will use to test the invariance:

```{.python .input #positional-information-rotary-position-embeddings-1}
%%tab pytorch
def rope(x, offset=0):
    """Rotate feature pairs of x (..., num_steps, d) by position angles."""
    d = x.shape[-1]
    pos = torch.arange(x.shape[-2], dtype=torch.float32,
                       device=x.device) + offset
    inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device) / d)
    theta = pos[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], -1).flatten(-2)
```

```{.python .input #positional-information-rotary-position-embeddings-1}
%%tab jax
def rope(x, offset=0):
    """Rotate feature pairs of x (..., num_steps, d) by position angles."""
    d = x.shape[-1]
    pos = jnp.arange(x.shape[-2], dtype=jnp.float32) + offset
    inv_freq = 10000.0 ** (-jnp.arange(0, d, 2) / d)
    theta = pos[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], -1).reshape(x.shape)
```

A numerical check of :eqref:`eq_rope-goal` shifts all positions jointly.
The full RoPE score matrix should remain unchanged. For contrast, we run the
same test on the additive sinusoidal encoding, where the four-term expansion
leaves absolute position in the score:

```{.python .input #positional-information-rotary-position-embeddings-2}
%%tab pytorch
torch.manual_seed(0)
T, d = 32, 64
q, k = torch.randn(T, d), torch.randn(T, d)

scores = lambda off: rope(q, off) @ rope(k, off).T / math.sqrt(d)
for off in (1, 17, 480):
    print(f'RoPE, all positions shifted by {off:>3}: max score change '
          f'{(scores(off) - scores(0)).abs().max():.1e}')

P = sinusoidal_encoding(600, d)
added = lambda off: (q + P[off:off+T]) @ (k + P[off:off+T]).T / math.sqrt(d)
print(f'additive sinusoidal, shifted by  17: max score change '
      f'{(added(17) - added(0)).abs().max():.1e}')
```

```{.python .input #positional-information-rotary-position-embeddings-2}
%%tab jax
T, d = 32, 64
key1, key2 = jax.random.split(jax.random.key(0))
q, k = jax.random.normal(key1, (T, d)), jax.random.normal(key2, (T, d))

scores = lambda off: rope(q, off) @ rope(k, off).T / math.sqrt(d)
for off in (1, 17, 480):
    print(f'RoPE, all positions shifted by {off:>3}: max score change '
          f'{jnp.abs(scores(off) - scores(0)).max():.1e}')

P = sinusoidal_encoding(600, d)
added = lambda off: (q + P[off:off+T]) @ (k + P[off:off+T]).T / math.sqrt(d)
print(f'additive sinusoidal, shifted by  17: max score change '
      f'{jnp.abs(added(17) - added(0)).max():.1e}')
```

RoPE's scores are unchanged up to floating-point round-off (three or more
orders of magnitude below the score scale in these runs) at every shift,
including one far beyond where any additive table would end; the additive
encoding moves the scores by an amount comparable to the scores themselves.
Relative position is therefore encoded by the architecture rather than
learned from data. This property does not guarantee accurate predictions at
offsets absent from the training data, which the next experiment tests.

## Extrapolation Beyond the Training Length

Every scheme above fixes permutation blindness at the training length. They
differ when evaluated outside the training range. A length-extrapolation test
trains at context $L$ and compares perplexity at $L$ with perplexity at, for
example, $4L$. Two more schemes were designed for this setting.

### ALiBi and NoPE

*ALiBi* (attention with linear biases) :cite:`Press.Smith.Lewis.2022`
encodes no positions at all. It instead subtracts a distance penalty from
every attention score: for query position $i$ and key position $j \leq i$
in head $h$ of $H$,

$$
\mathrm{score}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} - m_h\,(i - j), \qquad m_h = 2^{-8h/H}.
$$
:eqlabel:`eq_alibi-def`

Each head discounts distant tokens at its own geometric rate. Head 1 uses a
steep penalty and head $H$ a much smaller one; the slope formula
$2^{-8h/H}$ is the paper's choice for head counts that are powers of two,
with other $H$ interpolating the same geometric sequence. The bias depends
only on distance, so it is defined for every context length, and a distance
of 400 remains a valid input to the bias rather than an unallocated table
index. The cost is a fixed
recency prior the model cannot fully unlearn.

*NoPE* (no positional encoding) uses the asymmetry omitted from the
proposition :cite:`Kazemnejad.Padhi.Ramamurthy.ea.2023`: a causal model can
omit explicit position representations. The mask already breaks the symmetry —
position 0 attends over one token, position 100 over a hundred and one — so
the number of tokens competing in each softmax is itself positional
information, and a decoder can in principle recover relative order from it.
Because it uses neither a table nor a rotation, NoPE has no explicit
position range. Its effectiveness depends on how much usable positional
signal the mask supplies.

### An Attention-Only Language Model

We compare extrapolation by training a deliberately minimal model that is
also analyzed later in the chapter: a token embedding, a stack of
residual causal attention blocks, and an output head tied to the embedding
:cite:`Press.Wolf.2017`. There is no feed-forward network and no
normalization layer, and the attention projections carry no bias terms by
default (a `bias` switch restores them;
:numref:`sec_what-attention-computes` has one use for it). Besides the
embedding table, which doubles as the output head, attention is the only trainable cross-position operation. All mixing across
positions therefore passes through attention, which makes the model easier
to analyze in
:numref:`sec_what-attention-computes`. It is the attention-only cousin of
the `TinyLM` of :numref:`subsec_tinylm`, and the positional scheme is a
constructor argument taking `'learned'`, `'sinusoidal'`, `'rope'`,
`'alibi'`, or `'none'`, implemented exactly as in the equations above.

```{.python .input #positional-information-an-attention-only-language-model}
%%tab pytorch
class TinyCharLM(nn.Module):  #@save
    """Attention-only character-level language model."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False):
        super().__init__()
        self.num_heads, self.pos = num_heads, pos
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if pos == 'sinusoidal':
            theta = torch.arange(max_len)[:, None] / 10000 ** (
                torch.arange(0, num_hiddens, 2) / num_hiddens)
            P = torch.stack([torch.sin(theta), torch.cos(theta)], -1)
            self.register_buffer('P', P.reshape(max_len, num_hiddens))
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(num_hiddens, 3 * num_hiddens, bias=bias),
            proj=nn.Linear(num_hiddens, num_hiddens, bias=bias)))
            for _ in range(num_blks)])

    def _rope(self, x):
        d = x.shape[-1]
        pos = torch.arange(x.shape[-2], dtype=torch.float32, device=x.device)
        inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device) / d)
        theta = pos[:, None] * inv_freq[None, :]
        cos, sin = torch.cos(theta), torch.sin(theta)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], -1).flatten(-2)

    def _alibi(self, T, device):
        h = torch.arange(1, self.num_heads + 1, device=device)
        slopes = 2.0 ** (-8.0 * h / self.num_heads)
        pos = torch.arange(T, device=device, dtype=torch.float32)
        return slopes[:, None, None] * (pos[None, :] - pos[:, None])

    def _attend(self, blk, H):
        B, T, D = H.shape
        q, k, v = blk['qkv'](H).chunk(3, -1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in (q, k, v))
        if self.pos == 'rope':
            q, k = self._rope(q), self._rope(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if self.pos == 'alibi':
            scores = scores + self._alibi(T, H.device)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                     device=H.device), 1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return blk['proj'](out), weights

    def _embed(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
        if self.pos == 'sinusoidal':
            H = H + self.P[:X.shape[1]]
        return H

    def forward(self, X):
        H = self._embed(X)
        for blk in self.blks:
            out, _ = self._attend(blk, H)
            H = H + out
        return F.linear(H, self.token_emb.weight)  # Tied output head

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        H, maps = self._embed(X), []
        for blk in self.blks:
            out, weights = self._attend(blk, H)
            maps.append(weights)
            H = H + out
        return maps
```

```{.python .input #positional-information-an-attention-only-language-model}
%%tab jax
class TinyCharLM(nnx.Module):  #@save
    """Attention-only character-level language model."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_heads, self.pos = num_heads, pos
        init = nnx.initializers.normal(0.02)
        self.token_emb = nnx.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        if pos == 'sinusoidal':
            theta = jnp.arange(max_len)[:, None] / 10000 ** (
                jnp.arange(0, num_hiddens, 2) / num_hiddens)
            P = jnp.stack([jnp.sin(theta), jnp.cos(theta)], -1)
            self.P = nnx.Cache(P.reshape(max_len, num_hiddens))
        self.blks = nnx.List([nnx.Dict(
            qkv=nnx.Linear(num_hiddens, 3 * num_hiddens, use_bias=bias,
                           rngs=rngs),
            proj=nnx.Linear(num_hiddens, num_hiddens, use_bias=bias,
                            rngs=rngs))
            for _ in range(num_blks)])

    def _rope(self, x):
        d = x.shape[-1]
        pos = jnp.arange(x.shape[-2], dtype=jnp.float32)
        inv_freq = 10000.0 ** (-jnp.arange(0, d, 2) / d)
        theta = pos[:, None] * inv_freq[None, :]
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return jnp.stack([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], -1).reshape(x.shape)

    def _alibi(self, T):
        h = jnp.arange(1, self.num_heads + 1)
        slopes = 2.0 ** (-8.0 * h / self.num_heads)
        pos = jnp.arange(T, dtype=jnp.float32)
        return slopes[:, None, None] * (pos[None, :] - pos[:, None])

    def _attend(self, blk, H):
        B, T, D = H.shape
        q, k, v = jnp.split(blk['qkv'](H), 3, axis=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
                   for u in (q, k, v))
        if self.pos == 'rope':
            q, k = self._rope(q), self._rope(k)
        scores = q @ k.swapaxes(-2, -1) / math.sqrt(q.shape[-1])
        if self.pos == 'alibi':
            scores = scores + self._alibi(T)
        mask = jnp.triu(jnp.ones((T, T), dtype=bool), 1)
        scores = jnp.where(mask, jnp.finfo(scores.dtype).min, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        out = (weights @ v).swapaxes(1, 2).reshape(B, T, D)
        return blk['proj'](out), weights

    def _embed(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        if self.pos == 'sinusoidal':
            H = H + self.P[:X.shape[1]]
        return H

    def __call__(self, X):
        H = self._embed(X)
        for blk in self.blks:
            out, _ = self._attend(blk, H)
            H = H + out
        return self.token_emb.attend(H)  # Tied output head

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        H, maps = self._embed(X), []
        for blk in self.blks:
            out, weights = self._attend(blk, H)
            maps.append(weights)
            H = H + out
        return maps
```

### The Experiment

We train one fixed-seed copy per scheme on the character-level Time Machine corpus of
:numref:`sec_text-sequence` at context length 128, using the same fixed-step
training loop as :numref:`sec_adam`. Each run takes well under a minute on
one GPU. The learned and sinusoidal tables are allocated out to `max_len`
positions to permit longer evaluation. Only the first 128 rows of the
learned table receive gradients during training.

```{.python .input #positional-information-the-experiment-1}
%%tab pytorch
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)
schemes = ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
models = {}
for pos in schemes:
    torch.manual_seed(0)
    model = TinyCharLM(len(data.vocab), pos=pos)
    losses = d2l.train_lm(model, data,
                          torch.optim.AdamW(model.parameters(), lr=1e-3,
                                            weight_decay=0.0), 3000)
    models[pos] = model
    print(f'{pos:>10}: final training loss {sum(losses[-100:]) / 100:.2f}')
```

```{.python .input #positional-information-the-experiment-1}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)
schemes = ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
models = {}
for pos in schemes:
    model = TinyCharLM(len(data.vocab), pos=pos, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.0),
                              wrt=nnx.Param)
    losses = d2l.train_lm(model, data, optimizer, 3000)
    models[pos] = model
    print(f'{pos:>10}: final training loss {sum(losses[-100:]) / 100:.2f}')
```

Every model above was trained on length-$128$ sequences. We evaluate
validation perplexity at that training
context, $n = 128$, and at two and four times this length ($n = 256$ and
$512$). The first point is in range, while the other two measure
extrapolation on the same held-out text:

```{.python .input #positional-information-the-experiment-2}
%%tab pytorch
def eval_ppl(model, data):
    device = d2l.try_gpu()
    model.to(device).eval()
    losses = []
    with torch.no_grad():
        for X, Y in data.val_dataloader():
            logits = model(X.to(device))
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                Y.to(device).reshape(-1)).item())
    model.train()
    return math.exp(sum(losses) / len(losses))

contexts = (128, 256, 512)
ppls = {pos: [] for pos in schemes}
for ctx in contexts:
    eval_data = d2l.TimeMachine(batch_size=16, num_steps=ctx,
                                tokenization='char', num_train=100000,
                                num_val=3000)
    for pos in schemes:
        ppls[pos].append(eval_ppl(models[pos], eval_data))
for pos in schemes:
    print(f'{pos:>10}: ' + '  '.join(f'{p:7.1f}' for p in ppls[pos]))
d2l.plot(list(contexts), [ppls[pos] for pos in schemes],
         'evaluation context length', 'validation perplexity',
         legend=list(schemes), yscale='log',
         fmts=('-', 'm--', 'g-.', 'r:', 'c-'))
```

```{.python .input #positional-information-the-experiment-2}
%%tab jax
def eval_ppl(model, data):
    @nnx.jit
    def batch_loss(model, X, Y):
        logits = model(X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()
    losses = [float(batch_loss(model, jnp.asarray(X), jnp.asarray(Y)))
              for X, Y in data.val_dataloader()]
    return math.exp(sum(losses) / len(losses))

contexts = (128, 256, 512)
ppls = {pos: [] for pos in schemes}
for ctx in contexts:
    eval_data = d2l.TimeMachine(batch_size=16, num_steps=ctx,
                                tokenization='char', num_train=100000,
                                num_val=3000)
    for pos in schemes:
        ppls[pos].append(eval_ppl(models[pos], eval_data))
for pos in schemes:
    print(f'{pos:>10}: ' + '  '.join(f'{p:7.1f}' for p in ppls[pos]))
d2l.plot(list(contexts), [ppls[pos] for pos in schemes],
         'evaluation context length', 'validation perplexity',
         legend=list(schemes), yscale='log',
         fmts=('-', 'm--', 'g-.', 'r:', 'c-'))
```

For this fixed seed and training protocol, explicit position information
improves perplexity at the training length: RoPE is lowest, followed by the
learned table, while sinusoidal embeddings and ALiBi are similar and NoPE is
higher. This comparison does not isolate the cause of the differences; for
example, describing ALiBi's result as a consequence of its recency bias would
require a separate intervention.

At contexts longer than those seen during training, the learned and
sinusoidal absolute schemes deteriorate. The learned table has untrained rows
beyond position 128, whereas the sinusoidal table supplies vectors outside the
training distribution. RoPE also deteriorates in this run. Although its score
depends on relative offset, offsets beyond 127 did not occur during training,
so relative parameterization alone does not guarantee length extrapolation.
ALiBi and NoPE remain approximately flat in this experiment. Since each curve
comes from one seed in each framework, these observations illustrate the
possible behavior of the schemes; they do not establish a stable ranking or
an estimate of run-to-run variation.

### Extending a Trained Model's Context

At an unseen offset such as 400, RoPE rotates the high-frequency pairs
through angles absent from the training data. Because these angles vary
continuously, they can be *rescaled* so that the deployed range maps into the
trained range. Evaluating at length $4L$ with every angle multiplied by $1/4$
makes position $4\delta$ look like the familiar $\delta$: interpolation
instead of extrapolation. This is *position interpolation*
:cite:`Chen.Wong.Chen.ea.2023`, which (with a brief fine-tune) extended
Llama from 2k to 32k context; YaRN :cite:`Peng.Quesnelle.Fan.ea.2024`
refines it by rescaling the fast, position-discriminating frequencies
differently from the slow, content-carrying ones. Many long-context RoPE
models use schemes from this family. Note
that the recipe has two halves: rescaling *and* a brief fine-tune at the
scaled angles. Exercise 3 walks through both halves on our character model;
in our runs, rescaling alone increases perplexity because compressing the
angle range makes neighboring characters less distinguishable in the
high-frequency pairs. Rescaling followed by a few hundred fine-tuning steps
recovers most of the lost performance.

## Summary

Unmasked attention is permutation equivariant. Absolute position embeddings
break this symmetry by adding a position-dependent vector to each token;
sinusoidal embeddings define the vectors analytically, while learned tables are
learned only at positions used during training. RoPE rotates query and key feature
pairs so their inner product depends on relative offset. ALiBi adds a per-head
linear distance penalty, and NoPE uses only the positional asymmetry supplied by
the causal mask. In the fixed-seed character-model experiment, RoPE has the
lowest perplexity at the training length, while both absolute encodings and
RoPE degrade at four times that length. ALiBi and NoPE are approximately
stable in that run. Position interpolation and YaRN rescale RoPE frequencies
to map longer contexts into the trained range.

## Exercises

1. Causal attention is not permutation equivariant. Demonstrate this
   numerically: apply a causal mask to the shuffle experiment of the first
   section and measure the output difference. Then explain precisely where
   the equivariance proof breaks, and why the *first* position's output is
   nevertheless unchanged by any permutation that fixes it.
2. How much position information leaks through the causal mask? Train a
   linear probe (one linear layer) to predict the position index $i$ from
   the final hidden state $\mathbf{h}_i$ of the trained `'none'` model, and
   compare its accuracy against the same probe on the `'rope'` model.
3. Implement position interpolation: add a `scale` argument to
   `TinyCharLM._rope` that multiplies every angle, and re-evaluate the
   trained `'rope'` model at context 512 with `scale=128/512`. You will
   find that zero-shot rescaling alone does not help this character-level
   model — explain why, considering what the fastest frequency pairs are
   responsible for. Then complete the published recipe: fine-tune for a few
   hundred steps at context 512 with the scaled angles, re-evaluate, and
   compare the result against the ALiBi row of the experiment.
4. The experiment adds the sinusoidal table at its classic full amplitude,
   while the token embeddings are initialized two orders of magnitude
   smaller. Scale the table by 0.02 instead, retrain, and evaluate at all
   three lengths. Explain what the model degenerates into and which row of
   the results table it now resembles.
5. RoPE's base constant sets the frequency ladder. Retrain the `'rope'`
   variant with the base changed from 10000 to 100 and to 1000000, and
   compare both training-length perplexity and extrapolation. Which
   frequencies does each change affect, and why does the base matter more
   at length 512 than at length 128?
6. Visualize `models['rope'].attention_weights(X)` for a validation batch
   and compare with the `'none'` model. Which heads attend locally, which
   uniformly? Reconcile what you see with the two models' perplexity gap at
   the training length.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.4]{.kicker}

Positional information<br>
**permutation blindness · sinusoids and tables · RoPE · train short, test long**
:::
:::

::: {.slide title="Attention ignores order"}
[A one-line theorem]{.kicker}

**Proposition.** Unmasked self-attention is *permutation equivariant*:
$f(\boldsymbol{\Pi}\mathbf{X}) = \boldsymbol{\Pi} f(\mathbf{X})$ for every
permutation matrix $\boldsymbol{\Pi}$.

**Proof sketch.** Projections act row-wise; scores become
$\boldsymbol{\Pi}\mathbf{S}\boldsymbol{\Pi}^\top$; row-softmax commutes with
permutations; $\boldsymbol{\Pi}^\top\boldsymbol{\Pi} = \mathbf{I}$. $\blacksquare$

. . .

- Stacking equivariant layers preserves equivariance; additional depth does
  not encode order.
- Without positions, attention cannot distinguish "dog bites man" from a
  corresponding permutation such as "man bites dog".
:::

::: {.slide title="The shuffle check"}
@!positional-information-attention-ignores-order

::: {.d2l-note}
A **causal mask** breaks the symmetry because position $i$ attends over
$i+1$ tokens. NoPE uses this asymmetry.
:::
:::

::: {.slide title="Sinusoidal encodings"}
Add a designed position vector to each token
(Vaswani et al., 2017):

$$p_{i, 2j} = \sin\big(i/10000^{2j/d}\big), \qquad p_{i, 2j+1} = \cos\big(i/10000^{2j/d}\big)$$

@positional-information-sinusoidal-encodings-1
:::

::: {.slide title="A continuous binary counter"}
Low bits flip quickly and high bits slowly; sinusoidal encodings provide a
smooth analogue:

@!positional-information-sinusoidal-encodings-3

Learned tables (BERT, GPT-2) instead train one free vector per position.
Rows beyond the training length receive no gradient.
:::

::: {.slide title="Sinusoidal encodings as rotations"}
Moving from position $i$ to $i + \delta$ is a **rotation** of each
two-column pair, by an angle depending only on $\delta$:

$$\begin{bmatrix} \cos(\delta\omega_j) & \sin(\delta\omega_j) \\ -\sin(\delta\omega_j) & \cos(\delta\omega_j) \end{bmatrix}\begin{bmatrix} p_{i, 2j} \\ p_{i, 2j+1} \end{bmatrix} = \begin{bmatrix} p_{i+\delta, 2j} \\ p_{i+\delta, 2j+1} \end{bmatrix}$$

With additive encoding, the model must learn to exploit this structure, and
a query--key product of sums has four terms. RoPE puts the rotation directly
into the score.
:::

::: {.slide title="Rotary position embeddings (RoPE)"}
[The default in most current open-weights models]{.kicker}

Demand scores that see only the offset:
$(\mathbf{R}_i \mathbf{q})^\top(\mathbf{R}_j \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{j-i}\, \mathbf{k}$
⇒ the $\mathbf{R}_i$ form a rotation group: rotate feature pair $m$ of
queries and keys by $i\,\omega_m$.

@fig:mdl-attention-rope-rotation
:::

::: {.slide title="RoPE in code"}
@positional-information-rotary-position-embeddings-1
:::

::: {.slide title="Invariance, measured"}
Shifting all positions jointly should leave RoPE scores unchanged:

@!positional-information-rotary-position-embeddings-2

Relative position is now a property of the **architecture**, not something
training must discover.
:::

::: {.slide title="Extrapolation beyond the training length"}
Contexts grow after deployment. Two schemes designed for extrapolation:

- **ALiBi** (Press et al., 2022): no encoding; subtract a per-head linear
  distance penalty
  $\mathrm{score}_{ij} = \mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d} - m_h\,(i-j)$,
  slopes $m_h = 2^{-8h/H}$ (power-of-two $H$). The bias remains defined at
  distance 400 because it does not use a position table.
- **NoPE** (Kazemnejad et al., 2023): no explicit position representation;
  positional information can still arise from the causal mask.
:::

::: {.slide title="An attention-only language model"}
`TinyCharLM`: token embedding + stacked residual causal attention + tied
head. It has **no FFN, LayerNorm, or projection biases**, so attention is
the only operation that mixes information across positions. Positional scheme is a
constructor choice:
`'learned' · 'sinusoidal' · 'rope' · 'alibi' · 'none'`.

@positional-information-an-attention-only-language-model
:::

::: {.slide title="Five models, one corpus"}
Character-level Time Machine, context 128, 3000 steps each:

@!positional-information-the-experiment-1
:::

::: {.slide title="Results at four times the training length"}
@!positional-information-the-experiment-2

- Training length: RoPE best (~5), learned close behind, sinusoidal ≈
  ALiBi (~7), NoPE worst (~9).
- Length 512: absolute schemes and RoPE degrade substantially because offsets
  above 127 were not observed during training.
- **ALiBi: flat.** NoPE: flat, from a weaker start.
:::

::: {.slide title="Extending a trained model's context"}
RoPE angles are continuous, so rescaling can map length $4L$ into the
trained range: **position interpolation** (Chen et al., 2023) took Llama
2k → 32k with a brief fine-tune; **YaRN** (Peng et al., 2024) rescales
fast and slow frequencies differently.

Long-context RoPE models commonly use this interpolation strategy.
:::

::: {.slide title="Recap"}
- Unmasked attention is permutation equivariant, as shown analytically and
  numerically.
- Absolute encodings restore order. Sinusoidal encodings remain defined
  beyond the training length, while learned-table rows there are untrained.
- RoPE rotates queries and keys: scores depend only on offsets, by
  construction.
- In this fixed-seed experiment, absolute encodings and RoPE degrade beyond
  the training length, while ALiBi and NoPE remain stable. PI and YaRN
  extend RoPE by interpolation.
- `TinyCharLM` is an attention-only model whose positional scheme is a
  constructor argument; a later section uses it to inspect trained attention.
:::
