# Positional Information
:label:`sec_positional-information`

The sentences "dog bites man" and "man bites dog" contain the same three
tokens. Fed to the self-attention layer of
:numref:`sec_multihead-attention`, they produce the same three output
vectors, only listed in a different order. Each output averages value
vectors with weights computed from content alone, and nothing in that
computation records where a token stands. A classifier that pools these
outputs cannot tell who bit whom. Attention has to be given the positions
from outside.

There are several ways to do so. We can add a position vector to each
token, rotate queries and keys by angles that grow with position, or
subtract from each score a penalty that grows with distance. All of them
work at the sequence lengths a model was trained on, and beyond those
lengths they behave very differently. That difference matters whenever a
model trained on short documents is asked to read long ones. We first
prove the symmetry that has to be broken, then build the schemes that
break it. Finally, a small attention-only language model, trained on short
sequences, shows what each scheme does on sequences four times longer and
how a trained model can be stretched to the longer context.

```{.python .input #positional-information}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #positional-information}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Permutation Equivariance
:label:`subsec_pos-ignores-order`

Before choosing a remedy, we state the symmetry precisely, since every
scheme in this section is a way of breaking it. Let the rows of
$\mathbf{X} \in \mathbb{R}^{n \times d}$ hold a sequence of $n$ token
representations. One head of self-attention, :eqref:`eq_self-attention`
of :numref:`sec_multihead-attention`, forms the score matrix $\mathbf{S} = \mathbf{X}\mathbf{W}_q
(\mathbf{X}\mathbf{W}_k)^\top / \sqrt{d_h}$ and returns $f(\mathbf{X}) =
\mathrm{softmax}(\mathbf{S})\, \mathbf{X}\mathbf{W}_v$, with the softmax
taken over each row. We consider it at evaluation time, with dropout
disabled and with no mask or position-dependent bias.

The claim is that shuffling the tokens shuffles the outputs in the same
way and changes nothing else: each token's output is the same as before,
only in a different row. A shuffle of the $n$ rows is a permutation matrix
$\boldsymbol{\Pi}$, which reorders the rows of whatever it multiplies from
the left.

**Proposition (permutation equivariance).** For every permutation matrix
$\boldsymbol{\Pi}$, $\;f(\boldsymbol{\Pi} \mathbf{X}) = \boldsymbol{\Pi}
f(\mathbf{X})$.

**Proof.** The projections act on each row separately, so
$(\boldsymbol{\Pi}\mathbf{X})\mathbf{W} = \boldsymbol{\Pi}(\mathbf{X}\mathbf{W})$
for each of $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$. The score matrix
$\mathbf{S}$ becomes $\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top$.
Softmax normalizes each row by itself, so it commutes with permuting the
rows, and it treats the entries of a row alike, so it commutes with
permuting the columns:
$\mathrm{softmax}(\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top) =
\boldsymbol{\Pi}\, \mathrm{softmax}(\mathbf{S})\, \boldsymbol{\Pi}^\top$.
Multiplying by the permuted values,
$\boldsymbol{\Pi}\, \mathrm{softmax}(\mathbf{S})\, \boldsymbol{\Pi}^\top
\boldsymbol{\Pi} \mathbf{X}\mathbf{W}_v = \boldsymbol{\Pi} f(\mathbf{X})$
because $\boldsymbol{\Pi}^\top \boldsymbol{\Pi} = \mathbf{I}$. Concatenating
heads and applying $\mathbf{W}_o$ row-wise preserves the property, so it
holds for multi-head attention too. $\blacksquare$

The layer therefore does not use a token's position in the sequence. We
check the proposition on the multi-head layer of
:numref:`sec_multihead-attention` by shuffling eight random tokens. The
shuffled input should produce the shuffled output, up to round-off.

```{.python .input #positional-information-attention-ignores-order}
%%tab pytorch
attention = d2l.MultiHeadAttention(num_hiddens=64, num_heads=4)
X = torch.randn(1, 8, 64)
perm = torch.randperm(8)
Y = attention(X, X, X)
Y_perm = attention(X[:, perm], X[:, perm], X[:, perm])
print(f'max |Y[perm] - Y_perm|: {(Y[:, perm] - Y_perm).abs().max():.2e}')
```

```{.python .input #positional-information-attention-ignores-order}
%%tab jax
attention = d2l.MultiHeadAttention(num_hiddens=64, num_heads=4,
                                   rngs=nnx.Rngs(d2l.get_key()))
key_X, key_perm = jax.random.split(d2l.get_key())
X = jax.random.normal(key_X, (1, 8, 64))
perm = jax.random.permutation(key_perm, 8)
Y = attention(X, X, X)
Y_perm = attention(X[:, perm], X[:, perm], X[:, perm])
print(f'max |Y[perm] - Y_perm|: {jnp.abs(Y[:, perm] - Y_perm).max():.2e}')
```

The two outputs agree to float32 round-off; the difference may even come
out exactly zero. Two caveats qualify this result. First, depth does not
help: a stack of equivariant layers, read out per token, is equivariant
too, and so is a feed-forward layer applied to each token. Second, the
proposition is about unmasked attention. A causal mask singles out each
position by how many predecessors it may see, which breaks the symmetry.
We return to this asymmetry below, in a model with no position encoding at
all. Every other scheme in this section gives each token its position
explicitly.

## Absolute Position Embeddings

The original transformer attaches a *position vector* $\mathbf{p}_i \in
\mathbb{R}^d$ to each position $i$ and feeds $\mathbf{x}_i + \mathbf{p}_i$
into the first layer :cite:`Vaswani.Shazeer.Parmar.ea.2017`. With the
vectors stacked as the rows of $\mathbf{P}$, shuffling the tokens gives the
input $\boldsymbol{\Pi}\mathbf{X} + \mathbf{P}$, whereas equivariance would
need $\boldsymbol{\Pi}(\mathbf{X} + \mathbf{P})$, and the proof fails at its
first step. Two tokens with identical embeddings at different positions
enter attention as different vectors. The position vectors can be learned or
designed, and the learned ones are the simpler idea.

### Learned Positions

A learned scheme makes $\mathbf{P}$ a trainable embedding table, one free
vector per position, as in BERT and GPT-2
:cite:`Devlin.Chang.Lee.ea.2018,Radford.Wu.Child.ea.2019`. It concedes that
we do not know the right encoding and lets gradient descent find one. GPT-2
small, for instance, learns 1,024 such vectors. The limitation of a learned
table is visible before any training. The table has one row for each
position up to some maximum length, and a row that no training example ever
used still holds its random initialization. To such a table, position 500 is
not "a bit beyond position 128". Its row carries no information at all.

### Sinusoidal Encodings
:label:`subsec_pos-sinusoidal`

The designed choice gives every position a code before any training, and
for every position, however far. Row $i$ of $\mathbf{P}$ holds sines and
cosines of $i$ at $d/2$ geometrically spaced frequencies,

$$
p_{i, 2m} = \sin\left(\frac{i}{10000^{2m/d}}\right), \qquad p_{i, 2m+1} = \cos\left(\frac{i}{10000^{2m/d}}\right).
$$
:eqlabel:`eq_sinusoidal-def`

Columns $2m$ and $2m{+}1$ share the frequency $\omega_m = 1/10000^{2m/d}$,
which falls from one radian per position for the first pair to nearly
$1/10000$ for the last. The design looks arbitrary until we plot it. We
implement the table as a small function and look at four adjacent columns
of a table of width 32: two pairs, whose periods of 35 and 63 positions
both fit in the plot.

```{.python .input #positional-information-sinusoidal-encodings-1}
%%tab pytorch
def sinusoidal_encoding(max_len, num_hiddens):  #@save
    """The (max_len, num_hiddens) table of sinusoidal position codes."""
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
def sinusoidal_encoding(max_len, num_hiddens):  #@save
    """The (max_len, num_hiddens) table of sinusoidal position codes."""
    theta = jnp.arange(max_len)[:, None] / 10000 ** (
        jnp.arange(0, num_hiddens, 2) / num_hiddens)
    return jnp.stack([jnp.sin(theta), jnp.cos(theta)],
                     -1).reshape(max_len, num_hiddens)

P = sinusoidal_encoding(60, 32)
d2l.plot(jnp.arange(60), P[:, 6:10].T, xlabel='position',
         figsize=(6, 2.5), legend=[f'column {d}' for d in range(6, 10)])
```

Adjacent column pairs share a frequency (one sine, one cosine), and the
frequency falls as the column index grows. This is binary counting made
continuous. When we count $0, 1, 2, \ldots$ in binary, the lowest bit flips
at every step, the next every two steps, the next every four. Fast columns
tell neighboring positions apart, whereas slow columns tell broad regions
apart.

The heatmap of the whole table shows the same structure. Each row is a
multi-frequency fingerprint of its position. Unlike bits, the values vary
smoothly, so nearby positions get nearby codes.

```{.python .input #positional-information-sinusoidal-encodings-3}
%%tab pytorch, jax
d2l.show_heatmaps(P[None, None], xlabel='column (encoding dimension)',
                  ylabel='row (position)', figsize=(3.5, 4), cmap='Blues')
```

Language models have mostly moved on from this table. It survives in
diffusion models, which use it to encode the noise level of a sample, and
it matters here for a property that leads directly to the next scheme,
rotary embeddings.

### Sinusoidal Encodings as Rotations

Designed sinusoids have a relative-position structure that a learned table
lacks. The original paper chose them because, for every fixed offset
$\delta$, the row $\mathbf{p}_{i+\delta}$ is a linear function of the row
$\mathbf{p}_i$. The linear map can be written down for each column pair:

$$
\begin{bmatrix} \cos(\delta \omega_m) & \sin(\delta \omega_m) \\  -\sin(\delta \omega_m) & \cos(\delta \omega_m) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2m} \\  p_{i, 2m+1} \\ \end{bmatrix}
= \begin{bmatrix} \sin\left((i+\delta) \omega_m\right) \\  \cos\left((i+\delta) \omega_m\right) \\ \end{bmatrix}
= \begin{bmatrix} p_{i+\delta, 2m} \\  p_{i+\delta, 2m+1} \\ \end{bmatrix},
$$
:eqlabel:`eq_sinusoidal-rotation`

by the angle-addition identities. The map from position $i$ to position
$i + \delta$ is a *rotation* of each two-column pair, by an angle $\delta
\omega_m$ that depends only on the offset $\delta$, never on $i$. The table
contains relative position. Does attention see it?

Only indirectly. The model receives $\mathbf{x}_i + \mathbf{p}_i$, and a
first-layer head scores the inputs at positions $i$ and $j$ with the
matrix $\mathbf{A} = \mathbf{W}_q \mathbf{W}_k^\top / \sqrt{d_h}$. Expanding
the score gives four terms:

$$
(\mathbf{x}_i + \mathbf{p}_i)^\top \mathbf{A}\, (\mathbf{x}_j + \mathbf{p}_j)
= \underbrace{\mathbf{x}_i^\top \mathbf{A}\, \mathbf{x}_j}_{\textrm{content--content}}
+ \underbrace{\mathbf{x}_i^\top \mathbf{A}\, \mathbf{p}_j}_{\textrm{content--position}}
+ \underbrace{\mathbf{p}_i^\top \mathbf{A}\, \mathbf{x}_j}_{\textrm{position--content}}
+ \underbrace{\mathbf{p}_i^\top \mathbf{A}\, \mathbf{p}_j}_{\textrm{position--position}}.
$$
:eqlabel:`eq_pos-four-terms`

The first term ignores position. The middle two pair the content of one
token with the absolute position of the other: $\mathbf{x}_i^\top
\mathbf{A}\, \mathbf{p}_j$ depends on where token $j$ stands, not on how
far it is from token $i$, so moving both tokens by the same amount changes
it. The last term is the only one that can carry the offset, and it
carries the offset alone only when $\mathbf{A}$ acts on each frequency pair
as a scaled rotation and never mixes two pairs (Exercise
:exref:`ex_positional-information-additive-codes` proves it). Relative
position is there, but the projections must extract it from one term of
four while the two cross terms tie the score to absolute positions. Rotary
embeddings build the rotation into the comparison of query and key
instead.

## Rotary Position Embeddings
:label:`subsec_pos-rope`

*Rotary position embeddings* (RoPE) :cite:`Su.Lu.Pan.ea.2021` skip the
addition and act on queries and keys where they meet, in the score. Let us
write down what we want and see what it forces. We look for a
transformation $\mathbf{R}_i$, applied to the query at position $i$ and to
the key at position $j$, and we require that the score depend on the two
positions only through the offset $j - i$:

$$
(\mathbf{R}_i \mathbf{q})^\top (\mathbf{R}_j \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_i^\top \mathbf{R}_j \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{j-i} \mathbf{k} \quad \textrm{for all } i, j.
$$
:eqlabel:`eq_rope-goal`

The requirement $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$ pins
down the kind of map we need. Take first $i = j = 0$:
$\mathbf{R}_0^\top \mathbf{R}_0 = \mathbf{R}_0$. The left side is
symmetric, so $\mathbf{R}_0$ is symmetric, and then $\mathbf{R}_0^2 =
\mathbf{R}_0$: it is an orthogonal projection. Any projection other than
the identity would discard some query directions at every position, so we
take $\mathbf{R}_0 = \mathbf{I}$. With $i = j$ the requirement now reads
$\mathbf{R}_i^\top \mathbf{R}_i = \mathbf{I}$, so every $\mathbf{R}_i$ is
orthogonal and preserves lengths. With $j = 0$ it gives $\mathbf{R}_{-i} =
\mathbf{R}_i^\top$. Replacing $i$ by $-i$ in the requirement then gives
$\mathbf{R}_{-i}^\top \mathbf{R}_j = \mathbf{R}_i \mathbf{R}_j =
\mathbf{R}_{i+j}$. Hence $\mathbf{R}_i = \mathbf{R}_1^i$: one orthogonal
matrix, applied once per position.

Which orthogonal matrices are there? In a suitable orthonormal basis, every
orthogonal matrix is block diagonal, with $2 \times 2$ rotations and
possibly some entries $+1$ or $-1$ on the diagonal. A $+1$ is a coordinate
that never turns and a $-1$ one that flips sign at every step; two $+1$
entries form a rotation by $0$, and two $-1$ entries a rotation by $\pi$.
The change of basis can be absorbed into the learned projections
$\mathbf{W}_q$ and $\mathbf{W}_k$. So the requirement leaves surprisingly
little freedom: apart from at most one coordinate that stays fixed and one
that flips, every solution rotates pairs of coordinates, and only the angle
by which each pair turns per position is left to choose. RoPE chooses the
sinusoidal frequencies, now applied *multiplicatively* and with the head
width $d_h$ in place of the model width $d$. It splits the $d_h$
coordinates of each query and key into $d_h/2$ pairs and rotates pair $m$
at position $i$ by the angle $i\,\omega_m$, with $\omega_m =
10000^{-2m/d_h}$:

$$
\mathbf{R}_i = \begin{bmatrix} R(i\omega_0) & & \\ & \ddots & \\ & & R(i\omega_{d_h/2-1}) \end{bmatrix}, \qquad R(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}.
$$
:eqlabel:`eq_rope-def`

Each pair turns at its own rate: the first by one radian per position, the
last by less than a thousandth of a radian. :numref:`fig_rope-rotation`
shows one pair plane. Shifting both positions by the same amount rotates
query and key together and preserves the angle between them, and with it
their dot product. Rotations also preserve norms, so RoPE never inflates or
shrinks a token's content. The fast pairs separate neighboring positions
sharply, while the slow pairs barely move across a sentence and carry
content largely unchanged. This division into fast and slow pairs returns
at the end of the section. Only queries and keys are rotated. Rotating the
values as well would make each output depend on the absolute positions of
the keys it reads, the very dependence the construction was built to
remove.

RoPE is the positional scheme of most open-weights language models released
since 2023, and of the schemes in this section it is the one to know.

![Rotary embeddings rotate each two-dimensional feature pair of the query and the key by an angle proportional to position. Shifting both positions by the same amount, here by 3, rotates both vectors together and leaves the angle between them ($2\theta + \varphi$ in both panels), and hence the attention score, unchanged. Here $\theta = \omega_m$ is the angle by which pair $m$ turns per position, and $\varphi$ is the angle between the unrotated query and key.](../img/mdl-attention-rope-rotation.svg)
:label:`fig_rope-rotation`

`rope_tables` computes the cosines and sines of all the angles, and `rope`
rotates the even/odd coordinate pairs of its input with them. The argument
`offset` adds the same number to every position, which is what we need to
test the invariance. The other arguments keep their defaults until the end
of the section.

```{.python .input #positional-information-rotary-position-embeddings-1}
%%tab pytorch
def rope_tables(n, d, offset=0, base=10000.0, scale=None,  #@save
                device=None):
    """Cosines and sines, each (n, d/2), of the angles of the positions
    offset, ..., offset + n - 1."""
    pos = torch.arange(n, dtype=torch.float32, device=device) + offset
    omega = base ** (-torch.arange(0, d, 2, device=device) / d)
    if scale is not None:  # One angle factor per pair, shape (d/2,)
        omega = omega * scale
    theta = pos[:, None] * omega[None, :]
    return torch.cos(theta), torch.sin(theta)

def rope(x, offset=0, base=10000.0, scale=None, tables=None):  #@save
    """Rotate feature pairs of x (..., num_steps, d) by position angles;
    tables from rope_tables take the place of offset, base, and scale."""
    if tables is None:
        tables = rope_tables(*x.shape[-2:], offset, base, scale, x.device)
    # Angles in fp32, tables in the dtype of x: bf16 queries stay bf16
    cos, sin = (u.to(x.dtype) for u in tables)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], -1).flatten(-2)
```

```{.python .input #positional-information-rotary-position-embeddings-1}
%%tab jax
def rope_tables(n, d, offset=0, base=10000.0, scale=None):  #@save
    """Cosines and sines, each (n, d/2), of the angles of the positions
    offset, ..., offset + n - 1."""
    pos = jnp.arange(n, dtype=jnp.float32) + offset
    omega = base ** (-jnp.arange(0, d, 2) / d)
    if scale is not None:  # One angle factor per pair, shape (d/2,)
        omega = omega * scale
    theta = pos[:, None] * omega[None, :]
    return jnp.cos(theta), jnp.sin(theta)

def rope(x, offset=0, base=10000.0, scale=None, tables=None):  #@save
    """Rotate feature pairs of x (..., num_steps, d) by position angles;
    tables from rope_tables take the place of offset, base, and scale."""
    if tables is None:
        tables = rope_tables(*x.shape[-2:], offset, base, scale)
    # Angles in fp32, tables in the dtype of x: bf16 queries stay bf16
    cos, sin = (u.astype(x.dtype) for u in tables)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], -1).reshape(x.shape)
```

If :eqref:`eq_rope-goal` holds, shifting all positions together must leave
every RoPE score unchanged, up to round-off. We test this on 32 random
queries and keys of width $d_h = 64$ (the variable `d` in the cell). For
contrast we run the same test on the additive sinusoidal encoding, with $\mathbf{A}$ the identity
scaled by $1/\sqrt{d_h}$. Its position--position term depends on the
offset alone (Exercise
:exref:`ex_positional-information-distinct-positions-distinct-codes`), but
the two cross terms of :eqref:`eq_pos-four-terms` move with the shift, so
we expect the scores to change by about as much as they vary.

```{.python .input #positional-information-rotary-position-embeddings-2}
%%tab pytorch
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
key_q, key_k = jax.random.split(d2l.get_key())
q, k = jax.random.normal(key_q, (T, d)), jax.random.normal(key_k, (T, d))
# Full float32 matmuls: GPU default (TF32) round-off would hide the check
dot = lambda a, b: jnp.matmul(a, b.T, precision='highest') / math.sqrt(d)

scores = lambda off: dot(rope(q, off), rope(k, off))
for off in (1, 17, 480):
    print(f'RoPE, all positions shifted by {off:>3}: max score change '
          f'{jnp.abs(scores(off) - scores(0)).max():.1e}')

P = sinusoidal_encoding(600, d)
added = lambda off: dot(q + P[off:off+T], k + P[off:off+T])
print(f'additive sinusoidal, shifted by  17: max score change '
      f'{jnp.abs(added(17) - added(0)).max():.1e}')
```

At every shift the RoPE scores change only by float32 round-off, around
$10^{-6}$ to $10^{-5}$, while the scores themselves are of order one. They
stay at round-off even for a shift of 480, far beyond where a table of 128
rows would end. The additive encoding moves the scores by almost 3,
comparable to the scores themselves. With RoPE, relative position is part of
the architecture rather than something learned from data. That does not make
its predictions at offsets never seen in training accurate, and the
experiment below tests them.

## Distance Penalties and No Encoding

The schemes so far put position into the vectors that attention compares.
Two further schemes leave the vectors alone. Both were proposed with long
contexts in mind, and both reappear in the experiment.

*ALiBi* (attention with linear biases) :cite:`Press.Smith.Lewis.2022` adds
no position vectors and rotates nothing. It subtracts from every score a
penalty proportional to the distance between query and key. For query
position $i$ and key position $j \leq i$ in head $h$ of $H$,

$$
\mathrm{score}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} - m_h\,(i - j), \qquad m_h = 2^{-8h/H}.
$$
:eqlabel:`eq_alibi-def`

Each head discounts distant tokens at its own rate, and the rates form a
geometric sequence. For $H = 4$ the slopes are $1/4, 1/16, 1/64, 1/256$,
so head 1 sees a short window and head 4 hundreds of positions. (The
formula $2^{-8h/H}$ is the paper's rule for head counts that are powers of
two.) With slope $1/4$, a key 40 positions back already loses a factor
$e^{-10}$ of its weight. The bias is a formula in the distance, not a
table, so a distance of 400 is as valid an input as a distance of 4.
Nothing in ALiBi can break at a long context. The drawback is a recency
prior built into every score, which large content scores can offset but
training cannot remove.

*NoPE* (no positional encoding) :cite:`Kazemnejad.Padhi.Ramamurthy.ea.2023`
uses the asymmetry that the proposition leaves out. In a causal model the
mask already breaks the symmetry. Position 0 attends over one token,
position 100 over a hundred and one. The number of tokens competing in each
softmax is itself positional information, and a decoder can in principle
recover order from it. NoPE has no table and no angles, so it has no
explicit position range. How well it works depends on how much of this
signal the model learns to use.

## Extrapolation Beyond the Training Length

Within the training length every scheme above gives the model a way to
tell positions apart. The schemes differ on longer sequences. A
length-extrapolation test, the experiment with which ALiBi was introduced,
trains at a context of $L$ tokens and compares perplexity at $L$ with
perplexity at multiples of $L$. We take $L = 128$ bytes and test at 256 and
512.

### The Language Model TinyCharLM
:label:`subsec_tinycharlm`

We need a model in which position can enter only through the scheme under
test, and which is small enough to train many times. `TinyCharLM` is such a
model. It reads bytes, embeds each in 128 dimensions, passes the result
through two residual blocks of causal attention with four heads of width 32,
and predicts the next byte with an output head tied to the embedding
:cite:`Press.Wolf.2017`. There is no feed-forward network and no
normalization layer, and the attention projections carry no bias terms by
default (a `bias` switch, used in :numref:`sec_what-attention-computes`,
restores them). Only attention mixes information across positions. It is the attention-only cousin of the `TinyLM` of
:numref:`subsec_tinylm`, with the same vocabulary of 256 bytes.

The constructor argument `pos` selects the scheme: `'learned'`,
`'sinusoidal'`, `'rope'`, `'alibi'`, or `'none'`. Two methods matter in
this section: `embed`, where the absolute schemes act, and `attend`, where
the relative ones act. The method `stream` runs the blocks and also
collects their attention maps, which `attention_weights` returns. We look
at those maps once below, and :numref:`sec_what-attention-computes` reads
them closely.

```{.python .input #positional-information-an-attention-only-language-model}
%%tab pytorch
class TinyCharLM(nn.Module):  #@save
    """Attention-only language model."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False):
        super().__init__()
        assert pos in ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
        self.num_heads, self.pos = num_heads, pos
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if pos == 'sinusoidal':
            self.register_buffer('P', sinusoidal_encoding(max_len,
                                                          num_hiddens))
        if pos == 'rope':  # Base and per-pair angle factors of the rotation
            self.rope_base = 10000.0
            self.register_buffer('rope_scale',
                                 torch.ones(num_hiddens // num_heads // 2))
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(num_hiddens, 3 * num_hiddens, bias=bias),
            proj=nn.Linear(num_hiddens, num_hiddens, bias=bias)))
            for _ in range(num_blks)])

    def stream(self, X, num_blks=None):
        """Residual stream after the first num_blks blocks (default: all),
        and the attention weights of those blocks."""
        H, maps = self.embed(X), []
        for blk in self.blks[:num_blks]:
            out, weights = self.attend(blk, H)
            H, maps = H + out, maps + [weights]
        return H, maps

    def forward(self, X):
        return F.linear(self.stream(X)[0], self.token_emb.weight)  # Tied

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        return self.stream(X)[1]
```

```{.python .input #positional-information-an-attention-only-language-model}
%%tab jax
class TinyCharLM(nnx.Module):  #@save
    """Attention-only language model."""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False, *, rngs):
        assert pos in ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
        self.num_heads, self.pos = num_heads, pos
        init = nnx.initializers.normal(0.02)
        self.token_emb = d2l.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        if pos == 'sinusoidal':
            self.P = nnx.Cache(sinusoidal_encoding(max_len, num_hiddens))
        if pos == 'rope':  # Base and per-pair angle factors of the rotation
            self.rope_base = 10000.0
            self.rope_scale = nnx.Cache(
                jnp.ones(num_hiddens // num_heads // 2))
        self.blks = nnx.List([nnx.Dict(
            qkv=d2l.Linear(num_hiddens, 3 * num_hiddens, use_bias=bias,
                           rngs=rngs),
            proj=d2l.Linear(num_hiddens, num_hiddens, use_bias=bias,
                            rngs=rngs))
            for _ in range(num_blks)])

    def stream(self, X, num_blks=None):
        """Residual stream after the first num_blks blocks (default: all),
        and the attention weights of those blocks."""
        H, maps = self.embed(X), []
        for blk in self.blks[:num_blks]:
            out, weights = self.attend(blk, H)
            H, maps = H + out, maps + [weights]
        return H, maps

    def __call__(self, X):
        return self.token_emb.attend(self.stream(X)[0])  # Tied output head

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        return self.stream(X)[1]
```

The absolute schemes act once, on the input: `embed` adds the learned or
the sinusoidal table to the token embeddings. The sinusoidal table enters
divided by $\sqrt{d}$, where $d = 128$ is the model width. Its entries have
magnitude about $0.7$ and the token embeddings start with entries of about
$0.02$, so the raw table would outweigh them 35 to 1; scaled, it is about
three times their size. The original transformer multiplies the token
embeddings by $\sqrt{d}$ instead. Here the output head reuses the token
embeddings, so that factor would scale every logit as well, and the
sinusoidal model would differ from the others in more than its positions.
As written, the five models differ only in where position enters.

```{.python .input #positional-information-an-attention-only-language-model-1}
%%tab pytorch
@d2l.add_to_class(TinyCharLM)  #@save
def embed(self, X):
    """Token embeddings plus the absolute position code, if any."""
    H = self.token_emb(X)
    if self.pos == 'learned':
        H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
    if self.pos == 'sinusoidal':  # Table scaled to the token embeddings
        H = H + self.P[:X.shape[1]] / math.sqrt(H.shape[-1])
    return H
```

```{.python .input #positional-information-an-attention-only-language-model-1}
%%tab jax
@d2l.add_to_class(TinyCharLM)  #@save
def embed(self, X):
    """Token embeddings plus the absolute position code, if any."""
    H = self.token_emb(X)
    if self.pos == 'learned':
        H = H + self.pos_emb(jnp.arange(X.shape[1]))
    if self.pos == 'sinusoidal':  # Table scaled to the token embeddings
        H = H + self.P[:X.shape[1]] / math.sqrt(H.shape[-1])
    return H
```

The relative schemes act inside every block, on the queries, keys, and
scores. `attend` is causal multi-head attention, with `d2l.masked_softmax`
and `d2l.causal_mask` of :numref:`sec_attention-scoring-functions`, in
which RoPE rotates the queries and the keys and ALiBi adds the bias
$-m_h\,(i - j)$ of :eqref:`eq_alibi-def` to the scores. With the other
schemes it is plain causal attention.

```{.python .input #positional-information-an-attention-only-language-model-2}
%%tab pytorch
@d2l.add_to_class(TinyCharLM)  #@save
def alibi(self, T, device):
    """ALiBi bias -m_h (i - j) of every head, (num_heads, T, T)."""
    h = torch.arange(1, self.num_heads + 1, device=device)
    slopes = 2.0 ** (-8.0 * h / self.num_heads)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    return slopes[:, None, None] * (pos[None, :] - pos[:, None])

@d2l.add_to_class(TinyCharLM)  #@save
def attend(self, blk, H):
    """Output of attention block blk on the stream H, and its weights."""
    B, T, D = H.shape
    q, k, v = blk['qkv'](H).chunk(3, -1)
    q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
               for u in (q, k, v))
    if self.pos == 'rope':
        q = rope(q, base=self.rope_base, scale=self.rope_scale)
        k = rope(k, base=self.rope_base, scale=self.rope_scale)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if self.pos == 'alibi':
        scores = scores + self.alibi(T, H.device)
    weights = d2l.masked_softmax(scores, d2l.causal_mask(T, device=H.device))
    out = (weights @ v).transpose(1, 2).reshape(B, T, D)
    return blk['proj'](out), weights
```

```{.python .input #positional-information-an-attention-only-language-model-2}
%%tab jax
@d2l.add_to_class(TinyCharLM)  #@save
def alibi(self, T):
    """ALiBi bias -m_h (i - j) of every head, (num_heads, T, T)."""
    h = jnp.arange(1, self.num_heads + 1)
    slopes = 2.0 ** (-8.0 * h / self.num_heads)
    pos = jnp.arange(T, dtype=jnp.float32)
    return slopes[:, None, None] * (pos[None, :] - pos[:, None])

@d2l.add_to_class(TinyCharLM)  #@save
def attend(self, blk, H):
    """Output of attention block blk on the stream H, and its weights."""
    B, T, D = H.shape
    q, k, v = jnp.split(blk['qkv'](H), 3, axis=-1)
    q, k, v = (u.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
               for u in (q, k, v))
    if self.pos == 'rope':
        q = rope(q, base=self.rope_base, scale=self.rope_scale[...])
        k = rope(k, base=self.rope_base, scale=self.rope_scale[...])
    scores = q @ k.swapaxes(-2, -1) / math.sqrt(q.shape[-1])
    if self.pos == 'alibi':
        scores = scores + self.alibi(T)
    weights = d2l.masked_softmax(scores, d2l.causal_mask(T))
    out = (weights @ v).swapaxes(1, 2).reshape(B, T, D)
    return blk['proj'](out), weights
```

:begin_tab:`jax`
The token table is `d2l.Embed` of :numref:`subsec_utils_embed`, which
computes the gradient of the lookup faster than `nnx.Embed` on bytes of
text, whose frequencies are skewed.
:end_tab:

### The Experiment
:label:`subsec_pos-experiment`

We train each scheme three times from independent initializations, so that
every number comes with the range over which it varies between runs. The
data are `d2l.WikiText103`, the bytes of Wikipedia articles on which
`TinyLM` was trained (:numref:`subsec_tinylm`), in minibatches of 64
windows of 128 bytes. `d2l.train_steps` (:numref:`sec_test_models`) runs a
fixed number of optimizer steps, here 3,000 steps of AdamW without weight
decay, each on a fresh minibatch, and returns the loss of every step;
`d2l.final_loss` averages the last hundred. The learned and sinusoidal
tables are allocated out to `max_len` $= 512$ positions so that the models
can be evaluated on longer sequences, but only the first 128 rows of the
learned table ever receive a nonzero gradient.

```{.python .input #positional-information-the-experiment-1}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=128)
schemes = ('learned', 'sinusoidal', 'rope', 'alibi', 'none')

def train_tiny(**kwargs):
    """A TinyCharLM after 3000 steps, and its final training loss."""
    model = TinyCharLM(len(data.vocab), **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.0)
    return model, d2l.final_loss(d2l.train_steps(model, data, optimizer,
                                                 3000))

runs = {}  # Three independently initialized models per scheme
for pos in schemes:
    trained = [train_tiny(pos=pos) for _ in range(3)]
    runs[pos] = [model for model, _ in trained]
    print(f'{pos:>10}: final training loss '
          + ', '.join(f'{loss:.2f}' for _, loss in trained))
models = {pos: ms[0] for pos, ms in runs.items()}  # The first run of each
```

```{.python .input #positional-information-the-experiment-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
schemes = ('learned', 'sinusoidal', 'rope', 'alibi', 'none')

def train_tiny(**kwargs):
    """A TinyCharLM after 3000 steps, and its final training loss."""
    model = TinyCharLM(len(data.vocab), **kwargs,
                       rngs=nnx.Rngs(d2l.get_key()))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=0.0),
                              wrt=nnx.Param)
    return model, d2l.final_loss(d2l.train_steps(model, data, optimizer,
                                                 3000))

runs = {}  # Three independently initialized models per scheme
for pos in schemes:
    trained = [train_tiny(pos=pos) for _ in range(3)]
    runs[pos] = [model for model, _ in trained]
    print(f'{pos:>10}: final training loss '
          + ', '.join(f'{loss:.2f}' for _, loss in trained))
models = {pos: ms[0] for pos, ms in runs.items()}  # The first run of each
```

All fifteen runs converge, the three runs of each scheme within about 0.1
of each other in loss, and RoPE already ends lowest, near 2.0 against 2.2
to 2.5 for the others.

Before looking at validation perplexities, we can predict from the
mechanisms which schemes should survive a longer context. At context 512
the learned table reaches rows 128 to 511, which still hold their random
initialization. The sinusoidal table supplies rows that exist and vary
smoothly, but whose slow columns take values never seen in training. RoPE
has no table to run out of, but its slow pairs have periods longer than
128 positions. At offsets beyond 127 they turn through angles they never
reached during training, and the projections were never fit to scores
computed from those angles. ALiBi's bias is a formula in the distance and
has nothing to run out of. NoPE is the subtle case. It has no table and no
angles, but the signal it can read, the number of keys in each softmax,
also takes values beyond 128 that training never produced. So we expect
the first three schemes to degrade and ALiBi to hold, and NoPE to degrade
only if it has learned to use position at all.

We evaluate validation perplexity at the training context, $n = 128$, and
at $n = 256$ and $512$. All three contexts read the same validation bytes.
We cut the validation split into consecutive windows of 128 bytes with
`data.windows` and regroup them into windows of 256 and 512, so that only
the context available to each prediction changes. `d2l.lm_perplexity`
(:numref:`sec_test_models`) sums the cross-entropy over all predicted bytes
and converts it to a perplexity once at the end. The table gives, for each
scheme, the mean over the three runs and, in parentheses, the lowest and
the highest. One consequence of the causal mask is worth keeping in mind
while reading it. In a window of 512 bytes, the predictions at positions 0
to 127 see exactly what they saw at context 128 and do not change. Any
change at context 512 comes from positions 128 to 511, and the average
over the whole window dilutes it.

```{.python .input #positional-information-the-experiment-3}
%%tab pytorch, jax
train_len, batch = data.num_steps, 16
contexts = (train_len, 2 * train_len, 4 * train_len)
block = batch * max(contexts)  # Whole minibatches at every context
num_bytes = (len(data.val) - 1) // block * block
X, Y = data.windows(data.val, range(0, num_bytes, train_len))
eval_sets = {}
for n in contexts:  # The same bytes, regrouped into windows of n
    Xn, Yn = X.reshape(-1, n), Y.reshape(-1, n)
    eval_sets[n] = [(Xn[i:i + batch], Yn[i:i + batch])
                    for i in range(0, len(Xn), batch)]

def perplexities(model):
    return [d2l.lm_perplexity(model, eval_sets[n]) for n in contexts]

def mean_range(values):
    """Mean and (min-max) of the runs, one column of the table."""
    return f'{np.mean(values):5.1f} ({min(values):.1f}-{max(values):.1f})'

ppls = {pos: [perplexities(model) for model in ms] for pos, ms in runs.items()}
print(f'{num_bytes:,} validation bytes, mean (min-max) of three runs')
print(f'{"context":>10}: ' + ''.join(f'{n:>18}' for n in contexts))
for pos in schemes:  # zip(*...) turns runs x contexts into contexts x runs
    print(f'{pos:>10}: '
          + ''.join(f'{mean_range(col):>18}' for col in zip(*ppls[pos])))
```

How good is a perplexity of 7 or 10 per byte? A useful yardstick is a model
that predicts each byte from the previous byte alone. Counting the byte
pairs of the training split gives this *bigram* model in closed form, and
we evaluate it on the same validation bytes.

```{.python .input #positional-information-byte-bigram-baseline}
%%tab pytorch, jax
V = len(data.vocab)
counts = np.ones((V, V))  # Byte pair counts, plus one for unseen pairs
for s in range(0, len(data.train) - 1, 10**7):  # Chunks of 10^7 bytes
    a = data.train[s:s + 10**7 + 1].astype(np.int64)
    counts += np.bincount(a[:-1] * V + a[1:], minlength=V * V).reshape(V, V)
bigram = counts / counts.sum(1, keepdims=True)  # P(next byte | this byte)
x, y = np.asarray(X).reshape(-1), np.asarray(Y).reshape(-1)
print(f'byte bigram: perplexity {np.exp(-np.log(bigram[x, y]).mean()):.1f}')
```

At the training length RoPE is far ahead, at 7.3 with runs from 7.2 to 7.4.
The learned table, the sinusoidal table, and ALiBi follow at about 9 to 10,
all three about 15% below the bigram's 11.6. The sinusoidal range overlaps
both others. The learned table sits below ALiBi in every run, by 0.3 to
0.4, a gap that is small next to RoPE's lead and on which nothing below
depends. NoPE sits on the bigram. That is less mysterious than it looks:
the bigram predicts from the byte just before the one to be predicted,
which is the input byte at the same position and already sits in the
residual stream. Without positions, two attention blocks trained for 3,000
steps add nothing measurable to what that one byte says. This is a
statement about this small model and this budget. Larger causal models
trained for longer do learn to read position from the mask, and come close
to models with explicit encodings :cite:`Haviv.Ram.Press.ea.2022`.

RoPE's lead is far larger than the spread between runs, and the shift test
suggests why. Under RoPE, a head that looks at the previous byte needs one
fixed setting of its weights for every position (Exercise
:exref:`ex_positional-information-offset-preference-rope` builds such a
head). With an additive table the same head has to learn the offset from
absolute codes, through the four terms of :eqref:`eq_pos-four-terms`. A
model this small, with no feed-forward layers, leans heavily on such
heads, so the lead is large here. At the scale of the ALiBi paper the
schemes come much closer at the training length
:cite:`Press.Smith.Lewis.2022`, and large models choose RoPE for its
relative-position structure and for how readily it can be extended.

Beyond the training length the prediction holds for the three schemes that
have something to run out of. At context 512 the learned table roughly
doubles its perplexity or worse, the sinusoidal table rises by about 60%,
and RoPE, the best model at 128, ends at nearly four times its in-range
value on average. The model that used position best in range has the most
to lose out of range. The sinusoidal table degrades least of the three, in
every run: its new rows continue the old ones smoothly, while the learned
table's new rows were never trained. ALiBi holds its level at every
context. So does NoPE, but its flat row says nothing about the scheme: a
model that never learned to use position has no position to lose.

ALiBi's flat curve needs a closer look, because the bias alone predicts
where its heads can look. With slope $1/16$, a key 128 positions back keeps
a factor $e^{-8}$, about $3 \times 10^{-4}$, of the weight of the nearest
key. With slope $1/64$ it keeps $e^{-2} \approx 0.14$, and with slope
$1/256$ it keeps $e^{-1/2} \approx 0.6$. The two steep heads should
therefore see nothing beyond the training window, and the two gentle heads
should spread a good part of their weight there. For every head of the
first trained ALiBi model we measure, on 16 validation windows of 512
bytes, the share of its attention weight that lands on keys 128 or more
positions back, averaged over the queries at positions 128 to 511. The
first row gives the share that the bias alone assigns when all content
scores are equal.

```{.python .input #positional-information-alibi-weight-beyond-the-window}
%%tab pytorch
alibi_model, T = models['alibi'], max(contexts)
device = next(alibi_model.parameters()).device
X_eval = eval_sets[T][0][0].to(device)  # 16 windows of 512 bytes
i = torch.arange(T, device=device)
far = i[:, None] - i[None, :] >= train_len  # Offsets never trained on

def far_weight(weights):
    """Per head: mean weight on far keys, over queries from train_len on."""
    return (weights * far).sum(-1)[..., train_len:].mean((0, -1)).tolist()

with torch.inference_mode():
    maps = alibi_model.attention_weights(X_eval)
    bias_only = d2l.masked_softmax(alibi_model.alibi(T, device)[None],
                                   d2l.causal_mask(T, device=device))
print(f'{"head":>10}: ' + ''.join(f'{h:>9}' for h in range(1, 5)))
for name, weights in [('bias alone', bias_only)] + [
        (f'block {b}', w) for b, w in enumerate(maps, 1)]:
    print(f'{name:>10}: ' + ''.join(f'{w:9.3f}' for w in far_weight(weights)))
```

```{.python .input #positional-information-alibi-weight-beyond-the-window}
%%tab jax
alibi_model, T = models['alibi'], max(contexts)
X_eval = eval_sets[T][0][0]  # 16 windows of 512 bytes
i = jnp.arange(T)
far = i[:, None] - i[None, :] >= train_len  # Offsets never trained on

def far_weight(weights):
    """Per head: mean weight on far keys, over queries from train_len on."""
    return (weights * far).sum(-1)[..., train_len:].mean((0, -1)).tolist()

maps = alibi_model.attention_weights(X_eval)
bias_only = d2l.masked_softmax(alibi_model.alibi(T)[None], d2l.causal_mask(T))
print(f'{"head":>10}: ' + ''.join(f'{h:>9}' for h in range(1, 5)))
for name, weights in [('bias alone', bias_only)] + [
        (f'block {b}', w) for b, w in enumerate(maps, 1)]:
    print(f'{name:>10}: ' + ''.join(f'{w:9.3f}' for w in far_weight(weights)))
```

In how much weight they put beyond the training window, the trained heads
follow the bias: block 1 matches the bias alone to the third digit, and
block 2 to within about 0.01. Heads 1 and 2 put essentially no weight
there, and heads 3 and 4 put about 11 to 12% and about 40%. So the model
does read far keys. Does it use them? We evaluate each of the three ALiBi
models at context 512 once more with every key 128 or more positions back
removed, by a bias of $-\infty$ on those offsets.

```{.python .input #positional-information-alibi-without-far-keys}
%%tab pytorch
class WindowedALiBi(TinyCharLM):
    """ALiBi whose queries see only the window most recent keys."""
    def __init__(self, *args, window, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def alibi(self, T, device):
        i = torch.arange(T, device=device)
        return super().alibi(T, device).masked_fill(
            i[:, None] - i[None, :] >= self.window, -math.inf)

full, windowed = [], []
for model in runs['alibi']:
    model_w = WindowedALiBi(len(data.vocab), pos='alibi',
                            window=train_len).to(device)
    model_w.load_state_dict(model.state_dict())  # Same trained weights
    full.append(d2l.lm_perplexity(model, eval_sets[T]))
    windowed.append(d2l.lm_perplexity(model_w, eval_sets[T]))
for name, ppl in ('all keys', full), (f'keys < {train_len} back', windowed):
    print(f'{name:>16}: perplexity at {T} {mean_range(ppl)}')
```

```{.python .input #positional-information-alibi-without-far-keys}
%%tab jax
class WindowedALiBi(TinyCharLM):
    """ALiBi whose queries see only the window most recent keys."""
    def __init__(self, *args, window, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def alibi(self, T):
        i = jnp.arange(T)
        return jnp.where(i[:, None] - i[None, :] >= self.window, -jnp.inf,
                         super().alibi(T))

full, windowed = [], []
for model in runs['alibi']:
    model_w = WindowedALiBi(len(data.vocab), pos='alibi', window=train_len,
                            rngs=nnx.Rngs(d2l.get_key()))
    nnx.update(model_w, nnx.state(model))  # Same trained weights
    full.append(d2l.lm_perplexity(model, eval_sets[T]))
    windowed.append(d2l.lm_perplexity(model_w, eval_sets[T]))
for name, ppl in ('all keys', full), (f'keys < {train_len} back', windowed):
    print(f'{name:>16}: perplexity at {T} {mean_range(ppl)}')
```

The two rows agree to the printed digit, in the mean and in the lowest and
the highest run. How can 40% of a head's weight vanish without effect? The
match with the bias alone says that whatever the content scores of heads 3
and 4 select, they do not favor near keys over far ones. Such a head
computes a broad, recency-weighted average of the values. Cut off at 128
positions, the average is renormalized over the near keys, and the
unchanged perplexity says that an average over the last 128 bytes of an
article tells the model as much about the next byte as an average over the
last 512. The probe does not show the averaging directly, but it is the
reading that fits both measurements.

ALiBi's curve is therefore flat for two reasons: its bias is defined at
every distance, so nothing breaks, and what its gentle heads read beyond
the training window adds nothing to the predictions. The extra 384 bytes
of context bring no gain either, since the model is no better at 512 than
at 128. A flat curve shows that a scheme does not break at a longer
context. It does not show that the model uses that context.

### Extending a Trained Model's Context
:label:`subsec_pos-context-extension`

RoPE gave the best model at the training length and one of the worst at
four times that length. Can the trained model be repaired without training
it again? Only some of its pairs meet anything new beyond the training
length. A pair whose period $2\pi/\omega_m$ is at most 128 positions
completes a full turn within the training offsets, so every angle it
produces at offset 400 lies within $\omega_m/2$ of an angle it already took
during training. For the head width $d_h = 32$ of `TinyCharLM`, the
periods $2\pi \cdot 10000^{m/16}$ of the first pairs are 6.3, 11, 20, 35,
63, and 112 positions, and the seventh has a period of 199. Six of the 16
pairs are therefore fast in this sense. The other ten reach angles at long
offsets that they never took during training. Because the angles vary
continuously, they can be *rescaled* so that the deployed range maps into
the trained one. Multiplying every angle by $128/512$ makes the offset
$4\delta$ look like the familiar $\delta$, so the model interpolates
instead of extrapolating. This is *position interpolation*
:cite:`Chen.Wong.Chen.ea.2023`, which extended Llama models from 2,048 to
32,768 tokens with at most 1,000 fine-tuning steps.

Position interpolation rescales every pair, including the fast ones that
had already met every angle. Should the factor apply to all pairs, or only
to the slow pairs that meet new angles? Two arguments of `rope_tables` and
`rope` that we have kept at their defaults let us try both: `scale`
multiplies the angle of each pair by its own factor, and `base` replaces
the constant $10^4$. `TinyCharLM` stores them as the attributes
`rope_scale` and `rope_base`, so that they can be changed after training.
We apply each rescaling to the three trained RoPE models without any
training: the uniform factor $128/512$ for all pairs, and the same factor
for the slow pairs only. The cell first counts the slow pairs, which should
come to the ten found above.

```{.python .input #positional-information-extending-a-trained-model-s-context}
%%tab pytorch
rope_model = models['rope']
d_head = 2 * rope_model.rope_scale.numel()
periods = (2 * math.pi
           * rope_model.rope_base ** (torch.arange(0, d_head, 2) / d_head))
slow = periods > train_len
print(f'{int(slow.sum())} of {d_head // 2} pairs have periods above '
      f'{train_len}')
ones = torch.ones(d_head // 2)
shrink = train_len / max(contexts)  # Longest context into the trained one
factors = {'unscaled': ones, 'all pairs': ones * shrink,
           'slow pairs': torch.where(slow, ones * shrink, ones)}
print(f'{"context":>10}: ' + ''.join(f'{n:>18}' for n in contexts))
for name, factor in factors.items():  # Each of the three trained RoPE runs
    ppl = []
    for rope_model in runs['rope']:
        rope_model.rope_scale.copy_(factor)
        ppl.append(perplexities(rope_model))
        rope_model.rope_scale.copy_(ones)
    print(f'{name:>10}: ' + ''.join(f'{mean_range(col):>18}'
                                    for col in zip(*ppl)))
```

```{.python .input #positional-information-extending-a-trained-model-s-context}
%%tab jax
rope_model = models['rope']
d_head = 2 * rope_model.rope_scale.shape[0]
periods = (2 * math.pi
           * rope_model.rope_base ** (jnp.arange(0, d_head, 2) / d_head))
slow = periods > train_len
print(f'{int(slow.sum())} of {d_head // 2} pairs have periods above '
      f'{train_len}')
ones = jnp.ones(d_head // 2)
shrink = train_len / max(contexts)  # Longest context into the trained one
factors = {'unscaled': ones, 'all pairs': ones * shrink,
           'slow pairs': jnp.where(slow, ones * shrink, ones)}
print(f'{"context":>10}: ' + ''.join(f'{n:>18}' for n in contexts))
for name, factor in factors.items():  # Each of the three trained RoPE runs
    ppl = []
    for rope_model in runs['rope']:
        rope_model.rope_scale[...] = factor
        ppl.append(perplexities(rope_model))
        rope_model.rope_scale[...] = ones
    print(f'{name:>10}: ' + ''.join(f'{mean_range(col):>18}'
                                    for col in zip(*ppl)))
```

The uniform factor fails outright. Its perplexity jumps to about 40
already at the training length and stays there at every context, worse
than the bigram's 11.6 and, at 512, worse even than leaving the angles
alone. The fast pairs, which needed no help, now turn four times more
slowly, and the angles by which the heads told neighboring bytes apart are
gone.

Rescaling only the slow pairs brings the perplexity at context 512 down to
about 9, a third of the unscaled value, without changing a single weight.
The repair is not free. At the training length the perplexity of the
rescaled model rises by about 0.7 to 0.8 (about 8 against 7.3), more than
the spread between runs, and at 512 it stays some 20 to 25% above what the
unscaled model reaches at 128. No weight was ever trained on compressed
angles, which is why the published recipes follow the rescaling with a
short fine-tune at the new length; Exercise
:exref:`ex_positional-information-position-interpolation` carries it out on
this model.

The split into fast and slow pairs is the core of YaRN
:cite:`Peng.Quesnelle.Fan.ea.2024`, which also ramps between the two groups
and adds a softmax temperature. NTK-aware scaling, a community method that
the YaRN paper formalizes, reaches a similar effect by changing only the
base $b = 10^4$. To grow the context by a factor $s$ (here $s = 4$), it
replaces $b$ by $b' = b\, s^{d_h/(d_h-2)}$. Pair $m$ then turns more
slowly by the factor $s^{2m/(d_h-2)}$, which grows smoothly from 1 for the
fastest pair to exactly $s$ for the slowest. Training with a larger base
is the trained form of the same idea, and :numref:`subsec_attention-fr-rope`
lists the bases that released models use.

What should one choose? For a model that will be trained, or at least
fine-tuned, at the lengths it must serve, RoPE. Its scores depend on the
offset alone by construction, and its failure beyond the training length
sits in the slow pairs, where rescaling repairs most of the damage. The
recommendation does not rest on the size of its lead in range, which
belongs to our small model. ALiBi's flat curve
guards against breaking, but in our model it read nothing useful past the
training window. Whatever the choice, it decides what happens at 512
bytes. A model that reads a novel meets a different limit first: one
attention score for every pair of positions.

## Summary

Unmasked self-attention is permutation equivariant, so a model built from
it cannot tell where a token stands unless it is told. Absolute schemes add
a position vector to each token, from a learned table whose rows beyond
the training length are never trained, or from a sinusoidal table whose
shifts are rotations of column pairs; added to the tokens, either leaves
absolute position in the score. Requiring that every score depend only on
the offset forces rotations onto queries and keys, which is RoPE, while
ALiBi subtracts a per-head distance penalty and NoPE relies on the causal
mask alone. In a small attention-only byte model trained at context 128,
RoPE gives by far the lowest perplexity in range, and at four times that
length the learned table, the sinusoidal table, and RoPE degrade while
ALiBi stays flat, because what it reads beyond the training window
contributes nothing. For RoPE only the slow pairs meet unseen angles, and
rescaling just those recovers most of the loss without any training, while
rescaling every pair ruins the model. The slow-pair repair costs a little
at the training length, which is why practical recipes add a short
fine-tune.

## Exercises

1. **Distinct positions, distinct codes.** Let $\mathbf{p}_i$ be row $i$ of
   the sinusoidal table :eqref:`eq_sinusoidal-def` of width $d$, with the
   frequencies $\omega_m = 1/10000^{2m/d}$.
    1. Show that
       $$\mathbf{p}_i^\top \mathbf{p}_{i+\delta} = \sum_{m=0}^{d/2-1} \cos(\delta\,\omega_m),$$
       derive $\|\mathbf{p}_i - \mathbf{p}_{i+\delta}\|^2$ from it, and
       conclude that both quantities depend on the offset $\delta$ alone.
    1. Conclude that $\mathbf{p}_i = \mathbf{p}_{i'}$ only if
       $(i - i')\,\omega_m$ is a multiple of $2\pi$ for every $m$, and that
       no two distinct integer positions share a row, whatever the value of
       `max_len`.
    1. The fastest pair has period $2\pi$, so it nearly repeats at integer
       offsets close to multiples of $2\pi$. Find the offset $\delta < 512$
       at which its angle comes closest to a multiple of $2\pi$, and give
       the residual angle. For $d = 32$, compare
       $\|\mathbf{p}_i - \mathbf{p}_{i+\delta}\|$ at this offset with its
       value at $\delta = 1$, say which pairs keep the two positions of the
       near repeat apart, and relate the answer to the binary counting
       of :numref:`subsec_pos-sinusoidal`.
   :exlabel:`ex_positional-information-distinct-positions-distinct-codes`

1. [code] **Untrained rows.** The table `pos_emb` of `models['learned']`
   has `max_len = 512` rows, of which training at context 128 used only the
   first 128.
    1. Construct a new `TinyCharLM` with `pos='learned'`, copy its table
       `pos_emb` before training, and train it as in the experiment
       of :numref:`subsec_pos-experiment`. Explain from Adam's two moment
       estimates why rows 128 to 511 of the trained table are bitwise equal
       to the copy, and check the claim. By
       what factor would decoupled weight decay with coefficient $0.1$ at
       the learning rate $10^{-3}$ have shrunk these rows over the 3000
       training steps?
    1. Compute the cosine similarities between all 512 rows of the trained
       table and display them with `d2l.show_heatmaps`. Describe the three
       blocks of the result: trained rows against trained rows, untrained
       against untrained, and trained against untrained.
   :exlabel:`ex_positional-information-untrained-rows`

1. **Additive codes in the score.** With the additive encoding of
   :numref:`subsec_pos-sinusoidal`, a first-layer head scores the inputs at
   positions $i$ and $j$ by the four terms of :eqref:`eq_pos-four-terms`,
   with $\mathbf{A} = \mathbf{W}_q \mathbf{W}_k^\top / \sqrt{d_h}$.
    1. Let $\mathbf{J}$ be the rotation of the plane by $\pi/2$. Using
       :eqref:`eq_sinusoidal-rotation`, show that the position--position
       term $\mathbf{p}_i^\top \mathbf{A}\, \mathbf{p}_j$ depends only on
       $j - i$ when $\mathbf{A}$ is block diagonal with $2 \times 2$ blocks
       $a_m \mathbf{I} + b_m \mathbf{J}$, one per frequency pair. Exercise :exref:`ex_positional-information-distinct-positions-distinct-codes`
       treats $\mathbf{A} = \mathbf{I}$. Then show that a single nonzero
       entry of $\mathbf{A}$ that couples two different frequency pairs
       destroys this property.
    1. Determine when the content--position term
       $\mathbf{x}_i^\top \mathbf{A}\, \mathbf{p}_j$ is unchanged for every
       shift of $j$. Use the answer to explain the score change of the
       additive encoding in the shift test of :numref:`subsec_pos-rope`,
       and why RoPE has no content--position term.
   :exlabel:`ex_positional-information-additive-codes`

1. [code] **Where extrapolation fails.** The perplexities of
   :numref:`subsec_pos-experiment` at context 512 average the loss over all
   positions of each sequence.
    1. Check numerically that in each of the five models the logits at
       positions 0 to 127 of a length-512 sequence agree, up to
       floating-point round-off, with those computed from its first 128
       tokens alone. Which two properties of `TinyCharLM` guarantee this
       for every scheme?
    1. Compute the validation loss at every position of the length-512
       sequences for the five models, average it over windows of 16
       positions, and plot it against position. For each scheme, where does
       the curve leave the level it holds below position 128, and does the
       loss jump at position 128 or grow gradually?
    1. Restrict every query of `models['rope']` to the 128 most recent
       keys, itself included, by adding a band constraint to the causal
       mask in a subclass that overrides `attend`, copy the trained
       weights into it, and repeat the measurement. Compare the windowed
       curve beyond position 128 with the unwindowed curve below it. Do
       offsets beyond 127 account for the degradation of RoPE at context
       512?
   :exlabel:`ex_positional-information-where-extrapolation-fails`

1. **Biases that keep the symmetry.** ● Suppose that a fixed matrix
   $\mathbf{B} \in \mathbb{R}^{n \times n}$ is added to the score matrix
   $\mathbf{S}$ of the proof before the softmax, so that the layer
   :eqref:`eq_self-attention` computes $\mathrm{softmax}(\mathbf{S} +
   \mathbf{B})\, \mathbf{X}\mathbf{W}_v$.
    1. Show that the proposition still holds for every permutation matrix
       $\boldsymbol{\Pi}$ with $\boldsymbol{\Pi}^\top \mathbf{B}
       \boldsymbol{\Pi} = \mathbf{B}$.
    1. Find all matrices $\mathbf{B}$ that satisfy this condition for every
       permutation, and describe how each of them changes the attention
       weights.
    1. Without the causal mask, the ALiBi bias of :eqref:`eq_alibi-def`
       extends symmetrically to $B_{ij} = -m_h\,|i - j|$. Determine the
       permutations that preserve this matrix, and state what the result
       implies for an unmasked layer that receives this bias and no other
       positional signal. Explain why the causal ALiBi of
       :eqref:`eq_alibi-def` does not share this symmetry.

1. **Offset preference in RoPE.** ● Split a query $\mathbf{q}$ and a key
   $\mathbf{k}$ into the $d_h/2$ two-dimensional pairs $\mathbf{q}^{(m)},
   \mathbf{k}^{(m)} \in \mathbb{R}^2$ on which :eqref:`eq_rope-def` acts,
   and let $\delta = j - i$ be the offset of the key position $j$ from the
   query position $i$.
    1. Show that the RoPE score :eqref:`eq_rope-goal` equals
       $$\sum_{m=0}^{d_h/2-1} \|\mathbf{q}^{(m)}\|\, \|\mathbf{k}^{(m)}\| \cos(\delta\,\omega_m - \psi_m),$$
       where $\psi_m$ is the angle from $\mathbf{k}^{(m)}$ to
       $\mathbf{q}^{(m)}$ (the $\varphi$ of :numref:`fig_rope-rotation`
       with the opposite sign).
    1. Choose the angles $\psi_m$ so that every pair contributes its
       maximum at the offset $\delta = -1$, the previous token. With all
       products $\|\mathbf{q}^{(m)}\|\, \|\mathbf{k}^{(m)}\|$ equal to $c$,
       show that the score, as a function of $\delta$, is $c$ times a
       shifted copy of the dot product $\mathbf{p}_i^\top
       \mathbf{p}_{i+\delta}$ of Exercise :exref:`ex_positional-information-distinct-positions-distinct-codes` for a sinusoidal table of width
       $d_h$.
    1. Take $d_h = 32$ and the query at position 127, which attends to the
       offsets $\delta = -127, \ldots, 0$. Let the attention logits equal
       these scores, with the factor $1/\sqrt{d_h}$ absorbed into $c$. Find
       the $c$ at which the offset $-1$ receives half of the softmax
       weight, and identify the offsets that compete most with it. Which
       pairs set the gap between the score at $-1$ and the scores at its
       neighbors?
   :exlabel:`ex_positional-information-offset-preference-rope`

1. [code] **The RoPE base.** ● In :eqref:`eq_rope-def` the frequencies are
   $\omega_m = 10000^{-2m/d_h}$, and the constant $10000$ is a choice.
    1. For the head width $d_h = 32$ of `TinyCharLM`, compute the periods
       $2\pi/\omega_m$ of the 16 pairs for the bases $100$, $10^4$, and
       $10^6$, and count the pairs that complete at least one full rotation
       within 128 positions. Which period does not depend on the base?
    1. Retrain the `'rope'` variant of `TinyCharLM` with bases $100$ and
       $10^6$ by setting its attribute `rope_base` after construction.
       Train three models with each of the two bases, and compare them with
       the three base-$10^4$ models of the experiment, so that differences
       between bases can be set against the variation across runs. Compare
       perplexity at 128, 256, and 512.
    1. Explain from the counts why the base is expected to matter more at
       length 512 than at 128, and check the expectation against your
       results.
   :exlabel:`ex_positional-information-rope-base`

1. [extended] **Position interpolation.** ● The last cell of
   :numref:`subsec_pos-context-extension` rescales the angles of the
   trained RoPE models without any training. This problem explains its
   result, adds the base change of NTK-aware scaling, and completes the
   recipe with a fine-tune.
    1. Show that a pair that completes a full rotation within the training
       offsets 0 to 127 has already met, to within $\omega_m/2$ modulo
       $2\pi$, every angle that a longer offset produces. Compute the gap
       of Exercise :exref:`ex_positional-information-offset-preference-rope`
       between the score at offset $-1$ and the scores at its neighbors for
       $d_h = 32$, before the rescaling, after the uniform rescaling by
       $128/512$, and after the rescaling of only the pairs whose period
       exceeds 128 positions. Relate the result to the perplexities at
       context 128 in the cell.
    1. Show that the base $b' = b\, s^{d_h/(d_h-2)}$ leaves $\omega_0$
       unchanged and divides $\omega_{d_h/2-1}$ by $s$. Set `rope_base` of
       `models['rope']` to this base for $s = 4$, evaluate at 128, 256,
       and 512 without training, and compare with the two rescalings of
       the cell.
    1. Fine-tune for a few hundred steps at context 512, on
       `d2l.WikiText103(batch_size=16, num_steps=512)`, with each of the two
       rescalings of the cell, each starting from a copy of the trained
       `models['rope']`. Re-evaluate at 128, 256, and 512, and compare with
       the unscaled model and with the ALiBi model of the experiment in
       :numref:`subsec_pos-experiment` at the same lengths.
   :exlabel:`ex_positional-information-position-interpolation`

[Discussions](https://d2l.discourse.group/t/36771)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.4]{.kicker}

Positional Information<br>
**permutation equivariance · learned and sinusoidal tables · RoPE · train short, test long**
:::
:::

::: {.slide title="Attention Alone Cannot Tell “Dog Bites Man” from “Man Bites Dog”"}
[The problem]{.kicker}

- Self-attention maps a shuffled sentence to the **same outputs,
  shuffled**.
- A classifier that pools the outputs cannot tell who bit whom.
- GPT-2 small learns a table of **1,024** position vectors: token 1,025
  has no position at all.

::: {.d2l-note}
How do we give attention a sense of position, and what happens beyond the
length it was trained on?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- prove that unmasked attention is permutation equivariant;
- read a shift of the sinusoidal table as a rotation, and see why adding
  it leaves absolute position in the score;
- derive RoPE from "scores depend only on the offset";
- predict which schemes break at four times the training length, and
  test the prediction;
- repair a trained RoPE model at four times its context by rescaling
  angles.

**Running example:** `TinyCharLM`, an attention-only byte model (width 128,
two blocks of four heads of width 32), trained at context 128, tested at
512.
:::

::: {.slide title="Positional Information Lets Attention See Order"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| Attention Scoring and Masking | scaled dot products, masks |
| Self-Attention and Multiple Heads | a sequence attends to itself |
| **Positional Information** | **where a token stands** |
| The Cost of Attention | quadratic cost, exact fast kernels |
| Windowed and Linear Attention | cheaper attention, what it gives up |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** self-attention, causal masks, the byte data and training
  loop of `TinyLM`.
- **Skips:** learned relative-position vectors, long-context training.
:::
:::
:::

::: {.slide title="Unmasked Self-Attention Is Permutation Equivariant"}
[Permutation equivariance]{.kicker}

**Proposition.** $f(\boldsymbol{\Pi}\mathbf{X}) = \boldsymbol{\Pi}
f(\mathbf{X})$ for every permutation matrix $\boldsymbol{\Pi}$, where
$f(\mathbf{X}) = \mathrm{softmax}\big(\mathbf{X}\mathbf{W}_q
(\mathbf{X}\mathbf{W}_k)^\top/\sqrt{d_h}\big)\, \mathbf{X}\mathbf{W}_v$
and $d_h$ is the head width.

**Proof sketch.** Projections act row-wise; scores become
$\boldsymbol{\Pi}\mathbf{S}\boldsymbol{\Pi}^\top$; row-softmax commutes with
permutations; $\boldsymbol{\Pi}^\top\boldsymbol{\Pi} = \mathbf{I}$. $\blacksquare$

. . .

- Stacking equivariant layers stays equivariant: depth does not help.
:::

::: {.slide title="Shuffled Tokens Give Shuffled Outputs"}
[Permutation equivariance]{.kicker}

Eight random tokens through a four-head layer, then shuffled; the outputs
agree up to float32 round-off:

@!positional-information-attention-ignores-order

::: {.d2l-note}
A **causal mask** breaks the symmetry: position $i$ attends over $i+1$
tokens.
:::
:::

::: {.slide title="A Learned Table Knows Only the Positions It Was Trained On"}
[Absolute embeddings]{.kicker}

- Add a vector $\mathbf{p}_i$ to the token at position $i$: a shuffle
  gives $\boldsymbol{\Pi}\mathbf{X} + \mathbf{P} \neq
  \boldsymbol{\Pi}(\mathbf{X} + \mathbf{P})$, and the symmetry breaks.
- **Learned** (BERT, GPT-2): one free vector per position.
- A row that no training example used keeps its random initialization.

::: {.d2l-note}
Trained at context 128, rows 128 to 511 of a 512-row table never
receive a gradient.
:::
:::

::: {.slide title="Sinusoids Give Each Position a Code at Every Frequency"}
[Absolute embeddings]{.kicker}

Pair $m$ of the $d$ columns turns at $\omega_m = 10000^{-2m/d}$ radians
per position (Vaswani et al., 2017):

$$p_{i, 2m} = \sin(i\,\omega_m), \qquad p_{i, 2m+1} = \cos(i\,\omega_m)$$

@!positional-information-sinusoidal-encodings-1
:::

::: {.slide title="Sinusoids Are a Continuous Binary Counter"}
[Absolute embeddings]{.kicker}

Counting in binary, the lowest bit flips every step, the next every two,
the next every four. The table does the same, smoothly:

@!positional-information-sinusoidal-encodings-3

Fast columns separate neighbors; slow columns separate regions.
:::

::: {.slide title="Check: Which of These Break the Symmetry?"}
[Check]{.kicker}

1. a second attention layer on top of the first;
2. the same learned vector added to every token;
3. a causal mask;
4. a different vector $\mathbf{p}_i$ added at each position.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 3 and 4. A stack of equivariant layers is equivariant, and a
vector shared by all rows moves with any shuffle.
:::
:::
:::

::: {.slide title="A Shift Rotates Every Column Pair by an Angle Set by the Offset"}
[Absolute embeddings]{.kicker}

$$\begin{bmatrix} \cos(\delta\omega_m) & \sin(\delta\omega_m) \\ -\sin(\delta\omega_m) & \cos(\delta\omega_m) \end{bmatrix}\begin{bmatrix} p_{i, 2m} \\ p_{i, 2m+1} \end{bmatrix} = \begin{bmatrix} p_{i+\delta, 2m} \\ p_{i+\delta, 2m+1} \end{bmatrix}$$

- The angle $\delta\omega_m$ depends on the offset $\delta$, never on $i$.

::: {.d2l-note}
The table contains relative position. Does attention see it?
:::
:::

::: {.slide title="An Added Code Ties the Score to Absolute Positions"}
[Absolute embeddings]{.kicker}

With $\mathbf{A} = \mathbf{W}_q\mathbf{W}_k^\top/\sqrt{d_h}$, the score of
positions $i$ and $j$ has four terms:

$$(\mathbf{x}_i + \mathbf{p}_i)^\top \mathbf{A} (\mathbf{x}_j + \mathbf{p}_j) = \mathbf{x}_i^\top\mathbf{A}\mathbf{x}_j + \mathbf{x}_i^\top\mathbf{A}\mathbf{p}_j + \mathbf{p}_i^\top\mathbf{A}\mathbf{x}_j + \mathbf{p}_i^\top\mathbf{A}\mathbf{p}_j$$

- content–content: no position;
- the two cross terms: content of one token, **absolute** position of the
  other;
- position–position: offset only if $\mathbf{A}$ acts on each frequency
  pair as a scaled rotation and never mixes two pairs.
:::

::: {.slide title="Requiring Offset-Only Scores Forces Rotations"}
[Rotary embeddings]{.kicker}

Require $(\mathbf{R}_i \mathbf{q})^\top(\mathbf{R}_j \mathbf{k}) =
\mathbf{q}^\top \mathbf{R}_{j-i}\, \mathbf{k}$ for queries at $i$, keys
at $j$:

- I. $i = j = 0$: $\mathbf{R}_0$ is a projection; keep every direction,
  so $\mathbf{R}_0 = \mathbf{I}$.
- II. $i = j$: $\mathbf{R}_i^\top \mathbf{R}_i = \mathbf{I}$, each
  $\mathbf{R}_i$ is orthogonal.
- III. $j = 0$: $\mathbf{R}_{-i} = \mathbf{R}_i^\top$, so
  $\mathbf{R}_i = \mathbf{R}_1^i$.
- IV. An orthogonal $\mathbf{R}_1$ rotates coordinate pairs (in a basis the
  projections absorb).

::: {.d2l-note}
**RoPE:** rotate pair $m$ of every query and key at position $i$ by
$i\,\omega_m$, with $\omega_m = 10000^{-2m/d_h}$.
:::
:::

::: {.slide title="Shifting Both Positions Rotates q and k Together"}
[Rotary embeddings]{.kicker}

![](../img/mdl-attention-rope-rotation.svg){width=80%}

$\theta = \omega_m$ per position; $\varphi$ is the angle between the
unrotated $\mathbf{q}$ and $\mathbf{k}$. After any common shift the angle
between the two vectors, and the score, stay the same. Values are not
rotated.
:::

::: {.slide title="Check: How Far Apart Are the Query and the Key?"}
[Check]{.kicker}

Pair 0 turns one radian per position. Before rotation, the pair-0 parts of
a query and a key point the same way. The query sits at position 3, the
key at position 5. What angle separates them after RoPE? And at positions
103 and 105?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Two radians in both cases: the query turns by 3 or 103
radians, the key by 5 or 105, and only the offset of 2 survives.
:::
:::
:::

::: {.slide title="RoPE Rotates Even/Odd Pairs by Position Angles" only="pytorch"}
[Rotary embeddings]{.kicker}

~~~python
def rope(x, offset=0, base=10000.0):
    n, d = x.shape[-2:]
    pos = torch.arange(n, dtype=torch.float32) + offset
    omega = base ** (-torch.arange(0, d, 2) / d)
    theta = pos[:, None] * omega[None, :]      # (n, d/2) angles
    cos, sin = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]       # even/odd pairs
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], -1).flatten(-2)
~~~

`d2l.rope` adds per-pair `scale` factors for context extension.
:::

::: {.slide title="RoPE Rotates Even/Odd Pairs by Position Angles" only="jax"}
[Rotary embeddings]{.kicker}

~~~python
def rope(x, offset=0, base=10000.0):
    n, d = x.shape[-2:]
    pos = jnp.arange(n, dtype=jnp.float32) + offset
    omega = base ** (-jnp.arange(0, d, 2) / d)
    theta = pos[:, None] * omega[None, :]      # (n, d/2) angles
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]       # even/odd pairs
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], -1).reshape(x.shape)
~~~

`d2l.rope` adds per-pair `scale` factors for context extension.
:::

::: {.slide title="Shifting All Positions Leaves RoPE Scores Unchanged"}
[Rotary embeddings]{.kicker}

32 random queries and keys of width 64, all positions shifted together:

@!positional-information-rotary-position-embeddings-2

::: {.d2l-note}
RoPE changes only at round-off, even at shift 480. The additive code's
cross terms move the scores by almost 3, as much as the scores vary.
:::
:::

::: {.slide title="ALiBi Penalizes Distance; NoPE Uses Only the Mask"}
[Two more schemes]{.kicker}

- **ALiBi** (Press et al., 2022): no position vectors; subtract a
  per-head penalty,
  $\mathrm{score}_{ij} = \mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_h} - m_h\,(i-j)$,
  with $m_h = 2^{-8h/H}$; for $H = 4$: $1/4, 1/16, 1/64, 1/256$.
- Slope $1/4$: a key 40 back loses a factor $e^{-10}$. A formula in the
  distance, not a table, so nothing can break.
- **NoPE** (Kazemnejad et al., 2023): no encoding; a causal model can
  read position from how many keys each query sees.
:::

::: {.slide title="Position Enters TinyCharLM in One of Three Places"}
[The experiment]{.kicker}

| scheme | acts on | how | in the code |
|---|---|---|---|
| learned, sinusoidal | the input | add $\mathbf{p}_i$ to the token embedding | `embed`, once |
| RoPE | queries and keys | rotate them | `attend`, every block |
| ALiBi | the scores | subtract $m_h\,(i - j)$ | `attend`, every block |
| NoPE | nothing | only the causal mask | — |

- Tied output head, no feed-forward layer, no normalization.
- WikiText-103 bytes; 3,000 AdamW steps at context 128; three runs per
  scheme.
:::

::: {.slide title="Trained at 128: RoPE Is Lowest, NoPE Sits at Bigram Level"}
[The experiment]{.kicker}

Validation perplexity per byte, mean (min–max) of three runs:

@!positional-information-the-experiment-3

@!positional-information-byte-bigram-baseline

::: {.d2l-note}
At 128: RoPE lowest at 7.3; learned, sinusoidal, and ALiBi all about 15%
below the bigram; NoPE at the bigram, whose current byte is already in the
residual stream.
:::
:::

::: {.slide title="Check: Which Logits Change Between Context 128 and 512?"}
[Check]{.kicker}

The same models read the first 128 bytes of a validation sequence,
then the first 512. At which positions can the logits differ?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Only at positions 128 to 511. Causal attention lets position
$i$ see only positions $0, \ldots, i$, so positions 0 to 127 compute the
same logits in both runs. The whole change in perplexity comes from the
positions beyond the training length.
:::
:::
:::

::: {.slide title="Each Scheme Holds or Fails Beyond 128 for Its Own Reason"}
[The experiment]{.kicker}

- **Learned:** rows 128 to 511 never trained; doubles or worse.
- **Sinusoidal:** slow columns take new values, smoothly; about 60%
  worse, the mildest.
- **RoPE:** slow pairs reach unseen angles; nearly 4× worse on average.
- **ALiBi:** a formula in the distance, nothing to run out of; flat.
- **NoPE:** flat too, but at bigram level: no position learned, none to
  lose.

::: {.d2l-note}
The model that used position best in range has the most to lose out of
range.
:::
:::

::: {.slide title="ALiBi Stays Flat Because Far Keys Contribute Nothing"}
[The experiment]{.kicker}

Share of each head's weight on keys 128 or more back, then perplexity at
512 with those keys removed:

@!positional-information-alibi-weight-beyond-the-window

@!positional-information-alibi-without-far-keys

::: {.d2l-note}
Heads 3 and 4 read far keys, as the bias dictates; removing them changes
nothing. They average broadly, and the last 128 bytes say as much as the
last 512. Flat perplexity does not mean the model uses the longer context.
:::
:::

::: {.slide title="Only the Slow Pairs Meet New Angles Beyond 128"}
[Context extension]{.kicker}

- Head width 32, base $10^4$: periods $2\pi \cdot 10^{4m/16}$ are 6.3,
  11, 20, 35, 63, 112 (fast) and 199 to 35,000 (slow).
- A fast pair completes a turn within 128 positions: every angle at offset
  400 was already seen in training.
- A slow pair meets new angles. Rescale them so that 512 maps into 128.
- **Position interpolation** (Chen et al., 2023) multiplies *all* angles by
  $128/512$. All pairs, or only the slow ones?
:::

::: {.slide title="Rescaling Only the Slow Pairs Repairs Context 512 Without Training"}
[Context extension]{.kicker}

The three trained RoPE models, angle factors changed, no weight updated:

@!positional-information-extending-a-trained-model-s-context

::: {.d2l-note}
All pairs: about 40 everywhere, worse than the bigram. Slow pairs: about 9
at 512, a third of unscaled, at a cost of about 0.7 to 0.8 at 128. A short
fine-tune completes the recipe.
:::
:::

::: {.slide title="YaRN and NTK-Aware Scaling Spare the Fast Pairs"}
[Context extension]{.kicker}

- **YaRN** (Peng et al., 2024): interpolate only the slow pairs, ramp
  between the groups, add a softmax temperature.
- **NTK-aware base** $b' = b\, s^{d_h/(d_h-2)}$: pair $m$ slows by
  $s^{2m/(d_h-2)}$, from 1 for the fastest pair to $s$ for the slowest.
- Training with a larger base is the trained form of the same idea.
:::

::: {.slide title="Check: Which Base Grows the Context Fourfold?"}
[Check]{.kicker}

Head width $d_h = 32$, base $b = 10^4$, growth $s = 4$. Compute the
NTK-aware base $b' = b\, s^{d_h/(d_h-2)}$. Which pair keeps its frequency?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $b' = 10^4 \cdot 4^{32/30} \approx 4.4 \times 10^4$. Pair 0
turns one radian per position for every base; the slowest pair turns four
times slower.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Unmasked attention is permutation equivariant: shuffled inputs give
  shuffled outputs.
- Added codes (learned, sinusoidal) leave absolute position in the score;
  RoPE's rotations make it depend on the offset alone.
- Trained at 128: RoPE 7.3, NoPE at the bigram's 11.6. At 512: learned,
  sinusoidal, and RoPE degrade; ALiBi stays flat but reads nothing useful
  beyond 128.
- Rescaling RoPE's slow pairs brings 512 to about 9 without training.

**Next:** attention scores every pair of positions. What does that cost,
and how much of it can be avoided without changing the result?
:::
