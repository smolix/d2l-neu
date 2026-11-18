# Windowed and Linear Attention
:label:`sec_windowed-linear-attention`

Let us hand a language model a whole novel and ask it a question about the
ending. Every attention layer compares each new token with every token
before it, a hundred thousand of them or more. Does the last word of the
book really need every word of the first chapter? Usually it does not: what
a token needs is mostly nearby. Sometimes it does, because the answer is a
name that appeared once, early on. The fused kernel of
:numref:`sec_attention-at-scale` treats every token as the hard case. It
removed the stored score matrix but none of the scores, so doubling the context
still quadruples the arithmetic, and the keys and values kept for
generation still grow with every token. The score matrix is gone; the
arithmetic and the cache remain.

The question became pressing around 2019 and 2020, when the first
transformers for long documents ran out of memory. Of the many remedies
proposed then, two simple ones survive, in some form, in the models of
today. The first scores fewer pairs: each query looks only at its most
recent keys, a *sliding window*
:cite:`child2019generating,beltagy2020longformer`. The second changes the
score so that the sum over keys can be computed before any query arrives,
and attention becomes a recurrent network whose state is a matrix
:cite:`Katharopoulos.Vyas.Pappas.ea.2020`. Both make the cost linear in
$n$. Neither computes what dense attention computes, and the question of
this section is what each of them gives up. The other remedies are collected
in :numref:`sec_attention-further-reading`.

We take windows first. They need one more mask beyond those of
:numref:`sec_attention-scoring-functions`, and counting the tiles that a
fused kernel can skip predicts how much they save. We then derive linear
attention, implement it in three equivalent forms, and test it on what
softmax attention does best: putting nearly all of a query's weight on one
key. Finally we time the window against dense attention and say which
mechanism to use. The cells that call fused kernels expect a CUDA GPU.

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-2}
%%tab pytorch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
```

## Sliding-Window Attention

### The Band Mask

The simplest way to score fewer pairs is locality: each query attends only
to the $w$ most recent positions, itself included. With the masks of
:numref:`sec_attention-scoring-functions` this is one more condition joined
by AND. A key $j$ is allowed for query $i$ if it is not in the future,
$j \le i$, and not too far in the past, $i - j < w$. The causal triangle
becomes a band of width $w$ along the diagonal.

A small case shows the saving. On 12 positions the causal mask keeps 78 of
the 144 pairs. A window of $w = 4$ keeps 42: query 7, for instance, reads
keys 4, 5, 6 and 7, and the first three queries read what little precedes
them. For large $n$ the band holds about $nw$ pairs where the triangle
holds $n^2/2$, and the mask itself is one line of code.

```{.python .input #attention-at-scale-attention-through-a-window}
%%tab pytorch
def window_mask(n, w, device=None):
    """Causal band: query i may attend to keys i - w + 1, ..., i."""
    i = torch.arange(n, device=device)
    return d2l.causal_mask(n, device=device) & (i[:, None] - i[None, :] < w)
```

```{.python .input #attention-at-scale-attention-through-a-window}
%%tab jax
def window_mask(n, w):
    """Causal band: query i may attend to keys i - w + 1, ..., i."""
    i = jnp.arange(n)
    return d2l.causal_mask(n) & (i[:, None] - i[None, :] < w)
```

A window is a bet that what a token needs is nearby. For most tokens of a
text the bet is good. It is lost exactly where it hurts, on the name
introduced three pages earlier.

### Receptive Field Growth with Depth
:label:`subsec_attn-receptive-field`

A window of width $w$ excludes direct interactions beyond $w$ tokens, but
depth widens the receptive field, as it does in a stack of convolutions
(:numref:`chap_cnn`). A query in layer 2 attends to keys that are
themselves layer-1 outputs, each of which already summarizes its own
window. Information therefore travels up to $w - 1$ positions per layer,
and after $L$ layers a token's receptive field spans

$$1 + L(w - 1)$$

positions. With $w = 4$, one layer reaches 4 positions, two layers 7, and
four layers 13. Mistral 7B attends through a window of 4,096 in each of its
32 layers :cite:`Jiang.Sablayrolles.Mensch.ea.2023`, so on paper its last
layer reaches $1 + 32 \cdot 4095 = 131{,}041$ positions.

Reach is not use. Full self-attention connects any two positions in one
layer (:numref:`sec_multihead-attention`); a stack of windows needs about
$n/w$ hops, one per layer. A fact that must travel 100,000 tokens through
Mistral's windows has to be relayed by at least 25 of its 32 layers, each
carrying it almost the full width of its window. At every hop the fact is
one entry of a weighted average over thousands of positions, and it must
win enough weight at each hop to arrive at all.

How much weight does it keep? Let us take the simplest case, layers that
average their window with equal weights. With $w = 4$ and four layers, the
farthest token the last layer can reach, 12 positions back, arrives through
a single path of four hops, each of which gives it a quarter: its weight is
$(1/4)^4 = 1/256$. One global layer averaging the same 13 positions would
give it $1/13$, twenty times more. Near the edge of its reach, a stack of
windows that merely averages passes on very little; a distant fact arrives
only if every hop on its path puts most of its weight on it.
Nothing in the architecture guarantees that training finds such a relay. A
stack of windows therefore assumes that few dependencies need more than a
small number of hops.

### Global Layers

More recent models interleave windowed layers with a few layers of full
attention, called *global* layers. A global layer connects every pair in
one hop again, and the windowed layers keep both the arithmetic and the
memory of generation small. A model that generates text keeps the keys and
values of all past tokens so that it need not recompute them
(:numref:`sec_multihead-attention`). This store is the *KV cache*, and a
windowed layer needs only the last $w$ of its entries. Gemma 3 follows
every five windowed layers of width 1,024 with one global layer
:cite:`Gemma.Team.2025`.

How much cache does that save? With $r$ windowed layers per global layer,
each windowed layer caches $\min(n, w)$ positions and the global layer all
$n$, against $(r + 1)n$ positions for global layers throughout. The cache
therefore shrinks to the fraction

$$
\frac{r \min(n, w) + n}{(r + 1)\, n}
$$

of the all-global cache. Once $n$ is much larger than $w$, only the global
layers count, and the fraction tends to $1/(r + 1)$. For Gemma 3's $r = 5$
and $w = 1024$, it is about a quarter at $n = 8192$ and approaches $1/6$ as
the context grows.

### Tiles in a Fused Kernel

The mask defines the result, but by itself it saves no work. A masked
softmax over a band still scores all $n^2$ pairs and then discards most of
them. An implementation that saves work must never form the excluded
scores. One way is to process the sequence in blocks of $w$ queries. A
query can attend only to keys in its own block and in the block before it,
so each block of queries needs a $w \times 2w$ block of scores. The total
is $2nw$ scores instead of $n^2$, linear in $n$ for fixed $w$. The function
below does exactly this. Its only fiddly part is the first block, which has
no block before it; that index arithmetic can be skimmed.

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab pytorch
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d = Q.shape
    assert n % w == 0, 'blocks must tile the sequence exactly'
    Qb, Kb, Vb = (X.reshape(n // w, w, -1) for X in (Q, K, V))
    # Each query block attends to its own key block and the one before it
    prev = lambda X: torch.cat([torch.zeros_like(X[:1]), X[:-1]])
    Kb, Vb = torch.cat([prev(Kb), Kb], 1), torch.cat([prev(Vb), Vb], 1)
    i = torch.arange(w, device=Q.device)[:, None]      # Query in its block
    j = torch.arange(-w, w, device=Q.device)[None, :]  # Key offset from it
    first = torch.arange(n // w, device=Q.device)[:, None, None] == 0
    mask = (j <= i) & (i - j < w) & ~(first & (j < 0))  # No keys before 0
    scores = Qb @ Kb.transpose(-1, -2) / math.sqrt(d)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab jax
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d = Q.shape
    assert n % w == 0, 'blocks must tile the sequence exactly'
    Qb, Kb, Vb = (X.reshape(n // w, w, -1) for X in (Q, K, V))
    # Each query block attends to its own key block and the one before it
    prev = lambda X: jnp.concatenate([jnp.zeros_like(X[:1]), X[:-1]])
    Kb = jnp.concatenate([prev(Kb), Kb], 1)
    Vb = jnp.concatenate([prev(Vb), Vb], 1)
    i = jnp.arange(w)[:, None]                         # Query in its block
    j = jnp.arange(-w, w)[None, :]                     # Key offset from it
    first = jnp.arange(n // w)[:, None, None] == 0
    mask = (j <= i) & (i - j < w) & ~(first & (j < 0))  # No keys before 0
    scores = Qb @ Kb.swapaxes(-1, -2) / math.sqrt(d)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

A fused kernel cannot skip single scores. It computes in tiles of
$b \times b$ scores held in the small fast memory next to the arithmetic
units, and it can skip a tile only if the mask excludes all of it. Tiles
inside the mask are computed without masking, and only tiles that straddle
its edge apply the mask entry by entry. `windowed_attention` is this scheme
done by hand, with tiles of size $w$: each block of queries visits the tile
on the diagonal and the one to its left. The cost of a mask in a fused
kernel is therefore the fraction of tiles it touches. We count it at
$n = 2048$ for tiles of $b = 128$, a common size.

```{.python .input #attention-at-scale-a-linear-cost-implementation-3}
%%tab pytorch
def tiles_visited(mask, b):
    """Fraction of b-by-b tiles of a mask that hold at least one True."""
    T = mask.shape[0] // b
    return mask.reshape(T, b, T, b).any(dim=(1, 3)).float().mean().item()

n, b, w = 2048, 128, 128
print(f'causal:  {tiles_visited(d2l.causal_mask(n), b):.1%} of tiles')
print(f'window:  {tiles_visited(window_mask(n, w), b):.1%} of tiles')
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-3}
%%tab jax
def tiles_visited(mask, b):
    """Fraction of b-by-b tiles of a mask that hold at least one True."""
    T = mask.shape[0] // b
    return float(mask.reshape(T, b, T, b).any(axis=(1, 3)).mean())

n, b, w = 2048, 128, 128
print(f'causal:  {tiles_visited(d2l.causal_mask(n), b):.1%} of tiles')
print(f'window:  {tiles_visited(window_mask(n, w), b):.1%} of tiles')
```

A causal mask touches just over half of the tiles, and a window of 128
touches about one in eight: the diagonal tile and its left neighbor in each
row of tiles. A windowed fused kernel therefore costs time proportional to
$nw$, rounded up to whole tiles, and narrowing the window below the tile
size saves nothing more.

Fused kernels take the window as an argument. We check the fused kernel of
each framework, together with the blocked `windowed_attention`, against the
masked definition.

```{.python .input #attention-at-scale-a-linear-cost-implementation-2}
%%tab pytorch
n, w, d_h = 2048, 128, 64
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
reference = d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
                               window_mask(n, w, Q.device)) @ V
# FlexAttention takes the band as a rule on positions and evaluates it once
# per tile; torch.compile turns it into a kernel that visits only kept tiles
band = lambda batch, head, q, k: (k <= q) & (q - k < w)
block_mask = create_block_mask(band, None, None, n, n, device=Q.device)
fused = torch.compile(flex_attention)(Q[None, None], K[None, None],
                                      V[None, None], block_mask=block_mask)
for name, O in (('blocked', windowed_attention(Q, K, V, w)),
                ('fused kernel', fused[0, 0])):
    print(f'{name:>12}: maximum deviation '
          f'{float((O - reference).abs().max()):.2e}')
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-2}
%%tab jax
n, w, d_h = 2048, 128, 64
Q, K, V = (jax.random.normal(k, (n, d_h))
           for k in jax.random.split(d2l.get_key(), 3))
with jax.default_matmul_precision('highest'):
    reference = d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
                                   window_mask(n, w)) @ V
    # The window as an argument: w - 1 keys back, none ahead. The default
    # XLA implementation runs in float32; cuDNN's needs 16-bit inputs
    fused = jax.nn.dot_product_attention(
        Q[None, :, None], K[None, :, None], V[None, :, None],
        is_causal=True, local_window_size=(w - 1, 0))
    for name, O in (('blocked', windowed_attention(Q, K, V, w)),
                    ('fused kernel', fused[0, :, 0])):
        print(f'{name:>12}: maximum deviation '
              f'{float(jnp.abs(O - reference).max()):.2e}')
```

Both agree with the definition to within float32 rounding, below
$10^{-6}$.

:begin_tab:`pytorch`
FlexAttention :cite:`Dong.Feng.Guessous.ea.2025` takes the mask as a
Python function of the query and key positions. `create_block_mask`
evaluates it at every query--key pair, once, before the kernel runs, and
condenses the result into a `BlockMask` that records for each tile whether
the mask excludes all of it, allows all of it, or cuts through it: the tile
map counted above. Compiled with `torch.compile`, `flex_attention` becomes
a kernel in the style of FlashAttention that skips the excluded tiles and
applies the rule entry by entry only in the tiles the mask cuts through.
:end_tab:

:begin_tab:`jax`
`local_window_size=(w - 1, 0)` lets each query see $w - 1$ earlier keys and
none after it, which together with `is_causal=True` is the band of
`window_mask`. The check runs XLA's default implementation in float32; the
timing at the end of the section uses the cuDNN kernel, which needs 16-bit
inputs.
:end_tab:

## Linear Attention

### Kernelizing the Score

The second way keeps every key and changes the score. In
:numref:`sec_queries-keys-values`, attention pooling weighted each value in
proportion to a similarity kernel $K(\mathbf{q}, \mathbf{k})$ of query and
key, normalized over the keys. Softmax attention is the case
$K(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$,
with $d_h$ the width of a head. Nothing in the recipe requires the
exponential. Suppose instead that the kernel *factorizes* into features of
the query and features of the key,
$K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ for
some feature map $\phi$. What changes?

Let us start without the causal mask, where the answer takes one line.
Stack the features of the $n$ queries and of the $n$ keys as the rows of
$\phi(\mathbf{Q})$ and $\phi(\mathbf{K})$, and the values as the rows of
$\mathbf{V}$, an $n \times d_v$ matrix with $d_v$ the width of a value.
Leaving out the normalization for a moment, the outputs are

$$
\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\,\mathbf{V}
= \phi(\mathbf{Q})\,\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr).
$$

The two sides differ only in where the brackets go, and matrix products are
associative. The left side forms the $n \times n$ matrix of kernel values,
as softmax attention must. The right side first forms a $d_h \times d_v$
matrix, once, and multiplies the features of every query by it: about
$2 n d_h d_v$ multiply-adds instead of $n^2 (d_h + d_v)$. The normalizer
works the same way, with a column of ones in place of $\mathbf{V}$. Softmax
cannot do this, because the exponential sits between $\mathbf{q}$ and
$\mathbf{k}$ and keeps them from separating.

The causal mask lets query $t$ use only the keys $s \le t$, so one sum over
all keys becomes a running sum. We write the output for query $t$ as a row
vector $\mathbf{o}_t^\top$, one row of the output matrix. With a factorized
kernel, causal attention reads

$$
\mathbf{o}_t^\top
= \frac{\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)\,
        \mathbf{v}_s^\top}
       {\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}
= \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}
       {\phi(\mathbf{q}_t)^\top \mathbf{z}_t},
\qquad
\mathbf{S}_t = \sum_{s \leq t} \phi(\mathbf{k}_s) \mathbf{v}_s^\top,
\quad
\mathbf{z}_t = \sum_{s \leq t} \phi(\mathbf{k}_s).
$$
:eqlabel:`eq_linear-attn`

The second equality is the whole idea. The factor $\phi(\mathbf{q}_t)^\top$
is the same in every term, so it moves out of both sums, and what remains,
$\mathbf{S}_t$ and $\mathbf{z}_t$, does not depend on the query at all.
Instead of $t$ scores, each query needs one product of a vector with a
matrix that summarizes everything before it. This is *linear attention*,
named and analyzed by :citet:`Katharopoulos.Vyas.Pappas.ea.2020`.

Which feature map? It must keep every kernel value positive; otherwise the
weights are no longer those of an average, and the normalizer can vanish.
Following Katharopoulos et al., we apply $\phi(x) = \mathrm{elu}(x) + 1$
elementwise, where the exponential linear unit is $\mathrm{elu}(x) = x$ for
$x > 0$ and $e^x - 1$ otherwise. On the positive side $\phi(x) = x + 1$, on
the negative side $\phi(x) = e^x$: a ReLU shifted up by one, with the corner
rounded off, and positive everywhere. Positive is not the same as bounded
away from zero. For a component below about $-17$, $\mathrm{elu}(x) + 1$
rounds to zero in single precision, so the code adds $\epsilon = 10^{-6}$
to the normalizer, as the implementation of that paper does. This is
numerical hygiene. It changes no result in this section and can be
forgotten once the code runs.

The sums in :eqref:`eq_linear-attn` satisfy the recurrences

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\, \mathbf{v}_t^\top,
\qquad
\mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t),
$$
:eqlabel:`eq_linear-attn-recurrence`

which define a state of fixed size, $d_h \times d_v$ numbers for
$\mathbf{S}$ and $d_h$ for $\mathbf{z}$. Each arriving token writes its
value into the state as an outer product, and each query reads from it.
Linear attention is a recurrent network whose hidden state is a matrix. The
state acts as an associative memory, with $\phi(\mathbf{k}_t)$ the address
and $\mathbf{v}_t$ the content. The idea is older than the transformer:
:citet:`Schlag.Irie.Schmidhuber.2021` showed that this layer is a fast
weight programmer of the early 1990s, a network that rewrites the weights
of another network as it reads. The same computation has two
implementations. The *parallel* form computes all states at once with
cumulative sums, the natural form for training on whole sequences. The
*recurrent* form carries $(\mathbf{S}, \mathbf{z})$ forward one token at a
time in constant memory, as generation requires, like every RNN of
:numref:`chap_rnn`.

```{.python .input #attention-at-scale-kernelizing-the-score-1}
%%tab pytorch
def phi(X):
    return F.elu(X) + 1

def linear_attention_parallel(Q, K, V, eps=1e-6):
    """Training form: all outputs at once via cumulative sums."""
    Qp, Kp = phi(Q), phi(K)
    S = torch.cumsum(Kp[:, :, None] * V[:, None, :], dim=0)  # (n, d, d_v)
    z = torch.cumsum(Kp, dim=0)                              # (n, d)
    den = (Qp * z).sum(-1, keepdim=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d, d_v) state, updated token by token."""
    Qp, Kp = phi(Q), phi(K)
    S = torch.zeros(Q.shape[-1], V.shape[-1], device=Q.device)
    z = torch.zeros(Q.shape[-1], device=Q.device)
    outputs = []
    for t in range(Q.shape[0]):
        S = S + Kp[t][:, None] * V[t][None, :]   # Outer-product write
        z = z + Kp[t]
        outputs.append(Qp[t] @ S / (Qp[t] @ z + eps))  # Query the memory
    return torch.stack(outputs)
```

```{.python .input #attention-at-scale-kernelizing-the-score-1}
%%tab jax
def phi(X):
    return jax.nn.elu(X) + 1

def linear_attention_parallel(Q, K, V, eps=1e-6):
    """Training form: all outputs at once via cumulative sums."""
    Qp, Kp = phi(Q), phi(K)
    S = jnp.cumsum(Kp[:, :, None] * V[:, None, :], axis=0)  # (n, d, d_v)
    z = jnp.cumsum(Kp, axis=0)                              # (n, d)
    den = (Qp * z).sum(-1, keepdims=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d, d_v) state, updated token by token."""
    Qp, Kp = phi(Q), phi(K)
    def step(carry, qkv):
        S, z = carry
        q, k, v = qkv
        S = S + k[:, None] * v[None, :]          # Outer-product write
        z = z + k
        return (S, z), q @ S / (q @ z + eps)     # Query the memory
    init = (jnp.zeros((Q.shape[-1], V.shape[-1])), jnp.zeros(Q.shape[-1]))
    _, outputs = jax.lax.scan(step, init, (Qp, Kp, V))
    return outputs
```

### Chunked Computation

Neither form suits a GPU during training. The parallel form stores the state
at every position, $n$ matrices of $d_h \times d_v$ numbers: for one head
with $d_h = d_v = 64$ at $n = 16{,}384$, that is 256 MiB in fp32. At this
length the states take a quarter of the memory of the score matrix they
replace, but the states grow as $n d_h d_v$ and the score matrix as $n^2$,
so for $n$ below $d_h d_v = 4096$ the states are the larger of the two.
The recurrent form stores one state but takes $n$ sequential steps of
matrix--vector work, which leaves the matrix units of a GPU idle. Practical
kernels divide the sequence into chunks of $c$ positions
:cite:`Yang.Wang.Shen.ea.2024`, much as online softmax divided the keys
into blocks. Inside a chunk, the outputs come from the quadratic form, a
causally masked $c \times c$ matrix of kernel values. Across chunks, the
state carries the sum over all earlier chunks. For a chunk that starts at
position $t_0$, with $\mathbf{S}$ and $\mathbf{z}$ the sums over all
positions before $t_0$,

$$
\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\,
    \mathbf{v}_s^\top}
  {\phi(\mathbf{q}_t)^\top \mathbf{z}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)}.
$$
:eqlabel:`eq_linear-attn-chunked`

This equation is the one step of the section that may take a second
reading. It is the recurrent and the parallel form glued together at the
chunk boundaries. The first term of the numerator reads the memory of the
earlier chunks, as the recurrent form would, and the second attends within
the chunk, as the parallel form would. Both are matrix products. For each
chunk, a $c \times d_h$ by $d_h \times c$ product gives the kernel values,
a $c \times c$ by $c \times d_v$ product mixes the values, and the state
advances by a $d_h \times c$ by $c \times d_v$ product. The total work is
$O(nc(d_h + d_v) + n d_h d_v)$, linear in $n$; with $c = 64$, as in the
code, the two terms are of the same order. Only one state per chunk is
stored.

```{.python .input #attention-at-scale-chunked-computation-1}
%%tab pytorch
def linear_attention_chunked(Q, K, V, c=64, eps=1e-6):
    """Training form in chunks: quadratic within, recurrent across."""
    n, d = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d), phi(K).reshape(n // c, c, d)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = torch.cumsum(Kp.transpose(-1, -2) @ Vc, dim=0)   # (n/c, d, d_v)
    z = torch.cumsum(Kp.sum(1), dim=0)                   # (n/c, d)
    S = torch.cat([torch.zeros_like(S[:1]), S[:-1]])
    z = torch.cat([torch.zeros_like(z[:1]), z[:-1]])
    A = torch.tril(Qp @ Kp.transpose(-1, -2))            # (n/c, c, c)
    num = Qp @ S + A @ Vc                                # Past + this chunk
    den = Qp @ z[..., None] + A.sum(-1, keepdim=True) + eps
    return (num / den).reshape(n, -1)
```

```{.python .input #attention-at-scale-chunked-computation-1}
%%tab jax
def linear_attention_chunked(Q, K, V, c=64, eps=1e-6):
    """Training form in chunks: quadratic within, recurrent across."""
    n, d = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d), phi(K).reshape(n // c, c, d)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = jnp.cumsum(Kp.swapaxes(-1, -2) @ Vc, axis=0)     # (n/c, d, d_v)
    z = jnp.cumsum(Kp.sum(1), axis=0)                    # (n/c, d)
    S = jnp.concatenate([jnp.zeros_like(S[:1]), S[:-1]])
    z = jnp.concatenate([jnp.zeros_like(z[:1]), z[:-1]])
    A = jnp.tril(Qp @ Kp.swapaxes(-1, -2))               # (n/c, c, c)
    num = Qp @ S + A @ Vc                                # Past + this chunk
    den = Qp @ z[..., None] + A.sum(-1, keepdims=True) + eps
    return (num / den).reshape(n, -1)
```

The parallel, recurrent and chunked forms add the same terms in three
orders, so they should agree up to rounding. The check also prints the size
of the recurrent state for one head of width 64.

```{.python .input #attention-at-scale-kernelizing-the-score-2}
%%tab pytorch
n = 512
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
recurrent = linear_attention_recurrent(Q, K, V)
for name, f in (('parallel', linear_attention_parallel),
                ('chunked', linear_attention_chunked)):
    err = (f(Q, K, V) - recurrent).abs().max()
    print(f'{name:>8}: maximum deviation {float(err):.2e}')
print(f'recurrent state: {d_h}x{d_h} + {d_h} floats '
      f'= {(d_h * d_h + d_h) * 4 / 1024:.0f} KiB at any sequence length')
```

```{.python .input #attention-at-scale-kernelizing-the-score-2}
%%tab jax
n = 512
Q, K, V = (jax.random.normal(k, (n, d_h))
           for k in jax.random.split(d2l.get_key(), 3))
with jax.default_matmul_precision('highest'):
    recurrent = linear_attention_recurrent(Q, K, V)
    for name, f in (('parallel', linear_attention_parallel),
                    ('chunked', linear_attention_chunked)):
        err = jnp.abs(f(Q, K, V) - recurrent).max()
        print(f'{name:>8}: maximum deviation {float(err):.2e}')
print(f'recurrent state: {d_h}x{d_h} + {d_h} floats '
      f'= {(d_h * d_h + d_h) * 4 / 1024:.0f} KiB at any sequence length')
```

The three forms agree to within $4 \times 10^{-7}$; they differ only in what
they keep in memory and in how much of the work is matrix products. At
generation time this layer carries 16 KiB of state, whatever the context
length. A softmax head of the same width caches the key and the value of
every past token, $2 d_h \cdot 4 = 512$ bytes per token in fp32. Its cache
outgrows the linear-attention state at the 33rd token, reaches 4 MiB at
$n = 8192$, and keeps growing with every generated token.

### Capacity and Sharpness

What does linear attention give up for its constant memory? The state
$\mathbf{S}_t$ is a lossy summary. It holds $d_h \times d_v$ numbers however
many tokens it has absorbed, so as the context grows the writes interfere,
and retrieving one particular value becomes approximate. Positive features
make the interference unavoidable. Since $\phi(\mathbf{q})^\top
\phi(\mathbf{k}_s) > 0$ for every key, every read picks up a share of every
value ever written. A query can single out one key only if the features of
that key overlap little with those of all the others, and for features that
are never negative, little overlap means nearly disjoint supports. $d_h$
features leave room for at most about $d_h$ such keys, while a long context
holds thousands. Softmax attention keeps every key separately and has no
such limit.

The second loss is sharpness, and it appears even with eight keys and
plenty of room in the state. Softmax attention has a temperature.
Multiplying a query by a factor $\beta$ multiplies every score by $\beta$,
and the exponential turns a gap $\Delta$ between two scores into a ratio
$e^{\beta \Delta}$ of weights, which grows without bound: a large enough
query puts nearly all of its weight on its best key. Linear attention has
no such dial. For large $\beta$, the negative components of
$\beta\mathbf{q}$ give $e^{\beta x} \approx 0$ and the positive ones
$\beta x + 1 \approx \beta x$, so $\phi(\beta\mathbf{q})$ is close to
$\beta\,\mathrm{relu}(\mathbf{q})$, and the factor $\beta$ cancels between
numerator and normalizer. We should therefore see softmax weights approach
one-hot as the queries grow, and linear-attention weights that hardly move.
The test uses 1,000 random queries and 8 random keys of width 64, so that
uniform weights would be $1/8 = 0.125$.

```{.python .input #windowed-linear-attention-sharpening}
%%tab pytorch
Q, K = torch.randn(1000, d_h), torch.randn(8, d_h)  # 8 keys: uniform is 0.125
for beta in (1, 4, 16):  # Scale the queries up
    soft = torch.softmax(beta * Q @ K.T / math.sqrt(d_h), dim=-1)
    lin = phi(beta * Q) @ phi(K).T
    lin = lin / lin.sum(-1, keepdim=True)
    print(f'queries x{beta:<2}: mean largest weight, softmax '
          f'{soft.amax(-1).mean():.2f}, elu+1 {lin.amax(-1).mean():.2f}')
```

```{.python .input #windowed-linear-attention-sharpening}
%%tab jax
kq, kk = jax.random.split(d2l.get_key())
Q, K = jax.random.normal(kq, (1000, d_h)), jax.random.normal(kk, (8, d_h))
for beta in (1, 4, 16):  # Scale the queries up; 8 keys: uniform is 0.125
    soft = jax.nn.softmax(beta * Q @ K.T / math.sqrt(d_h), axis=-1)
    lin = phi(beta * Q) @ phi(K).T
    lin = lin / lin.sum(-1, keepdims=True)
    print(f'queries x{beta:<2}: mean largest weight, softmax '
          f'{soft.max(-1).mean():.2f}, elu+1 {lin.max(-1).mean():.2f}')
```

The largest softmax weight goes from about a third to about three quarters
to more than nine tenths. The largest $\mathrm{elu} + 1$ weight stays
between 0.14 and 0.16, barely above uniform. Multiplied by sixteen, a query
makes softmax nearly certain of its answer and leaves linear attention
about as undecided as before. (Exercise
:exref:`ex_attention-at-scale-feature-map` asks the same of other feature
maps.) Peaked weights are still possible through key features that barely
overlap, but only for the few keys that the capacity argument allows. What
linear attention lacks is the one dial that sharpens every comparison at
once.

## Choosing a Mechanism

### Windowed and Dense Kernels Compared

Before choosing, we should know how much time a window saves in practice.
The tile count predicts it before we run anything. With tiles of 128 and
$T = n/128$ rows of tiles, a causal kernel visits $T(T + 1)/2$ tiles. A
window of 128 visits $2T - 1$: the diagonal tile and its left neighbor in
each row, and only the diagonal tile in the first row. If time were
proportional to tiles, the window would be faster than dense causal
attention by the ratio $T(T + 1)/(2(2T - 1))$, which grows like $T/4$: 1.4
at 512 tokens, 64 at 32,768.

For the ratio to measure tiles and nothing else, both arms must run in the
same kernel. Each framework below times dense causal attention and the
window $w = 128$ in one fused kernel, on one sequence with 32 heads of
width 64, in bfloat16, the precision in which attention is usually trained.
The cell prints the measured speedup next to the tile-count prediction and
plots both times on log--log axes.

```{.python .input #attention-at-scale-the-price-of-attention-measured}
%%tab pytorch
flex = torch.compile(flex_attention)
w = b = 128  # Window width; tile size of the kernel
masks = {'dense causal': lambda batch, head, q, k: k <= q,
         f'window (w={w})': lambda batch, head, q, k: (k <= q) & (q - k < w)}
H, lengths = 32, [512, 1024, 2048, 4096, 8192, 16384, 32768]
times = {name: [] for name in masks}
for n in lengths:
    X = torch.randn(1, H, n, d_h, device=d2l.try_gpu(), dtype=torch.bfloat16)
    for name, rule in masks.items():
        block_mask = create_block_mask(rule, None, None, n, n, device=X.device)
        times[name].append(d2l.Benchmark(
            lambda: flex(X, X, X, block_mask=block_mask)).time * 1e3)
    T = n // b  # Rows of b x b tiles; causal T(T+1)/2, window 2T - 1
    dense, window = (t[-1] for t in times.values())
    print(f'n = {n:6,}: window {dense / window:4.1f}x faster, '
          f'tile count predicts {T * (T + 1) / 2 / (2 * T - 1):4.1f}x')
d2l.plot(lengths, list(times.values()), 'sequence length n', 'time (ms)',
         legend=list(times), xscale='log', yscale='log', figsize=(4.5, 3))
```

```{.python .input #attention-at-scale-the-price-of-attention-measured}
%%tab jax
# cuDNN takes (batch, length, heads, width); it skips the tiles outside the
# band only together with is_causal=True
w = b = 128  # Window width; tile size of the kernel
masks = {'dense causal': {},
         f'window (w={w})': {'local_window_size': (w - 1, 0)}}
kernels = {name: jax.jit(lambda X, kw=kw: jax.nn.dot_product_attention(
    X, X, X, is_causal=True, implementation='cudnn', **kw))
    for name, kw in masks.items()}
H, lengths = 32, [512, 1024, 2048, 4096, 8192, 16384, 32768]
times = {name: [] for name in masks}
for n in lengths:
    X = jax.random.normal(d2l.get_key(), (1, n, H, d_h), dtype=jnp.bfloat16)
    for name, f in kernels.items():
        times[name].append(d2l.Benchmark(lambda: f(X)).time * 1e3)
    T = n // b  # Rows of b x b tiles; causal T(T+1)/2, window 2T - 1
    dense, window = (t[-1] for t in times.values())
    print(f'n = {n:6,}: window {dense / window:4.1f}x faster, '
          f'tile count predicts {T * (T + 1) / 2 / (2 * T - 1):4.1f}x')
d2l.plot(lengths, list(times.values()), 'sequence length n', 'time (ms)',
         legend=list(times), xscale='log', yscale='log', figsize=(4.5, 3))
```

The prediction holds in shape and falls short in size. In our measurement,
up to about a thousand tokens the two kernels take the same time: a fixed
cost per call dominates, and there are few tiles to skip. (With more
sequences or heads per call, that fixed cost would be shared, and the
window would start to save time at shorter lengths.) From a few thousand
tokens on, the time of dense attention grows three to four times with
every doubling of $n$, approaching the fourfold growth of its tile count, while
the time of the window about doubles. At 32,768 tokens the window is more
than thirty times faster, though short of 64.

Most of the shortfall has a simple cause. A tile that lies entirely inside
the mask is computed without masking; a tile that straddles the edge of the
mask must apply it entry by entry, which costs more. Of the causal kernel's
tiles, only the $T$ on the diagonal straddle the edge, and all the others
lie inside the mask. Every tile the band visits straddles an edge. A
window's tiles are all of the expensive kind.

The figure measures cost alone. The window computes a different function
from dense attention, and what it loses (the hops of
:numref:`subsec_attn-receptive-field`) does not show up in milliseconds.

### The Default and Its Alternatives

Which of these should a model use? Exact attention in a fused kernel is the
default. It computes the right function, and on short sequences a window
saves little time. Windows are the remedy that most released long-context
models adopt, and they adopt it in combination: windowed layers in most of
the stack save arithmetic in proportion to the tiles they skip and cache in
proportion to $w/n$, while a few global layers keep a one-hop path between
any two positions.

Plain linear attention is not used as derived here. At equal size it models
language less well, and it is weakest on tasks that must recall one
particular earlier token
:cite:`Arora.Eyuboglu.Zhang.ea.2024,Yang.Wang.Shen.ea.2024`, just as its
lossy state and blunt weights suggest. Two changes brought it close to
softmax attention on language-modeling benchmarks. The state learns to
forget: a decay computed from each input multiplies $\mathbf{S}_{t-1}$
before the new write, so that old content fades when new content needs the
room (Exercise :exref:`ex_attention-at-scale-decaying-state` works out a
constant decay). And models keep a few softmax layers for recall;
MiniMax-01 places one after every seven linear-attention layers
:cite:`MiniMax.2025`. Whether a fixed-size state can be taught to forget
and to write selectively enough to compete with a cache that keeps
everything is the question of :numref:`chap_modern_rnn`, where gated
descendants of this recurrence become linear state space models.

## Summary

To go below the $n^2$ cost of exact attention, a model must either score
fewer pairs or change the score. A sliding window keeps each query's $w$
most recent keys; a stack of $L$ windows reaches $1 + L(w - 1)$ positions
but relays distant facts through many hops, so released models interleave
a few global layers. A fused kernel saves work only in whole tiles, and the
tile count predicts the measured speedup in shape, though not fully in
size, because every tile of a band is masked entry by entry. A kernel that
factorizes, $\phi(\mathbf{q})^\top\phi(\mathbf{k})$, turns attention into
a recurrent network with a $d_h \times d_v$ matrix state, computed in
chunks for training and step by step for generation. That state has
constant size but is a lossy summary, and its weights do not sharpen when
the query grows, which is why exact attention in a fused kernel remains the
default.

## Exercises

1. [code] **Tiles of a fused kernel.** A fused kernel works on tiles of
   $b \times b$ scores and skips a tile only if the mask excludes all of it.
   Let $T = n/b$.
    1. Show that a causal mask leaves $T(T+1)/2$ of the $T^2$ tiles, and
       that a window of width $w = kb$, for a whole number $k < T$, leaves
       $T(k + 1) - k(k + 1)/2$. Check both counts with `tiles_visited`.
       How does the count change when $w$ is not a multiple of $b$?
    1. Several documents are packed into one sequence of $n = 2048$ tokens,
       and the mask combines causality with "same document" as in the
       masks of :numref:`sec_attention-scoring-functions`. Using
       `tiles_visited` with $b = 128$, compare four documents of 512 tokens
       with documents of 1,000, 600, 300, and 148 tokens. Explain why
       documents whose boundaries do not fall on tile boundaries cost more
       tiles.

1. [code] **The feature map.** The feature map $\phi(x) =
   \mathrm{elu}(x) + 1$ in `phi` is one choice among many. Compare it with
   $\phi(x) = \mathrm{relu}(x)$ and $\phi(x) = \exp(x)$, both applied
   elementwise, on $n = 512$ random queries and keys of width `d_h`.
    1. For each map, compute the causal linear-attention weight matrix
       with entries
       $$\frac{\phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}{\sum_{s' \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{s'})}, \qquad s \leq t,$$
       and measure its maximum absolute difference from the causal softmax
       weight matrix `d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
       d2l.causal_mask(n))`, and the mean entropy of its rows.
    1. Under which condition on $\mathbf{q}_t$ and the keys
       $\mathbf{k}_1, \ldots, \mathbf{k}_t$ does the ReLU map produce an
       undefined row? Derive the probability of this event for row $t$
       when all entries are independent standard normal, and evaluate it
       for $t = 1$ at width $64$ and at width $4$.
    1. The section scaled the queries by $\beta \in \{1, 4, 16\}$ and
       watched the largest weight. Do the same for the ReLU and exponential
       maps, measuring the mean row entropy instead. Derive, for each of
       the three maps, the limit of the normalized weights as
       $\beta \to \infty$, and explain why none of them becomes one-hot
       where softmax does.
   :exlabel:`ex_attention-at-scale-feature-map`

1. **An exact feature map.** ● The softmax kernel $K(\mathbf{q},
   \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d_h})$ also
   factorizes, but only with infinitely many features.
    1. Expand $K$ in its Taylor series, and show that the degree-$p$
       term equals $\phi_p(\mathbf{q})^\top \phi_p(\mathbf{k})$ for a
       feature map $\phi_p$ of dimension $d_h^p$ built from the $p$-fold
       tensor product of its argument.
    1. Truncate the series after degree $2$. Show that the truncated kernel
       is positive for every pair of $\mathbf{q}$ and $\mathbf{k}$, so the
       normalizer of :eqref:`eq_linear-attn` cannot vanish, and give the
       dimension of the feature map that concatenates $\phi_0$, $\phi_1$,
       and $\phi_2$. For $d_h = 64$ and values of width $64$, compute the
       size of the state $(\mathbf{S}, \mathbf{z})$ in single precision and
       the length $n$ at which the key--value cache of a softmax head
       reaches the same size. A truncation of this kind is the feature map
       of Based :cite:`Arora.Eyuboglu.Zhang.ea.2024`.
    1. Take as given that the $n \times n$ matrix with entries
       $K(\mathbf{x}_i, \mathbf{x}_j)$ is positive definite for
       distinct $\mathbf{x}_1, \ldots, \mathbf{x}_n$. Show that no feature
       map of dimension $D < n$ reproduces $K$ exactly on these points
       when queries and keys coincide.

1. [extended] **Attention variants in a language model.** ● Whether the
   quality gap between exact and linear attention matters depends on the
   task. Measure it on `TinyCharLM` of
   :numref:`sec_positional-information`, with `pos='learned'` for every
   variant so that position enters the same way.
    1. Subclass `TinyCharLM` and override `attend` twice: once with a
       causal band mask of width $w$ in place of the causal mask, and once
       with the causal linear-attention weights of :eqref:`eq_linear-attn`
       for $\phi(x) = \mathrm{elu}(x) + 1$, computed explicitly as a full
       weight matrix over the context so that the method still returns its
       weights. Add $\epsilon = 10^{-6}$ to the normalizer, as in
       `linear_attention_parallel`; without it, a row whose features
       underflow divides zero by zero and training ends in NaN.
    1. Train dense attention, windows of width $w \in \{4, 16, 64\}$, and
       linear attention with the protocol of
       :numref:`subsec_pos-experiment`, and report validation perplexity at
       context $128$. Relate the window results to the reach $1 + 2(w-1)$
       of the two-block model.
    1. Build validation sequences of $128$ bytes, each a random
       $64$-byte excerpt of the validation text followed by the same
       excerpt again, and compare the mean loss on the second half with
       that on the first half for every variant. At the budget of
       :numref:`subsec_pos-experiment`, expect even dense attention to
       gain little from the copy. Train the dense model and the window
       $w = 16$ longer, or on data that repeats, until the dense model's
       second-half loss falls clearly below its first-half loss, and
       report the budget at which that happens. Which variants then
       exploit the repetition, and how does the answer relate to the reach
       of the windowed models and to the fixed-size state of linear
       attention?

1. [code] **A decaying state.** ● Give the state of
   :eqref:`eq_linear-attn-recurrence` a scalar decay $\gamma \in (0, 1)$,
   $$\mathbf{S}_t = \gamma\, \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \gamma\, \mathbf{z}_{t-1} + \phi(\mathbf{k}_t).$$
    1. Show that the parallel form now weights the contribution of position
       $s$ to the output at $t$ by $\gamma^{t-s}$, and implement it by
       rescaling the cumulative sums of `linear_attention_parallel` with
       powers of $\gamma$.
    1. Add the decay to `linear_attention_recurrent`, and verify that the
       two forms agree for $n = 512$ and $\gamma = 0.99$.
    1. The rescaling multiplies position $s$ by $\gamma^{-s}$ before the
       cumulative sum. For which $n$ does this overflow single precision at
       $\gamma = 0.9$ and at $\gamma = 0.5$?
    1. Add the decay to `linear_attention_chunked`: rescale each position
       relative to the start of its chunk, and carry the state from one
       chunk to the next with the decayed recurrence. Show that no factor
       larger than $\gamma^{-c}$ appears, run the result for $\gamma = 0.5$,
       $n = 512$, and $c = 64$, and compare it with the recurrent form.
    1. Compute the lag at which the weight $\gamma^{t-s}$ falls below
       $0.01$ for $\gamma = 0.99$. A task requires recalling one token
       from $2000$ positions back, while the tokens in between are filler.
       Explain why no fixed $\gamma$ serves both that token and the
       filler, and what a decay computed from each token's own input could
       do instead.
   :exlabel:`ex_attention-at-scale-decaying-state`

[Discussions](https://d2l.discourse.group/t/36882)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.6]{.kicker}

Windowed and Linear Attention<br>
**sliding windows · tiles · linear attention · measured cost**
:::
:::

::: {.slide title="Exact Fused Attention Still Visits 32,896 Tiles at 32K Tokens"}
[The problem]{.kicker}

A fused kernel no longer stores the $n \times n$ score matrix, but it still
computes every score, in tiles of $128 \times 128$.

| $n = 32{,}768$ tokens, $T = 256$ rows of tiles | tiles visited |
|:--|--:|
| dense causal: $T(T+1)/2$ | 32,896 |
| window of the last 128 keys: $2T - 1$ | 511 |

::: {.d2l-note}
The score matrix is gone; the arithmetic and the cache remain.
:::
:::

::: {.slide title="Goals: Two Routes Below n², and What Each Gives Up"}
[Goals]{.kicker}

- compute the **reach** of stacked windows and the **cache** they save;
- **count tiles** and predict a fused kernel's speedup;
- derive **linear attention** as a recurrence;
- show why scaling the query **does not sharpen** it;
- **choose** a mechanism.

Running example: one head of width $d_h = 64$, as in GPT-2 small.
:::

::: {.slide title="This Deck Covers Cheaper Attention, Not Learned Sparsity"}
[Scope]{.kicker}

| Chapter 10 | |
|:--|:--|
| queries, keys, values · scoring and masks · self-attention and heads · positions · cost of attention | done |
| **windowed and linear attention** | **here** |
| attention circuits | next |

Skipped: sparsity chosen from content, random-feature approximations of
softmax, gated recurrences (the state space chapter).
:::

::: {.slide title="Two Routes Below n²: Score Fewer Pairs, or Factorize the Score"}
[Idea]{.kicker}

| route | change | cost | gives up |
|:--|:--|:--|:--|
| sliding window | each query sees its $w$ latest keys | $O(nw)$ | direct long-range paths |
| linear attention | $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}) \to \phi(\mathbf{q})^\top\phi(\mathbf{k})$ | $O(n)$, fixed state | sharp recall |

$n$: sequence length; $\phi$: a feature map.
:::

::: {.slide title="A Sliding Window Turns the Causal Triangle into a Band"}
[Windows]{.kicker}

Query $i$ may read key $j$ if $j \le i$ and $i - j < w$:

@attention-at-scale-attention-through-a-window

12 positions: causal keeps 78 of 144 pairs, a window of $w = 4$ keeps 42.
Large $n$: about $nw$ pairs instead of $n^2/2$.
:::

::: {.slide title="Depth Restores the Reach: 1 + L(w − 1) Positions"}
[Windows]{.kicker}

Each layer moves information up to $w - 1$ positions back.

| layers $L$ | 1 | 2 | 4 | 32 |
|:--|--:|--:|--:|--:|
| reach $1 + L(w-1)$, $w = 4$ | 4 | 7 | 13 | 97 |
| reach, $w = 4096$ (Mistral 7B) | 4,096 | 8,191 | 16,381 | 131,041 |

[Jiang et al. 2023]{.small}
:::

::: {.slide title="Reach Is Not Use: Averaging Windows Pass On 1/256"}
[Windows]{.kicker}

- Full attention: any two positions in **one** layer.
- Windows: about $n/w$ hops; 100,000 tokens at $w = 4096$ need at least
  25 of Mistral's 32 layers.
- Equal-weight windows, $w = 4$, 4 layers: the token 12 back gets
  $(1/4)^4 = 1/256$; one global layer gives it $1/13$.

::: {.d2l-note}
A distant fact arrives only if every hop puts most of its weight on it.
:::
:::

::: {.slide title="Global Layers Keep One-Hop Paths; Windows Shrink the Cache"}
[Windows]{.kicker}

$r$ windowed layers per global layer; KV cache relative to all-global:
$$\frac{r\min(n, w) + n}{(r+1)\,n} \;\to\; \frac{1}{r+1}$$

| Gemma 3: $r = 5$, $w = 1024$ | 8K | 32K | 128K |
|:--|--:|--:|--:|
| cache vs. all-global | 0.27 | 0.19 | 0.17 |

[Gemma Team 2025]{.small}
:::

::: {.slide title="Check: Reach and Cache of a 24-Layer Model"}
[Check]{.kicker}

24 layers, windows of $w = 512$. How far back does the last layer reach?
With one global layer after every three windowed ones, what fraction of
the cache remains at $n = 4096$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1 + 24 \cdot 511 = 12{,}265$ positions.
$(3 \cdot 512 + 4096)/(4 \cdot 4096) \approx 0.34$, tending to $1/4$.
:::
:::
:::

::: {.slide title="A Fused Kernel Skips Work Only in Whole Tiles"}
[Tiles]{.kicker}

Tiles of $b \times b = 128 \times 128$ scores at $n = 2048$:

@!attention-at-scale-a-linear-cost-implementation-3

- A tile is skipped only if the mask excludes **all** of it.
- Window: the diagonal tile and its left neighbor per row.

::: {.d2l-note}
Time grows as $nw$, rounded up to whole tiles.
:::
:::

::: {.slide title="Check: Tiles for a Narrower and a Wider Window"}
[Check]{.kicker}

$b = 128$, $n = 2048$, so 16 rows of tiles. A window of $w = 128$ touches 31
of the 256 tiles. How many does $w = 32$ touch? And $w = 512$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $w = 32$: still 31; narrowing below a tile saves nothing.
$w = 512$: up to 5 tiles per row, $16 \cdot 5 - 10 = 70$.
:::
:::
:::

::: {.slide title="Fused Kernels Take the Window as an Argument and Match the Mask"}
[Tiles]{.kicker}

Blocked `windowed_attention` and a fused window kernel, against the masked
softmax definition:

@!attention-at-scale-a-linear-cost-implementation-2

PyTorch: FlexAttention takes the band as a rule on positions; its block
mask records which tiles to skip. JAX: `local_window_size=(w - 1, 0)` with
`is_causal=True`.
:::

::: {.slide title="Moving the Brackets Removes the n × n Matrix"}
[Linear attention]{.kicker}

Replace $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ by
$\phi(\mathbf{q})^\top\phi(\mathbf{k})$. Without a mask, up to the
normalization:

$$\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\mathbf{V}
= \phi(\mathbf{Q})\bigl(\phi(\mathbf{K})^\top\mathbf{V}\bigr)$$

Left: an $n \times n$ matrix. Right: a $d_h \times d_v$ matrix, formed once
($d_v$: width of a value).

::: {.d2l-note}
Same product, other brackets: $n^2$ becomes $n$.
:::

[Katharopoulos et al. 2020]{.small}
:::

::: {.slide title="Causal Linear Attention Is an RNN with a Matrix State"}
[Linear attention]{.kicker}

$\phi = \mathrm{elu} + 1 > 0$, with $\mathrm{elu}(x) = x$ for $x > 0$ and
$e^x - 1$ otherwise. The query factors out of the running sums:

$$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t}$$
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)$$

::: {.d2l-note}
State $\mathbf{S}$: a $d_h \times d_v$ matrix, written by outer products,
read with the query.
:::
:::

::: {.slide title="Three Forms, One Function: Parallel, Recurrent, Chunked"}
[Linear attention]{.kicker}

| form | stores | work | used for |
|:--|:--|:--|:--|
| parallel (cumulative sums) | $n$ states | elementwise products, a cumulative sum | short training sequences |
| recurrent | one state | $n$ sequential steps | generation |
| chunked, chunks of $c$ | one state per chunk | matrix products | training |

Chunked: a causally masked $c \times c$ product within a chunk, the carried
state $\mathbf{S}$ across chunks. Parallel at $n = 16{,}384$, one head:
256 MiB of states.
:::

::: {.slide title="The State Stays 16 KiB While a KV Cache Grows"}
[Linear attention]{.kicker}

@!attention-at-scale-kernelizing-the-score-2

- The three forms agree to within $4 \times 10^{-7}$.
- A softmax head of width 64 caches 512 bytes per token: larger than the
  state from the 33rd token, 4 MiB at 8,192.

::: {.d2l-note}
The price of constant size: features are positive, so every read mixes in
every write.
:::
:::

::: {.slide title="Scaling the Query Sharpens Softmax but Not elu + 1"}
[Linear attention]{.kicker}

1,000 random queries, 8 random keys (uniform weight 0.125), queries
multiplied by $\beta$:

@!windowed-linear-attention-sharpening

- Softmax: weight ratios $e^{\beta\Delta}$ grow without bound.
- $\phi(\beta\mathbf{q}) \approx \beta\,\mathrm{relu}(\mathbf{q})$: $\beta$
  cancels in the normalization.
:::

::: {.slide title="Check: The ReLU Feature Map"}
[Check]{.kicker}

Take $\phi(x) = \mathrm{relu}(x)$. What happens to the linear-attention
weights of a query when it is multiplied by 16?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Nothing. $\phi(16\mathbf{q}) = 16\,\phi(\mathbf{q})$, and the
factor 16 cancels exactly between numerator and normalizer.
:::
:::
:::

::: {.slide title="The Window Is More Than 30× Faster at 32K Tokens, Short of the Tiles' 64"}
[Measured]{.kicker}

@!attention-at-scale-the-price-of-attention-measured

Both arms in one fused kernel; one sequence, 32 heads of width 64, bfloat16.
:::

::: {.slide title="Every Tile the Band Visits Is a Masked Tile"}
[Measured]{.kicker}

| | tiles visited | tiles on the mask's edge |
|:--|--:|:--|
| dense causal | $T(T+1)/2$ | only the $T$ diagonal tiles |
| window, $w = b$ | $2T - 1$ | all of them |

- Short sequences: a fixed cost per call, no saving.
- Long sequences: dense 3–4× per doubling of $n$, window about 2×.

::: {.d2l-note}
Tile counts predict the shape; masked tiles explain most of the shortfall.
:::
:::

::: {.slide title="Default to Exact Fused Attention; Interleave Windows for Long Contexts"}
[Choice]{.kicker}

| mechanism | cost | cache | used |
|:--|:--|:--|:--|
| exact, fused kernel | $n^2$ time | all keys | the default |
| windows + few global layers | mostly $nw$ | mostly $w$ keys | long-context models |
| plain linear attention | $n$ | one matrix | not as is |

- Linear attention improved by a decay that depends on the input, plus a
  few softmax layers (MiniMax-01: one in eight).

::: {.d2l-note}
Scaling the query does not sharpen linear attention, and it recalls poorly.
:::

[Arora et al. 2024; Yang et al. 2024; MiniMax 2025]{.small}
:::

::: {.slide title="Check: A Window Narrower than a Tile"}
[Check]{.kicker}

At $n = 32{,}768$ with tiles of 128, a window of $w = 64$ replaces
$w = 128$. Does the kernel get faster?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. Each row of tiles still visits its diagonal tile and its
left neighbor, 511 tiles in all, and every one straddles the mask's edge.
:::
:::
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Window: $O(nw)$ work, reach $1 + L(w - 1)$; Gemma 3 keeps about a
  quarter of the cache at 8K.
- Tiles at 2,048 tokens: causal about half, window about one in eight.
- Linear attention: 16 KiB of state per head; a larger query does not
  sharpen its weights.
- At 32K tokens the window is more than 30× faster, short of the tiles' 64.

Next: what a trained head computes.
:::
