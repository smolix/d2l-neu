# Windowed and Linear Attention
:label:`sec_windowed-linear-attention`

<!-- WRITER: opening from the reader (worklist 6.B1). Section :numref:`sec_attention-at-scale` ended with exact attention that no longer stores the score matrix but still computes all $n^2$ scores; to go below that, a model must score fewer pairs or change the score so that the sum over keys can be done before the query arrives. What does each give up? The cells below need a CUDA GPU. -->

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

## Windowed and Sparse Attention

### Sliding-Window Attention

FlashAttention removes the $n^2$ *memory*, but every query still scores
every key, so the arithmetic remains quadratic. To reduce it too, we must
decide that some query--key pairs are not worth scoring. The simplest
useful rule is locality: each query attends only to the $w$ most recent
positions. With the masks of :numref:`sec_attention-scoring-functions`, the
causal mask becomes a band: causal, and at most $w - 1$ positions back.

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

<!-- WRITER: the random band heatmap cell (attention-at-scale-sliding-window-attention-1) is cut (worklist 6.A2); the mask figure of :numref:`sec_attention-scoring-functions` shows the band. The lead sentence "With a window of 4, each row of random scores keeps its weight on the query's own position and the three before it:" went with it. -->

### Receptive Field Growth with Depth
:label:`subsec_attn-receptive-field`

A window of width $w$ excludes direct interactions beyond $w$ tokens, but
depth increases the receptive field as in CNNs (:numref:`chap_cnn`). A query
in layer 2 attends to keys that are themselves layer-1 outputs, each of
which already summarizes its own window. Information therefore hops up to
$w-1$ positions per layer, and after $L$ layers a token's receptive field
spans $1 + L(w-1)$ positions. We can compute this directly. The band mask,
read as an adjacency matrix, composes across layers by boolean matrix
multiplication, and its $L$-th power says which positions can influence
which through $L$ layers.

<!-- WRITER: the receptive-field cell (attention-at-scale-depth-restores-the-reach) is cut (COST-12; it printed 8, 15, 29 = the formula). The two sentences before this marker ("We can compute this directly. The band mask, read as an adjacency matrix, ...") and "The count matches the formula, and the heatmaps show the band widening layer by layer." below refer to it; per worklist 6.B2 its space goes to the caveat that reach is not use. -->

The count matches the formula, and the heatmaps show the band widening layer
by layer. Mistral 7B uses this growth: it attends through a window of 4096 in
each of its 32 layers, for a reach beyond a hundred thousand tokens
:cite:`Jiang.Sablayrolles.Mensch.ea.2023`. :numref:`tab_cnn-rnn-attn`
describes what is given up. A stack of windows has path length
$O(n/w)$ instead of $O(1)$, so with $w$ in the thousands
it assumes that few dependencies need more than a small number of hops.

More recent models therefore interleave windowed layers with a few layers of
full attention, called *global* layers. Gemma 3 follows every five windowed
layers of width 1024 with one global layer :cite:`Gemma.Team.2025`, and
gpt-oss alternates global layers with windows of 128 tokens
:cite:`OpenAI.2025`. The global layers restore the $O(1)$ path
length. The windowed layers keep both the arithmetic and the memory of
generation small: a model that generates text caches the keys and values of
past tokens (:numref:`sec_multihead-attention`), and a windowed layer needs
to keep only the last $w$ of them. With $r$ windowed layers per global
layer, the cache shrinks to

$$
\frac{r \min(n, w) + n}{(r + 1)\, n}
$$

of the cache of a stack of global layers only: each of the $r$ windowed
layers caches $\min(n, w)$ positions and the global layer all $n$, against
$(r+1)n$ positions for global layers throughout. For Gemma 3's $r = 5$ and
$w = 1024$, this is about a quarter at $n = 8192$, and it approaches $1/6$
as the context grows.

Llama 4 also drops the positional encoding from its global layers, which
then work like the NoPE of :numref:`sec_positional-information`. It keeps
the rotary encoding only in its local layers, which attend within chunks of
the sequence and therefore see only short offsets :cite:`Meta.2025`. The aim
is a model that runs at contexts longer than those it was trained on.

### A Linear-Cost Implementation

The mask defines the result, but by itself it saves no work. A masked
softmax over a band still scores all $n^2$ pairs and then discards most of
them. The efficient implementation processes the sequence
in blocks of $w$ queries. A query can attend only to keys in its own block
and in the block before it, so each block scores a $w \times 2w$ tile. The
total work is $2nw$ scores instead of $n^2$, which is linear in $n$ for fixed
$w$.

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
$b \times b$ scores held in on-chip memory, and it can skip a tile only if
the mask excludes all of it. Tiles inside the mask are computed without
masking, and only tiles that straddle its edge apply the mask entry by
entry. The cost of a mask in a fused kernel is therefore the fraction of
tiles it touches. We count it for tiles of $b = 128$, the usual size.

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
size saves nothing more. Fused kernels take the window as part of their
interface, and we check that their output equals that of
`windowed_attention`:

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

<!-- WRITER: the two equivalence checks are now one cell (id -2, after the tile count; -4 deleted): it prints two deviations, blocked and fused kernel, both against the mask definition (one sentence, structural 3). "The blocked version should agree with the mask-based definition." above the old -2 is removed; the sentence "Fused kernels take the window as part of their interface, and we check that their output equals that of `windowed_attention`:" now leads this cell. The PyTorch print of BlockMask sparsity is gone (it equals 100% minus the window's 12.1% from tiles_visited). The FlexAttention compile requirement is now a code comment (COST-11); the two tab notes below go to at most two sentences each (6.B3). The JAX fused-kernel exactness check runs XLA's implementation (float32); `implementation='cudnn'` raises "Q must be fp16/bf16/fp8..." in float32, so cuDNN appears only in the timing cell. -->

:begin_tab:`pytorch`
FlexAttention :cite:`Dong.Feng.Guessous.ea.2025` makes this tile map the
interface to a fused kernel. The user writes the mask as a function of the
query and key positions, here the band, and `create_block_mask` evaluates it
once per tile. The resulting `BlockMask` is the tile map counted above, and
its sparsity is the percentage of tiles skipped. `flex_attention` with this
block mask equals `windowed_attention`. Under `torch.compile`, as here, it
generates a kernel in the style of FlashAttention that visits only the kept
tiles; called without compilation, it would compute the full score
matrix. Every mask built from the pieces of
:numref:`sec_attention-scoring-functions` (windows, packed documents,
prefixes) then runs at the speed of a fused kernel.
:end_tab:

:begin_tab:`jax`
`jax.nn.dot_product_attention` takes the window as an argument:
`local_window_size=(w - 1, 0)` lets each query see $w - 1$ earlier keys and
none after it, and with `is_causal=True` this is the band of `window_mask`.
The output equals that of `windowed_attention`. XLA's default
implementation applies the window as a mask over the full score matrix;
`implementation='cudnn'` passes it to the fused kernel as a parameter
instead.
:end_tab:

Fixed windows decide by position what may be attended, before the content is
seen. Natively sparse attention instead learns which blocks of keys each
query visits, at the same block granularity, and runs as a fused kernel
that visits only the chosen blocks :cite:`Yuan.Gao.Dai.ea.2025`.

## Linear Attention in Recurrent Form

### Kernelizing the Score

The third approach changes the softmax itself. Attention pooling with a
fixed kernel (:numref:`sec_queries-keys-values`) weights each value in
proportion to a similarity kernel, and the exponentiated dot product
$\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d})$ is one kernel among many. Suppose
we choose instead a kernel that *factorizes*,
$k(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ for
some elementwise feature map $\phi$. Causal attention with this kernel is

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

Because the query enters linearly, it factors out of the sum. The
remaining terms, $\mathbf{S}_t$ and $\mathbf{z}_t$, do not depend on the
query. Reordering the sums therefore removes the explicit pairwise
interaction and its quadratic cost. Instead of $t$ scores, each query needs
one matrix-vector product against a running summary
:cite:`Katharopoulos.Vyas.Pappas.ea.2020`. Following that paper, we take
$\phi(x) = \mathrm{elu}(x) + 1$, which keeps features positive so that the
normalizer is positive. Positive is not the same as large: when training
drives a query or key component far below zero, $\mathrm{elu}(x) + 1 =
e^x$ underflows. The code therefore adds $\epsilon = 10^{-6}$ to the
normalizer, as the implementation of that paper does.

The sums in :eqref:`eq_linear-attn` satisfy the recurrences

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\, \mathbf{v}_t^\top,
\qquad
\mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t),
$$
:eqlabel:`eq_linear-attn-recurrence`

which define a fixed-size state of $d \times d_v$ numbers. The state is
updated by an outer-product write as each token arrives, then read by the
query. Linear attention is a recurrent network whose hidden state is a
matrix. The state acts as an associative memory, with $\phi(\mathbf{k}_t)$
the address and $\mathbf{v}_t$ the content.
:citet:`Schlag.Irie.Schmidhuber.2021` showed that this layer is the fast
weight programmer of the early 1990s. The same computation has two
implementations. The *parallel* form for training computes all states at
once with cumulative sums. The *recurrent* form for generation carries
$(\mathbf{S}, \mathbf{z})$ forward one token at a time, in constant memory,
like every RNN in :numref:`chap_rnn`.

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

<!-- WRITER: the check of the parallel and recurrent forms (kernelizing-the-score-2) moved below the chunked form, merged with the chunked check (worklist 6.A4); the lead "Equations ... should agree up to floating-point rounding." went with it. The paragraphs on the 16 KiB state and "The state $\mathbf{S}_t$ is a lossy summary." moved below that check, followed by the new sharpening cell. -->

### Chunked Computation

Neither form suits a GPU during training. The parallel form stores the state
at every position, $n$ matrices of $d \times d_v$ numbers: 256 MiB for one
head at $n = 16{,}384$ in fp32, more than the score matrix it replaces until
$n$ exceeds $d\, d_v = 4096$. The recurrent form stores one state but takes
$n$ sequential steps of matrix--vector work, which leaves the matrix units
of a GPU idle. Practical kernels divide the sequence into chunks of $c$
positions, as online softmax divided the keys :cite:`Yang.Wang.Shen.ea.2024`.
Inside a chunk, the outputs come from the quadratic form, a causally masked
$c \times c$ matrix of kernel values. Across chunks, the state carries the
sum over all earlier chunks. For a chunk that starts at position $t_0$, with
$\mathbf{S}$ and $\mathbf{z}$ the sums over all positions before $t_0$,

$$
\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\,
    \mathbf{v}_s^\top}
  {\phi(\mathbf{q}_t)^\top \mathbf{z}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)}.
$$
:eqlabel:`eq_linear-attn-chunked`

The first term of the numerator reads the memory of the earlier chunks, and
the second attends within the chunk. Both are matrix products. For each
chunk, a $c \times d$ by $d \times c$ product gives the kernel values, and
the state advances by a $d \times c$ by $c \times d_v$ product. The total
work is $O(ncd + nd\,d_v)$, linear in $n$, and only one state per
chunk is stored.

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

The parallel, recurrent and chunked forms sum the same terms in three
orders, so they should agree up to floating-point rounding.
<!-- WRITER: placeholder lead for the merged check (kernelizing-the-score-2, now here; chunked-computation-2 deleted). One sentence (structural 3); it prints two deviations against the recurrent form and the state size. -->

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

The three forms agree to floating-point rounding. They differ only in what
they keep in memory and in how much of the work is matrix products.

At generation time this linear-attention layer with $d_h=64$ carries
16 KiB of state, whatever the context length. A softmax head instead caches
every past key and value, $2nd_h\cdot4$ bytes in fp32, or about 4 MiB at
$n=8192$. Its cache grows linearly with the generated sequence, whereas the
recurrent state stays fixed.

The state $\mathbf{S}_t$ is a lossy summary. It holds $d \times d_v$ numbers
however many tokens it has absorbed, so as the context grows the writes
interfere, and retrieving one particular value becomes approximate. Softmax
attention keeps every key and can put nearly all of a query's weight on one
of them. A kernel with positive features such as $\mathrm{elu} + 1$ cannot
become nearly one-hot in the same way: scaling up a query sharpens the
softmax without limit but leaves the normalized weights of this kernel almost
unchanged (Exercise :exref:`ex_attention-at-scale-feature-map`).

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

<!-- WRITER: new sharpening cell (COST-15). 1000 random queries, 8 random keys of width 64; mean over queries of the largest weight in the row. Measured (both tabs, three draws each in dev): softmax 0.36-0.37 at x1, 0.76-0.77 at x4, 0.94 at x16; elu+1 0.14-0.15 at x1, 0.15 at x4 and x16 (uniform 0.125). Softmax sharpens toward one key; the elu+1 weights barely move. The paragraph above ends with the claim and an exercise pointer; it now rests on this cell. Why: for large beta, phi(beta q) -> beta relu(q) + const, so the normalized weights tend to a fixed limit. -->

### Linear Attention in Practice

Plain kernelized attention of this kind did not replace softmax attention in
language models. At equal size it models language less well, and it is
weakest on tasks that must recall one particular earlier token
:cite:`Arora.Eyuboglu.Zhang.ea.2024,Yang.Wang.Shen.ea.2024`. Two changes
made the recurrent form competitive. First, the state learns to forget: a
decay computed from each input multiplies $\mathbf{S}_{t-1}$ before the new
write, so that old content fades when new content needs the room
(Exercise :exref:`ex_attention-at-scale-decaying-state` works out the case
of a constant decay).
Second, models keep a few softmax layers for recall. MiniMax-01, for
instance, places one softmax attention layer after every seven
linear-attention layers :cite:`MiniMax.2025`. A recurrence with a
fixed-size state that is updated linearly, as in
:eqref:`eq_linear-attn-recurrence`, is a *linear state space model*.
:numref:`sec_ssm` develops these models and their input-dependent decay from
this starting point, and :numref:`sec_matrix-state` extends
`linear_attention_chunked` to a state that decays.

## Measured Running Time

We now time the three strategies of this section on equal terms, as the
sequence grows from 512 to 32,768 tokens: dense causal attention in the
fused kernel, windowed attention with $w = 128$, and chunked linear
attention. All three process 32 heads of width 64 at once, for instance a
minibatch of 4 sequences with 8 heads each. They run in bfloat16, the
precision in which attention is trained, and `vmap` maps the one-head
functions of this section over the heads. Enough work per call keeps the
fixed cost of launching kernels from hiding the growth with $n$. The
milliseconds depend on the GPU, but the slopes on the log--log plot follow
from the operation counts.
<!-- WRITER: the timing is redesigned (COST-17; worklist 6.A6). Both arms run in ONE fused kernel family per tab: PyTorch `flex_attention` under `torch.compile` with a causal and with a band BlockMask; JAX `jax.nn.dot_product_attention(..., implementation='cudnn', is_causal=True)` without and with `local_window_size=(127, 0)`. Linear attention is no longer timed (its teaching implementation says little about a tuned kernel). 32 heads of width 64, bfloat16, 512 to 32,768 tokens. No vmap. State the prediction before the cell: a causal kernel visits T(T+1)/2 tiles of 128 x 128, a window of 128 visits 2T - 1 (the diagonal tile and its left neighbor in each row, one in the first), so the speedup should approach (T+1)/4 for T = n/128 tile rows, 64 at 32,768 tokens. The cell prints, per length, the measured speedup and this tile-count prediction. Measured (dev, two executions per tab, see the report): dense time grows about fourfold per doubling from 4,096 tokens on and the window's about twofold from 8,192 on; at 512-1,024 tokens both take about 0.1 ms and the speedup is about 1 (prediction 1.4-2.4); at 32,768 tokens the window is 30 to 55 times faster than dense (prediction 64.4). The shortfall: every tile the band visits straddles an edge of the mask (both are partial tiles), so it pays for masking entry by entry and for a fixed cost per row of tiles; in the causal kernel nearly every visited tile is interior. Quote only "about fourfold"/"about twofold", "several tens of times at 32,768 tokens", and "below the tile prediction"; no per-framework numbers in shared prose. The sentences on vmap, "Enough work per call keeps the fixed cost of launching kernels from hiding the growth with $n$" (delete, COST-17) and "three strategies" are now false. -->

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

<!-- WRITER: the reading paragraph below and the two tab notes that explained away tab differences are obsolete (the tab notes are deleted, worklist 6.A6). Linear attention is no longer in the figure; "linear attention is also faster during generation ..." as the last word goes (6.B6). The new reading is one shared paragraph from the printed speedups (see the marker above the cell), followed by the closing judgment (6.B6). The TinyCharLM quality cell (6.A7) was NOT added: its dev-time acceptance test failed (the dense model does not read the 64-byte copy; numbers in the report), so Exercise 9 stays whole and the ending rests on the sharpening cell and the reach caveat. -->

At short lengths all three take well under a millisecond, and fixed
per-call costs rather than the operation counts decide which is fastest. At
long lengths the time of dense attention grows about fourfold with every
doubling of $n$, while the times of the windowed and the chunked linear
mechanisms grow about twofold. At 32,768 tokens each of them therefore
needs well under half the time of dense attention, and the gap widens with
every further doubling. Dense attention remains the default at moderate
lengths, where it is exact and fast. At long contexts the mechanisms of
linear cost are faster, and linear attention is also faster during
generation, where its state stays the size of one matrix.

## Summary

<!-- WRITER: three or four sentences for this section (worklist 6.B7): windows, tiles, linear attention as a recurrence with a lossy matrix state, and the measured cost. -->

## Exercises

1. [code] **Tiles of a fused kernel.** A fused kernel works on tiles of
   $b \times b$ scores and skips a tile only if the mask excludes all of it.
   Let $T = n/b$.
    1. Show that a causal mask leaves $T(T+1)/2$ of the $T^2$ tiles, and
       that a window of width $w$, a multiple of $b$, leaves about
       $T(w/b + 1)$. Check both counts with `tiles_visited`. Below which
       width does narrowing the window stop saving tiles?
    1. Several documents are packed into one sequence of $n = 2048$ tokens,
       and the mask combines causality with "same document" as in
       :numref:`sec_attention-scoring-functions`, using `tiles_visited` and
       $b = 128$. Compare four documents of 512 tokens with documents of
       1,000, 600, 300, and 148 tokens. Explain why documents whose
       boundaries do not fall on tile boundaries cost more tiles.
    1. A kernel that receives the mask as an $n \times n$ boolean tensor
       must read it from memory. Compare the size of that tensor at
       $n = 32{,}768$ with the size of $\mathbf{Q}$, $\mathbf{K}$, and
       $\mathbf{V}$ for 8 heads of width 64 in bfloat16, and explain why
       FlexAttention evaluates the mask as a rule, once per tile.

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
    1. Multiply the queries by $\beta \in \{1, 4, 16\}$, and recompute the
       mean row entropy for the three maps and for the softmax weights.
       Explain from a property of each $\phi$ which maps cannot sharpen
       their weights in this way, and what this implies for approximating
       a softmax row that is nearly one-hot.
   :exlabel:`ex_attention-at-scale-feature-map`

1. **An exact feature map.** ● The softmax kernel $\kappa(\mathbf{q},
   \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})$ also
   factorizes, but only with infinitely many features.
    1. Expand $\kappa$ in its Taylor series, and show that the degree-$p$
       term equals $\phi_p(\mathbf{q})^\top \phi_p(\mathbf{k})$ for a
       feature map $\phi_p$ of dimension $d^p$ built from the $p$-fold
       tensor product of its argument.
    1. Truncate the series after degree $2$. Show that the truncated kernel
       is positive for every pair of $\mathbf{q}$ and $\mathbf{k}$, so the
       normalizer of :eqref:`eq_linear-attn` cannot vanish, and give the
       dimension of the feature map that concatenates $\phi_0$, $\phi_1$,
       and $\phi_2$. For $d = 64$ and values of width $64$, compute the
       size of the state $(\mathbf{S}, \mathbf{z})$ in single precision and
       the length $n$ at which the key--value cache of a softmax head
       reaches the same size. A truncation of this kind is the feature map
       of Based :cite:`Arora.Eyuboglu.Zhang.ea.2024`.
    1. Take as given that the $n \times n$ matrix with entries
       $\kappa(\mathbf{x}_i, \mathbf{x}_j)$ is positive definite for
       distinct $\mathbf{x}_1, \ldots, \mathbf{x}_n$. Show that no feature
       map of dimension $D < n$ reproduces $\kappa$ exactly on these points
       when queries and keys coincide.

1. [extended] **Attention variants in a language model.** ● Whether the
   quality gap between exact and linear attention matters depends on the
   task. Measure it on `TinyCharLM` of
   :numref:`sec_positional-information`, with `pos='learned'` for every
   variant so that position enters the same way.
    1. Subclass `TinyCharLM` and override `attend` twice: once with a
       causal band mask of width $w$ in place of the causal mask, and once
       with the causal linear-attention weights of :eqref:`eq_linear-attn`
       for $\phi(x) = \mathrm{elu}(x) + 1$, computed explicitly as a
       $T \times T$ matrix so that the method still returns its weights.
       Add $\epsilon = 10^{-6}$ to the normalizer, as in
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
   <!-- WRITER: exercise 4 updated (lib review 5): part 1 now tells readers to add eps = 1e-6 (the section's three forms carry it too; dev: without it the linear arm diverged to NaN in one run of three in both frameworks); part 3 re-posed because at the protocol's budget the dense two-block model barely reads the copy (dev second-minus-first-half loss -0.004 to -0.020 nats). Dev numbers, both tabs, converged runs: validation perplexity at 128 dense 9.5-10.1, window w = 16 8.3-9.3, linear 11.2-11.5. -->

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

<!-- WRITER: the deck below is the second half of the old attention-at-scale deck, moved as it was. It needs an opener with a number, goals, map, the sharpening check, the redesigned timing, what each mechanism gives up, and a closer and recap (worklist 6.B9). -->

::: {.slide title="A Sliding Window Turns the Causal Mask into a Band"}
[Windows]{.kicker}

Each query attends to its $w$ most recent positions:
`causal_mask(n) & (i - j < w)`.

$2nw$ scores instead of $n^2$: linear in $n$ for fixed $w$.
:::

::: {.slide title="Depth Restores the Reach: 1 + L(w − 1) Positions"}
[Windows]{.kicker}

| layers $L$ | 1 | 2 | 4 | 32 |
|:--|--:|--:|--:|--:|
| reach $1 + L(w-1)$, $w = 8$ | 8 | 15 | 29 | 225 |

Mistral 7B: $w = 4096$ in 32 layers reaches beyond 100,000 tokens, but
the path length is $O(n/w)$ instead of $O(1)$.
:::

::: {.slide title="Global Layers Keep Direct Paths; Windows Shrink the Cache"}
[Windows]{.kicker}

$r$ windowed layers per global layer: KV cache relative to all-global
$= \dfrac{r\min(n, w) + n}{(r+1)\,n}$.

| Gemma 3: $r = 5$, $w = 1024$ | 8K | 32K | 128K |
|:--|--:|--:|--:|
| cache vs. all-global | 0.27 | 0.19 | 0.17 |

Global layers restore $O(1)$ paths; Llama 4 also drops RoPE there
(as in NoPE).

[Gemma Team 2025; OpenAI 2025; Meta 2025]{.small}
:::

::: {.slide title="A Fused Kernel Skips Work Only in Whole Tiles"}
[Windows]{.kicker}

Blocked windowed attention matches the masked definition:

@!attention-at-scale-a-linear-cost-implementation-2

Tiles of $128 \times 128$ scores at $n = 2048$: causal touches about half,
a window of 128 about one in eight.

::: {.d2l-note}
FlexAttention: write the mask as a rule on positions; it is evaluated once
per tile and compiled into a fused kernel.
:::
:::

::: {.slide title="Check: Tiles for a Narrower Window"}
[Check]{.kicker}

Tiles of $b = 128$, $n = 2048$. A window of $w = 128$ touches 31 of the 256
tiles. How many does a window of $w = 32$ touch? Of $w = 512$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $w = 32$: still 31, the diagonal tile and its left neighbor in
each row; narrowing below a tile saves nothing. $w = 512$: up to 5 tiles
per row, $16 \cdot 5 - 10 = 70$.
:::
:::
:::

::: {.slide title="A Factorized Kernel Turns Attention into a Recurrence"}
[Linear attention]{.kicker}

Replace $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d})$ by
$\phi(\mathbf{q})^\top\phi(\mathbf{k})$, with $\phi = \mathrm{elu} + 1 > 0$.
The query factors out of the sum:

$$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t}$$
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)$$

::: {.d2l-note}
A recurrent network with a $d \times d_v$ matrix state: write by outer
products, read with the query.
:::

[Katharopoulos et al. 2020]{.small}
:::

::: {.slide title="The State Stays 16 KiB While a KV Cache Grows"}
[Linear attention]{.kicker}

@!attention-at-scale-kernelizing-the-score-2

- Parallel and chunked forms agree with the recurrent form.
- One head, $d_h = 64$: 16 KiB of state at any length; a softmax KV cache
  is 4 MiB at 8,192 tokens.
- Lossy: the state cannot hold every token, and positive features cannot
  become nearly one-hot.
:::

::: {.slide title="Chunks Turn Linear Attention into Matrix Products"}
[Linear attention]{.kicker}

Chunks of $c$ positions:

- **inside a chunk:** a causally masked $c \times c$ matrix of kernel
  values;
- **across chunks:** carry $\mathbf{S}$ and $\mathbf{z}$, one update per
  chunk.

Work $O(ncd + nd\,d_v)$, all matrix products; one state per chunk.

The same terms in a third order: the result matches the recurrent form.
:::

::: {.slide title="Practical Linear Attention Adds a Decay and a Few Softmax Layers"}
[Linear attention]{.kicker}

- Plain linear attention models language less well and fails at recalling
  one particular token.
- **Forgetting:** an input-dependent decay multiplies $\mathbf{S}_{t-1}$
  before each write.
- **Hybrids:** MiniMax-01 places one softmax layer after every seven
  linear-attention layers.

::: {.d2l-note}
A linear recurrence with a fixed-size state is a state space model.
:::

[Arora et al. 2024; Yang et al. 2024; MiniMax 2025]{.small}
:::

<!-- WRITER: the timing slide below shows the redesigned cell (dense causal against window w = 128 in one fused kernel per tab, no linear attention; prints speedup against the tile-count prediction). Its title and text ("Windowed and Linear as n", "windowed and chunked linear about 2x", "well under half") are out of date. Code step deck fixes: the random band heatmap (cut) is gone from the first window slide; the reach slide shows the formula as a table in place of the cut receptive-field cell; the chunk slide no longer embeds the deleted chunked-computation-2 (the merged check kernelizing-the-score-2 on the 16 KiB slide prints both deviations). The new sharpening cell `windowed-linear-attention-sharpening` has no slide yet (6.B9). -->

::: {.slide title="Dense Time Grows as n², Windowed and Linear as n"}
[Measured]{.kicker}

@!attention-at-scale-the-price-of-attention-measured

32 heads of width 64, bfloat16. At long lengths dense grows about 4× per
doubling of $n$, windowed and chunked linear about 2×; at 32,768 tokens
both need well under half of dense's time. Short lengths measure launch
costs, not arithmetic.
:::
