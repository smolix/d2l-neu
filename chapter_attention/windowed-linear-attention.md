# Windowed and Linear Attention
:label:`sec_windowed-linear-attention`

Fused kernels (:numref:`sec_attention-at-scale`) eliminate the $O(n^2)$
intermediate score matrix, yet both their $\Theta(n^2 d)$ FLOP count and
their key--value cache still grow with sequence length $n$. When processing a
100,000-token document, however, most tokens depend primarily on local
context, while only occasional lookups require long-range retrieval.

Two primary architectural strategies reduce attention's computational cost to
$O(n)$:

1. **Sliding-window attention** :cite:`child2019generating,beltagy2020longformer`
   restricts each query to its $w$ most recent keys, reducing per-layer work
   to $O(nw)$ and bounding the key--value cache at $w$ tokens.
2. **Linear attention** :cite:`Katharopoulos.Vyas.Pappas.ea.2020` replaces
   the softmax kernel $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with a
   factorized kernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$, allowing keys
   and values to be accumulated into a fixed-size $d_h \times d_v$ recurrent
   state matrix.

Both approaches compute a different function from dense softmax attention.
This section analyzes how sliding windows trade single-hop reach for tile
sparsity, derives the parallel, recurrent, and chunked forms of linear
attention, and demonstrates why factorized feature maps lack the sharpening
temperature of softmax.

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

In causal *sliding-window attention* of width $w$, query $i$ attends to key
$j$ only if $j \le i$ and $i - j < w$. This conjunction of causality and
locality replaces the lower triangle of $n(n+1)/2$ active pairs with a
diagonal band of at most $nw$ pairs:

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

### Receptive Field Growth with Depth
:label:`subsec_attn-receptive-field`

Just as stacked convolutions expand their receptive field with depth
(:numref:`chap_cnn`), stacking $L$ sliding-window layers of width $w$ allows
information to propagate $w - 1$ steps backward per layer, yielding a
theoretical receptive field of

$$1 + L(w - 1)$$

tokens. For example, Mistral 7B v0.1
:cite:`Jiang.Sablayrolles.Mensch.ea.2023` used $w = 4{,}096$ across $L = 32$
layers, spanning up to $131{,}041$ tokens. However, theoretical reach differs from direct retrieval: whereas
full self-attention connects any two tokens in a single hop, a stack of
windows requires $\lceil \Delta / (w - 1) \rceil$ sequential hops across
layers to relay a fact across distance $\Delta$. A relayed fact is also
diluted at every hop. :numref:`fig_window-receptive-field` shows the smallest
case: under uniform window averaging ($w = 4$, $L = 4$), a token at the
boundary $\Delta = 12$ arrives along a single four-hop path with weight
$(1/4)^4 = 1/256$, compared to $1/13$ in a single global layer.

![Stacked windows against one global layer. Left: with four windowed layers of width $w = 4$, the output at the top right depends on a cone of earlier positions, and the input $\Delta = 12$ positions back reaches it along a single path of four hops (blue), with weight $(1/4)^4 = 1/256$ under uniform attention weights. Right: one global layer connects the same two positions in one hop, with weight $1/13$.](../img/mdl-attention-window-receptive-field.svg)
:label:`fig_window-receptive-field`

### Global Layers

To preserve one-hop long-range retrieval while retaining most of the memory
and compute savings of local windows, modern long-context architectures
interleave $r$ windowed layers with one full (*global*) attention layer—for
example, Gemma 3 uses $r = 5$ windowed layers ($w = 1{,}024$) per global
layer :cite:`Gemma.Team.2025`. Because a windowed layer caches only the last
$\min(n, w)$ keys and values during generation (:numref:`sec_kv-cache`),
interleaving reduces the total KV cache to the fraction

$$
\frac{r \min(n, w) + n}{(r + 1)\, n} \;\xrightarrow{n \gg w}\; \frac{1}{r + 1}
$$

of an all-global stack ($\approx 27\%$ at $n = 8{,}192$ and $1/6$
asymptotically for $r = 5, w = 1{,}024$).

### Tiles in a Fused Kernel

Applying `window_mask` inside a dense $n \times n$ matrix multiplication
still computes all $n^2$ scores. To realize $O(nw)$ time and memory, an
implementation must evaluate only the active band—for instance, by grouping
queries into blocks of size $w$ and attending only to the current and
immediately preceding key blocks ($2nw$ scores total):

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab pytorch
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d_h = Q.shape
    assert n % w == 0, 'blocks must tile the sequence exactly'
    Qb, Kb, Vb = (X.reshape(n // w, w, -1) for X in (Q, K, V))
    # Each query block attends to its own key block and the one before it
    prev = lambda X: torch.cat([torch.zeros_like(X[:1]), X[:-1]])
    Kb, Vb = torch.cat([prev(Kb), Kb], 1), torch.cat([prev(Vb), Vb], 1)
    i = torch.arange(w, device=Q.device)[:, None]      # Query in its block
    j = torch.arange(-w, w, device=Q.device)[None, :]  # Key offset from it
    first = torch.arange(n // w, device=Q.device)[:, None, None] == 0
    mask = (j <= i) & (i - j < w) & ~(first & (j < 0))  # No keys before 0
    scores = Qb @ Kb.transpose(-1, -2) / math.sqrt(d_h)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab jax
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d_h = Q.shape
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
    scores = Qb @ Kb.swapaxes(-1, -2) / math.sqrt(d_h)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

In a tiled GPU kernel such as FlashAttention or FlexAttention
:cite:`Dong.Feng.Guessous.ea.2025`, computation proceeds in $b \times b$
tiles ($b = 128$ by default in FlexAttention), and a tile is skipped only
when *all* $b^2$ of its entries are masked out. At $n = 2{,}048$ and $b = w = 128$, the causal
mask visits $T(T+1)/2$ of the $T^2$ tiles ($53.1\%$ for $T = n/b = 16$),
whereas the sliding window visits only the diagonal tile and its left
neighbor in each tile row ($2T - 1 = 31$ tiles, or $12.1\%$;
:numref:`fig_window-tiles`):

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

![Tiles visited by a fused kernel, for $T = 16$ tile rows. Solid entries are allowed scores, and the faint remainder of a square marks a tile that the mask cuts but the kernel must still visit. The causal mask (left) visits $T(T+1)/2 = 136$ tiles and cuts only the $16$ on the diagonal. The window of width $w = b$ (right) visits the diagonal tile and its left neighbor in each row, $2T - 1 = 31$ tiles, and cuts every one of them.](../img/mdl-attention-window-tiles.svg)
:label:`fig_window-tiles`

Both `windowed_attention` and the framework's fused sliding-window kernel
match the
dense masked definition up to float32 round-off:

:begin_tab:`pytorch`
The fused kernel is `flex_attention` with `create_block_mask`.
:end_tab:

:begin_tab:`jax`
The fused kernel is `dot_product_attention` with `local_window_size=(w - 1, 0)`.
:end_tab:

```{.python .input #attention-at-scale-a-linear-cost-implementation-2}
%%tab pytorch
n, w, d_h = 2048, 128, 64
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
reference = d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
                               window_mask(n, w, Q.device)) @ V
# create_block_mask evaluates the band at every position and records which
# tiles are empty, full or cut by it; compiled, the kernel skips the empty ones
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

### Windowed and Dense Kernels Compared

With $T = n / 128$ tile rows, the ratio of active causal tiles to active
windowed tiles ($w = b = 128$) is $T(T + 1) / (2(2T - 1)) \approx T/4$,
predicting a $64\times$ theoretical tile reduction at $n = 32{,}768$ ($T =
256$). We benchmark both masks inside the same fused bfloat16 kernel across
$n \in [512, 32{,}768]$ ($H = 32$ heads, $d_h = 64$):

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

Beyond $n \approx 2{,}048$, dense causal runtime scales quadratically while
windowed runtime scales linearly, reaching a $> 30\times$ speedup at $n =
32{,}768$. This is between one half and three quarters of the $64\times$ that
the tile count predicts. The count treats every visited tile as the same
amount of work, so it is an upper bound on the gain and not an estimate of it.

:begin_tab:`pytorch`
In FlexAttention every visited tile of a width-$b$ band is cut by the mask
boundary (:numref:`fig_window-tiles`) and pays for elementwise masking,
whereas the interior tiles of the causal triangle run unmasked.
:end_tab:

:begin_tab:`jax`
cuDNN does not document how it tiles the score matrix, so $b = 128$ is our
assumption here.
:end_tab:

## Linear Attention

### Kernelizing the Score

Rather than restricting which pairs are scored, *linear attention*
:cite:`Katharopoulos.Vyas.Pappas.ea.2020` replaces the exponential similarity
$\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with a factorized kernel
$K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ using a
nonnegative elementwise feature map $\phi: \mathbb{R}^{d_h} \to
\mathbb{R}_+^{d_h}$. By the associativity of matrix multiplication,

$$
\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\,\mathbf{V}
= \phi(\mathbf{Q})\,\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr),
$$

avoiding the $n \times n$ matrix $\phi(\mathbf{Q})\phi(\mathbf{K})^\top$ in
favor of a $d_h \times d_v$ summary matrix. Under a causal mask ($s \le t$),
factoring $\phi(\mathbf{q}_t)^\top$ out of the numerator and denominator sums
yields

$$
\begin{aligned}
\mathbf{o}_t^\top
&= \frac{\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)\,
        \mathbf{v}_s^\top}
       {\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}
= \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}
       {\phi(\mathbf{q}_t)^\top \mathbf{z}_t}, \\
\mathbf{S}_t &= \sum_{s \leq t} \phi(\mathbf{k}_s) \mathbf{v}_s^\top,
\qquad
\mathbf{z}_t = \sum_{s \leq t} \phi(\mathbf{k}_s).
\end{aligned}
$$
:eqlabel:`eq_linear-attn`

Following :citet:`Katharopoulos.Vyas.Pappas.ea.2020`, we choose $\phi(x) =
\mathrm{elu}(x) + 1 > 0$ elementwise (with a small $\epsilon = 10^{-6}$ in
the denominator against float32 underflow). The prefix sums $(\mathbf{S}_t,
\mathbf{z}_t)$ satisfy the linear recurrence

$$
\begin{aligned}
\mathbf{S}_t &= \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\, \mathbf{v}_t^\top, \\
\mathbf{z}_t &= \mathbf{z}_{t-1} + \phi(\mathbf{k}_t),
\end{aligned}
$$
:eqlabel:`eq_linear-attn-recurrence`

turning causal attention into an RNN whose hidden state is a $d_h \times d_v$
associative memory matrix updated by outer products (a *fast weight
programmer*, :cite:`Schlag.Irie.Schmidhuber.2021`). We can evaluate
:eqref:`eq_linear-attn` either in parallel via cumulative sums (`cumsum`) or
recurrently step by step in $O(d_h d_v)$ memory:

```{.python .input #attention-at-scale-kernelizing-the-score-1}
%%tab pytorch
def phi(X):
    return F.elu(X) + 1

def linear_attention_parallel(Q, K, V, eps=1e-6):
    """Training form: all outputs at once via cumulative sums."""
    Qp, Kp = phi(Q), phi(K)
    S = torch.cumsum(Kp[:, :, None] * V[:, None, :], dim=0)  # (n, d_h, d_v)
    z = torch.cumsum(Kp, dim=0)                              # (n, d_h)
    den = (Qp * z).sum(-1, keepdim=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d_h, d_v) state, updated token by token."""
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
    S = jnp.cumsum(Kp[:, :, None] * V[:, None, :], axis=0)  # (n, d_h, d_v)
    z = jnp.cumsum(Kp, axis=0)                              # (n, d_h)
    den = (Qp * z).sum(-1, keepdims=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d_h, d_v) state, updated token by token."""
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

During training on GPUs, materializing all $n$ intermediate matrices
$\mathbf{S}_t \in \mathbb{R}^{d_h \times d_v}$ in `linear_attention_parallel`
consumes $O(n d_h d_v)$ memory (256 MiB per head at $n = 16{,}384, d_h = d_v
= 64$), whereas `linear_attention_recurrent` executes $n$ sequential vector
steps that underutilize GPU matrix cores. Partitioning the sequence into
chunks of length $c$ :cite:`Yang.Wang.Shen.ea.2024` combines both:for a
chunk starting at $t_0$ with pre-chunk state $(\mathbf{S}, \mathbf{z})$,

$$
\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\,
    \mathbf{v}_s^\top}
  {\phi(\mathbf{q}_t)^\top \mathbf{z}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)},
$$
:eqlabel:`eq_linear-attn-chunked`

so both intra-chunk attention ($c \times c$) and inter-chunk state propagation
use dense matrix multiplications in $O(nc(d_h + d_v) + n d_h d_v)$ FLOPs and
store only $n/c$ chunk states:

```{.python .input #attention-at-scale-chunked-computation-1}
%%tab pytorch
def linear_attention_chunked(Q, K, V, c=64, eps=1e-6):
    """Training form in chunks: quadratic within, recurrent across."""
    n, d_h = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d_h), phi(K).reshape(n // c, c, d_h)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = torch.cumsum(Kp.transpose(-1, -2) @ Vc, dim=0)   # (n/c, d_h, d_v)
    z = torch.cumsum(Kp.sum(1), dim=0)                   # (n/c, d_h)
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
    n, d_h = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d_h), phi(K).reshape(n // c, c, d_h)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = jnp.cumsum(Kp.swapaxes(-1, -2) @ Vc, axis=0)     # (n/c, d_h, d_v)
    z = jnp.cumsum(Kp.sum(1), axis=0)                    # (n/c, d_h)
    S = jnp.concatenate([jnp.zeros_like(S[:1]), S[:-1]])
    z = jnp.concatenate([jnp.zeros_like(z[:1]), z[:-1]])
    A = jnp.tril(Qp @ Kp.swapaxes(-1, -2))               # (n/c, c, c)
    num = Qp @ S + A @ Vc                                # Past + this chunk
    den = Qp @ z[..., None] + A.sum(-1, keepdims=True) + eps
    return (num / den).reshape(n, -1)
```

All three forms compute the exact same outputs up to float32 rounding, and
the recurrent state occupies a constant 16 KiB per head ($64 \times 64 + 64$
floats) regardless of sequence length:

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

### Capacity and Sharpness

Compressing an arbitrary number of key--value pairs into a $d_h \times d_v$
state matrix incurs two fundamental costs:

1. **Finite associative capacity:** Because $\phi(\mathbf{k}_s) \in
   \mathbb{R}_+^{d_h}$ is nonnegative, a query can isolate a single stored
   value without cross-talk only if key feature vectors are nearly orthogonal
   (nearly disjoint supports), limiting exact retrieval to at most $d_h$
   keys.
2. **Lack of temperature sharpening:** In softmax attention, scaling a query
   by $\beta > 1$ exponentiates score differences ($e^{\beta \Delta}$),
   driving the distribution toward a one-hot lookup as $\beta \to \infty$. In
   linear attention with $\phi(x) = \mathrm{elu}(x) + 1$, scaling $\mathbf{q}$
   by large $\beta$ yields $\phi(\beta \mathbf{q}) \approx \beta\,
   \mathrm{relu}(\mathbf{q})$, so the scale factor $\beta$ cancels out between
   numerator and denominator (Exercise
   :exref:`ex_attention-at-scale-feature-map`):

```{.python .input #windowed-linear-attention-sharpening}
%%tab pytorch
Q = torch.randn(1000, d_h, device=d2l.try_gpu())
K = torch.randn(8, d_h, device=d2l.try_gpu())
for beta in (1, 4, 16):  # Scale the queries up; 8 keys: uniform is 0.125
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

### Recall Under Each Mechanism

Both limits, and the limit of a window, appear in the simplest lookup we can
pose. We store $N$ random key--value pairs of width $d_h = 64$, query with one
of the stored keys, and count a hit when the output has a larger inner product
with the right value than with any other stored value. Four mechanisms answer
the query: dense softmax attention; a window that sees only the last $w = 16$
pairs; linear attention with the elu+1 feature map; and the *outer-product
state* $\mathbf{q}^\top \sum_i \mathbf{k}_i \mathbf{v}_i^\top$, which is
linear attention with the identity feature map and no normalizer. The last
one sets the feature map aside, so that it shows the capacity of a $d_h
\times d_v$ state by itself:

```{.python .input #windowed-linear-attention-recall-under-each-mechanism}
%%tab pytorch
def recall(N, w=16, trials=1000):
    """Look up one of N stored pairs by its key; fraction of hits per rule."""
    K, V = (torch.randn(trials, N, d_h, device=d2l.try_gpu())
            for _ in range(2))
    target = torch.randint(0, N, (trials,), device=K.device)
    rows = torch.arange(trials, device=K.device)
    q = K[rows, target][:, None]                       # (trials, 1, d_h)
    scores = q @ K.transpose(1, 2) / math.sqrt(d_h)    # (trials, 1, N)
    recent = torch.arange(N, device=K.device) >= N - w  # The last w pairs
    lin = phi(q) @ phi(K).transpose(1, 2)
    outputs = {'softmax': torch.softmax(scores, -1) @ V,
               f'window (w={w})': d2l.masked_softmax(scores, recent) @ V,
               'elu+1': lin / lin.sum(-1, keepdim=True) @ V,
               'outer-product state': q @ (K.transpose(1, 2) @ V)}
    # A hit: of the N stored values, the target is nearest to the output
    return {name: float(((O @ V.transpose(1, 2)).argmax(-1)[:, 0] == target)
                        .float().mean()) for name, O in outputs.items()}

for N in (16, 64, 256, 1024):
    print(f'N = {N:4}: ' + ', '.join(f'{name} {hit:.2f}'
                                     for name, hit in recall(N).items()))
```

```{.python .input #windowed-linear-attention-recall-under-each-mechanism}
%%tab jax
def recall(N, w=16, trials=1000):
    """Look up one of N stored pairs by its key; fraction of hits per rule."""
    key_K, key_V, key_t = jax.random.split(d2l.get_key(), 3)
    K, V = (jax.random.normal(k, (trials, N, d_h)) for k in (key_K, key_V))
    target = jax.random.randint(key_t, (trials,), 0, N)
    q = K[jnp.arange(trials), target][:, None]         # (trials, 1, d_h)
    scores = q @ K.swapaxes(1, 2) / math.sqrt(d_h)     # (trials, 1, N)
    recent = jnp.arange(N) >= N - w                    # The last w pairs
    lin = phi(q) @ phi(K).swapaxes(1, 2)
    outputs = {'softmax': jax.nn.softmax(scores, axis=-1) @ V,
               f'window (w={w})': d2l.masked_softmax(scores, recent) @ V,
               'elu+1': lin / lin.sum(-1, keepdims=True) @ V,
               'outer-product state': q @ (K.swapaxes(1, 2) @ V)}
    # A hit: of the N stored values, the target is nearest to the output
    return {name: float(((O @ V.swapaxes(1, 2)).argmax(-1)[:, 0] == target)
                        .mean()) for name, O in outputs.items()}

for N in (16, 64, 256, 1024):
    print(f'N = {N:4}: ' + ', '.join(f'{name} {hit:.2f}'
                                     for name, hit in recall(N).items()))
```

Softmax attention finds the pair at every $N$. The window finds it only when
the pair lies among the last $w$, so its recall is about $w/N$: perfect at
$N = 16$, a quarter at $N = 64$, and a few percent at $N = 1{,}024$. More
striking is elu+1 linear attention, which fails even at $N = 16$, where the
state has four times as many rows as there are pairs to store. Its weights
cannot sharpen, so the output is close to the average of all stored values,
and the right one barely stands out.

The outer-product state does much better, and it shows the capacity limit
cleanly. For the query $\mathbf{q} = \mathbf{k}_j$ its output is
$\|\mathbf{k}_j\|^2\, \mathbf{v}_j \approx d_h \mathbf{v}_j$ plus crosstalk
$\sum_{i \ne j} (\mathbf{k}_j^\top \mathbf{k}_i)\, \mathbf{v}_i$ from the
other pairs. Each crosstalk coefficient has size about $\sqrt{d_h}$, so the
noise relative to the signal is about $\sqrt{(N - 1)/d_h}$. Recall is nearly
perfect up to $N \approx d_h$, drops to about two thirds at $N = 4 d_h$, and
has collapsed at $N = 16 d_h$. Softmax attention keeps all $N$ pairs and
recalls at every $N$; a state of fixed size recalls only while $N$ stays near
its width.

## Choosing a Mechanism

* **Exact fused attention** is the default for short-to-medium contexts,
  providing exact one-hop retrieval in $O(n)$ working memory.
* **Interleaved sliding-window and global layers** are the standard for
  long-context transformers (e.g., Gemma 3), cutting KV-cache memory and tile
  FLOPs on local layers while preserving one-hop global routing on periodic
  global layers.
* **Linear attention** requires data-dependent decay gates (so old entries
  can be forgotten; see Exercise :exref:`ex_attention-at-scale-decaying-state`
  and :numref:`chap_modern_rnn`) and hybrid interleaving with full softmax
  layers (e.g., MiniMax-01, :cite:`MiniMax.2025`) to handle exact associative
  recall :cite:`Arora.Eyuboglu.Zhang.ea.2024,Yang.Wang.Shen.ea.2024`.

## Summary

* Sliding-window attention restricts each query to the $w$ most recent keys,
  achieving $O(nw)$ time and $O(w)$ cache memory; $L$ stacked layers span
  $1 + L(w - 1)$ tokens via multi-hop relays and are typically interleaved
  with periodic global attention layers.
* Tiled GPU kernels skip work only on completely empty $b \times b$ tiles,
  yielding speedups proportional to the fraction of skipped tiles (tempered by
  boundary masking overhead).
* Linear attention factorizes the similarity kernel as $\phi(\mathbf{q})^\top
  \phi(\mathbf{k})$, turning causal attention into an $O(n)$ recurrence with
  a constant-size $d_h \times d_v$ matrix state that can be trained in chunks.
* Because a fixed-size state with nonnegative features cannot sharpen weights
  via query scaling or store more than $O(d_h)$ orthogonal keys without
  interference, practical linear-attention models add gating and hybrid
  softmax layers.

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

Windowed, Sparse, and Linear Attention<br>**sliding-window masks · multi-hop receptive fields · tile sparsity · kernelized linear attention · recurrent & chunked forms**
:::
:::

::: {.slide title="Why Fused Exact Attention Still Costs O(n²) Time"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- FlashAttention eliminates the $O(n^2)$ score matrix in HBM, yet at $n = 32{,}768$ ($T = 256$ tile rows of $b = 128$) it still visits $T(T+1)/2 = \mathbf{32{,}896}$ tiles and caches all $32{,}768$ KV tokens!
- To cut arithmetic below $\Theta(n^2 d)$ and bound generation memory, we must either **score fewer pairs** (windows: $511$ tiles) or **factorize the similarity kernel** (linear attention: $d_h \times d_v$ state).
:::

::: {.col .fig .big}
@fig:ch10-attn-win-tile-wall
:::
:::
:::

::: {.slide title="Two Architectural Routes to O(n) Attention"}
[Design Trade-Offs]{.kicker}

::: {.cols .vc}
::: {.col}
- **Route 1 — Sliding-Window Attention:** Restricts each query $i$ to its $w$ most recent keys ($i - w < j \le i$), preserving exact softmax sharpness locally at $O(nw)$ cost.
- **Route 2 — Linear Attention:** Replaces $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with a nonnegative feature map product $\phi(\mathbf{q})^\top\phi(\mathbf{k})$, collapsing all past keys/values into a fixed $d_h \times d_v$ state matrix.
:::

::: {.col .fig .big}
@fig:ch10-attn-win-two-routes
:::
:::
:::

::: {.slide title="Sliding-Window Attention: The Causal Band Mask"}
[Local Attention]{.kicker}

A causal sliding window of width $w$ permits query $i$ to attend to key $j$ if and only if $j \le i$ (causality) and $i - j < w$ (locality):

@attention-at-scale-attention-through-a-window

- **Boolean Conjunction:** `window_mask(n, w)` is the logical AND (`&`) of `d2l.causal_mask(n)` and the offset constraint `i[:, None] - i[None, :] < w`.
- **Linear Pair Count:** Whereas the causal triangle retains $n(n+1)/2$ active pairs, the sliding band retains at most $nw$ pairs ($\approx 4.2 \times 10^6$ vs. $5.4 \times 10^8$ at $n = 32{,}768, w = 128$).
:::

::: {.slide title="Receptive Field Growth Across Stacked Window Layers"}
[Multi-Hop Propagation]{.kicker}

::: {.cols .vc}
::: {.col}
- Stacking $L$ sliding-window layers of width $w$ steps up to $w - 1$ positions back per layer:
  $$\textrm{Receptive Field}(L, w) = 1 + L(w - 1)$$
- **Theoretical Reach vs. Effective Attenuation:** Reaching distance $\Delta$ requires $\lceil \Delta / (w - 1) \rceil$ hops across layers; under uniform window weights ($w = 4, L = 4$), a boundary token at $\Delta = 12$ arrives with weight $(1/4)^4 = 1/256$.
:::

::: {.col .fig}
@fig:ch10-attn-win-receptive-field
:::
:::
:::

::: {.slide title="Interleaving Windowed and Global Attention Layers"}
[Hybrid Long-Context Architectures]{.kicker}

::: {.cols .vc}
::: {.col}
- Modern long-context transformers interleave $r$ windowed layers with $1$ full global layer:
  $$\frac{\textrm{Hybrid Cache}}{\textrm{All-Global Cache}} = \frac{r \min(n, w) + n}{(r + 1)\,n} \;\xrightarrow{\;n \gg w\;}\; \frac{1}{r + 1}$$
- **Gemma 3 ($r = 5, w = 1{,}024$):** Shrinks the KV cache to **$17.3\%$** at $n = 128\text{K}$ while guaranteeing a 1-hop global attention path every 6th layer!
:::

::: {.col .fig}
@fig:ch10-attn-win-hybrid-schedule
:::
:::
:::

::: {.slide title="Realizing O(nw) Cost via Blocked Band Evaluation"}
[Blocked Implementation]{.kicker}

Passing `window_mask(n, w)` into a dense $n \times n$ matrix product still computes all $n^2$ scores! To run in true $O(nw)$ time and memory, `windowed_attention` tiles queries into blocks of $w$ and scores each block only against its own and the previous key block ($2nw$ scores):

@attention-at-scale-a-linear-cost-implementation-1
:::

::: {.slide title="How Fused Kernels Exploit Block-Mask Tile Sparsity"}
[Hardware Tile Skipping]{.kicker}

::: {.cols .vc}
::: {.col}
- Tiled GPU kernels (FlashAttention, flex-attention-style kernels, cuDNN) partition the $n \times n$ score matrix into $b \times b$ tiles (typically $b = 128$) and skip a tile **only if all $b^2$ of its entries are masked out**.
- At $n = 2{,}048$ and $b = w = 128$ ($T = n/b = 16$ tile rows):
  - **Causal mask:** Visits $T(T+1)/2 = 136$ of $256$ tiles (**$53.1\%$**).
  - **Sliding window ($w = 128$):** Visits only the diagonal tile and its immediate left neighbor per row: $2T - 1 = 31$ tiles (**$12.1\%$**).
:::

::: {.col .fig}
@!attention-at-scale-a-linear-cost-implementation-3

@!attention-at-scale-a-linear-cost-implementation-2
:::
:::
:::

::: {.slide title="Benchmarking Windowed vs. Dense Causal Kernels" layout="tight"}
[Wall-Clock Scaling up to 32K]{.kicker}

::: {.cols .vc}
::: {.col}
- We time dense causal vs. sliding-window ($w = 128$) attention inside the same compiled `bfloat16` fused kernel ($H = 32, d_h = 64$) from $n = 512$ to $n = 32{,}768$:
- **Linear vs. Quadratic Slope:** Above $n = 2{,}048$, dense runtime quadruples per doubling of $n$, while windowed runtime grows linearly ($2\times$ per doubling).
- **$> 30\times$ Speedup at $32\text{K}$:** One half to three quarters of the $64.4\times$ tile ratio. The tile count treats every visited tile as equal work, so it is an upper bound on the gain.

@!attention-at-scale-the-price-of-attention-measured[out=1]
:::

::: {.col .fig .narrow}
@!attention-at-scale-the-price-of-attention-measured[out=2]
:::
:::
:::

::: {.slide title="Linear Attention: Factorizing the Similarity Kernel"}
[Factorized Kernels]{.kicker}

::: {.cols .vc}
::: {.col}
- **Linear attention** (Katharopoulos et al., 2020) replaces $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with $K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ where $\phi(x) = \mathrm{elu}(x) + 1 > 0$:
  $$\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\,\mathbf{V} \;=\; \phi(\mathbf{Q})\,\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr)$$
- **The Associativity Trick:** Computing $\phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{d_h \times d_v}$ first reduces cost from $\Theta(n^2 d_h)$ to **$\Theta(n d_h d_v)$** without ever forming an $n \times n$ matrix!
:::

::: {.col .fig}
@fig:ch10-attn-win-associativity
:::
:::
:::

::: {.slide title="Causal Linear Attention as a Fast-Weight RNN"}
[Recurrent Matrix State]{.kicker}

::: {.cols .vc}
::: {.col}
- Under a causal mask ($s \le t$), factoring $\phi(\mathbf{q}_t)^\top$ out of the sums turns causal linear attention into an exact recurrence:
  $$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t + \epsilon}, \qquad \mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top$$
- **Constant Generation State:** $(\mathbf{S}_t, \mathbf{z}_t)$ has fixed size $d_h d_v + d_h$ ($16.25\text{ KiB}$ in `float32` at $d_h = d_v = 64$) for any sequence length $n$!
:::

::: {.col .fig}
@fig:ch10-attn-win-fast-weight-rnn
:::
:::
:::

::: {.slide title="Parallel and Recurrent Linear Attention in Code"}
[Two Equivalent Evaluations]{.kicker}

Because the state recurrence $\mathbf{S}_t = \sum_{s \le t} \phi(\mathbf{k}_s)\mathbf{v}_s^\top$ is a prefix sum, we can evaluate it in parallel via `cumsum` or sequentially step by step. The parallel (training) form:

@-attention-at-scale-kernelizing-the-score-1[1-10]
:::

::: {.slide title="Parallel and Recurrent Linear Attention in Code (cont.)"}
[Two Equivalent Evaluations]{.kicker}

The recurrent (generation) form carries a $(d_h, d_v)$ state and updates it token by token:

@-attention-at-scale-kernelizing-the-score-1[12-]
:::

::: {.slide title="Chunked Linear Attention: Hardware-Friendly Training"}
[Chunkwise Parallel Form]{.kicker}

::: {.cols .vc}
::: {.col}
- **Chunked Decomposition (Yang et al., 2024):** Split the sequence into $n/c$ chunks of length $c$ ($c = 64$); for a chunk starting at $t_0$ with pre-chunk state $(\mathbf{S}, \mathbf{z})$:
  $$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S} + \sum_{t_0 \le s \le t} \big(\phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\big)\,\mathbf{v}_s^\top}{\phi(\mathbf{q}_t)^\top \mathbf{z} + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s) + \epsilon}$$
- Runs both inter-chunk ($\mathbf{Q}_p \mathbf{S}$) and intra-chunk ($\mathrm{tril}(\mathbf{Q}_p \mathbf{K}_p^\top)\mathbf{V}_c$) terms as dense Tensor Core GEMMs while storing only $n/c$ chunk states!
:::

::: {.col .fig}
@fig:ch10-attn-win-chunked-gemm
:::
:::
:::

::: {.slide title="Implementing and Verifying Chunked Linear Attention"}
[Chunked Code & Equivalence]{.kicker}

`linear_attention_chunked` matches both other forms to `float32` precision ($\sim 10^{-7}$):

@attention-at-scale-chunked-computation-1

@!attention-at-scale-kernelizing-the-score-2
:::

::: {.slide title="Three Execution Modes of Linear Attention Compared"}
[Systems Comparison]{.kicker}

::: {.cols .vc}
::: {.col}
- **Parallel (`cumsum`):** Materializes all $n$ state matrices ($256\text{ MiB}$ at $n = 16\text{K}$); useful only for short debug sequences.
- **Recurrent Loop:** Constant $16\text{ KiB}$ state ($O(1)$ per step); ideal for token-by-token generation.
- **Chunked ($c = 64$):** Combines Tensor Core GEMMs with $64\times$ smaller state memory for fast parallel training and prefill.
:::

::: {.col .fig .big}
@fig:ch10-attn-win-three-modes
:::
:::
:::

::: {.slide title="Limits of Linear Attention: Capacity and Sharpness"}
[Fundamental Limitations]{.kicker}

Compressing an unbounded history of key–value pairs into a fixed $d_h \times d_v$ matrix with a finite feature map $\phi$ incurs two fundamental retrieval penalties:

::: {.cols .vc}
::: {.col}
- **1. Finite Orthogonal Capacity ($\le d_h$ Keys):** Because $\mathbf{S}_t = \sum_{s \le t} \phi(\mathbf{k}_s)\mathbf{v}_s^\top$ sums outer products in $\mathbb{R}^{d_h \times d_v}$, retrieving $\mathbf{v}_s$ without cross-talk requires $\phi(\mathbf{q}_t)$ to be orthogonal to $\phi(\mathbf{k}_{s'})$ for $s' \neq s$—possible for at most $d_h$ keys!
- **2. No Temperature Sharpening:** In softmax, scaling $\mathbf{q} \mapsto \beta \mathbf{q}$ exponentiates score gaps ($e^{\beta\Delta} \to \infty$), converging to a 1-hot lookup. For $\phi(x) = \mathrm{elu}(x) + 1$, large $\beta$ gives $\phi(\beta \mathbf{q}) \approx \beta\,\mathrm{relu}(\mathbf{q})$, and $\beta$ **cancels between numerator and denominator**!
:::

::: {.col .fig}
@!windowed-linear-attention-sharpening
:::
:::
:::

::: {.slide title="Engineering Guide: Choosing an Attention Mechanism"}
[Architectural Decision Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **Short/medium contexts ($n \le 32\text{K}$):** Exact fused FlashAttention ($\Theta(n^2 d)$ time, full 1-hop recall).
- **Long contexts ($n = 128\text{K}\text{--}1\text{M}$):** Interleaved sliding window + periodic global layers (e.g., Gemma 3 $5{:}1$ schedule cuts KV cache to $17\%$).
- **Constant-state streaming:** Gated linear attention / modern RNNs with input-dependent decay gates $\gamma_t$ + hybrid softmax layers.
:::

::: {.col .fig .big}
@fig:ch10-attn-win-decision-guide
:::
:::
:::

::: {.slide title="Concept Check: Tiles, Caches, and Linear Kernels"}
[Check Your Understanding]{.kicker}

1. **Sub-Tile Window Width:** In a fused kernel with tile size $b = 128$ at sequence length $n = 2{,}048$ ($T = 16$ tile rows), a causal window of width $w = 128$ visits $2T - 1 = 31$ tiles. How many $128 \times 128$ tiles does the kernel visit if you shrink the window to $w = 32$?
2. **ReLU Feature Map Scaling:** If you use $\phi(x) = \mathrm{relu}(x)$ in linear attention and multiply a query vector $\mathbf{q}_t$ by $\beta = 16$ (assuming $\phi(\mathbf{q}_t)^\top \mathbf{z}_t > 0$), by how much do the attention weights change?
3. **Degree-2 Taylor Feature Map (Based):** Expanding $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}) \approx 1 + \frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_h}} + \frac{(\mathbf{q}^\top\mathbf{k})^2}{2d_h}$ factorizes into a feature map of dimension $D = 1 + d_h + d_h^2$. For $d_h = 64$, how large is the state $\mathbf{S}_t \in \mathbb{R}^{D \times d_v}$ ($d_v = 64$), and at what sequence length $n$ does a softmax KV cache exceed it?
:::

::: {.slide title="Concept Check: Tiles, Caches, and Linear Kernels (answers)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. **Still $31$ tiles!** Because a kernel skips only completely empty $128 \times 128$ tiles, any window $2 \le w \le 128$ crosses the boundary into the left-neighbor tile and visits the exact same $2T - 1 = 31$ tiles.
2. **Zero change.** Because $\mathrm{relu}(16x) = 16\,\mathrm{relu}(x)$ is positively homogeneous of degree 1, the factor $16$ cancels identically between numerator and denominator.
3. **$D = 1 + 64 + 4{,}096 = 4{,}161$**, so $\mathbf{S}_t$ holds $4{,}161 \times 64$ floats ($1.02\text{ MiB}$ in fp32), matching a softmax KV cache ($2 \times 64$ floats/token) at **$n = 2{,}081$ tokens**.
:::
:::

::: {.slide title="Summary: Windowed and Linear Attention"}
[Key Takeaways]{.kicker}

| Concept / Variant | Mathematical Rule | Time & Memory Cost | Key Trade-Off / Engineering Lesson |
| :--- | :--- | :--- | :--- |
| **Causal Sliding Window** | $j \le i$ and $i - j < w$ | $\Theta(nwd)$ FLOPs, $O(wd)$ cache | $L$ layers reach $1 + L(w-1)$ tokens via multi-hop relays |
| **Hybrid Window + Global** | $r$ window layers per $1$ global layer | Cache fraction $\to \dfrac{1}{r+1}$ for $n \gg w$ | Preserves 1-hop long-range retrieval at a fraction of the KV cache |
| **Tiled Block Sparsity** | Skip $b \times b$ tile iff all $b^2$ entries `False` | Window $w=b$ visits $2T-1$ vs. $T(T+1)/2$ tiles | $> 30\times$ speedup at $n=32\text{K}$; align $w$ and packed docs to tile size $b$ |
| **Linear Attention** | $\mathbf{o}_t^\top = \dfrac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t + \epsilon}$ | $\Theta(n d_h d_v)$ FLOPs, **$O(d_h d_v)$ state** | Associativity $(\phi(\mathbf{Q})\phi(\mathbf{K})^\top)\mathbf{V} = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{V})$ removes $n^2$ |
| **Chunked Linear Form** | Intra-chunk $c \times c$ + inter-chunk $\mathbf{Q}_p\mathbf{S}$ | $\Theta(nc d_h + n d_h d_v)$ FLOPs | Runs on GPU Tensor Cores without storing $n$ full state matrices |
| **Capacity & Sharpness** | $\operatorname{rank}(\mathbf{S}_t) \le d_h$; $\beta$ cancels for $\mathrm{elu}+1$ | Stores at most $d_h$ orthogonal keys | Requires data-dependent decay gates $\gamma_t$ and hybrid softmax layers |
:::
