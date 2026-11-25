# The KV Cache and Cache Reduction
:label:`sec_kv-cache`

Look closely at what `GPT.generate` (:numref:`subsec_gpt-sampling`) does at decoding step $t$: to sample token $t+1$, it runs a full forward pass over the entire prefix $x_{1:t}$ of length $t$, reads only the last logit vector $\mathbf{z}_t \in \mathbb{R}^V$, appends $x_{t+1}$, and on step $t+1$ feeds $x_{1:t+1}$ through all $L$ blocks all over again. Because causal attention is lower-triangular, representation $s \le t$ at every layer depends only on tokens $1, \dots, s$, so recomputing the keys and values at positions $1, \dots, t-1$ on step $t$ reproduces the exact same tensors computed on step $t-1$. Across $T$ generated tokens, this redundant recomputation passes $O(T^2)$ tokens through the weight matrices where $T$ would do.

Storing past keys and values in a **key--value (KV) cache** (:numref:`fig_kv-cache`) eliminates this redundancy: at step $t$, each layer projects only the single new token $x_t$ into $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t)$, writes $(\mathbf{k}_t, \mathbf{v}_t)$ into slot $t$ of a pre-allocated buffer, and attends $\mathbf{q}_t$ against the cached prefix $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$.

Eliminating redundant FLOPs, however, exposes a hardware bottleneck: **memory capacity and memory bandwidth**. During single-token decoding, every generated token must stream both the model weights and the entire KV cache from GPU memory (HBM) to perform a handful of matrix--vector products. At long contexts or large batches, the KV cache grows to tens of gigabytes and caps decoding throughput. The first half of this section builds the cached forward pass and analyzes **prefill**, **decode**, and **batching** on the hardware roofline. The second half (:numref:`sec_kv-cache-reduction`) compresses the KV cache along its three dimensions: head count (**GQA/MQA**), channel width (**low-rank key--value projections**), and sequence length (**sliding windows with attention sinks**).

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #kv-cache-generation-and-the-kv-cache-1}
%%tab jax
from functools import partial
```

## From Recomputation to Caching

![One decoding step at position $t$. Naive generation recomputes every key, value, and score row of the prefix and then uses only the last row. With a KV cache, the past keys and values are read from memory, and the step computes a single new key, value, and score row.](../img/mdl-transformers-kv-cache.svg)
:label:`fig_kv-cache`

### The Cached Forward Pass
:label:`subsec_kv-cache-cached-forward`

Why do we cache keys and values $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$ at each block rather than queries $\mathbf{Q}_{1:t}$ or residual states $\mathbf{H}_{1:t}$? Under causal masking, query $\mathbf{q}_s$ ($s < t$) is used only when emitting token $s+1$ and is never referenced again. And caching the *post-RoPE* keys $\mathbf{k}_s$ and values $\mathbf{v}_s$ (each of shape $(B, H_{\textrm{kv}}, 1, d_h)$) means past tokens never need to be re-projected by $\mathbf{W}_k, \mathbf{W}_v$ or re-rotated by RoPE.

`init_cache` pre-allocates zeroed key and value buffers `ks, vs` of shape $(B, H_{\textrm{kv}}, T_{\max}, d_h)$ for each of the $L$ blocks. :numref:`fig_mdl-transformers-kv-buffer` shows one such buffer in the middle of generation: the slots before position $t$ are filled, the current step writes slot $t$, and the remaining slots are still zero and hidden by the mask.

![The cache buffer of one block during the step at position $t$ (one sequence, one key--value head). Slots $0, \dots, t-1$ hold the keys and values of earlier steps. The step writes $\mathbf{k}_t$ and $\mathbf{v}_t$ into slot $t$. The slots after $t$ are still zero and the mask hides them, so query $t$ attends to slots $0, \dots, t$.](../img/mdl-transformers-kv-buffer.svg)
:label:`fig_mdl-transformers-kv-buffer`

`forward_cached(model, X, ks, vs, t)` processes a slice `X` of $n$ new tokens starting at position offset `t` (passing `offset=t` to RoPE or indexing `pos_emb[t:t+n]`), writes their keys and values into `[:, :, t:t+n, :]`, and attends against `[:, :, :t+n, :]`:

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab pytorch
def init_cache(model, batch_size, max_len):
    """Zeroed keys and values: for every block one buffer of each, of
    shape (batch, kv heads, max_len, head width)."""
    attn = model.blks[0].attention
    shape = (batch_size, attn.num_kv_heads, max_len,
             attn.kv_hiddens // attn.num_kv_heads)
    p = next(model.parameters())
    cache = [torch.zeros(shape, device=p.device, dtype=p.dtype)
             for _ in range(2 * len(model.blks))]
    for u in cache:  # Fixed addresses, which a compiled step writes into
        torch._dynamo.mark_static_address(u)
    return cache[::2], cache[1::2]

def forward_cached(model, X, ks, vs, t):
    """Logits after the last of the new tokens X, which sit at positions
    t, t + 1, ...; writes their keys and values into the cache."""
    assert all(blk.pre_norm for blk in model.blks), 'pre-norm blocks only'
    T = X.shape[1]
    pos = t + torch.arange(T, device=X.device)  # t: an int or a tensor
    H, tables = model.token_emb(X), None
    if model.pos == 'learned':
        H = H + model.pos_emb(pos)
    else:  # RoPE angles of positions t, ..., t + T - 1, for every block
        tables = d2l.rope_tables(T, model.head_width, t, device=X.device)
    # Query t + i sees the filled positions 0, ..., t + i of the buffer
    mask = torch.arange(ks[0].shape[2], device=X.device) <= pos[:, None]
    for i, blk in enumerate(model.blks):  # Pre-norm, as in d2l.GPT
        attn = blk.attention
        q, k, v = attn.qkv(blk.norm1(H), t, tables)  # Rotated at t, ...
        ks[i][:, :, pos], vs[i][:, :, pos] = k, v    # Written in place
        # Attention over the whole buffer; each group of query heads
        # reads one key-value head: (batch, kv heads, group, T, max_len)
        q = q.unflatten(1, (attn.num_kv_heads, -1))
        S = q @ ks[i][:, :, None].transpose(-1, -2) / math.sqrt(q.shape[-1])
        Y = d2l.masked_softmax(S, mask) @ vs[i][:, :, None]
        H = H + attn.W_o(Y.flatten(1, 2).transpose(1, 2).flatten(2))
        H = H + blk.ffn(blk.norm2(H))
    return F.linear(model.norm(H[:, -1]), model.token_emb.weight), ks, vs
```

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab jax
def init_cache(model, batch_size, max_len):
    """Zeroed keys and values: for every block one buffer of each, of
    shape (batch, max_len, kv heads, head width)."""
    attn = model.blks[0].attention
    shape = (batch_size, max_len, attn.num_kv_heads,
             attn.kv_hiddens // attn.num_kv_heads)
    dtype = model.token_emb.embedding[...].dtype
    cache = [jnp.zeros(shape, dtype) for _ in range(2 * len(model.blks))]
    return cache[::2], cache[1::2]

def forward_cached(model, X, ks, vs, t):
    """Logits after the last of the new tokens X, which sit at positions
    t, t + 1, ...; writes their keys and values into the cache."""
    assert all(blk.pre_norm for blk in model.blks), 'pre-norm blocks only'
    B, T = X.shape
    pos = t + jnp.arange(T)  # t: an int or a traced value
    H, tables = model.token_emb(X), None
    if model.pos == 'learned':
        H = H + model.pos_emb(pos)
    else:  # RoPE angles of positions t, ..., t + T - 1, for every block
        tables = d2l.rope_tables(T, model.head_width, t)
    # Query t + i sees the filled positions 0, ..., t + i of the buffer
    mask = jnp.arange(ks[0].shape[1]) <= pos[:, None]
    ks, vs = list(ks), list(vs)
    for i, blk in enumerate(model.blks):  # Pre-norm, as in d2l.GPT
        attn = blk.attention
        q, k, v = (u.swapaxes(1, 2)  # (batch, T, heads, head width)
                   for u in attn.qkv(blk.norm1(H), t, tables))
        # A write past the end of the buffer would not fail: XLA moves the
        # start back, so the caller keeps t + T within max_len
        ks[i] = jax.lax.dynamic_update_slice(ks[i], k, (0, t, 0, 0))
        vs[i] = jax.lax.dynamic_update_slice(vs[i], v, (0, t, 0, 0))
        Y = jax.nn.dot_product_attention(q, ks[i], vs[i], mask=mask)
        H = H + attn.W_o(Y.reshape(B, T, -1))
        H = H + blk.ffn(blk.norm2(H))
    return model.token_emb.attend(model.norm(H[:, -1])), ks, vs
```

Using `forward_cached`, `generate_cached` splits generation into two stages: (1) **prefill**, which processes the entire prompt of length $T_p$ in one parallel call (`t = 0, n = T_p`), and (2) **decode**, which processes one newly sampled token per step (`n = 1` at offset `t = T_p, T_p+1, ...`):

```{.python .input #kv-cache-the-cached-forward-pass-2}
%%tab pytorch
def sample(logits, temperature=1.0, top_k=None):
    """One token per row of logits (batch, vocab), as generate draws it."""
    logits = logits / temperature
    if top_k is not None:
        cutoff = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    return torch.multinomial(F.softmax(logits, -1), 1)

# One-token steps of fixed shape, compiled once and replayed as CUDA graphs
# (mode='reduce-overhead' sets the first option; the second turns off a
# one-pass softmax that does not apply to a single query over a long cache)
decode_step = torch.compile(forward_cached, dynamic=True, options={
    'triton.cudagraphs': True, 'online_softmax': False})

@torch.inference_mode()
def decode(model, logits, ks, vs, t, num_tokens, temperature=1.0,
           top_k=None):
    """Sample num_tokens tokens per sequence, the first from logits and
    each next one after a step at positions t, t + 1, ...; returns them as
    (batch, num_tokens), and the cache."""
    pos = torch.arange(t, t + num_tokens, device=logits.device)
    tokens = [sample(logits, temperature, top_k)]
    for i in range(num_tokens - 1):  # No step after the last token
        logits, ks, vs = decode_step(model, tokens[-1], ks, vs, pos[i])
        tokens.append(sample(logits, temperature, top_k))
    return torch.cat(tokens, 1), ks, vs

@d2l.add_to_class(d2l.GPT)
def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample as generate does, but never recompute the prefix."""
    assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
    X = torch.tensor(prefix, device=next(self.parameters()).device)[None]
    ks, vs = init_cache(self, 1, len(prefix) + num_tokens)
    with torch.inference_mode():  # Prefill: the whole prompt in one call
        logits, ks, vs = forward_cached(self, X, ks, vs, 0)
    tokens = decode(self, logits, ks, vs, len(prefix), num_tokens,
                    temperature, top_k)[0]
    return list(prefix) + tokens[0].tolist()
```

```{.python .input #kv-cache-the-cached-forward-pass-2}
%%tab jax
def sample(logits, key, temperature=1.0, top_k=None):
    """One token per row of logits (batch, vocab), as generate draws it."""
    logits = logits / temperature
    if top_k is not None:
        cutoff = jax.lax.top_k(logits, top_k)[0][:, -1:]
        logits = jnp.where(logits < cutoff, -jnp.inf, logits)
    return jax.random.categorical(key, logits)[:, None]

@partial(jax.jit, static_argnames=('graphdef', 'num_tokens', 'top_k'),
         donate_argnames=('ks', 'vs'))
def decode(graphdef, params, X, ks, vs, t, num_tokens, key,
           temperature=1.0, top_k=None):
    """Feed X at positions t, t + 1, ..., then sample num_tokens tokens
    per sequence; returns them as (batch, num_tokens), and the cache."""
    model = nnx.merge(graphdef, params)
    logits, ks, vs = forward_cached(model, X, ks, vs, t)  # Prefill
    first = sample(logits, jax.random.fold_in(key, 0), temperature, top_k)
    tokens = jnp.zeros((X.shape[0], num_tokens), jnp.int32).at[:, 0].set(
        first[:, 0])
    def step(i, carry):  # Feed token i - 1, sample token i
        ks, vs, tokens = carry
        logits, ks, vs = forward_cached(model, tokens[:, i - 1, None], ks, vs,
                                        t + X.shape[1] + i - 1)
        token = sample(logits, jax.random.fold_in(key, i), temperature, top_k)
        return ks, vs, tokens.at[:, i].set(token[:, 0])
    ks, vs, tokens = jax.lax.fori_loop(1, num_tokens, step, (ks, vs, tokens))
    return tokens, ks, vs

@d2l.add_to_class(d2l.GPT)
def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample as generate does, but never recompute the prefix."""
    assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
    ks, vs = init_cache(self, 1, len(prefix) + num_tokens)
    tokens = decode(*nnx.split(self), jnp.asarray(prefix)[None], ks, vs, 0,
                    num_tokens, d2l.get_key(), temperature, top_k)[0]
    return list(prefix) + jax.device_get(tokens)[0].tolist()
```

### Checking Exact Equivalence
:label:`subsec_kv-cache-same-logits`

Before timing `generate_cached`, we verify that prefilling a prompt and stepping one token at a time through the cache produces the **exact same logits** (to within floating-point rounding $\sim 10^{-6}$) as running `model(X)` on the full sequence at once, for both `pos='rope'` and `pos='learned'`:

```{.python .input #kv-cache-checking-the-cache-2}
%%tab pytorch
gpt = d2l.GPT
```

```{.python .input #kv-cache-checking-the-cache-2}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

We now instantiate a 3-block `GPT` under both rotary (`pos='rope'`) and learned absolute (`pos='learned'`) positional embeddings, prefill the first $16$ tokens of a $48$-token sequence in one call, step through the remaining $32$ positions one token at a time, and measure the maximum absolute discrepancy against the full uncached forward pass `model(x)`:

```{.python .input #kv-cache-same-logits-measured-1}
%%tab pytorch
for pos in ('rope', 'learned'):
    model = gpt(vocab_size=97, num_hiddens=128, num_heads=4, num_blks=3,
                pos=pos).eval()
    x = torch.randint(0, 97, (1, 48))
    ks, vs = init_cache(model, 1, 48)
    with torch.inference_mode():
        full = model(x)[0, 15:]  # Logits after positions 15, ..., 47
        logits, ks, vs = forward_cached(model, x[:, :16], ks, vs, 0)
        outs = [logits]
        for t in range(16, 48):
            logits, ks, vs = forward_cached(model, x[:, t:t + 1], ks, vs, t)
            outs.append(logits)
    err = (torch.cat(outs) - full).abs().max()
    print(f'{pos}: max |full - cached| = {err:.1e}')
```

```{.python .input #kv-cache-same-logits-measured-1}
%%tab jax
with jax.default_matmul_precision('highest'):
    for pos in ('rope', 'learned'):
        model = gpt(vocab_size=97, num_hiddens=128, num_heads=4,
                    num_blks=3, pos=pos)
        model.eval()
        x = jax.random.randint(d2l.get_key(), (1, 48), 0, 97)
        full = model(x)[0, 15:]  # Logits after positions 15, ..., 47
        ks, vs = init_cache(model, 1, 48)
        logits, ks, vs = forward_cached(model, x[:, :16], ks, vs, 0)
        outs = [logits]
        for t in range(16, 48):
            logits, ks, vs = forward_cached(model, x[:, t:t + 1], ks, vs, t)
            outs.append(logits)
        err = jnp.abs(jnp.concatenate(outs) - full).max()
        print(f'{pos}: max |full - cached| = {err:.1e}')
```

### Measuring the Quadratic-to-Linear Speedup
:label:`subsec_kv-cache-speedup`

On a 12-layer GPT-2-sized model ($d = 768, H = 12, L = 12$), comparing the wall-clock time of one uncached forward pass over $n$ tokens (`time_forward`, which naive decoding repeats at *every* step) against one cached decode step at context length $n$ (`time_decode`) shows why caching is mandatory for long-context generation:

```{.python .input #kv-cache-same-logits-measured-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32
device = d2l.try_gpu()
model = gpt(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
            max_len=4096).to(device).eval()  # GPT-2's vocabulary and size
print(f'{d2l.num_params(model) / 1e6:.0f}M parameters')
```

```{.python .input #kv-cache-same-logits-measured-2}
%%tab jax
model = gpt(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
            max_len=4096)  # GPT-2's vocabulary and size
model.eval()
print(f'{d2l.num_params(model) / 1e6:.0f}M parameters')
```

Next, we define two timing helpers using `d2l.Benchmark`: `time_forward(model, n)` measures a compiled full-prefix forward pass over $n$ tokens, while `time_decode(model, batch_size, n)` averages the wall-clock latency of $32$ single-token cached decode steps starting at context length $n$:

```{.python .input #kv-cache-checking-the-cache-3}
%%tab pytorch
def time_forward(model, n):
    """Seconds for a compiled forward pass over n random tokens."""
    device = next(model.parameters()).device
    x = torch.randint(0, model.token_emb.num_embeddings, (1, n),
                      device=device)
    forward = torch.compile(model)
    with torch.inference_mode():
        return d2l.Benchmark(lambda: forward(x)[:, -1]).time

def time_decode(model, batch_size, n):
    """Seconds per decode step for batch_size sequences at context n,
    averaged over 32 steps."""
    device = next(model.parameters()).device
    logits = torch.zeros(batch_size, model.token_emb.num_embeddings,
                         device=device)  # Uniform: random first tokens
    ks, vs = init_cache(model, batch_size, n + 32)
    return d2l.Benchmark(
        lambda: decode(model, logits, ks, vs, n, 33)).time / 32
```

```{.python .input #kv-cache-checking-the-cache-3}
%%tab jax
@partial(jax.jit, static_argnames='graphdef')
def last_logits(graphdef, params, X):
    """The step of naive generation: a forward pass over all of X."""
    return nnx.merge(graphdef, params)(X)[:, -1]

def time_forward(model, n):
    """Seconds for a forward pass over n random tokens."""
    graphdef, params = nnx.split(model)
    x = jax.random.randint(d2l.get_key(), (1, n), 0,
                           model.token_emb.num_embeddings)
    return d2l.Benchmark(lambda: last_logits(graphdef, params, x)).time

def time_decode(model, batch_size, n):
    """Seconds per decode step for batch_size sequences at context n,
    averaged over 32 steps."""
    graphdef, params = nnx.split(model)
    x = jax.random.randint(d2l.get_key(), (batch_size, 1), 0,
                           model.token_emb.num_embeddings)
    cache = init_cache(model, batch_size, n + 32)
    def run():
        nonlocal cache  # decode consumes the cache and returns it updated
        tokens, *cache = decode(graphdef, params, x, *cache, n - 1, 32,
                                d2l.get_key())
        return tokens
    return d2l.Benchmark(run).time / 32
```

Sweeping the context length $n$ across powers of two from $256$ to $4{,}032$ tokens contrasts the linear growth of each uncached step against the nearly flat latency of a cached decode step:

```{.python .input #kv-cache-checking-the-cache-4}
%%tab pytorch, jax
lengths, t_naive, t_cached = [256, 512, 1024, 2048, 4032], [], []
for n in lengths:
    t_naive.append(time_forward(model, n))
    t_cached.append(time_decode(model, 1, n))
    print(f'n={n:4d}: naive {1e3 * t_naive[-1]:5.1f} ms, '
          f'cached {1e3 * t_cached[-1]:4.1f} ms per token')
```

Plotting both per-token latency curves on log--log axes shows a naive step whose cost grows in proportion to the context length $n$ and a cached step that is nearly flat (the new query is still scored against all $n$ cached keys, but this cost has a small constant):

```{.python .input #kv-cache-checking-the-cache-1}
%%tab pytorch, jax
d2l.plot(lengths, [[1e3 * t for t in t_naive], [1e3 * t for t in t_cached]],
         'context length', 'ms per token', legend=['naive', 'cached'],
         xscale='log', yscale='log')
```

Cumulated over an end-to-end generation of $256$ new tokens after a $3{,}584$-token prompt, replacing `model.generate` with `model.generate_cached` yields more than an order-of-magnitude wall-clock speedup:

```{.python .input #kv-cache-same-logits-measured-3}
%%tab pytorch, jax
prefix = list(range(3584))  # Any token ids will do for timing
for name, gen in (('naive ', model.generate),
                  ('cached', model.generate_cached)):
    t = d2l.Benchmark(lambda: gen(prefix, 256), warmup=1, repeats=1).time
    print(f'{name}: 256 tokens in {t:4.1f} s = {256 / t:5.1f} tokens/s')
```

## Memory Capacity and the Hardware Roofline
:label:`subsec_kv-cache-memory`

For a model with $L$ blocks, batch size $B$, sequence length $n$, and $H_{\textrm{kv}}$ key--value heads of width $d_h$ stored at $b$ bytes per number ($b = 4$ for FP32, $2$ for BF16/FP16, $1$ for FP8), each block stores two tensors ($\mathbf{K}$ and $\mathbf{V}$) of shape $(B, H_{\textrm{kv}}, n, d_h)$. The total KV cache footprint is therefore

$$
M_{\textrm{KV}}(n, B) \;=\; 2 \, b \, B \, L \, n \, H_{\textrm{kv}} \, d_h \;\;\textrm{bytes}.
$$
:eqlabel:`eq_kv-cache-bytes`

```{.python .input #kv-cache-memory-required-by-the-kv-cache-1}
%%tab pytorch, jax
#@save
def kv_cache_bytes(model, num_tokens, batch_size=1, bytes_per_elem=4):
    """Bytes of the keys and values of num_tokens positions in every block
    of a GPT, for batch_size sequences."""
    attn = model.blks[0].attention
    return (2 * len(model.blks) * attn.kv_hiddens * num_tokens * batch_size
            * bytes_per_elem)
```

Evaluating `kv_cache_bytes` on our 124M-parameter FP32 model at $n = 4{,}096$ tokens confirms that the allocated cache buffers match :eqref:`eq_kv-cache-bytes` ($288\textrm{ MiB}$) and already exceed half the $471\textrm{ MiB}$ footprint of the model weights:

```{.python .input #kv-cache-the-memory-bill}
%%tab pytorch, jax
bytes_per_param = 4  # fp32 weights
weight_bytes = bytes_per_param * d2l.num_params(model)
ks, vs = init_cache(model, 1, 4096)
print(f'cache buffers {sum(u.nbytes for u in ks + vs) / 2**20:.0f} MiB, '
      f'formula {kv_cache_bytes(model, 4096) / 2**20:.0f} MiB, '
      f'weights {weight_bytes / 2**20:.0f} MiB')
```

### Prefill vs. Decode on the Roofline
:label:`subsec_kv-cache-prefill-decode`

Why did a 2,048-token prefill take nearly the same order of magnitude of time per call as some fifteen single-token decode steps? The **roofline model** (:numref:`fig_kv-cache-roofline`) explains the difference via **arithmetic intensity** (FLOPs performed per byte transferred between HBM and the compute units):

- **Prefill ($n = 2048$ tokens in parallel, compute-bound):** Each weight matrix $\mathbf{W}$ is read from HBM once and multiplied by $B n$ token vectors ($2 B n N_{\textrm{params}}$ FLOPs across $b_w N_{\textrm{params}}$ weight bytes), giving an arithmetic intensity of $\sim 2 B n / b_w \gg 80\textrm{ FLOPs/byte}$—well above the GPU's ridge point, saturating the tensor cores.
- **Single-stream decode ($B = 1, n = 1$ token per step, memory-bandwidth bound):** Each step performs only $2 N_{\textrm{params}}$ matrix--vector FLOPs while streaming **all** $b_w N_{\textrm{params}}$ weight bytes plus $M_{\textrm{KV}}(n, 1)$ cache bytes from HBM, giving an arithmetic intensity of $\le 2 / b_w \approx 0.5\textrm{ FLOPs/byte}$ (in FP32, or $1.0\textrm{ FLOP/byte}$ in BF16)—more than $100\times$ below the compute roof.

![The fp32 roofline of an RTX 4090. Attainable throughput is the smaller of the compute roof, about 83 TFLOP/s, and the arithmetic intensity times the memory bandwidth of about 1 TB/s; the two meet at the ridge point of about 82 FLOPs per byte. A decode step of a GPT-2-sized model at a context of 2,048 tokens performs 0.38 FLOPs per byte and sits far down the bandwidth slope; a prefill of 2,048 tokens, at about 780 counting weights and cache only, sits under the roof.](../img/mdl-transformers-roofline.svg)
:label:`fig_kv-cache-roofline`

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch, jax
t_prefill = time_forward(model, 2048)
t_decode = time_decode(model, 1, 2048)
```

Comparing the resulting token throughputs and arithmetic intensities against a $1\textrm{ TB/s}$ HBM bandwidth budget shows why single-stream decode sits near the memory-bandwidth ceiling while a $2{,}048$-token prefill achieves more than $100\times$ higher throughput:

```{.python .input #kv-cache-prefill-and-decode-1}
%%tab pytorch, jax
bandwidth = 1e12  # bytes/s, about an RTX 4090
flops = 2 * d2l.num_params(model)  # One decode step at context 2048
moved = weight_bytes + kv_cache_bytes(model, 2048)
print(f'prefill: {2048 / t_prefill:6.0f} tokens/s')
print(f'decode:  {1 / t_decode:6.0f} tokens/s, bandwidth ceiling '
      f'{bandwidth / moved:.0f} tokens/s')
print(f'FLOPs per byte: decode {flops / moved:.2f}, '
      f'prefill about {2048 * flops / moved:.0f}')
```

### Batched Decoding
:label:`subsec_kv-cache-batching`

Because a single-stream decode step ($B = 1$) spends its time streaming the model's weight bytes $M_{\textrm{weights}} = b_w N_{\textrm{params}}$ from HBM, decoding a batch of $B$ sequences simultaneously reuses each weight read across all $B$ tokens while reading $B$ distinct KV caches ($B \, M_{\textrm{KV}}(n, 1)$ bytes). At memory bandwidth $\mathcal{B}_{\textrm{HBM}}$ (bytes/s), the bandwidth-limited decode throughput (tokens/s) is

$$
\textrm{Throughput}(B, n) \;\le\; \frac{B \, \mathcal{B}_{\textrm{HBM}}}{M_{\textrm{weights}} + B \, M_{\textrm{KV}}(n, 1)}.
$$
:eqlabel:`eq_kv-cache-batch-throughput`

At small batch sizes ($B \, M_{\textrm{KV}} \ll M_{\textrm{weights}}$), throughput grows nearly linearly with $B$ ("free batching"); as $B$ or $n$ grows large ($B \, M_{\textrm{KV}} \gg M_{\textrm{weights}}$), KV cache reads dominate HBM traffic and throughput saturates at the **KV-bandwidth ceiling** $\mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$:

```{.python .input #kv-cache-batched-decoding}
%%tab pytorch, jax
batches, measured, bounds = [1, 4, 16, 64], [], []
for n in (128, 1024):
    measured.append([b / time_decode(model, b, n) for b in batches])
    nc = kv_cache_bytes(model, n)  # One sequence's cache
    bounds.append([b * bandwidth / (weight_bytes + b * nc) for b in batches])
    print(f'n={n}: ' + ', '.join(f'{r:.0f}' for r in measured[-1])
          + ' tokens/s')
d2l.plot(batches, measured + bounds, 'batch size', 'tokens per second',
         legend=['n=128', 'n=1024'], fmts=('C0-o', 'C1-s', 'C0:', 'C1:'),
         xscale='log', yscale='log', legend_loc='lower right')
```

## Shrinking the KV Cache
:label:`sec_kv-cache-reduction`

Because both memory capacity (:eqref:`eq_kv-cache-bytes`) and the large-batch decode throughput ceiling (:eqref:`eq_kv-cache-batch-throughput`) are governed by the per-sequence cache size $M_{\textrm{KV}}(n, 1) = 2 b L n H_{\textrm{kv}} d_h$, modern Transformer architectures compress the product $n \cdot H_{\textrm{kv}} \cdot d_h$ along three orthogonal axes:

1. **Fewer key--value heads ($H_{\textrm{kv}} < H$):** Grouped-query attention (GQA) :cite:`Ainslie.Lee-Thorp.Jong.ea.2023` and multi-query attention (MQA) :cite:`Shazeer.2019` share each key--value head across $G = H / H_{\textrm{kv}}$ query heads (:numref:`fig_gqa`).
2. **Narrower per-token representations ($2 H_{\textrm{kv}} d_h \to r$):** Low-rank key--value projections and Multi-Head Latent Attention (MLA) :cite:`Liu.Feng.Xue.ea.2024` compress all heads' keys and values into a low-dimensional latent vector (:numref:`subsec_kv-cache-low-rank`).
3. **Shorter cached context ($n \to w + s$):** Sliding-window attention evicts tokens older than a window $w$, while keeping $s \ge 1$ initial **attention sink** tokens :cite:`Xiao.Tian.Chen.ea.2024` so the softmax denominator remains stable (:numref:`subsec_kv-cache-sinks`).

![Query heads and the key--value heads they read. Multi-head attention gives every query head its own key--value head; grouped-query attention shares one per group of query heads; multi-query attention shares a single one across all of them. The cache scales with the number of key--value heads.](../img/mdl-transformers-gqa.svg)
:label:`fig_gqa`

### Axis 1: Grouped-Query Attention and Checkpoint Uptraining
:label:`subsec_kv-cache-gqa-quality`

Our `CausalAttention` class (:numref:`sec_gpt`) already accepts `num_kv_heads`: setting $H_{\textrm{kv}} = H / G$ shrinks $\mathbf{W}_{qkv}$ and cuts $M_{\textrm{KV}}$ by a factor of $G = H / H_{\textrm{kv}}$. Moreover, :citet:`Ainslie.Lee-Thorp.Jong.ea.2023` showed that an existing multi-head checkpoint ($H_{\textrm{kv}} = H$) can be converted into a GQA or MQA model by **mean-pooling** each group of $G$ key and value projection heads (`pool_heads`) and **uptraining** for a small fraction ($\sim 5\text{--}10\%$) of the original training budget.

We compare training our 6-block byte-level `GPT` ($H = 8$) from scratch for 1,000 steps at $H_{\textrm{kv}} \in \{8, 2\}$ (and at $H_{\textrm{kv}} = 2$ for only 50 steps) against converting the trained $H_{\textrm{kv}} = 8$ checkpoint down to $H_{\textrm{kv}} = 2$, by mean-pooling and by keeping the first head of each group, both zero-shot (0 steps) and after 50 and 100 steps ($10\%$) of uptraining, in two runs:

```{.python .input #kv-cache-reduction-cache-size-and-validation-loss-1}
%%tab pytorch
def pool_heads(target, source, pool):
    """Load the weights of source into target, which has fewer key-value
    heads. Each key (and value) projection of target is the mean of those
    of a group of consecutive heads of source (pool='mean'), or the first
    of them (pool='first')."""
    src, tgt = source.blks[0].attention, target.blks[0].attention
    group = src.num_kv_heads // tgt.num_kv_heads
    state = source.state_dict()
    for name, W in state.items():
        if name.endswith('W_qkv.weight'):  # Rows: d queries, keys, values
            q, k, v = W.split([W.shape[1], src.kv_hiddens, src.kv_hiddens])
            k, v = (u.unflatten(0, (tgt.num_kv_heads, group, -1))
                    for u in (k, v))
            k, v = ((u.mean(1) if pool == 'mean' else u[:, 0]).flatten(0, 1)
                    for u in (k, v))
            state[name] = torch.cat([q, k, v])
    target.load_state_dict(state)
    return target.to(next(source.parameters()).device)
```

```{.python .input #kv-cache-reduction-cache-size-and-validation-loss-1}
%%tab jax
def pool_heads(target, source, pool):
    """Load the weights of source into target, which has fewer key-value
    heads. Each key (and value) projection of target is the mean of those
    of a group of consecutive heads of source (pool='mean'), or the first
    of them (pool='first')."""
    src, tgt = source.blks[0].attention, target.blks[0].attention
    group = src.num_kv_heads // tgt.num_kv_heads
    def convert(path, W):
        if 'W_qkv' not in jax.tree_util.keystr(path):
            return W
        d = W.shape[0]  # Kernel columns: d queries, keys, values
        q, k, v = jnp.split(W, [d, d + src.kv_hiddens], axis=1)
        k, v = (u.reshape(d, tgt.num_kv_heads, group, -1) for u in (k, v))
        k, v = ((u.mean(2) if pool == 'mean' else u[:, :, 0]).reshape(d, -1)
                for u in (k, v))
        return jnp.concatenate([q, k, v], axis=1)
    nnx.update(target, jax.tree_util.tree_map_with_path(
        convert, nnx.state(source, nnx.Param)))
    return target
```

We now train our 6-block byte-level `GPT` on `WikiText103` at $H_{\textrm{kv}} \in \{8, 2\}$ and evaluate `pool_heads` with both `'mean'` and `'first'` head pooling across $0$, $50$, and $100$ uptraining steps:

```{.python .input #kv-cache-cache-against-quality}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)

def trained(num_kv_heads, num_steps=1000):
    """A d2l.GPT trained from scratch with the recipe of the GPT section."""
    lm = gpt(len(data.vocab), num_kv_heads=num_kv_heads)
    optimizer = d2l.LMOptimizer(lm, lr=2e-3, num_steps=num_steps,
                                warmup=min(100, num_steps // 10))
    d2l.train_steps(lm, data, optimizer, num_steps)
    return lm

def val_loss(lm):
    return math.log(d2l.lm_perplexity(lm, data.val_dataloader()))

for run in (1, 2):
    mha = trained(8)
    print(f'run {run}, trained from scratch: H_kv=8 {val_loss(mha):.2f}, '
          f'H_kv=2 {val_loss(trained(2)):.2f}, '
          f'H_kv=2 for 50 steps only {val_loss(trained(2, 50)):.2f}')
    for pool in ('mean', 'first'):
        lm = pool_heads(gpt(len(data.vocab), num_kv_heads=2), mha, pool)
        losses = [val_loss(lm)]
        optimizer = d2l.LMOptimizer(lm, lr=3e-4, num_steps=10**9, warmup=0)
        for _ in range(2):  # Uptraining at a constant rate
            d2l.train_steps(lm, data, optimizer, 50)
            losses.append(val_loss(lm))
        print(f'  H_kv=8 converted by {pool:5s}: {losses[0]:.2f}, after 50 '
              f'steps {losses[1]:.2f}, after 100 steps {losses[2]:.2f}')
```

Sharing keys and values across 4 query heads ($H_{\textrm{kv}} = 2$, a $4\times$ smaller KV cache) costs no measurable quality when trained from scratch (1.34--1.35 against 1.34 nats/byte). Head pooling starts near 2.8 nats/byte, and 100 steps of uptraining bring the mean-pooled model to about 1.47, roughly 90% of the way back, while 50 steps from scratch reach only about 2.7 and keeping the first head of each group trails mean-pooling after uptraining.

### Axis 2: Low-Rank Key--Value Projections
:label:`subsec_kv-cache-low-rank`

Instead of reducing the number of heads $H_{\textrm{kv}}$, can we compress the $2 d$-dimensional concatenated key--value vector $[\mathbf{k}_t; \mathbf{v}_t] \in \mathbb{R}^{2d}$ of each token at layer $i$ into a rank-$r$ subspace ($r \ll 2d$)? We test this directly on pretrained **GPT-2 Small** (`124M`, where $d = 768$ and $2d = 1536$) on a passage of $T = 1024$ tokens of The Time Machine:

```{.python .input #kv-cache-compressing-the-cache-further-1}
%%tab pytorch
gpt2, enc = d2l.load_gpt2().to(device), d2l.gpt2_tokenizer()
fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', '../data',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')
tokens = torch.tensor(enc.encode(open(fname).read()), device=device)
x, calib = tokens[None, :1024], tokens[None, 1024:2048]
print(f'{len(tokens)} tokens of The Time Machine')
```

```{.python .input #kv-cache-compressing-the-cache-further-1}
%%tab jax
gpt2, enc = d2l.load_gpt2(), d2l.gpt2_tokenizer()
fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', '../data',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')
tokens = jnp.asarray(enc.encode(open(fname).read()))
x, calib = tokens[None, :1024], tokens[None, 1024:2048]
print(f'{len(tokens)} tokens of The Time Machine')
```

To intercept and transform the keys and values inside every layer without rewriting `GPT`, we define `forward_blocks(model, X, attn_fn)`, which delegates block $i$'s attention to a callback `attn_fn(q, k, v, i)`, and verify that standard causal attention matches `gpt2(x)` to within $10^{-5}$ while establishing our baseline `tail_loss` on positions $512 \dots 1023$:

```{.python .input #kv-cache-compressing-the-cache-further-2}
%%tab pytorch
def forward_blocks(model, X, attn_fn):
    """The forward pass of a d2l.GPT with attention computed by
    attn_fn(q, k, v, i) in block i, on inputs of shape (batch, heads,
    length, head width)."""
    assert all(blk.pre_norm for blk in model.blks), 'pre-norm blocks only'
    H = model.token_emb(X)
    if model.pos == 'learned':
        H = H + model.pos_emb(torch.arange(X.shape[1], device=X.device))
    for i, blk in enumerate(model.blks):
        attn = blk.attention
        Y = attn_fn(*attn.qkv(blk.norm1(H)), i)
        H = H + attn.W_o(Y.transpose(1, 2).flatten(2))
        H = H + blk.ffn(blk.norm2(H))
    return F.linear(model.norm(H), model.token_emb.weight)

def tail_loss(logits, X):
    """Mean loss of the next-token predictions from position 512 on."""
    return F.cross_entropy(logits[0, 512:-1], X[0, 513:])

causal = lambda q, k, v, i: F.scaled_dot_product_attention(
    q, k, v, is_causal=True)
with torch.inference_mode():
    logits = forward_blocks(gpt2, x, causal)
    print(f'helper vs model: max deviation '
          f'{(logits - gpt2(x)).abs().max():.1e}')
    full = tail_loss(logits, x)
print(f'full attention: loss {full:.2f}, perplexity {full.exp():.0f}')
```

```{.python .input #kv-cache-compressing-the-cache-further-2}
%%tab jax
def forward_blocks(model, X, attn_fn):
    """The forward pass of a d2l.GPT with attention computed by
    attn_fn(q, k, v, i) in block i, on inputs of shape (batch, heads,
    length, head width)."""
    assert all(blk.pre_norm for blk in model.blks), 'pre-norm blocks only'
    H = model.token_emb(X)
    if model.pos == 'learned':
        H = H + model.pos_emb(jnp.arange(X.shape[1]))
    for i, blk in enumerate(model.blks):
        attn = blk.attention
        Y = attn_fn(*attn.qkv(blk.norm1(H)), i)
        H = H + attn.W_o(Y.swapaxes(1, 2).reshape(*X.shape, -1))
        H = H + blk.ffn(blk.norm2(H))
    return model.token_emb.attend(model.norm(H))

def tail_loss(logits, X):
    """Mean loss of the next-token predictions from position 512 on."""
    return optax.softmax_cross_entropy_with_integer_labels(
        logits[0, 512:-1], X[0, 513:]).mean()

def causal(q, k, v, i):
    """Fused causal attention; the kernel wants (batch, length, ...)."""
    return jax.nn.dot_product_attention(
        *(u.swapaxes(1, 2) for u in (q, k, v)), is_causal=True).swapaxes(1, 2)

logits = forward_blocks(gpt2, x, causal)
print(f'helper vs model: max deviation '
      f'{jnp.abs(logits - gpt2(x)).max():.1e}')
full = tail_loss(logits, x)
print(f'full attention: loss {full:.2f}, perplexity {jnp.exp(full):.0f}')
```

Using the next 1,024 tokens of the book (`calib`) to fit an empirical PCA/SVD basis $\mathbf{U}_i \in \mathbb{R}^{2d \times r}$ for each layer's concatenated key--value rows (`fit_bases`) and projecting the keys and values of the evaluated passage `x` onto the top $r \in \{512, 256, 128\}$ directions (`projected`) measures how compressible GPT-2's KV cache is even without retraining:

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab pytorch
def kv_rows(k, v):
    """Keys and values of all heads of each token as one row: (T, 2d)."""
    return torch.cat([u[0].transpose(0, 1).flatten(1) for u in (k, v)], 1)

def fit_bases(model, X):
    """Right singular vectors of the key-value rows of X in every block."""
    bases = []
    def fit(q, k, v, i):
        bases.append(torch.linalg.svd(kv_rows(k, v), full_matrices=False)[2])
        return causal(q, k, v, i)
    forward_blocks(model, X, fit)
    return bases

def projected(bases, r):
    """Attention with keys and values projected onto the top r bases."""
    def attn_fn(q, k, v, i):
        P = bases[i][:r]  # (r, 2d), orthonormal rows
        KV = kv_rows(k, v) @ P.T @ P
        k, v = (u.unflatten(1, (k.shape[1], -1)).transpose(0, 1)[None]
                for u in KV.chunk(2, 1))
        return causal(q, k, v, i)
    return attn_fn

with torch.inference_mode():
    bases = fit_bases(gpt2, calib)
    for r in (512, 256, 128):
        loss = tail_loss(forward_blocks(gpt2, x, projected(bases, r)), x)
        print(f'rank {r} of 768: loss {loss:.2f}, '
              f'perplexity {loss.exp():.0f}')
```

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab jax
def kv_rows(k, v):
    """Keys and values of all heads of each token as one row: (T, 2d)."""
    return jnp.concatenate(
        [u[0].swapaxes(0, 1).reshape(u.shape[2], -1) for u in (k, v)], 1)

def fit_bases(model, X):
    """Right singular vectors of the key-value rows of X in every block."""
    bases = []
    def fit(q, k, v, i):
        bases.append(jnp.linalg.svd(kv_rows(k, v), full_matrices=False)[2])
        return causal(q, k, v, i)
    forward_blocks(model, X, fit)
    return bases

def projected(bases, r):
    """Attention with keys and values projected onto the top r bases."""
    def attn_fn(q, k, v, i):
        P = bases[i][:r]  # (r, 2d), orthonormal rows
        KV = kv_rows(k, v) @ P.T @ P
        k, v = (u.reshape(u.shape[0], k.shape[1], -1).swapaxes(0, 1)[None]
                for u in jnp.split(KV, 2, axis=1))
        return causal(q, k, v, i)
    return attn_fn

bases = fit_bases(gpt2, calib)
for r in (512, 256, 128):
    loss = tail_loss(forward_blocks(gpt2, x, projected(bases, r)), x)
    print(f'rank {r} of 768: loss {loss:.2f}, '
          f'perplexity {jnp.exp(loss):.0f}')
```

Because each block computes its $2d$ keys and values from one $d$-dimensional input with one matrix and one bias, they lie in an affine subspace of only $d = 768$ dimensions, the "of 768" in the output. Even post-hoc linear projection compresses GPT-2's KV vectors by $6\times$ ($r = 256$ out of $2d = 1536$) with no measurable degradation (3.61 against 3.58 nats), while $r = 128$ costs more than half a nat; when the low-rank bottleneck is trained end-to-end as in DeepSeek-V2/V3's **Multi-Head Latent Attention (MLA)** :cite:`Liu.Feng.Xue.ea.2024`, a single 512-dimensional latent vector replaces all heads' keys and values at $>16\times$ compression.

### Axis 3: Sliding Windows and Attention Sinks
:label:`subsec_kv-cache-sinks`

Our third axis bounds the sequence length $n$ stored in the cache to a rolling window of the $w$ most recent tokens (:numref:`sec_windowed-linear-attention`). However, if we apply a pure sliding window ($w = 256$) to pretrained GPT-2 on a 1,024-token passage, cross-entropy on the tail tokens ($t \ge 512$) jumps from $\sim 3.6$ nats to over $8$ nats.

To see why, `sink_mass` measures the average attention probability that queries at positions $t \ge 64$ assign to **token 0** (the very first token of the sequence) in each of GPT-2's 12 layers:

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab pytorch
def sink_mass(model, X):
    """Mean attention weight on token 0 of the queries from position 64
    on, one entry per block."""
    masses = []
    def watch(q, k, v, i):  # Explicit causal attention
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        alpha = d2l.masked_softmax(scores, d2l.causal_mask(
            q.shape[2], device=q.device))
        masses.append(alpha[..., 64:, 0].mean())
        return alpha @ v
    forward_blocks(model, X, watch)
    return masses

with torch.inference_mode():
    masses = sink_mass(gpt2, x)
print('mean attention weight on token 0 (uniform would be about 0.003)')
print('block ' + ''.join(f'{i:6d}' for i in range(len(masses))))
print('weight' + ''.join(f'{m:6.2f}' for m in torch.stack(masses).tolist()))
```

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab jax
def sink_mass(model, X):
    """Mean attention weight on token 0 of the queries from position 64
    on, one entry per block."""
    masses = []
    def watch(q, k, v, i):  # Explicit causal attention
        scores = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
        alpha = d2l.masked_softmax(scores, d2l.causal_mask(q.shape[2]))
        masses.append(alpha[..., 64:, 0].mean())
        return alpha @ v
    forward_blocks(model, X, watch)
    return masses

masses = sink_mass(gpt2, x)
print('mean attention weight on token 0 (uniform would be about 0.003)')
print('block ' + ''.join(f'{i:6d}' for i in range(len(masses))))
print('weight' + ''.join(f'{m:6.2f}' for m in jax.device_get(masses)))
```

In the middle and upper layers of GPT-2, queries place **30% to nearly 60% of their entire attention weight on token 0**, regardless of what word sits at position 0. Because softmax probabilities must sum to $1$, attention heads use the earliest token(s), the only positions visible to *every* causal query, as a default **attention sink** :cite:`Xiao.Tian.Chen.ea.2024` when no local token matches. Once a pure sliding window slides past position 0 ($t \ge w$), removing token 0 from the softmax denominator forces that $30\text{--}60\%$ probability mass onto the remaining window tokens and scrambles the residual stream.

Keeping just $s \in \{1, 4\}$ initial sink tokens fixed at the front of the cache alongside the $w = 256$ most recent tokens ($w + s$ cached tokens in total) restores GPT-2's validation loss almost completely to its full-context baseline:

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab pytorch
j = torch.arange(x.shape[1], device=device)
allowed = d2l.causal_mask(x.shape[1], device=device)
w = 256  # The window: each query sees itself and the 255 tokens before it
window = allowed & (j[:, None] - j[None, :] < w)
with torch.inference_mode():
    for nsink in (0, 1, 4):
        mask = window | (allowed & (j[None, :] < nsink))
        loss = tail_loss(forward_blocks(
            gpt2, x, lambda q, k, v, i: F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask)), x)
        print(f'window {w} + {nsink} sink tokens: loss {loss:.2f}, '
              f'perplexity {loss.exp():.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {full.exp():.0f}')
```

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab jax
j = jnp.arange(x.shape[1])
allowed = d2l.causal_mask(x.shape[1])
w = 256  # The window: each query sees itself and the 255 tokens before it
window = allowed & (j[:, None] - j[None, :] < w)
for nsink in (0, 1, 4):
    mask = window | (allowed & (j[None, :] < nsink))
    loss = tail_loss(forward_blocks(
        gpt2, x, lambda q, k, v, i: jax.nn.dot_product_attention(
            *(u.swapaxes(1, 2) for u in (q, k, v)),
            mask=mask).swapaxes(1, 2)), x)
    print(f'window {w} + {nsink} sink tokens: loss {loss:.2f}, '
          f'perplexity {jnp.exp(loss):.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {jnp.exp(full):.0f}')
```

## Comparison of Cache-Reduction Methods

:Summary of KV cache reduction mechanisms along the three factors of $M_{\textrm{KV}} = 2 b B L n H_{\textrm{kv}} d_h$.
:label:`tab_kv-cache-reduction`

| Method | Factor reduced | Stored numbers per sequence | Needs training? |
|:--|:--|:--|:--|
| **Shared KV heads (GQA, MQA)** :cite:`Shazeer.2019,Ainslie.Lee-Thorp.Jong.ea.2023` | Heads $H_{\textrm{kv}} < H$ | $2 L n H_{\textrm{kv}} d_h$ | Yes: train from scratch or uptrain ($\sim 5\text{--}10\%$) |
| **Low-rank latent (MLA)** :cite:`Liu.Feng.Xue.ea.2024` | Width $2 H_{\textrm{kv}} d_h \to r$ | $\approx L n r$ (plus a small rotary key) | Yes for MLA; a post-hoc SVD basis works to $\sim 6\times$ |
| **Sliding window with sinks** :cite:`Xiao.Tian.Chen.ea.2024` | Length $n \to w + s$ | $2 L (w + s) H_{\textrm{kv}} d_h$ | Works zero-shot on pretrained models when $s \ge 1$ |

## Summary

**KV caching** stores the post-RoPE keys and values of past tokens so each decoding step projects only the single new token (`n = 1`), reducing projection work across $T$ generated tokens from $O(T^2)$ to $O(T)$ token evaluations (and attention scoring from $O(T^3)$ to $O(T^2)$) while producing identical logits. On the hardware roofline (:numref:`fig_kv-cache-roofline`), serving splits into compute-bound **prefill** ($n = T_p$ prompt tokens processed in parallel at high arithmetic intensity) and memory-bandwidth-bound **decode** ($n = 1$ token per step at low arithmetic intensity). **Batched decoding** amortizes weight reads across $B$ concurrent sequences (:eqref:`eq_kv-cache-batch-throughput`), scaling throughput linearly with $B$ until KV cache traffic dominates and throughput saturates at $\mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$. To raise both capacity and large-batch throughput, modern architectures compress $M_{\textrm{KV}} = 2 b L n H_{\textrm{kv}} d_h$ (:eqref:`eq_kv-cache-bytes`) along three orthogonal axes (:numref:`tab_kv-cache-reduction`): head count (**GQA/MQA**), channel width (**low-rank KV / MLA**), and sequence length (**sliding windows with attention sinks**).

## Exercises

1. **Cache memory and batch limits of a 70B model.** Llama 3.1 70B has $L = 80$ layers, hidden size $d = 8192$, $H = 64$ query heads ($d_h = 128$), and $H_{\textrm{kv}} = 8$ key--value heads, stored in BF16 ($b = 2$ bytes).
   1. Use :eqref:`eq_kv-cache-bytes` to compute the KV cache size $M_{\textrm{KV}}(n, 1)$ of a single sequence at full context length $n = 131{,}072$ ($128\textrm{K}$), and compare it with what the cache would cost under standard MHA ($H_{\textrm{kv}} = 64$).
   2. On an 8-GPU node with $8 \times 80 = 640\textrm{ GB}$ of total HBM (of which $140\textrm{ GB}$ holds the BF16 model weights), what is the maximum batch size $B$ of $128\textrm{K}$-token sequences that can fit in the remaining $500\textrm{ GB}$ under GQA ($H_{\textrm{kv}} = 8$) versus MHA ($H_{\textrm{kv}} = 64$)?
   :exlabel:`ex_kv-cache-70b`

2. **Batched decode throughput and the KV-bandwidth ceiling.** Serve the 70B model of :exref:`ex_kv-cache-70b` at context length $n = 32{,}768$ on a system with aggregate HBM bandwidth $\mathcal{B}_{\textrm{HBM}} = 24\textrm{ TB/s}$ and $M_{\textrm{weights}} = 140\textrm{ GB}$.
   1. Evaluate the bandwidth upper bound :eqref:`eq_kv-cache-batch-throughput` at batch sizes $B \in \{1, 8, 32, 128, 512\}$ for both GQA ($H_{\textrm{kv}} = 8$) and MHA ($H_{\textrm{kv}} = 64$).
   2. At what batch size $B_{1/2}$ does each configuration reach half of its asymptotic ceiling $\lim_{B \to \infty} \textrm{Throughput}(B, n) = \mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$?
   :exlabel:`ex_kv-cache-70b-batch`

3. **Prefix sharing across $K$ completions.** Suppose $K = 16$ sequences in a batch share a common system prompt of $n_{\textrm{shared}} = 4096$ tokens and each has generated $n_{\textrm{unique}} = 256$ unique tokens. Modify the denominator of :eqref:`eq_kv-cache-batch-throughput` when the shared prefix KV cache is stored and read **once** per step for all $K$ sequences, and compute the resulting throughput speedup at $K = 16$ for the 70B GQA model.
   :exlabel:`ex_kv-cache-shared-reads`

4. **INT8 / FP8 KV cache quantization.** Store the cached keys and values of GPT-2 in `forward_blocks` using symmetric per-token 8-bit integer quantization: for each token vector $\mathbf{u} \in \mathbb{R}^d$, compute scale $s = \max_j |u_j| / 127$, quantize $\tilde{\mathbf{u}} = s \cdot \textrm{round}(\mathbf{u} / s)$, and use $\tilde{\mathbf{u}}$ in attention. Measure the change in `tail_loss` on `x` when quantizing keys and values to 8 bits ($2\times$ cache byte reduction) versus 4 bits (`max / 7`, $4\times$ reduction).
   :exlabel:`ex_kv-cache-quantization`

5. **Attention sinks and a virtual sink logit ($\textrm{softmax}_1$).** :citet:`Xiao.Tian.Chen.ea.2024` and gpt-oss (:numref:`tab_modern-recipe`) note that a head needs an attention sink because standard softmax forces attention weights to sum to $1$.
   1. Replace the softmax in `forward_blocks` with $\textrm{softmax}_1(\mathbf{s})_j = \frac{e^{s_j}}{e^{b} + \sum_{m} e^{s_m}}$ (adding a non-token sink logit $b = 5.0$ to the denominator without adding any value vector).
   2. Evaluate the pure sliding window ($w = 256, s = 0$ kept sink tokens) on GPT-2 with $b = 5.0$. How much of the $s = 0$ loss spike does the virtual denominator logit eliminate?
   :exlabel:`ex_kv-cache-sink-logit`

6. **Absorbing low-rank key--value reconstructions into $\mathbf{W}_q$ and $\mathbf{W}_o$.** Suppose layer $\ell$ compresses each token $j$ into a single latent vector $\mathbf{c}_j \in \mathbb{R}^r$ ($r \ll d_h H$), from which head $m$'s key and value are reconstructed via $\mathbf{k}_{j,m} = \mathbf{c}_j \mathbf{B}_{k,m}$ and $\mathbf{v}_{j,m} = \mathbf{c}_j \mathbf{B}_{v,m}$ ($\mathbf{B}_{k,m}, \mathbf{B}_{v,m} \in \mathbb{R}^{r \times d_h}$).
   1. Show that $\mathbf{q}_{t,m}\mathbf{k}_{j,m}^\top = (\mathbf{x}_t \mathbf{W}_{q,m} \mathbf{B}_{k,m}^\top)\mathbf{c}_j^\top$ and $\bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{v}_{j,m}\bigr)\mathbf{W}_{o,m} = \bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{c}_j\bigr)(\mathbf{B}_{v,m}\mathbf{W}_{o,m})$.
   2. Explain why absorbing $\mathbf{B}_{k,m}^\top$ into $\mathbf{W}_{q,m}$ and $\mathbf{B}_{v,m}$ into $\mathbf{W}_{o,m}$ during decoding avoids ever materializing the full keys $\mathbf{k}_{j,m}$ or values $\mathbf{v}_{j,m}$ in HBM, and explain why DeepSeek-V2/V3 applies RoPE to a separate narrow decoupled key $\mathbf{k}_j^{\textrm{RoPE}}$ rather than rotating $\mathbf{c}_j \mathbf{B}_{k,m}$.
   :exlabel:`ex_kv-cache-absorbed`

7. **Implementing a fixed-size streaming ring-buffer cache.** Write a `StreamingCache` variant of `init_cache` and `forward_cached` that pre-allocates only $s + w$ token slots per layer (retaining the first $s = 4$ sink positions in slots $0 \dots s-1$ and overwriting slots $s + ((t - s) \bmod w)$ in a circular buffer for $t \ge s$), and verify that it generates 2,048 tokens using constant $O(s + w)$ cache memory. `gpt2` has only 1,024 learned positions, so either use a model with rotary positions (`pos='rope'`), whose position index may grow without bound, or assign each token of `gpt2` the position of its cache slot, as :citet:`Xiao.Tian.Chen.ea.2024` do.
   :exlabel:`ex_kv-cache-streaming`

[Discussions](https://d2l.discourse.group/t/36776)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.3]{.kicker}

The KV Cache, the Hardware Roofline, and Cache Reduction<br>**KV caching · prefill vs. decode · arithmetic intensity · batched throughput · GQA & uptraining · low-rank MLA · attention sinks**
:::
:::

::: {.slide title="Naive Generation Performs 250× Redundant Work"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Naive `GPT.generate` ($O(T^2)$ work):** Re-runs the full prefix $3{,}584, 3{,}585, \dots, 3{,}839$ at every generated token ($950{,}144$ token evaluations).
- **Why past recomputation is redundant:** Under a causal mask, $(\mathbf{k}_s, \mathbf{v}_s)$ at position $s < t$ never changes when token $t$ arrives.
- **Cached generation ($O(T)$ work):** Processes the prompt once and each new token once ($3{,}840$ evaluations—**$250\times$ less work**).
:::

::: {.col .fig .big}
@fig:ch11-tf-kv-recompute-vs-cache
:::
:::
:::

::: {.slide title="Causality Freezes Past Keys and Values"}
[The Caching Principle]{.kicker}

::: {.cols .vc}
::: {.col}
- **What to cache at each block:** Query $\mathbf{q}_s$ ($s < t$) is used only when emitting token $s+1$ and never needed again. Only the post-RoPE keys $\mathbf{K}_{1:t}$ and values $\mathbf{V}_{1:t}$ are read by future steps.
- **Stage 1 — Prefill ($n = T_p$ tokens):** Processes the entire prompt in one parallel forward pass (`offset = 0`), populating slots $0, \dots, T_p - 1$ of the cache.
- **Stage 2 — Decode ($n = 1$ token/step):** Projects only the newest token $x_t$ into $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t)$, writes slot $t$, and attends $\mathbf{q}_t$ across cached $(\mathbf{K}_{0:t}, \mathbf{V}_{0:t})$.
:::

::: {.col .fig}
@fig:mdl-transformers-kv-cache
:::
:::
:::

::: {.slide title="Pre-Allocated Static Buffers and Position Offsets"}
[Buffer Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **Fixed-shape buffer allocation (`init_cache`):** Pre-allocates zeroed tensors `ks[i], vs[i]` of shape $(B, H_{\text{kv}}, T_{\max}, d_h)$ per block before generation begins.
- **Position offset `t` & boolean mask:** Rotates queries/keys at offset $t$ and masks unwritten slots $> t$ (`arange(T_max) <= pos[:, None]`), keeping tensor shapes static so the compiler compiles the decode step **once**.
:::

::: {.col .fig .big}
@fig:ch11-tf-kv-static-ring-buffer
:::
:::
:::

::: {.slide title="The Cached Step: Write One Slot, Read the Prefix" only="pytorch"}
[Implementation]{.kicker}

Inside `forward_cached`, each pre-norm block projects only the new tokens `X`, writes `k, v` into `pos = t + arange(T)` in place, and attends across the buffer:

```{.python #kv-cache-exercises-1}
H = model.token_emb(X)                                # New tokens at pos = t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = blk.attention.qkv(blk.norm1(H), t, tables)  # RoPE rotated at offset t
    ks[i][:, :, pos], vs[i][:, :, pos] = k, v             # In-place cache update
    q = q.unflatten(1, (attn.num_kv_heads, -1))           # Group query heads per KV head
    S = q @ ks[i][:, :, None].transpose(-1, -2) / math.sqrt(q.shape[-1])
    Y = d2l.masked_softmax(S, mask) @ vs[i][:, :, None]
    H = H + attn.W_o(Y.flatten(1, 2).transpose(1, 2).flatten(2))
    H = H + blk.ffn(blk.norm2(H))
```

::: {.d2l-note}
Marking cache tensors with `torch._dynamo.mark_static_address(u)` allows `torch.compile` with `triton.cudagraphs=True` to replay the entire decode step with near-zero Python dispatch overhead.
:::
:::

::: {.slide title="The Cached Step: Write One Slot, Read the Prefix" only="jax"}
[Implementation]{.kicker}

Inside `forward_cached`, each pre-norm block projects only the new tokens `X`, updates slice `(0, t, 0, 0)` of `ks[i], vs[i]`, and attends across the buffer:

```{.python #kv-cache-exercises-2}
H = model.token_emb(X)                                # New tokens at pos = t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = (u.swapaxes(1, 2) for u in blk.attention.qkv(blk.norm1(H), t, tables))
    ks[i] = jax.lax.dynamic_update_slice(ks[i], k, (0, t, 0, 0))
    vs[i] = jax.lax.dynamic_update_slice(vs[i], v, (0, t, 0, 0))
    Y = jax.nn.dot_product_attention(q, ks[i], vs[i], mask=mask)
    H = H + blk.attention.W_o(Y.reshape(B, T, -1))
    H = H + blk.ffn(blk.norm2(H))
```

::: {.d2l-note}
Passing `donate_argnames=('ks', 'vs')` to `@jax.jit` around `jax.lax.fori_loop` allows XLA to mutate the donated KV cache buffers in place without allocating copies.
:::
:::

::: {.slide title="Verifying Exact Logit Equivalence"}
[Correctness Verification]{.kicker}

KV caching is **not an approximation**: prefilling $16$ tokens and then stepping through $32$ one by one gives the **same logits** as one uncached pass over all $48$ (to rounding, $\sim 10^{-7}$), for rotary and learned positions alike:

@kv-cache-same-logits-measured-1
:::

::: {.slide title="Empirical Speedup: Over 20× Faster at 4,000 Tokens"}
[Wall-Clock Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- **Naive step time grows linearly with context $n$:** Recomputing the entire prefix on a 124M-parameter model reaches $33\text{--}53\text{ ms/token}$ at $n = 4{,}032$, making a 256-token continuation take over $8\text{ seconds}$.
- **Cached step time stays flat ($\sim 1\text{ ms/token}$):** Projecting a single token takes $0.8\text{--}1.3\text{ ms}$ across the entire $4{,}096$-token range.

@!kv-cache-same-logits-measured-3
:::

::: {.col .fig}
@!kv-cache-checking-the-cache-1
:::
:::
:::

::: {.slide title="The KV Cache Memory Footprint Formula"}
[Memory Accounting]{.kicker}

::: {.cols .vc}
::: {.col}
- Storing both $\mathbf{K}$ and $\mathbf{V}$ across $L$ blocks, batch size $B$, length $n$, and $H_{\text{kv}}$ heads of width $d_h$ at $b$ bytes/element requires:
  $$M_{\mathrm{KV}}(n, B) = 2 \, b \, B \, L \, n \, H_{\mathrm{kv}} \, d_h \;\;\text{bytes}$$
- In GPT-2 Small (FP32, $72\text{ KiB/token}$), $n = 4{,}096$ occupies $288\text{ MiB}$ at $B = 1$ and **$4.5\text{ GiB}$ at $B = 16$** ($10\times$ the weights).

@kv-cache-the-memory-bill
:::

::: {.col .fig}
@fig:ch11-tf-kv-footprint-formula
:::
:::
:::

::: {.slide title="Why Single-Token Decode Is Memory-Bandwidth Bound"}
[The Bandwidth Floor]{.kicker}

::: {.cols .vc}
::: {.col}
- **Compute vs. memory transfer:** On GPT-2 Small ($B = 1$), $2N \approx 2.5 \times 10^8\text{ FLOPs}$ takes $0.003\text{ ms}$ at $83\text{ TFLOP/s}$, yet streaming $498\text{ MB}$ of FP32 weights across $1\text{ TB/s}$ HBM imposes a hard **$0.50\text{ ms/token}$** transfer floor.
- **Measured latency ($0.8\text{--}1.3\text{ ms/token}$):** Sits within $1.6\text{--}2\times$ of the physical HBM bandwidth limit.
:::

::: {.col .fig .big}
@fig:ch11-tf-kv-bandwidth-floor
:::
:::
:::

::: {.slide title="Prefill vs. Decode on the Hardware Roofline"}
[The Roofline Model]{.kicker}

::: {.cols .vc}
::: {.col}
- **Arithmetic intensity ($I = \text{FLOPs} / \text{Bytes}$):** The ratio of peak compute ($83\text{ TFLOP/s}$) to HBM bandwidth ($1\text{ TB/s}$) defines the hardware **ridge point** at $I^* \approx 82\text{ FLOPs/byte}$.
- **Decode ($n = 1, B = 1$):** Performs $2N$ FLOPs while reading $4N + M_{\text{KV}}$ bytes ($I \approx 0.38\text{ FLOPs/byte}$ at context $2{,}048$)—$200\times$ below the ridge point (**bandwidth-bound**).
- **Prefill ($n = 2{,}048$ in parallel):** Reuses each weight read across $2{,}048$ tokens ($I \approx 780\text{ FLOPs/byte}$)—well above the ridge point (**compute-bound**).
:::

::: {.col .fig}
@fig:mdl-transformers-roofline
:::
:::
:::

::: {.slide title="Measuring Prefill vs. Decode Throughput"}
[Roofline Verification]{.kicker}

::: {.cols .vc}
::: {.col}
- Benchmarking a 2,048-token prefill against a single-token decode step at context $2{,}048$ confirms the $100\times$ gap between the two serving regimes:
- **TTFT** is governed by compute-bound prefill, whereas **ITL** is governed by memory-bandwidth-bound decode.

@kv-cache-prefill-and-decode-1
:::

::: {.col .fig}
@fig:ch11-tf-kv-ttft-vs-itl
:::
:::
:::

::: {.slide title="Batched Decoding and the KV-Bandwidth Ceiling"}
[Amortizing Weight Reads]{.kicker}

Decoding $B$ sequences simultaneously reads the model weights $M_{\text{weights}}$ **once** per step for all $B$ tokens, but must still read $B$ distinct KV caches ($B \cdot M_{\text{KV}}(n, 1)$ bytes):

$$\mathrm{Throughput}(B, n) \le \frac{B \, \mathcal{B}_{\mathrm{HBM}}}{M_{\mathrm{weights}} + B \, M_{\mathrm{KV}}(n, 1)} \;\xrightarrow{\;B \to \infty\;}\; \frac{\mathcal{B}_{\mathrm{HBM}}}{M_{\mathrm{KV}}(n, 1)}$$

- **Small-batch regime ($B \, M_{\text{KV}} \ll M_{\text{weights}}$):** Weight reads dominate HBM traffic, so total throughput scales nearly linearly with $B$ ("free batching").
- **Half-saturation batch size ($B_{1/2} = M_{\text{weights}} / M_{\text{KV}}(n, 1)$):** The batch size where total KV cache bytes equal the model weight bytes ($B_{1/2} \approx 53$ at $n = 128$, but $B_{1/2} \approx 6.6$ at $n = 1{,}024$).
:::

::: {.slide title="Measured Decode Throughput vs. Batch Size"}
[Amortizing Weight Reads]{.kicker}

@!kv-cache-batched-decoding
:::

::: {.slide title="Three Orthogonal Axes for Shrinking the KV Cache"}
[Cache Compression Overview]{.kicker}

::: {.cols .vc}
::: {.col}
- Because both VRAM capacity and the large-batch throughput ceiling $\mathcal{B}_{\text{HBM}} / M_{\text{KV}}(n, 1)$ scale inversely with $M_{\text{KV}}(n, 1) = 2 b L n H_{\text{kv}} d_h$, modern LLMs compress all three factors:
  1. **Head count ($H_{\text{kv}}$):** GQA and MQA ($4\text{--}8\times$).
  2. **Feature width ($2 H_{\text{kv}} d_h$):** Low-rank SVD / MLA ($6\text{--}16\times$).
  3. **Sequence length ($n$):** Sliding window $w$ + $s$ sinks.
:::

::: {.col .fig .big}
@fig:ch11-tf-kv-three-axes
:::
:::
:::

::: {.slide title="Axis 1: Grouped-Query (GQA) and Multi-Query Attention"}
[Sharing Key–Value Heads]{.kicker}

::: {.cols .vc}
::: {.col}
- **Multi-Head Attention (MHA, $H_{\text{kv}} = H$):** Every query head has its own private key and value head.
- **Multi-Query Attention (MQA, $H_{\text{kv}} = 1$, Shazeer 2019):** All $H$ query heads share a single key and value head, cutting the cache by $H\times$.
- **Grouped-Query Attention (GQA, $1 < H_{\text{kv}} < H$, Ainslie et al. 2023):** Partitions $H$ query heads into $H_{\text{kv}}$ groups of size $G = H / H_{\text{kv}}$, cutting the cache by $G\times$ while matching MHA quality (used in Llama 3 and Qwen3).
:::

::: {.col .fig}
@fig:mdl-transformers-gqa
:::
:::
:::

::: {.slide title="Converting MHA Checkpoints via GQA Uptraining"}
[Checkpoint Uptraining]{.kicker}

Can we convert an already-trained multi-head checkpoint ($H_{\text{kv}} = H$) into a GQA model ($H_{\text{kv}} = H/G$) without retraining from scratch?

$$\mathbf{W}_k^{(\mathcal{G})} = \frac{1}{G}\sum_{h \in \mathcal{G}} \mathbf{W}_k^{(h)}, \qquad \mathbf{W}_v^{(\mathcal{G})} = \frac{1}{G}\sum_{h \in \mathcal{G}} \mathbf{W}_v^{(h)}$$

@!kv-cache-cache-against-quality

- **Trained from scratch:** GQA ($H_{\text{kv}} = 2$, $4\times$ smaller cache) matches MHA ($H_{\text{kv}} = 8$) at **$1.34\text{ nats/byte}$**.
- **Mean-pooling + $5\text{--}10\%$ uptraining:** Zero-shot conversion starts near $2.8\text{ nats}$, but just $100$ steps ($10\%$) of uptraining recovers to **$1.47\text{--}1.48\text{ nats}$**—far ahead of keeping only the first head (`first`) or training from scratch for 50 steps ($2.7\text{ nats}$).
:::

::: {.slide title="Axis 2: Low-Rank KV Subspaces & Latent Attention (MLA)"}
[Low-Rank KV Compression]{.kicker}

::: {.cols .vc}
::: {.col}
- **Post-hoc SVD projection ($6\times$):** Projecting GPT-2's $1{,}536$-dim $[\mathbf{k}_t; \mathbf{v}_t]$ onto its top $r = 256$ singular vectors preserves perplexity ($37$ vs. $36$).
- **Multi-Head Latent Attention (MLA, DeepSeek-V2/V3):** Caches a shared latent $\mathbf{c}_t \in \mathbb{R}^{r}$ ($r = 512$) and absorbs the up-projections into $\mathbf{W}_q$ and $\mathbf{W}_o$ during decode.

@!kv-cache-low-rank-keys-and-values
:::

::: {.col .fig}
@fig:ch11-tf-kv-mla-absorption
:::
:::
:::

::: {.slide title="Axis 3: Sliding Windows Fail Without Attention Sinks"}
[The Attention Sink Phenomenon]{.kicker}

::: {.cols .vc}
::: {.col}
- On pretrained GPT-2, a pure sliding window ($w = 256, s = 0$) causes tail perplexity to jump from **$36$ to $\approx 3{,}840$**.
- **Why?** Because softmax weights must sum to $1$, idle heads place $30\%\text{--}57\%$ of their weight onto **token 0**. Evicting token 0 once $t \ge w$ scrambles the residual stream.

@!kv-cache-a-window-needs-a-sink-1
:::

::: {.col .fig}
@fig:ch11-tf-kv-sink-window
:::
:::
:::

::: {.slide title="Restoring Sliding-Window Perplexity With Sink Tokens"}
[StreamingLLM & Sinks]{.kicker}

Retaining just $s \in \{1, 4\}$ initial **attention sink** tokens permanently in cache slots $0 \dots s-1$ alongside the $w = 256$ rolling window slots ($w + s$ cached tokens total, Xiao et al. 2024) cures the collapse zero-shot:

@!kv-cache-a-window-needs-a-sink-2

::: {.d2l-note}
Keeping a single sink token ($s = 1$) drops perplexity from $\approx 3{,}840$ back to $47$ ($s = 4$ reaches $43$, vs. $36$ for full $1{,}024$-token attention). Modern architectures (such as `gpt-oss`) go one step further by giving each attention head a learnable virtual sink bias in the softmax denominator.
:::
:::

::: {.slide title="Concept Check: Roofline and KV Cache Sizing"}
[Check Your Understanding]{.kicker}

Work through these three serving calculations:

1. **70B GQA cache footprint:** Llama 3.1 70B has $L = 80$ layers, $H = 64$ query heads, $H_{\text{kv}} = 8$ KV heads of width $d_h = 128$, and BF16 weights ($140\text{ GB}$) and cache ($b = 2\text{ B}$). How many kilobytes per token does its KV cache use, and at what context length $n$ does a single sequence's cache equal the $140\text{ GB}$ weights (with GQA vs. MHA)?
2. **Half-saturation batch size:** For GPT-2 Small in FP32 ($M_{\text{weights}} \approx 498\text{ MB}$, $72\text{ KiB/token}$), at what batch size $B_{1/2}$ does KV cache traffic equal weight traffic at context $n = 4{,}096$, and what is the asymptotic decode throughput ceiling at $1\text{ TB/s}$?
3. **Window + sink memory:** How much memory does a streaming cache with window $w = 256$ and $s = 4$ sinks use on GPT-2 Small (FP32) when generating $100{,}000$ tokens?
:::

::: {.slide title="Concept Check: Roofline and KV Cache Sizing (answers)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. **$2 \times 2 \times 80 \times 8 \times 128 = 327{,}680\text{ B} = 320\text{ KiB/token}$** under GQA ($8\times$ smaller than MHA's $2.5\text{ MiB/token}$). One sequence equals the $140\text{ GB}$ weights at **$n \approx 427\text{K}$ tokens** with GQA, vs. only **$53\text{K}$ tokens** with MHA.
2. At $n = 4{,}096$, one cache is $302\text{ MB}$, so **$B_{1/2} = 498 / 302 \approx 1.6$ sequences**, and throughput can never exceed $10^{12} / (3.02 \times 10^8) \approx \mathbf{3{,}310\text{ tokens/s}}$ at any batch size.
3. **Constant $(256 + 4) \times 72\text{ KiB} = 18.3\text{ MiB}$** at any sequence length (vs. $6.9\text{ GiB}$ for an unwindowed $100\text{K}$ cache).
:::
:::

::: {.slide title="Summary: Serving Regimes and KV Cache Compression"}
[Key Takeaways]{.kicker}

| Concept / Method | Mathematical Relation / Rule | Systems & Architectural Takeaway |
| :--- | :--- | :--- |
| **Cached Decoding** | Cache post-RoPE $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$, step with $n = 1$ | Cuts the weight FLOPs of $T$-token generation from $O(T^2)$ to $O(T)$ with identical logits |
| **KV Cache Footprint** | $M_{\text{KV}}(n, B) = 2 b B L n H_{\text{kv}} d_h$ | Grows linearly with batch $B$ and context $n$ ($288\text{ MiB}$ at $n = 4096$ on GPT-2) |
| **Prefill vs. Decode** | $I_{\text{prefill}} \approx \frac{2Bn}{b_w} \gg I^*$; $I_{\text{decode}} \le \frac{2}{b_w} \ll I^*$ | Prefill is **compute-bound** ($>10^5\text{ tok/s}$); decode is **bandwidth-bound** ($\sim 10^3\text{ tok/s}$) |
| **Batched Decode** | $\text{Throughput}(B,n) \le \frac{B \mathcal{B}_{\text{HBM}}}{M_{\text{weights}} + B M_{\text{KV}}(n,1)}$ | Linear speedup for $B \ll B_{1/2}$; saturates at $\mathcal{B}_{\text{HBM}} / M_{\text{KV}}(n,1)$ |
| **GQA / MQA** | Share 1 KV head across $G = H / H_{\text{kv}}$ query heads | Shrinks cache by $G\times$; MHA checkpoints convert via mean-pooling + $5\text{--}10\%$ uptraining |
| **Low-Rank / MLA & Sinks** | Rank-$r$ latent $\mathbf{c}_t \in \mathbb{R}^r$; window $w + s$ sinks | Compresses width $6\text{--}16\times$; $s \ge 1$ sink tokens prevent sliding-window softmax collapse |
:::
