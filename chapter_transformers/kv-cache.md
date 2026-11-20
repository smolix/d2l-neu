# The KV Cache and Cache Reduction
:label:`sec_kv-cache`

Look closely at what `GPT.generate` (:numref:`subsec_gpt-sampling`) does at decoding step $t$: to sample token $t+1$, it runs a full forward pass over the entire prefix $x_{1:t}$ of length $t$, reads only the last logit vector $\mathbf{z}_t \in \mathbb{R}^V$, appends $x_{t+1}$, and on step $t+1$ feeds $x_{1:t+1}$ through all $L$ blocks all over again. Because causal attention is lower-triangular—representation $s \le t$ at every layer depends only on tokens $1, \dots, s$—recomputing the keys and values at positions $1, \dots, t-1$ on step $t$ reproduces the exact same tensors computed on step $t-1$. Across $T$ generated tokens, this redundant recomputation turns linear per-step work into $O(T^2)$ total work.

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

`init_cache` pre-allocates zeroed key and value buffers `ks, vs` of shape $(B, H_{\textrm{kv}}, T_{\max}, d_h)$ for each of the $L$ blocks, and `forward_cached(model, X, ks, vs, t)` processes a slice `X` of $n$ new tokens starting at position offset `t` (passing `offset=t` to RoPE or indexing `pos_emb[t:t+n]`), writes their keys and values into `[:, :, t:t+n, :]`, and attends against `[:, :, :t+n, :]`:

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

```{.python .input #kv-cache-checking-the-cache-4}
%%tab pytorch, jax
lengths, t_naive, t_cached = [256, 512, 1024, 2048, 4032], [], []
for n in lengths:
    t_naive.append(time_forward(model, n))
    t_cached.append(time_decode(model, 1, n))
    print(f'n={n:4d}: naive {1e3 * t_naive[-1]:5.1f} ms, '
          f'cached {1e3 * t_cached[-1]:4.1f} ms per token')
```

```{.python .input #kv-cache-checking-the-cache-1}
%%tab pytorch, jax
d2l.plot(lengths, [[1e3 * t for t in t_naive], [1e3 * t for t in t_cached]],
         'context length', 'ms per token', legend=['naive', 'cached'],
         xscale='log', yscale='log')
```

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

Why did a 2,048-token prefill take nearly the same order of magnitude of time per call as a handful of single-token decode steps? The **roofline model** (:numref:`fig_kv-cache-roofline`) explains the difference via **arithmetic intensity** (FLOPs performed per byte transferred between HBM and the compute units):

- **Prefill ($n = 2048$ tokens in parallel, compute-bound):** Each weight matrix $\mathbf{W}$ is read from HBM once and multiplied by $B n$ token vectors ($2 B n N_{\textrm{params}}$ FLOPs across $b_w N_{\textrm{params}}$ weight bytes), giving an arithmetic intensity of $\sim 2 B n / b_w \gg 80\textrm{ FLOPs/byte}$—well above the GPU's ridge point, saturating the tensor cores.
- **Single-stream decode ($B = 1, n = 1$ token per step, memory-bandwidth bound):** Each step performs only $2 N_{\textrm{params}}$ matrix--vector FLOPs while streaming **all** $b_w N_{\textrm{params}}$ weight bytes plus $M_{\textrm{KV}}(n, 1)$ cache bytes from HBM, giving an arithmetic intensity of $\le 2 / b_w \approx 0.5\textrm{ FLOPs/byte}$ (in FP32, or $1.0\textrm{ FLOP/byte}$ in BF16)—more than $100\times$ below the compute roof!

![The fp32 roofline of an RTX 4090. Attainable throughput is the smaller of the compute roof, about 83 TFLOP/s, and the arithmetic intensity times the memory bandwidth of about 1 TB/s; the two meet at the ridge point of about 80 FLOPs per byte. A decode step of a GPT-2-sized model at a context of 2,048 tokens performs 0.38 FLOPs per byte and sits far down the bandwidth slope; a prefill of 2,048 tokens, at about 780 counting weights and cache only, sits under the roof.](../img/mdl-transformers-roofline.svg)
:label:`fig_kv-cache-roofline`

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch, jax
t_prefill = time_forward(model, 2048)
t_decode = time_decode(model, 1, 2048)
```

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

---

## Shrinking the KV Cache
:label:`sec_kv-cache-reduction`

Because both memory capacity (:eqref:`eq_kv-cache-bytes`) and the large-batch decode throughput ceiling (:eqref:`eq_kv-cache-batch-throughput`) are governed by the per-sequence cache size $M_{\textrm{KV}}(n, 1) = 2 b L n H_{\textrm{kv}} d_h$, modern Transformer architectures compress the product $n \cdot H_{\textrm{kv}} \cdot d_h$ along three orthogonal axes:

1. **Fewer key--value heads ($H_{\textrm{kv}} < H$):** Grouped-query attention (GQA) :cite:`Ainslie.Lee-Thorp.deJong.ea.2023` and multi-query attention (MQA) :cite:`Shazeer.2019` share each key--value head across $G = H / H_{\textrm{kv}}$ query heads (:numref:`fig_gqa`).
2. **Narrower per-token representations ($2 H_{\textrm{kv}} d_h \to r$):** Low-rank key--value projections and Multi-Head Latent Attention (MLA) :cite:`Liu.Feng.Xue.ea.2024` compress all heads' keys and values into a low-dimensional latent vector (:numref:`subsec_kv-cache-low-rank`).
3. **Shorter cached context ($n \to w + s$):** Sliding-window attention evicts tokens older than a window $w$, while keeping $s \ge 1$ initial **attention sink** tokens :cite:`Xiao.Tian.Chen.ea.2024` so the softmax denominator remains stable (:numref:`subsec_kv-cache-sinks`).

![Query heads and the key--value heads they read. Multi-head attention gives every query head its own key--value head; grouped-query attention shares one per group of query heads; multi-query attention shares a single one across all of them. The cache scales with the number of key--value heads.](../img/mdl-transformers-gqa.svg)
:label:`fig_gqa`

### Axis 1: Grouped-Query Attention and Checkpoint Uptraining
:label:`subsec_kv-cache-gqa-quality`

Our `CausalAttention` class (:numref:`sec_gpt`) already accepts `num_kv_heads`: setting $H_{\textrm{kv}} = H / G$ shrinks $\mathbf{W}_{qkv}$ and cuts $M_{\textrm{KV}}$ by a factor of $G = H / H_{\textrm{kv}}$. Moreover, :citet:`Ainslie.Lee-Thorp.deJong.ea.2023` showed that an existing multi-head checkpoint ($H_{\textrm{kv}} = H$) can be converted into a GQA or MQA model by **mean-pooling** each group of $G$ key and value projection heads (`pool_heads`) and **uptraining** for a small fraction ($\sim 5\text{--}10\%$) of the original training budget.

We compare training our 6-block byte-level `GPT` ($H = 8$) from scratch for 1,000 steps at $H_{\textrm{kv}} \in \{8, 4, 2, 1\}$ against mean-pooling the trained $H_{\textrm{kv}} = 8$ checkpoint down to $H_{\textrm{kv}} \in \{2, 1\}$ both zero-shot (0 steps) and with 100 steps ($10\%$) of uptraining:

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

Sharing keys and values across 4 query heads ($H_{\textrm{kv}} = 2$, a $4\times$ smaller KV cache) or even all 8 query heads ($H_{\textrm{kv}} = 1$, an $8\times$ smaller KV cache) costs only $\sim 0.01\text{--}0.03$ nats/byte when trained from scratch, and 100 steps of uptraining recovers nearly the entire quality gap after head pooling.

### Axis 2: Low-Rank Key--Value Projections
:label:`subsec_kv-cache-low-rank`

Instead of reducing the number of heads $H_{\textrm{kv}}$, can we compress the $2 d$-dimensional concatenated key--value vector $[\mathbf{k}_t; \mathbf{v}_t] \in \mathbb{R}^{2d}$ of each token at layer $i$ into a rank-$r$ subspace ($r \ll 2d$)? We test this directly on pretrained **GPT-2 Small** (`124M`, where $d = 768$ and $2d = 1536$) over WikiText-103 passages of length $T = 1024$:

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

Using the first half of the validation windows to fit an empirical PCA/SVD basis $\mathbf{U}_i \in \mathbb{R}^{2d \times r}$ for each layer's concatenated key--value rows (`fit_bases`) and projecting the held-out windows onto the top $r \in \{768, 384, 256, 192, 128, 64\}$ directions (`projected`) measures how compressible GPT-2's KV cache is even without retraining:

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

Even post-hoc linear projection compresses GPT-2's KV vectors by $4\times$ ($r = 384$ out of $2d = 1536$) with minimal degradation in validation loss; when the low-rank bottleneck is trained end-to-end as in DeepSeek-V2/V3's **Multi-Head Latent Attention (MLA)** :cite:`Liu.Feng.Xue.ea.2024`, a single 512-dimensional latent vector replaces all heads' keys and values at $>16\times$ compression.

### Axis 3: Sliding Windows and Attention Sinks
:label:`subsec_kv-cache-sinks`

Our third axis bounds the sequence length $n$ stored in the cache to a rolling window of the $w$ most recent tokens (:numref:`sec_windowed-linear-attention`). However, if we apply a pure sliding window ($w = 256$) to pretrained GPT-2 on a 1,024-token passage, cross-entropy on the tail tokens ($t \ge 512$) explodes from $\sim 3.1$ nats to over $6.5$ nats!

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

In the middle and upper layers of GPT-2, queries place **40% to 70% of their entire attention weight on token 0**, regardless of what word sits at position 0! Because softmax probabilities must sum to $1$, attention heads use the earliest token(s)—the only positions visible to *every* causal query—as a default **attention sink** :cite:`Xiao.Tian.Chen.ea.2024` when no local token matches. Once a pure sliding window slides past position 0 ($t \ge w$), removing token 0 from the softmax denominator forces that $40\text{--}70\%$ probability mass onto the remaining window tokens and scrambles the residual stream.

Keeping just $s \in \{1, 2, 4\}$ initial sink tokens fixed at the front of the cache alongside the $w = 256$ most recent tokens ($w + s$ cached tokens in total) restores GPT-2's validation loss almost completely to its full-context baseline:

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
| **Shared KV heads (GQA, MQA)** :cite:`Shazeer.2019,Ainslie.Lee-Thorp.deJong.ea.2023` | Heads $H_{\textrm{kv}} < H$ | $2 L n H_{\textrm{kv}} d_h$ | Yes: train from scratch or uptrain ($\sim 5\text{--}10\%$) |
| **Low-rank latent (MLA)** :cite:`Liu.Feng.Xue.ea.2024` | Width $2 H_{\textrm{kv}} d_h \to r$ | $\approx L n r$ (plus a small rotary key) | Yes for MLA; a post-hoc SVD basis works to $\sim 4\times$ |
| **Sliding window with sinks** :cite:`Xiao.Tian.Chen.ea.2024` | Length $n \to w + s$ | $2 L (w + s) H_{\textrm{kv}} d_h$ | Works zero-shot on pretrained models when $s \ge 1$ |

## Summary

* **KV caching** stores the post-RoPE keys and values of past tokens so each decoding step projects only the single new token (`n = 1`), reducing total work for generating $T$ tokens from $O(T^2)$ to $O(T)$ while producing identical logits.
* Serving splits into two hardware regimes on the roofline (:numref:`fig_kv-cache-roofline`): **prefill** ($n = T_p$ prompt tokens in parallel, high arithmetic intensity, compute-bound) and **decode** ($n = 1$ token per step, low arithmetic intensity, memory-bandwidth bound).
* **Batched decoding** amortizes weight reads across $B$ sequences (:eqref:`eq_kv-cache-batch-throughput`), scaling throughput linearly with $B$ until KV cache traffic dominates and throughput saturates at $\mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$.
* The per-sequence cache footprint $M_{\textrm{KV}} = 2 b L n H_{\textrm{kv}} d_h$ (:eqref:`eq_kv-cache-bytes`) can be compressed along three orthogonal axes (:numref:`tab_kv-cache-reduction`): head count (**GQA/MQA**), channel width (**low-rank KV / MLA**), and sequence length (**sliding window + attention sinks**).

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

4. **INT8 / FP8 KV cache quantization.** Store the cached keys and values of GPT-2 in `forward_blocks` using symmetric per-token 8-bit integer quantization: for each token vector $\mathbf{u} \in \mathbb{R}^d$, compute scale $s = \max_j |u_j| / 127$, quantize $\tilde{\mathbf{u}} = s \cdot \textrm{round}(\mathbf{u} / s)$, and use $\tilde{\mathbf{u}}$ in attention. Measure the change in `tail_loss` on `X_eval` when quantizing keys and values to 8 bits ($2\times$ cache byte reduction) versus 4 bits (`max / 7`, $4\times$ reduction).
:exlabel:`ex_kv-cache-quantization`

5. **Attention sinks and a virtual sink logit ($\textrm{softmax}_1$).** :citet:`Xiao.Tian.Chen.ea.2024` and gpt-oss (:numref:`tab_modern-recipe`) note that a head needs an attention sink because standard softmax forces attention weights to sum to $1$.
   1. Replace the softmax in `forward_blocks` with $\textrm{softmax}_1(\mathbf{s})_j = \frac{e^{s_j}}{e^{b} + \sum_{m} e^{s_m}}$ (adding a non-token sink logit $b = 5.0$ to the denominator without adding any value vector).
   2. Evaluate the pure sliding window ($w = 256, s = 0$ kept sink tokens) on GPT-2 with $b = 5.0$. How much of the $s = 0$ loss spike does the virtual denominator logit eliminate?
:exlabel:`ex_kv-cache-sink-logit`

6. **Absorbing low-rank key--value reconstructions into $\mathbf{W}_q$ and $\mathbf{W}_o$.** Suppose layer $\ell$ compresses each token $j$ into a single latent vector $\mathbf{c}_j \in \mathbb{R}^r$ ($r \ll d_h H$), from which head $m$'s key and value are reconstructed via $\mathbf{k}_{j,m} = \mathbf{c}_j \mathbf{B}_{k,m}$ and $\mathbf{v}_{j,m} = \mathbf{c}_j \mathbf{B}_{v,m}$ ($\mathbf{B}_{k,m}, \mathbf{B}_{v,m} \in \mathbb{R}^{r \times d_h}$).
   1. Show that $\mathbf{q}_{t,m}\mathbf{k}_{j,m}^\top = (\mathbf{x}_t \mathbf{W}_{q,m} \mathbf{B}_{k,m}^\top)\mathbf{c}_j^\top$ and $\bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{v}_{j,m}\bigr)\mathbf{W}_{o,m} = \bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{c}_j\bigr)(\mathbf{B}_{v,m}\mathbf{W}_{o,m})$.
   2. Explain why absorbing $\mathbf{B}_{k,m}^\top$ into $\mathbf{W}_{q,m}$ and $\mathbf{B}_{v,m}$ into $\mathbf{W}_{o,m}$ during decoding avoids ever materializing the full keys $\mathbf{k}_{j,m}$ or values $\mathbf{v}_{j,m}$ in HBM, and explain why DeepSeek-V2/V3 applies RoPE to a separate narrow decoupled key $\mathbf{k}_j^{\textrm{RoPE}}$ rather than rotating $\mathbf{c}_j \mathbf{B}_{k,m}$.
:exlabel:`ex_kv-cache-absorbed`

7. **Implementing a fixed-size streaming ring-buffer cache.** Write a `StreamingCache` variant of `init_cache` and `forward_cached` that pre-allocates only $s + w$ token slots per layer (retaining the first $s = 4$ sink positions in slots $0 \dots s-1$ and overwriting slots $s + ((t - s) \bmod w)$ in a circular buffer for $t \ge s$), and verify that it generates 2,048 tokens from `gpt2` using constant $O(s + w)$ cache memory.
:exlabel:`ex_kv-cache-streaming`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.3]{.kicker}

The KV Cache<br>
**reusing past keys and values · prefill and decode · the roofline · batching**
:::
:::

::: {.slide title="Naive Generation Does 250 Times the Necessary Work"}
[The problem]{.kicker}

GPT-2 small, $N$ = 124M parameters, writes 256 tokens after a 3,584-token
prompt.

- Naive `generate`: one forward pass per new token, over 3,584, …, 3,839
  tokens: $2N \sum n \approx 2.4 \times 10^{14}$ FLOPs.
- Read the prompt once, then each new token once:
  $2N \cdot 3{,}840 \approx 10^{12}$ FLOPs.

::: {.d2l-note}
Causal attention never changes the keys and values of past tokens.
**Store them**, and the repetition disappears.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write a cached forward pass with a position offset and preallocated
  buffers;
- estimate the floor of a decode step from the bytes it reads;
- compute the cache size of any transformer from its shape;
- place prefill and decode on the roofline: reading a prompt is limited by
  arithmetic, writing text by memory bandwidth;
- predict where batching stops helping.

**Running example:** GPT-2 small (124M; 12 blocks, 12 heads of width 64),
72 KiB of cache per token in fp32.
:::

::: {.slide title="A Cache Makes the Trained GPT Fast to Sample"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| **The KV Cache** | **reuse past keys and values** |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT`, causal masks, RoPE, top-$k$ sampling.
- **Skips:** serving engines (paging, continuous batching).
:::
:::
:::

::: {.slide title="Causality Freezes the Keys and Values of the Past"}
::: {.cols .vc}
::: {.col}
- Step $t$ needs one new query against $\mathbf{k}_0, \ldots, \mathbf{k}_t$.
- Positions $0, \ldots, t-1$ never see position $t$: their keys and values
  are the same as one step ago.
- **Prefill:** the whole prompt in one pass, filling the cache.
- **Decode:** one token per pass, reading the cache.
:::

::: {.col .fig .big}
@fig:mdl-transformers-kv-cache
:::
:::
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="pytorch"}
```{.python #kv-cache-exercises-1}
H = model.token_emb(X)              # new tokens at pos = t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = blk.attention.qkv(blk.norm1(H), t)      # RoPE offset t
    ks[i][:, :, pos], vs[i][:, :, pos] = k, v         # write in place
    q = q.unflatten(1, (H_kv, -1))         # query groups per kv head
    S = q @ ks[i][:, :, None].transpose(-1, -2) / math.sqrt(d_h)
    Y = d2l.masked_softmax(S, mask) @ vs[i][:, :, None]
    Y = Y.flatten(1, 2).transpose(1, 2).flatten(2)
    H = H + blk.attention.W_o(Y)
    H = H + blk.ffn(blk.norm2(H))
```

- Buffers of fixed size; the mask hides positions not yet written, so every
  decode step has the same shapes.
- `torch.compile` records the step once and replays it as a CUDA graph.
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="jax"}
```{.python #kv-cache-exercises-2}
H = model.token_emb(X)                    # new tokens at t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = (u.swapaxes(1, 2)
               for u in blk.attention.qkv(blk.norm1(H), t))
    ks[i] = jax.lax.dynamic_update_slice(ks[i], k, (0, t, 0, 0))
    vs[i] = jax.lax.dynamic_update_slice(vs[i], v, (0, t, 0, 0))
    Y = jax.nn.dot_product_attention(q, ks[i], vs[i], mask=mask)
    H = H + blk.attention.W_o(Y.reshape(B, T, -1))
    H = H + blk.ffn(blk.norm2(H))
```

- Fixed-size buffers; the mask hides positions not yet written.
- Inside one compiled `fori_loop`, XLA writes the cache in place.
:::

::: {.slide title="Cached and Full Logits Agree to Rounding Error"}
@!kv-cache-same-logits-measured-1

Prefill 16 tokens, decode 32, compare with one forward pass over all 48,
for rotary and for learned positions.
:::

::: {.slide title="At 4,000 Tokens a Cached Step Is Over 20 Times Faster"}
@!kv-cache-checking-the-cache-1

124M-parameter model, untrained, both steps compiled. Naive: $2Nn$ FLOPs,
grows more slowly than $n$ below about 1,000 tokens, tens of ms at 4,032.
Cached: 0.8 to 1.3 ms over the whole range.
:::

::: {.slide title="A Decode Step Cannot Beat One Read of the Weights"}
| GPT-2 small, one sequence | |
|---|---|
| fp32 weights, 124M parameters | $\approx 0.5$ GB |
| bandwidth of an RTX 4090 | $\approx 1$ TB/s |
| floor per decode step | $\approx 0.5$ ms |
| measured, short context | 0.8--0.9 ms |

- Extra 0.3--0.4 ms: a *fixed cost per step* (small kernels,
  matrix--vector products below peak bandwidth).
- The cache read adds the rest as $n$ grows.
- Only one read serving several tokens beats the floor: batching,
  speculative decoding.

::: {.d2l-note}
For one sequence, decoding speed is set by **bytes read**, not FLOPs.
:::
:::

::: {.slide title="Check: The Saving Without a Prompt"}
[Check]{.kicker}

A model with $N$ = 124M parameters writes $T$ = 4,096 tokens from an empty
prompt, $2N$ FLOPs per token of a forward pass. By what factor does the cache cut the FLOPs? With
a 3,584-token prompt and 256 new tokens it was about 250.

::: {.fragment .fade-up}
- Naive: $2N \sum_{n=1}^{T} n \approx NT^2 \approx 2.1 \times 10^{15}$.
- Cached: $2NT \approx 10^{12}$, so the factor is about $T/2 \approx 2{,}000$.
- The saving is roughly the number of generated tokens: all of them after a
  long prompt, half of them without one.
:::
:::

::: {.slide title="The Cache Grows with Layers, Heads, Tokens, and Sequences"}
$$\textrm{cache bytes} = 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n \cdot b \cdot (\textrm{bytes per element})$$

$L$ blocks, $H_\textrm{kv}$ key--value heads of width $d_h$, $n$ tokens, $b$
sequences.

- GPT-2 small, fp32: $2 \cdot 12 \cdot 12 \cdot 64 \cdot 4$ B = 72 KiB per
  token.
- 4,096 tokens: 288 MiB, more than half of the weights, 471 MiB
  (494 MB).

@!kv-cache-the-memory-bill
:::

::: {.slide title="A Decode Step Reads the Whole Model for One Token"}
Arithmetic intensity = FLOPs per byte moved. Roofline: time $\geq$
max(FLOPs / peak, bytes / bandwidth). An RTX 4090: 83 TFLOP/s, 1 TB/s, so
the bounds meet at the *ridge point*, about 80 FLOPs per byte.

| step | FLOPs | bytes | FLOPs per byte |
|---|---|---|---|
| decode, 1 token | $2N$ | $4N$ + cache | below 1 |
| prefill, $n$ tokens | $2Nn$ | $4N$ + cache + activations | about $n/2$, less with cache and activations |

::: {.d2l-note}
Prefill is limited by arithmetic, decode by memory bandwidth.
:::

[Pope et al., 2023]{.small}
:::

::: {.slide title="Decode Sits Far Down the Bandwidth Slope"}
::: {.cols .vc}
::: {.col .fig .big}
@fig:mdl-transformers-roofline
:::

::: {.col .narrow}
- Decode step: 0.38 FLOP per byte, 200 times below the ridge.
- Prefill of 2,048 tokens: about 780 (weights and cache only), under the
  compute roof.
:::
:::
:::

::: {.slide title="Prefill Reads Over 100,000 Tokens per Second, Decode About 1,000"}
@!kv-cache-prefill-and-decode-1

- Same layers, same GPU: decoding is more than a hundred times slower.
- Prefill reaches a third to a half of the 335,000 tokens/s that 83 TFLOP/s
  allows; decode about 60% of the bandwidth ceiling.
- Each misses its bound by less than 3×; the roofline gets the 100× right.
- Serving reports both: time to first token (prefill), tokens/s (decode).
:::

::: {.slide title="Batching Amortizes Weights but Not the KV Cache"}
$b$ sequences, weights $W$ bytes, cache $c_\textrm{kv}$ bytes per token,
context $n$, bandwidth $\beta$:

$$\textrm{tokens per second} \approx \frac{b\beta}{W + bnc_\textrm{kv}}
\;\xrightarrow{\;b \to \infty\;}\; \frac{\beta}{nc_\textrm{kv}}$$

- GPT-2 small, fp32: $W$ = 494 MB, $c_\textrm{kv}$ = 72 KiB.
- Caches equal the weights at $b \approx 52$ for $n = 128$, but at
  $b \approx 6.5$ for $n = 1024$.
:::

::: {.slide title="Long Contexts Cap the Gain from Batching"}
@!kv-cache-batched-decoding

Solid: measured. Dotted: the bandwidth bound at 1 TB/s.

- $n = 128$: 64 sequences decode about 28× as fast as one.
- $n = 1024$: from $b = 16$ to 64, only 1.4 to 1.5× more, to about 10,000
  tokens/s.
:::

::: {.slide title="Check: Where the Cache Outweighs the Weights"}
[Check]{.kicker}

GPT-2 small in fp32 at a context of $n = 4096$ tokens. At which batch size
does the cache equal the weights, and what is the ceiling on tokens per
second at 1 TB/s?

::: {.fragment .fade-up}
- $nc_\textrm{kv} = 4096 \cdot 72$ KiB = 288 MiB $\approx$ 302 MB per
  sequence, so $b \approx 494 / 302 \approx 1.6$.
- Ceiling $\beta / (nc_\textrm{kv}) \approx 10^{12} / (3.0 \times 10^8)
  \approx 3{,}300$ tokens per second, whatever the batch.
:::
:::

::: {.slide title="Check: The Cache of a 70B Model"}
[Check]{.kicker}

Llama 3.1 70B: 80 layers, 64 query heads sharing 8 key--value heads of
width 128, 16-bit cache, 140 GB of weights, contexts up to 128K tokens. How
many cache bytes per token, and at which context does one sequence's cache
reach the weights?

::: {.fragment .fade-up}
- $2 \cdot 80 \cdot 8 \cdot 128 \cdot 2$ B = 320 KiB per token.
- $140 \times 10^9 / 327{,}680 \approx 430{,}000$ tokens, beyond the
  context; without sharing (64 key--value heads), about 53,000.
:::

[Grattafiori et al., 2024]{.small}
:::

::: {.slide title="The Cache Removes the Arithmetic, and Bandwidth Then Sets the Speed"}
[Wrap-up]{.kicker}

- Causality lets a cache replace $\approx 2.4 \times 10^{14}$ FLOPs by
  $\approx 10^{12}$ for 256 tokens after 3,584, with the same logits to
  rounding error.
- A decode step costs at least one read of the weights: 0.5 ms for
  GPT-2 small in fp32; measured 0.8 to 1.3 ms.
- GPT-2 small caches 72 KiB per token in fp32: 288 MiB at 4,096 tokens.
- Prefill over 100,000 tokens/s, decode about 1,000: arithmetic against
  bandwidth.
- Batching shares the weight reads, not the cache reads: at 1,024 tokens
  throughput levels off near 10,000 tokens/s.

**Next:** shrinking the cache: fewer key--value heads, fewer dimensions,
fewer tokens.
:::

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.4]{.kicker}

Shrinking the KV Cache<br>
**shared key--value heads · uptraining · low-rank keys and values · windows and sinks**
:::
:::

::: {.slide title="Without Shared Heads, Llama 3 8B Would Cache Four Times Its Weights"}
[The problem]{.kicker}

$$\textrm{cache bytes} = 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n \cdot b \cdot (\textrm{bytes per element})$$

$L$ blocks, $H_\textrm{kv}$ key--value heads of width $d_h$, $b$ sequences
of $n$ tokens. Llama 3 8B: $L$ = 32, $d_h$ = 128, 32 query heads, 16-bit
cache, 16 GB of weights; one sequence of 128K tokens.

- One key--value head per query head ($H_\textrm{kv}$ = 32): 512 KiB per
  token, 64 GiB.
- $H_\textrm{kv}$ = 8, as released: 128 KiB per token, 16 GiB.

::: {.d2l-note}
Every factor of the formula is a lever: heads, width per token, tokens.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count what sharing key--value heads saves in parameters and cache;
- convert a multi-head checkpoint to grouped heads and uptrain it;
- bound the dimension of a token's keys and values, and test a low-rank
  cache on GPT-2;
- explain why a sliding window must keep the first token.

**Running examples:** the byte-level GPT (4.8M parameters, 8 heads,
validation loss about 1.35 nats per byte) and GPT-2 small (12 blocks, 12
heads of width 64, 72 KiB per token in fp32).
:::

::: {.slide title="Shrinking the Cache Makes Long Contexts and Large Batches Fit"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse past keys and values |
| **Shrinking the KV Cache** | **fewer heads, narrower or shorter caches** |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** the cache formula, `num_kv_heads`, GPT-2.
- **Skips:** quantized caches (an exercise), recurrent states (next
  chapter), eviction policies (Language Models part).
:::
:::
:::

::: {.slide title="MQA and GQA Divide the Cache by the Group Size"}
@fig:mdl-transformers-gqa

- Query heads differ in how they weight positions; that can come from
  their query projections alone.
- MQA: one key--value head for all $H$ query heads, cache $\div H$.
- GQA: $H_\textrm{kv}$ heads, each read by $H/H_\textrm{kv}$ consecutive
  query heads. Llama 3 8B, Mistral 7B: 32 query heads, 8 key--value heads.
:::

::: {.slide title="Fewer Key--Value Heads Shrink Two Projections and the Cache"}
@!kv-cache-a-pluggable-implementation-2

One block, $d$ = 256, $H$ = 8 query heads of width 32.

- $\mathbf{W}_q, \mathbf{W}_o$ stay $d \times d$; $\mathbf{W}_k,
  \mathbf{W}_v$ shrink to $d \times H_\textrm{kv} d_h$.
- Predicted: $4d^2$ = 262,144 parameters with 8 key--value heads,
  $2.25\,d^2$ = 147,456 with 1.

::: {.d2l-note}
At most half of the attention parameters go; the cache falls by the full
factor $H/H_\textrm{kv}$.
:::
:::

::: {.slide title="Averaging a Group's Projections Converts a Checkpoint"}
[Uptraining]{.kicker}

Trained model with $H$ key--value heads; target $H_\textrm{kv}$ groups of
$H/H_\textrm{kv}$ consecutive heads. For a group $\mathcal{G}$:

$$\mathbf{W}_k^{(\mathcal{G})} = \frac{H_\textrm{kv}}{H} \sum_{h \in \mathcal{G}} \mathbf{W}_k^{(h)},
\qquad \mathbf{W}_v^{(\mathcal{G})} = \frac{H_\textrm{kv}}{H} \sum_{h \in \mathcal{G}} \mathbf{W}_v^{(h)}$$

Queries, output projection, FFN, and norms are copied. Control: keep only
the first head of each group.

::: {.d2l-note}
Then train on for a few percent of the original steps: *uptraining*.
:::
:::

::: {.slide title="Converting Breaks the Model; 50 Steps Recover Most of It"}
@!kv-cache-cache-against-quality

- Two key--value heads from the start: 1.34--1.35, as good as eight (runs
  agree within about 0.01).
- Converted: 2.7--2.9, worse than a bigram model (2.45). After 50 steps
  about 1.6; 50 steps from scratch: 2.6--2.8.
- After 100 steps: 1.45--1.48, still 0.1 or more above; averaging leads
  the first head by over 0.1 in every run.

::: {.d2l-note}
Conversion breaks only the match of queries to keys; the rest survives.
:::
:::

::: {.slide title="At Scale, the Cost of Sharing Depends on the Model"}
[Evidence]{.kicker}

| study | training | 8 key--value groups | 1 key--value head |
|---|---|---|---|
| T5-XXL, avg. score | uptrained, 5% | 47.2 → 47.1 | 46.6 |
| Llama 2, 30B | from scratch | about as good | a little worse |
| DeepSeek-V2, 7B, MMLU | from scratch | 45 → 41 | 38 |

::: {.d2l-note}
Most open models train with GQA from the start, at $H/8$ to $H/4$: four
to eight times less cache per token.
:::

[Ainslie et al., 2023; Touvron et al., 2023; DeepSeek-AI, 2024]{.small}
:::

::: {.slide title="Check: Sharing Heads in GPT-2"}
[Check]{.kicker}

GPT-2 small: 12 blocks, $d$ = 768, 12 heads of width 64, fp32; its full
cache at 1,024 tokens is 72 MiB. Share each key--value head among three
query heads ($H_\textrm{kv}$ = 4). How large is the cache, and how many
attention weights does a block keep?

::: {.fragment .fade-up}
- Cache $\div 3$: 24 MiB at 1,024 tokens.
- $\mathbf{W}_k, \mathbf{W}_v$ become $768 \times 256$:
  $2 \cdot 768^2 + 2 \cdot 768 \cdot 256 \approx 1.57$M weights against
  $4 \cdot 768^2 \approx 2.36$M, a third fewer.
:::
:::

::: {.slide title="GPT-2's 1,536 Keys and Values Are an Affine Function of 768 Numbers"}
[Low rank]{.kicker}

Per block and token: $[\mathbf{k}; \mathbf{v}] = \mathbf{W}\mathbf{x} +
\mathbf{b}$, with $\mathbf{x} \in \mathbb{R}^{768}$ the normalized input
and $\mathbf{W} \in \mathbb{R}^{1536 \times 768}$.

- For any weights, every token's keys and values lie in a 768-dimensional
  subspace, shifted by $\mathbf{b}$.
- Caching $\mathbf{x}$ would be exact at half the bytes, but every read
  recomputes $\mathbf{W}\mathbf{x}$.

::: {.d2l-note}
The real question: how far below 768 can a cache go?
:::
:::

::: {.slide title="GPT-2's Keys and Values Need Only a Third of Their 768 Dimensions"}
@!kv-cache-low-rank-keys-and-values

- Basis: top $r$ right singular vectors of the key--value rows of a
  *different* passage; the cache stores $r$ numbers per token.
- Full attention 3.58. Ranks 512 and 256: within 0.03, no difference one
  passage resolves. Rank 128 (a sixth): +0.56.

::: {.d2l-note}
The subspace carries over between passages, so a model can fix it once.
:::
:::

::: {.slide title="MLA Learns the Compression During Training"}
[Multi-head latent attention]{.kicker}

- A trained down-projection maps each token to one latent; the cache
  stores only the latent.
- The up-projections fold into the query and output projections: decoding
  never rebuilds keys or values.
- RoPE's rotation depends on each key's position and cannot be folded, so
  MLA caches a small rotary key per token.

::: {.d2l-note}
DeepSeek-V2: a latent of 512 for a hidden state of 5,120, tenfold.
:::

[DeepSeek-AI, 2024]{.small}
:::

::: {.slide title="From Block 3 On, GPT-2 Puts a Third to Half of Its Attention on Token 0"}
@!kv-cache-a-window-needs-a-sink-1

- A query's weights sum to one. A head with nothing to retrieve still puts
  them somewhere: on the first token, which every query sees. This is an
  *attention sink*.
:::

::: {.slide title="One Sink Token Is Worth a Factor of 80 in Perplexity"}
@!kv-cache-a-window-needs-a-sink-2

- Window of 256 tokens plus $s$ sink tokens, the first $s$ of the text; no
  training. $s$ = 0: perplexity 36 → over 3,800. $s$ = 1: 47. $s$ = 4: 43.
- Newer models give each head a learned sink logit instead (gpt-oss).

::: {.d2l-note}
Recent tokens were nearly enough; the window removed where heads put
unneeded weight.
:::
:::

::: {.slide title="Each Method Shrinks One Factor of the Cache Formula"}
| method | factor reduced | stored numbers | needs training? |
|---|---|---|---|
| GQA, MQA | heads $H_\textrm{kv}$ | $2LnH_\textrm{kv}d_h$, fewer heads | yes, or uptrain |
| latent (MLA) | width per token | about $Lnr$ + a rotary key | yes (a fitted basis: no) |
| window + sinks | length $n$ | $2L(w+s)H_\textrm{kv}d_h$ | no; better if pretrained with a sink |

A window combines with either other row; GQA and a latent are
alternatives. Quantization (2 to 4 bits) combines with all three. A
fixed-size recurrent state forgets no token outright.
:::

::: {.slide title="Check: A Window with Sinks for GPT-2"}
[Check]{.kicker}

GPT-2 small in fp32 caches 72 KiB per token. With a window of $w$ = 256
and $s$ = 4 sink tokens, how large is the cache at 1,024 tokens, and at
100,000?

::: {.fragment .fade-up}
- $260 \cdot 72$ KiB $\approx$ 18 MiB at any length, against 72 MiB for
  the full cache at 1,024 tokens.
- At 100,000 tokens the full cache would need about 6.9 GiB (if the
  positions allowed it).
:::
:::

::: {.slide title="Share Heads First; a Latent or a Window with Sinks Shrinks the Rest"}
[Wrap-up]{.kicker}

- Sharing heads: 8 → 2 key--value heads cut the byte model's cache
  fourfold at no measurable loss when trained that way; a converted
  checkpoint needs uptraining, and averaging beats keeping one head.
- GPT-2's keys and values are a function of 768 numbers; a basis of 256
  from another passage costs nothing measurable, one of 128 half a nat.
- A window bounds the cache only if it keeps the attention sink.

**Next:** encoders, decoders, and cross-attention.
:::
