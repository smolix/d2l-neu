# The KV Cache
:label:`sec_kv-cache`

The GPT of :numref:`sec_gpt` writes text one token at a time, and its
`generate` method produces each new token by running the model over the
whole text so far. A chat assistant also writes one token at a time. If it
were built like our GPT, it would answer a long question by reading the
whole question again for every word of its answer. Does a model have to
reprocess everything before it to choose the next token?

It does not. Attention is causal, so the keys and
values of earlier tokens never change once computed. The *KV cache* stores
them, and each step then runs the model on the new token alone. For a long
prompt this removes all but a small fraction of the arithmetic.

This section builds the cache and measures the speedup. It then turns to the
problem the cache creates. Generation becomes limited by memory rather than
by arithmetic, because every generated token reads all weights of the model
and the whole cache from memory. The cache of a long context, or of many
users served together, outgrows the weights. We derive the size of the
cache, measure how much faster a model reads a prompt than it writes text,
and see why batching helps less as contexts grow. Then we shrink the cache
in three ways: by sharing keys and values across heads, by compressing them
to fewer dimensions, and by bounding the context with a sliding window that
keeps a few early tokens. The experiments use `d2l.GPT` from
:numref:`sec_gpt`, and the released GPT-2 weights for the effects that only
a trained model shows.
<!-- WRITER: the split moved the second half (sharing heads, low rank,
windows and sinks) to :numref:`sec_kv-cache-reduction`; the roadmap ends with
batching and one sentence pointing there, and GPT-2's weights are no longer
used in this section (every model here is untrained). -->

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

A forward pass of a model with $N$ parameters over $t$ tokens costs about
$2Nt$ floating-point operations, because every parameter takes part in one
multiplication and one addition per token. The attention scores add work
that grows with $t^2$ (:numref:`sec_attention-at-scale`). At the lengths
we run here it stays below $2Nt$, and we leave it out of the count, although
at a few thousand tokens it is no longer small. Naive generation calls this
forward pass once per new token, over a history that grows by one token per
step. Producing $T$ tokens therefore costs about
$\sum_{t} 2Nt \approx NT^2$ operations, quadratic in the length of the
text. Of the logits that each pass computes, only the last row is used.

Most of that work is repeated. At step $t$ the model needs, in every layer,
the attention output for one new query $\mathbf{q}_t$ against the keys
$\mathbf{k}_1, \ldots, \mathbf{k}_t$ and values $\mathbf{v}_1, \ldots,
\mathbf{v}_t$. The causal mask guarantees that positions $1, \ldots, t-1$
never see position $t$, so their hidden states, and therefore their keys
and values, are the same at step $t$ as they were at step $t-1$. A KV
cache stores, for every layer, the keys and values computed so far. Each
step then runs the model on the one new token, writes its key and value
into the cache, and attends to everything stored.
:numref:`fig_kv-cache` contrasts the two schedules. A cached step costs
about $2N$ operations, a forward pass over a single token, plus the scores
of one query against $t$ keys. For a model of GPT-2's size, with 124
million parameters, writing 256 tokens after a prompt of 3,584 tokens takes
about $2.4 \times 10^{14}$ operations without the cache. Reading the prompt
once and then processing each new token once takes about $10^{12}$, some
250 times less.

![One decoding step at context length $t$. Naive generation recomputes every key, value, and score row of the prefix and then uses only the last row. With a KV cache, the past keys and values are read from memory, and the step computes a single new key, value, and score row.](../img/mdl-transformers-kv-cache.svg)
:label:`fig_kv-cache`

### The Cached Forward Pass
:label:`subsec_kv-cache-cached-forward`

Two details need care. The first is position. RoPE rotates queries and keys
by their absolute positions, so a step must know the position $t$ at which
its new tokens start. `GPT.CausalAttention.qkv` takes this offset as an
argument, and a model with learned positions looks up the embeddings of
positions $t, t+1, \ldots$ instead of $0, 1, \ldots$. The RoPE angles
depend on the positions alone, so a call computes their tables once with
`d2l.rope_tables` and passes them to every block. The second is storage.
Appending each new key to a growing tensor would copy the whole cache at
every step, work that grows with the context. We therefore allocate the
cache once, at its final size: for every block, one buffer of keys and one
of values with room for `max_len` positions. A call with $T$ new tokens
writes their keys and values into positions $t, \ldots, t+T-1$ and attends
to the filled positions $0, \ldots, t+T-1$.

The same function serves both phases of generation. In *prefill*, the
whole prompt enters at $t = 0$, and its queries need the causal mask. In
*decode*, one token at position $t$ attends to all filled positions. The
mask `d2l.causal_mask(T, t + T)` covers both cases, because it treats the
$T$ queries as the last $T$ of $t + T$ positions
(:numref:`sec_attention-scoring-functions`).
<!-- WRITER: both tabs now attend over the whole preallocated buffer, with a
mask over all max_len positions that also hides the positions not yet
written (the mask of d2l.causal_mask(T, t + T), extended with False); this
keeps the shapes of a decode step the same at every t, which compiling the
step needs. The position t may be a tensor (PyTorch) or a traced value (JAX).
-->

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

:begin_tab:`pytorch`
The buffers are written in place, so `forward_cached` returns the same `ks`
and `vs` that it received; returning them keeps one calling convention for
the functions below. The slice `ks[i, :, :, :t + T]` is a view into the
buffer, not a copy. Computing the RoPE tables once per call instead of in
every block saves about a quarter of the time of a decode step, which, as
we will see, goes mostly to launching small kernels.
<!-- WRITER: this note no longer describes the code (KV-3, KV-4: cut "a
quarter"). New facts, one sentence at most: the buffers are written in place
by index (`ks[i][:, :, pos] = k`); `init_cache` gives each buffer a fixed
address (`torch._dynamo.mark_static_address`) so that a compiled step can
write into it when replayed as a CUDA graph; attention is written out
(scores, `d2l.masked_softmax`, weighted sum), with the query heads grouped
per key-value head, because the fused kernel is slow for one query over a
long masked buffer at batch 1 (4.3 ms against 1.1 ms per step at n = 2048,
measured). -->
:end_tab:

:begin_tab:`jax`
A JAX array cannot be changed in place, and a compiled function needs
shapes that do not change from call to call. `forward_cached` therefore
returns new buffers made by `dynamic_update_slice`, and attention reads the
whole buffer, with a mask that hides the positions not yet written. Inside a
compiled loop, as below, XLA turns these updates into writes in place. The
buffers keep the (batch, length, heads, head width) layout of the fused
kernel `jax.nn.dot_product_attention`, so no step transposes the cache.
The mask is that of `d2l.causal_mask(T, t + T)`, extended with False over
the positions not yet written. We spell it out because inside a compiled
loop `t` is a traced value, while the helper needs concrete sizes to build
its array.
`dynamic_update_slice` does not fail on a write past the end of the
buffer; it moves the start index back silently, so the caller must keep
$t + T$ within `max_len`.
<!-- WRITER: the mask is now shared prose; the clamping pitfall is a code
comment; the cache is one buffer per block in both tabs (a list). One
sentence remains: arrays are immutable, `dynamic_update_slice` returns new
buffers that XLA writes in place inside the compiled loop. -->
:end_tab:

Generation is a prefill call over the prompt followed by one decode call
per new token. `decode` draws each token from the last logits, after the
temperature and top-$k$ truncation of :numref:`subsec_gpt-sampling`, and
feeds it back. `generate_cached` gives it the interface of `generate`.
<!-- WRITER: PyTorch `decode` now starts from the logits of the prefill
(`generate_cached` runs the prefill, an eager `forward_cached` over the
prompt) and runs each one-token step through `decode_step`, a
`torch.compile` of `forward_cached` replayed as CUDA graphs; JAX `decode`
compiles the prefill and the loop together. Neither runs a step after the
last token any more. One shared clause: both tabs compile the step so that
the time measures the GPU's work, not Python's
(:numref:`sec_compilation`); `sample` repeats two lines of `generate`. -->

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

:begin_tab:`jax`
`decode` compiles the whole generation loop. `nnx.split` separates the
model into its structure, `graphdef`, which the compiled function treats as
a constant, and its arrays, `params`, which it takes as input; `nnx.merge`
reassembles the model inside. `lax.fori_loop` runs the decode steps on the
device and carries the cache from step to step, so XLA updates the buffers
in place and no token reaches the host before the end. `decode` also
*donates* `ks` and `vs` (`donate_argnames`): XLA may build the new cache in
their memory instead of in a copy, so the arrays passed in must not be used
again, and `decode` returns the updated cache in their place. The function
compiles once for each prompt length, buffer size, and `num_tokens`.
Calling a forward step through `nnx.jit` once per token would instead walk
the module's Python object graph at every call, which for a model of a
dozen blocks can take longer than the step itself.
:end_tab:

### Checking the Cache
:label:`subsec_kv-cache-same-logits`

The cached path must reproduce the logits of the full forward pass. For
both positional schemes, we prefill the first 16 tokens of a random
sequence, decode the remaining 32 one at a time, and compare the logits
after each position with those of one forward pass over the whole
sequence. We build every model of this section through one shorthand:

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

:begin_tab:`jax`
We set matrix multiplications to full float32 precision for this
comparison. On GPUs with tensor cores, JAX multiplies float32 matrices in
the lower-precision TF32 format by default, which rounds the two orders of
computation differently and would hide that they compute the same values.
:end_tab:

The two paths agree to rounding error.

### Measuring the Speedup
:label:`subsec_kv-cache-speedup`

Next we time one generation step at
growing context lengths, on an untrained model of GPT-2's size: 12 blocks
of width 768, 124 million parameters. The arithmetic does not depend on
the weights.

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

Two helpers time the two kinds of step. The naive step, `time_forward`, is
a forward pass over all $n$ tokens. The cached step, `time_decode`, is a
decode call against a cache of $n$ positions, averaged over 32 steps, for a
batch of sequences; we reuse it below. Both time with `d2l.Benchmark`,
which calls a function a few times to warm up, waits for the device to
finish, and averages the time of ten further calls.
<!-- WRITER: both helpers now time compiled code in both tabs (PyTorch
`time_forward` runs `torch.compile(model)`; `time_decode` runs `decode` from
uniform logits, 32 compiled steps). -->

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

We time both steps at five context lengths:

```{.python .input #kv-cache-checking-the-cache-4}
%%tab pytorch, jax
lengths, t_naive, t_cached = [256, 512, 1024, 2048, 4032], [], []
for n in lengths:
    t_naive.append(time_forward(model, n))
    t_cached.append(time_decode(model, 1, n))
    print(f'n={n:4d}: naive {1e3 * t_naive[-1]:5.1f} ms, '
          f'cached {1e3 * t_cached[-1]:4.1f} ms per token')
```

On logarithmic axes, a time proportional to $n$ is a line of slope one.

```{.python .input #kv-cache-checking-the-cache-1}
%%tab pytorch, jax
d2l.plot(lengths, [[1e3 * t for t in t_naive], [1e3 * t for t in t_cached]],
         'context length', 'ms per token', legend=['naive', 'cached'],
         xscale='log', yscale='log')
```

At long contexts the naive step grows at least in proportion to $n$, as
$2Nn$ says it must, and faster where the attention scores, whose work grows
with $n^2$, add a sizable share. The cached step grows far more slowly. The
time to read the cache does grow with $n$, but even at 4,000 tokens the
cache is smaller than the weights, and every step also has a fixed cost
that does not depend on $n$. At the longest context the cached step is
more than five times faster, and beyond about a thousand tokens the gap widens with
every doubling of the context. At short contexts the two curves approach
each other and can even cross. A step of either kind then spends much of
its time on fixed costs, such as launching the kernels of twelve blocks,
rather than on arithmetic.
<!-- WRITER: new numbers (direct runs 2026-11-18, both tabs compiled):
cached 0.8 ms at n = 256 to 1.2-1.3 ms at 4,032 in both tabs; naive 2.7
(PyTorch) and 1.6 (JAX) ms at 256, 34 and 53 ms at 4,032. The curves no
longer approach or cross at short contexts: the cached step is 2 to 3 times
faster at 256 and 27 to 41 times at 4,032. "more than five times faster"
goes. The cached step's floor is one read of every weight, about 0.5 ms
(KV-6); a step takes about 0.8 ms at short contexts, and the time grows
with n by the cache read. The fixed-cost explanation (launching kernels)
no longer applies; explain the fixed cost once here (KV-9) as the GPU's
own kernels: matrix-vector products that do not reach the full bandwidth
and about a hundred small kernels per step (profiled: 0.92 ms of GPU time
per step at n = 1,024, 0.46 ms in the block matrix-vector products at
about 740 GB/s, 0.17 ms for the output layer). -->

End to end, we generate 256 tokens after a prompt of 3,584 tokens with
both methods, after one warm-up call each:

```{.python .input #kv-cache-same-logits-measured-3}
%%tab pytorch, jax
prefix = list(range(3584))  # Any token ids will do for timing
for name, gen in (('naive ', model.generate),
                  ('cached', model.generate_cached)):
    t = d2l.Benchmark(lambda: gen(prefix, 256), warmup=1, repeats=1).time
    print(f'{name}: 256 tokens in {t:4.1f} s = {256 / t:5.1f} tokens/s')
```

:begin_tab:`jax`
The naive `generate` of :numref:`subsec_gpt-sampling` keeps its context in
a buffer of fixed shape, so every step here runs the forward pass over all
`max_len` = 4,096 positions, slightly more than the 3,584 to 3,840 tokens
of the context.
:end_tab:

<!-- WRITER: end to end now 8.0 s against 0.46 s in PyTorch (17 times) and
15.3 s against 0.4 s in JAX (38 times); the naive loop is the eager
library `generate` in PyTorch and runs over all 4,096 positions in JAX.
Both are far below the factor of 250 in operations. -->
The cached loop is more than five times faster, though by far less than
the factor of about 250 in operations. Every decoded token still reads all the weights of
the model, as :numref:`subsec_kv-cache-prefill-decode` shows. Serving
systems report the two phases separately: the *time to first token*
measures the prefill, and the *tokens per second* measure the decode loop.

## Memory and Bandwidth
:label:`subsec_kv-cache-memory`

The cache removes the repeated arithmetic, but it has to be stored, and it
is read again at every step, so we start with its size. The size of the
cache follows from the shape of its
buffers. Each of the $L$ blocks stores one key and one value vector of
dimension $H_\textrm{kv} d_h$ per token, for $H_\textrm{kv}$ key--value
heads of width $d_h$, so a batch of $b$ sequences of length $n$ holds

$$
\textrm{cache bytes} \;=\; 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n
\cdot b \cdot (\textrm{bytes per element}),
$$
:eqlabel:`eq_kv-cache-bytes`

where the leading 2 counts keys and values. So far $H_\textrm{kv}$ equals
the number $H$ of query heads; :numref:`subsec_kv-cache-gqa-quality`
breaks that equality. For our GPT-2-sized model in fp32 the cache takes
$2 \cdot 12 \cdot 12 \cdot 64 \cdot 4$ bytes $= 72$ KiB per token. At a
context of 4,096 tokens that is 288 MiB, more than half the size of the
model's weights. A deployed model has dozens of layers, contexts of
hundreds of thousands of tokens, and many users served at once. Then the
cache, not the weights, fills the accelerator; an exercise works this out
for a 70B model. We save the formula for any `d2l.GPT`.

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

The buffers of `init_cache` have this size, so a cache that does not fit
into memory fails when it is allocated, before generation starts.

```{.python .input #kv-cache-the-memory-bill}
%%tab pytorch, jax
bytes_per_param = 4  # fp32 weights
weight_bytes = bytes_per_param * d2l.num_params(model)
ks, vs = init_cache(model, 1, 4096)
print(f'cache buffers {sum(u.nbytes for u in ks + vs) / 2**20:.0f} MiB, '
      f'formula {kv_cache_bytes(model, 4096) / 2**20:.0f} MiB, '
      f'weights {weight_bytes / 2**20:.0f} MiB')
```

### Prefill and Decode
:label:`subsec_kv-cache-prefill-decode`

At each generated token the whole cache is read from memory, and so is
every weight of the model, because a forward pass over one token multiplies
its activations by every weight matrix. Which resource limits a computation,
arithmetic or memory traffic, depends on its *arithmetic intensity*: the
number of floating-point operations per byte moved between memory and the
processor. In the *roofline* model of performance, which
:numref:`chap_performance` develops, a computation takes at least its
operations divided by the peak rate of the processor. It also takes at
least its bytes divided by the memory bandwidth, and the larger of the two
bounds sets its time. An RTX 4090 performs up to about 83 TFLOP/s in fp32 but moves only
about 1 TB/s, so a computation needs about 80 FLOPs per byte to keep the
arithmetic units busy.

![The fp32 roofline of an RTX 4090. Attainable throughput is the smaller of the compute roof, about 83 TFLOP/s, and the arithmetic intensity times the memory bandwidth of about 1 TB/s; the two meet at the ridge point near 82 FLOPs per byte. A decode step of a GPT-2-sized model at a context of 2,048 tokens performs 0.38 FLOPs per byte and sits far down the bandwidth slope; a prefill of 2,048 tokens, at about 780, sits under the roof.](../img/mdl-transformers-roofline.svg)
:label:`fig_kv-cache-roofline`
<!-- WRITER: cite :numref:`fig_kv-cache-roofline` where the ridge point is
named (KV-7); the caption may say that the performance chapter draws the
bf16 roofline of the same card. PyTorch fp32 matmuls now may use tf32
(`torch.set_float32_matmul_precision('high')`, as in the performance
chapter); the RTX 4090's tf32 peak equals its fp32 peak. -->

A decode step performs about $2N$ operations and reads at least the $4N$
bytes of the fp32 weights plus the cache, an intensity below one FLOP per
byte. A prefill over $n$ tokens performs $2Nn$ operations with the same
weight traffic, because each weight is read once and used for every token,
so its intensity is about $n$ times higher. Prefill is therefore limited by
arithmetic and decode by memory bandwidth
:cite:`Pope.Douglas.Chowdhery.ea.2023`. We time a forward pass over a
2,048-token prompt and the decoding of 32 tokens after it.

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch, jax
t_prefill = time_forward(model, 2048)
t_decode = time_decode(model, 1, 2048)
```

We compare both rates with the intensity of a decode step and with the
rate that a bandwidth of 1 TB/s would allow.

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

<!-- WRITER: new numbers: prefill 152,000 (PyTorch) and 120,000 (JAX)
tokens/s, 36 to 45% of the 335,000 that 83 TFLOP/s would allow; decode 960
and 982 tokens/s against the ceiling of 1,550, about 62% of it in both
tabs; the ratio prefill/decode is about 150 and 120, so "about a hundred
times more slowly", not "an order of magnitude" and not "hundreds". The
gap to the ceiling is the GPU's own kernels (see the speedup subsection),
no longer launches from Python. -->
The model reads the prompt at about a hundred thousand tokens per second and
generates tokens at least an order of magnitude more slowly, with the same
layers on the same hardware. The decode rate also stays below the
bandwidth ceiling, because a fixed cost per step, such as launching the
kernels of twelve blocks, adds to the time to read the weights. The bytes
moved per token, weights and cache together, still set the limit that no
implementation can beat. A smaller cache therefore allows faster decoding,
longer contexts, or more sequences at once.

### Batched Decoding
:label:`subsec_kv-cache-batching`

A decode step for one sequence reads all $W$ bytes of the weights to
produce one token. A step for a batch of $b$ sequences reads the weights
once and produces $b$ tokens, because every sequence multiplies its
activations by the same matrices. The cache gains nothing from batching.
Each sequence reads its own $nc$ bytes, where $c$ is the cache size per
token from :eqref:`eq_kv-cache-bytes`. A step therefore moves about
$W + bnc$ bytes. When the memory bandwidth $\beta$ limits the step, the
throughput is

$$
\textrm{tokens per second} \approx \frac{b\beta}{W + bnc}.
$$
:eqlabel:`eq_kv-cache-batch-throughput`

For small $b$ the throughput grows in proportion to $b$. Once the caches
outweigh the weights, at $b > W/(nc)$, it levels off toward $\beta/(nc)$, a
ceiling set by the cache alone. For our fp32 model, $W \approx 0.5$ GB and
$c = 72$ KiB. The caches equal the weights at a batch of about 53 for a
context of 128 tokens, but already at a batch of about 7 for 1,024 tokens. We
measure the decode throughput at both contexts for batches of 1 to 64
sequences and plot it next to the bound of
:eqref:`eq_kv-cache-batch-throughput` with $\beta = 1$ TB/s.

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

The bound keeps growing over the whole range of batches at a context of
128 tokens, since the caches reach the size of the weights only near a
batch of 53. At 1,024 tokens it grows little beyond a batch of 16 and
approaches the ceiling set by the caches. Batching spreads the weight
reads over many sequences, but every sequence still reads its own cache.
At small batches the measured curves stay below the bound, by the fixed
cost per step of the previous subsection.
<!-- WRITER: both tabs now show the bound's shape (n = 1,024: PyTorch 1,110,
2,776, 6,493, 9,889; JAX 1,074, 3,360, 7,143, 9,951 tokens/s at b = 1, 4,
16, 64; bound 1,755 to 12,016). The measured curves sit at about 60% of the
bound at b = 1 and reach about 80% at b = 64, n = 1,024, where the cache
reads dominate. At n = 128 they keep growing (PyTorch 1,183 to 32,414, JAX
1,253 to 36,029). The legend now names only the measured curves; the dotted
lines are the bounds, in the matching colours, and the prose says so. Both
tab notes go (S4): the PyTorch note is false now, the JAX note says what
the shared prose says. -->

:begin_tab:`pytorch`
Here the fixed cost dominates almost the whole range. Each decode step
launches several hundred small kernels from Python, one or more per
operation of every block, and launching them takes longer than the GPU
needs to run them, weight and cache reads included. The time per step
therefore hardly depends on the batch or the context: the measured
throughput grows in proportion to the batch, far below the bound. Only at
64 sequences of 1,024 tokens does the GPU's work, most of it reading the
caches, catch up with the launches.
:numref:`sec_compilation` removes this cost by recording the kernels of a
step once and replaying them (CUDA graphs) or by compiling the step; only
then does the bound of :eqref:`eq_kv-cache-batch-throughput` appear in the
measurement.
:end_tab:

:begin_tab:`jax`
`decode` compiles the whole loop of decode steps, so the fixed cost per step
is small, and the measured curves follow the bound: at 1,024 tokens the
throughput grows little beyond a batch of 16.
:end_tab:

When many users are served, the batch changes from step to step, because
requests arrive and finish at different times and with contexts of
different lengths. Serving systems therefore admit and retire sequences at
every decode step, which is called *continuous batching*
:cite:`Yu.Jeong.Kim.ea.2022`. They also store the cache in fixed-size
blocks located through a table, like the pages of virtual memory, instead
of in one buffer of maximal length per sequence
:cite:`Kwon.Li.Zhuang.ea.2023`. :numref:`sec_model_serving` shows
these engines in use, together with speculative decoding, and the Language
Models part of the book builds their mechanisms.

## Summary

Naive generation runs a full forward pass for every new token, so its
dense-layer cost is quadratic in the length of the text. A KV cache stores
the keys and values of past tokens, which causality leaves unchanged, and
makes that cost linear. It needs one preallocated buffer per layer and the
position offset of each new token, and it reproduces the logits of the full
forward pass to rounding error. The cache holds
$2 L H_\textrm{kv} d_h n b$ elements for $L$ blocks of $H_\textrm{kv}$
key--value heads of width $d_h$, which for long contexts and many
sequences exceeds the weights. Prefill is limited by arithmetic and decode
by memory bandwidth, because every decoded token reads all weights and its
whole cache. Batching shares the weight reads but not the cache reads.
<!-- WRITER: the Summary keeps four sentences on this half; the sentences on GQA, MQA, MLA and windows moved to kv-cache-reduction.md. -->


## Exercises

1. **Cost of generation.** A model with $N$ parameters reads a prompt of
   $P$ tokens and then generates $T$ tokens. Count $2N$ dense-layer FLOPs
   per token of a forward pass.
    1. Count the dense-layer FLOPs of a naive schedule that runs a forward
       pass over the whole history before every new token, and of
       `generate_cached`, which runs one prefill over the prompt and then
       single-token steps. Evaluate the ratio of the two counts for the
       end-to-end run of :numref:`subsec_kv-cache-speedup`, with
       $P = 3584$ and $T = 256$.
    1. The end-to-end run is faster with the cache by much less than this
       ratio of FLOPs. Explain the difference with the arithmetic
       intensities of the two kinds of forward pass, as
       :numref:`subsec_kv-cache-prefill-decode` computes them.

1. [code] **A missing position offset.** `forward_cached` rotates the
   queries and keys of new tokens by their absolute positions.
    1. Drop the offset from the RoPE rotation, so that every decoded token
       is rotated as if it sat at position 0, and rerun the check of
       :numref:`subsec_kv-cache-same-logits` for `pos='rope'`. Compare the
       deviation with that of the correct cache.
    1. Take a decoded query at true position $t$. Determine the relative
       offset that the faulty cache uses against a prompt key at position
       $j$ and against the key of an earlier decoded token. Which
       information about the decoded tokens does the model lose entirely?
    1. Make the corresponding mistake for learned positions, so that every
       decoded token receives the embedding of position 0. Prefill GPT-2
       with the first 512 tokens of the passage `x`, feed the remaining
       tokens one at a time, and compare the loss on positions 512 onward,
       the loss that `tail_loss` measures, for the faulty and the correct
       cache.
       <!-- WRITER: `x`, `tail_loss` and the GPT-2 setup moved to
       :numref:`sec_kv-cache-reduction`; make this part self-contained
       (load GPT-2 with `d2l.load_gpt2`, plain cross-entropy from position
       512). `forward_cached` now takes a tensor position, and the PyTorch
       tab compiles its decode steps. -->

1. **Cache of a 70B model.** Llama-2-70B :cite:`touvron2023llama2` has 80
   layers and 64 query heads of dimension 128, and it uses GQA with 8
   key--value heads. Its weights occupy about 140 GB in 16-bit precision,
   and its cache is stored in 16-bit precision as well.
    1. Use :eqref:`eq_kv-cache-bytes` to compute the cache bytes per
       token, and the context length at which the cache of a single
       sequence reaches the size of the weights.
    1. Repeat both computations for the same model with 64 key--value
       heads, that is, without sharing, and with a single key--value head
       (MQA).
    1. The weights are split across two GPUs with 80 GB each. For each of
       the three configurations, find how many sequences of 4096 tokens fit
       into the memory that the weights leave free, ignoring activations.

1. **Shared weight reads.** ● Use the roofline model of
   :numref:`subsec_kv-cache-prefill-decode`: a forward pass takes at least
   its FLOPs divided by the peak rate of 83 TFLOP/s and at least its bytes
   moved divided by the bandwidth of 1 TB/s. The weights of `model` are
   stored in fp32.
    1. Ignore the cache. A forward pass of `model` over $m$ new tokens
       reads every weight once, whether the tokens belong to $m$ sequences
       of a batch or are $m$ drafted tokens of one sequence. Find the
       largest $m$ for which the pass takes no longer than a single-token
       step. Explain why this bound makes it cheap to check several drafted
       tokens in one pass, as speculative decoding does
       :cite:`Leviathan.Kalman.Matias.2023`.
    1. Now include the cache, which each of $b$ sequences at context $n$
       reads separately. Derive the arithmetic intensity of one decode step
       of `model`, counting the attention FLOPs and the cache bytes.
       Evaluate it at $n = 2048$ for $b = 1$ and in the limit
       $b \to \infty$, and compare both values with the ridge of about 80
       FLOPs per byte. How does the limit change when the $H$ query heads
       share $H_\textrm{kv}$ key--value heads?

[Discussions](https://d2l.discourse.group/t/36776)

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

- Naive `generate`: one forward pass over the whole text per token,
  $\sum_t 2Nt \approx 2.4 \times 10^{14}$ FLOPs.
- Read the prompt once, then each new token once: $\approx 10^{12}$ FLOPs.

::: {.d2l-note}
Causal attention never changes the keys and values of past tokens.
**Store them**, and the repetition disappears.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write a cached forward pass with a position offset and preallocated
  buffers;
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
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| **The KV Cache** | **reuse keys and values while sampling** |
| Shrinking the KV Cache | fewer heads, fewer dimensions, fewer tokens |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT`, causal masks, RoPE, sampling with temperature and
  top-$k$.
- **Skips:** serving engines (paged memory, continuous batching,
  speculative decoding): Language Models part.
:::
:::
:::

::: {.slide title="Causality Freezes the Keys and Values of the Past"}
::: {.cols .vc}
::: {.col}
- Step $t$ needs one new query against $\mathbf{k}_0, \ldots, \mathbf{k}_t$.
- Positions $0, \ldots, t-1$ never see position $t$: their keys and values
  are the same as one step ago.
:::

::: {.col .fig .big}
@fig:mdl-transformers-kv-cache
:::
:::
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="pytorch"}
```{.python #kv-cache-exercises-1}
H = model.token_emb(X)                 # T new tokens at pos = t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = blk.attention.qkv(blk.norm1(H), t)        # RoPE offset t
    ks[i][:, :, pos], vs[i][:, :, pos] = k, v           # write in place
    S = q.unflatten(1, (H_kv, -1)) @ ks[i][:, :, None].transpose(-1, -2)
    Y = d2l.masked_softmax(S / math.sqrt(d_h), mask) @ vs[i][:, :, None]
    H = H + blk.attention.W_o(Y.flatten(1, 2).transpose(1, 2).flatten(2))
    H = H + blk.ffn(blk.norm2(H))
```

- Buffers of fixed size; the mask hides positions not yet written, so every
  decode step has the same shapes.
- `torch.compile` records the step once and replays it as a CUDA graph.
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="jax"}
```{.python #kv-cache-exercises-2}
H = model.token_emb(X)                     # T new tokens at t, t+1, ...
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

::: {.slide title="The Cached Step Grows Far More Slowly Than the Naive One"}
@!kv-cache-checking-the-cache-1

124M-parameter model, untrained: the arithmetic does not depend on the
weights. Both steps are compiled, so the time is the GPU's work. Naive grows
at least like $2Nn$; cached adds the cache read to the weight read.
:::

::: {.slide title="Check: Operations of Naive and Cached Generation"}
[Check]{.kicker}

$N$ = 124M, prompt $P$ = 3,584 tokens, $T$ = 256 new tokens, $2N$ FLOPs per
token of a forward pass. How many FLOPs does each schedule need?

::: {.fragment .fade-up}
- Naive: $2N \sum_{t=P+1}^{P+T} t \approx 2 \cdot 1.24 \times 10^8 \cdot
  256 \cdot 3712 \approx 2.4 \times 10^{14}$.
- Cached: $2N (P + T) \approx 9.5 \times 10^{11}$, about 250 times less.
:::
:::

::: {.slide title="The Cache Grows with Layers, Heads, Tokens, and Sequences"}
$$\textrm{cache bytes} = 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n \cdot b \cdot (\textrm{bytes per element})$$

$L$ blocks, $H_\textrm{kv}$ key--value heads of width $d_h$, $n$ tokens, $b$
sequences.


- GPT-2 small, fp32: $2 \cdot 12 \cdot 12 \cdot 64 \cdot 4$ B = 72 KiB per
  token.
- 4,096 tokens: 288 MiB, more than half of the 470 MiB of weights.

@!kv-cache-the-memory-bill
:::

::: {.slide title="A Decode Step Reads the Whole Model for One Token"}
Arithmetic intensity = FLOPs per byte moved. An RTX 4090: 83 TFLOP/s,
1 TB/s, so the *ridge point* lies near 80 FLOPs per byte.

| step | FLOPs | bytes | FLOPs per byte |
|---|---|---|---|
| decode, 1 token | $2N$ | $4N$ + cache | below 1 |
| prefill, $n$ tokens | $2Nn$ | $4N$ + cache | about $n/2$, less with the cache |

::: {.d2l-note}
Prefill is limited by arithmetic, decode by memory bandwidth.
:::
:::

::: {.slide title="Decode Sits Far Down the Bandwidth Slope"}
::: {.cols .vc}
::: {.col .fig .big}
@fig:mdl-transformers-roofline
:::

::: {.col .narrow}
- Decode step: 0.38 FLOP per byte, 200 times below the ridge.
- Prefill of 2,048 tokens: about 780, under the compute roof.
:::
:::
:::

::: {.slide title="Prefill Reads Over 100,000 Tokens per Second, Decode About 1,000"}
@!kv-cache-prefill-and-decode-1

- Same layers, same GPU: decoding is about a hundred times slower.
- Decode reaches about 60% of the bandwidth ceiling; the rest is the GPU's
  own kernels, not Python.
:::

::: {.slide title="Batching Amortizes Weights but Not the KV Cache"}
$b$ sequences, weights $W$ bytes, cache $c_\textrm{kv}$ bytes per token,
context $n$, bandwidth $\beta$:

$$\textrm{tokens per second} \approx \frac{b\beta}{W + bnc_\textrm{kv}}
\;\xrightarrow{\;b \to \infty\;}\; \frac{\beta}{nc_\textrm{kv}}$$

- GPT-2 small, fp32: $W \approx$ 0.5 GB, $c_\textrm{kv}$ = 72 KiB.
- Caches equal the weights at $b \approx 52$ for $n = 128$, but at
  $b \approx 6.5$ for $n = 1024$.
:::

::: {.slide title="Long Contexts Cap the Gain from Batching"}
@!kv-cache-batched-decoding

Solid: measured. Dotted: the bandwidth bound at 1 TB/s. At $n = 1024$ the
throughput levels off near 10,000 tokens/s beyond $b = 16$; at $n = 128$ it
keeps growing.
:::

::: {.slide title="Check: Where the Cache Outweighs the Weights"}
[Check]{.kicker}

GPT-2 small in fp32 at a context of $n = 4096$ tokens. At which batch size
does the cache equal the weights, and what is the ceiling on tokens per
second at 1 TB/s?

::: {.fragment .fade-up}
- $nc_\textrm{kv} = 4096 \cdot 72$ KiB = 288 MiB per sequence, so
  $b \approx 500 / 302 \approx 1.7$.
- Ceiling $\beta / (nc_\textrm{kv}) \approx 10^{12} / (3.0 \times 10^8)
  \approx 3{,}300$ tokens per second, whatever the batch.
:::
:::

::: {.slide title="Check: The Cache of a 70B Model"}
[Check]{.kicker}

Llama-2-70B: 80 layers, heads of width 128, 8 key--value heads, 16-bit
cache. How many bytes per token, and at which context does one sequence's
cache reach the 140 GB of weights?

::: {.fragment .fade-up}
- $2 \cdot 80 \cdot 8 \cdot 128 \cdot 2$ B = 320 KiB per token.
- $140 \times 10^9 / 327{,}680 \approx 430{,}000$ tokens; without GQA
  (64 key--value heads), about 53,000.
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Causality lets a cache replace $\approx 2.4 \times 10^{14}$ FLOPs by
  $\approx 10^{12}$ for 256 tokens after 3,584, with the same logits to
  rounding error.
- GPT-2 small caches 72 KiB per token in fp32: 288 MiB at 4,096 tokens.
- Prefill is limited by arithmetic, decode by bandwidth; batching shares
  the weight reads, not the cache reads.

**Next:** shrinking the cache: fewer key--value heads, fewer dimensions,
fewer tokens.
:::
