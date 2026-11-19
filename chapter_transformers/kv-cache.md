# The KV Cache
:label:`sec_kv-cache`

A chat assistant writes its answer one token at a time, and so does the
GPT of :numref:`sec_gpt`. Its `generate` method produces each new token by
running the model over the whole text so far. An assistant built that way
would answer a long question by reading the whole question again for every
word of its answer. Does a model have to reprocess everything before it to
choose the next token?

It does not. Attention is causal, so the keys and values of earlier tokens
never change once computed. The *KV cache* stores them, and each step then
runs the model on the new token alone. For a long prompt this removes all
but a small fraction of the arithmetic.

This section builds the cache, checks it, and measures what it saves. The
saving turns out to be smaller than the count of operations promises, and
the reason is the second subject of the section. Once the cache is in
place, generation is limited by memory rather than by arithmetic, because
every new token reads all the weights of the model and its whole cache. We
derive the size of the cache, measure how much faster a model reads a
prompt than it writes text, and see why serving many users at once helps
less as their contexts grow. Every model in this section is an untrained
`d2l.GPT` of :numref:`sec_gpt`: time and memory do not care what the
weights have learned.

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

Let us first count what naive generation costs. A forward pass of a model
with $N$ parameters over $n$ tokens costs about $2Nn$ floating-point
operations, because every parameter takes part in one multiplication and
one addition per token. Attention adds work that grows faster
(:numref:`sec_attention-at-scale`): a token at context length $n$ computes
$n$ scores and a weighted sum of $n$ values, about $4nd$ operations in each
of $L$ blocks of width $d$. This term is not small: for a model of GPT-2's
shape at about 4,000 tokens, $4ndL$ is about 60% of the $2N$ of the dense
layers. We leave it out of the count all the same, because it hardly
changes the comparison we are after. The cached schedule below computes
attention over the prompt as well, and both counts grow by about the same
fraction. Naive generation calls the forward pass once per new token, over
a history that grows by one token per step. Producing $T$ tokens therefore
costs about $\sum_{n=1}^{T} 2Nn \approx NT^2$ operations, quadratic in the
length of the text, and each pass uses only the last row of the logits it
computes.

Most of that work is repeated. We count positions from 0. At step $t$ the
new token sits at position $t$, and every layer needs the attention output
for its query $\mathbf{q}_t$ against the keys $\mathbf{k}_0, \ldots,
\mathbf{k}_t$ and values $\mathbf{v}_0, \ldots, \mathbf{v}_t$. The causal
mask guarantees that positions $0, \ldots, t-1$ never see position $t$, so
their hidden states, and therefore their keys and values, are the same as
one step earlier. A KV cache stores, for every layer, the keys and values
computed so far. Each step then runs the model on the one new token, writes
its key and value into the cache, and attends to everything stored.
:numref:`fig_kv-cache` contrasts the two schedules.

A cached step costs about $2N$ operations, a forward pass over a single
token, plus the scores of one query against the cache. To see what this
saves, take a model of GPT-2's size, $N = 124$ million, that answers a long
question of $P = 3{,}584$ tokens with $T = 256$ tokens, a case that stays
within the 4,096 positions of the model we time below. The naive schedule
runs forward passes over $3{,}584, 3{,}585, \ldots, 3{,}839$ tokens, in all

$$
2N \sum_{n=3584}^{3839} n \;\approx\; 2N \cdot 256 \cdot 3712
\;\approx\; 2.4 \times 10^{14}
$$

operations. The cached schedule reads the prompt once and each new token
once, about $2N(P + T) \approx 9.5 \times 10^{11}$ operations. The cache
saves a factor of about 250, close to the number of generated tokens, and
for a good reason: a single naive step over the whole text costs about as
much as the entire cached run.

![One decoding step at position $t$. Naive generation recomputes every key, value, and score row of the prefix and then uses only the last row. With a KV cache, the past keys and values are read from memory, and the step computes a single new key, value, and score row.](../img/mdl-transformers-kv-cache.svg)
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
of values with room for `max_len` positions and a slot for every key--value
head, of which a model may have fewer than query heads
(:numref:`subsec_kv-head-sharing`). A call writes the keys and values of
its new tokens into the next free positions, starting at $t$, and attends
to all filled positions.

The same function serves both phases of generation. In *prefill*, the
whole prompt enters at $t = 0$, and its queries need the causal mask. In
*decode*, one token at position $t$ attends to all filled positions. One
mask covers both: the query at position $t + i$ may read the buffer
positions $0, \ldots, t + i$. This is the causal mask of
:numref:`sec_attention-scoring-functions` laid over the whole buffer, and
it also hides the positions not yet written. Attending over the whole
buffer rather than over its filled part wastes a little work. In exchange,
a decode step has the same shapes at every position, so that we can compile
it once and reuse it for every token. The mask is the one line of the code
below that needs thought. Everything else is the forward pass of `d2l.GPT`,
written out block by block.

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
The buffers are written in place by index, and `init_cache` gives each a
fixed address so that the compiled step below can write into it. We write
attention out with `d2l.masked_softmax` because
`F.scaled_dot_product_attention` is slow for a single query over a long
masked buffer.
:end_tab:

:begin_tab:`jax`
JAX arrays cannot be changed in place, so `forward_cached` returns new
buffers made by `dynamic_update_slice`; inside the compiled loop below, XLA
turns these updates into writes in place.
:end_tab:

Generation is a prefill call over the prompt followed by one decode call
per new token. `sample` draws a token from the last logits with the
temperature and top-$k$ truncation of :numref:`subsec_gpt-sampling`,
repeating two lines of `generate`. `decode` feeds each token back, and
`generate_cached` gives the whole loop the interface of `generate`. The
decode step is compiled once and replayed for every token, so that its
time is set by the work of the GPU rather than by Python's.

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
`decode` compiles the prefill and the whole loop of decode steps into one
function, which receives the model as the two parts that `nnx.split`
returns. It *donates* the cache buffers to XLA, which may then reuse their
memory, so the arrays passed in must not be used again.
:end_tab:

### Checking the Cache
:label:`subsec_kv-cache-same-logits`

A cache that is fast but wrong is worse than none, so before timing
anything we check that the cached path reproduces the logits of the full
forward pass. A small random model suffices. For both positional schemes,
we prefill the first 16 tokens of a random sequence, decode the remaining
32 one at a time, and compare the logits after each position with those of
one forward pass over the whole sequence. Every model of this section is
built through a shorthand, `gpt`, that gives it fresh random weights.

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
We set matrix products to full float32 precision here, since by default
JAX multiplies float32 matrices on this GPU in the coarser TF32 format,
whose rounding would hide that the two paths compute the same values.
:end_tab:

The largest deviation is about $10^{-6}$ for both positional schemes, which
is rounding error in float32: the two paths compute the same logits.

### Measuring the Speedup
:label:`subsec_kv-cache-speedup`

How much faster is the cache in practice? We time one generation step at
growing context lengths, on an untrained model of GPT-2's size: 12 blocks
of width 768, 124 million parameters.

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

Two helpers time the two kinds of step, both compiled. `time_forward` times
the naive step, a forward pass over all $n$ tokens. `time_decode` times one
decode step at context $n$, averaged over 32 steps; it takes a batch size,
which :numref:`subsec_kv-cache-batching` varies. Both use the `d2l.Benchmark`
of :numref:`subsec_async_dispatch`, which warms up, waits for the device,
and averages ten calls.

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

What should we expect? A naive step over $n$ tokens performs $2Nn$
operations, so once it has enough tokens to keep the arithmetic units busy,
its time should grow at least in proportion to $n$. A cached step performs
only $2N$, but it has to read every weight from memory once: half a
gigabyte of fp32 weights, at the roughly 1 TB/s of an RTX 4090, takes about
half a millisecond, however short the context. That is a floor, and a real
step can only be slower. We time both steps at five context lengths.

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

The naive curve bears out its prediction only from about a thousand tokens
on. Below that it rises more slowly than the context, because part of its
time does not depend on $n$ at all: a naive step, too, reads every weight
and runs every small operation of the model, which is all that a cached
step does. Beyond about a thousand tokens it grows faster than the
context, since the attention scores, whose work grows with $n^2$, add
their share, and at 4,032 tokens it takes tens of milliseconds. The cached
step never falls below its floor. It stays between 0.8 and 1.3 ms over the
whole range, and what grows is the time to read the cache, which even at
4,032 tokens is smaller than the weights. At 4,032 tokens the cached step
is more than twenty times faster than the naive one; at 256 tokens the gap
shrinks to two or three times.

Even the shortest cached step takes 0.8 to 0.9 ms, not the half
millisecond that reading the weights requires. The difference is a *fixed
cost per step*, which does not depend on the context, and it has two
sources. First, the weight reads themselves run below the peak bandwidth,
because the products of a single token are matrix--vector products. Second,
every small operation between them, a normalization, a softmax, an
addition, runs as a kernel of its own that takes microseconds whatever its
size. Python is not the cause: the step is compiled, and what Python still
does between steps overlaps with the work of the GPU.

The fixed cost depends on the GPU and the implementation, and better software
shrinks it. The weight read does not shrink. It is a hard floor: no step that
produces one token of this model in fp32 can run faster on this GPU. The only
way around it is to let one read of the weights serve several tokens, which is
what batching (:numref:`subsec_kv-cache-batching`) and speculative decoding
(Exercise :exref:`ex_kv-cache-shared-reads`) do.

The end-to-end run is the case we counted at the start: 256 tokens after
a prompt of 3,584 tokens, with each method, after one warm-up call.

```{.python .input #kv-cache-same-logits-measured-3}
%%tab pytorch, jax
prefix = list(range(3584))  # Any token ids will do for timing
for name, gen in (('naive ', model.generate),
                  ('cached', model.generate_cached)):
    t = d2l.Benchmark(lambda: gen(prefix, 256), warmup=1, repeats=1).time
    print(f'{name}: 256 tokens in {t:4.1f} s = {256 / t:5.1f} tokens/s')
```

:begin_tab:`pytorch`
The naive loop is the `generate` of :numref:`subsec_gpt-sampling`, which
runs uncompiled. At these lengths the work of the GPU dominates each step,
so this changes the time only a little.
:end_tab:

:begin_tab:`jax`
The naive `generate` of :numref:`subsec_gpt-sampling` runs every step over
its fixed buffer of 4,096 positions, slightly more than the context.
:end_tab:

The cached loop is more than ten times faster, far less than the factor of
250 in operations. Each of its 256 steps costs at least one read of the
weights, however few operations it performs, while a naive step over
thousands of tokens has enough arithmetic to keep the GPU busy. Operations
are therefore a poor measure of time, and
:numref:`subsec_kv-cache-prefill-decode` names a better one.

## Memory and Bandwidth
:label:`subsec_kv-cache-memory`

The cache removes the repeated arithmetic, but it has to be stored, and it
is read again at every step. How large is it? The answer follows from the
shape of its buffers. Each of the $L$ blocks stores one key and one value
vector of dimension $H_\textrm{kv} d_h$ per token, for $H_\textrm{kv}$
key--value heads of width $d_h$, so a batch of $b$ sequences of length $n$
holds

$$
\textrm{cache bytes} \;=\; 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n
\cdot b \cdot (\textrm{bytes per element}),
$$
:eqlabel:`eq_kv-cache-bytes`

where the leading 2 counts keys and values. The cache is linear in every
factor. Two of them, the length of the context and the batch, are set by
how the model is used rather than by its design. In our GPT every query
head has its own key--value head, so $H_\textrm{kv}$ equals the number $H$
of query heads; letting groups of query heads share one key--value head
(:numref:`subsec_kv-head-sharing`) shrinks this factor. For our
GPT-2-sized model in fp32 the cache takes
$2 \cdot 12 \cdot 12 \cdot 64 \cdot 4$ bytes $= 72$ KiB per token. At a
context of 4,096 tokens that is 288 MiB, more than half the size of the
model's weights. A deployed model has dozens of layers, contexts of a
hundred thousand tokens and more, and many users served at once. Then the
caches, not the weights, fill the accelerator, as Exercises
:exref:`ex_kv-cache-70b` and :exref:`ex_kv-cache-70b-batch` show for a 70B
model. We save the formula for any `d2l.GPT`.

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

`init_cache` allocates the whole cache at once, so a cache that does not
fit into memory fails before generation starts, not halfway through an
answer. Its buffers should have the size of the formula, and we compare the
two for a context of 4,096 tokens.

```{.python .input #kv-cache-the-memory-bill}
%%tab pytorch, jax
bytes_per_param = 4  # fp32 weights
weight_bytes = bytes_per_param * d2l.num_params(model)
ks, vs = init_cache(model, 1, 4096)
print(f'cache buffers {sum(u.nbytes for u in ks + vs) / 2**20:.0f} MiB, '
      f'formula {kv_cache_bytes(model, 4096) / 2**20:.0f} MiB, '
      f'weights {weight_bytes / 2**20:.0f} MiB')
```

Formula and allocation agree. At 288 MiB a sequence, a 24 GB card holds
fewer than a hundred such caches next to the weights of this small model,
so at long contexts the cache, not the model, decides how many users one
GPU can serve.

### Prefill and Decode
:label:`subsec_kv-cache-prefill-decode`

At each generated token the whole cache is read from memory, and so is
every weight of the model, because a forward pass over one token multiplies
its activations by every weight matrix. A decode step is, to a good
approximation, a read of the whole model from memory with a little
arithmetic on the side. A naive step is the other extreme. It is a prefill
of the whole text, the same computation that reads a prompt, and the
2,048-token point of our timing plot measured one of each. To say why the
two behave so differently we need one idea from the study of performance.

Which resource limits a computation, arithmetic or memory traffic, depends
on its *arithmetic intensity*: the number of floating-point operations it
performs per byte it moves between memory and the processor. A product of a
matrix with one vector uses each entry it reads once; a product of the same
matrix with a thousand vectors uses each entry a thousand times, for the
same traffic.

The *roofline* model turns this into two lower bounds on the time of a
computation: its operations divided by the peak arithmetic rate, and its bytes
divided by the memory bandwidth $\beta$. The larger bound sets the time. The
model is crude, since it ignores caches, latency, and kernels that do little
of either, but it asks the right first question of any computation: is it
waiting for arithmetic or for memory?

An RTX 4090 performs up to about 83 TFLOP/s in fp32, and as many in the
TF32 format that our timed matrix products may use, but it moves only about
1 TB/s. The two bounds are equal at about 80 FLOPs per byte, the *ridge
point* of this GPU. A computation below the ridge point waits for memory;
one above it keeps the arithmetic units busy.

Where do the two phases sit? A decode step performs about $2N$ operations
and reads at least the $4N$ bytes of the fp32 weights plus the cache, an
intensity below half a FLOP per byte. A prefill over $n$ tokens performs
$2Nn$ operations with the same weight traffic, because each weight is read
once and used for every token, so its intensity is about $n$ times higher.
At a context of 2,048 tokens the decode step reaches about 0.38 FLOPs per
byte, some two hundred times below the ridge point. The prefill reaches
about 780, ten times above it, although this count leaves out the
activations that a prefill also moves and is therefore an upper estimate.
:numref:`fig_kv-cache-roofline` marks both on the roofline of this GPU.
Prefill is limited by arithmetic and decode by memory bandwidth
:cite:`Pope.Douglas.Chowdhery.ea.2023`.

![The fp32 roofline of an RTX 4090. Attainable throughput is the smaller of the compute roof, about 83 TFLOP/s, and the arithmetic intensity times the memory bandwidth of about 1 TB/s; the two meet at the ridge point of about 80 FLOPs per byte. A decode step of a GPT-2-sized model at a context of 2,048 tokens performs 0.38 FLOPs per byte and sits far down the bandwidth slope; a prefill of 2,048 tokens, at about 780 counting weights and cache only, sits under the roof.](../img/mdl-transformers-roofline.svg)
:label:`fig_kv-cache-roofline`

The two bounds also predict rates. Arithmetic allows a prefill of at most
$83 \times 10^{12} / 2N \approx 335{,}000$ tokens per second. Bandwidth
allows a decode of at most $\beta$ divided by the bytes of one step, the
494 MB of weights and the 151 MB of cache at 2,048 tokens, which comes to
about 1,550 tokens per second. We time the same two steps as at the
2,048-token point of the timing plot: a forward pass over a 2,048-token
prompt, and the decoding of 32 tokens after it.

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch, jax
t_prefill = time_forward(model, 2048)
t_decode = time_decode(model, 1, 2048)
```

We print both rates next to the bandwidth ceiling and the two intensities.

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

The model reads the prompt at well over a hundred thousand tokens per
second and writes text at about a thousand, more than a hundred times more
slowly, with the same layers on the same GPU. This is why serving systems
report the two phases separately: the *time to first token* is set by the
prefill, and the tokens per second that follow by the decode loop.

Neither phase reaches its bound. The prefill runs at between a third and a
half of the arithmetic peak, and the decode at about 60% of the bandwidth
ceiling, where the fixed cost per step of :numref:`subsec_kv-cache-speedup`
accounts for the rest. Each phase misses its bound by less than a factor of
three, while the two phases differ by more than a factor of a hundred: the
roofline gets the large number right. Since the bytes that a decode step
moves, weights and cache together, set a limit that no implementation can
beat, a smaller cache allows faster decoding, longer contexts, or more
sequences at once.

### Batched Decoding
:label:`subsec_kv-cache-batching`

A server answers many users at once, and the floor of a decode step
suggests how it should do so. A step for one sequence reads all $W$ bytes
of the weights to produce one token. A step for a batch of $b$ sequences
reads the weights once and produces $b$ tokens, because every sequence
multiplies its activations by the same matrices. The cache gains nothing
from batching: each sequence reads its own $nc_\textrm{kv}$ bytes, where
$c_\textrm{kv}$ is the cache size per token from :eqref:`eq_kv-cache-bytes`.
A step therefore moves about $W + bnc_\textrm{kv}$ bytes. When the memory
bandwidth $\beta$ limits the step, the throughput is

$$
\textrm{tokens per second} \approx \frac{b\beta}{W + bnc_\textrm{kv}}.
$$
:eqlabel:`eq_kv-cache-batch-throughput`

Batching spreads the weight reads over many sequences, but every sequence
still reads its own cache. For small $b$ the throughput grows in proportion to
$b$. Once the caches outweigh the weights, at $b > W/(nc_\textrm{kv})$, it
levels off toward $\beta/(nc_\textrm{kv})$, a ceiling set by the cache alone.
For our fp32 model, $W = 494$ MB and $c_\textrm{kv} = 72$ KiB. The caches
equal the weights at a batch of about 52 for a context of 128 tokens, but
already at a batch of about 6.5 for 1,024 tokens. At 1,024 tokens the ceiling
is about 13,000 tokens per second, however many sequences we batch. We measure
the decode throughput at both contexts for batches of 1 to 64 sequences and
plot it next to the bound of :eqref:`eq_kv-cache-batch-throughput` with
$\beta = 1$ TB/s.

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

The solid lines are the measurements and the dotted lines the bounds, in
matching colors. At 128 tokens both keep rising over the whole range, and
64 sequences decode about 28 times as fast as one, since the caches reach
the size of the weights only near a batch of 52. At 1,024 tokens the
curves bend as predicted: from 16 sequences to 64 the batch grows fourfold
and the throughput only by a factor of 1.4 to 1.5, to about 10,000 tokens
per second. The measured curves stay below the bounds by the fixed cost
per step, at about 60% of the bound for a single sequence and about 80% for
64 sequences of 1,024 tokens, where reading the caches dominates the step.

A real server does not keep a fixed batch, since requests arrive and
finish at different times. It admits and retires sequences at every decode
step, which is called continuous batching :cite:`Yu.Jeong.Kim.ea.2022`, and
it stores each cache in fixed-size pages rather than in one buffer of
maximal length :cite:`Kwon.Li.Zhuang.ea.2023`. A cache can even outlive
its request: when many requests begin with the same long system prompt, a
server that keeps the cache of that prompt prefills it only once
:cite:`Zheng.Yin.Xie.ea.2024`. :numref:`sec_model_serving` shows these
techniques at work in a serving system.

At long contexts, then, serving is a memory problem twice over: the caches
decide how many sequences fit into memory, and they decide how fast those
sequences decode. Three factors of :eqref:`eq_kv-cache-bytes` are choices
of the model's designer: the number of key--value heads, the width stored
per token, and the number of tokens kept, which a window can make smaller
than the context. :numref:`sec_kv-cache-reduction` asks how far each can
shrink before the model notices.

## Summary

A KV cache stores the keys and values of past tokens, which causality
leaves unchanged, so each step of generation runs the model on the new
token alone and still reproduces the full logits to rounding error. The
cache holds $2 L H_\textrm{kv} d_h n b$ elements for $L$ blocks of
$H_\textrm{kv}$ key--value heads of width $d_h$, 72 KiB per token for a
model of GPT-2's size in fp32, and at long contexts and large batches it
outgrows the weights. Prefill uses each weight it reads for every token of
the prompt and is limited by arithmetic. Decode reads all weights and its
cache for a single token and is limited by bandwidth, so that no decode
step can be faster than one read of the weights; on our GPU the model
writes text more than a hundred times more slowly than it reads a prompt.
Batching shares the weight reads among sequences but not the cache reads,
so at long contexts the throughput levels off at the bandwidth divided by
the cache of one sequence.


## Exercises

1. **Cost of generation.** A model with $N$ parameters reads a prompt of
   $P$ tokens and then generates $T$ tokens. Count $2N$ dense-layer FLOPs
   per token of a forward pass.
    1. Count the dense-layer FLOPs of a naive schedule that runs a forward
       pass over the whole history before every new token, and of
       `generate_cached`, which runs one prefill over the prompt and then
       single-token steps. Show that their ratio is about
       $T(P + T/2)/(P + T)$, which gives the factor of 250 of
       :numref:`subsec_kv-cache-speedup` for $P = 3584$ and $T = 256$.
       What does the ratio approach for $T \ll P$, and for $P = 0$? For
       which $T$ does the cache save less than a factor of 10, whatever
       the prompt?
    1. Use the two bounds of the roofline model of
       :numref:`subsec_kv-cache-prefill-decode`, a peak of 83 TFLOP/s and a
       bandwidth of 1 TB/s, with fp32 weights and the cache of
       :eqref:`eq_kv-cache-bytes`, to find a lower bound on the time of
       each schedule for the same run. Compare the ratio of the two bounds
       with the ratio of FLOPs and with the measured ratio. Which of the
       two schedules runs further from its bound, and why might it?

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
       decoded token receives the embedding of position 0. Load the
       released GPT-2 with `d2l.load_gpt2()`, which uses learned
       positions, and encode the first 1,024 tokens of *The Time Machine*
       with `d2l.gpt2_tokenizer()`. Prefill the first 512 tokens, feed the
       remaining ones one at a time, and compare the mean cross-entropy of
       the predictions of the tokens at positions 513 to 1,023 for the
       faulty and the correct cache.

1. **Cache of a 70B model.** Llama 3.1 70B
   :cite:`Grattafiori.Dubey.Jauhri.ea.2024` has 80 layers and 64 query
   heads of dimension 128, and it shares its keys and values among them
   with $H_\textrm{kv} = 8$ key--value heads. Its weights occupy about
   140 GB in 16-bit precision, its cache is stored in 16-bit precision as
   well, and it accepts contexts of up to 131,072 tokens.
   :exlabel:`ex_kv-cache-70b`
    1. Use :eqref:`eq_kv-cache-bytes` to compute the cache bytes per
       token, and the context length at which the cache of a single
       sequence reaches the size of the weights. Can a single sequence get
       there?
    1. Repeat both computations for the same model with
       $H_\textrm{kv} = 64$, that is, without sharing, and with
       $H_\textrm{kv} = 1$.

1. **Batching a 70B model.** Serve the model of Exercise
   :exref:`ex_kv-cache-70b` on eight GPUs with 80 GB each, whose memory
   bandwidths add up to $\beta = 25$ TB/s, at contexts of
   $n = 32{,}768$ tokens.
   :exlabel:`ex_kv-cache-70b-batch`
    1. Use :eqref:`eq_kv-cache-batch-throughput` to find the batch at
       which the caches equal the weights, the throughput of a single
       sequence, and the ceiling that the throughput approaches for large
       batches.
    1. Ignoring activations, how many sequences fit into the memory that
       the weights leave free? What throughput does that batch reach, as a
       fraction of the ceiling? Which limit binds first, memory or
       bandwidth?
    1. Repeat both parts for $H_\textrm{kv} = 64$. By what factor does
       sharing keys and values raise the throughput of the whole server,
       and through which of the two limits does it act?

1. **Shared weight reads.** ● Use the roofline model of
   :numref:`subsec_kv-cache-prefill-decode`: a forward pass takes at least
   its FLOPs divided by the peak rate of 83 TFLOP/s and at least its bytes
   moved divided by the bandwidth of 1 TB/s. The weights of `model` are
   stored in fp32.
   :exlabel:`ex_kv-cache-shared-reads`
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
       $b \to \infty$, and compare both values with the ridge point of
       about 80 FLOPs per byte. How does the limit change when the $H$
       query heads share $H_\textrm{kv}$ key--value heads?

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
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| **The KV Cache** | **reuse keys and values while sampling** |
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
