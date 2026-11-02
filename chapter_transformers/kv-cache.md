# Generation and the KV Cache
:label:`sec_kv-cache`

The `generate` method of :numref:`sec_gpt` produces each new token by
running the model over the whole text so far. A forward pass costs about
$2N$ floating-point operations per token for a model with $N$ parameters,
so for a model of GPT-2's size, with 124 million parameters, writing 256
tokens after a prompt of 3,584 tokens takes about $2.4 \times 10^{14}$
operations. Reading the prompt once and then processing each new token once
would take about $10^{12}$, some 250 times less. The difference is
repetition. Attention is causal, so the keys and values of earlier tokens
never change once computed, and the *KV cache* stores them instead of
recomputing them.

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

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch.nn import functional as F
```

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
from functools import partial
import jax
from jax import numpy as jnp
import math
import optax
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
of one query against $t$ keys.

![One decoding step at context length $t$. Naive generation recomputes every key, value, and score row of the prefix and then uses only the last row. With a KV cache, the past keys and values are read from memory, and the step computes a single new key, value, and score row.](../img/mdl-transformers-kv-cache.svg)
:label:`fig_kv-cache`

### The Cached Forward Pass
:label:`subsec_kv-cache-cached-forward`

Two details need care. The first is position. RoPE rotates queries and keys
by their absolute positions, so a step must know the position $t$ at which
its new tokens start. `GPT.CausalAttention.qkv` takes this offset as an
argument, and a model with learned positions looks up the embeddings of
positions $t, t+1, \ldots$ instead of $0, 1, \ldots$. The second is storage.
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

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab pytorch
def init_cache(model, batch_size, max_len):
    """Zeroed keys and values for every block: two tensors of shape
    (blocks, batch, kv heads, max_len, head width)."""
    attn = model.blks[0].attention
    shape = (len(model.blks), batch_size, attn.num_kv_heads, max_len,
             attn.kv_hiddens // attn.num_kv_heads)
    p = next(model.parameters())
    return (torch.zeros(shape, device=p.device, dtype=p.dtype),
            torch.zeros(shape, device=p.device, dtype=p.dtype))

@torch.inference_mode()
def forward_cached(model, X, ks, vs, t):
    """Logits after the last of the new tokens X, which sit at positions
    t, t + 1, ...; writes their keys and values into the cache."""
    T = X.shape[1]
    H = model.token_emb(X)
    if model.pos == 'learned':
        H = H + model.pos_emb(torch.arange(t, t + T, device=X.device))
    mask = d2l.causal_mask(T, t + T, X.device)
    for i, blk in enumerate(model.blks):  # Pre-norm, as in d2l.GPT
        attn = blk.attention
        q, k, v = attn.qkv(blk.norm1(H), t)  # Rotated at positions t, ...
        ks[i, :, :, t:t + T], vs[i, :, :, t:t + T] = k, v
        Y = F.scaled_dot_product_attention(
            q, ks[i, :, :, :t + T], vs[i, :, :, :t + T], attn_mask=mask,
            enable_gqa=attn.num_kv_heads < attn.num_heads)
        H = H + attn.W_o(Y.transpose(1, 2).flatten(2))
        H = H + blk.ffn(blk.norm2(H))
    return F.linear(model.norm(H[:, -1]), model.token_emb.weight), ks, vs
```

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab jax
def init_cache(model, batch_size, max_len):
    """Zeroed keys and values for every block: two arrays of shape
    (blocks, batch, max_len, kv heads, head width)."""
    attn = model.blks[0].attention
    shape = (len(model.blks), batch_size, max_len, attn.num_kv_heads,
             attn.kv_hiddens // attn.num_kv_heads)
    return jnp.zeros(shape), jnp.zeros(shape)

def forward_cached(model, X, ks, vs, t):
    """Logits after the last of the new tokens X, which sit at positions
    t, t + 1, ...; writes their keys and values into the cache."""
    B, T = X.shape
    H = model.token_emb(X)
    if model.pos == 'learned':
        H = H + model.pos_emb(t + jnp.arange(T))
    # Query t + i sees the filled positions 0, ..., t + i of the buffer
    mask = jnp.arange(ks.shape[2]) <= t + jnp.arange(T)[:, None]
    for i, blk in enumerate(model.blks):  # Pre-norm, as in d2l.GPT
        attn = blk.attention
        q, k, v = (u.swapaxes(1, 2) for u in attn.qkv(blk.norm1(H), t))
        ks = jax.lax.dynamic_update_slice(ks, k[None], (i, 0, t, 0, 0))
        vs = jax.lax.dynamic_update_slice(vs, v[None], (i, 0, t, 0, 0))
        Y = jax.nn.dot_product_attention(q, ks[i], vs[i], mask=mask)
        H = H + attn.W_o(Y.reshape(B, T, -1))
        H = H + blk.ffn(blk.norm2(H))
    return model.token_emb.attend(model.norm(H[:, -1])), ks, vs
```

:begin_tab:`pytorch`
The buffers are written in place, so `forward_cached` returns the same `ks`
and `vs` that it received; returning them keeps one calling convention for
the functions below. The slice `ks[i, :, :, :t + T]` is a view into the
buffer, not a copy.
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
:end_tab:

Generation is a prefill call over the prompt followed by one decode call
per new token. `decode` draws each token from the last logits, after the
temperature and top-$k$ truncation of :numref:`subsec_gpt-sampling`, and
feeds it back. `generate_cached` gives it the interface of `generate`.

```{.python .input #kv-cache-the-cached-forward-pass-2}
%%tab pytorch
def sample(logits, temperature=1.0, top_k=None):
    """One token per row of logits (batch, vocab), as generate draws it."""
    logits = logits / temperature
    if top_k is not None:
        cutoff = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    return torch.multinomial(F.softmax(logits, -1), 1)

@torch.inference_mode()
def decode(model, X, ks, vs, t, num_tokens, temperature=1.0, top_k=None):
    """Feed X at positions t, t + 1, ..., then sample num_tokens tokens
    per sequence; returns them as (batch, num_tokens), and the cache."""
    logits, ks, vs = forward_cached(model, X, ks, vs, t)
    t, tokens = t + X.shape[1], []
    for i in range(num_tokens):
        tokens.append(sample(logits, temperature, top_k))
        logits, ks, vs = forward_cached(model, tokens[-1], ks, vs, t + i)
    return torch.cat(tokens, 1), ks, vs

@d2l.add_to_class(d2l.GPT)
def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample as generate does, but never recompute the prefix."""
    assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
    X = torch.tensor(prefix, device=next(self.parameters()).device)[None]
    ks, vs = init_cache(self, 1, len(prefix) + num_tokens)
    tokens = decode(self, X, ks, vs, 0, num_tokens, temperature, top_k)[0]
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
    logits, ks, vs = forward_cached(model, X, ks, vs, t)
    def step(i, carry):
        logits, ks, vs, tokens = carry
        token = sample(logits, jax.random.fold_in(key, i), temperature, top_k)
        logits, ks, vs = forward_cached(model, token, ks, vs,
                                        t + X.shape[1] + i)
        return logits, ks, vs, tokens.at[:, i].set(token[:, 0])
    tokens = jnp.zeros((X.shape[0], num_tokens), dtype=jnp.int32)
    _, ks, vs, tokens = jax.lax.fori_loop(0, num_tokens, step,
                                          (logits, ks, vs, tokens))
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
sequence.

```{.python .input #kv-cache-same-logits-measured-1}
%%tab pytorch
for pos in ('rope', 'learned'):
    model = d2l.GPT(vocab_size=97, num_hiddens=128, num_heads=4,
                    num_blks=3, pos=pos).eval()
    x = torch.randint(0, 97, (1, 48))
    with torch.inference_mode():
        full = model(x)[0, 15:]  # Logits after positions 15, ..., 47
    ks, vs = init_cache(model, 1, 48)
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
        model = d2l.GPT(vocab_size=97, num_hiddens=128, num_heads=4,
                        num_blks=3, pos=pos, rngs=nnx.Rngs(d2l.get_key()))
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

The two paths agree to rounding error. Next we time one generation step at
growing context lengths, on an untrained model of GPT-2's size: 12 blocks
of width 768, 124 million parameters. The arithmetic does not depend on
the weights. The naive step is a forward pass over all $n$ tokens. The
cached step is a decode call against a cache of $n$ positions, averaged
over 32 steps; the helper `time_decode` measures it for a batch of
sequences, and we reuse it below. We time with `d2l.Benchmark`, which
calls a function a few times to warm up, waits for the device to finish,
and averages the time of ten further calls.

```{.python .input #kv-cache-same-logits-measured-2}
%%tab pytorch
device = d2l.try_gpu()
model = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
                num_blks=12, max_len=4096).to(device).eval()
print(f'{d2l.num_params(model) / 1e6:.0f}M parameters')

def time_decode(model, batch_size, n):
    """Seconds per decode step for batch_size sequences at context n,
    averaged over 32 steps."""
    x = torch.randint(0, 50257, (batch_size, 1), device=device)
    ks, vs = init_cache(model, batch_size, n + 32)
    return d2l.Benchmark(
        lambda: decode(model, x, ks, vs, n - 1, 31)).time / 32

lengths, t_naive, t_cached = [256, 512, 1024, 2048, 4032], [], []
for n in lengths:
    x = torch.randint(0, 50257, (1, n), device=device)
    with torch.inference_mode():
        t_naive.append(d2l.Benchmark(lambda: model(x)[:, -1]).time)
    t_cached.append(time_decode(model, 1, n))
    print(f'n={n:4d}: naive {1e3 * t_naive[-1]:5.1f} ms, '
          f'cached {1e3 * t_cached[-1]:4.1f} ms per token')
```

```{.python .input #kv-cache-same-logits-measured-2}
%%tab jax
model = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
                num_blks=12, max_len=4096, rngs=nnx.Rngs(d2l.get_key()))
model.eval()
graphdef, params = nnx.split(model)
print(f'{d2l.num_params(model) / 1e6:.0f}M parameters')

@partial(jax.jit, static_argnames='graphdef')
def last_logits(graphdef, params, X):
    """The step of naive generation: a forward pass over all of X."""
    return nnx.merge(graphdef, params)(X)[:, -1]

def time_decode(model, batch_size, n):
    """Seconds per decode step for batch_size sequences at context n,
    averaged over 32 steps."""
    graphdef, params = nnx.split(model)
    x = jax.random.randint(d2l.get_key(), (batch_size, 1), 0, 50257)
    cache = init_cache(model, batch_size, n + 32)
    def run():
        nonlocal cache  # decode consumes the cache and returns it updated
        tokens, *cache = decode(graphdef, params, x, *cache, n - 1, 31,
                                d2l.get_key())
        return tokens
    return d2l.Benchmark(run).time / 32

lengths, t_naive, t_cached = [256, 512, 1024, 2048, 4032], [], []
for n in lengths:
    x = jax.random.randint(d2l.get_key(), (1, n), 0, 50257)
    t_naive.append(d2l.Benchmark(
        lambda: last_logits(graphdef, params, x)).time)
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
several times faster, and beyond about a thousand tokens the gap widens with
every doubling of the context. At short contexts the two curves approach
each other and can even cross. A step of either kind then spends much of
its time on fixed costs, such as launching the kernels of twelve blocks,
rather than on arithmetic.

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

The cached loop is several times faster, though by far less than the
factor of about 250 in operations. Every decoded token still reads all the weights of
the model, as :numref:`subsec_kv-cache-prefill-decode` shows. Serving
systems report the two phases separately: the *time to first token*
measures the prefill, and the *tokens per second* measure the decode loop.

## Memory and Bandwidth
:label:`subsec_kv-cache-memory`

The size of the cache follows from the shape of its buffers. Each of the
$L$ blocks stores one key and one value vector of dimension
$H_\textrm{kv} d_h$ per token, for $H_\textrm{kv}$ key--value heads of width
$d_h$, so a batch of $b$ sequences of length $n$ holds

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
ks, vs = init_cache(model, 1, 4096)
print(f'cache buffers {(ks.nbytes + vs.nbytes) / 2**20:.0f} MiB, '
      f'formula {kv_cache_bytes(model, 4096) / 2**20:.0f} MiB, '
      f'weights {4 * d2l.num_params(model) / 2**20:.0f} MiB')
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

A decode step performs about $2N$ operations and reads at least the $4N$
bytes of the fp32 weights plus the cache, an intensity below one FLOP per
byte. A prefill over $n$ tokens performs $2Nn$ operations with the same
weight traffic, because each weight is read once and used for every token,
so its intensity is about $n$ times higher. Prefill is therefore limited by
arithmetic and decode by memory bandwidth
:cite:`Pope.Douglas.Chowdhery.ea.2023`. We time a forward pass over a
2,048-token prompt and the decoding of 32 tokens after it.

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch
x = torch.randint(0, 50257, (1, 2048), device=device)
with torch.inference_mode():
    t_prefill = d2l.Benchmark(lambda: model(x)[:, -1]).time
t_decode = time_decode(model, 1, 2048)
```

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab jax
x = jax.random.randint(d2l.get_key(), (1, 2048), 0, 50257)
t_prefill = d2l.Benchmark(lambda: last_logits(graphdef, params, x)).time
t_decode = time_decode(model, 1, 2048)
```

We compare both rates with the intensity of a decode step and with the
rate that a bandwidth of 1 TB/s would allow.

```{.python .input #kv-cache-prefill-and-decode-1}
%%tab pytorch, jax
N, cache = d2l.num_params(model), kv_cache_bytes(model, 2048)
flops, moved = 2 * N, 4 * N + cache  # One decode step at context 2048
print(f'prefill: {2048 / t_prefill:6.0f} tokens/s')
print(f'decode:  {1 / t_decode:6.0f} tokens/s, bandwidth ceiling '
      f'{1e12 / moved:.0f} tokens/s')
print(f'FLOPs per byte: decode {flops / moved:.2f}, '
      f'prefill about {2048 * flops / moved:.0f}')
```

The model reads the prompt at tens of thousands of tokens per second and
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
batches, curves = [1, 4, 16, 64], []
for n in (128, 1024):
    rates = [b / time_decode(model, b, n) for b in batches]
    W, nc = 4 * d2l.num_params(model), kv_cache_bytes(model, n)
    curves += [rates, [b * 1e12 / (W + b * nc) for b in batches]]
    print(f'n={n}: ' + ', '.join(f'{r:.0f}' for r in rates) + ' tokens/s')
d2l.plot(batches, curves, 'batch size', 'tokens per second',
         legend=['n=128', 'n=128 bound', 'n=1024', 'n=1024 bound'],
         fmts=('C0-o', 'C0:', 'C1-s', 'C1:'), xscale='log', yscale='log')
```

The bound keeps growing over the whole range of batches at a context of
128 tokens, since the caches reach the size of the weights only near a
batch of 53. At 1,024 tokens it grows little beyond a batch of 16 and
approaches the ceiling set by the caches. Batching spreads the weight
reads over many sequences, but every sequence still reads its own cache.
At small batches the measured curves stay below the bound, by the fixed
cost per step of the previous subsection.

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

## Sharing Keys and Values across Heads

In :eqref:`eq_kv-cache-bytes`, the factor $H_\textrm{kv}$ counts
key--value heads, and nothing in the attention mechanism forces it to equal
the number $H$ of query heads. *Multi-query attention* (MQA) keeps all $H$
query heads but a single key--value head shared by all of them
:cite:`Shazeer.2019`, which shrinks the cache by a factor of $H$ at a
usually slight cost in quality. *Grouped-query attention* (GQA) lies in
between, with $H_\textrm{kv}$ key--value heads, each serving a group of
$H / H_\textrm{kv}$ query heads :cite:`Ainslie.Lee-Thorp.Jong.ea.2023`.
:numref:`fig_gqa` shows the sharing patterns. GQA with $H_\textrm{kv}$ between
$H/8$ and $H/4$ is the common choice in current open models: Llama 3 8B
and Mistral 7B both run 32 query heads against 8 key--value heads
:cite:`Grattafiori.Dubey.Jauhri.ea.2024,Jiang.Sablayrolles.Mensch.ea.2023`.

![Query heads and the key--value heads they read. Multi-head attention gives every query head its own key--value head; grouped-query attention shares one per group of query heads; multi-query attention shares a single one across all of them. The cache scales with the number of key--value heads.](../img/mdl-transformers-gqa.svg)
:label:`fig_gqa`

### Grouped Key--Value Heads in Code

`d2l.GPT` and its attention `GPT.CausalAttention` take the number of
key--value heads as the argument `num_kv_heads` (:numref:`sec_gpt`). The
fused projection $\mathbf{W}_{qkv}$ then produces $H$ query heads but only
$H_\textrm{kv}$ key heads and $H_\textrm{kv}$ value heads. Query head $h$ reads
key--value head $\lfloor h H_\textrm{kv} / H \rfloor$, so each group of
$H / H_\textrm{kv}$ consecutive query heads shares one, and the cache
stores each key--value head once. With
$H_\textrm{kv} = H$ the layer is ordinary multi-head attention, and the cached
forward pass above handles any $H_\textrm{kv}$, since `init_cache` sizes the
buffers by `num_kv_heads`.

```{.python .input #kv-cache-a-pluggable-implementation-1}
%%tab pytorch
attn = d2l.GPT.CausalAttention(256, num_heads=8, num_kv_heads=2)
q, k, v = attn.qkv(torch.ones(2, 10, 256))
print(f'queries {tuple(q.shape)}, keys {tuple(k.shape)}, '
      f'values {tuple(v.shape)}')
```

```{.python .input #kv-cache-a-pluggable-implementation-1}
%%tab jax
attn = d2l.GPT.CausalAttention(256, num_heads=8, num_kv_heads=2,
                               rngs=nnx.Rngs(d2l.get_key()))
q, k, v = attn.qkv(jnp.ones((2, 10, 256)))
print(f'queries {q.shape}, keys {k.shape}, values {v.shape}')
```

Eight query heads of width 32 face two key heads and two value heads. The
parameter count shows where the saving comes from: $\mathbf{W}_k$ and
$\mathbf{W}_v$ shrink with $H_\textrm{kv}$, so the attention parameters fall from
$4d^2$ toward $2d^2$, and the cache per token shrinks by the factor
$H / H_\textrm{kv}$.

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab pytorch
for G in (8, 4, 2, 1):
    attn = d2l.GPT.CausalAttention(256, 8, num_kv_heads=G)
    print(f'H_kv={G}: {d2l.num_params(attn):6d} parameters, cache '
          f'{2 * attn.kv_hiddens * 4:5d} bytes per token and block')
```

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab jax
for G in (8, 4, 2, 1):
    attn = d2l.GPT.CausalAttention(256, 8, num_kv_heads=G,
                                   rngs=nnx.Rngs(d2l.get_key()))
    print(f'H_kv={G}: {d2l.num_params(attn):6d} parameters, cache '
          f'{2 * attn.kv_hiddens * 4:5d} bytes per token and block')
```

### Cache Size and Validation Loss
:label:`subsec_kv-cache-gqa-quality`

Sharing helps only if the model stays nearly as good. To measure its
effect, we train the byte-level model of
:numref:`subsec_gpt-training` with its recipe, once for each
number of key--value heads from $H_\textrm{kv} = 8$ (full multi-head
attention) down to $H_\textrm{kv} = 1$ (MQA). The data are the bytes of the
Wikipedia articles of `d2l.WikiText103`, and `d2l.LMOptimizer` runs AdamW
with warmup, cosine decay, and gradient clipping for 1,000 steps. The
model almost never reads the same passage twice, so the validation loss at
the end of training measures what it has learned.

```{.python .input #kv-cache-cache-against-quality}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=128)
for G in (8, 4, 2, 1):
    lm = d2l.GPT(len(data.vocab), num_kv_heads=G)
    optimizer = d2l.LMOptimizer(lm, lr=2e-3, num_steps=1000, warmup=100)
    d2l.train_lm(lm, data, optimizer, 1000)
    print(f'H_kv={G}: validation loss '
          f'{math.log(d2l.lm_perplexity(lm, data)):.2f}, cache '
          f'{kv_cache_bytes(lm, 1) / 1024:4.1f} KiB per token')
```

```{.python .input #kv-cache-cache-against-quality}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
for G in (8, 4, 2, 1):
    lm = d2l.GPT(len(data.vocab), num_kv_heads=G,
                 rngs=nnx.Rngs(d2l.get_key()))
    optimizer = d2l.LMOptimizer(lm, lr=2e-3, num_steps=1000, warmup=100)
    d2l.train_lm(lm, data, optimizer, 1000)
    print(f'H_kv={G}: validation loss '
          f'{math.log(d2l.lm_perplexity(lm, data)):.2f}, cache '
          f'{kv_cache_bytes(lm, 1) / 1024:4.1f} KiB per token')
```

The cache shrinks eightfold from the first row to the last. All four
losses lie within a few hundredths of a nat, so at this scale even a single
key--value head causes no loss we can measure. At large scale MQA loses a
little quality, while GQA with $H/8$ key--value heads comes close to
multi-head attention :cite:`Ainslie.Lee-Thorp.Jong.ea.2023`. A multi-head
checkpoint can also be *uptrained* into a grouped one. Averaging the key
and value projections within each group and training on for about 5% of
the original steps recovers most of the quality, and this is how GQA
spread through existing model families.

## Compressing the Cache Further

Sharing heads reduces one factor of :eqref:`eq_kv-cache-bytes`. Two other
factors can shrink as well: the width of what is stored per token, and the
number of tokens stored. Our small byte-level models are too weak to show
how a trained model uses its cache, so we study both factors on the
released GPT-2.
`d2l.load_gpt2` builds it as a `d2l.GPT` with the weights loaded as in
:numref:`subsec_gpt-loading`, and `d2l.gpt2_tokenizer` returns its
tokenizer. We encode *The Time Machine* and keep its first 1,024 tokens as
the evaluation passage `x` and the next 1,024 tokens as a second passage
`calib`.

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

Each experiment below changes only what happens inside attention, so one
helper serves them all: a forward pass that walks GPT-2's blocks as the
class does, but computes attention with a function that we supply. We
check that the helper with ordinary causal attention reproduces the model,
and compute a baseline loss on the second half of the passage, from
position 512 on, where every predicted token has a long history.

```{.python .input #kv-cache-compressing-the-cache-further-2}
%%tab pytorch
def forward_blocks(model, X, attn_fn):
    """GPT-2's forward pass with attention computed by attn_fn(q, k, v, i)
    in block i, on inputs of shape (batch, heads, length, head width)."""
    H = model.token_emb(X) + model.pos_emb(
        torch.arange(X.shape[1], device=X.device))
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
    """GPT-2's forward pass with attention computed by attn_fn(q, k, v, i)
    in block i, on inputs of shape (batch, heads, length, head width)."""
    H = model.token_emb(X) + model.pos_emb(jnp.arange(X.shape[1]))
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

### Low-Rank Keys and Values
:label:`subsec_kv-cache-low-rank`

GQA shrinks the cache by storing fewer key--value heads. A different
approach keeps all heads. It assumes that the keys and values of a token,
$2 \times 768 = 1536$ numbers in GPT-2, lie close to a subspace of much
lower dimension, so that the cache can store the $r$ coordinates of each
token in that subspace. *Multi-head latent attention* (MLA), around which
DeepSeek built its V2 and V3 models, implements this idea
:cite:`DeepSeek-AI.2024`. A trained down-projection compresses the keys
and values of each token jointly into one latent vector, and the cache
stores only the latent. In DeepSeek-V2 the latent has 512 dimensions,
against 32,768 numbers of keys and values per token in its 128 heads.
Trained up-projections reconstruct the keys and values of every head when
they are read. RoPE needs one more piece. Its rotation depends on the
position, so it cannot be folded into a shared projection, and MLA carries
a small separate rotary key per token.

Whether the keys and values of a trained model admit such a compression
can be tested on GPT-2 without training anything. In every block we stack
the keys and values of each token of `calib` as one row of a matrix, and
compute its top $r$ right singular vectors. By the Eckart--Young theorem
they span the $r$-dimensional subspace closest to these rows
(:numref:`sec_mdl-svd-low-rank`). We then run the model on `x` with every
key and value replaced by its projection onto this subspace, which a cache
could store as $r$ coordinates per token. The basis comes from another
passage, as a trained projection would, so no token of `x` influences how
the keys and values of `x` are compressed.

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab pytorch
def kv_rows(k, v):
    """Keys and values of all heads of each token as one row: (T, 2d)."""
    return torch.cat([u[0].transpose(0, 1).flatten(1) for u in (k, v)], 1)

bases = []  # Right singular vectors of each block, fitted on calib
def fit_basis(q, k, v, i):
    bases.append(torch.linalg.svd(kv_rows(k, v), full_matrices=False)[2])
    return causal(q, k, v, i)

def projected(r):
    def attn_fn(q, k, v, i):
        P = bases[i][:r]  # (r, 2d), orthonormal rows
        KV = kv_rows(k, v) @ P.T @ P
        k, v = (u.unflatten(1, (k.shape[1], -1)).transpose(0, 1)[None]
                for u in KV.chunk(2, 1))
        return causal(q, k, v, i)
    return attn_fn

with torch.inference_mode():
    forward_blocks(gpt2, calib, fit_basis)
    for r in (512, 256, 128):
        loss = tail_loss(forward_blocks(gpt2, x, projected(r)), x)
        print(f'rank {r} of 1536: loss {loss:.2f}, '
              f'perplexity {loss.exp():.0f}')
```

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab jax
def kv_rows(k, v):
    """Keys and values of all heads of each token as one row: (T, 2d)."""
    return jnp.concatenate(
        [u[0].swapaxes(0, 1).reshape(u.shape[2], -1) for u in (k, v)], 1)

bases = []  # Right singular vectors of each block, fitted on calib
def fit_basis(q, k, v, i):
    bases.append(jnp.linalg.svd(kv_rows(k, v), full_matrices=False)[2])
    return causal(q, k, v, i)

def projected(r):
    def attn_fn(q, k, v, i):
        P = bases[i][:r]  # (r, 2d), orthonormal rows
        KV = kv_rows(k, v) @ P.T @ P
        k, v = (u.reshape(u.shape[0], k.shape[1], -1).swapaxes(0, 1)[None]
                for u in jnp.split(KV, 2, axis=1))
        return causal(q, k, v, i)
    return attn_fn

forward_blocks(gpt2, calib, fit_basis)
for r in (512, 256, 128):
    loss = tail_loss(forward_blocks(gpt2, x, projected(r)), x)
    print(f'rank {r} of 1536: loss {loss:.2f}, '
          f'perplexity {jnp.exp(loss):.0f}')
```

With a basis of rank 512, a third of the width, the loss matches that of
full attention, even a few hundredths of a nat below it, since dropping the
weakest directions also drops some noise. At rank 256, a sixth, it rises by
only a few hundredths of a nat, although the basis never saw the passage.
At rank 128 the loss rises by about half a nat. The keys and values of a
trained model thus lie close to a subspace much smaller than their width,
and the subspace carries over from one passage of the book to the next.
MLA builds on this structure but learns its projections during training
instead of fitting them afterward, so the model can adapt to the
compression.

### Sliding Windows and Attention Sinks
:label:`subsec_kv-cache-sinks`

The remaining factor is length. Keeping only the most recent $w$ tokens in
the cache, and evicting the oldest as generation proceeds, bounds its
memory for any length of text. This is the rolling-buffer form of the
sliding-window attention of :numref:`sec_attention-at-scale`. Before
evicting anything, we measure where a trained model sends its attention:
the average weight that GPT-2's queries, from position 64 on, place on the
first token of the passage.

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab pytorch
masses = []  # Mean weight on token 0, one entry per block
def watch(q, k, v, i):
    """Explicit causal attention that records the weight on token 0."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    alpha = d2l.masked_softmax(scores, d2l.causal_mask(
        q.shape[2], device=q.device))
    masses.append(alpha[..., 64:, 0].mean())
    return alpha @ v

with torch.inference_mode():
    forward_blocks(gpt2, x, watch)
print('mean attention weight on token 0 (uniform would be about 0.003)')
print('block ' + ''.join(f'{i:6d}' for i in range(len(masses))))
print('weight' + ''.join(f'{m:6.2f}' for m in torch.stack(masses).tolist()))
```

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab jax
masses = []  # Mean weight on token 0, one entry per block
def watch(q, k, v, i):
    """Explicit causal attention that records the weight on token 0."""
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    alpha = d2l.masked_softmax(scores, d2l.causal_mask(q.shape[2]))
    masses.append(alpha[..., 64:, 0].mean())
    return alpha @ v

forward_blocks(gpt2, x, watch)
print('mean attention weight on token 0 (uniform would be about 0.003)')
print('block ' + ''.join(f'{i:6d}' for i in range(len(masses))))
print('weight' + ''.join(f'{m:6.2f}' for m in jax.device_get(masses)))
```

From block 3 on, the queries place between about a third and more
than half of their attention on the first token, a hundred times its
uniform share or more. :citet:`Xiao.Tian.Chen.ea.2024` call such positions
*attention sinks* and trace them to the softmax. The weights of every query
sum to one, so a head with nothing to retrieve must still put its weight
somewhere. The first token, which every query can see, becomes that place.

A rolling buffer of the most recent $w$ tokens evicts this token as soon as
the text is longer than the window, and the softmax then spreads its weight
over tokens that the heads had chosen to ignore. StreamingLLM therefore
keeps the first few tokens in the cache next to the window
:cite:`Xiao.Tian.Chen.ea.2024`. We reproduce both the failure and the fix
by evaluating the passage with attention restricted to a window of 256
tokens plus a number of retained *sink tokens* at the start.

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab pytorch
j = torch.arange(x.shape[1], device=device)
allowed = d2l.causal_mask(x.shape[1], device=device)
window = allowed & (j[:, None] - j[None, :] < 256)
with torch.inference_mode():
    for nsink in (0, 1, 4):
        mask = window | (allowed & (j[None, :] < nsink))
        loss = tail_loss(forward_blocks(
            gpt2, x, lambda q, k, v, i: F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask)), x)
        print(f'window 256 + {nsink} sink tokens: loss {loss:.2f}, '
              f'perplexity {loss.exp():.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {full.exp():.0f}')
```

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab jax
j = jnp.arange(x.shape[1])
allowed = d2l.causal_mask(x.shape[1])
window = allowed & (j[:, None] - j[None, :] < 256)
for nsink in (0, 1, 4):
    mask = window | (allowed & (j[None, :] < nsink))
    loss = tail_loss(forward_blocks(
        gpt2, x, lambda q, k, v, i: jax.nn.dot_product_attention(
            *(u.swapaxes(1, 2) for u in (q, k, v)),
            mask=mask).swapaxes(1, 2)), x)
    print(f'window 256 + {nsink} sink tokens: loss {loss:.2f}, '
          f'perplexity {jnp.exp(loss):.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {jnp.exp(full):.0f}')
```

Without sink tokens, the window raises the perplexity from 36 to several
thousand. Keeping the first token removes almost all of that increase, and
with four tokens the loss comes within about 0.2 nat of full attention.
The rest of the gap is what the window loses by dropping older tokens. Our
mask keeps the original position of every token. A real rolling buffer
must also decide which positions the retained entries carry, and an
exercise works this out.

Models can also be trained not to need a sink. gpt-oss gives every head of
every attention layer a learned *sink logit*, an extra entry in the
denominator of its softmax that absorbs unneeded weight, so the first token
no longer has to serve this role :cite:`OpenAI.2025`.
:citet:`Qiu.Wang.Zheng.ea.2025` multiply the output of each head by a
sigmoid gate computed from the layer's input, so a head with nothing to
retrieve can return nothing. In their models with 1.7 and 15 billion
parameters, the average attention on the first token fell from 47% to 5%.

### Comparison of Cache-Reduction Methods

For one sequence of $n$ tokens, the baseline cache stores
$2 L n H_\textrm{kv} d_h$ elements for $L$ blocks of $H_\textrm{kv}$
key--value heads of width $d_h$. The table compares the methods of this
section on that count. Constants and auxiliary buffers depend on the
implementation.

| method | reduced factor | stored elements | information given up | training requirement | evidence in this section |
|:--|:--|:--|:--|:--|:--|
| GQA/MQA | key--value heads | $2LnH_\textrm{kv}d_h$ with fewer heads | separate keys and values for every query head | train with sharing, or uptrain a checkpoint | character model trained at each $H_\textrm{kv}$; cited large-model ablations |
| MLA | width per token | about $Lnr$, plus a small rotary key | keys and values outside an $r$-dimensional latent | learned compression and reconstruction maps | fixed rank-$r$ basis on GPT-2; cited trained MLA models |
| sliding window with sinks | cached length | $2L(w+s)H_\textrm{kv}d_h$ for window $w$ and $s$ sinks | tokens outside the window and the sinks | train or adapt to the restricted pattern | GPT-2 on one passage; cited StreamingLLM experiments |
| quantization | bytes per element | $2LnH_\textrm{kv}d_h$ at 2 to 4 bits each instead of 16 | precision of every stored number | none: applied to a trained model | an exercise on GPT-2; cited KIVI experiments |
| linear attention | the token-indexed cache | fixed state matrices, independent of $n$ | exact softmax attention and recall of individual tokens | replace and train the attention mechanism | derivation and experiment in :numref:`sec_attention-at-scale` |

Quantization combines with every other row, because it changes only the
number of bits that store each number. At two to four bits, quantizing
keys per channel and values per token keeps generation quality close to
that of the full cache :cite:`Liu.Yuan.Jin.ea.2024`. An exercise tests why
this grouping works. Other rows combine too. Hybrid stacks
(:numref:`chap_modern_rnn`) pair recurrent layers that keep a fixed state,
such as linear attention, with a few full-attention layers, so that only
those few layers keep a growing cache.

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
whole cache. Batching shares the weight reads but not the cache reads. GQA
and MQA shrink the cache by sharing key--value heads at little cost in
quality, and low-rank latents such as MLA shrink the width stored per
token. A sliding window bounds the length, provided it keeps the attention
sinks that trained models rely on.

## Exercises

1. **Cost of generation.** A model with $N$ parameters reads a prompt of
   $P$ tokens and then generates $T$ tokens. Count $2N$ dense-layer FLOPs
   per token of a forward pass.
    1. Count the dense-layer FLOPs of a naive schedule that runs a forward
       pass over the whole history before every new token, and of
       `generate_cached`, which runs one prefill over the prompt and then
       single-token steps. Evaluate the ratio of the two counts for the
       end-to-end run of :numref:`subsec_kv-cache-same-logits`, with
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

1. [code] **A basis fitted on the passage itself.** The experiment of
   :numref:`subsec_kv-cache-low-rank` fits its basis on a second passage.
    1. Fit the basis instead on the keys and values of `x` itself, the
       passage that `tail_loss` evaluates, and compare the losses at ranks
       512, 256, and 128 with those of the basis from `calib` at the same
       rank.
    1. Explain why this comparison is unfair: which later tokens influence
       how the key and value of token $t$ are compressed, and why can such
       an influence lower the loss of predicting token $t+1$?
    1. A server could instead fit the basis on the prompt, here the first
       512 tokens of `x`, and apply it to every later token. Implement this
       variant and compare it with the other two. Which of the three could
       a deployed cache use?

1. [code] **Quantized keys and values.** Store the keys and values of
   GPT-2 in 4 bits: round each number to one of 16 levels spaced evenly
   between the minimum and the maximum of its group, where a group is
   either one channel (one coordinate over all tokens of a head) or one
   token (all coordinates of one token in a head).
    1. Use `forward_blocks` to evaluate `tail_loss` on `x` for the four
       combinations of grouping keys and values per channel or per token.
       Which combination works best?
    1. Measure, in a few blocks, the largest absolute value of each key
       channel and of each value channel. Use the result to explain your
       answer to the previous question, and compare it with the choice of
       :citet:`Liu.Yuan.Jin.ea.2024`.
    1. How many bits per number does each grouping cost, counting the
       minimum and the maximum stored with every group in 16 bits, for a
       context of 1024 tokens?

1. [code] **Sinks as no-ops.** :citet:`Xiao.Tian.Chen.ea.2024` interpret a
   sink as the position where a head places attention it does not need.
   gpt-oss makes this role explicit by giving every head a learned logit
   $a_\textrm{sink}$ in the denominator of its softmax
   :cite:`OpenAI.2025`. For the query at position $t$ with scores
   $a_{t,0}, \ldots, a_{t,t}$ on the tokens at positions $0, \ldots, t$,
   the weight on token $j$ is
   $$\alpha_{t,j} = \frac{\exp(a_{t,j})}{\exp(a_\textrm{sink}) + \sum_{i=0}^{t} \exp(a_{t,i})}.$$
    1. Show that such a head computes the output of ordinary softmax
       attention over the $t + 1$ tokens plus one extra key, whose score is
       $a_\textrm{sink}$ for every query and whose value is the zero
       vector. Which values can the total weight on the $t + 1$ tokens
       take?
    1. GPT-2 has no sink logit, so its weights are those of the formula
       with $a_\textrm{sink} = -\infty$. Extend `watch` to record, in every
       block, the norm of the value vector $\mathbf{v}_0$ of token 0
       divided by the mean value norm of the other tokens. Record also the
       norm of $\alpha_{t,0}\, \mathbf{v}_0$ divided by the norm of the head
       output at position $t$, averaged over heads and over the queries
       from position 64 on. In which blocks does token 0 behave like the
       zero-value key of the previous sub-problem?
    1. Repeat the measurement of attention mass on a 1024-token passage
       that starts 100 tokens later in the book, so that a different token
       sits at position 0. Does the high mass stay at position 0?

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

1. [code] **Uptraining to GQA.** ● :citet:`Ainslie.Lee-Thorp.Jong.ea.2023`
   convert a multi-head checkpoint into a grouped one by mean-pooling the
   key and value projections of the heads in each group, and then continue
   training for 5 percent of the original number of steps. Apply this
   recipe to the byte-level model that
   :numref:`subsec_kv-cache-gqa-quality` trains.
    1. Train the $H_\textrm{kv} = 8$ configuration for 1,000 steps. Convert it to
       $H_\textrm{kv} = 2$ by averaging the key and value columns of
       $\mathbf{W}_{qkv}$ over each group of four consecutive heads, the
       grouping that `num_kv_heads` uses, and measure the validation loss
       without further training. Repeat the conversion with the first head
       of each group in place of the average.
    1. Continue training the mean-pooled model for 50 steps at a small
       constant learning rate. Compare its validation loss with that of the
       $H_\textrm{kv} = 8$ model and with that of an $H_\textrm{kv} = 2$ model trained
       from scratch for 1,050 steps.

1. **Absorbed projections.** ● In MLA, every token $j$ of a layer caches
   one latent vector $\mathbf{c}_j \in \mathbb{R}^r$, and head $h$
   reconstructs its key and value as $\mathbf{k}^{(h)}_j =
   \mathbf{U}^{(h)}_K \mathbf{c}_j$ and $\mathbf{v}^{(h)}_j =
   \mathbf{U}^{(h)}_V \mathbf{c}_j$, with up-projections
   $\mathbf{U}^{(h)}_K, \mathbf{U}^{(h)}_V \in \mathbb{R}^{d_h \times r}$
   :cite:`DeepSeek-AI.2024`. Head $h$ has the query $\mathbf{q}^{(h)}_t$ at
   position $t$ and the attention output $\mathbf{o}^{(h)}_t$, and the
   layer returns $\sum_h \mathbf{W}^{(h)}_O \mathbf{o}^{(h)}_t$ with
   $\mathbf{W}^{(h)}_O \in \mathbb{R}^{d \times d_h}$. Ignore the
   $1/\sqrt{d_h}$ scaling of the scores.
    1. Show that a decode step can compute every score and the layer output
       from the cached latents alone, without forming any key or value, by
       folding $\mathbf{U}^{(h)}_K$ into the query and $\mathbf{U}^{(h)}_V$
       into the output projection.
    1. With RoPE, in the notation of :numref:`sec_positional-information`,
       the score becomes
       $(\mathbf{R}_t \mathbf{q}^{(h)}_t)^\top \mathbf{R}_j
       \mathbf{U}^{(h)}_K \mathbf{c}_j$. Show that folding no longer
       produces a matrix that is independent of $j$. MLA therefore appends
       to every key a rotary part $\mathbf{R}_j \mathbf{k}^R_j$, where
       $\mathbf{k}^R_j \in \mathbb{R}^{d_R}$ is shared by all heads and
       cached next to $\mathbf{c}_j$, and it appends a rotary part to every
       query. Show that the score then splits into a term that admits the
       folding and a term computed from the cached $\mathbf{k}^R_j$.
    1. DeepSeek-V2 uses $H = 128$ heads with $d_h = 128$, $r = 512$, and
       $d_R = 64$. For one layer and one cached token, compare the number
       of cached elements and the FLOPs of the scores and weighted sums of
       a decode step under multi-head attention with cached keys and values
       and under MLA with folded projections. How many key--value heads
       would GQA need for a cache of the same size? What does the ratio of
       FLOPs to cached elements imply for the decode step of
       :numref:`subsec_kv-cache-prefill-decode`?

1. [extended] **A streaming cache.** ●● The mask-based experiment of
   :numref:`subsec_kv-cache-sinks` restricts attention without evicting any
   entry, and it keeps the original position of every token.
    1. Change `forward_cached` so that the cache of each block is a rolling
       buffer. It holds the first $s$ entries and the most recent $w$
       entries, the current token included, and overwrites the oldest of
       the remaining entries. Prefill GPT-2 with the first 256 tokens of
       the passage `x`, feed the remaining tokens one at a time with
       $w = 256$ and $s = 4$, and confirm that the loss on positions 512
       onward matches the mask-based result.
    1. GPT-2 cannot decode beyond position 1023, where its position table
       ends. A RoPE model can stream indefinitely if the positions of the
       cached entries are reassigned. Decide what the buffer must store for
       each entry and which positions the rotation should use, so that
       every query--key offset stays below $s + w$. Explain why learned
       position embeddings do not admit the same reassignment.
    1. Train the byte-level `d2l.GPT` of
       :numref:`subsec_kv-cache-gqa-quality`, with its default RoPE
       attention, at context 128. Stream the first 10,000 bytes of the
       validation text, `data.val` read in order, through your buffer with
       $s + w = 128$. Compare the mean loss for $s = 4$ and for $s = 0$
       with that of a full forward pass over the most recent 128 bytes at
       every step.

[Discussions](https://d2l.discourse.group/t/36776)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.3]{.kicker}

Generation and the KV Cache<br>
**reusing past keys and values · prefill and decode · batching · GQA, low rank, windows, and sinks**
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
- tell why reading a prompt is limited by arithmetic and writing text by
  memory bandwidth;
- predict where batching stops helping;
- shrink the cache by sharing heads, by a low-rank basis, and by a window
  with sinks.

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
| **Generation and the KV Cache** | **reuse keys and values while sampling** |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Mixture of Experts | more parameters per FLOP |
| Scaling Laws | loss against parameters, data, compute |
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
- Step $t$ needs one new query against $\mathbf{k}_1, \ldots, \mathbf{k}_t$.
- Positions $1, \ldots, t-1$ never see position $t$: their keys and values
  are the same as one step ago.
:::

::: {.col .fig .big}
@fig:mdl-transformers-kv-cache
:::
:::
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="pytorch"}
```{.python #kv-cache-exercises-1}
H = model.token_emb(X)                     # T new tokens at t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = blk.attention.qkv(blk.norm1(H), t)   # RoPE offset t
    ks[i, :, :, t:t + T], vs[i, :, :, t:t + T] = k, v
    Y = F.scaled_dot_product_attention(
        q, ks[i, :, :, :t + T], vs[i, :, :, :t + T], attn_mask=mask)
    H = H + blk.attention.W_o(Y.transpose(1, 2).flatten(2))
    H = H + blk.ffn(blk.norm2(H))
```

- Buffers allocated once at full length: no copy of the cache per step.
- Prefill: $T$ = prompt length at $t = 0$. Decode: $T = 1$.
:::

::: {.slide title="A Cached Step Writes Its Slot and Reads the Prefix" only="jax"}
```{.python #kv-cache-exercises-2}
H = model.token_emb(X)                     # T new tokens at t, t+1, ...
for i, blk in enumerate(model.blks):
    q, k, v = (u.swapaxes(1, 2)
               for u in blk.attention.qkv(blk.norm1(H), t))
    ks = jax.lax.dynamic_update_slice(ks, k[None], (i, 0, t, 0, 0))
    vs = jax.lax.dynamic_update_slice(vs, v[None], (i, 0, t, 0, 0))
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
weights. Naive grows at least like $2Nn$; cached adds the cache read to a fixed
cost per step.
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
1 TB/s, so about 80 FLOPs per byte to keep the arithmetic units busy.

| step | FLOPs | bytes | FLOPs per byte |
|---|---|---|---|
| decode, 1 token | $2N$ | $4N$ + cache | below 1 |
| prefill, $n$ tokens | $2Nn$ | $4N$ + cache | about $n/2$, less with the cache |

::: {.d2l-note}
Prefill is limited by arithmetic, decode by memory bandwidth.
:::
:::

::: {.slide title="Prefill Reads Tens of Thousands of Tokens per Second"}
@!kv-cache-prefill-and-decode-1

- Same layers, same GPU: decoding is at least an order of magnitude
  slower.
- Below the bandwidth ceiling: a fixed cost per step, such as launching the
  kernels of 12 blocks, adds to the weight read.
:::

::: {.slide title="Batching Amortizes Weights but Not the KV Cache"}
$b$ sequences, weights $W$ bytes, cache $c$ bytes per token, context $n$,
bandwidth $\beta$:

$$\textrm{tokens per second} \approx \frac{b\beta}{W + bnc}
\;\xrightarrow{\;b \to \infty\;}\; \frac{\beta}{nc}$$

- GPT-2 small, fp32: $W \approx$ 0.5 GB, $c$ = 72 KiB.
- Caches equal the weights at $b \approx 53$ for $n = 128$, but at
  $b \approx 7$ for $n = 1024$.
:::

::: {.slide title="Long Contexts Cap the Gain from Batching" only="jax"}
@!kv-cache-batched-decoding

Solid: measured. Dotted: the bandwidth bound at 1 TB/s. The compiled
decode loop follows the bound: at $n = 1024$ little gain beyond $b = 16$.
:::

::: {.slide title="A Fixed Cost per Step Hides the Bandwidth Bound" only="pytorch"}
::: {.cols .vc}
::: {.col}
Solid: measured. Dotted: the bandwidth bound at 1 TB/s.

- Eager PyTorch launches several hundred kernels per decode step; that
  takes longer than the GPU's work, except at $b = 64$, $n = 1024$.
- So throughput grows $\propto b$ almost throughout; CUDA graphs or
  compilation (:numref:`sec_compilation`) expose the bound.
:::

::: {.col .fig .big}
@!kv-cache-batched-decoding
:::
:::
:::

::: {.slide title="Check: Where the Cache Outweighs the Weights"}
[Check]{.kicker}

GPT-2 small in fp32 at a context of $n = 4096$ tokens. At which batch size
does the cache equal the weights, and what is the ceiling on tokens per
second at 1 TB/s?

::: {.fragment .fade-up}
- $nc = 4096 \cdot 72$ KiB = 288 MiB per sequence, so $b \approx
  500 / 302 \approx 1.7$.
- Ceiling $\beta / (nc) \approx 10^{12} / (3.0 \times 10^8) \approx 3{,}300$
  tokens per second, whatever the batch.
:::
:::

::: {.slide title="MQA and GQA Divide the Cache by the Group Size"}
@fig:mdl-transformers-gqa

- MQA: one key--value head for all $H$ query heads, cache $\div H$.
- GQA: $H_\textrm{kv}$ heads, one per group. Llama 3 8B and Mistral 7B: 32 query
  heads, 8 key--value heads.
:::

::: {.slide title="At This Scale, One Key--Value Head Costs Nothing Measurable"}
@!kv-cache-cache-against-quality

- Same byte-level GPT and recipe, $H_\textrm{kv} \in \{8, 4, 2, 1\}$: cache
  $8\times$ smaller from top to bottom, losses within a few hundredths.
- Large models (Ainslie et al., 2023): MQA loses a little, GQA at $H/8$
  comes close to multi-head attention; uptraining converts a checkpoint
  with 5% of its training steps.
:::

::: {.slide title="GPT-2's Keys and Values Fit a Fixed Low-Rank Basis"}
@!kv-cache-low-rank-keys-and-values

- Basis: top right singular vectors of keys and values on a *different*
  passage; cache $r$ numbers per token instead of 1,536.
- MLA (DeepSeek-V2/V3) learns this compression in training.
:::

::: {.slide title="GPT-2 Puts Much of Its Attention on Token 0"}
@!kv-cache-a-window-needs-a-sink-1

- Every query's weights sum to one; with nothing to retrieve, a head puts
  them on the first token, an *attention sink* (StreamingLLM, 2024).
:::

::: {.slide title="A Window Needs a Sink"}
@!kv-cache-a-window-needs-a-sink-2

- Evicting token 0 spreads its weight over tokens the heads ignored.
- Training without sinks: a learned sink logit per head (gpt-oss), or a
  sigmoid gate on each head's output (gated attention, 2025).
:::

::: {.slide title="Each Method Shrinks One Factor of the Cache Formula"}
| method | factor | cost |
|---|---|---|
| GQA, MQA | key--value heads $H_\textrm{kv}$ | train or uptrain with sharing |
| MLA, low rank | width per token | learned latent, separate rotary key |
| window + sinks | tokens $n$ | tokens outside the window |
| quantization | bytes per element | rounding error |
| linear attention | the whole cache | exact softmax recall |
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
- Heads (GQA), width (low rank, MLA), and length (window plus sinks)
  each shrink one factor of the cache.

**Next:** encoders, decoders, and cross-attention.
:::
