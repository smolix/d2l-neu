# Generation and the KV Cache
:label:`sec_kv-cache`

The `generate` method of :numref:`sec_gpt` reruns the full forward pass over
the history for every new token. This section measures the resulting cost
and introduces the *KV cache*. Because causal attention prevents past states
from depending on future tokens, the keys and values already computed for
the prefix can be stored and reused.
Caching makes the dominant dense-layer work linear in the generated
length. (Attention itself still reads the whole growing cache at every
step, so its cumulative cost stays quadratic, a term that is real but
subdominant at the sizes we run.) Caching also converts a compute problem into
a *memory* problem, so the second half of this section analyzes its storage cost.
We derive the cache-size formula and check it against the allocator, see
why generating tokens is bound by memory bandwidth while reading a prompt
is bound by arithmetic, and then shrink the cache three ways: sharing keys
and values across heads (MQA and GQA),
compressing them to a low-rank latent (the idea behind MLA), and bounding
the context with a sliding window and retaining the *attention sink*. The
experiments use the `d2l.GPT` class of the previous section, including GPT-2, which
supplies the phenomena that our one-minute character models are too small
to show.

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
from safetensors.torch import load_file
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
import time
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #kv-cache-generation-and-the-kv-cache}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
from safetensors import numpy as safetensors_numpy
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
import time
```

## From Recompute to Cache

Start with the accounting. A forward pass of a model with $N$ parameters
over $t$ tokens costs about $2Nt$ floating-point operations — every
parameter participates in one multiply–add per token — plus the attention
score work of :numref:`sec_attention-at-scale`, which is quadratic in $t$
but subdominant at the model sizes we run here. Naive generation calls
this forward pass once per token, over a history that grows by one each
step: producing $T$ tokens after a prompt costs roughly
$\sum_{t} 2Nt \approx N T^2$ operations, quadratic in the length of the
text; of the logits it computes, all but the last row is thrown away.

Almost all of that work is literally repeated. At step $t$ the model
needs, in every layer, the attention output for one new query
$\mathbf{q}_t$ against the keys and values $\mathbf{k}_{1..t},
\mathbf{v}_{1..t}$. The causal mask guarantees that positions
$1, \ldots, t-1$ never see position $t$, so their hidden states — and
therefore their keys and values — are the same at step $t$ as they were at
step $t-1$. A KV cache stores a per-layer buffer of all keys and
values computed so far, and at each step run the transformer on *one*
token, appending its $\mathbf{k}_t, \mathbf{v}_t$ to the buffer and
attending against the whole of it. :numref:`fig_kv-cache` contrasts the
two schedules. The cost of a cached step is one forward pass over a single
token, about $2N$ operations, independent of $t$.

![One decoding step at context length $t$. Naive generation recomputes every key, value, and score row of the prefix and then uses only the last row. With a KV cache, the past keys and values are read from memory, and the step computes a single new key, value, and score row.](../img/mdl-transformers-kv-cache.svg)
:label:`fig_kv-cache`

### The Cached Forward Pass

Two details need care. First, positions: RoPE rotates queries and keys by
their *absolute* position, so the attention step must know where in the
sequence its tokens sit — the `_rope` helper of `d2l.GPT` always counted
from zero, and we now need an `offset`. (A learned position table needs
the same offset, one line in the model forward.) Second, the interface:
the step receives only the *new* tokens plus the cache, instead of the
whole history.

:begin_tab:`pytorch`
We extend the `CausalAttention` of :numref:`sec_gpt` with a
`forward_step` that reads and grows a per-layer cache, held as a
dictionary of two tensors that we concatenate onto. Two cases arrive
here: *prefill*, where the whole prompt enters at once into an empty
cache and the mask must be causal, and *decode*, where a single token
attends to everything stored. These are the only two shapes the step
supports — appending a multi-token chunk to a non-empty cache would need
an offset causal mask that the `is_causal` flag cannot express, so the
assert below refuses the combination.
:end_tab:

:begin_tab:`jax`
Growing a tensor by one slot per step is unsuitable for JAX because every
new length creates a new shape and triggers a fresh XLA compilation. The
naive `generate` of :numref:`sec_gpt` already solved
this problem once, by allocating the token buffer at its final size and
overwriting it. The cache follows the same discipline, and it is exactly
how production JAX serving works: preallocate K and V buffers of shape
`(layers, batch, max_len, heads, head_dim)`, write each step's keys and
values into their slots with `dynamic_update_slice`, and mask attention
so that queries only see filled positions. The function compiles once
for the prompt shape and once for the single-token step, and the decode
loop then reuses that compilation at every step. Static shapes enable the
compiled path in this setting.
:end_tab:

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab pytorch
def rope(x, offset=0):
    """Rotary rotation of x at absolute positions offset, offset+1, ..."""
    d = x.shape[-1]
    pos = offset + torch.arange(x.shape[-2], dtype=torch.float32,
                                device=x.device)
    inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device) / d)
    theta = pos[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], -1).flatten(-2)

@d2l.add_to_class(d2l.GPT.CausalAttention)
def forward_step(self, X, cache):
    """Attention for the new tokens X against cached keys and values."""
    B, T, D = X.shape
    assert not (cache and T > 1), \
        'chunked append to a non-empty cache is unsupported ' \
        '(is_causal would misalign the mask)'
    q, k, v = self.W_qkv(X).chunk(3, -1)
    q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
               for u in (q, k, v))
    offset = cache['k'].shape[2] if cache else 0
    if self.rope:
        q, k = rope(q, offset), rope(k, offset)
    if cache:
        k = torch.cat([cache['k'], k], dim=2)
        v = torch.cat([cache['v'], v], dim=2)
    # contiguous(): at prefill k and v are still views into the fused QKV
    # projection, and caching a view would pin the whole 3x-wide buffer
    cache['k'], cache['v'] = k.contiguous(), v.contiguous()
    # Prefill (T > 1, empty cache) needs the causal mask; a single decoded
    # token (T = 1) attends to the entire cache
    Y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
    return self.W_o(Y.transpose(1, 2).reshape(B, T, D))
```

```{.python .input #kv-cache-the-cached-forward-pass-1}
%%tab jax
def rope(x, offset=0):
    """Rotary rotation of x at absolute positions offset, offset+1, ..."""
    d = x.shape[-1]
    pos = offset + jnp.arange(x.shape[1])
    inv_freq = 10000.0 ** (-jnp.arange(0, d, 2) / d)
    theta = pos[:, None] * inv_freq[None, :]
    cos = jnp.cos(theta)[:, None, :]  # broadcast over heads
    sin = jnp.sin(theta)[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], -1).reshape(x.shape)

def init_cache(model, batch_size, max_len):
    """Preallocated K and V buffers, one slab per layer."""
    attn = model.blks[0].attention
    head_dim = attn.W_o.in_features // attn.num_heads
    shape = (len(model.blks), batch_size, max_len, attn.num_heads, head_dim)
    return jnp.zeros(shape), jnp.zeros(shape)
```

The model-level step walks the blocks the way `d2l.GPT.forward` does,
handing each block's attention its cache; a `generate_cached` then
repeats the sampling loop of :numref:`sec_gpt`, prefixed by a single
prefill call, with each subsequent step feeding in one token.

```{.python .input #kv-cache-the-cached-forward-pass-2}
%%tab pytorch
@d2l.add_to_class(d2l.GPT)
@torch.no_grad()
def forward_cached(self, X, caches):
    """Forward for new tokens only, extending one cache per block."""
    offset = caches[0]['k'].shape[2] if caches[0] else 0
    H = self.token_emb(X)
    if self.pos == 'learned':
        H = H + self.pos_emb(torch.arange(offset, offset + X.shape[1],
                                          device=X.device))
    for blk, cache in zip(self.blks, caches):  # pre-norm arrangement
        H = H + blk.attention.forward_step(blk.norm1(H), cache)
        H = H + blk.ffn(blk.norm2(H))
    return F.linear(self.norm(H), self.token_emb.weight)

@d2l.add_to_class(d2l.GPT)
@torch.no_grad()
def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
    """Sample as generate does, but never recompute the prefix."""
    device = next(self.parameters()).device
    caches = [{} for _ in self.blks]
    ids = list(prefix)
    X = torch.tensor(prefix, device=device)[None]
    for _ in range(num_tokens):
        logits = self.forward_cached(X, caches)[0, -1] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, top_k).values[-1]
            logits[logits < cutoff] = -torch.inf
        ids.append(int(torch.multinomial(F.softmax(logits, -1), 1)))
        X = torch.tensor([ids[-1]], device=device)[None]
    return ids
```

```{.python .input #kv-cache-the-cached-forward-pass-2}
%%tab jax
@nnx.jit
def cached_forward(model, ks, vs, X, t):
    """Logits for the last of the new tokens X at positions t, t+1, ...;
    writes their keys and values into the cache slots at those positions."""
    B, T = X.shape
    H = model.token_emb(X)
    if model.pos == 'learned':
        H = H + model.pos_emb(t + jnp.arange(T))
    mask = (jnp.arange(ks.shape[2])[None, :]
            <= (t + jnp.arange(T))[:, None])[None, None]
    for i, blk in enumerate(model.blks):  # pre-norm arrangement
        attn, Y = blk.attention, blk.norm1(H)
        q, k, v = jnp.split(attn.W_qkv(Y), 3, axis=-1)
        q, k, v = (u.reshape(B, T, attn.num_heads, -1) for u in (q, k, v))
        if attn.rope:
            q, k = rope(q, t), rope(k, t)
        # Callers must keep t + T <= max_len: an out-of-range write does
        # not fail here -- dynamic_update_slice clamps its start index
        # silently. (t is traced under jit, so generate_cached asserts.)
        ks = jax.lax.dynamic_update_slice(ks, k[None], (i, 0, t, 0, 0))
        vs = jax.lax.dynamic_update_slice(vs, v[None], (i, 0, t, 0, 0))
        Y = jax.nn.dot_product_attention(q, ks[i], vs[i], mask=mask)
        H = H + attn.W_o(Y.reshape(B, T, -1))
        H = H + blk.ffn(blk.norm2(H))
    return model.token_emb.attend(model.norm(H))[:, -1], ks, vs

@d2l.add_to_class(d2l.GPT)
def generate_cached(self, prefix, num_tokens, seed=0, temperature=1.0,
                    top_k=None):
    """Sample as generate does, but never recompute the prefix."""
    assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
    ks, vs = init_cache(self, 1, self.max_len)
    logits, ks, vs = cached_forward(self, ks, vs,
                                    jnp.asarray(prefix)[None], jnp.array(0))
    ids, key = list(prefix), jax.random.key(seed)
    for t in range(len(prefix), len(prefix) + num_tokens):
        l = logits[0] / temperature
        if top_k is not None:
            l = jnp.where(l < jnp.sort(l)[-top_k], -jnp.inf, l)
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, l)))
        logits, ks, vs = cached_forward(self, ks, vs,
                                        jnp.asarray([[ids[-1]]]),
                                        jnp.array(t))
    return ids
```

### Same Logits, Measured

Caching should preserve the model's logits. We compare the cached path against
the full forward pass
on both positional schemes: prefill sixteen tokens, then decode one at a
time, and stack the resulting logits against those of a single
whole-sequence call.

```{.python .input #kv-cache-same-logits-measured-1}
%%tab pytorch
device = d2l.try_gpu()
for pos in ('rope', 'learned'):
    torch.manual_seed(0)
    model = d2l.GPT(vocab_size=97, num_hiddens=128, num_heads=4,
                    num_blks=3, pos=pos).to(device).eval()
    x = torch.randint(0, 97, (1, 48), device=device)
    with torch.no_grad():
        full = model(x)
    caches = [{} for _ in model.blks]
    outs = [model.forward_cached(x[:, :16], caches)]
    for t in range(16, 48):
        outs.append(model.forward_cached(x[:, t:t + 1], caches))
    err = (torch.cat(outs, 1) - full).abs().max()
    print(f'{pos}: max |full - cached| = {err:.2e}')
```

```{.python .input #kv-cache-same-logits-measured-1}
%%tab jax
with jax.default_matmul_precision('highest'):
    for pos in ('rope', 'learned'):
        model = d2l.GPT(vocab_size=97, num_hiddens=128, num_heads=4,
                        num_blks=3, pos=pos, rngs=nnx.Rngs(0))
        model.eval()
        x = jax.random.randint(jax.random.key(1), (1, 48), 0, 97)
        full = model(x)
        ks, vs = init_cache(model, 1, 64)
        logits, ks, vs = cached_forward(model, ks, vs, x[:, :16],
                                        jnp.array(0))
        outs = [logits]
        for t in range(16, 48):
            logits, ks, vs = cached_forward(model, ks, vs, x[:, t:t + 1],
                                            jnp.array(t))
            outs.append(logits)
        err = jnp.abs(jnp.stack([o[0] for o in outs]) - full[0, 15:]).max()
        print(f'{pos}: max |full - cached| = {err:.2e}')
```

:begin_tab:`jax`
(We pin matrix multiplications to full fp32 for the comparison, as in
:numref:`sec_attention-at-scale`: by default they run in TF32 on this
hardware, which perturbs the two computation orders differently at the
$10^{-4}$ level and would obscure the fact that they are the same
computation.)
:end_tab:

The cached and full computations agree to floating-point rounding. We next
time a single generation step at growing context length on a GPT-2-sized
instance of
our class — 124M parameters, untrained, since arithmetic does not care
what the weights are. The naive step is a full forward pass over $n$
tokens; the cached step is a forward pass over one token against a cache
of length $n-1$.

```{.python .input #kv-cache-same-logits-measured-2}
%%tab pytorch
torch.manual_seed(0)
model = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
                num_blks=12, max_len=4096).to(device).eval()
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')

def timeit(f, reps=10):
    f()                             # warm up
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps

lengths, t_naive, t_cached = [512, 1024, 2048, 4096], [], []
with torch.no_grad():
    for n in lengths:
        x = torch.randint(0, 50257, (1, n), device=device)
        t_naive.append(timeit(lambda: model(x)[:, -1]) * 1e3)
        caches = [{} for _ in model.blks]
        model.forward_cached(x[:, :-1], caches)   # a cache of length n-1
        one = x[:, -1:]
        # copy the dicts so every timed call extends the same-length cache
        t_cached.append(timeit(lambda: model.forward_cached(
            one, [dict(c) for c in caches])) * 1e3)
        print(f'n={n:5d}: naive {t_naive[-1]:6.2f} ms/token, '
              f'cached {t_cached[-1]:5.2f} ms/token')
d2l.plot(lengths, [t_naive, t_cached], 'context length', 'ms per token',
         legend=['naive', 'cached'], xscale='log', yscale='log')
```

```{.python .input #kv-cache-same-logits-measured-2}
%%tab jax
model = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
                num_blks=12, max_len=4096, rngs=nnx.Rngs(0))
model.eval()
n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f'{n_params / 1e6:.1f}M parameters')

@nnx.jit
def naive_step(model, buf, t):
    """The step inside the naive generate: a full forward pass."""
    return model(buf)[0, t - 1]

def timeit(f, reps=10):
    f().block_until_ready()         # warm up (and compile)
    t0 = time.time()
    for _ in range(reps):
        f().block_until_ready()
    return (time.time() - t0) / reps

lengths, t_naive, t_cached = [512, 1024, 2048, 4096], [], []
for n in lengths:
    buf = jnp.zeros((1, n), dtype=jnp.int32)
    t_naive.append(
        timeit(lambda: naive_step(model, buf, jnp.array(n - 1))) * 1e3)
    ks, vs = init_cache(model, 1, n)
    one = jnp.zeros((1, 1), dtype=jnp.int32)
    t_cached.append(timeit(lambda: cached_forward(
        model, ks, vs, one, jnp.array(n - 1))[0]) * 1e3)
    print(f'n={n:5d}: naive {t_naive[-1]:6.2f} ms/token, '
          f'cached {t_cached[-1]:5.2f} ms/token')
d2l.plot(lengths, [t_naive, t_cached], 'context length', 'ms per token',
         legend=['naive', 'cached'], xscale='log', yscale='log')
```

Read the plot from the right. At long contexts the naive step grows
roughly linearly with $n$, as $2Nn$ arithmetic says it must, while the
cached step stays flat; at a context of four thousand tokens the gap in
our runs is six- to sevenfold, and it widens with every further
doubling of context. At short contexts the two curves *merge* — below
about a thousand tokens this model is so small that a step of either kind
is dominated by launching a dozen blocks' worth of GPU kernels, not by
computing. The cache is not a constant-factor optimization: it removes a term
that grows with context. A measurable latency reduction appears once that term
is large relative to kernel-launch and other fixed overheads.
End to end, the effect on `generate` is what the per-step curve promises:

```{.python .input #kv-cache-same-logits-measured-3}
%%tab pytorch
prefix = list(range(3584))
for name, gen in (('naive ', model.generate),
                  ('cached', model.generate_cached)):
    torch.manual_seed(0)
    gen(list(range(8)), 8)          # warm up
    torch.manual_seed(0)
    torch.cuda.synchronize()
    t0 = time.time()
    gen(prefix, 256)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{name}: 256 tokens after a 3584-token prompt: {dt:4.1f}s '
          f'= {256 / dt:5.1f} tokens/s')
```

```{.python .input #kv-cache-same-logits-measured-3}
%%tab jax
prefix = list(range(3584))
for name, gen in (('naive ', model.generate),
                  ('cached', model.generate_cached)):
    t0 = time.time()
    gen(prefix, 256, seed=0)
    dt = time.time() - t0
    print(f'{name}: 256 tokens after a 3584-token prompt: {dt:4.1f}s '
          f'= {256 / dt:5.1f} tokens/s')
```

:begin_tab:`pytorch`
Several times faster at this prompt length, with the naive quadratic
growing worse from here and the cached line staying put. Production
serving systems generate this way; "time to first token" is our
prefill call, and "tokens per second" is the cached decode loop.
:end_tab:

:begin_tab:`jax`
More than twice as fast at this prompt length — each timing includes one
XLA compilation for its shapes, so the steady-state gap is larger still, as
the per-step curve above shows. Production serving systems generate this
way; "time to first token" is our prefill call, and "tokens per second"
is the cached decode loop.
:end_tab:

## Memory Required by the KV Cache

The cache consumes GPU memory. Each of the $n_\textrm{layers}$ layers stores one key and one value
vector of dimension $n_\textrm{kv} \cdot d_\textrm{head}$ per token, so a
batch of $b$ sequences of length $n$ holds

$$
\textrm{cache bytes} \;=\; 2 \cdot n_\textrm{layers} \cdot n_\textrm{kv}
\cdot d_\textrm{head} \cdot n \cdot b \cdot (\textrm{bytes per element}),
$$
:eqlabel:`eq_kv-cache-bytes`

where the leading 2 counts K and V and $n_\textrm{kv}$ is the number of
key-value heads (equal to the number of query heads, for now: the next
section breaks that equality). For our GPT-2-sized model in fp32 that is
$2 \cdot 12 \cdot 12 \cdot 64 \cdot 4$ bytes $= 72$ KiB per token: at a
context of 4096 the cache holds 288 MiB, more than half the size of the
124M-parameter model itself. Scale the formula up to a modern deployment
— dozens of layers, contexts in the hundreds of thousands, batches of
concurrent users — and the cache, not the weights, is what fills the
accelerator; the exercises make this concrete for a 70B model. Formulas
should be checked, not trusted:

```{.python .input #kv-cache-the-memory-bill}
%%tab pytorch
for n in (1024, 2048, 4096):
    caches = [{} for _ in model.blks]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    model.forward_cached(torch.randint(0, 50257, (1, n), device=device),
                         caches)
    torch.cuda.synchronize()
    measured = torch.cuda.memory_allocated() - base
    formula = 2 * 12 * 12 * 64 * n * 4
    print(f'n={n}: allocator grew {measured / 2**20:6.1f} MiB, '
          f'formula says {formula / 2**20:6.1f} MiB')
```

```{.python .input #kv-cache-the-memory-bill}
%%tab jax
ks, vs = init_cache(model, 1, 4096)
formula = 2 * 12 * 12 * 64 * 4096 * 4
print(f'cache buffers: {(ks.nbytes + vs.nbytes) / 2**20:.0f} MiB, '
      f'formula says {formula / 2**20:.0f} MiB')
```

:begin_tab:`pytorch`
The allocator's growth matches :eqref:`eq_kv-cache-bytes` to within a
rounding error — what remains allocated after a prefill is exactly the
cache tensors. That exactness took one deliberate line: `forward_step`
stores `contiguous()` copies, because the freshly projected keys and
values are *views* into the fused QKV buffer, and caching a view keeps
the whole three-times-wider buffer alive. In an early draft of this cell
the allocator reported twice the formula, and that line is how the gap
was closed; measuring memory is how such quiet leaks are found.
:end_tab:

:begin_tab:`jax`
In JAX the check is almost circular, and that is itself the lesson: the
fixed-size discipline allocates the full cache up front, so
:eqref:`eq_kv-cache-bytes` is not a prediction about hidden allocator
behavior but the literal size of two arrays you created. A serving
configuration that does not fit fails at initialization, not mid-request.
:end_tab:

### Prefill Is Compute-Bound, Decode Is Memory-Bound

Why obsess over the size of a buffer that merely *sits* in memory?
Because at generation time it does not sit — it moves. Every decoded
token must read every cached key and value once, and, more importantly,
must read *every parameter of the model* once: that is what "a forward
pass over one token" means. The useful lens is *arithmetic intensity*:
FLOPs performed per byte of memory traffic. A decode step performs about
$2N$ FLOPs and moves at least the $4N$ bytes of fp32 weights plus the
cache, an intensity below one FLOP per byte. A prefill over $n$ tokens
performs $2Nn$ FLOPs against the *same* weight traffic (the weights are
read once and reused for every token in the batch of positions), an
intensity roughly $n$ times higher. A modern GPU is balanced at far
higher intensities (an RTX 4090 delivers up to about $83$ TFLOP/s of
fp32 compute but only about $1$ TB/s of memory bandwidth, a ridge near
80 FLOPs per byte), so prefill lands comfortably compute-bound while
single-stream decode cannot even in principle keep the arithmetic units
busy: it is a memory-bandwidth workload
:cite:`Pope.Douglas.Chowdhery.ea.2023`. The two phases of one `generate`
call live on opposite sides of the roofline.

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab pytorch
N = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    x = torch.randint(0, 50257, (1, 2048), device=device)
    t_prefill = timeit(
        lambda: model.forward_cached(x, [{} for _ in model.blks]), reps=5)
    caches = [{} for _ in model.blks]
    model.forward_cached(x, caches)
    one = x[:, :1]
    t_decode = timeit(
        lambda: model.forward_cached(one, [dict(c) for c in caches]))
print(f'prefill: 2048 tokens in {t_prefill * 1e3:4.0f} ms = '
      f'{2048 / t_prefill:6.0f} tokens/s')
print(f'decode:  one token in {t_decode * 1e3:6.2f} ms = '
      f'{1 / t_decode:6.0f} tokens/s')
cache = 2 * 12 * 12 * 64 * 2048 * 4
flops, moved = 2 * N, 4 * N + cache
print(f'decode intensity: {flops / 1e6:.0f} MFLOP / {moved / 1e6:.0f} MB '
      f'= {flops / moved:.1f} FLOP/byte (GPU ridge is near 80)')
print(f'prefill intensity: about {flops * 2048 / moved:.0f} FLOP/byte')
print(f'bandwidth ceiling for decode: about '
      f'{1.0e12 / moved:.0f} tokens/s')
```

```{.python .input #kv-cache-prefill-is-compute-bound-decode-is-memory-bound}
%%tab jax
N = n_params
x = jnp.zeros((1, 2048), dtype=jnp.int32)
ks, vs = init_cache(model, 1, 2048)
t_prefill = timeit(
    lambda: cached_forward(model, ks, vs, x, jnp.array(0))[0], reps=5)
one = jnp.zeros((1, 1), dtype=jnp.int32)
t_decode = timeit(
    lambda: cached_forward(model, ks, vs, one, jnp.array(2047))[0])
print(f'prefill: 2048 tokens in {t_prefill * 1e3:4.0f} ms = '
      f'{2048 / t_prefill:6.0f} tokens/s')
print(f'decode:  one token in {t_decode * 1e3:6.2f} ms = '
      f'{1 / t_decode:6.0f} tokens/s')
cache = 2 * 12 * 12 * 64 * 2048 * 4
flops, moved = 2 * N, 4 * N + cache
print(f'decode intensity: {flops / 1e6:.0f} MFLOP / {moved / 1e6:.0f} MB '
      f'= {flops / moved:.1f} FLOP/byte (GPU ridge is near 80)')
print(f'prefill intensity: about {flops * 2048 / moved:.0f} FLOP/byte')
print(f'bandwidth ceiling for decode: about '
      f'{1.0e12 / moved:.0f} tokens/s')
```

The measurement shows that this model reads a two-thousand-token prompt at
tens of thousands of tokens per second and then generates at under a
hundred — nearly three orders of magnitude apart, on identical hardware,
running identical layers. Note also the gap between our measured decode rate and
the bandwidth ceiling the arithmetic promises: a Python loop that
launches every kernel of every block for each token incurs overheads that
production engines reduce with compiled decode loops and CUDA graphs.
Nevertheless, decode speed is set largely by the *bytes moved per token*,
including weights and cache. Reducing the cache can therefore increase
decode speed, permit longer contexts, or support more concurrent users.

## Sharing Keys and Values across Heads

Look again at :eqref:`eq_kv-cache-bytes`: the factor $n_\textrm{kv}$ is
the number of key-value heads, and nothing in the attention mechanism
forces it to equal the number of query heads. Queries do the asking;
keys and values are the library being consulted. *Multi-query attention*
(MQA) keeps all $H$ query heads but a single shared key-value head
:cite:`Shazeer.2019`, shrinking the cache by a factor of $H$ at a
usually slight cost in quality. *Grouped-query attention* (GQA)
interpolates: $H_{kv}$
key-value heads, each serving a contiguous group of $H / H_{kv}$ query
heads :cite:`Ainslie.Lee-Thorp.Jong.ea.2023`. :numref:`fig_gqa` shows
the sharing patterns. GQA with $H_{kv} = H/4$ to $H/8$ is close to the
universal choice of current open models — Llama 3 runs 32 query heads
against 8 key-value heads, Mistral 7B the same
:cite:`Grattafiori.Dubey.Jauhri.ea.2024,Jiang.Sablayrolles.Mensch.ea.2023`.

![Query heads and the key-value heads they read. Multi-head attention gives every query head its own key-value head; grouped-query attention shares one per group of query heads; multi-query attention shares a single one across all of them. The cache scales with the number of key-value heads.](../img/mdl-transformers-gqa.svg)
:label:`fig_gqa`

### A Pluggable Implementation

`GQAAttention` below generalizes the causal attention of
:numref:`sec_gpt`: `W_q` still produces $H$ query heads, but `W_k` and
`W_v` produce only $H_{kv}$ heads, and the fused kernel broadcasts each
key-value head across its query group (in PyTorch via `enable_gqa`; the
JAX kernel accepts grouped shapes natively). We keep the
`(queries, keys, values, valid_lens)` call shape of
`d2l.MultiHeadAttention`, so the class drops into
`d2l.TransformerBlock`'s `attn_factory` hook, and with `rope=True` into
`d2l.GPT` as well; that hook is the seam :numref:`sec_transformer-block`
built for exactly this purpose. Setting $H_{kv} = H$ recovers standard
multi-head attention exactly.

```{.python .input #kv-cache-a-pluggable-implementation-1}
%%tab pytorch
class GQAAttention(nn.Module):  #@save
    """Causal multi-head attention with num_kv_heads shared KV heads."""
    def __init__(self, num_hiddens, num_heads, num_kv_heads, bias=False,
                 rope=False, causal=True):
        super().__init__()
        assert num_hiddens % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = num_hiddens // num_heads
        self.rope, self.causal = rope, causal
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_kv_heads * self.head_dim,
                             bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_kv_heads * self.head_dim,
                             bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def _rope(self, x):  # the rotation of d2l.GPT, kept self-contained
        d = x.shape[-1]
        pos = torch.arange(x.shape[-2], dtype=torch.float32,
                           device=x.device)
        inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device) / d)
        theta = pos[:, None] * inv_freq[None, :]
        cos, sin = torch.cos(theta), torch.sin(theta)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, queries, keys, values, valid_lens=None):
        B, T, D = queries.shape
        q = self.W_q(queries).reshape(B, T, self.num_heads, -1)
        k = self.W_k(keys).reshape(B, -1, self.num_kv_heads, self.head_dim)
        v = self.W_v(values).reshape(B, -1, self.num_kv_heads,
                                     self.head_dim)
        q, k, v = (u.transpose(1, 2) for u in (q, k, v))
        if self.rope:
            q, k = self._rope(q), self._rope(k)
        mask, causal = None, self.causal
        if valid_lens is not None:      # mask padding (and causality)
            S = k.shape[2]
            if valid_lens.dim() == 1:
                valid_lens = valid_lens[:, None].expand(B, T)
            mask = (torch.arange(S, device=q.device)[None, None, :]
                    < valid_lens[:, :, None])[:, None]
            if causal:
                i = torch.arange(T, device=q.device)[:, None]
                j = torch.arange(S, device=q.device)[None, :]
                mask = mask & (j <= i + S - T)
            causal = False
        Y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=causal,
                                           enable_gqa=True)
        return self.W_o(Y.transpose(1, 2).reshape(B, T, -1))
```

```{.python .input #kv-cache-a-pluggable-implementation-1}
%%tab jax
class GQAAttention(nnx.Module):  #@save
    """Causal multi-head attention with num_kv_heads shared KV heads."""
    def __init__(self, num_hiddens, num_heads, num_kv_heads, bias=False,
                 rope=False, causal=True, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        assert num_hiddens % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = num_hiddens // num_heads
        self.rope, self.causal = rope, causal
        self.W_q = nnx.Linear(num_hiddens, num_hiddens, use_bias=bias,
                              rngs=rngs)
        self.W_k = nnx.Linear(num_hiddens, num_kv_heads * self.head_dim,
                              use_bias=bias, rngs=rngs)
        self.W_v = nnx.Linear(num_hiddens, num_kv_heads * self.head_dim,
                              use_bias=bias, rngs=rngs)
        self.W_o = nnx.Linear(num_hiddens, num_hiddens, use_bias=bias,
                              rngs=rngs)

    def _rope(self, x):  # the rotation of d2l.GPT, kept self-contained
        d = x.shape[-1]
        pos = jnp.arange(x.shape[1], dtype=jnp.float32)
        inv_freq = 10000.0 ** (-jnp.arange(0, d, 2) / d)
        theta = pos[:, None] * inv_freq[None, :]
        cos = jnp.cos(theta)[:, None, :]
        sin = jnp.sin(theta)[:, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return jnp.stack([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], -1).reshape(x.shape)

    def __call__(self, queries, keys, values, valid_lens=None):
        B, T, D = queries.shape
        q = self.W_q(queries).reshape(B, T, self.num_heads, -1)
        k = self.W_k(keys).reshape(B, -1, self.num_kv_heads, self.head_dim)
        v = self.W_v(values).reshape(B, -1, self.num_kv_heads,
                                     self.head_dim)
        if self.rope:
            q, k = self._rope(q), self._rope(k)
        mask, causal = None, self.causal
        if valid_lens is not None:      # mask padding (and causality)
            S = k.shape[1]
            if valid_lens.ndim == 1:
                valid_lens = jnp.broadcast_to(valid_lens[:, None], (B, T))
            mask = (jnp.arange(S)[None, None, :]
                    < valid_lens[:, :, None])[:, None]
            if causal:
                i, j = jnp.arange(T)[:, None], jnp.arange(S)[None, :]
                mask = mask & (j <= i + S - T)
            causal = False
        Y = jax.nn.dot_product_attention(q, k, v, mask=mask,
                                         is_causal=causal)
        return self.W_o(Y.reshape(B, T, -1)), None
```

It honors the block contract, and its parameter count records where the
saving comes from: `W_k` and `W_v` shrink with $H_{kv}$ (attention
parameters go from $4d^2$ toward $2d^2$), while the cache line of
:eqref:`eq_kv-cache-bytes` shrinks in proportion.

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab pytorch
X = torch.ones(2, 10, 256)
blk = d2l.TransformerBlock(
    256, num_heads=8,
    attn_factory=lambda: GQAAttention(256, 8, 2, causal=False))
d2l.check_shape(blk(X, torch.tensor([7, 4])), X.shape)
for G in (8, 4, 2, 1):
    attn = GQAAttention(256, 8, G)
    n = sum(p.numel() for p in attn.parameters())
    print(f'H_kv={G}: {n:6d} parameters, '
          f'cache {2 * G * 32 * 4:4d} bytes per token per layer')
```

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab jax
X = jnp.ones((2, 10, 256))
blk = d2l.TransformerBlock(
    256, num_heads=8,
    attn_factory=lambda rngs: GQAAttention(256, 8, 2, causal=False,
                                           rngs=rngs))
d2l.check_shape(blk(X, jnp.array([7, 4])), X.shape)
for G in (8, 4, 2, 1):
    attn = GQAAttention(256, 8, G)
    n = sum(p.size for p in jax.tree.leaves(nnx.state(attn, nnx.Param)))
    print(f'H_kv={G}: {n:6d} parameters, '
          f'cache {2 * G * 32 * 4:4d} bytes per token per layer')
```

### GQA Cache Size and Validation Loss

What does the sharing cost in modeling power? We train the
:numref:`sec_gpt` configuration on the character-level Time Machine at
every group count from $H_{kv} = 8$ (full multi-head) down to $1$ (MQA),
swapping the attention in each block after construction. Because the
validation loss of this small model bottoms out within a few hundred
steps and then climbs into memorization (:numref:`sec_gpt`), we compare
models at their *best* validation loss over a fixed 600-step budget.
Each row below is one fixed-seed run evaluated on the same validation split.
This design can reveal differences larger than ordinary run-to-run variation,
but it cannot estimate a confidence interval or establish equivalence.

```{.python .input #kv-cache-cache-against-quality}
%%tab pytorch
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)

def val_loss(model, data):
    model.to(device).eval()
    with torch.no_grad():
        losses = [F.cross_entropy(
            model(X.to(device)).flatten(0, 1), Y.to(device).flatten())
            for X, Y in data.val_dataloader()]
    model.train()
    return sum(l.item() for l in losses) / len(losses)

for G in (8, 4, 2, 1):
    torch.manual_seed(0)
    lm = d2l.GPT(len(data.vocab), num_hiddens=256, num_heads=8,
                 num_blks=6)
    for blk in lm.blks:
        blk.attention = GQAAttention(256, 8, G, rope=True)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3,
                                  weight_decay=0.0)
    best = float('inf')
    for chunk in range(6):
        d2l.train_lm(lm, data, optimizer, 100)
        best = min(best, val_loss(lm, data))
    print(f'H_kv={G}: best validation loss {best:.2f}, '
          f'cache {2 * 6 * G * 32 * 4 / 1024:4.1f} KiB per token')
```

```{.python .input #kv-cache-cache-against-quality}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char',
                       num_train=100000, num_val=3000)

@nnx.jit
def batch_loss(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

def val_loss(model, data):
    model.eval()
    losses = [float(batch_loss(model, jnp.asarray(X), jnp.asarray(Y)))
              for X, Y in data.val_dataloader()]
    model.train()
    return sum(losses) / len(losses)

for G in (8, 4, 2, 1):
    rngs = nnx.Rngs(0)
    lm = d2l.GPT(len(data.vocab), num_hiddens=256, num_heads=8,
                 num_blks=6, rngs=rngs)
    for blk in lm.blks:
        blk.attention = GQAAttention(256, 8, G, rope=True, rngs=rngs)
    optimizer = nnx.Optimizer(lm, optax.adamw(1e-3, weight_decay=0.0),
                              wrt=nnx.Param)
    best = float('inf')
    for chunk in range(6):
        d2l.train_lm(lm, data, optimizer, 100)
        best = min(best, val_loss(lm, data))
    print(f'H_kv={G}: best validation loss {best:.2f}, '
          f'cache {2 * 6 * G * 32 * 4 / 1024:4.1f} KiB per token')
```

The cache shrinks eightfold from top to bottom of the table. Each configuration
uses one initialization per framework, 600 training steps, and a validation
corpus configured with `num_val=3000`; losses are displayed to 0.01 nat. The
rows differ by only a few hundredths, but without repeated seeds this protocol
provides no uncertainty estimate or equivalence test. In larger-scale
ablations, full
MQA incurs a small quality reduction, while GQA at $H/8$ matches multi-head
attention under the protocol of
:citet:`Ainslie.Lee-Thorp.Jong.ea.2023` — and an existing multi-head
checkpoint can be *uptrained* into a grouped one for a small fraction of
its original compute, which is how GQA spread through the open-model
world so quickly.

## Compressing the Cache Further

Sharing heads is one axis. This closing section looks at the two other
levers deployed systems pull: compressing the *width* of each cached
token further, and bounding the *length* of what is cached. The second
comes with a failure mode that makes it subtle. Our minute-trained
character models are too small to exhibit these phenomena, so we probe
the real GPT-2, reloading it exactly as in :numref:`sec_gpt` (weights
and tokenizer files are already pinned in `d2l.DATA_HUB`), along with the
raw text of *The Time Machine* as a 1024-token evaluation passage.

```{.python .input #kv-cache-compressing-the-cache-further-1}
%%tab pytorch
HF_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    HF_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    HF_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    HF_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')
enc = tiktoken.Encoding(
    'gpt2', explicit_n_vocab=50257,
    pat_str=d2l.BPETokenizer.GPT2_PATTERN,
    mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
        d2l.download('gpt2-merges', '../data/gpt2'),
        d2l.download('gpt2-encoder', '../data/gpt2')),
    special_tokens={'<|endoftext|>': 50256})

gpt2 = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
               num_blks=12, max_len=1024, pos='learned', norm='layer',
               act='gelu', pre_norm=True, bias=True)
weights = load_file(d2l.download('gpt2-weights', '../data/gpt2'))
with torch.no_grad():
    gpt2.token_emb.weight.copy_(weights['wte.weight'])
    gpt2.pos_emb.weight.copy_(weights['wpe.weight'])
    gpt2.norm.weight.copy_(weights['ln_f.weight'])
    gpt2.norm.bias.copy_(weights['ln_f.bias'])
    for i, blk in enumerate(gpt2.blks):
        modules = {f'h.{i}.ln_1': blk.norm1, f'h.{i}.ln_2': blk.norm2,
                   f'h.{i}.attn.c_attn': blk.attention.W_qkv,
                   f'h.{i}.attn.c_proj': blk.attention.W_o,
                   f'h.{i}.mlp.c_fc': blk.ffn.W_1,
                   f'h.{i}.mlp.c_proj': blk.ffn.W_2}
        for key, module in modules.items():
            W = weights[key + '.weight']
            module.weight.copy_(W.T if W.ndim == 2 else W)  # Conv1D layout
            module.bias.copy_(weights[key + '.bias'])
gpt2.to(device).eval()

fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', '../data',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')
ids = enc.encode(open(fname).read())[:1024]
x = torch.tensor(ids, device=device)[None]
print(f'{len(ids)} tokens of The Time Machine')
```

```{.python .input #kv-cache-compressing-the-cache-further-1}
%%tab jax
HF_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    HF_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    HF_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    HF_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')
enc = tiktoken.Encoding(
    'gpt2', explicit_n_vocab=50257,
    pat_str=d2l.BPETokenizer.GPT2_PATTERN,
    mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
        d2l.download('gpt2-merges', '../data/gpt2'),
        d2l.download('gpt2-encoder', '../data/gpt2')),
    special_tokens={'<|endoftext|>': 50256})

gpt2 = d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
               num_blks=12, max_len=1024, pos='learned', norm='layer',
               act='gelu', pre_norm=True, bias=True)
weights = safetensors_numpy.load_file(
    d2l.download('gpt2-weights', '../data/gpt2'))
gpt2.token_emb.embedding[...] = jnp.asarray(weights['wte.weight'])
gpt2.pos_emb.embedding[...] = jnp.asarray(weights['wpe.weight'])
gpt2.norm.scale[...] = jnp.asarray(weights['ln_f.weight'])
gpt2.norm.bias[...] = jnp.asarray(weights['ln_f.bias'])
for i, blk in enumerate(gpt2.blks):
    for norm, key in ((blk.norm1, f'h.{i}.ln_1'),
                      (blk.norm2, f'h.{i}.ln_2')):
        norm.scale[...] = jnp.asarray(weights[key + '.weight'])
        norm.bias[...] = jnp.asarray(weights[key + '.bias'])
    for linear, key in ((blk.attention.W_qkv, f'h.{i}.attn.c_attn'),
                        (blk.attention.W_o, f'h.{i}.attn.c_proj'),
                        (blk.ffn.W_1, f'h.{i}.mlp.c_fc'),
                        (blk.ffn.W_2, f'h.{i}.mlp.c_proj')):
        linear.kernel[...] = jnp.asarray(weights[key + '.weight'])
        linear.bias[...] = jnp.asarray(weights[key + '.bias'])
gpt2.eval()

fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', '../data',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')
ids = enc.encode(open(fname).read())[:1024]
x = jnp.asarray(ids)[None]
print(f'{len(ids)} tokens of The Time Machine')
```

Each experiment below changes only what happens *inside* attention, so
one helper suffices: a forward pass that walks GPT-2's blocks exactly as
the class does, but lets us substitute the attention computation per
block. First a sanity check that the helper with stock attention
reproduces the model, and a baseline loss on the second half of the
passage (positions 512 onward, so every evaluated token has a long
history to draw on).

```{.python .input #kv-cache-compressing-the-cache-further-2}
%%tab pytorch
def forward_blocks(model, X, attn_fn):
    """GPT-2 forward with a pluggable attention computation per block."""
    H = model.token_emb(X) + model.pos_emb(
        torch.arange(X.shape[1], device=X.device))
    for i, blk in enumerate(model.blks):
        A, Y = blk.attention, blk.norm1(H)
        B, T, D = Y.shape
        q, k, v = (u.reshape(B, T, A.num_heads, -1).transpose(1, 2)
                   for u in A.W_qkv(Y).chunk(3, -1))
        H = H + A.W_o(attn_fn(q, k, v, i).transpose(1, 2).reshape(B, T, D))
        H = H + blk.ffn(blk.norm2(H))
    return F.linear(model.norm(H), model.token_emb.weight)

def tail_loss(logits):
    """Mean per-token loss on the second half of the passage."""
    return F.cross_entropy(logits[0, 512:-1], x[0, 513:])

stock = lambda q, k, v, i: F.scaled_dot_product_attention(
    q, k, v, is_causal=True)
with torch.no_grad():
    logits = forward_blocks(gpt2, x, stock)
    print(f'helper vs stock forward: max deviation '
          f'{(logits - gpt2(x)).abs().max():.1e}')
    full = tail_loss(logits)
print(f'full attention: loss {full:.2f}, perplexity {full.exp():.0f}')
```

```{.python .input #kv-cache-compressing-the-cache-further-2}
%%tab jax
def forward_blocks(model, X, attn_fn):
    """GPT-2 forward with a pluggable attention computation per block."""
    H = model.token_emb(X) + model.pos_emb(jnp.arange(X.shape[1]))
    for i, blk in enumerate(model.blks):
        A, Y = blk.attention, blk.norm1(H)
        B, T, D = Y.shape
        q, k, v = (u.reshape(B, T, A.num_heads, -1)
                   for u in jnp.split(A.W_qkv(Y), 3, axis=-1))
        H = H + A.W_o(attn_fn(q, k, v, i).reshape(B, T, D))
        H = H + blk.ffn(blk.norm2(H))
    return model.token_emb.attend(model.norm(H))

def tail_loss(logits):
    """Mean per-token loss on the second half of the passage."""
    return optax.softmax_cross_entropy_with_integer_labels(
        logits[0, 512:-1], x[0, 513:]).mean()

stock = lambda q, k, v, i: jax.nn.dot_product_attention(q, k, v,
                                                        is_causal=True)
logits = forward_blocks(gpt2, x, stock)
print(f'helper vs stock forward: max deviation '
      f'{jnp.abs(logits - gpt2(x)).max():.1e}')
full = tail_loss(logits)
print(f'full attention: loss {full:.2f}, perplexity {jnp.exp(full):.0f}')
```

### Low-Rank Keys and Values

GQA shrinks the cache by storing fewer key-value heads. A different bet:
keep all heads, but suppose that a token's concatenated key and value
vectors do not really span the full width they occupy (for GPT-2,
$2 \times 768 = 1536$ numbers), and store only their coordinates in some
lower-dimensional subspace. That is the idea behind *multi-head latent
attention* (MLA), the mechanism DeepSeek built its V2 and V3 models
around :cite:`DeepSeek-AI.2024`: a trained projection compresses each
token's keys and values jointly into one latent vector (rank 512, versus
tens of thousands of raw KV dimensions, in DeepSeek-V2), the cache
stores only the latent, and per-head keys and values are reconstructed
by trained up-projections at read time.

We can test the load-bearing assumption — that realized keys and values
are approximately low-rank — directly on GPT-2. In each layer we take
the actual K and V matrices of our passage, replace them by their best
rank-$r$ approximation (via SVD, jointly across the concatenated KV
width of 1536), and run the model with every layer so truncated. This
uses an oracle factorization computed on the very sequence being
evaluated, so it is an upper bound on what a fixed trained projection
could do; MLA's contribution is making the projection trainable, and
handling the one real obstacle: RoPE's position-dependent rotation does
not commute with a shared down-projection, which MLA solves by carrying
a small separate rotary component per token. None of that machinery is
needed to check the premise:

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab pytorch
def lowrank_kv(r):
    def attn_fn(q, k, v, i):
        B, H, T, hd = k.shape
        KV = torch.cat([k.transpose(1, 2).reshape(T, -1),
                        v.transpose(1, 2).reshape(T, -1)], dim=1)
        U, S, Vh = torch.linalg.svd(KV, full_matrices=False)
        KVr = (U[:, :r] * S[:r]) @ Vh[:r]     # best rank-r approximation
        kr = KVr[:, :H * hd].reshape(1, T, H, hd).transpose(1, 2)
        vr = KVr[:, H * hd:].reshape(1, T, H, hd).transpose(1, 2)
        return F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    return attn_fn

with torch.no_grad():
    for r in (256, 128):
        loss = tail_loss(forward_blocks(gpt2, x, lowrank_kv(r)))
        print(f'rank {r} of 1536 ({1536 // r}x smaller cache): '
              f'loss {loss:.2f}, perplexity {loss.exp():.0f}')
```

```{.python .input #kv-cache-low-rank-keys-and-values}
%%tab jax
def lowrank_kv(r):
    def attn_fn(q, k, v, i):
        B, T, H, hd = k.shape
        KV = jnp.concatenate([k.reshape(T, -1), v.reshape(T, -1)], axis=1)
        U, S, Vh = jnp.linalg.svd(KV, full_matrices=False)
        KVr = (U[:, :r] * S[:r]) @ Vh[:r]     # best rank-r approximation
        kr = KVr[:, :H * hd].reshape(1, T, H, hd)
        vr = KVr[:, H * hd:].reshape(1, T, H, hd)
        return jax.nn.dot_product_attention(q, kr, vr, is_causal=True)
    return attn_fn

for r in (256, 128):
    loss = tail_loss(forward_blocks(gpt2, x, lowrank_kv(r)))
    print(f'rank {r} of 1536 ({1536 // r}x smaller cache): '
          f'loss {loss:.2f}, perplexity {jnp.exp(loss):.0f}')
```

At rank 256, loss on this one passage changes by less than the displayed
precision; at rank 128 it increases. This oracle SVD is fitted to the evaluated
sequence, so it neither implements a cache reduction nor shows that one fixed
projection would generalize. It establishes only that the realized K/V matrix
for this passage admits a low-rank approximation. MLA instead learns a shared
projection during training and retains a separate rotary component.

### Sliding Windows and Attention Sinks

The remaining lever is length: cap the cache at the last $w$ tokens,
evicting the oldest as generation proceeds, and memory is bounded
forever — the rolling-buffer form of the sliding-window attention of
:numref:`sec_attention-at-scale`. Before evicting anything, it is worth
asking where a trained model actually *sends* its attention. We measure
the average weight that GPT-2's queries (from position 64 on) place on
the single first token of the sequence:

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab pytorch
masses = []
def watch(q, k, v, i):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    j = torch.arange(scores.shape[-1], device=scores.device)
    scores.masked_fill_(j[None, :] > j[:, None],
                        torch.finfo(scores.dtype).min)
    alpha = torch.softmax(scores, -1)
    masses.append(alpha[..., 64:, 0].mean().item())
    return alpha @ v

with torch.no_grad():
    forward_blocks(gpt2, x, watch)
print('mean attention weight on token 0 (uniform would be ~0.003):')
d2l.print_wrapped('  '.join(f'layer {i}: {m:.2f}'
                            for i, m in enumerate(masses)))
```

```{.python .input #kv-cache-a-window-needs-a-sink-1}
%%tab jax
masses = []
def watch(q, k, v, i):
    qh, kh, vh = (u.transpose(0, 2, 1, 3) for u in (q, k, v))
    scores = qh @ kh.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    j = jnp.arange(scores.shape[-1])
    scores = jnp.where(j[None, :] > j[:, None],
                       jnp.finfo(scores.dtype).min, scores)
    alpha = jax.nn.softmax(scores, axis=-1)
    masses.append(float(alpha[..., 64:, 0].mean()))
    return (alpha @ vh).transpose(0, 2, 1, 3)

forward_blocks(gpt2, x, watch)
print('mean attention weight on token 0 (uniform would be ~0.003):')
d2l.print_wrapped('  '.join(f'layer {i}: {m:.2f}'
                            for i, m in enumerate(masses)))
```

From the middle of the stack upward, the measured GPT-2 heads assign roughly a
third to a half of their attention mass to the first token, far above a uniform
reference. :citet:`Xiao.Tian.Chen.ea.2024` call such positions *attention
sinks* and interpret them as locations that absorb probability mass when a
head has no useful retrieval. Our measurement establishes the weight pattern
for one passage; it does not test that functional explanation. The effect of
the first token also depends on its value vector, which we have not measured.

Evicting that token, as a rolling buffer of the most recent $w$ entries would,
removes this high-attention position:
the discarded mass is renormalized onto tokens the head actively chose
not to attend to. StreamingLLM retains the first few tokens in the cache
alongside the sliding window
:cite:`Xiao.Tian.Chen.ea.2024`. We reproduce both the failure and the
fix, evaluating our passage with attention restricted to a 256-token
window plus a varying number of retained *sink tokens*:

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab pytorch
j = torch.arange(x.shape[1], device=device)
causal = j[None, :] <= j[:, None]
window = causal & (j[:, None] - j[None, :] < 256)
with torch.no_grad():
    for nsink in (0, 1, 4):
        mask = window | (causal & (j[None, :] < nsink))
        loss = tail_loss(forward_blocks(
            gpt2, x, lambda q, k, v, i: F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask)))
        print(f'window 256 + {nsink} sink tokens: loss {loss:.2f}, '
              f'perplexity {loss.exp():.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {full.exp():.0f}')
```

```{.python .input #kv-cache-a-window-needs-a-sink-2}
%%tab jax
j = jnp.arange(x.shape[1])
causal = j[None, :] <= j[:, None]
window = causal & (j[:, None] - j[None, :] < 256)
for nsink in (0, 1, 4):
    mask = window | (causal & (j[None, :] < nsink))
    loss = tail_loss(forward_blocks(
        gpt2, x, lambda q, k, v, i: jax.nn.dot_product_attention(
            q, k, v, mask=mask[None, None])))
    print(f'window 256 + {nsink} sink tokens: loss {loss:.2f}, '
          f'perplexity {jnp.exp(loss):.0f}')
print(f'full attention:            loss {full:.2f}, '
      f'perplexity {jnp.exp(full):.0f}')
```

In this passage, windowed attention without retained sink tokens raises
perplexity from the mid-thirties into the thousands. Retaining one initial
token removes most of this increase, and four retained tokens bring loss
within a fraction of a nat of full attention. The comparison supports a sink
effect for this checkpoint and passage, but it does not isolate all effects of
discarding older context. (Our mask-based experiment
keeps original position indices; a real rolling buffer also has to
handle positions carefully as entries are evicted, which is an
exercise.) :citet:`Xiao.Tian.Chen.ea.2024` provide broader experiments on
attention sinks, and gpt-oss provides one concrete design with an
explicit learned sink logit per head in its sliding-window layers
:cite:`OpenAI.2025`, which frees the first token from double duty.

### Comparison of Cache-Reduction Methods

For one sequence, the baseline stores $2L n H_{kv}d_h$ elements. The table
uses the same contract for each alternative; constants and auxiliary buffers
depend on the implementation.

| method | reduced dimension | stored state | information no longer available | training requirement | evidence used here |
|:--|:--|:--|:--|:--|:--|
| GQA/MQA | KV heads | $2LnH_{kv}d_h$ | independent K/V projections for every query head | train with sharing or uptrain a checkpoint | fixed-seed character model; cited large-model ablations |
| MLA | width per token | approximately $Lnr$ plus a rotary component | full K/V coordinates, reconstructed from an $r$-dimensional latent | learned compression and reconstruction maps | oracle SVD on one GPT-2 passage; cited trained MLA models |
| sliding window with sinks | cached length | $2L(w+s)H_{kv}d_h$ for window $w$ and $s$ sinks | tokens outside the window and retained sinks | train or adapt for the restricted pattern | one GPT-2 passage; cited StreamingLLM experiments |
| linear attention | token-indexed cache | fixed feature-state matrices, independent of $n$ | exact softmax attention and tokenwise recall | replace and train the attention mechanism | derivation and toy experiment in :numref:`sec_attention-at-scale` |

Hybrid stacks (:numref:`chap_modern_rnn`) combine recurrent layers with a
smaller number of full-attention layers, so only the latter retain a growing
tokenwise cache. Beyond the cache proper, the decode-side bandwidth accounting
of this section is also
what *speculative decoding* exploits, drafting several tokens with a smaller
model and verifying them together in a parallel forward pass
:cite:`Leviathan.Kalman.Matias.2023`. The roofline economics behind that
arithmetic are made precise in :numref:`sec_hardware`; the systems-level
story of speculative decoding and the serving engines around it belongs
to the Language Models part.

## Summary

Naive generation repeats a full forward pass for each token and therefore
has quadratic dense-layer cost in the generated length. Reusing past keys
and values makes this part linear, although attention still reads a growing
cache. A cached implementation requires a per-layer key--value buffer and a
position offset for RoPE; JAX uses a preallocated, fixed-shape buffer with
indexed updates. The cached and uncached implementations agree to numerical
precision, and caching gives several-fold faster decoding at long contexts.
For a batch of $b$ sequences, the cache contains $2 \cdot
n_\textrm{layers} \cdot n_\textrm{kv} \cdot d_\textrm{head} \cdot n \cdot b$
elements. Multiplying by the bytes per element gives the tensor storage; this
count excludes allocator and metadata overhead. Prefill is compute-bound, whereas
decoding is usually memory-bandwidth-bound. GQA reduces storage by sharing
keys and values across query heads; in our experiment, reducing eight KV
heads to one reduces the cache eightfold, while the fixed-seed validation
losses differ by only a few hundredths of a nat. This is not an equivalence
test. MLA instead compresses the width of the cache. Sliding
windows bound its length; for checkpoints that rely on initial attention
sinks, retaining those positions can avoid the loss increase observed in our
single-passage test. Linear-attention and
hybrid architectures provide further alternatives.

## Exercises

1. Work out the KV cache of Llama-2-70B, which uses 80 layers, 64 query
   heads of dimension 128, and GQA with 8 key-value heads, storing the
   cache in 16-bit precision. How many bytes per token? At what context
   length does a single sequence's cache reach the size of the weights
   (140 GB in fp16)? Redo both numbers for the same model without GQA
   (64 key-value heads) and for MQA (1 key-value head).
2. Break the cache deliberately: remove the `offset` from the cached
   RoPE path, so decoded tokens are rotated as if they sat at position
   zero, and rerun the correctness check. How large is the logit
   deviation, and why does it grow with the length of the prefix? What
   is the corresponding bug for the `pos='learned'` scheme?
3. Our decode measurement ran a batch of one. Extend `generate_cached`
   to a batch of prompts of equal length and measure tokens per second
   at batch sizes 1, 4, 16, and 64 at a fixed context. Explain the shape
   of the curve with the arithmetic-intensity argument: which bytes are
   read once per *step* and which once per *sequence*?
4. The PyTorch implementation grows the cache by `torch.cat`, copying
   the whole buffer every step; the JAX one preallocates and writes into
   slots, but hands the jitted function fresh buffer copies unless the
   arguments are donated. Fix either one: preallocate in PyTorch (write
   into a slice of a `max_len` buffer), or pass `donate_argnums` for the
   cache arguments in JAX. Measure the per-step latency at context 4096
   before and after your fix.
5. Give `GQAAttention` a `forward_step` in the style of this section, so
   that the cache stores only the $H_{kv}$ key-value heads. Verify
   correctness against the full forward pass, and confirm that measured
   cache memory shrinks by $H / H_{kv}$.
6. Implement a true rolling buffer: cap the per-layer cache at the 4
   sink entries plus the most recent $w - 4$, evicting the rest as
   generation proceeds past $w$ tokens. Decide what position index a
   retained entry should keep (consider both the `'rope'` and
   `'learned'` schemes), state your choice, and check the model's loss
   on a long passage against the mask-based experiment of this section.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.3]{.kicker}

Generation and the KV cache<br>
**reusing past states · cache memory · GQA, MLA, windows, and sinks**
:::
:::

::: {.slide title="Naive Autoregressive Generation Recomputes the Prefix"}
The naive `generate`: one full forward pass per token.

- Step $t$ costs $\approx 2Nt$ FLOPs; $T$ tokens cost
  $\approx NT^2$ — quadratic, and all but the last logit row discarded.
- But causality freezes the past: $\mathbf{k}_{1..t-1}, \mathbf{v}_{1..t-1}$
  are *identical* at every future step. Store them.

@fig:mdl-transformers-kv-cache
:::

::: {.slide title="The cached attention step"}
Two details: RoPE needs the **absolute position offset**, and the step
receives only the new tokens plus the cache.

@kv-cache-the-cached-forward-pass-1
:::

::: {.slide title="Prefill once, then decode one token at a time"}
@kv-cache-the-cached-forward-pass-2

::: {.d2l-note}
JAX: growing shapes would recompile every step — preallocate
`(layers, batch, max_len, heads, head_dim)` and write slots with
`dynamic_update_slice`. Static shapes are how real serving works.
:::
:::

::: {.slide title="Cached and Uncached Logits Agree"}
@!kv-cache-same-logits-measured-1

- Prefill 16, decode 32, stack the logits against one full forward pass:
  agreement to floating-point rounding, both positional schemes.
:::

::: {.slide title="Per-step latency vs. context"}
124M-parameter model, untrained (arithmetic does not care):

@!kv-cache-same-logits-measured-2
:::

::: {.slide title="Reading the curve"}
- Naive grows with context ($2Nn$); cached stays **flat**.
- About $5\times$ at a 4k context in our runs, doubling with every further
  doubling of context.
- Curves *merge* at short context: a 12-block step is launch-bound there —
  the cache removes a term that grows, not a constant factor.

@!kv-cache-same-logits-measured-3
:::

::: {.slide title="KV-cache memory"}
$$\textrm{cache bytes} = 2 \cdot n_\textrm{layers} \cdot n_\textrm{kv} \cdot d_\textrm{head} \cdot n \cdot b \cdot (\textrm{bytes/elem})$$

72 KiB per token here: at 4k context, 288 MiB — more than half the model.

@!kv-cache-the-memory-bill

::: {.d2l-note}
In PyTorch, caching a *view* of the fused QKV buffer retained twice the storage
predicted by the formula; `contiguous()` removed the excess allocation.
:::
:::

::: {.slide title="Prefill is compute-bound, decode is memory-bound"}
Every decoded token reads **all weights + the whole cache** for $2N$ FLOPs:

@!kv-cache-prefill-is-compute-bound-decode-is-memory-bound

- Same layers, same GPU: tens of thousands vs. under a hundred tokens/s.
- Decode speed depends on bytes moved per token, so reducing the cache can
  increase throughput.
:::

::: {.slide title="Sharing keys and values: MQA and GQA"}
Queries may use a different number of heads from keys and values; nothing
forces $n_\textrm{kv} = H$.

@fig:mdl-transformers-gqa

- MQA (Shazeer, 2019): one KV head — cache $\div H$.
- GQA (Ainslie et al., 2023): $H_{kv}$ heads, one per query group —
  Llama 3 and Mistral run $H/4$; uptraining made adoption cheap.
:::

::: {.slide title="A pluggable GQAAttention"}
`W_k`, `W_v` produce $H_{kv}$ heads; the fused kernel broadcasts per group.
Drops into `TransformerBlock`'s `attn_factory` and into `d2l.GPT`;
$H_{kv} = H$ recovers multi-head attention exactly.

@!kv-cache-a-pluggable-implementation-2
:::

::: {.slide title="GQA Cache Size and Validation Loss"}
Same GPT config, $H_{kv} \in \{8, 4, 2, 1\}$, best validation loss over a
fixed budget:

@!kv-cache-cache-against-quality

- Cache shrinks $8\times$; with one seed and values rounded to 0.01 nat,
  the experiment supplies no uncertainty estimate or equivalence test.
- Ainslie et al. (2023) provide separate large-model ablations of MQA and GQA.
:::

::: {.slide title="Low-rank keys and values — the MLA idea"}
Keep all heads, store a low-rank latent instead (DeepSeek-V2/V3).
Oracle check on GPT-2: truncate every layer's realized K/V to rank $r$ of
1536.

@!kv-cache-low-rank-keys-and-values

- Rank 256 changes loss by less than displayed precision on this passage;
  rank 128 increases it. This oracle, passage-specific SVD is not MLA; MLA
  learns a shared projection during training.
:::

::: {.slide title="Where trained attention actually goes"}
@!kv-cache-a-window-needs-a-sink-1

- On the displayed passage, several GPT-2 layers assign a **third to a half**
  of their attention mass to token 0. StreamingLLM (2024) calls this pattern
  an attention sink and tests its role more broadly.
:::

::: {.slide title="A window needs a sink"}
Compare a 256-token window with variants that retain initial sink positions:

@!kv-cache-a-window-needs-a-sink-2

- On this passage, the window-only mask raises perplexity from 36 to thousands.
- One retained initial token removes most of the increase; gpt-oss ships a
  learned sink logit per head.
:::

::: {.slide title="Cache-reduction methods"}
The methods address different factors in the cache formula:

- **Width**: GQA (fewer KV heads), MLA (low-rank latent).
- **Length**: sliding window + sinks.
- **Remove it**: linear attention → fixed recurrent state (§10.5);
  hybrids interleave a few full-attention layers (ch. 12).
- Decode bandwidth arithmetic also powers **speculative decoding** —
  draft cheap, verify at prefill price (→ Computational Performance).
:::

::: {.slide title="Recap"}
- Because causality leaves past states unchanged, caching their keys and
  values reduces cumulative dense-layer work from quadratic to linear in
  generated length without changing logits. Cache reads remain cumulatively
  quadratic but are subdominant here.
- Cache bytes $= 2 L\, n_\textrm{kv} d_\textrm{head}\, n\, b$ — verified
  against the allocator; it *moves* every step.
- Prefill is compute-bound, while decode speed is often limited by the
  number of cache bytes transferred per token.
- GQA: 8x smaller cache in the fixed-seed experiment; quality equivalence is
  not established.
- MLA compresses width; windows bound length while retaining designated sink
  positions.
:::
