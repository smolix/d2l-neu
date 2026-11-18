# Shrinking the KV Cache
:label:`sec_kv-cache-reduction`

<!-- WRITER: new opening (structure.md section 3.4): from the reader, the cache
formula of :eqref:`eq_kv-cache-bytes` restated in one display, the 70B number,
and the ranking (sharing heads, a learned latent, a window with sinks;
quantization and linear attention in one sentence each). The cells below come
from the second half of the old kv-cache.md with their ids kept. -->

```{.python .input #kv-cache-reduction-shrinking-the-kv-cache-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #kv-cache-reduction-shrinking-the-kv-cache-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #kv-cache-reduction-shrinking-the-kv-cache-2}
%%tab pytorch
gpt, device = d2l.GPT, d2l.try_gpu()
```

```{.python .input #kv-cache-reduction-shrinking-the-kv-cache-2}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

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
$H / H_\textrm{kv}$. We count both in a `d2l.GPT` of a single block.

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab pytorch, jax
for G in (8, 4, 2, 1):
    lm = gpt(256, num_hiddens=256, num_heads=8, num_blks=1, num_kv_heads=G)
    print(f'H_kv={G}: {d2l.num_params(lm.blks[0].attention):6d} '
          f'parameters, cache {d2l.kv_cache_bytes(lm, 1):5d} bytes per token '
          'and block')
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
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
for G in (8, 4, 2, 1):
    lm = gpt(len(data.vocab), num_kv_heads=G)
    optimizer = d2l.LMOptimizer(lm, lr=2e-3, num_steps=1000, warmup=100)
    d2l.train_steps(lm, data, optimizer, 1000)
    loss = math.log(d2l.lm_perplexity(lm, data.val_dataloader()))
    print(f'H_kv={G}: validation loss {loss:.2f}, cache '
          f'{d2l.kv_cache_bytes(lm, 1) / 1024:4.1f} KiB per token')
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
        print(f'rank {r} of 1536: loss {loss:.2f}, '
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
sliding-window attention of :numref:`sec_windowed-linear-attention`. Before
evicting anything, we measure where a trained model sends its attention:
the average weight that GPT-2's queries, from position 64 on, place on the
first token of the passage.

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
| linear attention | the token-indexed cache | fixed state matrices, independent of $n$ | exact softmax attention and recall of individual tokens | replace and train the attention mechanism | derivation and experiment in :numref:`sec_windowed-linear-attention` |

Quantization combines with every other row, because it changes only the
number of bits that store each number. At two to four bits, quantizing
keys per channel and values per token keeps generation quality close to
that of the full cache :cite:`Liu.Yuan.Jin.ea.2024`. An exercise tests why
this grouping works. Other rows combine too. Hybrid stacks
(:numref:`chap_modern_rnn`) pair recurrent layers that keep a fixed state,
such as linear attention, with a few full-attention layers, so that only
those few layers keep a growing cache.

## Summary

<!-- WRITER: four sentences; the three below come from the old Summary of
kv-cache.md. -->
GQA
and MQA shrink the cache by sharing key--value heads at little cost in
quality, and low-rank latents such as MLA shrink the width stored per
token. A sliding window bounds the length, provided it keeps the attention
sinks that trained models rely on.

## Exercises

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

<!-- Discussions: a new Discourse topic is needed for this page (tools/create_forum_topics.py --create). -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.4]{.kicker}

Shrinking the KV Cache<br>
**shared key--value heads · low-rank keys and values · windows and sinks**
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
