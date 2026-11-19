# Shrinking the KV Cache
:label:`sec_kv-cache-reduction`

The cache of the previous section removed the repeated arithmetic of
generation and put a new limit in its place. Every generated token reads
the whole cache from memory, and the cache grows with every token of every
sequence in the batch. Its size is a product of six factors
(:numref:`subsec_kv-cache-memory`):

$$
\textrm{cache bytes} \;=\; 2 \cdot L \cdot H_\textrm{kv} \cdot d_h \cdot n
\cdot b \cdot (\textrm{bytes per element}),
$$

for $L$ blocks, $H_\textrm{kv}$ key--value heads of width $d_h$, and $b$
sequences of $n$ tokens. Problem :exref:`ex_kv-cache-70b` of
:numref:`sec_kv-cache` put numbers into this formula for Llama 2 70B. Built
like our GPT, with a key--value head for each of its 64 query heads, that
model would cache 2.5 MiB per token, and thirteen conversations of 4,096
tokens would fill as much memory as its 140 GB of weights. The released
model caches eight times less.

How can a cache shrink without breaking the model? The formula is a product,
and each factor is a place to cut. Three of them are worth learning, and
they are not equally important. The one that matters most is the number of
key--value heads $H_\textrm{kv}$, which shrinks when several query heads
share one. Most open models released since 2023 do this, and our
`d2l.GPT` does it with one constructor argument; whether it costs anything
is a question we can test. The width that each token stores in each block,
$2 H_\textrm{kv} d_h$ numbers in all, shrinks further when a token stores a
short summary from which its keys and values are rebuilt. DeepSeek's models
learn such a summary during training. The length $n$ shrinks when the cache
keeps only a window of recent tokens, which matters for text that never
ends; a trained model then needs a few of its first tokens kept as well,
for a reason that is far from obvious. Two more ways to shrink the cache
get no more than a pointer here: storing each number in fewer bits, and
replacing the cache by a state of fixed size.

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

We met shared keys and values once before, in the copy-both layer of
:numref:`subsec_kv-head-sharing`, a task built so that a single head must
fail. The heads of that layer had to differ in how they weight the
positions (:numref:`subsec_one-mixture-bottleneck`), but the difference
could come from their query projections alone. Their keys could be shared,
and so could their values, as long as the shared value heads together had
room for all the coordinates of a value. Let us now take the idea from a toy
task to language models.

In multi-head attention every query head has a key head and a value head of
its own, so $H_\textrm{kv}$ equals the number $H$ of query heads.
*Multi-query attention* (MQA) goes to the other extreme: all $H$ query heads
read a single key--value head, and the cache shrinks by the factor $H$
:cite:`Shazeer.2019`. *Grouped-query attention* (GQA) lies in between. It
keeps $H_\textrm{kv}$ key--value heads, each read by a group of $H /
H_\textrm{kv}$ query heads, and shrinks the cache by the size of the group
:cite:`Ainslie.Lee-Thorp.Jong.ea.2023`. :numref:`fig_gqa` shows the three
patterns. A language model offers no clean count like that of the copy-both
layer, so how far sharing can go is a question for experiment. In current
open models the common choice is $H_\textrm{kv}$ between $H/8$ and $H/4$;
Llama 3 8B and Mistral 7B both run 32 query heads against 8 key--value heads
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
$H_\textrm{kv} = H$ the layer is ordinary multi-head attention. The cached
forward pass of :numref:`subsec_kv-cache-cached-forward` needs no change,
since `init_cache` sizes its buffers by `num_kv_heads`.

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

Eight query heads of width 32 face two key heads and two value heads. What
does the sharing save? $\mathbf{W}_q$ and $\mathbf{W}_o$ stay $d \times d$,
while $\mathbf{W}_k$ and $\mathbf{W}_v$ shrink to $d \times H_\textrm{kv}
d_h$. For $d = 256$ and $H = 8$ we therefore expect $4d^2 = 262{,}144$
attention parameters with eight key--value heads and $2.25\, d^2 =
147{,}456$ with one. The cache per token and block holds $2 H_\textrm{kv}
d_h$ numbers, 2,048 bytes in fp32 with eight heads, and shrinks with
$H_\textrm{kv}$. We count both in a `d2l.GPT` of a single block.

```{.python .input #kv-cache-a-pluggable-implementation-2}
%%tab pytorch, jax
for H_kv in (8, 4, 2, 1):
    lm = gpt(256, num_hiddens=256, num_heads=8, num_blks=1, num_kv_heads=H_kv)
    print(f'H_kv={H_kv}: {d2l.num_params(lm.blks[0].attention):6d} '
          f'parameters, cache {d2l.kv_cache_bytes(lm, 1):5d} bytes per token '
          'and block')
```

The counts match the prediction. Sharing removes at most half of the
attention parameters, a modest saving in a model's size, but it divides the
cache by the full size of the group, and the cache is what grows with the
context.

### Shared Heads and Validation Loss
:label:`subsec_kv-cache-gqa-quality`

What does sharing cost in quality? There are two ways to arrive at a model
with shared heads, and the question has a different answer for each. We can
train with $H_\textrm{kv} < H$ from the first step. Or we can take a model
already trained with multi-head attention and convert it, which is what
Ainslie et al. proposed for checkpoints that cost too much to train again
:cite:`Ainslie.Lee-Thorp.Jong.ea.2023`. Their conversion replaces the key
projections of each group of $H / H_\textrm{kv}$ consecutive heads by
their average, and does the same for the value projections. For group $g$,

$$
\mathbf{W}_k^{(g)} = \frac{H_\textrm{kv}}{H} \sum_{h \in g}
\mathbf{W}_k^{(h)}, \qquad
\mathbf{W}_v^{(g)} = \frac{H_\textrm{kv}}{H} \sum_{h \in g}
\mathbf{W}_v^{(h)},
$$

while the query and output projections, the feed-forward networks, and the
normalizations are copied unchanged. The averaged key is not the key that
any query head was trained to match, so the converted model must then be
trained a little longer. This short continuation is called *uptraining*.
The function `pool_heads` performs the conversion, by averaging or, as a
control, by keeping only the first head of each group.

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

We run the experiment on the byte-level model of :numref:`sec_gpt`: the
default `d2l.GPT` with six blocks of width 256 and eight heads, about 4.8
million parameters, trained for 1,000 steps on the bytes of WikiText-103
with the recipe of that section. Repeated runs of it end within about 0.01
nats of each other (:numref:`subsec_gpt-run-to-run`), and this is the
yardstick for every difference below. The cell compares three ways to
arrive at two key--value heads, each shared by four query heads:

1. Train with two key--value heads from the start, next to the usual eight.
1. Convert the trained eight-head model, by averaging and by keeping the
   first head, and uptrain each conversion at a small constant learning
   rate for 50 steps and then 50 more, 5% and 10% of the original training.
1. As a control, train the two-head model from scratch for only 50 steps.
   It shows how much of the converted model's quality those 50 steps alone
   could have produced.

Everything runs twice.

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

Trained with two key--value heads from the start, the model ends at 1.34
to 1.35 nats per byte in every run, where the model with eight heads ends.
A cache four times smaller costs nothing that this model can resolve.

Conversion is another matter. Right after it, both converted models reach
a loss of about 2.7 to 2.9, worse than the bigram level of 2.45 that a
model reading only the previous byte reaches (:numref:`subsec_gpt-training`).
How can a model this broken have kept anything worth keeping? The
eight-head model's knowledge is still in its embeddings, its feed-forward
networks, and its query and output projections. What the conversion broke
is the match between each query head and the keys it was trained against,
and that match takes few steps to restore. Fifty steps of uptraining bring
the averaged model to about 1.6. That closes about four fifths of the gap
between the converted model and one trained with shared heads, while fifty
steps of training from scratch reach only 2.6 to 2.8. After 100 steps the
averaged model stands at 1.45 to 1.48, still 0.1 nat or more above the
model trained with shared heads, ten times the yardstick. After either
length of uptraining it is ahead of the first-head conversion by more than
a tenth of a nat in every run. Averaging keeps a part of every head's
projection, while keeping the first head throws three of the four away;
Ainslie et al. found the same order on T5.

Our model is small, and at a larger scale the cost of sharing is not
always zero. Ainslie et al. uptrained T5-XXL for 5% of its pretraining
steps. With eight key--value groups its average score on their tasks fell
by a tenth of a point, from 47.2 to 47.1, while inference became five
times faster per sample; with a single key--value head the score fell to
46.6.

Two ablations of models trained with sharing from the start disagree with
each other. In Llama 2's, at 30 billion parameters, eight key--value heads
did about as well as full multi-head attention, and a single head a little
worse :cite:`touvron2023llama2`. In DeepSeek-V2's, with 7B models trained
on 1.33 trillion tokens, accuracy on the multiple-choice benchmark MMLU
fell from 45% with full heads to 41% with eight groups and to 38% with one
:cite:`DeepSeek-AI.2024`. The two studies differ in size, data, and
benchmark, so the cost of sharing depends on the model and on what it is
asked. No single experiment settles it.

Practice has settled the question anyway. The models that adopted GQA,
among them Llama 2 70B, Mistral 7B, and Llama 3, train with it from the
start; uptraining was proposed to convert existing checkpoints, not as the
usual route to a grouped model. A cache eight times smaller holds
eight times as much context or eight times as many sequences, and every
decoded token reads eight times fewer cache bytes. Against this, a loss that
is zero in some studies and a few points in others is one that most model
builders accept. Sharing heads is the cheapest factor of four to eight in a
transformer's memory, and it is the first one to reach for. DeepSeek, which
measured the larger loss, went looking for another factor to cut.

## Narrower and Shorter Caches

Two factors remain: the width of what each token stores, and the number of
tokens stored. How far each can shrink depends on how a trained model uses
its keys and values, so we need a model trained on far more text than our
byte-level one: the released GPT-2. `d2l.load_gpt2` builds it as a
`d2l.GPT` with the weights loaded as in :numref:`subsec_gpt-loading`, and
`d2l.gpt2_tokenizer` returns its tokenizer. We encode *The Time Machine*
and keep its first 1,024 tokens as the evaluation passage `x`. The next
1,024 tokens form a second passage `calib`, on which we will fit
compressions without looking at `x`.

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

The helper reproduces the model without a single rounding difference, and
full attention predicts the second half of the passage with a loss of 3.58
nats per token, a perplexity of 36.

### Low-Rank Keys and Values
:label:`subsec_kv-cache-low-rank`

GQA removes whole heads. A different idea keeps every head and asks whether
a token stores as many numbers as it seems to. Each block of GPT-2 caches
768 key and 768 value numbers per token, 1,536 in all. But it computes them
from one vector, the normalized input $\mathbf{x}$ of the block, which has
768 numbers, with one matrix and one bias:

$$
\begin{bmatrix} \mathbf{k} \\ \mathbf{v} \end{bmatrix}
= \mathbf{W} \mathbf{x} + \mathbf{b},
\qquad \mathbf{W} \in \mathbb{R}^{1536 \times 768}.
$$

Whatever the weights, the keys and values of all tokens therefore lie in a
768-dimensional subspace, shifted by $\mathbf{b}$. Half of the cache is
redundant before any training has happened. A cache could store
$\mathbf{x}$ and reproduce every key and value exactly from half the bytes,
although every decode step would then multiply every cached $\mathbf{x}$
by $\mathbf{W}$ again. The interesting question is how far below 768
we can go, and that depends on what training has done to the keys and
values.

GPT-2 lets us answer it without training anything. In every block we stack
the keys and values of each token of `calib` as one row of a matrix and
compute its top $r$ right singular vectors. By the Eckart--Young theorem
they span the $r$-dimensional subspace closest to these rows
(:numref:`sec_mdl-svd-low-rank`). We then run the model on `x` with every
key and value replaced by its projection onto this subspace, which a cache
could store as $r$ coordinates per token. The basis is fixed before `x`
arrives, as any projection built into a model must be, so no token of `x`
influences how the keys and values of `x` are compressed. We try $r =
512$, 256, and 128, two thirds, one third, and one sixth of the 768
dimensions available.

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

At ranks 512 and 256 the loss is 3.55 and 3.61, against 3.58 for full
attention. On 511 predictions of one passage, a difference of 0.03 in
either direction is no difference we can claim. Only at rank 128 does the
compression hurt: the loss rises by more than half a nat, and the
perplexity from 36 to 63. Two facts come out of this. Projected onto a
third of the 768 dimensions they could fill, GPT-2's keys and values
predict as well as before, as far as one passage can tell. And the subspace that does it was fitted on
another passage, so it carries over from one part of the book to the next.
The second fact is the one a cache needs: a basis that had to be refitted
for every text could not be built into a model.

*Multi-head latent attention* (MLA), around which DeepSeek built its V2 and
V3 models, puts such a projection into the architecture
:cite:`DeepSeek-AI.2024`. It learns the projection during training instead
of fitting it afterward, so the rest of the model can adapt to the
compression. A trained down-projection compresses the keys and values of
each token jointly into one latent vector, the cache stores only the
latent, and trained up-projections map the latent to the keys and values of
every head. In DeepSeek-V2 the latent has 512 numbers. Against the 32,768
key and value numbers of its 128 heads, that looks like a factor of 64. But
the keys and values are computed from a hidden state of 5,120 numbers, and
by the argument above that is the fair yardstick: the latent compresses by
a factor of ten.

Two details complete the design. The up-projections need not be applied to
the cache at all. They can be folded into the queries and into the output
projection, so that a decode step works on the cached latents directly, as
Exercise :exref:`ex_kv-cache-absorbed` shows. RoPE spoils this for the
keys: its rotation depends on the position of each key and sits between the
query and the up-projection, so it cannot be folded in. MLA therefore
caches a small separate rotary key per token.

### Sliding Windows and Attention Sinks
:label:`subsec_kv-cache-sinks`

The remaining factor is length. A cache that keeps only the most recent $w$
tokens, and evicts the oldest as generation proceeds, needs the same memory
after a million tokens as after a thousand. This is the rolling-buffer form
of the sliding-window attention of :numref:`sec_windowed-linear-attention`,
and it is what a model needs for a stream of text that never ends. It looks
harmless for a trained model, whose next token rarely depends on text a
thousand tokens back. Before evicting anything, though, we should check
where a trained model sends its attention. We measure the average weight
that GPT-2's queries place on the first token of the passage. The queries
before position 64 are left out, since they see so few tokens that any of
them receives a large share.

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

The first two blocks hardly look at the first token, and block 2 gives it
about a tenth of its attention. From block 3 on, the queries place between
about a third and more than half of their attention on the first token, a
hundred times its uniform share or more. This is the
*attention sink* of :numref:`subsec_cross-attention-alignment`. The weights
of every query sum to one, so a head with nothing to retrieve must still put
its weight somewhere, and the first token, which every query can see,
becomes that place :cite:`Xiao.Tian.Chen.ea.2024`.

A rolling buffer of the most recent $w$ tokens evicts this token as soon as
the text is longer than the window, and the softmax then spreads its weight
over tokens that the heads had chosen to ignore. StreamingLLM therefore
keeps the first few tokens in the cache next to the window
:cite:`Xiao.Tian.Chen.ea.2024`. We reproduce both the failure and the fix
by evaluating the passage with attention restricted to a window of 256
tokens plus a number of retained *sink tokens* at the start, with no
training at all.

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

The window alone is a disaster: the perplexity rises from 36 to more than
3,800. Keeping a single token brings it back to 47. That token is the first
word of the book, *The*, and it says nothing about sentences hundreds of
tokens later. Four sink tokens bring the loss within about 0.2 nat of full
attention. The 256 most recent tokens were nearly enough all
along. What the window took away was the place where the heads put the
weight they did not need. One token out of 1,024 is worth a factor of 80 in
perplexity, not for what it says but for where it sits. The remaining gap
of 0.2 nat is the cost of the window: older tokens are gone, and the model
was not trained on this pattern.

Our mask keeps the original position of every token. A real rolling buffer
cannot do so forever: GPT-2's learned positions end at 1,023, and a stream
does not. Exercise :exref:`ex_kv-cache-streaming` builds the buffer and
decides which positions its entries carry. Recent models remove the need
for a sink token by giving each head a learned logit that can absorb
unwanted weight :cite:`OpenAI.2025`, which Exercise
:exref:`ex_kv-cache-sink-logit` analyses.

## Comparison of Cache-Reduction Methods

For one sequence of $n$ tokens the full cache stores $2 L n H_\textrm{kv}
d_h$ numbers. :numref:`tab_kv-cache-reduction` lists which factor each
method of this section reduces and whether it needs training. Constants
and auxiliary buffers depend on the implementation.

:Three ways to shrink the KV cache of one sequence of $n$ tokens, with $L$ blocks, $H_\textrm{kv}$ key--value heads of width $d_h$, a latent of $r$ numbers, a window of $w$ tokens, and $s$ sink tokens.
:label:`tab_kv-cache-reduction`

| method | factor reduced | stored numbers | needs training? |
|:--|:--|:--|:--|
| shared key--value heads (GQA, MQA) | heads $H_\textrm{kv}$ | $2LnH_\textrm{kv}d_h$ with $H_\textrm{kv} < H$ | yes: train with sharing, or uptrain a checkpoint |
| low-rank latent (MLA) | width $2H_\textrm{kv}d_h$ | about $Lnr$, plus a small rotary key | yes for MLA; a basis fitted after training works up to a point |
| sliding window with sinks | length $n$ | $2L(w+s)H_\textrm{kv}d_h$ | no; better if pretrained with a sink token |

A window combines with either of the other two methods; Mistral 7B, for
instance, reads grouped heads within a sliding window
:cite:`Jiang.Sablayrolles.Mensch.ea.2023`. Shared heads and a latent are
alternatives rather than partners, since one latent already serves all
heads. A fourth factor, the bytes per element, combines with all three.
Stored in two to four bits instead of sixteen, the cache keeps generation
quality close to that of the full cache if keys are rounded in groups per
channel, one coordinate across tokens, and values in groups per token
:cite:`Liu.Yuan.Jin.ea.2024`. Exercise :exref:`ex_kv-cache-quantization`
finds out why this grouping works.

For a model trained today, sharing heads is the default and the only
question is how many. A latent has to be built in before training. A
window with sinks can be added to a trained model, as GPT-2 showed, though
it works better if the model was pretrained with a sink token. The window
bounds the cache by forgetting every token more than $w$ positions back,
except the sinks. A memory of fixed size that still summarizes the whole text needs a
state, such as that of the linear attention of
:numref:`sec_windowed-linear-attention`. Hybrid stacks, in which only a few
attention layers keep a cache, shrink the factor $L$ instead
(:numref:`chap_modern_rnn`).

## Summary

The KV cache is a product of factors, and three of them can shrink: the
number of key--value heads, the width stored per token, and the number of
tokens kept. Sharing key--value heads among groups of query heads divides
the cache by the size of the group; trained that way from the start, it
cost our byte-level model nothing measurable, and with brief uptraining a
converted checkpoint recovers most of what the conversion lost. GPT-2's
keys and values are an affine function of 768 numbers, and a basis of a
third of that dimension, fitted on another passage, leaves the loss
unchanged within what one passage can resolve; MLA learns such a
compression during training. A sliding window bounds the cache for any
length of text, provided it keeps the attention sink on the first token,
without which GPT-2's perplexity rises a hundredfold.

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
       512 tokens of `x`, keep the prompt's own keys and values exact, and
       apply the basis to every later token. Implement this variant and
       compare it with the other two. Which of the three could a deployed
       cache use?

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
   :exlabel:`ex_kv-cache-quantization`

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
   :exlabel:`ex_kv-cache-sink-logit`

1. [code] **Longer uptraining and other group sizes.** ● The conversion
   of :numref:`subsec_kv-cache-gqa-quality` stops after 100 steps, while
   the converted model is still above one trained with shared heads from
   the start.
    1. Continue uptraining the mean-pooled model for 500 steps and measure
       the validation loss every 50 steps. After how many steps does it come
       within the run-to-run spread of :numref:`subsec_gpt-run-to-run` of
       the $H_\textrm{kv} = 2$ model trained from scratch? Repeat with a
       learning rate that decays to zero over the 500 steps in place of
       the constant rate.
    1. Convert the trained $H_\textrm{kv} = 8$ model with `pool_heads` to
       $H_\textrm{kv} = 4$ and to $H_\textrm{kv} = 1$ as well. How do the
       loss before uptraining and the loss after 50 steps depend on the
       size of the groups?

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
   :exlabel:`ex_kv-cache-absorbed`

1. [extended] **A streaming cache.** ●● The mask-based experiment of
   :numref:`subsec_kv-cache-sinks` restricts attention without evicting any
   entry, and it keeps the original position of every token.
    1. Copy `forward_cached` from :numref:`sec_kv-cache` and change it so
       that the cache of each block is a rolling buffer. It holds the first
       $s$ entries and the most recent $w$ entries, the current token
       included, and overwrites the oldest of the remaining entries.
       Prefill GPT-2 with the first 256 tokens of the passage `x`, feed
       the remaining tokens one at a time with
       $w = 256$ and $s = 4$, and confirm that the loss on positions 512
       onward matches the mask-based result.
    1. GPT-2 cannot decode beyond position 1023, where its position table
       ends. A RoPE model can stream indefinitely if the positions of the
       cached entries are reassigned. Decide what the buffer must store for
       each entry and which positions the rotation should use, so that
       every query--key offset stays below $s + w$. Explain why learned
       position embeddings do not admit the same reassignment.
   :exlabel:`ex_kv-cache-streaming`

[Discussions](https://d2l.discourse.group/t/36883)

<!-- slides -->

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
| A GPT from Scratch | a trained LM; GPT-2 loaded |
| The KV Cache | reuse keys and values |
| **Shrinking the KV Cache** | **fewer heads, dimensions, tokens** |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as tokens |
| Scaling Laws | loss against size and data |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | settings of deployed models |
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
$H/H_\textrm{kv}$ consecutive heads. For group $g$:

$$\mathbf{W}_k^{(g)} = \frac{H_\textrm{kv}}{H} \sum_{h \in g} \mathbf{W}_k^{(h)},
\qquad \mathbf{W}_v^{(g)} = \frac{H_\textrm{kv}}{H} \sum_{h \in g} \mathbf{W}_v^{(h)}$$

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

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Sharing heads: 8 → 2 key--value heads cut the byte model's cache
  fourfold at no measurable loss when trained that way; a converted
  checkpoint needs uptraining, and averaging beats keeping one head.
- GPT-2's keys and values are a function of 768 numbers; a basis of 256
  from another passage costs nothing measurable, one of 128 half a nat.
- A window bounds the cache only if it keeps the attention sink.

**Next:** encoders, decoders, and cross-attention.
:::
