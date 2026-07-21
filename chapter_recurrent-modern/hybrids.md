# Hybrid Architectures
:label:`sec_hybrids`

Every memory in this chapter has been a fixed block of numbers. We gated
it, linearized it, made it selective, taught it to edit itself, and
finally recognized it as a regressor fitting itself at test time — and
none of that changed its size. :numref:`sec_test-time-regression` ended
on the question the whole chapter has been postponing: when the state is
not enough, how much genuine attention must a model keep? This section
answers with measurements. It is also the delivery of a promise made
from the other side of the divide: the cache-relief map of
:numref:`sec_kv-cache` ended with a rung it could only gesture at —
interleave a few full-attention layers into a mostly recurrent stack, so
that most layers pay no cache at all while a few retain exact recall —
and :numref:`sec_scaling-laws` called the bet on that rung "that
chapter's story". This is that story.

The plan follows the trade itself. First the *limit*: what a fixed state
cannot do, with the counting theorem — conditions included — and the
production symptoms that made hybrids necessary. Then the *price*: only
attention layers pay a
growing cache, an asymmetry worth a factor of eight in serving memory at
long context. Then the *experiment*, the section's centerpiece: three
matched models — pure linear recurrence, pure attention, and a hybrid
differing by a single layer — swept until the fixed state saturates,
with a language-modeling panel that shows why perplexity did not see the
problem. Finally the *engineering*: the measured design rules from
shipped systems, a recipe table from Jamba to Kimi Linear, and the
distillation shortcut that turns a pretrained transformer into a hybrid
for a fraction of the training bill.

*Prerequisites: the KV-cache accounting of :numref:`sec_kv-cache`
(:eqref:`eq_kv-cache-bytes`); the matrix state, its capacity
proposition, and the duality of :numref:`sec_matrix-state`; and the
recall task of :numref:`subsec_dn-trained`.*

```{.python .input #hybrids-hybrid-architectures}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

```{.python .input #hybrids-hybrid-architectures}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import numpy as np
import optax
```

## What a Fixed State Cannot Do
:label:`subsec_hy-limits`

Selectivity fixed content-blindness. The delta rule fixed the additive
write. Test-time learning fixed staleness. None of them fixed capacity,
and none can: a state of $N$ numbers at any fixed numerical precision
holds a bounded number of bits, and
information theory does not care how cleverly the update rule was
chosen. To reproduce $k$ arbitrary tokens from a vocabulary of size
$V$, *something* in the model must hold $k \log_2 V$ bits from the
moment they appear to the moment they are needed. Our own measurements
have circled this wall all chapter. The crowding proposition of
:numref:`subsec_ms-capacity` is the wall seen from below: expected
squared read
error $(n-1)/d_k$ after $n$ random-key writes, interference growing
linearly in
how much you store. And the transition ladder of :numref:`tab_dn-ladder`
is the wall seen from the side: each rung changed what the state can
*compute* — forgetting, erasure, reflections, group words — while the
number of bits it holds never moved.

:citet:`Jelassi.Brandfonbrener.Kakade.ea.2024` make the limit a theorem,
using the purest recall task there is: copying. Both directions of their
comparison carry conditions, so we state the two results together, with
matching quantifiers.

> **The copy bounds** :cite:`Jelassi.Brandfonbrener.Kakade.ea.2024`.
> *Transformer, upper bound:* there is a depth-two transformer of width
> $O(n \log D)$ that copies any string over an alphabet of size $D$ up
> to length $D^n$, *provided no length-$n$ substring repeats* — the
> construction addresses the archive by hashed $n$-grams, and the
> attention window over everything so far supplies the storage, growing
> with the sequence. *Generalized SSM, lower bound:* any model that
> pushes the sequence through a fixed, finite state set $\mathcal{S}$ —
> every architecture in this chapter, at any fixed precision — errs on
> copying a string drawn *uniformly at random* from the $D^L$ strings of
> length $L$ with probability at least $1 - |\mathcal{S}|/D^L$; once the
> state holds fewer than about $L \log_2 D$ bits, it gets most such
> copies wrong.

Neither half reduces to "attention remembers, recurrence cannot". The
upper bound is a construction, valid for strings without repeated
$n$-grams; the lower bound is a counting argument over the uniform copy
distribution, and it says nothing about a narrower workload — a state
that stores just the strings its training distribution favors evades
it. What the count does rule out is a fixed state that copies
*arbitrary* strings much longer than its bit budget: there are $D^L$
required outputs and only $|\mathcal{S}|$ states to route them through,
and no parameterization, gating, or training trick appears anywhere in
that ratio. Empirically the crossover is not close: on synthetic
phone-book
lookup, the smallest transformer they test, at 410M parameters, beats a
2.8B-parameter Mamba once the book exceeds roughly seventy entries.

Where does the wall bite in real language?
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` located it with *multi-query
associative recall* (MQAR): a stream writes key–value bindings, later
re-presents keys in arbitrary order, and the model must produce each
key's value — our overwrite task of :numref:`subsec_dn-trained`,
embedded in text. Across seventeen trained language models, 82% of the
perplexity gap between efficient architectures and attention fell on
the recall slice — the tokens whose prediction requires retrieving a
binding seen once before. Attention solves MQAR at a width independent
of sequence length; recurrent models solve it only while the bindings
in play fit in their state — the crowding picture again, now
diagnosed in the wild.

At production scale the failure has a name. In NVIDIA's controlled
8B-parameter comparison, a pure Mamba-2 model matched a transformer on
perplexity and most benchmarks while scoring 29% on five-shot MMLU
against the transformer's 46%, and its phone-book recall collapsed once
the book grew past a few hundred tokens; the authors call the mode
*fuzzy memory* — the model returns an answer that shares digits with
the right one :cite:`Waleffe.Byeon.Riach.ea.2024`. The practitioner's
version of the diagnosis is blunter: ablate the few full-attention
layers of a shipped hybrid and needle-in-a-haystack retrieval drops to
roughly zero. A handful of full layers carries retrieval; the linear
majority carries cheap throughput.

:citet:`Gu.2025` offers a framing for this trade worth carrying to the
end of the section. A recurrent state is a *brain*: a compressed,
always-on working memory at constant cost per step, which must decide at
write time what will matter later, and therefore forgets. The
transformer's growing key–value cache is a *database*: a lossless log of
everything, pay-per-query at ever-growing cost, which never has to
predict what will matter because it keeps it all. Neither dominates,
because they fail on different bills — the brain loses exact recall, the
database loses the economics. The obvious synthesis, a brain in front of
a database, is where production actually landed, and the rest of this
section is about its terms.

## The Economics
:label:`subsec_hy-economics`

If the fixed state loses the recall fight, why not concede and serve
pure transformers? Because of the bill that :numref:`sec_kv-cache`
measured: cache bytes $= 2 \cdot n_\textrm{layers} \cdot n_\textrm{kv}
\cdot d_\textrm{head} \cdot n \cdot b$ times the element size,
:eqref:`eq_kv-cache-bytes`, growing linearly in the context length
$n$. The factor that matters here is $n_\textrm{layers}$: *the cache is
paid per attention layer*. A recurrent layer holds a state whose shape
:eqref:`eq_ms-recurrence` fixes at $d_k \times d_v$ numbers per head
and whose bytes the per-layer formula :eqref:`eq_ms-state-bytes`
prices. Priced concretely, a Mamba-2 layer at $d_\textrm{model} = 4096$
with the usual expansion factor of two (so $d_\textrm{inner} = 8192$)
and state width $d_\textrm{state} = 128$ carries $8192 \cdot 128
\approx 1.05\textrm{M}$ state elements — about 2 MB at the fp16
serving precision that formula's headline uses, with the small
convolution buffer adding another two percent
— and that figure is the same at token one hundred and at token one
million. A full-attention layer with grouped queries at the same width
pays 4 KB *per token*: past a few hundred tokens the growing archive
dwarfs the flat state, and by 128K context one attention layer holds
512 MB against the recurrent layer's 2 MB.
:numref:`fig_hy-cache` draws the consequence for a whole model: replace
most attention layers with recurrent ones and the memory bill drops by
almost exactly the fraction of attention you removed, because the
surviving attention layers are the only ones still paying rent.

![Persistent decode state against context length for a 32-layer model at production width (GQA with $8$ key–value heads of dimension 128, Mamba-2-sized states, 16-bit). The count covers only the per-user state a server holds between decode steps — KV cache plus recurrent state — excluding parameters, activations, and workspace. A pure transformer's cache reaches 32 GB at 256K context; a pure recurrent stack stays near 64 MB at every length; a 4-of-32 hybrid pays the attention fraction of the transformer bill.](../img/mdl-modernrnn-hybrid-cache.svg)
:label:`fig_hy-cache`

Those curves are not hypothetical: they are the memory column of
Jamba's model card. At 256K context, Jamba — 4 attention layers out of
32, the rest Mamba — reports a 4 GB key–value cache where the
comparably sized Mixtral carries 32 GB and a Llama-2-70B-class
transformer 128 GB :cite:`Lieber.Lenz.Bata.ea.2024`. An eightfold saving
on the resource that decides how many concurrent users fit on an
accelerator (recall from :numref:`sec_kv-cache` that decode is
bandwidth-bound, so cache bytes are also the currency of generation
speed) is the kind of number that redraws architectures, and it comes
from nothing more than the layer count in :eqref:`eq_kv-cache-bytes`.

One structural remark before we measure the other side of the trade.
The cache relief of :numref:`sec_kv-cache` came in flavors that
*compose*: GQA and MLA shrink the width of each cached token, sliding
windows bound its lifetime, quantization shrinks its bytes, and a
deployment can apply all three to the same cache. Replacing attention
with recurrence removes the growing term rather than shrinking it. The
recurrent state is a tensor like any other — it can be quantized or held
in lower precision too — but it is already constant in context length,
so there is no growth left to attack; the contrast that matters is not
compressible-versus-not but *what grows with context and what does not*.
A hybrid gets both moves: the recurrent majority contributes only
constant state, and the attention minority keeps a cache that GQA, MLA,
and quantization still shrink (the recipe table below shows shipped
hybrids doing exactly this). That is the last rung of the cache-relief
map, delivered. What remains is the quantitative question the map could
not answer: how few attention layers can a model keep and still recall
like a transformer?

## The Experiment: One Attention Layer Rescues Recall
:label:`subsec_hy-experiment`

We answer at teaching scale, with machinery this book has already
built. Three models, identical in every dimension — depth, width,
heads, embeddings, MLPs, parameter count to within a percent (the model
cell below prints the counts) — except
for the token mixer inside each block. The task is MQAR, sized so that
the sweep crosses the fixed state's capacity. The claim to test is the
one the last two sections set up: recall should collapse for the pure
recurrent stack when the bindings exceed what its state can hold, a
single mid-stack attention layer should restore it, and language-model
loss should barely notice any of it.

### Three Matched Models

The task generator is the one from :numref:`subsec_dn-trained` with the
re-binding knob removed and the *load* knob exposed: each sequence
writes `num_pairs` distinct key–value bindings, then queries every key
in random order. Sweeping `num_pairs` sweeps the number of bits the
model must carry from write phase to query phase.

```{.python .input #hybrids-three-matched-models-1}
%%tab pytorch, jax
def make_recall(num_seqs, num_pairs, num_keys=64, num_values=32, seed=0):
    """Write num_pairs distinct bindings, then query each key once."""
    task_rng = np.random.default_rng(seed)
    P, T = num_pairs, 2 * num_pairs
    QUERY = num_values                    # A reserved "query" content token
    keys = np.zeros((num_seqs, T), np.int64)
    vals = np.full((num_seqs, T), QUERY, np.int64)
    tgt = np.full((num_seqs, T), -1, np.int64)
    for b in range(num_seqs):
        ks = task_rng.choice(num_keys, size=P, replace=False)
        vs = task_rng.integers(0, num_values, size=P)
        keys[b, :P], vals[b, :P] = ks, vs
        qorder = task_rng.permutation(P)
        keys[b, P:] = ks[qorder]
        tgt[b, P:] = vs[qorder]
    return keys, vals, tgt                # tgt = -1 outside the query phase
```

The recurrent mixer is *scalar-gated* linear attention: an
input-dependent decay $a_t$, one number per head per token, applied to a
matrix state per head. On the decay ladder of
:numref:`fig_ms-decay-ladder` this is Mamba-2's rung
:cite:`Dao.Gu.2024`, one below the per-coordinate diagonal gate that
defines GLA proper :cite:`Yang.Wang.Shen.ea.2024`; we take the scalar
rung because it is the cheapest transition that forgets, and name the
class for what it is: `ScalarGatedMixer`. We train it through the
quadratic dual $\mathbf{Y} = (\mathbf{L} \circ
\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ of :eqref:`eq_ms-semiseparable`,
which :numref:`subsec_ms-duality` verified equal to the recurrence to
float32 rounding — so this is a genuine linear-recurrence model,
computed the fast way.

Three lines of the implementation are load-bearing, and each encodes a
lesson that cost real debugging time, so we state them as prose rather
than leaving them as comments. First, **a fresh gate must retain**.
With a default-initialized gate the decay comes out around
$a \approx 0.5$ per token — a state half-life of one token — and the
untrained model destroys the write phase before the query phase
arrives; it then sits at chance recall for every load we test, and
training rarely digs it out. Initializing the gate bias at $-4.5$ gives
$a \approx 0.99$, a half-life of about seventy tokens, and the problem
disappears. Every production cell in this family makes the same move:
S4D's small initial step sizes (:numref:`subsec_s4d`), Mamba's
$\Delta$ bias, the retention bias we gave Gated DeltaNet in
:numref:`subsec_dn-gated`. Second, **mask before exponentiating**. The
dual builds $\mathbf{L}$ as $\exp(\textrm{cum}_i - \textrm{cum}_j)$;
above the diagonal that exponent is large and *positive*, and once
training pushes some head's decay toward zero — which language modeling
does within a few hundred steps — the $\exp$ overflows to infinity and
`inf * 0 = nan` kills the run. Masking with $-\infty$ *before* the
$\exp$, as `segsum` did in :numref:`subsec_ms-chunked`, makes the upper
triangle an exact zero instead. In our runs the recall task does not
trigger this
overflow; the language-modeling panel below does, reliably. A bug that
only fires on one of your two benchmarks is the expensive kind. Third,
**normalize the read-out**. The memory read $\mathbf{S}_t^\top
\mathbf{q}_t$ is a sum over everything the state has accumulated, so
its scale grows with how much has been written — order $\sqrt{T}$ at
initialization — while the residual stream it joins does not. This
family therefore normalizes the *output*, not a carried normalizer
state (the design decision discussed in :numref:`sec_matrix-state`,
and the RMSNorm inside our Gated DeltaNet cell): a per-head RMSNorm on
the read-out before the output projection. Without it, our four-block
stacks train erratically at this sequence length — in one framework
the hybrid never leaves chance recall — because the untrained
recurrent blocks flood the residual stream and the attention layer
downstream never sees the tokens.

```{.python .input #hybrids-three-matched-models-2}
%%tab pytorch
class ScalarGatedMixer(nn.Module):
    """Scalar-gated linear attention (Mamba-2's rung), via the dual."""
    def __init__(self, num_hiddens, num_heads):
        super().__init__()
        self.num_heads, self.d_head = num_heads, num_hiddens // num_heads
        self.W_qkv = nn.Linear(num_hiddens, 3 * num_hiddens, bias=False)
        self.W_g = nn.Linear(num_hiddens, num_heads)
        nn.init.constant_(self.W_g.bias, -4.5)     # A fresh gate must retain
        self.norm = nn.RMSNorm(self.d_head)        # Normalize the read-out
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)

    def forward(self, X, *_):
        B, T, d = X.shape
        q, k, v = self.W_qkv(X).reshape(B, T, self.num_heads, 3,
                                        self.d_head).unbind(3)
        log_a = -F.softplus(self.W_g(X))           # log a_t < 0, per head
        cum = log_a.cumsum(1)
        logL = cum[:, :, None, :] - cum[:, None, :, :]   # (B, T, T, heads)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                       device=X.device))
        logL = logL.masked_fill(~causal[None, :, :, None], -torch.inf)
        scores = torch.einsum('bihd,bjhd->bijh', q, k) / math.sqrt(self.d_head)
        Y = torch.einsum('bijh,bjhd->bihd', torch.exp(logL) * scores, v)
        return self.W_o(self.norm(Y).reshape(B, T, d))
```

```{.python .input #hybrids-three-matched-models-2}
%%tab jax
class ScalarGatedMixer(nnx.Module):
    """Scalar-gated linear attention (Mamba-2's rung), via the dual."""
    def __init__(self, num_hiddens, num_heads, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_heads, self.d_head = num_heads, num_hiddens // num_heads
        self.W_qkv = nnx.Linear(num_hiddens, 3 * num_hiddens, use_bias=False,
                                rngs=rngs)
        self.W_g = nnx.Linear(num_hiddens, num_heads, rngs=rngs,
                              bias_init=nnx.initializers.constant(-4.5))
        self.norm = nnx.RMSNorm(self.d_head, rngs=rngs)  # Normalize read-out
        self.W_o = nnx.Linear(num_hiddens, num_hiddens, use_bias=False,
                              rngs=rngs)

    def __call__(self, X, *_):
        B, T, d = X.shape
        qkv = self.W_qkv(X).reshape(B, T, self.num_heads, 3, self.d_head)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        log_a = -jax.nn.softplus(self.W_g(X))      # log a_t < 0, per head
        cum = jnp.cumsum(log_a, 1)
        logL = cum[:, :, None, :] - cum[:, None, :, :]   # (B, T, T, heads)
        causal = jnp.tril(jnp.ones((T, T), bool))
        logL = jnp.where(causal[None, :, :, None], logL, -jnp.inf)
        scores = jnp.einsum('bihd,bjhd->bijh', q, k) / math.sqrt(self.d_head)
        Y = jnp.einsum('bijh,bjhd->bihd', jnp.exp(logL) * scores, v)
        return self.W_o(self.norm(Y).reshape(B, T, d)), None
```

The attention mixer needs no new code at all: it is the causal
multi-head attention of :numref:`sec_gpt`, dropped into the
configurable block of :numref:`sec_transformer-block` through its
`attn_factory` hook — the same hook the GPT model itself uses. A stack
is then a *string*: one letter per block, `'G'` for the scalar-gated
recurrence, `'A'` for full attention. The entire architectural question
of this section fits in three such strings.

```{.python .input #hybrids-three-matched-models-3}
%%tab pytorch, jax
LAYOUTS = {'linear': 'GGGG', 'attention': 'AAAA', 'hybrid': 'GGAG'}
```

The hybrid places its one attention layer third of four — mid-stack,
not first, a choice the design rules below will justify. Everything
around the mixers (pre-norm residuals, MLPs, embeddings, readout) is
identical across the three models; the key and value channels are
embedded separately and summed, as in :numref:`subsec_dn-trained`, plus
a learned position embedding shared by all three (a `pos=False` switch
drops it, and a guard in `forward` refuses sequences longer than the
table — the first exercise relies on both).

```{.python .input #hybrids-three-matched-models-4}
%%tab pytorch
def make_blocks(layout, num_hiddens, num_heads):
    """'G' = scalar-gated recurrence block, 'A' = full-attention block."""
    gated = lambda: ScalarGatedMixer(num_hiddens, num_heads)
    attn = lambda: d2l.GPT.CausalAttention(num_hiddens, num_heads)
    return nn.ModuleList([
        d2l.TransformerBlock(num_hiddens, num_heads, norm='layer', act='gelu',
                             attn_factory=attn if c == 'A' else gated)
        for c in layout])

class RecallModel(nn.Module):
    def __init__(self, layout, num_keys, num_values, max_len,
                 num_hiddens=32, num_heads=4, pos=True):
        super().__init__()
        self.key_emb = nn.Embedding(num_keys, num_hiddens)
        self.val_emb = nn.Embedding(num_values + 1, num_hiddens)
        self.pos_emb = nn.Embedding(max_len, num_hiddens) if pos else None
        self.blocks = make_blocks(layout, num_hiddens, num_heads)
        self.norm = nn.LayerNorm(num_hiddens)
        self.head = nn.Linear(num_hiddens, num_values)

    def forward(self, keys, vals):
        X = self.key_emb(keys) + self.val_emb(vals)
        if self.pos_emb is not None:
            assert keys.shape[1] <= self.pos_emb.num_embeddings, \
                'sequence exceeds the position table: raise max_len'
            X = X + self.pos_emb(torch.arange(keys.shape[1],
                                              device=keys.device))
        for blk in self.blocks:
            X = blk(X)
        return self.head(self.norm(X))

for name, layout in LAYOUTS.items():
    model = RecallModel(layout, num_keys=64, num_values=32, max_len=128)
    print(f'{name:>10}: '
          f'{sum(p.numel() for p in model.parameters()):,} parameters')
```

```{.python .input #hybrids-three-matched-models-4}
%%tab jax
def make_blocks(layout, num_hiddens, num_heads, rngs):
    """'G' = scalar-gated recurrence block, 'A' = full-attention block."""
    gated = lambda rngs: ScalarGatedMixer(num_hiddens, num_heads, rngs=rngs)
    attn = lambda rngs: d2l.GPT.CausalAttention(num_hiddens, num_heads,
                                                rngs=rngs)
    return nnx.List([
        d2l.TransformerBlock(num_hiddens, num_heads, norm='layer', act='gelu',
                             attn_factory=attn if c == 'A' else gated,
                             rngs=rngs)
        for c in layout])

class RecallModel(nnx.Module):
    def __init__(self, layout, num_keys, num_values, max_len,
                 num_hiddens=32, num_heads=4, pos=True, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.key_emb = nnx.Embed(num_keys, num_hiddens, rngs=rngs)
        self.val_emb = nnx.Embed(num_values + 1, num_hiddens, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, num_hiddens, rngs=rngs) \
            if pos else None
        self.blocks = make_blocks(layout, num_hiddens, num_heads, rngs)
        self.norm = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.head = nnx.Linear(num_hiddens, num_values, rngs=rngs)

    def __call__(self, keys, vals):
        X = self.key_emb(keys) + self.val_emb(vals)
        if self.pos_emb is not None:
            assert keys.shape[1] <= self.pos_emb.num_embeddings, \
                'sequence exceeds the position table: raise max_len'
            X = X + self.pos_emb(jnp.arange(keys.shape[1]))
        for blk in self.blocks:
            X = blk(X)
        return self.head(self.norm(X))

for name, layout in LAYOUTS.items():
    model = RecallModel(layout, num_keys=64, num_values=32, max_len=128,
                        rngs=nnx.Rngs(0))
    n_params = sum(x.size for x in
                   jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f'{name:>10}: {n_params:,} parameters')
```

The printed counts back the "matched" claim with numbers: 58,544
(linear), 57,984 (attention), and 58,404 (hybrid) parameters — a spread
of 560, just under one percent, coming from the gate projection and
read-out norm that each scalar-gated mixer carries and the attention
mixer does not. Parameter count is not the whole story of a fair
comparison (the mixers do different amounts of compute per token, which
:numref:`tab_hy-ledger` prices), but it rules out the
crudest confound: none of the recall differences below can come from
one model simply being bigger.

### The Recall Sweep

Width is the experiment's one carefully chosen number. At
`num_hiddens=32` with four heads, each head's state is $8 \times 8$.
The random-key proposition of :numref:`subsec_ms-capacity` — used here
as a sizing heuristic, not as the theory of this model: a trained
four-layer stack with learned keys is far from the proposition's single
random-key memory — suggests a head crowds at order-eight bindings, so
a sweep of `num_pairs` from 4 to 64 should cross the whole stack's
capacity mid-sweep. (Widen the model and the cliff should move right;
the counting bound above says it cannot disappear.) One
protocol note: small-scale MQAR results are notoriously sensitive to
the learning rate, so :citet:`Arora.Eyuboglu.Timalsina.ea.2024` sweep
it per configuration and report the best. We piloted their grid
$\{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$ and pin the winner,
$3 \times 10^{-3}$, for every cell below; at $3 \times 10^{-4}$,
*every* architecture sits near chance at the largest loads, and a
careless single-rate comparison at that setting would report no
architecture gap at all. Optimization noise can manufacture or erase
the effect you are looking for — sweep or pin deliberately, and say
which.

```{.python .input #hybrids-the-recall-sweep-1}
%%tab pytorch
def train_recall(layout, num_pairs, lr=3e-3, num_keys=64, num_values=32,
                 epochs=16, batch_size=128):
    device = d2l.try_gpu()
    torch.manual_seed(0)
    num_train = 8192 if num_pairs <= 16 else 16384
    data = [torch.tensor(x, device=device) for x in
            make_recall(num_train, num_pairs, num_keys, num_values, seed=1)
            + make_recall(2048, num_pairs, num_keys, num_values, seed=2)]
    Xk, Xv, y, Vk, Vv, Vy = data
    model = RecallModel(layout, num_keys, num_values,
                        2 * num_pairs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i in torch.randperm(num_train, device=device).split(batch_size):
            loss = F.cross_entropy(
                model(Xk[i], Xv[i]).reshape(-1, num_values),
                y[i].reshape(-1), ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred, mask = model(Vk, Vv).argmax(-1), Vy != -1
        return float((pred[mask] == Vy[mask]).float().mean())

loads = [4, 8, 16, 32, 64]
recall = {name: [train_recall(layout, P) for P in loads]
          for name, layout in LAYOUTS.items()}
print(f"{'pairs':>10}" + ''.join(f'{P:>7}' for P in loads))
for name, accs in recall.items():
    print(f'{name:>10}' + ''.join(f'{a:>7.3f}' for a in accs))
print(f"{'chance':>10}" + f'{1 / 32:>7.3f}' * len(loads))
```

```{.python .input #hybrids-the-recall-sweep-1}
%%tab jax
def train_recall(layout, num_pairs, lr=3e-3, num_keys=64, num_values=32,
                 epochs=16, batch_size=128):
    num_train = 8192 if num_pairs <= 16 else 16384
    Xk, Xv, y = map(jnp.array, make_recall(num_train, num_pairs, num_keys,
                                           num_values, seed=1))
    Vk, Vv, Vy = map(jnp.array, make_recall(2048, num_pairs, num_keys,
                                            num_values, seed=2))
    model = RecallModel(layout, num_keys, num_values, 2 * num_pairs,
                        rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, keys, vals, tgt):
        def loss_fn(model):
            mask = tgt != -1
            losses = optax.softmax_cross_entropy_with_integer_labels(
                model(keys, vals), jnp.where(mask, tgt, 0))
            return (losses * mask).sum() / mask.sum()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    perm_rng = np.random.default_rng(0)
    for epoch in range(epochs):
        for i in np.split(perm_rng.permutation(num_train),
                          num_train // batch_size):
            train_step(model, optimizer, Xk[i], Xv[i], y[i])
    pred, mask = model(Vk, Vv).argmax(-1), Vy != -1
    return float((jnp.where(mask, pred == Vy, False)).sum() / mask.sum())

loads = [4, 8, 16, 32, 64]
recall = {name: [train_recall(layout, P) for P in loads]
          for name, layout in LAYOUTS.items()}
print(f"{'pairs':>10}" + ''.join(f'{P:>7}' for P in loads))
for name, accs in recall.items():
    print(f'{name:>10}' + ''.join(f'{a:>7.3f}' for a in accs))
print(f"{'chance':>10}" + f'{1 / 32:>7.3f}' * len(loads))
```

```{.python .input #hybrids-the-recall-sweep-2}
%%tab pytorch, jax
d2l.set_figsize((5, 3))
for name, marker in zip(LAYOUTS, ['o', 's', 'd']):
    d2l.plt.semilogx(loads, recall[name], marker=marker, label=name, base=2)
d2l.plt.axhline(1 / 32, color='black', lw=1, ls=':')
d2l.plt.xlabel('key-value pairs per sequence')
d2l.plt.ylabel('recall accuracy')
d2l.plt.legend();
assert min(recall['attention']) > 0.95 and min(recall['hybrid']) > 0.85
assert recall['hybrid'][-1] > 0.95 > recall['linear'][-1]  # One layer rescues
assert recall['linear'][-1] < 0.6      # The fixed state has saturated
```

The three curves are the section's thesis in one picture. The pure
linear-recurrence stack degrades as the load grows and collapses at 64
pairs, where it recovers fewer than half of the queries — consistent
with the crowding heuristic's cliff for $8 \times 8$ states asked to
hold 64 bindings, though read that as a diagnostic analogy: the
proposition's random-key assumptions do not cover this trained,
four-layer model. The pure attention stack scores 1.000 at every load
in both frameworks' runs: its
"state" at the query is the whole write phase, so load never crowds
it. And the hybrid, three quarters of which is the same recurrence
that just collapsed, tracks attention closely — in our PyTorch run it
too scores 1.000 at every load; in JAX it dips to roughly 0.92–0.94
mid-sweep
and returns to 0.99–1.00 at the two largest loads, exactly the loads
that break the linear stack. One mid-stack attention layer buys back
the
recall deficit at this scale — the miniature of the finding, in
controlled studies at the hundred-million-parameter scale, that adding
attention to a Mamba backbone lifts recall benchmarks by about thirty
points while moving reasoning benchmarks by single digits
:cite:`Lee.Yu.Zhang.ea.2025`.

### The Language-Modeling Panel

If the deficit is that dramatic, why did pure recurrent language models
ever look competitive? Because the deficit hides from the training
objective. We train the same three stacks — same widths, same blocks,
nothing changed but the input pipeline — as character-level language
models on *The Time Machine* of :numref:`sec_rnn-scratch`, with a
128-character context so that the window is long enough for recall to
matter in principle.

```{.python .input #hybrids-the-language-modeling-panel}
%%tab pytorch
class CharLM(nn.Module):
    def __init__(self, layout, vocab_size, max_len, num_hiddens=32,
                 num_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, num_hiddens)
        self.pos_emb = nn.Embedding(max_len, num_hiddens)
        self.blocks = make_blocks(layout, num_hiddens, num_heads)
        self.norm = nn.LayerNorm(num_hiddens)
        self.head = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X):
        pos = torch.arange(X.shape[1], device=X.device)
        H = self.emb(X) + self.pos_emb(pos)
        for blk in self.blocks:
            H = blk(H)
        return self.head(self.norm(H))

def train_lm(layout, data, lr=3e-3):
    device, vocab = d2l.try_gpu(), len(data.vocab)
    torch.manual_seed(0)
    model = CharLM(layout, vocab, data.num_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for X, y in data.train_dataloader():
        X, y = X.to(device), y.to(device)
        loss = F.cross_entropy(model(X).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        total = count = 0
        for X, y in data.val_dataloader():
            X, y = X.to(device), y.to(device)
            total += float(F.cross_entropy(model(X).reshape(-1, vocab),
                                           y.reshape(-1), reduction='sum'))
            count += y.numel()
    return total / count

data = d2l.TimeMachine(batch_size=64, num_steps=128, num_train=51200,
                       num_val=4096, tokenization='char')
print(f"{'model':>10} {'val loss':>9} {'bits/char':>10}")
lm_loss = {}
for name, layout in LAYOUTS.items():
    lm_loss[name] = train_lm(layout, data)
    print(f'{name:>10} {lm_loss[name]:>9.2f} '
          f'{lm_loss[name] / math.log(2):>10.2f}')
assert max(lm_loss.values()) - min(lm_loss.values()) < 0.15
```

```{.python .input #hybrids-the-language-modeling-panel}
%%tab jax
class CharLM(nnx.Module):
    def __init__(self, layout, vocab_size, max_len, num_hiddens=32,
                 num_heads=4, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.emb = nnx.Embed(vocab_size, num_hiddens, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, num_hiddens, rngs=rngs)
        self.blocks = make_blocks(layout, num_hiddens, num_heads, rngs)
        self.norm = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.head = nnx.Linear(num_hiddens, vocab_size, rngs=rngs)

    def __call__(self, X):
        H = self.emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blocks:
            H = blk(H)
        return self.head(self.norm(H))

def train_lm(layout, data, lr=3e-3):
    vocab = len(data.vocab)
    model = CharLM(layout, vocab, data.num_steps, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, X, y):
        def loss_fn(model):
            return optax.softmax_cross_entropy_with_integer_labels(
                model(X), y).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    for X, y in data.train_dataloader():
        train_step(model, optimizer, jnp.asarray(X), jnp.asarray(y))
    total = count = 0
    for X, y in data.val_dataloader():
        losses = optax.softmax_cross_entropy_with_integer_labels(
            model(jnp.asarray(X)), jnp.asarray(y))
        total += float(losses.sum())
        count += losses.size
    return total / count

data = d2l.TimeMachine(batch_size=64, num_steps=128, num_train=51200,
                       num_val=4096, tokenization='char')
print(f"{'model':>10} {'val loss':>9} {'bits/char':>10}")
lm_loss = {}
for name, layout in LAYOUTS.items():
    lm_loss[name] = train_lm(layout, data)
    print(f'{name:>10} {lm_loss[name]:>9.2f} '
          f'{lm_loss[name] / math.log(2):>10.2f}')
assert max(lm_loss.values()) - min(lm_loss.values()) < 0.15
```

All three land within roughly a tenth of a nat of one another — about
a tenth
of a bit per character — and in our runs the pure-linear stack matches
or beats the pure-attention stack: the model that just
collapsed on
the recall sweep is, by the language-modeling objective, the equal of
the models that aced it.
This is the cautionary dissociation, and it is the single most
important measurement in this section. Scope it first: one pass over
one small corpus at one seed, so it demonstrates that perplexity *can*
sit still while exact recall collapses, not that it always does. The
larger-scale evidence says the dissociation is the rule, though:
across seventeen trained language models most of the
efficient-architecture perplexity gap concentrated on the rare recall
tokens :cite:`Arora.Eyuboglu.Timalsina.ea.2024`, and in controlled
hybrid-ratio sweeps recall climbs steeply with the attention fraction
while perplexity barely responds :cite:`Wang.Zhu.Abreu.ea.2025`. The
mechanism is the same at both scales: next-character prediction on
ordinary text is dominated by local structure that a compressed,
always-on state models well, and the tokens that require exact
long-range retrieval are too rare to move the average. Perplexity was
blind to
the recall deficit — which is why the field needed MQAR-style probes
and needle-in-a-haystack evaluations to see it at all
:cite:`Arora.Eyuboglu.Timalsina.ea.2024,Waleffe.Byeon.Riach.ea.2024`,
and why a perplexity-matched "efficient" model can quietly fail as a
retrieval engine in production. Benchmark what you actually need.

### The Memory Bill, Measured

The third panel is the other side of the trade, on our own models. At
generation time an attention layer must carry its keys and values for
the whole context, :eqref:`eq_kv-cache-bytes`; a scalar-gated layer
carries its
$d_k \times d_v$ state per head, full stop. As in
:numref:`fig_hy-cache`, we count only the *persistent decode state* —
the bytes a server must hold for a user between decode steps.
Parameters, activations, and workspace are excluded; they are nearly
identical across the three stacks anyway, as the parameter counts
above showed. Counting those bytes per model at
our toy width (float32, batch of one):

```{.python .input #hybrids-the-memory-bill-measured}
%%tab pytorch, jax
num_heads, d_head, fp_bytes = 4, 8, 4
kv_layer = lambda T: 2 * T * num_heads * d_head * fp_bytes
state_layer = num_heads * d_head * d_head * fp_bytes

lengths = 2 ** np.arange(6, 17)
d2l.set_figsize((5, 3))
for name, layout in LAYOUTS.items():
    per_model = [sum(kv_layer(T) if c == 'A' else state_layer
                     for c in layout) / 2**10 for T in lengths]
    d2l.plt.loglog(lengths, per_model, label=name, base=2)
d2l.plt.xlabel('context length (tokens)')
d2l.plt.ylabel('persistent decode state (KiB)')
d2l.plt.legend();
```

The attention stack's line grows linearly with context; the linear
stack's is
flat at four kilobytes at every length; and the hybrid's line grows
with
exactly one quarter of the attention stack's slope — one of its four
layers still pays per token. That slope ratio *is* the attention
fraction, and it is the entire economic content of the hybrid design:
:numref:`fig_hy-cache` is this same plot at production width, where the
gap between the lines is measured in tens of gigabytes. Both sides of
the trade are now on the table, from our own runs: the hybrid recalls
like the attention model and pays like the recurrent one, plus one
quarter of the rent.

What the experiment shows: at this scale, on this diagnostic, one
attention layer in four recovers pure attention's recall while paying
a quarter of its growing state — with the caveats already given
(single seeded runs, a learning rate the pilot sweep pinned, a
capacity heuristic rather than a theorem). What it does not show: that
the result transfers to billion-parameter models trained on language.
For that, the ablation studies and shipped configurations of the next
two sections are the evidence.

## Design Rules, Measured
:label:`subsec_hy-design`

Our experiment fixed one design by fiat: one attention layer in four,
mid-stack, full attention, one recurrence. Each of those choices is an
axis, and by now each has been swept — by ablation studies at the
hundred-million-to-billion scale, and by the engineering teams whose
models the recipe table below records. :numref:`fig_hy-stacks` shows
where four representative systems landed.

![Four shipped answers to "where does the attention go?", drawn from each release's config at the variant named in the label. Jamba 52B repeats a 1-attention-in-8 block; Nemotron-H 8B spreads 4 attention layers on an 11-layer period, none at the front; Samba 3.8B alternates Mamba with sliding-window attention only; Zamba2 7B re-enters two weight-shared attention blocks 13 times along an 81-layer Mamba-2 backbone.](../img/mdl-modernrnn-hybrid-stacks.svg)
:label:`fig_hy-stacks`

**How much attention?** Less than you would guess, with a measurable
knee. The founding ablation is NVIDIA's 8B study
:cite:`Waleffe.Byeon.Riach.ea.2024`: sweeping the attention fraction,
validation loss was best near 8% attention and *degraded above it* —
more attention is not monotonically better — yielding the recipe
(roughly 43% Mamba-2, 7% attention, 50% MLP layers) that Nemotron-H and
Granite ship nearly verbatim. Sequential hybrids in production cluster
at 8–12.5% full attention (Jamba and MiniMax at 12.5%, Nemotron-H near
8%, Granite at 10%), with Hunyuan-TurboS lowest at 5.5%
:cite:`Hunyuan.Team.2025`. The gated-DeltaNet family sits deliberately
higher: Qwen3-Next fixes one full-attention layer per four
:cite:`Qwen.Team.2025`, and Kimi Linear's ablation of the ratio
$\{0{:}1, 1{:}1, 3{:}1, 7{:}1\}$ places the knee at 3:1 — at 7:1
training loss matched but validation degraded, at 1:1 nothing improved
but the bill, and pure attention (0:1) was strictly worse
:cite:`Kimi.Team.2025b`. Controlled academic sweeps agree from the
other side: recall climbs steeply with the attention fraction up to
about 1:3 while perplexity stays nearly flat across the whole range
:cite:`Wang.Zhu.Abreu.ea.2025` — the ratio axis is a recall dial, not
a perplexity dial, which is our dissociation panel again, now as a
design tool.

**Where does it go?** Evenly spread, and — in every report so far —
not first.
:citet:`Waleffe.Byeon.Riach.ea.2024` found no placement better than
even spacing with the first layer recurrent; the systematic study of
:citet:`Bae.Acun.Lin.ea.2025` is blunter — "never place Transformer
blocks at the front" — and finds middle placement optimal, which is why
our toy hybrid put its one attention layer third of four. Samba
supplies the sharpest single datapoint: inserting one full-attention
layer at the *front* of its stack made length extrapolation collapse
(perplexity exploding by 16K context), where the same capacity spent
elsewhere extrapolated to 256K :cite:`Ren.Liu.Lu.ea.2024`. The shipped
models agree: Nemotron-H's four attention layers sit at depths 8, 19,
30, 41 of 52 :cite:`NVIDIA.2025`; Granite's at 6, 16, 26, 36 of 40
:cite:`Granite.Team.2025`; Jamba's at 4, 12, 20, 28 of 32
:cite:`Lieber.Lenz.Bata.ea.2024`. None of these puts attention at
layer one. Read the rule as a heuristic distilled from a handful of
systematic studies plus shipped practice — consistent so far, but
extracted from a young design space, not derived from anything.

**Sequential or parallel?** Interleaving whole layers is not the only
composition. Hymba runs attention heads and Mamba-2 heads *in parallel
inside every layer*, on the same input, outputs normalized and fused —
and its controlled ablation has the parallel form beating the
equivalent sequential stack (45.2% against 44.1% average on its
commonsense suite), the argument being that the two mixers see the same
representation rather than each other's output
:cite:`Dong.Fu.Diao.ea.2024`. Falcon-H1 makes the same bet with a
continuously tunable split of channels between the attention and SSM
branches :cite:`Zuo.Velikanov.Chahed.ea.2025`, and the systematic
comparison of :citet:`Bae.Acun.Lin.ea.2025` places intra-layer hybrids
on the best quality-efficiency Pareto frontier, slightly ahead of
inter-layer. Sequential remains the production default — it is simpler
to schedule and to distill into — but the parallel results say the
margin is real, not folklore.

**Sharing the attention you keep.** Zyphra's Zamba pushes the economics
one step further: a single attention block whose *weights are shared*,
re-entered every six Mamba layers (thirteen times in the 7B model), so
global attention costs one block of parameters; Zamba2 refines this to
two shared blocks applied alternately, each re-entry specialized by a
cheap LoRA adapter
:cite:`Glorioso.Anthony.Tokpanov.ea.2024,Glorioso.Anthony.Tokpanov.ea.2024b`.
Parameter sharing is orthogonal to the sequential-parallel axis, and it
sharpens the section's moral: if a few attention layers carry
retrieval, perhaps they do not even need to be *different* layers.

**The pieces interact.** The components of a hybrid are not chosen
independently, and the cleanest evidence is a negative result. AI21
tested Mamba-2 — larger state, better standalone model — inside Jamba
and *rejected* it: the Mamba-1-plus-attention combination trained
better, their hypothesis being that once full-attention layers can pool
the entire context, the recurrence no longer needs the larger state
that made Mamba-2 win in isolation :cite:`Lieber.Lenz.Bata.ea.2024`.
The same report found no substantial quality difference between 1:3 and
1:7 attention ratios at their scale, so they shipped the cheaper. Both
decisions generalize: a hybrid's recurrent half should be judged *in
the presence of* its attention half, and where quality saturates,
economics decides. Sliding-window attention re-enters here as a middle
option — Samba's alternation of Mamba with windowed attention contains
no global layer at all, yet recovers most of the recall gap and
extrapolates to 256K, because a little *local* exact attention is
already most of what retrieval needs :cite:`Ren.Liu.Lu.ea.2024`, the
production echo of Based's finding that a small sliding window plus a
linear global memory recovers over 90% of full attention's recall
:cite:`Arora.Eyuboglu.Zhang.ea.2024`.

## The Recipe Table
:label:`subsec_hy-recipe`

As in :numref:`sec_scaling-laws`, we close the survey with the shipped
configurations themselves — one row per production hybrid, verified
against each model's technical report or released configuration. The
convergence is looser than the transformer recipe's (this design space
is younger), but the clustering is already unmistakable: a strong
recurrence from the ladder of :numref:`tab_dn-ladder`, roughly a tenth
to a quarter of layers as attention, evenly spread and none first,
with the attention layers wearing the cache compressions of
:numref:`sec_kv-cache`.

:Shipped hybrid recipes, 2024–2025. "Attention" counts full-attention layers (Samba's window-only attention noted); positions are 1-indexed. Layer counts follow each release's own convention — one entry per mixer block, so a Zamba2 entry is one backbone block and its 81 is not comparable to Jamba's 32 without the config; check the source config before comparing depths across rows. The context column gives the longest context the release claims to support (the configured maximum, where a config is public); training context and long-range evaluation length are different numbers, listed separately where the report distinguishes them (Samba trains at 4K and evaluates recall at 256K). For the recurrence column see :numref:`sec_mamba` and :numref:`sec_deltanet`; for the attention variants see :numref:`sec_kv-cache`.
:label:`tab_hy-recipe`

| model | layers | attention: count, positions | attention variant | recurrence | context |
|:--|:--|:--|:--|:--|:--|
| Jamba (52B-A12B) :cite:`Lieber.Lenz.Bata.ea.2024` | 32 | 4 (12.5%), at 4/12/20/28 | full, GQA $32{:}8$ | Mamba-1 | 256K |
| Samba (3.8B) :cite:`Ren.Liu.Lu.ea.2024` | 64 | none full; SWA every 2nd layer | SWA 2048, near-MQA | Mamba-1 | 4K train, 256K recall |
| Zamba2 (7B) :cite:`Glorioso.Anthony.Tokpanov.ea.2024b` | 81 | 13 re-entries of 2 *shared* blocks | full MHA $32{:}32$, LoRA per call | Mamba-2 | 4K |
| Nemotron-H (8B) :cite:`NVIDIA.2025` | 52 | 4 (7.7%), at 8/19/30/41 | full, GQA $32{:}8$ | Mamba-2 | 8K |
| Granite 4.0-H (32B-A9B) :cite:`Granite.Team.2025` | 40 | 4 (10%), at 6/16/26/36 | full, GQA $32{:}8$, NoPE | Mamba-2 | 128K |
| MiniMax-01 (456B-A46B) :cite:`MiniMax.2025` | 80 | 10 (12.5%), every 8th | full, GQA $64{:}8$ | lightning attention | 1M |
| Qwen3-Next (80B-A3B) :cite:`Qwen.Team.2025` | 48 | 12 (25%), every 4th | gated attn, GQA $16{:}2$ | gated DeltaNet | 262K |
| Kimi Linear (48B-A3B) :cite:`Kimi.Team.2025b` | 27 | 7 (26%), 3 KDA per MLA | MLA, NoPE | KDA (gated DeltaNet) | 1M |
| Falcon-H1 (34B) :cite:`Zuo.Velikanov.Chahed.ea.2025` | — | parallel: attn $\parallel$ SSM channels, every layer | full, GQA | Mamba-2 | 256K |

Read down the columns and the section's arguments reappear as shipped
hardware. The recurrence column is the last three sections' family —
Mamba-1/2 from :numref:`sec_mamba`, gated DeltaNet and its per-channel
refinement KDA from :numref:`subsec_dn-gated` (the cell we trained on
our scoreboard is the one Qwen3-Next ships). The attention-variant
column shows the composition argument in action: Kimi Linear pairs its
recurrence with MLA, Granite and Kimi drop positional encodings on the
attention layers (NoPE), Qwen3-Next keeps only 2 key–value heads — the
surviving cache, being most of the memory bill, receives every
compression :numref:`sec_kv-cache` taught. And the context column
explains the investment: the models built for the longest contexts are
hybrids of this table's shape, because :eqref:`eq_kv-cache-bytes` made
the alternative a hardware bill.

## Distillation, and Where This Leaves Us
:label:`subsec_hy-distill`

One practical question remains: must a hybrid be pretrained from
scratch? No — a pretrained transformer can be *converted*, and the
state-space duality explains why conversion has a head start without
guaranteeing it. The duality of :numref:`subsec_ms-duality` is an exact
statement about *linear* attention: a semiseparable mixing matrix can
be computed as a recurrence or as a masked matmul. A pretrained
transformer's *softmax* attention is not in that family, so its weights
are not a recurrence in disguise — but its projections and per-layer
mixing patterns are close enough to serve as a *warm start*:
initialize the student's recurrence from the attention projections and
learn the rest. MOHAWK distills in
three stages — match each layer's mixing matrix (the attention pattern
against the recurrence's semiseparable approximation of it), then match
hidden
states, then fine-tune end to end — and turns Phi-1.5 into a
Mamba-architecture model using about 3B tokens, under one percent of a
from-scratch pretraining budget, with quality above same-size
from-scratch Mamba models :cite:`Bick.Li.Xing.ea.2024`. "The Mamba in
the Llama" initializes the recurrence directly from the attention
projections and converts Llama-3 into a hybrid, keeping a fraction of
attention layers intact; quality falls smoothly as attention is
removed, the shipped configuration keeps about a quarter, and the
distilled hybrid still solves needle-in-a-haystack at twenty times its
training length — while the zero-attention variant does not
:cite:`Wang.Paliotta.May.ea.2024`. Both are learned approximations,
not exact conversions: the duality supplies the initialization and the
layer-wise matching targets, and the distillation data supplies
whatever softmax attention computed that a linear recurrence cannot.
The retained attention layers are
where the recall lives, which by this point in the section is
not a surprise; it is the design rule, observed a third way.

### The Chapter in One Table

The chapter's architectures differ in what they compute; what they
*charge* lines up in one table. Each row prices one layer of one
architecture on the axes this section has been trading against each
other: work and persistent state when decoding token $t$, work and
sequential depth when training on a length-$T$ sequence, and whether
the training-time parallel form computes the same function as the
step-by-step form in real arithmetic (floating-point reassociation
still changes rounding, which :numref:`subsec_ms-duality` measured).
Every "constant cost", "linear time", or "same computation" claim made
along the way resolves to a cell here, and each row cites the section
that established it.

:Per-layer contracts of the chapter's architectures. $d$ = model width, $h$ = heads of dimension $d_k = d_v$, $N$ = SSM state width per channel, $C$ = chunk length, $t$ = decode position, $T$ = training sequence length. Work entries are leading-order; MLP, normalization, and projection costs shared by all rows are omitted. "Exact" means the parallel and sequential forms coincide in real arithmetic.
:label:`tab_hy-ledger`

| architecture | decode work per token | persistent decode state | training work | sequential depth | parallel form exact? |
|:--|:--|:--|:--|:--|:--|
| LSTM / GRU (:numref:`sec_lstm`) | $O(d^2)$ | $O(d)$: $\mathbf{h}_t$ (and $\mathbf{c}_t$) | $O(T d^2)$ | $T$ | none: nonlinearity inside the recurrence (minGRU removes it, :numref:`subsec_mingru`) |
| diagonal SSM, S4D (:numref:`subsec_s4d`) | $O(d N)$ | $d N$ numbers | $O(T d N)$ work-efficient scan, or $O(T \log T\, d)$ FFT convolution (:numref:`subsec_ssm-conv`) | $O(\log T)$ (:numref:`subsec_parallel-scans`) | yes |
| Mamba-1 (:numref:`subsec_selective-ssm`) | $O(d N)$ | $d_\textrm{inner} N$ numbers | $O(T d N)$ work-efficient scan | $O(\log T)$ | yes: selective steps are still affine |
| SSD / Mamba-2 (:numref:`subsec_ms-duality`) | $O(h\, d_k d_v)$ | $h\, d_k d_v$ numbers | $O(T C d)$ chunked matmuls (:numref:`subsec_ms-chunked`) | $T/C$ | yes, verified to float rounding |
| GLA (:numref:`subsec_ms-decay-ladder`) | $O(h\, d_k d_v)$ | $h\, d_k d_v$ numbers | $O(T C d)$ chunked | $T/C$ | yes |
| DeltaNet (:numref:`subsec_dn-wy`) | $O(h\, d_k d_v)$ | $h\, d_k d_v$ numbers | $O(T C d)$ chunked WY | $T/C$ | yes: the WY form is exact algebra |
| softmax attention (:numref:`sec_gpt`, :numref:`sec_kv-cache`) | $O(t\, d)$ — grows | $2 t d$ numbers — grows (:eqref:`eq_kv-cache-bytes`) | $O(T^2 d)$ | $O(1)$ | trivially: the parallel form is the definition |
| 1-in-4 hybrid (this section) | $\tfrac{3}{4} O(h\, d_k d_v) + \tfrac{1}{4} O(t\, d)$ | constant $+\ \tfrac14$ of the cache | mixed | per component | per component |

Read across the attention row and the SSD row and the whole section
reappears: attention pays $t$-dependent work and state for exact recall;
the matrix-state family pays a constant for a bounded memory; the
hybrid row is a convex combination with the attention fraction as the
mixing weight. Read down the exactness column for the chapter's other
running theme: every linear-recurrence training trick — scan, chunk,
dual, WY — is a reassociation of the same arithmetic, not an
approximation, which is what made their speed free of modeling cost.

So: where does this leave the fixed state? This chapter gave it five
upgrades, and this section drew its boundary. The fixed state lost the
exact-recall fight — by a counting bound under its stated assumptions,
and by a collapse
measurable in an afternoon at width 32 — and it won the economics, not
narrowly: flat kilobytes against gigabytes that grow with every
token of context. Production stopped treating those as competing claims
and shipped both, in proportions this section measured from three
independent directions — ablation sweeps, shipped configurations, and
distillation experiments, all pointing at a small attention minority
carrying retrieval through a cheap recurrent majority. What this
section did *not* cover is the systems story that makes the recurrent
majority fast in practice — the chunked forms of
:numref:`subsec_ms-chunked` living as fused kernels, which belongs to
:numref:`chap_performance`; and the pretraining and post-training of
full-scale language models, hybrid or not, together with the serving
stacks that exploit their mostly flat memory bill, which are the subject
of the Language Models part. The pendulum question from this chapter's
introduction — whether attention remains on top on January 1, 2027 —
stays open. But notice what would settle it: not a better gate or a
cleverer write rule, both of which this chapter taught, but whether a
learned, compressed memory can be trusted with retrieval. That is the
measured question, and you now own every tool it is measured with.

## Summary

A fixed-size state cannot perform unbounded exact recall: reproducing
$k$ tokens from a vocabulary of $V$ requires $k \log_2 V$ bits in the
state, and a finite-state model's error on copying uniformly random
length-$L$ strings is at least
$1 - |\mathcal{S}|/D^L$; the failure is measurable as fuzzy recall
on phone-book and needle tasks long before it shows in perplexity. The
economics run the other way: by :eqref:`eq_kv-cache-bytes` only
attention layers pay a context-proportional cache, so a hybrid's
persistent decode state
is the attention fraction times the transformer's — Jamba's
measured 4 GB against Mixtral's 32 GB at 256K. Our three matched stacks
(parameter counts printed, spread under one percent)
made both claims concrete: the pure scalar-gated stack's recall
collapsed
once the load crossed roughly its $8 \times 8$-per-head state size, one
mid-stack attention layer restored recall at the loads
that broke it (1.000 throughout in our PyTorch run; 0.99–1.00 at the
post-collapse loads in JAX, with a mid-sweep dip to roughly 0.92–0.94),
language-modeling loss barely separated the three stacks (the
pure-linear model matched or beat pure attention in our
runs), and the persistent decode state grew
with slope equal to the attention fraction. The shipped design rules:
attention fractions cluster at 5–25%, with a measured knee near 1:3 for
the DeltaNet family and near 8% for Mamba-2 hybrids; placement is even
and, in every report so far, not front; parallel fusion and weight
sharing are live
refinements; and the components interact (Jamba chose Mamba-1 *because*
attention was present). Pretrained transformers distill into hybrids
for under a percent of the original token budget — a learned
approximation the duality warm-starts, with the retained
attention layers carrying recall. :numref:`tab_hy-ledger` collects the
per-layer cost and state contracts of every architecture in the
chapter. Three implementation lessons from the
experiment: initialize decay gates to retain, mask in log space before
exponentiating a decay kernel, and normalize the memory read-out
before it joins the residual stream.

## Exercises

1. *Placement, self-discovered.* [extended] Rerun the recall sweep
   with the
   attention layer first (`'AGGG'`) and last (`'GGGA'`) instead of
   mid-stack. Then probe length generalization — carefully, because
   `RecallModel`'s learned position table is a confound: it is built
   with `max_len = 2 * num_pairs`, so a longer evaluation sequence
   would index past it (the guard in `forward` stops you), and even a
   longer table leaves the extra rows untrained at evaluation time.
   Remove the confound by dropping absolute positions: adapt
   `train_recall` to build its models with `pos=False` — the recall
   task is content-addressed, so none of the three mixers needs the
   position table; verify this first by reproducing the sweep's
   accuracy at `num_pairs=32`. Then extend `make_recall` to insert
   filler tokens between the write and query phases (reserve one extra
   key index for the filler and pass `num_keys=65`), train at zero
   padding, and evaluate the same models on sequences padded by 32.
   Which placement degrades, and does your
   result match Samba's report that a single *front* attention layer
   breaks length extrapolation :cite:`Ren.Liu.Lu.ea.2024`?
1. *Price a million tokens.* [conceptual] Using
   :eqref:`eq_kv-cache-bytes`, compute
   the persistent decode state at a 1M-token context, batch of one,
   16-bit,
   for three 48-layer models with $n_\textrm{kv} = 8$ and
   $d_\textrm{head} = 128$: a pure transformer, a pure recurrent stack
   whose per-layer state matches Mamba-2 at $d_\textrm{model} = 4096$
   (expansion two, $d_\textrm{state} = 128$: $2 \cdot 4096 \cdot 128$
   elements per :eqref:`eq_ms-state-bytes`, ignoring the small
   convolution buffer), and
   a 12.5%
   hybrid. How many concurrent 1M-token users fit in 80 GB of spare
   HBM under each design? Compare your hybrid figure against Jamba's
   published 4 GB at 256K.
1. *The ratio axis.* [short-code] Sweep the attention fraction at
   fixed depth:
   layouts `'GGGG'`, `'GGAG'`, `'GAGA'`, `'AAAA'`. Plot recall at
   `num_pairs=64` and decode memory at 64K context (from the
   memory-panel cell) against the fraction. Where is the knee at this
   scale, and how does it compare with Kimi Linear's reported 3:1
   :cite:`Kimi.Team.2025b`?
1. *A window instead.* [extended] Replace the hybrid's full-attention
   mixer with
   sliding-window attention of window 16 (mask scores outside the
   window before the softmax). At which loads does the windowed hybrid
   match the full one, and where does it break? Explain both regimes:
   at what load does every query's answer still sit within the window,
   and what does this say about Samba's window-only design
   :cite:`Ren.Liu.Lu.ea.2024`?
1. *Bit accounting.* [conceptual] The linear stack's total state is
   $4$ layers
   $\times\, 4$ heads $\times\, 8 \times 8$ floats. Estimate its
   capacity in bindings via the crowding heuristic of
   :numref:`subsec_ms-capacity` (interference $(n-1)/d_k$ per head
   against unit signal) and via raw bits ($k \log_2 V$ at $V = 32$
   values against one float carrying, say, 8 useful bits). Where do
   the two accounts place the cliff, and where did the measured sweep
   put it?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.7]{.kicker}

Hybrid Architectures<br>
**what a fixed state cannot do · only attention pays rent · one layer rescues recall · the shipped recipes**
:::
:::

::: {.slide title="The wall no update rule moves"}
Five upgrades this chapter — gate, linearize, select, edit, learn — and
**none changed the state's size**.

- Reproducing $k$ tokens from vocabulary $V$ needs $k \log_2 V$ bits,
  held from write to query.
- Copy bound (Jelassi et al. 2024): a finite-state model errs with
  probability $\ge 1 - |\mathcal{S}|/D^L$ on **uniformly random**
  length-$L$ strings — a counting argument, no training trick appears
  in it.

. . .

Measured: a **410M transformer beats a 2.8B Mamba** on phone-book
lookup past ~70 entries.
:::

::: {.slide title="The diagnosis: associative recall"}
Where the wall bites in real text (Zoology, Arora et al. 2024):

- **82%** of the perplexity gap between efficient models and attention
  sits on the *recall slice* — tokens requiring retrieval of a binding
  seen once.
- Attention solves MQAR at width independent of length; a fixed state
  only while bindings fit.

. . .

Production symptom: *fuzzy recall* — MMLU 29 vs 46 at matched 8B;
ablate a hybrid's few attention layers and needle-in-a-haystack drops
to ~0.
:::

::: {.slide title="Only attention pays rent"}
Per layer, at production width (16-bit):

- Full attention (GQA $8 \times 128$): **4 KB per token** — 512 MB per
  layer at 128K context.
- Mamba-2 state: **~2 MB, constant** in context length.

@fig:mdl-modernrnn-hybrid-cache

The hybrid's bill = attention fraction × the transformer's.
:::

::: {.slide title="The Jamba bill, 256K context"}
| model | KV cache |
|:--|--:|
| Llama-2-70B-class | 128 GB |
| Mixtral | 32 GB |
| **Jamba (4 attn of 32 layers)** | **4 GB** |

. . .

Recurrence removes the growing term; the constant state can still be
quantized, but nothing grows. The surviving attention cache takes
GQA / MLA / quantization on top.
:::

::: {.slide title="Three matched models, one letter apart"}
Same depth, width, heads, MLPs, embeddings — only the mixer string
differs:

@hybrids-three-matched-models-3

- `'G'` = `ScalarGatedMixer`: scalar-per-head decay, **Mamba-2's rung**
  of the decay ladder (not GLA's per-coordinate gate), trained via the
  quadratic dual.
- `'A'` = the causal attention of ch. 11, through
  `d2l.TransformerBlock`'s `attn_factory` hook.
- Printed parameter counts: 58,544 / 57,984 / 58,404 — spread < 1%.
:::

::: {.slide title="Three lessons before training"}
**A fresh gate must retain.** Default init → $a \approx 0.5$: state
half-life of one token, chance recall at every load. Bias $-4.5$ →
$a \approx 0.99$. (S4D, Mamba, Gated DeltaNet all ship this trick.)

. . .

**Mask before exp.** The dual's $\exp(\textrm{cum}_i - \textrm{cum}_j)$
is large and *positive* above the diagonal; trained decays → 0 make it
overflow: `inf * 0 = nan`. Mask with $-\infty$ in log space first —
only the LM panel triggers it, never the recall task.

. . .

**Normalize the read-out.** $\mathbf{S}^\top \mathbf{q}$ grows with
what the state has accumulated ($\sqrt{T}$ at init); per-head RMSNorm
before $W_o$, or the recurrent blocks flood the residual stream and
the attention layer downstream never sees the tokens.
:::

::: {.slide title="The sweep: one layer buys back recall"}
@!hybrids-the-recall-sweep-2

- Linear: degradation, then collapse at 64 pairs — the crowding
  heuristic's cliff for $8 \times 8$ head states (a diagnostic
  analogy, not a theorem about this trained stack).
- Hybrid = 3/4 the *same* recurrence + one mid-stack attention layer:
  **1.000 at every load (PyTorch)**; JAX dips to ~0.92–0.94 mid-sweep,
  then 0.99–1.00 at the loads that break the linear stack. Attention:
  1.000 throughout.
:::

::: {.slide title="Perplexity misses the deficit"}
Same three stacks as character LMs on *The Time Machine*:

@!hybrids-the-language-modeling-panel

- All within ~0.1 nat; the **pure-linear model matches or
  beats pure attention** — while collapsing on recall. (One pass, one
  small corpus, one seed: a demonstration.)
- At scale the dissociation is the rule: the perplexity gap sits on
  the rare recall tokens (Zoology); ratio sweeps move recall, barely
  perplexity (2507.06457). MQAR and needle probes exist because the
  training objective did not see the deficit.
:::

::: {.slide title="Persistent decode state, measured"}
@!hybrids-the-memory-bill-measured

- KV cache + recurrent state only — parameters and activations are
  the same across stacks.
- Attention: grows linearly. Linear: 4 KiB at every length.
- Hybrid: **slope = attention fraction** (here 1/4).

. . .

Recalls like attention, pays like recurrence, plus a quarter of the
rent.
:::

::: {.slide title="Design rules, measured"}
- **Ratio:** shipped 5–25% attention; ablation optimum ~8% (NVIDIA);
  DeltaNet-family knee at 3:1 (Kimi ablation). Recall climbs to ~1:3;
  perplexity flat throughout.
- **Placement:** evenly spread, **not first in any report so far** —
  one front attention layer broke Samba's length extrapolation. A
  heuristic from a young design space, not a law.
- **Parallel** (Hymba, Falcon-H1): fuse per layer; beats sequential in
  controlled ablation.
- **Sharing** (Zamba2): re-enter two weight-tied attention blocks
  along the backbone.

@fig:mdl-modernrnn-hybrid-stacks
:::

::: {.slide title="The pieces interact"}
AI21 tested Mamba-2 in Jamba — the better standalone model — and
**rejected it**.

- With attention layers pooling the full context, the recurrence no
  longer needs the bigger state.
- No quality difference 1:3 vs 1:7 at their scale → shipped the
  cheaper.

. . .

Judge the recurrent half *in the presence of* the attention half;
where quality saturates, economics decides.
:::

::: {.slide title="The recipe table"}
| model | layers | attention | recurrence | context |
|:--|--:|:--|:--|--:|
| Jamba | 32 | 4 full, GQA | Mamba-1 | 256K |
| Nemotron-H | 52 | 4 full, GQA | Mamba-2 | 8K |
| Granite 4.0-H | 40 | 4 full, NoPE | Mamba-2 | 128K |
| Qwen3-Next | 48 | 12, GQA $16{:}2$ | gated DeltaNet | 262K |
| Kimi Linear | 27 | 7 MLA, NoPE | KDA | 1M |

Every row: evenly spread, none first; the surviving attention wears
every compression of ch. 11.
:::

::: {.slide title="Distillation: duality warm-starts, data does the rest"}
Duality is exact for **linear** attention only; softmax attention is
not a recurrence in disguise. Conversion is a *learned approximation*:

- **MOHAWK**: match mixing matrices → hidden states → fine-tune;
  Phi-1.5 → Mamba with ~3B tokens (<1% of scratch).
- **Mamba-in-the-Llama**: init recurrence from attention projections,
  keep ~1/4 of attention layers — recall survives at 20× training
  length; remove them all and it does not.
:::

::: {.slide title="The chapter in one table"}
Per layer: decode work / persistent state / training depth / exact?

| architecture | decode | state | depth | exact? |
|:--|:--|:--|:--|:--|
| LSTM/GRU | $O(d^2)$ | $O(d)$ | $T$ | no parallel form |
| S4D, Mamba-1 | $O(dN)$ | $dN$ | $O(\log T)$ | yes |
| SSD / GLA / DeltaNet | $O(h d_k d_v)$ | $h d_k d_v$ | $T/C$ | yes |
| softmax attention | $O(td)$ ↑ | $2td$ ↑ | $O(1)$ | by definition |
| 1-in-4 hybrid | mix | const + cache/4 | mix | per component |

. . .

The fixed state **lost exact recall** (a counting bound, plus our
measured collapse) and **won the economics** (by a factor that grows
with context). Production stopped choosing. Kernels → Computational
Performance; pretrained stacks → the Language Models part.
:::
