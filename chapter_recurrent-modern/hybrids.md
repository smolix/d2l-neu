# Hybrid Architectures
:label:`sec_hybrids`

A user pastes a table of 64 product codes and prices into a prompt and asks
for the prices of 20 of them. A programmer asks for a 2,000-token function to
be copied with one line changed. A transformer answers such requests by
looking back: every earlier token is still in its cache, and attention finds
the ones that match. A model with a fixed-size state has only what it chose
to store when the tokens went by, before it knew which of them would be
asked for. Every model in this chapter improved how such a state is written,
from gates to selective dynamics to the delta rule, but none of them can
make it hold more than its size allows. Retrieval from the context is a
large part of how language models are used, in few-shot prompts, in answers
grounded in retrieved documents, and in editing code, so this limit
matters.

This section measures the limit and shows the remedy that most recent large
recurrent models adopt. We first count how much a fixed state can hold. We
then train four small models on a recall task with a growing number of
pairs: two built from the recurrent layers of the previous sections, one
built from attention, and a *hybrid* in which one layer in four is
attention. We ask whether this one layer is enough to recall as well as a
transformer, and what the hybrid saves in memory, by counting that memory
for models of realistic size.

```{.python .input #hybrids-hybrid-architectures}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #hybrids-hybrid-architectures}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
```

## Limits of a Fixed-Size State

### Counting Bits

A state of $N$ numbers stored with $b$ bits each can be in at most
$2^{Nb}$ different configurations. Suppose a model reads $k$ tokens, each
drawn from a vocabulary of $V$ tokens, and must later reproduce all of them.
The $V^k$ possible inputs must then lead to $V^k$ different states, since two
inputs that lead to the same state receive the same answer. This requires

$$N b \ge k \log_2 V.$$
:eqlabel:`eq_hy-counting`

The state must hold $\log_2 V$ bits per token to be recalled, and the number
of tokens it can recall grows in proportion to its size and not at all with
the length of the context. A transformer's cache grows by one key and one
value per token, so it never runs out. :citet:`Jelassi.Brandfonbrener.Kakade.ea.2024`
turn this argument into a lower bound on the state size needed to copy, and
show that small transformers learn to copy strings far longer than
recurrent models of the same size can.

The bound counts bits that a perfect encoding would use, and it is far from
tight for the models below. Their four recurrent layers hold 512 numbers each,
2,048 numbers of 32 bits in all, many times the 256 bits that 64 values of 4
bits take. A learned matrix memory uses its numbers far less efficiently.
:numref:`sec_matrix-state` showed that after $n$ writes with random unit keys of dimension $d_k$, the
expected squared read error is $(n-1)/d_k$, so a head reads back about $d_k$
pairs before the interference between keys is as large as the values
themselves. The models below have eight heads per layer with keys of
dimension 8, so we expect each head to recall only a few pairs cleanly.

### Associative Recall

To measure recall we use a version of *multi-query associative recall*
:cite:`Arora.Eyuboglu.Timalsina.ea.2024`. A sequence begins with $n$ pairs
of a key and a value, for distinct keys drawn from 64 and values drawn from
16, and ends with a query for every key, in random order. At each query, the
model must output the value that was paired with that key. Between the pairs
and the queries, a filler token pads the sequence to 128 tokens.
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` found that this kind of recall
accounts for most of the gap in perplexity on natural text between attention
and the gated convolution models they studied. To keep the task about
memory rather than about combining neighboring tokens, each pair is one
token, so the vocabulary has a token for each of the $64 \times 16$ pairs,
one for each key as a query, one for each value as an answer, and the
filler.

```{.python .input #hybrids-associative-recall-1}
%%tab pytorch, jax
def recall_batch(rng, batch_size, num_pairs, num_keys=64, num_values=16,
                 num_steps=128):
    """Key-value pairs as single tokens, then a query for every key.

    Token ids: values, then pairs, then queries, then the filler."""
    keys = np.argsort(rng.random((batch_size, num_keys)), 1)[:, :num_pairs]
    values = rng.integers(0, num_values, (batch_size, num_pairs))
    pair_ids = num_values + keys * num_values + values
    order = np.argsort(rng.random((batch_size, num_pairs)), 1)
    query_ids = num_values + num_keys * num_values + np.take_along_axis(
        keys, order, 1)
    filler = num_values + num_keys * num_values + num_keys
    X = np.full((batch_size, num_steps), filler)
    X[:, :num_pairs], X[:, -num_pairs:] = pair_ids, query_ids
    return X, np.take_along_axis(values, order, 1)  # Answers to the queries

num_keys, num_values = 64, 16
vocab_size = num_values + num_keys * num_values + num_keys + 1  # And filler
X, Y = recall_batch(np.random.default_rng(0), 1, 4)
print('tokens:', X[0, :4], '...', X[0, -4:], ' answers:', Y[0])
```

## Recall with One Attention Layer

### Four Models

All four models are instances of `d2l.SeqLM` with four blocks of width 64
and eight heads per layer, so that keys and values have 8 dimensions; only
the mixers differ. Two use the recurrent
layers of the previous sections, gated linear attention
(:numref:`sec_matrix-state`) and Gated DeltaNet (:numref:`sec_deltanet`).
The third uses causal softmax attention in every block, and the fourth,
the hybrid, replaces the first of its four Gated DeltaNet layers by
attention. The attention mixer is the multi-head attention of
:numref:`chap_attention`, computed with the fused attention function of each
framework. It needs no positional encoding for this task, because a query
finds its pair by content alone.

```{.python .input #hybrids-four-models-1}
%%tab pytorch
class CausalAttention(nn.Module):
    """Causal multi-head softmax attention, with the interface of a mixer."""
    def __init__(self, num_hiddens, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(num_hiddens, 3 * num_hiddens, bias=False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)

    def forward(self, X):  # X: (batch_size, num_steps, num_hiddens)
        B, T, d = X.shape
        Q, K, V = (Z.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for Z in self.W_qkv(X).chunk(3, -1))
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        return self.W_o(Y.transpose(1, 2).reshape(B, T, d))
```

```{.python .input #hybrids-four-models-1}
%%tab jax
class CausalAttention(nnx.Module):
    """Causal multi-head softmax attention, with the interface of a mixer."""
    def __init__(self, num_hiddens, num_heads, *, rngs):
        self.num_heads = num_heads
        self.W_qkv = nnx.Linear(num_hiddens, 3 * num_hiddens, use_bias=False,
                                rngs=rngs)
        self.W_o = nnx.Linear(num_hiddens, num_hiddens, use_bias=False,
                              rngs=rngs)

    def __call__(self, X):  # X: (batch_size, num_steps, num_hiddens)
        B, T, d = X.shape
        Q, K, V = (Z.reshape(B, T, self.num_heads, -1)
                   for Z in jnp.split(self.W_qkv(X), 3, -1))
        Y = jax.nn.dot_product_attention(Q, K, V, is_causal=True)
        return self.W_o(Y.reshape(B, T, d))
```

A string describes the mixers of a model, one letter per block.

```{.python .input #hybrids-four-models-2}
%%tab pytorch
def recall_model(layout, vocab_size, num_hiddens=64, num_heads=8):
    """SeqLM whose mixers are given by letters: L(inear attention),
    G(ated DeltaNet), A(ttention)."""
    make = {'L': d2l.GatedLinearAttention, 'G': d2l.GatedDeltaNet,
            'A': CausalAttention}
    return d2l.SeqLM(vocab_size, num_hiddens,
                     [make[c](num_hiddens, num_heads) for c in layout])
```

```{.python .input #hybrids-four-models-2}
%%tab jax
def recall_model(layout, vocab_size, num_hiddens=64, num_heads=8, seed=0):
    """SeqLM whose mixers are given by letters: L(inear attention),
    G(ated DeltaNet), A(ttention)."""
    rngs = nnx.Rngs(seed)
    make = {'L': d2l.GatedLinearAttention, 'G': d2l.GatedDeltaNet,
            'A': CausalAttention}
    return d2l.SeqLM(vocab_size, num_hiddens, [
        make[c](num_hiddens, num_heads, rngs=rngs) for c in layout], rngs=rngs)
```

### The Recall Sweep

Each model is trained once on fresh batches whose number of pairs cycles
through 8, 16, 32, and 64, with the loss computed at the query positions
only, and is then tested separately at each number of pairs. Training uses
AdamW with learning rate 0.003 for 4,000 steps of 64 sequences.

```{.python .input #hybrids-the-recall-sweep-1}
%%tab pytorch
def train_recall(model, pairs=(8, 16, 32, 64), num_steps=4000, lr=3e-3):
    """Train on recall batches with a cycling number of pairs; return the
    test accuracy for each number of pairs."""
    device, rng = d2l.try_gpu(), np.random.default_rng(0)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(num_steps):
        n = pairs[step % len(pairs)]
        X, Y = (torch.tensor(A, device=device) for A in recall_batch(rng, 64, n))
        loss = F.cross_entropy(model(X)[:, -n:].flatten(0, 1), Y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = {}
    for n in pairs:
        X, Y = (torch.tensor(A, device=device)
                for A in recall_batch(np.random.default_rng(1), 1024, n))
        with torch.inference_mode():
            accuracy[n] = (model(X)[:, -n:].argmax(-1) == Y).float().mean().item()
    return accuracy
```

```{.python .input #hybrids-the-recall-sweep-1}
%%tab jax
@nnx.jit
def mqar_step(model, optimizer, X, Y):
    def loss_fn(model):
        logits = model(X)[:, -Y.shape[1]:]  # Predictions at the queries
        return optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def mqar_predict(model, X):
    return model(X).argmax(-1)

def train_recall(model, pairs=(8, 16, 32, 64), num_steps=4000, lr=3e-3):
    """Train on recall batches with a cycling number of pairs; return the
    test accuracy for each number of pairs."""
    rng = np.random.default_rng(0)
    optimizer = nnx.Optimizer(model, optax.adamw(lr, weight_decay=0.01),
                              wrt=nnx.Param)
    step = nnx.cached_partial(mqar_step, model, optimizer)  # Compile once
    predict = nnx.cached_partial(mqar_predict, model)
    for i in range(num_steps):
        n = pairs[i % len(pairs)]
        step(*recall_batch(rng, 64, n))
    accuracy = {}
    for n in pairs:
        X, Y = recall_batch(np.random.default_rng(1), 1024, n)
        accuracy[n] = float((predict(X)[:, -n:] == Y).mean())
    return accuracy
```

```{.python .input #hybrids-the-recall-sweep-2}
%%tab pytorch
layouts = {'gated linear attention': 'LLLL', 'Gated DeltaNet': 'GGGG',
           'attention': 'AAAA', 'hybrid': 'AGGG'}
accuracy = {}
for name, layout in layouts.items():
    torch.manual_seed(0)
    accuracy[name] = train_recall(recall_model(layout, vocab_size))
```

```{.python .input #hybrids-the-recall-sweep-2}
%%tab jax
layouts = {'gated linear attention': 'LLLL', 'Gated DeltaNet': 'GGGG',
           'attention': 'AAAA', 'hybrid': 'AGGG'}
accuracy = {name: train_recall(recall_model(layout, vocab_size))
            for name, layout in layouts.items()}
```

```{.python .input #hybrids-the-recall-sweep-3}
%%tab pytorch, jax
pairs = list(next(iter(accuracy.values())))
for name, acc in accuracy.items():
    print(f'{name:24s}', ' '.join(f'{acc[n]:.2f}' for n in pairs))
d2l.plot(pairs, [list(acc.values()) for acc in accuracy.values()],
         'pairs in the sequence', 'recall accuracy', legend=list(accuracy),
         figsize=(4.5, 3))
```

With 8 pairs, every model recalls nearly all values. As the number of pairs
grows, the two recurrent models lose more of them. At 64 pairs, gated linear
attention recalls only two thirds to four fifths of the values, depending on
the run, and Gated DeltaNet about nine in ten or more: its corrected writes
lose less of the state to interference, but its losses still grow with the
number of pairs. Attention recalls nearly
every pair at every load, and so does the hybrid, although three of its four
layers are recurrent. Its single attention layer looks up the pairs that the
recurrent state cannot hold.

### Language Modeling

Does the difference show on ordinary text? We train two more models on the
character-level Penn Treebank with the recipe of :numref:`sec_lstm`: the
transformer `d2l.TinyLM` of :numref:`sec_adam`, two blocks of width 128 with
learned position embeddings, and a hybrid `d2l.SeqLM` of the same width
whose first block is Gated DeltaNet and second block attention. With this
recipe, Gated DeltaNet alone reached a perplexity of about 3.2.

```{.python .input #hybrids-language-modeling-1}
%%tab pytorch, jax
data = d2l.PennTreebank(batch_size=64, num_steps=128)
```

```{.python .input #hybrids-language-modeling-2}
%%tab pytorch
torch.manual_seed(0)
models = {'attention (TinyLM)': d2l.TinyLM(len(data.vocab), d_model=128,
                                           num_heads=4, num_blks=2, max_len=128),
          'hybrid': d2l.SeqLM(len(data.vocab), 128, [
              d2l.GatedDeltaNet(128, 4), CausalAttention(128, 4)])}
for name, model in models.items():
    print(f'{name:20s} perplexity {d2l.fit_lm(model, data):.2f}')
```

```{.python .input #hybrids-language-modeling-2}
%%tab jax
rngs = nnx.Rngs(0)
models = {'attention (TinyLM)': d2l.TinyLM(len(data.vocab), d_model=128,
                                           num_heads=4, num_blks=2, max_len=128,
                                           rngs=rngs),
          'hybrid': d2l.SeqLM(len(data.vocab), 128, [
              d2l.GatedDeltaNet(128, 4, rngs=rngs),
              CausalAttention(128, 4, rngs=rngs)], rngs=rngs)}
for name, model in models.items():
    print(f'{name:20s} perplexity {d2l.fit_lm(model, data):.2f}')
```

The transformer and the hybrid both reach a perplexity of about 3.3,
slightly behind Gated DeltaNet alone. Language modeling at this scale barely
registers the difference in recall, because predicting the next character of
ordinary text rarely requires looking up a pair stored long before. A
comparison by perplexity alone can therefore miss what a model cannot
recall. Note also the order of the layers in this hybrid. Its attention
layer has no positional encoding; in the recall task it finds pairs by
content and needs no order, but in text it needs to know the order of the
tokens, which the recurrent layer before it provides.

## Inference Memory

During generation a model keeps, for every sequence it is serving, what it
needs to compute the next token. For an attention layer this is the
key–value cache of :numref:`sec_kv-cache`: one key and one value per past
token for each of the $n_\textrm{kv}$ key–value heads of dimension
$d_\textrm{head}$, that is,

$$2 \cdot n_\textrm{kv} \cdot d_\textrm{head} \cdot (\textrm{bytes per number})$$

bytes per token and layer. With grouped-query attention, 8 key–value heads
of dimension 128 in 16-bit numbers, that is 4 KB per token. A recurrent
layer keeps its state, whose size does not depend on the length: a Mamba-2
layer in a model of width 4,096 has 8,192 channels with 128 states each,
about one million numbers, or 2 MB. The attention layer's cache overtakes
this state after about 500 tokens, and at 128,000 tokens it holds 512 MB,
256 times as much.

Only the attention layers of a hybrid have caches that grow, so its memory
per sequence is the transformer's times the fraction of layers that are
attention, plus a constant. :numref:`fig_hy-cache` shows the effect for a
model of 32 layers. Jamba :cite:`Lieber.Lenz.Bata.ea.2024`, with 4 attention
layers among 32, reports a cache of 4 GB at a context of 256,000 tokens,
against 32 GB for a transformer of similar size. Memory per sequence decides
how many sequences a GPU can serve at once, and because generating a token
reads the whole cache from memory (:numref:`sec_kv-cache`), it also sets
how fast each token is generated.

![Memory held per sequence during generation, against context length, for a 32-layer model with grouped-query attention (8 key–value heads of dimension 128) and Mamba-2-sized recurrent states, in 16-bit numbers. A transformer's cache reaches 32 GB at 256K tokens; a purely recurrent model stays at 64 MB at every length; a hybrid with 4 attention layers pays an eighth of the transformer's cache.](../img/mdl-modernrnn-hybrid-cache.svg)
:label:`fig_hy-cache`

The chapter asked how to build a memory of fixed size that trains in
parallel, forgets the right things, stores many associations, and can
overwrite them. Its answers, gated linear recurrences computed by scans,
selective state space models, matrix memories trained in chunks, and the
delta rule, make the fixed state as useful as possible. The answer of
practice adds a small memory that grows: a few attention layers, which
retrieve exactly what the fixed state cannot hold. How few, where to place
them, and whether some of them can attend to a window of recent tokens
instead of the whole context are questions that current models answer
differently; :numref:`sec_ssm-further-reading` lists some of their choices.

## Summary

A state of fixed size holds a fixed number of bits, so a recurrent model can
recall only as many items from its context as its state can store, while an
attention layer keeps every token and can always look back. On multi-query
associative recall, gated linear attention and Gated DeltaNet lose more
pairs as the number of pairs grows, Gated DeltaNet fewer than gated linear
attention. Attention recalls all of them, and so does a hybrid with one
attention layer among four. Language-modeling perplexity
barely shows the difference. During generation only the attention layers
keep a cache that grows with the context, so a hybrid needs the
transformer's cache times its fraction of attention layers, an eighth for a
model with one attention layer in eight. Most recent large models with
recurrent layers are hybrids for this reason.

## Exercises

1. **The counting bound.** A model reads a string of $k$ tokens from a
   vocabulary of $V$ and must then reproduce it.
    1. Prove :eqref:`eq_hy-counting` for a model whose state between reading
       and writing has $N$ numbers of $b$ bits and whose output is a
       deterministic function of that state.
    1. Suppose instead that the model may output a wrong string with
       probability $\epsilon$ when the string is drawn uniformly. Show that
       $Nb \ge (1 - \epsilon)\, k \log_2 V - 1$. (Hint: count the strings
       that each state can decode correctly.)
    1. How many tokens of a vocabulary of 50,000 can a model with 48 layers
       of Mamba-2 state, 2 MB each in 16-bit numbers, copy at most? Why does
       no model of this size come close in practice?

1. **Memory for many users.** A GPU with 80 GB serves a model of 32
   layers, with the cache sizes of :numref:`fig_hy-cache`, after 16 GB of
   the GPU's memory are taken by the weights.
    1. How many sequences of 128,000 tokens can the GPU serve at once with
       the transformer, the hybrid with 4 attention layers, and the
       recurrent model?
    1. Grouped-query attention with fewer key–value heads, and 8-bit
       numbers for the cache, each halve the cache. Apply both to the
       transformer and to the hybrid, and repeat the count. Which change
       matters more for the hybrid?

1. [code] **More pairs.** Extend the recall task to 128 keys, sequences of
   256 tokens, and up to 128 pairs. Train Gated DeltaNet and the hybrid.
   Where does each model lose pairs, and how does the result compare with
   the counting bound and with the interference estimate of the section?

1. [code] **Where and how much attention?** Train the hybrids `GAGG`,
   `GGGA`, and `GAGA` with the setup of the recall sweep.
    1. Compare their recall with that of `AGGG`. Does the position of the
       attention layer matter? Consider what the attention layer receives
       in each case: the embedding of each token, or the output of
       recurrent layers that have mixed each token with the ones before it.
    1. At 64 pairs `AGGG` already recalls every value. Repeat the
       comparison with the larger task of the previous exercise. Does a
       second attention layer help there?

1. [code] **A window instead.** Replace the attention mixer by attention
   that sees only the last $w$ tokens, by masking the scores of earlier
   tokens.
    1. Train the hybrid with $w = 32$ and $w = 128$ and compare recall at
       every number of pairs. Which queries can a window of 32 answer, and
       which must the recurrent layers answer?
    1. How large is the cache of this layer during generation? What does
       the comparison suggest about combining windows with recurrence?

1. [code] **Recall in text.** In the Penn Treebank validation text, mark
   the positions whose character completes a string of at least eight
   characters that already occurred earlier in the same window. Compare
   the loss of Gated DeltaNet, the hybrid, and `d2l.TinyLM` at the marked
   positions and at all other positions. Does the hybrid's advantage on
   recall show up where text repeats itself?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.6]{.kicker}

Hybrid architectures<br>
**a fixed state holds fixed bits · associative recall · one attention layer in four · memory per sequence**
:::
:::

::: {.slide title="A Fixed State Stores What It Chose Before the Question"}
[The problem]{.kicker}

A prompt lists 64 product codes and prices, then asks for 20 of them.

- A transformer looks back: every token is still in its cache.
- A recurrent model has only what it wrote into its state as the tokens
  went by, before it knew which would be asked for.

::: {.d2l-note}
Better writes (gates, selectivity, the delta rule) do not make a state
hold more than its size allows.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
- bound the recall of a fixed state by counting bits;
- measure recall of four models as the number of stored pairs grows;
- compute the memory per sequence of attention, recurrent, and hybrid
  models.

**Running example:** a Mamba-2 layer's state of about 2 MB against a
grouped-query attention cache of 4 KB per token.
:::

::: {.slide title="Recalling k Tokens Needs k log₂ V Bits"}
$V^k$ possible inputs must reach $V^k$ different states:

$$N b \ge k \log_2 V$$

$N$ numbers of $b$ bits; $k$ tokens from a vocabulary of $V$.

- Recall grows with the state, not with the context.
- A KV cache grows by one key and one value per token: it never runs out.
:::

::: {.slide title="A Learned Memory Uses Its Bits Poorly"}
- Our recurrent models: 4 layers × 8 heads × 8 × 8 = 2,048 numbers of
  32 bits; 64 values of 4 bits take 256 bits.
- But a head with keys of dimension $d_k$ reads back about $d_k$ random
  pairs before interference equals the signal: $(n - 1)/d_k$.
- With $d_k = 8$: a few pairs per head.
:::

::: {.slide title="Multi-Query Associative Recall"}
$n$ pairs (key from 64, value from 16) as single tokens, fillers, then a
query for every key in random order. Answer each with its value.

@!hybrids-associative-recall-1

Chance: 1/16.
:::

::: {.slide title="Four Models Differ Only in Their Mixers"}
`d2l.SeqLM`, four blocks of width 64, eight heads per layer:

| model | mixers |
|---|---|
| gated linear attention | L L L L |
| Gated DeltaNet | G G G G |
| attention | A A A A |
| **hybrid** | **A** G G G |

Attention without positional encoding: a query finds its pair by content.
Trained once with 8, 16, 32, 64 pairs; tested at each.
:::

::: {.slide title="One Attention Layer Restores Recall"}
@!hybrids-the-recall-sweep-3
:::

::: {.slide title="Check: Bits Against Pairs"}
The four recurrent layers hold 2,048 numbers of 32 bits.

1. By the counting bound, how many 4-bit values could they hold?
2. Why does Gated DeltaNet already lose pairs at 64?

::: {.fragment .fade-up}
1. $2048 \cdot 32 / 4 = 16{,}384$ values.
2. Interference: keys of 8 dimensions overlap once a head holds more than
   a few pairs; the bound counts bits a perfect code would use.
:::
:::

::: {.slide title="Perplexity Barely Shows the Gap"}
@!hybrids-language-modeling-2

- Gated DeltaNet alone: about 3.2.
- The next character rarely depends on a pair stored long before.

::: {.d2l-note .warn}
Benchmark what you need: perplexity alone can miss what a model cannot
recall.
:::
:::

::: {.slide title="Attention Stores 4 KB per Token, Mamba-2 About 2 MB in All"}
Per layer, per sequence, 16-bit numbers:

| layer | memory |
|---|---|
| GQA, 8 KV heads × 128 | $2 \cdot 8 \cdot 128 \cdot 2$ B = 4 KB per token |
| Mamba-2, width 4,096 | $8192 \cdot 128 \cdot 2$ B ≈ 2 MB, any length |

Crossover near 500 tokens; at 128 K tokens, 512 MB against 2 MB.
:::

::: {.slide title="Only the Attention Layers of a Hybrid Keep a Cache"}
![](../img/mdl-modernrnn-hybrid-cache.svg){width=85%}

Jamba, 4 attention layers of 32: 4 GB at 256 K tokens, against 32 GB.
:::

::: {.slide title="Check: Memory of a Hybrid"}
32 layers, 4 of them attention, 128 K tokens, numbers as above.

1. Memory per sequence of the hybrid?
2. Of the transformer?

::: {.fragment .fade-up}
1. $4 \cdot 512$ MB $+ 28 \cdot 2$ MB $\approx 2.1$ GB.
2. $32 \cdot 512$ MB $= 16$ GB: eight times as much.
:::
:::

::: {.slide title="Recap: One Template, Six Memories"}
| | state per layer | update | recall | PTB |
|---|---|---|---|---|
| LSTM, GRU | vector | nonlinear, sequential | short | 3.0 |
| minGRU | vector | $\mathbf{a}_t\odot\mathbf{h} + \mathbf{b}_t$, scan | | 3.2 |
| S4D | $d \times N$ | fixed decays, FFT | by distance | 3.35 |
| Mamba | $H \times N$ | selective decays, scan | selective | 2.95 |
| gated linear attention | $d_v \times d_k$ | $a_t\mathbf{S} + \mathbf{v}\mathbf{k}^\top$, chunks | interference | 3.25 |
| Gated DeltaNet | $d_v \times d_k$ | delta rule, chunks | rebinds | 3.2 |
| + one attention layer | + KV cache | | exact | |

**Open:** how few attention layers, where, and whether windows suffice.
:::
