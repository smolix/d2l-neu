# Hybrid Architectures
:label:`sec_hybrids`

Every model in this chapter improved how a fixed-size state is written,
from gates to selective dynamics to the delta rule, but none of them can
make the state hold more than its size allows. How much can a state of
fixed size recall from its context, and when does that matter? Consider a
user who pastes a table of 64 product codes and prices into a prompt and
asks for the prices of 20 of them. A transformer answers such a request by
looking back: every earlier token is still in its cache, and attention
finds the ones that match. A model with a fixed-size state has only what it
chose to store when the tokens went by, before it knew which of them would
be asked for. Retrieval from the context is a large part of how language
models are used, in few-shot prompts, in answers grounded in retrieved
documents, and in editing code, where a programmer may ask for a
2,000-token function to be copied with one line changed.

The remedy that most recent large recurrent models adopt is modest: keep a
few attention layers among the recurrent ones. Before we trust it, we want
to know how much a fixed state can hold, and where a small one should start
to fail. We then train small models with and without one attention layer,
on a recall task and on text, and count what a transformer, a recurrent
model, and a hybrid of production size keep in memory while they generate.
Since this section closes the chapter, it ends by setting the chapter's
memories side by side and asking which one to use.

```{.python .input #hybrids-hybrid-architectures}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #hybrids-hybrid-architectures}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Limits of a Fixed-Size State

### Counting Bits

Let us start with the crudest argument, which needs nothing but counting: a
state cannot tell apart more inputs than it has configurations. A state of
$M$ numbers stored with $b$ bits each can be in at most $2^{Mb}$ different
configurations. Suppose a model reads $k$ tokens, each drawn from a
vocabulary of $V$ tokens, and must later reproduce all of them. The $V^k$
possible inputs must then lead to $V^k$ different states, since two inputs
that lead to the same state receive the same answer. This requires

$$M b \ge k \log_2 V.$$
:eqlabel:`eq_hy-counting`

The state must hold $\log_2 V$ bits per token to be recalled, and the number of
tokens it can recall grows in proportion to its size and not at all with the
length of the context. A transformer's cache grows by one key and one value per
token, so it never runs out. :citet:`Jelassi.Brandfonbrener.Kakade.ea.2024`
turn this argument into a lower bound on the state size needed to copy, and
show that small transformers learn to copy strings far longer than recurrent
models of the same size can.

The bound counts the bits that a perfect encoding would use, and for the
small recurrent model of this section it is far from tight. Its two Gated
DeltaNet layers have eight heads each, every head with a $4 \times 4$
state, so the two layers hold 256 numbers, or 8,192 bits at 32 bits per
number. We will train it on the associative recall task of
:numref:`sec_matrix-state`, in which a sequence stores up to 64 pairs, with
keys drawn from 128 possible ones and values from 4. Every query asks for a
key that was seen, so a table of 2 bits for each of the 128 possible keys,
256 bits in all, answers every query, and the 64 values alone need 128
bits. The state therefore has more than thirty times the bits that the task
needs. If bits were all that mattered, it would recall every pair. A matrix
memory, however, is a poor encoder of bits, and the reason is geometric.

### Overlapping Keys

A matrix memory writes by adding outer products $\mathbf{v}\mathbf{k}^\top$
and reads with a linear map, so the pairs it holds interfere. As
:numref:`sec_matrix-state` showed, a read with one unit key returns the
value stored under it plus the values of all other stored keys, each
weighted by the overlap of the two keys. For random unit keys of dimension
$d_k$ the mean squared overlap of two keys is $1/d_k$, and what such a
memory holds grows with the number of entries of its state, $d_k d_v$ for
each head.

The delta rule of :numref:`sec_deltanet` does not simply add. It moves the
memory toward the least-squares fit of the pairs, and for linearly
independent keys, of which a head holds at most $d_k$, that fit returns
every value exactly. With more keys than dimensions it cannot. The keys are
then linearly dependent: some key is a combination of the others, and a
linear read must return the same combination of their values, whatever value
was stored under that key. So both write rules let the pairs interfere once
the keys outnumber the dimensions, and the overlap of the keys says how
much. A trained model chooses its keys, however. Could it choose them so
that they hardly overlap?

Not when there are more keys than dimensions. A head of dimension $d_k$ has
room for only $d_k$ orthogonal keys, and the next proposition says how much
any larger set of keys must overlap, however it is chosen.

**Proposition.** Let $\mathbf{k}_1, \ldots, \mathbf{k}_m$ be unit vectors
in $\mathbb{R}^{d_k}$, with $m > d_k$. Then the mean of $(\mathbf{k}_i^\top
\mathbf{k}_j)^2$ over all pairs $i \neq j$ is at least

$$\frac{m - d_k}{d_k\,(m - 1)}.$$
:eqlabel:`eq_hy-welch`

**Proof.** Let $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top$.
Its trace is $\sum_i \|\mathbf{k}_i\|^2 = m$, so its $d_k$ eigenvalues sum
to $m$, and by the Cauchy–Schwarz inequality their squares sum to at least
$m^2/d_k$. The sum of the squared eigenvalues of the symmetric matrix
$\mathbf{G}$ is $\|\mathbf{G}\|_F^2 = \sum_{i,j} (\mathbf{k}_i^\top
\mathbf{k}_j)^2$, and the $m$ terms with $i = j$ equal one. The other $m(m -
1)$ terms therefore sum to at least $m^2/d_k - m$. $\blacksquare$

This is a bound of :citet:`Welch.1974`. It is instructive to put in the
sizes of our models. Sixty-four keys in a head of dimension 4 have a mean
squared overlap of at least $60/(4 \cdot 63) \approx 0.24$, while random
keys have $1/4$. Once the keys far outnumber the dimensions, the best keys
that training can find are hardly better than random ones. In a head of
dimension 16 the floor drops to $48/(16 \cdot 63) \approx 0.05$. The bound
grows toward $1/d_k$ as $m$ grows, so a head that stores more pairs than it
has dimensions reads each of them with interference from the others, and no
training removes it.

The bound is severe. With 63 other keys, each overlapping the one we read by
about 0.24 in square on average, an additive read in a head of dimension 4
picks up far more interference than signal. Yet a read need only pick the
right one of four values, not reproduce it, and the eight heads of a layer
each have keys of their own and a slice of every value. The bound does not
say how a trained network spreads its pairs over heads and layers, so it
predicts a trend, not a recall rate.

What should we expect, then, when we train such models? First, recall
should fall as the number of stored pairs grows, and a state with four
times the entries should fall later. :numref:`sec_matrix-state` found that
what gated linear attention holds roughly doubles when its state doubles.
If the delta rule behaves the same way, a model with four times the state
should recall 64 pairs about as well as the small one recalls 16. Second, an
attention layer keeps every pair it reads in its cache, so a model that has
one should not lose pairs at all, however many there are.

## Recall with One Attention Layer

### Four Models

The task is the multi-query associative recall of
:numref:`sec_matrix-state` with the sizes above: up to 64 pairs, keys
drawn from 128 and values from 4, so that guessing is right one time in
four. All four models are `d2l.SeqLM` with two blocks of width 32, and they
differ only in their mixers. Two use Gated DeltaNet in both blocks, either
with 8 heads of size 4, a state of $8 \cdot 4^2 = 128$ entries per layer, or
with 2 heads of size 16, four times the state at about the same parameter
count. The third uses attention in both blocks, and the fourth, the hybrid,
uses Gated DeltaNet and then attention.

The attention mixer is `d2l.AttentionMixer` of :numref:`sec_lstm`, placed
behind the same short convolution as the recurrent layers and without a
position encoding. We return to the convolution once the results are in.
In `recall_model` a string names the mixers, one letter per block. As in
:numref:`sec_matrix-state`, the decays of the recurrent layers start with
memories of 256 to 2,048 steps, longer than any sequence of the task.

```{.python .input #hybrids-four-models-2}
%%tab pytorch
def recall_model(layout, num_heads=8, conv_width=4, num_hiddens=32):
    """SeqLM whose mixers are given by letters: G(ated DeltaNet) with
    num_heads heads, A(ttention) with two; tokens: 4 values, 128 keys, and
    the filler."""
    make = {'G': lambda: d2l.GatedDeltaNet(
                num_hiddens, num_heads, conv_width=conv_width,
                timescales=(256, 2048)),
            'A': lambda: d2l.AttentionMixer(num_hiddens, num_heads=2,
                                            conv_width=4, rope=False)}
    return d2l.SeqLM(4 + 128 + 1, num_hiddens, [make[c]() for c in layout])
```

```{.python .input #hybrids-four-models-2}
%%tab jax
def recall_model(layout, num_heads=8, conv_width=4, num_hiddens=32):
    """SeqLM whose mixers are given by letters: G(ated DeltaNet) with
    num_heads heads, A(ttention) with two; tokens: 4 values, 128 keys, and
    the filler."""
    rngs = nnx.Rngs(d2l.get_key())  # A new initialization each call
    make = {'G': lambda: d2l.GatedDeltaNet(
                num_hiddens, num_heads, conv_width=conv_width,
                timescales=(256, 2048), rngs=rngs),
            'A': lambda: d2l.AttentionMixer(num_hiddens, num_heads=2,
                                            conv_width=4, rope=False,
                                            rngs=rngs)}
    return d2l.SeqLM(4 + 128 + 1, num_hiddens, [make[c]() for c in layout],
                     rngs=rngs)
```

### The Recall Sweep

`d2l.train_recall` trains each model as in :numref:`sec_matrix-state`,
for 2,000 updates on loads that cycle through 4, 8, 16, 32, and 64 pairs,
and then measures recall at each load on new sequences. Each model is
trained twice, from fresh weights on fresh data, and the table prints both
runs.

```{.python .input #hybrids-the-recall-sweep-2}
%%tab pytorch, jax
arms = {'Gated DeltaNet': dict(layout='GG'), 'attention': dict(layout='AA'),
        'hybrid': dict(layout='GA'),
        'Gated DeltaNet, 4x state': dict(layout='GG', num_heads=2)}
recall = {}
for name, arm in arms.items():
    models = [recall_model(**arm) for _ in range(2)]
    recall[name] = [d2l.train_recall(model, num_keys=128) for model in models]
    print(f'{name:25s} {d2l.num_params(models[0]):,} parameters')
```

```{.python .input #hybrids-the-recall-sweep-3}
%%tab pytorch, jax
pairs = list(recall['hybrid'][0])
runs = {name: np.array([[run[n] for n in pairs] for run in two])
        for name, two in recall.items()}  # Two rows, one per run
d2l.plot(pairs, [r.mean(0) for r in runs.values()], 'pairs in the sequence',
         'recall, mean of two runs', legend=list(runs), xscale='log',
         ylim=(0, 1.05), fmts=('-o', '-s', '--x', '-^'), figsize=(4.5, 3),
         legend_loc='lower left')
d2l.plt.xticks(pairs, pairs)  # Ticks at the numbers of pairs
print(f'{"pairs":25s}', ''.join(f'{n:>11d}' for n in pairs))
for name, r in runs.items():
    print(f'{name:25s}', ''.join(f'{a:6.2f}/{b:.2f}' for a, b in r.T))
```

Both predictions hold, and the first with room to spare. With 4 pairs
every model recalls nearly every value. From there Gated DeltaNet loses
pairs as the load grows, to two thirds or a little less of 32 pairs and
about half of 64. The model with four times the state does better than we
guessed: it recalls 0.93 to 0.97 of 64 pairs, about as well as the small
state recalls 8, not 16, and every one of its runs is above every run of
the small state from 16 pairs on. Four times the entries hold at least
four times the pairs, so what :numref:`sec_matrix-state` found for gated
linear attention holds for the delta rule as well. The two runs of each
model differ far less than the models do.

The hybrid recalls every pair at every load in every run, and so does
attention behind the same convolution. Attention was built to keep every
pair, so its row is no surprise. The hybrid's row is the result that
matters: one attention layer in place of the second recurrent layer removes
the fall altogether. That layer can look up every pair, so what the
recurrent layer loses no longer matters. At these lengths, however, the
attention layer is not cheap. At the last query of a sequence with 64 pairs
its cache holds the keys and values of 192 tokens, 12,288 numbers, 24 times
the larger recurrent state. A little attention saves memory only when the
context is long and most layers are recurrent, which is the count we make at
the end of this section.

These models are small on purpose. A matrix memory of 256 numbers is
overloaded by 64 pairs, while a large model has about a million numbers of
state in each layer and meets its limit only at far longer contexts. The
experiment therefore shows the mechanism, not the size of the effect in a
real model.
That evidence comes from :citet:`Waleffe.Byeon.Riach.ea.2024`, who trained
transformers, Mamba models, and hybrids with billions of parameters on the
same data: the pure Mamba models fell behind the transformer on tasks that
copy from the context or look things up in it, and a hybrid with a small
fraction of attention layers matched or exceeded the transformer.

### The Short Convolution

Every mixer in the sweep reads its input through a short causal
convolution, and the attention layer was given one deliberately. A value
arrives one token after its key, but a matrix memory writes one key and
one value per position. Without a view of the token before it, the
position of a value does not know which key it belongs to. A convolution of
width 4 lets each position draw on the three tokens before it, so one write
can store a key together with its value. Attention gains the same thing:
the convolution brings each key into the position of the value that follows
it, a query then finds that position by content, and the lookup needs no
position encoding. To see how much this matters, we train the Gated
DeltaNet model with a convolution of width 1, which only scales and shifts
each channel and mixes nothing across steps.

```{.python .input #hybrids-the-recall-sweep-4}
%%tab pytorch, jax
no_conv = [d2l.train_recall(recall_model('GG', conv_width=1), num_keys=128)
           for _ in range(2)]  # Width 1: only scales and shifts each channel
print(f'{"pairs":25s}', ''.join(f'{n:>11d}' for n in pairs))
for name, (a, b) in [('Gated DeltaNet', recall['Gated DeltaNet']),
                     ('without the convolution', no_conv)]:
    print(f'{name:25s}', ''.join(f'{a[n]:6.2f}/{b[n]:.2f}' for n in pairs))
```

Every run without the convolution recalls fewer pairs than every run with
it, at every load, by 0.06 to 0.38, and the difference is largest at 8 to
16 pairs. Even 4 pairs, which fit easily into the state, are no longer
recalled reliably. The model still does better than guessing at every
load, so its two layers find some other way to bring a value together with
its key, but not one as good as a convolution over four tokens. A memory
can hold only what reaches it.

## Language Modeling at Equal Size

Does the difference show on ordinary text? In :numref:`sec_lstm` and
:numref:`sec_mamba`, a stack of attention layers ended behind every
recurrent stack of the same width and depth. The recall sweep suggests
where attention might win instead: if predicting a byte requires a pair
stored long before, attention and the hybrid should gain at the late
positions of a window, where more has been stored. We therefore use the
text benchmark of :numref:`sec_lstm`, byte-level WikiText-103 in windows of
128 bytes, and look at the loss by position in the window. Each model
trains for 1,000 steps of `d2l.train_text` on 64 windows per step, and
`d2l.text_table` reports the validation loss in nats per byte over the
whole window and at bytes 0 to 15, 16 to 63, and 64 to 127.

All three models are `d2l.SeqLM` with four blocks of width 256, and their
parameter counts lie within a tenth of each other. One has Gated DeltaNet
in every block, with 4 heads of size 64, and the hybrid replaces the
fourth of these by attention. The third is the attention model of
:numref:`sec_lstm`, with 8 heads and rotary positions in every block.
These attention layers, the hybrid's included, have no short convolution:
they are the layers that the earlier sections trained on this benchmark,
not those of the recall sweep.

```{.python .input #hybrids-language-modeling-3}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
```

```{.python .input #hybrids-language-modeling-4}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32
text_mixers = {'G': lambda d: d2l.GatedDeltaNet(d, num_heads=4),
               'A': lambda d: d2l.AttentionMixer(d, num_heads=8)}

def text_model(layout, d):  # 256 byte values; one block per letter
    return d2l.SeqLM(256, d, [text_mixers[c](d) for c in layout]).to(
        d2l.try_gpu())
```

```{.python .input #hybrids-language-modeling-4}
%%tab jax
text_mixers = {
    'G': lambda d, rngs: d2l.GatedDeltaNet(d, num_heads=4, rngs=rngs),
    'A': lambda d, rngs: d2l.AttentionMixer(d, num_heads=8, rngs=rngs)}

def text_model(layout, d):  # 256 byte values; one block per letter
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.SeqLM(256, d, [text_mixers[c](d, rngs) for c in layout],
                     rngs=rngs)
```

```{.python .input #hybrids-language-modeling-5}
%%tab pytorch, jax
arms = {'Gated DeltaNet': ('GGGG', 256), 'hybrid': ('GGGA', 256),
        'attention': ('AAAA', 256)}  # Width 256, four blocks each
trained = d2l.text_table(text_model, arms, data)
```

Gated DeltaNet ends 0.05 to 0.06 nats per byte below attention over the
whole window. The gap is smallest over the first 16 bytes, about 0.04, and
0.05 to 0.06 from byte 16 to the end, far more than the two runs of either
model differ. Attention does not catch up at the late positions, where the
prediction said it should. The hybrid lands between the two and much
nearer to Gated DeltaNet: every hybrid run ends behind every Gated DeltaNet
run, in every band, but by about 0.01 or less, a small fraction of
attention's gap. On this benchmark, trading one recurrent layer in four for
attention raised the loss a little in every run. Nor do the late positions
help any model much: every model's loss falls by at most 0.004 from bytes
16 to 63 to bytes 64 to 127. On these windows the next byte depends on the
bytes just before it, and almost never on a pair stored a hundred bytes
earlier. With larger models and longer contexts,
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` traced most of the gap in
perplexity between attention and the gated convolution models they studied
to recall of this kind. On our windows of 128 bytes there is no such gap to
explain; here attention is the model that trails.

Gated DeltaNet, the model that recalls the fewest pairs, thus has the
lowest loss on text of the three. A low average loss on short windows is
no evidence that a model can look things up. Where text does repeat
itself, as with a name or a number seen earlier in the window, the
difference should appear, and an exercise measures it there.

At this length the recurrent model does not train faster either. A Gated
DeltaNet step takes about two to two and a half times as long as an
attention step (the last column of the table), and the hybrid lies between
them. The reason is most likely the number of separate GPU operations, not
the amount of arithmetic: at 128 tokens the quadratic work of attention is
small and runs in one fused kernel, while the chunked delta rule runs a
sequence of small matrix products and a triangular solve. The advantages of
a fixed state lie elsewhere: at long contexts, where the work of attention
per token grows with the length (:numref:`subsec_linear-recurrence-cost`),
and above all in the memory it needs while it generates.

## Inference Memory

During generation a model keeps, for every sequence it is serving, what it
needs to compute the next token. This memory per sequence decides how many
sequences a GPU can serve at once, and because generating a token reads all
of it from memory, it also sets how fast each token is generated. For an
attention layer it is the key–value cache of :numref:`sec_kv-cache`: one
key and one value per past token for each of the $n_\textrm{kv}$ key–value
heads of dimension $d_\textrm{head}$, that is,

$$2 \cdot n_\textrm{kv} \cdot d_\textrm{head} \cdot (\textrm{bytes per number})$$

bytes per token and layer. With grouped-query attention, 8 key–value heads
of dimension 128 in 16-bit numbers, that is 4 KB per token. A recurrent
layer keeps its state, whose size does not depend on the length: a Mamba-2
layer in a model of width 4,096 has 8,192 channels with 128 states each,
about one million numbers, or 2 MB. The attention layer's cache overtakes
this state after about 500 tokens, and at a context of 128K tokens it holds
512 MB, 256 times as much.

Only the attention layers of a hybrid have caches that grow, so its memory
per sequence is the transformer's times the fraction of layers that are
attention, plus a constant. :numref:`sec_windowed-linear-attention` did the
same sum for windowed attention layers interleaved with global ones; here a
state of fixed size takes the place of each window's cache.
:numref:`fig_hy-cache` shows the effect for a model of 32 layers.

Released hybrids bear this out. Jamba :cite:`Lieber.Lenz.Bata.ea.2024`,
with 4 attention layers among 32, reports a cache of 4 GB at a context of
256,000 tokens, against 32 GB for a transformer of similar size. Later
hybrids build their recurrent layers on the delta rule of
:numref:`sec_deltanet`: Qwen3-Next :cite:`Qwen.Team.2025` interleaves
Gated DeltaNet layers with full attention layers, three to one, and Kimi
Linear :cite:`Kimi.Team.2025b` does the same with a variant of Gated
DeltaNet whose decay acts on each key coordinate.

![Memory held per sequence during generation, against context length, for a 32-layer model with grouped-query attention (8 key–value heads of dimension 128) and Mamba-2-sized recurrent states, in 16-bit numbers. A transformer's cache reaches 32 GB at 256K tokens; a purely recurrent model stays at 64 MB at every length; a hybrid with 4 attention layers holds an eighth of the transformer's cache.](../img/mdl-modernrnn-hybrid-cache.svg)
:label:`fig_hy-cache`

## The Chapter's Memories Compared
:label:`subsec_hy-memories`

:numref:`tab_hy-memories` sets the memories of this chapter side by side,
with attention and the hybrid for comparison. For each it gives what a
layer keeps while it generates, how it is trained, what the synthetic task
of its section showed, and its loss on the text benchmark of
:numref:`sec_lstm`, where every model has four blocks of width $d = 256$.
The table writes $N$ for the states per channel of S4D and Mamba,
$H = 2d$ for the channels of a Mamba layer, $h$ for the number of heads,
and $T$ for the number of tokens read so far. Losses from different
sections that differ by less than about 0.02 are within the spread between
executions of the same model.

:The memories of this chapter. Text: validation loss in nats per byte on byte-level WikiText-103, windows of 128 bytes, two runs per model; parameters in parentheses.
:label:`tab_hy-memories`

| Memory | State per layer at generation | Training | Synthetic task | Text |
| :-- | :-- | :-- | :-- | :-- |
| LSTM, GRU | $2d$, $d$ | one step after another | gates that start near one keep a symbol for 200 steps; started at one half, they lose it by 50 | 1.29 to 1.30 (4.3 M, 3.8 M) |
| minGRU | $d$ | scan | at chance after 100 steps; with the timescales of S4D, recalls after 1,000 | 1.34 to 1.35 (2.8 M) |
| S4D | $dN$ | convolution or scan | recalls after 1,000 steps; at most one in five on selective copying | not trained |
| Mamba | $HN$ | scan | 0.99 to 1.00 on selective copying | 1.30 (4.0 M) |
| gated linear attention | $d^2/h$ | chunks | about half of 64 pairs with 128 entries, more with 512 | not trained |
| Gated DeltaNet | $d^2/h$ | chunks with a triangular solve | 0.95 or more of rewritten keys, against about 0.83 to 0.92 for gated linear attention | 1.32 to 1.33 (3.6 M) |
| attention | $2dT$ | fused attention over all positions | every pair at every load | 1.37 to 1.39 (3.3 M) |
| hybrid | recurrent states plus $2dT$ per attention layer | both | every pair, with attention in one of two blocks | 1.33 to 1.34 (3.5 M) |

Read down the last column, and the chapter's newest layers do not win: on
windows of 128 bytes no layer beats the LSTM of 1997, Mamba is level with
it, and Gated DeltaNet is about 0.03 behind. Read across the rows, and the
reason for the newer layers appears. They train in parallel, by scans or by
chunks of matrix products, and the matrix states of gated linear attention
and Gated DeltaNet hold pairs that can be looked up and, with the delta
rule, rewritten. A benchmark of short windows tests neither.

Which memory, then, should one use? For a model that generates long texts,
the answer of practice is a hybrid: delta-rule layers for most of the
depth, with one attention layer in every four to eight, so that the cache
is a fraction of a transformer's and recall stays close to a transformer's.
For a small model that processes a stream, such as speech or sensor
readings, where nothing from the distant past has to be reproduced exactly,
a gated recurrence still serves well, and its state is all the memory it
needs. What nobody knows yet is how few attention layers are enough, where
in the stack they belong, and whether attention over a window of recent
tokens (:numref:`sec_windowed-linear-attention`) can replace some of them.
Released models answer these questions differently
(:numref:`sec_ssm-further-reading`).

## Summary

A state of fixed size holds a fixed number of bits, so a recurrent model
can recall only as many items from its context as its state can store,
while an attention layer keeps every token and can look back. A matrix
memory falls far short even of this count. By Welch's bound, more keys than
dimensions must overlap however they are chosen, and overlapping keys make
the stored pairs interfere, under additive writes and under the delta rule
alike. On multi-query associative recall, a small Gated DeltaNet model
recalls about half of 64 pairs, and four times the entries hold at least
four times the pairs. A hybrid with attention in place of the second
recurrent layer recalls every pair at every load, as attention does.
Without the short convolution that brings each key to its value, the
recurrent model recalls fewer pairs at every load. On windows of 128 bytes
of text none of this shows: Gated DeltaNet is ahead of attention, and the
hybrid trails it by about 0.01 or less. During generation only the
attention layers keep a cache that grows with the context, so a hybrid
needs the transformer's cache times its fraction of attention layers, an
eighth for one attention layer in eight, plus a constant. Most recent
large models with recurrent layers are hybrids built this way.

## Exercises

1. **The counting bound.** A model reads a string of $k$ tokens from a
   vocabulary of $V$ and must then reproduce it.
    1. Prove :eqref:`eq_hy-counting` for a model whose state between reading
       and writing has $M$ numbers of $b$ bits and whose output is a
       deterministic function of that state.
    1. Suppose instead that the model may output a wrong string with
       probability at most $\epsilon$ when the string is drawn uniformly.
       Show that $Mb \ge k \log_2 V + \log_2(1 - \epsilon)$. (Hint: count
       the strings that each state can decode correctly.)
    1. How many tokens of a vocabulary of 50,000 can a model with 48 layers
       of Mamba-2 state, 2 MB each in 16-bit numbers, copy at most? Why does
       no model of this size come close in practice?

1. **Memory for many users.** A GPU with 80 GB serves a model of 32
   layers, with the cache sizes of :numref:`fig_hy-cache`, after 16 GB of
   the GPU's memory are taken by the weights.
    1. How many sequences of 128K tokens can the GPU serve at once with the
       transformer, the hybrid with 4 attention layers, and the recurrent
       model?
    1. Grouped-query attention with fewer key–value heads, and 8-bit
       numbers for the cache, each halve the cache. Apply both to the
       transformer and to the hybrid, and repeat the count. Which of the two
       changes could also shrink the hybrid's recurrent states, and what
       share of the hybrid's memory do those states take after both
       changes?

1. **Overlap and training.**
    1. How much better than random keys can training make the keys of a
       head? Divide the bound :eqref:`eq_hy-welch` by $1/d_k$, the mean
       squared overlap of random unit keys, and evaluate the ratio for 64
       keys in dimensions 4, 8, 16, and 32. In which of these heads can
       trained keys be clearly better than random ones?
    1. [code] Train the small Gated DeltaNet model of the sweep with 4,000
       and 8,000 updates (`num_updates` of `d2l.train_recall`). Does its
       recall of 32 and 64 pairs approach the hybrid's? What does the
       answer say about whether the small model lacks training or
       capacity?
    1. [code] Draw the keys from 64 and from 512 possible ones (`num_keys`,
       with the vocabulary of `recall_model` changed to match). A sequence
       still stores at most 64 pairs. Why can the number of possible keys
       change recall at all?

1. [code] **Where and how much attention?**
    1. Train the hybrid `AG`, with attention in the first block, and
       compare its recall with that of `GA`. Does the position of the
       attention layer matter? Consider which layer can bring each key
       together with the value that follows it.
    1. Train four-block models `GGGG`, `GGGA`, and `GAGA` of the same
       width. Do two more recurrent layers move the recall curve as far as
       four times the state per layer did, and does one attention layer
       among four still recall every pair?

1. [code] **A window instead.** Replace the attention mixer by attention
   that sees only the last $w$ tokens, by masking the scores of earlier
   tokens.
    1. Train the hybrid with $w = 32$ and $w = 128$ and compare recall at
       every number of pairs. Which queries can a window of 32 answer, and
       which must the recurrent layers answer?
    1. How large is the cache of this layer during generation? What does
       the comparison suggest about combining windows with recurrence?

1. [code] **Recall in text.** In the validation windows of byte-level
   WikiText-103, mark the positions whose byte completes a string of at
   least eight bytes that already occurred earlier in the same window.
    1. Compare the loss of the trained Gated DeltaNet, hybrid, and
       attention models (`trained`) at the marked positions and at all
       other positions. Does attention's advantage on recall show up where
       text repeats itself?
    1. Repeat with windows of 512 bytes (`num_steps=512`, with fewer
       windows per step). How does the share of marked positions change,
       and with it the difference between the models?

1. [code] **Rewritten keys and attention.** Train the four models of the
   sweep on the task in which some keys are written twice, so that the
   answer is the newer of two values
   (`pairs=(16, 32, 64), num_keys=128, num_rebound=8, gap=2` in
   `d2l.train_recall`, since the default loads start at 4 pairs, too few
   to rewrite 8 keys). The third array returned by
   `d2l.associative_recall` marks the queries of rewritten keys; score
   them as `recall_by_kind` in :numref:`sec_deltanet` does.
    1. Attention keeps both values in its cache, and our attention mixer
       has no position encoding. How could it tell the newer value from the
       older? Measure its recall of rewritten keys.
    1. Compare attention, the hybrid, and Gated DeltaNet on rewritten keys.
       Which of the two memories of the hybrid answers these queries?

[Discussions](https://d2l.discourse.group/t/36785)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.8]{.kicker}

Hybrid architectures<br>
**counting bits · overlapping keys · associative recall · one attention layer · memory per sequence · the chapter's memories**
:::
:::

::: {.slide title="A Fixed State Stores What It Chose Before the Question"}
[The problem]{.kicker}

A prompt lists 64 product codes and prices, then asks for 20 of them.

- A transformer looks back: every token is still in its cache.
- A recurrent model has only what it stored, before it knew the question.

::: {.d2l-note}
Better writes do not make a state hold more than its size allows.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- bound the recall of a fixed state by counting bits;
- predict where a small state loses pairs;
- measure one attention layer on recall and on text;
- compute memory per sequence, and choose a memory.

**Running example:** a Mamba-2 layer, 2 MB of state, against a
grouped-query cache of 4 KB per token.
:::

::: {.slide title="The Last Step of the Chapter Puts Back a Little Attention"}
[Where we are]{.kicker}

Gates → linear recurrences and scans → state space models → selectivity →
matrix memories → the delta rule → **hybrids**.

- Here: what no fixed state can do, and the cheapest repair.
- Not here: hybrids at scale, windowed attention, GPU kernels.
:::

::: {.slide title="Recalling k Tokens Needs k log₂ V Bits"}
[Counting bits]{.kicker}

$V^k$ possible inputs must reach $V^k$ different states:

$$M b \ge k \log_2 V$$

$M$ numbers of $b$ bits; $k$ tokens from a vocabulary of $V$.

- Recall grows with the state, not with the context.
- A KV cache grows by one key and one value per token: it never runs out.
:::

::: {.slide title="The Small Recall Model Has 30× the Bits That 64 Pairs Need"}
[Counting bits]{.kicker}

Recall task below: up to 64 pairs, keys from 128, values from 4.
Model: two Gated DeltaNet layers, 8 heads of $4 \times 4$ state each.

| | bits |
|---|---|
| state: 2 layers × 8 heads × 4 × 4 numbers × 32 bits | 8,192 |
| a table of 2 bits per possible key: 128 × 2 | 256 |
| the 64 values alone: 64 × 2 | 128 |

::: {.d2l-note}
Bits are not the limit: a matrix memory is a poor encoder of bits.
:::
:::

::: {.slide title="Pairs in a Matrix Memory Interfere"}
[Overlapping keys]{.kicker}

Write $\mathbf{S} \leftarrow \mathbf{S} + \mathbf{v}\mathbf{k}^\top$; read with unit key $\mathbf{k}_j$:

$$\mathbf{S}\mathbf{k}_j = \mathbf{v}_j + \sum_{i \neq j}
(\mathbf{k}_i^\top \mathbf{k}_j)\, \mathbf{v}_i$$

- Random unit keys of dimension $d_k$: mean squared overlap $1/d_k$.
- What the memory holds grows with its entries: $d_k d_v$ per head, keys
  of size $d_k$, values of size $d_v$.
- Delta rule: fits the pairs, but $m > d_k$ keys are linearly dependent,
  so no linear read returns every value.
- Can training pick keys that hardly overlap?
:::

::: {.slide title="More Keys Than Dimensions Must Overlap"}
[Welch's bound]{.kicker}

Unit $\mathbf{k}_1, \ldots, \mathbf{k}_m \in \mathbb{R}^{d_k}$, $m > d_k$:

$$\operatorname{mean}_{i \neq j} (\mathbf{k}_i^\top \mathbf{k}_j)^2 \ge
\frac{m - d_k}{d_k\,(m - 1)}$$

Proof: $\mathbf{G} = \sum_i \mathbf{k}_i\mathbf{k}_i^\top$ has trace $m$
and rank $\le d_k$, so $\|\mathbf{G}\|_F^2 \ge m^2/d_k$.

| 64 keys | floor | random |
|---|---|---|
| $d_k = 4$ | 0.24 | 0.25 |
| $d_k = 16$ | 0.05 | 0.06 |

::: {.d2l-note}
Once keys far outnumber dimensions, the best keys are hardly better than
random ones.
:::
:::

::: {.slide title="Check: Predict Before We Train"}
[Check]{.kicker}

1. The state grows 4× at the same parameter count. Where should recall
   of 64 pairs land?
2. One of two layers becomes attention. What happens to recall?

::: {.fragment .fade-up}
1. About where the small state recalls 16 pairs, if pairs grow in
   proportion to entries, as they did for gated linear attention.
2. No fall: the cache holds every pair it has read.
:::
:::

::: {.slide title="Multi-Query Associative Recall"}
[The task]{.kicker}

`k₁ v₁  k₂ v₂  …  kₙ vₙ  |  k₂ kₙ k₁ …`

- $n$ distinct keys from 128, each followed by a value from 4.
- Then every key again, in random order: answer the value that followed.
- $n$ from 4 to 64. Guessing: 1 in 4.
:::

::: {.slide title="Four Models Differ Only in Their Mixers"}
[Four models]{.kicker}

| model | mixers | state per recurrent layer |
|---|---|---|
| Gated DeltaNet | G G, 8 heads of 4 | 128 entries |
| Gated DeltaNet, 4× state | G G, 2 heads of 16 | 512 entries |
| attention | A A | cache of every pair |
| **hybrid** | G **A** | 128 entries + cache |

Two blocks of width 32, 35,000 to 38,000 parameters; the same short
convolution before every mixer; attention without a position encoding;
each trained twice.
:::

::: {.slide title="One Attention Layer Removes the Fall; 4× State Moves It Right"}
@!hybrids-the-recall-sweep-3
:::

::: {.slide title="4× State Recalls 64 Pairs as Well as the Small State Recalls 8"}
[Recall]{.kicker}

- Gated DeltaNet: two thirds or a little less of 32 pairs, half of 64.
- 4× state: 0.93–0.97 of 64, like the small state at 8 (0.93–0.96),
  better than the guess of 16: at least 4× the pairs.
- Hybrid, attention: every pair, every load, every run.
- At 64 pairs the attention cache holds 192 tokens, 12,288 numbers: 24×
  the larger state. A little attention saves memory only at long
  contexts, among mostly recurrent layers.

::: {.d2l-note .warn}
Tiny on purpose (256 numbers): the mechanism, not the effect at scale.
:::
:::

::: {.slide title="Without the Short Convolution, Every Run Recalls Less"}
[The convolution]{.kicker}

@!hybrids-the-recall-sweep-4

A value comes one token after its key; a convolution of width 4 lets one
write store both. Width 1 only scales and shifts each channel.

::: {.d2l-note}
A memory can hold only what reaches it.
:::
:::

::: {.slide title="On Text, Attention Is Behind; the Hybrid Trails Slightly"}
[Text]{.kicker}

@!hybrids-language-modeling-5

Byte-level WikiText-103, windows of 128 bytes; four blocks of width 256;
loss in nats per byte, overall and by position in the window; last column:
time per training step. Attention here: 8 heads, rotary positions, no
convolution.
:::

::: {.slide title="The Model That Recalls Least Has the Lowest Text Loss"}
[Text]{.kicker}

| mixers | recall of 64 pairs | text, nats per byte |
|---|---|---|
| Gated DeltaNet | about half | 1.32 to 1.33 |
| hybrid | all | 1.33 to 1.34 |
| attention | all | 1.37 to 1.39 |

::: {.d2l-note .warn}
A low loss on short windows is no evidence that a model can look things
up.
:::
:::

::: {.slide title="Check: Why Does No Model Gain Much After Byte 16?"}
[Check]{.kicker}

Every model's loss falls by at most 0.004 from bytes 16–63 to bytes
64–127, and attention stays 0.05 to 0.06 behind Gated DeltaNet. What does
that say about this benchmark?

::: {.fragment .fade-up}
The next byte depends on the bytes just before it. A pair stored a hundred
bytes back rarely matters, so the benchmark does not test recall.
:::
:::

::: {.slide title="Attention Stores 4 KB per Token, Mamba-2 About 2 MB in All"}
[Inference memory]{.kicker}

Per layer, per sequence, 16-bit numbers:

| layer | memory |
|---|---|
| GQA, 8 KV heads × 128 | $2 \cdot 8 \cdot 128 \cdot 2$ B = 4 KB per token |
| Mamba-2, width 4,096 | $8192 \cdot 128 \cdot 2$ B ≈ 2 MB, any length |

Crossover near 500 tokens; at 128K tokens, 512 MB against 2 MB.
:::

::: {.slide title="Only the Attention Layers of a Hybrid Keep a Cache"}
[Inference memory]{.kicker}

![](../img/mdl-modernrnn-hybrid-cache.svg){width=80%}

Jamba, 4 attention layers of 32: 4 GB at 256K tokens, against 32 GB.
Qwen3-Next: Gated DeltaNet and attention, three to one; Kimi Linear: a
variant that decays each key coordinate, also three to one.
:::

::: {.slide title="Check: Memory of a Hybrid"}
[Check]{.kicker}

32 layers, 4 of them attention, 128K tokens, numbers as above.

1. Memory per sequence of the hybrid?
2. Of the transformer?

::: {.fragment .fade-up}
1. $4 \cdot 512$ MB $+ 28 \cdot 2$ MB $\approx 2.1$ GB.
2. $32 \cdot 512$ MB $= 16$ GB: eight times as much.
:::
:::

::: {.slide title="The Chapter's Memories Side by Side"}
[The chapter]{.kicker}

| memory | state per layer | training | text |
|---|---|---|---|
| LSTM, GRU | $2d$, $d$ | step by step | 1.29–1.30 |
| minGRU | $d$ | scan | 1.34–1.35 |
| S4D | $dN$ | convolution or scan | |
| Mamba | $HN$ | scan | 1.30 |
| gated linear attention | $d^2/h$ | chunks | |
| Gated DeltaNet | $d^2/h$ | chunks + solve | 1.32–1.33 |
| attention | $2dT$ | fused, all positions | 1.37–1.39 |
| hybrid | states + $2dT$ per attention layer | both | 1.33–1.34 |

Text: nats per byte, width $d = 256$; $N$ states per channel, $H = 2d$
Mamba channels, $h$ heads, $T$ tokens read.
:::

::: {.slide title="On Short Windows No Layer Beats the 1997 LSTM"}
[The chapter]{.kicker}

- Newer layers train in parallel and store pairs to look up or rewrite;
  short windows test neither.
- Long-context generation: delta-rule layers, one attention layer in four
  to eight.
- A small streaming model: a gated recurrence.

**Open:** how few attention layers, where, and whether windows suffice.
:::

::: {.slide title="Recap"}
- Recall $\le Mb/\log_2 V$ tokens; a matrix memory far less, since more
  than $d_k$ keys must overlap.
- 64 pairs: Gated DeltaNet half; 4× state 0.93–0.97; one attention
  layer, all.
- 128-byte text: attention 0.05–0.06 behind Gated DeltaNet; hybrid about
  0.01 behind, in every run.
- 128K tokens, 32 layers: 16 GB of cache; 2.1 GB with 4 attention layers.

Next: what models cost on real hardware, and how to make them fast.
:::
