# Hybrid Architectures
:label:`sec_hybrids`

Every model in this chapter improved how a fixed-size state is written,
from gates to selective dynamics to the delta rule, but none of them can
make the state hold more than its size allows. Does that size limit what a
model can recall from its context? Consider a user who pastes a table of 64
product codes and prices into a prompt and asks for the prices of 20 of
them. A transformer answers such a request by looking back: every earlier
token is still in its cache, and attention finds the ones that match. A
model with a fixed-size state has only what it chose to store when the
tokens went by, before it knew which of them would be asked for. Retrieval
from the context is a large part of how language models are used, in
few-shot prompts, in answers grounded in retrieved documents, and in
editing code, where a programmer may ask for a 2,000-token function to be
copied with one line changed. This limit therefore matters in practice.

This section measures the limit and shows the remedy that most recent large
recurrent models adopt. We first count how much a fixed state can hold. We
then train four small models on a recall task with a growing number of
pairs: two built from the recurrent layers of the previous sections, one
built from attention, and a *hybrid* in which one of its two layers is
attention. We ask whether this one layer is enough to recall as well as a
transformer, and what the hybrid saves in memory, by counting that memory
for models of realistic size.

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

The simplest limit comes from counting: a state cannot tell apart more
inputs than it has configurations. A state of $N$ numbers stored with $b$
bits each can be in at most $2^{Nb}$ different configurations. Suppose a model reads $k$ tokens, each
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

<!-- WRITER: the run below now draws keys from 128, not 512 (as :numref:`sec_matrix-state`). HY-2 then reads:
recording which 64 of 128 keys were seen takes log2 C(128, 64) = 124 bits, plus 2 bits for each of the 64 values:
about 250 bits in all (with 512 keys it would be 402). The recurrent models still have 2 layers x 8 heads x 4 x 4 =
256 numbers of 32 bits = 8,192 bits, so "many times" stays true. The paragraph "The task below draws its keys from
512 possible ones, and the recurrent models have 256 numbers of state ..." is false in its numbers and goes with the
slots. -->
The bound counts bits that a perfect encoding would use, and it is far from
tight for the models below. Their two recurrent layers hold 128 numbers
each, eight heads with a state of $4 \times 4$, so 256 numbers of 32 bits in
all, many times the 128 bits that 64 values of 2 bits take. A matrix memory
cannot pack bits like that: it writes by adding outer products and reads
with a linear map, and the pairs it holds interfere.
:numref:`sec_matrix-state` showed that after $n$ writes with random unit
keys of dimension $d_k$, the expected squared read error is $(n-1)/d_k$. A
trained model chooses its keys and values, however, and it can do much
better than random ones.

### Slots and Overlapping Keys

How much better? Suppose that a value needs one number, as a value from a
small set does. A head with a $d_v \times d_k$ state can then give each key
a slot of its own: the key's value goes into one row of the state, along a
key direction orthogonal to those of the other keys assigned to that row.
Reading with the key returns its value and nothing else. A row has room for
$d_k$ orthogonal directions, so a state of $N$ numbers has $N$ slots, and a
model whose task draws keys from at most $N$ possible ones can store any
pairs without interference. With more possible keys than slots, some row
must host more keys than it has dimensions, and these keys overlap,
whatever training does. The next proposition says by how much.

**Proposition.** Let $\mathbf{k}_1, \ldots, \mathbf{k}_m$ be unit vectors
in $\mathbb{R}^d$, with $m > d$. Then the mean of $(\mathbf{k}_i^\top
\mathbf{k}_j)^2$ over all pairs $i \neq j$ is at least

$$\frac{m - d}{d\,(m - 1)}.$$
:eqlabel:`eq_hy-welch`

**Proof.** Let $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top$.
Its trace is $\sum_i \|\mathbf{k}_i\|^2 = m$, so its $d$ eigenvalues sum to
$m$, and by the Cauchy–Schwarz inequality their squares sum to at least
$m^2/d$. The sum of the squared eigenvalues of the symmetric matrix
$\mathbf{G}$ is $\|\mathbf{G}\|_F^2 = \sum_{i,j} (\mathbf{k}_i^\top
\mathbf{k}_j)^2$, and the $m$ terms with $i = j$ equal one. The other $m(m -
1)$ terms therefore sum to at least $m^2/d - m$. $\blacksquare$

This is a bound of :citet:`Welch.1974`. For $m \gg d$ it approaches $1/d$,
the mean squared overlap of random unit keys in the proof of
:eqref:`eq_ms-capacity-law`. When a row hosts $m > d$ possible keys, a read
of one of them picks up the values of the other stored keys of that row,
as in :eqref:`eq_ms-retrieval-error`, with weights whose mean square is at
least the bound. The more pairs a sequence stores, the more stored keys
share a row, and the larger the interference.

The task below draws its keys from 512 possible ones, and the recurrent
models have 256 numbers of state, so on average two possible keys share each
slot. With a few pairs, the stored keys rarely meet in a row; with 64,
many of them do. We therefore expect the recurrent models to recall nearly
every pair when there are few, and to lose more and more of them as the
number of pairs grows, since no training removes the overlap. Attention has
no such limit, because its cache holds every pair it has read.

### Associative Recall

To measure recall we use *multi-query associative recall*
:cite:`Arora.Eyuboglu.Timalsina.ea.2024`. A sequence begins with $n$ pairs,
each a key token followed by a value token, for distinct keys drawn from
512 and values drawn from 4. Filler tokens follow, and the sequence ends
with every key once more, in random order, as a query. At each query, the
model must output the value that followed that key. Sequences have 256
tokens, so they hold up to 64 pairs with their queries.
:citet:`Arora.Eyuboglu.Timalsina.ea.2024` found that this kind of recall
accounts for most of the gap in perplexity on natural text between attention
and the gated convolution models they studied. The 512 keys exceed the 256
slots of the recurrent models; the 4 values let one number hold a value.

<!-- WRITER: the task and its harness now live in :numref:`sec_matrix-state` (`d2l.associative_recall`,
`d2l.train_recall`, saved there; the old cell `hybrids-associative-recall-1` moved with its id). Restate the task
here in one sentence with that pointer. Facts of this section's run that differ from the paragraph above: keys
are drawn from 128, as in :numref:`sec_matrix-state` (`num_keys=128`; with 512 keys the run-to-run spread of the
larger-state arm was twice as wide, up to 0.18 at 64 pairs, see the results note), values from 4; every sequence is as short as its pairs and queries allow (a multiple of 32 tokens, at most 192, no block
of filler), so "Filler tokens follow" and "Sequences have 256 tokens" are false. The "slots" sentence goes with
the slot construction (structure, HY-3). -->

## Recall with One Attention Layer

### Four Models

To see whether a single attention layer gives a recurrent model the recall
of a transformer, we compare a hybrid with models built from one kind of
layer. All four models are instances of `d2l.SeqLM` with two blocks of
width 32; only the mixers differ. Two use the recurrent layers of the
previous sections, gated linear attention (:numref:`sec_matrix-state`) and
Gated DeltaNet (:numref:`sec_deltanet`), with eight heads, so that keys and
values have 4 dimensions. The third uses causal softmax attention in both
blocks, and the fourth, the hybrid, replaces the second of its two Gated
DeltaNet layers by attention. The attention mixer is computed like the
multi-head attention of :numref:`chap_attention`, with the fused attention
function of each framework, behind the same short convolution as the
recurrent layers, so that the mixers differ only in their memory. The
attention mixer has no position encoding of its own, like the NoPE layers
of :numref:`sec_positional-information`, and the convolution gives it all
the order it needs here: it brings each key into the position of the value
that follows it, and a query then finds that position by content. Without
it, two layers of attention with rotary position embeddings
(:numref:`subsec_pos-rope`) must form an *induction head*, the first
copying each key into the position after it and the second looking the
query up there. They learn this too, but only after a plateau of a few
thousand training steps whose length varies from run to run, and some runs
are still on it at the end of training. Attention gets two heads of
dimension 16: its cache grows with the context and has no slots to count,
so the size of its heads does not change the comparison.

<!-- WRITER: the four models are now: Gated DeltaNet (two blocks, 8 heads of size 4: 8 x 4 x 4 = 128 entries of
state per layer, 37,989 parameters), attention (two blocks, 34,821), the hybrid (Gated DeltaNet, then attention;
36,405), and Gated DeltaNet with four times the state (2 heads of size 16: 2 x 16 x 16 = 512 entries per layer,
37,197 parameters, so the state grows at an equal parameter count, as in the head-size experiment of
:numref:`sec_matrix-state`). Gated linear attention is no longer an arm (:numref:`sec_deltanet` compares the write
rules); every sentence about it, and the plateau/induction-head passage (HY-6), goes. The attention mixer is
`d2l.AttentionMixer(32, num_heads=2, conv_width=4, rope=False)` of :numref:`sec_lstm`: the old `ConvAttention`
(a short convolution over queries, keys and values, then fused causal attention, no position encoding). So the
NoPE sentence stays true. Measured reason for rope=False (one clause at most, or a code comment): with rotary
positions the two-layer attention model stayed on a plateau (about 0.4 recall at 64 pairs) in 3 of 18 PyTorch dev
runs; without them in 0 of 10, and in every run since. -->

<!-- WRITER: the decay gates now start with memories of 256 to 2,048 steps, as in
:numref:`sec_matrix-state` (longer than any sequence, at most 192 tokens); "must survive up to 256 steps" and "64 to
1,024" are false. The last sentence of this paragraph (shorter start, gated linear attention markedly slower) is a
claim no cell prints: cut it. `recall_model(layout, num_heads=8, conv_width=4)`: one letter per block, G or A;
num_heads sets the Gated DeltaNet head count (2 gives the larger state); conv_width=1 is the ablation below. -->
A string describes the mixers of a model, one letter per block. The pairs
must survive up to 256 steps, so the decay gates of the recurrent layers
start with memories of 64 to 1,024 steps rather than the 4 to 256 steps of
:numref:`sec_matrix-state`. As the long-range recall experiment of
:numref:`sec_ssm` showed, memories that start short are slow to learn to
last: with the shorter start, the recurrent models below learn the task
more slowly, gated linear attention markedly so.

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

Each model is trained once on fresh batches whose number of pairs cycles
through 4, 8, 16, 32, and 64, with the loss computed at the query positions
only by `d2l.train_queries` of :numref:`subsec_delayed-recall`, and is then
tested separately at each number of pairs. Training uses AdamW for 16,000
steps of 256 sequences, with gradients clipped to norm 1 and the learning
rate of :numref:`sec_batch_norm` (`d2l.cosine_warmup`): a linear warmup
over 800 steps to 0.006, then a half-cosine down to zero.

<!-- WRITER: training is now `d2l.train_recall` with its defaults (:numref:`sec_matrix-state`): 2,000 updates of
256 sequences whose number of pairs cycles through 4, 8, 16, 32, 64, AdamW, gradients clipped to norm 1, peak
learning rate 0.01 reached after a tenth of the updates, then a half-cosine (`d2l.cosine_warmup`: cite
:eqref:`eq_cosine_warmup`, defined in `chapter_convolutional-modern/alexnet.md`, not `sec_batch_norm`, HY-8);
recall is measured on 1,024 new sequences per load. Each of the four models is trained twice, from fresh weights
on fresh data, and the table prints both runs. "16,000 steps", "800 steps", "0.006" are false. Cost on one RTX 4090:
the sweep and the ablation take about 8 minutes in either framework; the whole notebook 10 to 13. -->

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

With 4 pairs, all four models recall nearly every value. Beyond that Gated
DeltaNet loses pairs, and the more pairs a sequence holds, the more it
loses: it recalls about nine tenths of 32 pairs and about three quarters
of 64. Gated linear attention recalls fewer from 8 pairs on, and its curve
varies more from run to run. It first sits on a plateau, recalling about
half of 4 pairs, for a number of training steps that varies from a few
thousand to more than ten thousand, and only then learns the task. The
reference implementation of the layer, from the flash-linear-attention
library of its authors :cite:`Yang.Wang.Shen.ea.2024`, trained on the same
task with the same recipe, sits on the same plateau for as long, in one of
three runs for all 16,000 steps, so the plateau belongs to the model and
not to our code. The hybrid recalls every pair at every load, although one
of its two layers is recurrent, and so does attention in almost every run.
The attention layer of the hybrid looks up the pairs that the recurrent
state cannot hold.

The recurrent curves still rise slowly at the end of training. Trained
twice as long, Gated DeltaNet recalls about 95% of 32 pairs and 80 to 90%
of 64, and gated linear attention still recalls fewer. With keys drawn
from 128 possible ones, fewer than the slots, the same long training
brings Gated DeltaNet to 90% or more of 64 pairs; the gap to the task with 512 keys is the part of the
curve that the slots explain. The delta rule pays off as the regression
view of :numref:`sec_deltanet` suggests. An additive memory returns, with
every value it reads, the values of all stored keys that overlap the query,
while the corrected writes fit the stored pairs and remove much of that
interference. Keys that share a slot overlap, so the correction matters
most when the slots fill up.

<!-- WRITER: results, replacing the two paragraphs above (they read a one-run 16,000-step table with a
gated-linear-attention row; the plateau, reference-implementation, doubled-training, 128-key and "delta rule pays
off" passages report runs no cell prints: cut, HY-9 to HY-11). Measured (two runs per cell; three notebook executions per tab and dev runs of the
same code and settings: 12 runs per arm for Gated DeltaNet and its 4x arm, 8 to 10 for the others; chance 0.25),
recall at 4 / 8 / 16 / 32 / 64 pairs: Gated DeltaNet 0.98-1.00 / 0.91-0.96 / 0.74-0.86 / 0.58-0.68 / 0.47-0.55;
Gated DeltaNet with 4x state 1.00 / 1.00 / 0.99-1.00 / 0.96-1.00 / 0.86-0.99; attention 1.00 at every load in every
run; hybrid 1.00 at every load in every run. Two runs of one cell differ by at most 0.08 for Gated DeltaNet and 0.11
for the 4x arm (both at 64 pairs). Firm lessons, both tabs: (1) the fixed state loses pairs as the load grows, from
nearly all of 4 to about half of 64; (2) one attention layer in place of the second Gated DeltaNet layer removes the
fall entirely (the hybrid equals attention: every pair, every load, every run); (3) four times the state, at the same
parameter count, moves the curve right: at 64 pairs 0.86 to 0.99 against 0.47 to 0.55, every 4x run above every
base run from 16 pairs on, and in each cell by more than three times the larger two-run range (the acceptance test:
in the last executions 0.96/0.99 against 0.55/0.47 in PyTorch, 0.93/0.98 against 0.51/0.50 in JAX). Prediction
from :numref:`sec_matrix-state`'s capacity in entries: the 4x arm at 64 pairs (about 0.95) recalls about as well
as the base arm at 8 to 16 pairs (0.91 to 0.96, 0.74 to 0.86), so "the curve moves right roughly in proportion to
the state" holds loosely; do not claim an exact factor. Say plainly that the models are tiny on purpose (a state of
256 numbers) so that 64 pairs overload them; the experiment shows the mechanism, not the size of the effect in a
real model (evidence at scale: :citet:`Waleffe.Byeon.Riach.ea.2024`). "Attention behind the same convolution"
once in this paragraph (HY-7): it is not the transformer of chapter 11. No sentence on the write rule (HY-11). -->

```{.python .input #hybrids-the-recall-sweep-4}
%%tab pytorch, jax
no_conv = [d2l.train_recall(recall_model('GG', conv_width=1), num_keys=128)
           for _ in range(2)]  # Width 1: each channel is only scaled
print(f'{"pairs":25s}', ''.join(f'{n:>11d}' for n in pairs))
for name, (a, b) in [('Gated DeltaNet', recall['Gated DeltaNet']),
                     ('without the convolution', no_conv)]:
    print(f'{name:25s}', ''.join(f'{a[n]:6.2f}/{b[n]:.2f}' for n in pairs))
```

<!-- WRITER: the ablation cell (new, S2/HY-12): the same Gated DeltaNet model with a convolution of width 1, which
only scales each channel and so mixes nothing across steps; two runs, printed beside the two runs of the sweep.
Measured (same runs as the sweep, both tabs; 8 runs without the convolution), recall at 4 / 8 / 16 / 32
/ 64 pairs without the convolution: 0.65-0.90 / 0.54-0.74 / 0.45-0.58 / 0.39-0.46 / 0.35-0.40, against Gated
DeltaNet 0.98-1.00 / 0.91-0.96 / 0.74-0.86 / 0.58-0.68 / 0.47-0.55. Every run without the convolution is below every
run with it at every load; the gap is largest at few pairs (4: about 0.1 to 0.35; 64: about 0.1). Its two runs can
differ by up to 0.23 at 4 pairs (JAX dev 0.65/0.88), so quote a range, not a value. Last executions: PyTorch
0.85/0.89 ... 0.38/0.40, JAX 0.79/0.80 ... 0.37/0.36. The paragraph below keeps its judgment and the mechanism (the value follows its key one token
later; the convolution brings the two together for one write); everything after "Gated DeltaNet learns more slowly
and recalls fewer pairs at every load" (gated linear attention, unit keys, output gate, SiLU) reports runs no cell
prints: cut, or one clause on unit keys with its reason from :numref:`sec_deltanet`. "learns more slowly" is not
measured; "recalls fewer pairs at every load" is. -->

The parts around the memory matter as much as the write rule. The value
follows its key one token later, and the short convolution is what brings
the two together for a single write. Without it, gated linear attention
recalls barely more than chance even with 4 pairs, which fit easily into
its state, and Gated DeltaNet learns more slowly and recalls fewer pairs at
every load. Without unit keys, Gated DeltaNet does not learn the task at
all. Without the output gate, Gated DeltaNet recalls several points less at
16 pairs and more; for gated linear attention the change stays within its
spread from run to run. The SiLU after the convolution is the exception:
it makes no difference on text, and on this task it costs Gated DeltaNet a
few points at the highest loads.

### Language Modeling

<!-- WRITER: setup now (the Penn Treebank / `d2l.TinyLM` run is retired, S4/HY-7): the text benchmark of
:numref:`sec_lstm` (byte-level WikiText-103, windows of 128 bytes, 64 per step, `d2l.train_text`: LMOptimizer at
peak 2e-3 for 1,000 steps), `d2l.SeqLM` of width 256 with four blocks, three models that differ only in their
mixers: Gated DeltaNet in all four blocks (4 heads of size 64), the hybrid (Gated DeltaNet in the first three,
attention in the fourth), and attention in all four (`d2l.AttentionMixer(256, num_heads=8)`, with rotary positions,
the attention model of :numref:`sec_lstm` and :numref:`sec_mamba`). Two runs each; the table prints parameters,
validation loss in nats per byte, the loss at positions 0-15, 16-63 and 64-127, and ms per training step. The
question from :numref:`sec_mamba`: on real text, does the fixed state fall behind, and does one attention layer
close the gap? -->
Does the difference show on ordinary text? We train two more models on the
character-level Penn Treebank with the recipe of :numref:`sec_lstm`: the
transformer `d2l.TinyLM` of :numref:`sec_test_models`, two blocks of width 128
with learned position embeddings, and a hybrid `d2l.SeqLM` of the same width
whose first block is Gated DeltaNet and second block attention, as in the
recall experiment. With this recipe, Gated DeltaNet alone reached a perplexity
of about 3.1.

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

<!-- WRITER: reading, replacing the paragraph below (its perplexities and the TinyLM comparison are gone; "A
comparison by perplexity alone can therefore miss what a model cannot recall" is not supported by this run: cut,
or keep as the motivation of Exercise 6). Measured (two runs per cell, three notebook executions per tab plus a dev run in
JAX; width 256, four blocks, 1,000 steps of 64 x 128 bytes): Gated DeltaNet 3,573,536 parameters, 1.325 to 1.331
nats per byte; hybrid 3,501,848, 1.329 to 1.335; attention 3,286,784, 1.377 to 1.386. Bands 0-15 / 16-63 / 64-127:
Gated DeltaNet 1.518-1.523 / 1.299-1.305 / 1.297-1.303; hybrid 1.523-1.528 / 1.303-1.310 / 1.300-1.307; attention
1.554-1.562 / 1.353-1.362 / 1.350-1.360. Both tabs agree. Firm lessons: (1) on 128-byte windows the fixed state does
not fall behind: Gated DeltaNet is about 0.05 nats per byte ahead of attention at the same width and depth, in every
band, more than five times any two-run range (at most 0.008); (2) one attention layer in four changes the loss by
less than 0.01 (the hybrid is 0.002 to 0.008 behind Gated DeltaNet in every pair of runs, with overlapping ranges
over all runs: do not claim it is worse, say it is the same to within 0.01); (3) no model gains after byte 16
(16-63 and 64-127 within 0.004 for each). So the recall gap of the sweep does not show in the average loss on
128-byte windows, where few predictions need a pair stored long before; Exercise 6 measures where text repeats.
Place: the other mixers of the chapter on this benchmark, from their sections: LSTM 1.29 to 1.30, GRU about 1.30,
Mamba 1.295 to 1.299, minGRU and RNN about 1.35, attention 1.38 (:numref:`sec_lstm`, :numref:`sec_mamba`). Cost per
training step (tab notes only, never compared across tabs in the shared text): PyTorch (eager) Gated DeltaNet 50 to
52 ms, hybrid 42 to 45, attention 20 to 22; JAX (jitted) 17 to 18, 14 to 15, 9. In both tabs the Gated DeltaNet
step costs about two to two and a half times the attention step at this length, and the hybrid sits between. -->
The hybrid reaches a perplexity of about 3.15, close to Gated DeltaNet
alone, and the transformer between about 3.25 and 3.4, depending on the
run. Language modeling at this scale barely
registers the difference in recall, because predicting the next character
of ordinary text rarely requires looking up a pair stored long before. A
comparison by perplexity alone can therefore miss what a model cannot
recall.

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
attention, plus a constant. This is the accounting of windowed layers
interleaved with global ones in :numref:`sec_windowed-linear-attention`, with a
state of fixed size in place of each window's cache.
:numref:`fig_hy-cache` shows the effect for a model of 32 layers. Jamba
:cite:`Lieber.Lenz.Bata.ea.2024`, with 4 attention layers among 32, reports a cache of 4 GB at a context of 256,000 tokens,
against 32 GB for a transformer of similar size. Memory per sequence decides
how many sequences a GPU can serve at once, and because generating a token
reads the whole cache from memory (:numref:`sec_kv-cache`), it also sets
how fast each token is generated.

![Memory held per sequence during generation, against context length, for a 32-layer model with grouped-query attention (8 key–value heads of dimension 128) and Mamba-2-sized recurrent states, in 16-bit numbers. A transformer's cache reaches 32 GB at 256K tokens; a purely recurrent model stays at 64 MB at every length; a hybrid with 4 attention layers holds an eighth of the transformer's cache.](../img/mdl-modernrnn-hybrid-cache.svg)
:label:`fig_hy-cache`

The chapter asked how to build a memory of fixed size that trains in
parallel, forgets the right things, stores many associations, and can
overwrite them. Its answers, gated linear recurrences computed by scans,
selective state space models, matrix memories trained in chunks, and the
delta rule, make the fixed state as useful as possible. The answer of
practice adds a small memory that grows: a few attention layers, which
retrieve exactly what the fixed state cannot hold. How few, where to place
them, and whether some of them can attend to a window of recent tokens
(:numref:`sec_windowed-linear-attention`) instead of the whole context are
questions that current models answer differently;
:numref:`sec_ssm-further-reading` lists some of their choices.

## Summary

<!-- WRITER: rewrite from the new results (structure: no slots). False now: "every key needs a slot of its own",
"about three quarters of 64" (Gated DeltaNet recalls about half of 64 pairs at this budget), the gated-linear-attention
sentence, "both curves still rise with longer training" (no cell prints it), "Language-modeling perplexity barely
shows the difference" (replace with the text result). True and measured: the hybrid and attention recall every pair
at every load in every run; four times the state at an equal parameter count recalls more at every load from 8
pairs on; without the short convolution Gated DeltaNet recalls fewer pairs at every load. -->

A state of fixed size holds a fixed number of bits, so a recurrent model can
recall only as many items from its context as its state can store, while an
attention layer keeps every token and can always look back. A matrix memory,
which writes by adding outer products and reads linearly, holds at best about
one value per number of its state: every key needs a slot of its own. When
the task can ask about more keys than the state has slots, keys that share a
slot overlap, by at least Welch's bound, and recall falls as the number of
stored pairs grows. On multi-query associative recall, Gated DeltaNet
recalls nearly every pair when there are few and about three quarters of 64,
and gated linear attention, whose writes only add, recalls fewer; both
curves still rise with longer training. Without the short convolution,
which brings each key together with its value, both recall far less.
A hybrid whose second layer is attention recalls every pair, and so does
attention in almost every run.
Language-modeling perplexity barely shows the difference. During generation
only the attention layers keep a cache that grows with the context, so a
hybrid needs the transformer's cache times its fraction of attention layers,
an eighth for a model with one attention layer in eight. Most recent large
models with recurrent layers are hybrids for this reason.

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

<!-- WRITER: Exercise 3 rewritten without slots (structure): "16,000 and 32,000 steps" no longer match the
recipe (2,000 updates, `d2l.train_recall(model, num_updates=...)`); keys from 512 vs 128 is `num_keys=128`; part 2
(width 64) is close to the 4x-state arm now in the text (2 heads of 16 at width 32): ask instead for 1 head of 32,
or for width 64. -->
1. [code] **Keys and slots.** The recurrent models of the recall sweep
   have 256 slots and the task draws keys from 512.
    1. Train Gated DeltaNet for 16,000 and for 32,000 steps, once with keys
       drawn from 512 and once from 128, fewer than the slots. Where do the
       four curves differ? Which differences come from training time and
       which from the number of slots?
    1. Keep 512 keys and double the width to 64, with eight heads. How many
       slots does the model have now, and how does its recall curve move?
    1. Compute the bound :eqref:`eq_hy-welch` for a row of dimension 4 that
       hosts 8 possible keys, and estimate how often two stored keys share a
       row in a sequence of 64 pairs.

1. [code] **Where and how much attention?**
    1. Train the hybrid `AG`, with attention in the first block, and
       compare its recall with that of `GA`. Does the position of the
       attention layer matter? Consider which layer can bring each key
       together with the value that follows it.
    1. Train four-block models `GGGG`, `GGGA`, and `GAGA` of the same
       width. Do two more recurrent layers add as many slots as the
       section's count predicts, and does one attention layer among four
       still recall every pair?

1. [code] **A window instead.** Replace the attention mixer by attention
   that sees only the last $w$ tokens, by masking the scores of earlier
   tokens.
    1. Train the hybrid with $w = 32$ and $w = 128$ and compare recall at
       every number of pairs. Which queries can a window of 32 answer, and
       which must the recurrent layers answer?
    1. How large is the cache of this layer during generation? What does
       the comparison suggest about combining windows with recurrence?

<!-- WRITER: Exercise 6 moves to WikiText bytes (structure): `d2l.WikiText103`, the text models of the section
(`trained`, or `text_model('GGGG', 256)` and the other layouts), `d2l.train_text`; positions whose byte completes
a string of at least eight bytes that already occurred earlier in the same 128-byte window. `d2l.TinyLM` and Penn
Treebank are no longer used in the section. -->
1. [code] **Recall in text.** In the Penn Treebank validation text, mark
   the positions whose character completes a string of at least eight
   characters that already occurred earlier in the same window. Compare
   the loss of Gated DeltaNet, the hybrid, and `d2l.TinyLM` at the marked
   positions and at all other positions. Does the hybrid's advantage on
   recall show up where text repeats itself?

[Discussions](https://d2l.discourse.group/)

[Discussions](https://d2l.discourse.group/t/36785)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.7]{.kicker}

Hybrid architectures<br>
**a fixed state holds fixed bits · slots and overlapping keys · associative recall · one attention layer · memory per sequence**
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

::: {.slide title="A Matrix Memory Holds About One Value per Number"}
- Our recurrent models: 2 layers × 8 heads × 4 × 4 = 256 numbers of
  32 bits; 64 values of 2 bits take only 128 bits.
- But the memory adds outer products and reads linearly.
- Best case, one *slot* per number: a value goes into one row, along a
  key direction orthogonal to the other keys of that row.
- At most $N$ possible keys: every pair stored without interference.
- More possible keys than slots: keys that share a row must overlap.
:::

::: {.slide title="Keys That Share a Row Must Overlap"}
For unit vectors $\mathbf{k}_1, \ldots, \mathbf{k}_m \in \mathbb{R}^d$,
$m > d$ (Welch):

$$\operatorname{mean}_{i \neq j} (\mathbf{k}_i^\top \mathbf{k}_j)^2 \ge
\frac{m - d}{d\,(m - 1)}$$

- Proof: $\mathbf{G} = \sum_i \mathbf{k}_i\mathbf{k}_i^\top$ has trace
  $m$ and rank $\le d$, so $\|\mathbf{G}\|_F^2 \ge m^2/d$.
- $m \gg d$: about $1/d$, as for random keys. Training cannot avoid it.
- Task: 512 possible keys, 256 slots. More stored pairs, more stored keys
  sharing a row, more interference.
:::

::: {.slide title="Multi-Query Associative Recall"}
$n$ pairs, each a key token (one of 512) followed by a value token (one of
4); fillers up to 256 tokens; then every key again, in random order. Answer
each query with the value that followed its key.

<!-- DECK: the example cell moved to the matrix-state section with its id (a placeholder cannot reach another
file); show the layout in text, e.g. `k1 v1 k2 v2 ... | k2 k1 ...`, or drop this slide and restate the task on the
next one. Sequences are now as short as pairs and queries allow (at most 192 tokens, no filler). -->

Chance: 1/4. Up to 64 pairs per sequence.
:::

<!-- DECK: four models now: Gated DeltaNet G G (8 heads of 4), attention A A (2 heads of 16, no position
encoding), hybrid G A, and Gated DeltaNet with 4x state G G (2 heads of 16); no gated-linear-attention row, no
induction head; memories start at 256 to 2,048 steps; trained twice each with `d2l.train_recall` (2,000 updates of
256 sequences). -->
::: {.slide title="Four Models Differ Only in Their Mixers"}
`d2l.SeqLM`, two blocks of width 32:

| model | mixers |
|---|---|
| gated linear attention | L L (8 heads of 4) |
| Gated DeltaNet | G G (8 heads of 4) |
| attention | A A (2 heads of 16) |
| **hybrid** | G **A** |

- Every mixer starts with the same short convolution: it moves each key
  into its value's position, and attention finds it there by content.
- Without it, attention must form an induction head over two layers,
  after a plateau of varying length.
- Decay gates start with memories of 64 to 1,024 steps.
- Trained once with 4 to 64 pairs, 16,000 steps of 256 sequences; tested
  at each number of pairs.
:::

::: {.slide title="One Attention Layer Restores Recall"}
@!hybrids-the-recall-sweep-3
:::

<!-- DECK: rewrite from the new table (two runs, four arms; numbers in the WRITER note of the results); no
plateau, no reference implementation, no doubled training, no 128-key run, no slots. -->
::: {.slide title="Recall Falls with the Number of Pairs, Except with Attention"}
- Gated DeltaNet: nearly all of 4 pairs, about nine tenths of 32, about
  three quarters of 64.
- Gated linear attention: fewer from 8 pairs on; it first sits on a plateau
  (about half of 4 pairs) for a varying number of steps, as does the
  reference implementation.
- The hybrid: every pair at every load; attention in almost every run.
- Twice as long: Gated DeltaNet about 95% of 32 and 80–90% of 64;
  with 128 possible keys (fewer than the 256 slots), 90% or more of 64.
- Corrected writes fit the stored pairs and remove much of the
  interference of overlapping keys.
:::

<!-- DECK: from the ablation cell only (`@!hybrids-the-recall-sweep-4`): with and without the short
convolution, two runs each; drop the other three rows and "worth about 0.1 in perplexity". -->
::: {.slide title="The Parts Around the Memory Matter"}
| without | gated linear attention | Gated DeltaNet |
|---|---|---|
| short convolution | near chance | fewer pairs at every load |
| unit queries and keys | no consistent change | does not learn |
| output gate | within the spread | several points less |
| SiLU | no consistent change | a few points more at 32–64 pairs |

The value follows its key one token later: the convolution brings the two
together for one write. On text, the convolution, SiLU, and output gate
together are worth about 0.1 in perplexity.
:::

::: {.slide title="Check: Bits Against Pairs"}
The two recurrent layers hold 256 numbers of 32 bits.

1. By the counting bound, how many 2-bit values could they hold?
2. Why do they already lose pairs at 32?

::: {.fragment .fade-up}
1. $256 \cdot 32 / 2 = 4{,}096$ values.
2. A linear read gives about one slot per number: 256 slots for 512
   possible keys, 64 rows of dimension 4. Among 32 stored keys, some share
   a row, and keys that share a row overlap.
:::
:::

::: {.slide title="Perplexity Barely Shows the Gap"}
<!-- DECK: was the Penn Treebank cell (retired); now the text table on WikiText bytes. Title and bullets must
follow the new reading (no perplexity 3.1; see the WRITER note after the text cell). -->
@!hybrids-language-modeling-5

- Gated DeltaNet alone: about 3.1.
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

<!-- DECK: the PTB column is gone with Penn Treebank; the closing table takes the text loss at equal size on
WikiText bytes from 12.1, 12.3, 12.5 and this section (Gated DeltaNet about 1.33, the hybrid about 1.33, attention
about 1.38 nats per byte). -->
::: {.slide title="Recap: One Template, Six Memories"}
| | state per layer | update | recall | PTB |
|---|---|---|---|---|
| LSTM, GRU | vector | nonlinear, sequential | short | 3.0 |
| minGRU | vector | $\mathbf{a}_t\odot\mathbf{h} + \mathbf{b}_t$, scan | | 3.2 |
| S4D | $d \times N$ | fixed decays, FFT | by distance | 3.35 |
| Mamba | $H \times N$ | selective decays, scan | selective | 3.0 |
| gated linear attention | $d_v \times d_k$ | $\mathbf{S}\,\textrm{diag}(\boldsymbol{\alpha}_t) + \mathbf{v}\mathbf{k}^\top$, chunks | interference | 3.1 |
| Gated DeltaNet | $d_v \times d_k$ | delta rule, chunks | rebinds | 3.1 |
| + one attention layer | + KV cache | | exact | |

**Open:** how few attention layers, where, and whether windows suffice.
:::
