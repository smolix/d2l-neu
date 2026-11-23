# Hybrid Architectures
:label:`sec_hybrids`

<!-- execution: locked; requires: gpu-2h -->

Every model in this chapter has improved *how* a fixed-size state is updated—from gates to selective dynamics to error-correcting delta rules—yet none of them can store more information than the state's size allows. When a user pastes a table of 64 product codes and prices into a prompt and asks for 20 of them, a Transformer looks back across its full KV cache and retrieves the exact tokens, whereas a purely recurrent model has only what it compressed into its fixed state before knowing which items would be queried. In-context retrieval—in few-shot prompting, retrieval-augmented generation, and code editing—is where fixed-state sequence models hit a hard ceiling.

The remedy adopted by the recurrent language models in production use is pragmatic: interleave a small fraction of full attention layers among the recurrent blocks. Below we establish two fundamental limits on fixed-state recall—an information-theoretic bit-counting bound and Welch's geometric lower bound on key overlap—compare pure `GatedDeltaNet`, pure attention, and a `GA` hybrid on multi-query associative recall and byte-level WikiText-103, ablate the short depthwise convolution that pairs adjacent key and value tokens, quantify inference KV-cache savings at 256K context length, and summarize all sequence memories covered in this chapter.

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

By the pigeonhole principle, a state of $M$ numbers stored at $b$ bits each has at most $2^{Mb}$ distinct configurations. To distinguish and reproduce all $V^k$ possible sequences of $k$ tokens drawn from a vocabulary of size $V$, the state must satisfy :cite:`Jelassi.Brandfonbrener.Kakade.ea.2024`:

$$M b \ge k \log_2 V.$$
:eqlabel:`eq_hy-counting`

Thus the maximum number of tokens a fixed state can recall grows linearly with state size $M$ and is completely independent of context length $T$, whereas a Transformer's KV cache grows linearly with $T$ and never saturates.

In practice, linear matrix memories hit a geometric limit long before saturating their raw bit count. In our associative recall benchmark below, a sequence stores up to 64 pairs with keys drawn from 128 symbols and values from 4 symbols (at most $128 \times 2 = 256$ bits of table information, or $64 \times 2 = 128$ bits for the 64 values), while two `GatedDeltaNet` layers with 8 heads of $4 \times 4$ float32 states hold $256 \times 32 = 8{,}192$ bits—more than $30\times$ the required bits. Why does a matrix memory lose pairs well below its raw bit count?

### Overlapping Keys

Whether a matrix head $\mathbf{S} \in \mathbb{R}^{d_v \times d_k}$ is updated additively (:numref:`sec_matrix-state`) or via the delta rule toward the least-squares fit (:numref:`sec_deltanet`), exact linear readout $\mathbf{S}\mathbf{k}_j = \mathbf{v}_j$ for arbitrary values requires the stored keys to be linearly independent—and $\mathbb{R}^{d_k}$ holds at most $d_k$ linearly independent vectors. Once the number of stored keys $m$ exceeds the head dimension $d_k$, the keys *must* overlap no matter how training chooses them.

**Proposition (Welch's Bound).** Let $\mathbf{k}_1, \ldots, \mathbf{k}_m$ be unit vectors in $\mathbb{R}^{d_k}$ with $m > d_k$. Then the average squared inner product over all distinct pairs $i \neq j$ is at least :cite:`Welch.1974`:

$$\frac{m - d_k}{d_k\,(m - 1)}.$$
:eqlabel:`eq_hy-welch`

**Proof.** Let $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_k \times d_k}$. Its trace is $\sum_{i=1}^m \|\mathbf{k}_i\|^2 = m$, so its $d_k$ eigenvalues sum to $m$. By Cauchy–Schwarz, the sum of its squared eigenvalues—which equals $\|\mathbf{G}\|_F^2 = \sum_{i,j} (\mathbf{k}_i^\top \mathbf{k}_j)^2$—is at least $m^2 / d_k$. Subtracting the $m$ diagonal terms $(\mathbf{k}_i^\top \mathbf{k}_i)^2 = 1$ leaves at least $m^2 / d_k - m = m(m - d_k)/d_k$ across the $m(m - 1)$ off-diagonal pairs. $\blacksquare$

In a head of dimension $d_k = 4$, $m = 64$ unit keys must have mean squared overlap at least $60 / (4 \times 63) \approx 0.24$—almost indistinguishable from random unit vectors ($1/d_k = 0.25$). Widening the head to $d_k = 16$ lowers the Welch floor to $48 / (16 \times 63) \approx 0.05$.

This predicts two clear experimental signatures on multi-query associative recall:

1. A pure `GatedDeltaNet` stack will degrade as the number of stored pairs $n$ grows past its state capacity, while quadrupling its per-layer state entries (from $8 \times 4^2 = 128$ to $2 \times 16^2 = 512$ at fixed width $d = 32$) will shift the degradation to roughly $4\times$ higher pair loads.
2. Replacing just one of the two recurrent blocks with full causal attention (`GA`) will eliminate the capacity drop entirely, because the attention layer retains every token in its cache.

## Recall with One Attention Layer

### Four Models

We test four two-block `d2l.SeqLM` models ($d = 32$) on multi-query associative recall (`num_keys=128`, `num_values=4`, loads $n \in \{4, 8, 16, 32, 64\}$):

- **`Gated DeltaNet` (`GG`):** 8 heads of size 4 ($128$ state entries/layer);
- **`Gated DeltaNet, 4x state` (`GG, num_heads=2`):** 2 heads of size 16 ($512$ state entries/layer, same parameter count);
- **`attention` (`AA`):** 2-head causal attention (`d2l.AttentionMixer`) with the same width-4 depthwise causal convolution on $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ and no positional encoding (`rope=False`);
- **`hybrid` (`GA`):** `GatedDeltaNet` in block 1 followed by `AttentionMixer` in block 2.

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
jax.config.update('jax_default_matmul_precision', 'float32')  # Not TF32

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

We train two seeds of each architecture for 2,000 updates with `d2l.train_recall` and evaluate accuracy across pair loads.

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

We now average the recall curves across the two runs of each arm and plot accuracy as a function of the number of stored pairs $n \in \{4, 8, 16, 32, 64\}$:

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

Both predictions hold:

1. **State capacity scaling in `GatedDeltaNet`:** With 128 state entries/layer (`GG`), recall drops from about $1.00$ at 4 pairs to between $0.6$ and $0.7$ at 32 pairs and about $0.5$ at 64 pairs. Quadrupling the state to 512 entries/layer (`Gated DeltaNet, 4x state`) keeps recall near $1.00$ through 32 pairs. At 64 pairs it recalls most pairs, between about $0.8$ and $1.0$ depending on the run.
2. **One attention layer restores exact recall:** Both `attention` (`AA`) and the `hybrid` (`GA`) achieve $1.00$ recall across every load up to 64 pairs in every run—mirroring the large-scale findings of :citet:`Waleffe.Byeon.Riach.ea.2024`.

### The Short Convolution

Why did we include a width-4 depthwise causal convolution in front of our recurrent and attention mixers? In associative recall, each value arrives at token $t$ immediately *after* its key at token $t - 1$. Without local token mixing (`conv_width=1`), a single matrix-state layer at step $t$ sees only the value token and cannot bind it to the preceding key token. Let us ablate the convolution by setting `conv_width=1` in `GG`:

```{.python .input #hybrids-the-recall-sweep-4}
%%tab pytorch, jax
no_conv = [d2l.train_recall(recall_model('GG', conv_width=1), num_keys=128)
           for _ in range(2)]  # Width 1: only scales and shifts each channel
print(f'{"pairs":25s}', ''.join(f'{n:>11d}' for n in pairs))
for name, (a, b) in [('Gated DeltaNet', recall['Gated DeltaNet']),
                     ('without the convolution', no_conv)]:
    print(f'{name:25s}', ''.join(f'{a[n]:6.2f}/{b[n]:.2f}' for n in pairs))
```

Removing the width-4 causal convolution degrades recall at every single load (by roughly $10$ to $35$ percentage points, dropping even 4-pair recall to between $0.8$ and $0.9$).

## Language Modeling at Equal Size

How do pure `GatedDeltaNet` (`GGGG`), a 3-to-1 hybrid (`GGGA`), and pure causal `attention` (`AAAA`) compare on our byte-level WikiText-103 benchmark (`d2l.SeqLM`, 4 blocks at width $d = 256$)? We ask twice: first on the 128-byte windows used throughout this chapter, and then on windows of 1,024 bytes, to see whether a longer context separates a fixed state from a growing cache. Both times we train for 8,000 steps and not for the 1,000 steps of the earlier sections, because a model learns to use distant context late in training. The losses below are therefore lower than those of the earlier text tables and cannot be compared with them.

### Short Windows

We first load the 128-byte `WikiText103` dataset:

```{.python .input #hybrids-language-modeling-3}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
```

The comparison needs one more arm to be fair. Mamba, gated linear attention, and Gated DeltaNet all mix neighboring tokens with a width-4 causal convolution before their recurrence, while the attention mixer of our text benchmarks has had none. A lead of `GatedDeltaNet` over `attention` could therefore belong to the convolution rather than to the memory. We give attention the same convolution in a fourth arm, as we did in the recall experiment.

To this end, `text_model(layout, d)` constructs a 4-block `d2l.SeqLM` whose mixers follow a four-letter `layout` string of `'G'` (`GatedDeltaNet`, 4 heads), `'A'` (`AttentionMixer`, 8 heads), `'C'` (the same attention behind a width-4 causal convolution), and `'R'` (a `GRU`, for the longer windows below):

```{.python .input #hybrids-language-modeling-4}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32
text_mixers = {'R': lambda d: d2l.Recurrent(d2l.init_lecun(
                   nn.GRU(d, d, batch_first=True))),
               'G': lambda d: d2l.GatedDeltaNet(d, num_heads=4),
               'A': lambda d: d2l.AttentionMixer(d, num_heads=8),
               'C': lambda d: d2l.AttentionMixer(d, num_heads=8,
                                                 conv_width=4)}

def text_model(layout, d):  # 256 byte values; one block per letter
    return d2l.SeqLM(256, d, [text_mixers[c](d) for c in layout]).to(
        d2l.try_gpu())
```

```{.python .input #hybrids-language-modeling-4}
%%tab jax
jax.config.update('jax_default_matmul_precision',
                  'tensorfloat32')  # fp32 matmuls may use tf32
text_mixers = {
    'R': lambda d, rngs: nnx.RNN(nnx.GRUCell(
        d, d, recurrent_kernel_init=nnx.initializers.lecun_normal(),
        rngs=rngs), unroll=8),
    'G': lambda d, rngs: d2l.GatedDeltaNet(d, num_heads=4, rngs=rngs),
    'A': lambda d, rngs: d2l.AttentionMixer(d, num_heads=8, rngs=rngs),
    'C': lambda d, rngs: d2l.AttentionMixer(d, num_heads=8, conv_width=4,
                                            rngs=rngs)}

def text_model(layout, d):  # 256 byte values; one block per letter
    rngs = nnx.Rngs(d2l.get_key())
    return d2l.SeqLM(256, d, [text_mixers[c](d, rngs) for c in layout],
                     rngs=rngs)
```

We train two runs each of `'GGGG'`, `'GGGA'`, `'AAAA'`, and `'CCCC'` at width $d = 256$ for 8,000 steps via `d2l.text_table`:

```{.python .input #hybrids-language-modeling-5}
%%tab pytorch, jax
arms = {'Gated DeltaNet': ('GGGG', 256), 'hybrid': ('GGGA', 256),
        'attention': ('AAAA', 256),
        'attention, conv': ('CCCC', 256)}  # Width 256, four blocks each
trained = d2l.text_table(text_model, arms, data, num_steps=8000)
```

On 128-byte windows three of the four arms finish together: `Gated DeltaNet` at about $1.16$ nats per byte, and the `GGGA` hybrid and attention with the convolution at about $1.15$. Only `attention` without the convolution stands apart, about $0.03$ behind at $1.18$. At this length the loss is dominated by local syntax: it hardly changes after the first 16 bytes of a window (:numref:`subsec_deep-rnn`), so the table mostly measures how well a mixer combines neighboring bytes.

The convolution thus accounts for the whole deficit of plain attention. With it, attention is level with the hybrid. Both are ahead of `Gated DeltaNet` by $0.01$ at most, which is about what separates two runs of the same arm, so we read no ranking into it. On short windows neither the matrix memory nor the cache has an advantage that local mixing does not explain, and the hybrid's single attention layer adds nothing that we can measure. That layer matters where exact long-range lookup is needed (Exercise :exref:`ex_hybrids-recall-in-text`). Does it matter on real text once the windows are long enough?

### Long Windows

Windows of 128 bytes cannot show what a fixed state loses against a growing cache, nor what a vector state loses against a matrix memory. We therefore train four mixers on windows of 1,024 bytes: a stack of `GRU`s (`'RRRR'`), `Gated DeltaNet`, the `GGGA` hybrid, and attention with the convolution. The batch size drops from 64 to 8, so that every step still sees the same number of bytes and the 8,000 steps the same amount of text.

```{.python .input #hybrids-language-modeling-at-equal-size-1}
%%tab pytorch, jax
long_data = d2l.WikiText103(batch_size=8, num_steps=1024)  # Same bytes/step
long_arms = {'GRU': ('RRRR', 256), 'Gated DeltaNet': ('GGGG', 256),
             'hybrid': ('GGGA', 256), 'attention, conv': ('CCCC', 256)}
long_trained = d2l.text_table(text_model, long_arms, long_data,
                              num_steps=8000)  # Long context needs training
```

The average over a window hides where in the window each model gains. A model that uses its whole context keeps improving as the position grows, whereas one that summarizes only the recent bytes stops improving early. To tell the two apart, we average the validation loss of the two runs over bands of positions $[a, 2a)$ that double in length, from bytes 16–31 up to bytes 512–1,023:

```{.python .input #hybrids-language-modeling-at-equal-size-2}
%%tab pytorch, jax
losses = {name: np.mean([d2l.loss_by_position(model, long_data)
                         for model in models], 0)  # Mean of the two runs
          for name, models in long_trained.items()}
starts = [16, 32, 64, 128, 256, 512]  # Bands [a, 2a) of positions
bands = {name: [loss[a:2 * a].mean() for a in starts]
         for name, loss in losses.items()}
d2l.plot(starts, list(bands.values()), 'positions [a, 2a) in the window',
         'validation loss (nats/byte)', legend=list(bands), xscale='log',
         fmts=('-o', '-s', '--x', '-^'), figsize=(4.5, 3))
d2l.plt.xticks(starts, starts)  # Ticks at the first position of each band
print(f'{"positions":16s}', ''.join(f'{f"{a}-{2 * a - 1}":>10s}'
                                   for a in starts))
for name, band in bands.items():
    print(f'{name:16s}', ''.join(f'{v:10.3f}' for v in band))
```

Longer windows help every model that we trained at both lengths. `Gated DeltaNet` ends between $1.11$ and $1.12$ and the hybrid between $1.07$ and $1.08$, against $1.16$ and $1.15$ on windows of 128 bytes. Attention with the convolution ends at about $1.07$ in most runs. About one run in five ends near $1.12$ instead, for a reason that we give below. The `GRU` reaches $1.09$. It is the slowest arm by a wide margin, at about twice the time per step of the next slowest or more, because it alone processes a window one step after another.

The three arms that were level on short windows are level no longer, and the columns of the table show where they part. From the first 128 bytes of a window to the last 512, the loss of the `GRU` falls by about $0.07$ and that of `Gated DeltaNet` by about $0.08$. The hybrid gains about $0.13$, and attention, in most runs, about $0.16$. The plot locates the gain more closely. Up to byte 127 the fixed states hold their own: the `GRU` is lowest in every band, attention is highest in the first bands, and the hybrid cannot be told from pure `Gated DeltaNet`. Beyond byte 128 the `GRU` and `Gated DeltaNet` gain little more, about $0.03$ by the last band. The hybrid keeps falling. It leaves `Gated DeltaNet` behind at byte 128, and from byte 256 on it is below the `GRU` as well. At bytes 512–1,023 it leads `Gated DeltaNet` by about $0.05$ and the `GRU` by about $0.02$. All of this lead comes from the one attention layer in its fourth block.

Attention falls the furthest, though not in every run. In most runs it ends at $1.03$ to $1.04$ on bytes 512–1,023, about $0.02$ ahead of the hybrid and the best of the four models. About one run in five lags. It ends at around $1.12$ overall and at about $1.10$ in the last band, no better than `Gated DeltaNet`. The plot shows the mean of the two runs in the table, so where one of them is such a run, the attention curve ends above the hybrid's and not below it.

The two kinds of run differ by a single event. For the first thousand steps or so, all runs follow the same training loss. In most runs the loss then falls suddenly by about $0.05$, at some step between 1,000 and 5,000. A lagging run declines smoothly and never shows this drop. The drop is the moment at which attention learns to copy. Text repeats itself: a name, a date, or a phrase often recurs within a kilobyte. A model that finds the earlier occurrence of the bytes it has just read, and predicts the byte that followed them there, gains at every such repeat. :citet:`Olsson.Elhage.Nanda.ea.2022` call this circuit an *induction head*, and we met it in :numref:`sec_what-attention-computes`. We can test for it directly. On windows that contain the same passage of 512 bytes twice, a run that has had its drop predicts the second copy at $0.6$ to $0.8$ of the loss of the first. A lagging run predicts both copies at nearly the same loss, through all 8,000 steps. On the first 128 bytes of a window, where there is little to copy from, the two kinds of run cannot be told apart.

A lagging run is not unstable. Its loss does not diverge, and its gradients are as large as those of the other runs. It is simply late. The step at which the circuit forms varies with the seed, and the cosine schedule leaves little learning rate after about step 6,000, so a run that has not formed the circuit by then does not form it at all. The short convolution helps here as it did in the recall task: without it, most runs have had no drop by step 5,000. We also tried other learning rates, no weight decay, and a different number of attention heads. The misses became rarer in one case and disappeared in none. Sudden drops of this kind are typical when a network must assemble a circuit from parts that are of little use on their own, and they are a good reason to train more than one seed before ranking two architectures.

With a long context and enough training, then, text shows a weaker form of what the recall task showed. The growing cache gains the most from distant bytes, a single attention layer among three recurrent ones recovers a good part of that gain, and a fixed state alone gains the least. On windows of 128 bytes there are no distant bytes to use, which is why the short-window table cannot show this.

## Inference Memory

During autoregressive generation, per-sequence memory determines both maximum serving batch size and memory-bandwidth-bound decoding speed:

- **Grouped-query attention layer** ($n_\textrm{kv} = 8$ key–value heads of dimension $d_\textrm{head} = 128$ in 16-bit floats): stores $2 \times n_\textrm{kv} \times d_\textrm{head} \times 2 = 4\textrm{ KB}$ *per token*—reaching $512\textrm{ MB}$ per layer at $T = 128\textrm{K}$ tokens.
- **Recurrent / matrix-state layer** (e.g., Mamba-2 at width $4{,}096$ with $8{,}192$ channels and state dimension $128$ in 16-bit floats): stores a fixed $8{,}192 \times 128 \times 2 = 2\textrm{ MB}$ *regardless of context length* (overtaken by the attention layer's KV cache after just $512$ tokens).

In a hybrid architecture, only the attention layers accumulate a growing KV cache (:numref:`fig_hy-cache`). A 32-layer model with 4 attention layers (1 in 8, as in Jamba :cite:`Lieber.Lenz.Bata.ea.2024`) cuts the 256K-token cache from $32\textrm{ GB}$ to $4\textrm{ GB}$, while 3-to-1 Gated DeltaNet hybrids such as Qwen3-Next :cite:`Qwen.Team.2025` and Kimi Linear :cite:`Kimi.Team.2025b` reduce the KV cache by $4\times$.

![Memory held per sequence during generation, against context length, for a 32-layer model with grouped-query attention (8 key–value heads of dimension 128) and Mamba-2-sized recurrent states, in 16-bit numbers. A transformer's cache reaches 32 GB at 256K tokens; a purely recurrent model stays at 64 MB at every length; a hybrid with 4 attention layers holds an eighth of the transformer's cache.](../img/mdl-modernrnn-hybrid-cache.svg)
:label:`fig_hy-cache`

## The Chapter's Memories Compared
:label:`subsec_hy-memories`

:numref:`tab_hy-memories` compares every sequence mixer built in this chapter alongside causal attention and the hybrid (with $d$ denoting model width, $N$ state dimension, $H = 2d$ expanded channels in Mamba, $h$ number of heads, and $T$ sequence length).

:The memories of this chapter. Text: validation loss in nats per byte on byte-level WikiText-103, windows of 128 bytes, two runs per model; parameters in parentheses. The models of this section were trained for 8,000 steps and the others for 1,000, so only losses with the same number of steps can be compared. On windows of 1,024 bytes after 8,000 steps, attention with the convolution leads in most runs, ahead of the hybrid, the GRU, and Gated DeltaNet.
:label:`tab_hy-memories`

| Memory | State per layer at generation | Training | Synthetic task | Text |
| :-- | :-- | :-- | :-- | :-- |
| LSTM, GRU | $2d$, $d$ | one step after another | gates that start near one keep a symbol for 200 steps; started at one half, they lose it by 50 | 1.29 to 1.30 (4.3 M, 3.8 M) |
| minGRU | $d$ | scan | at chance after 100 steps; with the timescales of S4D, recalls after 1,000 | 1.34 to 1.35 (2.8 M) |
| S4D | $dN$ | convolution or scan | recalls after 1,000 steps; at most one in five on selective copying | not trained |
| Mamba | $HN$ | scan | 0.98 or more on selective copying | 1.30 (4.0 M) |
| gated linear attention | $d^2/h$ | chunks | about half of 64 pairs with 128 entries, more with 512 | not trained |
| Gated DeltaNet | $d^2/h$ | chunks with a triangular solve | 0.95 or more of rewritten keys, against about 0.9 for gated linear attention | 1.16 after 8,000 steps (3.6 M) |
| attention | $2dT$ | fused attention over all positions | every pair at every load | 1.38 to 1.39; after 8,000 steps 1.18, and 1.15 with the width-4 convolution (3.3 M) |
| hybrid | recurrent states plus $2dT$ per attention layer | both | every pair, with attention in one of two blocks | 1.15 after 8,000 steps (3.5 M) |

Reading across the columns summarizes the design trade-offs:

- For **low-power streaming or continuous sensor/audio processing** where exact lookup of arbitrary past tokens is not required, classical gated recurrences or selective SSMs provide compact $O(1)$-state processing.
- For **long-context foundation models**, interleaving chunk-parallel delta-rule matrix recurrences (`GatedDeltaNet`) or selective SSMs (`Mamba-2`) with a small ratio of full attention layers ($1{:}3$ to $1{:}7$) combines parallel linear-time training, $4\times$–$8\times$ smaller inference KV caches, and exact content-addressable recall.

## Summary

A fixed-size state of $M$ numbers at $b$ bits each can distinguish at most $2^{Mb}$ histories ($Mb \ge k \log_2 V$ to recall $k$ tokens from vocabulary $V$). In matrix-state layers, capacity is constrained even earlier by geometry: by Welch's bound, $m > d_k$ unit keys in $\mathbb{R}^{d_k}$ must have average squared overlap at least $(m - d_k)/(d_k(m - 1))$, causing cross-key interference under both additive and delta-rule updates. On multi-query associative recall, a pure `GatedDeltaNet` model degrades once pair count outgrows its state entries, whereas quadrupling state entries scales capacity proportionally and replacing a single recurrent block with full attention (`GA`) restores $100\%$ recall across all loads. A width-4 causal convolution in front of the mixer is essential so adjacent key and value tokens can bind within a single layer. On byte-level text the differences are smaller, and they depend on the length of the window. On windows of 128 bytes, `Gated DeltaNet`, the hybrid, and attention with the convolution finish within about $0.01$ nats per byte of one another after 8,000 steps, and the deficit of plain attention, about $0.03$, disappears once it has the same convolution. On windows of 1,024 bytes the models separate beyond byte 128: attention gains the most from distant context in most runs, the hybrid follows, and pure `Gated DeltaNet` gains the least. During generation, only the attention layers of a hybrid require a growing KV cache, cutting inference cache memory in proportion to the fraction of attention layers.

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

1. [code] **Recall in text.** In the validation windows of byte-level
   WikiText-103, mark the positions whose byte completes a string of at
   least eight bytes that already occurred earlier in the same window.
   :exlabel:`ex_hybrids-recall-in-text`
    1. Compare the loss of the trained Gated DeltaNet, hybrid, and
       attention models (`trained`) at the marked positions and at all
       other positions. Does attention's advantage on recall show up where
       text repeats itself?
    1. Repeat with windows of 512 bytes (`num_steps=512`, with fewer
       windows per step). How does the share of marked positions change,
       and with it the difference between the models?

1. [code] **Watching the copy circuit form.** Build validation windows of
   1,024 bytes whose second half repeats the first. Train attention with
   the convolution (`'CCCC'`) on ordinary windows of 1,024 bytes, and
   every 250 steps record the training loss and the ratio of the loss on
   the second half of the repeated windows to the loss on the first.
    1. At which step does the ratio fall below 1, and does the training
       loss drop at the same step?
    1. Repeat with several seeds. How much does the step of the drop vary,
       and what does a run look like that has no drop within 8,000 steps?

[Discussions](https://d2l.discourse.group/t/36785)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.8]{.kicker}

Hybrid Architectures: Combining Recurrence with Global Attention<br>**information & Welch capacity bounds · Gated DeltaNet + attention hybrids · local convolution binding · KV-cache scaling**
:::
:::

::: {.slide title="Why Every Fixed-Size State Hits an In-Context Ceiling"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Pre-Query Compression Limit:** A fixed recurrent state $\mathbf{S}_t$ must compress all $64$ prompt pairs *before* knowing which $20$ keys will be queried.
- **Post-Query Attention Lookup:** Full attention retains the raw KV table for exact random access at $O(T)$ memory cost.
- **The Hybrid Principle:** Interleave 1 sparse attention layer per 3–7 recurrent layers.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-table-lookup-ceiling
:::
:::
:::

::: {.slide title="Limit 1: The Information-Theoretic Bit-Counting Bound"}
[Pigeonhole Capacity Limit]{.kicker}

::: {.cols .vc}
::: {.col}
$$M b \ge k \log_2 V \implies k_{\max} \le \frac{M b}{\log_2 V}$$

- **Abundant Raw Bits:** Our 2-layer `GG` model ($8$ heads of $4 \times 4$) holds **$8{,}192$ `float32` bits**, whereas $64$ four-valued answers carry just **$128$ bits** ($>30\times$ smaller).
- **Linear-Algebraic Bottleneck:** Matrix memories saturate rank $d_k$ long before exhausting raw float bits.
:::

::: {.col .fig}
@fig:ch12-mrnn-pigeonhole-vs-algebra
:::
:::
:::

::: {.slide title="Limit 2: Linear Independence and Welch's Overlap Bound"}
[Geometric Capacity Limit]{.kicker}

::: {.cols .vc}
::: {.col}
$$\operatorname{mean}_{i \neq j}\,(\mathbf{k}_i^\top \mathbf{k}_j)^2 \ge \frac{m - d_k}{d_k\,(m - 1)} \quad (m > d_k)$$

- **Rank-$d_k$ Gram Deficit:** $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_k \times d_k}$ has trace $m$ and rank $\le d_k$, so Cauchy–Schwarz forces $\|\mathbf{G}\|_F^2 \ge m^2 / d_k$.
- **Universal Linear Floor:** Applies to *every* linear matrix readout $\mathbf{y} = \mathbf{S}\mathbf{q}$ (`GLA`, `DeltaNet`, `Longhorn`).
:::

::: {.col .fig}
@fig:ch12-mrnn-welch-bound-geometry
:::
:::
:::

::: {.slide title="What Welch's Bound Predicts for Trained Key Embeddings"}
[Theory to Experiment]{.kicker}

::: {.cols .vc}
::: {.col}
- **At $d_k = 4, m = 64$:** Welch's lower bound ($0.2381$) is **$95.2\%$** of random overlap ($1/4 = 0.2500$), leaving gradient descent almost zero room to reduce cross-talk.
- **At $d_k = 16, m = 64$:** Welch's bound drops $5\times$ to **$0.0476$**, predicting a $4\times$ capacity jump when head width quadruples at fixed model width $d = 32$.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-welch-dk4-vs-dk16
:::
:::
:::

::: {.slide title="Four 2-Block Architectures on Associative Recall"}
[Controlled Experimental Design]{.kicker}

::: {.cols .vc}
::: {.col}
- **Four Matched 2-Block Stacks ($d = 32$):**
  1. `GG` ($8$ heads of $4 \times 4 = 128$ floats/layer).
  2. `GG, 4x state` ($2$ heads of $16 \times 16 = 512$ floats/layer, same params).
  3. `AA` ($2$ full causal attention layers).
  4. `GA` hybrid (`GatedDeltaNet` $\to$ `Attention`, $50\%$ KV cache).
:::

::: {.col .fig .big}
@fig:ch12-mrnn-four-recall-layouts
:::
:::
:::

::: {.slide title="Multi-Query Associative Recall: 4 to 64 Key–Value Pairs"}
[Recall Sweep Results]{.kicker}

Mean recall across two training runs ($2{,}000$ AdamW updates cycling across pair loads $n \in \{4, 8, 16, 32, 64\}$; chance $= 0.25$):

@!hybrids-the-recall-sweep-3
:::

::: {.slide title="1 Attention Layer Lifts the Ceiling; 4x State Shifts It"}
[Interpreting the Recall Sweep]{.kicker}

::: {.cols .vc}
::: {.col}
- **4× State Scaling (`GG` vs. `GG, 4x state`):** Quadrupling head entries from $128$ to $512$ floats/layer lifts 64-pair recall from **$\approx 0.5$ to between $0.8$ and $1.0$** at equal parameter count.
- **Why Short Conv Matters:** Adjacent tokens $(k_i, v_i)$ require width-4 causal convolution so step $t$'s outer product $\mathbf{v}_t \mathbf{k}_t^\top$ binds both tokens together.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-conv4-key-value-binding
:::
:::
:::

::: {.slide title="Why the Width-4 Causal Convolution Is Essential"}
[Local Token Binding Ablation]{.kicker}

What happens if we remove the width-$4$ depthwise causal convolution in front of `GatedDeltaNet` by setting `conv_width=1`?

@!hybrids-the-recall-sweep-4

- **Why `conv_width=1` fails even at $4$ pairs:** In our sequence format, key token $k_i$ arrives at step $t - 1$ and its value token $v_i$ arrives at step $t$. Without local receptive-field mixing (`conv_width=1`), step $t$'s linear projections $(\mathbf{k}_t, \mathbf{v}_t)$ see only the value token $v_i$ and cannot bind it to the preceding key token $k_i$ inside a single layer!
- **Width-4 causal convolution bridges adjacent tokens:** Mixing $4$ consecutive steps into $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ lets step $t$ extract $\mathbf{k}_t$ from position $t-1$ and $\mathbf{v}_t$ from position $t$ in a single outer-product write.
:::

::: {.slide title="WikiText-103 Bytes: GGGG, GGGA Hybrid, and Attention"}
[Language Modeling Benchmark]{.kicker}

Comparing 4-block width-256 `Gated DeltaNet` (`GGGG`), a 3-to-1 `hybrid` (`GGGA`, attention in block 4), pure causal `attention` (`AAAA`), and attention behind the same width-4 causal convolution (`CCCC`) on 128-byte WikiText-103 windows, trained for $8{,}000$ steps:

@!hybrids-language-modeling-5
:::

::: {.slide title="Loss by Position on 1,024-Byte Windows"}
[Longer Context Separates the Memories]{.kicker}

::: {.cols .vc}
::: {.col}
- **Same bytes per step:** batch size $8$ instead of $64$, windows of $1{,}024$ bytes, $8{,}000$ steps; mean of two runs over bands of positions $[a, 2a)$.
- **Fixed states flatten:** up to byte $127$ the `GRU` is lowest; past byte $128$ the `GRU` and `GGGG` gain only about $0.03$ more.
- **The hybrid keeps falling:** level with `GGGG` up to byte $127$, then about $0.05$ ahead of it and $0.02$ ahead of the `GRU` at bytes $512$–$1{,}023$.
- **Attention gains the most, in most runs:** highest at first, $1.03$–$1.04$ in the last band. About one run in five stays near $1.10$: its induction head, which copies what followed an earlier occurrence of the context, has not formed by the time the cosine schedule has used up the learning rate. Such a run lifts the averaged curve above the hybrid's.
:::

::: {.col .fig .big}
@!hybrids-language-modeling-at-equal-size-2[out=2]
:::
:::
:::

::: {.slide title="Pitfall: Short-Window Loss Masks Retrieval Deficits"}
[Benchmarking Insight]{.kicker}

::: {.cols .vc}
::: {.col}
- **Short-Window Loss (`T = 128`, $8{,}000$ steps):** `GGGG` ($\approx 1.16$ nats/byte), `GGGA` and attention with the convolution (both $\approx 1.15$) finish together; plain `AAAA` trails at $1.18$ only for lack of local mixing. The loss barely changes after byte 16 and says nothing about recall.
- **Long-Range Recall (`m = 64` pairs, 2 blocks):** `GG` ($d_k = 4$) saturates at $\approx 50\%$, whereas the hybrid `GA` achieves **$100\%$ recall**. On $1{,}024$-byte windows the 3:1 `GGGA` (**$1.07$–$1.08$ nats/byte**) leads `GGGG` ($1.11$–$1.12$) at $25\%$ of the KV cache.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-perplexity-vs-retrieval
:::
:::
:::

::: {.slide title="Inference Memory Scaling: GQA Cache vs. Fixed State"}
[Serving & Generation Footprint]{.kicker}

During autoregressive decoding at scale ($16$-bit `float16` / `bfloat16`), per-sequence memory determines maximum batch size and memory-bandwidth latency:

| Layer Type (Width $d = 4{,}096$, $16$-bit Floats) | Per-Layer Memory Formula | At $T = 512$ | At $T = 128\text{K}$ | At $T = 256\text{K}$ |
| :--- | :--- | :---: | :---: | :---: |
| **Grouped-Query Attention ($8$ KV heads $\times 128$)** | $2 \times 8 \times 128 \times 2\text{ B} \times T = 4\text{ KB} \times T$ | $2\text{ MB}$ | **$512\text{ MB}$** | **$1{,}024\text{ MB}$** |
| **Mamba-2 / Gated DeltaNet ($8{,}192 \times 128$ state)** | $8{,}192 \times 128 \times 2\text{ B} = \mathbf{2\text{ MB}}$ | $2\text{ MB}$ | **$2\text{ MB}$** | **$2\text{ MB}$** |

- **Crossover at 512 tokens:** By $T = 128\text{K}$ tokens, a single GQA layer's KV cache is **$256\times$ larger** than a recurrent matrix state ($512\text{ MB}$ vs. $2\text{ MB}$).
:::

::: {.slide title="KV Cache Savings in Frontier Hybrid Architectures"}
[32-Layer Memory Scaling]{.kicker}

::: {.cols .vc}
::: {.col}
- **32-layer Transformer at $256\text{K}$ tokens:** $32 \times 1\text{ GB} = \mathbf{32\text{ GB}}$ of KV cache *per sequence*.
- **1-in-8 Hybrid (Jamba, $4$ attention + $28$ Mamba layers):** $4 \times 1\text{ GB} + 28 \times 2\text{ MB} \approx \mathbf{4.06\text{ GB}}$ (**$8\times$ reduction**).
- **1-in-4 Hybrid (Qwen3-Next & Kimi Linear, 3:1 Gated DeltaNet + Attention):** Cuts KV cache by **$4\times$** while retaining full in-context retrieval.
:::

::: {.col .fig .big}
![](../img/mdl-modernrnn-hybrid-cache.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Master Comparison of Chapter §{num:chap_modern_rnn} Sequence Mixers (1/2)"}
[Architectural Synthesis]{.kicker}

| Sequence Mixer | Generation State / Layer | Parallel Training Method | Key Synthetic Benchmark Result | WikiText-103 ($d=256$) |
| :--- | :---: | :--- | :--- | :---: |
| **LSTM / GRU (§{sec_lstm})** | $2d$ / $d$ | Sequential ($T$ steps) | $1.00$ at $T=200$ recall with $+5$ keep-gate bias | **$1.29$–$1.30$** ($4.3\text{M}$ / $3.8\text{M}$) |
| **minGRU (§{sec_linear-recurrence})** | $d$ | Associative scan | $1.00$ at $T=1000$ recall with S4D timescales | $1.34$–$1.35$ ($2.8\text{M}$) |
| **S4D (§{sec_ssm})** | $dN$ | FFT conv or scan | $1.00$ at $T=1000$ fixed delay; $\le 0.20$ selective copy | — |
| **Mamba (§{sec_mamba})** | $HN$ ($H=2d$) | Scan + adjoint scan | **$0.98+$** on $1{,}024$-step selective copying | **$1.30$** ($4.0\text{M}$) |
:::

::: {.slide title="Master Comparison of Chapter §{num:chap_modern_rnn} Sequence Mixers (2/2)"}
[Architectural Synthesis]{.kicker}

| Sequence Mixer | Generation State / Layer | Parallel Training Method | Key Synthetic Benchmark Result | WikiText-103 ($d=256$) |
| :--- | :---: | :--- | :--- | :---: |
| **Gated Linear Attn (§{sec_matrix-state})** | $d^2 / h$ | Chunked GEMM + scan | Recall scales $\propto d^2/h$; superposes rebound keys | — |
| **Gated DeltaNet (§{sec_deltanet})** | $d^2 / h$ | Chunked solve + scan | **$0.96+$** on short-gap rebound keys; saturates at $n \gg d_k$ | $1.16$ at $8{,}000$ steps ($3.6\text{M}$) |
| **Causal Attention** | $2dT$ | Flash / fused GEMMs | **$1.00$** at all pair loads; $O(T^2)$ compute & $O(T)$ cache | $1.38$–$1.39$; at $8{,}000$ steps $1.18$, $1.15$ with conv ($3.3\text{M}$) |
| **Hybrid (`GA` / `GGGA`, §{sec_hybrids})** | Fixed states + sparse $2dT$ | Chunked + attention | **$1.00$** at all pair loads with $4\times$–$8\times$ smaller cache | $1.15$ at $8{,}000$ steps ($3.5\text{M}$) |
:::

::: {.slide title="When to Choose Which Sequence Architecture"}
[Design Guidelines]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unbounded Streaming & Edge (`Mamba` / `S4D` / `GRU`):** Best for audio, robotics, and sensor streams requiring strictly constant $O(1)$ memory and latency per step.
- **Long-Context Foundation Models (3:1 or 7:1 Hybrid):** Interleaving chunked `GatedDeltaNet` / `Mamba-2` with sparse GQA layers delivers fast prefill, $4\times$–$8\times$ smaller KV caches, and $100\%$ exact retrieval.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-architecture-decision-tree
:::
:::
:::

::: {.slide title="Concept Check: Welch's Bound and Hybrid Memory"}
[Check Your Understanding]{.kicker}

Work out these three capstone calculations:

1. According to Welch's bound, what is the minimum possible average squared inner product among $m = 33$ unit keys in a head of dimension $d_k = 8$ versus random unit keys?
2. An $80\text{ GB}$ GPU holds $16\text{ GB}$ of model weights, leaving $64\text{ GB}$ for inference caches. At $T = 128\text{K}$ tokens ($512\text{ MB}$ per GQA layer, $2\text{ MB}$ per recurrent layer), how many concurrent $128\text{K}$-token sequences can a 32-layer Transformer serve versus a 32-layer hybrid with $4$ attention layers?
3. Why does a 1-layer matrix memory with `conv_width=1` fail when keys and values arrive on consecutive tokens $(t-1, t)$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Welch floor: $\frac{33 - 8}{8 \times (33 - 1)} = \frac{25}{256} \approx \mathbf{0.0977}$ (compared to $1/8 = \mathbf{0.1250}$ for random unit keys).
2. **32-layer Transformer:** $32 \times 0.5\text{ GB} = 16\text{ GB/seq} \implies 64 / 16 = \mathbf{4\text{ sequences}}$. **32-layer Hybrid ($4$ attention + $28$ recurrent):** $4 \times 512\text{ MB} + 28 \times 2\text{ MB} \approx 2.10\text{ GB/seq} \implies \lfloor 64 / 2.10 \rfloor = \mathbf{30\text{ sequences}}$ ($7.5\times$ higher serving concurrency!).
3. With `conv_width=1`, $\mathbf{k}_t$ and $\mathbf{v}_t$ at step $t$ are functions of $x_t$ (the value token) alone, so $\mathbf{v}_t \mathbf{k}_t^\top$ never sees the key token $x_{t-1}$; a width-4 causal convolution mixes $(x_{t-1}, x_t)$ into the same step.
:::
:::

::: {.slide title="Summary: Hybrid Architectures and Fixed-State Limits"}
[Key Takeaways]{.kicker}

| Principle / Component | Mathematical / Architectural Result | Practical Implication |
| :--- | :--- | :--- |
| **Bit-Counting Bound** | $M b \ge k \log_2 V \implies k_{\max} = O(M)$ | Fixed-state recall capacity is independent of context length $T$ |
| **Welch's Overlap Bound** | $\operatorname{mean}_{i \neq j}(\mathbf{k}_i^\top \mathbf{k}_j)^2 \ge \frac{m - d_k}{d_k(m - 1)}$ for $m > d_k$ | Linear matrix readout degrades once stored keys outnumber head width $d_k$ |
| **State Width Scaling** | $h$ heads of $d_h \times d_h$ hold $d \cdot d_h$ entries | Quadrupling head dimension $d_h$ at fixed width $d$ quadruples pair recall capacity |
| **Short Causal Convolution** | Width-4 depthwise causal `Conv1d` before QKV | Essential for binding adjacent key and value tokens into a single matrix write |
| **Recurrent–Attention Hybrid** | 3:1 or 7:1 `GatedDeltaNet` / `Mamba-2` + GQA | Restores $100\%$ associative recall while cutting $128\text{K}$–$256\text{K}$ KV caches by $4\times$–$8\times$ |
:::

