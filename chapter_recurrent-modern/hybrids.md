# Hybrid Architectures
:label:`sec_hybrids`

Every model in this chapter has improved *how* a fixed-size state is updated—from gates to selective dynamics to error-correcting delta rules—yet none of them can store more information than the state's size allows. When a user pastes a table of 64 product codes and prices into a prompt and asks for 20 of them, a Transformer looks back across its full KV cache and retrieves the exact tokens, whereas a purely recurrent model has only what it compressed into its fixed state before knowing which items would be queried. In-context retrieval—in few-shot prompting, retrieval-augmented generation, and code editing—is where fixed-state sequence models hit a hard ceiling.

The remedy adopted by modern frontier recurrent architectures is pragmatic: interleave a small fraction of full attention layers among the recurrent blocks. Below we establish two fundamental limits on fixed-state recall—an information-theoretic bit-counting bound and Welch's geometric lower bound on key overlap—compare pure `GatedDeltaNet`, pure attention, and a `GA` hybrid on multi-query associative recall and byte-level WikiText-103, ablate the short depthwise convolution that pairs adjacent key and value tokens, quantify inference KV-cache savings at 256K context length, and summarize all sequence memories covered in this chapter.

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
1. **State capacity scaling in `GatedDeltaNet`:** With 128 state entries/layer (`GG`), recall drops from $1.00$ at 4 pairs to $\sim 0.65$ at 32 pairs and $\sim 0.48$ at 64 pairs. Quadrupling the state to 512 entries/layer (`Gated DeltaNet, 4x state`) sustains $0.93$–$0.97$ recall all the way to 64 pairs.
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

Removing the width-4 causal convolution degrades recall at every single load (by $6$ to $38$ percentage points, dropping even 4-pair recall below $0.95$).

## Language Modeling at Equal Size

How do pure `GatedDeltaNet` (`GGGG`), a 3-to-1 hybrid (`GGGA`), and pure causal `attention` (`AAAA`) compare on our byte-level WikiText-103 benchmark (`d2l.SeqLM`, 4 blocks at width $d = 256$, 128-byte windows, 1,000 steps)?

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

On short 128-byte windows, `Gated DeltaNet` (`1.326`) and the `GGGA` hybrid (`1.335`) both outperform pure `attention` (`1.383`) by $0.05$–$0.06$ nats/byte: average next-byte loss on short contexts is dominated by local syntax, while the hybrid's single attention layer stands ready for exact long-range lookup when repeated spans occur (see Exercise 5).

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

Reading across the columns summarizes the design trade-offs:
- For **low-power streaming or continuous sensor/audio processing** where exact lookup of arbitrary past tokens is not required, classical gated recurrences or selective SSMs provide compact $O(1)$-state processing.
- For **long-context foundation models**, interleaving chunk-parallel delta-rule matrix recurrences (`GatedDeltaNet`) or selective SSMs (`Mamba-2`) with a small ratio of full attention layers ($1{:}3$ to $1{:}7$) combines parallel linear-time training, $4\times$–$8\times$ smaller inference KV caches, and exact content-addressable recall.

## Summary

A fixed-size state of $M$ numbers at $b$ bits each can distinguish at most $2^{Mb}$ histories ($Mb \ge k \log_2 V$ to recall $k$ tokens from vocabulary $V$). In matrix-state layers, capacity is constrained even earlier by geometry: by Welch's bound, $m > d_k$ unit keys in $\mathbb{R}^{d_k}$ must have average squared overlap at least $(m - d_k)/(d_k(m - 1))$, causing cross-key interference under both additive and delta-rule updates. On multi-query associative recall, a pure `GatedDeltaNet` model degrades once pair count outgrows its state entries, whereas quadrupling state entries scales capacity proportionally and replacing a single recurrent block with full attention (`GA`) restores $100\%$ recall across all loads. A width-4 causal convolution in front of the mixer is essential so adjacent key and value tokens can bind within a single layer. During generation, only the attention layers of a hybrid require a growing KV cache, cutting inference cache memory in proportion to the fraction of attention layers.

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
    1. Compare the loss of the trained Gated DeltaNet, hybrid, and
       attention models (`trained`) at the marked positions and at all
       other positions. Does attention's advantage on recall show up where
       text repeats itself?
    1. Repeat with windows of 512 bytes (`num_steps=512`, with fewer
       windows per step). How does the share of marked positions change,
       and with it the difference between the models?

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
