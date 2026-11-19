# The Modern Recipe
:label:`sec_modern-recipe`

Open the configuration file that ships with a released language model and
you find a page of settings: the number of layers, the width, the number of
query heads and of key--value heads, the base of the rotary embedding, the
width of the feed-forward network, the epsilon of RMSNorm. A reader who has
worked through this chapter and the previous one has built every one of
them. The file no longer describes a foreign machine. It reads like a call
of our own constructor with unfamiliar arguments.

That leaves the question a practitioner asks of every new model: which of
these settings matter? Some are the same in almost every file, others
change from one family to the next. The answer turns out to be short. Most
settings have converged, and a reader can take them as given. Families
differ in two places, the size of the KV cache and the use of experts, and
both decide what a token costs. This section reads seven families in one
table, counts what their choices cost in bytes and parameters, and builds
two of them with `d2l.GPT`, next to GPT-2 for contrast. It adds no
component; it reads the chapter's block as released models configure it.

```{.python .input #modern-recipe-the-modern-recipe-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #modern-recipe-the-modern-recipe-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## A Configuration File

Let us begin with one file, that of Llama 3 8B
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Each of its 32 layers runs 32
query heads of width 128, and every group of four query heads shares one
key--value head, so a layer has 8 key--value heads. This sharing is
grouped-query attention (:numref:`sec_kv-cache-reduction`). How much
memory does one token of context take? Every layer stores one key and one
value per key--value head, at two bytes per number in bf16, so the cache
grows by

$$
2 \times 32 \times 8 \times 128 \times 2\ \textrm{bytes} = 131{,}072\
\textrm{bytes} = 128\ \textrm{KiB}
$$

per token. The factors are keys and values, layers, key--value heads, head
width and bytes per number: the cache formula of
:numref:`subsec_kv-cache-memory` for a single token. A context of 128K
tokens therefore holds 16 GiB, more than the 15 GiB that the model's 8.03
billion weights take in the same precision. Had every query head kept its
own key and value, as in GPT-2, the cache would be four times larger: 512
KiB per token, 64 GiB for that context. One integer in the file, the number
of key--value heads, moves the memory of a long context by a factor of
four. No line about normalization or activations moves it at all.

## Seven Open-Weights Families

Llama 3 is one family of many. :numref:`tab_modern-recipe` collects the
architecture sections of the technical reports of seven open-weights
families: Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023`, Llama 3,
Qwen3 :cite:`Yang.Li.Yang.ea.2025`, OLMo 2 and 3
:cite:`OLMo.2025,OLMo3.2025`, DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024`,
Gemma 3 :cite:`Gemma.Team.2025`, and gpt-oss :cite:`OpenAI.2025`. In the
attention column, GQA $H{:}H_\textrm{kv}$ means $H$ query heads sharing
$H_\textrm{kv}$ key--value heads.

:Architectures of seven open-weights model families, as reported in their technical reports, 2023 to 2025. Attention and cache: :numref:`sec_kv-cache-reduction`. Normalization: :numref:`sec_transformer-block`. Positions: :numref:`sec_positional-information`. Mixture of experts: :numref:`sec_moe`.
:label:`tab_modern-recipe`

| model | attention and cache | normalization | positions | FFN | dense or MoE |
|:--|:--|:--|:--|:--|:--|
| Mistral 7B (2023) | GQA $32{:}8$; sliding window 4096 | RMSNorm, pre | RoPE, base $10^4$ | SwiGLU | dense |
| Llama 3 (2024) | GQA $32{:}8$ (8B), $64{:}8$ (70B), $128{:}8$ (405B) | RMSNorm, pre | RoPE, base $5 \times 10^5$ | SwiGLU | dense |
| Qwen3 (2025) | GQA: $64{:}8$ dense (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, pre; QK-norm | RoPE, base $10^6$; YaRN for long context | SwiGLU | dense, and MoE: 128 experts, 8 active |
| OLMo 2/3 (2024/25) | MHA (7B), GQA (32B); OLMo 3: window 4096 on 3 of 4 layers | RMSNorm on sublayer outputs; QK-norm | RoPE | SwiGLU | dense |
| DeepSeek-V3 (2024) | MLA: KV compressed to a 512-dim latent | RMSNorm, pre | RoPE on a separate 64-dim key shared by all heads | SwiGLU | MoE: 256 experts plus 1 shared, 8 active |
| Gemma 3 (2025) | GQA; local:global $5{:}1$, window 1024 | RMSNorm, pre and post; QK-norm | RoPE, base $10^6$ global, $10^4$ local | GeGLU | dense |
| gpt-oss (2025) | GQA $64{:}8$; window 128 on alternate layers; learned sinks | RMSNorm, pre | RoPE; YaRN | SwiGLU, clamped | MoE: 128 or 32 experts, 4 active |

Every row is the block of :numref:`sec_transformer-block` in the causal
wiring of :numref:`sec_gpt`, and the rows agree on most of its settings.
All of them use rotary positions and a gated feed-forward network. (GeGLU
is SwiGLU with GELU in place of SiLU, and gpt-oss clamps the inputs of its
SwiGLU to a fixed range.) All of them normalize with RMSNorm what enters a
sublayer or what leaves it, and none normalizes the residual stream
itself: most normalize the input (pre-norm), OLMo the output, and Gemma 3
both. None uses the post-norm of the original transformer, which
normalizes the stream after every addition. Two settings need no column,
because the rows barely differ on them: only gpt-oss keeps biases, in its
attention, and no row uses dropout. Dropout is absent for the reason
:numref:`sec_gpt` gave for our own model: a model that sees almost no
passage twice has little overfitting for dropout to prevent.

Why have these settings converged? Our reading is that they are
cheap to test, since a small model already shows their effect. Our own
model of 4.8 million parameters settled two of them in :numref:`sec_gpt`.
Without a warmup, the post-norm stack stalled at the unigram level where
pre-norm trained, and SwiGLU ended about 0.06 nats below GELU. Settings
that small experiments can decide have been decided, and the columns that
hold them can be read quickly.

The rows differ on two settings. The first is how far each family shrinks
the KV cache: by the group size of grouped-query attention, by sliding
windows on some or all layers in four rows, and by a compressed latent in
DeepSeek-V3, all from :numref:`sec_kv-cache-reduction`. The second is
capacity per unit of computation: three of the seven families replace the
feed-forward network by a mixture of experts (:numref:`sec_moe`). The best
value of either depends on how a model will be used, on the length of its
contexts, the size of its batches and the memory of the machines that
serve it, so it is no surprise that families still choose differently. The
settings the rows share decide how well a model learns; the settings on
which they differ decide what each token costs.

This gives an order in which to read a new model's file when sizing a
deployment. Read the attention column first, since for long contexts the
cache can outgrow the weights, as it does for Llama 3 8B. Then read the
last column: a mixture of experts needs memory for every parameter it
stores but computation only for those it runs.

## Mixtures of Experts in Released Models

:numref:`sec_moe` built a layer that stores $E$ experts and runs $k$ of
them for each token, so that the stored parameters grow with $E$ and the
computation per token with $k$. :numref:`tab_moe-experts` lists how four
released models set these two numbers, how wide they make each expert
relative to the model width $d$, and the ratio of the parameters they store
to those one token uses. The first row is Mixtral 8x7B, whose 46.7 billion
stored and 12.9 billion active parameters :numref:`sec_moe` counted by
hand.

:Routed-expert configurations of released MoE language models.
:label:`tab_moe-experts`

| Model | Experts per layer | Active $k$ | Expert width / $d$ | Shared expert | Stored / active parameters |
|---|---|---|---|---|---|
| Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` | 8 | 2 | 3.5 | none | 46.7B / 12.9B = 3.6 |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | 256 | 8 | 0.29 | 1 | 671B / 37B = 18 |
| Qwen3-235B :cite:`Yang.Li.Yang.ea.2025` | 128 | 8 | 0.38 | none | 235B / 22B = 11 |
| gpt-oss-120b :cite:`OpenAI.2025` | 128 | 4 | 1.0 | none | 117B / 5.1B = 23 |

The three later models store many more experts than Mixtral, 128 or 256
per layer, and run 4 to 8 of them per token. The last column can be
predicted from the two before it. If the experts were the whole model, the
ratio of stored to active parameters would be $E/k$: 4 for Mixtral, 32 for
DeepSeek-V3, 16 for Qwen3 and 32 for gpt-oss. Attention, the embeddings and
any shared expert run for every token, so they pull the ratio below $E/k$,
to 3.6 for Mixtral and 11 to 23 for the later models. A token of
gpt-oss-120b runs about one parameter in twenty-three.

Two of the three, DeepSeek-V3 and Qwen3, also make each expert narrower
than the model itself, the fine-grained design of
:numref:`subsec_moe-fine-grained`. Narrow experts let a model run more of
them per token for the same computation. They do not raise the ratio by
themselves: gpt-oss, whose ratio is the highest, keeps its experts as wide
as the model. The models also disagree on the shared expert. DeepSeek's
models have used shared experts since DeepSeekMoE, two in DeepSeek-V2
:cite:`DeepSeek-AI.2024` and one in DeepSeek-V3, whereas Qwen3 and gpt-oss
use none. How many experts a model should have, how wide, and how many
tokens it needs before its stored parameters help are questions of
scaling, of the kind that :numref:`sec_scaling-laws` studied for dense
models. For mixtures the answer is not settled. Four rows give four
different designs, which is a fair summary of the state of the question.

## Configurations of Our Block

The block that every row shares is the one we built, so rows of the table
become calls of `d2l.GPT`. We build them at the size of our byte-level
model, width $d = 256$ with 8 query heads and six blocks, through the same
shorthand as :numref:`sec_scaling-laws`, so the default row is the
4.8-million-parameter model of :numref:`sec_gpt` itself.

```{.python .input #modern-recipe-the-modern-recipe-2}
%%tab pytorch
gpt = d2l.GPT
```

```{.python .input #modern-recipe-the-modern-recipe-2}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

The defaults of `d2l.GPT` are the settings the table shares: pre-norm
RMSNorm, SwiGLU, rotary positions and no biases. A modern row therefore
needs one flag, the number of key--value heads. Llama 3 8B gives each
key--value head four query heads, which with our 8 query heads means
`num_kv_heads=2`; Qwen3-32B gives it eight, so `num_kv_heads=1`. GPT-2,
from 2019, predates most of the shared settings. It needs four flags to
undo the defaults, for learned positions, LayerNorm, GELU and biases, and
its row measures what the changes to the block since then did to the
parameters and the cache. The cell below prints each row's parameters and
its cache per token, both against the default model.

Before running it, let us predict both columns, starting with the easier
one. Each of the six blocks stores a key and a value of width 256 per
token, in 4-byte floats, so the default model caches $2 \times 6 \times
256 \times 4 = 12{,}288$ bytes, or 12 KiB, per token. Sharing divides this
by the size of the group: 3 KiB for Llama 3 8B, 1.5 KiB for Qwen3-32B.

The parameters take one more step. Of the $12d^2$ parameters of a block
(:numref:`sec_transformer-block`), $4d^2$ sit in the four attention
projections, and sharing shrinks only two of those matrices, $\mathbf{W}_k$
and $\mathbf{W}_v$. Groups of four shrink each from $d \times d$ to
$d \times d/4$, which saves $2 \cdot \tfrac{3}{4} d^2 = 1.5 d^2$ per block, an
eighth of the block; groups of eight save $1.75 d^2$. Over six blocks of width
256 that comes to 0.59 and 0.69 million parameters.

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab pytorch, jax
recipes = {'default': {},
           'GPT-2': dict(pos='learned', norm='layer', act='gelu',
                         bias=True),
           'Llama 3 8B': dict(num_kv_heads=2),
           'Qwen3-32B': dict(num_kv_heads=1)}
counts = {}
for name, cfg in recipes.items():
    model = gpt(256, num_hiddens=256, num_heads=8, num_blks=6, **cfg)
    counts[name] = d2l.num_params(model), d2l.kv_cache_bytes(model, 1)
n_0, kv_0 = counts['default']
print(f'{"model":10s} {"parameters":>10s} {"vs default":>11s} '
      f'{"cache/token":>13s} {"vs default":>12s}')
for name, (n, kv) in counts.items():
    print(f'{name:10s} {n / 1e6:9.2f}M {n / n_0 - 1:+11.0%} '
          f'{kv / 1024:9.1f} KiB {kv / kv_0:12.3f}')
```

Both predictions hold. The default model has 4.79 million parameters and
caches 12.0 KiB per token. The Llama 3 8B row has $4.79 - 4.20 = 0.59$
million fewer parameters and the Qwen3-32B row $4.79 - 4.10 = 0.69$
million fewer, 12% and 14% of the model, while their caches are a quarter
and an eighth of the default one. The two savings differ so much because
they are fractions of different things. The cache holds nothing but keys
and values, so sharing shrinks all of it. The parameters hold keys and
values in two of the twelve $d^2$ of a block, so no amount of sharing can
remove more than a sixth of them.

The parameters are not what makes sharing attractive, though. The quality
is, and this cell does not measure it. In
:numref:`sec_kv-cache-reduction`, our model trained with two key--value
heads from the start ended at the 1.34 to 1.35 nats per byte of the model
with eight, in every run, and repeated runs of that model differ by about
0.01 nats. Few changes to a transformer save so much memory at so little
loss, and every family in the table makes some version of this one: six
of the seven share key--value heads, and the seventh, DeepSeek-V3,
compresses keys and values into a latent instead.

GPT-2's row moves the other way. Its biases and its table of learned
positions add 6% to the parameters, almost all of it in the table, and
none of its four flags touches the cache. Measured in these two columns,
five years of changes to the block moved our model by 6% of its parameters
and not at all in its cache, while the single flag of the Llama row
divides the cache by four.

The remaining columns of :numref:`tab_modern-recipe` are not flags of
`d2l.GPT`, but none of them needs a new block either.
`d2l.TransformerBlock` takes two hooks, `attn_factory` for the attention
sublayer and `ffn_factory` for the feed-forward network; `d2l.GPT` fills
the first from its class attribute `CausalAttention` and passes the second
on to its blocks. QK-norm (:numref:`subsec_qk-norm`) normalizes queries
and keys inside attention, so it enters as a new attention class: a
subclass of `d2l.GPT` replaces `CausalAttention`, and an exercise of
:numref:`sec_transformer-block` builds such an attention. A sliding window
changes the mask that the attention applies, and it saves memory only if
the cache also evicts what the mask no longer reaches, as a rolling buffer
does (:numref:`subsec_kv-cache-sinks`). Sink tokens kept at the start of
the text change only which positions the mask keeps. The learned sinks of
gpt-oss are a different device: one learned logit per head, added to the
denominator of the softmax. A latent cache stores one compressed vector
per token in place of the keys and values and reconstructs them when it
attends; :numref:`subsec_kv-cache-low-rank` built the same idea with a
fixed basis. The experts of the MoE rows enter through `ffn_factory`, as
in :numref:`sec_moe`.

The newest designs have begun to change the block itself. Hybrid models
keep full attention in a fraction of their layers and replace the rest by
linear recurrences whose state does not grow with the context
:cite:`Kimi.Team.2025b`, and Llama 4 removes the positional encoding from
some of its attention layers :cite:`Meta.2025`. Whether either will join
the shared recipe, nobody knows yet. The first of them, recurrence
returning to the stack, is the subject of :numref:`chap_modern_rnn`.

## Summary

Released open-weights models are the chapter's block in its causal wiring,
and most of its settings have converged: RMSNorm that leaves the residual
stream unnormalized, a gated feed-forward network, rotary positions, no
dropout, and biases only in the attention of gpt-oss. The families differ
in the two settings that decide what a token costs, the size of the KV
cache and the use of experts. Llama 3 8B caches 128 KiB per token with 8
key--value heads, a quarter of what a key--value head per query head would
need, and recent mixtures store 11 to 23 times the parameters that a token
uses. Sharing key--value heads shrinks the cache by the full group size
but the parameters by at most a sixth, 12% and 14% at our width, and two
key--value heads trained from the start lost nothing that our runs could
measure. The newest designs, hybrids with linear recurrences and layers
without positions, depart from the block itself.

## Exercises

1. **A window against a long context.** Mistral 7B has the cache settings
   of Llama 3 8B (32 layers, 8 key--value heads of width 128, bf16), but
   every layer attends only to the last 4,096 tokens. Its cache therefore
   stops growing at 512 MiB per sequence, where that of Llama 3 8B reaches
   16 GiB at 128K tokens. A token outside the window can still affect the
   output, through the positions that attended to it in earlier layers.
    1. How far back can information reach through 32 such layers?
    1. Why is this reach an upper bound and not a guarantee? What would you
       measure to find how far information actually travels?

1. **Llama 3 8B from its settings.** Llama 3 8B has a vocabulary of
   128,256 tokens, width 4,096, 32 layers, 32 query heads of width 128
   sharing 8 key--value heads, a SwiGLU network of width 14,336, RMSNorm,
   no biases, and separate input and output embeddings.
    1. Count the parameters of the embeddings, of one block's attention, of
       one block's feed-forward network, and of the whole model. Compare
       with the reported 8.03 billion.
    1. What fraction of the parameters sits in the embeddings? Use the
       comparison of $Vd$ with $12Ld^2$ from :numref:`sec_scaling-laws` to
       explain why it is so much larger than in our byte-level model.
    1. How many parameters would the model have with a key--value head per
       query head? Compare the relative change with the 12% of our cell.

1. **Stored and active parameters of Qwen3-235B.** The MoE model of Qwen3
   has a vocabulary of 151,936, width 4,096, and 94 layers. Each layer has
   64 query heads of width 128 sharing 4 key--value heads, and 128 SwiGLU
   experts of width 1,536, of which a router selects 8 per token; there is
   no shared expert, and input and output embeddings are separate.
    1. Count the stored and the active parameters, including the router,
       and compare with the 235B and 22B of :numref:`tab_moe-experts`.
    1. What fraction of the stored parameters sits in the experts?
    1. In bf16, how many bytes must a server hold for the weights, and how
       many must it read to produce one token when the batch holds a single
       sequence?

1. **Which cache is smallest?** Consider a model with 36 layers and 8
   key--value heads of width 128 in bf16, so that full attention costs
   4 KiB per token and layer. Compare four layouts: (a) full attention in
   every layer; (b) the pattern of Gemma 3, five layers with a window of
   1,024 tokens for every layer with full attention; (c) the pattern of
   gpt-oss, a window of 128 tokens on alternate layers; (d) a latent of 576
   numbers per token and layer in place of the keys and values, as in
   DeepSeek-V3 (512 for the latent and 64 for the rotary key).
    1. For each layout, compute the cache per token of context for
       sequences of 1K, 32K and 128K tokens.
    1. Which layout is smallest at 1K tokens, and which at 128K? At what
       length do (b) and (d) cross?
    1. None of these numbers shows what each layout gives up. Name it for
       each, using :numref:`sec_kv-cache-reduction`.

1. **A cache budget.** You serve a model with 32 layers and 32 query heads
   of width 128 in bf16, and 9 GiB of memory remain for the cache. You must
   hold 8 sequences of 32K tokens at once.
    1. With full attention in every layer, what is the largest number of
       key--value heads (a divisor of 32) that fits?
    1. Now put a window of 4,096 tokens on three of every four layers, as
       OLMo 3 does. What is the largest number now?
    1. Which of the two designs would you choose? What evidence from
       :numref:`sec_kv-cache-reduction` would you want before deciding?

<!-- Discussions: a new Discourse topic is needed for this page (tools/create_forum_topics.py --create). -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.9]{.kicker}

The Modern Recipe<br>
**one block, a handful of settings · the cache · experts**
:::
:::

::: {.slide title="At 128K Tokens, Llama 3 8B's Cache Outweighs Its Weights"}
[The problem]{.kicker}

| Llama 3 8B, bf16 | bytes |
|:--|--:|
| weights: 8.03B parameters | 15 GiB |
| KV cache of one 128K-token sequence | 16 GiB |
| the same cache with 32 key--value heads instead of 8 | 64 GiB |

::: {.d2l-note .warn}
A model's configuration file lists a dozen settings. One integer in it, the
number of key--value heads, moves the memory of a long context fourfold.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read a released model as settings of the chapter's one block;
- say which settings have converged and which still differ, and why;
- compute the KV cache per token from a row, and stored against active
  parameters for a mixture of experts;
- predict what sharing key--value heads saves, and check it with
  `d2l.GPT`.

**Running example:** Llama 3 8B; then the default `d2l.GPT`, GPT-2's
settings, and the Llama 3 8B and Qwen3-32B rows at our width of 256.
:::

::: {.slide title="Every Component Is Built; Released Models Are Settings of Them"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN, residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse past keys and values |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| Mixture of Experts | more parameters per FLOP |
| **The Modern Recipe** | **released models as settings** |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT` and its flags.
- **Skips:** training data and post-training (Language Models part).
:::
:::
:::

::: {.slide title="Llama 3 8B Caches 128 KiB per Token, a Quarter of Full Heads"}
32 query heads per layer, in groups of four; each group shares one
key--value head, so a layer has 8. Per token, every layer stores one key
and one value per key--value head:

$$\underbrace{2}_{K,\,V} \times \underbrace{32}_{\textrm{layers}} \times
\underbrace{8}_{\textrm{KV heads}} \times \underbrace{128}_{\textrm{head
width}} \times \underbrace{2\ \textrm{B}}_{\textrm{bf16}} = 128\
\textrm{KiB}$$

- One key--value head per query head (32): 512 KiB per token.
- No setting for normalization or activations changes this number.

::: {.d2l-note}
For long contexts, the cache, not the weights, sets the memory.
:::
:::

::: {.slide title="Open Models Share One Block and Differ in the Cache and the FFN"}
| model | attention + cache | norm | FFN / experts |
|:--|:--|:--|:--|
| Mistral 7B | GQA 32:8, window 4096 | RMS pre | SwiGLU, dense |
| Llama 3 8B | GQA 32:8 | RMS pre | SwiGLU, dense |
| Qwen3 | GQA 64:8 dense, 64:4 MoE | RMS pre + QK | SwiGLU, dense & MoE 128/8 |
| DeepSeek-V3 | MLA: 512-d latent | RMS pre | SwiGLU, MoE 256+1/8 |
| Gemma 3 | GQA, local:global 5:1 | RMS pre+post + QK | GeGLU, dense |
| gpt-oss | GQA 64:8, window 128 on half the layers, learned sinks | RMS pre | SwiGLU, MoE 128/4 |

::: {.d2l-note}
Shared settings (also RoPE, no dropout) decide how well a model learns;
the differing ones decide what each token costs.
:::

[GQA $H{:}H_\textrm{kv}$: $H$ query heads, $H_\textrm{kv}$ key--value
heads. MoE $E/k$: experts stored / run per token, +1 shared.]{.small}
:::

::: {.slide title="Recent Mixtures Store 11 to 23 Times What a Token Uses"}
| model | $E$ | $k$ | width$/d$ | shared | stored / active |
|---|--:|--:|--:|---|--:|
| Mixtral 8x7B | 8 | 2 | 3.5 | none | 46.7B / 12.9B = 3.6 |
| DeepSeek-V3 | 256 | 8 | 0.29 | 1 | 671B / 37B = 18 |
| Qwen3-235B | 128 | 8 | 0.38 | none | 235B / 22B = 11 |
| gpt-oss-120b | 128 | 4 | 1.0 | none | 117B / 5.1B = 23 |

- Experts alone would give stored/active $= E/k$: 4, 32, 16, 32.
- Attention, embeddings and shared experts run for every token and pull
  the ratio lower.
- Narrow experts ($d$: model width) let $k$ grow at equal compute; gpt-oss,
  with the highest ratio, keeps wide ones.

::: {.d2l-note}
Memory for every stored parameter, computation only for the active ones.
:::
:::

::: {.slide title="Check: What Does Sharing Save at Width 256?"}
[Check]{.kicker}

Width $d = 256$, 8 query heads, six blocks, fp32. Groups of four, or of
eight, query heads share one key--value head. How much smaller are the
cache and the parameters?

. . .

- Cache per token: $2 \times 6 \times 256 \times 4\ \textrm{B} = 12$ KiB,
  divided by the group size: 3 and 1.5 KiB.
- Parameters: a block holds about $12d^2$, of which $\mathbf{W}_k$ and
  $\mathbf{W}_v$ are $2d^2$.
- Groups of 4 save $1.5d^2$ per block: 0.59M; groups of 8, $1.75d^2$:
  0.69M.

::: {.d2l-note}
The cache is all keys and values; the parameters are not, so sharing saves
at most a sixth of them.
:::
:::

::: {.slide title="Sharing Shrinks the Cache 4× and 8×, the Parameters 12–14%"}
@!scaling-laws-recipe-rows-as-constructor-calls

- GPT-2 (2019): learned positions, LayerNorm, GELU, biases: +6%
  parameters, same cache.
- Llama 3 8B and Qwen3-32B: the default plus `num_kv_heads`.
- Quality: two key--value heads trained from the start end at the loss of
  eight, 1.34 to 1.35 nats per byte (Shrinking the KV Cache).
:::

::: {.slide title="The Other Settings Enter Through Two Hooks"}
Each block takes `attn_factory` (the attention sublayer) and `ffn_factory`
(the FFN); `d2l.GPT` fills the first from its class `CausalAttention`.

| setting | where it enters |
|:--|:--|
| QK-norm | a new attention class: a subclass replaces `CausalAttention` |
| sliding window, kept sink tokens | a different mask, and a cache that evicts old entries |
| learned sinks (gpt-oss) | one learned logit per head in the softmax denominator |
| latent cache (MLA) | an attention module that caches one latent per token |
| experts | a mixture layer, through `ffn_factory` |

::: {.d2l-note}
Nothing in the table needs a new block: only new attention and FFN
modules.
:::
:::

::: {.slide title="Check: Llama 3 8B's Parameter Count from Its Settings"}
[Check]{.kicker}

Vocabulary 128,256, width 4,096, 32 layers, GQA 32:8 with head width 128,
SwiGLU width 14,336, separate input and output embeddings. How many
parameters?

. . .

- Embeddings: $2 \times 128{,}256 \times 4{,}096 = 1.05$B.
- Per layer: attention $2 \cdot 4096^2 + 2 \cdot 4096 \cdot 1024 = 42$M
  (keys and values: 8 heads × 128 = 1,024 wide), SwiGLU
  $3 \cdot 4096 \cdot 14{,}336 = 176$M.
- $1.05\textrm{B} + 32 \times 218\textrm{M} = 8.03$B.
:::

::: {.slide title="Released Models Share the Block and Differ in What a Token Costs"}
[Wrap-up]{.kicker}

- Released models are the chapter's block: RMSNorm (mostly pre-norm),
  gated FFN, RoPE, mostly no biases, no dropout.
- They differ in what a token costs: the cache (Llama 3 8B: 128 KiB per
  token, a quarter of full heads) and experts (11 to 23 times more stored
  than active).
- At our width, `num_kv_heads` cuts the cache 4× and 8× for 12% and 14%
  fewer parameters; two key--value heads trained from the start lose
  nothing our runs could measure.
- Sizing a deployment: read the attention and cache column first, then the
  experts.

**Next:** hybrids that replace most attention layers with linear
recurrences.
:::
