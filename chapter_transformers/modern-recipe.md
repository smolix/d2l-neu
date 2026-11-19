# The Modern Recipe
:label:`sec_modern-recipe`

<!-- WRITER: new opening (structure.md section 3.9), from the reader, in the
voice of a chapter close. The body below moved from the end of scaling-laws.md
("Configurations of Deployed Models"), cell id kept; its first sentence
("The models trained on such budgets") and its last ("trained at budgets that
the law of this section helps to plan", SL-19) still read as the end of the
scaling section. tab_moe-experts and its paragraph arrived from moe.md
(verbatim, plus the stored/active column of MOE-20); "Departures from the
Shared Block" is still in further-reading.md and moves here as the closing
(writer). Small case for the opening, checked: Llama 3 8B in bf16 caches
32 layers x 8 KV heads x 128 x 2 (keys and values) x 2 bytes = 131,072 bytes
= 128 KiB per token, a quarter of the 512 KiB that 32 KV heads would need;
16 GiB at 128K tokens. -->

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

```{.python .input #modern-recipe-the-modern-recipe-2}
%%tab pytorch
gpt = d2l.GPT
```

```{.python .input #modern-recipe-the-modern-recipe-2}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

The models trained on such budgets are built from the components of this
chapter. :numref:`tab_modern-recipe` compiles the architecture sections of
seven open-weights families: Mistral 7B
:cite:`Jiang.Sablayrolles.Mensch.ea.2023`, Llama 3
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, Qwen3
:cite:`Yang.Li.Yang.ea.2025`, OLMo 2 and 3 :cite:`OLMo.2025,OLMo3.2025`,
DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024`, Gemma 3
:cite:`Gemma.Team.2025`, and GPT-OSS :cite:`OpenAI.2025`.

:Architectures of seven open-weights model families, as reported in their technical reports (2023–2025). Attention and cache: :numref:`sec_kv-cache`. Normalization: :numref:`sec_transformer-block`. Positions: :numref:`sec_positional-information`. Mixture of experts: :numref:`sec_moe`.
:label:`tab_modern-recipe`

| model | attention and cache | normalization | positions | FFN | dense or MoE |
|:--|:--|:--|:--|:--|:--|
| Mistral 7B (2023) | GQA $32{:}8$; sliding window 4096 | RMSNorm, pre | RoPE, $\theta = 10^4$ | SwiGLU | dense |
| Llama 3 (2024) | GQA $32{:}8$ (8B), $64{:}8$ (70B), $128{:}8$ (405B) | RMSNorm, pre | RoPE, $\theta = 5 \times 10^5$ | SwiGLU | dense |
| Qwen3 (2025) | GQA: $64{:}8$ dense (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, pre; QK-norm | RoPE, $\theta = 10^6$; YaRN for long context | SwiGLU | dense, and MoE: 128 experts, 8 active |
| OLMo 2/3 (2024/25) | MHA (7B), GQA (32B); OLMo 3: window 4096 on 3 of 4 layers | RMSNorm on sublayer outputs; QK-norm | RoPE | SwiGLU | dense |
| DeepSeek-V3 (2024) | MLA: KV compressed to a 512-dim latent | RMSNorm, pre | RoPE on a decoupled 64-dim slice per head | SwiGLU | MoE: 256 experts plus 1 shared, 8 active |
| Gemma 3 (2025) | GQA; local:global $5{:}1$, window 1024 | RMSNorm, pre and post; QK-norm | RoPE, $\theta = 10^6$ global, $10^4$ local | GeGLU | dense |
| GPT-OSS (2025) | GQA $64{:}8$; window 128 on alternate layers; learned sinks | RMSNorm, pre | RoPE; YaRN | SwiGLU, clamped | MoE: 128 or 32 experts, 4 active |

Every row is the block of :numref:`sec_transformer-block` in the causal
wiring of :numref:`sec_gpt`, and the rows agree on most of its settings: a
gated feed-forward network, RoPE, RMSNorm with an unnormalized residual
stream, no biases in most rows, and no dropout in any. (OLMo normalizes each
sublayer's output before it joins the stream rather than its input, which
also leaves the stream itself unnormalized.) The absence of dropout is a
consequence of scale: a model that reads trillions of tokens about once has nothing to
memorize. The rows differ on two axes. The first is the size of the KV
cache: grouped-query attention in most rows, sliding windows on some
layers, and a compressed latent in DeepSeek-V3, all from
:numref:`sec_kv-cache`.
The second is capacity per FLOP: three of the seven families use a mixture
of experts (:numref:`sec_moe`). The newest designs depart further. Hybrid
models replace most attention layers with the linear recurrences of
:numref:`chap_modern_rnn` :cite:`Kimi.Team.2025b`, and Llama 4 removes
positional encoding from some of its attention layers :cite:`Meta.2025`.

Released models make different choices along these axes, as
:numref:`tab_moe-experts` shows.

<!-- WRITER: the lead-in sentence above came from moe.md, where "these axes"
were the number and width of experts; here it needs its own antecedent.
MOE-20: "most of them" in the paragraph after the table is two of the three
later models (DeepSeek-V3 and Qwen3 have experts narrower than d; gpt-oss
has width 1.0). The new last column is each model's reported total over its
reported active parameters. -->

:Routed-expert configurations of released MoE language models.
:label:`tab_moe-experts`

| Model | Experts per layer | Active $k$ | Expert width / $d$ | Shared expert | Stored / active parameters |
|---|---|---|---|---|---|
| Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` | 8 | 2 | 3.5 | none | 46.7B / 12.9B = 3.6 |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | 256 | 8 | 0.29 | 1 | 671B / 37B = 18 |
| Qwen3-235B :cite:`Yang.Li.Yang.ea.2025` | 128 | 8 | 0.38 | none | 235B / 22B = 11 |
| gpt-oss-120b :cite:`OpenAI.2025` | 128 | 4 | 1.0 | none | 117B / 5.1B = 23 |

The three later models use many more experts than Mixtral and between 4
and 8 active per token, and most of them make each expert narrower than
the model itself. They disagree on the shared expert: DeepSeek's models
have used shared experts since DeepSeekMoE, two in DeepSeek-V2
:cite:`DeepSeek-AI.2024` and one in DeepSeek-V3, whereas Qwen3 uses none.
How many experts a model should have, how wide, and how many tokens it
needs before its stored parameters help are questions of scaling, of the
kind that :numref:`sec_scaling-laws` asks for dense models.

At width 256 and six blocks, three rows of the table become calls of our
constructor. The modern rows need only the number of key--value heads,
because the defaults of `d2l.GPT` are the recipe the table shares. Llama 3
8B shares each key--value head among 4 query heads, Qwen3-32B among 8. The
cell prints the parameter count and the bytes of KV cache per token in
32-bit floats:

<!-- WRITER: the cell now has a fourth row, the default `d2l.GPT` with no
flags (SL-18), and prints each row against it. Both tabs print:
default 4.99M parameters (+0%), 12.0 KiB of cache per token (1.000);
GPT-2 5.26M (+6%), 12.0 KiB (1.000); Llama 3 8B 4.40M (-12%), 3.0 KiB
(0.250); Qwen3-32B 4.30M (-14%), 1.5 KiB (0.125). "three rows" becomes the
default and three rows; the shape check is gone from the cell. -->

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab pytorch, jax
recipes = {'default': {},
           'GPT-2': dict(pos='learned', norm='layer', act='gelu',
                         bias=True),
           'Llama 3 8B': dict(num_kv_heads=2),
           'Qwen3-32B': dict(num_kv_heads=1)}
counts = {}
for name, cfg in recipes.items():
    model = gpt(1024, num_hiddens=256, num_heads=8, num_blks=6, **cfg)
    counts[name] = d2l.num_params(model), d2l.kv_cache_bytes(model, 1)
n_0, kv_0 = counts['default']
print(f'{"model":10s} {"parameters":>10s} {"vs default":>11s} '
      f'{"cache/token":>13s} {"vs default":>12s}')
for name, (n, kv) in counts.items():
    print(f'{name:10s} {n / 1e6:9.2f}M {n / n_0 - 1:+11.0%} '
          f'{kv / 1024:9.1f} KiB {kv / kv_0:12.3f}')
```

Sharing key--value heads removes part of the key and value projections, a
tenth of the parameters or so at this width, but it shrinks the cache by
the full factor of the sharing, 4 and 8. The settings the flags do not
reach would enter as new attention modules. QK-norm
(:numref:`subsec_qk-norm`) enters through the `attn_factory` hook of
`TransformerBlock`, as in an exercise of :numref:`sec_transformer-block`.
A sliding window is a mask (:numref:`sec_windowed-linear-attention`,
:numref:`sec_kv-cache`). A learned latent cache replaces the stored
keys and values, which :numref:`sec_kv-cache` sketches with a fixed basis.
The expert layers of the MoE rows enter through `ffn_factory`
(:numref:`sec_moe`). Deployed models are thus
configurations of the chapter's one block, trained at budgets that the law
of this section helps to plan.

<!-- WRITER: "a tenth of the parameters or so" is now printed: 12% (Llama 3
8B) and 14% (Qwen3-32B) below the default, against cache factors 0.25 and
0.125. The latent cache is now in sec_kv-cache-reduction, not sec_kv-cache
(check both pointers in this paragraph and in the table caption). Closing:
"Departures from the Shared Block" (further-reading.md) moves in here,
shortened (structure 3.9 item 5). -->

## Summary

<!-- WRITER: summary of the section (structure.md section 3.9). -->

<!-- Discussions: a new Discourse topic is needed for this page (tools/create_forum_topics.py --create). -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.9]{.kicker}

The modern recipe<br>
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
A model card lists a dozen settings. One of them, the number of
key--value heads, decides whether the model serves long contexts on one GPU.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read a released model as settings of the chapter's one block;
- say which settings have converged and which still differ;
- compute the KV cache per token from a row, and stored against active
  parameters for a mixture of experts;
- build the rows with `d2l.GPT` flags and count what they change.

**Running example:** GPT-2 small against Llama 3 8B, and both at our
width of 256.
:::

::: {.slide title="Every Component Is Built; Released Models Are Settings of Them"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model |
| The KV Cache | reuse keys and values |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, ... | the other two wirings |
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

::: {.slide title="Open Models Share One Block and Differ in the Cache and the FFN"}
| model | attention + cache | norm | FFN / experts |
|:--|:--|:--|:--|
| Mistral 7B | GQA 32:8, window 4096 | RMS pre | SwiGLU, dense |
| Llama 3 8B | GQA 32:8 | RMS pre | SwiGLU, dense |
| Qwen3 | GQA 64:8 dense, 64:4 MoE | RMS pre + QK | SwiGLU, dense & MoE 128/8 |
| DeepSeek-V3 | MLA: 512-d latent | RMS pre | SwiGLU, MoE 256+1/8 |
| Gemma 3 | GQA, local:global 5:1 | RMS pre+post + QK | GeGLU, dense |
| GPT-OSS | GQA 64:8, window 128 alt., sinks | RMS pre | SwiGLU, MoE 128/4 |

All: RoPE, **no dropout**: a model that sees almost no token twice has
little overfitting for dropout to prevent.

[GQA $h{:}g$: $h$ query heads share $g$ key--value heads. As reported in
the technical reports, 2023 to 2025.]{.small}
:::

::: {.slide title="Llama 3 8B Caches 128 KiB per Token, a Quarter of Full Heads"}
Per token, every layer stores one key and one value per key--value head:

$$\underbrace{32}_{\textrm{layers}} \times \underbrace{8}_{\textrm{KV heads}}
\times \underbrace{128}_{\textrm{head width}} \times
\underbrace{2}_{K,\,V} \times \underbrace{2\ \textrm{B}}_{\textrm{bf16}}
= 128\ \textrm{KiB}$$

- 32 key--value heads: 512 KiB per token.
- 128K tokens: 16 GiB, more than the 15 GiB of weights.

::: {.d2l-note}
The cache column is the first to read when sizing a deployment.
:::
:::

::: {.slide title="Released Models Use Many Narrow Experts"}
| model | experts | $k$ | width$/d$ | shared | stored / active |
|---|--:|--:|--:|---|--:|
| Mixtral 8x7B | 8 | 2 | 3.5 | none | 46.7B / 12.9B = 3.6 |
| DeepSeek-V3 | 256 | 8 | 0.29 | 1 | 671B / 37B = 18 |
| Qwen3-235B | 128 | 8 | 0.38 | none | 235B / 22B = 11 |
| gpt-oss-120b | 128 | 4 | 1.0 | none | 117B / 5.1B = 23 |

- $k$ of the $E$ experts run per token: stored parameters grow with $E$,
  the work per token with $k$.
- The later models store 11 to 23 times what a token uses.
:::

::: {.slide title="Check: How Large Is Qwen3-32B's Cache at 32K Tokens?"}
[Check]{.kicker}

Qwen3-32B: 64 layers, 64 query heads sharing 8 key--value heads of width
128, bf16. How many bytes per token, and for one 32K-token sequence?

. . .

- $64 \times 8 \times 128 \times 2 \times 2\ \textrm{B} = 256$ KiB per token.
- $256\ \textrm{KiB} \times 32{,}768 = 8$ GiB; with 64 key--value heads,
  64 GiB.
:::

::: {.slide title="At Equal Width, Sharing Heads Cuts the Cache 4× and 8×, the Parameters 12–14%"}
@!scaling-laws-recipe-rows-as-constructor-calls

- Width 256, six blocks, 8 query heads; vocabulary 1,024.
- GPT-2: learned positions, LayerNorm, GELU, biases. Llama 3 8B and
  Qwen3-32B: the default plus `num_kv_heads`.
:::

::: {.slide title="The Other Settings Enter Through Two Hooks"}
| setting | where it enters |
|:--|:--|
| QK-norm | a new attention module, through `attn_factory` |
| sliding window, sinks | a different mask |
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
- Per layer: attention $2 \cdot 4096^2 + 2 \cdot 4096 \cdot 1024 = 42$M,
  SwiGLU $3 \cdot 4096 \cdot 14{,}336 = 176$M.
- $1.05\textrm{B} + 32 \times 218\textrm{M} = 8.03$B.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Released models are the chapter's block: pre-norm RMSNorm, gated FFN,
  RoPE, no biases, no dropout.
- They differ in the cache (key--value sharing, windows, a latent) and in
  experts: Llama 3 8B caches 128 KiB per token; MoE models store 11 to 23
  times their active parameters.
- At our width, `d2l.GPT` with `num_kv_heads` gives the cache 4× and 8×
  smaller for 12% and 14% fewer parameters.

**Next:** state space models, which replace most attention layers in
hybrid designs.
:::
