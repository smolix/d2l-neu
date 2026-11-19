# The Modern Recipe
:label:`sec_modern-recipe`

<!-- WRITER: new opening (structure.md section 3.9), from the reader, in the
voice of a chapter close. The body below moved from the end of scaling-laws.md
("Configurations of Deployed Models"), cell id kept; its first sentence
("The models trained on such budgets") and its last ("trained at budgets that
the law of this section helps to plan", SL-19) still read as the end of the
scaling section. tab_moe-experts and its paragraph arrive from moe.md, and
"Departures from the Shared Block" from further-reading.md (other steps). -->

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

At width 256 and six blocks, three rows of the table become calls of our
constructor. The modern rows need only the number of key--value heads,
because the defaults of `d2l.GPT` are the recipe the table shares. Llama 3
8B shares each key--value head among 4 query heads, Qwen3-32B among 8. The
cell prints the parameter count and the bytes of KV cache per token in
32-bit floats:

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab pytorch, jax
recipes = {'GPT-2': dict(pos='learned', norm='layer', act='gelu',
                         bias=True),
           'Llama 3 8B': dict(num_kv_heads=2),
           'Qwen3-32B': dict(num_kv_heads=1)}
for name, cfg in recipes.items():
    model = gpt(1024, num_hiddens=256, num_heads=8, num_blks=6, **cfg)
    d2l.check_shape(model(d2l.zeros((2, 16), dtype=d2l.int32)),
                    (2, 16, 1024))
    print(f'{name:10s} {d2l.num_params(model) / 1e6:.2f}M parameters, '
          f'{d2l.kv_cache_bytes(model, 1) / 1024:4.1f} KiB of cache '
          'per token')
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

::: {.slide title="Open Models Share One Block and Differ in the Cache and the FFN"}
| model | attention + cache | norm | FFN / experts |
|:--|:--|:--|:--|
| Mistral 7B | GQA 32:8, window 4096 | RMS pre | SwiGLU, dense |
| Llama 3 8B | GQA 32:8 | RMS pre | SwiGLU, dense |
| Qwen3 | GQA 64:8 dense, 64:4 MoE | RMS pre + QK | SwiGLU, dense & MoE 128/8 |
| DeepSeek-V3 | MLA: 512-d latent | RMS pre | SwiGLU, MoE 256+1/8 |
| Gemma 3 | GQA, local:global 5:1 | RMS pre+post + QK | GeGLU, dense |
| GPT-OSS | GQA 64:8, window 128 alt., sinks | RMS pre | SwiGLU, MoE 128/4 |

All: RoPE, **no dropout**: a model that reads its tokens about once has
nothing to memorize.
:::

::: {.slide title="At Equal Width, the Rows Differ by Their Cache"}
@!scaling-laws-recipe-rows-as-constructor-calls

- The modern rows need one flag: the defaults of `d2l.GPT` are the shared
  recipe.
- Key--value sharing shrinks the cache 4× and 8×, the parameters by about
  a tenth.
:::
