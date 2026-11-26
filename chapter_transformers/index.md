# Transformers
:label:`chap_transformers`

:numref:`chap_attention` developed the multi-head attention layer, positional encodings, and attention variants that scale to long contexts. Attention alone, however, is only a token-mixing operator: without per-token nonlinear transformations, residual connections, and normalization, stacking attention layers does not yield a trainable deep network. This chapter assembles attention into complete **Transformer** architectures :cite:`Vaswani.Shazeer.Parmar.ea.2017` and studies how they are trained, served, and scaled.

We begin by building and training the causal stack from the ground up. :numref:`sec_transformer-block` pairs multi-head attention with a position-wise feed-forward network inside a pre-norm residual stream (`RMSNorm` and `SwiGLU`), showing why residual projection scaling ($1/\sqrt{2L}$) and query--key normalization keep signal norms and attention logits bounded across depth. :numref:`sec_gpt` stacks these causal blocks between token embeddings and a weight-tied language-modeling head, trains a byte-level GPT on WikiText-103 against a matched recurrent baseline, loads pretrained GPT-2 weights (`124M`) to examine temperature and top-$k$ sampling, and distills the architectural recipe shared by modern open-weights LLMs (:numref:`sec_modern-recipe`). Because naive autoregressive generation recomputes the entire prefix at every step, :numref:`sec_kv-cache` introduces the key--value cache, contrasts compute-bound prefill with memory-bandwidth-bound decode on the hardware roofline, and compresses the cache across head count (grouped-query and multi-query attention), channel width (low-rank key--value projections), and sequence length (sliding windows with attention sinks).

Not every task is causal text generation. :numref:`sec_transformer` contrasts causal decoders with bidirectional encoders trained via masked language modeling :cite:`Devlin.Chang.Lee.ea.2018` and sequence-to-sequence encoder--decoders coupled by cross-attention, including fixed-size latent bottlenecks :cite:`Jaegle.Gimeno.Brock.ea.2021`. :numref:`sec_vision-transformer` then applies the bidirectional encoder directly to images by projecting non-overlapping pixel patches into token embeddings :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, comparing learned 2D positional structure and data efficiency against a parameter-matched ResNet.

Finally, we turn to scaling and conditional computation. :numref:`sec_scaling-laws` counts parameters ($N \approx 12 L d^2$) and training FLOPs ($C \approx 6 N D$), trains a 15-model grid, uses the published Chinchilla power-law fits :cite:`kaplan2020scaling,hoffmann2022training` to draw IsoFLOP curves and the compute-optimal split, and explains why lifetime inference cost pushes modern pretraining toward smaller models trained far past the Chinchilla token-per-parameter ratio. :numref:`sec_moe` decouples stored parameter capacity from per-token FLOPs by replacing the dense feed-forward network with a bank of routed experts :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017,fedus2022switch`, comparing auxiliary-loss balancing against loss-free router bias control :cite:`Wang.Gao.Zeng.ea.2024`, and :numref:`sec_transformers-further-reading` surveys extensions across normalization placement, latent attention, and speculative decoding.

```toc
:maxdepth: 2

transformer-block
gpt
kv-cache
encoders-decoders
vision-transformer
scaling-laws
moe
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11]{.kicker}

Transformers: From One Block to Frontier Architectures<br>**residual stream · pre-norm & SwiGLU · GPT · KV cache & roofline · encoders & ViT · scaling laws · Mixture of Experts**
:::
:::

::: {.slide title="Why Attention Alone Is Not Enough"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Missing per-token nonlinearity:** Without a position-wise FFN, stacked attention layers cannot synthesize hierarchical nonlinear features within each token vector.
- **Depth instability & bandwidth walls:** Deep stacks require pre-norm residual streams, and generation streams all $12 L d^2$ block weights and the KV cache from HBM.
:::

::: {.col .fig .big}
@fig:ch11-tf-why-block
:::
:::
:::

::: {.slide title="The Four Architectural Pillars of Chapter §{num:chap_transformers}"}
[Architectural Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Build & Train (§§{sec_transformer-block}–§{num:sec_gpt}):** Pre-norm, `RMSNorm`, `QK-Norm`, $1/\sqrt{2L}$ scaling, `SwiGLU`, and decoder-only GPT pretraining.
- **2. Serve & Compress (§{sec_kv-cache}):** KV caching, roofline analysis, GQA, low-rank MLA, and attention sinks.
- **3. Rewire & Scale (§§{sec_transformer}–§{num:sec_moe}):** Encoders, cross-attention, ViT patches, Chinchilla scaling laws, and sparse Mixture of Experts (MoE).
:::

::: {.col .fig .big}
@fig:ch11-tf-four-pillars
:::
:::
:::

::: {.slide title="Concrete Single-GPU Benchmarks Across Chapter §{num:chap_transformers}"}
[Running Examples]{.kicker}

::: {.cols .vc}
::: {.col}
- Every architectural mechanism in this chapter is verified on reproducible models that train or profile on a single GPU in minutes:
- **End-to-end validation:** From 32-layer initialization audits and byte-level WikiText-103 pretraining to RTX 4090 roofline profiling, a 15-model scaling grid, and 8-expert MoE routing.
:::

::: {.col .fig .big}
@fig:ch11-tf-benchmarks
:::
:::
:::

::: {.slide title="From the 2017 Transformer to the Modern Recipe"}
[The Modern Recipe]{.kicker}

::: {.cols .vc}
::: {.col}
- Every model in this chapter is an instance of one parameterized constructor whose defaults match frontier open-weights LLMs (`RMSNorm`, `RoPE`, `SwiGLU`, GQA/MLA, MoE).
- **Up next (§{sec_transformer-block} The Transformer Block):** How pre-norm placement, `RMSNorm`, `QK-Norm`, and `SwiGLU` prevent token collapse and keep gradients well-behaved across depth.
:::

::: {.col .fig .big}
@fig:ch11-tf-modern-recipe
:::
:::
:::
