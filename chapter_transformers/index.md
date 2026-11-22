# Transformers
:label:`chap_transformers`

:numref:`chap_attention` developed the multi-head attention layer, positional encodings, and attention variants that scale to long contexts. Attention alone, however, is only a token-mixing operator: without per-token nonlinear transformations, residual connections, and normalization, stacking attention layers does not yield a trainable deep network. This chapter assembles attention into complete **Transformer** architectures :cite:`Vaswani.Shazeer.Parmar.ea.2017` and studies how they are trained, served, and scaled.

We proceed from a single layer to frontier model families in eight steps:

1. **The Transformer block** (:numref:`sec_transformer-block`) pairs multi-head attention with a position-wise feed-forward network inside a pre-norm residual stream (`RMSNorm` and `SwiGLU`), and shows why residual projection scaling ($1/\sqrt{2L}$) and query--key normalization keep signal norms and attention logits bounded across depth.
2. **Decoder-only language models (GPT)** (:numref:`sec_gpt`) stack causal Transformer blocks between token embeddings and a language-modeling head. We train a byte-level GPT on WikiText-103, compare it against a matched recurrent baseline, load pretrained GPT-2 weights (`124M`) to inspect generation under temperature, top-$k$, and top-$p$ sampling, and summarize the architectural cheat sheet of modern open-weights LLMs (:numref:`sec_modern-recipe`).
3. **The key--value cache and cache reduction** (:numref:`sec_kv-cache`) turn quadratic autoregressive decoding into linear per-step decoding by caching past keys and values, analyze the memory-bandwidth roofline of prefill versus decode, and compress the cache across three axes (:numref:`sec_kv-cache-reduction`): head count (grouped-query and multi-query attention), sequence length (sliding-window buffers with attention sinks), and head dimension (low-rank key--value projections).
4. **Encoders and encoder--decoders** (:numref:`sec_transformer`) contrast causal decoders with bidirectional encoders trained via masked language modeling :cite:`Devlin.Chang.Lee.ea.2018` and sequence-to-sequence encoder--decoders wired with cross-attention, including fixed-size latent bottlenecks :cite:`Jaegle.Gimeno.Brock.ea.2021`.
5. **Vision Transformers (ViT)** (:numref:`sec_vision-transformer`) apply the encoder architecture directly to images by projecting non-overlapping pixel patches into token embeddings :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, comparing learned 2D positional structure and data efficiency against a parameter-matched ResNet.
6. **Scaling laws** (:numref:`sec_scaling-laws`) count parameters ($N \approx 12 L d^2$) and training FLOPs ($C \approx 6 N D$), fit power-law IsoFLOP curves :cite:`kaplan2020scaling,hoffmann2022training`, and show why inference demand shifts modern pretraining toward smaller models trained on many more tokens.
7. **Mixture of experts (MoE)** (:numref:`sec_moe`) decouples total parameter capacity from per-token compute by replacing the dense feed-forward network with a bank of routed experts :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017,fedus2022switch`, comparing auxiliary-loss balancing against loss-free bias control :cite:`DeepSeek-AI.2024`.
8. **Further reading** (:numref:`sec_transformers-further-reading`) surveys architectural refinements beyond the baseline recipe, including normalization placement, multi-head latent attention, speculative decoding, and retrieval-augmented interfaces.

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
- **End-to-end validation:** From 32-layer initialization audits and byte-level WikiText-103 pretraining to RTX 4090 roofline profiling, 15-model IsoFLOP fits, and 8-expert MoE routing.
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
