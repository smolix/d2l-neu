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

Multi-head attention mixes information across sequence positions, but on its own it applies only linear projections around a weighted sum of values.

::: {.cols}
::: {.col}
- **Missing per-token nonlinearity:** Without a position-wise feed-forward network (FFN), stacked attention layers cannot compute hierarchical nonlinear features within each token vector.
- **Depth instability at initialization:** Stacking 32 post-norm blocks with standard Xavier weights collapses all token representations onto a single vector (token spread $\sim 10^{-4}$) before training even begins.
:::

::: {.col}
- **The memory-bandwidth wall of generation:** Autoregressive decoding streams all model weights and a growing key–value (KV) cache from GPU HBM for every single generated token.
- **Unifying architectural insight:** Nearly every modern foundation model across text, code, speech, and vision is built by repeating and rewiring **one configurable Transformer block**.
:::
:::

::: {.d2l-note}
In GPT-2 Small ($124\text{M}$ parameters, $L = 12, d = 768$), the 12 blocks hold $12 \times 12 d^2 \approx 85\text{M}$ parameters ($4d^2$ in attention, $8d^2$ in the FFN), while the weight-tied token embedding table ($50{,}257 \times 768 \approx 38.6\text{M}$) and $1{,}024$ learned positions ($0.8\text{M}$) hold the remaining $39\text{M}$.
:::
:::

::: {.slide title="The Four Architectural Pillars of Chapter 11"}
[Architectural Roadmap]{.kicker}

We progress from a single residual block to frontier training, serving, and scaling in seven connected sections:

| Pillar | Chapter 11 Sections | Core Engineering & Mathematical Question |
| :--- | :--- | :--- |
| **1. Build & Train** | **§11.1** Transformer Block · **§11.2** GPT | Why do pre-norm, `RMSNorm`, `QK-Norm`, $1/\sqrt{2L}$ scaling, and `SwiGLU` stabilize deep language models? |
| **2. Serve & Compress** | **§11.3** The KV Cache & Cache Reduction | How does KV caching turn $O(T^2)$ decoding into $O(T)$, and how do GQA, MLA, and sinks beat the bandwidth roofline? |
| **3. Rewire Modalities** | **§11.4** Encoders–Decoders · **§11.5** ViT | When should we use bidirectional MLM, cross-attention, latent bottlenecks, or image patch tokens? |
| **4. Scale Capacity** | **§11.6** Scaling Laws · **§11.7** MoE | How do we allocate a $C \approx 6ND$ FLOP budget (Chinchilla) or decouple stored capacity from active FLOPs? |
:::

::: {.slide title="Concrete Single-GPU Benchmarks Across Chapter 11"}
[Running Examples]{.kicker}

Every architectural mechanism in this chapter is tested on reproducible models that train or run on a single GPU in minutes:

| Section | Concrete Benchmark & Empirical Verification |
| :--- | :--- |
| **§11.1 Transformer Block** | 32-layer signal propagation & gradient audit comparing post-norm vs. pre-norm with $1/\sqrt{2L}$ scaling |
| **§11.2 Decoder-Only (GPT)** | 4.8M-parameter byte-level GPT on WikiText-103 vs. matched 4.7M RNN; loading OpenAI's 124M GPT-2 Small |
| **§11.3 KV Cache & Reduction** | RTX 4090 roofline profiling of prefill vs. batched decode; GQA uptraining, SVD low-rank KV, and attention sinks |
| **§11.4 Encoders & Cross-Attn** | Masked byte modeling on WikiText-103; exact sequence reversal alignment; Perceiver $O(MN)$ latent bottleneck |
| **§11.5 Vision Transformer** | 6.5M-parameter ViT ($6 \times 6$ grid of $16 \times 16$ patches) vs. parameter-matched ResNet on $96 \times 96$ Fashion-MNIST |
| **§11.6–11.7 Scaling & MoE** | 15-model IsoFLOP scaling grid ($N \in [0.3\text{M}, 14\text{M}]$); 8-expert Top-1 MoE GPT vs. dense GPT with loss-free bias control |
:::

::: {.slide title="From the 2017 Transformer to the Modern Recipe"}
[The Modern Recipe]{.kicker}

Every model in this chapter is an instance of one parameterized constructor whose defaults match frontier open-weights LLMs:

| Component | 2017 Transformer / GPT-2 Baseline | Modern Frontier Recipe (Llama 3, Qwen3, DeepSeek-V3) |
| :--- | :--- | :--- |
| **Normalization** | Mean-centered `LayerNorm` with bias | Pre-norm **`RMSNorm`** (no bias) + optional per-head **`QK-Norm`** |
| **Positional Encoding** | Sinusoidal or learned absolute table $\mathbf{P}$ | **Rotary Position Embeddings (RoPE)** inside every attention layer |
| **Feed-Forward Network** | Two-matrix ReLU/GELU MLP ($d_{\text{ff}} = 4d$) | Three-matrix gated **`SwiGLU`** ($d_{\text{ff}} \approx \frac{8}{3}d$, matched $8d^2$ budget) |
| **KV Cache Footprint** | Multi-head attention ($H_{\text{kv}} = H$) | **Grouped-Query Attention (GQA)**, low-rank **MLA**, or sliding windows + sinks |
| **Capacity vs. FLOPs** | Dense FFN ($N_{\text{active}} = N_{\text{total}}$) | **Sparse Mixture of Experts (MoE)** with fine-grained + shared experts |

::: {.d2l-note}
**Up next (§11.1 The Transformer Block):** How pre-norm placement, `RMSNorm`, `QK-Norm`, and `SwiGLU` prevent token collapse and keep gradients well-behaved across depth.
:::
:::
