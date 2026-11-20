# Transformers
:label:`chap_transformers`

:numref:`chap_attention` developed the multi-head attention layer, positional encodings, and attention variants that scale to long contexts. Attention alone, however, is only a token-mixing operator: without per-token nonlinear transformations, residual connections, and normalization, stacking attention layers does not yield a trainable deep network. This chapter assembles attention into complete **Transformer** architectures :cite:`Vaswani.Shazeer.Parmar.ea.2017` and studies how they are trained, served, and scaled.

We proceed from a single layer to frontier model families in eight steps:

1. **The Transformer block** (:numref:`sec_transformer-block`) pairs multi-head attention with a position-wise feed-forward network inside a pre-norm residual stream (`RMSNorm` and `SwiGLU`), and shows why residual projection scaling ($1/\sqrt{2L}$) and query--key normalization keep signal norms and attention logits bounded across depth.
2. **Decoder-only language models (GPT)** (:numref:`sec_gpt`) stack causal Transformer blocks between token embeddings and a language-modeling head. We train a byte-level GPT on WikiText-103, compare it against a matched recurrent baseline, load pretrained GPT-2 weights (`124M`) to inspect generation under temperature, top-$k$, and top-$p$ sampling, and summarize the architectural cheat sheet of modern open-weights LLMs (:numref:`sec_modern-recipe`).
3. **The key--value cache and cache reduction** (:numref:`sec_kv-cache`) turn quadratic autoregressive decoding into linear per-step decoding by caching past keys and values, analyze the memory-bandwidth roofline of prefill versus decode, and compress the cache across three axes (:numref:`sec_kv-cache-reduction`): head count (grouped-query and multi-query attention), sequence length (sliding-window buffers with attention sinks), and head dimension (low-rank key--value projections).
4. **Encoders and encoder--decoders** (:numref:`sec_transformer`) contrast causal decoders with bidirectional encoders trained via masked language modeling :cite:`Devlin.Chang.Lee.ea.2018` and sequence-to-sequence encoder--decoders wired with cross-attention, including fixed-size latent bottlenecks :cite:`Jaegle.Gimeno.Brock.ea.2021`.
5. **Vision Transformers (ViT)** (:numref:`sec_vision-transformer`) apply the encoder architecture directly to images by projecting non-overlapping pixel patches into token embeddings :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`, comparing learned 2D positional structure and data efficiency against a parameter-matched ResNet.
6. **Scaling laws** (:numref:`sec_scaling-laws`) count parameters ($N \approx 12 L d^2$) and training FLOPs ($C \approx 6 N D$), fit power-law IsoFLOP curves :cite:`Kaplan.McCandlish.Henighan.ea.2020,Hoffmann.Borgeaud.Mensch.ea.2022`, and show why inference demand shifts modern pretraining toward smaller models trained on many more tokens.
7. **Mixture of experts (MoE)** (:numref:`sec_moe`) decouples total parameter capacity from per-token compute by replacing the dense feed-forward network with a bank of routed experts :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017,Fedus.Zoph.Shazeer.2022`, comparing auxiliary-loss balancing against loss-free bias control :cite:`DeepSeek-AI.2024`.
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

One block, many models<br>
**the block · GPT · KV cache · encoders and decoders · ViT · scaling laws · experts · the modern recipe**
:::
:::

::: {.slide title="Twelve Copies of One Block Make GPT-2 Small"}
[The problem]{.kicker}

- One block: attention ($4d^2$ parameters) and a feed-forward network
  ($8d^2$), each adding to a residual stream. At width $d = 768$ that is
  $12d^2 \approx 7.1$M.
- Deeper, it can fail before training starts: after 32 post-norm blocks
  with Xavier weights, all tokens agree to about one part in $10^4$.
- Generating, every token reads all the weights and a growing cache: at
  4,096 tokens a GPT-2-sized model reads 288 MiB of cache besides 471 MiB
  of weights.

**Check:** twelve blocks hold 85M parameters, but GPT-2 small has 124M.
Where are the other 39M?

::: {.fragment .fade-up}
::: {.d2l-note}
In the token embedding, $50{,}257 \times 768 \approx 38.6$M, which the
output layer reuses, and in 1,024 learned positions, 0.8M.
:::
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- build the block, and say when a deep post-norm stack maps all tokens
  toward one vector and what pre-norm changes;
- train a GPT beside a recurrent network of its size, and load GPT-2 into
  it;
- implement a KV cache, bound the speed of decoding by its bandwidth, and
  shrink the cache by sharing key–value heads;
- count parameters and FLOPs, split a compute budget, and route tokens to
  experts;
- wire encoders, encoder–decoders and ViTs, and read a released model as
  settings of the block.

**Idea:** most problems of depth and scale are answered by a setting of
one block. **Running examples:** a 4.8M-parameter byte-level GPT; GPT-2
small (124M, 12 blocks of width 768).
:::

::: {.slide title="The Block and GPT Are the Core; the Cache Comes Next"}
[Scope and place]{.kicker}

- **Assumes:** attention with masks and rotary positions; AdamW with
  warmup.
- **Core:** the block and GPT.
- **Next:** the KV cache and how to shrink it, which explain why a long
  conversation costs more per word.
- **Can wait:** the vision transformer; encoders until the encoder chapter
  of the language-model part.
- **Cost:** one RTX 4090; every notebook under a quarter of an hour.
- **Leaves out:** data, fine-tuning, serving, quantization, kernels,
  parallelism.
:::

::: {.slide title="Build a GPT, Make It Generate, Then Rewire, Scale and Read It"}
[Chapter map]{.kicker}

| | section | question |
|--------|------------------------------|------------------------------------------|
| **build** | The Transformer Block | what does a block need besides attention? |
| | A GPT from Scratch | how do blocks become a trained language model? |
| **generate** | The KV Cache | how do we generate without recomputing? |
| | Shrinking the KV Cache | which factor of the cache can shrink? |
| **rewire** | Encoders, Decoders, and Cross-Attention | which wiring suits which task? |
| | Vision Transformer | can the same blocks classify images? |
| **scale** | Scaling Laws | what does a run cost, and how is a budget split? |
| | Mixture of Experts | how do we store many FFNs and run few? |
| **read** | The Modern Recipe | which settings do released models choose? |
| | Further Reading | what does the chapter leave out? |
:::

::: {.slide title="Every Section Runs on Models That Fit One GPU"}
[Running examples]{.kicker}

| section | example |
|------------------------------|--------------------------------------------------|
| The Transformer Block | 32 random blocks: post-LN and pre-LN with Xavier weights, post-LN at GPT-2's weight scale; one vector through LayerNorm and RMSNorm |
| A GPT from Scratch | a 4.8M-parameter GPT on WikiText-103 bytes; a recurrent network of the same size on the same bytes; GPT-2 small |
| The KV Cache | an untrained GPT-2-sized model: cached decoding, prefill, batched decode |
| Shrinking the KV Cache | the byte GPT with 8 and with 2 key–value heads; GPT-2 small: low-rank keys, sinks |
| Encoders, Decoders, and Cross-Attention | masked bytes; sequence reversal; learned queries |
| Vision Transformer | Fashion-MNIST at $96 \times 96$, 36 patches |
| Scaling Laws | fifteen GPTs: five sizes, three data budgets |
| Mixture of Experts | an 8-expert GPT against a dense one |
| The Modern Recipe | GPT-2, Llama 3 8B and Qwen3-32B as constructor calls at our model's size |
:::
