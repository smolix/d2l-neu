# Transformers
:label:`chap_transformers`

Attention lets every position read the others, but on its own it makes a
weak model: each output is a weighted average of linear projections,
and nothing computes a nonlinear feature of a single position. A
*transformer* block adds a feed-forward network that transforms each
position on its own, and both sublayers add their outputs to a residual
stream that carries the signal through the stack. GPT-2 small stacks twelve
such blocks of width 768 into 124 million parameters. Open models
hundreds of times larger use the same block with a few settings changed:
where the normalization sits, the activation of the feed-forward network,
how positions enter, how many key--value heads attention keeps, and whether
a block has one feed-forward network or many.

Each setting answers a problem of depth or scale. A deep stack can lose its
signal at initialization and diverge in training. Generating one token at a
time recomputes the whole context unless keys and values are kept, and then
reading what is kept, not the arithmetic, sets the speed. More parameters
cost more computation unless most of them sit idle for any one token. And a
run that costs millions has to be sized before it starts. We build one
implementation in which each setting is a constructor argument, and measure
what each changes on small models.

:numref:`sec_transformer-block` asks what a block needs besides attention,
and where its normalization must sit for a deep stack to train.
:numref:`sec_gpt` asks how a stack of blocks becomes a language model, how
to train it, and whether the same code runs the released GPT-2.
:numref:`sec_kv-cache` asks how to generate without recomputing the
context, and how to shrink the cache that generation then reads.
:numref:`sec_transformer` asks what changes when the causal mask is removed,
or a second stack is joined by cross-attention.
:numref:`sec_vision-transformer` asks whether the same blocks can classify
an image cut into patches. :numref:`sec_moe` asks how a model can store many
feed-forward networks and run only a few for each token.
:numref:`sec_scaling-laws` asks what a training run costs, and how to split
a budget between parameters and data.

The chapter treats the architecture of base models. Pretraining at scale,
data, fine-tuning, serving systems, and quantization belong to the Language
Models part, kernels and parallelism to
:numref:`chap_performance`, and recurrent alternatives to a growing cache to
:numref:`chap_modern_rnn`. Related topics that the chapter does not teach
are collected in :numref:`sec_transformers-further-reading`.

```toc
:maxdepth: 2

transformer-block
gpt
kv-cache
encoders-decoders
vision-transformer
moe
scaling-laws
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online. Related topics that
the chapter does not teach are collected in
:numref:`sec_transformers-further-reading`.

**Build-alongs and courses**

- [Let's Build GPT: From Scratch, in Code — Andrej Karpathy (2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the video counterpart of :numref:`sec_gpt`: a character-level GPT assembled and trained in real time; [nanoGPT](https://github.com/karpathy/nanoGPT) and [build-nanogpt](https://github.com/karpathy/build-nanogpt) are its repository forms, and [nanochat](https://github.com/karpathy/nanochat) applies the same approach to a full chat system, with its training costs stated in dollars.
- [The Annotated Transformer — Harvard NLP (2018, refreshed 2022)](https://nlp.seas.harvard.edu/annotated-transformer/) — the original encoder–decoder, implemented line by line next to the text of the paper, the wiring of :numref:`sec_transformer`.
- [Build a Large Language Model (From Scratch) — Sebastian Raschka (2024)](https://github.com/rasbt/LLMs-from-scratch) — code for a book-length version of :numref:`sec_transformer-block` through :numref:`sec_kv-cache`, with notebooks for GQA, sliding windows, and from-scratch ports of current open models (the book itself is paid).
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — a course whose first assignment builds RMSNorm, RoPE, SwiGLU, causal attention, and the training loop of :numref:`sec_transformer-block` and :numref:`sec_gpt` from primitives; the lectures are on YouTube.
- [CMU Advanced NLP, minLlama Assignment](https://www.phontron.com/class/anlp-fall2024/) — build a Llama-style decoder and load pretrained weights into it, the exercise of loading GPT-2 in :numref:`sec_gpt` at a larger scale.

**Architecture papers**

- [On Layer Normalization in the Transformer Architecture — Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) — the analysis of pre-norm and post-norm behind the signal-propagation experiment of :numref:`sec_transformer-block`, and the reason post-norm models need warmup, which :numref:`sec_gpt` tests.
- [GLU Variants Improve Transformer — Shazeer (2020)](https://arxiv.org/abs/2002.05202) — the four-page note whose matched-parameter comparison :numref:`sec_transformer-block` repeats in miniature.
- [GQA: Training Generalized Multi-Query Transformer Models — Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) — grouped-query attention as :numref:`sec_kv-cache` uses it, including the uptraining recipe that converts multi-head checkpoints.
- [Efficient Streaming Language Models with Attention Sinks — Xiao et al. (2024)](https://arxiv.org/abs/2309.17453) — the first-token sink and the windowed cache that keeps it, which :numref:`sec_kv-cache` measures on GPT-2.
- [Perceiver IO: A General Architecture for Structured Inputs and Outputs — Jaegle et al. (2022)](https://arxiv.org/abs/2107.14795) — learned queries that read an input of any size and write an output of any shape, the interface of :numref:`sec_transformer`.
- [An Image Is Worth 16x16 Words — Dosovitskiy et al. (2021)](https://arxiv.org/abs/2010.11929) — the ViT paper behind :numref:`sec_vision-transformer`, whose question of scale against built-in assumptions the section's small experiment asks in a different regime.
- [Switch Transformers — Fedus et al. (2022)](https://arxiv.org/abs/2101.03961) and [DeepSeek-V3 Technical Report — DeepSeek-AI (2024)](https://arxiv.org/abs/2412.19437) — the two ends of the design range in :numref:`sec_moe`: top-1 routing with an auxiliary balancing loss, and fine-grained experts balanced by a bias.
- [The Big LLM Architecture Comparison — Sebastian Raschka (2025, maintained)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — current open models compared on the same few axes, a maintained version of :numref:`tab_modern-recipe` in :numref:`sec_scaling-laws`.

**The arithmetic of scale**

- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/transformer-inference-arithmetic/) — per-token inference FLOPs, cache storage, and bandwidth limits, the accounting of :numref:`sec_kv-cache`, compared with a real system.
- [Transformer Math 101 — EleutherAI (2023)](https://blog.eleuther.ai/transformer-math/) — the training-side companion: where $6ND$ comes from and what it predicts, the count that :numref:`sec_scaling-laws` checks against the framework.
- [Training Compute-Optimal Large Language Models — Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) — Chinchilla, the tokens-per-parameter result whose law :numref:`sec_scaling-laws` fits and whose bend its miniature study shows.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — the parallelism and memory engineering that the budgets of :numref:`sec_scaling-laws` require beyond one GPU, treated in :numref:`chap_performance`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11]{.kicker}

One block, many models<br>
**the block · GPT · KV cache · encoders and decoders · ViT · experts · scaling laws**
:::
:::

::: {.slide title="Twelve Copies of One Block Make GPT-2 Small"}
[The problem]{.kicker}

- One block: attention plus a feed-forward network, both added to a
  residual stream; $12d^2 pprox 7.1$M parameters at $d = 768$.
- Twelve blocks, embeddings, and a tied head: 124M parameters.
- Open models hundreds of times larger change a few settings: norm
  placement, activation, positions, key–value heads, experts.
- Each setting fixes a problem of depth or scale: unstable training, slow
  generation, costly parameters, unplanned budgets.

::: {.d2l-note}
Which few choices make one block trainable deep, fast to generate from,
cheap per parameter, and predictable in cost?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- build the block and explain why pre-norm stacks train where post-norm
  stacks stall;
- write a GPT, train it, and load GPT-2's released weights into it;
- implement a KV cache and estimate the memory and bandwidth of decoding;
- wire encoders, encoder–decoders, ViTs, and mixtures of experts from the
  same block;
- count parameters and FLOPs, and split a compute budget between model
  size and data.

**Running example:** GPT-2 small, 124M parameters, 12 blocks of width 768.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| The Transformer Block | what does a block need besides attention? |
| A GPT from Scratch | how do blocks become a trained language model? |
| Generation and the KV Cache | how do we generate without recomputing? |
| Encoders, Decoders, and Cross-Attention | what do the other wirings give? |
| Vision Transformer | can the same blocks classify images? |
| Mixture of Experts | how do we store many FFNs and run few? |
| Scaling Laws | what does a run cost, and how should a budget be split? |
:::

::: {.slide title="Every Section Measures Its Claims on Models That Train on One GPU"}
[Running examples]{.kicker}

| section | testbed |
|---|---|
| The Transformer Block | 32 random blocks; a small byte model on WikiText-103 |
| A GPT from Scratch | a 4.8M-parameter GPT on WikiText-103 bytes; GPT-2 small |
| Generation and the KV Cache | GPT-2 small: cached decoding, low-rank keys, sinks |
| Encoders, Decoders, and Cross-Attention | masked bytes; sequence reversal; learned queries |
| Vision Transformer | Fashion-MNIST at $96 \times 96$, 36 patches |
| Mixture of Experts | an 8-expert GPT against a dense one |
| Scaling Laws | fifteen GPTs: five sizes, three data budgets |
:::
