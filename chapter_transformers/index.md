# Transformers
:label:`chap_transformers`

:numref:`chap_attention` developed attention as a mechanism for exchanging
information between positions. A transformer places attention inside a
*block* with a position-wise feed-forward network, residual connections,
and normalization. Stacking these blocks yields models for machine
translation, language modeling, images, audio, and protein structure.
Although these applications differ, most current transformer models can
be described by a small set of architectural choices.

We therefore begin by building a configurable GPT class: normalization type and placement,
activation, positional scheme, and the attention and feed-forward
modules are all constructor arguments. Every subsequent section changes
one component and measures the difference. At the end of the chapter,
the configurations in a table of current models can be expressed as
constructor arguments. The training runs finish in minutes on a single
GPU. We state when conclusions depend on this small scale and compare the
computational cost with that of production training.

We first develop the transformer block and compare normalization placement,
RMSNorm, QK-norm, and gated feed-forward networks. We then assemble a GPT,
train it on a small text corpus, and load the published GPT-2 weights into
the same implementation. The section on generation derives the KV cache and
compares grouped-query attention, low-rank compression, and sliding windows
with attention sinks.

The remaining sections examine encoder, decoder, and encoder--decoder
architectures; apply an encoder to image patches; and replace dense
feed-forward networks with mixture-of-experts layers. The chapter concludes
by deriving parameter and FLOP counts, conducting a small scaling study, and
comparing the configurations of several current models.

The 2017 transformer was an encoder–decoder for translation, normalized after each sublayer,
with sinusoidal positions and a plain ReLU network. What survived is
the block; nearly every choice around it changed, and the changes
concentrate on three axes — stability at depth (pre-norm, RMSNorm,
QK-norm), the memory required for generation (grouped queries, cache
compression, windows), and capacity (experts). Independent model families
have adopted increasingly similar choices. This common structure allows us to treat the architecture as a
configurable model rather than as a collection of unrelated designs.

This chapter assumes the treatment of tokenization in
:numref:`chap_rnn` and optimizers in :numref:`chap_optimization`; both
are used here without comment. Pretraining corpora, instruction tuning,
and everything downstream of the base model belong to the Language
Models part, as does BERT, whose encoder this chapter's taxonomy
locates but does not train. Vision applications beyond the ViT itself
are the Image Models part's. Kernels, parallelism, quantization, and
serving systems belong to the Computational Performance chapter. Here
they appear only as black-box fused-attention calls, whose kernels are
that chapter's subject, and as one closing sentence on speculative
decoding. And the state-space alternative to
attention has its own chapter, :numref:`chap_modern_rnn`, which picks
continues the discussion of alternatives to a growing key--value cache.

```toc
:maxdepth: 2

transformer-block
gpt
kv-cache
encoders-decoders
vision-transformer
moe
scaling-laws
```

## Resources and Further Reading {.unnumbered}

The resources are grouped into model construction, architecture, and
scaling. All are freely available unless noted.

**Build-alongs**

- [Let's build GPT: from scratch, in code — Andrej Karpathy (2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the video counterpart of :numref:`sec_gpt`: a character-level GPT assembled and trained in real time; [nanoGPT](https://github.com/karpathy/nanoGPT) and [build-nanogpt](https://github.com/karpathy/build-nanogpt) are its repository forms, and [nanochat](https://github.com/karpathy/nanochat) extends the same discipline to a full chat system with stated dollar costs.
- [The Annotated Transformer — Harvard NLP (2018, refreshed 2022)](https://nlp.seas.harvard.edu/annotated-transformer/) — the original encoder–decoder implemented line by line against the paper; the format this book's executable sections descend from, and the best companion to :numref:`sec_transformer`.
- [Build a Large Language Model (From Scratch) — Sebastian Raschka (2024)](https://github.com/rasbt/LLMs-from-scratch) — a book-length version of :numref:`sec_transformer-block` through :numref:`sec_kv-cache`, with bonus notebooks for GQA, sliding windows, and from-scratch ports of current open models.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — the course whose first assignment is this chapter as graded homework: BPE, RMSNorm, RoPE, SwiGLU, causal attention, and the training loop, all from primitives, with lectures on YouTube.
- [CMU Advanced NLP, minLlama assignment](https://www.phontron.com/class/anlp-fall2024/) — build a Llama-style decoder and load real pretrained weights into it, the same payoff as :numref:`sec_gpt`'s GPT-2 cell at larger scale.

**The architecture record**

- [The Big LLM Architecture Comparison — Sebastian Raschka (2025, maintained)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — seventeen current models on the same few axes; the living version of :numref:`sec_scaling-laws`'s recipe table, updated as new models ship.
- [On Layer Normalization in the Transformer Architecture — Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) — the pre-norm/post-norm analysis behind :numref:`sec_transformer-block`'s signal-propagation experiment, and the paper that explains why warmup exists.
- [GLU Variants Improve Transformer — Shazeer (2020)](https://arxiv.org/abs/2002.05202) — the four-page note whose matched-parameter sweep :numref:`sec_transformer-block` reproduces in miniature.
- [GQA: Training Generalized Multi-Query Transformer Models — Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) — grouped-query attention as :numref:`sec_kv-cache` implements it, including the uptraining recipe that converted existing checkpoints.
- [An Image is Worth 16x16 Words — Dosovitskiy et al. (2021)](https://arxiv.org/abs/2010.11929) — the ViT paper behind :numref:`sec_vision-transformer`, with the scale-versus-inductive-bias evidence our small-scale experiment recreates from the losing side.
- [Switch Transformers — Fedus et al. (2021)](https://arxiv.org/abs/2101.03961) and [DeepSeek-V3 — DeepSeek-AI (2024)](https://arxiv.org/abs/2412.19437) — the two poles of :numref:`sec_moe`: top-1 routing with an auxiliary balancing loss, and fine-grained experts balanced without one.

**The arithmetic of scale**

- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/transformer-inference-arithmetic/) — the napkin-math discipline behind :numref:`sec_kv-cache`'s memory-bill section: 2P FLOPs per token, cache bytes, and why decode is bandwidth-bound, checked against a real system.
- [Transformer Math 101 — EleutherAI (2023)](https://blog.eleuther.ai/transformer-math/) — the training-side companion: where 6ND comes from and what it predicts, the accounting :numref:`sec_scaling-laws` verifies against a profiler.
- [Training Compute-Optimal Large Language Models — Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) — Chinchilla: the tokens-per-parameter result whose small-scale shadow is the bend in :numref:`sec_scaling-laws`'s miniature study.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — what happens past one GPU: the parallelism and memory engineering this chapter deliberately leaves to the Computational Performance chapter.
