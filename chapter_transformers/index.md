# Transformers
:label:`chap_transformers`

:numref:`chap_attention` studied attention the way one studies an organ
in isolation; this chapter builds the organism. A transformer wraps the
attention layer in a *block*: attention moves information between
positions, a position-wise feed-forward network transforms it in place,
and residual connections with normalization keep the signal usable when
blocks are stacked dozens or hundreds deep. That block, stacked and
wired in one of three ways, is the entire architecture. It has carried
machine translation, then language modeling, then images, audio, and
protein structure, with so little structural change since 2017 that the
differences between today's frontier models fit in a small table — one
this chapter ends by writing down and instantiating.

Our method follows from that observation. Early in the chapter we build
a single configurable GPT class: normalization type and placement,
activation, positional scheme, and the attention and feed-forward
modules are all constructor arguments. Every subsequent section changes
one component, re-runs the model, and measures the difference, so that
by the close of the chapter the recipe table's rows are not
descriptions but constructor calls. Every training run here finishes in
minutes on one GPU; where the small scale genuinely changes the story —
and it sometimes does — the prose says so, and stated cost anchors
connect our runs to what production training actually spends.

The sections, in order. :numref:`sec_transformer-block` dissects the
block: where normalization goes (we measure signal propagation through
32 randomly initialized blocks and watch the post-norm arrangement
starve its own attention gradients), why RMSNorm and QK-norm displaced
plain LayerNorm, and what gated feed-forward networks buy, ending with
the configurable block. :numref:`sec_gpt` assembles the blocks into a
language model, trains the modern configuration on real text, samples
from it, and then loads the published GPT-2 weights into the same class
— the 2019 model is one setting of the flags, and our hand-built code
generates with it. :numref:`sec_kv-cache` turns to generation, where
the economics invert: decoding is bound by memory traffic, not
arithmetic, and the key–value cache, grouped-query attention, low-rank
cache compression, and sliding windows with attention sinks are the
escalating responses; we implement and measure each.
:numref:`sec_transformer` steps back to the taxonomy — encoder-only,
decoder-only, encoder–decoder as three wirings of the same block — and
treats cross-attention fully, including its most general form: a small
learned latent array that reads an arbitrarily long input, the idea
behind the Perceiver family and the input adapters of current
multimodal models. :numref:`sec_vision-transformer` feeds the
architecture images by cutting them into patch tokens, and lets the
result teach the lesson: at our scale a matched convolutional
network wins, because the transformer must learn the locality a
convolution assumes. :numref:`sec_moe` decouples parameters from
computation with mixture-of-experts layers, where the interesting
problem is not the idea but keeping the router from collapsing — we
train the failure and both standard repairs. :numref:`sec_scaling-laws`
closes with the arithmetic of scale: counting FLOPs, a miniature
scaling study whose bend reproduces the data-starvation effect behind
compute-optimal training, and the modern recipe table.

The history is short enough to tell in full. The 2017 transformer was
an encoder–decoder for translation, normalized after each sublayer,
with sinusoidal positions and a plain ReLU network. What survived is
the block; nearly every choice around it changed, and the changes
concentrate on three axes — stability at depth (pre-norm, RMSNorm,
QK-norm), the memory bill of generation (grouped queries, cache
compression, windows), and capacity (experts). Convergent evolution
across independent labs has made new open models look more alike each
year, which is what makes the architecture teachable as one object
with options rather than a zoo.

What this chapter is not. Tokenization was settled in
:numref:`chap_rnn` and optimizers in :numref:`chap_optimization`; both
are used here without comment. Pretraining corpora, instruction tuning,
and everything downstream of the base model belong to the Language
Models part, as does BERT, whose encoder this chapter's taxonomy
locates but does not train. Vision applications beyond the ViT itself
are the Image Models part's. Kernels, parallelism, quantization, and
serving systems belong to the Computational Performance chapter; here
they appear only as black-box fused-attention calls, whose kernels are
that chapter's subject, and one closing sentence on speculative
decoding. And the state-space alternative to
attention has its own chapter, :numref:`chap_modern_rnn`, which picks
up exactly where :numref:`sec_kv-cache`'s cache-relief map leaves off.

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

Grouped by the chapter's arc: building the model, the architecture
record, and the arithmetic of scale. All free unless noted.

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
