# Attention
:label:`chap_attention`

Recurrent networks summarize a variable-length sequence in a fixed-dimensional
state. This state can become a bottleneck when later predictions require
specific information from earlier positions. Attention instead retains a
representation at every position. A query is compared with a key for each
position, the comparison scores are normalized into weights, and the output is
a weighted average of the corresponding values. The terms *query*, *key*, and
*value* emphasize the connection to database lookup, while the weighted average
makes the operation differentiable.

Attention was introduced in neural machine translation to let a decoder access
all source representations rather than a single sentence vector
:cite:`Bahdanau.Cho.Bengio.2014`. The Transformer later removed recurrence and
used attention as its primary sequence-mixing operation
:cite:`Vaswani.Shazeer.Parmar.ea.2017`; that architecture is developed in
:numref:`chap_transformers`. Attention is now used in models for text, images,
speech, and biological sequences. This chapter studies the mechanism itself:
its algebra, positional information, computational cost, and the circuits that
trained attention layers can implement.

The chapter follows six dependencies. A soft lookup first defines queries,
keys, values, and normalized weights. Learned scoring functions replace fixed
kernels, and multiple heads provide several independently projected lookups.
Because this operation is permutation equivariant, sequence models must then
represent position explicitly or through a causal mechanism. The resulting
all-pairs interaction has quadratic cost, which motivates exact memory-saving,
sparse, and kernelized alternatives. Finally, an attention-only model provides
a controlled setting in which attention patterns and value transformations can
be analyzed together.

The experiments use a small character-level attention-only language model.
Complete Transformers add feed-forward layers, normalization, and a broader
training procedure; :numref:`chap_transformers` develops those components.
Optimizer details appear in :numref:`chap_optimization`, and
:numref:`chap_modern_rnn` develops the recurrent and state-space side of the
linear-attention correspondence.

```toc
:maxdepth: 2

queries-keys-values
attention-scoring
multihead-attention
positional-information
attention-at-scale
what-attention-computes
```

## Resources and Further Reading {.unnumbered}

These references cover the mechanism, its computational cost, positional
representations, and circuit analysis.

**Visual introductions**

- [Attention in transformers, step-by-step — 3Blue1Brown (2024)](https://www.3blue1brown.com/lessons/attention) — a geometric introduction to queries, keys, values, and attention weights.
- [The Illustrated Transformer — Jay Alammar (2018)](https://jalammar.github.io/illustrated-transformer/) — visual explanations of self-attention and the Transformer architecture.
- [Transformers from scratch — Peter Bloem (2019)](https://peterbloem.nl/blog/transformers) — derives self-attention as a permutation-equivariant operation before assembling the architecture.

**Foundational papers**

- [Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) — introduces learned alignment for neural machine translation.
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — introduces scaled dot-product and multi-head attention in the Transformer.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al. (2021)](https://arxiv.org/abs/2104.09864) — develops rotary position embeddings.
- [Train Short, Test Long — Press et al. (2022)](https://arxiv.org/abs/2108.12409) — introduces ALiBi and evaluates length extrapolation.

**The cost of attention**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — organizes exact attention around memory traffic using tiled online softmax.
- [Transformers are RNNs — Katharopoulos et al. (2020)](https://arxiv.org/abs/2006.16236) — derives the recurrent form of factorized-kernel attention.
- [The Transformer Family v2.0 — Lilian Weng (2023)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — surveys attention variants and related architectural choices.

**What attention computes**

- [A Mathematical Framework for Transformer Circuits — Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) — develops the QK/OV factorization and residual-stream view for attention-only transformers.
- [In-context Learning and Induction Heads — Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — studies induction heads and their relation to in-context learning.
- [Thinking Like Transformers (Transformer Puzzles) — Sasha Rush](https://github.com/srush/Transformer-Puzzles) — provides RASP exercises on computations expressible by fixed-depth attention models.

**Exercises**

- [Stanford CS224n, self-attention and transformers assignment](https://web.stanford.edu/class/cs224n/) — exercises on self-attention, positional information, and Transformer components.
