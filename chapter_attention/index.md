# Attention
:label:`chap_attention`

A language model that has read a thousand tokens often needs one of them
again: a name from the first paragraph, a variable defined at the top of a
file. A recurrent network must carry that token forward in a state of fixed
size. After 1,024 tokens, a recurrent network as wide as GPT-2 small holds
768 numbers about them, all written before anyone knew which token would be
asked for.
Attention keeps a representation of every position instead and lets each
query choose what to read. It scores the query against a key at every
position, turns the scores into weights, and returns the weighted average of
the values stored there. The terms *query*, *key*, and *value* come from
database lookup, and the weighted average makes the lookup differentiable.

Attention first let a translation model consult every source word instead of
one sentence vector :cite:`Bahdanau.Cho.Bengio.2014`, and the Transformer
made it the only operation that mixes positions
:cite:`Vaswani.Shazeer.Parmar.ea.2017`. Reading every position raises
problems that recurrent networks never had. A weighted average ignores the
order of its inputs. Scoring all pairs of positions costs time and memory
that grow with the square of the length. And a map of attention weights
shows where a head looks, not what it computes.

:numref:`sec_queries-keys-values` asks how a lookup can be made
differentiable, starting from classical kernel regression.
:numref:`sec_attention-scoring-functions` asks which score a network should
learn, and how masks keep a query away from padding and from the future.
:numref:`sec_multihead-attention` asks what one weighting per query cannot
express, and what several heads, and heads that share keys and values,
provide. :numref:`sec_positional-information` asks how an operation that is
blind to order learns where each token stands, and what happens beyond the
length it was trained on. :numref:`sec_attention-at-scale` asks how to cut
the cost of scoring all pairs of positions: by computing exact attention
with less memory, by restricting each query to a window, or by a recurrence
of linear cost. :numref:`sec_what-attention-computes` asks
what trained heads compute, and which experiments can tell.

The chapter studies the attention layer by itself, in models built from
attention alone. :numref:`chap_transformers` adds the feed-forward layers,
normalization, and training recipe that make it a transformer, and
:numref:`chap_modern_rnn` develops the recurrent side of linear attention.
Related topics that the chapter does not teach are collected in
:numref:`sec_attention-further-reading`.

```toc
:maxdepth: 2

queries-keys-values
attention-scoring
multihead-attention
positional-information
attention-at-scale
what-attention-computes
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online. Related topics that
the chapter does not teach are collected in
:numref:`sec_attention-further-reading`.

**Visual introductions**

- [Attention in Transformers, Step-by-Step — 3Blue1Brown (2024)](https://www.3blue1brown.com/lessons/attention) — a geometric introduction to queries, keys, values, and attention weights, the lookup of :numref:`sec_queries-keys-values` drawn as vectors in an embedding space.
- [The Illustrated Transformer — Jay Alammar (2018)](https://jalammar.github.io/illustrated-transformer/) — pictures of scaled dot-product and multi-head attention with the shapes of every matrix, the computation of :numref:`sec_attention-scoring-functions` and :numref:`sec_multihead-attention`.
- [Transformers from Scratch — Peter Bloem (2019)](https://peterbloem.nl/blog/transformers) — derives self-attention as an operation on sets, with the permutation equivariance that :numref:`sec_positional-information` proves and then breaks.

**Foundational papers**

- [Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) — the first learned alignment, whose additive score :numref:`sec_attention-scoring-functions` compares with the dot product.
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — scaled dot-product attention, multi-head attention, and sinusoidal positions, the starting point of :numref:`sec_attention-scoring-functions` through :numref:`sec_positional-information`.
- [Fast Transformer Decoding: One Write-Head Is All You Need — Shazeer (2019)](https://arxiv.org/abs/1911.02150) — multi-query attention, the extreme case of the shared keys and values of :numref:`sec_multihead-attention`, motivated by the memory traffic of decoding.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al. (2021)](https://arxiv.org/abs/2104.09864) — rotary embeddings, derived as in :numref:`sec_positional-information` from the requirement that scores depend only on offsets.
- [Train Short, Test Long — Press et al. (2022)](https://arxiv.org/abs/2108.12409) — ALiBi and the length-extrapolation experiment that :numref:`sec_positional-information` repeats on a small model.
- [YaRN: Efficient Context Window Extension of Large Language Models — Peng et al. (2023)](https://arxiv.org/abs/2309.00071) — position interpolation, the NTK-aware base change, and YaRN's treatment of fast and slow pairs, the context extension of :numref:`sec_positional-information`.

**The cost of attention**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — the tiled online softmax of :numref:`sec_attention-at-scale`, with the memory-traffic analysis that explains its speed.
- [Transformers Are RNNs — Katharopoulos et al. (2020)](https://arxiv.org/abs/2006.16236) — linear attention and its recurrent form, the third remedy of :numref:`sec_attention-at-scale`.
- [The Transformer Family Version 2.0 — Lilian Weng (2023)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — a survey of attention variants, sparse patterns, and efficient approximations around :numref:`sec_attention-at-scale`.

**What attention computes**

- [A Mathematical Framework for Transformer Circuits — Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) — the residual stream and the QK and OV circuits that :numref:`sec_what-attention-computes` reads from trained weights.
- [In-Context Learning and Induction Heads — Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — induction heads in small and large models, the circuit that :numref:`sec_what-attention-computes` trains and dissects.
- [Thinking Like Transformers (Transformer Puzzles) — Sasha Rush](https://github.com/srush/Transformer-Puzzles) — puzzles in RASP, a language for what fixed-depth attention can compute, a companion to the expressivity arguments of :numref:`sec_what-attention-computes`.

**Courses**

- [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — lectures and an assignment on self-attention, positions, and transformer components, the material of :numref:`sec_attention-scoring-functions` through :numref:`sec_positional-information`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10]{.kicker}

Reading every position<br>
**soft lookup · scoring and masks · heads · positions · cost · circuits**
:::
:::

::: {.slide title="A Fixed-Size State Must Guess What Will Be Asked"}
[The problem]{.kicker}

- A recurrent model that has read 1,024 tokens keeps 768 numbers about
  them (the width of GPT-2 small).
- They are written before anyone knows which token a later prediction
  needs.
- Attention keeps one vector per position and lets each query score every
  key, weight the scores, and average the values.
- New problems: no sense of order, a cost quadratic in the length, and
  weights that do not explain themselves.

::: {.d2l-note}
How do we score, mask, split, and position a lookup over every position,
keep its cost down, and find out what a trained head does?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- implement scaled dot-product attention, as a soft lookup, with padding
  and causal masks;
- prove what one head cannot express, and build multi-head attention with
  shared keys and values;
- encode position with absolute embeddings, RoPE, or ALiBi, and extend a
  trained context;
- compute exact attention without its score matrix, and cut its cost with
  windows or linear attention;
- read QK and OV circuits from weights and test them by ablation.

**Running example:** one head of GPT-2 small, width 64 in a model of width
768, reading a context of 1,024 tokens.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| Queries, Keys, and Values | how can a lookup be made differentiable? |
| Attention Scoring and Masking | which score, and which pairs may count? |
| Multi-Head and Cross-Attention | what can one weighting per query not express? |
| Positional Information | how does an order-blind operation learn position? |
| The Cost of Attention | how do we cut the cost of all pairs of positions? |
| Attention Circuits | what do trained heads compute, and how can we tell? |
:::

::: {.slide title="Every Section Tests Its Claims on a Small, Checkable Example"}
[Running examples]{.kicker}

| section | example |
|---|---|
| Queries, Keys, and Values | Nadaraya–Watson regression on noisy 1-D data |
| Attention Scoring and Masking | random queries and keys; padded and causal masks |
| Multi-Head and Cross-Attention | a task that needs two values per query |
| Positional Information | `TinyCharLM` on WikiText-103 bytes, five schemes |
| The Cost of Attention | one GPT-2-small head; timings up to 32,768 tokens |
| Attention Circuits | `TinyCharLM` on repeated random patterns |
:::
