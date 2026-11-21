# Attention
:label:`chap_attention`

The recurrent models of :numref:`chap_rnn` read a sequence one token at a
time and compress everything they have seen into a hidden state of fixed
size. If a prediction near the end of a long document depends on a detail
from the first paragraph, the recurrent network must have written that detail
into its hidden state long before knowing it would be needed. Rather than
compressing history into a single vector, can a network retain a
representation at every position and retrieve only what the current
prediction requires?

*Attention* does exactly this. At each lookup, a *query* vector representing
what the current position seeks is scored against a *key* vector at every
candidate position. A softmax converts those scores into nonnegative weights
that sum to one, and the layer returns the weighted average of the *value*
vectors stored at those positions. Because a weighted average is smooth,
gradient descent can train the projections that produce queries, keys, and
values end to end. Running several such lookups in parallel gives *multi-head
attention*. Introduced for neural machine translation so a decoder could
consult every source word directly :cite:`Bahdanau.Cho.Bengio.2014`,
attention became the sole position-mixing operation in the Transformer
:cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Replacing recurrence with all-pairs lookup introduces three new challenges:

1. **Order blindness:** A weighted sum is invariant to permutations of its
   inputs, so positional information must be injected explicitly.
2. **Quadratic scaling:** Scoring all pairs of $n$ tokens requires $O(n^2)$
   operations and memory, motivating both IO-aware exact kernels and sub-quadratic
   approximations.
3. **Mechanistic interpretation:** Attention weights show *where* a head
   reads, not *what* function it computes across layers.

The first three sections construct the layer: differentiable key--value
lookup (:numref:`sec_queries-keys-values`), scaled dot-product scoring and
boolean masking (:numref:`sec_attention-scoring-functions`), and multi-head
self-attention (:numref:`sec_multihead-attention`).
:numref:`sec_positional-information` addresses sequence order through
absolute, rotary (RoPE), and relative (ALiBi) positional encodings. The next
two sections address computational scale: :numref:`sec_attention-at-scale`
derives tiled online-softmax kernels (FlashAttention) that avoid
materializing the $n \times n$ score matrix, and
:numref:`sec_windowed-linear-attention` examines sliding-window and linear
attention variants. Finally, :numref:`sec_what-attention-computes` analyzes
trained attention circuits through the residual stream, QK/OV decompositions,
and two-layer induction heads, and :numref:`sec_attention-further-reading`
surveys extensions and primary literature.

Readers heading directly to :numref:`chap_transformers` need only the first
four sections and the residual-stream formulation at the start of
:numref:`sec_what-attention-computes`; the sections on computational cost can
be read alongside :numref:`sec_kv-cache` and :numref:`chap_modern_rnn`.

```toc
:maxdepth: 2

queries-keys-values
attention-scoring
multihead-attention
positional-information
attention-at-scale
windowed-linear-attention
what-attention-computes
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10]{.kicker}

Attention Mechanisms: Differentiable Content-Based Retrieval<br>**queries, keys & values · scaled dot-product · multi-head & GQA · RoPE & ALiBi · FlashAttention · linear attention · circuits**
:::
:::

::: {.slide title="From Fixed-Size State Bottlenecks to Direct Lookup"}
[Motivation]{.kicker}

Recurrent networks compress an entire sequence into a single fixed-width hidden vector before knowing which early details a future prediction will require:

::: {.cols .vc}
::: {.col}
- **The Recurrent Bottleneck:** A model of width $d = 768$ (GPT-2 small) reading $n = 1{,}024$ tokens must squeeze $1{,}024 \times 768 \approx 786{,}000$ numbers of history into $768$ floats.
- **Premature Compression:** Details from paragraph one must be written into $\mathbf{h}_t$ long before the final query arrives, across an $O(n)$-step gradient chain.
:::

::: {.col}
- **Content-Based Retrieval:** Instead of summarizing history into one vector, keep one representation at every position and retrieve relevant context on demand.
- **Differentiable Soft Lookup:** Replace discontinuous database matching with a softmax-weighted average of stored values, trainable end to end by backpropagation.
:::
:::

::: {.d2l-note}
Introduced for neural machine translation (Bahdanau et al., 2014), attention became the sole token-mixing operator in the Transformer (Vaswani et al., 2017).
:::
:::

::: {.slide title="The Core Primitive: Queries, Keys, and Values"}
[Core Mechanism]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three Distinct Roles:** Each lookup compares a **query** $\mathbf{q} \in \mathbb{R}^d$ (what the current position seeks) against $m$ **keys** $\mathbf{k}_i \in \mathbb{R}^d$ (what each candidate position offers).
- **Convex Value Mixing:** Compatibility scores $a(\mathbf{q}, \mathbf{k}_i)$ pass through a softmax to form nonnegative weights $\alpha_i$ summing to $1$:
  $$\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\,\mathbf{v}_i$$
- **Parallel Heads:** One lookup forms an attention **head**; running $H$ heads side by side lets a token retrieve multiple arguments at once.
:::

::: {.col .fig}
![Soft lookup scores a query against keys and returns a convex combination of values.](../img/mdl-attention-soft-lookup.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Three Challenges Created by All-Pairs Lookup"}
[Architectural Trade-Offs]{.kicker}

Replacing sequential recurrence with all-pairs content lookup eliminates the $O(n)$ path length but creates three new engineering and theoretical challenges:

| Challenge | Why It Happens in Attention | Solutions Developed in Chapter 10 |
| :--- | :--- | :--- |
| **1. Order Blindness** | A weighted sum $\sum_i \alpha_i \mathbf{v}_i$ is permutation equivariant and ignores word order | Absolute tables, rotary embeddings (**RoPE**), and distance penalties (**ALiBi**) in **§10.4** |
| **2. Quadratic Cost** | Scoring all pairs of $n$ tokens costs $\Theta(n^2 d)$ FLOPs and $O(n^2)$ naive memory | Tiled **online softmax** (**FlashAttention**, **§10.5**), sliding windows, and **linear attention** (**§10.6**) |
| **3. Mechanistic Opacity** | Attention weights show *where* a head looks, not *what* function it computes | Residual stream, **QK/OV circuit** decomposition, and two-layer **induction heads** (**§10.7**) |

::: {.d2l-note}
At $n = 1{,}024$, a single head evaluates $1{,}024^2 \approx 1.05 \times 10^6$ pairs; at $n = 131{,}072$, a naive `float32` score matrix for one head requires $64\text{ GiB}$.
:::
:::

::: {.slide title="Architecture and Roadmap of Chapter 10"}
[Chapter Roadmap]{.kicker}

We first construct the multi-head self-attention layer (§10.1–10.3), then restore sequence order (§10.4), scale to long contexts (§10.5–10.6), and analyze trained circuits (§10.7):

| Part | Section | Central Mathematical & Systems Question |
| :--- | :--- | :--- |
| **1. Build the Layer** | **§10.1 Queries, Keys, and Values** | How does soft key–value lookup unify database retrieval and kernel regression? |
| **1. Build the Layer** | **§10.2 Attention Scoring & Masking** | Why scale dot products by $1/\sqrt{d}$, and how do boolean masks compose? |
| **1. Build the Layer** | **§10.3 Self-Attention & Multi-Head** | Why does a single head lose half of two values, and how does GQA shrink caches? |
| **2. Sequence Order** | **§10.4 Positional Information** | How does RoPE enforce relative-offset scores and extrapolate beyond training length? |
| **3. Scale & Speed** | **§10.5 The Cost of Attention** | How does blockwise online softmax compute exact attention in $O(n)$ memory? |
| **3. Scale & Speed** | **§10.6 Windowed & Linear Attention** | What do sliding windows and factorized kernel recurrences trade for $O(n)$ time? |
| **4. Interpretation** | **§10.7 Attention Circuits** | How do QK and OV circuits compose across two layers to form an induction head? |
:::

::: {.slide title="Running Benchmarks and Study Guide"}
[Pedagogical Thread]{.kicker}

Every section tests its claims on an executable benchmark and counts exact FLOPs and bytes on a reference GPT-2 small head ($d = 768, H = 12, d_h = 64$):

| Section | Executable Benchmark / Model | Key Quantitative Takeaway |
| :--- | :--- | :--- |
| **§10.1–10.2** | Nadaraya–Watson 1D regression; `ToyAligner` permutation task | Bandwidth $\sigma$ and $1/\sqrt{d}$ act as softmax temperatures preventing gradient saturation |
| **§10.3** | *"Alice gave Bob the book"*; Gaussian **copy-both** task | One head incurs $\ge 1/2$ relative MSE; two heads reach $\sim 10^{-7}$ at identical $4d^2$ params |
| **§10.4** | `TinyCharLM` on WikiText-103 bytes ($n = 128$ $\to$ $512$) | RoPE gives lowest in-range perplexity ($7.3$); slow-pair interpolation repairs $4\times$ context |
| **§10.5–10.6** | Chunked online softmax; fused Flash/FlexAttention up to $n = 32\text{K}$ | Online softmax cuts memory from $O(n^2)$ to $O(n)$; window $w = 128$ gives $> 30\times$ speedup |
| **§10.7** | `TinyCharLM` on variable-period repeated random patterns | Two blocks learn in-context induction ($\sim 90\%$ accuracy); one block stalls at $\sim 10\%$ |

::: {.d2l-note}
**Fast track to Chapter 11 (Transformers):** Read **§10.1–10.4** and the residual-stream opening of **§10.7**; study **§10.5–10.6** alongside KV caching and modern RNNs.
:::
:::
