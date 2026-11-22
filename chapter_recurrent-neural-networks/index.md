# Sequence Models
:label:`chap_rnn`

Having spent two chapters on images, we close this part of the book by turning
to sequential data. Every model we have built so far expects an input of fixed
shape—a tabular vector of attributes or a $28 \times 28$ grid of pixels—and
treats each example independently. A sentence, an audio recording, a video, or
a stream of sensor measurements behaves quite differently: its length varies
from one example to the next, and its entries depend strongly on one another
and on the order in which they arrive. Consider the opening sentence of H. G.
Wells' *The Time Machine*, which serves as our running dataset throughout this
chapter: "The Time Traveller (for so it will be convenient to speak of him)
was expounding a recondite matter to us." Guessing the final word requires
reading the nineteen words before it in order. How can a single model make
that prediction when the preceding text might be arbitrarily long?

Claude Shannon framed this question as a guessing game :cite:`Shannon.1951`,
estimating the predictability of printed English by asking human readers to
guess a passage one character at a time. A *language model* plays the same
game: it assigns a probability to the next *token*—whether a character, a
subword, or a whole word—given the tokens that preceded it, and can generate
new text by feeding each chosen token back in as its next input.

Because the history already read grows with every step, restricting attention
to a fixed window of recent tokens discards everything earlier. A *recurrent
neural network* (RNN) instead maintains a fixed-size *hidden state* that
summarizes the sequence read so far, updating that state with the same
parameters after every token. Even where modern large language models replace
recurrence with attention (:numref:`chap_attention`) or structured state space
models (:numref:`chap_modern_rnn`), the core machinery of this
chapter—subword tokenization, perplexity and bits-per-byte evaluation, and
autoregressive decoding—carries over unchanged.

We begin by formulating autoregressive sequence forecasting and observing why
multistep rollouts degrade much faster than one-step predictions
(:numref:`sec_sequence`). Next, we convert raw text into subword token
sequences via byte-level Byte Pair Encoding (:numref:`sec_text-sequence`) and
build count-based $n$-gram language models alongside the perplexity and
bits-per-byte metrics (:numref:`sec_language-model`). We then formulate and
implement recurrent neural network language models from scratch and with
high-level APIs (:numref:`sec_rnn`), stabilizing training with gradient
clipping and comparing the learned representations against Kneser--Ney
$n$-grams. Finally, :numref:`sec_bptt` derives backpropagation through time to
explain why vanilla RNN gradients vanish or explode, and
:numref:`sec_decoding` studies greedy search, beam search, and truncated
sampling methods for generating text.

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
bptt
decoding
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §8]{.kicker}

Sequence Models and Recurrent Neural Networks<br>**autoregression · BPE subword tokenization · language models · perplexity & bits per byte · RNNs · BPTT · decoding**
:::
:::

::: {.slide title="Why Fixed-Shape Feedforward Models Fail on Sequences"}
[Motivation & Core Problem]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unbounded causal context:** Completing the 20-word opening sentence of H. G. Wells' *The Time Machine* requires integrating nineteen preceding words in exact temporal order.
- **Combinatorial context explosion:** Tracking 20 tokens over a 1,024-token vocabulary yields $1024^{20} \approx 10^{60}$ contexts; even for 3-word triples ($n = 3$), $87\%$ of held-out triples never appear in training.
- **Bounded-state compression:** A sequence model must compress an arbitrarily long prefix $x_{<t} = (x_1, \ldots, x_{t-1})$ into a fixed-size representation with constant parameter count and $O(1)$ compute per step.
:::

::: {.col .fig .big}
@fig:ch08-rnn-fixed-vs-seq
:::
:::
:::

::: {.slide title="Chapter 8 Roadmap: From Time Series to Decoding"}
[Architectural Progression]{.kicker}

::: {.cols .vc}
::: {.col}
- **Data & representation (§§8.1–8.2):** Formulate chain-rule autoregression ($m$-step windows vs. latent state $\mathbf{h}_t$) and convert raw UTF-8 strings into subword IDs via pre-tokenized Byte Pair Encoding (BPE).
- **Probabilistic scoring & RNNs (§§8.3–8.4):** Benchmark Kneser–Ney $n$-grams in tokenizer-invariant bits per byte ($\textrm{bpb}$), then replace count tables with a shared-weight $\tanh$ recurrent cell stabilized by global gradient clipping.
- **Credit assignment & decoding (§§8.5–8.6):** Derive Backpropagation Through Time (BPTT) to explain why gradients vanish or explode, and compare beam search against truncated samplers ($\text{top-}k$, $\text{top-}p$, $\text{min-}p$).
:::

::: {.col .fig .big}
@fig:ch08-rnn-roadmap
:::
:::
:::

::: {.slide title="End-to-End Quantitative Benchmarks on Wells' Novella"}
[Empirical Anchors]{.kicker}

::: {.cols .vc}
::: {.col}
- **Rollout drift & subword compression (§§8.1–8.2):** Linear AR($4$) achieves $0.05$ 1-step MSE on a noisy sine wave but degrades $5\times$ to $0.25$ on 64-step rollout; BPE-1024 compresses *The Time Machine* to $2.78$ bytes/token with zero OOVs.
- **Count vs. neural language models (§§8.3–8.4):** Kneser–Ney cuts word trigram PPL from $2{,}481 \to 300$ ($1.90$ bpb on BPE-1024); clipped `RNNLMScratch` ($512$-byte state, $89\%$ params in vocabulary layers) reaches $2.12$ bpb.
- **Gradient horizon & generation (§§8.5–8.6):** Despite $\rho(\mathbf{W}_{\textrm{hh}}) \approx 1.80$, $\tanh'$ contraction ($\approx 0.27$) decays gradients below $1\%$ within $8$ steps; truncated samplers restore `distinct-3` $\approx 0.99$.
:::

::: {.col .fig .big}
@fig:ch08-rnn-wells-benchmarks
:::
:::
:::

::: {.slide title="Concept Check: Designing an End-to-End Language Model"}
[Check Your Understanding]{.kicker}

Before diving into §8.1, test how the components of the sequence modeling pipeline interact:

1. **Tokenizer vs. Perplexity:** Model A uses a 27-character vocabulary and achieves test perplexity $\textrm{PPL} = 6.5$. Model B uses a 1,024-token BPE vocabulary ($2.78$ bytes/token) and achieves $\textrm{PPL} = 60.0$. Which model compresses the test text better in bits per byte?
2. **State Memory vs. Sequence Length:** How many bytes of hidden-state memory does a single-layer `float32` RNN with $h = 128$ units require to process a 32-token window versus a 1,000,000-token book?
3. **Training vs. Inference Mismatch:** Why can an autoregressive model achieve low 1-step cross-entropy loss during teacher-forced training yet degenerate into repetitive or off-topic text during free generation?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Model B wins:** Character Model A (1 byte/char) uses $\log_2(6.5) / 1.0 \approx 2.70$ bits/byte, whereas BPE Model B uses $\log_2(60.0) / 2.78 \approx 5.91 / 2.78 \approx 2.13$ bits/byte. Raw perplexity is incomparable across different token granularities.
2. **Identical ($512$ bytes):** $128 \times 4\text{ bytes} = 512\text{ bytes}$ per sequence regardless of length $T$, because $\mathbf{h}_t$ overwrites $\mathbf{h}_{t-1}$ in place at every step.
3. **Exposure bias & rollout compounding:** Teacher forcing always feeds ground-truth prefixes $x_{<t}$, whereas generation feeds back self-sampled tokens $\hat{x}_t$; a single low-probability tail token or deterministic cycle pushes the state $\mathbf{h}_t$ out of the training distribution.
:::
:::

::: {.slide title="Key Takeaways: The Sequence Modeling Pipeline"}
[Chapter Synthesis]{.kicker}

::: {.cols .vc}
::: {.col}
- **Universal subwords & invariant scoring (§§8.2–8.3):** Pre-tokenized byte-level BPE eliminates `<unk>` tokens while reserving special IDs above $|\mathcal{V}_{\textrm{BPE}}|$; bits per byte ($\textrm{bpb}$) compares any tokenization on a common lossless-compression scale.
- **Constant-memory recurrence & BPTT (§§8.4–8.5):** Shared weights update a fixed-size state $\mathbf{H}_t \in \mathbb{R}^{b \times h}$ in $O(1)$ per step; truncated BPTT bounds memory to $O(T_0 bh)$ and global norm clipping prevents exploding gradient steps.
- **Controlled autoregressive decoding (§8.6):** Length-normalized beam search serves constrained tasks, while temperature + $\text{top-}p / \text{min-}p$ tail truncation eliminates repetition loops in open-ended generation.
:::

::: {.col .fig .big}
@fig:ch08-rnn-pipeline-summary
:::
:::
:::

