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

Feedforward networks and CNNs assume fixed-dimensional inputs $\mathbf{x} \in \mathbb{R}^d$ drawn independently, whereas text, speech, and time series vary in length and depend strictly on temporal order:

- **Unbounded causal context:** Completing the 20-word opening sentence of H. G. Wells' *The Time Machine* (*"The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to ___"*) requires integrating nineteen preceding words in exact order.
- **Combinatorial context explosion:** Tracking 20 tokens of history over a modest 1,024-token vocabulary yields $1024^{20} \approx 10^{60}$ distinct contexts; even for 3-word triples ($n = 3$), $87\%$ of held-out triples in *The Time Machine* never appear in training.
- **Bounded-state compression:** A sequence model must compress an arbitrarily long prefix $x_{<t} = (x_1, \ldots, x_{t-1})$ into a fixed-size representation with constant parameter count and $O(1)$ compute per step.

::: {.d2l-note}
**The Unifying Pipeline:** Even when modern LLMs replace recurrent cells with self-attention or structured state space models, the core pipeline of this chapter—**byte-level BPE tokenization**, **chain-rule language modeling**, **bits-per-byte evaluation**, and **autoregressive decoding**—transfers unchanged.
:::
:::

::: {.slide title="Chapter 8 Roadmap: From Time Series to Decoding"}
[Architectural Progression]{.kicker}

Each section builds one layer of an end-to-end autoregressive sequence modeling and generation system:

| Section | Central Question | Core Mathematical & Systems Mechanism |
|---|---|---|
| **§8.1 Working with Sequences** | How do we predict a series from its own past? | Chain-rule factorization, $m$-step Markov windows vs. latent state $\mathbf{h}_t$, 1-step vs. $k$-step rollout drift |
| **§8.2 From Text to Tokens** | How do we convert raw strings into integer IDs? | Character vs. word granularity trade-off, UTF-8 byte alphabet, Byte Pair Encoding (BPE), regex pre-tokenization |
| **§8.3 Language Models** | How do we estimate $P(x_{1:T})$ and compare tokenizers? | Count-based $n$-grams, Zipf's law sparsity, Laplace vs. interpolated Kneser–Ney, Perplexity ($\textrm{PPL}$) & Bits per Byte ($\textrm{bpb}$) |
| **§8.4 Recurrent Neural Networks** | How does a learned hidden state replace fixed windows? | $\mathbf{H}_t = \tanh(\mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}})$, teacher forcing, global gradient clipping $\min(1, \theta/\|\mathbf{g}\|)$ |
| **§8.5 Backpropagation Through Time** | Why do vanilla RNN gradients vanish or explode? | Unrolled backward recursion $\mathbf{G}_t = \nabla_{\mathbf{O}_t} L\,\mathbf{W}_{\textrm{hq}}^\top + \mathbf{G}_{t+1}\mathbf{W}_{\textrm{hh}}^\top$, matrix powers $(\mathbf{W}_{\textrm{hh}}^\top)^{j-t}$, truncated BPTT |
| **§8.6 Decoding and Generation** | How do we turn next-token distributions into fluent text? | Greedy argmax cycles, length-normalized beam search, temperature $\tau$, and tail truncation (top-$k$, top-$p$, min-$p$) |
:::

::: {.slide title="End-to-End Quantitative Benchmarks on Wells' Novella"}
[Empirical Anchors]{.kicker}

Every section validates its mathematical analysis against concrete measurements on *The Time Machine* (~30,000 words, 179,462 UTF-8 bytes, 65,741 BPE-1024 tokens):

| Section | Experimental Setup | Key Empirical Finding |
|---|---|---|
| **§8.1 Sequences** | Linear AR ($m = 4$) on $x_t = \sin(0.01t) + \epsilon_t$ ($\sigma = 0.2$) | 1-step MSE $\approx 0.05$ (noise floor), but 64-step rollout MSE degrades $5\times$ to $\approx 0.25$ as learned moving average damps oscillation |
| **§8.2 BPE Tokens** | Byte-level BPE ($|\mathcal{V}| = 1{,}024 + 3$ specials) with `GPT2_PATTERN` | Matches OpenAI `tiktoken` (`gpt2`) token-for-token; achieves $2.78$ bytes/token ($67\%$ of 50k/200k production vocabularies) with zero OOVs |
| **§8.3 $n$-Grams** | Unigram, bigram, trigram across characters, BPE-1024, and words | $87\%$ of validation word trigrams are unseen; Kneser–Ney cuts word trigram PPL from $2{,}481 \to 300$ and BPE-1024 trigram to $1.90$ bpb |
| **§8.4 RNN LM** | `RNNLMScratch` & `RNNLM` ($d = 64 \to 512$, $h = 128$, $T = 32$) | $222{,}915$ baseline parameters ($89\%$ in embedding/output layers, $512$-byte state); clipping at $\theta = 1$ prevents explosion and reaches $\approx 2.1$ bpb |
| **§8.5 BPTT** | Backward Jacobian norm probe across $L = 64$ validation steps | Despite spectral radius $\rho(\mathbf{W}_{\textrm{hh}}) \approx 1.8$, mean $\tanh'$ factor $0.27$ causes gradient norm to decay below $1\%$ within $8$ steps |
| **§8.6 Decoding** | 100-token rollouts across 20 prompts under 6 decoding strategies | Greedy and beam ($w = 4$) maximize likelihood but collapse into loops (`distinct-3` $\ll 0.5$); top-$k$/top-$p$/min-$p$ restore `distinct-3` $\approx 0.99$ |
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

| Pipeline Stage | Classical / Naive Approach | Modern / Neural Solution in Chapter 8 | Core Invariant / Equation |
|---|---|---|---|
| **1. Tokenization (§8.2)** | Fixed word list ($+\texttt{<unk>}$) or raw characters | Pre-tokenized byte-level BPE over 256 UTF-8 bytes | Replay learned pair merges in ascending rank order; special IDs sit above $|\mathcal{V}_{\textrm{BPE}}|$ |
| **2. Sequence Probability (§8.1, §8.3)** | Unsmoothed count tables over $m$-token Markov windows | Chain-rule product of next-token conditionals smoothed or parameterized by neural state | $P(x_{1:T}) = \prod_{t=1}^T P(x_t \mid x_{<t})$, evaluated via $\textrm{bpb} = \frac{1}{B}\sum_{t=1}^T -\log_2 P(x_t \mid x_{<t})$ |
| **3. Recurrent Representation (§8.4)** | Lookup table or $m$-gram concatenation growing with window | Shared-weight recurrent cell updating a fixed-size latent summary $\mathbf{H}_t \in \mathbb{R}^{b \times h}$ | $\mathbf{H}_t = \tanh(\mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}})$, $\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_{\textrm{hq}} + \mathbf{b}_{\textrm{q}}$ |
| **4. Credit Assignment (§8.5)** | Full-corpus backprop or unclipped SGD updates | Truncated BPTT across detached windows of length $T_0$ with global norm clipping | $\mathbf{G}_t = \nabla_{\mathbf{O}_t} L\,\mathbf{W}_{\textrm{hq}}^\top + (\mathbf{G}_{t+1} \odot (1 - \mathbf{H}_{t+1}^2))\mathbf{W}_{\textrm{hh}}^\top$; clip $\mathbf{g} \leftarrow \min(1, \theta/\|\mathbf{g}\|)\mathbf{g}$ |
| **5. Autoregressive Decoding (§8.6)** | Greedy argmax or raw unpruned multinomial sampling | Length-normalized beam search (constrained tasks) or temperature + top-$k$/top-$p$/min-$p$ (open-ended) | Truncate unreliable tail of $P_\tau(x \mid x_{<t}) \propto \exp(o_x/\tau)$, renormalize, and carry state $\mathbf{H}_t$ in $O(1)$ per step |
:::

