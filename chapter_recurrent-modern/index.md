# State Space Models
:label:`chap_modern_rnn`

The recurrent networks of :numref:`chap_rnn` read text one token after another and carry what they have read in a fixed-size *state*. Generating the millionth token therefore costs as much as generating the tenth. The Transformers of :numref:`chap_transformers` keep every token instead: answering a question about a long report holds a key and a value for every token of the report (:numref:`sec_kv-cache`), so both memory and per-token work grow with context length. Yet in :numref:`sec_gpt` a plain recurrent network as large as our GPT fell behind after the first few bytes of context and never caught up. Can a better recurrent network close that gap, and at what length does the growing cost of attention begin to matter?

Two obstacles stand in the way. Classical recurrent networks update their state nonlinearly, forcing training to proceed one step at a time and leaving most of a GPU idle. And a fixed-size state must forget, so it has to choose what to keep. The answers come from three directions: gates from the 1990s :cite:`Hochreiter.Schmidhuber.1997,Gers.Schmidhuber.Cummins.2000`, linear dynamical systems from control theory :cite:`Gu.Dao.Ermon.ea.2020,Gu.Goel.Re.2022`, and attention without the softmax :cite:`Katharopoulos.Vyas.Pappas.ea.2020`. They meet in recurrent layers that train in parallel and compute a form of linear attention. Every recurrent layer in this chapter has a state, an *update* that folds in each token, and a *readout* that computes the output. Control engineers call such a system a *state space model*; the chapter's title uses the name in this broad sense, while :numref:`sec_ssm` uses it in the narrow sense of a linear system in continuous time.

We begin with gates, which let a network learn what to keep, and with the 2014 encoder–decoder, whose single state vector creates a bottleneck we can measure. Three changes then make the state fast and selective: a linear update that can be evaluated across all positions in parallel and timed against attention (:numref:`subsec_linear-recurrence-cost`); multi-timescale decays discretized from continuous time; and input-dependent step sizes that let each token decide how much it writes and forgets. Extending the state from a vector to a matrix turns it into an associative memory of key–value pairs whose capacity we can calculate and whose write rule we can sharpen with the delta rule. Finally, interleaving a few full attention layers restores the exact recall that no fixed state possesses—a hybrid pattern adopted by several frontier models :cite:`Lieber.Lenz.Bata.ea.2024,Qwen.Team.2025`.

On a first reading, :numref:`sec_seq2seq`, the continuous-time derivation in :numref:`sec_ssm`, and the chunked algorithms of :numref:`sec_matrix-state` and :numref:`sec_deltanet` can wait. Small synthetic tasks, each designed to isolate one memory capability, test the mechanisms, and the byte-level text benchmark of :numref:`sec_gpt` checks how each mixer performs on language.

```toc
:maxdepth: 2

lstm
seq2seq
linear-recurrence
ssm
mamba
matrix-state
deltanet
hybrids
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12]{.kicker}

State Space Models and Modern Recurrent Architectures<br>**gated recurrences · parallel associative scans · structured & selective SSMs · matrix memories · delta rule · hybrids**
:::
:::

::: {.slide title="The Long-Context Dilemma: KV Cache vs. Fixed State"}
[Motivation]{.kicker}

Answering questions over a long report or codebase exposes a fundamental memory trade-off between attention and recurrence:

::: {.cols}
::: {.col}
- **Causal Attention ($O(T)$ KV Cache):** Retains a key and value vector for every past token. With $8$ KV heads of width $128$ in $16$-bit floats, each layer stores $2 \times 8 \times 128 \times 2 = 4\text{ KB}$ per token—reaching **$512\text{ MB}$ per layer** ($16\text{ GB}$ across $32$ layers) at $T = 128\text{K}$ tokens.
- **Quadratic training work:** Pairwise scores grow as $O(T^2 d)$, dominating projection FLOPs once $T \gg d$.
:::

::: {.col}
- **Modern Recurrent Layers ($O(1)$ State):** Compress past context into a fixed-size state. A Mamba-2 layer at width $4{,}096$ ($8{,}192$ channels $\times$ $128$ states in $16$-bit floats) stores **$2\text{ MB}$ total** at any sequence length.
- **Crossover at 512 tokens:** Beyond $2\text{ MB} / 4\text{ KB} = 512$ tokens, the recurrent state is smaller, and generating token $1{,}000{,}000$ costs the same as token $10$.
:::
:::

::: {.d2l-note}
Can a fixed-size recurrent state train in parallel on GPUs and match Transformer quality—and what can no fixed state ever recall?
:::
:::

::: {.slide title="Two Obstacles to Modernizing Recurrent Networks"}
[Core Challenges]{.kicker}

Classical RNNs fell behind Transformers for two structural reasons that this chapter resolves step by step:

::: {.cols}
::: {.col}
- **Obstacle 1 — Sequential Training ($O(T)$ Depth):** Classical RNNs, LSTMs, and GRUs pass $\mathbf{h}_{t-1}$ through nonlinear activations at every step, forcing $T$ sequential updates that leave GPU Tensor Cores idle.
- **The Parallel Fix (§12.3–12.5):** Make the state transition *affine* in the state so steps compose via parallel associative scans in $O(\log_2 T)$ rounds or FFT convolutions.
:::

::: {.col}
- **Obstacle 2 — Fixed-Capacity Forgetting:** A fixed state vector overwrites old signals or superposes values without key-addressable lookup or targeted erasure.
- **The Memory Fix (§12.4–12.8):** Discretize multi-timescale decay rates, make step sizes input-selective, expand the state into a key–value matrix updated by the delta rule, and interleave sparse attention.
:::
:::

::: {.d2l-note .rule}
**Unifying Abstraction:** Every sequence mixer in this chapter is defined by a **state**, a **transition/write update** folding in token $t$, and a **readout** producing the output.
:::
:::

::: {.slide title="Architectural Progression Across Chapter 12"}
[Chapter Roadmap]{.kicker}

Each section upgrades one mathematical component of the recurrent state machine:

| Section | Core Innovation | State Transition & Write Rule | Training Algorithm |
| :--- | :--- | :--- | :--- |
| **§12.1 Gated RNNs** | Learned elementwise decay gates (LSTM, GRU) | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ | Sequential ($T$ steps) |
| **§12.2 Seq2Seq** | Single-vector encoder–decoder bottleneck | $\mathbf{c} = \mathbf{h}_T^{\text{enc}} \to \mathbf{s}_{t'} = g(y_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1})$ | Sequential ($T + T'$ steps) |
| **§12.3 Linear Recurrence** | Input-only gates (`MinGRU`) & associative scan | $\mathbf{h}_t = \mathbf{a}(\mathbf{x}_t) \odot \mathbf{h}_{t-1} + \mathbf{b}(\mathbf{x}_t)$ | Parallel scan ($\lceil\log_2 T\rceil$ rounds) |
| **§12.4 State Space Models** | Discretized ODE rates $\lambda_n$ & step $\Delta$ (`S4D`) | $h_{t,n} = e^{\Delta \lambda_n} h_{t-1,n} + \bar{b}_n x_t$ | FFT convolution or scan |
| **§12.5 Selective SSMs** | Token-dependent $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ (`Mamba`) | $\mathbf{S}_t = \mathbf{E}_t \odot \mathbf{S}_{t-1} + (\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\mathbf{B}_t^\top$ | Parallel scan + reverse-scan VJP |
| **§12.6 Matrix-State RNNs** | Outer-product KV memory & duality (`GLA`) | $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | Chunked matmuls + scan |
| **§12.7 The Delta Rule** | Read-before-write error step (`GatedDeltaNet`) | $\mathbf{S}_t = a_t \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | Chunked triangular solve + scan |
| **§12.8 Hybrid Models** | Sparse full attention among recurrent blocks | Fixed matrix states + sparse KV caches | Chunked recurrence + attention |
:::

::: {.slide title="Synthetic Stress Tests and the Language Benchmark"}
[Empirical Methodology]{.kicker}

We pair synthetic tasks that isolate individual memory mechanisms with a unified byte-level language model (`SeqLM`):

| Benchmark Task | Task Structure | What It Measures & Where It Appears |
| :--- | :--- | :--- |
| **Delayed Recall** | 1 target symbol followed by $T-1$ fillers ($T \le 1{,}000$) | Gradient propagation & initial decay timescales (**§12.1, §12.4**) |
| **String Reversal** | Reversing $T \in [8, 64]$ random letters through 1 state vector | Effective bits packed per state coordinate (**§12.2**) |
| **Selective Copying** | 4 marked symbols at random positions among $1{,}024$ fillers | Content-selective gating vs. time-invariant lag filters (**§12.5**) |
| **Associative Recall** | $n \in [4, 64]$ key–value pairs queried in random order | Outer-product capacity $O(d_k d_v)$ & Welch overlap floor (**§12.6, §12.8**) |
| **Key Rebinding** | Overwriting 8 of 16 keys with new values after a short gap | Error-correcting delta rule vs. additive superposition (**§12.7**) |
| **WikiText-103 Bytes** | 4-block pre-norm `SeqLM` ($d = 256$, $128$-byte windows) | Controlled language modeling loss (nats/byte) & step latency (**§12.1–12.8**) |
:::

::: {.slide title="How to Navigate Chapter 12"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core architectural line:** Follow **§12.1 (Gates)** $\to$ **§12.3 (Linear Recurrences & Parallel Scans)** $\to$ **§12.5 (Selective SSMs / Mamba)** $\to$ **§12.6–12.7 (Matrix Memories & DeltaNet)** $\to$ **§12.8 (Hybrid Architectures)**.
- **Historical & information-theoretic foundation (§12.2):** Revisits the 2014 encoder–decoder and translation metrics (`chrF`, `BLEU`) to measure how rapidly a single state vector saturates as input length grows.
:::

::: {.col}
- **Mathematical derivations that can wait on first reading:** Continuous-time zero-order-hold discretization in **§12.4** and the block-semiseparable chunked algorithms in **§12.6–12.7**.
- **Controlled code comparisons:** Every mixer plugs into the identical pre-norm residual `SeqLM` backbone with LeCun initialization so differences reflect the sequence mixer alone.
:::
:::

::: {.d2l-note}
**Up next (§12.1 Gated Recurrent Networks):** How sigmoid gates create a learned elementwise decay path—and why gate bias initialization decides whether training discovers long-range memory.
:::
:::

