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

::: {.cols .vc}
::: {.col}
- **Causal Attention ($O(T)$ KV Cache):** Stores 4 KB/token ($8$ KV heads $\times 128$ fp16), growing to **$512\text{ MB}$/layer** at $T = 128\text{K}$ with $O(T^2 d)$ score FLOPs.
- **Recurrent State ($O(1)$ Memory):** Compresses history into a fixed **$2\text{ MB}$** state (Mamba-2 at $d=4{,}096$), crossing the KV cache at $T = 512$ tokens.
- **Central Question:** Can an $O(1)$ state train in parallel on GPUs and match Transformer recall?
:::

::: {.col .fig .big}
@fig:ch12-mrnn-kv-vs-state
:::
:::
:::

::: {.slide title="Two Obstacles to Modernizing Recurrent Networks"}
[Core Challenges]{.kicker}

::: {.cols .vc}
::: {.col}
- **Obstacle 1 — $O(T)$ Sequential Depth:** Nonlinear $\tanh(\mathbf{W}_{\textrm{hh}}\mathbf{h}_{t-1})$ blocks GPU parallelism; affine transitions (§{sec_linear-recurrence}–§{num:sec_mamba}) run in $O(\log_2 T)$ parallel scan rounds.
- **Obstacle 2 — Fixed-Vector Decay:** A 1D state $\mathbf{h}_t \in \mathbb{R}^d$ blurs facts; multi-timescale $\Delta_t$, matrix states $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$, and the delta rule (§{sec_ssm}–§{num:sec_deltanet}) add key-value addressing.
- **Unifying State-Space View:** Every mixer has a **state**, a **write update**, and a **readout**.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-two-obstacles
:::
:::
:::

::: {.slide title="Architectural Progression Across Chapter §{num:chap_modern_rnn}"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **From Gates to Linear Scans (§{sec_lstm}–§{num:sec_mamba}):** Progresses from sequential LSTM/GRU gates and Seq2Seq bottlenecks to parallel `minGRU` scans, multi-timescale `S4D`, and selective `Mamba`.
- **From Vectors to Matrix Memories (§{sec_matrix-state}–§{num:sec_deltanet}):** Expands the state into an outer-product key–value matrix (`GLA`) sharpened by online read-before-write error correction (`GatedDeltaNet`).
- **Sparse Attention Hybrids (§{sec_hybrids}):** Interleaves 1 full attention layer per 3 recurrent layers to break the Welch overlap ceiling.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-roadmap
:::
:::
:::

::: {.slide title="Synthetic Stress Tests and the Language Benchmark"}
[Empirical Methodology]{.kicker}

::: {.cols .vc}
::: {.col}
- **Five Targeted Capability Probes:** Delayed Recall (half-life horizon), String Reversal (bits/float capacity), Selective Copying (token-dependent $\Delta_t$), Associative Recall (Welch rank limit), and Key Rebinding (delta rule erasure).
- **Unified Language Benchmark (`SeqLM`):** Every sequence mixer is evaluated inside the exact same 4-block pre-norm residual stack on byte-level WikiText-103.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-stress-tests
:::
:::
:::

::: {.slide title="How to Navigate Chapter §{num:chap_modern_rnn}"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core architectural line:** Follow **§{sec_lstm} (Gates)** $\to$ **§{sec_linear-recurrence} (Linear Recurrences & Parallel Scans)** $\to$ **§{sec_mamba} (Selective SSMs / Mamba)** $\to$ **§{sec_matrix-state}–§{num:sec_deltanet} (Matrix Memories & DeltaNet)** $\to$ **§{sec_hybrids} (Hybrid Architectures)**.
- **Historical & information-theoretic foundation (§{sec_seq2seq}):** Revisits the 2014 encoder–decoder and translation metrics (`chrF`, `BLEU`) to measure how rapidly a single state vector saturates as input length grows.
:::

::: {.col}
- **Mathematical derivations that can wait on first reading:** Continuous-time zero-order-hold discretization in **§{sec_ssm}** and the block-semiseparable chunked algorithms in **§{sec_matrix-state}–§{num:sec_deltanet}**.
- **Controlled code comparisons:** Every mixer plugs into the identical pre-norm residual `SeqLM` backbone with LeCun initialization so differences reflect the sequence mixer alone.
:::
:::

::: {.d2l-note}
**Up next (§{sec_lstm} Gated Recurrent Networks):** How sigmoid gates create a learned elementwise decay path—and why gate bias initialization decides whether training discovers long-range memory.
:::
:::

