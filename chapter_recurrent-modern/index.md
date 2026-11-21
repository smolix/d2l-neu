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

A memory of fixed size<br>
**trained in parallel, taught what to forget, made to store and correct pairs, and helped by a little attention**
:::
:::

::: {.slide title="A Transformer's Cache Grows With the Context; a Recurrent State Does Not"}
[The problem]{.kicker}

- Attention keeps a key and a value per token. With 8 key–value heads of
  width 128 in 16-bit numbers: $2 \times 8 \times 128 \times 2$ bytes
  $= 4$ KB per token and layer, 512 MB at 128K tokens.
- A recurrent layer of a width-4,096 model, with the sizes of Mamba-2,
  keeps 8,192 channels of 128 numbers: $8{,}192 \times 128 \times 2$ bytes
  $= 2$ MB, at any length.

**Check:** after how many tokens does the cache outgrow the state?

::: {.fragment .fade-up}
::: {.d2l-note}
$2\text{ MB} / 4\text{ KB} = 512$ tokens. Beyond that, the recurrent layer
holds less, and the millionth token costs as much as the tenth.
:::
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- explain why a gate keeps a value for hundreds of steps, and why its
  starting value decides whether training finds that memory;
- compute a linear recurrence at all $T$ positions in $\log_2 T$ rounds;
- set a state's decays by timescale, and let each token choose them;
- store key–value pairs in a matrix state, predict how many it holds, and
  correct a pair by reading before writing;
- say what no fixed state can recall, and what a few attention layers
  cost and repair.

**Idea:** every model is a state, an update and a readout.
:::

::: {.slide title="Gates, the Linear Update and the Matrix Memory Are the Core"}
[Scope and place]{.kicker}

- **Assumes:** recurrent networks and BPTT; attention, including linear
  attention.
- **Core:** gates as a learned decay; the parallel linear update; the
  matrix memory and its capacity; the delta rule (read before writing);
  recall of many pairs, with and without attention.
- **History:** the 2014 encoder–decoder and its one-state bottleneck.
- **Can wait:** the derivation from continuous time; the chunked
  algorithms.
- **Leaves out:** GPU kernels, rankings of current models.
:::

::: {.slide title="Make the Update Linear, Then Selective, Then Correct; Then Put Attention Back"}
[Chapter map]{.kicker}

| section | question | answer |
|---|---|---|
| Gated Recurrent Networks | how to keep a value for long? | gates, a learned decay |
| Sequence-to-Sequence Learning | how much of an input fits through one state? | less as the input grows |
| Linear Recurrences and Parallel Scans | how to train on all positions at once? | a linear update and a scan |
| State Space Models | how fast should a state forget? | decays set by timescale |
| Selective State Space Models | how to keep only what matters? | step sizes chosen by the token |
| Matrix-State Recurrences | how to store pairs, and how many? | outer products; capacity set by size |
| The Delta Rule | how to overwrite a pair? | read before writing |
| Hybrid Architectures | what can no fixed state do? | recall of many pairs; a few attention layers |
| Further Reading | what does the chapter leave out? | practice, other memories, theory, other domains |
:::

::: {.slide title="Small Synthetic Tasks Test Memory; One Text Benchmark Tests Language"}
[Running examples]{.kicker}

- **Delayed recall:** a symbol, then random fillers, then name the symbol.
- **Reversal:** read a random string, then write it backwards.
- **Selective copying:** a few marked symbols among fillers; output them
  in order.
- **Associative recall:** key–value pairs, then every key again as a
  query; answer its value.
- **Text:** WikiText-103 bytes in windows of 128; four blocks of width 256
  (wider for the controls matched in parameters), and only the mixing
  layer changes.

Each task needs one mechanism and nothing else; the text checks that a
layer still learns language.
:::

::: {.slide title="Each Section Tests Its Mechanism on One of These"}
[Running examples]{.kicker}

| section | example |
|---|---|
| Gated Recurrent Networks | delayed recall over 20 to 200 steps; gated stacks and attention on text |
| Sequence-to-Sequence Learning | reversing strings of 8 to 64 letters through one state |
| Linear Recurrences and Parallel Scans | GRU against a minimal GRU on text; GRU, scan and attention timed up to 16,384 tokens |
| State Space Models | delayed recall over 1,000 steps |
| Selective State Space Models | selective copying, with and without selectivity; the selective model on text |
| Matrix-State Recurrences | random keys in a matrix; associative recall at three head sizes |
| The Delta Rule | a key written twice; recall after keys are rebound |
| Hybrid Architectures | associative recall of 4 to 64 pairs; the cache of a 32-layer model |
:::
