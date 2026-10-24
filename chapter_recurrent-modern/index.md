# State Space Models
:label:`chap_modern_rnn`

A language model that generates text one token at a time has to remember
what it has read. A transformer keeps everything: it stores a key and a value
for every previous token, so the memory it needs, and the work per new token,
grow with the length of the context. A recurrent network keeps a summary of
fixed size instead, its *state*, and updates it once per token. Generating
the millionth token then costs as much as generating the tenth.

A fixed-size summary raises two problems. Classical recurrent networks update
the state with a nonlinear function of the previous state, so training has to
process a sequence one step at a time and leaves most of a GPU idle. And a
state of fixed size must forget, which raises the question of what it should
keep and how much it can hold.

:numref:`sec_lstm` shows how gates let a network learn what to keep, and why
gated networks still train one step at a time. :numref:`sec_ssm` makes the
update linear in the state, which lets us compute the states at all
positions at once, and borrows from physics a principled way to choose how
fast the state forgets. :numref:`sec_mamba` lets each input decide how much
to write and how much to forget. :numref:`sec_matrix-state` turns the state
into a matrix that stores key–value pairs, which is linear attention, and
asks how many pairs it can hold. :numref:`sec_deltanet` lets a write correct
an old association instead of adding to it. The last section,
:numref:`sec_hybrids`, shows what no fixed-size state can do and how a few
attention layers repair it.

We assume you know how a recurrent network carries a hidden state, why its
gradients vanish over long spans, and how attention reads keys and values.
The chapter teaches algorithms rather than GPU kernels; fused kernels are the
subject of :numref:`chap_performance`.

```toc
:maxdepth: 2

lstm
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

State space models<br>
**memories of fixed size that train in parallel, forget the right things, store pairs, and correct them**
:::
:::

::: {.slide title="A Transformer's Memory Grows With the Context; a Recurrent One Does Not"}
- Attention keeps a key and a value for every token: at 128 K tokens, a
  grouped-query attention layer holds 512 MB per sequence.
- A recurrent layer keeps a state of fixed size: about 2 MB for a Mamba-2
  layer, at any length.
- The millionth token costs as much as the tenth.

::: {.d2l-note}
Two problems: train such a memory **in parallel**, and decide **what to
keep** in a fixed size.
:::
:::

::: {.slide title="Map of the Chapter"}
| question | answer |
|---|---|
| how to keep a value for long? | gates, a learned decay |
| how to train in parallel? | linear updates, scans; SSMs for timescales |
| how to keep only what matters? | step sizes chosen by the input |
| how to store pairs? | a matrix state: linear attention, in chunks |
| how to overwrite a pair? | the delta rule |
| what can no fixed state do? | exact recall; add a few attention layers |

**Running examples:** character-level Penn Treebank for every model;
delayed recall, selective copying, and associative recall for memory.
:::
