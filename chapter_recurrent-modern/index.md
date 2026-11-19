# State Space Models
:label:`chap_modern_rnn`

The recurrent networks of :numref:`chap_rnn` read a text the way we do,
one token after another, and carry what they have read in a summary of
fixed size, their *state*. Generating the millionth token therefore costs
as much as generating the tenth. The transformers of
:numref:`chap_transformers` keep every token instead. A chat assistant
that answers a question about a long report holds a key and a value for
every token of the report (:numref:`sec_kv-cache`), so its memory, and the
work for each word of the answer, grow with the length of the report. The recurrent way is cheap, but is it good enough? In
:numref:`sec_gpt` a plain recurrent network as large as the GPT fell
behind it after the first few bytes of context and never caught up. Can a
better recurrent network close that gap, and at what length does the
growing cost of attention begin to matter?

Two things stand in the way. Classical recurrent networks update the state
nonlinearly, so training processes a sequence one step at a time and
leaves most of a GPU idle. And a state of fixed size must forget, so it
has to choose what to keep. The answers come from three directions: gates
from the 1990s
:cite:`Hochreiter.Schmidhuber.1997,Gers.Schmidhuber.Cummins.2000`, linear
dynamical systems from physics and control theory
:cite:`Gu.Dao.Ermon.ea.2020,Gu.Goel.Re.2022`, and attention without the
softmax :cite:`Katharopoulos.Vyas.Pappas.ea.2020`. The three meet in
recurrent layers that train in parallel and compute a form of attention.
Every recurrent layer in the chapter has a state, an *update* that folds
in each token, and a *readout* that computes the output. Control engineers
call such a system a *state space model*. The chapter's title uses the
name in this wide sense, and :numref:`sec_ssm` in the narrow one, a linear
system in continuous time.

We begin with gates, which let a network learn what to keep, and with the
encoder–decoder of 2014, whose single state is a bottleneck we can
measure. Three changes then make the state fast and selective: an update
that is linear, so that it can be computed at every position at once and
timed against attention (:numref:`subsec_linear-recurrence-cost`); decays
read off a system in continuous time, which say how fast each channel
should forget; and a token that decides how much it writes and forgets, so
that the state can keep a name and let the commas pass. That last change is the hardest step of the chapter. The state then
turns out to be a memory of key–value pairs, whose capacity we can
calculate and whose write rule we can improve. Finally a little attention
restores the recall that no fixed state has, which is where practice has
arrived: several recent large models keep one attention layer in every
four to eight :cite:`Lieber.Lenz.Bata.ea.2024,Qwen.Team.2025`.

On a first reading, :numref:`sec_seq2seq`, the derivation from continuous
time in :numref:`sec_ssm` and the chunked algorithms of
:numref:`sec_matrix-state` and :numref:`sec_deltanet` can wait. Small
synthetic tasks, each built to need one mechanism, test the memories, and
the byte-level text of :numref:`sec_gpt` checks that a layer still learns
language. We assume linear attention
(:numref:`sec_windowed-linear-attention`). Our models are small, and how
these layers fare at scale is left to the papers we cite: the chapter
teaches mechanisms and algorithms, not a ranking of current models or
their GPU kernels.

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

## Resources and Further Reading {.unnumbered}

Start with Gu's essay on the tradeoffs of state space models and
transformers: in a few pages it argues that compressing a sequence and
looking up its tokens are different jobs, which is the point
:numref:`sec_hybrids` measures. Each entry names the section it goes with,
and all are freely accessible online. Topics that the chapter does not
teach are collected in :numref:`sec_ssm-further-reading`.

**Implementations**

- [The Annotated S4](https://srush.github.io/annotated-s4/), Sasha Rush and Sidd Karamcheti (2022): the full S4, of which :numref:`sec_ssm` teaches the diagonal case, implemented line by line beside the paper as runnable code, much like the sections of this book.
- [state-spaces/mamba](https://github.com/state-spaces/mamba), Albert Gu and Tri Dao: the authors' reference implementation, whose `ssd_minimal` listing reduces the chunked algorithm of Mamba-2 to a page (:numref:`sec_matrix-state`). [mamba-minimal](https://github.com/johnma2006/mamba-minimal) holds the architecture of :numref:`sec_mamba` in one readable PyTorch file, without fused kernels.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), fla-org: GPU kernels in Triton for gated linear attention, DeltaNet, Gated DeltaNet and their relatives, the production form of the layers of :numref:`sec_matrix-state` and :numref:`sec_deltanet`.
- [Gated DeltaNet for Linear Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet), Sebastian Raschka (2025): a bonus chapter of *Build a Large Language Model (From Scratch)* that implements a simplified Gated DeltaNet layer and explains the hybrids of Qwen3-Next and Kimi Linear, three linear layers to one attention layer; a build-along for :numref:`sec_deltanet` and :numref:`sec_hybrids`.
- [zoology](https://github.com/HazyResearch/zoology), HazyResearch (2023): the code of a study of recall in efficient language models, with the multi-query associative recall task that :numref:`sec_matrix-state` and :numref:`sec_hybrids` train on in small form.

**Explanations**

- [On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/), Albert Gu (2025): the essay recommended above, best read after :numref:`sec_hybrids`, whose recall experiment measures its main claim.
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state), Maarten Grootendorst (2024): more than fifty figures, from the basics of state space models to the selective scan; a gentle companion to :numref:`sec_ssm` and :numref:`sec_mamba`.
- [State Space Duality (Mamba-2), parts I to IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/), Albert Gu and Tri Dao (2024): the authors' account of why a gated recurrence and masked attention compute the same thing, the duality of :numref:`sec_matrix-state`, from the model to the algorithm and the systems.
- [DeltaNet Explained, parts I to III](https://sustcsonglin.github.io/blog/2024/deltanet-1/), Songlin Yang (2024), with her [Linear Attention and Beyond slides](https://github.com/sustcsonglin/linear-attention-and-beyond-slides): the delta rule, its training in chunks, and the design space of linear attention, from one of its main contributors; a second view of :numref:`sec_deltanet`.

**A survey and a lecture**

- [Test-Time Regression](https://arxiv.org/abs/2501.12352), Ke Alexander Wang, Jiaxin Shi and Emily B. Fox (2025): the framework behind the regression view of memory at the end of :numref:`sec_deltanet`, worked out for attention, linear attention and many recurrent layers.
- [CMU 10-423 Generative AI, Lecture 22: State Space Models](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/slides/lecture22-ssm.pdf), Matt Gormley and Pat Virtue (spring 2025): lecture slides on S4 and Mamba; a second treatment of :numref:`sec_ssm` and :numref:`sec_mamba`.

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
