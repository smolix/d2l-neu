# State Space Models
:label:`chap_modern_rnn`

The recurrent networks of :numref:`chap_rnn` read a text the way we do,
one token after another, and carry what they have read in a hidden state.
The transformers of :numref:`chap_transformers` keep every token instead.
When a chat assistant answers a question about a long report, it holds a
key and a value for every token of the report (:numref:`sec_kv-cache`), so
its memory, and the work for each word of the answer, grow with the length
of the report. Can a model read a long text with a memory of fixed size and
still model language well?

A recurrent network keeps a summary of fixed size, its *state*, and updates
it once per token, so generating the millionth token costs as much as
generating the tenth. A state of fixed size raises two problems. Classical
recurrent networks update the state nonlinearly, so training processes a
sequence one step at a time and leaves most of a GPU idle. And a state of fixed size must forget, which raises the question of what it should
keep and how much it can hold. The answers in this chapter come from three
directions: gates from the 1990s :cite:`Hochreiter.Schmidhuber.1997`, linear
dynamical systems from physics and control theory
:cite:`Gu.Dao.Ermon.ea.2020,Gu.Goel.Re.2022`, and attention without the
softmax :cite:`Katharopoulos.Vyas.Pappas.ea.2020`. The three meet in
recurrent layers that train in parallel and compute a form of attention.
Several recent large models interleave such layers with a few attention
layers :cite:`Lieber.Lenz.Bata.ea.2024,Qwen.Team.2025`.

:numref:`sec_lstm` asks how a network can learn what to keep.
:numref:`sec_ssm` asks how to compute the states at all positions at once,
and how fast a state should forget. :numref:`sec_mamba` asks how each input
can decide what to write and what to forget. :numref:`sec_matrix-state` asks
how a state can store key–value pairs, and how many it can hold.
:numref:`sec_deltanet` asks how a memory can correct a pair instead of
adding to it. :numref:`sec_hybrids` asks what no fixed-size state can do,
and how a few attention layers repair it.

We assume the recurrent networks of :numref:`chap_rnn` and the attention of
:numref:`chap_attention`; linear attention and its computation in chunks
(:numref:`sec_attention-at-scale`) are the starting point of
:numref:`sec_matrix-state`. The field changes quickly, so the chapter
teaches the mechanisms (state, update, readout, and cost) rather than a
ranking of current models. It teaches algorithms rather than GPU kernels,
which are the subject of :numref:`chap_performance`, and
:numref:`sec_ssm-further-reading` points to the topics it has no room for.

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

## Resources and Further Reading {.unnumbered}

The resources are grouped into implementations to build from, explanations
of the ideas, surveys, and course counterparts. All are freely accessible
online. Research topics that the chapter does not teach are collected in
:numref:`sec_ssm-further-reading`.

**Annotated implementations**

- [The Annotated S4 — Sasha Rush (2022)](https://srush.github.io/annotated-s4/) and [Mamba: The Hard Way (2024)](https://srush.github.io/annotated-mamba/hard.html) — S4 and the Mamba scan implemented line by line against the papers, in the executable-textbook format this book shares; companions to :numref:`sec_ssm` and :numref:`sec_mamba`.
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal) — the architecture of :numref:`sec_mamba` in one readable PyTorch file, without the fused kernels.
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — the authors' reference implementation, including the `ssd_minimal` listing that reduces Mamba-2's chunked algorithm (:numref:`sec_matrix-state`) to a page.
- [flash-linear-attention — fla-org](https://github.com/fla-org/flash-linear-attention) — Triton kernels for gated linear attention, DeltaNet, Gated DeltaNet, RWKV, and their relatives; the production form of the chunked layers of :numref:`sec_matrix-state` and :numref:`sec_deltanet`.
- [Gated DeltaNet, from scratch — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) — a bonus chapter of *Build a Large Language Model (From Scratch)* that implements the Gated DeltaNet layer of Qwen3-Next and Kimi Linear and their three-to-one hybrid ratio; a build-along for :numref:`sec_deltanet` and :numref:`sec_hybrids`.
- [zoology — HazyResearch](https://github.com/HazyResearch/zoology) — the synthetic associative-recall benchmarks behind the recall experiments of :numref:`sec_hybrids`.

**The ideas, explained**

- [A Visual Guide to Mamba and State Space Models — Maarten Grootendorst (2024)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — more than fifty figures from the basics of SSMs to the selective scan; a gentle introduction to :numref:`sec_ssm` and :numref:`sec_mamba`.
- [State Space Duality (Mamba-2), parts I–IV — Albert Gu and Tri Dao (2024)](https://goombalab.github.io/blog/2024/mamba2-part1-model/) — the authors' development of the duality between gated recurrences and masked attention of :numref:`sec_matrix-state`: model, theory, algorithm, and systems.
- [DeltaNet Explained, parts I–III — Songlin Yang (2024)](https://sustcsonglin.github.io/blog/2024/deltanet-1/) and her [Linear Attention and Beyond slides](https://github.com/sustcsonglin/linear-attention-and-beyond-slides) — the delta rule, its chunked training, and the design space of linear attention, from one of its main contributors; a second view of :numref:`sec_deltanet`.
- [On the Tradeoffs of SSMs and Transformers — Albert Gu (2025)](https://goombalab.github.io/blog/2025/tradeoffs/) — the argument that compressing a sequence and looking up its tokens are different jobs and that good models need both; the question :numref:`sec_hybrids` measures.
- [ASAP seminar series](https://asap-seminar.github.io/) — an ongoing online seminar on sequence-model architectures, where the topics of this chapter continue.

**Surveys**

- [Test-Time Regression — Wang, Shi, and Fox (2025)](https://arxiv.org/abs/2501.12352) — the framework behind the regression view of memory at the end of :numref:`sec_deltanet`, worked out for attention, linear attention, and many recurrent layers.
- [Speed Always Wins — Sun et al. (2025)](https://arxiv.org/abs/2508.09834) — a survey of linear sequence models, sparse attention, mixtures of experts, and hybrids, covering the efficient architectures around this chapter.

**Courses**

- [Stanford CS336: Language Modeling from Scratch, Lecture 4](https://cs336.stanford.edu/) — attention alternatives and mixtures of experts: this chapter's material as one lecture of a course on building language models.
- [CMU 10-423 Generative AI, Lecture 22: State Space Models](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/slides/lecture22-ssm.pdf) — lecture slides on S4 and Mamba by Gormley and Virtue; a second treatment of :numref:`sec_ssm` and :numref:`sec_mamba`.

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
