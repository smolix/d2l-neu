# State Space Models
:label:`chap_modern_rnn`

You now own two kinds of memory. The transformer of
:numref:`chap_transformers` keeps everything: its key–value cache is an
archive that grows with every token, answers exact-recall questions by
lookup, and presents a bill that :numref:`sec_kv-cache` measured directly
— a bill that in production sets the economics of long-context serving.
The recurrent networks of :numref:`chap_rnn` keep one thing: a fixed-size
state, updated in place, whose cost per token never changes and whose
memory of the past is whatever it managed to squeeze in. This chapter
asks the question that sits between those designs: how far can a fixed
state actually go? Its answer, which is the field's answer as of this
writing, comes as five verbs. *Gate* the state, so that learned sigmoids
decide what is written and what is erased. *Linearize* it, so that
training parallelizes across the sequence. *Select*, so that the dynamics
read the data as it flows past. *Edit*, so that a write can correct the
memory instead of piling on top of it. *Learn*, the reading under which
all of these turn out to be one algorithm fitting a regressor at test
time. And then the truce that production systems have settled on:
*hybridize*, keeping a few layers of genuine attention inside a mostly
recurrent stack.

The first three verbs are the chapter's classical spine.
:numref:`sec_lstm` builds the gate once, carefully: starting from the
observation that only an additive state passes gradients unattenuated, it
derives the LSTM's three gates and the GRU's two, trains both against the
vanilla RNN under one fixed recipe, and keeps the honest scoreboard — at
a ten-epoch budget the cheaper GRU wins while the LSTM merely matches the
baseline until its initialization is repaired, because architecture and
optimization cannot be judged separately. :numref:`sec_ssm` deletes the
nonlinearity from the state path: what remains of the GRU is an affine
recurrence that an associative scan evaluates in logarithmic depth, and
the state space view rebuilds the same object from continuous time, where
discretization *derives* the gate rather than positing it and the HiPPO
theory says what a state should be for its memory of the past to be
provably good. The section trains an S4D classifier on 784-pixel
image sequences, then runs the trained model both ways (all pixels at
once through the scan, one pixel at a time through a carried state,
verifying that the two agree) and prints the punchline: the model's entire
memory of an arbitrarily long history is a kilobyte-scale state, the same
at token one hundred as at token one hundred thousand, where a
transformer's cache would have grown a thousandfold. :numref:`sec_mamba`
restores what linearization gave up. A selective-copying task that no
time-invariant model can solve motivates making the step size a function
of the input (the forget gate derived a third time), and the resulting
Mamba block solves the copy task, tops the chapter's three-answers
scoreboard, and generates text at constant cost per token through its own
stepped path.

The next three sections carry the story to the present.
:numref:`sec_matrix-state` is where this chapter's road meets the one
from :numref:`chap_attention`: linear attention's matrix state and the
selective recurrence are one template that varies only in how it forgets.
The section measures the memory's capacity law: after $n$ unit-norm
writes into key width $d_k$ the squared read error is $(n-1)/d_k$, and
the measured curves sit on that prediction across three widths. It then
climbs the decay ladder from RetNet to Mamba-2 to GLA, and proves the
promised state-space duality — a gated linear recurrence and masked
attention are the same matrix computed in two contraction orders, with
the chunked third order being how the whole family trains at scale.
:numref:`sec_deltanet` changes the write rule. A memory that can only add
fails when a key must be re-bound: in the section's flagship experiment,
recall of the latest value roughly halves by two writes per key and
approaches chance by eight, a collapse that end-to-end training does not
escape, while the delta rule (read first, then write only the correction)
holds recall at 1.000 throughout and turns out to be one step of
gradient descent on a recall loss, running inside the forward pass. The
section makes it trainable with a triangular solve, gates it into the
Gated DeltaNet cell that several production models now ship, and shows
that the new transition genuinely computes: a single eigenvalue explains
why letting the write strength exceed one buys parity at any length.
:numref:`sec_test-time-regression` then supplies the theory the instances
have been hinting at: every memory in this chapter maintains itself by
solving a weighted regression of values on keys at test time. Softmax
attention is the Nadaraya–Watson estimator (closing a loop opened in
:numref:`sec_attention-pooling`, whose one learnable bandwidth the
section finally trains); linear attention is least squares with the key
covariance deleted; the delta rule is one explicit gradient step; and a
measured spectrum from a single online pass to the batch solve confirms
that more solving buys a better memory. Two models fall out of the view
rather than being designed: Longhorn, whose gate is the closed form of an
implicit update, and Titans, a memory that is itself a small network
trained inside the forward pass; a drifting-target experiment ends the
section with the statistical reason forgetting exists at all.

:numref:`sec_hybrids` closes the chapter where production begins. A fixed
state loses the exact-recall fight (the section quantifies what it
cannot copy, and why language-modeling loss hides the deficit), but only
attention layers pay a growing cache, so shipped systems interleave a few
of them into a mostly recurrent stack. The centerpiece experiment trains
three matched models, a pure recurrent stack, a pure attention stack, and
a hybrid with a single attention layer mid-stack, and watches that one
layer buy back the recall the recurrent stack loses while perplexity
barely moves; measured design rules for how much attention to keep and
where to put it, and a recipe table of shipped hybrids from Jamba to Kimi
Linear, turn the trade into engineering. One recipe threads all of these
experiments together: every trained language model in the chapter runs on
the *Time Machine* text of :numref:`sec_rnn-scratch` — the classical spine
and the Gated DeltaNet row on one shared scoreboard, the hybrid stacks on
their own matched panel — and the mechanistic experiments (capacity,
overwrite, the regression spectrum) run in seconds on a CPU.

The history here is a pendulum. The LSTM
:cite:`Hochreiter.Schmidhuber.1997` made recurrence trainable and carried
speech recognition and translation through the 2010s; the transformer
displaced it almost completely, and for a few years recurrence looked
finished. It returned through an unexpected door: S4
:cite:`Gu.Goel.Re.2022` arrived from continuous-time modeling rather than
the RNN lineage, Mamba :cite:`Gu.Dao.2023` made the dynamics selective
and competitive with transformers on language, and the state-space
duality of Mamba-2 :cite:`Dao.Gu.2024` collapsed the wall between the
returning recurrences and the linear attention that transformer
researchers had been simplifying toward. After that the lineages merged
outright: delta-rule cells and attention–recurrence hybrids now ship
inside production language models from many labs at once. Whether the
pendulum swings all the way back is a question with a date on it: a
public wager between Jonathan Frankle (yes) and Sasha Rush (no) resolves
on January 1, 2027, on whether a transformer-like model still holds the
state of the art in most benchmarked language tasks
([isattentionallyouneed.com](https://www.isattentionallyouneed.com/)).
This chapter takes no side; it teaches what each side is counting on.

A word on what this chapter is not. It teaches algorithms, not kernels:
the chunked forms here are twenty-line teaching implementations, and the
Triton kernels, memory hierarchies, and serving systems that make them
fast belong to :numref:`chap_performance`. It trains no large models:
pretraining recipes, data pipelines, and everything downstream of a base
model belong to the Language Models part. The efficient-attention
taxonomy stays in :numref:`chap_attention`, which already implemented the
surviving variants; applications of state space models to vision, audio,
and genomics are out of scope; and the fast-moving family of
test-time-training architectures beyond Titans is fenced off at a
pointer in the resources below. What remains is one adversary, met six
ways: the fixed-size state, and the measured question of how much
attention a model must keep when the state is not enough.

```toc
:maxdepth: 2

lstm
ssm
mamba
matrix-state
deltanet
test-time-regression
hybrids
```

## Resources and Further Reading {.unnumbered}

Grouped by the chapter's arc: implementations to build from, the
explanations behind the ideas, the papers that organize the field, and
course counterparts. All are freely accessible online.

**Annotated implementations**

- [The Annotated S4 — Sasha Rush (2022)](https://srush.github.io/annotated-s4/) and [Mamba: The Hard Way (2024)](https://srush.github.io/annotated-mamba/hard.html) — S4 and the Mamba scan implemented line by line against the papers, in the executable-textbook format this book shares; the closest companions to :numref:`sec_ssm` and :numref:`sec_mamba`.
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal) — the architecture of :numref:`sec_mamba` in one readable PyTorch file, deliberately without the kernels.
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — the authors' reference implementation, including the `ssd_minimal` listing that distills Mamba-2's chunked algorithm (:numref:`sec_matrix-state`) to a page.
- [flash-linear-attention — fla-org](https://github.com/fla-org/flash-linear-attention) — production Triton kernels for GLA, DeltaNet, RWKV, and their relatives; the industrial form of the chunked cells taught in :numref:`sec_matrix-state` and :numref:`sec_deltanet`.
- [Gated DeltaNet, from scratch — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) — a bonus chapter of *LLMs from Scratch* implementing Qwen3-Next's linear-attention layer, hybrid ratio and all; a build-along for :numref:`sec_deltanet` and :numref:`sec_hybrids`.
- [zoology — HazyResearch](https://github.com/HazyResearch/zoology) — the synthetic associative-recall harness behind the recall results that :numref:`sec_hybrids` reproduces at teaching scale.

**The ideas, explained**

- [A Visual Guide to Mamba and State Space Models — Maarten Grootendorst (2024)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — more than fifty custom figures from SSM basics to the selective scan; the gentlest on-ramp to :numref:`sec_ssm` and :numref:`sec_mamba`.
- [State Space Duality (Mamba-2), parts I–IV — Albert Gu and Tri Dao (2024)](https://goombalab.github.io/blog/2024/mamba2-part1-model/) — the authors' own four-part development of the duality that :numref:`sec_matrix-state` teaches: model, theory, algorithm, systems.
- [DeltaNet Explained, parts I–III — Songlin Yang (2024)](https://sustcsonglin.github.io/blog/2024/deltanet-1/) and her [Linear Attention and Beyond slides](https://github.com/sustcsonglin/linear-attention-and-beyond-slides) — the delta rule, the WY trick, and the whole linear-attention design space, from the researcher behind much of it; :numref:`sec_deltanet` in its original voice.
- [On the Tradeoffs of SSMs and Transformers — Albert Gu (2025)](https://goombalab.github.io/blog/2025/tradeoffs/) — the argument, adapted from a widely given talk, that compression and lookup are different jobs and the best models will do both; the thesis :numref:`sec_hybrids` prices out.
- [ASAP seminar series](https://asap-seminar.github.io/) — an ongoing virtual seminar on sequence-model architectures; where the topics of this chapter continue past its cutoff, including the post-Titans test-time-training line.

**Papers that organize the field**

- [Test-Time Regression — Wang, Shi, and Fox (2025)](https://arxiv.org/abs/2501.12352) — the unifying frame of :numref:`sec_test-time-regression` at full mathematical strength; the closest thing this chapter has to a companion paper.
- [Speed Always Wins — Sun et al. (2025)](https://arxiv.org/abs/2508.09834) — an eighty-page survey of linear sequence modeling, sparse attention, mixtures of experts, hybrids, and diffusion language models; the field-scale map for everything this chapter had to leave out.

**Course counterparts**

- [Stanford CS336: Language Modeling from Scratch, Lecture 4](https://cs336.stanford.edu/) — attention alternatives and mixture of experts: this chapter's material as one lecture of the from-scratch language-modeling course.
- [CMU 10-423 Generative AI, Lecture 22: State Space Models](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/slides/lecture22-ssm.pdf) — a careful lecture-notes treatment of S4 and Mamba (Gormley and Virtue); a second angle on :numref:`sec_ssm` and :numref:`sec_mamba`.
