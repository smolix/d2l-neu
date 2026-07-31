# State Space Models
:label:`chap_modern_rnn`

Transformers retain a key--value pair for every token, so their cache grows
with the context. Recurrent models instead update a fixed-size state. This
reduces inference memory, but it limits how much information the model can
retain and retrieve. The chapter studies this tradeoff and the main
mechanisms used to improve fixed-state sequence models.

:numref:`sec_lstm` begins with multiplicative gates in the LSTM and GRU.
:numref:`sec_ssm` then makes the state update linear, which permits
parallel evaluation by an associative scan, and derives state space models
by discretizing continuous linear dynamics. :numref:`sec_mamba` makes
those dynamics input-dependent so that the model can select which tokens
affect the state.

The remaining sections compare matrix-valued state, editable memory,
online regression, and hybrid architectures. :numref:`sec_matrix-state`
connects linear attention to selective recurrence: both use a matrix
state, while their transitions determine how previous writes decay.
The section measures the memory's capacity law: after $n$ independent
random unit-norm writes into key width $d_k$ the expected squared read
error is $(n-1)/d_k$, and
the measured curves sit on that prediction across three widths. It then
compares the scalar and diagonal transitions used by RetNet, Mamba-2,
and GLA, and derives state-space duality — a gated linear recurrence and masked
attention are the same matrix computed in two contraction orders, with
the chunked third order being how these models train at scale.
:numref:`sec_deltanet` changes the write rule. A memory that can only add
fails when a key must be re-bound: in the section's flagship experiment,
recall of the latest value roughly halves by two writes per key and
approaches chance by eight. End-to-end training does not prevent this
failure in the deliberately restricted memory class tested there,
while the delta rule (read first, then write only the correction)
holds recall essentially perfect throughout and turns out to be one step of
gradient descent on a recall loss, running inside the forward pass. The
section makes it trainable with a triangular solve, gates it into the
Gated DeltaNet cell that several production models now ship, and shows
that the new transition genuinely computes: a single eigenvalue explains
why letting the write strength exceed one makes parity representable at
any length.
:numref:`sec_test-time-regression` interprets these updates as approximate
online regression of values on keys. Softmax
attention is the Nadaraya–Watson estimator (closing a loop opened in
:numref:`sec_attention-pooling`, whose one learnable bandwidth the
section finally trains); linear attention is least squares with the key
covariance deleted; the delta rule is one explicit gradient step; and a
measured spectrum from a single online pass to the batch solve confirms
how additional optimization steps affect test error. The same view
recovers Longhorn, whose gate is the closed form of an
implicit update, and Titans, a memory that is itself a small network
updated inside the forward pass. A drifting-target experiment then shows
why discounting old observations can reduce tracking error in a
nonstationary stream.

:numref:`sec_hybrids` compares fixed-state and attention layers. A fixed
state has limited exact-recall capacity, whereas an attention cache grows
with context length. Several deployed architectures therefore interleave
attention and recurrent layers. The section trains
three matched models, a pure recurrent stack, a pure attention stack, and
a hybrid with a single attention layer mid-stack, and watches that one
layer recover most or all of the recall lost by the recurrent stack
(roughly 0.92 to 1.00 across the sweep in our runs) while perplexity
barely moves. It then relates this tradeoff to engineering choices:
measured design rules for how much attention to keep and where to put it,
and a recipe table of shipped hybrids from Jamba to Kimi
Linear. One recipe threads all of these
experiments together: every trained language model in the chapter runs on
the *Time Machine* text of :numref:`sec_rnn-scratch` — the classical spine
and the Gated DeltaNet row on one shared scoreboard, the hybrid stacks on
their own matched panel — and the mechanistic experiments (capacity,
overwrite, the regression spectrum) run in seconds on a CPU.

The LSTM
:cite:`Hochreiter.Schmidhuber.1997` made recurrence trainable and carried
speech recognition and translation through the 2010s; the transformer
displaced it in many sequence applications. S4
:cite:`Gu.Goel.Re.2022` arrived from continuous-time modeling rather than
the RNN lineage, Mamba :cite:`Gu.Dao.2023` made the dynamics selective
and competitive with transformers on language, and the state-space
duality of Mamba-2 :cite:`Dao.Gu.2024` connected gated recurrences with
masked linear attention. Delta-rule cells and attention–recurrence
hybrids now combine elements of both lineages. Because these developments
change quickly, this chapter emphasizes state, update, readout, and
complexity rather than predicting which family will dominate.

A word on the name, and on what this chapter is not. We use *state space
models* the way the field now uses it: as the umbrella term for the whole
fixed-state family — gated RNNs, linear recurrences, selective SSMs,
matrix memories, test-time learners, and their hybrids — and not only for
the continuous-time linear systems from which :numref:`sec_ssm` takes the
term (that section also notes what the phrase means to a statistician,
which is different again). The chapter teaches algorithms, not kernels:
the chunked forms here are twenty-line teaching implementations, and the
Triton kernels and memory hierarchies that make them fast belong to
:numref:`chap_performance`. It trains no large models: the Language
Models part owns pretraining recipes, data pipelines, the serving stacks
that turn a trained model into a service, and everything downstream of a
base model. The efficient-attention
taxonomy stays in :numref:`chap_attention`, which already implemented the
surviving variants; applications of state space models to vision, audio,
and genomics are out of scope; and the fast-moving family of
test-time-training architectures beyond Titans is fenced off at a
pointer in the resources below. What remains is one adversary, met six
ways: the fixed-size state, and the measured question of how much
attention a model must keep when the state is not enough.

Two maps are worth carrying into the chapter. The first pins down its
most overloaded word. *State* names five related but distinct objects in
the sections ahead:

| What "state" means | Where | Typical shape | At autoregressive inference |
| :-- | :-- | :-- | :-- |
| RNN hidden vector $\mathbf{H}_t$ (plus the LSTM cell $\mathbf{C}_t$) | :numref:`sec_lstm` | $h$ numbers per layer | carried, updated in place |
| Continuous-time latent $\mathbf{x}(t)$ | :numref:`sec_ssm` | $N$ numbers per channel | analysis object; only its discretization runs |
| Discretized SSM state $\mathbf{x}_t$ | :numref:`sec_ssm`, :numref:`sec_mamba` | $(H, N)$ block per layer | carried, updated in place |
| Matrix fast weight $\mathbf{S}_t$ | :numref:`sec_matrix-state`, :numref:`sec_deltanet` | $d_k \times d_v$ per head | carried, updated in place |
| Inner-loop parameters of a memory network | :numref:`sec_test-time-regression` | a small MLP's weights | carried, updated by inner gradient steps |

All but the second are one idea at different granularities: the numbers a
fixed-memory model carries from token to token. The KV cache of
:numref:`sec_kv-cache` is the contrast class, per-token storage that
grows with the context; "state" in this chapter never means that.

The second map is for reading the experiments. Each probes one property,
and each has a confounder worth knowing before its conclusion arrives:

| Experiment | Probes | Main confounder |
| :-- | :-- | :-- |
| Sequential-image classification (:numref:`sec_ssm`) | long-range mixing (mean-pool readout) vs. state retention (final-step readout) | the readout decides which is measured; the LSTM baseline is initialization-sensitive |
| Selective copying (:numref:`sec_mamba`) | content-dependent selection | a deep network around LTI mixers earns partial credit without selectivity |
| Random-key capacity (:numref:`sec_matrix-state`) | additive-memory interference vs. key width | assumes independent isotropic keys; learned keys are neither |
| Overwrite task (:numref:`sec_deltanet`) | key re-binding: additive vs. delta writes | the trained baseline is a deliberately restricted memory class |
| Parity vs. length (:numref:`sec_deltanet`) | representability vs. trainability of sign-flipping transitions | optimization noise across seeds and lengths |
| Solver spectrum (:numref:`sec_test-time-regression`) | value of more inner-solver compute | the estimators optimize related, not identical, objectives |
| Hybrid recall sweep (:numref:`sec_hybrids`) | exact recall vs. attention budget | position handling and parameter matching |
| LM scoreboards (:numref:`sec_lstm`, :numref:`sec_mamba`, :numref:`sec_hybrids`) | end-to-end quality at teaching scale | one seeded run each; optimizer and parameter-count asymmetries |

When a section's conclusion reads stronger than its table, this map is
the antidote.

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
