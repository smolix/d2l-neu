# Chapter 12 Rewrite: Outline

Status: proposal for approval (2026-10-23). Governing documents:
`docs/writing-guide.md`, `docs/code-guide.md`, `docs/slide-guide.md`, and the
chapter notes in `reviews/writing-review-2026-10.md`,
`code-review-2026-10.md`, `figure-review-2026-10.md`, and
`slide-outlines-2026-10.md`. Section digests of the current text, with
stored results checked against the prose, are the basis for the decisions
below.

## 1. What the Chapter Teaches

A model that reads a sequence has to remember what it has read. A transformer
keeps a key and a value for every token, so its memory and its work per token
grow with the context. A recurrent model keeps a state of fixed size. The
chapter asks how to build a fixed-size memory that (a) trains in parallel on
a GPU, (b) forgets the right things, (c) stores as many associations as its
size allows, (d) can overwrite what it stored, and (e) where such a memory
must fail. Every model in the chapter is written in one form,

$$\text{state}_t = \text{transition}_t(\text{state}_{t-1}) + \text{write}_t, \qquad \text{output}_t = \text{read}_t(\text{state}_t),$$

and each section changes one part of it.

| § | Section | State | Transition | Write | Training | What changes |
|---|---|---|---|---|---|---|
| 1 | Gated recurrent networks | vector $\mathbf{h}$ | nonlinear, depends on $\mathbf{h}_{t-1}$ | gated | sequential | gates fix vanishing gradients |
| 2 | Linear recurrences and scans | vector $\mathbf{h}$ | diagonal $\mathbf{a}_t(x_t)$ | $\mathbf{b}_t(x_t)$ | parallel scan | gates read only the input |
| 3 | State space models | $d \times N$ | fixed $e^{\Delta A}$ | $\Delta B x_t$ | convolution or scan | many timescales, stable by construction |
| 4 | Selective state space models | $d \times N$ | $e^{\Delta_t A}$ | $\Delta_t B_t x_t$ | scan | dynamics depend on the input |
| 5 | Matrix-state recurrences | $d_v \times d_k$ | $a_t \mathbf{S}$ | $\mathbf{v}_t\mathbf{k}_t^\top$ | chunked | an associative memory; linear attention |
| 6 | The delta rule | $d_v \times d_k$ | $\mathbf{S}(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$ | $\beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | chunked | writes correct instead of add |
| 7 | Hybrid architectures | state + some KV caches | | | | a few attention layers restore recall |

## 2. Scope and Size

- PyTorch and JAX only (Advanced part).
- About 18,000–20,000 prose words, half the current 37,000. Seven sections and
  an optional Further Reading notebook. `test-time-regression.md` is folded
  into the delta-rule section; the linear-recurrence half of `ssm.md` becomes
  its own section.
- Labels referenced from other chapters are kept: `chap_modern_rnn`,
  `sec_lstm`, `sec_gru`, `sec_ssm`, `sec_mamba`, `subsec_parallel-scans`,
  `subsec_capstone` (`fast-transformer.md:1548` refers to `MambaLM` there).
  `d2l.GRU` keeps its signature (`seq2seq.md` uses it).
- Every claim in the text is checked against the stored output of both tabs
  before the section is done (writing guide §9).

## 3. Chapter Introduction (draft, 330 words)

> A language model that generates text one token at a time has to remember
> what it has read. A transformer keeps everything: it stores a key and a
> value for every previous token, so the memory it needs, and the work per
> new token, grow with the length of the context. A recurrent network keeps a
> summary of fixed size instead, its *state*, and updates it once per token.
> Generating the millionth token then costs as much as generating the tenth.
>
> A fixed-size summary raises two problems. Classical recurrent networks
> update the state with a nonlinear function of the previous state, so
> training has to process a sequence one step at a time and leaves most of a
> GPU idle. And a state of fixed size must forget, which raises the question
> of what it should keep and how much it can hold.
>
> :numref:`sec_lstm` shows how gates let a network learn what to keep, and
> why gated networks still train one step at a time. :numref:`sec_linear-recurrence`
> makes the update linear in the state, which lets us compute every position
> at once. :numref:`sec_ssm` borrows from physics a principled way to choose
> how fast the state forgets, and :numref:`sec_mamba` lets each input decide
> how much to write and how much to forget. :numref:`sec_matrix-state` turns
> the state into a matrix that stores key–value pairs, which is linear
> attention, and asks how many pairs it can hold. :numref:`sec_deltanet`
> lets a write correct an old association instead of adding to it. The last
> section, :numref:`sec_hybrids`, shows what no fixed-size state can do and
> how a few attention layers repair it.
>
> We assume you know how a recurrent network carries a hidden state, why its
> gradients vanish over long spans, and how attention reads keys and values.
> The chapter teaches algorithms rather than GPU kernels; fused kernels are
> the subject of :numref:`chap_performance`.

## 4. Sections

Word targets are prose only, excluding code and exercises.

### 12.1 Gated Recurrent Networks (`lstm.md`, `sec_lstm`; 2,500 words)

- **Idea.** A gate is a learned decay: multiplying the state by a number in
  $(0,1)$ chosen per unit and per step lets gradients survive long spans.
- **Opening problem.** A plain RNN's gradient over $k$ steps is a product of
  $k$ Jacobians (restated from ch. 8 in one equation), so it vanishes or
  explodes. Number: with a factor of 0.9 per step, 50 steps leave 0.005.
- **Smallest case.** The scalar accumulator $c_t = c_{t-1} + x_t$ (gradient
  exactly 1), then $c_t = f\,c_{t-1} + i\,x_t$: half-life $\ln 2/\ln(1/f)$.
- **Body.** `## Gates` (scalar case, sigmoid gate); `## Long Short-Term Memory`
  (cell, three gates, figure, scratch implementation, concise layer);
  `## Gated Recurrent Units` (`sec_gru`; convex blend, reset gate; `d2l.GRU`);
  `## Deep and Bidirectional Networks` (one subsection, short; causal-decoder
  warning kept).
- **Experiment.** RNN, LSTM, GRU on the shared language model task (§5
  below), first three rows of the chapter's results table.
- **Limit → next.** The gates read $\mathbf{h}_{t-1}$, so step $t$ waits for
  step $t-1$: training is sequential.
- **Cut.** Lineage paragraphs; "Gates in Other Architectures" survey and
  table; per-framework initialization narration; sample generation; the
  "Experimental scope" paragraph.

### 12.2 Linear Recurrences and Parallel Scans (new `linear-recurrence.md`, `sec_linear-recurrence`, `subsec_parallel-scans`; 2,500 words)

This is the sample section, written in full for review.

- **Idea.** If the gates read only the input, the update is linear in the
  state, and a linear recurrence can be computed for all positions at once.
- **Opening problem.** A GRU on a 4,096-token sequence makes 4,096 dependent
  steps; a GPU that could do them together waits.
- **Smallest case.** $h_t = a h_{t-1} + x_t$ with $a = 0.9$: the state is a
  geometrically weighted sum of the inputs; half-life about 7 steps; $a =
  0.999$ gives about 693 (writing guide §5.1).
- **Body.** `## Linear Recurrences` (scalar, then vector with per-unit decay,
  then input-dependent decay $a_t$, the minGRU as the GRU with its gates cut
  off from the state); `## Parallel Scans` (affine maps compose; the
  combine; a scan in $\log_2 T$ rounds; `linear_scan`; check against the
  loop; timing); `## A Linear Recurrent Language Model` (`MinGRU` on the
  shared task: it trains several times faster than the GRU and comes within a
  few points of its perplexity).
- **Limit → next.** The minGRU's state has one number per channel and its
  decay rate is learned from scratch; nothing tells it which timescales to
  cover.
- **Cut.** The GRU-equation derivation by citation; the "load-bearing"
  framing; per-tab prose.

### 12.3 State Space Models (`ssm.md`, `sec_ssm`; 2,500 words)

- **Idea.** Writing the recurrence as a discretized differential equation
  gives a stable decay by construction, a range of timescales from one
  initialization, and, when the dynamics are fixed, a second way to compute
  the model: a convolution.
- **Opening problem.** A recurrence learned from scratch has no reason to
  cover timescales from 10 to 1,000 steps, and a decay outside $(0,1)$
  diverges.
- **Smallest case.** $h'(t) = \lambda h(t) + x(t)$, $\lambda = -1$; zero-order
  hold gives $a = e^{\Delta\lambda}$: $\Delta = 0.01$ gives $a = 0.990$, $\Delta
  = 0.1$ gives $a = 0.905$. The step size acts as a gate.
- **Body.** `## Continuous-Time Dynamics` (the ODE, why continuous time);
  `## Discretization` (zero-order hold, the map from $\mathrm{Re}\,\lambda < 0$
  to $|a| < 1$ as a new figure, state expansion to $N$ states per channel);
  `## The Convolutional Form` (unrolling to a kernel; FFT training; three views
  figure); `## S4D` (diagonal SSM layer and block, initialization of $A$ and
  $\Delta$ in one paragraph with the HiPPO reason in one sentence; a long
  sequence task; one token at a time at inference, state bytes against a KV
  cache).
- **Experiment.** S4D on the shared language model task (table row) and on
  one long-sequence task. The current sequential Fashion-MNIST comparison
  reaches opposite conclusions in the two tabs (LSTM 0.818 vs S4D 0.813 in
  PyTorch; S4D five points ahead in JAX) and its clipping claim compares
  against an unclipped run that was never stored. Replace it with a task
  whose answer depends on a long span and on timescale coverage, run
  through the FFT path in both frameworks; keep it only if both tabs agree.
- **Limit → next.** The dynamics are the same at every position, so the
  weight of an input depends on its distance, not on what it is: "whether the
  token ten steps ago was the key fact of the paragraph or a comma, its
  influence is the same" (kept from `mamba.md:66`).
- **Cut / moved.** Control-theory interpretation, HiPPO derivation, NPLR/DPLR,
  S5, the four-rule discretization table, "four meanings of stable" box →
  Further Reading (HiPPO, control view) or exercises.

### 12.4 Selective State Space Models (`mamba.md`, `sec_mamba`; 3,000 words)

- **Idea.** Letting the input set the step size and the write and read
  directions makes the state keep what matters and skip what does not, and
  the scan still applies.
- **Opening problem.** Selective copying: four marked symbols scattered among
  256 filler tokens must be reproduced in order. The LTI S4D stalls (0.58 /
  0.63, chance 0.125); an LSTM solves it (0.98 / 0.99).
- **What changes.** Stated precisely as a difference from the minGRU and the
  S4D: Mamba combines the minGRU's input-dependent decay with the SSM's
  $N$-dimensional state per channel and timescale initialization, and adds
  input-dependent write and read vectors $B_t$, $C_t$. (The current text and
  its `three-answers` figure call the minGRU content-blind; it is not.)
- **Body.** `## The Selectivity Problem` (task, LTI failure, the minGRU as a
  control: run it; its result decides whether the section says selectivity
  alone suffices or state size also matters); `## Selective Dynamics`
  (equations, scan unchanged, convolution lost); `## The Mamba Block`
  (block, `subsec_capstone` language model row, stepping one token at a
  time); `## Selectivity Measured` (replaces the Δ boxplot, see below).
- **The Δ claim.** The stored boxplot averages $\Delta$ over 96 channels and
  shows filler $\Delta$ above symbol $\Delta$ in three of four panels, the
  opposite of the prose. Replace it with the contribution of each input
  position to the state at the query step, for the trained Mamba and the
  trained S4D on the same sequences: LTI contributions decay with distance;
  selective contributions concentrate on the marked symbols wherever they
  occur. Ship the figure only if both tabs show it; otherwise state what the
  model does and leave the mechanism to an exercise with interventions on
  $\Delta$.
- **Cut.** "Prosecution/witness" framing; simplification table; per-framework
  result narration; sampling cell; the `three-answers` figure (replaced by
  the chapter table in §1 of this outline).

### 12.5 Matrix-State Recurrences (`matrix-state.md`, `sec_matrix-state`; 2,800 words)

- **Idea.** A matrix-valued state stores key–value associations; linear
  attention with decay is exactly such a recurrence, and it can be computed
  as masked attention, as a scan, or in chunks that mix the two.
- **Opening problem.** A vector state of $d$ numbers stores one number per
  channel. To recall "the value that went with this key", the state must
  store pairs.
- **Smallest case.** Two orthogonal keys in $\mathbb{R}^2$; write two pairs,
  read both back exactly; write a third and see the interference.
- **Body.** `## An Associative Memory` (outer-product write, read by a query;
  restate ch. 10's linear attention in one equation; the interference
  proposition with its three-line proof, error $(n-1)/d_k$, and the stored
  experiment that sits on the prediction); `## Decay` (scalar $a_t$ =
  Mamba-2, diagonal $\boldsymbol{\alpha}_t$ = GLA, each in one sentence of
  what it changes); `## The Dual Form` (unroll: $(\mathbf{L}\circ
  \mathbf{Q}\mathbf{K}^\top)\mathbf{V}$, a gated recurrence is attention with
  a learned decaying mask; numerical check); `## Chunked Computation`
  (attention within chunks, state across; figure; timing with chunk sizes
  that compute finite values only).
- **Limit → next.** Every write adds: a key written twice returns the sum of
  both values.
- **Cut.** "Ladder/rung/crowding/ledger" vocabulary; seven-model family table
  and notation box; Mamba-3 digression; mLSTM and its stabilizer (Further
  Reading); the $a_t = 1$ check that is true by construction; the C = 1024
  timing row computed on NaNs.

### 12.6 The Delta Rule (`deltanet.md`, `sec_deltanet`; 3,000 words)

- **Idea.** Reading the memory before writing and storing only the error
  turns each write into one gradient step on a recall loss, so a key can be
  rebound to a new value.
- **Opening problem.** Write key $k$ with value 1, then with value 2: an
  additive memory returns 3 (more generally the sum). Hand-built experiment:
  recall of the latest value 1.00, 0.51, 0.30, 0.22 after 1, 2, 4, 8 writes;
  the delta rule stays at 1.00 (both tabs agree; trained models agree).
- **Body.** `## Additive Writes` (the failure, measured);
  `## Writes as Gradient Steps` (recall loss, its gradient is key times
  prediction error, the update, $\beta$ as write strength; the transition
  $\mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$ has eigenvalue $1-\beta$
  along $\mathbf{k}$: projection at $\beta = 1$, reflection at $\beta = 2$,
  with a new figure and the short hand-set parity demonstration);
  `## Gated DeltaNet` (decay and correction together; language model row;
  chunked training stated as the chunk idea of §12.5 plus one triangular
  solve, with the numerical check); `## Memory as Regression` (one table:
  attention = kernel regression over all pairs, linear attention = an
  uncentered cross-moment, decay = weighted samples, delta rule = one SGD
  step, the implicit step and why it cannot overshoot; one paragraph).
- **Limit → next.** However it writes, a state of fixed size holds a fixed
  number of bits.
- **Cut / moved.** RWKV-7, DeltaProduct and the formal-language survey, the
  trained parity experiment and seed narration, Titans, the
  Nadaraya–Watson bandwidth warm-up (187 lines repeating ch. 10), model tours
  → Further Reading or nowhere.

### 12.7 Hybrid Architectures (`hybrids.md`, `sec_hybrids`; 2,300 words)

- **Idea.** A fixed state holds a fixed number of bits, so tasks that need
  exact recall of many items need memory that grows; a few attention layers
  provide it at a fraction of a transformer's cache.
- **Opening problem.** Recall 64 key–value pairs after reading them once.
- **Body.** `## Limits of a Fixed-Size State` (the counting argument: $k$
  tokens from a vocabulary of $V$ need $k\log_2 V$ bits); `## Inference
  Memory` (per-layer cache with units: about 4 KB per token for a GQA layer
  against about 1.3 MB fixed for a Mamba-2 layer, crossing near 320 tokens;
  figure); `## Recall with One Attention Layer` (three matched stacks, recall
  sweep: one attention layer in four restores recall; the language model
  panel cannot see the difference). 
- **Fixes.** The "0.92–0.94" JAX dip is not in the store; "Measured Inference
  State" plots a formula (replace by the figure or a real byte count); no
  "cliff" language.
- **Cut.** Design-rules survey, recipe table of shipped hybrids, distillation
  summaries → Further Reading; the "pendulum question" callback.

### Further Reading (`further-reading.md`; 1,500 words, optional)

HiPPO and the control-theory view of SSMs; S4's NPLR form; mLSTM/xLSTM;
RWKV-7, DeltaProduct and negative eigenvalues; Longhorn and Titans as
different inner solvers; formal-language expressivity results; shipped
hybrids and where they place attention; distillation from transformers.
Each topic: the problem, the one thing it changes, a citation with a clause.

## 5. Shared Infrastructure

- **One language-model task,** defined once and restated where used: *The
  Time Machine* with the 1,024-token BPE vocabulary of ch. 8, 32-token
  windows, the same optimizer, schedule, and step budget for every model,
  models matched in parameter count, one `lm_perplexity` helper that
  computes the same quantity in both tabs (the current `val_ppl` does not).
  One results table grows through the chapter (§12.1 RNN/LSTM/GRU, §12.2
  minGRU, §12.3 S4D, §12.4 Mamba, §12.5 linear attention, §12.6 Gated
  DeltaNet, §12.7 attention and hybrid). Differences within run-to-run noise
  are reported as "about the same"; the text says plainly that this task does
  not separate the models and that the recall tasks do.
- **Recall tasks:** selective copying (§12.4), overwrite (§12.6), multi-query
  associative recall (§12.7), from one generator module.
- **Library (`#@save`, PyTorch and JAX):** `linear_scan(a, b, h0=None)`
  replacing three scan signatures; `MinGRU`; `S4D` with `discretize`, `kernel`
  (FFT path), and `step`; `SelectiveSSM`, `Mamba`; `GatedDeltaNet`;
  `SeqStack(mixer, num_layers, ...)`, one residual stack for every mixer, used
  by the hybrids; `lm_perplexity`. Each saved symbol has a consumer.
- **Notation table** at the start of the chapter: $T$, $d$, $N$, $\mathbf{x}_t$,
  $\mathbf{h}_t$, $\mathbf{S}_t$, $a_t$, $\Delta_t$, $A, B, C$, $\mathbf{q},
  \mathbf{k}, \mathbf{v}$, $\beta_t$, $\mathbf{L}$, chunk length $m$ (not $C$).
  One meaning per symbol.
- **Figures.** Keep: LSTM and GRU cells, scan tree (fix caption colors),
  three views, SSM block, selective copying, Mamba block, semiseparable mask,
  hybrid cache. New: discretization map (left half-plane to unit disk),
  effective contributions selective vs LTI, delta-rule geometry (eigenvalue
  along $\mathbf{k}$), matrix memory write/read. Drop: three-answers, decay
  ladder (replaced by one sentence per decay form), hybrid stacks, HiPPO
  reconstruction.
- **Slides.** Rebuilt after the text, two lectures as in
  `slide-outlines-2026-10.md`, running example the Mamba-2 state against a
  GQA cache.
- **Exercises.** 84 now. Keep the good ones that fit the new body (most are
  substantive), re-anchor references, drop those tied to cut material; about
  six to eight per section.

## 6. Decisions Needed

1. **Split `ssm.md`** into linear recurrences (§12.2) and state space models
   (§12.3). Recommended.
2. **Fold `test-time-regression.md` into §12.6** and delete the section.
   Recommended.
3. **Further Reading notebook** for the cut material. Recommended.
4. **One training recipe for all language models** (AdamW at a matched step
   budget) instead of the ch. 8 SGD recipe for the RNNs and Adam for the rest.
   Recommended: the current table mixes optimizers, so its orderings mean
   little, and they flip between tabs.
5. **Replace sequential Fashion-MNIST** in §12.3 by a long-span task, or keep
   it with the FFT path if both tabs then agree.

## 7. Order of Work

1. This outline and the sample (introduction and §12.2), for review.
2. Shared infrastructure: language-model task, `lm_perplexity`, `linear_scan`,
   `SeqStack`, recall-task generator.
3. Sections in order 12.1 → 12.7, each verified against both tabs' stored
   outputs, then Further Reading, slides, exercises.
4. Re-execute the chapter's notebooks (PyTorch and JAX), capture, render, and
   check every figure and claim before the pull request.
