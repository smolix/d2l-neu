# Review + redesign proposal: ch. 12 "State Space Models" (2026-07-19)

*Proposal only; no chapter edits made. Commissioned by Alex 2026-07-19: rebuild
`chapter_recurrent-modern/` into a top-5 teaching chapter now that it sits
after Attention (ch. 10) and Transformers (ch. 11). Mandated coverage:
traditional variants (LSTM and friends, brief xLSTM), the main story "Mamba
and what follows" (SSM variants, DeltaNet + a DeltaProduct mention), one
notebook on test-time regression / forecasting (Longhorn, Titans, picking the
Nadaraya-Watson thread back up), and hybrid architectures. Grounded in six
research reports (current-state review incl. full dependency sweep; external
literature sweep; courses/pedagogy sweep; hybrids deep-dive incl. the MLSS
deck; implementation prototyping; legacy d2l NW mining). The core new
experiments — delta-vs-Hebbian overwrite (mechanistic + trained), chunked ==
sequential-loop parity for both the GLA/SSD form and the WY delta form, the
parity/eigenvalue demonstrations, the test-time-regression spectrum, and
micro-Titans — were **run on CPU and verified** before being proposed; §7
lists what remains unpiloted and gates it. Reports live in the session
scratchpad. A Fable critic pass reviewed this document against the reports
and the repo (verdict: adopt-with-changes); its corrections are incorporated
below.*

---

## 1. Verdict and the chapter at a glance

The existing SSM spine (12.3 `ssm.md`, 12.4 `mamba.md`) is strong and stays.
The chapter's problems are (a) it stops in 2023 — no matrix-state family, no
DeltaNet, no test-time-regression view, no hybrids, all of which are now the
production mainstream; (b) it still *argues as if attention comes next* (the
relocation happened on disk but not in the prose); (c) its central bargain —
constant-memory recurrent inference — is claimed but never demonstrated (no
trained model can step token-by-token; generation re-runs the prefix); and
(d) `seq2seq.md` no longer belongs in it.

**Proposed structure — 7 sections (3 revised, 4 new), seq2seq relocated:**

| § | File | Title (working) | Status | One-line story |
|---|------|-----------------|--------|----------------|
| 12.1 | `lstm.md` | Gated Recurrence | revise/slim | Gate it: LSTM, GRU, why gates outlived the cell; brief xLSTM |
| 12.2 | `ssm.md` | Linear Recurrence and State Space Models | revise | Linearize it: minGRU, scans, S4D, HiPPO — **+ step it token-by-token** |
| 12.3 | `mamba.md` | Selective State Space Models | revise/slim | Select it: Mamba; limits/frontier coda moves out |
| 12.4 | `matrix-state.md` | The Matrix State: Linear Attention to Mamba-2 | **new** | The two roads meet: decay ladder, SSD duality, chunked training |
| 12.5 | `deltanet.md` | DeltaNet: Memory That Edits | **new** | Delta vs Hebbian writes; chunked WY; the expressivity ladder |
| 12.6 | `test-time-regression.md` | Learning at Test Time | **new** | One recipe generates the family; NW → Longhorn → Titans; forecasting |
| 12.7 | `hybrids.md` | Hybrid Architectures | **new** | What a fixed state cannot do, and the production answer |
| — | `seq2seq.md` | Machine Translation and Encoder-Decoders | **moves out** | → Language Models part (recommendation; see §4) |

The chapter keeps its three-answers thesis (gate / linearize / select) and
extends it with the two post-Mamba answers: **edit** (the delta rule) and
**learn at test time** (regression view), closing with **hybridize** (the
production compromise). One adversary throughout, as before: the fixed-size
state.

Everything below is evidence and specification for this table.

---

## 2. What the relocation changed

### 2a. Inbound obligations the chapter must now honor (all grep-verified)

The preceding chapters make five explicit promises about this chapter:

1. `chapter_attention/index.md`: this chapter "will confront their limit
   directly: a fixed-size state must eventually discard something"; linear
   attention "turns attention back into a recurrence and **hands the story
   to** `chap_modern_rnn`"; this chapter "**owns the recurrence side** of the
   correspondence."
2. `chapter_attention/attention-at-scale.md` §"The Bridge to State Space
   Models": establishes, with taught + numerically verified code, the
   matrix-state recurrence `S_t = S_{t-1} + φ(k_t)v_tᵀ` (feature map
   `φ = elu+1`, parallel and recurrent forms proven equal), and states that
   Mamba-2's state-space duality "makes the correspondence exact." Its
   Exercise 6 pre-builds scalar decay `S_t = γS_{t-1} + φ(k_t)v_tᵀ`.
3. `chapter_transformers/kv-cache.md` cache-relief map: "…the **hybrid stacks
   of `chap_modern_rnn`** interleave a few full-attention layers into a
   mostly-linear model so that most layers pay no cache at all while a few
   retain exact recall." `chapter_transformers/index.md`: this chapter "picks
   up exactly where `sec_kv-cache`'s cache-relief map leaves off."
4. `chapter_transformers/scaling-laws.md`: hybrids "bet that exact global
   lookup is worth its price only a few times per stack; where that bet pays,
   and where it fails, **is that chapter's story**."
5. `chapter_attention/queries-keys-values.md`: Nadaraya-Watson regression is
   established as fixed-kernel attention pooling (keys = inputs, values =
   labels, query = eval point), with the **learned bandwidth explicitly
   demoted to an exercise** (the "use SGD to learn a good bandwidth…
   leave-one-out" prompt) — the open thread the test-time-regression section
   picks up.

The proposed 12.4 discharges (2), 12.7 discharges (3) and (4), 12.6
discharges (5), and the retained limits material in 12.7 discharges (1).

### 2b. Wrong-way references (must flip regardless of any other decision)

The chapter still calls attention "the next part" in `index.md:44-45` and
`seq2seq.md:1102-1108` (+ slide), and mamba.md's intro still frames a
hand-off to a successor; `mamba.md:758-760` says the comparison with
attention "must wait until we have built attention" (now false — it is
built); the final mamba slide says "attention, next chapter."
`mamba.md:149-154` glosses induction heads that
`sec_what-attention-computes` now teaches with running code. `lstm.md`'s
transformer-MLP/SwiGLU material can now cite `sec_transformer-block` (which
reproduces the SwiGLU sweep) as taught material. Full line
inventory in the current-state report §6. Conversely two already-flipped refs
in `mamba.md:1312-1316, 1351-1352` are correct and stay.

### 2c. Standing content gaps (relocation-independent, confirmed still true)

- **G1 — no recurrent-inference demonstration.** `S4D.forward`,
  `SelectiveSSM.forward`, `Mamba.forward` implement only the parallel path;
  `Mamba.forward` returns `(…, None)` for state, and the capstone's
  generation re-runs the **entire prefix per token**. The chapter's central
  "parallel training AND constant-memory inference" bargain is asserted, not
  shown. Fixed in 12.2/12.3 below; it also becomes the concrete backward link
  to ch. 11's KV-cache economics (the SSM's "cache" is a fixed `(H,N)`
  state).
- **G4 — S4D/S4DBlock restated verbatim** (~120 lines × tabs) in `mamba.md`.
  Fix by `#@save`-ing them in 12.2.
- Framework tabs exceed policy everywhere: lstm/seq2seq carry 4, ssm/mamba
  carry 3. The Advanced part is PyTorch + JAX; the rewrite drops
  tf/mxnet tabs from all ch. 12 files.

---

## 3. What is preserved untouched (the spine)

Confirmed by the current-state pass and left alone by this proposal: the
accumulator→gate derivation and the honest LSTM/GRU scoreboard (12.1); the
minGRU linearization-by-deletion story and `eq_affine_recurrence`; the
parallel-scan derivation, tree figure, and timing benchmark; the ZOH "step
size is a gate" box; the three-views (loop == scan == conv) numerical
verification; HiPPO at reconstruction-figure altitude; the S4D-vs-LSTM
initialization narrative (12.2); the selective-copy experiment, the
"gate derived a third time" thread, "what selectivity costs" (conv dies, scan
survives), the Mamba block, and the three-answers capstone with its honest
caveats (12.3). These are the chapter's crown jewels; the redesign builds
around them, not over them.

---

## 4. The structural call: `seq2seq.md` moves to the Language Models part

**Evidence (dependency sweep, grep-complete):**

- The entire MT apparatus — `MTFraEng`, `Seq2Seq`, `Seq2SeqEncoder`,
  `predict_step` (with its dead `save_attention_weights` hook), `chrf`,
  `bleu` — has **zero code consumers outside the file**. (`d2l.LSTM` and
  `d2l.GRU` are defined in `lstm.md`, not here, and stay: ssm.md/mamba.md use
  them as baselines; they do not pin seq2seq.) The old attention chapter that
  consumed the MT apparatus was replaced by ch. 11's transformer
  encoder-decoder, which reimplements everything on a synthetic task. The
  only genuine library reuse from the file is `show_list_len_pair_hist`
  (consumed by `word-embedding-dataset.md`), which survives as a `#@save`
  wherever the file lands.
- The encoder-decoder abstraction is now **taught twice**: seq2seq's RNN
  version and ch. 11 `encoders-decoders.md`'s three-wirings treatment with
  cross-attention, T5/Whisper, and Perceiver. The fixed-vector bottleneck
  that seq2seq builds toward has already been *used* — ch. 10's index and
  attention-scoring tell the 2014 bottleneck story as attention's own
  motivation.
- Inbound references are prose pointers only — six external files reference
  the file's labels or names: `attention-scoring.md` (ch. 10, forward ref),
  `text-sequence.md`, `word2vec-pretraining.md`,
  `natural-language-processing-pretraining/index.md`,
  `sentiment-analysis-and-dataset.md`, `mdl-statistics.md`, plus
  `init.md:394` which mentions the *function name* `init_seq2seq` in prose
  (not a label). All label pointers keep resolving wherever the file lives;
  the reference *direction* is the one thing placement changes (next
  paragraph).

**Why it should leave ch. 12 rather than stay reframed:** the chapter's story
is now a single ladder — gate → linearize → select → reconcile-with-attention
→ edit → learn-at-test-time → hybridize. A 1,300-line RNN machine-translation
build (dataset engineering, teacher forcing, chrF, beam search on real data)
interrupts that ladder at position two, teaches an abstraction ch. 11 already
owns, and motivates an escape (attention) the reader took two chapters ago.
It is good material — the masked-loss teaching, the measured bottleneck, and
chrF-over-BLEU modernization are worth keeping *somewhere* — but every
argument for its old position pointed forward to a chapter that now sits
behind the reader. Its pedagogical role in ch. 12 is over.

**Destination — recommendation: the OPENING of the Language Models part**,
i.e. ahead of the word2vec content — either as the first section of
`chapter_natural-language-processing-pretraining/` (ch. 17) or as a new
part-opening sequence-transduction file. The position matters: ch. 17
already back-references seq2seq (`word2vec-pretraining.md:77` "As described
in `sec_seq2seq`…", `…pretraining/index.md:10` cites
`sec_machine_translation`), so landing the file *after* them (e.g. in
ch. 18) would invert those into forward references. Reframe in one
paragraph: "the application that motivated both the encoder-decoder
abstraction and attention itself, built here with the recurrent tools of
`chap_modern_rnn` — the historical baseline against which everything in this
part is measured." All its references become backward references there. Its
4-framework tabs can survive unchanged for now (the Language Models part is
not yet under the PT+JAX policy), deferring tab surgery to the LM pass. The
bottleneck section stays with it, reworded from "the subject of the next
part" to "the reason attention won (`chap_attention`)."

**Alternative (if Alex prefers not to touch another part now):** park
seq2seq as ch. 12's final section, reframed as an application coda with a
"swap the GRU encoder for the chapter's Mamba stack" exercise. Costs: the
chapter ends on 2014 material after a hybrids capstone, and the ladder still
breaks — just at the end instead of the middle. Not recommended.

**`lstm.md` stays but slims.** LSTM + GRU are load-bearing (baselines used by
12.2/12.3, `sec_lstm` referenced from three upstream chapters, and the
"gate it" rung of the thesis). Deep RNNs and bidirectional RNNs have zero
external references; compress both into one "Depth and Direction" subsection
of roughly a third the current length — keep the 2-layer LSTM cell, the
bi-LSTM shape check, and the "an encoder, not a generator" lesson with its
ELMo→BERT pointer; drop the rest. The "Gates beyond Recurrent Networks"
table stays (it already has an xLSTM row) and gains the **brief xLSTM
mention** Alex asked for: sLSTM = exponential gating with a log-space
stabilizer (one paragraph, cite `Beck.ea.2024`), with the matrix-memory
mLSTM deferred one sentence forward to 12.4 where it lands as a family-table
row. The transformer-MLP and SwiGLU references flip to backward citations of
`sec_transformer-block`.

---

## 5. The new arc, section by section

Notation decision, binding for the whole chapter: adopt ch. 10's convention —
state `S ∈ R^{d_k×d_v}` with keys indexing rows, written
`S_t = D_t S_{t-1} + k_t v_tᵀ` (outer-product write), read `o_t = S_tᵀ q_t`.
DeltaNet's transition then acts from the left: `S_t = (I − β_t k_t k_tᵀ)
S_{t-1} + β_t k_t v_tᵀ`. The papers genuinely disagree on conventions
(Katharopoulos vs RetNet/GLA vs Yang's DeltaNet are transposes of one
another); the chapter picks one, states it once, and translates. A margin
note flags the transpose issue so readers can map to any paper.

### 12.1 `lstm.md` — revise (see §4 for the slimming spec)

Beyond slimming: flip the transformer references backward; anchor the
scoreboard prose unchanged; end the "Gates beyond" section with the hinge
sentence it already has (gates → linear recurrence → "the seed of minGRU…"),
which now carries the whole chapter.

### 12.2 `ssm.md` — revise

1. **Add `## Inference, One Token at a Time`** after `subsec_s4d` (fixes G1
   for the LTI half). LTI stepping is three lines from the discretized
   update; carry `(H, N)` state across calls, assert parity with the scan
   output on the trained sequential-Fashion-MNIST S4D, and time stepped
   generation against prefix-re-running as the prefix grows. This is also
   where the chapter cashes the ch. 11 backward reference: a transformer
   generating with a KV cache pays memory linear in length (measured in
   `sec_kv-cache`); the SSM pays a constant `(H,N)` — one figure, two
   curves. `#@save` a `step()` on `S4D`.
2. **`#@save` `S4D` and `S4DBlock`** (fixes G4; `mamba.md` then imports).
   Note: mamba.md's tensorflow copy of S4D is not byte-identical (it adds
   `@tf.recompute_grad`), but tf tabs are being dropped, so the dedup is
   clean for the two surviving tabs.
3. Prose flips: none needed for attention (the section already argues from
   CNNs — preserved), but the intro's "at inference we still update one token
   at a time… exactly like any RNN" claim now points at the new subsection
   instead of dangling.
4. One margin box near the ZOH/"step size is a gate" material: Gu's exact
   identity that **LSTM/GRU gating is the backward-Euler discretization of
   the same linear ODE** (thesis §2.4.2) — upgrading the chapter's
   "gate derived three times" motif from analogy to identity in three
   sentences.

### 12.3 `mamba.md` — revise and slim

1. Drop the verbatim S4D restatement (use `d2l.S4D`/`d2l.S4DBlock`).
2. **Add recurrent stepping for the selective model**: a `step()` for
   `SelectiveSSM`/`MambaBlock` carrying `(state, conv_buffer)` — the
   depthwise conv needs a width-4 rolling buffer, which is exactly what
   production Mamba inference caches (a point worth one paragraph). Verify
   stepped == scanned on the trained capstone model, then regenerate the
   capstone's text sample with the O(1) path (removing the O(T²)
   prefix-re-running contradiction).
3. Backward-cite induction heads (`sec_what-attention-computes`) at the
   associative-recall motivation; flip `:29-31` and `:758-760` — the
   transformer comparison is now **made** (one short paragraph + pointer),
   not deferred.
4. **Move out the two coda sections.** `## What a Fixed State Cannot Do`
   relocates to open 12.7 (it is the hybrids motivation, and its
   brains-vs-databases framing now composes with ch. 11's measured cache
   bill). `## The Recurrent Frontier` dissolves: its siblings-converged
   paragraph is superseded by 12.4/12.5 (which teach the siblings), its
   hybrids paragraph by 12.7, its SSD deferral by 12.4. mamba.md ends on
   "Selective Copying, Revisited" + a two-sentence handoff to 12.4 ("we
   built selectivity from the state-space side; the next section arrives at
   the same object from the attention side — and proves it is the same").
   The final slide's "attention, next chapter" is re-authored accordingly.

### 12.4 `matrix-state.md` — NEW. "The Matrix State: From Linear Attention to Mamba-2"

The reconciliation section: ch. 10 ended linear attention as a recurrence
with identity decay; ch. 12.2-12.3 built selective SSMs from continuous time.
This section proves the two roads meet, and en route teaches the decay
ladder and how these models actually train.

- `## Two Roads to One Recurrence` — restate `eq_linear-attn-recurrence`
  (backward ref). One paragraph must account for the state ch. 10 actually
  taught: the reader verified the **pair (S, z)** with feature map
  `φ = elu+1` and normalized read `o = φ(q)ᵀS / φ(q)ᵀz`; the modern family
  drops the explicit normalizer z (and mostly the feature map) in favor of
  output normalization — say so, and note that xLSTM's mLSTM is the one
  family member that keeps a normalizer state, a loop the family table
  closes. Then the capacity problem: retrieval error
  `Sᵀk_j = v_j + Σ_{i≠j}(k_i·k_j)v_i` (exact for unit-norm keys — state the
  assumption; L2-normalized keys are also DeltaNet's shipped default),
  capacity capped at d orthogonal keys ("the enemy of memory is not time;
  it's other memories" — cite Yang). **Make the capacity claim a measured
  law**: a cheap CPU sweep of recall accuracy vs number of stored pairs (and
  vs d) — the recall-capacity accounting no textbook currently does, and the
  quantity 12.7's hybrid trade is priced in. Forgetting as the first fix:
  scalar decay (RetNet; ch. 10's decay exercise redeemed) → input-dependent
  scalar (Mamba-2) → diagonal/vector gate (GLA, RWKV-6) — each one line
  added to the same recurrence, weighted-least-squares reading noted in a
  sentence (fully developed in 12.6).
- `## The State-Space Duality` — taught constructively per the Gu/Dao
  series: unroll the scalar-gated recurrence into the semiseparable matrix
  `M = L ∘ (QKᵀ)` with `L_{ij} = a_i···a_{j+1}`; the pivot observation "if
  all a_t = 1, L is literally the causal mask" makes masked linear attention
  and the SSM the same matrix; quadratic (attention) mode and linear
  (recurrent) mode are two contraction orders. Code: compute both, assert
  allclose (prototyped; the SSD-equivalence check passed). This *discharges
  the SSD promise* the old chapter deferred to a chapter that didn't exist.
- `## Chunked Computation: Mostly Matmul, a Little Scan` — the ~25-line
  segsum/chunked form (verified == sequential loop to ~1e-5 in prototype):
  diagonal blocks are attention-like matmuls, cross-chunk is a short state
  recurrence. One timing cell. This is how GLA/Mamba-2/DeltaNet train at
  scale, and why the state size jumped 16 → 64-256 for free (tensor cores).
  Prose altitude for Mamba-1-vs-Mamba-2 hardware trade-offs; brief Mamba-3
  pointer (trapezoidal discretization) as a frontier sentence.
- `## The Family, So Far` — first appearance of the master table (state,
  transition structure, write rule) for the Hebbian half: linear attention /
  RetNet / GLA / Mamba-1 / Mamba-2 / RWKV-6 / **mLSTM (the xLSTM
  matrix-memory cell, with its normalizer — the promised brief xLSTM
  treatment)**. The delta column is left conspicuously open with one
  sentence: every model in this table only ever *adds* to its memory; the
  next section is about the write rule that can *edit*. (License note: mLSTM
  is reimplemented from the paper's equations; the NX-AI repo is AGPL-style
  and must not be copied.)

Experiments: chunked == sequential loop was **prototyped and verified**
(~1e-5 GLA/SSD form, ~1e-6 WY delta); the quadratic-dual == recurrence
allclose is a small addition written at implementation time (the algorithm
is verified against the authors' reference `ssd_minimal.py`, but our cell
was not piloted — pilot before prose); chunked timing; the recall-capacity
sweep above; a state-size/cache accounting cell (ties to `sec_kv-cache`).
Training run: optional tiny GLA on the capstone LM task for one scoreboard
row (minutes, 1 GPU; unpiloted — pilot first); everything else is CPU-fine.

### 12.5 `deltanet.md` — NEW. "DeltaNet: Memory That Edits"

The centerpiece novelty. No course anywhere assigns building any of this
(verified across every syllabus the pedagogy sweep checked — nine named,
including CS336's five assignment handouts); the best treatments are blogs.
This section is the first executable textbook treatment.

- `## The Trouble with Adding` — flagship experiment (mechanistic, then
  trained; both prototyped): store key→value pairs where later writes
  **overwrite** earlier keys; query for the latest value. Hebbian recall
  collapses 1.000 → 0.502 → 0.285 → 0.194 as overwrites/key go 1→8; the
  delta rule holds 1.000 throughout. Trained 2-layer models reproduce it
  (Hebbian 0.53/0.34/0.26 at R=2/4/6; delta 1.000; 12-46 s per run on CPU).
  One design constraint that must be stated (discovered in prototyping): the
  write address (q,k) must be computed from the key alone, or a trained
  Hebbian model escapes superposition by giving each pair its own address.
- `## The Delta Rule` — derivation as error correction: retrieve, subtract,
  write the correction; `S_t = S_{t-1} − β_t k_t(k_tᵀS_{t-1} − v_tᵀ)`; then
  the one-line reveal that this is **one SGD step on ½‖Sᵀk_t − v_t‖²**
  while the Hebbian write is one SGD step on `−⟨Sᵀk_t, v_t⟩` (Widrow-Hoff
  1960 meets fast-weight programmers; cite Schlag/Irie/Schmidhuber 2021).
  The four-line recurrent implementation (error form). β as a learned
  write-strength gate.
- `## Training It: the WY Trick` — sequential updates couple through
  `k_i·k_j`, so a chunk's product of Householders is one triangular solve
  (~18-line chunked form, verified == loop to ~1e-6 in prototype). Altitude:
  teach the solve, cite the fla-org kernels for the production form.
- `## Gating and the Modern Cell` — Gated DeltaNet: `α_t(I − β_t k_tk_tᵀ)`,
  decay decoupled from the write step ("the AdamW of memory updates" — decay
  decoupled from learning rate); one scoreboard LM run (minutes, 1 GPU).
  One-paragraph siblings: RWKV-7 (generalized delta, diag + rank-1,
  Apache-licensed reference is 10 lines) and Kimi's KDA as shipped
  derivatives — this replaces mamba.md's faith-based "siblings converged"
  paragraph with taught machinery.
- `## What the Transition Can Compute` — the expressivity ladder, the best
  cheap experiment found in the whole sweep, **specified exactly as
  prototyped** (two constraints are load-bearing and must be pinned in the
  notebook, or the demo silently fails): (i) the trained contrast uses a
  *pure multiplicative* recurrence `h_t = a_t ⊙ h_{t-1}` with nonzero `h_0`
  and **no input drive** — an earlier prototype variant with a drive term
  and zero init put both eigenvalue ranges at chance; (ii) sequences stay
  short (train ≤16-24). Measured: `a ∈ (0,1)` sits at chance at every
  length (0.478-0.511) while `a ∈ (−1,1)` solves parity at T≤16 (1.000)
  and degrades into an optimization horizon by T=24 (0.831, 2 of 3 seeds) —
  quote at that stability, not as an unqualified solve. The Householder
  `(I − βkkᵀ)` demonstration is **mechanistic** (hand-set, not trained —
  the single-axis trained version is an optimization trap and the prose
  says so): eigenvalue `1−β` along k, so β=1 erases (parity lost, →0.506)
  and β=2 **reflects** (exact at every length tested, 1.000 at T=64) —
  grounding DeltaNet's `allow_neg_eigval` in one eigenvalue. TC⁰ ceiling
  (Merrill et al., ICML 2024) at citation altitude; **DeltaProduct as the
  mandated quick mention**: n_h Householders per token compose reflections
  → group word problems, one paragraph + one table row, no implementation.

### 12.6 `test-time-regression.md` — NEW. "Learning at Test Time" (one notebook, per the brief)

The unification, placed after the instances so it lands as a revelation
rather than an axiom. Also the section that redeems ch. 10's open exercise.

- `## Learning the Bandwidth` — pick the NW thread back up exactly where
  `sec_attention-pooling` left it: the legacy parametric NW cell (a single
  learnable bandwidth, leave-one-out construction, 5-epoch SGD loop, "the
  attention weights sharpen") is reinstated in modern form — the
  previously-dropped half of the classic notebook, and ch. 10's
  learned-bandwidth exercise answered in prose (refer to it by content, not
  number).
- `## One Recipe` — the Wang-Shi-Fox frame: memory = solve a weighted
  regression of values on keys at test time; retrieval = evaluate at the
  query. Three choices: weights γ, function class M, solver. Then the
  reveal table, each row now *already familiar*: NW/softmax attention =
  nonparametric kernel smoothing (softmax attention **is** the NW estimator
  — closing a loop opened in ch. 10); linear attention = least squares with
  `KᵀK ≈ I` dropped ("a crude associative memory that ignores key
  covariance"); decay = weighted LS; DeltaNet = one SGD step; multi-pass =
  batch GD. The spectrum figure (prototyped): NW 0.027 / online-GD 1-pass
  0.089 → 30-pass 0.026 / batch ridge 0.016 test-MSE on one dataset, with
  visually distinct fits.
- `## Deriving the Gate: Longhorn` — the proximal/implicit step
  `s_t = argmin ‖s−s_{t-1}‖² + β_t‖sᵀk_t − x_t‖²` has a closed form whose
  step size `Δ_t = β_t/(1+β_tk_tᵀk_t)` is *derived*, not designed — "Mamba's
  gate is a design choice; Longhorn derives it." Short cell verifying the
  closed form equals the argmin numerically (unpiloted — a few lines against
  a brute-force minimizer; pilot before prose).
- `## Deeper Memories: Titans` — the memory as a module trained at test
  time: gradient = momentary surprise, momentum = past surprise, weight
  decay = forgetting. Micro-Titans prototyped: linear case is delta +
  momentum + decay in ~20 lines (no autograd needed — and measurably a
  *softer* overwriter than pure delta: 1.00/0.92/0.20/0.11 recall at
  R=1/2/4/8, a teaching contrast, not a bug); MLP memory via
  `torch.func.grad`/JAX grad in ~40 lines, memorizes 16 associations in ~1 s
  on CPU. MAC/MAG/MAL variants at one-paragraph altitude.
- `## Regression That Tracks: the Forecasting Connection` — Alex's
  forecasting hook: streaming regression against a **drifting** target
  `w*`; undecayed accumulation goes stale, geometric decay tracks, the
  implicit step tracks with a derived rate. One figure. This states in
  miniature why test-time adaptation matters for nonstationary sequences —
  and why these memories forecast.
- Scope fence: this is deliberately ONE notebook. Post-Titans literature
  (Miras, Atlas, MesaNet, TTT-done-right…) is a Resources pointer, not
  content. Everything here runs on CPU in seconds-to-a-minute; per policy it
  should not require a GPU.

### 12.7 `hybrids.md` — NEW. "Hybrid Architectures" (capstone)

- `## What a Fixed State Cannot Do` — relocated and sharpened from mamba.md:
  the k·log₂V bit argument, the copying theorem (any fixed-state model's
  copy error ≥ 1 − |S|/D^L; a 410M transformer beats a 2.8B Mamba past ~70
  phone-book entries), MQAR as the diagnosis (82% of the pure-vs-attention
  perplexity gap is associative recall), and the production symptom: ablate
  the few full-attention layers of a shipped hybrid and needle-in-a-haystack
  drops to ~0 ("fuzzy recall" — the MLSS framing, near-verbatim).
- `## The Economics` — only attention layers pay a growing cache: the
  flat-vs-linear state figure (Mamba-2 ~4 MB/layer constant vs GQA KV
  growing to ~512 MB/layer at 128k — the MLSS plot, redrawn in house style);
  Jamba's measured 256K bill (4 GB vs Mixtral's 32 GB, 8×, because 4 of 32
  layers are attention). Backward refs to `sec_kv-cache` throughout; this is
  the cache-relief map's last rung, delivered.
- `## The Experiment: One Attention Layer Rescues Recall` — the section's
  centerpiece (harness 90% exists in mamba.md): three matched few-M-param
  models — pure selective-SSM stack, pure attention stack (ch. 11 blocks),
  and a 1-in-4 hybrid with the attention layer mid-stack — swept over recall
  difficulty (pairs × length) plus a short LM-loss panel. Expected shape
  (from Zoology/Based at small scale): recall collapses for the pure SSM as
  pairs exceed state capacity, the hybrid tracks pure attention, and **LM
  loss stays nearly flat across all three** — the cautionary dissociation
  (perplexity hides the recall deficit) is itself a headline teaching
  result. **This is the one unpiloted centerpiece in the proposal, and it is
  a hard gate: pilot it before any prose is written.** Known risk:
  small-scale MQAR results are hyperparameter-sensitive (Zoology swept
  learning rates per configuration); if the dissociation does not reproduce
  cleanly in single seeded runs, the fallback is the copy-length
  generalization axis (train ≤L, test >L: pure SSM cliffs, hybrid
  extrapolates — the cheaper theory-backed demo) plus the memory panel,
  which is arithmetic and always works. Second panel: measured state/cache
  memory vs generation length for all three. Honest-results policy applies:
  qualitative conclusions only, accuracy-vs-difficulty curves, single seeded
  runs.
- `## Design Rules, Measured` — the verified design axes: **ratio**
  (shipping sequential hybrids cluster at ~8-12.5% full attention, with
  Hunyuan-TurboS lower still at 5.5%; the gated-DeltaNet family sits at 25%;
  Kimi's ablation places the knee at 3:1; ratio studies show recall rising
  to ~1:3 while perplexity stays flat); **placement** (evenly dispersed,
  never the front layer — Samba reports a single front full-attention layer
  breaking length extrapolation, and the systematic study of arXiv
  2510.04800 finds middle placement optimal); **sequential vs parallel**
  (Hymba/Falcon-H1 fuse per-layer; Hymba's own controlled ablation has
  parallel beating sequential 45.19% vs 44.07%, the sharpest internal
  evidence, alongside the external Pareto study); **the parameter-sharing
  axis** (Zamba's one attention block reused 13×); and **the
  recurrence-attention interaction** (AI21 tested and rejected Mamba-2 for
  Jamba — "the Mamba-1-Attention combination works better," hypothesizing
  the larger state matters less once attention layers can pool the full
  context; and found no substantial difference between 1:3 and 1:7, so
  picked the cheaper — one paragraph: the components are not chosen
  independently). SWA counts as "a little exact attention" (Samba:
  local-only attention already recovers most recall and extrapolates to
  256K).
- `## The Recipe Table` — the chapter-closing analogue of ch. 11's
  recipe table (the pattern that worked): one row per shipped hybrid —
  Jamba, Samba, Zamba2, Nemotron-H, Granite 4.0-H, MiniMax-01, Qwen3-Next,
  Kimi Linear, Falcon-H1 — columns: total layers, attention fraction and
  positions, attention variant (full/SWA/GQA/MLA), recurrence
  (Mamba-1/2, gated DeltaNet, KDA, lightning), context. Every cell verified
  against config.json or paper in the hybrids report; MOHAWK is
  arXiv:2408.10189 (a frequently miscited ID).
- `## Distillation, and Where This Leaves Us` — one paragraph each:
  transformers distill into hybrids cheaply (MOHAWK: <1% of from-scratch
  tokens; Mamba-in-the-Llama: keep ~¼ of attention layers and recall
  survives — the retained attention layers are exactly what preserves it);
  closing synthesis returning to the chapter's adversary: the fixed state
  lost the exact-recall fight and won the economics, and production models
  stopped choosing. Hand-offs forward: kernels/systems to
  `chap_computational-performance`, pretrained stacks to the Language Models
  part.

### `index.md` — full rewrite (the "solid front discussion")

Model on the ch. 10/ch. 11 index essays: (1) the question — you have now
met both memories, the transformer's growing archive (ch. 10-11) and the
RNN's fixed state (ch. 7); this chapter is about how far the fixed state
goes, told as five verbs: gate, linearize, select, edit, learn — and the
truce, hybridize. (2) The section-by-section tour. (3) The history
paragraph (1997 LSTM → 2020-23 S4/Mamba → 2024-26 delta/hybrid
convergence); the public Frankle-Rush wager on whether transformers still
top the leaderboards on 2027-01-01 (isattentionallyouneed.com) makes a good
dated-bet hook here — framed as a bet with a date, not a live question the
book answers. (4) What this chapter is not: no kernels/systems
(ch. 13), no pretraining (Language Models part), the efficient-attention
taxonomy stays in ch. 10, applications of SSMs to vision/audio are fenced
off. Plus a Resources and Further Reading section (below).

**Exercises** (each new section ships exercises with teeth, per the ch. 10/11
bar — sketch, finalized at implementation): 12.4 — derive the chunked form's
FLOP count vs the quadratic dual and find the crossover chunk size; extend
the capacity sweep to decayed writes. 12.5 — implement the β∈[0,2] toggle in
the trained recall model and measure what changes (and what doesn't);
DeltaProduct's n_h=2 on the S3 word problem (guided). 12.6 — swap the
spectrum's solver for RLS/Woodbury and place it on the table; derive
Longhorn's Δ_t from the prox objective. 12.7 — re-run the sweep with the
attention layer first vs mid-stack (the Samba front-layer result, now
self-discovered); price a 1M-token context for each of the three models
using the ch. 11 cache formula.

---

## 6. Cross-cutting specifications

**Figures** (all generated, house style, via `tools/gen_mdl_*` + the
mdl-figure skill): the flat-vs-linear state/cache plot; the family-tree /
decay-ladder diagram (Yang-style: none → scalar → diagonal → diag+rank-1 →
products → dense, with the delta/Hebbian fork); the semiseparable-matrix
block decomposition (diagonal blocks = attention, off-diagonal = state
passing); a hybrid layer-stack schematic (ratio + placement). Slide decks:
all re-authored (68 existing slide divs across the four current files are
tied to cells that will move); new-figure copies must land in `img/auto/`.

**#@save contracts** (additions to the library): `S4D`, `S4DBlock`,
`S4D.step`, `MambaBlock.step` (12.2/12.3); `DeltaNetLayer` (or equivalent)
if 12.7's hybrid stack wants to reuse it — otherwise keep 12.5's classes
notebook-local and have 12.7 import the attention block from ch. 11's saved
classes plus the chapter's own Mamba block. Decide at implementation time;
default to fewer saves.

**Runtime budget:** every new experiment was CPU-prototyped in seconds to
~1 min; the trained scoreboard runs (GLA row, Gated DeltaNet row, the
three-model hybrid sweep) are minutes each on one GPU at existing capstone
scale. No HEAVY scheduler entries anticipated. Per Alex's policy, 12.6 and
the mechanistic cells of 12.5 should run on CPU (no meaningful GPU benefit).

**Citations to add to `d2l.bib`** (verified IDs): 2501.12352 (test-time
regression), 2407.14207 (Longhorn), 2501.00663 (Titans), 2102.11174
(fast-weight programmers), 2406.06484 (parallelizing DeltaNet), 2412.06464
(Gated DeltaNet), 2502.10297 (DeltaProduct), 2503.14456 (RWKV-7), 2405.21060
(Mamba-2/SSD), 2405.04517 (xLSTM), 2404.08819 (Illusion of State),
2411.12537 (negative eigenvalues), 2402.01032 (Repeat After Me), 2312.04927
(Zoology/MQAR), 2402.18668 (Based), 2406.07887 (NVIDIA empirical study),
2507.06457 (hybrid ratio analysis), 2510.04800 (hybrid systematic analysis),
2403.19887 (Jamba), 2406.07522 (Samba), 2504.03624 (Nemotron-H), 2501.08313
(MiniMax-01), 2510.26692 (Kimi Linear), 2411.13676 (Hymba), 2408.10189
(MOHAWK — note the commonly miscited 2408.10248), 2408.15237
(Mamba-in-the-Llama), 2508.09834 (the survey — Resources only).

**Resources and Further Reading** (index.md, all click-verified by the
sweeps): the Gu/Dao SSD blog series (parts I-IV); The Annotated S4 and
Mamba: The Hard Way (Rush); Songlin Yang's DeltaNet Explained I-III and her
"Linear Attention and Beyond" tutorial slides; Grootendorst's Visual Guide
to Mamba (visual on-ramp); Gu's "Tradeoffs of SSMs and Transformers" talk +
blog (databases vs brains); the ASAP seminar series; reference repos:
fla-org/flash-linear-attention, mamba-minimal, ssd_minimal, zoology; the
test-time-regression paper as the chapter's companion paper; the survey
(2508.09834) as the field map; Raschka's DeltaNet bonus chapter; CS336
Lecture 4 and CMU 10-423 Lecture 22 as the course counterparts.

**Scope fences** (what the chapter deliberately does NOT cover): sparse
attention and efficient-full-attention (ch. 10/11 territory); MoE (ch. 11);
kernels/Triton/serving (ch. 13); diffusion LMs; vision/audio/genomics SSM
applications; the post-Titans TTT zoo beyond a Resources pointer; LRA
benchmarking (historical citation only, as now).

---

## 7. Build/ops plan (implementation phase, for the record)

Per-file recapture (PT+JAX) after each section lands; the seq2seq move is a
CHAPTER_NUMBERING + `_quarto.yml` + outputs-tree move (its 4-framework
outputs move with it) with label preservation; reference-check afterward:
`mdl-statistics.md`, `text-sequence.md`, `word2vec-pretraining.md`,
`natural-language-processing-pretraining/index.md`,
`sentiment-analysis-and-dataset.md`, `attention-scoring.md` (label
pointers), and `init.md:394` (a prose mention of the `init_seq2seq`
function name — keep the function `#@save`d). One upstream fix rides along:
`chapter_attention/attention-at-scale.md:1196`'s closing slide hardcodes
"the state-space recurrence of ch. 13" — recurrent-modern is ch. 12; fix to
a label-driven reference per the no-hardcoded-numbers rule.

**Size and risk, named:** the current chapter is 5,828 lines / 68 slide
divs; at ch. 10/11 density the plan lands around 8,700-9,100 lines and ~95
slide divs — the largest chapter in the book — plus 4→2-tab surgery on
lstm.md and 3→2 on ssm/mamba. The pre-committed trim lever if scope must
shrink: fold 12.4 into 12.5 (§8.4), or thin 12.4's four jobs (it is the
densest new section). Slides re-authored per section; `make -B slides-<fw>
SLIDES_FILTER=...` per file; full render + PDF check at the end (watch the
`$`-digit and smallmatrix tripwires).

**Pilot gates (what was and wasn't prototyped):** prototyped and verified on
CPU — overwrite (mechanistic + trained), chunked==loop (both forms),
parity/eigenvalue (diagonal trained + Householder mechanistic), TTR
spectrum, micro-Titans (linear + MLP). **Unpiloted, pilot before prose:**
the 12.7 three-model recall sweep (hard gate; fallback specified in §5),
the GLA and Gated-DeltaNet LM scoreboard rows, the S4D/Mamba `step()`
parity cells, the SSD quadratic-dual allclose, the Longhorn closed-form
check. The pilots in `scratchpad/ssm-proto/` transfer as starting points.
Author agents get `docs/writing-avoid.md` and the results-precision policy;
every quoted number above regenerates at capture time and prose must round
to run-to-run stability. Implementation sequencing mirrors the ch. 10/11
build: pilot first, then write prose around measured outputs; two agents at
a time.

## 8. Open questions for Alex — RESOLVED 2026-07-19

**Decisions (Alex, 2026-07-19):** (1) seq2seq moves to the Language Models
part — "seriously out of date anyway and mostly for historical reasons";
reframe accordingly (historical baseline, not living material). (2) Title
stays "State Space Models". (3) bi-RNN: compress significantly. (4) Scope
approved — 7 sections; "this is a very important chapter." Build ops: max 2
agents at a time; pause at 85% usage and resume after reset.

*(original questions kept below for the record)*

1. **seq2seq destination**: Language Models part (recommended) vs ch. 12
   applications-coda vs defer the move to the LM modernization pass (keep it
   rendering in place, reframed minimally)?
2. **Chapter title**: keep "State Space Models" (set in the 2026-07-17
   restructure)? The content now spans delta rules/test-time
   regression/hybrids; "State Space Models" remains the field's umbrella
   usage, so the proposal keeps it — flagging in case "Recurrent Language
   Models" or similar is preferred.
3. **Depth/bi-RNN compression** to ~⅓ (proposed) vs full removal?
4. **7 sections** (4 new notebooks) is the proposed scope — comparable to
   the ch. 10/11 rebuild per-chapter. Trim candidate if needed: fold 12.4
   into 12.5 (denser, one fewer deck; costs the clean two-roads-meet
   structure).
