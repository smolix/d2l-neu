# Attention & Transformers: Chapter Review and Rewrite Proposal

*2026-07-17. Inputs: file-by-file content review of
`chapter_attention-mechanisms-and-transformers/`, a repo-wiring map (labels,
`#@save` contract, neighboring-chapter coverage), an external survey of how the
material is taught in 2024–26 (Stanford CS336/CS224n/CS25, CMU 11-711/11-667,
Berkeley CS182/CS294-158, Princeton COS597, MIT 6.5940; Karpathy nanoGPT/
nanochat, Annotated Transformer, UvA notebooks, Raschka's from-scratch book +
architecture comparison, transformer-circuits, kipply/EleutherAI transformer
math), and an adversarial review pass (feasibility, ordering, build-contract
mechanics). Deliverable: a proposal for Alex to react to — no content has been
rewritten yet.*

## 1. Verdict on the current chapter

Chapter-wide salvage is roughly **50–55%**. The scaffolding is sound and parts
were recently modernized (Flax NNX tabs, chrF metric, warmup, autoregressive
`pos_offset`, fp16-safe masking notes), but the *arc* is 2017–2020 and
MT-centric: attention is motivated as an enhancement to RNN encoder–decoders,
the capstone artifact is a **post-LN / ReLU / sinusoidal encoder–decoder
translator**, no decoder-only LM is ever built, and the closing survey
(`large-pretraining-transformers.md`, zero code cells) is a BERT/GPT-3-era time
capsule. Verified absent from the whole chapter: RoPE, ALiBi/NoPE, RMSNorm,
pre-LN as default (only ViT has it), SwiGLU, GQA/MQA/MLA, FlashAttention,
KV cache as a taught concept, any trained decoder-only model, sparse/linear
attention beyond hand-waves. The relative-position rotation derivation in
`self-attention-and-positional-encoding.md` stops literally one step short of
RoPE. None of the chapter has received the Advanced-part PyTorch+JAX-only cut
(all files still carry TF, most carry MXNet).

Per-file salvage: queries-keys-values 75% · attention-scoring-functions 80% ·
vision-transformer 80% · multihead-attention 70% · self-attention-and-PE 65% ·
transformer 55% · index 40% · bahdanau-attention 35% · attention-pooling 30% ·
large-pretraining-transformers 20%.

Meanwhile every serious 2026 course (CS336 A1, CMU minLlama, CMU 11-667 HW2,
CS224n A3) has students build the **modern decoder — pre-norm RMSNorm + RoPE +
SwiGLU + GQA — as the default transformer**, with 2017 taught as history.

## 2. Proposed decomposition: two chapters, 13 notebooks

The part structure already reserves both slots: "Attention" and the
`chapter_transformers/` placeholder. Proposal:

- **Chapter "Attention"** (6 notebooks) — the *mechanism*, application-agnostic:
  lookup → scores → heads → positions → cost/variants → what attention
  computes. Needs no transformer; its two trained models are a single tiny
  shared attention-only LM (see A4).
- **Chapter "Transformers"** (7 notebooks) — the *architecture family*, built
  on the Princeton-seminar spine made literal: **one configurable
  block/GPT class (`norm ∈ {layer,rms}`, `act ∈ {gelu,swiglu}`,
  `pos ∈ {learned,rope}`, pluggable FFN factory)** established early; every
  subsequent section swaps one component and re-runs. The modern-recipe table
  in the closing notebook then has a live counterpart: every row is a
  constructor call.

This matches how CMU 11-667 teaches it (mechanism lecture vs "architecture
advancement" lecture) and how CS336/minLlama structure the build.

### Chapter A — Attention (6 notebooks + index)

**A0 `index.md`.** Rewritten: attention as the shared primitive of LLMs,
vision, audio, and science models; honest but brief history (Bahdanau →
Vaswani); chapter map. The "grab BERT and fine-tune" thesis goes.

**A1 `queries-keys-values.md` — Queries, Keys, and Values.** (salvage ~75% of
current file + compressed remnant of attention-pooling)
- Attention as differentiable soft lookup over a keyed database; softmax
  normalization.
- *Attention pooling with fixed kernels*: Nadaraya–Watson compressed from a
  full section to one demo — a normalized similarity kernel *is* attention
  with hand-picked weights, which motivates learning them. Carries the
  `sec_attention-pooling` label. The 4-panel kernel-drawing cell (house-rule
  violation) becomes a pre-generated figure or is dropped.
- Visualization with `d2l.show_heatmaps` + identity-matrix sanity check. (The
  `#@save` itself relocates to ch. 3 — see §6.)
- Keep the ∇_q attention = key-covariance exercise.

**A2 `attention-scoring.md` — Scoring and Masking.** (salvage ~80% of
attention-scoring-functions + the Bahdanau story compressed to a vignette)
- Dot-product attention; the variance argument for 1/√d **plus a new
  demonstration**: softmax saturation/gradient collapse vs dimension.
- Masking: padding + causal, dtype-safe −inf idiom, `masked_softmax` (#@save),
  batch matmul (`subsec_batch_dot` label preserved — word2vec-pretraining
  references it).
- *From alignment to attention* (history subsection): additive scoring and the
  Bahdanau soft-alignment story. The alignment figure must be house-generated
  (schematic in `gen_mdl_*` style), NOT the paper screenshot. **No RNN
  attention decoder is built; the 200-epoch GRU MT run is deleted.** This also
  severs the chapter's backwards dependency on `d2l.Seq2Seq`/`d2l.Encoder`/
  `d2l.Decoder` from ch. 13's seq2seq.md.

**A3 `multihead-attention.md` — Multi-Head and Cross-Attention.** (salvage
~70%; kept lean)
- Why one head is not enough: the CS224n construction (a single head must
  average where copying is needed; multi-head fixes it), verified numerically.
- `MultiHeadAttention` (#@save) with the reshape trick and same-FLOPs
  accounting across heads.
- Self- vs cross-attention: one function, two wirings; a toy demo where
  sequence A queries sequence B with an interpretable alignment.
- (The CNN/RNN/self-attention complexity table moves to A5, where it belongs.)

**A4 `positional-information.md` — Positional Information.** (upgrade; ~65%)
- Attention ignores order: permutation-equivariance proof + shuffle experiment.
- Absolute encodings: sinusoidal (kept, incl. binary-counting intuition) and
  learned embeddings — now framed as baselines.
- **RoPE as the headline**: continue the existing 2×2 rotation derivation to
  its conclusion, implement it, and check numerically that scores depend only
  on relative offset. Every open model uses it; CS336/CS224n/CMU all require
  implementing it.
- Extrapolation: ALiBi; NoPE (the causal mask leaks position); a
  train-short/test-long experiment; PI/YaRN as pointers. **Introduces the
  chapter's one trained model**: a tiny `#@save` attention-only char-LM
  (embeddings + stacked attention + tied head), trained on real text at
  context 128, evaluated at 512 — a small delta after ch. 8's RNN LMs, reused
  in A6. *Pilot the task before writing prose*: on synthetic copy tasks the
  canonical ordering (absolute blows up, RoPE degrades, ALiBi flat) can
  invert; real-text char-LM is the robust setting.

**A5 `attention-at-scale.md` — The Cost of Attention.** (new; the strongest
new notebook)
- Opens with the CNN/RNN/self-attention complexity + path-length table
  (moved from the old self-attention section — the best era-independent asset
  in the chapter).
- Counting FLOPs and memory for one attention **layer**; formula vs
  measurement (measure in PT; verify analytically in the JAX tab — XLA
  preallocation makes direct measurement awkward, and we say why). Model-level
  6ND accounting belongs to B7, not here.
- Exact attention without the n×n matrix: online softmax in ~20 lines
  (chunked), exactness check, memory curve; this *is* the FlashAttention idea;
  point at the fused SDPA backend for the measured version.
- Windowed/sparse attention: sliding window, receptive field grows with depth
  (Mistral/Longformer context; one sentence on trainable sparsity, NSA-style).
- **Linear attention is a recurrent network**: kernelize the score, show the
  parallel and recurrent forms agree to numerical precision — a linear RNN
  with matrix state. One shared log-log time/memory-vs-n figure over
  dense/windowed/linear anchors the notebook.
- **Treaty with ch. 13 (SSMs)**: A5 owns kernelization + the
  parallel≡recurrent equivalence (the state-space-duality bridge in
  miniature); mamba.md's Recurrent Frontier keeps hybrids, the
  RWKV/xLSTM/GLA convergence story, and the matrix-state picture — A5 does not
  re-tell them, and mamba.md's "deferred to the attention machinery of the
  next chapter" promise is rewritten to point backward at A5 **in the same
  landing**, not "when ch. 13 is next touched".
- One paragraph, not a section, on the Performer/Linformer/Reformer
  approximation zoo (absent from every 2024–26 syllabus).

**A6 `what-attention-computes.md` — What Attention Computes.** (new; the
chapter's differentiator — taught executably nowhere else)
- **Owns the residual-stream view** (B1 back-references it, never re-teaches):
  QK circuit (where to attend) vs OV circuit (what gets moved), in the
  **strictly attention-only** setting (no FFN, no LayerNorm — the Anthropic
  framework's actual setting, and what keeps this notebook inside the
  mechanism chapter).
- Induction-heads lab on the A4 model: repeated random token sequences; the
  prefix-match-then-copy head and the loss-bump phase change appear in
  minutes. **Claims are existence-not-location** (the bump's epoch is
  seed-dependent; the results-precision policy forbids quoting it).
- In-context learning as pattern completion; close with the honest
  "what do attention weights mean" caveat inherited from the old chapter.

### Chapter B — Transformers (7 notebooks + index)

**B0 `index.md`.** One block, many wirings; the chapter builds a modern GPT
and then explains every axis on which deployed 2026 models differ from it.

**B1 `transformer-block.md` — The Transformer Block.**
- Block = attention + position-wise FFN writing into the residual stream
  (back-ref A6).
- Normalization, evidence at initialization: stack N blocks, measure
  activation/gradient norms vs depth — post-LN grows/shrinks geometrically,
  pre-LN stays flat (deterministic, seconds-cheap, seed-noise-free; the
  training-divergence version of this claim is a one-line ablation in B2,
  where a trainable model legitimately exists). RMSNorm; QK-norm (OLMo-2
  post-norm re-litigation as a one-sentence aside).
- FFN: ReLU → GELU → SwiGLU matched-parameter sweep; the 8/3 width ratio.
- Assemble the **configurable block** (#@save; flags for norm/act/pos) —
  modern config (pre-norm RMSNorm + SwiGLU) is the default.

**B2 `gpt.md` — A GPT from Scratch.** (the centerpiece)
- From blocks to a language model: embeddings, causal blocks, tied head; the
  GPT class takes the block configuration (and a pluggable FFN factory, which
  B6 uses).
- Train the modern config on real text at char level, minutes on one GPU
  (tokenization/BPE is ch. 8 material — cross-reference); loss curves; honest
  cost anchors (what this run costs vs GPT-2-class training). The
  `norm_first=False` divergence ablation lives here.
- Generation: temperature/sampling, cross-ref ch. 8's decoding section.
- **Load real GPT-2 weights**: instantiate the *GPT-2 configuration* of the
  same class (LayerNorm + GELU + learned positions — this is exactly what the
  flags are for; the modern config cannot accept GPT-2 weights) and generate.
  Weights mirrored through `d2l.DATA_HUB` with a pinned hash (no live HF
  fetches in the capture pipeline); `tiktoken` for the tokenizer; mind the
  Conv1D transpose; JAX side loads via safetensors→numpy (no torch in that
  venv).

**B3 `kv-cache.md` — Generation and the KV Cache.**
- Recompute vs cache: implement the cache, measure tokens/sec before/after
  (JAX tab uses a fixed-size cache with index updates — static shapes taught
  as a feature, it is how real serving works; warm up before timing, quote
  rough ratios only).
- The memory bill: cache-bytes formula vs measurement; **prefill is
  compute-bound, decode is memory-bound** — taught properly here as a real
  subsection (comp-perf's hardware page predates the roofline framing; this
  chapter cannot outsource it).
- MQA/GQA: sweep group count from H to 1; quality-vs-cache curve.
- Cache compression: low-rank KV compression as the *idea* behind MLA (prose +
  small demo; a faithful decoupled-RoPE MLA is too much code for a
  subsection); sliding-window rolling buffer **with the attention-sinks
  caveat** (StreamingLLM; GPT-OSS ships sinks — teaching the window without
  the sink teaches a method that visibly breaks); speculative decoding as a
  closing pointer (deep home: the future comp-perf pass).
- Optional payoff cell if it pilots well: load SmolLM2-135M (GQA) into our
  class — the modern-checkpoint counterpart of B2's GPT-2 load.

**B4 `encoders-decoders.md` — Encoders, Decoders, and Cross-Attention.**
(absorbs the taxonomy from large-pretraining-transformers.md, the enc-dec
material of transformer.md, and the Perceiver material — cross-attention as
interface)
- Three wirings of one block (encoder-only / decoder-only / encoder–decoder),
  attention-pattern figures redrawn in house style.
- Encoder-only: bidirectional attention, a tiny masked-token demo; BERT
  pointer → ch. 17 (+ ModernBERT one-liner so the story doesn't end in 2019).
- Encoder–decoder: cross-attention in full on a small self-contained toy task
  (alignment interpretable by construction) — deliberately NOT the full MT
  pipeline, whose dataset/trainer classes live in ch. 13.
- **Cross-attention as interface**: a learned latent array attends into an
  arbitrary-length input — the Perceiver idea (O(MN) latent bottleneck), with
  one cost-curve cell and Perceiver IO as a paragraph; its living descendants
  (Flamingo resampler, Q-Former, VLM query tokens) as pointers.
- Which wiring when: T5/Whisper (enc-dec), BERT-descendants (enc-only),
  everything else (dec-only); DETR/Flamingo depth → Image-Models part.

**B5 `vision-transformer.md` — Vision Transformer.** (carry over; ~80%
salvage, the best existing file)
- Keep: patchify-as-convolution, pre-norm block, GELU, learnable positions,
  cls token, Fashion-MNIST training, honest ViT-loses-at-small-scale
  discussion. Drop the TF tab. Add: the learned position-embedding 2-D
  similarity grid; sharpen the same-parameter ViT-vs-CNN inductive-bias
  framing (the loss *is* the lesson).

**B6 `moe.md` — Mixture of Experts.**
- Conditional computation: parameters vs FLOPs decoupled.
- Routing and balancing triptych: no balancing (expert collapse) → auxiliary
  loss → aux-free bias, with expert-usage histograms.
- Fine-grained + shared experts (DeepSeek design); Mixtral/Qwen3 configs.
- Swap the FFN in our GPT for a small MoE (via B2's FFN factory) and measure.

**B7 `scaling-laws.md` — Scaling Laws and the Modern Recipe.** (the close;
retitled to avoid colliding with ch. 9's "Scaling Up" = muP/LR-transfer, which
it cross-references and explicitly does not overlap)
- Counting parameters and FLOPs: 6ND, 2 FLOPs/param/token; our model's
  numbers vs the profiler.
- A miniature scaling study: 4–5 model sizes on a fixed corpus, log-log L(N).
  **Embrace the bend**: points line up in the power-law regime and depart
  where small data saturates large models — that departure *is* the
  Chinchilla lesson. Fit nothing beyond "roughly a straight line", quote no
  exponent decimals (results-precision policy, structural not disclaimed).
  The chapter's heaviest notebook; pilot first.
- **The modern recipe table**: Llama-3 / Qwen3 / OLMo-2/3 / DeepSeek-V3 /
  Gemma-3 / Mistral on the axes attention (GQA/MLA/sliding ratio), norm
  (placement, QK-norm), positions (RoPE/NoPE), FFN, dense/MoE, dropout
  (none at scale). Punchline: convergent evolution — everything deployed in
  2026 is still recognizably the 2017 transformer; the deltas are stability
  and cache/capacity engineering. Every row is a constructor call to B1/B2's
  configurable class.
- Where the field is moving: linear-attention hybrids (→ ch. 13), long
  context, and a pointer to the Language-Models part for data/post-training.

## 3. What moves elsewhere, what dies

- **`large-pretraining-transformers.md` — retire the file.** The
  enc/dec/dec-only taxonomy (and its three genuinely useful figures) moves
  into B4 *with code*. The BERT/T5/GPT-3 tour, the paper-screenshot scaling
  PNGs (copyright exposure), and the "LLMs" survey are superseded; anything
  current-day (post-training, in-context learning, frontier landscape)
  belongs to the **Language Models part intro (ch. 17)** when that part is
  rewritten.
- **`bahdanau-attention.md` — retire as a section**; survives as the history
  vignette in A2. Deletes a 200-epoch GRU MT training run of a museum-piece
  architecture and the forward dependency on ch. 13's seq2seq classes.
- **`attention-pooling.md` — retire as a section**; survives as one demo + a
  figure in A1. (If the full Nadaraya–Watson treatment is worth preserving
  anywhere it would be the statistics appendix chapter, but the compressed
  aside likely suffices — the section calls itself "entirely optional" today.)
- **Not moving in**: tokenization/BPE (ch. 8 has it), optimizers/schedules/muP
  (ch. 9), kernels/parallelism/quantization/serving (Computational
  Performance's remit; B3 names speculative decoding and points there),
  SFT/RLHF/data curation (Language Models part), DETR/Flamingo/DiT depth
  (Image Models / Diffusion), Decision Transformer (RL). Noted for later:
  Computational Performance is CNN-era (hardware.md still quotes DDR4/RTX
  2080 Ti) and needs a transformer-age pass — out of scope here, but if it
  ends up downstream of these chapters its constants should be refreshed.

## 4. Ordering recommendation

Current Advanced order: 9 Optimization · 10 Attention · 11 Computational
Performance · 12 Transformers (placeholder) · 13 SSM · 14 RL · 15 GANs · 16
Diffusion. Splitting into slots 10 and 12 leaves Computational Performance
wedged between the two halves of one story.

**Recommend: 10 Attention · 11 Transformers · 12 State Space Models · 13
Computational Performance.** Attention→Transformers adjacent; the SSM chapter
then directly receives A5's linear-attention bridge and B3's cache-relief
handoff (and mamba.md's state-space-duality deferral rewrites to a *backward*
reference); Computational Performance — which needs its own transformer-age
modernization anyway — sits after both chapters whose workloads it should
discuss. Fallback if moving SSMs feels premature: the minimal 11↔12 swap
(… 10 Attention · 11 Transformers · 12 Comp-Perf · 13 SSM …). Either way the
change is mechanical: reorder `_quarto.yml` + `CHAPTER_NUMBERING` (dict order
= yml order invariant); labels renumber automatically.

## 5. Compute budget

All notebooks PT+JAX, single GPU, ≤7.5 GB, minutes-scale: the GPT build (B2)
and induction-heads lab (A6) are nanoGPT-scale; the GPT-2 loading cell is 124M
params (fine even on CPU); ViT is the existing 10-epoch Fashion-MNIST run. The
only heavy notebook is the scaling study (B7), bounded like ch. 9's
lr-scheduler (~10 min/framework). Nothing needs HEAVY_GPU/multi-GPU marking a
priori. **Four experiments get piloted before prose is written** (A4
extrapolation, A6 bump variability, B3 JAX timing, B7 scaling bend), with
claim templates fixed in advance per the results-precision policy.

## 6. Contracts and mechanics

The build scans **every** `chapter_*/*.md` by glob (`Makefile` `SRC_MDS`),
independent of `_quarto.yml`; same-name `#@save` collisions resolve by silent
last-writer-wins; docstrings embed `Defined in :numref:` so *moving* a
`#@save` block flags every consuming notebook lib-fingerprint-stale. Hence:

- **`show_heatmaps` relocates to ch. 3** (`chapter_linear-classification/
  softmax-regression-scratch.md`), its first consumer — Basics, where all four
  framework tabs legitimately live. A1 uses it via `d2l.`.
- **A "legacy attention lib" quarantine file** (e.g.
  `chapter_natural-language-processing-pretraining/legacy-attention-lib.md`,
  not listed in `_quarto.yml`, never executed — `make lib` reads source text)
  carries the `#@save` blocks BERT still needs, in **all four frameworks**:
  `TransformerEncoderBlock` (frozen 2017-faithful: post-LN, ReLU, original
  signature, inheriting `d2l.Module`/`nnx.Module` — NOT `d2l.Encoder`, whose
  home is ch. 13) **plus tf/mxnet variants** of `MultiHeadAttention`,
  `masked_softmax`, `DotProductAttention`, `PositionalEncoding` (the new
  chapters define PT/JAX versions only; without the quarantine, retiring the
  old chapter deletes the tf/mxnet symbols and breaks ch. 17's tf/mxnet
  notebooks). Name-collision rule: quarantine carries **no** PT/JAX symbol
  that the new chapters also define; the frozen block keeps its PT/JAX copies
  only because the new chapters deliberately do not reuse that class name
  (their block is the configurable `TransformerBlock`).
  **Rejected**: modernizing `TransformerEncoderBlock` now and re-capturing
  BERT later — between the two events the committed store would hold BERT
  outputs no longer reproducible from source. Freeze now; modernize
  atomically with the BERT re-capture in the Language-Models pass, which then
  deletes exactly one quarantine file.
- **`TransformerEncoder` (the full stack) is dropped from the library** — no
  external consumer (BERT builds its own encoder from the block), and keeping
  it would re-import the `d2l.Encoder` forward dependency.
- **Scheduled re-capture** (fingerprint staleness from relocated `#@save`
  blocks): ch. 3 softmax-regression-scratch (×4 fw, `show_heatmaps` docstring)
  and the NLP-pretraining/-applications trees (×4 fw, relocated attention
  symbols) — a GPU-box step of landing, listed in the implementation plan.
- **Labels**: **retire `chap_attention-and-transformers`** — its 12 external
  refs have heterogeneous intent (some mean the mechanism, some the
  architecture); each re-points explicitly to `chap_attention` or
  `chap_transformers` (one grep, twelve decisions). Other externally-
  referenced labels carry over: `sec_attention-scoring-functions` (math
  appendix ×4 → A2), `subsec_batch_dot` (→ A2), `sec_attention-pooling`
  (→ A1), `sec_multihead-attention` (→ A3),
  `sec_self-attention-and-positional-encoding` (→ A4),
  `sec_vision-transformer` (→ B5), `sec_transformer` (→ B4; bert.md's ref
  means "where the encoder block lives", which is B4),
  `sec_large-pretraining-transformers` (cnn-design, language-model ×2 → B4).
- **Directories**: new `chapter_attention/` + populate `chapter_transformers/`;
  retire `chapter_attention-mechanisms-and-transformers/` (sources, outputs
  store, slides) once the new chapters land — same retirement pattern as the
  legacy math appendix.
- **Frameworks**: PT+JAX only; old tf/mxnet output trees deleted with the old
  chapter (modulo the quarantine file, which has no outputs).
- **Figures**: all new/redrawn figures via the house generator
  (`tools/gen_mdl_*` style, byte-idempotent); paper-screenshot PNGs retired;
  the A2 Bahdanau alignment figure is house-generated, not the paper's.
  Slides: all ~13 decks rewritten as part of each notebook's authoring.

## 7. Deliberately not covered (over-stuffing guard)

The strongest courses run 2 mechanism lectures + 1 architecture lecture and
push depth into one from-scratch build. Accordingly: no efficient-attention
zoo beyond one paragraph, no Triton/CUDA kernel content, no full MT pipeline,
no BERT-era pretraining objectives (→ ch. 17), no RLHF/RAG/agents, no
long-context methods beyond A4's extrapolation experiment + PI/YaRN pointers,
no muP/optimizer content (ch. 9 owns it), no faithful MLA implementation
(idea only), no serving systems (B3 points at comp-perf's future pass).

## 8. Open questions for Alex

1. **Ordering**: approve 10 Attention · 11 Transformers · 12 SSM · 13
   Comp-Perf (recommended), or the minimal 11↔12 swap?
2. **A6 (induction heads)**: in as a full section (my call — it is the
   chapter's signature and no competing textbook has it executable), or
   demoted to optional?
3. **Perceiver placement**: folded into B4 as "cross-attention as interface"
   (recommended — full subsection with executable cost-curve cells; it
   appears in no 2024–26 syllabus as a standalone topic), or standalone
   notebook as originally sketched?
4. **BERT coupling**: confirm freeze-faithful `TransformerEncoderBlock` in a
   quarantine lib file now; modernize + re-capture BERT atomically in the
   Language-Models pass.
5. **B2's GPT-2 loading**: confirm mirroring the 124M checkpoint through
   `d2l.DATA_HUB` (~500 MB asset, pinned hash) is acceptable for the data
   pipeline.
