# Chapter Overview — chapter_transformers

Seven sections, 41 existing exercises (matches the prior style review's count exactly).
Stanford CS336 ("Language Modeling from Scratch") is the dominant external source
by a wide margin: its `assignment1-basics` (tokenizer, RMSNorm/RoPE/SwiGLU
transformer, training loop, temperature/top-p decoding) maps almost exactly onto
transformer-block.md and gpt.md, and its `assignment3-scaling` (query a hosted
training API, build IsoFLOP profiles, predict a compute-optimal configuration at
a held-out FLOPs budget) is close enough to scaling-laws.md's own miniature study
that it earns high-overlap "adapted from" credit on two new problems there.
MIT 6.5940 (TinyML/EfficientML) supplies course-confirmed but detail-thin support
for kv-cache.md (KV-cache-adjacent quantization/deployment labs); CMU 11-667 has a
confirmed "Transformer from scratch" assignment but nothing scaling-, cache-, or
MoE-specific. "Understanding Deep Learning" (Prince) ch. 12 contributes two
concrete compute-accounting problems (12.8, 12.9) usable for vision-transformer.md
and otherwise treats this material expositorily rather than as posed problems.
The clearest negative finding: KV-cache internals (GQA/MLA/attention sinks) and
MoE routing/load-balancing have essentially **no** academic problem-set tradition
anywhere surveyed — both remain lecture-and-paper topics, which is itself notable
given this chapter builds working implementations of both from scratch. This
chapter's existing exercises are the strongest-reviewed material in the book
(clean, quantitatively precise, code-grounded per the prior style review);
disposition is accordingly keep-heavy: 39 of 41 kept, 2 rewritten (both in
vision-transformer.md, fixing a formatting defect and an underspecified clause),
0 dropped. 56 problems are proposed in total across the 7 sections.

---

## chapter_transformers/transformer-block.md — The Transformer Block

**Topic:** Where normalization sits in a transformer block (post-LN vs. pre-LN vs.
newer placements), RMSNorm and QK-norm, and the feed-forward network (classic MLP
vs. SwiGLU/gated variants at matched parameter count).

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review this is a clean, uniform, defect-free set in which
every item names an exact code change and an exact measurement; nothing here
needs fixing, so external material is additions only.

**External sources found:**
- Stanford CS336, `assignment1-basics` (2024/2025) — students implement RMSNorm,
  a SwiGLU-gated FFN, and a full pre-norm transformer block from scratch, graded
  by numerical unit tests (`adapters.py`) rather than free-form questions — the
  single closest external match to this section's own content — https://github.com/stanford-cs336/assignment1-basics
- CMU 11-667, "Large Language Models: Methods and Applications" (Fall 2024),
  Assignment #2 "Implement a Transformer from scratch" — a parallel build-and-test
  task covering the same block internals, confirmed at the assignment-topic level
  — https://2024.cmu-llms.org/assignments/
- Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture" —
  already the book's own citation for post-LN's warmup sensitivity; no course
  surveyed turns its warmup finding into a hands-on problem, which is a genuine
  gap our own addition below fills — https://arxiv.org/abs/2002.04745
- "Understanding Deep Learning" (Prince), ch. 12 — covers norm placement and
  gated activations in exposition, but its end-of-chapter problems (12.8, 12.9;
  see vision-transformer.md below) concentrate on attention/patch cost accounting,
  not block-internal design choices — no norm-placement or SwiGLU problem found
  here, an explicit coverage gap for this specific angle — https://udlbook.github.io/udlbook/

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Fit the pre-norm growth exponent.** At initialization the
   pre-norm stream RMS should grow like $\sqrt{1+2N}$ after $N$ blocks. Extend
   the signal-propagation experiment to fit the growth exponent from a log-log
   plot, and explain why the attention sublayer contributes less than the FFN
   sublayer early in the stack.
   *Provenance:* original (existing ex1).
1. [short-code] **A third normalization arrangement.** OLMo 2 normalizes after
   each sublayer but off the residual stream. Add this arrangement to
   `MiniBlock` and rerun the signal-propagation experiment; report which of
   post-LN's pathologies it avoids and what happens to the stream RMS.
   *Provenance:* original (existing ex2).
1. [short-code] **Initialization scale and collapse depth.** Rerun the
   signal-propagation experiment with each framework's default initialization
   and with all weights scaled by an extra factor of 0.5; report how the
   collapse depth changes and relate it to GPT-1-era trainability.
   *Provenance:* original (existing ex3).
1. [short-code] **Race the gated FFN variants.** Implement ReGLU and GEGLU by
   changing one line of `FeedForward`, race all four variants at matched
   parameters on `CharLM`, and compare the ranking against Shazeer (2020).
   *Provenance:* original (existing ex4).
1. [conceptual] **Exact parameter census.** Derive the exact parameter count of
   `TransformerBlock(d, h)` for both `act` settings and check it against the
   census cell; compute how far the Llama-7B configuration's rounded SwiGLU
   width lands from $8d^2$.
   *Provenance:* original (existing ex5).
1. [short-code] **Implement QK-norm.** Wrap `d2l.MultiHeadAttention` in an
   `attn_factory` that applies RMSNorm to queries and keys before the dot
   product, and measure attention-logit standard deviation with and without it
   as `num_hiddens` scales from 64 to 1024.
   *Provenance:* original (existing ex6).
1. [short-code] **Reproduce the warmup finding for post-LN.** Train the post-LN
   `CharLM` at the learning rate that failed outright in this section, but with
   a linear warmup over $W$ steps for $W \in \{0, 50, 200, 800\}$. Report the
   minimal $W$ at which training escapes the unigram-entropy plateau, and
   compare it against the gradient-attenuation gap this section already
   measured at initialization.
   *Provenance:* inspired by Xiong et al. 2020 (already cited in-text for this
   claim; overlap low — the paper motivates the question, but the specific
   warmup sweep and its connection to this section's own gradient measurement
   is original).
1. [short-code] **Gemma 3's arrangement, numerically.** Implement the fourth
   entry of :numref:`fig_norm-taxonomy` (normalize each branch on both sides)
   in `MiniBlock` and rerun the signal-propagation experiment alongside the
   other three arrangements already tested. Where does its stream-RMS growth
   and token-spread trajectory fall relative to pure pre-LN and pure post-LN?
   *Provenance:* original (completes the taxonomy figure the chapter already
   draws but only partially exercises).

---

## chapter_transformers/gpt.md — A GPT from Scratch

**Topic:** Assembling a decoder-only GPT (embeddings, causal masking, RoPE vs.
learned positions), training and breaking it, sampling/decoding, and loading
released GPT-2 weights into the from-scratch class.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review this is a clean, well-scoped set where every item
names an exact quantity to compute or measure; strong enough to keep in full.

**External sources found:**
- Stanford CS336, `assignment1-basics` — the closest external match in the whole
  chapter: BPE tokenizer, RoPE/learned-position transformer, training loop, and
  temperature/top-$p$ decoding, all graded against fixed test adapters —
  https://github.com/stanford-cs336/assignment1-basics
- CMU 11-667, Assignment #2 "Implement a Transformer from scratch" — same
  build-a-decoder-LM genre, confirmed at the assignment-topic level —
  https://2024.cmu-llms.org/assignments/
- Karpathy, nanoGPT / minGPT (github.com/karpathy/nanoGPT, github.com/karpathy/minGPT)
  — the most widely used from-scratch GPT reference implementation and informal
  teaching material (paired with the "Let's build GPT" video); it has no graded
  problem set or rubric of its own, so we note it as a practice resource rather
  than a citable exercise source.
- Radford et al. 2019 (GPT-2) and Devlin et al. 2018-adjacent weight-loading
  practice — no course surveyed turns "load a real released checkpoint into a
  from-scratch class" into a homework problem the way this section already does;
  this looks like a genuine gap that the section's own exercise 2 already fills.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Account for GPT-2's parameters.** Using the block census of
   :numref:`sec_transformer-block` plus the embeddings, account for every one
   of GPT-2's 124.4M parameters; report what fraction sits in the embeddings
   and why weight tying makes "124M" a slightly generous figure.
   *Provenance:* original (existing ex1).
1. [short-code] **Break the weight-loading with one flag.** Instantiate the
   GPT-2 configuration with `norm='rms'` and attempt to load the released
   weights; report which line fails and list every tensor/parameter left
   without a match under the modern `pos='rope', act='swiglu'` flags.
   *Provenance:* original (existing ex2).
1. [short-code] **Add nucleus sampling.** Implement top-$p$ sampling in
   `generate`, then compare GPT-2 continuations under top-$k=50$ vs. $p=0.9$
   for a confident prompt and an uncertain one.
   *Provenance:* original (existing ex3).
1. [conceptual] **Temperature limits and empirical entropy.** Show that
   $\tau \to 0$ recovers greedy decoding and $\tau \to \infty$ recovers the
   uniform distribution; measure empirical sample entropy at
   $\tau \in \{0.5,1,2,4\}$ and plot against both limits.
   *Provenance:* original (existing ex4).
1. [short-code] **RoPE extrapolation inside a full model.** Measure the trained
   char model's validation loss at contexts 128/256/512 and test whether naive
   RoPE extrapolation fails inside a full transformer as it did in isolation;
   apply position interpolation and report the change.
   *Provenance:* original (existing ex5).
1. [short-code] **Bits-per-character head-to-head.** Compare GPT-2 and the char
   model on the same held-out passage in bits per character, accounting for
   GPT-2's larger tokenizer and its 40 GB pretraining corpus.
   *Provenance:* original (existing ex6).
1. [short-code] **Shape-check before you download.** Before loading real GPT-2
   weights, write a test that derives the expected parameter *names and shapes*
   of the GPT-2-124M configuration by hand from the constructor flags, and
   checks them against `gpt2.state_dict()`/`nnx.state(gpt2, ...)` — catching a
   naming or shape mismatch before any checkpoint is downloaded.
   *Provenance:* inspired by CS336 `assignment1-basics`'s test-driven
   `adapters.py` pattern of checking implementations against fixed reference
   structure (overlap low — different target, same "test the seam before you
   trust it" method).
1. [extended] **Train a BPE tokenizer and compare against the char model.**
   Train (not just apply) a small byte-pair-encoding vocabulary on the Time
   Machine corpus at a chosen vocabulary size, using the BPE machinery of
   :numref:`sec_text-sequence`; retrain this section's `GPT` with the new
   tokenizer and compare validation bits-per-character against the
   character-level model on the same corpus. Does the text's claim that "BPE
   becomes advantageous on much larger corpora" hold at this corpus's size?
   *Provenance:* adapted from CS336 `assignment1-basics`, where training a BPE
   tokenizer is itself a separately graded component (overlap medium — same
   BPE-training task, applied to this section's own comparison rather than
   CS336's pipeline).

---

## chapter_transformers/kv-cache.md — Generation and the KV Cache

**Topic:** Why naive generation is quadratic, the KV cache and its memory cost,
prefill-vs-decode roofline analysis, and shrinking the cache via GQA/MQA, MLA,
and sliding windows with attention sinks.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review every item already states a measurable quantity
and, where relevant, an expected before/after comparison; nothing here needs
fixing.

**External sources found:**
- MIT 6.5940, "TinyML and Efficient Deep Learning Computing" (Fall 2024) —
  confirmed course structure: Lab 4 "LLM compression" and Lab 5 "LLM deployment
  on laptop" (deploying Llama-2-7B on a laptop), with lecture content explicitly
  covering KV-cache reduction via multi-query and grouped-query attention; exact
  lab problem text was not accessible, so this is a course-level, not
  problem-level, confirmation — https://hanlab.mit.edu/courses/2024-fall-65940
- Stanford CS336, `assignment2-systems` — students profile and benchmark a
  transformer LM and implement FlashAttention2 in Triton; the methodology
  (measure, change one thing, remeasure) transfers directly to this section's
  cache-timing exercises, though the assignment targets attention kernels and
  distributed training rather than the KV cache specifically —
  https://github.com/stanford-cs336/assignment2-systems
- Xiao et al. 2024 (StreamingLLM / attention sinks) — already the book's own
  primary citation for the sliding-window-plus-sink experiment; no course
  surveyed has turned it into a homework problem.
- **Explicit finding:** across every source searched (CS336, CMU 11-667, MIT
  6.5940, general web search), KV-cache internals — the memory formula, GQA/MQA,
  MLA's low-rank compression, attention sinks — have no dedicated academic
  problem-set tradition. This remains a systems-engineering topic taught via
  lecture and primary papers (Shazeer 2019, Ainslie et al. 2023, DeepSeek-AI
  2024, Xiao et al. 2024, all already cited in-text) rather than posed as
  exercises anywhere we found; this section's own hands-on cache-memory
  verification and cache-reduction experiments are unusually rare pedagogically.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Size the Llama-2-70B cache.** Work out its KV cache in 16-bit
   precision (80 layers, 64 query heads of dimension 128, GQA with 8 KV heads);
   find the context length at which one sequence's cache reaches the size of
   the fp16 weights, and redo both numbers without GQA and with MQA.
   *Provenance:* original (existing ex1).
1. [short-code] **Break the cache's RoPE offset on purpose.** Remove the
   `offset` from the cached RoPE path so decoded tokens rotate as if at
   position zero; measure how the resulting logit deviation grows with prefix
   length, and identify the analogous bug for `pos='learned'`.
   *Provenance:* original (existing ex2).
1. [short-code] **Batch the cached decode loop.** Extend `generate_cached` to a
   batch of equal-length prompts and measure tokens/second at batch sizes
   1/4/16/64; explain the curve's shape via which bytes are read once per step
   vs. once per sequence.
   *Provenance:* original (existing ex3).
1. [short-code] **Fix the wasteful cache-growth implementation.** Preallocate
   the PyTorch cache into a `max_len` buffer (instead of `torch.cat` every
   step), or pass `donate_argnums` for the JAX cache; measure per-step latency
   at context 4096 before and after.
   *Provenance:* original (existing ex4).
1. [short-code] **Give GQA its own cached step.** Add a `forward_step` to
   `GQAAttention` so the cache stores only $H_{kv}$ heads; verify correctness
   against the full forward pass and confirm cache memory shrinks by
   $H/H_{kv}$.
   *Provenance:* original (existing ex5).
1. [short-code] **Implement a true rolling buffer.** Cap the per-layer cache at
   4 sink entries plus the most recent $w-4$, evicting the rest as generation
   proceeds; decide and justify a position-index policy for retained entries
   under both `'rope'` and `'learned'`, and check loss against the mask-based
   experiment.
   *Provenance:* original (existing ex6).
1. [short-code] **Quantize the KV cache.** Store cached keys and values in an
    8-bit quantized format (per-tensor or per-channel scale) instead of fp32,
   dequantizing just before the attention matmul; measure the resulting cache
   memory reduction and any perplexity change on the same GPT-2 passage used
   in this section's rank-reduction experiment.
   *Provenance:* inspired by MIT 6.5940's LLM-compression/deployment labs,
   which cover quantization for efficient LLM inference (overlap low — we
   design the specific quantization scheme and evaluation ourselves; the labs'
   exact tasks were not accessible to us).
1. [conceptual] **Find the roofline crossover on two GPUs.** Using this
   section's arithmetic-intensity argument, derive symbolically the context
   length $n^*$ at which prefill and decode reach equal tokens/second, in
   terms of parameter count $N$, peak FLOP/s $F$, and bandwidth $B$. Evaluate
   $n^*$ for the RTX 4090 numbers already given in-text and for a GPU of your
   choice with published specs, and compare against this section's own
   measured crossover.
   *Provenance:* original (extends the section's single-GPU roofline argument
   to a second device — a natural gap since only one GPU's ridge point is
   discussed in-text).

---

## chapter_transformers/moe.md — Mixture of Experts

**Topic:** Conditional computation via top-$k$ token-choice routing, routing
collapse, and two load-balancing repairs (an auxiliary loss vs. a
gradient-free bias controller).

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review this set is clean and code-grounded; the informal
"triptych" label in ex3/ex6 is flagged as loose terminology but explicitly not a
defect (the referent is unambiguous), so nothing needs rewriting.

**External sources found:**
- Fedus et al. 2022 (Switch Transformer) and Lepikhin et al. 2021 (GShard) —
  already the book's primary citations for the balancing loss and capacity
  factor; no course surveyed has built a graded assignment directly implementing
  either mechanism.
- Stanford CS336 — has a lecture explicitly titled "Mixture of Experts" (Lecture
  4), confirming MoE is taught at the flagship course for this chapter, but we
  could not confirm a graded assignment (as opposed to lecture) built around
  implementing routing or load balancing — https://cs336.stanford.edu/
- CMU 10-423/623/723 "Generative AI" (Matt Gormley) — has a dedicated lecture
  slide deck, "Mixture of Experts" (Lecture 16), confirming the same
  lecture-only pattern at a second institution —
  http://www.cs.cmu.edu/~mgormley/courses/10423-f24/slides/lecture16-moe-ink.pdf
- **Explicit finding:** across every source searched, implementing MoE routing
  and load balancing appears as *lecture* content at multiple institutions
  (Stanford CS336, CMU 10-423) but we found no published, graded homework
  assignment that has students implement top-$k$ routing or a load-balancing
  mechanism from scratch. This section's own toy-scale, hands-on routing-collapse
  and balancing-method comparison is accordingly unusual as a teaching exercise
  rather than a lecture topic.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Derive the router's gradient, and Mixtral's escape.** Derive
   $\partial\mathcal{L}/\partial z_j$ for the router logits under raw-probability
   weighting; then show that under Mixtral-style renormalized gating the $k=1$
   router gradient vanishes identically, and explain why $k=2$ escapes this.
   *Provenance:* original (existing ex1).
1. [conceptual] **Capacity factor and dropped tokens.** Explain why production
   MoE systems need a fixed per-expert token buffer (capacity factor $c$),
   what happens to dropped tokens' representations, and why batch-level
   balance matters for hardware efficiency even under tolerable statistical
   imbalance; characterize $c=1$ and large $c$.
   *Provenance:* original (existing ex2).
1. [short-code] **Rerun the triptych at $k=2$.** Repeat the three-way balancing
   comparison with `num_active=2`; report whether the unbalanced run still
   collapses, and explain any difference via the feedback-loop argument.
   *Provenance:* original (existing ex3).
1. [short-code] **Sweep both balancing knobs.** Sweep the auxiliary-loss weight
   $\alpha$ and the bias update speed $u$ each over
   $\{0,10^{-3},10^{-2},10^{-1},1\}$; plot training loss and dead-expert count
   against each, and characterize the failure mode at large $u$ vs. large
   $\alpha$.
   *Provenance:* original (existing ex4).
1. [short-code] **Implement gathered (not masked) routing.** Route tokens to
   their assigned experts via index-and-gather in PyTorch instead of dense
   masking; verify it matches the dense layer's output and measure at what
   expert count the gathered version becomes faster.
   *Provenance:* original (existing ex5).
1. [short-code] **Add a shared expert.** Add one always-on `FeedForward`
   alongside the $k$ routed experts (DeepSeek-style); compare against the
   plain layer at matched active parameters and measure whether the shared
   expert changes how quickly routed experts specialize (usage entropy over
   training).
   *Provenance:* original (existing ex6).
1. [short-code] **Fine-grained experts at fixed budget.** Reconfigure the MoE
   GPT of this section to use 32 quarter-width experts with 4 active,
   matching stored and active parameter counts to the trained 8-experts/
   1-active configuration, and rerun the same 600-step comparison. Does finer
   granularity change the best validation loss or the usage-entropy
   trajectory under the bias controller?
   *Provenance:* original (the section's own :numref:`tab_moe-experts`
   discusses the granularity axis descriptively but runs no experiment varying
   it — a direct gap).
1. [short-code] **Combine both balancing methods.** Run the auxiliary loss
   ($\alpha=0.01$) and the bias controller ($u=0.01$) simultaneously on the
   triptych task, and compare final training loss and usage entropy against
   each method alone and against no balancing at all.
   *Provenance:* original (a natural fourth arm of the section's own three-run
   comparison, which frames the two repairs as alternatives but never tests
   them jointly).

---

## chapter_transformers/scaling-laws.md — Scaling Laws and Current Transformer Configurations

**Topic:** Counting non-embedding parameters and the $6ND$ FLOPs approximation,
a miniature multi-size scaling experiment, the Chinchilla compute-optimal law,
and a survey of 2023–2025 open-weight model configurations.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review, ex6 in particular is a genuine "predict twice,
then run" exercise with two explicit competing predictions and a stated
adjudication criterion; this is the strongest-reviewed set in the chapter and
needs no changes.

**External sources found:**
- **Stanford CS336, `assignment3-scaling`** (Spring 2024/2025) — the single best
  external match in the entire chapter. Students query a hosted "training API"
  with model hyperparameters (layers, embedding size, heads, batch size,
  learning rate) and a desired FLOPs budget; the API returns only the final
  training loss (no real training happens locally). Students build IsoFLOP
  profiles across several compute budgets, fit a scaling law, and must predict
  the loss-minimizing configuration for a real 1e19-FLOP budget, delivering a
  plot that extrapolates the fit to $10^{24}$ FLOPs — https://github.com/stanford-cs336/assignment3-scaling
- Kaplan et al. 2020 ("Scaling Laws for Neural Language Models") and Hoffmann et
  al. 2022 ("Training Compute-Optimal Large Language Models") — already the
  book's own primary citations for $L(N,D)$ and the compute-optimal allocation;
  no course surveyed assigns the analytical Lagrangian derivation
  ($\alpha A/N^\alpha=\beta B/D^\beta$) as a standalone problem — students are
  universally asked to fit/extrapolate empirically (as CS336 does) rather than
  derive analytically, which is itself a useful finding.
- Stanford CS336, Lecture 9 "Scaling laws basics" — the lecture underpinning
  `assignment3-scaling`; confirms the course treats scaling laws as its
  flagship empirical-methods topic — https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf
- CMU 11-667 — its four confirmed assignments cover data preparation,
  transformer-from-scratch, retrieval/tool-use, and bias/evaluation; no
  scaling-laws-specific assignment found — an explicit gap at this institution.
- **Explicit finding:** no external source surveyed poses "express reported
  model configurations as a shared table/constructor-call comparison" (this
  section's own exercise 3) — that framing appears to be original to this book.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Derive $6ND$ exactly for our configuration.** Write the
   per-block matmul parameter count at width 256, add the tied head, multiply
   out $6ND$ for one batch, and compare against the profiler's analytic line;
   find the context length at which the attention-score term equals the
   linear-layer work.
   *Provenance:* original (existing ex1).
1. [conceptual] **Run the Chinchilla check on our own sweep.** At twenty
   tokens per parameter, compute how many tokens each of the five sizes would
   want against the corpus's unique-character supply; relate which sizes
   exceed the ratio to where the measured curve bends, and argue (using
   Muennighoff et al. 2023) whether doubling the passes would rescue the
   largest model.
   *Provenance:* original (existing ex2).
1. [short-code] **Add a model family to the recipe table.** Read the
   architecture section of an open-weights report this section does not cover
   (e.g. Kimi K2) and fill in every column of :numref:`tab_modern-recipe`;
   identify which cells map onto `d2l.GPT` constructor flags, which need the
   factory seams, and which need machinery this chapter has not built.
   *Provenance:* original (existing ex3).
1. [conceptual] **When does the embedding dominate?** Derive the condition on
   $V$, $d$, $L$ under which embedding parameters reach at least half the
   total, check it against GPT-2's numbers, and explain why scaling-law fits
   improve when embeddings are excluded from $N$.
   *Provenance:* original (existing ex4).
1. [short-code] **Freeze the learning rate and watch the sweep lie.** Rerun the
   five-size sweep with the learning rate frozen at $10^{-3}$; report which
   points move and in which direction, and state what a naive reader would
   wrongly conclude from the resulting plot.
   *Provenance:* original (existing ex5).
1. [short-code] **Predict, then run, a sixth size.** Before running a width-512,
   ten-block, ~31M-parameter model, predict its validation loss two ways (line
   extrapolation vs. assume the corpus is saturated), estimate its FLOP cost,
   then run it and report which prediction was closer.
   *Provenance:* original (existing ex6).
1. [short-code] **An IsoFLOP sweep, not just a parameter sweep.** Re-run this
   section's miniature study as an IsoFLOP sweep: for two or three fixed
   compute budgets $C$, vary $(N,D)$ together so $6ND\approx C$ at each point,
   and find the loss-minimizing $N$ at each $C$. Does the resulting
   compute-optimal $N$ vs. $C$ relationship look power-law-consistent over
   your necessarily narrow range?
   *Provenance:* adapted from Stanford CS336 `assignment3-scaling`, whose
   central deliverable is exactly an IsoFLOP-profile-based fit used to predict
   a compute-optimal configuration at a held-out budget (overlap high — cite
   CS336 on adoption).
1. [extended] **Predict a held-out point from a partial fit.** Split the
   section's five sweep sizes into a training subset (the three smallest) and
   a held-out point (the largest); fit $L(N,D)=E+AN^{-\alpha}$ (dropping the
   data term, since $D$ is fixed) to the training subset only, predict the
   held-out model's validation loss, and compare against its measured value.
   Report your prediction error and discuss whether three points warrant the
   extrapolation.
   *Provenance:* adapted from CS336 `assignment3-scaling`'s core
   fit-small-predict-large methodology (overlap medium — we reuse the
   fit-then-predict protocol on this chapter's own five-point sweep rather
   than CS336's hosted training API).

---

## chapter_transformers/encoders-decoders.md — Encoders, Decoders, and Cross-Attention

**Topic:** The three transformer wirings (encoder-only, decoder-only,
encoder–decoder), the masked-language-modeling objective, verifying learned
cross-attention against a known alignment, and cross-attention with learned
(Perceiver-style) queries as a general interface.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0
— per the prior style review this file has no defects or clarity flags at all
and correctly uses `<mask>` as a code span throughout; a clean set to keep in
full.

**External sources found:**
- Devlin et al. 2018 (BERT) — already the book's own citation; this section's
  own exercise 1 already implements BERT's 80/10/10 masking rule directly, so
  the paper is both a text citation and an already-adopted exercise basis.
- Stanford CS224N, default final project "minBERT and Downstream Tasks" —
  students implement BERT's embeddings, self-attention, and transformer layers
  from scratch, then fine-tune/evaluate on sentiment analysis, paraphrase
  detection, and semantic textual similarity; confirmed via a public student
  project report referencing the assignment (exact year of report: unclear,
  project is a recurring default option) — https://web.stanford.edu/class/cs224n/
- Michigan EECS 498-007/598-005 (Justin Johnson), Assignment 5 — students
  implement scaled dot-product attention, multi-head attention, and full
  Transformer building blocks following "Attention Is All You Need," then
  visualize the learned attention weights on an image-captioning task —
  architecturally adjacent but the application (captioning) and visualization
  practice differ from this section's alignment-verification task —
  https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html
- Jaegle et al. 2021/2022 (Perceiver / Perceiver IO) and Carion et al. 2020
  (DETR) — already the book's own citations for learned-query cross-attention;
  **explicit finding:** no university course surveyed has a homework assignment
  implementing a Perceiver-style latent bottleneck or DETR-style learned object
  queries — this architectural family appears to be a "read the paper" topic
  everywhere we looked, making this section's own hands-on Perceiver
  implementation and cost-curve measurement unusually rare pedagogically.

**Proposed problem set** (8 problems, our reference format):
1. [short-code] **Implement BERT's 80/10/10 masking rule.** Replace the chosen
   position with `<mask>` 80% of the time, a random token 10%, and the
   original token 10%; compare masked accuracy and the loss on inputs
   containing no `<mask>` at all.
   *Provenance:* original (existing ex1).
1. [short-code] **Mask adjacent characters.** Mask two adjacent characters
   instead of one and evaluate the loss at both positions; explain the change
   using the per-position context analysis of this section.
   *Provenance:* original (existing ex2).
1. [short-code] **Widen the encoder–decoder and watch the map delocalize.**
   Rerun the alignment check and heatmaps at `num_blks=2`; report what happens
   to the argmax hit rate and attention mass on the true source position, and
   reconcile it with the chapter's warnings about reading attention maps.
   *Provenance:* original (existing ex3).
1. [short-code] **Change the task and predict the heatmap first.** Change
   `sample_batch` to a copy task, predict the heatmap, then verify; then try
   reverse-then-copy and predict the alignment in each half.
   *Provenance:* original (existing ex4).
1. [short-code] **Sweep the number of latents.** Sweep $M\in\{16,64,256\}$ in
   the cost-curve experiment, locate where the crossover with full
   self-attention moves, and derive the FLOP count of `PerceiverEncoder` as a
   function of $M$, $N$, $d$.
   *Provenance:* original (existing ex5).
1. [short-code] **Build a Perceiver IO output head.** Add a second learned
   query array of length $K$ that cross-attends into the latent summary to
   produce shape $(B,K,d)$; verify the shape and argue the total cost is
   $O(MN+M^2+KM)$.
   *Provenance:* original (existing ex6).
1. [short-code] **Probe the frozen encoder, minBERT-style.** Freeze the
   trained `TransformerEncoder` and train a small linear probe on top of its
   (unmasked) forward pass to predict a simple derived per-window label (e.g.,
   whether the window contains the letter "e" more than twice); compare probe
   accuracy using representations from an untrained vs. a trained encoder.
   *Provenance:* inspired by Stanford CS224N's minBERT default project, which
   evaluates a from-scratch BERT encoder via downstream-task probes rather
   than the masked-loss-by-position analysis already in this section (overlap
   low — different downstream task and probe design).
1. [short-code] **Look for off-diagonal head specialization.** Following the
   attention-visualization practice of Michigan EECS 498's Transformer
   assignment, plot the full cross-attention heatmap (not just argmax
   accuracy) for 3 example sequences from the widened (`num_blks=2`)
   encoder–decoder of problem 3 above; identify whether any of the 4 heads
   shows a systematic off-diagonal preference (e.g., attending one position
   ahead of the exact reverse) across all 3 examples, or only idiosyncratically
   per example.
   *Provenance:* inspired by Michigan EECS 498-007/598-005 Assignment 5's
   attention-weight visualization exercise (overlap low — different
   task/architecture, shared visualize-and-interpret methodology).

---

## chapter_transformers/vision-transformer.md — Vision Transformer

**Topic:** Patchify-as-strided-convolution, the learnable `<cls>` token and
position embeddings, and a matched-budget comparison between a ViT and a CNN
on Fashion-MNIST.

**Current exercises:** 5; disposition: keep 3, rewrite 2, drop 0
— per the prior style review this file is the chapter's one "legacy-style"
outlier (repeated-1 numbering, only 5 items): ex2 escapes the `<cls>` token as
an HTML entity ("&lt;cls&gt;") where the rest of the file's own slides use a
plain code span, and ex1's opening clause asks about `img_size`'s effect on
training time with no stated range before pivoting to a well-specified
sub-question — both are minor, mechanical fixes, not conceptual problems, so we
rewrite rather than drop.

**External sources found:**
- "Understanding Deep Learning" (Prince), ch. 12, **Problem 12.8** — poses
  exactly the quadratic-growth-in-patch-count question for ViT compute and asks
  for a mitigation using the book's own figure 12.15 — https://udlbook.github.io/udlbook/
- "Understanding Deep Learning" (Prince), ch. 12, **Problem 12.9** — asks
  students to compute the computation required to embed a 16×16-patch,
  512-dim grid and compare it against a DaViT-style transformer's cost —
  https://udlbook.github.io/udlbook/
- Michigan EECS 498-007/598-005, Assignment 5 — implements Transformer
  building blocks (not patchify/ViT specifically) in an image-captioning
  context; adjacent shared machinery (attention, multi-head projections) but
  not a ViT-vs-CNN comparison — https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html
- Touvron et al. 2021 (DeiT) and Liu et al. 2021 (Swin) — already the book's
  own citations for closing the ViT/CNN gap and reinstating locality,
  respectively; no course surveyed assigns implementing either as homework —
  this section's own exercise 3 (add augmentation, compare to DeiT) is already
  an unusually direct translation of DeiT's finding into a hands-on exercise.
- **Explicit finding:** ViT-specific homework is sparser than for text
  transformers across the sources surveyed — computer-vision courses mostly
  treat it as a lecture topic (an attention lecture) rather than a
  dedicated train-and-compare assignment; UDL's two chapter-12 problems are
  the most concrete external items found, and both are compute-accounting
  problems rather than train-and-compare exercises, which is exactly the gap
  this section's own ViT-vs-CNN and position-embedding-similarity experiments
  fill.

**Proposed problem set** (8 problems, our reference format):
1. [short-code] **`img_size` and `patch_size` vs. training time.** Predict what
   halving `patch_size` to 8 does to sequence length and to the cost of an
   attention layer, then measure time per epoch at `img_size` $\in\{64,96,128\}$
   with `patch_size` fixed at 16, and separately at `patch_size` $\in\{8,16\}$
   with `img_size` fixed at 96.
   *Provenance:* original (rewrite of existing ex1 — same task, now with an
   explicit range stated up front instead of pivoting mid-sentence).
1. [short-code] **Classify from pooled patches instead of `<cls>`.** Project
   the averaged patch representations to the output instead of the `<cls>`
   token's representation; implement this change and report the accuracy
   effect.
   *Provenance:* original (rewrite of existing ex2 — same task, `<cls>` now
   written as a code span throughout instead of an HTML entity).
1. [short-code] **Augment and retrain both models.** Add random horizontal
   flips and random crops to the training pipeline, retrain both the ViT and
   the CNN, and report whether the accuracy gap between them shrinks; relate
   the result to the DeiT recipe.
   *Provenance:* original (existing ex3).
1. [short-code] **Permute the patches at test time.** Apply a fixed random
   permutation to the 36 patch tokens after patch embedding (position
   embeddings left in place) and measure the trained ViT's accuracy; report
   what the size of the drop says about reliance on position embeddings.
   *Provenance:* original (existing ex4).
1. [short-code] **Interpolate position embeddings to a new resolution.**
   Reshape the 36 patch embeddings to $6\times6\times d$, resize spatially to
   a $9\times9$ grid, flatten back, and verify the model still classifies far
   better than chance at $144\times144$ input without retraining.
   *Provenance:* original (existing ex5).
1. [conceptual] **Where does ViT compute stop being linear?** Using this
   section's `PatchEmbedding`/`img_size` machinery, derive how doubling
   `img_size` at fixed `patch_size` changes (a) patch count $m$, (b) one
   attention layer's FLOPs, and (c) one FFN layer's FLOPs; report which term
   dominates first, and at what patch-grid size.
   *Provenance:* adapted from "Understanding Deep Learning" Problem 12.8
   (overlap medium — same quadratic-cost question, grounded in this section's
   own module and image size rather than the book's general figure 12.15
   sketch).
1. [short-code] **Is the patchify stem ever the bottleneck?** Compute the exact
   multiply-add count of this section's `PatchEmbedding` for a $96\times96\times3$
   image at `patch_size` 16 vs. 8, and compare both against the FLOPs of one
   `ViTBlock`'s attention layer at the resulting token count; report at which
   patch size the stem stops being negligible next to one transformer block.
   *Provenance:* adapted from "Understanding Deep Learning" Problem 12.9
   (overlap medium — same patch-embedding compute comparison, applied to this
   section's own module and image size rather than a DaViT comparison).
1. [extended] **A DeiT-style distillation micro-experiment.** Train the
   `CompactResNet` of this section as a teacher, then train a small ViT
   student with an added distillation loss (KL divergence to the teacher's
   softened logits, weight $\lambda$, temperature $T$) alongside the
   classification loss, at the same 10-epoch budget. Does distillation from
   the parameter-matched CNN close any of this section's measured accuracy
   gap? Report accuracy for at least two choices of $\lambda$.
   *Provenance:* inspired by Touvron et al. 2021 (DeiT), already cited
   in-text (overlap low — DeiT's actual recipe uses a distillation token and a
   different data/schedule; we adapt only the core distillation-loss idea to
   this section's small Fashion-MNIST setup).
