# Chapter Overview: chapter_recurrent-modern

This chapter (gated recurrence through Mamba, matrix-state/SSD, DeltaNet, test-time
regression, and hybrids) covers the newest material in the book, and its own 45
existing exercises are already excellent: precise, quantitative, and — per the prior
style review — essentially free of the "try things and see" vagueness flagged
elsewhere. Disposition below is therefore keep-heavy: 44 of 45 kept as-is; only
hybrids.md's exercise 1 needs a structural rewrite (six chained instructions split
into lettered subparts). The dominant finding across six of the seven sections
(everything past gated recurrence) is that there is essentially **no course-homework
tradition yet**: S4 (2022), Mamba (2023), Mamba-2/SSD (2024), DeltaNet's revival
(2024), Titans/test-time-regression (2025), and hybrids (Jamba/Samba/Kimi Linear,
2024-2025) are all too recent. The two strongest courses that teach this material at
all — Stanford CS336 and CMU 10-423/623 — cover it in one lecture each with zero
accompanying homework, and every annotated-implementation or blog source checked
(Annotated S4, Grootendorst's visual guide, Gu & Dao's State Space Duality series,
Songlin Yang's DeltaNet Explained, Raschka's Gated DeltaNet chapter, mamba-minimal,
flash-linear-attention) is expository or reference code with no posed exercises. Only
gated recurrence sits in a mature exercise tradition (SLP3, CS224n, CMU 11-785) — and
even there, SLP3's current 3rd-edition draft has dropped its chapter-end exercises
entirely. The single most reusable external source is HazyResearch's Zoology
blogposts/MQAR task, already the methodological basis of this chapter's own recall
experiments. Totals: 7 sections, 45 existing exercises (keep 44, rewrite 1, drop 0),
35 new problems proposed (5 per section).

---

## chapter_recurrent-modern/lstm.md — Gated Recurrence

**Topic:** Multiplicative gating (LSTM, GRU) as the fix for vanishing/exploding BPTT
gradients; depth and bidirectionality as structural axes.
**Current exercises:** 10; disposition: keep 10, rewrite 0, drop 0 — every item names
an exact ablation or comparison and a metric (perplexity, wall-clock, parameter
count); the style review flagged zero clarity defects in this file.

**External sources found:**
- Stanford CS224n, Assignment #3 (Spring 2024), "Neural Machine Translation with
  RNNs" — students implement a bidirectional-LSTM encoder / unidirectional-LSTM
  decoder with multiplicative attention from scratch across parts (a)-(f), then
  answer a written question, part (g), on why the encoder's padding mask must zero
  out attention scores at PAD positions before the softmax. Verified by fetching the
  handout directly. — https://web.stanford.edu/class/cs224n/assignments/a3_spr24_student_handout.pdf
- CMU 11-785 (Introduction to Deep Learning), HW3 Part 1, "RNNs and GRUs and CTCs and
  Search, Oh My!!" (Spring 2021 offering; the same HW recurs most years, later moved
  to HW4) — a from-scratch numpy implementation of RNN and GRU cells (forward *and*
  backward, no autodiff permitted) inside a personal "mytorch" library, plus
  multiple-choice conceptual questions and greedy/beam search. Verified by fetching
  the writeup. — https://deeplearning.cs.cmu.edu/F21/document/homework/HW3/HW3P1_writeup.pdf
- Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft, Chapter 13
  "RNNs and LSTMs" — checked the chapter PDF directly: it ends at "13.9 Summary" →
  Historical Notes → bibliography, with **no "Exercises" heading at all**. Finding: a
  flagship NLP textbook's current draft has no chapter-end problem set for this
  topic (unlike its own older editions' earlier chapters). —
  https://web.stanford.edu/~jurafsky/slp3/13.pdf
- colah's blog, "Understanding LSTM Networks" (2015) — confirmed purely expository;
  closes with reading recommendations (attention, Grid LSTMs), not posed exercises. —
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Proposed problem set** (5 problems):
1. [short-code] **Backward pass by hand.** Derive and implement the backward pass of
   `LSTMScratch` (gradients into each gate and into $\mathbf{C}_{t-1}$) without
   autodiff, in the spirit of CMU 11-785's "no autodiff toolboxes" rule. Verify every
   gradient against `torch.autograd`/`jax.grad` on a short random sequence to a
   stated tolerance (e.g. $10^{-4}$ relative error).
   *Provenance:* adapted from CMU 11-785, HW3 Part 1 (2021) (overlap medium — same
   from-scratch-backward discipline, applied to this section's own cell).
1. [short-code] **Padding masks in a batched LSTM.** Batch *Time Machine* training
   windows of two different lengths by right-padding the shorter ones with a PAD id;
   naively averaging the loss over the padded batch changes it. Build a boolean mask
   (paralleling CS224n's `enc_masks`) that zeroes the loss and gradient contribution
   of PAD positions in `LSTMScratch`'s training loop, and verify the padded batch
   reproduces the same per-token loss as running each sequence unpadded and singly.
   *Provenance:* inspired by CS224n, Assignment #3 (Sp24), part 1(g) (overlap low —
   same masking idea, moved from NMT encoder padding to a batched language model).
1. [conceptual] **Why the cell-state path doesn't vanish geometrically.** Using the
   $\mathbf{C}_{t-1} \to \mathbf{C}_t$ path of :eqref:`lstm_update` (additive, gated
   by $\mathbf{F}_t$, with no shared weight matrix in this specific path), derive the
   exact expression for $\partial \mathbf{C}_t / \partial \mathbf{C}_{t-k}$ as a
   product of $k$ forget-gate values. Explain why this product need not shrink
   geometrically the way the vanilla RNN's $\rho^k$ bound does, even for large $k$.
   *Provenance:* inspired by the unrolled-computation-graph framing CMU 11-785 uses
   to introduce this homework (overlap low — same pedagogical device, applied to the
   gated path rather than the vanilla one).
1. [extended] **Recall capacity of a fixed-size LSTM state.** Build a small
   write-then-query task: feed `LSTMScratch` a sequence of $n$ random key–value
   symbol pairs, then query each key using only the final $(\mathbf{H}_T,
   \mathbf{C}_T)$. Train and measure retrieval accuracy as $n$ grows past the hidden
   size $h$. At what ratio of $n$ to $h$ does accuracy visibly break down?
   *Provenance:* original.
1. [short-code] **Visualizing gate activations over time.** Instrument a trained
   `LSTMScratch` model to log $\mathbf{F}_t, \mathbf{I}_t, \mathbf{O}_t$ per hidden
   unit across a validation sequence, and plot them as a colah-style heatmap over
   time. Identify (by a variance-over-time threshold you define) at least one
   "long-memory" unit (near-constant $\mathbf{C}_t$) and one "short-memory" unit.
   *Provenance:* inspired by colah's blog's diagram style (overlap low — colah's
   post poses no exercises itself; this converts its expository visuals into a
   checkable task).

---

## chapter_recurrent-modern/ssm.md — Linear Recurrence and State Space Models

**Topic:** Linearizing the recurrence for parallel scan; continuous-time state space
models, discretization, and HiPPO memory initialization (S4/S4D).
**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — every item
specifies exact sweep ranges, tolerances, or an explicit question; the style review
flagged zero clarity defects.

**External sources found:**
- Sasha Rush (and Sidd Karamcheti), "The Annotated S4" (2022) — the book's own
  closest companion (cited in :numref:`chap_modern_rnn`'s bibliography). Fetched
  directly: an annotated-code-walkthrough in the Annotated-Transformer lineage
  (narrative prose interleaved with executable cells); the fetched content showed no
  distinct "Exercises" heading, consistent with the rest of that series, though the
  page is long enough that this is not an exhaustive negative. —
  https://srush.github.io/annotated-s4/
- Maarten Grootendorst, "A Visual Guide to Mamba and State Space Models" (2024) —
  confirmed via fetch: purely visual/explanatory (50+ custom figures); the article
  explicitly directs readers wanting hands-on practice to the Annotated S4 and the
  Mamba paper instead of posing its own exercises. —
  https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
- CMU 10-423/623 (Generative AI, Gormley & Virtue), Lecture 22 "State Space Models" —
  the book's own bibliography source. Fetched the course's coursework page directly:
  its five listed homeworks (PyTorch Primer, LLMs, Image Generation, Applying/
  Adapting LLMs, Multimodal Foundation Models) include **none** on SSMs — the
  material is lecture-only. — http://www.cs.cmu.edu/~mgormley/courses/10423-f25/coursework.html
- Stanford CS336 (Language Modeling from Scratch) — fetched the course site directly:
  Lecture 4 covers "attention alternatives" (including SSMs, per the schedule), but
  none of the five posted assignments (Basics, Systems, Scaling, Data, Alignment/RL)
  touch SSMs or Mamba. — https://cs336.stanford.edu/

**Proposed problem set** (5 problems):
1. [short-code] **Scan depth vs. sequential depth, measured.** Using the section's
   own `wall_clock` helper, time `sequential_scan` vs. `associative_scan` as $T$ grows
   over $\{128, 512, 2048, 8192, 32768\}$. Fit a log-log line to each and report both
   slopes. Does the scan's slope look sub-linear as theory predicts, and at what $T$
   does it overtake the sequential version despite doing more total work?
   *Provenance:* original.
1. [conceptual] **HiPPO by hand for $N=2$.** Write out :eqref:`eq_hippo` explicitly
   for $N=2$ and verify by hand that its eigenvalues are real and negative. Compare
   its off-diagonal entries to the diagonal initialization $a_n = -(n+1)$ used
   elsewhere in the section: which entries does the diagonal approximation discard,
   and what kind of cross-channel information could they in principle carry?
   *Provenance:* original.
1. [short-code] **A cross-check, Annotated-S4 style.** The Annotated S4 project
   validates a new kernel formula by computing it two independent ways before
   trusting it at scale. Do the same here: for the diagonal system $a_n = -(n+1)$ at
   a chosen $\Delta$, compute the discretized kernel (a) by unrolling the ZOH
   recurrence of :eqref:`eq_zoh_diag` step by step, and (b) via the closed-form
   kernel of :eqref:`eq_ssm_kernel`. Confirm agreement to floating-point tolerance
   for the first 50 taps.
   *Provenance:* inspired by The Annotated S4 (Rush & Karamcheti, 2022) (overlap low
   — same two-derivations debugging discipline, applied to this section's own S4D).
1. [conceptual] **Pricing the state, Mamba-guide style.** Grootendorst's guide
   centers on the SSM's fixed per-layer state (width $N$ per channel) versus an
   unbounded KV cache. Compute `S4D`'s exact per-layer state size (in numbers, not
   bytes) at `num_hiddens=256, num_states=4`, and compare it against a same-width
   single-head attention cache ($2 \times$ `num_hiddens` per token) at context
   lengths 512, 4096, and 65536. At what context length does the attention cache
   first exceed the SSM's fixed state?
   *Provenance:* inspired by Maarten Grootendorst's visual guide (2024) (overlap low
   — its central figure, turned into a computed comparison with this section's shapes).
1. [short-code] **S4D's kernel, before and after training.** Materialize `S4D`'s
   convolution kernel once at initialization and again after training on the
   sequential-image task; overlay both for a few channels. Does training mainly
   reshape the decay envelope (how fast the kernel falls off) or the oscillation
   frequency, and which change would help :numref:`subsec_ssm-step`'s streaming task?
   *Provenance:* original.

---

## chapter_recurrent-modern/mamba.md — Selective State Space Models

**Topic:** Making the SSM's step size and dynamics input-dependent (selectivity)
while keeping a parallel scan; the residual Mamba block.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — each item
specifies a concrete ablation and comparison metric; zero clarity defects flagged.

**External sources found:**
- CMU 10-423/623, Lecture 22 (same slide deck as ssm.md, covers Mamba too) — same
  finding: no accompanying homework (verified above).
- Stanford CS336, Lecture 4 (same lecture as ssm.md) — same finding: no assignment
  covers Mamba (verified above).
- johnma2006/mamba-minimal — the book's own bibliography source. Fetched directly:
  confirmed a single-file reference implementation ("deliberately without the
  kernels") with a demo notebook, no tutorial or posed exercises. —
  https://github.com/johnma2006/mamba-minimal
- HazyResearch, "Zoology" blogpost series (Blogposts 0–2, Dec 2023) and repo — found
  via search and abstract/description: a research writeup (not a course) introducing
  multi-query associative recall (MQAR) as the diagnostic explaining why sub-
  quadratic mixers underperform attention on recall, motivating this chapter's own
  recall experiments (credited in :numref:`chap_modern_rnn`'s bibliography). No
  posed exercises, but MQAR is a directly reusable task template. —
  https://hazyresearch.stanford.edu/blog/2023-12-11-zoology0-intro ,
  https://github.com/HazyResearch/zoology

**Proposed problem set** (5 problems):
1. [short-code] **MQAR instead of marked-symbol copying.** Zoology argues that
   recalling a handful of *marked* symbols (this section's `selective_copy`) is
   easier than full multi-query associative recall, where every position is a
   key–value pair. Modify the task generator so `num_marked == num_steps`, rerun the
   Mamba-vs-LTI-baseline comparison of existing exercise 2 at `num_states=4` and
   `16`, and report whether the accuracy "capacity cliff" moves to shorter sequences.
   *Provenance:* adapted from HazyResearch's Zoology, Blogpost 1 (2023) (overlap
   medium — reuses their MQAR task definition inside this section's own harness).
1. [conceptual] **Predict the ablation, then check it.** For each of $\Delta_t$,
   $\mathbf{B}_t$, $\mathbf{C}_t$ held input-*independent* in turn (existing exercise
   1 runs two such ablations), state in advance which specific failure mode of
   selective copying you expect and why — before running anything. Then check your
   three predictions against existing exercise 1's results.
   *Provenance:* original (a predict-first framing already used elsewhere in this
   chapter, e.g. deltanet.md's existing exercise 1).
1. [short-code] **One-file readability, mamba-minimal style.** mamba-minimal packs
   the whole architecture into one file "deliberately without the kernels," as a
   readability benchmark. Count non-blank, non-comment lines needed to define
   `SelectiveSSM` + `MambaBlock` + `Mamba` in this section vs. mamba-minimal's
   `model.py`, and name the single biggest source of any length difference.
   *Provenance:* inspired by johnma2006/mamba-minimal (overlap low — same "how many
   lines does this really take" framing, applied to this section's own listing).
1. [short-code] **Selective vs. LTI FLOPs at inference.** Using the stepped `step`
   function from existing exercise 5, count the exact multiply-accumulates needed to
   advance the state by one token for `SelectiveSSM` (input-dependent
   $\Delta,\mathbf{B},\mathbf{C}$) vs. `S4D`'s LTI recurrence (:numref:`sec_ssm`,
   fixed $\Delta,\mathbf{B},\mathbf{C}$). Does selectivity add asymptotic per-token
   cost, or only a constant-factor overhead from computing $\Delta_t,
   \mathbf{B}_t,\mathbf{C}_t$?
   *Provenance:* original.
1. [extended] **Reproduce one Zoology finding at teaching scale.** Zoology reports
   near-perfect MQAR accuracy for attention and a plateau for sub-quadratic mixers as
   the number of keys grows. Using the MQAR harness from problem 1 above plus a
   plain scaled-dot-product-attention encoder of matched width, plot accuracy vs.
   number of keys for attention-only and Mamba-only. Does your Mamba curve degrade
   the way Zoology's sub-quadratic mixers do?
   *Provenance:* adapted from HazyResearch's Zoology, Blogposts 0–2 (2023) (overlap
   medium — reproduces their qualitative recall gap at this book's scale).

---

## chapter_recurrent-modern/matrix-state.md — Matrix-State Recurrences and State Space Duality

**Topic:** The matrix-valued fast-weight recurrence unifying linear attention,
RetNet, GLA, and Mamba-2; capacity, decay forms, and the chunked (SSD) algorithm.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — each item names
exact sweep values and the quantity to report; zero clarity defects flagged.

**External sources found:**
- Albert Gu and Tri Dao, "State Space Duality (Mamba-2)," Part I (2024) — the book's
  own bibliography source for exactly this section's duality argument. Fetched
  directly: purely explanatory (Introduction → SSD Model → State Space Duality →
  Best of Both Worlds → Mamba-2 Architecture), no posed exercises. —
  https://goombalab.github.io/blog/2024/mamba2-part1-model/
- Songlin Yang, "Linear Attention and Beyond" slides and interactive tutorial (with
  Sasha Rush, 2025) — confirmed to exist (GitHub repo + recorded tutorial covering
  linear attention through Mamba-2/DeltaNet/TTT/Titans) via search, but the slide
  deck's internal content was not independently fetched, so presence/absence of a
  posed problem inside it is unconfirmed either way. —
  https://github.com/sustcsonglin/linear-attention-and-beyond-slides
- flash-linear-attention (fla-org) — the book's own bibliography source for
  production kernels. Fetched directly: confirmed a kernel/library repo (RetNet,
  GLA, DeltaNet, Mamba, RWKV) with usage docs and a separate training framework
  ("Flame"), no tutorial with posed exercises. —
  https://github.com/fla-org/flash-linear-attention
- **Finding:** no course homework on matrix-state recurrences / SSD was found
  anywhere (CMU 10-423's parallel lecture, "State Space Models + Hybrid Models," has
  the same no-homework pattern as its SSM lecture).

**Proposed problem set** (5 problems):
1. [short-code] **Reading fla-org's real GLA kernel.** flash-linear-attention ships a
   production kernel for gated linear attention — the same per-coordinate decay
   model as existing exercise 4's `segsum`-based teaching implementation. Find the
   file implementing GLA's chunked recurrence in the repo, and write a short
   correspondence table mapping each piece of your own `matrix_state_chunked` to its
   named counterpart there, plus one sentence on what the production version does
   that yours doesn't.
   *Provenance:* inspired by flash-linear-attention (overlap low — operationalizes
   the book's own bibliography pointer into a concrete reading task).
1. [conceptual] **Decay ladder, compared at fixed parameter count.** Write the
   transition $\mathbf{D}_t$ for RetNet, GLA, and Mamba-2 from
   :eqref:`eq_ms-decay-ladder`. Holding the number of decay parameters per head
   fixed, which model's decay can represent the widest range of distinct effective-
   memory horizons (:numref:`sec_ssm`'s definition) simultaneously within one head,
   and why does that follow from scalar vs. per-coordinate vs. input-dependent decay?
   *Provenance:* original.
1. [short-code] **Capacity law with correlated keys.** Existing exercise 2 assumes
   independent unit-norm keys — a confounder :numref:`chap_modern_rnn`'s own
   experiment table names explicitly. Redo the capacity sweep with keys correlated at
   $\rho \in \{0, 0.3, 0.6\}$ (mix each key with a shared random direction) at fixed
   $\gamma=1$, and compare measured retrieval error against the $(n-1)/d_k$
   prediction of :eqref:`eq_ms-retrieval-error`. At which $\rho$ does the prediction
   visibly break, and in which direction?
   *Provenance:* original (directly answers the confounder the book's own index
   names for this experiment).
1. [short-code] **mLSTM stabilizer, ablated.** Instrument `mlstm_naive` to log the
   running max of its unnormalized accumulator during a forward pass at forget
   pre-activation mean 2, and plot it on a log scale beside `mlstm_stabilized`'s
   bounded equivalent. At which step does the naive version's value pass float32's
   max representable magnitude, and does that match existing exercise 5's prediction?
   *Provenance:* original.
1. [extended] **Mapping the FLOP-optimal chunk size.** Existing exercise 1 finds
   $C \approx d$ optimal at one $(T, d)$ pair. Sweep the chunked-vs-dual FLOP ratio
   across $d \in \{64, 128, 256\}$ and $T \in \{2048, 8192, 32768\}$, and plot the
   FLOP-optimal $C$ against both. Does the optimum track $d$ alone, or does it also
   depend on $T$ once measured across a grid?
   *Provenance:* original.

---

## chapter_recurrent-modern/deltanet.md — DeltaNet and Corrective Memory Updates

**Topic:** The delta rule (read-before-write) as a fix for additive-memory
re-binding failure; the WY chunked algorithm; gating and expressivity.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — each item
specifies exact verification targets (tolerance, FLOP counts, a tracking-length
bound); the style review flagged only a cosmetic bold-bracket inconsistency, no
content issues.

**External sources found:**
- Songlin Yang, "DeltaNet Explained," Part I (2024) — the section's own closest
  primary-voice companion. Fetched directly: purely explanatory (Linear Attention as
  RNN → DeltaNet: Linear Attention with Delta Rule), no posed exercises; uses a
  "correcting a mis-aimed shot" analogy for the delta rule. —
  https://sustcsonglin.github.io/blog/2024/deltanet-1/
- Sebastian Raschka, "Gated DeltaNet, from scratch" bonus chapter of *LLMs from
  Scratch* (covering Qwen3-Next's linear-attention layer) — fetched directly:
  confirmed a pure code walkthrough with a memory-savings helper script, no posed
  exercises. — https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet
- Songlin Yang, "Linear Attention and Beyond" slides/tutorial — same as noted under
  matrix-state.md: confirmed to exist, internal exercise content unconfirmed.
- flash-linear-attention — confirmed (see matrix-state.md) to be a kernel library
  with no posed exercises, but it ships a production DeltaNet kernel.

**Proposed problem set** (5 problems):
1. [short-code] **Raschka's memory-savings script, on this section's own model.**
   Raschka's bonus chapter computes state-vs-KV-cache memory savings for Qwen3-
   Next's actual configuration. Write the analogous computation for this section's
   own `GatedDeltaNet` (its `num_heads`, per-head width) at context lengths 4K, 64K,
   and 1M, fp16. At what context length does your model's fixed state undercut one
   attention layer's cache, and how does that crossover compare to Raschka's
   reported numbers for the much larger Qwen3-Next?
   *Provenance:* adapted from Sebastian Raschka's Gated DeltaNet chapter (overlap
   medium — same memory-accounting script, re-run at this section's own scale).
1. [conceptual] **The delta rule as least squares, made precise.** Songlin Yang
   motivates the delta rule with "correcting a mis-aimed shot" rather than
   overwriting. Starting from :eqref:`eq_dn-recall-loss` and :eqref:`eq_dn-gradient`,
   write out the gradient term by term to show a delta-rule step of size $\beta_t$ is
   exactly one gradient step on the per-token recall loss, and identify which term
   plays the role of "how far the last shot missed."
   *Provenance:* inspired by Songlin Yang's "DeltaNet Explained, Part I" (2024)
   (overlap low — same analogy, formalized against this section's own equations).
1. [short-code] **A third overwrite regime: bursty re-binding.** Existing exercises 1
   -2 test uniform re-binding rates and random keys. Modify `make_task` so a fixed
   "hot" pair (2 of the 8 keys) is rebound on every step while the rest rebind at
   rate $R$, and measure delta-rule recall on the hot pair vs. the rest separately.
   Does :numref:`subsec_dn-trained`'s near-perfect recall survive this concentration?
   *Provenance:* original.
1. [short-code] **RWKV-7's transition, implemented and checked.**
   :eqref:`eq_dn-rwkv7` generalizes the gated update further. Implement it as a new
   `rwkv7_recurrent` alongside `delta_recurrent`/`delta_recurrent_matrix`, and verify
   a token-by-token loop against a batched matrix-form version to floating-point
   tolerance, mirroring the section's own verification style.
   *Provenance:* original.
1. [extended] **Word problems beyond $S_3$.** Existing exercise 6 hand-constructs a
   2-reflection cell tracking $S_3$ perfectly. Work out how many reflections $n_h$ a
   single delta micro-step needs to track $S_4$ (24 elements) instead, implement that
   many micro-steps per token following :numref:`subsec_dn-reflection`'s
   construction, and verify perfect tracking on random words up to length 64. What is
   the smallest $n_h$ that works, and does it match your hand count?
   *Provenance:* original (escalates the section's own existing exercise 6 to a
   larger group).

---

## chapter_recurrent-modern/test-time-regression.md — Learning at Test Time

**Topic:** Attention, linear attention, the delta rule, Longhorn, and Titans unified
as online regression of values on keys, with weighting, function class, and
optimizer as the three design axes.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — every item names
an exact derivation target or sweep with a numeric comparison; the highest crossref
density in the chapter, but no clarity defects flagged.

**External sources found:**
- Wang, Shi, and Fox, "Test-Time Regression" (2025) — the section's own explicit
  companion paper (:cite:`Wang.Shi.Fox.2025`). Fetched the abstract directly:
  confirmed a unifying-framework research paper (memorization as test-time
  regression, recovering Transformers/linear attention/SSMs as special cases of
  weighting/function-class/optimizer choices); a paper, so no exercises, as expected.
  — https://arxiv.org/abs/2501.12352
- Behrouz, Zhong, and Mirrokni, "Titans: Learning to Memorize at Test Time" (2025) —
  found via search/secondary summaries (not independently fetched from arXiv): a
  neural long-term memory module optimized inside the forward pass via a "surprise"
  gradient signal; no coursework found referencing it. —
  https://arxiv.org/abs/2501.00663
- ASAP seminar series — fetched directly: confirmed a passive virtual-seminar talk
  listing (schedule, recordings, Discord/Zoom links), no coursework or exercises. —
  https://asap-seminar.github.io/
- **Finding:** no course homework on test-time regression, Longhorn, or Titans was
  found anywhere — this is the newest material in the chapter (all cited work is
  2025) and sits entirely outside any current course's homework.

**Proposed problem set** (5 problems):
1. [short-code] **A recipe-table checkpoint, from the paper itself.** The Test-Time
   Regression paper organizes fast-weight variants by (weighting, function class,
   optimizer) exactly like this section's :numref:`tab_ttr-recipe`. Pick one entry
   from the paper's table not yet in this section's own table, implement its update
   rule using the `nadaraya_watson`/`nw_attention`/`objective` scaffolding, and add a
   measured row using the spectrum experiment of :numref:`subsec_ttr-spectrum`.
   *Provenance:* adapted from Wang, Shi, and Fox (2025) (overlap medium — extends
   this section's own recipe table using a variant documented in its source paper).
1. [conceptual] **Titans' surprise metric, derived from the regression view.** Titans
   motivates its update with a "surprise" signal resembling a per-token
   reconstruction gradient. Starting from :eqref:`eq_ttr-longhorn-objective` and the
   momentum extension :eqref:`eq_ttr-titans`, show that Titans' surprise term is
   exactly the gradient contribution in :eqref:`eq_ttr-titans` before momentum is
   applied, and state what the momentum term adds beyond a memoryless gradient step.
   *Provenance:* inspired by Titans (Behrouz, Zhong, Mirrokni, 2025) (overlap low —
   the paper's intuition, formalized against this section's own equations, which the
   section itself already connects to Titans).
1. [short-code] **Beyond Gaussian: a Laplace test-time kernel.**
   :numref:`subsec_ttr-bandwidth` learns a Gaussian kernel's bandwidth by leave-one-
   out gradient descent. Swap in a Laplace kernel ($\exp(-|q-k|/\sigma)$) inside
   `nadaraya_watson`/`loo_loss`, retrain $\sigma$, and compare the learned fit and
   final leave-one-out loss to the section's Gaussian result on the same data. Does
   the optimal $\sigma$ differ in a way explained by the two kernels' tail behavior?
   *Provenance:* original.
1. [short-code] **A fourth solver on the spectrum.** :numref:`subsec_ttr-two-loops`'s
   inner-solver comparison places the delta-rule step and 30-pass online GD on a
   spectrum ending at the batch ridge solution. Add a diagonal (Adagrad-style)
   preconditioned update inside `online_gd`, and place it on the same plot. Does
   per-coordinate adaptation move it closer to or further from the batch solution
   than plain online GD at matched step budget?
   *Provenance:* original.
1. [extended] **Consistent vs. random-walk drift, at scale.** Existing exercise 5
   sweeps momentum $\eta$ under both drift types at one state dimension. Extend
   :numref:`subsec_ttr-tracking`'s experiment to $d \in \{8, 32, 128\}$ crossed with
   both drift types (holding drift speed comparable across types, however you define
   that), and plot tracking error vs. $d$ for the best $\eta$ per cell. Does the
   optimal $\eta$ shrink, grow, or stay flat as $d$ grows, per drift type?
   *Provenance:* original (a gridded extension of the section's own existing
   exercise 5).

---

## chapter_recurrent-modern/hybrids.md — Hybrid Architectures

**Topic:** Interleaving fixed-state (recurrent/linear-attention) and full-attention
layers to trade cache growth against exact-recall capacity; measured recall sweeps,
design rules, and distillation from pretrained transformers.
**Current exercises:** 5; disposition: keep 4, rewrite 1, drop 0 — exercise 1 chains
roughly six sequential instructions into one un-lettered list item (flagged in the
prior style review as the chapter's worst clarity offender, though each individual
step is clear); the other four are cleanly scoped as written.

**External sources found:**
- HazyResearch, "Zoology" blogposts and repo — already the methodological basis for
  this section's own recall task (credited in :numref:`chap_modern_rnn`'s
  bibliography); confirmed (see mamba.md) to be a research writeup with no posed
  exercises, but its MQAR task is directly reusable here too. —
  https://github.com/HazyResearch/zoology
- Jamba, Samba, and Kimi Linear (the three hybrid architectures this section's own
  recipe table draws on) — confirmed via search to have no associated course
  homework anywhere; coverage is limited to the original papers and secondary
  technical write-ups (e.g. Medium explainers), not coursework.
- ASAP seminar series — confirmed (see test-time-regression.md) to be a passive talk
  listing with no coursework; the section this chapter is closest to for "where this
  topic continues."
- CMU 10-423/623, Lecture 21, "State Space Models + Hybrid Models" — same course as
  ssm.md/mamba.md; same finding, no dedicated homework.
- **Finding:** no course-exercise tradition exists yet for hybrid attention/recurrent
  architectures at any institution checked.

**Proposed problem set** (5 problems):
1. [extended] **Placement, self-discovered** *(rewrite of existing exercise 1 —
   content unchanged, split into independently checkable steps)*.
   1. Rerun the recall sweep with the attention layer first (`'AGGG'`) and last
      (`'GGGA'`) instead of mid-stack; report recall for each at `num_pairs=32`.
   1. Note the position-table confound: `RecallModel` is built with `max_len = 2 *
      num_pairs`, so a longer evaluation sequence either hits the `forward` guard or
      exercises untrained position rows.
   1. Remove the confound: retrain the three placements with `pos=False`, and
      confirm the `num_pairs=32` sweep still reproduces the original accuracies.
   1. Extend `make_recall` to insert filler tokens between write and query phases
      (reserve one extra key index, `num_keys=65`); train at zero padding, then
      evaluate at 32 tokens of padding.
   1. Report which placement degrades most under padding, and whether this matches
      Samba's report that a single *front* attention layer breaks length
      extrapolation :cite:`Ren.Liu.Lu.ea.2024`.
   *Provenance:* original (restructuring of the book's own existing exercise 1 to
   fix the chained-instructions defect the style review flagged).
1. [short-code] **HazyResearch's MQAR, not this section's phase-separated recall.**
   This section's `make_recall` segregates writes then queries into two phases;
   Zoology's MQAR interleaves a query after each write instead. Modify `make_recall`
   accordingly, retrain the section's three matched models (`'GGGG'`, mid-stack
   hybrid, `'AAAA'`), and compare recall against the section's own phase-separated
   numbers at matched `num_pairs`. Does interleaving hurt the pure recurrent stack
   specifically, or all three equally?
   *Provenance:* adapted from HazyResearch's Zoology (2023) (overlap medium — the
   harness this chapter's own bibliography already credits for the recall results).
1. [conceptual] **Pricing Kimi Linear's actual ratio.** Existing exercise 3 compares
   the measured recall/memory knee against Kimi Linear's reported 3:1 ratio in the
   abstract. Look up Kimi Linear's actual per-block layout (already summarized in
   this section's own recipe table, :numref:`tab_hy-recipe`) and redo existing
   exercise 2's 1M-token, 80GB pricing exercise using that real layout in place of
   the "12.5% hybrid" already computed there. How many more or fewer concurrent
   1M-token users does the real ratio support?
   *Provenance:* original (a more precise version of the section's own existing
   exercise 2, anchored to a configuration already named in the text).
1. [short-code] **Distillation vs. training from scratch.**
   :numref:`subsec_hy-distill` describes converting a trained transformer into a
   hybrid rather than training one from scratch, but does not code it up. Take the
   section's own trained pure-attention (`'AAAA'`) character-level LM, initialize a
   `'GGGA'` hybrid's attention layer from the corresponding transformer layer's
   weights (leaving the recurrent layers freshly initialized), and briefly fine-tune
   on the same data. Compare final perplexity and the training curve against the
   section's from-scratch `'GGGA'` hybrid.
   *Provenance:* original (operationalizes a procedure this section already
   describes in prose).
1. [conceptual] **Counting bound, revisited for GQA.** :numref:`subsec_hy-limits`'s
   counting bound assumes a plain multi-head attention cache. Redo existing exercise
   2's 1M-token pricing exercise for a transformer using grouped-query attention (4
   query heads per KV head, $n_\textrm{kv}=8$ unchanged, so 32 total query heads
   share 8 KV heads), and recompute how many concurrent users fit in 80 GB. Does GQA
   alone close enough of the gap to make the recurrent/hybrid design unnecessary at
   this scale, or does the advantage survive?
   *Provenance:* original (extends the section's own existing exercise 2 to a
   cache-saving technique already covered in :numref:`sec_kv-cache`).
