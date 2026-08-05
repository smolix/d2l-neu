# Exercise Catalog — chapter_attention

**Chapter overview.** The two strongest external matches are Stanford CS224N
Assignment 4 ("Self-Attention, Transformers, and Pretraining," Winter 2025)
and Stanford CS336 Assignments 1–2 ("Language Modeling from Scratch," Spring
2026): CS224N's Q1/Q2 independently reconstruct this chapter's own
single-head-must-average proposition and permutation-equivariance proof from
different arguments, and CS336's A1 has students implement RoPE, scaled
dot-product attention, and causal multi-head attention from a from-scratch
interface, while its A2 has them write a Triton FlashAttention-2 kernel and
benchmark it — a near-exact systems-level complement to attention-at-scale.md.
Simon Prince's *Understanding Deep Learning* (Problems 12.1, 12.3–12.5, 12.7,
12.10) supplies a third independent derivation for nearly every major claim in
the chapter. ARENA's mechanistic-interpretability curriculum is the best match
for what-attention-computes.md, built on the same QK/OV framework. Coverage
gaps with no verified external homework tradition: Nadaraya–Watson kernel
regression, the masking numerical-precision idiom, MQA/GQA, dilated/Longformer
windows, position interpolation/YaRN, and the linear-attention–to–Mamba
connection — all remain paper- or textbook-only topics in the courses
surveyed. Five of six files (all but queries-keys-values.md) were already
defect-free and code-grounded per the prior style review, and this catalog
keeps essentially all of their exercises intact; external material mostly
corroborates rather than replaces this chapter's own (unusually strong)
exercise sets.

---

## chapter_attention/queries-keys-values.md — Queries, Keys, and Values

**Topic:** Attention as a differentiable soft lookup over (key, value) pairs;
softmax turns any scoring function into valid weights; Nadaraya–Watson kernel
regression as attention with a hand-picked (unlearned) kernel.

**Current exercises:** 6; disposition: keep 3, rewrite 2, drop 1 — this is the
chapter's one "bare-prompt" outlier per the prior style review. Three items
(the covariance-gradient proof, the unit-sphere simplification, the
SGD-bandwidth exercise) are concrete and code-grounded and are kept unchanged;
three (classical-database matching, "design a differentiable search engine,"
a Squeeze-and-Excitation reading prompt) are unmodified legacy d2l prompts
with no deliverable or success criterion and are rewritten or dropped.

**External sources found:**
- Stanford CS224N, Assignment 4 ("Self-Attention, Transformers, and
  Pretraining"), Q1(a)–(b), Winter 2025 — asks when the softmax distribution
  concentrates almost all its weight on one key (and what that implies for
  the output), then asks students to design a query that instead averages
  exactly two orthogonal-keyed values — https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.1 (MIT Press,
  2023) — counts the weights/biases needed to form queries, keys, and values
  and the resulting number of attention weights, then compares against a
  fully-connected network over the same inputs — https://github.com/udlbook/udlbook/blob/main/UDL_Answer_Booklet_Students.pdf
- Nadaraya–Watson kernel regression itself has no external homework tradition
  beyond this book's own classic-edition lineage (checked D2L's earlier
  "Attention Pooling" chapter, which is this book's own ancestor, not an
  independent source). This is a finding, not an omission: outside
  deep-learning coursework treats Nadaraya–Watson as a nonparametric-statistics
  topic, not an attention exercise.
- No verified external exercise treats "attention as differentiable database
  lookup" as its own homework framing — CS224N and CS336 both jump straight to
  scaled dot-product attention without this section's kernel-regression
  detour, a coverage gap this book's own approach fills rather than borrows.

**Proposed problem set** (7 problems):
1. [conceptual] **When attention copies.** State precisely what must be true
   of a query $\mathbf{q}$ and the keys $\{\mathbf{k}_i\}$ for the softmax
   weights to place almost all their mass on a single key $\mathbf{k}_j$
   (deliverable: one sentence giving the condition, e.g. in terms of the
   score gap to the runner-up), then describe the resulting output in that
   regime.
   1. Now go the other way: given two orthonormal keys $\mathbf{k}_a,
      \mathbf{k}_b$, exhibit a query $\mathbf{q}$ (as a function of
      $\mathbf{k}_a,\mathbf{k}_b$) for which the output is approximately
      $\tfrac12(\mathbf{v}_a+\mathbf{v}_b)$, and verify it numerically against
      `eq_softmax_attention`.
   *Provenance:* adapted from Stanford CS224N A4 Q1(a)-(b), 2025 (overlap
   high; cite on adoption).
1. [conceptual] **Attention's gradient is a covariance.** (Section's existing
   exercise 2, unchanged: prove $\nabla_{\mathbf q}\mathrm{Attention}(\mathbf
   q,\mathcal D) = \mathrm{Cov}_{p(\mathbf k_i;\mathbf q)}[\mathbf k_i]$ under
   $a(\mathbf q,\mathbf k_i)=\mathbf q^\top\mathbf k_i$, $\mathbf k_i=\mathbf
   v_i$.)
   *Provenance:* original (kept from the section's existing set).
1. [short-code] **Boxcar and triangular kernels in action.** Extend
   `nadaraya_watson` to accept the boxcar and triangular kernels of
   `fig_attention_kernels` instead of only the Gaussian, matching bandwidths
   so all three touch zero at the same distance, then re-run the $\sigma=0.5$
   regression and plot all three fits and their attention-weight heatmaps
   side by side. Deliverable: a plot showing where the boxcar kernel's hard
   cutoff visibly differs from the Gaussian's smooth falloff, with a
   one-paragraph explanation tied to which training points each query can
   "see" near a gap in `x_train`.
   *Provenance:* original (extends the section's own kernel gallery, which is
   illustrated but never coded beyond the Gaussian).
1. [conceptual] **Dot products on the sphere.** (Section's existing exercise
   5, unchanged: simplify $\|\mathbf q-\mathbf k\|^2$ under $\|\mathbf
   x\|=1$; sets up the next section's dot-product motivation.)
   *Provenance:* original (kept).
1. [short-code] **Learning the bandwidth, honestly.** (Section's existing
   exercise 6, unchanged: SGD-learned $\sigma$, with and without excluding
   $(x_i,y_i)$ from its own estimate.)
   *Provenance:* original (kept).
1. [conceptual] **Counting attention's parameters.** For a soft lookup over
   $m$ key–value pairs with $\mathbf{q},\mathbf{k}_i\in\mathbb{R}^{d_k}$,
   $\mathbf{v}_i\in\mathbb{R}^{d_v}$, and a learned bilinear score
   $\mathbf{q}^\top\mathbf{M}\mathbf{k}_i$: count the parameters needed to
   compute all $m$ scores, and compare against a fully-connected network
   mapping the concatenated database (size $m(d_k+d_v)$) directly to a
   $d_v$-dimensional output. Deliverable: two closed-form counts and one
   sentence on how each scales with $m$, connecting back to the section's
   claim that attention uses "a fixed set of parameters" regardless of
   database size.
   *Provenance:* adapted from Simon Prince, *Understanding Deep Learning*,
   Problem 12.1 (overlap medium; cite on adoption).
1. [extended] **A tiny differentiable lookup table.** Build a database of 20
   (key, value) pairs where keys are random unit vectors in $\mathbb R^{16}$
   and values are one-hot class labels; generate queries as noisy copies of a
   random subset of keys ($\mathbf q = \mathbf k_i + \epsilon$). Sweep the
   Gaussian-kernel bandwidth and report retrieval accuracy (does $\arg\max_i
   \alpha(\mathbf q,\mathbf k_i)$ recover the true $i$?) against noise level,
   then repeat with $m$ scaled up to 200 keys. Deliverable: an
   accuracy-vs-noise curve at two database sizes and a short explanation of
   why growing $m$ alone (with fixed key separation) degrades retrieval even
   without adding noise.
   *Provenance:* inspired by the section's own exercise 3 ("design a
   differentiable search engine"), reworked with a concrete dataset, sweep,
   and success criterion (overlap low).

---

## chapter_attention/attention-scoring.md — Attention Scoring and Masking

**Topic:** Deriving scaled dot-product attention from the Gaussian kernel; why
$1/\sqrt d$ matters (measured via entropy and softmax-Jacobian norm); masked
softmax for padding/causality and its numerical-precision pitfalls; composing
masks by logical AND; batched attention; additive attention as the historical
alternative.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — per the
prior style review this file is clean and code-grounded throughout (no
defects, no clarity flags). It is one of the chapter's strongest sets and is
kept entirely intact, with external material added rather than substituted.

**External sources found:**
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.4 (MIT Press,
  2023) — gives five softmax inputs including one outlier ($z_3=100$) and asks
  for all 25 partial derivatives $\partial y_i/\partial z_j$ — exactly the
  softmax Jacobian this section's exercise 4 derives and measures, an
  independently-converging exercise design — https://github.com/udlbook/udlbook/blob/main/UDL_Answer_Booklet_Students.pdf
- Stanford CS336, Assignment 1 ("Building a Transformer LM"), Problem
  (scaled_dot_product_attention), Spring 2026 — has students implement scaled
  dot-product attention taking a boolean mask (`True` = attend) of shape
  `(seq_len, seq_len)`, the complementary convention to this section's
  `valid_lens`-based `masked_softmax` — https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf
- Stanford CS336, Assignment 1, Problem (softmax), Spring 2026 — the standard
  max-subtraction numerical-stability trick for softmax overflow; a different,
  complementary numerical concern from this section's mask-fill-value idiom,
  since it addresses the exponentiation itself rather than the masked score.
- No external exercise was found that poses this section's specific
  concern — which mask *fill value* is safe across dtypes, and why $-10^6$
  and literal $-\infty$ are both wrong — as a homework question; this appears
  to be an angle original to this book.
- No external exercise was found comparing additive and dot-product attention
  on FLOPs/wall-clock (exercise 6); CS224N and CS336 each implement only one
  scoring function, without the historical additive-vs-dot-product comparison
  this section stages.

**Proposed problem set** (7 problems):
1. [short-code] **Distance-based attention.** (Section's existing exercise 1,
   unchanged: modify `DotProductAttention` to implement distance-based
   attention using $\|\mathbf k_i\|^2$.)
   *Provenance:* original (kept).
1. [short-code] **Queries and keys of different width.** (Existing exercise
   2, unchanged: score with $\mathbf q^\top \mathbf M \mathbf k$.)
   *Provenance:* original (kept).
1. [conceptual] **Counting the cost of one attention layer.** (Existing
   exercise 3, unchanged: FLOP and memory-bandwidth scaling of
   `eq_softmax_QK_V`.)
   *Provenance:* original (kept).
1. [conceptual] **The softmax Jacobian, derived and measured.** (Existing
   exercise 4, unchanged: derive $\mathrm{diag}(\boldsymbol\alpha) -
   \boldsymbol\alpha\boldsymbol\alpha^\top$, verify against autograd, compute
   its Frobenius norm at one-hot and uniform $\boldsymbol\alpha$, connect to
   the saturation experiment.) Optionally check your derivative formula
   against the five-value example $z=(-3,1,100,5,-1)$ from an equivalent
   exercise elsewhere, and confirm which of your two computed norms (one-hot
   or uniform) that outlier-containing vector's Jacobian most resembles.
   *Provenance:* original (kept); independently converges with Simon Prince,
   *Understanding Deep Learning*, Problem 12.4 (noted for the optional
   numeric check).
1. [conceptual] **Masking at the edges.** (Existing exercise 5, unchanged:
   valid length 0, $-\infty$ vs. finite-min, when a fully-masked query can
   arise.)
   *Provenance:* original (kept).
1. [short-code] **Additive vs. dot-product attention, timed.** (Existing
   exercise 6, unchanged: count parameters/FLOPs for both scorers, implement
   batched additive attention, time both at $d=h=64,256$.)
   *Provenance:* original (kept).
1. [short-code] **Two masking conventions, reconciled.** Write a converter
   between this section's `valid_lens`-based masking and CS336's boolean
   "`True` = attend" convention (shape `(seq_len, seq_len)`), and verify on
   ten random padding/causal configurations that `masked_softmax` under both
   conventions produces identical weights. Deliverable: the converter
   function plus a printed max-absolute-difference of 0 (or a caught bug)
   across the ten test cases.
   *Provenance:* adapted from Stanford CS336 Assignment 1, Problem
   (scaled_dot_product_attention), 2026 (overlap medium; cite on adoption).

---

## chapter_attention/multihead-attention.md — Multi-Head and Cross-Attention

**Topic:** A single head must average (proved and measured on a
value-blind "copy-both" task); learned multi-head attention as $h$
independently-projected heads recombined linearly, at no leading
parameter/FLOP cost when per-head width is $d/h$; self- vs. cross-attention
as two wirings of the same function.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — another
clean, defect-free file per the prior style review. Kept entirely, with one
problem added because an unusually close external precedent surfaced during
research.

**External sources found:**
- Stanford CS224N, Assignment 4, Q1(b)-(e), Winter 2025 — poses, with a
  different (Gaussian-perturbation) argument, essentially the same
  progression this section builds from scratch: design a query that averages
  two orthogonal-keyed values (Q1b); show this averaging is fragile once one
  key's covariance becomes anisotropic (Q1c); show that a second,
  independently-parameterized query head fixes it (Q1d-e) —
  https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.5 (MIT Press,
  2023) — asks why per-head dimension $D/H$ makes multi-head attention's
  implementation efficient, the same question this section's exercise 4
  derives quantitatively via FLOP counting — https://github.com/udlbook/udlbook/blob/main/UDL_Answer_Booklet_Students.pdf
- No external course exercise was found on multi-query/grouped-query
  attention (exercise 5); it remains largely a systems/efficiency-paper topic
  (Shazeer 2019; Ainslie et al. 2023) rather than assigned homework in the
  courses surveyed — a coverage gap, not an oversight.
- No external exercise was found asking students to design a
  head-importance/pruning experiment with the
  "jointly-prunable-but-not-individually" caveat (exercise 6); this looks
  like a framing original to this book.

**Proposed problem set** (7 problems):
1. [short-code] **Does a nonlinear readout help?** (Existing exercise 1,
   unchanged: MLP readout on Gaussian values, then $\pm1$-valued values.)
   *Provenance:* original (kept).
1. [short-code] **How close to degenerate can two heads get?** (Existing
   exercise 2, unchanged: noise + condition number as $\epsilon\to0$.)
   *Provenance:* original (kept).
1. [short-code] **Trained heads on the letter alignment.** (Existing exercise
   3, unchanged: 4-head `MultiHeadAttention` on "attention"/"translation".)
   *Provenance:* original (kept).
1. [conceptual] **Deriving the FLOP crossover.** (Existing exercise 4,
   unchanged: derive `eq_multihead-flops`; crossover $n$ at $d=512$.)
   *Provenance:* original (kept); independently converges with Simon Prince,
   *Understanding Deep Learning*, Problem 12.5.
1. [conceptual] **Multi-query and grouped-query attention.** (Existing
   exercise 5, unchanged: parameter/FLOP change under MQA/GQA.)
   *Provenance:* original (kept).
1. [conceptual] **Pruning heads without breaking pairs.** (Existing exercise
   6, unchanged: design a head-importance experiment; guard against jointly
   prunable pairs.)
   *Provenance:* original (kept).
1. [conceptual] **A second route to the same bound.** Redo the "one head must
   average" argument with CS224N's construction instead of this section's:
   keys are randomly sampled $\mathbf k_i\sim\mathcal N(\boldsymbol\mu_i,
   \Sigma_i)$ with orthonormal, unit-norm means $\boldsymbol\mu_i$. First take
   $\Sigma_i=\alpha I$ for vanishingly small $\alpha$ and design a query
   recovering $\tfrac12(\mathbf v_a+\mathbf v_b)$; then let $\Sigma_a=\alpha I
   + \tfrac12\boldsymbol\mu_a\boldsymbol\mu_a^\top$ (key $a$'s magnitude, not
   direction, becomes highly variable) and describe qualitatively what
   happens to the recovered mixture across resamples. Deliverable: one query
   expression per regime and a one-paragraph qualitative description — the
   failure mode that this section's own two-head construction then repairs,
   exactly as it repairs the value-blind case.
   *Provenance:* adapted from Stanford CS224N Assignment 4, Q1(b)-(c), 2025
   (overlap high; cite on adoption).

---

## chapter_attention/positional-information.md — Positional Information

**Topic:** Unmasked self-attention is permutation-equivariant (proved and
checked numerically); absolute (sinusoidal/learned), rotary (RoPE), and
distance-based (ALiBi/NoPE) schemes for supplying position; a controlled
experiment training five variants of the same tiny attention-only LM and
comparing perplexity at and beyond the training length.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the
cleanest file in the chapter per the prior style review (no defects, no
clarity flags). Every existing exercise is kept, and two are added because
the external tradition for this specific topic (RoPE) is unusually strong and
well-matched.

**External sources found:**
- Stanford CS224N, Assignment 4, Q2, Winter 2025 — proves $Z_\mathrm{perm}=
  PZ$ for a permuted input under given identities, then asks why this is
  problematic for text and whether adding a sinusoidal position vector fixes
  it, and whether two different positions can receive the same embedding —
  the same permutation-equivariance argument this section opens with, posed
  independently — https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.3 (MIT Press,
  2023) — an independent third proof of the identical permutation-equivariance
  proposition, using the book's own $\beta,\Omega$ notation for the
  query/key/value projections — https://github.com/udlbook/udlbook/blob/main/UDL_Answer_Booklet_Students.pdf
- Stanford CS336, Assignment 1, Section 3.4.3 and Problem (rope), Spring
  2026 — has students implement a `RotaryPositionalEmbedding` module from a
  from-scratch interface (`theta`, `d_k`, `max_seq_len`, applied via
  precomputed sin/cos buffers indexed by an explicit `token_positions`
  tensor) — a more general implementation contract than this section's
  fixed-order `rope()` helper — https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf
- Stanford CS336, Assignment 1, Problem (no_pos_emb), Spring 2026 — an
  ablation removing position information entirely from a RoPE-based
  transformer and comparing learning curves against the RoPE baseline, the
  same 'none'-vs-'rope' comparison this section's own five-way experiment
  includes.
- No external course exercise was found asking students to implement or
  evaluate position interpolation or YaRN (exercise 3); both remain
  paper-level techniques (Chen et al. 2023; Peng et al. 2024) without a
  verified homework tradition.

**Proposed problem set** (8 problems):
1. [short-code] **Causality breaks the symmetry.** (Existing exercise 1,
   unchanged: causal-mask shuffle experiment; explain where the equivariance
   proof breaks and why position 0's output is unchanged by
   permutations that fix it.)
   *Provenance:* original (kept); independently converges with Stanford
   CS224N A4 Q2(a) and Simon Prince's Problem 12.3, both of which pose the
   unmasked-case proof this exercise's masked-case contrast builds on.
1. [short-code] **How much position leaks through the mask?** (Existing
   exercise 2, unchanged: linear probe on 'none' vs. 'rope' final hidden
   state.)
   *Provenance:* original (kept).
1. [short-code] **Position interpolation, both halves.** (Existing exercise
   3, unchanged: rescale RoPE angles, then fine-tune at the scaled angles.)
   *Provenance:* original (kept).
1. [short-code] **When the position table dominates.** (Existing exercise 4,
   unchanged: scale the sinusoidal table to 0.02 and retrain.)
   *Provenance:* original (kept).
1. [short-code] **RoPE's base constant.** (Existing exercise 5, unchanged:
   retrain with base 100 and 1,000,000.)
   *Provenance:* original (kept).
1. [conceptual] **Reading the learned heads.** (Existing exercise 6,
   unchanged: visualize `rope` vs. `none` attention weights.)
   *Provenance:* original (kept).
1. [short-code] **RoPE against a from-scratch interface.** Re-implement
   `rope()` to CS336's specification: a class taking `theta`, `d_k`, and
   `max_seq_len` at construction, precomputing sin/cos buffers, and applying
   rotation via an explicit `token_positions` tensor of arbitrary leading
   batch shape (rather than assuming positions $0,\ldots,T{-}1$ in order).
   Verify it reproduces `TinyCharLM._rope`'s output on contiguous positions,
   then use the explicit-positions interface to rotate a batch where every
   example shares the same relative offset but starts from a *different*
   absolute position — something the section's own `offset`-only
   implementation cannot express in a single call. Deliverable: the class, a
   numerical agreement check, and the mixed-start-position example.
   *Provenance:* adapted from Stanford CS336 Assignment 1, Section 3.4.3 and
   Problem (rope), 2026 (overlap medium; cite on adoption).
1. [conceptual] **Can two positions share an encoding?** For the sinusoidal
   table of `eq_sinusoidal-def`, can two distinct positions $i\neq i'$ within
   `max_len` ever receive identical rows $\mathbf p_i = \mathbf p_{i'}$?
   Answer for sinusoidal encodings, then answer the analogous question for
   RoPE: can two distinct offsets $\delta\neq\delta'$ produce the same
   rotation $\mathbf R_\delta = \mathbf R_{\delta'}$, and if so, at what
   offset does the fastest-rotating feature pair first repeat? Deliverable: a
   yes/no with justification for each scheme, plus the numerical period of
   the fastest pair for the base-10000 encoding used in this section.
   *Provenance:* adapted from Stanford CS224N Assignment 4, Q2(b)(ii), 2025
   (overlap low; the question is asked there of the additive sinusoidal
   scheme only, extended here to RoPE's rotation-repeat analogue).

---

## chapter_attention/attention-at-scale.md — The Cost of Attention

**Topic:** Attention's quadratic time/memory cost, measured against a real
GPU allocator; exact memory-linear computation via online softmax (the
FlashAttention idea); sliding-window/sparse attention and receptive-field
growth with depth; linear attention as an exact recurrence with a fixed-size
state, and its connection to state-space models.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — dense,
compound, code-grounded exercises with no defects per the prior style review.
Kept in full, with one addition drawn from the strongest external match found
for the whole chapter.

**External sources found:**
- Stanford CS336, Assignment 2 ("Systems and Parallelism"), Problem
  (pytorch_attention), Spring 2026 — benchmarks a bare attention layer (no
  multi-head) across a $d_\mathrm{model}\in\{16,32,64,128\}\times
  n\in\{256,\ldots,16384\}$ grid, reports timings or out-of-memory errors, and
  asks for a memory-usage accounting at the smallest OOM configuration —
  essentially the same experiment as this section's own GPU-memory/timing
  measurements, run as a graded assignment rather than an in-text demo —
  https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf
- Stanford CS336, Assignment 2, Problems (flash_forward)/(flash_backward)/
  (flash_benchmarking), Spring 2026 — has students write an actual
  FlashAttention-2 forward and backward pass as a Triton kernel (tiling,
  online softmax, the logsumexp trick, recomputation in the backward pass)
  and benchmark it against the naive and `torch.compile`d versions — a
  graded implementation of the algorithm this section explains and
  implements at the chunked-NumPy/JAX level, one rung further toward real
  hardware.
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.10 (MIT Press,
  2023) — replaces $\exp(\mathbf k^\top\mathbf q)$ with a factorizing map
  $g(\mathbf k)^\top g(\mathbf q)$ and asks students to show this lets the
  key-side sum be precomputed, cutting incremental decode cost from $O(ND)$
  to $O(D')$ — the same kernel-factorization argument behind this section's
  linear-attention derivation, posed independently and from the
  decoding-cost angle — https://github.com/udlbook/udlbook/blob/main/UDL_Answer_Booklet_Students.pdf
- Simon J.D. Prince, *Understanding Deep Learning*, Problem 12.7 — asks what
  extra computation is needed to extend a precomputed masked self-attention
  layer by one new token, i.e. the incremental cost of KV-cached decoding
  that this section's linear-attention subsection contrasts against (a fixed
  16 KiB state vs. a linearly growing cache).
- No external course exercise was found on dilated/Longformer-style sliding
  windows (exercise 4) or on adding a learned/scalar decay to the
  linear-attention recurrence (exercise 6, which connects to Mamba); both
  remain paper-level topics (Beltagy et al. 2020; Dao and Gu 2024) without a
  verified homework tradition in the courses surveyed.

**Proposed problem set** (7 problems):
1. [conceptual] **The windowed FLOP crossover.** (Existing exercise 1,
   unchanged: 90%-quadratic threshold; dense-vs-windowed FLOP ratio.)
   *Provenance:* original (kept).
1. [conceptual] **The memory budget of chunking.** (Existing exercise 2,
   unchanged: peak-memory formula for `chunked_attention`; largest chunk size
   under 100 MB.)
   *Provenance:* original (kept).
1. [short-code] **Why the running maximum matters.** (Existing exercise 3,
   unchanged: drop the running max, run in fp16 at 10x score scale.)
   *Provenance:* original (kept).
1. [short-code] **Dilated windows.** (Existing exercise 4, unchanged:
   implement a dilated band mask; recompute receptive-field growth.)
   *Provenance:* original (kept).
1. [short-code] **Choosing a feature map.** (Existing exercise 5, unchanged:
   compare $\mathrm{elu}(x)+1$, $\mathrm{relu}(x)$, $\exp(x)$ as $\phi$.)
   *Provenance:* original (kept); independently converges with Simon Prince,
   *Understanding Deep Learning*, Problem 12.10.
1. [short-code] **A decaying state.** (Existing exercise 6, unchanged: scalar
   decay $\gamma$ in the linear-attention recurrence; verify
   parallel–recurrent agreement.)
   *Provenance:* original (kept).
1. [conceptual] **The cost of one more token.** For dense causal attention
   with a KV cache of length $t$, count the FLOPs and the bytes read from the
   cache to produce token $t{+}1$'s output; do the same for linear
   attention's recurrent form (`linear_attention_recurrent`). Deliverable:
   two closed-form per-token costs, one growing with $t$ and one constant in
   $t$, and a one-paragraph explanation of why the *steady-state* size
   comparison this section already draws (16 KiB vs. a growing cache) is a
   special case of a difference present at every single decoding step, not
   only in the limit.
   *Provenance:* adapted from Simon Prince, *Understanding Deep Learning*,
   Problem 12.7 (overlap medium; cite on adoption).

---

## chapter_attention/what-attention-computes.md — What Attention Computes

**Topic:** Attention-only Transformers factor into QK circuits (where) and OV
circuits (what) acting on a residual stream; representable functions grow
with depth (bigrams → skip-trigrams → induction); a from-scratch experiment
trains a two-layer model on repeated random tokens and verifies an induction
circuit behaviorally, in attention maps, and in the weights.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the
chapter's most technically dense file, with no defects per the prior style
review. The entire existing set converges closely with a live, actively
maintained curriculum (ARENA) built around the same Elhage et al./Olsson et
al. framework, so it is kept in full and supplemented with one exercise from
a different tradition (discrete/RASP attention) for contrast.

**External sources found:**
- ARENA (Alignment Research Engineer Accelerator) mechanistic-interpretability
  curriculum, Chapter 1: Transformer Interpretability, "Finding induction
  heads" and "Reverse-engineering induction circuits" sections, Callum
  McDougall et al. — has students locate induction heads from
  attention-pattern stripes on repeated random tokens, use `TransformerLens`
  hooks to ablate individual heads and measure the effect on loss, and
  multiply through QK and OV matrices (via a `FactoredMatrix` class) to test
  composition between a previous-token head and a candidate induction head —
  the same three checks (attention pattern, ablation, weight-level
  composition) this section's exercises 2-4 perform by hand —
  https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/
- Elhage et al., "A Mathematical Framework for Transformer Circuits," and
  Olsson et al., "In-context Learning and Induction Heads" (Anthropic,
  2021/2022) — already in this book's own further-reading list; ARENA's
  exercises are themselves built directly on this framework, so the two
  sources corroborate rather than compete.
- Sasha Rush, "Thinking Like Transformers" / Transformer Puzzles — already in
  this book's further-reading list; poses a different, discrete/RASP style of
  exercise (compose primitives like "first index of a match" and "shift by
  one position" into hard-attention programs) rather than this section's
  soft, weight-level analysis of a trained model —
  https://github.com/srush/Transformer-Puzzles
- No external exercise was found asking students to hand-construct a copying
  head with fixed (untrained) weights (exercise 1) or to diagnose a
  train/test vocabulary-split failure through the QK/OV algebra (exercise 6);
  both appear original to this book.

**Proposed problem set** (7 problems):
1. [short-code] **A copying head, by hand.** (Existing exercise 1, unchanged:
   hand-construct a skip-trigram copying head with no training.)
   *Provenance:* original (kept).
1. [short-code] **Ablating the circuit.** (Existing exercise 2, unchanged:
   zero out block-2 heads in turn, then the two strongest together.)
   *Provenance:* original (kept); independently converges with ARENA's
   induction-head ablation exercises (TransformerLens hooks).
1. [short-code] **Positional schemes and induction.** (Existing exercise 3,
   unchanged: retrain with `pos='none'` and `pos='alibi'`.)
   *Provenance:* original (kept).
1. [short-code] **K-composition.** (Existing exercise 4, unchanged: compute
   $\mathbf E\mathbf W_q^\top\mathbf W_k\mathbf W_\mathrm{OV}^{(1)}\mathbf
   E^\top$ vs. the direct path.)
   *Provenance:* original (kept); independently converges with ARENA's
   "reverse-engineering induction circuits" composition-score exercises.
1. [short-code] **A third occurrence.** (Existing exercise 5, unchanged:
   evaluate on a 3x-repeated length-21 pattern.)
   *Provenance:* original (kept).
1. [short-code] **A vocabulary split.** (Existing exercise 6, unchanged:
   train on tokens 0–47, evaluate on 48–63.)
   *Provenance:* original (kept).
1. [conceptual] **The same rule, in hard attention.** Using the RASP-style
   primitives "first index of a match" and "shift by $k$ positions" (in the
   style of Transformer Puzzles' index- and shift-based challenges), write
   pseudocode for a discrete, hard-attention program that solves the
   induction task: given the current token, find its most recent previous
   occurrence and output the token that followed it. Compare this
   formulation against the soft, two-layer QK/OV circuit of this section:
   which parts of the discrete program map onto the previous-token head and
   which onto the induction head, and what does the discrete version do
   (that the soft circuit cannot) when the current token has *two* prior
   occurrences with different successors? Deliverable: the pseudocode, an
   explicit head-to-primitive mapping, and one paragraph on the
   tie-breaking question.
   *Provenance:* inspired by Sasha Rush, Transformer Puzzles (overlap low).
