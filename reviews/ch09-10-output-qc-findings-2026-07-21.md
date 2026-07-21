# ch9 (Optimization) + ch10 (Attention) — review findings (2026-07-21)

Rendered-output + content QC of the two Advanced-part chapters that preceded the
11–13 work. Both were **content-redesigned** earlier (07-13→07-19:
`ch09-optimization-redesign`, `ch10-structure-review`,
`attention-transformers-review-and-proposal`) but never received the
rendered-output QC pass that 11–13 got this session (`ch11/12/13-output-qc`,
coverage/reconcile QC, `comp-perf-review-findings`). This file is that pass.

**Method.** Two external expert reviews (`optimization_feedback.md`,
`attention_feedback.md`) exist — the ch9/ch10 analogs of `performance_feedback.md`.
They review the *post-redesign* chapters but from a working tree that was edited
afterwards, so every item was **re-triaged against the current committed source**
(verdicts: CONFIRMED / FIXED / PARTIAL / DISAGREE). Our output-QC lens (figure
house-style, crossref, table captions, experimental-precision policy, stale
JAX/NNX API, ANSI leaks, wide output, `$`-before-digit) is layered on top.

**Already clean (both chapters).** Coverage parity = PyTorch+JAX only (no
TF/MXNet trees). HTML crossrefs resolve (0 `?@`/leaked `@sec-` markers). The
house-style `mdl-opt-*` / `mdl-attention-*` figures are high quality (visually
inspected — no label↔line or numeric defects).

---

## ch10 (Attention) — triage

Reviewer's 7 top items: **3 fully fixed** by the redesign, **3 partial** (the
dangerous parts corrected; guards/labels/disclaimers outstanding), **2 live and
Colab-breaking**. Many section-level items were also already redesigned away.

### REJECTED — `nnx.view` is NOT a bug (feedback + first-pass triage were wrong)
The external feedback claimed "the current pinned Flax/NNX API no longer provides
`nnx.view`" and called it "the same class of Colab failure observed elsewhere";
the first triage pass accepted this. **It is false.** Verified: `flax.nnx.view`
exists in the pinned **flax 0.12.7** (`flax.nnx.module.view`), is the standard
house idiom used throughout **`d2l/jax.py` itself** (lines 411, 414, 1126, 2339,
3124, 3155) and across **70 source files in 10+ chapters** (builders-guide,
convolutional-modern, GANs, NLP, recurrent-*, computer-vision, …), and ch10's JAX
outputs executed clean (no error/deprecation). The 4 ch10 calls
(`attention-scoring.md:498`, `multihead-attention.md:372,442,471`) are correct and
consistent with the library. **No action.** (Root cause of the false positive: the
triage was primed with the feedback's unverified claim; an independent API check
overturned it.)

### Live runnability item (verify against book convention before acting)
- **`attention-at-scale.md` GPU-only — CONFIRMED present, severity TBD.** The
  PyTorch `peak_memory`/`wall_clock` helpers call `torch.cuda.synchronize/
  reset_peak_memory_stats/max_memory_allocated` unconditionally (`:173-178,
  217-221`), plus `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` + `float16`
  (`:450,459`); JAX uses `implementation='cudnn'` (`:482`). This runs on Colab's
  **GPU** runtime (the default for DL notebooks) but fails on a **CPU** runtime.
  Whether that's a defect depends on the book's convention for scale/perf
  notebooks (see convention check below) — if other GPU-scale notebooks (e.g.
  ch13 comp-perf) are also unguarded GPU-only, at-scale is consistent, and a
  capability-gate is a nice-to-have graceful-skip, not a critical bug. If the book
  generally guards CPU, at-scale is an inconsistency worth fixing.
  **Convention check result:** the book's GPU-scale/perf notebooks are
  GPU-oriented and unguarded — the ch13 comp-perf chapter we rebuilt, reviewed,
  and shipped this cycle has `multi-gpu-practice.md` (0 `is_available` guards),
  `fast-transformer.md`, `memory-precision.md`, plus ch11 `scaling-laws.md`, ch12
  `matrix-state.md`/`ssm.md`, all making the same unconditional `torch.cuda.*`
  calls. So at-scale is **consistent with shipped precedent, not a ch10-specific
  defect.** Disposition: the feedback's graceful-CPU-skip idea is reasonable *as a
  book-wide policy*, but applying it to at-scale alone would make it inconsistent
  with the rest. Treat as a book-wide convention question, out of scope for a
  ch10 fix. (Colab defaults to a GPU runtime for these notebooks anyway.)

### Partial (dangerous part handled; residue outstanding)
- **Masked-softmax safety (C1) — PARTIAL.** `attention-scoring.md:214-220,365-372`,
  exercise `:645`. Prose now thoroughly *warns* ("a uniform average over invalid
  values is still garbage, so callers must guarantee ≥1 valid key"), but the
  reviewer's concrete ask — an `assert` of ≥1 valid key (or zero-out fully-masked
  rows) — is not in the `masked_softmax` code. Add the guard.
- **Cost-table like-for-like (C5) — PARTIAL.** The "three orders of magnitude"
  overclaim is FIXED (now "one to two orders past the crossover", worked example
  d=4096,n=131072 → 16×). Remaining: comparison table (`attention-at-scale.md:86-90`)
  still lists self-attention as `O(n²d)` while charging CNN/RNN their `d²`
  transforms; the QKVO `8nd²` reappears 30 lines later. Label the row "(mixing
  only)" or add the full `Θ(nd²+n²d)`.
- **Single-head "provably" scope (H6) — PARTIAL.** Assumptions are stated locally
  (position-only keys, isotropic Gaussian values, single mixture, linear readout)
  and phrasing is scoped; missing one sentence that the separation does *not*
  extend to deeper / content-dependent / residual one-head models
  (`multihead-attention.md:11`, summary `:529`, index `:46`).

### Fixed by the redesign — no action
Gaussian→dot-product LayerNorm claim (C2, gone; framed as a modeling choice with
the Gaussian as the equal-norm special case), tied-embedding bigram (C3, now
"Gram matrix, rank ≤ d, not an arbitrary |V|×|V| table"), windowed path length
(C4, now "gives up the O(1) path length … keeping O(n/w)"), deprecated Flax
`.value` (now `variable[...]`, JAX outputs warning-free), JAX chunked
`V.reshape(...d)`→`V.shape[-1]` + tail/assert, windowed `n%w`/dtype asserts, RoPE
"every pair carries content", ALiBi power-of-two restriction, "only
sequence-mixing machinery", reading-vs-generation "entire difference", "heads are
free" (hedged), "folds heads into the batch" (hedged), diagonal-argmax now
"in some runs" (recaptured 0.94 PT / 0.77 JAX), "training longer widens the gap"
(removed), permutation-equivariance bag-of-tokens overclaim.

### Lower-severity, still open
- softmax "gradient never vanishes identically" (`queries-keys-values.md:99`,
  `attention-scoring.md:87`) — imprecise; the softmax Jacobian always has 𝟙 in
  its null space. Say "the Jacobian is never the zero matrix."
- entropy < 0.2 ⇒ "weight > 0.9 on a single key" (`attention-scoring.md:143`) —
  the code measures entropy, never mean-max-probability; measure it or drop the
  0.9.
- "gradients stop flowing to everything upstream of the scores" (`:86`) — too
  broad (V, output proj, residual still carry gradient).
- benchmark provenance (hw/dtype/backend) not adjacent to timing plots
  (`attention-at-scale.md:447-489,858-909`); perplexity plot lacks a training-
  length marker (`positional-information.md:633`).
- index/intro undated universals ("analogy is exact except…", "essentially every
  current model", "the variants that survived") — part of the writing pass.

### Figures (our lens)
Three non-house-style legacy SVGs remain amid seven house-style
`mdl-attention-*`: `qkv.svg` (`queries-keys-values.md:105` + slide),
`multi-head-attention.svg` (`multihead-attention.md:181` + slide),
`cnn-rnn-self-attention.svg` (`attention-at-scale.md:64` + slide). All three are
static includes with **no drawing code** (nothing to strip from notebooks) —
regenerate in house style. `qkv.svg` is the weakest (generic box-and-arrow); the
feedback wants it replaced with a worked soft-lookup (one query, three keys,
scores, normalized weights, value mixture). `cnn-rnn-self-attention.svg` is worth
keeping as a historical schematic but its cost table must be made consistent (C5).

---

## ch9 (Optimization) — triage

**Headline: the feedback is substantially stale — all twelve P0 items are already
FIXED.** The chapter `.md` files are git-clean and history includes
`d502d788 "ch9: apply Codex-review fix pass (text+code)"`; the reviewer's working
tree predates it. Each P0 was re-checked against current `file:line` with the math
verified directly:

1. Muon steepest-descent — **FIXED** `muon.md:63-74`: "returns *normalized*
   gradient descent −ηg/‖g‖₂," regularized model given, direction/step separated.
   (So the `mdl-opt-norm-balls` ℓ₂ panel showing −g/‖g‖₂ is now *consistent*, not a
   defect.)
2. Scaling single-example factorization — **FIXED** `scaling.md:224-248` ("For a
   single example… sign does not factorize for a minibatch").
3. muP output bias — **FIXED** `scaling.md:306` (`fc_out(features(X)/m)` = divides
   features, bias unscaled).
4. Batch-size Adam scaling — **FIXED** `batch-size.md:551-559` (full η'=√kη,
   β'ᵢ=1−k(1−βᵢ), ε'=ε/√k, admissible range).
5. MS-vs-RMS noise radius — **FIXED** (consistently "squared radius ∝ η").
6. Robbins–Monro α=½ — **FIXED** `sgd.md:200-203` (α∈(½,1], "edge of the window").
7. AdamW decomposition — **FIXED** `adamw.md:65-77` ("schematically… both moments
   built from the penalized gradient").
8. AdamW "restores semantics" — **FIXED** (now "restores SGD semantics"; three
   ideas separated).
9. Newton–Schulz rank — **FIXED** `muon.md:183-188` ("rank-deficient stays
   rank-deficient").
10. Muon scratch-vs-library momentum — **FIXED** (both Nesterov).
11. Practice clipping+Adam cap — **FIXED** `practice.md:227-236` (cap removed,
    ≈3 transient bound; clip fraction printed).
12. Practice EMA+BatchNorm — **FIXED (claim-level)** `practice.md:347-355` (demo
    still averages BN stats, labeled approximate — minor residual).

### Genuine remaining ch9 work (P1/P2 + our lens)
- **Figure straggler — "code teaches, it does not draw" (our lens, also flagged by
  feedback).** `optimization-intro.md` (Landscapes) draws its illustrative figures
  **inline with matplotlib** — the 3-D saddle wireframe (`:152-166`, worst), plus
  local-minima (`:120`), x³ (`:141`), tanh (`:201`), risk/empirical-risk (`:69`) —
  while the rest of ch9 uses pre-generated house-style `mdl-opt-*` SVGs (and a
  generator `tools/gen_opt_figures.py` exists). One-style-per-chapter inconsistency;
  reproduce these as SVGs. (`gd`/`sgd`/`adam` trajectory plots and the `adamw`
  heatmap compute results — legitimate, keep.)
- **Landscapes math precision (P1)** — second-order test (`:172-176`) omits the
  PSD/zero-eigenvalue **inconclusive** case; x³ "saddle" not distinguished from a
  stationary inflection (`:137`); the coin-flip Hessian-eigenvalue model persists
  (`:177`, hedged).
- **Schedules methodology (P1)** — PyTorch decay comparisons are **unseeded**
  (JAX side is seeded) and **selected on the test set** (no val split); Animator
  puts **train loss and accuracy on one shared y-axis** (`lr-scheduler.md:120`).
  Heavy hedging mitigates but doesn't fix.
- **AdamW scale-invariance (P1)** — "most of a transformer is scale-invariant"
  (`adamw.md:444-446`) imprecise for pre-LN residual (residual stream bypasses the
  norm).
- **Tone/P2** — "first credible challenger" (`index.md:44`, `muon.md:10`; Shampoo
  predates); gradient-accumulation "arithmetically identical" lacks BN/dropout/
  clipping caveats (`minibatch-sgd.md:277`); "frameworks apply *exactly* the
  equation" (`adam.md:340`, `adamw.md:167`); `d2l.train_lm` reused for CNNs
  (misleading name); no dated Hyperball note. (Absolute-word audit is *largely
  already clean* — "honest" down to 3, "universal/settled" now mostly used to deny
  universality; experimental-precision policy is compliant — margins as ranges,
  methodology limits disclosed not overclaimed.)
- **Enhancement requests NOT done** (P1/P2, optional): scratch-vs-library one-step
  conformance tests, an update-diagnostics dashboard, per-layer Δh/update-ratio muP
  plots, two-batch-estimator error bars. These add author-side test infrastructure;
  defer unless wanted.

---

## Consolidated fix-priority shortlist (both chapters)

Both feedback files reviewed **pre-fix-pass** working trees; the redesign +
Codex fix pass already resolved the alarming material (all ch9 P0 math; ch10
C2/C3/C4). **Nothing Colab-breaking remains** — the ch10 `nnx.view` "bug" is a
false positive, and at-scale GPU-only matches shipped convention. Genuine
remaining work, ordered:

1. **Trivial source hygiene.** Delete the stray `%load_ext d2lbook.tab` /
   `interact_select` cells: `adam.md:1-4`, `adamw.md:1-4` (stripped by the
   preprocessor, so no render impact; but Colab-export + consistency).
2. **Caption defect.** Add caption + `:label:` to the muP learning-rate table
   `scaling.md:277` (renders unnumbered now) — fold into the muP-table touch.
3. **ch10 content partials.** Masked-softmax code guard (assert ≥1 valid key,
   `attention-scoring.md:224`); cost-table like-for-like label/`Θ(nd²+n²d)` row
   (`attention-at-scale.md:86`); one-sentence single-head scope disclaimer
   (`multihead-attention.md:11`).
4. **ch9 Landscapes figures → house style.** Reproduce the inline-drawn
   `optimization-intro.md` figures (esp. the 3-D saddle) as `mdl-opt-*` SVGs.
5. **ch9 P1 precision/methodology.** Landscapes second-order-test inconclusive
   case + x³/saddle convention; seed the schedule comparisons and split
   loss/accuracy axes; qualify AdamW pre-LN scale-invariance.
6. **ch10 lower-severity.** Softmax "Jacobian never zero-matrix" wording; the
   entropy→0.9 max-weight claim (measure or drop); "gradients stop flowing to
   everything upstream" scope; benchmark provenance + perplexity training-length
   marker.
7. **Figures — 3 ch10 stragglers.** Regenerate `qkv.svg` (→ worked soft-lookup),
   `multi-head-attention.svg`, `cnn-rnn-self-attention.svg` in house style.
8. **Absolute-word / dating pass** across prose **and** slides (both chapters;
   mostly ch10 + a few ch9 residuals: "first credible challenger", undated
   frontier, index universals).

Not a fix: ch10 `nnx.view` (working house idiom); at-scale CPU-guarding (book-wide
convention question, not ch10-specific).

---

## Output-QC cross-cutting scan

- **Table caption/label — one real defect.** `scaling.md:277-281` (the muP
  learning-rate-rules table) has **no `:Caption` and no `:label:`** — it renders
  as a bare, unnumbered `<table>` (confirmed: `_book/chapter_optimization/
  scaling.html:1984` has no `<figcaption>` / table number, unlike the correctly
  labelled `practice.html` table). It's never `:numref:`-referenced (prose calls
  it "the table" at `scaling.md:496`), so it's a pure omission, not a broken ref.
  Fix alongside the muP P0 (the table content changes anyway). The other 3 book
  tables (`muon.md:151` `tab_muon_norms`, `practice.md:46` `tab_practice_recipes`,
  `attention-at-scale.md:83` `tab_cnn-rnn-attn`) are correctly captioned
  caption→label→table. (Slide-deck table reproductions carry no label by
  design — not flagged.)
- **Stray preprocessor cell — `adam.md:2-3` AND `adamw.md:2-3`** carry a leftover
  `%load_ext d2lbook.tab` / `tab.interact_select('pytorch','jax')` cell. The
  preprocessor strips it (`d2l_preprocess.py:97-98`), so it does **not** leak into
  the `.qmd`/HTML — but it's pre-rewrite source cruft the other 11 ch9 files (and
  all of ch10, ch12, ch13) don't have. Delete for consistency. (Not a render bug.)
- **Clean** (verified, no action): crossref leakage (0 `?@`/leaked directives in
  either chapter's HTML); duplicate/undefined labels (43 labels, no dups vs 915
  repo-wide; all `:numref:`/`:eqref:` resolve); ANSI escapes in outputs (none —
  ch9/10 unlike ch13's JAX sharding viz); **wide output (widest line 82 chars —
  well under the ~110 wrap threshold; no profiler tables or list dumps, so ch9/10
  need nothing from the verbatim-wrap patch)**; `$`-before-digit PDF tripwire
  (0 genuine closing-`$`-before-digit); `jax.tree_map` (correctly uses modern
  `jax.tree.map`); bare `%%tab all` (none — all properly `%%tab pytorch/jax`).
