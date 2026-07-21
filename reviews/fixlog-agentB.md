# Fixlog — agent B (memory-precision.md, multiple-gpus.md, fig_ring_allreduce)

Scope: §2.5 (MP-1..MP-6), §2.6 (MG-1..MG-7) of
`reviews/comp-perf-review-findings-2026-07-20.md`, plus the §3 calibration
pass on these two files. GPUs 2,3 (CUDA_VISIBLE_DEVICES=2,3). All code
verified by executing cells extracted verbatim from the .md files in the
right venvs (scratch scripts under scratchpad/agentB/).

## Status — all findings DONE

- [x] **MP-1** — jax mixed-precision cell now times three configurations:
  `jax.default_matmul_precision('highest')` (true fp32), the default, and
  bf16. On-box verification: JAX's default dot precision is byte-identical
  to `'high'` (tf32-class); `'highest'` differs (max |Δ| 0.072 on a
  2048-dot). Measured 20.2 / 13.0 / 6.7 ms → tf32→bf16 = 1.93× (pytorch tab
  1.5×). Shared prose says "about one and a half to two times,
  framework- and shape-dependent"; intro names the opposite framework
  defaults (PyTorch tf32 off, JAX tf32-class on); one clause states
  `default_matmul_precision` controls dot *compute*, not storage dtype.
  Summary bullet + slide updated. "Both report the same physics" →
  "same design point" (§3).
- [x] **MP-2** — checkpointing cells now completion-time fwd+bwd via
  `d2l.Benchmark` in both tabs. (a) pytorch grads (params + X) cleared
  between the two runs — measured peaks in notebook context:
  4318→2306 MB (saving 2012 MB; the old 2507 checkpointed figure carried
  ~200 MB of stale grads); (b) jax init split into 32 keys (W1≠W2 now);
  (c) "never for speed (it costs speed)" → "normally costs time; its
  purpose is memory (rare cache-effect exceptions...)". Timings:
  pytorch 53.7→62.7 ms (+17%), jax 50.4→58.7 ms (+16%); prose/slide quote
  "about 15–20% more time" and "~2 GB of activations released / peak cut
  by half or more" (jax plan 3230→1216 MB, saving 2014 MB — the 2 GB
  matches across tabs).
- [x] **MP-3** — cell prints `max_memory_reserved` alongside allocated;
  new paragraph reconciles 403 vs 524 MB (activations ~13 MB, autograd/
  cuBLAS workspaces, fused-optimizer temporaries ~ one gradient set —
  verified: peak pre-step 224 MB vs 524 after `opt.step()` — allocator
  rounding), names allocated-vs-reserved once with :numref:`sec_use_gpu`
  cross-ref, frames 16P as this configuration's (fp32+Adam) floor, adds
  the workload-dependence clause for activations.
- [x] **MP-4** — equivalence conditions sentence added (additive loss,
  equal micro-batches/weighted mean, RNG/dropout, no BatchNorm-style
  batch-coupled state, clip/step at global boundary); "same work in the
  same time" → "identical arithmetic as more, smaller, lower-intensity
  GEMM launches — usually slightly slower"; Ladder list + slide updated
  (compile "usually free", bf16 "~1.5–2×", accumulate "≈ same wall-clock,
  never faster").
- [x] **MP-5** — exercise 4 re-posed: construct underflow deliberately
  (scaled loss), verify fp16 grads flush to zero while bf16 survives,
  then GradScaler; asks why full-run failure is workload-dependent.
- [x] **MP-6** — "what stand" → "what stands"; added "don't compare the
  two frameworks' digits head-to-head: a compiler's plan is not an
  allocator's high-water mark".
- [x] **MG-1** — THE deviation to know about (see Deviations): plain
  `jax.lax.pmean(grads)` per the spec is STILL wrong at jax 0.10.2.
  Root cause found via jaxpr: under `shard_map` with `check_vma=True`,
  differentiating *replicated* (P()) params makes AD's transpose insert
  `psum_invariant` itself — `jax.grad` returns already-summed grads. The
  committed cell's explicit `psum` therefore double-counted (4× mean at
  k=2, not the review's assumed 2×), and `pmean` on top yields sum·2/2 =
  still the sum. Fix: `jax.lax.pcast(params, 'data', to='varying')`
  before `jax.grad` (grads stay device-local) then `jax.lax.pmean` — the
  collective stays explicit and correct ("the allreduce, in one line"
  survives). Prose explains the transpose-inserts-the-sum subtlety.
  One-step k=1-vs-k=2 invariant cell added in BOTH tabs (new cell id
  `multiple-gpus-data-parallelism-by-hand-7`), run on GPUs 2,3:
  pytorch max Δ-diff 5.8e-10, jax 1.9e-9 (magnitudes ~1e-2) — both pass;
  jax delta was verified equal to a single-device full-batch reference
  gradient step (2.3e-9). jax `train()` now prints test accuracy
  (jitted lenet over the val loader, params pulled to device 0),
  mirroring pytorch.
- [x] **MG-2** — editing artifact rewritten: "models with high compute
  per byte of gradient traffic, large per-device batches, and fast links
  all push in your favor".
- [x] **MG-3** — "independent of k" replaced everywhere (prose, eq
  follow-up, summary, slide, figure text) with nearly-constant /
  bounded-by-2N phrasing, exact factor kept. Convention sentence added
  before the measurement (bytes per device per allreduce; NCCL busbw is a
  different normalization). Optional α sentence added (per-hop latency,
  why trees win small messages). Figure regenerated; generator verified
  byte-idempotent (identical md5 across two runs); PNG visually
  inspected — no overlaps, layout intact.
- [x] **MG-4** — post-measurement paragraph split with :begin_tab:.
  pytorch: "roughly twenty GB/s" star copy (measured 18.6 GB/s this box,
  store 20.7) + NCCL busbw aside. jax: "a few GB/s" psum (measured 5.5,
  store 4.5), explains XLA→NCCL, host-staged, lands below the naive copy.
  Shared lead-in names what each tab measures. "read the cost model
  honestly" → "against what we measured".
- [x] **MG-5** — (a) "more arithmetic and more *aggregate* memory";
  (b) intro inverted honestly (fast fabrics shrink a constant, the
  accounting matters more at frontier scale; our slow fabric makes it
  loud); (c) pytorch per-batch `torch.cuda.synchronize()` moved to the
  epoch boundary, syncing *every* device before `timer.stop()`
  (completion-honest); measured 1.67→2.09 s/epoch at k=1→2 (still
  slower on 2 GPUs — story intact); "per-step synchronization" in the
  no-speedup paragraph updated to "per-step dispatch"; (d) "we elide
  them here" clause deleted (loop is shown); (e) the XLA "Delay kernel
  timed out" stderr warning fires during first compilation/autotune —
  the new invariant cell compiles train_step for k=1 and k=2 at exactly
  the training shapes *before* the train cells, so the timed cells'
  outputs render clean; the warning, if it appears at all, lands in the
  invariant cell (it is contention-dependent; on quiet capture runs it
  may not appear).
- [x] **MG-6** — strong-vs-weak scaling definitions added in §Accounting
  right after eq_dp_cost (strong = fixed global B, what the model
  states; weak = fixed per-device batch), with pointer that 13.6's sweep
  is weak-scaling and a :numref:`sec_batch_size` trajectory caveat.
  Summary bullet carries the two names.
- [x] **MG-7** — "rings and trees, chosen per message size and topology,
  alongside hardware-specific schemes"; parameter-server: recsys kept as
  the book-relevant example, exclusivity dropped ("wherever sparse or
  asynchronous state dominates"). Summary + Lineage slide updated.
- [x] §3 calibration on my two files: "honest" now appears once
  (load-bearing, 13.5 intro); "physics" zero.
- [x] `add_cell_ids` run (2 new ids assigned, no existing id touched);
  `tools/lint_source.py` clean on both files; `#@save` bodies
  (split_batch pt+jax) byte-identical (verified via diff grep).

## Verification summary (all on CUDA_VISIBLE_DEVICES=2,3)

- Full per-framework verbatim cell-extraction runs of both files pass:
  memory-precision {pytorch, jax}, multiple-gpus {pytorch, jax}.
- multiple-gpus pytorch: invariant 5.82e-10; train 1.67/2.09 s/epoch
  (acc .79/.79); allreduce 18.59 GB/s over 28.9 ms.
- multiple-gpus jax: invariant 1.86e-09; train 0.54/0.92 s/epoch
  (acc .80/.79); psum 5.50 GB/s over 97.5 ms.
- figure: mdl-perf-ring-allreduce.svg regenerated, idempotent, inspected.

## RECONCILE AFTER CAPTURE

Re-check prose numbers against blessed outputs for these cells (both
frameworks unless noted):

- `memory-precision-measuring-memory-1` (pt): now prints reserved too
  (524 alloc / 547 reserved here); prose says "overshoots by roughly 30%".
- `memory-precision-mixed-precision`: three timings both tabs; prose
  "one and a half to two times", jax default = tf32-class claim.
- `memory-precision-activation-checkpointing`: new peaks (pt 4318→2306;
  jax temp 3230→1216) + new fwd+bwd timings (+16–17%); prose "~2 GB",
  "half or more", "15–20% longer". NOTE pt peaks include ~600 MB of live
  tensors left by the mixed-precision cell — context-dependent; if cell
  order ever changes, re-check.
- `multiple-gpus-data-parallelism-by-hand-7` (NEW, both tabs): prose says
  "around 1e-9 against magnitudes near 1e-2".
- `multiple-gpus-data-parallelism-by-hand-5` / `-6`: timing semantics
  changed (epoch-boundary sync; jax adds accuracy line). Prose is
  qualitative only ("outright slower in both frameworks" — held at
  1.67→2.09 and 0.54→0.92 here; if capture ever shows 2-GPU ≥ parity,
  soften to "no faster").
- `multiple-gpus-the-accounting`: prose "roughly twenty GB/s" (pt tab) /
  "a few GB/s" (jax tab); measured 18.6 / 5.5 here vs store 20.7 / 4.5.

## Deviations from the spec

1. **MG-1 implementation differs from the spec's literal prescription.**
   Spec: replace psum with `jax.lax.pmean(grads, 'data')`. At jax 0.10.2
   that still yields the SUM (AD already psums grads of replicated params
   via `psum_invariant`; pmean-on-top = sum·k/k). The invariant the spec
   mandated caught this. Implemented instead:
   `pcast(params, 'data', to='varying')` → `jax.grad` → `pmean`, which
   passes the invariant exactly and keeps the collective explicit.
   (`jax.lax.pvary` is removed at 0.10; `pcast(..., to='varying')` is its
   replacement.) Also note the review §1 table's "effective lr = k·lr"
   diagnosis understates the committed bug: explicit psum on top of the
   implicit one gave 2k·lr-equivalent gradients at k=2.
2. **MP-2 prose numbers**: checkpointed peak changed more than expected
   because clearing grads removed ~200 MB from the second run's baseline;
   ratio quoted as "half or more" + absolute "~2 GB released" (stable and
   identical across tabs) instead of a bare ratio.
3. `make figures` on this box also rewrites 8 math-appendix SVGs
   (mdl-cal-/mdl-la-/mdl-opt-/mdl-clf-*) with real path differences —
   pre-existing generator/environment drift, NOT from my change. I
   reverted those 8 via `git checkout --`; only
   `img/mdl-perf-ring-allreduce.svg` remains modified. The perf generator
   itself is byte-idempotent (md5-verified). **Central pass beware**: the
   §7 "make figures idempotence" gate will trip on those unrelated
   figures on this box.
4. Benign NCCL warning on this GPU pair ("Failed to register GPU memory
   with clique... Cuda failure 500") appears on stderr for 2-GPU jax
   runs; collectives verified numerically correct. Environmental, not a
   notebook defect; may or may not appear in capture (whole-box).

## Not touched (other agents' scope)

compilation.md, hardware.md, index.md, performance-model.md diffs in the
worktree are agent A's. multi-gpu-practice.md / fast-transformer.md are
agent C's. legacy-multigpu-lib.md untouched.
