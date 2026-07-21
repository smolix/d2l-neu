# Fixlog — agent E (fast-transformer.md: JAX tab build-out)

Scope: build the JAX version of ch. 13.7 (`chapter_computational-performance/
fast-transformer.md`), per Alex's directive. Operating rules: findings §0.
Hard constraint: pytorch cells byte-identical (no recapture); jax cells added
under paired `#<id>`s; prose split with `:begin_tab:` where the stories
diverge; slides audited per framework; index.md capstone sentence updated
(that sentence only).

## Status: COMPLETE — capture-path OOM fixed and GATED THROUGH THE SCHEDULER

### Capture-path fix (2026-07-21) — R3 fp32@512 control OOM

**Symptom** (coordinator-reported): under `make run-notebooks-jax` (scheduler
env) the notebook died `RESOURCE_EXHAUSTED: Out of memory ... 15.14GiB` at the
R3 fp32@512 control. My direct `.venv-jax` runs missed it (JAX default
frac 0.75 + prealloc true packs differently).

**Faithful local repro built.** Confirmed diagnostic (temporarily added to
the jax imports cell, then removed): the scheduler runs this as a WHOLE-BOX
notebook on `[GPU 0,1,2,3]` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94`
(coordinator said ~0.95; the scheduler computes 0.94 for this box) and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (from `EXTRA_ENV_jax`). **No
env-precedence bug** — the kernel sees exactly the scheduler's frac. Repro
env: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=0,1,2,3` reproduced the OOM at the same line.

**Root cause (traced with `device.memory_stats()`), deeper than the
coordinator's c512-co-residence hypothesis:**
- `.compile()` does NOT reserve device memory (verified: `bytes_in_use`
  flat after compile); `jax.clear_caches()` does NOT shrink a
  `prealloc=false` pool (retained, never returned to OS).
- The real culprit is **R0**: an *un-jitted* train step keeps every op's
  intermediate live at once (no XLA buffer reuse), peaking ~8 GiB at
  batch 64 — and prealloc=false RETAINS that 8 GiB pool for the rest of the
  process. Even ONE un-jitted step bloats to 8 GiB (single-step peak, not
  accumulation — verified warmup=1/timed=1).
- `8 GiB (retained) + 15.14 GiB (fp32@512 contiguous) = 23.14 > 22.8` cap →
  the fp32@512 RUN can never fit once R0 has run. bf16@512 (~9 GiB) DOES fit
  (8+9<22.8), so only the fp32 *control* was the victim.
- Reordering within R3 (fp32-first, clear, bf16), and reading-plan-then-run,
  both still OOM'd — because R0's upstream bloat, not R3 co-residence, is the
  wall. Confirmed empirically across 5 probe variants.

**Fix chosen (coordinator option 1, extended upstream):** run the fp32@512
control **up front, on a clean allocator, before R0 bloats the pool**, on a
*throwaway* model/optimizer, and stash `ctl512`+`plan32_gib` for R3. The
clean-pool fp32@512 grows the pool to ~20 GiB, which then absorbs every later
rung (R0 8, R1/R2 small, bf16@512 9, R4 2, R5 per-device) by reuse. New cell
`#fast-transformer-the-subject-3` (jax-only) at the end of "The Subject",
framed as an on-theme memory lesson ("a measurement whose *scheduling* is
dictated by memory — §13.4 arriving early"). R3 cell now runs only bf16@512
and reads its plan; it uses the stashed control. **R0 stays at batch 64**, so
the ~20× jit ratio and the same-batch honesty are preserved; the full
batch-512 comparison (bf16 run + both plans + fp32 run) and the "fp32@512
slower than fp32@64 / neither rung pays alone" lesson are all intact.

**GATE PASSED (the real capture path):**
`rm _notebooks/jax/chapter_computational-performance/fast-transformer.executed`,
regenerated notebooks (`make notebooks-jax`), then the scheduler scoped to
this one notebook (`notebook_scheduler.py --frameworks jax --files
chapter_computational-performance/fast-transformer.md` with the box's detected
`D2L_GPU_*` env — scoped with `--files` instead of the unscoped
`make run-notebooks-jax` only to avoid re-running all 154 jax notebooks; same
scheduler, same per-notebook env). Result:
`=== scheduler done: 1/1 ok, 0 failed ===`, `fast-transformer.ipynb: OK
(361s) [GPU 0,1,2,3]`, exit 0. Diagnostic confirmed `frac=0.94 prealloc=false
ndev=4`. Diagnostic then removed; committed jax cells verified byte-identical
to a prior no-diagnostic run that was green at frac 0.95.

### Authoritative SCHEDULER-CAPTURE numbers (prose matches these)

fp32 control @512 (up front) 216,198 t/s, plan 15.1 GiB; R0 un-jitted 13,683;
R1 jit 254,330 (**19×**); classify 765 GFLOP / 32.2 ms / 24 TFLOP/s; naive
"bf16" 0.99× (receipt float32); R2 threaded 230,192 (**0.91×**), plans
1149/1989 MiB; R3 bf16@512 424,186 (**1.84×**), 43 TFLOP/s, fp32 control
216,198 → **bf16 pays 1.96×**, plans **8.9 / 15.1 GiB**; R4 ckpt 376,494
(**0.89×**), 1.8 GiB; R5 floors 249k/498k, k=2 199k (0.78×), k=4 300k
(1.18×); fast config k=2 516k (61%), k=4 986k (58%); cumulative R1→R3
**1.67×** (R0→R3 31×); learning 2.84 → 0.06. Wall 361s (well under budget).

**Prose reconciled to these:** R4 "about a seventh" → "about a tenth"
(0.89× = 11% cost), waterfall jax "most of the way to 2×" → "about 1.7×"
(1.67×), R3 fp32-control rate "under twenty TFLOP/s" → "around twenty
TFLOP/s, no better than the batch-64 step" (≈20 vs R1's 24). All other
hedges still hold: R1/R0 "around twenty-fold" (19×); R2 "barely moves"
(0.91×); bf16 pays "most of a factor of two" (1.96×); "fp32@512 slower than
fp32@64" (216k<254k); R5 k=2 "dip below one GPU" (0.78×). RECONCILE list
below is now satisfied against the capture.

---

## Status (original build): COMPLETE

Final verification run (v7, cells extracted from the committed .md text,
.venv-jax, all 4 GPUs, ~285 s wall): exit 0, stderr EMPTY, all asserts
pass. Cells unchanged since v7 (diff-verified); subsequent edits were
prose/slides only. PyTorch cells byte-identical vs HEAD (extract+diff);
pytorch slide deck output byte-identical vs HEAD (gen_slides diff);
`audit_outputs.py --verify-fresh` → "no stale notebooks / store is clean
and fresh" (pytorch fast-transformer manifest fresh; jax manifest does
not exist yet — to be created at capture). Lint clean; add_cell_ids
idempotent; tab blocks balanced (37/37); preprocessor round-trips.

### v7 printed values (the reconcile baseline)

R0 13,943 t/s; first compile 6.9 s; R1 254,467 (18x); classify 765
GFLOP, 32.2 ms, 24 TFLOP/s; naive "bf16" 248,725 (0.98x), receipt
float32; R2 threaded: receipt bfloat16, plans 1149 vs 1989 MiB, 234,896
(0.92x); R3 454,628 (1.94x), 46 TFLOP/s, fp32 control 193,973 (bf16
pays 2.34x), plans 8.9 vs 15.1 GiB; R4 398,204 (0.88x), plan 1.8 GiB;
R5 floors 249k/498k, measured k=2 199k (0.78x), k=4 298k (1.17x); fast
config k=2 529k (58%), k=4 1001k (55%); cumulative R1→R3 1.79x (R0→R3
33x); learning 2.85 → 0.06.

### Late finding — the rung interaction (prose rewritten to it)

fp32@512 ≈ 194–196k reproducibly (probe_ctl: clean-pool 196k, after
everything 184k, in-sequence 194k) — i.e. **batch-up alone is a
negative rung for the fp32 program** (memory wall: 15.1 GiB plan,
materialized fp32 attention buffers), while bf16 alone was flat at 64
(dispatch wall). Only the pair pays (~1.9x). R3's prose, the waterfall
reading, the summary, and the "Rungs 3–4" slide all teach this
interaction explicitly.

## What was built

Fifteen jax cells (13 paired under existing pytorch ids, 2 jax-only:
`…-rungs-each-one-measured-9` bf16-threaded, `…-10` DP fast config), tabbed
prose throughout, per-framework slides (`only=` scoping), 2 new jax
exercises, jax summary. PyTorch cells verified byte-identical vs HEAD
(extract+diff of all 14 cells). Lint clean; add_cell_ids idempotent.

## The measured JAX ladder (verified end-to-end, 4×4090 box)

Numbers from v5/v6/v7 full-sequence runs of the extracted cells
(.venv-jax, jax 0.10.2 / flax 0.12.7):

- R0 un-jitted: ~14k tokens/s (very stable, 3 runs ±0.5%).
- R1 nnx.jit whole step: ~245–255k (18×); first compile ~7 s; parity
  jit-vs-eager max |Δ| ≈ 3.6e-4.
- classify: XLA cost_analysis 765 GFLOP/step → ~23–24 TFLOP/s achieved
  (≈ a third of the tf32 roof).
- R2a naive bf16 (cast arrays only): ≈1.00× — SILENTLY still fp32
  (flax modules remember compute dtype; `Embed.dtype = param_dtype` at
  construction promotes the bf16 table back). Receipt printed.
- R2b bf16 threaded (module dtypes + post-rope cast subclass): receipt
  bfloat16, planned temp ~1.15 vs 1.99 GiB fp32 — throughput ≈0.95–1.08×:
  NULL at batch 64. Diagnosis: per-step host/dispatch floor ~32 ms pins
  every batch-64 config (R1≈R2a≈R2b); fixed-tensor probes run the same
  step in ~14–20 ms. Prose presents the null as the overhead-regime
  classification.
- R3 batch-up 512: 441–473k (1.83–1.94× over R2), 45–46 TFLOP/s; the
  fp32-at-512 control *run* lands at ~194–196k — SLOWER than fp32@64:
  batch-up alone is negative for the fp32 program (memory wall). The
  win is the bf16×batch interaction. Plans: bf16 8.9 vs fp32 15.1 GiB.
- R4 checkpoint: 0.85–0.86×, plan 8.9 → 1.8 GiB. Mirror assert exact
  (max |Δ| = 0.0).
- R5 DP (fp32 jit, per-device 64): floor ~249k (t_comm ~34 ms ≈ t_cmp);
  measured k=2: 200–240k (0.78–0.98× of ONE GPU — at/below floor);
  k=4: 300–380k (1.2–1.6×, well below floor 2.0×).
- R5 fast config (bf16, per-device 512): k=2 ~538–546k (61% weak-scaling
  eff.), k=4 ~985–1056k (56–59%).
- Learning check: bf16+512, 300 steps: smoothed loss 2.8 → 0.09, assert
  passes.
- Whole jax notebook wall time ≈ 5.5 min single process (within budget).

## Deviations from the brief (all measurement-driven)

1. **R2 is a null rung at batch 64, not a win.** Directive expected bf16 to
   pay at R2; measured ≈1.0×. Root causes found and TAUGHT in the section:
   (a) naive array cast silently stays fp32 (flax module compute dtype +
   rope fp32-trig promotion — kernel refuses mixed dtypes loudly);
   (b) after the correct threading, batch-64 steps are pinned at a ~32 ms
   per-step host/dispatch floor, so halved device work doesn't move
   end-to-end. bf16's payoff lands inside R3 (fp32 control isolates it).
2. **No flash-attention rung in the section.** Explored (works:
   FlashGPT subclass, implementation='cudnn', needs bf16) but measured
   ~1.0× at context 128 — moved to a new exercise (context 128 vs 1024).
3. **R5 baseline uses the jitted step** (not eager) — the honest jax
   baseline; prediction uses tput1 and beta=4.5 GB/s (13.5's psum).
4. **DP mesh needs `axis_types=(jax.sharding.AxisType.Auto,)`** — at
   jax 0.10.2, `jax.make_mesh((k,),('data',))` traces the embedding
   gather under explicit sharding types and raises ShardingTypeError
   (13.6's ResNet convs don't hit this; a GPT does). One cell line + one
   prose parenthetical. NOTE for other chapters: any embedding-bearing
   model under 13.6's recipe needs this.
5. **jax data stream = on-device batcher**, not the tf.data loader: the
   stock loader costs ~1+ ms host per batch + per-epoch rebuild and taxed
   the fast rungs ~25–30% end-to-end (measured); the 5 MB corpus is staged
   on device once and shuffled there. Taught in the trap paragraph.
   Also: never re-time one cached batch (flatters by double-digit % via
   cache reuse) — taught in the metric note.
6. jax R3 reuses `data` at batch_size=512 (no second TimeMachine build —
   the jnp windows build costs ~50 s; pytorch tab keeps its own).

## RECONCILE AFTER CAPTURE (cell → prose claims to re-check)

- `…-rung-0-baseline-profiled@jax`: "~14k"; R1/R0 ratio "(18x)" — prose
  says "roughly twenty-fold/around twenty-fold" (slides "~20×") — holds
  for 17–18×; re-check print.
- `…-rungs-each-one-measured-1@jax`: first compile "~7 s" → prose
  "several seconds"; parity tolerance atol=1e-2 (margin ~30×).
- `…-measured-5@jax`: "765 GFLOP", "23–24 TFLOP/s" → prose "about a third
  of the card's ~83 tf32 TFLOP/s" (29% — "about a third" ok; re-check).
- `…-measured-2@jax` (naive): ratio ~0.98–1.01× → prose "Flat".
- `…-measured-9@jax`: plan "1149 vs 1989 MiB" → prose "drop by about
  two-fifths" (slide "~40%"); ratio 0.92–0.95× at sequence position →
  prose "barely moves/almost nothing"; WOBBLE acknowledged in prose.
  If capture prints >1.15×, soften "still barely moves".
- `…-measured-3@jax`: 1.83–1.94× over R2 ("most of a factor of two");
  45–46 TFLOP/s ("roughly doubles" vs classify's 23–24); fp32 control
  ~194–196k, i.e. SLOWER than fp32@64 ("slower than fp32 at batch 64",
  "under twenty TFLOP/s" = 18.1–18.3); bf16-vs-fp32@512 2.10–2.34×;
  plans 15.1/8.9 GiB ("roughly 15 GiB"/"around nine").
- `…-measured-4@jax`: 0.85–0.88× ("about a seventh"), plan 1.8 GiB
  ("several-fold" cut from 8.9).
- `…-measured-8@jax`: k=2 at/below floor; k=4 well below. Prose written
  to cover 0.78–0.98× (k=2) and 1.17–1.6× (k=4). (Current-design runs:
  199–200k and 298–300k — tight.)
- `…-measured-10@jax`: efficiencies 55–61% → prose "markedly higher".
- waterfall print: "cumulative R1→R3" 1.79× → prose "most of the way
  to 2×"; R0→R3 33× → "thirty-odd-fold".
- learning: 2.8 → 0.09.
- Metric-note prose: "flatters ... by double-digit percent (we measured
  that too)" — scratch-verified (fixed 604k vs cycled 441k; also
  15–25% in other runs).
- Trap prose: "taxed the fastest rungs below by roughly a quarter" —
  scratch-verified (tf.data 326k vs staged 422–450k for the bf16 step;
  v5-vs-v6 R3 441 vs 451 within noise).

## Open risks

- Batch-64 numbers (R1, R2a, R2b) carry a host-state sensitivity: at
  fixed sequence position they reproduced within ±8% across three runs,
  but measurements at other positions in a warm process ran up to ~1.5×
  faster. Prose hedges accordingly and even teaches the wobble. If a
  future capture lands R2b ≈1.2–1.4×, the R2 narration ("barely moves")
  needs softening — flagged above.
- The nnx.jit per-call graph-walk overhead is part of the ~32 ms floor;
  a future flax may shrink it and lift batch-64 numbers (direction:
  ratios compress toward 1.0 for R2, R3 gain shrinks). All prose hedged
  ("in our runs", "most of a factor of two").
- `NCCL_LOCAL_REGISTER=0` kept in imports (13.6 precedent); no warnings
  observed in any run (stderr empty end-to-end).
- capture must run the jax notebook via the scheduler as usual; runtime
  ~5.5 min + pytorch unchanged (no pytorch re-execution needed or wanted).
