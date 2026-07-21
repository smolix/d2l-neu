# Fixlog — agent C (multi-gpu-practice.md, fast-transformer.md)

Scope: §2.7 (MGP-1..MGP-7), §2.8 (FT-1..FT-8) of
`reviews/comp-perf-review-findings-2026-07-20.md`, plus §3 calibration on
these two files. GPUs 0,1 (CUDA_VISIBLE_DEVICES=0,1). **ALL FINDINGS
IMPLEMENTED and verified end-to-end at k∈{1,2}; status COMPLETE** (both
files: full extracted cell sequences executed in the right venvs, zero
warnings, all asserts pass; lint clean; add_cell_ids idempotent; #@save
bodies byte-identical vs HEAD; nothing outside scope touched).

## MAJOR DISCOVERY (13.7) — read before reconciling

The captured story "R1 compile buys ~4%" was a **measurement artifact**,
and fixing it rewrites the section's ratios:

1. **Ragged-batch retrace inside the timing window.** TimeMachine's final
   train batch is 16 (10000 % 64): a new shape → dynamo retrace lands
   inside R1/R2's measured windows. Fix: streams drop the ragged tail
   (`if X.shape[0] == dm.batch_size`); `unique_graphs: 1` confirms.
2. **torch.profiler's CUPTI instrumentation outlives its cell** — for the
   whole process. Eager reads ~264k tokens/s before any profiling, ~221k
   after (permanently; the store's R0 = 221455 matches the taxed value to
   0.1%). Launch-bound configs pay ~16%, batch-512 configs ~0% → the
   store's per-rung ratios were distorted. Fix (verified, incl. set after
   torch import, repeat-profiling still works): `TEARDOWN_CUPTI=1` env in
   the imports cell.

Clean steady-state (multiple readings, ±0.2%, GPUs idle, clocks pinned
2715 MHz): R0 264k; R1 346k (**1.31×**); R2 474–495k (**~1.4×**); R3
622–623k (**~1.3×**, peak 8.6 GiB; fp32-512 control 16.6 GiB); R4 545k
(0.88×, 3.1 GiB); **cumulative R0→R3 2.36×** (the store's 2.80× divided
by a CUPTI-taxed baseline). Section prose/summary/slides rewritten to the
measured truth + a "profile your experiment first" trap paragraph, new
forensics exercise 6. Note: today's thermal claims in the old prose (idle
200 MHz, boost-then-throttle, "twice as fast") do NOT reproduce (clocks
pinned 2715 MHz idle→load, 42–47 °C) — thermal paragraph replaced by a
hedged warmup-discipline paragraph (FT-8a).

## Findings — all done

- **MGP-1 done** — parent pre-downloads Fashion-MNIST quietly
  (redirect_stdout+stderr); `_ =` write_text; sidecar + rank JSONs under
  `ddp_scratch/`; stale `rank*.json` unlinked per launch; context-managed
  file IO (notebook + scripts).
- **MGP-2 done** — weak-scaling named where the sweep is introduced
  (+ trajectory sentence → sec_batch_size); print says "% weak-scaling
  efficiency"; prose quotes 88%/82% as the cell prints; ADDED
  strong-scaling sweep (global 512, per-rank 512/k, same ks guard) with
  its own cell + combined plot; prose holds at k=2 (conventions nearly
  coincide there — measured weak 88% / strong 89% — divergence framed as
  the k=4 thing to watch).
- **MGP-3 done** — `comm_probe.py` sidecar (no_sync vs synced, 30 timed
  steps, barrier-aligned, `device_id=` in init_process_group so the
  barrier warning is gone): measured ~15–16 ms/step at k=2 vs printed
  2N/β prediction 20 ms (β=4.5e9 commented as this box's NCCL effective
  bytes/device/s); prose "within tens of percent" + subtraction-is-an-
  estimate caveat.
- **MGP-4 done** — the JAX climax is real: (a) ResNet18 + nnx.Optimizer
  instantiated, state explicitly replicated onto the mesh (`replicate`
  helper — required: mixed committed devices error otherwise), one jitted
  step on a real sharded batch, loss printed (2.47);
  (b) visualize_array_sharding of batch (sharded) AND last Linear kernel
  (replicated, via `kernel[...]` — `.value` deprecation-warns at flax
  0.12.7); (c) receipt: `train_step.lower(...).compile().as_text()`,
  grep all-reduce → 36 ops, largest 6.4M floats, HLO line printed; prose
  reads the 2-big(gradients, XLA-bucketed)+34-small(BN statistics —
  global under GSPMD, unlike DDP's per-replica BN) structure;
  (d) weak-scaling sweep k∈{1,2,4} guarded (measured k=1 2683, k=2 4390,
  1.64×, 82%; prose notes faster-per-device ⇒ comm relatively larger ⇒
  lower efficiency than DDP, and pre-staged arrays ⇒ not a framework
  shoot-out); (e) one-step k=1-vs-k=2 delta invariant (measured 1.1e-4 vs
  updates ~2.3e-2; prose frames tf32/BN reassociation across two compiled
  programs); (f) `import optax` (and os/re/time) in the jax imports cell;
  `opt.update(grads)` fixed to the flax 0.12.7 `opt.update(model, grads)`
  signature (the old code could never have run).
  Also: `NCCL_LOCAL_REGISTER=0` set in the jax imports cell — XLA's NCCL
  buffer registration fails env-wide (cuda error 500) and would print a
  warning on every collective (reproduced on 13.5's psum cell too).
- **MGP-5 done** — FSDP wrapper "still imports at our pin, deprecated
  legacy path"; reach-for-FSDP = memory inequality with few-billion kept
  as this card class's threshold; fig_fsdp_lifecycle caption gains
  "under one simplified reshard-after-forward policy".
- **MGP-6 done** — exercise 2 replaced (extend no_sync probe to k=4;
  compare 2(k-1)/k vs flat 2N/β); prerequisites line now points at 13.2
  for why-the-fabric-is-slow and 13.5 for what-the-collective-costs;
  exercise 5 says 88%→82% weak-scaling.
- **MGP-7 done** — (a) "loop body unchanged; scaffolding is the
  launcher's" (both spots, sampler/set_epoch/teardown named);
  (b) punchline → "one sharding vocabulary … though an effective plan is
  still model-aware"; (c) bridge names hierarchy of fabrics, rendezvous/
  elastic restart, stragglers; (d) 4-row collective table added
  (`tab_collectives`) after the FSDP mechanism; (e) summary + slides
  updated (weak-scaling labels, 88/82, comm-probe slide "Price the
  Fabric...", JAX demo slide + new "The Receipt" slide; "Honest Scaling"
  retitled "Weak Scaling, Measured").
- **FT-1 done** — rung 5 measured for real: `ddp_gpt/train_gpt_ddp.py`
  sidecar (d2l.GPT 512×6, char TimeMachine ctx 128 — downloaded by the
  parent's subject cell before any torchrun; per-rank batch 64 weak
  scaling; per-rank seeds), `ddp_gpt_tokens(k)` launcher, k∈{2,4} guarded
  by d2l.num_gpus(); prediction printed BEFORE measurement (t_comm ~34 ms
  vs t_cmp ~31 ms; no-overlap floor per k) — measured k=2: 297k vs floor
  253k (1.13× of one GPU); prose frames floor-vs-linear-ceiling with
  DDP's overlap explaining the gap; waterfall stays single-GPU, R5 its
  own clearly-marked multi-GPU cells; slide embeds the measurement cell.
  `torch.cuda.empty_cache()` before launches so ranks get the parent's
  allocator cache back.
- **FT-2 done** — 300-step learning run, fresh `make_gpt()`, compiled +
  bf16 + batch-512, losses kept on device (no host read in hot loop),
  smoothed first-20 vs last-20 printed (3.01 → 0.07), assert it fell;
  prose notes the tiny-corpus memorization so the low value reads right.
- **FT-3 done** — waterfall cell computes and prints
  `cumulative, R0 -> R3: {…:.2f}x`; prose says "about 2.4× in our runs —
  the number the cell prints". (Deviation from the spec's "say 2.8×" —
  see Deviations.)
- **FT-4 done** — `loss.backward()` moved outside autocast in step_bf16,
  step_ckpt, and the FT-2 loop; R2 prose names the discipline.
- **FT-5 done** — fp32 batch-512 peak-memory control (3 eager steps,
  16.6 GiB) printed beside R3; rung 3 reframed: batch-up is its own
  intensity rung; bf16 bought headroom (~half the footprint), not
  admission; end-to-end metric statement added after `throughput`
  (incl. measured <2% input-pipeline share).
- **FT-6 done** — (a) `assert torch.allclose(ckpt_gpt(X), model(X),
  atol=1e-6)` before compiling/timing R4 (measured bitwise 0.0);
  (b) eager-vs-compiled `assert torch.allclose(..., atol=1e-2, rtol=1e-3)`
  before R1's timing, grad-mode kept enabled (no_grad would trigger a
  dynamo recompile); measured max |Δ| ≈ 5e-3 (tf32 fusion reassociation),
  tolerance stated in prose.
- **FT-7 done** — compact re-profiles (row_limit=5, acc_events=True) as
  NEW cells after R1 and R3 with bottleneck-moved readings (R1: tail
  folds into CompiledFunction, mm share rises; R3: mm avg per-call grows
  ~3.4×); R0 profiler gains acc_events=True; intro promise weakened to
  "re-profiling as the bottleneck moves"; ALL profiler output verified
  warning-free (acc_events + TEARDOWN_CUPTI).
- **FT-8 done** — (a) thermal paragraph replaced (unreproducible clock
  claims dropped; hedged "whatever clock state the driver is in" +
  warmup discipline); (b) PyTorch-only signpost added to The Subject
  (pairs with IX-2); (c) waterfall labeled cumulative in prose, cell
  intro, plot title, and slide ("each bar inherits every choice to its
  left"); (d) slide "~4%" gone — replaced by measured ~1.3× (supersedes
  the spec's "a few percent" wording; see MAJOR DISCOVERY).
- **§3 calibration done** — zero "honest/honestly" and zero rhetorical
  "physics" remain in my two files (grep-verified); replaced by named
  properties (measured, completion-timed, end-to-end, "can be read
  cleanly", "a scaling curve you can price before you buy").

## Verification record

- 13.6 pytorch: full extracted sequence run (weak sweep k∈{1,2}: 2107 /
  3717 samples/s 88%; strong: 2092 / 3714 89%; comm probe 20 ms predicted
  vs 15 ms measured) — exit 0, no warnings (torchrun OMP banner appears
  only outside the scheduler, which sets OMP_NUM_THREADS=4).
- 13.6 jax: full sequence run — loss 2.47, both sharding visualizations,
  36 all-reduce receipt, sweep 2683/4390 (82%), invariant 1.1e-4 — exit
  0, zero warnings (NCCL_LOCAL_REGISTER placement validated in-sequence).
- 13.7: full sequence run — R0 263806, R1 345491 (1.31×), R2 474075
  (1.37×), R3 622771 (1.31×, 8.6 GiB, control 16.6), R4 545494 (0.88×,
  3.1 GiB), R5 k=2 297k vs floor 253k, cumulative 2.36×, learning 3.01→
  0.07 — exit 0, zero warnings, all asserts pass. Wall ≈ 2.5 min at k≤2;
  k=4 additions ≈ +1 min → comfortably inside the ~15 min budget.
- lint_source clean on both; add_cell_ids idempotent (0 new on rerun);
  #@save cells byte-identical vs HEAD (programmatic check); git status
  shows no out-of-scope modifications by me.
- New cell ids: multi-gpu-practice `ddp-really-run-4` (strong sweep),
  `ddp-really-run-5` (comm probe), `jax-…-collectives-3/-4/-5` (receipt/
  sweep/invariant); fast-transformer `rungs-each-one-measured-5/-6`
  (re-profiles), `-7/-8` (R5 script/measure), `the-waterfall-1`
  (learning run).

## RECONCILE AFTER CAPTURE

13.6 pytorch (`multi-gpu-practice`):
- `ddp-really-run-3`: prose quotes 88% (k=2) and 82% (k=4) and "about
  1.8× / 3.3×"; summary + Weak-Scaling slide + exercise 5 quote the same.
  k=4 numbers from whole-box capture must match; adjust all four spots
  together if they moved.
- `ddp-really-run-4` (NEW, k=4 strong sweep never run anywhere): prose is
  written to survive any outcome ("any gap that opens…is pure
  underutilization") but check it reads right against the printed k=4
  strong efficiency; expected ballpark 70–80%.
- `ddp-really-run-5`: prints predicted 20 ms vs measured (~15–16 ms in
  scratch); prose says "within tens of percent" — verify the captured
  subtraction stays within that envelope (whole-box thermal state may
  shift it a few ms).

13.6 jax:
- `…collectives-2`: loss print (scratch 2.47) — prose does not quote it.
  Visualizations at capture show 4 GPUs (scratch showed 2) — fine.
- `…collectives-3`: op count (scratch 36 at k=2) — prose says "dozens"
  and "(the exact count is a compiler artifact)"; verify count and that
  the largest is still the ~6.4M-float gradient sum at k=4.
- `…collectives-4`: k=4 weak efficiency unseen (k=2 scratch: 82%); prose
  quotes "82% already at two GPUs in our runs" — check the captured k=2
  line still prints 82±2% and that k=4 < k=2.
- `…collectives-5`: invariant magnitude (scratch 1.1–1.4e-4); prose says
  "about 10^-4" — confirm order of magnitude.

13.7 (`fast-transformer`) — all rungs re-print; prose is hedged to:
R1 "roughly 1.3×", first call "about two seconds", R2 "about 1.4×", R3
"roughly another 1.3×", control "snugly fits" (scratch 16.6 GiB), R4
"about a tenth" cost and "about 9 GiB down to 3", cumulative "about 2.4×
in our runs" (cell prints exact; scratch 2.36×), parity residue "about
10^-2" tolerance prose, R5 floor/measured (k=2 scratch: 253k/297k;
**k=4 entirely unseen** — prose says "roughly half of linear" for k=4,
verify against capture; expected floor 1.9×, measured ~2.0–2.4×),
learning-run pair (scratch 3.01→0.07; prose quotes no numbers).
Summary + slides carry the same magnitudes (1.3×/1.4×/1.3×/−10%/~2.4×).
If the whole-box capture's baseline R0 lands well below ~250k tokens/s
(other-GPU heat), the cumulative may drift toward 2.5–2.6× — prose says
"about 2.4×… the cell prints the exact figure", acceptable up to ~2.6;
beyond that, retouch the three "2.4×" mentions (waterfall prose, summary,
slide).

## FLAGS FOR CENTRAL PASS / SIBLINGS (outside my scope)

1. **index.md (agent A, IX-1)**: now says 13.7 "shows a case where
   [compilation] buys almost nothing — which is why diagnosis comes
   first". FALSE after the R1 artifact fix (compile buys ~1.3×). Needs
   rewording — e.g. point at R4/checkpointing as the technique that buys
   less than nothing, or drop the clause. THIS IS THE ONE CROSS-FILE
   CONTRADICTION my changes create.
2. **performance-model.md slides (agent A)**: literal `:eqref:`eq_roofline``
   inside a slide body (~line 71 of the slides block) — gen_slides does
   not process it; renders literally.
3. **13.5 jax psum cell (agent B)**: will likely capture the NCCL
   buffer-registration warning on re-execution (reproduced today on its
   exact code); `os.environ.setdefault('NCCL_LOCAL_REGISTER','0')` in the
   13.5 jax imports cell fixes it. 13.5's "Delay kernel timed out" E-line
   is a separate, known issue (MG-5e).
4. **Ground-truth table §1** rows for 13.6/13.7 are superseded by the
   numbers above (R1 1.04×→1.31×, R2 1.69×→~1.4×, R3 1.60×→~1.3×,
   cumulative 2.80×→~2.4×, R0 221k→~264k).

## Deviations (from the spec's letter, with rationale)

1. **FT-3**: spec said write "about 2.8×"; the honest cumulative after
   removing the two measurement artifacts is ~2.4× (the 2.80× divided a
   CUPTI-taxed baseline into an untaxed batch-512 number). The cell
   computes the ratio (spec's mechanism) and prose says "about 2.4×".
2. **FT-8d**: slide "~4%" was to become "a few percent"; it is now
   "~1.3×" because the underlying number was an artifact.
3. **R1/R0 narrative** rewritten beyond the spec's list (measured truth +
   the two traps taught in The Subject + new exercise 6). Justified by
   §0.5 (prose must quote re-execution-stable magnitudes — the old R1
   number was not even measurement-stable) and the chapter's own thesis.
4. Two environment lines added to imports cells (TEARDOWN_CUPTI in 13.7,
   NCCL_LOCAL_REGISTER in 13.6-jax), each with a one-line comment —
   required for clean, uncontaminated captured output.
5. MGP-4 invariant reported at default (tf32) matmul precision (~1e-4);
   'highest' tested (2e-5) but rejected as extra plumbing.
6. 13.7 exercise 5 re-posed (body now runs the eager DP measurement the
   old exercise asked for); new exercise: stack DDP on the full ladder.
7. The old thermal paragraph's specific claims were dropped entirely
   (not merely softened) — they do not reproduce on today's box
   (clocks pinned 2715 MHz idle→load; persistence behavior changed).

## Open risks

1. k=4 numbers (both files) are predictions/hedges until the whole-box
   capture; every dependent prose spot is listed above.
2. The R5 DDP measurement at capture shares the box with the parent
   notebook's CUDA context (rank 0 on GPU 0); empty_cache() mitigates,
   scratch runs were clean, but watch for OOM at k=4 in capture logs.
3. TEARDOWN_CUPTI is a kineto env knob, not a documented torch API;
   verified thoroughly at our pin (torch 2.11.0). If a future pin breaks
   it, the R0-after-profile throughput will silently sag ~16% — the
   "Profile Your Experiment First" prose documents the symptom.
4. jax sweep compile time (~15–20 s × 5 distinct shapes) makes the 13.6
   jax notebook ~4–5 min at capture; within budget but the longest jax
   notebook in the chapter.
5. R2's steady state fluctuated 474–495k across runs (~4%) — prose says
   "about 1.4×", which covers it; the printed ratio may read 1.37–1.43.
