# Fixlog — agent D (NCCL default-vs-configured story, 13.6/13.7 + pointers)

Scope: implement Alex's decision to adopt `NCCL_SHM_USE_CUDA_MEMCPY=1` as
ch. 13's multi-GPU running configuration and teach the
default-vs-configured story (root cause:
`reviews/nccl-p2p-investigation-2026-07-20.md`). Operating rules:
`reviews/comp-perf-review-findings-2026-07-20.md` §0.

## HEADLINE — adoption as specified is NOT shippable; scope pivoted

**`NCCL_SHM_USE_CUDA_MEMCPY=1` deadlocks PyTorch DDP training on this box**
(torch 2.11.0+cu128, bundled NCCL 2.28.9, driver 595.71.05), 100%
reproducible within 0–20 steps, across every escape hatch tried. The
investigation's "verified allclose, risk low" only exercised *bare,
sequential* collectives — those are indeed flawless and ~5× faster. Real
DDP workloads wedge. Full forensic matrix below.

Mid-task, Alex's framing directive arrived (recorded here, binding):
teach the switch as **pragmatic engineering** — a workaround for a
platform-specific NCCL performance bug, not a canonical config; the
transferable lesson is the workflow (measure against the wire, find the
stage, look for the escape hatch, re-validate per platform); toggle
exercise's moral = "workarounds must be re-validated per platform";
sidecar comments must say "works around … on this box; validate on
yours". The deadlock finding *strengthens* this framing, and the landed
design follows it:

- **All training sidecars run at NCCL defaults** (no env lines; scripts,
  probe, and 13.7's R5 byte-identical to the captured store except where
  noted). The captured sweep/probe/R5 numbers remain valid.
- **13.6 gains one new pytorch cell** (`multi-gpu-practice-ddp-really-run-6`):
  a bare-allreduce sidecar run twice via `torchrun` — default vs
  `NCCL_SHM_USE_CUDA_MEMCPY=1` passed as a visible env kwarg — printing
  effective bytes/device/s both ways and the ratio (~4.5 vs ~22 GB/s,
  ~5.0×). This is the default-vs-configured story, *measured*, in the
  only form that is stable on this box.
- **Prose** (13.6, after the no_sync probe): the general sentence Alex
  asked for (collective-library configuration alone — transport
  selection, topology assumptions — moves performance by factors, not
  percent; measure yours), the platform-specific-bug framing, and the
  honest reason the training runs keep the defaults: the workaround wins
  the microbenchmark and deadlocks the workload. Exercise 3 has the
  reader reproduce both halves (payload sweep of the bare comparison;
  flip the switch inside `train_ddp.py` and watch the k=2 sweep wedge —
  with a kill warning). Summary bullet + new slide
  ("Configured vs. Default: Factors, Not Percent") carry the same story.
- **13.5 (multiple-gpus.md), prose-only**: pytorch tab now frames the
  switch as a workaround for a platform-specific performance bug, "not a
  setting to copy blindly", pointing at 13.6's measurement *and its
  limits*; jax tab, summary bullet, and the "The Accounting" slide
  updated to match. No code cell touched (no jax recapture).
- **13.2 (hardware.md), prose-only**: "names the one-line cure" →
  "names the one-line workaround, which :numref:`sec_multi_gpu_concise`
  weighs". The "default fallback settings" framings verified still true
  (busbw ~2 GB/s claims are about the default; "flat from 2 to 4 GPUs"
  holds in both regimes: default 2.24→2.17, CE 11.2→10.6 busbw).
- **13.7 (fast-transformer.md)**: R5 sidecar + beta cell REVERTED to the
  captured state (verified: extracted pytorch cell sequence diff vs
  pre-edit extraction shows only my reverted lines; **no 13.7 recapture
  needed**). One prose-only touch in the R5 prediction paragraph: after
  "NCCL's own busbw reads ~2 GB/s" added "— the default-fallback figure:
  :numref:`sec_multi_gpu_concise` measures a five-fold-faster configured
  mode, and why these runs nonetheless keep the library's defaults".

## Deadlock forensics (all with GPUs 0,1 / 0-3 idle, exclusive)

Bare collectives under `NCCL_SHM_USE_CUDA_MEMCPY=1` — ALL PASS:
- allreduce 44.7/76/256 MB, k=2 and k=4 (15 timed iters): 11.0–11.3
  busbw k=2, 10.4–10.7 k=4 (matches investigation).
- broadcast 1→250 MB; coalesced broadcast 4×44 MB; allgather /
  reduce-scatter 1 KB→45 MB; all_gather_object; barrier; 50 back-to-back
  tiny broadcasts + allreduces. Transport confirmed `via SHM/CE/direct`
  with the env var set *inside* the script pre-init (the sidecar pattern).

Deadlocks — ALL HANG (timeout-killed):
| repro | config | result |
|---|---|---|
| real `train_ddp.py` k=2 | CE flag | hang at DDP() init (rings connect, then nothing) |
| DDP init `_sync_module_states` (127-tensor resnet18 state_dict, mixed fp32/int64) | CE flag | hang |
| same | + `NCCL_SHM_MEMCPY_MODE` 1/2/3 | hang |
| `_broadcast_coalesced` 16 grouped buckets × 4 MB | CE flag | hang (8×4 MB passes; 16 tiny pass) |
| same | + `NCCL_BUFFSIZE=16 MB` | passes — but 64×4 MB still hangs (fifo-capacity deadlock, threshold scales with fifo) |
| synthetic DDP ResNet training k=2/k=4 | CE + BUFFSIZE 16 MB | hang at step ~10 / step 0 |
| + `broadcast_buffers=False` | " | hang |
| + `bucket_cap_mb=1000` (single bucket) | " | hang |
| + one-bucket AND no buffer bcast | " | hang after step 5–9 |
| GPT 512×6 DDP training k=2 (no BN, 13.7's R5) | " | hang < step 20 |

Conclusion: grouped collectives beyond the SHM fifo capacity *and*
collectives running concurrently with autograd compute both wedge the
CE path. No documented knob combination
(`NCCL_SHM_MEMCPY_MODE` 1/2/3, `NCCL_BUFFSIZE` 16 MB, `NCCL_PROTO=Simple`,
one-bucket DDP, `broadcast_buffers=False`) makes DDP training survive.
This is precisely the "re-validate the workaround on the workload"
lesson, now taught by the section.

## Task-by-task status

1. **13.6 adopt env in sidecar + re-verify** — PIVOTED as above. Sidecars
   at defaults; new bare-collective cell carries the configured
   measurement; β stays 4.5e9 (its comment unchanged); weak/strong/probe
   cells' code untouched except the imports cell (`import os` added for
   the new cell's launcher env) — that changes prefix fingerprints, so
   **the whole 13.6 pytorch notebook needs recapture** (jax untouched —
   no jax cell modified; verified no `%%tab jax` cell in the diff).
2. **Exercise flip** — DONE in pivoted form: exercise 3 now assumes the
   bench cell exists, extends it, then has the reader flip the switch in
   the training sidecar and observe the wedge (kill warning included);
   morals: defaults-on-untuned-topology + re-validate workarounds
   per platform and per workload.
3. **13.7** — REVERTED to captured state (code); prose pointer added
   (directive framing); prediction/measurement story unchanged (floor
   253k / measured 297k at k=2; floor 506k / measured 455k at k=4, per
   the store). No recapture required.
4. **13.5/13.2 pointers** — DONE (see above), directive framing.
5. **End-to-end verification** — 13.6 full pytorch sequence re-run on the
   idle 4-GPU box (numbers below). 13.7 not re-run (code byte-identical
   to the green capture of today); its extracted-sequence diff is empty.

   **Fingerprint check against the store** (generated notebooks via
   `tools/gen_notebooks.py` into scratch, `prefix_fingerprints` from
   `tools/capture_outputs.py`):
   - `jax/multi-gpu-practice`: **0 stale, 0 missing, 0 orphaned** — no
     jax recapture.
   - `pytorch/multi-gpu-practice`: all 7 existing cells stale (prefix
     cascade from the imports cell's `import os`) + 1 missing
     (`ddp-really-run-6`, the new cell) — whole-file pytorch recapture,
     as intended.
   - `pytorch/fast-transformer`: **0 stale, 0 missing, 0 orphaned** — no
     recapture.
   - `#@save` cells in all four edited files byte-identical to git HEAD
     (programmatic check).
   - `tools/lint_source.py` clean on all four; `add_cell_ids` idempotent
     (0 assigned on re-run).

## Measured numbers (this session, exclusive box)

- Bare allreduce effective bytes/device/s (2N/t convention), k=2 GPUs 0,1:
  default 4.42–4.53 GB/s (44.7–256 MB payloads); configured 21.99–22.57;
  ratio ≈ 4.9–5.1×. k=4 configured: 13.9–14.2 (2N/t), busbw 10.4–10.7.
- nccl-tests-convention busbw k=2: 2.21–2.26 default; 10.99–11.29
  configured (matches the investigation's 2.24 / 11.24).
- **13.6 full extracted pytorch sequence (16 cells incl. the new bench
  cell): exit 0, no warnings beyond the expected torchrun OMP banner
  (absent under the scheduler), all launches clean.** Prints:
  - weak sweep: 2107 / 3716 (1.76×, 88%) / 6964 (3.30×, 83%) samples/s —
    store had 2105 / 3714 (88%) / 6931 (82%): k=4 efficiency is on the
    88/82-vs-83 noise edge (see RECONCILE).
  - strong sweep: 2092 / 3713 (1.78×, 89%) / 5881 (2.81×, 70%) — store:
    2094 / 3715 (89%) / 5876 (70%). Exact reproduction.
  - no_sync probe: predicted 20 ms, measured 15 (synced 133, no_sync
    117) — store: 20 / 15 (132 / 117). Exact reproduction.
  - NEW bench cell: "bare allreduce, effective bytes/device/s: 4.4 GB/s
    at the default, 22.3 GB/s configured (5.1x)"; standalone re-run
    after the directive-comment amendment: 4.4 / 22.2 (5.1x).
- The bench cell now carries the directive's comment at the switch's
  call site: "Works around an NCCL transport bottleneck on this
  P2P-less box (13.5's diagnosis) -- a platform-specific fix; validate
  on yours."
- **13.7 full extracted pytorch sequence re-run for verification** (code
  byte-identical to today's green capture): exit 0, zero warnings, all
  asserts pass. Prints (idle box): 18.9M params; R0 308,860 tokens/s;
  R1 346,648; R2 507,586; R3 623,483 (peak 8.6 GiB, fp32 control 16.6);
  R4 546,133 (0.88×, 3.1 GiB); R5 k=2 measured 297k, k=4 measured 455k
  (both EXACTLY reproduce the store); cumulative R0→R3 2.02×; learning
  run 3.01 → 0.07 (matches store).
- **FLAG — R0 is CPU-contention-sensitive** (eager dispatch-bound at
  batch 64): idle box 309k vs 263k under whole-box load (the store's
  capture condition). R1–R4 and both R5 sidecar numbers are stable; the
  R0-derived quantities (R1 ratio, cumulative, R5 floors) move with it —
  idle-box cumulative prints 2.02× vs the store's 2.37×, outside agent
  C's ≤2.6 hedge on the LOW side. **Do not recapture 13.7** (its code is
  unchanged); if it is ever recaptured, do it under normal scheduler
  load, or expect to retouch the "about 2.4×" mentions and R5-floor
  narration per agent C's reconcile list.

## RECONCILE AFTER CAPTURE (13.6 pytorch only)

- `multi-gpu-practice-ddp-really-run-6` (NEW): prints
  "`bare allreduce, effective bytes/device/s: X GB/s at the default,
  Y GB/s configured (Z x)`" — prose says "roughly five-fold" (X ≈ 4.5,
  Y ≈ 22, Z ≈ 4.8–5.1 across my runs); 13.5's pytorch tab says "roughly
  five-fold recovery"; 13.7's pointer says "five-fold-faster"; slide
  says "~5×". If the captured Z lands below ~4.5 (DRAM contention during
  whole-box capture), soften all four spots to "four-to-five-fold".
- `ddp-really-run-3` weak sweep: prose/summary/slide/exercise 5 quote
  88% (k=2) / 82% (k=4) and ~1.8×/3.3× — unchanged from the store;
  re-check the recaptured prints match (±1–2 pts OK; adjust all four
  spots together if they moved).
- `ddp-really-run-4` strong sweep: "nearly coincide at k=2 … part
  company as k grows" — store printed 89%/70%; verify shape holds.
- `ddp-really-run-5` probe: "within tens of percent" (store: 20 predicted
  vs 15 measured); verify envelope.
- The imports cell change means every pytorch cell in the file
  re-executes; jax cells must NOT be re-captured (unchanged).

## Open risks / flags for the central pass

1. The deadlock is torch/NCCL-pin-specific. If the torch pin moves, the
   bench cell stays valid either way, but exercise 3's "wedges within
   seconds" claim and the 13.6 prose beat should be re-tested at the new
   pin (a fixed NCCL would turn the deadlock claim false — the prose
   hedges with "on our box, with this NCCL build", but re-verify).
2. During whole-box capture the bare-bench cell adds 2 torchrun launches
   (~40 s total). It runs at k=2 on GPUs 0,1 via `--nproc-per-node=2`
   (guarded by `d2l.num_gpus() >= 2`).
3. The k=2/k=4 "flat" hardware.md claims were re-verified in both
   regimes; no numbers there needed changing.
4. If Alex still wants full adoption despite the deadlock, options that
   were NOT taken (and why): `NCCL_BUFFSIZE` enlargement (moves the
   grouped-collective threshold but training still wedges),
   `broadcast_buffers=False` / one-bucket DDP (still wedge, and gut the
   section's overlap story). A driver/NCCL upgrade or an NVIDIA bug
   report is the real path; the investigation doc + this matrix are the
   evidence package.
