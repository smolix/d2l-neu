# Ch. 13 comp-perf rebuild — Phase 0 pilot notes

*Per `reviews/comp-perf-implementation-brief.md` Part D Phase 0. Each pilot runs
as a scratch notebook executed the way the build executes notebooks (jupyter
nbconvert with the `d2l-<fw>` kernel; the scheduler pathway is the gold standard
for P1/P5). Verdicts here gate the structure of 13.6/13.7 and the width choice
for 13.7. Box: 4×RTX 4090, driver 595.71, no P2P; torch 2.11.0 / jax 0.10.2.*

---

## P1 — torchrun from a notebook cell (gates 13.6/13.7 structure)

**Status:** DONE (agent pilot-p1)

**Verdict: PASS.** A notebook cell executed under the build's nbconvert
pathway can write a minimal DDP script to a sidecar `.py` and launch it via
`torchrun --standalone --nproc-per-node=k` as a subprocess, cleanly, at k=2
**and** k=4, even when an earlier cell of the same notebook already touched
CUDA (the deliberate poison cell). The already-CUDA-touched parent kernel is
also unaffected afterward (post-launch matmul cell succeeds every time). No
STOP-and-reassess trigger hit.

**Reproduction of the build pathway (`tools/run_notebooks.py:execute_notebook`):**
read the function first — it runs
`sys.executable -m jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.timeout=<t> --ExecutePreprocessor.kernel_name=<kernel>
<nb>` via `subprocess.Popen(..., env=env)` with `CUDA_VISIBLE_DEVICES` injected
into `env` and no `cwd=` override (inherits the Make-driven repo-root cwd).
The pilot mirrored this exactly, with the one deliberate substitution the task
called for: `--ExecutePreprocessor.kernel_name=d2l-pytorch` (the registered
kernel) instead of `run_notebooks.py`'s ephemeral self-kernel — same
nbconvert invocation shape, run from `/home/smola/d2l-neu` (repo root):

```
CUDA_VISIBLE_DEVICES="0,1"      .venv-pytorch/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=d2l-pytorch pilot_run1.ipynb   # k=2 only
CUDA_VISIBLE_DEVICES="0,1,2,3"  .venv-pytorch/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=d2l-pytorch pilot_run2.ipynb   # k=2 and k=4
```

Both exited 0 (all assertions inside the notebook passed).

**Winning idiom (B):** resolve `torchrun` as the sibling binary of
`sys.executable`, not via `PATH`:

```python
torchrun_bin = str(pathlib.Path(sys.executable).parent / "torchrun")
argv = [torchrun_bin, "--standalone", f"--nproc-per-node={k}", str(script_path)]
subprocess.Popen(argv, cwd=..., env=os.environ.copy(), ...)
```

This is the idiom the book section should use. It needs no PATH assumption
and is guaranteed to exist alongside whichever `sys.executable` the kernel
is running (the `.venv-pytorch/bin/torchrun` console-script installed by the
same `uv sync` that installed torch).

**Launch-idiom matrix (all tested, both runs):**

| Idiom | argv | Result | Why |
|---|---|---|---|
| A — bare `"torchrun"` via PATH | `["torchrun", "--standalone", ...]` | **FAIL** — `FileNotFoundError: [Errno 2] No such file or directory: 'torchrun'` | `shutil.which("torchrun")` returned `None` **inside the kernel process** in both runs. The `d2l-pytorch` kernelspec's `argv` is an absolute path to `.venv-pytorch/bin/python` with no PATH injection (see `~/.local/share/jupyter/kernels/d2l-pytorch/kernel.json`), so the ipykernel process's inherited PATH (`/home/smola/.local/bin:/usr/local/cuda/bin:...:/usr/bin:...`) never includes the venv's `bin/`. **This is exactly the "works in a plain terminal, fails under nbconvert" case the pilot was built to catch** — a terminal with the venv activated (or Make's explicit `PATH="$(CURDIR)/.venv-pytorch/bin:$PATH"` prefix) would resolve this fine; the bare kernelspec does not. |
| B — `Path(sys.executable).parent / "torchrun"` | see above | **PASS** (primary/winning) | No PATH dependency. |
| C — `[sys.executable, "-m", "torch.distributed.run", ...]` | module invocation | **PASS** | Also PATH-independent; a fine fallback, marginally more verbose. |

**Evidence (returncodes / rank-line counts / grad-checksum agreement):**

- Run 1 (`CUDA_VISIBLE_DEVICES=0,1`, k=2 only — k=4 cell self-skips via a
  `torch.cuda.device_count() < 4` guard, "SKIP k=4: only 2 GPU(s) visible"):
  4 separate k=2 `--standalone` launches (idiom B ×3, idiom C ×1), **all
  `rc=0`, all exactly 2 parsed `RANK` lines, `world_size=2` on both, identical
  `grad_checksum=-9.17790222` on both ranks every time.** Idiom A: 1 launch,
  `FileNotFoundError` as above (expected/documented, not a torchrun failure).
- Run 2 (`CUDA_VISIBLE_DEVICES=0,1,2,3`, full notebook): k=2 primary-idiom
  launch **PASS** (`rc=0`, 2 ranks, checksum `-9.17790222`); **k=4
  primary-idiom launch PASS** (`rc=0`, 4 parsed `RANK` lines, `world_size=4`
  on all four, identical `grad_checksum=-6.21573782` on all four ranks —
  DDP's gradient allreduce genuinely synchronized across all 4 GPUs); plus
  the same idiom-C and back-to-back idiom-B k=2 checks, all PASS.
- `nvidia-smi --query-compute-apps` polled ~1/s during every launch and saved
  alongside each notebook's output: k=2 launches show exactly 2 worker PIDs,
  one each on `GPU-7ef3c7d1...` (index 0) and `GPU-7b266c81...` (index 1);
  the k=4 launch shows 4 worker PIDs, one on each of all 4 GPU UUIDs
  (indices 0-3) — physical-GPU assignment matches `--nproc-per-node` exactly.
  The parent kernel's own poisoned CUDA context (a distinct PID) stays
  visible on GPU 0 throughout every launch, undisturbed.
- **Post-launch parent health check:** after every k=2 *and* k=4 launch
  (including the two back-to-back k=2 relaunches), the parent kernel ran
  another `cuda:0` matmul with no exception and `torch.cuda.current_device()
  == 0` — launching/tearing down DDP subprocesses has zero observable effect
  on the already-CUDA-touched parent process.

**Launch overhead (informs the runtime budget):** consistently **~5.0–5.4 s**
wall-clock from `Popen` to the first parsed `RANK` line (torchrun spawn +
NCCL/rendezvous init dominates; k=4's ~5.4s was barely higher than k=2's
~5.0s — startup cost is close to k-independent in the 2-4 range on one node),
and **~6.4–7.1 s** total per cell including nvidia-smi-poll-thread and
reader-thread join overhead. Budget **≈6-8 s of pure launch/teardown per
torchrun cell**, independent of k, when sizing 13.6/13.7's cell timeouts and
prose about wall-clock cost.

**Rendezvous / port-clash finding:** `torchrun --help` documents
`--standalone` as "Start a local standalone rendezvous backend that is
represented by a C10d TCP store on a **free port**... any explicitly set
[rdzv] values are ignored" — i.e. it self-selects an available port every
invocation, no manual coordination needed. Empirically confirmed: each run
did **4 separate `--standalone` k=2 launches in immediate back-to-back
succession** (primary ×1, idiom C ×1, explicit back-to-back pair ×2) with
zero hangs or port-in-use errors across both runs (8 total back-to-back
launches). This matters because the real build scheduler may run other
notebooks concurrently — `--standalone` looks safe for that.

**Gotcha #1 (non-obvious, worth a callout in the book / a lesson for anyone
writing a similar multi-rank-prints-status-line cell):** the very first k=4
attempt (before a fix, evidence kept in
`pilot_run2_orig_executed_k4race.ipynb` / `primaryB_k4.stdout.log`) produced
only 3 *parseable* lines out of 4 real prints — the raw bytes showed
`"...grad_checksum=-6.21573782RANK 1/4 device=cuda:1 ...\n"` with **no
newline separating rank 2's and rank 1's lines**. Root cause: plain
`print(f"...", flush=True)` issues the line content and the trailing `"\n"`
as **two separate `write()` syscalls**; by default `torchrun` (no
`--redirects`/`--tee`) has every worker inherit the **same** piped stdout fd,
so two ranks' independent two-part writes can interleave when there are
enough concurrent writers. This never showed up across 4 separate k=2
launches (only 2 concurrent writers) but hit on the *very first* k=4 attempt
(4 concurrent writers) — a race whose odds scale with rank count, **not a
torchrun/nbconvert defect**; the DDP training itself was fully correct even
in the raced run (`rc=0`, all 4 ranks' printed checksums, once you account
for the missing newline by eye, agreed). Fix applied and reverified clean:
replace the print with one atomic syscall,
`os.write(1, line.encode())` (`line` includes its own trailing `"\n"`, total
size well under `PIPE_BUF` so POSIX guarantees atomicity) — k=4 then passed
cleanly on the next attempt with correctly delimited lines. **Recommendation
for 13.6/13.7:** if a cell has every rank print a status/verification line,
either use a single atomic write per line (as above) or the more idiomatic
real-world choice, rank-0-only logging — do not rely on bare multi-rank
`print(..., flush=True)` when output is captured through a pipe (which
nbconvert always does).

**Gotcha #2 (benign):** every launch's stderr carries torchrun's standard
`Setting OMP_NUM_THREADS environment variable for each process to be 1 in
default, to avoid your system being overloaded...` warning. Harmless, but if
13.6/13.7 wants clean cell output, set `OMP_NUM_THREADS` explicitly before
the `torchrun` call (or note in prose that the warning is expected/benign).

**Scratch artifacts** (kept in place, not committed):
`/tmp/claude-4002/-home-smola-d2l-neu/319ce40a-90af-4f30-bdd8-cfe83f878de0/scratchpad/pilot-p1/`
— `build_notebook.py` (notebook generator, incl. the final atomic-write
`train_ddp.py` source), `pilot_run1.ipynb` / `pilot_run2.ipynb` (final clean
executed notebooks — all cell outputs, nvidia-smi samples, timings inline),
`pilot_run1_orig_executed.ipynb` / `pilot_run2_orig_executed_k4race.ipynb`
(first-attempt raw evidence, including the k=4 stdout-interleave race),
`run{1,1b,2,2b}_nbconvert.log` (raw nbconvert stdout/stderr),
`*.stdout.log`/`*.stderr.log` per launch tag, `final_outputs.txt` /
`cell_outputs_run1.txt` (flattened readable transcripts).

No stray GPU processes were left running; verified via `nvidia-smi
--query-compute-apps` (empty) and `ps aux | grep torchrun` (empty) after
each run.

---

## P2 — DDP ResNet-18 throughput at k ∈ {1,2,4} (Fashion-MNIST-64)

**Status:** DONE (agent pilot-p2p3)

**Verdict: PASS.** Weak-scaling DDP throughput measured end-to-end through
the real notebook pathway (`jupyter nbconvert --execute --inplace`, kernel
`d2l-pytorch`, torchrun launched from inside the notebook via P1's winning
idiom B). `d2l.resnet18(10, 1)` on Fashion-MNIST resized to 64×64
(`transforms.Resize(64)` + `ToTensor()`, matching `d2l.load_data_fashion_mnist(...,
resize=64)`'s transform), fixed **per-rank batch 256** (weak scaling), SGD
lr=0.1, `torch.set_float32_matmul_precision('high')`, `cudnn.benchmark=True`.
Verified model parameter count: **11,175,818** (≈11.2M, matches the task's
cost-model constant). Epoch 1 = warmup (cuDNN autotune + `DistributedSampler`/
`DataLoader` worker spin-up + FashionMNIST already on disk, no download
needed), epoch 2 timed with `torch.cuda.synchronize()` immediately before/
after. k=1 ran through the identical torchrun harness (`--standalone
--nproc-per-node=1`), so launch/DDP overhead is apples-to-apples across all
three k values.

**Scaling curve / efficiency** (from the final nbconvert-executed notebook;
per-rank throughput was uniform across ranks at every k, as expected for
identical GPUs and a synchronized step):

| k | aggregate samples/s | speedup vs k=1 | efficiency |
|---|---|---|---|
| 1 | 2159 | 1.00x | 100% |
| 2 | 3819 (avg of 2 runs: 3818 / 3821) | 1.77x | ~90% |
| 4 | 7149 | 3.31x | ~85% |

Efficiency degrades gently and monotonically — no cliff at k=4 — consistent
with a fixed per-step compute cost (weak scaling) and a communication cost
that grows sub-linearly with k on a single node.

**t_comm estimate vs. cost-model prediction:** measured by differencing
timed full DDP steps against `model.no_sync()` steps (20 steps each, 5-step
warmup, `torch.cuda.synchronize()` bracketing every step), averaged across
ranks:

| k | measured t_comm | cost-model prediction (`2(k-1)/k × 44.7 MB / 2.2 GB/s`) | ratio measured/predicted |
|---|---|---|---|
| 2 | ~16 ms | ~20 ms | ~0.8 |
| 4 | ~25 ms | ~31 ms | ~0.8 |

The cost model **over**-predicts by a consistent ~20% at both k, i.e. the
simple ring-allreduce estimate is a reasonable, slightly-conservative
approximation on this no-P2P 4×4090 box — good enough for a book-prose
sentence ("allreduce cost model predicts X ms; measured ~0.8X"), not exact.
Ratio consistency across k (0.78 at k=2, 0.83 at k=4) is itself evidence the
measurement methodology (full-step vs. no_sync) is sound rather than noisy.

**Run-to-run noise (k=2, repeated inside the same notebook):** aggregate
samples/s 3817.7 vs. 3820.6 — **spread 0.07%**. Noise is negligible at this
batch/step count; the efficiency numbers above are not noise-limited.

**Runtime-budget check: PASS, comfortable margin.** The full sweep (k=1
baseline, k=2 run A, k=2 run B for the noise check, k=4), executed
end-to-end via `nbconvert` from repo root exactly like a real build, totaled
**250.9 s (4.2 min)** of measured cell time; total nbconvert wall clock
(incl. kernel startup) was under 5 min. 13.6's ~10 min runtime budget has
comfortable headroom left for prose/plotting cells even if the book keeps
both k=2 repeats for the noise-floor discussion; dropping the second k=2 run
would free another ~1 min if needed.

**API/idiom notes:** per-rank result files (`{tag}_rank{rank}.json`) written
directly by each rank, read back by the parent notebook process after
`torchrun` returns, **sidestep P1's Gotcha #1 entirely** (no shared-stdout
multi-rank print race to worry about) — worth recommending as the default
pattern for 13.6's DDP cells over rank-0-only logging or atomic
`os.write`, whenever a script needs to report *every* rank's own numbers
(not just rank 0's). `LazyConv2d`/`LazyBatchNorm2d` in `d2l.resnet18`
required one dummy forward pass to materialize parameters **before** wrapping
in `DDP` (`DDP` requires real `Parameter`s, not `UninitializedParameter`) —
worth a one-line callout in the book cell.

**Scratch artifacts** (kept in place, not committed):
`/tmp/claude-4002/-home-smola-d2l-neu/319ce40a-90af-4f30-bdd8-cfe83f878de0/scratchpad/pilot-p2p3/`
— `ddp_resnet.py` (sidecar DDP training script), `build_notebook_p2.py`
(notebook generator), `p2_pilot.ipynb` (final executed notebook, all cell
outputs inline), `results/*.json` (per-rank timing, incl. terminal
quick-iteration runs used to validate the script before the final nbconvert
pass).

No stray GPU processes after the run (`nvidia-smi --query-compute-apps`
empty, checked before and after inside the notebook itself and again from
the shell).

---

## P3 — 13.7 waterfall end-to-end at d=256 and d=512

**Status:** DONE (single-GPU rungs R0–R4 both widths; DP rung R5 deferred to
notebook execution, see below). Bench: `gpt_rung_bench.py` (scratch), reusing
`d2l.GPT(vocab, num_hiddens=w, num_blks=6)` + `d2l.TimeMachine(..., num_steps=128,
tokenization='char')` exactly as `chapter_transformers/gpt.md` does. Char vocab
= 28. Params: **d=256 → 4.73M**, **d=512 → ~18M**. AdamW, ctx 128, base batch 64.

**Width verdict: d=512.** The deciding rung is **bf16 (R2)**, which must be a
*positive* rung for the chapter (it is the headline precision technique). At
**d=256 bf16 goes BACKWARDS** — R2 (compile+bf16) is *slower* than R1
(compile-only): the char-vocab GPT's width-256 matmuls are too small to feed
the tensor cores, so bf16-autocast casting overhead dominates the tiny
tensor-core win. At **d=512 bf16 is a clean ~1.35× rung** (matmuls finally big
enough). Build 13.7 at **d=512**. Bonus: d=512's R0 noise floor is small and
clean (~3% across 3 runs) once the throttle prime + 60-step warmup is applied
(see the throttle trap below), so every rung clears noise there.

**Per-rung effects (clean, d=512; tokens/s single GPU):**

| rung | what | tokens/s | ratio vs prev | cumulative | note |
|---|---|---|---|---|---|
| R0 | eager, tf32-fair | ~273k (±3%) | — | 1.0× | baseline |
| R1 | + `torch.compile` | ~353k | ~1.3× | 1.3× | compile latency ~12.6 s first call, ~2 s cached |
| R2 | + bf16 autocast | ~475k | ~1.35× | ~1.7× | **positive at d=512** (negative at d=256) |
| R3 | + batch-up (512) | ~625k | ~1.3× | ~2.3× | freed bf16 memory buys batch; the roofline-climb rung (~9 GiB) |
| R4 | + activation ckpt | ~547k | 0.88× | ~2.0× | **deliberate negative rung** — memory not the binding constraint at this scale, so recompute only costs time |

Final clean d=512 waterfall (18.9M params): **R0 275k → R1 353k (1.28×) →
R2 479k (1.36×) → R3 625k (1.30×) → R4 547k (0.88×)**; cumulative ~2.3×
single-GPU, each positive rung clears the ~3% R0 noise. This is the plot
13.7 is written around.

(d=256 for contrast: R0 ~273k, R1 ~570–760k [compile helps MORE on the
smaller/overhead-bound model], **R2 ~484–656k < R1 — bf16 negative**, R3
~1565k stable [batch 1024, 8.5 GiB], R4 ~1304k negative. The bf16 sign flip
is the whole reason to prefer d=512.)

**Prose-precision guidance for 13.7 (per proposal §6.6):** quote rungs
qualitatively — "compile bought roughly a third; bf16 about 1.3–1.4×;
spending the freed memory on batch size is the biggest single win;
checkpointing measured as *not* helping here (a deliberate negative rung —
memory is not the binding constraint at this scale)". Do NOT quote raw
tokens/s decimals — they drift with the throttle state.

**`d2l.GPT` + `d2l.TimeMachine` reuse check: PASS.** Both reused verbatim
(no reimplementation). One friction: `d2l.GPT` has no built-in activation-
checkpointing toggle, so R4 wraps its exact submodules
(`token_emb`/`blks`/`norm`) in a thin `CheckpointedGPT` that only changes the
block-invocation loop to `torch.utils.checkpoint.checkpoint(blk, H,
use_reentrant=False)` — no change to `d2l/torch.py`. The book cell can do the
same wrapper inline.

**R5 (data-parallel rung, 2–4 GPU):** not separately micro-benchmarked here;
13.7 reuses 13.6's verified DDP harness (P1 idiom + P2 confirmed ResNet
scaling ~1.8×/2GPU, ~3.3×/4GPU). For the GPT the prediction from §13.5's
accounting is *communication-hungrier* than ResNet (transformer params ∝
compute, ~18M params = ~72 MB gradients to allreduce over the ~2 GB/s
host-staged fabric), so expect a modest 2-GPU gain and visibly sublinear
4-GPU scaling — the prediction-then-measure demonstration. Real numbers land
at notebook capture (Phase 4).

**⚠️ Throttle trap (verified on this box; flag for the book's benchmark
cells).** Persistence mode is OFF: idle SM clock ~210 MHz; under load the GPU
boosts to ~2700 MHz within ~1 s then **throttles down** over the next several
seconds as the die heats (measured eager step time drifted ~14 ms → stable
~29 ms over ~100 steps — a 2× drift that swamps rung-to-rung effects if warmup
is too short). Fix used: a ~12 s generic-matmul prime + 60-step per-rung
warmup → steady-state noise ~1–2%. **13.7's timing cells must warm up
properly** or the waterfall is thermally, not algorithmically, shaped. This is
the concrete, on-hardware instance of :numref:`sec_perf_model`'s warmup
discipline and worth a one-sentence callout in 13.7.

**Runtime:** each single-GPU width sweep (R0×3 + R1–R4, full warmup) runs in
a few minutes; well within 13.7's ~15 min/framework budget once the DP rung is
added. Scratch: `gpt_rung_bench.py`, `ddp_gpt.py`, `results/r256*.json`,
`results/r512.json`.

---

## P4 — memory-snapshot rendering path

**Status:** PASS (Part A pytorch), PASS (Part B jax). Both executed headlessly via
`jupyter nbconvert --to notebook --execute` from the repo root, kernels
`d2l-pytorch` / `d2l-jax`, `CUDA_VISIBLE_DEVICES=3` only. Scratch notebooks +
artifacts: `/tmp/.../pilot-p4/part_a_pytorch_memory.ipynb`,
`part_b_jax_memory.ipynb`, `snap.pickle`, `memory_trace_plot.png`,
`trace_plot.html`, `memory.prof` (paths below).

### Part A — PyTorch (`_record_memory_history` → `_dump_snapshot`)

Setup: 4×(Linear(2048,2048)+GELU)+Linear head, batch 256, Adam, 5 training
steps, `max_entries=100000`. `_dump_snapshot` wrote a 190,359-byte pickle;
`_record_memory_history(enabled=None)` stopped cleanly.

**Snapshot schema (torch 2.11.0+cu128), verified by loading the pickle in the
same notebook:**

- Top level: `dict` with exactly 4 keys — `segments`, `device_traces`,
  `allocator_settings`, `external_annotations`.
- `segments`: list (29 here), one per live allocator segment at dump time.
  Each: `device, address, total_size, allocated_size, active_size,
  requested_size, stream, segment_type ('small'/'large'), segment_pool_id,
  is_expandable, frames, blocks`. Each `blocks[i]`:
  `address, size, requested_size, state ('active_allocated'/'inactive'),
  frames`.
- `device_traces`: list of 1 (one entry per visible CUDA device — just
  `cuda:0` under our `CUDA_VISIBLE_DEVICES=3`), each a **chronological event
  list** (685 events for our 5 steps). Event keys: `action, addr, size,
  stream, time_us, compile_context, user_metadata, frames` (frames = a full
  Python call stack, 85 frames on the first event — this is what makes the
  trace flame-graph-capable but also why raw event dumps are noisy).
  Observed `action` values + counts: `alloc` 250, `free_requested` 203,
  `free_completed` 203, `segment_alloc` 29. **Gotcha:** frees are split into
  a `free_requested`/`free_completed` pair — a cumulative-bytes reducer must
  consume only one side (`free_completed`) or it double-subtracts; likewise
  every `segment_alloc` (the underlying `cudaMalloc`) is immediately followed
  by a same-address/same-size `alloc` event, so a reducer that sums `alloc`
  and `free_completed` only (skipping `segment_alloc`) is correct and does
  NOT double count the cudaMalloc.
- `allocator_settings`: dict of the live CUDACachingAllocator config
  (`PYTORCH_CUDA_ALLOC_CONF`, `max_split_size`, `garbage_collection_threshold`,
  `expandable_segments`, `roundup_power2_divisions`, ...).
- `external_annotations`: list (20 here) of `{stage: START/END, name, device,
  time_us}` — auto-captured optimizer/autograd phase markers (e.g.
  `Optimizer.zero_grad#Adam.zero_grad`, `Optimizer.step#Adam.step`); not
  needed for the basic plot but a nice future "which phase" x-axis overlay.

**Recommended book artifact — CONFIRMED WORKING, visually inspected:** a
matplotlib line plot of cumulative allocated bytes vs. event index,
reconstructed by walking `device_traces[0]` and doing `cum += size` on
`action == 'alloc'`, `cum -= size` on `action == 'free_completed'` (all other
actions ignored, per the gotcha above), plus a horizontal reference line at
`torch.cuda.max_memory_allocated()`. Rendered PNG
(`memory_trace_plot.png`, 65,612 bytes) shows **five clean sawtooth cycles**
— forward pass growing, backward+optimizer-step shrinking — with the peak
line sitting right at each spike top. This is pedagogically legible as-is.
One refinement for the real book cell (not done in the pilot): call
`torch.cuda.reset_peak_memory_stats()` right after building the
model/optimizer and before `_record_memory_history`, so the peak line is
scoped exactly to the 5 visualized steps rather than including earlier
cell activity; also consider plotting against `time_us` (already a field)
instead of raw event index for an honest wall-clock x-axis.

**`torch.cuda._memory_viz.trace_plot(snapshot)`:** works headlessly (returns
in ~0.002 s, no error) and produces valid HTML (254,132 bytes for our
190 KB snapshot). **Verdict: not embeddable, confirmed not just assumed.**
Two independent disqualifiers: (1) the HTML's `<script type="module">` does
`import ... from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"`
— an external, network-dependent, **unpinned** (`@main`) JS resource; it
cannot render standalone/offline and could silently break if that file moves
upstream. This alone rules it out for a static, offline-renderable Quarto
book page. (2) even ignoring that, it inlines the *entire* snapshot pickle
as base64 inside the HTML (~1.3× the pickle size here; scales up hard for a
real multi-epoch run with thousands of allocations). Matches the brief's
hypothesis exactly: works, but wrong shape for house style — at most a
"open this file in a browser with network access" aside, never an embedded
output.

**Basic counters path (the anatomy-table numbers):** after
`reset_peak_memory_stats()` + 3 more training steps: `memory_allocated` =
362.18 MB, `max_memory_allocated` = 446.11 MB, `memory_reserved` =
`max_memory_reserved` = 484.44 MB (reserved == max_reserved because the
allocator didn't need to grow further in this short run). These four calls
are cheap, synchronous, no snapshot/pickle involved — good for the anatomy
table regardless of which trace-level artifact 13.4 chooses.

### Part B — JAX (`save_device_memory_profile`, `memory_stats`, AOT `memory_analysis`)

Setup: comparable 4-block flax/nnx MLP (width 2048, batch 256), `nnx.Optimizer`
+ optax Adam, `@nnx.jit`-ed step, 5 steps, under
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` / `XLA_PYTHON_CLIENT_PREALLOCATE=false`
to coexist on GPU 3. **Gotcha at flax 0.12.7:** a bare Python `list` of child
`nnx.Linear` modules as a Module attribute now raises `ValueError: Found data
... assigned to static attribute` (stricter nnx pytree checking than earlier
flax) — must wrap as `nnx.List([...])`, matching the house pattern already
used in `chapter_builders-guide/model-construction.md` and
`chapter_attention/positional-information.md`.

- **`jax.profiler.save_device_memory_profile(path)`**: writes headlessly,
  no error. Format confirmed = gzip'd pprof/perftools `Profile` protobuf
  (magic bytes `\x1f\x8b`, `gzip.decompress` succeeds: 3,134 → 7,997 bytes
  for this toy step). **Decoding:** no `pprof`/`perfetto`/`gviz_api` package
  is importable in `.venv-jax` (none installed; the uv-managed venv has no
  `pip` to add one ad hoc). However `tensorflow` **is** a genuine, committed
  dependency of the `jax` extra already (`pyproject.toml`: pulled in because
  `d2l/jax.py` uses `tf.data`), and it ships a compiled pprof schema at
  `tensorflow.core.profiler.profile_pb2` that **does** parse the file: 22
  samples, 2 sample types (`('allocations','count')`, `('space','bytes')`),
  110 locations. So decoding is reproducible in this repo's own venv, not a
  fluke — but it's an internal/private TF proto import never documented for
  this purpose, requires manually walking `sample`/`location`/`string_table`
  to reconstruct anything visual (no one-line render), and — more
  fundamentally — it's a **single point-in-time live-allocation snapshot by
  call site** (a flame-graph source), not a before/after or over-time trace,
  so it cannot produce the same "sawtooth across training steps" picture
  PyTorch's trace gives even with full decode effort. **Verdict: decodable in
  principle, wrong shape of data and too much plumbing to be the book's JAX
  artifact.**
- **`jax.local_devices()[0].memory_stats()`** keys at jax 0.10.2: `num_allocs,
  bytes_in_use, peak_bytes_in_use, largest_alloc_size, bytes_limit,
  bytes_reserved, peak_bytes_reserved, largest_free_block_bytes, pool_bytes,
  peak_pool_bytes`. Sample values: `num_allocs=669, bytes_in_use=255,976,704,
  peak_bytes_in_use=507,763,712, bytes_limit=8,839,495,680,
  pool_bytes=peak_pool_bytes=538,968,064`; `bytes_reserved`/
  `peak_bytes_reserved` read 0 (expected — no arena preallocated under
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`). A single live snapshot, not a
  time-series; fine for the anatomy table or a small before/after bar, not
  a curve.
- **AOT path**, `jax.jit(step_fn).lower(params, opt_state, x, y).compile()`:
  works directly on a plain functional reimplementation of the same
  architecture/width/batch (not the `nnx.jit`-wrapped training step itself —
  staying with nnx would require manually splitting the Module into a static
  graphdef + array-state pytree before calling `.lower()`, which is plumbing
  orthogonal to the memory-introspection question being piloted).
  `compiled.memory_analysis()` returns a `jaxlib._jax.CompiledMemoryStats`
  exposing exactly: `alias_size_in_bytes, argument_size_in_bytes,
  generated_code_size_in_bytes, host_alias_size_in_bytes,
  host_argument_size_in_bytes, host_generated_code_size_in_bytes,
  host_output_size_in_bytes, host_temp_size_in_bytes, output_size_in_bytes,
  peak_memory_in_bytes, serialized_buffer_assignment_proto,
  temp_size_in_bytes` — every field the brief named, all confirmed present
  at 0.10.2, plus a bonus `peak_memory_in_bytes` (the single most useful
  number, directly comparable to PyTorch's `max_memory_allocated`). Sample
  values: `argument_size_in_bytes=255,975,428, output_size_in_bytes=
  251,781,384, temp_size_in_bytes=4,376, alias_size_in_bytes=
  generated_code_size_in_bytes=0, peak_memory_in_bytes=507,756,820`.
  **Gotcha:** `serialized_buffer_assignment_proto` comes back empty (`b''`)
  under default compile options — don't rely on it without extra XLA flags;
  the six scalar `*_size_in_bytes`/`peak_memory_in_bytes` fields need no flag
  and are sufficient. `compiled.cost_analysis()` (bonus, requested) returns a
  flat dict of ~44 keys mixing a couple of genuinely useful aggregates
  (`flops=30,413,545,472`, `'bytes accessed'=977,678,592`) with dozens of
  compiler-internal per-operand/per-fusion indexed keys (`'utilization17{}'`,
  `"bytes accessedout{2}"`, ...) whose indices are not stable across
  recompiles/code changes — not book-table material as a whole; cherry-pick
  `flops` / `'bytes accessed'` only if used at all.

**Verdict (JAX tab contrast):** YES — "PyTorch counts allocations at
runtime; XLA plans memory at compile time" is fully backed by working code
at our pins. The JAX tab should show the **AOT triple**
(`jax.jit(step_fn).lower(...).compile()` → `compiled.memory_analysis()`),
plotted/tabled from `peak_memory_in_bytes`, `argument_size_in_bytes`,
`output_size_in_bytes`, `temp_size_in_bytes` — computed with **zero device
execution** (contrast directly against PyTorch's trace, which necessarily
required 5 real steps to exist). `save_device_memory_profile` should NOT be
the JAX tab's rendered artifact (wrong data shape, no clean decode path);
optionally mention it in prose as "JAX also has a pprof-format live-snapshot
profiler for flame-graph tooling" without embedding its output. Optionally
pair the AOT numbers with a real post-execution
`jax.local_devices()[0].memory_stats()` read (`bytes_in_use`,
`peak_bytes_in_use`) as a one-line "compile-time estimate vs. runtime
reality" coda if 13.4 wants that extra beat.

**Recommended embedding for 13.4 (summary):**

| Framework | Embed | Do NOT embed |
|---|---|---|
| PyTorch | matplotlib cumulative-bytes-vs-event-index plot from `_dump_snapshot`'s `device_traces`, + `max_memory_allocated()` line | `trace_plot()` HTML (network-dependent, unpinned CDN import, inlines full base64 pickle) |
| JAX | matplotlib bar/table of `compiled.memory_analysis()` fields (AOT, zero execution) | `save_device_memory_profile()` pprof blob (needs an internal TF proto import to decode; wrong artifact shape — live snapshot, not a trace) |

Both frameworks' `#@save d2l.Benchmark`-adjacent counters
(`torch.cuda.max_memory_allocated` / `memory_allocated` /
`jax.local_devices()[0].memory_stats()`) are cheap, synchronous, and safe for
the anatomy table regardless of the plot choice above.

---

<!-- APPLIED 2026-07-20 (Phase 4): the P5 whole-box scheduler spec below is now
landed in tools/runtime_env.py (WHOLE_BOX_NOTEBOOKS = {multi-gpu-practice,
fast-transformer}; notebook_resource → ('gpu','all','all')) and
tools/notebook_scheduler.py (Item.slots, Scheduler._reserve whole-box branch +
per-GPU spg, _release). Dry-run over the 7 ch13 files confirmed: whole-box
notebooks each reserve [GPU 0,1,2,3] exclusively and serialize; 2-GPU
multiple-gpus gets a pair; INVARIANTS OK. Also updated MULTI_GPU_NOTEBOOKS
(dropped the 3 deleted files, kept multiple-gpus). -->

## P5 — JAX 4-GPU under the scheduler (whole-box reservation)

**Status:** DONE (agent pilot-p5)

**Verdict: PASS.** A JAX notebook that builds a `Mesh` over every visible
device, shards a batch with `NamedSharding`, runs a jitted sharded training
step, and uses `jax.shard_map` + `lax.psum` executes correctly under the
build's real scheduler pathway (`tools/notebook_scheduler.py` →
`make <stamp>` → `EXEC_RULE` → `run_one_notebook.py` assigned mode →
`nbconvert`) once the scheduler is taught a **whole-box** resource class. The
minimal change is a third marking table (`WHOLE_BOX_NOTEBOOKS`) plus one
admission branch in `Scheduler._reserve`; it degrades correctly to "all 2
GPUs" on a 2-GPU host and needs no per-GPU-count special-casing in the JAX
fraction formula. No STOP-and-reassess trigger hit.

### The Phase-3 spec: proposed diff

`notebook_resource` gets a third resolution rule ahead of `MULTI_GPU_NOTEBOOKS`,
returning a `('gpu', 'all', 'all')` sentinel — host-sized, never a literal
count. `Scheduler._reserve` gets one admission branch for the sentinel:
"every GPU must be entirely free" (a different admission test in kind from
the existing "≥ spg free on n emptiest GPUs" test, so it cannot share that
code path) that reserves each GPU's own full slot capacity. Selection,
release, and the JAX fraction formula are then generalized from a scalar
`spg` to a per-chosen-GPU list, which makes the fraction formula a **strict
generalization** of the existing one (identical arithmetic result when
`per_gpu_spg` is constant across chosen GPUs, for any `n_gpus`):

```diff
diff --git a/tools/notebook_scheduler.py b/tools/notebook_scheduler.py
index 0ffd6489..24a93373 100644
--- a/tools/notebook_scheduler.py
+++ b/tools/notebook_scheduler.py
@@ -79,6 +79,8 @@ class Item:
     def slots(self):  # total slots, for reporting / big-first tie-break
         if self.req[0] == "cpu":
             return 1
+        if self.req[1] == "all":
+            return -1  # host-sized; unknown until reserve() resolves it
         return self.req[1] * self.req[2]
 
 
@@ -221,22 +223,46 @@ class Scheduler:
             return {"cuda": "", "cpu_cores": self.core_groups[slot],
                     "_": ("cpu", slot), "_fw": it.fw}
         _, ngpu, spg = it.req
-        # GPUs with at least spg free, most-free first (spread load)
-        cand = sorted((g for g in range(self.num_gpus) if self.gpu_free[g] >= spg),
-                      key=lambda g: self.gpu_free[g], reverse=True)
-        if len(cand) < ngpu:
-            return None
-        chosen = sorted(cand[:ngpu])
-        for g in chosen:
-            self.gpu_free[g] -= spg
+        if ngpu == "all":
+            # Whole-box: every GPU on the host (never a hardcoded count —
+            # self.num_gpus degrades correctly on a 2-GPU host), each at its
+            # own full slot capacity. This is an exclusive reservation: it
+            # only fits when EVERY GPU is entirely free, not just >= some
+            # slot count on a subset — a fundamentally different admission
+            # test from the "n emptiest GPUs with >= spg free" test below, so
+            # it gets its own branch here. Nothing below this branch (GPU
+            # selection persistence, release, or the JAX fraction formula)
+            # special-cases the number of GPUs.
+            if any(free < cap for free, cap in zip(self.gpu_free, self.gpu_cap)):
+                return None
+            chosen = list(range(self.num_gpus))
+            per_gpu_spg = list(self.gpu_cap)  # reserve each card's own full capacity
+        else:
+            # GPUs with at least spg free, most-free first (spread load)
+            cand = sorted((g for g in range(self.num_gpus) if self.gpu_free[g] >= spg),
+                          key=lambda g: self.gpu_free[g], reverse=True)
+            if len(cand) < ngpu:
+                return None
+            chosen = sorted(cand[:ngpu])
+            per_gpu_spg = [spg] * len(chosen)
+        for g, s in zip(chosen, per_gpu_spg):
+            self.gpu_free[g] -= s
         env = {}
         if it.fw == "jax":   # jax preallocates: cap to the slots we reserved
-            vram = min(self.gpu_vram[g] for g in chosen)
-            frac = min(0.95, max(0.05, (spg * self.mib_per_slot) / vram))
+            # Per-GPU fraction, take the tightest (min) across the chosen
+            # cards so no GPU is ever over-committed. This is a strict
+            # generalization of the old scalar formula: when per_gpu_spg is
+            # the same value on every chosen GPU (the 1x/2x1/2x2 cases), the
+            # min-over-GPUs of (spg*mib/vram_g) reduces to spg*mib/min(vram_g)
+            # — byte-for-byte the previous formula, for any n_gpus. No
+            # n_gpus-value special-casing anywhere in this expression.
+            frac = min(0.95, max(0.05, min(
+                (s * self.mib_per_slot) / self.gpu_vram[g]
+                for g, s in zip(chosen, per_gpu_spg))))
             env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{frac:.2f}"
         self.fw_inflight[it.fw] += 1
         return {"cuda": ",".join(str(g) for g in chosen), "cpu_cores": None,
-                "extra_env": env, "_": ("gpu", chosen, spg), "_fw": it.fw}
+                "extra_env": env, "_": ("gpu", chosen, per_gpu_spg), "_fw": it.fw}
 
     def _release(self, asg):
         self.fw_inflight[asg["_fw"]] -= 1
@@ -244,9 +270,9 @@ class Scheduler:
         if kind == "cpu":
             self.cpu_free.append(asg["_"][1])
         else:
-            _, chosen, spg = asg["_"]
-            for g in chosen:
-                self.gpu_free[g] += spg
+            _, chosen, per_gpu_spg = asg["_"]
+            for g, s in zip(chosen, per_gpu_spg):
+                self.gpu_free[g] += s
 
     def _track(self):
         for g in range(self.num_gpus):
diff --git a/tools/runtime_env.py b/tools/runtime_env.py
index f97cb688..4001c35c 100644
--- a/tools/runtime_env.py
+++ b/tools/runtime_env.py
@@ -108,6 +108,26 @@ MULTI_GPU_NOTEBOOKS = {
     "chapter_optimization/minibatch-sgd.ipynb",
 }
 
+# Notebooks that need an EXCLUSIVE, whole-box GPU reservation: every GPU the
+# host has, at that GPU's full slot capacity — nothing else may share any
+# card while one of these runs. Unlike MULTI_GPU_NOTEBOOKS (a fixed 2 GPUs,
+# 1-2 slots each, happy to share the rest of each card with other jobs),
+# these build a jax.sharding.Mesh over every jax.devices() entry in one
+# process (13.6/13.7's Mesh + NamedSharding + jitted sharded-training-step
+# notebooks) and therefore must see ALL physical GPUs via CUDA_VISIBLE_DEVICES
+# with no other tenant preallocating VRAM on any of them.
+#
+# The requirement is deliberately host-sized, not a hardcoded GPU count: on
+# a 4-GPU box this reserves 4 GPUs; on a 2-GPU box (CI / a smaller render
+# host) it degrades to reserving both GPUs, matching the chapter's
+# 2-GPU-demonstrability rule (the notebook itself must never hardcode 4 —
+# only `len(jax.devices())`). See notebook_resource()'s ('gpu', 'all', 'all')
+# case and Scheduler._reserve in notebook_scheduler.py.
+WHOLE_BOX_NOTEBOOKS = {
+    "chapter_computational-performance/multi-gpu-practice.ipynb",
+    "chapter_computational-performance/fast-transformer.ipynb",
+}
+
 # (framework, relative-path) → number of global GPU slots to hold for
 # this single notebook. Used for notebooks that overflow the standard
 # per-process VRAM budget. Each entry causes run_one_notebook to flock
@@ -146,17 +166,24 @@ TWO_GPU_SLOTS_PER = {
 def notebook_resource(framework, rel, uses_gpu):
     """Resource requirement of one notebook, for the unified scheduler:
 
-        ('cpu',)         → 1 CPU slot
-        ('gpu', n, spg)  → spg GPU slots on EACH of n distinct GPUs:
-                           (1,1) default 1 slot · (1,2) 2 slots on one GPU
-                           (2,1) "2x1" (1 each on 2 GPUs) · (2,2) "2x2"
+        ('cpu',)           → 1 CPU slot
+        ('gpu', n, spg)    → spg GPU slots on EACH of n distinct GPUs:
+                             (1,1) default 1 slot · (1,2) 2 slots on one GPU
+                             (2,1) "2x1" (1 each on 2 GPUs) · (2,2) "2x2"
+        ('gpu', 'all', 'all') → whole-box: every GPU on the host, every slot
+                             on each (host-sized, never a literal count —
+                             resolved against the live GPU pool at reserve
+                             time, see Scheduler._reserve).
 
     A GPU slot is GPU_MIB_PER_SLOT (7.5 GiB) of VRAM. Default is 1 slot;
     memory-heavy single-GPU notebooks (HEAVY_GPU_NOTEBOOKS) take 2 on one GPU;
-    data-parallel notebooks (MULTI_GPU_NOTEBOOKS) take 2 GPUs at 1 (or 2) each.
+    data-parallel notebooks (MULTI_GPU_NOTEBOOKS) take 2 GPUs at 1 (or 2) each;
+    whole-box notebooks (WHOLE_BOX_NOTEBOOKS) take every GPU, fully.
     """
     if rel in CPU_ONLY_NOTEBOOKS:
         return ('cpu',)
+    if rel in WHOLE_BOX_NOTEBOOKS:
+        return ('gpu', 'all', 'all')
     if rel in MULTI_GPU_NOTEBOOKS:
         return ('gpu', 2, TWO_GPU_SLOTS_PER.get((framework, rel), 1))
     heavy = HEAVY_GPU_NOTEBOOKS.get((framework, rel))
```

Populate `WHOLE_BOX_NOTEBOOKS` with the real 13.6/13.7 relpaths once they
land: `chapter_computational-performance/multi-gpu-practice.ipynb` and
`chapter_computational-performance/fast-transformer.ipynb` (both PyTorch and
JAX tabs — the resource *class* is framework-agnostic even though only JAX
needs the fraction math; PyTorch's torchrun rung in the same notebooks also
needs all physical GPUs for its `--nproc-per-node` launch, see P1).

**Design-constraint check (against the brief's three requirements):**

- **(a) 2-GPU degradation:** `self.num_gpus` (derived from
  `D2L_GPU_SLOTS_PER`'s length, e.g. `detect_resources.py`'s per-host probe)
  is never hardcoded; `chosen = list(range(self.num_gpus))` is 2 items on a
  2-GPU host. **Verified for real** below.
- **(b) No n_gpus-value special-casing:** the one branch added is on the
  sentinel (`'all'` vs. a concrete int), which is a *selection-strategy*
  distinction ("all-or-nothing admission" vs. "top-n emptiest with partial
  availability"), not a branch on whether `num_gpus` is 2, 3, 4, or 13. The
  JAX fraction line is a single expression for both cases (proved algebraically
  above and empirically below on 2-GPU, 4-GPU, and a simulated heterogeneous
  3×24+1×32 GiB host).
- **(c) Marking-table style:** `WHOLE_BOX_NOTEBOOKS` is a bare-relpath set,
  same shape as `MULTI_GPU_NOTEBOOKS` (framework-agnostic; the class applies
  regardless of which tab needs it).

**Scope note:** this diff only touches the *scheduler* pathway
(`notebook_scheduler.py` + `runtime_env.py`), as instructed. The **standalone**
fallback (`run_one_notebook.py`'s `notebook_mode()`/`acquire_all_gpus`, used by
a bare `make _notebooks/jax/<ch>/<f>.executed` with no scheduler) still only
knows `'multi-gpu'` (2-GPU) vs `'gpu'`/`'cpu'`, and does **not** currently
special-case a whole-box notebook or scale `XLA_PYTHON_CLIENT_MEM_FRACTION`
for it (this is exactly the standalone/scheduler asymmetry flagged in the
task's ops context, today observed for `HEAVY_GPU_NOTEBOOKS` too). Phase 3
should add a parallel `'whole-box'` mode there (`acquire_all_gpus` already
takes all currently-detected GPUs — it would need only the same JAX-fraction
line, computed against the real per-GPU cap instead of a fixed constant) so a
surgical single-notebook refresh of 13.6/13.7 doesn't silently under- or
over-allocate. Also: docs/notebook-scheduler.md's marking table (§2) should
gain the `('gpu', 'all', 'all')` row when this lands for real.

**Operational caveat for Phase 3:** because whole-box admission requires
*every* GPU fully idle, a whole-box notebook dispatched mid-queue during a
full `make all` will simply wait every round until the entire pool drains,
then hold it alone for the notebook's ~10–15 min (per the brief's runtime
budget) before the pool refills. This is correct but worth expecting — it
reads as "the box looks stalled on one job" in the scheduler log, not a hang.

### Verification (real scheduler pathway, 4×RTX 4090, exclusive box)

**Toy notebook** (`/tmp/.../pilot-p5/p5_wholebox.ipynb`, kernel `d2l-jax`):
imports jax/flax.nnx/optax; asserts `len(jax.devices())` against the
`CUDA_VISIBLE_DEVICES` entry count (never hardcodes 4); builds a small
`nnx.Module` MLP (256→512→512→10) + `nnx.Optimizer`(optax.adam); a `Mesh`
over `jax.devices()` on axis `'data'`; `NamedSharding(mesh, P('data', None))`
+ `jax.device_put` for a synthetic 512-row batch; an `@nnx.jit` sharded
training step (20 steps); `jax.debug.visualize_array_sharding`; a
`jax.shard_map`+`lax.psum` cell; per-device `memory_stats()` at the end.
Before touching any GPU, the full cell logic was smoke-tested twice on CPU
(`XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu`) —
once as a flat script, once through real `nbconvert` — which caught a real
bug (see Gotcha #1 below) cheaply.

**Harness:** the diff above was applied to the real
`tools/runtime_env.py`/`tools/notebook_scheduler.py`, the notebook was placed
at `_notebooks/jax/chapter_computational-performance/p5_wholebox.ipynb` (the
scheduler's real, gitignored scratch tree — enumerated by `build_worklist`'s
`root.rglob("*.ipynb")` exactly like any real notebook), and its relpath was
added to `WHOLE_BOX_NOTEBOOKS` (a third, pilot-only entry alongside the two
placeholders above). Driven via:

```
D2L_GPU_SLOTS_PER="3,3,3,3" D2L_GPU_VRAM_PER="24564,24564,24564,24564" \
D2L_GPU_MIB_PER_SLOT=7680 D2L_CPU_SLOTS=4 \
  python3 tools/notebook_scheduler.py --frameworks jax \
  --files "chapter_computational-performance/p5_wholebox" --force-all
```

This is option (a) from the task, fully real: `notebook_scheduler.py`'s
worklist enumeration, `Scheduler._reserve`'s new whole-box branch, the
`_make_env`-injected `D2L_ASSIGNED_CUDA`/`XLA_PYTHON_CLIENT_MEM_FRACTION`, the
shelled `make --old-file ... <stamp>`, the real `EXEC_RULE` (venv,
`LD_LIBRARY_PATH`, `EXTRA_ENV_jax` incl. `XLA_PYTHON_CLIENT_PREALLOCATE=false`),
`run_one_notebook.py`'s **assigned mode** (`D2L_ASSIGNED_CUDA` set → skips its
own flock, calls `execute_notebook` directly), and the real
`jupyter nbconvert --execute --inplace --kernel_name=d2l-jax` — i.e. every
link in the chain the brief names, exercised for real, not simulated. (A
`--dry-run` pass was also run first as a zero-risk sanity check of the
reservation/invariant logic before any GPU touch; see below.)

**4-GPU whole-box run — PASS.** Scheduler line: `[jax]
chapter_computational-performance/p5_wholebox.ipynb: [GPU 0,1,2,3] running
(scheduled)` → `OK (16s)`. Cell 1 printed `CUDA_VISIBLE_DEVICES: 0,1,2,3`,
`XLA_PYTHON_CLIENT_MEM_FRACTION: 0.94` — exactly the value the patched
`_reserve` computes for 3 slots/24 GiB card (`3×7680/24564 = 0.9376 → 0.94`).
`jax.devices()` returned 4 `CudaDevice`s, matching the assertion.
`nvidia-smi --query-compute-apps` sampled mid-run showed **one PID (45769)
with a memory-usage row on all four physical GPU UUIDs simultaneously** —
the concrete "one process holds the whole box" evidence:

```
45769, .../python, 722 MiB, GPU-7ef3c7d1... (idx 0)
45769, .../python, 584 MiB, GPU-7b266c81... (idx 1)
45769, .../python, 584 MiB, GPU-915651cf... (idx 2)
45769, .../python, 584 MiB, GPU-3312b7bc... (idx 3)
```

**Fraction check — matches, via the right metric.** `nvidia-smi`'s raw
`memory.used` stayed low (~0.5–0.7 GiB/card) during the run, **not** ~21–22
GiB — this is *expected*, not a miss: the real build sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (`Makefile`'s `EXTRA_ENV_jax`), so JAX
grows on demand instead of eagerly grabbing the fraction up front, and this
toy model's real footprint is tiny. The correct way to verify the *cap*
actually landed is each device's own `memory_stats()['bytes_limit']`, printed
by the last cell: **`bytes_limit = 23,737,663,488` bytes (≈22.1 GiB) — uniform
across all 4 devices.** That's `0.94 × (CUDA free bytes at client init)`, a
few hundred MB below `0.94 × nvidia-smi's reported 24,564 MiB` because XLA
sizes the fraction against `cudaMemGetInfo` free bytes (already net of driver
context overhead), not the raw advertised total — a benign, expected
delta, not a computation bug. **Confirmed: the 0.94 fraction reached the JAX
client on all 4 GPUs, computed correctly by the patched scheduler and
threaded end-to-end through the real dispatch chain.**

**Training / sharding / collective checks — all PASS:** loss `250.91 → 124.03`
over 20 steps (more than halved, per the notebook's own assertion);
`X.sharding` / `jax.debug.visualize_array_sharding` showed the batch correctly
split 4 ways along `'data'` (rendered as the expected 4-block colored HTML
table, confirmed by inspecting the executed notebook's raw output — not just
its text stream); `l1 kernel sharding` showed `P()` (replicated, as expected
for small params under the default auto-sharding of `NamedSharding`+`jit`).
**`jax.shard_map` + `lax.psum` — top-level import path confirmed working at
0.10.2**: `from jax import shard_map` then calling `shard_map(fn, mesh=...,
in_specs=..., out_specs=...)` directly (see Gotcha #1) computed a
sharded-mean-via-psum that matched `jnp.mean` with **max abs error = 0.0**.

**2-GPU degradation — PASS.** Same notebook file, unmodified, driven through
the same real pathway with `D2L_GPU_SLOTS_PER="3,3" D2L_GPU_VRAM_PER="24564,24564"`
(simulating a 2-GPU host on the real 4-GPU box — the scheduler simply never
learns GPUs 2/3 exist). Scheduler line: `[GPU 0,1] running (scheduled)` →
`OK (15s)`. `nvidia-smi --query-compute-apps` during the run showed the same
single PID with rows on GPU UUIDs for index 0 and 1 **only** (indices 2/3
untouched). Cell output: `CUDA_VISIBLE_DEVICES: 0,1`, fraction **still
0.94** (same per-GPU cap → same formula result, confirming the generalization
is n_gpus-independent), `jax.devices()` → 2 `CudaDevice`s, the assertion
adapted automatically (`expected = len(CUDA_VISIBLE_DEVICES split)`, never a
literal 4), `Mesh('data': 2, ...)` built and trained correctly (loss
`250.91 → 124.03`), `shard_map`+`psum` again exact (err 0.0). **The notebook
never hardcodes a device count anywhere; it degrades cleanly.**

**`--dry-run` sanity check (pure reservation-logic unit test, no
subprocess/GPU touch)**, run before any real GPU execution, also exercised:
a 4×24 GiB pool (→ `cuda=0,1,2,3`, frac `0.94`, admission blocks a second
item while whole-box is in flight, full release on completion), a 2×16 GiB
pool (→ `cuda=0,1`, frac `0.94`), a simulated heterogeneous 3×24+1×32 GiB
pool (→ frac `0.94`, i.e. `min(3×7680/24564, 4×7680/32768) = min(0.938,
0.9375) → 0.94` — the min-over-GPUs generalization working as designed), and
a regression check that an ordinary `('gpu', 2, 1)` `MULTI_GPU_NOTEBOOKS`
item is byte-for-byte unaffected (`cuda=0,1`, frac `0.31`, only 2 of 4 GPUs'
free-counts touched).

**Gotcha #1 (notebook-authoring, caught by the CPU smoke test before any GPU
run):** `from jax import shard_map` binds the name directly to the
**callable** — `jax.shard_map` *is* the `shard_map` function at the
top-level, not a submodule namespace like the deprecated
`jax.experimental.shard_map.shard_map`. Writing `shard_map.shard_map(...)`
(the old shape) raises `AttributeError: 'function' object has no attribute
'shard_map'`. The correct 0.10.2 top-level idiom is `shard_map(fn,
mesh=..., in_specs=..., out_specs=...)(x)` directly. Worth a callout in
13.5/13.6 prose since it's a natural mistake when migrating from the old
import.

**Gotcha #2 (build-script bug, not a JAX finding):** the pilot's first notebook
generator wrote each cell's `"source"` as `str.split("\n")` (no line
terminators kept), which nbformat/nbconvert then joined into **one run-on
line per cell with no separators** — a `SyntaxError` that looked like a JAX
problem but wasn't. Fixed with `str.splitlines(keepends=True)`. Recorded here
in case a future book-generation script makes the same mistake.

**memory_stats() is GPU/TPU-only:** confirmed (again, consistent with P4)
that it returns `None` on the CPU backend — expected, guarded in the
notebook, not an issue on the real GPU runs (both runs printed real
`bytes_in_use`/`peak_bytes_in_use`/`bytes_limit` dicts for every device).

**What was exercised for real vs. simulated:**

| Layer | Real or simulated |
|---|---|
| `notebook_resource` → `('gpu','all','all')` sentinel | **Real** (patched `runtime_env.py`) |
| `build_worklist` enumeration of the scratch notebook | **Real** (`_notebooks/jax/...` `rglob`) |
| `Scheduler._reserve` whole-box branch + JAX fraction formula | **Real**, on the actual 4-GPU pool and (via env override) a simulated 2-GPU / heterogeneous pool |
| `make <stamp>` shell-out, `EXEC_RULE`, `D2L_ASSIGNED_CUDA` env plumbing | **Real** |
| `run_one_notebook.py` assigned-mode dispatch | **Real** |
| `jupyter nbconvert --execute` under kernel `d2l-jax` | **Real**, on real GPUs, both 4-GPU and 2-GPU-simulated configs |
| Standalone (non-scheduler) whole-box dispatch in `run_one_notebook.py` | **Not implemented / not exercised** — flagged as a Phase-3 follow-up above |
| A truly separate 2-GPU *physical* host | **Not available** — degradation verified by restricting `D2L_GPU_SLOTS_PER` to 2 entries on the real 4-GPU box, which exercises the exact same code path a real 2-GPU host's `detect_resources.py` output would feed in |

**Cleanup:** prototype edits reverted (`git checkout -- tools/runtime_env.py
tools/notebook_scheduler.py`; `git status`/`git diff` both clean on both
files, confirmed). Scratch notebook, `.executed` stamp, `.d` file, and
auto-generated `.provenance.json` removed from `_notebooks/jax/...`. No
stray GPU processes (`nvidia-smi --query-compute-apps` empty, `ps aux | grep
nbconvert\|ipykernel` empty) after both runs. Scratch artifacts (not
committed) at
`/tmp/claude-4002/-home-smola-d2l-neu/319ce40a-90af-4f30-bdd8-cfe83f878de0/scratchpad/pilot-p5/`:
`build_notebook.py` (generator), `p5_wholebox.ipynb` (source),
`p5_wholebox_4gpu_executed.ipynb` / `p5_wholebox_2gpu_executed.ipynb`
(executed evidence), `p5_cpu_smoke.ipynb` (pre-GPU smoke test),
`scheduler_wholebox.diff` (the diff above, as a file), `run4gpu.log` /
`run2gpu.log` (scheduler stdout).

---

## STOP-and-reassess triggers (from the brief)

- P1 fails both idioms (torchrun subprocess AND fork harness) → restructure.
- P5 fails both paths → reassess scheduler marks.
- P3 finds no width where rungs clear noise → re-scope 13.7's rung list.

---

## MEASURED OUTPUTS from the real capture run (2026-07-20) — prose ground truth

These are the numbers the executed notebooks actually printed; §13.2/13.5/13.7
prose must match THESE (not the pilots' pre-run estimates). Reconciliation pass
applied after capture.

- **13.1** timing trap: naive 0.87 ms vs synchronized 9.12 ms (~10× lie ✓);
  elementwise add/mul/sin/sigmoid all ~0.14 ms (bandwidth-bound ✓); unfused
  chain 1.28 ms vs 1 op 0.15 ms.
- **13.2** HBM bandwidth: measured **0.80 TB/s** (spec ~1.0); pinned H2D
  **24.3 GB/s** vs pageable 13.4 GB/s.
- **13.5 (pytorch)**: hand-rolled star allreduce **20.7 GB/s** effective on the
  256 MB payload (this is the RAW PCIe host-staged copy rate — NOT the ~2 GB/s
  NCCL *busbw* the framework audit measured; different metric, big gap worth a
  theory-vs-practice sentence). LeNet: 1 GPU 2.18 s/epoch (acc 0.78), 2 GPU
  **2.36 s/epoch** (acc 0.81) — only ~8% slower, NOT ~30%. Fix 13.5's "~2 GB/s"
  and "~30% slower" claims; the honest lesson is "LeNet is too small to amortize
  parallelism (halved per-device batch + Python overhead)", with the raw copy
  itself being fast.
- **13.6 (pytorch DDP, NCCL)**: 1 GPU 2107 samples/s, 2 GPU 3710 (**1.76×,
  88%**), 4 GPU 6911 (**3.28×, 82%**). Matches P2 pilot and my prose (~1.8×/
  ~3.3×). NCCL busbw here is the ~2 GB/s figure; ResNet is compute-dense enough
  to hide it → good scaling.
- **13.7 (pytorch waterfall, d=512, ~18.9M params)**: R0 221k → R1 compile
  **229k (1.04×)** → R2 bf16 **389k (1.69×)** → R3 batch-up **621k (1.60×,
  peak 8.7 GiB)** → R4 checkpoint **545k (0.88×, peak 3.1 GiB)**. Cumulative to
  R3 ≈ **2.8×**. Fixes needed: compile bought only ~4% (not "a third"); bf16 is
  the ~1.7× win; checkpointing DID cut peak memory 8.7→3.1 GiB (my draft said
  "unchanged" — wrong) for ~12% more time.
