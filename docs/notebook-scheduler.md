# Notebook execution scheduler

Reference for `tools/notebook_scheduler.py` — the component that executes the
book's notebooks (the long pole of a build: ~100 min for all 524 on the 4×4090
box). It is invoked by `make run-all-notebooks` (all frameworks) and
`make run-notebooks-<fw>` (one framework); `make all` calls the former.

The design has one guiding idea: **it is a queue, served as resources become
available.** Every notebook is a task on a single queue; a task starts the moment
the slots it needs are free. There is no phased plan, no per-framework barrier,
and no per-notebook lock — those were tried and removed (see
[§7 History](#7-history--what-was-tried-and-removed)).

---

## 1. Resources

`tools/detect_resources.py` measures the host and emits the slot pools. Two
resource types, sized independently:

### GPU slots — one per 7.5 GiB of VRAM, **per physical GPU**

`GPU_MIB_PER_SLOT = 7680` (7.5 GiB). For each GPU *i*:

    slots(i) = floor(VRAM_i / 7680)

Slots are tracked **per physical GPU**, not as a flat pool, so heterogeneous
GPUs simply contribute different counts:

| Host | per-GPU slots (`GPU_SLOTS_PER`) | total |
|---|---|---|
| 4 × 24 GiB (RTX 4090) | `3,3,3,3` | 12 |
| 1 × 32 GiB (RTX 5090) | `4` | 4 |
| 3 × 24 GiB + 1 × 32 GiB | `3,3,3,4` | 13 |
| 2 × 16 GiB | `2,2` | 4 |

`detect_resources.py` reports this with `--report` and exposes
`--get GPU_SLOTS_PER` (e.g. `3,3,3,3`) and `--get GPU_VRAM_PER` (per-GPU MiB,
used for the JAX memory fraction). Override the per-slot size with
`GPU_MIB_PER_SLOT=...`.

### CPU slots — one per 8 cores (min 1)

    CPU_SLOTS = max(1, floor(cores / 8))     # CPU_PER_LIGHT = 8

A 6-core laptop still gets 1 CPU slot. Each CPU slot is pinned to a dedicated
**core group** (`sched_setaffinity`) so CPU notebooks don't oversubscribe: the
cores are split into `CPU_SLOTS` contiguous groups (~8 cores each on a 64-core
box) and slot *i* gets group *i*.

### CPU-only / macOS hosts

On a host with **no GPU** (e.g. an Apple-silicon Mac) `detect_resources.py`
reports `0` GPUs, so the queue is served entirely from the CPU pool — only the
notebooks classified CPU-only (`CPU_ONLY_NOTEBOOKS` in `tools/runtime_env.py`)
run; GPU notebooks have nothing to schedule onto and are left to the freshness
gate to *defer* (build-system.md §3.3a). CPU affinity pinning uses
`os.sched_setaffinity`, which is **Linux-only**; off Linux it degrades to a
no-op (the slot pool still bounds concurrency, the cores just aren't pinned), so
the scheduler imports and runs unchanged on macOS. In practice on a laptop you
rarely invoke the scheduler directly — the single-notebook refresh path
(`gmake -B _notebooks/<fw>/<ch>/<f>.executed`, via `run_one_notebook.py`) is the
common CPU-only flow.

The Makefile hands all of this to the scheduler via `SCHED_ENV`
(`D2L_GPU_SLOTS_PER`, `D2L_GPU_VRAM_PER`, `D2L_GPU_MIB_PER_SLOT`, `D2L_CPU_SLOTS`).
Each is derived once from `detect_resources.py` and can be overridden on the
command line (e.g. `gmake CPU_SLOTS=2 run-all-notebooks`).

---

## 2. Per-notebook resource requirement

What a single notebook needs is decided by
`runtime_env.notebook_resource(framework, rel, uses_gpu)`, which returns one of:

| Requirement | Meaning | Marked by |
|---|---|---|
| `('cpu',)` | 1 CPU slot | default for a non-GPU notebook |
| `('gpu', 1, 1)` | 1 GPU slot (one GPU) | **default** for a GPU notebook |
| `('gpu', 1, 2)` | 2 slots on **one** GPU (memory-heavy) | `HEAVY_GPU_NOTEBOOKS[(fw, rel)] = 2` |
| `('gpu', 2, 1)` | **2×1** — 1 slot on **each of two** GPUs | `rel ∈ MULTI_GPU_NOTEBOOKS` |
| `('gpu', 2, 2)` | **2×2** — 2 slots on **each of two** GPUs | `MULTI_GPU_NOTEBOOKS` + `TWO_GPU_SLOTS_PER[(fw, rel)] = 2` |
| `('gpu', n, 'all')` | **n whole cards** — every slot on each of n GPUs, rest of the box free for others | `FULL_CARD_NOTEBOOKS[rel] = n` |
| `('gpu', 'all', 'all')` | **whole box** — every GPU, every slot on each | `rel ∈ WHOLE_BOX_NOTEBOOKS` |

The tuple is `('gpu', n_gpus, slots_per_gpu)`: *n_gpus* distinct GPUs, each
contributing *slots_per_gpu* slots. A GPU slot is 7.5 GiB of VRAM, so a notebook
that peaks above 7.5 GiB on a card must be marked for 2 slots on that card.

Resolution order in `notebook_resource`:

0. `rel ∈ GPU_ONLY_NOTEBOOKS` or `(fw, rel) ∈ GPU_FRAMEWORK_NOTEBOOKS` →
   `('gpu', 1, 1)`; `rel ∈ CPU_ONLY_NOTEBOOKS` → `('cpu',)`. Both lists
   exist because the keyword scan is textual and unreliable (a bare "GPU" in a
   base64 output blob reads as GPU; a JAX notebook that never names a device
   reads as CPU). In ch. 17 the seven sections that train on Fashion-MNIST
   (denoising, image-diffusion, samplers, design-space, latent-diffusion,
   few-step, discrete-diffusion) are in `GPU_ONLY_NOTEBOOKS`: both frameworks
   put the model and data on the GPU (PyTorch through `d2l.try_gpu()`, JAX on
   its default device). On the pre-rebuild sections a GPU took a PyTorch run
   from 1–7 min on four CPU threads to under a minute and a JAX run from
   7–51 min on a CPU slot (#95) to 1–1.5 min, and the old image-diffusion JAX
   run exceeded the 3600 s cell timeout on a CPU. denoising is listed although
   most of its cells are 2-D: its one-step JAX image denoiser is too slow on a
   CPU slot. The rebuild plans 1–10 GPU-min per notebook per framework
   (latent-diffusion about 50 in PyTorch and 20 in JAX after the DiT
   scaling curve was added); the times are re-measured at the first
   scheduler run. gaussian-paths, which uses only the 2-D running example,
   is GPU-only as well: its two 40k-step models take about 5 min (PyTorch)
   and under 1 min (JAX) on a GPU against 8–9 min each on 8 CPU cores. The
   per-framework set `GPU_FRAMEWORK_NOTEBOOKS` overrides a
   CPU-only listing for one framework only; it is empty at present.
   `run_one_notebook.notebook_mode`, `file_uses_gpu` and
   `scan_notebook_manifests.source_execution_class` check both GPU lists
   before any keyword scan, so the single-notebook `make -B …executed` path
   agrees with the scheduler. `source_execution_class` takes the framework too,
   so the Make stamp lists (`EXECUTED_GPU_<fw>`) and the freshness gate's
   capability tiers (build-system.md §3.3a) classify all of these as `gpu`:
   a CPU-only host defers them like any GPU notebook.
   `rel ∈ WHOLE_BOX_NOTEBOOKS` → `('gpu', 'all', 'all')` — a
   notebook that scales over every visible GPU (a `jax.sharding.Mesh` over
   `jax.devices()`, `torchrun --nproc-per-node` up to the device count). They
   run on any k ≥ 2; whole-box is used so the captured scaling tables include
   the k = 4 rows the prose quotes. `rel ∈ FULL_CARD_NOTEBOOKS` →
   `('gpu', n, 'all')` — a notebook that sizes its own work to the visible
   cards' memory (the async HPO notebooks start `total_memory // 7 GB`
   syne-tune workers per visible GPU), or one whose prose quotes timings,
   bandwidths or peak memory that another tenant on the card would skew
   (`gpus-devices-memory`, `kv-cache`, and ch. 13's sections 13.1–13.5;
   `multiple-gpus` takes two whole cards). `FULL_CARD_NOTEBOOKS` is checked
   before `MULTI_GPU_NOTEBOOKS`.
1. `rel ∈ MULTI_GPU_NOTEBOOKS` → `('gpu', 2, TWO_GPU_SLOTS_PER.get((fw, rel), 1))`
   — a data-parallel notebook genuinely uses two GPUs; default 1 slot each
   (`2×1`), bumped to 2 each (`2×2`) for a framework that allocates >7.5 GiB on a
   card.
2. `(fw, rel) ∈ HEAVY_GPU_NOTEBOOKS` → `('gpu', 1, n)` — memory-heavy on a single
   GPU.
3. `uses_gpu` (the `file_uses_gpu` keyword heuristic, which also checks sibling
   frameworks) → `('gpu', 1, 1)`. The keywords are `runtime_env.GPU_KEYWORDS`
   (`try_gpu`, `cuda`, `GPU`, `Trainer(`, `d2l.train`, `fit_lm(`, ...); a new
   training helper whose name matches none of them must be added there, or
   its notebooks are scheduled on a CPU slot.
   The single-notebook path (`make -B _notebooks/<fw>/<ch>/<f>.executed`)
   applies the same heuristic.
4. otherwise → `('cpu',)`.

### The marking tables (`tools/runtime_env.py`)

```python
MULTI_GPU_NOTEBOOKS = {            # data-parallel: need 2 GPUs
    "chapter_builders-guide/gpus-devices-memory.ipynb",
    "chapter_computational-performance/multiple-gpus.ipynb",
}

WHOLE_BOX_NOTEBOOKS = {            # every GPU, fully; nothing shares a card
    "chapter_computational-performance/multi-gpu-practice.ipynb",
    "chapter_computational-performance/fast-transformer.ipynb",
}

FULL_CARD_NOTEBOOKS = {            # n whole cards, whatever their size
    "chapter_builders-guide/gpus-devices-memory.ipynb": 2,     # timings
    "chapter_transformers/kv-cache.ipynb": 1,                  # timings
    "chapter_computational-performance/performance-model.ipynb": 1,
    "chapter_computational-performance/hardware.ipynb": 1,
    "chapter_computational-performance/compilation.ipynb": 1,
    "chapter_computational-performance/memory-precision.ipynb": 1,
    "chapter_computational-performance/multiple-gpus.ipynb": 2,
    "chapter_hyperparameter-optimization/rs-async.ipynb": 1,   # syne-tune
    "chapter_hyperparameter-optimization/sh-async.ipynb": 1,   # syne-tune
}

HEAVY_GPU_NOTEBOOKS = {            # 2 slots on ONE gpu (excerpt)
    ("jax", "chapter_generative-adversarial-networks/dcgan.ipynb"): 2,
    ("tensorflow", "chapter_computer-vision/ssd.ipynb"): 2,
    ("tensorflow", "chapter_natural-language-processing-pretraining/bert-pretraining.ipynb"): 2,
    ("tensorflow", "chapter_computer-vision/fine-tuning.ipynb"): 2,
}

TWO_GPU_SLOTS_PER = {             # per-(fw, rel) override: 2x2 instead of 2x1
    # e.g. ("tensorflow", "chapter_computational-performance/multiple-gpus.ipynb"): 2,
}

PINNED_GPU_NOTEBOOKS = {          # which physical GPUs, not only how many
    "chapter_computational-performance/multiple-gpus.ipynb": (0, 1),
}
```

**Pinned GPUs.** `PINNED_GPU_NOTEBOOKS[rel]` names the physical GPUs
(`nvidia-smi` indices) a notebook must run on. It does not change the
requirement above; it narrows the GPUs `reserve` may choose from to the pinned
set (§3.3). Use it when the prose quotes a number that depends on *which* cards
the notebook gets. `multiple-gpus` (13.5) measures the allreduce bandwidth β
between two cards, and on the 4×4090 box (no peer-to-peer; two pairs behind two
PCIe host bridges, `nvidia-smi topo -m`) that depends on the pair: NCCL reaches
β ≈ 4.9 GB/s on GPUs 0,1, ≈ 6.2 on 2,3 and ≈ 7.3–7.4 on 0,2 or 1,3 (2026-11-02,
256 MB per device, both frameworks). Pinned to 0,1, every capture measures the
same pair; the standalone path (§4) shows the notebook every GPU, and the
notebook takes the first two, so it also runs on 0,1. A pin naming a GPU the
host does not have is ignored. CUDA numbers identical cards in PCI bus order,
as `nvidia-smi` does; on a box with mixed cards set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
before relying on a pin.

**To change a notebook's footprint:** add it to the right table (above). A
single-GPU notebook that OOMs → `HEAVY_GPU_NOTEBOOKS[(fw,rel)] = 2`. A
data-parallel notebook that OOMs at `2×1` → `TWO_GPU_SLOTS_PER[(fw,rel)] = 2`.
`TWO_GPU_SLOTS_PER` starts empty and is filled in empirically. A notebook that
spreads work over GPUs it was not given (failures name a GPU outside its
`[GPU n]` reservation, or a trial OOMs on a card another job holds) is a bug in
whatever ignores `CUDA_VISIBLE_DEVICES` — fix that rather than widening the
reservation. Example: syne-tune 0.3.x counts GPUs with `nvidia-smi --list-gpus`
and pins trial *i* to physical GPU *i*; `tools/check_runtime_deps.py pytorch`
(run by the `.venv-pytorch/.runtime-deps` target after every venv sync) patches
it to honour the parent's `CUDA_VISIBLE_DEVICES`.

---

## 3. Scheduling algorithm

### 3.1 The queue order is framework-grouped

`build_worklist` enumerates **framework by framework**
(`pytorch → tensorflow → jax → mxnet`), and within each framework by relpath,
keeping only stale notebooks (stamp missing, or older than the `.ipynb` or any
`.d`-listed dependency; `--force-all` includes everything). The result is one
flat list:

    [pytorch/a, pytorch/b, …, pytorch/~141,
     tensorflow/a, …, jax/a, …, mxnet/a, … ]

This ordering is the **only** correctness mechanism for cross-framework safety —
see [§3.4](#34-why-no-barrier-and-no-lock).

### 3.2 Dispatch loop — first-fit, served as slots free

A single loop (holding a lock) repeats until the queue is empty and nothing is
in flight:

```
each round:
    for item in queue (in framework order):
        asg = reserve(item)          # None if its slots aren't free right now
        if asg: start item on asg; remove from queue
    if nothing was started this round:
        wait until a running notebook finishes (which frees slots)
```

It is **first-fit**: the round scans the whole queue and starts *every* item that
fits right now, **skipping** (not blocking on) items that don't. So a 2-GPU
notebook that can't find two GPUs with a free slot is passed over and smaller
items behind it keep the pools full; it starts on a later round when a gap opens.

### 3.3 Reservation — `reserve(item)`

- **CPU:** take any free CPU slot → its core group → `D2L_ASSIGNED_CPU_CORES`.
- **GPU `('gpu', n, spg)`:** find the GPUs that currently have ≥ *spg* free slots,
  pick the *n* **emptiest** of them (spread load), reserve *spg* on each. The
  chosen GPU indices become `D2L_ASSIGNED_CUDA` (e.g. `"2"`, or `"0,1"`). If fewer
  than *n* qualifying GPUs exist, return `None` (try again next round).
  For a pinned notebook (`PINNED_GPU_NOTEBOOKS`) the candidates are its pinned
  GPUs only, so it waits for them even while other GPUs are free. Like any
  whole-card request, it is skipped while other jobs hold slots on those cards
  and may therefore start late in a busy run, at the latest when the GPU queue
  drains; it never blocks the items behind it.
- **JAX only:** because JAX preallocates, the scheduler also sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION = clamp(spg × 7.5 GiB / card_VRAM, 0.05, 0.95)`
  so a JAX process is capped to exactly the slots it reserved (e.g. 1 slot on a
  24 GiB card → 0.31; 2 of 3 slots → 0.63). A card whose slots are **all**
  reserved by the job (whole-box `('gpu','all','all')`, `('gpu', n, 'all')`, or
  *spg* equal to that card's capacity) gets the full-card fraction
  `FULL_CARD_MEM_FRACTION = 0.94` instead, whatever the slot count; the rest of
  the card is left to the CUDA context and cuDNN/cuBLAS workspaces. Before
  2026-10-27 the slot formula applied there too, which tied a whole-box job's
  arena to `D2L_GPU_SLOTS_PER`: under `2,2,2,2` the JAX fast-transformer got
  0.62 (15 GiB) and OOMed at its 13.5 GiB fp32 step, while `3,3,3,3` gave 0.94.
  With several cards the tightest card sets the fraction. PyTorch/TF/MXNet
  allocate on demand and need no fraction.

Releasing on completion returns the slots to their GPUs / CPU pool and wakes the
loop.

### 3.4 Why no barrier and no lock

The same notebook exists in up to four frameworks and they share `data/` (e.g.
`kaggle-cifar10` / `kaggle-dog` reorganize `data/.../train_valid_test/` with
non-atomic `shutil.copy`). Running two framework variants of the **same** notebook
at once is the one real hazard (a half-written image → OpenCV
`imdecode_ !buf.empty()`; concurrent `d2l/*.py` rebuilds → `partially initialized
module`).

The **framework-grouped order alone** prevents this: a notebook's pytorch and jax
variants are ~one framework apart in the queue (~130 dispatches), and the slot
pool is only ~12–20 wide, so by the time the jax variant is dispatched the
pytorch variant finished long ago. They are **never in flight together** — with
no barrier and no per-relpath mutex. (At ~130 notebooks/framework the probability
of overlap is effectively zero; this is intentional, not luck.)

Consequences of "just a queue":

- **No per-framework barrier** — there is no point where the build waits for a
  framework to finish. The tail of pytorch's GPU work overlaps the head of
  tensorflow's; only *different* notebooks overlap, which is always safe.
- **CPU notebooks flow ahead.** GPU and CPU pools are independent and the scan
  covers the whole queue, so a later framework's CPU notebooks land on free CPU
  slots while earlier frameworks' GPU notebooks are still running. CPU work tends
  to finish early; the GPUs are the long pole.
- **2-GPU notebooks are mixed in** with 1-GPU and CPU notebooks — no separate
  multi-GPU phase or gating. Per-GPU slot accounting keeps them from
  double-booking a card.

---

## 4. Dispatch mechanism

Each task is run by shelling **`make <stamp>`** (one `make` per notebook), not by
importing frameworks in-process — so the scheduler stays framework-agnostic and
reuses the Makefile's tested per-framework recipe:

```
make --no-print-directory \
     -o .preprocess.stamp -o d2l/.built -o d2l/<fw>.py -o _notebooks/<fw>/.generated \
     _notebooks/<fw>/<rel>.executed
```

with env: the chosen device(s) in `D2L_ASSIGNED_CUDA` (and `D2L_ASSIGNED_CPU_CORES`
for CPU items), plus the JAX fraction when applicable.

- **Env reuse, no duplication.** The `.executed` pattern rule (`EXEC_RULE`) sets
  the venv (`UV_PROJECT_ENVIRONMENT`, `PATH`), `LD_LIBRARY_PATH` for the NVIDIA
  libs, the per-framework thread/memory tuning (`EXTRA_ENV_<fw>`: `OMP=2`,
  `TF_FORCE_GPU_ALLOW_GROWTH`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, …), runs
  `run_one_notebook.py`, tees to `logs/run-<fw>-<ts>.log`, and `@touch`es the
  stamp on success. The scheduler reuses all of it.
- **Assigned mode.** `run_one_notebook._run_once` sees `D2L_ASSIGNED_CUDA` and
  **skips its own flock-based slot acquisition** — the scheduler already placed
  it. It sets `CUDA_VISIBLE_DEVICES` to the assignment, applies the CPU affinity,
  and (for MXNet) keeps the short cuDNN-init serialization. Pipefail means a
  notebook failure makes `make <stamp>` exit non-zero, so `@touch` is skipped (no
  stamp) and the scheduler records the failure.
- **`-o` (`--old-file`) guards.** Because each notebook is a *separate* `make`,
  without these the inner makes would race to rebuild the **shared** d2l library /
  preprocess / notebook set and corrupt `d2l/*.py` mid-import (one race once wiped
  the hand-maintained preamble → `partially initialized module`). Those phase
  stamps are built **once upfront** by `make all` (lib → notebooks → run-all);
  `-o` tells every inner make to treat them as up-to-date.
- **Warm `.pyc`.** Before dispatching, the scheduler runs `compileall -q -f d2l`
  once (serially) so the first burst of concurrent imports finds a valid
  `d2l/__pycache__/*.pyc` and doesn't race on writing it. All four venvs are the
  same CPython, so one compile warms the shared cache.
- **No per-dispatch `detect_resources`.** The inner make's `?=` resource knobs are
  pre-set in the child env (computed once at scheduler start) so it doesn't shell
  `detect_resources` / `nvidia-smi` on every dispatch.

### Locked notebooks

A notebook whose source carries `<!-- execution: locked; … -->`
(`docs/build-system.md` §3.7, `tools/locks.py`) never goes on the queue. Before
dispatch the scheduler materializes each stale locked notebook from the
committed `outputs/` store in-process (no slot, no GPU), touches its stamp and
prints `[fw] chapter/x.ipynb: LOCKED (skipped) — materialized from …`; the
summary adds `, N LOCKED (skipped)` (only when N > 0, so unlocked runs print
exactly what they did before). A locked notebook that cannot be materialized
(no manifest) is listed as `FAILED: … (LOCKED: not materialized)`.
`run_one_notebook.py` applies the same rule to a direct `make <stamp>`.
A notebook that was materialized while locked keeps a `<stem>.materialized`
file; once its source is no longer locked, the scheduler queues it, removes its
stamp before dispatch and executes it (printing `was materialized from
outputs/ while locked; no longer locked, so it is executed`).
`--rerun-locked --files …` (`make rerun-locked`) inverts the filter: it queues
only the named locked notebooks, removes their stamps and provenance sidecars,
and sets `D2L_RERUN_LOCKED=1` so the inner `run_one_notebook.py` executes them.

### Standalone fallback

A direct `make _notebooks/<fw>/<rel>.executed` (no scheduler — used for rebuilding
one notebook) has no `D2L_ASSIGNED_CUDA`, so `run_one_notebook` falls back to its
own flock-based slot acquisition, and `serialize_dataset_prep` guards the
shared-dataset notebooks. This path is unchanged and self-contained.

---

## 5. Configuration & overrides

| Knob | Default | Effect |
|---|---|---|
| `GPU_MIB_PER_LIGHT` / `GPU_MIB_PER_SLOT` | 7680 | VRAM per GPU slot |
| `CPU_PER_LIGHT` | 8 | cores per CPU slot |
| `D2L_GPU_SLOTS_PER` | detected | per-GPU slot capacities, e.g. `3,3,3,3` |
| `D2L_CPU_SLOTS` | detected | CPU slots |
| `--frameworks a,b` | all four | restrict to a subset (`run-notebooks-<fw>` passes one) |
| `--files "a.md b.md"` | all | restrict to a subset of source notebooks (whitespace/comma-separated `chapter_x/foo.md`); `make refresh-stale` passes the audit-stale set here for a parallel re-run |
| `--force-all` | off | ignore stamp freshness when *selecting* notebooks. The inner `make <stamp>` still sees an up-to-date stamp as done and executes nothing, so a notebook whose `.executed` stamp `make notebooks` kept fresh is reported `OK` without running. That happens when cells were only **moved or deleted**: `gen_notebooks.py` (`_merge_outputs_from_prev`) counts a code change only when a cell's id+source changed, and then re-touches the stamp. After reordering or cutting cells, delete the stamps first (`rm _notebooks/*/<chapter>/<file>.executed`) and run the scheduler without `--force-all` (2026-11-10, ch. 1 pass) |
| `--dry-run` | off | simulate dispatch (no execution); checks per-GPU invariants |
| `--rerun-locked` | off | with `--files`: execute exactly the named LOCKED notebooks (`make rerun-locked`); refuses unlocked files |
| `D2L_GPU_LIVE_GATE=1` | off | shared hosts: admit a GPU job to a card only when `nvidia-smi` shows enough free memory (see below) |
| `NOTEBOOK_TIMEOUTS` (`tools/runtime_env.py`) | 3600 s | per-notebook wall-clock limit for notebooks that legitimately run longer (e.g. 17.6's DiT scaling curve, 7200 s); `run_one_notebook.py --timeout N` overrides it |
| `D2L_GPU_LIVE_HEADROOM_MIB` | 2048 | extra free MiB required beyond `slots × GPU_MIB_PER_SLOT`; a reservation that fills the card (whole-box, or 3 slots on 24 GB) instead needs free ≥ VRAM − headroom, i.e. an idle card |
| `D2L_GPU_LIVE_STABLE_S` | 180 | a card must have had one slot's worth free this long, so a momentary gap between another user's jobs is never taken |
| `D2L_GPU_LIVE_SETTLE_S` | 90 | pause on a card after a dispatch while the new job allocates |

**Shared hosts.** The slot pools are static: they assume the GPUs belong to the
build. When other users' jobs share the cards, set `D2L_GPU_LIVE_GATE=1` (and,
to be conservative, fewer slots per card). CPU notebooks run as usual; GPU
notebooks wait, re-checking every 15 s, until a card has held enough free
memory for `D2L_GPU_LIVE_STABLE_S`. The scheduler prints the live free memory
every 10 minutes. Used for the 2026-10 full re-execution, when another user's
vLLM fleet occupied about 23 GiB of every card:

```bash
D2L_GPU_LIVE_GATE=1 make run-all-notebooks GPU_SLOTS_PER=2,2,2,2
```

Examples:

```bash
make run-all-notebooks                     # everything, autodetected
make run-notebooks-jax                      # one framework
GPU_MIB_PER_SLOT=15360 make run-all-notebooks   # 15 GiB/slot (fewer, bigger slots)
python tools/notebook_scheduler.py --dry-run --force-all   # plan + invariant check
python tools/notebook_scheduler.py --files "chapter_preliminaries/ndarray.md"  # one notebook, all fw
```

---

## 6. Observability & failure handling

- Per-notebook lines: `[fw] chapter/x: [GPU 0,1] running (scheduled)` then
  `OK (Ns) [GPU 0,1] (k/total done)` — the device(s), and a running completed
  count.
- A final summary: `=== scheduler done: N/total ok, M failed ===` and a
  `FAILED:` list. Full per-notebook output is in `logs/run-<fw>-<ts>.log`; the
  cause of a failure is dumped to `logs/nb-errors/<fw>/<rel>.log`.
- A failed notebook leaves **no stamp**, so a re-run picks it up; one failure does
  **not** block other notebooks (there are no inter-notebook dependencies).
- `--dry-run` prints peak per-GPU and CPU usage and asserts they never exceed the
  pools (`INVARIANTS OK`).

---

## 7. History — what was tried and removed

Earlier revisions over-engineered this and caused real failures; recording the
dead ends so they aren't reintroduced:

- **Two background `make -jN` queues + per-notebook flock** (original). Coupled
  the job count to the slot count and left a per-framework tail.
- **Framework-interleaved order + per-relpath mutex + framework-fair dispatch +
  multi-GPU gating.** This let all four frameworks run the same notebook
  "sequenced" by a lock — but the lock at the wrong layer, plus concurrent
  `make` processes, produced the `cifar10` data race and the `d2l/*.py`
  preamble-wipe race. Replaced by the framework-grouped queue, which makes those
  races impossible by construction with far less machinery.
- **GPU pair-packing for multi-GPU notebooks.** Replaced by the general per-GPU
  slot model (a 2-GPU notebook just reserves slots on two GPUs like any other
  requirement), so multi-GPU notebooks mix freely with the rest.

See also: `docs/build-system.md` §6.7 (summary), §6.8 (why HTML render is a single
`quarto render`, not parallel), and §6.7's MXNet cross-thread-CUDA gotcha.
