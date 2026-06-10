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

The tuple is `('gpu', n_gpus, slots_per_gpu)`: *n_gpus* distinct GPUs, each
contributing *slots_per_gpu* slots. A GPU slot is 7.5 GiB of VRAM, so a notebook
that peaks above 7.5 GiB on a card must be marked for 2 slots on that card.

Resolution order in `notebook_resource`:

1. `rel ∈ MULTI_GPU_NOTEBOOKS` → `('gpu', 2, TWO_GPU_SLOTS_PER.get((fw, rel), 1))`
   — a data-parallel notebook genuinely uses two GPUs; default 1 slot each
   (`2×1`), bumped to 2 each (`2×2`) for a framework that allocates >7.5 GiB on a
   card.
2. `(fw, rel) ∈ HEAVY_GPU_NOTEBOOKS` → `('gpu', 1, n)` — memory-heavy on a single
   GPU.
3. `uses_gpu` (the `file_uses_gpu` keyword heuristic, which also checks sibling
   frameworks) → `('gpu', 1, 1)`.
4. otherwise → `('cpu',)`.

### The marking tables (`tools/runtime_env.py`)

```python
MULTI_GPU_NOTEBOOKS = {            # data-parallel: need 2 GPUs
    "chapter_builders-guide/use-gpu.ipynb",
    "chapter_computational-performance/multiple-gpus.ipynb",
    "chapter_computational-performance/multiple-gpus-concise.ipynb",
    "chapter_computational-performance/auto-parallelism.ipynb",
    "chapter_computational-performance/async-computation.ipynb",
    "chapter_optimization/minibatch-sgd.ipynb",
}

HEAVY_GPU_NOTEBOOKS = {            # 2 slots on ONE gpu
    ("tensorflow", "chapter_computer-vision/ssd.ipynb"): 2,
    ("tensorflow", "chapter_natural-language-processing-pretraining/bert-pretraining.ipynb"): 2,
    ("tensorflow", "chapter_computer-vision/fine-tuning.ipynb"): 2,
}

TWO_GPU_SLOTS_PER = {             # per-(fw, rel) override: 2x2 instead of 2x1
    # e.g. ("tensorflow", "chapter_computational-performance/multiple-gpus.ipynb"): 2,
}
```

**To change a notebook's footprint:** add it to the right table (above). A
single-GPU notebook that OOMs → `HEAVY_GPU_NOTEBOOKS[(fw,rel)] = 2`. A
data-parallel notebook that OOMs at `2×1` → `TWO_GPU_SLOTS_PER[(fw,rel)] = 2`.
`TWO_GPU_SLOTS_PER` starts empty and is filled in empirically.

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
- **JAX only:** because JAX preallocates, the scheduler also sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION = clamp(spg × 7.5 GiB / card_VRAM, 0.05, 0.95)`
  so a JAX process is capped to exactly the slots it reserved (e.g. 1 slot on a
  24 GiB card → 0.31; 2 slots → 0.63). PyTorch/TF/MXNet allocate on demand and
  need no fraction.

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
| `--force-all` | off | ignore stamp freshness, run everything |
| `--dry-run` | off | simulate dispatch (no execution); checks per-GPU invariants |

Examples:

```bash
make run-all-notebooks                     # everything, autodetected
make run-notebooks-jax                      # one framework
GPU_MIB_PER_SLOT=15360 make run-all-notebooks   # 15 GiB/slot (fewer, bigger slots)
python tools/notebook_scheduler.py --dry-run --force-all   # plan + invariant check
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
