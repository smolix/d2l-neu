# PyTorch / JAX API currency audit for the rebuilt "Computational Performance" chapter

Scope: audit current (mid-2026) PyTorch and JAX APIs for perf/compilation/multi-GPU
training, **at the repo's exact pinned versions**, and assess in-notebook feasibility
on the book's actual 4×RTX 4090 build box. All claims below are either (a) sourced
from official docs/release notes (cited), or (b) **empirically verified on this box**
in this session — those are marked **[VERIFIED HERE]** with the command/output that
proves it. Test artifacts live in
`/tmp/claude-4002/-home-smola-d2l-neu/0b526508-b3d5-4a42-8bee-4374612287b8/scratchpad/nbtest/`.

No repo file was modified.

## 0. Pinned versions (from `pyproject.toml`) and box facts

| | value |
|---|---|
| torch | **2.11.0** (`+cu128` on linux, `pytorch-cu128` index) |
| torchvision | **0.26.0** |
| jax | **0.10.2** (`jax[cuda12]`) |
| flax | **0.12.7** (NNX stack) |
| optax | **0.2.8** |
| orbax-checkpoint | **0.12.0** |
| GPUs | 4× NVIDIA GeForce **RTX 4090**, 24564 MiB each |
| driver / CUDA | 595.71.05 / **CUDA 13.2** [VERIFIED HERE, `nvidia-smi`] |
| CPU / RAM | 64 logical cores, 251 GiB RAM [VERIFIED HERE, `nproc`/`free -h`] |
| notebook execution | `jupyter nbconvert --to notebook --execute --inplace` in a **subprocess per notebook**, one ipykernel process, via `tools/run_notebooks.py:execute_notebook` — **not** `torchrun`, no outer MPI/multi-process launcher |

## 1. Hardware topology and the NCCL/P2P story — the central constraint

**[VERIFIED HERE]** `nvidia-smi topo -m` on the actual build box:

```
        GPU0  GPU1  GPU2  GPU3
GPU0     X    PHB   NODE  NODE
GPU1    PHB    X    NODE  NODE
GPU2    NODE  NODE   X    PHB
GPU3    NODE  NODE  PHB    X
```

No `NV#` anywhere — **no NVLink**. Best case (GPU0↔GPU1, GPU2↔GPU3) is `PHB` (through
the CPU's PCIe host bridge); the cross pairs are `NODE` (through the NUMA
interconnect as well). This is a consumer motherboard with two GPUs behind each of
two root ports, not a server board with a PCIe switch fabric — no `PXB`/`PIX` either.

**[VERIFIED HERE]** `nvidia-smi topo -p2p r` — every pair reports **`CNS` (chipset
not supported)**: P2P is unavailable between every GPU pair on this box. Confirmed a
second way at the CUDA level:

```python
>>> torch.cuda.can_device_access_peer(0, 1)  # and every other pair
False
```

This matches NVIDIA's documented driver-level policy: **P2P (and NVLink) is disabled
on GeForce/RTX consumer cards** (Ada/RTX 40-series included) — it is not a topology
accident, it is a deliberate segmentation between GeForce and workstation/datacenter
SKUs. Community workarounds exist (patched kernel modules, `NCCL_P2P_LEVEL` env
tricks) but they are out of scope for a book that must build unattended on stock
drivers — and several NCCL/nccl-tests GitHub issues report `all_reduce` **hangs** on
4090s until P2P is explicitly disabled, i.e., the disabled state is the *safe* one.
(Sources: [NCCL#1614](https://github.com/NVIDIA/nccl/issues/1614),
[NCCL#814](https://github.com/NVIDIA/nccl/issues/814),
[nccl-tests#117](https://github.com/NVIDIA/nccl-tests/issues/117),
[Puget Systems](https://www.pugetsystems.com/labs/hpc/problems-with-rtx4090-multigpu-and-amd-vs-intel-vs-rtx6000ada-or-rtx3090/).)

**[VERIFIED HERE]** Real NCCL allreduce numbers on this box (`torch.distributed`,
NCCL backend, 256 MB fp32 tensor, ring/tree channels, `NCCL_DEBUG=INFO`):

```
world_size=2  time=119ms  algbw=2.25 GB/s  busbw=2.25 GB/s
world_size=4  time=187ms  algbw=1.43 GB/s  busbw=2.15 GB/s
```

NCCL log confirms the mechanism: `P2P is disabled between connected GPUs 1 and 0`
(repeated for every pair), `NVLS multicast support is not available`, `isAllDirectP2p
0 isAllCudaP2p 0` — NCCL builds its usual ring/tree channels but every hop is a
**staged copy through host (pinned) memory**, not a direct GPU-GPU transfer. Bus
bandwidth **does not improve going from 2→4 GPUs** (2.25→2.15 GB/s busbw) because the
bottleneck is the per-GPU PCIe↔host staging step, not link count. For comparison: a
plain `.to(device)` cross-device copy (no NCCL) also works — CUDA transparently
falls back to a host-staged `cudaMemcpy` when P2P access isn't available — but it's
the same slow path, not a distinct mechanism.

**Teaching-level takeaway:** on this box, **allreduce bandwidth is ~2–2.5 GB/s
regardless of GPU count**, roughly two orders of magnitude below an NVLink box
(hundreds of GB/s) and well below what you'd get even on a P2P-capable
PCIe-Gen4-only server (~20–25 GB/s). This is real and worth teaching explicitly
(it's a great, honest illustration of "why NVLink/NVSwitch exist" and "why datacenter
GPUs disable this segmentation") — but it also means **any multi-GPU demo whose story
is "look how much faster 4 GPUs are" will be communication-bound and unconvincing**
unless the per-step compute is large enough to hide a ~2 GB/s allreduce (see §7).
JAX's collectives (via XLA's NCCL client) hit the same P2P-disabled ceiling — see §3.

## 2. `nbconvert` + `torch.multiprocessing` — the hard blocker, empirically confirmed

The book's build has no `torchrun`/`mp.spawn`-friendly launcher: each notebook is one
ipykernel process started by `jupyter nbconvert --execute`. This matters enormously
for anything that needs `torch.distributed` (DDP, FSDP2), because those need
multiple *processes*, which inside a notebook means spinning them up from a cell with
`torch.multiprocessing`.

**[VERIFIED HERE]** Built two real notebooks (kernel `d2l-pytorch`, executed exactly
as the pipeline does: `jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.kernel_name=d2l-pytorch`), with the worker function defined
inline in the cell (as every d2l notebook cell is — no importable helper module):

**(a) `mp.spawn(worker, ...)` (PyTorch's documented DDP idiom) → fails:**
```
AttributeError: Can't get attribute 'worker' on <module '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>
```
`mp.spawn` hardcodes `start_method='spawn'`, which re-executes `__main__` in the
child to unpickle the target function. Under `nbconvert`/`ipykernel`, `__main__` is
the kernel launcher module, not the notebook's cell namespace — so a function defined
in a cell is **not picklable by reference** and every child crashes at unpickle time.
This is not a build-box quirk; it is inherent to how `ipykernel` executes cells, and
it reproduces the well-known "can't pickle local object" class of multiprocessing
error that anyone doing `mp.spawn` from Jupyter hits.

**(b) `torch.multiprocessing.start_processes(fn, ..., start_method='fork')`, called
before any CUDA context exists in the parent → works:**
```
main pid 58717 cuda initialized before spawn: False
done
result=[3.0, 3.0, 3.0, 3.0] pid=58942
```
`fork` duplicates the whole process image (including the live `worker` function in
the kernel's namespace), so pickling is not needed — this is the idiom that survives
`nbconvert`.

**(c) Same fork call, but preceded by one cell that touches CUDA in the parent
(`torch.ones(1, device='cuda:0')`) → fails:**
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```
This is the standard CUDA-fork hazard, and it is exactly the trap a real d2l notebook
falls into: **`torch.cuda.device_count()` / `torch.cuda.is_available()` do *not*
initialize a CUDA context [VERIFIED HERE]**, but the very next thing most notebooks
do — `torch.ones(1, device='cuda')`, `d2l.try_gpu()` used to move a tensor, a
single-GPU baseline run in the main process — **does**, and that permanently
forecloses `fork`-based multiprocessing for the rest of that kernel's life.

**Net conclusion — this is the load-bearing fact for the whole multi-GPU section
of the rebuild:**

- `mp.spawn` (the API PyTorch's own DDP tutorials tell you to use) is **unusable
  as-written** inside this build's notebooks.
- The workaround (`start_processes(..., start_method='fork')`) **works, but only if
  the notebook is disciplined about never touching CUDA in the main/parent process
  before the fork call** — including no earlier "let's check `torch.cuda
  .device_count()` and then run a single-GPU baseline for comparison" cell, which is
  precisely the structure of the *existing* `multiple-gpus-concise.md` (single-GPU
  `train()` call in the main process, executed directly, *before* the 2-GPU cell).
  A DDP/FSDP2 rebuild must either (i) run *every* GPU-touching cell — including the
  single-GPU baseline — through the same fork-based multiprocess harness, or (ii) put
  the baseline in a separate notebook. Document this prominently; it is exactly the
  kind of gotcha that will silently break on the next `torch` bump if unstated.
- This constraint doesn't exist for the *from-scratch*, no-`torch.distributed`
  pattern the current `multiple-gpus.md` already uses (explicit `.to(device)` copies,
  manual gather/broadcast in a Python loop, all in one process, no NCCL). That
  pattern is unaffected by any of this and **[VERIFIED HERE]** still runs correctly
  today (cross-device `.to()` copy tested, correct result, uses the same P2P-less
  staged-copy fallback CUDA provides transparently).

## 3. JAX: fully notebook-friendly, confirmed

**[VERIFIED HERE]**: a single `.venv-jax` process sees all 4 GPUs with no
multiprocessing:
```python
>>> jax.devices()
[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
>>> jax.local_device_count()
4
```
`Mesh` / `NamedSharding` / `PartitionSpec` / `jax.device_put(..., sharding)` /
`jax.shard_map` all ran correctly in-process **[VERIFIED HERE]**. A `shard_map`-based
psum allreduce over the same 256 MB payload used for the NCCL test above gave:
```
k=2  time=60ms  busbw=4.46 GB/s
k=4  time=47ms  busbw=8.57 GB/s
```
i.e., noticeably **better than raw `torch.distributed`+NCCL on identical hardware**
(4.5–8.6 GB/s vs. 2.15–2.25 GB/s), though still nowhere near an NVLink box, and it
scaled 2→4 (unlike the PyTorch number) — XLA's collective lowering is evidently doing
something more effective with the same P2P-less transport (possibly better chunking/
pipelining across more channels). One benign warning appears every run and is safe to
mention-and-ignore in the book: `Failed to register GPU memory with clique ...
ncclCommRegister ... 'named symbol not found'` — this is NCCL's zero-copy buffer
*registration* fast path failing (unsupported on this driver/GPU combo for the P2P-
less case) and falling back; **execution still completes with a correct result**.

## 4. PyTorch API currency table (at 2.11.0)

| Area | Current/recommended | Deprecated/legacy | Notes |
|---|---|---|---|
| Graph compilation | `torch.compile()` (modes `default`, `reduce-overhead`, `max-autotune`) | `torch.jit.script` / `torch.jit.trace` (**TorchScript**) | **[VERIFIED HERE]**: `torch.jit.script(net)` on 2.11 emits `DeprecationWarning: torch.jit.script is deprecated. Please switch to torch.compile or torch.export.` **This directly hits the existing repo**: `chapter_computational-performance/hybridize.md` teaches `torch.jit.script`/`net.hybridize()`-style symbolic programming as PyTorch's "hybridize" story — that content is built on a now-deprecated API and should be rewritten around `torch.compile` in the rebuild (out of this task's file scope, but flagging is in scope: it's the #1 concrete kill-list item). |
| torch.compile maturity | Default mode is stable/mature for both inference and training as of 2.11; `reduce-overhead` (CUDA-graph based) is the standard choice once shapes are static; `max-autotune` for kernel-search when compile-time budget allows. Regional compilation (`torch.compile` applied per repeated block, e.g. each transformer layer) is the documented way to cut cold-compile time on multi-layer models. Dynamic-shape support continues to mature (2.11 adds dynamic-shape LSTM tracing/export). | — | 2.11 highlights: FlexAttention gets a FlashAttention‑4 backend on **Hopper/Blackwell only** — not applicable on this box's Ada (sm_89) RTX 4090s. `TORCH_COMPILE_OVERRIDE_BACKENDS` added for graph-by-graph backend bisection. Source: [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/). |
| CUDA graphs | `torch.compile(mode="reduce-overhead")` (CUDAGraph Trees — shared memory pool, default) for the common case; `torch.cuda.graph` / `torch.cuda.make_graphed_callables` for hand-rolled capture of a custom training loop. | — | CUDA graphs need **static shapes**; not worth teaching as a separate primitive when `reduce-overhead` gets you 90% of it declaratively — good candidate to mention, not demo standalone. |
| Profiling | `torch.profiler.profile(activities=[CPU,CUDA])` + `prof.export_chrome_trace(...)` (view at `chrome://tracing`) or `torch.profiler.tensorboard_trace_handler(...)` (mutually exclusive with chrome export in one run). | legacy `torch.autograd.profiler` namespace | Stable, current API; good notebook fit — traces are just files, no distributed anything. |
| GPU timing | `torch.cuda.Event(enable_timing=True)` + `.record()`/`.synchronize()`/`elapsed_time()`, or wall-clock + `torch.cuda.synchronize()`. | — | Unchanged, standard idiom. |
| Memory profiling | `torch.cuda.memory_allocated()`, `max_memory_allocated()`, `torch.cuda.memory._record_memory_history()` + `torch.cuda.memory._dump_snapshot("f.pickle")`, then drag-and-drop into **pytorch.org/memory_viz**, or generate an HTML locally via `torch/cuda/_memory_viz.py`. | — | `_dump_snapshot`/`_record_memory_history` are underscore-prefixed ("private" but the documented/blessed way — see [PyTorch's own memory-viz blog series](https://pytorch.org/blog/understanding-gpu-memory-1/)). Notebook-friendly: writes a local file, no server needed for the basic allocation-over-time plot (the interactive HTML viewer is a static page). |
| Mixed precision | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`; **omit `GradScaler`** for bf16 (only fp16 needs loss scaling, since fp16 has a much narrower exponent range and gradients can underflow to zero). | fp16 + `GradScaler` still correct but only needed on GPUs without native bf16 (pre-Ampere) — irrelevant here, Ada supports bf16 natively. | **[VERIFIED HERE]**: bf16 autocast vs. TF32 fp32 on a 3-conv-layer/256-batch toy model gave **1.93× speedup** (52.7 ms → 27.3 ms/step) — real, robust, book-quotable effect size (see §7). |
| TF32 / fp32 matmul precision | `torch.set_float32_matmul_precision('high')` (or `torch.backends.cuda.matmul.fp32_precision = 'tf32'` / `torch.backends.cudnn.conv.fp32_precision = 'tf32'`). **Default as of recent PyTorch is `'highest'` — i.e. TF32 is OFF by default for matmul.** | `torch.backends.cuda.matmul.allow_tf32 = True` / `torch.backends.cudnn.allow_tf32` (bool flags) — deprecated after 2.9, replaced by the string-valued `fp32_precision` setters. | **[VERIFIED HERE]**: compiling a conv net without setting this emits inductor's own warning: `TensorFloat32 tensor cores ... available but not enabled. Consider setting torch.set_float32_matmul_precision('high')`. This is an easy "free" 2–3× matmul win the book should call out explicitly as a one-line prerequisite before any perf comparison, or risk misleadingly slow "eager fp32" baselines. |
| Memory layout | `model.to(memory_format=torch.channels_last)` + NHWC-friendly conv inputs — still current best practice for conv-heavy models under cuDNN/AMP. | — | Cheap, real, orthogonal win; pairs well with AMP demo. |
| Attention kernels | `torch.nn.functional.scaled_dot_product_attention` dispatches among Flash / cuDNN-attention / memory-efficient / math backends automatically; `torch.nn.attention.sdpa_kernel(...)` context manager for explicit backend pinning. | — | On this box's Ada GPUs, standard FlashAttention-2-class and cuDNN-attention backends apply; FlashAttention-4 (2.11's new FlexAttention backend) targets Hopper/Blackwell only, not usable here. |
| Activation checkpointing | `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` — **`use_reentrant` must now be passed explicitly** (an unset value has been an error since 2.9). | `use_reentrant=True` (reentrant/nested-autograd variant) — still available, but documented as incompatible with some DDP/FSDP configurations; non-reentrant is the recommended default. | Direct API-currency fix vs. any pre-2.9-era d2l snippet that calls `checkpoint(fn, *args)` with no `use_reentrant` kwarg. |
| Data-parallel wrapper | `torch.nn.parallel.DistributedDataParallel` (DDP) — **"recommended... even if there is only a single node"** per PyTorch's own docs. | `torch.nn.DataParallel` — **not formally removed** in 2.11 (a 2021 RFC proposed deprecating it, never fully executed) but the docstring explicitly steers to DDP; single-process, thread-based, GIL-bound, doesn't scale past ~a couple GPUs. **[VERIFIED HERE]**: `nn.DataParallel(model, device_ids=[0,1])` still runs without error or warning on 2.11. | The *existing* `multiple-gpus-concise.md` already uses `nn.DataParallel` with an inline comment noting DDP is preferred "for production" — that comment is accurate and can stay, but the rebuild should seriously consider leading with DDP instead now that mp.spawn-under-fork is a known-working (if fussy) pattern (§2, §7). |
| Sharded/large-model training | **FSDP2**: `torch.distributed.fsdp.fully_shard` + `DeviceMesh` + `DTensor`. In-place module conversion (no wrapper shell), per-parameter dim-0 sharding, `HSDP` via a 2D mesh (`Replicate()`, `Shard(0)`). | **FSDP1** (`torch.distributed.fsdp.FullyShardedDataParallel` wrapper class) — multiple independent sources describe it as **deprecated as of PyTorch 2.11.0+**, not backward compatible with FSDP2, scheduled for eventual removal. | `DeviceMesh`/`DTensor` themselves are the stable substrate FSDP2 is built on (production-used in TorchTitan); standalone Tensor-Parallel APIs on top of DTensor are still marked experimental — irrelevant for a minimal single-node FSDP2 demo. |
| Streams / async | `torch.cuda.Stream()`, `with torch.cuda.stream(s):`, `torch.cuda.Event` for cross-stream sync, `.record_stream()` for tensors crossing streams. | — | Unchanged, standard; already partly covered conceptually (not at the streams-API level) in `async-computation.md`. |
| Host↔device transfer | `tensor.to(device, non_blocking=True)` from `pin_memory=True` host tensors (or `DataLoader(pin_memory=True)`); overlap with compute via streams. | — | Unchanged, standard. |
| Collective API shape | `torch.distributed.init_process_group(backend='nccl', ...)`, then `dist.all_reduce`, `dist.all_gather`, `dist.broadcast`, `dist.barrier`; new in 2.11: **differentiable functional collectives** (API-unstable) allowing backprop through a collective op directly. | — | For teaching, `dist.all_reduce(tensor, op=ReduceOp.SUM)` is the one line that matters; world_size=1 "distributed" runs are possible but pedagogically empty (single rank, no real communication) — don't bother demoing that combination. |

## 5. JAX API currency table (at jax 0.10.2 / flax 0.12.7)

| Area | Current/recommended | Deprecated/legacy | Notes |
|---|---|---|---|
| JIT | `jax.jit(f, static_argnums=..., donate_argnums=...)`. `donate_argnums` lets XLA reuse an input buffer's memory for the output (real memory savings, e.g. in-place-style optimizer updates); `static_argnums` forces retracing per distinct value (use for small/hashable control-flow-affecting args, not shapes). | — | Standard, mature, no change at this pin. |
| AOT compilation | **[VERIFIED HERE]** 3-stage API: `traced = jax.jit(f).trace(*args)` → `lowered = traced.lower()` → `compiled = lowered.compile()` → `compiled(*args)`. `compiled.cost_analysis()` returns a dict with a real `'flops'` estimate (**[VERIFIED HERE]**: `268697600.0` flops for a 512×512 matmul+sum — matches the analytic `2*512^3 + …` order of magnitude) plus bytes-accessed/utilization fields — genuinely useful for a roofline demo without running anything. | `jax.xla_computation` (older AOT entry point) | Good notebook fit: purely single-process, deterministic, fast. |
| Timing idiom | Always call `.block_until_ready()` (or `jax.block_until_ready(pytree)`) before stopping a wall-clock timer — JAX dispatch is async. | — | Unchanged; already used correctly in the existing chapter's jax cells. |
| Profiling | `jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True)` context manager (view in Perfetto/TensorBoard/XProf); `jax.profiler.device_memory_profile()` for a memory snapshot (pprof-compatible), always-on instrumentation under the hood. | — | Notebook-friendly, single process, no distributed setup needed. |
| Multi-device parallelism ("modern") | `jax.jit` + **automatic parallelization**: shard inputs with `jax.device_put(x, NamedSharding(mesh, PartitionSpec(...)))`, jit the function, XLA infers/compiles the distributed execution plan and inserts communication automatically. For **explicit** control over communication and collectives inside the traced function: `jax.shard_map(f, mesh=..., in_specs=..., out_specs=...)` (now a **top-level** `jax.shard_map`, moved out of `jax.experimental` — **[VERIFIED HERE]** `from jax import shard_map` works at 0.10.2; the old `jax.experimental.shard_map` import still works but **[VERIFIED HERE]** raises `DeprecationWarning: jax.experimental.shard_map is deprecated in v0.8.0. Use jax.shard_map instead.`). | `jax.pmap` — **as of JAX 0.8.0, `pmap`'s C++ implementation was replaced**; it now internally re-lowers to `jax.jit` + `jax.shard_map` and, per JAX's own migration guide, "**the new implementation is not a perfect replacement for the original**" (rank-reducing vs. rank-preserving semantics differ, implicit resharding differs). Migration window for the *old* pmap behavior (`JAX_PMAP_SHMAP_MERGE=0` opt-out) was documented to close **Jan 15, 2026** — already past as of this audit. **[VERIFIED HERE]**: `jax.pmap` still runs with no warning at 0.10.2 (it's the compat shim working correctly), but it is functionally legacy now. | **This directly hits the existing repo**: both `multiple-gpus.md` (`jax.pmap(..., axis_name=...)`) and `multiple-gpus-concise.md` (`nnx.pmap`) build their JAX multi-GPU story entirely on `pmap`. Per JAX's own docs this is now a compatibility wrapper around `jit`+`shard_map`, not the primitive itself — the rebuild should lead with `jit`+`NamedSharding` (implicit) and/or `shard_map` (explicit collectives), matching what `docs.jax.dev`'s current parallelism docs teach. |
| Flax NNX + sharding | `nnx.jit` (shard-inference from input `jax.Array` shardings, defaults to replicate if unspecified) combined with `jax.lax.with_sharding_constraint` inside the traced function, or plain `jax.jit` over NNX's split/merge (`nnx.state`/`nnx.update`) functional core. | `nnx.pmap` / `nnx.vmap`-per-replica pattern (what the existing `multiple-gpus-concise.md` uses to build one model+optimizer replica per device) — works, but is downstream of the `pmap` legacy status above. | Flax's own docs (`flax.readthedocs.io/.../flax_gspmd.html`) now lead with the jit+sharding story for scaling NNX modules, not pmap. |
| Activation checkpointing | `jax.checkpoint(f)` (alias `jax.remat(f)` — **[VERIFIED HERE]** both exist, both work, functionally identical, not literally the same object but no deprecation warning on either). | — | Straightforward, notebook-friendly; good pairing with a memory-profiling demo. |
| Precision control | `jax.default_matmul_precision('bfloat16'|'tensorfloat32'|'float32')` (context manager or global), or pass `precision=` to `jnp.dot`/`lax.dot_general` per-call. | — | Direct JAX analogue of PyTorch's TF32 story — worth a one-line side-by-side in the book. |
| Memory stats | `jax.profiler.device_memory_profile()` (pprof format), or informally inspect `x.devices()` / `jax.live_arrays()` for what's currently resident. No PyTorch-style `max_memory_allocated()` running counter — the model is closer to "always-on stack-trace-tagged sampling profiler" than a manually-queried counter. | — | Framing this difference explicitly (JAX: profile-based; PyTorch: counter + snapshot-based) is a good pedagogical contrast. |
| XLA flags | `XLA_FLAGS` env var, e.g. `--xla_gpu_enable_latency_hiding_scheduler=true`; **must be set before the first JAX op / backend init** — setting it mid-notebook after JAX has already run anything has no effect. | — | Not recommended as a teaching demo (fiddly, hardware/version-specific, easy to get a silently-ignored flag). Worth one sentence ("here be dragons, set before first import") rather than a code cell. |
| Collectives inside `shard_map` | `jax.lax.psum`, `jax.lax.pmean`, `jax.lax.all_gather`, `jax.lax.all_to_all` — same primitives as under `pmap`, now invoked inside a `shard_map`-transformed function against a named `Mesh` axis instead of an implicit `pmap` axis. | — | **[VERIFIED HERE]** a `shard_map`+`psum` allreduce over a `Mesh` ran correctly (§3) — this is the idiom to teach. |

## 6. Collectives at a teaching level

- **NCCL ring vs. tree**: ring allreduce is bandwidth-optimal at large message size
  (each of $k$ GPUs sends/receives $2(k-1)/k$ of the data — the "busbw" formula used
  above); tree algorithms cut latency at small message size / large GPU counts (log
  depth vs. linear). NCCL auto-selects per message size; on this box the choice is
  moot because the **staged host copy**, not the ring/tree topology, is the
  bottleneck (§1) — a good "in theory vs. in practice" teaching moment.
- **`torch.distributed` collective shape**: `init_process_group(backend, rank,
  world_size)` once per process, then `dist.all_reduce(tensor, op=ReduceOp.SUM)` /
  `all_gather` / `broadcast` / `barrier`, `destroy_process_group()` at the end. DDP
  and FSDP2 both build on exactly this collective layer — worth showing the raw
  collective once (as the existing from-scratch section does, just without NCCL
  today) before introducing DDP/FSDP2 as "the same collectives, orchestrated for
  you."
- **JAX collectives**: no separate init step — collectives are just primitives
  (`psum`, `pmean`, ...) invoked *inside* a `shard_map`-transformed function against a
  named mesh axis; the "process group" concept doesn't exist for single-controller,
  single-node JAX the way it does for `torch.distributed`. This is a genuine, teachable
  API-philosophy contrast between the two frameworks.

## 7. Eight candidate demo areas — feasibility, wall-time, effect size, minimal code

Framework coverage per CLAUDE.md's Advanced-part policy: **PyTorch + JAX only** (no
new TF/MXNet content; this chapter is in the Advanced part).

### 1. Benchmark eager vs. `torch.compile` / `jax.jit`
**Feasibility: solid, in-notebook, single process, no GPU-count dependency.**
Wall-time: seconds (compile) + seconds (timed loop) — well under a minute total.
Effect size: **[VERIFIED HERE]** on a small 2-conv-layer/128-batch model, eager
6.24 ms/step vs. compiled steady-state 4.74 ms/step = **1.32×** (first compiled call
took 3.7 s — show that cost explicitly, it's part of the lesson). This is a modest,
somewhat model-size-dependent number — pick a meatier model (e.g. the book's
ResNet-18-ish network already in `multiple-gpus-concise.md`) for a more convincing
ratio, and **always call `torch.set_float32_matmul_precision('high')` first** or the
eager baseline is needlessly slow and the comparison is polluted by a config choice,
not compilation. Robust/reproducible: yes — compile speedups from kernel fusion are
stable across re-executions (not stochastic), though the *magnitude* will drift a bit
with model/batch size, so quote a range, not a single decimal (per CLAUDE.md's
prose-precision rule).

```python
# PyTorch
torch.set_float32_matmul_precision('high')
model = build_model().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
def step(m, x):
    opt.zero_grad(set_to_none=True)
    loss = m(x).sum(); loss.backward(); opt.step()
for _ in range(10): step(model, x)          # warmup
torch.cuda.synchronize(); t0 = time.time()
for _ in range(50): step(model, x)
torch.cuda.synchronize(); eager_t = (time.time()-t0)/50

cmodel = torch.compile(model)
step(cmodel, x); torch.cuda.synchronize()   # 1st call: compiles (~seconds)
for _ in range(10): step(cmodel, x)         # warmup compiled path
torch.cuda.synchronize(); t0 = time.time()
for _ in range(50): step(cmodel, x)
torch.cuda.synchronize(); compiled_t = (time.time()-t0)/50
```
```python
# JAX
def loss_fn(params, x): ...
grad_fn = jax.jit(jax.grad(loss_fn))
grad_fn(params, x)  # warmup / triggers compile
x.block_until_ready()
t0 = time.time()
for _ in range(50): g = grad_fn(params, x)
jax.block_until_ready(g); jit_t = (time.time()-t0)/50

nojit_fn = jax.grad(loss_fn)  # eager
t0 = time.time()
for _ in range(50): g = nojit_fn(params, x)
jax.block_until_ready(g); eager_t = (time.time()-t0)/50
```

### 2. Profile a training step (`torch.profiler` / `jax.profiler`)
**Feasibility: solid, in-notebook, single process.** Wall-time: seconds to run,
trace files are small (KBs–low MBs) for a handful of steps. Effect size: N/A (this is
a diagnostic demo, not a speedup demo) — the "result" is a trace artifact / table of
top ops, which is exactly reproducible in *shape* (same ops dominate) even if exact
microsecond numbers wobble run to run; fine for a book that re-executes regularly as
long as the prose doesn't quote specific microsecond figures.
```python
# PyTorch
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True) as prof:
    for _ in range(5):
        step(model, x)
        prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```
```python
# JAX
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
    for _ in range(5):
        loss = grad_fn(params, x)
    jax.block_until_ready(loss)
# view with: tensorboard --logdir /tmp/jax-trace  (or the XProf viewer)
```

### 3. Roofline-style matmul FLOPs sweep
**Feasibility: excellent — the single cleanest demo in this list.** Purely
single-process, single-GPU, no communication, no multiprocessing gotchas at all.
Wall-time: a handful of seconds per size, a sweep of ~8 sizes is well under a minute.
Effect size: **highly robust** — measured TFLOP/s vs. matrix size is a real hardware
characteristic (ramp-up to peak as size grows, memory-bound at small size,
compute-bound at large size), reproducible run to run within a few percent. JAX's
`cost_analysis()['flops']` (**[VERIFIED HERE]**, exact analytic FLOP count with no
execution) makes a particularly elegant version: compare the *analytic* FLOP count
against measured wall-clock time to get achieved TFLOP/s, no manual FLOP-counting
formula needed.
```python
# PyTorch
for n in [256, 512, 1024, 2048, 4096, 8192]:
    a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    for _ in range(3): torch.mm(a, b)          # warmup
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): c = torch.mm(a, b)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    tflops = 2 * n**3 / dt / 1e12
    print(n, f'{tflops:.1f} TFLOP/s')
```
```python
# JAX
for n in [256, 512, 1024, 2048, 4096, 8192]:
    a = jnp.array(np.random.randn(n, n), dtype=jnp.bfloat16)
    b = jnp.array(np.random.randn(n, n), dtype=jnp.bfloat16)
    f = jax.jit(jnp.dot)
    compiled = f.lower(a, b).compile()
    flops = compiled.cost_analysis()['flops']   # exact, no manual formula
    compiled(a, b).block_until_ready()          # warmup
    t0 = time.time()
    for _ in range(10): c = compiled(a, b)
    c.block_until_ready()
    dt = (time.time() - t0) / 10
    print(n, f'{flops/dt/1e12:.1f} TFLOP/s')
```

### 4. AMP speedup (bf16 autocast)
**Feasibility: excellent.** Wall-time: seconds. Effect size: **[VERIFIED HERE]**
**1.93×** (52.7 ms → 27.3 ms/step) on a modest 3-conv-layer/256-batch model with TF32
already enabled on the fp32 baseline (i.e. this is bf16-vs-TF32, the honest
comparison, not bf16-vs-unaccelerated-fp32 which would inflate the number
artificially). This is a robust, reproducible, book-quotable win — Ada's tensor cores
give a genuine ~2× on compute-bound conv/matmul workloads, and the ratio is stable
run to run (no stochastic element). Recommend explicitly contrasting against an
*unfair* fp32-without-TF32 baseline once, to make the TF32 lesson from item 1 land,
then use the fair TF32 baseline for the "real" AMP number.
```python
# PyTorch — see §4 table row for GradScaler guidance (omit for bf16)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(x).sum()
loss.backward(); opt.step()
```
```python
# JAX — precision is a config, not a wrapper
with jax.default_matmul_precision('bfloat16'):
    loss, grads = jax.value_and_grad(loss_fn)(params, x)
# or per-op: jnp.dot(a, b, precision=jax.lax.Precision.DEFAULT)  # bf16-class
```

### 5. Memory profiling + activation checkpointing
**Feasibility: good, in-notebook, single process.** Wall-time: seconds to run the
forward/backward passes being profiled; the snapshot file itself is small for a short
run. Effect size: activation-checkpointing memory savings are a **real, large,
robust** effect (commonly 30–60%+ peak-memory reduction at the cost of ~30% more
compute from recomputation) — good, stable book material, though the exact percentage
depends heavily on model depth/width, so report qualitatively ("recomputing
activations trades roughly Δcompute% more time for Δmemory% less peak memory") rather
than a single precise pair of numbers. The `_dump_snapshot` → memory_viz.py / hosted
viewer path is notebook-friendly (writes a file; no live server needed for the static
HTML render).
```python
# PyTorch
torch.cuda.memory._record_memory_history(max_entries=100_000)
loss = model(x).sum(); loss.backward()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
print(f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak')

def blk(x):  # a residual block, e.g.
    return residual_block(x)
y = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)  # ~half the
                                                                     # activations saved
```
```python
# JAX
def block(params, x): ...
ckpt_block = jax.checkpoint(block)   # recompute in backward instead of storing
loss, grads = jax.value_and_grad(lambda p: ckpt_block(p, x).sum())(params)
# memory: jax.profiler.device_memory_profile() -> pprof; inspect resident bytes
```

### 6. Async dispatch / streams overlap
**Feasibility: OK but the *demonstrable effect* is fragile.** Wall-time: seconds.
Effect size: **caution** — overlap wins are typically small and workload-shape
dependent (need genuinely independent, well-sized compute+copy work to see a clean
win over the Python/launch overhead noise), and PyTorch's default stream already
dispatches asynchronously (this is what `async-computation.md` already teaches
conceptually). A dedicated multi-stream demo risks showing a marginal, noisy
speedup that won't survive re-execution cleanly — **recommend keeping this
conceptual/qualitative** (as the current chapter mostly does) rather than adding a
new quantitative multi-stream benchmark cell, unless a specific H2D-copy-overlaps-
compute scenario is deliberately sized to make the win obvious and stable (large
enough copy, large enough independent compute, several repetitions averaged).
```python
# PyTorch — H2D copy overlapped with independent compute
copy_stream = torch.cuda.Stream()
with torch.cuda.stream(copy_stream):
    x_next = x_next_cpu.to(device, non_blocking=True)  # needs pinned host memory
y = model(x_current)                                    # runs concurrently
torch.cuda.current_stream().wait_stream(copy_stream)
```
JAX's dispatch is async by construction (every op returns a future-like array
immediately; `block_until_ready()` is the only place synchronization happens) — there
isn't a separate "streams API" to teach at the user level; frame this as a philosophy
difference (explicit streams in PyTorch vs. everything-is-already-async-and-XLA-
schedules-it in JAX) rather than a second code demo.

### 7. From-scratch data parallelism on 2–4 GPUs (no `torch.distributed`)
**Feasibility: excellent — the one guaranteed-safe multi-GPU-in-notebook pattern.**
This is exactly what `multiple-gpus.md` already does (manual `.to(device)` copies,
Python-loop allreduce, all single-process) and it demonstrably still runs correctly
on 2.11 (**[VERIFIED HERE]**, cross-device `.to()` copy works). Wall-time: matches
what's already captured in the store (order of a few seconds per epoch at
Fashion-MNIST/LeNet scale — 10 epochs × 2 configs is a couple of minutes). Effect
size: **honestly negative for tiny models** — the existing chapter's own captured
numbers show 2 GPUs ~30% *slower* than 1 for LeNet-on-Fashion-MNIST, and that's a
*more* honest story on this box than most boxes, precisely because §1's ~2 GB/s
staged-copy bandwidth makes the hand-rolled allreduce's overhead even more dominant
here than on a P2P-capable machine. Keep this framing — it's pedagogically correct
and now has a stronger hardware-grounded justification ("this box has no P2P/NVLink,
so naive allreduce is unusually expensive here — exactly why production DDP/FSDP
exist"). For the "second GPU pays for itself" flip promised at the end of the
existing section, a meatier model (the ResNet-18 already used in
`multiple-gpus-concise.md`) is needed — worth actually re-verifying that flip's
current numbers on this box given the topology, rather than assuming it still holds.
```python
# PyTorch — unchanged shape from the existing chapter; still the safe pattern
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```
```python
# JAX — no multiprocessing needed either way; this is where JAX's story is simplest:
# shard the batch across the mesh, jit the step, let XLA insert the collective.
mesh = jax.sharding.Mesh(jax.devices()[:k], ('d',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d'))
X, y = jax.device_put(X, sharding), jax.device_put(y, sharding)

@jax.jit
def train_step(params, X, y):
    grads = jax.grad(loss_fn)(params, X, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)  # XLA all-reduces
                                                                   # under the hood
```

### 8. End-to-end DDP / FSDP2 (PyTorch) or jax-sharded training (JAX)
**Feasibility: JAX — excellent, no caveats (§3). PyTorch — feasible but fragile**,
and this is the item to scope carefully. DDP/FSDP2 both require `torch.distributed`,
which requires multiple processes, which (§2) requires `start_processes(...,
start_method='fork')` called **before any CUDA touch in the parent kernel** —
workable, but it constrains notebook structure (the single-GPU baseline, if any, must
also go through the same subprocess harness, or into a separate notebook) and it is
exactly the kind of gotcha that breaks silently on a future torch bump if the ordering
invariant isn't documented in the notebook itself. Wall-time: a handful of epochs at
Fashion-MNIST/ResNet-18 scale, a few minutes. Effect size: **communication-bound on
this box** (§1) — expect a *real but modest* speedup at 2 GPUs for a compute-heavy
enough model, likely diminishing or flat at 4 GPUs given the flat ~2–2.5 GB/s busbw
observed regardless of GPU count; do not promise near-linear scaling, and re-verify
empirically before publishing a number (the committed store already shows this
tension for the from-scratch version — DDP will be faster than the hand-rolled
allreduce, per item 7's own logic, but not textbook-linear). Recommend: **DDP demo,
not FSDP2**, for the notebook — FSDP2 is real and current (§4 table), but its value
proposition (fit models that don't fit on one GPU) doesn't show up on toy models that
fit trivially in 24 GB, so a from-scratch-motivated audience gets a better lesson from
"DDP replaces our hand-rolled allreduce with NCCL and process parallelism" than from
FSDP2's sharding story, which needs a model large enough to matter — describe FSDP2
in prose/a code sketch as the scaling answer, but don't force a live demo of it purely
to check a box.
```python
# PyTorch DDP — must run via start_processes(fork), BEFORE any parent CUDA use
def worker(rank, world_size, ...):
    import os, torch, torch.distributed as dist
    os.environ['MASTER_ADDR'] = '127.0.0.1'; os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = DDP(build_model().to(rank), device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for X, y in shard_of(train_iter, rank, world_size):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(X.to(rank)), y.to(rank))
        loss.backward()          # DDP all-reduces gradients here automatically
        opt.step()
    dist.destroy_process_group()

torch.multiprocessing.start_processes(
    worker, args=(world_size,), nprocs=world_size, start_method='fork')
```
```python
# FSDP2 sketch (prose/code-sketch only, not necessarily a live multi-minute demo)
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh('cuda', (world_size,))
model = build_big_model().to(rank)
for module in model.transformer_blocks:      # shard per repeated block
    fully_shard(module, mesh=mesh)
fully_shard(model, mesh=mesh)                # shard the rest
```
```python
# JAX — genuinely simplest end-to-end story, still single-process
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
def shard(x): return jax.device_put(
    x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data')))
state = create_train_state(...)  # replicated by default
train_step = jax.jit(train_step_fn)  # XLA infers the collective from shardings
for epoch in range(num_epochs):
    for X, y in train_iter:
        state = train_step(state, shard(X), shard(y))
```

## 8. Kill list / must-fix and hard blockers — condensed

**Deprecated/legacy APIs the rebuild should not teach as current:**
1. `torch.jit.script` / `torch.jit.trace` (TorchScript) — **[VERIFIED]** emits
   `DeprecationWarning` on 2.11 itself, docs say use `torch.compile`/`torch.export`.
   Directly hits `chapter_computational-performance/hybridize.md`'s existing content.
2. `torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32`
   (bool flags) — deprecated after 2.9 in favor of the string `fp32_precision` API /
   `torch.set_float32_matmul_precision`.
3. FSDP1 (`torch.distributed.fsdp.FullyShardedDataParallel` wrapper) — described as
   deprecated as of 2.11.0+; FSDP2 (`fully_shard`) is the only path forward.
4. `jax.pmap` (and `nnx.pmap`) — not removed, but reimplemented as a `jit`+
   `shard_map` compatibility shim since JAX 0.8.0 with documented semantic gaps;
   current JAX docs teach `jit`+sharding / `shard_map` instead. Directly hits both
   `multiple-gpus.md` and `multiple-gpus-concise.md`'s existing JAX content.
5. `jax.experimental.shard_map` import path — deprecated, use top-level
   `jax.shard_map` (confirmed present at the pinned 0.10.2).
6. `torch.utils.checkpoint.checkpoint(fn, *args)` without an explicit
   `use_reentrant=` kwarg — has been a hard error since 2.9; any old snippet needs
   `use_reentrant=False` added.
7. `nn.DataParallel` — not removed, still runs fine, but PyTorch's own docs say
   "recommended to use DistributedDataParallel... even if there is only a single
   node"; keep it only as a one-line contrast, not the primary pattern.

**Hard blockers for the in-notebook build (not API deprecations, but constraints on
what can be demoed at all):**
- **No NVLink, no P2P, on any GPU pair on this box** — confirmed at three
  independent levels (`nvidia-smi topo -p2p`, `torch.cuda.can_device_access_peer`,
  NCCL's own log lines). All multi-GPU communication is host-staged.
- **Measured NCCL allreduce busbw ≈ 2–2.5 GB/s, flat from 2→4 GPUs** — any demo
  whose punchline is "more GPUs = proportionally faster" will not hold up; the honest
  story is communication-bound scaling, and that's worth teaching explicitly as a
  contrast with NVLink/NVSwitch datacenter boxes.
- **`mp.spawn` fails outright under `nbconvert`/ipykernel** (confirmed: pickling
  `__main__`-defined worker functions fails). The working substitute,
  `start_processes(..., start_method='fork')`, only works if the parent kernel
  process **never touches CUDA before the fork call** — a real structural constraint
  on notebook cell ordering that must be documented in-notebook, not just known by
  the author, or it will silently break the next time someone reorders cells or the
  next torch bump changes fork/CUDA interaction subtleties.
- JAX has **no such blocker** — single process sees all 4 GPUs, `jit`+sharding /
  `shard_map` work standalone with no multiprocessing, no fork/CUDA-init ordering
  constraint, and (per §3) achieved noticeably higher collective bandwidth than raw
  `torch.distributed`+NCCL on identical hardware in this session's test. This is a
  strong argument for **leading the multi-GPU section with JAX**, or at minimum
  being explicit that PyTorch's multi-process story requires care this book's build
  environment doesn't get for free.
