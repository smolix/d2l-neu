# JAX-side survey (relayed)

NOTE: This was produced by a sub-agent spawned during the research fan-out ("Survey
framework docs and speedrun lore"). It tried to message its parent agent directly and
failed, so the content was delivered to the main session instead. Preserved here
verbatim so the synthesis stage has it even if the parent agent's report missed it.

---

# JAX-side survey: how "computational performance" is taught (2024–2026)

Scope: single-node + single-node multi-GPU + compilation. All URLs verified live on **docs.jax.dev** (JAX's living docs — pages don't carry per-page dates, so I note feature/API maturity instead). Pedagogy classified as (a) first-principles, (b) measurement, (c) API-walkthrough.

## 1. XLA compilation via `jax.jit`
- **URL:** https://docs.jax.dev/en/latest/jit-compilation.html · thinking-in-JAX/jaxpr primer https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html · AOT https://docs.jax.dev/en/latest/aot.html · tracing https://docs.jax.dev/en/latest/tracing.html
- **What it teaches:** trace-once with tracer objects → build a **jaxpr** → lower to **XLA HLO** → XLA **fuses** into optimized device kernels → cache keyed on input **shape/dtype (avals)**. Outline: How transformations work → JIT compiling → **"Why can't we just JIT everything?"** → Marking arguments static → JIT and caching.
- **Pedagogy: (a) first-principles + (b) measurement.** Shows `jax.make_jaxpr()` so you *see* the captured graph; times SELU **5.85 ms → 659 µs (~9×)** with `%timeit`; caching demo ~439 ms vs 2.95 ms.
- **Key contrasts with torch.compile (for the chapter):** (i) **functional/trace-based, not Python-bytecode capture** → **NO graph breaks**; unsupported Python simply can't touch traced values. (ii) Python control flow can't branch on a traced value → `TracerBoolConversionError`; fix = `static_argnums`/`static_argnames` (recompiles per static value) or `lax.cond`/`lax.scan`. (iii) **recompilation on new shape/dtype** is the JAX-specific footgun. (iv) XLA **auto-fuses**, so JAX rarely needs manual fusion tricks.
- **Best teaching device:** `jax.make_jaxpr()` printout (side effects vanish → "trace captures only tensor ops; purity matters") + the 9× `%timeit`.
- **`donate_argnums`:** documented in the AOT / `jax.jit` API pages (buffer donation — reuse an input's memory for the output; JAX's in-place-memory-reuse analogue). The intro jit tutorial omits it.

## 2. Rematerialization — `jax.checkpoint` / `jax.remat`
- **URL:** https://docs.jax.dev/en/latest/gradient-checkpointing.html · deep autodiff notebook https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html
- **What it teaches:** autodiff default = **store all intermediates ("residuals")**; `jax.checkpoint` switches to **recompute-in-backward** (compute-for-memory). Outline: core concept → how autodiff saves residuals → **policies** (`dots_with_no_batch_dims_saveable`, `checkpoint_dots`, `checkpoint_name`) → offload-to-CPU + **recursive O(log D) memory** → jit/`lax.scan` integration.
- **Pedagogy: (a) first-principles + empirical inspection** via `print_saved_residuals()` (you *watch* which tensors are kept).
- **Best teaching device:** `print_saved_residuals` before/after, plus the **policy** idea — control what's saved *without editing the model* (cleaner than PyTorch's wrap-the-module `checkpoint()`). Docs explicitly note XLA optimization often *reduces the need* vs eager frameworks.

## 3. Sharding / single-node multi-GPU — **THE key contrast**
- **URLs:** intro https://docs.jax.dev/en/latest/sharded-computation.html · deep notebook https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html · manual https://docs.jax.dev/en/latest/notebooks/shard_map.html · module https://docs.jax.dev/en/latest/jax.sharding.html
- **What it teaches:** build a **Mesh** (named device grid) → describe layout with **PartitionSpec** wrapped in **NamedSharding** → place data via `jax.device_put` → **pass sharded arrays into an ordinary `jax.jit` function** → the **GSPMD** partitioner shards all intermediates and **inserts the collectives automatically**. `jax.debug.visualize_array_sharding` shows the layout. `shard_map` is the manual escape hatch: **per-shard local code with explicit collectives** (`lax.psum`, etc.).
- **Pedagogy: (a/c) — API walkthrough on one big first-principle.** Framing to steal verbatim: **"you annotate how inputs/outputs are sharded; the compiler figures out the communication."**
- **PyTorch contrast (headline of the whole multi-GPU section):** DDP/FSDP are **explicit/imperative** — you wrap the model and know an all-reduce (DDP) or reduce-scatter+all-gather (FSDP) fires; separate APIs per strategy. JAX is **declarative/compiler-driven (GSPMD)** — same physical collectives, but you never write them; **one mechanism spans data-/tensor-/FSDP-style sharding** (just change the PartitionSpec). `shard_map` ↔ explicit end; `jit`+sharding ↔ automatic end.
- **Best teaching device:** the `visualize_array_sharding` grid before/after a jitted matmul — you literally see the array split across devices and the result come back sharded.

## 4. Mixed precision
- **URLs:** DeepMind **jmp** https://github.com/google-deepmind/jmp · **MPX** paper https://arxiv.org/abs/2507.03312 (2025) · bf16 discussion https://github.com/jax-ml/jax/discussions/30106
- **What it teaches:** JAX has **no autocast and no global GradScaler**; you **set dtypes explicitly**. `jmp.Policy(compute_dtype, param_dtype, output_dtype)` casts params/activations; `jmp.LossScale` (static or **dynamic**) does fp16 loss-scaling by hand. **bf16 is the default JAX idiom** (fp32-range exponent → usually no scaling needed).
- **Pedagogy: (c) API/library** — less pedagogical than PyTorch's AMP recipe.
- **Contrast:** PyTorch = *implicit* op-by-op autocast + automatic GradScaler; JAX = *explicit* policy threaded through the model, loss-scaling only if you opt into fp16. Cleaner mental model (every cast visible), more manual.

## 5. Profiling
- **URLs:** https://docs.jax.dev/en/latest/profiling.html · device memory https://docs.jax.dev/en/latest/device_memory_profiling.html · benchmarking https://docs.jax.dev/en/latest/benchmarking.html
- **What it teaches:** `jax.profiler.trace("/tmp/…", create_perfetto_link=True)` → **Perfetto**, or **TensorBoard/XProf** (Trace Viewer, **Graph Viewer showing the HLO + sharding**, Memory Viewer). `jax.profiler.save_device_memory_profile()` → pprof for OOM hunting.
- **Pedagogy: (b) measurement/methodology.** The XProf **HLO Graph Viewer** has no PyTorch analogue — shows fusion and sharding decisions ("see what the compiler did").

## 6. Custom kernels — Pallas (the Triton analogue)
- **URLs:** https://docs.jax.dev/en/latest/pallas/index.html · quickstart https://docs.jax.dev/en/latest/pallas/quickstart.html · design https://docs.jax.dev/en/latest/pallas/design/design.html
- **What it teaches:** a **JAX kernel language** — write tile/block-level kernels in `jax.numpy`-style Python; **on GPU lowers via Triton, on TPU via Mosaic**, so kernels are **portable across GPU/TPU** (Triton itself is GPU-only). Docs: Quickstart ("Hello world"/add kernel) → grids/BlockSpecs → software pipelining → TPU/GPU backend guides (incl. matmul/FlashAttention-style example) → design notes. **Flagged experimental.**
- **Pedagogy: (c) API walkthrough, progressive.** Framing for the chapter: **because XLA auto-fuses, JAX users reach for hand-kernels far less than eager-PyTorch users**; Pallas is the escape hatch for cases (FlashAttention-like) XLA can't fuse well.

## 7. Async dispatch
- **URL:** https://docs.jax.dev/en/latest/async_dispatch.html (+ benchmarking page)
- **What it teaches:** JAX **dispatches asynchronously** (like PyTorch CUDA) — Python "runs ahead", enqueuing work so the accelerator never idles. **Consequence taught as the #1 benchmarking pitfall:** a naive timer measures only *dispatch*, not execution → you must **`.block_until_ready()`** (or pull to host).
- **Pedagogy: (a) first-principles + (b) the timing gotcha.** Best device: the wrong-vs-right timing snippet (looks instant → add `block_until_ready()` → real number). Direct parallel to PyTorch's `torch.cuda.synchronize()`.

---

## PyTorch vs JAX — pedagogical contrast table

| Topic | PyTorch | JAX | Contrast to teach |
|---|---|---|---|
| **Compilation** | `torch.compile`: **bytecode capture** (Dynamo) + guards + **graph breaks**; Inductor→Triton | `jax.jit`: **trace values → jaxpr → XLA HLO**; **no graph breaks**; recompiles on shape/dtype; static args | imperative-with-fallback vs functional-pure; graph breaks vs recompilation |
| **Sharding / multi-GPU** | **explicit** DDP (all-reduce) / FSDP (reduce-scatter+all-gather); separate APIs per strategy | **declarative** Mesh+PartitionSpec+jit; **GSPMD inserts collectives**; one API for DP/TP/FSDP; `shard_map` = manual | "write the collective" vs "annotate layout, compiler writes it" — **the headline contrast** |
| **Checkpointing** | `torch.utils.checkpoint` wraps a module | `jax.checkpoint`/`remat` + **policies**; `print_saved_residuals` | wrap-module vs functional decorator + save-policy |
| **Mixed precision** | **autocast** (implicit op dtypes) + **GradScaler** (auto) | **explicit** dtypes / `jmp.Policy`; bf16-by-default; manual loss scale | implicit/automatic vs explicit/manual |
| **Custom kernels** | Triton (GPU-only), CUDA C++ | **Pallas** (GPU via Triton, **TPU via Mosaic — portable**); needed less (XLA auto-fuses) | eager needs fusion tricks; XLA fuses for free |
| **Profiling** | `torch.profiler` + TensorBoard/Chrome trace | `jax.profiler` + Perfetto/**XProf incl. HLO Graph Viewer** | both timeline-based; JAX adds compiler-graph + sharding view |

**Two throughlines for the chapter:** (1) the deepest PyTorch↔JAX split is **sharding** — imperative collectives vs compiler-synthesized (GSPMD); build the multi-GPU section around that contrast. (2) JAX pushes work onto the **XLA compiler** (auto-fusion, GSPMD comms, remat interplay), so it teaches *fewer manual perf tricks* but *more "understand what the compiler did"* (jaxpr/HLO viewer) — a clean complement to PyTorch's checklist style.
