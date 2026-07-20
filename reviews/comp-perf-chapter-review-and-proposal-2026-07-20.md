# Review + redesign proposal: ch. 13 "Computational Performance" (2026-07-20)

> **STATUS (2026-07-20): APPROVED by Alex with five decisions** — hardware
> stays (properties here, buyers guide in ch. 29); serving softened upstream
> and deferred to the Language Models part; torchrun pilot-first; the
> mxnet/tf ride-along must NOT render on the site (use `LIB_ONLY_FILES`, not
> the ch. 19 edit proposed in §6.2/Q4); budget approved with a new 2-GPU
> demonstrability rule. The binding amendments and the full implementation
> plan are in **`reviews/comp-perf-implementation-brief.md`** — read that
> first; where it amends this document, the brief wins.

*Proposal only; no chapter edits made. Commissioned by Alex 2026-07-20: rebuild
`chapter_computational-performance/` into a top-5 teaching chapter suitable for
CMU/Stanford/Berkeley/MIT/UW. Mandate: single-node multi-GPU training in, multi-
machine distributed out (deferred to the Language Models part); compute graphs
and compilation in, with detail; teach how it is done today, not the history;
hardware background may draw on the `smolix/mlss-efficiency` deck. Grounded in
five research reports (existing-chapter review, 2024–26 curriculum survey,
framework-API audit **with live measurements on the build box**, MLSS-deck
inventory, whole-book coverage scan). All load-bearing feasibility claims below
were empirically verified on the 4×RTX 4090 box at the repo's pins (torch
2.11.0, jax 0.10.2, flax 0.12.7) — marked [measured]. Where research conflicts
or a call was needed, this document makes it and says why.*

---

## 1. Verdict and the chapter at a glance

The chapter is the oldest surviving spine in the book. Its pedagogical skeleton
is MXNet's (imperative/symbolic "hybridize", the async-engine reveal, KVStore
push/pull), its PyTorch path teaches `torch.jit.script` — which now emits a
`DeprecationWarning` on our own pinned torch 2.11 [measured] — its multi-GPU
"concise" section ships `nn.DataParallel`, its JAX tabs build on `pmap` (a
compatibility shim since JAX 0.8), and its hardware section describes Skylake,
Turing, DDR4, and Optane (discontinued 2022). Meanwhile the surrounding book
moved: ch. 10.5 fully teaches FlashAttention with a benchmark, ch. 12 teaches
parallel scans and Mamba's recompute-don't-store kernel argument, ch. 11.7 uses
XLA's `cost_analysis()` for FLOP accounting, and ch. 29 owns the modern
DDP/FSDP/torchrun library map. Salvage is roughly **25–30%** — but the salvage
includes two crown jewels (the from-scratch data-parallel implementation and
the ring-allreduce derivation) and about twenty inbound promises that make
this chapter the book's designated home for the systems story.

The rebuild is organized around one method, stated in §13.1 and executed in
every section: **measure → classify the regime (compute-, bandwidth-, or
overhead-bound) → apply the matching fix → re-measure.** The roofline /
arithmetic-intensity model is the chapter's map; every later technique is
introduced as "which bound it relieves." This is the consensus spine of every
strong 2024–26 treatment (CS336, CMU 15-442, the DeepMind scaling book, Horace
He), and no existing course or text teaches the full loop end-to-end in
runnable notebooks — that gap is the chapter's differentiator.

**Proposed structure — 7 sections + index (4 new files, 2 rewritten in place,
3 deleted):**

| § | File | Title (working) | Status | One-line story |
|---|------|-----------------|--------|----------------|
| 13.1 | `performance-model.md` | The Performance Model | **new** | FLOPs, bytes, roofline, three regimes; how to time GPUs without lying to yourself; the method |
| 13.2 | `hardware.md` | Hardware | rewrite in place | Where bytes live and why compute outruns bandwidth; 2026 numbers; our own box as worked example |
| 13.3 | `compilation.md` | Compute Graphs and Compilation | **new** (replaces `hybridize.md`) | `torch.compile` vs `jax.jit`: capture, fusion, graph breaks, overhead; see what the compiler did |
| 13.4 | `memory-precision.md` | Memory and Precision | **new** | The memory anatomy of a training step; bf16; activation checkpointing; gradient accumulation |
| 13.5 | `multiple-gpus.md` | Multi-GPU from First Principles | rewrite in place | From-scratch data parallelism; derive ring allreduce; measure the communication bill honestly |
| 13.6 | `multi-gpu-practice.md` | Multi-GPU in Practice | **new** (replaces `multiple-gpus-concise.md`) | DDP + the FSDP idea vs JAX's declarative sharding — "annotate the layout, the compiler writes the collectives" |
| 13.7 | `fast-transformer.md` | Case Study: Making a Transformer Fast | **new** | The whole ladder applied to ch. 11's GPT, one measured waterfall |
| — | `index.md` | — | rewrite | The method essay; scope fences; Resources |
| — | `async-computation.md`, `auto-parallelism.md`, `parameterserver.md` | — | **deleted** | Content absorbed into 13.1/13.3/13.5/13.6 (see §4) |

Framework coverage: **PyTorch + JAX** throughout (Advanced-part policy); the
tf/mxnet tabs and their committed outputs are removed, with two library-
contract relocations to keep untouched chapters building (§6.2).

Everything below is evidence and specification for this table.

---

## 2. Ground truth the plan is built on

### 2a. Inbound obligations (all grep-verified)

~20 `:numref:` references from 18 files target `chap_performance`; three more
target specific subsections. Each promise, and where the plan discharges it:

| Promise (source) | Discharged by |
|---|---|
| "Full treatment of asynchrony, streams, and multi-device parallelism" (`builders-guide/gpus-devices-memory.md`, 4×, once per tab) | 13.1 (async dispatch, timing, barriers; streams at concept level) + 13.6 (multi-device) |
| "How to get performance out of the operations you already have"; kernel *authoring* explicitly out of scope for the book (`builders-guide/custom-layers.md`) | chapter-wide; the no-kernel-authoring fence restated in `index.md`; Triton/Pallas at pointer altitude in 13.3 |
| Sharded/lazy checkpoint loading "when models get that big" (`builders-guide/saving-loading.md`, 4×) | 13.4 (memory anatomy: why models stop fitting) + 13.6 (FSDP idea) + explicit hand-off to `sec_training_systems` |
| Gradient accumulation "we return to in `chap_performance`" (`optimization/minibatch-sgd.md`) | 13.4 (a dedicated subsection — this promise is currently satisfied *nowhere* in the chapter) |
| Data parallelism as the machinery that turns big batches into wall-clock (`optimization/batch-size.md`) | 13.5 |
| ZeRO-style sharding + comm/compute overlap (`optimization/adamw.md`, `practice.md`) | 13.6 (the allreduce = reduce-scatter + all-gather identity → FSDP; DDP overlap) |
| "Triton kernels, memory hierarchies, and serving systems that make them fast belong to ch. 13" (`recurrent-modern/index.md`, `hybrids.md`) | memory hierarchy → 13.2; kernels/fusion → 13.3; **serving systems → partially; see Open Question 2** |
| "The systems-level story belongs to ch. 13" re speculative decoding/serving (`transformers/kv-cache.md`) | 13.2's prefill-vs-decode roofline reading covers the economics; serving *engines* flagged (Open Question 2) |
| "Data pipelines, custom kernels, and the parallelism of ch. 13" (`transformers/gpt.md`); "the systems work of ch. 13" (`scaling-laws.md`) | 13.5/13.6/13.7 |
| `tf.function` graph compilation forward-ref (`preliminaries/ndarray.md` → `sec_hybridize`) | 13.3 (label renamed; one-line inbound fix, §6.3) |
| `split_batch` prose ref (`nlp-applications/natural-language-inference-attention.md` → `sec_multi_gpu`) | label kept on 13.5; `#@save` kept |
| "Recall the introduction to multi-GPU training" (`computer-vision/image-augmentation.md` → `sec_multi_gpu_concise`) | label kept on 13.6 |

### 2b. What the book already teaches — the non-duplication contract

The rebuilt chapter builds *on* these, cites them, and must not re-teach them:

- **FlashAttention / IO-aware fusion** — fully taught with a measured >10×
  benchmark in ch. 10.5 (`attention-at-scale.md`). 13.3 teaches fusion as a
  *general* compiler mechanism and cites 10.5 as the handwritten pinnacle.
- **Parallel scans, chunked recurrences, Mamba's recompute-in-backward kernel
  argument** — ch. 12. 13.4's checkpointing section backward-cites Mamba's
  "hardware-aware" paragraph as the trade the reader already met.
- **`cost_analysis()` FLOP accounting** — ch. 11.7 scaling-laws already uses
  `jax.jit(...).lower(...).compile().cost_analysis()`; 13.1 reuses the idiom.
- **Shape stability for JIT** — taught as data-loading discipline in ch. 2/3
  (`drop_remainder=train`); 13.3's recompilation section closes that loop.
- **`d2l.Timer`** (ch. 9), **`try_gpu`/`try_all_gpus`** (ch. 5) — inherited,
  not redefined. `d2l.Benchmark` (currently defined here, zero external
  consumers) is upgraded and moves to 13.1 (§6.2).
- **ch. 29 `sec_training_systems`** — owns the production-library map (DDP,
  FSDP2, DeepSpeed, torchrun, what-to-use-at-which-scale, checkpointing long
  runs), display-only. Division of labor, stated in both chapters: **ch. 13
  earns the concepts at notebook scale; ch. 29 names the products at
  datacenter scale.** 13.6 closes with the pointer. `sec_hardware_buyers`
  (ch. 29) keeps the buyer's-guide altitude; 13.2 keeps the architectural one.
- **BN-fold-into-conv** — `convolutional-modern/` territory; not reclaimed.

### 2c. What the build box can actually demonstrate [all measured]

The API audit ran live experiments on the actual 4×RTX 4090 build machine.
These facts shape the design; the plan treats them as teaching assets:

1. **No P2P, no NVLink, on any GPU pair** (`nvidia-smi topo -p2p` reports CNS
   everywhere; `torch.cuda.can_device_access_peer` = False; NVIDIA disables
   P2P on GeForce by policy). Every inter-GPU byte stages through host memory.
2. **NCCL allreduce bus bandwidth ≈ 2.2 GB/s, flat from 2→4 GPUs** (256 MB
   fp32 payload). JAX's XLA collectives on identical hardware: 4.5 GB/s at 2
   GPUs, 8.6 at 4. Both are two-plus orders of magnitude below an NVLink
   domain (~1.8 TB/s per B200). Any "more GPUs = proportionally faster" demo
   is dead on arrival; the *accounting* of when data parallelism pays is
   therefore the lesson, and our topology makes it vivid rather than
   embarrassing. 13.5/13.6/13.7 are designed around this.
3. **`mp.spawn` fails outright under nbconvert** (cell-defined workers are
   not picklable from ipykernel's `__main__`). `start_processes(...,
   start_method='fork')` works *iff the parent kernel never touches CUDA
   first*. Untested but likely-clean third path: launching `torchrun` from a
   notebook cell as a subprocess (§4.5, Open Question 3).
4. **JAX is fully notebook-friendly**: one process sees all 4 GPUs;
   `Mesh`/`NamedSharding`/`jax.shard_map` all verified in-process.
5. **Robust, quotable single-GPU effects**: `torch.compile` 1.32× on a small
   conv net (first call ~4 s); bf16 autocast **1.93× vs a fair TF32
   baseline**; TF32 is *off* by default at 2.11 (`'highest'`) and Inductor
   itself warns to enable it — the "set `torch.set_float32_matmul_precision
   ('high')` before benchmarking anything" lesson is mandatory hygiene.
6. `jax` AOT `cost_analysis()['flops']` returns exact analytic FLOPs without
   execution — the elegant basis for the roofline sweep.

### 2d. Ordering confirmed

Position 13 (after 9 Optimization, 10 Attention, 11 Transformers, 12 SSMs)
works and is better than the chapter has ever had it: 13.7 can profile and
accelerate a real `d2l.GPT` (ch. 11, `num_hiddens=256`, 6 blocks, char-level
*Time Machine*), 13.3 can cite ch. 10.5's fused-attention benchmark, and the
optimization chapter's forward promises all point the right way. Nothing
after ch. 13 (RL, GANs, diffusion) depends on it. No ordering change
proposed. The preface's stale roadmap sentence is out of scope (its rewrite
is separately deferred, per CLAUDE.md).

---

## 3. What is preserved (the salvage list)

1. **The from-scratch data-parallel implementation** (`multiple-gpus.md`):
   manual parameter broadcast, hand-rolled allreduce, per-device
   forward/backward, then "NCCL does this properly." The single best teaching
   device in the chapter; 13.5 keeps it as its spine, PT + JAX.
2. **The ring-allreduce derivation** (`parameterserver.md`): reduce-scatter +
   all-gather, per-link traffic `2(k−1)/k · N` independent of ring size —
   timeless, and the identity that later turns DDP into FSDP/ZeRO. Absorbed
   into 13.5.
3. **The eager-vs-compiled framing and the `Benchmark` device**
   (`hybridize.md`) — re-grounded in `torch.compile` + `jax.jit`.
4. **The async barrier/blocker taxonomy** (`async-computation.md`): `.item()`,
   `.numpy()`, `print(tensor)`, `.nonzero()` force syncs; don't read the loss
   every step. Recentered on CUDA streams / `block_until_ready` in 13.1.
5. **hardware.md's pedagogical skeleton**: latency-numbers framing, burst vs
   random access, cache hierarchy, the training-vs-inference distinction, and
   most of its 13 principle-teaching exercises. Every number and product name
   is replaced (§13.2).
6. The load-bearing labels: `chap_performance`, `sec_hardware`,
   `sec_multi_gpu`, `sec_multi_gpu_concise` (and `sec_hybridize`'s single
   inbound, handled by rename+fix).
7. The library contracts `d2l.split_batch` and `d2l.resnet18` (§6.2).

---

## 4. The structural calls

**4.1 Measurement before silicon (13.1 before 13.2).** The curriculum survey
found every strong course opens with first-principles FLOP/byte accounting,
none with an API. We go one step further: 13.1 *measures* the phenomena on
our own box (the matmul efficiency ramp, the async-timing trap) and states
the roofline with the GPU's two numbers taken as given; 13.2 then explains
where those two numbers come from and why their ratio keeps worsening. Effect:
hardware is motivated by data the reader has already produced, and the
chapter's method (measure first) is enacted from page one.

**4.2 Hardware stays in the chapter (recommendation).** `sec_hardware` has
zero inbound references, so relocation to the Tools appendix is *possible* —
but the roofline needs the silicon story adjacent, Alex's own instinct ("we
need a solid understanding anyway to support the rest of the book") argues
for keeping it, and every surveyed course teaches hardware inside the
performance sequence, not as an appendix. Ch. 29's `sec_hardware_buyers`
already covers the what-should-I-buy altitude, so there is no duplication:
13.2 is *why the machine behaves this way*, 29 is *what to purchase*.
Recommended: stay, rebuilt on the MLSS deck's June-2026-verified numbers.
(Flagged as Open Question 1 per Alex's explicit ask.)

**4.3 `parameterserver.md` is cut.** The ring-allreduce derivation (its one
piece of gold) moves into 13.5; the push/pull KV-store abstraction survives
as one historical paragraph there (parameter servers today live on mainly in
recommender-system embedding tables — a pointer to ch. 22's territory and to
`sec_training_systems`). Zero inbound references; the modern distributed
story already lives, correctly, in ch. 29. This also discharges the "teach
today, parameter server historical at most" requirement directly.

**4.4 `async-computation.md` and `auto-parallelism.md` dissolve.** Async
dispatch is not a *topic* in 2026, it is the #1 measurement pitfall and a
scheduling fact — that is 13.1's job. Auto-parallelism's real payload
(independent ops overlap; compute overlaps communication) reappears where it
pays: kernel-level overlap in 13.3's overhead discussion, and comm/compute
overlap as DDP's headline mechanism in 13.6. The existing overlap benchmark
does not reproduce on the capture hardware (the pytorch tab already confesses
this) and is retired rather than fixed — the honest replacement is DDP's
measured overlap. `fig_frontends` (MXNet's multi-language diagram) dies.

**4.5 The PyTorch multi-process idiom: `torchrun` from a cell, fork as
fallback.** DDP/FSDP need multiple processes; the audit proved `mp.spawn`
cannot work under nbconvert and that fork-before-any-parent-CUDA does
[measured]. But the fork discipline is fragile (one stray `.to('cuda')` cell
above the fork poisons the kernel forever) and is not what anyone runs in
production. The plan's primary idiom is instead: write the training script
from a cell (`%%writefile`-style), then execute `torchrun --standalone
--nproc-per-node=k train_ddp.py` as a subprocess — fresh processes, no
fork/CUDA-ordering constraint, and *exactly* the launcher ch. 29 and the
PyTorch docs teach. This was not piloted by the audit (it tested the
`torch.multiprocessing` family only); nothing in its findings rules it out,
and it is a plain subprocess launch. **Hard pilot gate before any 13.6/13.7
prose** (§7); the verified fork harness is the specified fallback, with the
discipline documented in-notebook. Either way, the single-GPU baseline runs
through the same harness at world-size 1, so the comparison is
apples-to-apples and the notebook structure is fork-safe by construction.
(Open Question 3.)

**4.6 FSDP2: concept + code sketch, not a live demo.** FSDP's value
proposition — fit what doesn't fit — is invisible on models that occupy a few
hundred MB of a 24 GB card, and FSDP1 is deprecated at our pin. 13.6 teaches
the *identity* (allreduce = reduce-scatter + all-gather) and the ZeRO
shard-what's-redundant table with a `fully_shard`/`DeviceMesh` sketch
(`eval: false`), then hands off to `sec_training_systems`. A live demo would
be checkbox theater; the identity is the durable content.

**4.7 The case study is the differentiator (13.7).** Modern treatments end
with an integrative wall-clock exercise (CS336 A2, modded-nanoGPT); none
ship it as a runnable textbook notebook. Taking ch. 11's own GPT down a
measured waterfall — profile, classify, fix, re-measure at every rung —
is the chapter's closing argument and the book's proof that the method is
real. It also gives every earlier section a place its technique visibly pays.

**4.8 Framework policy per demo.** Both frameworks everywhere except:
(a) memory *snapshot* tooling — PyTorch's `_record_memory_history` →
memory_viz has no JAX twin; the JAX tab uses `device_memory_profile` +
compiler `memory_analysis()` and the prose names the philosophical difference
(counter/snapshot vs profiler/compiler-report); (b) CUDA-graphs/
`reduce-overhead` — PyTorch-only demo (XLA amortizes launches by
construction; the JAX tab says exactly that); (c) 13.6's process story —
necessarily asymmetric: PyTorch = multi-process launcher, JAX = single
process + sharding; the asymmetry *is* the lesson (explicit collectives vs
GSPMD). PyTorch tabs lead by book convention; the JAX declarative-sharding
contrast is 13.6's deliberate climax.

---

## 5. The new arc, section by section

Binding conventions for all sections: one imports cell per framework;
`torch.set_float32_matmul_precision('high')` in the PyTorch imports cell of
every notebook that times anything (with one sentence; taught properly in
13.4); all illustrative figures pre-generated via a new
`tools/gen_mdl_perf_figures.py` (§6.1); every timed cell uses the upgraded
sync-aware `d2l.Benchmark` (§6.2); prose quotes only re-execution-stable
magnitudes (§6.6).

### 13.1 `performance-model.md` — The Performance Model (`sec_perf_model`, NEW)

*Learning goals: FLOPs/bytes/latency vocabulary; arithmetic intensity and the
roofline; the three regimes; why naive GPU timing lies; profiler first
contact; the measure→classify→fix→re-measure method.*

1. **Open with the hook**: one matmul kernel, two sizes, ~50× difference in
   achieved TFLOP/s. Same silicon, same op. The section exists to explain
   this plot.
2. **`## Counting: FLOPs, Bytes, and Arithmetic Intensity`** — intensity =
   FLOPs/bytes moved; the matmul algebra (for `X[B,D]·W[D,F]`, intensity ≈ B
   when B ≪ D,F); the roofline: performance = min(peak FLOP/s, intensity ×
   bandwidth); ridge point = peak/bandwidth, computed for the reader's own
   card from `torch.cuda.get_device_properties`/spec numbers (RTX 4090:
   ~165 TF bf16 / ~1.0 TB/s ⇒ ridge ≈ 165 FLOP/byte). Figure:
   `fig_roofline` (schematic, house style).
3. **`## Measuring Without Lying`** — async dispatch (Python enqueues, the
   GPU runs behind); the wrong-vs-right timing demo (naive timer reports
   ~nothing; add `torch.cuda.synchronize()` / `.block_until_ready()` and get
   the real number) [audit demo 1, verified]; CUDA events as the precision
   option; the barrier taxonomy salvaged from `async-computation.md`
   (`.item()`, `.cpu()`, `print`, `.nonzero()`), with the "don't read the
   loss every step" rule and a measured cost. Figure: `fig_async_timeline`.
   The upgraded `#@save d2l.Benchmark` lands here.
4. **`## The Sweep`** — matmul TFLOP/s vs size (256→8192, bf16): PyTorch
   computes FLOPs analytically (2n³); JAX gets them from
   `compiled.cost_analysis()['flops']` [verified exact] — the ramp
   (bandwidth-bound) bending into the roof (compute-bound), plotted against
   the roofline. This is the audit's cleanest demo (single GPU, reproducible
   within a few percent).
5. **`## Three Regimes`** — Horace He's taxonomy as the diagnostic
   vocabulary: compute-bound (live with it or change format — 13.2/13.4),
   bandwidth-bound (fuse — 13.3), overhead-bound (capture/replay — 13.3).
   A bandwidth-regime demonstration: an elementwise chain whose runtime
   stays flat as compute-per-byte grows, until it doesn't. The fix is
   *deliberately deferred* to 13.3 — the section only diagnoses. Figure:
   `fig_regimes`.
6. **`## The Profiler`** — first contact: `torch.profiler` /
   `jax.profiler.trace` on a small training step; the CPU-time vs
   device-time table as the regime detector (CPU ≫ CUDA time = overhead-
   bound); Chrome/Perfetto trace export named, not belabored. Close with the
   method box: **measure → classify → fix → re-measure** — "the rest of this
   chapter is this loop, applied."

Demos: 4 (timing trap; sweep; elementwise chain; profiler table) — all
single-GPU, all verified-class. Salvages: async-computation's taxonomy +
mental model; hybridize's Benchmark. Runtime: ~3 min, 1 GPU. Slides: the
50× hook; roofline + ridge; the timing trap; three regimes; the method loop.
Exercises: compute your card's ridge point from specs and compare to the
sweep's knee; find the batch size where `B×4096·4096×4096` crosses regimes
and check against intensity ≈ B; measure the cost of `print(loss)` per step;
profile a dataloader-starved loop and classify it.

### 13.2 `hardware.md` — Hardware (`sec_hardware`, REWRITE)

*Learning goals: explain 13.1's two numbers from silicon; the memory
hierarchy with 2026 magnitudes; tensor cores and the format ladder;
interconnects, including our own box; energy; the three growth rates.*

All numbers from the MLSS deck's June-2026-verified set (its `verification/`
pass checked 87 hardware claims; the flagged errors are already fixed
upstream). Every figure redrawn as a house-style matplotlib generator — the
deck's charts are inline TikZ (data, not images), so this is "port the
numbers, redraw," and **none** of the old chapter's NVIDIA marketing
PNG/JPGs (turing.png, tensorcore.jpg, …) survive.

1. **`## Where Bytes Live`** — the hierarchy: registers/SRAM (~100 MB total
   on-die, ~TB/s-to-tens-of-TB/s) → HBM/GDDR (tens–hundreds of GB, ~1–8
   TB/s) → host DRAM (hundreds of GB, ~0.5 TB/s) → NVMe (TBs, ~14 GB/s) →
   network. The 2026 bandwidth ladder and latency ladder figures
   (`fig_bandwidth_ladder`, `fig_latency_ladder` — replacing both stale
   latency tables); "every chip boundary costs an order of magnitude."
   Burst-vs-random access and the cache lesson from the old section survive
   compressed; false sharing demotes to an exercise.
2. **`## Why Compute Outruns Bandwidth`** — the shoreline argument (compute ∝
   die area, I/O ∝ perimeter; doubling the side buys 4× compute, 2× I/O);
   HBM-on-interposer as the countermeasure; the three growth rates (compute
   ~4×/generation, bandwidth ~2×, capacity ~1.7×) — the reason the ridge
   point keeps climbing and this chapter stays relevant. Figure:
   `fig_shoreline`.
3. **`## The GPU`** — SMs and warps at concept altitude (enough to read a
   profiler, not a CUDA course); tensor cores as matmul-shaped silicon; the
   format ladder fp32 → tf32 → bf16/fp16 → fp8 → fp4 with to-scale bit-layout
   figure (`fig_float_formats`) and the two-word rule (same 8-bit exponent =
   same range; every halving wins twice — 2× FLOP/s *and* half the bytes);
   the spec table (H100 / B200 / RTX 4090 / RTX 5090) as the one
   numbers-dense table, dense-not-sparse convention footnoted. One compact
   cross-vendor row-set (AMD MI355X, TPU v7, Trainium2) to make the
   same-physics point; the edge-device menu and market-pricing notes from
   the deck are **cut** (fastest-decaying content).
4. **`## The CPU's Role`** — reframed from the old microarchitecture tour to
   what matters for DL: the orchestrator that launches kernels (5–15 µs
   each), feeds data (pinned memory, `non_blocking=True`), and runs the
   input pipeline. One measured cell: H2D bandwidth, pageable vs pinned.
5. **`## Interconnects`** — PCIe vs NVLink generations; then **our box as
   the worked example**: 4×4090, no P2P (GeForce segmentation), every
   GPU-to-GPU byte staged through host memory, measured ~2 GB/s allreduce —
   against a GB200 NVL72's 1.8 TB/s per GPU. "Why datacenter fabrics exist,"
   taught from the machine the book is built on. Figure: `fig_pcie_topology`
   (our topology, PHB/NODE, host-staged path highlighted).
6. **`## Energy`** — the pJ ladder (INT8 add 0.03 → fp32 mul 3.7 → DRAM read
   ~2000): *a DRAM access costs ~500 multiplies; arithmetic is free, moving
   operands is the budget*. Figure: `fig_energy_ladder`.
7. **`## Reading the Roofline: Two Workloads`** — the prefill-vs-decode
   worked example (prefill intensity ~ context length → compute-bound;
   batch-1 decode intensity ~1, bytes = whole model per token →
   bandwidth-bound; tokens/s ≈ bandwidth / bytes-per-token), tied to ch. 11's
   KV-cache accounting by backward reference. This is where the kv-cache
   chapter's "systems story" promise gets its substance (Open Question 2
   handles the serving-engine remainder). Ends with the actionable
   rules-of-thumb table (adapted from the deck's "useful numbers," training-
   slanted: bytes/param under Adam ≈ 16–20 with mixed precision; kernel
   launch 5–15 µs; PCIe vs HBM vs NVLink magnitudes).

Demos (light, ~4 short cells): device query; measured HBM bandwidth on a
large elementwise op (lands within tens of percent of the 1.0 TB/s spec —
stable); pinned-vs-pageable H2D; all else prose + figures. Salvages: the old
section's framing, structure, and most exercises (renumbered; principle-
teaching ones kept, Turing/Optane-specific ones replaced). Runtime: ~2 min,
1 GPU. Slides: the two ladders; shoreline; format ladder; our box vs an
NVL72; the three growth rates. Exercises: decode vs prefill intensity for
ch. 11's GPT; energy budget of one epoch from the pJ ladder; measure the
cache cliff (working-set sweep); why HBM sits on the interposer; the false-
sharing experiment (from the old section, now an exercise).

### 13.3 `compilation.md` — Compute Graphs and Compilation (`sec_compilation`, NEW; replaces `hybridize.md`)

*Learning goals: eager vs captured execution; how each framework captures a
graph; what the compiler does with it (fusion); graph breaks vs
recompilation; the overhead regime and CUDA graphs; introspecting compiled
programs.*

1. **`## The Graph Was Always There`** — autograd already builds a graph
   (backward ref to ch. 2/5); eager execution walks it one kernel at a time:
   every op = a launch (5–15 µs, 13.2) + an HBM round trip (13.1's bandwidth
   regime). Capturing the graph lets a compiler see across ops. One
   paragraph of history at footnote altitude: the symbolic-vs-imperative
   framework wars ended in "eager by default + a tracing compiler"; the old
   hybridize framing is retired, not re-taught. Figure: `fig_compute_graph`
   (regenerated, house style).
2. **`## Capture: Two Philosophies`** — `torch.compile`: Dynamo bytecode
   capture, guards, **graph breaks** (fall back to Python, keep going) —
   demonstrated with `torch._dynamo.explain` on a data-dependent `if`;
   `jax.jit`: functional tracing to a **jaxpr** (shown via `jax.make_jaxpr`;
   the vanishing-`print` demo — "the trace captures tensor ops only; purity
   matters"), lowered to XLA-HLO — **no graph breaks, but recompilation on
   new shapes/dtypes** (the JAX footgun; `static_argnums` and `lax.cond` as
   the escape hatches; backward ref to ch. 3's `drop_remainder` discipline,
   promise redeemed). The contrast table (bytecode-with-fallback vs
   pure-trace; breaks vs retraces). Figure: `fig_compile_pipelines` (Dynamo →
   AOTAutograd → Inductor/Triton ∥ jaxpr → HLO → XLA).
3. **`## What the Compiler Does: Fusion`** — pick up 13.1's deferred
   bandwidth-regime demo and *cure it*: the unfused elementwise chain vs the
   same chain under `torch.compile`/`jax.jit` — fewer HBM round trips,
   roughly 2× [stable effect class, verified]. State the general rule
   (fusion trades kernel launches and memory traffic for nothing); cite
   ch. 10.5's FlashAttention as the handwritten, IO-aware pinnacle of the
   same idea — **not re-taught**. One paragraph: when the compiler can't
   fuse, people write kernels — Triton (PyTorch/Inductor's backend) and
   Pallas (JAX; GPU via Triton, TPU via Mosaic) at pointer altitude,
   honoring `custom-layers.md`'s "we do not teach kernel authoring" fence.
4. **`## Whole-Step Compilation, Measured`** — compile the full training
   step (fwd+bwd+optimizer) of a mid-sized model in both frameworks; show
   the first-call compile cost (seconds) explicitly, then steady state
   [1.32× measured on a small conv net; a meatier model is chosen at pilot
   time to make the ratio comfortably quotable]. JAX side: same experiment,
   plus `.lower().compile()` AOT staging and `memory_analysis()` — "see what
   the compiler decided" (the introspection theme JAX does uniquely well).
5. **`## The Overhead Regime: Capture and Replay`** — many-small-ops model
   (deep stack of thin layers): CPU dispatch can't keep the GPU fed; the
   profiler shows gaps; `torch.compile(mode="reduce-overhead")` (CUDA
   graphs: record once, replay with one launch) closes them. PyTorch-only
   demo; the JAX tab explains why XLA doesn't need a separate mechanism.
   Static-shape requirement stated.
6. **`## When Compilation Hurts`** — compile latency vs amortization; graph
   breaks in hot loops; shape churn; the honest checklist of when *not* to
   bother. One paragraph on portability/serialization (torch.export,
   StableHLO) as the modern successor to the old serialization deep-dive —
   prose only, no demo.

Demos: 5. Salvages: hybridize's concept arc + benchmark shape;
auto-parallelism's "independent ops schedule together" one-liner lands in
the overhead discussion. Runtime: ~5 min (dominated by compile times), 1
GPU. Slides: the graph was always there; two capture philosophies; fusion
before/after; reduce-overhead; when not to compile. Exercises: introduce a
graph break, find it with `explain`, fix it; force a jit retrace via shapes,
fix with padding or `static_argnums`; sweep model depth at fixed width and
find where `reduce-overhead` wins; compile 13.1's sweep and explain where
compiled and eager coincide (large matmuls were already one kernel).

### 13.4 `memory-precision.md` — Memory and Precision (`sec_memory_precision`, NEW)

*Learning goals: where a training step's memory goes; how to measure it;
mixed precision done right; activation checkpointing / remat; gradient
accumulation. The "it doesn't fit" rung of the ladder.*

1. **`## The Memory Anatomy of a Training Step`** — params + grads + optimizer
   states + activations, with the bytes/param table (fp32 SGD: 12; fp32 Adam:
   16; mixed-precision Adam with fp32 master copies: ~18–20 — consistent with
   ch. 9's AdamW discussion, backward ref) and activations' batch×seq×width×
   depth scaling. Figure: `fig_memory_anatomy` (stacked composition over the
   step timeline: activations grow through forward, shrink through backward).
   This section is where `saving-loading.md`'s "when models get that big"
   promise begins to be paid.
2. **`## Measuring Memory`** — PyTorch: `max_memory_allocated`,
   `_record_memory_history` → snapshot → memory_viz (the blessed-if-
   underscored path, per the audit); JAX: `device_memory_profile` +
   `compiled.memory_analysis()` — and the taught contrast: PyTorch counts
   allocations at runtime, XLA *plans* memory at compile time. Verify the
   anatomy table's prediction against the measured peak on a GPT stack.
3. **`## Mixed Precision`** — the tensor-core requirement (13.2); the
   TF32 lesson done properly: one *unfair* fp32-`'highest'` baseline shown
   once, then the fair TF32 baseline for the real comparison [audit-verified
   framing]; bf16 autocast **without GradScaler** as the modern default
   [1.93× measured, quotable as "about 2×"], fp16+GradScaler as the
   pre-Ampere legacy in one paragraph; JAX: precision is explicit — dtypes
   threaded through the model, `jax.default_matmul_precision`, `jmp` named;
   the implicit-vs-explicit contrast stated. Both time *and* memory reported
   (the AMP-recipe experiment shape, adopted chapter-wide).
4. **`## Activation Checkpointing`** — the trade: recompute the forward
   inside backward instead of storing; PyTorch `torch.utils.checkpoint`
   (`use_reentrant=False`, mandatory kwarg since 2.9 — kill-list item); JAX
   `jax.checkpoint`/`remat` **with policies** (`checkpoint_dots`,
   `print_saved_residuals` — you *watch* what is kept; the cleaner mental
   model, said so); measured on a stacked-block model: large activation-
   memory reduction for roughly a third more step time [robust effect class;
   quote qualitatively]. Backward ref: ch. 12's Mamba recompute-in-kernel
   paragraph — same trade, one level down.
5. **`## Gradient Accumulation`** — the minibatch-sgd promise, finally
   discharged: k micro-batches, one step; `B_global = B_device · k` (· ranks,
   forward pointer to 13.5); parity check demo (accumulated vs full-batch
   losses match at matched global batch); when it's the right tool (memory-
   bound, not speed-bound — it *costs* time).
6. **`## The Ladder So Far`** — recap box: doesn't fit → bf16 → checkpoint →
   accumulate → still too slow → more devices (next section).

Demos: 5, all single-GPU, single-process. Runtime: ~5 min, 1 GPU. Salvage:
none from the old chapter (this file is the biggest pure gap the old chapter
had). Slides: the anatomy bar; snapshot screenshot-style figure; fair-vs-
unfair baselines; checkpoint trade; the ladder. Exercises: budget the
largest GPT (d, L) trainable in 24 GB under {fp32 Adam, bf16, bf16+ckpt};
derive per-token activation bytes for ch. 11's `TransformerBlock` and check
against the profiler; implement every-√n-blocks checkpointing and measure;
show fp16-without-scaler diverging on a deep net (then fix with GradScaler).

### 13.5 `multiple-gpus.md` — Multi-GPU from First Principles (`sec_multi_gpu`, REWRITE)

*Learning goals: data parallelism built by hand; collectives as the primitive;
ring allreduce derived; the communication cost model; when data parallelism
pays. The crown jewel, kept and sharpened.*

1. **`## Three Ways to Split`** — data / layer (pipeline) / tensor
   partitioning, one figure (`fig_splitting`, regenerated), with the honest
   scope sentence: this chapter builds data parallelism; the other two only
   pay at scales this part defers to the Language Models chapters. The
   historical AlexNet-across-two-GPUs beat in one sentence (+ citation, no
   figure). The "GPU memory is a solved problem" claim from the old file is
   deleted with prejudice — 13.4 just taught the opposite.
2. **`## Data Parallelism by Hand`** — the preserved from-scratch
   implementation, PT tab: `get_params`/broadcast, `#@save split_batch`,
   hand-rolled star allreduce (gather on GPU 0, broadcast back), per-device
   forward/backward, manual SGD; LeNet on Fashion-MNIST, 2 GPUs. **The
   measured result is a slowdown** (~30% in the committed store) — kept
   deliberately, now with a sharper diagnosis via 13.1's method: profile it,
   see the allreduce dominate, *measure* the hand-allreduce's effective
   bandwidth (~2 GB/s on our host-staged topology [measured]) and compare to
   the model's compute per step. The failure is the syllabus.
3. **`## Doing Better: Ring Allreduce`** — salvaged derivation from
   `parameterserver.md`, now in its right home: star moves (k−1)·N bytes
   through one hub; reduce-scatter + all-gather around a ring moves
   `2(k−1)/k · N` per link — bandwidth-optimal, independent of k. Figure:
   `fig_ring_allreduce` (step-by-step, house style). Stated forward hook:
   *this identity — allreduce = reduce-scatter + all-gather — is also the
   seed of FSDP/ZeRO (13.6).* NCCL implements ring/tree and picks per
   message size; on our box the transport (host-staged copies), not the
   algorithm, is the ceiling — theory-vs-practice paragraph.
4. **`## The JAX Tab, Explicit on Purpose`** — the same from-scratch
   semantics via `jax.shard_map` + `lax.psum` (top-level API at our pin;
   `pmap` retired per the kill list): the collective is *visible in the
   code*, matching the PT tab's hand-rolled loop; measured psum bandwidth
   [4.5–8.6 GB/s measured]. This sets up 13.6's reveal (jit can write the
   collective for you).
5. **`## The Accounting`** — the cost model: `t_step(k) ≈ t_compute(B/k) +
   t_comm(2(k−1)/k · P·bytes / BW)`; plug in measured numbers; predict when
   a second GPU pays (bigger model, bigger per-device batch, faster link);
   the scaling-efficiency definition used in 13.6/13.7. One paragraph of
   history closes the section: parameter servers (push/pull, sharded
   aggregation — the authors' own lineage, cited) organized this
   communication for the multi-machine asynchronous era; synchronous
   collectives won for dense training; the PS pattern survives in recsys
   embedding systems (ch. 22) and the production landscape lives in
   `sec_training_systems`.

Demos: from-scratch training (2 GPUs), hand-allreduce bandwidth measurement,
shard_map psum, the accounting cell (analytic + measured inputs). Salvages:
the entire from-scratch spine; parameterserver's derivation + history;
`fig_splitting`/`fig_data_parallel` regenerated. `#@save`: `split_batch`
(PT/JAX). Runtime: ~8 min, **2 GPUs** (scheduler-marked). Slides: three
splits; the hand-rolled loop; the measured slowdown + diagnosis; ring
derivation; the accounting. Exercises: extend the hand-rolled version to
k=4 and measure; implement ring allreduce with `.to()` copies and test
whether it beats star on a host-staged box (it barely can — explain);
compute the ring's traffic for ResNet-18's 11M params at our measured
bandwidth and predict 13.6's result; gradient quantization thought
experiment (fp32→bf16 gradients halve `t_comm` — what breaks?).

### 13.6 `multi-gpu-practice.md` — Multi-GPU in Practice (`sec_multi_gpu_concise` kept, NEW FILE; replaces `multiple-gpus-concise.md`)

*Learning goals: DDP mechanics (processes, buckets, overlap); the FSDP/ZeRO
idea; JAX's declarative sharding as the headline contrast; measured
end-to-end scaling on 2–4 GPUs; the one-paragraph multi-node bridge.*

1. **`## What Our Hand-Rolled Loop Lacked`** — three deficits of 13.5's
   implementation: no overlap (allreduce waits for the whole backward), one
   Python process (GIL, one dispatcher for k GPUs), star topology. DDP fixes
   all three: one process per GPU, gradient bucketing, **allreduce
   overlapped with the backward pass** — the compute/communication overlap
   idea (formerly auto-parallelism.md's) finally shown where it earns money.
   Figure: `fig_ddp_overlap` (timeline: backward and bucket-allreduces
   interleaved).
2. **`## DDP, Really Run`** — the launcher reality stated plainly: DDP means
   multiple processes; in a notebook we write the training script from a
   cell and launch `torchrun --standalone --nproc-per-node=k` (primary
   idiom; §4.5, pilot-gated; fallback: the verified fork harness with the
   no-CUDA-in-parent rule stated in prose *and* enforced by structure — all
   GPU work including the k=1 baseline goes through the harness).
   `init_process_group("nccl")` → `DDP(model)` → the loop is *unchanged from
   single-GPU* — that's the selling point. Model: `#@save d2l.resnet18` on
   Fashion-MNIST-64 (compute-dense enough for the box; contract preserved
   for ch. 19). **Measured**: throughput at k ∈ {1, 2, 4}; efficiency
   confronted with 13.5's accounting (predicted vs measured — on our
   P2P-less box expect real-but-modest 2-GPU gains and visibly sublinear 4-GPU
   scaling; the agreement between prediction and measurement is the result).
   One line on what an NVLink box changes (t_comm shrinks ~100×; the same
   accounting, different constant). `nn.DataParallel` gets exactly one
   contrast sentence (single-process, GIL-bound, legacy) — kill-list item.
3. **`## Sharding the Redundant: the FSDP Idea`** — every rank holds
   identical params, grads, optimizer states — k−1 copies of everything are
   redundant. ZeRO's ladder (shard states → grads → params; memory ∕ k);
   the 13.5 identity cashed in: replace allreduce with reduce-scatter (each
   rank keeps its shard's grads) + all-gather (params re-assembled
   just-in-time per layer, freed after). `fully_shard`/`DeviceMesh` sketch,
   `eval: false`; FSDP1 named as deprecated (kill list). When you need it
   (models past ~a few B params; not our 11M-param demo — say so). Figure:
   `fig_fsdp_lifecycle`. Hand-off: `sec_training_systems` for the
   production map. This pays adamw/practice's ZeRO promises.
4. **`## JAX: Annotate the Layout, the Compiler Writes the Collectives`** —
   the headline contrast, built to land as a reveal: define a `Mesh`, give
   the batch a `NamedSharding(P('data'))`, `jax.device_put` the arrays, jit
   the *unchanged* training step — GSPMD partitions the computation and
   inserts the psum 13.5 wrote by hand; `jax.debug.visualize_array_sharding`
   shows the layout before and after (the audit verified the whole path
   in-process). Same ResNet, same k-sweep, measured. Then the punchline
   paragraph: changing the `PartitionSpec` — not the code — moves between
   data-, tensor-, and FSDP-style sharding; one mechanism spans what PyTorch
   exposes as three APIs. `shard_map` (13.5) is the manual end of the same
   spectrum. The contrast table (explicit/imperative vs
   declarative/compiler-driven).
5. **`## When One Node Is Not Enough`** — the bridge, one paragraph, promise-
   precise: tensor/pipeline/expert parallelism extend "shard something" to
   models and fleets where a single machine loses; the Language Models part
   takes it from here with data big enough to warrant it; the library map is
   `sec_training_systems`.

Demos: torchrun DDP sweep (pilot-gated); JAX sharded sweep;
`visualize_array_sharding`; optional profiler trace showing overlap
(pilot-gated, drop without regret if fragile). `#@save`: `resnet18`
(PT/JAX). Runtime: ~10 min, **4 GPUs** (scheduler-marked; JAX variant needs
whole-box reservation — one process claims all four devices). Slides: what
the hand-rolled loop lacked; DDP overlap timeline; the identity → FSDP; the
sharding-grid reveal; predicted-vs-measured scaling. Exercises: vary DDP
`bucket_cap_mb` and measure; run the fallback fork harness and deliberately
break the no-parent-CUDA rule to see the failure mode; write the
`PartitionSpec` for a tensor-parallel matmul and `visualize_array_sharding`
it; size ZeRO-3 for a 7B-param model on 8×80 GB (arithmetic).

### 13.7 `fast-transformer.md` — Case Study: Making a Transformer Fast (`sec_fast_transformer`, NEW)

*Learning goals: none new — that is the point. Execute the whole method on a
real model and attribute every win to a section.*

1. **`## The Subject`** — ch. 11's `d2l.GPT` (char-level *Time Machine*,
   ctx 128), possibly widened (d = 256 → 512 at pilot time so each rung's
   effect clears the noise floor); the metric: tokens/second, measured with
   13.1's discipline; a short real training run at the end to show the
   accelerated configuration still learns.
2. **`## Rung 0: Baseline, Profiled`** — eager, TF32-fair; profile; classify
   (at this scale, substantially overhead/bandwidth-bound — small kernels,
   thin matmuls).
3. **`## Rungs, Each One Measured`** — the waterfall, one subsection per
   rung, each ending with re-profile + re-classify:
   compile (13.3) → bf16 (13.4) → *spend the saved memory on batch size*
   (13.4's anatomy: bigger per-device batch climbs the roofline — the rung
   that ties precision to the roofline, and typically the biggest single
   win on thin models) → activation checkpointing discussed and **measured
   as not helping here** (memory isn't the binding constraint at this scale
   — a deliberate negative rung; knowing when a technique doesn't apply is
   the method too) → data parallel across 2–4 GPUs via 13.6's machinery,
   with the efficiency *predicted first* from 13.5's accounting (transformer
   params ∝ compute, so DP on a ~2 GB/s fabric is communication-hungry;
   the prediction-then-confirmation is the demonstration).
4. **`## The Waterfall`** — one data plot (computed, not pre-generated):
   tokens/s per rung, annotated with the regime each rung attacked. Prose
   quotes ratios rounded to halves ("compile bought roughly a third; bf16
   about 2×; four GPUs about …× at …% efficiency — close to what §13.5's
   model predicts for this fabric").
5. **`## The Lore, and the Ladder Beyond`** — modded-nanoGPT's speedrun as
   the cultural closer: the record run stacks compiled block-sparse
   attention, a better optimizer, fp8 — each mappable to a section of this
   chapter (or to ch. 9); fp8 tensor cores exist on our Ada GPUs but the
   FP8-training recipe is deferred (Language Models part). Close the chapter
   by restating the method — measure, classify, fix, re-measure — now
   demonstrated seven sections deep.

Demos: 5–6 timed configurations + profiles; PyTorch DP rung via the 13.6
harness (the notebook is structured so all rungs run through it, or the DP
rung runs via torchrun — pilot decides). Runtime: ~15 min, **4 GPUs**
(scheduler-marked). Slides: the subject; rung-by-rung waterfall build-up
(one reveal per rung); the negative rung; the speedrun lore. Exercises: add
a rung (`channels_last`? pinned-memory dataloading? `sdpa_kernel` backend
pinning?) and measure; apply the ladder to ch. 12's Mamba capstone; apply it
to ch. 11's ViT; take a rung that helped here and construct a model where it
doesn't (then explain with the profiler).

### `index.md` — rewrite (`chap_performance` kept)

The method essay, modeled on the ch. 10/11 index register: (1) the third
axis — the book so far chose models and data; this chapter is about the
machine, and one method for using it well; the roofline as the chapter's
map, the regimes as its vocabulary. (2) The section tour (7 sentences).
(3) One honest paragraph about the box the book is built on — consumer
GPUs, no NVLink — and why that is a teaching instrument, not a limitation
(readers' hardware looks like ours, not like an NVL72; the accounting
transfers, the constants don't). (4) What this chapter is not: multi-node
parallelism (Language Models part), kernel authoring (fence from ch. 5.4,
pointers to Triton/Pallas/GPU-MODE), serving engines (see Open Question 2),
the production-library map and buying advice (ch. 29). (5) Resources and
Further Reading (§6.7).

---

## 6. Cross-cutting specifications

### 6.1 Figures

New generator `tools/gen_mdl_perf_figures.py` importing the shared house
style (mdl-figure skill; byte-idempotent; black axes/labels, no
text-line collisions, render-and-inspect loop mandatory). ~15 figures:

| Figure | § | Source of truth |
|---|---|---|
| `fig_roofline` (schematic, ridge annotated) | 13.1 | scaling-book/NVIDIA algebra |
| `fig_async_timeline` (dispatch queue vs GPU) | 13.1 | replaces `fig_asyncgraph`/`fig_threading` |
| `fig_regimes` (three-regime diagnostic) | 13.1 | Horace He taxonomy |
| `fig_bandwidth_ladder`, `fig_latency_ladder` (2026) | 13.2 | MLSS r2 dossier data |
| `fig_memory_hierarchy` (pyramid w/ sizes+BW) | 13.2 | MLSS §2.1 |
| `fig_shoreline` (area vs perimeter) | 13.2 | MLSS §2.2 |
| `fig_float_formats` (to-scale bit layouts) | 13.2 | MLSS §2.5 |
| `fig_energy_ladder` (pJ/op) | 13.2 | Horowitz/Dally via MLSS |
| `fig_pcie_topology` (our box, host-staged path) | 13.2 | `nvidia-smi topo` [measured] |
| `fig_compute_graph` (regenerated) | 13.3 | replaces stale SVG |
| `fig_compile_pipelines` (Dynamo∥XLA) | 13.3 | torch.compiler docs / JAX docs |
| `fig_memory_anatomy` (step-timeline composition) | 13.4 | Ultra-Scale §2 shape |
| `fig_splitting`, `fig_data_parallel` (regenerated) | 13.5 | old chapter, house style |
| `fig_ring_allreduce` (RS+AG steps) | 13.5 | parameterserver.md derivation |
| `fig_ddp_overlap` (bucketed backward timeline) | 13.6 | DDP paper/tutorial |
| `fig_fsdp_lifecycle` (shard→gather→compute→free→RS) | 13.6 | FSDP2 tutorial |

Deleted: all 25 existing figure/table labels, including every NVIDIA
marketing raster (`turing.png`, `tensorcore.jpg`, `turing-processing-
block.png`, `frontends.png`, `latencynumbers.png`) and the 2.8 MB
`twogpu.svg`. Data plots (13.1's sweep, 13.7's waterfall, scaling curves)
are computed in-notebook via `d2l.plot` — allowed, they teach measured
results.

### 6.2 `#@save` contracts and library migration

| Symbol | Old home | New home | Notes |
|---|---|---|---|
| `Benchmark` | hybridize.md | 13.1 | upgraded: warmup + device-sync (torch `synchronize` / jax `block_until_ready`) built in; zero external consumers, safe to change signature |
| `split_batch` | multiple-gpus.md | 13.5 (kept) | PT/JAX tabs; label `sec_multi_gpu` kept so docstrings/refs stay true |
| `resnet18` | multiple-gpus-concise.md | 13.6 (kept) | PT/JAX; consumed by ch. 19 (`image-augmentation.md`, `kaggle-cifar10.md`) |
| `split_batch`, `resnet18` (mxnet, tf copies) | this chapter | **relocate to `chapter_computer-vision/image-augmentation.md`** | ch. 13 drops mxnet/tf tabs, but `make lib` builds `d2l/mxnet.py`/`tensorflow.py` from source `#@save` blocks — deleting the tabs here would break the still-4-framework ch. 18/19 notebooks (their mxnet tabs call `d2l.split_batch`; tf/mxnet call `d2l.resnet18`). Mechanical ride-along edit to the consuming chapter (Open Question 4); alternative home: ch. 29 `utils.md` |

No other new library surface: 13.7 reuses `d2l.GPT`/`d2l.train_lm` (ch. 11),
`d2l.Timer` (ch. 9), `try_all_gpus` (ch. 5). Default to fewer saves.

### 6.3 Label and reference migration (complete)

| Label | Inbound | Fate |
|---|---|---|
| `chap_performance` | ~20 refs, 18 files | **keep** on index.md — untouched |
| `sec_hybridize` | 1 (`preliminaries/ndarray.md`, TF tab) | **rename → `sec_compilation`** on 13.3; fix the one inbound line (the old name is dead MXNet vocabulary; one-line cost) |
| `sec_multi_gpu` | 1 (`nli-attention.md`) | **keep** on 13.5 |
| `sec_multi_gpu_concise` | 1 (`image-augmentation.md`) | **keep** on 13.6 (label survives the file rename; accurate enough, zero-risk) |
| `sec_hardware` | 0 | keep on 13.2 (free either way) |
| `sec_async`, `sec_auto_para`, `sec_parameterserver` | 0 each | **die** with their files |
| all 25 `fig_*`/`table_*` | 0 external | die; replaced per §6.1 |
| new: `sec_perf_model`, `sec_memory_precision`, `sec_fast_transformer` | — | created |

Files: `hybridize.md` → `compilation.md`, `multiple-gpus-concise.md` →
`multi-gpu-practice.md` (outputs-tree + slide-stamp moves in §7);
`async-computation.md`, `auto-parallelism.md`, `parameterserver.md` deleted
with their outputs (all four framework trees).

### 6.4 Deprecated-API kill list (never appears as taught material)

`torch.jit.script`/`trace` (deprecation-warns at our pin) · `allow_tf32`
bool flags (→ `set_float32_matmul_precision`) · FSDP1 wrapper (→
`fully_shard` sketch) · `nn.DataParallel` (one contrast sentence only) ·
`checkpoint(...)` without `use_reentrant=` (hard error since 2.9) ·
`jax.pmap`/`nnx.pmap` (compat shim since 0.8; both current files use it) ·
`jax.experimental.shard_map` import path (→ top-level `jax.shard_map`).

### 6.5 Runtime budget and scheduler marks

| Notebook | GPU-time (est., per fw) | GPUs | Scheduler note |
|---|---|---|---|
| 13.1 | ~3 min | 1 | — |
| 13.2 | ~2 min | 1 | — |
| 13.3 | ~5 min | 1 | compile latency dominates |
| 13.4 | ~5 min | 1 | — |
| 13.5 | ~8 min | 2 | 2-GPU mark; CPU-box gate defers |
| 13.6 | ~10 min | 4 | 4-GPU/whole-box; JAX single process claims all devices — exclusive reservation (cf. the JAX memory-fraction scheduler note) |
| 13.7 | ~15 min | 4 | whole-box |

Total ≈ 45–50 min/framework ≈ 1.5–1.7 GPU-hours per full 2-framework
recapture — heavier than the old chapter (which trained almost nothing) but
modest against the vision chapters, and only 13.5–13.7 gate on multi-GPU
hardware. On single-GPU or CPU hosts the capability-aware freshness gate
defers them by design.

### 6.6 Results-precision policy (what prose may quote, per demo)

| Demo | Quotable | Never quote |
|---|---|---|
| matmul sweep | the shape (ramp→roof); "from a few percent of peak to near the spec number" | exact TFLOP/s decimals |
| timing trap | "the naive timer reports close to nothing" | µs figures |
| compile | "tens of percent to about 2×, model-dependent"; first call "seconds" | 1.32× as a stable constant |
| fusion chain | "about 2×" | — |
| bf16 | "about 2×" (1.93 measured, robust) | third digits |
| allreduce BW | "a couple of GB/s on our box; orders of magnitude below an NVLink domain" | flat-vs-scaling cross-framework rankings (config-sensitive; one careful sentence max on XLA's better use of the same links) |
| checkpointing | "a large fraction of activation memory for roughly a third more step time" | exact % pairs |
| memory anatomy | ratios exact-by-construction (Adam states = 2× params); measured GB rounded to tenths | allocator-noise digits |
| DP scaling | efficiency to the nearest ~5–10%; "prediction and measurement agree to within tens of percent" | per-run seconds |
| waterfall | per-rung ratios rounded to halves | cumulative precise multipliers |

House rules apply throughout: no per-seed decimals, no conclusions on
differences inside run-to-run noise, single seeded runs.

### 6.7 Citations (add to `d2l.bib`; verify IDs at implementation) and Resources

New bib entries: Williams/Waterman/Patterson 2009 (roofline, CACM) ·
Ansel et al. 2024 (PyTorch 2 / TorchDynamo+Inductor, ASPLOS) · Li et al.
2020 (PyTorch DDP, VLDB; 2006.15704) · Xu et al. 2021 (GSPMD; 2105.04663) ·
Zhao et al. 2023 (PyTorch FSDP; 2304.11277) · Micikevicius et al. 2018
(mixed precision; 1710.03740) · Micikevicius et al. 2022 (FP8 formats;
2209.05433) · Chen et al. 2016 (gradient checkpointing; 1604.06174) ·
Patarasuk & Yuan 2009 (bandwidth-optimal allreduce, JPDC) · Sergeev &
Del Balso 2018 (Horovod; 1802.05799) · Krizhevsky 2014 (one weird trick;
1404.5997) · Frostig et al. 2018 (JAX tracing, MLSys) · Tillet et al. 2019
(Triton, MAPL) · Horowitz 2014 (energy, ISSCC). Already present (verify,
don't duplicate): Dao 2022 (FlashAttention), Rajbhandari 2020 (ZeRO),
Li et al. 2014 (parameter server), Goyal 2017 (via ch. 9).

Resources and Further Reading (index.md, ch. 9 format — link, access note,
tie-back to a `:numref:`; all URLs verified live by the survey):
**Books/long-form**: How to Scale Your Model (DeepMind; the roofline→
collectives→sharding companion to 13.1/13.5/13.6) · Ultra-Scale Playbook
(HF; the memory-anatomy and parallelism ladder behind 13.4/13.6) ·
Making Deep Learning Go Brrrr (Horace He; 13.1's three regimes in original
form). **Courses**: CS336 (Stanford) · 15-442 ML Systems (CMU) · GPU-MODE
lecture series (the kernel-authoring path this book fences off).
**Docs/tutorials**: NVIDIA DL-performance/GEMM guide (13.1's two-GEMM
regime flip) · PyTorch performance tuning guide, AMP recipe (13.4's
experiment template), profiler recipe, `torch.compile` tutorial +
architecture docs, FSDP2 tutorial, CUDA-graphs blog · JAX: jit-compilation,
sharded-computation, profiling pages (13.3/13.6's JAX spine) ·
`smolix/mlss-efficiency` (the June-2026 hardware numbers behind 13.2, incl.
the inference-side chapters this book defers). **Lore**: modded-nanoGPT
speedrun log (13.7's closing frame) · Lilian Weng, "How to Train Really
Large Models on Many GPUs?" (the ZeRO taxonomy at blog altitude).

### 6.8 Scope fences (what ch. 13 deliberately does NOT cover, and who owns it)

Multi-node training — tensor/pipeline/context/expert parallelism, network
fabrics (→ Language Models part; bridge paragraph in 13.6) · kernel
authoring in CUDA/Triton/Pallas (→ fenced off book-wide per ch. 5.4;
pointers only) · serving engines — continuous batching, paged KV,
speculative decoding (→ Open Question 2; the *economics* appear in 13.2's
roofline reading) · quantization for inference — GPTQ/AWQ/formats-as-
compression (→ future efficiency treatment; 13.2 teaches formats as
*training* precisions only) · FlashAttention internals (→ ch. 10.5) ·
scan/chunked-recurrence algorithms (→ ch. 12) · production library map,
torchrun ops, cluster checkpointing, GPU buying advice (→ ch. 29) ·
data-pipeline engineering beyond one checklist mention (→ tuning-guide
pointer).

---

## 7. Build/ops plan (implementation phase, for the record)

**Pilot gates, in order, before any prose:** (P1) `torchrun` launched from
an nbconvert-executed cell — the §4.5 primary idiom; if it fails, fall back
to the verified fork harness and restructure 13.6/13.7 accordingly (all
GPU cells through the harness). (P2) DDP ResNet-18 throughput at k ∈
{1,2,4} — the honest scaling curve 13.6's prose is written around. (P3) The
13.7 waterfall end-to-end at two candidate model widths; pick the width
whose rungs clear noise. (P4) Memory-snapshot rendering path
(`_dump_snapshot` → viewable artifact) inside the build. (P5) JAX 4-GPU
notebooks under the scheduler (whole-box reservation; memory-fraction note
from the ops memory applies). Single-GPU demos (13.1–13.4) are
verified-class per the audit and need no pre-gates beyond normal capture.

Then per-file: figures first (`gen_mdl_perf_figures.py`, render-and-inspect
loop), section prose written around measured outputs, recapture via the
scheduler (capture refuses manual-nbconvert provenance; check "0 failed"
before capturing), slides re-authored per section (`make -B slides-<fw>
SLIDES_FILTER=...`). File moves (`hybridize.md`→`compilation.md`,
`multiple-gpus-concise.md`→`multi-gpu-practice.md`, three deletions, three
creations) ride through `_quarto.yml` + the outputs tree in one commit;
delete the chapter's tf/mxnet output trees; relocate the two mxnet/tf
`#@save` blocks to `image-augmentation.md` (Open Question 4) **before**
deleting them here, and run `make lib` + execute the affected ch. 18/19
mxnet/tf notebooks to prove the contract held. Fix the one `sec_hybridize`
inbound in `preliminaries/ndarray.md`. Full render + PDF check at the end
(the `$`-digit and smallmatrix tripwires; the chapter is table-heavy).
Ride-along doc fix: CLAUDE.md's "Current status" still lists the pre-SSM
chapter order (comp-perf as ch. 11) — update when this lands. Sequencing
mirrors the ch. 12 build: max two author agents, pilot-first, prose after
measurement.

**Size estimate:** 7 notebooks + index at ch. 10–12 density ≈ 7,500–8,500
lines, ~85–95 slide divs — comparable to the approved SSM rebuild. Trim
lever if scope must shrink: fold 13.2's cross-vendor material to a
paragraph and compress 13.3's serialization/portability and overhead
subsections; 13.7 is the last thing to cut (it is the chapter's argument).

---

## 8. Open questions for Alex

1. **Hardware placement.** Keep the rebuilt hardware section as 13.2
   (recommended — the roofline needs the silicon adjacent, and ch. 29
   already owns the buyer's-guide altitude), or move it to the Tools
   appendix? Sub-decision if it stays: the compact cross-vendor table is
   in, the edge-device/price menu is cut (fastest-decaying data) — OK?
2. **The serving promise.** Ch. 12 (`index.md`, `hybrids.md`) and ch. 11
   (`kv-cache.md`) currently name "serving systems" as ch. 13 territory.
   This plan covers the inference *economics* (13.2's prefill-vs-decode
   roofline) but not serving engines (batching, paged KV, speculative
   decoding — a chapter's worth, and your MLSS deck is its obvious source).
   Approve two one-line softenings upstream (pointing serving to the
   Language Models part / a future efficiency chapter), or should ch. 13
   grow a serving section now? (Recommended: soften; keep ch. 13 training-
   focused per the mandate.)
3. **The PyTorch multi-process idiom.** Primary plan: write the DDP script
   from a cell and launch `torchrun` as a subprocess (production-idiomatic,
   no fork/CUDA ordering hazard, but *not yet piloted* under nbconvert).
   Fallback: the empirically verified fork harness with the documented
   no-CUDA-in-parent discipline. Approve pilot-first-then-decide, with
   fork as the specified fallback?
4. **Ride-along edit to ch. 19.** Relocating the mxnet/tf copies of
   `split_batch`/`resnet18` into `computer-vision/image-augmentation.md`
   keeps `make lib` emitting them for the still-4-framework ch. 18/19
   notebooks after this chapter drops its tf/mxnet tabs. It is a mechanical
   move into a chapter not otherwise touched — OK? (Alternative: park them
   in ch. 29's `utils.md`.)
5. **GPU budget for the case study.** 13.7 as specified costs ~15 min of
   4-GPU time per framework per recapture (chapter total ~45–50 min/fw).
   Acceptable, or should the waterfall be trimmed (2 GPUs, fewer rungs)?
