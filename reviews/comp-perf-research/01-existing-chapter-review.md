# Existing "Computational Performance" chapter — critical review

*Read-only review of `chapter_computational-performance/` (currently ch. 13) for a
later modernization pass. Source-of-truth `.md` files only. 2026-07-20.*

## Orientation

The chapter is 8 files: `index.md` + 7 sections. Two are text-only conceptual
(hardware, parameterserver), one is a text-only front matter (index), and five
carry code notebooks. The chapter is ~5 years old (d2l-en heritage) with a
recent JAX-port + prose-polish pass layered on top, but the *pedagogical spine
is still MXNet's* (imperative/symbolic/hybrid engine, async backend, KVStore
push/pull). It predates `torch.compile`, `DistributedDataParallel`/FSDP as the
default, modern JAX (`jit`/`pmap`/`shard_map`), and every current accelerator.

**Framework tabs:** every code notebook ships all four (mxnet, pytorch,
tensorflow, jax). Per the Advanced-part policy (ch. 9–16 = **PyTorch + JAX
only**), the tf/mxnet tabs and their committed outputs are slated for removal.

**Code-cell census** (```` ```{.python .input} ```` fences; per-fw `#@tab`):

| Section | fences | mxnet | pytorch | tf | jax | shape |
|---|---|---|---|---|---|---|
| index | 0 | – | – | – | – | prose |
| hybridize | 24 | **9** | 4 | 4 | 4 | MXNet-heavy (serialization deep-dive is mxnet-only) |
| async-computation | 21 | **7** | 4 | 6 | 4 | MXNet-heavy |
| auto-parallelism | 24 | 6 | 6 | 6 | 6 | even |
| hardware | 0 | – | – | – | – | prose |
| multiple-gpus | 47 | 12 | 12 | 12 | 11 | even, biggest notebook |
| multiple-gpus-concise | 27 | 9 | 6 | 6 | 6 | MXNet-heavy |
| parameterserver | 0 | – | – | – | – | prose |

**Outputs store:** all four framework trees have captured outputs for this
chapter (`outputs/{pytorch,jax,tensorflow,mxnet}/chapter_computational-performance/`).
Notebooks are *light*: the four small JSON manifests are 1.5–5 KB each (toy
matmul/benchmark timings, no training). The only heavy artifacts are the
`multiple-gpus/` and `multiple-gpus-concise/` subdirs (~34–40 KB each — a
handful of test-accuracy SVGs from 10-epoch Fashion-MNIST runs). TF has a
`multiple-gpus-concise.json` but no captured SVG subdir. Nothing in this
chapter is GPU-expensive to recapture except the two multi-GPU training plots
(and those *need ≥2 GPUs*).

---

## Codex feedback triage — comp-perf items

**Finding: there are none.** `reviews/ch10-11-codex-feedback-triage-2026-07-19.md`
is entirely about **ch. 10 Attention** and **ch. 11 Transformers** (current
numbering) — masked-softmax, KV-cache, GQA/MLA, MoE accounting, scaling laws,
ViT patch-embedding, tokenization. It contains **zero** items touching
hybridize / async / auto-parallelism / hardware / multi-GPU / parameter-server.
The task's premise ("under the OLD numbering ch11 was this chapter") does not
hold for this file — the triage's ch10/11 are attention/transformers, not
comp-perf. No prior external feedback on this chapter exists in that document.
(The one transferable meta-lesson: the triage's house-policy reminders —
CPU-measurement notebooks need no CPU fallback, purge "honest"/absolutes, quote
only precision that survives re-execution — apply to any rewrite here.)

---

## Per-section review

### `index.md` — chapter front matter
**Teaches:** 12-line intro + `toc`. Frames the chapter as "imperative
programming, symbolic programming, asynchronous computing, automatic
parallelism, and multi-GPU computation."
**Staleness:** the framing sentence omits *hardware* and *parameter servers*
entirely (two of the seven sections), so the index already under-describes its
own contents. "reducing training time without affecting accuracy" is a 2019
framing; the modern motivation is scaling to models that *don't fit at all* on
one device. No labels referenced inbound except `chap_performance` (heavily —
see below).
**Salvageable:** the `:label:`chap_performance`` anchor (20+ inbound refs — a
hard constraint). Little else; the intro needs a full rewrite around the modern
performance story (compilation, memory, parallelism ladder).

### `hybridize.md` — "Compilers and Interpreters" (`sec_hybridize`)
**Teaches:** imperative vs symbolic programming → "hybrid" programming → hybridize
a `Sequential` MLP → benchmark eager-vs-compiled → serialize the graph. Figure
`fig_compute_graph` (computegraph.svg). 6 slides.
**Framework skew:** the *most MXNet-centric* code section. 9 mxnet cells vs 4
each. The entire **Serialization** deep-dive (§ `HybridNet`, the `F`
symbol-vs-ndarray argument, the "print statements vanish after hybridize" demo,
`export` → `my_mlp-symbol.json`) is **mxnet-only** and teaches an
MXNet-internals mental model that no longer generalizes. The core narrative
("historically frameworks chose imperative *or* symbolic; hybrid combines them")
is MXNet's Gluon origin story.
**Staleness:**
- PyTorch path uses `torch.jit.script` / `torch.jit` — **superseded by
  `torch.compile`** (TorchDynamo/Inductor, the default since PT 2.0). Scripting
  is legacy; the chapter never mentions `torch.compile`.
- The pytorch tab already hedges that the CPU MLP shows "little or no
  improvement" — i.e. the headline demo doesn't actually demonstrate its
  headline on the captured hardware.
- TF/Theano/CNTK framing ("Theano, CNTK formulate models symbolically") is
  historical trivia now; CNTK/Theano are dead.
- JAX tab is the healthiest (jit is genuinely central to JAX) but is bolted on
  rather than leading.
**Salvageable:** the imperative-vs-symbolic *concept* and the eager-vs-compiled
*benchmark device* (`Benchmark` context manager, `#@save`) are keepers. The
serialization/portability idea survives but must be re-grounded in
`torch.compile` + `torch.export` / StableHLO, not TorchScript + MXNet symbol
JSON. `fig_compute_graph` is a generic dataflow diagram — regenerate in house
style. Exercise "can you improve a previous model's perf by reimplementing it"
is fine.

### `async-computation.md` — "Asynchronous Computation" (`sec_async`)
**Teaches:** framework frontend/backend split; async dispatch; the dependency
graph; barriers/blockers (`.numpy()`, `.item()`, print); why mid-loop syncs
stall the pipeline. Figures `fig_frontends` (frontends.png), `fig_asyncgraph`,
`fig_threading`. 7 slides.
**Framework skew:** written around **MXNet's async engine** (`npx.waitall`,
`z.wait_to_read`). The whole "Asynchrony via Backend" reveal ("MXNet is orders
of magnitude faster than numpy — something else must be going on") is the MXNet
demo. PyTorch and JAX are *async-by-default*, so for them the "surprise" is
muted and the section reads as retrofitted. `fig_frontends` literally shows
"Python, R, Scala, C++" MXNet frontends.
**Staleness:**
- `frontends.png` is an MXNet-multilanguage diagram — off-message for a
  PyTorch/JAX book.
- The $t_1 + 10000\,t_2 + t_3$ analysis (frontend/backend queue model) is a nice
  mental model but framed in MXNet's threading terms.
- TF tab carries `tf.experimental.async_scope()` and heavy eager-vs-`@tf.function`
  scaffolding that will be deleted with the TF tab.
- Barrier list is still correct and useful (`.item()`/`.numpy()`/print force
  sync — timeless).
**Salvageable:** the async mental model, the barrier/blocker taxonomy, and the
"don't `print(loss)` every step" guidance are genuinely worth keeping — but
should be recentered on the CUDA-stream model (PyTorch `torch.cuda.synchronize`,
JAX `block_until_ready`) rather than the MXNet frontend/backend thread story.
`fig_asyncgraph`/`fig_threading` are generic and regenerable. This material may
merge into a single "how async + compilation actually run" section.

### `auto-parallelism.md` — "Automatic Parallelism" (`sec_auto_para`)
**Teaches:** dependency-graph-driven parallelism; run two independent matmuls on
2 GPUs and show combined < sum; overlap compute with device→host copy. Figure
`fig_twogpu` (twogpu.svg, **2.8 MB** — bloated). 4 slides.
**Framework skew:** now fairly even (6/6/6/6). Concept (backend schedules
independent ops in parallel) is universal.
**Staleness:**
- **Requires ≥2 GPUs** to execute — expensive/gated to recapture.
- The pytorch tab's honest confession is the standout: on the captured single
  RTX 4090 the compute+copy overlap benchmark is *slower* than the sum of parts
  (~4.7 s vs 0.1 + 3.2), with a long paragraph of caveats. Pedagogically candid,
  but it means the section's headline result (overlap saves time) **does not
  reproduce** on the capture hardware — a rewrite must either fix the demo or
  reframe it.
- Nsight link is `nsight-compute-2019_5` (dated).
- Cites `Hadjis.Zhang.Mitliagkas.ea.2016` (CPU+GPU training) — dated but fine as
  history.
**Salvageable:** the "independent ops run in parallel for free" demo and the
compute/communication-overlap idea are worth keeping (overlap is exactly what
DDP/FSDP exploit). `fig_twogpu` concept is good but the 2.8 MB SVG must be
regenerated lean and in house style. Could merge with async-computation into one
"async + auto-parallel scheduling" unit.

### `hardware.md` — "Hardware" (`sec_hardware`)
**Teaches:** latency numbers every programmer should know; computer anatomy;
memory (DRAM burst reads); storage (HDD/SSD/NVMe/cloud); CPU microarch,
vectorization, cache, false sharing; GPUs/TPUs, tensor cores; networks/buses;
two latency tables. Nine figures. **No code, no exercises tied to code.**
**Framework skew:** none (hardware-agnostic prose).
**Staleness — this is the worst-dated section; every dated claim:**
- **CPUs:** "AMD Threadripper 3 has 64 PCIe 4.0 lanes"; "Zen 3 Threadripper …
  8 slots"; "AMD Epyc 3 … 256 MB cache"; "AMD's EPYC 3 has 128 lanes, Intel's
  Xeon up to 48 lanes"; "Ryzen 9 (20 lanes), Core i9 (16 lanes)"; "consumer-grade
  Intel CPUs have 24 lanes"; Intel **Skylake** quad-core as the exemplar CPU
  (`fig_skylake`, 2015); **ARM Cortex A77** microarch (`fig_cortexa77`, 2019).
  All ~2019–2020 parts; current is Zen 5 / Arrow Lake / Grace, PCIe 5.0.
- **Memory:** "CPU RAM is typically DDR4, 20–25 GB/s per module" — now **DDR5**;
  HBM referenced as "HBM (…) limited to high-end server chips such as the NVIDIA
  Volta V100" — now HBM3e on H100/H200/B200/MI300.
- **GPU (all Turing/Volta era):** "NVIDIA's RTX 2080 Ti has a 352-bit bus …
  4,352 CUDA cores … one NVLink at reduced 100 Gbit/s"; "Titan series … GDDR6
  over 500 GB/s"; "V100 preferable for training, Turing T4 for inference";
  the entire **Turing** deep-dive (`fig_turing_processing_block`, `fig_turing`,
  `fig_tensorcore`, TU102, tensor cores "4×4 to 16×16"). Current is
  Ada/Hopper/Blackwell; tensor cores now do FP8/FP4, NVLink is 4th/5th gen
  (~900 GB/s), the "T4 for inference / V100 for training" split is obsolete.
- **Storage:** HDD "7,200 RPM, 16 TB on 9 platters, ~100 IOPs"; SSD "100k–500k
  IOPs, 1–3 GB/s"; "NVMe up to 4 lanes, 8 GB/s on PCIe 4.0" — now PCIe 5.0 NVMe
  ~14 GB/s; **Intel Optane** cited twice in the latency table (product line
  **discontinued** 2022).
- **Buses:** "PCIe 4.0 up to 32 GB/s 16-lane" (now PCIe 5.0/6.0); "NVLink up to
  300 Gbit/s per link, V100 six links, RTX 2080 Ti one link".
- **Latency tables** (`table_latency_numbers`, `table_latency_numbers_tesla`):
  Broadwell E5-2690v4, DC P3608 NVMe, DC S3510 SATA SSD, "40GB NVLink ~33 GB/s",
  "PCIe 3.0 x16 ~12 GB/s" — all ~2016–2018 measurements. Sourced from Jeff
  Dean's **2010** Stanford talk + Colin Scott's interactive page.
- **Dead/stale links:** Intel OpenVino `01.org` (defunct domain); cs152 `sp19`;
  Hennessy & Patterson **2011** edition (now 6th ed. 2019).
**Salvageable:** the *pedagogical structure* is excellent and largely timeless —
the "latency numbers every programmer should know" framing, burst-read vs
random-access, the cache hierarchy + false-sharing lesson, vectorization/SIMD,
roofline-style "match algorithm to bandwidth" reasoning, the training-vs-inference
accelerator distinction, and most of the 13 exercises (they teach principles,
not numbers). The *numbers and product names* must all be refreshed (DDR5, PCIe
5, HBM3e, Hopper/Blackwell, FP8/FP4 tensor cores, current NVLink). The
photographic PNG/JPG figures (`turing.png` 804 KB, `tensorcore.jpg` 464 KB,
`turing-processing-block.png`, `frontends.png`, `latencynumbers.png`) are
NVIDIA-marketing / third-party screenshots — likely copyright-encumbered and
Turing-era; replace with house-style diagrams or drop. **No inbound refs to
`sec_hardware` from the rest of the book** — it is a free-standing island, so it
can be freely renamed, cut, or relocated (e.g. into the Tools appendix).

### `multiple-gpus.md` — "Training on Multiple GPUs" (`sec_multi_gpu`)
**Teaches:** the three ways to split work (network / layerwise / **data**
parallelism); then a **from-scratch data-parallel implementation** — manual
`get_params` broadcast, hand-rolled `allreduce` (star gather on GPU 0 +
broadcast), `split_batch`, `train_batch` (per-GPU forward/backward → allreduce →
per-GPU SGD), on a from-scratch LeNet + Fashion-MNIST. Figures
`fig_alexnet_original`, `fig_splitting`, `fig_data_parallel`. 11 slides.
**Framework skew:** even (12 each). JAX tab uses `pmap`/`lax.pmean` (idiomatic);
PT/TF/MXNet do the manual star-allreduce.
**Staleness:**
- **Requires ≥2 GPUs.** Captured 2-GPU run is a **~30% regression** vs 1 GPU
  (LeNet too small to amortize sync) — the file is admirably honest about this
  ("*the worst case* for data-parallel training"), but it means the flagship
  from-scratch demo produces a *negative* result on the capture box.
- AWS instance zoo (g4dn.12xlarge, p3.16xlarge, **p2.16xlarge/16 GPUs**) is
  dated; p2 is retired.
- "GPU memory used to be a problem … by now this issue has been resolved for all
  but the most unusual cases" — **flatly false in the LLM era**; model/tensor/
  pipeline/sharded parallelism exists *precisely* because models don't fit.
- The `nn.parallel.scatter`, `torch.no_grad` manual SGD, and hand allreduce are
  teaching scaffolding, not how anyone trains today.
**Salvageable:** **the from-scratch data-parallel implementation is the crown
jewel of the chapter and MUST NOT be lost.** Manually broadcasting parameters
and hand-writing allreduce (then revealing NCCL ring-allreduce does this
optimally) is a classic, irreplaceable teaching device — it demystifies
DDP/FSDP. Keep it (PyTorch + JAX), but pick a model big enough that 2 GPUs
actually help (the file itself points at "ResNet on ImageNet"), and frame it as
the on-ramp to modern DDP. `fig_splitting`/`fig_data_parallel` are good
conceptual diagrams — regenerate in house style; `fig_alexnet_original`
(historical model-parallel AlexNet) is a nice history beat. Inbound ref from
`natural-language-processing-applications/natural-language-inference-attention.md`
to `sec_multi_gpu` (the `split_batch` mention) — a rename constraint.

### `multiple-gpus-concise.md` — "Concise Implementation …" (`sec_multi_gpu_concise`)
**Teaches:** the same data-parallel training via high-level APIs on a modified
ResNet-18 + Fashion-MNIST resized to 64×64; PyTorch `nn.DataParallel`, TF
`MirroredStrategy`, JAX `nnx.pmap`/`vmap`, MXNet `gluon.Trainer`. 8 slides.
**Framework skew:** MXNet-heavy prose (9 cells; init-on-device deep-dive is
mxnet-only) but the pytorch/jax paths are clean.
**Staleness:**
- **PyTorch uses `nn.DataParallel`** — the deprecated single-process pattern.
  The file *already* hedges ("PyTorch recommends `DistributedDataParallel`") but
  still *ships* `DataParallel` as the worked example. A modernization must lead
  with **DDP** (and mention FSDP), not `DataParallel`.
- Requires ≥2 GPUs.
- p2.16xlarge/16-GPU exercise again.
- JAX tab uses `nnx.pmap`/`nnx.vmap` + `nnx.Optimizer` — reasonably current, but
  `pmap` itself is now the older JAX multi-device path (`shard_map` + `jit` with
  `Mesh`/`NamedSharding` is the modern idiom; the from-scratch section already
  uses `Mesh`/`NamedSharding` in one cell).
**Salvageable:** the "framework wraps it in one line; same numerical recipe" pay
off is worth keeping — re-cast as DDP (PyTorch) + `jit`/`shard_map` or `pmap`
(JAX). The modified-ResNet-18 (`#@save resnet18`) is reused and small. Inbound
ref from `computer-vision/image-augmentation.md` to `sec_multi_gpu_concise` — a
rename constraint. This section is a strong **merge** candidate with
`multiple-gpus.md` (from-scratch → concise in one section).

### `parameterserver.md` — "Parameter Servers" (`sec_parameterserver`)
**Teaches:** distributed training across machines; bandwidth hierarchy math
(aggregate-on-GPU-0 vs CPU vs sharded); ring synchronization / ring-allreduce
derivation; multi-machine parameter-server architecture; key–value store
push/pull abstraction. 8 figures. **No code, no slides.**
**Framework skew:** framework-agnostic prose, but the KVStore push/pull framing
is MXNet's `ps-lite`/KVStore abstraction, and the citations are the authors' own
parameter-server lineage (Smola 2010, Ahmed 2012, Li 2014).
**Staleness — framing is a generation out of date:**
- Presents the **parameter server as the present tense of distributed training**.
  In 2026 the dominant paradigm is **synchronous ring/tree all-reduce (NCCL) via
  DDP**, and **sharded data parallelism (ZeRO/FSDP)** + tensor/pipeline
  parallelism for large models. The PS is now mostly historical / async-RL /
  recsys-embedding territory.
- Hardware numbers: "NVLink up to 100 GB/s across 6 links", "PCIe 4.0 32 GB/s",
  "100 GbE = 10 GB/s", **8×V100 / p3.16xlarge / DGX-2**, "consumer Intel CPUs 24
  lanes", "160 MB gradients" — all Volta-era.
- The ring-sync derivation itself (bandwidth-optimal, cost independent of ring
  size, reduce-scatter + all-gather) is **timeless and excellent** — it is
  exactly ring-allreduce, just introduced as a PS optimization.
- **Critically overlaps with an already-modern section:** `chapter_appendix-
  tools-for-deep-learning/training-systems.md` ("Model Training",
  `sec_training_systems`, 247 lines) already teaches the *modern* version — the
  scaling ladder (1 GPU → DDP → FSDP/ZeRO → tensor/pipeline parallel), where
  memory goes, the library landscape (NCCL, DDP, FSDP, DeepSpeed/ZeRO), "what to
  use at which scale", and checkpointing long runs. `adamw.md`, `practice.md`,
  `batch-size.md` etc. cite `chap_performance` *and* `sec_training_systems`
  together. Any rewrite MUST reconcile these two so the modern distributed story
  lives in one place.
**Salvageable:** the ring-allreduce derivation and the bandwidth-hierarchy
reasoning are the keepers; the push/pull KV-store abstraction is worth a
paragraph as history. **No inbound refs to `sec_parameterserver`** — free to
cut/merge/relocate. Strong **cut-or-merge-into-training-systems** candidate.

---

## Inbound reference inventory (rename/deletion constraints)

Grepped all `chapter_*/*.md` (excluding this chapter) for the 36 labels defined
here. **Only four labels are referenced from outside:**

| Label | # inbound | Referencing files |
|---|---|---|
| `chap_performance` | ~20 | preface/index; builders-guide/{saving-loading, custom-layers, gpus-devices-memory}; optimization/{adamw, practice, batch-size, minibatch-sgd}; transformers/{kv-cache, gpt, scaling-laws}; recurrent-modern/{index, hybrids} |
| `sec_hybridize` | 1 | preliminaries/ndarray.md |
| `sec_multi_gpu` | 1 | natural-language-processing-applications/natural-language-inference-attention.md |
| `sec_multi_gpu_concise` | 1 | computer-vision/image-augmentation.md |

**Everything else is free.** `sec_hardware`, `sec_async`, `sec_auto_para`,
`sec_parameterserver`, and **all 25 figure/table labels** have **zero** inbound
references — they can be renamed, merged, or cut without breaking the rest of the
book. `chap_performance` is the load-bearing anchor: whatever the chapter becomes,
keep that label (or redirect the ~20 citations).

Outbound (this chapter depends on): `sec_use_gpu`, `sec_lenet`, `sec_resnet`,
`sec_minibatch_sgd`, `sec_batch_norm`, `sec_numerical_stability` — all still
exist; keep them wired.

---

## What a modernization MUST NOT lose

1. **The from-scratch data-parallel implementation** (`multiple-gpus.md`): manual
   parameter broadcast + hand-rolled allreduce, then "NCCL does this optimally."
   The single best teaching device in the chapter.
2. **The ring-allreduce derivation** (`parameterserver.md`): cost independent of
   ring size, reduce-scatter + all-gather — timeless, and the intellectual core
   of DDP.
3. **The imperative-vs-symbolic / eager-vs-compiled framing** (`hybridize.md`) —
   re-grounded in `torch.compile` + JAX `jit`, not TorchScript/MXNet symbol.
4. **The async barrier/blocker taxonomy** (`async-computation.md`): what forces a
   sync and why mid-loop reads stall the pipeline.
5. **hardware.md's pedagogical skeleton**: latency numbers, burst vs random,
   cache hierarchy + false sharing, roofline "match algorithm to bandwidth,"
   train-vs-inference hardware — refresh the numbers, keep the lessons and
   exercises.
6. The `chap_performance` label and the four inbound-referenced section labels.

## Things a modernization should retire

- TorchScript / MXNet hybridize / `nn.DataParallel` as *the* worked examples
  (→ `torch.compile`, DDP/FSDP).
- Parameter server as the present tense of distributed training (→ NCCL
  ring-allreduce, ZeRO/FSDP; reconcile with `sec_training_systems`).
- All Turing/Volta/DDR4/Optane/PCIe-3-4 hardware numbers and the
  copyright-encumbered NVIDIA marketing PNG/JPGs.
- The AWS instance zoo (p2.16xlarge, p3.16xlarge, g4dn) and the "GPU memory is a
  solved problem" claim.
- tf/mxnet tabs and their outputs (policy).

---

## Verdict table

| Section | Verdict | One-line justification |
|---|---|---|
| `index.md` | **Rewrite** | Reframe around the modern perf story (compile → memory → parallelism ladder); keep `chap_performance` label + 20 inbound refs. |
| `hybridize.md` | **Rewrite** | Concept survives; strip MXNet serialization deep-dive, replace TorchScript with `torch.compile`; keep the eager-vs-compiled benchmark. |
| `async-computation.md` | **Rewrite / merge** | Keep async model + barrier taxonomy; drop MXNet frontend/backend framing + frontends.png; candidate to merge with auto-parallelism. |
| `auto-parallelism.md` | **Rewrite / merge** | Keep "independent ops parallelize for free" + compute/comm overlap; fix the non-reproducing overlap demo; merge with async into one scheduling section. |
| `hardware.md` | **Keep skeleton, refresh all numbers** (or **move** to Tools appendix) | Excellent timeless structure, but every product/number is 2019-era and figures are encumbered; no inbound refs, so relocatable. |
| `multiple-gpus.md` | **Keep (core) + rewrite framing** | The from-scratch data-parallel + hand allreduce is the must-keep jewel; use a model where 2 GPUs actually help; re-aim at DDP. |
| `multiple-gpus-concise.md` | **Merge into `multiple-gpus` + modernize** | Same lesson via high-level API; lead with DDP/FSDP (not `DataParallel`); fold from-scratch→concise into one section. |
| `parameterserver.md` | **Cut / merge into `sec_training_systems`** | Ring-allreduce derivation is gold, rest is superseded by the already-modern training-systems.md; no inbound refs. |
