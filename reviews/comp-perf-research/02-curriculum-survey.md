# How Deep-Learning Computational Performance Is Taught Today (2024–2026)

**A curriculum survey to inform the rebuilt d2l "Computational Performance" chapter.**

Scope of the target chapter (set by Alex Smola): **single-node performance + single-node
multi-GPU training. Multi-machine distributed is OUT** (deferred to a later Language Models
part). **Compute graphs & compilation are IN.** Modern practice, not history. It is a
**teaching** chapter (understanding + runnable notebooks), PyTorch + JAX only.

This survey covers three source classes — (A) university courses, (B) canonical
first-principles / scaling texts, (C) framework docs + practitioner "speedrun" lore — and
extracts, for each, *which* performance topics are taught, in *what order*, and by *what
pedagogy* (first-principles analysis vs. profiling/measurement vs. API walkthrough).

All URLs below were fetched/verified live during the survey (July 2026). Dates noted per source.

---

## 0. The sources at a glance (verified)

### A. University courses

| Course | Institution / instructors | Latest confirmed offering | URL |
|---|---|---|---|
| **10-414 / 10-714 Deep Learning Systems** | CMU · Kolter, T. Chen | **Fall 2025** | https://dlsyscourse.org/ |
| **15-442 / 15-642 Machine Learning Systems** ★ | CMU · T. Chen, Z. Jia | **Spring 2026** | https://mlsyscourse.org/ · https://mlsyscourse.org/schedule |
| **CS336 Language Modeling from Scratch** ★ | Stanford · P. Liang, Hashimoto | **Spring 2025** | https://cs336.stanford.edu/ · https://github.com/stanford-cs336/spring2025-lectures |
| **CS149 Parallel Computing** | Stanford · Fatahalian, Olukotun | Fall 2024 | https://cs149.stanford.edu/fall24 |
| **6.5940 TinyML & Efficient DL Computing** | MIT · Song Han | Fall 2024 (no FA25) | https://hanlab.mit.edu/courses/2024-fall-65940 |
| **CS267 Applications of Parallel Computers** | Berkeley · Buluç, Demmel | Spring 2025 | https://sites.google.com/lbl.gov/cs267-spr2025 |
| **CS294-162 ML Systems (AI-Sys)** | Berkeley · Gonzalez, Zaharia, Stoica | Fall 2024 | https://ucbsky.github.io/aisys-fa2024/ |
| **CSE 599W Systems for ML** (ancestor) | UW · T. Chen, Krishnamurthy | ~2018 (historical) | https://dlsys.cs.washington.edu/ |
| ETH "How to Write Fast Numerical Code" (roofline lecture, non-ML) | ETH · Püschel | 2024 | https://acl.inf.ethz.ch/teaching/fastcode/2024/ |

★ = best structural match for the intended chapter.

### B. First-principles / scaling texts

| Source | Author / org | Date | URL |
|---|---|---|---|
| **How to Scale Your Model** ★ (flagship) | Google DeepMind (Austin, Douglas, Frostig, …) | Feb 2025, updated (GPU ch. added) | https://jax-ml.github.io/scaling-book/ |
| **Ultra-Scale Playbook** | HuggingFace / Nanotron | Feb 2025 | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| **Making Deep Learning Go Brrrr From First Principles** ★ | Horace He (PyTorch) | ~2022 (still canonical) | https://horace.io/brrr_intro.html |
| **GPU-MODE** (formerly CUDA-MODE) lectures | GPU-MODE community (Saroufim, Köpf) | ongoing since 2024 | https://github.com/gpu-mode/lectures |
| **Roofline** (original paper) | Williams, Waterman, Patterson | CACM, Apr 2009 | https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/ |

### C. Framework docs + practitioner lore

| Source | Org / author | Date/version | URL |
|---|---|---|---|
| **NVIDIA DL Performance Guide** (GPU Perf Background, **Matrix-Multiplication Background**) ★ | NVIDIA | living | https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html |
| **PyTorch Performance Tuning Guide** | PyTorch | v2.13 | https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html |
| **torch.compile** tutorial + `torch.compiler` architecture docs | PyTorch | 2.0+ | https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html · https://docs.pytorch.org/docs/stable/torch.compiler.html |
| **CUDA Graphs** blog | PyTorch | — | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ |
| **FSDP2** tutorial | PyTorch | v2.13 | https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html |
| **AMP / mixed precision** recipe ★ | PyTorch | v2.13 | https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html |
| **PyTorch Profiler** recipe | PyTorch | v2.3+ | https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html |
| **FlashAttention** ★ (IO-awareness case study) | Dao et al. | 2022 (FA-2 2023, FA-3 2024) | https://arxiv.org/abs/2205.14135 |
| **modded-nanoGPT speedrun** ★ | Keller Jordan et al. | living log | https://github.com/KellerJordan/modded-nanoGPT |
| **How to Train Really Large Models on Many GPUs?** | Lilian Weng | 2021, upd. 2022 | https://lilianweng.github.io/posts/2021-09-25-train-large/ |

---

## 1. Topic × source matrix

Legend: ● core / dedicated treatment · ○ covered · �– light/mentioned · ✗ absent · **†** = topic is
mostly *multi-node* in that source (out of the chapter's scope even where present).

### 1a. University courses

| Topic | 10-414 | 15-442 | CS336 | CS149 | 6.5940 | CS267 | CS294 | 599W |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Roofline / arithmetic intensity | ○ | ○ | ● | ● | �– | ● | �– | �– |
| Memory hierarchy / bandwidth (HBM/SRAM) | ● | ● | ● | ● | ○ | ● | ○ | ○ |
| Kernels & fusion | ● | ● | ● | ○ | ● | ○ | ● | ● |
| Compilation / graph tracing (compile/XLA/TVM) | ○ | ● | ○ | ✗ | ○ | ✗ | ● | ● |
| Mixed precision (fp16/bf16/fp8) | �– | ○ | ● | ✗ | ● | ✗ | ○ | �– |
| Memory opt / activation-grad checkpointing | ● | ● | ● | ✗ | ● | ○ | ● | ● |
| Profiling (tools + methodology) | ○ | �– | ● | ● | ○ | ○ | ○ | �– |
| Async exec / compute-comm overlap | ○ | ○ | ● | ○ | ○ | ● | ○ | ● |
| Data parallelism (DDP) | ● | ● | ● | ○ | ● | ○ | ○ | ● |
| Collectives / allreduce (ring/NCCL) | ○ | ● | ● | ○ | ● | ● | ○ | ● |
| ZeRO / FSDP sharding | ○ | ● | ● | ✗ | ○ | ✗ | ○ | ✗ |
| Tensor parallelism † | ○ | ● | ● | ✗ | ● | ○ | ● | ○ |
| Pipeline parallelism † | ○ | ● | ● | ✗ | ● | ○ | ○ | ○ |
| Inference perf (KV cache, batching, spec-dec) | ○ | ● | ● | �– | ● | ✗ | ● | ● |
| Quantization | ✗ | ✗ | ○ | ✗ | ● | ✗ | ● | ○ |
| GPU architecture basics | ● | ● | ● | ● | ○ | ● | ○ | ● |
| GEMM / matmul performance | ● | ● | ● | ○ | ○ | ● | ○ | ● |
| Custom kernels (CUDA/Triton) | ● | ● | ● | ● | ● | ○ | ● | ○ |

### 1b. First-principles / scaling texts + framework docs + lore

| Topic | Scaling Book | Ultra-Scale | Horace "Brrrr" | GPU-MODE | Roofline'09 | NVIDIA GEMM | PT Tuning | torch.compile | CUDA Graphs | FSDP2 | AMP | Profiler | FlashAttn | nanoGPT | L.Weng |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Roofline / arith. intensity | ● | ○ | ● | �– | ● | ● | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ● | ✗ | ✗ |
| Memory hierarchy (HBM/SRAM) | ● | ○ | ● | ● | ● | ● | �– | �– | �– | ✗ | ✗ | ○ | ● | ○ | ✗ |
| Kernels & fusion | ○ | ● | ● | ● | ✗ | ○ | ● | ● | ✗ | ✗ | �– | ✗ | ● | ● | ✗ |
| Compilation / graph tracing | ● | �– | ○ | ○ | ✗ | ✗ | ● | ● | ● | ✗ | ✗ | ✗ | ✗ | ● | ✗ |
| Mixed precision (fp16/bf16/fp8) | ● | ● | �– | ● | ✗ | ○ | ● | �– | ✗ | ● | ● | ✗ | �– | ● | ○ |
| Mem opt / activation checkpointing | ○ | ● | ✗ | �all | ✗ | ✗ | ● | ✗ | ✗ | �– | ✗ | ✗ | ● | ✗ | ● |
| Profiling | ● | ● | ● | ● | ✗ | ○ | �– | ○ | ✗ | ✗ | ○ | ● | ✗ | ● | ✗ |
| Async / compute-comm overlap | ● | ● | ● | ○ | ✗ | ✗ | ● | ✗ | ● | ● | ✗ | ○ | �– | �– | �– |
| Data parallelism (DDP) | ● | ● | ✗ | ○ | ✗ | ✗ | ● | ✗ | ✗ | ● | ✗ | ✗ | ✗ | ○ | ● |
| Collectives / allreduce | ● | ● | ✗ | ● | ✗ | ✗ | �– | ✗ | ✗ | ● | ✗ | ✗ | ✗ | �– | �– |
| ZeRO / FSDP sharding | ● | ● | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ | ● | ✗ | ✗ | ✗ | �– | ● |
| Tensor parallelism † | ● | ● | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ | �– | ✗ | ✗ | ✗ | ○ | ● |
| Pipeline parallelism † | ● | ● | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ● |
| Inference perf (KV cache, batching) | ● | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | �– | ✗ |
| Quantization | ○ | �CP | ✗ | ● | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | �FA3 | ○ | ✗ |

(Cells like "�all/◊CP/◊FA3" = present but partial/only in a successor version.)

**One-line read of the matrix.** The topics that light up across *both* halves are: memory
hierarchy, arithmetic intensity / roofline, kernels & fusion, compilation, mixed precision,
activation checkpointing, profiling, async/overlap, data parallelism + collectives, and
FSDP/ZeRO sharding. Tensor/pipeline parallelism are widely taught but are the natural
scope-boundary (they only pay off across many devices/nodes). Inference-perf and quantization
are strong in the *systems/efficiency* courses (15-442, 6.5940, CS294) and the scaling book,
weaker in the first-principles perf texts.

---

## 2. Ranked "core canon" vs. electives vs. defer

### CORE CANON — essentially every strong treatment covers these (teach in the chapter)

Ranked by cross-source ubiquity and how load-bearing they are:

1. **The roofline / arithmetic-intensity model** — the organizing frame. (Scaling Book Ch. 1,
   NVIDIA GEMM guide, Horace He's three regimes, Roofline'09, CS336 L5, CS267, CS149.) Every
   serious modern treatment starts here or converges on it.
2. **Memory hierarchy & bandwidth** (registers/SRAM ↔ HBM ↔ DRAM; bandwidth as the true
   bottleneck). Inseparable from #1. (All of the above + FlashAttention.)
3. **The three performance regimes: compute-bound / memory-bandwidth-bound / overhead-bound**
   — Horace He's taxonomy, now the field's default diagnostic vocabulary.
4. **Kernel fusion** as the cure for the bandwidth regime (fewer HBM round-trips). (Horace He,
   Ultra-Scale §11, NVIDIA, FlashAttention, torch.compile/Inductor, GPU-MODE.)
5. **Compilation / graph capture** — `torch.compile` (Dynamo → AOTAutograd → Inductor/Triton)
   and `jax.jit` → XLA. What tracing buys, what breaks it. (15-442, 599W, torch.compile docs,
   Scaling Book.) **Explicitly in-scope per the author.**
6. **Async execution & launch overhead** — the CPU dispatch queue running ahead of the GPU;
   CUDA graphs / `reduce-overhead`; "avoid CPU-GPU sync" footguns. (Horace He, CUDA-graphs
   blog, PT tuning guide, profiler.)
7. **Mixed precision** — fp16+GradScaler / bf16 (no scaler) / fp8; Tensor-Core requirement;
   buys *both* speed and memory. (AMP recipe, NVIDIA, Ultra-Scale, scaling book, 6.5940.)
8. **Profiling methodology** — measure → classify the regime → fix. (PyTorch/JAX profiler,
   CS336 A2, GPU-MODE, scaling book Ch. 9.) The connective tissue between all topics.
9. **Activation / gradient checkpointing (rematerialization)** — trade compute for memory.
   (15-442 dedicated lecture, Ultra-Scale §2, Lilian Weng, FlashAttention backward, 6.5940.)
10. **GEMM / matmul performance** — tiling, Tensor Cores, tile/wave quantization; the critical
    batch size where a matmul crosses from memory- to compute-bound. (NVIDIA GEMM guide,
    CS267, 15-442 "GEMM on modern GPUs", scaling book.)
11. **Data parallelism + collectives** — DDP, ring all-reduce, gradient bucketing, overlapping
    the gradient all-reduce with the backward pass. The *first* multi-GPU technique. (All
    systems courses, FSDP tutorial, Ultra-Scale §3, tuning guide.)
12. **FSDP / ZeRO sharding** — the single-node multi-GPU payoff: shard optimizer states →
    gradients → parameters; the identity **all-reduce = reduce-scatter + all-gather** turns DDP
    into ZeRO. (FSDP2 tutorial, Ultra-Scale §4, scaling book, Lilian Weng, 15-442, CS336.)

### ELECTIVES — strong, common, but not universal; include as depth/optional sections

- **FlashAttention as a worked IO-awareness case study** (near-canonical; the single best
  concrete instance of #2–#4 — see §3).
- **Inference-time performance** — KV cache, prefill vs. decode, continuous/dynamic batching,
  PagedAttention, speculative decoding. (Scaling Book Ch. 7–8, 15-442, 6.5940, CS294.) Central
  to systems courses; a natural "single-node serving" elective, but leans toward its own chapter.
- **Quantization** (PTQ/int8/int4 for inference). (6.5940's signature, CS294, GPU-MODE.)
  Overlaps mixed precision; belongs more to an efficiency/inference chapter.
- **Writing a custom kernel** (Triton, or JAX/Pallas) — high-value hands-on capstone (CS336 A2
  FlashAttention-2 in Triton), but a big lift for a general chapter.
- **`channels_last` / memory-format, cuDNN autotune, data-loading pipeline** (PT tuning guide) —
  practical single-GPU wins worth a short "checklist" treatment.

### EXPLICITLY MULTI-NODE / DEFER (out of scope per author — mention as a bridge, don't teach)

- **Tensor parallelism** and **sequence parallelism** (Megatron-style) — only pays off with
  fast intra-node interconnect and beyond; the natural boundary.
- **Pipeline parallelism** (GPipe / 1F1B / interleaved / zero-bubble) — inherently multi-stage
  across devices; pipeline-bubble analysis is elegant but belongs to the distributed chapter.
- **Context / ring-attention parallelism, expert parallelism (MoE), 3D/5D parallelism** —
  Ultra-Scale §5–10; squarely multi-node.
- **Multi-machine collectives / network topology, parameter servers** — deferred (and see §5:
  parameter servers are now a historical footnote).

*Pedagogical bridge to keep:* teach DP + FSDP fully on one node, then state in one paragraph
that TP/PP/CP extend the same "shard something across devices" idea when one node is not enough
— forward-referencing the later Language Models part.

---

## 3. Best teaching resources per core topic + the best pedagogical device seen

For each core topic: the 2–3 best resources (URL) and the single best device to emulate.

**1. Roofline / arithmetic intensity**
- Best: **NVIDIA Matrix-Multiplication Background** (https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html); **Scaling Book Ch. 1** (https://jax-ml.github.io/scaling-book/roofline/); **Roofline'09**.
- Best device: **the two-GEMM contrast.** NVIDIA: on a V100 (Tensor-Core ratio ≈ **139 FLOP/byte**), an `8192×128×8192` GEMM has intensity **124 → memory-bound**; growing K to `8192×8192×8192` gives intensity **2730 → math-bound**. *Same operation, one number changed, regime flips.* Pair with the scaling book's algebra: for `X[B,D]·Y[D,F]`, intensity `≈ B` (when B≪D,F), so "a bf16 matmul is compute-bound only once the per-device token batch exceeds ~240 (TPU v5e) / ~295 (H100)" — **the critical-batch-size = ridge-point argument.** Emulatable notebook: sweep matrix size / batch, plot achieved FLOP/s (log-log), watch the memory-bound ramp bend into the compute-bound roof.

**2. Memory hierarchy & bandwidth**
- Best: **Horace He "Brrrr"** (https://horace.io/brrr_intro.html); **FlashAttention** (https://arxiv.org/abs/2205.14135); **CS336 L5 GPUs**.
- Best device: **Horace's factory/warehouse analogy** — compute = a fast small factory (SRAM/registers), DRAM = a big slow warehouse, bandwidth = the trucks; compute has outgrown bandwidth for generations, so the factory increasingly idles waiting on trucks. Plus FlashAttention's **memory pyramid** ("SRAM ~19 TB/s but ~1000× smaller than HBM ~1.5 TB/s"). Emulatable: microbenchmark that increases compute-per-byte (`x*=2` repeated N times) and shows runtime *flat* (memory-bound) until N crosses a threshold, then linear.

**3. Three regimes (compute / bandwidth / overhead)**
- Best: **Horace He "Brrrr"** — the definitive statement.
- Best device: the **regime table + the `x.cos().cos()` fusion example** (unfused = 4 global-mem ops, fused = 2 → ~2×; "a fused `x.cos().cos()` costs almost the same as one `x.cos()` — which is why all activation functions cost about the same regardless of op count"). And **"in the time Python does one FLOP, an A100 could do 9.75M FLOPs"** to make the overhead regime visceral. Diagnosis-before-optimization is the thesis: *know your regime first.*

**4. Kernel fusion**
- Best: **FlashAttention** (the money example); **torch.compile/TorchInductor** (https://docs.pytorch.org/docs/stable/torch.compiler.html); **Ultra-Scale §11**.
- Best device: FlashAttention's **"tile → online-softmax → fuse → recompute-in-backward"** storyline — fuse the whole attention into one kernel, never materialize the N×N matrix in HBM, recompute it in the backward pass from saved stats. Concrete before/after: **GPT-2 3× faster, 5–20× less memory.** Ties fusion directly back to the roofline (attention is memory-bound, so cut HBM traffic, not FLOPs).

**5. Compilation / graph capture** *(explicitly in-scope)*
- Best: **torch.compile tutorial + `torch.compiler` architecture docs**; **Scaling Book Ch. 10** (JAX/XLA); **CMU 15-442 ML-compiler lectures** (https://mlsyscourse.org/schedule).
- Best device: **the eager-vs-compiled median-time benchmark with an explicit warm-up print** (tutorial: 4096² matmul, eager 0.87 ms → compiled 0.37 ms = **2.34×**, first call 0.54 s to teach amortization) **plus the graph-break narrative** — a data-dependent `if` splits the graph and costs the speedup, motivating *why* tracing needs guards. Frame the compiler as a 4-stage pipeline: **Dynamo (capture) → AOTAutograd (backward graph) → Inductor/Triton (fuse+codegen) → CUDA graph (replay).** JAX contrast: `jax.jit` traces a *functional* program to XLA HLO — no graph breaks, but recompiles on shape/dtype change (see §6).

**6. Async execution & launch overhead**
- Best: **CUDA Graphs blog** (https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/); **PT tuning guide** ("avoid CPU-GPU synchronization" item); **Horace He** (async section).
- Best device: the **"CPU-bound at small batch → GPU bubbles" timeline diagram** — kernels are so small that the GPU idles between launches while Python dispatches; a CUDA graph captures the sequence once and replays it with one ~10 µs launch. Complementary teaching footgun list: `.item()`, `.cpu()`, `print(tensor)`, `.nonzero()` are hidden syncs that stall the pipeline.

**7. Mixed precision**
- Best: **PyTorch AMP recipe** (https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html); **NVIDIA mixed-precision guide**; **Ultra-Scale §11**.
- Best device: the **AMP recipe's incremental "add one thing, re-measure time AND memory"** structure — fixed workload, four stages (default → +autocast → +GradScaler → all), each reporting *both* wall-time and peak GPU memory. This is the single best experiment *template* in the whole survey; the chapter should reuse its shape everywhere. Teaches the two ideas cleanly: autocast picks per-op dtype (fp16 matmul, fp32 reductions); GradScaler rescales the loss so small fp16 grads don't underflow (bf16 needs no scaler). Gains are **Tensor-Core-only, ~2–3×.**

**8. Profiling methodology**
- Best: **PyTorch Profiler recipe** (https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html); **Scaling Book Ch. 9** (JAX/XLA, Perfetto); **GPU-MODE Lec 1/8/16**.
- Best device: the **CPU-time-vs-CUDA-time table** (`key_averages().table()`) that exposes whether you are launch-bound (CPU time ≫ CUDA time) or compute-bound — this *is* the "which regime am I in?" measurement, and it dictates the fix (async/CUDA-graph vs. fusion vs. bigger tensors). Chrome/Perfetto trace export for the timeline view.

**9. Activation / gradient checkpointing**
- Best: **CMU 15-442 "Memory Optimizations: rematerialization & offload"**; **Ultra-Scale §2**; **Lilian Weng**.
- Best device: the **memory-anatomy breakdown of a transformer** (params + grads + Adam optimizer states + activations) from Ultra-Scale, with a live memory-profiler trace, *then* show activation recompute collapsing the activation term — motivate the technique by the bar it removes. Quantify the compute-for-memory trade (one extra forward per checkpointed block).

**10. GEMM / matmul performance**
- Best: **NVIDIA Matrix-Multiplication Background**; **CS267 cache-blocked matmul assignment**; **CMU 15-442 "GEMM on Modern GPUs"** (https://mlsyscourse.org/slides/modern-gpu-gemm/).
- Best device: **CS267's hand-tuned cache-blocked matmul graded against the roofline** — students *write* a tiled GEMM and measure achieved vs. peak. Plus NVIDIA's **tile/wave-quantization** lesson (why a matmul dim that isn't a multiple of the tile wastes a whole "wave" of SMs — a surprising, memorable perf cliff).

**11. Data parallelism + collectives**
- Best: **PyTorch DDP + FSDP tutorial**; **Ultra-Scale §3**; **Scaling Book** (collective-cost derivations).
- Best device: the scaling book's **closed-form cost of each collective** (all-gather / reduce-scatter / all-reduce as bytes-on-the-wire) → a *communication roofline* whose ridge depends on model dim, contrasted with the compute roofline; plus DDP's **overlap the gradient all-reduce with the backward pass** (bucketing) — the first "hide communication under compute" lesson.

**12. FSDP / ZeRO sharding**
- Best: **FSDP2 tutorial** (https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html); **Lilian Weng** (ZeRO stage taxonomy); **Ultra-Scale §4**.
- Best device: the **all-reduce = reduce-scatter + all-gather identity** — the one equation that converts DDP (replicate everything) into FSDP/ZeRO (shard parameters, all-gather just-in-time for fwd/bwd, free after, reduce-scatter the grads). Pair with Lilian Weng's **ZeRO memory table** ("which of the three — params/grads/optimizer states — do we shard, and memory drops ~1/N"). The FSDP param-lifecycle animation (sharded → all-gather → compute → free → reduce-scatter) + a peak-memory-vs-DDP plot.

---

## 4. Narrative-arc observations — the order the best treatments use, and why

Across the strongest treatments the sequencing has *converged*. Five observations:

**(i) First-principles FLOP/byte accounting comes BEFORE any API.** The best courses open by
hand-counting FLOPs and bytes to derive a bound (CS336 L2 "resource accounting" → L5 rooflines;
Scaling Book Ch. 1; CS267's roofline lecture; NVIDIA's guide). Not one strong source opens with
`torch.compile`. **Implication:** the chapter should lead with roofline / arithmetic intensity,
so every later technique is explained as "which bound it relieves," not as a bag of tips.

**(ii) The universal spine is: hardware/roofline → single-GPU kernels & fusion → memory/precision
→ compilation & async → then multi-GPU (DP → sharding).** This exact ordering recurs from UW
599W → both CMU courses → CS336 → the scaling book. Concretely (the safe table of contents):
*GPU architecture & memory hierarchy → arithmetic intensity / roofline → GEMM & Tensor Cores →
the three regimes → kernel fusion (FlashAttention as the case study) → mixed precision → memory
optimization (checkpointing) → compilation (graph capture) & async/CUDA-graphs → profiling as
the connective method → data parallelism + collectives → FSDP/ZeRO → (bridge to TP/PP/distributed).*

**(iii) "Each section removes the previous section's wall."** The scaling texts (Ultra-Scale,
Lilian Weng) are structured as a bottleneck ladder: single GPU runs out of memory → recompute +
mixed precision → still too slow → data-parallel across GPUs → optimizer/grad/param memory is
redundant → shard it (ZeRO/FSDP) → the tensors themselves are too big → shard them (TP, *next
chapter*). Motivate every technique by the specific wall it knocks down, shown with a memory or
time measurement, *before* naming the API.

**(iv) Profiling is taught as the connective methodology, not a standalone topic.** The best
sequencing interleaves "measure → classify the regime → apply the matching remedy → re-measure"
(CS336 A2 chains profile → custom kernel → parallelism; the AMP recipe re-measures at every
step). No single existing source teaches this whole measure→classify→fix loop end-to-end — that
gap is the chapter's natural spine and its differentiator.

**(v) Modern treatments end with an integrative wall-clock case study.** CS336's Assignment 2
(profile → write FlashAttention-2 in Triton → memory-efficient distributed training) and the
**modded-nanoGPT speedrun** are the model closer: take one concrete training run and drive its
wall-clock down, attributing each win to a mechanism already taught. modded-nanoGPT's "what moved
the needle" is a ready-made waterfall: **FlexAttention (compiled block-sparse attention) ≈ 30% —
the single biggest jump**, then the **Muon** optimizer + architecture modernization (≈ halved
steps), then **low precision (bf16 baseline → fp8 on head/MLP)**, then **fused Triton kernels /
FlashAttention-3**, with `torch.compile` + bf16 assumed as table stakes underneath everything.

---

## 5. What modern treatments deliberately SKIP that older ones taught

Things a 2015–2019 "performance/distributed DL" treatment emphasized that current curricula drop
(or demote to a footnote) — the d2l chapter should skip these too:

1. **Parameter servers** (async PS, stale-gradient SGD, PS vs. worker topologies). All-reduce /
   collective-based data parallelism won; PS survives only as a one-line historical note (if at
   all) in current syllabi. The old d2l chapter's parameter-server section is exactly this.
2. **The symbolic-vs-imperative ("define-and-run" vs. "define-by-run") debate.** Once a headline
   framework-design argument (TF1 graphs vs. PyTorch eager, MXNet's hybridize). Settled: everyone
   is eager-by-default *plus a tracing compiler* (`torch.compile`, `jax.jit`). Teach the compiler
   as an optimization layer, not as a philosophical fork. Drop `HybridSequential`/`hybridize` and
   the imperative/symbolic framing.
3. **Manual hybrid front-ends / hand-written static graphs** (MXNet Gluon hybridize, TF1
   `Session.run`, hand-built `tf.Graph`). Replaced by automatic capture; no one hand-authors a
   static graph anymore.
4. **Asynchronous / bounded-staleness SGD as a *performance* device.** Modern single-node
   training is synchronous; async is a niche distributed topic, not a core perf lever.
5. **fp32-everywhere and "just use more/bigger GPUs" framing.** Replaced by bf16/fp8 as the
   default and by *arithmetic-intensity-aware* reasoning (make the hardware busy) rather than
   raw device count.
6. **NCCL/ring-allreduce internals as a headline topic.** Modern treatments *use* the collective
   and reason about its cost (bytes on the wire), but teach ring-vs-tree allreduce mechanics only
   in passing — the abstraction (all-reduce cost model) matters more than the implementation.
7. **CPU/multi-core parallelism and MPI-style HPC** (still central in CS267/CS149) is *not* what
   a DL perf chapter foregrounds today — GPUs, the memory hierarchy, and collectives are.
8. **Over-indexing on convolution/im2col performance.** Older DL-perf material centered CNN
   kernels; modern material centers **matmul + attention** (Transformers), with FlashAttention as
   the canonical kernel case study.

---

## 6. PyTorch vs. JAX pedagogy (framework-contrast section)

The live 2024–2026 *pedagogical* corpus is overwhelmingly PyTorch-centric (tuning guide,
AMP recipe, nanoGPT speedrun, GPU-MODE). JAX's equivalents exist as high-quality official
docs (docs.jax.dev — living, undated pages) plus the scaling book (the *canonical* JAX perf
text). The chapter should present the two frameworks as the same ideas with different
capture/execution models. Mapping per topic:

**1. Compilation — `jax.jit` → XLA.** (https://docs.jax.dev/en/latest/jit-compilation.html ·
thinking-in-JAX https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html · AOT
https://docs.jax.dev/en/latest/aot.html) Trace-once with tracer objects → build a **jaxpr** →
lower to **XLA HLO** → XLA **auto-fuses** into kernels → cache keyed on input shape/dtype.
Pedagogy: **(a) first-principles + (b) measurement** — `jax.make_jaxpr()` lets you *see* the
captured graph; `%timeit` shows SELU 5.85 ms → 659 µs (~9×). **Contrasts with `torch.compile`
to teach:** functional/trace-based, not bytecode capture → **no graph breaks** (unsupported
Python simply can't touch traced values); Python control flow can't branch on a traced value
(`TracerBoolConversionError`) → fix with `static_argnums` or `lax.cond`/`lax.scan`;
**recompilation on new shape/dtype** is the JAX-specific footgun; XLA auto-fuses so JAX needs
fewer manual fusion tricks. `donate_argnums` = JAX's in-place buffer-reuse analogue. Best
device: `make_jaxpr()` printout (side effects vanish → "trace captures only tensor ops; purity
matters").

**2. Checkpointing — `jax.checkpoint`/`jax.remat`.**
(https://docs.jax.dev/en/latest/gradient-checkpointing.html) Default autodiff stores all
residuals; `remat` switches to recompute-in-backward. Distinctive: **save-policies**
(`checkpoint_dots`, `checkpoint_name`) let you control *what is saved without editing the
model*, and `print_saved_residuals()` lets you *watch* which tensors are kept. Pedagogy:
**(a) first-principles + empirical inspection.** Cleaner than PyTorch's wrap-the-module
`torch.utils.checkpoint`.

**3. Sharding / single-node multi-GPU — THE headline contrast.**
(intro https://docs.jax.dev/en/latest/sharded-computation.html · deep
https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
· manual `shard_map` https://docs.jax.dev/en/latest/notebooks/shard_map.html) Build a **Mesh**
→ describe layout with **PartitionSpec** in **NamedSharding** → pass sharded arrays into an
ordinary `jax.jit` function → the **GSPMD** partitioner shards intermediates and **inserts the
collectives automatically**. `shard_map` is the manual escape hatch (per-shard local code with
explicit `lax.psum` etc.). **The chapter's headline framework contrast:** PyTorch DDP/FSDP are
*explicit/imperative* (you wrap the model and know an all-reduce or reduce-scatter+all-gather
fires; separate APIs per strategy); JAX is *declarative/compiler-driven* — same physical
collectives, never hand-written, **one mechanism (change the PartitionSpec) spans
data-/tensor-/FSDP-style sharding.** Best device: `jax.debug.visualize_array_sharding` grid
before/after a jitted matmul — you literally see the array split across devices and the result
come back sharded.

**4. Mixed precision.** (DeepMind **jmp** https://github.com/google-deepmind/jmp) JAX has **no
autocast and no global GradScaler** — you set dtypes explicitly; `jmp.Policy(compute/param/
output dtype)` casts, `jmp.LossScale` (static/dynamic) does fp16 scaling by hand; **bf16 is the
default idiom** (usually no scaling). Pedagogy: **(c) API/library** — less pedagogical than the
PyTorch AMP recipe. Contrast: PyTorch implicit op-by-op autocast + automatic scaler vs JAX
explicit policy threaded through the model (every cast visible, more manual).

**5. Profiling — `jax.profiler`.** (https://docs.jax.dev/en/latest/profiling.html)
`jax.profiler.trace(..., create_perfetto_link=True)` → **Perfetto**, or **TensorBoard/XProf**
(Trace Viewer + a **Graph Viewer showing the HLO + sharding** with no PyTorch analogue — "see
what the compiler did"). Pedagogy **(b) measurement**.

**6. Custom kernels — Pallas (the Triton analogue).**
(https://docs.jax.dev/en/latest/pallas/index.html) A JAX kernel language: tile/block-level
kernels in `jax.numpy`-style Python; **on GPU lowers via Triton, on TPU via Mosaic → portable
across GPU/TPU** (Triton is GPU-only). Flagged experimental. Framing: **because XLA auto-fuses,
JAX users reach for hand-kernels far less than eager-PyTorch users;** Pallas is the escape hatch
for FlashAttention-like cases XLA can't fuse.

**7. Async dispatch.** (https://docs.jax.dev/en/latest/async_dispatch.html) JAX dispatches
asynchronously like PyTorch CUDA — Python runs ahead enqueuing work. Taught as the **#1
benchmarking pitfall:** a naive timer measures only *dispatch* → you must `.block_until_ready()`
(direct parallel to PyTorch's `torch.cuda.synchronize()`).

### PyTorch vs. JAX contrast table (to reproduce in the chapter)

| Topic | PyTorch | JAX | Contrast to teach |
|---|---|---|---|
| **Compilation** | `torch.compile`: bytecode capture (Dynamo) + guards + **graph breaks**; Inductor→Triton | `jax.jit`: trace values → jaxpr → XLA HLO; **no graph breaks**; recompiles on shape/dtype; static args | imperative-with-fallback vs functional-pure; graph breaks vs recompilation |
| **Sharding / multi-GPU** | **explicit** DDP (all-reduce) / FSDP (reduce-scatter+all-gather); separate APIs per strategy | **declarative** Mesh+PartitionSpec+jit; **GSPMD inserts collectives**; one API for DP/TP/FSDP; `shard_map` = manual | "write the collective" vs "annotate layout, compiler writes it" — the headline contrast |
| **Checkpointing** | `torch.utils.checkpoint` wraps a module | `jax.checkpoint`/`remat` + **policies**; `print_saved_residuals` | wrap-module vs functional decorator + save-policy |
| **Mixed precision** | **autocast** (implicit op dtypes) + **GradScaler** (auto) | explicit dtypes / `jmp.Policy`; bf16-by-default; manual loss scale | implicit/automatic vs explicit/manual |
| **Custom kernels** | Triton (GPU-only), CUDA C++ | **Pallas** (GPU via Triton, TPU via Mosaic — portable); needed less (XLA auto-fuses) | eager needs fusion tricks; XLA fuses for free |
| **Profiling** | `torch.profiler` + TensorBoard/Chrome trace | `jax.profiler` + Perfetto / **XProf incl. HLO Graph Viewer** | both timeline-based; JAX adds compiler-graph + sharding view |

**Two through-lines for the chapter:** (1) the deepest PyTorch↔JAX split is **sharding** —
imperative collectives vs compiler-synthesized (GSPMD); build the multi-GPU section around that
contrast. (2) JAX pushes work onto the **XLA compiler** (auto-fusion, GSPMD comms, remat
interplay), so it teaches *fewer manual perf tricks* but *more "understand what the compiler
did"* (jaxpr/HLO viewer) — a clean complement to PyTorch's checklist style. Canonical JAX perf
reference throughout: **How to Scale Your Model** (jax-ml.github.io/scaling-book).

---

## 7. Bottom-line recommendations for the chapter

- **Open with the roofline** and the three regimes; make arithmetic intensity the spine so every
  later technique is "which bound it relieves."
- **Adopt the AMP-recipe experiment shape** everywhere: fixed workload, add one technique,
  re-measure *both* wall-time and peak memory.
- **Use FlashAttention as the memory-hierarchy set-piece** and **modded-nanoGPT as the closing
  wall-clock waterfall.**
- **Teach compilation as a 4-stage pipeline** (capture → backward → fuse/codegen → replay), and
  present `torch.compile` and `jax.jit` side by side as the same idea with different capture
  mechanisms (bytecode vs. functional trace).
- **Do DP + FSDP fully on one node**, then bridge in one paragraph to TP/PP as the "when one node
  isn't enough" topic deferred to the distributed chapter.
- **Skip** parameter servers, the symbolic/imperative debate, and manual hybrid engines (§5).
