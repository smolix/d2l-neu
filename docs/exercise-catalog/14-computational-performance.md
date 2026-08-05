# Chapter Overview: chapter_computational-performance

This chapter's 7 sections (42 exercises total) are already the strongest-specified
in the book: the prior style review found zero clarity flags across all 42 items
and only 3 formatting defects, though also zero use of the book's new
named+tagged exercise style. Best external match overall: **Stanford CS336**
("Language Modeling from Scratch") **Assignment 2 (Systems)** — an end-to-end
profile → Triton FlashAttention2 → DDP → optimizer-state-sharding pipeline on a
real Transformer that mirrors fast-transformer.md and multi-gpu-practice.md
almost exactly, and whose Lecture 2 "Resource Accounting" teaches the identical
16-bytes-per-parameter rule as memory-precision.md. Second best: **UC Berkeley
CS267**'s "Assignment 1: Optimizing Matrix Multiplication" (measured FLOP/s vs.
theoretical peak, cache blocking) is the direct ancestor of the roofline/ridge-
point exercises in performance-model.md and hardware.md. **UW CSE 599W**
(Tianqi Chen, 2018) Assignment 2 (TVM graph executor, mandatory 10x-speedup
matmul schedule) is the best analogue for compilation.md, though newer courses
have largely dropped "build a tracing compiler" as homework. Ring-allreduce's
best source isn't a course at all — Andrew Gibiansky's 2017 Baidu SVAIL
engineering blog derives the exact cost formula multiple-gpus.md builds; the
PyTorch Distributed (2020) and ZeRO (2019) papers cover DDP bucketing and
stage-3 sharding respectively. Real gaps with **no good external exercise
tradition**: CPU false sharing (hardware.md), reproducing an NCCL transport
hang (multi-gpu-practice.md), and a dedicated gradient-accumulation sweep
(memory-precision.md) — all classic-but-uncovered or box-specific material.
`dlsyscourse.org` (CMU/UW 10-414/714), despite being the most obvious "build
your own DL systems" course, stops at single-node NDArray/CPU-GPU backends and
never reaches distributed training or graph-fusion as a graded assignment —
notable given the course's reputation. MIT 6.5940 is well-matched in spirit but
its labs target post-training compression/quantization for inference, not this
chapter's training-time concerns — cited only as low-overlap background.
Because the existing sets are already excellent, this catalog is disposition-
heavy on **keep**; new problems are added only where a genuinely stronger
external framing exists, and the only rewrites are the 4 formatting fixes
already flagged by the prior style review (hardware.md ex2/ex3,
fast-transformer.md ex5/ex8) — no exercise anywhere in the chapter is dropped.

---

## chapter_computational-performance/performance-model.md — The Performance Model

**Topic:** the roofline-style performance model (arithmetic intensity vs. a
machine's ridge point) and the discipline of correct, synchronized GPU timing.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — all five are
concrete measurement tasks with a stated comparison and no formatting defects;
this is one of the chapter's two strongest sets (tied with memory-precision.md)
and should not be diluted.

**External sources found:**
- UC Berkeley, CS267 "Applications of Parallel Computers," Assignment 1 "Optimizing Matrix Multiplication" (Spring 2024) — students write an optimized single-threaded `dgemm` for NERSC's Perlmutter and report measured GFLOP/s as a percentage of the machine's computed theoretical peak (3.5 GHz × 4-wide vector × 2 pipelines × 2 FMA ops = 56 GFLOP/s) — https://sites.google.com/lbl.gov/cs267-spr2024/hw-1
- Modular, "Mojo GPU Puzzles," Puzzle 16 "Roofline Model" — walks through computing arithmetic intensity and sustained performance for a naive matmul kernel on an A100, as prerequisite reasoning for a later tiling/fusion puzzle (not itself a graded exercise) — https://puzzles.modular.com/puzzle_16/roofline.html
- GPU MODE, "reference-kernels" repository — hosts graded, leaderboard-scored kernel-optimization problems (including "PMPP practice problems" derived from *Programming Massively Parallel Processors*) with a defined reference-implementation/test-case submission format — https://github.com/gpu-mode/reference-kernels
- GPU MODE lecture series, Lecture 1 "Profiling and Integrating CUDA kernels in PyTorch" — covers exactly this section's CPU-vs-device timing discipline via `torch.profiler`, though as lecture content rather than a graded problem — https://github.com/gpu-mode/lectures
- NERSC, "Roofline Performance Model" documentation — reference/tool documentation (ERT, Nsight Compute) with one small case study sweeping a kernel's loop-iteration count to shift its arithmetic intensity; not exercise-formatted, cited as technique background only — https://docs.nersc.gov/tools/performance/roofline/

**Proposed problem set** (6 problems):
1. [conceptual] **Ridge point from spec sheet.** Compute the ridge point of the reader's own GPU from its specification sheet and compare it against a measured matmul-size sweep; redo for fp32 vs. bf16 and report where the measured curve leaves the bandwidth wall.
   *Provenance:* original (existing ex1, kept).
1. [short-code] **Batch-size crossover.** For a fixed-width batched matmul, predict the batch size at which the operation crosses the GPU's ridge point, then measure achieved TFLOP/s for $B \in \{1, 8, 64, 512, 4096\}$ and check the prediction.
   *Provenance:* original (existing ex2, kept).
1. [short-code] **Print-statement tax.** Add a `print(loss)` on every training iteration and measure the per-step slowdown as a function of batch size; explain why the *relative* damage shrinks as the batch grows.
   *Provenance:* original (existing ex3, kept).
1. [short-code] **Classify a real training step.** Profile a training step whose `DataLoader` uses `num_workers=0` and classify it into one of the three regimes using the CPU-vs-device-time rule; state whether it is a GPU problem at all.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Synchronization triggers.** Explain in one sentence why `x.nonzero()` forces a synchronization while `x + 1` does not, and find one more operation with the same property.
   *Provenance:* original (existing ex5, kept).
1. [short-code] **Peak-fraction and a blocked rewrite.** Measure achieved TFLOP/s as a percentage of theoretical peak across the matmul-size sweep (CS267 HW1's reporting convention), then rewrite a bandwidth-bound elementwise operation to process a large tensor in fixed-size chunks instead of all at once, and check whether chunking changes achieved bandwidth.
   *Provenance:* adapted from UC Berkeley CS267 Assignment 1 (overlap low — the "report as % of theoretical peak" convention is adopted; the hand-written-C register/cache blocking is not, since the book has not introduced CUDA-level code at this point).

---

## chapter_computational-performance/hardware.md — Hardware

**Topic:** where peak FLOP/s and bandwidth come from; the memory hierarchy,
bandwidth/latency ladders, chip-boundary cost rule, energy of data movement,
interconnects, and false sharing.
**Current exercises:** 8; disposition: keep 6, rewrite 2, drop 0 — content is
uniformly well-specified (the review's clarity check found zero issues); the
two rewrites are pure formatting fixes already flagged by the prior style
review (ex2's crammed inline (a)/(b) lettering, ex3's unmatched `$` that opens
an unterminated math span through the rest of the section) — no content
changes, so nothing is lost by fixing them.

**External sources found:**
- UC Berkeley, CS267, Assignment 1 "Optimizing Matrix Multiplication" (Spring 2024) — same source as performance-model.md; its blocking/register-tiling techniques are the direct ancestor of this section's cache-hierarchy reasoning — https://sites.google.com/lbl.gov/cs267-spr2024/hw-1
- CMU/UW, 10-414/714 "Deep Learning Systems" (dlsyscourse.org, 2025 offering), Homework 3 "Hardware Acceleration" — builds a CPU and CUDA NDArray backend including a cache-aware tiled CPU matmul and a GPU version using cooperative fetching and shared-memory register tiling — https://github.com/dlsyscourse/hw3 (assignment schedule confirmed at https://dlsyscourse.org/assignments/)
- UW, CSE 599W "Systems for ML" (Tianqi Chen, 2018), Assignment 2 "Graph Executor with TVM" — requires a matmul schedule that hits at least 10x speedup over the default via blocking, vectorization, and loop permutation, i.e. hand-tuning for the same memory hierarchy this section describes — https://github.com/wyc-ruiker/CSE-599W-2018/tree/master/assignment2 (student-maintained mirror; the original dlsys-course.github.io assignment pages are no longer reachable directly)
- NERSC, "Roofline Performance Model" documentation — bandwidth/latency measurement tooling (ERT, Nsight Compute) relevant to this section's cache-cliff and ladder figures; reference material, not an exercise — https://docs.nersc.gov/tools/performance/roofline/
- GPU MODE lecture series, Lecture 8 "CUDA Performance Checklist" and Lecture 37 "Introduction to SASS & GPU Microarchitecture" — lecture coverage of hardware-aware optimization and cache/occupancy effects; inspirational only, not graded problems — https://github.com/gpu-mode/lectures
- **No good external exercise tradition found** for CPU false sharing / cache-line padding (this section's ex7) specifically in an ML-systems context — it is classic computer-architecture material (textbooks like Hennessy & Patterson, already cited in this section's prose) but no ML course was found posing it as a homework problem.

**Proposed problem set** (8 problems — already at the range's upper bound; no additions needed given the strength of the existing set):
1. [conceptual] **Ridge point across GPU generations.** Compute the ridge point for several real GPUs at bf16 and fp8; reconcile the direction of movement with the shoreline argument and explain why the long-run trend differs from the two most recent steps.
   *Provenance:* original (existing ex1, kept).
1. [conceptual] **Prefill vs. decode regime.** Estimate a GPT model's prefill arithmetic intensity and its decode tokens-per-second bound from its parameter count and the reader's GPU specs, and identify which regime each is in.
   *Provenance:* original (existing ex2, kept — reformat the inline "(a) ... (b) ..." into a nested list; content unchanged).
1. [conceptual] **Cost of an epoch's memory traffic.** Estimate the DRAM-traffic energy of one Fashion-MNIST training epoch and convert it to a dollar figure at a stated electricity price.
   *Provenance:* original (existing ex3, kept — fix the unmatched `$` so the following prose does not render as one open math span; content unchanged).
1. [short-code] **Measuring the cache cliff.** Time a simple elementwise operation across tensor sizes from 1 MB to 2 GB, plot achieved bandwidth against size, and locate the memory-hierarchy level transitions.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Why HBM sits on an interposer.** Answer using the shoreline (wire-count) argument, then check the answer against the actual signal-wire counts of a DIMM socket versus an HBM stack's interface.
   *Provenance:* original (existing ex5, kept).
1. [short-code] **Burst vs. random access.** Compare summing a large tensor in natural order against summing it through a random index permutation, and explain the bandwidth ratio using the address-setup-versus-streaming model.
   *Provenance:* original (existing ex6, kept).
1. [short-code] **False sharing.** Write a two-thread CPU program where each thread increments its own counter, first adjacent in memory and then padded to separate cache lines; measure and explain the resulting speedup.
   *Provenance:* original (existing ex7, kept — no external exercise tradition found for this specific task; see chapter overview).
1. [conceptual] **Cheapest fixes for a starved GPU.** Given a DataLoader that delivers batches slower than the GPU consumes them, rank the three cheapest interventions in order of expected payoff.
   *Provenance:* original (existing ex8, kept).

---

## chapter_computational-performance/compilation.md — Compute Graphs and Compilation

**Topic:** eager vs. traced/compiled execution (`torch.compile`/`jax.jit`),
graph breaks, retracing, kernel fusion, and when compilation pays off.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — every item
is a concrete debugging/measurement task tied to a named API, with no defects
found in the prior review; the one real cross-reference defect the review
flagged (a hardcoded "§13.6" instead of `:numref:`) is in fast-transformer.md,
not here.

**External sources found:**
- UW, CSE 599W "Systems for ML" (Tianqi Chen, 2018), Assignment 2 "Graph Executor with TVM" — closest external analogue: build a computation-graph executor and a matmul schedule required to clear a 10x speedup target via blocking/vectorization/loop reordering — the same "trace, then optimize the schedule" structure this section teaches, in an older toolchain — https://github.com/wyc-ruiker/CSE-599W-2018/tree/master/assignment2
- CMU/UW, 10-414/714 "Deep Learning Systems" (dlsyscourse.org, 2025) — checked all 5 homeworks directly (HW0-HW4 via github.com/dlsyscourse); HW3 covers CPU/GPU NDArray backends and HW4 covers convolution/RNN/LSTM ops, but **no homework covers a tracing compiler or operator-fusion pass** as a graded exercise — their only compilation-adjacent material is a lecture Colab notebook, not an assignment — https://dlsyscourse.org/assignments/, https://colab.research.google.com/github/dlsyscourse/lecture14/blob/main/14_hardware_acceleration_architecture_overview.ipynb
- PyTorch official documentation, "torch.compile Troubleshooting" and "Use `fullgraph=True` to Identify and Eliminate Graph Breaks" — the canonical reference for `TORCH_LOGS=trace_bytecode`/`trace_source` and the `fullgraph=True` workflow this section's ex1 exercises; reference documentation, not a course exercise, but directly citable as the technique's primary source — https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html, https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_true.html
- **No good external exercise tradition found** for "diagnose and fix a graph break with `torch._dynamo.explain`" or "diagnose a `jax.jit` retrace" specifically as course homework — GPU MODE's lecture series (checked directly) is CUDA/Triton/CUTLASS-focused and has no lecture on TorchDynamo internals; this appears to be a genuinely under-taught skill relative to how central it is to modern practice.

**Proposed problem set** (6 problems):
1. [short-code] **Fixing a graph break.** Introduce a data-dependent `if` into a `torch.compile`d function, locate the break with `torch._dynamo.explain`, rewrite the control flow with `torch.where`, and confirm the break count drops to zero; report the change in steady-state time.
   *Provenance:* original (existing ex1, kept).
1. [short-code] **Diagnosing a `jax.jit` retrace.** Force retraces by calling a jitted function with three different input lengths in a loop; fix it by padding to a common length; explain why `static_argnums` is not a fix here, verifying with a counter, and state when it is the right tool.
   *Provenance:* original (existing ex2, kept).
1. [short-code] **Capture-and-replay crossover.** Sweep the depth of a thin-layer stack at fixed width and plot eager time against `reduce-overhead` time; find the depth at which capture-and-replay starts to win and explain why the crossover exists.
   *Provenance:* original (existing ex3, kept).
1. [conceptual] **When compilation is a no-op.** Compile the matmul-size sweep from the performance-model section; identify the sizes where compiled time equals eager time and explain why in terms of what a single large matmul already is.
   *Provenance:* original (existing ex4, kept).
1. [short-code] **Repaying the compile tax.** Time the first call and the tenth call of a compiled mid-sized training step; compute how many steady-state steps are needed to repay the first-call compile cost.
   *Provenance:* original (existing ex5, kept).
1. [short-code] **Hit a speedup target.** Pick a fusible chain of elementwise operations from earlier in the section; without changing the math, search compilation modes/backend flags until steady-state throughput is at least 5x the eager baseline, and report which configuration got there.
   *Provenance:* inspired by UW CSE 599W Assignment 2's "clear at least 10x speedup" framing (overlap low — the target-speedup structure is adopted; the TVM schedule-search mechanics are not, since this section's toolchain is `torch.compile`/`jax.jit` configuration rather than a hand-written schedule).

---

## chapter_computational-performance/memory-precision.md — Memory and Precision

**Topic:** the 16-bytes-per-parameter training-memory rule, mixed-precision
training and loss scaling, activation checkpointing, and gradient accumulation.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the review
found this the strongest file in the chapter (ex4's deliberate-failure-then-fix
structure was singled out as exemplary); no defects, no clarity issues.

**External sources found:**
- Stanford, CS336 "Language Modeling from Scratch," Lecture 2 "PyTorch, Resource Accounting" and Assignment 1 ("basics") — teaches the identical accounting: FLOPs $\approx 6 \times$ tokens $\times$ parameters, and memory as parameters + gradients + optimizer state at 16 bytes/parameter for fp32 Adam, then asks students to compute peak memory/FLOPs for a stated Transformer configuration — near-exact overlap with this section's ex1/ex2 — https://cs336.stanford.edu/
- NVIDIA, "Train With Mixed Precision" developer guide — gives a real worked number: an SSD network loses 31% of its gradient values to fp16 underflow with no loss scaling, recovered with a scale of 8 (or a scale of 32,768 to keep 99.9% of values); documents the dynamic loss-scaling heuristic (grow by 2x every ~2000 good steps, halve on overflow) that `GradScaler` implements — https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
- Micikevicius, Narang, Alben, Diamos, Elsen, Garcia, Ginsburg, Houston, Kuchaiev, Venkatesh, Wu, "Mixed Precision Training," ICLR 2018 (arXiv:1710.03740) — the origin paper for the fp32-master-weights-plus-loss-scaling scheme that `GradScaler` implements, i.e. the mechanism this section's ex4 breaks and repairs.
- Chen, Xu, Zhang, Guestrin, "Training Deep Nets with Sublinear Memory Cost," 2016 (arXiv:1604.06174) — origin of the $O(\sqrt{n})$-memory checkpointing result behind this section's ex3; note on verification: the paper states $O(\sqrt{n})$ memory as the *achieved* complexity of their specific checkpointing-every-$k$-th-layer scheme (with one extra forward pass), not as a formally proven "the interval $\sqrt{n}$ is memory-optimal" theorem — cite carefully if adopted.
- MIT 6.5940 "TinyML and Efficient Deep Learning Computing" (Song Han), Lab 2 "Quantization" — the HAN Lab course's closest lab to this section's topic, but it targets post-training/inference weight quantization (int8/int4) for deployment, not training-time mixed precision, checkpointing, or accumulation — low overlap, cited as adjacent background only — https://hanlab.mit.edu/courses/2024-fall-65940
- **No good external exercise tradition found** for a dedicated gradient-accumulation micro-batch sweep (this section's ex5) — every source checked treats gradient accumulation as a one-line implementation note rather than a subject worth its own homework problem.

**Proposed problem set** (6 problems):
1. [short-code] **Largest trainable model in 24 GB.** Using only the memory anatomy, compute the largest (width, depth) trainable under fp32+Adam, bf16+Adam, and bf16+Adam+checkpointing, then verify one point against `max_memory_allocated`.
   *Provenance:* original (existing ex1, kept).
1. [conceptual] **Per-token activation bytes.** Derive the per-token activation bytes of one Transformer block as a function of width and sequence length, and check the formula against a measured peak.
   *Provenance:* original (existing ex2, kept).
1. [short-code] **Checkpoint every √n-th block.** Implement checkpointing at that interval for a deep block stack, measure peak memory and step time, and explain why $\sqrt{n}$ is the memory-optimal interval.
   *Provenance:* original (existing ex3, kept; on adoption, cite Chen et al. 2016 (arXiv:1604.06174) as the source of the $O(\sqrt{n})$ result — overlap high, with the caveat above about how the paper itself frames the result).
1. [short-code] **Breaking and fixing fp16.** Deliberately underflow fp16 gradients by scaling the loss down before backward under `autocast(dtype=torch.float16)` with no `GradScaler`; confirm bf16 survives the same step; add a `GradScaler`, confirm the fix, and explain why real failures are workload-dependent.
   *Provenance:* original (existing ex4, kept; on adoption, cite Micikevicius et al., ICLR 2018 (arXiv:1710.03740) for the scaling scheme, and optionally NVIDIA's mixed-precision guide for the concrete 31%-underflow/scale-of-8 worked number as a real-world anchor — overlap medium).
1. [short-code] **Micro-batching at fixed global batch.** With a fixed global batch, sweep the micro-batch size and plot peak memory and wall-clock per optimizer step; confirm memory tracks the micro-batch while the update is unchanged.
   *Provenance:* original (existing ex5, kept — no external exercise tradition found for this specific sweep; see chapter overview).
1. [conceptual] **Memory accounting for a different optimizer.** Redo ex1's memory-anatomy derivation symbolically for SGD-with-momentum (one extra state per parameter) instead of Adam (two), and predict how the largest-trainable-model answer shifts.
   *Provenance:* adapted from Stanford CS336 Lecture 2 "Resource Accounting" (overlap medium — the accounting method is borrowed directly; the specific optimizer comparison is new).

---

## chapter_computational-performance/multiple-gpus.md — Multi-GPU from First Principles

**Topic:** hand-written data parallelism, the ring-allreduce derivation, and the
data-parallel step-time cost model.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — dense but
concretely scoped; every item names a specific quantity to compute and an
equation to check it against, with no formatting or clarity issues found.

**External sources found:**
- Andrew Gibiansky (Baidu SVAIL), "Bringing HPC Techniques to Deep Learning," 2017 — the widely-cited engineering-blog derivation of the ring-allreduce cost formula: each of $N$ devices sends/receives $N-1$ times in each of the scatter-reduce and allgather phases, transferring $K/N$ values per round, giving total traffic $2(N-1)K/N$ — independent of $N$; this is almost certainly the same derivation this section's :eqref:`eq_ring_traffic` follows — http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
- Li, Zhao, Varma, et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," 2020 (arXiv:2006.15704) — documents gradient bucketing and compute/communication overlap, the production mechanisms this section explicitly defers to multi-gpu-practice.md.
- Rajbhandari, Rasley, Ruwase, He, "ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models," 2019 (arXiv:1910.02054) — background for the sharding direction this section is building toward; the actual sizing exercise belongs to multi-gpu-practice.md.
- **No dedicated university homework found** asking students to derive ring-allreduce from scratch — UW CSE 599W's assignments (checked directly) stop at single-node graph/kernel optimization and never reach a distributed-training assignment; this topic's exercise tradition lives in engineering blogs and systems papers (Gibiansky, PyTorch, ZeRO) rather than course problem sets — a genuine, notable gap for a topic this central to the field.

**Proposed problem set** (6 problems):
1. [short-code] **Scaling the hand-rolled loop.** Extend the from-scratch data-parallel loop to $k=4$ and measure sec/epoch at $k \in \{1,2,4\}$; explain whether the slowdown grows, shrinks, or holds using the cost equation.
   *Provenance:* original (existing ex1, kept).
1. [short-code] **Ring allreduce vs. star.** Implement ring allreduce with explicit device-to-device copies and test whether it beats the star `allreduce` on a slow-fabric machine; explain the (barely) positive result in terms of interconnect topology.
   *Provenance:* original (existing ex2, kept).
1. [short-code] **Predicting communication time.** Compute the ring's per-device traffic for a real model's parameter count in fp32, and use a measured bandwidth to predict communication time per step, ready to compare against a later real measurement.
   *Provenance:* original (existing ex3, kept).
1. [conceptual] **Halving gradient bytes.** A thought experiment: sending gradients in bf16 instead of fp32 halves the bytes moved; identify what could break and which term of the cost equation the halving actually helps.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Scaling batch with device count.** Scale the batch size with $k$ so each device keeps a full per-device batch; identify which cost-equation term now dominates and whether the second GPU still "pays" for itself.
   *Provenance:* original (existing ex5, kept).
1. [conceptual] **Checking the primary derivation.** Read Gibiansky's 2017 ring-allreduce cost derivation and check it term-by-term against this section's :eqref:`eq_ring_traffic`; identify the one bookkeeping difference (total bytes $K$ vs. per-parameter accounting).
   *Provenance:* adapted from Gibiansky, "Bringing HPC Techniques to Deep Learning," 2017 (overlap high — cite on adoption, since this is very likely the book's own underlying source for the derivation).

---

## chapter_computational-performance/multi-gpu-practice.md — Multi-GPU in Practice

**Topic:** production data parallelism (PyTorch DDP vs. JAX sharding
annotations), weak-scaling measurement, tensor parallelism, and ZeRO/FSDP-style
sharding.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — every item
is dense but concretely scoped (specific env vars, specific sweep ranges,
specific numbers to reconcile against equations); the review flagged no
defects and no clarity issues, including the unusually elaborate ex3 with its
explicit safety warning.

**External sources found:**
- Stanford, CS336 "Language Modeling from Scratch," Assignment 2 "Systems" (Spring 2024/2025/2026 offerings) — implements DDP and optimizer-state sharding (ZeRO-style) end to end on a real Transformer and benchmarks the result, directly analogous to this section's ex1/ex2/ex5 pattern of "measure, then explain the mechanism" — https://github.com/stanford-cs336/assignment2-systems
- Rajbhandari, Rasley, Ruwase, He, "ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models," 2019 (arXiv:1910.02054) — origin of the stage-1/2/3 sharding scheme and the per-GPU memory accounting this section's ex5 asks students to redo by hand for a 7B-parameter model.
- Li, Zhao, Varma, et al., "PyTorch Distributed," 2020 (arXiv:2006.15704) — documents DDP's gradient-bucketing mechanism, the subject of this section's ex1, though the paper itself does not give a specific numeric bucket-size recommendation — reference/background rather than a worked answer.
- GPU MODE lecture series, Lecture 17 "GPU Collective Communication (NCCL)" — lecture coverage of the exact mechanism ex2/ex3 measure; inspirational only, not a graded exercise — https://github.com/gpu-mode/lectures
- **No external exercise tradition found** for reproducing an NCCL transport-selection hang (this section's ex3, second half) — a box-specific defaults-vs-workaround pitfall unlikely to appear in any general course; a legitimate, distinctive finding rather than a gap to fill.

**Proposed problem set** (6 problems):
1. [short-code] **Bucket size sweep.** Vary DDP's `bucket_cap_mb` and measure throughput at $k=2$; explain why there is an optimum by identifying what a too-small bucket costs and what a too-large one costs.
   *Provenance:* original (existing ex1, kept; on adoption, cite Li et al. 2020 (arXiv:2006.15704) for the bucketing mechanism's documentation — overlap low, since the paper does not itself supply a bucket-size answer).
1. [short-code] **Communication scaling law.** Extend the `no_sync()` gradient-accumulation measurement to $k=4$; compare the growth in per-step communication time against two candidate scaling laws and decide which fits, drawing a conclusion about how NCCL schedules the transfer on this fabric.
   *Provenance:* original (existing ex2, kept).
1. [short-code] **Reproducing two fabric behaviors.** Extend the bare-collective bandwidth comparison across a range of payload sizes to find each transport's saturation point, then flip a documented NCCL environment variable and observe the run wedge (with an explicit safety warning to be ready to kill the launcher).
   *Provenance:* original (existing ex3, kept — no external tradition found for this box-specific task).
1. [short-code] **Tensor-parallel sharding.** Write the `PartitionSpec` that shards a weight matrix's output features across the device mesh, visualize the resulting sharding, and compare the communication pattern to the batch-sharded (data-parallel) case.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Sizing ZeRO stage 3.** Size ZeRO stage 3 (parameters, gradients, and optimizer states all sharded) for a 7-billion-parameter model on 8 GPUs with 80 GB each; redo the arithmetic for plain DDP (no sharding) and explain the difference.
   *Provenance:* original (existing ex5, kept; on adoption, cite Rajbhandari et al. 2019 (arXiv:1910.02054) for the sharding scheme this exercise sizes — overlap high).
1. [conceptual] **Predicting weak-scaling efficiency.** Given measured weak-scaling efficiency at $k=2$ and $k=4$, predict the efficiency at $k=8$ on the same fabric using the cost equation, stating the scaling assumption the prediction makes.
   *Provenance:* original (existing ex6, kept).

---

## chapter_computational-performance/fast-transformer.md — Case Study: Optimizing a Transformer

**Topic:** end-to-end, incremental optimization of a real GPT-style Transformer
— compilation, mixed precision, larger batches, activation checkpointing, and
data parallelism, applied and measured one technique at a time.
**Current exercises:** 8; disposition: keep 6, rewrite 2, drop 0 — technically
the strongest-specified file alongside memory-precision.md; the two rewrites
fix a defect and a pedagogical issue the prior review flagged (ex8's literal
"§13.6" instead of `:numref:`, and ex5 stating the experimental outcome before
asking the reader to explain it, which pre-empts the discovery every sibling
exercise otherwise preserves) — both are surface fixes, content is unchanged.

**External sources found:**
- Stanford, CS336 "Language Modeling from Scratch," Assignment 2 "Systems" — the closest external analogue in the entire chapter: students profile/benchmark a real Transformer, replace attention with a Triton FlashAttention2 kernel, and layer on DDP and optimizer-state sharding, reporting an ablation-style comparison exactly as this section does — https://github.com/stanford-cs336/assignment2-systems
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022 (arXiv:2205.14135) — origin of the tiled, non-materializing attention kernel this section's ex7 (JAX) fuses in; the paper reports concrete speedups (e.g. 15% end-to-end on BERT-large at sequence length 512, 3x on GPT-2 at 1K) that can anchor a "how big a win should this be" sanity check.
- GPU MODE lecture series, Lecture 12 "Flash Attention" and Lecture 1 "Profiling and Integrating CUDA kernels in PyTorch" — lecture coverage of exactly this section's two central techniques (fused attention, end-to-end profiling); inspirational only, not graded problems — https://github.com/gpu-mode/lectures
- **The "incremental ablation case study" format itself has a thin external tradition** outside CS336 and GPU MODE — most courses teach individual techniques (compilation, precision, parallelism) as separate labs rather than chaining them on one model and reporting cumulative, signed effects; this section's structure is unusually mature relative to what was found elsewhere.

**Proposed problem set** (8 problems — already at the range's upper bound; no additions needed):
1. [short-code] **A negative result at small width.** Reproduce the negative bf16 configuration at the smaller hidden width and confirm bf16 is slower than compile-alone; profile both and explain, via the roofline model, why small matmuls do not benefit from tensor cores.
   *Provenance:* original (existing ex1, kept).
1. [short-code] **Add your own optimization.** Add one configuration of the reader's choosing (`channels_last` memory format, pinned-memory dataloading, or SDPA backend pinning), measure it, and slot it into the cumulative comparison table, stating which regime it attacks.
   *Provenance:* original (existing ex2, kept).
1. [extended] **Port the sequence to a different model.** Apply the whole optimization sequence to a different model (the Mamba capstone or a ViT); identify which configurations change sign and explain why using the arithmetic-intensity argument.
   *Provenance:* original (existing ex3, kept).
1. [short-code] **Construct a reversal.** Take a configuration that helped at this model's scale and construct a different model or configuration where it hurts; explain the reversal using the memory anatomy of memory-precision.md.
   *Provenance:* original (existing ex4, kept).
1. [short-code] **DDP on the optimized model.** Stack data parallelism on top of the full compiled, bf16 sequence at $k=2$ and $k=4$; measure tokens/s and explain, via the multi-GPU cost model, why scaling efficiency comes out higher here than for the eager baseline.
   *Provenance:* original (existing ex5, kept — rephrased so the reader discovers the efficiency gap rather than being told it in advance; content otherwise unchanged).
1. [short-code] **Finding a hidden retrace.** Remove the ragged-batch filter, reproduce a timing measurement once corrupted by an in-window retrace, and find the retrace two ways: in the throughput number and via compiler logging.
   *Provenance:* original (existing ex6, kept).
1. [short-code] **(JAX) Fusing the attention kernel.** Replace the default attention lowering, which materializes the full score matrix, with a fused kernel; verify numerical parity, then measure tokens/s and the compiler's planned temporary-buffer bytes at two context lengths, and explain why context length matters more than batch size for this fix.
   *Provenance:* original (existing ex7, kept; on adoption, cite Dao et al. 2022 (arXiv:2205.14135) as the fused-kernel's source and Stanford CS336 Assignment 2 as the closest hands-on analogue — overlap medium).
1. [short-code] **(JAX) Sharding parameters as well as batch.** Extend Configuration 5's batch-only sharding to also shard parameters across the mesh (the FSDP pattern) via `PartitionSpec`; read the compiled HLO for the inserted collectives, measure tokens/s at $k=2,4$, and explain why this does not pay off at this model's parameter count.
   *Provenance:* original (existing ex8, kept — fix the literal "§13.6" to a proper `:numref:` cross-reference; content otherwise unchanged).

---

## Totals

- Sections covered: 7 (all sections in chapter_computational-performance with a
  `## Exercises` heading).
- Existing exercises reviewed: 42. Disposition: **keep 40, rewrite 4** (2 in
  hardware.md, 2 in fast-transformer.md — all 4 are formatting/phrasing fixes
  with content unchanged), **drop 0**.
- Proposed problem sets: 46 total problems across the 7 sections (6, 8, 6, 6,
  6, 6, 8) — 40 are the kept originals (renamed/tagged per house style), 6 are
  new, each tied to a verified external source.
- Best sources: Stanford CS336 Assignment 2 "Systems" (best overall match —
  relevant to 3 of 7 sections), UC Berkeley CS267 Assignment 1 (best match for
  performance-model.md/hardware.md), UW CSE 599W Assignment 2 (best match for
  compilation.md), Andrew Gibiansky's 2017 Baidu ring-allreduce derivation
  (best match for multiple-gpus.md).
- Sections/sub-topics with no good external exercise tradition: CPU false
  sharing (hardware.md), the NCCL transport-hang reproduction
  (multi-gpu-practice.md), and the gradient-accumulation micro-batch sweep
  (memory-precision.md) — all noted as findings, not failures.
