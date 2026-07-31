# Computational Performance
:label:`chap_performance`

The same model and optimization method can differ substantially in
wall-clock time and memory use across implementations. The difference
depends on how well the program uses arithmetic units, limits data movement,
amortizes framework overhead, and allocates device memory. This chapter
develops a quantitative method for analyzing these effects.

The roofline model compares arithmetic throughput with memory bandwidth.
Arithmetic intensity, measured in FLOPs per byte moved, determines which
resource bounds an operation. A third regime appears when dispatch overhead
dominates both. The practical procedure is to measure a program, classify
its limiting regime, apply a corresponding optimization, and measure again.
:numref:`sec_perf_model` introduces this procedure and the remaining
sections apply it.

The sections proceed from individual operations to full models on multiple
GPUs. :numref:`sec_perf_model` establishes the roofline, the three
regimes, and the measurement discipline the frameworks' asynchronous
dispatch makes non-optional. :numref:`sec_hardware` explains where the
roofline's two numbers come from — the memory hierarchy, the tensor cores
and their numerical formats, interconnects, and energy costs
underneath them all — using our own four-GPU box as the worked example.
:numref:`sec_compilation` targets the bandwidth and overhead regimes:
capturing the compute graph can reduce both, and the section contrasts
`torch.compile`'s bytecode capture with `jax.jit`'s tracing.
:numref:`sec_memory_precision` turns to memory use: the allocation of a
training step, mixed precision, activation checkpointing, and gradient
accumulation — the techniques that decide whether a model fits.
:numref:`sec_multi_gpu` builds data parallelism from scratch, derives the
ring allreduce, and measures communicated bytes and latency;
:numref:`sec_multi_gpu_concise` replaces the hand-rolled version with
production data parallelism and contrasts PyTorch's explicit collectives
with JAX's declarative sharding. Finally :numref:`sec_fast_transformer`
runs the complete procedure on a GPT model. The case study uses each
framework in its usual idiom —
`torch.compile`, autocast, and DDP on the PyTorch side; `jax.jit`,
explicit bf16 threading, and declarative sharding on the JAX side — and
the two waterfalls diverge exactly where the frameworks do: what a
compiled execution changes, and how data parallelism is launched.

A word on the machine this chapter is built on, because it shapes what
you will see. The book's build box is four consumer RTX 4090 GPUs with no
NVLink and — a deliberate market segmentation — no peer-to-peer transfer:
every byte between two GPUs is staged through host memory over PCIe — tens
of gigabytes per second at best, roughly two orders of magnitude below a
datacenter NVLink fabric. This is not a handicap to apologize for; it is a
teaching instrument. Most readers'
multi-GPU machines look like ours, not like a datacenter rack with a
terabyte-per-second fabric, and a slow interconnect makes the *accounting*
of parallel training impossible to ignore. The constants in this chapter
are ours; the reasoning transfers to any machine, and the sections are
written so that a reader on two GPUs, or one, sees the same story.

## What This Chapter Is Not {.unnumbered}

This chapter has a limited systems scope.
*Multi-node* training is the province of the Language Models part, which
has data large enough to warrant it: splitting a model across machines
with tensor, pipeline, or expert parallelism, and the network fabrics
that make it possible, all belong there. :numref:`sec_multi_gpu_concise`
builds the bridge and stops at the water's edge.
*Kernel authoring* in CUDA, Triton,
or Pallas is fenced off book-wide (:numref:`sec_custom_layer`); we teach
how to get performance from the operations you already have, and point to
the resources below for those who want to write their own.
*Serving engines* belong to the Language Models part as well: continuous
batching, paged key–value caches, speculative decoding, all the systems
that turn a trained model into a low-latency service. This chapter
teaches the inference *economics* (the prefill-versus-decode roofline
reading of :numref:`sec_hardware`) but not the engines that exploit them.
*Quantization as compression* for inference is likewise deferred;
:numref:`sec_hardware` teaches formats as *training* precisions only. The
production library map and hardware-selection guidance both live in the Tools appendix
(:numref:`sec_training_systems`, :numref:`sec_hardware_buyers`): which
distributed framework to reach for at which scale, how to checkpoint a
long run, how to launch across a cluster. This chapter earns the concepts
at notebook scale, and the appendix names the products at datacenter
scale.

## Resources and Further Reading {.unnumbered}

The references below follow the chapter's arc — the roofline and the
regimes, the compiler and the memory, the collectives and the case study.
All are freely accessible online. Where this chapter fences off a topic —
kernel authoring, multi-node parallelism, serving — the resource that owns
it is flagged as such.

**Books and long-form**

- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/) — free; the roofline-to-collectives-to-sharding companion to this chapter written in the same spirit, working the arithmetic-intensity accounting of :numref:`sec_perf_model` and the collective-communication cost model of :numref:`sec_multi_gpu` all the way up to datacenter scale.
- [The Ultra-Scale Playbook — Hugging Face](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — free; it develops the memory budgeting of :numref:`sec_memory_precision`, three-dimensional parallelism, and the ZeRO/FSDP sharding analysis referenced by :numref:`sec_multi_gpu_concise`.
- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html) — free; the compute-bound / bandwidth-bound / overhead-bound taxonomy of :numref:`sec_perf_model` in its original form, from one of the authors of `torch.compile`.

**Courses and video lectures**

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — free; the systems assignments — profiling, a `torch.compile` and mixed-precision pass, a multi-GPU training run — are the graded version of :numref:`sec_perf_model` through :numref:`sec_fast_transformer`; lectures on YouTube.
- [CMU 15-442 / 15-642: Machine Learning Systems](https://mlsyscourse.org/) — free; a full ML-systems course covering the hardware, compilation, and parallelism this chapter compresses into seven sections, with lecture notes and assignments.
- [GPU MODE lecture series](https://www.youtube.com/@GPUMODE) — free; the kernel-authoring path this book fences off at :numref:`sec_custom_layer` — CUDA, Triton, and FlashAttention internals — taught from the ground up for readers who want to write the kernels :numref:`sec_compilation` only calls.

**Foundational and current papers**

- [Roofline: An Insightful Visual Performance Model — Williams, Waterman & Patterson (2009)](https://doi.org/10.1145/1498765.1498785) — free; the original roofline paper, the source of the map in :numref:`fig_roofline` and the ridge-point reasoning the whole chapter hangs on.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation — Ansel et al. (2024)](https://doi.org/10.1145/3620665.3640366) — free; how TorchDynamo captures a graph from Python bytecode and Inductor compiles it, the mechanism behind :numref:`sec_compilation`'s `torch.compile` tab.
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training — Li et al. (2020)](https://arxiv.org/abs/2006.15704) — free; the gradient bucketing and computation–communication overlap that make DDP faster than the hand-rolled allreduce of :numref:`sec_multi_gpu`, measured in :numref:`sec_multi_gpu_concise`.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — Rajbhandari et al. (2020)](https://arxiv.org/abs/1910.02054) — free; it derives the staged sharding of optimizer states, gradients, and parameters used by FSDP in :numref:`sec_multi_gpu_concise`.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — free; the master-weights-and-loss-scaling recipe of :numref:`sec_memory_precision`, and the reason bf16 needs no scaler where fp16 does.
- [Training Deep Nets with Sublinear Memory Cost — Chen et al. (2016)](https://arxiv.org/abs/1604.06174) — free; the recompute-in-backward trade of :numref:`sec_memory_precision`'s activation checkpointing, and the $\sqrt{n}$-checkpointing exercise.

**Tutorials, notes, and lore**

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [Automatic Mixed Precision recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — free; the checklist behind :numref:`sec_memory_precision`'s mixed-precision experiment and the pinned-memory and `channels_last` wins the exercises pursue.
- [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), [JAX: jit compilation](https://docs.jax.dev/en/latest/jit-compilation.html), and [JAX: sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html) — free; the framework-side spine of :numref:`sec_compilation` and :numref:`sec_multi_gpu_concise`, including the graph-break and recompilation footguns those sections warn about.
- [Understanding GPU Memory — PyTorch blog](https://pytorch.org/blog/understanding-gpu-memory-1/) — free; the `_record_memory_history` snapshot path :numref:`sec_memory_precision` reads to draw the memory-anatomy sawtooth.
- [How to Train Really Large Models on Many GPUs? — Lilian Weng (2021)](https://lilianweng.github.io/posts/2021-09-25-train-large/) — free; the ZeRO and parallelism taxonomy of :numref:`sec_multi_gpu_concise` at blog altitude, a gentle on-ramp to the Ultra-Scale Playbook.
- [modded-nanogpt — Keller Jordan et al.](https://github.com/KellerJordan/modded-nanogpt) — free; the GPT speedrun that closes :numref:`sec_fast_transformer`: every record stacks techniques from this chapter (compiled block-sparse attention, a better optimizer, fp8) and documents the wall-clock win, the evidence culture the chapter's waterfall imitates.

```toc
:maxdepth: 2

performance-model
hardware
compilation
memory-precision
multiple-gpus
multi-gpu-practice
fast-transformer
```
