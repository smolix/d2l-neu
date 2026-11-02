# Computational Performance
:label:`chap_performance`

An RTX 4090, the GPU that runs this book's notebooks, performs 165
trillion floating-point operations per second on bf16 matrices, but it
reads and writes its memory at only one trillion bytes per second. A
product of two large matrices performs thousands of operations for every
byte it moves, so it can keep the arithmetic units busy. Adding one to
every entry of a bf16 matrix performs one operation for every four bytes,
so it runs at a quarter of a trillion operations per second at most, less
than a six-hundredth of the peak. A training step contains operations of
both kinds, and each of them also costs some microseconds of Python before
the GPU starts it. Written one way, a step keeps the card busy; written
another way, the same model with the same optimizer spends most of its time
waiting.

The earlier chapters ran their models as written and took the time of a
step and its memory as given. That stops working when a run takes days, when a model does not fit on the card, or when a second GPU
makes training slower instead of faster. This chapter teaches a method in
its place: measure a program, find the resource that limits it, apply the
change that relieves that resource, and measure again.

:numref:`sec_perf_model` asks which of three resources limits an
operation (the arithmetic units, the memory bandwidth, or the cost of
issuing it) and how to time GPU code correctly. :numref:`sec_hardware`
asks where a machine's arithmetic rate and bandwidth come from, and why
their ratio tends to grow. :numref:`sec_compilation` asks how a compiler removes memory traffic and
launch overhead that eager execution cannot avoid.
:numref:`sec_memory_precision` asks where the memory of a training step
goes, and how a model of a billion parameters can train on one card.
:numref:`sec_multi_gpu` builds data parallelism by hand and asks when a
second GPU shortens a step.
:numref:`sec_multi_gpu_concise` asks how production libraries hide
communication behind computation and divide a model that no single GPU
holds. :numref:`sec_fast_transformer` asks how far the whole method takes
the training step of a GPT.

The chapter stays within one machine with up to four GPUs, and it uses the
kernels that the frameworks provide instead of writing new ones
(:numref:`sec_custom_layer`). Training across machines, serving engines,
and quantization for inference belong to the Language Models part.

```toc
:maxdepth: 2

performance-model
hardware
compilation
memory-precision
multiple-gpus
multi-gpu-practice
fast-transformer
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online. Related topics that
the chapter does not teach, such as other accelerators, kernels written by
hand, compressed optimizer state, and collective algorithms beyond the
ring, are collected in :numref:`sec_performance-further-reading`.

**Books and long-form texts**

- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/) — rooflines, sharding notation, and the cost of collectives worked out for TPUs and GPUs in JAX; the arithmetic of :numref:`sec_perf_model` and the cost model of :numref:`sec_multi_gpu`, carried on to clusters of thousands of chips.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — the memory budget of a training step, data parallelism, and ZeRO sharding, each with measurements, from :numref:`sec_memory_precision` through :numref:`sec_multi_gpu_concise`, and on to parallelism across many machines.
- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html) — the division of a program's time into compute, memory bandwidth, and overhead that :numref:`sec_perf_model` measures, in a short essay by one of the authors of `torch.compile`.

**Courses and lectures**

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — assignments that profile a transformer, apply compilation and mixed precision, and train it on several GPUs, the work of :numref:`sec_perf_model` through :numref:`sec_fast_transformer` as graded problems; lectures on YouTube.
- [CMU 15-442/15-642: Machine Learning Systems](https://mlsyscourse.org/) — a full course on the hardware, compilers, and parallelism of :numref:`sec_hardware` through :numref:`sec_multi_gpu_concise`, with lecture notes and assignments.
- [GPU MODE lecture series](https://www.youtube.com/@GPUMODE) — CUDA, Triton, and the internals of fused attention kernels, for readers who want to write the kernels that :numref:`sec_compilation` only generates or calls.

**Foundational papers**

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures — Williams, Waterman & Patterson (2009)](https://doi.org/10.1145/1498765.1498785) — the original roofline and its ridge point, the model of :numref:`sec_perf_model`.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation — Ansel et al. (2024)](https://doi.org/10.1145/3620665.3640366) — how TorchDynamo captures a graph from Python bytecode and Inductor compiles it, the PyTorch side of :numref:`sec_compilation`.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — fp32 master weights and loss scaling, the recipe of :numref:`subsec_mp-mixed`, and the reason fp16 needs a scaler where bf16 does not.
- [Training Deep Nets with Sublinear Memory Cost — Chen et al. (2016)](https://arxiv.org/abs/1604.06174) — recomputing activations in the backward pass instead of storing them, the trade of :numref:`subsec_mp-checkpointing`, with a placement of the stored layers that makes memory grow as the square root of the depth.
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training — Li et al. (2020)](https://arxiv.org/abs/2006.15704) — gradient buckets and the overlap of the allreduce with the backward pass, the design of DDP in :numref:`sec_multi_gpu_concise`.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — Rajbhandari et al. (2020)](https://arxiv.org/abs/1910.02054) — the three stages of sharding optimizer state, gradients, and weights of :numref:`subsec_mgp-fsdp`.

**Tutorials and notes**

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [Automatic Mixed Precision recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — a checklist of settings that :numref:`sec_hardware` and :numref:`sec_memory_precision` explain, from pinned memory to autocast.
- [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), [JAX: just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html), and [JAX: sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html) — the framework documentation behind :numref:`sec_compilation` and :numref:`sec_multi_gpu_concise`, including graph breaks, recompilation, and sharded arrays.
- [Understanding GPU Memory — PyTorch blog (2023)](https://pytorch.org/blog/understanding-gpu-memory-1/) — recording and reading memory snapshots with `torch.cuda.memory._record_memory_history`, the tool of :numref:`subsec_mp-measuring`.
- [How to Train Really Large Models on Many GPUs? — Lilian Weng (2021)](https://lilianweng.github.io/posts/2021-09-25-train-large/) — a survey of data, pipeline, and tensor parallelism and of ZeRO, the background of :numref:`sec_multi_gpu` and :numref:`sec_multi_gpu_concise`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13]{.kicker}

Where the time and memory of training go<br>
**performance model · hardware · compilation · memory and precision · multiple GPUs · a case study**
:::
:::

::: {.slide title="One Card Runs Some Operations at Its Peak and Others at a Six-Hundredth"}
[The problem]{.kicker}

- RTX 4090: 165 TFLOP/s in bf16, but only 1 TB/s to its memory.
- A large matrix product: thousands of FLOPs per byte, so it can run
  near the peak.
- Adding 1 to a bf16 matrix: one FLOP per 4 bytes, at most
  0.25 TFLOP/s.
- Every operation also costs microseconds of Python before it starts.
- A 1B-parameter GPT with Adam in fp32 needs 15 GiB of the card's 24 GiB
  before it stores a single activation.

::: {.d2l-note}
Which resource limits a training step, and which change relieves it, on
one GPU and on several?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- place an operation on the roofline and time GPU code correctly;
- explain where arithmetic rate, bandwidth, and capacity come from;
- fuse and replay kernels with `torch.compile` or `jax.jit`;
- fit a model with bf16, checkpointing, and gradient accumulation;
- build data parallelism by hand and predict its speedup;
- overlap communication and shard the training state across GPUs;
- speed up the training step of a GPT one measured change at a time.

**Running example:** one RTX 4090 and a GPT of a billion parameters, then
the four RTX 4090s of our build machine.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| The Performance Model | which resource limits an operation, and how do we time it? |
| Hardware | where do arithmetic rate and bandwidth come from? |
| Compute Graphs and Compilation | how does a compiler remove memory traffic and overhead? |
| Memory and Precision | where does the memory of a step go, and how does a model fit? |
| Multi-GPU from First Principles | when does a second GPU shorten a step? |
| Multi-GPU in Practice | how do libraries hide communication and shard a model? |
| Case Study: Optimizing a Transformer | how far does the method take a GPT? |
:::

::: {.slide title="Every Section Measures Its Claims on Our Own Machine"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| The Performance Model | matrix products of size 256 to 8,192; elementwise operations; an MLP step |
| Hardware | decoding with a 1B-parameter GPT; copies between host and GPUs |
| Compute Graphs and Compilation | an eight-operation activation; the MLP of the 1B GPT; an MLP step |
| Memory and Precision | the 1B GPT on one 24 GiB card |
| Multi-GPU from First Principles | LeNet on Fashion-MNIST, on two GPUs |
| Multi-GPU in Practice | ResNet-18 on up to four GPUs; a 1.9B-parameter GPT for sharding |
| Case Study: Optimizing a Transformer | the GPT of the Transformers chapter at width 512, on one to four GPUs |
:::
