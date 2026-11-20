# Computational Performance
:label:`chap_performance`

Until now we have trained every model by writing the network, starting the
training loop, and waiting. We took the time of a step and its memory as
given. That stops working when a run takes days, a model does not fit on
the card, or a second GPU makes training slower. Consider the GPT of
:numref:`sec_gpt`, made twice as wide to give the GPU more work.
Dividing the arithmetic of one training step by the peak rate of the GPU
gives the shortest time the step could take. The measured step takes
several times longer. Where does the rest of the time go?

Computer architects met this question long before deep learning did. By
the 1990s processors had become so much faster than their memory that many
programs spent much of their time waiting for data. In 2009 Williams,
Waterman, and Patterson drew the consequence as one picture, the
*roofline* :cite:`Williams.Waterman.Patterson.2009`: a program runs no
faster than its arithmetic units allow, and no faster than memory can
deliver the bytes it needs. A GPU driven from Python adds a third limit,
the cost of issuing each operation. Which of the three holds a step back
decides which change speeds it up. This view explains choices that earlier
chapters made, such as large minibatches and 16-bit numbers
(:numref:`sec_numerics`). It also gives a method: count the arithmetic and
the bytes of a step, predict its time, measure it, and explain the
difference. The difference is where the time went.

Everything rests on the first section (:numref:`sec_perf_model`), which
introduces one quantity, the arithmetic an operation performs per byte it
moves, and puts the method to work. Hardware (:numref:`sec_hardware`)
explains where the peak rate and the bandwidth come from; a reader in a
hurry can take its tables. Compilation (:numref:`sec_compilation`) and
fewer bits with recomputation (:numref:`sec_memory_precision`) are the
one-GPU tools most readers will use every day. Three sections then add
GPUs. A second GPU helps only when each card has enough work, and on our
cards, which meet only through host memory, the wires between them matter
as much as the GPUs. Dividing the weights, gradients, and optimizer state
among the GPUs trains a model that no single card holds, but every step
must move them between the cards. The case study
(:numref:`sec_fast_transformer`) returns to the GPT above, which fits on
one card, and accounts for its step in milliseconds.

We assume you can time GPU code and read its peak memory
(:numref:`sec_use_gpu`), know bf16 and tf32 (:numref:`sec_numerics`), and
have met the GPT and its key–value cache (:numref:`sec_gpt`,
:numref:`sec_kv-cache`). Every measurement comes from one host with four
RTX 4090 cards, PyTorch 2.11, and JAX 0.12. On your machine the times will
differ; the counts of FLOPs and bytes will not, and neither will the
method that turns them into predictions. The first four sections need one
GPU and the later ones up to four; reading needs none.

Kernels written by hand are left to
:numref:`sec_performance-further-reading`, and training across machines,
serving engines, and quantization for inference to the Language Models
part.

```toc
:maxdepth: 2

performance-model
hardware
compilation
memory-precision
multiple-gpus
multi-gpu-practice
sharding
fast-transformer
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with, and all are freely accessible online. Read Horace
He's essay before or beside :numref:`sec_perf_model`: it takes about
fifteen minutes and describes the same three limits in another voice. The scaling book of
Austin et al. is the natural second text for the whole chapter. Related
topics that the chapter does not teach, such as other accelerators,
kernels written by hand, compressed optimizer state, and collective
algorithms beyond the ring, are collected in
:numref:`sec_performance-further-reading`.

**Books and long-form texts**

- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/) — rooflines, sharding notation, and the cost of collectives worked out for TPUs and GPUs in JAX; the arithmetic of :numref:`sec_perf_model` and the cost model of :numref:`sec_multi_gpu`, carried on to clusters of thousands of chips.
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — the memory budget of a training step, data parallelism, and ZeRO sharding, each with measurements, from :numref:`sec_memory_precision` through :numref:`sec_sharding`, and on to parallelism across many machines.
- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html) — the division of a program's time into compute, memory bandwidth, and overhead that :numref:`sec_perf_model` measures, in a short essay by a PyTorch compiler developer.

**Courses and lectures**

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) — assignments that profile a transformer, apply compilation and mixed precision, and train it on several GPUs, the work of :numref:`sec_perf_model` through :numref:`sec_fast_transformer` as graded problems; lectures on YouTube.
- [CMU 15-442/15-642: Machine Learning Systems](https://mlsyscourse.org/) — a full course on the hardware, compilers, and parallelism of :numref:`sec_hardware` through :numref:`sec_sharding`, with lecture notes and assignments.
- [GPU MODE lecture series](https://www.youtube.com/@GPUMODE) — CUDA, Triton, and the internals of fused attention kernels, for readers who want to write the kernels that :numref:`sec_compilation` only generates or calls.

**Foundational papers**

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures — Williams, Waterman & Patterson (2009)](https://doi.org/10.1145/1498765.1498785) — the original roofline and its ridge point, the model of :numref:`sec_perf_model`.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation — Ansel et al. (2024)](https://doi.org/10.1145/3620665.3640366) — how TorchDynamo captures a graph from Python bytecode and Inductor compiles it, the PyTorch side of :numref:`sec_compilation`.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — fp32 master weights and loss scaling, the recipe of :numref:`subsec_mp-mixed`, and why the narrow exponent range of fp16 needs a loss scaler, a reason that the eight exponent bits of bf16 remove.
- [Training Deep Nets with Sublinear Memory Cost — Chen et al. (2016)](https://arxiv.org/abs/1604.06174) — recomputing activations in the backward pass instead of storing them, the trade of :numref:`subsec_mp-checkpointing`, with a placement of the stored layers that makes memory grow as the square root of the depth.
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training — Li et al. (2020)](https://arxiv.org/abs/2006.15704) — gradient buckets and the overlap of the allreduce with the backward pass, the design of DDP in :numref:`sec_multi_gpu_concise`.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — Rajbhandari et al. (2020)](https://arxiv.org/abs/1910.02054) — the three stages of sharding optimizer state, gradients, and weights of :numref:`sec_sharding`.

**Tutorials and notes**

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [Automatic Mixed Precision recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — a checklist of settings that :numref:`sec_hardware` and :numref:`sec_memory_precision` explain, from pinned memory to autocast.
- [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), [JAX: just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html), and [JAX: sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html) — the framework documentation behind :numref:`sec_compilation`, :numref:`sec_multi_gpu_concise`, and :numref:`sec_sharding`, including graph breaks, recompilation, and sharded arrays.
- [Understanding GPU Memory — PyTorch blog (2023)](https://pytorch.org/blog/understanding-gpu-memory-1/) — recording and reading snapshots of the allocator's memory over a training step, the tool of :numref:`subsec_mp-measuring`.
- [How to Train Really Large Models on Many GPUs? — Lilian Weng (2021)](https://lilianweng.github.io/posts/2021-09-25-train-large/) — a survey of data, pipeline, and tensor parallelism and of ZeRO, the background of :numref:`sec_multi_gpu` through :numref:`sec_sharding`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13]{.kicker}

Where the time and memory of training go<br>
**performance model · hardware · compilation · memory and precision · multiple GPUs · sharding · a case study**
:::
:::

::: {.slide title="Arithmetic, Memory Traffic, the Host, and Capacity Each Limit Some Steps"}
[The problem]{.kicker}

- **Arithmetic:** an 8,192-wide bf16 product does about 2,700 FLOPs per
  byte and runs at the RTX 4090's peak of about 165 TFLOP/s.
- **Memory traffic:** adding 1 to a bf16 matrix reads 2 bytes and writes 2
  per FLOP; at 1 TB/s that is at most 0.25 TFLOP/s, a six-hundredth of the
  peak.
- **Host:** every operation issued from Python costs microseconds before
  the GPU starts it.
- **Capacity:** a 1B-parameter GPT with Adam in fp32 fills 15 GiB of the
  card's 24 GiB before it stores a single activation.

::: {.d2l-note}
Which limit holds a step back, on one GPU and on several, decides which
change speeds it up.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- predict an operation's time from its FLOPs and bytes, and time GPU code
  correctly;
- speed up one GPU with compilation and bf16, and fit larger minibatches
  with checkpointing and gradient accumulation;
- predict when a second GPU shortens a step, and measure what overlap
  hides;
- train a model that no card holds by dividing its weights, gradients, and
  optimizer state among the GPUs;
- account for a GPT's training step in milliseconds.

**Method:** count FLOPs and bytes, predict the time, measure, explain the
difference. **Running examples:** a 1B-parameter GPT for memory and
sharding, a 19M-parameter GPT for the case study; one RTX 4090, then four
in one host.
:::

::: {.slide title="Five Numbers Describe Our Machine" only="pytorch"}
[Our machine in five numbers]{.kicker}

| symbol | what it is | value | measured by |
|---|---|---|---|
| $P$ | peak tensor-core rate | 165 TFLOP/s bf16 (83 tf32, rated) | an 8,192-wide bf16 product reaches it |
| $\beta$ | GPU memory bandwidth | 1.0 TB/s rated, about 0.9 measured | elementwise operations moving 512 MiB |
| $h$ | host time per operation issued from Python | about 13 µs | tiny products the GPU runs in a few µs |
| $\beta_\textrm{host}$ | pinned host-to-GPU copy | about 24 GB/s | copying a pinned buffer |
| $\beta_\textrm{link}$ | NCCL allreduce rate between GPUs 0 and 1 | about 2.4 GB/s, through host memory | a bare 256 MB allreduce |

::: {.d2l-note}
Ridge point, where the two roofs meet: $P/\beta \approx 164$ FLOP per byte
at the rated $\beta$. The link between the GPUs moves bytes about 400 times
more slowly than the card's own memory.
:::
:::

::: {.slide title="Five Numbers Describe Our Machine" only="jax"}
[Our machine in five numbers]{.kicker}

| symbol | what it is | value | measured by |
|---|---|---|---|
| $P$ | peak tensor-core rate | 165 TFLOP/s bf16 (83 tf32, rated) | an 8,192-wide bf16 product reaches it |
| $\beta$ | GPU memory bandwidth | 1.0 TB/s rated, about 0.9 measured | elementwise operations moving 512 MiB |
| $h$ | host time per call of a jitted function | about 60–100 µs | tiny products the GPU runs in a few µs |
| $\beta_\textrm{host}$ | pinned host-to-GPU copy | about 24 GB/s | copying a pinned buffer |
| $\beta_\textrm{link}$ | NCCL allreduce rate between GPUs 0 and 1 | about 2.4 GB/s, through host memory | a bare 256 MB allreduce |

::: {.d2l-note}
Ridge point, where the two roofs meet: $P/\beta \approx 164$ FLOP per byte
at the rated $\beta$. The link between the GPUs moves bytes about 400 times
more slowly than the card's own memory.
:::
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | GPUs |
|---|---|---|
| The Performance Model | which resource limits an operation, and what time does it predict? | 1 |
| Hardware | where do arithmetic rate and bandwidth come from? | 1 |
| Computation Graphs and Compilation | how does a compiler remove memory traffic and overhead? | 1 |
| Memory and Precision | where does the memory of a step go, and how does a model fit? | 1 |
| Multi-GPU from First Principles | when does a second GPU shorten a step? | 2 |
| Data Parallelism in Practice | how do production tools drive every GPU, and what does overlap hide? | 2–4 |
| Sharding the Training State | what does it cost to train a model that no card holds? | 2 |
| Case Study: Optimizing a Transformer | where do the milliseconds of a GPT step go? | 1–4 |

**Not here:** kernels written by hand; training across machines, serving, and
quantization for inference (Language Models part).
:::

::: {.slide title="A 1B-Parameter GPT Returns in Hardware, Memory, and Sharding"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| The Performance Model | a skinny product across the ridge; products of size 256 to 8,192; elementwise operations; an MLP step |
| Hardware | **the decode bound of a 1B GPT**; sequential and random reads; copies between host and GPU |
| Computation Graphs and Compilation | an eight-operation activation; an MLP step, compiled and replayed |
| Memory and Precision | **the 1B GPT (18 blocks, width 2,048)** on one 24 GiB card |
| Multi-GPU from First Principles | LeNet on Fashion-MNIST, on one and two GPUs |
| Data Parallelism in Practice | ResNet-18 on one, two, and four GPUs |
| Sharding the Training State | **the 1B GPT made twice as deep: 36 blocks, 1.9B parameters**, on two GPUs |
| Case Study: Optimizing a Transformer | a 19M-parameter GPT (width 512), on one to four GPUs |
:::
