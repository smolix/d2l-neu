# Computational Performance
:label:`chap_performance`

Until now we have trained every model by writing the network, starting the
training loop, and waiting, taking step time and memory as given. That stops
working when a run takes days, a model does not fit on the card, or a second
GPU makes training slower. Consider the GPT of :numref:`sec_gpt`, made twice
as wide to give the GPU more work. Dividing the arithmetic of one training
step by the GPU's peak rate gives the shortest time the step could take, yet
the measured step takes several times longer. Where does the rest of the time
go?

Computer architects met this question long before deep learning did. By the
1990s processors had outpaced their memory so far that many programs spent
most of their time waiting for data. In 2009 Williams, Waterman, and
Patterson summarized the consequence in the *roofline* model
:cite:`Williams.Waterman.Patterson.2009`: a program runs no faster than its
arithmetic units allow, and no faster than memory can deliver the bytes it
needs. A GPU driven from Python adds a third limit, the host cost of issuing
each operation. Which of the three holds a step back decides which change
speeds it up. This view explains choices made in earlier chapters, such as
large minibatches and 16-bit numbers (:numref:`sec_numerics`), and gives a
method: count the arithmetic and the bytes of a step, predict its time,
measure it, and explain the difference.

Everything rests on :numref:`sec_perf_model`, which introduces *arithmetic
intensity*—the FLOPs an operation performs per byte it moves—and puts the
method to work. Hardware (:numref:`sec_hardware`) explains where peak
arithmetic and bandwidth come from. Compilation (:numref:`sec_compilation`)
and the memory techniques of :numref:`sec_memory_precision` are the single-GPU
tools most practitioners use every day. Three sections then add GPUs: data
parallelism from first principles (:numref:`sec_multi_gpu`) and in practice
(:numref:`sec_multi_gpu_concise`), followed by sharding the weights,
gradients, and optimizer state across cards (:numref:`sec_sharding`) so that
models larger than any single GPU can train. Finally, the case study
(:numref:`sec_fast_transformer`) returns to the GPT above on one and four
GPUs and accounts for its step in milliseconds.

We assume you can time GPU code and read its peak memory
(:numref:`sec_use_gpu`), know bf16 and tf32 (:numref:`sec_numerics`), and
have met the GPT and its key–value cache (:numref:`sec_gpt`,
:numref:`sec_kv-cache`). Every measurement comes from one host with four
RTX 4090 cards, PyTorch 2.11, and JAX 0.12. On your machine the times will
differ; the counts of FLOPs and bytes will not, nor will the method that
turns them into predictions.

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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13]{.kicker}

Computational Performance: Roofline Models, Hardware, Compilers, and Multi-GPU Scaling<br>**arithmetic intensity · roofline & dispatch bounds · memory & precision · fusion & CUDA graphs · collectives & FSDP**
:::
:::

::: {.slide title="Four Bottlenecks Govern Training Time and Capacity"}
[The Central Problem]{.kicker}

Dividing a training step's FLOPs by peak GPU throughput rarely predicts wall-clock time because four distinct resources can limit a step:

- **Arithmetic Throughput ($P$):** An $8192 \times 8192$ `bfloat16` matrix product performs $\sim 2{,}700\text{ FLOP/byte}$ and saturates the RTX 4090's Tensor Cores near $165\text{ TFLOP/s}$.
- **Memory Bandwidth ($\beta$):** Adding $1$ to a `bfloat16` matrix reads $2\text{ B}$ and writes $2\text{ B}$ per FLOP ($I = 0.25\text{ FLOP/byte}$), capping throughput at $0.25\text{ TFLOP/s}$ ($1/660$ of peak).
- **Host Dispatch Overhead ($h$):** Launching an eager GPU kernel from Python costs $10\text{--}100\,\mu\text{s}$ on the CPU, leaving the accelerator idle between microsecond kernels.
- **Memory Capacity ($C$):** Training a 1B-parameter GPT with `float32` Adam consumes $16N \approx 15.1\text{ GiB}$ of a $24\text{ GiB}$ card before storing a single token's activations.

::: {.d2l-note}
**The Systems Method:** Count FLOPs and bytes $\to$ predict the binding regime and step time $\to$ measure with synchronized benchmarks and kernel profilers $\to$ explain the gap.
:::
:::

::: {.slide title="The Three-Floor Time Model and the Roofline Ridge"}
[Unifying Analytical Model]{.kicker}

Every dispatched operation or training step is bounded below by the slowest of three physical floors—host launch latency, arithmetic compute, and memory traffic:

$$t \;\gtrsim\; \max\!\left(h,\; \frac{\textrm{FLOPs}}{P},\; \frac{\textrm{bytes}}{\beta}\right), \qquad \textrm{Throughput} \;\le\; \min(P,\; I\beta)$$

- **Arithmetic Intensity ($I = \textrm{FLOPs}/\textrm{bytes}$):** Counts how many floating-point operations are performed per byte transferred between off-chip DRAM and on-chip SRAM/registers.
- **The Ridge Point ($I^* = P/\beta$):** The crossover intensity where the sloped bandwidth roof $I\beta$ meets the flat compute roof $P$ ($I^* \approx 164\text{ FLOP/byte}$ in `bfloat16` on the RTX 4090; $\sim 295$ on H100).
- **Multi-GPU Scaling:** Distributing a batch across $k$ GPUs adds collective communication $\alpha + \frac{2(k-1)}{k} S \beta_{\textrm{link}}$, shortening a step only when compute savings exceed link latency and transfer time.
:::

::: {.slide title="Five Hardware Constants Characterize Our Benchmark Host"}
[Reference Machine]{.kicker}

Throughout Chapter 13, we test every analytical prediction against measurements on a single host equipped with four RTX 4090 GPUs ($24\text{ GiB}$ each):

| Symbol | Hardware Parameter | Value on Our Workstation | How We Measure It |
| :--- | :--- | :--- | :--- |
| $P$ | Peak Tensor Core throughput | $165.2\text{ TFLOP/s}$ (`bf16`), $82.6$ (`tf32`) | Square $8192 \times 8192$ `bfloat16` matrix product |
| $\beta$ | Off-chip GPU DRAM bandwidth | $1.01\text{ TB/s}$ rated ($\sim 0.9\text{ TB/s}$ achieved) | Streaming $512\text{ MiB}$ elementwise kernels (`x + 1`) |
| $I^*$ | Roofline ridge point ($P / \beta$) | $164\text{ FLOP/byte}$ (`bf16`), $82$ (`tf32`) | Batch-size sweep $B \in [1, 1024]$ on a $16384^2$ weight |
| $h$ | Host dispatch latency per call | $\sim 13\,\mu\text{s}$ (PyTorch op), $50\text{--}100\,\mu\text{s}$ (JAX JIT) | Tiny $256 \times 256$ ops where GPU busy time is $\sim 2\,\mu\text{s}$ |
| $\beta_{\textrm{link}}$ | Inter-GPU collective time/byte | $\sim 0.42\text{ ns/B}$ ($\sim 2.4\text{ GB/s}$ bus bandwidth) | Bare $256\text{ MiB}$ NCCL allreduce staged via host PCIe DRAM |
:::

::: {.slide title="Architectural Roadmap of Chapter 13"}
[Chapter Structure]{.kicker}

The chapter progresses from single-kernel analytical bounds to single-GPU compiler and memory optimizations, and finally to multi-GPU scaling:

| Section | Core Systems Question | GPUs | Concrete Running Benchmark |
| :--- | :--- | :---: | :--- |
| **§13.1 Performance Model** | Which resource binds an op, and how do we time it? | 1 | Skinny matmuls across $I^*$; `gelu_ish`; 3-layer MLP |
| **§13.2 Hardware Anatomy** | Where do $P$, $\beta$, sector reads, and link rates come from? | 1 | 1B GPT decode bound; random gather; pinned PCIe copies |
| **§13.3 Compilation** | How do graph capture, fusion, and CUDA graphs work? | 1 | 8-op `gelu_ish` ($9\times$ traffic cut); 60-layer thin stack |
| **§13.4 Memory & Precision** | How many bytes per param and token fit in $24\text{ GiB}$? | 1 | 1B GPT (18 blocks, $d = 2048$): `bf16`, remat, accumulation |
| **§13.5–13.6 Multi-GPU & DDP** | When does data parallelism scale, and what hides comm? | 2–4 | Ring allreduce; LeNet vs. ResNet-18 bucketing & overlap |
| **§13.7–13.8 Sharding & Case Study** | How does FSDP shard $16N$, and how fast can a GPT run? | 1–4 | 1.9B GPT (36 blocks) via FSDP; 19M GPT step-by-step ($6\times$) |
:::

::: {.slide title="How to Study This Chapter"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Always predict before measuring:** Before running any benchmark cell, write down the FLOPs, bytes moved, and predicted floor $\max(h, \textrm{FLOPs}/P, \textrm{bytes}/\beta)$. When a measurement disagrees with your paper estimate, the discrepancy reveals an uncounted memory transfer, hidden host barrier, or SM contention.
:::

::: {.col}
- **Verify numerical equivalence first:** An optimization that silently alters gradients, batch-norm statistics, or causal masking is a bug, not a speedup. Every section pairs timing benchmarks with one-step gradient checks (`allclose`) and end-to-end validation loss gates.
:::
:::

::: {.d2l-note}
**Up next (§13.1 The Performance Model):** Defining arithmetic intensity, constructing the roofline bound, avoiding asynchronous timing traps, and profiling a complete MLP training step.
:::
:::
