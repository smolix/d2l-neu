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

::: {.cols .vc}
::: {.col}
- **Compute ($P$) vs. Bandwidth ($\beta$):** Large $8192^2$ `bf16` GEMMs saturate Tensor Cores near $165\text{ TFLOP/s}$, while elementwise adds ($I = 0.25$) stall at $0.25\text{ TFLOP/s}$.
- **Host Dispatch ($h$):** Launching eager kernels from Python costs $10\text{--}100\,\mu\text{s}$ per op, leaving the GPU idle between microsecond calls.
- **Memory Capacity ($C$):** A 1B GPT with `fp32` Adam consumes $16N \approx 15.1\text{ GiB}$ of a $24\text{ GiB}$ card before storing a single token's activations.
:::

::: {.col .fig .big}
@fig:ch13-perf-four-bottlenecks
:::
:::
:::

::: {.slide title="The Three-Floor Time Model and the Roofline Ridge"}
[Unifying Analytical Model]{.kicker}

::: {.cols .vc}
::: {.col}
$$t \;\gtrsim\; \max\!\left(h,\; \frac{\textrm{FLOPs}}{P},\; \frac{\textrm{bytes}}{\beta}\right)$$

- **Arithmetic Intensity ($I = \textrm{FLOPs}/\textrm{bytes}$):** Counts operations performed per byte moved between off-chip DRAM and on-chip registers/SRAM.
- **The Ridge Point ($I^* = P/\beta$):** Where the bandwidth roof $I\beta$ meets the compute roof $P$ ($I^* \approx 164\text{ FLOP/B}$ in `bf16` on RTX 4090; $\sim 295$ on H100).
- **Multi-GPU Scaling:** Distributing across $k$ GPUs adds collective latency and transfer time $\alpha + \frac{2(k-1)}{k}S\beta_{\textrm{link}}$.
:::

::: {.col .fig}
@fig:ch13-perf-three-floors
:::
:::
:::

::: {.slide title="Five Hardware Constants Characterize Our Benchmark Host"}
[Reference Machine]{.kicker}

::: {.cols .vc}
::: {.col}
- **Compute & Bandwidth ($P, \beta, I^*$):** Four $24\text{ GiB}$ RTX 4090s deliver $P = 165.2\text{ TFLOP/s}$ (`bf16`) and $\beta = 1.01\text{ TB/s}$ DRAM bandwidth ($I^* \approx 164\text{ FLOP/B}$).
- **Host Launch ($h$):** Single-op dispatch costs $\sim 13\,\mu\text{s}$ in PyTorch and $50\text{--}100\,\mu\text{s}$ in un-jitted JAX.
- **Inter-GPU Link ($\beta_{\textrm{link}}$):** Host-staged PCIe 4.0 allreduces run at $\sim 2.4\text{ GB/s}$ bus bandwidth ($0.42\text{ ns/B}$).
:::

::: {.col .fig .big}
@fig:ch13-perf-host-constants
:::
:::
:::

::: {.slide title="Architectural Roadmap of Chapter 13"}
[Chapter Structure]{.kicker}

::: {.cols .vc}
::: {.col}
- **§13.1–13.2 Roofline & Hardware:** Arithmetic intensity, synchronized timing, memory hierarchy, SMs, Tensor Cores, and the $20\text{ ms}$ sweep invariant.
- **§13.3–13.4 Compilers & Memory:** Symbolic tracing, operator fusion ($9\times$), CUDA graphs, $16N$ Adam state, `bf16`, rematerialization, and accumulation.
- **§13.5–13.8 Multi-GPU & Case Study:** Ring allreduce, DDP bucketing, ZeRO/FSDP sharding, and a $6\times$ step-by-step GPT speedup.
:::

::: {.col .fig .big}
@fig:ch13-perf-roadmap
:::
:::
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
