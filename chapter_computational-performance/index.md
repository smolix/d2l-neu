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
| — | pinned host-to-GPU copy rate | about 23 GB/s | copying a pinned buffer |
| $\beta_\textrm{link}$ | time per byte a GPU sends in an NCCL allreduce, GPUs 0 and 1 | about 0.42 ns per byte (bus bandwidth about 2.4 GB/s), through host memory | a bare 256 MB allreduce |

::: {.d2l-note}
Ridge point, where the two roofs meet: $P/\beta \approx 164$ FLOP per byte
at the rated $\beta$. $\beta$ is a rate, $\beta_\textrm{link}$ a time per
byte, as in the literature on collectives: a message of $S$ bytes costs
$\alpha + S\,\beta_\textrm{link}$. The link between the GPUs moves bytes
about 400 times more slowly than the card's own memory.
:::
:::

::: {.slide title="Five Numbers Describe Our Machine" only="jax"}
[Our machine in five numbers]{.kicker}

| symbol | what it is | value | measured by |
|---|---|---|---|
| $P$ | peak tensor-core rate | 165 TFLOP/s bf16 (83 tf32, rated) | an 8,192-wide bf16 product reaches it |
| $\beta$ | GPU memory bandwidth | 1.0 TB/s rated, about 0.9 measured | elementwise operations moving 512 MiB |
| $h$ | host time per call of a jitted function | about 50–110 µs | tiny products the GPU runs in a few µs |
| — | pinned host-to-GPU copy rate | about 23 GB/s | copying a pinned buffer |
| $\beta_\textrm{link}$ | time per byte a GPU sends in an NCCL allreduce, GPUs 0 and 1 | about 0.42 ns per byte (bus bandwidth about 2.4 GB/s), through host memory | a bare 256 MB allreduce |

::: {.d2l-note}
Ridge point, where the two roofs meet: $P/\beta \approx 164$ FLOP per byte
at the rated $\beta$. $\beta$ is a rate, $\beta_\textrm{link}$ a time per
byte, as in the literature on collectives: a message of $S$ bytes costs
$\alpha + S\,\beta_\textrm{link}$. The link between the GPUs moves bytes
about 400 times more slowly than the card's own memory.
:::
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | GPUs |
|-----------------------------------|---------------------------------------------------------|--------|
| The Performance Model | which resource limits an operation, and how long does it take? | 1 |
| Hardware | where do arithmetic rate and bandwidth come from? | 1 |
| Computation Graphs and Compilation | how does a compiler cut memory traffic and overhead? | 1 |
| Memory and Precision | where does a step's memory go, and how does a model fit? | 1 |
| Multi-GPU from First Principles | when does a second GPU shorten a step? | 2 |
| Data Parallelism in Practice | what do production tools and overlap gain? | 2–4 |
| Sharding the Training State | what does a model that no card holds cost to train? | 2 |
| Case Study: Optimizing a Transformer | where do the milliseconds of a GPT step go? | 1–4 |

**Not here:** hand-written kernels; many machines, serving, and inference
quantization (Language Models part).
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
