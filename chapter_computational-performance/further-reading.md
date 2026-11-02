# Further Reading
:label:`sec_performance-further-reading`

The sections of this chapter teach one representative of each idea: one
roofline with one memory bandwidth, one GPU and its links, one compiler per
framework, three ways to save memory, one collective algorithm, and one
transformer made fast on one machine. This section collects, in the order
of the chapter, topics they had no room for. They range from finer
performance models and other accelerators to hand-written kernels, further
ways to save memory, other collective algorithms, older forms of
distributed training, and the topics beyond one machine that the Language
Models part takes over. Each topic states the problem it
addresses and what it changes relative to the chapter. Implementations,
tutorials, and courses for the material that the chapter does teach are
listed in its introduction (:numref:`chap_performance`). Nothing else in
the book depends on this section.

## Measuring and Modeling

### One Roof per Memory Level

The roofline of :numref:`sec_perf_model` has a single bandwidth $\beta$,
that of GPU memory. Data that a kernel reuses from the L2 cache or from
shared memory arrives faster, so a kernel with a low intensity against
GPU memory can still run above the roof that this intensity predicts. The
cache-aware roofline :cite:`Ilic.Pratas.Sousa.2014` draws one bandwidth
slope per level of the memory hierarchy and counts the bytes each level
delivers. :citet:`Yang.Wang.Farrell.ea.2020` collect the FLOPs and the
bytes per level from hardware counters with Nsight Compute and place the
kernels of a deep learning model against an L1, an L2, and a memory roof at
once. The single number $I$ of the chapter becomes one intensity per level,
and the level whose roof is lowest for its intensity limits the kernel.

### Latency Hiding

The bandwidth of :numref:`subsec_hw-bytes` is reached only when enough
requests are in flight to cover the latency of each one, as the exercise
on Little's law in :numref:`sec_hardware` computes. A GPU keeps them in
flight by switching among many resident warps. :citet:`Volkov.2016` shows
that the number of warps needed follows from the latency and the
throughput of each unit, that independent instructions within one thread
can replace additional warps, and that common GPU performance models
mispredict throughput because they ignore this. The analysis explains why
small kernels and kernels with few threads stay below the bandwidth roof that
their intensity predicts.

### Device Timers and System Profilers

The `Benchmark` of :numref:`sec_perf_model` waits for the device and reads
the host's clock, so it measures what a step costs in wall-clock time.
CUDA events (`torch.cuda.Event(enable_timing=True)`) take timestamps on the
GPU's own stream instead. They time a region of device work without
draining the queue, which suits measurements inside a training loop, but
they miss host time and so cannot detect the overhead-bound regime. The
framework profilers of :numref:`subsec_perf-profiler` attribute device
time to kernels. NVIDIA's Nsight Systems records the whole timeline of a
process, with Python threads, CUDA calls, kernels, copies, and NCCL
collectives on one axis, and shows idle gaps whose cause lies outside the
framework. Nsight Compute measures a single kernel's achieved FLOP/s and
bytes against both roofs.

### The Time of a Transformer Step

The profile of :numref:`sec_fast_transformer` found the matrix products to
be the largest share of a step but not all of it.
:citet:`Ivanov.Dryden.Ben-Nun.ea.2021` made the same measurement on BERT.
The tensor contractions perform over 99% of the FLOPs of a training step
but take only 61% of its runtime, and over a third of the runtime goes to
normalizations and elementwise operations, which are bandwidth-bound.
Fusing these operators and choosing data layouts that suit each kernel
reduced the memory traffic and made a BERT encoder layer 1.3 times faster
than the frameworks of the time. The study applies the bandwidth regime of
:numref:`sec_perf_model` to a whole model and makes the case for the
compilers of :numref:`sec_compilation`.

### Utilization as a Reported Number

The case study ends with the fraction of the peak that its fastest step
reaches. Reports of large training runs quote the same quantity as *model
FLOPs utilization* (MFU) :cite:`chowdhery2022palm`. MFU is the FLOPs that
the model needs per token, times the tokens per second, divided by the
peak of the hardware. *Hardware FLOPs utilization* also counts the
FLOPs of recomputation, so activation checkpointing
(:numref:`subsec_mp-checkpointing`) raises it while it lowers MFU
:cite:`Korthikanti.Casper.Lym.ea.2022`. Only MFU compares systems
fairly, because it credits only the work that the model needs.

### Time to a Target Loss

Throughput in tokens per second rewards any change that makes a step
faster, even one that makes training need more steps. The modded-nanoGPT
speedrun :cite:`Jordan.2024` ranks changes by the wall-clock time to train
a small GPT to a fixed validation loss on one node of eight GPUs. Its
records combine the tools of this chapter (compiled and fused kernels,
fused attention, fp8 products for some layers) with changes to the
optimization, such as the Muon optimizer of :numref:`sec_muon`. Under this
measure a larger minibatch can lose even when it raises throughput, once
it grows past the critical batch size of :numref:`sec_batch_size`.

## Hardware and Number Formats

### Other Accelerators

The table of :numref:`tab_gpu_specs` lists NVIDIA parts only. Accelerators
from other vendors follow the same design of stacked memory beside a die
of matrix units, so the roofline describes them with two different
numbers. In late 2026, AMD's Instinct MI355X has 288 GB of HBM3e at
8 TB/s and runs 2,517 dense bf16 TFLOP/s, a ridge point of about 310
FLOP/byte. Google's TPU7x (Ironwood) has 192 GiB at 7.4 TB/s and 2,307
bf16 TFLOP/s, a ridge point of about the same size. Intel's Gaudi 3 has
128 GB at 3.7 TB/s and runs 1,678 TFLOP/s in bf16 and in fp8 alike. AWS's
Trainium2 has 96 GiB at 2.9 TB/s and 667 bf16 TFLOP/s. The vendors'
datasheets give these values.

The ridge points of these accelerators range from about 230 FLOP/byte
(Trainium2) to about 450 (Gaudi 3), against 281 and 295 for the B200 and
the H100. The rates per format differ more: fp8 runs no faster than bf16
on Gaudi 3 and on TPU v6e, and fp6 runs at the fp4 rate on the MI355X but
at the fp8 rate on NVIDIA's Blackwell GPUs.

### Unified Memory

A single stream of text is decoded at a rate bounded by the memory
bandwidth divided by the bytes of the weights
(:eqref:`eq_hw_decode_bound`), and only if the weights fit at all. Apple's
M3 Ultra shares up to 512 GB of memory between its CPU and GPU at
819 GB/s. A model of 70 billion parameters stored in 8 bits occupies about
70 GB: it does not fit on an RTX 4090, while on the M3 Ultra the bound
allows about a dozen tokens per second. Machines of this kind trade a low
compute roof for capacity, which suits decoding a single stream of a large
model and suits training poorly. The Tools appendix
(:numref:`sec_hardware_buyers`) covers the choice between such machines
and GPUs.

### Block-Scaled Formats in Detail

:numref:`sec_hardware` explains why 4-bit formats need a scale for every
small block of values. The OCP microscaling formats
:cite:`Rouhani.Zhao.More.ea.2023` share one power-of-two scale, stored in
8 bits, among 32 values; the family covers 8-, 6-, and 4-bit floating
point and 8-bit integers, and adds a quarter bit per value. NVIDIA's
NVFP4 shares a scale in fp8 among 16 values and adds one fp32 scale per
tensor. It spends half a bit per value on scales, and its scales need not
be powers of two, so they follow outliers more closely. Which format
runs at which rate is a decision of each chip's designers, as the
accelerators above show.

## Compilers and Kernels

### Writing Fused Kernels

The compilers of :numref:`sec_compilation` fuse chains of elementwise
operations and reductions, but they do not find algorithmic rewrites such
as FlashAttention. Triton :cite:`Tillet.Kung.Cox.2019` makes a block of a
tensor the unit of programming. The author of a kernel decides the tiling
and which values stay on the chip, and the compiler handles the threads,
the vector instructions, and the layout in shared memory. Inductor emits
Triton for the GPU kernels it generates, so its output is a readable
starting point. Pallas plays the same role in JAX, lowering through Triton
or Mosaic GPU on GPUs and through Mosaic on TPUs. The successive versions
of FlashAttention show how a kernel follows the hardware.
FlashAttention-2 :cite:`Dao.2024` divides the work among thread blocks by
query tiles, and FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024`
overlaps data movement with the matrix products and the exponentials on
Hopper GPUs and adds fp8.

### Compile Time at Scale

The break-even point of :numref:`subsec_comp-wholestep` counts the time a
compiler takes. For a large model this time grows to minutes, and it
recurs in every process and for every new shape. Regional compilation
compiles one repeated block, such as a transformer layer
(`torch.compiler.nested_compile_region`), and reuses the result for every
copy. Compilation caches keep compiled programs across processes:
`torch.compiler.save_cache_artifacts` and `load_cache_artifacts` in
PyTorch, and the persistent compilation cache (`jax_compilation_cache_dir`)
in JAX. Autotuning goes the other way and spends more compile time for
faster kernels. `torch.compile(mode='max-autotune')` times several tilings
and kernel templates for each matrix product and keeps the fastest, and
XLA autotunes its GPU matrix products and fusions in the same way.

### Kernel Replay beyond Inference

:numref:`subsec_comp-overhead` records and replays a forward pass. A
training step adds the backward pass, the optimizer update, and buffers
that persist from step to step. The CUDA graph trees behind
`mode='reduce-overhead'` record the forward and backward passes as
separate graphs that share one memory pool, record again for every new
input shape, and overwrite the outputs of the previous replay, so a
training loop marks the start of each step
(`torch.compiler.cudagraph_mark_step_begin()`). XLA records runs of
kernels as CUDA graphs, which it calls command buffers, by default.

### Exporting a Captured Graph

:numref:`sec_compilation` captures a graph only to compile it for the same
Python process. The same graph can be saved and run without Python.
`torch.export` captures a whole program with no graph breaks, failing where
`torch.compile` would break, and replaces the older TorchScript. A jitted
JAX function lowers to StableHLO, a versioned format that other compilers
and runtimes accept. Exported graphs are how models reach phones,
browsers, and serving systems, which the Language Models part takes up.

### Earlier Graph-Building Designs

Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` and TensorFlow 1
:cite:`Abadi.Barham.Chen.ea.2016` made the user build a symbolic graph and
then run it, which gave the compiler the whole program but made the
program hard to write and to debug. Eager frameworks reversed the choice,
and the tracing compilers of :numref:`sec_compilation` recover the graph
from eager code. :citet:`Frostig.Johnson.Leary.2018` describe the
tracing of Python functions into XLA from which `jax.jit` grew.

## Memory

### Smaller Optimizer State

Of the 16 bytes per parameter of :numref:`subsec_mp-anatomy`, Adam's two
moment estimates take 8. Eight-bit Adam
:cite:`Dettmers.Lewis.Shleifer.ea.2022` stores both in 8 bits, quantized
block by block so that one large entry affects only its own block, and
matches the results of the fp32 optimizer with 2 bytes per parameter
instead of 8. Adafactor :cite:`Shazeer.Stern.2018` keeps, for each weight
matrix, only the sums of the second moment over its rows and over its
columns, and reconstructs the moment of each entry from the two. Its state
grows with the sum of a matrix's dimensions instead of their product.

### Offloading to Host Memory and Disk

Host memory is larger than GPU memory by an order of magnitude and
cheaper per byte. ZeRO-Offload :cite:`Ren.Rajbhandari.Aminabadi.ea.2021`
keeps the optimizer state in host memory and runs the update on the CPU,
so the GPU holds the weights and the activations only. The drawback is
traffic over PCIe. In fp32 the gradients travel to the host and the
updated weights return, 8 GB per step for a model of a billion
parameters, which takes a sizable fraction of a second at the 23 GB/s of
:numref:`subsec_hw-cpu`. Large minibatches hide this time behind the
computation. ZeRO-Infinity :cite:`Rajbhandari.Ruwase.Rasley.ea.2021`
extends the hierarchy to NVMe drives. JAX exposes offloading through its
checkpoint policies, which can send named activations to host memory
instead of discarding them.

### Choosing What to Recompute

:numref:`subsec_mp-checkpointing` stores the input of every block and
recomputes the rest. Which values to store under a memory budget is an
optimization problem. For a chain of identical steps, binomial
checkpointing :cite:`Griewank.Walther.2008` minimizes the recomputation for
a given number of stored states. Checkmate
:cite:`Jain.Jain.Nrusimha.ea.2020` solves the problem for a general graph
as an integer linear program, with the measured cost of each operation.
Compilers make the choice themselves: when AOTAutograd splits the joint
graph of the forward and backward pass, it recomputes cheap elementwise
results in the backward pass instead of storing them, and its memory
budget (`torch._functorch.config.activation_memory_budget`) sets how far
it goes.

### Fragmentation

The memory that a training step needs and the memory that it can obtain
differ by fragmentation. PyTorch's caching allocator keeps freed blocks for
reuse, and when the shapes change from step to step, as with variable
sequence lengths, free memory splits into pieces that no new tensor fits.
An out-of-memory error then reports a large amount "reserved but
unallocated". Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
lets the allocator grow a segment in place instead of allocating new ones,
which removes most of this waste. JAX allocates most of the device's
memory on its first use of a GPU (:numref:`sec_use_gpu`) and manages it
in its own pool. XLA plans the buffers of a compiled program ahead of
time, which leaves the allocator fewer and larger requests.

## Communication

### Collective Algorithms beyond the Ring

The ring allreduce of :numref:`subsec_mg-ring` moves the fewest bytes, but
it takes $2(k-1)$ sequential steps, each with a fixed cost. For small
messages on many devices that cost dominates. Trees reduce and broadcast in
about $\log_2 k$ steps, and NCCL combines two binary trees in which every
device is a leaf in one and an inner node in the other, which keeps all
links busy. Recursive halving and doubling, a reduce-scatter and an
all-gather in $\log_2 k$ steps each, reach the bandwidth of the ring with
the step count of a tree when $k$ is a power of two
:cite:`Thakur.Rabenseifner.Gropp.2005`. Collective libraries pick among
these algorithms by message size and topology, and machines with several
levels of links run a ring at each level.

### Bus Bandwidth

Collective benchmarks such as NVIDIA's nccl-tests report two rates. The
*algorithm bandwidth* is the message size divided by the time. The *bus
bandwidth* of an allreduce is the algorithm bandwidth times $2(k-1)/k$,
the rate at which each link of an ideal ring carries data, so it can be
compared across $k$. The $\beta$ of :numref:`sec_multi_gpu` is defined by
the time $2N/\beta$ of a large allreduce of $N$ bytes, so it is twice the
algorithm bandwidth. It equals the bus bandwidth times $k/(k-1)$, twice the
bus bandwidth at $k = 2$.

### Parameter Servers and Asynchronous Updates

Before collectives became the default for dense models, large-scale
learning summed gradients on parameter servers: machines that each hold a
shard of the parameters and answer two requests, *push* a gradient into
the sum and *pull* the current values :cite:`Li.Andersen.Park.ea.2014`.
Sharding the parameters over many servers lets the bandwidth of the
aggregation grow with the cluster, and the interface admits updates
computed from stale parameters. DistBelief
:cite:`Dean.Corrado.Monga.ea.2012` trained with such asynchronous updates.
Asynchrony and the related local SGD, in which each replica takes several
steps between averages, communicate less often instead of faster, and
both change the optimization as well as its speed. Synchronous data
parallelism therefore remains the default for dense models. Parameter
servers survive where the parameters are too large and too sparse for
every device to hold them, as in the embedding tables of recommender
systems (:numref:`chap_recsys`).

### Large Minibatches

Weak scaling (:numref:`sec_multi_gpu`) keeps the minibatch per GPU fixed,
so the global minibatch grows with the number of GPUs, and the learning
rate has to follow. :citet:`Goyal.Dollar.Girshick.ea.2017` trained a
ResNet-50 on ImageNet with minibatches of 8,192 images to the accuracy of
small minibatches by scaling the learning rate in proportion to the
minibatch and warming it up over the first epochs. :numref:`sec_batch_size`
explains why the gain stops at the critical batch size.

### Parallelism Interfaces

PyTorch offers each form of parallelism as its own interface:
`DistributedDataParallel`; `fully_shard` for sharded data parallelism;
`torch.distributed.tensor.parallel` for splitting the matrices of a layer
across devices; and `torch.distributed.pipelining` for splitting the
layers into stages. A device mesh with several dimensions combines them,
for example sharding within a machine and replicating across machines.
JAX expresses all of them as placements of arrays on a mesh with named
axes, and the compiler's partitioner inserts the collectives, a design
that GSPMD :cite:`Xu.Lee.Chen.ea.2021` introduced.

## Beyond One Machine

The Language Models part takes up the topics that need several machines or
serve trained models; each gets one line here. Tensor, pipeline, and
expert parallelism split a model too large for data parallelism
:cite:`shoeybi2019megatron,Narayanan.Shoeybi.Casper.ea.2021,Lepikhin.Lee.Xu.ea.2021`.
Training in fp8 and fp4 needs scales per tensor or per block and fp32
accumulation :cite:`Micikevicius.Stosic.Burgess.ea.2022,Liu.Feng.Xue.ea.2024`.
Serving engines batch requests continuously and page the key–value cache
:cite:`Kwon.Li.Zhuang.ea.2023`. Quantization compresses trained weights to
4 or 8 bits for inference.

[Discussions](https://d2l.discourse.group/)
