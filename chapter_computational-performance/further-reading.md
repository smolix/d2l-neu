# Further Reading
:label:`sec_performance-further-reading`

The sections of this chapter teach one representative of each idea: one
roofline with one memory bandwidth, one GPU and its links, one compiler per
framework, three ways to save memory on one card, one collective algorithm,
one way to divide a model's state among GPUs, and one transformer made fast
on one machine. A real training run brings the rest of the field one
problem at a time, and the problems are not equally urgent. Four of the
entries below, in the order of the chapter, will matter in a first real
run: "The Input Pipeline" (for anyone who trains on images or audio),
"Fragmentation and Preallocation", "Compile Time at Scale", and
"Utilization and Time to a Target Loss". The finer performance models and
the collective algorithms are for readers who will write kernels or run
clusters. Theano is history, and parameter servers mostly so. Function and
flag names are those of late 2026; the ideas will outlast the names.
Implementations, tutorials, and courses for what the chapter does teach are
listed in its introduction (:numref:`chap_performance`). Nothing else in
the book depends on this section.

## Measuring and Modeling

### Finer Performance Models

The roofline of :numref:`sec_perf_model` errs in both directions. Some
kernels run above its bandwidth roof, because tensors that stay in the
72 MB L2 cache, or data reused from shared memory, arrive faster than
$\beta$.
The cache-aware roofline :cite:`Ilic.Pratas.Sousa.2014` draws one bandwidth
roof per level of the memory hierarchy; the hierarchical roofline also
gives a kernel one intensity per level, and the lowest bound limits it.
:citet:`Yang.Wang.Farrell.ea.2020` read these per-level bytes from hardware
counters for deep learning kernels. Other kernels stay below the roof
because they keep too few bytes in flight (Little's law,
:numref:`subsec_hw-gpu`). :citet:`Volkov.2016` shows that independent
instructions within one thread can stand in for extra warps, and that
earlier GPU performance models mispredict throughput by up to 1.7 times,
mostly by misapplying Little's law. For a first diagnosis the plain
roofline is enough; the refinements matter once a kernel sits near a roof
and the question is which one.

### Device Timers and System Profilers

The `Benchmark` of :numref:`sec_perf_model` drains the queue before it
reads the clock, which would slow the training loop it measures. CUDA
events (`torch.cuda.Event`) take timestamps on the GPU's stream without
draining it, but their interval includes the gaps in which the GPU waited
for the host: they time an overhead-bound step without diagnosing it. The
profiler of :numref:`subsec_perf-profiler` separates busy from idle time
for one process, and its traces open as a timeline in a viewer such as
Perfetto. NVIDIA's Nsight Systems adds what lies outside the framework,
putting the operating system's threads, the copies, and the NCCL calls of
all processes on one timeline; Nsight Compute measures a single kernel
against both roofs.

### The Input Pipeline

Readers who train on images or audio often meet a starved loader before
they meet Python overhead: the GPU waits while the CPU decodes and augments
the next minibatch. The case study never meets it, because its corpus sits
on the GPU. Once :numref:`subsec_hw-cpu` has given the number of worker
processes the GPU needs, the remaining question is what to do when the host
has fewer free cores than that; the usual answer is to decode on the GPU,
as NVIDIA's DALI library does. :citet:`Mohan.Phanishayee.Raniwala.ea.2021`
measure such data stalls and remove much of the stall time with a cache
built for the access pattern of training and by sharing preprocessing among
the jobs of a hyperparameter search.

### Utilization and Time to a Target Loss

The account that ends the case study (:numref:`subsec_ft-account`) was made
for BERT by :citet:`Ivanov.Dryden.Ben-Nun.ea.2021`, who went on to fuse the
bandwidth-bound kernels between the products and made an encoder layer 1.3
times faster. The case study's `MFU` column is the number that reports of
large runs quote :cite:`chowdhery2022palm`. *Hardware FLOPs utilization*
also counts recomputed FLOPs, so under activation checkpointing it exceeds
MFU, which checkpointing lowers :cite:`Korthikanti.Casper.Lym.ea.2022`.
Only MFU compares systems fairly, because it credits only the work that the
model needs.

Throughput in tokens per second rewards any change that makes a step
faster, even one that makes training need more steps. In the case study the
larger minibatch raised tokens per second and still ended at a higher loss,
after more tokens, than the GPT of :numref:`sec_gpt`
(:numref:`subsec_ft-waterfall`). The modded-nanoGPT speedrun
:cite:`Jordan.2024` therefore ranks changes by the wall-clock time to train
a small GPT to a fixed validation loss on eight GPUs, a number that a
faster step improves only if it does not cost more steps than it saves. Its
records at the time of writing pair this chapter's kernels and formats with
the Muon optimizer (:numref:`sec_muon`).

## Hardware and Number Formats

### Other Accelerators

The chapter applies unchanged to accelerators that :numref:`tab_gpu_specs`
does not list. AMD's Instinct MI355X, Google's TPU7x, Intel's Gaudi 3, and
AWS's Trainium2 all place stacked memory beside a die of matrix units.
Their ridge points (published dense bf16 rate over memory bandwidth) range
from about 230 FLOP/byte (Trainium2) to 450 or more (Gaudi 3), 1.4 to 3
times the 164 of our RTX 4090. The regimes of :numref:`sec_perf_model`
carry over, and more of a step falls under the bandwidth roof, so fusion
and narrow formats matter more on these chips, not less. What differs is
the software, the formats each chip runs fast, and the price, which
:numref:`sec_hardware_buyers` weighs.

### Unified Memory

A single stream of text is decoded no faster than the memory bandwidth
divided by the bytes of the weights (:numref:`subsec_hw-two-workloads`),
and only if the weights fit at all. Apple's M3 Ultra shares up to 512 GB of
memory between its CPU and GPU at 819 GB/s. A 70-billion-parameter model in
8 bits, about 70 GB, does not fit on an RTX 4090, while on the M3 Ultra the
bound allows about a dozen tokens per second. Machines of this kind trade a
low compute roof for capacity, which suits decoding a single stream of a
large model and suits training poorly.

### Block-Scaled Formats in Detail

The block scales of :numref:`subsec_hw-gpu` cost bits. In the OCP microscaling
formats :cite:`Rouhani.Zhao.More.ea.2023` an 8-bit power-of-two scale shared by
32 values adds a quarter bit per value. NVIDIA's NVFP4 shares an fp8 scale
among 16 values, half a bit per value, plus one fp32 scale per tensor. For the
extra quarter bit its scales need not be powers of two, so they can bring each
block's largest value close to 6, the top of the fp4 range, where a
power-of-two scale can only bring it within a factor of two. Either way a 4-bit
number costs at most 4.5 bits, under a third of the 16 bits of bf16.

## Compilers and Kernels

### Writing Fused Kernels

The compilers of :numref:`sec_compilation` fuse chains of elementwise
operations and reductions, but they do not find algorithmic rewrites such
as FlashAttention. Triton :cite:`Tillet.Kung.Cox.2019` makes a block of a
tensor the unit of programming: the author decides the tiling and which
values stay on the chip, and the compiler handles threads, vector
instructions, and shared memory. Pallas is JAX's language of the same kind.
Most readers will never write a kernel, and those who do usually start by
reading one: Inductor's Triton output for a familiar model. The versions of
FlashAttention show a kernel following the hardware: FlashAttention-2
:cite:`Dao.2024` divides the work among thread blocks by query tiles, and
FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024` overlaps data
movement with the products and exponentials on Hopper GPUs and adds fp8.

### Compile Time at Scale

Compiling the small step of :numref:`subsec_comp-wholestep` takes seconds;
compiling a large model takes minutes, again in every process and for every
new shape. Regional compilation compiles one repeated block, such as a
transformer layer, once for all its copies
(`torch.compiler.nested_compile_region`), and caches keep compiled programs
across processes (`torch.compiler.save_cache_artifacts`,
`jax_compilation_cache_dir`). Autotuning spends compile time instead of
saving it: it times several tilings of each matrix product and keeps the
fastest (`mode='max-autotune'`; XLA autotunes as it compiles). It lengthens
the first run for a gain in every step, a trade that the break-even count
$n^*$ of that subsection weighs.

### Kernel Replay in Training

A recorded CUDA graph fixes its shapes and addresses
(:numref:`subsec_comp-overhead`), and a training loop strains both. Shapes
change, as with the short last minibatch of an epoch; the CUDA graph trees
behind `mode='reduce-overhead'` record the forward and backward passes once
per shape, all in one shared memory pool. Tensors kept across steps, such
as a logged loss, are overwritten by the next replay unless they are cloned
first. The trees report an error when an overwritten output is read, and
where they cannot tell that a new step has begun, the loop marks it
(`torch.compiler.cudagraph_mark_step_begin`) so that the previous step's
memory can be reused.

### Exporting a Captured Graph

A captured graph can also leave Python. `torch.export` captures a whole
program, failing with an error where `torch.compile` would insert a graph
break, and replaces the older TorchScript; a jitted JAX function lowers to
StableHLO, a versioned format that other compilers accept. Exported graphs
are how models reach phones, browsers, and serving systems.

### Earlier Graph-Building Designs

Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` and TensorFlow 1
:cite:`Abadi.Barham.Chen.ea.2016` made the user build a symbolic graph and
then run it, which gave the compiler the whole program and made the program
hard to write and to debug. Eager frameworks reversed the choice, and
tracing compilers recover the graph from eager code;
:citet:`Frostig.Johnson.Leary.2018` describe the tracing into XLA from
which `jax.jit` grew.

## Memory

### Smaller Optimizer State

Adam's two moment estimates take 8 of the 16 bytes per parameter of
:numref:`subsec_mp-anatomy`. Rather than divide them among GPUs
(:numref:`sec_sharding`), two methods shrink them. Eight-bit Adam
:cite:`Dettmers.Lewis.Shleifer.ea.2022` quantizes both block by block and
reaches the accuracy of 32-bit Adam with 2 bytes of state per parameter
instead of 8. Adafactor :cite:`Shazeer.Stern.2018` by default keeps no
first moment, and of each weight matrix's second moment only the row and
column sums, so its state grows with the sum of the dimensions, not their
product. Of the two, 8-bit Adam changes less: it is still Adam, while
Adafactor is another optimizer with its own settings to tune.

### Offloading to Host Memory and Disk

Host memory is an order of magnitude larger than GPU memory and cheaper per
byte. A model of three billion parameters needs 48 GB at the 16 bytes per
parameter of :numref:`subsec_mp-anatomy`, twice our card. ZeRO-Offload
:cite:`Ren.Rajbhandari.Aminabadi.ea.2021` keeps the fp32 weights and both
moments in host memory and runs the update on the CPU, leaving the card
12 GB of 16-bit weights and gradients. Gradients out and weights back then
move 12 GB per step, about half a second at the 23 GB/s of our pinned
copies (:numref:`subsec_hw-cpu`). At six FLOPs per parameter and
token and half the bf16 peak, the card processes about 4,600 tokens per
second, so a minibatch of a few thousand tokens computes long enough to
hide the copy. The CPU's Adam step over all three billion parameters is the
other cost. ZeRO-Infinity :cite:`Rajbhandari.Ruwase.Rasley.ea.2021` adds
NVMe drives to the hierarchy.

### Choosing What to Recompute

Storing the input of every block, as :numref:`subsec_mp-checkpointing`
does, is one choice among many. Binomial checkpointing
:cite:`Griewank.Walther.2008` minimizes the recomputation of a chain for a
given number of stored states, and Checkmate
:cite:`Jain.Jain.Nrusimha.ea.2020` solves a general graph as an integer
linear program. Compilers choose too: AOTAutograd recomputes cheap
elementwise results when that saves memory traffic, and more under a memory
budget (a private setting,
`torch._functorch.config.activation_memory_budget`). In JAX the choice is a
policy passed to `jax.checkpoint`, such as saving the matrix products and
recomputing the rest.

### Fragmentation and Preallocation

A training step can run out of memory with gigabytes free. PyTorch's caching
allocator keeps freed blocks for reuse. When shapes change from step to step,
as with variable sequence lengths, the free memory splits into pieces that no
new tensor fits, and the error reports a large amount "reserved but
unallocated". Letting the allocator grow a segment in place removes most of
this waste (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), as the error
message suggests. Under JAX the first surprise is a card that `nvidia-smi`
shows nearly full whatever the model: a process reserves 75% of the card's
memory at its first operation, so a second JAX process on the same card can
fail (`XLA_PYTHON_CLIENT_PREALLOCATE=false` turns this off,
`XLA_CLIENT_MEM_FRACTION` sets the share).

## Communication and Scale

### Collective Algorithms beyond the Ring

The ring allreduce of :numref:`subsec_mg-ring` moves the fewest bytes but
takes $2(k-1)$ sequential steps, each with a fixed cost, which dominates
for small messages on many devices. Trees need a number of steps that grows
like $\log_2 k$; NCCL combines two binary trees in which every device is a
leaf in one and an inner node in the other, so all links stay busy.
Recursive halving and doubling reach the bandwidth of the ring with the
step count of a tree when $k$ is a power of two
:cite:`Thakur.Rabenseifner.Gropp.2005`. For four GPUs and gradients of tens
of megabytes the ring is the right choice; the others pay off for small
messages or many devices, and libraries pick among them by message size and
topology. NVIDIA's nccl-tests prints the inverse of our
$\beta_\textrm{link}$ (:numref:`sec_multi_gpu`) as *bus bandwidth*, next
to an *algorithm bandwidth*, the message size over the time, which equals
the bus bandwidth at $k = 2$ and falls to about half of it for large $k$.

### Parameter Servers and Asynchronous Updates

Before collectives became the default, large-scale learning summed
gradients on parameter servers: machines that each hold a shard of the
parameters and answer two requests, *push* a gradient and *pull* the
current values :cite:`Li.Andersen.Park.ea.2014`. The interface admits
updates computed from stale parameters, as in DistBelief
:cite:`Dean.Corrado.Monga.ea.2012`. Asynchrony stops waiting for the
slowest worker, and local SGD :cite:`Stich.2019` averages the replicas only
every few steps; both change the optimization, not just its speed.
Synchronous data parallelism therefore remains the default for dense
models. Parameter servers survive where the parameters are too large and
too sparse for every device to hold, as in the embedding tables of
recommender systems (:numref:`chap_recsys`).

### Parallelism Interfaces

PyTorch offers each form of parallelism as its own interface
(`DistributedDataParallel`, `fully_shard` for the sharding of
:numref:`sec_sharding`, `torch.distributed.tensor.parallel`,
`torch.distributed.pipelining`) and combines them on a device mesh. JAX
expresses all of them as placements of arrays on a mesh with named axes,
and the compiler inserts the collectives, a design that GSPMD
:cite:`Xu.Lee.Chen.ea.2021` introduced.

### Beyond One Machine

The Language Models part takes up the topics that need several machines or
serve trained models. Tensor, pipeline, and expert parallelism split a
model whose step is too slow or too communication-heavy under sharded data
parallelism
:cite:`shoeybi2019megatron,Narayanan.Shoeybi.Casper.ea.2021,Lepikhin.Lee.Xu.ea.2021`
(:numref:`chap_lm_parallelism`). Training in fp8 needs scales per tensor or
per block and fp32 accumulation
:cite:`Micikevicius.Stosic.Burgess.ea.2022,Liu.Feng.Xue.ea.2024`
(:numref:`sec_lm_low_precision`). Serving engines batch requests
continuously :cite:`Yu.Jeong.Kim.ea.2022` and page the key–value cache
:cite:`Kwon.Li.Zhuang.ea.2023` (:numref:`chap_lm_serving`). Quantization
compresses trained weights to 4 or 8 bits (:numref:`chap_lm_compression`),
and :eqref:`eq_hw_decode_bound` says why it pays: halving the bytes of the
weights doubles the bound on single-stream decoding, while faster
arithmetic does not move it at all.

[Discussions](https://d2l.discourse.group/t/36854)
