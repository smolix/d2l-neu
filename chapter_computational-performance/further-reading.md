# Further Reading
:label:`sec_performance-further-reading`

The sections of this chapter teach one representative of each idea: one
roofline with one memory bandwidth, one GPU and its links, one compiler per
framework, three ways to save memory on one card, one collective algorithm,
one way to divide a model's state among GPUs, and one transformer made fast
on one machine. Two open-access books make natural companions to the whole
chapter: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
(Austin et al., 2025) works out rooflines, sharding notation, and collective
costs for TPUs and GPUs, and [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
(Hugging Face, 2025) surveys memory budgets, data parallelism, ZeRO, and
multi-node parallelism with empirical measurements. For a quick introduction
to the three regimes of :numref:`sec_perf_model`, see Horace He's essay
:cite:`He.2022`.

## Measuring and Modeling

### Finer Performance Models

The roofline of :numref:`sec_perf_model` :cite:`Williams.Waterman.Patterson.2009`
errs in both directions. Some kernels run above its bandwidth roof because
tensors that fit in the 72 MB L2 cache or shared memory arrive faster than
$\beta$. The cache-aware roofline :cite:`Ilic.Pratas.Sousa.2014` draws one
bandwidth roof per level of the memory hierarchy, and
:citet:`Yang.Wang.Farrell.ea.2020` read these per-level bytes from hardware
counters for deep learning kernels. Other kernels stay below the roof because
they keep too few bytes in flight (Little's law, :numref:`subsec_hw-gpu`).
:citet:`Volkov.2016` shows that independent instructions within one thread can
stand in for extra warps, and that earlier GPU performance models mispredict
throughput by up to 1.7 times by misapplying Little's law.

### Device Timers and System Profilers

`d2l.Benchmark` (:numref:`sec_perf_model`) drains the queue before reading the
clock, which would slow a live training loop. CUDA events
(`torch.cuda.Event`) take timestamps on the GPU's stream without draining it,
though their interval includes any gaps in which the GPU waited for the host.
The profiler of :numref:`subsec_perf-profiler` separates busy from idle time
for one process and exports traces for Perfetto. NVIDIA's Nsight Systems
places OS threads, memory copies, and NCCL calls across all processes on one
timeline, while Nsight Compute measures a single kernel against both roofs.

### The Input Pipeline

Readers who train on images or audio often meet a starved data loader before
they meet Python overhead: the GPU waits while the CPU decodes and augments
the next minibatch. When the host has fewer free cores than the GPU needs
(:numref:`subsec_hw-cpu`), decoding can move to the GPU (e.g., via NVIDIA
DALI). :citet:`Mohan.Phanishayee.Raniwala.ea.2021` analyze data stalls across
training jobs and remove much of the stall time with a training-aware cache
and shared preprocessing across hyperparameter sweeps.

### Utilization and Time to a Target Loss

The kernel breakdown that ends the case study (:numref:`subsec_ft-account`)
was carried out for BERT by :citet:`Ivanov.Dryden.Ben-Nun.ea.2021`, who fused
the bandwidth-bound kernels between matrix products to speed up an encoder
layer by $1.3\times$. The case study's `MFU` column is the standard metric
reported for large training runs :cite:`chowdhery2022palm`. *Hardware FLOPs
utilization* (HFU) also counts recomputed FLOPs, so under activation
checkpointing it exceeds MFU :cite:`Korthikanti.Casper.Lym.ea.2022`; only MFU
compares systems fairly because it credits only the work the model requires.

Throughput in tokens per second rewards any change that shortens a step, even
one that requires more steps to converge. The modded-nanoGPT speedrun
:cite:`Jordan.2024` therefore ranks changes by wall-clock time to train a GPT
to a fixed validation loss on eight GPUs, pairing compiled kernels and
low-precision formats with the Muon optimizer (:numref:`sec_muon`).

## Hardware and Number Formats

### Other Accelerators

The roofline analysis applies unchanged to accelerators beyond
:numref:`tab_gpu_specs`. AMD's Instinct MI355X, Google's TPU7x, Intel's
Gaudi 3, and AWS's Trainium2 all pair stacked HBM with matrix units. Their
bf16 ridge points range from about 230 FLOP/byte (Trainium2) to over 450
(Gaudi 3)—$1.4\times$ to $3\times$ the 164 FLOP/byte of our RTX 4090—making
fusion and narrow formats even more important.

### Unified Memory

Single-stream decoding runs no faster than memory bandwidth divided by the
weight bytes (:numref:`subsec_hw-two-workloads`), provided the weights fit.
Apple's M3 Ultra shares up to 512 GB of memory between CPU and GPU at
819 GB/s. A 70B model in 8 bits (~70 GB) does not fit on a 24 GB card, whereas
on the M3 Ultra the bandwidth bound allows roughly a dozen tokens per second.
Such unified-memory machines trade a lower compute roof for high capacity,
suiting single-stream inference rather than training.

### Block-Scaled Formats in Detail

The block scales of :numref:`subsec_hw-gpu` add a small bit overhead. In the
OCP microscaling formats :cite:`Rouhani.Zhao.More.ea.2023`, an 8-bit
power-of-two scale shared by 32 values adds 0.25 bits per value. NVIDIA's
NVFP4 shares an fp8 scale among 16 values (0.5 bits per value) plus one fp32
scale per tensor, allowing each block's largest value to map near 6 (the top
of the fp4 range) rather than only within a factor of two. Either way, a
4-bit weight or activation costs at most 4.5 bits.

## Compilers and Kernels

### Writing Fused Kernels

The compilers of :numref:`sec_compilation` fuse elementwise chains and
reductions (:cite:`Ansel.Yang.He.ea.2024` for PyTorch 2), but they do not
discover tiled algorithmic rewrites such as FlashAttention. Triton
:cite:`Tillet.Kung.Cox.2019` (and JAX's Pallas) makes a tensor tile the unit
of programming: the author specifies the tiling and on-chip buffering while
the compiler manages threads, vector instructions, and shared memory.
FlashAttention-2 :cite:`Dao.2024` parallelizes across query tiles, and
FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024` overlaps data movement
with tensor-core products and exponentials on Hopper GPUs while adding fp8
support.

### Compile Time at Scale

Compiling a small step (:numref:`subsec_comp-wholestep`) takes seconds,
whereas compiling a large model can take minutes per process and shape.
Regional compilation compiles a repeated block (such as a Transformer layer)
once for all its copies (`torch.compiler.nested_compile_region`), and
persistent caches share compiled artifacts across processes
(`torch.compiler.save_cache_artifacts`, `jax_compilation_cache_dir`).
Autotuning (`mode='max-autotune'` in PyTorch; built into XLA) benchmarks
multiple matrix-product tilings at compile time, trading a longer initial
compilation for faster steps (:numref:`subsec_comp-wholestep`).

### Kernel Replay in Training

A recorded CUDA graph fixes shapes and memory addresses
(:numref:`subsec_comp-overhead`). The CUDA graph trees behind
`mode='reduce-overhead'` record forward and backward passes once per shape in
a shared memory pool. Tensors retained across steps (such as a logged loss)
are overwritten on the next replay unless cloned, and when the runtime cannot
detect a step boundary automatically, `torch.compiler.cudagraph_mark_step_begin`
marks it explicitly.

### Exporting and Earlier Graph Designs

A captured graph can also run outside Python: `torch.export` captures a full
graph strictly without graph breaks, and a jitted JAX function lowers to
StableHLO for deployment on mobile, web, and serving runtimes. Earlier
frameworks such as Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` and
TensorFlow 1 :cite:`Abadi.Barham.Chen.ea.2016` required users to construct
static symbolic graphs explicitly; tracing compilers recover the same whole-program
graph from ordinary eager code :cite:`Frostig.Johnson.Leary.2018`.

## Memory

### Smaller Optimizer State

Adam's two moment estimates occupy 8 of the 16 bytes per parameter in
:numref:`subsec_mp-anatomy`. Rather than sharding them across GPUs
(:numref:`sec_sharding`), 8-bit Adam :cite:`Dettmers.Lewis.Shleifer.ea.2022`
quantizes both moments block-by-block to 2 bytes per parameter without losing
accuracy, while Adafactor :cite:`Shazeer.Stern.2018` omits the first moment
and factors the second moment into row and column sums.

### Offloading to Host Memory and Disk

Host DRAM is an order of magnitude larger and cheaper than GPU memory.
ZeRO-Offload :cite:`Ren.Rajbhandari.Aminabadi.ea.2021` keeps fp32 master
weights and Adam moments in host memory and runs the optimizer update on the
CPU, leaving only 16-bit weights and gradients on the GPU; ZeRO-Infinity
:cite:`Rajbhandari.Ruwase.Rasley.ea.2021` extends offloading to NVMe drives.

### Choosing What to Recompute

Storing only block inputs (:numref:`subsec_mp-checkpointing`,
:cite:`Chen.Xu.Zhang.ea.2016`) is one point on a trade-off curve. Binomial
checkpointing :cite:`Griewank.Walther.2008` minimizes recomputation for a
fixed number of stored states in a chain, and Checkmate
:cite:`Jain.Jain.Nrusimha.ea.2020` solves general computation graphs via
integer linear programming. Modern compilers also apply selective
recomputation automatically (AOTAutograd's min-cut partitioner in PyTorch) or
via explicit policies (`jax.checkpoint_policies` in JAX).

### Fragmentation and Preallocation

When tensor shapes vary across steps, PyTorch's caching allocator can split
free memory into non-contiguous fragments ("reserved but unallocated");
setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allows segments to
grow in place. In JAX, the default allocator preallocates 75% of GPU memory on
startup (`XLA_PYTHON_CLIENT_PREALLOCATE=false` disables preallocation, and
`XLA_PYTHON_CLIENT_ALLOCATOR=vmm` uses CUDA virtual memory management).

## Communication and Scale

### Collective Algorithms beyond the Ring

The ring allreduce (:numref:`subsec_mg-ring`) minimizes total bytes moved but
takes $2(k-1)$ sequential hops. Tree algorithms scale their hop count as
$\log_2 k$, and NCCL pairs two complementary binary trees so that every device
is a leaf in one and an internal node in the other. When $k$ is a power of
two, recursive halving and doubling achieves the ring's byte volume in
$\log_2 k$ steps :cite:`Thakur.Rabenseifner.Gropp.2005`.

### Parameter Servers and Asynchronous Updates

Before collective libraries became standard, distributed training aggregated
gradients on parameter servers :cite:`Dean.Corrado.Monga.ea.2012,Li.Andersen.Park.ea.2014`,
supporting asynchronous updates and local SGD :cite:`Stich.2019`. Because
stale gradients alter optimization dynamics, synchronous allreduce/ZeRO
(:cite:`Li.Zhao.Varma.ea.2020,Rajbhandari.Rasley.Ruwase.ea.2020`) is now
standard for dense models, while parameter servers remain common for massive
sparse recommender tables (:numref:`chap_recsys`).

### Parallelism Interfaces and Beyond One Machine

PyTorch exposes each parallelism strategy through a composable API
(`DistributedDataParallel`, `fully_shard`, `torch.distributed.tensor.parallel`,
`torch.distributed.pipelining`) on a device mesh, whereas JAX expresses all of
them declaratively via array placements on a named mesh compiled with GSPMD
/ Shardy :cite:`Xu.Lee.Chen.ea.2021`. Multi-node tensor, pipeline, and expert
parallelism :cite:`shoeybi2019megatron,Narayanan.Shoeybi.Casper.ea.2021,Lepikhin.Lee.Xu.ea.2021`
(:numref:`chap_lm_parallelism`), fp8 training
:cite:`Micikevicius.Stosic.Burgess.ea.2022,Liu.Feng.Xue.ea.2024`
(:numref:`sec_lm_low_precision`), continuous batching and PagedAttention
:cite:`Yu.Jeong.Kim.ea.2022,Kwon.Li.Zhuang.ea.2023`
(:numref:`chap_lm_serving`), and weight quantization
(:numref:`chap_lm_compression`) are developed in the Language Models part.

[Discussions](https://d2l.discourse.group/t/36854)
