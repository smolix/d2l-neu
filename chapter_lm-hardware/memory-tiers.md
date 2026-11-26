# Memory and Storage Tiers
:label:`sec_lm_memory_tiers`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.4 Memory and Storage Tiers · T3 (T1 reduced)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** State can live on a slower and larger tier whenever the link to that tier moves it faster than the GPU consumes it, and the offload condition says which bytes qualify.
**Opening:** The reader has the ladders of `subsec_hw-bytes` for one GPU and pinned host memory from `subsec_hw-cpu`. One concrete example: Qwen3-8B's optimizer state does not fit one 40 GB card, host memory on the node holds it twenty times over, and host memory is twenty times slower. The open question: which bytes of a workload (weights, optimizer state, activations, KV cache, dataset shards) can sit one tier down without stalling the GPU, and which cannot.
**Teaches:**
- Every tier with a measured bandwidth, latency, capacity and an ordinal cost rank: registers and shared memory, L2, HBM, host DDR5 and the coherent LPDDR5X of Grace-class nodes, CXL-attached memory, local NVMe, parallel file systems, object stores, hard disks; the table is a data file with a source and a date per row, and the plot overlays measured points on the book's ladders.
- What lives where: weights, optimizer state and activations (training); the KV cache and its tiers in serving (HBM, then host memory, then shared SSD-backed stores); datasets and checkpoints (one sentence; 6.9 owns the checkpoint write).
- Offloading as moving state down a tier: optimizer-state offload sends gradients down and updated weights up per step, so it hides behind compute only above a token count per step that depends on the link, not on the model size; the CPU's own Adam pass has the same form; KV reuse as load against recompute.
- Data loading at scale: a node training at a million tokens per second reads two megabytes per second of text, so text pretraining is never storage-bound and the dataset path is about shard size, resumable cursors (4.1) and the per-open latency, not bandwidth; what is storage-bound is preprocessing (Chapter 3), multimodal data and checkpoints; streaming loaders over object stores in one sentence each.
**Derivation:** The offload condition. A step of F FLOPs that needs B bytes from the far side of a link of bandwidth beta_l stalls unless F / P_eff >= B / beta_l, that is unless the FLOPs per byte moved exceed the link's ridge P_eff / beta_l; for optimizer-state offload with F = 6 N T and 2 N bytes per direction the condition is T >= P_eff / (3 beta_l), independent of N (blueprint ch05 5.3 D1). Smallest numeric case: with P_eff = 250 TFLOP/s and a pinned host link of 25 GB/s per direction the step needs at least about 3,300 tokens per GPU to hide the transfer; the CPU's Adam pass at about 30 bytes per parameter over host memory bandwidth adds its own threshold, which the cell prints. Second statement, checked in a cell: a read of s bytes at t(s) = alpha + s / beta achieves a fraction s / (alpha beta + s) of the bandwidth, so shards must exceed 9 alpha beta for 90% (blueprint 5.3 D3).
**Notebook:**
1. Question: do the measured tiers of one node place each byte of Qwen3-8B's training and serving state where the offload condition says it can go, and does a toy offloaded run cross from stalled to overlapped near the predicted token count?
2. Measurements on the 8x A100 node, each under the microbenchmark contract with exact commands and tested parsers: host-to-device and device-to-host sweeps (pinned and pageable, per NUMA node, with `nvbandwidth` and `d2l.hw_h2d_bench`), NVMe sequential and random reads with `fio` at several block sizes and queue depths, GPUDirect Storage against a bounce buffer with `gdsio` (compatibility mode labelled as such), the shared file system and, if an endpoint exists, an object store.
3. Fit alpha and beta per tier with `d2l.fit_alpha_beta` (weighted least squares, weights 1/t^2); print the 90% shard size per tier.
4. The toy offloaded run: a 1B-parameter configuration with optimizer state in pinned host memory, step time against tokens per step; the crossover from stalled to overlapped against the predicted threshold (agreement within a factor of 1.5 is the test).
5. KV load against recompute: for Qwen3-8B with 5.3's prefill model, the crossover prefix length per tier.
6. Reduced T1 configuration: one build-box GPU, the build-box NVMe, pinned and pageable host copies, the Qwen3-0.6B state; `config.tier='T1'`.
7. Claims that survive noise: the fitted tier bandwidths are within 20% of their specifications where a specification exists; the offloaded run crosses within the stated factor; the per-node text read rate is three orders of magnitude below NVMe bandwidth (a computed fact, checked by one measured epoch).
**Figures:** `fig_lm_hw_tiers` (plot: measured bandwidth and latency per tier on the book's ladders); `fig_lm_hw_offload_condition` (plot: step time against tokens per step, stalled and overlapped regimes, predicted threshold); `fig_lm_hw_state_placement` (where each byte of training and serving state lives, by tier).
**Restate / forward:** `subsec_hw-bytes`, `fig_bandwidth_ladder`, `fig_latency_ladder` (the ladders are extended, not re-derived); `subsec_hw-cpu` (pinned against pageable copies); the "Ridge point of a link" exercise of `sec_hardware` (a link has a ridge point: one sentence); `sec_read_write` (file layout is not re-taught); `subsec_mp-anatomy`; 4.1's `d2l.PackedStream` (shard format and cursor). Forward: 6.9 (checkpoint writes use these bandwidths).
**Exercises:** 6 to 8. Derive the batch at which KV offload to host memory overlaps decode for Qwen3-8B (derive); derive the host memory bandwidth the CPU Adam pass needs at 8 GPUs per node (derive); measure the page-cache effect by reading a file twice with and without `O_DIRECT` (measure); measure NVMe random reads at queue depths 1, 8 and 32 (measure); check the shard-size rule against the fitted alpha and beta of the shared file system (check); open: a two-tier KV store for the serving engine of Chapter 13, with the crossover prefix as its admission rule.
**Writer notes:** Pitfalls (blueprint 5.3): reading a file twice measures the page cache, use `O_DIRECT` or drop caches and say which; pageable copies are staged by the driver, so benchmarking only pinned memory overstates what `.to(device)` achieves; `nvbandwidth` under-reports with small buffers (use at least 512 MiB); NVMe "is fast" for large sequential reads at high queue depth only; GPUDirect Storage in compatibility mode is not GPUDirect; CXL and hard-disk rows are table rows only, no derivation depends on them. Hardware fallbacks: no `nvidia-fs` on the node (report compatibility mode, omit the GPUDirect bar); no root for a page-cache drop (`O_DIRECT` through `fio`, `dd iflag=direct`); no object-store endpoint (skip with a committed log from a previous allocation); Grace-tier numbers come from a paper and are labelled so. The checkpoint save and load measurement belongs to 6.9 together with `d2l.ckpt_stall` and `ckpt_records.jsonl`; `lm-hw-node-storage` keeps `tiers.json` (consumed by 13.6; unmeasured tiers omitted, never zero-filled). Exports: `d2l.tier_table`, `d2l.hw_h2d_bench`, `d2l.parse_nvbandwidth`, `d2l.parse_fio`, `d2l.parse_gdsio`, `d2l.fit_alpha_beta` (the one alpha-beta fitter of the part), `d2l.offload_breakeven`, `d2l.kv_load_vs_recompute`. Do not re-teach the ladders, pinned memory, safetensors. Detail: blueprint ch05 5.3 (commands M1 to M5, D1 to D3, tier table columns). The streaming-loader paragraph names MosaicML Streaming and WebDataset, one sentence each, as implementations of the shard rule.
**Sources:**
- Papers: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) (optimizer-state offload, the case of D1); [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) (offload across all tiers including NVMe); [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865) (KV and weight offload for inference); [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) (KV tiers across servers); [Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning](https://arxiv.org/abs/2408.14158) (cluster storage design); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (storage for checkpoints and data, one paragraph).
- Code and documentation: [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/) (`gdsio`, compatibility mode); [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (host-device sweeps); [fio](https://github.com/axboe/fio) (NVMe and file-system reads); [3FS](https://github.com/deepseek-ai/3FS) (a parallel file system built for this workload); [LMCache](https://github.com/LMCache/LMCache) (KV tiers in an engine); [MosaicML Streaming](https://github.com/mosaicml/streaming) and [WebDataset](https://github.com/webdataset/webdataset) (shard-based loaders over object stores).
- Courses, tutorials and blogs: Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (storage and I/O chapters); [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (the tier hierarchy); [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832) (dated contrast only).
-->


<!-- Placeholder (prose): opening (spec 22.4 part 1). The reader has the
ladders of subsec_hw-bytes for one GPU and pinned host memory from
subsec_hw-cpu. Qwen3-8B's mixed-precision training state (16 bytes per
parameter, subsec_mp-anatomy) fits neither one 24 GB card nor all four of
this machine's cards, host memory holds it twice over, and the PCIe link to
host memory (cell host-link) is about forty times slower than the card's
own memory (22.1's committed DRAM rate). The question: which bytes of a
workload (weights, optimizer state, activations, KV cache, dataset shards)
can sit one tier down without stalling the GPU? Idea: state can live on a
slower, larger tier whenever the link moves it faster than the GPU consumes
it. -->

```{.python .input #memory-tiers-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #memory-tiers-extras}
%%tab pytorch
import pandas as pd
```

```{.python .input #memory-tiers-opening}
%%tab pytorch
specs = d2l.chip_specs()
chip = specs.loc['RTX 4090']
cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
train = d2l.TrainStepModel(cfg_8b, chip, recipe='mixed16')
state = train.state['mixed16'] * train.N  # bytes held per parameter x N
n_cards, host = d2l.host_gpu_count(), d2l.host_memory_bytes()
print(f'Qwen3-8B training state, mixed precision: {state / 1e9:.0f} GB')
print(f'GPU memory: {chip.mem_gb:.0f} GB per card, '
      f'{n_cards * chip.mem_gb:.0f} GB on this machine\'s {n_cards} cards')
print(f'host memory: {host / 2**30:.0f} GiB, {host / state:.1f} times the '
      f'state')
```

## A Ladder of Tiers
:label:`subsec_lm_hw_tier_ladder`

<!-- Placeholder (prose): the table extends fig_bandwidth_ladder and
fig_latency_ladder downward (NVMe, parallel file system, object store, hard
disk) and sideways (coherent LPDDR5X, CXL); the ladders are not re-derived.
Every row is a data file row with a source and a retrieval date (the
artifact holds both; the sources are the A100 datasheet, the Ada
whitepaper, the Grace page, AMD's EPYC 7002 datasheet, the PCI-SIG rates,
the Solidigm P41 Plus product brief, the 3FS README, Cloudflare R2 and the
Seagate Exos datasheet; latencies of the GPU and DRAM rows from the MLSS
2026 latency ladder). The cost rank is ordinal (1 = cheapest per byte),
never a dollar figure (prices live in 22.8's table). Rows without a vendor
bandwidth (shared memory, L2) are measured in sec_lm_accelerator_design;
CXL and the hard disk are table rows only. Latency Numbers Every Programmer
Should Know as a dated contrast only. One sentence: d2l.tier_table loads the
sourced table and prints it (with a campaign's measured columns when one is
given). -->

```{.python .input #memory-tiers-tier-table}
%%tab pytorch
tiers = d2l.tier_table()
```

### The Host Link
:label:`subsec_lm_hw_host_link`

<!-- Placeholder (prose): a copy between host memory and the GPU crosses
the PCIe link: 16 lanes at 16 GT/s with 128b/130b encoding, 16 x 16 x
128/130 / 8 = 31.5 GB/s per direction (the A100 datasheet's 64 GB/s counts
both directions). Pinned memory is copied by the GPU's copy engines
directly; pageable memory is first staged by the driver into a pinned
buffer (subsec_hw-cpu), so benchmarking only pinned memory overstates what
.to(device) achieves. Small copies are latency-bound: t(s) = alpha + s /
beta, fitted by weighted least squares (weights 1 / t^2, fit_alpha_beta,
the part's one fitter); a sweep that stops at a few megabytes
under-reports beta. -->

```{.python .input #memory-tiers-h2d-bench}
%%tab pytorch
def hw_h2d_bench(nbytes, pinned=True, direction='h2d', numa=None,  #@save
                 device=None, repeats=10, huge_pages=False):
    """Milliseconds per copy (median and MAD over repeats) and GB/s of
    nbytes between host memory, pinned or pageable, and GPU memory: host to
    device ('h2d') or back ('d2h'). CUDA events bracket enough back-to-back
    copies for a few milliseconds; a pageable copy blocks the host, a
    pinned one is queued. The host buffer lives on NUMA node numa if given,
    in 2 MiB pages if huge_pages."""
    host, gpu = d2l._copy_buffers(nbytes, pinned, numa, device, huge_pages)
    src, dst = (host, gpu) if direction == 'h2d' else (gpu, host)
    copies = d2l._copies_per_timing(nbytes)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    ms = []
    with torch.cuda.device(gpu.device):
        for _ in range(repeats + 1):  # the first round warms up
            start.record()
            for _ in range(copies):
                dst.copy_(src, non_blocking=pinned)
            stop.record()
            stop.synchronize()
            ms.append(start.elapsed_time(stop) / copies)
    med, mad = d2l._hw_stats(ms[1:])
    return dict(ms=med, mad=mad, gbs=nbytes / med / 1e6)
```

<!-- Placeholder (prose): the alpha-beta model t(s) = alpha + s / beta is
linear in alpha and 1 / beta; weights 1 / t^2 make the fit relative, so the
small latency-bound sizes count as much as the large bandwidth-bound ones.
-->

```{.python .input #memory-tiers-fit-alpha-beta}
%%tab pytorch
def fit_alpha_beta(sizes, times, weights=None):  #@save
    """Fit t = alpha + s / beta to transfer times (seconds) at sizes
    (bytes) by weighted least squares, weights 1 / t^2 by default (each
    point's relative error counts alike). Returns (alpha, beta, cov) with
    alpha in seconds, beta in bytes per second and cov the 2 x 2
    covariance of (alpha, beta)."""
    s = np.asarray(sizes, dtype=float)
    t = np.asarray(times, dtype=float)
    w = 1 / t**2 if weights is None else np.asarray(weights, dtype=float)
    X = np.stack([np.ones_like(s), s], axis=1)  # columns: alpha, 1 / beta
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    alpha, g = A @ (XtW @ t)
    r = t - X @ np.array([alpha, g])
    dof = max(len(t) - 2, 1)
    cov_ag = A * (w @ r**2) / dof
    J = np.array([[1.0, 0.0], [0.0, -1 / g**2]])  # d(alpha, beta)/d(alpha, g)
    return float(alpha), float(1 / g), J @ cov_ag @ J.T
```

<!-- Placeholder (prose): the sweep, 4 KiB to 1 GiB in factors of 2,
pinned (4 KiB pages, what pin_memory gives), pinned in 2 MiB pages and
pageable, both directions. The data are not alpha-beta shaped: pinned h2d
sits near 19 GB/s from 1 to 64 MiB and climbs only for the largest copies,
and pinned d2h in 4 KiB pages falls off at 1 GiB while the 2 MiB-page
buffer holds about 26 GB/s both ways (this box's IOMMU translates every
DMA, DMA-FQ: fewer, larger pages miss its translation cache less). So beta
is the plateau of the copies of 256 MiB and more (median and range), alpha
the fit of fit_alpha_beta on the copies up to 1 MiB (claim M1: pinned h2d
against the per-direction ceiling, quoted at 2 digits; pageable markedly
slower). The link idles at Gen 1 and trains to Gen 4 under load (sysfs read
before the sweep, nvidia-smi during it): a link read at rest looks eight
times slower than it is. -->

```{.python .input #memory-tiers-host-link}
%%tab pytorch
h2d = d2l.hw_h2d_sweep(sizes=2**np.arange(12, 31),
                       modes=('pinned', 'pinned-2M', 'pageable'),
                       directions=('h2d', 'd2h'))
links = d2l.fit_h2d(h2d, alpha_below=2**20, plateau_from=2**28)
beta_pinned = links.loc[('pinned', 'h2d'), 'beta']  # bytes/s, the plateau
d2l.plot_h2d(h2d, links, pcie_gen=4, lanes=16)
assert links.loc[('pageable', 'h2d'), 'beta'] < beta_pinned  # staging costs
```

<!-- Placeholder (prose): the four cards at once, asked as a question with
two outcomes (claim M2): the EPYC 7502 has 128 Gen 4 lanes and about 200
GB/s of DRAM, so four links at about 25 GB/s each may scale, or a shared
resource may bind (the two host bridges of fig_pcie_topology, each shared
by a pair of cards, or host DRAM, measured next against its 8 x 25.6 GB/s).
The measurement needs every card, so the page shows the committed run and
re-measures one card the same way. On one socket in NPS1 mode the four
root complexes look like one NUMA node; the node campaign shows a
two-socket layout. NVIDIA's nvbandwidth (copy engines, 512 MiB buffers,
each card alone; its output is read by d2l.parse_nvbandwidth) cross-checks
the single-card figures in both directions. Host DRAM: STREAM's kernels
(McCalpin) at 8, 16 and 32 threads, the best of each (a memory-bound
kernel peaks below all cores), counted as STREAM counts them (bytes read plus
bytes written, no write-allocate traffic): a summing read, a copy and the
triad a = b + s c. The datasheet's 8 x 25.6 GB/s is a peak no kernel
reaches; the triad is the achievable rate, and the CPU optimizer's bound
below uses it. -->

<!-- Placeholder (prose): fix pass F2: the host-link lines now label the
plateau as "the median of the copies of 256 MiB and more, and their range"
(M1: pinned h2d 23 GB/s, range 21-24, is not a confidence interval), and
the four-card output gains a per-bridge line (M2: about 36 GB/s h2d per host
bridge, two cards behind it; all four cards 98% of 2 x 36: the bridge
binds, not DRAM). -->

```{.python .input #memory-tiers-four-cards}
%%tab pytorch
agg = d2l.hw_h2d_concurrent('box')
dram = d2l.hw_host_mem_bench(threads=(8, 16, 32))  # read, copy, triad
```

### Drives and Object Stores
:label:`subsec_lm_hw_storage`

<!-- Placeholder (prose): NVMe with fio. Reading a file twice measures the
page cache, so every read uses O_DIRECT (no root needed); the file is
written once first, so reads hit flash, not unwritten space; an NVMe drive
is fast only for large reads at depth: queue depth (outstanding requests)
times block size must cover the drive's internal parallelism. fio's JSON
is read by d2l.parse_fio (bandwidth, IOPS, mean and 99th-percentile
latency, and the sectors each drive read: disk_util). GDS (gdsio) is absent
here; d2l.parse_gdsio is tested on NVIDIA's documented output, and
compatibility mode is not GPUDirect: it is a bounce buffer in disguise. -->

<!-- Placeholder (prose): the sweep (claim M3): block sizes 4 KiB to 4 MiB,
sequential and random reads, one outstanding read or 64 for each of four
jobs, on a 16 GiB file, each point the median of three processes. Random
small reads are bound by IOPS, large ones by bandwidth; the knee is at
bs = beta / IOPS. Two drives in RAID 1 hold the same bytes, so a read can
come from either: at depth the pair reads faster than one drive's
datasheet, so it is compared with two drives' (fio's per-drive counters
show the split). The live check reads a
2 GiB scratch file at two points (skipped without the fio binary or below
10 GB free). -->

```{.python .input #memory-tiers-nvme}
%%tab pytorch
fio = d2l.hw_fio_sweep('box')
```

<!-- Placeholder (prose): the object store, the book's own R2 bucket over
the public internet: range GETs of 64 KiB to 64 MiB of one registered file,
one stream, and eight streams at once. The first byte waits for DNS, TCP,
TLS and the server (alpha in tens of milliseconds); one stream then
delivers tens of MB/s; parallel streams add up, which is why loaders over
object stores keep many requests in flight. -->

```{.python .input #memory-tiers-object-store}
%%tab pytorch
obj = d2l.hw_object_store('box')
```

<!-- Placeholder (prose): the ladders with this machine's points (figure
fig_lm_hw_tiers, computed: bandwidth and latency per tier; the GPU's shared
memory, L2 and GDDR from sec_lm_accelerator_design's committed run, the RTX
4090's, beside the table's A100 rows). Claim M8: measured within 20% of a
datasheet that states the same quantity: the host link (plateau), GDDR and
the RAID 1 pair against two drives. Not compared: the four host links
(drawn against 4 x the ceiling; M2 answers what binds) and host DRAM, whose
8 x 25.6 GB/s is a peak no kernel reaches (its STREAM fractions are
printed above). The node's DRAM, CXL, LPDDR5X and the hard disk have no
point here. -->

```{.python .input #memory-tiers-tiers-measured}
%%tab pytorch
measured = d2l.tier_measurements(links, agg, dram, fio, obj)
compare = d2l.plot_tiers(tiers, measured)  # datasheet bars, measured dots
d2l.show_table(compare, digits={'datasheet': 1, 'measured': 1,
                                'fraction': 2})
```

<!-- Placeholder (prose): D3, the shard rule (claim M6). A read of s bytes
takes t(s) = alpha + s / beta and so achieves s / (alpha beta + s) of the
bandwidth; 90% needs s >= 9 alpha beta. For the host link that is a few
megabytes, for one NVMe reader about a megabyte, for the object store tens
of megabytes: shards are large because of alpha, not beta. -->

```{.python .input #memory-tiers-shard-rule}
%%tab pytorch
rule = d2l.shard_rule(links, fio, obj)
```

## Where Each Byte Lives
:label:`subsec_lm_hw_placement`

<!-- Placeholder (prose): training keeps weights, gradients, optimizer
state and activations in GPU memory unless offloaded (16 bytes per
parameter in mixed precision; activations recomputed or offloaded);
serving keeps weights and the KV cache in GPU memory, then host memory,
then shared SSD-backed stores (Mooncake, LMCache); datasets and checkpoints
live on drives and object stores (one sentence; Chapter 23 owns the
checkpoint write, by :numref: if the label resolves). Figure
fig_lm_hw_state_placement (generator state-placement, illustrative): each
kind of state by tier. -->

## The Offload Condition
:label:`subsec_lm_hw_offload`

<!-- Placeholder (prose): D1. A step of F FLOPs that needs B bytes from the
far side of a link of bandwidth beta_l stalls unless F / P_eff >= B /
beta_l: the FLOPs per byte moved must exceed the link's ridge P_eff /
beta_l (the "ridge point of a link" exercise of sec_hardware, one
sentence). Optimizer-state offload (ZeRO-Offload): per step and GPU, T
tokens cost 6 N T FLOPs, the bf16 gradients go down (2N bytes) while the
updated bf16 weights come up (2N bytes) on the full-duplex link: no stall
iff T >= T_link = P_eff / (3 beta_l), independent of N. The plan's A100
example (250 TFLOP/s, 25 GB/s: about 3,300 tokens) is quoted as the
plan's. -->

```{.python .input #memory-tiers-offload-breakeven}
%%tab pytorch
def offload_breakeven(p_eff, link_bw, bytes_per_param_per_dir=2):  #@save
    """T_link, the tokens per GPU and step above which the transfers of an
    offloaded optimizer (bytes_per_param_per_dir down and as many up per
    parameter, full duplex) hide behind the step's 6 N T FLOPs:
    6 N T / p_eff >= b N / link_bw, so T >= p_eff b / (6 link_bw)."""
    return p_eff * bytes_per_param_per_dir / (6 * link_bw)
```

<!-- Placeholder (prose): D1', the CPU's own threshold (review H5). The CPU
AdamW takes s_adam seconds per parameter, so overlap also needs 6 N T /
P_eff >= N s_adam, T >= T_cpu = P_eff s_adam / 6, again independent of N.
The update moves q bytes of host memory per parameter, counted from what
it does: the bf16 gradient read (2) and written as fp32 (4); AdamW reads
weight, gradient and both moments (16) and writes weight and moments (12);
the fp32 weight read (4) and written as bf16 (2): q = 40 bytes of operands,
counted as STREAM counts, so the bound to compare with is q / B_triad (the
triad above, same counting). Part of each tensor's traffic stays in the 128
MiB L3 (a 4M-parameter tensor's fp32 gradient is written and read again at
once), so the measured s_adam can sit near or even below that bound: the
text says "within X of the host-memory bound", not "bandwidth bound". The
update runs on 8 threads: it is memory bound, 8 threads are within about
15% of 32, and the offloaded run below shows what 32 cost. P_eff is 22.3's
calibration (the best bf16 product: the rate the GPU could reach), beta_l
the pinned host link's plateau; the cell says which threshold binds at
that rate. -->

```{.python .input #memory-tiers-offload-threshold}
%%tab pytorch
calib = d2l.hw_calibrate(log=d2l.hw_box_log('box-microbench'))
P_eff = calib['P_eff_bf16']  # FLOP/s, the GPU's best bf16 product
s_adam = d2l.hw_cpu_adam_bench(threads=8)  # seconds per parameter
q = 2 + 4 + 16 + 12 + 4 + 2  # bytes of operands per parameter
B_triad = dram['triad']  # GB/s, so q / B_triad is in ns
T_link = d2l.offload_breakeven(P_eff, beta_pinned)
T_cpu = P_eff * s_adam / 6
print(f'CPU update: {s_adam * 1e9:.2f} ns per parameter; q / B_triad = '
      f'{q} B / {B_triad:.0f} GB/s = {q / B_triad:.2f} ns')
print(f'at P_eff = {P_eff / 1e12:.0f} TFLOP/s: T_link = P_eff / (3 beta_l) '
      f'= {round(T_link, -2):,.0f} tokens per GPU')
print(f'T_cpu = P_eff s_adam / 6 = {round(T_cpu, -2):,.0f} tokens per GPU: '
      f'the {"CPU" if T_cpu > T_link else "link"} binds')
```

<!-- Placeholder (prose): the toy offloaded run (claim M5). A Qwen3-style
model of 0.82 billion parameters (d = 2,048, 16 layers, 16 query and 8 KV
heads, SwiGLU 5,632, vocabulary 32,768, tied), bf16 weights and gradients
on the GPU, fp32 master weights and AdamW moments in pinned host memory (12
bytes per parameter, about 10 GB). A step of T tokens runs micro-batches
of 4,096 (one shorter sequence below that) with gradient accumulation and
chunked_lm_loss. ZeRO-Offload's delayed parameter update, with its
one-step staleness: on the last micro-batch each finished gradient goes
down on a side stream, the CPU updates that tensor as it lands, and its new
weights go up on another stream while the GPU already runs the next step
on the previous weights; the update must be in before the step after
that. The overlap window is the whole step, and the step time is predicted
by max(t_compute, t_link, t_cpu). The toy does not train at P_eff: the
prediction uses its own rate (the compute-only arm, about half of P_eff),
the link alone, and the CPU update's time during the offloaded steps (the
worker times it; it runs 30-50% slower there than alone, which points to
contention with the DMA and the thread feeding the GPU: not yet measured),
so T_link and T_cpu of the previous cell are what the GPU could reach, not
this run's. Measured at 1,024 to 131,072 tokens: offloaded with the
optimizer on 8 threads and on 32, compute only, and the link and the CPU
update alone. The measured
crossover is the corner of a fit of max(a, b T) to the offloaded steps;
above it the offloaded step should cost what compute alone costs (the
excess is printed). With 32 optimizer threads the GPU's own work slows
(the busy time printed): the threads compete for cores and memory with the
thread that launches the GPU's kernels, so a CPU optimizer should leave
that thread its cores. GPU memory: weights,
two staging copies, gradients and activations, about 14 GiB. Without the
delayed update a per-layer tail stays exposed (an exercise). -->

```{.python .input #memory-tiers-offload-run}
%%tab pytorch
run = d2l.hw_offload_run('box')
d2l.plot_offload(run)  # predicted from the run's own rate and times
```

<!-- Placeholder (prose): D2, KV load against recompute. Loading a cached
prefix of n tokens from a tier costs alpha + n k_KV / beta; recomputing it
costs prefill_time(n) of 22.3's ServingModel at P_eff, whose attention term
grows with n^2. The table gives, per tier, both times at 8,192 tokens and
the shortest prefix from which loading is at least as fast ("any": from
one token, since recomputing even a short prefix reads every weight once;
"never": recompute wins up to the model's 40,960-token context).
Recompute uses 22.3's measured rates (P_eff and beta_eff); each tier its
fitted alpha and beta from the shard rule, the drives as one reader at
queue depth 1 (2.6 GB/s, not the 7.5 GB/s of many requests at depth).
Mooncake and LMCache build serving engines on this rule. -->

```{.python .input #memory-tiers-kv-def}
%%tab pytorch
def kv_load_vs_recompute(cfg, n, link_bw, p_eff, chip=None,  #@save
                         alpha=0.0, n_max=None, beta_eff=None):
    """Seconds to load the KV cache of prefixes of n tokens over a link
    (alpha + n k_KV / link_bw) and to recompute them (ServingModel's
    prefill_time at p_eff and the memory rate beta_eff, by default the
    chip's datasheet; chip None: FLOPs alone), and the crossover: the
    shortest prefix up to n_max (the model's context) from which loading
    is at least as fast (inf if none)."""
    chip = d2l._flops_only_chip(p_eff) if chip is None else chip
    m = d2l.ServingModel(cfg, chip, calib=dict(
        P_eff_bf16=p_eff, beta_eff=beta_eff or chip.mem_bw_tbs * 1e12))
    k = m.kv_bytes_per_token()  # bytes of K and V per token, all layers
    n_max = n_max or cfg.max_position_embeddings

    def times(x):
        return alpha + x * k / link_bw, np.array([m.prefill_time(int(v))
                                                  for v in x])
    grid = np.unique(np.geomspace(1, n_max, 400).astype(int))
    load, recompute = times(grid)
    loses = np.flatnonzero(load > recompute)  # the last prefix it loses at
    first = 0 if len(loses) == 0 else loses[-1] + 1
    load_n, recompute_n = times(np.asarray(n))
    return dict(n=np.asarray(n), load_s=load_n, recompute_s=recompute_n,
                k_kv=k, n_max=n_max,
                crossover=grid[first] if first < len(grid) else np.inf)
```

```{.python .input #memory-tiers-kv-recompute}
%%tab pytorch
prefix = 8192  # tokens of a cached prompt
kv = {tier: d2l.kv_load_vs_recompute(cfg_8b, [prefix], r.beta, P_eff, chip,
                                     alpha=r.alpha,
                                     beta_eff=calib['beta_eff'])
      for tier, r in rule.iterrows()}  # fitted alpha, beta; NVMe: one reader
d2l.show_table(pd.DataFrame({
    'load 8K (ms)': [x['load_s'][0] * 1e3 for x in kv.values()],
    'recompute 8K (ms)': [x['recompute_s'][0] * 1e3 for x in kv.values()],
    'load wins from': [d2l.prefix_words(x['crossover'], x['n_max'])
                       for x in kv.values()]}, index=kv))
```

## Data Loading
:label:`subsec_lm_hw_dataloading`

<!-- Placeholder (prose): a node training at a million tokens per second
reads two megabytes per second of text (2 bytes per token in chapter 20's
shards), three orders of magnitude below one drive (claim M4, computed and
checked by one measured read). What is storage-bound instead:
preprocessing (Chapter 20), multimodal data and checkpoints. The dataset
path is about shard size (the rule above), resumable cursors (chapter 21's
PackedStream) and per-open latency, not bandwidth. MosaicML Streaming and
WebDataset (one sentence each) implement the shard rule over object
stores. -->

```{.python .input #memory-tiers-data-rate}
%%tab pytorch
shards = d2l.fwedu_shards('100m')
read = d2l.hw_shard_read(shards, tokens=2**26)
tokens_per_s = 1e6  # a node training at a million tokens per second
need = tokens_per_s * shards.dtype.itemsize  # bytes/s of text
print(f'needed at 1M tokens/s: {need / 1e6:.0f} MB/s, '
      f'{round(read["drive_bps"] / need, -2):,.0f}x below the drive')
assert read['drive_bps'] > 1000 * need  # three orders of magnitude
```

<!-- Placeholder (prose): the node campaign (C-T3, pending; claim M7):
per-NUMA host-link sweeps on a two-socket node, GPUDirect Storage against
a bounce buffer, a shared file system; the sentence is rewritten when the
log is committed. -->
<!-- campaign: lm-hw-node-storage-v1 -->

```{.python .input #memory-tiers-node-twin}
%%tab pytorch
node = d2l.tier_table(d2l.hw_campaign_log('node-storage'))
```

## Summary

<!-- Placeholder (prose): 3-6 sentences; forward to sec_lm_server_paths
(the links between devices, not tiers) and to Chapter 23's checkpoint
writes by :numref: if the label resolves. -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.4.md part 9: the
batch at which KV offload to host memory overlaps decode for Qwen3-8B
(derive); the host bandwidth the CPU Adam needs at 8 GPUs per node
(derive); the page-cache effect with and without O_DIRECT (measure, 2
min); the exposed tail of the offloaded run without the delayed update
(measure, 3 min); NVMe random reads at QD 1, 8, 32 (measure, 3 min); the
shard rule against the object store's fit (check); open: a two-tier KV
store for the serving engine with the crossover prefix as admission
rule. -->
