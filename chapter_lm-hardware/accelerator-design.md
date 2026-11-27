# Accelerator Design
:label:`sec_lm_accelerator_design`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.1 Accelerator Design · T0 (T2 microbenchmarks)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A chip is a budget of die area, power and package perimeter, and three ratios computed from its datasheet place it among its competitors.
**Opening:** The reader has `tab_gpu_specs` with four GPUs and the shoreline argument of `subsec_hw-shoreline`: compute grows with area, bandwidth with perimeter. One concrete example: two datasheets, one part with twice the FLOP/s and half the memory bandwidth of the other, and one decode step of Qwen3-8B to run on each; which is faster is not obvious from either number alone. The open question: what single plane lets a reader place any accelerator, from an A100 to a wafer-scale part, and read off what it is good for.
**Teaches:**
- The three budgets: die area (reticle limit, chiplets) split among matrix units, on-chip SRAM and memory controllers; watts per chip and per rack; package perimeter, which caps the number of memory stacks and I/O lanes. Restated from `subsec_hw-shoreline`, not re-derived.
- Off-chip memory choices and what each buys: HBM (bandwidth, few stacks, cost per GB), GDDR (cheap bandwidth, little capacity), LPDDR (capacity per dollar and per watt), SRAM only (very high bandwidth, tiny capacity per chip).
- Five datasheet numbers (peak FLOP/s per format, memory bandwidth, capacity, scale-up bandwidth, scale-out bandwidth) and three ratios: the ridge point P/beta, the sweep time C/beta (seconds to read all of memory once), and network bytes per FLOP. The specification table is a data file with a source URL and a retrieval date per row: A100 to B200 and Rubin (announced), TPU v7 Ironwood, Trainium3, MI455X (Helios), Groq's LPU and Cerebras wafer-scale, Rubin CPX as the GDDR prefill part; one sentence on the NVIDIA licence of Groq's inference technology (December 2025).
- The design plane: ridge point against sweep time, with a side strip for parts that publish no dense 16-bit figure; what the training, prefill and decode corners of the plane mean (one sentence each; 5.3 develops them).
**Derivation:** Sweep time is a floor on per-token decode latency for a model that fills memory: if W + b L k_KV = C, every decode step reads at least C bytes, so t_step >= C/beta independent of the batch, and tensor parallelism over c chips leaves C/beta unchanged because both scale by c (blueprint ch05 5.1 D2). Smallest numeric case: the A100 PCIe 40 GB row gives 40 GB / 1.555 TB/s = 26 ms, so no model that fills the card decodes faster than about 39 tokens per second per user on one or on eight of them. Second derivation, two sentences: an SRAM-only chip of capacity S_c holds a model on N_c = ceil(bytes / (eta S_c)) chips, and chip-seconds per token are N_c times the stage time (blueprint 5.1 D3); the dollar side is 5.8.
**Notebook:**
1. Question: do the datasheet ratios predict which chip is faster for each of the three workloads, and does one A100 reproduce its datasheet's memory ladder?
2. T0: load `d2l.chip_specs()`, add the ratios with `d2l.derive_ratios`, plot the design plane; the four `tab_gpu_specs` rows are a regression test (loading asserts they match the book table).
3. T0: the sweep-time floor for every row for Qwen3-8B at L = 8,192, and the SRAM chip count for Qwen3-8B and a 70B configuration at eta = 0.8 (a parameter, printed).
4. T2 (one A100 PCIe 40 GB): the bandwidth ladder of `subsec_hw-bytes` measured on the A100: HBM copy and read-only bandwidth, L2 and shared-memory bandwidth (two short `load_inline` kernels; committed logs if `nvcc` is absent), each plotted against the datasheet line; the GEMM-by-shape sweep is 5.2's, not repeated here. Check: within 15% of the datasheet for HBM; the L2 and shared-memory numbers are reported against the whitepaper with no tolerance claim.
5. T1 twin: the same functions on the reader's GPU at most 8 GiB, with an FP8 series on Ada (A100 has no FP8 tensor cores).
6. Claim that survives noise: the measured HBM bandwidth is within 15% of the datasheet on both hosts; the ordering of chips in the plane is a datasheet fact, not a measurement.
**Figures:** `fig_lm_hw_chip_budget` (area, power and perimeter split among matrix units, SRAM, memory controllers and links); `fig_lm_hw_accelerator_plane` (plot: ridge point against sweep time, one marker per row, the three workload corners annotated); `fig_lm_hw_a100_ladder` (plot: measured bandwidth per level against the datasheet line).
**Restate / forward:** `subsec_hw-shoreline` (compute scales with area, bandwidth with perimeter); `tab_gpu_specs` (the four rows, extended, not replaced); `subsec_hw-gpu` (tensor cores halve time per format step); `subsec_hw-bytes` and `fig_bandwidth_ladder` (the ladder is the target the measurement reproduces); `eq_hw_decode_bound` (the single-stream bound is the b = 1 case of the floor). Forward: 5.3.
**Exercises:** 6 to 8. Derive the sweep-time floor for an MoE whose active experts are a fraction of memory (derive); derive how many Cerebras-class chips hold Qwen3-235B-A22B and the implied per-token chip-seconds (derive); measure the half-peak message size of a device-to-device copy on the reader's GPU (measure); check which rows of the table are bidirectional aggregates and recompute the per-direction scale-up bandwidth (check); open: add a row for a chip released after the edition and place it on the plane from its datasheet alone.
**Writer notes:** The table stores dense figures only; the A100 datasheet prints sparse and dense side by side and mixing them is the commonest error (blueprint 5.1 pitfalls). Vendor memory bandwidth for multi-die parts is aggregate over dies; one die cannot read the other's HBM at that rate. A copy kernel counts read plus write, a read-only kernel read only; the plot states which. TF32 is a compute format with 32-bit storage. Rows updated per `sota.md` A8: Trainium3 UltraServers (GA December 2025), Ironwood (GA), MI455X with Helios as the rack (launched July 2026), Rubin and Rubin CPX marked announced, Groq licence sentence. SRAM-chip dense 16-bit figures may be unpublished: store `NaN`, plot in the side strip. Clocks may not be lockable on the T2 host: record them and raise the claim threshold to 5x MAD for any ratio of two series not run back-to-back. `check_specs` compares `get_device_properties().name` and total memory with the `A100 PCIe 40 GB` row before drawing a datasheet line. Do not re-teach: the shoreline argument, tensor cores and formats (`subsec_hw-gpu`), the roofline (`eq_roofline`), the square-matmul sweep (`subsec_perf-sweep`). Exports: `d2l.chip_specs`, `d2l.derive_ratios`, `d2l.BenchLog` (with `.power_trace()` sampling pynvml at 10 Hz, read by 5.8), `d2l.env_fingerprint`, `d2l.hw_matmul_bench`, `d2l.hw_copy_bench`; artifacts `lm-hw-a100-microbench` (bandwidth entries; 5.2 adds the GEMM entries to the same tar) and `lm-hw-buildbox-microbench`. The tile-intensity derivation of blueprint 5.1 D1 is taught in 5.2. Detail: blueprint ch05 5.1 (specification table rows and sources, microbenchmark commands, pitfalls, risks). The Hopper dissection row (L2 hit rate from `ncu`) is an optional cell.
**Sources:**
- Papers: [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174) (the A100 ladder the T2 cells reproduce); [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499) (the Hopper counterpart, optional cell); [AI and Memory Wall](https://arxiv.org/abs/2403.14123) (compute outruns bandwidth across generations); [Chiplet Cloud](https://arxiv.org/abs/2307.02666) (SRAM-heavy chiplet design point); [WaferLLM](https://arxiv.org/abs/2502.04563) (wafer-scale inference, the capacity argument of D3); [TPU v4](https://arxiv.org/abs/2304.01433) (a non-GPU design with optical circuit switches).
- Code and documentation: Google, [Ironwood](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/) (TPU v7 row); AWS, [Trn3 UltraServers](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/) (Trainium3 row); AMD, [Helios](https://www.amd.com/en/products/rackscale-solutions/helios.html) (MI455X row and the rack); NVIDIA, [Rubin CPX announcement](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) (GDDR prefill part); Groq, [The LPU Explained](https://groq.com/blog/the-groq-lpu-explained) and the [NVIDIA licensing agreement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) (SRAM row and its 2025 status); Cerebras, [Inference 3x faster](https://www.cerebras.ai/blog/cerebras-inference-3x-faster) (wafer-scale row); NVIDIA, [Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (register-file and cluster facts used by 5.2).
- Courses, tutorials and blogs: [Scaling Book: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) (chip-level view); [Modal GPU Glossary](https://modal.com/gpu-glossary) (vocabulary); [GPU MODE lectures](https://github.com/gpu-mode/lectures) (microbenchmark methodology).
-->

<!-- Placeholder (prose): opening. Two datasheets, one part with twice the
FLOP/s and half the bandwidth of the other, and one decode step of Qwen3-8B
on each; the question of one plane that places any accelerator. -->

```{.python .input #accelerator-design-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #accelerator-design-extras}
%%tab pytorch
import pandas as pd
```

<!-- Placeholder (prose): the specification table is a data file, one row
per chip with its source and retrieval date; dense figures only. -->

```{.python .input #accelerator-design-chip-specs}
%%tab pytorch
def chip_specs():  #@save
    """The accelerator table, one row per chip (indexed by name): memory
    (GB, TB/s), on-chip SRAM (MB), dense peaks (TFLOP/s) in 16, 8 and 4
    bits, power (W), scale-up and scale-out bandwidth (GB/s), each row with
    its source and retrieval date. NaN where the vendor states nothing."""
    df = d2l._hw_static('hw-chip-specs').set_index('chip')
    d2l._check_specs(df)  # sources, dates, and the book's four rows
    return df
```

<!-- Placeholder (prose): three of its rows answer the opening's question
for an HBM GPU, a GDDR card and an SRAM-only chip. The GDDR prefill part
(Rubin CPX) publishes no bandwidth yet, so the GDDR row is the RTX 5090.
One decode step at batch 1 reads every weight once, so it takes at least
W / beta (the b = 1 case of eq_hw_decode_bound): the FLOP/s column does not
decide it. "Chips" is the least number that holds the weights at full
capacity; the SRAM cell below adds the usable fraction eta. -->

```{.python .input #accelerator-design-three-chips}
%%tab pytorch
specs = d2l.chip_specs()
cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
W_8b = d2l.config_memory(cfg_8b, 1, 1)['weights']  # bytes in bf16
three = specs.loc[['H100 SXM', 'RTX 5090', 'Groq LPU'],
                  ['mem_type', 'mem_gb', 'mem_bw_tbs', 'p16_tflops']].copy()
three['step_ms'] = W_8b / 1e9 / three.mem_bw_tbs  # GB / (TB/s) = ms
three['sweep_ms'] = three.mem_gb / three.mem_bw_tbs
three['chips'] = np.ceil(W_8b / (three.mem_gb * 1e9)).astype(int)
print(f'Qwen3-8B weights in bf16: W = {W_8b / 1e9:.1f} GB')
d2l.show_table(three, headers={
    'mem_type': 'memory', 'mem_gb': 'GB', 'mem_bw_tbs': 'TB/s',
    'p16_tflops': 'TFLOP/s', 'step_ms': 'W/beta (ms)',
    'sweep_ms': 'sweep (ms)', 'chips': 'chips, at least'})
```

## A Chip Is a Budget
:label:`subsec_lm_hw_budget`

<!-- Placeholder (prose): recap of subsec_hw-shoreline (one paragraph); the
three budgets (area: matrix units, SRAM, memory controllers and PHYs,
control; power per chip and per rack; perimeter: HBM stacks and SerDes
lanes; reticle limit about 858 mm^2, chiplets and interposers);
eq_lm_hw_areabudget; figure fig_lm_hw_chip_budget (generator chip-budget). -->

### HBM, GDDR, LPDDR and SRAM
:label:`subsec_lm_hw_memmenu`

<!-- Placeholder (prose): table tab_lm_hw_memmenu (HBM, GDDR, LPDDR, DDR5,
SRAM-only: bandwidth and capacity per device, interface width x rate, cost
rank, why it exists). An HBM stack has a 1,024-bit interface, so its
bandwidth is the stack count times the width times the per-pin rate; the
cell reads the per-pin rate off two datasheets (H100 SXM: five active HBM3
stacks; B200: eight HBM3e stacks). -->

```{.python .input #accelerator-design-hbm-stack}
%%tab pytorch
stack_bits = 1024  # HBM interface width per stack (JEDEC)
stacks = {'H100 SXM': 5, 'B200': 8}
for chip, n_stacks in stacks.items():
    beta = specs.loc[chip, 'mem_bw_tbs']  # TB/s
    rate = beta * 1e12 * 8 / (n_stacks * stack_bits) / 1e9  # Gb/s per pin
    print(f'{chip} ({specs.loc[chip, "mem_type"]}): {beta:.2f} TB/s over '
          f'{n_stacks} stacks x {stack_bits:,} pins = {rate:.1f} Gb/s per pin')
```

## Five Numbers and Three Ratios
:label:`subsec_lm_hw_ratios`

<!-- Placeholder (prose): the five datasheet numbers (peak FLOP/s per
format, memory bandwidth, capacity, scale-up and scale-out bandwidth) and
three ratios: the ridge P/beta, the sweep time C/beta (GB over TB/s is
milliseconds) and network bytes per FLOP; dense figures only; vendors state
scale-up links as bidirectional aggregates. -->

```{.python .input #accelerator-design-derive-ratios}
%%tab pytorch
def derive_ratios(df):  #@save
    """Add the design ratios to a chip table: the ridge point (FLOP/byte,
    dense 16-bit), the sweep time C / beta (ms), the scale-out and
    scale-up bytes per FLOP, and on-chip SRAM as a fraction of memory."""
    out = df.copy()
    out['ridge16'] = df.p16_tflops / df.mem_bw_tbs  # TFLOP/s over TB/s
    out['sweep_ms'] = df.mem_gb / df.mem_bw_tbs
    flops = df.p16_tflops * 1e12
    out['net_bytes_per_flop'] = df.scaleout_gbs * 1e9 / flops
    out['scaleup_bytes_per_flop'] = df.scaleup_gbs * 1e9 / flops
    out['hbm_fraction_sram'] = df.sram_mb / (1000 * df.mem_gb)
    return out
```

### The Design Plane
:label:`subsec_lm_hw_plane`

<!-- Placeholder (prose): how to read the plane: up is compute-rich for its
bandwidth (prefill, training), left is fast to sweep (decode); marker area is
capacity; chips without a published dense 16-bit figure sit in the strip
below. One sentence per corner; 22.3 develops them. -->

```{.python .input #accelerator-design-ratios-plane}
%%tab pytorch
ratios = d2l.derive_ratios(specs)
d2l.plot_chip_plane(ratios)
```

<!-- Placeholder (prose): the ordering in the plane is a datasheet fact, not
a measurement (claim C4). -->

### The Sweep-Time Floor
:label:`subsec_lm_hw_sweep_floor`

<!-- Placeholder (prose): derivation D2 and eq_lm_hw_sweep_floor: if
W + b L k_KV = C, every decode step reads at least C bytes, so
t_step >= C / beta for any batch, and tensor parallelism over c chips scales
C and beta alike. The A100 PCIe 40 GB: 40 GB / 1.555 TB/s = 25.7 ms, at most
39 tokens per second per user. -->

```{.python .input #accelerator-design-sweep-floor-def}
%%tab pytorch
def sweep_floor(specs, cfg, L, mem_util=0.9):  #@save
    """The decode floor of a model that fills each chip: the largest batch
    b_max of sequences of L tokens whose weights W and KV cache fit
    mem_util of memory, the step time (W + b_max L k_KV) / beta that every
    step at that batch needs (ms), and the per-user token rate it allows."""
    mem = d2l.config_memory(cfg, L, 1)
    W, kv = mem['weights'], mem['kv_cache']  # bytes; KV of one sequence
    C, beta = specs.mem_gb * 1e9, specs.mem_bw_tbs * 1e12
    b_max = np.floor((mem_util * C - W) / kv)
    b_max = b_max.where(b_max >= 1)  # NaN: the weights alone do not fit
    floor_ms = (W + b_max * kv) / beta * 1e3
    return pd.DataFrame({'sweep_ms': C / beta * 1e3, 'b_max': b_max,
                         'floor_ms': floor_ms, 'user_tps_max': 1e3 / floor_ms})
```

<!-- Placeholder (prose): the table keeps the chips with off-chip memory;
an SRAM-only chip holds no KV cache of this size, and the chip count below
is its lens. -->

```{.python .input #accelerator-design-sweep-floor}
%%tab pytorch
L = 8192  # tokens per sequence
off_chip = specs[specs.mem_type != 'SRAM']
floor = d2l.sweep_floor(off_chip, cfg_8b, L).dropna()
d2l.show_table(floor.sort_values('floor_ms'), digits={'b_max': 0}, headers={
    'sweep_ms': 'sweep (ms)', 'b_max': 'b_max', 'floor_ms': 'floor (ms)',
    'user_tps_max': 'tokens/s per user'})
```

<!-- Placeholder (prose): D3 in two sentences: an SRAM-only chip of
capacity S_c holds a model on N_c = ceil(bytes / (eta S_c)) chips, and
chip-seconds per token are N_c times the stage time; the dollar side is
22.8. -->

```{.python .input #accelerator-design-sram-chips-def}
%%tab pytorch
def sram_chips_needed(model_bytes, sram_bytes, eta=0.8):  #@save
    """Chips of sram_bytes each that hold model_bytes when a fraction eta
    of every chip is usable (the rest holds activations and buffers)."""
    return math.ceil(model_bytes / (eta * sram_bytes))
```

```{.python .input #accelerator-design-sram-chips}
%%tab pytorch
eta = 0.8  # usable fraction of each chip's SRAM
W_70b = d2l.config_memory(d2l.HW_SHAPES['dense-70b'], 1, 1)['weights']
for chip in ('Groq LPU', 'Cerebras WSE-3'):
    S_c = specs.loc[chip, 'mem_gb'] * 1e9
    for model, W in (('Qwen3-8B', W_8b), ('dense 70B', W_70b)):
        N_c = d2l.sram_chips_needed(W, S_c, eta)
        print(f'{model} ({W / 1e9:.0f} GB) on {chip} ({S_c / 1e9:g} GB '
              f'each, eta {eta}): {N_c} chip{"s" if N_c > 1 else ""}')
```

## Measuring One Chip
:label:`subsec_lm_hw_measure`

<!-- Placeholder (prose): the microbenchmark contract in one paragraph
(command, log, statistics: median and MAD of at least 5 timed repeats after
a warm-up, the headline the median of three processes' medians, their spread
the noise a gap must exceed; plot with the datasheet line, the twin on the
A100); the numbers below are the committed regeneration, and a "live check"
line repeats a few points on the reader's GPU. Every measured point also
goes into a log (d2l.BenchLog): one record per point with the command, its
parameters and results, the raw tool output, the environment below, and
whether another user's process shared the card; the energy section samples
power through the same log. First, the box as the measured object
(d2l.env_fingerprint: versions, clocks, power limits, PCIe link). -->

```{.python .input #accelerator-design-fingerprint}
%%tab pytorch
env = d2l.env_fingerprint()
d2l.show_env(env)
```

### The Bandwidth Ladder

<!-- Placeholder (prose): two kernels, a read and a copy (read plus write
counted). A kernel launch costs about 5 microseconds, the time L2 needs for
20 MB, so the read kernel streams a small working set many times (loads
cached in L2, not L1) and the copy is replayed in one CUDA graph; a working
set that fits the 72 MB L2 is served from L2 after its first pass, a larger
one from DRAM. The read kernel shrinks its blocks for a small working set so
that every SM stays busy. d2l.hw_bandwidth_ladder runs hw_copy_bench over the
working sets, each point after a short warm-up, and records the SM clock and
board power. -->

```{.python .input #accelerator-design-copy-bench}
%%tab pytorch
def hw_copy_bench(nbytes, kind='copy', repeats=20, graph=True):  #@save
    """Bandwidth (GB/s) of streaming nbytes of fp32 on the GPU: 'read'
    reads them `passes` times in one kernel (passes x nbytes moved), 'copy'
    copies them once (2 nbytes: read + write). Returns gbs, ms per call
    (median of repeats) and its MAD."""
    x = torch.empty(nbytes // 4, device='cuda').uniform_()
    if kind == 'read':
        passes = max(1, 2**28 // nbytes)  # at least 256 MiB per kernel
        fn, moved = (lambda: d2l.read_passes(x, passes)), passes * nbytes
    else:
        y = torch.empty_like(x)
        fn, moved = (lambda: y.copy_(x)), 2 * nbytes
    ms, mad = d2l._hw_stats(d2l._hw_time(fn, repeats, graph))
    return dict(gbs=moved / ms / 1e6, ms=ms, mad=mad)
```

```{.python .input #accelerator-design-ladder}
%%tab pytorch
ladder = d2l.hw_bandwidth_ladder(sizes=2**np.arange(18, 33),
                                 kinds=('read', 'copy'))
d2l.plot_ladder(ladder, specs,
                reference=d2l.hw_campaign_log('a100-microbench'))
```

### The Format Ladder

<!-- Placeholder (prose): one 8192^3 product in each format; FP32 on the
CUDA cores, TF32 (FP32 storage, a 10-bit-mantissa compute format) and the
16- and 8-bit formats on the tensor cores; the 4090's BF16 and FP8 figures
are the FP32-accumulate rates, half the FP16-accumulate figure on the vendor
page. FP8 needs a scale per tensor so that the largest entry lands near the
largest E4M3 value, 448. -->

<!-- Placeholder (prose): one product per format; FP8 takes both operands
through chapter 21's d2l.fp8_quant with one scale per tensor (cite
:numref:`sec_lm_low_precision`; fix F9 replaced this section's own
fp8_per_tensor, which returned the same pair) and multiplies with
torch._scaled_mm, which applies the two scales. Ruling D3: fp8 runs natively
on every GPU with fp8 tensor cores (sm_89 and newer, d2l.has_fp8_cores); on
one without them (the A100 of C-T2) the ladder records "no fp8 tensor cores"
instead of failing. -->

```{.python .input #accelerator-design-matmul-bench}
%%tab pytorch
def hw_matmul_op(m, n, k, dtype, device='cuda'):  #@save
    """A zero-argument (m x k) @ (k x n) product in a format: 'fp32',
    'tf32' (both FP32 storage), 'bf16', 'fp16', or 'fp8' (E4M3 with
    per-tensor scales through torch._scaled_mm, bf16 output; only on a GPU
    with fp8 tensor cores)."""
    if dtype == 'fp8' and not d2l.has_fp8_cores(device):
        raise ValueError(f'fp8 needs fp8 tensor cores (sm_89 or newer); '
                         f'{d2l.device_name(device)} has none')
    A = torch.randn(m, k, device=device)
    B = torch.randn(n, k, device=device)  # B.T is (k, n), column-major
    if dtype == 'fp8':
        (A8, s_A), (B8, s_B) = (d2l.fp8_quant(X, granularity='tensor')
                                for X in (A, B))
        return lambda: torch._scaled_mm(A8, B8.T, scale_a=s_A, scale_b=s_B,
                                        out_dtype=torch.bfloat16)
    store = {'fp32': torch.float32, 'tf32': torch.float32,
             'bf16': torch.bfloat16, 'fp16': torch.float16}[dtype]
    A, Bt = A.to(store), B.to(store).T
    return lambda: A @ Bt

def hw_matmul_bench(m, n, k, dtype, repeats=20, graph=True):  #@save
    """TFLOP/s of hw_matmul_op (2 m n k FLOPs per product): tflops, ms per
    product (median of repeats) and its MAD."""
    fn = hw_matmul_op(m, n, k, dtype)
    with d2l._hw_tf32(dtype == 'tf32'):
        ms, mad = d2l._hw_stats(d2l._hw_time(fn, repeats, graph))
    return dict(tflops=2 * m * n * k / ms / 1e9, ms=ms, mad=mad)
```

<!-- Placeholder (prose): d2l.hw_dtype_ladder runs hw_matmul_bench in each
format after a one-second warm-up and records the SM clock and board power;
the table shows the spread between the three processes, the noise any gap
between two formats must exceed. A fraction above 100% is read against the
SM clock column: the datasheet peaks assume the boost clock. -->

```{.python .input #accelerator-design-dtype-ladder}
%%tab pytorch
dtypes = d2l.hw_dtype_ladder(n=8192,
                             formats=('fp32', 'tf32', 'bf16', 'fp16', 'fp8'))
```

<!-- Placeholder (prose): C3's fp16 gap (fix F9, measured with torch.profiler
and clock locks): fp16 is about 8% below bf16 in the table (89% vs 97% of the
peak). cuBLAS picks a different kernel for fp16 (m16n8k16, 128x64 tile) than
for bf16 (m16n8k8, 64x128) at 8192^3; at the same locked clock the fp16 kernel
does up to 2% less work per cycle and draws 11-17% more power. At default
clocks fp16 runs at a lower SM clock (2.49 vs 2.58 GHz in the table) and does
about 4% less work per measured cycle. The clock and the locked-clock kernel
difference explain about 5 of the 8 points; why the per-cycle gap widens at
default clocks is not measured. Neither format sat at the 450 W limit (406 and
411 W in the table). The reduced-precision-reduction setting does not change
the kernel. -->

<!-- Placeholder (prose): ruling D3 in one or two sentences (fix pass F9,
F11): fp8 is native on Ada (the 330 TFLOP/s row above, about twice bf16) and
on every newer GPU; NVFP4 and MXFP4 are simulated on this Ada card and run
natively on Blackwell (sm_100, sm_103, sm_120, sm_121), both through chapter
21's implementation (cite :numref:`sec_lm_low_precision`: its simulator and
its native product with the capability check and probe GEMM); this section
re-implements none of the packing or scales. The 4-bit cell
(accelerator-design-fp4, d2l.hw_fp4_matmul) goes here once chapter 21's
native FP4 product lands (its fix F40; held, not on the ch21 branch as of
2026-11-27). On this box it prints two status lines and measures nothing
(no FP4 tensor cores on sm_89; simulated here, timed natively on sm_100 /
sm_120); on a Blackwell card it times the native product against the chip
row's FP4 peak, a live rate never quoted in the text. Say that the native
branch is written and unit-tested with the hardware detection mocked and
cannot run here. The summary table gains an FP4 row only for chips with an
FP4 datasheet figure (the 4090 has none). -->

### Shared Memory and Latency

<!-- Placeholder (prose): two Triton kernels (library): every SM re-reads a
small L1-resident block (on Ada, L1 and shared memory are one 128 KB SRAM per
SM, 128 bytes per clock; Triton has no explicit shared-memory load, so these
are L1 loads, and a simple load loop, not a tuned kernel), and one thread
chases a random cycle of pointers, so each load waits for the last: the
latency of L1, L2 and DRAM by working set. ncu counters need root on this
box (an exercise). -->

```{.python .input #accelerator-design-smem-latency}
%%tab pytorch
smem = d2l.hw_smem_bench()
lat = d2l.hw_latency_bench(sizes=2**np.arange(12, 31))
d2l.plot_latency(lat)
```

<!-- Placeholder (prose): fix pass F10: four cards of one model are not one
card. At the same SM and memory clocks the box's GPU 3 (the card of the
committed run) takes 98 ns per dependent L2 load and 220 ns per DRAM load,
GPUs 1 and 2 about 108 and 231 ns; L1 is 16-17 ns on all three. A live line
run on a card other than the committed run's says "another RTX 4090". The
live check re-runs the committed sweep in its order (the first sweep in a
fresh process reads about 8% high while the clocks ramp). -->

### Datasheet against Measured

<!-- Placeholder (prose): the summary table; claims C1 (DRAM read within 15%
of 1,008 GB/s), C2 (L2 a multiple of DRAM), C3 (BF16 fraction, FP8 about
twice BF16); TF32 is a compute format with 32-bit storage; the L1 row's
reference is a computed bound, not a datasheet figure; clocks are not locked
on this box, so a gap between two series counts only above the spread
between processes, and never without the clock and power record. -->

```{.python .input #accelerator-design-datasheet-vs-measured}
%%tab pytorch
d2l.datasheet_vs_measured(specs.loc['RTX 4090'], ladder, dtypes, smem)
```

### The A100 Twin
:label:`subsec_lm_hw_a100_twin`

<!-- Placeholder (prose): what the T2 campaign (C-T2, one A100 PCIe 40 GB)
measures and why it is the part's T2 chip; no FP8 tensor cores. Claim C5
(the A100 reproduces its datasheet HBM within 15%) is pending: part of the
A100 campaign. -->
<!-- campaign: lm-hw-a100-microbench-v1 -->

```{.python .input #accelerator-design-a100-twin}
%%tab pytorch
d2l.datasheet_vs_measured(specs.loc['A100 PCIe 40 GB'],
                          d2l.hw_campaign_log('a100-microbench'))
```

<!-- Placeholder (prose): close: forward to sec_lm_kernels (how a kernel
turns the budget into FLOP/s) and sec_lm_workload_hardware. -->

## Summary

<!-- Placeholder (prose): 3-6 sentences. -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.1.md part 10 (MoE
sweep floor; Cerebras chips for Qwen3-235B-A22B; half-peak message size of a
device-to-device copy; bidirectional aggregates; a new chip row; Nsight
Compute L2 hit rate where counters are allowed). -->
