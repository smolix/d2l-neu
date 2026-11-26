# Kernels, Tiles and Fusion
:label:`sec_lm_kernels`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.2 Kernels, Tiles and Fusion · T1 (T2 twin)").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A kernel is a schedule of tiles through the memory hierarchy, and its speed is set by how many FLOPs each byte that enters on-chip memory serves before it leaves.
**Opening:** The reader saw `torch.compile` fuse elementwise chains and cut traffic in `sec_compilation`, and read in `sec_attention-at-scale` that FlashAttention is the blockwise online softmax fitted to the memory hierarchy, with the remark that the book does not write it as a kernel. One concrete example: the reference model's attention at sequence 4,096 on the reader's GPU, run three ways, the chunked PyTorch version of `sec_attention-at-scale`, the same function under `torch.compile`, and the library's fused kernel; the three differ by an order of magnitude (to be measured). The open question: what does it take to write the fast version by hand, and what stops it short of the roofline.
**Teaches:**
- The programmer's view of 5.1's area budget: a thread block owns an output tile, its operands pass through shared memory and registers, and the tile is the unit of reuse; occupancy, waves and launch cost in one paragraph each (restating `subsec_hw-gpu`'s SM and tensor-core facts in one sentence).
- Tiled matmul: a T_m x T_n output tile with a K-slab of depth T_k loads b T_k (T_m + T_n) bytes for 2 T_m T_n T_k FLOPs, so its intensity is 2 T_m T_n / (b (T_m + T_n)); registers cap the tile; the ridge therefore needs reuse across tiles in L2 or by multicast; wave quantization (tiles against SMs) explains the saw-tooth of achieved FLOP/s against M; the GEMM list of a transformer (QKV, output, gate-up, down, head) at the token counts of training and of decode.
- Triton: program ids, block pointers, `tl.load`, `tl.dot`, `tl.store`, `num_warps`, `num_stages`, autotuning; a fused RMSNorm as the first kernel (about fifteen lines), then the attention forward: one program per query tile, a loop over key-value tiles carrying the running maximum, sum and accumulator, the causal skip of fully masked tiles, the final normalization.
- Where the compiler stops: Inductor fuses elementwise and reduction chains and emits Triton, and `max-autotune` picks tiles for a single matmul, but no compiler re-tiles a matmul chain across a softmax; FlashAttention-2's query-tile partition, and FlashAttention-3's warp specialization, asynchrony and FP8 on Hopper, in one paragraph each (the A100 has neither TMA nor FP8, so these stay prose); fused training kernels beyond attention (chunked cross-entropy that never materializes the logits, fused RMSNorm and RoPE) as the same idea applied to the rest of the step.
**Derivation:** The roof condition for a tile and the bytes attention moves. With R = P/beta and b bytes per element, a square tile reaches the compute roof only if T >= b R, and its FP32 accumulators then occupy 4 b^2 R^2 bytes, so the on-chip memory a roofline-bound tile needs grows with the square of the ridge point (blueprint ch05 5.1 D1). Smallest numeric case: the A100 PCIe 40 GB row gives R = 312 / 1.555 = 201 FLOP per byte for BF16, so T >= 402 and the accumulator tile is 4 x 402^2 = 646 KB, above the 256 KB register file of one SM; a single tile cannot reach the roof from HBM alone, and the kernel relies on L2 reuse across tiles. For attention with query tile B_q: every query tile reads all keys and values once, so the kernel reads (N / B_q) x 2 N d b bytes of K and V for 4 N^2 d FLOPs, an HBM intensity of 2 B_q / b, which is 128 FLOP per byte at B_q = 128 and b = 2, below the A100 ridge; the kernel is compute-bound only because one head's K and V (4,096 x 128 x 2 x 2 bytes = 2 MB) stay in the 40 MB L2, which the cell checks by counting bytes.
**Notebook:**
1. Question: does a hand-tiled Triton attention kernel for the reference model's shapes reach the same fraction of the measured matmul peak as the library kernel, and do tile size and warps move it in the direction the byte count predicts?
2. Model and data: random Q, K, V in BF16 at the reference model's shape (12 heads, head dimension 64, sequence 4,096, batch 8) and at Qwen3-8B's (32 query heads, 8 KV heads, head dimension 128); `d2l.gemm_shapes(cfg, tokens)` lists the matmul shapes of both models.
3. The shape sweep: achieved BF16 TFLOP/s of `torch.matmul` over the listed shapes at M from 512 to 16,384 tokens with `d2l.hw_matmul_bench` (5.1), on the reader's GPU and on the A100 twin; wave quantization is visible at multiples of the SM count.
4. The warm-up kernel: a Triton RMSNorm checked against `d2l.RMSNorm` to 1e-5 in FP32 and timed against the eager and compiled versions.
5. The attention kernel: `d2l.flash_attention_triton(q, k, v, causal=True, block_q=128, block_k=64, num_warps=4)`, checked against `torch.nn.functional.scaled_dot_product_attention` with an FP32 reference: maximum absolute error at most 2e-2 times the reference's maximum magnitude in BF16 (state the band; Triton's summation order differs by tile size, so the comparison is against FP32, not bitwise).
6. The sweep: TFLOP/s against `block_q` in {64, 128}, `block_k` in {32, 64, 128} and `num_warps` in {4, 8}, against the SDPA flash backend and the compiled chunked attention; the T2 twin commits the A100 series to `lm-hw-a100-microbench`.
7. Claim that survives noise: the book's kernel runs within a factor of two of the library kernel and at least ten times faster than the chunked PyTorch version; raising `block_q` lowers the K and V re-read count and raises throughput until registers spill, and the spill point is reported as measured, not predicted.
8. T1 is the native tier (Triton runs on Ampere and Ada within 8 GiB); the T2 twin only supplies the A100 numbers that 5.3's calibration uses.
**Figures:** `fig_lm_hw_tile_reuse` (one output tile, the two operand slabs it loads, bytes against FLOPs); `fig_lm_hw_flash_tiles` (a query tile sweeping the key-value tiles with the running maximum and sum, the causal skip shaded); `fig_lm_hw_shape_sweep` (plot: achieved TFLOP/s against M for the two models' GEMM shapes, SM-count multiples marked).
**Restate / forward:** `subsec_hw-gpu` (SMs, tensor cores, shared memory in one sentence); `eq_matmul_intensity` and `subsec_perf-sweep` (the square sweep is not repeated); `eq_online-softmax` (the three running statistics, not re-derived); `sec_compilation` (fusion cuts traffic, graph replay removes launch cost: restated, not re-measured); `sec_performance-further-reading` "Writing Fused Kernels" (its paragraph is this section's starting point). Forward: 12.2 (the split-KV decode kernel reuses this vocabulary).
**Exercises:** 7 to 9. Derive the K and V re-read count and the L2 condition for Qwen3-8B at sequence 32,768 (derive); derive the tile intensity for INT8 operands and the register bytes at the ridge (derive); measure the saw-tooth at M = 108 x 128 against 109 x 128 on an A100 with 108 SMs, or the reader's SM count (measure); measure the backward kernel from the Triton tutorial and the ratio of backward to forward time (measure); check that `torch.compile(mode='max-autotune')` on the chunked attention does not reach the fused kernel and read the generated Triton to say why (check); open: fuse the RMSNorm into the QKV projection's prologue and measure the saving against the two-kernel version.
**Writer notes:** This section owns the Triton introduction for the part; 12.2 restates program ids and block pointers in one paragraph and writes the decode kernel, and 7.x, 8.x and 11.x call Triton kernels only through `d2l` names. Triton is the version shipped with the pinned `torch`; the kernel cell is the one cell in this chapter allowed over thirty lines, with the reason stated in the text (as 12.2's kernel is in its chapter). Write the book's own kernel: causal only, no dropout, no backward in the main text (backward is an exercise and an optional A6a task); the Triton tutorial's fused-attention kernel is the test oracle for speed, never copied. Tolerance against an FP32 reference, not against SDPA bitwise. `torch.compile` on the chunked attention of `sec_attention-at-scale` is the compiler baseline arm; name what Inductor fused and what it did not from the generated code. Do not re-teach `torch.compile`, CUDA graphs (`sec_compilation`), the roofline (`sec_perf_model`), the online softmax (`sec_attention-at-scale`), or tensor-core formats (`subsec_hw-gpu`). Ada readers may see FP8 tensor cores; keep FP8 out (4.3 owns formats). Hardware fallback: if Triton fails to compile on the reader's driver, the cell prints the error and loads the committed A100 series; the T1 twin on the build box runs the full sweep within 8 GiB. The shape-sweep entries extend `lm-hw-a100-microbench` (5.1 keeps the bandwidth entries). Exports: `d2l.flash_attention_triton`, `d2l.triton_rmsnorm`, `d2l.gemm_shapes`. Blueprint ch05 5.1 D1 (tile intensity) and its numerical check are taught here; the register-file sizes come from the whitepapers with source strings. The Hopper-only paragraph names TMA and warp specialization (three sentences); the A6a kernel task is the attention forward (the CS336 pattern), with the chunked cross-entropy as the alternative.
**Sources:**
- Papers: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (the tiled schedule and the HBM-traffic bound); [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (query-tile partition, the kernel the book writes); [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (Hopper refinements, prose only); [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989) (fused RMSNorm, RoPE and chunked cross-entropy for training); [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174) (L2 and shared-memory numbers the derivation uses); Tillet, Kung and Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MAPL 2019; cited by title, the book's existing bibliography key).
- Code and documentation: [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) (test oracle); [flash-attention repository](https://github.com/Dao-AILab/flash-attention) (the library kernel); [Liger-Kernel repository](https://github.com/linkedin/Liger-Kernel) (training kernels in Triton); [CUTLASS](https://github.com/NVIDIA/cutlass) (tile hierarchy vocabulary); NVIDIA, [Matrix multiplication performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) (tile and wave quantization) and [CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) (thread blocks, shared memory); [Nsight Compute](https://docs.nvidia.com/nsight-compute/) (optional byte counting).
- Courses, tutorials and blogs: [CS336 assignment 2](https://github.com/stanford-cs336/assignment2-systems) (the Triton FlashAttention-2 assignment); [GPU MODE lectures](https://github.com/gpu-mode/lectures) (Triton and profiling lectures); [CMU 11-868 LLM Systems](https://llmsystem.github.io/llmsystem2026spring/) (GPU programming and FlashAttention lectures); [Modal GPU Glossary](https://modal.com/gpu-glossary).
-->

<!-- Placeholder (prose): opening. The reference model's attention at
sequence 4,096 on this GPU, three ways: the chunked PyTorch attention of
sec_attention-at-scale, the same function under torch.compile, and the fused
(flash) kernel behind SDPA. They differ by about two orders of magnitude.
The open question: what it takes to write the fast one by hand, and what
stops it short of the roofline. -->

```{.python .input #kernels-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #kernels-extras}
%%tab pytorch
import pandas as pd
import triton
import triton.language as tl
```

<!-- Placeholder (prose): the two attention shapes of the section: the
reference model (12 query heads sharing 4 key-value heads, head dimension
64, batch 8) and Qwen3-8B (32 query and 8 key-value heads, head dimension
128, batch 1), both at 4,096 tokens. A fused kernel re-reads one head's keys
and values for every query tile, so it matters whether they fit in L2: the
table counts their bytes against this GPU's L2. -->

```{.python .input #kernels-shapes}
%%tab pytorch
specs = d2l.chip_specs()
R = d2l.r2_config('r2').arch  # the reference model's architecture
cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
shapes = {'reference': (R, 8), 'Qwen3-8B': (cfg_8b, 1)}  # (config, batch)
seq = 4096  # tokens per sequence
d2l.attention_shapes(shapes, seq,
                     l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size)
```

<!-- Placeholder (prose): chunked_attention of sec_attention-at-scale, in
batched form: query heads of shape (B, H, N, d), keys and values with fewer
heads (GQA). The query heads that share a key-value head are stacked as rows
against it, so nothing is copied. Scores and running statistics in FP32,
products in BF16. -->

```{.python .input #kernels-chunked-attention}
%%tab pytorch
def chunked_causal_attention(q, k, v, chunk=512):  #@save
    """Exact causal attention of q (B, H, N, d) against k and v
    (B, H_kv, N, d), one chunk of keys at a time with the online softmax;
    the H // H_kv query heads sharing a key-value head are stacked as rows
    against it (GQA without copying k and v)."""
    B, H, N, d = q.shape
    H_kv = k.shape[1]
    Q = q.reshape(B, H_kv, H // H_kv * N, d)  # a group's queries as rows
    pos_q = torch.arange(N, device=q.device).repeat(H // H_kv)[:, None]
    lowest = torch.finfo(torch.float32).min
    m = torch.full((B, H_kv, Q.shape[2], 1), lowest, device=q.device)
    s = torch.zeros_like(m)
    O = torch.zeros(Q.shape, device=q.device)
    for start in range(0, N, chunk):
        Kc, Vc = k[:, :, start:start + chunk], v[:, :, start:start + chunk]
        pos_k = torch.arange(start, start + Kc.shape[2], device=q.device)
        a = (Q @ Kc.transpose(-2, -1)).float() / math.sqrt(d)
        a.masked_fill_(pos_k > pos_q, lowest)
        m_new = torch.maximum(m, a.amax(-1, keepdim=True))
        scale = torch.exp(m - m_new)  # rescale the past
        P = torch.exp(a - m_new)
        s = s * scale + P.sum(-1, keepdim=True)
        O = O * scale + (P.to(Vc.dtype) @ Vc).float()
        m = m_new
    return (O / s).to(q.dtype).view(B, H, N, d)
```

<!-- Placeholder (prose): the three arms on the reference shape. SDPA runs
only its flash backend here (d2l.sdpa_causal refuses a fallback to the math
kernel). TFLOP/s count the useful FLOPs of causal attention, 2 B H N^2 d_h
(half of the two products 4 B H N^2 d_h); the kernels column counts the
launches per call. -->

```{.python .input #kernels-three-ways}
%%tab pytorch
q, k, v = d2l.attention_inputs(R, batch=8, seq=seq)
ways = d2l.time_attention({
    'chunked': d2l.chunked_causal_attention,
    'compiled': torch.compile(d2l.chunked_causal_attention),
    'SDPA': d2l.sdpa_causal}, q, k, v)
```

<!-- Placeholder (prose): the open question, and the plan of the section:
the tile as the unit of reuse, a first Triton kernel, the attention kernel,
and where the compiler stops. -->

## Tiles and Reuse
:label:`subsec_lm_hw_tiles`

<!-- Placeholder (prose): one-sentence restatement of subsec_hw-gpu (SMs,
tensor cores, shared memory). A thread block owns an output tile; its
operands pass through shared memory and registers, and the tile is the unit
of reuse. Occupancy (one paragraph): blocks per SM limited by registers and
shared memory. Waves (one paragraph): tiles are dealt to SMs in rounds; a
last, partial round leaves SMs idle. Launch cost (one paragraph): a few
microseconds per kernel, which a fused kernel pays once. -->

### The Roof Condition for a Tile
:label:`subsec_lm_hw_tile_roof`

<!-- Placeholder (prose): derivation D1. A T_m x T_n output tile with a
K-slab of depth T_k loads b T_k (T_m + T_n) bytes for 2 T_m T_n T_k FLOPs:
I(T_m, T_n) = 2 T_m T_n / (b (T_m + T_n)) (eq_lm_hw_tile_roof). A square
tile reaches the ridge R = P / beta only if T >= b R, and its FP32
accumulators then take 4 T^2 = 4 b^2 R^2 bytes. Figure fig_lm_hw_tile_reuse
(generator tile-reuse): one output tile, two operand slabs, bytes against
FLOPs. -->

```{.python .input #kernels-tile-roof-def}
%%tab pytorch
def tile_intensity(Tm, Tn, b=2):  #@save
    """FLOPs per byte loaded of a Tm x Tn output tile: each K-slab of depth
    T_k brings b T_k (Tm + Tn) bytes for 2 Tm Tn T_k FLOPs."""
    return 2 * Tm * Tn / (b * (Tm + Tn))

def tile_roof(rows, b=2, rf_kib=256):  #@save
    """For each chip (a chip_specs row): the ridge R = P / beta (FLOP per
    byte, dense 16-bit), the smallest square tile T = ceil(b R) that
    reaches it from DRAM, its FP32 accumulators 4 T^2 (KiB), and the
    register file of one SM (KiB)."""
    R = rows.p16_tflops / rows.mem_bw_tbs  # TFLOP/s over TB/s
    T = np.ceil(b * R).astype(int)
    return pd.DataFrame({'ridge': R, 'T': T, 'acc_kib': 4 * T**2 / 2**10,
                         'rf_kib': rf_kib})
```

<!-- Placeholder (prose): the square tile's intensity is T / b; the two
chips need tiles whose accumulators exceed an SM's register file, so a
kernel reaches the roof only with reuse across tiles in L2. -->

```{.python .input #kernels-tile-roof}
%%tab pytorch
for T in (64, 128, 256, 512):
    print(f'{T} x {T} tile: {d2l.tile_intensity(T, T):.0f} FLOP per byte')
d2l.show_table(d2l.tile_roof(specs.loc[['RTX 4090', 'A100 PCIe 40 GB']]),
               digits={'ridge': 0}, headers={
                   'ridge': 'ridge FLOP/B', 'T': 'T >= bR',
                   'acc_kib': 'accumulator KiB',
                   'rf_kib': 'register KiB/SM'})
```

### The Shapes a Transformer Multiplies
:label:`subsec_lm_hw_gemm_shapes`

<!-- Placeholder (prose): the weight products of one decoder layer (QKV,
the attention output, gate-up, down) and the head, each an (M x K) @ (K x N)
product with M the number of tokens; training and prefill run them at
M = thousands of tokens, decode at M = the batch. -->

```{.python .input #kernels-gemm-shapes-def}
%%tab pytorch
def gemm_shapes(cfg, tokens):  #@save
    """The weight products of one forward pass over `tokens` tokens of a
    decoder (an Arch or a Transformers config): per product, the shape
    M x N x K of one layer's matmul and the GFLOP over all layers (the head
    once). The attention scores are not weight products."""
    L, d, H, H_kv, d_h, f, V, _ = d2l.arch_dims(cfg)  # _: tied or not
    products = [('QKV', (H + 2 * H_kv) * d_h, d, L), ('out', d, H * d_h, L),
                ('gate-up', 2 * f, d, L), ('down', d, f, L), ('head', V, d, 1)]
    return pd.DataFrame([
        dict(name=name, M=tokens, N=n, K=k, gflop=2 * tokens * n * k * c / 1e9)
        for name, n, k, c in products])
```

<!-- Placeholder (prose): the same products at M = 1 (decode at batch 1)
and M = 4,096 (prefill of one sequence): the intensity 2MNK / (b (MK + KN +
MN)) is about 1 FLOP per byte at M = 1 and above the ridge at 4,096. -->

```{.python .input #kernels-gemm-shapes}
%%tab pytorch
b = 2  # bytes per bf16 element
table = pd.concat({name: d2l.gemm_shapes(cfg, seq).set_index('name')
                   for name, (cfg, _) in shapes.items()})
for M in (1, seq):
    table[M] = 2 * M * table.N * table.K / (
        b * (M * table.K + table.K * table.N + M * table.N))
d2l.show_table(table[['N', 'K', 'gflop', 1, seq]], digits={1: 1, seq: 0},
               headers={'gflop': 'GFLOP*', 1: 'FLOP/B, M=1',
                        seq: f'FLOP/B, M={seq:,}'})
print(f'* all layers (the head once), M = {seq:,} tokens')
```

<!-- Placeholder (prose): the shape sweep: achieved BF16 TFLOP/s of each of
Qwen3-8B's weight products for M from 512 to 16,384 (hw_matmul_bench of
sec_lm_accelerator_design, each point after a short warm-up), against the
datasheet peak; the A100 twin's series joins it when campaign C-T2 has run.
Read the lines under the plot: most points sit at 160-177 TFLOP/s, above the
dashed datasheet line, because the card runs its SM clock above the 2.52 GHz
boost clock the datasheet assumes (the clock line gives the sweep's range).
One point drops by a third: the head product (N = 151,936) at M = 4,096
(about 109 TFLOP/s, as long per call as M = 6,144). It is deterministic, and
the kernel lines name the cause: cuBLAS's heuristic picked the same 128 x 128
tile without the multistage pipeline (no "stages_32x1") it uses at
M = 2,048. Lesson: a library's per-shape kernel choice can miss by a third;
measure the shapes the model actually runs. -->

```{.python .input #kernels-shape-sweep}
%%tab pytorch
sweep = d2l.hw_shape_sweep(cfg_8b, M=2**np.arange(9, 15))
d2l.plot_shape_sweep(sweep, specs,
                     reference=d2l.hw_campaign_log('a100-microbench'))
```

<!-- Placeholder (prose): wave quantization. With M = K = 4,096 and N in
steps of 64, the tile count ceil(M / T_m) ceil(N / T_n) crosses a multiple
of the 128 SMs every n_SM T_n T_m / M columns; just past each crossing a new,
nearly empty wave starts and TFLOP/s drops. Two arms: cuBLAS, which picks a
kernel (tile, split) per shape and so stays nearly flat, and a matmul with one
fixed 128 x 128 tile (d2l.tile_matmul, a dozen lines of Triton in the library;
the reader meets Triton in the next section), whose saw-tooth has the
predicted period of 512 columns, deep at two to three waves (~110 against
~160 TFLOP/s at N = 1,088 on the 4090) and shallow at sixteen. The lines
below give each arm's range, the measured period against
n_SM T_m T_n / M, and the tiles the cuBLAS kernels encode (three on the
4090). Lesson: wave quantization is real; libraries hide it by choosing a
tile per shape, a hand-written kernel must handle it (Exercise). -->

```{.python .input #kernels-sawtooth}
%%tab pytorch
wave = d2l.hw_wave_sweep(M=4096, K=4096, N=range(1024, 8193, 64))
d2l.plot_wave(wave)
```

## Writing Kernels in Triton
:label:`subsec_lm_hw_triton`

<!-- Placeholder (prose): Triton's model. A kernel is written for one
program (a thread block); tl.program_id names its tile; tl.load and
tl.store move whole blocks between memory and registers (pointer arithmetic
or block pointers, tl.make_block_ptr); tl.dot multiplies blocks on the
tensor cores; num_warps sets the threads per program and num_stages the
software pipelining of loads; triton.autotune picks among configurations.
The first kernel: RMSNorm, one program per row, one read and one write of
the row. -->

```{.python .input #kernels-rmsnorm-kernel}
%%tab pytorch
@triton.jit  #@save
def _rmsnorm_kernel(X, W, Y, d, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    x = tl.load(X + row * d + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / d + eps)
    tl.store(Y + row * d + cols, (x * r * w).to(Y.dtype.element_ty),
             mask=mask)

def triton_rmsnorm(x, weight, eps=1e-6):  #@save
    """RMSNorm over the last dimension of x in one Triton kernel (one
    program per row, FP32 accumulation)."""
    X = x.reshape(-1, x.shape[-1]).contiguous()
    Y = torch.empty_like(X)
    _rmsnorm_kernel[(X.shape[0],)](X, weight, Y, X.shape[1], eps,
                                   BLOCK=triton.next_power_of_2(X.shape[1]))
    return Y.view_as(x)
```

<!-- Placeholder (prose): checked against PyTorch's RMSNorm in FP32, then
timed against the formula written out in PyTorch (one kernel per operation),
the same formula under torch.compile, and torch's own fused kernel
(F.rms_norm); one read and one write of x, so GB/s is 2 |x| / t, compared
with the DRAM read rate measured in sec_lm_accelerator_design. -->

<!-- Placeholder (prose): the baseline, the formula as written in PyTorch:
each operation is its own kernel and reads or writes x. -->

```{.python .input #kernels-rmsnorm-formula}
%%tab pytorch
def rms_norm_formula(x, weight, eps=1e-6):  #@save
    """RMSNorm as written: x / sqrt(mean(x^2) + eps) * weight."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

```{.python .input #kernels-rmsnorm-check}
%%tab pytorch
d = 4096
x = torch.randn(8 * 4096, d, device='cuda')  # 8 sequences of 4,096 tokens
w = torch.randn(d, device='cuda')
d2l.check_close(d2l.triton_rmsnorm(x, w), F.rms_norm(x, (d,), w, eps=1e-6),
                atol=1e-5)
norms = d2l.time_fns({
    'Triton': d2l.triton_rmsnorm, 'formula': d2l.rms_norm_formula,
    'formula, compiled': torch.compile(d2l.rms_norm_formula),
    'F.rms_norm': lambda x, w: F.rms_norm(x, (d,), w, eps=1e-6)},
    x, w, nbytes=2 * x.nbytes)
```

### A Fused Attention Kernel
:label:`subsec_lm_hw_flash_kernel`

<!-- Placeholder (prose): the FlashAttention-2 forward in Triton. One
program per query tile of one head; a loop over key-value tiles carries the
running maximum m, sum s and accumulator of eq_online-softmax (restated in
one sentence); exp2 with the scale folded in log2 e; tiles entirely below the
diagonal need no mask, the diagonal tiles do, tiles above it are skipped
(the causal skip); the final normalization divides by s once. GQA: query
head h reads key-value head h // (H / H_kv). This is the one cell of the
chapter over thirty lines: the kernel is read as a whole (as the decode
kernel of sec_lm_decode_kernels). The bytes it moves: each query tile reads
all earlier keys and values once, an HBM intensity of 2 B_q / b = 128 FLOP
per byte at B_q = 128, below the 4090's ridge of 164; it runs near the roof
only because a head's keys and values (1 MiB at the reference shape, 2 MiB
for Qwen3-8B, cell above) stay in the 72 MiB L2. Figure
fig_lm_hw_flash_tiles (generator flash-tiles). -->

```{.python .input #kernels-flash-kernel}
%%tab pytorch
@triton.jit  #@save
def _attend(q, K_tile, V_tile, m, s, acc, start, rows, N, scale,
            MASK: tl.constexpr, CAUSAL: tl.constexpr, BLOCK_K: tl.constexpr):
    """One key-value tile of the online softmax of a query tile."""
    k = tl.load(K_tile, boundary_check=(0,))
    a = tl.dot(q, tl.trans(k)) * scale  # scores in log2 units
    if MASK:  # a tile on the diagonal, or past the last key
        cols = start + tl.arange(0, BLOCK_K)
        keep = cols[None, :] < N
        if CAUSAL:
            keep = keep & (rows[:, None] >= cols[None, :])
        a = tl.where(keep, a, float('-inf'))
    m_new = tl.maximum(m, tl.max(a, axis=1))
    alpha = tl.exp2(m - m_new)  # rescale the past
    p = tl.exp2(a - m_new[:, None])
    v = tl.load(V_tile, boundary_check=(0,))
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, s * alpha + tl.sum(p, axis=1), acc

@triton.jit  #@save
def _flash_fwd_kernel(Q, K, V, O, sm_scale, N, H, GROUP: tl.constexpr,
                      D: tl.constexpr, BLOCK_Q: tl.constexpr,
                      BLOCK_K: tl.constexpr, CAUSAL: tl.constexpr):
    i, bh = tl.program_id(0), tl.program_id(1)  # query tile, (batch, head)
    b, h = bh // H, bh % H
    q_base = bh * N * D
    kv_base = (b * (H // GROUP) + h // GROUP) * N * D  # the shared KV head
    Q_tile = tl.make_block_ptr(Q + q_base, (N, D), (D, 1), (i * BLOCK_Q, 0),
                               (BLOCK_Q, D), (1, 0))
    K_tile = tl.make_block_ptr(K + kv_base, (N, D), (D, 1), (0, 0),
                               (BLOCK_K, D), (1, 0))
    V_tile = tl.make_block_ptr(V + kv_base, (N, D), (D, 1), (0, 0),
                               (BLOCK_K, D), (1, 0))
    q = tl.load(Q_tile, boundary_check=(0,))
    m = tl.full([BLOCK_Q], float('-inf'), tl.float32)
    s = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)
    scale = sm_scale * 1.4426950408889634  # e^x = 2^(x log2 e)
    rows = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
    if CAUSAL:  # full tiles below the diagonal, then the diagonal tiles
        lo = i * BLOCK_Q // BLOCK_K * BLOCK_K
        hi = tl.minimum((i + 1) * BLOCK_Q, N)  # later keys: skipped
    else:
        lo, hi = N // BLOCK_K * BLOCK_K, N
    for start in range(0, lo, BLOCK_K):
        m, s, acc = _attend(q, K_tile, V_tile, m, s, acc, start, rows, N,
                            scale, False, CAUSAL, BLOCK_K)
        K_tile = tl.advance(K_tile, (BLOCK_K, 0))
        V_tile = tl.advance(V_tile, (BLOCK_K, 0))
    for start in range(lo, hi, BLOCK_K):
        m, s, acc = _attend(q, K_tile, V_tile, m, s, acc, start, rows, N,
                            scale, True, CAUSAL, BLOCK_K)
        K_tile = tl.advance(K_tile, (BLOCK_K, 0))
        V_tile = tl.advance(V_tile, (BLOCK_K, 0))
    O_tile = tl.make_block_ptr(O + q_base, (N, D), (D, 1), (i * BLOCK_Q, 0),
                               (BLOCK_Q, D), (1, 0))
    tl.store(O_tile, (acc / s[:, None]).to(O.dtype.element_ty),
             boundary_check=(0,))

def flash_attention_triton(q, k, v, causal=True, block_q=128,  #@save
                           block_k=64, num_warps=4, num_stages=2):
    """Attention forward of q (B, H, N, d) against k and v (B, H_kv, N, d)
    in one Triton kernel: one program per query tile and head, the online
    softmax over key-value tiles; no dropout, no backward. Offsets are
    32-bit: each tensor holds fewer than 2^31 elements."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, block_q), B * H)
    _flash_fwd_kernel[grid](q, k, v, o, 1 / math.sqrt(D), N, H,
                            GROUP=H // k.shape[1], D=D, BLOCK_Q=block_q,
                            BLOCK_K=block_k, CAUSAL=causal,
                            num_warps=num_warps, num_stages=num_stages)
    return o
```

<!-- Placeholder (prose): correctness against an FP32 reference (computed in
query chunks of 1,024), not bitwise against SDPA: the kernel sums in a
different order for every tile size. The band: the largest error at most
2e-2 of the largest output magnitude, in BF16. -->

```{.python .input #kernels-flash-check}
%%tab pytorch
for name, shape in shapes.items():
    d2l.attention_error(d2l.flash_attention_triton, shape, seq, name=name)
```

### Tile Size, Warps and the Library Kernel
:label:`subsec_lm_hw_kernel_sweep`

<!-- Placeholder (prose): the sweep: block_q in {64, 128}, block_k in {32,
64, 128}, num_warps in {4, 8} (num_stages 2) at both shapes, against SDPA's
flash kernel and the compiled chunked attention; the table keeps the three
best and the worst configuration per shape. Claims K1 (within 2x of SDPA)
and K2 (at least 10x the eager chunked version of the opening). A
configuration that needs more shared memory than an SM offers (99 KiB per
block on Ada) would not compile and a line under the table would name it;
on the 4090 all twelve compile (num_stages 2), so no such line appears.
Measured on the 4090: the best configuration runs at 103% of SDPA at the
reference shape (64 x 64, 4 warps) and 92% at Qwen3-8B (128 x 128, 8 warps);
the 64-row tiles at 8 warps run at about half speed at both shapes. -->

```{.python .input #kernels-flash-sweep}
%%tab pytorch
flash = d2l.flash_sweep(shapes, seq, block_q=(64, 128), block_k=(32, 64, 128),
                        num_warps=(4, 8))
```

<!-- Placeholder (prose): registers. Triton reports, per compiled
configuration, the registers per thread (at most 255) and the 32-bit words
spilled to local memory when a tile's accumulators and operands do not fit;
claim K3, stated as measured (the fallback wording), not as predicted:
- d_h = 64 (reference): block_q 128 does not raise throughput; the 64-row
  tiles at 4 warps are best (149-151 TF/s), since K and V come from L2
  anyway. 128 x 32 at 4 warps is spill-free (254 registers) at 147.
- d_h = 128 (Qwen3-8B): block_q 128 gains (64 x 32 -> 128 x 32 at 4 warps:
  116 -> 132 TF/s) even with 44 spilled words.
- Large spills cost: 128 x 128 at 4 warps spills 140 words (reference) and
  248 (Qwen3-8B) and runs at 127 and 76 TF/s.
- The strongest effect is warps against tile height: 64-row tiles at 8
  warps run at about half speed at both shapes (68-77 and 64-99 TF/s).
Do not write "the largest spill-free block_q at 4 warps was 64": 128 x 32 at
4 warps has no spills at the reference shape. -->

```{.python .input #kernels-spill}
%%tab pytorch
d2l.triton_spills(d2l.flash_attention_triton, flash)
```

## Where the Compiler Stops
:label:`subsec_lm_hw_compiler_limits`

<!-- Placeholder (prose): Inductor fuses elementwise and reduction chains
and emits Triton for them; max-autotune picks the tiles of a single matmul;
nothing re-tiles a chain of matmuls across a softmax. The list below is what
torch.compile made of the chunked attention (claim K4): library matmuls for
the two products of every chunk (8 QK^T GEMMs, 8 PV GEMMs), and Triton
kernels for the softmax arithmetic between them. Inductor fused the softmax
chain of all 8 chunks into one reduction, so it materializes every chunk's
scores (and P) in memory: 8 QK^T GEMMs, then one reduction, then 8 PV GEMMs.
That is also why the compiled arm peaks at about 7 GiB at the reference
shape (3.2 GB of BF16 scores plus 3.2 GB of P). -->

```{.python .input #kernels-inductor-kernels}
%%tab pytorch
d2l.inductor_kernels(torch.compile(d2l.chunked_causal_attention), q, k, v)
```

<!-- Placeholder (prose): FlashAttention-2's partition (one paragraph: query
tiles across programs, no inter-program reduction in the forward pass);
FlashAttention-3 on Hopper (one paragraph, prose only: TMA, warp
specialization, asynchrony, FP8; the A100 and the 4090 have no TMA);
fused training kernels beyond attention (one paragraph: chunked
cross-entropy that never materializes the logits, fused RMSNorm and RoPE;
Liger Kernel). -->

<!-- Placeholder (prose): the A100 twin (campaign C-T2) runs the same
sweeps on an A100 PCIe 40 GB (108 SMs, 40 MB L2); claim K6 (the same ratios
on the A100) is pending: part of the A100 campaign. -->
<!-- campaign: lm-hw-a100-microbench-v1 -->

```{.python .input #kernels-a100-twin}
%%tab pytorch
d2l.flash_sweep_table(d2l.hw_campaign_log('a100-microbench'))
```

<!-- Placeholder (prose): close: forward to sec_lm_workload_hardware (which
workloads sit on which side of the ridge) and to sec_lm_decode_kernels (the
split-KV decode kernel reuses this vocabulary). -->

## Summary

<!-- Placeholder (prose): 3-6 sentences. -->

## Exercises

<!-- Placeholder (prose): 7-9 exercises per S/spec/22.2.md part 8: K and V
re-reads and the L2 condition for Qwen3-8B at 32,768 tokens (derive); INT8
tile intensity and register bytes (derive); the saw-tooth at M = 128 x 128
against 129 x 128 on this GPU's 128 SMs, or 108 on an A100 (measure); the
backward kernel of the Triton tutorial and the backward/forward ratio
(measure); max-autotune on the chunked attention does not reach the fused
kernel, read the generated Triton (check); open: fuse RMSNorm into the QKV
prologue. -->
