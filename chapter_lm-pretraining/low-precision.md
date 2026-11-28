# Low-Precision Training
:label:`sec_lm_low_precision`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.3 Low-Precision Training · T1 (Ada or newer for native fp8)").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.4.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Low-precision training works when scales are chosen per block of values and the sensitive tensors stay in higher precision.
**Opening:** The reader trains in bf16 with fp32 master weights (`sec_memory_precision`) and knows the fp8 and block formats as layouts (`sec_numerics`, `sec_hardware`). An fp8 tensor core doubles matrix throughput, but one outlier in a tensor sets the scale for every other value in it, and a cell that quantizes a bf16 activation tensor with one scale shows most entries rounding to zero. What remains is to ask which tensors can take the format and which must not, and what the measured speedup is after everything else in the step is counted.
**Teaches:**
- Which GEMMs run in fp8 (the three per linear layer: forward, input gradient, weight gradient), what each operand's format is (E4M3 for weights and activations, E5M2 for the output gradient), and what stays in higher precision: fp32 master weights and optimizer moments, the loss, normalization, the embedding and head, attention probabilities, and in production recipes the sensitive projections.
- Scaling granularity: tensorwise, rowwise, blockwise (1x128 activation tiles and 128x128 weight blocks in the DeepSeek-V3 recipe, with fp32 promotion of partial sums every 128 elements because the tensor core's accumulator is narrower than fp32), delayed scaling from an amax history against current scaling, and the saturation and underflow statistics that tell them apart.
- Four-bit formats: MXFP4 (blocks of 32 with a power-of-two scale) and NVFP4 (blocks of 16 with an E4M3 scale and a tensor-level fp32 scale), stochastic rounding on gradients and a random Hadamard transform on the weight-gradient inputs; the production fact that Nemotron 3 was pretrained in NVFP4 for up to 25T tokens with the sensitive layers kept in higher precision, and adaptive block scaling as the refinement that narrows the gap to bf16.
- The cost model: fp8 beats bf16 on a GEMM only above a crossover size set by the cast traffic, and the end-to-end speedup is bounded by Amdahl's law with the GEMM share of the step (about 0.57 for the preset, so at most about 1.40x); datasheet ratios are not speedups.
- Where the formats run: fp8 native on Ada and Hopper, simulated elsewhere; NVFP4 and MXFP4 simulated on Ada and A100, native on Blackwell; the T1 tier line requires an Ada or newer card for the native arm.
**Derivation:** Block-scaling error against block size and in-block dynamic range. For the E2M1 grid (largest value 6, smallest normal 1, spacing 0.5 near the top) with one scale per block, the per-element normalized error in the relative regime lies between $2^{-2m}/48$ and $2^{-2m}/12$ (0.005 to 0.021 for one mantissa bit); one outlier of $\rho$ standard deviations in a block gives $\textrm{NMSE} \approx \rho^2 \Delta_s^2 / (12 Q^2) = \rho^2 / 1728$, saturating near $\rho \approx 24$; an E4M3 scale changes the effective range by at most 6%, a power-of-two scale is 1.3 to 2.5 times worse. Smallest case: a block of 16 Gaussian values with one $\rho = 8$ outlier has NMSE 0.0375 against the measured 0.0370. The notebook checks Gaussian blocks of size 8 to 1,024 (the model-to-measured ratio falls from 1.58 to 1.05) and the outlier formula at $\rho \in \{8, 16\}$ within 10%.
**Notebook:**
1. Question: does fp8 training match bf16 within seed noise on the same model and data, at what measured speedup, and what does simulated NVFP4 do to the loss?
2. Model and data: `d2l.PackedGPT` at the `fp8` preset (width 768, 8 blocks, 12 query and 4 key-value heads, SwiGLU 2,048, QK-norm on, 75.5M parameters, every width a multiple of 16 as fp8 GEMMs require) on the first 20M training tokens of the 100M shards, 1,221 steps of 16 sequences of 1,024, AdamW (0.9, 0.95), weight decay 0.1, peak learning rate 1e-3, WSD, z-loss 1e-4, fp32 master weights, `torch.compile` in every arm. Arms: bf16 seed 0 and seed 1 (the noise reference); torchao fp8 tensorwise (native on Ada); torchao fp8 rowwise (native if the card supports it, otherwise simulated and labelled); the from-scratch delayed-scaling fp8 with operand statistics; NVFP4 simulated on the MLP GEMMs (round-to-nearest forward, stochastic rounding on gradients, 16x16 weight blocks, Hadamard on the weight-gradient inputs; attention and head in bf16). 3.4's 20M proxy model is acceptable here only if its widths are multiples of 16; otherwise use this preset.
3. Check: `d2l.Fp8Linear(native=False)` against fp32 `F.linear` with relative Frobenius error within a factor two of the analytic value; native against simulated on Ada within `5e-3`; input and weight gradients against fp32 autograd within `1e-1`; the FP4 simulator returns on-grid values exactly, stochastic rounding is unbiased over $10^5$ draws within 1%, round-to-nearest is idempotent; the Hadamard transform is orthogonal to `1e-6`; a GEMM sweep at $16{,}384 \times n \times n$ for $n \in \{512, \ldots, 4096\}$ in a separate process locates the measured crossover.
4. Claim that survives noise: the fp8 tensorwise arm's final validation loss lies within the two bf16 seeds' spread; the end-to-end tokens-per-second ratio is reported as a median over 200 synchronized steps after warm-up with its interquartile range, below the Amdahl bound and possibly below one on this card; delayed scaling shows more saturation and underflow around warmup and decay than current scaling; the NVFP4 arm shows a visible loss gap at this scale (the NVFP4 paper reports parity at 12B parameters and 10T tokens, which the prose quotes as the paper's result).
5. Reused: `d2l.pretrain`, `d2l.r2_config('fp8')`, `d2l.PackedStream` (4.1, 4.2); `d2l.Benchmark` (`sec_perf_model`); `tab_gpu_specs` numbers for the 4090 (165 TFLOP/s bf16, 330 fp8, 1.0 TB/s). T1 is native on an Ada card; on an A100 the native arms are skipped and labelled.
**Figures:** `fig_lm_block_scaling` (a block of values, its shared scale, the grid and the outlier that crushes the rest); `fig_lm_fp8_loss_throughput` (plot: validation loss against tokens for the six arms, with a bar of tokens per second and the Amdahl bound); `fig_lm_lowp_stats` (plot: saturation and underflow fractions over training for delayed and current scaling).
**Restate / forward:** `sec_numerics` (E4M3 trades range for precision, E5M2 the reverse; fp32 master weights keep small updates); `sec_memory_precision` (the mixed-precision recipe and the 16 bytes per parameter); `sec_hardware` (microscaling blocks of 32 with a power-of-two scale, NVFP4 blocks of 16; hardware.md:322 defers the training recipes here, as do memory-precision.md:1248 and fast-transformer.md:1561). Forward: 11.5 (the same formats at inference, with quantization-aware distillation).
**Exercises:** 5 to 7. Derive: the error ratio of block sizes 16 and 128 for Gaussian values; which operands need E5M2 and why; the crossover size $n^*$ for forward-only GEMMs (half the training value). Measure: overflow under delayed scaling after a jump in activation scale (amax 6 against a history of 1 saturates at 448); the NVFP4 block-size and scale-format sweep on 5M tokens with two seeds. Open: a SwiGLU outlier miniature that destabilizes per-tensor scaling and the block size that fixes it.
**Writer notes:**
- The baseline is bf16 with `torch.compile`, never fp32 or tf32; time after compilation and after warm-up, with `cuda.synchronize`; a 2 to 4 minute run can be dominated by compile time, so measure steps 200 onward. GeForce cards run fp8 accumulation at a reduced rate, which is one reason the measured ratio may be below one; report it.
- fp8 storage is not fp8 training; fake quantization is not FP4 training; say both once. Delayed scaling silently clips after a jump, so log saturation. E5M2 forward and E4M3 gradients is the common mix-up.
- NVFP4 recipe details (which layers stay in higher precision, the Hadamard size, the stochastic-rounding placement) are copied from the NVFP4 pretraining paper and the Nemotron 3 report at authoring, with the sentence on kept layers quoting the report. DeepSeek-V3's ablation (relative loss error below a quarter of a percent) is quoted from its §3.3 after re-reading.
- Hardware fallbacks: torchao rowwise may not support sm89; the arm is guarded, falls back to simulated, and the torchao version is pinned. The A100 has no fp8 tensor cores: only the simulated arms and the T0 error cells run there. No H100 is in the tiers; an H100 sweep would be an appendix if one becomes available.
- `d2l.fake_quant_fp4` is the one FP4 simulator in the part; 7.5 and Chapter 11 import it (`01_interfaces.md`'s one-owner rule).
- Do not re-teach: the format tables and loss scaling (`sec_numerics`, `subsec_loss_scaling`), the mixed-precision recipe (`sec_memory_precision`), the block formats as layouts (`sec_hardware`). New here: per-GEMM operand formats, scaling granularity and mode, the block-scaling error, the cost model, the production recipe.
- Blueprint: ch04 §4.4 (`tab_lm_lowp_where`, the six-arm protocol, D1 to D3 with their numeric checks, the GEMM sweep, `tab_lm_lowp_recipes` for DeepSeek-V3 and the NVFP4 paper). The section adds the Nemotron 3 and Four Over Six paragraph and the tier line.
**Sources:**
- Papers: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (E4M3 and E5M2 and the training recipe); [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (fine-grained block scaling with fp32 promotion); [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) (instabilities that appear late in fp8 runs); [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) (the NVFP4 recipe: stochastic rounding, Hadamard, kept layers); [NVIDIA Nemotron 3](https://arxiv.org/abs/2512.20856) (NVFP4 pretraining in production and which layers stay in higher precision); [Four Over Six](https://arxiv.org/abs/2512.02010) (adaptive block scaling); [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) (MXFP4 and its shared scales); [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (the recipe the fp8 one extends).
- Code and documentation: [torchao](https://github.com/pytorch/ao) and its [float8 module](https://github.com/pytorch/ao/tree/main/torchao/float8) (`convert_to_float8_training`, tensorwise and rowwise recipes); [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (NVIDIA's fp8 recipes and delayed scaling); [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) (accumulation order and non-associativity).
- Courses, tutorials and blogs: [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/) (the production fp8 training path and its measured gains); [Alex Smola's MLSS efficiency tutorial](https://alex.smola.org/posts/45-mlss-efficiency/) (bytes, FLOPs and where precision pays).
-->

<!-- Placeholder (prose): opening. The reader trains in bf16 with fp32 master weights (sec_memory_precision) and knows the fp8 and block formats as layouts (sec_numerics, sec_hardware). An fp8 tensor core doubles matrix throughput on paper, but one scale per tensor lets one outlier set the scale for every other value; the cell below casts a real activation tensor of the reference model with one scale and with blocks, before the section builds the casts (fp8_quant, fake_quant_fp4 are defined below). The opening question: does fp8 training match bf16 within seed noise on the same model and data, at what measured speedup, and what does simulated NVFP4 do to the loss? -->

```{.python .input #low-precision-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #low-precision-opening}
%%tab pytorch
model = d2l.load_reference_models([6000])[6000]  # the reference run's end
X = d2l.mlp_input(model, block=len(model.blks) // 2, n_seqs=4, T=1024)

def dequant(q_scale):  # fp8 values times their scales, back in fp32
    q, scale = q_scale
    return q.float() * scale

casts = [('E4M3', 'one per tensor', dequant(d2l.fp8_quant(X))),
         ('E4M3', 'per 1x128 tile',
          dequant(d2l.fp8_quant(X, granularity='block1x128'))),
         ('E2M1', 'one per tensor',
          d2l.fake_quant_fp4(X.flatten(), X.numel(), scale='fp32')),
         ('E2M1', 'per 16 (NVFP4)', d2l.fake_quant_fp4(X))]
print(f'{"format":>6} {"scale":>15} {"zeroed":>7} {"NMSE":>7}')
for fmt, scale, Q in casts:
    Q = Q.reshape(X.shape)
    zeroed = ((Q == 0) & (X != 0)).float().mean()
    nmse = ((Q - X) ** 2).sum() / (X ** 2).sum()
    print(f'{fmt:>6} {scale:>15} {zeroed:>7.1%} {nmse:>7.4f}')
```

<!-- Placeholder (prose): read the table: with one scale for the whole tensor, E2M1 rounds most small entries to zero (quote the measured zeroed share; "most" only if above one half); blocks restore it. The E4M3 sentence follows the measured rows: E4M3 spans about 18 binades, so one tensor scale rarely zeroes an activation; if its two rows tie (as on random weights), say that E4M3's range absorbs one scale per tensor for this activation and let the E2M1 rows carry the point; claim a loss of small values only if the per-tensor row shows it. The rest of the section explains why, and what it costs in a training run. -->

## Which GEMMs, Which Formats, What Stays Wide

### Three GEMMs per Linear Layer

<!-- Placeholder (prose): a linear layer's training step is three GEMMs: Y = X W^T, dX = dY W, dW = dY^T X. fp8 training casts their operands: E4M3 for X and W (precision), E5M2 for dY (range; restate sec_numerics' format trade in one sentence; the common mix-up is the reverse). What stays wide in every recipe: fp32 master weights and AdamW moments (sec_memory_precision: 16 bytes per parameter, one sentence), the loss and the softmax, normalization, the embedding and the head, attention scores and probabilities, and in production recipes the sensitive projections. fp8 storage is not fp8 training. Table tab_lm_lowp_where (markdown): which experiment of this section runs where (native fp8 on Ada and Hopper; NVFP4 and MXFP4 simulated on Ada and A100, native on Blackwell; the T1 tier needs an Ada or newer card for the native arms). -->

### Casting to fp8

<!-- Placeholder (prose): fp8_quant: a scale maps the tile's largest magnitude (amax) to the format's largest value (448 for E4M3, 57,344 for E5M2), values beyond it saturate, values below half the smallest subnormal round to zero. Granularity: one scale per tensor, per row, per 1x128 tile (DeepSeek-V3's activations and gradients) or per 128x128 block (its weights; restate its fp32 promotion of partial sums every 128 elements as a cited fact). Current scaling takes the amax of the tensor being cast (one extra pass over it); delayed scaling takes the largest of the last 16 amaxes (no extra pass) and clips silently when the amax jumps. -->

```{.python .input #low-precision-fp8-quant}
%%tab pytorch
#@save
def fp8_quant(x, fmt='e4m3', granularity='tensor', scale_mode='current',
              state=None):
    """Cast x to fp8: (q, scale) with q.float() * scale ≈ x. One scale per
    tensor, row, 1x128 tile or 128x128 block; current scaling takes the
    amax of x, delayed scaling the largest of the last 16 amaxes kept in
    state (a dict the call updates)."""
    dtype = {'e4m3': torch.float8_e4m3fn, 'e5m2': torch.float8_e5m2}[fmt]
    fmax = torch.finfo(dtype).max  # 448 (E4M3) or 57,344 (E5M2)
    tiles, untile = d2l._fp8_tiles(x.detach().float(), granularity)
    amax = tiles.abs().amax(-1, keepdim=True)  # one per tile
    if scale_mode == 'delayed':  # state: the caller's dict, kept per operand
        if state is None:
            raise ValueError('delayed scaling needs a state dict')
        history = state.get('amax', amax.expand(16, *amax.shape))
        state['amax'] = torch.cat([history[1:], amax[None]])
        amax = history.amax(0)  # from the earlier calls only
    scale = amax.clamp(min=1e-12) / fmax
    q = (tiles / scale).clamp(-fmax, fmax).to(dtype)  # saturate, round
    return untile(q), untile(scale)
```

## Scaling Granularity and the Block-Scaling Error

### Error Against Block Size

<!-- Placeholder (prose): the derivation (plan 4.3 / blueprint D1): grid E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}, largest value Q = 6, smallest normal 1, subnormal spacing 0.5, m = 1 mantissa bit; scale s = amax / Q; treating the rounding error as uniform over the grid cell, E e^2 = s^2 Δ(z)^2 / 12. Two regimes: relative for |z| >= 1 (NMSE between 2^{-2m}/48 and 2^{-2m}/12, 0.005 to 0.021), absolute below (variance s^2 Δ_s^2 / 12, independent of the value). For Gaussian blocks the block maximum grows like sqrt(2 ln 2B), so larger blocks push more values into the coarse low cells; and the block maximum itself is represented exactly, which the model ignores, so it over-predicts most at small blocks. The cell measures both on 2^20 Gaussian values (fake_quant_fp4 with fp32 scales, built in the next section). -->

```{.python .input #low-precision-block-error}
%%tab pytorch
x = torch.randn(2 ** 20, generator=torch.Generator().manual_seed(0))
BLOCKS = [8, 16, 32, 64, 128, 256, 512, 1024]

def model_nmse(x, block):
    """The uniform-error model: each value's error spread over its cell."""
    xb = x.reshape(-1, block)
    s = xb.abs().amax(-1, keepdim=True) / 6  # the block's scale
    z = xb.abs() / s                          # in grid units, 0 to 6
    cell = torch.where(z < 2, 0.5, torch.where(z < 4, 1.0, 2.0))
    return float(((s * cell) ** 2 / 12).sum() / (xb ** 2).sum())

fp32_scales = lambda xb, b: d2l.fake_quant_fp4(xb, b, scale='fp32')
measured = [d2l.block_nmse(x, fp32_scales, b) for b in BLOCKS]
modelled = [model_nmse(x, b) for b in BLOCKS]
ratio = [m / e for m, e in zip(modelled, measured)]
assert 1 < ratio[-1] < ratio[0] < 1.65  # claim 2: the ratio falls toward 1
print(f'{"block":>5} {"measured":>9} {"model":>7} {"ratio":>6}')
for b, e, m, r in zip(BLOCKS, measured, modelled, ratio):
    print(f'{b:>5} {e:>9.4f} {m:>7.4f} {r:>6.2f}')
d2l.plot(BLOCKS, [measured, modelled], 'block size', 'NMSE',
         legend=['measured', 'model'], xscale='log', fmts=('-o', '--s'),
         figsize=(4.5, 3))
```

### One Outlier Sets the Scale

<!-- Placeholder (prose): one outlier of ρ standard deviations sets s = ρσ/6; once every other value sits in the absolute regime its error is s^2 Δ_s^2 / 12 per value, NMSE ≈ ρ^2 Δ_s^2 / (12 Q^2) = ρ^2 / 1728, quadratic in the in-block dynamic range. Near ρ ≈ 24 a typical (1σ) value falls below half the first grid step (s · 0.25 = ρσ/24) and rounds to zero: ρ ≈ 24 is the zeroing point, not a saturation point, and the formula still holds there; beyond about 30 it over-predicts (the zeroed values' error is at most their own size), and the NMSE approaches 1 only slowly. Read the measured table, not the plan: its "saturating near ρ ≈ 24" is wrong as an NMSE statement, and its "0.0375 against the measured 0.0370" swaps model and measurement (model 0.0370, measured 0.0375). -->

```{.python .input #low-precision-outlier}
%%tab pytorch
B, N_BLOCKS = 16, 10_000
gen = torch.Generator().manual_seed(1)

def outlier_blocks(rho):  # blocks of 16 with one rho-sigma outlier each
    xb = torch.randn(N_BLOCKS, B, generator=gen)
    xb[:, 0] = rho
    q = d2l.fake_quant_fp4(xb, B, scale='fp32')
    rest, q = xb[:, 1:], q[:, 1:]  # the values the outlier's scale crushes
    return float(((q - rest) ** 2).sum() / (rest ** 2).sum()), float(
        (q == 0).float().mean())

print(f'{"ρ":>3} {"model":>7} {"measured":>9} {"zeroed":>7}')
for rho in (8, 16, 24, 32):
    nmse, zeroed = outlier_blocks(rho)
    formula = rho ** 2 / 1728  # rho^2 Delta_s^2 / (12 Q^2)
    if rho <= 16:
        assert abs(nmse / formula - 1) < 0.1  # claim 3
    print(f'{rho:>3} {formula:>7.4f} {nmse:>9.4f} {zeroed:>7.0%}')
```

## Four-Bit Formats

### A Simulator for MXFP4 and NVFP4

<!-- Placeholder (prose): E2M1 has eight non-negative values; a block scale carries the range. MXFP4: blocks of 32 with a power-of-two (E8M0) scale, 2^(floor(log2 amax) - 2) as in the OCP spec, so amax / s lies in [4, 8) and the largest values can clip; NVFP4: blocks of 16 with an E4M3 scale under a per-tensor fp32 scale. fake_quant_fp4 quantizes and dequantizes: it measures numerics only, no speed and no memory saving (fake quantization is not FP4 training). It is the part's one FP4 simulator (chapters on compression import it). Round to nearest, or stochastic rounding: round up with probability equal to the distance from the lower grid value, so the expected result is the input. -->

```{.python .input #low-precision-fake-quant}
%%tab pytorch
#@save
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # the non-negative values

#@save
def fake_quant_fp4(x, block=16, scale='e4m3', rounding='rtn', dim=-1,
                   tensor_scale=True, generator=None):
    """x rounded to E2M1 times one scale per `block` values along dim.
    Scales: 'fp32'; 'e4m3' (NVFP4, under an fp32 tensor scale); 'pow2'
    (MXFP4's E8M0). Rounding: to nearest, ties to even ('rtn'), or
    stochastic ('sr')."""
    xt = x.float().movedim(dim, -1)
    xb = xt.reshape(-1, block)
    amax = xb.abs().amax(-1, keepdim=True)
    s = amax / 6  # the block's largest value lands on 6
    if scale == 'e4m3':  # block scales in E4M3, times one fp32 scale
        t = s.amax() / 448 if tensor_scale else s.new_ones(())
        t = torch.where(t > 0, t, 1.0)
        s = (s / t).to(torch.float8_e4m3fn).float() * t
    elif scale == 'pow2':  # 2^(floor(log2 amax) - 2): the top may clip
        s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    z = torch.where(s > 0, xb.abs() / s, 0.0).clamp(max=6)  # grid units
    grid = torch.tensor(E2M1, device=x.device)
    i = (torch.bucketize(z, grid, right=True) - 1).clamp(max=6)
    lo, hi = grid[i], grid[i + 1]  # the grid values around z
    frac = (z - lo) / (hi - lo)
    if rounding == 'sr':  # up with probability frac: unbiased
        up = frac > torch.rand(z.shape, generator=generator, device=z.device)
    else:  # to nearest; a tie goes to the even code (even index), as in E2M1
        up = (frac > 0.5) | ((frac == 0.5) & (i % 2 == 1))
    q = torch.where(up, hi, lo) * torch.sign(xb) * s
    return q.reshape(xt.shape).movedim(-1, dim).to(x.dtype)

#@save
def block_nmse(x, quantizer, block):
    """||Q(x) - x||^2 / ||x||^2 with Q = quantizer(blocks, block)."""
    xb = x.float().reshape(-1, block)
    return float(((quantizer(xb, block) - xb) ** 2).sum() / (xb ** 2).sum())
```

<!-- Placeholder (prose): the simulator's checks: values already on the grid come back exactly; stochastic rounding is unbiased; round-to-nearest is idempotent. Then the scale formats: an E4M3 scale moves amax / s by at most 2^-4 relative (at most about 6% of the grid's range), a power-of-two scale wastes up to a factor of two of the range or clips the largest values. -->

```{.python .input #low-precision-fp4-checks}
%%tab pytorch
gen = torch.Generator().manual_seed(2)
on_grid = torch.tensor(E2M1 + tuple(-v for v in E2M1)) / 4  # 16 values
exact = all(torch.equal(fake_quant_fp4(on_grid, 16, s), on_grid)
            for s in ('fp32', 'pow2'))
DRAWS = 10 ** 5
v = torch.tensor([0.3, 1.2, 2.5, 4.7])  # between grid values (scale 1)
blocks = torch.cat([v, torch.tensor([6.0]), torch.zeros(11)]).repeat(DRAWS, 1)
mean = fake_quant_fp4(blocks, 16, 'fp32', 'sr', generator=gen)[:, :4].mean(0)
bias = float(((mean - v) / v).abs().max())
x = torch.randn(2 ** 20, generator=gen)
q = fake_quant_fp4(x)
idempotent = torch.equal(fake_quant_fp4(q), q)
n_fp32, n_e4m3, n_pow2 = (  # NMSE per scale format, blocks of 16
    block_nmse(x, lambda xb, b, s=s: fake_quant_fp4(xb, b, s), 16)
    for s in ('fp32', 'e4m3', 'pow2'))
assert exact and idempotent and bias < 0.01  # claim 5
assert n_e4m3 / n_fp32 < 1.06 and 1.3 <= n_pow2 / n_e4m3 <= 2.5  # claim 4
print(f'on-grid: {"exact" if exact else "not exact"}')
print(f'stochastic rounding bias: {bias:.1%} ({DRAWS:,} draws)')
print(f'round-to-nearest idempotent: {idempotent}')
print(f'NMSE fp32 / E4M3 scale / power-of-two (block 16): '
      f'{n_fp32:.4f} / {n_e4m3:.4f} / {n_pow2:.4f}')
print(f'power-of-two vs E4M3: ×{n_pow2 / n_e4m3:.1f}')
```

### Rotating Outliers Away

<!-- Placeholder (prose): a random Hadamard transform (orthogonal: H H^T = I) applied to both operands along the reduction dimension leaves the product unchanged and spreads an outlier over the block, so no value is crushed to zero; the NVFP4 recipe applies a 16x16 one to the inputs of the weight-gradient GEMM. Say plainly what the cell measures: one outlier per block is the case rotation does not fix. The per-block scale already confines the outlier to its block; rotating spreads it into ±ρ/4 on every value, which lands the block in E2M1's coarse top cells, so the error on the 15 small values, the outlier's own error and the block's total all go up (quote the table). (The build also tried Student-t and log-normal blocks of 16: rotation did not lower the NMSE either.) The recipe's benefit is the NVFP4 paper's ablation at scale (quote it as the paper's result), not something one block shows. d2l._hadamard is the library's Sylvester construction. -->

```{.python .input #low-precision-hadamard}
%%tab pytorch
gen = torch.Generator().manual_seed(3)
H = d2l._hadamard(16, generator=gen)  # random column signs
orth_err = float((H @ H.T - torch.eye(16)).abs().max())
assert orth_err < 1e-6
xb = torch.randn(10_000, 16, generator=gen)
xb[:, 0] = 16  # one outlier of 16 standard deviations per block
q_plain = fake_quant_fp4(xb)
q_rot = fake_quant_fp4(xb @ H.T) @ H  # rotate, quantize, rotate back

def part_nmse(q, cols):  # the NMSE of the values in columns cols
    return float(((q - xb)[:, cols] ** 2).sum() / (xb[:, cols] ** 2).sum())

print(f'max |H Hᵀ − I| = {orth_err:.0e}')
print(f'{"NMSE of":>14} {"plain":>7} {"rotated":>8}')
for part, cols in (('the outlier', [0]), ('the other 15', slice(1, None)),
                   ('the block', slice(None))):
    print(f'{part:>14} {part_nmse(q_plain, cols):>7.4f} '
          f'{part_nmse(q_rot, cols):>8.4f}')
```

### Stochastic Rounding for Gradients

<!-- Placeholder (prose): why gradients need stochastic rounding: an update smaller than half a grid step rounds back to the same value every time under round-to-nearest, so a sum of many small updates is lost; stochastic rounding keeps each update in expectation. The cell adds 10,000 updates of 0.2 grid steps to a value held on a grid (in units of the local grid step); the same holds for bf16 weights late in WSD's decay (restate in one sentence why fp32 master weights matter there). -->

```{.python .input #low-precision-sr-sum}
%%tab pytorch
N_UPDATES, DELTA = 10_000, 0.2  # updates, each 0.2 of a grid step
u = torch.rand(N_UPDATES, generator=torch.Generator().manual_seed(4))
w_rtn, w_sr, rtn, sr = 0.0, 0.0, [], []
for k in range(N_UPDATES):
    w_rtn = round(w_rtn + DELTA)  # to the nearest grid value
    w_sr = math.floor(w_sr + DELTA + float(u[k]))  # up with prob. 0.2
    rtn.append(w_rtn)
    sr.append(w_sr)
exact = [DELTA * (k + 1) for k in range(N_UPDATES)]
assert w_rtn == 0  # claim 5: round-to-nearest loses every update
print(f'sum of {N_UPDATES:,} updates of {DELTA} steps: RTN {w_rtn:.1f}, '
      f'SR {w_sr:.1f} (exact {exact[-1]:,.0f})')
d2l.plot(list(range(1, N_UPDATES + 1)), [exact, sr, rtn], 'updates',
         'sum (grid steps)', legend=['exact', 'stochastic', 'to nearest'],
         fmts=('k:', '-', '--'), figsize=(4.5, 3))
```

## The Cost Model and the Six Arms

### fp8 and NVFP4 Linear Layers

<!-- Placeholder (prose): Fp8Linear: an nn.Linear whose three GEMMs run on fp8 operands through a custom autograd.Function (sec_custom_layer): forward casts X and W to E4M3, backward casts dY to E5M2 and reuses the saved fp8 X and W. Simulated: dequantize and multiply in fp32 (fp32 accumulation, as the tensor core does), which gives the numerics on any device and never a speedup; native: torch._scaled_mm on an Ada or newer card with tensor scales. .stats() reports saturation, underflow and relative error per operand since the last call (and the same tensors under a current scale, the replay the delayed arm is read against). -->

```{.python .input #low-precision-fp8-linear}
%%tab pytorch
#@save
class _Fp8Matmul(torch.autograd.Function):
    """Y = X W^T and its two backward GEMMs on fp8 operands."""
    @staticmethod
    def forward(ctx, X, W, lin, training):
        x = lin.quant(X, 'x', training)  # (q, scale), E4M3
        w = lin.quant(W, 'w', training)
        ctx.lin, ctx.dtypes = lin, (X.dtype, W.dtype)
        ctx.save_for_backward(*x, *w)
        return d2l._fp8_mm(x, w, lin.native, d2l._out_dtype(X), tb=True)

    @staticmethod
    def backward(ctx, dY):
        xq, xs, wq, ws = ctx.saved_tensors
        dy, native = ctx.lin.quant(dY, 'dy'), ctx.lin.native  # E5M2
        dX = d2l._fp8_mm(dy, (wq, ws), native, ctx.dtypes[0])  # dY W
        dW = d2l._fp8_mm(dy, (xq, xs), native, ctx.dtypes[1], ta=True)
        return dX, dW, None, None  # dW = dY^T X

#@save
class Fp8Linear(nn.Linear):
    """nn.Linear with fp8 GEMMs: X, W in E4M3, dY in E5M2, fp32
    accumulation; simulated, or native (torch._scaled_mm, Ada or newer).
    While self.track, .stats() reports each training step's casts."""
    def __init__(self, in_features, out_features, bias=False,
                 granularity='tensor', scale_mode='current', native=False):
        super().__init__(in_features, out_features, bias=bias)
        self.granularity, self.scale_mode = granularity, scale_mode
        self.native, self.track, self.records = native, False, []
        self.state = {'x': {}, 'w': {}, 'dy': {}}  # delayed: amax histories

    def quant(self, t, role, training=True):
        """A cast of operand role; outside training (no gradients, e.g. a
        validation pass) the amax history is read but not updated."""
        fmt = 'e5m2' if role == 'dy' else 'e4m3'
        gran = d2l._fp8_granularity(self.granularity, role)
        state = self.state[role] if training else dict(self.state[role])
        q, s = fp8_quant(t, fmt, gran, self.scale_mode, state)
        if self.track and training:
            self.records.append(d2l._fp8_record(t, q, s, fmt, gran, role))
        return q, s

    def forward(self, X):
        Y = _Fp8Matmul.apply(X.reshape(-1, X.shape[-1]), self.weight, self,
                             torch.is_grad_enabled())
        Y = Y.reshape(*X.shape[:-1], -1)
        return Y if self.bias is None else Y + self.bias

    def stats(self):  # the mean per operand since the last call
        table, self.records = d2l._fp8_stats_table(self.records), []
        return table
```

<!-- Placeholder (prose): the checks of Fp8Linear (CPU, simulated): the forward relative Frobenius error against fp32 F.linear lies within a factor two of the analytic value sqrt(2 · 2^{-6}/24) (two E4M3 casts, each with relative-regime NMSE 2^{-2m}/24 for values spread evenly over a binade); input and weight gradients against fp32 autograd within 1e-1 (E5M2 has two mantissa bits); on an Ada card the native path matches the simulated one to bf16 output rounding (≤ 5e-3); and delayed scaling after a jump of the amax by six against a history of ones saturates every value above the old amax at 448. -->

```{.python .input #low-precision-fp8-checks}
%%tab pytorch
torch.manual_seed(0)
lin = Fp8Linear(512, 256)  # simulated, on the CPU
Xc = torch.randn(1024, 512, requires_grad=True)
Y, Y_ref = lin(Xc), F.linear(Xc, lin.weight)
dY = torch.randn_like(Y)
dX, dW = torch.autograd.grad(Y, (Xc, lin.weight), dY)
dX_ref, dW_ref = torch.autograd.grad(Y_ref, (Xc, lin.weight), dY)
rel = lambda a, b: float((torch.dist(a, b) / b.norm()).detach())
analytic = math.sqrt(2 * 2 ** -6 / 24)  # two E4M3 casts
assert rel(Y, Y_ref) < 2 * analytic  # claim 6
assert rel(dX, dX_ref) < 0.1 and rel(dW, dW_ref) < 0.1
print(f'forward error: {rel(Y, Y_ref):.1e} (analytic {analytic:.1e})')
print(f'input / weight grad error: {rel(dX, dX_ref):.1e} / '
      f'{rel(dW, dW_ref):.1e}')
```

```{.python .input #low-precision-fp8-native}
%%tab pytorch
dev = d2l.try_gpu()
if d2l.has_fp8_cores(dev):  # Ada or newer
    sim = Fp8Linear(512, 256).to(dev)
    nat = Fp8Linear(512, 256, native=True).to(dev)
    nat.load_state_dict(sim.state_dict())
    Xg = torch.randn(1024, 512, device=dev, dtype=torch.bfloat16)
    err = rel(nat(Xg).float(), sim(Xg).float())
    assert err < 5e-3  # claim 6: bf16 output rounding, no more
    print(f'native vs simulated ({d2l.device_name(dev)}): {err:.1e}')
else:
    print('native vs simulated: skipped (needs an Ada or newer GPU)')
state = {}  # delayed scaling: a history of 16 amaxes near 1, then a jump
for _ in range(16):
    fp8_quant(2 * torch.rand(4096) - 1, scale_mode='delayed', state=state)
q, _ = fp8_quant(12 * torch.rand(4096) - 6, scale_mode='delayed', state=state)
saturated = float((q.float().abs() == 448).float().mean())
print(f'delayed scaling after a 6× jump: {saturated:.0%} saturated at 448')
```

<!-- Placeholder (prose): Fp4Linear implements the NVFP4 training recipe of the NVFP4 pretraining paper (simulated here; native on Blackwell, below): round-to-nearest for the forward operands, 16x16 blocks for the weights (so the forward and the input gradient read the same quantized W), stochastic rounding on the gradients, and a random 16x16 Hadamard transform over tokens on both inputs of the weight-gradient GEMM; which layers stay in higher precision is quoted from the paper and the Nemotron 3 report. On this box: numerics only, no speed. -->

```{.python .input #low-precision-fp4-linear}
%%tab pytorch
#@save
class _Fp4Matmul(torch.autograd.Function):
    """Y = X W^T with NVFP4 operands (the NVFP4 recipe): simulated, or
    on FP4 tensor cores where the GPU has them (d2l._fp4_mm)."""
    @staticmethod
    def forward(ctx, X, W, lin):
        ctx.lin = lin
        ctx.save_for_backward(X, W)
        Wq = d2l._fp4_weight(W, lin.block, lin.weight_block2d,
                             scale=lin.scale)
        Xq = fake_quant_fp4(X, lin.block, lin.scale)
        return d2l._fp4_mm(Xq, Wq.T, lin, d2l._out_dtype(X), raw=(X, W.T))

    @staticmethod
    def backward(ctx, dY):
        (X, W), lin = ctx.saved_tensors, ctx.lin
        q = lambda t, r, dim=-1: fake_quant_fp4(t, lin.block, lin.scale,
                                                rounding=r, dim=dim)
        sr = 'sr' if lin.sr_grads else 'rtn'  # stochastic for gradients
        Wq = d2l._fp4_weight(W, lin.block, lin.weight_block2d, dim=0,
                             scale=lin.scale)
        dX = d2l._fp4_mm(q(dY, sr), Wq, lin, X.dtype, raw=(dY, W))  # dY W
        if lin.H is not None:  # rotate each 16 tokens: dY^T X is unchanged
            X, dY = d2l._rotate_rows(X, lin.H), d2l._rotate_rows(dY, lin.H)
        dW = d2l._fp4_mm(q(dY, sr, 0).T, q(X, 'rtn', 0), lin, W.dtype,
                         raw=(dY.T, X), weight=False)  # not in W's tiles
        return dX, dW, None  # dW = dY^T X, blocks along the tokens

#@save
class Fp4Linear(nn.Linear):
    """nn.Linear with NVFP4 (or MXFP4: block=32, scale='pow2') training
    GEMMs; native=None: on FP4 tensor cores where the GPU has them."""
    def __init__(self, in_features, out_features, bias=False, block=16,
                 weight_block2d=True, hadamard=16, sr_grads=True,
                 scale='e4m3', native=None):
        super().__init__(in_features, out_features, bias=bias)
        self.block, self.weight_block2d = block, weight_block2d
        self.sr_grads, self.scale, self.native = sr_grads, scale, native
        H = (d2l._hadamard(hadamard, torch.Generator().manual_seed(0))
             if hadamard else None)  # a random Hadamard transform
        self.register_buffer('H', H, persistent=False)

    def forward(self, X):
        Y = _Fp4Matmul.apply(X.reshape(-1, X.shape[-1]), self.weight, self)
        Y = Y.reshape(*X.shape[:-1], -1)
        return Y if self.bias is None else Y + self.bias
```

<!-- Placeholder (prose): which path ran. Blackwell GPUs (B200 and B300, the RTX 50 series) have FP4 tensor cores and Fp4Linear uses them there: the same quantized operands (rounding, stochastic rounding and the Hadamard rotation stay in software) go to the hardware as packed E2M1 codes with their E4M3 block scales and the fp32 tensor scale (MXFP4: E8M0 scales per 32, native on B200/B300 only with torch 2.11), and the 16 x 16 weight tiles become the hardware's 1 x 16 blocks by repeating each tile's scale. The build box (4 x RTX 4090, compute capability 8.9) has none, so every FP4 number in this section comes from the simulator; the native path is tested on the CPU with the hardware mocked (tools/test_lm_low_precision.py: the codes and scales handed to the GEMM reconstruct the simulator's operands exactly in all three GEMMs) and has not run on Blackwell here (claim 13; campaign lowp-blackwell). Quote no native FP4 speed from a datasheet as if measured. -->

```{.python .input #low-precision-fp8-and-nvfp4-linear-layers-1}
%%tab pytorch
print(d2l.fp4_backend(d2l.try_gpu()))
```

### When fp8 Pays: the Crossover and Amdahl's Bound

<!-- Placeholder (prose): the cost model (blueprint D2): with bf16 peak P, fp8 peak rP and casts that read 2 bytes and write 1 byte per operand element at bandwidth W, a forward-only n x n x n GEMM gains (2 n^3 / ηP)(1 - 1/r) and pays c · 6 n^2 / W of cast traffic (two operands), so it wins above n*_fwd = 3 c η P r / ((r - 1) W) (990 for the 4090's 165 TFLOP/s, 1.0 TB/s, r = 2, c = η = 1; tab_gpu_specs). A training step has three GEMMs (6 n^3) against three operand casts (9 n^2 bytes) plus the transposed copies the backward GEMMs read (about 3 n^2 more), so its crossover is n* ≈ (1.5 to 2) c η P r / ((r - 1) W) ≈ 500 to 660 at c = η = 1: the forward-only crossover is 1.5 to 2 times the training one, not half of it (the plan has this backwards). Compare the sweep with the training form: a measured training crossover near 3,072 means c of about 4 to 6 for this sweep's casts (compute it from the measured crossover and quote it). The sweep times one linear layer's training step (forward and both backward products) at 16,384 x n x n and at this section's preset shapes, in a process of its own on a whole card: bf16, fp8 with the casts a step pays, and the fp8 GEMMs alone. GeForce Ada runs fp8 with fp32 accumulation at a reduced rate: one reason the measured ratio may stay below the datasheet's 2; the sweep measures it rather than assuming. Datasheet ratios are not speedups. -->

```{.python .input #low-precision-gemm-sweep}
%%tab pytorch
cfg = d2l.r2_config('fp8')  # the preset of the arms below
arch = cfg.arch
M = cfg.global_batch * arch.T  # tokens in one step of the arms
SHAPES = [(M, arch.d, n) for n in (arch.d, arch.H_kv * arch.d_h, arch.f)] + [
    (M, arch.f, arch.d)]  # W_q and W_o; W_k and W_v; W_g and W_1; W_2
sweep = d2l.fp8_gemm_sweep(m=M, ns=(512, 768, 1024, 1536, 2048, 3072, 4096),
                           shapes=SHAPES)
t = sweep.table.set_index('n')
ratio, n_max = t['fp8 with casts'] / t['bf16'], t.index[-1]
# claim 7: the crossover, the smallest n from which every larger n wins
wins = [n for n in ratio.index if (ratio[ratio.index >= n] > 1).all()]
if wins:
    print(f'fp8 with casts beats bf16 from n = {wins[0]:,} on '
          f'(n = {n_max:,}: ×{ratio[n_max]:.2f})')
else:
    print(f'fp8 with casts does not beat bf16 at n = {n_max:,} '
          f'(×{ratio[n_max]:.2f})')
print(f'fp8 without casts / bf16 at n = {n_max:,}: '
      f'×{t["fp8 GEMM only"][n_max] / t["bf16"][n_max]:.2f} (datasheet ×2)')
d2l.plot_gemm_sweep(sweep)
```

### Six Arms on One Model

<!-- Placeholder (prose): the fp8 preset: PackedGPT with width 768, 8 blocks, 12 query and 4 key-value heads, SwiGLU 2,048, QK-norm on, every width a multiple of 16 as fp8 GEMMs require; 20M training tokens of the 100M shards, 1,221 steps of 16 sequences of 1,024; AdamW (0.9, 0.95), weight decay 0.1, WSD with peak 1e-3, z-loss 1e-4, fp32 master weights, torch.compile in every arm. -->

```{.python .input #low-precision-preset}
%%tab pytorch
a = cfg.arch  # the preset of the sweep cell
with torch.device('meta'):  # shapes only, no memory
    P = sum(p.numel() for p in d2l.PackedGPT.from_arch(a).parameters())
widths = (a.d, a.f, a.H_kv * a.d_h)
tokens = cfg.steps * cfg.global_batch * a.T
print(f'{P:,} parameters; QK-norm {"on" if a.qk_norm else "off"}; widths '
      f'{", ".join(map(str, widths))}, multiples of 16: '
      f'{all(w % 16 == 0 for w in widths)}')
print(f'{cfg.steps:,} steps × {cfg.global_batch} × {a.T:,} = {tokens:,} '
      f'tokens')
print(f'WSD: warmup {cfg.warmup}, decay from step {cfg.steps - cfg.decay:,} '
      f'over {cfg.decay}; peak learning rate {cfg.peak_lr:g}')
```

<!-- Placeholder (prose): the arms. bf16 (four seeds: the noise reference); torchao fp8 tensorwise (native on Ada, two seeds paired with bf16's: a run's model and data seed are equal, so bf16 seed s and tensorwise seed s see the same initialization and batches); torchao rowwise (native if the card supports it, otherwise the simulated rowwise Fp8Linear, labelled in the native column); the from-scratch Fp8Linear with delayed scaling, recording operand statistics every 10 steps (native fp8 GEMMs run eagerly outside the compiled graph, 'eager' in the native column; its speed is not claimed); NVFP4 simulated on the MLP's three GEMMs with Fp4Linear, attention, embedding and head in bf16. torchao converts every block linear whose dimensions are multiples of 16 (the tied head is not converted). Throughput: the median over steps 200-400, timed after compilation with device synchronization, and its interquartile range; the baseline is bf16 with torch.compile, never fp32 or tf32. The runs take about an hour on one card, so a build loads the committed result. Note for the prose (from the build): torch 2.11 / Triton 3.6 miscompile a fused kernel that stores an fp8 tile in two layouts, so the arms and the sweep compile with a pass that moves fp8 copies out of fused kernels (the library says this in one sentence; one clause on the page at most). -->

```{.python .input #low-precision-arms}
%%tab pytorch
ARMS = {  # name: (recipe, seeds)
    'bf16': ('bf16', (0, 1, 2, 3)),
    'tensorwise': ('torchao-tensorwise', (0, 1)),
    'rowwise': ('torchao-rowwise', (0,)),
    'delayed': ('fp8-delayed', (0,)),
    'nvfp4-mlp': ('nvfp4-mlp', (0,))}
runs = d2l.lowp_arms(cfg, ARMS)  # speeds: '—' where fp8 runs outside compile
print(runs.table.to_string(index=False, na_rep='—', formatters={
    'val loss': '{:.3f}'.format, 'tok/s': '{:,.0f}'.format,
    'IQR': '{:,.0f}'.format, '÷bf16': '{:.2f}'.format,
    'GiB': '{:.1f}'.format}))
```

<!-- Placeholder (prose): Amdahl: only the block GEMMs are cast, so with share s of a bf16 step in them and a speedup r on exactly those GEMMs, the step speeds up at most 1 / ((1 - s) + s / r). s comes from a profile of the bf16 arm's seed 0 (steps 200-210); it is a measured time share, which is why it is far below the plan's FLOP share (s ≈ 0.57, bound ≈ 1.40 at r = 2: the datasheet version of the same arithmetic; attention and elementwise work dominate at width 768). r is the sweep's GEMM-only speedup at the preset shapes, each weighted by its bf16 time and by how many of it a block holds (two each of W_q/W_o, W_k/W_v, W_g/W_1, one W_2): casts only add time, so this is the true ceiling. The profile table then shows where the fp8 step's time actually goes, against the bf16 step: its block GEMMs (realized, not the sweep's), the fp8 copies of the compiler workaround (torch 2.11 / Triton 3.6; quote their measured ms and share, not just a clause), and the other kernels, where torchao's fused casts land. If the measured ratio in the arms table is below one, say so and name the reasons from these rows (casts and copies in "other kernels" and "fp8 copies", the GeForce fp8 accumulation rate as the gap between the realized and the datasheet GEMM speedup). -->

```{.python .input #low-precision-amdahl}
%%tab pytorch
ms = runs.profile.pivot_table('ms', 'kernels', 'arm', sort=False)  # per step
ms.loc['whole step'] = ms.sum()
s = ms.loc['block GEMMs', 'bf16'] / ms.loc['whole step', 'bf16']
COUNTS = [2, 2, 2, 1]  # per block: W_q, W_o | W_k, W_v | W_g, W_1 | W_2
r = float((sweep.shapes['bf16'] * COUNTS).sum() /
          (sweep.shapes['fp8 GEMM only'] * COUNTS).sum())
bound = 1 / ((1 - s) + s / r)  # casts only add time: an upper bound
print(f'GEMM share of a bf16 step: {s:.2f}; fp8 GEMM-only speedup: ×{r:.2f}')
print(f'end-to-end bound: ×{bound:.2f}')
print(ms[['bf16', 'tensorwise']].rename_axis(
    index=None, columns='ms per step').round(1).to_string())
d2l.plot_lowp_runs(runs, bound)
```

### Reading the Arms

<!-- Placeholder (prose): fp8 against seed noise: the four bf16 seeds give a mean m and a standard deviation s; one more bf16 seed falls in m ± t_3 s sqrt(1 + 1/4) with probability 0.95, so a tensorwise seed outside it is a 5% false alarm when fp8 is exact. Quote DeepSeek-V3's ablation (relative loss error below 0.25%, its §3.3) as a different model and recipe. If a tensorwise seed falls outside, that is a bug to root-cause before writing. -->

```{.python .input #low-precision-seed-noise}
%%tab pytorch
final = runs.losses.dropna(subset=['val_loss']).groupby(
    ['arm', 'seed'])['val_loss'].last()
bf16 = final['bf16'].to_numpy()
m, h = d2l.t_interval(bf16)  # mean, half-width of the 95% interval for it
sd = bf16.std(ddof=1)
half = h * math.sqrt(len(bf16) + 1)  # one more seed: t s sqrt(1 + 1/n)
lo, hi = m - half, m + half
print(f'bf16, {len(bf16)} seeds: {m:.3f} (SD {sd:.3f})')
print(f'95% interval for one more bf16 seed: [{lo:.3f}, {hi:.3f}]')
tw = final['tensorwise']
print(f'tensorwise seeds {", ".join(map(str, tw.index))}: '
      + ', '.join(f'{v:.3f} ({"inside" if lo <= v <= hi else "outside"})'
                  for v in tw))
```

<!-- Placeholder (prose): delayed against current scaling, on the same tensors: every 10 steps the delayed arm records, per operand, the share of values that saturate at the format's maximum and the share that underflow to zero under its delayed scales and under a current scale of the same tensor (current scaling never saturates by construction). Read the windows of ±50 steps around the end of warmup and the start of the decay, per operand (x and w in E4M3, dy in E5M2; averaging the roles hides the effect: in the smoke, dy underflow differed by about 100× while x and w were within 10%); "no difference at this scale" for a role whose delayed/current underflow ratio lies within 10% of 1. Saturation: current scaling never saturates, so any delayed saturation is the effect. Validation passes do not touch the amax histories (Fp8Linear reads them without updating outside training). -->

```{.python .input #low-precision-scaling-stats}
%%tab pytorch
WINDOW = 50  # steps on either side of a phase change
phases = {'warmup end': cfg.warmup, 'decay start': cfg.steps - cfg.decay}
print(f'{"in ppm":>11} {"operand":>7} {"saturated":>9} {"underflow":>9} '
      f'{"(current)":>9} {"ratio":>6}')
for phase, step in phases.items():
    near = runs.stats[(runs.stats['step'] - step).abs() <= WINDOW]
    ppm = 1e6 * near.groupby('role').mean(numeric_only=True)
    for role in ('x', 'w', 'dy'):
        sat, under, cur = ppm.loc[role, ['saturation', 'underflow',
                                         'underflow_current']]
        under_ratio = f'{under / cur:.2f}' if cur > 0 else '—'
        print(f'{phase:>11} {role:>7} {sat:>9.3g} {under:>9.3g} {cur:>9.3g} '
              f'{under_ratio:>6}')  # ratio: delayed / current
d2l.plot_lowp_stats(runs.stats, marks=(cfg.warmup, cfg.steps - cfg.decay))
```

<!-- Placeholder (prose): NVFP4 at this scale: the gap of the NVFP4 arm to the bf16 mean (name the path the cell prints from the arms table's native column: simulated on the build box, "native FP4" if the result was regenerated on Blackwell) in units of the bf16 seed SD; visible means outside the 95% interval and more than three SDs. The NVFP4 paper reports parity with fp8 at 12B parameters and 10T tokens; quote it as the paper's result (and Nemotron 3's NVFP4 pretraining for up to 25T tokens with the sensitive layers kept in higher precision, and Four Over Six's adaptive block scaling as the refinement that narrows the gap). A 75M model on 20M tokens is not evidence against the recipe at scale. Closing hand-off: the same formats at inference (chap_lm_compression). -->

```{.python .input #low-precision-nvfp4-gap}
%%tab pytorch
gap = final['nvfp4-mlp'].mean() - m
where = 'inside' if lo <= m + gap <= hi else 'outside'
path = runs.table.set_index('arm').loc['nvfp4-mlp', 'native']
path = {'eager': 'native FP4'}.get(path, path)  # 'simulated' off Blackwell
print(f'NVFP4 ({path}, MLP) − bf16 mean: {gap:+.3f} = {gap / sd:.1f} '
      f'seed SDs, {where} the 95% interval')
```

<!-- Placeholder (prose): table tab_lm_lowp_recipes (markdown): per production recipe (DeepSeek-V3's fine-grained fp8, the NVFP4 paper, Nemotron 3), the formats and granularity of the six GEMM operands, the rounding, and what stays in higher precision. Then the formats by hardware (fp8 native on Ada and Hopper; MXFP4/NVFP4 native on Blackwell; simulated elsewhere). -->

## Summary

<!-- Placeholder (prose): summary, 3-6 sentences. -->

## Exercises

<!-- Placeholder (prose): exercises 5-7 per spec part H: the error ratio of block sizes 16 and 128 for Gaussian values (derive); which operands need E5M2 and why; the crossover n* for forward-only GEMMs (1.5 to 2 times the training value, not half of it: one GEMM against two casts instead of three GEMMs against three casts and their transposed copies; optional sweep, 5 min); overflow under delayed scaling after a jump (amax 6 against a history of 1 saturates at 448; CPU); the NVFP4 block-size and scale-format sweep on 5M tokens with two seeds (one card, about 40 min); open: a SwiGLU outlier miniature that destabilizes per-tensor scaling, and the block size that fixes it (one card, about 20 min). -->
